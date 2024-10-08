import Mathlib

namespace no_solutions_exist_l134_134229

theorem no_solutions_exist (m n : ℤ) : ¬(m^2 = n^2 + 1954) :=
by sorry

end no_solutions_exist_l134_134229


namespace current_age_of_son_l134_134475

variables (S F : ℕ)

-- Define the conditions
def condition1 : Prop := F = 3 * S
def condition2 : Prop := F - 8 = 4 * (S - 8)

-- The theorem statement
theorem current_age_of_son (h1 : condition1 S F) (h2 : condition2 S F) : S = 24 :=
sorry

end current_age_of_son_l134_134475


namespace stratified_sampling_l134_134489

theorem stratified_sampling (total_students boys girls sample_size x y : ℕ)
  (h1 : total_students = 8)
  (h2 : boys = 6)
  (h3 : girls = 2)
  (h4 : sample_size = 4)
  (h5 : x + y = sample_size)
  (h6 : (x : ℚ) / boys = 3 / 4)
  (h7 : (y : ℚ) / girls = 1 / 4) :
  x = 3 ∧ y = 1 :=
by
  sorry

end stratified_sampling_l134_134489


namespace spinner_probability_l134_134376

-- Define the game board conditions
def total_regions : ℕ := 12  -- The triangle is divided into 12 smaller regions
def shaded_regions : ℕ := 3  -- Three regions are shaded

-- Define the probability calculation
def probability (total : ℕ) (shaded : ℕ): ℚ := shaded / total

-- State the proof problem
theorem spinner_probability :
  probability total_regions shaded_regions = 1 / 4 :=
by
  sorry

end spinner_probability_l134_134376


namespace store_earnings_correct_l134_134903

theorem store_earnings_correct :
  let graphics_cards_sold : ℕ := 10
  let hard_drives_sold : ℕ := 14
  let cpus_sold : ℕ := 8
  let ram_pairs_sold : ℕ := 4
  let graphics_card_price : ℝ := 600
  let hard_drive_price : ℝ := 80
  let cpu_price : ℝ := 200
  let ram_pair_price : ℝ := 60
  graphics_cards_sold * graphics_card_price +
  hard_drives_sold * hard_drive_price +
  cpus_sold * cpu_price +
  ram_pairs_sold * ram_pair_price = 8960 := 
by
  sorry

end store_earnings_correct_l134_134903


namespace not_all_roots_real_l134_134168

-- Define the quintic polynomial with coefficients a5, a4, a3, a2, a1, a0
def quintic_polynomial (a5 a4 a3 a2 a1 a0 : ℝ) (x : ℝ) : ℝ :=
  a5 * x^5 + a4 * x^4 + a3 * x^3 + a2 * x^2 + a1 * x + a0

-- Define a predicate for the existence of all real roots
def all_roots_real (a5 a4 a3 a2 a1 a0 : ℝ) : Prop :=
  ∀ r : ℝ, quintic_polynomial a5 a4 a3 a2 a1 a0 r = 0

-- Define the main theorem statement
theorem not_all_roots_real (a5 a4 a3 a2 a1 a0 : ℝ) :
  2 * a4^2 < 5 * a5 * a3 →
  ¬ all_roots_real a5 a4 a3 a2 a1 a0 :=
by
  sorry

end not_all_roots_real_l134_134168


namespace range_of_k_if_f_monotonically_increasing_l134_134658

noncomputable def f (k x : ℝ) : ℝ := k * x - Real.log x

theorem range_of_k_if_f_monotonically_increasing :
  (∀ (x : ℝ), 1 < x → 0 ≤ (k - 1 / x)) → k ∈ Set.Ici (1: ℝ) :=
by
  intro hyp
  have : ∀ (x : ℝ), 1 < x → 0 ≤ k - 1 / x := hyp
  sorry

end range_of_k_if_f_monotonically_increasing_l134_134658


namespace range_of_a_l134_134899

def proposition_p (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2 * x + a > 0

def proposition_q (a : ℝ) : Prop :=
  a - 1 > 1

theorem range_of_a (a : ℝ) :
  (proposition_p a ∨ proposition_q a) ∧ ¬ (proposition_p a ∧ proposition_q a) ↔ 1 < a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l134_134899


namespace evaluate_expression_l134_134615

theorem evaluate_expression : - (16 / 4 * 8 - 70 + 4^2 * 7) = -74 := by
  sorry

end evaluate_expression_l134_134615


namespace marks_in_chemistry_l134_134611

-- Define the given conditions
def marks_english := 76
def marks_math := 65
def marks_physics := 82
def marks_biology := 85
def average_marks := 75
def number_subjects := 5

-- Define the theorem statement to prove David's marks in Chemistry
theorem marks_in_chemistry :
  let total_marks := marks_english + marks_math + marks_physics + marks_biology
  let total_marks_all_subjects := average_marks * number_subjects
  let marks_chemistry := total_marks_all_subjects - total_marks
  marks_chemistry = 67 :=
sorry

end marks_in_chemistry_l134_134611


namespace min_and_max_f_l134_134132

noncomputable def f (x : ℝ) : ℝ := -2 * x + 1

theorem min_and_max_f :
  (∀ x, 0 ≤ x ∧ x ≤ 5 → f x ≥ -9) ∧ (∀ x, 0 ≤ x ∧ x ≤ 5 → f x ≤ 1) :=
by
  sorry

end min_and_max_f_l134_134132


namespace probability_age_less_than_20_l134_134177

theorem probability_age_less_than_20 (total_people : ℕ) (over_30_years : ℕ) 
  (less_than_20_years : ℕ) (h1 : total_people = 120) (h2 : over_30_years = 90) 
  (h3 : less_than_20_years = total_people - over_30_years) : 
  (less_than_20_years : ℚ) / total_people = 1 / 4 :=
by {
  sorry
}

end probability_age_less_than_20_l134_134177


namespace find_range_of_m_l134_134808

def proposition_p (m : ℝ) : Prop := 0 < m ∧ m < 1/3
def proposition_q (m : ℝ) : Prop := 0 < m ∧ m < 15
def proposition_r (m : ℝ) : Prop := proposition_p m ∨ proposition_q m
def proposition_s (m : ℝ) : Prop := proposition_p m ∧ proposition_q m = False
def range_of_m (m : ℝ) : Prop := 1/3 ≤ m ∧ m < 15

theorem find_range_of_m (m : ℝ) : proposition_r m ∧ proposition_s m → range_of_m m := by
  sorry

end find_range_of_m_l134_134808


namespace find_alpha_l134_134011

-- Declare the conditions
variables (α : ℝ) (h₀ : 0 < α) (h₁ : α < 90) (h₂ : Real.sin (α - 10 * Real.pi / 180) = Real.sqrt 3 / 2)

theorem find_alpha : α = 70 * Real.pi / 180 :=
sorry

end find_alpha_l134_134011


namespace long_diagonal_length_l134_134600

-- Define the lengths of the rhombus sides and diagonals
variables (a b : ℝ) (s : ℝ)
variable (side_length : ℝ)
variable (short_diagonal : ℝ)
variable (long_diagonal : ℝ)

-- Given conditions
def rhombus (side_length: ℝ) (short_diagonal: ℝ) : Prop :=
  side_length = 51 ∧ short_diagonal = 48

-- To prove: length longer diagonal is 90 units
theorem long_diagonal_length (side_length: ℝ) (short_diagonal: ℝ) (long_diagonal: ℝ) :
  rhombus side_length short_diagonal →
  long_diagonal = 90 :=
by
  sorry 

end long_diagonal_length_l134_134600


namespace john_streams_hours_per_day_l134_134282

theorem john_streams_hours_per_day :
  (∃ h : ℕ, (7 - 3) * h * 10 = 160) → 
  (∃ h : ℕ, h = 4) :=
sorry

end john_streams_hours_per_day_l134_134282


namespace function_passes_through_fixed_point_l134_134094

noncomputable def passes_through_fixed_point (a : ℝ) (h : a > 0 ∧ a ≠ 1) : Prop :=
  ∃ y : ℝ, y = a^(1-1) + 1 ∧ y = 2

theorem function_passes_through_fixed_point (a : ℝ) (h : a > 0 ∧ a ≠ 1) : passes_through_fixed_point a h :=
by
  sorry

end function_passes_through_fixed_point_l134_134094


namespace percentage_increase_in_second_year_l134_134238

def initial_deposit : ℝ := 5000
def first_year_balance : ℝ := 5500
def two_year_increase_percentage : ℝ := 21
def second_year_increase_percentage : ℝ := 10

theorem percentage_increase_in_second_year
  (initial_deposit first_year_balance : ℝ) 
  (two_year_increase_percentage : ℝ) 
  (h1 : first_year_balance = initial_deposit + 500) 
  (h2 : (initial_deposit * (1 + two_year_increase_percentage / 100)) = initial_deposit * 1.21) 
  : second_year_increase_percentage = 10 := 
sorry

end percentage_increase_in_second_year_l134_134238


namespace inequality_reciprocal_l134_134993

theorem inequality_reciprocal (a b : Real) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) : (1 / a < 1 / b) :=
by
  sorry

end inequality_reciprocal_l134_134993


namespace cube_edge_length_l134_134906

theorem cube_edge_length (surface_area : ℝ) (h : surface_area = 96) : ∃ (edge_length : ℝ), edge_length = 4 := 
by 
  sorry

end cube_edge_length_l134_134906


namespace part_I_solution_set_part_II_min_value_l134_134893

-- Define the function f(x)
def f (x : ℝ) : ℝ := x + 1 + |3 - x|

-- Prove the solution set of the inequality f(x) ≤ 6 for x ≥ -1 is -1 ≤ x ≤ 4
theorem part_I_solution_set (x : ℝ) (h1 : x ≥ -1) : f x ≤ 6 ↔ (-1 ≤ x ∧ x ≤ 4) :=
by
  sorry

-- Define the condition for the minimum value of f(x)
def min_f := 4

-- Prove the minimum value of 2a + b under the given constraints
theorem part_II_min_value (a b : ℝ) (h2 : a > 0 ∧ b > 0) (h3 : 8 * a * b = a + 2 * b) : 2 * a + b ≥ 9 / 8 :=
by
  sorry

end part_I_solution_set_part_II_min_value_l134_134893


namespace identical_lines_pairs_count_l134_134143

theorem identical_lines_pairs_count : 
  ∃ P : Finset (ℝ × ℝ), (∀ p ∈ P, 
    (∃ a b, p = (a, b) ∧ 
      (∀ x y, 2 * x + a * y + b = 0 ↔ b * x + 3 * y - 9 = 0))) ∧ P.card = 2 :=
sorry

end identical_lines_pairs_count_l134_134143


namespace part1_a2_part1_a3_part2_general_formula_l134_134365

noncomputable def a : ℕ → ℚ
| 0 => 0
| 1 => 1
| n + 1 => (n + 1) * n / 2

noncomputable def S (n : ℕ) : ℚ := (n + 2) * a n / 3

theorem part1_a2 : a 2 = 3 := sorry

theorem part1_a3 : a 3 = 6 := sorry

theorem part2_general_formula (n : ℕ) (h : n > 0) : a n = n * (n + 1) / 2 := sorry

end part1_a2_part1_a3_part2_general_formula_l134_134365


namespace players_quit_game_l134_134521

variable (total_players initial num_lives players_left players_quit : Nat)
variable (each_player_lives : Nat)

theorem players_quit_game :
  (initial = 8) →
  (each_player_lives = 3) →
  (num_lives = 15) →
  players_left = num_lives / each_player_lives →
  players_quit = initial - players_left →
  players_quit = 3 :=
by
  intros h_initial h_each_player_lives h_num_lives h_players_left h_players_quit
  sorry

end players_quit_game_l134_134521


namespace guilty_D_l134_134854

def isGuilty (A B C D : Prop) : Prop :=
  ¬A ∧ (B → ∃! x, x ≠ A ∧ (x = C ∨ x = D)) ∧ (C → ∃! x₁ x₂, x₁ ≠ x₂ ∧ x₁ ≠ A ∧ x₂ ≠ A ∧ ((x₁ = B ∨ x₁ = D) ∧ (x₂ = B ∨ x₂ = D))) ∧ (¬A ∨ B ∨ C ∨ D)

theorem guilty_D (A B C D : Prop) (h : isGuilty A B C D) : D :=
by
  sorry

end guilty_D_l134_134854


namespace binomial_inequality_l134_134248

theorem binomial_inequality (n : ℕ) (x : ℝ) (h1 : 2 ≤ n) (h2 : |x| < 1) : 
  (1 - x)^n + (1 + x)^n < 2^n := 
by 
  sorry

end binomial_inequality_l134_134248


namespace area_triangle_BRS_l134_134454

def point := ℝ × ℝ
def x_intercept (p : point) : ℝ := p.1
def y_intercept (p : point) : ℝ := p.2

noncomputable def distance_from_y_axis (p : point) : ℝ := abs p.1

theorem area_triangle_BRS (B R S : point)
  (hB : B = (4, 10))
  (h_perp : ∃ m₁ m₂, m₁ * m₂ = -1)
  (h_sum_zero : x_intercept R + x_intercept S = 0)
  (h_dist : distance_from_y_axis B = 10) :
  ∃ area : ℝ, area = 60 := 
sorry

end area_triangle_BRS_l134_134454


namespace shoe_size_15_is_9point25_l134_134071

noncomputable def smallest_shoe_length (L : ℝ) := L
noncomputable def largest_shoe_length (L : ℝ) := L + 9 * (1/4 : ℝ)
noncomputable def length_ratio_condition (L : ℝ) := largest_shoe_length L = 1.30 * smallest_shoe_length L
noncomputable def shoe_length_size_15 (L : ℝ) := L + 7 * (1/4 : ℝ)

theorem shoe_size_15_is_9point25 : ∃ L : ℝ, length_ratio_condition L → shoe_length_size_15 L = 9.25 :=
by
  sorry

end shoe_size_15_is_9point25_l134_134071


namespace find_a_l134_134370

theorem find_a (a k : ℝ) (h1 : ∀ x, a * x^2 + 3 * x - k = 0 → x = 7) (h2 : k = 119) : a = 2 :=
by
  sorry

end find_a_l134_134370


namespace second_number_is_correct_l134_134583

theorem second_number_is_correct (A B C : ℝ) 
  (h1 : A + B + C = 157.5)
  (h2 : A / B = 14 / 17)
  (h3 : B / C = 2 / 3)
  (h4 : A - C = 12.75) : 
  B = 18.75 := 
sorry

end second_number_is_correct_l134_134583


namespace fill_tank_with_two_pipes_l134_134409

def Pipe (Rate : Type) := Rate

theorem fill_tank_with_two_pipes
  (capacity : ℝ)
  (three_pipes_fill_time : ℝ)
  (h1 : three_pipes_fill_time = 12)
  (pipe_rate : ℝ)
  (h2 : pipe_rate = capacity / 36) :
  2 * pipe_rate * 18 = capacity := 
by 
  sorry

end fill_tank_with_two_pipes_l134_134409


namespace gcd_1230_990_l134_134492

theorem gcd_1230_990 : Int.gcd 1230 990 = 30 := by
  sorry

end gcd_1230_990_l134_134492


namespace find_r_l134_134264

theorem find_r (k r : ℝ) (h1 : (5 = k * 3^r)) (h2 : (45 = k * 9^r)) : r = 2 :=
  sorry

end find_r_l134_134264


namespace resulting_polygon_sides_l134_134059

theorem resulting_polygon_sides :
  let square_sides := 4
  let pentagon_sides := 5
  let hexagon_sides := 6
  let heptagon_sides := 7
  let octagon_sides := 8
  let nonagon_sides := 9
  let decagon_sides := 10
  let shared_square_decagon := 2
  let shared_between_others := 2 * 5 -- 2 sides shared for pentagon to nonagon
  let total_shared_sides := shared_square_decagon + shared_between_others
  let total_unshared_sides := 
    square_sides + pentagon_sides + hexagon_sides + heptagon_sides + octagon_sides + nonagon_sides + decagon_sides
  total_unshared_sides - total_shared_sides = 37 := by
  sorry

end resulting_polygon_sides_l134_134059


namespace factorize_polynomial_l134_134943

theorem factorize_polynomial (x : ℝ) :
  x^4 + 2 * x^3 - 9 * x^2 - 2 * x + 8 = (x + 4) * (x - 2) * (x + 1) * (x - 1) :=
sorry

end factorize_polynomial_l134_134943


namespace find_k_b_l134_134863

-- Define the sets A and B
def A : Set (ℝ × ℝ) := { p | ∃ x y: ℝ, p = (x, y) }
def B : Set (ℝ × ℝ) := { p | ∃ x y: ℝ, p = (x, y) }

-- Define the mapping f
def f (p : ℝ × ℝ) (k b : ℝ) : ℝ × ℝ := (k * p.1, p.2 + b)

-- Define the conditions
def condition (f : (ℝ × ℝ) → ℝ × ℝ) :=
  f (3,1) = (6,2)

-- Statement: Prove that the values of k and b are 2 and 1 respectively
theorem find_k_b : ∃ (k b : ℝ), f (3, 1) k b = (6, 2) ∧ k = 2 ∧ b = 1 :=
by
  sorry

end find_k_b_l134_134863


namespace theorem_incorrect_statement_D_l134_134338

open Real

def incorrect_statement_D (φ : ℝ) (hφ : φ > 0) (x : ℝ) : Prop :=
  cos (2*x + φ) ≠ cos (2*(x - φ/2))

theorem theorem_incorrect_statement_D (φ : ℝ) (hφ : φ > 0) : 
  ∃ x : ℝ, incorrect_statement_D φ hφ x :=
by
  sorry

end theorem_incorrect_statement_D_l134_134338


namespace total_cost_is_correct_l134_134113

def num_children : ℕ := 5
def daring_children : ℕ := 3
def ferris_wheel_cost_per_child : ℕ := 5
def merry_go_round_cost_per_child : ℕ := 3
def ice_cream_cones_per_child : ℕ := 2
def ice_cream_cost_per_cone : ℕ := 8

def total_spent_on_ferris_wheel : ℕ := daring_children * ferris_wheel_cost_per_child
def total_spent_on_merry_go_round : ℕ := num_children * merry_go_round_cost_per_child
def total_spent_on_ice_cream : ℕ := num_children * ice_cream_cones_per_child * ice_cream_cost_per_cone

def total_spent : ℕ := total_spent_on_ferris_wheel + total_spent_on_merry_go_round + total_spent_on_ice_cream

theorem total_cost_is_correct : total_spent = 110 := by
  sorry

end total_cost_is_correct_l134_134113


namespace matrix_power_minus_l134_134856

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

end matrix_power_minus_l134_134856


namespace sum_of_absolute_values_of_coefficients_l134_134383

theorem sum_of_absolute_values_of_coefficients :
  ∀ (a a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℝ),
  (∀ x : ℝ, (1 - 3 * x) ^ 9 = a + a_1 * x + a_2 * x ^ 2 + a_3 * x ^ 3 + a_4 * x ^ 4 + a_5 * x ^ 5 + a_6 * x ^ 6 + a_7 * x ^ 7 + a_8 * x ^ 8 + a_9 * x ^ 9) →
  |a| + |a_1| + |a_2| + |a_3| + |a_4| + |a_5| + |a_6| + |a_7| + |a_8| + |a_9| = 4 ^ 9 :=
by
  intro a a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 h
  sorry

end sum_of_absolute_values_of_coefficients_l134_134383


namespace find_remainder_l134_134923

def dividend : ℝ := 17698
def divisor : ℝ := 198.69662921348313
def quotient : ℝ := 89
def remainder : ℝ := 14

theorem find_remainder :
  dividend = (divisor * quotient) + remainder :=
by 
  -- Placeholder proof
  sorry

end find_remainder_l134_134923


namespace ratio_sqrt5_over_5_l134_134655

noncomputable def radius_ratio (a b : ℝ) (h : π * b^2 - π * a^2 = 4 * π * a^2) : ℝ :=
a / b

theorem ratio_sqrt5_over_5 (a b : ℝ) (h : π * b^2 - π * a^2 = 4 * π * a^2) :
  radius_ratio a b h = 1 / Real.sqrt 5 := 
sorry

end ratio_sqrt5_over_5_l134_134655


namespace find_k_l134_134573

variable (c : ℝ) (k : ℝ)
variable (a : ℕ → ℝ) (S : ℕ → ℝ)

def geometric_sequence (a : ℕ → ℝ) (c : ℝ) : Prop :=
  ∀ n, a (n + 1) = c * a n

def sum_sequence (S : ℕ → ℝ) (k : ℝ) : Prop :=
  ∀ n, S n = 3^n + k

theorem find_k (c_ne_zero : c ≠ 0)
  (h_geo : geometric_sequence a c)
  (h_sum : sum_sequence S k)
  (h_a1 : a 1 = 3 + k)
  (h_a2 : a 2 = S 2 - S 1)
  (h_a3 : a 3 = S 3 - S 2) :
  k = -1 :=
sorry

end find_k_l134_134573


namespace find_first_hour_speed_l134_134529

variable (x : ℝ)

-- Conditions
def speed_second_hour : ℝ := 60
def average_speed_two_hours : ℝ := 102.5

theorem find_first_hour_speed (h1 : average_speed_two_hours = (x + speed_second_hour) / 2) : 
  x = 145 := 
by
  sorry

end find_first_hour_speed_l134_134529


namespace find_value_l134_134802

theorem find_value (x : ℝ) (h : x^2 - 2 * x = 1) : 2023 + 6 * x - 3 * x^2 = 2020 := 
by 
sorry

end find_value_l134_134802


namespace probability_triangle_or_circle_l134_134842

theorem probability_triangle_or_circle (total_figures triangles circles : ℕ) 
  (h1 : total_figures = 10) 
  (h2 : triangles = 4) 
  (h3 : circles = 3) : 
  (triangles + circles) / total_figures = 7 / 10 :=
by
  sorry

end probability_triangle_or_circle_l134_134842


namespace sum_of_numbers_le_1_1_l134_134098

theorem sum_of_numbers_le_1_1 :
  let nums := [1.4, 0.9, 1.2, 0.5, 1.3]
  let filtered := nums.filter (fun x => x <= 1.1)
  filtered.sum = 1.4 :=
by
  let nums := [1.4, 0.9, 1.2, 0.5, 1.3]
  let filtered := nums.filter (fun x => x <= 1.1)
  have : filtered = [0.9, 0.5] := sorry
  have : filtered.sum = 1.4 := sorry
  exact this

end sum_of_numbers_le_1_1_l134_134098


namespace alvin_earns_14_dollars_l134_134269

noncomputable def total_earnings (total_marbles : ℕ) (percent_white percent_black : ℚ)
  (price_white price_black price_colored : ℚ) : ℚ :=
  let white_marbles := percent_white * total_marbles
  let black_marbles := percent_black * total_marbles
  let colored_marbles := total_marbles - white_marbles - black_marbles
  (white_marbles * price_white) + (black_marbles * price_black) + (colored_marbles * price_colored)

theorem alvin_earns_14_dollars :
  total_earnings 100 (20/100) (30/100) 0.05 0.10 0.20 = 14 := by
  sorry

end alvin_earns_14_dollars_l134_134269


namespace min_a2_b2_c2_l134_134960

theorem min_a2_b2_c2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 2 * a + 3 * b + 5 * c = 100) : 
  a^2 + b^2 + c^2 ≥ (5000 / 19) :=
by
  sorry

end min_a2_b2_c2_l134_134960


namespace bed_width_is_4_feet_l134_134470

def total_bags : ℕ := 16
def soil_per_bag : ℕ := 4
def bed_length : ℝ := 8
def bed_height : ℝ := 1
def num_beds : ℕ := 2

theorem bed_width_is_4_feet :
  (total_bags * soil_per_bag / num_beds) = (bed_length * 4 * bed_height) :=
by
  sorry

end bed_width_is_4_feet_l134_134470


namespace spending_record_l134_134905

-- Definitions based on conditions
def deposit_record (x : ℤ) : ℤ := x
def spend_record (x : ℤ) : ℤ := -x

-- Theorem statement
theorem spending_record (x : ℤ) (hx : x = 500) : spend_record x = -500 := by
  sorry

end spending_record_l134_134905


namespace inequality_proof_l134_134831

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

end inequality_proof_l134_134831


namespace ratio_of_common_differences_l134_134330

variable (x y d1 d2 : ℝ)

theorem ratio_of_common_differences (d1_nonzero : d1 ≠ 0) (d2_nonzero : d2 ≠ 0) 
  (seq1 : x + 4 * d1 = y) (seq2 : x + 5 * d2 = y) : d1 / d2 = 5 / 4 := 
sorry

end ratio_of_common_differences_l134_134330


namespace employee_n_salary_l134_134536

variable (m n : ℝ)

theorem employee_n_salary 
  (h1 : m + n = 605) 
  (h2 : m = 1.20 * n) : 
  n = 275 :=
by
  sorry

end employee_n_salary_l134_134536


namespace find_petra_age_l134_134263

namespace MathProof
  -- Definitions of the given conditions
  variables (P M : ℕ)
  axiom sum_of_ages : P + M = 47
  axiom mother_age_relation : M = 2 * P + 14
  axiom mother_actual_age : M = 36

  -- The proof goal which we need to fill later
  theorem find_petra_age : P = 11 :=
  by
    -- Using the axioms we have
    sorry -- Proof steps, which you don't need to fill according to the instructions
end MathProof

end find_petra_age_l134_134263


namespace total_number_of_letters_l134_134249

def jonathan_first_name_letters : Nat := 8
def jonathan_surname_letters : Nat := 10
def sister_first_name_letters : Nat := 5
def sister_surname_letters : Nat := 10

theorem total_number_of_letters : 
  jonathan_first_name_letters + jonathan_surname_letters + sister_first_name_letters + sister_surname_letters = 33 := 
by 
  sorry

end total_number_of_letters_l134_134249


namespace percentage_increase_is_20_l134_134726

def number_of_students_this_year : ℕ := 960
def number_of_students_last_year : ℕ := 800

theorem percentage_increase_is_20 :
  ((number_of_students_this_year - number_of_students_last_year : ℕ) / number_of_students_last_year * 100) = 20 := 
by
  sorry

end percentage_increase_is_20_l134_134726


namespace pie_slices_l134_134236

theorem pie_slices (total_pies : ℕ) (sold_pies : ℕ) (gifted_pies : ℕ) (left_pieces : ℕ) (eaten_fraction : ℚ) :
  total_pies = 4 →
  sold_pies = 1 →
  gifted_pies = 1 →
  eaten_fraction = 2/3 →
  left_pieces = 4 →
  (total_pies - sold_pies - gifted_pies) * (left_pieces * 3 / (1 - eaten_fraction)) / (total_pies - sold_pies - gifted_pies) = 6 :=
by
  sorry

end pie_slices_l134_134236


namespace ratio_QP_l134_134713

noncomputable def P : ℚ := 11 / 6
noncomputable def Q : ℚ := 5 / 2

theorem ratio_QP : Q / P = 15 / 11 := by 
  sorry

end ratio_QP_l134_134713


namespace parking_lot_perimeter_l134_134265

theorem parking_lot_perimeter (a b : ℝ) (h₁ : a^2 + b^2 = 625) (h₂ : a * b = 168) :
  2 * (a + b) = 62 :=
sorry

end parking_lot_perimeter_l134_134265


namespace total_amount_shared_l134_134799

theorem total_amount_shared (jane mike nora total : ℝ) 
  (h1 : jane = 30) 
  (h2 : jane / 2 = mike / 3) 
  (h3 : mike / 3 = nora / 8) 
  (h4 : total = jane + mike + nora) : 
  total = 195 :=
by
  sorry

end total_amount_shared_l134_134799


namespace interest_first_year_correct_interest_second_year_correct_interest_third_year_correct_l134_134215

noncomputable def principal_first_year : ℝ := 9000
noncomputable def interest_rate_first_year : ℝ := 0.09
noncomputable def principal_second_year : ℝ := principal_first_year * (1 + interest_rate_first_year)
noncomputable def interest_rate_second_year : ℝ := 0.105
noncomputable def principal_third_year : ℝ := principal_second_year * (1 + interest_rate_second_year)
noncomputable def interest_rate_third_year : ℝ := 0.085

noncomputable def compute_interest (principal : ℝ) (rate : ℝ) : ℝ :=
  principal * rate

theorem interest_first_year_correct :
  compute_interest principal_first_year interest_rate_first_year = 810 := by
  sorry

theorem interest_second_year_correct :
  compute_interest principal_second_year interest_rate_second_year = 1034.55 := by
  sorry

theorem interest_third_year_correct :
  compute_interest principal_third_year interest_rate_third_year = 922.18 := by
  sorry

end interest_first_year_correct_interest_second_year_correct_interest_third_year_correct_l134_134215


namespace sufficiency_s_for_q_l134_134738

variables {q r s : Prop}

theorem sufficiency_s_for_q (h₁ : r → q) (h₂ : ¬(q → r)) (h₃ : r ↔ s) : s → q ∧ ¬(q → s) :=
by
  sorry

end sufficiency_s_for_q_l134_134738


namespace math_problem_l134_134591

noncomputable def x : ℝ := -2

def A (x : ℝ) : Set ℝ := {1, 3, x^2}
def B (x : ℝ) : Set ℝ := {1, 2 - x}
def C1 : Set ℝ := {1, 3}
def C2 : Set ℝ := {3, 4}

theorem math_problem
  (h1 : B x ⊆ A x) :
  x = -2 ∧ (B x ∪ C1 = A x ∨ B x ∪ C2 = A x) :=
by
  sorry

end math_problem_l134_134591


namespace jerry_total_logs_l134_134126

-- Given conditions
def pine_logs_per_tree := 80
def maple_logs_per_tree := 60
def walnut_logs_per_tree := 100

def pine_trees_cut := 8
def maple_trees_cut := 3
def walnut_trees_cut := 4

-- Formulate the problem
theorem jerry_total_logs : 
  pine_logs_per_tree * pine_trees_cut + 
  maple_logs_per_tree * maple_trees_cut + 
  walnut_logs_per_tree * walnut_trees_cut = 1220 := 
by 
  -- Placeholder for the actual proof
  sorry

end jerry_total_logs_l134_134126


namespace total_bottles_in_market_l134_134199

theorem total_bottles_in_market (j w : ℕ) (hj : j = 34) (hw : w = 3 / 2 * j + 3) : j + w = 88 :=
by
  sorry

end total_bottles_in_market_l134_134199


namespace S_contains_finite_but_not_infinite_arith_progressions_l134_134664

noncomputable def S : Set ℤ := {n | ∃ k : ℕ, n = Int.floor (k * Real.pi)}

theorem S_contains_finite_but_not_infinite_arith_progressions :
  (∀ (k : ℕ), ∃ (a d : ℤ), ∀ (i : ℕ) (h : i < k), (a + i * d) ∈ S) ∧
  ¬(∃ (a d : ℤ), ∀ (n : ℕ), (a + n * d) ∈ S) :=
by
  sorry

end S_contains_finite_but_not_infinite_arith_progressions_l134_134664


namespace second_train_further_l134_134027

-- Define the speeds of the two trains
def speed_train1 : ℝ := 50
def speed_train2 : ℝ := 60

-- Define the total distance between points A and B
def total_distance : ℝ := 1100

-- Define the distances traveled by the two trains when they meet
def distance_train1 (t: ℝ) : ℝ := speed_train1 * t
def distance_train2 (t: ℝ) : ℝ := speed_train2 * t

-- Define the meeting condition
def meeting_condition (t: ℝ) : Prop := distance_train1 t + distance_train2 t = total_distance

-- Prove the distance difference
theorem second_train_further (t: ℝ) (h: meeting_condition t) : distance_train2 t - distance_train1 t = 100 :=
sorry

end second_train_further_l134_134027


namespace sandra_coffee_l134_134067

theorem sandra_coffee (S : ℕ) (H1 : 2 + S = 8) : S = 6 :=
by
  sorry

end sandra_coffee_l134_134067


namespace sum_of_solutions_l134_134080

theorem sum_of_solutions (a b c : ℝ) (h : a = 1 ∧ b = -9 ∧ c = 20 ∧ ∀ x, a * x^2 + b * x + c = 0) : 
  -b / a = 9 :=
by
  -- The proof is omitted here (hence the 'sorry')
  sorry

end sum_of_solutions_l134_134080


namespace proposition_true_and_negation_false_l134_134303

theorem proposition_true_and_negation_false (a b : ℝ) : 
  (a + b ≥ 2 → (a ≥ 1 ∨ b ≥ 1)) ∧ ¬(a + b ≥ 2 → (a < 1 ∧ b < 1)) :=
by {
  sorry
}

end proposition_true_and_negation_false_l134_134303


namespace final_score_is_83_l134_134944

def running_score : ℕ := 90
def running_weight : ℚ := 0.5

def fancy_jump_rope_score : ℕ := 80
def fancy_jump_rope_weight : ℚ := 0.3

def jump_rope_score : ℕ := 70
def jump_rope_weight : ℚ := 0.2

noncomputable def final_score : ℚ := 
  running_score * running_weight + 
  fancy_jump_rope_score * fancy_jump_rope_weight + 
  jump_rope_score * jump_rope_weight

theorem final_score_is_83 : final_score = 83 := 
  by
    sorry

end final_score_is_83_l134_134944


namespace cash_refund_per_bottle_l134_134251

-- Define the constants based on the conditions
def bottles_per_month : ℕ := 15
def cost_per_bottle : ℝ := 3.0
def bottles_can_buy_with_refund : ℕ := 6
def months_per_year : ℕ := 12

-- Define the total number of bottles consumed in a year
def total_bottles_per_year : ℕ := bottles_per_month * months_per_year

-- Define the total refund in dollars after 1 year
def total_refund_amount : ℝ := bottles_can_buy_with_refund * cost_per_bottle

-- Define the statement we need to prove
theorem cash_refund_per_bottle :
  total_refund_amount / total_bottles_per_year = 0.10 :=
by
  -- This is where the steps would be completed to prove the theorem
  sorry

end cash_refund_per_bottle_l134_134251


namespace sum_of_first_2n_terms_l134_134291

-- Definitions based on conditions
variable (n : ℕ) (S : ℕ → ℝ)

-- Conditions
def condition1 : Prop := S n = 24
def condition2 : Prop := S (3 * n) = 42

-- Statement to be proved
theorem sum_of_first_2n_terms {n : ℕ} (S : ℕ → ℝ) 
    (h1 : S n = 24) (h2 : S (3 * n) = 42) : S (2 * n) = 36 := by
  sorry

end sum_of_first_2n_terms_l134_134291


namespace gcd_lcm_product_eq_abc_l134_134845

theorem gcd_lcm_product_eq_abc (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) :
  let D := Nat.gcd (Nat.gcd a b) c
  let m := Nat.lcm (Nat.lcm a b) c
  D * m = a * b * c :=
by
  sorry

end gcd_lcm_product_eq_abc_l134_134845


namespace meaningful_expression_condition_l134_134797

theorem meaningful_expression_condition (x : ℝ) : (x > 1) ↔ (∃ y : ℝ, y = 2 / Real.sqrt (x - 1)) :=
by
  sorry

end meaningful_expression_condition_l134_134797


namespace reciprocal_of_negative_one_sixth_l134_134659

theorem reciprocal_of_negative_one_sixth : ∃ x : ℚ, - (1/6) * x = 1 ∧ x = -6 :=
by
  use -6
  constructor
  . sorry -- Need to prove - (1 / 6) * (-6) = 1
  . sorry -- Need to verify x = -6

end reciprocal_of_negative_one_sixth_l134_134659


namespace part1_solution_part2_solution_l134_134589

def part1 (m : ℝ) (x1 : ℝ) (x2 : ℝ) : Prop :=
  (m * x1 - 2) * (m * x2 - 2) = 4

theorem part1_solution : part1 (1/3) 9 18 :=
by 
  sorry

def part2 (m x1 x2 : ℕ) : Prop :=
  ((m * x1 - 2) * (m * x2 - 2) = 4)

def count_pairs : ℕ := 7

theorem part2_solution 
  (m x1 x2 : ℕ) 
  (h_pos : m > 0 ∧ x1 > 0 ∧ x2 > 0) : 
  ∃ c, c = count_pairs ∧ 
  (part2 m x1 x2) :=
by 
  sorry

end part1_solution_part2_solution_l134_134589


namespace max_paths_from_A_to_F_l134_134062

-- Define the points and line segments.
inductive Point
| A | B | C | D | E | F

-- Define the edges of the graph as pairs of points.
def edges : List (Point × Point) :=
  [(Point.A, Point.B), (Point.A, Point.E), (Point.A, Point.D),
   (Point.B, Point.C), (Point.B, Point.E),
   (Point.C, Point.F),
   (Point.D, Point.E), (Point.D, Point.F),
   (Point.E, Point.F)]

-- A path is valid if it passes through each point and line segment only once.
def valid_path (path : List (Point × Point)) : Bool :=
  -- Check that each edge in the path is unique and forms a sequence from A to F.
  sorry

-- Calculate the maximum number of different valid paths from point A to point F.
def max_paths : Nat :=
  List.length (List.filter valid_path (List.permutations edges))

theorem max_paths_from_A_to_F : max_paths = 9 :=
by sorry

end max_paths_from_A_to_F_l134_134062


namespace train_length_is_300_l134_134511

noncomputable def speed_kmph : Float := 90
noncomputable def speed_mps : Float := (speed_kmph * 1000) / 3600
noncomputable def time_sec : Float := 12
noncomputable def length_of_train : Float := speed_mps * time_sec

theorem train_length_is_300 : length_of_train = 300 := by
  sorry

end train_length_is_300_l134_134511


namespace cone_cube_volume_ratio_l134_134546

theorem cone_cube_volume_ratio (s : ℝ) (h : ℝ) (r : ℝ) (π : ℝ) 
  (cone_inscribed_in_cube : r = s / 2 ∧ h = s ∧ π > 0) :
  ((1/3) * π * r^2 * h) / (s^3) = π / 12 :=
by
  sorry

end cone_cube_volume_ratio_l134_134546


namespace johns_subtraction_l134_134007

theorem johns_subtraction 
  (a : ℕ) 
  (h₁ : (51 : ℕ)^2 = (50 : ℕ)^2 + 101) 
  (h₂ : (49 : ℕ)^2 = (50 : ℕ)^2 - b) 
  : b = 99 := 
by 
  sorry

end johns_subtraction_l134_134007


namespace qy_length_l134_134162

theorem qy_length (Q : Type*) (C : Type*) (X Y Z : Q) (QX QZ QY : ℝ) 
  (h1 : 5 = QX)
  (h2 : QZ = 2 * (QY - QX))
  (PQ_theorem : QX * QY = QZ^2) :
  QY = 10 :=
by
  sorry

end qy_length_l134_134162


namespace find_divided_number_l134_134116

-- Declare the constants and assumptions
variables (d q r : ℕ)
variables (n : ℕ)
variables (h_d : d = 20)
variables (h_q : q = 6)
variables (h_r : r = 2)
variables (h_def : n = d * q + r)

-- State the theorem we want to prove
theorem find_divided_number : n = 122 :=
by
  sorry

end find_divided_number_l134_134116


namespace second_discarded_number_l134_134438

theorem second_discarded_number (S : ℝ) (X : ℝ) (h1 : S / 50 = 62) (h2 : (S - 45 - X) / 48 = 62.5) : X = 55 := 
by
  sorry

end second_discarded_number_l134_134438


namespace equations_solutions_l134_134680

-- Definition and statement for Equation 1
noncomputable def equation1_solution1 : ℝ :=
  (-3 + Real.sqrt 17) / 4

noncomputable def equation1_solution2 : ℝ :=
  (-3 - Real.sqrt 17) / 4

-- Definition and statement for Equation 2
def equation2_solution : ℝ :=
  -6

-- Theorem proving the solutions to the given equations
theorem equations_solutions :
  (∃ x : ℝ, 2 * x^2 + 3 * x = 1 ∧ (x = equation1_solution1 ∨ x = equation1_solution2)) ∧
  (∃ x : ℝ, 3 / (x - 2) = 5 / (2 - x) - 1 ∧ x = equation2_solution) :=
by
  sorry

end equations_solutions_l134_134680


namespace seminar_total_cost_l134_134176

theorem seminar_total_cost 
  (regular_fee : ℝ)
  (discount_rate : ℝ)
  (num_teachers : ℕ) 
  (food_allowance_per_teacher : ℝ)
  (total_cost : ℝ)
  (h1 : regular_fee = 150)
  (h2 : discount_rate = 0.05)
  (h3 : num_teachers = 10) 
  (h4 : food_allowance_per_teacher = 10)
  (h5 : total_cost = regular_fee * num_teachers * (1 - discount_rate) + food_allowance_per_teacher * num_teachers) :
  total_cost = 1525 := 
sorry

end seminar_total_cost_l134_134176


namespace only_rational_root_is_one_l134_134209

-- Define the polynomial
def polynomial_3x5_minus_2x4_plus_5x3_minus_x2_minus_7x_plus_2 (x : ℚ) : ℚ :=
  3 * x^5 - 2 * x^4 + 5 * x^3 - x^2 - 7 * x + 2

-- The main theorem stating that 1 is the only rational root
theorem only_rational_root_is_one : 
  ∀ x : ℚ, polynomial_3x5_minus_2x4_plus_5x3_minus_x2_minus_7x_plus_2 x = 0 ↔ x = 1 :=
by
  sorry

end only_rational_root_is_one_l134_134209


namespace maximum_value_of_f_l134_134737

noncomputable def f : ℝ → ℝ :=
  fun x => -x^2 * (x^2 + 4*x + 4)

theorem maximum_value_of_f :
  ∀ x : ℝ, x ≠ 0 → x ≠ -2 → x ≠ 1 → x ≠ -3 → f x ≤ 0 ∧ f 0 = 0 :=
by
  sorry

end maximum_value_of_f_l134_134737


namespace locus_C2_angle_measure_90_l134_134429

variable (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : b < a)

-- Conditions for Question 1
def ellipse_C1 (x y : ℝ) : Prop := 
  (x^2 / a^2) + (y^2 / b^2) = 1

variable (x0 y0 x1 y1 : ℝ)
variable (hA : ellipse_C1 a b x0 y0)
variable (hE : ellipse_C1 a b x1 y1)
variable (h_perpendicular : x1 * x0 + y1 * y0 = 0)

theorem locus_C2 :
  ∀ (x y : ℝ), ellipse_C1 a b x y → 
  x ≠ 0 → y ≠ 0 → 
  (x^2 / a^2 + y^2 / b^2 = (a^2 - b^2)^2 / (a^2 + b^2)^2) := 
sorry

-- Conditions for Question 2
def circle_C3 (x y : ℝ) : Prop := 
  x^2 + y^2 = 1

theorem angle_measure_90 :
  (a^2 + b^2)^3 = a^2 * b^2 * (a^2 - b^2)^2 → 
  ∀ (x y : ℝ), ellipse_C1 a b x y → 
  circle_C3 x y → 
  (∃ (theta : ℝ), θ = 90) := 
sorry

end locus_C2_angle_measure_90_l134_134429


namespace denise_spent_l134_134578

theorem denise_spent (price_simple : ℕ) (price_meat : ℕ) (price_fish : ℕ)
  (price_milk_smoothie : ℕ) (price_fruit_smoothie : ℕ) (price_special_smoothie : ℕ)
  (julio_spent_more : ℕ) :
  price_simple = 7 →
  price_meat = 11 →
  price_fish = 14 →
  price_milk_smoothie = 6 →
  price_fruit_smoothie = 7 →
  price_special_smoothie = 9 →
  julio_spent_more = 6 →
  ∃ (d_price : ℕ), (d_price = 14 ∨ d_price = 17) :=
by
  sorry

end denise_spent_l134_134578


namespace steps_already_climbed_l134_134257

-- Definitions based on conditions
def total_stair_steps : ℕ := 96
def steps_left_to_climb : ℕ := 22

-- Theorem proving the number of steps already climbed
theorem steps_already_climbed : total_stair_steps - steps_left_to_climb = 74 := by
  sorry

end steps_already_climbed_l134_134257


namespace geometric_series_sum_l134_134568

theorem geometric_series_sum :
  ∀ (a r : ℤ) (n : ℕ),
  a = 3 → r = -2 → n = 10 →
  (a * ((r ^ n - 1) / (r - 1))) = -1024 :=
by
  intros a r n ha hr hn
  rw [ha, hr, hn]
  sorry

end geometric_series_sum_l134_134568


namespace discount_for_multiple_rides_l134_134306

-- Definitions based on given conditions
def ferris_wheel_cost : ℝ := 2.0
def roller_coaster_cost : ℝ := 7.0
def coupon_value : ℝ := 1.0
def total_tickets_needed : ℝ := 7.0

-- The proof problem
theorem discount_for_multiple_rides : 
  (ferris_wheel_cost + roller_coaster_cost) - (total_tickets_needed - coupon_value) = 2.0 :=
by
  sorry

end discount_for_multiple_rides_l134_134306


namespace problem_statement_l134_134418

noncomputable def f (x k : ℝ) := x^3 / (2^x + k * 2^(-x))

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def k2_eq_1_is_nec_but_not_suff (f : ℝ → ℝ) (k : ℝ) : Prop :=
  (k^2 = 1) → (is_even_function f → k = -1 ∧ ¬(k = 1))

theorem problem_statement (k : ℝ) :
  k2_eq_1_is_nec_but_not_suff (λ x => f x k) k :=
by
  sorry

end problem_statement_l134_134418


namespace find_k_value_l134_134537

theorem find_k_value (k : ℝ) : 
  (∃ x1 x2 x3 x4 : ℝ,
    x1 ≠ 0 ∧ x2 ≠ 0 ∧ x3 ≠ 0 ∧ x4 ≠ 0 ∧
    (x1^2 - 1) * (x1^2 - 4) = k ∧
    (x2^2 - 1) * (x2^2 - 4) = k ∧
    (x3^2 - 1) * (x3^2 - 4) = k ∧
    (x4^2 - 1) * (x4^2 - 4) = k ∧
    x1 ≠ x2 ∧ x1 < x2 ∧ x2 < x3 ∧ x3 < x4 ∧
    x4 - x3 = x3 - x2 ∧ x2 - x1 = x4 - x3) → 
  k = 7/4 := 
by
  sorry

end find_k_value_l134_134537


namespace paint_two_faces_red_l134_134734

theorem paint_two_faces_red (f : Fin 8 → ℕ) (H : ∀ i, 1 ≤ f i ∧ f i ≤ 8) : 
  (∃ pair_count : ℕ, pair_count = 9 ∧
    ∀ i j, i < j → f i + f j ≤ 7 → true) :=
sorry

end paint_two_faces_red_l134_134734


namespace find_a_l134_134456

noncomputable def A (a : ℝ) : ℝ × ℝ := (a, 2)
def B : ℝ × ℝ := (5, 1)
noncomputable def C (a : ℝ) : ℝ × ℝ := (-4, 2 * a)

def collinear (A B C : ℝ × ℝ) : Prop :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

theorem find_a (a : ℝ) : collinear (A a) B (C a) ↔ a = 4 :=
by
  sorry

end find_a_l134_134456


namespace joe_eats_different_fruits_l134_134961

noncomputable def joe_probability : ℚ :=
  let single_fruit_prob := (1 / 3) ^ 4
  let all_same_fruit_prob := 3 * single_fruit_prob
  let at_least_two_diff_fruits_prob := 1 - all_same_fruit_prob
  at_least_two_diff_fruits_prob

theorem joe_eats_different_fruits :
  joe_probability = 26 / 27 :=
by
  -- The proof is omitted for this task
  sorry

end joe_eats_different_fruits_l134_134961


namespace simplify_expression_l134_134506

theorem simplify_expression (p q r s : ℝ) (hp : p ≠ 6) (hq : q ≠ 7) (hr : r ≠ 8) (hs : s ≠ 9) :
    (p - 6) / (8 - r) * (q - 7) / (6 - p) * (r - 8) / (7 - q) * (s - 9) / (9 - s) = 1 := by
  sorry

end simplify_expression_l134_134506


namespace zero_is_a_root_of_polynomial_l134_134567

theorem zero_is_a_root_of_polynomial :
  (12 * (0 : ℝ)^4 + 38 * (0)^3 - 51 * (0)^2 + 40 * (0) = 0) :=
by simp

end zero_is_a_root_of_polynomial_l134_134567


namespace height_of_parallelogram_l134_134175

theorem height_of_parallelogram (Area Base : ℝ) (h1 : Area = 180) (h2 : Base = 18) : Area / Base = 10 :=
by
  sorry

end height_of_parallelogram_l134_134175


namespace mango_rate_is_50_l134_134425

theorem mango_rate_is_50 (quantity_grapes kg_grapes_perkg quantity_mangoes total_paid cost_grapes cost_mangoes rate_mangoes : ℕ) 
  (h1 : quantity_grapes = 8) 
  (h2 : kg_grapes_perkg = 70) 
  (h3 : quantity_mangoes = 9) 
  (h4 : total_paid = 1010)
  (h5 : cost_grapes = quantity_grapes * kg_grapes_perkg)
  (h6 : cost_mangoes = total_paid - cost_grapes)
  (h7 : rate_mangoes = cost_mangoes / quantity_mangoes) : 
  rate_mangoes = 50 :=
by sorry

end mango_rate_is_50_l134_134425


namespace sample_size_l134_134255

theorem sample_size (F n : ℕ) (FR : ℚ) (h1: F = 36) (h2: FR = 1/4) (h3: FR = F / n) : n = 144 :=
by 
  sorry

end sample_size_l134_134255


namespace fraction_zero_implies_x_half_l134_134351

theorem fraction_zero_implies_x_half (x : ℝ) (h₁ : (2 * x - 1) / (x + 2) = 0) (h₂ : x ≠ -2) : x = 1 / 2 :=
by sorry

end fraction_zero_implies_x_half_l134_134351


namespace kaleb_candy_problem_l134_134445

-- Define the initial problem with given conditions

theorem kaleb_candy_problem :
  ∀ (total_boxes : ℕ) (given_away_boxes : ℕ) (pieces_per_box : ℕ),
    total_boxes = 14 →
    given_away_boxes = 5 →
    pieces_per_box = 6 →
    (total_boxes - given_away_boxes) * pieces_per_box = 54 :=
by
  intros total_boxes given_away_boxes pieces_per_box
  intros h1 h2 h3
  -- Use assumptions
  sorry

end kaleb_candy_problem_l134_134445


namespace ratio_rate_down_to_up_l134_134798

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

end ratio_rate_down_to_up_l134_134798


namespace power_addition_identity_l134_134185

theorem power_addition_identity : 
  (-2)^23 + 5^(2^4 + 3^3 - 4^2) = -8388608 + 5^27 := by
  sorry

end power_addition_identity_l134_134185


namespace draws_alternate_no_consecutive_same_color_l134_134297

-- Defining the total number of balls and the count of each color.
def total_balls : ℕ := 15
def white_balls : ℕ := 5
def black_balls : ℕ := 5
def red_balls : ℕ := 5

-- Defining the probability that the draws alternate in colors with no two consecutive balls of the same color.
def probability_no_consecutive_same_color : ℚ := 162 / 1001

theorem draws_alternate_no_consecutive_same_color :
  (white_balls + black_balls + red_balls = total_balls) →
  -- The resulting probability based on the given conditions.
  probability_no_consecutive_same_color = 162 / 1001 := by
  sorry

end draws_alternate_no_consecutive_same_color_l134_134297


namespace problem_statement_l134_134523

noncomputable def myFunction (f : ℝ → ℝ) := 
  (∀ x y : ℝ, f (x^3 + y^3) = (x + y) * (f x ^ 2 - f x * f y + f y ^ 2)) 

theorem problem_statement (f : ℝ → ℝ) 
  (h : myFunction f) : 
  ∀ x : ℝ, f (2005 * x) = 2005 * f x :=
sorry

end problem_statement_l134_134523


namespace proof_correct_word_choice_l134_134474

def sentence_completion_correct (word : String) : Prop :=
  "Most of them are kind, but " ++ word ++ " is so good to me as Bruce" = "Most of them are kind, but none is so good to me as Bruce"

theorem proof_correct_word_choice : 
  (sentence_completion_correct "none") → 
  ("none" = "none") := 
by
  sorry

end proof_correct_word_choice_l134_134474


namespace set_intersection_complement_l134_134855

-- Definitions corresponding to conditions
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | -1 < x ∧ x < 3}
def B : Set ℝ := {x | x > 1}

-- Statement to prove
theorem set_intersection_complement : A ∩ (U \ B) = {x | -1 < x ∧ x ≤ 1} := by
  sorry

end set_intersection_complement_l134_134855


namespace mutually_exclusive_iff_complementary_l134_134413

variables {Ω : Type} (A₁ A₂ : Set Ω) (S : Set Ω)

/-- Proposition A: Events A₁ and A₂ are mutually exclusive. -/
def mutually_exclusive : Prop := A₁ ∩ A₂ = ∅

/-- Proposition B: Events A₁ and A₂ are complementary. -/
def complementary : Prop := A₁ ∩ A₂ = ∅ ∧ A₁ ∪ A₂ = S

/-- Proposition A is a necessary but not sufficient condition for Proposition B. -/
theorem mutually_exclusive_iff_complementary :
  mutually_exclusive A₁ A₂ → (complementary A₁ A₂ S → mutually_exclusive A₁ A₂) ∧
  (¬(mutually_exclusive A₁ A₂ → complementary A₁ A₂ S)) :=
by
  sorry

end mutually_exclusive_iff_complementary_l134_134413


namespace find_number_of_girls_l134_134947

-- Definitions
variables (B G : ℕ)
variables (total children_holding_boys_hand children_holding_girls_hand : ℕ)
variables (children_counted_twice : ℕ)

-- Conditions
axiom cond1 : B + G = 40
axiom cond2 : children_holding_boys_hand = 22
axiom cond3 : children_holding_girls_hand = 30
axiom cond4 : total = 40

-- Goal
theorem find_number_of_girls (h : children_counted_twice = children_holding_boys_hand + children_holding_girls_hand - total) :
  G = 24 :=
sorry

end find_number_of_girls_l134_134947


namespace is_decreasing_on_interval_l134_134040

open Set Real

def f (x : ℝ) : ℝ := x^3 - x^2 - x

def f' (x : ℝ) : ℝ := 3 * x^2 - 2 * x - 1

theorem is_decreasing_on_interval :
  ∀ x ∈ Ioo (-1 / 3 : ℝ) 1, f' x < 0 :=
by
  intro x hx
  sorry

end is_decreasing_on_interval_l134_134040


namespace production_time_l134_134389

-- Define the conditions
def machineProductionRate (machines: ℕ) (units: ℕ) (hours: ℕ): ℕ := units / machines / hours

-- The question we need to answer: How long will it take for 10 machines to produce 100 units?
theorem production_time (h1 : machineProductionRate 5 20 10 = 4 / 10)
  : 10 * 0.4 * 25 = 100 :=
by sorry

end production_time_l134_134389


namespace ratio_of_areas_is_16_l134_134967

-- Definitions and conditions
variables (a b : ℝ)

-- Given condition: Perimeter of the larger square is 4 times the perimeter of the smaller square
def perimeter_relation (ha : a = 4 * b) : Prop := a = 4 * b

-- Theorem to prove: Ratio of the area of the larger square to the area of the smaller square is 16
theorem ratio_of_areas_is_16 (ha : a = 4 * b) : (a^2 / b^2) = 16 :=
by
  sorry

end ratio_of_areas_is_16_l134_134967


namespace train_length_l134_134717

theorem train_length (x : ℕ)
  (h1 : ∀ (x : ℕ), (790 + x) / 33 = (860 - x) / 22) : x = 200 := by
  sorry

end train_length_l134_134717


namespace sum_first_third_numbers_l134_134896

theorem sum_first_third_numbers (A B C : ℕ)
    (h1 : A + B + C = 98)
    (h2 : A * 3 = B * 2)
    (h3 : B * 8 = C * 5)
    (h4 : B = 30) :
    A + C = 68 :=
by
-- Data is sufficient to conclude that A + C = 68
sorry

end sum_first_third_numbers_l134_134896


namespace even_number_representation_l134_134462

-- Definitions for conditions
def even_number (k : Int) : Prop := ∃ m : Int, k = 2 * m
def perfect_square (n : Int) : Prop := ∃ p : Int, n = p * p
def sum_representation (a b : Int) : Prop := ∃ k : Int, a + b = 2 * k ∧ perfect_square (a * b)
def difference_representation (d k e : Int) : Prop := d * (d - 2 * k) = e * e

-- The theorem statement
theorem even_number_representation {k : Int} (hk : even_number k) :
  (∃ a b : Int, sum_representation a b ∧ 2 * k = a + b) ∨
  (∃ d e : Int, difference_representation d k e ∧ d ≠ 0) :=
sorry

end even_number_representation_l134_134462


namespace solve_modulo_problem_l134_134876

theorem solve_modulo_problem (n : ℤ) :
  0 ≤ n ∧ n < 19 ∧ 38574 % 19 = n % 19 → n = 4 := by
  sorry

end solve_modulo_problem_l134_134876


namespace quadratic_properties_l134_134698

theorem quadratic_properties (a b c : ℝ) (h1 : a < 0) (h2 : a * (-1 : ℝ)^2 + b * (-1 : ℝ) + c = 0) (h3 : -b = 2 * a) :
  (a - b + c = 0) ∧ 
  (∀ m : ℝ, a * m^2 + b * m + c ≤ -4 * a) ∧ 
  (∀ (x1 x2 : ℝ), (a * x1^2 + b * x1 + c + 1 = 0) ∧ (a * x2^2 + b * x2 + c + 1 = 0) ∧ x1 < x2 → x1 < -1 ∧ x2 > 3) :=
by
  sorry

end quadratic_properties_l134_134698


namespace sum_of_max_values_l134_134912

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (Real.sin x - Real.cos x)

theorem sum_of_max_values : (f π + f (3 * π)) = (Real.exp π + Real.exp (3 * π)) := 
by sorry

end sum_of_max_values_l134_134912


namespace find_u5_l134_134714

theorem find_u5 
  (u : ℕ → ℝ)
  (h_rec : ∀ n, u (n + 2) = 3 * u (n + 1) + 2 * u n)
  (h_u3 : u 3 = 9)
  (h_u6 : u 6 = 243) : 
  u 5 = 69 :=
sorry

end find_u5_l134_134714


namespace average_sequence_x_l134_134764

theorem average_sequence_x (x : ℚ) (h : (5050 + x) / 101 = 50 * x) : x = 5050 / 5049 :=
by
  sorry

end average_sequence_x_l134_134764


namespace maximum_z_l134_134253

theorem maximum_z (x y z : ℝ) (h1 : x + y + z = 5) (h2 : x * y + y * z + z * x = 3) : z ≤ 13 / 3 :=
sorry

end maximum_z_l134_134253


namespace minimum_value_of_f_l134_134495

noncomputable def f (x : ℝ) : ℝ := (4 * x^2 + 8 * x + 13) / (6 * (1 + Real.exp (-x)))

theorem minimum_value_of_f : ∀ x : ℝ, 0 ≤ x → f x ≥ f 0 :=
by
  intro x hx
  unfold f
  admit

end minimum_value_of_f_l134_134495


namespace franks_age_l134_134227

variable (F G : ℕ)

def gabriel_younger_than_frank : Prop := G = F - 3
def total_age_is_seventeen : Prop := F + G = 17

theorem franks_age (h1 : gabriel_younger_than_frank F G) (h2 : total_age_is_seventeen F G) : F = 10 :=
by
  sorry

end franks_age_l134_134227


namespace ellipse_standard_equation_l134_134819

theorem ellipse_standard_equation :
  ∃ (a b c : ℝ),
    2 * a = 10 ∧
    c / a = 3 / 5 ∧
    b^2 = a^2 - c^2 ∧
    (∀ x y : ℝ, (x^2 / 16) + (y^2 / 25) = 1) :=
by
  sorry

end ellipse_standard_equation_l134_134819


namespace markus_more_marbles_l134_134407

theorem markus_more_marbles :
  let mara_bags := 12
  let marbles_per_mara_bag := 2
  let markus_bags := 2
  let marbles_per_markus_bag := 13
  let mara_marbles := mara_bags * marbles_per_mara_bag
  let markus_marbles := markus_bags * marbles_per_markus_bag
  mara_marbles + 2 = markus_marbles := 
by
  sorry

end markus_more_marbles_l134_134407


namespace quadratic_roots_distinct_l134_134758

theorem quadratic_roots_distinct (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + 2*x1 + m = 0 ∧ x2^2 + 2*x2 + m = 0) →
  m < 1 := 
by
  sorry

end quadratic_roots_distinct_l134_134758


namespace fraction_product_cube_l134_134809

theorem fraction_product_cube :
  ((5 : ℚ) / 8)^3 * ((4 : ℚ) / 9)^3 = (125 : ℚ) / 5832 :=
by
  sorry

end fraction_product_cube_l134_134809


namespace g_of_1001_l134_134465

noncomputable def g : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) : g (x * y) + x = x * g y + g x
axiom g_of_1 : g 1 = -3

theorem g_of_1001 : g 1001 = -2001 := 
by sorry

end g_of_1001_l134_134465


namespace percentage_time_in_park_l134_134309

/-- Define the number of trips Laura takes to the park. -/
def number_of_trips : ℕ := 6

/-- Define time spent at the park per trip in hours. -/
def time_at_park_per_trip : ℝ := 2

/-- Define time spent walking per trip in hours. -/
def time_walking_per_trip : ℝ := 0.5

/-- Define the total time for all trips. -/
def total_time_for_all_trips : ℝ := (time_at_park_per_trip + time_walking_per_trip) * number_of_trips

/-- Define the total time spent in the park for all trips. -/
def total_time_in_park : ℝ := time_at_park_per_trip * number_of_trips

/-- Prove that the percentage of the total time spent in the park is 80%. -/
theorem percentage_time_in_park : total_time_in_park / total_time_for_all_trips * 100 = 80 :=
by
  sorry

end percentage_time_in_park_l134_134309


namespace sum_of_squares_bounds_l134_134510

theorem sum_of_squares_bounds (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hxy : x + y = 10) : 
  (x^2 + y^2 ≤ 100) ∧ (x^2 + y^2 ≥ 50) :=
by 
  sorry

end sum_of_squares_bounds_l134_134510


namespace minimal_fence_length_l134_134337

-- Define the conditions as assumptions
axiom side_length : ℝ
axiom num_paths : ℕ
axiom path_length : ℝ

-- Assume the conditions given in the problem
axiom side_length_value : side_length = 50
axiom num_paths_value : num_paths = 13
axiom path_length_value : path_length = 50

-- Define the theorem to be proved
theorem minimal_fence_length : (num_paths * path_length) = 650 := by
  -- The proof goes here
  sorry

end minimal_fence_length_l134_134337


namespace find_x_l134_134181

def vector_a : ℝ × ℝ := (1, 2)
def vector_b (x : ℝ) : ℝ × ℝ := (x, -1)
def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ := v₁.1 * v₂.1 + v₁.2 * v₂.2

theorem find_x
  (h : dot_product vector_a (vector_b x) = 0) :
  x = 2 :=
by
  sorry

end find_x_l134_134181


namespace distance_between_trees_l134_134157

theorem distance_between_trees (n : ℕ) (L : ℝ) (d : ℝ) (h1 : n = 26) (h2 : L = 700) (h3 : d = L / (n - 1)) : d = 28 :=
sorry

end distance_between_trees_l134_134157


namespace x_squared_plus_y_squared_l134_134607

theorem x_squared_plus_y_squared (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 2) : x^2 + y^2 = 21 := 
by 
  sorry

end x_squared_plus_y_squared_l134_134607


namespace right_triangle_perimeter_l134_134605

theorem right_triangle_perimeter (area leg1 : ℕ) (h_area : area = 180) (h_leg1 : leg1 = 30) :
  ∃ leg2 hypotenuse perimeter, 
    (2 * area = leg1 * leg2) ∧ 
    (hypotenuse^2 = leg1^2 + leg2^2) ∧ 
    (perimeter = leg1 + leg2 + hypotenuse) ∧ 
    (perimeter = 42 + 2 * Real.sqrt 261) :=
by
  sorry

end right_triangle_perimeter_l134_134605


namespace transistors_in_2010_l134_134140

-- Define initial conditions
def initial_transistors : ℕ := 500000
def years_passed : ℕ := 15
def tripling_period : ℕ := 3
def tripling_factor : ℕ := 3

-- Define the function to compute the number of transistors after a number of years
noncomputable def final_transistors (initial : ℕ) (years : ℕ) (period : ℕ) (factor : ℕ) : ℕ :=
  initial * factor ^ (years / period)

-- State the proposition we aim to prove
theorem transistors_in_2010 : final_transistors initial_transistors years_passed tripling_period tripling_factor = 121500000 := 
by 
  sorry

end transistors_in_2010_l134_134140


namespace max_value_of_f_l134_134237

def f (x : ℝ) : ℝ := 12 * x - 4 * x^2

theorem max_value_of_f : ∀ x : ℝ, f x ≤ 9 :=
by
  have h₁ : ∀ x : ℝ, 12 * x - 4 * x^2 ≤ 9
  { sorry }
  exact h₁

end max_value_of_f_l134_134237


namespace seedling_probability_l134_134186

theorem seedling_probability (germination_rate survival_rate : ℝ)
    (h_germ : germination_rate = 0.9) (h_survival : survival_rate = 0.8) : 
    germination_rate * survival_rate = 0.72 :=
by
  rw [h_germ, h_survival]
  norm_num

end seedling_probability_l134_134186


namespace factoring_sum_of_coefficients_l134_134661

theorem factoring_sum_of_coefficients 
  (a b c d e f g h j k : ℤ)
  (h1 : 64 * x^6 - 729 * y^6 = (a * x + b * y) * (c * x^2 + d * x * y + e * y^2) * (f * x + g * y) * (h * x^2 + j * x * y + k * y^2)) :
  a + b + c + d + e + f + g + h + j + k = 30 :=
sorry

end factoring_sum_of_coefficients_l134_134661


namespace inequality_problem_l134_134035

-- Define a and the condition that expresses the given problem as an inequality
variable (a : ℝ)

-- The inequality to prove
theorem inequality_problem : a - 5 > 2 * a := sorry

end inequality_problem_l134_134035


namespace hotel_cost_l134_134305

/--
Let the total cost of the hotel be denoted as x dollars.
Initially, the cost for each of the original four colleagues is x / 4.
After three more colleagues joined, the cost per person becomes x / 7.
Given that the amount paid by each of the original four decreased by 15,
prove that the total cost of the hotel is 140 dollars.
-/
theorem hotel_cost (x : ℕ) (h : x / 4 - 15 = x / 7) : x = 140 := 
by
  sorry

end hotel_cost_l134_134305


namespace david_recreation_l134_134364

theorem david_recreation (W : ℝ) (P : ℝ) 
  (h1 : 0.95 * W = this_week_wages) 
  (h2 : 0.5 * this_week_wages = recreation_this_week)
  (h3 : 1.1875 * (P / 100) * W = recreation_this_week) : P = 40 :=
sorry

end david_recreation_l134_134364


namespace diaries_ratio_l134_134631

variable (initial_diaries : ℕ)
variable (final_diaries : ℕ)
variable (lost_fraction : ℚ)
variable (bought_diaries : ℕ)

theorem diaries_ratio 
  (h1 : initial_diaries = 8)
  (h2 : final_diaries = 18)
  (h3 : lost_fraction = 1 / 4)
  (h4 : ∃ x : ℕ, (initial_diaries + x - lost_fraction * (initial_diaries + x) = final_diaries) ∧ x = 16) :
  (16 / initial_diaries : ℚ) = 2 := 
by
  sorry

end diaries_ratio_l134_134631


namespace unique_solution_l134_134104

def is_solution (f : ℝ → ℝ) : Prop :=
  ∀ (x : ℝ) (hx : 0 < x), 
    ∃! (y : ℝ) (hy : 0 < y), 
      x * f y + y * f x ≤ 2

theorem unique_solution : ∀ (f : ℝ → ℝ), 
  is_solution f ↔ (∀ x, 0 < x → f x = 1 / x) :=
by
  intros
  sorry

end unique_solution_l134_134104


namespace mark_reading_time_l134_134744

variable (x y : ℕ)

theorem mark_reading_time (x y : ℕ) : 
  7 * x + y = 7 * x + y :=
by
  sorry

end mark_reading_time_l134_134744


namespace interval_representation_l134_134815

def S : Set ℝ := {x | -1 < x ∧ x ≤ 3}

theorem interval_representation : S = Set.Ioc (-1) 3 :=
sorry

end interval_representation_l134_134815


namespace sarah_score_l134_134245

theorem sarah_score (s g : ℝ) (h1 : s = g + 50) (h2 : (s + g) / 2 = 110) : s = 135 := 
by 
  sorry

end sarah_score_l134_134245


namespace trig_identity_proof_l134_134962

theorem trig_identity_proof :
  (Real.cos (10 * Real.pi / 180) * Real.sin (70 * Real.pi / 180) - Real.cos (80 * Real.pi / 180) * Real.sin (20 * Real.pi / 180)) = (Real.sqrt 3 / 2) :=
by
  sorry

end trig_identity_proof_l134_134962


namespace yuna_has_biggest_number_l134_134388

-- Define the collections
def yoongi_collected : ℕ := 4
def jungkook_collected : ℕ := 6 - 3
def yuna_collected : ℕ := 5

-- State the theorem
theorem yuna_has_biggest_number :
  yuna_collected > yoongi_collected ∧ yuna_collected > jungkook_collected :=
by
  sorry

end yuna_has_biggest_number_l134_134388


namespace insurance_compensation_zero_l134_134942

noncomputable def insured_amount : ℝ := 500000
noncomputable def deductible : ℝ := 0.01
noncomputable def actual_damage : ℝ := 4000

theorem insurance_compensation_zero :
  actual_damage < insured_amount * deductible → 0 = 0 := by
sorry

end insurance_compensation_zero_l134_134942


namespace find_time_to_fill_tank_l134_134648

noncomputable def time_to_fill_tanker (TA : ℝ) : Prop :=
  let RB := 1 / 40
  let fill_time := 29.999999999999993
  let half_fill_time := fill_time / 2
  let RAB := (1 / TA) + RB
  (RAB * half_fill_time = 1 / 2) → (TA = 120)

theorem find_time_to_fill_tank : ∃ TA, time_to_fill_tanker TA :=
by
  use 120
  sorry

end find_time_to_fill_tank_l134_134648


namespace neg_exists_exp_l134_134515

theorem neg_exists_exp (p : Prop) :
  (¬ (∃ x : ℝ, Real.exp x < 0)) = (∀ x : ℝ, Real.exp x ≥ 0) :=
by
  sorry

end neg_exists_exp_l134_134515


namespace value_of_a_l134_134704

def quadratic_vertex (a b c : ℤ) (x : ℤ) : ℤ :=
  a * x^2 + b * x + c

def vertex_form (a h k x : ℤ) : ℤ :=
  a * (x - h)^2 + k

theorem value_of_a (a b c : ℤ) (h k x1 y1 x2 y2 : ℤ) (H_vert : h = 2) (H_vert_val : k = 3)
  (H_point : x1 = 1) (H_point_val : y1 = 5) (H_graph : ∀ x, quadratic_vertex a b c x = vertex_form a h k x) :
  a = 2 :=
by
  sorry

end value_of_a_l134_134704


namespace condition_s_for_q_condition_r_for_q_condition_p_for_s_l134_134212

variables {p q r s : Prop}

-- Given conditions from a)
axiom h₁ : r → p
axiom h₂ : q → r
axiom h₃ : s → r
axiom h₄ : q → s

-- The corresponding proof problems based on c)
theorem condition_s_for_q : (s ↔ q) :=
by sorry

theorem condition_r_for_q : (r ↔ q) :=
by sorry

theorem condition_p_for_s : (s → p) :=
by sorry

end condition_s_for_q_condition_r_for_q_condition_p_for_s_l134_134212


namespace avg_A_lt_avg_B_combined_avg_eq_6_6_l134_134859

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

end avg_A_lt_avg_B_combined_avg_eq_6_6_l134_134859


namespace factor_expression_l134_134643

theorem factor_expression (x : ℝ) : 16 * x^2 + 8 * x = 8 * x * (2 * x + 1) := 
by
  sorry

end factor_expression_l134_134643


namespace value_of_x_plus_inv_x_l134_134917

theorem value_of_x_plus_inv_x (x : ℝ) (h : x + (1 / x) = v) (hr : x^2 + (1 / x)^2 = 23) : v = 5 :=
sorry

end value_of_x_plus_inv_x_l134_134917


namespace triangle_area_gt_half_l134_134901

-- We are given two altitudes h_a and h_b such that both are greater than 1
variables {a h_a h_b : ℝ}

-- Conditions: h_a > 1 and h_b > 1
axiom ha_gt_one : h_a > 1
axiom hb_gt_one : h_b > 1

-- Prove that the area of the triangle is greater than 1/2
theorem triangle_area_gt_half :
  ∃ a : ℝ, a > 1 ∧ ∃ h_a : ℝ, h_a > 1 ∧ (1 / 2) * a * h_a > (1 / 2) :=
by {
  sorry
}

end triangle_area_gt_half_l134_134901


namespace seventh_term_arith_seq_l134_134503

/-- 
The seventh term of an arithmetic sequence given that the sum of the first five terms 
is 15 and the sixth term is 7.
-/
theorem seventh_term_arith_seq (a d : ℚ) 
  (h1 : 5 * a + 10 * d = 15) 
  (h2 : a + 5 * d = 7) : 
  a + 6 * d = 25 / 3 := 
sorry

end seventh_term_arith_seq_l134_134503


namespace plane_divided_by_n_lines_l134_134353

-- Definition of the number of regions created by n lines in a plane
def regions (n : ℕ) : ℕ :=
  if n = 0 then 1 else (n * (n + 1)) / 2 + 1 -- Using the given formula directly

-- Theorem statement to prove the formula holds
theorem plane_divided_by_n_lines (n : ℕ) : 
  regions n = (n * (n + 1)) / 2 + 1 :=
sorry

end plane_divided_by_n_lines_l134_134353


namespace total_people_counted_l134_134936

-- Definitions based on conditions
def people_second_day : ℕ := 500
def people_first_day : ℕ := 2 * people_second_day

-- Theorem statement
theorem total_people_counted : people_first_day + people_second_day = 1500 := 
by 
  sorry

end total_people_counted_l134_134936


namespace proof_set_intersection_l134_134207

def set_M := {x : ℝ | x^2 - 2*x - 8 ≤ 0}
def set_N := {x : ℝ | Real.log x ≥ 0}
def set_answer := {x : ℝ | 1 ≤ x ∧ x ≤ 4}

theorem proof_set_intersection : 
  (set_M ∩ set_N) = set_answer := 
by 
  sorry

end proof_set_intersection_l134_134207


namespace children_l134_134642

variable (C : ℝ) -- Define the weight of a children's book

theorem children's_book_weight :
  (9 * 0.8 + 7 * C = 10.98) → C = 0.54 :=
by  
sorry

end children_l134_134642


namespace quadratic_inequality_for_all_x_l134_134200

theorem quadratic_inequality_for_all_x (a : ℝ) :
  (∀ x : ℝ, (a^2 + a) * x^2 - a * x + 1 > 0) ↔ (-4 / 3 < a ∧ a < -1) ∨ a = 0 :=
sorry

end quadratic_inequality_for_all_x_l134_134200


namespace arithmetic_sequence_properties_l134_134695

/-- In an arithmetic sequence {a_n}, let S_n represent the sum of the first n terms, 
and it is given that S_6 < S_7 and S_7 > S_8. 
Prove that the correct statements among the given options are: 
1. The common difference d < 0 
2. S_9 < S_6 
3. S_7 is definitively the maximum value among all sums S_n. -/
theorem arithmetic_sequence_properties 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ)
  (h_arith_seq : ∀ n, S (n + 1) = S n + a (n + 1))
  (h_S6_lt_S7 : S 6 < S 7)
  (h_S7_gt_S8 : S 7 > S 8) :
  (a 7 > 0 ∧ a 8 < 0 ∧ ∃ d, ∀ n, a (n + 1) = a n + d ∧ d < 0 ∧ S 9 < S 6 ∧ ∀ n, S n ≤ S 7) :=
by
  -- Proof omitted
  sorry

end arithmetic_sequence_properties_l134_134695


namespace parakeets_per_cage_is_2_l134_134865

variables (cages : ℕ) (parrots_per_cage : ℕ) (total_birds : ℕ)

def number_of_parakeets_each_cage : ℕ :=
  (total_birds - cages * parrots_per_cage) / cages

theorem parakeets_per_cage_is_2
  (hcages : cages = 4)
  (hparrots_per_cage : parrots_per_cage = 8)
  (htotal_birds : total_birds = 40) :
  number_of_parakeets_each_cage cages parrots_per_cage total_birds = 2 := 
by
  sorry

end parakeets_per_cage_is_2_l134_134865


namespace number_of_months_in_season_l134_134742

def games_per_month : ℝ := 323.0
def total_games : ℝ := 5491.0

theorem number_of_months_in_season : total_games / games_per_month = 17 := 
by
  sorry

end number_of_months_in_season_l134_134742


namespace simplify_expression_l134_134256

theorem simplify_expression :
  20 * (14 / 15) * (2 / 18) * (5 / 4) = 70 / 27 :=
by sorry

end simplify_expression_l134_134256


namespace find_quadratic_eq_with_given_roots_l134_134320

theorem find_quadratic_eq_with_given_roots (A z x1 x2 : ℝ) 
  (h1 : A * z * x1^2 + x1 * x1 + x2 = 0) 
  (h2 : A * z * x2^2 + x1 * x2 + x2 = 0) : 
  (A * z * x^2 + x1 * x - x2 = 0) :=
by
  sorry

end find_quadratic_eq_with_given_roots_l134_134320


namespace remainder_when_four_times_n_minus_9_l134_134109

theorem remainder_when_four_times_n_minus_9
  (n : ℤ) (h : n % 5 = 3) : (4 * n - 9) % 5 = 3 := 
by 
  sorry

end remainder_when_four_times_n_minus_9_l134_134109


namespace find_k_l134_134004

noncomputable def f (a b c : ℤ) (x : ℤ) := a * x^2 + b * x + c

theorem find_k (a b c k : ℤ) 
  (h1 : f a b c 1 = 0) 
  (h2 : 50 < f a b c 7) (h2' : f a b c 7 < 60) 
  (h3 : 70 < f a b c 8) (h3' : f a b c 8 < 80) 
  (h4 : 5000 * k < f a b c 100) (h4' : f a b c 100 < 5000 * (k + 1)) : 
  k = 3 := 
sorry

end find_k_l134_134004


namespace ceil_sqrt_sum_l134_134838

theorem ceil_sqrt_sum :
  ⌈Real.sqrt 3⌉₊ + ⌈Real.sqrt 27⌉₊ + ⌈Real.sqrt 243⌉₊ = 24 :=
by
  have h1 : 1 < Real.sqrt 3 ∧ Real.sqrt 3 < 2 := by sorry
  have h2 : 5 < Real.sqrt 27 ∧ Real.sqrt 27 < 6 := by sorry
  have h3 : 15 < Real.sqrt 243 ∧ Real.sqrt 243 < 16 := by sorry
  sorry

end ceil_sqrt_sum_l134_134838


namespace contractor_fine_per_absent_day_l134_134467

noncomputable def fine_per_absent_day (total_days : ℕ) (pay_per_day : ℝ) (total_amount_received : ℝ) (days_absent : ℕ) : ℝ :=
  let days_worked := total_days - days_absent
  let earned := days_worked * pay_per_day
  let fine := (earned - total_amount_received) / days_absent
  fine

theorem contractor_fine_per_absent_day :
  fine_per_absent_day 30 25 425 10 = 7.5 := by
  sorry

end contractor_fine_per_absent_day_l134_134467


namespace collinear_points_value_l134_134788

/-- 
If the points (2, a, b), (a, 3, b), and (a, b, 4) are collinear, 
then the value of a + b is 7.
-/
theorem collinear_points_value (a b : ℝ) (h_collinear : ∃ l : ℝ → ℝ × ℝ × ℝ, 
  l 0 = (2, a, b) ∧ l 1 = (a, 3, b) ∧ l 2 = (a, b, 4) ∧ 
  ∀ t s : ℝ, l t = l s → t = s) :
  a + b = 7 :=
sorry

end collinear_points_value_l134_134788


namespace weekly_crab_meat_cost_l134_134374

-- Declare conditions as definitions
def dishes_per_day : ℕ := 40
def pounds_per_dish : ℝ := 1.5
def cost_per_pound : ℝ := 8
def closed_days_per_week : ℕ := 3
def days_per_week : ℕ := 7

-- Define the Lean statement to prove the weekly cost
theorem weekly_crab_meat_cost :
  let days_open_per_week := days_per_week - closed_days_per_week
  let pounds_per_day := dishes_per_day * pounds_per_dish
  let daily_cost := pounds_per_day * cost_per_pound
  let weekly_cost := daily_cost * (days_open_per_week : ℝ)
  weekly_cost = 1920 :=
by
  sorry

end weekly_crab_meat_cost_l134_134374


namespace possible_values_of_n_l134_134765

theorem possible_values_of_n :
  let a := 1500
  let max_r2 := 562499
  let total := max_r2
  let perfect_squares := (750 : Nat)
  total - perfect_squares = 561749 := by
    sorry

end possible_values_of_n_l134_134765


namespace concentric_circles_ratio_l134_134581

theorem concentric_circles_ratio (R r k : ℝ) (hr : r > 0) (hRr : R > r) (hk : k > 0)
  (area_condition : π * (R^2 - r^2) = k * π * r^2) :
  R / r = Real.sqrt (k + 1) :=
by
  sorry

end concentric_circles_ratio_l134_134581


namespace all_fruits_fallen_by_twelfth_day_l134_134635

noncomputable def magical_tree_falling_day : Nat :=
  let total_fruits := 58
  let initial_day_falls := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].foldl (· + ·) 0
  let continuation_falls := [1, 2].foldl (· + ·) 0
  let total_days := initial_day_falls + continuation_falls
  12

theorem all_fruits_fallen_by_twelfth_day :
  magical_tree_falling_day = 12 :=
by
  sorry

end all_fruits_fallen_by_twelfth_day_l134_134635


namespace additional_life_vests_needed_l134_134480

def num_students : ℕ := 40
def num_instructors : ℕ := 10
def life_vests_on_hand : ℕ := 20
def percent_students_with_vests : ℕ := 20

def total_people : ℕ := num_students + num_instructors
def students_with_vests : ℕ := (percent_students_with_vests * num_students) / 100
def total_vests_available : ℕ := life_vests_on_hand + students_with_vests

theorem additional_life_vests_needed : 
  total_people - total_vests_available = 22 :=
by 
  sorry

end additional_life_vests_needed_l134_134480


namespace fraction_inequality_l134_134800

theorem fraction_inequality (a b m : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : m > 0) : 
  (b / a) < (b + m) / (a + m) := 
sorry

end fraction_inequality_l134_134800


namespace units_digit_of_x_l134_134102

theorem units_digit_of_x 
  (a x : ℕ) 
  (h1 : a * x = 14^8) 
  (h2 : a % 10 = 9) : 
  x % 10 = 4 := 
by 
  sorry

end units_digit_of_x_l134_134102


namespace problem_l134_134064

noncomputable def f (x a : ℝ) : ℝ := (1/2) * x ^ 2 - x - a * Real.log (x - a)

def monotonicity_f (a : ℝ) : Prop :=
  if a = 0 then
    ∀ x : ℝ, 0 < x → (x < 1 → f x 0 < f (x + 1) 0) ∧ (x > 1 → f x 0 > f (x + 1) 0)
  else if a > 0 then
    ∀ x : ℝ, a < x → (x < a + 1 → f x a < f (x + 1) a) ∧ (x > a + 1 → f x a > f (x + 1) a)
  else if -1 < a ∧ a < 0 then
    ∀ x : ℝ, 0 < x → (x < a + 1 → f x a < f (x + 1) a) ∧ (x > a + 1 → f (x + 1) a > f x a)
  else if a = -1 then
    ∀ x : ℝ, -1 < x → f x (-1) < f (x + 1) (-1)
  else
    ∀ x : ℝ, a < x → (x < 0 → f (x + 1) a > f x a) ∧ (0 < x → f x a > f (x + 1) a)

noncomputable def g (x a : ℝ) : ℝ := f (x + a) a - a * (x + (1/2) * a - 1)

def extreme_points (x₁ x₂ a : ℝ) : Prop :=
  x₁ < x₂ ∧ ∀ x : ℝ, 0 < x → x < 1 → g x a = 0

theorem problem (a : ℝ) (x₁ x₂ : ℝ) (hx : extreme_points x₁ x₂ a) (h_dom : -1/4 < a ∧ a < 0) :
  0 < f x₁ a - f x₂ a ∧ f x₁ a - f x₂ a < 1/2 := sorry

end problem_l134_134064


namespace largest_multiple_of_6_neg_greater_than_neg_150_l134_134752

theorem largest_multiple_of_6_neg_greater_than_neg_150 : 
  ∃ m : ℤ, m % 6 = 0 ∧ -m > -150 ∧ m = 144 :=
by
  sorry

end largest_multiple_of_6_neg_greater_than_neg_150_l134_134752


namespace speed_of_faster_train_approx_l134_134709

noncomputable def speed_of_slower_train_kmph : ℝ := 40
noncomputable def speed_of_slower_train_mps : ℝ := speed_of_slower_train_kmph * 1000 / 3600
noncomputable def distance_train1 : ℝ := 250
noncomputable def distance_train2 : ℝ := 500
noncomputable def total_distance : ℝ := distance_train1 + distance_train2
noncomputable def crossing_time : ℝ := 26.99784017278618
noncomputable def relative_speed_train_crossing : ℝ := total_distance / crossing_time
noncomputable def speed_of_faster_train_mps : ℝ := relative_speed_train_crossing - speed_of_slower_train_mps
noncomputable def speed_of_faster_train_kmph : ℝ := speed_of_faster_train_mps * 3600 / 1000

theorem speed_of_faster_train_approx : abs (speed_of_faster_train_kmph - 60.0152) < 0.001 :=
by 
  sorry

end speed_of_faster_train_approx_l134_134709


namespace compute_A_3_2_l134_134457

namespace Ackermann

def A : ℕ → ℕ → ℕ
| 0, n     => n + 1
| m + 1, 0 => A m 1
| m + 1, n + 1 => A m (A (m + 1) n)

theorem compute_A_3_2 : A 3 2 = 12 :=
sorry

end Ackermann

end compute_A_3_2_l134_134457


namespace minimum_value_function_l134_134789

theorem minimum_value_function :
  ∀ x : ℝ, x ≥ 0 → (∃ y : ℝ, y = (3 * x^2 + 9 * x + 20) / (7 * (2 + x)) ∧
    (∀ z : ℝ, z ≥ 0 → (3 * z^2 + 9 * z + 20) / (7 * (2 + z)) ≥ y)) ∧
    (∃ x0 : ℝ, x0 = 0 ∧ y = (3 * x0^2 + 9 * x0 + 20) / (7 * (2 + x0)) ∧ y = 10 / 7) :=
by
  sorry

end minimum_value_function_l134_134789


namespace pocket_knife_value_l134_134130

noncomputable def value_of_pocket_knife (n : ℕ) : ℕ :=
  if h : n = 0 then 0 else
    let total_rubles := n * n
    let tens (x : ℕ) := x / 10
    let units (x : ℕ) := x % 10
    let e := units n
    let d := tens n
    let remaining := total_rubles - ((total_rubles / 10) * 10)
    if remaining = 6 then 4 else sorry

theorem pocket_knife_value (n : ℕ) : value_of_pocket_knife n = 2 := by
  sorry

end pocket_knife_value_l134_134130


namespace complement_intersection_l134_134042

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem complement_intersection (U M N : Set ℕ) (hU : U = {1, 2, 3, 4})
  (hM : M = {1, 2, 3}) (hN : N = {2, 3, 4}) : (U \ (M ∩ N)) = {1, 4} := 
by
  sorry

end complement_intersection_l134_134042


namespace license_plate_palindrome_probability_l134_134729

-- Definitions for the problem conditions
def count_letter_palindromes : ℕ := 26 * 26
def total_letter_combinations : ℕ := 26 ^ 4

def count_digit_palindromes : ℕ := 10 * 10
def total_digit_combinations : ℕ := 10 ^ 4

def prob_letter_palindrome : ℚ := count_letter_palindromes / total_letter_combinations
def prob_digit_palindrome : ℚ := count_digit_palindromes / total_digit_combinations
def prob_both_palindrome : ℚ := (count_letter_palindromes * count_digit_palindromes) / (total_letter_combinations * total_digit_combinations)

def prob_atleast_one_palindrome : ℚ :=
  prob_letter_palindrome + prob_digit_palindrome - prob_both_palindrome

def p_q_sum : ℕ := 775 + 67600

-- Statement of the problem to be proved
theorem license_plate_palindrome_probability :
  prob_atleast_one_palindrome = 775 / 67600 ∧ p_q_sum = 68375 :=
by { sorry }

end license_plate_palindrome_probability_l134_134729


namespace manny_had_3_pies_l134_134732

-- Definitions of the conditions
def number_of_classmates : ℕ := 24
def number_of_teachers : ℕ := 1
def slices_per_pie : ℕ := 10
def slices_left : ℕ := 4

-- Number of people including Manny
def number_of_people : ℕ := number_of_classmates + number_of_teachers + 1

-- Total number of slices eaten
def slices_eaten : ℕ := number_of_people

-- Total number of slices initially
def total_slices : ℕ := slices_eaten + slices_left

-- Number of pies Manny had
def number_of_pies : ℕ := (total_slices / slices_per_pie) + 1

-- Theorem statement
theorem manny_had_3_pies : number_of_pies = 3 := by
  sorry

end manny_had_3_pies_l134_134732


namespace total_cost_is_53_l134_134356

-- Defining the costs and quantities as constants
def sandwich_cost : ℕ := 4
def soda_cost : ℕ := 3
def num_sandwiches : ℕ := 7
def num_sodas : ℕ := 10
def discount : ℕ := 5

-- Get the cost of sandwiches purchased
def cost_of_sandwiches : ℕ := num_sandwiches * sandwich_cost

-- Get the cost of sodas purchased
def cost_of_sodas : ℕ := num_sodas * soda_cost

-- Calculate the total cost before discount
def total_cost_before_discount : ℕ := cost_of_sandwiches + cost_of_sodas

-- Calculate the total cost after discount
def total_cost_after_discount : ℕ := total_cost_before_discount - discount

-- The theorem stating that the total cost is 53 dollars
theorem total_cost_is_53 : total_cost_after_discount = 53 :=
by
  sorry

end total_cost_is_53_l134_134356


namespace minimum_value_of_f_l134_134931

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem minimum_value_of_f :
  (∀ x : ℝ, x > 0 → f x ≥ -1 / Real.exp 1) ∧ (∃ x : ℝ, x > 0 ∧ f x = -1 / Real.exp 1) :=
by
  sorry

end minimum_value_of_f_l134_134931


namespace television_hours_watched_l134_134786

theorem television_hours_watched (minutes_per_day : ℕ) (days_per_week : ℕ) (weeks : ℕ)
  (h1 : minutes_per_day = 45) (h2 : days_per_week = 4) (h3 : weeks = 2):
  (minutes_per_day * days_per_week / 60) * weeks = 6 :=
by
  sorry

end television_hours_watched_l134_134786


namespace solve_for_x_l134_134411

-- Lean 4 statement for the problem
theorem solve_for_x (x : ℝ) (h : (x + 1)^3 = -27) : x = -4 := by
  sorry

end solve_for_x_l134_134411


namespace number_of_people_l134_134486

theorem number_of_people (clinks : ℕ) (h : clinks = 45) : ∃ x : ℕ, x * (x - 1) / 2 = clinks ∧ x = 10 :=
by
  sorry

end number_of_people_l134_134486


namespace parallel_lines_l134_134846

theorem parallel_lines :
  (∃ m: ℚ, (∀ x y: ℚ, (4 * y - 3 * x = 16 → y = m * x + (16 / 4)) ∧
                      (-3 * x - 4 * y = 15 → y = -m * x - (15 / 4)) ∧
                      (4 * y + 3 * x = 16 → y = -m * x + (16 / 4)) ∧
                      (3 * y + 4 * x = 15) → False)) :=
sorry

end parallel_lines_l134_134846


namespace snow_white_last_trip_l134_134273

noncomputable def dwarfs : List String := ["Happy", "Grumpy", "Dopey", "Bashful", "Sleepy", "Doc", "Sneezy"]

theorem snow_white_last_trip : ∃ dwarfs_in_last_trip : List String, 
  dwarfs_in_last_trip = ["Grumpy", "Bashful", "Doc"] :=
by
  sorry

end snow_white_last_trip_l134_134273


namespace exists_positive_integers_abcd_l134_134743

theorem exists_positive_integers_abcd (m : ℤ) : ∃ (a b c d : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ (a * b - c * d = m) := by
  sorry

end exists_positive_integers_abcd_l134_134743


namespace hitting_first_third_fifth_probability_hitting_exactly_three_out_of_five_probability_l134_134731

noncomputable def probability_hitting_first_third_fifth (P : ℚ) : ℚ :=
  P * (1 - P) * P * (1 - P) * P

noncomputable def binomial_coefficient (n k : ℕ) : ℕ := 
  Nat.choose n k

noncomputable def probability_hitting_exactly_three_out_of_five (P : ℚ) : ℚ :=
  binomial_coefficient 5 3 * P^3 * (1 - P)^2

theorem hitting_first_third_fifth_probability :
  probability_hitting_first_third_fifth (3/5) = 108/3125 := by
  sorry

theorem hitting_exactly_three_out_of_five_probability :
  probability_hitting_exactly_three_out_of_five (3/5) = 216/625 := by
  sorry

end hitting_first_third_fifth_probability_hitting_exactly_three_out_of_five_probability_l134_134731


namespace ratio_boys_to_girls_l134_134759

theorem ratio_boys_to_girls
  (b g : ℕ) 
  (h1 : b = g + 6) 
  (h2 : b + g = 36) : b / g = 7 / 5 :=
sorry

end ratio_boys_to_girls_l134_134759


namespace sum_of_squares_of_roots_l134_134783

theorem sum_of_squares_of_roots (s1 s2 : ℝ) (h1 : s1 * s2 = 4) (h2 : s1 + s2 = 16) : s1^2 + s2^2 = 248 :=
by
  sorry

end sum_of_squares_of_roots_l134_134783


namespace marge_funds_l134_134922

theorem marge_funds (initial_winnings : ℕ)
    (tax_fraction : ℕ)
    (loan_fraction : ℕ)
    (savings_amount : ℕ)
    (investment_fraction : ℕ)
    (tax_paid leftover_for_loans savings_after_loans final_leftover final_leftover_after_investment : ℕ) :
    initial_winnings = 12006 →
    tax_fraction = 2 →
    leftover_for_loans = initial_winnings / tax_fraction →
    loan_fraction = 3 →
    savings_after_loans = leftover_for_loans / loan_fraction →
    savings_amount = 1000 →
    final_leftover = leftover_for_loans - savings_after_loans - savings_amount →
    investment_fraction = 5 →
    final_leftover_after_investment = final_leftover - (savings_amount / investment_fraction) →
    final_leftover_after_investment = 2802 :=
by
  intros
  sorry

end marge_funds_l134_134922


namespace f_at_3_l134_134763

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * x - 1

-- The theorem to prove
theorem f_at_3 : f 3 = 5 := sorry

end f_at_3_l134_134763


namespace contractor_fine_per_day_l134_134314

theorem contractor_fine_per_day
    (total_days : ℕ) 
    (work_days_fine_amt : ℕ) 
    (total_amt : ℕ) 
    (absent_days : ℕ) 
    (worked_days : ℕ := total_days - absent_days)
    (earned_amt : ℕ := worked_days * work_days_fine_amt)
    (fine_per_day : ℚ)
    (total_fine : ℚ := absent_days * fine_per_day) : 
    (earned_amt - total_fine = total_amt) → 
    fine_per_day = 7.5 :=
by
  intros h
  -- proof here is omitted
  sorry

end contractor_fine_per_day_l134_134314


namespace gcd_of_n13_minus_n_l134_134834

theorem gcd_of_n13_minus_n : 
  ∀ n : ℤ, n ≠ 0 → 2730 ∣ (n ^ 13 - n) :=
by sorry

end gcd_of_n13_minus_n_l134_134834


namespace total_rainbow_nerds_l134_134082

-- Definitions based on the conditions
def num_purple_candies : ℕ := 10
def num_yellow_candies : ℕ := num_purple_candies + 4
def num_green_candies : ℕ := num_yellow_candies - 2

-- The statement to be proved
theorem total_rainbow_nerds : num_purple_candies + num_yellow_candies + num_green_candies = 36 := by
  -- Using the provided definitions to automatically infer
  sorry

end total_rainbow_nerds_l134_134082


namespace no_perfect_square_integers_l134_134051

open Nat

def Q (x : ℤ) : ℤ := x^4 + 4 * x^3 + 10 * x^2 + 4 * x + 29

theorem no_perfect_square_integers : ∀ x : ℤ, ¬∃ a : ℤ, Q x = a^2 :=
by
  sorry

end no_perfect_square_integers_l134_134051


namespace repetend_of_5_over_13_l134_134953

theorem repetend_of_5_over_13 : (∃ r : ℕ, r = 384615) :=
by
  let d := 13
  let n := 5
  let r := 384615
  -- Definitions to use:
  -- d is denominator 13
  -- n is numerator 5
  -- r is the repetend 384615
  sorry

end repetend_of_5_over_13_l134_134953


namespace inverse_proportion_l134_134497

theorem inverse_proportion (x y : ℝ) (k : ℝ) (h1 : x^2 * y^4 = k)
  (h2 : 6^2 * 2^4 = k) (hy : y = 4) : x^2 = 2.25 :=
by
  sorry

end inverse_proportion_l134_134497


namespace three_f_l134_134970

noncomputable def f (x : ℝ) : ℝ := sorry

theorem three_f (x : ℝ) (hx : 0 < x) (h : ∀ y > 0, f (3 * y) = 5 / (3 + y)) :
  3 * f x = 45 / (9 + x) :=
by
  sorry

end three_f_l134_134970


namespace ap_minus_aq_eq_8_l134_134703

theorem ap_minus_aq_eq_8 (S_n : ℕ → ℤ) (a_n : ℕ → ℤ) (p q : ℕ) 
  (h1 : ∀ n, S_n n = n^2 - 5 * n) 
  (h2 : ∀ n ≥ 2, a_n n = S_n n - S_n (n - 1)) 
  (h3 : p - q = 4) :
  a_n p - a_n q = 8 := sorry

end ap_minus_aq_eq_8_l134_134703


namespace quadratic_expression_value_l134_134868

-- Given conditions
variables (a : ℝ) (h : 2 * a^2 + 3 * a - 2022 = 0)

-- Prove the main statement
theorem quadratic_expression_value :
  2 - 6 * a - 4 * a^2 = -4042 :=
sorry

end quadratic_expression_value_l134_134868


namespace car_trip_proof_l134_134877

def initial_oil_quantity (y : ℕ → ℕ) : Prop :=
  y 0 = 50

def consumption_rate (y : ℕ → ℕ) : Prop :=
  ∀ t, y t = y (t - 1) - 5

def relationship_between_y_and_t (y : ℕ → ℕ) : Prop :=
  ∀ t, y t = 50 - 5 * t

def oil_left_after_8_hours (y : ℕ → ℕ) : Prop :=
  y 8 = 10

theorem car_trip_proof (y : ℕ → ℕ) :
  initial_oil_quantity y ∧ consumption_rate y ∧ relationship_between_y_and_t y ∧ oil_left_after_8_hours y :=
by
  -- the proof goes here
  sorry

end car_trip_proof_l134_134877


namespace sin_C_l134_134276

variable {A B C : ℝ}

theorem sin_C (hA : A = 90) (hcosB : Real.cos B = 3/5) : Real.sin (90 - B) = 3/5 :=
by
  sorry

end sin_C_l134_134276


namespace arithmetic_sequence_a5_zero_l134_134708

variable {a : ℕ → ℤ}
variable {d : ℤ}

theorem arithmetic_sequence_a5_zero 
  (h1 : ∀ n, a (n + 1) = a n + d) 
  (h2 : d ≠ 0)
  (h3 : a 3 + a 9 = a 10 - a 8) : 
  a 5 = 0 := sorry

end arithmetic_sequence_a5_zero_l134_134708


namespace sufficient_but_not_necessary_condition_l134_134646

theorem sufficient_but_not_necessary_condition (m : ℝ) :
  (∃ x : ℝ, x^2 + 2 * x + m = 0) ↔ m < 1 :=
by
  sorry

end sufficient_but_not_necessary_condition_l134_134646


namespace five_nat_numbers_product_1000_l134_134746

theorem five_nat_numbers_product_1000 :
  ∃ (a b c d e : ℕ), 
    a * b * c * d * e = 1000 ∧ 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
    c ≠ d ∧ c ≠ e ∧ 
    d ≠ e := 
by
  sorry

end five_nat_numbers_product_1000_l134_134746


namespace negation_equiv_l134_134675

-- Original proposition
def original_proposition (x : ℝ) : Prop := x > 0 ∧ x^2 - 5 * x + 6 > 0

-- Negated proposition
def negated_proposition : Prop := ∀ x : ℝ, x > 0 → x^2 - 5 * x + 6 ≤ 0

-- Statement of the theorem to prove
theorem negation_equiv : ¬(∃ x : ℝ, original_proposition x) ↔ negated_proposition :=
by sorry

end negation_equiv_l134_134675


namespace linear_eq_rewrite_l134_134774

theorem linear_eq_rewrite (x y : ℝ) (h : 2 * x - y = 3) : y = 2 * x - 3 :=
by
  sorry

end linear_eq_rewrite_l134_134774


namespace at_least_one_triangle_l134_134169

theorem at_least_one_triangle {n : ℕ} (h1 : n ≥ 2) (points : Finset ℕ) (segments : Finset (ℕ × ℕ)) : 
(points.card = 2 * n) ∧ (segments.card = n^2 + 1) → 
∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ ((a, b) ∈ segments ∨ (b, a) ∈ segments) ∧ ((b, c) ∈ segments ∨ (c, b) ∈ segments) ∧ ((c, a) ∈ segments ∨ (a, c) ∈ segments) := 
by 
  sorry

end at_least_one_triangle_l134_134169


namespace solve_system_of_equations_l134_134093

theorem solve_system_of_equations (x y : ℝ) (h1 : 2 * x + 3 * y = 7) (h2 : 4 * x - 3 * y = 5) : x = 2 ∧ y = 1 :=
by
    -- The proof is not required, so we put a sorry here.
    sorry

end solve_system_of_equations_l134_134093


namespace perpendicular_lines_a_equals_one_l134_134213

theorem perpendicular_lines_a_equals_one
  (a : ℝ)
  (l1 : ∀ x y : ℝ, x - 2 * y + 1 = 0)
  (l2 : ∀ x y : ℝ, 2 * x + a * y - 1 = 0)
  (perpendicular : ∀ x y : ℝ, (x - 2 * y + 1 = 0) ∧ (2 * x + a * y - 1 = 0) → 
    (-(1 / -2) * -(2 / a)) = -1) :
  a = 1 :=
by
  sorry

end perpendicular_lines_a_equals_one_l134_134213


namespace c_zero_roots_arithmetic_seq_range_f1_l134_134361

section problem

variable (b : ℝ)
def f (x : ℝ) := x^3 + 3 * b * x^2 + 0 * x + (-2 * b^3)
def f' (x : ℝ) := 3 * x^2 + 6 * b * x + 0

-- Proving c = 0 if f(x) is increasing on (-∞, 0) and decreasing on (0, 2)
theorem c_zero (h_inc : ∀ x < 0, f' b x > 0) (h_dec : ∀ x > 0, f' b x < 0) : 0 = 0 := sorry

-- Proving f(x) = 0 has two other distinct real roots x1 and x2 different from -b, forming an arithmetic sequence
theorem roots_arithmetic_seq (hb : ∀ x : ℝ, f b x = 0 → (x = -b ∨ -b ≠ x)) : 
    ∃ (x1 x2 : ℝ), x1 ≠ -b ∧ x2 ≠ -b ∧ x1 + x2 = -2 * b := sorry

-- Proving the range of values for f(1) when the maximum value of f(x) is less than 16
theorem range_f1 (h_max : ∀ x : ℝ, f b x < 16 ) : 0 ≤ f b 1 ∧ f b 1 < 11 := sorry

end problem

end c_zero_roots_arithmetic_seq_range_f1_l134_134361


namespace ceil_eq_intervals_l134_134174

theorem ceil_eq_intervals (x : ℝ) :
  (⌈⌈ 3 * x ⌉ + 1 / 2⌉ = ⌈ x - 2 ⌉) ↔ (-1 : ℝ) ≤ x ∧ x < -2 / 3 := 
by
  sorry

end ceil_eq_intervals_l134_134174


namespace expression_positive_l134_134010

theorem expression_positive (a b : ℝ) (h : a ≠ 0 ∨ b ≠ 0) : 5 * a ^ 2 - 6 * a * b + 5 * b ^ 2 > 0 :=
by
  sorry

end expression_positive_l134_134010


namespace largest_k_l134_134028

-- Define the system of equations and conditions
def system_valid (x y k : ℝ) : Prop := 
  2 * x + y = k ∧ 
  3 * x + y = 3 ∧ 
  x - 2 * y ≥ 1

-- Define the proof problem as a theorem in Lean
theorem largest_k (x y : ℝ) :
  ∀ k : ℝ, system_valid x y k → k ≤ 2 := 
sorry

end largest_k_l134_134028


namespace meridian_students_l134_134925

theorem meridian_students
  (eighth_to_seventh_ratio : Nat → Nat → Prop)
  (seventh_to_sixth_ratio : Nat → Nat → Prop)
  (r1 : ∀ a b, eighth_to_seventh_ratio a b ↔ 7 * b = 4 * a)
  (r2 : ∀ b c, seventh_to_sixth_ratio b c ↔ 10 * c = 9 * b) :
  ∃ a b c, eighth_to_seventh_ratio a b ∧ seventh_to_sixth_ratio b c ∧ a + b + c = 73 :=
by
  sorry

end meridian_students_l134_134925


namespace prob_lamp_first_factory_standard_prob_lamp_standard_l134_134958

noncomputable def P_B1 : ℝ := 0.35
noncomputable def P_B2 : ℝ := 0.50
noncomputable def P_B3 : ℝ := 0.15

noncomputable def P_B1_A : ℝ := 0.70
noncomputable def P_B2_A : ℝ := 0.80
noncomputable def P_B3_A : ℝ := 0.90

-- Question A
theorem prob_lamp_first_factory_standard : P_B1 * P_B1_A = 0.245 :=
by 
  sorry

-- Question B
theorem prob_lamp_standard : (P_B1 * P_B1_A) + (P_B2 * P_B2_A) + (P_B3 * P_B3_A) = 0.78 :=
by 
  sorry

end prob_lamp_first_factory_standard_prob_lamp_standard_l134_134958


namespace angles_supplementary_l134_134488

theorem angles_supplementary (A B : ℕ) (h1 : A > 0) (h2 : B > 0) (h3 : A + B = 180) (h4 : ∃ k : ℕ, k ≥ 1 ∧ A = k * B) : ∃ S : Finset ℕ, S.card = 17 ∧ (∀ a ∈ S, ∃ k : ℕ, k * (180 / (k + 1)) = a ∧ A = a) :=
by
  sorry

end angles_supplementary_l134_134488


namespace factorization_of_difference_of_squares_l134_134839

theorem factorization_of_difference_of_squares (m n : ℝ) : m^2 - n^2 = (m + n) * (m - n) := 
by sorry

end factorization_of_difference_of_squares_l134_134839


namespace constant_term_in_expansion_l134_134560

theorem constant_term_in_expansion : 
  let a := (x : ℝ)
  let b := - (2 / Real.sqrt x)
  let n := 6
  let general_term (r : Nat) : ℝ := Nat.choose n r * a * (b ^ (n - r))
  (∀ x : ℝ, ∃ (r : Nat), r = 4 ∧ (1 - (n - r) / 2 = 0) →
  general_term 4 = 60) :=
by
  sorry

end constant_term_in_expansion_l134_134560


namespace algebraic_expression_l134_134472

def a (x : ℕ) := 2005 * x + 2009
def b (x : ℕ) := 2005 * x + 2010
def c (x : ℕ) := 2005 * x + 2011

theorem algebraic_expression (x : ℕ) : 
  a x ^ 2 + b x ^ 2 + c x ^ 2 - a x * b x - b x * c x - c x * a x = 3 :=
by
  sorry

end algebraic_expression_l134_134472


namespace solve_equations_l134_134412

theorem solve_equations (x y : ℝ) (h1 : (x + y) / x = y / (x + y)) (h2 : x = 2 * y) :
  x = 0 ∧ y = 0 :=
by
  sorry

end solve_equations_l134_134412


namespace perfect_square_trinomial_l134_134242

theorem perfect_square_trinomial (k : ℤ) : (∀ x : ℤ, x^2 + 2 * (k + 1) * x + 16 = (x + (k + 1))^2) → (k = 3 ∨ k = -5) :=
by
  sorry

end perfect_square_trinomial_l134_134242


namespace billiard_expected_reflections_l134_134988

noncomputable def expected_reflections : ℝ :=
  (2 / Real.pi) * (3 * Real.arccos (1 / 4) - Real.arcsin (3 / 4) + Real.arccos (3 / 4))

theorem billiard_expected_reflections :
  expected_reflections = (2 / Real.pi) * (3 * Real.arccos (1 / 4) - Real.arcsin (3 / 4) + Real.arccos (3 / 4)) :=
by
  sorry

end billiard_expected_reflections_l134_134988


namespace isosceles_triangle_perimeter_l134_134603

-- Define the given conditions
def is_isosceles_triangle (a b c : ℕ) : Prop :=
  (a = b ∨ b = c ∨ c = a)

def valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Theorem based on the problem statement and conditions
theorem isosceles_triangle_perimeter (a b : ℕ) (P : is_isosceles_triangle a b 5) (Q : is_isosceles_triangle b a 10) :
  valid_triangle a b 5 → valid_triangle b a 10 → a + b + 5 = 25 :=
by sorry

end isosceles_triangle_perimeter_l134_134603


namespace problem_l134_134979

open Real

noncomputable def f (x : ℝ) : ℝ := log x / log 2

theorem problem (f : ℝ → ℝ) (h : ∀ (x y : ℝ), f (x * y) = f x + f y) : 
  (∀ x : ℝ, f x = log x / log 2) :=
sorry

end problem_l134_134979


namespace probability_of_draw_l134_134509

-- Define the probabilities as given conditions
def P_A : ℝ := 0.4
def P_A_not_losing : ℝ := 0.9

-- Define the probability of drawing
def P_draw : ℝ :=
  P_A_not_losing - P_A

-- State the theorem to be proved
theorem probability_of_draw : P_draw = 0.5 := by
  sorry

end probability_of_draw_l134_134509


namespace negation_of_proposition_l134_134198

theorem negation_of_proposition (m : ℝ) : 
  (¬ ∀ x : ℝ, x^2 + 2*x + m ≤ 0) ↔ (∃ x : ℝ, x^2 + 2*x + m > 0) :=
sorry

end negation_of_proposition_l134_134198


namespace find_m_l134_134167

theorem find_m (m : ℕ) (h_pos : 0 < m) 
  (h_intersection : ∃ (x y : ℤ), 13 * x + 11 * y = 700 ∧ y = m * x - 1) : 
  m = 6 :=
sorry

end find_m_l134_134167


namespace total_work_completed_in_days_l134_134349

theorem total_work_completed_in_days (T : ℕ) :
  (amit_days amit_worked ananthu_days remaining_work : ℕ) → 
  amit_days = 3 → amit_worked = amit_days * (1 / 15) → 
  ananthu_days = 36 → 
  remaining_work = 1 - amit_worked  →
  (ananthu_days * (1 / 45)) = remaining_work →
  T = amit_days + ananthu_days →
  T = 39 := 
sorry

end total_work_completed_in_days_l134_134349


namespace four_digit_positive_integers_count_l134_134995

def first_two_digit_choices : Finset ℕ := {2, 3, 6}
def last_two_digit_choices : Finset ℕ := {3, 7, 9}

theorem four_digit_positive_integers_count :
  (first_two_digit_choices.card * first_two_digit_choices.card) *
  (last_two_digit_choices.card * (last_two_digit_choices.card - 1)) = 54 := by
sorry

end four_digit_positive_integers_count_l134_134995


namespace robot_material_handling_per_hour_min_num_type_A_robots_l134_134812

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

end robot_material_handling_per_hour_min_num_type_A_robots_l134_134812


namespace num_people_present_l134_134640

-- Given conditions
def associatePencilCount (A : ℕ) : ℕ := 2 * A
def assistantPencilCount (B : ℕ) : ℕ := B
def associateChartCount (A : ℕ) : ℕ := A
def assistantChartCount (B : ℕ) : ℕ := 2 * B

def totalPencils (A B : ℕ) : ℕ := associatePencilCount A + assistantPencilCount B
def totalCharts (A B : ℕ) : ℕ := associateChartCount A + assistantChartCount B

-- Prove the total number of people present
theorem num_people_present (A B : ℕ) (h1 : totalPencils A B = 11) (h2 : totalCharts A B = 16) : A + B = 9 :=
by
  sorry

end num_people_present_l134_134640


namespace gcd_79625_51575_l134_134106

theorem gcd_79625_51575 : Nat.gcd 79625 51575 = 25 :=
by
  sorry

end gcd_79625_51575_l134_134106


namespace count_squares_3x3_grid_count_squares_5x5_grid_l134_134286

/-- Define a mathematical problem: 
  Prove that the number of squares with all four vertices on the dots in a 3x3 grid is 4.
  Prove that the number of squares with all four vertices on the dots in a 5x5 grid is 50.
-/

def num_squares_3x3 : Nat := 4
def num_squares_5x5 : Nat := 50

theorem count_squares_3x3_grid : 
  ∀ (grid_size : Nat), grid_size = 3 → (∃ (dots_on_square : Bool), ∀ (distance_between_dots : Real), (dots_on_square = true → num_squares_3x3 = 4)) := 
by 
  intros grid_size h1
  exists true
  intros distance_between_dots 
  sorry

theorem count_squares_5x5_grid : 
  ∀ (grid_size : Nat), grid_size = 5 → (∃ (dots_on_square : Bool), ∀ (distance_between_dots : Real), (dots_on_square = true → num_squares_5x5 = 50)) :=
by 
  intros grid_size h1
  exists true
  intros distance_between_dots 
  sorry

end count_squares_3x3_grid_count_squares_5x5_grid_l134_134286


namespace balloon_difference_l134_134223

theorem balloon_difference 
  (your_balloons : ℕ := 7) 
  (friend_balloons : ℕ := 5) : 
  your_balloons - friend_balloons = 2 := 
by 
  sorry

end balloon_difference_l134_134223


namespace overtime_pay_correct_l134_134243

theorem overtime_pay_correct
  (overlap_slow : ℝ := 69) -- Slow clock minute-hand overlap in minutes
  (overlap_normal : ℝ := 12 * 60 / 11) -- Normal clock minute-hand overlap in minutes
  (hours_worked : ℝ := 8) -- The normal working hours a worker believes working
  (hourly_wage : ℝ := 4) -- The normal hourly wage
  (overtime_rate : ℝ := 1.5) -- Overtime pay rate
  (expected_overtime_pay : ℝ := 2.60) -- The expected overtime pay
  
  : hours_worked * (overlap_slow / overlap_normal) * hourly_wage * (overtime_rate - 1) = expected_overtime_pay :=
by
  sorry

end overtime_pay_correct_l134_134243


namespace ac_bc_ratios_l134_134969

theorem ac_bc_ratios (A B C : ℝ) (m n : ℕ) (h : AC / BC = m / n) : 
  if m ≠ n then
    ((AC / AB = m / (m+n) ∧ BC / AB = n / (m+n)) ∨ 
     (AC / AB = m / (n-m) ∧ BC / AB = n / (n-m)))
  else 
    (AC / AB = 1 / 2 ∧ BC / AB = 1 / 2) := sorry

end ac_bc_ratios_l134_134969


namespace expression_evaluation_l134_134992

theorem expression_evaluation : (50 - (2050 - 250)) + (2050 - (250 - 50)) = 100 := by
  sorry

end expression_evaluation_l134_134992


namespace factorize_square_difference_l134_134766

open Real

theorem factorize_square_difference (m n : ℝ) :
  m ^ 2 - 4 * n ^ 2 = (m + 2 * n) * (m - 2 * n) :=
sorry

end factorize_square_difference_l134_134766


namespace simplify_expression_l134_134981

theorem simplify_expression :
  (Real.sin (Real.pi / 6) + (1 / 2) - 2007^0 + abs (-2) = 2) :=
by
  sorry

end simplify_expression_l134_134981


namespace find_base_of_triangle_l134_134690

-- Given data
def perimeter : ℝ := 20 -- The perimeter of the triangle
def tangent_segment : ℝ := 2.4 -- The segment of the tangent to the inscribed circle contained between the sides

-- Define the problem and expected result
theorem find_base_of_triangle (a b c : ℝ) (P : a + b + c = perimeter)
  (tangent_parallel_base : ℝ := tangent_segment):
  a = 4 ∨ a = 6 :=
sorry

end find_base_of_triangle_l134_134690


namespace length_of_pencils_l134_134310

theorem length_of_pencils (length_pencil1 : ℕ) (length_pencil2 : ℕ)
  (h1 : length_pencil1 = 12) (h2 : length_pencil2 = 12) : length_pencil1 + length_pencil2 = 24 :=
by
  sorry

end length_of_pencils_l134_134310


namespace determine_avery_height_l134_134123

-- Define Meghan's height
def meghan_height : ℕ := 188

-- Define range of players' heights
def height_range : ℕ := 33

-- Define the predicate to determine Avery's height
def avery_height : ℕ := meghan_height - height_range

-- The theorem we need to prove
theorem determine_avery_height : avery_height = 155 := by
  sorry

end determine_avery_height_l134_134123


namespace spiders_hired_l134_134745

theorem spiders_hired (total_workers beavers : ℕ) (h_total : total_workers = 862) (h_beavers : beavers = 318) : (total_workers - beavers) = 544 := by
  sorry

end spiders_hired_l134_134745


namespace product_first_8_terms_l134_134795

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

end product_first_8_terms_l134_134795


namespace incorrect_option_l134_134897

noncomputable def f : ℝ → ℝ := sorry
def is_odd (g : ℝ → ℝ) := ∀ x, g (-(2 * x + 1)) = -g (2 * x + 1)
def is_even (g : ℝ → ℝ) := ∀ x, g (x + 2) = g (-x + 2)

theorem incorrect_option (h₁ : is_odd f) (h₂ : is_even f) (h₃ : ∀ x, 2 ≤ x ∧ x ≤ 3 → f x = 3 - x) :
  ¬ (∀ x, f x = f (-x - 2)) :=
by
  sorry

end incorrect_option_l134_134897


namespace math_problem_l134_134504

def f (x : ℝ) : ℝ := sorry

theorem math_problem (n s : ℕ)
  (h1 : f 1 = 2)
  (h2 : ∀ x y : ℝ, f (x^2 + y^2) = (x + y) * (f x - f y))
  (hn : n = 1)
  (hs : s = 6) :
  n * s = 6 := by
  sorry

end math_problem_l134_134504


namespace problem_l134_134638

section Problem
variables {n : ℕ } {k : ℕ} 

theorem problem (n : ℕ) (k : ℕ) (a : ℕ) (n_i : Fin k → ℕ) (h1 : ∀ i j, i ≠ j → Nat.gcd (n_i i) (n_i j) = 1) 
  (h2 : ∀ i, a^n_i i % n_i i = 1) (h3 : ∀ i, ¬(n_i i ∣ a - 1)) :
  ∃ (x : ℕ), x > 1 ∧ a^x % x = 1 ∧ x ≥ 2^(k + 1) - 2 := by
  sorry
end Problem

end problem_l134_134638


namespace students_enrolled_for_german_l134_134211

-- Defining the total number of students
def class_size : Nat := 40

-- Defining the number of students enrolled for both English and German
def enrolled_both : Nat := 12

-- Defining the number of students enrolled for only English and not German
def enrolled_only_english : Nat := 18

-- Using the conditions to define the number of students who enrolled for German
theorem students_enrolled_for_german (G G_only : Nat) 
  (h_class_size : G_only + enrolled_only_english + enrolled_both = class_size) 
  (h_G : G = G_only + enrolled_both) : 
  G = 22 := 
by
  -- placeholder for proof
  sorry

end students_enrolled_for_german_l134_134211


namespace like_terms_product_l134_134233

theorem like_terms_product :
  ∀ (m n : ℕ),
    (-x^3 * y^n) = (3 * x^m * y^2) → (m = 3 ∧ n = 2) → m * n = 6 :=
by
  intros m n h1 h2
  sorry

end like_terms_product_l134_134233


namespace value_of_x_l134_134476

theorem value_of_x (x : ℝ) (h : 0.75 * 600 = 0.50 * x) : x = 900 :=
by
  sorry

end value_of_x_l134_134476


namespace Rick_is_three_times_Sean_l134_134637

-- Definitions and assumptions
def Fritz_money : ℕ := 40
def Sean_money : ℕ := (Fritz_money / 2) + 4
def total_money : ℕ := 96

-- Rick's money can be derived from total_money - Sean_money
def Rick_money : ℕ := total_money - Sean_money

-- Claim to be proven
theorem Rick_is_three_times_Sean : Rick_money = 3 * Sean_money := 
by 
  -- Proof steps would go here
  sorry

end Rick_is_three_times_Sean_l134_134637


namespace find_a1_l134_134088

theorem find_a1 (a : ℕ → ℝ) (h1 : ∀ n : ℕ, n > 0 → a (n + 1) = 1 / (1 - a n)) (h2 : a 8 = 2)
: a 1 = 1 / 2 :=
sorry

end find_a1_l134_134088


namespace kara_uses_28_cups_of_sugar_l134_134814

theorem kara_uses_28_cups_of_sugar (S W : ℕ) (h1 : S + W = 84) (h2 : S * 2 = W) : S = 28 :=
by sorry

end kara_uses_28_cups_of_sugar_l134_134814


namespace angle_terminal_side_equiv_l134_134630

theorem angle_terminal_side_equiv (α : ℝ) (k : ℤ) :
  (∃ k : ℤ, α = 30 + k * 360) ↔ (∃ β : ℝ, β = 30 ∧ α % 360 = β % 360) :=
by
  sorry

end angle_terminal_side_equiv_l134_134630


namespace number_of_digits_in_N_l134_134020

noncomputable def N : ℕ := 2^12 * 5^8

theorem number_of_digits_in_N : (Nat.digits 10 N).length = 10 := by
  sorry

end number_of_digits_in_N_l134_134020


namespace smallest_whole_number_larger_than_perimeter_l134_134702

-- Define the sides of the triangle
def side1 : ℕ := 7
def side2 : ℕ := 23

-- State the conditions using the triangle inequality theorem
def triangle_inequality_satisfied (s : ℕ) : Prop :=
  (side1 + side2 > s) ∧ (side1 + s > side2) ∧ (side2 + s > side1)

-- The proof statement
theorem smallest_whole_number_larger_than_perimeter
  (s : ℕ) (h : triangle_inequality_satisfied s) : 
  ∃ n : ℕ, n = 60 ∧ ∀ p : ℕ, (p > side1 + side2 + s) → (p ≥ n) :=
sorry

end smallest_whole_number_larger_than_perimeter_l134_134702


namespace range_of_a_l134_134946

theorem range_of_a (x y : ℝ) (a : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y + 4 = 2 * x * y) :
  x^2 + 2 * x * y + y^2 - a * x - a * y + 1 ≥ 0 ↔ a ≤ 17 / 4 := 
sorry

end range_of_a_l134_134946


namespace inequality_proof_l134_134686

theorem inequality_proof (x y z : ℝ) (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z ≥ 0) : 
  (x*y^2/z + y*z^2/x + z*x^2/y) ≥ (x^2 + y^2 + z^2) := by
  sorry

end inequality_proof_l134_134686


namespace consecutive_integers_product_no_3n_plus_1_product_consecutive_no_n_cubed_plus_5n_plus_4_product_consecutive_no_permutation_23456780_product_consecutive_l134_134860

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

end consecutive_integers_product_no_3n_plus_1_product_consecutive_no_n_cubed_plus_5n_plus_4_product_consecutive_no_permutation_23456780_product_consecutive_l134_134860


namespace John_avg_speed_l134_134806

theorem John_avg_speed :
  ∀ (initial final : ℕ) (time : ℕ),
    initial = 27372 →
    final = 27472 →
    time = 4 →
    ((final - initial) / time) = 25 :=
by
  intros initial final time h_initial h_final h_time
  sorry

end John_avg_speed_l134_134806


namespace geometric_sum_l134_134258

open BigOperators

noncomputable def geom_sequence (a q : ℚ) (n : ℕ) : ℚ := a * q ^ n

noncomputable def sum_geom_sequence (a q : ℚ) (n : ℕ) : ℚ := 
  if q = 1 then a * n
  else a * (1 - q ^ (n + 1)) / (1 - q)

theorem geometric_sum (a q : ℚ) (h_a : a = 1) (h_S3 : sum_geom_sequence a q 2 = 3 / 4) :
  sum_geom_sequence a q 3 = 5 / 8 :=
sorry

end geometric_sum_l134_134258


namespace max_product_of_slopes_l134_134444

theorem max_product_of_slopes 
  (m₁ m₂ : ℝ)
  (h₁ : m₂ = 3 * m₁)
  (h₂ : abs ((m₂ - m₁) / (1 + m₁ * m₂)) = Real.sqrt 3) :
  m₁ * m₂ ≤ 2 :=
sorry

end max_product_of_slopes_l134_134444


namespace number_in_2019th_field_l134_134228

theorem number_in_2019th_field (f : ℕ → ℕ) (h1 : ∀ n, 0 < f n) (h2 : ∀ n, f n * f (n+1) * f (n+2) = 2018) :
  f 2018 = 1009 := sorry

end number_in_2019th_field_l134_134228


namespace marked_price_percentage_l134_134317

theorem marked_price_percentage
  (CP MP SP : ℝ)
  (h_profit : SP = 1.08 * CP)
  (h_discount : SP = 0.8307692307692308 * MP) :
  MP = CP * 1.3 :=
by sorry

end marked_price_percentage_l134_134317


namespace part1_part2_i_part2_ii_l134_134793

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

end part1_part2_i_part2_ii_l134_134793


namespace negation_of_proposition_l134_134371

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, 1 < x → (Real.log x / Real.log 2) + 4 * (Real.log 2 / Real.log x) > 4)) ↔
  (∃ x : ℝ, 1 < x ∧ (Real.log x / Real.log 2) + 4 * (Real.log 2 / Real.log x) ≤ 4) :=
sorry

end negation_of_proposition_l134_134371


namespace spadesuit_calculation_l134_134994

def spadesuit (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spadesuit_calculation : spadesuit 5 (spadesuit 3 2) = 0 :=
by
  sorry

end spadesuit_calculation_l134_134994


namespace solution_l134_134296

noncomputable def x1 : ℝ := sorry
noncomputable def x2 : ℝ := sorry
noncomputable def x3 : ℝ := sorry
noncomputable def x4 : ℝ := sorry
noncomputable def x5 : ℝ := sorry
noncomputable def x6 : ℝ := sorry
noncomputable def x7 : ℝ := sorry
noncomputable def x8 : ℝ := sorry

axiom cond1 : x1 + 4 * x2 + 9 * x3 + 16 * x4 + 25 * x5 + 36 * x6 + 49 * x7 + 64 * x8 = 10
axiom cond2 : 4 * x1 + 9 * x2 + 16 * x3 + 25 * x4 + 36 * x5 + 49 * x6 + 64 * x7 + 81 * x8 = 40
axiom cond3 : 9 * x1 + 16 * x2 + 25 * x3 + 36 * x4 + 49 * x5 + 64 * x6 + 81 * x7 + 100 * x8 = 170

theorem solution : 16 * x1 + 25 * x2 + 36 * x3 + 49 * x4 + 64 * x5 + 81 * x6 + 100 * x7 + 121 * x8 = 400 := 
by
  sorry

end solution_l134_134296


namespace Mike_watches_TV_every_day_l134_134446

theorem Mike_watches_TV_every_day :
  (∃ T : ℝ, 
  (3 * (T / 2) + 7 * T = 34) 
  → T = 4) :=
by
  let T := 4
  sorry

end Mike_watches_TV_every_day_l134_134446


namespace divisor_unique_l134_134779

theorem divisor_unique {b : ℕ} (h1 : 826 % b = 7) (h2 : 4373 % b = 8) : b = 9 :=
sorry

end divisor_unique_l134_134779


namespace voltage_relationship_l134_134990

variables (x y z : ℝ) -- Coordinates representing positions on the lines
variables (I R U : ℝ) -- Representing current, resistance, and voltage respectively

-- Conditions translated into Lean
def I_def := I = 10^x
def R_def := R = 10^(-2 * y)
def U_def := U = 10^(-z)
def coord_relation := x + z = 2 * y

-- The final theorem to prove V = I * R under given conditions
theorem voltage_relationship : I = 10^x → R = 10^(-2 * y) → U = 10^(-z) → (x + z = 2 * y) → U = I * R :=
by 
  intros hI hR hU hXYZ
  sorry

end voltage_relationship_l134_134990


namespace arithmetic_sequence_eighth_term_l134_134840

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

end arithmetic_sequence_eighth_term_l134_134840


namespace Jason_spent_correct_amount_l134_134813

namespace MusicStore

def costFlute : Real := 142.46
def costMusicStand : Real := 8.89
def costSongBook : Real := 7.00
def totalCost : Real := 158.35

theorem Jason_spent_correct_amount :
  costFlute + costMusicStand + costSongBook = totalCost :=
sorry

end MusicStore

end Jason_spent_correct_amount_l134_134813


namespace find_C_l134_134324

theorem find_C (A B C : ℕ) (hA : A = 509) (hAB : A = B + 197) (hCB : C = B - 125) : C = 187 := 
by 
  sorry

end find_C_l134_134324


namespace johns_salary_before_raise_l134_134785

variable (x : ℝ)

theorem johns_salary_before_raise (h : x + 0.3333 * x = 80) : x = 60 :=
by
  sorry

end johns_salary_before_raise_l134_134785


namespace candies_per_block_l134_134197

theorem candies_per_block (candies_per_house : ℕ) (houses_per_block : ℕ) (h1 : candies_per_house = 7) (h2 : houses_per_block = 5) :
  candies_per_house * houses_per_block = 35 :=
by 
  -- Placeholder for the formal proof
  sorry

end candies_per_block_l134_134197


namespace tod_north_distance_l134_134077

-- Given conditions as variables
def speed : ℕ := 25  -- speed in miles per hour
def time : ℕ := 6    -- time in hours
def west_distance : ℕ := 95  -- distance to the west in miles

-- Prove the distance to the north given conditions
theorem tod_north_distance : time * speed - west_distance = 55 := by
  sorry

end tod_north_distance_l134_134077


namespace point_A_in_fourth_quadrant_l134_134500

def Point := ℤ × ℤ

def is_in_fourth_quadrant (p : Point) : Prop :=
  p.1 > 0 ∧ p.2 < 0

def point_A : Point := (3, -2)
def point_B : Point := (2, 5)
def point_C : Point := (-1, -2)
def point_D : Point := (-2, 2)

theorem point_A_in_fourth_quadrant : is_in_fourth_quadrant point_A :=
  sorry

end point_A_in_fourth_quadrant_l134_134500


namespace find_intersection_pair_l134_134350

def cubic_function (x : ℝ) : ℝ := x^3 - 3*x + 2

def linear_function (x y : ℝ) : Prop := x + 4*y = 4

def intersection_points (x y : ℝ) : Prop := 
  linear_function x y ∧ y = cubic_function x

def sum_x_coord (points : List (ℝ × ℝ)) : ℝ :=
  points.map Prod.fst |>.sum

def sum_y_coord (points : List (ℝ × ℝ)) : ℝ :=
  points.map Prod.snd |>.sum

theorem find_intersection_pair (x1 x2 x3 y1 y2 y3 : ℝ) 
  (h1 : intersection_points x1 y1)
  (h2 : intersection_points x2 y2)
  (h3 : intersection_points x3 y3)
  (h_sum_x : sum_x_coord [(x1, y1), (x2, y2), (x3, y3)] = 0) :
  sum_y_coord [(x1, y1), (x2, y2), (x3, y3)] = 3 :=
sorry

end find_intersection_pair_l134_134350


namespace total_birds_distance_l134_134538

def birds_flew_collectively : Prop := 
  let distance_eagle := 15 * 2.5
  let distance_falcon := 46 * 2.5
  let distance_pelican := 33 * 2.5
  let distance_hummingbird := 30 * 2.5
  let distance_hawk := 45 * 3
  let distance_swallow := 25 * 1.5
  let total_distance := distance_eagle + distance_falcon + distance_pelican + distance_hummingbird + distance_hawk + distance_swallow
  total_distance = 482.5

theorem total_birds_distance : birds_flew_collectively := by
  -- proof goes here
  sorry

end total_birds_distance_l134_134538


namespace basket_white_ball_probability_l134_134313

noncomputable def basket_problem_proof : Prop :=
  let P_A := 1 / 2
  let P_B := 1 / 2
  let P_W_given_A := 2 / 5
  let P_W_given_B := 1 / 4
  let P_W := P_A * P_W_given_A + P_B * P_W_given_B
  let P_A_given_W := (P_A * P_W_given_A) / P_W
  P_A_given_W = 8 / 13

theorem basket_white_ball_probability :
  basket_problem_proof :=
  sorry

end basket_white_ball_probability_l134_134313


namespace team_plays_60_games_in_division_l134_134972

noncomputable def number_of_division_games (N M : ℕ) (hNM : N > M) (hM : M > 5) (h_total : 4 * N + 5 * M = 90) : ℕ :=
  4 * N

theorem team_plays_60_games_in_division (N M : ℕ) 
  (hNM : N > M) (hM : M > 5) (h_total : 4 * N + 5 * M = 90) 
  : number_of_division_games N M hNM hM h_total = 60 := 
sorry

end team_plays_60_games_in_division_l134_134972


namespace simplify_expression_l134_134900

theorem simplify_expression (x : ℝ) : 
  (3 * x - 4) * (2 * x + 10) - (x + 3) * (3 * x - 2) = 3 * x^2 + 15 * x - 34 := 
by
  sorry

end simplify_expression_l134_134900


namespace find_k_l134_134895

theorem find_k (k : ℕ) (h1 : k > 0) (h2 : 15 * k^4 < 120) : k = 1 := 
  sorry

end find_k_l134_134895


namespace first_place_beat_joe_l134_134582

theorem first_place_beat_joe (joe_won joe_draw first_place_won first_place_draw points_win points_draw : ℕ) 
    (h1 : joe_won = 1) (h2 : joe_draw = 3) (h3 : first_place_won = 2) (h4 : first_place_draw = 2)
    (h5 : points_win = 3) (h6 : points_draw = 1) : 
    (first_place_won * points_win + first_place_draw * points_draw) - (joe_won * points_win + joe_draw * points_draw) = 2 :=
by
   sorry

end first_place_beat_joe_l134_134582


namespace probability_two_female_one_male_l134_134653

-- Define basic conditions
def total_contestants : Nat := 7
def female_contestants : Nat := 4
def male_contestants : Nat := 3
def choose_count : Nat := 3

-- Calculate combinations (binomial coefficients)
def comb (n k : Nat) : Nat := Nat.choose n k

-- Define the probability calculation steps in Lean
def total_ways := comb total_contestants choose_count
def favorable_ways_female := comb female_contestants 2
def favorable_ways_male := comb male_contestants 1
def favorable_ways := favorable_ways_female * favorable_ways_male

theorem probability_two_female_one_male :
  (favorable_ways : ℚ) / (total_ways : ℚ) = 18 / 35 := by
  sorry

end probability_two_female_one_male_l134_134653


namespace arithmetic_geometric_mean_inequality_l134_134517

theorem arithmetic_geometric_mean_inequality (a b c : ℝ) (h₁ : 0 ≤ a) (h₂ : 0 ≤ b) (h₃ : 0 ≤ c) :
  (a + b + c) / 3 ≥ (a * b * c) ^ (1 / 3) :=
sorry

end arithmetic_geometric_mean_inequality_l134_134517


namespace booknote_unique_letters_count_l134_134325

def booknote_set : Finset Char := {'b', 'o', 'k', 'n', 't', 'e'}

theorem booknote_unique_letters_count : booknote_set.card = 6 :=
by
  sorry

end booknote_unique_letters_count_l134_134325


namespace no_such_P_exists_l134_134312

theorem no_such_P_exists (P : Polynomial ℤ) (r : ℕ) (r_ge_3 : r ≥ 3) (a : Fin r → ℤ)
  (distinct_a : ∀ i j, i ≠ j → a i ≠ a j)
  (P_cycle : ∀ i, P.eval (a i) = a ⟨(i + 1) % r, sorry⟩)
  : False :=
sorry

end no_such_P_exists_l134_134312


namespace ken_got_1750_l134_134525

theorem ken_got_1750 (K : ℝ) (h : K + 2 * K = 5250) : K = 1750 :=
sorry

end ken_got_1750_l134_134525


namespace smallest_sum_of_squares_l134_134460

theorem smallest_sum_of_squares (x y : ℕ) (h : x^2 - y^2 = 221) : x^2 + y^2 ≥ 229 :=
sorry

end smallest_sum_of_squares_l134_134460


namespace square_of_positive_difference_l134_134736

theorem square_of_positive_difference {y : ℝ}
  (h : (45 + y) / 2 = 50) :
  (|y - 45|)^2 = 100 :=
by
  sorry

end square_of_positive_difference_l134_134736


namespace geometric_means_insertion_l134_134585

noncomputable def is_geometric_progression (s : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ) (r_pos : r > 0), ∀ n, s (n + 1) = s n * r

theorem geometric_means_insertion (s : ℕ → ℝ) (n : ℕ)
  (h : is_geometric_progression s)
  (h_pos : ∀ i, s i > 0) :
  ∃ t : ℕ → ℝ, is_geometric_progression t :=
sorry

end geometric_means_insertion_l134_134585


namespace gcd_of_powers_l134_134760

theorem gcd_of_powers (a b c : ℕ) (h1 : a = 2^105 - 1) (h2 : b = 2^115 - 1) (h3 : c = 1023) :
  Nat.gcd a b = c :=
by sorry

end gcd_of_powers_l134_134760


namespace larger_number_of_two_l134_134152

theorem larger_number_of_two (A B : ℕ) (hcf : ℕ) (factor1 factor2 : ℕ) 
  (h_hcf : hcf = 28) (h_factors : A % hcf = 0 ∧ B % hcf = 0) 
  (h_f1 : factor1 = 12) (h_f2 : factor2 = 15)
  (h_lcm : Nat.lcm A B = hcf * factor1 * factor2)
  (h_coprime : Nat.gcd (A / hcf) (B / hcf) = 1)
  : max A B = 420 := 
sorry

end larger_number_of_two_l134_134152


namespace sum_of_n_values_l134_134575

theorem sum_of_n_values (n1 n2 : ℚ) (h1 : 3 * n1 - 8 = 5) (h2 : 3 * n2 - 8 = -5) :
  n1 + n2 = 16 / 3 := 
sorry

end sum_of_n_values_l134_134575


namespace parabola_focus_coordinates_l134_134794

theorem parabola_focus_coordinates (x y : ℝ) (h : x = 2 * y^2) : (x, y) = (1/8, 0) :=
sorry

end parabola_focus_coordinates_l134_134794


namespace xiao_ming_correct_answers_l134_134397

theorem xiao_ming_correct_answers :
  ∃ (m n : ℕ), m + n = 20 ∧ 5 * m - n = 76 ∧ m = 16 := 
by
  -- Definitions of points for correct and wrong answers
  let points_per_correct := 5 
  let points_deducted_per_wrong := 1

  -- Contestant's Scores and Conditions
  have contestant_a : 20 * points_per_correct - 0 * points_deducted_per_wrong = 100 := by sorry
  have contestant_b : 19 * points_per_correct - 1 * points_deducted_per_wrong = 94 := by sorry
  have contestant_c : 18 * points_per_correct - 2 * points_deducted_per_wrong = 88 := by sorry
  have contestant_d : 14 * points_per_correct - 6 * points_deducted_per_wrong = 64 := by sorry
  have contestant_e : 10 * points_per_correct - 10 * points_deducted_per_wrong = 40 := by sorry

  -- Xiao Ming's conditions translated to variables m and n
  have xiao_ming_conditions : (∃ m n : ℕ, m + n = 20 ∧ 5 * m - n = 76) := by sorry

  exact ⟨16, 4, rfl, rfl, rfl⟩

end xiao_ming_correct_answers_l134_134397


namespace inequality_one_inequality_two_l134_134964

-- Definitions of the three positive real numbers and their sum of reciprocals squared is equal to 1
variables {a b c : ℝ}
variable (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
variable (h_sum : (1 / a^2) + (1 / b^2) + (1 / c^2) = 1)

-- First proof that (1/a + 1/b + 1/c) <= sqrt(3)
theorem inequality_one (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : (1 / a^2) + (1 / b^2) + (1 / c^2) = 1) :
  (1 / a) + (1 / b) + (1 / c) ≤ Real.sqrt 3 :=
sorry

-- Second proof that (a^2/b^4) + (b^2/c^4) + (c^2/a^4) >= 1
theorem inequality_two (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : (1 / a^2) + (1 / b^2) + (1 / c^2) = 1) :
  (a^2 / b^4) + (b^2 / c^4) + (c^2 / a^4) ≥ 1 :=
sorry

end inequality_one_inequality_two_l134_134964


namespace minimum_value_of_f_symmetry_of_f_monotonic_decreasing_f_l134_134933

noncomputable def f (x : Real) : Real := Real.cos (2*x) - 2*Real.sin x + 1

theorem minimum_value_of_f : ∃ x : Real, f x = -2 := sorry

theorem symmetry_of_f : ∀ x : Real, f x = f (π - x) := sorry

theorem monotonic_decreasing_f : ∀ x y : Real, 0 < x ∧ x < y ∧ y < π / 2 → f y < f x := sorry

end minimum_value_of_f_symmetry_of_f_monotonic_decreasing_f_l134_134933


namespace special_discount_percentage_l134_134311

theorem special_discount_percentage (original_price discounted_price : ℝ) (h₀ : original_price = 80) (h₁ : discounted_price = 68) : 
  ((original_price - discounted_price) / original_price) * 100 = 15 :=
by 
  sorry

end special_discount_percentage_l134_134311


namespace solution_to_problem_l134_134626

theorem solution_to_problem
  {x y z : ℝ}
  (h1 : xy / (x + y) = 1 / 3)
  (h2 : yz / (y + z) = 1 / 5)
  (h3 : zx / (z + x) = 1 / 6) :
  xyz / (xy + yz + zx) = 1 / 7 :=
by sorry

end solution_to_problem_l134_134626


namespace stacy_savings_for_3_pairs_l134_134803

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

end stacy_savings_for_3_pairs_l134_134803


namespace total_perimeter_of_compound_shape_l134_134597

-- Definitions of the conditions from the original problem
def triangle1_side : ℝ := 10
def triangle2_side : ℝ := 6
def shared_side : ℝ := 6

-- A theorem to represent the mathematically equivalent proof problem
theorem total_perimeter_of_compound_shape 
  (t1s : ℝ := triangle1_side) 
  (t2s : ℝ := triangle2_side)
  (ss : ℝ := shared_side) : 
  t1s = 10 ∧ t2s = 6 ∧ ss = 6 → 3 * t1s + 3 * t2s - ss = 42 := 
by
  sorry

end total_perimeter_of_compound_shape_l134_134597


namespace platform_length_l134_134117

theorem platform_length (train_length : ℕ) (time_cross_platform : ℕ) (time_cross_pole : ℕ) (train_speed : ℕ) (L : ℕ)
  (h1 : train_length = 500) 
  (h2 : time_cross_platform = 65) 
  (h3 : time_cross_pole = 25) 
  (h4 : train_speed = train_length / time_cross_pole)
  (h5 : train_speed = (train_length + L) / time_cross_platform) :
  L = 800 := 
sorry

end platform_length_l134_134117


namespace unique_real_solution_l134_134025

theorem unique_real_solution :
  ∃! (a b : ℝ), 2 * (a^2 + 1) * (b^2 + 1) = (a + 1)^2 * (ab + 1) ∧ a = 1 ∧ b = 1 :=
by
  sorry

end unique_real_solution_l134_134025


namespace carrot_cakes_in_february_l134_134924

theorem carrot_cakes_in_february :
  (∃ (cakes_in_oct : ℕ) (cakes_in_nov : ℕ) (cakes_in_dec : ℕ) (cakes_in_jan : ℕ) (monthly_increase : ℕ),
      cakes_in_oct = 19 ∧
      cakes_in_nov = 21 ∧
      cakes_in_dec = 23 ∧
      cakes_in_jan = 25 ∧
      monthly_increase = 2 ∧
      cakes_in_february = cakes_in_jan + monthly_increase) →
  cakes_in_february = 27 :=
  sorry

end carrot_cakes_in_february_l134_134924


namespace find_p_q_r_l134_134452

def f (x : ℝ) : ℝ := x^2 + 2*x + 2
def g (x p q r : ℝ) : ℝ := x^3 + 2*x^2 + 6*p*x + 4*q*x + r

noncomputable def roots_sum_f := -2
noncomputable def roots_product_f := 2

theorem find_p_q_r (p q r : ℝ) (h1 : ∀ x, f x = 0 → g x p q r = 0) :
  (p + q) * r = 0 :=
sorry

end find_p_q_r_l134_134452


namespace ratio_mara_janet_l134_134041

variables {B J M : ℕ}

/-- Janet has 9 cards more than Brenda --/
def janet_cards (B : ℕ) : ℕ := B + 9

/-- Mara has 40 cards less than 150 --/
def mara_cards : ℕ := 150 - 40

/-- They have a total of 211 cards --/
axiom total_cards_eq (B : ℕ) : B + janet_cards B + mara_cards = 211

/-- Mara has a multiple of Janet's number of cards --/
axiom multiples_cards (J M : ℕ) : J * 2 = M

theorem ratio_mara_janet (B J M : ℕ) (h1 : janet_cards B = J)
  (h2 : mara_cards = M) (h3 : J * 2 = M) :
  (M / J : ℕ) = 2 :=
sorry

end ratio_mara_janet_l134_134041


namespace inheritance_amount_l134_134826

theorem inheritance_amount (x : ℝ)
  (federal_tax_rate : ℝ := 0.25)
  (state_tax_rate : ℝ := 0.15)
  (total_taxes_paid : ℝ := 16000)
  (H : (federal_tax_rate * x) + (state_tax_rate * (1 - federal_tax_rate) * x) = total_taxes_paid) :
  x = 44138 := sorry

end inheritance_amount_l134_134826


namespace least_integer_gt_square_l134_134740

theorem least_integer_gt_square (x : ℝ) (y : ℝ) (h1 : x = 2) (h2 : y = Real.sqrt 3) :
  ∃ (n : ℤ), n = 14 ∧ n > (x + y) ^ 2 := by
  sorry

end least_integer_gt_square_l134_134740


namespace inequality_solution_range_l134_134214

theorem inequality_solution_range (a : ℝ) :
  (∃ x : ℝ, |x + 2| + |x| ≤ a) ↔ a ≥ 2 :=
by
  sorry

end inequality_solution_range_l134_134214


namespace graph_passes_through_quadrants_l134_134076

theorem graph_passes_through_quadrants (k : ℝ) (h : k < 0) :
  ∀ (x y : ℝ), (y = k * x - k) → 
    ((0 < x ∧ 0 < y) ∨ (x < 0 ∧ 0 < y) ∨ (x < 0 ∧ y < 0)) :=
by
  sorry

end graph_passes_through_quadrants_l134_134076


namespace sms_message_fraudulent_l134_134761

-- Define the conditions as properties
def messageArrivedNumberKnown (msg : String) (numberKnown : Bool) : Prop :=
  msg = "SMS message has already arrived" ∧ numberKnown = true

def fraudDefinition (acquisition : String -> Prop) : Prop :=
  ∀ (s : String), acquisition s = (s = "acquisition of property by third parties through deception or gaining the trust of the victim")

-- Define the main proof problem statement
theorem sms_message_fraudulent (msg : String) (numberKnown : Bool) (acquisition : String -> Prop) :
  messageArrivedNumberKnown msg numberKnown ∧ fraudDefinition acquisition →
  acquisition "acquisition of property by third parties through deception or gaining the trust of the victim" :=
  sorry

end sms_message_fraudulent_l134_134761


namespace probability_good_or_excellent_l134_134502

noncomputable def P_H1 : ℚ := 5 / 21
noncomputable def P_H2 : ℚ := 10 / 21
noncomputable def P_H3 : ℚ := 6 / 21

noncomputable def P_A_given_H1 : ℚ := 1
noncomputable def P_A_given_H2 : ℚ := 1
noncomputable def P_A_given_H3 : ℚ := 1 / 3

noncomputable def P_A : ℚ := 
  P_H1 * P_A_given_H1 + 
  P_H2 * P_A_given_H2 + 
  P_H3 * P_A_given_H3

theorem probability_good_or_excellent : P_A = 17 / 21 :=
by
  sorry

end probability_good_or_excellent_l134_134502


namespace complement_of_angle_l134_134832

variable (α : ℝ)

axiom given_angle : α = 63 + 21 / 60

theorem complement_of_angle :
  90 - α = 26 + 39 / 60 :=
by
  sorry

end complement_of_angle_l134_134832


namespace simplify_fraction_l134_134851

theorem simplify_fraction :
  (45 * (14 / 25) * (1 / 18) * (5 / 11) : ℚ) = 7 / 11 := 
by sorry

end simplify_fraction_l134_134851


namespace roots_difference_is_one_l134_134857

noncomputable def quadratic_eq (p : ℝ) :=
  ∃ (α β : ℝ), (α ≠ β) ∧ (α - β = 1) ∧ (α ^ 2 - p * α + (p ^ 2 - 1) / 4 = 0) ∧ (β ^ 2 - p * β + (p ^ 2 - 1) / 4 = 0)

theorem roots_difference_is_one (p : ℝ) : quadratic_eq p :=
  sorry

end roots_difference_is_one_l134_134857


namespace quadratic_complete_square_l134_134038

theorem quadratic_complete_square :
  ∃ a b c : ℤ, (8 * x^2 - 48 * x - 320 = a * (x + b)^2 + c) ∧ (a + b + c = -387) :=
sorry

end quadratic_complete_square_l134_134038


namespace initial_water_amount_l134_134016

theorem initial_water_amount (W : ℝ) 
  (evap_per_day : ℝ := 0.0008) 
  (days : ℤ := 50) 
  (percentage_evap : ℝ := 0.004) 
  (evap_total : ℝ := evap_per_day * days) 
  (evap_eq : evap_total = percentage_evap * W) : 
  W = 10 := 
by
  sorry

end initial_water_amount_l134_134016


namespace great_dane_weight_l134_134920

def weight_problem (C P G : ℝ) : Prop :=
  (P = 3 * C) ∧ (G = 3 * P + 10) ∧ (C + P + G = 439)

theorem great_dane_weight : ∃ (C P G : ℝ), weight_problem C P G ∧ G = 307 :=
by
  sorry

end great_dane_weight_l134_134920


namespace condition_for_diff_of_roots_l134_134629

/-- Statement: For a quadratic equation of the form x^2 + px + q = 0, if the difference of the roots is a, then the condition a^2 - p^2 = -4q holds. -/
theorem condition_for_diff_of_roots (p q a : ℝ) (h : ∀ x : ℝ, x^2 + p * x + q = 0 → ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ x1 - x2 = a) :
  a^2 - p^2 = -4 * q :=
sorry

end condition_for_diff_of_roots_l134_134629


namespace trigonometric_identity_l134_134097

open Real

noncomputable def sin_alpha (x y : ℝ) : ℝ :=
  y / sqrt (x^2 + y^2)

noncomputable def tan_alpha (x y : ℝ) : ℝ :=
  y / x

theorem trigonometric_identity (x y : ℝ) (h_x : x = 3/5) (h_y : y = -4/5) :
  sin_alpha x y * tan_alpha x y = 16/15 :=
by {
  -- math proof to be provided here
  sorry
}

end trigonometric_identity_l134_134097


namespace neg_prop_p_equiv_l134_134473

variable {x : ℝ}

def prop_p : Prop := ∃ x ≥ 0, 2^x = 3

theorem neg_prop_p_equiv : ¬prop_p ↔ ∀ x ≥ 0, 2^x ≠ 3 :=
by sorry

end neg_prop_p_equiv_l134_134473


namespace john_sells_20_woodburnings_l134_134496

variable (x : ℕ)

theorem john_sells_20_woodburnings (price_per_woodburning cost profit : ℤ) 
  (h1 : price_per_woodburning = 15) (h2 : cost = 100) (h3 : profit = 200) :
  (profit = price_per_woodburning * x - cost) → 
  x = 20 :=
by
  intros h_profit
  rw [h1, h2, h3] at h_profit
  linarith

end john_sells_20_woodburnings_l134_134496


namespace circle_line_intersection_zero_l134_134768

theorem circle_line_intersection_zero (x_0 y_0 r : ℝ) (hP : x_0^2 + y_0^2 < r^2) :
  ∀ (x y : ℝ), (x^2 + y^2 = r^2) → (x_0 * x + y_0 * y = r^2) → false :=
by
  sorry

end circle_line_intersection_zero_l134_134768


namespace carousel_ticket_cost_l134_134201

theorem carousel_ticket_cost :
  ∃ (x : ℕ), 
  (2 * 5) + (3 * x) = 19 ∧ x = 3 :=
by
  sorry

end carousel_ticket_cost_l134_134201


namespace problem_1_solution_set_problem_2_min_value_l134_134232

-- Problem (1)
def f (x : ℝ) : ℝ := 4 - |x| - |x - 3|

theorem problem_1_solution_set :
  {x : ℝ | f (x + 3/2) ≥ 0} = {x | -2 ≤ x ∧ x ≤ 2} :=
by
  sorry

-- Problem (2)
theorem problem_2_min_value (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) 
  (h : 1/(3*p) + 1/(2*q) + 1/r = 4) : 
  3*p + 2*q + r ≥ 9/4 :=
by
  sorry

end problem_1_solution_set_problem_2_min_value_l134_134232


namespace time_to_finish_all_problems_l134_134423

def mathProblems : ℝ := 17.0
def spellingProblems : ℝ := 15.0
def problemsPerHour : ℝ := 8.0
def totalProblems : ℝ := mathProblems + spellingProblems

theorem time_to_finish_all_problems : totalProblems / problemsPerHour = 4.0 :=
by
  sorry

end time_to_finish_all_problems_l134_134423


namespace martha_total_cost_l134_134986

def weight_cheese : ℝ := 1.5
def weight_meat : ℝ := 0.55    -- converting grams to kg
def weight_pasta : ℝ := 0.28   -- converting grams to kg
def weight_tomatoes : ℝ := 2.2

def price_cheese_per_kg : ℝ := 6.30
def price_meat_per_kg : ℝ := 8.55
def price_pasta_per_kg : ℝ := 2.40
def price_tomatoes_per_kg : ℝ := 1.79

def tax_cheese : ℝ := 0.07
def tax_meat : ℝ := 0.06
def tax_pasta : ℝ := 0.08
def tax_tomatoes : ℝ := 0.05

def total_cost : ℝ :=
  let cost_cheese := weight_cheese * price_cheese_per_kg * (1 + tax_cheese)
  let cost_meat := weight_meat * price_meat_per_kg * (1 + tax_meat)
  let cost_pasta := weight_pasta * price_pasta_per_kg * (1 + tax_pasta)
  let cost_tomatoes := weight_tomatoes * price_tomatoes_per_kg * (1 + tax_tomatoes)
  cost_cheese + cost_meat + cost_pasta + cost_tomatoes

theorem martha_total_cost : total_cost = 19.9568 := by
  sorry

end martha_total_cost_l134_134986


namespace consecutive_numbers_product_l134_134155

theorem consecutive_numbers_product : 
  ∃ n : ℕ, (n + n + 1 = 11) ∧ (n * (n + 1) * (n + 2) = 210) :=
sorry

end consecutive_numbers_product_l134_134155


namespace K_time_for_distance_l134_134479

theorem K_time_for_distance (s : ℝ) (hs : s > 0) :
  (let K_time := 45 / s
   let M_speed := s - 1 / 2
   let M_time := 45 / M_speed
   K_time = M_time - 3 / 4) -> K_time = 45 / s := 
by
  sorry

end K_time_for_distance_l134_134479


namespace problem1_problem2_problem3_problem4_l134_134450

-- statement for problem 1
theorem problem1 : -5 + 8 - 2 = 1 := by
  sorry

-- statement for problem 2
theorem problem2 : (-3) * (5/6) / (-1/4) = 10 := by
  sorry

-- statement for problem 3
theorem problem3 : -3/17 + (-3.75) + (-14/17) + (15/4) = -1 := by
  sorry

-- statement for problem 4
theorem problem4 : -(1^10) - ((13/14) - (11/12)) * (4 - (-2)^2) + (1/2) / 3 = -(5/6) := by
  sorry

end problem1_problem2_problem3_problem4_l134_134450


namespace find_c_k_l134_134262

theorem find_c_k (a b : ℕ → ℕ) (c : ℕ → ℕ) (k : ℕ) (d r : ℕ) 
  (h1 : ∀ n, a n = 1 + (n-1)*d)
  (h2 : ∀ n, b n = r^(n-1))
  (h3 : ∀ n, c n = a n + b n)
  (h4 : c (k-1) = 80)
  (h5 : c (k+1) = 500) :
  c k = 167 := sorry

end find_c_k_l134_134262


namespace jane_book_pages_l134_134074

theorem jane_book_pages (x : ℝ) :
  (x - (1 / 4 * x + 10) - (1 / 5 * (x - (1 / 4 * x + 10)) + 20) - (1 / 2 * (x - (1 / 4 * x + 10) - (1 / 5 * (x - (1 / 4 * x + 10)) + 20)) + 25) = 75) → x = 380 :=
by
  sorry

end jane_book_pages_l134_134074


namespace sqrt_meaningful_real_domain_l134_134396

theorem sqrt_meaningful_real_domain (x : ℝ) (h : 6 - 4 * x ≥ 0) : x ≤ 3 / 2 :=
by sorry

end sqrt_meaningful_real_domain_l134_134396


namespace john_loses_probability_eq_3_over_5_l134_134359

-- Definitions used directly from the conditions in a)
def probability_win := 2 / 5
def probability_lose := 1 - probability_win

-- The theorem statement
theorem john_loses_probability_eq_3_over_5 : 
  probability_lose = 3 / 5 := 
by
  sorry -- proof is to be filled in later

end john_loses_probability_eq_3_over_5_l134_134359


namespace find_2a6_minus_a4_l134_134166

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 2) = 2 * a (n + 1) - a n

theorem find_2a6_minus_a4 {a : ℕ → ℤ} 
  (h_seq : is_arithmetic_sequence a)
  (h_cond : a 1 + 3 * a 8 + a 15 = 120) : 
  2 * a 6 - a 4 = 24 :=
by
  sorry

end find_2a6_minus_a4_l134_134166


namespace prob_xi_eq_12_l134_134586

noncomputable def prob_of_draws (total_draws red_draws : ℕ) (prob_red prob_white : ℚ) : ℚ :=
    (Nat.choose (total_draws - 1) (red_draws - 1)) * (prob_red ^ (red_draws - 1)) * (prob_white ^ (total_draws - red_draws)) * prob_red

theorem prob_xi_eq_12 :
    prob_of_draws 12 10 (3 / 8) (5 / 8) = 
    (Nat.choose 11 9) * (3 / 8)^9 * (5 / 8)^2 * (3 / 8) :=
by sorry

end prob_xi_eq_12_l134_134586


namespace value_of_a_if_1_in_S_l134_134419

variable (a : ℤ)
def S := { x : ℤ | 3 * x + a = 0 }

theorem value_of_a_if_1_in_S (h : 1 ∈ S a) : a = -3 :=
sorry

end value_of_a_if_1_in_S_l134_134419


namespace min_disks_required_l134_134836

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

end min_disks_required_l134_134836


namespace distribute_positions_l134_134565

theorem distribute_positions :
  let positions := 11
  let classes := 6
  ∃ total_ways : ℕ, total_ways = Nat.choose (positions - 1) (classes - 1) ∧ total_ways = 252 :=
by
  let positions := 11
  let classes := 6
  have : Nat.choose (positions - 1) (classes - 1) = 252 := by sorry
  exact ⟨Nat.choose (positions - 1) (classes - 1), this, this⟩

end distribute_positions_l134_134565


namespace geometric_sequence_condition_l134_134128

theorem geometric_sequence_condition {a : ℕ → ℝ} (h_geom : ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n) : 
  (a 3 * a 5 = 16) ↔ a 4 = 4 :=
sorry

end geometric_sequence_condition_l134_134128


namespace Shekar_marks_in_Science_l134_134246

theorem Shekar_marks_in_Science (S : ℕ) (h : (76 + S + 82 + 67 + 85) / 5 = 75) : S = 65 :=
sorry

end Shekar_marks_in_Science_l134_134246


namespace system_of_equations_solution_system_of_inequalities_solution_l134_134490

-- Problem (1): Solve the system of equations
theorem system_of_equations_solution :
  ∃ (x y : ℝ), (3 * (y - 2) = x - 1) ∧ (2 * (x - 1) = 5 * y - 8) ∧ (x = 7) ∧ (y = 4) :=
by
  sorry

-- Problem (2): Solve the system of linear inequalities
theorem system_of_inequalities_solution :
  ∃ (x : ℝ), (3 * x ≤ 2 * x + 3) ∧ ((x + 1) / 6 - 1 < (2 * (x + 1)) / 3) ∧ (-3 < x) ∧ (x ≤ 3) :=
by
  sorry

end system_of_equations_solution_system_of_inequalities_solution_l134_134490


namespace sum_of_remainders_and_smallest_n_l134_134015

theorem sum_of_remainders_and_smallest_n (n : ℕ) (h : n % 20 = 11) :
    (n % 4 + n % 5 = 4) ∧ (∃ (k : ℕ), k > 2 ∧ n = 20 * k + 11 ∧ n > 50) := by
  sorry

end sum_of_remainders_and_smallest_n_l134_134015


namespace expand_polynomials_l134_134369

-- Define the given polynomials
def poly1 (x : ℝ) : ℝ := 12 * x^2 + 5 * x - 3
def poly2 (x : ℝ) : ℝ := 3 * x^3 + 2

-- Define the expected result of the polynomial multiplication
def expected (x : ℝ) : ℝ := 36 * x^5 + 15 * x^4 - 9 * x^3 + 24 * x^2 + 10 * x - 6

-- State the theorem
theorem expand_polynomials (x : ℝ) :
  (poly1 x) * (poly2 x) = expected x :=
by
  sorry

end expand_polynomials_l134_134369


namespace product_gcd_lcm_l134_134989

-- Define the numbers
def a : ℕ := 24
def b : ℕ := 60

-- Define the gcd and lcm
def gcd_ab := Nat.gcd a b
def lcm_ab := Nat.lcm a b

-- Statement to prove: the product of gcd and lcm of 24 and 60 equals 1440
theorem product_gcd_lcm : gcd_ab * lcm_ab = 1440 := by
  -- gcd_ab = 12
  -- lcm_ab = 120
  -- Thus, 12 * 120 = 1440
  sorry

end product_gcd_lcm_l134_134989


namespace max_value_f_l134_134547

noncomputable def f (x : ℝ) : ℝ := 8 * Real.sin x - Real.tan x

theorem max_value_f : 
  ∃ x ∈ Set.Ioo 0 (Real.pi / 2), ∀ y ∈ Set.Ioo 0 (Real.pi / 2), f y ≤ f x ∧ f x = 3 * Real.sqrt 3 :=
by
  sorry

end max_value_f_l134_134547


namespace reading_time_equal_l134_134829

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

end reading_time_equal_l134_134829


namespace g_2002_equals_1_l134_134784

theorem g_2002_equals_1 (f : ℝ → ℝ)
  (hf1 : f 1 = 1)
  (hf2 : ∀ x : ℝ, f (x + 5) ≥ f x + 5)
  (hf3 : ∀ x : ℝ, f (x + 1) ≤ f x + 1)
  (g : ℝ → ℝ := fun x => f x + 1 - x)
  : g 2002 = 1 :=
by
  sorry

end g_2002_equals_1_l134_134784


namespace original_six_digit_number_is_285714_l134_134221

theorem original_six_digit_number_is_285714 
  (N : ℕ) 
  (h1 : ∃ x, N = 200000 + x ∧ 10 * x + 2 = 3 * (200000 + x)) :
  N = 285714 := 
sorry

end original_six_digit_number_is_285714_l134_134221


namespace min_difference_xue_jie_ti_neng_li_l134_134873

theorem min_difference_xue_jie_ti_neng_li : 
  ∀ (shu hsue jie ti neng li zhan shi : ℕ), 
  shu = 8 ∧ hsue = 1 ∧ jie = 4 ∧ ti = 3 ∧ neng = 9 ∧ li = 5 ∧ zhan = 7 ∧ shi = 2 →
  (shu * 1000 + hsue * 100 + jie * 10 + ti) = 1842 →
  (neng * 10 + li) = 95 →
  1842 - 95 = 1747 := 
by
  intros shu hsue jie ti neng li zhan shi h_digits h_xue_jie_ti h_neng_li
  sorry

end min_difference_xue_jie_ti_neng_li_l134_134873


namespace complement_union_l134_134149

namespace SetProof

variable (U M N : Set ℕ)
variable (H_U : U = {1, 2, 3, 4, 5})
variable (H_M : M = {1, 2})
variable (H_N : N = {3, 4})

theorem complement_union :
  (U \ (M ∪ N)) = {5} := by
  sorry

end SetProof

end complement_union_l134_134149


namespace solution_xyz_uniqueness_l134_134081

theorem solution_xyz_uniqueness (x y z : ℝ) :
  x + y + z = 3 ∧ x^2 + y^2 + z^2 = 3 ∧ x^3 + y^3 + z^3 = 3 → x = 1 ∧ y = 1 ∧ z = 1 :=
by
  sorry

end solution_xyz_uniqueness_l134_134081


namespace exists_rectangle_with_properties_l134_134762

variables {e a φ : ℝ}

-- Define the given conditions
def diagonal_diff (e a : ℝ) := e - a
def angle_between_diagonals (φ : ℝ) := φ

-- The problem to prove
theorem exists_rectangle_with_properties (e a φ : ℝ) 
  (h_diff : diagonal_diff e a = e - a) 
  (h_angle : angle_between_diagonals φ = φ) : 
  ∃ (rectangle : Type) (A B C D : rectangle), 
    (e - a = e - a) ∧ 
    (φ = φ) := 
sorry

end exists_rectangle_with_properties_l134_134762


namespace trapezoid_bases_12_and_16_l134_134587

theorem trapezoid_bases_12_and_16 :
  ∀ (h R : ℝ) (a b : ℝ),
    (R = 10) →
    (h = (a + b) / 2) →
    (∀ k m, ((k = 3/7 * h) ∧ (m = 4/7 * h) ∧ (R^2 = k^2 + (a/2)^2) ∧ (R^2 = m^2 + (b/2)^2))) →
    (a = 12) ∧ (b = 16) :=
by
  intros h R a b hR hMid eqns
  sorry

end trapezoid_bases_12_and_16_l134_134587


namespace find_a_8_l134_134673

noncomputable def sequence_a (a : ℕ → ℤ) : Prop :=
  a 1 = 3 ∧ ∃ b : ℕ → ℤ, (∀ n : ℕ, 0 < n → b n = a (n + 1) - a n) ∧
  b 3 = -2 ∧ b 10 = 12

theorem find_a_8 (a : ℕ → ℤ) (h : sequence_a a) : a 8 = 3 :=
sorry

end find_a_8_l134_134673


namespace part_a_part_b_l134_134954

noncomputable def arithmetic_progression_a (a₁: ℕ) (r: ℕ) : ℕ :=
  a₁ + 3 * r

theorem part_a (a₁: ℕ) (r: ℕ) (h_a₁ : a₁ = 2) (h_r : r = 3) : arithmetic_progression_a a₁ r = 11 := 
by 
  sorry

noncomputable def arithmetic_progression_formula (d: ℕ) (r: ℕ) (n: ℕ) : ℕ :=
  d + (n - 1) * r

theorem part_b (a3: ℕ) (a6: ℕ) (a9: ℕ) (a4_plus_a7_plus_a10: ℕ) (a_sum: ℕ) (h_a3 : a3 = 3) (h_a6 : a6 = 6) (h_a9 : a9 = 9) 
  (h_a4a7a10 : a4_plus_a7_plus_a10 = 207) (h_asum : a_sum = 553) 
  (h_eqn1: 3 * a3 + a6 * 2 = 207) (h_eqn2: a_sum = 553): 
  arithmetic_progression_formula 9 10 11 = 109 := 
by 
  sorry

end part_a_part_b_l134_134954


namespace least_faces_combined_l134_134204

noncomputable def least_number_of_faces (c d : ℕ) : ℕ :=
c + d

theorem least_faces_combined (c d : ℕ) (h_cge8 : c ≥ 8) (h_dge8 : d ≥ 8)
  (h_sum9_prob : 8 / (c * d) = 1 / 2 * 16 / (c * d))
  (h_sum15_prob : ∃ m : ℕ, m / (c * d) = 1 / 15) :
  least_number_of_faces c d = 28 := sorry

end least_faces_combined_l134_134204


namespace find_range_of_m_l134_134216

def has_two_distinct_real_roots (m : ℝ) : Prop :=
  m^2 - 4 > 0

def inequality_holds_for_all_real_x (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2 * (m + 1) * x + m * (m + 1) > 0

def p (m : ℝ) : Prop := has_two_distinct_real_roots m
def q (m : ℝ) : Prop := inequality_holds_for_all_real_x m

theorem find_range_of_m (m : ℝ) :
  (p m ∨ q m) ∧ ¬(p m ∧ q m) → (m > 2 ∨ (-2 ≤ m ∧ m < -1)) :=
sorry

end find_range_of_m_l134_134216


namespace blueberries_count_l134_134888

theorem blueberries_count
  (initial_apples : ℕ)
  (initial_oranges : ℕ)
  (initial_blueberries : ℕ)
  (apples_eaten : ℕ)
  (oranges_eaten : ℕ)
  (remaining_fruits : ℕ)
  (h1 : initial_apples = 14)
  (h2 : initial_oranges = 9)
  (h3 : apples_eaten = 1)
  (h4 : oranges_eaten = 1)
  (h5 : remaining_fruits = 26) :
  initial_blueberries = 5 := 
by
  sorry

end blueberries_count_l134_134888


namespace correct_expression_after_removing_parentheses_l134_134620

variable (a b c : ℝ)

theorem correct_expression_after_removing_parentheses :
  -2 * (a + b - 3 * c) = -2 * a - 2 * b + 6 * c :=
sorry

end correct_expression_after_removing_parentheses_l134_134620


namespace number_of_matching_pages_l134_134358

theorem number_of_matching_pages : 
  ∃ (n : Nat), n = 13 ∧ ∀ x, 1 ≤ x ∧ x ≤ 63 → (x % 10 = (64 - x) % 10) ↔ x % 10 = 2 ∨ x % 10 = 7 :=
by
  sorry

end number_of_matching_pages_l134_134358


namespace robert_ate_more_l134_134017

variable (robert_chocolates : ℕ) (nickel_chocolates : ℕ)
variable (robert_ate_9 : robert_chocolates = 9) (nickel_ate_2 : nickel_chocolates = 2)

theorem robert_ate_more : robert_chocolates - nickel_chocolates = 7 :=
  by
    sorry

end robert_ate_more_l134_134017


namespace determine_c_l134_134822

theorem determine_c (c : ℝ) :
  let vertex_x := -(-10 / (2 * 1))
  let vertex_y := c - ((-10)^2 / (4 * 1))
  ((5 - 0)^2 + (vertex_y - 0)^2 = 10^2)
  → (c = 25 + 5 * Real.sqrt 3 ∨ c = 25 - 5 * Real.sqrt 3) :=
by
  sorry

end determine_c_l134_134822


namespace find_a_l134_134254

-- Definitions of the conditions
def A (a : ℝ) : Set ℝ := {a + 2, (a + 1) ^ 2, a ^ 2 + 3 * a + 3}

-- The proof goal
theorem find_a (a : ℝ) (h : 1 ∈ A a) : a = 0 := 
by 
  sorry

end find_a_l134_134254


namespace madeline_rent_l134_134231

noncomputable def groceries : ℝ := 400
noncomputable def medical_expenses : ℝ := 200
noncomputable def utilities : ℝ := 60
noncomputable def emergency_savings : ℝ := 200
noncomputable def hourly_wage : ℝ := 15
noncomputable def hours_worked : ℕ := 138
noncomputable def total_expenses_and_savings : ℝ := groceries + medical_expenses + utilities + emergency_savings
noncomputable def total_earnings : ℝ := hourly_wage * hours_worked

theorem madeline_rent : total_earnings - total_expenses_and_savings = 1210 := by
  sorry

end madeline_rent_l134_134231


namespace pencil_sharpening_time_l134_134090

theorem pencil_sharpening_time (t : ℕ) :
  let hand_crank_rate := 45
  let electric_rate := 20
  let sharpened_by_hand := (60 * t) / hand_crank_rate
  let sharpened_by_electric := (60 * t) / electric_rate
  (sharpened_by_electric = sharpened_by_hand + 10) → 
  t = 6 :=
by
  intros hand_crank_rate electric_rate sharpened_by_hand sharpened_by_electric h
  sorry

end pencil_sharpening_time_l134_134090


namespace fewest_occupied_seats_l134_134333

theorem fewest_occupied_seats (n m : ℕ) (h₁ : n = 150) (h₂ : (m * 4 + 3 < 150)) : m = 37 :=
by
  sorry

end fewest_occupied_seats_l134_134333


namespace exists_positive_int_solutions_l134_134274

theorem exists_positive_int_solutions (a : ℕ) (ha : a > 2) :
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x^2 - y^2 = a^2 :=
by
  sorry

end exists_positive_int_solutions_l134_134274


namespace total_pencils_l134_134739

theorem total_pencils (pencils_per_child : ℕ) (children : ℕ) (hp : pencils_per_child = 2) (hc : children = 8) :
  pencils_per_child * children = 16 :=
by
  sorry

end total_pencils_l134_134739


namespace angle_measure_l134_134508

theorem angle_measure (x : ℝ) 
  (h1 : 5 * x + 12 = 180 - x) : x = 28 := by
  sorry

end angle_measure_l134_134508


namespace complex_multiplication_example_l134_134850

def imaginary_unit (i : ℂ) : Prop := i^2 = -1

theorem complex_multiplication_example (i : ℂ) (h : imaginary_unit i) :
  (3 + i) * (1 - 2 * i) = 5 - 5 * i := 
by
  sorry

end complex_multiplication_example_l134_134850


namespace joan_mortgage_payoff_l134_134891

/-- Joan's mortgage problem statement. -/
theorem joan_mortgage_payoff (a r : ℕ) (total : ℕ) (n : ℕ) : a = 100 → r = 3 → total = 12100 → 
    total = a * (1 - r^n) / (1 - r) → n = 5 :=
by intros ha hr htotal hgeom; sorry

end joan_mortgage_payoff_l134_134891


namespace seventh_grade_caps_collection_l134_134978

theorem seventh_grade_caps_collection (A B C : ℕ) 
  (h1 : A = (B + C) / 2) 
  (h2 : B = (A + C) / 3)
  (h3 : C = 150) : A + B + C = 360 := 
by 
  sorry

end seventh_grade_caps_collection_l134_134978


namespace min_value_f_l134_134193

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 + 24 * x + 128 / x^3

theorem min_value_f : ∃ x > 0, f x = 168 :=
by
  sorry

end min_value_f_l134_134193


namespace income_calculation_l134_134421

theorem income_calculation (savings : ℕ) (income_ratio : ℕ) (expenditure_ratio : ℕ) 
  (ratio_condition : income_ratio = 5 ∧ expenditure_ratio = 4) (savings_condition : savings = 3800) :
  income_ratio * savings / (income_ratio - expenditure_ratio) = 19000 :=
by
  sorry

end income_calculation_l134_134421


namespace number_of_tables_l134_134748

theorem number_of_tables (x : ℕ) (h : 2 * (x - 1) + 3 = 65) : x = 32 :=
sorry

end number_of_tables_l134_134748


namespace remainder_when_added_then_divided_l134_134550

def num1 : ℕ := 2058167
def num2 : ℕ := 934
def divisor : ℕ := 8

theorem remainder_when_added_then_divided :
  (num1 + num2) % divisor = 5 := 
sorry

end remainder_when_added_then_divided_l134_134550


namespace green_flowers_count_l134_134405

theorem green_flowers_count :
  ∀ (G R B Y T : ℕ),
    T = 96 →
    R = 3 * G →
    B = 48 →
    Y = 12 →
    G + R + B + Y = T →
    G = 9 :=
by
  intros G R B Y T
  intro hT
  intro hR
  intro hB
  intro hY
  intro hSum
  sorry

end green_flowers_count_l134_134405


namespace jessy_initial_reading_plan_l134_134342

theorem jessy_initial_reading_plan (x : ℕ) (h : (7 * (3 * x + 2) = 140)) : x = 6 :=
sorry

end jessy_initial_reading_plan_l134_134342


namespace false_proposition_l134_134522

open Classical

variables (a b : ℝ) (x : ℝ)

def P := ∃ (a b : ℝ), (0 < a) ∧ (0 < b) ∧ (a + b = 1) ∧ ((1 / a) + (1 / b) = 3)
def Q := ∀ (x : ℝ), x^2 - x + 1 ≥ 0

theorem false_proposition :
  (¬ P ∧ ¬ Q) = false → (¬ P ∨ ¬ Q) = true → (¬ P ∨ Q) = true → (¬ P ∧ Q) = true :=
sorry

end false_proposition_l134_134522


namespace det_my_matrix_l134_134614

def my_matrix : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![3, 0, 1], ![-5, 5, -4], ![3, 3, 6]]

theorem det_my_matrix : my_matrix.det = 96 := by
  sorry

end det_my_matrix_l134_134614


namespace maria_earnings_l134_134392

def cost_of_brushes : ℕ := 20
def cost_of_canvas : ℕ := 3 * cost_of_brushes
def cost_per_liter_of_paint : ℕ := 8
def liters_of_paint : ℕ := 5
def cost_of_paint : ℕ := liters_of_paint * cost_per_liter_of_paint
def total_cost : ℕ := cost_of_brushes + cost_of_canvas + cost_of_paint
def selling_price : ℕ := 200

theorem maria_earnings : (selling_price - total_cost) = 80 := by
  sorry

end maria_earnings_l134_134392


namespace volleyball_team_selection_l134_134459

noncomputable def volleyball_squad_count (n m k : ℕ) : ℕ :=
  n * (Nat.choose m k)

theorem volleyball_team_selection :
  volleyball_squad_count 12 11 7 = 3960 :=
by
  sorry

end volleyball_team_selection_l134_134459


namespace total_potatoes_l134_134189

theorem total_potatoes (jane_potatoes mom_potatoes dad_potatoes : Nat) 
  (h1 : jane_potatoes = 8)
  (h2 : mom_potatoes = 8)
  (h3 : dad_potatoes = 8) :
  jane_potatoes + mom_potatoes + dad_potatoes = 24 :=
by
  sorry

end total_potatoes_l134_134189


namespace maria_needs_flour_l134_134878

-- Definitions from conditions
def cups_of_flour_per_cookie (c : ℕ) (f : ℚ) : ℚ := f / c

def total_cups_of_flour (cps_per_cookie : ℚ) (num_cookies : ℕ) : ℚ := cps_per_cookie * num_cookies

-- Given values
def cookies_20 := 20
def flour_3 := 3
def cookies_100 := 100

theorem maria_needs_flour :
  total_cups_of_flour (cups_of_flour_per_cookie cookies_20 flour_3) cookies_100 = 15 :=
by
  sorry -- Proof is omitted

end maria_needs_flour_l134_134878


namespace point_in_second_quadrant_l134_134478

def P : ℝ × ℝ := (-5, 4)

theorem point_in_second_quadrant (p : ℝ × ℝ) (hx : p.1 = -5) (hy : p.2 = 4) : p.1 < 0 ∧ p.2 > 0 :=
by
  sorry

example : P.1 < 0 ∧ P.2 > 0 :=
  point_in_second_quadrant P rfl rfl

end point_in_second_quadrant_l134_134478


namespace y_increase_by_18_when_x_increases_by_12_l134_134336

theorem y_increase_by_18_when_x_increases_by_12
  (h_slope : ∀ x y: ℝ, (4 * y = 6 * x) ↔ (3 * y = 2 * x)) :
  ∀ Δx : ℝ, Δx = 12 → ∃ Δy : ℝ, Δy = 18 :=
by
  sorry

end y_increase_by_18_when_x_increases_by_12_l134_134336


namespace binom_13_10_eq_286_l134_134725

noncomputable def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binom_13_10_eq_286 : binomial 13 10 = 286 := by
  sorry

end binom_13_10_eq_286_l134_134725


namespace rationalize_denominator_l134_134804

theorem rationalize_denominator 
  (cbrt32_eq_2cbrt4 : (32:ℝ)^(1/3) = 2 * (4:ℝ)^(1/3))
  (cbrt16_eq_2cbrt2 : (16:ℝ)^(1/3) = 2 * (2:ℝ)^(1/3))
  (cbrt64_eq_4 : (64:ℝ)^(1/3) = 4) :
  1 / ((4:ℝ)^(1/3) + (32:ℝ)^(1/3)) = ((2:ℝ)^(1/3)) / 6 :=
  sorry

end rationalize_denominator_l134_134804


namespace area_of_square_II_l134_134956

theorem area_of_square_II {a b : ℝ} (h : a > b) (d : ℝ) (h1 : d = a - b)
    (A1_A : ℝ) (h2 : A1_A = (a - b)^2 / 2) (A2_A : ℝ) (h3 : A2_A = 3 * A1_A) :
  A2_A = 3 * (a - b)^2 / 2 := by
  sorry

end area_of_square_II_l134_134956


namespace problem_part1_problem_part2_l134_134532

noncomputable def f (a : ℝ) (x : ℝ) := 2 * Real.log x + a / x
noncomputable def g (a : ℝ) (x : ℝ) := (x / 2) * f a x - a * x^2 - x

theorem problem_part1 (a : ℝ) :
  (∀ x : ℝ, f a x = 0 → x > 0) ↔ 0 < a ∧ a < 2/Real.exp 1 := sorry

theorem problem_part2 (a : ℝ) (x₁ x₂ : ℝ) (h₁ : x₁ < x₂) (h₂ : g a x₁ = 0) (h₃ : g a x₂ = 0) :
  0 < a ∧ a < 2/Real.exp 1 → Real.log x₁ + 2 * Real.log x₂ > 3 := sorry

end problem_part1_problem_part2_l134_134532


namespace hillary_descending_rate_l134_134060

def baseCampDistance : ℕ := 4700
def hillaryClimbingRate : ℕ := 800
def eddyClimbingRate : ℕ := 500
def hillaryStopShort : ℕ := 700
def departTime : ℕ := 6 -- time is represented in hours from midnight
def passTime : ℕ := 12 -- time is represented in hours from midnight

theorem hillary_descending_rate :
  ∃ r : ℕ, r = 1000 := by
  sorry

end hillary_descending_rate_l134_134060


namespace seeds_in_bucket_C_l134_134624

theorem seeds_in_bucket_C (A B C : ℕ) (h1 : A + B + C = 100) (h2 : A = B + 10) (h3 : B = 30) : C = 30 :=
by
  -- Placeholder for the actual proof
  sorry

end seeds_in_bucket_C_l134_134624


namespace no_integer_solutions_l134_134757

theorem no_integer_solutions :
  ¬ ∃ (x y z : ℤ),
    x^6 + x^3 + x^3 * y + y = 147^157 ∧
    x^3 + x^3 * y + y^2 + y + z^9 = 157^147 :=
by
  sorry

end no_integer_solutions_l134_134757


namespace find_divisor_l134_134458

-- Condition Definitions
def dividend : ℕ := 725
def quotient : ℕ := 20
def remainder : ℕ := 5

-- Target Proof Statement
theorem find_divisor (divisor : ℕ) (h : dividend = divisor * quotient + remainder) : divisor = 36 := by
  sorry

end find_divisor_l134_134458


namespace number_of_green_fish_and_carp_drawn_is_6_l134_134652

-- Definitions/parameters from the problem
def total_fish := 80 + 20 + 40 + 40 + 20
def sample_size := 20
def number_of_green_fish := 20
def number_of_carp := 40
def probability_of_being_drawn := sample_size / total_fish

-- Theorem to prove the combined number of green fish and carp drawn is 6
theorem number_of_green_fish_and_carp_drawn_is_6 :
  (number_of_green_fish + number_of_carp) * probability_of_being_drawn = 6 := 
by {
  -- Placeholder for the actual proof
  sorry
}

end number_of_green_fish_and_carp_drawn_is_6_l134_134652


namespace relationship_between_p_and_q_l134_134442

theorem relationship_between_p_and_q (p q : ℝ) 
  (h : ∃ x : ℝ, (x^2 + p*x + q = 0) ∧ (2*x)^2 + p*(2*x) + q = 0) :
  2 * p^2 = 9 * q :=
sorry

end relationship_between_p_and_q_l134_134442


namespace annuity_payment_l134_134701

variable (P : ℝ) (A : ℝ) (i : ℝ) (n1 n2 : ℕ)

-- Condition: Principal amount
axiom principal_amount : P = 24000

-- Condition: Annual installment for the first 5 years
axiom annual_installment : A = 1500 

-- Condition: Annual interest rate
axiom interest_rate : i = 0.045 

-- Condition: Years before equal annual installments
axiom years_before_installment : n1 = 5 

-- Condition: Years for repayment after the first 5 years
axiom repayment_years : n2 = 7 

-- Remaining debt after n1 years
noncomputable def remaining_debt_after_n1 : ℝ :=
  P * (1 + i) ^ n1 - A * ((1 + i) ^ n1 - 1) / i

-- Annual payment for n2 years to repay the remaining debt
noncomputable def annual_payment (D : ℝ) : ℝ :=
  D * (1 + i) ^ n2 / (((1 + i) ^ n2 - 1) / i)

axiom remaining_debt_amount : remaining_debt_after_n1 P A i n1 = 21698.685 

theorem annuity_payment : annual_payment (remaining_debt_after_n1 P A i n1) = 3582 := by
  sorry

end annuity_payment_l134_134701


namespace molly_age_l134_134299

theorem molly_age : 14 + 6 = 20 := by
  sorry

end molly_age_l134_134299


namespace george_elaine_ratio_l134_134544

-- Define the conditions
def time_jerry := 3
def time_elaine := 2 * time_jerry
def time_kramer := 0
def total_time := 11

-- Define George's time based on the given total time condition
def time_george := total_time - (time_jerry + time_elaine + time_kramer)

-- Prove the ratio of George's time to Elaine's time is 1:3
theorem george_elaine_ratio : time_george / time_elaine = 1 / 3 :=
by
  -- Lean proof would go here
  sorry

end george_elaine_ratio_l134_134544


namespace sandbox_area_l134_134363

def sandbox_length : ℕ := 312
def sandbox_width : ℕ := 146

theorem sandbox_area : sandbox_length * sandbox_width = 45552 := by
  sorry

end sandbox_area_l134_134363


namespace hillary_descending_rate_correct_l134_134749

-- Define the conditions in Lean
def base_to_summit := 5000 -- height from base camp to the summit
def departure_time := 6 -- departure time in hours after midnight (6:00)
def summit_time_hillary := 5 -- time taken by Hillary to reach 1000 ft short of the summit
def passing_time := 12 -- time when Hillary and Eddy pass each other (12:00)
def climb_rate_hillary := 800 -- Hillary's climbing rate in ft/hr
def climb_rate_eddy := 500 -- Eddy's climbing rate in ft/hr
def stop_short := 1000 -- distance short of the summit Hillary stops at

-- Define the correct answer based on the conditions
def descending_rate_hillary := 1000 -- Hillary's descending rate in ft/hr

-- Create the theorem to prove Hillary's descending rate
theorem hillary_descending_rate_correct (base_to_summit departure_time summit_time_hillary passing_time climb_rate_hillary climb_rate_eddy stop_short descending_rate_hillary : ℕ) :
  (descending_rate_hillary = 1000) :=
sorry

end hillary_descending_rate_correct_l134_134749


namespace tamia_bell_pepper_pieces_l134_134399

theorem tamia_bell_pepper_pieces :
  let bell_peppers := 5
  let slices_per_pepper := 20
  let initial_slices := bell_peppers * slices_per_pepper
  let half_slices_cut := initial_slices / 2
  let small_pieces := half_slices_cut * 3
  let total_pieces := (initial_slices - half_slices_cut) + small_pieces
  total_pieces = 200 :=
by
  sorry

end tamia_bell_pepper_pieces_l134_134399


namespace product_of_two_numbers_l134_134346

theorem product_of_two_numbers (a b : ℝ) (h1 : a + b = 70) (h2 : a - b = 10) : a * b = 1200 := 
by
  sorry

end product_of_two_numbers_l134_134346


namespace average_age_choir_l134_134048

theorem average_age_choir (S_f S_m S_total : ℕ) (avg_f : ℕ) (avg_m : ℕ) (females males total : ℕ)
  (h1 : females = 8) (h2 : males = 12) (h3 : total = 20)
  (h4 : avg_f = 25) (h5 : avg_m = 40)
  (h6 : S_f = avg_f * females) 
  (h7 : S_m = avg_m * males) 
  (h8 : S_total = S_f + S_m) :
  (S_total / total) = 34 := by
  sorry

end average_age_choir_l134_134048


namespace two_digit_number_l134_134436

theorem two_digit_number (x y : Nat) : 
  10 * x + y = 10 * x + y := 
by 
  sorry

end two_digit_number_l134_134436


namespace article_usage_correct_l134_134170

def blank1 := "a"
def blank2 := ""  -- Representing "不填" (no article) as an empty string for simplicity

theorem article_usage_correct :
  (blank1 = "a" ∧ blank2 = "") :=
by
  sorry

end article_usage_correct_l134_134170


namespace add_to_fraction_l134_134833

theorem add_to_fraction (x : ℕ) :
  (3 + x) / (11 + x) = 5 / 9 ↔ x = 7 :=
by
  sorry

end add_to_fraction_l134_134833


namespace base_n_system_digits_l134_134206

theorem base_n_system_digits (N : ℕ) (h : N ≥ 6) :
  ((N - 1) ^ 4).digits N = [N-4, 5, N-4, 1] :=
by
  sorry

end base_n_system_digits_l134_134206


namespace football_basketball_problem_l134_134068

theorem football_basketball_problem :
  ∃ (football_cost basketball_cost : ℕ),
    (3 * football_cost + basketball_cost = 230) ∧
    (2 * football_cost + 3 * basketball_cost = 340) ∧
    football_cost = 50 ∧
    basketball_cost = 80 ∧
    ∃ (basketballs footballs : ℕ),
      (basketballs + footballs = 20) ∧
      (footballs < basketballs) ∧
      (80 * basketballs + 50 * footballs ≤ 1400) ∧
      ((basketballs = 11 ∧ footballs = 9) ∨
       (basketballs = 12 ∧ footballs = 8) ∨
       (basketballs = 13 ∧ footballs = 7)) :=
by
  sorry

end football_basketball_problem_l134_134068


namespace transportable_load_l134_134100

theorem transportable_load 
  (mass_of_load : ℝ) 
  (num_boxes : ℕ) 
  (box_capacity : ℝ) 
  (num_trucks : ℕ) 
  (truck_capacity : ℝ) 
  (h1 : mass_of_load = 13.5) 
  (h2 : box_capacity = 0.35) 
  (h3 : truck_capacity = 1.5) 
  (h4 : num_trucks = 11)
  (boxes_condition : ∀ (n : ℕ), n * box_capacity ≥ mass_of_load) :
  mass_of_load ≤ num_trucks * truck_capacity :=
by
  sorry

end transportable_load_l134_134100


namespace polygon_sides_l134_134928

theorem polygon_sides (n : ℕ) :
  (n - 2) * 180 = 3 * 360 - 180 → n = 5 := by
  intro h
  sorry

end polygon_sides_l134_134928


namespace shifted_parabola_is_correct_l134_134403

-- Define the initial parabola
def initial_parabola (x : ℝ) : ℝ :=
  -((x - 1) ^ 2) + 2

-- Define the shifted parabola
def shifted_parabola (x : ℝ) : ℝ :=
  -((x + 1 - 1) ^ 2) + 4

-- State the theorem
theorem shifted_parabola_is_correct :
  ∀ x : ℝ, shifted_parabola x = -x^2 + 4 :=
by
  -- Proof would go here
  sorry

end shifted_parabola_is_correct_l134_134403


namespace sum_of_s_and_t_eq_neg11_l134_134907

theorem sum_of_s_and_t_eq_neg11 (s t : ℝ) 
  (h1 : ∀ x, x = 3 → x^2 + s * x + t = 0)
  (h2 : ∀ x, x = -4 → x^2 + s * x + t = 0) :
  s + t = -11 :=
sorry

end sum_of_s_and_t_eq_neg11_l134_134907


namespace blanket_cost_l134_134921

theorem blanket_cost (x : ℝ) 
    (h₁ : 200 + 750 + 2 * x = 1350) 
    (h₂ : 2 + 5 + 2 = 9) 
    (h₃ : (200 + 750 + 2 * x) / 9 = 150) : 
    x = 200 :=
by
    have h_total : 200 + 750 + 2 * x = 1350 := h₁
    have h_avg : (200 + 750 + 2 * x) / 9 = 150 := h₃
    sorry

end blanket_cost_l134_134921


namespace technicians_count_l134_134129

-- Variables
variables (T R : ℕ)
-- Conditions from the problem
def avg_salary_all := 8000
def avg_salary_tech := 12000
def avg_salary_rest := 6000
def total_workers := 30
def total_salary := avg_salary_all * total_workers

-- Equations based on conditions
def eq1 : T + R = total_workers := sorry
def eq2 : avg_salary_tech * T + avg_salary_rest * R = total_salary := sorry

-- Proof statement (external conditions are reused for clarity)
theorem technicians_count : T = 10 :=
by sorry

end technicians_count_l134_134129


namespace ratio_fourth_to_third_l134_134402

theorem ratio_fourth_to_third (third_graders fifth_graders fourth_graders : ℕ) (H1 : third_graders = 20) (H2 : fifth_graders = third_graders / 2) (H3 : third_graders + fifth_graders + fourth_graders = 70) : fourth_graders / third_graders = 2 := by
  sorry

end ratio_fourth_to_third_l134_134402


namespace evaluate_expression_l134_134879

theorem evaluate_expression (x y : ℤ) (h₁ : x = 3) (h₂ : y = 4) : 
  (x^4 + 3 * x^2 - 2 * y + 2 * y^2) / 6 = 22 :=
by
  -- Conditions from the problem
  rw [h₁, h₂]
  -- Sorry is used to skip the proof
  sorry

end evaluate_expression_l134_134879


namespace compare_neg_two_and_neg_one_l134_134554

theorem compare_neg_two_and_neg_one : -2 < -1 :=
by {
  -- Proof is omitted
  sorry
}

end compare_neg_two_and_neg_one_l134_134554


namespace social_media_usage_in_week_l134_134178

def days_in_week : ℕ := 7
def daily_phone_usage : ℕ := 16
def daily_social_media_usage : ℕ := daily_phone_usage / 2

theorem social_media_usage_in_week :
  daily_social_media_usage * days_in_week = 56 :=
by
  sorry

end social_media_usage_in_week_l134_134178


namespace arithmetic_sequence_general_formula_l134_134818

theorem arithmetic_sequence_general_formula
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n : ℕ, S n = n * (a 1 + a n) / 2)
  (h2 : a 4 - a 2 = 4)
  (h3 : S 3 = 9)
  : ∀ n : ℕ, a n = 2 * n - 1 := 
sorry

end arithmetic_sequence_general_formula_l134_134818


namespace diet_equivalence_l134_134148

variable (B E L D A : ℕ)

theorem diet_equivalence :
  (17 * B = 170 * L) →
  (100000 * A = 50 * L) →
  (10 * B = 4 * E) →
  12 * E = 600000 * A :=
sorry

end diet_equivalence_l134_134148


namespace tangent_line_at_1_1_l134_134335

noncomputable def f (x : ℝ) : ℝ := x / (2 * x - 1)

theorem tangent_line_at_1_1 :
  let m := -((2 * 1 - 1 - 2 * 1) / (2 * 1 - 1)^2) -- Derivative evaluated at x = 1
  let tangent_line (x y : ℝ) := x + y - 2
  ∀ x y : ℝ, tangent_line x y = 0 → (f x = y ∧ x = 1 → y = 1 → m = -1) :=
by
  sorry

end tangent_line_at_1_1_l134_134335


namespace least_three_digit_with_product_l134_134916

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digits_product (n : ℕ) (p : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n % 100) / 10
  let d3 := n % 10
  d1 * d2 * d3 = p

theorem least_three_digit_with_product (p : ℕ) : ∃ n : ℕ, is_three_digit n ∧ digits_product n p ∧ 
  ∀ m : ℕ, is_three_digit m ∧ digits_product m p → n ≤ m :=
by
  use 116
  sorry

end least_three_digit_with_product_l134_134916


namespace intersection_range_l134_134911

noncomputable def f (a : ℝ) (x : ℝ) := a * x
noncomputable def g (x : ℝ) := Real.log x
noncomputable def F (a : ℝ) (x : ℝ) := f a x - g x

theorem intersection_range (a : ℝ) :
  (∃ x1 x2 : ℝ, 0 < x1 ∧ 0 < x2 ∧ x1 ≠ x2 ∧ f a x1 = g x1 ∧ f a x2 = g x2) ↔
  0 < a ∧ a < 1 / Real.exp 1 := by
  sorry

end intersection_range_l134_134911


namespace positive_integers_solution_l134_134105

open Nat

theorem positive_integers_solution (a b m n : ℕ) (r : ℕ) (h_pos_a : 0 < a)
  (h_pos_b : 0 < b) (h_pos_m : 0 < m) (h_pos_n : 0 < n) 
  (h_gcd : Nat.gcd m n = 1) :
  (a^2 + b^2)^m = (a * b)^n ↔ a = 2^r ∧ b = 2^r ∧ m = 2 * r ∧ n = 2 * r + 1 :=
sorry

end positive_integers_solution_l134_134105


namespace find_nonzero_q_for_quadratic_l134_134825

theorem find_nonzero_q_for_quadratic :
  ∃ (q : ℝ), q ≠ 0 ∧ (∀ (x1 x2 : ℝ), (q * x1^2 - 8 * x1 + 2 = 0 ∧ q * x2^2 - 8 * x2 + 2 = 0) → x1 = x2) ↔ q = 8 :=
by
  sorry

end find_nonzero_q_for_quadratic_l134_134825


namespace calculate_selling_price_l134_134484

noncomputable def originalPrice : ℝ := 120
noncomputable def firstDiscountRate : ℝ := 0.30
noncomputable def secondDiscountRate : ℝ := 0.15
noncomputable def taxRate : ℝ := 0.08

def discountedPrice1 (originalPrice firstDiscountRate : ℝ) : ℝ :=
  originalPrice * (1 - firstDiscountRate)

def discountedPrice2 (discountedPrice1 secondDiscountRate : ℝ) : ℝ :=
  discountedPrice1 * (1 - secondDiscountRate)

def finalPrice (discountedPrice2 taxRate : ℝ) : ℝ :=
  discountedPrice2 * (1 + taxRate)

theorem calculate_selling_price : 
  finalPrice (discountedPrice2 (discountedPrice1 originalPrice firstDiscountRate) secondDiscountRate) taxRate = 77.112 := 
sorry

end calculate_selling_price_l134_134484


namespace area_of_fourth_square_l134_134997

open Real

theorem area_of_fourth_square
  (EF FG GH : ℝ)
  (hEF : EF = 5)
  (hFG : FG = 7)
  (hGH : GH = 8) :
  let EG := sqrt (EF^2 + FG^2)
  let EH := sqrt (EG^2 + GH^2)
  EH^2 = 138 :=
by
  sorry

end area_of_fourth_square_l134_134997


namespace set_inter_complement_l134_134360

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

theorem set_inter_complement :
  U = {1, 2, 3, 4, 5, 6, 7} ∧ A = {1, 2, 3, 4} ∧ B = {3, 5, 6} →
  A ∩ (U \ B) = {1, 2, 4} :=
by
  sorry

end set_inter_complement_l134_134360


namespace shortest_remaining_side_l134_134463

theorem shortest_remaining_side (a b c : ℝ) (h₁ : a = 5) (h₂ : c = 13) (h₃ : a^2 + b^2 = c^2) : b = 12 :=
by
  rw [h₁, h₂] at h₃
  sorry

end shortest_remaining_side_l134_134463


namespace isosceles_triangle_construction_l134_134381

noncomputable def isosceles_triangle_construction_impossible 
  (hb lb : ℝ) : Prop :=
  ∀ (α β : ℝ), 
  3 * β ≠ α

theorem isosceles_triangle_construction : 
  ∃ (hb lb : ℝ), isosceles_triangle_construction_impossible hb lb :=
sorry

end isosceles_triangle_construction_l134_134381


namespace proof_no_natural_solutions_l134_134432

noncomputable def no_natural_solutions : Prop :=
  ∀ x y : ℕ, y^2 ≠ x^2 + x + 1

theorem proof_no_natural_solutions : no_natural_solutions :=
by
  intros x y
  sorry

end proof_no_natural_solutions_l134_134432


namespace simplify_expression_l134_134179

theorem simplify_expression : Real.sqrt (25 * Real.sqrt (15 * Real.sqrt 9)) = 5 * Real.sqrt 15 :=
by
  sorry

end simplify_expression_l134_134179


namespace tony_running_speed_l134_134882

theorem tony_running_speed :
  (∀ R : ℝ, (4 / 2 * 60) + 2 * ((4 / R) * 60) = 168 → R = 10) :=
sorry

end tony_running_speed_l134_134882


namespace value_of_f5_and_f_neg5_l134_134590

noncomputable def f (a b c x : ℝ) : ℝ := a * x^7 - b * x^5 + c * x^3 + 2

theorem value_of_f5_and_f_neg5 (a b c : ℝ) (m : ℝ) (h : f a b c (-5) = m) :
  f a b c 5 + f a b c (-5) = 4 :=
sorry

end value_of_f5_and_f_neg5_l134_134590


namespace canteen_consumption_l134_134295

theorem canteen_consumption :
  ∀ (x : ℕ),
    (x + (500 - x) + (200 - x)) = 700 → 
    (500 - x) = 7 * (200 - x) →
    x = 150 :=
by
  sorry

end canteen_consumption_l134_134295


namespace fraction_irreducible_iff_l134_134259

-- Define the condition for natural number n
def is_natural (n : ℕ) : Prop :=
  True  -- All undergraduate natural numbers abide to True

-- Main theorem formalized in Lean 4
theorem fraction_irreducible_iff (n : ℕ) :
  (∃ (g : ℕ), g = 1 ∧ (∃ a b : ℕ, 2 * n * n + 11 * n - 18 = a * g ∧ n + 7 = b * g)) ↔ 
  (n % 3 = 0 ∨ n % 3 = 1) :=
by sorry

end fraction_irreducible_iff_l134_134259


namespace circle_properties_l134_134627

noncomputable def pi : Real := 3.14
variable (C : Real) (diameter : Real) (radius : Real) (area : Real)

theorem circle_properties (h₀ : C = 31.4) :
  radius = C / (2 * pi) ∧
  diameter = 2 * radius ∧
  area = pi * radius^2 ∧
  radius = 5 ∧
  diameter = 10 ∧
  area = 78.5 :=
by
  sorry

end circle_properties_l134_134627


namespace concrete_volume_is_six_l134_134853

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

end concrete_volume_is_six_l134_134853


namespace pauline_convertibles_l134_134220

theorem pauline_convertibles : 
  ∀ (total_cars regular_percentage truck_percentage sedan_percentage sports_percentage suv_percentage : ℕ),
  total_cars = 125 →
  regular_percentage = 38 →
  truck_percentage = 12 →
  sedan_percentage = 17 →
  sports_percentage = 22 →
  suv_percentage = 6 →
  (total_cars - (regular_percentage * total_cars / 100 + truck_percentage * total_cars / 100 + sedan_percentage * total_cars / 100 + sports_percentage * total_cars / 100 + suv_percentage * total_cars / 100)) = 8 :=
by
  intros
  sorry

end pauline_convertibles_l134_134220


namespace determine_f_function_l134_134750

variable (f : ℝ → ℝ)

theorem determine_f_function (x : ℝ) (h : f (1 - x) = 1 + x) : f x = 2 - x := 
sorry

end determine_f_function_l134_134750


namespace set_D_cannot_form_triangle_l134_134913

theorem set_D_cannot_form_triangle : ¬ (∃ a b c : ℝ, a = 2 ∧ b = 4 ∧ c = 6 ∧ 
  (a + b > c ∧ a + c > b ∧ b + c > a)) :=
by {
  sorry
}

end set_D_cannot_form_triangle_l134_134913


namespace evaluate_expression_l134_134345

theorem evaluate_expression : (4^150 * 9^152) / 6^301 = 27 / 2 := 
by 
  -- skipping the actual proof
  sorry

end evaluate_expression_l134_134345


namespace discount_policy_l134_134593

-- Define the prices of the fruits
def lemon_price := 2
def papaya_price := 1
def mango_price := 4

-- Define the quantities Tom buys
def lemons_bought := 6
def papayas_bought := 4
def mangos_bought := 2

-- Define the total amount paid by Tom
def amount_paid := 21

-- Define the total number of fruits bought
def total_fruits_bought := lemons_bought + papayas_bought + mangos_bought

-- Define the total cost without discount
def total_cost_without_discount := 
  (lemons_bought * lemon_price) + 
  (papayas_bought * papaya_price) + 
  (mangos_bought * mango_price)

-- Calculate the discount
def discount := total_cost_without_discount - amount_paid

-- The discount policy
theorem discount_policy : discount = 3 ∧ total_fruits_bought = 12 :=
by 
  sorry

end discount_policy_l134_134593


namespace distance_walked_hazel_l134_134935

theorem distance_walked_hazel (x : ℝ) (h : x + 2 * x = 6) : x = 2 :=
sorry

end distance_walked_hazel_l134_134935


namespace base_four_to_base_ten_of_20314_eq_568_l134_134552

-- Define what it means to convert a base-four number to base-ten
def base_four_to_base_ten (digits : List ℕ) : ℕ :=
  digits.reverse.enum.foldr (λ ⟨index, digit⟩ acc => acc + digit * 4^index) 0

-- Define the specific base-four number 20314_4 as a list of its digits
def num_20314_base_four : List ℕ := [2, 0, 3, 1, 4]

-- Theorem stating that the base-ten equivalent of 20314_4 is 568
theorem base_four_to_base_ten_of_20314_eq_568 : base_four_to_base_ten num_20314_base_four = 568 := sorry

end base_four_to_base_ten_of_20314_eq_568_l134_134552


namespace min_value_expr_l134_134275

theorem min_value_expr (x : ℝ) (h : x > -3) : ∃ m, (∀ y > -3, 2 * y + (1 / (y + 3)) ≥ m) ∧ m = 2 * Real.sqrt 2 - 6 :=
by
  sorry

end min_value_expr_l134_134275


namespace number_of_divisors_that_are_multiples_of_2_l134_134241

-- Define the prime factorization of 540
def prime_factorization_540 : ℕ × ℕ × ℕ := (2, 3, 5)

-- Define the constraints for a divisor to be a multiple of 2
def valid_divisor_form (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 3 ∧ 0 ≤ c ∧ c ≤ 1

noncomputable def count_divisors (prime_info : ℕ × ℕ × ℕ) : ℕ :=
  let (p1, p2, p3) := prime_info
  2 * 4 * 2 -- Correspond to choices for \( a \), \( b \), and \( c \)

theorem number_of_divisors_that_are_multiples_of_2 (p1 p2 p3 : ℕ) (h : prime_factorization_540 = (p1, p2, p3)) :
  ∃ (count : ℕ), count = 16 :=
by
  use count_divisors (2, 3, 5)
  sorry

end number_of_divisors_that_are_multiples_of_2_l134_134241


namespace grazing_time_for_36_cows_l134_134398

-- Defining the problem conditions and the question in Lean 4
theorem grazing_time_for_36_cows :
  ∀ (g r b : ℕ), 
    (24 * 6 * b = g + 6 * r) →
    (21 * 8 * b = g + 8 * r) →
    36 * 3 * b = g + 3 * r :=
by
  intros
  sorry

end grazing_time_for_36_cows_l134_134398


namespace problem_l134_134753

variable (a b : ℝ)

theorem problem (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = 4) : a^2 + b^2 ≥ 8 := 
sorry

end problem_l134_134753


namespace number_of_rows_is_ten_l134_134293

-- Definition of the arithmetic sequence
def arithmetic_sequence_sum (n : ℕ) : ℕ :=
  n * (3 * n + 1) / 2

-- The main theorem to prove
theorem number_of_rows_is_ten :
  (∃ n : ℕ, arithmetic_sequence_sum n = 145) ↔ n = 10 :=
by
  sorry

end number_of_rows_is_ten_l134_134293


namespace square_roots_of_16_l134_134727

theorem square_roots_of_16 :
  {y : ℤ | y^2 = 16} = {4, -4} :=
by
  sorry

end square_roots_of_16_l134_134727


namespace arithmetic_seq_sum_l134_134892

theorem arithmetic_seq_sum (a d : ℕ) (S : ℕ → ℕ) (n : ℕ) :
  S 6 = 36 →
  S 12 = 144 →
  S (6 * n) = 576 →
  (∀ m, S m = m * (2 * a + (m - 1) * d) / 2) →
  n = 4 := 
by
  intros h1 h2 h3 h4
  sorry

end arithmetic_seq_sum_l134_134892


namespace triangle_third_side_length_l134_134008

theorem triangle_third_side_length (A B C : Type) 
  (AB : ℝ) (AC : ℝ) 
  (angle_ABC angle_ACB : ℝ) 
  (BC : ℝ) 
  (h1 : AB = 7) 
  (h2 : AC = 21) 
  (h3 : angle_ABC = 3 * angle_ACB) 
  : 
  BC = (some_correct_value ) := 
sorry

end triangle_third_side_length_l134_134008


namespace n_m_odd_implies_sum_odd_l134_134733

theorem n_m_odd_implies_sum_odd {n m : ℤ} (h : Odd (n^2 + m^2)) : Odd (n + m) :=
by
  sorry

end n_m_odd_implies_sum_odd_l134_134733


namespace number_of_students_without_A_l134_134886

theorem number_of_students_without_A (total_students : ℕ) (A_chemistry : ℕ) (A_physics : ℕ) (A_both : ℕ) (h1 : total_students = 40)
    (h2 : A_chemistry = 10) (h3 : A_physics = 18) (h4 : A_both = 5) :
    total_students - (A_chemistry + A_physics - A_both) = 17 :=
by {
  sorry
}

end number_of_students_without_A_l134_134886


namespace arithmetic_sequence_75th_term_l134_134039

theorem arithmetic_sequence_75th_term (a d n : ℕ) (h1 : a = 2) (h2 : d = 4) (h3 : n = 75) : 
  a + (n - 1) * d = 298 :=
by 
  sorry

end arithmetic_sequence_75th_term_l134_134039


namespace joe_total_toy_cars_l134_134300

def joe_toy_cars (initial_cars additional_cars : ℕ) : ℕ :=
  initial_cars + additional_cars

theorem joe_total_toy_cars : joe_toy_cars 500 120 = 620 := by
  sorry

end joe_total_toy_cars_l134_134300


namespace compare_abc_l134_134566

noncomputable def a : ℝ := Real.log 0.3 / Real.log 2
noncomputable def b : ℝ := Real.exp (0.3 * Real.log 2)
noncomputable def c : ℝ := Real.exp (0.2 * Real.log 0.3)

theorem compare_abc : a < c ∧ c < b :=
by
  -- The proof will be provided here.
  sorry

end compare_abc_l134_134566


namespace calculate_div_expression_l134_134724

variable (x y : ℝ)

theorem calculate_div_expression : (6 * x^3 * y^2) / (-3 * x * y) = -2 * x^2 * y := by
  sorry

end calculate_div_expression_l134_134724


namespace sequence_general_formula_l134_134101

theorem sequence_general_formula (a : ℕ → ℕ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n, a (n + 1) = a n + 2 * n) :
  ∀ n, a n = n^2 - n + 1 :=
by sorry

end sequence_general_formula_l134_134101


namespace max_distance_convoy_l134_134079

structure Vehicle :=
  (mpg : ℝ) (min_gallons : ℝ)

def SUV : Vehicle := ⟨12.2, 10⟩
def Sedan : Vehicle := ⟨52, 5⟩
def Motorcycle : Vehicle := ⟨70, 2⟩

def total_gallons : ℝ := 21

def total_distance (SUV_gallons Sedan_gallons Motorcycle_gallons : ℝ) : ℝ :=
  SUV.mpg * SUV_gallons + Sedan.mpg * Sedan_gallons + Motorcycle.mpg * Motorcycle_gallons

theorem max_distance_convoy (SUV_gallons Sedan_gallons Motorcycle_gallons : ℝ) :
  SUV_gallons + Sedan_gallons + Motorcycle_gallons = total_gallons →
  SUV_gallons >= SUV.min_gallons →
  Sedan_gallons >= Sedan.min_gallons →
  Motorcycle_gallons >= Motorcycle.min_gallons →
  total_distance SUV_gallons Sedan_gallons Motorcycle_gallons = 802 :=
sorry

end max_distance_convoy_l134_134079


namespace maximum_a3_S10_l134_134134

-- Given definitions of the problem
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def conditions (a : ℕ → ℝ) : Prop :=
  is_arithmetic_sequence a ∧ (∀ n, a n > 0) ∧ (a 1 + a 3 + a 8 = a 4 ^ 2)

-- The problem statement
theorem maximum_a3_S10 (a : ℕ → ℝ) (h : conditions a) : 
  (∃ S : ℝ, S = a 3 * ((10 / 2) * (a 1 + a 10)) ∧ S ≤ 375 / 4) :=
sorry

end maximum_a3_S10_l134_134134


namespace steps_in_staircase_l134_134610

theorem steps_in_staircase (h1 : 120 / 20 = 6) (h2 : 180 / 6 = 30) : 
  ∃ n : ℕ, n = 30 :=
by
  -- the proof is omitted
  sorry

end steps_in_staircase_l134_134610


namespace parabola_focus_l134_134777

theorem parabola_focus : 
  ∀ x y : ℝ, y = - (1 / 16) * x^2 → ∃ f : ℝ × ℝ, f = (0, -4) := 
by
  sorry

end parabola_focus_l134_134777


namespace solve_for_x_l134_134086

theorem solve_for_x (x : ℝ) : 45 - 5 = 3 * x + 10 → x = 10 :=
by
  sorry

end solve_for_x_l134_134086


namespace initial_markup_percentage_l134_134735

theorem initial_markup_percentage (C : ℝ) (M : ℝ) 
  (h1 : ∀ S_1 : ℝ, S_1 = C * (1 + M))
  (h2 : ∀ S_2 : ℝ, S_2 = C * (1 + M) * 1.25)
  (h3 : ∀ S_3 : ℝ, S_3 = C * (1 + M) * 1.25 * 0.94)
  (h4 : ∀ S_3 : ℝ, S_3 = C * 1.41) : 
  M = 0.2 :=
by
  sorry

end initial_markup_percentage_l134_134735


namespace find_coefficient_b_l134_134375

variable (a b c p : ℝ)

def parabola (x : ℝ) := a * x^2 + b * x + c

theorem find_coefficient_b (h_vertex : ∀ x, parabola a b c x = a * (x - p)^2 + p)
                           (h_y_intercept : parabola a b c 0 = -3 * p)
                           (hp_nonzero : p ≠ 0) :
  b = 8 / p :=
by
  sorry

end find_coefficient_b_l134_134375


namespace det_B_squared_minus_3B_l134_134781

open Matrix
open Real

variable {α : Type*} [Fintype α] {n : ℕ}
variable [DecidableEq α]

noncomputable def B : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![2, 4],
  ![1, 3]
]

theorem det_B_squared_minus_3B : det (B * B - 3 • B) = -8 := sorry

end det_B_squared_minus_3B_l134_134781


namespace quadratic_has_two_distinct_real_roots_l134_134343

theorem quadratic_has_two_distinct_real_roots :
  ∀ x : ℝ, ∃ r1 r2 : ℝ, r1 ≠ r2 ∧ (x^2 - 2 * x - 6 = 0 ∧ x = r1 ∨ x = r2) :=
by sorry

end quadratic_has_two_distinct_real_roots_l134_134343


namespace intersection_M_N_l134_134173

def M : Set ℝ := { x | x^2 + x - 2 < 0 }
def N : Set ℝ := { x | 0 < x ∧ x ≤ 2 }

theorem intersection_M_N : M ∩ N = { x | 0 < x ∧ x < 1 } :=
by
  sorry

end intersection_M_N_l134_134173


namespace cos_4_arccos_l134_134747

theorem cos_4_arccos (y : ℝ) (hy1 : y = Real.arccos (2/5)) (hy2 : Real.cos y = 2/5) : 
  Real.cos (4 * y) = -47 / 625 := 
by 
  sorry

end cos_4_arccos_l134_134747


namespace decreasing_function_positive_l134_134543

variable {f : ℝ → ℝ}

axiom decreasing (h : ℝ → ℝ) : ∀ x1 x2, x1 < x2 → h x1 > h x2

theorem decreasing_function_positive (h_decreasing: ∀ x1 x2: ℝ, x1 < x2 → f x1 > f x2)
    (h_condition: ∀ x: ℝ, f x / (deriv^[2] f x) + x < 1) :
  ∀ x : ℝ, f x > 0 := 
by
  sorry

end decreasing_function_positive_l134_134543


namespace algebra_expression_value_l134_134499

theorem algebra_expression_value (m : ℝ) (hm : m^2 - m - 1 = 0) : m^2 - m + 2008 = 2009 :=
by
  sorry

end algebra_expression_value_l134_134499


namespace problem1_problem2_l134_134909

theorem problem1 :
  (-1)^2 + (Real.pi - 2022)^0 + 2 * Real.sin (60 * Real.pi / 180) - abs (1 - Real.sqrt 3) = 3 :=
by 
  sorry

theorem problem2 (x : ℝ) :
  (2 / (x + 1) + 1 = x / (x - 1)) → x = 3 :=
by 
  sorry

end problem1_problem2_l134_134909


namespace value_of_k_l134_134061

def f (x : ℝ) := 4 * x ^ 2 - 5 * x + 6
def g (x : ℝ) (k : ℝ) := 2 * x ^ 2 - k * x + 1

theorem value_of_k :
  (f 5 - g 5 k = 30) → k = -10 := 
by 
  sorry

end value_of_k_l134_134061


namespace ackermann_3_2_l134_134769

-- Define the Ackermann function
def ackermann : ℕ → ℕ → ℕ
| 0, n => n + 1
| (m + 1), 0 => ackermann m 1
| (m + 1), (n + 1) => ackermann m (ackermann (m + 1) n)

-- Prove that A(3, 2) = 29
theorem ackermann_3_2 : ackermann 3 2 = 29 := by
  sorry

end ackermann_3_2_l134_134769


namespace Freddy_journey_time_l134_134705

/-- Eddy and Freddy start simultaneously from city A. Eddy travels to city B, Freddy travels to city C.
    Eddy takes 3 hours from city A to city B, which is 900 km. The distance between city A and city C is
    300 km. The ratio of average speed of Eddy to Freddy is 4:1. Prove that Freddy takes 4 hours to travel. -/
theorem Freddy_journey_time (t_E : ℕ) (d_AB : ℕ) (d_AC : ℕ) (r : ℕ) (V_E V_F t_F : ℕ)
    (h1 : t_E = 3)
    (h2 : d_AB = 900)
    (h3 : d_AC = 300)
    (h4 : r = 4)
    (h5 : V_E = d_AB / t_E)
    (h6 : V_E = r * V_F)
    (h7 : t_F = d_AC / V_F)
  : t_F = 4 := 
  sorry

end Freddy_journey_time_l134_134705


namespace largest_multiple_of_15_less_than_500_l134_134006

-- Define the conditions
def is_multiple_of_15 (n : Nat) : Prop :=
  ∃ (k : Nat), n = 15 * k

def is_positive (n : Nat) : Prop :=
  n > 0

def is_less_than_500 (n : Nat) : Prop :=
  n < 500

-- Define the main theorem statement
theorem largest_multiple_of_15_less_than_500 :
  ∀ (n : Nat), is_multiple_of_15 n ∧ is_positive n ∧ is_less_than_500 n → n ≤ 495 :=
sorry

end largest_multiple_of_15_less_than_500_l134_134006


namespace find_nm_2023_l134_134334

theorem find_nm_2023 (n m : ℚ) (h : (n + 9)^2 + |m - 8| = 0) : (n + m) ^ 2023 = -1 := by
  sorry

end find_nm_2023_l134_134334


namespace rounding_increases_value_l134_134217

theorem rounding_increases_value (a b c d : ℕ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (rounded_a : ℕ := a + 1)
  (rounded_b : ℕ := b - 1)
  (rounded_c : ℕ := c + 1)
  (rounded_d : ℕ := d + 1) :
  (rounded_a * rounded_d) / rounded_b + rounded_c > (a * d) / b + c := 
sorry

end rounding_increases_value_l134_134217


namespace unoccupied_volume_correct_l134_134018

-- Define the conditions given in the problem
def tank_length := 12 -- inches
def tank_width := 8 -- inches
def tank_height := 10 -- inches
def water_fraction := 1 / 3
def ice_cube_side := 1 -- inches
def num_ice_cubes := 12

-- Calculate the occupied volume
noncomputable def tank_volume : ℝ := tank_length * tank_width * tank_height
noncomputable def water_volume : ℝ := tank_volume * water_fraction
noncomputable def ice_cube_volume : ℝ := ice_cube_side^3
noncomputable def total_ice_volume : ℝ := ice_cube_volume * num_ice_cubes
noncomputable def total_occupied_volume : ℝ := water_volume + total_ice_volume

-- Calculate the unoccupied volume
noncomputable def unoccupied_volume : ℝ := tank_volume - total_occupied_volume

-- State the problem
theorem unoccupied_volume_correct : unoccupied_volume = 628 := by
  sorry

end unoccupied_volume_correct_l134_134018


namespace find_phi_l134_134000

open Real

noncomputable def f (x φ : ℝ) : ℝ := cos (2 * x + φ)
noncomputable def g (x φ : ℝ) : ℝ := cos (2 * x - π/2 + φ)

theorem find_phi 
  (h1 : 0 < φ) 
  (h2 : φ < π) 
  (symmetry_condition : ∀ x, g (π/2 - x) φ = g (π/2 + x) φ) 
  : φ = π / 2 
:= by 
  sorry

end find_phi_l134_134000


namespace gcd_of_11121_and_12012_l134_134408

def gcd_problem : Prop :=
  gcd 11121 12012 = 1

theorem gcd_of_11121_and_12012 : gcd_problem :=
by
  -- Proof omitted
  sorry

end gcd_of_11121_and_12012_l134_134408


namespace count_inverses_modulo_11_l134_134665

theorem count_inverses_modulo_11 : (Finset.filter (λ a => Nat.gcd a 11 = 1) (Finset.range 11)).card = 10 :=
  by
  sorry

end count_inverses_modulo_11_l134_134665


namespace inequality_with_xy_l134_134645

theorem inequality_with_xy
  (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x * y = 4) :
  (1 / (x + 3)) + (1 / (y + 3)) ≤ 2 / 5 :=
sorry

end inequality_with_xy_l134_134645


namespace pqrs_sum_l134_134029

theorem pqrs_sum (p q r s : ℝ)
  (h1 : (∀ x : ℝ, x^2 - 12*p*x - 13*q = 0 → x = r ∨ x = s))
  (h2 : (∀ x : ℝ, x^2 - 12*r*x - 13*s = 0 → x = p ∨ x = q))
  (h3 : p ≠ q) (h4 : p ≠ r) (h5 : p ≠ s) (h6 : q ≠ r) (h7 : q ≠ s) (h8 : r ≠ s) :
  p + q + r + s = 2028 :=
sorry

end pqrs_sum_l134_134029


namespace number_of_dimes_l134_134279

theorem number_of_dimes (p n d : ℕ) (h1 : p + n + d = 50) (h2 : p + 5 * n + 10 * d = 200) : d = 14 := 
sorry

end number_of_dimes_l134_134279


namespace range_of_x2_y2_l134_134196

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - x^4

theorem range_of_x2_y2 (x y : ℝ) (h : x^2 + y^2 = 2 * x) : 
  0 ≤ x^2 * y^2 ∧ x^2 * y^2 ≤ 27 / 16 :=
sorry

end range_of_x2_y2_l134_134196


namespace xy_value_l134_134219

theorem xy_value {x y : ℝ} (h1 : x - y = 6) (h2 : x^3 - y^3 = 162) : x * y = 21 :=
by
  sorry

end xy_value_l134_134219


namespace find_natural_triples_l134_134401

open Nat

noncomputable def satisfies_conditions (a b c : ℕ) : Prop :=
  (a + b) % c = 0 ∧ (b + c) % a = 0 ∧ (c + a) % b = 0

theorem find_natural_triples :
  ∀ (a b c : ℕ), satisfies_conditions a b c ↔
    (∃ a, (a = b ∧ b = c) ∨ 
          (a = b ∧ c = 2 * a) ∨ 
          (b = 2 * a ∧ c = 3 * a) ∨ 
          (b = 3 * a ∧ c = 2 * a) ∨ 
          (a = 2 * b ∧ c = 3 * b) ∨ 
          (a = 3 * b ∧ c = 2 * b)) :=
sorry

end find_natural_triples_l134_134401


namespace evaluate_g_x_plus_2_l134_134182

theorem evaluate_g_x_plus_2 (x : ℝ) (h₁ : x ≠ -3/2) (h₂ : x ≠ 2) : 
  (2 * (x + 2) + 3) / ((x + 2) - 2) = (2 * x + 7) / x :=
by 
  sorry

end evaluate_g_x_plus_2_l134_134182


namespace total_duration_of_running_l134_134287

-- Definition of conditions
def constant_speed_1 : ℝ := 18
def constant_time_1 : ℝ := 3
def next_distance : ℝ := 70
def average_speed_2 : ℝ := 14

-- Proof statement
theorem total_duration_of_running : 
    let distance_1 := constant_speed_1 * constant_time_1
    let time_2 := next_distance / average_speed_2
    distance_1 = 54 ∧ time_2 = 5 → (constant_time_1 + time_2 = 8) :=
sorry

end total_duration_of_running_l134_134287


namespace min_value_a_plus_b_plus_c_l134_134898

-- Define the main conditions
variables {a b c : ℝ}
variables (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
variables (h_eq : a^2 + 2*a*b + 4*b*c + 2*c*a = 16)

-- Define the theorem
theorem min_value_a_plus_b_plus_c : 
  (∀ a b c : ℝ, (a > 0 ∧ b > 0 ∧ c > 0) ∧ (a^2 + 2*a*b + 4*b*c + 2*c*a = 16) → a + b + c ≥ 4) :=
sorry

end min_value_a_plus_b_plus_c_l134_134898


namespace not_always_divisible_by_40_l134_134045

open Nat

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

theorem not_always_divisible_by_40 (p : ℕ) (hp_prime : is_prime p) (hp_geq7 : p ≥ 7) : ¬ (∀ p : ℕ, is_prime p ∧ p ≥ 7 → 40 ∣ (p^2 - 1)) := 
sorry

end not_always_divisible_by_40_l134_134045


namespace job_completion_time_l134_134315

def time (hours : ℕ) (minutes : ℕ) : ℕ := hours * 60 + minutes

noncomputable def start_time : ℕ := time 9 45
noncomputable def half_completion_time : ℕ := time 13 0  -- 1:00 PM in 24-hour time format

theorem job_completion_time :
  ∃ finish_time, finish_time = time 16 15 ∧
  (half_completion_time - start_time) * 2 = finish_time - start_time :=
by
  sorry

end job_completion_time_l134_134315


namespace algebraic_expression_value_l134_134883

theorem algebraic_expression_value (a b : ℝ) (h₁ : a^2 - 3 * a + 1 = 0) (h₂ : b^2 - 3 * b + 1 = 0) :
  (1 / (a^2 + 1) + 1 / (b^2 + 1)) = 1 :=
sorry

end algebraic_expression_value_l134_134883


namespace possible_values_of_f_zero_l134_134084

theorem possible_values_of_f_zero (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (x + y) = f x * f y) :
  f 0 = 0 ∨ f 0 = 1 :=
by
  sorry

end possible_values_of_f_zero_l134_134084


namespace elevation_after_descend_l134_134618

theorem elevation_after_descend (initial_elevation : ℕ) (rate : ℕ) (time : ℕ) (final_elevation : ℕ) 
  (h_initial : initial_elevation = 400) 
  (h_rate : rate = 10) 
  (h_time : time = 5) 
  (h_final : final_elevation = initial_elevation - rate * time) : 
  final_elevation = 350 := 
by 
  sorry

end elevation_after_descend_l134_134618


namespace time_for_Harish_to_paint_alone_l134_134339

theorem time_for_Harish_to_paint_alone (H : ℝ) (h1 : H > 0) (h2 :  (1 / 6 + 1 / H) = 1 / 2 ) : H = 3 :=
sorry

end time_for_Harish_to_paint_alone_l134_134339


namespace inverse_proportion_l134_134415

theorem inverse_proportion (a : ℝ) (b : ℝ) (k : ℝ) : 
  (a = k / b^2) → 
  (40 = k / 12^2) → 
  (a = 10) → 
  b = 24 := 
by
  sorry

end inverse_proportion_l134_134415


namespace find_divisor_l134_134574

theorem find_divisor (dividend quotient remainder divisor : ℕ) 
  (h_dividend : dividend = 125) 
  (h_quotient : quotient = 8) 
  (h_remainder : remainder = 5) 
  (h_equation : dividend = (divisor * quotient) + remainder) : 
  divisor = 15 := by
  sorry

end find_divisor_l134_134574


namespace prime_digit_B_l134_134982

-- Mathematical description
def six_digit_form (B : Nat) : Nat := 3 * 10^5 + 0 * 10^4 + 3 * 10^3 + 7 * 10^2 + 0 * 10^1 + B

-- Prime condition
def is_prime (n : Nat) : Prop := n > 1 ∧ (∀ m : Nat, m ∣ n → m = 1 ∨ m = n)

theorem prime_digit_B (B : Nat) : is_prime (six_digit_form B) ↔ B = 3 :=
sorry

end prime_digit_B_l134_134982


namespace elise_hospital_distance_l134_134938

noncomputable def distance_to_hospital (total_fare: ℝ) (base_price: ℝ) (toll_price: ℝ) 
(tip_percent: ℝ) (cost_per_mile: ℝ) (increase_percent: ℝ) (toll_count: ℕ) : ℝ :=
let base_and_tolls := base_price + (toll_price * toll_count)
let fare_before_tip := total_fare / (1 + tip_percent)
let distance_fare := fare_before_tip - base_and_tolls
let original_travel_fare := distance_fare / (1 + increase_percent)
original_travel_fare / cost_per_mile

theorem elise_hospital_distance : distance_to_hospital 34.34 3 2 0.15 4 0.20 3 = 5 := 
sorry

end elise_hospital_distance_l134_134938


namespace sheela_total_income_l134_134881

-- Define the monthly income as I
def monthly_income (I : Real) : Prop :=
  4500 = 0.28 * I

-- Define the annual income computed from monthly income
def annual_income (I : Real) : Real :=
  I * 12

-- Define the interest earned from savings account 
def interest_savings (principal : Real) (monthly_rate : Real) : Real :=
  principal * (monthly_rate * 12)

-- Define the interest earned from fixed deposit
def interest_fixed (principal : Real) (annual_rate : Real) : Real :=
  principal * annual_rate

-- Overall total income after one year calculation
def overall_total_income (annual_income : Real) (interest_savings : Real) (interest_fixed : Real) : Real :=
  annual_income + interest_savings + interest_fixed

-- Given conditions
variable (I : Real)
variable (principal_savings : Real := 4500)
variable (principal_fixed : Real := 3000)
variable (monthly_rate_savings : Real := 0.02)
variable (annual_rate_fixed : Real := 0.06)

-- Theorem statement to be proved
theorem sheela_total_income :
  monthly_income I →
  overall_total_income (annual_income I) 
                      (interest_savings principal_savings monthly_rate_savings)
                      (interest_fixed principal_fixed annual_rate_fixed)
  = 194117.16 :=
by
  sorry

end sheela_total_income_l134_134881


namespace parabola_intersection_sum_l134_134534

theorem parabola_intersection_sum : 
  ∃ x_0 y_0 : ℝ, (y_0 = x_0^2 + 15 * x_0 + 32) ∧ (x_0 = y_0^2 + 49 * y_0 + 593) ∧ (x_0 + y_0 = -33) :=
by
  sorry

end parabola_intersection_sum_l134_134534


namespace first_train_left_time_l134_134224

-- Definitions for conditions
def speed_first_train := 45
def speed_second_train := 90
def meeting_distance := 90

-- Prove the statement
theorem first_train_left_time (T : ℝ) (time_meeting : ℝ) :
  (time_meeting - T = 2) →
  (∀ t, 0 ≤ t → t ≤ 1 → speed_first_train * t ≤ meeting_distance) →
  (∀ t, 1 ≤ t → speed_first_train * (T + t) + speed_second_train * (t - 1) = meeting_distance) →
  (time_meeting = 2 + T) :=
by
  sorry

end first_train_left_time_l134_134224


namespace quadratic_roots_product_sum_l134_134283

theorem quadratic_roots_product_sum :
  (∀ d e : ℝ, 3 * d^2 + 4 * d - 7 = 0 ∧ 3 * e^2 + 4 * e - 7 = 0 →
   (d + 1) * (e + 1) = - 8 / 3) := by
sorry

end quadratic_roots_product_sum_l134_134283


namespace distance_between_parabola_vertices_l134_134712

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem distance_between_parabola_vertices :
  distance (0, 3) (0, -1) = 4 := 
by {
  -- Proof omitted here
  sorry
}

end distance_between_parabola_vertices_l134_134712


namespace problem_statement_l134_134775

noncomputable def a (k : ℕ) : ℝ := 2^k / (3^(2^k) + 1)
noncomputable def A : ℝ := (Finset.range 10).sum (λ k => a k)
noncomputable def B : ℝ := (Finset.range 10).prod (λ k => a k)

theorem problem_statement : A / B = (3^(2^10) - 1) / 2^47 - 1 / 2^36 := 
by
  sorry

end problem_statement_l134_134775


namespace man_twice_son_age_l134_134880

theorem man_twice_son_age (S M Y : ℕ) (h1 : S = 18) (h2 : M = S + 20) 
  (h3 : M + Y = 2 * (S + Y)) : Y = 2 :=
by
  -- Proof steps can be added here later
  sorry

end man_twice_son_age_l134_134880


namespace value_three_std_devs_less_than_mean_l134_134225

-- Define the given conditions as constants.
def mean : ℝ := 16.2
def std_dev : ℝ := 2.3

-- Translate the question into a proof statement.
theorem value_three_std_devs_less_than_mean : mean - 3 * std_dev = 9.3 :=
by sorry

end value_three_std_devs_less_than_mean_l134_134225


namespace smallest_n_l134_134937

theorem smallest_n (n : ℕ) (h1 : ∃ a : ℕ, 5 * n = a^2) (h2 : ∃ b : ℕ, 3 * n = b^3) (h3 : ∀ m : ℕ, m > 0 → (∃ a : ℕ, 5 * m = a^2) → (∃ b : ℕ, 3 * m = b^3) → n ≤ m) : n = 1125 := 
sorry

end smallest_n_l134_134937


namespace speed_conversion_l134_134021

theorem speed_conversion (speed_mps: ℝ) (conversion_factor: ℝ) (expected_speed_kmph: ℝ):
  speed_mps * conversion_factor = expected_speed_kmph :=
by
  let speed_mps := 115.00919999999999
  let conversion_factor := 3.6
  let expected_speed_kmph := 414.03312
  sorry

end speed_conversion_l134_134021


namespace part1_part2_l134_134778

variables {A B C a b c : ℝ}

-- Condition: sides opposite to angles A, B, and C are a, b, and c respectively and 4b * sin A = sqrt 7 * a
def condition1 : 4 * b * Real.sin A = Real.sqrt 7 * a := sorry

-- Prove that sin B = sqrt 7 / 4
theorem part1 (h : 4 * b * Real.sin A = Real.sqrt 7 * a) :
  Real.sin B = Real.sqrt 7 / 4 := sorry

-- Condition: a, b, and c form an arithmetic sequence with a common difference greater than 0
def condition2 : 2 * b = a + c := sorry

-- Prove that cos A - cos C = sqrt 7 / 2
theorem part2 (h1 : 4 * b * Real.sin A = Real.sqrt 7 * a) (h2 : 2 * b = a + c) :
  Real.cos A - Real.cos C = Real.sqrt 7 / 2 := sorry

end part1_part2_l134_134778


namespace rectangle_area_l134_134671

theorem rectangle_area {H W : ℝ} (h_height : H = 24) (ratio : W / H = 0.875) :
  H * W = 504 :=
by 
  sorry

end rectangle_area_l134_134671


namespace number_of_intersections_l134_134666

noncomputable def y1 (x: ℝ) : ℝ := (x - 1) ^ 4
noncomputable def y2 (x: ℝ) : ℝ := 2 ^ (abs x) - 2

theorem number_of_intersections : (∃ x₁ x₂ x₃ x₄ : ℝ, y1 x₁ = y2 x₁ ∧ y1 x₂ = y2 x₂ ∧ y1 x₃ = y2 x₃ ∧ y1 x₄ = y2 x₄ ∧ x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄) :=
sorry

end number_of_intersections_l134_134666


namespace green_apples_count_l134_134599

variables (G R : ℕ)

def total_apples_collected (G R : ℕ) : Prop :=
  R + G = 496

def relation_red_green (G R : ℕ) : Prop :=
  R = 3 * G

theorem green_apples_count (G R : ℕ) (h1 : total_apples_collected G R) (h2 : relation_red_green G R) :
  G = 124 :=
by sorry

end green_apples_count_l134_134599


namespace area_of_triangle_HFG_l134_134526

noncomputable def calculate_area_of_triangle (A B C : (ℝ × ℝ)) :=
  1/2 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem area_of_triangle_HFG :
  let A := (0, 0)
  let B := (2, 0)
  let C := (2, 4)
  let D := (0, 4)
  let E := (2, 2)
  let F := (1, 4)
  let G := (0, 2)
  let H := ((2 + 1 + 0) / 3, (2 + 4 + 2) / 3)
  calculate_area_of_triangle H F G = 2/3 :=
by
  sorry

end area_of_triangle_HFG_l134_134526


namespace fraction_absent_l134_134302

theorem fraction_absent (total_students present_students : ℕ) (h1 : total_students = 28) (h2 : present_students = 20) : 
  (total_students - present_students) / total_students = 2 / 7 :=
by
  sorry

end fraction_absent_l134_134302


namespace quadratic_inequality_solution_non_empty_l134_134934

theorem quadratic_inequality_solution_non_empty
  (a b c : ℝ) (h : a < 0) :
  ∃ x : ℝ, ax^2 + bx + c < 0 :=
sorry

end quadratic_inequality_solution_non_empty_l134_134934


namespace volume_of_remaining_sphere_after_hole_l134_134699

noncomputable def volume_of_remaining_sphere (R : ℝ) : ℝ :=
  let volume_sphere := (4 / 3) * Real.pi * R^3
  let volume_cylinder := (4 / 3) * Real.pi * (R / 2)^3
  volume_sphere - volume_cylinder

theorem volume_of_remaining_sphere_after_hole : 
  volume_of_remaining_sphere 5 = (500 * Real.pi) / 3 :=
by
  sorry

end volume_of_remaining_sphere_after_hole_l134_134699


namespace remaining_lives_l134_134226

theorem remaining_lives (initial_players quit1 quit2 player_lives : ℕ) (h1 : initial_players = 15) (h2 : quit1 = 5) (h3 : quit2 = 4) (h4 : player_lives = 7) :
  (initial_players - quit1 - quit2) * player_lives = 42 :=
by
  sorry

end remaining_lives_l134_134226


namespace interest_rate_is_4_l134_134570

-- Define the conditions based on the problem statement
def principal : ℕ := 500
def time : ℕ := 8
def simple_interest : ℕ := 160

-- Assuming the formula for simple interest
def simple_interest_formula (P R T : ℕ) : ℕ := P * R * T / 100

-- The interest rate we aim to prove
def interest_rate : ℕ := 4

-- The statement we want to prove: Given the conditions, the interest rate is 4%
theorem interest_rate_is_4 : simple_interest_formula principal interest_rate time = simple_interest := by
  -- The proof steps would go here
  sorry

end interest_rate_is_4_l134_134570


namespace arithmetic_sequence_d_range_l134_134874

theorem arithmetic_sequence_d_range (d : ℝ) :
  (10 + 4 * d > 0) ∧ (10 + 5 * d < 0) ↔ (-5/2 < d) ∧ (d < -2) :=
by
  sorry

end arithmetic_sequence_d_range_l134_134874


namespace differences_multiple_of_nine_l134_134044

theorem differences_multiple_of_nine (S : Finset ℕ) (hS : S.card = 10) (h_unique : ∀ {x y : ℕ}, x ∈ S → y ∈ S → x ≠ y → x ≠ y) : 
  ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ (a - b) % 9 = 0 :=
by
  sorry

end differences_multiple_of_nine_l134_134044


namespace problem_statement_l134_134036

def f (x : ℝ) : ℝ := sorry

theorem problem_statement
  (cond1 : ∀ {x y w : ℝ}, x > y → f x + x ≥ w → w ≥ f y + y → ∃ (z : ℝ), z ∈ Set.Icc y x ∧ f z = w - z)
  (cond2 : ∃ (u : ℝ), 0 ∈ Set.range f ∧ ∀ a ∈ Set.range f, u ≤ a)
  (cond3 : f 0 = 1)
  (cond4 : f (-2003) ≤ 2004)
  (cond5 : ∀ x y : ℝ, f x * f y = f (x * f y + y * f x + x * y)) :
  f (-2003) = 2004 := sorry

end problem_statement_l134_134036


namespace discount_equation_l134_134782

theorem discount_equation (x : ℝ) : 280 * (1 - x) ^ 2 = 177 := 
by 
  sorry

end discount_equation_l134_134782


namespace diff_of_squares_not_2018_l134_134527

theorem diff_of_squares_not_2018 (a b : ℕ) (h : a > b) : ¬(a^2 - b^2 = 2018) :=
by {
  -- proof goes here
  sorry
}

end diff_of_squares_not_2018_l134_134527


namespace number_of_consecutive_sum_sets_eq_18_l134_134975

theorem number_of_consecutive_sum_sets_eq_18 :
  ∃! (S : ℕ → ℕ) (n a : ℕ), (n ≥ 2) ∧ (S n = (n * (2 * a + n - 1)) / 2) ∧ (S n = 18) :=
sorry

end number_of_consecutive_sum_sets_eq_18_l134_134975


namespace janet_additional_money_needed_l134_134872

def janet_savings : ℕ := 2225
def monthly_rent : ℕ := 1250
def advance_months : ℕ := 2
def deposit : ℕ := 500

theorem janet_additional_money_needed :
  (advance_months * monthly_rent + deposit - janet_savings) = 775 :=
by
  sorry

end janet_additional_money_needed_l134_134872


namespace soda_difference_l134_134469

-- Define the number of regular soda bottles
def R : ℕ := 79

-- Define the number of diet soda bottles
def D : ℕ := 53

-- The theorem that states the number of regular soda bottles minus the number of diet soda bottles is 26
theorem soda_difference : R - D = 26 := 
by
  sorry

end soda_difference_l134_134469


namespace inequality_abc_l134_134792

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    1 / (a * b * c) + 1 ≥ 3 * (1 / (a^2 + b^2 + c^2) + 1 / (a + b + c)) :=
by
  sorry

end inequality_abc_l134_134792


namespace intersection_M_N_eq_set_l134_134165

-- Define sets M and N
def M : Set ℝ := {x : ℝ | x^2 < 4}
def N : Set ℝ := {x : ℝ | x^2 - 2*x - 3 < 0}

-- The theorem to be proved
theorem intersection_M_N_eq_set : (M ∩ N) = {x : ℝ | -1 < x ∧ x < 2} := by
  sorry

end intersection_M_N_eq_set_l134_134165


namespace Jill_arrives_30_minutes_before_Jack_l134_134230

theorem Jill_arrives_30_minutes_before_Jack
  (d : ℝ) (v_J : ℝ) (v_K : ℝ)
  (h₀ : d = 3) (h₁ : v_J = 12) (h₂ : v_K = 4) :
  (d / v_K - d / v_J) * 60 = 30 :=
by
  sorry

end Jill_arrives_30_minutes_before_Jack_l134_134230


namespace find_k_of_collinear_points_l134_134387

theorem find_k_of_collinear_points :
  ∃ k : ℚ, ∀ (x1 y1 x2 y2 x3 y3 : ℚ), (x1, y1) = (4, 10) → (x2, y2) = (-3, k) → (x3, y3) = (-8, 5) → 
  ((y2 - y1) * (x3 - x2) = (y3 - y2) * (x2 - x1)) → k = 85 / 12 :=
by
  sorry

end find_k_of_collinear_points_l134_134387


namespace cost_in_chinese_yuan_l134_134441

theorem cost_in_chinese_yuan
  (usd_to_nad : ℝ := 8)
  (usd_to_cny : ℝ := 5)
  (sculpture_cost_nad : ℝ := 160) :
  sculpture_cost_nad / usd_to_nad * usd_to_cny = 100 := 
by
  sorry

end cost_in_chinese_yuan_l134_134441


namespace quadratic_distinct_zeros_range_l134_134682

theorem quadratic_distinct_zeros_range (k : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 - (k+1)*x1 + k + 4 = 0 ∧ x2^2 - (k+1)*x2 + k + 4 = 0)
  ↔ k ∈ (Set.Iio (-3) ∪ Set.Ioi 5) :=
by
  sorry

end quadratic_distinct_zeros_range_l134_134682


namespace actual_price_of_food_l134_134453

theorem actual_price_of_food (P : ℝ) (h : 1.32 * P = 132) : P = 100 := 
by
  sorry

end actual_price_of_food_l134_134453


namespace remainder_div_101_l134_134688

theorem remainder_div_101 : 
  9876543210 % 101 = 68 := 
by 
  sorry

end remainder_div_101_l134_134688


namespace Eve_total_running_distance_l134_134505

def Eve_walked_distance := 0.6

def Eve_ran_distance := Eve_walked_distance + 0.1

theorem Eve_total_running_distance : Eve_ran_distance = 0.7 := 
by sorry

end Eve_total_running_distance_l134_134505


namespace students_in_class_l134_134448

theorem students_in_class (total_pencils : ℕ) (pencils_per_student : ℕ) (n: ℕ) 
    (h1 : total_pencils = 18) 
    (h2 : pencils_per_student = 9) 
    (h3 : total_pencils = n * pencils_per_student) : 
    n = 2 :=
by 
  sorry

end students_in_class_l134_134448


namespace Nina_saves_enough_to_buy_video_game_in_11_weeks_l134_134647

-- Definitions (directly from conditions)
def game_cost : ℕ := 50
def tax_rate : ℚ := 10 / 100
def sales_tax (cost : ℕ) (rate : ℚ) : ℚ := cost * rate
def total_cost (cost : ℕ) (tax : ℚ) : ℚ := cost + tax
def weekly_allowance : ℕ := 10
def savings_rate : ℚ := 1 / 2
def weekly_savings (allowance : ℕ) (rate : ℚ) : ℚ := allowance * rate
def weeks_to_save (total_cost : ℚ) (savings_per_week : ℚ) : ℚ := total_cost / savings_per_week

-- Theorem to prove
theorem Nina_saves_enough_to_buy_video_game_in_11_weeks :
  weeks_to_save
    (total_cost game_cost (sales_tax game_cost tax_rate))
    (weekly_savings weekly_allowance savings_rate) = 11 := by
-- We skip the proof for now, as per instructions
  sorry

end Nina_saves_enough_to_buy_video_game_in_11_weeks_l134_134647


namespace minimum_value_of_expression_l134_134622

noncomputable def min_squared_distance (a b c d : ℝ) : ℝ :=
  (a - c)^2 + (b - d)^2

theorem minimum_value_of_expression
  (a b c d : ℝ)
  (h1 : 4 * a^2 + b^2 - 8 * b + 12 = 0)
  (h2 : c^2 - 8 * c + 4 * d^2 + 12 = 0) :
  min_squared_distance a b c d = 42 - 16 * Real.sqrt 5 :=
sorry

end minimum_value_of_expression_l134_134622


namespace river_flow_speed_l134_134722

theorem river_flow_speed (v : ℝ) :
  (6 - v ≠ 0) ∧ (6 + v ≠ 0) ∧ ((48 / (6 - v)) + (48 / (6 + v)) = 18) → v = 2 := 
by
  sorry

end river_flow_speed_l134_134722


namespace rahim_books_from_first_shop_l134_134272

variable (books_first_shop_cost : ℕ)
variable (second_shop_books : ℕ)
variable (second_shop_books_cost : ℕ)
variable (average_price_per_book : ℕ)
variable (number_of_books_first_shop : ℕ)

theorem rahim_books_from_first_shop
  (h₁ : books_first_shop_cost = 581)
  (h₂ : second_shop_books = 20)
  (h₃ : second_shop_books_cost = 594)
  (h₄ : average_price_per_book = 25)
  (h₅ : (books_first_shop_cost + second_shop_books_cost) = (number_of_books_first_shop + second_shop_books) * average_price_per_book) :
  number_of_books_first_shop = 27 :=
sorry

end rahim_books_from_first_shop_l134_134272


namespace polar_to_rectangular_l134_134555

theorem polar_to_rectangular (r θ : ℝ) (h₁ : r = 6) (h₂ : θ = Real.pi / 2) :
  (r * Real.cos θ, r * Real.sin θ) = (0, 6) :=
by
  sorry

end polar_to_rectangular_l134_134555


namespace cone_volume_increase_l134_134976

open Real

noncomputable def cone_volume (r h : ℝ) : ℝ := (1 / 3) * π * r^2 * h
noncomputable def new_height (h : ℝ) : ℝ := 2 * h
noncomputable def new_volume (r h : ℝ) : ℝ := cone_volume r (new_height h)

theorem cone_volume_increase (r h : ℝ) : new_volume r h = 2 * (cone_volume r h) :=
by
  sorry

end cone_volume_increase_l134_134976


namespace remainder_3_pow_20_div_5_l134_134672

theorem remainder_3_pow_20_div_5 : (3 ^ 20) % 5 = 1 := 
by {
  sorry
}

end remainder_3_pow_20_div_5_l134_134672


namespace smallest_positive_integer_divisible_by_8_11_15_l134_134628

-- Define what it means for a number to be divisible by another
def divisible_by (n m : ℕ) : Prop :=
  ∃ k : ℕ, n = k * m

-- Define a function to find the least common multiple of three numbers
noncomputable def lcm_three (a b c : ℕ) : ℕ :=
  Nat.lcm a (Nat.lcm b c)

-- Statement of the theorem
theorem smallest_positive_integer_divisible_by_8_11_15 : 
  ∀ n : ℕ, (n > 0) ∧ divisible_by n 8 ∧ divisible_by n 11 ∧ divisible_by n 15 ↔ n = 1320 :=
sorry -- Proof is omitted

end smallest_positive_integer_divisible_by_8_11_15_l134_134628


namespace max_b_div_a_plus_c_l134_134609

-- Given positive numbers a, b, c
-- equation: b^2 + 2(a + c)b - ac = 0
-- Prove: ∀ a b c : ℝ (h : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : b^2 + 2*(a + c)*b - a*c = 0),
--         b/(a + c) ≤ (Real.sqrt 5 - 2)/2

theorem max_b_div_a_plus_c (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_eq : b^2 + 2 * (a + c) * b - a * c = 0) :
  b / (a + c) ≤ (Real.sqrt 5 - 2) / 2 :=
sorry

end max_b_div_a_plus_c_l134_134609


namespace unoccupied_volume_in_container_l134_134321

-- defining constants
def side_length_container := 12
def side_length_ice_cube := 3
def number_of_ice_cubes := 8
def water_fill_fraction := 3 / 4

-- defining volumes
def volume_container := side_length_container ^ 3
def volume_water := volume_container * water_fill_fraction
def volume_ice_cube := side_length_ice_cube ^ 3
def total_volume_ice := volume_ice_cube * number_of_ice_cubes
def volume_unoccupied := volume_container - (volume_water + total_volume_ice)

-- The theorem to be proved
theorem unoccupied_volume_in_container : volume_unoccupied = 216 := by
  -- Proof steps will go here
  sorry

end unoccupied_volume_in_container_l134_134321


namespace april_plant_arrangement_l134_134195

theorem april_plant_arrangement :
  let basil_plants := 5
  let tomato_plants := 4
  let total_units := (basil_plants - 2) + 1 + 1
  (Nat.factorial total_units) * (Nat.factorial tomato_plants) * (Nat.factorial 2) = 5760 :=
by
  sorry

end april_plant_arrangement_l134_134195


namespace right_triangle_c_squared_value_l134_134202

theorem right_triangle_c_squared_value (a b c : ℕ) (h : a = 9) (k : b = 12) (right_triangle : True) :
  c^2 = a^2 + b^2 ∨ c^2 = b^2 - a^2 :=
by sorry

end right_triangle_c_squared_value_l134_134202


namespace expression_equals_answer_l134_134190

noncomputable def verify_expression : ℚ :=
  15 * (1 / 17) * 34 - (1 / 2)

theorem expression_equals_answer :
  verify_expression = 59 / 2 :=
by
  sorry

end expression_equals_answer_l134_134190


namespace highest_x_value_satisfies_equation_l134_134471

theorem highest_x_value_satisfies_equation:
  ∃ x, x ≤ 4 ∧ (∀ x1, x1 ≤ 4 → x1 = 4 ↔ (15 * x1^2 - 40 * x1 + 18) / (4 * x1 - 3) + 7 * x1 = 9 * x1 - 2) :=
by
  sorry

end highest_x_value_satisfies_equation_l134_134471


namespace point_on_y_axis_l134_134003

theorem point_on_y_axis (m n : ℝ) (h : (m, n).1 = 0) : m = 0 :=
by
  sorry

end point_on_y_axis_l134_134003


namespace support_percentage_l134_134562

theorem support_percentage (men women : ℕ) (support_men_percentage support_women_percentage : ℝ) 
(men_support women_support total_support : ℕ)
(hmen : men = 150) 
(hwomen : women = 850) 
(hsupport_men_percentage : support_men_percentage = 0.55) 
(hsupport_women_percentage : support_women_percentage = 0.70) 
(hmen_support : men_support = 83) 
(hwomen_support : women_support = 595)
(htotal_support : total_support = men_support + women_support) :
  ((total_support : ℝ) / (men + women) * 100) = 68 :=
by
  -- Insert the proof here to verify each step of the calculation and rounding
  sorry

end support_percentage_l134_134562


namespace remainder_of_trailing_zeroes_in_factorials_product_l134_134633

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def product_factorials (n : ℕ) : ℕ :=
  (List.range (n + 1)).foldr (λ x acc => acc * factorial x) 1 

def trailing_zeroes (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n / 5 + trailing_zeroes (n / 5))

def trailing_zeroes_in_product (n : ℕ) : ℕ :=
  (List.range (n + 1)).foldr (λ x acc => acc + trailing_zeroes x) 0 

theorem remainder_of_trailing_zeroes_in_factorials_product :
  let N := trailing_zeroes_in_product 150
  N % 500 = 45 :=
by
  sorry

end remainder_of_trailing_zeroes_in_factorials_product_l134_134633


namespace number_subtraction_l134_134278

theorem number_subtraction
  (x : ℕ) (y : ℕ)
  (h1 : x = 30)
  (h2 : 8 * x - y = 102) : y = 138 :=
by 
  sorry

end number_subtraction_l134_134278


namespace productivity_increase_correct_l134_134548

def productivity_increase (that: ℝ) :=
  ∃ x : ℝ, (x + 1) * (x + 1) * 2500 = 2809

theorem productivity_increase_correct :
  productivity_increase (0.06) :=
by
  sorry

end productivity_increase_correct_l134_134548


namespace cubic_roots_reciprocal_sum_l134_134210

theorem cubic_roots_reciprocal_sum {α β γ : ℝ} 
  (h₁ : α + β + γ = 6)
  (h₂ : α * β + β * γ + γ * α = 11)
  (h₃ : α * β * γ = 6) :
  (1 / α^2) + (1 / β^2) + (1 / γ^2) = 49 / 36 := 
by 
  sorry

end cubic_roots_reciprocal_sum_l134_134210


namespace exists_rhombus_with_given_side_and_diag_sum_l134_134674

-- Define the context of the problem
variables (a s : ℝ)

-- Necessary definitions for a rhombus
structure Rhombus (side diag_sum : ℝ) :=
  (side_length : ℝ)
  (diag_sum : ℝ)
  (d1 d2 : ℝ)
  (side_length_eq : side_length = side)
  (diag_sum_eq : d1 + d2 = diag_sum)
  (a_squared : 2 * (side_length)^2 = d1^2 + d2^2)

-- The proof problem
theorem exists_rhombus_with_given_side_and_diag_sum (a s : ℝ) : 
  ∃ (r : Rhombus a (2*s)), r.side_length = a ∧ r.diag_sum = 2 * s :=
by
  sorry

end exists_rhombus_with_given_side_and_diag_sum_l134_134674


namespace problem1_problem2_l134_134205

def p (x a : ℝ) : Prop := x^2 + 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := (x^2 - 6*x - 72 <= 0) ∧ (x^2 + x - 6 > 0)

theorem problem1 (x : ℝ) (a : ℝ) (h : a = -1): (p x a ∨ q x) → (-6 ≤ x ∧ x < -3) ∨ (1 < x ∧ x ≤ 12) :=
sorry

theorem problem2 (a : ℝ): (¬ ∃ x : ℝ, p x a) → (¬ ∃ x : ℝ, q x) → (-4 ≤ a ∧ a ≤ -2) :=
sorry

end problem1_problem2_l134_134205


namespace max_marks_l134_134247

theorem max_marks (T : ℝ) (h : 0.33 * T = 165) : T = 500 := 
by {
  sorry
}

end max_marks_l134_134247


namespace proof_problem_l134_134730

-- Conditions: p and q are solutions to the quadratic equation 3x^2 - 5x - 8 = 0
def is_solution (p q : ℝ) : Prop := (3 * p^2 - 5 * p - 8 = 0) ∧ (3 * q^2 - 5 * q - 8 = 0)

-- Question: Compute the value of (3 * p^2 - 3 * q^2) / (p - q) given the conditions
theorem proof_problem (p q : ℝ) (h : is_solution p q) :
  (3 * p^2 - 3 * q^2) * (p - q)⁻¹ = 5 := sorry

end proof_problem_l134_134730


namespace exists_acute_triangle_l134_134406

-- Define the segments as a list of positive real numbers
variables (a b c d e : ℝ) (h0 : a > 0) (h1 : b > 0) (h2 : c > 0) (h3 : d > 0) (h4 : e > 0)
(h_order : a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ d ≥ e)

-- Conditions: Any three segments can form a triangle
variables (h_triangle_1 : a + b > c ∧ a + c > b ∧ b + c > a)
variables (h_triangle_2 : a + b > d ∧ a + d > b ∧ b + d > a)
variables (h_triangle_3 : a + b > e ∧ a + e > b ∧ b + e > a)
variables (h_triangle_4 : a + c > d ∧ a + d > c ∧ c + d > a)
variables (h_triangle_5 : a + c > e ∧ a + e > c ∧ c + e > a)
variables (h_triangle_6 : a + d > e ∧ a + e > d ∧ d + e > a)
variables (h_triangle_7 : b + c > d ∧ b + d > c ∧ c + d > b)
variables (h_triangle_8 : b + c > e ∧ b + e > c ∧ c + e > b)
variables (h_triangle_9 : b + d > e ∧ b + e > d ∧ d + e > b)
variables (h_triangle_10 : c + d > e ∧ c + e > d ∧ d + e > c)

-- Prove that there exists an acute-angled triangle 
theorem exists_acute_triangle : ∃ (x y z : ℝ), (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e) ∧
                                        (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e) ∧
                                        (z = a ∨ z = b ∨ z = c ∨ z = d ∨ z = e) ∧ 
                                        x + y > z ∧ x + z > y ∧ y + z > x ∧ 
                                        x^2 < y^2 + z^2 := 
sorry

end exists_acute_triangle_l134_134406


namespace regular_polygon_sides_l134_134889

theorem regular_polygon_sides (n : ℕ) (h : 1 < n) (exterior_angle : ℝ) (h_ext : exterior_angle = 18) :
  n * exterior_angle = 360 → n = 20 :=
by 
  sorry

end regular_polygon_sides_l134_134889


namespace twenty_five_percent_of_five_hundred_is_one_twenty_five_l134_134482

theorem twenty_five_percent_of_five_hundred_is_one_twenty_five :
  let percent := 0.25
  let amount := 500
  percent * amount = 125 :=
by
  sorry

end twenty_five_percent_of_five_hundred_is_one_twenty_five_l134_134482


namespace find_ticket_cost_l134_134918

-- Define the initial amount Tony had
def initial_amount : ℕ := 20

-- Define the amount Tony paid for a hot dog
def hot_dog_cost : ℕ := 3

-- Define the amount Tony had after buying the ticket and the hot dog
def remaining_amount : ℕ := 9

-- Define the function to find the baseball ticket cost
def ticket_cost (t : ℕ) : Prop := initial_amount - t - hot_dog_cost = remaining_amount

-- The statement to prove
theorem find_ticket_cost : ∃ t : ℕ, ticket_cost t ∧ t = 8 := 
by 
  existsi 8
  unfold ticket_cost
  simp
  exact sorry

end find_ticket_cost_l134_134918


namespace binom_10_3_eq_120_l134_134520

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l134_134520


namespace find_h_l134_134136

theorem find_h (h : ℝ) : (∀ x : ℝ, x^2 - 4 * h * x = 8) 
    ∧ (∀ r s : ℝ, r + s = 4 * h ∧ r * s = -8 → r^2 + s^2 = 18) 
    → h = (Real.sqrt 2) / 4 ∨ h = -(Real.sqrt 2) / 4 :=
by
  sorry

end find_h_l134_134136


namespace slices_with_all_toppings_l134_134601

-- Definitions
def slices_with_pepperoni (x y w : ℕ) : ℕ := 15 - x - y + w
def slices_with_mushrooms (x z w : ℕ) : ℕ := 16 - x - z + w
def slices_with_olives (y z w : ℕ) : ℕ := 10 - y - z + w

-- Problem's total validation condition
axiom total_slices_with_at_least_one_topping (x y z w : ℕ) :
  15 + 16 + 10 - x - y - z - 2 * w = 24

-- Statement to prove
theorem slices_with_all_toppings (x y z w : ℕ) (h : 15 + 16 + 10 - x - y - z - 2 * w = 24) : w = 2 :=
sorry

end slices_with_all_toppings_l134_134601


namespace triangle_min_diff_l134_134203

variable (XY YZ XZ : ℕ) -- Declaring the side lengths as natural numbers

theorem triangle_min_diff (h1 : XY < YZ ∧ YZ ≤ XZ) -- Condition for side length relations
  (h2 : XY + YZ + XZ = 2010) -- Condition for the perimeter
  (h3 : XY + YZ > XZ)
  (h4 : XY + XZ > YZ)
  (h5 : YZ + XZ > XY) :
  (YZ - XY) = 1 := -- Statement that the smallest possible value of YZ - XY is 1
sorry

end triangle_min_diff_l134_134203


namespace exponent_relation_l134_134696

theorem exponent_relation (a : ℝ) (m n : ℕ) (h1 : a^m = 9) (h2 : a^n = 3) : a^(m - n) = 3 := 
sorry

end exponent_relation_l134_134696


namespace common_ratio_is_4_l134_134034

theorem common_ratio_is_4 
  (a : ℕ → ℝ) -- The geometric sequence
  (r : ℝ) -- The common ratio
  (h_geo_seq : ∀ n, a (n + 1) = r * a n) -- Definition of geometric sequence
  (h_condition : ∀ n, a n * a (n + 1) = 16 ^ n) -- Given condition
  : r = 4 := 
  sorry

end common_ratio_is_4_l134_134034


namespace predict_HCl_formed_l134_134280

-- Define the initial conditions and chemical reaction constants
def initial_moles_CH4 : ℝ := 3
def initial_moles_Cl2 : ℝ := 6
def volume : ℝ := 2

-- Define the reaction stoichiometry constants
def stoich_CH4_to_HCl : ℝ := 2
def stoich_CH4 : ℝ := 1
def stoich_Cl2 : ℝ := 2

-- Declare the hypothesis that reaction goes to completion
axiom reaction_goes_to_completion : Prop

-- Define the function to calculate the moles of HCl formed
def moles_HCl_formed : ℝ :=
  initial_moles_CH4 * stoich_CH4_to_HCl

-- Prove the predicted amount of HCl formed is 6 moles under the given conditions
theorem predict_HCl_formed : reaction_goes_to_completion → moles_HCl_formed = 6 := by
  sorry

end predict_HCl_formed_l134_134280


namespace max_magnitude_value_is_4_l134_134542

noncomputable def max_value_vector_magnitude (θ : ℝ) : ℝ :=
  let a := (Real.cos θ, Real.sin θ)
  let b := (Real.sqrt 3, -1)
  let vector := (2 * a.1 - b.1, 2 * a.2 + 1)
  Real.sqrt (vector.1 ^ 2 + vector.2 ^ 2)

theorem max_magnitude_value_is_4 (θ : ℝ) : 
  ∃ θ : ℝ, max_value_vector_magnitude θ = 4 :=
sorry

end max_magnitude_value_is_4_l134_134542


namespace cubic_polynomial_Q_l134_134147

noncomputable def Q (x : ℝ) : ℝ := 27 * x^3 - 162 * x^2 + 297 * x - 156

theorem cubic_polynomial_Q {a b c : ℝ} 
  (h_roots : ∀ x, x^3 - 6 * x^2 + 11 * x - 6 = 0 → x = a ∨ x = b ∨ x = c)
  (h_vieta_sum : a + b + c = 6)
  (h_vieta_prod_sum : ab + bc + ca = 11)
  (h_vieta_prod : abc = 6)
  (hQ : Q a = b + c) 
  (hQb : Q b = a + c) 
  (hQc : Q c = a + b) 
  (hQ_sum : Q (a + b + c) = -27) :
  Q x = 27 * x^3 - 162 * x^2 + 297 * x - 156 :=
by { sorry }

end cubic_polynomial_Q_l134_134147


namespace graph_location_l134_134382

theorem graph_location (k : ℝ) (H : k > 0) :
    (∀ x : ℝ, (0 < x → 0 < y) → (y = 2/x) → (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0)) :=
by
    sorry

end graph_location_l134_134382


namespace find_r_k_l134_134271

theorem find_r_k :
  ∃ r k : ℚ, (∀ t : ℚ, (∃ x y : ℚ, (x = r + 3 * t ∧ y = 2 + k * t) ∧ y = 5 * x - 7)) ∧ 
            r = 9 / 5 ∧ k = -4 :=
by {
  sorry
}

end find_r_k_l134_134271


namespace ratio_of_area_to_perimeter_l134_134066

noncomputable def side_length := 10
noncomputable def altitude := (side_length * (Real.sqrt 3 / 2))
noncomputable def area := (1 / 2) * side_length * altitude
noncomputable def perimeter := 3 * side_length

theorem ratio_of_area_to_perimeter (s : ℝ) (h : ℝ) (A : ℝ) (P : ℝ) 
  (h1 : s = 10) 
  (h2 : h = s * (Real.sqrt 3 / 2)) 
  (h3 : A = (1 / 2) * s * h) 
  (h4 : P = 3 * s) :
  A / P = 5 * Real.sqrt 3 / 6 := by
  sorry

end ratio_of_area_to_perimeter_l134_134066


namespace sum_cubed_identity_l134_134801

theorem sum_cubed_identity
  (p q r : ℝ)
  (h1 : p + q + r = 5)
  (h2 : pq + pr + qr = 7)
  (h3 : pqr = -10) :
  p^3 + q^3 + r^3 = -10 := 
by
  sorry

end sum_cubed_identity_l134_134801


namespace last_number_in_first_set_l134_134160

variables (x y : ℕ)

def mean (a b c d e : ℕ) : ℕ :=
  (a + b + c + d + e) / 5

theorem last_number_in_first_set :
  (mean 28 x 42 78 y = 90) ∧ (mean 128 255 511 1023 x = 423) → y = 104 :=
by 
  sorry

end last_number_in_first_set_l134_134160


namespace sum_of_six_angles_l134_134022

theorem sum_of_six_angles (a1 a2 a3 a4 a5 a6 : ℕ) (h1 : a1 + a3 + a5 = 180)
                           (h2 : a2 + a4 + a6 = 180) : 
                           a1 + a2 + a3 + a4 + a5 + a6 = 360 := 
by
  -- omitted proof
  sorry

end sum_of_six_angles_l134_134022


namespace value_of_each_walmart_gift_card_l134_134180

variable (best_buy_value : ℕ) (best_buy_count : ℕ) (walmart_count : ℕ) (points_sent_bb : ℕ) (points_sent_wm : ℕ) (total_returnable : ℕ)

-- Conditions given in the problem
def conditions : Prop :=
  best_buy_value = 500 ∧
  best_buy_count = 6 ∧
  walmart_count = 9 ∧
  points_sent_bb = 1 ∧
  points_sent_wm = 2 ∧
  total_returnable = 3900

-- Result to prove
theorem value_of_each_walmart_gift_card : conditions best_buy_value best_buy_count walmart_count points_sent_bb points_sent_wm total_returnable →
  (total_returnable - ((best_buy_count - points_sent_bb) * best_buy_value)) / (walmart_count - points_sent_wm) = 200 :=
by
  intros h
  rcases h with
    ⟨hbv, hbc, hwc, hsbb, hswm, htr⟩
  sorry

end value_of_each_walmart_gift_card_l134_134180


namespace minimum_red_points_for_square_l134_134483

/-- Given a circle divided into 100 equal segments with points randomly colored red. 
Prove that the minimum number of red points needed to ensure at least four red points 
form the vertices of a square is 76. --/
theorem minimum_red_points_for_square (n : ℕ) (h : n = 100) (red_points : Finset ℕ)
  (hred : red_points.card ≥ 76) (hseg : ∀ i j : ℕ, i ≤ j → (j - i) % 25 ≠ 0 → ¬ (∃ a b c d : ℕ, 
  a ∈ red_points ∧ b ∈ red_points ∧ c ∈ red_points ∧ d ∈ red_points ∧ 
  (a + b + c + d) % n = 0)) : 
  ∃ a b c d : ℕ, a ∈ red_points ∧ b ∈ red_points ∧ c ∈ red_points ∧ d ∈ red_points ∧ 
  (a + b + c + d) % n = 0 :=
sorry

end minimum_red_points_for_square_l134_134483


namespace fifth_equation_pattern_l134_134866

theorem fifth_equation_pattern :
  1^3 + 2^3 + 3^3 + 4^3 + 5^3 + 6^3 = 21^2 := 
by sorry

end fifth_equation_pattern_l134_134866


namespace length_of_single_row_l134_134539

-- Define smaller cube properties and larger cube properties
def side_length_smaller_cube : ℕ := 5  -- in cm
def side_length_larger_cube : ℕ := 100  -- converted from 1 meter to cm

-- Prove that the row of smaller cubes is 400 meters long
theorem length_of_single_row :
  let num_smaller_cubes := (side_length_larger_cube / side_length_smaller_cube) ^ 3
  let length_in_cm := num_smaller_cubes * side_length_smaller_cube
  let length_in_m := length_in_cm / 100
  length_in_m = 400 :=
by
  sorry

end length_of_single_row_l134_134539


namespace value_of_b_l134_134594

theorem value_of_b (b : ℝ) (f g : ℝ → ℝ) :
  (∀ x, f x = 2 * x^2 - b * x + 3) ∧ 
  (∀ x, g x = 2 * x^2 + b * x + 3) ∧ 
  (∀ x, g x = f (x + 6)) →
  b = 12 :=
by
  sorry

end value_of_b_l134_134594


namespace perpendicular_line_through_P_l134_134728

open Real

/-- Define the point P as (-1, 3) -/
def P : ℝ × ℝ := (-1, 3)

/-- Define the line equation -/
def line1 (x y : ℝ) : Prop := x + 2 * y - 3 = 0

/-- Define the perpendicular line equation -/
def perpendicular_line (x y : ℝ) : Prop := 2 * x - y + 5 = 0

/-- The theorem stating that P lies on the perpendicular line to the given line -/
theorem perpendicular_line_through_P : ∀ x y, P = (x, y) → line1 x y → perpendicular_line x y :=
by
  sorry

end perpendicular_line_through_P_l134_134728


namespace find_all_k_l134_134862

theorem find_all_k :
  ∃ (k : ℝ), ∃ (v : ℝ × ℝ), v ≠ 0 ∧ (∃ (v₀ v₁ : ℝ), v = (v₀, v₁) 
  ∧ (3 * v₀ + 6 * v₁) = k * v₀ ∧ (4 * v₀ + 3 * v₁) = k * v₁) 
  ↔ k = 3 + 2 * Real.sqrt 6 ∨ k = 3 - 2 * Real.sqrt 6 :=
by
  -- here goes the proof
  sorry

end find_all_k_l134_134862


namespace C_share_l134_134235

-- Conditions in Lean definition
def ratio_A_C (A C : ℕ) : Prop := 3 * C = 2 * A
def ratio_A_B (A B : ℕ) : Prop := 3 * B = A
def total_profit : ℕ := 60000

-- Lean statement
theorem C_share (A B C : ℕ) (h1 : ratio_A_C A C) (h2 : ratio_A_B A B) : (C * total_profit) / (A + B + C) = 20000 :=
  by
  sorry

end C_share_l134_134235


namespace polynomial_expansion_coefficient_a8_l134_134687

theorem polynomial_expansion_coefficient_a8 :
  let a := 1
  let a_1 := 10
  let a_2 := 45
  let a_3 := 120
  let a_4 := 210
  let a_5 := 252
  let a_6 := 210
  let a_7 := 120
  let a_8 := 45
  let a_9 := 10
  let a_10 := 1
  a_8 = 45 :=
by {
  sorry
}

end polynomial_expansion_coefficient_a8_l134_134687


namespace Jans_original_speed_l134_134323

theorem Jans_original_speed
  (doubled_speed : ℕ → ℕ) (skips_after_training : ℕ) (time_in_minutes : ℕ) (original_speed : ℕ) :
  (∀ (s : ℕ), doubled_speed s = 2 * s) → 
  skips_after_training = 700 → 
  time_in_minutes = 5 → 
  (original_speed = (700 / 5) / 2) → 
  original_speed = 70 := 
by
  intros h1 h2 h3 h4
  exact h4

end Jans_original_speed_l134_134323


namespace inscribed_circle_radius_DEF_l134_134780

noncomputable def radius_inscribed_circle (DE DF EF : ℕ) : ℝ :=
  let s := (DE + DF + EF) / 2
  let K := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  K / s

theorem inscribed_circle_radius_DEF :
  radius_inscribed_circle 26 16 20 = 5 * Real.sqrt 511.5 / 31 :=
by
  sorry

end inscribed_circle_radius_DEF_l134_134780


namespace henry_collection_cost_l134_134087

def initial_figures : ℕ := 3
def total_needed : ℕ := 8
def cost_per_figure : ℕ := 6

theorem henry_collection_cost : 
  let needed_figures := total_needed - initial_figures
  let total_cost := needed_figures * cost_per_figure
  total_cost = 30 := 
by
  let needed_figures := total_needed - initial_figures
  let total_cost := needed_figures * cost_per_figure
  sorry

end henry_collection_cost_l134_134087


namespace result_after_subtraction_l134_134816

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

end result_after_subtraction_l134_134816


namespace line_through_two_points_l134_134260

theorem line_through_two_points (A B : ℝ × ℝ) (hA : A = (1, 2)) (hB : B = (3, 4)) :
  ∃ k b : ℝ, (∀ x y : ℝ, (y = k * x + b) ↔ ((x, y) = A ∨ (x, y) = B)) ∧ (k = 1) ∧ (b = 1) := 
by
  sorry

end line_through_two_points_l134_134260


namespace jenny_investment_l134_134390

theorem jenny_investment :
  ∃ (m r : ℝ), m + r = 240000 ∧ r = 6 * m ∧ r = 205714.29 :=
by
  sorry

end jenny_investment_l134_134390


namespace harly_initial_dogs_l134_134926

theorem harly_initial_dogs (x : ℝ) 
  (h1 : 0.40 * x + 0.60 * x + 5 = 53) : 
  x = 80 := 
by 
  sorry

end harly_initial_dogs_l134_134926


namespace union_A_B_l134_134516

-- Define set A
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 0}

-- Define set B
def B : Set ℝ := {x | x^2 > 1}

-- Prove the union of A and B is the expected result
theorem union_A_B : A ∪ B = {x | x ≤ 0 ∨ x > 1} :=
by
  sorry

end union_A_B_l134_134516


namespace Kenny_jumping_jacks_wednesday_l134_134683

variable (Sunday Monday Tuesday Wednesday Thursday Friday Saturday : ℕ)
variable (LastWeekTotal : ℕ := 324)
variable (SundayJumpingJacks : ℕ := 34)
variable (MondayJumpingJacks : ℕ := 20)
variable (TuesdayJumpingJacks : ℕ := 0)
variable (SomeDayJumpingJacks : ℕ := 64)
variable (FridayJumpingJacks : ℕ := 23)
variable (SaturdayJumpingJacks : ℕ := 61)

def Kenny_jumping_jacks_this_week (WednesdayJumpingJacks ThursdayJumpingJacks : ℕ) : ℕ :=
  SundayJumpingJacks + MondayJumpingJacks + TuesdayJumpingJacks + WednesdayJumpingJacks + ThursdayJumpingJacks + FridayJumpingJacks + SaturdayJumpingJacks

def Kenny_jumping_jacks_to_beat (weekTotal : ℕ) : ℕ :=
  LastWeekTotal + 1

theorem Kenny_jumping_jacks_wednesday : 
  ∃ (WednesdayJumpingJacks ThursdayJumpingJacks : ℕ), 
  Kenny_jumping_jacks_this_week WednesdayJumpingJacks ThursdayJumpingJacks = LastWeekTotal + 1 ∧ 
  (WednesdayJumpingJacks = 59 ∧ ThursdayJumpingJacks = 64) ∨ (WednesdayJumpingJacks = 64 ∧ ThursdayJumpingJacks = 59) :=
by
  sorry

end Kenny_jumping_jacks_wednesday_l134_134683


namespace parts_of_second_liquid_l134_134009

theorem parts_of_second_liquid (x : ℝ) :
    (0.10 * 5 + 0.15 * x) / (5 + x) = 11.42857142857143 / 100 ↔ x = 2 :=
by
  sorry

end parts_of_second_liquid_l134_134009


namespace maciek_total_cost_l134_134386

theorem maciek_total_cost :
  let p := 4
  let cost_of_chips := 1.75 * p
  let pretzels_cost := 2 * p
  let chips_cost := 2 * cost_of_chips
  let t := pretzels_cost + chips_cost
  t = 22 :=
by
  sorry

end maciek_total_cost_l134_134386


namespace smallest_b_not_divisible_by_5_l134_134706

theorem smallest_b_not_divisible_by_5 :
  ∃ b : ℕ, b > 2 ∧ ¬ (5 ∣ (2 * b^3 - 1)) ∧ ∀ b' > 2, ¬ (5 ∣ (2 * (b'^3) - 1)) → b = 6 :=
by
  sorry

end smallest_b_not_divisible_by_5_l134_134706


namespace eval_poly_at_2_l134_134844

def f (x : ℝ) : ℝ := 4 * x^5 - 3 * x^3 + 2 * x^2 + 5 * x + 1

theorem eval_poly_at_2 :
  f 2 = 123 :=
by
  sorry

end eval_poly_at_2_l134_134844


namespace sally_has_18_nickels_and_total_value_98_cents_l134_134689

-- Define the initial conditions
def pennies_initial := 8
def nickels_initial := 7
def nickels_from_dad := 9
def nickels_from_mom := 2

-- Define calculations based on the initial conditions
def total_nickels := nickels_initial + nickels_from_dad + nickels_from_mom
def value_pennies := pennies_initial
def value_nickels := total_nickels * 5
def total_value := value_pennies + value_nickels

-- State the theorem to prove the correct answers
theorem sally_has_18_nickels_and_total_value_98_cents :
  total_nickels = 18 ∧ total_value = 98 := 
by {
  -- Proof goes here
  sorry
}

end sally_has_18_nickels_and_total_value_98_cents_l134_134689


namespace Chicago_White_Sox_loss_l134_134053

theorem Chicago_White_Sox_loss :
  ∃ (L : ℕ), (99 = L + 36) ∧ (L = 63) :=
by
  sorry

end Chicago_White_Sox_loss_l134_134053


namespace range_of_a_l134_134150

def A (x : ℝ) : Prop := x^2 - 6*x + 5 ≤ 0
def B (x a : ℝ) : Prop := x < a + 1

theorem range_of_a (a : ℝ) : (∃ x : ℝ, A x ∧ B x a) ↔ a > 0 := by
  sorry

end range_of_a_l134_134150


namespace nested_sqrt_eq_two_l134_134005

theorem nested_sqrt_eq_two (y : ℝ) (h : y = Real.sqrt (2 + y)) : y = 2 :=
by
  sorry

end nested_sqrt_eq_two_l134_134005


namespace probability_neither_red_nor_purple_l134_134807

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

end probability_neither_red_nor_purple_l134_134807


namespace inequality_holds_range_of_expression_l134_134669

noncomputable def f (x : ℝ) : ℝ := |2 * x - 1|
noncomputable def g (x : ℝ) : ℝ := f x + f (x - 1)

theorem inequality_holds (x : ℝ) : f x < |x - 2| + 4 ↔ x ∈ Set.Ioo (-5 : ℝ) 3 := by
  sorry

theorem range_of_expression (m n : ℝ) (h : m + n = 2) (hm : m > 0) (hn : n > 0) :
  (m^2 + 2) / m + (n^2 + 1) / n ∈ Set.Ici ((7 + 2 * Real.sqrt 2) / 2) := by
  sorry

end inequality_holds_range_of_expression_l134_134669


namespace conic_sections_parabolas_l134_134057

theorem conic_sections_parabolas (x y : ℝ) :
  (y^6 - 9*x^6 = 3*y^3 - 1) → 
  ((y^3 = 3*x^3 + 1) ∨ (y^3 = -3*x^3 + 1)) := 
by 
  sorry

end conic_sections_parabolas_l134_134057


namespace trigonometric_identity_l134_134103

theorem trigonometric_identity : 
  (Real.sin (42 * Real.pi / 180) * Real.cos (18 * Real.pi / 180) - 
   Real.cos (138 * Real.pi / 180) * Real.cos (72 * Real.pi / 180)) = 
  (Real.sqrt 3 / 2) :=
by
  sorry

end trigonometric_identity_l134_134103


namespace initial_number_of_rabbits_is_50_l134_134663

-- Initial number of weasels
def initial_weasels := 100

-- Each fox catches 4 weasels and 2 rabbits per week
def weasels_caught_per_fox_per_week := 4
def rabbits_caught_per_fox_per_week := 2

-- There are 3 foxes
def num_foxes := 3

-- After 3 weeks, 96 weasels and rabbits are left
def weasels_and_rabbits_left := 96
def weeks := 3

theorem initial_number_of_rabbits_is_50 :
  (initial_weasels + (initial_weasels + weasels_and_rabbits_left)) - initial_weasels = 50 :=
by
  sorry

end initial_number_of_rabbits_is_50_l134_134663


namespace original_price_of_article_l134_134650

theorem original_price_of_article (new_price : ℝ) (reduction_percentage : ℝ) (original_price : ℝ) 
  (h_reduction : reduction_percentage = 56/100) (h_new_price : new_price = 4400) :
  original_price = 10000 :=
sorry

end original_price_of_article_l134_134650


namespace jason_stacked_bales_l134_134267

theorem jason_stacked_bales (initial_bales : ℕ) (final_bales : ℕ) (stored_bales : ℕ) 
  (h1 : initial_bales = 73) (h2 : final_bales = 96) : stored_bales = final_bales - initial_bales := 
by
  rw [h1, h2]
  sorry

end jason_stacked_bales_l134_134267


namespace xiaoming_problem_l134_134927

theorem xiaoming_problem :
  (- 1 / 24) / (1 / 3 - 1 / 6 + 3 / 8) = - 1 / 13 :=
by
  sorry

end xiaoming_problem_l134_134927


namespace functional_inequality_solution_l134_134491

theorem functional_inequality_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, 2 + f x * f y ≤ x * y + 2 * f (x + y + 1)) ↔ (∀ x : ℝ, f x = x + 2) :=
by
  sorry

end functional_inequality_solution_l134_134491


namespace math_problem_l134_134535

theorem math_problem :
  3 ^ (2 + 4 + 6) - (3 ^ 2 + 3 ^ 4 + 3 ^ 6) + (3 ^ 2 * 3 ^ 4 * 3 ^ 6) = 1062242 :=
by
  sorry

end math_problem_l134_134535


namespace percentage_fraction_l134_134430

theorem percentage_fraction (P : ℚ) (hP : P < 35) (h : (P / 100) * 180 = 42) : P = 7 / 30 * 100 :=
by
  sorry

end percentage_fraction_l134_134430


namespace contradiction_example_l134_134697

theorem contradiction_example (a b c d : ℝ) 
  (h1 : a + b = 1) 
  (h2 : c + d = 1) 
  (h3 : ac + bd > 1) : 
  ¬ (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) → 
  a < 0 ∨ b < 0 ∨ c < 0 ∨ d < 0 :=
by
  sorry

end contradiction_example_l134_134697


namespace range_of_a_l134_134644

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 2 - a ≤ x ∧ x ≤ 2 + a}
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 5}

-- Define the theorem to be proved
theorem range_of_a (a : ℝ) (h₁ : A a ⊆ B) (h₂ : 2 - a < 2 + a) : 0 < a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l134_134644


namespace value_of_expression_l134_134344

variables (u v w : ℝ)

theorem value_of_expression (h1 : u = 3 * v) (h2 : w = 5 * u) : 2 * v + u + w = 20 * v :=
by sorry

end value_of_expression_l134_134344


namespace union_is_correct_l134_134091

def A : Set ℕ := {1, 2, 4}
def B : Set ℕ := {2, 6}

theorem union_is_correct : A ∪ B = {1, 2, 4, 6} := by
  sorry

end union_is_correct_l134_134091


namespace function_y_neg3x_plus_1_quadrants_l134_134451

theorem function_y_neg3x_plus_1_quadrants :
  ∀ (x : ℝ), (∃ y : ℝ, y = -3 * x + 1) ∧ (
    (x < 0 ∧ y > 0) ∨ -- Second quadrant
    (x > 0 ∧ y > 0) ∨ -- First quadrant
    (x > 0 ∧ y < 0)   -- Fourth quadrant
  )
:= sorry

end function_y_neg3x_plus_1_quadrants_l134_134451


namespace initial_milk_quantity_l134_134656

theorem initial_milk_quantity (A B C D : ℝ) (hA : A > 0)
  (hB : B = 0.55 * A)
  (hC : C = 1.125 * A)
  (hD : D = 0.8 * A)
  (hTransferBC : B + 150 = C - 150 + 100)
  (hTransferDC : C - 50 = D - 100)
  (hEqual : B + 150 = D - 100) : 
  A = 1000 :=
by sorry

end initial_milk_quantity_l134_134656


namespace gcd_of_fraction_in_lowest_terms_l134_134163

theorem gcd_of_fraction_in_lowest_terms (n : ℤ) (h : n % 2 = 1) : Int.gcd (2 * n + 2) (3 * n + 2) = 1 := 
by 
  sorry

end gcd_of_fraction_in_lowest_terms_l134_134163


namespace committee_size_l134_134639

theorem committee_size (n : ℕ)
  (h : ((n - 2 : ℕ) : ℚ) / ((n - 1) * (n - 2) / 2 : ℚ) = 0.4) :
  n = 6 :=
by
  sorry

end committee_size_l134_134639


namespace prob_none_given_not_A_l134_134145

-- Definitions based on the conditions
def prob_single (h : ℕ → Prop) : ℝ := 0.2
def prob_double (h1 h2 : ℕ → Prop) : ℝ := 0.1
def prob_triple_given_AB : ℝ := 0.5

-- Assume that h1, h2, and h3 represent the hazards A, B, and C respectively.
variables (A B C : ℕ → Prop)

-- The ultimate theorem we want to prove
theorem prob_none_given_not_A (P : ℕ → Prop) :
  ((1 - (0.2 * 3 + 0.1 * 3) + (prob_triple_given_AB * (prob_single A + prob_double A B))) / (1 - 0.2) = 11 / 9) :=
by
  sorry

end prob_none_given_not_A_l134_134145


namespace triangle_perfect_square_l134_134464

theorem triangle_perfect_square (a b c : ℤ) (h : ∃ h₁ h₂ h₃ : ℤ, (1/2) * a * h₁ = (1/2) * b * h₂ ∧ (1/2) * b * h₂ = (1/2) * c * h₃ ∧ (h₁ = h₂ + h₃)) :
  ∃ k : ℤ, a^2 + b^2 + c^2 = k^2 :=
by
  sorry

end triangle_perfect_square_l134_134464


namespace corn_syrup_content_sport_formulation_l134_134864

def standard_ratio_flavoring : ℕ := 1
def standard_ratio_corn_syrup : ℕ := 12
def standard_ratio_water : ℕ := 30

def sport_ratio_flavoring_to_corn_syrup : ℕ := 3 * standard_ratio_flavoring
def sport_ratio_flavoring_to_water : ℕ := standard_ratio_flavoring / 2

def sport_ratio_flavoring : ℕ := 1
def sport_ratio_corn_syrup : ℕ := sport_ratio_flavoring * sport_ratio_flavoring_to_corn_syrup
def sport_ratio_water : ℕ := (sport_ratio_flavoring * standard_ratio_water) / 2

def water_content_sport_formulation : ℕ := 30

theorem corn_syrup_content_sport_formulation : 
  (sport_ratio_corn_syrup / sport_ratio_water) * water_content_sport_formulation = 2 :=
by
  sorry

end corn_syrup_content_sport_formulation_l134_134864


namespace total_driving_routes_l134_134121

def num_starting_points : ℕ := 4
def num_destinations : ℕ := 3

theorem total_driving_routes (h1 : ¬(num_starting_points = 0)) (h2 : ¬(num_destinations = 0)) : 
  num_starting_points * num_destinations = 12 :=
by
  sorry

end total_driving_routes_l134_134121


namespace smallest_m_l134_134043

theorem smallest_m (m : ℤ) (h : 2 * m + 1 ≥ 0) : m ≥ 0 :=
sorry

end smallest_m_l134_134043


namespace find_pairs_l134_134377

theorem find_pairs (x y : ℕ) (h1 : 0 < x ∧ 0 < y)
  (h2 : ∃ p : ℕ, Prime p ∧ (x + y = 2 * p))
  (h3 : (x! + y!) % (x + y) = 0) : ∃ p : ℕ, Prime p ∧ x = p ∧ y = p :=
by
  sorry

end find_pairs_l134_134377


namespace like_terms_sum_l134_134073

theorem like_terms_sum (m n : ℕ) (h1 : m + 1 = 1) (h2 : 3 = n) : m + n = 3 :=
by sorry

end like_terms_sum_l134_134073


namespace dance_team_recruitment_l134_134870

theorem dance_team_recruitment 
  (total_students choir_students track_field_students dance_students : ℕ)
  (h1 : total_students = 100)
  (h2 : choir_students = 2 * track_field_students)
  (h3 : dance_students = choir_students + 10)
  (h4 : total_students = track_field_students + choir_students + dance_students) : 
  dance_students = 46 :=
by {
  -- The proof goes here, but it is not required as per instructions
  sorry
}

end dance_team_recruitment_l134_134870


namespace probability_target_A_destroyed_probability_exactly_one_target_destroyed_l134_134996

-- Definition of probabilities
def prob_A_hits_target_A := 1 / 2
def prob_A_hits_target_B := 1 / 2
def prob_B_hits_target_A := 1 / 3
def prob_B_hits_target_B := 2 / 5

-- The event of target A being destroyed
def prob_target_A_destroyed := prob_A_hits_target_A * prob_B_hits_target_A

-- The event of target B being destroyed
def prob_target_B_destroyed := prob_A_hits_target_B * prob_B_hits_target_B

-- Complementary events
def prob_target_A_not_destroyed := 1 - prob_target_A_destroyed
def prob_target_B_not_destroyed := 1 - prob_target_B_destroyed

-- Exactly one target being destroyed
def prob_exactly_one_target_destroyed := 
  (prob_target_A_destroyed * prob_target_B_not_destroyed) +
  (prob_target_B_destroyed * prob_target_A_not_destroyed)

theorem probability_target_A_destroyed : prob_target_A_destroyed = 1 / 6 := by
  -- Proof needed here
  sorry

theorem probability_exactly_one_target_destroyed : prob_exactly_one_target_destroyed = 3 / 10 := by
  -- Proof needed here
  sorry

end probability_target_A_destroyed_probability_exactly_one_target_destroyed_l134_134996


namespace sin_minus_cos_eq_pm_sqrt_b_l134_134677

open Real

/-- If θ is an acute angle such that cos(2θ) = b, then sin(θ) - cos(θ) = ±√b. -/
theorem sin_minus_cos_eq_pm_sqrt_b (θ b : ℝ) (hθ : 0 < θ ∧ θ < π / 2) (hcos2θ : cos (2 * θ) = b) :
  sin θ - cos θ = sqrt b ∨ sin θ - cos θ = -sqrt b :=
sorry

end sin_minus_cos_eq_pm_sqrt_b_l134_134677


namespace day_100_M_minus_1_is_Tuesday_l134_134032

variable {M : ℕ}

-- Given conditions
def day_200_M_is_Monday (M : ℕ) : Prop :=
  ((200 % 7) = 6)

def day_300_M_plus_2_is_Monday (M : ℕ) : Prop :=
  ((300 % 7) = 6)

-- Statement to prove
theorem day_100_M_minus_1_is_Tuesday (M : ℕ) 
  (h1 : day_200_M_is_Monday M) 
  (h2 : day_300_M_plus_2_is_Monday M) 
  : (((100 + (365 - 200)) % 7 + 7 - 1) % 7 = 2) :=
sorry

end day_100_M_minus_1_is_Tuesday_l134_134032


namespace last_two_digits_of_7_pow_10_l134_134326

theorem last_two_digits_of_7_pow_10 :
  (7 ^ 10) % 100 = 49 := by
  sorry

end last_two_digits_of_7_pow_10_l134_134326


namespace quadratic_roots_l134_134046

theorem quadratic_roots {x y : ℝ} (h1 : x + y = 8) (h2 : |x - y| = 10) :
  ∃ (a b c : ℝ), a ≠ 0 ∧ (x^2 - 8*x - 9 = 0) ∧ (y^2 - 8*y - 9 = 0) :=
by
  sorry

end quadratic_roots_l134_134046


namespace correct_mark_l134_134820

theorem correct_mark (x : ℕ) (S_Correct S_Wrong : ℕ) (n : ℕ) :
  n = 26 →
  S_Wrong = S_Correct + (83 - x) →
  (S_Wrong : ℚ) / n = (S_Correct : ℚ) / n + 1 / 2 →
  x = 70 :=
by
  intros h1 h2 h3
  sorry

end correct_mark_l134_134820


namespace sum_of_tens_and_units_digit_of_8_pow_100_l134_134354

noncomputable def tens_digit (n : ℕ) : ℕ := (n / 10) % 10
noncomputable def units_digit (n : ℕ) : ℕ := n % 10
noncomputable def sum_of_digits (n : ℕ) := tens_digit n + units_digit n

theorem sum_of_tens_and_units_digit_of_8_pow_100 : sum_of_digits (8 ^ 100) = 13 :=
by 
  sorry

end sum_of_tens_and_units_digit_of_8_pow_100_l134_134354


namespace tenth_number_in_twentieth_row_l134_134047

def arrangement : ∀ n : ℕ, ℕ := -- A function defining the nth number in the sequence.
  sorry

-- A function to get the nth number in the mth row, respecting the arithmetic sequence property.
def number_in_row (m n : ℕ) : ℕ := 
  sorry

theorem tenth_number_in_twentieth_row : number_in_row 20 10 = 426 :=
  sorry

end tenth_number_in_twentieth_row_l134_134047


namespace angle_B_is_pi_over_3_range_of_expression_l134_134466

variable {A B C a b c : ℝ}

-- Conditions
def sides_opposite_angles (A B C : ℝ) (a b c : ℝ): Prop :=
  (2 * c - a) * Real.cos B - b * Real.cos A = 0

-- Part 1: Prove B = π/3
theorem angle_B_is_pi_over_3 (h : sides_opposite_angles A B C a b c) : 
    B = Real.pi / 3 := 
  sorry

-- Part 2: Prove the range of sqrt(3) * (sin A + sin(C - π/6)) is (1, 2]
theorem range_of_expression (h : 0 < A ∧ A < 2 * Real.pi / 3) : 
    (1:ℝ) < Real.sqrt 3 * (Real.sin A + Real.sin (C - Real.pi / 6)) 
    ∧ Real.sqrt 3 * (Real.sin A + Real.sin (C - Real.pi / 6)) ≤ 2 := 
  sorry

end angle_B_is_pi_over_3_range_of_expression_l134_134466


namespace expression_simplification_l134_134828

theorem expression_simplification (a b : ℤ) : 
  2 * (2 * a - 3 * b) - 3 * (2 * b - 3 * a) = 13 * a - 12 * b :=
by
  sorry

end expression_simplification_l134_134828


namespace problem_1_problem_2_l134_134557

noncomputable def f (x : ℝ) (a : ℝ) := Real.sqrt (a - x^2)

-- First proof problem statement: 
theorem problem_1 (a : ℝ) (x : ℝ) (A B : Set ℝ) (h1 : a = 4) (h2 : A = {x : ℝ | -2 ≤ x ∧ x ≤ 2}) (h3 : B = {y : ℝ | 0 ≤ y ∧ y ≤ 2}) : 
  (A ∩ B = {y : ℝ | 0 ≤ y ∧ y ≤ 2}) :=
sorry

-- Second proof problem statement:
theorem problem_2 (a : ℝ) (h : 1 ∈ {y : ℝ | 0 ≤ y ∧ y ≤ Real.sqrt a}) : a ≥ 1 :=
sorry

end problem_1_problem_2_l134_134557


namespace log_inequality_l134_134380

theorem log_inequality (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x ≠ 1) (h4 : y ≠ 1) :
    (Real.log y / Real.log x + Real.log x / Real.log y > 2) →
    (x ≠ y ∧ ((x > 1 ∧ y > 1) ∨ (x < 1 ∧ y < 1))) :=
by
    sorry

end log_inequality_l134_134380


namespace tammy_earnings_after_3_weeks_l134_134118

noncomputable def oranges_picked_per_day (num_trees : ℕ) (oranges_per_tree : ℕ) : ℕ :=
  num_trees * oranges_per_tree

noncomputable def packs_sold_per_day (oranges_per_day : ℕ) (oranges_per_pack : ℕ) : ℕ :=
  oranges_per_day / oranges_per_pack

noncomputable def total_packs_sold_in_weeks (packs_per_day : ℕ) (days_in_week : ℕ) (num_weeks : ℕ) : ℕ :=
  packs_per_day * days_in_week * num_weeks

noncomputable def money_earned (total_packs : ℕ) (price_per_pack : ℕ) : ℕ :=
  total_packs * price_per_pack

theorem tammy_earnings_after_3_weeks :
  let num_trees := 10
  let oranges_per_tree := 12
  let oranges_per_pack := 6
  let price_per_pack := 2
  let days_in_week := 7
  let num_weeks := 3
  oranges_picked_per_day num_trees oranges_per_tree /
  oranges_per_pack *
  days_in_week *
  num_weeks *
  price_per_pack = 840 :=
by {
  sorry
}

end tammy_earnings_after_3_weeks_l134_134118


namespace value_of_first_equation_l134_134985

theorem value_of_first_equation (x y a : ℝ) 
  (h₁ : 2 * x + y = a) 
  (h₂ : x + 2 * y = 10) 
  (h₃ : (x + y) / 3 = 4) : 
  a = 12 :=
by 
  sorry

end value_of_first_equation_l134_134985


namespace second_smallest_packs_hot_dogs_l134_134754

theorem second_smallest_packs_hot_dogs 
    (n : ℕ) 
    (k : ℤ) 
    (h1 : 10 * n ≡ 4 [MOD 8]) 
    (h2 : n = 4 * k + 2) : 
    n = 6 :=
by sorry

end second_smallest_packs_hot_dogs_l134_134754


namespace discriminant_divisible_l134_134244

theorem discriminant_divisible (a b: ℝ) (n: ℤ) (h: (∃ x1 x2: ℝ, 2018*x1^2 + a*x1 + b = 0 ∧ 2018*x2^2 + a*x2 + b = 0 ∧ x1 - x2 = n)): 
  ∃ k: ℤ, a^2 - 4 * 2018 * b = 2018^2 * k := 
by 
  sorry

end discriminant_divisible_l134_134244


namespace find_2023rd_digit_of_11_div_13_l134_134065

noncomputable def decimal_expansion_repeating (n d : Nat) : List Nat := sorry

noncomputable def decimal_expansion_digit (n d pos : Nat) : Nat :=
  let repeating_block := decimal_expansion_repeating n d
  repeating_block.get! ((pos - 1) % repeating_block.length)

theorem find_2023rd_digit_of_11_div_13 :
  decimal_expansion_digit 11 13 2023 = 8 := by
  sorry

end find_2023rd_digit_of_11_div_13_l134_134065


namespace tangents_product_l134_134144

theorem tangents_product (x y : ℝ) 
  (h1 : Real.tan x - Real.tan y = 7) 
  (h2 : 2 * Real.sin (2 * (x - y)) = Real.sin (2 * x) * Real.sin (2 * y)) :
  Real.tan x * Real.tan y = -7/6 := 
sorry

end tangents_product_l134_134144


namespace perimeter_of_star_is_160_l134_134268

-- Define the radius of the circles
def radius := 5 -- in cm

-- Define the diameter based on radius
def diameter := 2 * radius

-- Define the side length of the square
def side_length_square := 2 * diameter

-- Define the side length of each equilateral triangle
def side_length_triangle := side_length_square

-- Define the perimeter of the four-pointed star
def perimeter_star := 8 * side_length_triangle

-- Statement: The perimeter of the star is 160 cm
theorem perimeter_of_star_is_160 :
  perimeter_star = 160 := by
    sorry

end perimeter_of_star_is_160_l134_134268


namespace gold_coins_percentage_l134_134827

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

end gold_coins_percentage_l134_134827


namespace binary_to_decimal_l134_134823

theorem binary_to_decimal : (1 * 2^0 + 1 * 2^1 + 0 * 2^2 + 0 * 2^3 + 1 * 2^4 + 1 * 2^5) = 51 :=
by
  sorry

end binary_to_decimal_l134_134823


namespace hours_per_shift_l134_134433

def hourlyWage : ℝ := 4.0
def tipRate : ℝ := 0.15
def shiftsWorked : ℕ := 3
def averageOrdersPerHour : ℝ := 40.0
def totalEarnings : ℝ := 240.0

theorem hours_per_shift :
  (hourlyWage + averageOrdersPerHour * tipRate) * (8 * shiftsWorked) = totalEarnings := 
sorry

end hours_per_shift_l134_134433


namespace ratio_a3_b3_l134_134362

theorem ratio_a3_b3 (a : ℝ) (ha : a ≠ 0)
  (h1 : a = b₁)
  (h2 : a * q * b = 2)
  (h3 : b₄ = 8 * a * q^3) :
  (∃ r : ℝ, r = -5 ∨ r = -3.2) :=
by
  sorry

end ratio_a3_b3_l134_134362


namespace find_M_l134_134915

theorem find_M :
  ∃ (M : ℕ), 1001 + 1003 + 1005 + 1007 + 1009 = 5100 - M ∧ M = 75 :=
by
  sorry

end find_M_l134_134915


namespace chord_bisected_line_eq_l134_134685

theorem chord_bisected_line_eq (x y : ℝ) (hx1 : x^2 + 4 * y^2 = 36) (hx2 : (4, 2) = ((x1 + x2) / 2, (y1 + y2) / 2)) :
  x + 2 * y - 8 = 0 :=
sorry

end chord_bisected_line_eq_l134_134685


namespace arithmetic_sequence_problem_l134_134875

variable {a_n : ℕ → ℤ}
variable {S_n : ℕ → ℤ}
variable (h_arith_seq : ∀ n, a_n n = a_n 1 + (n - 1) * d)
variable (h_S_n : ∀ n, S_n n = (n * (a_n 1 + a_n n)) / 2)

theorem arithmetic_sequence_problem
  (h1 : S_n 5 = 2 * a_n 5)
  (h2 : a_n 3 = -4) :
  a_n 9 = -22 := sorry

end arithmetic_sequence_problem_l134_134875


namespace total_oranges_in_buckets_l134_134569

theorem total_oranges_in_buckets (a b c : ℕ) 
  (h1 : a = 22) 
  (h2 : b = a + 17) 
  (h3 : c = b - 11) : 
  a + b + c = 89 := 
by {
  sorry
}

end total_oranges_in_buckets_l134_134569


namespace coin_toss_5_times_same_side_l134_134428

noncomputable def probability_of_same_side (n : ℕ) : ℝ :=
  (1 / 2) ^ n

theorem coin_toss_5_times_same_side :
  probability_of_same_side 5 = 1 / 32 :=
by 
  -- The goal is to prove (1/2)^5 = 1/32
  sorry

end coin_toss_5_times_same_side_l134_134428


namespace LCM_of_18_and_27_l134_134393

theorem LCM_of_18_and_27 : Nat.lcm 18 27 = 54 := by
  sorry

end LCM_of_18_and_27_l134_134393


namespace count_divisible_digits_l134_134721

def is_divisible (a b : ℕ) : Prop := b ≠ 0 ∧ a % b = 0

theorem count_divisible_digits :
  ∃! (s : Finset ℕ), s = {n | n ∈ Finset.range 10 ∧ n ≠ 0 ∧ is_divisible (25 * n) n} ∧ (Finset.card s = 3) := 
by
  sorry

end count_divisible_digits_l134_134721


namespace total_animal_eyes_l134_134234

def frogs_in_pond := 20
def crocodiles_in_pond := 6
def eyes_per_frog := 2
def eyes_per_crocodile := 2

theorem total_animal_eyes : (frogs_in_pond * eyes_per_frog + crocodiles_in_pond * eyes_per_crocodile) = 52 := by
  sorry

end total_animal_eyes_l134_134234


namespace solve_for_b_l134_134050

noncomputable def P (x a b d c : ℝ) : ℝ := x^4 + a * x^3 + b * x^2 + d * x + c

theorem solve_for_b (a b d c : ℝ) (h1 : -a = d) (h2 : d = 1 + a + b + d + c) (h3 : c = 8) :
    b = -17 :=
by
  sorry

end solve_for_b_l134_134050


namespace monthly_income_l134_134858

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

end monthly_income_l134_134858


namespace inequality_a4b_to_abcd_l134_134711

theorem inequality_a4b_to_abcd (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  a^4 * b + b^4 * c + c^4 * d + d^4 * a ≥ a * b * c * d * (a + b + c + d) :=
by
  sorry

end inequality_a4b_to_abcd_l134_134711


namespace intersection_M_N_l134_134395

def M : Set ℝ := {x | |x| ≤ 2}
def N : Set ℝ := {x | x^2 + 2 * x - 3 ≤ 0}
def intersection : Set ℝ := {x | -2 ≤ x ∧ x ≤ 1}

theorem intersection_M_N : M ∩ N = intersection := by
  sorry

end intersection_M_N_l134_134395


namespace trains_pass_each_other_l134_134821

noncomputable def time_to_pass (speed1 speed2 distance : ℕ) : ℚ :=
  (distance : ℚ) / ((speed1 + speed2) : ℚ) * 60

theorem trains_pass_each_other :
  time_to_pass 60 80 100 = 42.86 := sorry

end trains_pass_each_other_l134_134821


namespace range_of_sum_l134_134608

theorem range_of_sum (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : a + b + 3 = a * b) : 
a + b ≥ 6 := 
sorry

end range_of_sum_l134_134608


namespace solution_pairs_l134_134887

theorem solution_pairs (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b)
    (h_coprime: Nat.gcd (2 * a - 1) (2 * b + 1) = 1) 
    (h_divides : (a + b) ∣ (4 * a * b + 1)) :
    ∃ n : ℕ, a = n ∧ b = n + 1 :=
by
  -- statement
  sorry

end solution_pairs_l134_134887


namespace isosceles_right_triangle_C_coordinates_l134_134281

theorem isosceles_right_triangle_C_coordinates :
  ∃ C : ℝ × ℝ, (let A : ℝ × ℝ := (1, 0)
                let B : ℝ × ℝ := (3, 1) 
                ∃ (x y: ℝ), C = (x, y) ∧ 
                ((x-1)^2 + y^2 = 10) ∧ 
                (((x-3)^2 + (y-1)^2 = 10))) ∨
                ((x = 2 ∧ y = 3) ∨ (x = 4 ∧ y = -1)) :=
by
  sorry

end isosceles_right_triangle_C_coordinates_l134_134281


namespace positive_sum_inequality_l134_134651

theorem positive_sum_inequality 
  (a b c : ℝ) 
  (a_pos : 0 < a) 
  (b_pos : 0 < b) 
  (c_pos : 0 < c) : 
  (a^2 + ab + b^2) * (b^2 + bc + c^2) * (c^2 + ca + a^2) ≥ (ab + bc + ca)^3 := 
by 
  sorry

end positive_sum_inequality_l134_134651


namespace swimming_pool_width_l134_134634

theorem swimming_pool_width
  (length : ℝ)
  (lowered_height_inches : ℝ)
  (removed_water_gallons : ℝ)
  (gallons_per_cubic_foot : ℝ)
  (volume_for_removal : ℝ)
  (width : ℝ) :
  length = 60 → 
  lowered_height_inches = 6 →
  removed_water_gallons = 4500 →
  gallons_per_cubic_foot = 7.5 →
  volume_for_removal = removed_water_gallons / gallons_per_cubic_foot →
  width = volume_for_removal / (length * (lowered_height_inches / 12)) →
  width = 20 :=
by
  intros h_length h_lowered_height h_removed_water h_gallons_per_cubic_foot h_volume_for_removal h_width
  sorry

end swimming_pool_width_l134_134634


namespace intersection_A_B_l134_134368

def A : Set ℤ := {-2, 0, 1, 2}
def B : Set ℤ := { x | -2 ≤ x ∧ x ≤ 1 }

theorem intersection_A_B : A ∩ B = {-2, 0, 1} := by
  sorry

end intersection_A_B_l134_134368


namespace find_acute_angle_x_l134_134949

def a_parallel_b (x : ℝ) : Prop :=
  let a := (Real.sin x, 3 / 4)
  let b := (1 / 3, 1 / 2 * Real.cos x)
  b.1 * a.2 = a.1 * b.2

theorem find_acute_angle_x (x : ℝ) (h : a_parallel_b x) : x = Real.pi / 4 :=
by
  sorry

end find_acute_angle_x_l134_134949


namespace probability_of_selection_l134_134063

theorem probability_of_selection : 
  ∀ (n k : ℕ), n = 121 ∧ k = 20 → (P : ℚ) = 20 / 121 :=
by
  intros n k h
  sorry

end probability_of_selection_l134_134063


namespace range_of_a_l134_134159

variable (a : ℝ)

theorem range_of_a (ha : a ≥ 1/4) : ¬ ∃ x : ℝ, a * x^2 + x + 1 < 0 := sorry

end range_of_a_l134_134159


namespace find_principal_l134_134756

theorem find_principal (x y : ℝ) : 
  (2 * x * y / 100 = 400) → 
  (2 * x * y + x * y^2 / 100 = 41000) → 
  x = 4000 := 
by
  sorry

end find_principal_l134_134756


namespace correct_answer_is_B_l134_134657

-- Define what it means to be a quadratic equation in one variable
def is_quadratic_in_one_variable (eq : ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x : ℝ, eq x ↔ (a * x ^ 2 + b * x + c = 0)

-- Conditions:
def eqA (x : ℝ) : Prop := 2 * x + 1 = 0
def eqB (x : ℝ) : Prop := x ^ 2 + 1 = 0
def eqC (x y : ℝ) : Prop := y ^ 2 + x = 1
def eqD (x : ℝ) : Prop := 1 / x + x ^ 2 = 1

-- Theorem statement: Prove which equation is a quadratic equation in one variable
theorem correct_answer_is_B : is_quadratic_in_one_variable eqB :=
sorry  -- Proof is not required as per the instructions

end correct_answer_is_B_l134_134657


namespace sin_cos_eq_sqrt2_l134_134965

theorem sin_cos_eq_sqrt2 (x : ℝ) (h0 : 0 ≤ x) (h1 : x ≤ 2 * Real.pi) (h2 : Real.sin x - Real.cos x = Real.sqrt 2) :
  x = (3 * Real.pi) / 4 :=
sorry

end sin_cos_eq_sqrt2_l134_134965


namespace all_points_lie_on_circle_l134_134662

theorem all_points_lie_on_circle {s : ℝ} :
  let x := (s^2 - 1) / (s^2 + 1)
  let y := (2 * s) / (s^2 + 1)
  x^2 + y^2 = 1 :=
by
  sorry

end all_points_lie_on_circle_l134_134662


namespace parabola_directrix_y_neg1_l134_134751

-- We define the problem given the conditions.
def parabola_directrix (p : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 = 4 * y → y = -p

-- Now we state what needs to be proved.
theorem parabola_directrix_y_neg1 : parabola_directrix 1 :=
by
  sorry

end parabola_directrix_y_neg1_l134_134751


namespace all_fruits_sold_in_market_market_vs_orchard_income_diff_total_profit_l134_134572

-- Conditions
def investment := 13500  -- in yuan
def total_yield := 19000 -- in kg
def price_orchard := 4   -- in yuan/kg
def price_market (x : ℝ) := x -- in yuan/kg
def market_daily_sale := 1000 -- in kg/day

-- Part 1: Days to sell all fruits in the market
theorem all_fruits_sold_in_market (x : ℝ) (h : x > 4) : total_yield / market_daily_sale = 19 :=
by
  sorry

-- Part 2: Income difference between market and orchard sales
theorem market_vs_orchard_income_diff (x : ℝ) (h : x > 4) : total_yield * price_market x - total_yield * price_orchard = 19000 * x - 76000 :=
by
  sorry

-- Part 3: Total profit from selling partly in the orchard and partly in the market
theorem total_profit (x : ℝ) (h : x > 4) : 6000 * price_orchard + (total_yield - 6000) * price_market x - investment = 13000 * x + 10500 :=
by
  sorry

end all_fruits_sold_in_market_market_vs_orchard_income_diff_total_profit_l134_134572


namespace fettuccine_to_tortellini_ratio_l134_134718

-- Definitions based on the problem conditions
def total_students := 800
def preferred_spaghetti := 320
def preferred_fettuccine := 200
def preferred_tortellini := 160
def preferred_penne := 120

-- Theorem to prove that the ratio is 5/4
theorem fettuccine_to_tortellini_ratio :
  (preferred_fettuccine : ℚ) / (preferred_tortellini : ℚ) = 5 / 4 :=
sorry

end fettuccine_to_tortellini_ratio_l134_134718


namespace volume_maximized_at_r_5_h_8_l134_134284

noncomputable def V (r : ℝ) : ℝ := (Real.pi / 5) * (300 * r - 4 * r^3)

/-- (1) Given that the total construction cost is 12000π yuan, 
express the volume V as a function of the radius r, and determine its domain. -/
def volume_function (r : ℝ) (h : ℝ) (cost : ℝ) : Prop :=
  cost = 12000 * Real.pi ∧
  h = 1 / (5 * r) * (300 - 4 * r^2) ∧
  V r = Real.pi * r^2 * h ∧
  0 < r ∧ r < 5 * Real.sqrt 3

/-- (2) Prove V(r) is maximized when r = 5 and h = 8 -/
theorem volume_maximized_at_r_5_h_8 :
  ∀ (r : ℝ) (h : ℝ) (cost : ℝ), volume_function r h cost → 
  ∃ (r_max : ℝ) (h_max : ℝ), r_max = 5 ∧ h_max = 8 ∧ ∀ x, 0 < x → x < 5 * Real.sqrt 3 → V x ≤ V r_max :=
by
  intros r h cost hvolfunc
  sorry

end volume_maximized_at_r_5_h_8_l134_134284


namespace initial_earning_members_l134_134524

theorem initial_earning_members (average_income_before: ℝ) (average_income_after: ℝ) (income_deceased: ℝ) (n: ℝ)
    (H1: average_income_before = 735)
    (H2: average_income_after = 650)
    (H3: income_deceased = 990)
    (H4: n * average_income_before - (n - 1) * average_income_after = income_deceased)
    : n = 4 := 
by 
  rw [H1, H2, H3] at H4
  linarith


end initial_earning_members_l134_134524


namespace solve_for_x_l134_134512

namespace proof_problem

-- Define the operation a * b = 4 * a * b
def star (a b : ℝ) : ℝ := 4 * a * b

-- Given condition rewritten in terms of the operation star
def equation (x : ℝ) : Prop := star x x + star 2 x - star 2 4 = 0

-- The statement we intend to prove
theorem solve_for_x (x : ℝ) : equation x → (x = 2 ∨ x = -4) :=
by
  -- Proof omitted
  sorry

end proof_problem

end solve_for_x_l134_134512


namespace find_number_l134_134847

theorem find_number (number : ℝ) (h : 0.001 * number = 0.24) : number = 240 :=
sorry

end find_number_l134_134847


namespace best_player_total_hits_l134_134654

theorem best_player_total_hits
  (team_avg_hits_per_game : ℕ)
  (games_played : ℕ)
  (total_players : ℕ)
  (other_players_avg_hits_next_6_games : ℕ)
  (correct_answer : ℕ)
  (h1 : team_avg_hits_per_game = 15)
  (h2 : games_played = 5)
  (h3 : total_players = 11)
  (h4 : other_players_avg_hits_next_6_games = 6)
  (h5 : correct_answer = 25) :
  ∃ total_hits_of_best_player : ℕ,
  total_hits_of_best_player = correct_answer := by
  sorry

end best_player_total_hits_l134_134654


namespace min_value_eq_ab_squared_l134_134187

noncomputable def min_value (x a b : ℝ) : ℝ := 1 / (x^a * (1 - x)^b)

theorem min_value_eq_ab_squared (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) :
  ∃ x, 0 < x ∧ x < 1 ∧ min_value x a b = (a + b)^2 :=
by
  sorry

end min_value_eq_ab_squared_l134_134187


namespace triangle_right_angle_l134_134999

theorem triangle_right_angle (α β γ : ℝ) (h1 : α + β + γ = 180) (h2 : γ = α + β) : γ = 90 :=
by
  sorry

end triangle_right_angle_l134_134999


namespace infinite_solutions_l134_134592

theorem infinite_solutions (b : ℝ) :
  (∀ x : ℝ, 4 * (3 * x - b) = 3 * (4 * x + 16)) ↔ b = -12 :=
by
  sorry

end infinite_solutions_l134_134592


namespace shortest_distance_between_stations_l134_134660

/-- 
Given two vehicles A and B shuttling between two locations,
with Vehicle A stopping every 0.5 kilometers and Vehicle B stopping every 0.8 kilometers,
prove that the shortest distance between two stations where Vehicles A and B do not stop at the same place is 0.1 kilometers.
-/
theorem shortest_distance_between_stations :
  ∀ (dA dB : ℝ), (dA = 0.5) → (dB = 0.8) → ∃ δ : ℝ, (δ = 0.1) ∧ (∀ n m : ℕ, dA * n ≠ dB * m → abs ((dA * n) - (dB * m)) = δ) :=
by
  intros dA dB hA hB
  use 0.1
  sorry

end shortest_distance_between_stations_l134_134660


namespace fraction_of_x_l134_134208

theorem fraction_of_x (w x y f : ℝ) (h1 : 2 / w + f * x = 2 / y) (h2 : w * x = y) (h3 : (w + x) / 2 = 0.5) : f = 2 / x - 2 := 
sorry

end fraction_of_x_l134_134208


namespace find_hansol_weight_l134_134904

variable (H : ℕ)

theorem find_hansol_weight (h : H + (H + 4) = 88) : H = 42 :=
by
  sorry

end find_hansol_weight_l134_134904


namespace simplify_expression_l134_134968

theorem simplify_expression (w : ℝ) : 3 * w^2 + 6 * w^2 + 9 * w^2 + 12 * w^2 + 15 * w^2 + 24 = 45 * w^2 + 24 :=
by
  sorry

end simplify_expression_l134_134968


namespace find_m_parallel_l134_134533

def vector_parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = (k • v) ∨ v = (k • u)

theorem find_m_parallel (m : ℝ) (a b : ℝ × ℝ) (h_a : a = (-1, 1)) (h_b : b = (3, m)) 
  (h_parallel : vector_parallel a (a.1 + b.1, a.2 + b.2)) : m = -3 := 
by 
  sorry

end find_m_parallel_l134_134533


namespace factor_expression_l134_134434

theorem factor_expression (b : ℝ) : 56 * b^3 + 168 * b^2 = 56 * b^2 * (b + 3) :=
by
  sorry

end factor_expression_l134_134434


namespace shelves_of_picture_books_l134_134540

-- Define the conditions
def n_mystery : ℕ := 5
def b_per_shelf : ℕ := 4
def b_total : ℕ := 32

-- State the main theorem to be proven
theorem shelves_of_picture_books :
  (b_total - n_mystery * b_per_shelf) / b_per_shelf = 3 :=
by
  -- The proof is omitted
  sorry

end shelves_of_picture_books_l134_134540


namespace total_bird_families_l134_134240

-- Declare the number of bird families that flew to Africa
def a : Nat := 47

-- Declare the number of bird families that flew to Asia
def b : Nat := 94

-- Condition that Asia's number of bird families matches Africa + 47 more
axiom h : b = a + 47

-- Prove the total number of bird families is 141
theorem total_bird_families : a + b = 141 :=
by
  -- Insert proof here
  sorry

end total_bird_families_l134_134240


namespace prove_monomial_l134_134447

-- Definitions and conditions from step a)
def like_terms (x y : ℕ) := 
  x = 2 ∧ x + y = 5

-- Main statement to be proved
theorem prove_monomial (x y : ℕ) (h : like_terms x y) : 
  1 / 2 * x^3 - 1 / 6 * x * y^2 = 1 :=
by
  sorry

end prove_monomial_l134_134447


namespace hypotenuse_length_l134_134078

theorem hypotenuse_length (a b c : ℝ) (h1: a^2 + b^2 + c^2 = 2500) (h2: c^2 = a^2 + b^2) : 
  c = 25 * Real.sqrt 10 := 
sorry

end hypotenuse_length_l134_134078


namespace cookies_none_of_ingredients_l134_134439

theorem cookies_none_of_ingredients (c : ℕ) (o : ℕ) (r : ℕ) (a : ℕ) (total_cookies : ℕ) :
  total_cookies = 48 ∧ c = total_cookies / 3 ∧ o = (3 * total_cookies + 4) / 5 ∧ r = total_cookies / 2 ∧ a = total_cookies / 8 → 
  ∃ n, n = 19 ∧ (∀ k, k = total_cookies - max c (max o (max r a)) → k ≤ n) :=
by sorry

end cookies_none_of_ingredients_l134_134439


namespace ceil_sub_self_eq_half_l134_134494

theorem ceil_sub_self_eq_half (n : ℤ) (x : ℝ) (h : x = n + 1/2) : ⌈x⌉ - x = 1/2 :=
by
  sorry

end ceil_sub_self_eq_half_l134_134494


namespace intersection_setA_setB_l134_134288

namespace Proof

def setA : Set ℝ := {x | ∃ y : ℝ, y = x + 1}
def setB : Set ℝ := {y | ∃ x : ℝ, y = 2^x}

theorem intersection_setA_setB : (setA ∩ setB) = {y | 0 < y} :=
by
  sorry

end Proof

end intersection_setA_setB_l134_134288


namespace equalize_foma_ierema_l134_134191

theorem equalize_foma_ierema (F E Y : ℕ) 
  (h1 : F - 70 = E + 70) 
  (h2 : F - 40 = Y) 
  (h3 : Y = E + 70) 
  : ∃ x : ℕ, x = 55 ∧ F - x = E + x :=
by
  use 55
  sorry

end equalize_foma_ierema_l134_134191


namespace parabola_line_intersection_distance_l134_134998

theorem parabola_line_intersection_distance :
  ∀ (x y : ℝ), x^2 = -4 * y ∧ y = x - 1 ∧ x^2 + 4 * x + 4 = 0 →
  abs (y - -1 + (-1 - y)) = 8 :=
by
  sorry

end parabola_line_intersection_distance_l134_134998


namespace factor_1_factor_2_triangle_is_isosceles_l134_134811

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

end factor_1_factor_2_triangle_is_isosceles_l134_134811


namespace find_x_l134_134561

theorem find_x (x : ℝ) (h : 0.25 * x = 0.10 * 500 - 5) : x = 180 :=
by
  sorry

end find_x_l134_134561


namespace geometric_sequence_vertex_property_l134_134941

theorem geometric_sequence_vertex_property (a b c d : ℝ) 
  (h_geom : ∃ r : ℝ, b = a * r ∧ c = b * r ∧ d = c * r)
  (h_vertex : b = 1 ∧ c = 2) : a * d = b * c :=
by sorry

end geometric_sequence_vertex_property_l134_134941


namespace restaurant_bill_split_l134_134111

def original_bill : ℝ := 514.16
def tip_rate : ℝ := 0.18
def number_of_people : ℕ := 9
def final_amount_per_person : ℝ := 67.41

theorem restaurant_bill_split :
  final_amount_per_person = (1 + tip_rate) * original_bill / number_of_people :=
by
  sorry

end restaurant_bill_split_l134_134111


namespace find_x_l134_134584

theorem find_x (x : ℝ) (hx : x > 0) (h : Real.sqrt (12*x) * Real.sqrt (5*x) * Real.sqrt (7*x) * Real.sqrt (21*x) = 21) : 
  x = 21 / 97 :=
by
  sorry

end find_x_l134_134584


namespace part_one_part_two_l134_134636

def f (a x : ℝ) : ℝ := |a - 4 * x| + |2 * a + x|

theorem part_one (x : ℝ) : f 1 x ≥ 3 ↔ x ≤ 0 ∨ x ≥ 2 / 5 := 
sorry

theorem part_two (a x : ℝ) : f a x + f a (-1 / x) ≥ 10 := 
sorry

end part_one_part_two_l134_134636


namespace inequality_solution_l134_134723

theorem inequality_solution (x : ℝ) :
  (x > -4 ∧ x < -5 / 3) ↔ 
  (2 * x + 3) / (3 * x + 5) > (4 * x + 1) / (x + 4) := 
sorry

end inequality_solution_l134_134723


namespace raghu_investment_approx_l134_134767

-- Define the investments
def investments (R : ℝ) : Prop :=
  let Trishul := 0.9 * R
  let Vishal := 0.99 * R
  let Deepak := 1.188 * R
  R + Trishul + Vishal + Deepak = 8578

-- State the theorem to prove that Raghu invested approximately Rs. 2103.96
theorem raghu_investment_approx : 
  ∃ R : ℝ, investments R ∧ abs (R - 2103.96) < 1 :=
by
  sorry

end raghu_investment_approx_l134_134767


namespace range_of_m_l134_134556

theorem range_of_m (m : ℝ) (h : 1 < m) :
  (∀ x : ℝ, -3 ≤ x ∧ x ≤ 1 → -m ≤ x ∧ x ≤ m - 1) → (3 ≤ m) :=
by
  sorry  -- The proof will be constructed here.

end range_of_m_l134_134556


namespace smallest_n_integer_price_l134_134437

theorem smallest_n_integer_price (p : ℚ) (h : ∃ x : ℕ, p = x ∧ 1.06 * p = n) : n = 53 :=
sorry

end smallest_n_integer_price_l134_134437


namespace wrench_force_l134_134373

theorem wrench_force (F L k: ℝ) (h_inv: ∀ F L, F * L = k) (h_given: F * 12 = 240 * 12) : 
  (∀ L, (L = 16) → (F = 180)) ∧ (∀ L, (L = 8) → (F = 360)) := by 
sorry

end wrench_force_l134_134373


namespace not_square_n5_plus_7_l134_134885

theorem not_square_n5_plus_7 (n : ℕ) (h : n > 1) : ¬ ∃ k : ℕ, k^2 = n^5 + 7 := 
by
  sorry

end not_square_n5_plus_7_l134_134885


namespace solve_x_l134_134948

theorem solve_x : ∃ x : ℝ, 2^(Real.log 5 / Real.log 2) = 3 * x + 4 ∧ x = 1 / 3 :=
by
  use 1 / 3
  sorry

end solve_x_l134_134948


namespace common_ratio_geometric_sequence_l134_134668

noncomputable def a (n : ℕ) : ℝ := sorry
noncomputable def S (n : ℕ) : ℝ := sorry

theorem common_ratio_geometric_sequence
  (a3_eq : a 3 = 2 * S 2 + 1)
  (a4_eq : a 4 = 2 * S 3 + 1)
  (geometric_seq : ∀ n, a (n+1) = a 1 * (q ^ n))
  (h₀ : a 1 ≠ 0)
  (h₁ : q ≠ 0) :
  q = 3 :=
sorry

end common_ratio_geometric_sequence_l134_134668


namespace difference_divisible_l134_134037

theorem difference_divisible (a b n : ℕ) (h : n % 2 = 0) (hab : a + b = 61) :
  (47^100 - 14^100) % 61 = 0 := by
  sorry

end difference_divisible_l134_134037


namespace find_multiple_l134_134357

theorem find_multiple (x m : ℕ) (h₁ : x = 69) (h₂ : x - 18 = m * (86 - x)) : m = 3 :=
by
  sorry

end find_multiple_l134_134357


namespace box_length_l134_134869

theorem box_length :
  ∃ (length : ℝ), 
  let box_height := 8
  let box_width := 10
  let block_height := 3
  let block_width := 2
  let block_length := 4
  let num_blocks := 40
  let box_volume := box_height * box_width * length
  let block_volume := block_height * block_width * block_length
  num_blocks * block_volume = box_volume ∧ length = 12 := by
  sorry

end box_length_l134_134869


namespace green_tractor_price_l134_134030

-- Define the conditions
def salary_based_on_sales (r_ct : Nat) (r_price : ℝ) (g_ct : Nat) (g_price : ℝ) : ℝ :=
  0.1 * r_ct * r_price + 0.2 * g_ct * g_price

-- Define the problem's Lean statement
theorem green_tractor_price
  (r_ct : Nat) (g_ct : Nat)
  (r_price : ℝ) (total_salary : ℝ)
  (h_rct : r_ct = 2)
  (h_gct : g_ct = 3)
  (h_rprice : r_price = 20000)
  (h_salary : total_salary = 7000) :
  ∃ g_price : ℝ, salary_based_on_sales r_ct r_price g_ct g_price = total_salary ∧ g_price = 5000 :=
by
  sorry

end green_tractor_price_l134_134030


namespace sum_base6_l134_134501

theorem sum_base6 : 
  ∀ (a b : ℕ) (h1 : a = 4532) (h2 : b = 3412),
  (a + b = 10414) :=
by
  intros a b h1 h2
  rw [h1, h2]
  sorry

end sum_base6_l134_134501


namespace angles_in_quadrilateral_l134_134347

theorem angles_in_quadrilateral (A B C D : ℝ)
    (h : A / B = 1 / 3 ∧ B / C = 3 / 5 ∧ C / D = 5 / 6)
    (sum_angles : A + B + C + D = 360) :
    A = 24 ∧ D = 144 := 
by
    sorry

end angles_in_quadrilateral_l134_134347


namespace revenue_for_recent_quarter_l134_134715

noncomputable def previous_year_revenue : ℝ := 85.0
noncomputable def percentage_fall : ℝ := 43.529411764705884
noncomputable def recent_quarter_revenue : ℝ := previous_year_revenue - (previous_year_revenue * (percentage_fall / 100))

theorem revenue_for_recent_quarter : recent_quarter_revenue = 48.0 := 
by 
  sorry -- Proof is skipped

end revenue_for_recent_quarter_l134_134715


namespace find_ratio_l134_134564

-- Definitions and conditions
def sides_form_right_triangle (x d : ℝ) : Prop :=
  x > d ∧ d > 0 ∧ (x^2 + (x^2 - d)^2 = (x^2 + d)^2)

-- The theorem stating the required ratio
theorem find_ratio (x d : ℝ) (h : sides_form_right_triangle x d) : 
  x / d = 8 :=
by
  sorry

end find_ratio_l134_134564


namespace rectangular_box_in_sphere_radius_l134_134092

theorem rectangular_box_in_sphere_radius (a b c s : ℝ) 
  (h1 : a + b + c = 40) 
  (h2 : 2 * a * b + 2 * b * c + 2 * a * c = 608) 
  (h3 : (2 * s)^2 = a^2 + b^2 + c^2) : 
  s = 16 * Real.sqrt 2 :=
by
  sorry

end rectangular_box_in_sphere_radius_l134_134092


namespace parabola_directrix_l134_134328

theorem parabola_directrix (x : ℝ) :
  (∃ y : ℝ, y = (x^2 - 8*x + 12) / 16) →
  ∃ directrix : ℝ, directrix = -17 / 4 :=
by
  sorry

end parabola_directrix_l134_134328


namespace algebra_books_cannot_be_determined_uniquely_l134_134676

theorem algebra_books_cannot_be_determined_uniquely (A H S M E : ℕ) (pos_A : A > 0) (pos_H : H > 0) (pos_S : S > 0) 
  (pos_M : M > 0) (pos_E : E > 0) (distinct : A ≠ H ∧ A ≠ S ∧ A ≠ M ∧ A ≠ E ∧ H ≠ S ∧ H ≠ M ∧ H ≠ E ∧ S ≠ M ∧ S ≠ E ∧ M ≠ E) 
  (cond1: S < A) (cond2: M > H) (cond3: A + 2 * H = S + 2 * M) : 
  E = 0 :=
sorry

end algebra_books_cannot_be_determined_uniquely_l134_134676


namespace determine_s_l134_134606

theorem determine_s 
  (s : ℝ) 
  (h : (3 * x^3 - 2 * x^2 + x + 6) * (2 * x^3 + s * x^2 + 3 * x + 5) =
       6 * x^6 + s * x^5 + 5 * x^4 + 17 * x^3 + 10 * x^2 + 33 * x + 30) : 
  s = 4 :=
by
  sorry

end determine_s_l134_134606


namespace speed_of_stream_l134_134107

theorem speed_of_stream (vs : ℝ) (h : ∀ (d : ℝ), d / (57 - vs) = 2 * (d / (57 + vs))) : vs = 19 :=
by
  sorry

end speed_of_stream_l134_134107


namespace age_difference_l134_134577

variable (Patrick_age Michael_age Monica_age : ℕ)

theorem age_difference 
  (h1 : ∃ x : ℕ, Patrick_age = 3 * x ∧ Michael_age = 5 * x)
  (h2 : ∃ y : ℕ, Michael_age = 3 * y ∧ Monica_age = 5 * y)
  (h3 : Patrick_age + Michael_age + Monica_age = 245) :
  Monica_age - Patrick_age = 80 := by 
sorry

end age_difference_l134_134577


namespace number_of_answer_choices_l134_134588

theorem number_of_answer_choices (n : ℕ) (H1 : (n + 1)^4 = 625) : n = 4 :=
sorry

end number_of_answer_choices_l134_134588


namespace Kimberley_collected_10_pounds_l134_134884

variable (K H E total : ℝ)

theorem Kimberley_collected_10_pounds (h_total : total = 35) (h_Houston : H = 12) (h_Ela : E = 13) :
    K + H + E = total → K = 10 :=
by
  intros h_sum
  rw [h_Houston, h_Ela] at h_sum
  linarith

end Kimberley_collected_10_pounds_l134_134884


namespace range_of_a_l134_134108

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 < x ∧ x < 4 → a < x ∧ x < 5) → a ≤ 1 := 
sorry

end range_of_a_l134_134108


namespace compute_expression_l134_134810

theorem compute_expression : 3 * 3^4 - 9^19 / 9^17 = 162 := by
  sorry

end compute_expression_l134_134810


namespace total_games_played_l134_134125

-- Define the conditions as Lean 4 definitions
def games_won : Nat := 12
def games_lost : Nat := 4

-- Prove the total number of games played is 16
theorem total_games_played : games_won + games_lost = 16 := 
by
  -- Place a proof placeholder
  sorry

end total_games_played_l134_134125


namespace find_a_b_and_m_range_l134_134329

-- Definitions and initial conditions
def f (x : ℝ) (a b m : ℝ) : ℝ := 2*x^3 + a*x^2 + b*x + m
def f_prime (x : ℝ) (a b : ℝ) : ℝ := 6*x^2 + 2*a*x + b

-- Problem statement
theorem find_a_b_and_m_range (a b m : ℝ) :
  (∀ x, f_prime x a b = 6 * (x + 0.5)^2 - k) →
  f_prime 1 a b = 0 →
  a = 3 ∧ b = -12 ∧ -20 < m ∧ m < 7 :=
sorry

end find_a_b_and_m_range_l134_134329


namespace part_one_part_two_l134_134141

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 5 then (16 / (9 - x) - 1) else (11 - (2 / 45) * x ^ 2)

theorem part_one (k : ℝ) (h : 1 ≤ k ∧ k ≤ 4) : k * (16 / (9 - 3) - 1) = 4 → k = 12 / 5 :=
by sorry

theorem part_two (y x : ℝ) (h_y : y = 4) :
  (1 ≤ x ∧ x ≤ 5 ∧ 4 * (16 / (9 - x) - 1) ≥ 4) ∨
  (5 < x ∧ x ≤ 15 ∧ 4 * (11 - (2/45) * x ^ 2) ≥ 4) :=
by sorry

end part_one_part_two_l134_134141


namespace ny_mets_fans_count_l134_134379

-- Define the known ratios and total fans
def ratio_Y_to_M (Y M : ℕ) : Prop := 3 * M = 2 * Y
def ratio_M_to_R (M R : ℕ) : Prop := 4 * R = 5 * M
def total_fans (Y M R : ℕ) : Prop := Y + M + R = 330

-- Define what we want to prove
theorem ny_mets_fans_count (Y M R : ℕ) (h1 : ratio_Y_to_M Y M) (h2 : ratio_M_to_R M R) (h3 : total_fans Y M R) : M = 88 :=
sorry

end ny_mets_fans_count_l134_134379


namespace seeds_in_big_garden_l134_134830

-- Definitions based on conditions
def total_seeds : ℕ := 42
def small_gardens : ℕ := 3
def seeds_per_small_garden : ℕ := 2
def seeds_planted_in_small_gardens : ℕ := small_gardens * seeds_per_small_garden

-- Proof statement
theorem seeds_in_big_garden : total_seeds - seeds_planted_in_small_gardens = 36 :=
sorry

end seeds_in_big_garden_l134_134830


namespace line_equation_of_point_and_slope_angle_l134_134218

theorem line_equation_of_point_and_slope_angle 
  (p : ℝ × ℝ) (θ : ℝ)
  (h₁ : p = (-1, 2))
  (h₂ : θ = 45) :
  ∃ (a b c : ℝ), a * (p.1) + b * (p.2) + c = 0 ∧ (a * 1 + b * 1 = c) :=
sorry

end line_equation_of_point_and_slope_angle_l134_134218


namespace find_product_abcd_l134_134598

def prod_abcd (a b c d : ℚ) :=
  4 * a - 2 * b + 3 * c + 5 * d = 22 ∧
  2 * (d + c) = b - 2 ∧
  4 * b - c = a + 1 ∧
  c + 1 = 2 * d

theorem find_product_abcd (a b c d : ℚ) (h : prod_abcd a b c d) :
  a * b * c * d = -30751860 / 11338912 :=
sorry

end find_product_abcd_l134_134598


namespace rational_root_of_polynomial_l134_134849

-- Polynomial definition
def P (x : ℚ) : ℚ := 3 * x^4 - 7 * x^3 + 4 * x^2 + 6 * x - 8

-- Theorem statement
theorem rational_root_of_polynomial : ∀ x : ℚ, P x = 0 ↔ x = -1 :=
by
  sorry

end rational_root_of_polynomial_l134_134849


namespace total_bricks_proof_l134_134952

-- Define the initial conditions
def initial_courses := 3
def bricks_per_course := 400
def additional_courses := 2

-- Compute the number of bricks removed from the last course
def bricks_removed_from_last_course (bricks_per_course: ℕ) : ℕ :=
  bricks_per_course / 2

-- Calculate the total number of bricks
def total_bricks (initial_courses : ℕ) (bricks_per_course : ℕ) (additional_courses : ℕ) (bricks_removed : ℕ) : ℕ :=
  (initial_courses + additional_courses) * bricks_per_course - bricks_removed

-- Given values and the proof problem
theorem total_bricks_proof :
  total_bricks initial_courses bricks_per_course additional_courses (bricks_removed_from_last_course bricks_per_course) = 1800 :=
by
  sorry

end total_bricks_proof_l134_134952


namespace jerry_weighted_mean_l134_134971

noncomputable def weighted_mean (aunt uncle sister cousin friend1 friend2 friend3 friend4 friend5 : ℝ)
    (eur_to_usd gbp_to_usd cad_to_usd : ℝ) (family_weight friends_weight : ℝ) : ℝ :=
  let uncle_usd := uncle * eur_to_usd
  let friend3_usd := friend3 * eur_to_usd
  let friend4_usd := friend4 * gbp_to_usd
  let cousin_usd := cousin * cad_to_usd
  let family_sum := aunt + uncle_usd + sister + cousin_usd
  let friends_sum := friend1 + friend2 + friend3_usd + friend4_usd + friend5
  family_sum * family_weight + friends_sum * friends_weight

theorem jerry_weighted_mean : 
  weighted_mean 9.73 9.43 7.25 20.37 22.16 23.51 18.72 15.53 22.84 
               1.20 1.38 0.82 0.40 0.60 = 85.4442 := 
sorry

end jerry_weighted_mean_l134_134971


namespace num_common_tangents_l134_134319

-- Define the first circle
def circle1 (x y : ℝ) : Prop := (x + 2) ^ 2 + y ^ 2 = 4
-- Define the second circle
def circle2 (x y : ℝ) : Prop := (x - 2) ^ 2 + (y - 1) ^ 2 = 9

-- Prove that the number of common tangent lines between the given circles is 2
theorem num_common_tangents : ∃ (n : ℕ), n = 2 ∧
  -- The circles do not intersect nor are they internally tangent
  (∀ (x y : ℝ), ¬(circle1 x y ∧ circle2 x y) ∧ 
  -- There exist exactly n common tangent lines
  ∃ (C : ℕ), C = n) :=
sorry

end num_common_tangents_l134_134319


namespace kevin_total_distance_l134_134867

def v1 : ℝ := 10
def t1 : ℝ := 0.5
def v2 : ℝ := 20
def t2 : ℝ := 0.5
def v3 : ℝ := 8
def t3 : ℝ := 0.25

theorem kevin_total_distance : v1 * t1 + v2 * t2 + v3 * t3 = 17 := by
  sorry

end kevin_total_distance_l134_134867


namespace simplify_root_product_l134_134694

theorem simplify_root_product : 
  (625:ℝ)^(1/4) * (125:ℝ)^(1/3) = 25 :=
by
  have h₁ : (625:ℝ) = 5^4 := by norm_num
  have h₂ : (125:ℝ) = 5^3 := by norm_num
  rw [h₁, h₂]
  norm_num
  sorry

end simplify_root_product_l134_134694


namespace total_games_for_18_players_l134_134984

-- Define the number of players
def num_players : ℕ := 18

-- Define the function to calculate total number of games
def total_games (n : ℕ) : ℕ := n * (n - 1) * 2

-- Theorem statement asserting the total number of games for 18 players
theorem total_games_for_18_players : total_games num_players = 612 :=
by
  -- proof goes here
  sorry

end total_games_for_18_players_l134_134984


namespace championship_outcomes_l134_134596

theorem championship_outcomes :
  ∀ (students events : ℕ), students = 4 → events = 3 → students ^ events = 64 :=
by
  intros students events h_students h_events
  rw [h_students, h_events]
  exact rfl

end championship_outcomes_l134_134596


namespace percentage_of_profit_if_no_discount_l134_134493

-- Conditions
def discount : ℝ := 0.05
def profit_w_discount : ℝ := 0.216
def cost_price : ℝ := 100
def expected_profit : ℝ := 28

-- Proof statement
theorem percentage_of_profit_if_no_discount :
  ∃ (marked_price selling_price_no_discount : ℝ),
    selling_price_no_discount = marked_price ∧
    (marked_price - cost_price) / cost_price * 100 = expected_profit :=
by
  -- Definitions and logic will go here
  sorry

end percentage_of_profit_if_no_discount_l134_134493


namespace original_cost_of_pencil_l134_134787

theorem original_cost_of_pencil (final_price discount: ℝ) (h_final: final_price = 3.37) (h_disc: discount = 0.63) : 
  final_price + discount = 4 :=
by
  sorry

end original_cost_of_pencil_l134_134787


namespace carols_father_gave_5_peanuts_l134_134331

theorem carols_father_gave_5_peanuts : 
  ∀ (c: ℕ) (f: ℕ), c = 2 → c + f = 7 → f = 5 :=
by
  intros c f h1 h2
  sorry

end carols_father_gave_5_peanuts_l134_134331


namespace speed_in_still_water_l134_134553

/-- A man can row upstream at 37 km/h and downstream at 53 km/h, 
    prove that the speed of the man in still water is 45 km/h. --/
theorem speed_in_still_water 
  (upstream_speed : ℕ) 
  (downstream_speed : ℕ)
  (h1 : upstream_speed = 37)
  (h2 : downstream_speed = 53) : 
  (upstream_speed + downstream_speed) / 2 = 45 := 
by 
  sorry

end speed_in_still_water_l134_134553


namespace wrestler_teams_possible_l134_134070

theorem wrestler_teams_possible :
  ∃ (team1 team2 team3 : Finset ℕ),
  (team1 ∪ team2 ∪ team3 = {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧
  (team1 ∩ team2 = ∅) ∧ (team1 ∩ team3 = ∅) ∧ (team2 ∩ team3 = ∅) ∧
  (team1.card = 3) ∧ (team2.card = 3) ∧ (team3.card = 3) ∧
  (team1.sum id = 15) ∧ (team2.sum id = 15) ∧ (team3.sum id = 15) ∧
  (∀ x ∈ team1, ∀ y ∈ team2, x > y) ∧
  (∀ x ∈ team2, ∀ y ∈ team3, x > y) ∧
  (∀ x ∈ team3, ∀ y ∈ team1, x > y) := sorry

end wrestler_teams_possible_l134_134070


namespace mushroom_mass_decrease_l134_134033

theorem mushroom_mass_decrease :
  ∀ (initial_mass water_content_fresh water_content_dry : ℝ),
  water_content_fresh = 0.8 →
  water_content_dry = 0.2 →
  (initial_mass * (1 - water_content_fresh) / (1 - water_content_dry) = initial_mass * 0.25) →
  (initial_mass - initial_mass * 0.25) / initial_mass = 0.75 :=
by
  intros initial_mass water_content_fresh water_content_dry h_fresh h_dry h_dry_mass
  sorry

end mushroom_mass_decrease_l134_134033


namespace point_of_tangency_l134_134075

noncomputable def f (x a : ℝ) : ℝ := Real.exp x + a * Real.exp (-x)
noncomputable def f_deriv (x a : ℝ) : ℝ := Real.exp x - a * Real.exp (-x)

theorem point_of_tangency (a : ℝ) (h1 : ∀ x, f_deriv (-x) a = -f_deriv x a)
  (h2 : ∃ x0, f_deriv x0 1 = 3/2) :
  ∃ x0 y0, x0 = Real.log 2 ∧ y0 = f (Real.log 2) 1 ∧ y0 = 5/2 :=
by
  sorry

end point_of_tangency_l134_134075


namespace find_natrual_numbers_l134_134617

theorem find_natrual_numbers (k n : ℕ) (A B : Matrix (Fin n) (Fin n) ℤ) 
  (h1 : k ≥ 1) 
  (h2 : n ≥ 2) 
  (h3 : A ^ 3 = 0) 
  (h4 : A ^ k * B + B * A = 1) : 
  k = 1 ∧ Even n := 
sorry

end find_natrual_numbers_l134_134617


namespace marble_price_proof_l134_134772

noncomputable def price_per_colored_marble (total_marbles white_percentage black_percentage white_price black_price total_earnings : ℕ) : ℕ :=
  let white_marbles := total_marbles * white_percentage / 100
  let black_marbles := total_marbles * black_percentage / 100
  let colored_marbles := total_marbles - (white_marbles + black_marbles)
  let earnings_from_white := white_marbles * white_price
  let earnings_from_black := black_marbles * black_price
  let earnings_from_colored := total_earnings - (earnings_from_white + earnings_from_black)
  earnings_from_colored / colored_marbles

theorem marble_price_proof : price_per_colored_marble 100 20 30 5 10 1400 = 20 := 
sorry

end marble_price_proof_l134_134772


namespace tangent_line_equation_l134_134977

theorem tangent_line_equation :
  ∀ (x : ℝ) (y : ℝ), y = 4 * x - x^3 → 
  (x = -1) → (y = -3) →
  (∀ (m : ℝ), m = 4 - 3 * (-1)^2) →
  ∃ (line_eq : ℝ → ℝ), (∀ x, line_eq x = x - 2) :=
by
  sorry

end tangent_line_equation_l134_134977


namespace inequality_proof_l134_134435

theorem inequality_proof (x1 x2 x3 : ℝ) (hx1 : 0 < x1) (hx2 : 0 < x2) (hx3 : 0 < x3) :
  (x1^2 + x2^2 + x3^2)^3 / (x1^3 + x2^3 + x3^3)^2 ≤ 3 :=
sorry

end inequality_proof_l134_134435


namespace geometric_sequence_inequality_l134_134945

variable (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ q : ℝ)

-- Conditions
def geometric_sequence (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ q : ℝ) : Prop :=
  a₂ = a₁ * q ∧
  a₃ = a₁ * q^2 ∧
  a₄ = a₁ * q^3 ∧
  a₅ = a₁ * q^4 ∧
  a₆ = a₁ * q^5 ∧
  a₇ = a₁ * q^6 ∧
  a₈ = a₁ * q^7

theorem geometric_sequence_inequality
  (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ q : ℝ)
  (h_seq : geometric_sequence a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ q)
  (h_a₁_pos : 0 < a₁)
  (h_q_ne_1 : q ≠ 1) :
  a₁ + a₈ > a₄ + a₅ :=
by 
-- Proof omitted
sorry

end geometric_sequence_inequality_l134_134945


namespace find_x2_plus_y2_l134_134841

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

end find_x2_plus_y2_l134_134841


namespace probability_factor_of_36_is_1_over_4_l134_134142

def totalPositiveIntegers := 36

def isDivisor (n: Nat) : Prop := ∃ (a b: Nat), (0 ≤ a ∧ a ≤ 2) ∧ (0 ≤ b ∧ b ≤ 2) ∧ (n = 2^a * 3^b)

def numDivisors := 9
def probability := (numDivisors : ℚ) / (totalPositiveIntegers : ℚ)

theorem probability_factor_of_36_is_1_over_4 :
  probability = (1 / 4 : ℚ) :=
sorry

end probability_factor_of_36_is_1_over_4_l134_134142


namespace circles_ACD_and_BCD_orthogonal_l134_134131

-- Define mathematical objects and conditions
variables (A B C D : Point) -- Points in general position on the plane
variables (circle : Point → Point → Point → Circle)

-- Circles intersect orthogonally property
def orthogonal_intersection (c1 c2 : Circle) : Prop :=
  -- Definition of orthogonal intersection of circles goes here (omitted for brevity)
  sorry

-- Given conditions
def circles_ABC_and_ABD_orthogonal : Prop :=
  orthogonal_intersection (circle A B C) (circle A B D)

-- Theorem statement
theorem circles_ACD_and_BCD_orthogonal (h : circles_ABC_and_ABD_orthogonal A B C D circle) :
  orthogonal_intersection (circle A C D) (circle B C D) :=
sorry

end circles_ACD_and_BCD_orthogonal_l134_134131


namespace nine_possible_xs_l134_134266

theorem nine_possible_xs :
  ∃! (x : ℕ) (h1 : 1 ≤ x) (h2 : x ≤ 33) (h3 : 25 ≤ x),
    ∀ n, (1 ≤ n ∧ n ≤ 3 → n * x < 100 ∧ (n + 1) * x ≥ 100) :=
sorry

end nine_possible_xs_l134_134266


namespace mark_total_spending_l134_134670

theorem mark_total_spending:
  let cost_per_pound_tomatoes := 5
  let pounds_tomatoes := 2
  let cost_per_pound_apples := 6
  let pounds_apples := 5
  let cost_tomatoes := cost_per_pound_tomatoes * pounds_tomatoes
  let cost_apples := cost_per_pound_apples * pounds_apples
  let total_spending := cost_tomatoes + cost_apples
  total_spending = 40 :=
by
  sorry

end mark_total_spending_l134_134670


namespace range_of_f_l134_134327

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then |x| - 1 else Real.sin x ^ 2

theorem range_of_f : Set.range f = Set.Ioi (-1) := 
  sorry

end range_of_f_l134_134327


namespace gum_boxes_l134_134049

theorem gum_boxes (c s t g : ℕ) (h1 : c = 2) (h2 : s = 5) (h3 : t = 9) (h4 : c + s + g = t) : g = 2 := by
  sorry

end gum_boxes_l134_134049


namespace f_12_16_plus_f_16_12_l134_134054

noncomputable def f : ℕ × ℕ → ℕ :=
sorry

axiom ax1 : ∀ (x : ℕ), f (x, x) = x
axiom ax2 : ∀ (x y : ℕ), f (x, y) = f (y, x)
axiom ax3 : ∀ (x y : ℕ), (x + y) * f (x, y) = y * f (x, x + y)

theorem f_12_16_plus_f_16_12 : f (12, 16) + f (16, 12) = 96 :=
by sorry

end f_12_16_plus_f_16_12_l134_134054


namespace vacation_cost_l134_134974

theorem vacation_cost (C : ℝ)
  (h1 : C / 5 - C / 8 = 60) :
  C = 800 :=
sorry

end vacation_cost_l134_134974


namespace perpendicular_chords_cosine_bound_l134_134124

theorem perpendicular_chords_cosine_bound 
  (a b : ℝ) 
  (h_ab : a > b) 
  (h_b0 : b > 0) 
  (θ1 θ2 : ℝ) 
  (x y : ℝ → ℝ) 
  (h_ellipse : ∀ t, x t = a * Real.cos t ∧ y t = b * Real.sin t) 
  (h_theta1 : ∃ t1, (x t1 = a * Real.cos θ1 ∧ y t1 = b * Real.sin θ1)) 
  (h_theta2 : ∃ t2, (x t2 = a * Real.cos θ2 ∧ y t2 = b * Real.sin θ2)) 
  (h_perpendicular: θ1 = θ2 + π / 2 ∨ θ1 = θ2 - π / 2) :
  0 ≤ |Real.cos (θ1 - θ2)| ∧ |Real.cos (θ1 - θ2)| ≤ (a ^ 2 - b ^ 2) / (a ^ 2 + b ^ 2) :=
sorry

end perpendicular_chords_cosine_bound_l134_134124


namespace other_root_of_equation_l134_134641

theorem other_root_of_equation (m : ℝ) :
  (∃ (x : ℝ), 3 * x^2 + m * x = -2 ∧ x = -1) →
  (∃ (y : ℝ), 3 * y^2 + m * y + 2 = 0 ∧ y = -(-2 / 3)) :=
by
  sorry

end other_root_of_equation_l134_134641


namespace greatest_third_side_l134_134183

theorem greatest_third_side
  (a b : ℕ)
  (h₁ : a = 7)
  (h₂ : b = 10)
  (c : ℕ)
  (h₃ : a + b + c ≤ 30)
  (h₄ : 3 < c)
  (h₅ : c ≤ 13) :
  c = 13 := 
sorry

end greatest_third_side_l134_134183


namespace grabbed_books_l134_134541

-- Definitions from conditions
def initial_books : ℕ := 99
def boxed_books : ℕ := 3 * 15
def room_books : ℕ := 21
def table_books : ℕ := 4
def kitchen_books : ℕ := 18
def current_books : ℕ := 23

-- Proof statement
theorem grabbed_books : (boxed_books + room_books + table_books + kitchen_books = initial_books - (23 - current_books)) → true := sorry

end grabbed_books_l134_134541


namespace maximal_regions_convex_quadrilaterals_l134_134468

theorem maximal_regions_convex_quadrilaterals (n : ℕ) (hn : n ≥ 1) : 
  ∃ a_n : ℕ, a_n = 4*n^2 - 4*n + 2 :=
by
  sorry

end maximal_regions_convex_quadrilaterals_l134_134468


namespace sufficient_but_not_necessary_condition_l134_134056

theorem sufficient_but_not_necessary_condition (b c : ℝ) :
  (∃ x0 : ℝ, (x0^2 + b * x0 + c) < 0) ↔ (c < 0) ∨ true :=
sorry

end sufficient_but_not_necessary_condition_l134_134056


namespace linear_eq_m_minus_2n_zero_l134_134318

theorem linear_eq_m_minus_2n_zero (m n : ℕ) (x y : ℝ) 
  (h1 : 2 * x ^ (m - 1) + 3 * y ^ (2 * n - 1) = 7)
  (h2 : m - 1 = 1) (h3 : 2 * n - 1 = 1) : 
  m - 2 * n = 0 := 
sorry

end linear_eq_m_minus_2n_zero_l134_134318


namespace jerry_can_escape_l134_134720

theorem jerry_can_escape (d : ℝ) (V_J V_T : ℝ) (h1 : (1 / 5) < d) (h2 : d < (1 / 4)) (h3 : V_T = 4 * V_J) :
  (4 * d) / V_J < 1 / (2 * V_J) :=
by
  sorry

end jerry_can_escape_l134_134720


namespace arithmeticSeqModulus_l134_134908

-- Define the arithmetic sequence
def arithmeticSeqSum (a d l : ℕ) : ℕ :=
  let n := (l - a) / d + 1
  n * (a + l) / 2

-- The main theorem to prove
theorem arithmeticSeqModulus : arithmeticSeqSum 2 5 102 % 20 = 12 := by
  sorry

end arithmeticSeqModulus_l134_134908


namespace solve_inequality_l134_134069

def solution_set_of_inequality : Set ℝ := {x : ℝ | -4 < x ∧ x < 2}

theorem solve_inequality (x : ℝ) (h : (2 - x) / (x + 4) > 0) : x ∈ solution_set_of_inequality :=
by
  sorry

end solve_inequality_l134_134069


namespace transformed_parabola_correct_l134_134693

def f (x : ℝ) : ℝ := (x + 2)^2 + 3
def g (x : ℝ) : ℝ := (x - 1)^2 + 1

theorem transformed_parabola_correct :
  ∀ x : ℝ, g x = f (x - 3) - 2 := by
  sorry

end transformed_parabola_correct_l134_134693


namespace simplify_and_evaluate_l134_134058

-- Definitions of given conditions
def a := 1
def b := 2

-- Statement of the theorem
theorem simplify_and_evaluate : (a * b + (a^2 - a * b) - (a^2 - 2 * a * b) = 4) :=
by
  -- Using sorry to indicate the proof is to be completed
  sorry

end simplify_and_evaluate_l134_134058


namespace numLinesTangentToCircles_eq_2_l134_134580

noncomputable def lineTangents (A B : Point) (dAB rA rB : ℝ) : ℕ :=
  if dAB < rA + rB then 2 else 0

theorem numLinesTangentToCircles_eq_2
  (A B : Point) (dAB rA rB : ℝ)
  (hAB : dAB = 4) (hA : rA = 3) (hB : rB = 2) :
  lineTangents A B dAB rA rB = 2 := by
  sorry

end numLinesTangentToCircles_eq_2_l134_134580


namespace negation_of_p_l134_134632

-- Define the proposition p
def p : Prop := ∀ x : ℝ, 2 * x^2 + 1 > 0

-- State the negation of p
theorem negation_of_p : ¬p ↔ ∃ x : ℝ, 2 * x^2 + 1 ≤ 0 := sorry

end negation_of_p_l134_134632


namespace reynald_soccer_balls_l134_134292

theorem reynald_soccer_balls (total_balls basketballs_more soccer tennis baseball more_baseballs volleyballs : ℕ) 
(h_total_balls: total_balls = 145) 
(h_basketballs_more: basketballs_more = 5)
(h_tennis: tennis = 2 * soccer)
(h_more_baseballs: more_baseballs = 10)
(h_volleyballs: volleyballs = 30) 
(sum_eq: soccer + (soccer + basketballs_more) + tennis + (soccer + more_baseballs) + volleyballs = total_balls) : soccer = 20 := 
by
  sorry

end reynald_soccer_balls_l134_134292


namespace find_c_l134_134939

theorem find_c (x c : ℤ) (h1 : 3 * x + 8 = 5) (h2 : c * x - 7 = 1) : c = -8 :=
sorry

end find_c_l134_134939


namespace miley_discount_rate_l134_134824

theorem miley_discount_rate :
  let cost_per_cellphone := 800
  let number_of_cellphones := 2
  let amount_paid := 1520
  let total_cost_without_discount := cost_per_cellphone * number_of_cellphones
  let discount_amount := total_cost_without_discount - amount_paid
  let discount_rate := (discount_amount / total_cost_without_discount) * 100
  discount_rate = 5 := by
    sorry

end miley_discount_rate_l134_134824


namespace inverse_propositions_l134_134890

-- Given conditions
lemma right_angles_equal : ∀ θ1 θ2 : ℝ, θ1 = θ2 → (θ1 = 90 ∧ θ2 = 90) :=
sorry

lemma equal_angles_right : ∀ θ1 θ2 : ℝ, (θ1 = 90 ∧ θ2 = 90) → (θ1 = θ2) :=
sorry

-- Theorem to be proven
theorem inverse_propositions :
  (∀ θ1 θ2 : ℝ, θ1 = θ2 → (θ1 = 90 ∧ θ2 = 90)) ↔
  (∀ θ1 θ2 : ℝ, (θ1 = 90 ∧ θ2 = 90) → (θ1 = θ2)) :=
sorry

end inverse_propositions_l134_134890


namespace total_hours_proof_l134_134440

-- Definitions and conditions
def kate_hours : ℕ := 22
def pat_hours : ℕ := 2 * kate_hours
def mark_hours : ℕ := kate_hours + 110

-- Statement of the proof problem
theorem total_hours_proof : pat_hours + kate_hours + mark_hours = 198 := by
  sorry

end total_hours_proof_l134_134440


namespace frame_width_proof_l134_134270

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

end frame_width_proof_l134_134270


namespace graph_passes_through_quadrants_l134_134164

theorem graph_passes_through_quadrants :
  (∃ x, x > 0 ∧ -1/2 * x + 2 > 0) ∧  -- Quadrant I condition: x > 0, y > 0
  (∃ x, x < 0 ∧ -1/2 * x + 2 > 0) ∧  -- Quadrant II condition: x < 0, y > 0
  (∃ x, x > 0 ∧ -1/2 * x + 2 < 0) := -- Quadrant IV condition: x > 0, y < 0
by
  sorry

end graph_passes_through_quadrants_l134_134164


namespace simplify_and_evaluate_expression_l134_134741

theorem simplify_and_evaluate_expression (x : ℤ) (h : x = -2) : 
  2 * x * (x - 3) - (x - 2) * (x + 1) = 16 :=
by
  sorry

end simplify_and_evaluate_expression_l134_134741


namespace spending_on_clothes_transport_per_month_l134_134378

noncomputable def monthly_spending_on_clothes_transport (S : ℝ) : ℝ :=
  0.2 * S

theorem spending_on_clothes_transport_per_month :
  ∃ (S : ℝ), (monthly_spending_on_clothes_transport S = 1584) ∧
             (12 * S - (12 * 0.6 * S + 12 * monthly_spending_on_clothes_transport S) = 19008) :=
by
  sorry

end spending_on_clothes_transport_per_month_l134_134378


namespace dodecahedron_interior_diagonals_l134_134613

-- Definition of a dodecahedron based on given conditions
structure Dodecahedron :=
  (vertices : ℕ)
  (faces : ℕ)
  (vertices_per_face : ℕ)
  (faces_per_vertex : ℕ)
  (interior_diagonals : ℕ)

-- Conditions provided in the problem
def dodecahedron : Dodecahedron :=
  { vertices := 20,
    faces := 12,
    vertices_per_face := 5,
    faces_per_vertex := 3,
    interior_diagonals := 130 }

-- The theorem to prove that given a dodecahedron structure, it has the correct number of interior diagonals
theorem dodecahedron_interior_diagonals (d : Dodecahedron) : d.interior_diagonals = 130 := by
  sorry

end dodecahedron_interior_diagonals_l134_134613


namespace find_mistake_l134_134817

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

end find_mistake_l134_134817


namespace arithmetic_sequence_sum_l134_134122

theorem arithmetic_sequence_sum
  (a1 : ℤ) (S : ℕ → ℤ) (d : ℤ)
  (H1 : a1 = -2017)
  (H2 : (S 2013 : ℤ) / 2013 - (S 2011 : ℤ) / 2011 = 2)
  (H3 : ∀ n : ℕ, S n = n * a1 + (n * (n - 1) / 2) * d) :
  S 2017 = -2017 :=
by
  sorry

end arithmetic_sequence_sum_l134_134122


namespace sum_of_numbers_l134_134115

theorem sum_of_numbers :
  2.12 + 0.004 + 0.345 = 2.469 :=
sorry

end sum_of_numbers_l134_134115


namespace sqrt_product_simplification_l134_134023

variable (q : ℝ)
variable (hq : q ≥ 0)

theorem sqrt_product_simplification : 
  (Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (3 * q)) = 21 * q * Real.sqrt (2 * q) := 
  sorry

end sqrt_product_simplification_l134_134023


namespace savings_by_going_earlier_l134_134455

/-- Define the cost of evening ticket -/
def evening_ticket_cost : ℝ := 10

/-- Define the cost of large popcorn & drink combo -/
def food_combo_cost : ℝ := 10

/-- Define the discount percentage on tickets from 12 noon to 3 pm -/
def ticket_discount : ℝ := 0.20

/-- Define the discount percentage on food combos from 12 noon to 3 pm -/
def food_combo_discount : ℝ := 0.50

/-- Prove that the total savings Trip could achieve by going to the earlier movie is $7 -/
theorem savings_by_going_earlier : 
  (ticket_discount * evening_ticket_cost) + (food_combo_discount * food_combo_cost) = 7 := by
  sorry

end savings_by_going_earlier_l134_134455


namespace votes_switched_l134_134790

theorem votes_switched (x : ℕ) (total_votes : ℕ) (half_votes : ℕ) 
  (votes_first_round : ℕ) (votes_second_round_winner : ℕ) (votes_second_round_loser : ℕ)
  (cond1 : total_votes = 48000)
  (cond2 : half_votes = total_votes / 2)
  (cond3 : votes_first_round = half_votes)
  (cond4 : votes_second_round_winner = half_votes + x)
  (cond5 : votes_second_round_loser = half_votes - x)
  (cond6 : votes_second_round_winner = 5 * votes_second_round_loser) :
  x = 16000 := by
  -- Proof will go here
  sorry

end votes_switched_l134_134790


namespace circumcircle_radius_of_triangle_l134_134684

theorem circumcircle_radius_of_triangle
  (A B C : Type)
  [MetricSpace A]
  [MetricSpace B]
  [MetricSpace C]
  (AB BC : ℝ)
  (angle_ABC : ℝ)
  (hAB : AB = 4)
  (hBC : BC = 4)
  (h_angle_ABC : angle_ABC = 120) :
  ∃ (R : ℝ), R = 4 := by
  sorry

end circumcircle_radius_of_triangle_l134_134684


namespace Fiona_Less_Than_Charles_l134_134930

noncomputable def percentDifference (a b : ℝ) : ℝ :=
  ((a - b) / a) * 100

theorem Fiona_Less_Than_Charles : percentDifference 600 (450 * 1.1) = 17.5 :=
by
  sorry

end Fiona_Less_Than_Charles_l134_134930


namespace perimeter_triangle_formed_by_parallel_lines_l134_134692

-- Defining the side lengths of the triangle ABC
def AB := 150
def BC := 270
def AC := 210

-- Defining the lengths of the segments formed by intersections with lines parallel to the sides of ABC
def length_lA := 65
def length_lB := 60
def length_lC := 20

-- The perimeter of the triangle formed by the intersection of the lines
theorem perimeter_triangle_formed_by_parallel_lines :
  let perimeter : ℝ := 5.71 + 20 + 83.33 + 65 + 91 + 60 + 5.71
  perimeter = 330.75 := by
  sorry

end perimeter_triangle_formed_by_parallel_lines_l134_134692


namespace symmetric_circle_equation_l134_134481

theorem symmetric_circle_equation :
  ∀ (x y : ℝ),
    (x^2 + y^2 - 6 * x + 8 * y + 24 = 0) →
    (x - 3 * y - 5 = 0) →
    (∃ x₀ y₀ : ℝ, (x₀ - 1)^2 + (y₀ - 2)^2 = 1) :=
by
  sorry

end symmetric_circle_equation_l134_134481


namespace value_of_a_l134_134192

theorem value_of_a (a : ℝ) : 
  ({2, 3} : Set ℝ) ⊆ ({1, 2, a} : Set ℝ) → a = 3 :=
by
  sorry

end value_of_a_l134_134192


namespace sum_of_reciprocals_eq_two_l134_134700

theorem sum_of_reciprocals_eq_two (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 20) : 1 / x + 1 / y = 2 := by
  sorry

end sum_of_reciprocals_eq_two_l134_134700


namespace problem_a_problem_b_l134_134770

section ProblemA

variable (x : ℝ)

theorem problem_a :
  x ≠ 0 ∧ x ≠ -3/8 ∧ x ≠ 3/7 →
  2 + 5 / (4 * x) - 15 / (4 * x * (8 * x + 3)) = 2 * (7 * x + 1) / (7 * x - 3) →
  x = 9 := by
  sorry

end ProblemA

section ProblemB

variable (x : ℝ)

theorem problem_b :
  x ≠ 0 →
  2 / x + 1 / x^2 - (7 + 10 * x) / (x^2 * (x^2 + 7)) = 2 / (x + 3 / (x + 4 / x)) →
  x = 4 := by
  sorry

end ProblemB

end problem_a_problem_b_l134_134770


namespace complement_intersection_l134_134384

open Set

variable (U M N : Set ℕ)
variable (U_def : U = {1, 2, 3, 4, 5, 6})
variable (M_def : M = {2, 3})
variable (N_def : N = {1, 4})

theorem complement_intersection (U M N : Set ℕ) (U_def : U = {1, 2, 3, 4, 5, 6}) (M_def : M = {2, 3}) (N_def : N = {1, 4}) :
  (U \ M) ∩ (U \ N) = {5, 6} := by
  sorry

end complement_intersection_l134_134384


namespace number_of_democrats_in_senate_l134_134002

/-
This Lean statement captures the essence of the problem: proving the number of Democrats in the Senate (S_D) is 55,
under given conditions involving the House's and Senate's number of Democrats and Republicans.
-/

theorem number_of_democrats_in_senate
  (D R S_D S_R : ℕ)
  (h1 : D + R = 434)
  (h2 : R = D + 30)
  (h3 : S_D + S_R = 100)
  (h4 : S_D * 4 = S_R * 5) :
  S_D = 55 := by
  sorry

end number_of_democrats_in_senate_l134_134002


namespace inscribed_square_length_l134_134667

-- Define the right triangle PQR with given sides
variables (PQ QR PR : ℕ)
variables (h s : ℚ)

-- Given conditions
def right_triangle_PQR : Prop := PQ = 5 ∧ QR = 12 ∧ PR = 13
def altitude_Q_to_PR : Prop := h = (PQ * QR) / PR
def side_length_of_square : Prop := s = h * (1 - h / PR)

theorem inscribed_square_length (PQ QR PR h s : ℚ) 
    (right_triangle_PQR : PQ = 5 ∧ QR = 12 ∧ PR = 13)
    (altitude_Q_to_PR : h = (PQ * QR) / PR) 
    (side_length_of_square : s = h * (1 - h / PR)) 
    : s = 6540 / 2207 := by
  -- we skip the proof here as requested
  sorry

end inscribed_square_length_l134_134667


namespace compare_minus_abs_val_l134_134914

theorem compare_minus_abs_val :
  -|(-8)| < -6 := 
sorry

end compare_minus_abs_val_l134_134914


namespace closed_pipe_length_l134_134861

def speed_of_sound : ℝ := 333
def fundamental_frequency : ℝ := 440

theorem closed_pipe_length :
  ∃ l : ℝ, l = 0.189 ∧ fundamental_frequency = speed_of_sound / (4 * l) :=
by
  sorry

end closed_pipe_length_l134_134861


namespace a_2_value_l134_134507

theorem a_2_value (a a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℝ) (x : ℝ) :
  x^3 + x^10 = a + a1 * (x+1) + a2 * (x+1)^2 + a3 * (x+1)^3 + a4 * (x+1)^4 + a5 * (x+1)^5 +
  a6 * (x+1)^6 + a7 * (x+1)^7 + a8 * (x+1)^8 + a9 * (x+1)^9 + a10 * (x+1)^10 → 
  a2 = 42 :=
by
  sorry

end a_2_value_l134_134507


namespace cos_30_eq_sqrt3_div_2_l134_134417

theorem cos_30_eq_sqrt3_div_2 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 :=
by
  sorry

end cos_30_eq_sqrt3_div_2_l134_134417


namespace g_sum_even_function_l134_134410

def g (a b c d x : ℝ) : ℝ := a * x ^ 8 + b * x ^ 6 - c * x ^ 4 + d * x ^ 2 + 5

theorem g_sum_even_function 
  (a b c d : ℝ) 
  (h : g a b c d 2 = 4)
  : g a b c d 2 + g a b c d (-2) = 8 :=
by
  sorry

end g_sum_even_function_l134_134410


namespace seed_germination_probability_l134_134449

-- Define necessary values and variables
def n : ℕ := 3
def p : ℚ := 0.7
def k : ℕ := 2

-- Define the binomial probability formula
def binomial_probability (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k) * (p^k) * ((1 - p)^(n - k))

-- State the proof problem
theorem seed_germination_probability :
  binomial_probability n k p = 0.441 := 
sorry

end seed_germination_probability_l134_134449


namespace gcd_9247_4567_eq_1_l134_134277

theorem gcd_9247_4567_eq_1 : Int.gcd 9247 4567 = 1 := sorry

end gcd_9247_4567_eq_1_l134_134277


namespace calculate_permutation_sum_l134_134158

noncomputable def A (n k : ℕ) : ℕ := n.factorial / (n - k).factorial

theorem calculate_permutation_sum (n : ℕ) (h1 : 3 ≤ n) (h2 : n ≤ 3) :
  A (2 * n) (n + 3) + A 4 (n + 1) = 744 := by
  sorry

end calculate_permutation_sum_l134_134158


namespace monochromatic_triangle_probability_l134_134619

noncomputable def probability_of_monochromatic_triangle_in_hexagon : ℝ := 0.968324

theorem monochromatic_triangle_probability :
  ∃ (H : Hexagon), probability_of_monochromatic_triangle_in_hexagon = 0.968324 :=
sorry

end monochromatic_triangle_probability_l134_134619


namespace sum_of_interior_edges_l134_134372

-- Define the problem parameters
def width_of_frame : ℝ := 2 -- width of the frame pieces in inches
def exposed_area : ℝ := 30 -- exposed area of the frame in square inches
def outer_edge_length : ℝ := 6 -- one of the outer edge length in inches

-- Define the statement to prove
theorem sum_of_interior_edges :
  ∃ (y : ℝ), (6 * y - 2 * (y - width_of_frame * 2) = exposed_area) ∧
  (2 * (6 - width_of_frame * 2) + 2 * (y - width_of_frame * 2) = 7) :=
sorry

end sum_of_interior_edges_l134_134372


namespace problem_l134_134239

noncomputable def f (x : ℝ) : ℝ := Real.log x - x + 1

theorem problem (a b : ℝ) (h₀ : 0 < a) (h₁ : a < b) : 
  ((f b - f a) / (b - a) < 1 / (a * (a + 1))) :=
by
  sorry -- Proof steps go here

end problem_l134_134239


namespace find_a_l134_134681

theorem find_a (x y z a : ℝ) (h1 : ∃ k : ℝ, x = 3 * k ∧ y = 4 * k ∧ z = 7 * k) 
              (h2 : x + y + z = 70) 
              (h3 : y = 15 * a - 5) : 
  a = 5 / 3 := 
by sorry

end find_a_l134_134681


namespace tangent_line_at_one_l134_134322

noncomputable def f (x : ℝ) := Real.log x + 2 * x^2 - 4 * x

theorem tangent_line_at_one :
  let slope := (1/x + 4*x - 4) 
  let y_val := -2 
  ∃ (A : ℝ) (B : ℝ) (C : ℝ), A = 1 ∧ B = -1 ∧ C = -3 ∧ (∀ (x y : ℝ), f x = y → A * x + B * y + C = 0) :=
by
  sorry

end tangent_line_at_one_l134_134322


namespace last_digit_of_3_pow_2004_l134_134602

theorem last_digit_of_3_pow_2004 : (3 ^ 2004) % 10 = 1 := by
  sorry

end last_digit_of_3_pow_2004_l134_134602


namespace weekly_spending_l134_134477

-- Definitions based on the conditions outlined in the original problem
def weekly_allowance : ℝ := 50
def hours_per_week : ℕ := 30
def hourly_wage : ℝ := 9
def weeks_per_year : ℕ := 52
def first_year_allowance : ℝ := weekly_allowance * weeks_per_year
def second_year_earnings : ℝ := (hourly_wage * hours_per_week) * weeks_per_year
def total_car_cost : ℝ := 15000
def additional_needed : ℝ := 2000
def total_savings : ℝ := first_year_allowance + second_year_earnings

-- The amount Thomas needs over what he has saved
def total_needed : ℝ := total_savings + additional_needed
def amount_spent_on_self : ℝ := total_needed - total_car_cost
def total_weeks : ℕ := 2 * weeks_per_year

theorem weekly_spending :
  amount_spent_on_self / total_weeks = 35 := by
  sorry

end weekly_spending_l134_134477


namespace sequence_eventually_constant_l134_134146

theorem sequence_eventually_constant (n : ℕ) (h : n ≥ 1) : 
  ∃ s, ∀ k ≥ s, (2 ^ (2 ^ k) % n) = (2 ^ (2 ^ (k + 1)) % n) :=
sorry

end sequence_eventually_constant_l134_134146


namespace circular_garden_remaining_grass_area_l134_134133

noncomputable def remaining_grass_area (diameter : ℝ) (path_width: ℝ) : ℝ :=
  let radius := diameter / 2
  let circle_area := Real.pi * radius^2
  let path_area := path_width * diameter
  circle_area - path_area

theorem circular_garden_remaining_grass_area :
  remaining_grass_area 10 2 = 25 * Real.pi - 20 := sorry

end circular_garden_remaining_grass_area_l134_134133


namespace length_decrease_by_33_percent_l134_134950

theorem length_decrease_by_33_percent (L W L_new : ℝ) 
  (h1 : L * W = L_new * 1.5 * W) : 
  L_new = (2 / 3) * L ∧ ((1 - (2 / 3)) * 100 = 33.33) := 
by
  sorry

end length_decrease_by_33_percent_l134_134950


namespace local_max_2_l134_134332

noncomputable def f (x m n : ℝ) := 2 * Real.log x - (1 / 2) * m * x^2 - n * x

theorem local_max_2 (m n : ℝ) (h : n = 1 - 2 * m) :
  ∃ m : ℝ, -1/2 < m ∧ (∀ x : ℝ, x > 0 → (∃ U : Set ℝ, IsOpen U ∧ (2 ∈ U) ∧ (∀ y ∈ U, f y m n ≤ f 2 m n))) :=
sorry

end local_max_2_l134_134332


namespace Kolya_is_correct_Valya_is_incorrect_l134_134127

noncomputable def Kolya_probability (x : ℝ) : ℝ :=
  let r := 1 / (x + 1)
  let p := 1 / x
  let s := x / (x + 1)
  let q := (x - 1) / x
  r / (1 - s * q)

noncomputable def Valya_probability_not_losing (x : ℝ) : ℝ :=
  let r := 1 / (x + 1)
  let p := 1 / x
  let s := x / (x + 1)
  let q := (x - 1) / x
  r / (1 - s * q)

theorem Kolya_is_correct (x : ℝ) (hx : x > 0) : Kolya_probability x = 1 / 2 :=
by
  sorry

theorem Valya_is_incorrect (x : ℝ) (hx : x > 0) : Valya_probability_not_losing x = 1 / 2 :=
by
  sorry

end Kolya_is_correct_Valya_is_incorrect_l134_134127


namespace sampling_interval_divisor_l134_134513

theorem sampling_interval_divisor (P : ℕ) (hP : P = 524) (k : ℕ) (hk : k ∣ P) : k = 4 :=
by
  sorry

end sampling_interval_divisor_l134_134513


namespace solve_equation_l134_134352

noncomputable def equation_solution (x : ℝ) : Prop :=
  (3 / x = 2 / (x - 2)) ∧ x ≠ 0 ∧ x - 2 ≠ 0

theorem solve_equation : (equation_solution 6) :=
  by
    sorry

end solve_equation_l134_134352


namespace max_value_of_function_l134_134052

noncomputable def max_value (x : ℝ) : ℝ := 3 * Real.sin x + 2

theorem max_value_of_function : 
  ∀ x : ℝ, (- (Real.pi / 2)) ≤ x ∧ x ≤ 0 → max_value x ≤ 2 :=
sorry

end max_value_of_function_l134_134052


namespace inequality_solution_set_l134_134530

theorem inequality_solution_set (a : ℝ) (x : ℝ) (h : (a - 1) * x > 2) : x < 2 / (a - 1) ↔ a < 1 :=
by
  sorry

end inequality_solution_set_l134_134530


namespace minimum_value_of_function_l134_134837

theorem minimum_value_of_function (x : ℝ) (hx : x > 4) : 
    (∃ y : ℝ, y = x + 9 / (x - 4) ∧ (∀ z : ℝ, (∃ w : ℝ, w > 4 ∧ z = w + 9 / (w - 4)) → z ≥ 10) ∧ y = 10) :=
sorry

end minimum_value_of_function_l134_134837


namespace sin_of_angle_in_first_quadrant_l134_134298

theorem sin_of_angle_in_first_quadrant (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.tan α = 3 / 4) : Real.sin α = 3 / 5 :=
by
  sorry

end sin_of_angle_in_first_quadrant_l134_134298


namespace helga_tried_on_66_pairs_of_shoes_l134_134771

variables 
  (n1 n2 n3 n4 n5 n6 : ℕ)
  (h1 : n1 = 7)
  (h2 : n2 = n1 + 2)
  (h3 : n3 = 0)
  (h4 : n4 = 2 * (n1 + n2 + n3))
  (h5 : n5 = n2 - 3)
  (h6 : n6 = n1 + 5)
  (total : ℕ := n1 + n2 + n3 + n4 + n5 + n6)

theorem helga_tried_on_66_pairs_of_shoes : total = 66 :=
by sorry

end helga_tried_on_66_pairs_of_shoes_l134_134771


namespace percentage_decrease_l134_134151

theorem percentage_decrease (A C : ℝ) (h1 : C > A) (h2 : A > 0) (h3 : C = 1.20 * A) : 
  ∃ y : ℝ, A = C - (y/100) * C ∧ y = 50 / 3 :=
by {
  sorry
}

end percentage_decrease_l134_134151


namespace find_ks_l134_134184

theorem find_ks (k : ℕ) : 
  (∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ m * (m + k) = n * (n + 1)) ↔ k ≠ 2 ∧ k ≠ 3 :=
by sorry

end find_ks_l134_134184


namespace no_term_un_eq_neg1_l134_134518

theorem no_term_un_eq_neg1 (p : ℕ) [hp_prime: Fact (Nat.Prime p)] (hp_odd: p % 2 = 1) (hp_not_five: p ≠ 5) :
  ∀ n : ℕ, ∀ u : ℕ → ℤ, ((u 0 = 0) ∧ (u 1 = 1) ∧ (∀ k, k ≥ 2 → u (k-2) = 2 * u (k-1) - p * u k)) → 
    (u n ≠ -1) :=
  sorry

end no_term_un_eq_neg1_l134_134518


namespace gcd_ab_a2b2_eq_1_or_2_l134_134755

theorem gcd_ab_a2b2_eq_1_or_2
  (a b : Nat)
  (h_coprime : Nat.gcd a b = 1) :
  Nat.gcd (a + b) (a^2 + b^2) = 1 ∨ Nat.gcd (a + b) (a^2 + b^2) = 2 :=
by {
  sorry
}

end gcd_ab_a2b2_eq_1_or_2_l134_134755


namespace product_of_series_l134_134902

theorem product_of_series :
  (1 - 1/2^2) * (1 - 1/3^2) * (1 - 1/4^2) * (1 - 1/5^2) * (1 - 1/6^2) *
  (1 - 1/7^2) * (1 - 1/8^2) * (1 - 1/9^2) * (1 - 1/10^2) = 11 / 20 :=
by 
  sorry

end product_of_series_l134_134902


namespace find_two_digit_numbers_l134_134316

theorem find_two_digit_numbers :
  ∀ (a b : ℕ), (1 ≤ a ∧ a ≤ 9) → (0 ≤ b ∧ b ≤ 9) → (10 * a + b = 3 * a * b) → (10 * a + b = 15 ∨ 10 * a + b = 24) :=
by
  intros
  sorry

end find_two_digit_numbers_l134_134316


namespace marked_price_correct_l134_134719

noncomputable def marked_price (original_price discount_percent purchase_price profit_percent final_price_percent : ℝ) := 
  (purchase_price * (1 + profit_percent)) / final_price_percent

theorem marked_price_correct
  (original_price : ℝ)
  (discount_percent : ℝ)
  (profit_percent : ℝ)
  (final_price_percent : ℝ)
  (purchase_price : ℝ := original_price * (1 - discount_percent))
  (expected_marked_price : ℝ) :
  original_price = 40 →
  discount_percent = 0.15 →
  profit_percent = 0.25 →
  final_price_percent = 0.90 →
  expected_marked_price = 47.20 →
  marked_price original_price discount_percent purchase_price profit_percent final_price_percent = expected_marked_price := 
by
  intros
  sorry

end marked_price_correct_l134_134719


namespace smallest_number_divisible_l134_134301

theorem smallest_number_divisible (n : ℕ) 
    (h1 : (n - 20) % 15 = 0) 
    (h2 : (n - 20) % 30 = 0)
    (h3 : (n - 20) % 45 = 0)
    (h4 : (n - 20) % 60 = 0) : 
    n = 200 :=
sorry

end smallest_number_divisible_l134_134301


namespace problem1_problem2_l134_134156

theorem problem1 :
  (2 / 3) * Real.sqrt 24 / (-Real.sqrt 3) * (1 / 3) * Real.sqrt 27 = - (4 / 3) * Real.sqrt 6 :=
sorry

theorem problem2 :
  Real.sqrt 3 * Real.sqrt 12 + (Real.sqrt 3 + 1)^2 = 10 + 2 * Real.sqrt 3 :=
sorry

end problem1_problem2_l134_134156


namespace trigonometric_identity_l134_134514

open Real

noncomputable def acute (x : ℝ) := 0 < x ∧ x < π / 2

theorem trigonometric_identity 
  {α β : ℝ} (hα : acute α) (hβ : acute β) (h : cos α > sin β) :
  α + β < π / 2 :=
sorry

end trigonometric_identity_l134_134514


namespace nancy_games_this_month_l134_134595

-- Define the variables and conditions from the problem
def went_games_last_month : ℕ := 8
def plans_games_next_month : ℕ := 7
def total_games : ℕ := 24

-- Let's calculate the games this month and state the theorem
def games_last_and_next : ℕ := went_games_last_month + plans_games_next_month
def games_this_month : ℕ := total_games - games_last_and_next

-- The theorem statement
theorem nancy_games_this_month : games_this_month = 9 := by
  -- Proof is omitted for the sake of brevity
  sorry

end nancy_games_this_month_l134_134595


namespace number_of_bags_of_chips_l134_134055

theorem number_of_bags_of_chips (friends : ℕ) (amount_per_friend : ℕ) (cost_per_bag : ℕ) (total_amount : ℕ) (number_of_bags : ℕ) : 
  friends = 3 → amount_per_friend = 5 → cost_per_bag = 3 → total_amount = friends * amount_per_friend → number_of_bags = total_amount / cost_per_bag → number_of_bags = 5 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end number_of_bags_of_chips_l134_134055


namespace find_number_l134_134487

-- Define the conditions and the theorem
theorem find_number (number : ℝ)
  (h₁ : ∃ w : ℝ, w = (69.28 * number) / 0.03 ∧ abs (w - 9.237333333333334) ≤ 1e-10) :
  abs (number - 0.004) ≤ 1e-10 :=
by
  sorry

end find_number_l134_134487


namespace find_m_l134_134531

theorem find_m (m : ℝ) (h : |m| = |m + 2|) : m = -1 :=
sorry

end find_m_l134_134531


namespace smallest_z_l134_134385

-- Given conditions
def distinct_consecutive_even_positive_perfect_cubes (w x y z : ℕ) : Prop :=
  w^3 + x^3 + y^3 = z^3 ∧
  ∃ a b c d : ℕ, 
    a < b ∧ b < c ∧ c < d ∧
    2 * a = w ∧ 2 * b = x ∧ 2 * c = y ∧ 2 * d = z

-- The smallest value of z proving the equation holds
theorem smallest_z (w x y z : ℕ) (h : distinct_consecutive_even_positive_perfect_cubes w x y z) : z = 12 :=
  sorry

end smallest_z_l134_134385


namespace shaded_area_eq_63_l134_134119

noncomputable def rect1_width : ℕ := 4
noncomputable def rect1_height : ℕ := 12
noncomputable def rect2_width : ℕ := 5
noncomputable def rect2_height : ℕ := 7
noncomputable def overlap_width : ℕ := 4
noncomputable def overlap_height : ℕ := 5

theorem shaded_area_eq_63 :
  (rect1_width * rect1_height) + (rect2_width * rect2_height) - (overlap_width * overlap_height) = 63 := by
  sorry

end shaded_area_eq_63_l134_134119


namespace marie_finishes_ninth_task_at_730PM_l134_134929

noncomputable def start_time : ℕ := 8 * 60 -- 8:00 AM in minutes
noncomputable def end_time_task_3 : ℕ := 11 * 60 + 30 -- 11:30 AM in minutes
noncomputable def total_tasks : ℕ := 9
noncomputable def tasks_done_by_1130AM : ℕ := 3
noncomputable def end_time_task_9 : ℕ := 19 * 60 + 30 -- 7:30 PM in minutes

theorem marie_finishes_ninth_task_at_730PM
    (h1 : start_time = 480) -- 8:00 AM
    (h2 : end_time_task_3 = 690) -- 11:30 AM
    (h3 : total_tasks = 9)
    (h4 : tasks_done_by_1130AM = 3)
    (h5 : end_time_task_9 = 1170) -- 7:30 PM
    : end_time_task_9 = start_time + ((end_time_task_3 - start_time) / tasks_done_by_1130AM) * total_tasks :=
sorry

end marie_finishes_ninth_task_at_730PM_l134_134929


namespace painting_time_l134_134222

theorem painting_time (karl_time leo_time : ℝ) (t : ℝ) (break_time : ℝ) : 
  karl_time = 6 → leo_time = 8 → break_time = 0.5 → 
  (1 / karl_time + 1 / leo_time) * (t - break_time) = 1 :=
by
  intros h_karl h_leo h_break
  rw [h_karl, h_leo, h_break]
  -- sorry to skip the proof
  sorry

end painting_time_l134_134222


namespace tutors_meet_again_l134_134416

theorem tutors_meet_again (tim uma victor xavier: ℕ) (h1: tim = 5) (h2: uma = 6) (h3: victor = 9) (h4: xavier = 8) :
  Nat.lcm (Nat.lcm tim uma) (Nat.lcm victor xavier) = 360 := 
by 
  rw [h1, h2, h3, h4]
  show Nat.lcm (Nat.lcm 5 6) (Nat.lcm 9 8) = 360
  sorry

end tutors_meet_again_l134_134416


namespace value_of_f_15_l134_134194

def f (n : ℕ) : ℕ := n^2 + 2*n + 19

theorem value_of_f_15 : f 15 = 274 := 
by 
  -- Add proof here
  sorry

end value_of_f_15_l134_134194


namespace greatest_b_for_no_real_roots_l134_134414

theorem greatest_b_for_no_real_roots :
  ∀ (b : ℤ), (∀ x : ℝ, x^2 + (b : ℝ) * x + 12 ≠ 0) ↔ b ≤ 6 := sorry

end greatest_b_for_no_real_roots_l134_134414


namespace perimeter_of_square_is_32_l134_134426

-- Given conditions
def radius := 4
def diameter := 2 * radius
def side_length_of_square := diameter

-- Question: What is the perimeter of the square?
def perimeter_of_square := 4 * side_length_of_square

-- Proof statement
theorem perimeter_of_square_is_32 : perimeter_of_square = 32 :=
sorry

end perimeter_of_square_is_32_l134_134426


namespace total_min_waiting_time_total_max_waiting_time_total_expected_waiting_time_l134_134019

variables (a b: ℕ) (n m: ℕ)

def C (x y : ℕ) : ℕ := x.choose y

def T_min (a n m : ℕ) : ℕ :=
  a * C n 2 + a * m * n + b * C m 2

def T_max (a n m : ℕ) : ℕ :=
  a * C n 2 + b * m * n + b * C m 2

def E_T (a b n m : ℕ) : ℕ :=
  C (n + m) 2 * ((b * m + a * n) / (m + n))

theorem total_min_waiting_time (a b : ℕ) : T_min 1 5 3 = 40 :=
  by sorry

theorem total_max_waiting_time (a b : ℕ) : T_max 1 5 3 = 100 :=
  by sorry

theorem total_expected_waiting_time (a b : ℕ) : E_T 1 5 5 3 = 70 :=
  by sorry

end total_min_waiting_time_total_max_waiting_time_total_expected_waiting_time_l134_134019


namespace radius_ratio_in_right_triangle_l134_134932

theorem radius_ratio_in_right_triangle (PQ QR PR PS SR : ℝ)
  (h₁ : PQ = 5) (h₂ : QR = 12) (h₃ : PR = 13)
  (h₄ : PS + SR = PR) (h₅ : PS / SR = 5 / 8)
  (r_p r_q : ℝ)
  (hr_p : r_p = (1 / 2 * PQ * PS / 3) / ((PQ + PS / 3 + PS) / 3))
  (hr_q : r_q = (1 / 2 * QR * SR) / ((PS / 3 + QR + SR) / 3)) :
  r_p / r_q = 175 / 576 :=
sorry

end radius_ratio_in_right_triangle_l134_134932


namespace dividend_is_144_l134_134290

theorem dividend_is_144 
  (Q : ℕ) (D : ℕ) (M : ℕ)
  (h1 : M = 6 * D)
  (h2 : D = 4 * Q) 
  (Q_eq_6 : Q = 6) : 
  M = 144 := 
sorry

end dividend_is_144_l134_134290


namespace negation_proposition_l134_134716

theorem negation_proposition (x : ℝ) : ¬ (x ≥ 1 → x^2 - 4 * x + 2 ≥ -1) ↔ (x < 1 ∧ x^2 - 4 * x + 2 < -1) :=
by
  sorry

end negation_proposition_l134_134716


namespace find_p_q_l134_134559

variable (p q : ℝ)
def f (x : ℝ) : ℝ := x^2 + p * x + q

theorem find_p_q:
  (p, q) = (-6, 7) →
  ∀ x : ℝ, (1 ≤ x ∧ x ≤ 5) → |f p q x| ≤ 2 :=
by
  sorry

end find_p_q_l134_134559


namespace intersection_subset_proper_l134_134120

-- Definitions of P and Q
def P : Set ℝ := {1, 2, 3, 4}
def Q : Set ℝ := { x | 0 ≤ x ∧ x ≤ 3 }

-- The problem statement to prove
theorem intersection_subset_proper : P ∩ Q ⊂ P := by
  sorry

end intersection_subset_proper_l134_134120


namespace total_amount_divided_l134_134604

theorem total_amount_divided 
    (A B C : ℝ) 
    (h1 : A = (2 / 3) * (B + C)) 
    (h2 : B = (2 / 3) * (A + C)) 
    (h3 : A = 160) : 
    A + B + C = 400 := 
by 
  sorry

end total_amount_divided_l134_134604


namespace age_ratio_3_2_l134_134498

/-
Define variables: 
  L : ℕ -- Liam's current age
  M : ℕ -- Mia's current age
  y : ℕ -- number of years until the age ratio is 3:2
-/

theorem age_ratio_3_2 (L M : ℕ) 
  (h1 : L - 4 = 2 * (M - 4)) 
  (h2 : L - 10 = 3 * (M - 10)) 
  (h3 : ∃ y, (L + y) * 2 = (M + y) * 3) : 
  ∃ y, y = 8 :=
by
  sorry

end age_ratio_3_2_l134_134498


namespace ratio_of_x_intercepts_l134_134649

theorem ratio_of_x_intercepts (b s t : ℝ) (h_b : b ≠ 0)
  (h1 : 0 = 8 * s + b)
  (h2 : 0 = 4 * t + b) :
  s / t = 1 / 2 :=
by
  sorry

end ratio_of_x_intercepts_l134_134649


namespace bobby_toy_cars_in_5_years_l134_134919

noncomputable def toy_cars_after_n_years (initial_cars : ℕ) (percentage_increase : ℝ) (n : ℕ) : ℝ :=
initial_cars * (1 + percentage_increase)^n

theorem bobby_toy_cars_in_5_years :
  toy_cars_after_n_years 25 0.75 5 = 410 := by
  -- 25 * (1 + 0.75)^5 
  -- = 25 * (1.75)^5 
  -- ≈ 410.302734375
  -- After rounding
  sorry

end bobby_toy_cars_in_5_years_l134_134919


namespace unwanted_texts_per_week_l134_134616

-- Define the conditions as constants
def messages_per_day_old : ℕ := 20
def messages_per_day_new : ℕ := 55
def days_per_week : ℕ := 7

-- Define the theorem stating the problem
theorem unwanted_texts_per_week (messages_per_day_old messages_per_day_new days_per_week 
  : ℕ) : (messages_per_day_new - messages_per_day_old) * days_per_week = 245 :=
by
  sorry

end unwanted_texts_per_week_l134_134616


namespace tetrahedron_volume_l134_134710

noncomputable def volume_tetrahedron (A₁ A₂ : ℝ) (θ : ℝ) (d : ℝ) : ℝ :=
  (A₁ * A₂ * Real.sin θ) / (3 * d)

theorem tetrahedron_volume:
  ∀ (PQ PQR PQS : ℝ) (θ : ℝ),
  PQ = 5 → PQR = 20 → PQS = 18 → θ = Real.pi / 4 → volume_tetrahedron PQR PQS θ PQ = 24 * Real.sqrt 2 :=
by
  intros
  unfold volume_tetrahedron
  sorry

end tetrahedron_volume_l134_134710


namespace original_radius_of_cylinder_l134_134153

theorem original_radius_of_cylinder (r y : ℝ) 
  (h₁ : 3 * π * ((r + 5)^2 - r^2) = y) 
  (h₂ : 5 * π * r^2 = y)
  (h₃ : 3 > 0) :
  r = 7.5 :=
by
  sorry

end original_radius_of_cylinder_l134_134153


namespace opponent_final_score_l134_134366

theorem opponent_final_score (x : ℕ) (h : x + 29 = 39) : x = 10 :=
by {
  sorry
}

end opponent_final_score_l134_134366


namespace zookeeper_feeding_problem_l134_134485

noncomputable def feeding_ways : ℕ :=
  sorry

theorem zookeeper_feeding_problem :
  feeding_ways = 2880 := 
sorry

end zookeeper_feeding_problem_l134_134485


namespace correct_conclusion_l134_134545

theorem correct_conclusion (x : ℝ) (hx : x > 1/2) : -2 * x + 1 < 0 :=
by
  -- sorry placeholder
  sorry

end correct_conclusion_l134_134545


namespace evaluate_expression_l134_134835

theorem evaluate_expression :
  (π - 2023) ^ 0 + |(-9)| - 3 ^ 2 = 1 :=
by
  sorry

end evaluate_expression_l134_134835


namespace problem_statement_l134_134294

noncomputable def repeating_decimal_to_fraction (n : ℕ) : ℚ :=
  -- Conversion function for repeating two-digit decimals to fractions
  n / 99

theorem problem_statement :
  (repeating_decimal_to_fraction 63) / (repeating_decimal_to_fraction 21) = 3 :=
by
  -- expected simplification and steps skipped
  sorry

end problem_statement_l134_134294


namespace line_intersects_parabola_at_one_point_l134_134154

theorem line_intersects_parabola_at_one_point (k : ℝ) : (∃ y : ℝ, -y^2 - 4 * y + 2 = k) ↔ k = 6 :=
by 
  sorry

end line_intersects_parabola_at_one_point_l134_134154


namespace ratio_nephews_l134_134013

variable (N : ℕ) -- The number of nephews Alden has now.
variable (Alden_had_50 : Prop := 50 = 50)
variable (Vihaan_more_60 : Prop := Vihaan = N + 60)
variable (Together_260 : Prop := N + (N + 60) = 260)

theorem ratio_nephews (N : ℕ) 
  (H1 : Alden_had_50)
  (H2 : Vihaan_more_60)
  (H3 : Together_260) :
  50 / N = 1 / 2 :=
by
  sorry

end ratio_nephews_l134_134013


namespace sin_theta_value_l134_134089

theorem sin_theta_value (f : ℝ → ℝ)
  (hx : ∀ x, f x = 3 * Real.sin x - 8 * Real.cos (x / 2) ^ 2)
  (h_cond : ∀ x, f x ≤ f θ) : Real.sin θ = 3 / 5 := 
sorry

end sin_theta_value_l134_134089


namespace hyperbola_eccentricity_l134_134843

theorem hyperbola_eccentricity 
  (p1 p2 : ℝ × ℝ)
  (asymptote_passes_through_p1 : p1 = (1, 2))
  (hyperbola_passes_through_p2 : p2 = (2 * Real.sqrt 2, 4)) :
  ∃ e : ℝ, e = Real.sqrt 5 :=
by
  sorry

end hyperbola_eccentricity_l134_134843


namespace sarah_min_days_l134_134576

theorem sarah_min_days (r P B : ℝ) (x : ℕ) (h_r : r = 0.1) (h_P : P = 20) (h_B : B = 60) :
  (P + r * P * x ≥ B) → (x ≥ 20) :=
by
  sorry

end sarah_min_days_l134_134576


namespace evaluate_P_l134_134959

noncomputable def P (x : ℝ) : ℝ := x^3 - 6*x^2 - 5*x + 4

theorem evaluate_P (y : ℝ) (z : ℝ) (hz : ∀ n : ℝ, z * P y = P (y - n) + P (y + n)) : P 2 = -22 := by
  sorry

end evaluate_P_l134_134959


namespace fraction_left_handed_l134_134138

def total_participants (k : ℕ) := 15 * k

def red (k : ℕ) := 5 * k
def blue (k : ℕ) := 5 * k
def green (k : ℕ) := 3 * k
def yellow (k : ℕ) := 2 * k

def left_handed_red (k : ℕ) := (1 / 3) * red k
def left_handed_blue (k : ℕ) := (2 / 3) * blue k
def left_handed_green (k : ℕ) := (1 / 2) * green k
def left_handed_yellow (k : ℕ) := (1 / 4) * yellow k

def total_left_handed (k : ℕ) := left_handed_red k + left_handed_blue k + left_handed_green k + left_handed_yellow k

theorem fraction_left_handed (k : ℕ) : 
  (total_left_handed k) / (total_participants k) = 7 / 15 := 
sorry

end fraction_left_handed_l134_134138


namespace kim_branch_marking_l134_134955

theorem kim_branch_marking (L : ℝ) (rem_frac : ℝ) (third_piece : ℝ) (F : ℝ) :
  L = 3 ∧ rem_frac = 0.6 ∧ third_piece = 1 ∧ L * rem_frac = 1.8 → F = 1 / 15 :=
by sorry

end kim_branch_marking_l134_134955


namespace class_gpa_l134_134114

theorem class_gpa (n : ℕ) (h_n : n = 60)
  (n1 : ℕ) (h_n1 : n1 = 20) (gpa1 : ℕ) (h_gpa1 : gpa1 = 15)
  (n2 : ℕ) (h_n2 : n2 = 15) (gpa2 : ℕ) (h_gpa2 : gpa2 = 17)
  (n3 : ℕ) (h_n3 : n3 = 25) (gpa3 : ℕ) (h_gpa3 : gpa3 = 19) :
  (20 * 15 + 15 * 17 + 25 * 19 : ℕ) / 60 = 1717 / 100 := 
sorry

end class_gpa_l134_134114


namespace range_of_a_l134_134424

def is_increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) :=
∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y

theorem range_of_a (a : ℝ) :
  (∀ x y, 3 ≤ x ∧ x ≤ y → (x^2 - 2*a*x + 2) ≤ (y^2 - 2*a*y + 2)) → a ≤ 3 := 
sorry

end range_of_a_l134_134424


namespace jebb_take_home_pay_l134_134096

-- We define the given conditions
def tax_rate : ℝ := 0.10
def total_pay : ℝ := 650

-- We define the function for the tax amount
def tax_amount (pay : ℝ) (rate : ℝ) : ℝ := pay * rate

-- We define the function for take-home pay
def take_home_pay (pay : ℝ) (rate : ℝ) : ℝ := pay - tax_amount pay rate

-- We state the theorem that needs to be proved
theorem jebb_take_home_pay : take_home_pay total_pay tax_rate = 585 := 
by
  -- The proof is omitted.
  sorry

end jebb_take_home_pay_l134_134096


namespace total_players_is_59_l134_134443

-- Define the number of players from each sport.
def cricket_players : ℕ := 16
def hockey_players : ℕ := 12
def football_players : ℕ := 18
def softball_players : ℕ := 13

-- Define the total number of players as the sum of the above.
def total_players : ℕ :=
  cricket_players + hockey_players + football_players + softball_players

-- Prove that the total number of players is 59.
theorem total_players_is_59 :
  total_players = 59 :=
by
  unfold total_players
  unfold cricket_players
  unfold hockey_players
  unfold football_players
  unfold softball_players
  sorry

end total_players_is_59_l134_134443


namespace a_gives_b_head_start_l134_134625

theorem a_gives_b_head_start (Va Vb L H : ℝ) 
    (h1 : Va = (20 / 19) * Vb)
    (h2 : L / Va = (L - H) / Vb) : 
    H = (1 / 20) * L := sorry

end a_gives_b_head_start_l134_134625


namespace smallest_fraction_numerator_l134_134420

theorem smallest_fraction_numerator :
  ∃ (a b : ℕ), (10 ≤ a ∧ a ≤ 99) ∧ (10 ≤ b ∧ b ≤ 99) ∧ (a * 4 > b * 3) ∧ (a = 73) := 
sorry

end smallest_fraction_numerator_l134_134420


namespace students_passing_course_l134_134980

theorem students_passing_course :
  let students_three_years_ago := 200
  let increase_factor := 1.5
  let students_two_years_ago := students_three_years_ago * increase_factor
  let students_last_year := students_two_years_ago * increase_factor
  let students_this_year := students_last_year * increase_factor
  students_this_year = 675 :=
by
  sorry

end students_passing_course_l134_134980


namespace incorrect_statement_d_l134_134391

noncomputable def x := Complex.mk (-1/2) (Real.sqrt 3 / 2)
noncomputable def y := Complex.mk (-1/2) (-Real.sqrt 3 / 2)

theorem incorrect_statement_d : (x^12 + y^12) ≠ 1 := by
  sorry

end incorrect_statement_d_l134_134391


namespace max_value_f1_solve_inequality_f2_l134_134528

def f_1 (x : ℝ) : ℝ := |x + 1| - |x - 1|

theorem max_value_f1 : ∃ x, f_1 x = 2 :=
sorry

def f_2 (x : ℝ) : ℝ := |2 * x - 1| - |x - 1|

theorem solve_inequality_f2 (x : ℝ) : f_2 x ≥ 1 ↔ x ≤ -1 ∨ x ≥ 1 :=
sorry

end max_value_f1_solve_inequality_f2_l134_134528


namespace inequality_count_l134_134012

theorem inequality_count
  (x y a b : ℝ)
  (hx_pos : 0 < x)
  (hy_pos : 0 < y)
  (ha_pos : 0 < a)
  (hb_pos : 0 < b)
  (hx_lt_one : x < 1)
  (hy_lt_one : y < 1)
  (hx_lt_a : x < a)
  (hy_lt_b : y < b)
  (h_sum : x + y = a - b) :
  ({(x + y < a + b), (x - y < a - b), (x * y < a * b)}:Finset Prop).card = 3 :=
by
  sorry

end inequality_count_l134_134012


namespace problem_statement_l134_134963

variables {a c b d : ℝ} {x y q z : ℕ}

-- Given conditions:
def condition1 (a c : ℝ) (x q : ℕ) : Prop := a^(x + 1) = c^(q + 2)
def condition2 (a c : ℝ) (y z : ℕ) : Prop := c^(y + 3) = a^(z+ 4)

-- Goal statement
theorem problem_statement (a c : ℝ) (x y q z : ℕ) (h1 : condition1 a c x q) (h2 : condition2 a c y z) :
  (q + 2) * (z + 4) = (y + 3) * (x + 1) :=
sorry

end problem_statement_l134_134963


namespace matrix_pow_six_identity_l134_134307

variable {n : Type} [Fintype n] [DecidableEq n]
variables {A B C : Matrix n n ℂ}

theorem matrix_pow_six_identity 
  (h1 : A^2 = B^2) (h2 : B^2 = C^2) (h3 : B^3 = A * B * C + 2 * (1 : Matrix n n ℂ)) : 
  A^6 = 1 :=
by 
  sorry

end matrix_pow_six_identity_l134_134307


namespace identify_wrong_operator_l134_134014

def original_expr (x y z w u v p q : Int) : Int := x + y - z + w - u + v - p + q
def wrong_expr (x y z w u v p q : Int) : Int := x + y - z - w - u + v - p + q

theorem identify_wrong_operator :
  original_expr 3 5 7 9 11 13 15 17 ≠ -4 →
  wrong_expr 3 5 7 9 11 13 15 17 = -4 :=
by
  sorry

end identify_wrong_operator_l134_134014


namespace min_diff_between_y_and_x_l134_134987

theorem min_diff_between_y_and_x (x y z : ℤ)
    (h1 : x < y)
    (h2 : y < z)
    (h3 : Even x)
    (h4 : Odd y)
    (h5 : Odd z)
    (h6 : z - x = 9) :
    y - x = 1 := 
  by sorry

end min_diff_between_y_and_x_l134_134987


namespace num_integers_contains_3_and_4_l134_134563

theorem num_integers_contains_3_and_4 
  (n : ℕ) (h1 : 500 ≤ n) (h2 : n < 1000) :
  (∀ a b c : ℕ, n = 100 * a + 10 * b + c → (b = 3 ∧ c = 4) ∨ (b = 4 ∧ c = 3)) → 
  n = 10 :=
sorry

end num_integers_contains_3_and_4_l134_134563


namespace interest_rate_borrowed_l134_134519

variables {P : Type} [LinearOrderedField P]

def borrowed_amount : P := 9000
def lent_interest_rate : P := 0.06
def gain_per_year : P := 180
def per_cent : P := 100

theorem interest_rate_borrowed (r : P) (h : borrowed_amount * lent_interest_rate - gain_per_year = borrowed_amount * r) : 
  r = 0.04 :=
by sorry

end interest_rate_borrowed_l134_134519


namespace five_crows_two_hours_l134_134422

-- Define the conditions and the question as hypotheses
def crows_worms (crows worms hours : ℕ) := 
  (crows = 3) ∧ (worms = 30) ∧ (hours = 1)

theorem five_crows_two_hours 
  (c: ℕ) (w: ℕ) (h: ℕ)
  (H: crows_worms c w h)
  : ∃ worms_eaten : ℕ, worms_eaten = 100 :=
by
  sorry

end five_crows_two_hours_l134_134422


namespace tan_product_cos_conditions_l134_134137

variable {α β : ℝ}

theorem tan_product_cos_conditions
  (h1 : Real.cos (α + β) = 2 / 3)
  (h2 : Real.cos (α - β) = 1 / 3) :
  Real.tan α * Real.tan β = -1 / 3 :=
sorry

end tan_product_cos_conditions_l134_134137


namespace ice_skating_rinks_and_ski_resorts_2019_l134_134308

theorem ice_skating_rinks_and_ski_resorts_2019 (x y : ℕ) :
  x + y = 1230 →
  2 * x + 212 + y + 288 = 2560 →
  x = 830 ∧ y = 400 :=
by {
  sorry
}

end ice_skating_rinks_and_ski_resorts_2019_l134_134308


namespace race_prob_l134_134871

theorem race_prob :
  let pX := (1 : ℝ) / 8
  let pY := (1 : ℝ) / 12
  let pZ := (1 : ℝ) / 6
  pX + pY + pZ = (3 : ℝ) / 8 :=
by
  sorry

end race_prob_l134_134871


namespace twelve_times_reciprocal_sum_l134_134394

theorem twelve_times_reciprocal_sum (a b c : ℚ) (h₁ : a = 1/3) (h₂ : b = 1/4) (h₃ : c = 1/6) :
  12 * (a + b + c)⁻¹ = 16 := 
by
  sorry

end twelve_times_reciprocal_sum_l134_134394


namespace hyperbola_s_squared_l134_134172

theorem hyperbola_s_squared 
  (s : ℝ) 
  (a b : ℝ) 
  (h1 : a = 3)
  (h2 : b^2 = 144 / 13) 
  (h3 : (2, s) ∈ {p : ℝ × ℝ | (p.2)^2 / a^2 - (p.1)^2 / b^2 = 1}) :
  s^2 = 441 / 36 :=
by sorry

end hyperbola_s_squared_l134_134172


namespace acute_angles_relation_l134_134691

theorem acute_angles_relation (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h : Real.sin α = (1 / 2) * Real.sin (α + β)) : α < β :=
sorry

end acute_angles_relation_l134_134691


namespace ab_sum_l134_134551

theorem ab_sum (a b : ℤ) (h1 : |a| = 3) (h2 : |b| = 7) (h3 : |a - b| = b - a) : a + b = 10 ∨ a + b = 4 :=
by
  sorry

end ab_sum_l134_134551


namespace length_B1C1_l134_134957

variable (AC BC : ℝ) (A1B1 : ℝ) (T : ℝ)

/-- Given a right triangle ABC with legs AC = 3 and BC = 4, and transformations
  of points to A1, B1, and C1 where A1B1 = 1 and angle B1 = 90 degrees,
  prove that the length of B1C1 is 12. -/
theorem length_B1C1 (h1 : AC = 3) (h2 : BC = 4) (h3 : A1B1 = 1) 
  (TABC : T = 6) (right_triangle_ABC : true) (right_triangle_A1B1C1 : true) : 
  B1C1 = 12 := 
sorry

end length_B1C1_l134_134957


namespace triangle_inequality_theorem_l134_134431

def is_triangle (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem triangle_inequality_theorem :
  ¬ is_triangle 2 3 5 ∧ is_triangle 5 6 10 ∧ ¬ is_triangle 1 1 3 ∧ ¬ is_triangle 3 4 9 :=
by {
  -- Proof goes here
  sorry
}

end triangle_inequality_theorem_l134_134431


namespace range_of_a_l134_134910

noncomputable def f (a x : ℝ) : ℝ :=
  (1 / 2) * x^2 - a * x + (a - 1) * Real.log x

theorem range_of_a (a : ℝ) (h1 : 1 < a) :
  (∀ x1 x2 : ℝ, 0 < x1 ∧ 0 < x2 ∧ x1 > x2 → f a x1 - f a x2 > x2 - x1) ↔ (1 < a ∧ a ≤ 5) :=
by
  -- The proof is omitted
  sorry

end range_of_a_l134_134910


namespace determine_a_for_line_l134_134139

theorem determine_a_for_line (a : ℝ) (h : a ≠ 0)
  (intercept_condition : ∃ (k : ℝ), 
    ∀ x y : ℝ, (a * x - 6 * y - 12 * a = 0) → (x = 12) ∧ (y = 2 * a * x / 6) ∧ (12 = 3 * (-2 * a))) : 
  a = -2 :=
by
  sorry

end determine_a_for_line_l134_134139


namespace evaluate_f_at_2_l134_134026

def f (x : ℕ) : ℕ := 5 * x + 2

theorem evaluate_f_at_2 : f 2 = 12 := by
  sorry

end evaluate_f_at_2_l134_134026


namespace sum_of_50th_terms_l134_134250

open Nat

-- Definition of arithmetic sequence
def arithmetic_sequence (a₁ d n : ℕ) : ℕ := a₁ + (n - 1) * d

-- Definition of geometric sequence
def geometric_sequence (g₁ r n : ℕ) : ℕ := g₁ * r^(n - 1)

-- Prove the sum of the 50th terms of the given sequences
theorem sum_of_50th_terms : 
  arithmetic_sequence 3 6 50 + geometric_sequence 2 3 50 = 297 + 2 * 3^49 :=
by
  sorry

end sum_of_50th_terms_l134_134250


namespace expected_value_of_die_is_475_l134_134805

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

end expected_value_of_die_is_475_l134_134805


namespace solve_equation_l134_134099

theorem solve_equation (x : ℝ) : (x - 1) * (x + 3) = 5 ↔ x = 2 ∨ x = -4 := by
  sorry

end solve_equation_l134_134099


namespace perimeter_of_rectangle_l134_134348

theorem perimeter_of_rectangle (DC BC P : ℝ) (hDC : DC = 12) (hArea : 1/2 * DC * BC = 30) : P = 2 * (DC + BC) → P = 34 :=
by
  sorry

end perimeter_of_rectangle_l134_134348


namespace tan_3theta_eq_2_11_sin_3theta_eq_22_125_l134_134940

variable {θ : ℝ}

-- First, stating the condition \(\tan \theta = 2\)
axiom tan_theta_eq_2 : Real.tan θ = 2

-- Stating the proof problem for \(\tan 3\theta = \frac{2}{11}\)
theorem tan_3theta_eq_2_11 : Real.tan (3 * θ) = 2 / 11 :=
by 
  sorry

-- Stating the proof problem for \(\sin 3\theta = \frac{22}{125}\)
theorem sin_3theta_eq_22_125 : Real.sin (3 * θ) = 22 / 125 :=
by 
  sorry

end tan_3theta_eq_2_11_sin_3theta_eq_22_125_l134_134940


namespace number_of_possible_values_of_a_l134_134340

theorem number_of_possible_values_of_a :
  ∃ a_values : Finset ℕ, 
    (∀ a ∈ a_values, 5 ∣ a) ∧ 
    (∀ a ∈ a_values, a ∣ 30) ∧ 
    (∀ a ∈ a_values, 0 < a) ∧ 
    a_values.card = 4 :=
by
  sorry

end number_of_possible_values_of_a_l134_134340


namespace inverse_of_congruence_implies_equal_area_l134_134289

-- Definitions to capture conditions and relationships
def congruent_triangles (T1 T2 : Triangle) : Prop :=
  -- Definition agrees with congruency of two triangles
  sorry

def equal_areas (T1 T2 : Triangle) : Prop :=
  -- Definition agrees with equal areas of two triangles
  sorry

-- Statement to prove the inverse proposition
theorem inverse_of_congruence_implies_equal_area :
  (∀ T1 T2 : Triangle, congruent_triangles T1 T2 → equal_areas T1 T2) →
  (∀ T1 T2 : Triangle, equal_areas T1 T2 → congruent_triangles T1 T2) :=
  sorry

end inverse_of_congruence_implies_equal_area_l134_134289


namespace sum_of_roots_l134_134991

theorem sum_of_roots (a b c : ℝ) (h_eq : a = 1) (h_b : b = -5) (h_c : c = 6) :
  (-b / a) = 5 := by
sorry

end sum_of_roots_l134_134991


namespace legacy_total_earnings_l134_134679

def floors := 4
def rooms_per_floor := 10
def hours_per_room := 6
def hourly_rate := 15
def total_rooms := floors * rooms_per_floor
def total_hours := total_rooms * hours_per_room
def total_earnings := total_hours * hourly_rate

theorem legacy_total_earnings :
  total_earnings = 3600 :=
by
  -- Proof to be filled in
  sorry

end legacy_total_earnings_l134_134679


namespace must_be_negative_when_x_is_negative_l134_134951

open Real

theorem must_be_negative_when_x_is_negative (x : ℝ) (h : x < 0) : x^3 < 0 ∧ -x^4 < 0 := 
by
  sorry

end must_be_negative_when_x_is_negative_l134_134951


namespace marble_cut_percentage_first_week_l134_134776

theorem marble_cut_percentage_first_week :
  ∀ (W1 W2 : ℝ), 
  W1 = W2 / 0.70 → 
  W2 = 124.95 / 0.85 → 
  (300 - W1) / 300 * 100 = 30 :=
by
  intros W1 W2 h1 h2
  sorry

end marble_cut_percentage_first_week_l134_134776


namespace not_divisible_l134_134983

theorem not_divisible (n k : ℕ) : ¬ (5 ^ n + 1) ∣ (5 ^ k - 1) :=
sorry

end not_divisible_l134_134983


namespace odd_function_neg_value_l134_134024

theorem odd_function_neg_value
  (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_pos : ∀ x, 0 ≤ x → f x = x^2 + 2 * x) :
  ∀ x, x < 0 → f x = -x^2 + 2 * x :=
by
  intros x hx
  -- The proof would go here
  sorry

end odd_function_neg_value_l134_134024


namespace fraction_students_say_like_actually_dislike_l134_134571

theorem fraction_students_say_like_actually_dislike :
  let n := 200
  let p_l := 0.70
  let p_d := 0.30
  let p_ll := 0.85
  let p_ld := 0.15
  let p_dd := 0.80
  let p_dl := 0.20
  let num_like := p_l * n
  let num_dislike := p_d * n
  let num_ll := p_ll * num_like
  let num_ld := p_ld * num_like
  let num_dd := p_dd * num_dislike
  let num_dl := p_dl * num_dislike
  let total_say_like := num_ll + num_dl
  (num_dl / total_say_like) = 12 / 131 := 
by
  sorry

end fraction_students_say_like_actually_dislike_l134_134571


namespace first_group_men_l134_134252

theorem first_group_men (M : ℕ) (h : M * 15 = 25 * 24) : M = 40 := sorry

end first_group_men_l134_134252


namespace count_perfect_squares_diff_two_consecutive_squares_l134_134031

theorem count_perfect_squares_diff_two_consecutive_squares:
  (∃ n : ℕ, n = 71 ∧ 
            ∀ a : ℕ, (a < 20000 → 
            (∃ b : ℕ, a^2 = (b+1)^2 - b^2))) :=
sorry

end count_perfect_squares_diff_two_consecutive_squares_l134_134031


namespace overall_percentage_gain_is_0_98_l134_134095

noncomputable def original_price : ℝ := 100
noncomputable def increased_price := original_price * 1.32
noncomputable def after_first_discount := increased_price * 0.90
noncomputable def final_price := after_first_discount * 0.85
noncomputable def overall_gain := final_price - original_price
noncomputable def overall_percentage_gain := (overall_gain / original_price) * 100

theorem overall_percentage_gain_is_0_98 :
  overall_percentage_gain = 0.98 := by
  sorry

end overall_percentage_gain_is_0_98_l134_134095


namespace average_cost_is_2_l134_134707

noncomputable def total_amount_spent (apples_quantity bananas_quantity oranges_quantity apples_cost bananas_cost oranges_cost : ℕ) : ℕ :=
  apples_quantity * apples_cost + bananas_quantity * bananas_cost + oranges_quantity * oranges_cost

noncomputable def total_number_of_fruits (apples_quantity bananas_quantity oranges_quantity : ℕ) : ℕ :=
  apples_quantity + bananas_quantity + oranges_quantity

noncomputable def average_cost (apples_quantity bananas_quantity oranges_quantity apples_cost bananas_cost oranges_cost : ℕ) : ℚ :=
  (total_amount_spent apples_quantity bananas_quantity oranges_quantity apples_cost bananas_cost oranges_cost : ℚ) /
  (total_number_of_fruits apples_quantity bananas_quantity oranges_quantity : ℚ)

theorem average_cost_is_2 :
  average_cost 12 4 4 2 1 3 = 2 := 
by
  sorry

end average_cost_is_2_l134_134707


namespace necessary_and_sufficient_condition_for_tangency_l134_134161

-- Given conditions
variables (ρ θ D E : ℝ)

-- Definition of the circle in polar coordinates and the condition for tangency with the radial axis
def circle_eq : Prop := ρ = D * Real.cos θ + E * Real.sin θ

-- Statement of the proof problem
theorem necessary_and_sufficient_condition_for_tangency :
  (circle_eq ρ θ D E) → (D = 0 ∧ E ≠ 0) :=
sorry

end necessary_and_sufficient_condition_for_tangency_l134_134161


namespace quadratic_rewrite_ab_value_l134_134171

theorem quadratic_rewrite_ab_value:
  ∃ a b c : ℤ, (∀ x: ℝ, 16*x^2 + 40*x + 18 = (a*x + b)^2 + c) ∧ a * b = 20 :=
by
  -- We'll add the definitions derived from conditions here
  sorry

end quadratic_rewrite_ab_value_l134_134171


namespace cost_per_trip_l134_134355

theorem cost_per_trip (cost_per_pass : ℕ) (num_passes : ℕ) (trips_oldest : ℕ) (trips_youngest : ℕ) :
    cost_per_pass = 100 →
    num_passes = 2 →
    trips_oldest = 35 →
    trips_youngest = 15 →
    (cost_per_pass * num_passes) / (trips_oldest + trips_youngest) = 4 := by
  sorry

end cost_per_trip_l134_134355


namespace symmetric_points_x_axis_l134_134791

theorem symmetric_points_x_axis (m n : ℤ) (h1 : m + 1 = 1) (h2 : 3 = -(n - 2)) : m - n = 1 :=
by
  sorry

end symmetric_points_x_axis_l134_134791


namespace main_l134_134367

def prop_p (x0 : ℝ) : Prop := x0 > -2 ∧ 6 + abs x0 = 5
def p : Prop := ∃ x : ℝ, prop_p x

def q : Prop := ∀ x : ℝ, x < 0 → x^2 + 4 / x^2 ≥ 4

def r : Prop := ∀ x y : ℝ, abs x + abs y ≤ 1 → abs y / (abs x + 2) ≤ 1 / 2
def not_r : Prop := ∃ x y : ℝ, abs x + abs y > 1 ∧ abs y / (abs x + 2) > 1 / 2

theorem main : ¬ p ∧ ¬ p ∨ r ∧ (p ∧ q) := by
  sorry

end main_l134_134367


namespace correctly_subtracted_value_l134_134083

theorem correctly_subtracted_value (x : ℤ) (h1 : 122 = x - 64) : 
  x - 46 = 140 :=
by
  -- Proof goes here
  sorry

end correctly_subtracted_value_l134_134083


namespace shortest_path_correct_l134_134400

noncomputable def shortest_path_length (length width height : ℕ) : ℝ :=
  let diagonal := Real.sqrt ((length + height)^2 + width^2)
  Real.sqrt 145

theorem shortest_path_correct :
  ∀ (length width height : ℕ),
    length = 4 → width = 5 → height = 4 →
    shortest_path_length length width height = Real.sqrt 145 :=
by
  intros length width height h1 h2 h3
  rw [h1, h2, h3]
  sorry

end shortest_path_correct_l134_134400


namespace max_non_managers_l134_134110

-- Definitions of the problem conditions
variable (m n : ℕ)
variable (h : m = 8)
variable (hratio : (7:ℚ) / 24 < m / n)

-- The theorem we need to prove
theorem max_non_managers (m n : ℕ) (h : m = 8) (hratio : ((7:ℚ) / 24 < m / n)) :
  n ≤ 27 := 
sorry

end max_non_managers_l134_134110


namespace sufficient_but_not_necessary_condition_l134_134973

variable {x k : ℝ}

def p (x k : ℝ) : Prop := x ≥ k
def q (x : ℝ) : Prop := (2 - x) / (x + 1) < 0

theorem sufficient_but_not_necessary_condition (h_suff : ∀ x, p x k → q x) (h_not_necessary : ∃ x, q x ∧ ¬p x k) : k > 2 :=
sorry

end sufficient_but_not_necessary_condition_l134_134973


namespace expression_evaluation_l134_134549

theorem expression_evaluation :
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) = 5^128 - 4^128 :=
sorry

end expression_evaluation_l134_134549


namespace minimize_quadratic_l134_134001

theorem minimize_quadratic (y : ℝ) : 
  ∃ m, m = 3 * y ^ 2 - 18 * y + 11 ∧ 
       (∀ z : ℝ, 3 * z ^ 2 - 18 * z + 11 ≥ m) ∧ 
       m = -16 := 
sorry

end minimize_quadratic_l134_134001


namespace F_is_even_l134_134461

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

noncomputable def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

noncomputable def F (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  (x^3 - 2*x) * f x

theorem F_is_even (f : ℝ → ℝ) (h_odd : is_odd_function f) (h_nonzero : f 1 ≠ 0) :
  is_even_function (F f) :=
sorry

end F_is_even_l134_134461


namespace time_to_cross_signal_pole_l134_134341

-- Given conditions
def length_of_train : ℝ := 300
def time_to_cross_platform : ℝ := 39
def length_of_platform : ℝ := 1162.5

-- The question to prove
theorem time_to_cross_signal_pole :
  (length_of_train / ((length_of_train + length_of_platform) / time_to_cross_platform)) = 8 :=
by
  sorry

end time_to_cross_signal_pole_l134_134341


namespace potassium_bromate_molecular_weight_l134_134072

def potassium_atomic_weight : Real := 39.10
def bromine_atomic_weight : Real := 79.90
def oxygen_atomic_weight : Real := 16.00
def oxygen_atoms : Nat := 3

theorem potassium_bromate_molecular_weight :
  potassium_atomic_weight + bromine_atomic_weight + oxygen_atoms * oxygen_atomic_weight = 167.00 :=
by
  sorry

end potassium_bromate_molecular_weight_l134_134072


namespace bob_shuck_2_hours_l134_134773

def shucking_rate : ℕ := 10  -- oysters per 5 minutes
def minutes_per_hour : ℕ := 60
def hours : ℕ := 2
def minutes : ℕ := hours * minutes_per_hour
def interval : ℕ := 5  -- minutes per interval
def intervals : ℕ := minutes / interval
def num_oysters (intervals : ℕ) : ℕ := intervals * shucking_rate

theorem bob_shuck_2_hours : num_oysters intervals = 240 := by
  -- leave the proof as an exercise
  sorry

end bob_shuck_2_hours_l134_134773


namespace no_solution_to_a_l134_134427

theorem no_solution_to_a (x : ℝ) :
  (4 * x - 1) / 6 - (5 * x - 2 / 3) / 10 + (9 - x / 2) / 3 ≠ 101 / 20 := 
sorry

end no_solution_to_a_l134_134427


namespace find_three_digit_numbers_l134_134796
open Nat

theorem find_three_digit_numbers (n : ℕ) (h1 : 100 ≤ n) (h2 : n < 1000) (h3 : ∀ (k : ℕ), n^k % 1000 = n % 1000) : n = 625 ∨ n = 376 :=
sorry

end find_three_digit_numbers_l134_134796


namespace nonnegative_integer_solutions_l134_134579

theorem nonnegative_integer_solutions :
  {ab : ℕ × ℕ | 3 * 2^ab.1 + 1 = ab.2^2} = {(0, 2), (3, 5), (4, 7)} :=
by
  sorry

end nonnegative_integer_solutions_l134_134579


namespace arithmetic_sequence_primes_l134_134623

theorem arithmetic_sequence_primes (a : ℕ) (d : ℕ) (primes_seq : ∀ n : ℕ, n < 15 → Nat.Prime (a + n * d))
  (distinct_primes : ∀ m n : ℕ, m < 15 → n < 15 → m ≠ n → a + m * d ≠ a + n * d) :
  d > 30000 := 
sorry

end arithmetic_sequence_primes_l134_134623


namespace fare_collected_from_I_class_l134_134261

theorem fare_collected_from_I_class (x y : ℕ) 
  (h_ratio_passengers : 4 * x = 4 * x) -- ratio of passengers 1:4
  (h_ratio_fare : 3 * y = 3 * y) -- ratio of fares 3:1
  (h_total_fare : 7 * 3 * x * y = 224000) -- total fare Rs. 224000
  : 3 * x * y = 96000 := 
by
  sorry

end fare_collected_from_I_class_l134_134261


namespace find_k_value_l134_134558

def line (k : ℝ) (x y : ℝ) : Prop := 3 - 2 * k * x = -4 * y

def on_line (k : ℝ) : Prop := line k 5 (-2)

theorem find_k_value (k : ℝ) : on_line k → k = -0.5 :=
by
  sorry

end find_k_value_l134_134558


namespace sequence_positive_from_26_l134_134135

-- Define the sequence with its general term formula.
def a_n (n : ℕ) : ℤ := 4 * n - 102

-- State the theorem that for all n ≥ 26, a_n > 0.
theorem sequence_positive_from_26 (n : ℕ) (h : n ≥ 26) : a_n n > 0 := by
  sorry

end sequence_positive_from_26_l134_134135


namespace worksheets_turned_in_l134_134621

def initial_worksheets : ℕ := 34
def graded_worksheets : ℕ := 7
def remaining_worksheets : ℕ := initial_worksheets - graded_worksheets
def current_worksheets : ℕ := 63

theorem worksheets_turned_in :
  current_worksheets - remaining_worksheets = 36 :=
by
  sorry

end worksheets_turned_in_l134_134621


namespace sphere_radius_l134_134304

theorem sphere_radius (R : ℝ) (h : 4 * Real.pi * R^2 = 4 * Real.pi) : R = 1 :=
by
  sorry

end sphere_radius_l134_134304


namespace boat_man_mass_l134_134852

theorem boat_man_mass (L B h : ℝ) (rho g : ℝ): 
  L = 3 → B = 2 → h = 0.015 → rho = 1000 → g = 9.81 → (rho * L * B * h * g) / g = 9 :=
by
  intros
  simp_all
  sorry

end boat_man_mass_l134_134852


namespace ryan_weekly_commuting_time_l134_134848

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

end ryan_weekly_commuting_time_l134_134848


namespace inverse_of_g_l134_134285

theorem inverse_of_g : 
  ∀ (g g_inv : ℝ → ℝ) (p q r s : ℝ),
  (∀ x, g x = (3 * x - 2) / (x + 4)) →
  (∀ x, g_inv x = (p * x + q) / (r * x + s)) →
  (∀ x, g (g_inv x) = x) →
  q / s = 2 / 3 :=
by
  intros g g_inv p q r s h_g h_g_inv h_g_ginv
  sorry

end inverse_of_g_l134_134285


namespace daniel_candy_removal_l134_134112

theorem daniel_candy_removal (n k : ℕ) (h1 : n = 24) (h2 : k = 4) : ∃ m : ℕ, n % k = 0 → m = 0 :=
by
  sorry

end daniel_candy_removal_l134_134112


namespace problem1_problem2_l134_134612

theorem problem1 (a b : ℤ) (h₁ : |a| = 5) (h₂ : |b| = 2) (h₃ : a > b) : a + b = 7 ∨ a + b = 3 := 
by sorry

theorem problem2 (a b : ℤ) (h₁ : |a| = 5) (h₂ : |b| = 2) (h₃ : |a + b| = |a| - |b|) : (a = -5 ∧ b = 2) ∨ (a = 5 ∧ b = -2) := 
by sorry

end problem1_problem2_l134_134612


namespace x_minus_y_value_l134_134085

theorem x_minus_y_value (x y : ℝ) (h1 : x^2 = 4) (h2 : |y| = 3) (h3 : x + y < 0) : x - y = 1 ∨ x - y = 5 := by
  sorry

end x_minus_y_value_l134_134085


namespace determine_m_l134_134894

theorem determine_m 
  (f : ℝ → ℝ) 
  (m : ℕ) 
  (h_nat: 0 < m) 
  (h_f: ∀ x, f x = x ^ (m^2 - 2 * m - 3)) 
  (h_no_intersection: ∀ x, f x ≠ 0) 
  (h_symmetric_origin : ∀ x, f (-x) = -f x) : 
  m = 2 :=
by
  sorry

end determine_m_l134_134894


namespace area_of_triangle_BFE_l134_134188

theorem area_of_triangle_BFE (A B C D E F : ℝ × ℝ) (u v : ℝ) 
  (h_rectangle : (0, 0) = A ∧ (3 * u, 0) = B ∧ (3 * u, 3 * v) = C ∧ (0, 3 * v) = D)
  (h_E : E = (0, 2 * v))
  (h_F : F = (2 * u, 0))
  (h_area_rectangle : 3 * u * 3 * v = 48) :
  ∃ (area : ℝ), area = |3 * u * 2 * v| / 2 ∧ area = 24 :=
by 
  sorry

end area_of_triangle_BFE_l134_134188


namespace range_of_m_l134_134404

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Define A as the set of real numbers satisfying 2x^2 - x = 0
def A : Set ℝ := {x | 2 * x^2 - x = 0}

-- Define B based on the parameter m as the set of real numbers satisfying mx^2 - mx - 1 = 0
def B (m : ℝ) : Set ℝ := {x | m * x^2 - m * x - 1 = 0}

-- Define the condition (¬U A) ∩ B = ∅
def condition (m : ℝ) : Prop := (U \ A) ∩ B m = ∅

theorem range_of_m : ∀ m : ℝ, condition m → -4 ≤ m ∧ m ≤ 0 :=
by
  sorry

end range_of_m_l134_134404


namespace rabbit_travel_time_l134_134966

theorem rabbit_travel_time :
  let distance := 2
  let speed := 5
  let hours_to_minutes := 60
  (distance / speed) * hours_to_minutes = 24 := by
sorry

end rabbit_travel_time_l134_134966


namespace find_quadratic_eq_l134_134678

theorem find_quadratic_eq (x y : ℝ) 
  (h₁ : x + y = 10)
  (h₂ : |x - y| = 6) :
  x^2 - 10 * x + 16 = 0 :=
sorry

end find_quadratic_eq_l134_134678
