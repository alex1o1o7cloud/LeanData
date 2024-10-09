import Mathlib

namespace mean_goals_correct_l1897_189764

-- Definitions based on problem conditions
def players_with_3_goals := 4
def players_with_4_goals := 3
def players_with_5_goals := 1
def players_with_6_goals := 2

-- The total number of goals scored
def total_goals := (3 * players_with_3_goals) + (4 * players_with_4_goals) + (5 * players_with_5_goals) + (6 * players_with_6_goals)

-- The total number of players
def total_players := players_with_3_goals + players_with_4_goals + players_with_5_goals + players_with_6_goals

-- The mean number of goals
def mean_goals := total_goals.toFloat / total_players.toFloat

theorem mean_goals_correct : mean_goals = 4.1 := by
  sorry

end mean_goals_correct_l1897_189764


namespace parabola_directrix_l1897_189712

theorem parabola_directrix
  (p : ℝ) (hp : p > 0)
  (O : ℝ × ℝ := (0,0))
  (Focus_F : ℝ × ℝ := (p / 2, 0))
  (Point_P : ℝ × ℝ)
  (Point_Q : ℝ × ℝ)
  (H1 : Point_P.1 = p / 2 ∧ Point_P.2^2 = 2 * p * Point_P.1)
  (H2 : Point_P.1 = Point_P.1) -- This comes out of the perpendicularity of PF to x-axis
  (H3 : Point_Q.2 = 0)
  (H4 : ∃ k_OP slope_OP, slope_OP = 2 ∧ ∃ k_PQ slope_PQ, slope_PQ = -1 / 2 ∧ k_OP * k_PQ = -1)
  (H5 : abs (Point_Q.1 - Focus_F.1) = 6) :
  x = -3 / 2 := 
sorry

end parabola_directrix_l1897_189712


namespace orange_preference_percentage_l1897_189740

theorem orange_preference_percentage 
  (red blue green yellow purple orange : ℕ)
  (total : ℕ)
  (h_red : red = 75)
  (h_blue : blue = 80)
  (h_green : green = 50)
  (h_yellow : yellow = 45)
  (h_purple : purple = 60)
  (h_orange : orange = 55)
  (h_total : total = red + blue + green + yellow + purple + orange) :
  (orange * 100) / total = 15 :=
by
sorry

end orange_preference_percentage_l1897_189740


namespace man_age_difference_l1897_189794

theorem man_age_difference (S M : ℕ) (h1 : S = 22) (h2 : M + 2 = 2 * (S + 2)) : M - S = 24 :=
by
  sorry

end man_age_difference_l1897_189794


namespace nancy_crystal_beads_l1897_189747

-- Definitions of given conditions
def price_crystal : ℕ := 9
def price_metal : ℕ := 10
def sets_metal : ℕ := 2
def total_spent : ℕ := 29

-- Statement of the proof problem
theorem nancy_crystal_beads : ∃ x : ℕ, price_crystal * x + price_metal * sets_metal = total_spent ∧ x = 1 := by
  sorry

end nancy_crystal_beads_l1897_189747


namespace exists_three_numbers_sum_to_zero_l1897_189744

theorem exists_three_numbers_sum_to_zero (s : Finset ℤ) (h_card : s.card = 101) (h_abs : ∀ x ∈ s, |x| ≤ 99) :
  ∃ (a b c : ℤ), a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b + c = 0 :=
by {
  sorry
}

end exists_three_numbers_sum_to_zero_l1897_189744


namespace sum_of_coefficients_l1897_189735

-- Define the polynomial P(x)
def P (x : ℝ) : ℝ := 5 * (2 * x^8 - 3 * x^5 + 9 * x^3 - 6) + 4 * (7 * x^6 - 2 * x^3 + 8)

-- Statement to prove that the sum of the coefficients of P(x) is 62
theorem sum_of_coefficients : P 1 = 62 := sorry

end sum_of_coefficients_l1897_189735


namespace machine_A_sprockets_per_hour_l1897_189748

theorem machine_A_sprockets_per_hour :
  ∀ (A T : ℝ),
    (T > 0 ∧
    (∀ P Q, P = 1.1 * A ∧ Q = 330 / P ∧ Q = 330 / A + 10) →
      A = 3) := 
by
  intro A T
  intro h
  sorry

end machine_A_sprockets_per_hour_l1897_189748


namespace angle_between_plane_and_base_l1897_189716

variable (α k : ℝ)
variable (hα : ∀ S A B C : ℝ, S ≠ A ∧ S ≠ B ∧ S ≠ C ∧ A ≠ B ∧ A ≠ C ∧ B ≠ C)
variable (h_ratio : ∀ A D S : ℝ, AD / DS = k)

theorem angle_between_plane_and_base (α k : ℝ) 
  (hα : ∀ S A B C : ℝ, S ≠ A ∧ S ≠ B ∧ S ≠ C ∧ A ≠ B ∧ A ≠ C ∧ B ≠ C)
  (h_ratio : ∀ A D S : ℝ, AD / DS = k) 
  : ∃ γ : ℝ, γ = Real.arctan (k / (k + 3) * Real.tan α) :=
by
  sorry

end angle_between_plane_and_base_l1897_189716


namespace democrats_ratio_l1897_189726

theorem democrats_ratio (F M: ℕ) 
  (h_total_participants : F + M = 810)
  (h_female_democrats : 135 * 2 = F)
  (h_male_democrats : (1 / 4) * M = 135) : 
  (270 / 810 = 1 / 3) :=
by 
  sorry

end democrats_ratio_l1897_189726


namespace cyclist_speed_north_l1897_189724

theorem cyclist_speed_north (v : ℝ) :
  (∀ d t : ℝ, d = 50 ∧ t = 1 ∧ 40 * t + v * t = d) → v = 10 :=
by
  sorry

end cyclist_speed_north_l1897_189724


namespace christina_rearrangements_l1897_189730

-- define the main conditions
def rearrangements (n : Nat) : Nat := Nat.factorial n

def half (n : Nat) : Nat := n / 2

def time_for_first_half (r : Nat) : Nat := r / 12

def time_for_second_half (r : Nat) : Nat := r / 18

def total_time_in_minutes (t1 t2 : Nat) : Nat := t1 + t2

def total_time_in_hours (t : Nat) : Nat := t / 60

-- statement proving that the total time will be 420 hours
theorem christina_rearrangements : 
  rearrangements 9 = 362880 →
  half (rearrangements 9) = 181440 →
  time_for_first_half 181440 = 15120 →
  time_for_second_half 181440 = 10080 →
  total_time_in_minutes 15120 10080 = 25200 →
  total_time_in_hours 25200 = 420 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end christina_rearrangements_l1897_189730


namespace tangential_quadrilateral_perpendicular_diagonals_l1897_189715

-- Define what it means for a quadrilateral to be tangential
def is_tangential_quadrilateral (a b c d : ℝ) : Prop :=
  a + c = b + d

-- Define what it means for a quadrilateral to be a kite
def is_kite (a b c d : ℝ) : Prop :=
  a = b ∧ c = d

-- Define what it means for the diagonals of a quadrilateral to be perpendicular
def diagonals_perpendicular (a b c d : ℝ) : Prop :=
  sorry -- Actual geometric definition needs to be elaborated

-- Main statement to prove
theorem tangential_quadrilateral_perpendicular_diagonals (a b c d : ℝ) :
  is_tangential_quadrilateral a b c d → 
  (diagonals_perpendicular a b c d ↔ is_kite a b c d) := 
sorry

end tangential_quadrilateral_perpendicular_diagonals_l1897_189715


namespace difference_between_new_and_original_l1897_189711

variables (x y : ℤ) -- Declaring variables x and y as integers

-- The original number is represented as 10*x + y, and the new number after swapping is 10*y + x.
-- We need to prove that the difference between the new number and the original number is -9*x + 9*y.
theorem difference_between_new_and_original (x y : ℤ) :
  (10 * y + x) - (10 * x + y) = -9 * x + 9 * y :=
by
  sorry -- Proof placeholder

end difference_between_new_and_original_l1897_189711


namespace jelly_beans_in_jar_y_l1897_189704

-- Definitions of the conditions
def total_beans : ℕ := 1200
def number_beans_in_jar_y (y : ℕ) := y
def number_beans_in_jar_x (y : ℕ) := 3 * y - 400

-- The main theorem to be proven
theorem jelly_beans_in_jar_y (y : ℕ) :
  number_beans_in_jar_x y + number_beans_in_jar_y y = total_beans → 
  y = 400 := 
by
  sorry

end jelly_beans_in_jar_y_l1897_189704


namespace parallel_lines_regular_ngon_l1897_189796

def closed_n_hop_path (n : ℕ) (a : Fin (n + 1) → Fin n) : Prop :=
∀ i j : Fin n, a (i + 1) + a i = a (j + 1) + a j → i = j

theorem parallel_lines_regular_ngon (n : ℕ) (a : Fin (n + 1) → Fin n):
  (Even n → ∃ i j : Fin n, i ≠ j ∧ a (i + 1) + a i = a (j + 1) + a j) ∧
  (Odd n → ¬(∃ i j : Fin n, i ≠ j ∧ a (i + 1) + a i = a (j + 1) + a j ∧ ∀ k l : Fin n, k ≠ l → a (k + 1) + k ≠ a (l + 1) + l)) :=
by
  sorry

end parallel_lines_regular_ngon_l1897_189796


namespace fraction_equals_half_l1897_189737

def numerator : ℤ := 1 - 2 + 4 - 8 + 16 - 32 + 64
def denominator : ℤ := 2 - 4 + 8 - 16 + 32 - 64 + 128

theorem fraction_equals_half : (numerator : ℚ) / (denominator : ℚ) = 1 / 2 :=
by
  sorry

end fraction_equals_half_l1897_189737


namespace perpendicular_and_intersection_l1897_189738

variables (x y : ℚ)

def line1 := 4 * y - 3 * x = 15
def line4 := 3 * y + 4 * x = 15

theorem perpendicular_and_intersection :
  (4 * y - 3 * x = 15) ∧ (3 * y + 4 * x = 15) →
  let m1 := (3 : ℚ) / 4
  let m4 := -(4 : ℚ) / 3
  m1 * m4 = -1 ∧
  ∃ x y : ℚ, 4*y - 3*x = 15 ∧ 3*y + 4*x = 15 ∧ x = 15/32 ∧ y = 35/8 :=
by
  sorry

end perpendicular_and_intersection_l1897_189738


namespace least_stamps_l1897_189707

theorem least_stamps (s t : ℕ) (h : 5 * s + 7 * t = 48) : s + t = 8 :=
by sorry

end least_stamps_l1897_189707


namespace count_yellow_balls_l1897_189799

theorem count_yellow_balls (total white green yellow red purple : ℕ) (prob : ℚ)
  (h_total : total = 100)
  (h_white : white = 50)
  (h_green : green = 30)
  (h_red : red = 9)
  (h_purple : purple = 3)
  (h_prob : prob = 0.88) :
  yellow = 8 :=
by
  -- The proof will be here
  sorry

end count_yellow_balls_l1897_189799


namespace percent_correct_l1897_189761

theorem percent_correct (x : ℕ) : 
  (5 * 100.0 / 7) = 71.43 :=
by
  sorry

end percent_correct_l1897_189761


namespace proof_part1_proof_part2_l1897_189789

-- Proof problem for the first part (1)
theorem proof_part1 (m : ℝ) : m^3 * m^6 + (-m^3)^3 = 0 := 
by
  sorry

-- Proof problem for the second part (2)
theorem proof_part2 (a : ℝ) : a * (a - 2) - 2 * a * (1 - 3 * a) = 7 * a^2 - 4 * a := 
by
  sorry

end proof_part1_proof_part2_l1897_189789


namespace no_solutions_to_equation_l1897_189786

theorem no_solutions_to_equation : ¬∃ x : ℝ, (x ≠ 0) ∧ (x ≠ 5) ∧ ((2 * x ^ 2 - 10 * x) / (x ^ 2 - 5 * x) = x - 3) :=
by
  sorry

end no_solutions_to_equation_l1897_189786


namespace area_of_triangle_ABC_l1897_189702

theorem area_of_triangle_ABC (A B C : ℝ) (a b c : ℝ) 
  (h1 : b = 2) (h2 : c = 3) (h3 : C = 2 * B): 
  ∃ S : ℝ, S = 1/2 * b * c * (Real.sin A) ∧ S = 15 * (Real.sqrt 7) / 16 :=
by
  sorry

end area_of_triangle_ABC_l1897_189702


namespace find_speed_range_l1897_189773

noncomputable def runningErrorB (v : ℝ) : ℝ := abs ((300 / v) - 7)
noncomputable def runningErrorC (v : ℝ) : ℝ := abs ((480 / v) - 11)

theorem find_speed_range (v : ℝ) :
  (runningErrorB v + runningErrorC v ≤ 2) →
  33.33 ≤ v ∧ v ≤ 48.75 := sorry

end find_speed_range_l1897_189773


namespace mass_percentage_O_in_CaO_l1897_189734

theorem mass_percentage_O_in_CaO :
  (16.00 / (40.08 + 16.00)) * 100 = 28.53 :=
by
  sorry

end mass_percentage_O_in_CaO_l1897_189734


namespace eleventh_term_of_sequence_l1897_189776

def inversely_proportional_sequence (a : ℕ → ℝ) (c : ℝ) : Prop :=
  ∀ n : ℕ, a n * a (n + 1) = c

theorem eleventh_term_of_sequence :
  ∃ a : ℕ → ℝ,
    (a 1 = 3) ∧
    (a 2 = 6) ∧
    inversely_proportional_sequence a 18 ∧
    a 11 = 3 :=
by
  sorry

end eleventh_term_of_sequence_l1897_189776


namespace total_pages_read_l1897_189708

-- Define the average pages read by Lucas for the first four days.
def day1_4_avg : ℕ := 42

-- Define the average pages read by Lucas for the next two days.
def day5_6_avg : ℕ := 50

-- Define the pages read on the last day.
def day7 : ℕ := 30

-- Define the total number of days for which measurement is provided.
def total_days : ℕ := 7

-- Prove that the total number of pages Lucas read is 298.
theorem total_pages_read : 
  4 * day1_4_avg + 2 * day5_6_avg + day7 = 298 := 
by 
  sorry

end total_pages_read_l1897_189708


namespace triangle_inequality_l1897_189779

theorem triangle_inequality (a b c S : ℝ)
  (h : a ≠ b ∧ b ≠ c ∧ c ≠ a)   -- a, b, c are sides of a non-isosceles triangle
  (S_def : S = Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))) :
  (a^3) / ((a - b) * (a - c)) + (b^3) / ((b - c) * (b - a)) + (c^3) / ((c - a) * (c - b)) > 2 * 3^(3/4) * S :=
by
  sorry

end triangle_inequality_l1897_189779


namespace inequality_ab2_bc2_ca2_leq_27_div_8_l1897_189797

theorem inequality_ab2_bc2_ca2_leq_27_div_8 (a b c : ℝ) (h : a ≥ b) (h1 : b ≥ c) (h2 : c ≥ 0) (h3 : a + b + c = 3) :
  ab^2 + bc^2 + ca^2 ≤ 27 / 8 :=
sorry

end inequality_ab2_bc2_ca2_leq_27_div_8_l1897_189797


namespace multiple_of_second_number_l1897_189759

def main : IO Unit := do
  IO.println s!"Proof problem statement in Lean 4."

theorem multiple_of_second_number (x m : ℕ) 
  (h1 : 19 = m * x + 3) 
  (h2 : 19 + x = 27) : 
  m = 2 := 
sorry

end multiple_of_second_number_l1897_189759


namespace twins_age_l1897_189714

theorem twins_age (x : ℕ) (h : (x + 1) * (x + 1) = x * x + 5) : x = 2 :=
by
  sorry

end twins_age_l1897_189714


namespace range_of_a_l1897_189725

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ (∃ x : ℝ, x^2 + 2 * a * x + a + 2 = 0) → a ≤ -1 :=
by
  sorry

end range_of_a_l1897_189725


namespace proof_of_a_b_and_T_l1897_189768

-- Define sequences and the given conditions

def a (n : ℕ) : ℕ := 2^(n-1)

def b (n : ℕ) : ℕ := 2 * n

def S (n : ℕ) : ℕ := 2^n - 1

def c (n : ℕ) : ℚ := 1 / ((b n)^2 - 1)

def T (n : ℕ) : ℚ := (n : ℚ) / (2 * n + 1)

axiom b_condition : ∀ n : ℕ, n > 0 → (b n + 2 * n = 2 * (b (n-1)) + 4)

axiom S_condition : ∀ n : ℕ, S n = 2^n - 1

theorem proof_of_a_b_and_T (n : ℕ) (h : n > 0) : 
  (∀ k, a k = 2^(k-1)) ∧ 
  (∀ k, b k = 2 * k) ∧ 
  (∀ k, T k = (k : ℚ) / (2 * k + 1)) := by
  sorry

end proof_of_a_b_and_T_l1897_189768


namespace necessary_not_sufficient_l1897_189705

-- Define the function y = x^2 - 2ax + 1
def quadratic_function (a x : ℝ) : ℝ := x^2 - 2 * a * x + 1

-- Define strict monotonicity on the interval [1, +∞)
def strictly_increasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ I → y ∈ I → x < y → f x < f y

-- Define the condition for the function to be strictly increasing on [1, +∞)
def condition_strict_increasing (a : ℝ) : Prop :=
  strictly_increasing_on (quadratic_function a) (Set.Ici 1)

-- The condition to prove
theorem necessary_not_sufficient (a : ℝ) :
  condition_strict_increasing a → (a ≤ 0) := sorry

end necessary_not_sufficient_l1897_189705


namespace hyperbola_real_axis_length_l1897_189778

theorem hyperbola_real_axis_length :
  (∃ a : ℝ, (∀ x y : ℝ, (x^2 / 9 - y^2 = 1) → (2 * a = 6))) :=
sorry

end hyperbola_real_axis_length_l1897_189778


namespace domain_of_f_l1897_189781

theorem domain_of_f (x : ℝ) : (1 - x > 0) ∧ (2 * x + 1 > 0) ↔ - (1 / 2 : ℝ) < x ∧ x < 1 :=
by
  sorry

end domain_of_f_l1897_189781


namespace range_of_a_l1897_189758

theorem range_of_a (a : ℝ) : (∀ x ∈ Set.Icc (-2 : ℝ) 3, 2 * x > x ^ 2 + a) → a < -8 :=
by sorry

end range_of_a_l1897_189758


namespace area_proportions_and_point_on_line_l1897_189784

theorem area_proportions_and_point_on_line (T : ℝ × ℝ) :
  (∃ r s : ℝ, T = (r, s) ∧ s = -(5 / 3) * r + 10 ∧ 1 / 2 * 6 * s = 7.5) 
  ↔ T.1 + T.2 = 7 :=
by { sorry }

end area_proportions_and_point_on_line_l1897_189784


namespace acorns_given_is_correct_l1897_189783

-- Define initial conditions
def initial_acorns : ℕ := 16
def remaining_acorns : ℕ := 9

-- Define the number of acorns given to her sister
def acorns_given : ℕ := initial_acorns - remaining_acorns

-- Theorem statement
theorem acorns_given_is_correct : acorns_given = 7 := by
  sorry

end acorns_given_is_correct_l1897_189783


namespace basic_printer_total_price_l1897_189719

theorem basic_printer_total_price (C P : ℝ) (hC : C = 1500) (hP : P = (1/3) * (C + 500 + P)) : C + P = 2500 := 
by
  sorry

end basic_printer_total_price_l1897_189719


namespace find_area_of_plot_l1897_189733

def area_of_plot (B : ℝ) (L : ℝ) (A : ℝ) : Prop :=
  L = 0.75 * B ∧ B = 21.908902300206645 ∧ A = L * B

theorem find_area_of_plot (B L A : ℝ) (h : area_of_plot B L A) : A = 360 := by
  sorry

end find_area_of_plot_l1897_189733


namespace binomial_expansion_terms_l1897_189769

theorem binomial_expansion_terms (x n : ℝ) (hn : n = 8) : 
  ∃ t, t = 3 :=
  sorry

end binomial_expansion_terms_l1897_189769


namespace infinite_geometric_series_sum_l1897_189757

theorem infinite_geometric_series_sum :
  let a := (5 : ℚ) / 3
  let r := -(3 : ℚ) / 4
  ∑' n : ℕ, a * r ^ n = 20 / 21 := by
  sorry

end infinite_geometric_series_sum_l1897_189757


namespace multiply_expression_l1897_189731

variable (y : ℝ)

theorem multiply_expression : 
  (16 * y^3) * (12 * y^5) * (1 / (4 * y)^3) = 3 * y^5 := by
  sorry

end multiply_expression_l1897_189731


namespace truck_distance_on_7_gallons_l1897_189754

theorem truck_distance_on_7_gallons :
  ∀ (d : ℝ) (g₁ g₂ : ℝ), d = 240 → g₁ = 5 → g₂ = 7 → (d / g₁) * g₂ = 336 :=
by
  intros d g₁ g₂ h₁ h₂ h₃
  rw [h₁, h₂, h₃]
  sorry

end truck_distance_on_7_gallons_l1897_189754


namespace katy_brownies_l1897_189774

-- Define the conditions
def ate_monday : ℕ := 5
def ate_tuesday : ℕ := 2 * ate_monday

-- Define the question
def total_brownies : ℕ := ate_monday + ate_tuesday

-- State the proof problem
theorem katy_brownies : total_brownies = 15 := by
  sorry

end katy_brownies_l1897_189774


namespace remaining_area_is_correct_l1897_189742

-- Define the given conditions:
def original_length : ℕ := 25
def original_width : ℕ := 35
def square_side : ℕ := 7

-- Define a function to calculate the area of the original cardboard:
def area_original : ℕ := original_length * original_width

-- Define a function to calculate the area of one square corner:
def area_corner : ℕ := square_side * square_side

-- Define a function to calculate the total area removed:
def total_area_removed : ℕ := 4 * area_corner

-- Define a function to calculate the remaining area:
def area_remaining : ℕ := area_original - total_area_removed

-- The theorem we want to prove:
theorem remaining_area_is_correct : area_remaining = 679 := by
  -- Here, we would provide the proof if required, but we use sorry for now.
  sorry

end remaining_area_is_correct_l1897_189742


namespace cosine_identity_example_l1897_189720

theorem cosine_identity_example {α : ℝ} (h : Real.sin (π / 3 - α) = 1 / 3) : Real.cos (π / 3 + 2 * α) = -7 / 9 :=
by sorry

end cosine_identity_example_l1897_189720


namespace combined_PPF_two_females_combined_PPF_two_males_combined_PPF_male_female_l1897_189703

-- Definition: Combined PPF for two females
theorem combined_PPF_two_females (K : ℝ) (h : K ≤ 40) :
  (∀ K₁ K₂, K = K₁ + K₂ →  40 - 2 * K₁ + 40 - 2 * K₂ = 80 - 2 * K) := sorry

-- Definition: Combined PPF for two males
theorem combined_PPF_two_males (K : ℝ) (h : K ≤ 16) :
  (∀ K₁ K₂, K₁ = 0.5 * K → K₂ = 0.5 * K → 64 - K₁^2 + 64 - K₂^2 = 128 - 0.5 * K^2) := sorry

-- Definition: Combined PPF for one male and one female (piecewise)
theorem combined_PPF_male_female (K : ℝ) :
  (K ≤ 1 → (∀ K₁ K₂, K₁ = K → K₂ = 0 → 64 - K₁^2 + 40 - 2 * K₂ = 104 - K^2)) ∧
  (1 < K ∧ K ≤ 21 → (∀ K₁ K₂, K₁ = 1 → K₂ = K - 1 → 64 - K₁^2 + 40 - 2 * K₂ = 105 - 2 * K)) ∧
  (21 < K ∧ K ≤ 28 → (∀ K₁ K₂, K₁ = K - 20 → K₂ = 20 → 64 - K₁^2 + 40 - 2 * K₂ = 40 * K - K^2 - 336)) := sorry

end combined_PPF_two_females_combined_PPF_two_males_combined_PPF_male_female_l1897_189703


namespace sum_of_x_and_y_l1897_189717

-- Define integers x and y
variables (x y : ℤ)

-- Define conditions
def condition1 : Prop := x - y = 200
def condition2 : Prop := y = 250

-- Define the main statement
theorem sum_of_x_and_y (h1 : condition1 x y) (h2 : condition2 y) : x + y = 700 := 
by
  sorry

end sum_of_x_and_y_l1897_189717


namespace number_of_social_science_papers_selected_is_18_l1897_189766

def total_social_science_papers : ℕ := 54
def total_humanities_papers : ℕ := 60
def total_other_papers : ℕ := 39
def total_selected_papers : ℕ := 51

def number_of_social_science_papers_selected : ℕ :=
  (total_social_science_papers * total_selected_papers) / (total_social_science_papers + total_humanities_papers + total_other_papers)

theorem number_of_social_science_papers_selected_is_18 :
  number_of_social_science_papers_selected = 18 :=
by 
  -- Proof to be provided
  sorry

end number_of_social_science_papers_selected_is_18_l1897_189766


namespace line_through_points_l1897_189790

theorem line_through_points (x1 y1 x2 y2 : ℝ) (h1 : (x1, y1) = (2, 8)) (h2 : (x2, y2) = (5, 2)) :
  ∃ m b : ℝ, (∀ x, y = m * x + b → (x, y) = (2,8) ∨ (x, y) = (5, 2)) ∧ (m + b = 10) :=
by
  sorry

end line_through_points_l1897_189790


namespace kaeli_problems_per_day_l1897_189752

-- Definitions based on conditions
def problems_solved_per_day_marie_pascale : ℕ := 4
def total_problems_marie_pascale : ℕ := 72
def total_problems_kaeli : ℕ := 126

-- Number of days both took should be the same
def number_of_days : ℕ := total_problems_marie_pascale / problems_solved_per_day_marie_pascale

-- Kaeli solves 54 more problems than Marie-Pascale
def extra_problems_kaeli : ℕ := 54

-- Definition that Kaeli's total problems solved is that of Marie-Pascale plus 54
axiom kaeli_total_problems (h : total_problems_marie_pascale + extra_problems_kaeli = total_problems_kaeli) : True

-- Now to find x, the problems solved per day by Kaeli
def x : ℕ := total_problems_kaeli / number_of_days

-- Prove that x = 7
theorem kaeli_problems_per_day (h : total_problems_marie_pascale + extra_problems_kaeli = total_problems_kaeli) : x = 7 := by
  sorry

end kaeli_problems_per_day_l1897_189752


namespace smallest_positive_integer_x_l1897_189772

def smallest_x (x : ℕ) : Prop :=
  x > 0 ∧ (450 * x) % 625 = 0

theorem smallest_positive_integer_x :
  ∃ x : ℕ, smallest_x x ∧ ∀ y : ℕ, smallest_x y → x ≤ y ∧ x = 25 :=
by {
  sorry
}

end smallest_positive_integer_x_l1897_189772


namespace train_crosses_second_platform_in_20_sec_l1897_189722

theorem train_crosses_second_platform_in_20_sec
  (length_train : ℝ)
  (length_first_platform : ℝ)
  (time_first_platform : ℝ)
  (length_second_platform : ℝ)
  (time_second_platform : ℝ):

  length_train = 100 ∧
  length_first_platform = 350 ∧
  time_first_platform = 15 ∧
  length_second_platform = 500 →
  time_second_platform = 20 := by
  sorry

end train_crosses_second_platform_in_20_sec_l1897_189722


namespace group_made_l1897_189770

-- Definitions based on the problem's conditions
def teachers_made : Nat := 28
def total_products : Nat := 93

-- Theorem to prove that the group made 65 recycled materials
theorem group_made : total_products - teachers_made = 65 := by
  sorry

end group_made_l1897_189770


namespace a_n_formula_l1897_189788

variable {a : ℕ+ → ℝ}  -- Defining a_n as a sequence from positive natural numbers to real numbers
variable {S : ℕ+ → ℝ}  -- Defining S_n as a sequence from positive natural numbers to real numbers

-- Given conditions
axiom S_def (n : ℕ+) : S n = a n / 2 + 1 / a n - 1
axiom a_pos (n : ℕ+) : a n > 0

-- Conjecture to be proved
theorem a_n_formula (n : ℕ+) : a n = Real.sqrt (2 * n + 1) - Real.sqrt (2 * n - 1) := 
sorry -- proof to be done

end a_n_formula_l1897_189788


namespace geometric_sequence_fifth_term_is_32_l1897_189755

-- Defining the geometric sequence conditions
variables (a r : ℝ)

def third_term := a * r^2 = 18
def fourth_term := a * r^3 = 24
def fifth_term := a * r^4

theorem geometric_sequence_fifth_term_is_32 (h1 : third_term a r) (h2 : fourth_term a r) : 
  fifth_term a r = 32 := 
by
  sorry

end geometric_sequence_fifth_term_is_32_l1897_189755


namespace find_X_l1897_189777

def r (X Y : ℕ) : ℕ := X^2 + Y^2

theorem find_X (X : ℕ) (h : r X 7 = 338) : X = 17 := by
  sorry

end find_X_l1897_189777


namespace complement_set_P_l1897_189749

open Set

theorem complement_set_P (P : Set ℝ) (hP : P = {x : ℝ | x ≥ 1}) : Pᶜ = {x : ℝ | x < 1} :=
sorry

end complement_set_P_l1897_189749


namespace petya_wins_max_margin_l1897_189746

theorem petya_wins_max_margin {P1 P2 V1 V2 : ℕ} 
  (h1 : P1 = V1 + 9)
  (h2 : V2 = P2 + 9)
  (h3 : P1 + P2 + V1 + V2 = 27)
  (h4 : P1 + P2 > V1 + V2) :
  ∃ m : ℕ, m = 9 ∧ P1 + P2 - (V1 + V2) = m :=
by
  sorry

end petya_wins_max_margin_l1897_189746


namespace survey_population_l1897_189721

-- Definitions based on conditions
def number_of_packages := 10
def dozens_per_package := 10
def sets_per_dozen := 12

-- Derived from conditions
def total_sets := number_of_packages * dozens_per_package * sets_per_dozen

-- Populations for the proof
def population_quality : ℕ := total_sets
def population_satisfaction : ℕ := total_sets

-- Proof statement
theorem survey_population:
  (population_quality = 1200) ∧ (population_satisfaction = 1200) := by
  sorry

end survey_population_l1897_189721


namespace find_circle_center_l1897_189795

-- Define the conditions as hypotheses
def line1 (x y : ℝ) : Prop := 5 * x - 2 * y = 40
def line2 (x y : ℝ) : Prop := 5 * x - 2 * y = 10
def line_center_constraint (x y : ℝ) : Prop := 3 * x - 4 * y = 0

-- Define the function for the equidistant line
def line_eq (x y : ℝ) : Prop := 5 * x - 2 * y = 25

-- Prove that the center of the circle satisfying the given conditions is (50/7, 75/14)
theorem find_circle_center (x y : ℝ) 
(h1 : line_eq x y)
(h2 : line_center_constraint x y) : 
(x = 50 / 7 ∧ y = 75 / 14) :=
sorry

end find_circle_center_l1897_189795


namespace goose_eggs_count_l1897_189741

theorem goose_eggs_count (E : ℕ) 
  (hatch_ratio : ℝ := 1/4)
  (survival_first_month_ratio : ℝ := 4/5)
  (survival_first_year_ratio : ℝ := 3/5)
  (survived_first_year : ℕ := 120) :
  ((survival_first_year_ratio * (survival_first_month_ratio * hatch_ratio * E)) = survived_first_year) → E = 1000 :=
by
  intro h
  sorry

end goose_eggs_count_l1897_189741


namespace inf_coprime_naturals_l1897_189732

theorem inf_coprime_naturals (a b : ℤ) (h : a ≠ b) : 
  ∃ᶠ n in Filter.atTop, Nat.gcd (Int.natAbs (a + n)) (Int.natAbs (b + n)) = 1 := 
sorry

end inf_coprime_naturals_l1897_189732


namespace minimum_positive_period_of_f_is_pi_l1897_189775

noncomputable def f (x : ℝ) : ℝ :=
  (Real.cos (x + Real.pi / 4))^2 - (Real.sin (x + Real.pi / 4))^2

theorem minimum_positive_period_of_f_is_pi :
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T', T' > 0 ∧ (∀ x, f (x + T') = f x) → T' ≥ T) ∧ T = Real.pi :=
sorry

end minimum_positive_period_of_f_is_pi_l1897_189775


namespace find_a_l1897_189765

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 3^x + a / (3^x + 1)

theorem find_a (a : ℝ) : 
  (∀ x : ℝ, 3^x + a / (3^x + 1) ≥ 5) ∧ (∃ x : ℝ, 3^x + a / (3^x + 1) = 5) 
  → a = 9 := 
by 
  intro h
  sorry

end find_a_l1897_189765


namespace factor_correct_l1897_189718

noncomputable def p (b : ℝ) : ℝ := 221 * b^2 + 17 * b
def factored_form (b : ℝ) : ℝ := 17 * b * (13 * b + 1)

theorem factor_correct (b : ℝ) : p b = factored_form b := by
  sorry

end factor_correct_l1897_189718


namespace triangle_inequality_sum_l1897_189760

theorem triangle_inequality_sum (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) :
  (c / (a + b)) + (a / (b + c)) + (b / (c + a)) > 1 :=
by
  sorry

end triangle_inequality_sum_l1897_189760


namespace slope_of_tangent_at_point_l1897_189793

theorem slope_of_tangent_at_point (x : ℝ) (y : ℝ) (h_curve : y = x^3)
    (h_slope : 3*x^2 = 3) : (x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -1) := 
sorry

end slope_of_tangent_at_point_l1897_189793


namespace value_of_mathematics_l1897_189771

def letter_value (n : ℕ) : ℤ :=
  -- The function to assign values based on position modulo 8
  match n % 8 with
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 0
  | 5 => -1
  | 6 => -2
  | 7 => -3
  | 0 => 0
  | _ => 0 -- This case is practically unreachable

def letter_position (c : Char) : ℕ :=
  -- The function to find the position of a character in the alphabet
  c.toNat - 'a'.toNat + 1

def value_of_word (word : String) : ℤ :=
  -- The function to calculate the sum of values of letters in the word
  word.foldr (fun c acc => acc + letter_value (letter_position c)) 0

theorem value_of_mathematics : value_of_word "mathematics" = 6 := 
  by
    sorry -- Proof to be completed

end value_of_mathematics_l1897_189771


namespace expression_simplified_l1897_189787

theorem expression_simplified (d : ℤ) (h : d ≠ 0) :
  let a := 24
  let b := 61
  let c := 96
  a + b + c = 181 ∧ 
  (15 * d ^ 2 + 7 * d + 15 + (3 * d + 9) ^ 2 = a * d ^ 2 + b * d + c) := by
{
  sorry
}

end expression_simplified_l1897_189787


namespace range_of_x_l1897_189753

noncomputable def f (x : ℝ) : ℝ := Real.log (1 + |x|) - 1 / (1 + x^2)

theorem range_of_x :
  ∀ x : ℝ, (f x > f (2*x - 1)) ↔ (1/3 < x ∧ x < 1) :=
by
  sorry

end range_of_x_l1897_189753


namespace q_value_at_2_l1897_189727

def q (x d e : ℤ) : ℤ := x^2 + d*x + e

theorem q_value_at_2 (d e : ℤ) 
  (h1 : ∃ p : ℤ → ℤ, ∀ x, x^4 + 8*x^2 + 49 = (q x d e) * (p x))
  (h2 : ∃ r : ℤ → ℤ, ∀ x, 2*x^4 + 5*x^2 + 36*x + 7 = (q x d e) * (r x)) :
  q 2 d e = 5 := 
sorry

end q_value_at_2_l1897_189727


namespace perfect_square_k_l1897_189791

theorem perfect_square_k (a b k : ℝ) (h : ∃ c : ℝ, a^2 + 2*(k-3)*a*b + 9*b^2 = (a + c*b)^2) : 
  k = 6 ∨ k = 0 := 
sorry

end perfect_square_k_l1897_189791


namespace smallest_part_when_divided_l1897_189763

theorem smallest_part_when_divided (total : ℝ) (a b c : ℝ) (h_total : total = 150)
                                   (h_a : a = 3) (h_b : b = 5) (h_c : c = 7/2) :
                                   min (min (3 * (total / (a + b + c))) (5 * (total / (a + b + c)))) ((7/2) * (total / (a + b + c))) = 3 * (total / (a + b + c)) :=
by
  -- Mathematical steps have been omitted
  sorry

end smallest_part_when_divided_l1897_189763


namespace cannot_be_simultaneous_squares_l1897_189713

theorem cannot_be_simultaneous_squares (x y : ℕ) :
  ¬ (∃ a b : ℤ, x^2 + y = a^2 ∧ y^2 + x = b^2) :=
by
  sorry

end cannot_be_simultaneous_squares_l1897_189713


namespace weight_of_a_l1897_189756

theorem weight_of_a (a b c d e : ℝ)
  (h1 : (a + b + c) / 3 = 84)
  (h2 : (a + b + c + d) / 4 = 80)
  (h3 : e = d + 8)
  (h4 : (b + c + d + e) / 4 = 79) :
  a = 80 :=
by
  sorry

end weight_of_a_l1897_189756


namespace problem_evaluation_l1897_189709

theorem problem_evaluation : (726 * 726) - (725 * 727) = 1 := 
by 
  sorry

end problem_evaluation_l1897_189709


namespace registration_methods_l1897_189767

-- Define the number of students and groups
def num_students : ℕ := 4
def num_groups : ℕ := 3

-- Theorem stating the total number of different registration methods
theorem registration_methods : (num_groups ^ num_students) = 81 := 
by sorry

end registration_methods_l1897_189767


namespace flower_growth_l1897_189706

theorem flower_growth (total_seeds : ℕ) (seeds_per_bed : ℕ) (max_grow_per_bed : ℕ) (h1 : total_seeds = 55) (h2 : seeds_per_bed = 15) (h3 : max_grow_per_bed = 60) : total_seeds ≤ 55 :=
by
  -- use the given conditions
  have h4 : total_seeds = 55 := h1
  sorry -- Proof goes here, omitted as instructed

end flower_growth_l1897_189706


namespace smallest_n_satisfying_conditions_l1897_189780

theorem smallest_n_satisfying_conditions : ∃ (n : ℕ), (n > 0) ∧ (∀ (a b : ℤ), ∃ (α β : ℝ), 
    α ≠ β ∧ (0 < α) ∧ (α < 1) ∧ (0 < β) ∧ (β < 1) ∧ (n * α^2 + a * α + b = 0) ∧ (n * β^2 + a * β + b = 0)
 ) ∧ (∀ (m : ℕ), m > 0 ∧ m < n → ¬ (∀ (a b : ℤ), ∃ (α β : ℝ), 
    α ≠ β ∧ (0 < α) ∧ (α < 1) ∧ (0 < β) ∧ (β < 1) ∧ (m * α^2 + a * α + b = 0) ∧ (m * β^2 + a * β + b = 0))) := 
sorry

end smallest_n_satisfying_conditions_l1897_189780


namespace solve_for_a_l1897_189723

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if h : x >= 0 then 4^x else 2^(a - x)

theorem solve_for_a (a : ℝ) (h : a ≠ 1) (h_f_eq : f a (1 - a) = f a (a - 1)) : a = 1 / 2 :=
by
  sorry

end solve_for_a_l1897_189723


namespace solution_interval_l1897_189750

theorem solution_interval (x : ℝ) : (x^2 / (x - 5)^2 > 0) ↔ (x ∈ Set.Iio 0 ∪ Set.Ioi 0 ∩ Set.Iio 5 ∪ Set.Ioi 5) :=
by
  sorry

end solution_interval_l1897_189750


namespace value_of_coefficients_l1897_189743

theorem value_of_coefficients (a₀ a₁ a₂ a₃ : ℤ) (x : ℤ) :
  (5 * x + 4) ^ 3 = a₀ + a₁ * x + a₂ * x ^ 2 + a₃ * x ^ 3 →
  x = -1 →
  (a₀ + a₂) - (a₁ + a₃) = -1 :=
by
  sorry

end value_of_coefficients_l1897_189743


namespace shares_of_c_l1897_189701

theorem shares_of_c (a b c : ℝ) (h1 : 3 * a = 4 * b) (h2 : 4 * b = 7 * c) (h3 : a + b + c = 427): 
  c = 84 :=
by {
  sorry
}

end shares_of_c_l1897_189701


namespace area_of_trapezoid_l1897_189700

noncomputable def triangle_XYZ_is_isosceles : Prop := 
  ∃ (X Y Z : Type) (XY XZ : ℝ), XY = XZ

noncomputable def identical_smaller_triangles (area : ℝ) (num : ℕ) : Prop := 
  num = 9 ∧ area = 3

noncomputable def total_area_large_triangle (total_area : ℝ) : Prop := 
  total_area = 135

noncomputable def trapezoid_contains_smaller_triangles (contained : ℕ) : Prop :=
  contained = 4

theorem area_of_trapezoid (XYZ_area smaller_triangle_area : ℝ) 
    (num_smaller_triangles contained_smaller_triangles : ℕ) : 
    triangle_XYZ_is_isosceles → 
    identical_smaller_triangles smaller_triangle_area num_smaller_triangles →
    total_area_large_triangle XYZ_area →
    trapezoid_contains_smaller_triangles contained_smaller_triangles →
    (XYZ_area - contained_smaller_triangles * smaller_triangle_area) = 123 :=
by
  intros iso smaller_triangles total_area contained
  sorry

end area_of_trapezoid_l1897_189700


namespace distance_PQ_parallel_x_max_distance_PQ_l1897_189710

open Real

def parabola (x : ℝ) : ℝ := x^2

/--
1. When PQ is parallel to the x-axis, find the distance from point O to PQ.
-/
theorem distance_PQ_parallel_x (m : ℝ) (h₁ : m ≠ 0) (h₂ : parabola m = 1) : 
  ∃ d : ℝ, d = 1 := by
  sorry

/--
2. Find the maximum value of the distance from point O to PQ.
-/
theorem max_distance_PQ (a b : ℝ) (h₁ : a * b = -1) (h₂ : ∀ x, ∃ y, y = a * x + b) :
  ∃ d : ℝ, d = 1 := by
  sorry

end distance_PQ_parallel_x_max_distance_PQ_l1897_189710


namespace initial_dragon_fruits_remaining_kiwis_l1897_189745

variable (h d k : ℕ)    -- h: initial number of cantaloupes, d: initial number of dragon fruits, k: initial number of kiwis
variable (d_rem : ℕ)    -- d_rem: remaining number of dragon fruits after all cantaloupes are used up
variable (k_rem : ℕ)    -- k_rem: remaining number of kiwis after all cantaloupes are used up

axiom condition1 : d = 3 * h + 10
axiom condition2 : k = 2 * d
axiom condition3 : d_rem = 130
axiom condition4 : (d - d_rem) = 2 * h
axiom condition5 : k_rem = k - 10 * h

theorem initial_dragon_fruits (h : ℕ) (d : ℕ) (k : ℕ) (d_rem : ℕ) : 
  3 * h + 10 = d → 
  2 * d = k → 
  d_rem = 130 →
  2 * h + d_rem = d → 
  h = 120 → 
  d = 370 :=
by 
  intros
  sorry

theorem remaining_kiwis (h : ℕ) (d : ℕ) (k : ℕ) (k_rem : ℕ) : 
  3 * h + 10 = d → 
  2 * d = k → 
  h = 120 →
  k_rem = k - 10 * h → 
  k_rem = 140 :=
by 
  intros
  sorry

end initial_dragon_fruits_remaining_kiwis_l1897_189745


namespace slope_of_intersection_points_l1897_189792

theorem slope_of_intersection_points : 
  (∀ t : ℝ, ∃ x y : ℝ, (2 * x + 3 * y = 10 * t + 4) ∧ (x + 4 * y = 3 * t + 3)) → 
  (∀ t1 t2 : ℝ, t1 ≠ t2 → ((2 * ((10 * t1 + 4)  / 2) + 3 * ((-5/3 * t1 - 2/3)) = (10 * t1 + 4)) ∧ (2 * ((10 * t2 + 4) / 2) + 3 * ((-5/3 * t2 - 2/3)) = (10 * t2 + 4))) → 
  (31 * (((-5/3 * t1 - 2/3) - (-5/3 * t2 - 2/3)) / ((10 * t1 + 4) / 2 - (10 * t2 + 4) / 2)) = -4)) :=
sorry

end slope_of_intersection_points_l1897_189792


namespace smallest_positive_difference_l1897_189751

theorem smallest_positive_difference (a b : ℤ) (h : 17 * a + 6 * b = 13) : (∃ n : ℤ, n > 0 ∧ n = a - b) → n = 17 :=
by sorry

end smallest_positive_difference_l1897_189751


namespace second_expression_l1897_189729

variable (a b : ℕ)

theorem second_expression (h : 89 = ((2 * a + 16) + b) / 2) (ha : a = 34) : b = 94 :=
by
  sorry

end second_expression_l1897_189729


namespace ratio_of_black_to_white_tiles_l1897_189728

theorem ratio_of_black_to_white_tiles
  (original_width : ℕ)
  (original_height : ℕ)
  (original_black_tiles : ℕ)
  (original_white_tiles : ℕ)
  (border_width : ℕ)
  (border_height : ℕ)
  (extended_width : ℕ)
  (extended_height : ℕ)
  (new_white_tiles : ℕ)
  (total_white_tiles : ℕ)
  (total_black_tiles : ℕ)
  (ratio_black_to_white : ℚ)
  (h1 : original_width = 5)
  (h2 : original_height = 6)
  (h3 : original_black_tiles = 12)
  (h4 : original_white_tiles = 18)
  (h5 : border_width = 1)
  (h6 : border_height = 1)
  (h7 : extended_width = original_width + 2 * border_width)
  (h8 : extended_height = original_height + 2 * border_height)
  (h9 : new_white_tiles = (extended_width * extended_height) - (original_width * original_height))
  (h10 : total_white_tiles = original_white_tiles + new_white_tiles)
  (h11 : total_black_tiles = original_black_tiles)
  (h12 : ratio_black_to_white = total_black_tiles / total_white_tiles) :
  ratio_black_to_white = 3 / 11 := 
sorry

end ratio_of_black_to_white_tiles_l1897_189728


namespace rectangle_perimeter_is_36_l1897_189739

theorem rectangle_perimeter_is_36 (a b : ℕ) (h : a ≠ b) (h1 : a * b = 2 * (2 * a + 2 * b) - 8) : 2 * (a + b) = 36 :=
  sorry

end rectangle_perimeter_is_36_l1897_189739


namespace book_cost_l1897_189762

theorem book_cost (x : ℝ) 
  (h1 : Vasya_has = x - 150)
  (h2 : Tolya_has = x - 200)
  (h3 : (x - 150) + (x - 200) / 2 = x + 100) : x = 700 :=
sorry

end book_cost_l1897_189762


namespace number_of_teams_l1897_189798

theorem number_of_teams (x : ℕ) (h : x * (x - 1) = 90) : x = 10 :=
sorry

end number_of_teams_l1897_189798


namespace martinez_family_combined_height_l1897_189736

def chiquita_height := 5
def mr_martinez_height := chiquita_height + 2
def mrs_martinez_height := chiquita_height - 1
def son_height := chiquita_height + 3
def combined_height := chiquita_height + mr_martinez_height + mrs_martinez_height + son_height

theorem martinez_family_combined_height : combined_height = 24 :=
by
  sorry

end martinez_family_combined_height_l1897_189736


namespace lcm_18_24_30_l1897_189785

theorem lcm_18_24_30 :
  let a := 18
  let b := 24
  let c := 30
  let lcm := 360
  (∀ x > 0, x ∣ a ∧ x ∣ b ∧ x ∣ c → x ∣ lcm) ∧ (∀ y > 0, y ∣ lcm → y ∣ a ∧ y ∣ b ∧ y ∣ c) :=
by {
  let a := 18
  let b := 24
  let c := 30
  let lcm := 360
  sorry
}

end lcm_18_24_30_l1897_189785


namespace sum_even_integers_12_to_46_l1897_189782

theorem sum_even_integers_12_to_46 : 
  let a1 := 12
  let d := 2
  let an := 46
  let n := (an - a1) / d + 1
  let Sn := n * (a1 + an) / 2
  Sn = 522 := 
by
  let a1 := 12 
  let d := 2 
  let an := 46
  let n := (an - a1) / d + 1 
  let Sn := n * (a1 + an) / 2
  sorry

end sum_even_integers_12_to_46_l1897_189782
