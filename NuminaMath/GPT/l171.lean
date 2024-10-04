import Mathlib

namespace sin_sum_triangle_l171_171622

theorem sin_sum_triangle (α β γ : ℝ) (h : α + β + γ = Real.pi) : 
  Real.sin α + Real.sin β + Real.sin γ ≤ (3 * Real.sqrt 3) / 2 :=
sorry

end sin_sum_triangle_l171_171622


namespace smallest_n_inequality_l171_171455

theorem smallest_n_inequality :
  ∃ n : ℕ, (∀ x y z w : ℝ, (x^2 + y^2 + z^2 + w^2)^2 ≤ n * (x^4 + y^4 + z^4 + w^4)) ∧
  (∀ m : ℕ, (∀ x y z w : ℝ, (x^2 + y^2 + z^2 + w^2)^2 ≤ m * (x^4 + y^4 + z^4 + w^4)) → n ≤ m) :=
sorry

end smallest_n_inequality_l171_171455


namespace min_cans_needed_l171_171106

theorem min_cans_needed (C : ℕ → ℕ) (H : C 1 = 15) : ∃ n, C n * n >= 64 ∧ ∀ m, m < n → C 1 * m < 64 :=
by
  sorry

end min_cans_needed_l171_171106


namespace rainfall_second_week_l171_171560

theorem rainfall_second_week (x : ℝ) (h1 : x + 1.5 * x = 20) : 1.5 * x = 12 := 
by {
  sorry
}

end rainfall_second_week_l171_171560


namespace smallest_N_l171_171812

theorem smallest_N (N : ℕ) (hN : N > 70) (hdv : 70 ∣ 21 * N) : N = 80 :=
sorry

end smallest_N_l171_171812


namespace infinite_sequences_B_intersect_with_A_infinite_l171_171659

theorem infinite_sequences_B_intersect_with_A_infinite (A B : ℕ → ℕ) (d : ℕ) :
  (∀ n, A n = 5 * n - 2) ∧ (∀ k, B k = k * d + 7 - d) →
  (∃ d, ∀ m, ∃ k, A m = B k) :=
by
  sorry

end infinite_sequences_B_intersect_with_A_infinite_l171_171659


namespace CE_length_l171_171726

theorem CE_length (AF ED AE area : ℝ) (hAF : AF = 30) (hED : ED = 50) (hAE : AE = 120) (h_area : area = 7200) : 
  ∃ CE : ℝ, CE = 138 :=
by
  -- omitted proof steps
  sorry

end CE_length_l171_171726


namespace smallest_possible_N_l171_171795

theorem smallest_possible_N (N : ℕ) (h1 : N > 70) (h2 : 70 ∣ 21 * N) : N = 80 :=
sorry

end smallest_possible_N_l171_171795


namespace intersection_AB_l171_171206

variable {x : ℝ}

def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {x | x > 0}

theorem intersection_AB : A ∩ B = {x | 0 < x ∧ x < 2} :=
by sorry

end intersection_AB_l171_171206


namespace max_balls_in_cube_l171_171701

noncomputable def volume_of_cube : ℝ := (5 : ℝ)^3

noncomputable def volume_of_ball : ℝ := (4 / 3) * Real.pi * (1 : ℝ)^3

theorem max_balls_in_cube (c_length : ℝ) (b_radius : ℝ) (h1 : c_length = 5)
  (h2 : b_radius = 1) : 
  ⌊volume_of_cube / volume_of_ball⌋ = 29 := 
by
  sorry

end max_balls_in_cube_l171_171701


namespace picture_frame_length_l171_171508

theorem picture_frame_length (h : ℕ) (l : ℕ) (P : ℕ) (h_eq : h = 12) (P_eq : P = 44) (perimeter_eq : P = 2 * (l + h)) : l = 10 :=
by
  -- proof would go here
  sorry

end picture_frame_length_l171_171508


namespace first_year_after_2020_with_sum_15_l171_171251

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem first_year_after_2020_with_sum_15 :
  ∀ n, n > 2020 → (sum_of_digits n = 15 ↔ n = 2058) := by
  sorry

end first_year_after_2020_with_sum_15_l171_171251


namespace distance_traveled_on_foot_l171_171420

theorem distance_traveled_on_foot (x y : ℝ) (h1 : x + y = 80) (h2 : x / 8 + y / 16 = 7) : x = 32 :=
by
  sorry

end distance_traveled_on_foot_l171_171420


namespace birdhouse_volume_difference_l171_171042

-- Definitions to capture the given conditions
def sara_width_ft : ℝ := 1
def sara_height_ft : ℝ := 2
def sara_depth_ft : ℝ := 2

def jake_width_in : ℝ := 16
def jake_height_in : ℝ := 20
def jake_depth_in : ℝ := 18

-- Convert Sara's dimensions to inches
def ft_to_in (x : ℝ) : ℝ := x * 12
def sara_width_in := ft_to_in sara_width_ft
def sara_height_in := ft_to_in sara_height_ft
def sara_depth_in := ft_to_in sara_depth_ft

-- Volume calculations
def volume (width height depth : ℝ) := width * height * depth
def sara_volume := volume sara_width_in sara_height_in sara_depth_in
def jake_volume := volume jake_width_in jake_height_in jake_depth_in

-- The theorem to prove the difference in volume
theorem birdhouse_volume_difference : sara_volume - jake_volume = 1152 := by
  -- Proof goes here
  sorry

end birdhouse_volume_difference_l171_171042


namespace inflation_two_years_correct_real_rate_of_return_correct_l171_171703

-- Define the calculation for inflation over two years
def inflation_two_years (r : ℝ) : ℝ :=
  ((1 + r)^2 - 1) * 100

-- Define the calculation for the real rate of return
def real_rate_of_return (r : ℝ) (infl_rate : ℝ) : ℝ :=
  ((1 + r * r) / (1 + infl_rate / 100) - 1) * 100

-- Prove the inflation over two years is 3.0225%
theorem inflation_two_years_correct :
  inflation_two_years 0.015 = 3.0225 :=
by
  sorry

-- Prove the real yield of the bank deposit is 11.13%
theorem real_rate_of_return_correct :
  real_rate_of_return 0.07 3.0225 = 11.13 :=
by
  sorry

end inflation_two_years_correct_real_rate_of_return_correct_l171_171703


namespace circle_symmetry_l171_171359

theorem circle_symmetry (a b : ℝ) 
  (h1 : ∀ x y : ℝ, (x - a)^2 + (y - b)^2 = 1 ↔ (x - 1)^2 + (y - 3)^2 = 1) 
  (symm_line : ∀ x y : ℝ, y = x + 1) : a + b = 2 :=
sorry

end circle_symmetry_l171_171359


namespace probability_of_sum_14_l171_171375

-- Define the set of faces on a tetrahedral die
def faces : Set ℕ := {2, 4, 6, 8}

-- Define the event where the sum of two rolls equals 14
def event_sum_14 (a b : ℕ) : Prop := a + b = 14 ∧ a ∈ faces ∧ b ∈ faces

-- Define the total number of outcomes when rolling two dice
def total_outcomes : ℕ := 16

-- Define the number of successful outcomes for the event where the sum is 14
def successful_outcomes : ℕ := 2

-- The probability of rolling a sum of 14 with two such tetrahedral dice
def probability_sum_14 : ℚ := successful_outcomes / total_outcomes

-- The theorem we want to prove
theorem probability_of_sum_14 : probability_sum_14 = 1 / 8 := 
by sorry

end probability_of_sum_14_l171_171375


namespace juan_european_stamps_total_cost_l171_171412

/-- Define the cost of European stamps collection for Juan -/
def total_cost_juan_stamps : ℝ := 
  -- Costs of stamps from the 1980s
  (15 * 0.07) + (11 * 0.06) + (14 * 0.08) +
  -- Costs of stamps from the 1990s
  (14 * 0.07) + (10 * 0.06) + (12 * 0.08)

/-- Prove that the total cost for European stamps from the 80s and 90s is $5.37 -/
theorem juan_european_stamps_total_cost : total_cost_juan_stamps = 5.37 :=
  by sorry

end juan_european_stamps_total_cost_l171_171412


namespace sum_of_powers_pattern_l171_171604

theorem sum_of_powers_pattern :
  (5 + 7) * (5^2 + 7^2) * (5^4 + 7^4) * (5^8 + 7^8) * (5^16 + 7^16) = 5^32 + 7^32 :=
  sorry

end sum_of_powers_pattern_l171_171604


namespace units_digit_L_L_15_l171_171242

def Lucas (n : ℕ) : ℕ :=
match n with
| 0 => 2
| 1 => 1
| n + 2 => Lucas n + Lucas (n + 1)

theorem units_digit_L_L_15 : (Lucas (Lucas 15)) % 10 = 7 := by
  sorry

end units_digit_L_L_15_l171_171242


namespace both_not_divisible_by_7_l171_171396

theorem both_not_divisible_by_7 {a b : ℝ} (h : ¬ (∃ k : ℤ, ab = 7 * k)) : ¬ (∃ m : ℤ, a = 7 * m) ∧ ¬ (∃ n : ℤ, b = 7 * n) :=
sorry

end both_not_divisible_by_7_l171_171396


namespace balls_probability_l171_171916

theorem balls_probability :
  let total_ways := Nat.choose 24 4
  let ways_bw := Nat.choose 10 2 * Nat.choose 8 2
  let ways_br := Nat.choose 10 2 * Nat.choose 6 2
  let ways_wr := Nat.choose 8 2 * Nat.choose 6 2
  let target_ways := ways_bw + ways_br + ways_wr
  (target_ways : ℚ) / total_ways = 157 / 845 := by
  sorry

end balls_probability_l171_171916


namespace math_problem_l171_171634

variable {p q r x y : ℝ}

theorem math_problem (h1 : p / q = 6 / 7)
                     (h2 : p / r = 8 / 9)
                     (h3 : q / r = x / y) :
                     x = 28 ∧ y = 27 ∧ 2 * p + q = (19 / 6) * p := 
by 
  sorry

end math_problem_l171_171634


namespace arrangements_count_l171_171731

-- Definitions of students and grades
inductive Student : Type
| A | B | C | D | E | F
deriving DecidableEq

inductive Grade : Type
| first | second | third
deriving DecidableEq

-- A function to count valid arrangements
def valid_arrangements (assignments : Student → Grade) : Bool :=
  assignments Student.A = Grade.first ∧
  assignments Student.B ≠ Grade.third ∧
  assignments Student.C ≠ Grade.third ∧
  (assignments Student.A = Grade.first) ∧
  ((assignments Student.B = Grade.second ∧ assignments Student.C = Grade.second ∧ 
    (assignments Student.D ≠ Grade.first ∨ assignments Student.E ≠ Grade.first ∨ assignments Student.F ≠ Grade.first)) ∨
   ((assignments Student.B ≠ Grade.second ∨ assignments Student.C ≠ Grade.second) ∧ 
    (assignments Student.B ≠ Grade.first ∨ assignments Student.C ≠ Grade.first)))

theorem arrangements_count : 
  ∃ (count : ℕ), count = 9 ∧
  count = (Nat.card { assign : Student → Grade // valid_arrangements assign } : ℕ) := sorry

end arrangements_count_l171_171731


namespace triangle_altitude_sum_l171_171448

-- Problem Conditions
def line_eq (x y : ℝ) : Prop := 10 * x + 8 * y = 80

-- Altitudes Length Sum
theorem triangle_altitude_sum :
  ∀ x y : ℝ, line_eq x y → 
  ∀ (a b c: ℝ), a = 8 → b = 10 → c = 40 / Real.sqrt 41 →
  a + b + c = (18 * Real.sqrt 41 + 40) / Real.sqrt 41 :=
by
  sorry

end triangle_altitude_sum_l171_171448


namespace mutually_exclusive_not_complementary_l171_171643

open Finset

-- Definitions for the number of white and black balls in a bag.
def white_balls : ℕ := 3
def black_balls : ℕ := 4
def total_balls : ℕ := white_balls + black_balls

-- Definitions of events ①, ②, ③, and ④.
def event1 (drawn : Finset ℕ) : Prop :=
  (drawn.card = 1 ∧ (drawn ⊆ range white_balls) ∧ (drawn = range white_balls))

def event2 (drawn : Finset ℕ) : Prop :=
  (∃ ball, ball ∈ drawn ∧ ball < white_balls) ∧ ∀ ball, ball ∈ drawn → ball ≥ white_balls

def event3 (drawn : Finset ℕ) : Prop :=
  (∃ ball, ball ∈ drawn ∧ ball < white_balls) ∧ drawn.card ≥ 2

def event4 (drawn : Finset ℕ) : Prop :=
  (∃ ball, ball ∈ drawn ∧ ball < white_balls) ∧ (∃ ball, ball ∈ drawn ∧ ball ≥ white_balls)

-- The question to prove.
theorem mutually_exclusive_not_complementary :
  (∀ e1 e2, event1 e1 → event1 e2 → e1 = e2 ∨ disjoint e1 e2) ∧
  ¬(∀ e, (event2 e ∨ event3 e ∨ event4 e) → event1 e) :=
sorry

end mutually_exclusive_not_complementary_l171_171643


namespace carpet_dimensions_l171_171590
open Real

theorem carpet_dimensions (x y : ℝ) 
  (h1 : x > 0)
  (h2 : y > 0)
  (h3 : ∃ k: ℝ, y = k * x)
  (h4 : ∃ α β: ℝ, α + k * β = 50 ∧ k * α + β = 55)
  (h5 : ∃ γ δ: ℝ, γ + k * δ = 38 ∧ k * γ + δ = 55) :
  x = 25 ∧ y = 50 :=
by sorry

end carpet_dimensions_l171_171590


namespace total_area_of_storage_units_l171_171679

theorem total_area_of_storage_units (total_units remaining_units : ℕ) 
    (size_8_by_4 length width unit_area_200 : ℕ)
    (h1 : total_units = 42)
    (h2 : remaining_units = 22)
    (h3 : length = 8)
    (h4 : width = 4)
    (h5 : unit_area_200 = 200) 
    (h6 : ∀ i : ℕ, i < 20 → unit_area_8_by_4 = length * width) 
    (h7 : ∀ j : ℕ, j < 22 → unit_area_200 = 200) :
    total_area_of_all_units = 5040 :=
by
  let unit_area_8_by_4 := length * width
  let total_area_20_units := 20 * unit_area_8_by_4
  let total_area_22_units := 22 * unit_area_200
  let total_area_of_all_units := total_area_20_units + total_area_22_units
  sorry

end total_area_of_storage_units_l171_171679


namespace simplify_expression_l171_171517

theorem simplify_expression : 1 - (1 / (1 + Real.sqrt 5)) + (1 / (1 - Real.sqrt 5)) = 1 := 
by sorry

end simplify_expression_l171_171517


namespace ratio_boys_to_girls_l171_171207

variables (B G : ℤ)

def boys_count : ℤ := 50
def girls_count (B : ℤ) : ℤ := B + 80

theorem ratio_boys_to_girls : 
  (B = boys_count) → 
  (G = girls_count B) → 
  ((B : ℚ) / (G : ℚ) = 5 / 13) :=
by
  sorry

end ratio_boys_to_girls_l171_171207


namespace sqrt_10_bounds_l171_171256

theorem sqrt_10_bounds : 10 > 9 ∧ 10 < 16 → 3 < Real.sqrt 10 ∧ Real.sqrt 10 < 4 := 
by 
  sorry

end sqrt_10_bounds_l171_171256


namespace solve_equation_l171_171612

theorem solve_equation (x : ℝ) : x^4 + (4 - x)^4 = 272 ↔ x = 2 + Real.sqrt 6 ∨ x = 2 - Real.sqrt 6 := 
by sorry

end solve_equation_l171_171612


namespace smallest_possible_value_of_N_l171_171808

theorem smallest_possible_value_of_N :
  ∃ N : ℕ, N > 70 ∧ (21 * N) % 70 = 0 ∧ N = 80 :=
by
  existsi 80
  split
  · norm_num
  split
  · norm_num
  sorry

end smallest_possible_value_of_N_l171_171808


namespace area_of_rectangle_l171_171487

variables {group_interval rate : ℝ}

theorem area_of_rectangle (length_of_small_rectangle : ℝ) (height_of_small_rectangle : ℝ) :
  (length_of_small_rectangle = group_interval) → (height_of_small_rectangle = rate / group_interval) →
  length_of_small_rectangle * height_of_small_rectangle = rate :=
by
  intros h_length h_height
  rw [h_length, h_height]
  exact mul_div_cancel' rate (by sorry)

end area_of_rectangle_l171_171487


namespace possible_values_x_l171_171275

-- Define the conditions
def gold_coin_worth (x y : ℕ) (g s : ℝ) : Prop :=
  g = (1 + x / 100.0) * s ∧ s = (1 - y / 100.0) * g

-- Define the main theorem statement
theorem possible_values_x : ∀ (x y : ℕ) (g s : ℝ), gold_coin_worth x y g s → 
  (∃ (n : ℕ), n = 12) :=
by
  -- Definitions based on given conditions
  intro x y g s h
  obtain ⟨hx, hy⟩ := h

  -- Placeholder for proof; skip with sorry
  sorry

end possible_values_x_l171_171275


namespace simplify_and_evaluate_expression_l171_171241

theorem simplify_and_evaluate_expression (x : ℝ) (hx : x = 4) :
  (1 / (x + 2) + 1) / ((x^2 + 6 * x + 9) / (x^2 - 4)) = 2 / 7 :=
by
  sorry

end simplify_and_evaluate_expression_l171_171241


namespace line_relation_in_perpendicular_planes_l171_171182

-- Let's define the notions of planes and lines being perpendicular/parallel
variables {α β : Plane} {a : Line}

def plane_perpendicular (α β : Plane) : Prop := sorry -- definition of perpendicular planes
def line_perpendicular_plane (a : Line) (β : Plane) : Prop := sorry -- definition of a line being perpendicular to a plane
def line_parallel_plane (a : Line) (α : Plane) : Prop := sorry -- definition of a line being parallel to a plane
def line_in_plane (a : Line) (α : Plane) : Prop := sorry -- definition of a line lying in a plane

-- The theorem stating the relationship given the conditions
theorem line_relation_in_perpendicular_planes 
  (h1 : plane_perpendicular α β) 
  (h2 : line_perpendicular_plane a β) : 
  line_parallel_plane a α ∨ line_in_plane a α :=
sorry

end line_relation_in_perpendicular_planes_l171_171182


namespace sum_of_sines_leq_3_sqrt3_over_2_l171_171621

theorem sum_of_sines_leq_3_sqrt3_over_2 (α β γ : ℝ) (h : α + β + γ = Real.pi) :
  Real.sin α + Real.sin β + Real.sin γ ≤ 3 * Real.sqrt 3 / 2 :=
sorry

end sum_of_sines_leq_3_sqrt3_over_2_l171_171621


namespace difference_highest_lowest_score_l171_171098

-- Definitions based on conditions
def total_innings : ℕ := 46
def avg_innings : ℕ := 61
def highest_score : ℕ := 202
def avg_excl_highest_lowest : ℕ := 58
def innings_excl_highest_lowest : ℕ := 44

-- Calculated total runs
def total_runs : ℕ := total_innings * avg_innings
def total_runs_excl_highest_lowest : ℕ := innings_excl_highest_lowest * avg_excl_highest_lowest
def sum_of_highest_lowest : ℕ := total_runs - total_runs_excl_highest_lowest
def lowest_score : ℕ := sum_of_highest_lowest - highest_score

theorem difference_highest_lowest_score 
  (h1: total_runs = total_innings * avg_innings)
  (h2: avg_excl_highest_lowest * innings_excl_highest_lowest = total_runs_excl_highest_lowest)
  (h3: sum_of_highest_lowest = total_runs - total_runs_excl_highest_lowest)
  (h4: highest_score = 202)
  (h5: lowest_score = sum_of_highest_lowest - highest_score)
  : highest_score - lowest_score = 150 :=
by
  -- We only need to state the theorem, so we can skip the proof.
  -- The exact statements of conditions and calculations imply the result.
  sorry

end difference_highest_lowest_score_l171_171098


namespace range_of_m_for_inequality_l171_171201

-- Define the condition
def condition (x : ℝ) := x ∈ Set.Iic (-1)

-- Define the inequality for proving the range of m
def inequality_holds (m x : ℝ) : Prop := (m - m^2) * 4^x + 2^x + 1 > 0

-- Prove the range of m for the given conditions such that the inequality holds
theorem range_of_m_for_inequality :
  (∀ (x : ℝ), condition x → inequality_holds m x) ↔ (-2 < m ∧ m < 3) :=
sorry

end range_of_m_for_inequality_l171_171201


namespace sufficient_but_not_necessary_not_necessary_l171_171914

theorem sufficient_but_not_necessary (a : ℝ) (h : a > 0) : (|a| > 0) := by
  sorry

theorem not_necessary (a : ℝ) : |a| > 0 → ¬(a = 0) ∧ (a ≠ 0 → |a| > 0 ∧ (¬(a > 0) → (|a| > 0))) := by
  sorry

end sufficient_but_not_necessary_not_necessary_l171_171914


namespace percentage_games_won_l171_171453

theorem percentage_games_won 
  (P_first : ℝ)
  (P_remaining : ℝ)
  (total_games : ℕ)
  (H1 : P_first = 0.7)
  (H2 : P_remaining = 0.5)
  (H3 : total_games = 100) :
  True :=
by
  -- To prove the percentage of games won is 70%
  have percentage_won : ℝ := P_first
  have : percentage_won * 100 = 70 := by sorry
  trivial

end percentage_games_won_l171_171453


namespace cj_more_stamps_than_twice_kj_l171_171735

variable (C K A : ℕ) (x : ℕ)

theorem cj_more_stamps_than_twice_kj :
  (C = 2 * K + x) →
  (K = A / 2) →
  (C + K + A = 930) →
  (A = 370) →
  (x = 25) →
  (C - 2 * K = 5) :=
by
  intros h1 h2 h3 h4 h5
  sorry

end cj_more_stamps_than_twice_kj_l171_171735


namespace percent_correct_both_l171_171091

-- Definitions based on given conditions in the problem
def P_A : ℝ := 0.63
def P_B : ℝ := 0.50
def P_not_A_and_not_B : ℝ := 0.20

-- Definition of the desired result using the inclusion-exclusion principle based on the given conditions
def P_A_and_B : ℝ := P_A + P_B - (1 - P_not_A_and_not_B)

-- Theorem stating our goal: proving the probability of both answering correctly is 0.33
theorem percent_correct_both : P_A_and_B = 0.33 := by
  sorry

end percent_correct_both_l171_171091


namespace find_m_l171_171059

theorem find_m (x₁ x₂ y₁ y₂ : ℝ) (m : ℝ) 
  (h_parabola_A : y₁ = 2 * x₁^2) 
  (h_parabola_B : y₂ = 2 * x₂^2) 
  (h_symmetry : y₂ - y₁ = 2 * (x₂^2 - x₁^2)) 
  (h_product : x₁ * x₂ = -1/2) 
  (h_midpoint : (y₂ + y₁) / 2 = (x₂ + x₁) / 2 + m) :
  m = 3 / 2 :=
by
  sorry

end find_m_l171_171059


namespace inflation_over_two_years_real_yield_deposit_second_year_l171_171706

-- Inflation problem setup and proof
theorem inflation_over_two_years :
  ((1 + 0.015) ^ 2 - 1) * 100 = 3.0225 :=
by sorry

-- Real yield problem setup and proof
theorem real_yield_deposit_second_year :
  ((1.07 * 1.07) / (1 + 0.030225) - 1) * 100 = 11.13 :=
by sorry

end inflation_over_two_years_real_yield_deposit_second_year_l171_171706


namespace plane_distance_last_10_seconds_l171_171243

theorem plane_distance_last_10_seconds (s : ℝ → ℝ) (h : ∀ t, s t = 60 * t - 1.5 * t^2) : 
  s 20 - s 10 = 150 := 
by 
  sorry

end plane_distance_last_10_seconds_l171_171243


namespace range_of_expression_l171_171026

theorem range_of_expression (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 ≤ β ∧ β ≤ π / 2) :
  -π / 6 < 2 * α - β / 3 ∧ 2 * α - β / 3 < π :=
sorry

end range_of_expression_l171_171026


namespace additional_pairs_of_snakes_l171_171507

theorem additional_pairs_of_snakes (total_snakes breeding_balls snakes_per_ball additional_snakes_per_pair : ℕ)
  (h1 : total_snakes = 36) 
  (h2 : breeding_balls = 3)
  (h3 : snakes_per_ball = 8) 
  (h4 : additional_snakes_per_pair = 2) :
  (total_snakes - (breeding_balls * snakes_per_ball)) / additional_snakes_per_pair = 6 :=
by
  sorry

end additional_pairs_of_snakes_l171_171507


namespace train_cross_time_l171_171427

def length_of_train : Float := 135.0 -- in meters
def speed_of_train_kmh : Float := 45.0 -- in kilometers per hour
def length_of_bridge : Float := 240.03 -- in meters

def speed_of_train_ms : Float := speed_of_train_kmh * 1000.0 / 3600.0

def total_distance : Float := length_of_train + length_of_bridge

def time_to_cross : Float := total_distance / speed_of_train_ms

theorem train_cross_time : time_to_cross = 30.0024 :=
by
  sorry

end train_cross_time_l171_171427


namespace sin_90_eq_one_l171_171152

-- Definition of the rotation by 90 degrees counterclockwise
def rotate90 (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, p.1)

-- Definition of the sine function for a 90 degree angle
def sin90 : ℝ :=
  let initial_point := (1, 0)
  let rotated_point := rotate90 initial_point
  rotated_point.2

-- Theorem to be proven: sin90 should be equal to 1
theorem sin_90_eq_one : sin90 = 1 :=
by
  sorry

end sin_90_eq_one_l171_171152


namespace max_gcd_13n_plus_4_8n_plus_3_l171_171599

theorem max_gcd_13n_plus_4_8n_plus_3 : ∃ n : ℕ, n > 0 ∧ Int.gcd (13 * n + 4) (8 * n + 3) = 11 := 
sorry

end max_gcd_13n_plus_4_8n_plus_3_l171_171599


namespace smallest_solution_l171_171070

theorem smallest_solution : ∃ x : ℝ, x^4 - 50 * x^2 + 625 = 0 ∧ ∀ y : ℝ, y^4 - 50 * y^2 + 625 = 0 → x ≤ y :=
by {
  use -5,
  split,
  sorry, -- here would be the proof that (-5)^4 - 50 * (-5)^2 + 625 = 0
  intros y hy,
  sorry -- here would be the proof that for any y such that y^4 - 50 * y^2 + 625 = 0, -5 ≤ y
}

end smallest_solution_l171_171070


namespace Gary_final_amount_l171_171766

theorem Gary_final_amount
(initial_amount dollars_snake dollars_hamster dollars_supplies : ℝ)
(h1 : initial_amount = 73.25)
(h2 : dollars_snake = 55.50)
(h3 : dollars_hamster = 25.75)
(h4 : dollars_supplies = 12.40) :
  initial_amount + dollars_snake - dollars_hamster - dollars_supplies = 90.60 :=
by
  sorry

end Gary_final_amount_l171_171766


namespace sin_90_eq_1_l171_171136

theorem sin_90_eq_1 :
  let θ := 90 : ℝ in
  let cos_θ := real.cos θ in
  let sin_θ := real.sin θ in 
  let rotation_matrix := ![![cos_θ, -sin_θ], ![sin_θ, cos_θ]] in
  let point := ![1, 0] in
  let rotated_point := matrix.mul_vec rotation_matrix point in
  rotated_point = ![0, 1] → 
  sin_θ = 1 :=
by
  sorry

end sin_90_eq_1_l171_171136


namespace max_pasture_area_l171_171728

/-- A rectangular sheep pasture is enclosed on three sides by a fence, while the fourth side uses the 
side of a barn that is 500 feet long. The fence costs $10 per foot, and the total budget for the 
fence is $2000. Determine the length of the side parallel to the barn that will maximize the pasture area. -/
theorem max_pasture_area (length_barn : ℝ) (cost_per_foot : ℝ) (budget : ℝ) :
  length_barn = 500 ∧ cost_per_foot = 10 ∧ budget = 2000 → 
  ∃ x : ℝ, x = 100 ∧ (∀ y : ℝ, y ≥ 0 → 
    (budget / cost_per_foot) ≥ 2*y + x → 
    (y * x ≤ y * 100)) :=
by
  sorry

end max_pasture_area_l171_171728


namespace temperature_representation_l171_171741

theorem temperature_representation (a : ℤ) (b : ℤ) (h1 : a = 8) (h2 : b = -5) :
    b < 0 → b = -5 :=
by
  sorry

end temperature_representation_l171_171741


namespace area_of_inscribed_octagon_l171_171932

open Real

def regular_octagon_area {r : ℝ} (octagon : Prop) : ℝ :=
  if octagon then 8 * (1 / 2 * r^2 * sin (π / 4)) else 0

theorem area_of_inscribed_octagon (r : ℝ) (h1 : r = 3) (h2 : ∀ octagon, octagon → regular_octagon_area octagon = 18 * sqrt 2) :
  regular_octagon_area true = 18 * sqrt 2 :=
by
  rw [← h2 true]
  sorry

end area_of_inscribed_octagon_l171_171932


namespace trapezoid_circumscribed_l171_171244

noncomputable def is_circumscribed (t: trapezoid) := sorry -- Definition needed

variables (α β : Real)
variables (ABCD : trapezoid)
variables (AD BC : Real)

-- Angles at the base AD
def angles_at_base (t: trapezoid) (α β : Real) : Prop :=
  -- The assertion here is symbolic, real proof would need concrete definitions
  sorry

-- The main theorem to prove
theorem trapezoid_circumscribed (ABCD : trapezoid) (α β : Real) (AD BC : ℝ)
  (h_angles : angles_at_base ABCD α β) :
  (is_circumscribed ABCD) ↔ (BC / AD = tan(α) * tan(β)) :=
sorry

end trapezoid_circumscribed_l171_171244


namespace factorial_trailing_zeros_base_8_l171_171996

/-- Number of trailing zeros of 15! in base 8 is 3 -/
theorem factorial_trailing_zeros_base_8 : number_of_trailing_zeros_in_base 15! 8 = 3 := sorry

end factorial_trailing_zeros_base_8_l171_171996


namespace alice_speed_exceeds_l171_171051

theorem alice_speed_exceeds (distance : ℕ) (v_bob : ℕ) (time_diff : ℕ) (v_alice : ℕ)
  (h_distance : distance = 220)
  (h_v_bob : v_bob = 40)
  (h_time_diff : time_diff = 1/2) : 
  v_alice > 44 := 
sorry

end alice_speed_exceeds_l171_171051


namespace milk_needed_for_one_batch_l171_171247

-- Define cost of one batch given amount of milk M
def cost_of_one_batch (M : ℝ) : ℝ := 1.5 * M + 6

-- Define cost of three batches
def cost_of_three_batches (M : ℝ) : ℝ := 3 * cost_of_one_batch M

theorem milk_needed_for_one_batch : ∃ M : ℝ, cost_of_three_batches M = 63 ∧ M = 10 :=
by
  sorry

end milk_needed_for_one_batch_l171_171247


namespace total_dining_bill_before_tip_l171_171417

-- Define total number of people
def numberOfPeople : ℕ := 6

-- Define the individual payment
def individualShare : ℝ := 25.48

-- Define the total payment
def totalPayment : ℝ := numberOfPeople * individualShare

-- Define the tip percentage
def tipPercentage : ℝ := 0.10

-- Total payment including tip expressed in terms of the original bill B
def totalPaymentWithTip (B : ℝ) : ℝ := B + B * tipPercentage

-- Prove the total dining bill before the tip
theorem total_dining_bill_before_tip : 
    ∃ B : ℝ, totalPayment = totalPaymentWithTip B ∧ B = 139.89 :=
by
    sorry

end total_dining_bill_before_tip_l171_171417


namespace password_problem_l171_171844

theorem password_problem (n : ℕ) :
  (n^4 - n * (n - 1) * (n - 2) * (n - 3) = 936) → n = 6 :=
by
  sorry

end password_problem_l171_171844


namespace total_weight_is_1kg_total_weight_in_kg_eq_1_l171_171577

theorem total_weight_is_1kg 
  (weight_msg : ℕ := 80)
  (weight_salt : ℕ := 500)
  (weight_detergent : ℕ := 420) :
  (weight_msg + weight_salt + weight_detergent) = 1000 := by
sorry

theorem total_weight_in_kg_eq_1 
  (total_weight_g : ℕ := weight_msg + weight_salt + weight_detergent) :
  (total_weight_g = 1000) → (total_weight_g / 1000 = 1) := by
sorry

end total_weight_is_1kg_total_weight_in_kg_eq_1_l171_171577


namespace find_number_l171_171533

theorem find_number (x : ℕ) (h : 5 + x = 20) : x = 15 :=
sorry

end find_number_l171_171533


namespace ratio_of_length_to_width_l171_171528

variable (P W L : ℕ)
variable (ratio : ℕ × ℕ)

theorem ratio_of_length_to_width (h1 : P = 336) (h2 : W = 70) (h3 : 2 * L + 2 * W = P) : ratio = (7, 5) :=
by
  sorry

end ratio_of_length_to_width_l171_171528


namespace max_value_k_l171_171963

theorem max_value_k (x y k : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < k)
(h4 : 4 = k^2 * (x^2 / y^2 + 2 + y^2 / x^2) + k^3 * (x / y + y / x)) : 
k ≤ 4 * (Real.sqrt 2) - 4 :=
by sorry

end max_value_k_l171_171963


namespace Lisa_earns_15_more_than_Tommy_l171_171857

variables (total_earnings Lisa_earnings Tommy_earnings : ℝ)

-- Conditions
def condition1 := total_earnings = 60
def condition2 := Lisa_earnings = total_earnings / 2
def condition3 := Tommy_earnings = Lisa_earnings / 2

-- Theorem to prove
theorem Lisa_earns_15_more_than_Tommy (h1: condition1) (h2: condition2) (h3: condition3) : 
  Lisa_earnings - Tommy_earnings = 15 :=
sorry

end Lisa_earns_15_more_than_Tommy_l171_171857


namespace find_highway_speed_l171_171572

def car_local_distance := 40
def car_local_speed := 20
def car_highway_distance := 180
def average_speed := 44
def speed_of_car_on_highway := 60

theorem find_highway_speed :
  car_local_distance / car_local_speed + car_highway_distance / speed_of_car_on_highway = (car_local_distance + car_highway_distance) / average_speed :=
by
  sorry

end find_highway_speed_l171_171572


namespace fraction_pow_zero_l171_171699

theorem fraction_pow_zero :
  (4310000 / -21550000 : ℝ) ≠ 0 →
  (4310000 / -21550000 : ℝ) ^ 0 = 1 :=
by
  intro h
  sorry

end fraction_pow_zero_l171_171699


namespace bonnets_per_orphanage_l171_171034

/--
Mrs. Young makes bonnets for kids in the orphanage.
On Monday, she made 10 bonnets.
On Tuesday and Wednesday combined she made twice more than on Monday.
On Thursday she made 5 more than on Monday.
On Friday she made 5 less than on Thursday.
She divided up the bonnets evenly and sent them to 5 orphanages.
Prove that the number of bonnets Mrs. Young sent to each orphanage is 11.
-/
theorem bonnets_per_orphanage :
  let monday := 10
  let tuesday_wednesday := 2 * monday
  let thursday := monday + 5
  let friday := thursday - 5
  let total_bonnets := monday + tuesday_wednesday + thursday + friday
  let orphanages := 5
  total_bonnets / orphanages = 11 :=
by
  sorry

end bonnets_per_orphanage_l171_171034


namespace triangle_inequality_part_a_triangle_inequality_part_b_l171_171090

variable {a b c S : ℝ}

/-- Part (a): Prove that for any triangle ABC, the inequality a^2 + b^2 + c^2 ≥ 4 √3 S holds
    where equality holds if and only if ABC is an equilateral triangle. -/
theorem triangle_inequality_part_a (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (area_S : S > 0) :
  a^2 + b^2 + c^2 ≥ 4 * Real.sqrt 3 * S :=
sorry

/-- Part (b): Prove that for any triangle ABC,
    the inequality a^2 + b^2 + c^2 - (a - b)^2 - (b - c)^2 - (c - a)^2 ≥ 4 √3 S
    holds where equality also holds if and only if a = b = c. -/
theorem triangle_inequality_part_b (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (area_S : S > 0) :
  a^2 + b^2 + c^2 - (a - b)^2 - (b - c)^2 - (c - a)^2 ≥ 4 * Real.sqrt 3 * S :=
sorry

end triangle_inequality_part_a_triangle_inequality_part_b_l171_171090


namespace national_park_sightings_l171_171010

def january_sightings : ℕ := 26

def february_sightings : ℕ := 3 * january_sightings

def march_sightings : ℕ := february_sightings / 2

def total_sightings : ℕ := january_sightings + february_sightings + march_sightings

theorem national_park_sightings : total_sightings = 143 := by
  sorry

end national_park_sightings_l171_171010


namespace actual_price_of_food_l171_171089

theorem actual_price_of_food (P : ℝ) (h : 1.32 * P = 132) : P = 100 := 
by
  sorry

end actual_price_of_food_l171_171089


namespace smallest_possible_value_of_N_l171_171825

-- Declares the context and required constraints
theorem smallest_possible_value_of_N (N : ℕ) (h1 : 70 < N) (h2 : 70 ∣ 21 * N) : N = 80 :=
by
  sorry

end smallest_possible_value_of_N_l171_171825


namespace smallest_possible_N_l171_171793

theorem smallest_possible_N (N : ℕ) (h1 : N > 70) (h2 : 70 ∣ 21 * N) : N = 80 :=
sorry

end smallest_possible_N_l171_171793


namespace total_games_played_l171_171720

theorem total_games_played (n : ℕ) (h : n = 7) : (n.choose 2) = 21 := by
  sorry

end total_games_played_l171_171720


namespace smallest_possible_value_of_N_l171_171810

theorem smallest_possible_value_of_N :
  ∃ N : ℕ, N > 70 ∧ (21 * N) % 70 = 0 ∧ N = 80 :=
by
  existsi 80
  split
  · norm_num
  split
  · norm_num
  sorry

end smallest_possible_value_of_N_l171_171810


namespace factor_polynomial_l171_171978

theorem factor_polynomial (a b c : ℝ) : 
  a^3 * (b^2 - c^2) + b^3 * (c^2 - b^2) + c^3 * (a^2 - b^2) = (a - b) * (b - c) * (c - a) * (a * b + a * c + b * c) :=
by 
  sorry

end factor_polynomial_l171_171978


namespace multiply_exp_result_l171_171400

theorem multiply_exp_result : 121 * (5 ^ 4) = 75625 :=
by
  sorry

end multiply_exp_result_l171_171400


namespace min_convergence_to_exponential_l171_171410

open ProbabilityTheory

variable {α : Type*} {μ : Measure α}

/-- Given a sequence of positive i.i.d. random variables ξ₁, ξ₂, ... with a distribution function F.
    If F(x) = λx + o(x) as x → 0 for some λ > 0, then n ξₘᵢₙ converges in distribution to η, where η
    is an exponentially distributed random variable with parameter λ. -/
theorem min_convergence_to_exponential
  (ξ : ℕ → α → ℝ)
  (F : ℝ → ℝ)
  (h_indep : IndepFun μ ξ)
  (h_iid : ∀ k, IdentDistrib (ξ 0) (ξ k))
  (h_F : ∀ x, F x = λ * x + asymptotics.small_o x)
  (λ_pos : λ > 0) :
  tendsto_in_distrib (λ n, n * inf (λ i, ξ i) n) (Exponential μ λ) :=
sorry

end min_convergence_to_exponential_l171_171410


namespace machine_x_widgets_per_hour_l171_171227

-- Definitions of the variables and conditions
variable (Wx Wy Tx Ty: ℝ)
variable (h1: Tx = Ty + 60)
variable (h2: Wy = 1.20 * Wx)
variable (h3: Wx * Tx = 1080)
variable (h4: Wy * Ty = 1080)

-- Statement of the problem to prove
theorem machine_x_widgets_per_hour : Wx = 3 := by
  sorry

end machine_x_widgets_per_hour_l171_171227


namespace tanya_dan_error_l171_171240

theorem tanya_dan_error 
  (a b c d e f g : ℤ)
  (h₁ : a < b) (h₂ : b < c) (h₃ : c < d) (h₄ : d < e) (h₅ : e < f) (h₆ : f < g)
  (h₇ : a % 2 = 1) (h₈ : b % 2 = 1) (h₉ : c % 2 = 1) (h₁₀ : d % 2 = 1) 
  (h₁₁ : e % 2 = 1) (h₁₂ : f % 2 = 1) (h₁₃ : g % 2 = 1)
  (h₁₄ : (a + b + c + d + e + f + g) / 7 - d = 3 / 7) :
  false :=
by sorry

end tanya_dan_error_l171_171240


namespace fourth_number_on_board_eighth_number_on_board_l171_171263

theorem fourth_number_on_board (medians : List ℚ) (hmed : medians = [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5]) :
  ∃ (numbers : List ℚ), numbers.length ≥ 4 ∧ median numbers[3] = 2 :=
sorry

theorem eighth_number_on_board (medians : List ℚ) (hmed : medians = [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5]) :
  ∃ (numbers : List ℚ), numbers.length ≥ 8 ∧ median numbers[7] = 2 :=
sorry

end fourth_number_on_board_eighth_number_on_board_l171_171263


namespace find_B_l171_171792

variables {a b c A B C : ℝ}

-- Conditions
axiom given_condition_1 : (c - b) / (c - a) = (Real.sin A) / (Real.sin C + Real.sin B)

-- Law of Sines
axiom law_of_sines_1 : (c - b) / (c - a) = a / (c + b)

-- Law of Cosines
axiom law_of_cosines_1 : Real.cos B = (a^2 + c^2 - b^2) / (2 * a * c)

-- Target
theorem find_B : B = Real.pi / 3 := 
sorry

end find_B_l171_171792


namespace speed_difference_between_lucy_and_sam_l171_171586

noncomputable def average_speed (distance : ℚ) (time_minutes : ℚ) : ℚ :=
  distance / (time_minutes / 60)

theorem speed_difference_between_lucy_and_sam :
  let distance := 6
  let lucy_time := 15
  let sam_time := 45
  let lucy_speed := average_speed distance lucy_time
  let sam_speed := average_speed distance sam_time
  (lucy_speed - sam_speed) = 16 :=
by
  sorry

end speed_difference_between_lucy_and_sam_l171_171586


namespace sin_ninety_deg_l171_171142

theorem sin_ninety_deg : Real.sin (Float.pi / 2) = 1 := 
by sorry

end sin_ninety_deg_l171_171142


namespace inequality_solution_l171_171957

theorem inequality_solution (x : ℝ) :
  (1 / (x^2 + 1) > 4 / x + 19 / 10) ↔ x ∈ Set.Ioo (-2 : ℝ) (0 : ℝ) :=
sorry

end inequality_solution_l171_171957


namespace right_triangle_hypotenuse_l171_171647

theorem right_triangle_hypotenuse (a b : ℕ) (a_val : a = 4) (b_val : b = 5) :
    ∃ c : ℝ, c^2 = (a:ℝ)^2 + (b:ℝ)^2 ∧ c = Real.sqrt 41 :=
by
  sorry

end right_triangle_hypotenuse_l171_171647


namespace number_of_distinguishable_large_triangles_l171_171100

theorem number_of_distinguishable_large_triangles (colors : Fin 8) :
  ∃(large_triangles : Fin 960), true :=
by
  sorry

end number_of_distinguishable_large_triangles_l171_171100


namespace seven_whole_numbers_cannot_form_ten_consecutive_pairwise_sums_l171_171969

theorem seven_whole_numbers_cannot_form_ten_consecutive_pairwise_sums
  (a1 a2 a3 a4 a5 a6 a7 : Nat) :
  ¬ ∃ (s : Finset Nat), (s = {a1 + a2, a1 + a3, a1 + a4, a1 + a5, a1 + a6, a1 + a7,
                             a2 + a3, a2 + a4, a2 + a5, a2 + a6, a2 + a7,
                             a3 + a4, a3 + a5, a3 + a6, a3 + a7,
                             a4 + a5, a4 + a6, a4 + a7,
                             a5 + a6, a5 + a7,
                             a6 + a7}) ∧
  (∃ (n : Nat), s = {n, n+1, n+2, n+3, n+4, n+5, n+6, n+7, n+8, n+9}) := 
sorry

end seven_whole_numbers_cannot_form_ten_consecutive_pairwise_sums_l171_171969


namespace truncated_cone_radius_l171_171832

theorem truncated_cone_radius (R: ℝ) (l: ℝ) (h: 0 < l)
  (h1 : ∃ (r: ℝ), r = (R + 5) / 2 ∧ (5 + r) = (1 / 2) * (R + r))
  : R = 25 :=
sorry

end truncated_cone_radius_l171_171832


namespace range_of_f_l171_171979

noncomputable def f (x : ℝ) : ℝ := sin x / cos (x + π / 6)

theorem range_of_f :
  set.range (λ x, f x) = set.Icc ((sqrt 3 - 1) / 2) 1 :=
  sorry

end range_of_f_l171_171979


namespace star_is_addition_l171_171413

theorem star_is_addition (star : ℝ → ℝ → ℝ) 
  (H : ∀ a b c : ℝ, star (star a b) c = a + b + c) : 
  ∀ a b : ℝ, star a b = a + b :=
by
  sorry

end star_is_addition_l171_171413


namespace parametric_equations_l171_171926

variables (t : ℝ)
def x_velocity : ℝ := 9
def y_velocity : ℝ := 12
def init_x : ℝ := 1
def init_y : ℝ := 1

theorem parametric_equations :
  (x = init_x + x_velocity * t) ∧ (y = init_y + y_velocity * t) :=
sorry

end parametric_equations_l171_171926


namespace number_of_spinsters_l171_171562

-- Given conditions
variables (S C : ℕ)
axiom ratio_condition : S / C = 2 / 9
axiom difference_condition : C = S + 63

-- Theorem to prove
theorem number_of_spinsters : S = 18 :=
sorry

end number_of_spinsters_l171_171562


namespace B_pow_97_l171_171853

open Matrix

def B : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![0, 1, 0],
  ![0, 0, 1],
  ![1, 0, 0]
]

theorem B_pow_97 : B ^ 97 = B := by
  sorry

end B_pow_97_l171_171853


namespace Amanda_family_paint_walls_l171_171434

theorem Amanda_family_paint_walls :
  let num_people := 5
  let rooms_with_4_walls := 5
  let rooms_with_5_walls := 4
  let walls_per_room_4 := 4
  let walls_per_room_5 := 5
  let total_walls := (rooms_with_4_walls * walls_per_room_4) + (rooms_with_5_walls * walls_per_room_5)
  total_walls / num_people = 8 :=
by
  -- We add a sorry to skip proof
  sorry

end Amanda_family_paint_walls_l171_171434


namespace min_distance_symmetry_l171_171683

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x^2 + x + 1

def line (x y : ℝ) : Prop := 2 * x - y = 3

theorem min_distance_symmetry :
  ∀ (P Q : ℝ × ℝ),
    line P.1 P.2 → line Q.1 Q.2 →
    (exists (x : ℝ), P = (x, f x)) ∧
    (exists (x : ℝ), Q = (x, f x)) →
    ∃ (d : ℝ), d = 2 * Real.sqrt 5 :=
sorry

end min_distance_symmetry_l171_171683


namespace sin_90_deg_l171_171146

theorem sin_90_deg : Real.sin (90 * Real.pi / 180) = 1 := 
by
  sorry

end sin_90_deg_l171_171146


namespace decrease_percent_in_revenue_l171_171563

theorem decrease_percent_in_revenue 
  (T C : ℝ) 
  (original_revenue : ℝ := T * C)
  (new_tax : ℝ := 0.80 * T)
  (new_consumption : ℝ := 1.15 * C)
  (new_revenue : ℝ := new_tax * new_consumption) :
  ((original_revenue - new_revenue) / original_revenue) * 100 = 8 := 
sorry

end decrease_percent_in_revenue_l171_171563


namespace sum_of_sins_is_zero_l171_171614

variable {x y z : ℝ}

theorem sum_of_sins_is_zero
  (h1 : Real.sin x = Real.tan y)
  (h2 : Real.sin y = Real.tan z)
  (h3 : Real.sin z = Real.tan x) :
  Real.sin x + Real.sin y + Real.sin z = 0 :=
sorry

end sum_of_sins_is_zero_l171_171614


namespace cost_price_of_computer_table_l171_171686

/-- The owner of a furniture shop charges 20% more than the cost price. 
    Given that the customer paid Rs. 3000 for the computer table, 
    prove that the cost price of the computer table was Rs. 2500. -/
theorem cost_price_of_computer_table (CP SP : ℝ) (h1 : SP = CP + 0.20 * CP) (h2 : SP = 3000) : CP = 2500 :=
by {
  sorry
}

end cost_price_of_computer_table_l171_171686


namespace compute_sin_90_l171_171149

noncomputable def sin_90_eq_one : Prop :=
  let angle_0_point := (1, 0) in
  let angle_90_point := (0, 1) in
  (angle_90_point.y = 1)  ∧ ∀ θ : ℝ, θ = 90 → Real.sin (θ * (Real.pi / 180)) = 1

theorem compute_sin_90 : sin_90_eq_one := 
by 
  -- the proof steps go here
  sorry

end compute_sin_90_l171_171149


namespace elijah_total_cards_l171_171609

-- Define the conditions
def num_decks : ℕ := 6
def cards_per_deck : ℕ := 52

-- The main statement that we need to prove
theorem elijah_total_cards : num_decks * cards_per_deck = 312 := by
  -- We skip the proof
  sorry

end elijah_total_cards_l171_171609


namespace chomp_game_configurations_l171_171645

/-- Number of valid configurations such that 0 ≤ a_1 ≤ a_2 ≤ ... ≤ a_5 ≤ 7 is 330 -/
theorem chomp_game_configurations :
  let valid_configs := {a : Fin 6 → Fin 8 // (∀ i j, i ≤ j → a i ≤ a j)}
  Fintype.card valid_configs = 330 :=
sorry

end chomp_game_configurations_l171_171645


namespace smallest_value_of_x_l171_171266

theorem smallest_value_of_x (x : ℝ) (h : 6 * x ^ 2 - 37 * x + 48 = 0) : x = 13 / 6 :=
sorry

end smallest_value_of_x_l171_171266


namespace smallest_N_value_proof_l171_171803

def smallest_value_N (N : ℕ) : Prop :=
  N > 70 ∧ (21 * N) % 70 = 0

theorem smallest_N_value_proof : ∃ N, smallest_value_N N ∧ (∀ M, smallest_value_N M → N ≤ M) :=
  sorry

end smallest_N_value_proof_l171_171803


namespace smallest_N_l171_171814

theorem smallest_N (N : ℕ) (hN : N > 70) (hdv : 70 ∣ 21 * N) : N = 80 :=
sorry

end smallest_N_l171_171814


namespace polynomial_evaluation_l171_171007

theorem polynomial_evaluation (x y : ℝ) (h : 2 * x^2 + 3 * y + 3 = 8) : 6 * x^2 + 9 * y + 8 = 23 :=
sorry

end polynomial_evaluation_l171_171007


namespace emily_irises_after_addition_l171_171060

theorem emily_irises_after_addition
  (initial_roses : ℕ)
  (added_roses : ℕ)
  (ratio_irises_roses : ℕ)
  (ratio_roses_irises : ℕ)
  (h_ratio : ratio_irises_roses = 3 ∧ ratio_roses_irises = 7)
  (h_initial_roses : initial_roses = 35)
  (h_added_roses : added_roses = 30) :
  ∃ irises_after_addition : ℕ, irises_after_addition = 27 :=
  by
    sorry

end emily_irises_after_addition_l171_171060


namespace eggs_problem_solution_l171_171432

theorem eggs_problem_solution :
  ∃ (n x : ℕ), 
  (120 * n = 206 * x) ∧
  (n = 103) ∧
  (x = 60) :=
by sorry

end eggs_problem_solution_l171_171432


namespace emily_patches_difference_l171_171862

theorem emily_patches_difference (h p : ℕ) (h_eq : p = 3 * h) :
  (p * h) - ((p + 5) * (h - 3)) = (4 * h + 15) :=
by
  sorry

end emily_patches_difference_l171_171862


namespace denomination_of_remaining_notes_eq_500_l171_171925

-- Definitions of the given conditions:
def total_money : ℕ := 10350
def total_notes : ℕ := 126
def n_50_notes : ℕ := 117

-- The theorem stating what we need to prove
theorem denomination_of_remaining_notes_eq_500 :
  ∃ (X : ℕ), X = 500 ∧ total_money = (n_50_notes * 50 + (total_notes - n_50_notes) * X) :=
by
sorry

end denomination_of_remaining_notes_eq_500_l171_171925


namespace gcd_polynomials_l171_171309

-- Define a as a multiple of 1836
def is_multiple_of (a b : ℤ) : Prop := ∃ k : ℤ, a = k * b

-- Problem statement: gcd of the polynomial expressions given the condition
theorem gcd_polynomials (a : ℤ) (h : is_multiple_of a 1836) : Int.gcd (2 * a^2 + 11 * a + 40) (a + 4) = 4 :=
by
  sorry

end gcd_polynomials_l171_171309


namespace swim_club_member_count_l171_171391

theorem swim_club_member_count :
  let total_members := 60
  let passed_percentage := 0.30
  let passed_members := total_members * passed_percentage
  let not_passed_members := total_members - passed_members
  let preparatory_course_members := 12
  not_passed_members - preparatory_course_members = 30 :=
by
  sorry

end swim_club_member_count_l171_171391


namespace polygon_sides_l171_171329

/-- If the sum of the interior angles of a polygon is three times the sum of its exterior angles,
    then the number of sides of the polygon is 8. -/
theorem polygon_sides (n : ℕ) (h1 : 180 * (n - 2) = 3 * 360) : n = 8 :=
sorry

end polygon_sides_l171_171329


namespace probability_of_same_color_is_correct_l171_171097

-- Definitions from the problem conditions
def red_marbles := 6
def white_marbles := 7
def blue_marbles := 8
def total_marbles := red_marbles + white_marbles + blue_marbles -- 21

-- Calculate the probability of drawing 4 red marbles
def P_all_red := (red_marbles / total_marbles) * ((red_marbles - 1) / (total_marbles - 1)) * ((red_marbles - 2) / (total_marbles - 2)) * ((red_marbles - 3) / (total_marbles - 3))

-- Calculate the probability of drawing 4 white marbles
def P_all_white := (white_marbles / total_marbles) * ((white_marbles - 1) / (total_marbles - 1)) * ((white_marbles - 2) / (total_marbles - 2)) * ((white_marbles - 3) / (total_marbles - 3))

-- Calculate the probability of drawing 4 blue marbles
def P_all_blue := (blue_marbles / total_marbles) * ((blue_marbles - 1) / (total_marbles - 1)) * ((blue_marbles - 2) / (total_marbles - 2)) * ((blue_marbles - 3) / (total_marbles - 3))

-- Total probability of drawing 4 marbles of the same color
def P_all_same_color := P_all_red + P_all_white + P_all_blue

-- Proof that the total probability is equal to the given correct answer
theorem probability_of_same_color_is_correct : P_all_same_color = 240 / 11970 := by
  sorry

end probability_of_same_color_is_correct_l171_171097


namespace brownie_pieces_count_l171_171030

def pan_width : ℕ := 24
def pan_height : ℕ := 15
def brownie_width : ℕ := 3
def brownie_height : ℕ := 2

theorem brownie_pieces_count : (pan_width * pan_height) / (brownie_width * brownie_height) = 60 := by
  sorry

end brownie_pieces_count_l171_171030


namespace problem_solution_l171_171447

theorem problem_solution :
  (1/3⁻¹) - Real.sqrt 27 + 3 * Real.tan (Real.pi / 6) + (Real.pi - 3.14)^0 = 4 - 2 * Real.sqrt 3 := by
  sorry

end problem_solution_l171_171447


namespace sin_90_degree_l171_171139

-- Definitions based on conditions
def unit_circle_point (angle : ℝ) : ℝ × ℝ :=
  if angle = 90 * (π / 180) then (0, 1) else sorry

def sin_usual (angle : ℝ) : ℝ :=
  (unit_circle_point angle).snd

-- The main theorem as per the question and conditions
theorem sin_90_degree : sin_usual (90 * (π / 180)) = 1 :=
by
  sorry

end sin_90_degree_l171_171139


namespace probability_of_X_l171_171394

variable (P : Prop → ℝ)
variable (event_X event_Y : Prop)

-- Defining the conditions
variable (hYP : P event_Y = 2 / 3)
variable (hXYP : P (event_X ∧ event_Y) = 0.13333333333333333)

-- Proving that the probability of selection of X is 0.2
theorem probability_of_X : P event_X = 0.2 := by
  sorry

end probability_of_X_l171_171394


namespace triangle_inscribed_in_semicircle_l171_171722

variables {R : ℝ} (P Q R' : ℝ) (PR QR : ℝ)
variables (hR : 0 < R) (h_pq_diameter: P = -R ∧ Q = R)
variables (h_pr_square_qr_square : PR^2 + QR^2 = 4 * R^2)
variables (t := PR + QR)

theorem triangle_inscribed_in_semicircle (h_pos_pr : 0 < PR) (h_pos_qr : 0 < QR) : 
  t^2 ≤ 8 * R^2 :=
sorry

end triangle_inscribed_in_semicircle_l171_171722


namespace total_books_read_l171_171715

-- Given conditions
variables (c s : ℕ) -- variable c represents the number of classes, s represents the number of students per class

-- Main statement to prove
theorem total_books_read (h1 : ∀ a, a = 7) (h2 : ∀ b, b = 12) :
  84 * c * s = 84 * c * s :=
by
  sorry

end total_books_read_l171_171715


namespace sin_90_eq_1_l171_171126

-- Define the unit circle
def unit_circle (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define the sine of 90 degrees using radians
def sin_90_degrees : ℝ := unit_circle (Real.pi / 2).snd

-- State the theorem
theorem sin_90_eq_1 : sin_90_degrees = 1 :=
by
  sorry

end sin_90_eq_1_l171_171126


namespace sales_difference_l171_171570

-- Definitions of the conditions
def daily_avg_sales_pastries := 20 * 2
def daily_avg_sales_bread := 10 * 4
def daily_avg_sales := daily_avg_sales_pastries + daily_avg_sales_bread

def today_sales_pastries := 14 * 2
def today_sales_bread := 25 * 4
def today_sales := today_sales_pastries + today_sales_bread

-- Statement to be proved
theorem sales_difference : today_sales - daily_avg_sales = 48 :=
by {
  -- Unpack the definitions
  simp [daily_avg_sales_pastries, daily_avg_sales_bread, daily_avg_sales],
  simp [today_sales_pastries, today_sales_bread, today_sales],
  -- Computation,
  -- daily_avg_sales == 20 * 2 + 10 * 4 == 80,
  -- today_sales == 14 * 2 + 25 * 4 == 128
  -- therefore, 128 - 80 == 48,
  -- QED.
  sorry
}

end sales_difference_l171_171570


namespace sqrt_of_25_l171_171116

theorem sqrt_of_25 : ∃ x : ℝ, x^2 = 25 ∧ (x = 5 ∨ x = -5) :=
by {
  sorry
}

end sqrt_of_25_l171_171116


namespace smallest_of_seven_consecutive_even_numbers_l171_171521

theorem smallest_of_seven_consecutive_even_numbers (n : ℤ) :
  (n - 6) + (n - 4) + (n - 2) + n + (n + 2) + (n + 4) + (n + 6) = 448 → 
  (n - 6) = 58 :=
by
  sorry

end smallest_of_seven_consecutive_even_numbers_l171_171521


namespace decision_block_has_two_exits_l171_171085

-- Define the conditions based on the problem
def output_block_exits := 1
def processing_block_exits := 1
def start_end_block_exits := 0
def decision_block_exits := 2

-- The proof statement
theorem decision_block_has_two_exits :
  (output_block_exits = 1) ∧
  (processing_block_exits = 1) ∧
  (start_end_block_exits = 0) ∧
  (decision_block_exits = 2) →
  decision_block_exits = 2 :=
by
  sorry

end decision_block_has_two_exits_l171_171085


namespace eighth_group_number_correct_stratified_sampling_below_30_correct_l171_171208

noncomputable def systematic_sampling_eighth_group_number 
  (total_employees : ℕ) (sample_size : ℕ) (groups : ℕ) (fifth_group_number : ℕ) : ℕ :=
  let interval := total_employees / groups
  let initial_number := fifth_group_number - 4 * interval
  initial_number + 7 * interval

theorem eighth_group_number_correct :
  systematic_sampling_eighth_group_number 200 40 40 22 = 37 :=
  sorry

noncomputable def stratified_sampling_below_30_persons 
  (total_employees : ℕ) (sample_size : ℕ) (percent_below_30 : ℕ) : ℕ :=
  (percent_below_30 * sample_size) / 100

theorem stratified_sampling_below_30_correct :
  stratified_sampling_below_30_persons 200 40 40 = 16 :=
  sorry

end eighth_group_number_correct_stratified_sampling_below_30_correct_l171_171208


namespace hiking_rate_up_the_hill_l171_171088

theorem hiking_rate_up_the_hill (r_down : ℝ) (t_total : ℝ) (t_up : ℝ) (r_up : ℝ) :
  r_down = 6 ∧ t_total = 3 ∧ t_up = 1.2 → r_up * t_up = 9 * t_up :=
by
  intro h
  let ⟨hrd, htt, htu⟩ := h
  sorry

end hiking_rate_up_the_hill_l171_171088


namespace C_share_l171_171405

-- Conditions in Lean definition
def ratio_A_C (A C : ℕ) : Prop := 3 * C = 2 * A
def ratio_A_B (A B : ℕ) : Prop := 3 * B = A
def total_profit : ℕ := 60000

-- Lean statement
theorem C_share (A B C : ℕ) (h1 : ratio_A_C A C) (h2 : ratio_A_B A B) : (C * total_profit) / (A + B + C) = 20000 :=
  by
  sorry

end C_share_l171_171405


namespace inflation_two_years_real_rate_of_return_l171_171708

-- Proof Problem for Question 1
theorem inflation_two_years :
  ((1 + 0.015)^2 - 1) * 100 = 3.0225 :=
by
  sorry

-- Proof Problem for Question 2
theorem real_rate_of_return :
  ((1.07 * 1.07) / (1 + 0.030225) - 1) * 100 = 11.13 :=
by
  sorry

end inflation_two_years_real_rate_of_return_l171_171708


namespace smallest_solution_l171_171071

theorem smallest_solution : ∃ x : ℝ, x^4 - 50 * x^2 + 625 = 0 ∧ ∀ y : ℝ, y^4 - 50 * y^2 + 625 = 0 → x ≤ y :=
by {
  use -5,
  split,
  sorry, -- here would be the proof that (-5)^4 - 50 * (-5)^2 + 625 = 0
  intros y hy,
  sorry -- here would be the proof that for any y such that y^4 - 50 * y^2 + 625 = 0, -5 ≤ y
}

end smallest_solution_l171_171071


namespace range_of_a_l171_171322

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ (3 / 2)^x = (2 + 3 * a) / (5 - a)) ↔ a ∈ Set.Ioo (-2 / 3) (3 / 4) :=
by
  sorry

end range_of_a_l171_171322


namespace five_in_range_for_all_b_l171_171971

noncomputable def f (x b : ℝ) := x^2 + b * x - 3

theorem five_in_range_for_all_b : ∀ (b : ℝ), ∃ (x : ℝ), f x b = 5 := by 
  sorry

end five_in_range_for_all_b_l171_171971


namespace evaluate_expression_l171_171028

noncomputable def a : ℕ := 2
noncomputable def b : ℕ := 1

theorem evaluate_expression : (1 / 2)^(b - a + 1) = 1 :=
by
  sorry

end evaluate_expression_l171_171028


namespace selling_price_is_correct_l171_171040

noncomputable def purchase_price : ℝ := 36400
noncomputable def repair_costs : ℝ := 8000
noncomputable def profit_percent : ℝ := 54.054054054054056

noncomputable def total_cost := purchase_price + repair_costs
noncomputable def selling_price := total_cost * (1 + profit_percent / 100)

theorem selling_price_is_correct :
    selling_price = 68384 := by
  sorry

end selling_price_is_correct_l171_171040


namespace golden_section_AC_correct_l171_171313

noncomputable def golden_ratio : ℝ := (1 + Real.sqrt 5) / 2
noncomputable def segment_length := 20
noncomputable def golden_section_point (AB AC BC : ℝ) (h1 : AB = AC + BC) (h2 : AC > BC) (h3 : AB = segment_length) : Prop :=
  AC = (Real.sqrt 5 - 1) / 2 * AB

theorem golden_section_AC_correct :
  ∃ (AC BC : ℝ), (AC + BC = segment_length) ∧ (AC > BC) ∧ (AC = 10 * (Real.sqrt 5 - 1)) :=
by
  sorry

end golden_section_AC_correct_l171_171313


namespace compute_sin_90_l171_171147

noncomputable def sin_90_eq_one : Prop :=
  let angle_0_point := (1, 0) in
  let angle_90_point := (0, 1) in
  (angle_90_point.y = 1)  ∧ ∀ θ : ℝ, θ = 90 → Real.sin (θ * (Real.pi / 180)) = 1

theorem compute_sin_90 : sin_90_eq_one := 
by 
  -- the proof steps go here
  sorry

end compute_sin_90_l171_171147


namespace smallest_N_l171_171817

theorem smallest_N (N : ℕ) (hN : N > 70) (hDiv : 70 ∣ 21 * N) : N = 80 :=
sorry

end smallest_N_l171_171817


namespace ratio_Polly_to_Pulsar_l171_171347

theorem ratio_Polly_to_Pulsar (P Po Pe : ℕ) (k : ℕ) (h1 : P = 10) (h2 : Po = k * P) (h3 : Pe = Po / 6) (h4 : P + Po + Pe = 45) : Po / P = 3 :=
by 
  -- Skipping the proof, but this sets up the Lean environment
  sorry

end ratio_Polly_to_Pulsar_l171_171347


namespace player1_wins_11th_round_l171_171900

noncomputable def egg_strength_probability (n : ℕ) : ℚ :=
  (n - 1) / n

theorem player1_wins_11th_round :
  let player1_wins_first_10_rounds := true,
      total_rounds := 11,
      new_egg := 12 in
  player1_wins_first_10_rounds → egg_strength_probability total_rounds = 11 / 12 :=
by
  intros
  exact sorry

end player1_wins_11th_round_l171_171900


namespace bridge_length_l171_171716

variable (speed : ℝ) (time_minutes : ℝ)
variable (time_hours : ℝ := time_minutes / 60)

theorem bridge_length (h1 : speed = 5) (h2 : time_minutes = 15) : 
  speed * time_hours = 1.25 := by
  sorry

end bridge_length_l171_171716


namespace max_gcd_13n_plus_4_8n_plus_3_l171_171598

theorem max_gcd_13n_plus_4_8n_plus_3 : ∃ n : ℕ, n > 0 ∧ Int.gcd (13 * n + 4) (8 * n + 3) = 11 := 
sorry

end max_gcd_13n_plus_4_8n_plus_3_l171_171598


namespace cricket_run_target_l171_171213

/-- Assuming the run rate in the first 15 overs and the required run rate for the next 35 overs to
reach a target, prove that the target number of runs is 275. -/
theorem cricket_run_target
  (run_rate_first_15 : ℝ := 3.2)
  (overs_first_15 : ℝ := 15)
  (run_rate_remaining_35 : ℝ := 6.485714285714286)
  (overs_remaining_35 : ℝ := 35)
  (runs_first_15 := run_rate_first_15 * overs_first_15)
  (runs_remaining_35 := run_rate_remaining_35 * overs_remaining_35)
  (target_runs := runs_first_15 + runs_remaining_35) :
  target_runs = 275 := by
  sorry

end cricket_run_target_l171_171213


namespace find_n_l171_171960

theorem find_n :
  ∃ (n : ℤ), (4 ≤ n ∧ n ≤ 8) ∧ (n % 5 = 2) ∧ (n = 7) :=
by
  sorry

end find_n_l171_171960


namespace smallest_possible_value_of_N_l171_171827

-- Declares the context and required constraints
theorem smallest_possible_value_of_N (N : ℕ) (h1 : 70 < N) (h2 : 70 ∣ 21 * N) : N = 80 :=
by
  sorry

end smallest_possible_value_of_N_l171_171827


namespace find_cost_10_pound_bag_l171_171574

def cost_5_pound_bag : ℝ := 13.82
def cost_25_pound_bag : ℝ := 32.25
def minimum_required_weight : ℝ := 65
def maximum_required_weight : ℝ := 80
def least_possible_cost : ℝ := 98.75
def cost_10_pound_bag (cost : ℝ) : Prop :=
  ∃ n m l, 
    (n * 5 + m * 10 + l * 25 ≥ minimum_required_weight) ∧
    (n * 5 + m * 10 + l * 25 ≤ maximum_required_weight) ∧
    (n * cost_5_pound_bag + m * cost + l * cost_25_pound_bag = least_possible_cost)

theorem find_cost_10_pound_bag : cost_10_pound_bag 2 := 
by
  sorry

end find_cost_10_pound_bag_l171_171574


namespace latest_time_for_temperature_at_60_l171_171834

theorem latest_time_for_temperature_at_60
  (t : ℝ) (h : -t^2 + 10 * t + 40 = 60) : t = 12 :=
sorry

end latest_time_for_temperature_at_60_l171_171834


namespace prob_first_given_defective_correct_l171_171288

-- Definitions from problem conditions
def first_box : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
def second_box : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
def defective_first_box : Set ℕ := {1, 2, 3}
def defective_second_box : Set ℕ := {1, 2}

-- Probability values as defined
def prob_first_box : ℚ := 1 / 2
def prob_second_box : ℚ := 1 / 2
def prob_defective_given_first : ℚ := 3 / 10
def prob_defective_given_second : ℚ := 1 / 10

-- Calculation of total probability of defective component
def prob_defective : ℚ := (prob_first_box * prob_defective_given_first) + (prob_second_box * prob_defective_given_second)

-- Bayes' Theorem application to find the required probability
def prob_first_given_defective : ℚ := (prob_first_box * prob_defective_given_first) / prob_defective

-- Lean statement to verify the computed probability is as expected
theorem prob_first_given_defective_correct : prob_first_given_defective = 3 / 4 :=
by
  unfold prob_first_given_defective prob_defective
  sorry

end prob_first_given_defective_correct_l171_171288


namespace probability_sum_is_4_l171_171253

open Real

def rounding (x : ℝ) : ℤ :=
  if x - floor x < 0.5 then floor x else ceil x

theorem probability_sum_is_4 (x : ℝ) (h : 0 ≤ x ∧ x ≤ 3.5)
  (hx : 0.5 ≤ x ∧ x < 1.5 ∨ 1.5 ≤ x ∧ x < 2) :
  let rounded_sum := rounding x + rounding (3.5 - x) 
  in (rounded_sum = 4) → (3 / 7 : ℝ) :=
sorry

end probability_sum_is_4_l171_171253


namespace non_adjacent_arrangement_l171_171973

-- Define the number of people
def numPeople : ℕ := 8

-- Define the number of specific people who must not be adjacent
def numSpecialPeople : ℕ := 3

-- Define the number of general people who are not part of the specific group
def numGeneralPeople : ℕ := numPeople - numSpecialPeople

-- Permutations calculation for general people
def permuteGeneralPeople : ℕ := Nat.factorial numGeneralPeople

-- Number of gaps available after arranging general people
def numGaps : ℕ := numGeneralPeople + 1

-- Permutations calculation for special people placed in the gaps
def permuteSpecialPeople : ℕ := Nat.descFactorial numGaps numSpecialPeople

-- Total permutations
def totalPermutations : ℕ := permuteSpecialPeople * permuteGeneralPeople

theorem non_adjacent_arrangement :
  totalPermutations = Nat.descFactorial 6 3 * Nat.factorial 5 := by
  sorry

end non_adjacent_arrangement_l171_171973


namespace lindas_nickels_l171_171506

theorem lindas_nickels
  (N : ℕ)
  (initial_dimes : ℕ := 2)
  (initial_quarters : ℕ := 6)
  (initial_nickels : ℕ := N)
  (additional_dimes : ℕ := 2)
  (additional_quarters : ℕ := 10)
  (additional_nickels : ℕ := 2 * N)
  (total_coins : ℕ := 35)
  (h : initial_dimes + initial_quarters + initial_nickels + additional_dimes + additional_quarters + additional_nickels = total_coins) :
  N = 5 := by
  sorry

end lindas_nickels_l171_171506


namespace intersection_eq_l171_171633

def A : Set ℕ := {1, 2, 4, 6, 8}
def B : Set ℕ := {x | ∃ k ∈ A, x = 2 * k}

theorem intersection_eq : A ∩ B = {2, 4, 8} := by
  sorry

end intersection_eq_l171_171633


namespace recurring_decimal_as_fraction_l171_171269

theorem recurring_decimal_as_fraction :
  0.53 + (247 / 999) * 0.001 = 53171 / 99900 :=
by
  sorry

end recurring_decimal_as_fraction_l171_171269


namespace ticTacToe_CarlWins_l171_171947

def ticTacToeBoard := Fin 3 × Fin 3

noncomputable def countConfigurations : Nat := sorry

theorem ticTacToe_CarlWins :
  countConfigurations = 148 :=
sorry

end ticTacToe_CarlWins_l171_171947


namespace smallest_N_l171_171822

theorem smallest_N (N : ℕ) (hN : N > 70) (hDiv : 70 ∣ 21 * N) : N = 80 :=
sorry

end smallest_N_l171_171822


namespace tangent_line_circle_l171_171762

theorem tangent_line_circle : 
  ∃ (k : ℚ), (∀ x y : ℚ, ((x - 3) ^ 2 + (y - 4) ^ 2 = 25) 
               → (3 * x + 4 * y - 25 = 0)) :=
sorry

end tangent_line_circle_l171_171762


namespace shoe_store_total_shoes_l171_171689

theorem shoe_store_total_shoes (b k : ℕ) (h1 : b = 22) (h2 : k = 2 * b) : b + k = 66 :=
by
  sorry

end shoe_store_total_shoes_l171_171689


namespace dave_deleted_apps_l171_171295

def apps_initial : ℕ := 23
def apps_left : ℕ := 5
def apps_deleted : ℕ := apps_initial - apps_left

theorem dave_deleted_apps : apps_deleted = 18 := 
by
  sorry

end dave_deleted_apps_l171_171295


namespace rectangle_area_change_l171_171370

theorem rectangle_area_change (L W : ℝ) (h : L * W = 540) :
  (0.8 * L) * (1.15 * W) ≈ 497 :=
by
  -- Given the initial area condition
  -- Calculate the new area after changing the dimensions
  have : 0.8 * L * (1.15 * W) = 0.92 * L * W,
  { ring },
  equiv_rw ← h at this,
  rw ← (show 0.92 * 540 = 496.8, by norm_num) at this,
  ring
      
  -- Show the new area is approximately equal to 497 square centimeters
  sorry

end rectangle_area_change_l171_171370


namespace birdhouse_volume_difference_l171_171041

theorem birdhouse_volume_difference :
  let sara_width := 1
  let sara_height := 2
  let sara_depth := 2
  let jake_width := 16 / 12
  let jake_height := 20 / 12
  let jake_depth := 18 / 12
  let sara_volume := sara_width * sara_height * sara_depth
  let jake_volume := jake_width * jake_height * jake_depth
  let volume_difference := sara_volume - jake_volume
  volume_difference ≈ 0.668 :=
by
  sorry

end birdhouse_volume_difference_l171_171041


namespace quarters_spent_l171_171239

theorem quarters_spent (original : ℕ) (remaining : ℕ) (q : ℕ) 
  (h1 : original = 760) 
  (h2 : remaining = 342) 
  (h3 : q = original - remaining) : q = 418 := 
by
  sorry

end quarters_spent_l171_171239


namespace area_of_circle_l171_171954

open Real

theorem area_of_circle :
  ∃ (A : ℝ), (∀ x y : ℝ, (x^2 + y^2 - 4 * x + 2 * y - 11 = 0) → A = 16 * π) :=
sorry

end area_of_circle_l171_171954


namespace arithmetic_sequence_sum_l171_171976

variable {α : Type*} [LinearOrderedField α]

def sum_n_terms (a₁ d : α) (n : ℕ) : α :=
  n / 2 * (2 * a₁ + (n - 1) * d)

theorem arithmetic_sequence_sum 
  (a₁ : α) (h : sum_n_terms a₁ 1 4 = 1) :
  sum_n_terms a₁ 1 8 = 18 := by
  sorry

end arithmetic_sequence_sum_l171_171976


namespace arithmetic_geometric_sequence_l171_171541

theorem arithmetic_geometric_sequence (x y z : ℤ) :
  (x ≠ y ∧ y ≠ z ∧ x ≠ z) ∧
  ((x + y + z = 6) ∧ (y - x = z - y) ∧ (y^2 = x * z)) →
  (x = -4 ∧ y = 2 ∧ z = 8 ∨ x = 8 ∧ y = 2 ∧ z = -4) :=
by
  intros h
  sorry

end arithmetic_geometric_sequence_l171_171541


namespace sin_90_eq_1_l171_171137

theorem sin_90_eq_1 :
  let θ := 90 : ℝ in
  let cos_θ := real.cos θ in
  let sin_θ := real.sin θ in 
  let rotation_matrix := ![![cos_θ, -sin_θ], ![sin_θ, cos_θ]] in
  let point := ![1, 0] in
  let rotated_point := matrix.mul_vec rotation_matrix point in
  rotated_point = ![0, 1] → 
  sin_θ = 1 :=
by
  sorry

end sin_90_eq_1_l171_171137


namespace m_range_l171_171985

open Real

-- Define the points
def A : ℝ × ℝ := (-1, 2)
def B : ℝ × ℝ := (2, -1)

-- Define the line equation
def line_eq (x y m : ℝ) : Prop := x - 2*y + m = 0

-- Theorem: m must belong to the interval [-4, 5]
theorem m_range (m : ℝ) : (line_eq A.1 A.2 m) → (line_eq B.1 B.2 m) → -4 ≤ m ∧ m ≤ 5 := 
sorry

end m_range_l171_171985


namespace Isabel_afternoon_runs_l171_171217

theorem Isabel_afternoon_runs (circuit_length morning_runs weekly_distance afternoon_runs : ℕ)
  (h_circuit_length : circuit_length = 365)
  (h_morning_runs : morning_runs = 7)
  (h_weekly_distance : weekly_distance = 25550)
  (h_afternoon_runs : weekly_distance = morning_runs * circuit_length * 7 + afternoon_runs * circuit_length) :
  afternoon_runs = 21 :=
by
  -- The actual proof goes here
  sorry

end Isabel_afternoon_runs_l171_171217


namespace arithmetic_progression_common_difference_zero_l171_171211

theorem arithmetic_progression_common_difference_zero {a d : ℤ} (h₁ : a = 12) 
  (h₂ : ∀ n : ℕ, a + n * d = (a + (n + 1) * d + a + (n + 2) * d) / 2) : d = 0 :=
  sorry

end arithmetic_progression_common_difference_zero_l171_171211


namespace bags_total_weight_l171_171956

noncomputable def total_weight_of_bags (x y z : ℕ) : ℕ := x + y + z

theorem bags_total_weight (x y z : ℕ) (h1 : x + y = 90) (h2 : y + z = 100) (h3 : z + x = 110) :
  total_weight_of_bags x y z = 150 :=
by
  sorry

end bags_total_weight_l171_171956


namespace negation_of_proposition_l171_171881

theorem negation_of_proposition : (¬ ∃ x : ℝ, x > 0 ∧ x^2 - 5 * x + 6 > 0) ↔ ∀ x : ℝ, x > 0 → x^2 - 5 * x + 6 ≤ 0 := by
  sorry

end negation_of_proposition_l171_171881


namespace parabola_intersections_l171_171697

-- Definitions of the parabolas
def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 9 * x + 1
def parabola2 (x : ℝ) : ℝ := 2 * x^2 - 5 * x + 3

-- The statement to be proven
theorem parabola_intersections :
    (parabola1 (2 + Real.sqrt 6) = 13 + 3 * Real.sqrt 6) ∧ 
    (parabola1 (2 - Real.sqrt 6) = 13 - 3 * Real.sqrt 6) ∧
    (parabola2 (2 + Real.sqrt ) = 13 + 3 * Real.sqrt 6) ∧ 
    (parabola2 (2 - Real.sqrt 6) = 13 - 3 * Real.sqrt 6) := 
by
  sorry

end parabola_intersections_l171_171697


namespace not_countably_additive_l171_171294

noncomputable def nu_n (n : ℕ) (B : set ℝ) : MeasureTheory.Measure ℝ :=
  MeasureTheory.Measure.restrict (MeasureTheory.Measure.lebesgue) (univ \ Icc (-n : ℝ) (n : ℝ))

def seq_non_increasing (B : set ℝ) : Prop :=
  ∀ n : ℕ, nu_n (n + 1) B ≤ nu_n n B

def limit_measure_nu (B : set ℝ) : MeasureTheory.Measure ℝ :=
  MeasureTheory.Measure.of_real (if MeasureTheory.Measure.lebesgue B = ∞ then ∞ else 0)

theorem not_countably_additive (B : set ℝ) (hB : MeasureTheory.Measure.lebesgue B < ∞) 
  (hB_union : ∀ (k : ℕ), disjoint (λ i : ℕ, B i) (B \ (⋃ j < k, B j))) :
  limit_measure_nu (⋃ k, B k) ≠ ∑ k, limit_measure_nu (B k) :=
sorry

end not_countably_additive_l171_171294


namespace sequence_inequality_l171_171889

theorem sequence_inequality (a : ℕ → ℕ) (strictly_increasing : ∀ n, a n < a (n + 1))
  (sum_condition : ∀ m : ℕ, ∃ i j : ℕ, m = a i + a j) :
  ∀ n, a n ≤ n^2 :=
by sorry

end sequence_inequality_l171_171889


namespace smallest_solution_x4_minus_50x2_plus_625_eq_0_l171_171077

theorem smallest_solution_x4_minus_50x2_plus_625_eq_0 : ∃ x : ℝ, x^4 - 50 * x^2 + 625 = 0 ∧ ∀ y : ℝ, y^4 - 50 * y^2 + 625 = 0 → x ≤ y := 
sorry

end smallest_solution_x4_minus_50x2_plus_625_eq_0_l171_171077


namespace quad_form_b_c_sum_l171_171385

theorem quad_form_b_c_sum :
  ∃ (b c : ℝ), (b + c = -10) ∧ (∀ x : ℝ, x^2 - 20 * x + 100 = (x + b)^2 + c) :=
by
  sorry

end quad_form_b_c_sum_l171_171385


namespace triangle_perimeter_l171_171384

theorem triangle_perimeter : 
  ∀ (f : ℝ → ℝ), 
  (∀ x, f x = 4 * (1 - x / 3)) →
  ∃ (A B C : ℝ × ℝ), 
  A = (3, 0) ∧ 
  B = (0, 4) ∧ 
  C = (0, 0) ∧ 
  dist A B + dist B C + dist C A = 12 :=
by
  sorry

end triangle_perimeter_l171_171384


namespace greatest_possible_value_of_x_l171_171867

theorem greatest_possible_value_of_x
    (x : ℕ)
    (h1 : x > 0)
    (h2 : x % 4 = 0)
    (h3 : x^3 < 8000) :
    x ≤ 16 :=
    sorry

end greatest_possible_value_of_x_l171_171867


namespace license_plate_palindrome_probability_l171_171338

theorem license_plate_palindrome_probability :
  let p := 507
  let q := 2028
  p + q = 2535 :=
by
  sorry

end license_plate_palindrome_probability_l171_171338


namespace find_a_plus_d_l171_171974

theorem find_a_plus_d (a b c d : ℝ) (h₁ : ab + bc + ca + db = 42) (h₂ : b + c = 6) : a + d = 7 := 
sorry

end find_a_plus_d_l171_171974


namespace difference_of_numbers_l171_171252

theorem difference_of_numbers (x y : ℕ) (h1 : x + y = 64) (h2 : y = 26) : x - y = 12 :=
sorry

end difference_of_numbers_l171_171252


namespace find_length_of_side_c_find_measure_of_angle_B_l171_171648

variable {A B C a b c : ℝ}

def triangle_problem (a b c A B C : ℝ) :=
  a * Real.cos B = 3 ∧
  b * Real.cos A = 1 ∧
  A - B = Real.pi / 6 ∧
  a^2 + c^2 - b^2 - 6 * c = 0 ∧
  b^2 + c^2 - a^2 - 2 * c = 0

theorem find_length_of_side_c (h : triangle_problem a b c A B C) :
  c = 4 :=
sorry

theorem find_measure_of_angle_B (h : triangle_problem a b c A B C) :
  B = Real.pi / 6 :=
sorry

end find_length_of_side_c_find_measure_of_angle_B_l171_171648


namespace part1_part2_l171_171226

def p (a : ℝ) : Prop := a^2 - 5*a - 6 > 0
def q (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a * x + 1 = 0 → x < 0

theorem part1 (a : ℝ) (hp : p a) : a ∈ Set.Iio (-1) ∪ Set.Ioi 6 :=
sorry

theorem part2 (a : ℝ) (h_or : p a ∨ q a) (h_and : ¬ (p a ∧ q a)) : a ∈ Set.Iio (-1) ∪ Set.Ioc 2 6 :=
sorry

end part1_part2_l171_171226


namespace min_n_for_circuit_l171_171065

theorem min_n_for_circuit
  (n : ℕ) 
  (p_success_component : ℝ)
  (p_work_circuit : ℝ) 
  (h1 : p_success_component = 0.5)
  (h2 : p_work_circuit = 1 - p_success_component ^ n) 
  (h3 : p_work_circuit ≥ 0.95) :
  n ≥ 5 := 
sorry

end min_n_for_circuit_l171_171065


namespace recurrence_relation_holds_sequence_converges_to_sqrt2_l171_171492

noncomputable def f (x : ℝ) : ℝ := x^2 - 2

def x_seq : ℕ → ℝ
| 0 := 2  -- Initial value x1 = 2
| (n+1) := (x_seq n)^2 / (2 * (x_seq n)) + 1 / (x_seq n / 2) -- x_{n+1} = (x_n^2 + 2) / (2 * x_n)

theorem recurrence_relation_holds : 
  ∀ n: ℕ, x_seq (n+1) = (x_seq n)^2 / (2 * (x_seq n)) + 1 / (x_seq n / 2) := 
by
  intro n
  sorry

theorem sequence_converges_to_sqrt2 : 
  tendsto x_seq at_top (nhds (sqrt 2)) := 
by 
  sorry

end recurrence_relation_holds_sequence_converges_to_sqrt2_l171_171492


namespace complement_of_M_in_U_l171_171504

namespace SetComplements

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 3, 5}
def complement_U_M : Set ℕ := U \ M

theorem complement_of_M_in_U :
  complement_U_M = {2, 4, 6} :=
by
  sorry

end SetComplements

end complement_of_M_in_U_l171_171504


namespace total_students_l171_171732

theorem total_students (n x : ℕ) (h1 : 3 * n + 48 = 6 * n) (h2 : 4 * n + x = 2 * n + 2 * x) : n = 16 :=
by
  sorry

end total_students_l171_171732


namespace vanAubel_theorem_l171_171503

variables (A B C O A1 B1 C1 : Type)
variables (CA1 A1B CB1 B1A CO OC1 : ℝ)

-- Given Conditions
axiom condition1 : CB1 / B1A = 1
axiom condition2 : CO / OC1 = 2

-- Van Aubel's theorem statement
theorem vanAubel_theorem : (CO / OC1) = (CA1 / A1B) + (CB1 / B1A) := by
  sorry

end vanAubel_theorem_l171_171503


namespace compute_difference_of_squares_l171_171746

theorem compute_difference_of_squares :
  (23 + 15) ^ 2 - (23 - 15) ^ 2 = 1380 := by
  sorry

end compute_difference_of_squares_l171_171746


namespace zeroes_at_end_base_8_of_factorial_15_l171_171997

theorem zeroes_at_end_base_8_of_factorial_15 : 
  let a := factorial 15
  in let num_twos := (∑ k in Icc 1 15, padicValRat 2 k)
  in num_twos / 3 = 3 :=
by {
  sorry
}

end zeroes_at_end_base_8_of_factorial_15_l171_171997


namespace perfect_square_461_l171_171271

theorem perfect_square_461 (x : ℤ) (y : ℤ) (hx : 5 ∣ x) (hy : 5 ∣ y) 
  (h : x^2 + 461 = y^2) : x^2 = 52900 :=
  sorry

end perfect_square_461_l171_171271


namespace union_P_complement_Q_l171_171197

-- Define sets P and Q
def P : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def Q : Set ℝ := {x | x^2 ≥ 4}

-- Define the complement of Q in ℝ
def C_RQ : Set ℝ := {x | -2 < x ∧ x < 2}

-- State the main theorem
theorem union_P_complement_Q : (P ∪ C_RQ) = {x | -2 < x ∧ x ≤ 3} := 
by
  sorry

end union_P_complement_Q_l171_171197


namespace easter_egg_battle_probability_l171_171901

theorem easter_egg_battle_probability (players : Type) [fintype players] [decidable_eq players]
  (egg_strength : players → ℕ) (p1 : players) (p2 : players) (n : ℕ) [decidable (p1 ≠ p2)] :
  (∀ i in finset.range n, egg_strength p1 > egg_strength p2) →
  let prob11thWin := 11 / 12 in
  11 / 12 = prob11thWin :=
by sorry

end easter_egg_battle_probability_l171_171901


namespace triangle_perimeter_l171_171641

theorem triangle_perimeter (a : ℕ) (h1 : a < 8) (h2 : a > 4) (h3 : a % 2 = 0) : 2 + 6 + a = 14 :=
  by
  sorry

end triangle_perimeter_l171_171641


namespace identical_answers_l171_171765
-- Import necessary libraries

-- Define the entities and conditions
structure Person :=
  (name : String)
  (always_tells_truth : Bool)

def Fyodor : Person := { name := "Fyodor", always_tells_truth := true }
def Sasha : Person := { name := "Sasha", always_tells_truth := false }

def answer (p : Person) : String :=
  if p.always_tells_truth then "Yes" else "No"

-- The theorem statement
theorem identical_answers :
  answer Fyodor = answer Sasha :=
by
  -- Proof steps will be filled in later
  sorry

end identical_answers_l171_171765


namespace Yvonne_probability_of_success_l171_171272

theorem Yvonne_probability_of_success
  (P_X : ℝ) (P_Z : ℝ) (P_XY_notZ : ℝ) :
  P_X = 1 / 3 →
  P_Z = 5 / 8 →
  P_XY_notZ = 0.0625 →
  ∃ P_Y : ℝ, P_Y = 0.5 :=
by
  intros hX hZ hXY_notZ
  existsi (0.5 : ℝ)
  sorry

end Yvonne_probability_of_success_l171_171272


namespace sequence_general_formula_l171_171178

theorem sequence_general_formula (a : ℕ → ℕ) (S : ℕ → ℕ) :
  a 2 = 4 →
  S 4 = 30 →
  (∀ n, n ≥ 2 → a (n + 1) + a (n - 1) = 2 * (a n + 1)) →
  ∀ n, a n = n^2 :=
by
  intros h1 h2 h3
  sorry

end sequence_general_formula_l171_171178


namespace postal_service_revenue_l171_171888

theorem postal_service_revenue 
  (price_colored : ℝ := 0.50)
  (price_bw : ℝ := 0.35)
  (price_golden : ℝ := 2.00)
  (sold_colored : ℕ := 578833)
  (sold_bw : ℕ := 523776)
  (sold_golden : ℕ := 120456) : 
  (price_colored * (sold_colored : ℝ) + 
  price_bw * (sold_bw : ℝ) + 
  price_golden * (sold_golden : ℝ) = 713650.10) :=
by
  sorry

end postal_service_revenue_l171_171888


namespace intersection_A_B_l171_171639

def set_A (x : ℝ) : Prop := 2 * x + 1 > 0
def set_B (x : ℝ) : Prop := abs (x - 1) < 2

theorem intersection_A_B : 
  {x : ℝ | set_A x} ∩ {x : ℝ | set_B x} = {x : ℝ | -1/2 < x ∧ x < 3} :=
by
  sorry

end intersection_A_B_l171_171639


namespace difference_of_squares_l171_171749

theorem difference_of_squares : (23 + 15)^2 - (23 - 15)^2 = 1380 := by
  sorry

end difference_of_squares_l171_171749


namespace smallest_possible_value_of_N_l171_171824

-- Declares the context and required constraints
theorem smallest_possible_value_of_N (N : ℕ) (h1 : 70 < N) (h2 : 70 ∣ 21 * N) : N = 80 :=
by
  sorry

end smallest_possible_value_of_N_l171_171824


namespace sum_of_digits_133131_l171_171687

noncomputable def extract_digits_sum (n : Nat) : Nat :=
  let digits := n.digits 10
  digits.foldl (· + ·) 0

theorem sum_of_digits_133131 :
  let ABCDEF := 665655 / 5
  extract_digits_sum ABCDEF = 12 :=
by
  sorry

end sum_of_digits_133131_l171_171687


namespace minimum_chess_pieces_l171_171066

theorem minimum_chess_pieces (n : ℕ) : 
  (n % 3 = 1) ∧ (n % 5 = 3) ∧ (n % 7 = 5) → 
  n = 103 :=
by 
  sorry

end minimum_chess_pieces_l171_171066


namespace difference_of_squares_l171_171748

theorem difference_of_squares : (23 + 15)^2 - (23 - 15)^2 = 1380 := by
  sorry

end difference_of_squares_l171_171748


namespace system_solution_l171_171460

theorem system_solution (x y : ℝ) (h1 : x + 5*y = 5) (h2 : 3*x - y = 3) : x + y = 2 := 
by
  sorry

end system_solution_l171_171460


namespace chord_intersection_probability_l171_171361

noncomputable def probability_chord_intersection : ℚ :=
1 / 3

theorem chord_intersection_probability 
    (A B C D : ℕ) 
    (total_points : ℕ) 
    (adjacent : A + 1 = B ∨ A = B + 1)
    (distinct : ∀ (A B C D : ℕ), A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
    (points_on_circle : total_points = 2023) :
    ∃ p : ℚ, p = probability_chord_intersection :=
by sorry

end chord_intersection_probability_l171_171361


namespace tangent_condition_l171_171615

def curve1 (x y : ℝ) : Prop := y = x ^ 3 + 2
def curve2 (x y m : ℝ) : Prop := y^2 - m * x = 1

theorem tangent_condition (m : ℝ) (h : ∃ x y : ℝ, curve1 x y ∧ curve2 x y m) :
  m = 4 + 2 * Real.sqrt 3 :=
sorry

end tangent_condition_l171_171615


namespace tank_filling_time_l171_171863

noncomputable def netWaterPerCycle (rateA rateB rateC : ℕ) : ℕ := rateA + rateB - rateC

noncomputable def totalTimeToFill (tankCapacity rateA rateB rateC cycleDuration : ℕ) : ℕ :=
  let netWater := netWaterPerCycle rateA rateB rateC
  let cyclesNeeded := tankCapacity / netWater
  cyclesNeeded * cycleDuration

theorem tank_filling_time :
  totalTimeToFill 750 40 30 20 3 = 45 :=
by
  -- replace "sorry" with the actual proof if required
  sorry

end tank_filling_time_l171_171863


namespace north_pond_ducks_l171_171670

-- Definitions based on the conditions
def ducks_lake_michigan : ℕ := 100
def twice_ducks_lake_michigan : ℕ := 2 * ducks_lake_michigan
def additional_ducks : ℕ := 6
def ducks_north_pond : ℕ := twice_ducks_lake_michigan + additional_ducks

-- Theorem to prove the answer
theorem north_pond_ducks : ducks_north_pond = 206 :=
by
  sorry

end north_pond_ducks_l171_171670


namespace otto_knives_l171_171345

theorem otto_knives (n : ℕ) (cost : ℕ) : 
  cost = 32 → 
  (n ≥ 1 → cost = 5 + ((min (n - 1) 3) * 4) + ((max 0 (n - 4)) * 3)) → 
  n = 9 :=
by
  intros h_cost h_structure
  sorry

end otto_knives_l171_171345


namespace sum_of_three_integers_mod_53_l171_171709

theorem sum_of_three_integers_mod_53 (a b c : ℕ) (h1 : a % 53 = 31) 
                                     (h2 : b % 53 = 22) (h3 : c % 53 = 7) : 
                                     (a + b + c) % 53 = 7 :=
by
  sorry

end sum_of_three_integers_mod_53_l171_171709


namespace min_value_am_gm_l171_171501

theorem min_value_am_gm (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a / b) + (b / c) + (c / d) + (d / a) ≥ 4 := 
sorry

end min_value_am_gm_l171_171501


namespace sum_of_products_two_at_a_time_l171_171534

theorem sum_of_products_two_at_a_time (a b c : ℝ)
  (h1 : a^2 + b^2 + c^2 = 267) 
  (h2 : a + b + c = 23) : 
  ab + bc + ac = 131 :=
by
  sorry

end sum_of_products_two_at_a_time_l171_171534


namespace max_2ab_plus_2bc_sqrt2_l171_171655

theorem max_2ab_plus_2bc_sqrt2 (a b c : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : a^2 + b^2 + c^2 = 1) :
  2 * a * b + 2 * b * c * Real.sqrt 2 ≤ Real.sqrt 3 :=
sorry

end max_2ab_plus_2bc_sqrt2_l171_171655


namespace melinda_math_books_probability_l171_171340

theorem melinda_math_books_probability :
  let boxes := 3
  let total_books := 15
  let math_books := 4
  let books_per_box := 5
  let favorable_ways := 8316
  let total_ways := (choose 15 5) * (choose 10 5) * (choose 5 5)
  (favorable_ways : ℚ) / total_ways = 769 / 100947 :=
by
  let boxes := 3
  let total_books := 15
  let math_books := 4
  let books_per_box := 5
  let favorable_ways := 8316
  let total_ways := (choose 15 5) * (choose 10 5) * (choose 5 5)
  sorry

end melinda_math_books_probability_l171_171340


namespace octagon_area_l171_171942

noncomputable def regular_octagon_area_inscribed_circle_radius3 : ℝ :=
  18 * Real.sqrt 2

theorem octagon_area
  (r : ℝ)
  (h : r = 3)
  (octagon_inscribed : ∀ (x : ℝ), x = r * 3 * Real.sin (π / 8)): 
  regular_octagon_area_inscribed_circle_radius3 = 18 * Real.sqrt 2 :=
by
  sorry

end octagon_area_l171_171942


namespace walls_divided_equally_l171_171436

-- Define the given conditions
def num_people : ℕ := 5
def num_rooms : ℕ := 9
def rooms_with_4_walls : ℕ := 5
def walls_per_room_4 : ℕ := 4
def rooms_with_5_walls : ℕ := 4
def walls_per_room_5 : ℕ := 5

-- Calculate the total number of walls
def total_walls : ℕ := (rooms_with_4_walls * walls_per_room_4) + (rooms_with_5_walls * walls_per_room_5)

-- Define the expected result
def walls_per_person : ℕ := total_walls / num_people

-- Theorem statement: Each person should paint 8 walls.
theorem walls_divided_equally : walls_per_person = 8 := by
  sorry

end walls_divided_equally_l171_171436


namespace carpet_dimensions_l171_171588

theorem carpet_dimensions (a b : ℕ) 
  (h1 : a^2 + b^2 = 38^2 + 55^2) 
  (h2 : a^2 + b^2 = 50^2 + 55^2) 
  (h3 : a ≤ b) : 
  (a = 25 ∧ b = 50) ∨ (a = 50 ∧ b = 25) :=
by {
  -- The proof would go here
  sorry
}

end carpet_dimensions_l171_171588


namespace savings_from_discount_l171_171852

-- Define the initial price
def initial_price : ℝ := 475.00

-- Define the discounted price
def discounted_price : ℝ := 199.00

-- The theorem to prove the savings amount
theorem savings_from_discount : initial_price - discounted_price = 276.00 :=
by 
  -- This is where the actual proof would go
  sorry

end savings_from_discount_l171_171852


namespace total_triangles_in_geometric_figure_l171_171738

noncomputable def numberOfTriangles : ℕ :=
  let smallest_triangles := 3 + 2 + 1
  let medium_triangles := 2
  let large_triangle := 1
  smallest_triangles + medium_triangles + large_triangle

theorem total_triangles_in_geometric_figure : numberOfTriangles = 9 := by
  unfold numberOfTriangles
  sorry

end total_triangles_in_geometric_figure_l171_171738


namespace actual_distance_traveled_l171_171717

theorem actual_distance_traveled 
  (D : ℝ)
  (h1 : ∃ (D : ℝ), D/12 = (D + 36)/20)
  : D = 54 :=
sorry

end actual_distance_traveled_l171_171717


namespace weird_fraction_implies_weird_power_fraction_l171_171865

theorem weird_fraction_implies_weird_power_fraction 
  (a b c : ℝ) (k : ℕ) 
  (h1 : (1/a) + (1/b) + (1/c) = (1/(a + b + c))) 
  (h2 : Odd k) : 
  (1 / (a^k) + 1 / (b^k) + 1 / (c^k) = 1 / (a^k + b^k + c^k)) := 
by 
  sorry

end weird_fraction_implies_weird_power_fraction_l171_171865


namespace sin_ninety_degrees_l171_171132

theorem sin_ninety_degrees : Real.sin (90 * Real.pi / 180) = 1 := 
by
  sorry

end sin_ninety_degrees_l171_171132


namespace find_other_subject_given_conditions_l171_171283

theorem find_other_subject_given_conditions :
  ∀ (P C M : ℕ),
  P = 65 →
  (P + C + M) / 3 = 85 →
  (P + M) / 2 = 90 →
  ∃ (S : ℕ), (P + S) / 2 = 70 ∧ S = C :=
by
  sorry

end find_other_subject_given_conditions_l171_171283


namespace find_antonym_word_l171_171284

-- Defining the condition that the word means "rarely" or "not often."
def means_rarely_or_not_often (word : String) : Prop :=
  word = "seldom"

-- Theorem statement: There exists a word such that it meets the given condition.
theorem find_antonym_word : 
  ∃ word : String, means_rarely_or_not_often word :=
by
  use "seldom"
  unfold means_rarely_or_not_often
  rfl

end find_antonym_word_l171_171284


namespace find_a7_l171_171214

def seq (a : ℕ → ℚ) : Prop :=
  a 1 = -4/3 ∧ (∀ n, a (n + 2) = 1 / (a n + 1))

theorem find_a7 (a : ℕ → ℚ) (h : seq a) : a 7 = 2 :=
by
  sorry

end find_a7_l171_171214


namespace adam_money_ratio_l171_171433

theorem adam_money_ratio 
  (initial_dollars: ℕ) 
  (spent_dollars: ℕ) 
  (remaining_dollars: ℕ := initial_dollars - spent_dollars) 
  (ratio_numerator: ℕ := remaining_dollars / Nat.gcd remaining_dollars spent_dollars) 
  (ratio_denominator: ℕ := spent_dollars / Nat.gcd remaining_dollars spent_dollars) 
  (h_initial: initial_dollars = 91) 
  (h_spent: spent_dollars = 21) 
  (h_gcd: Nat.gcd (initial_dollars - spent_dollars) spent_dollars = 7) :
  ratio_numerator = 10 ∧ ratio_denominator = 3 := by
  sorry

end adam_money_ratio_l171_171433


namespace sum_of_values_of_m_l171_171642

-- Define the inequality conditions
def condition1 (x m : ℝ) : Prop := (x - m) / 2 ≥ 0
def condition2 (x : ℝ) : Prop := x + 3 < 3 * (x - 1)

-- Define the equation constraint for y
def fractional_equation (y m : ℝ) : Prop := (3 - y) / (2 - y) + m / (y - 2) = 3

-- Sum function for the values of m
def sum_of_m (m1 m2 m3 : ℝ) : ℝ := m1 + m2 + m3

-- Main theorem
theorem sum_of_values_of_m : sum_of_m 3 (-3) (-1) = -1 := 
by { sorry }

end sum_of_values_of_m_l171_171642


namespace find_x_l171_171216

-- Define the condition variables
variables (y z x : ℝ) (Y Z X : ℝ)
-- Primary conditions given in the problem
variable (h_y : y = 7)
variable (h_z : z = 6)
variable (h_cosYZ : Real.cos (Y - Z) = 15 / 16)

-- The main theorem to prove
theorem find_x (h_y : y = 7) (h_z : z = 6) (h_cosYZ : Real.cos (Y - Z) = 15 / 16) :
  x = Real.sqrt 22 :=
sorry

end find_x_l171_171216


namespace sin_90_degrees_l171_171123

theorem sin_90_degrees : Real.sin (Float.pi / 2) = 1 :=
by
  sorry

end sin_90_degrees_l171_171123


namespace rectangle_area_given_diagonal_l171_171887

noncomputable def area_of_rectangle (x : ℝ) : ℝ :=
  1250 - x^2 / 2

theorem rectangle_area_given_diagonal (P : ℝ) (x : ℝ) (A : ℝ) :
  P = 100 → x^2 = (P / 2)^2 - 2 * A → A = area_of_rectangle x :=
by
  intros hP hx
  sorry

end rectangle_area_given_diagonal_l171_171887


namespace compute_ab_val_l171_171682

variables (a b : ℝ)

theorem compute_ab_val
  (h1 : b^2 - a^2 = 25)
  (h2 : a^2 + b^2 = 64) :
  |a * b| = Real.sqrt 868.5 :=
sorry

end compute_ab_val_l171_171682


namespace probability_of_cube_divides_product_l171_171067

open Finset

noncomputable def cube_divides_probability : ℚ :=
let S : Finset ℕ := {2, 3, 4, 6, 8, 9} in
let comb : Finset (Finset ℕ) := S.powerset.filter (λ s, s.card = 3) in
let favorable : Finset (Finset ℕ) := comb.filter (λ s, let l := s.to_list.sorted (· ≤ ·) in (l.nth_le 0 (by simp [list.sorted_nth_le])) ^ 3 ∣ (l.nth_le 1 (by simp [list.sorted_nth_le])) * (l.nth_le 2 (by simp [list.sorted_nth_le]))) in
(favorable.card : ℚ) / (comb.card : ℚ)

theorem probability_of_cube_divides_product (S : Finset ℕ) (hS : S = {2, 3, 4, 6, 8, 9}) :
  cube_divides_probability = 1 / 5 := by
  rw [cube_divides_probability, hS]
  sorry

end probability_of_cube_divides_product_l171_171067


namespace converse_of_statement_l171_171310

variables (a b : ℝ)

theorem converse_of_statement :
  (a + b ≤ 2) → (a ≤ 1 ∨ b ≤ 1) :=
by
  sorry

end converse_of_statement_l171_171310


namespace single_colony_habitat_limit_reach_time_l171_171723

noncomputable def doubling_time (n : ℕ) : ℕ := 2^n

theorem single_colony_habitat_limit_reach_time :
  ∀ (S : ℕ), ∀ (n : ℕ), doubling_time (n + 1) = S → doubling_time (2 * (n - 1)) = S → n + 1 = 16 :=
by
  intros S n H1 H2
  sorry

end single_colony_habitat_limit_reach_time_l171_171723


namespace internal_common_tangent_bisects_arc_l171_171873

-- Define the centers and point of tangencies
variables {A B C : Point} -- Centers of circles

-- Circles defined with respective centers and radii
variables {r1 r2 r3 : ℝ} -- Radii of the circles
variables {K1 : Circle} (hK1 : K1.center = A ∧ K1.radius = r1)
variables {K2 : Circle} (hK2 : K2.center = B ∧ K2.radius = r2)
variables {K3 : Circle} (hK3 : K3.center = C ∧ K3.radius = r3)

-- Points of tangencies and the external common tangent
variables {T P Q : Point} -- Points of tangency and intersection points
variables (HT : tangent_point K1 K2 = T)
variables (HPQ : external_common_tangent K1 K2 K3 = (P, Q))

-- The proof goal: The internal common tangent bisects the arc PQ closer to T
theorem internal_common_tangent_bisects_arc 
  (hT : externally_tangent K1 K2)
  (hPQRST : common_tangent_meeting_points K1 K2 K3 = (P, Q)) 
  (h_arc_condition : closer_arc_bisected_by_internal_tangent K1 K2 K3 PQ T) :
  bisects_internal_common_tangent K1 K2 K3 PQ T :=
sorry

end internal_common_tangent_bisects_arc_l171_171873


namespace fraction_divisible_by_n_l171_171649

theorem fraction_divisible_by_n (a b n : ℕ) (h1 : a ≠ b) (h2 : n > 0) (h3 : n ∣ (a^n - b^n)) : n ∣ ((a^n - b^n) / (a - b)) :=
by
  sorry

end fraction_divisible_by_n_l171_171649


namespace bonnets_per_orphanage_l171_171036

theorem bonnets_per_orphanage :
  let monday_bonnets := 10
  let tuesday_wednesday_bonnets := 2 * monday_bonnets
  let thursday_bonnets := monday_bonnets + 5
  let friday_bonnets := thursday_bonnets - 5
  let total_bonnets := monday_bonnets + tuesday_wednesday_bonnets + thursday_bonnets + friday_bonnets
  let orphanages := 5
  total_bonnets / orphanages = 11 :=
by
  let monday_bonnets := 10
  let tuesday_wednesday_bonnets := 2 * monday_bonnets
  let thursday_bonnets := monday_bonnets + 5
  let friday_bonnets := thursday_bonnets - 5
  let total_bonnets := monday_bonnets + tuesday_wednesday_bonnets + thursday_bonnets + friday_bonnets
  let orphanages := 5
  sorry

end bonnets_per_orphanage_l171_171036


namespace first_group_work_done_l171_171096

-- Define work amounts with the conditions given
variable (W : ℕ) -- amount of work 3 people can do in 3 days
variable (work_rate : ℕ → ℕ → ℕ) -- work_rate(p, d) is work done by p people in d days

-- Conditions
axiom cond1 : work_rate 3 3 = W
axiom cond2 : work_rate 6 3 = 6 * W

-- The proof statement
theorem first_group_work_done : work_rate 3 3 = 2 * W :=
by
  sorry

end first_group_work_done_l171_171096


namespace min_max_sums_l171_171530

theorem min_max_sums (a b c d e f g : ℝ) 
    (h0 : 0 ≤ a) (h1 : 0 ≤ b) (h2 : 0 ≤ c)
    (h3 : 0 ≤ d) (h4 : 0 ≤ e) (h5 : 0 ≤ f) 
    (h6 : 0 ≤ g) (h_sum : a + b + c + d + e + f + g = 1) :
    (min (max (a + b + c) 
              (max (b + c + d) 
                   (max (c + d + e) 
                        (max (d + e + f) 
                             (e + f + g))))) = 1 / 3) :=
sorry

end min_max_sums_l171_171530


namespace ratio_of_luxury_to_suv_l171_171917

variable (E L S : Nat)

-- Conditions
def condition1 := E * 2 = L * 3
def condition2 := E * 1 = S * 4

-- The statement to prove
theorem ratio_of_luxury_to_suv 
  (h1 : condition1 E L)
  (h2 : condition2 E S) :
  L * 3 = S * 8 :=
by sorry

end ratio_of_luxury_to_suv_l171_171917


namespace factorization_cd_c_l171_171374

theorem factorization_cd_c (C D : ℤ) (h : ∀ y : ℤ, 20*y^2 - 117*y + 72 = (C*y - 8) * (D*y - 9)) : C * D + C = 25 :=
sorry

end factorization_cd_c_l171_171374


namespace price_per_ton_max_tons_l171_171332

variable (x y m : ℝ)

def conditions := x = y + 100 ∧ 2 * x + y = 1700

theorem price_per_ton (h : conditions x y) : x = 600 ∧ y = 500 :=
  sorry

def budget_conditions := 10 * (600 - 100) + 1 * 500 ≤ 5600

theorem max_tons (h : budget_conditions) : 600 * m + 500 * (10 - m) ≤ 5600 → m ≤ 6 :=
  sorry

end price_per_ton_max_tons_l171_171332


namespace smallest_n_for_factorization_l171_171967

theorem smallest_n_for_factorization :
  ∃ n : ℤ, (∀ A B : ℤ, A * B = 60 ↔ n = 5 * B + A) ∧ n = 56 :=
by
  sorry

end smallest_n_for_factorization_l171_171967


namespace vector_dot_cross_product_l171_171224

open Matrix


def a : Fin 3 → ℝ := ![2, -4, 1]
def b : Fin 3 → ℝ := ![3, 0, 2]
def c : Fin 3 → ℝ := ![-1, 3, 2]
def d : Fin 3 → ℝ := ![4, -1, 0]

theorem vector_dot_cross_product :
  let ab := ∀ i, a i - b i
  let bc := ∀ i, b i - c i
  let cd := ∀ i, c i - d i
  (ab 0, ab 1, ab 2) • ![bc 0, bc 1, bc 2] ⨯ ![cd 0, cd 1, cd 2] = 45 :=
by
  -- Proof goes here
  sorry

end vector_dot_cross_product_l171_171224


namespace smallest_N_l171_171815

theorem smallest_N (N : ℕ) (hN : N > 70) (hdv : 70 ∣ 21 * N) : N = 80 :=
sorry

end smallest_N_l171_171815


namespace regular_octagon_area_l171_171937

-- Define the problem conditions
def inscribed_circle_radius : ℝ := 3
def central_angle : ℝ := 360 / 8
def side_length (r : ℝ) : ℝ := 2 * r * sin (central_angle / 2 * real.pi / 180)

-- State the problem to be proven
theorem regular_octagon_area (r : ℝ) (h : r = inscribed_circle_radius) :
  8 * (1/2 * (side_length r) * r * sin (central_angle / 2 * real.pi / 180)) = 18 * real.sqrt(3) * (2 - real.sqrt(2)) :=
sorry

end regular_octagon_area_l171_171937


namespace evaluate_expression_l171_171160

noncomputable def math_expr (x c : ℝ) : ℝ := (x^2 + c)^2 - (x^2 - c)^2

theorem evaluate_expression (x : ℝ) (c : ℝ) (hc : 0 < c) : 
  math_expr x c = 4 * x^2 * c :=
by sorry

end evaluate_expression_l171_171160


namespace congruence_solutions_count_number_of_solutions_l171_171464

theorem congruence_solutions_count (x : ℕ) (hx_pos : x > 0) (hx_lt : x < 200) :
  (x + 17) % 52 = 75 % 52 ↔ x = 6 ∨ x = 58 ∨ x = 110 ∨ x = 162 :=
by sorry

theorem number_of_solutions :
  (∃ x : ℕ, (0 < x ∧ x < 200 ∧ (x + 17) % 52 = 75 % 52)) ∧
  (∃ x1 x2 x3 x4 : ℕ, x1 = 6 ∧ x2 = 58 ∧ x3 = 110 ∧ x4 = 162) ∧
  4 = 4 :=
by sorry

end congruence_solutions_count_number_of_solutions_l171_171464


namespace hundreds_digit_25fac_minus_20fac_zero_l171_171549

theorem hundreds_digit_25fac_minus_20fac_zero :
  ((25! - 20!) % 1000) / 100 % 10 = 0 := by
  sorry

end hundreds_digit_25fac_minus_20fac_zero_l171_171549


namespace minimum_value_at_zero_l171_171469

noncomputable def f (x : ℝ) : ℝ := (x - 1) * Real.exp (x - 1)

theorem minimum_value_at_zero : ∀ x : ℝ, f 0 ≤ f x :=
by
  sorry

end minimum_value_at_zero_l171_171469


namespace joe_initial_tests_l171_171652

theorem joe_initial_tests (S n : ℕ) (h1 : S = 60 * n) (h2 : (S - 45) = 65 * (n - 1)) : n = 4 :=
by {
  sorry
}

end joe_initial_tests_l171_171652


namespace segments_before_returning_to_A_l171_171357

-- Definitions based on given conditions
def concentric_circles := Type -- Placeholder for circles definition
def angle_ABC := 60 -- angle value 60 degrees
def minor_arc_AC := 120 -- angle formed by minor arc AC

-- Problem translated into Lean
theorem segments_before_returning_to_A (n m: ℕ) (h1: angle_ABC = 60)
(h2: minor_arc_AC = 2 * angle_ABC) 
(h3: ∀ i, i < n → (120 * i = 360 * (m + i))): 
  n = 3 := 
by
sorным-polyveryt-очовторенioodingsAdding sorry as we are not required to prove the statement, just to write it.
sorry

end segments_before_returning_to_A_l171_171357


namespace sum_of_integers_is_18_l171_171868

theorem sum_of_integers_is_18 (a b c d : ℕ) 
  (h1 : a * b + c * d = 38)
  (h2 : a * c + b * d = 34)
  (h3 : a * d + b * c = 43) : 
  a + b + c + d = 18 := 
  sorry

end sum_of_integers_is_18_l171_171868


namespace probability_triangle_or_circle_l171_171344

theorem probability_triangle_or_circle (total_figures triangles circles : ℕ) 
  (h1 : total_figures = 10) 
  (h2 : triangles = 4) 
  (h3 : circles = 3) : 
  (triangles + circles) / total_figures = 7 / 10 :=
by
  sorry

end probability_triangle_or_circle_l171_171344


namespace equivalence_l171_171478

theorem equivalence (a b c : ℝ) (h : a + c = 2 * b) : a^2 + 8 * b * c = (2 * b + c)^2 := 
by 
  sorry

end equivalence_l171_171478


namespace sum_of_b_values_l171_171608

theorem sum_of_b_values :
  let discriminant (b : ℝ) := (b + 6) ^ 2 - 4 * 3 * 12
  ∃ b1 b2 : ℝ, discriminant b1 = 0 ∧ discriminant b2 = 0 ∧ b1 + b2 = -12 :=
by sorry

end sum_of_b_values_l171_171608


namespace sin_sum_triangle_l171_171623

theorem sin_sum_triangle (α β γ : ℝ) (h : α + β + γ = Real.pi) : 
  Real.sin α + Real.sin β + Real.sin γ ≤ (3 * Real.sqrt 3) / 2 :=
sorry

end sin_sum_triangle_l171_171623


namespace system_of_equations_solution_l171_171672

variable {x y : ℝ}

theorem system_of_equations_solution
  (h1 : x^2 + x * y * Real.sqrt (x * y) + y^2 = 25)
  (h2 : x^2 - x * y * Real.sqrt (x * y) + y^2 = 9) :
  (x, y) = (1, 4) ∨ (x, y) = (4, 1) ∨ (x, y) = (-1, -4) ∨ (x, y) = (-4, -1) :=
by
  sorry

end system_of_equations_solution_l171_171672


namespace paul_homework_average_l171_171756

def hoursOnWeeknights : ℕ := 2 * 5
def hoursOnWeekend : ℕ := 5
def totalHomework : ℕ := hoursOnWeeknights + hoursOnWeekend
def practiceNights : ℕ := 2
def daysAvailable : ℕ := 7 - practiceNights
def averageHomeworkPerNight : ℕ := totalHomework / daysAvailable

theorem paul_homework_average :
  averageHomeworkPerNight = 3 := 
by
  -- sorry because we skip the proof
  sorry

end paul_homework_average_l171_171756


namespace minimum_value_sqrt_m2_n2_l171_171496

theorem minimum_value_sqrt_m2_n2 
  (a b m n : ℝ)
  (h1 : a^2 + b^2 = 3)
  (h2 : m*a + n*b = 3) : 
  ∃ (k : ℝ), k = Real.sqrt 3 ∧ Real.sqrt (m^2 + n^2) = k :=
by
  sorry

end minimum_value_sqrt_m2_n2_l171_171496


namespace cube_inequality_sufficient_and_necessary_l171_171691

theorem cube_inequality_sufficient_and_necessary (a b : ℝ) :
  (a > b ↔ a^3 > b^3) := 
sorry

end cube_inequality_sufficient_and_necessary_l171_171691


namespace simplify_expression_l171_171951

theorem simplify_expression (a b : ℚ) : (14 * a^3 * b^2 - 7 * a * b^2) / (7 * a * b^2) = 2 * a^2 - 1 := 
by 
  sorry

end simplify_expression_l171_171951


namespace eval_expression_l171_171444

theorem eval_expression :
  72 + (120 / 15) + (18 * 19) - 250 - (360 / 6) = 112 :=
by sorry

end eval_expression_l171_171444


namespace eval_expr_at_x_eq_neg6_l171_171669

-- Define the given condition
def x : ℤ := -4

-- Define the expression to be simplified and evaluated
def expr (x y : ℤ) : ℤ := ((x + y)^2 - y * (2 * x + y) - 8 * x) / (2 * x)

-- The theorem stating the result of the evaluated expression
theorem eval_expr_at_x_eq_neg6 (y : ℤ) : expr (-4) y = -6 := 
by
  sorry

end eval_expr_at_x_eq_neg6_l171_171669


namespace value_of_a7_l171_171321

-- Define the geometric sequence and its properties
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

-- Define the conditions of the problem
variables (a : ℕ → ℝ) (h_geom : is_geometric_sequence a) (h_pos : ∀ n : ℕ, a n > 0) (h_product : a 3 * a 11 = 16)

-- Conjecture that we aim to prove
theorem value_of_a7 : a 7 = 4 :=
by {
  sorry
}

end value_of_a7_l171_171321


namespace total_time_spent_l171_171381

def chess_game_duration_hours : ℕ := 20
def chess_game_duration_minutes : ℕ := 15
def additional_analysis_time : ℕ := 22
def total_expected_time : ℕ := 1237

theorem total_time_spent : 
  (chess_game_duration_hours * 60 + chess_game_duration_minutes + additional_analysis_time) = total_expected_time :=
  by
    sorry

end total_time_spent_l171_171381


namespace new_area_is_497_l171_171371

noncomputable def rect_area_proof : Prop :=
  ∃ (l w l' w' : ℝ),
    -- initial area condition
    l * w = 540 ∧ 
    -- conditions for new dimensions
    l' = 0.8 * l ∧
    w' = 1.15 * w ∧
    -- final area calculation
    l' * w' = 497

theorem new_area_is_497 : rect_area_proof := by
  sorry

end new_area_is_497_l171_171371


namespace max_principals_in_10_years_l171_171390

theorem max_principals_in_10_years :
  ∀ (term_length : ℕ) (P : ℕ → Prop),
  (∀ n, P n → 3 ≤ n ∧ n ≤ 5) → 
  ∃ (n : ℕ), (n ≤ 10 / 3 ∧ P n) ∧ n = 3 :=
by
  sorry

end max_principals_in_10_years_l171_171390


namespace no_integer_solutions_l171_171965

theorem no_integer_solutions (m n : ℤ) (h1 : m ^ 3 + n ^ 4 + 130 * m * n = 42875) (h2 : m * n ≥ 0) :
  false :=
sorry

end no_integer_solutions_l171_171965


namespace g_10_equals_100_l171_171154

-- Define the function g and the conditions it must satisfy.
def g : ℕ → ℝ := sorry

axiom g_2 : g 2 = 4

axiom g_condition : ∀ m n : ℕ, m ≥ n → g (m + n) + g (m - n) = (g (2 * m) + g (2 * n)) / 2

-- Prove the required statement.
theorem g_10_equals_100 : g 10 = 100 :=
by sorry

end g_10_equals_100_l171_171154


namespace monotonic_f_on_interval_l171_171003

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (x + Real.pi / 10) - 2

theorem monotonic_f_on_interval : 
  ∀ x y : ℝ, 
    x ∈ Set.Icc (Real.pi / 2) (7 * Real.pi / 5) → 
    y ∈ Set.Icc (Real.pi / 2) (7 * Real.pi / 5) → 
    x ≤ y → 
    f x ≤ f y :=
sorry

end monotonic_f_on_interval_l171_171003


namespace trailing_zeroes_in_500_factorial_l171_171948

theorem trailing_zeroes_in_500_factorial : ∀ n = 500, (∑ k in range (nat.log 5 500 + 1), 500 / 5^k) = 124 :=
by
  sorry

end trailing_zeroes_in_500_factorial_l171_171948


namespace triangle_inequality_third_side_l171_171776

theorem triangle_inequality_third_side (x : ℝ) (h1 : 5 > 0) (h2 : 8 > 0) :
  3 < x ∧ x < 13 ↔ (5 + 8 > x) ∧ (5 + x > 8) ∧ (8 + x > 5) :=
by 
  -- Placeholder for proof
  sorry

end triangle_inequality_third_side_l171_171776


namespace tina_husband_brownies_days_l171_171393

variable (d : Nat)

theorem tina_husband_brownies_days : 
  (exists (d : Nat), 
    let total_brownies := 24
    let tina_daily := 2
    let husband_daily := 1
    let total_daily := tina_daily + husband_daily
    let shared_with_guests := 4
    let remaining_brownies := total_brownies - shared_with_guests
    let final_leftover := 5
    let brownies_eaten := remaining_brownies - final_leftover
    brownies_eaten = d * total_daily) → d = 5 := 
by
  sorry

end tina_husband_brownies_days_l171_171393


namespace man_born_year_l171_171419

theorem man_born_year (x : ℕ) : 
  (x^2 - x = 1806) ∧ (x^2 - x < 1850) ∧ (40 < x) ∧ (x < 50) → x = 43 :=
by
  sorry

end man_born_year_l171_171419


namespace guess_x_30_guess_y_127_l171_171316

theorem guess_x_30 : 120 = 4 * 30 := 
  sorry

theorem guess_y_127 : 87 = 127 - 40 := 
  sorry

end guess_x_30_guess_y_127_l171_171316


namespace fourth_number_is_two_eighth_number_is_two_l171_171260

theorem fourth_number_is_two
  (notebook : List ℚ)
  (h_notebook : notebook = [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5]) :
  ∃ (board : List ℚ), board.length ≥ 4 ∧ board !! 3 = some 2 :=
by
  sorry

theorem eighth_number_is_two
  (notebook : List ℚ)
  (h_notebook : notebook = [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5]) :
  ∃ (board : List ℚ), board.length ≥ 8 ∧ board !! 7 = some 2 :=
by
  sorry

end fourth_number_is_two_eighth_number_is_two_l171_171260


namespace sum_mnp_l171_171750

noncomputable def volume_of_parallelepiped := 2 * 3 * 4
noncomputable def volume_of_extended_parallelepipeds := 
  2 * (1 * 2 * 3 + 1 * 2 * 4 + 1 * 3 * 4)
noncomputable def volume_of_quarter_cylinders := 
  4 * (1 / 4 * Real.pi * 1^2 * (2 + 3 + 4))
noncomputable def volume_of_spherical_octants := 
  8 * (1 / 8 * (4 / 3) * Real.pi * 1^3)

noncomputable def total_volume := 
  volume_of_parallelepiped + volume_of_extended_parallelepipeds + 
  volume_of_quarter_cylinders + volume_of_spherical_octants

theorem sum_mnp : 228 + 85 + 3 = 316 := by
  sorry

end sum_mnp_l171_171750


namespace smallest_possible_value_of_N_l171_171809

theorem smallest_possible_value_of_N :
  ∃ N : ℕ, N > 70 ∧ (21 * N) % 70 = 0 ∧ N = 80 :=
by
  existsi 80
  split
  · norm_num
  split
  · norm_num
  sorry

end smallest_possible_value_of_N_l171_171809


namespace geometric_sequence_ratio_l171_171019

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 2 + a 8 = 15) 
  (h2 : a 3 * a 7 = 36) 
  (h_geom : ∀ n, a (n + 1) = a n * q) : 
  (a 19 / a 13 = 4) ∨ (a 19 / a 13 = 1 / 4) :=
by
  sorry

end geometric_sequence_ratio_l171_171019


namespace sin_90_degrees_l171_171124

theorem sin_90_degrees : Real.sin (Float.pi / 2) = 1 :=
by
  sorry

end sin_90_degrees_l171_171124


namespace trajectory_of_M_l171_171774

theorem trajectory_of_M
  (A : ℝ × ℝ := (3, 0))
  (P_circle : ∀ (P : ℝ × ℝ), P.1^2 + P.2^2 = 1)
  (M_midpoint : ∀ (P M : ℝ × ℝ), M = ((P.1 + 3) / 2, P.2 / 2) → M.1 = x ∧ M.2 = y) :
  (∀ (x y : ℝ), (x - 3/2)^2 + y^2 = 1/4) := 
sorry

end trajectory_of_M_l171_171774


namespace complete_the_square_l171_171904

theorem complete_the_square (x : ℝ) : (x^2 - 6 * x + 8 = 0) -> ((x - 3)^2 = 1) :=
by
  intro h
  sorry

end complete_the_square_l171_171904


namespace union_of_sets_l171_171790

open Set

variable (a : ℤ)

def setA : Set ℤ := {1, 3}
def setB (a : ℤ) : Set ℤ := {a + 2, 5}

theorem union_of_sets (h : {3} = setA ∩ setB a) : setA ∪ setB a = {1, 3, 5} :=
by
  sorry

end union_of_sets_l171_171790


namespace octagon_area_l171_171934

theorem octagon_area 
  (r : ℝ) 
  (h_regular : true) 
  (h_inscribed : true)
  (h_radius : r = 3) : 
  ∃ A, A = 18 * real.sqrt 2 :=
by
  sorry

end octagon_area_l171_171934


namespace sum_of_four_numbers_eq_zero_l171_171459

theorem sum_of_four_numbers_eq_zero
  (x y s t : ℝ)
  (h₀ : x ≠ y)
  (h₁ : x ≠ s)
  (h₂ : x ≠ t)
  (h₃ : y ≠ s)
  (h₄ : y ≠ t)
  (h₅ : s ≠ t)
  (h_eq : (x + s) / (x + t) = (y + t) / (y + s)) :
  x + y + s + t = 0 := by
sorry

end sum_of_four_numbers_eq_zero_l171_171459


namespace least_number_to_add_to_4499_is_1_l171_171564

theorem least_number_to_add_to_4499_is_1 (x : ℕ) : (4499 + x) % 9 = 0 → x = 1 := sorry

end least_number_to_add_to_4499_is_1_l171_171564


namespace largest_multiple_of_8_less_than_neg_63_l171_171069

theorem largest_multiple_of_8_less_than_neg_63 : 
  ∃ n : ℤ, (n < -63) ∧ (∃ k : ℤ, n = 8 * k) ∧ (∀ m : ℤ, (m < -63) ∧ (∃ l : ℤ, m = 8 * l) → m ≤ n) :=
sorry

end largest_multiple_of_8_less_than_neg_63_l171_171069


namespace inequality_solution_l171_171199

theorem inequality_solution (x y : ℝ) (h : 5 * x > -5 * y) : x + y > 0 :=
sorry

end inequality_solution_l171_171199


namespace CarriageSharingEquation_l171_171441

theorem CarriageSharingEquation (x : ℕ) :
  (x / 3 + 2 = (x - 9) / 2) ↔
  (3 * ((x - 9) / 2) + 2 * 3 = x / 3 + 2) ∧ 
  (2 * ((x - 9) / 2) + 9 = x ∨ 2 * ((x - 9) / 2) + 9 < x) ∧ 
  (x / 3 + 2 < 3 * (x / 2) + 2 * 2 ∨ x / 3 + 2 = 3 * (x / 2) + 2 * 2) :=
sorry

end CarriageSharingEquation_l171_171441


namespace time_left_after_council_room_is_zero_l171_171037

-- Define the conditions
def totalTimeAllowed : ℕ := 30
def travelToSchoolTime : ℕ := 25
def walkToLibraryTime : ℕ := 3
def returnBooksTime : ℕ := 4
def walkToCouncilRoomTime : ℕ := 5
def submitProjectTime : ℕ := 3

-- Calculate time spent up to the student council room
def timeSpentUpToCouncilRoom : ℕ :=
  travelToSchoolTime + walkToLibraryTime + returnBooksTime + walkToCouncilRoomTime + submitProjectTime

-- Question: How much time is left after leaving the student council room to reach the classroom without being late?
theorem time_left_after_council_room_is_zero (totalTimeAllowed travelToSchoolTime walkToLibraryTime returnBooksTime walkToCouncilRoomTime submitProjectTime : ℕ):
  totalTimeAllowed - timeSpentUpToCouncilRoom = 0 := by
  sorry

end time_left_after_council_room_is_zero_l171_171037


namespace smallest_N_value_proof_l171_171801

def smallest_value_N (N : ℕ) : Prop :=
  N > 70 ∧ (21 * N) % 70 = 0

theorem smallest_N_value_proof : ∃ N, smallest_value_N N ∧ (∀ M, smallest_value_N M → N ≤ M) :=
  sorry

end smallest_N_value_proof_l171_171801


namespace inequality_holds_l171_171236

theorem inequality_holds (a b : ℝ) : (6 * a - 3 * b - 3) * (a ^ 2 + a ^ 2 * b - 2 * a ^ 3) ≤ 0 :=
sorry

end inequality_holds_l171_171236


namespace sum_perpendiculars_in_regular_hexagon_l171_171566

open EuclideanGeometry

variables {P Q R A B C D E F O : Point}

-- Definition of regular hexagon and its properties
def is_regular_hexagon (A B C D E F : Point) : Prop :=
  is_regular_polygon 6 [A, B, C, D, E, F]

-- Perpendicular drops
def perpendicular_from_point_to_line (A P : Point) (l : Line) : Prop :=
  is_perpendicular (Line.through A P) l

-- Regular Hexagon and perpendicular properties
variables (h1 : is_regular_hexagon A B C D E F)
          (h2 : perpendicular_from_point_to_line A P (Line.through C D))
          (h3 : perpendicular_from_point_to_line A Q (Line.through E F))
          (h4 : perpendicular_from_point_to_line A R (Line.through B C))
          (h5 : center_of A B C D E F = O)
          (h6 : dist O P = 1)

theorem sum_perpendiculars_in_regular_hexagon (h1 h2 h3 h4 h5 h6) : 
  dist A P + dist A Q + dist A R = 3 * Real.sqrt 3 := 
sorry

end sum_perpendiculars_in_regular_hexagon_l171_171566


namespace total_students_l171_171913

theorem total_students (n1 n2 : ℕ) (h1 : (158 - 140)/(n1 + 1) = 2) (h2 : (158 - 140)/(n2 + 1) = 3) :
  n1 + n2 + 2 = 15 :=
sorry

end total_students_l171_171913


namespace ellipse_AB_length_l171_171308

theorem ellipse_AB_length :
  ∀ (F1 F2 A B : ℝ × ℝ) (x y : ℝ),
  (x^2 / 25 + y^2 / 9 = 1) →
  (F1 = (5, 0) ∨ F1 = (-5, 0)) →
  (F2 = (if F1 = (5, 0) then (-5, 0) else (5, 0))) →
  ({p : ℝ × ℝ | p.1^2 / 25 + p.2^2 / 9 = 1} A ∨ {p : ℝ × ℝ | p.1^2 / 25 + p.2^2 / 9 = 1} B) →
  ((A = F1) ∨ (B = F1)) →
  (abs (F2.1 - A.1) + abs (F2.2 - A.2) + abs (F2.1 - B.1) + abs (F2.2 - B.2) = 12) →
  abs (A.1 - B.1) + abs (A.2 - B.2) = 8 :=
by
  sorry

end ellipse_AB_length_l171_171308


namespace sin_ninety_deg_l171_171143

theorem sin_ninety_deg : Real.sin (Float.pi / 2) = 1 := 
by sorry

end sin_ninety_deg_l171_171143


namespace minimum_value_of_f_l171_171203

noncomputable def f (x : ℝ) := 2 * x + 18 / x

theorem minimum_value_of_f :
  ∃ x > 0, f x = 12 ∧ ∀ y > 0, f y ≥ 12 :=
by
  sorry

end minimum_value_of_f_l171_171203


namespace perimeter_right_triangle_l171_171426

-- Given conditions
def area : ℝ := 200
def b : ℝ := 20

-- Mathematical problem
theorem perimeter_right_triangle :
  ∀ (x c : ℝ), 
  (1 / 2) * b * x = area →
  c^2 = x^2 + b^2 →
  x + b + c = 40 + 20 * Real.sqrt 2 := 
  by
  sorry

end perimeter_right_triangle_l171_171426


namespace range_x_f_inequality_l171_171980

noncomputable def f (x : ℝ) : ℝ := Real.log (1 - |x|) + 1 / (x^2 + 1)

theorem range_x_f_inequality :
  (∀ x : ℝ, f (2 * x + 1) ≥ f x) ↔ x ∈ Set.Icc (-1 : ℝ) (-1 / 3) := sorry

end range_x_f_inequality_l171_171980


namespace smallest_possible_N_l171_171798

theorem smallest_possible_N (N : ℕ) (h1 : N > 70) (h2 : 70 ∣ 21 * N) : N = 80 :=
sorry

end smallest_possible_N_l171_171798


namespace find_a1_l171_171854

theorem find_a1 (f : ℝ → ℝ) (a : ℕ → ℝ) (h₀ : ∀ x, f x = (x - 1)^3 + x + 2)
(h₁ : ∀ n, a (n + 1) = a n + 1/2)
(h₂ : f (a 1) + f (a 2) + f (a 3) + f (a 4) + f (a 5) + f (a 6) = 18) :
a 1 = -1 / 4 :=
by
  sorry

end find_a1_l171_171854


namespace steps_to_school_l171_171908

-- Define the conditions as assumptions
def distance : Float := 900
def step_length : Float := 0.45

-- Define the statement to be proven
theorem steps_to_school (x : Float) : step_length * x = distance → x = 2000 := by
  intro h
  sorry

end steps_to_school_l171_171908


namespace big_joe_height_is_8_l171_171111

variable (Pepe_height Frank_height Larry_height Ben_height BigJoe_height : ℝ)

axiom Pepe_height_def : Pepe_height = 4.5
axiom Frank_height_def : Frank_height = Pepe_height + 0.5
axiom Larry_height_def : Larry_height = Frank_height + 1
axiom Ben_height_def : Ben_height = Larry_height + 1
axiom BigJoe_height_def : BigJoe_height = Ben_height + 1

theorem big_joe_height_is_8 :
  BigJoe_height = 8 :=
sorry

end big_joe_height_is_8_l171_171111


namespace partnership_profit_l171_171339

noncomputable def total_profit
  (P : ℝ)
  (mary_investment : ℝ := 700)
  (harry_investment : ℝ := 300)
  (effort_share := P / 3 / 2)
  (remaining_share := 2 / 3 * P)
  (total_investment := mary_investment + harry_investment)
  (mary_share_remaining := (mary_investment / total_investment) * remaining_share)
  (harry_share_remaining := (harry_investment / total_investment) * remaining_share) : Prop :=
  (effort_share + mary_share_remaining) - (effort_share + harry_share_remaining) = 800

theorem partnership_profit : ∃ P : ℝ, total_profit P ∧ P = 3000 :=
  sorry

end partnership_profit_l171_171339


namespace andrew_brian_ratio_l171_171866

-- Definitions based on conditions extracted from the problem
variables (A S B : ℕ)

-- Conditions
def steven_shirts : Prop := S = 72
def brian_shirts : Prop := B = 3
def steven_andrew_relation : Prop := S = 4 * A

-- The goal is to prove the ratio of Andrew's shirts to Brian's shirts is 6
theorem andrew_brian_ratio (A S B : ℕ) 
  (h1 : steven_shirts S) 
  (h2 : brian_shirts B)
  (h3 : steven_andrew_relation A S) :
  A / B = 6 := by
  sorry

end andrew_brian_ratio_l171_171866


namespace sin_90_eq_one_l171_171120

noncomputable theory
open Real

/--
The sine of an angle in the unit circle is the y-coordinate of the point at that angle from the positive x-axis.
Rotating the point (1,0) by 90 degrees counterclockwise about the origin results in the point (0,1).
Prove that \(\sin 90^\circ = 1\).
-/
theorem sin_90_eq_one : sin (90 * (real.pi / 180)) = 1 :=
by
  -- Definitions and conditions for the unit circle and sine function
  let angle := 90 * (real.pi / 180)
  have h1 : (cos angle, sin angle) = (0, 1),
  { sorry },
  -- Desired conclusion
  exact h1.2

end sin_90_eq_one_l171_171120


namespace trajectory_of_moving_circle_l171_171198

def circle1 (x y : ℝ) := (x + 4) ^ 2 + y ^ 2 = 2
def circle2 (x y : ℝ) := (x - 4) ^ 2 + y ^ 2 = 2

theorem trajectory_of_moving_circle (x y : ℝ) : 
  (x = 0) ∨ (x ^ 2 / 2 - y ^ 2 / 14 = 1) := 
  sorry

end trajectory_of_moving_circle_l171_171198


namespace cube_rolling_impossible_l171_171416

-- Definitions
def paintedCube : Type := sorry   -- Define a painted black-and-white cube.
def chessboard : Type := sorry    -- Define the chessboard.
def roll (c : paintedCube) (b : chessboard) : Prop := sorry   -- Define the rolling over the board visiting each square exactly once.
def matchColors (c : paintedCube) (b : chessboard) : Prop := sorry   -- Define the condition that colors match on contact.

-- Theorem
theorem cube_rolling_impossible (c : paintedCube) (b : chessboard)
  (h1 : roll c b) : ¬ matchColors c b := sorry

end cube_rolling_impossible_l171_171416


namespace value_of_a1_a3_a5_l171_171472

theorem value_of_a1_a3_a5 (a a1 a2 a3 a4 a5 : ℤ) (h : (2 * x + 1) ^ 5 = a + a1 * x + a2 * x ^ 2 + a3 * x ^ 3 + a4 * x ^ 4 + a5 * x ^ 5) :
  a1 + a3 + a5 = 122 :=
by
  sorry

end value_of_a1_a3_a5_l171_171472


namespace smallest_N_l171_171816

theorem smallest_N (N : ℕ) (hN : N > 70) (hdv : 70 ∣ 21 * N) : N = 80 :=
sorry

end smallest_N_l171_171816


namespace pine_saplings_in_sample_l171_171921

-- Definitions based on conditions
def total_saplings : ℕ := 30000
def pine_saplings : ℕ := 4000
def sample_size : ℕ := 150

-- Main theorem to prove
theorem pine_saplings_in_sample : (pine_saplings * sample_size) / total_saplings = 20 :=
by sorry

end pine_saplings_in_sample_l171_171921


namespace valid_third_side_length_l171_171778

theorem valid_third_side_length {x : ℝ} (h1 : 5 + 8 > x) (h2 : 5 + x > 8) (h3 : 8 + x > 5) : x = 6 :=
by
  -- Given 5 + 8 > x, 5 + x > 8, 8 + x > 5
  have range1 : 13 > x := h1,
  have range2 : x > 3 := (by linarith [h2]),
  have _ : 3 < 6 ∧ 6 < 13 := by norm_num,
  linarith

#check valid_third_side_length

end valid_third_side_length_l171_171778


namespace possible_third_side_l171_171785

theorem possible_third_side {x : ℕ} (h_option_A : x = 2) (h_option_B : x = 3) (h_option_C : x = 6) (h_option_D : x = 13) : 3 < x ∧ x < 13 ↔ x = 6 :=
by
  sorry

end possible_third_side_l171_171785


namespace circles_do_not_intersect_first_scenario_circles_do_not_intersect_second_scenario_l171_171543

-- Define radii of the circles
def r1 : ℝ := 3
def r2 : ℝ := 5

-- Statement for first scenario (distance = 9)
theorem circles_do_not_intersect_first_scenario (d : ℝ) (h : d = 9) : ¬ (|r1 - r2| ≤ d ∧ d ≤ r1 + r2) :=
by sorry

-- Statement for second scenario (distance = 1)
theorem circles_do_not_intersect_second_scenario (d : ℝ) (h : d = 1) : d < |r1 - r2| ∨ ¬ (|r1 - r2| ≤ d ∧ d ≤ r1 + r2) :=
by sorry

end circles_do_not_intersect_first_scenario_circles_do_not_intersect_second_scenario_l171_171543


namespace volume_difference_l171_171043

def sara_dimensions : (ℤ × ℤ × ℤ) := (1 * 12, 2 * 12, 2 * 12) -- dimensions in inches
def jake_dimensions : (ℤ × ℤ × ℤ) := (16, 20, 18) -- dimensions already in inches

def volume (dims : (ℤ × ℤ × ℤ)) : ℤ :=
  dims.1 * dims.2 * dims.3

theorem volume_difference :
  volume sara_dimensions - volume jake_dimensions = 1152 :=
by
  sorry

end volume_difference_l171_171043


namespace product_of_primes_l171_171702

theorem product_of_primes :
  (7 * 97 * 89) = 60431 :=
by
  sorry

end product_of_primes_l171_171702


namespace fewest_presses_to_original_l171_171473

theorem fewest_presses_to_original (x : ℝ) (hx : x = 16) (f : ℝ → ℝ)
    (hf : ∀ y : ℝ, f y = 1 / y) : (f (f x)) = x :=
by
  sorry

end fewest_presses_to_original_l171_171473


namespace new_area_of_rectangle_l171_171369

theorem new_area_of_rectangle (L W : ℝ) (h : L * W = 540) : 
  let L' := 0.8 * L,
      W' := 1.15 * W,
      A' := L' * W' in
  A' = 497 := 
by
  sorry

end new_area_of_rectangle_l171_171369


namespace common_root_of_two_equations_l171_171172

theorem common_root_of_two_equations (m x : ℝ) :
  (m * x - 1000 = 1001) ∧ (1001 * x = m - 1000 * x) → (m = 2001 ∨ m = -2001) :=
by
  sorry

end common_root_of_two_equations_l171_171172


namespace area_DEFG_l171_171103

-- Define points and the properties of the rectangle ABCD
variable (A B C D E G F : Type)
variables (area_ABCD : ℝ) (Eg_parallel_AB_CD Df_parallel_AD_BC : Prop)
variable (E_position_AD : ℝ) (G_position_CD : ℝ) (F_midpoint_BC : Prop)
variables (length_abcd width_abcd : ℝ)

-- Assumptions based on given conditions
axiom h1 : area_ABCD = 150
axiom h2 : E_position_AD = 1 / 3
axiom h3 : G_position_CD = 1 / 3
axiom h4 : Eg_parallel_AB_CD
axiom h5 : Df_parallel_AD_BC
axiom h6 : F_midpoint_BC

-- Theorem to prove the area of DEFG
theorem area_DEFG : length_abcd * width_abcd / 3 = 50 :=
    sorry

end area_DEFG_l171_171103


namespace trajectory_of_M_lines_perpendicular_l171_171847

-- Define the given conditions
def parabola (P : ℝ × ℝ) : Prop :=
  P.1 ^ 2 = P.2

def midpoint_condition (P M : ℝ × ℝ) : Prop :=
  P.1 = 1/2 * M.1 ∧ P.2 = M.2

def trajectory_condition (M : ℝ × ℝ) : Prop :=
  M.1 ^ 2 = 4 * M.2

theorem trajectory_of_M (P M : ℝ × ℝ) (H1 : parabola P) (H2 : midpoint_condition P M) : 
  trajectory_condition M :=
sorry

-- Define the conditions for the second part
def line_through_F (A B : ℝ × ℝ) (F : ℝ × ℝ): Prop :=
  ∃ k : ℝ, A.2 = k * A.1 + F.2 ∧ B.2 = k * B.1 + F.2

def perpendicular_feet (A B A1 B1 : ℝ × ℝ) : Prop :=
  A1 = (A.1, -1) ∧ B1 = (B.1, -1)

def perpendicular_lines (A1 B1 F : ℝ × ℝ) : Prop :=
  let v1 := (-A1.1, F.2 - A1.2)
  let v2 := (-B1.1, F.2 - B1.2)
  v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem lines_perpendicular (A B A1 B1 F : ℝ × ℝ) (H1 : trajectory_condition A) (H2 : trajectory_condition B) 
(H3 : line_through_F A B F) (H4 : perpendicular_feet A B A1 B1) :
  perpendicular_lines A1 B1 F :=
sorry

end trajectory_of_M_lines_perpendicular_l171_171847


namespace sin_90_degree_l171_171138

-- Definitions based on conditions
def unit_circle_point (angle : ℝ) : ℝ × ℝ :=
  if angle = 90 * (π / 180) then (0, 1) else sorry

def sin_usual (angle : ℝ) : ℝ :=
  (unit_circle_point angle).snd

-- The main theorem as per the question and conditions
theorem sin_90_degree : sin_usual (90 * (π / 180)) = 1 :=
by
  sorry

end sin_90_degree_l171_171138


namespace regular_octagon_area_l171_171940

-- Definitions based on conditions
def is_regular_octagon (p : ℝ → Prop) : Prop := 
  ∀ θ, 0 ≤ θ ∧ θ < 7 * (π / 4) → p θ = p (θ + π / 4)

def inscribed_in_circle (p : ℝ → Prop) (r : ℝ) : Prop :=
  ∀ θ, 0 ≤ θ ∧ θ < 2 * π → p θ = r

-- The proof statement
theorem regular_octagon_area 
  (r : ℝ) (h_r : r = 3) 
  (p : ℝ → Prop)
  (h_regular : is_regular_octagon p)
  (h_inscribed : inscribed_in_circle p r) :
  ∃ a : ℝ, a = 14.92 := 
sorry

end regular_octagon_area_l171_171940


namespace face_value_of_shares_l171_171414

/-- A company pays a 12.5% dividend to its investors. -/
def div_rate := 0.125

/-- An investor gets a 25% return on their investment. -/
def roi_rate := 0.25

/-- The investor bought the shares at Rs. 20 each. -/
def purchase_price := 20

theorem face_value_of_shares (FV : ℝ) (div_rate : ℝ) (roi_rate : ℝ) (purchase_price : ℝ) 
  (h1 : purchase_price * roi_rate = div_rate * FV) : FV = 40 :=
by sorry

end face_value_of_shares_l171_171414


namespace simplified_expression_value_at_4_l171_171551

theorem simplified_expression (x : ℝ) (h : x ≠ 5) : (x^2 - 3*x - 10) / (x - 5) = x + 2 := 
sorry

theorem value_at_4 : (4 : ℝ)^2 - 3*4 - 10 / (4 - 5) = 6 := 
sorry

end simplified_expression_value_at_4_l171_171551


namespace agatha_amount_left_l171_171587

noncomputable def initial_amount : ℝ := 60
noncomputable def frame_cost : ℝ := 15 * (1 - 0.10)
noncomputable def wheel_cost : ℝ := 25 * (1 - 0.05)
noncomputable def seat_cost : ℝ := 8 * (1 - 0.15)
noncomputable def handlebar_tape_cost : ℝ := 5
noncomputable def bell_cost : ℝ := 3
noncomputable def hat_cost : ℝ := 10 * (1 - 0.25)

noncomputable def total_cost : ℝ :=
  frame_cost + wheel_cost + seat_cost + handlebar_tape_cost + bell_cost + hat_cost

noncomputable def amount_left : ℝ := initial_amount - total_cost

theorem agatha_amount_left : amount_left = 0.45 :=
by
  -- interim calculations would go here
  sorry

end agatha_amount_left_l171_171587


namespace trig_identity_proof_l171_171255

theorem trig_identity_proof :
  let sin240 := - (Real.sin (120 * Real.pi / 180))
  let tan240 := Real.tan (240 * Real.pi / 180)
  Real.sin (600 * Real.pi / 180) + tan240 = Real.sqrt 3 / 2 :=
by
  sorry

end trig_identity_proof_l171_171255


namespace first_number_is_seven_l171_171692

variable (x y : ℝ)

theorem first_number_is_seven (h1 : x + y = 10) (h2 : 2 * x = 3 * y + 5) : x = 7 :=
sorry

end first_number_is_seven_l171_171692


namespace amount_given_to_beggar_l171_171918

variable (X : ℕ)
variable (pennies_initial : ℕ := 42)
variable (pennies_to_farmer : ℕ := 22)
variable (pennies_after_farmer : ℕ := 20)

def amount_to_boy (X : ℕ) : ℕ :=
  (20 - X) / 2 + 3

theorem amount_given_to_beggar : 
  (X = 12) →  (pennies_initial - pennies_to_farmer - X) / 2 + 3 + 1 = pennies_initial - pennies_to_farmer - X :=
by
  intro h
  subst h
  sorry

end amount_given_to_beggar_l171_171918


namespace max_value_of_g_l171_171451

noncomputable def g (x : ℝ) : ℝ := 4 * x - x^4

theorem max_value_of_g : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 → g x ≤ 3 ∧ (∃ x0, x0 = 1 ∧ g x0 = 3) :=
by
  sorry

end max_value_of_g_l171_171451


namespace cos_2alpha_minus_pi_over_6_l171_171029

theorem cos_2alpha_minus_pi_over_6 (α : ℝ) (hα : 0 < α ∧ α < π / 2) (hSin : Real.sin (α + π / 6) = 3 / 5) :
  Real.cos (2 * α - π / 6) = 24 / 25 :=
sorry

end cos_2alpha_minus_pi_over_6_l171_171029


namespace value_of_S_l171_171509

-- Defining the condition as an assumption
def one_third_one_eighth_S (S : ℝ) : Prop :=
  (1 / 3) * (1 / 8) * S = (1 / 4) * (1 / 6) * 120

-- The statement we need to prove
theorem value_of_S (S : ℝ) (h : one_third_one_eighth_S S) : S = 120 :=
by
  sorry

end value_of_S_l171_171509


namespace prove_p_value_l171_171594

noncomputable def binomial_coefficient (n k: ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Assign the given binomial coefficient
def C_7_4 : ℕ := binomial_coefficient 7 4

-- Define v in terms of binomial distribution
def v (p : ℝ) : ℝ := C_7_4 * p^4 * (1 - p)^3

-- Given condition
axiom v_value : v 0.7 = 343 / 3125

theorem prove_p_value : ∃ p : ℝ, v p = 343 / 3125 ∧ p = 0.7 :=
by
  sorry

end prove_p_value_l171_171594


namespace geometric_sequence_first_term_l171_171764

theorem geometric_sequence_first_term (a r : ℝ) (h1 : a * r^2 = 18) (h2 : a * r^4 = 162) : a = 2 := by
  sorry

end geometric_sequence_first_term_l171_171764


namespace circles_intersect_l171_171788

theorem circles_intersect (R r d: ℝ) (hR: R = 7) (hr: r = 4) (hd: d = 8) : (R - r < d) ∧ (d < R + r) :=
by
  rw [hR, hr, hd]
  exact ⟨by linarith, by linarith⟩

end circles_intersect_l171_171788


namespace sum_terms_sequence_l171_171022

noncomputable def geometric_sequence := ℕ → ℝ

variables (a : geometric_sequence)
variables (r : ℝ) (h_pos : ∀ n, a n > 0)

-- Geometric sequence condition
axiom geom_seq (n : ℕ) : a (n + 1) = a n * r

-- Given condition
axiom h_condition : a 3 * a 5 + a 2 * a 10 + 2 * a 4 * a 6 = 100

-- The goal is to prove that a_4 + a_6 = 10
theorem sum_terms_sequence : a 4 + a 6 = 10 :=
by
  sorry

end sum_terms_sequence_l171_171022


namespace coordinates_with_respect_to_origin_l171_171876

theorem coordinates_with_respect_to_origin (x y : ℤ) (hx : x = 3) (hy : y = -2) : (x, y) = (3, -2) :=
by
  sorry

end coordinates_with_respect_to_origin_l171_171876


namespace aaronFoundCards_l171_171595

-- Given conditions
def initialCardsAaron : ℕ := 5
def finalCardsAaron : ℕ := 67

-- Theorem statement
theorem aaronFoundCards : finalCardsAaron - initialCardsAaron = 62 :=
by
  sorry

end aaronFoundCards_l171_171595


namespace factorial_ends_with_base_8_zeroes_l171_171989

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def highestPowerOfFactorInFactorial (n p : ℕ) : ℕ :=
  if p = 1 then n else
  Nat.div (n - 1) (p - 1)

theorem factorial_ends_with_base_8_zeroes (n : ℕ) : 
  highestPowerOfFactorInFactorial 15 8 = 3 := 
sorry

end factorial_ends_with_base_8_zeroes_l171_171989


namespace spinner_even_product_probability_l171_171893

-- Define the outcomes for the two spinners
def spinner1 := {0, 2}
def spinner2 := {1, 3, 5}

-- Define the event that the product is even
def is_product_even (a b : ℕ) : Prop := (a * b) % 2 = 0

-- Theorem statement
theorem spinner_even_product_probability : 
  ∀ (a ∈ spinner1) (b ∈ spinner2), is_product_even a b :=
by
  sorry

end spinner_even_product_probability_l171_171893


namespace total_people_correct_l171_171298

-- Define the daily changes as given conditions
def daily_changes : List ℝ := [1.6, 0.8, 0.4, -0.4, -0.8, 0.2, -1.2]

-- Define the total number of people given 'a' and daily changes
def total_people (a : ℝ) : ℝ :=
  7 * a + daily_changes.sum

-- Lean statement for proving the total number of people
theorem total_people_correct (a : ℝ) : 
  total_people a = 7 * a + 13.2 :=
by
  -- This statement needs a proof, so we leave a placeholder 'sorry'
  sorry

end total_people_correct_l171_171298


namespace cube_edge_length_l171_171725

theorem cube_edge_length
  (length_base : ℝ) (width_base : ℝ) (rise_level : ℝ) (volume_displaced : ℝ) (volume_cube : ℝ) (edge_length : ℝ)
  (h_base : length_base = 20) (h_width : width_base = 15) (h_rise : rise_level = 3.3333333333333335)
  (h_volume_displaced : volume_displaced = length_base * width_base * rise_level)
  (h_volume_cube : volume_cube = volume_displaced)
  (h_edge_length_eq : volume_cube = edge_length ^ 3)
  : edge_length = 10 :=
by
  sorry

end cube_edge_length_l171_171725


namespace area_of_50th_ring_l171_171605

-- Definitions based on conditions:
def garden_area : ℕ := 9
def ring_area (n : ℕ) : ℕ := 9 * ((2 * n + 1) ^ 2 - (2 * (n - 1) + 1) ^ 2) / 2

-- Theorem to prove:
theorem area_of_50th_ring : ring_area 50 = 1800 := by sorry

end area_of_50th_ring_l171_171605


namespace sum_of_three_is_odd_implies_one_is_odd_l171_171238

theorem sum_of_three_is_odd_implies_one_is_odd 
  (a b c : ℤ) 
  (h : (a + b + c) % 2 = 1) : 
  a % 2 = 1 ∨ b % 2 = 1 ∨ c % 2 = 1 := 
sorry

end sum_of_three_is_odd_implies_one_is_odd_l171_171238


namespace school_team_profit_is_333_l171_171104

noncomputable def candy_profit (total_bars : ℕ) (price_800_bars : ℕ) (price_400_bars : ℕ) (sold_600_bars_price : ℕ) (remaining_600_bars_price : ℕ) : ℚ :=
  let cost_800_bars := 800 / 3
  let cost_400_bars := 400 / 4
  let total_cost := cost_800_bars + cost_400_bars
  let revenue_sold_600_bars := 600 / 2
  let revenue_remaining_600_bars := (600 * 2) / 3
  let total_revenue := revenue_sold_600_bars + revenue_remaining_600_bars
  total_revenue - total_cost

theorem school_team_profit_is_333 :
  candy_profit 1200 3 4 2 2 = 333 := by
  sorry

end school_team_profit_is_333_l171_171104


namespace scientific_notation_example_l171_171363

theorem scientific_notation_example :
  (0.000000007: ℝ) = 7 * 10^(-9 : ℝ) :=
sorry

end scientific_notation_example_l171_171363


namespace smallest_solution_x4_minus_50x2_plus_625_eq_0_l171_171075

theorem smallest_solution_x4_minus_50x2_plus_625_eq_0 :
  ∃ (x : ℝ), (x * x * x * x - 50 * x * x + 625 = 0) ∧ (∀ y, (y * y * y * y - 50 * y * y + 625 = 0) → x ≤ y) :=
sorry

end smallest_solution_x4_minus_50x2_plus_625_eq_0_l171_171075


namespace price_of_each_brownie_l171_171848

variable (B : ℝ)

theorem price_of_each_brownie (h : 4 * B + 10 + 28 = 50) : B = 3 := by
  -- proof steps would go here
  sorry

end price_of_each_brownie_l171_171848


namespace smallest_N_value_proof_l171_171802

def smallest_value_N (N : ℕ) : Prop :=
  N > 70 ∧ (21 * N) % 70 = 0

theorem smallest_N_value_proof : ∃ N, smallest_value_N N ∧ (∀ M, smallest_value_N M → N ≤ M) :=
  sorry

end smallest_N_value_proof_l171_171802


namespace ellipse_focus_value_l171_171877

theorem ellipse_focus_value (k : ℝ) (hk : 5 * (0:ℝ)^2 - k * (2:ℝ)^2 = 5) : k = -1 :=
by
  sorry

end ellipse_focus_value_l171_171877


namespace factorial_base8_trailing_zeros_l171_171992

-- Define the factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n + 1) * fact n

-- Define the function to count the largest power of a prime p dividing n!
def prime_power_in_factorial (p n : ℕ) : ℕ :=
  if p = 1 then 0 else
  let rec aux k := if k ≤ 0 then 0 else (n / k) + aux (k / p)
  in aux p

-- Define the function to compute number of trailing zeros in base b
def trailing_zeros_in_base (n b : ℕ) : ℕ :=
  let p := match (nat.find_greatest_prime_divisor b) with
           | some p' => p'
           | none => 1
           end
  in (prime_power_in_factorial p n) / (nat.find_greatest_power_of_prime b)

-- Define the statement
theorem factorial_base8_trailing_zeros : trailing_zeros_in_base 15 8 = 3 := by
  sorry

end factorial_base8_trailing_zeros_l171_171992


namespace perfect_square_trinomial_l171_171318

theorem perfect_square_trinomial (k : ℝ) : 
  ∃ a : ℝ, (x^2 - k*x + 1 = (x + a)^2) → (k = 2 ∨ k = -2) :=
by
  sorry

end perfect_square_trinomial_l171_171318


namespace cylinder_radius_l171_171920

theorem cylinder_radius
  (r h : ℝ) (S : ℝ) (h_cylinder : h = 8) (S_surface : S = 130 * Real.pi)
  (surface_area_eq : S = 2 * Real.pi * r^2 + 2 * Real.pi * r * h) :
  r = 5 :=
by
  sorry

end cylinder_radius_l171_171920


namespace sqrt_sum_fractions_eq_l171_171739

theorem sqrt_sum_fractions_eq :
  (Real.sqrt ((1 / 25) + (1 / 36)) = (Real.sqrt 61) / 30) :=
by
  sorry

end sqrt_sum_fractions_eq_l171_171739


namespace lisa_earns_more_than_tommy_l171_171858

theorem lisa_earns_more_than_tommy {total_earnings : ℤ} (h1 : total_earnings = 60) :
  let lisa_earnings := total_earnings / 2
  let tommy_earnings := lisa_earnings / 2
  lisa_earnings - tommy_earnings = 15 :=
by
  sorry

end lisa_earns_more_than_tommy_l171_171858


namespace min_unit_cubes_l171_171009

theorem min_unit_cubes (l w h : ℕ) (S : ℕ) (hS : S = 52) 
  (hSurface : 2 * (l * w + l * h + w * h) = S) : 
  ∃ l w h, l * w * h = 16 :=
by
  -- start the proof here
  sorry

end min_unit_cubes_l171_171009


namespace digit_problem_l171_171157

theorem digit_problem (A B C D E F : ℕ) (hABC : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ D ≠ E ∧ D ≠ F ∧ E ≠ F) 
    (h1 : 100 * A + 10 * B + C = D * 100000 + A * 10000 + E * 1000 + C * 100 + F * 10 + B)
    (h2 : 100 * C + 10 * B + A = E * 100000 + D * 10000 + C * 1000 + A * 100 + B * 10 + F) : 
    A = 3 ∧ B = 6 ∧ C = 4 ∧ D = 1 ∧ E = 2 ∧ F = 9 := 
sorry

end digit_problem_l171_171157


namespace a_b_c_relationship_l171_171526

noncomputable def a (f : ℝ → ℝ) : ℝ := 25 * f (0.2^2)
noncomputable def b (f : ℝ → ℝ) : ℝ := f 1
noncomputable def c (f : ℝ → ℝ) : ℝ := - (Real.log 3 / Real.log 5) * f (Real.log 5 / Real.log 3)

axiom odd_function (f : ℝ → ℝ) : ∀ x, f (-x) = -f x
axiom decreasing_g (f : ℝ → ℝ) : ∀ (x1 x2 : ℝ), 0 < x1 → x1 < x2 → 0 < x2 → (f x1 / x1) > (f x2 / x2)

theorem a_b_c_relationship (f : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x) 
  (h_decreasing : ∀ (x1 x2 : ℝ), 0 < x1 → x1 < x2 → 0 < x2 → (f x1 / x1) > (f x2 / x2)) :
  a f > b f ∧ b f > c f :=
sorry

end a_b_c_relationship_l171_171526


namespace smallest_N_l171_171819

theorem smallest_N (N : ℕ) (hN : N > 70) (hDiv : 70 ∣ 21 * N) : N = 80 :=
sorry

end smallest_N_l171_171819


namespace sharon_highway_speed_l171_171493

theorem sharon_highway_speed:
  ∀ (total_distance : ℝ) (highway_time : ℝ) (city_time: ℝ) (city_speed : ℝ),
  total_distance = 59 → highway_time = 1 / 3 → city_time = 2 / 3 → city_speed = 45 →
  (total_distance - city_speed * city_time) / highway_time = 87 :=
by
  intro total_distance highway_time city_time city_speed
  intro h_total_distance h_highway_time h_city_time h_city_speed
  rw [h_total_distance, h_highway_time, h_city_time, h_city_speed]
  sorry

end sharon_highway_speed_l171_171493


namespace octagon_area_l171_171941

noncomputable def regular_octagon_area_inscribed_circle_radius3 : ℝ :=
  18 * Real.sqrt 2

theorem octagon_area
  (r : ℝ)
  (h : r = 3)
  (octagon_inscribed : ∀ (x : ℝ), x = r * 3 * Real.sin (π / 8)): 
  regular_octagon_area_inscribed_circle_radius3 = 18 * Real.sqrt 2 :=
by
  sorry

end octagon_area_l171_171941


namespace right_triangle_hypotenuse_unique_l171_171488

theorem right_triangle_hypotenuse_unique :
  ∃ (a b c : ℚ) (d e : ℕ), 
    (c^2 = a^2 + b^2) ∧
    (a = 10 * e + d) ∧
    (c = 10 * d + e) ∧
    (d + e = 11) ∧
    (d ≠ e) ∧
    (a = 56) ∧
    (b = 33) ∧
    (c = 65) :=
by {
  sorry
}

end right_triangle_hypotenuse_unique_l171_171488


namespace spherical_to_rectangular_conversion_l171_171606

noncomputable def convert_spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_conversion :
  convert_spherical_to_rectangular 8 (5 * Real.pi / 4) (Real.pi / 4) = (-4, -4, 4 * Real.sqrt 2) :=
by
  sorry

end spherical_to_rectangular_conversion_l171_171606


namespace scientific_notation_correct_l171_171367

-- The number to be converted to scientific notation
def number : ℝ := 0.000000007

-- The scientific notation consisting of a coefficient and exponent
structure SciNotation where
  coeff : ℝ
  exp : ℤ
  def valid (sn : SciNotation) : Prop := sn.coeff ≥ 1 ∧ sn.coeff < 10

-- The proposed scientific notation for the number
def sciNotationOfNumber : SciNotation :=
  { coeff := 7, exp := -9 }

-- The proof statement
theorem scientific_notation_correct : SciNotation.valid sciNotationOfNumber ∧ number = sciNotationOfNumber.coeff * 10 ^ sciNotationOfNumber.exp :=
by
  sorry

end scientific_notation_correct_l171_171367


namespace subset_M_N_l171_171449

def M : Set ℝ := {-1, 1}
def N : Set ℝ := { x | (1 / x < 2) }

theorem subset_M_N : M ⊆ N :=
by
  sorry -- Proof omitted as per the guidelines

end subset_M_N_l171_171449


namespace sum_of_fractions_l171_171759

theorem sum_of_fractions :
  (1 / (1^2 * 2^2) + 1 / (2^2 * 3^2) + 1 / (3^2 * 4^2) + 1 / (4^2 * 5^2)
  + 1 / (5^2 * 6^2) + 1 / (6^2 * 7^2)) = 48 / 49 := 
by
  sorry

end sum_of_fractions_l171_171759


namespace cid_earnings_l171_171118

theorem cid_earnings :
  let model_a_oil_change_cost := 20
  let model_a_repair_cost := 30
  let model_a_wash_cost := 5
  let model_b_oil_change_cost := 25
  let model_b_repair_cost := 40
  let model_b_wash_cost := 8
  let model_c_oil_change_cost := 30
  let model_c_repair_cost := 50
  let model_c_wash_cost := 10

  let model_a_oil_changes := 5
  let model_a_repairs := 10
  let model_a_washes := 15
  let model_b_oil_changes := 3
  let model_b_repairs := 4
  let model_b_washes := 10
  let model_c_oil_changes := 2
  let model_c_repairs := 6
  let model_c_washes := 5

  let total_earnings := 
      (model_a_oil_change_cost * model_a_oil_changes) +
      (model_a_repair_cost * model_a_repairs) +
      (model_a_wash_cost * model_a_washes) +
      (model_b_oil_change_cost * model_b_oil_changes) +
      (model_b_repair_cost * model_b_repairs) +
      (model_b_wash_cost * model_b_washes) +
      (model_c_oil_change_cost * model_c_oil_changes) +
      (model_c_repair_cost * model_c_repairs) +
      (model_c_wash_cost * model_c_washes)

  total_earnings = 1200 := by
  sorry

end cid_earnings_l171_171118


namespace compute_sin_90_l171_171148

noncomputable def sin_90_eq_one : Prop :=
  let angle_0_point := (1, 0) in
  let angle_90_point := (0, 1) in
  (angle_90_point.y = 1)  ∧ ∀ θ : ℝ, θ = 90 → Real.sin (θ * (Real.pi / 180)) = 1

theorem compute_sin_90 : sin_90_eq_one := 
by 
  -- the proof steps go here
  sorry

end compute_sin_90_l171_171148


namespace scientific_notation_correct_l171_171368

-- The number to be converted to scientific notation
def number : ℝ := 0.000000007

-- The scientific notation consisting of a coefficient and exponent
structure SciNotation where
  coeff : ℝ
  exp : ℤ
  def valid (sn : SciNotation) : Prop := sn.coeff ≥ 1 ∧ sn.coeff < 10

-- The proposed scientific notation for the number
def sciNotationOfNumber : SciNotation :=
  { coeff := 7, exp := -9 }

-- The proof statement
theorem scientific_notation_correct : SciNotation.valid sciNotationOfNumber ∧ number = sciNotationOfNumber.coeff * 10 ^ sciNotationOfNumber.exp :=
by
  sorry

end scientific_notation_correct_l171_171368


namespace smallest_N_l171_171820

theorem smallest_N (N : ℕ) (hN : N > 70) (hDiv : 70 ∣ 21 * N) : N = 80 :=
sorry

end smallest_N_l171_171820


namespace find_number_of_children_l171_171276

-- Definition of the problem
def total_cost (num_children num_adults : ℕ) (price_child price_adult : ℕ) : ℕ :=
  num_children * price_child + num_adults * price_adult

-- Given conditions
def conditions (X : ℕ) :=
  let num_adults := X + 25 in
  total_cost X num_adults 8 15 = 720

theorem find_number_of_children :
  ∃ X : ℕ, conditions X ∧ X = 15 :=
by
  sorry

end find_number_of_children_l171_171276


namespace total_ducks_and_ducklings_l171_171861

theorem total_ducks_and_ducklings : 
  let ducks1 := 2
  let ducklings1 := 5
  let ducks2 := 6
  let ducklings2 := 3
  let ducks3 := 9
  let ducklings3 := 6 
  let total_ducklings := ducks1 * ducklings1 + ducks2 * ducklings2 + ducks3 * ducklings3
  let total_ducks := ducks1 + ducks2 + ducks3
  in total_ducks + total_ducklings = 99 := 
by {
  let ducks1 := 2
  let ducklings1 := 5
  let ducks2 := 6
  let ducklings2 := 3
  let ducks3 := 9
  let ducklings3 := 6
  let total_ducklings := ducks1 * ducklings1 + ducks2 * ducklings2 + ducks3 * ducklings3
  let total_ducks := ducks1 + ducks2 + ducks3
  show total_ducks + total_ducklings = 99
  sorry
}

end total_ducks_and_ducklings_l171_171861


namespace inequality_solution_l171_171353

theorem inequality_solution (x : ℝ) :
  (6 * (x ^ 3 - 8) * (Real.sqrt (x ^ 2 + 6 * x + 9)) / ((x ^ 2 + 2 * x + 4) * (x ^ 2 + x - 6)) ≥ x - 2) ↔
  (x ∈ Set.Iic (-4) ∪ Set.Ioo (-3) 2 ∪ Set.Ioo 2 8) := sorry

end inequality_solution_l171_171353


namespace calculate_number_of_boys_l171_171872

theorem calculate_number_of_boys (old_average new_average misread correct_weight : ℝ) (number_of_boys : ℕ)
  (h1 : old_average = 58.4)
  (h2 : misread = 56)
  (h3 : correct_weight = 61)
  (h4 : new_average = 58.65)
  (h5 : (number_of_boys : ℝ) * old_average + (correct_weight - misread) = (number_of_boys : ℝ) * new_average) :
  number_of_boys = 20 :=
by
  sorry

end calculate_number_of_boys_l171_171872


namespace sin_90_eq_1_l171_171128

-- Define the unit circle
def unit_circle (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define the sine of 90 degrees using radians
def sin_90_degrees : ℝ := unit_circle (Real.pi / 2).snd

-- State the theorem
theorem sin_90_eq_1 : sin_90_degrees = 1 :=
by
  sorry

end sin_90_eq_1_l171_171128


namespace find_k_b_find_x_when_y_neg_8_l171_171846

theorem find_k_b (k b : ℤ) (h1 : -20 = 4 * k + b) (h2 : 16 = -2 * k + b) : k = -6 ∧ b = 4 := 
sorry

theorem find_x_when_y_neg_8 (x : ℤ) (k b : ℤ) (h_k : k = -6) (h_b : b = 4) (h_target : -8 = k * x + b) : x = 2 := 
sorry

end find_k_b_find_x_when_y_neg_8_l171_171846


namespace rational_m_abs_nonneg_l171_171830

theorem rational_m_abs_nonneg (m : ℚ) : m + |m| ≥ 0 :=
by sorry

end rational_m_abs_nonneg_l171_171830


namespace cycle_selling_price_l171_171287

theorem cycle_selling_price (initial_price : ℝ)
  (first_discount_percent : ℝ) (second_discount_percent : ℝ) (third_discount_percent : ℝ)
  (first_discounted_price : ℝ) (second_discounted_price : ℝ) :
  initial_price = 3600 →
  first_discount_percent = 15 →
  second_discount_percent = 10 →
  third_discount_percent = 5 →
  first_discounted_price = initial_price * (1 - first_discount_percent / 100) →
  second_discounted_price = first_discounted_price * (1 - second_discount_percent / 100) →
  final_price = second_discounted_price * (1 - third_discount_percent / 100) →
  final_price = 2616.30 :=
by
  intros
  sorry

end cycle_selling_price_l171_171287


namespace difference_between_multiplication_and_subtraction_l171_171554

theorem difference_between_multiplication_and_subtraction (x : ℤ) (h1 : x = 11) :
  (3 * x) - (26 - x) = 18 := by
  sorry

end difference_between_multiplication_and_subtraction_l171_171554


namespace unit_digit_product_zero_l171_171552

def unit_digit (n : ℕ) : ℕ := n % 10

theorem unit_digit_product_zero :
  let a := 785846
  let b := 1086432
  let c := 4582735
  let d := 9783284
  let e := 5167953
  let f := 3821759
  let g := 7594683
  unit_digit (a * b * c * d * e * f * g) = 0 := 
by {
  sorry
}

end unit_digit_product_zero_l171_171552


namespace sin_90_deg_l171_171144

theorem sin_90_deg : Real.sin (90 * Real.pi / 180) = 1 := 
by
  sorry

end sin_90_deg_l171_171144


namespace min_value_of_quadratic_expression_l171_171675

variable (x y z : ℝ)

theorem min_value_of_quadratic_expression 
  (h1 : 2 * x + 2 * y + z + 8 = 0) : 
  (x - 1)^2 + (y + 2)^2 + (z - 3)^2 = 9 :=
sorry

end min_value_of_quadratic_expression_l171_171675


namespace additional_savings_in_cents_l171_171064

/-
The book has a cover price of $30.
There are two discount methods to compare:
1. First $5 off, then 25% off.
2. First 25% off, then $5 off.
Prove that the difference in final costs (in cents) between these two discount methods is 125 cents.
-/
def book_price : ℝ := 30
def discount_cash : ℝ := 5
def discount_percentage : ℝ := 0.25

def final_price_apply_cash_first (price : ℝ) (cash_discount : ℝ) (percentage_discount : ℝ) : ℝ :=
  (price - cash_discount) * (1 - percentage_discount)

def final_price_apply_percentage_first (price : ℝ) (percentage_discount : ℝ) (cash_discount : ℝ) : ℝ :=
  (price * (1 - percentage_discount)) - cash_discount

def savings_comparison (price : ℝ) (cash_discount : ℝ) (percentage_discount : ℝ) : ℝ :=
  (final_price_apply_cash_first price cash_discount percentage_discount) - 
  (final_price_apply_percentage_first price percentage_discount cash_discount)

theorem additional_savings_in_cents : 
  savings_comparison book_price discount_cash discount_percentage * 100 = 125 :=
  by sorry

end additional_savings_in_cents_l171_171064


namespace largest_possible_number_of_sweets_in_each_tray_l171_171392

-- Define the initial conditions as given in the problem statement
def tim_sweets : ℕ := 36
def peter_sweets : ℕ := 44

-- Define the statement that we want to prove
theorem largest_possible_number_of_sweets_in_each_tray :
  Nat.gcd tim_sweets peter_sweets = 4 :=
by
  sorry

end largest_possible_number_of_sweets_in_each_tray_l171_171392


namespace euler_totient_divisibility_l171_171668

theorem euler_totient_divisibility (n : ℕ) (h : n > 0) : n ∣ Nat.totient (2^n - 1) := by
  sorry

end euler_totient_divisibility_l171_171668


namespace property_damage_worth_40000_l171_171117

-- Definitions based on conditions in a)
def medical_bills : ℝ := 70000
def insurance_rate : ℝ := 0.80
def carl_payment : ℝ := 22000
def carl_rate : ℝ := 0.20

theorem property_damage_worth_40000 :
  ∃ P : ℝ, P = 40000 ∧ 
    (carl_payment = carl_rate * (P + medical_bills)) :=
by
  sorry

end property_damage_worth_40000_l171_171117


namespace arithmetic_progression_roots_geometric_progression_roots_harmonic_sequence_roots_l171_171046

-- Arithmetic Progression
theorem arithmetic_progression_roots (a b c : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 - x2 = x2 - x3 ∧ x1 + x2 + x3 = -a ∧ x1 * x2 + x2 * x3 + x1 * x3 = b ∧ -x1 * x2 * x3 = c) 
  ↔ (b = (2 * a^3 + 27 * c) / (9 * a)) :=
sorry

-- Geometric Progression
theorem geometric_progression_roots (a b c : ℝ) :
  (∃ x1 x2 x3 : ℝ, x2 / x1 = x3 / x2 ∧ x1 + x2 + x3 = -a ∧ x1 * x2 + x2 * x3 + x1 * x3 = b ∧ -x1 * x2 * x3 = c) 
  ↔ (b = a * c^(1/3)) :=
sorry

-- Harmonic Sequence
theorem harmonic_sequence_roots (a b c : ℝ) :
  (∃ x1 x2 x3 : ℝ, (x1 - x2) / (x2 - x3) = x1 / x3 ∧ x1 + x2 + x3 = -a ∧ x1 * x2 + x2 * x3 + x1 * x3 = b ∧ -x1 * x2 * x3 = c) 
  ↔ (a = (2 * b^3 + 27 * c) / (9 * b^2)) :=
sorry

end arithmetic_progression_roots_geometric_progression_roots_harmonic_sequence_roots_l171_171046


namespace quarts_of_water_needed_l171_171696

-- Definitions of conditions
def total_parts := 5 + 2 + 1
def total_gallons := 3
def quarts_per_gallon := 4
def water_parts := 5

-- Lean proof statement
theorem quarts_of_water_needed :
  (water_parts : ℚ) * ((total_gallons * quarts_per_gallon) / total_parts) = 15 / 2 :=
by sorry

end quarts_of_water_needed_l171_171696


namespace height_of_picture_frame_l171_171231

-- Definitions of lengths and perimeter
def length : ℕ := 10
def perimeter : ℕ := 44

-- Perimeter formula for a rectangle
def rectangle_perimeter (L H : ℕ) : ℕ := 2 * (L + H)

-- Theorem statement: Proving the height is 12 inches based on given conditions
theorem height_of_picture_frame : ∃ H : ℕ, rectangle_perimeter length H = perimeter ∧ H = 12 := by
  sorry

end height_of_picture_frame_l171_171231


namespace area_of_regular_octagon_in_circle_l171_171929

/-- Define a regular octagon and inscribe it in a circle of radius 3 units, 
    finding the exact area in square units in simplest radical form -/
theorem area_of_regular_octagon_in_circle 
(radius : ℝ) (h_radius : radius = 3) : 
  ∃ (a : ℝ), a = (8 * (1 / 2 * (2 * radius * real.sin (real.pi / 8))^2 * real.sin (real.pi / 4))) :=
by sorry

end area_of_regular_octagon_in_circle_l171_171929


namespace interpretation_of_k5_3_l171_171200

theorem interpretation_of_k5_3 (k : ℕ) (hk : 0 < k) : (k^5)^3 = k^5 * k^5 * k^5 :=
by sorry

end interpretation_of_k5_3_l171_171200


namespace trees_planted_l171_171895

theorem trees_planted (interval trail_length : ℕ) (h1 : interval = 30) (h2 : trail_length = 1200) : 
  trail_length / interval = 40 :=
by
  sorry

end trees_planted_l171_171895


namespace age_of_child_l171_171523

theorem age_of_child (H W C : ℕ) (h1 : (H + W) / 2 = 23) (h2 : (H + 5 + W + 5 + C) / 3 = 19) : C = 1 := by
  sorry

end age_of_child_l171_171523


namespace time_spent_washing_car_l171_171229

theorem time_spent_washing_car (x : ℝ) 
  (h1 : x + (1/4) * x = 100) : x = 80 := 
sorry  

end time_spent_washing_car_l171_171229


namespace third_side_length_l171_171786

theorem third_side_length (x : ℝ) (h1 : 5 > 0) (h2 : 8 > 0) (h3 : 3 < x < 13) : (5 < x + 8) ∧ (x < 5 + 8) ∧ (5 < x + 3) ∧ (x < 8 + 5) := 
by
  sorry

end third_side_length_l171_171786


namespace right_triangle_hypotenuse_l171_171489

theorem right_triangle_hypotenuse (a b c : ℕ) (h : a = 3) (h' : b = 4) (hc : c^2 = a^2 + b^2) : c = 5 := 
by
  -- proof goes here
  sorry

end right_triangle_hypotenuse_l171_171489


namespace necessary_but_not_sufficient_condition_l171_171373

def isEllipse (a b : ℝ) (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ x y : ℝ, f x y = 1

theorem necessary_but_not_sufficient_condition (a b : ℝ) (h : a > 0 ∧ b > 0) : 
  isEllipse a b (λ x y => a * x^2 + b * y^2) → ¬(∃ x y : ℝ, a * x^2 + b * y^2 = 1) :=
sorry

end necessary_but_not_sufficient_condition_l171_171373


namespace find_m_l171_171193

variable (a m x : ℝ)

noncomputable def quadratic_function : ℝ → ℝ := λ x, -a * x^2 + 2 * a * x + 3

theorem find_m (h1 : a > 0) (h2 : quadratic_function a m = 3) (h3 : m ≠ 0) : m = 2 := 
sorry

end find_m_l171_171193


namespace profit_percentage_l171_171578

theorem profit_percentage (C S : ℝ) (hC : C = 800) (hS : S = 1080) :
  ((S - C) / C) * 100 = 35 := 
by
  sorry

end profit_percentage_l171_171578


namespace exists_n_ge_1_le_2020_l171_171665

theorem exists_n_ge_1_le_2020
  (a : ℕ → ℕ)
  (h_distinct : ∀ i j : ℕ, 1 ≤ i → i ≤ 2020 → 1 ≤ j → j ≤ 2020 → i ≠ j → a i ≠ a j)
  (h_periodic1 : a 2021 = a 1)
  (h_periodic2 : a 2022 = a 2) :
  ∃ n : ℕ, 1 ≤ n ∧ n ≤ 2020 ∧ a n ^ 2 + a (n + 1) ^ 2 ≥ a (n + 2) ^ 2 + n ^ 2 + 3 := 
sorry

end exists_n_ge_1_le_2020_l171_171665


namespace num_trailing_zeroes_500_factorial_l171_171949

-- Define the function to count factors of a prime p in n!
def count_factors_in_factorial (n p : ℕ) : ℕ :=
  if p = 0 then 0 else
    (n / p) + (n / (p ^ 2)) + (n / (p ^ 3)) + (n / (p ^ 4))

theorem num_trailing_zeroes_500_factorial : 
  count_factors_in_factorial 500 5 = 124 :=
sorry

end num_trailing_zeroes_500_factorial_l171_171949


namespace find_number_l171_171476

theorem find_number (N : ℝ) 
    (h : 0.20 * ((0.05)^3 * 0.35 * (0.70 * N)) = 182.7) : 
    N = 20880000 :=
by
  -- proof to be filled
  sorry

end find_number_l171_171476


namespace greatest_length_of_pieces_l171_171651

theorem greatest_length_of_pieces (a b c : ℕ) (ha : a = 48) (hb : b = 60) (hc : c = 72) :
  Nat.gcd (Nat.gcd a b) c = 12 := by
  sorry

end greatest_length_of_pieces_l171_171651


namespace negation_proposition_l171_171885

theorem negation_proposition (h : ∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) :
  ¬(∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) = ∃ x : ℝ, x^2 - 2*x + 4 > 0 :=
by
  sorry

end negation_proposition_l171_171885


namespace volume_of_inscribed_cube_l171_171282

theorem volume_of_inscribed_cube (S : ℝ) (π : ℝ) (V : ℝ) (r : ℝ) (s : ℝ) :
    S = 12 * π → 4 * π * r^2 = 12 * π → s = 2 * r → V = s^3 → V = 8 :=
by
  sorry

end volume_of_inscribed_cube_l171_171282


namespace percent_of_dollar_in_pocket_l171_171545

def value_of_penny : ℕ := 1  -- value of one penny in cents
def value_of_nickel : ℕ := 5  -- value of one nickel in cents
def value_of_half_dollar : ℕ := 50 -- value of one half-dollar in cents

def pennies : ℕ := 3  -- number of pennies
def nickels : ℕ := 2  -- number of nickels
def half_dollars : ℕ := 1  -- number of half-dollars

def total_value_in_cents : ℕ :=
  (pennies * value_of_penny) + (nickels * value_of_nickel) + (half_dollars * value_of_half_dollar)

def value_of_dollar_in_cents : ℕ := 100

def percent_of_dollar (value : ℕ) (total : ℕ) : ℚ := (value / total) * 100

theorem percent_of_dollar_in_pocket : percent_of_dollar total_value_in_cents value_of_dollar_in_cents = 63 :=
by
  sorry

end percent_of_dollar_in_pocket_l171_171545


namespace smallest_positive_phi_l171_171323

open Real

theorem smallest_positive_phi :
  (∃ k : ℤ, (2 * φ + π / 4 = π / 2 + k * π)) →
  (∀ k, φ = π / 8 + k * π / 2) → 
  0 < φ → 
  φ = π / 8 :=
by
  sorry

end smallest_positive_phi_l171_171323


namespace number_of_routes_from_A_to_L_is_6_l171_171068

def A_to_B_or_E : Prop := True
def B_to_A_or_C_or_F : Prop := True
def C_to_B_or_D_or_G : Prop := True
def D_to_C_or_H : Prop := True
def E_to_A_or_F_or_I : Prop := True
def F_to_B_or_E_or_G_or_J : Prop := True
def G_to_C_or_F_or_H_or_K : Prop := True
def H_to_D_or_G_or_L : Prop := True
def I_to_E_or_J : Prop := True
def J_to_F_or_I_or_K : Prop := True
def K_to_G_or_J_or_L : Prop := True
def L_from_H_or_K : Prop := True

theorem number_of_routes_from_A_to_L_is_6 
  (h1 : A_to_B_or_E)
  (h2 : B_to_A_or_C_or_F)
  (h3 : C_to_B_or_D_or_G)
  (h4 : D_to_C_or_H)
  (h5 : E_to_A_or_F_or_I)
  (h6 : F_to_B_or_E_or_G_or_J)
  (h7 : G_to_C_or_F_or_H_or_K)
  (h8 : H_to_D_or_G_or_L)
  (h9 : I_to_E_or_J)
  (h10 : J_to_F_or_I_or_K)
  (h11 : K_to_G_or_J_or_L)
  (h12 : L_from_H_or_K) : 
  6 = 6 := 
by 
  sorry

end number_of_routes_from_A_to_L_is_6_l171_171068


namespace min_sum_of_grid_numbers_l171_171292

-- Definition of the 2x2 grid and the problem conditions
variables (a b c d : ℕ)
variables (a_pos : a > 0) (b_pos : b > 0) (c_pos : c > 0) (d_pos : d > 0)

-- Lean statement for the minimum sum proof problem
theorem min_sum_of_grid_numbers :
  a + b + c + d + a * b + c * d + a * c + b * d = 2015 → a + b + c + d = 88 :=
by
  sorry

end min_sum_of_grid_numbers_l171_171292


namespace temperature_on_last_day_l171_171886

noncomputable def last_day_temperature (T1 T2 T3 T4 T5 T6 T7 : ℕ) (mean : ℕ) : ℕ :=
  8 * mean - (T1 + T2 + T3 + T4 + T5 + T6 + T7)

theorem temperature_on_last_day 
  (T1 T2 T3 T4 T5 T6 T7 mean x : ℕ)
  (hT1 : T1 = 82) (hT2 : T2 = 80) (hT3 : T3 = 84) 
  (hT4 : T4 = 86) (hT5 : T5 = 88) (hT6 : T6 = 90) 
  (hT7 : T7 = 88) (hmean : mean = 86) 
  (hx : x = last_day_temperature T1 T2 T3 T4 T5 T6 T7 mean) :
  x = 90 := by
  sorry

end temperature_on_last_day_l171_171886


namespace geometric_sequence_a4_l171_171020

-- Define the terms of the geometric sequence
variable {a : ℕ → ℝ}

-- Define the conditions of the problem
def a2_cond : Prop := a 2 = 2
def a6_cond : Prop := a 6 = 32

-- Define the theorem we want to prove
theorem geometric_sequence_a4 (a2_cond : a 2 = 2) (a6_cond : a 6 = 32) : a 4 = 8 := by
  sorry

end geometric_sequence_a4_l171_171020


namespace probability_different_color_and_label_sum_more_than_3_l171_171016

-- Definitions for the conditions:
structure Coin :=
  (color : Bool) -- True for Yellow, False for Green
  (label : Nat)

def coins : List Coin := [
  Coin.mk true 1,
  Coin.mk true 2,
  Coin.mk false 1,
  Coin.mk false 2,
  Coin.mk false 3
]

def outcomes : List (Coin × Coin) :=
  [(coins[0], coins[1]), (coins[0], coins[2]), (coins[0], coins[3]), (coins[0], coins[4]),
   (coins[1], coins[2]), (coins[1], coins[3]), (coins[1], coins[4]),
   (coins[2], coins[3]), (coins[2], coins[4]), (coins[3], coins[4])]

def different_color_and_label_sum_more_than_3 (c1 c2 : Coin) : Bool :=
  c1.color ≠ c2.color ∧ (c1.label + c2.label > 3)

def valid_outcomes : List (Coin × Coin) :=
  outcomes.filter (λ p => different_color_and_label_sum_more_than_3 p.fst p.snd)

-- Proof statement:
theorem probability_different_color_and_label_sum_more_than_3 :
  (valid_outcomes.length : ℚ) / (outcomes.length : ℚ) = 3 / 10 :=
by
  sorry

end probability_different_color_and_label_sum_more_than_3_l171_171016


namespace ticket_cost_l171_171584

theorem ticket_cost 
  (V G : ℕ)
  (h1 : V + G = 320)
  (h2 : V = G - 212) :
  40 * V + 15 * G = 6150 := 
by
  sorry

end ticket_cost_l171_171584


namespace inverse_sine_function_l171_171056

theorem inverse_sine_function :
  ∀ x : ℝ, x ∈ set.Icc (-(Real.pi / 2)) (Real.pi / 2) → ∀ y : ℝ, y = Real.sin x →
  (x ∈ set.Icc (-1) 1 → Real.arcsin y = x) :=
by
  sorry

end inverse_sine_function_l171_171056


namespace polynomial_correct_l171_171450

-- Define the set of pairs
def xy_pairs : List (ℕ × ℕ) := [(1, 1), (2, 7), (3, 19), (4, 37), (5, 61)]

-- Define the polynomial function
def polynomial (x : ℕ) : ℕ := 3 * x ^ 2 - 3 * x + 1

-- State that for all pairs (x, y) in xy_pairs, polynomial(x) == y
theorem polynomial_correct : ∀ (p : ℕ × ℕ), p ∈ xy_pairs → polynomial p.fst = p.snd :=
by
  intros p hp
  cases hp <;> cases p
  norm_num
  sorry
  norm_num
  sorry
  norm_num
  sorry
  norm_num
  sorry
  norm_num

end polynomial_correct_l171_171450


namespace negative_correction_is_correct_l171_171919

-- Define the constants given in the problem
def gain_per_day : ℚ := 13 / 4
def set_time : ℚ := 8 -- 8 A.M. on April 10
def end_time : ℚ := 15 -- 3 P.M. on April 19
def days_passed : ℚ := 9

-- Calculate the total time in hours from 8 A.M. on April 10 to 3 P.M. on April 19
def total_hours_passed : ℚ := days_passed * 24 + (end_time - set_time)

-- Calculate the gain in time per hour
def gain_per_hour : ℚ := gain_per_day / 24

-- Calculate the total gained time over the total hours passed
def total_gain : ℚ := total_hours_passed * gain_per_hour

-- The negative correction m to be subtracted
def correction : ℚ := 2899 / 96

theorem negative_correction_is_correct :
  total_gain = correction :=
by
-- skipping the proof
sorry

end negative_correction_is_correct_l171_171919


namespace triangle_OAB_area_range_l171_171379

noncomputable def area_of_triangle_OAB (m : ℝ) : ℝ :=
  4 * Real.sqrt (64 * m^2 + 4 * 64)

theorem triangle_OAB_area_range :
  ∀ m : ℝ, 64 ≤ area_of_triangle_OAB m :=
by
  intro m
  sorry

end triangle_OAB_area_range_l171_171379


namespace thirteen_pow_seven_mod_nine_l171_171673

theorem thirteen_pow_seven_mod_nine : (13^7 % 9 = 4) :=
by {
  sorry
}

end thirteen_pow_seven_mod_nine_l171_171673


namespace people_per_column_in_second_arrangement_l171_171209
-- Lean 4 Statement

theorem people_per_column_in_second_arrangement :
  ∀ P X : ℕ, (P = 30 * 16) → (12 * X = P) → X = 40 :=
by
  intros P X h1 h2
  sorry

end people_per_column_in_second_arrangement_l171_171209


namespace largest_integer_less_than_80_with_remainder_3_when_divided_by_5_l171_171961

theorem largest_integer_less_than_80_with_remainder_3_when_divided_by_5 : 
  ∃ x : ℤ, x < 80 ∧ x % 5 = 3 ∧ (∀ y : ℤ, y < 80 ∧ y % 5 = 3 → y ≤ x) :=
sorry

end largest_integer_less_than_80_with_remainder_3_when_divided_by_5_l171_171961


namespace triangle_area_is_31_5_l171_171398

def point := (ℝ × ℝ)

def A : point := (2, 3)
def B : point := (9, 3)
def C : point := (5, 12)

def triangle_area (A B C : point) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem triangle_area_is_31_5 :
  triangle_area A B C = 31.5 :=
by
  -- Placeholder for the proof
  sorry

end triangle_area_is_31_5_l171_171398


namespace sqrt_inequality_l171_171168

theorem sqrt_inequality (x : ℝ) : abs ((x^2 - 9) / 3) < 3 ↔ -Real.sqrt 18 < x ∧ x < Real.sqrt 18 :=
by
  sorry

end sqrt_inequality_l171_171168


namespace avg_diff_l171_171108

theorem avg_diff (n : ℕ) (m : ℝ) (mistake : ℝ) (true_value : ℝ)
   (h_n : n = 30) (h_mistake : mistake = 15) (h_true_value : true_value = 105) 
   (h_m : m = true_value - mistake) : 
   (m / n) = 3 := 
by
  sorry

end avg_diff_l171_171108


namespace members_count_l171_171257

theorem members_count
  (n : ℝ)
  (h1 : 191.25 = n / 4) :
  n = 765 :=
by
  sorry

end members_count_l171_171257


namespace find_xyz_l171_171698

theorem find_xyz : ∃ (x y z : ℕ), x + y + z = 12 ∧ 7 * x + 5 * y + 8 * z = 79 ∧ x = 5 ∧ y = 4 ∧ z = 3 :=
by
  sorry

end find_xyz_l171_171698


namespace Elaine_rent_percentage_l171_171222

variable (E : ℝ) (last_year_rent : ℝ) (this_year_rent : ℝ)

def Elaine_last_year_earnings (E : ℝ) : ℝ := E

def Elaine_last_year_rent (E : ℝ) : ℝ := 0.20 * E

def Elaine_this_year_earnings (E : ℝ) : ℝ := 1.25 * E

def Elaine_this_year_rent (E : ℝ) : ℝ := 0.30 * (1.25 * E)

theorem Elaine_rent_percentage 
  (E : ℝ) 
  (last_year_rent := Elaine_last_year_rent E)
  (this_year_rent := Elaine_this_year_rent E) :
  (this_year_rent / last_year_rent) * 100 = 187.5 := 
by sorry

end Elaine_rent_percentage_l171_171222


namespace sum_of_last_three_coefficients_l171_171270

open scoped BigOperators

/-- The sum of the last three coefficients of the polynomial expansion of (1 - 2/x)^7 is 29. -/
theorem sum_of_last_three_coefficients : 
  let f := λ (x : ℚ), (1 - 2 / x)
  let coeffs := (∑ k in Finset.range 8, (choose 7 k) * (-2:ℚ) ^ k * x ^ (7 - k))
  let last_three := (coeffs.coeff 0) + (coeffs.coeff 1) + (coeffs.coeff 2)
  last_three = (29:ℚ) :=
by 
  sorry

end sum_of_last_three_coefficients_l171_171270


namespace leq_sum_l171_171619

open BigOperators

theorem leq_sum (x : Fin 3 → ℝ) (hx_pos : ∀ i, 0 < x i) (hx_sum : ∑ i, x i = 1) :
  (∑ i, 1 / (1 + (x i)^2)) ≤ 27 / 10 :=
sorry

end leq_sum_l171_171619


namespace square_of_third_side_l171_171006

theorem square_of_third_side (a b : ℕ) (h1 : a = 4) (h2 : b = 5) 
    (h_right_triangle : (a^2 + b^2 = c^2) ∨ (b^2 + c^2 = a^2)) : 
    (c = 9) ∨ (c = 41) :=
sorry

end square_of_third_side_l171_171006


namespace correct_option_is_C_l171_171083

-- Our conditions as mathematical expressions
def condition_A : Prop := (+6) + (-13) = +7
def condition_B : Prop := (+6) + (-13) = -19
def condition_C : Prop := (+6) + (-13) = -7
def condition_D : Prop := (-5) + (-3) = 8

-- The proposition we need to prove
theorem correct_option_is_C : condition_C ∧ ¬condition_A ∧ ¬condition_B ∧ ¬condition_D :=
by 
  sorry

end correct_option_is_C_l171_171083


namespace find_y_l171_171018

theorem find_y (DEG EFG y : ℝ) 
  (h1 : DEG = 150)
  (h2 : EFG = 40)
  (h3 : DEG = EFG + y) :
  y = 110 :=
by
  sorry

end find_y_l171_171018


namespace fish_to_rice_value_l171_171205

variable (f l r : ℝ)

theorem fish_to_rice_value (h1 : 5 * f = 3 * l) (h2 : 2 * l = 7 * r) : f = 2.1 * r :=
by
  sorry

end fish_to_rice_value_l171_171205


namespace larger_number_l171_171897

theorem larger_number (x y : ℕ) (h1 : x + y = 47) (h2 : x - y = 3) : max x y = 25 :=
sorry

end larger_number_l171_171897


namespace player1_wins_11th_round_probability_l171_171898

-- Definitions based on the conditions
def egg_shell_strength (n : ℕ) : ℝ := sorry
def player1_won_first_10_rounds : Prop := sorry

-- Main theorem
theorem player1_wins_11th_round_probability
  (h : player1_won_first_10_rounds) :
  Prob (egg_shell_strength 11 > egg_shell_strength 12) = 11 / 12 := sorry

end player1_wins_11th_round_probability_l171_171898


namespace john_task_completion_l171_171023

theorem john_task_completion (J : ℝ) (h : 5 * (1 / J + 1 / 10) + 5 * (1 / J) = 1) : J = 20 :=
by
  sorry

end john_task_completion_l171_171023


namespace sum_of_squares_five_consecutive_ints_not_perfect_square_l171_171909

theorem sum_of_squares_five_consecutive_ints_not_perfect_square (n : ℤ) :
  ∀ k : ℤ, k^2 ≠ 5 * (n^2 + 2) := 
sorry

end sum_of_squares_five_consecutive_ints_not_perfect_square_l171_171909


namespace algebraic_expression_l171_171461

theorem algebraic_expression (x : ℝ) (h : x - 1/x = 3) : x^2 + 1/x^2 = 11 := 
by
  sorry

end algebraic_expression_l171_171461


namespace area_of_shaded_region_l171_171354

theorem area_of_shaded_region:
  let b := 10
  let h := 6
  let n := 14
  let rect_length := 2
  let rect_height := 1.5
  (n * rect_length * rect_height - (1/2 * b * h)) = 12 := 
by
  sorry

end area_of_shaded_region_l171_171354


namespace number_of_employees_excluding_manager_l171_171245

theorem number_of_employees_excluding_manager 
  (avg_salary : ℕ)
  (manager_salary : ℕ)
  (new_avg_salary : ℕ)
  (n : ℕ)
  (T : ℕ)
  (h1 : avg_salary = 1600)
  (h2 : manager_salary = 3700)
  (h3 : new_avg_salary = 1700)
  (h4 : T = n * avg_salary)
  (h5 : T + manager_salary = (n + 1) * new_avg_salary) :
  n = 20 :=
by
  sorry

end number_of_employees_excluding_manager_l171_171245


namespace emily_wrong_questions_l171_171758

variable (E F G H : ℕ)

theorem emily_wrong_questions (h1 : E + F + 4 = G + H) 
                             (h2 : E + H = F + G + 8) 
                             (h3 : G = 6) : 
                             E = 8 :=
sorry

end emily_wrong_questions_l171_171758


namespace evaluate_expression_l171_171769

theorem evaluate_expression :
  let a := 1
  let b := 10
  let c := 100
  let d := 1000
  (a + b + c - d) + (a + b - c + d) + (a - b + c + d) + (-a + b + c + d) = 2222 :=
by
  let a := 1
  let b := 10
  let c := 100
  let d := 1000
  sorry

end evaluate_expression_l171_171769


namespace average_weight_of_arun_l171_171718

variable (weight : ℝ)

def arun_constraint := 61 < weight ∧ weight < 72
def brother_constraint := 60 < weight ∧ weight < 70
def mother_constraint := weight ≤ 64
def father_constraint := 62 < weight ∧ weight < 73
def sister_constraint := 59 < weight ∧ weight < 68

theorem average_weight_of_arun : 
  (∃ w : ℝ, arun_constraint w ∧ brother_constraint w ∧ mother_constraint w ∧ father_constraint w ∧ sister_constraint w) →
  (63.5 = (63 + 64) / 2) := 
by
  sorry

end average_weight_of_arun_l171_171718


namespace perfect_squares_diff_consecutive_l171_171471

theorem perfect_squares_diff_consecutive (h1 : ∀ a : ℕ, a^2 < 1000000 → ∃ b : ℕ, a^2 = (b + 1)^2 - b^2) : 
  (∃ n : ℕ, n = 500) := 
by 
  sorry

end perfect_squares_diff_consecutive_l171_171471


namespace opposite_difference_five_times_l171_171845

variable (a b : ℤ) -- Using integers for this example

theorem opposite_difference_five_times (a b : ℤ) : (-a - 5 * b) = -(a) - (5 * b) := 
by
  -- The proof details would be filled in here
  sorry

end opposite_difference_five_times_l171_171845


namespace find_m_eq_2_l171_171188

theorem find_m_eq_2 (a m : ℝ) (h1 : a > 0) (h2 : -a * m^2 + 2 * a * m + 3 = 3) (h3 : m ≠ 0) : m = 2 :=
by
  sorry

end find_m_eq_2_l171_171188


namespace Alfred_repair_cost_l171_171730

noncomputable def scooter_price : ℕ := 4700
noncomputable def sale_price : ℕ := 5800
noncomputable def gain_percent : ℚ := 9.433962264150944
noncomputable def gain_value (repair_cost : ℚ) : ℚ := sale_price - (scooter_price + repair_cost)

theorem Alfred_repair_cost : ∃ R : ℚ, gain_percent = (gain_value R / (scooter_price + R)) * 100 ∧ R = 600 :=
by
  sorry

end Alfred_repair_cost_l171_171730


namespace bakery_ratio_l171_171483

theorem bakery_ratio (F B : ℕ) 
    (h1 : F = 10 * B)
    (h2 : F = 8 * (B + 60))
    (sugar : ℕ)
    (h3 : sugar = 3000) :
    sugar / F = 5 / 4 :=
by sorry

end bakery_ratio_l171_171483


namespace coefficient_x4_in_expression_l171_171165

theorem coefficient_x4_in_expression : 
  let expr := 4 * (Polynomial.C 1 * Polynomial.X ^ 4 - Polynomial.C 2 * Polynomial.X ^ 5) + 
              3 * (Polynomial.C 1 * Polynomial.X ^ 2 - Polynomial.C 1 * Polynomial.X ^ 4 - Polynomial.C 2 * Polynomial.X ^ 6) - 
              (Polynomial.C 5 * Polynomial.X ^ 5 - Polynomial.C 2 * Polynomial.X ^ 4)
  in Polynomial.coeff expr 4 = 3 := 
by 
  sorry

end coefficient_x4_in_expression_l171_171165


namespace difference_of_squares_l171_171747

theorem difference_of_squares : (23 + 15)^2 - (23 - 15)^2 = 1380 := by
  sorry

end difference_of_squares_l171_171747


namespace solve_for_x_l171_171457

variable (a b x : ℝ)

def operation (a b : ℝ) : ℝ := (a + 5) * b

theorem solve_for_x (h : operation x 1.3 = 11.05) : x = 3.5 :=
by
  sorry

end solve_for_x_l171_171457


namespace sellingPrice_is_459_l171_171032

-- Definitions based on conditions
def costPrice : ℝ := 540
def markupPercentage : ℝ := 0.15
def discountPercentage : ℝ := 0.2608695652173913

-- Calculating the marked price based on the given conditions
def markedPrice (cp : ℝ) (markup : ℝ) : ℝ := cp + (markup * cp)

-- Calculating the discount amount based on the marked price and the discount percentage
def discount (mp : ℝ) (discountPct : ℝ) : ℝ := discountPct * mp

-- Calculating the selling price
def sellingPrice (mp : ℝ) (discountAmt : ℝ) : ℝ := mp - discountAmt

-- Stating the final proof problem
theorem sellingPrice_is_459 :
  sellingPrice (markedPrice costPrice markupPercentage) (discount (markedPrice costPrice markupPercentage) discountPercentage) = 459 :=
by
  sorry

end sellingPrice_is_459_l171_171032


namespace soldier_initial_consumption_l171_171012

theorem soldier_initial_consumption :
  ∀ (s d1 n : ℕ) (c2 d2 : ℝ), 
    s = 1200 → d1 = 30 → n = 528 → c2 = 2.5 → d2 = 25 → 
    36000 * (x : ℝ) = 108000 → x = 3 := 
by {
  sorry
}

end soldier_initial_consumption_l171_171012


namespace derivative_y_over_x_l171_171912

noncomputable def x (t : ℝ) : ℝ := (t^2 * Real.log t) / (1 - t^2) + Real.log (Real.sqrt (1 - t^2))
noncomputable def y (t : ℝ) : ℝ := (t / Real.sqrt (1 - t^2)) * Real.arcsin t + Real.log (Real.sqrt (1 - t^2))

theorem derivative_y_over_x (t : ℝ) (ht : t ≠ 0) (h1 : t ≠ 1) (hneg1 : t ≠ -1) : 
  (deriv y t) / (deriv x t) = (Real.arcsin t * Real.sqrt (1 - t^2)) / (2 * t * Real.log t) :=
by
  sorry

end derivative_y_over_x_l171_171912


namespace min_value_am_gm_l171_171500

theorem min_value_am_gm (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a / b) + (b / c) + (c / d) + (d / a) ≥ 4 := 
sorry

end min_value_am_gm_l171_171500


namespace scientific_notation_example_l171_171364

theorem scientific_notation_example :
  (0.000000007: ℝ) = 7 * 10^(-9 : ℝ) :=
sorry

end scientific_notation_example_l171_171364


namespace range_of_slope_angle_l171_171894

theorem range_of_slope_angle (P A B : ℝ × ℝ) (hP : P = (0, -1)) (hA : A = (1, -2)) (hB : B = (2, 1)) :
  ∀ l, (∃ m : ℝ, l = λ x, m * (x - P.1) + P.2 ∧ (-1 ≤ m ∧ m ≤ 1) ∧ ∃ t ∈ Icc 0 1, (1 - t) • A + t • B = l x) →
  ∃ θ : ℝ, θ ∈ Icc 0 (π / 4) ∨ θ ∈ Icc (3 * π / 4) π ∧ real.tan θ = m :=
by
  sorry

end range_of_slope_angle_l171_171894


namespace income_after_tax_l171_171408

def poor_income_perc (x : ℝ) : ℝ := x

def middle_income_perc (x : ℝ) : ℝ := 4 * x

def rich_income_perc (x : ℝ) : ℝ := 5 * x

def rich_tax_rate (x : ℝ) : ℝ := (x^2 / 4) + x

def post_tax_rich_income (x : ℝ) : ℝ := rich_income_perc x * (1 - rich_tax_rate x)

def tax_collected (x : ℝ) : ℝ := rich_income_perc x - post_tax_rich_income x

def tax_to_poor (x : ℝ) : ℝ := (3 / 4) * tax_collected x

def tax_to_middle (x : ℝ) : ℝ := (1 / 4) * tax_collected x

def new_poor_income (x : ℝ) : ℝ := poor_income_perc x + tax_to_poor x

def new_middle_income (x : ℝ) : ℝ := middle_income_perc x + tax_to_middle x

def new_rich_income (x : ℝ) : ℝ := post_tax_rich_income x

theorem income_after_tax (x : ℝ) (h : 10 * x = 100) :
  new_poor_income x + new_middle_income x + new_rich_income x = 100 := by
  sorry

end income_after_tax_l171_171408


namespace jelly_price_l171_171418

theorem jelly_price (d1 h1 d2 h2 : ℝ) (P1 : ℝ)
    (hd1 : d1 = 2) (hh1 : h1 = 5) (hd2 : d2 = 4) (hh2 : h2 = 8) (P1_cond : P1 = 0.75) :
    ∃ P2 : ℝ, P2 = 2.40 :=
by
  sorry

end jelly_price_l171_171418


namespace common_root_equation_l171_171174

theorem common_root_equation {m : ℝ} (x : ℝ) (h1 : m * x - 1000 = 1001) (h2 : 1001 * x = m - 1000 * x) : m = 2001 ∨ m = -2001 :=
by
  -- Skipping the proof details
  sorry

end common_root_equation_l171_171174


namespace factorial_base_8_zeroes_l171_171994

theorem factorial_base_8_zeroes (n : ℕ) :
  n = 15 →
  largest_power_8_dividing_factorial_n = 3 :=
begin
  assume hn : n = 15,
  -- Definitions and setup based on conditions
  let k := largest_power_of_prime_dividing_factorial 2 n,
  have hk : k = 11, 
  { sorry }, -- Sum the factors of 2 as shown in the solution steps
  have hp8 : largest_power_8_dividing_factorial_n = k / 3,
  { sorry }, -- Calculate the integer division k / 3 to find power of 8 division
  rw hn at *,
  exact eq.trans hp8.symm (nat.div_eq_of_lt_trans (nat.lt_succ_self 2 * (k / 3))),
end

end factorial_base_8_zeroes_l171_171994


namespace max_remainder_is_8_l171_171422

theorem max_remainder_is_8 (d q r : ℕ) (h1 : d = 9) (h2 : q = 6) (h3 : r < d) : 
  r ≤ (d - 1) :=
by 
  sorry

end max_remainder_is_8_l171_171422


namespace bucket_proof_l171_171112

variable (CA : ℚ) -- capacity of Bucket A
variable (CB : ℚ) -- capacity of Bucket B
variable (SA_init : ℚ) -- initial amount of sand in Bucket A
variable (SB_init : ℚ) -- initial amount of sand in Bucket B

def bucket_conditions : Prop := 
  CB = (1 / 2) * CA ∧
  SA_init = (1 / 4) * CA ∧
  SB_init = (3 / 8) * CB

theorem bucket_proof (h : bucket_conditions CA CB SA_init SB_init) : 
  (SA_init + SB_init) / CA = 7 / 16 := 
  by sorry

end bucket_proof_l171_171112


namespace polynomial_divisible_by_p_l171_171495

open Nat

/-- Define the sequence of polynomials as described -/
def Q : ℕ → ℕ → ℤ 
| 0, x := 1
| 1, x := x
| (n + 1), x := x * Q n x + n * Q (n - 1) x

/-- Main theorem to prove the divisibility -/
theorem polynomial_divisible_by_p (p x : ℕ) (hp : p > 2 ∧ Prime p) : 
    p ∣ (Q p x - x ^ p) := sorry

end polynomial_divisible_by_p_l171_171495


namespace roots_cubic_polynomial_l171_171027

theorem roots_cubic_polynomial (r s t : ℝ)
  (h₁ : 8 * r^3 + 1001 * r + 2008 = 0)
  (h₂ : 8 * s^3 + 1001 * s + 2008 = 0)
  (h₃ : 8 * t^3 + 1001 * t + 2008 = 0)
  (h₄ : r + s + t = 0) :
  (r + s)^3 + (s + t)^3 + (t + r)^3 = 753 := 
sorry

end roots_cubic_polynomial_l171_171027


namespace distance_between_first_and_last_student_l171_171540

theorem distance_between_first_and_last_student 
  (n : ℕ) (d : ℕ)
  (students : n = 30) 
  (distance_between_students : d = 3) : 
  n - 1 * d = 87 := 
by
  sorry

end distance_between_first_and_last_student_l171_171540


namespace compute_diff_of_squares_l171_171742

theorem compute_diff_of_squares : (65^2 - 35^2 = 3000) :=
by
  sorry

end compute_diff_of_squares_l171_171742


namespace least_positive_t_l171_171743

theorem least_positive_t
  (α : ℝ) (hα : 0 < α ∧ α < π / 2)
  (ht : ∃ t, 0 < t ∧ (∃ r, (Real.arcsin (Real.sin α) * r = Real.arcsin (Real.sin (3 * α)) ∧ 
                            Real.arcsin (Real.sin (3 * α)) * r = Real.arcsin (Real.sin (5 * α)) ∧
                            Real.arcsin (Real.sin (5 * α)) * r = Real.arcsin (Real.sin (t * α))))) :
  t = 6 :=
sorry

end least_positive_t_l171_171743


namespace jill_spent_10_percent_on_food_l171_171342

theorem jill_spent_10_percent_on_food 
  (T : ℝ)                         
  (h1 : 0.60 * T = 0.60 * T)    -- 60% on clothing
  (h2 : 0.30 * T = 0.30 * T)    -- 30% on other items
  (h3 : 0.04 * (0.60 * T) = 0.024 * T)  -- 4% tax on clothing
  (h4 : 0.08 * (0.30 * T) = 0.024 * T)  -- 8% tax on other items
  (h5 : 0.048 * T = (0.024 * T + 0.024 * T)) -- total tax is 4.8%
  : 0.10 * T = (T - (0.60*T + 0.30*T)) :=
by
  -- Proof is omitted
  sorry

end jill_spent_10_percent_on_food_l171_171342


namespace calculation_result_l171_171445

theorem calculation_result:
  (-1:ℤ)^3 - 8 / (-2) + 4 * abs (-5) = 23 := by
  sorry

end calculation_result_l171_171445


namespace smallest_n_in_range_l171_171456

theorem smallest_n_in_range (n : ℤ) (h1 : 4 ≤ n ∧ n ≤ 12) (h2 : n ≡ 2 [ZMOD 9]) : n = 11 :=
sorry

end smallest_n_in_range_l171_171456


namespace scientific_notation_correct_l171_171365

theorem scientific_notation_correct :
  0.000000007 = 7 * 10^(-9) :=
by
  sorry

end scientific_notation_correct_l171_171365


namespace ages_sum_l171_171505

theorem ages_sum (a b c : ℕ) (h1 : a = b) (h2 : a * b * c = 72) : a + b + c = 14 :=
by sorry

end ages_sum_l171_171505


namespace fallen_sheets_l171_171573

/-- The number of sheets that fell out of a book given the first page is 163
    and the last page contains the same digits but arranged in a different 
    order and ends with an even digit.
-/
theorem fallen_sheets (h1 : ∃ n, n = 163 ∧ 
                        ∃ m, m ≠ n ∧ (m = 316) ∧ 
                        m % 2 = 0 ∧ 
                        (∃ p1 p2 p3 q1 q2 q3, 
                         (p1, p2, p3) ≠ (q1, q2, q3) ∧ 
                         p1 ≠ q1 ∧ p2 ≠ q2 ∧ p3 ≠ q3 ∧ 
                         n = p1 * 100 + p2 * 10 + p3 ∧ 
                         m = q1 * 100 + q2 * 10 + q3)) :
  ∃ k, k = 77 :=
by
  sorry

end fallen_sheets_l171_171573


namespace star_comm_l171_171235

section SymmetricOperation

variable {S : Type*} 
variable (star : S → S → S)
variable (symm : ∀ a b : S, star a b = star (star b a) (star b a)) 

theorem star_comm (a b : S) : star a b = star b a := 
by 
  sorry

end SymmetricOperation

end star_comm_l171_171235


namespace bonnets_per_orphanage_l171_171033

/--
Mrs. Young makes bonnets for kids in the orphanage.
On Monday, she made 10 bonnets.
On Tuesday and Wednesday combined she made twice more than on Monday.
On Thursday she made 5 more than on Monday.
On Friday she made 5 less than on Thursday.
She divided up the bonnets evenly and sent them to 5 orphanages.
Prove that the number of bonnets Mrs. Young sent to each orphanage is 11.
-/
theorem bonnets_per_orphanage :
  let monday := 10
  let tuesday_wednesday := 2 * monday
  let thursday := monday + 5
  let friday := thursday - 5
  let total_bonnets := monday + tuesday_wednesday + thursday + friday
  let orphanages := 5
  total_bonnets / orphanages = 11 :=
by
  sorry

end bonnets_per_orphanage_l171_171033


namespace zeroes_at_end_base_8_of_factorial_15_l171_171998

theorem zeroes_at_end_base_8_of_factorial_15 : 
  let a := factorial 15
  in let num_twos := (∑ k in Icc 1 15, padicValRat 2 k)
  in num_twos / 3 = 3 :=
by {
  sorry
}

end zeroes_at_end_base_8_of_factorial_15_l171_171998


namespace trigonometric_identity_l171_171304

theorem trigonometric_identity (α : ℝ) (h1 : 0 < α) (h2 : α < π) (h3 : Real.tan α = -2) :
  2 * (Real.sin α)^2 - (Real.sin α) * (Real.cos α) + (Real.cos α)^2 = 11 / 5 := by
  sorry

end trigonometric_identity_l171_171304


namespace hindi_books_count_l171_171535

theorem hindi_books_count (H : ℕ) (h1 : 22 = 22) (h2 : Nat.choose 23 H = 1771) : H = 3 :=
sorry

end hindi_books_count_l171_171535


namespace trigonometric_identity_proof_l171_171657

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < π / 2)
variable (hβ : 0 < β ∧ β < π / 2)
variable (h : Real.tan α = (1 + Real.sin β) / Real.cos β)

theorem trigonometric_identity_proof : 2 * α - β = π / 2 := 
by 
  sorry

end trigonometric_identity_proof_l171_171657


namespace max_gcd_l171_171600

theorem max_gcd (n : ℕ) (h : 0 < n) : ∀ n, ∃ d ≥ 1, d ∣ 13 * n + 4 ∧ d ∣ 8 * n + 3 → d ≤ 9 :=
begin
  sorry
end

end max_gcd_l171_171600


namespace probability_black_white_l171_171923

structure Jar :=
  (black_balls : ℕ)
  (white_balls : ℕ)
  (green_balls : ℕ)

def total_balls (j : Jar) : ℕ :=
  j.black_balls + j.white_balls + j.green_balls

def choose (n k : ℕ) : ℕ := n.choose k

theorem probability_black_white (j : Jar) (h_black : j.black_balls = 3) (h_white : j.white_balls = 3) (h_green : j.green_balls = 1) :
  (choose 3 1 * choose 3 1) / (choose (total_balls j) 2) = 3 / 7 :=
by
  sorry

end probability_black_white_l171_171923


namespace find_beta_l171_171618

open Real

theorem find_beta (α β : ℝ) (h1 : cos α = 1 / 7) (h2 : cos (α - β) = 13 / 14)
  (h3 : 0 < β) (h4 : β < α) (h5 : α < π / 2) : β = π / 3 :=
by
  sorry

end find_beta_l171_171618


namespace variance_ξ_l171_171101

variable (P : ℕ → ℝ) (ξ : ℕ)

-- conditions
axiom P_0 : P 0 = 1 / 5
axiom P_1 : P 1 + P 2 = 4 / 5
axiom E_ξ : (0 * P 0 + 1 * P 1 + 2 * P 2) = 1

-- proof statement
theorem variance_ξ : (0 - 1)^2 * P 0 + (1 - 1)^2 * P 1 + (2 - 1)^2 * P 2 = 2 / 5 :=
by sorry

end variance_ξ_l171_171101


namespace arithmetic_seq_a12_l171_171462

def arithmetic_seq (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
  ∀ n : ℕ, a n = a1 + (n - 1) * d

theorem arithmetic_seq_a12 (a : ℕ → ℝ) (a1 d : ℝ) 
  (h_arith : arithmetic_seq a a1 d)
  (h7_and_9 : a 7 + a 9 = 16)
  (h4 : a 4 = 1) :
  a 12 = 15 :=
by
  sorry

end arithmetic_seq_a12_l171_171462


namespace product_of_roots_l171_171336

theorem product_of_roots (p q r : ℝ)
  (h1 : ∀ x : ℝ, (3 * x^3 - 9 * x^2 + 5 * x - 15 = 0) → (x = p ∨ x = q ∨ x = r)) :
  p * q * r = 5 := by
  sorry

end product_of_roots_l171_171336


namespace Amanda_family_paint_walls_l171_171435

theorem Amanda_family_paint_walls :
  let num_people := 5
  let rooms_with_4_walls := 5
  let rooms_with_5_walls := 4
  let walls_per_room_4 := 4
  let walls_per_room_5 := 5
  let total_walls := (rooms_with_4_walls * walls_per_room_4) + (rooms_with_5_walls * walls_per_room_5)
  total_walls / num_people = 8 :=
by
  -- We add a sorry to skip proof
  sorry

end Amanda_family_paint_walls_l171_171435


namespace number_of_girls_l171_171233

theorem number_of_girls {total_children boys girls : ℕ} 
  (h_total : total_children = 60) 
  (h_boys : boys = 18) 
  (h_girls : girls = total_children - boys) : 
  girls = 42 := by 
  sorry

end number_of_girls_l171_171233


namespace verify_number_of_true_props_l171_171474

def original_prop (a : ℝ) : Prop := a > -3 → a > 0
def converse_prop (a : ℝ) : Prop := a > 0 → a > -3
def inverse_prop (a : ℝ) : Prop := a ≤ -3 → a ≤ 0
def contrapositive_prop (a : ℝ) : Prop := a ≤ 0 → a ≤ -3

theorem verify_number_of_true_props :
  (¬ original_prop a ∧ converse_prop a ∧ inverse_prop a ∧ ¬ contrapositive_prop a) → (2 = 2) := sorry

end verify_number_of_true_props_l171_171474


namespace find_s_l171_171515

noncomputable def utility (hours_math hours_frisbee : ℝ) : ℝ :=
  (hours_math + 2) * hours_frisbee

theorem find_s (s : ℝ) :
  utility (10 - 2 * s) s = utility (2 * s + 4) (3 - s) ↔ s = 3 / 2 := 
by 
  sorry

end find_s_l171_171515


namespace choose_three_cards_of_different_suits_l171_171637

/-- The number of ways to choose 3 cards from a standard deck of 52 cards,
if all three cards must be of different suits -/
theorem choose_three_cards_of_different_suits :
  let n := 4
  let r := 3
  let suits_combinations := Nat.choose n r
  let cards_per_suit := 13
  let total_ways := suits_combinations * (cards_per_suit ^ r)
  total_ways = 8788 :=
by
  sorry

end choose_three_cards_of_different_suits_l171_171637


namespace smallest_possible_value_of_N_l171_171806

theorem smallest_possible_value_of_N :
  ∃ N : ℕ, N > 70 ∧ (21 * N) % 70 = 0 ∧ N = 80 :=
by
  existsi 80
  split
  · norm_num
  split
  · norm_num
  sorry

end smallest_possible_value_of_N_l171_171806


namespace y_pow_expression_l171_171532

theorem y_pow_expression (y : ℝ) (h : y + 1/y = 3) : y^13 - 5 * y^9 + y^5 = 0 :=
sorry

end y_pow_expression_l171_171532


namespace sin_ninety_degrees_l171_171133

theorem sin_ninety_degrees : Real.sin (90 * Real.pi / 180) = 1 := 
by
  sorry

end sin_ninety_degrees_l171_171133


namespace regular_octagon_area_l171_171938

-- Define the problem conditions
def inscribed_circle_radius : ℝ := 3
def central_angle : ℝ := 360 / 8
def side_length (r : ℝ) : ℝ := 2 * r * sin (central_angle / 2 * real.pi / 180)

-- State the problem to be proven
theorem regular_octagon_area (r : ℝ) (h : r = inscribed_circle_radius) :
  8 * (1/2 * (side_length r) * r * sin (central_angle / 2 * real.pi / 180)) = 18 * real.sqrt(3) * (2 - real.sqrt(2)) :=
sorry

end regular_octagon_area_l171_171938


namespace common_tangent_lines_l171_171465

theorem common_tangent_lines (m : ℝ) (hm : 0 < m) :
  (∀ x y : ℝ, x^2 + y^2 - (4 * m + 2) * x - 2 * m * y + 4 * m^2 + 4 * m + 1 = 0 →
     (y = 0 ∨ y = 4 / 3 * x - 4 / 3)) :=
by sorry

end common_tangent_lines_l171_171465


namespace find_m_eq_2_l171_171189

theorem find_m_eq_2 (a m : ℝ) (h1 : a > 0) (h2 : -a * m^2 + 2 * a * m + 3 = 3) (h3 : m ≠ 0) : m = 2 :=
by
  sorry

end find_m_eq_2_l171_171189


namespace number_of_valid_strings_l171_171964

def count_valid_strings (n : ℕ) : ℕ :=
  4^n - 3 * 3^n + 3 * 2^n - 1

theorem number_of_valid_strings (n : ℕ) :
  count_valid_strings n = 4^n - 3 * 3^n + 3 * 2^n - 1 :=
by sorry

end number_of_valid_strings_l171_171964


namespace find_median_of_100_l171_171839

noncomputable def median_of_set (s : Finset ℝ) : ℝ :=
if h : ∃ median, is_median s median then classical.some h else 0

def is_median (s : Finset ℝ) (m : ℝ) : Prop :=
∃ (f : ℤ), (f : ℝ) = (card s : ℝ) / 2 ∧
    ∃ (low : Finset ℝ) (high : Finset ℝ),
        low ⊆ s ∧ high ⊆ s ∧
        card low = floor ((card s : ℝ) / 2) ∧
        card high = ceil ((card s : ℝ) / 2) ∧
        (∀ x ∈ low, x ≤ m) ∧ (∀ x ∈ high, x ≥ m)

theorem find_median_of_100 (s : Finset ℝ) (h_size : s.card = 100)
(h1 : ∃ x ∈ s, median_of_set (s.erase x) = 78)
(h2 : ∃ y ∈ s, median_of_set (s.erase y) = 66) :
  median_of_set s = 72 := by
sorry

end find_median_of_100_l171_171839


namespace find_f_of_3_l171_171972

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  x^7 + a*x^5 + b*x - 5

theorem find_f_of_3 (a b : ℝ) (h : f (-3) a b = 5) : f 3 a b = -15 := by
  sorry

end find_f_of_3_l171_171972


namespace special_hash_calculation_l171_171685

-- Definition of the operation #
def special_hash (a b : ℤ) : ℚ := 2 * a + (a / b) + 3

-- Statement of the proof problem
theorem special_hash_calculation : special_hash 7 3 = 19 + 1/3 := 
by 
  sorry

end special_hash_calculation_l171_171685


namespace x_share_of_profit_l171_171406

-- Define the problem conditions
def investment_x : ℕ := 5000
def investment_y : ℕ := 15000
def total_profit : ℕ := 1600

-- Define the ratio simplification
def ratio_x : ℕ := 1
def ratio_y : ℕ := 3
def total_ratio_parts : ℕ := ratio_x + ratio_y

-- Define the profit division per part
def profit_per_part : ℕ := total_profit / total_ratio_parts

-- Lean 4 statement to prove
theorem x_share_of_profit : profit_per_part * ratio_x = 400 := sorry

end x_share_of_profit_l171_171406


namespace range_of_a_for_inequality_l171_171005

open Real

theorem range_of_a_for_inequality (a : ℝ) : 
  (∀ x : ℝ, ¬(a*x^2 - |x + 1| + 2*a < 0)) ↔ a ≥ (sqrt 3 + 1) / 4 := 
by
  sorry

end range_of_a_for_inequality_l171_171005


namespace combined_apples_sold_l171_171727

theorem combined_apples_sold (red_apples green_apples total_apples : ℕ) 
    (h1 : red_apples = 32) 
    (h2 : green_apples = (3 * (32 / 8))) 
    (h3 : total_apples = red_apples + green_apples) : 
    total_apples = 44 :=
by
  sorry

end combined_apples_sold_l171_171727


namespace tickets_difference_vip_general_l171_171583

theorem tickets_difference_vip_general (V G : ℕ) 
  (h1 : V + G = 320) 
  (h2 : 40 * V + 10 * G = 7500) : G - V = 34 := 
by
  sorry

end tickets_difference_vip_general_l171_171583


namespace num_integer_pairs_satisfying_m_plus_n_eq_mn_l171_171058

theorem num_integer_pairs_satisfying_m_plus_n_eq_mn : 
  ∃ (m n : ℤ), (m + n = m * n) ∧ ∀ (m n : ℤ), (m + n = m * n) → 
  (m = 0 ∧ n = 0) ∨ (m = 2 ∧ n = 2) :=
by
  sorry

end num_integer_pairs_satisfying_m_plus_n_eq_mn_l171_171058


namespace cost_of_ice_cream_l171_171740

theorem cost_of_ice_cream (x : ℝ) (h1 : 10 * x = 40) : x = 4 :=
by sorry

end cost_of_ice_cream_l171_171740


namespace triangle_BC_length_l171_171833

theorem triangle_BC_length (A B C X : Type) (AB AC BC BX CX : ℕ)
  (h1 : AB = 75)
  (h2 : AC = 85)
  (h3 : BC = BX + CX)
  (h4 : BX * (BX + CX) = 1600)
  (h5 : BX + CX = 80) :
  BC = 80 :=
by
  sorry

end triangle_BC_length_l171_171833


namespace find_c_l171_171055

-- Defining the variables and conditions given in the problem
variables (a b c : ℝ)

-- Conditions
def vertex_condition : Prop := (2, -3) = (a * (-3)^2 + b * (-3) + c, -3)
def point_condition : Prop := (7, -1) = (a * (-1)^2 + b * (-1) + c, -1)

-- Problem Statement
theorem find_c 
  (h_vertex : vertex_condition a b c)
  (h_point : point_condition a b c) :
  c = 53 / 4 :=
sorry

end find_c_l171_171055


namespace smallest_possible_value_of_N_l171_171826

-- Declares the context and required constraints
theorem smallest_possible_value_of_N (N : ℕ) (h1 : 70 < N) (h2 : 70 ∣ 21 * N) : N = 80 :=
by
  sorry

end smallest_possible_value_of_N_l171_171826


namespace inverse_proportion_quadrants_l171_171324

theorem inverse_proportion_quadrants (k b : ℝ) (h1 : b > 0) (h2 : k < 0) :
  ∀ x : ℝ, (x > 0 → (y = kb / x) → y < 0) ∧ (x < 0 → (y = kb / x) → y > 0) :=
by
  sorry

end inverse_proportion_quadrants_l171_171324


namespace only_pairs_satisfying_conditions_l171_171761

theorem only_pairs_satisfying_conditions (a b : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) :
  (b^2 + b + 1) % a = 0 ∧ (a^2 + a + 1) % b = 0 → a = 1 ∧ b = 1 :=
by
  sorry

end only_pairs_satisfying_conditions_l171_171761


namespace term_in_census_is_population_l171_171052

def term_for_entire_set_of_objects : String :=
  "population"

theorem term_in_census_is_population :
  term_for_entire_set_of_objects = "population" :=
sorry

end term_in_census_is_population_l171_171052


namespace find_radius_l171_171372

noncomputable def radius (π : ℝ) : Prop :=
  ∃ r : ℝ, π * r^2 + 2 * r - 2 * π * r = 12 ∧ r = Real.sqrt (12 / π)

theorem find_radius (π : ℝ) (hπ : π > 0) : 
  radius π :=
sorry

end find_radius_l171_171372


namespace gcd_pair_sum_ge_prime_l171_171666

theorem gcd_pair_sum_ge_prime
  (n : ℕ)
  (h_prime: Prime (2*n - 1))
  (a : Fin n → ℕ)
  (h_distinct: ∀ i j : Fin n, i ≠ j → a i ≠ a j) :
  ∃ i j : Fin n, i ≠ j ∧ (a i + a j) / Nat.gcd (a i) (a j) ≥ 2*n - 1 := sorry

end gcd_pair_sum_ge_prime_l171_171666


namespace correct_answer_l171_171556

variable (x : ℝ)

theorem correct_answer : {x : ℝ | x^2 + 2*x + 1 = 0} = {-1} :=
by sorry -- the actual proof is not required, just the statement

end correct_answer_l171_171556


namespace cost_per_bag_l171_171220

-- Definitions and variables based on the conditions
def sandbox_length : ℝ := 3  -- Sandbox length in feet
def sandbox_width : ℝ := 3   -- Sandbox width in feet
def bag_area : ℝ := 3        -- Area of one bag of sand in square feet
def total_cost : ℝ := 12     -- Total cost to fill up the sandbox in dollars

-- Statement to prove
theorem cost_per_bag : (total_cost / (sandbox_length * sandbox_width / bag_area)) = 4 :=
by
  sorry

end cost_per_bag_l171_171220


namespace special_blend_probability_l171_171286

/-- Define the probability variables and conditions -/
def visit_count : ℕ := 6
def special_blend_prob : ℚ := 3 / 4
def non_special_blend_prob : ℚ := 1 / 4

/-- The binomial coefficient for choosing 5 days out of 6 -/
def choose_6_5 : ℕ := Nat.choose 6 5

/-- The probability of serving the special blend exactly 5 times out of 6 -/
def prob_special_blend_5 : ℚ := (choose_6_5 : ℚ) * (special_blend_prob ^ 5) * (non_special_blend_prob ^ 1)

/-- Statement to prove the desired probability -/
theorem special_blend_probability :
  prob_special_blend_5 = 1458 / 4096 :=
by
  sorry

end special_blend_probability_l171_171286


namespace polygon_sides_l171_171326

theorem polygon_sides (x : ℕ) (h1 : 180 * (x - 2) = 3 * 360) : x = 8 := by
  sorry

end polygon_sides_l171_171326


namespace g_of_g_of_g_of_20_l171_171502

def g (x : ℕ) : ℕ :=
  if x < 10 then x^2 - 9 else x - 15

theorem g_of_g_of_g_of_20 : g (g (g 20)) = 1 := by
  -- Proof steps would go here
  sorry

end g_of_g_of_g_of_20_l171_171502


namespace largest_multiples_of_3_is_9999_l171_171907

theorem largest_multiples_of_3_is_9999 :
  ∃ n : ℕ, (n = 9999 ∧ n < 10000 ∧ 1000 ≤ n ∧ 3 ∣ n) ∧ 
  (∀ k : ℕ, (k < 10000 ∧ 1000 ≤ k ∧ 3 ∣ k) → k ≤ n) :=
by
  sorry

end largest_multiples_of_3_is_9999_l171_171907


namespace smallest_value_of_3a_plus_2_l171_171638

theorem smallest_value_of_3a_plus_2 (a : ℝ) (h : 8 * a^2 + 10 * a + 6 = 2) : 
  ∃ (x : ℝ), x = 3 * a + 2 ∧ x = -1 :=
by
  sorry

end smallest_value_of_3a_plus_2_l171_171638


namespace smallest_N_value_proof_l171_171804

def smallest_value_N (N : ℕ) : Prop :=
  N > 70 ∧ (21 * N) % 70 = 0

theorem smallest_N_value_proof : ∃ N, smallest_value_N N ∧ (∀ M, smallest_value_N M → N ≤ M) :=
  sorry

end smallest_N_value_proof_l171_171804


namespace derivative_f_at_2_l171_171184

noncomputable def f (x : ℝ) : ℝ := (x + 1) * (x - 1)

theorem derivative_f_at_2 : (deriv f 2) = 4 := by
  sorry

end derivative_f_at_2_l171_171184


namespace green_marbles_l171_171303

theorem green_marbles 
  (total_marbles : ℕ)
  (red_marbles : ℕ)
  (at_least_blue_marbles : ℕ)
  (h1 : total_marbles = 63) 
  (h2 : at_least_blue_marbles ≥ total_marbles / 3) 
  (h3 : red_marbles = 38) 
  : ∃ green_marbles : ℕ, total_marbles - red_marbles - at_least_blue_marbles = green_marbles ∧ green_marbles = 4 :=
by
  sorry

end green_marbles_l171_171303


namespace value_of_e_over_f_l171_171775

theorem value_of_e_over_f 
    (a b c d e f : ℝ) 
    (h1 : a * b * c = 1.875 * d * e * f)
    (h2 : a / b = 5 / 2)
    (h3 : b / c = 1 / 2)
    (h4 : c / d = 1)
    (h5 : d / e = 3 / 2) : 
    e / f = 1 / 3 :=
by
  sorry

end value_of_e_over_f_l171_171775


namespace find_missing_employee_l171_171428

-- Definitions based on the problem context
def employee_numbers : List Nat := List.range (52)
def sample_size := 4

-- The given conditions, stating that these employees are in the sample
def in_sample (x : Nat) : Prop := x = 6 ∨ x = 32 ∨ x = 45 ∨ x = 19

-- Define systematic sampling method condition
def systematic_sample (nums : List Nat) (size interval : Nat) : Prop :=
  nums = List.map (fun i => 6 + i * interval % 52) (List.range size)

-- The employees in the sample must include 6
def start_num := 6
def interval := 13
def expected_sample := [6, 19, 32, 45]

-- The Lean theorem we need to prove
theorem find_missing_employee :
  systematic_sample expected_sample sample_size interval ∧
  in_sample 6 ∧ in_sample 32 ∧ in_sample 45 →
  in_sample 19 :=
by
  sorry

end find_missing_employee_l171_171428


namespace integer_pair_condition_l171_171981

theorem integer_pair_condition (m n : ℤ) (h : (m^2 + m * n + n^2 : ℚ) / (m + 2 * n) = 13 / 3) : m + 2 * n = 9 :=
sorry

end integer_pair_condition_l171_171981


namespace find_a_plus_b_l171_171631

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + b * Real.log x

theorem find_a_plus_b (a b : ℝ) :
  (∃ x : ℝ, x = 1 ∧ f a b x = 1 / 2 ∧ (deriv (f a b)) 1 = 0) →
  a + b = -1/2 :=
by
  sorry

end find_a_plus_b_l171_171631


namespace gasVolume_at_20_l171_171458

variable (V : ℕ → ℕ)

/-- Given conditions:
 1. The gas volume expands by 3 cubic centimeters for every 5 degree rise in temperature.
 2. The volume is 30 cubic centimeters when the temperature is 30 degrees.
  -/
def gasVolume : Prop :=
  (∀ T ΔT, ΔT = 5 → V (T + ΔT) = V T + 3) ∧ V 30 = 30

theorem gasVolume_at_20 :
  gasVolume V → V 20 = 24 :=
by
  intro h
  -- Proof steps would go here.
  sorry

end gasVolume_at_20_l171_171458


namespace circumcircle_radius_of_right_triangle_l171_171724

theorem circumcircle_radius_of_right_triangle (a b c : ℝ) (h1: a = 8) (h2: b = 6) (h3: c = 10) (h4: a^2 + b^2 = c^2) : (c / 2) = 5 := 
by
  sorry

end circumcircle_radius_of_right_triangle_l171_171724


namespace solution_sum_l171_171317

theorem solution_sum (m n : ℝ) (h₀ : m ≠ 0) (h₁ : m^2 + m * n - m = 0) : m + n = 1 := 
by 
  sorry

end solution_sum_l171_171317


namespace least_positive_integer_condition_l171_171265

theorem least_positive_integer_condition :
  ∃ b : ℕ, b > 0 ∧
    b % 3 = 2 ∧
    b % 5 = 4 ∧
    b % 6 = 5 ∧
    b % 7 = 6 ∧
    ∀ n : ℕ, (n > 0 ∧ n % 3 = 2 ∧ n % 5 = 4 ∧ n % 6 = 5 ∧ n % 7 = 6) → n ≥ b :=
    ∃ b : ℕ, b = 209 := sorry

end least_positive_integer_condition_l171_171265


namespace median_of_100_numbers_l171_171840

theorem median_of_100_numbers 
  (numbers : List ℝ)
  (h_len : numbers.length = 100)
  (h_median_99_1 : ∀ num ∈ numbers, median (numbers.erase num) = 78 → num ∈ numbers)
  (h_median_99_2 : ∀ num ∈ numbers, median (numbers.erase num) = 66 → num ∈ numbers) :
  median numbers = 72 :=
sorry

end median_of_100_numbers_l171_171840


namespace intersection_with_x_axis_l171_171613

noncomputable def f (x : ℝ) : ℝ := 
  (3 * x - 1) * (Real.sqrt (9 * x^2 - 6 * x + 5) + 1) + 
  (2 * x - 3) * (Real.sqrt (4 * x^2 - 12 * x + 13)) + 1

theorem intersection_with_x_axis :
  ∃ x : ℝ, f x = 0 ∧ x = 4 / 5 :=
by
  sorry

end intersection_with_x_axis_l171_171613


namespace sin_90_deg_l171_171145

theorem sin_90_deg : Real.sin (90 * Real.pi / 180) = 1 := 
by
  sorry

end sin_90_deg_l171_171145


namespace smallest_possible_value_of_N_l171_171807

theorem smallest_possible_value_of_N :
  ∃ N : ℕ, N > 70 ∧ (21 * N) % 70 = 0 ∧ N = 80 :=
by
  existsi 80
  split
  · norm_num
  split
  · norm_num
  sorry

end smallest_possible_value_of_N_l171_171807


namespace bob_total_questions_l171_171602

theorem bob_total_questions (q1 q2 q3 : ℕ) : 
  q1 = 13 ∧ q2 = 2 * q1 ∧ q3 = 2 * q2 → q1 + q2 + q3 = 91 :=
by
  intros
  sorry

end bob_total_questions_l171_171602


namespace magic_square_S_divisible_by_3_l171_171212

-- Definitions of the 3x3 magic square conditions
def is_magic_square (a : ℕ → ℕ → ℤ) (S : ℤ) : Prop :=
  (a 0 0 + a 0 1 + a 0 2 = S) ∧
  (a 1 0 + a 1 1 + a 1 2 = S) ∧
  (a 2 0 + a 2 1 + a 2 2 = S) ∧
  (a 0 0 + a 1 0 + a 2 0 = S) ∧
  (a 0 1 + a 1 1 + a 2 1 = S) ∧
  (a 0 2 + a 1 2 + a 2 2 = S) ∧
  (a 0 0 + a 1 1 + a 2 2 = S) ∧
  (a 0 2 + a 1 1 + a 2 0 = S)

-- Main theorem statement
theorem magic_square_S_divisible_by_3 :
  ∀ (a : ℕ → ℕ → ℤ) (S : ℤ),
    is_magic_square a S →
    S % 3 = 0 :=
by
  -- Here we assume the existence of the proof
  sorry

end magic_square_S_divisible_by_3_l171_171212


namespace evaluate_expression_l171_171223

noncomputable def a : ℝ := 2 * Real.sqrt 2 + 3 * Real.sqrt 3 + 4 * Real.sqrt 6
noncomputable def b : ℝ := -2 * Real.sqrt 2 + 3 * Real.sqrt 3 + 4 * Real.sqrt 6
noncomputable def c : ℝ := 2 * Real.sqrt 2 - 3 * Real.sqrt 3 + 4 * Real.sqrt 6
noncomputable def d : ℝ := -2 * Real.sqrt 2 - 3 * Real.sqrt 3 + 4 * Real.sqrt 6

theorem evaluate_expression : (1/a + 1/b + 1/c + 1/d)^2 = 952576 / 70225 := by
  sorry

end evaluate_expression_l171_171223


namespace tshirts_per_package_l171_171230

-- Definitions based on the conditions
def total_tshirts : ℕ := 70
def num_packages : ℕ := 14

-- Theorem to prove the number of t-shirts per package
theorem tshirts_per_package : total_tshirts / num_packages = 5 := by
  -- The proof is omitted, only the statement is provided as required.
  sorry

end tshirts_per_package_l171_171230


namespace player1_wins_11th_round_l171_171902

theorem player1_wins_11th_round (player1_wins_first_10 : ∀ (round : ℕ), round < 10 → player1_wins round) : 
  prob_winning_11th_round player1 = 11 / 12 :=
sorry

end player1_wins_11th_round_l171_171902


namespace smallest_solution_l171_171072

theorem smallest_solution : ∃ x : ℝ, x^4 - 50 * x^2 + 625 = 0 ∧ ∀ y : ℝ, y^4 - 50 * y^2 + 625 = 0 → x ≤ y :=
by {
  use -5,
  split,
  sorry, -- here would be the proof that (-5)^4 - 50 * (-5)^2 + 625 = 0
  intros y hy,
  sorry -- here would be the proof that for any y such that y^4 - 50 * y^2 + 625 = 0, -5 ≤ y
}

end smallest_solution_l171_171072


namespace ratio_y_to_x_l171_171438

-- Define the setup as given in the conditions
variables (c x y : ℝ)

-- Condition 1: Selling price x results in a loss of 20%
def condition1 : Prop := x = 0.80 * c

-- Condition 2: Selling price y results in a profit of 25%
def condition2 : Prop := y = 1.25 * c

-- Theorem: Prove the ratio of y to x is 25/16 given the conditions
theorem ratio_y_to_x (c : ℝ) (h1 : condition1 c x) (h2 : condition2 c y) : y / x = 25 / 16 := 
sorry

end ratio_y_to_x_l171_171438


namespace final_state_probability_l171_171013

-- Define the initial state and conditions of the problem
structure GameState where
  raashan : ℕ
  sylvia : ℕ
  ted : ℕ
  uma : ℕ

-- Conditions: each player starts with $2, and the game evolves over 500 rounds
def initial_state : GameState :=
  { raashan := 2, sylvia := 2, ted := 2, uma := 2 }

def valid_statements (state : GameState) : Prop :=
  state.raashan = 2 ∧ state.sylvia = 2 ∧ state.ted = 2 ∧ state.uma = 2

-- Final theorem statement
theorem final_state_probability :
  let states := 500 -- representing the number of rounds
  -- proof outline implies that after the games have properly transitioned and bank interactions, the probability is calculated
  -- state after the transitions
  ∃ (prob : ℚ), prob = 1/4 ∧ valid_statements initial_state :=
  sorry

end final_state_probability_l171_171013


namespace Paul_average_homework_l171_171757

theorem Paul_average_homework :
  let weeknights := 5,
      weekend_homework := 5,
      night_homework := 2,
      nights_no_homework := 2,
      days_in_week := 7,
      total_homework := weekend_homework + night_homework * weeknights,
      available_nights := days_in_week - nights_no_homework,
      average_homework_per_night := total_homework / available_nights
  in average_homework_per_night = 3 := 
by
  sorry

end Paul_average_homework_l171_171757


namespace total_students_l171_171099

theorem total_students (a : ℕ) (h1: (71 * ((3480 - 69 * a) / 2) + 69 * (a - (3480 - 69 * a) / 2)) = 3480) : a = 50 :=
by
  -- Proof to be provided here
  sorry

end total_students_l171_171099


namespace real_estate_commission_l171_171102

theorem real_estate_commission (r : ℝ) (P : ℝ) (C : ℝ) (h : r = 0.06) (hp : P = 148000) : C = P * r :=
by
  -- Definitions and proof steps will go here.
  sorry

end real_estate_commission_l171_171102


namespace weight_computation_requires_initial_weight_l171_171829

-- Let's define the conditions
variable (initial_weight : ℕ) -- The initial weight of the pet; needs to be provided
def yearly_gain := 11  -- The pet gains 11 pounds each year
def age := 8  -- The pet is 8 years old

-- Define the goal to be proved
def current_weight_computable : Prop :=
  initial_weight ≠ 0 → initial_weight + (yearly_gain * age) ≠ 0

-- State the theorem
theorem weight_computation_requires_initial_weight : ¬ ∃ current_weight, initial_weight + (yearly_gain * age) = current_weight :=
by {
  sorry
}

end weight_computation_requires_initial_weight_l171_171829


namespace find_n_for_integer_roots_l171_171315

theorem find_n_for_integer_roots (n : ℤ):
    (∃ x y : ℤ, x ≠ y ∧ x^2 + (n+1)*x + (2*n - 1) = 0 ∧ y^2 + (n+1)*y + (2*n - 1) = 0) →
    (n = 1 ∨ n = 5) :=
sorry

end find_n_for_integer_roots_l171_171315


namespace smallest_N_l171_171821

theorem smallest_N (N : ℕ) (hN : N > 70) (hDiv : 70 ∣ 21 * N) : N = 80 :=
sorry

end smallest_N_l171_171821


namespace algebra_identity_l171_171307

theorem algebra_identity (x y : ℝ) (h1 : x + y = 2) (h2 : x - y = 4) : x^2 - y^2 = 8 := by
  sorry

end algebra_identity_l171_171307


namespace combined_tax_rate_correct_l171_171719

noncomputable def combined_tax_rate (income_john income_ingrid tax_rate_john tax_rate_ingrid : ℝ) : ℝ :=
  let tax_john := tax_rate_john * income_john
  let tax_ingrid := tax_rate_ingrid * income_ingrid
  let total_tax := tax_john + tax_ingrid
  let combined_income := income_john + income_ingrid
  total_tax / combined_income * 100

theorem combined_tax_rate_correct :
  combined_tax_rate 56000 74000 0.30 0.40 = 35.69 := by
  sorry

end combined_tax_rate_correct_l171_171719


namespace factorial_ends_with_base_8_zeroes_l171_171990

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def highestPowerOfFactorInFactorial (n p : ℕ) : ℕ :=
  if p = 1 then n else
  Nat.div (n - 1) (p - 1)

theorem factorial_ends_with_base_8_zeroes (n : ℕ) : 
  highestPowerOfFactorInFactorial 15 8 = 3 := 
sorry

end factorial_ends_with_base_8_zeroes_l171_171990


namespace total_ducks_and_ducklings_l171_171860

theorem total_ducks_and_ducklings :
  let ducks1 := 2
  let ducklings1 := 5
  let ducks2 := 6
  let ducklings2 := 3
  let ducks3 := 9
  let ducklings3 := 6
  (ducks1 + ducks2 + ducks3) + (ducks1 * ducklings1 + ducks2 * ducklings2 + ducks3 * ducklings3) = 99 :=
by
  sorry

end total_ducks_and_ducklings_l171_171860


namespace determine_m_l171_171190

theorem determine_m (a m : ℝ) (h : a > 0) (h2 : (m, 3) ∈ set_of (λ p : ℝ × ℝ, p.2 = -a * p.1 ^ 2 + 2 * a * p.1 + 3)) (h3 : m ≠ 0) : m = 2 :=
sorry

end determine_m_l171_171190


namespace hundreds_digit_of_25_fact_minus_20_fact_l171_171548

theorem hundreds_digit_of_25_fact_minus_20_fact : (25! - 20!) % 1000 / 100 = 0 := 
  sorry

end hundreds_digit_of_25_fact_minus_20_fact_l171_171548


namespace median_of_100_set_l171_171838

theorem median_of_100_set 
  (S : Finset ℝ) (h_card : S.card = 100)
  (h_remove1 : ∃ x ∈ S, median (S.erase x) = 78)
  (h_remove2 : ∃ y ∈ S, median (S.erase y) = 66) : 
  median S = 72 :=
by
  sorry

end median_of_100_set_l171_171838


namespace correct_fourth_number_correct_eighth_number_l171_171264

-- Condition: Initial number on the board and sequence of medians
def initial_board : List ℝ := [1]
def medians : List ℝ := [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5]

-- The number written fourth is 2
def fourth_number_written (board : List ℝ) : ℝ := 2

-- The number written eighth is also 2
def eighth_number_written (board : List ℝ) : ℝ := 2

-- Formalizing the conditions and assertions
theorem correct_fourth_number :
  ∃ board : List ℝ, 
    board.head = 1 ∧ 
    -- Assume the sequence of medians can be calculated from the board
    (calculate_medians_from_board board = medians) ∧
    fourth_number_written board = 2 := 
sorry

theorem correct_eighth_number :
  ∃ board : List ℝ, 
    board.head = 1 ∧ 
    -- Assume the sequence of medians can be calculated from the board
    (calculate_medians_from_board board = medians) ∧
    eighth_number_written board = 2 := 
sorry

-- Function to calculate medians from the board (to be implemented)
noncomputable def calculate_medians_from_board (board : List ℝ) : List ℝ := sorry

end correct_fourth_number_correct_eighth_number_l171_171264


namespace average_age_of_students_l171_171330

variable (A : ℕ) -- We define A as a natural number representing average age

-- Define the conditions
def num_students : ℕ := 32
def staff_age : ℕ := 49
def new_average_age := A + 1

-- Definition of total age including the staff
def total_age_with_staff := 33 * new_average_age

-- Original condition stated as an equality
def condition : Prop := num_students * A + staff_age = total_age_with_staff

-- Theorem statement asserting that the average age A is 16 given the condition
theorem average_age_of_students : condition A → A = 16 :=
by sorry

end average_age_of_students_l171_171330


namespace find_m_l171_171192

variable (a m x : ℝ)

noncomputable def quadratic_function : ℝ → ℝ := λ x, -a * x^2 + 2 * a * x + 3

theorem find_m (h1 : a > 0) (h2 : quadratic_function a m = 3) (h3 : m ≠ 0) : m = 2 := 
sorry

end find_m_l171_171192


namespace sequence_may_or_may_not_be_arithmetic_l171_171481

theorem sequence_may_or_may_not_be_arithmetic (a : ℕ → ℕ) 
  (h1 : a 0 = 1) (h2 : a 1 = 2) (h3 : a 2 = 3) 
  (h4 : a 3 = 4) (h5 : a 4 = 5) : 
  ¬(∀ n, a (n + 1) - a n = 1) → 
  (∀ n, a (n + 1) - a n = 1) ∨ ¬(∀ n, a (n + 1) - a n = 1) :=
by
  sorry

end sequence_may_or_may_not_be_arithmetic_l171_171481


namespace total_canoes_built_by_End_of_May_l171_171734

noncomputable def total_canoes_built (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

theorem total_canoes_built_by_End_of_May :
  total_canoes_built 7 2 5 = 217 :=
by
  -- The proof would go here.
  sorry

end total_canoes_built_by_End_of_May_l171_171734


namespace ratio_of_a_to_b_l171_171092

variable (a b c d : ℝ)

theorem ratio_of_a_to_b (h1 : c = 0.20 * a) (h2 : c = 0.10 * b) : a = (1 / 2) * b :=
by
  sorry

end ratio_of_a_to_b_l171_171092


namespace find_difference_l171_171674

variable (d : ℕ) (A B : ℕ)
open Nat

theorem find_difference (hd : d > 7)
  (hAB : d * A + B + d * A + A = d * d + 7 * d + 4)  (hA_gt_B : A > B):
  A - B = 3 :=
sorry

end find_difference_l171_171674


namespace last_score_is_65_l171_171646

-- Define the scores and the problem conditions
def scores := [65, 72, 75, 80, 85, 88, 92]
def total_sum := 557
def remaining_sum (score : ℕ) : ℕ := total_sum - score

-- Define a property to check divisibility
def divisible_by (n d : ℕ) : Prop := n % d = 0

-- The main theorem statement
theorem last_score_is_65 :
  (∀ s ∈ scores, divisible_by (remaining_sum s) 6) ∧ divisible_by total_sum 7 ↔ scores = [65, 72, 75, 80, 85, 88, 92] :=
sorry

end last_score_is_65_l171_171646


namespace midpoint_of_interception_l171_171166

theorem midpoint_of_interception (x1 x2 y1 y2 : ℝ) 
  (h1 : y1^2 = 4 * x1) 
  (h2 : y2^2 = 4 * x2) 
  (h3 : y1 = x1 - 1) 
  (h4 : y2 = x2 - 1) : 
  ( (x1 + x2) / 2, (y1 + y2) / 2 ) = (3, 2) :=
by 
  sorry

end midpoint_of_interception_l171_171166


namespace find_q_l171_171986

-- Define the conditions and the statement to prove
theorem find_q (p q : ℝ) (hp1 : p > 1) (hq1 : q > 1) 
  (h1 : 1 / p + 1 / q = 3 / 2)
  (h2 : p * q = 9) : q = 6 := 
sorry

end find_q_l171_171986


namespace smaller_number_in_ratio_l171_171395

noncomputable def LCM (a b : ℕ) : ℕ := (a * b) / Nat.gcd a b

theorem smaller_number_in_ratio (x : ℕ) (a b : ℕ) (h1 : a = 4 * x) (h2 : b = 5 * x) (h3 : LCM a b = 180) : a = 36 := 
by
  sorry

end smaller_number_in_ratio_l171_171395


namespace negation_of_exists_statement_l171_171882

theorem negation_of_exists_statement :
  ¬ (∃ x0 : ℝ, x0 > 0 ∧ x0^2 - 5 * x0 + 6 > 0) ↔ ∀ x : ℝ, x > 0 → x^2 - 5 * x + 6 ≤ 0 :=
by
  sorry

end negation_of_exists_statement_l171_171882


namespace sin_90_eq_1_l171_171135

theorem sin_90_eq_1 :
  let θ := 90 : ℝ in
  let cos_θ := real.cos θ in
  let sin_θ := real.sin θ in 
  let rotation_matrix := ![![cos_θ, -sin_θ], ![sin_θ, cos_θ]] in
  let point := ![1, 0] in
  let rotated_point := matrix.mul_vec rotation_matrix point in
  rotated_point = ![0, 1] → 
  sin_θ = 1 :=
by
  sorry

end sin_90_eq_1_l171_171135


namespace triangle_third_side_length_l171_171783

theorem triangle_third_side_length (a b : ℝ) (x : ℝ) (h₁ : a = 5) (h₂ : b = 8) (hx : x ∈ {2, 3, 6, 13}) :
  3 < x ∧ x < 13 → x = 6 :=
by sorry

end triangle_third_side_length_l171_171783


namespace sum_of_reciprocals_l171_171389

theorem sum_of_reciprocals (a b : ℝ) (h_sum : a + b = 15) (h_prod : a * b = 225) :
  (1 / a) + (1 / b) = 1 / 15 :=
by 
  sorry

end sum_of_reciprocals_l171_171389


namespace shelves_used_l171_171107

-- Define the initial conditions
def initial_stock : Float := 40.0
def additional_stock : Float := 20.0
def books_per_shelf : Float := 4.0

-- Define the total number of books
def total_books : Float := initial_stock + additional_stock

-- Define the number of shelves
def number_of_shelves : Float := total_books / books_per_shelf

-- The proof statement that needs to be proven
theorem shelves_used : number_of_shelves = 15.0 :=
by
  -- The proof will go here
  sorry

end shelves_used_l171_171107


namespace minimum_time_needed_l171_171401

-- Define the task times
def review_time : ℕ := 30
def rest_time : ℕ := 30
def boil_water_time : ℕ := 15
def homework_time : ℕ := 25

-- Define the minimum time required (Xiao Ming can boil water while resting)
theorem minimum_time_needed : review_time + rest_time + homework_time = 85 := by
  -- The proof is omitted with sorry
  sorry

end minimum_time_needed_l171_171401


namespace factor_expression_l171_171611

theorem factor_expression (x : ℝ) : 
  3 * x * (x - 5) + 4 * (x - 5) = (3 * x + 4) * (x - 5) :=
by
  sorry

end factor_expression_l171_171611


namespace isabel_earnings_l171_171411

theorem isabel_earnings :
  ∀ (bead_necklaces gem_necklaces cost_per_necklace : ℕ),
    bead_necklaces = 3 →
    gem_necklaces = 3 →
    cost_per_necklace = 6 →
    (bead_necklaces + gem_necklaces) * cost_per_necklace = 36 := by
sorry

end isabel_earnings_l171_171411


namespace systematic_sampling_first_group_l171_171397

theorem systematic_sampling_first_group 
  (total_students sample_size group_size group_number drawn_number : ℕ)
  (h1 : total_students = 160)
  (h2 : sample_size = 20)
  (h3 : total_students = sample_size * group_size)
  (h4 : group_number = 16)
  (h5 : drawn_number = 126) 
  : (drawn_lots_first_group : ℕ) 
      = ((drawn_number - ((group_number - 1) * group_size + 1)) + 1) :=
sorry


end systematic_sampling_first_group_l171_171397


namespace exponent_rule_example_l171_171950

theorem exponent_rule_example {a : ℝ} : (a^3)^4 = a^12 :=
by {
  sorry
}

end exponent_rule_example_l171_171950


namespace sum_of_three_integers_mod_53_l171_171710

theorem sum_of_three_integers_mod_53 (a b c : ℕ) (h1 : a % 53 = 31) 
                                     (h2 : b % 53 = 22) (h3 : c % 53 = 7) : 
                                     (a + b + c) % 53 = 7 :=
by
  sorry

end sum_of_three_integers_mod_53_l171_171710


namespace remainder_div_741147_6_l171_171550

theorem remainder_div_741147_6 : 741147 % 6 = 3 :=
by
  sorry

end remainder_div_741147_6_l171_171550


namespace range_of_a_l171_171630

-- Defining the function f(x)
def f (a x : ℝ) := x^2 + (a^2 - 1) * x + (a - 2)

-- The statement of the problem in Lean 4
theorem range_of_a (a : ℝ) (h : ∃ x1 x2 : ℝ, x1 > 1 ∧ x2 < 1 ∧ f a x1 = 0 ∧ f a x2 = 0) : -2 < a ∧ a < 1 :=
by
  sorry -- Proof is omitted

end range_of_a_l171_171630


namespace common_ratio_of_geometric_progression_l171_171331

theorem common_ratio_of_geometric_progression (a1 q : ℝ) (S3 : ℝ) (a2 : ℝ)
  (h1 : S3 = a1 * (1 + q + q^2))
  (h2 : a2 = a1 * q)
  (h3 : a2 + S3 = 0) :
  q = -1 := 
  sorry

end common_ratio_of_geometric_progression_l171_171331


namespace greatest_integer_equality_l171_171970

theorem greatest_integer_equality (m : ℝ) (h : m ≥ 3) :
  Int.floor ((m * (m + 1)) / (2 * (2 * m - 1))) = Int.floor ((m + 1) / 4) :=
  sorry

end greatest_integer_equality_l171_171970


namespace number_of_mappings_A_to_B_number_of_mappings_B_to_A_l171_171890

theorem number_of_mappings_A_to_B (A B : Finset ℕ) (hA : A.card = 5) (hB : B.card = 4) :
  (B.card ^ A.card) = 4^5 :=
by sorry

theorem number_of_mappings_B_to_A (A B : Finset ℕ) (hA : A.card = 5) (hB : B.card = 4) :
  (A.card ^ B.card) = 5^4 :=
by sorry

end number_of_mappings_A_to_B_number_of_mappings_B_to_A_l171_171890


namespace number_of_trailing_zeroes_base8_l171_171999

theorem number_of_trailing_zeroes_base8 (n : ℕ) (hn : n = 15) : 
  (trailing_zeroes_base8 (factorial 15)) = 3 := 
by
  sorry

end number_of_trailing_zeroes_base8_l171_171999


namespace max_gcd_l171_171601

theorem max_gcd (n : ℕ) (h : 0 < n) : ∀ n, ∃ d ≥ 1, d ∣ 13 * n + 4 ∧ d ∣ 8 * n + 3 → d ≤ 9 :=
begin
  sorry
end

end max_gcd_l171_171601


namespace households_subscribing_to_F_l171_171644

theorem households_subscribing_to_F
  (x y : ℕ)
  (hx : x ≥ 1)
  (h_subscriptions : 1 + 4 + 2 + 2 + 2 + y = 2 + 2 + 4 + 3 + 5 + x)
  : y = 6 :=
sorry

end households_subscribing_to_F_l171_171644


namespace distinct_digits_mean_l171_171678

theorem distinct_digits_mean (M : ℕ) :
  (∀ n, n ∈ {9, 99, 999, 9999, 99999, 999999, 9999999, 99999999, 999999999} → M = (9 + 99 + 999 + 9999 + 99999 + 999999 + 9999999 + 99999999 + 999999999) / 9) →
  M = 123456789 ∧ (∀ d : ℕ, d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} → d ≠ 0 → d ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) :=
by 
  sorry

end distinct_digits_mean_l171_171678


namespace garbage_bill_problem_l171_171228

theorem garbage_bill_problem
  (R : ℝ)
  (trash_bins : ℝ := 2)
  (recycling_bins : ℝ := 1)
  (weekly_trash_cost_per_bin : ℝ := 10)
  (weeks_per_month : ℝ := 4)
  (discount_rate : ℝ := 0.18)
  (fine : ℝ := 20)
  (final_bill : ℝ := 102) :
  (trash_bins * weekly_trash_cost_per_bin * weeks_per_month + recycling_bins * R * weeks_per_month)
  - discount_rate * (trash_bins * weekly_trash_cost_per_bin * weeks_per_month + recycling_bins * R * weeks_per_month)
  + fine = final_bill →
  R = 5 := 
by
  sorry

end garbage_bill_problem_l171_171228


namespace find_m_l171_171186

theorem find_m (a m : ℝ) (h_pos : a > 0) (h_points : (m, 3) ∈ set_of (λ x : ℝ × ℝ, ∃ x_val : ℝ, x.snd = -a * (x_val)^2 + 2 * a * x_val + 3)) (h_non_zero : m ≠ 0) : m = 2 := 
sorry

end find_m_l171_171186


namespace c_10_value_l171_171658

def c : ℕ → ℤ
| 0 => 3
| 1 => 9
| (n + 1) => c n * c (n - 1)

theorem c_10_value : c 10 = 3^89 :=
by
  sorry

end c_10_value_l171_171658


namespace sin_90_eq_one_l171_171151

-- Definition of the rotation by 90 degrees counterclockwise
def rotate90 (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, p.1)

-- Definition of the sine function for a 90 degree angle
def sin90 : ℝ :=
  let initial_point := (1, 0)
  let rotated_point := rotate90 initial_point
  rotated_point.2

-- Theorem to be proven: sin90 should be equal to 1
theorem sin_90_eq_one : sin90 = 1 :=
by
  sorry

end sin_90_eq_one_l171_171151


namespace sum_of_prime_factors_of_91_l171_171267

theorem sum_of_prime_factors_of_91 : 
  (¬ (91 % 2 = 0)) ∧ 
  (¬ (91 % 3 = 0)) ∧ 
  (¬ (91 % 5 = 0)) ∧ 
  (91 = 7 * 13) →
  (7 + 13 = 20) := 
by 
  intros h
  sorry

end sum_of_prime_factors_of_91_l171_171267


namespace octagon_area_correct_l171_171935

-- Define the radius of the circle
def radius : ℝ := 3
-- Define the expected area of the regular octagon
def expected_area : ℝ := 54 * (2 - Real.sqrt 2) * Real.sqrt 3

-- The Lean 4 theorem statement
theorem octagon_area_correct : 
  ∀ (R : ℝ) (hR : R = radius), 
  ∃ (A : ℝ), A = expected_area := 
by
  intro R hR
  use expected_area
  sorry

end octagon_area_correct_l171_171935


namespace product_of_last_two_digits_l171_171204

theorem product_of_last_two_digits (A B : ℕ) (h1 : A + B = 12) (h2 : (10 * A + B) % 8 = 0) : A * B = 32 :=
by
  sorry

end product_of_last_two_digits_l171_171204


namespace two_digit_number_is_24_l171_171538

-- Definitions from the problem conditions
def is_two_digit_number (n : ℕ) := n ≥ 10 ∧ n < 100

def tens_digit (n : ℕ) := n / 10

def ones_digit (n : ℕ) := n % 10

def condition_2 (n : ℕ) := tens_digit n = ones_digit n - 2

def condition_3 (n : ℕ) := 3 * tens_digit n * ones_digit n = n

-- The proof problem statement
theorem two_digit_number_is_24 (n : ℕ) (h1 : is_two_digit_number n)
  (h2 : condition_2 n) (h3 : condition_3 n) : n = 24 := by
  sorry

end two_digit_number_is_24_l171_171538


namespace root_equation_alpha_beta_property_l171_171004

theorem root_equation_alpha_beta_property {α β : ℝ} (h1 : α^2 + α - 1 = 0) (h2 : β^2 + β - 1 = 0) :
    α^2 + 2 * β^2 + β = 4 :=
by
  sorry

end root_equation_alpha_beta_property_l171_171004


namespace prove_a_zero_l171_171684

-- Define two natural numbers a and b
variables (a b : ℕ)

-- Condition: For every natural number n, 2^n * a + b is a perfect square
def condition := ∀ n : ℕ, ∃ k : ℕ, 2^n * a + b = k^2

-- Statement to prove: a = 0
theorem prove_a_zero (h : condition a b) : a = 0 := sorry

end prove_a_zero_l171_171684


namespace vehicle_height_limit_l171_171430

theorem vehicle_height_limit (h : ℝ) (sign : String) (cond : sign = "Height Limit 4.5 meters") : h ≤ 4.5 :=
sorry

end vehicle_height_limit_l171_171430


namespace fourth_number_is_2_eighth_number_is_2_l171_171262

-- Conditions as given in the problem
def initial_board := [1]

/-- Medians recorded in Mitya's notebook for the first 10 numbers -/
def medians := [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5]

/-- Prove that the fourth number written on the board is 2 given initial conditions. -/
theorem fourth_number_is_2 (board : ℕ → ℤ)  
  (h1 : board 0 = 1)
  (h2 : medians = [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5])
  : board 3 = 2 :=
sorry

/-- Prove that the eighth number written on the board is 2 given initial conditions. -/
theorem eighth_number_is_2 (board : ℕ → ℤ) 
  (h1 : board 0 = 1)
  (h2 : medians = [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5])
  : board 7 = 2 :=
sorry

end fourth_number_is_2_eighth_number_is_2_l171_171262


namespace two_zeros_range_l171_171470

noncomputable def f (x k : ℝ) : ℝ := x * Real.exp x - k

theorem two_zeros_range (k : ℝ) : -1 / Real.exp 1 < k ∧ k < 0 → ∃! x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 k = 0 ∧ f x2 k = 0 :=
by
  sorry

end two_zeros_range_l171_171470


namespace semicircle_area_increase_l171_171281

noncomputable def area_semicircle (r : ℝ) : ℝ :=
  (1 / 2) * Real.pi * r^2

noncomputable def percent_increase (initial final : ℝ) : ℝ :=
  ((final - initial) / initial) * 100

theorem semicircle_area_increase :
  let r_long := 6
  let r_short := 4
  let area_long := 2 * area_semicircle r_long
  let area_short := 2 * area_semicircle r_short
  percent_increase area_short area_long = 125 :=
by
  let r_long := 6
  let r_short := 4
  let area_long := 2 * area_semicircle r_long
  let area_short := 2 * area_semicircle r_short
  have : area_semicircle r_long = 18 * Real.pi := by sorry
  have : area_semicircle r_short = 8 * Real.pi := by sorry
  have : area_long = 36 * Real.pi := by sorry
  have : area_short = 16 * Real.pi := by sorry
  have : percent_increase area_short area_long = 125 := by sorry
  exact this

end semicircle_area_increase_l171_171281


namespace min_xy_of_conditions_l171_171498

open Real

theorem min_xy_of_conditions
  (x y : ℝ)
  (hx_pos : 0 < x) 
  (hy_pos : 0 < y) 
  (hxy_eq : 1 / (2 + x) + 1 / (2 + y) = 1 / 3) : 
  xy ≥ 16 :=
by
  sorry

end min_xy_of_conditions_l171_171498


namespace price_increase_problem_l171_171249

variable (P P' x : ℝ)

theorem price_increase_problem
  (h1 : P' = P * (1 + x / 100))
  (h2 : P = P' * (1 - 23.076923076923077 / 100)) :
  x = 30 :=
by
  sorry

end price_increase_problem_l171_171249


namespace inequality_of_powers_l171_171237

theorem inequality_of_powers (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (hab : a ≤ b) (hbc : b ≤ c) (hcd : c ≤ d) :
  a^b * b^c * c^d * d^a ≥ b^a * c^b * d^c * a^d := 
sorry

end inequality_of_powers_l171_171237


namespace eiffel_tower_model_height_l171_171565

theorem eiffel_tower_model_height 
  (H1 : ℝ) (W1 : ℝ) (W2 : ℝ) (H2 : ℝ)
  (h1 : H1 = 324)
  (w1 : W1 = 8000000)  -- converted 8000 tons to 8000000 kg
  (w2 : W2 = 1)
  (h_eq : (H2 / H1)^3 = W2 / W1) : 
  H2 = 1.62 :=
by
  rw [h1, w1, w2] at h_eq
  sorry

end eiffel_tower_model_height_l171_171565


namespace geometric_sequence_sum_l171_171603

theorem geometric_sequence_sum :
  let a := (1/2 : ℚ)
  let r := (1/3 : ℚ)
  let n := 6
  let S_n := a * (1 - r^n) / (1 - r)
  S_n = 364 / 243 :=
by
  sorry

end geometric_sequence_sum_l171_171603


namespace correctly_calculated_value_l171_171402

theorem correctly_calculated_value (x : ℕ) (h : 5 * x = 40) : 2 * x = 16 := 
by {
  sorry
}

end correctly_calculated_value_l171_171402


namespace smallest_N_l171_171813

theorem smallest_N (N : ℕ) (hN : N > 70) (hdv : 70 ∣ 21 * N) : N = 80 :=
sorry

end smallest_N_l171_171813


namespace ada_original_seat_l171_171045

theorem ada_original_seat {positions : Fin 6 → Fin 6} 
  (Bea Ceci Dee Edie Fred Ada: Fin 6)
  (h1: Ada = 0)
  (h2: positions (Bea + 1) = Bea)
  (h3: positions (Ceci - 2) = Ceci)
  (h4: positions Dee = Edie ∧ positions Edie = Dee)
  (h5: positions Fred = Fred) :
  Ada = 1 → Bea = 1 → Ceci = 3 → Dee = 4 → Edie = 5 → Fred = 6 → Ada = 1 :=
by
  intros
  sorry

end ada_original_seat_l171_171045


namespace apple_weight_susan_l171_171733

theorem apple_weight_susan 
  (P : ℚ) (B : ℚ) (T : ℚ) (S : ℚ) (sliced_weight : ℚ) (w : ℚ)
  (hP : P = 38.25)
  (hB : B = P + 8.5)
  (hT : T = (3 / 8) * B)
  (hsliced : sliced_weight = (T / 2) * 75)
  (hS : S = (1 / 2) * T + 7)
  (h90 : w = (S * 0.9) * 150) 
  :
  w = 2128.359375 := 
  sorry

end apple_weight_susan_l171_171733


namespace parallel_vectors_implies_value_of_x_l171_171791

-- Define the vectors a and b
def a : ℝ × ℝ := (4, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 3)

-- Define the condition for parallel vectors
def parallel (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, (u.1 = k * v.1) ∧ (u.2 = k * v.2)

-- The proof statement
theorem parallel_vectors_implies_value_of_x : ∀ (x : ℝ), parallel a (b x) → x = 6 :=
by
  intro x
  intro h
  sorry

end parallel_vectors_implies_value_of_x_l171_171791


namespace valid_license_plates_count_l171_171429

-- Define the number of choices for letters and digits
def num_letters : ℕ := 26
def num_digits : ℕ := 10

-- Define the total number of valid license plates
def num_valid_license_plates : ℕ := num_letters^3 * num_digits^3

-- Theorem stating that the number of valid license plates is 17,576,000
theorem valid_license_plates_count :
  num_valid_license_plates = 17576000 :=
by
  sorry

end valid_license_plates_count_l171_171429


namespace smallest_sector_angle_divided_circle_l171_171851

theorem smallest_sector_angle_divided_circle : ∃ a d : ℕ, 
  (2 * a + 7 * d = 90) ∧ 
  (8 * (a + (a + 7 * d)) / 2 = 360) ∧ 
  a = 38 := 
by
  sorry

end smallest_sector_angle_divided_circle_l171_171851


namespace compute_difference_of_squares_l171_171745

theorem compute_difference_of_squares :
  (23 + 15) ^ 2 - (23 - 15) ^ 2 = 1380 := by
  sorry

end compute_difference_of_squares_l171_171745


namespace kelseys_sister_age_in_2021_l171_171221

-- Definitions based on given conditions
def kelsey_birth_year : ℕ := 1999 - 25
def sister_birth_year : ℕ := kelsey_birth_year - 3

-- Prove that Kelsey's older sister is 50 years old in 2021
theorem kelseys_sister_age_in_2021 : (2021 - sister_birth_year) = 50 :=
by
  -- Add proof here
  sorry

end kelseys_sister_age_in_2021_l171_171221


namespace problem_statement_l171_171110

-- Assume F is a function defined such that given the point (4,4) is on the graph y = F(x)
def F : ℝ → ℝ := sorry

-- Hypothesis: (4, 4) is on the graph of y = F(x)
axiom H : F 4 = 4

-- We need to prove that F(4) = 4
theorem problem_statement : F 4 = 4 :=
by exact H

end problem_statement_l171_171110


namespace expression_evaluates_to_47_l171_171289

theorem expression_evaluates_to_47 : 
  (3 * 4 * 5) * ((1 / 3) + (1 / 4) + (1 / 5)) = 47 := 
by 
  sorry

end expression_evaluates_to_47_l171_171289


namespace solve_for_S_l171_171512

theorem solve_for_S (S : ℝ) (h : (1 / 3) * (1 / 8) * S = (1 / 4) * (1 / 6) * 120) : S = 120 :=
sorry

end solve_for_S_l171_171512


namespace CarriageSharingEquation_l171_171442

theorem CarriageSharingEquation (x : ℕ) :
  (x / 3 + 2 = (x - 9) / 2) ↔
  (3 * ((x - 9) / 2) + 2 * 3 = x / 3 + 2) ∧ 
  (2 * ((x - 9) / 2) + 9 = x ∨ 2 * ((x - 9) / 2) + 9 < x) ∧ 
  (x / 3 + 2 < 3 * (x / 2) + 2 * 2 ∨ x / 3 + 2 = 3 * (x / 2) + 2 * 2) :=
sorry

end CarriageSharingEquation_l171_171442


namespace smallest_solution_x4_minus_50x2_plus_625_eq_0_l171_171074

theorem smallest_solution_x4_minus_50x2_plus_625_eq_0 :
  ∃ (x : ℝ), (x * x * x * x - 50 * x * x + 625 = 0) ∧ (∀ y, (y * y * y * y - 50 * y * y + 625 = 0) → x ≤ y) :=
sorry

end smallest_solution_x4_minus_50x2_plus_625_eq_0_l171_171074


namespace multiple_people_sharing_carriage_l171_171439

theorem multiple_people_sharing_carriage (x : ℝ) : 
  (x / 3) + 2 = (x - 9) / 2 :=
sorry

end multiple_people_sharing_carriage_l171_171439


namespace triangle_area_l171_171404

theorem triangle_area :
  ∃ (A : ℝ),
  let a := 65
  let b := 60
  let c := 25
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  a = 65 ∧ b = 60 ∧ c = 25 ∧ s = 75 ∧  area = 750 :=
by
  let a := 65
  let b := 60
  let c := 25
  let s := (a + b + c) / 2
  use Real.sqrt (s * (s - a) * (s - b) * (s - c))
  -- We would prove the conditions and calculations here, but we skip the proof parts
  sorry

end triangle_area_l171_171404


namespace binary_arith_proof_l171_171113

theorem binary_arith_proof :
  let a := 0b1101110  -- binary representation of 1101110_2
  let b := 0b101010   -- binary representation of 101010_2
  let c := 0b100      -- binary representation of 100_2
  (a * b / c) = 0b11001000010 :=  -- binary representation of the final result
by
  sorry

end binary_arith_proof_l171_171113


namespace gina_total_pay_l171_171617

noncomputable def gina_painting_pay : ℕ :=
let roses_per_hour := 6
let lilies_per_hour := 7
let rose_order := 6
let lily_order := 14
let pay_per_hour := 30

-- Calculate total time (in hours) Gina spends to complete the order
let time_for_roses := rose_order / roses_per_hour
let time_for_lilies := lily_order / lilies_per_hour
let total_time := time_for_roses + time_for_lilies

-- Calculate the total pay
let total_pay := total_time * pay_per_hour

total_pay

-- The theorem that Gina gets paid $90 for the order
theorem gina_total_pay : gina_painting_pay = 90 := by
  sorry

end gina_total_pay_l171_171617


namespace probability_at_least_one_peanut_ball_l171_171280

def glutinous_rice_balls := {sesame := 1, peanut := 2, red_bean_paste := 3}

-- Define the event of selecting at least one peanut filling glutinous rice ball
noncomputable def event := Pr[ at_least_one_peanut_ball | select_two_glutinous_rice_balls ]

theorem probability_at_least_one_peanut_ball :
  event = 3/5 := by sorry

end probability_at_least_one_peanut_ball_l171_171280


namespace real_solutions_count_l171_171531

theorem real_solutions_count :
  ∃ S : Set ℝ, (∀ x : ℝ, x ∈ S ↔ (|x-2| + |x-3| = 1)) ∧ (S = Set.Icc 2 3) :=
sorry

end real_solutions_count_l171_171531


namespace translation_equivalence_l171_171377

def f₁ (x : ℝ) : ℝ := 4 * (x + 3)^2 - 4
def f₂ (x : ℝ) : ℝ := 4 * (x - 3)^2 + 4

theorem translation_equivalence :
  (∀ x : ℝ, f₁ (x + 6) = 4 * (x + 9)^2 + 4) ∧
  (∀ x : ℝ, f₁ x  - 8 = 4 * (x + 3)^2 - 4) :=
by sorry

end translation_equivalence_l171_171377


namespace dice_product_not_odd_probability_l171_171896

theorem dice_product_not_odd_probability :
  let odd_faces := {1, 3, 5}
  let even_faces := {2, 4, 6}
  let total_outcomes := 6 * 6
  let odd_product_outcomes := 3 * 3
  let even_product_outcomes := total_outcomes - odd_product_outcomes
  let probability : ℚ := even_product_outcomes / total_outcomes
  probability = 3 / 4 :=
by
  sorry

end dice_product_not_odd_probability_l171_171896


namespace multiple_people_sharing_carriage_l171_171440

theorem multiple_people_sharing_carriage (x : ℝ) : 
  (x / 3) + 2 = (x - 9) / 2 :=
sorry

end multiple_people_sharing_carriage_l171_171440


namespace quadratic_least_value_l171_171170

variable (a b c : ℝ)

theorem quadratic_least_value (h_a_pos : a > 0)
  (h_c_eq : ∀ x : ℝ, a * x^2 + b * x + c ≥ 9) :
  c = 9 + b^2 / (4 * a) :=
by
  sorry

end quadratic_least_value_l171_171170


namespace probability_at_least_40_cents_heads_l171_171360

-- Definitions of the values of the coins
def penny_value : ℕ := 1
def nickel_value : ℕ := 5
def dime_value : ℕ := 10
def quarter_value : ℕ := 25
def fifty_cent_piece_value : ℕ := 50

-- Define what it means to have at least 40 cents
def at_least_40_cents (c1 c2 c3 c4 c5 : Bool) : Prop :=
  (if c1 then penny_value else 0) +
  (if c2 then nickel_value else 0) +
  (if c3 then dime_value else 0) +
  (if c4 then quarter_value else 0) +
  (if c5 then fifty_cent_piece_value else 0) ≥ 40

-- Calculate the probability of a successful outcome
theorem probability_at_least_40_cents_heads :
  (Finset.filter (λ outcome : Finset (Fin 5), at_least_40_cents 
    (outcome.contains 0)
    (outcome.contains 1)
    (outcome.contains 2)
    (outcome.contains 3)
    (outcome.contains 4))
    (Finset.powerset (Finset.range 5))).card.toRat / 32 = 9 / 16 := sorry

end probability_at_least_40_cents_heads_l171_171360


namespace total_area_to_paint_proof_l171_171924

def barn_width : ℝ := 15
def barn_length : ℝ := 20
def barn_height : ℝ := 8
def door_width : ℝ := 3
def door_height : ℝ := 7
def window_width : ℝ := 2
def window_height : ℝ := 4

noncomputable def wall_area (width length height : ℝ) : ℝ := 2 * (width * height + length * height)
noncomputable def door_area (width height : ℝ) (num: ℕ) : ℝ := width * height * num
noncomputable def window_area (width height : ℝ) (num: ℕ) : ℝ := width * height * num

noncomputable def total_area_to_paint : ℝ := 
  let total_wall_area := wall_area barn_width barn_length barn_height
  let total_door_area := door_area door_width door_height 2
  let total_window_area := window_area window_width window_height 3
  let net_wall_area := total_wall_area - total_door_area - total_window_area
  let ceiling_floor_area := barn_width * barn_length * 2
  net_wall_area * 2 + ceiling_floor_area

theorem total_area_to_paint_proof : total_area_to_paint = 1588 := by
  sorry

end total_area_to_paint_proof_l171_171924


namespace log_relationships_l171_171628

theorem log_relationships (c d y : ℝ) (hc : c > 0) (hd : d > 0) (hy : y > 0) :
  9 * (Real.log y / Real.log c)^2 + 5 * (Real.log y / Real.log d)^2 = 18 * (Real.log y)^2 / (Real.log c * Real.log d) →
  d = c^(1 / Real.sqrt 3) ∨ d = c^(Real.sqrt 3) ∨ d = c^(1 / Real.sqrt (6 / 10)) ∨ d = c^(Real.sqrt (6 / 10)) :=
sorry

end log_relationships_l171_171628


namespace edith_books_total_l171_171299

-- Define the conditions
def novels := 80
def writing_books := novels * 2

-- Theorem statement
theorem edith_books_total : novels + writing_books = 240 :=
by sorry

end edith_books_total_l171_171299


namespace sin_90_eq_1_l171_171127

-- Define the unit circle
def unit_circle (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define the sine of 90 degrees using radians
def sin_90_degrees : ℝ := unit_circle (Real.pi / 2).snd

-- State the theorem
theorem sin_90_eq_1 : sin_90_degrees = 1 :=
by
  sorry

end sin_90_eq_1_l171_171127


namespace compute_difference_of_squares_l171_171744

theorem compute_difference_of_squares :
  (23 + 15) ^ 2 - (23 - 15) ^ 2 = 1380 := by
  sorry

end compute_difference_of_squares_l171_171744


namespace river_joe_collected_money_l171_171348

theorem river_joe_collected_money :
  let price_catfish : ℤ := 600 -- in cents to avoid floating point issues
  let price_shrimp : ℤ := 350 -- in cents to avoid floating point issues
  let total_orders : ℤ := 26
  let shrimp_orders : ℤ := 9
  let catfish_orders : ℤ := total_orders - shrimp_orders
  let total_catfish_sales : ℤ := catfish_orders * price_catfish
  let total_shrimp_sales : ℤ := shrimp_orders * price_shrimp
  let total_money_collected : ℤ := total_catfish_sales + total_shrimp_sales
  total_money_collected = 13350 := -- in cents, so $133.50 is 13350 cents
by
  sorry

end river_joe_collected_money_l171_171348


namespace option_d_can_form_triangle_l171_171086

noncomputable def satisfies_triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem option_d_can_form_triangle : satisfies_triangle_inequality 2 3 4 :=
by {
  -- Using the triangle inequality theorem to check
  sorry
}

end option_d_can_form_triangle_l171_171086


namespace solve_inequality_system_l171_171047

theorem solve_inequality_system (x : ℝ) 
  (h1 : 3 * x - 1 > x + 1) 
  (h2 : (4 * x - 5) / 3 ≤ x) 
  : 1 < x ∧ x ≤ 5 :=
by
  sorry

end solve_inequality_system_l171_171047


namespace luke_bus_time_l171_171031

theorem luke_bus_time
  (L : ℕ)   -- Luke's bus time to work in minutes
  (P : ℕ)   -- Paula's bus time to work in minutes
  (B : ℕ)   -- Luke's bike time home in minutes
  (h1 : P = 3 * L / 5) -- Paula's bus time is \( \frac{3}{5} \) of Luke's bus time
  (h2 : B = 5 * L)     -- Luke's bike time is 5 times his bus time
  (h3 : L + P + B + P = 504) -- Total travel time is 504 minutes
  : L = 70 := 
sorry

end luke_bus_time_l171_171031


namespace children_count_l171_171277

theorem children_count (C A : ℕ) (h1 : 15 * A + 8 * C = 720) (h2 : A = C + 25) : C = 15 := 
by
  sorry

end children_count_l171_171277


namespace sum_of_remainders_mod_l171_171711

theorem sum_of_remainders_mod (a b c : ℕ) (h1 : a % 53 = 31) (h2 : b % 53 = 22) (h3 : c % 53 = 7) :
  (a + b + c) % 53 = 7 :=
by
  sorry

end sum_of_remainders_mod_l171_171711


namespace probability_of_winning_11th_round_l171_171899

-- Definitions of the conditions
def player1_wins_ten_rounds (eggs : List ℕ) : Prop :=
  ∀ i, i < 10 → eggs.indexOf (eggs.nthLe 0 (i+1)) < eggs.indexOf (eggs.nthLe 1 (i+1))

def is_strongest (egg : ℕ) (eggs : List ℕ) : Prop :=
  egg = List.maximum (0 :: eggs)

-- The proof to show the probability of winning the 11th round
theorem probability_of_winning_11th_round
  (eggs : List ℕ) : player1_wins_ten_rounds eggs →
  (1 - 1 / (length eggs + 1) = 11 / 12) :=
by
  sorry

end probability_of_winning_11th_round_l171_171899


namespace average_salary_all_employees_l171_171490

-- Define the given conditions
def average_salary_officers : ℝ := 440
def average_salary_non_officers : ℝ := 110
def number_of_officers : ℕ := 15
def number_of_non_officers : ℕ := 480

-- Define the proposition we need to prove
theorem average_salary_all_employees :
  let total_salary_officers := average_salary_officers * number_of_officers
  let total_salary_non_officers := average_salary_non_officers * number_of_non_officers
  let total_salary_all_employees := total_salary_officers + total_salary_non_officers
  let total_number_of_employees := number_of_officers + number_of_non_officers
  let average_salary_all_employees := total_salary_all_employees / total_number_of_employees
  average_salary_all_employees = 120 :=
by {
  -- Skipping the proof steps
  sorry
}

end average_salary_all_employees_l171_171490


namespace triangle_inequality_third_side_l171_171777

theorem triangle_inequality_third_side (x : ℝ) (h1 : 5 > 0) (h2 : 8 > 0) :
  3 < x ∧ x < 13 ↔ (5 + 8 > x) ∧ (5 + x > 8) ∧ (8 + x > 5) :=
by 
  -- Placeholder for proof
  sorry

end triangle_inequality_third_side_l171_171777


namespace sum_zero_inv_sum_zero_a_plus_d_zero_l171_171499

theorem sum_zero_inv_sum_zero_a_plus_d_zero 
  (a b c d : ℝ) (h1 : a ≤ b ∧ b ≤ c ∧ c ≤ d) 
  (h2 : a + b + c + d = 0) 
  (h3 : 1/a + 1/b + 1/c + 1/d = 0) :
  a + d = 0 := 
  sorry

end sum_zero_inv_sum_zero_a_plus_d_zero_l171_171499


namespace amount_y_gets_each_rupee_x_gets_l171_171729

-- Given conditions
variables (x y z a : ℝ)
variables (h_y_share : y = 36) (h_total : x + y + z = 156) (h_z : z = 0.50 * x)

-- Proof problem
theorem amount_y_gets_each_rupee_x_gets (h : 36 / x = a) : a = 9 / 20 :=
by {
  -- The proof is omitted and replaced with 'sorry'.
  sorry
}

end amount_y_gets_each_rupee_x_gets_l171_171729


namespace min_students_in_class_l171_171011

noncomputable def min_possible_students (b g : ℕ) : Prop :=
  (3 * b) / 4 = 2 * (2 * g) / 3 ∧ b = (16 * g) / 9

theorem min_students_in_class : ∃ (b g : ℕ), min_possible_students b g ∧ b + g = 25 :=
by
  sorry

end min_students_in_class_l171_171011


namespace unique_two_digit_number_l171_171892

theorem unique_two_digit_number (x y : ℕ) (h1 : 10 ≤ 10 * x + y ∧ 10 * x + y < 100) (h2 : 3 * y = 2 * x) (h3 : y + 3 = x) : 10 * x + y = 63 :=
by
  sorry

end unique_two_digit_number_l171_171892


namespace average_home_runs_correct_l171_171525

-- Define the number of players hitting specific home runs
def players_5_hr : ℕ := 3
def players_7_hr : ℕ := 2
def players_9_hr : ℕ := 1
def players_11_hr : ℕ := 2
def players_13_hr : ℕ := 1

-- Calculate the total number of home runs and total number of players
def total_hr : ℕ := 5 * players_5_hr + 7 * players_7_hr + 9 * players_9_hr + 11 * players_11_hr + 13 * players_13_hr
def total_players : ℕ := players_5_hr + players_7_hr + players_9_hr + players_11_hr + players_13_hr

-- Calculate the average number of home runs
def average_home_runs : ℚ := total_hr / total_players

-- The theorem we need to prove
theorem average_home_runs_correct : average_home_runs = 73 / 9 :=
by
  sorry

end average_home_runs_correct_l171_171525


namespace total_cups_needed_l171_171425

def servings : Float := 18.0
def cups_per_serving : Float := 2.0

theorem total_cups_needed : servings * cups_per_serving = 36.0 :=
by
  sorry

end total_cups_needed_l171_171425


namespace distance_between_lines_is_two_l171_171466

noncomputable def distance_between_parallel_lines : ℝ := 
  let A1 := 3
  let B1 := 4
  let C1 := -3
  let A2 := 6
  let B2 := 8
  let C2 := 14
  (|C2 - C1| : ℝ) / Real.sqrt (A2^2 + B2^2)

theorem distance_between_lines_is_two :
  distance_between_parallel_lines = 2 := by
  sorry

end distance_between_lines_is_two_l171_171466


namespace probability_two_defective_phones_l171_171105

theorem probability_two_defective_phones (total_smartphones : ℕ) 
  (typeA amountA defectiveA : ℕ) 
  (typeB amountB defectiveB : ℕ)
  (typeC amountC defectiveC : ℕ)
  (hA : amountA = 100) (hB : amountB = 80) (hC : amountC = 70)
  (hDA : defectiveA = 30) (hDB : defectiveB = 25) (hDC : defectiveC = 21)
  (hTotal : total_smartphones = 250) :
  let P_first_pick := (defectiveA + defectiveB + defectiveC) / total_smartphones,
      P_second_pick := (defectiveA + defectiveB + defectiveC - 1) / (total_smartphones - 1)
  in P_first_pick * P_second_pick ≈ 0.0916 :=
by
  -- Definitions and setup
  let total_defective := defectiveA + defectiveB + defectiveC
  have hTotal_defective: total_defective = 76 := by 
    rw [hDA, hDB, hDC]; norm_num
    
  -- Probabilities
  let P_first_pick := (total_defective : ℚ) / total_smartphones
  have hP_first_pick: P_first_pick = 76 / 250 := by rw [hTotal_defective, hTotal]; norm_num
  
  let P_second_pick := (total_defective - 1 : ℚ) / (total_smartphones - 1)
  
  -- Final probability calculation
  let P_both_defective := P_first_pick * P_second_pick
  have approx_P_both_defective: P_both_defective ≈ 0.0916 := 
    by 
      have eq1: (76 : ℚ) / 250 = 0.304 := by norm_num
      have eq2: (75 : ℚ) / 249 = 0.3012 := by norm_num
      rw [hP_first_pick, eq2]
      have eq3: 0.304 * 0.3012 = 0.0915632 := by norm_num
      convert eq3
  exact approx_P_both_defective

end probability_two_defective_phones_l171_171105


namespace min_value_expression_l171_171225

theorem min_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  (x + y) * (1 / x + 1 / y) ≥ 6 := 
by
  sorry

end min_value_expression_l171_171225


namespace geometric_sequence_ratio_l171_171467

noncomputable def geometric_sequence_pos (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a n > 0 ∧ a (n + 1) = a n * q

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) (h : geometric_sequence_pos a q) (h_q : q^2 = 4) :
  (a 2 + a 3) / (a 3 + a 4) = 1 / 2 :=
sorry

end geometric_sequence_ratio_l171_171467


namespace number_of_teams_l171_171537

-- Define the problem context
variables (n : ℕ)

-- Define the conditions
def total_games (n : ℕ) : ℕ := n * (n - 1) / 2

-- The theorem we want to prove
theorem number_of_teams (h : total_games n = 55) : n = 11 :=
sorry

end number_of_teams_l171_171537


namespace value_of_a_l171_171319

theorem value_of_a (a : ℕ) (h : a ^ 3 = 21 * 35 * 45 * 35) : a = 105 :=
by
  sorry

end value_of_a_l171_171319


namespace range_f_iff_l171_171629

noncomputable def f (m x : ℝ) : ℝ :=
  Real.log ((m^2 - 3 * m + 2) * x^2 + 2 * (m - 1) * x + 5)

theorem range_f_iff (m : ℝ) :
  (∀ y ∈ Set.univ, ∃ x, f m x = y) ↔ (m = 1 ∨ (2 < m ∧ m ≤ 9/4)) := 
by
  sorry

end range_f_iff_l171_171629


namespace oblique_projection_correctness_l171_171259

structure ProjectionConditions where
  intuitive_diagram_of_triangle_is_triangle : Prop
  intuitive_diagram_of_parallelogram_is_parallelogram : Prop

theorem oblique_projection_correctness (c : ProjectionConditions)
  (h1 : c.intuitive_diagram_of_triangle_is_triangle)
  (h2 : c.intuitive_diagram_of_parallelogram_is_parallelogram) :
  c.intuitive_diagram_of_triangle_is_triangle ∧ c.intuitive_diagram_of_parallelogram_is_parallelogram :=
by
  sorry

end oblique_projection_correctness_l171_171259


namespace new_interest_rate_l171_171378

theorem new_interest_rate 
    (i₁ : ℝ) (r₁ : ℝ) (p : ℝ) (additional_interest : ℝ) (i₂ : ℝ) (r₂ : ℝ)
    (h1 : r₁ = 0.05)
    (h2 : i₁ = 101.20)
    (h3 : additional_interest = 20.24)
    (h4 : i₂ = i₁ + additional_interest)
    (h5 : p = i₁ / (r₁ * 1))
    (h6 : i₂ = p * r₂ * 1) :
  r₂ = 0.06 :=
by
  sorry

end new_interest_rate_l171_171378


namespace fraction_meaningful_condition_l171_171482

theorem fraction_meaningful_condition (m : ℝ) : (m + 3 ≠ 0) → (m ≠ -3) :=
by
  intro h
  sorry

end fraction_meaningful_condition_l171_171482


namespace sin_90_eq_1_l171_171130

theorem sin_90_eq_1 : Real.sin (Float.pi / 2) = 1 := by
  sorry

end sin_90_eq_1_l171_171130


namespace sum_of_remainders_mod_l171_171712

theorem sum_of_remainders_mod (a b c : ℕ) (h1 : a % 53 = 31) (h2 : b % 53 = 22) (h3 : c % 53 = 7) :
  (a + b + c) % 53 = 7 :=
by
  sorry

end sum_of_remainders_mod_l171_171712


namespace radius_of_circle_with_area_3_14_l171_171399

theorem radius_of_circle_with_area_3_14 (A : ℝ) (π : ℝ) (hA : A = 3.14) (hπ : π = 3.14) (h_area : A = π * r^2) : r = 1 :=
by
  sorry

end radius_of_circle_with_area_3_14_l171_171399


namespace take_home_pay_correct_l171_171362

def jonessa_pay : ℝ := 500
def tax_deduction_percent : ℝ := 0.10
def insurance_deduction_percent : ℝ := 0.05
def pension_plan_deduction_percent : ℝ := 0.03
def union_dues_deduction_percent : ℝ := 0.02

def total_deductions : ℝ :=
  jonessa_pay * tax_deduction_percent +
  jonessa_pay * insurance_deduction_percent +
  jonessa_pay * pension_plan_deduction_percent +
  jonessa_pay * union_dues_deduction_percent

def take_home_pay : ℝ := jonessa_pay - total_deductions

theorem take_home_pay_correct : take_home_pay = 400 :=
  by
  sorry

end take_home_pay_correct_l171_171362


namespace expression_evaluation_l171_171736

theorem expression_evaluation : 2 + 3 * 4 - 5 + 6 * (2 - 1) = 15 := 
by sorry

end expression_evaluation_l171_171736


namespace trig_identity_l171_171254

theorem trig_identity (α : ℝ) :
  (Real.cos (α - 35 * Real.pi / 180) * Real.cos (25 * Real.pi / 180 + α) +
   Real.sin (α - 35 * Real.pi / 180) * Real.sin (25 * Real.pi / 180 + α)) = 1 / 2 :=
by
  sorry

end trig_identity_l171_171254


namespace correct_average_of_15_numbers_l171_171421

theorem correct_average_of_15_numbers
  (initial_average : ℝ)
  (num_numbers : ℕ)
  (incorrect1 incorrect2 correct1 correct2 : ℝ)
  (initial_average_eq : initial_average = 37)
  (num_numbers_eq : num_numbers = 15)
  (incorrect1_eq : incorrect1 = 52)
  (incorrect2_eq : incorrect2 = 39)
  (correct1_eq : correct1 = 64)
  (correct2_eq : correct2 = 27) :
  (initial_average * num_numbers - incorrect1 - incorrect2 + correct1 + correct2) / num_numbers = 37 :=
by
  rw [initial_average_eq, num_numbers_eq, incorrect1_eq, incorrect2_eq, correct1_eq, correct2_eq]
  sorry

end correct_average_of_15_numbers_l171_171421


namespace divide_triangle_into_two_equal_parts_l171_171625

-- Definitions for the problem
variable {A B C P Q R : Type} [PlaneGeometry A B C]

-- Additional conditions required
axiom P_on_perimeter_But_Not_Vertex : lies_on_perimeter_but_not_vertex P A B C
axiom Q_inside_triangle : lies_inside_triangle Q A B C
axiom R_on_perimeter : lies_on_perimeter R A B C

theorem divide_triangle_into_two_equal_parts :
  ∃ R : Point, 
    lies_on_perimeter R (triangle A B C) ∧ 
    polygonal_line_divides_area_eq P Q R (triangle A B C) :=
begin
  sorry
end

end divide_triangle_into_two_equal_parts_l171_171625


namespace negation_of_proposition_l171_171880

theorem negation_of_proposition : (¬ ∃ x : ℝ, x > 0 ∧ x^2 - 5 * x + 6 > 0) ↔ ∀ x : ℝ, x > 0 → x^2 - 5 * x + 6 ≤ 0 := by
  sorry

end negation_of_proposition_l171_171880


namespace find_b_and_area_l171_171333

open Real

variables (a c : ℝ) (A b S : ℝ)

theorem find_b_and_area 
  (h1 : a = sqrt 7) 
  (h2 : c = 3) 
  (h3 : A = π / 3) :
  (b = 1 ∨ b = 2) ∧ (S = 3 * sqrt 3 / 4 ∨ S = 3 * sqrt 3 / 2) := 
by sorry

end find_b_and_area_l171_171333


namespace benjamin_trip_odd_number_conditions_l171_171341

theorem benjamin_trip_odd_number_conditions (a b c : ℕ) 
  (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a)
  (h4 : a + b + c ≤ 9) 
  (h5 : ∃ x : ℕ, 60 * x = 99 * (c - a)) :
  a^2 + b^2 + c^2 = 35 := 
sorry

end benjamin_trip_odd_number_conditions_l171_171341


namespace solve_for_S_l171_171511

theorem solve_for_S (S : ℝ) (h : (1 / 3) * (1 / 8) * S = (1 / 4) * (1 / 6) * 120) : S = 120 :=
sorry

end solve_for_S_l171_171511


namespace number_of_students_preferring_dogs_l171_171955

-- Define the conditions
def total_students : ℕ := 30
def dogs_video_games_chocolate_percentage : ℚ := 0.50
def dogs_movies_vanilla_percentage : ℚ := 0.10
def cats_video_games_chocolate_percentage : ℚ := 0.20
def cats_movies_vanilla_percentage : ℚ := 0.15

-- Define the target statement to prove
theorem number_of_students_preferring_dogs : 
  (dogs_video_games_chocolate_percentage + dogs_movies_vanilla_percentage) * total_students = 18 :=
by
  sorry

end number_of_students_preferring_dogs_l171_171955


namespace pqrs_product_l171_171653

noncomputable def P := (Real.sqrt 2007 + Real.sqrt 2008)
noncomputable def Q := (-Real.sqrt 2007 - Real.sqrt 2008)
noncomputable def R := (Real.sqrt 2007 - Real.sqrt 2008)
noncomputable def S := (-Real.sqrt 2008 + Real.sqrt 2007)

theorem pqrs_product : P * Q * R * S = -1 := by
  sorry

end pqrs_product_l171_171653


namespace determine_M_l171_171196

noncomputable def M : Set ℤ :=
  {a | ∃ k : ℕ, k > 0 ∧ 6 = k * (5 - a)}

theorem determine_M : M = {-1, 2, 3, 4} :=
  sorry

end determine_M_l171_171196


namespace fourth_number_is_two_eighth_number_is_two_l171_171261

-- Conditions:
-- 1. Initial number on the board is 1
-- 2. Sequence of medians observed by Mitya

def initial_number : ℕ := 1
def medians : list ℚ := [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5]

-- Required proof statements:

-- a) The fourth number written on the board is 2
theorem fourth_number_is_two (numbers : list ℕ) (h_initial : numbers.head = initial_number)
  (h_medians : ∀ k, medians.nth k = some (list.median (numbers.take (k + 1)))) :
  numbers.nth 3 = some 2 :=
sorry

-- b) The eighth number written on the board is 2
theorem eighth_number_is_two (numbers : list ℕ) (h_initial : numbers.head = initial_number)
  (h_medians : ∀ k, medians.nth k = some (list.median (numbers.take (k + 1)))) :
  numbers.nth 7 = some 2 :=
sorry

end fourth_number_is_two_eighth_number_is_two_l171_171261


namespace find_C_l171_171536

theorem find_C (A B C : ℕ) (h0 : 3 * A - A = 10) (h1 : B + A = 12) (h2 : C - B = 6) (h3 : A ≠ B) (h4 : B ≠ C) (h5 : C ≠ A) 
: C = 13 :=
sorry

end find_C_l171_171536


namespace walls_divided_equally_l171_171437

-- Define the given conditions
def num_people : ℕ := 5
def num_rooms : ℕ := 9
def rooms_with_4_walls : ℕ := 5
def walls_per_room_4 : ℕ := 4
def rooms_with_5_walls : ℕ := 4
def walls_per_room_5 : ℕ := 5

-- Calculate the total number of walls
def total_walls : ℕ := (rooms_with_4_walls * walls_per_room_4) + (rooms_with_5_walls * walls_per_room_5)

-- Define the expected result
def walls_per_person : ℕ := total_walls / num_people

-- Theorem statement: Each person should paint 8 walls.
theorem walls_divided_equally : walls_per_person = 8 := by
  sorry

end walls_divided_equally_l171_171437


namespace geometric_seq_reciprocal_sum_l171_171491

noncomputable def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop := ∀ n, a (n + 1) = a n * r

theorem geometric_seq_reciprocal_sum
  (a : ℕ → ℝ) (r : ℝ)
  (h_geom : geometric_sequence a r)
  (h1 : a 2 * a 5 = -3/4)
  (h2 : a 2 + a 3 + a 4 + a 5 = 5/4) :
  (1 / a 2) + (1 / a 3) + (1 / a 4) + (1 / a 5) = -5/3 := sorry

end geometric_seq_reciprocal_sum_l171_171491


namespace determine_x_l171_171607

theorem determine_x (x : ℚ) (n : ℤ) (d : ℚ) 
  (h_cond : x = n + d)
  (h_floor : n = ⌊x⌋)
  (h_d : 0 ≤ d ∧ d < 1)
  (h_eq : ⌊x⌋ + x = 17 / 4) :
  x = 9 / 4 := sorry

end determine_x_l171_171607


namespace number_of_triplets_l171_171627

theorem number_of_triplets (N : ℕ) (a b c : ℕ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h4 : 2017 ≥ 10 * a) (h5 : 10 * a ≥ 100 * b) (h6 : 100 * b ≥ 1000 * c) : 
  N = 574 := 
sorry

end number_of_triplets_l171_171627


namespace inflation_over_two_years_real_yield_deposit_second_year_l171_171705

-- Inflation problem setup and proof
theorem inflation_over_two_years :
  ((1 + 0.015) ^ 2 - 1) * 100 = 3.0225 :=
by sorry

-- Real yield problem setup and proof
theorem real_yield_deposit_second_year :
  ((1.07 * 1.07) / (1 + 0.030225) - 1) * 100 = 11.13 :=
by sorry

end inflation_over_two_years_real_yield_deposit_second_year_l171_171705


namespace smallest_possible_value_of_N_l171_171805

theorem smallest_possible_value_of_N :
  ∃ N : ℕ, N > 70 ∧ (21 * N) % 70 = 0 ∧ N = 80 :=
by
  existsi 80
  split
  · norm_num
  split
  · norm_num
  sorry

end smallest_possible_value_of_N_l171_171805


namespace range_values_y_div_x_l171_171468

-- Given the equation of the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4 * x - 6 * y + 12 = 0

-- Prove that the range of values for y / x is [ (6 - 2 * sqrt 3) / 3, (6 + 2 * sqrt 3) / 3 ]
theorem range_values_y_div_x :
  (∀ x y : ℝ, circle_eq x y → (∃ k : ℝ, y = k * x) → 
  ( (6 - 2 * Real.sqrt 3) / 3 ≤ y / x ∧ y / x ≤ (6 + 2 * Real.sqrt 3) / 3 )) :=
sorry

end range_values_y_div_x_l171_171468


namespace ratio_of_cost_to_selling_price_l171_171061

-- Define the given conditions
def cost_price (CP : ℝ) := CP
def selling_price (CP : ℝ) : ℝ := CP + 0.25 * CP

-- Lean statement for the problem
theorem ratio_of_cost_to_selling_price (CP SP : ℝ) (h1 : SP = selling_price CP) : CP / SP = 4 / 5 :=
by
  sorry

end ratio_of_cost_to_selling_price_l171_171061


namespace value_of_m_l171_171325

theorem value_of_m (x1 x2 m : ℝ) (h1 : x1 + x2 = 8) (h2 : x1 = 3 * x2) : m = 12 :=
by
  -- Proof will be provided here
  sorry

end value_of_m_l171_171325


namespace calculate_y_position_l171_171624

/--
Given a number line with equally spaced markings, if eight steps are taken from \( 0 \) to \( 32 \),
then the position \( y \) after five steps can be calculated.
-/
theorem calculate_y_position : 
    ∃ y : ℕ, (∀ (step length : ℕ), (8 * step = 32) ∧ (y = 5 * length) → y = 20) :=
by
  -- Provide initial definitions based on the conditions
  let step := 4
  let length := 4
  use (5 * length)
  sorry

end calculate_y_position_l171_171624


namespace valentino_chickens_l171_171343

variable (C : ℕ) -- Number of chickens
variable (D : ℕ) -- Number of ducks
variable (T : ℕ) -- Number of turkeys
variable (total_birds : ℕ) -- Total number of birds on the farm

theorem valentino_chickens (h1 : D = 2 * C) 
                            (h2 : T = 3 * D)
                            (h3 : total_birds = C + D + T)
                            (h4 : total_birds = 1800) :
  C = 200 := by
  sorry

end valentino_chickens_l171_171343


namespace eugene_initial_pencils_l171_171159

theorem eugene_initial_pencils (P : ℕ) (h1 : P + 6 = 57) : P = 51 :=
by
  sorry

end eugene_initial_pencils_l171_171159


namespace carpet_dimensions_l171_171591
open Real

theorem carpet_dimensions (x y : ℝ) 
  (h1 : x > 0)
  (h2 : y > 0)
  (h3 : ∃ k: ℝ, y = k * x)
  (h4 : ∃ α β: ℝ, α + k * β = 50 ∧ k * α + β = 55)
  (h5 : ∃ γ δ: ℝ, γ + k * δ = 38 ∧ k * γ + δ = 55) :
  x = 25 ∧ y = 50 :=
by sorry

end carpet_dimensions_l171_171591


namespace triangle_angle_contradiction_l171_171713

theorem triangle_angle_contradiction (A B C : ℝ) (h₁ : A + B + C = 180) (h₂ : A > 60) (h₃ : B > 60) (h₄ : C > 60) :
  false :=
by
  sorry

end triangle_angle_contradiction_l171_171713


namespace solve_equation_l171_171352

def equation_solution (x : ℝ) : Prop :=
  (x^2 + x + 1) / (x + 1) = x + 3

theorem solve_equation :
  ∃ x : ℝ, equation_solution x ∧ x = -2 / 3 :=
by
  sorry

end solve_equation_l171_171352


namespace f_at_3_l171_171311

noncomputable def f : ℝ → ℝ := sorry

axiom f_odd (x : ℝ) : f (-x) = -f x
axiom f_of_2 : f 2 = 1
axiom f_rec (x : ℝ) : f (x + 2) = f x + f 2

theorem f_at_3 : f 3 = 3 / 2 := 
by 
  sorry

end f_at_3_l171_171311


namespace arithmetic_seq_a5_l171_171773

variable (a : ℕ → ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ (n : ℕ), a (n + 1) - a n = a 1 - a 0

theorem arithmetic_seq_a5 (h1 : is_arithmetic_sequence a) (h2 : a 2 + a 8 = 12) :
  a 5 = 6 :=
by
  sorry

end arithmetic_seq_a5_l171_171773


namespace carpet_dimensions_l171_171589

theorem carpet_dimensions (a b : ℕ) 
  (h1 : a^2 + b^2 = 38^2 + 55^2) 
  (h2 : a^2 + b^2 = 50^2 + 55^2) 
  (h3 : a ≤ b) : 
  (a = 25 ∧ b = 50) ∨ (a = 50 ∧ b = 25) :=
by {
  -- The proof would go here
  sorry
}

end carpet_dimensions_l171_171589


namespace find_function_l171_171497

theorem find_function (a : ℝ) (f : ℝ → ℝ) (h : ∀ x y, (f x * f y - f (x * y)) / 4 = 2 * x + 2 * y + a) : a = -3 ∧ ∀ x, f x = x + 1 :=
by
  sorry

end find_function_l171_171497


namespace line_eq_l171_171529

-- Conditions
def circle_eq (x y : ℝ) (a : ℝ) : Prop :=
  (x + 1)^2 + (y - 2)^2 = 5 - a

def midpoint (x1 y1 x2 y2 xm ym : ℝ) : Prop :=
  2*xm = x1 + x2 ∧ 2*ym = y1 + y2

-- Theorem statement
theorem line_eq (a : ℝ) (h : a < 3) :
  circle_eq 0 1 a →
  ∃ l : ℝ → ℝ, (∀ x, l x = x - 1) :=
sorry

end line_eq_l171_171529


namespace find_third_vertex_l171_171258

open Real

-- Define the vertices of the triangle
def vertex1 : ℝ × ℝ := (9, 3)
def vertex2 : ℝ × ℝ := (0, 0)

-- Define the conditions
def on_negative_x_axis (p : ℝ × ℝ) : Prop :=
  p.2 = 0 ∧ p.1 < 0

def area_of_triangle (a b c : ℝ × ℝ) : ℝ :=
  0.5 * abs ((b.1 - a.1) * (c.2 - a.2) - (c.1 - a.1) * (b.2 - a.2))

-- Statement of the problem in Lean
theorem find_third_vertex :
  ∃ (vertex3 : ℝ × ℝ), 
    on_negative_x_axis vertex3 ∧ 
    area_of_triangle vertex1 vertex2 vertex3 = 45 ∧
    vertex3 = (-30, 0) :=
sorry

end find_third_vertex_l171_171258


namespace train2_length_is_230_l171_171403

noncomputable def train_length_proof : Prop :=
  let speed1_kmph := 120
  let speed2_kmph := 80
  let length_train1 := 270
  let time_cross := 9
  let speed1_mps := speed1_kmph * 1000 / 3600
  let speed2_mps := speed2_kmph * 1000 / 3600
  let relative_speed := speed1_mps + speed2_mps
  let total_distance := relative_speed * time_cross
  let length_train2 := total_distance - length_train1
  length_train2 = 230

theorem train2_length_is_230 : train_length_proof :=
  by
    sorry

end train2_length_is_230_l171_171403


namespace money_left_is_correct_l171_171423

noncomputable def total_income : ℝ := 800000
noncomputable def children_pct : ℝ := 0.2
noncomputable def num_children : ℝ := 3
noncomputable def wife_pct : ℝ := 0.3
noncomputable def donation_pct : ℝ := 0.05

noncomputable def remaining_income_after_donations : ℝ := 
  let distributed_to_children := total_income * children_pct * num_children
  let distributed_to_wife := total_income * wife_pct
  let total_distributed := distributed_to_children + distributed_to_wife
  let remaining_after_family := total_income - total_distributed
  let donation := remaining_after_family * donation_pct
  remaining_after_family - donation

theorem money_left_is_correct :
  remaining_income_after_donations = 76000 := 
by 
  sorry

end money_left_is_correct_l171_171423


namespace inequality_solution_l171_171094

-- Declare the constants m and n
variables (m n : ℝ)

-- State the conditions
def condition1 (x : ℝ) := m < 0
def condition2 := n = -m / 2

-- State the theorem
theorem inequality_solution (x : ℝ) (h1 : condition1 m n) (h2 : condition2 m n) : 
  nx - m < 0 ↔ x < -2 :=
sorry

end inequality_solution_l171_171094


namespace exists_infinite_B_with_property_l171_171660

-- Definition of the sequence A
def seqA (n : ℕ) : ℤ := 5 * n - 2

-- Definition of the sequence B with its general form
def seqB (k : ℕ) (d : ℤ) : ℤ := k * d + 7 - d

-- The proof problem statement
theorem exists_infinite_B_with_property :
  ∃ (B : ℕ → ℤ) (d : ℤ), B 1 = 7 ∧ 
  (∀ k, k > 1 → B k = B (k - 1) + d) ∧
  (∀ n : ℕ, ∃ (k : ℕ), seqB k d = seqA n) :=
sorry

end exists_infinite_B_with_property_l171_171660


namespace smallest_solution_x4_minus_50x2_plus_625_eq_0_l171_171076

theorem smallest_solution_x4_minus_50x2_plus_625_eq_0 : ∃ x : ℝ, x^4 - 50 * x^2 + 625 = 0 ∧ ∀ y : ℝ, y^4 - 50 * y^2 + 625 = 0 → x ≤ y := 
sorry

end smallest_solution_x4_minus_50x2_plus_625_eq_0_l171_171076


namespace symmetry_origin_points_l171_171179

theorem symmetry_origin_points (x y : ℝ) (h₁ : (x, -2) = (-3, -y)) : x + y = -1 :=
sorry

end symmetry_origin_points_l171_171179


namespace smallest_N_value_proof_l171_171800

def smallest_value_N (N : ℕ) : Prop :=
  N > 70 ∧ (21 * N) % 70 = 0

theorem smallest_N_value_proof : ∃ N, smallest_value_N N ∧ (∀ M, smallest_value_N M → N ≤ M) :=
  sorry

end smallest_N_value_proof_l171_171800


namespace find_x_l171_171246

theorem find_x (x : ℝ) :
  (1 / 3) * ((2 * x + 8) + (7 * x + 3) + (3 * x + 9)) = 5 * x^2 - 8 * x + 2 ↔ 
  x = (36 + Real.sqrt 2136) / 30 ∨ x = (36 - Real.sqrt 2136) / 30 := 
sorry

end find_x_l171_171246


namespace total_pages_in_book_l171_171355

-- Conditions
def hours_reading := 5
def pages_read := 2323
def increase_per_hour := 10
def extra_pages_read := 90

-- Main statement to prove
theorem total_pages_in_book (T : ℕ) :
  (∃ P : ℕ, P + (P + increase_per_hour) + (P + 2 * increase_per_hour) + 
   (P + 3 * increase_per_hour) + (P + 4 * increase_per_hour) = pages_read) ∧
  (pages_read = T - pages_read + extra_pages_read) →
  T = 4556 :=
by { sorry }

end total_pages_in_book_l171_171355


namespace neg_p_neither_sufficient_nor_necessary_l171_171306

-- Definitions of p and q as described
def p (x : ℝ) : Prop := abs (x + 1) > 2
def q (x : ℝ) : Prop := 5 * x - 6 ≤ x^2

-- Proving that ¬p is neither a sufficient nor necessary condition for q
theorem neg_p_neither_sufficient_nor_necessary (x : ℝ) : 
  ( ¬ p x → q x ) = false ∧ ( q x → ¬ p x ) = false := by
  sorry

end neg_p_neither_sufficient_nor_necessary_l171_171306


namespace initial_scissors_l171_171539

-- Define conditions as per the problem
def Keith_placed (added : ℕ) : Prop := added = 22
def total_now (total : ℕ) : Prop := total = 76

-- Define the problem statement as a theorem
theorem initial_scissors (added total initial : ℕ) (h1 : Keith_placed added) (h2 : total_now total) 
  (h3 : total = initial + added) : initial = 54 := by
  -- This is where the proof would go
  sorry

end initial_scissors_l171_171539


namespace factorial_base8_trailing_zeros_l171_171991

-- Define the factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n + 1) * fact n

-- Define the function to count the largest power of a prime p dividing n!
def prime_power_in_factorial (p n : ℕ) : ℕ :=
  if p = 1 then 0 else
  let rec aux k := if k ≤ 0 then 0 else (n / k) + aux (k / p)
  in aux p

-- Define the function to compute number of trailing zeros in base b
def trailing_zeros_in_base (n b : ℕ) : ℕ :=
  let p := match (nat.find_greatest_prime_divisor b) with
           | some p' => p'
           | none => 1
           end
  in (prime_power_in_factorial p n) / (nat.find_greatest_power_of_prime b)

-- Define the statement
theorem factorial_base8_trailing_zeros : trailing_zeros_in_base 15 8 = 3 := by
  sorry

end factorial_base8_trailing_zeros_l171_171991


namespace select_integers_divisible_l171_171025

theorem select_integers_divisible (k : ℕ) (s : Finset ℤ) (h₁ : s.card = 2 * 2^k - 1) :
  ∃ t : Finset ℤ, t ⊆ s ∧ t.card = 2^k ∧ (t.sum id) % 2^k = 0 :=
sorry

end select_integers_divisible_l171_171025


namespace find_tangent_line_to_curves_l171_171053

noncomputable def tangent_line_to_curves_tangent (t : ℝ) : Prop :=
  let f : ℝ → ℝ := λ x, Real.exp x
  let g : ℝ → ℝ := λ x, -x ^ 2 / 4
  let tangent_line (t : ℝ) (x : ℝ) : ℝ := Real.exp t * (x - t) + Real.exp t
  in
  (e^t + t - 1 = 0) ∧
  (∀ x, tangent_line t x = -x ^ 2 / 4 → y = x + 1)

theorem find_tangent_line_to_curves : ∃ t, tangent_line_to_curves_tangent t := sorry

end find_tangent_line_to_curves_l171_171053


namespace find_B_l171_171694

theorem find_B (A B : ℕ) (h₁ : 6 * A + 10 * B + 2 = 77) (h₂ : A ≤ 9) (h₃ : B ≤ 9) : B = 1 := sorry

end find_B_l171_171694


namespace tom_seashells_l171_171542

theorem tom_seashells (fred_seashells : ℕ) (total_seashells : ℕ) (tom_seashells : ℕ)
  (h1 : fred_seashells = 43)
  (h2 : total_seashells = 58)
  (h3 : total_seashells = fred_seashells + tom_seashells) : tom_seashells = 15 :=
by
  sorry

end tom_seashells_l171_171542


namespace angle_of_inclination_of_line_is_135_degrees_l171_171905

theorem angle_of_inclination_of_line_is_135_degrees :
  ∃ θ : ℝ, (x + y - 1 = 0) ∧ (θ = 135.0 * real.pi / 180.0) :=
sorry

end angle_of_inclination_of_line_is_135_degrees_l171_171905


namespace profit_percentage_is_25_l171_171593

variable (CP SP Profit ProfitPercentage : ℝ)
variable (hCP : CP = 192)
variable (hSP : SP = 240)
variable (hProfit : Profit = SP - CP)
variable (hProfitPercentage : ProfitPercentage = (Profit / CP) * 100)

theorem profit_percentage_is_25 :
  hCP → hSP → hProfit → hProfitPercentage → ProfitPercentage = 25 := by
  intros hCP hSP hProfit hProfitPercentage
  sorry

end profit_percentage_is_25_l171_171593


namespace intersection_M_N_l171_171977

open Set

-- Definitions of the sets M and N
def M : Set ℤ := {-1, 0, 1, 5}
def N : Set ℤ := {-2, 1, 2, 5}

-- The theorem stating that the intersection of M and N is {1, 5}
theorem intersection_M_N :
  M ∩ N = {1, 5} :=
  sorry

end intersection_M_N_l171_171977


namespace polygon_sides_l171_171328

/-- If the sum of the interior angles of a polygon is three times the sum of its exterior angles,
    then the number of sides of the polygon is 8. -/
theorem polygon_sides (n : ℕ) (h1 : 180 * (n - 2) = 3 * 360) : n = 8 :=
sorry

end polygon_sides_l171_171328


namespace jackson_running_increase_l171_171494

theorem jackson_running_increase
    (initial_miles_per_day : ℕ)
    (final_miles_per_day : ℕ)
    (weeks_increasing : ℕ)
    (total_weeks : ℕ)
    (h1 : initial_miles_per_day = 3)
    (h2 : final_miles_per_day = 7)
    (h3 : weeks_increasing = 4)
    (h4 : total_weeks = 5) :
    (final_miles_per_day - initial_miles_per_day) / weeks_increasing = 1 := 
by
  -- provided steps from solution
  sorry

end jackson_running_increase_l171_171494


namespace polygon_sides_l171_171327

theorem polygon_sides (x : ℕ) (h1 : 180 * (x - 2) = 3 * 360) : x = 8 := by
  sorry

end polygon_sides_l171_171327


namespace sin_90_eq_one_l171_171121

noncomputable theory
open Real

/--
The sine of an angle in the unit circle is the y-coordinate of the point at that angle from the positive x-axis.
Rotating the point (1,0) by 90 degrees counterclockwise about the origin results in the point (0,1).
Prove that \(\sin 90^\circ = 1\).
-/
theorem sin_90_eq_one : sin (90 * (real.pi / 180)) = 1 :=
by
  -- Definitions and conditions for the unit circle and sine function
  let angle := 90 * (real.pi / 180)
  have h1 : (cos angle, sin angle) = (0, 1),
  { sorry },
  -- Desired conclusion
  exact h1.2

end sin_90_eq_one_l171_171121


namespace find_root_and_coefficient_l171_171177

theorem find_root_and_coefficient (m: ℝ) (x: ℝ) (h₁: x ^ 2 - m * x - 6 = 0) (h₂: x = 3) :
  (x = 3 ∧ -2 = -6 / 3 ∨ m = 1) :=
by
  sorry

end find_root_and_coefficient_l171_171177


namespace min_geometric_ratio_l171_171314

theorem min_geometric_ratio (q : ℝ) (a : ℕ → ℝ) (h1 : ∀ n, a (n + 1) = a n * q)
(h2 : 1 < q) (h3 : q < 2) : q = 6 / 5 := by
  sorry

end min_geometric_ratio_l171_171314


namespace sin_90_eq_1_l171_171131

theorem sin_90_eq_1 : Real.sin (Float.pi / 2) = 1 := by
  sorry

end sin_90_eq_1_l171_171131


namespace sum_of_legs_of_right_triangle_l171_171527

theorem sum_of_legs_of_right_triangle
  (a b : ℕ)
  (h1 : a % 2 = 0)
  (h2 : b = a + 2)
  (h3 : a^2 + b^2 = 50^2) :
  a + b = 70 := by
  sorry

end sum_of_legs_of_right_triangle_l171_171527


namespace difference_in_sales_l171_171568

def daily_pastries : ℕ := 20
def daily_bread : ℕ := 10
def today_pastries : ℕ := 14
def today_bread : ℕ := 25
def price_pastry : ℕ := 2
def price_bread : ℕ := 4

theorem difference_in_sales : (daily_pastries * price_pastry + daily_bread * price_bread) - (today_pastries * price_pastry + today_bread * price_bread) = -48 :=
by
  -- Proof will go here
  sorry

end difference_in_sales_l171_171568


namespace complex_multiplication_l171_171093

theorem complex_multiplication :
  ∀ (i : ℂ), i * i = -1 → i * (1 + i) = -1 + i :=
by
  intros i hi
  sorry

end complex_multiplication_l171_171093


namespace ratio_b_c_l171_171181

theorem ratio_b_c (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 3)
  (h2 : a * b * c / (d * e * f) = 0.1875)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : e / f = 1 / 8) : 
  b / c = 3 :=
sorry

end ratio_b_c_l171_171181


namespace sequence_term_number_l171_171983

theorem sequence_term_number (n : ℕ) : (n ≥ 1) → (n + 3 = 17 ∧ n + 1 = 15) → n = 14 := 
by
  intro h1 h2
  sorry

end sequence_term_number_l171_171983


namespace describe_cylinder_l171_171843

noncomputable def cylinder_geometric_shape (c : ℝ) (r θ z : ℝ) : Prop :=
  r = c

theorem describe_cylinder (c : ℝ) (hc : 0 < c) :
  ∀ r θ z : ℝ, cylinder_geometric_shape c r θ z ↔ (r = c) :=
by
  sorry

end describe_cylinder_l171_171843


namespace mean_cars_l171_171087

theorem mean_cars (a b c d e : ℝ) (h1 : a = 30) (h2 : b = 14) (h3 : c = 14) (h4 : d = 21) (h5 : e = 25) : 
  (a + b + c + d + e) / 5 = 20.8 :=
by
  -- The proof will be provided here
  sorry

end mean_cars_l171_171087


namespace bedrooms_count_l171_171218

/-- Number of bedrooms calculation based on given conditions -/
theorem bedrooms_count (B : ℕ) (h1 : ∀ b, b = 20 * B)
  (h2 : ∀ lr, lr = 20 * B)
  (h3 : ∀ bath, bath = 2 * 20 * B)
  (h4 : ∀ out, out = 2 * (20 * B + 20 * B + 40 * B))
  (h5 : ∀ siblings, siblings = 3)
  (h6 : ∀ work_time, work_time = 4 * 60) : B = 3 :=
by
  -- proof will be provided here
  sorry

end bedrooms_count_l171_171218


namespace david_marks_in_biology_l171_171153

theorem david_marks_in_biology (marks_english marks_math marks_physics marks_chemistry : ℕ)
  (average_marks num_subjects total_marks_known : ℕ)
  (h1 : marks_english = 76)
  (h2 : marks_math = 65)
  (h3 : marks_physics = 82)
  (h4 : marks_chemistry = 67)
  (h5 : average_marks = 75)
  (h6 : num_subjects = 5)
  (h7 : total_marks_known = marks_english + marks_math + marks_physics + marks_chemistry)
  (h8 : total_marks_known = 290)
  : ∃ biology_marks : ℕ, biology_marks = 85 ∧ biology_marks = (average_marks * num_subjects) - total_marks_known :=
by
  -- placeholder for proof
  sorry

end david_marks_in_biology_l171_171153


namespace find_unknown_number_l171_171048

theorem find_unknown_number (x : ℕ) :
  (x + 30 + 50) / 3 = ((20 + 40 + 6) / 3 + 8) → x = 10 := by
    sorry

end find_unknown_number_l171_171048


namespace max_value_of_f_l171_171382

noncomputable def f (x : ℝ) : ℝ := x / Real.exp x

theorem max_value_of_f :
  ∃ x_max : ℝ, (∀ x : ℝ, f x ≤ f x_max) ∧ f 1 = 1 / Real.exp 1 := 
by {
  sorry
}

end max_value_of_f_l171_171382


namespace bonnets_per_orphanage_l171_171035

theorem bonnets_per_orphanage :
  let monday_bonnets := 10
  let tuesday_wednesday_bonnets := 2 * monday_bonnets
  let thursday_bonnets := monday_bonnets + 5
  let friday_bonnets := thursday_bonnets - 5
  let total_bonnets := monday_bonnets + tuesday_wednesday_bonnets + thursday_bonnets + friday_bonnets
  let orphanages := 5
  total_bonnets / orphanages = 11 :=
by
  let monday_bonnets := 10
  let tuesday_wednesday_bonnets := 2 * monday_bonnets
  let thursday_bonnets := monday_bonnets + 5
  let friday_bonnets := thursday_bonnets - 5
  let total_bonnets := monday_bonnets + tuesday_wednesday_bonnets + thursday_bonnets + friday_bonnets
  let orphanages := 5
  sorry

end bonnets_per_orphanage_l171_171035


namespace sine_of_smaller_angle_and_k_domain_l171_171248

theorem sine_of_smaller_angle_and_k_domain (α : ℝ) (k : ℝ) (AD : ℝ) (h0 : 1 < k) 
  (h1 : CD = AD * Real.tan (2 * α)) (h2 : BD = AD * Real.tan α) 
  (h3 : k = CD / BD) :
  k > 2 ∧ Real.sin (Real.pi / 2 - 2 * α) = 1 / (k - 1) := by
  sorry

end sine_of_smaller_angle_and_k_domain_l171_171248


namespace latest_departure_time_l171_171285

noncomputable def minutes_in_an_hour : ℕ := 60
noncomputable def departure_time : ℕ := 20 * minutes_in_an_hour -- 8:00 pm in minutes
noncomputable def checkin_time : ℕ := 2 * minutes_in_an_hour -- 2 hours in minutes
noncomputable def drive_time : ℕ := 45 -- 45 minutes
noncomputable def parking_time : ℕ := 15 -- 15 minutes
noncomputable def total_time_needed : ℕ := checkin_time + drive_time + parking_time -- Total time in minutes

theorem latest_departure_time : departure_time - total_time_needed = 17 * minutes_in_an_hour :=
by
  sorry

end latest_departure_time_l171_171285


namespace smallest_solution_x4_minus_50x2_plus_625_eq_0_l171_171073

theorem smallest_solution_x4_minus_50x2_plus_625_eq_0 :
  ∃ (x : ℝ), (x * x * x * x - 50 * x * x + 625 = 0) ∧ (∀ y, (y * y * y * y - 50 * y * y + 625 = 0) → x ≤ y) :=
sorry

end smallest_solution_x4_minus_50x2_plus_625_eq_0_l171_171073


namespace geometric_sequence_sum_l171_171772

theorem geometric_sequence_sum :
  ∃ (a : ℕ → ℕ) (q : ℕ), 
    a 1 = 1 ∧ 
    (∀ n, a n = q^(n-1)) ∧ 
    (4 * a 2, 2 * a 3, a 4 form arithmetic_sequence) ∧
    (a 2 + a 3 + a 4 = 14) := 
begin
  sorry
end

end geometric_sequence_sum_l171_171772


namespace woman_waits_time_after_passing_l171_171575

-- Definitions based only on the conditions in a)
def man_speed : ℝ := 5 -- in miles per hour
def woman_speed : ℝ := 25 -- in miles per hour
def waiting_time_man_minutes : ℝ := 20 -- in minutes

-- Equivalent proof problem statement
theorem woman_waits_time_after_passing :
  let waiting_time_man_hours := waiting_time_man_minutes / 60
  let distance_man : ℝ := man_speed * waiting_time_man_hours
  let relative_speed : ℝ := woman_speed - man_speed
  let time_woman_covers_distance_hours := distance_man / relative_speed
  let time_woman_covers_distance_minutes := time_woman_covers_distance_hours * 60
  time_woman_covers_distance_minutes = 5 :=
by
  sorry

end woman_waits_time_after_passing_l171_171575


namespace Taimour_painting_time_l171_171850

theorem Taimour_painting_time (T : ℝ) 
  (h1 : ∀ (T : ℝ), Jamshid_time = 0.5 * T) 
  (h2 : (1 / T + 2 / T) * 7 = 1) : 
    T = 21 :=
by
  sorry

end Taimour_painting_time_l171_171850


namespace inflation_two_years_real_rate_of_return_l171_171707

-- Proof Problem for Question 1
theorem inflation_two_years :
  ((1 + 0.015)^2 - 1) * 100 = 3.0225 :=
by
  sorry

-- Proof Problem for Question 2
theorem real_rate_of_return :
  ((1.07 * 1.07) / (1 + 0.030225) - 1) * 100 = 11.13 :=
by
  sorry

end inflation_two_years_real_rate_of_return_l171_171707


namespace fraction_equality_l171_171610

theorem fraction_equality : (16 : ℝ) / (8 * 17) = (1.6 : ℝ) / (0.8 * 17) := 
sorry

end fraction_equality_l171_171610


namespace banana_to_pear_equiv_l171_171946

/-
Given conditions:
1. 5 bananas cost as much as 3 apples.
2. 9 apples cost the same as 6 pears.
Prove the equivalence between 30 bananas and 12 pears.

We will define the equivalences as constants and prove the cost equivalence.
-/

variable (cost_banana cost_apple cost_pear : ℤ)

noncomputable def cost_equiv : Prop :=
  (5 * cost_banana = 3 * cost_apple) ∧ 
  (9 * cost_apple = 6 * cost_pear) →
  (30 * cost_banana = 12 * cost_pear)

theorem banana_to_pear_equiv :
  cost_equiv cost_banana cost_apple cost_pear :=
by
  sorry

end banana_to_pear_equiv_l171_171946


namespace number_of_cars_parked_l171_171849

-- Definitions for the given conditions
def total_area (length width : ℕ) : ℕ := length * width
def usable_area (total : ℕ) : ℕ := (8 * total) / 10
def cars_parked (usable : ℕ) (area_per_car : ℕ) : ℕ := usable / area_per_car

-- Given conditions
def length : ℕ := 400
def width : ℕ := 500
def area_per_car : ℕ := 10
def expected_cars : ℕ := 16000 -- correct answer from solution

-- Define a proof statement
theorem number_of_cars_parked : cars_parked (usable_area (total_area length width)) area_per_car = expected_cars := by
  sorry

end number_of_cars_parked_l171_171849


namespace max_gcd_13n_plus_4_8n_plus_3_l171_171596

theorem max_gcd_13n_plus_4_8n_plus_3 : 
  ∀ n : ℕ, n > 0 → ∃ d : ℕ, d = 7 ∧ ∀ k : ℕ, k = gcd (13 * n + 4) (8 * n + 3) → k ≤ d :=
by
  sorry

end max_gcd_13n_plus_4_8n_plus_3_l171_171596


namespace octagon_area_l171_171933

theorem octagon_area 
  (r : ℝ) 
  (h_regular : true) 
  (h_inscribed : true)
  (h_radius : r = 3) : 
  ∃ A, A = 18 * real.sqrt 2 :=
by
  sorry

end octagon_area_l171_171933


namespace smallest_possible_value_of_N_l171_171823

-- Declares the context and required constraints
theorem smallest_possible_value_of_N (N : ℕ) (h1 : 70 < N) (h2 : 70 ∣ 21 * N) : N = 80 :=
by
  sorry

end smallest_possible_value_of_N_l171_171823


namespace lines_perpendicular_l171_171158

variable (b : ℝ)

/-- Proof that if the given lines are perpendicular, then b must be 3 -/
theorem lines_perpendicular (h : b ≠ 0) :
    let l₁_slope := -3
    let l₂_slope := b / 9
    l₁_slope * l₂_slope = -1 → b = 3 :=
by
  intros slope_prod
  simp only [h]
  sorry

end lines_perpendicular_l171_171158


namespace overall_gain_percentage_l171_171927

noncomputable theory

-- Define conditions
def investment_stock_market := 5000
def investment_artwork := 10000
def investment_crypto := 15000
def returns_stock_market := 6000
def sale_price_artwork := 12000
def sales_tax_artwork := 0.05
def crypto_amount_rub := 17000
def conversion_rate := 1.03
def exchange_fee_crypto := 0.02

-- Define initial investment
def total_initial_investment := investment_stock_market + investment_artwork + investment_crypto

-- Define returns on investments
def net_return_artwork := sale_price_artwork - (sales_tax_artwork * sale_price_artwork)
def gross_return_crypto := crypto_amount_rub * conversion_rate
def net_return_crypto := gross_return_crypto - (exchange_fee_crypto * gross_return_crypto)

-- Define total returns
def total_returns := returns_stock_market + net_return_artwork + net_return_crypto

-- Define overall gain and gain percentage
def overall_gain := total_returns - total_initial_investment
def gain_percentage := (overall_gain / total_initial_investment) * 100

-- Lean statement to prove the overall gain percentage
theorem overall_gain_percentage :
  abs (gain_percentage - 15.20) < 0.01 :=
by
  sorry

end overall_gain_percentage_l171_171927


namespace domain_f_domain_g_intersection_M_N_l171_171001

namespace MathProof

open Set

def M : Set ℝ := { x | -2 < x ∧ x < 4 }
def N : Set ℝ := { x | x < 1 ∨ x ≥ 3 }

theorem domain_f :
  (M = { x : ℝ | -2 < x ∧ x < 4 }) := by
  sorry

theorem domain_g :
  (N = { x : ℝ | x < 1 ∨ x ≥ 3 }) := by
  sorry

theorem intersection_M_N : 
  (M ∩ N = { x : ℝ | (-2 < x ∧ x < 1) ∨ (3 ≤ x ∧ x < 4) }) := by
  sorry

end MathProof

end domain_f_domain_g_intersection_M_N_l171_171001


namespace line_intersects_x_axis_at_10_0_l171_171278

theorem line_intersects_x_axis_at_10_0 :
  let x1 := 9
  let y1 := 1
  let x2 := 5
  let y2 := 5
  let slope := (y2 - y1) / (x2 - x1)
  let y := 0
  ∃ x, (x - x1) * slope = y - y1 ∧ y = 0 → x = 10 := by
  sorry

end line_intersects_x_axis_at_10_0_l171_171278


namespace linda_total_distance_l171_171661

theorem linda_total_distance
  (miles_per_gallon : ℝ) (tank_capacity : ℝ) (initial_distance : ℝ) (refuel_amount : ℝ) (final_tank_fraction : ℝ)
  (fuel_used_first_segment : ℝ := initial_distance / miles_per_gallon)
  (initial_fuel_full : fuel_used_first_segment = tank_capacity)
  (total_fuel_after_refuel : ℝ := 0 + refuel_amount)
  (remaining_fuel_stopping : ℝ := final_tank_fraction * tank_capacity)
  (fuel_used_second_segment : ℝ := total_fuel_after_refuel - remaining_fuel_stopping)
  (distance_second_leg : ℝ := fuel_used_second_segment * miles_per_gallon) :
  initial_distance + distance_second_leg = 637.5 := by
  sorry

end linda_total_distance_l171_171661


namespace min_sum_of_consecutive_natural_numbers_l171_171063

theorem min_sum_of_consecutive_natural_numbers (a b c : ℕ) 
  (h1 : a + 1 = b)
  (h2 : a + 2 = c)
  (h3 : a % 9 = 0)
  (h4 : b % 8 = 0)
  (h5 : c % 7 = 0) :
  a + b + c = 1488 :=
sorry

end min_sum_of_consecutive_natural_numbers_l171_171063


namespace towels_folded_in_one_hour_l171_171219

theorem towels_folded_in_one_hour :
  let jane_rate := 12 * 5 -- Jane's rate in towels/hour
  let kyla_rate := 6 * 9  -- Kyla's rate in towels/hour
  let anthony_rate := 3 * 14 -- Anthony's rate in towels/hour
  let david_rate := 4 * 6 -- David's rate in towels/hour
  jane_rate + kyla_rate + anthony_rate + david_rate = 180 := 
by
  let jane_rate := 12 * 5
  let kyla_rate := 6 * 9
  let anthony_rate := 3 * 14
  let david_rate := 4 * 6
  show jane_rate + kyla_rate + anthony_rate + david_rate = 180
  sorry

end towels_folded_in_one_hour_l171_171219


namespace roots_difference_squared_l171_171654

-- Defining the solutions to the quadratic equation
def quadratic_equation_roots (a b : ℚ) : Prop :=
  (2 * a^2 - 7 * a + 6 = 0) ∧ (2 * b^2 - 7 * b + 6 = 0)

-- The main theorem we aim to prove
theorem roots_difference_squared (a b : ℚ) (h : quadratic_equation_roots a b) :
    (a - b)^2 = 1 / 4 := 
  sorry

end roots_difference_squared_l171_171654


namespace expression_nonnegative_l171_171156

theorem expression_nonnegative (x : ℝ) :
  0 <= x ∧ x < 3 → (2*x - 6*x^2 + 9*x^3) / (9 - x^3) ≥ 0 := 
by
  sorry

end expression_nonnegative_l171_171156


namespace simplify_expression_l171_171351

theorem simplify_expression :
  ((3 + 5 + 6 + 2) / 3) + ((2 * 3 + 4 * 2 + 5) / 4) = 121 / 12 :=
by
  sorry

end simplify_expression_l171_171351


namespace find_parallel_line_l171_171959

/-- 
Given a line l with equation 3x - 2y + 1 = 0 and a point A(1,1).
Find the equation of a line that passes through A and is parallel to l.
-/
theorem find_parallel_line (a b c : ℝ) (p_x p_y : ℝ) 
    (h₁ : 3 * p_x - 2 * p_y + c = 0) 
    (h₂ : p_x = 1 ∧ p_y = 1)
    (h₃ : a = 3 ∧ b = -2) :
    3 * x - 2 * y - 1 = 0 := 
by 
  sorry

end find_parallel_line_l171_171959


namespace symmetric_axis_and_vertex_l171_171693

theorem symmetric_axis_and_vertex (x : ℝ) : 
  (∀ x y, y = (1 / 2) * (x - 1)^2 + 6 → x = 1) 
  ∧ (1, 6) = (1, 6) :=
by 
  sorry

end symmetric_axis_and_vertex_l171_171693


namespace sin_ninety_deg_l171_171141

theorem sin_ninety_deg : Real.sin (Float.pi / 2) = 1 := 
by sorry

end sin_ninety_deg_l171_171141


namespace sin_90_eq_one_l171_171150

-- Definition of the rotation by 90 degrees counterclockwise
def rotate90 (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, p.1)

-- Definition of the sine function for a 90 degree angle
def sin90 : ℝ :=
  let initial_point := (1, 0)
  let rotated_point := rotate90 initial_point
  rotated_point.2

-- Theorem to be proven: sin90 should be equal to 1
theorem sin_90_eq_one : sin90 = 1 :=
by
  sorry

end sin_90_eq_one_l171_171150


namespace happy_children_count_l171_171662

theorem happy_children_count (total_children sad_children neither_children total_boys total_girls happy_boys sad_girls : ℕ)
  (h1 : total_children = 60)
  (h2 : sad_children = 10)
  (h3 : neither_children = 20)
  (h4 : total_boys = 18)
  (h5 : total_girls = 42)
  (h6 : happy_boys = 6)
  (h7 : sad_girls = 4) :
  ∃ happy_children, happy_children = 30 :=
  sorry

end happy_children_count_l171_171662


namespace smallest_N_value_proof_l171_171799

def smallest_value_N (N : ℕ) : Prop :=
  N > 70 ∧ (21 * N) % 70 = 0

theorem smallest_N_value_proof : ∃ N, smallest_value_N N ∧ (∀ M, smallest_value_N M → N ≤ M) :=
  sorry

end smallest_N_value_proof_l171_171799


namespace sin_90_eq_1_l171_171129

theorem sin_90_eq_1 : Real.sin (Float.pi / 2) = 1 := by
  sorry

end sin_90_eq_1_l171_171129


namespace range_of_a_for_three_distinct_zeros_l171_171002

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - 3 * x + a

theorem range_of_a_for_three_distinct_zeros : 
  ∀ a : ℝ, (∀ x y : ℝ, x ≠ y → f x a = 0 → f y a = 0 → (f (1:ℝ) a < 0 ∧ f (-1:ℝ) a > 0)) ↔ (-2 < a ∧ a < 2) := 
by
  sorry

end range_of_a_for_three_distinct_zeros_l171_171002


namespace third_side_length_l171_171787

theorem third_side_length (x : ℝ) (h1 : 5 > 0) (h2 : 8 > 0) (h3 : 3 < x < 13) : (5 < x + 8) ∧ (x < 5 + 8) ∧ (5 < x + 3) ∧ (x < 8 + 5) := 
by
  sorry

end third_side_length_l171_171787


namespace smallest_of_seven_consecutive_even_numbers_l171_171520

theorem smallest_of_seven_consecutive_even_numbers (n : ℤ) :
  (n - 6) + (n - 4) + (n - 2) + n + (n + 2) + (n + 4) + (n + 6) = 448 → 
  (n - 6) = 58 :=
by
  sorry

end smallest_of_seven_consecutive_even_numbers_l171_171520


namespace fraction_increase_l171_171300

variable (P A : ℝ)
variable (f : ℝ)

theorem fraction_increase (hP : P = 2880) (hA : A = 3645) (h : A = P * (1 + f) ^ 2) : f = 0.125 := by
  sorry

end fraction_increase_l171_171300


namespace polynomial_solution_l171_171301

noncomputable def polynomial_form (P : ℝ → ℝ) : Prop :=
∀ (x y z : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ (2 * x * y * z = x + y + z) →
(P x / (y * z) + P y / (z * x) + P z / (x * y) = P (x - y) + P (y - z) + P (z - x))

theorem polynomial_solution (P : ℝ → ℝ) : polynomial_form P → ∃ c : ℝ, ∀ x : ℝ, P x = c * (x ^ 2 + 3) := 
by 
  sorry

end polynomial_solution_l171_171301


namespace min_abs_sum_l171_171169

theorem min_abs_sum (x y : ℝ) : |x - 1| + |x| + |y - 1| + |y + 1| ≥ 3 :=
by sorry

end min_abs_sum_l171_171169


namespace ratio_of_a_over_3_to_b_over_2_l171_171561

theorem ratio_of_a_over_3_to_b_over_2 (a b : ℝ) (h1 : 2 * a = 3 * b) (h2 : a * b ≠ 0) : (a / 3) / (b / 2) = 1 :=
by
  sorry

end ratio_of_a_over_3_to_b_over_2_l171_171561


namespace possible_values_quotient_l171_171580

theorem possible_values_quotient (α : ℝ) (h_pos : α > 0) (h_rounded : ∃ (n : ℕ) (α1 : ℝ), α = n / 100 + α1 ∧ 0 ≤ α1 ∧ α1 < 1 / 100) :
  ∃ (values : List ℝ), values = [0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59,
                                  0.60, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69,
                                  0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79,
                                  0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89,
                                  0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99,
                                  1.00] :=
  sorry

end possible_values_quotient_l171_171580


namespace amount_for_second_shop_l171_171667

-- Definitions based on conditions
def books_from_first_shop : Nat := 65
def amount_first_shop : Float := 1160.0
def books_from_second_shop : Nat := 50
def avg_price_per_book : Float := 18.08695652173913
def total_books : Nat := books_from_first_shop + books_from_second_shop
def total_amount_spent : Float := avg_price_per_book * (total_books.toFloat)

-- The Lean statement to prove
theorem amount_for_second_shop : total_amount_spent - amount_first_shop = 920.0 := by
  sorry

end amount_for_second_shop_l171_171667


namespace largest_number_after_removal_l171_171952

theorem largest_number_after_removal :
  ∀ (s : Nat), s = 1234567891011121314151617181920 -- representing the start of the sequence
  → true
  := by
    sorry

end largest_number_after_removal_l171_171952


namespace find_a_l171_171975

variable (a b c : ℝ)

def f (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem find_a (h1 : a ≠ 0) (h2 : f a b c (-1) = 0)
    (h3 : ∀ x : ℝ, x ≤ f a b c x ∧ f a b c x ≤ (1/2) * (x^2 + 1)) :
  a = 1/2 :=
by
  sorry

end find_a_l171_171975


namespace inverse_h_l171_171656

-- definitions of f, g, and h
def f (x : ℝ) : ℝ := 4 * x - 5
def g (x : ℝ) : ℝ := 3 * x + 7
def h (x : ℝ) : ℝ := f (g x)

-- statement of the problem
theorem inverse_h (x : ℝ) : (∃ y : ℝ, h y = x) ∧ ∀ y : ℝ, h y = x → y = (x - 23) / 12 :=
by
  sorry

end inverse_h_l171_171656


namespace canteen_needs_bananas_l171_171571

-- Define the given conditions
def total_bananas := 9828
def weeks := 9
def days_in_week := 7
def bananas_in_dozen := 12

-- Calculate the required value and prove the equivalence
theorem canteen_needs_bananas : 
  (total_bananas / (weeks * days_in_week)) / bananas_in_dozen = 13 :=
by
  -- This is where the proof would go
  sorry

end canteen_needs_bananas_l171_171571


namespace table_tennis_expected_games_l171_171210

open ProbabilityTheory
open MeasureTheory
open Localization
open ENNReal

noncomputable def expected_games_stop : ℚ :=
  97 / 32

theorem table_tennis_expected_games :
  ∀ (E : Type) [Fintype E],
  let prob_A := (3 / 4 : ℚ)
  let prob_B := (1 / 4 : ℚ)
  let max_games := 6
  let xi : ℕ := expected_value_of_games (prob_A, prob_B) max_games
  (∀ i ∈ (finset.range max_games), i % 2 = 0 → xi <= max_games → xi ≤ max_games) →
  (∑ i in (finset.range max_games), i * (prob_A ^ (i / 2) * prob_B ^ (i / 2)) = expected_games_stop) := 
  sorry

end table_tennis_expected_games_l171_171210


namespace sin_90_degrees_l171_171125

theorem sin_90_degrees : Real.sin (Float.pi / 2) = 1 :=
by
  sorry

end sin_90_degrees_l171_171125


namespace price_of_turban_l171_171910

theorem price_of_turban (T : ℝ) (h1 : ∀ (T : ℝ), 3 / 4 * (90 + T) = 40 + T) : T = 110 :=
by
  sorry

end price_of_turban_l171_171910


namespace exists_prime_q_l171_171335

theorem exists_prime_q (p : ℕ) (hp : Nat.Prime p) (h2 : 2 < p) : 
  ∃ q : ℕ, Nat.Prime q ∧ q < p ∧ ¬ (p ^ 2 ∣ q ^ (p - 1) - 1) := 
sorry

end exists_prime_q_l171_171335


namespace initial_percentage_filled_l171_171915

theorem initial_percentage_filled (capacity : ℝ) (added : ℝ) (final_fraction : ℝ) (initial_water : ℝ) :
  capacity = 80 → added = 20 → final_fraction = 3/4 → 
  initial_water = (final_fraction * capacity - added) → 
  100 * (initial_water / capacity) = 50 :=
by
  intros
  sorry

end initial_percentage_filled_l171_171915


namespace pyramid_base_length_l171_171871

theorem pyramid_base_length (A s h : ℝ): A = 120 ∧ h = 40 ∧ (A = 1/2 * s * h) → s = 6 := 
by
  sorry

end pyramid_base_length_l171_171871


namespace initial_volume_of_mixture_l171_171576

-- Define the initial condition volumes for p and q
def initial_volumes (x : ℕ) : ℕ × ℕ := (3 * x, 2 * x)

-- Define the final condition volumes for p and q after adding 2 liters of q
def final_volumes (x : ℕ) : ℕ × ℕ := (3 * x, 2 * x + 2)

-- Define the initial total volume of the mixture
def initial_volume (x : ℕ) : ℕ := 5 * x

-- The theorem stating the solution
theorem initial_volume_of_mixture (x : ℕ) (h : 3 * x / (2 * x + 2) = 5 / 4) : 5 * x = 25 := 
by sorry

end initial_volume_of_mixture_l171_171576


namespace sin_ninety_degrees_l171_171134

theorem sin_ninety_degrees : Real.sin (90 * Real.pi / 180) = 1 := 
by
  sorry

end sin_ninety_degrees_l171_171134


namespace baker_sales_difference_l171_171569

/-!
  Prove that the difference in dollars between the baker's daily average sales and total sales for today is 48 dollars.
-/

theorem baker_sales_difference :
  let price_pastry := 2
  let price_bread := 4
  let avg_pastries := 20
  let avg_bread := 10
  let today_pastries := 14
  let today_bread := 25
  let daily_avg_sales := avg_pastries * price_pastry + avg_bread * price_bread
  let today_sales := today_pastries * price_pastry + today_bread * price_bread
  daily_avg_sales - today_sales = 48 :=
sorry

end baker_sales_difference_l171_171569


namespace choose_7_from_16_l171_171062

theorem choose_7_from_16 : (Nat.choose 16 7) = 11440 := 
by
  sorry

end choose_7_from_16_l171_171062


namespace sin_90_degree_l171_171140

-- Definitions based on conditions
def unit_circle_point (angle : ℝ) : ℝ × ℝ :=
  if angle = 90 * (π / 180) then (0, 1) else sorry

def sin_usual (angle : ℝ) : ℝ :=
  (unit_circle_point angle).snd

-- The main theorem as per the question and conditions
theorem sin_90_degree : sin_usual (90 * (π / 180)) = 1 :=
by
  sorry

end sin_90_degree_l171_171140


namespace gaoan_total_revenue_in_scientific_notation_l171_171585

theorem gaoan_total_revenue_in_scientific_notation :
  (21 * 10^9 : ℝ) = 2.1 * 10^9 :=
sorry

end gaoan_total_revenue_in_scientific_notation_l171_171585


namespace angie_total_taxes_l171_171109

theorem angie_total_taxes:
  ∀ (salary : ℕ) (N_1 N_2 N_3 N_4 T_1 T_2 T_3 T_4 U_1 U_2 U_3 U_4 left_over : ℕ),
  salary = 80 →
  N_1 = 12 → T_1 = 8 → U_1 = 5 →
  N_2 = 15 → T_2 = 6 → U_2 = 7 →
  N_3 = 10 → T_3 = 9 → U_3 = 6 →
  N_4 = 14 → T_4 = 7 → U_4 = 4 →
  left_over = 18 →
  T_1 + T_2 + T_3 + T_4 = 30 :=
by
  intros salary N_1 N_2 N_3 N_4 T_1 T_2 T_3 T_4 U_1 U_2 U_3 U_4 left_over
  sorry

end angie_total_taxes_l171_171109


namespace exists_gcd_one_l171_171038

theorem exists_gcd_one (p q r : ℤ) (hp : p > 0) (hq : q > 0) (hr : r > 0) 
  (h : Int.gcd p (Int.gcd q r) = 1) : ∃ a : ℤ, Int.gcd p (q + a * r) = 1 :=
sorry

end exists_gcd_one_l171_171038


namespace minor_premise_wrong_l171_171721

noncomputable def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = f x

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

def f (x : ℝ) : ℝ := x^2 + x

theorem minor_premise_wrong : ¬ is_even_function f ∧ ¬ is_odd_function f := 
by
  sorry

end minor_premise_wrong_l171_171721


namespace simplify_expression_l171_171966

noncomputable def simplified_result (a b : ℝ) (i : ℂ) (hi : i * i = -1) : ℂ :=
  (a + b * i) * (a - b * i)

theorem simplify_expression (a b : ℝ) (i : ℂ) (hi : i * i = -1) :
  simplified_result a b i hi = a^2 + b^2 := by
  sorry

end simplify_expression_l171_171966


namespace median_of_100_numbers_l171_171836

theorem median_of_100_numbers (numbers : List ℝ) (h_length : numbers.length = 100)
  (h_sorted : numbers.sorted (≤))
  (h_51 : numbers.nth_le 50 h_51_nat = 78) -- Note: nth_le is zero-indexed, so 51st element is 50th index.
  (h_50 : numbers.nth_le 49 h_50_nat = 66) : 
  (numbers.nth_le 49 h_50_nat + numbers.nth_le 50 h_51_nat) / 2 = 72 :=
sorry

end median_of_100_numbers_l171_171836


namespace coordinates_with_respect_to_origin_l171_171875

theorem coordinates_with_respect_to_origin (x y : ℤ) (h : (x, y) = (3, -2)) : (x, y) = (3, -2) :=
by
  -- Given that the point (x, y) is (3, -2)
  rw h

end coordinates_with_respect_to_origin_l171_171875


namespace factorial_base_8_zeroes_l171_171993

theorem factorial_base_8_zeroes (n : ℕ) :
  n = 15 →
  largest_power_8_dividing_factorial_n = 3 :=
begin
  assume hn : n = 15,
  -- Definitions and setup based on conditions
  let k := largest_power_of_prime_dividing_factorial 2 n,
  have hk : k = 11, 
  { sorry }, -- Sum the factors of 2 as shown in the solution steps
  have hp8 : largest_power_8_dividing_factorial_n = k / 3,
  { sorry }, -- Calculate the integer division k / 3 to find power of 8 division
  rw hn at *,
  exact eq.trans hp8.symm (nat.div_eq_of_lt_trans (nat.lt_succ_self 2 * (k / 3))),
end

end factorial_base_8_zeroes_l171_171993


namespace negation_of_exists_statement_l171_171884

theorem negation_of_exists_statement :
  ¬ (∃ x0 : ℝ, x0 > 0 ∧ x0^2 - 5 * x0 + 6 > 0) ↔ ∀ x : ℝ, x > 0 → x^2 - 5 * x + 6 ≤ 0 :=
by
  sorry

end negation_of_exists_statement_l171_171884


namespace combined_weight_l171_171479

-- Define the main proof problem
theorem combined_weight (student_weight : ℝ) (sister_weight : ℝ) :
  (student_weight - 5 = 2 * sister_weight) ∧ (student_weight = 79) → (student_weight + sister_weight = 116) :=
by
  sorry

end combined_weight_l171_171479


namespace expand_polynomial_l171_171161

theorem expand_polynomial (z : ℂ) :
  (3 * z^3 + 4 * z^2 - 5 * z + 1) * (2 * z^2 - 3 * z + 4) * (z - 1) = 3 * z^6 - 3 * z^5 + 5 * z^4 + 2 * z^3 - 5 * z^2 + 4 * z - 16 :=
by sorry

end expand_polynomial_l171_171161


namespace factorial_trailing_zeros_base_8_l171_171995

/-- Number of trailing zeros of 15! in base 8 is 3 -/
theorem factorial_trailing_zeros_base_8 : number_of_trailing_zeros_in_base 15! 8 = 3 := sorry

end factorial_trailing_zeros_base_8_l171_171995


namespace sheets_in_stack_l171_171279

theorem sheets_in_stack (n : ℕ) (thickness : ℝ) (height : ℝ) 
  (h1 : n = 400) (h2 : thickness = 4) (h3 : height = 10) : 
  n * height / thickness = 1000 := 
by 
  sorry

end sheets_in_stack_l171_171279


namespace regular_octagon_area_l171_171939

-- Definitions based on conditions
def is_regular_octagon (p : ℝ → Prop) : Prop := 
  ∀ θ, 0 ≤ θ ∧ θ < 7 * (π / 4) → p θ = p (θ + π / 4)

def inscribed_in_circle (p : ℝ → Prop) (r : ℝ) : Prop :=
  ∀ θ, 0 ≤ θ ∧ θ < 2 * π → p θ = r

-- The proof statement
theorem regular_octagon_area 
  (r : ℝ) (h_r : r = 3) 
  (p : ℝ → Prop)
  (h_regular : is_regular_octagon p)
  (h_inscribed : inscribed_in_circle p r) :
  ∃ a : ℝ, a = 14.92 := 
sorry

end regular_octagon_area_l171_171939


namespace total_cost_l171_171968

-- Definition of the conditions
def cost_sharing (x : ℝ) : Prop :=
  let initial_cost := x / 5
  let new_cost := x / 7
  initial_cost - 15 = new_cost

-- The statement we need to prove
theorem total_cost (x : ℝ) (h : cost_sharing x) : x = 262.50 := by
  sorry

end total_cost_l171_171968


namespace technicians_count_l171_171049

noncomputable def total_salary := 8000 * 21
noncomputable def average_salary_all := 8000
noncomputable def average_salary_technicians := 12000
noncomputable def average_salary_rest := 6000
noncomputable def total_workers := 21

theorem technicians_count :
  ∃ (T R : ℕ),
  T + R = total_workers ∧
  average_salary_technicians * T + average_salary_rest * R = total_salary ∧
  T = 7 :=
by
  sorry

end technicians_count_l171_171049


namespace reciprocal_geometric_sum_l171_171922

/-- The sum of the new geometric progression formed by taking the reciprocal of each term in the original progression,
    where the original progression has 10 terms, the first term is 2, and the common ratio is 3, is \( \frac{29524}{59049} \). -/
theorem reciprocal_geometric_sum :
  let a := 2
  let r := 3
  let n := 10
  let sn := (2 * (1 - r^n)) / (1 - r)
  let sn_reciprocal := (1 / a) * (1 - (1/r)^n) / (1 - 1/r)
  (sn_reciprocal = 29524 / 59049) :=
by
  sorry

end reciprocal_geometric_sum_l171_171922


namespace school_orchestra_members_l171_171386

theorem school_orchestra_members (total_members can_play_violin can_play_keyboard neither : ℕ)
    (h1 : total_members = 42)
    (h2 : can_play_violin = 25)
    (h3 : can_play_keyboard = 22)
    (h4 : neither = 3) :
    (can_play_violin + can_play_keyboard) - (total_members - neither) = 8 :=
by
  sorry

end school_orchestra_members_l171_171386


namespace find_rate_l171_171424

def plan1_cost (minutes : ℕ) : ℝ :=
  if minutes <= 500 then 50 else 50 + (minutes - 500) * 0.35

def plan2_cost (minutes : ℕ) (x : ℝ) : ℝ :=
  if minutes <= 1000 then 75 else 75 + (minutes - 1000) * x

theorem find_rate (x : ℝ) :
  plan1_cost 2500 = plan2_cost 2500 x → x = 0.45 := by
  sorry

end find_rate_l171_171424


namespace union_M_N_l171_171984

def U : Set ℝ := {x | -3 ≤ x ∧ x < 2}
def M : Set ℝ := {x | -1 < x ∧ x < 1}
def complement_U_N : Set ℝ := {x | 0 < x ∧ x < 2}
def N : Set ℝ := {x | -3 ≤ x ∧ x ≤ 0}

theorem union_M_N :
  M ∪ N = {x | -3 ≤ x ∧ x < 1} := 
sorry

end union_M_N_l171_171984


namespace set_equivalence_l171_171555

theorem set_equivalence :
  {p : ℝ × ℝ | p.1 + p.2 = 1 ∧ 2 * p.1 - p.2 = 2} = {(1, 0)} :=
by
  sorry

end set_equivalence_l171_171555


namespace sum_of_sines_leq_3_sqrt3_over_2_l171_171620

theorem sum_of_sines_leq_3_sqrt3_over_2 (α β γ : ℝ) (h : α + β + γ = Real.pi) :
  Real.sin α + Real.sin β + Real.sin γ ≤ 3 * Real.sqrt 3 / 2 :=
sorry

end sum_of_sines_leq_3_sqrt3_over_2_l171_171620


namespace possible_values_of_k_l171_171234

theorem possible_values_of_k (n : ℕ) (h : n ≥ 3) :
  ∃ t : ℕ, k = 2 ^ t ∧ 2 ^ t ≥ n :=
sorry

end possible_values_of_k_l171_171234


namespace monotonically_increasing_interval_l171_171767

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2*x) / Real.log (1/2)

theorem monotonically_increasing_interval :
  ∀ x, x ∈ Set.Iio (0 : ℝ) → StrictMono f :=
by
  sorry

end monotonically_increasing_interval_l171_171767


namespace smallest_solution_of_quartic_equation_l171_171080

theorem smallest_solution_of_quartic_equation :
  ∃ x : ℝ, (x^4 - 50 * x^2 + 625 = 0) ∧ ∀ y : ℝ, y^4 - 50 * y^2 + 625 = 0 → x ≤ y := sorry

end smallest_solution_of_quartic_equation_l171_171080


namespace smallest_possible_N_l171_171797

theorem smallest_possible_N (N : ℕ) (h1 : N > 70) (h2 : 70 ∣ 21 * N) : N = 80 :=
sorry

end smallest_possible_N_l171_171797


namespace valid_third_side_length_l171_171779

theorem valid_third_side_length {x : ℝ} (h1 : 5 + 8 > x) (h2 : 5 + x > 8) (h3 : 8 + x > 5) : x = 6 :=
by
  -- Given 5 + 8 > x, 5 + x > 8, 8 + x > 5
  have range1 : 13 > x := h1,
  have range2 : x > 3 := (by linarith [h2]),
  have _ : 3 < 6 ∧ 6 < 13 := by norm_num,
  linarith

#check valid_third_side_length

end valid_third_side_length_l171_171779


namespace predicted_customers_on_Saturday_l171_171274

theorem predicted_customers_on_Saturday 
  (breakfast_customers : ℕ)
  (lunch_customers : ℕ)
  (dinner_customers : ℕ)
  (prediction_factor : ℕ)
  (h1 : breakfast_customers = 73)
  (h2 : lunch_customers = 127)
  (h3 : dinner_customers = 87)
  (h4 : prediction_factor = 2) :
  prediction_factor * (breakfast_customers + lunch_customers + dinner_customers) = 574 :=  
by 
  sorry 

end predicted_customers_on_Saturday_l171_171274


namespace function_g_l171_171380

theorem function_g (g : ℝ → ℝ) (t : ℝ) :
  (∀ t, (20 * t - 14) = 2 * (g t) - 40) → (g t = 10 * t + 13) :=
by
  intro h
  have h1 : 20 * t - 14 = 2 * (g t) - 40 := h t
  sorry

end function_g_l171_171380


namespace common_root_of_two_equations_l171_171171

theorem common_root_of_two_equations (m x : ℝ) :
  (m * x - 1000 = 1001) ∧ (1001 * x = m - 1000 * x) → (m = 2001 ∨ m = -2001) :=
by
  sorry

end common_root_of_two_equations_l171_171171


namespace find_certain_number_l171_171057

theorem find_certain_number (x y : ℝ)
  (h1 : (28 + x + 42 + y + 104) / 5 = 62)
  (h2 : (48 + 62 + 98 + 124 + x) / 5 = 78) :
  y = 78 :=
by
  sorry

end find_certain_number_l171_171057


namespace larger_number_l171_171681

theorem larger_number (A B : ℝ) (h1 : A - B = 1650) (h2 : 0.075 * A = 0.125 * B) : A = 4125 :=
sorry

end larger_number_l171_171681


namespace multiply_scientific_notation_l171_171446

theorem multiply_scientific_notation (a b : ℝ) (e1 e2 : ℤ) 
  (h1 : a = 2) (h2 : b = 8) (h3 : e1 = 3) (h4 : e2 = 3) :
  (a * 10^e1) * (b * 10^e2) = 1.6 * 10^7 :=
by
  simp [h1, h2, h3, h4]
  sorry

end multiply_scientific_notation_l171_171446


namespace smallest_of_seven_even_numbers_sum_448_l171_171518

theorem smallest_of_seven_even_numbers_sum_448 :
  ∃ n : ℤ, n + (n+2) + (n+4) + (n+6) + (n+8) + (n+10) + (n+12) = 448 ∧ n = 58 := 
by
  sorry

end smallest_of_seven_even_numbers_sum_448_l171_171518


namespace milk_distribution_l171_171215

theorem milk_distribution 
  (x y z : ℕ)
  (h_total : x + y + z = 780)
  (h_equiv : 3 * x / 4 = 4 * y / 5 ∧ 3 * x / 4 = 4 * z / 7) :
  x = 240 ∧ y = 225 ∧ z = 315 := 
sorry

end milk_distribution_l171_171215


namespace possible_third_side_l171_171784

theorem possible_third_side {x : ℕ} (h_option_A : x = 2) (h_option_B : x = 3) (h_option_C : x = 6) (h_option_D : x = 13) : 3 < x ∧ x < 13 ↔ x = 6 :=
by
  sorry

end possible_third_side_l171_171784


namespace a_five_minus_a_divisible_by_five_l171_171346

theorem a_five_minus_a_divisible_by_five (a : ℤ) : 5 ∣ (a^5 - a) :=
by
  -- proof steps
  sorry

end a_five_minus_a_divisible_by_five_l171_171346


namespace points_where_star_is_commutative_are_on_line_l171_171296

def star (a b : ℝ) : ℝ := a * b * (a - b)

theorem points_where_star_is_commutative_are_on_line :
  {p : ℝ × ℝ | star p.1 p.2 = star p.2 p.1} = {p : ℝ × ℝ | p.1 = p.2} :=
by
  sorry

end points_where_star_is_commutative_are_on_line_l171_171296


namespace number_of_possible_scenarios_l171_171835

-- Definitions based on conditions
def num_companies : Nat := 5
def reps_company_A : Nat := 2
def reps_other_companies : Nat := 1
def total_speakers : Nat := 3

-- Problem statement
theorem number_of_possible_scenarios : 
  ∃ (scenarios : Nat), scenarios = 16 ∧ 
  (scenarios = 
    (Nat.choose reps_company_A 1 * Nat.choose 4 2) + 
    Nat.choose 4 3) :=
by
  sorry

end number_of_possible_scenarios_l171_171835


namespace fish_population_estimate_l171_171484

theorem fish_population_estimate :
  ∃ N : ℕ, (60 * 60) / 2 = N ∧ (2 / 60 : ℚ) = (60 / N : ℚ) :=
by
  use 1800
  simp
  sorry

end fish_population_estimate_l171_171484


namespace alyssa_gave_away_puppies_l171_171945

def start_puppies : ℕ := 12
def remaining_puppies : ℕ := 5

theorem alyssa_gave_away_puppies : 
  start_puppies - remaining_puppies = 7 := 
by
  sorry

end alyssa_gave_away_puppies_l171_171945


namespace find_ending_number_l171_171737

theorem find_ending_number (n : ℕ) 
  (h1 : ∃ numbers, numbers = (filter (λ x, x % 2 = 0) (list.range' 12 (n - 12 + 1)))) 
  (h2 : (list.sum (filter (λ x, x % 2 = 0) (list.range' 12 (n - 12 + 1)))) / (list.length (filter (λ x, x % 2 = 0) (list.range' 12 (n - 12 + 1)))) = 19) :
  n = 26 := 
sorry

end find_ending_number_l171_171737


namespace simplify_and_evaluate_at_x_eq_4_l171_171350

noncomputable def simplify_and_evaluate (x : ℚ) : ℚ :=
  (x - 1 - (3 / (x + 1))) / ((x^2 - 2*x) / (x + 1))

theorem simplify_and_evaluate_at_x_eq_4 : simplify_and_evaluate 4 = 3 / 2 := by
  sorry

end simplify_and_evaluate_at_x_eq_4_l171_171350


namespace nth_equation_l171_171232

-- Define the product of a list of integers
def prod_list (lst : List ℕ) : ℕ :=
  lst.foldl (· * ·) 1

-- Define the product of first n odd numbers
def prod_odds (n : ℕ) : ℕ :=
  prod_list (List.map (λ i => 2 * i - 1) (List.range n))

-- Define the product of the range from n+1 to 2n
def prod_range (n : ℕ) : ℕ :=
  prod_list (List.range' (n + 1) n)

-- The theorem to prove
theorem nth_equation (n : ℕ) (hn : 0 < n) : prod_range n = 2^n * prod_odds n := 
  sorry

end nth_equation_l171_171232


namespace travel_time_by_raft_l171_171388

variable (U V : ℝ) -- U: speed of the steamboat, V: speed of the river current
variable (S : ℝ) -- S: distance between cities A and B

-- Conditions
variable (h1 : S = 12 * U - 15 * V) -- Distance calculation, city B to city A
variable (h2 : S = 8 * U + 10 * V)  -- Distance calculation, city A to city B
variable (T : ℝ) -- Time taken on a raft

-- Proof problem
theorem travel_time_by_raft : T = 60 :=
by
  sorry


end travel_time_by_raft_l171_171388


namespace tammy_total_distance_l171_171869

-- Define the times and speeds for each segment and breaks
def initial_speed : ℝ := 55   -- miles per hour
def initial_time : ℝ := 2     -- hours
def road_speed : ℝ := 40      -- miles per hour
def road_time : ℝ := 5        -- hours
def first_break : ℝ := 1      -- hour
def drive_after_break_speed : ℝ := 50  -- miles per hour
def drive_after_break_time : ℝ := 15   -- hours
def hilly_speed : ℝ := 35     -- miles per hour
def hilly_time : ℝ := 3       -- hours
def second_break : ℝ := 0.5   -- hours
def finish_speed : ℝ := 60    -- miles per hour
def total_journey_time : ℝ := 36 -- hours

-- Define a function to calculate the segment distance
def distance (speed time : ℝ) : ℝ := speed * time

-- Define the total distance calculation
def total_distance : ℝ :=
  distance initial_speed initial_time +
  distance road_speed road_time +
  distance drive_after_break_speed drive_after_break_time +
  distance hilly_speed hilly_time +
  distance finish_speed (total_journey_time - (initial_time + road_time + drive_after_break_time + hilly_time + first_break + second_break))

-- The final proof statement
theorem tammy_total_distance : total_distance = 1735 :=
  sorry

end tammy_total_distance_l171_171869


namespace project_completion_by_B_l171_171650

-- Definitions of the given conditions
def person_A_work_rate := 1 / 10
def person_B_work_rate := 1 / 15
def days_A_worked := 3

-- Definition of the mathematical proof problem
theorem project_completion_by_B {x : ℝ} : person_A_work_rate * days_A_worked + person_B_work_rate * x = 1 :=
by
  sorry

end project_completion_by_B_l171_171650


namespace extreme_values_l171_171524

noncomputable def f (x : ℝ) : ℝ := x + 4 / x

theorem extreme_values (x : ℝ) (hx : x ≠ 0) :
  (x = -2 → f x = -4 ∧ ∀ y, y > -2 → f y > -4) ∧
  (x = 2 → f x = 4 ∧ ∀ y, y < 2 → f y > 4) :=
sorry

end extreme_values_l171_171524


namespace area_of_inscribed_octagon_l171_171931

open Real

def regular_octagon_area {r : ℝ} (octagon : Prop) : ℝ :=
  if octagon then 8 * (1 / 2 * r^2 * sin (π / 4)) else 0

theorem area_of_inscribed_octagon (r : ℝ) (h1 : r = 3) (h2 : ∀ octagon, octagon → regular_octagon_area octagon = 18 * sqrt 2) :
  regular_octagon_area true = 18 * sqrt 2 :=
by
  rw [← h2 true]
  sorry

end area_of_inscribed_octagon_l171_171931


namespace arithmetic_progression_power_of_two_l171_171592

theorem arithmetic_progression_power_of_two 
  (a d : ℤ) (n : ℕ) (k : ℕ) 
  (Sn : ℤ)
  (h_sum : Sn = 2^k)
  (h_ap : Sn = n * (2 * a + (n - 1) * d) / 2)  :
  ∃ m : ℕ, n = 2^m := 
sorry

end arithmetic_progression_power_of_two_l171_171592


namespace product_lcm_gcd_l171_171167

def a : ℕ := 6
def b : ℕ := 8

theorem product_lcm_gcd : Nat.lcm a b * Nat.gcd a b = 48 := by
  sorry

end product_lcm_gcd_l171_171167


namespace inflation_two_years_correct_real_rate_of_return_correct_l171_171704

-- Define the calculation for inflation over two years
def inflation_two_years (r : ℝ) : ℝ :=
  ((1 + r)^2 - 1) * 100

-- Define the calculation for the real rate of return
def real_rate_of_return (r : ℝ) (infl_rate : ℝ) : ℝ :=
  ((1 + r * r) / (1 + infl_rate / 100) - 1) * 100

-- Prove the inflation over two years is 3.0225%
theorem inflation_two_years_correct :
  inflation_two_years 0.015 = 3.0225 :=
by
  sorry

-- Prove the real yield of the bank deposit is 11.13%
theorem real_rate_of_return_correct :
  real_rate_of_return 0.07 3.0225 = 11.13 :=
by
  sorry

end inflation_two_years_correct_real_rate_of_return_correct_l171_171704


namespace emerie_dimes_count_l171_171273

variables (zain_coins emerie_coins num_quarters num_nickels : ℕ)
variable (emerie_dimes : ℕ)

-- Conditions as per part a)
axiom zain_has_more_coins : ∀ (e z : ℕ), z = e + 10
axiom total_zain_coins : zain_coins = 48
axiom emerie_coins_from_quarters_and_nickels : num_quarters = 6 ∧ num_nickels = 5
axiom emerie_known_coins : ∀ q n : ℕ, emerie_coins = q + n + emerie_dimes

-- The statement to prove
theorem emerie_dimes_count : emerie_coins = 38 → emerie_dimes = 27 := 
by 
  sorry

end emerie_dimes_count_l171_171273


namespace stuart_segments_to_start_point_l171_171356

-- Definitions of given conditions
def concentric_circles {C : Type} (large small : Set C) (center : C) : Prop :=
  ∀ (x y : C), x ∈ large → y ∈ large → x ≠ y → (x = center ∨ y = center)

def tangent_to_small_circle {C : Type} (chord : Set C) (small : Set C) : Prop :=
  ∀ (x y : C), x ∈ chord → y ∈ chord → x ≠ y → (∀ z ∈ small, x ≠ z ∧ y ≠ z)

def measure_angle (ABC : Type) (θ : ℝ) : Prop :=
  θ = 60

-- The theorem to solve the problem
theorem stuart_segments_to_start_point 
    (C : Type)
    {large small : Set C} 
    {center : C} 
    {chords : List (Set C)}
    (h_concentric : concentric_circles large small center)
    (h_tangent : ∀ chord ∈ chords, tangent_to_small_circle chord small)
    (h_angle : ∀ ABC ∈ chords, measure_angle ABC 60)
    : ∃ n : ℕ, n = 3 := 
  sorry

end stuart_segments_to_start_point_l171_171356


namespace median_of_set_l171_171837

open List

def is_median (l : List ℝ) (m : ℝ) : Prop :=
  l.length % 2 = 1 ∧ (sorted l) ∧ (l.nth (l.length / 2)).iget = m

theorem median_of_set (s : List ℝ) (h_len : s.length = 100)
  (h1 : ∃ n, is_median (s.erase n) 78)
  (h2 : ∃ n, is_median (s.erase n) 66) :
  is_median s 72 :=
sorry

end median_of_set_l171_171837


namespace forty_percent_of_number_l171_171664

theorem forty_percent_of_number (N : ℚ)
  (h : (1 / 4) * (1 / 3) * (2 / 5) * N = 15) : 0.40 * N = 180 := 
by 
  sorry

end forty_percent_of_number_l171_171664


namespace sin_90_eq_one_l171_171122

noncomputable theory
open Real

/--
The sine of an angle in the unit circle is the y-coordinate of the point at that angle from the positive x-axis.
Rotating the point (1,0) by 90 degrees counterclockwise about the origin results in the point (0,1).
Prove that \(\sin 90^\circ = 1\).
-/
theorem sin_90_eq_one : sin (90 * (real.pi / 180)) = 1 :=
by
  -- Definitions and conditions for the unit circle and sine function
  let angle := 90 * (real.pi / 180)
  have h1 : (cos angle, sin angle) = (0, 1),
  { sorry },
  -- Desired conclusion
  exact h1.2

end sin_90_eq_one_l171_171122


namespace paintable_area_correct_l171_171024

-- Defining lengths
def bedroom_length : ℕ := 15
def bedroom_width : ℕ := 11
def bedroom_height : ℕ := 9

-- Defining the number of bedrooms
def num_bedrooms : ℕ := 4

-- Defining the total area not to be painted per bedroom
def area_not_painted_per_bedroom : ℕ := 80

-- The total wall area calculation
def total_wall_area_per_bedroom : ℕ :=
  2 * (bedroom_length * bedroom_height) + 2 * (bedroom_width * bedroom_height)

-- The paintable wall area per bedroom calculation
def paintable_area_per_bedroom : ℕ :=
  total_wall_area_per_bedroom - area_not_painted_per_bedroom

-- The total paintable area across all bedrooms calculation
def total_paintable_area : ℕ :=
  paintable_area_per_bedroom * num_bedrooms

-- The theorem statement
theorem paintable_area_correct : total_paintable_area = 1552 := by
  sorry -- Proof is omitted

end paintable_area_correct_l171_171024


namespace unique_value_of_n_l171_171842

theorem unique_value_of_n
  (n t : ℕ) (h1 : t ≠ 0)
  (h2 : 15 * t + (n - 20) * t / 3 = (n * t) / 2) :
  n = 50 :=
by sorry

end unique_value_of_n_l171_171842


namespace complement_intersection_empty_l171_171305

open Set

-- Given definitions and conditions
def U : Set ℕ := {1, 2, 3}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 3}

-- Complement operation with respect to U
def C_U (X : Set ℕ) : Set ℕ := U \ X

-- The proof statement to be shown
theorem complement_intersection_empty :
  (C_U A ∩ C_U B) = ∅ := by sorry

end complement_intersection_empty_l171_171305


namespace distance_covered_by_wheel_l171_171431

noncomputable def pi_num : ℝ := 3.14159

noncomputable def wheel_diameter : ℝ := 14

noncomputable def number_of_revolutions : ℝ := 33.03002729754322

noncomputable def circumference : ℝ := pi_num * wheel_diameter

noncomputable def calculated_distance : ℝ := circumference * number_of_revolutions

theorem distance_covered_by_wheel : 
  calculated_distance = 1452.996 :=
sorry

end distance_covered_by_wheel_l171_171431


namespace greatest_b_not_in_range_l171_171700

theorem greatest_b_not_in_range (b : ℤ) : ∀ x : ℝ, ¬ (x^2 + (b : ℝ) * x + 20 = -9) ↔ b ≤ 10 :=
by
  sorry

end greatest_b_not_in_range_l171_171700


namespace StockPriceAdjustment_l171_171676

theorem StockPriceAdjustment (P₀ P₁ P₂ P₃ P₄ : ℝ) (january_increase february_decrease march_increase : ℝ) :
  P₀ = 150 →
  january_increase = 0.10 →
  february_decrease = 0.15 →
  march_increase = 0.30 →
  P₁ = P₀ * (1 + january_increase) →
  P₂ = P₁ * (1 - february_decrease) →
  P₃ = P₂ * (1 + march_increase) →
  142.5 <= P₃ * (1 - 0.17) ∧ P₃ * (1 - 0.17) <= 157.5 :=
by
  intros hP₀ hJanuaryIncrease hFebruaryDecrease hMarchIncrease hP₁ hP₂ hP₃
  sorry

end StockPriceAdjustment_l171_171676


namespace no_real_solution_intersection_l171_171753

theorem no_real_solution_intersection :
  ¬ ∃ x y : ℝ, (y = 8 / (x^3 + 4 * x + 3)) ∧ (x + y = 5) :=
by
  sorry

end no_real_solution_intersection_l171_171753


namespace value_of_S_l171_171510

-- Defining the condition as an assumption
def one_third_one_eighth_S (S : ℝ) : Prop :=
  (1 / 3) * (1 / 8) * S = (1 / 4) * (1 / 6) * 120

-- The statement we need to prove
theorem value_of_S (S : ℝ) (h : one_third_one_eighth_S S) : S = 120 :=
by
  sorry

end value_of_S_l171_171510


namespace daria_still_owes_l171_171953

-- Definitions of the given conditions
def saved_amount : ℝ := 500
def couch_cost : ℝ := 750
def table_cost : ℝ := 100
def lamp_cost : ℝ := 50

-- Calculation of total cost of the furniture
def total_cost : ℝ := couch_cost + table_cost + lamp_cost

-- Calculation of the remaining amount owed
def remaining_owed : ℝ := total_cost - saved_amount

-- Proof statement that Daria still owes $400 before interest
theorem daria_still_owes : remaining_owed = 400 := by
  -- Skipping the proof
  sorry

end daria_still_owes_l171_171953


namespace max_loaves_given_l171_171522

variables {a1 d : ℕ}

-- Mathematical statement: The conditions given in the problem
def arith_sequence_correct (a1 d : ℕ) : Prop :=
  (5 * a1 + 10 * d = 60) ∧ (2 * a1 + 7 * d = 3 * a1 + 3 * d)

-- Lean theorem statement
theorem max_loaves_given (a1 d : ℕ) (h : arith_sequence_correct a1 d) : a1 + 4 * d = 16 :=
sorry

end max_loaves_given_l171_171522


namespace little_ming_problem_solution_l171_171870

theorem little_ming_problem_solution :
  let number_of_ways := ∑ i in (Finset.range 10), Nat.choose 9 i
  number_of_ways = 512 :=
by
  -- The proof steps are not required according to the problem statement
  sorry

end little_ming_problem_solution_l171_171870


namespace common_root_equation_l171_171173

theorem common_root_equation {m : ℝ} (x : ℝ) (h1 : m * x - 1000 = 1001) (h2 : 1001 * x = m - 1000 * x) : m = 2001 ∨ m = -2001 :=
by
  -- Skipping the proof details
  sorry

end common_root_equation_l171_171173


namespace snakes_in_cage_l171_171443

theorem snakes_in_cage (snakes_hiding : Nat) (snakes_not_hiding : Nat) (total_snakes : Nat) 
  (h : snakes_hiding = 64) (nh : snakes_not_hiding = 31) : 
  total_snakes = snakes_hiding + snakes_not_hiding := by
  sorry

end snakes_in_cage_l171_171443


namespace smallest_solution_of_quartic_equation_l171_171079

theorem smallest_solution_of_quartic_equation :
  ∃ x : ℝ, (x^4 - 50 * x^2 + 625 = 0) ∧ ∀ y : ℝ, y^4 - 50 * y^2 + 625 = 0 → x ≤ y := sorry

end smallest_solution_of_quartic_equation_l171_171079


namespace sum_of_possible_k_l171_171856

theorem sum_of_possible_k (a b c k : ℂ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a)
  (h4 : a / (2 - b) = k) (h5 : b / (3 - c) = k) (h6 : c / (4 - a) = k) :
  k = 1 ∨ k = -1 ∨ k = -2 → k = 1 + (-1) + (-2) :=
by
  sorry

end sum_of_possible_k_l171_171856


namespace sqrt_square_multiply_l171_171291

theorem sqrt_square_multiply (a : ℝ) (h : a = 49284) :
  (Real.sqrt a)^2 * 3 = 147852 :=
by
  sorry

end sqrt_square_multiply_l171_171291


namespace sum_edge_lengths_rectangular_solid_vol_surface_area_geom_prog_l171_171928

theorem sum_edge_lengths_rectangular_solid_vol_surface_area_geom_prog
  (a r : ℝ)
  (volume_cond : a^3 * r^3 = 288)
  (surface_area_cond : 2 * (a^2 * r^4 + a^2 * r^2 + a^2 * r) = 288)
  (geom_prog : True) :
  4 * (a * r^2 + a * r + a) = 92 := 
sorry

end sum_edge_lengths_rectangular_solid_vol_surface_area_geom_prog_l171_171928


namespace line_through_parabola_intersects_vertex_l171_171982

theorem line_through_parabola_intersects_vertex (y x k : ℝ) :
  (y = 6 * x ∨ 6 * x - 5 * y - 24 = 0) ∧ 
  (∃ P Q : ℝ × ℝ, (P.1)^2 = 4 * P.2 ∧ (Q.1)^2 = 4 * Q.2 ∧ 
   (P = (0, 0) ∨ Q = (0, 0)) ∧ 
   (y = 6 * x ∨ 6 * x - 5 * y - 24 = 0)) := sorry

end line_through_parabola_intersects_vertex_l171_171982


namespace first_prize_ticket_numbers_l171_171943

theorem first_prize_ticket_numbers :
  {n : ℕ | n < 10000 ∧ (n % 1000 = 418)} = {418, 1418, 2418, 3418, 4418, 5418, 6418, 7418, 8418, 9418} :=
by
  sorry

end first_prize_ticket_numbers_l171_171943


namespace shares_total_amount_l171_171486

theorem shares_total_amount (Nina_portion : ℕ) (m n o : ℕ) (m_ratio n_ratio o_ratio : ℕ)
  (h_ratio : m_ratio = 2 ∧ n_ratio = 3 ∧ o_ratio = 9)
  (h_Nina : Nina_portion = 60)
  (hk := Nina_portion / n_ratio)
  (h_shares : m = m_ratio * hk ∧ n = n_ratio * hk ∧ o = o_ratio * hk) :
  m + n + o = 280 :=
by 
  sorry

end shares_total_amount_l171_171486


namespace three_letter_words_with_A_at_least_once_l171_171988

theorem three_letter_words_with_A_at_least_once :
  let total_words := 4^3
  let words_without_A := 3^3
  total_words - words_without_A = 37 :=
by
  let total_words := 4^3
  let words_without_A := 3^3
  sorry

end three_letter_words_with_A_at_least_once_l171_171988


namespace algebraic_expression_value_l171_171475

theorem algebraic_expression_value (a b : ℝ) 
  (h : |a + 2| + (b - 1)^2 = 0) : (a + b) ^ 2005 = -1 :=
by
  sorry

end algebraic_expression_value_l171_171475


namespace probability_of_one_standard_one_special_l171_171293

noncomputable def probability_exactly_one_standard_one_one_special_four : ℚ :=
  let six_sided_dice := {1, 2, 3, 4, 5, 6}
  let even_sided_dice := {2, 4, 6}
  let prob_standard_shows_1 := (1 / 6 : ℚ)
  let prob_standard_not_show_1 := (5 / 6 : ℚ)
  let prob_special_shows_4 := (1 / 3 : ℚ)
  let prob_special_not_show_4 := (2 / 3 : ℚ)
  let comb_five_choose_one := nat.choose 5 1
  comb_five_choose_one * prob_standard_shows_1 * prob_standard_not_show_1^4 *
  comb_five_choose_one * prob_special_shows_4 * prob_special_not_show_4^4

theorem probability_of_one_standard_one_special :
  probability_exactly_one_standard_one_one_special_four ≈ 0.132 := by
  sorry

end probability_of_one_standard_one_special_l171_171293


namespace sum_geometric_sequence_l171_171771

theorem sum_geometric_sequence 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n, a (n + 1) = a n * a 1)
  (h_a1 : a 1 = 1)
  (h_arithmetic : 4 * a 2 + a 4 = 2 * a 3) : 
  a 2 + a 3 + a 4 = 14 :=
sorry

end sum_geometric_sequence_l171_171771


namespace integer_values_of_a_l171_171751

variable (a b c x : ℤ)

theorem integer_values_of_a (h : (x - a) * (x - 12) + 4 = (x + b) * (x + c)) : a = 7 ∨ a = 17 := by
  sorry

end integer_values_of_a_l171_171751


namespace expand_binomials_l171_171162

variable {x y : ℝ}

theorem expand_binomials (x y : ℝ) : 
  (x + 5) * (3 * y + 15) = 3 * x * y + 15 * x + 15 * y + 75 := 
by
  sorry

end expand_binomials_l171_171162


namespace smallest_possible_N_l171_171796

theorem smallest_possible_N (N : ℕ) (h1 : N > 70) (h2 : 70 ∣ 21 * N) : N = 80 :=
sorry

end smallest_possible_N_l171_171796


namespace median_of_100_numbers_l171_171841

theorem median_of_100_numbers (x : Fin 100 → ℝ)
  (h1 : ∀ i j, i ≠ j → x i = 78 → x j = 66 → i = 51 ∧ j = 50 ∨ i = 50 ∧ j = 51)
  (h2 : ∀ i, i ≠ 51 → x 51 = 78)
  (h3 : ∀ i, i ≠ 50 → x 50 = 66) :
  (x 50 + x 51) / 2 = 72 :=
by sorry

end median_of_100_numbers_l171_171841


namespace total_marbles_l171_171859

def mary_marbles := 9
def joan_marbles := 3
def john_marbles := 7

theorem total_marbles :
  mary_marbles + joan_marbles + john_marbles = 19 :=
by
  sorry

end total_marbles_l171_171859


namespace line_containing_chord_l171_171000

variable {x y x₁ y₁ x₂ y₂ : ℝ}

def ellipse_eq (x y : ℝ) : Prop := (x^2 / 9 + y^2 / 4 = 1)

def midpoint_condition (x₁ x₂ : ℝ) (y₁ y₂ : ℝ) : Prop := 
  (x₁ + x₂ = 2) ∧ (y₁ + y₂ = 2)

theorem line_containing_chord (h₁ : ellipse_eq x₁ y₁) 
                               (h₂ : ellipse_eq x₂ y₂) 
                               (hmp : midpoint_condition x₁ x₂ y₁ y₂)
    : 4 * 1 + 9 * 1 - 13 = 0 := 
sorry

end line_containing_chord_l171_171000


namespace binom_13_11_eq_78_l171_171119

theorem binom_13_11_eq_78 : Nat.choose 13 11 = 78 := by
  sorry

end binom_13_11_eq_78_l171_171119


namespace octagon_area_correct_l171_171936

-- Define the radius of the circle
def radius : ℝ := 3
-- Define the expected area of the regular octagon
def expected_area : ℝ := 54 * (2 - Real.sqrt 2) * Real.sqrt 3

-- The Lean 4 theorem statement
theorem octagon_area_correct : 
  ∀ (R : ℝ) (hR : R = radius), 
  ∃ (A : ℝ), A = expected_area := 
by
  intro R hR
  use expected_area
  sorry

end octagon_area_correct_l171_171936


namespace new_apples_grew_l171_171039

-- The number of apples originally on the tree.
def original_apples : ℕ := 11

-- The number of apples picked by Rachel.
def picked_apples : ℕ := 7

-- The number of apples currently on the tree.
def current_apples : ℕ := 6

-- The number of apples left on the tree after picking.
def remaining_apples : ℕ := original_apples - picked_apples

-- The number of new apples that grew on the tree.
def new_apples : ℕ := current_apples - remaining_apples

-- The theorem we need to prove.
theorem new_apples_grew :
  new_apples = 2 := by
    sorry

end new_apples_grew_l171_171039


namespace value_of_m_l171_171195

theorem value_of_m (a m : ℝ) (h : a > 0) (hm : m ≠ 0) :
  (P : ℝ × ℝ) (P = (m, 3))
  (H : ∀ x : ℝ, -a * x^2 + 2 * a * x + 3 = 3 → x = 0 ∨ x = 2) :
  m = 2 :=
by
  sorry

end value_of_m_l171_171195


namespace set_intersection_example_l171_171789

theorem set_intersection_example :
  let A := { y | ∃ x, y = Real.log x / Real.log 2 ∧ x ≥ 3 }
  let B := { x | x^2 - 4 * x + 3 = 0 }
  A ∩ B = {3} :=
by
  sorry

end set_intersection_example_l171_171789


namespace prove_f_f_x_eq_4_prove_f_f_x_eq_5_l171_171176

variable (f : ℝ → ℝ)

-- Conditions
axiom f_of_4 : f (-2) = 4 ∧ f 2 = 4 ∧ f 6 = 4
axiom f_of_5 : f (-4) = 5 ∧ f 4 = 5

-- Intermediate Values
axiom f_inv_of_4 : f 0 = -2 ∧ f (-1) = 2 ∧ f 3 = 6
axiom f_inv_of_5 : f 2 = 4

theorem prove_f_f_x_eq_4 :
  {x : ℝ | f (f x) = 4} = {0, -1, 3} :=
by
  sorry

theorem prove_f_f_x_eq_5 :
  {x : ℝ | f (f x) = 5} = {2} :=
by
  sorry

end prove_f_f_x_eq_4_prove_f_f_x_eq_5_l171_171176


namespace triangle_third_side_length_l171_171780

theorem triangle_third_side_length (a b c : ℝ) (h₁ : a = 5) (h₂ : b = 8) :
  (a + b > c) → (a + c > b) → (b + c > a) → c = 6 :=
by
  intros h₃ h₄ h₅
  rw [h₁, h₂] at *
  -- Simplified inequalities from the solution step
  have h₆ : 5 + 8 > c := by rw [h₁, h₂]; exact h₃
  have h₇ : 8 + c > 5 := by rw h₂; exact h₄
  have h₈ : 5 + c > 8 := by rw h₁; exact h₅
  sorry

end triangle_third_side_length_l171_171780


namespace part1_part2_part3_l171_171831

variables (a b c : ℤ)
-- Condition: For all integer values of x, (ax^2 + bx + c) is a square number 
def quadratic_is_square_for_any_x (a b c : ℤ) : Prop :=
  ∀ x : ℤ, ∃ k : ℤ, a * x^2 + b * x + c = k^2

-- Question (1): Prove that 2a, 2b, c are all integers
theorem part1 (h : quadratic_is_square_for_any_x a b c) : 
  ∃ m n : ℤ, 2 * a = m ∧ 2 * b = n ∧ ∃ k₁ : ℤ, c = k₁ :=
sorry

-- Question (2): Prove that a, b, c are all integers, and c is a square number
theorem part2 (h : quadratic_is_square_for_any_x a b c) : 
  ∃ k₁ k₂ m n : ℤ, a = k₁ ∧ b = k₂ ∧ c = m^2 :=
sorry

-- Question (3): Prove that if (2) holds, it does not necessarily mean that 
-- for all integer values of x, (ax^2 + bx + c) is always a square number.
theorem part3 (a b c : ℤ) (h : ∃ k₁ k₂ m n : ℤ, a = k₁ ∧ b = k₂ ∧ c = m^2) : 
  ¬ quadratic_is_square_for_any_x a b c :=
sorry

end part1_part2_part3_l171_171831


namespace sum_on_simple_interest_is_1400_l171_171690

noncomputable def sum_placed_on_simple_interest : ℝ :=
  let P_c := 4000
  let r := 0.10
  let n := 1
  let t_c := 2
  let t_s := 3
  let A := P_c * (1 + r / n)^(n * t_c)
  let CI := A - P_c
  let SI := CI / 2
  100 * SI / (r * t_s)

theorem sum_on_simple_interest_is_1400 : sum_placed_on_simple_interest = 1400 := by
  sorry

end sum_on_simple_interest_is_1400_l171_171690


namespace length_of_ae_l171_171714

-- Definition of points and lengths between them
variables (a b c d e : Type)
variables (bc cd de ab ac : ℝ)

-- Given conditions
axiom H1 : bc = 3 * cd
axiom H2 : de = 8
axiom H3 : ab = 5
axiom H4 : ac = 11
axiom H5 : bc = ac - ab
axiom H6 : cd = bc / 3

-- Theorem to prove
theorem length_of_ae : ∀ ab bc cd de : ℝ, ae = ab + bc + cd + de := by
  sorry

end length_of_ae_l171_171714


namespace opposite_numbers_l171_171553

theorem opposite_numbers
  (odot otimes : ℝ)
  (x y : ℝ)
  (h1 : 6 * x + odot * y = 3)
  (h2 : 2 * x + otimes * y = -1)
  (h_add : 6 * x + odot * y + (2 * x + otimes * y) = 2) :
  odot + otimes = 0 := by
  sorry

end opposite_numbers_l171_171553


namespace final_remaining_money_l171_171636

-- Define conditions as given in the problem
def monthly_income : ℕ := 2500
def rent : ℕ := 700
def car_payment : ℕ := 300
def utilities : ℕ := car_payment / 2
def groceries : ℕ := 50
def expenses_total : ℕ := rent + car_payment + utilities + groceries
def remaining_money : ℕ := monthly_income - expenses_total
def retirement_contribution : ℕ := remaining_money / 2

-- State the theorem to be proven
theorem final_remaining_money : (remaining_money - retirement_contribution) = 650 := by
  sorry

end final_remaining_money_l171_171636


namespace smallest_solution_x4_minus_50x2_plus_625_eq_0_l171_171078

theorem smallest_solution_x4_minus_50x2_plus_625_eq_0 : ∃ x : ℝ, x^4 - 50 * x^2 + 625 = 0 ∧ ∀ y : ℝ, y^4 - 50 * y^2 + 625 = 0 → x ≤ y := 
sorry

end smallest_solution_x4_minus_50x2_plus_625_eq_0_l171_171078


namespace center_and_radius_of_circle_l171_171874

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2 * x + 6 * y + 6 = 0

-- State the theorem
theorem center_and_radius_of_circle :
  (∃ x₀ y₀ r, (∀ x y, circle_eq x y ↔ (x - x₀)^2 + (y - y₀)^2 = r^2) ∧
  x₀ = 1 ∧ y₀ = -3 ∧ r = 2) :=
by
  -- Proof is omitted
  sorry

end center_and_radius_of_circle_l171_171874


namespace num_dogs_with_spots_l171_171337

variable (D P : ℕ)

theorem num_dogs_with_spots (h1 : D / 2 = D / 2) (h2 : D / 5 = P) : (5 * P) / 2 = D / 2 := 
by
  have h3 : 5 * P = D := by
    sorry
  have h4 : (5 * P) / 2 = D / 2 := by
    rw [h3]
  exact h4

end num_dogs_with_spots_l171_171337


namespace path_shorter_factor_l171_171663

-- Declare variables
variables (x y z : ℝ)

-- Define conditions as hypotheses
def condition1 := x = 3 * (y + z)
def condition2 := 4 * y = z + x

-- State the proof statement
theorem path_shorter_factor (condition1 : x = 3 * (y + z)) (condition2 : 4 * y = z + x) :
  (4 * y) / z = 19 :=
sorry

end path_shorter_factor_l171_171663


namespace satisfies_equation_l171_171755

theorem satisfies_equation (a b c : ℤ) (h₁ : a = b) (h₂ : b = c + 1) :
  a * (a - b) + b * (b - c) + c * (c - a) = 3 := 
by 
  sorry

end satisfies_equation_l171_171755


namespace division_addition_l171_171290

theorem division_addition : (-300) / (-75) + 10 = 14 := by
  sorry

end division_addition_l171_171290


namespace abs_inequality_solution_set_l171_171763

-- Define the main problem as a Lean theorem statement
theorem abs_inequality_solution_set (x : ℝ) : 
  (|x - 5| + |x + 3| ≥ 10 ↔ (x ≤ -4 ∨ x ≥ 6)) :=
by {
  sorry
}

end abs_inequality_solution_set_l171_171763


namespace twenty_four_point_solution_l171_171017

theorem twenty_four_point_solution : (5 - (1 / 5)) * 5 = 24 := 
by 
  sorry

end twenty_four_point_solution_l171_171017


namespace problem_1_problem_2_problem_3_problem_4i_problem_4ii_problem_4iii_problem_4iv_l171_171095

-- Problem 1:
theorem problem_1 (a : ℕ → ℝ) (h₀ : a 1 = 1) (h₁ : ∀ n > 1, a n = 1 + 1 / a (n - 1)) :
  a 3 = 3 / 2 := sorry

-- Problem 2:
theorem problem_2 (a q : ℕ → ℝ) (S : ℕ → ℝ) (h₀ : q 1 = 1 / 2) (h₁ : ∀ n > 1, q n = q 1)
  (h₂ : ∀ n, S n = (a 1 * (1 - q 1 ^ n)) / (1 - q 1)) :
  (S 4 / a 4) = 15 := sorry

-- Problem 3:
theorem problem_3 (a b c S : ℝ) (h₀ : 2 * S = (a + b) ^ 2 - c ^ 2) :
  Real.tan (Real.atan2 a b c S) = -4 / 3 := sorry

-- Problem 4i:
theorem problem_4i (a : ℕ → ℝ) (h₀ : a 1 = 2) (h₁ : ∀ n ∈ ℕ*, a (n + 1) = 2 * a n - 1) :
  a 11 = 1025 := sorry

-- Problem 4ii:
theorem problem_4ii (a b : ℕ → ℝ) (h₀ : ∀ n ∈ ℕ*, a (n + 1) = 1 - 1 / (4 * a n))
  (h₁ : b n = 2 / (2 * a n - 1)) :
  ¬ (∀ n ∈ ℕ*, b (n + 2) - b (n + 1) = b (n + 1) - b n) := sorry

-- Problem 4iii:
theorem problem_4iii (a : ℕ → ℝ) (S : ℕ → ℝ) (h₀ : S n = n ^ 2 + 2 * n) :
  (∀ n ∈ ℕ*, 1 / a (n + 1) + 1 / a (n + 2) + ... + 1 / a (2 * n) ≥ 1 / 5) := sorry

-- Problem 4iv:
theorem problem_4iv (a : ℤ → ℕ → ℝ) (h₀ : (∀ n ∈ ℕ*, a (1 + 3 * n + 5 * n (n + 1) / 2 + ... + (2 * n - 1) * a n) = 2 ^ (n + 1)) :
  ¬ (∀ n ∈ ℕ*, a n = 2 ^ n / (2 * n - 1)) := sorry

end problem_1_problem_2_problem_3_problem_4i_problem_4ii_problem_4iii_problem_4iv_l171_171095


namespace convoy_length_after_checkpoint_l171_171415

theorem convoy_length_after_checkpoint
  (L_initial : ℝ) (v_initial : ℝ) (v_final : ℝ) (t_fin : ℝ)
  (H_initial_len : L_initial = 300)
  (H_initial_speed : v_initial = 60)
  (H_final_speed : v_final = 40)
  (H_time_last_car : t_fin = (300 / 1000) / 60) :
  L_initial * v_final / v_initial - (v_final * ((300 / 1000) / 60)) = 200 :=
by
  sorry

end convoy_length_after_checkpoint_l171_171415


namespace range_of_a_l171_171008

variable {x a : ℝ}

theorem range_of_a (h1 : x < 0) (h2 : 2 ^ x - a = 1 / (x - 1)) : 0 < a ∧ a < 2 :=
sorry

end range_of_a_l171_171008


namespace circle_inscribed_angles_l171_171680

theorem circle_inscribed_angles (O : Type) (circle : Set O) (A B C D E F G H I J K L : O) 
  (P : ℕ) (n : ℕ) (x_deg_sum y_deg_sum : ℝ)  
  (h1 : n = 12) 
  (h2 : x_deg_sum = 45) 
  (h3 : y_deg_sum = 75) :
  x_deg_sum + y_deg_sum = 120 :=
by
  /- Proof steps are not required -/
  apply sorry

end circle_inscribed_angles_l171_171680


namespace linear_eq_solution_l171_171640

theorem linear_eq_solution (m x : ℝ) (h : |m| = 1) (h1: 1 - m ≠ 0):
  x = -(1/2) :=
sorry

end linear_eq_solution_l171_171640


namespace parallel_line_through_point_l171_171054

-- Problem: Prove the equation of the line that passes through the point (1, 1)
-- and is parallel to the line 2x - y + 1 = 0 is 2x - y - 1 = 0.

theorem parallel_line_through_point (x y : ℝ) (c : ℝ) :
  (2*x - y + 1 = 0) → (x = 1) → (y = 1) → (2*1 - 1 + c = 0) → c = -1 → (2*x - y - 1 = 0) :=
by
  sorry

end parallel_line_through_point_l171_171054


namespace smallest_possible_N_l171_171794

theorem smallest_possible_N (N : ℕ) (h1 : N > 70) (h2 : 70 ∣ 21 * N) : N = 80 :=
sorry

end smallest_possible_N_l171_171794


namespace negation_of_proposition_l171_171879

theorem negation_of_proposition : (¬ ∃ x : ℝ, x > 0 ∧ x^2 - 5 * x + 6 > 0) ↔ ∀ x : ℝ, x > 0 → x^2 - 5 * x + 6 ≤ 0 := by
  sorry

end negation_of_proposition_l171_171879


namespace cloth_sold_l171_171582

theorem cloth_sold (C S M : ℚ) (P : ℚ) (hP : P = 1 / 3) (hG : 10 * S = (1 / 3) * (M * C)) (hS : S = (4 / 3) * C) : M = 40 := by
  sorry

end cloth_sold_l171_171582


namespace x_intercept_perpendicular_l171_171547

theorem x_intercept_perpendicular (k m x y : ℝ) (h1 : 4 * x - 3 * y = 12) (h2 : y = -3/4 * x + 3) :
  x = 4 :=
by
  sorry

end x_intercept_perpendicular_l171_171547


namespace number_of_elements_in_union_l171_171180

open Set

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {1, 2, 3}

theorem number_of_elements_in_union : ncard (A ∪ B) = 4 :=
by
  sorry

end number_of_elements_in_union_l171_171180


namespace exists_finite_set_with_subset_relation_l171_171409

-- Definition of an ordered set (E, ≤)
variable {E : Type} [LE E]

theorem exists_finite_set_with_subset_relation (E : Type) [LE E] :
  ∃ (F : Set (Set E)) (X : E → Set E), 
  (∀ (e1 e2 : E), e1 ≤ e2 ↔ X e2 ⊆ X e1) :=
by
  -- The proof is initially skipped, as per instructions
  sorry

end exists_finite_set_with_subset_relation_l171_171409


namespace plane_distance_l171_171514

variable (a b c p : ℝ)

def plane_intercept := (a ≠ 0) ∧ (b ≠ 0) ∧ (c ≠ 0) ∧
  (p = 1 / (Real.sqrt ((1 / a^2) + (1 / b^2) + (1 / c^2))))

theorem plane_distance
  (h : plane_intercept a b c p) :
  1 / a^2 + 1 / b^2 + 1 / c^2 = 1 / p^2 := 
sorry

end plane_distance_l171_171514


namespace min_value_fraction_l171_171312

theorem min_value_fraction (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x * y = x + 2 * y + 6) : 
  (∃ (z : ℝ), z = 1 / x + 1 / (2 * y) ∧ z ≥ 1 / 3) :=
sorry

end min_value_fraction_l171_171312


namespace weigh_1_to_10_kg_l171_171268

theorem weigh_1_to_10_kg (n : ℕ) : 1 ≤ n ∧ n ≤ 10 →
  ∃ (a b c : ℤ), 
    (abs a ≤ 1 ∧ abs b ≤ 1 ∧ abs c ≤ 1 ∧
    (n = a * 3 + b * 4 + c * 9)) :=
by sorry

end weigh_1_to_10_kg_l171_171268


namespace area_of_regular_octagon_in_circle_l171_171930

/-- Define a regular octagon and inscribe it in a circle of radius 3 units, 
    finding the exact area in square units in simplest radical form -/
theorem area_of_regular_octagon_in_circle 
(radius : ℝ) (h_radius : radius = 3) : 
  ∃ (a : ℝ), a = (8 * (1 / 2 * (2 * radius * real.sin (real.pi / 8))^2 * real.sin (real.pi / 4))) :=
by sorry

end area_of_regular_octagon_in_circle_l171_171930


namespace sqrt_div_add_l171_171616

theorem sqrt_div_add :
  let sqrt_0_81 := 0.9
  let sqrt_1_44 := 1.2
  let sqrt_0_49 := 0.7
  (Real.sqrt 1.1 / sqrt_0_81) + (sqrt_1_44 / sqrt_0_49) = 2.8793 :=
by
  -- Prove equality using the given conditions
  sorry

end sqrt_div_add_l171_171616


namespace arlo_stationery_count_l171_171688

theorem arlo_stationery_count (books pens : ℕ) (ratio_books_pens : ℕ × ℕ) (total_books : ℕ)
  (h_ratio : ratio_books_pens = (7, 3)) (h_books : total_books = 280) :
  books + pens = 400 :=
by
  sorry

end arlo_stationery_count_l171_171688


namespace harold_final_remaining_money_l171_171635

def harold_monthly_income : ℝ := 2500.00
def rent : ℝ := 700.00
def car_payment : ℝ := 300.00
def utilities_cost (car_payment : ℝ) : ℝ := car_payment / 2
def groceries : ℝ := 50.00
def total_expenses (rent car_payment utilities_cost groceries : ℝ) : ℝ :=
  rent + car_payment + utilities_cost + groceries
def remaining_money (income total_expenses : ℝ) : ℝ := income - total_expenses
def retirement_savings (remaining_money : ℝ) : ℝ := remaining_money / 2
def final_remaining (remaining_money retirement_savings : ℝ) : ℝ :=
  remaining_money - retirement_savings

theorem harold_final_remaining_money :
  final_remaining (remaining_money harold_monthly_income (total_expenses rent car_payment (utilities_cost car_payment) groceries))
         (retirement_savings (remaining_money harold_monthly_income (total_expenses rent car_payment (utilities_cost car_payment) groceries))) = 650.00 :=
by
  sorry

end harold_final_remaining_money_l171_171635


namespace bob_bakes_pie_in_6_minutes_l171_171944

theorem bob_bakes_pie_in_6_minutes (x : ℕ) (h_alice : 60 / 5 = 12)
  (h_condition : 12 - 2 = 60 / x) : x = 6 :=
sorry

end bob_bakes_pie_in_6_minutes_l171_171944


namespace triangle_third_side_length_l171_171782

theorem triangle_third_side_length (a b : ℝ) (x : ℝ) (h₁ : a = 5) (h₂ : b = 8) (hx : x ∈ {2, 3, 6, 13}) :
  3 < x ∧ x < 13 → x = 6 :=
by sorry

end triangle_third_side_length_l171_171782


namespace evaluate_expression_l171_171454

theorem evaluate_expression (x y z : ℝ) : 
  (x + (y + z)) - ((-x + y) + z) = 2 * x := 
by
  sorry

end evaluate_expression_l171_171454


namespace geometric_sequence_sum_condition_l171_171021

theorem geometric_sequence_sum_condition
  (a_1 r : ℝ) 
  (S₄ : ℝ := a_1 * (1 + r + r^2 + r^3)) 
  (S₈ : ℝ := S₄ + a_1 * (r^4 + r^5 + r^6 + r^7)) 
  (h₁ : S₄ = 1) 
  (h₂ : S₈ = 3) :
  a_1 * r^16 * (1 + r + r^2 + r^3) = 8 := 
sorry

end geometric_sequence_sum_condition_l171_171021


namespace smallest_N_l171_171811

theorem smallest_N (N : ℕ) (hN : N > 70) (hdv : 70 ∣ 21 * N) : N = 80 :=
sorry

end smallest_N_l171_171811


namespace smallest_N_l171_171818

theorem smallest_N (N : ℕ) (hN : N > 70) (hDiv : 70 ∣ 21 * N) : N = 80 :=
sorry

end smallest_N_l171_171818


namespace find_a_and_b_l171_171632

-- Define the line equation
def line (k : ℝ) (x : ℝ) : ℝ := k * x + 1

-- Define the curve equation
def curve (a b x : ℝ) : ℝ := x^3 + a * x + b

-- Define the derivative of the curve
def curve_derivative (a x : ℝ) : ℝ := 3 * x^2 + a

-- Main theorem to prove a = -1 and b = 3 given tangency conditions
theorem find_a_and_b 
  (k : ℝ) (a b : ℝ) (tangent_point : ℝ × ℝ)
  (h_tangent : tangent_point = (1, 3))
  (h_line : line k tangent_point.1 = tangent_point.2)
  (h_curve : curve a b tangent_point.1 = tangent_point.2)
  (h_slope : curve_derivative a tangent_point.1 = k) : 
  a = -1 ∧ b = 3 := 
by
  sorry

end find_a_and_b_l171_171632


namespace calculation_l171_171114

theorem calculation : (1 / 2) ^ (-2 : ℤ) + (-1 : ℝ) ^ (2022 : ℤ) = 5 := by
  sorry

end calculation_l171_171114


namespace smallest_prime_divides_sum_l171_171202

theorem smallest_prime_divides_sum :
  ∃ a, Prime a ∧ a ∣ (3 ^ 11 + 5 ^ 13) ∧
       ∀ b, Prime b → b ∣ (3 ^ 11 + 5 ^ 13) → a ≤ b :=
sorry

end smallest_prime_divides_sum_l171_171202


namespace power_of_two_has_half_nines_l171_171864

theorem power_of_two_has_half_nines (k : ℕ) (h : k > 1) :
  ∃ n : ℕ, (∃ m : ℕ, (k / 2 < m) ∧ 
            (10^k ∣ (2^n + m + 1)) ∧ 
            (2^n % (10^k) = 10^k - 1)) :=
sorry

end power_of_two_has_half_nines_l171_171864


namespace class_student_difference_l171_171695

theorem class_student_difference (A B : ℕ) (h : A - 4 = B + 4) : A - B = 8 := by
  sorry

end class_student_difference_l171_171695


namespace virginia_eggs_l171_171544

theorem virginia_eggs (initial_eggs : ℕ) (taken_eggs : ℕ) (result_eggs : ℕ) 
  (h_initial : initial_eggs = 200) 
  (h_taken : taken_eggs = 37) 
  (h_calculation: result_eggs = initial_eggs - taken_eggs) :
result_eggs = 163 :=
by {
  sorry
}

end virginia_eggs_l171_171544


namespace find_a_l171_171082

theorem find_a (a : ℝ) (h : (1 / Real.log 3 / Real.log a) + (1 / Real.log 5 / Real.log a) + (1 / Real.log 7 / Real.log a) = 1) : 
  a = 105 := 
sorry

end find_a_l171_171082


namespace find_zero_function_l171_171164

noncomputable def satisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x ^ 714 + y) = f (x ^ 2019) + f (y ^ 122)

theorem find_zero_function (f : ℝ → ℝ) (h : satisfiesCondition f) :
  ∀ x : ℝ, f x = 0 :=
sorry

end find_zero_function_l171_171164


namespace find_vector_c_l171_171987

def a : ℝ × ℝ := (1, -1)
def b : ℝ × ℝ := (1, 2)
def c : ℝ × ℝ := (2, 1)

def perp (v w : ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 = 0
def parallel (v w : ℝ × ℝ) : Prop := ∃ k : ℝ, w = (k * v.1, k * v.2)

theorem find_vector_c : 
  perp (c.1 + b.1, c.2 + b.2) a ∧ parallel (c.1 - a.1, c.2 + a.2) b :=
by 
  sorry

end find_vector_c_l171_171987


namespace prob_B_takes_second_shot_correct_prob_A_takes_ith_shot_correct_expected_A_shots_correct_l171_171513

noncomputable def prob_A_makes_shot : ℝ := 0.6
noncomputable def prob_B_makes_shot : ℝ := 0.8
noncomputable def prob_A_starts : ℝ := 0.5
noncomputable def prob_B_starts : ℝ := 0.5

noncomputable def prob_B_takes_second_shot : ℝ :=
  prob_A_starts * (1 - prob_A_makes_shot) + prob_B_starts * prob_B_makes_shot

theorem prob_B_takes_second_shot_correct :
  prob_B_takes_second_shot = 0.6 :=
  sorry

noncomputable def prob_A_takes_nth_shot (n : ℕ) : ℝ :=
  let p₁ := 0.5
  let recurring_prob := (1 / 6) * ((2 / 5)^(n-1))
  (1 / 3) + recurring_prob

theorem prob_A_takes_ith_shot_correct (i : ℕ) :
  prob_A_takes_nth_shot i = (1 / 3) + (1 / 6) * ((2 / 5)^(i - 1)) :=
  sorry

noncomputable def expected_A_shots (n : ℕ) : ℝ :=
  let geometric_sum := ((2 / 5)^n - 1) / (1 - (2 / 5))
  (1 / 6) * geometric_sum + (n / 3)

theorem expected_A_shots_correct (n : ℕ) :
  expected_A_shots n = (5 / 18) * (1 - (2 / 5)^n) + (n / 3) :=
  sorry

end prob_B_takes_second_shot_correct_prob_A_takes_ith_shot_correct_expected_A_shots_correct_l171_171513


namespace rational_division_example_l171_171546

theorem rational_division_example : (3 / 7) / 5 = 3 / 35 := by
  sorry

end rational_division_example_l171_171546


namespace tangent_line_relation_l171_171463

noncomputable def proof_problem (x1 x2 : ℝ) : Prop :=
  ((∃ (P Q : ℝ × ℝ),
    P = (x1, Real.log x1) ∧
    Q = (x2, Real.exp x2) ∧
    ∀ k : ℝ, Real.exp x2 = k ↔ k * (x2 - x1) = Real.log x1 - Real.exp x2) →
    (((x1 * Real.exp x2 = 1) ∧ ((x1 + 1) / (x1 - 1) + x2 = 0))))


theorem tangent_line_relation (x1 x2 : ℝ) (h : proof_problem x1 x2) : 
  (x1 * Real.exp x2 = 1) ∧ ((x1 + 1) / (x1 - 1) + x2 = 0) :=
sorry

end tangent_line_relation_l171_171463


namespace price_of_mixture_l171_171677

theorem price_of_mixture (P1 P2 P3 : ℝ) (h1 : P1 = 126) (h2 : P2 = 135) (h3 : P3 = 175.5) : 
  (P1 + P2 + 2 * P3) / 4 = 153 :=
by 
  -- Main goal is to show (126 + 135 + 2 * 175.5) / 4 = 153
  sorry

end price_of_mixture_l171_171677


namespace num_solutions_non_negative_reals_l171_171958

-- Define the system of equations as a function to express the cyclic nature
def system_of_equations (n : ℕ) (x : ℕ → ℝ) (k : ℕ) : Prop :=
  x (k + 1 % n) + (x (if k = 0 then n else k) ^ 2) = 4 * x (if k = 0 then n else k)

-- Define the main theorem stating the number of solutions
theorem num_solutions_non_negative_reals {n : ℕ} (hn : 0 < n) : 
  ∃ (s : Finset (ℕ → ℝ)), (∀ x ∈ s, ∀ k, 0 ≤ (x k) ∧ system_of_equations n x k) ∧ s.card = 2^n :=
sorry

end num_solutions_non_negative_reals_l171_171958


namespace b_is_arithmetic_sequence_a_general_formula_l171_171387

open Nat

-- Define the sequence a_n
def a : ℕ → ℤ
| 0     => 1
| 1     => 2
| (n+2) => 2 * (a (n+1)) - (a n) + 2

-- Define the sequence b_n
def b (n : ℕ) : ℤ := a (n+1) - a n

-- Part 1: The sequence b_n is an arithmetic sequence
theorem b_is_arithmetic_sequence : ∀ n : ℕ, b (n+1) - b n = 2 := by
  sorry

-- Part 2: Find the general formula for a_n
theorem a_general_formula : ∀ n : ℕ, a (n+1) = n^2 + 1 := by
  sorry

end b_is_arithmetic_sequence_a_general_formula_l171_171387


namespace determine_m_l171_171191

theorem determine_m (a m : ℝ) (h : a > 0) (h2 : (m, 3) ∈ set_of (λ p : ℝ × ℝ, p.2 = -a * p.1 ^ 2 + 2 * a * p.1 + 3)) (h3 : m ≠ 0) : m = 2 :=
sorry

end determine_m_l171_171191


namespace prime_gt_three_modulus_l171_171480

theorem prime_gt_three_modulus (p : ℕ) (hp : Prime p) (hp_gt3 : p > 3) : (p^2 + 12) % 12 = 1 := by
  sorry

end prime_gt_three_modulus_l171_171480


namespace value_of_m_l171_171194

theorem value_of_m (a m : ℝ) (h : a > 0) (hm : m ≠ 0) :
  (P : ℝ × ℝ) (P = (m, 3))
  (H : ∀ x : ℝ, -a * x^2 + 2 * a * x + 3 = 3 → x = 0 ∨ x = 2) :
  m = 2 :=
by
  sorry

end value_of_m_l171_171194


namespace minimum_point_translation_l171_171376

noncomputable def f (x : ℝ) : ℝ := |x| - 2

theorem minimum_point_translation :
  let minPoint := (0, f 0)
  let newMinPoint := (minPoint.1 + 4, minPoint.2 + 5)
  newMinPoint = (4, 3) :=
by
  sorry

end minimum_point_translation_l171_171376


namespace triangle_angle_zero_degrees_l171_171014

theorem triangle_angle_zero_degrees {a b c : ℝ} (h : (a + b + c) * (a + b - c) = 4 * a * b) :
  ∃ (C : ℝ), C = 0 ∧ c = 0 :=
sorry

end triangle_angle_zero_degrees_l171_171014


namespace find_base_b_l171_171477

-- Defining the conditions
def base_representation_784 (b : ℕ) : ℕ := 7 * b^2 + 8 * b + 4
def base_representation_28 (b : ℕ) : ℕ := 2 * b + 8

-- Theorem to prove that the base b is 10
theorem find_base_b (b : ℕ) (h : (base_representation_28 b)^2 = base_representation_784 b) : b = 10 :=
sorry

end find_base_b_l171_171477


namespace shaded_area_square_with_circles_l171_171626

-- Defining the conditions
def side_length : ℝ := 8
def radius : ℝ := 3

/-
The goal is to prove that the shaded area is equal to 
64 - 16 * real.sqrt 7 - 18 * real.arcsin (real.sqrt 7 / 4)
-/
theorem shaded_area_square_with_circles :
  let A_shaded := side_length^2 - 4 * (2 * real.sqrt(4^2 - radius^2)) - 2 * (4 * radius^2 * real.arcsin(real.sqrt(4^2 - radius^2) / 4)) / π
  A_shaded = 64 - 16 * real.sqrt 7 - 18 * real.arcsin (real.sqrt 7 / 4) := 
by {
  sorry
}

end shaded_area_square_with_circles_l171_171626


namespace income_distribution_after_tax_l171_171407

theorem income_distribution_after_tax (x : ℝ) (hx : 10 * x = 100) :
  let poor_income_initial := x
  let middle_income_initial := 4 * x
  let rich_income_initial := 5 * x
  let tax_rate := (x^2 / 4) + x
  let tax_collected := tax_rate * rich_income_initial / 100
  let poor_income_after := poor_income_initial + 3 / 4 * tax_collected
  let middle_income_after := middle_income_initial + 1 / 4 * tax_collected
  let rich_income_after := rich_income_initial - tax_collected
  poor_income_after = 0.23125 * 100 ∧
  middle_income_after = 0.44375 * 100 ∧
  rich_income_after = 0.325 * 100 :=
by {
  sorry
}

end income_distribution_after_tax_l171_171407


namespace log_division_simplification_l171_171044

theorem log_division_simplification (log_base_half : ℝ → ℝ) (log_base_half_pow5 :  log_base_half (2 ^ 5) = 5 * log_base_half 2)
  (log_base_half_pow1 : log_base_half (2 ^ 1) = 1 * log_base_half 2) :
  (log_base_half 32) / (log_base_half 2) = 5 :=
sorry

end log_division_simplification_l171_171044


namespace smallest_solution_of_quartic_equation_l171_171081

theorem smallest_solution_of_quartic_equation :
  ∃ x : ℝ, (x^4 - 50 * x^2 + 625 = 0) ∧ ∀ y : ℝ, y^4 - 50 * y^2 + 625 = 0 → x ≤ y := sorry

end smallest_solution_of_quartic_equation_l171_171081


namespace scientific_notation_correct_l171_171366

theorem scientific_notation_correct :
  0.000000007 = 7 * 10^(-9) :=
by
  sorry

end scientific_notation_correct_l171_171366


namespace sin_value_l171_171175

open Real

-- Define the given conditions
variables (x : ℝ) (h1 : cos (π + x) = 3 / 5) (h2 : π < x) (h3 : x < 2 * π)

-- State the problem to be proved
theorem sin_value : sin x = - 4 / 5 :=
by
  sorry

end sin_value_l171_171175


namespace DianasInitialSpeed_l171_171452

open Nat

theorem DianasInitialSpeed
  (total_distance : ℕ)
  (initial_time : ℕ)
  (tired_speed : ℕ)
  (total_time : ℕ)
  (distance_when_tired : ℕ)
  (initial_distance : ℕ)
  (initial_speed : ℕ)
  (initial_hours : ℕ) :
  total_distance = 10 →
  initial_time = 2 →
  tired_speed = 1 →
  total_time = 6 →
  distance_when_tired = tired_speed * (total_time - initial_time) →
  initial_distance = total_distance - distance_when_tired →
  initial_distance = initial_speed * initial_time →
  initial_speed = 3 := by
  sorry

end DianasInitialSpeed_l171_171452


namespace tenth_term_l171_171015

-- Define the conditions
variables {a d : ℤ}

-- The conditions of the problem
axiom third_term_condition : a + 2 * d = 10
axiom sixth_term_condition : a + 5 * d = 16

-- The goal is to prove the tenth term
theorem tenth_term : a + 9 * d = 24 :=
by
  sorry

end tenth_term_l171_171015


namespace smallest_of_seven_even_numbers_sum_448_l171_171519

theorem smallest_of_seven_even_numbers_sum_448 :
  ∃ n : ℤ, n + (n+2) + (n+4) + (n+6) + (n+8) + (n+10) + (n+12) = 448 ∧ n = 58 := 
by
  sorry

end smallest_of_seven_even_numbers_sum_448_l171_171519


namespace multiplication_equation_l171_171383

-- Define the given conditions
def multiplier : ℕ := 6
def product : ℕ := 168
def multiplicand : ℕ := product - 140

-- Lean statement for the proof
theorem multiplication_equation : multiplier * multiplicand = product := by
  sorry

end multiplication_equation_l171_171383


namespace solve_for_x_l171_171754

theorem solve_for_x :
  (∀ y : ℝ, 10 * x * y - 15 * y + 4 * x - 6 = 0) ↔ x = 3 / 2 :=
by
  sorry

end solve_for_x_l171_171754


namespace solution_set_of_inequality_l171_171891

theorem solution_set_of_inequality (x : ℝ) (h : x ≠ 0) : 
  (x + 1) / x ≤ 3 ↔ x ∈ Set.Iio (0 : ℝ) ∪ Set.Ici (1 / 2) :=
by sorry

end solution_set_of_inequality_l171_171891


namespace triangle_third_side_length_l171_171781

theorem triangle_third_side_length (a b c : ℝ) (h₁ : a = 5) (h₂ : b = 8) :
  (a + b > c) → (a + c > b) → (b + c > a) → c = 6 :=
by
  intros h₃ h₄ h₅
  rw [h₁, h₂] at *
  -- Simplified inequalities from the solution step
  have h₆ : 5 + 8 > c := by rw [h₁, h₂]; exact h₃
  have h₇ : 8 + c > 5 := by rw h₂; exact h₄
  have h₈ : 5 + c > 8 := by rw h₁; exact h₅
  sorry

end triangle_third_side_length_l171_171781


namespace functional_equation_solution_l171_171155

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x * f y - y * f x) = f (x * y) - x * y) →
  (f = id ∨ f = abs) :=
by sorry

end functional_equation_solution_l171_171155


namespace min_value_f_l171_171302

def f (x y : ℝ) : ℝ := x^2 + 4 * x * y + 5 * y^2 - 8 * x - 6 * y

theorem min_value_f : ∃ x y : ℝ, f x y = -9 / 5 :=
sorry

end min_value_f_l171_171302


namespace distance_from_origin_is_correct_l171_171579

-- Define the point (x, y) with given conditions
variables (x y : ℝ)

-- Given conditions
axiom h1 : y = 20
axiom h2 : dist (x, y) (2, 15) = 15
axiom h3 : x > 2

-- The theorem to prove
theorem distance_from_origin_is_correct :
  dist (x, y) (0, 0) = Real.sqrt (604 + 40 * Real.sqrt 2) :=
by
  -- Set h1, h2, and h3 as our constraints
  sorry

end distance_from_origin_is_correct_l171_171579


namespace correct_calculation_l171_171084

theorem correct_calculation : (6 + (-13)) = -7 :=
by
  sorry

end correct_calculation_l171_171084


namespace solveSALE_l171_171349

namespace Sherlocked

open Nat

def areDistinctDigits (d₁ d₂ d₃ d₄ d₅ d₆ : Nat) : Prop :=
  d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₁ ≠ d₄ ∧ d₁ ≠ d₅ ∧ d₁ ≠ d₆ ∧ 
  d₂ ≠ d₃ ∧ d₂ ≠ d₄ ∧ d₂ ≠ d₅ ∧ d₂ ≠ d₆ ∧ 
  d₃ ≠ d₄ ∧ d₃ ≠ d₅ ∧ d₃ ≠ d₆ ∧ 
  d₄ ≠ d₅ ∧ d₄ ≠ d₆ ∧ 
  d₅ ≠ d₆

theorem solveSALE :
  ∃ (S C A L E T : ℕ),
    SCALE - SALE = SLATE ∧
    areDistinctDigits S C A L E T ∧
    S < 10 ∧ C < 10 ∧ A < 10 ∧
    L < 10 ∧ E < 10 ∧ T < 10 ∧
    SALE = 1829 :=
by
  sorry

end Sherlocked

end solveSALE_l171_171349


namespace condition_equiv_l171_171050

theorem condition_equiv (p q : Prop) : (¬ (p ∧ q) ∧ (p ∨ q)) ↔ ((p ∨ q) ∧ (¬ p ↔ q)) :=
  sorry

end condition_equiv_l171_171050


namespace max_gcd_13n_plus_4_8n_plus_3_l171_171597

theorem max_gcd_13n_plus_4_8n_plus_3 : 
  ∀ n : ℕ, n > 0 → ∃ d : ℕ, d = 7 ∧ ∀ k : ℕ, k = gcd (13 * n + 4) (8 * n + 3) → k ≤ d :=
by
  sorry

end max_gcd_13n_plus_4_8n_plus_3_l171_171597


namespace ducks_at_North_Pond_l171_171671

theorem ducks_at_North_Pond :
  (∀ (ducks_Lake_Michigan : ℕ), ducks_Lake_Michigan = 100 → (6 + 2 * ducks_Lake_Michigan) = 206) :=
by
  intros ducks_Lake_Michigan hL
  rw [hL]
  norm_num
  rfl

end ducks_at_North_Pond_l171_171671


namespace treasure_coins_problem_l171_171855

theorem treasure_coins_problem (N m n t k s u : ℤ) 
  (h1 : N = (2/3) * (2/3) * (2/3) * (m - 1) - (2/3) - (2^2 / 3^2))
  (h2 : N = 3 * n)
  (h3 : 8 * (m - 1) - 30 = 81 * k)
  (h4 : m - 1 = 3 * t)
  (h5 : 8 * t - 27 * k = 10)
  (h6 : m = 3 * t + 1)
  (h7 : k = 2 * s)
  (h8 : 4 * t - 27 * s = 5)
  (h9 : t = 8 + 27 * u)
  (h10 : s = 1 + 4 * u)
  (h11 : 110 ≤ 81 * u + 25)
  (h12 : 81 * u + 25 ≤ 200) :
  m = 187 :=
sorry

end treasure_coins_problem_l171_171855


namespace find_value_of_sum_of_squares_l171_171183

theorem find_value_of_sum_of_squares (x y : ℝ) (h : x^2 + y^2 + x^2 * y^2 - 4 * x * y + 1 = 0) :
  (x + y)^2 = 4 :=
sorry

end find_value_of_sum_of_squares_l171_171183


namespace count_prime_numbers_in_sequence_l171_171297

theorem count_prime_numbers_in_sequence : 
  ∀ (k : Nat), (∃ n : Nat, 47 * (10^n * k + (10^(n-1) - 1) / 9) = 47) → k = 0 :=
  sorry

end count_prime_numbers_in_sequence_l171_171297


namespace find_m_l171_171187

theorem find_m (a m : ℝ) (h_pos : a > 0) (h_points : (m, 3) ∈ set_of (λ x : ℝ × ℝ, ∃ x_val : ℝ, x.snd = -a * (x_val)^2 + 2 * a * x_val + 3)) (h_non_zero : m ≠ 0) : m = 2 := 
sorry

end find_m_l171_171187


namespace not_necessarily_divisible_by_66_l171_171358

open Nat

-- Definition of what it means to be the product of four consecutive integers
def product_of_four_consecutive_integers (n : ℕ) : Prop :=
  ∃ k : ℤ, n = (k * (k + 1) * (k + 2) * (k + 3))

-- Lean theorem statement for the proof problem
theorem not_necessarily_divisible_by_66 (n : ℕ) 
  (h1 : product_of_four_consecutive_integers n) 
  (h2 : 11 ∣ n) : ¬ (66 ∣ n) :=
sorry

end not_necessarily_divisible_by_66_l171_171358


namespace intervals_of_increase_l171_171752

def f (x : ℝ) : ℝ := 2*x^3 - 6*x^2 + 7

theorem intervals_of_increase : 
  ∀ x : ℝ, (x < 0 ∨ x > 2) → (6*x^2 - 12*x > 0) :=
by
  -- Placeholder for proof
  sorry

end intervals_of_increase_l171_171752


namespace wrestling_match_student_count_l171_171250

theorem wrestling_match_student_count (n : ℕ) (h : n * (n - 1) / 2 = 91) : n = 14 := by
  sorry

end wrestling_match_student_count_l171_171250


namespace angelina_speed_l171_171559

theorem angelina_speed (v : ℝ) (h1 : 200 / v - 50 = 300 / (2 * v)) : 2 * v = 2 := 
by
  sorry

end angelina_speed_l171_171559


namespace total_gift_amount_l171_171485

-- Definitions based on conditions
def workers_per_block := 200
def number_of_blocks := 15
def worth_of_each_gift := 2

-- The statement we need to prove
theorem total_gift_amount : workers_per_block * number_of_blocks * worth_of_each_gift = 6000 := by
  sorry

end total_gift_amount_l171_171485


namespace probability_visible_l171_171516

-- Definitions of the conditions
def lap_time_sarah : ℕ := 120
def lap_time_sam : ℕ := 100
def start_to_photo_min : ℕ := 15
def start_to_photo_max : ℕ := 16
def photo_fraction : ℚ := 1/3
def shadow_start_interval : ℕ := 45
def shadow_duration : ℕ := 15

-- The theorem to prove
theorem probability_visible :
  let total_time := 60
  let valid_overlap_time := 13.33
  valid_overlap_time / total_time = 1333 / 6000 :=
by {
  sorry
}

end probability_visible_l171_171516


namespace inequality_ineq_l171_171770

variable (x y z : Real)

theorem inequality_ineq {x y z : Real} (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x^2 + y^2 + z^2 = 3) :
  (1 / (x^5 - x^2 + 3)) + (1 / (y^5 - y^2 + 3)) + (1 / (z^5 - z^2 + 3)) ≤ 1 :=
by 
  sorry

end inequality_ineq_l171_171770


namespace find_number_l171_171906

theorem find_number :
  ∃ x : ℕ, (8 * x + 5400) / 12 = 530 ∧ x = 120 :=
by
  sorry

end find_number_l171_171906


namespace function_increasing_no_negative_roots_l171_171185

noncomputable def f (a x : ℝ) : ℝ := a^x + (x - 2) / (x + 1)

theorem function_increasing (a : ℝ) (h : a > 1) : 
  ∀ (x1 x2 : ℝ), (-1 < x1) → (x1 < x2) → (f a x1 < f a x2) := 
by
  -- placeholder proof
  sorry

theorem no_negative_roots (a : ℝ) (h : a > 1) : 
  ∀ (x : ℝ), (x < 0) → (f a x ≠ 0) := 
by
  -- placeholder proof
  sorry

end function_increasing_no_negative_roots_l171_171185


namespace difference_between_percentages_l171_171567

noncomputable def number : ℝ := 140

noncomputable def percentage_65 (x : ℝ) : ℝ := 0.65 * x

noncomputable def fraction_4_5 (x : ℝ) : ℝ := 0.8 * x

theorem difference_between_percentages 
  (x : ℝ) 
  (hx : x = number) 
  : (fraction_4_5 x) - (percentage_65 x) = 21 := 
by 
  sorry

end difference_between_percentages_l171_171567


namespace factor_sum_l171_171760

theorem factor_sum :
  ∃ d e f : ℤ, (∀ x : ℤ, x^2 + 11 * x + 24 = (x + d) * (x + e)) ∧
              (∀ x : ℤ, x^2 + 9 * x - 36 = (x + e) * (x - f)) ∧
              d + e + f = 14 := by
  sorry

end factor_sum_l171_171760


namespace least_number_to_be_added_l171_171962

theorem least_number_to_be_added (k : ℕ) (h₁ : Nat.Prime 29) (h₂ : Nat.Prime 37) (H : Nat.gcd 29 37 = 1) : 
  (433124 + k) % Nat.lcm 29 37 = 0 → k = 578 :=
by 
  sorry

end least_number_to_be_added_l171_171962


namespace layla_goals_l171_171334

variable (L K : ℕ)
variable (average_score : ℕ := 92)
variable (goals_difference : ℕ := 24)
variable (total_games : ℕ := 4)

theorem layla_goals :
  K = L - goals_difference →
  (L + K) = (average_score * total_games) →
  L = 196 :=
by
  sorry

end layla_goals_l171_171334


namespace positive_real_x_condition_l171_171163

-- We define the conditions:
variables (x : ℝ)
#check (1 - x^4)
#check (1 + x^4)

-- The main proof statement:
theorem positive_real_x_condition (h1 : x > 0) 
    (h2 : (Real.sqrt (Real.sqrt (1 - x^4)) + Real.sqrt (Real.sqrt (1 + x^4)) = 1)) :
    (x^8 = 35 / 36) :=
sorry

end positive_real_x_condition_l171_171163


namespace largest_reflections_l171_171581

-- Define initial conditions
variables (A B D : Point)  -- Points in the plane
variable (n : ℕ) -- Number of reflections
variable (a : ℝ) -- Angle at reflection

-- Define reflection conditions
variable (AD CD : Line) -- Lines involved
variable (angleCDA : ℝ) -- Given angle

-- Lean statement of the equivalent proof problem
theorem largest_reflections (h_angle: angleCDA = 8)
                           (h_angle_unit: angleCDA * n ≤ 90):
  n ≤ 11 := 
  -- Given conditions above and problem's context
  sorry

end largest_reflections_l171_171581


namespace sqrt_of_25_l171_171115

theorem sqrt_of_25 : ∃ x : ℝ, x^2 = 25 ∧ (x = 5 ∨ x = -5) :=
by {
  sorry
}

end sqrt_of_25_l171_171115


namespace race_head_start_l171_171558

variable (vA vB L h : ℝ)
variable (hva_vb : vA = (16 / 15) * vB)

theorem race_head_start (hL_pos : L > 0) (hvB_pos : vB > 0) 
    (h_times_eq : (L / vA) = ((L - h) / vB)) : h = L / 16 :=
by
  sorry

end race_head_start_l171_171558


namespace smallest_possible_value_of_N_l171_171828

-- Declares the context and required constraints
theorem smallest_possible_value_of_N (N : ℕ) (h1 : 70 < N) (h2 : 70 ∣ 21 * N) : N = 80 :=
by
  sorry

end smallest_possible_value_of_N_l171_171828


namespace correct_number_of_outfits_l171_171557

-- Define the number of each type of clothing
def num_red_shirts := 4
def num_green_shirts := 4
def num_blue_shirts := 4
def num_pants := 10
def num_red_hats := 6
def num_green_hats := 6
def num_blue_hats := 4

-- Define the total number of outfits that meet the conditions
def total_outfits : ℕ :=
  (num_red_shirts * num_pants * (num_green_hats + num_blue_hats)) +
  (num_green_shirts * num_pants * (num_red_hats + num_blue_hats)) +
  (num_blue_shirts * num_pants * (num_red_hats + num_green_hats))

-- The proof statement asserting that the total number of valid outfits is 1280
theorem correct_number_of_outfits : total_outfits = 1280 := by
  sorry

end correct_number_of_outfits_l171_171557


namespace right_triangle_sides_unique_l171_171878

theorem right_triangle_sides_unique (a b c : ℕ) 
  (relatively_prime : Int.gcd (Int.gcd a b) c = 1) 
  (right_triangle : a ^ 2 + b ^ 2 = c ^ 2) 
  (increased_right_triangle : (a + 100) ^ 2 + (b + 100) ^ 2 = (c + 140) ^ 2) : 
  (a = 56 ∧ b = 33 ∧ c = 65) :=
by
  sorry 

end right_triangle_sides_unique_l171_171878


namespace robbers_can_divide_loot_equally_l171_171903

theorem robbers_can_divide_loot_equally (coins : List ℕ) (h1 : (coins.sum % 2 = 0)) 
    (h2 : ∀ k, (k % 2 = 1 ∧ 1 ≤ k ∧ k ≤ 2017) → k ∈ coins) :
  ∃ (subset1 subset2 : List ℕ), subset1 ∪ subset2 = coins ∧ subset1.sum = subset2.sum :=
by
  sorry

end robbers_can_divide_loot_equally_l171_171903


namespace triangle_side_lengths_condition_l171_171768

noncomputable def f (x k : ℝ) : ℝ := (x^2 + k*x + 1) / (x^2 + x + 1)

theorem triangle_side_lengths_condition (k : ℝ) :
  (∀ x1 x2 x3 : ℝ, x1 > 0 → x2 > 0 → x3 > 0 →
    (f x1 k) + (f x2 k) > (f x3 k) ∧ (f x2 k) + (f x3 k) > (f x1 k) ∧ (f x3 k) + (f x1 k) > (f x2 k))
  ↔ (-1/2 ≤ k ∧ k ≤ 4) :=
by
  sorry

end triangle_side_lengths_condition_l171_171768


namespace car_speed_l171_171320

theorem car_speed (rev_per_min : ℕ) (circ : ℝ) (h_rev : rev_per_min = 400) (h_circ : circ = 5) : 
  (rev_per_min * circ) * 60 / 1000 = 120 :=
by
  sorry

end car_speed_l171_171320


namespace negation_of_exists_statement_l171_171883

theorem negation_of_exists_statement :
  ¬ (∃ x0 : ℝ, x0 > 0 ∧ x0^2 - 5 * x0 + 6 > 0) ↔ ∀ x : ℝ, x > 0 → x^2 - 5 * x + 6 ≤ 0 :=
by
  sorry

end negation_of_exists_statement_l171_171883


namespace sufficient_balance_after_29_months_l171_171911

noncomputable def accumulated_sum (S0 : ℕ) (D : ℕ) (r : ℚ) (n : ℕ) : ℚ :=
  S0 * (1 + r)^n + D * ((1 + r)^n - 1) / r

theorem sufficient_balance_after_29_months :
  let S0 := 300000
  let D := 15000
  let r := (1 / 100 : ℚ) -- interest rate of 1%
  accumulated_sum S0 D r 29 ≥ 900000 :=
by
  sorry -- The proof will be elaborated later

end sufficient_balance_after_29_months_l171_171911
