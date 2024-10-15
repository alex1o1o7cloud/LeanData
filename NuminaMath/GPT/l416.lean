import Mathlib

namespace NUMINAMATH_GPT_blue_notebook_cost_l416_41661

theorem blue_notebook_cost 
  (total_spent : ℕ)
  (total_notebooks : ℕ)
  (red_notebooks : ℕ)
  (cost_per_red : ℕ)
  (green_notebooks : ℕ)
  (cost_per_green : ℕ)
  (blue_notebooks : ℕ)
  (total_cost_blue : ℕ)
  (cost_per_blue : ℕ)
  (h1 : total_spent = 37)
  (h2 : total_notebooks = 12)
  (h3 : red_notebooks = 3)
  (h4 : cost_per_red = 4)
  (h5 : green_notebooks = 2)
  (h6 : cost_per_green = 2)
  (h7 : total_cost_blue = total_spent - (red_notebooks * cost_per_red + green_notebooks * cost_per_green))
  (h8 : blue_notebooks = total_notebooks - (red_notebooks + green_notebooks))
  (h9 : cost_per_blue = total_cost_blue / blue_notebooks)
  : cost_per_blue = 3 :=
sorry

end NUMINAMATH_GPT_blue_notebook_cost_l416_41661


namespace NUMINAMATH_GPT_problem_solution_l416_41685

theorem problem_solution
  (p q : ℝ)
  (h₁ : p ≠ q)
  (h₂ : (x : ℝ) → (x - 5) * (x + 3) = 24 * x - 72 → x = p ∨ x = q)
  (h₃ : p > q) :
  p - q = 20 :=
sorry

end NUMINAMATH_GPT_problem_solution_l416_41685


namespace NUMINAMATH_GPT_maximum_value_conditions_l416_41638

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / (1 + x) - Real.log x

theorem maximum_value_conditions (x_0 : ℝ) (h_max : ∀ x : ℝ, f x ≤ f x_0) :
    f x_0 = x_0 ∧ f x_0 < 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_maximum_value_conditions_l416_41638


namespace NUMINAMATH_GPT_archer_probability_less_than_8_l416_41657

-- Define the conditions as probabilities for hitting the 10-ring, 9-ring, and 8-ring.
def p_10 : ℝ := 0.24
def p_9 : ℝ := 0.28
def p_8 : ℝ := 0.19

-- Define the probability that the archer scores at least 8.
def p_at_least_8 : ℝ := p_10 + p_9 + p_8

-- Calculate the probability of the archer scoring less than 8.
def p_less_than_8 : ℝ := 1 - p_at_least_8

-- Now, state the theorem to prove that this probability is equal to 0.29.
theorem archer_probability_less_than_8 : p_less_than_8 = 0.29 := by sorry

end NUMINAMATH_GPT_archer_probability_less_than_8_l416_41657


namespace NUMINAMATH_GPT_p_sufficient_not_necessary_for_q_l416_41641

-- Define the propositions p and q based on the given conditions
def p (α : ℝ) : Prop := α = Real.pi / 4
def q (α : ℝ) : Prop := Real.sin α = Real.cos α

-- Theorem that states p is a sufficient but not necessary condition for q
theorem p_sufficient_not_necessary_for_q (α : ℝ) : p α → (q α) ∧ ¬(q α → p α) :=
by
  sorry

end NUMINAMATH_GPT_p_sufficient_not_necessary_for_q_l416_41641


namespace NUMINAMATH_GPT_work_completion_in_8_days_l416_41678

/-- Definition of the individual work rates and the combined work rate. -/
def work_rate_A := 1 / 12
def work_rate_B := 1 / 24
def combined_work_rate := work_rate_A + work_rate_B

/-- The main theorem stating that A and B together complete the job in 8 days. -/
theorem work_completion_in_8_days (h1 : work_rate_A = 1 / 12) (h2 : work_rate_B = 1 / 24) : 
  1 / combined_work_rate = 8 :=
by
  sorry

end NUMINAMATH_GPT_work_completion_in_8_days_l416_41678


namespace NUMINAMATH_GPT_snowball_game_l416_41613

theorem snowball_game (x y z : ℕ) (h : 5 * x + 4 * y + 3 * z = 12) : 
  x = 1 ∧ y = 1 ∧ z = 1 := by
  sorry

end NUMINAMATH_GPT_snowball_game_l416_41613


namespace NUMINAMATH_GPT_classroom_lamps_total_ways_l416_41667

theorem classroom_lamps_total_ways (n : ℕ) (h : n = 4) : (2^n - 1) = 15 :=
by
  sorry

end NUMINAMATH_GPT_classroom_lamps_total_ways_l416_41667


namespace NUMINAMATH_GPT_length_QF_l416_41675

-- Define parabola C as y^2 = 8x
def is_on_parabola (P : ℝ × ℝ) : Prop :=
  P.2 * P.2 = 8 * P.1

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define the condition that Q is on the parabola and the line PF in the first quadrant
def is_intersection_and_in_first_quadrant (Q : ℝ × ℝ) (P : ℝ × ℝ) : Prop :=
  is_on_parabola Q ∧ Q.1 - Q.2 - 2 = 0 ∧ Q.1 > 0 ∧ Q.2 > 0

-- Define the vector relation between P, Q, and F
def vector_relation (P Q F : ℝ × ℝ) : Prop :=
  let vPQ := (Q.1 - P.1, Q.2 - P.2)
  let vQF := (F.1 - Q.1, F.2 - Q.2)
  (vPQ.1^2 + vPQ.2^2) = 2 * (vQF.1^2 + vQF.2^2)

-- Lean 4 statement of the proof problem
theorem length_QF (Q : ℝ × ℝ) (P : ℝ × ℝ) :
  is_on_parabola Q ∧ is_intersection_and_in_first_quadrant Q P ∧ vector_relation P Q focus → 
  dist Q focus = 8 + 4 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_length_QF_l416_41675


namespace NUMINAMATH_GPT_soccer_league_points_l416_41628

structure Team :=
  (name : String)
  (regular_wins : ℕ)
  (losses : ℕ)
  (draws : ℕ)
  (bonus_wins : ℕ)

def total_points (t : Team) : ℕ :=
  3 * t.regular_wins + t.draws + 2 * t.bonus_wins

def Team_Soccer_Stars : Team :=
  { name := "Team Soccer Stars", regular_wins := 18, losses := 5, draws := 7, bonus_wins := 6 }

def Lightning_Strikers : Team :=
  { name := "Lightning Strikers", regular_wins := 15, losses := 8, draws := 7, bonus_wins := 5 }

def Goal_Grabbers : Team :=
  { name := "Goal Grabbers", regular_wins := 21, losses := 5, draws := 4, bonus_wins := 4 }

def Clever_Kickers : Team :=
  { name := "Clever Kickers", regular_wins := 11, losses := 10, draws := 9, bonus_wins := 2 }

theorem soccer_league_points :
  total_points Team_Soccer_Stars = 73 ∧
  total_points Lightning_Strikers = 62 ∧
  total_points Goal_Grabbers = 75 ∧
  total_points Clever_Kickers = 46 ∧
  [Goal_Grabbers, Team_Soccer_Stars, Lightning_Strikers, Clever_Kickers].map total_points =
  [75, 73, 62, 46] := 
by
  sorry

end NUMINAMATH_GPT_soccer_league_points_l416_41628


namespace NUMINAMATH_GPT_Carter_reads_30_pages_in_1_hour_l416_41642

variables (C L O : ℕ)

def Carter_reads_half_as_many_pages_as_Lucy_in_1_hour (C L : ℕ) : Prop :=
  C = L / 2

def Lucy_reads_20_more_pages_than_Oliver_in_1_hour (L O : ℕ) : Prop :=
  L = O + 20

def Oliver_reads_40_pages_in_1_hour (O : ℕ) : Prop :=
  O = 40

theorem Carter_reads_30_pages_in_1_hour
  (C L O : ℕ)
  (h1 : Carter_reads_half_as_many_pages_as_Lucy_in_1_hour C L)
  (h2 : Lucy_reads_20_more_pages_than_Oliver_in_1_hour L O)
  (h3 : Oliver_reads_40_pages_in_1_hour O) : 
  C = 30 :=
by
  sorry

end NUMINAMATH_GPT_Carter_reads_30_pages_in_1_hour_l416_41642


namespace NUMINAMATH_GPT_no_positive_integer_has_product_as_perfect_square_l416_41614

theorem no_positive_integer_has_product_as_perfect_square:
  ¬ ∃ n : ℕ, n > 0 ∧ ∃ k : ℕ, n * (n + 1) = k * k :=
by
  sorry

end NUMINAMATH_GPT_no_positive_integer_has_product_as_perfect_square_l416_41614


namespace NUMINAMATH_GPT_clarinet_cost_correct_l416_41633

noncomputable def total_spent : ℝ := 141.54
noncomputable def song_book_cost : ℝ := 11.24
noncomputable def clarinet_cost : ℝ := total_spent - song_book_cost

theorem clarinet_cost_correct : clarinet_cost = 130.30 :=
by
  sorry

end NUMINAMATH_GPT_clarinet_cost_correct_l416_41633


namespace NUMINAMATH_GPT_find_x_l416_41647

theorem find_x (x : ℝ) (h : 1 / 7 + 7 / x = 15 / x + 1 / 15) : x = 105 := 
by 
  sorry

end NUMINAMATH_GPT_find_x_l416_41647


namespace NUMINAMATH_GPT_max_volume_is_correct_l416_41651

noncomputable def max_volume_of_inscribed_sphere (AB BC AA₁ : ℝ) (h₁ : AB = 6) (h₂ : BC = 8) (h₃ : AA₁ = 3) : ℝ :=
  let AC := Real.sqrt ((6 : ℝ) ^ 2 + (8 : ℝ) ^ 2)
  let r := (AB + BC - AC) / 2
  let sphere_radius := AA₁ / 2
  (4/3) * Real.pi * sphere_radius ^ 3

theorem max_volume_is_correct : max_volume_of_inscribed_sphere 6 8 3 (by rfl) (by rfl) (by rfl) = 9 * Real.pi / 2 := by
  sorry

end NUMINAMATH_GPT_max_volume_is_correct_l416_41651


namespace NUMINAMATH_GPT_no_positive_real_solution_l416_41616

open Real

theorem no_positive_real_solution (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) :
  ¬(∀ n : ℕ, 0 < n → (n - 2) / a ≤ ⌊b * n⌋ ∧ ⌊b * n⌋ < (n - 1) / a) :=
by sorry

end NUMINAMATH_GPT_no_positive_real_solution_l416_41616


namespace NUMINAMATH_GPT_quadratic_real_solution_l416_41604

theorem quadratic_real_solution (m : ℝ) (i : ℂ) (h_i : i * i = -1)
  (h_quad : ∃ z : ℝ, z^2 + (i * z) + m = 0) : m = 0 :=
sorry

end NUMINAMATH_GPT_quadratic_real_solution_l416_41604


namespace NUMINAMATH_GPT_emmy_rosa_ipods_total_l416_41680

theorem emmy_rosa_ipods_total :
  ∃ (emmy_initial rosa_current : ℕ), 
    emmy_initial = 14 ∧ 
    (emmy_initial - 6) / 2 = rosa_current ∧ 
    (emmy_initial - 6) + rosa_current = 12 :=
by
  sorry

end NUMINAMATH_GPT_emmy_rosa_ipods_total_l416_41680


namespace NUMINAMATH_GPT_actual_distance_traveled_l416_41698

variable (t : ℝ) -- let t be the actual time in hours
variable (d : ℝ) -- let d be the actual distance traveled at 12 km/hr

-- Conditions
def condition1 := 20 * t = 12 * t + 30
def condition2 := d = 12 * t

-- The target we want to prove
theorem actual_distance_traveled (t : ℝ) (d : ℝ) (h1 : condition1 t) (h2 : condition2 t d) : 
  d = 45 := by
  sorry

end NUMINAMATH_GPT_actual_distance_traveled_l416_41698


namespace NUMINAMATH_GPT_parabola_directrix_l416_41688

noncomputable def directrix_value (a : ℝ) : ℝ := -1 / (4 * a)

theorem parabola_directrix (a : ℝ) (h : directrix_value a = 2) : a = -1 / 8 :=
by
  sorry

end NUMINAMATH_GPT_parabola_directrix_l416_41688


namespace NUMINAMATH_GPT_polynomial_value_l416_41672

variable (a b : ℝ)

theorem polynomial_value :
  2 * a + 3 * b = 5 → 6 * a + 9 * b - 12 = 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_polynomial_value_l416_41672


namespace NUMINAMATH_GPT_slope_of_line_l416_41693

theorem slope_of_line (s x y : ℝ) (h1 : 2 * x + 3 * y = 8 * s + 5) (h2 : x + 2 * y = 3 * s + 2) :
  ∃ m c : ℝ, ∀ x y, x = m * y + c ∧ m = -7/2 :=
by
  sorry

end NUMINAMATH_GPT_slope_of_line_l416_41693


namespace NUMINAMATH_GPT_binary_to_decimal_correct_l416_41624

def binary_to_decimal : ℕ := 110011

theorem binary_to_decimal_correct : 
  binary_to_decimal = 51 := sorry

end NUMINAMATH_GPT_binary_to_decimal_correct_l416_41624


namespace NUMINAMATH_GPT_lowest_number_in_range_l416_41659

theorem lowest_number_in_range (y : ℕ) (h : ∀ x y : ℕ, 0 < x ∧ x < y) : ∃ x : ℕ, x = 999 :=
by
  existsi 999
  sorry

end NUMINAMATH_GPT_lowest_number_in_range_l416_41659


namespace NUMINAMATH_GPT_range_of_a_l416_41643

theorem range_of_a
  (a : ℝ)
  (h1 : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 ≥ a)
  (h2 : ∃ x0 : ℝ, x0^2 + 2*a*x0 + 2 - a = 0) :
  a ≤ -2 ∨ a = 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l416_41643


namespace NUMINAMATH_GPT_initial_card_count_l416_41658

theorem initial_card_count (x : ℕ) (h1 : (3 * (1/2) * ((x / 3) + (4 / 3))) = 34) : x = 64 :=
  sorry

end NUMINAMATH_GPT_initial_card_count_l416_41658


namespace NUMINAMATH_GPT_largest_element_lg11_l416_41646

variable (x y : ℝ)
variable (A : Set ℝ)  (B : Set ℝ)

-- Conditions
def condition1 : A = Set.insert (Real.log x) (Set.insert (Real.log y) (Set.insert (Real.log (x + y / x)) ∅)) := sorry
def condition2 : B = Set.insert 0 (Set.insert 1 ∅) := sorry
def condition3 : B ⊆ A := sorry

-- Statement
theorem largest_element_lg11 (x y : ℝ)

  (Aeq : A = Set.insert (Real.log x) (Set.insert (Real.log y) (Set.insert (Real.log (x + y / x)) ∅)))
  (Beq : B = Set.insert 0 (Set.insert 1 ∅))
  (subset : B ⊆ A) :
  ∃ M ∈ A, ∀ a ∈ A, a ≤ M ∧ M = Real.log 11 :=
sorry

end NUMINAMATH_GPT_largest_element_lg11_l416_41646


namespace NUMINAMATH_GPT_principal_amount_l416_41694

theorem principal_amount (P : ℕ) (R : ℕ) (T : ℕ) (SI : ℕ) 
  (h1 : R = 12)
  (h2 : T = 10)
  (h3 : SI = 1500) 
  (h4 : SI = (P * R * T) / 100) : P = 1250 :=
by sorry

end NUMINAMATH_GPT_principal_amount_l416_41694


namespace NUMINAMATH_GPT_shaded_area_quadrilateral_l416_41699

theorem shaded_area_quadrilateral :
  let large_square_area := 11 * 11
  let small_square_area_1 := 1 * 1
  let small_square_area_2 := 2 * 2
  let small_square_area_3 := 3 * 3
  let small_square_area_4 := 4 * 4
  let other_non_shaded_areas := 12 + 15 + 14
  let total_non_shaded := small_square_area_1 + small_square_area_2 + small_square_area_3 + small_square_area_4 + other_non_shaded_areas
  let shaded_area := large_square_area - total_non_shaded
  shaded_area = 35 := by
  sorry

end NUMINAMATH_GPT_shaded_area_quadrilateral_l416_41699


namespace NUMINAMATH_GPT_man_speed_l416_41687

theorem man_speed (time_in_minutes : ℝ) (distance_in_km : ℝ) (T : time_in_minutes = 24) (D : distance_in_km = 4) : 
  (distance_in_km / (time_in_minutes / 60)) = 10 := by
  sorry

end NUMINAMATH_GPT_man_speed_l416_41687


namespace NUMINAMATH_GPT_tom_beach_days_l416_41637

theorem tom_beach_days (total_seashells days_seashells : ℕ) (found_each_day total_found : ℕ) 
    (h1 : found_each_day = 7) (h2 : total_found = 35) : total_found / found_each_day = 5 := 
by 
  sorry

end NUMINAMATH_GPT_tom_beach_days_l416_41637


namespace NUMINAMATH_GPT_parallel_vectors_eq_l416_41668

theorem parallel_vectors_eq (x : ℝ) :
  let a := (x, 1)
  let b := (2, 4)
  (a.1 / b.1 = a.2 / b.2) → x = 1 / 2 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_parallel_vectors_eq_l416_41668


namespace NUMINAMATH_GPT_encore_songs_l416_41602

-- Definitions corresponding to the conditions
def repertoire_size : ℕ := 30
def first_set_songs : ℕ := 5
def second_set_songs : ℕ := 7
def average_songs_per_set_3_and_4 : ℕ := 8

-- The statement to prove
theorem encore_songs : (repertoire_size - (first_set_songs + second_set_songs)) - (2 * average_songs_per_set_3_and_4) = 2 := by
  sorry

end NUMINAMATH_GPT_encore_songs_l416_41602


namespace NUMINAMATH_GPT_quadratic_root_conditions_l416_41676

theorem quadratic_root_conditions (a b : ℝ)
    (h1 : ∃ k : ℝ, ∀ x : ℝ, x^2 + 2 * x + 3 - k = 0)
    (h2 : ∀ α β : ℝ, α * β = 3 - k ∧ k^2 = α * β + 3 * k) : 
    k = 3 := 
sorry

end NUMINAMATH_GPT_quadratic_root_conditions_l416_41676


namespace NUMINAMATH_GPT_value_of_s_in_base_b_l416_41626

noncomputable def b : ℕ :=
  10

def fourteen_in_b (b : ℕ) : ℕ :=
  b + 4

def seventeen_in_b (b : ℕ) : ℕ :=
  b + 7

def eighteen_in_b (b : ℕ) : ℕ :=
  b + 8

def five_thousand_four_and_four_in_b (b : ℕ) : ℕ :=
  5 * b ^ 3 + 4 * b ^ 2 + 4

def product_in_base_b_equals (b : ℕ) : Prop :=
  (fourteen_in_b b) * (seventeen_in_b b) * (eighteen_in_b b) = five_thousand_four_and_four_in_b b

def s_in_base_b (b : ℕ) : ℕ :=
  fourteen_in_b b + seventeen_in_b b + eighteen_in_b b

theorem value_of_s_in_base_b (b : ℕ) (h : product_in_base_b_equals b) : s_in_base_b b = 49 := by
  sorry

end NUMINAMATH_GPT_value_of_s_in_base_b_l416_41626


namespace NUMINAMATH_GPT_problem_part1_problem_part2_l416_41691

open Real

theorem problem_part1 (A B C : ℝ) (h1 : sin C * sin (A - B) = sin B * sin (C - A)) (h2 : A = 2 * B) : 
  C = 5 * π / 8 := 
sorry

theorem problem_part2 (a b c A B C : ℝ) (h1 : sin C * sin (A - B) = sin B * sin (C - A)) (h2 : A = 2 * B) (h3 : A + B + C = π):
  2 * a^2 = b^2 + c^2 :=
sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_l416_41691


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l416_41603

theorem geometric_sequence_common_ratio (a₁ : ℕ) (S₃ : ℕ) (q : ℤ) 
  (h₁ : a₁ = 2) (h₂ : S₃ = 6) : 
  (q = 1 ∨ q = -2) :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l416_41603


namespace NUMINAMATH_GPT_imaginary_unit_power_l416_41669

theorem imaginary_unit_power (i : ℂ) (n : ℕ) (h_i : i^2 = -1) : ∃ (n : ℕ), i^n = -1 :=
by
  use 6
  have h1 : i^4 = 1 := by sorry  -- Need to show i^4 = 1
  have h2 : i^6 = -1 := by sorry  -- Use i^4 and additional steps to show i^6 = -1
  exact h2

end NUMINAMATH_GPT_imaginary_unit_power_l416_41669


namespace NUMINAMATH_GPT_rectangle_length_width_ratio_l416_41615

-- Define the side lengths of the small squares and the large square
variables (s : ℝ)

-- Define the dimensions of the large square and the rectangle
def large_square_side : ℝ := 5 * s
def rectangle_length : ℝ := 5 * s
def rectangle_width : ℝ := s

-- State and prove the theorem
theorem rectangle_length_width_ratio : rectangle_length s / rectangle_width s = 5 :=
by sorry

end NUMINAMATH_GPT_rectangle_length_width_ratio_l416_41615


namespace NUMINAMATH_GPT_trebled_result_of_original_number_is_72_l416_41670

theorem trebled_result_of_original_number_is_72:
  ∀ (x : ℕ), x = 9 → 3 * (2 * x + 6) = 72 :=
by
  intro x h
  sorry

end NUMINAMATH_GPT_trebled_result_of_original_number_is_72_l416_41670


namespace NUMINAMATH_GPT_parabola_intersection_points_l416_41618

theorem parabola_intersection_points :
  let parabola1 := λ x : ℝ => 4*x^2 + 3*x - 1
  let parabola2 := λ x : ℝ => x^2 + 8*x + 7
  ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ = -4/3 ∧ y₁ = -17/9 ∧
                        x₂ = 2 ∧ y₂ = 27 ∧
                        parabola1 x₁ = y₁ ∧ 
                        parabola2 x₁ = y₁ ∧
                        parabola1 x₂ = y₂ ∧
                        parabola2 x₂ = y₂ :=
by {
  sorry
}

end NUMINAMATH_GPT_parabola_intersection_points_l416_41618


namespace NUMINAMATH_GPT_value_of_f_at_6_l416_41621

variable {R : Type*} [LinearOrderedField R]

noncomputable def f : R → R := sorry

-- Conditions
axiom odd_function (x : R) : f (-x) = -f x
axiom periodicity (x : R) : f (x + 2) = -f x

-- Theorem to prove
theorem value_of_f_at_6 : f 6 = 0 := by sorry

end NUMINAMATH_GPT_value_of_f_at_6_l416_41621


namespace NUMINAMATH_GPT_inequality_holds_l416_41650

variables {a b c : ℝ}

theorem inequality_holds (h1 : c < b) (h2 : b < a) (h3 : ac < 0) : ab > ac :=
sorry

end NUMINAMATH_GPT_inequality_holds_l416_41650


namespace NUMINAMATH_GPT_symmetry_x_y_axis_symmetry_line_y_neg1_l416_41605

-- Define point P
structure Point :=
  (x : ℝ)
  (y : ℝ)

def P : Point := { x := 1, y := 2 }

-- Condition for symmetry with respect to x-axis
def symmetric_x (p : Point) : Point :=
  { x := p.x, y := -p.y }

-- Condition for symmetry with respect to the line y = -1
def symmetric_line_y_neg1 (p : Point) : Point :=
  { x := p.x, y := 2 * 1 - p.y - 1 }

-- Theorem statements
theorem symmetry_x_y_axis : symmetric_x P = { x := 1, y := -2 } := sorry
theorem symmetry_line_y_neg1 : symmetric_line_y_neg1 { x := 1, y := -2 } = { x := 1, y := 3 } := sorry

end NUMINAMATH_GPT_symmetry_x_y_axis_symmetry_line_y_neg1_l416_41605


namespace NUMINAMATH_GPT_is_not_prime_390629_l416_41645

theorem is_not_prime_390629 : ¬ Prime 390629 :=
sorry

end NUMINAMATH_GPT_is_not_prime_390629_l416_41645


namespace NUMINAMATH_GPT_parallel_lines_condition_l416_41695

theorem parallel_lines_condition (a : ℝ) :
  (a = 3 / 2) ↔ (∀ x y : ℝ, (x + 2 * a * y - 1 = 0 → (a - 1) * x + a * y + 1 = 0) → (a = 3 / 2)) :=
sorry

end NUMINAMATH_GPT_parallel_lines_condition_l416_41695


namespace NUMINAMATH_GPT_find_certain_number_l416_41655

noncomputable def certain_number_is_square (n : ℕ) (x : ℕ) : Prop :=
  ∃ (y : ℕ), x * n = y * y

theorem find_certain_number : ∃ x, certain_number_is_square 3 x :=
by 
  use 1
  unfold certain_number_is_square
  use 3
  sorry

end NUMINAMATH_GPT_find_certain_number_l416_41655


namespace NUMINAMATH_GPT_investment2_rate_l416_41617

-- Define the initial conditions
def total_investment : ℝ := 10000
def investment1 : ℝ := 4000
def rate1 : ℝ := 0.05
def investment2 : ℝ := 3500
def income1 : ℝ := investment1 * rate1
def yearly_income_goal : ℝ := 500
def remaining_investment : ℝ := total_investment - investment1 - investment2
def rate3 : ℝ := 0.064
def income3 : ℝ := remaining_investment * rate3

-- The main theorem
theorem investment2_rate (rate2 : ℝ) : 
  income1 + income3 + investment2 * (rate2 / 100) = yearly_income_goal → rate2 = 4 := 
by 
  sorry

end NUMINAMATH_GPT_investment2_rate_l416_41617


namespace NUMINAMATH_GPT_order_of_reading_amounts_l416_41644

variable (a b c d : ℝ)

theorem order_of_reading_amounts (h1 : a + c = b + d) (h2 : a + b > c + d) (h3 : d > b + c) :
  a > d ∧ d > b ∧ b > c :=
by
  sorry

end NUMINAMATH_GPT_order_of_reading_amounts_l416_41644


namespace NUMINAMATH_GPT_solve_ab_cd_l416_41634

theorem solve_ab_cd (a b c d : ℝ) 
  (h1 : a + b + c = 3) 
  (h2 : a + b + d = -2) 
  (h3 : a + c + d = 5) 
  (h4 : b + c + d = 4) 
  : a * b + c * d = 26 / 9 := 
by {
  sorry
}

end NUMINAMATH_GPT_solve_ab_cd_l416_41634


namespace NUMINAMATH_GPT_lcm_48_180_l416_41662

theorem lcm_48_180 : Nat.lcm 48 180 = 720 :=
by 
  sorry

end NUMINAMATH_GPT_lcm_48_180_l416_41662


namespace NUMINAMATH_GPT_polygon_area_is_correct_l416_41679

def points : List (ℕ × ℕ) := [
  (0, 0), (10, 0), (20, 0), (30, 10),
  (0, 20), (10, 20), (20, 30), (10, 30),
  (0, 30), (20, 10), (30, 20), (10, 10)
]

def polygon_area (ps : List (ℕ × ℕ)) : ℕ := sorry

theorem polygon_area_is_correct :
  polygon_area points = 9 := sorry

end NUMINAMATH_GPT_polygon_area_is_correct_l416_41679


namespace NUMINAMATH_GPT_parabola_coordinates_l416_41692

theorem parabola_coordinates (x y : ℝ) (h_parabola : y^2 = 4 * x) (h_distance : (x - 1)^2 + y^2 = 100) :
  (x = 9 ∧ y = 6) ∨ (x = 9 ∧ y = -6) :=
by
  sorry

end NUMINAMATH_GPT_parabola_coordinates_l416_41692


namespace NUMINAMATH_GPT_gear_angular_speeds_ratio_l416_41632

noncomputable def gear_ratio (x y z w : ℕ) (ω_A ω_B ω_C ω_D : ℝ) :=
  x * ω_A = y * ω_B ∧ y * ω_B = z * ω_C ∧ z * ω_C = w * ω_D

theorem gear_angular_speeds_ratio (x y z w : ℕ) (ω_A ω_B ω_C ω_D : ℝ) 
  (h : gear_ratio x y z w ω_A ω_B ω_C ω_D) :
  ω_A / ω_B = y / x ∧ ω_B / ω_C = z / y ∧ ω_C / ω_D = w / z :=
by sorry

end NUMINAMATH_GPT_gear_angular_speeds_ratio_l416_41632


namespace NUMINAMATH_GPT_remainder_of_product_mod_5_l416_41612

theorem remainder_of_product_mod_5 : 
  (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 5 = 4 := by
  sorry

end NUMINAMATH_GPT_remainder_of_product_mod_5_l416_41612


namespace NUMINAMATH_GPT_pizza_shared_cost_l416_41611

theorem pizza_shared_cost (total_price : ℕ) (num_people : ℕ) (share: ℕ)
  (h1 : total_price = 40) (h2 : num_people = 5) : share = 8 :=
by
  sorry

end NUMINAMATH_GPT_pizza_shared_cost_l416_41611


namespace NUMINAMATH_GPT_converse_statement_2_true_implies_option_A_l416_41673

theorem converse_statement_2_true_implies_option_A :
  (∀ x : ℕ, x = 1 ∨ x = 2 → (x^2 - 3 * x + 2 = 0)) →
  (x = 1 ∨ x = 2) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_converse_statement_2_true_implies_option_A_l416_41673


namespace NUMINAMATH_GPT_sum_of_three_squares_l416_41664

theorem sum_of_three_squares (n : ℕ) (h : n = 100) : 
  ∃ (a b c : ℕ), a = 4 ∧ b^2 + c^2 = 84 ∧ a^2 + b^2 + c^2 = 100 ∧ 
  (a ≠ b ∧ b ≠ c ∧ a ≠ c ∨ (b = c ∧ a ≠ b)) ∧
  (4^2 + 7^2 + 6^2 = 100 ∧ 4^2 + 8^2 + 5^2 = 100 ∧ 4^2 + 9^2 + 1^2 = 100) ∧
  (4^2 + 6^2 + 7^2 ≠ 100 ∧ 4^2 + 5^2 + 8^2 ≠ 100 ∧ 4^2 + 1^2 + 9^2 ≠ 100 ∧ 
   4^2 + 4^2 + 8^2 ≠ 100 ∨ 4^2 + 8^2 + 4^2 ≠ 100) :=
sorry

end NUMINAMATH_GPT_sum_of_three_squares_l416_41664


namespace NUMINAMATH_GPT_find_x_collinear_l416_41619

theorem find_x_collinear (x : ℝ) (a b : ℝ × ℝ) (h_a : a = (2, 1)) (h_b : b = (x, -1)) 
  (h_collinear : ∃ k : ℝ, (a.1 - b.1, a.2 - b.2) = (k * b.1, k * b.2)) : x = -2 :=
by 
  -- the proof would go here
  sorry

end NUMINAMATH_GPT_find_x_collinear_l416_41619


namespace NUMINAMATH_GPT_Karl_max_score_l416_41649

def max_possible_score : ℕ :=
  69

theorem Karl_max_score (minutes problems : ℕ) (n_points : ℕ → ℕ) (time_1_5 : ℕ) (time_6_10 : ℕ) (time_11_15 : ℕ)
    (h1 : minutes = 15) (h2 : problems = 15)
    (h3 : ∀ n, n = n_points n)
    (h4 : ∀ i, 1 ≤ i ∧ i ≤ 5 → time_1_5 = 1)
    (h5 : ∀ i, 6 ≤ i ∧ i ≤ 10 → time_6_10 = 2)
    (h6 : ∀ i, 11 ≤ i ∧ i ≤ 15 → time_11_15 = 3) : 
    max_possible_score = 69 :=
  by
  sorry

end NUMINAMATH_GPT_Karl_max_score_l416_41649


namespace NUMINAMATH_GPT_socks_count_l416_41684

theorem socks_count :
  ∃ (x y z : ℕ), x + y + z = 12 ∧ x + 3 * y + 4 * z = 24 ∧ 1 ≤ x ∧ 1 ≤ y ∧ 1 <= z ∧ x = 7 :=
by
  sorry

end NUMINAMATH_GPT_socks_count_l416_41684


namespace NUMINAMATH_GPT_moving_circle_fixed_point_l416_41671

def parabola (p : ℝ × ℝ) : Prop := p.2^2 = 4 * p.1

def tangent_line (c : ℝ × ℝ) (r : ℝ) : Prop :=
  abs (c.1 + 1) = r

theorem moving_circle_fixed_point :
  ∀ (c : ℝ × ℝ) (r : ℝ),
    parabola c →
    tangent_line c r →
    (1, 0) ∈ {p : ℝ × ℝ | dist c p = r} :=
by
  intro c r hc ht
  sorry

end NUMINAMATH_GPT_moving_circle_fixed_point_l416_41671


namespace NUMINAMATH_GPT_sum_of_coefficients_l416_41663

theorem sum_of_coefficients (a a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 : ℝ) :
  (∀ x, (x^2 + 1) * (x - 2)^9 = a + a1 * (x - 1) + a2 * (x - 1)^2 + a3 * (x - 1)^3 + a4 * (x - 1)^4 +
        a5 * (x - 1)^5 + a6 * (x - 1)^6 + a7 * (x - 1)^7 + a8 * (x - 1)^8 + a9 * (x - 1)^9 + a10 * (x - 1)^10 + a11 * (x - 1)^11) →
  a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 = 2 := 
sorry

end NUMINAMATH_GPT_sum_of_coefficients_l416_41663


namespace NUMINAMATH_GPT_find_whole_number_M_l416_41652

-- Define the conditions
def condition (M : ℕ) : Prop :=
  21 < M ∧ M < 23

-- Define the main theorem to be proven
theorem find_whole_number_M (M : ℕ) (h : condition M) : M = 22 := by
  sorry

end NUMINAMATH_GPT_find_whole_number_M_l416_41652


namespace NUMINAMATH_GPT_problem_statement_l416_41620

noncomputable def floor_T (u v w x : ℝ) : ℤ :=
  ⌊u + v + w + x⌋

theorem problem_statement (u v w x : ℝ) (T : ℝ) (h₁: u^2 + v^2 = 3005) (h₂: w^2 + x^2 = 3005) (h₃: u * w = 1729) (h₄: v * x = 1729) :
  floor_T u v w x = 155 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l416_41620


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l416_41674

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ) (b : ℕ → ℚ) (T : ℕ → ℚ) :
  (a 3 = 7) ∧ (a 5 + a 7 = 26) →
  (∀ n, a n = 2 * n + 1) ∧
  (∀ n, S n = n^2 + 2 * n) ∧
  (∀ n, b n = 1 / ((a n)^2 - 1)) →
  (∀ n, T n = n / (4 * (n + 1))) := sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l416_41674


namespace NUMINAMATH_GPT_quadratic_solution_l416_41639

theorem quadratic_solution (x : ℝ) :
  (x^2 + 2 * x = 0) ↔ (x = 0 ∨ x = -2) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_solution_l416_41639


namespace NUMINAMATH_GPT_sin_C_of_arith_prog_angles_l416_41608

theorem sin_C_of_arith_prog_angles (A B C a b : ℝ) (h_abc : A + B + C = Real.pi)
  (h_arith_prog : 2 * B = A + C) (h_a : a = Real.sqrt 2) (h_b : b = Real.sqrt 3) :
  Real.sin C = (Real.sqrt 2 + Real.sqrt 6) / 4 :=
sorry

end NUMINAMATH_GPT_sin_C_of_arith_prog_angles_l416_41608


namespace NUMINAMATH_GPT_total_balloons_cost_is_91_l416_41607

-- Define the number of balloons and their costs for Fred, Sam, and Dan
def fred_balloons : ℕ := 10
def fred_cost_per_balloon : ℝ := 1

def sam_balloons : ℕ := 46
def sam_cost_per_balloon : ℝ := 1.5

def dan_balloons : ℕ := 16
def dan_cost_per_balloon : ℝ := 0.75

-- Calculate the total cost for each person’s balloons
def fred_total_cost : ℝ := fred_balloons * fred_cost_per_balloon
def sam_total_cost : ℝ := sam_balloons * sam_cost_per_balloon
def dan_total_cost : ℝ := dan_balloons * dan_cost_per_balloon

-- Calculate the total cost of all the balloons combined
def total_cost : ℝ := fred_total_cost + sam_total_cost + dan_total_cost

-- The main statement to be proved
theorem total_balloons_cost_is_91 : total_cost = 91 :=
by
  -- Recall that the previous individual costs can be worked out and added
  -- But for the sake of this statement, we use sorry to skip details
  sorry

end NUMINAMATH_GPT_total_balloons_cost_is_91_l416_41607


namespace NUMINAMATH_GPT_percent_increase_l416_41686

variable (P : ℝ)
def firstQuarterPrice := 1.20 * P
def secondQuarterPrice := 1.50 * P

theorem percent_increase:
  ((secondQuarterPrice P - firstQuarterPrice P) / firstQuarterPrice P) * 100 = 25 := by
  sorry

end NUMINAMATH_GPT_percent_increase_l416_41686


namespace NUMINAMATH_GPT_geese_count_l416_41653

theorem geese_count (initial : ℕ) (flown_away : ℕ) (left : ℕ) 
  (h₁ : initial = 51) (h₂ : flown_away = 28) : 
  left = initial - flown_away → left = 23 := 
by
  sorry

end NUMINAMATH_GPT_geese_count_l416_41653


namespace NUMINAMATH_GPT_initial_plants_count_l416_41623

theorem initial_plants_count (p : ℕ) 
    (h1 : p - 20 > 0)
    (h2 : (p - 20) / 2 > 0)
    (h3 : ((p - 20) / 2) - 1 > 0)
    (h4 : ((p - 20) / 2) - 1 = 4) : 
    p = 30 :=
by
  sorry

end NUMINAMATH_GPT_initial_plants_count_l416_41623


namespace NUMINAMATH_GPT_largest_angle_in_pentagon_l416_41665

theorem largest_angle_in_pentagon (P Q R S T : ℝ) 
          (h1 : P = 70) 
          (h2 : Q = 100)
          (h3 : R = S) 
          (h4 : T = 3 * R - 25)
          (h5 : P + Q + R + S + T = 540) : 
          T = 212 :=
by
  sorry

end NUMINAMATH_GPT_largest_angle_in_pentagon_l416_41665


namespace NUMINAMATH_GPT_log_relation_l416_41601

theorem log_relation (a b c: ℝ) (h₁: a = (Real.log 2) / 2) (h₂: b = (Real.log 3) / 3) (h₃: c = (Real.log 5) / 5) : c < a ∧ a < b :=
by
  sorry

end NUMINAMATH_GPT_log_relation_l416_41601


namespace NUMINAMATH_GPT_second_piece_weight_l416_41606

theorem second_piece_weight (w1 : ℝ) (s1 : ℝ) (s2 : ℝ) (w2 : ℝ) :
  (s1 = 4) → (w1 = 16) → (s2 = 6) → w2 = w1 * (s2^2 / s1^2) → w2 = 36 :=
by
  intro h_s1 h_w1 h_s2 h_w2
  rw [h_s1, h_w1, h_s2] at h_w2
  norm_num at h_w2
  exact h_w2

end NUMINAMATH_GPT_second_piece_weight_l416_41606


namespace NUMINAMATH_GPT_total_cost_proof_l416_41622

-- Definitions for the problem conditions
def basketball_cost : ℕ := 48
def volleyball_cost : ℕ := basketball_cost - 18
def basketball_quantity : ℕ := 3
def volleyball_quantity : ℕ := 5
def total_basketball_cost : ℕ := basketball_cost * basketball_quantity
def total_volleyball_cost : ℕ := volleyball_cost * volleyball_quantity
def total_cost : ℕ := total_basketball_cost + total_volleyball_cost

-- Theorem to be proved
theorem total_cost_proof : total_cost = 294 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_proof_l416_41622


namespace NUMINAMATH_GPT_age_problem_l416_41609

theorem age_problem (age x : ℕ) (h : age = 64) :
  (1 / 2 : ℝ) * (8 * (age + x) - 8 * (age - 8)) = age → x = 8 :=
by
  sorry

end NUMINAMATH_GPT_age_problem_l416_41609


namespace NUMINAMATH_GPT_sqrt_4_eq_pm2_l416_41696

theorem sqrt_4_eq_pm2 : {y : ℝ | y^2 = 4} = {2, -2} :=
by
  sorry

end NUMINAMATH_GPT_sqrt_4_eq_pm2_l416_41696


namespace NUMINAMATH_GPT_grace_apples_after_6_weeks_l416_41625

def apples_per_day_bella : ℕ := 6

def days_per_week : ℕ := 7

def fraction_apples_bella_consumes : ℚ := 1/3

def weeks : ℕ := 6

theorem grace_apples_after_6_weeks :
  let apples_per_week_bella := apples_per_day_bella * days_per_week
  let apples_per_week_grace := apples_per_week_bella / fraction_apples_bella_consumes
  let remaining_apples_week := apples_per_week_grace - apples_per_week_bella
  let total_apples := remaining_apples_week * weeks
  total_apples = 504 := by
  sorry

end NUMINAMATH_GPT_grace_apples_after_6_weeks_l416_41625


namespace NUMINAMATH_GPT_sqrt_meaningful_range_l416_41635

theorem sqrt_meaningful_range (x : ℝ) : 
  (x + 4) ≥ 0 ↔ x ≥ -4 :=
by sorry

end NUMINAMATH_GPT_sqrt_meaningful_range_l416_41635


namespace NUMINAMATH_GPT_not_divisible_by_3_l416_41690

theorem not_divisible_by_3 (n : ℤ) : (n^2 + 1) % 3 ≠ 0 := by
  sorry

end NUMINAMATH_GPT_not_divisible_by_3_l416_41690


namespace NUMINAMATH_GPT_range_of_a_l416_41660

/-- 
For the system of inequalities in terms of x 
    \begin{cases} 
    x - a < 0 
    ax < 1 
    \end{cases}
the range of values for the real number a such that the solution set is not empty is [-1, ∞).
-/
theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x - a < 0 ∧ a * x < 1) ↔ -1 ≤ a :=
by sorry

end NUMINAMATH_GPT_range_of_a_l416_41660


namespace NUMINAMATH_GPT_sid_money_left_after_purchases_l416_41600

theorem sid_money_left_after_purchases : 
  ∀ (original_money money_spent_on_computer money_spent_on_snacks half_of_original_money money_left final_more_than_half),
  original_money = 48 → 
  money_spent_on_computer = 12 → 
  money_spent_on_snacks = 8 →
  half_of_original_money = original_money / 2 → 
  money_left = original_money - (money_spent_on_computer + money_spent_on_snacks) → 
  final_more_than_half = money_left - half_of_original_money →
  final_more_than_half = 4 := 
by
  intros original_money money_spent_on_computer money_spent_on_snacks half_of_original_money money_left final_more_than_half
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_sid_money_left_after_purchases_l416_41600


namespace NUMINAMATH_GPT_rhombus_area_eq_54_l416_41636

theorem rhombus_area_eq_54
  (a b : ℝ) (eq_long_side : a = 4 * Real.sqrt 3) (eq_short_side : b = 3 * Real.sqrt 3)
  (rhombus_diagonal1 : ℝ := 9 * Real.sqrt 3) (rhombus_diagonal2 : ℝ := 4 * Real.sqrt 3) :
  (1 / 2) * rhombus_diagonal1 * rhombus_diagonal2 = 54 := by
  sorry

end NUMINAMATH_GPT_rhombus_area_eq_54_l416_41636


namespace NUMINAMATH_GPT_expected_value_winnings_l416_41648

def probability_heads : ℚ := 2 / 5
def probability_tails : ℚ := 3 / 5
def winnings_heads : ℚ := 4
def loss_tails : ℚ := -3

theorem expected_value_winnings : 
  (probability_heads * winnings_heads + probability_tails * loss_tails) = -1 / 5 := 
by
  -- calculation steps and proof would go here
  sorry

end NUMINAMATH_GPT_expected_value_winnings_l416_41648


namespace NUMINAMATH_GPT_tom_seashells_found_l416_41654

/-- 
Given:
- sally_seashells = 9 (number of seashells Sally found)
- jessica_seashells = 5 (number of seashells Jessica found)
- total_seashells = 21 (number of seashells found together)

Prove that the number of seashells that Tom found (tom_seashells) is 7.
-/
theorem tom_seashells_found (sally_seashells jessica_seashells total_seashells tom_seashells : ℕ)
  (h₁ : sally_seashells = 9) (h₂ : jessica_seashells = 5) (h₃ : total_seashells = 21) :
  tom_seashells = 7 :=
by
  sorry

end NUMINAMATH_GPT_tom_seashells_found_l416_41654


namespace NUMINAMATH_GPT_total_practice_hours_correct_l416_41682

-- Define the conditions
def daily_practice_hours : ℕ := 5 -- The team practices 5 hours daily
def missed_days : ℕ := 1 -- They missed practicing 1 day this week
def days_in_week : ℕ := 7 -- There are 7 days in a week

-- Calculate the number of days they practiced
def practiced_days : ℕ := days_in_week - missed_days

-- Calculate the total hours practiced
def total_practice_hours : ℕ := practiced_days * daily_practice_hours

-- Theorem to prove the total hours practiced is 30
theorem total_practice_hours_correct : total_practice_hours = 30 := by
  -- Start the proof; skipping the actual proof steps
  sorry

end NUMINAMATH_GPT_total_practice_hours_correct_l416_41682


namespace NUMINAMATH_GPT_infection_equation_l416_41610

-- Given conditions
def initially_infected : Nat := 1
def total_after_two_rounds : ℕ := 81
def avg_infect_per_round (x : ℕ) : ℕ := x

-- Mathematically equivalent proof problem
theorem infection_equation (x : ℕ) 
  (h1 : initially_infected = 1)
  (h2 : total_after_two_rounds = 81)
  (h3 : ∀ (y : ℕ), initially_infected + avg_infect_per_round y + (avg_infect_per_round y)^2 = total_after_two_rounds):
  (1 + x)^2 = 81 :=
by
  sorry

end NUMINAMATH_GPT_infection_equation_l416_41610


namespace NUMINAMATH_GPT_garden_width_l416_41689

theorem garden_width (L W : ℕ) 
  (area_playground : 192 = 16 * 12)
  (area_garden : 192 = L * W)
  (perimeter_garden : 64 = 2 * L + 2 * W) :
  W = 12 :=
by
  sorry

end NUMINAMATH_GPT_garden_width_l416_41689


namespace NUMINAMATH_GPT_find_fraction_l416_41627

-- Variables and Definitions
variables (x : ℚ)

-- Conditions
def condition1 := (2 / 3) / x = (3 / 5) / (7 / 15)

-- Theorem to prove the certain fraction
theorem find_fraction (h : condition1 x) : x = 14 / 27 :=
by sorry

end NUMINAMATH_GPT_find_fraction_l416_41627


namespace NUMINAMATH_GPT_area_relation_l416_41630

-- Define the areas of the triangles
variables (a b c : ℝ)

-- Define the condition that triangles T_a and T_c are similar (i.e., homothetic)
-- which implies the relationship between their areas.
theorem area_relation (ha : 0 < a) (hc : 0 < c) (habc : b = Real.sqrt (a * c)) : b = Real.sqrt (a * c) := by
  sorry

end NUMINAMATH_GPT_area_relation_l416_41630


namespace NUMINAMATH_GPT_problem_statement_l416_41629

variable {a b c d k : ℝ}

theorem problem_statement (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
    (h_pos : 0 < k)
    (h_sum_ab : a + b = k)
    (h_sum_cd : c + d = k^2)
    (h_roots1 : ∀ x, x^2 - 4*a*x - 5*b = 0 → x = c ∨ x = d)
    (h_roots2 : ∀ x, x^2 - 4*c*x - 5*d = 0 → x = a ∨ x = b) : 
    a + b + c + d = k + k^2 :=
sorry

end NUMINAMATH_GPT_problem_statement_l416_41629


namespace NUMINAMATH_GPT_arrange_banana_l416_41666

theorem arrange_banana : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let permutations := Nat.factorial total_letters
  let adjust_A := Nat.factorial count_A
  let adjust_N := Nat.factorial count_N
  permutations / (adjust_A * adjust_N) = 60 := by
  -- proof steps will be filled in here
  sorry

end NUMINAMATH_GPT_arrange_banana_l416_41666


namespace NUMINAMATH_GPT_closest_whole_number_l416_41656

theorem closest_whole_number :
  let x := (10^2001 + 10^2003) / (10^2002 + 10^2002)
  abs ((x : ℝ) - 5) < 1 :=
by 
  sorry

end NUMINAMATH_GPT_closest_whole_number_l416_41656


namespace NUMINAMATH_GPT_lunch_to_read_ratio_l416_41683

theorem lunch_to_read_ratio 
  (total_pages : ℕ) (pages_per_hour : ℕ) (lunch_hours : ℕ)
  (h₁ : total_pages = 4000)
  (h₂ : pages_per_hour = 250)
  (h₃ : lunch_hours = 4) :
  lunch_hours / (total_pages / pages_per_hour) = 1 / 4 := by
  sorry

end NUMINAMATH_GPT_lunch_to_read_ratio_l416_41683


namespace NUMINAMATH_GPT_possible_amounts_l416_41681

theorem possible_amounts (n : ℕ) : 
  ¬ (∃ x y : ℕ, 3 * x + 5 * y = n) ↔ n = 1 ∨ n = 2 ∨ n = 4 ∨ n = 7 :=
sorry

end NUMINAMATH_GPT_possible_amounts_l416_41681


namespace NUMINAMATH_GPT_calculate_expression_l416_41677

theorem calculate_expression :
  let a := 2^4
  let b := 2^2
  let c := 2^3
  (a^2 / b^3) * c^3 = 2048 :=
by
  sorry -- Proof is omitted as per instructions

end NUMINAMATH_GPT_calculate_expression_l416_41677


namespace NUMINAMATH_GPT_sequence_general_formula_l416_41640

theorem sequence_general_formula (a : ℕ → ℕ)
    (h1 : a 1 = 3) 
    (h2 : a 2 = 4) 
    (h3 : a 3 = 6) 
    (h4 : a 4 = 10) 
    (h5 : a 5 = 18) :
    ∀ n : ℕ, a n = 2^(n-1) + 2 :=
sorry

end NUMINAMATH_GPT_sequence_general_formula_l416_41640


namespace NUMINAMATH_GPT_smallest_integer_min_value_l416_41697

theorem smallest_integer_min_value :
  ∃ (A B C D : ℕ), 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ 
  B ≠ C ∧ B ≠ D ∧ 
  C ≠ D ∧ 
  (A + B + C + D) = 288 ∧ 
  D = 90 ∧ 
  (A = 21) := 
sorry

end NUMINAMATH_GPT_smallest_integer_min_value_l416_41697


namespace NUMINAMATH_GPT_simplify_polynomial_l416_41631

theorem simplify_polynomial (x : ℝ) (A B C D : ℝ) :
  (y = (x^3 + 12 * x^2 + 47 * x + 60) / (x + 3)) →
  (y = A * x^2 + B * x + C) →
  x ≠ D →
  A = 1 ∧ B = 9 ∧ C = 20 ∧ D = -3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_polynomial_l416_41631
