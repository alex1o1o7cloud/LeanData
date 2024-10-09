import Mathlib

namespace no_real_roots_iff_l387_38706

noncomputable def f (x a : ℝ) : ℝ := x^2 + 2*x + a

theorem no_real_roots_iff (a : ℝ) : (∀ x : ℝ, f x a ≠ 0) → a > 1 :=
  by
    sorry

end no_real_roots_iff_l387_38706


namespace find_value_of_m_and_n_l387_38710

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x^3 + 3*x^2 + m * x
noncomputable def g (x : ℝ) (n : ℝ) : ℝ := Real.log (x + 1) + n * x

theorem find_value_of_m_and_n (m n : ℝ) (h₀ : n > 0) 
  (h₁ : f (-1) m = -1) 
  (h₂ : ∀ x : ℝ, f x m = g x n → x = 0) :
  m + n = 5 := 
by 
  sorry

end find_value_of_m_and_n_l387_38710


namespace sequence_fifth_term_l387_38728

theorem sequence_fifth_term (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : a 2 = 2)
    (h₃ : ∀ n > 2, a n = a (n-1) + a (n-2)) : a 5 = 8 :=
sorry

end sequence_fifth_term_l387_38728


namespace cost_of_one_unit_each_l387_38767

variables (x y z : ℝ)

theorem cost_of_one_unit_each
  (h1 : 2 * x + 3 * y + z = 130)
  (h2 : 3 * x + 5 * y + z = 205) :
  x + y + z = 55 :=
by
  sorry

end cost_of_one_unit_each_l387_38767


namespace expected_value_of_N_l387_38788

noncomputable def expected_value_N : ℝ :=
  30

theorem expected_value_of_N :
  -- Suppose Bob chooses a 4-digit binary string uniformly at random,
  -- and examines an infinite sequence of independent random binary bits.
  -- Let N be the least number of bits Bob has to examine to find his chosen string.
  -- Then the expected value of N is 30.
  expected_value_N = 30 :=
by
  sorry

end expected_value_of_N_l387_38788


namespace ellipse_standard_equation_l387_38744

theorem ellipse_standard_equation
  (F : ℝ × ℝ)
  (e : ℝ)
  (eq1 : F = (0, 1))
  (eq2 : e = 1 / 2) :
  ∃ (a b : ℝ), a = 2 ∧ b ^ 2 = 3 ∧ (∀ x y : ℝ, (y ^ 2 / 4) + (x ^ 2 / 3) = 1) :=
by
  sorry

end ellipse_standard_equation_l387_38744


namespace both_participation_correct_l387_38781

-- Define the number of total participants
def total_participants : ℕ := 50

-- Define the number of participants in Chinese competition
def chinese_participants : ℕ := 30

-- Define the number of participants in Mathematics competition
def math_participants : ℕ := 38

-- Define the number of people who do not participate in either competition
def neither_participants : ℕ := 2

-- Define the number of people who participate in both competitions
def both_participants : ℕ :=
  chinese_participants + math_participants - (total_participants - neither_participants)

-- The theorem we want to prove
theorem both_participation_correct : both_participants = 20 :=
by
  sorry

end both_participation_correct_l387_38781


namespace possible_n_values_l387_38719

theorem possible_n_values (x y z n : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
    x^3 + y^3 + z^3 = n * x^2 * y^2 * z^2 → n = 1 ∨ n = 3 :=
by 
  sorry

end possible_n_values_l387_38719


namespace max_value_of_k_l387_38786

theorem max_value_of_k (m : ℝ) (k : ℝ) (h1 : 0 < m) (h2 : m < 1/2) 
  (h3 : ∀ m, 0 < m → m < 1/2 → (1 / m + 2 / (1 - 2 * m) ≥ k)) : k = 8 :=
sorry

end max_value_of_k_l387_38786


namespace roger_cookie_price_l387_38778

open Classical

theorem roger_cookie_price
  (art_base1 art_base2 art_height : ℕ) 
  (art_cookies_per_batch art_cookie_price roger_cookies_per_batch : ℕ)
  (art_area : ℕ := (art_base1 + art_base2) * art_height / 2)
  (total_dough : ℕ := art_cookies_per_batch * art_area)
  (roger_area : ℚ := total_dough / roger_cookies_per_batch)
  (art_total_earnings : ℚ := art_cookies_per_batch * art_cookie_price) :
  ∀ (roger_cookie_price : ℚ), roger_cookies_per_batch * roger_cookie_price = art_total_earnings →
  roger_cookie_price = 100 / 3 :=
sorry

end roger_cookie_price_l387_38778


namespace floor_sub_y_eq_zero_l387_38791

theorem floor_sub_y_eq_zero {y : ℝ} (h : ⌊y⌋ + ⌈y⌉ = 2 * y) : ⌊y⌋ - y = 0 :=
sorry

end floor_sub_y_eq_zero_l387_38791


namespace arithmetic_sequence_sum_mod_l387_38720

theorem arithmetic_sequence_sum_mod (a d l k S n : ℕ) 
  (h_seq_start : a = 3)
  (h_common_difference : d = 5)
  (h_last_term : l = 103)
  (h_sum_formula : S = (k * (3 + 103)) / 2)
  (h_term_count : k = 21)
  (h_mod_condition : 1113 % 17 = n)
  (h_range_condition : 0 ≤ n ∧ n < 17) : 
  n = 8 :=
by
  sorry

end arithmetic_sequence_sum_mod_l387_38720


namespace percentage_reduction_is_correct_l387_38735

def percentage_reduction_alcohol_concentration (V_original V_added : ℚ) (C_original : ℚ) : ℚ :=
  let V_total := V_original + V_added
  let Amount_alcohol := V_original * C_original
  let C_new := Amount_alcohol / V_total
  ((C_original - C_new) / C_original) * 100

theorem percentage_reduction_is_correct :
  percentage_reduction_alcohol_concentration 12 28 0.20 = 70 := by
  sorry

end percentage_reduction_is_correct_l387_38735


namespace compare_squares_l387_38704

theorem compare_squares (a b c : ℝ) : a^2 + b^2 + c^2 ≥ ab + bc + ca :=
by
  sorry

end compare_squares_l387_38704


namespace compare_2_pow_n_n_sq_l387_38783

theorem compare_2_pow_n_n_sq (n : ℕ) (h : n > 0) :
  (n = 1 → 2^n > n^2) ∧
  (n = 2 → 2^n = n^2) ∧
  (n = 3 → 2^n < n^2) ∧
  (n = 4 → 2^n = n^2) ∧
  (n ≥ 5 → 2^n > n^2) :=
by sorry

end compare_2_pow_n_n_sq_l387_38783


namespace quadratic_no_real_roots_l387_38725

theorem quadratic_no_real_roots (c : ℝ) : (∀ x : ℝ, x^2 + 2 * x + c ≠ 0) → c > 1 :=
by
  sorry

end quadratic_no_real_roots_l387_38725


namespace rice_mixture_ratio_l387_38741

theorem rice_mixture_ratio (x y z : ℕ) (h : 16 * x + 24 * y + 30 * z = 18 * (x + y + z)) : 
  x = 9 * y + 18 * z :=
by
  sorry

end rice_mixture_ratio_l387_38741


namespace chipmunk_families_left_l387_38795

theorem chipmunk_families_left (orig : ℕ) (left : ℕ) (h1 : orig = 86) (h2 : left = 65) : orig - left = 21 := by
  sorry

end chipmunk_families_left_l387_38795


namespace double_24_times_10_pow_8_l387_38798

theorem double_24_times_10_pow_8 : 2 * (2.4 * 10^8) = 4.8 * 10^8 :=
by
  sorry

end double_24_times_10_pow_8_l387_38798


namespace cedar_vs_pine_height_cedar_vs_birch_height_l387_38742

-- Define the heights as rational numbers
def pine_tree_height := 14 + 1/4
def birch_tree_height := 18 + 1/2
def cedar_tree_height := 20 + 5/8

-- Theorem to prove the height differences
theorem cedar_vs_pine_height :
  cedar_tree_height - pine_tree_height = 6 + 3/8 :=
by
  sorry

theorem cedar_vs_birch_height :
  cedar_tree_height - birch_tree_height = 2 + 1/8 :=
by
  sorry

end cedar_vs_pine_height_cedar_vs_birch_height_l387_38742


namespace largest_common_value_under_800_l387_38732

-- Let's define the problem conditions as arithmetic sequences
def sequence1 (a : ℤ) : Prop := ∃ n : ℤ, a = 4 + 5 * n
def sequence2 (a : ℤ) : Prop := ∃ m : ℤ, a = 7 + 8 * m

-- Now we state the theorem that the largest common value less than 800 is 799
theorem largest_common_value_under_800 : 
  ∃ a : ℤ, sequence1 a ∧ sequence2 a ∧ a < 800 ∧ ∀ b : ℤ, sequence1 b ∧ sequence2 b ∧ b < 800 → b ≤ a :=
sorry

end largest_common_value_under_800_l387_38732


namespace carol_pennies_l387_38723

variable (a c : ℕ)

theorem carol_pennies (h₁ : c + 2 = 4 * (a - 2)) (h₂ : c - 2 = 3 * (a + 2)) : c = 62 :=
by
  sorry

end carol_pennies_l387_38723


namespace sum_of_geometric_terms_l387_38775

noncomputable def geometric_sequence (a : ℕ → ℝ) :=
  ∃ q > 0, ∀ n, a (n + 1) = q * a n

theorem sum_of_geometric_terms {a : ℕ → ℝ} 
  (hseq : geometric_sequence a)
  (h_pos : ∀ n, a n > 0)
  (h_a1 : a 1 = 1)
  (h_sum135 : a 1 + a 3 + a 5 = 21) :
  a 2 + a 4 + a 6 = 42 :=
sorry

end sum_of_geometric_terms_l387_38775


namespace min_value_of_fraction_l387_38708

theorem min_value_of_fraction (m n : ℝ) (hm : 0 < m) (hn : 0 < n) (h : 2 * m + n = 1) : 
  (1 / m + 2 / n) = 8 := 
by 
  sorry

end min_value_of_fraction_l387_38708


namespace american_literature_marks_l387_38797

variable (History HomeEconomics PhysicalEducation Art AverageMarks NumberOfSubjects TotalMarks KnownMarks : ℕ)
variable (A : ℕ)

axiom marks_history : History = 75
axiom marks_home_economics : HomeEconomics = 52
axiom marks_physical_education : PhysicalEducation = 68
axiom marks_art : Art = 89
axiom average_marks : AverageMarks = 70
axiom number_of_subjects : NumberOfSubjects = 5

def total_marks (AverageMarks NumberOfSubjects : ℕ) : ℕ := AverageMarks * NumberOfSubjects

def known_marks (History HomeEconomics PhysicalEducation Art : ℕ) : ℕ := History + HomeEconomics + PhysicalEducation + Art

axiom total_marks_eq : TotalMarks = total_marks AverageMarks NumberOfSubjects
axiom known_marks_eq : KnownMarks = known_marks History HomeEconomics PhysicalEducation Art

theorem american_literature_marks :
  A = TotalMarks - KnownMarks := by
  sorry

end american_literature_marks_l387_38797


namespace number_of_books_l387_38715

theorem number_of_books (Maddie Luisa Amy Noah : ℕ)
  (H1 : Maddie = 15)
  (H2 : Luisa = 18)
  (H3 : Amy + Luisa = Maddie + 9)
  (H4 : Noah = Amy / 3)
  : Amy + Noah = 8 :=
sorry

end number_of_books_l387_38715


namespace abs_eq_three_system1_system2_l387_38707

theorem abs_eq_three : ∀ x : ℝ, |x| = 3 ↔ x = 3 ∨ x = -3 := 
by sorry

theorem system1 : ∀ x y : ℝ, (y * (x - 1) = 0) ∧ (2 * x + 5 * y = 7) → 
(x = 7 / 2 ∧ y = 0) ∨ (x = 1 ∧ y = 1) := 
by sorry

theorem system2 : ∀ x y : ℝ, (x * y - 2 * x - y + 2 = 0) ∧ (x + 6 * y = 3) ∧ (3 * x + y = 8) → 
(x = 1 ∧ y = 5) ∨ (x = 2 ∧ y = 2) := 
by sorry

end abs_eq_three_system1_system2_l387_38707


namespace correct_sum_after_digit_change_l387_38776

theorem correct_sum_after_digit_change :
  let d := 7
  let e := 8
  let num1 := 935641
  let num2 := 471850
  let correct_sum := num1 + num2
  let new_sum := correct_sum + 10000
  new_sum = 1417491 := 
sorry

end correct_sum_after_digit_change_l387_38776


namespace intersection_M_N_l387_38737

-- Define set M and N
def M : Set ℝ := {x | x - 1 < 0}
def N : Set ℝ := {x | x^2 - 5 * x + 6 > 0}

-- Problem statement to show their intersection
theorem intersection_M_N :
  M ∩ N = {x | x < 1} := 
sorry

end intersection_M_N_l387_38737


namespace find_other_number_l387_38733

theorem find_other_number (a b : ℕ) (h1 : Nat.gcd a b = 24) (h2 : Nat.lcm a b = 5040) (h3 : a = 240) : b = 504 :=
by {
  sorry
}

end find_other_number_l387_38733


namespace Earl_rate_36_l387_38703

theorem Earl_rate_36 (E : ℝ) (h1 : E + (2 / 3) * E = 60) : E = 36 :=
by {
  sorry
}

end Earl_rate_36_l387_38703


namespace little_twelve_conference_games_l387_38760

def teams_in_division : ℕ := 6
def divisions : ℕ :=  2

def games_within_division (t : ℕ) : ℕ := (t * (t - 1)) / 2 * 2

def games_between_divisions (d t : ℕ) : ℕ := t * t

def total_conference_games (d t : ℕ) : ℕ :=
  d * games_within_division t + games_between_divisions d t

theorem little_twelve_conference_games :
  total_conference_games divisions teams_in_division = 96 :=
by
  sorry

end little_twelve_conference_games_l387_38760


namespace students_catching_up_on_homework_l387_38785

-- Definitions for the given conditions
def total_students := 120
def silent_reading_students := (2/5 : ℚ) * total_students
def board_games_students := (3/10 : ℚ) * total_students
def group_discussions_students := (1/8 : ℚ) * total_students
def other_activities_students := silent_reading_students + board_games_students + group_discussions_students
def catching_up_homework_students := total_students - other_activities_students

-- Statement of the proof problem
theorem students_catching_up_on_homework : catching_up_homework_students = 21 := by
  sorry

end students_catching_up_on_homework_l387_38785


namespace greatest_integer_b_not_in_range_of_quadratic_l387_38755

theorem greatest_integer_b_not_in_range_of_quadratic :
  ∀ b : ℤ, (∀ x : ℝ, x^2 + (b : ℝ) * x + 20 ≠ 5) ↔ (b^2 < 60) ∧ (b ≤ 7) := by
  sorry

end greatest_integer_b_not_in_range_of_quadratic_l387_38755


namespace exp_7pi_over_2_eq_i_l387_38734

theorem exp_7pi_over_2_eq_i : Complex.exp (7 * Real.pi * Complex.I / 2) = Complex.I :=
by
  sorry

end exp_7pi_over_2_eq_i_l387_38734


namespace total_distance_traveled_is_960_l387_38754

-- Definitions of conditions
def first_day_distance : ℝ := 100
def second_day_distance : ℝ := 3 * first_day_distance
def third_day_distance : ℝ := second_day_distance + 110
def fourth_day_distance : ℝ := 150

-- The total distance traveled in four days
def total_distance : ℝ := first_day_distance + second_day_distance + third_day_distance + fourth_day_distance

-- Theorem statement
theorem total_distance_traveled_is_960 :
  total_distance = 960 :=
by
  sorry

end total_distance_traveled_is_960_l387_38754


namespace sum_even_1_to_200_l387_38729

open Nat

/-- The sum of all even numbers from 1 to 200 is 10100. --/
theorem sum_even_1_to_200 :
  let first_term := 2
  let last_term := 200
  let common_diff := 2
  let n := (last_term - first_term) / common_diff + 1
  let sum := n / 2 * (first_term + last_term)
  sum = 10100 :=
by
  let first_term := 2
  let last_term := 200
  let common_diff := 2
  let n := (last_term - first_term) / common_diff + 1
  let sum := n / 2 * (first_term + last_term)
  show sum = 10100
  sorry

end sum_even_1_to_200_l387_38729


namespace largest_angle_in_hexagon_l387_38749

theorem largest_angle_in_hexagon :
  ∀ (x : ℝ), (2 * x + 3 * x + 3 * x + 4 * x + 4 * x + 5 * x = 720) →
  5 * x = 1200 / 7 :=
by
  intros x h
  sorry

end largest_angle_in_hexagon_l387_38749


namespace tangent_circles_pass_through_homothety_center_l387_38759

-- Define the necessary structures
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def is_tangent_to_line (ω : Circle) (L : ℝ → ℝ) : Prop :=
  sorry -- Definition of tangency to a line

def is_tangent_to_circle (ω : Circle) (C : Circle) : Prop :=
  sorry -- Definition of tangency to another circle

theorem tangent_circles_pass_through_homothety_center
  (L : ℝ → ℝ) (C : Circle) (ω : Circle)
  (H_ext H_int : ℝ × ℝ)
  (H_tangency_line : is_tangent_to_line ω L)
  (H_tangency_circle : is_tangent_to_circle ω C) :
  ∃ P Q : ℝ × ℝ, 
    (is_tangent_to_line ω L ∧ is_tangent_to_circle ω C) →
    (P = Q ∧ (P = H_ext ∨ P = H_int)) :=
by
  sorry

end tangent_circles_pass_through_homothety_center_l387_38759


namespace two_solutions_exist_l387_38768

theorem two_solutions_exist 
  (a b c : ℝ) 
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h_equation : (1 / a) + (1 / b) + (1 / c) = (1 / (a + b + c))) : 
  ∃ (a' b' c' : ℝ), 
    ((a' = 1/3 ∧ b' = 1/3 ∧ c' = 1/3) ∨ (a' = -1/3 ∧ b' = -1/3 ∧ c' = -1/3)) := 
sorry

end two_solutions_exist_l387_38768


namespace exist_colored_points_r_gt_pi_div_sqrt3_exist_colored_points_r_gt_pi_div_2_l387_38711

theorem exist_colored_points_r_gt_pi_div_sqrt3 (r : ℝ) (hr : r > π / Real.sqrt 3) 
    (coloring : ℝ × ℝ → Prop) : 
    (∀ p, 0 ≤ p.1 ∧ p.1^2 + p.2^2 < r^2 → coloring p ∨ ¬coloring p) → 
    ∃ A B : ℝ × ℝ, (0 ≤ A.1 ∧ 0 ≤ B.1 ∧ A.1^2 + A.2^2 < r^2 ∧ B.1^2 + B.2^2 < r^2 ∧ dist A B = π ∧ coloring A = coloring B) :=
sorry

theorem exist_colored_points_r_gt_pi_div_2 (r : ℝ) (hr : r > π / 2)
    (coloring : ℝ × ℝ → Prop) : 
    (∀ p, 0 ≤ p.1 ∧ p.1^2 + p.2^2 < r^2 → coloring p ∨ ¬coloring p) → 
    ∃ A B : ℝ × ℝ, (0 ≤ A.1 ∧ 0 ≤ B.1 ∧ A.1^2 + A.2^2 < r^2 ∧ B.1^2 + B.2^2 < r^2 ∧ dist A B = π ∧ coloring A = coloring B) :=
sorry

end exist_colored_points_r_gt_pi_div_sqrt3_exist_colored_points_r_gt_pi_div_2_l387_38711


namespace mo_tea_cups_l387_38746

theorem mo_tea_cups (n t : ℤ) (h1 : 4 * n + 3 * t = 22) (h2 : 3 * t = 4 * n + 8) : t = 5 :=
by
  -- proof steps
  sorry

end mo_tea_cups_l387_38746


namespace range_of_expression_l387_38761

theorem range_of_expression (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hxy : x + y ≤ 1) : 
  ∃ (z : Set ℝ), z = Set.Icc (2 / 3) 4 ∧ (4*x^2 + 4*y^2 + (1 - x - y)^2) ∈ z :=
by
  sorry

end range_of_expression_l387_38761


namespace factorize_x3_minus_9x_factorize_a3b_minus_2a2b_plus_ab_l387_38702

theorem factorize_x3_minus_9x (x : ℝ) : x^3 - 9 * x = x * (x + 3) * (x - 3) :=
sorry

theorem factorize_a3b_minus_2a2b_plus_ab (a b : ℝ) : a^3 * b - 2 * a^2 * b + a * b = a * b * (a - 1)^2 :=
sorry

end factorize_x3_minus_9x_factorize_a3b_minus_2a2b_plus_ab_l387_38702


namespace sector_area_l387_38764

/--
The area of a sector with radius 6cm and central angle 15° is (3 * π / 2) cm².
-/
theorem sector_area (R : ℝ) (θ : ℝ) (h_radius : R = 6) (h_angle : θ = 15) :
    (S : ℝ) = (3 * Real.pi / 2) := by
  sorry

end sector_area_l387_38764


namespace MaryHasBlueMarbles_l387_38751

-- Define the number of blue marbles Dan has
def DanMarbles : Nat := 5

-- Define the relationship of Mary's marbles to Dan's marbles
def MaryMarbles : Nat := 2 * DanMarbles

-- State the theorem that we need to prove
theorem MaryHasBlueMarbles : MaryMarbles = 10 :=
by
  sorry

end MaryHasBlueMarbles_l387_38751


namespace intersection_distance_zero_l387_38718

noncomputable def A : Type := ℝ × ℝ

def P : A := (2, 0)

def line_intersects_parabola (x y : ℝ) : Prop :=
  y - 2 * x + 5 = 0 ∧ y^2 = 3 * x + 4

def distance (p1 p2 : A) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

theorem intersection_distance_zero :
  ∀ (A1 A2 : A),
  line_intersects_parabola A1.1 A1.2 ∧ line_intersects_parabola A2.1 A2.2 →
  (abs (distance A1 P - distance A2 P) = 0) :=
sorry

end intersection_distance_zero_l387_38718


namespace average_price_of_tshirts_l387_38743

theorem average_price_of_tshirts
  (A : ℝ)
  (total_cost_seven_remaining : ℝ := 7 * 505)
  (total_cost_three_returned : ℝ := 3 * 673)
  (total_cost_eight : ℝ := total_cost_seven_remaining + 673) -- since (1 t-shirt with price is included in the total)
  (total_cost_eight_eq : total_cost_eight = 8 * A) :
  A = 526 :=
by sorry

end average_price_of_tshirts_l387_38743


namespace snake_body_length_l387_38709

theorem snake_body_length (L : ℝ) (H : ℝ) (h1 : H = L / 10) (h2 : L = 10) : L - H = 9 :=
by
  sorry

end snake_body_length_l387_38709


namespace functional_equation_solution_l387_38762

noncomputable def f : ℝ → ℝ := sorry 

theorem functional_equation_solution :
  (∀ x y : ℝ, f (x * y + 1) = f x * f y - f y - x + 2) →
  10 * f 2006 + f 0 = 20071 :=
by
  intros h
  sorry

end functional_equation_solution_l387_38762


namespace part_a_part_b_part_c_l387_38736

/-- (a) Given that p = 33 and q = 216, show that the equation f(x) = 0 has 
three distinct integer solutions and the equation g(x) = 0 has two distinct integer solutions.
-/
theorem part_a (p q : ℕ) (h_p : p = 33) (h_q : q = 216) :
  (∃ x1 x2 x3 : ℤ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ (x1 * x2 * x3 = 216 ∧ x1 + x2 + x3 = 33 ∧ x1 = 0))
  ∧ (∃ y1 y2 : ℤ, y1 ≠ y2 ∧ (3 * y1 * y2 = 216 ∧ y1 + y1 = 22)) := sorry

/-- (b) Suppose that the equation f(x) = 0 has three distinct integer solutions 
and the equation g(x) = 0 has two distinct integer solutions. Prove the necessary conditions 
for p and q.
-/
theorem part_b (p q : ℕ) 
  (h_f : ∃ x1 x2 x3 : ℤ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ (x1 * x2 * x3 = q ∧ x1 + x2 + x3 = p))
  (h_g : ∃ y1 y2 : ℤ, y1 ≠ y2 ∧ (3 * y1 * y2 = q ∧ y1 + y1 = 2 * p)) :
  (∃ k : ℕ, p = 3 * k) ∧ (∃ l : ℕ, q = 9 * l) ∧ (∃ m n : ℕ, p^2 - 3 * q = m^2 ∧ p^2 - 4 * q = n^2) := sorry

/-- (c) Prove that there are infinitely many pairs of positive integers (p, q) for which:
1. The equation f(x) = 0 has three distinct integer solutions.
2. The equation g(x) = 0 has two distinct integer solutions.
3. The greatest common divisor of p and q is 3.
-/
theorem part_c :
  ∃ (p q : ℕ) (infinitely_many : ℕ → Prop),
  (∃ x1 x2 x3 : ℤ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ (x1 * x2 * x3 = q ∧ x1 + x2 + x3 = p))
  ∧ (∃ y1 y2 : ℤ, y1 ≠ y2 ∧ (3 * y1 * y2 = q ∧ y1 + y1 = 2 * p))
  ∧ ∃ k : ℕ, gcd p q = 3 ∧ infinitely_many k := sorry

end part_a_part_b_part_c_l387_38736


namespace solution_set_of_inequality_l387_38780

theorem solution_set_of_inequality : 
  {x : ℝ | x < x^2} = {x | x < 0} ∪ {x | x > 1} :=
by sorry

end solution_set_of_inequality_l387_38780


namespace parabola_directrix_eq_l387_38789

theorem parabola_directrix_eq (a : ℝ) (h : - a / 4 = - (1 : ℝ) / 4) : a = 1 := by
  sorry

end parabola_directrix_eq_l387_38789


namespace minimum_students_l387_38712

variables (b g : ℕ) -- Define variables for boys and girls

-- Define the conditions
def boys_passed : ℕ := (3 * b) / 4
def girls_passed : ℕ := (2 * g) / 3
def equal_passed := boys_passed b = girls_passed g

def total_students := b + g + 4

-- Statement to prove minimum students in the class
theorem minimum_students (h1 : equal_passed b g)
  (h2 : ∃ multiple_of_nine : ℕ, g = 9 * multiple_of_nine ∧ 3 * b = 4 * multiple_of_nine * 2) :
  total_students b g = 21 :=
sorry

end minimum_students_l387_38712


namespace parabola_focus_distance_l387_38756

theorem parabola_focus_distance
  (F P Q : ℝ × ℝ)
  (hF : F = (1 / 2, 0))
  (hP : ∃ y, P = (2 * y^2, y))
  (hQ : Q = (1 / 2, Q.2))
  (h_parallel : P.2 = Q.2)
  (h_distance : dist P Q = dist Q F) :
  dist P F = 2 :=
by
  sorry

end parabola_focus_distance_l387_38756


namespace car_r_speed_l387_38705

variable (v : ℝ)

theorem car_r_speed (h1 : (300 / v - 2 = 300 / (v + 10))) : v = 30 := 
sorry

end car_r_speed_l387_38705


namespace triangle_obtuse_at_15_l387_38790

-- Define the initial angles of the triangle
def x0 : ℝ := 59.999
def y0 : ℝ := 60
def z0 : ℝ := 60.001

-- Define the recurrence relations for the angles
def x (n : ℕ) : ℝ := (-2)^n * (x0 - 60) + 60
def y (n : ℕ) : ℝ := (-2)^n * (y0 - 60) + 60
def z (n : ℕ) : ℝ := (-2)^n * (z0 - 60) + 60

-- Define the obtuseness condition
def is_obtuse (a : ℝ) : Prop := a > 90

-- The main theorem stating the least positive integer n is 15 for which the triangle A_n B_n C_n is obtuse
theorem triangle_obtuse_at_15 : ∃ n : ℕ, n > 0 ∧ 
  (is_obtuse (x n) ∨ is_obtuse (y n) ∨ is_obtuse (z n)) ∧ n = 15 :=
sorry

end triangle_obtuse_at_15_l387_38790


namespace problem_solution_l387_38747

noncomputable def f (x : ℝ) : ℝ := (1 / 4) * (x + 1)^2

theorem problem_solution :
  (∀ x : ℝ, (0 < x ∧ x ≤ 5) → x ≤ f x ∧ f x ≤ 2 * |x - 1| + 1) →
  (f 1 = 4 * (1 / 4) + 1) →
  (∃ (t m : ℝ), m > 1 ∧ 
               (∀ x : ℝ, (1 ≤ x ∧ x ≤ m) → f t ≤ (1 / 4) * (x + t + 1)^2)) →
  (1 / 4 = 1 / 4) ∧ (m = 2) :=
by
  intros h1 h2 h3
  sorry

end problem_solution_l387_38747


namespace sqrt_sum_difference_product_l387_38796

open Real

theorem sqrt_sum_difference_product :
  (sqrt 3 + sqrt 2) * (sqrt 3 - sqrt 2) = 1 := by
  sorry

end sqrt_sum_difference_product_l387_38796


namespace projectile_height_35_l387_38757

theorem projectile_height_35 (t : ℝ) : 
  (∃ t : ℝ, -4.9 * t ^ 2 + 30 * t = 35 ∧ t > 0) → t = 10 / 7 := 
sorry

end projectile_height_35_l387_38757


namespace conditional_probability_event_B_given_event_A_l387_38740

-- Definitions of events A and B
def event_A := {outcomes | ∃ i j k, outcomes = (i, j, k) ∧ (i = 1 ∨ j = 1 ∨ k = 1)}
def event_B := {outcomes | ∃ i j k, outcomes = (i, j, k) ∧ (i + j + k = 1)}

-- Calculation of probabilities
def probability_AB := 3 / 8
def probability_A := 7 / 8

-- Prove conditional probability
theorem conditional_probability_event_B_given_event_A :
  (probability_AB / probability_A) = 3 / 7 :=
by
  sorry

end conditional_probability_event_B_given_event_A_l387_38740


namespace unshaded_squares_in_tenth_figure_l387_38724

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + d * (n - 1)

theorem unshaded_squares_in_tenth_figure :
  arithmetic_sequence 8 4 10 = 44 :=
by
  sorry

end unshaded_squares_in_tenth_figure_l387_38724


namespace find_a_for_even_function_l387_38721

theorem find_a_for_even_function :
  ∀ a : ℝ, (∀ x : ℝ, a * 3^x + 1 / 3^x = a * 3^(-x) + 1 / 3^(-x)) → a = 1 :=
by
  sorry

end find_a_for_even_function_l387_38721


namespace integer_solutions_count_l387_38716

theorem integer_solutions_count : 
  ∃ (S : Finset (ℤ × ℤ)), 
  (∀ x y, (x, y) ∈ S ↔ x^2 + x * y + 2 * y^2 = 29) ∧ 
  S.card = 4 := 
sorry

end integer_solutions_count_l387_38716


namespace solve_for_m_l387_38713

theorem solve_for_m (a_0 a_1 a_2 a_3 a_4 a_5 m : ℝ)
  (h1 : (x : ℝ) → (x + m)^5 = a_0 + a_1 * (x+1) + a_2 * (x+1)^2 + a_3 * (x+1)^3 + a_4 * (x+1)^4 + a_5 * (x+1)^5)
  (h2 : a_0 + a_1 + a_2 + a_3 + a_4 + a_5 = 32) :
  m = 2 :=
sorry

end solve_for_m_l387_38713


namespace sum_of_three_numbers_l387_38726

theorem sum_of_three_numbers :
  ∀ (a b c : ℕ), 
  a ≤ b ∧ b ≤ c → b = 10 →
  (a + b + c) / 3 = a + 20 →
  (a + b + c) / 3 = c - 30 →
  a + b + c = 60 :=
by
  sorry

end sum_of_three_numbers_l387_38726


namespace probability_of_passing_through_correct_l387_38748

def probability_of_passing_through (n k : ℕ) : ℚ :=
(2 * k * n - 2 * k^2 + 2 * k - 1) / n^2

theorem probability_of_passing_through_correct (n k : ℕ) (h1 : 1 ≤ k) (h2 : k ≤ n) :
  probability_of_passing_through n k = (2 * k * n - 2 * k^2 + 2 * k - 1) / n^2 := 
by
  sorry

end probability_of_passing_through_correct_l387_38748


namespace ratio_of_almonds_to_walnuts_l387_38727

theorem ratio_of_almonds_to_walnuts
  (A W : ℝ)
  (weight_almonds : ℝ)
  (total_weight : ℝ)
  (weight_walnuts : ℝ)
  (ratio : 2 * W = total_weight - weight_almonds)
  (given_almonds : weight_almonds = 107.14285714285714)
  (given_total_weight : total_weight = 150)
  (computed_weight_walnuts : weight_walnuts = 42.85714285714286)
  (proportion : A / (2 * W) = weight_almonds / weight_walnuts) :
  A / W = 5 :=
by
  sorry

end ratio_of_almonds_to_walnuts_l387_38727


namespace total_cost_l387_38739

def cost(M R F : ℝ) := 10 * M = 24 * R ∧ 6 * F = 2 * R ∧ F = 23

theorem total_cost (M R F : ℝ) (h : cost M R F) : 
  4 * M + 3 * R + 5 * F = 984.40 :=
by
  sorry

end total_cost_l387_38739


namespace sufficient_not_necessary_condition_l387_38738

-- Definitions of propositions
def propA (x : ℝ) : Prop := (x - 1)^2 < 9
def propB (x a : ℝ) : Prop := (x + 2) * (x + a) < 0

-- Lean statement of the problem
theorem sufficient_not_necessary_condition (a : ℝ) : 
  (∀ x, propA x → propB x a) ∧ (∃ x, ¬ propA x ∧ propB x a) ↔ a < -4 :=
sorry

end sufficient_not_necessary_condition_l387_38738


namespace exists_x_l387_38793

theorem exists_x (a b c : ℕ) (ha : 0 < a) (hc : 0 < c) :
  ∃ x : ℕ, (0 < x) ∧ (a ^ x + x) % c = b % c :=
sorry

end exists_x_l387_38793


namespace no_information_loss_chart_is_stem_and_leaf_l387_38758

theorem no_information_loss_chart_is_stem_and_leaf :
  "The correct chart with no information loss" = "Stem-and-leaf plot" :=
sorry

end no_information_loss_chart_is_stem_and_leaf_l387_38758


namespace reflection_across_x_axis_l387_38799

def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

theorem reflection_across_x_axis :
  reflect_x_axis (-2, -3) = (-2, 3) :=
by
  sorry

end reflection_across_x_axis_l387_38799


namespace fraction_expression_l387_38772

theorem fraction_expression :
  (3 / 7 + 5 / 8) / (5 / 12 + 2 / 9) = 531 / 322 :=
by sorry

end fraction_expression_l387_38772


namespace complement_U_A_correct_l387_38784

-- Define the universal set U and set A
def U : Set Int := {-1, 0, 2}
def A : Set Int := {-1, 0}

-- Define the complement of A in U
def complement_U_A : Set Int := {x | x ∈ U ∧ x ∉ A}

-- Theorem stating the required proof
theorem complement_U_A_correct : complement_U_A = {2} :=
by
  sorry -- Proof will be filled in

end complement_U_A_correct_l387_38784


namespace graph_of_eq_hyperbola_l387_38782

theorem graph_of_eq_hyperbola (x y : ℝ) : (x + y)^2 = x^2 + y^2 + 1 → ∃ a b : ℝ, a * b = x * y ∧ a * b = 1/2 := by
  sorry

end graph_of_eq_hyperbola_l387_38782


namespace find_a5_div_b5_l387_38752

-- Definitions
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop := ∃ d, ∀ n, a (n + 1) = a n + d
def sum_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ := n * (a 0 + a (n - 1)) / 2

-- Main statement
theorem find_a5_div_b5 (a b : ℕ → ℤ) (S T : ℕ → ℤ)
  (h1 : is_arithmetic_sequence a)
  (h2 : is_arithmetic_sequence b)
  (h3 : ∀ n : ℕ, S n = sum_first_n_terms a n)
  (h4 : ∀ n : ℕ, T n = sum_first_n_terms b n)
  (h5 : ∀ n : ℕ, S n * (3 * n + 1) = 2 * n * T n) :
  (a 5 : ℚ) / b 5 = 9 / 14 :=
by
  sorry

end find_a5_div_b5_l387_38752


namespace area_ratio_equilateral_triangl_l387_38774

theorem area_ratio_equilateral_triangl (x : ℝ) :
  let sA : ℝ := x 
  let sB : ℝ := 3 * sA
  let sC : ℝ := 5 * sA
  let sD : ℝ := 4 * sA
  let area_ABC := (Real.sqrt 3 / 4) * (sA ^ 2)
  let s := (sB + sC + sD) / 2
  let area_A'B'C' := Real.sqrt (s * (s - sB) * (s - sC) * (s - sD))
  (area_A'B'C' / area_ABC) = 8 * Real.sqrt 3 := by
  sorry

end area_ratio_equilateral_triangl_l387_38774


namespace sum_of_digits_is_13_l387_38700

theorem sum_of_digits_is_13:
  ∀ (a b c d : ℕ),
  b + c = 10 ∧
  c + d = 1 ∧
  a + d = 2 →
  a + b + c + d = 13 :=
by {
  sorry
}

end sum_of_digits_is_13_l387_38700


namespace square_garden_tiles_l387_38730

theorem square_garden_tiles (n : ℕ) (h : 2 * n - 1 = 25) : n^2 = 169 :=
by
  sorry

end square_garden_tiles_l387_38730


namespace units_digit_7_pow_2023_l387_38765

-- We start by defining a function to compute units digit of powers of 7 modulo 10.
def units_digit_of_7_pow (n : ℕ) : ℕ :=
  (7 ^ n) % 10

-- Define the problem statement: the units digit of 7^2023 is equal to 3.
theorem units_digit_7_pow_2023 : units_digit_of_7_pow 2023 = 3 := sorry

end units_digit_7_pow_2023_l387_38765


namespace Tony_packs_of_pens_l387_38714

theorem Tony_packs_of_pens (T : ℕ) 
  (Kendra_packs : ℕ := 4) 
  (pens_per_pack : ℕ := 3) 
  (Kendra_keep : ℕ := 2) 
  (Tony_keep : ℕ := 2)
  (friends_pens : ℕ := 14) 
  (total_pens_given : Kendra_packs * pens_per_pack - Kendra_keep + 3 * T - Tony_keep = friends_pens) :
  T = 2 :=
by {
  sorry
}

end Tony_packs_of_pens_l387_38714


namespace greatest_possible_n_l387_38779

theorem greatest_possible_n (n : ℤ) (h1 : 102 * n^2 ≤ 8100) : n ≤ 8 :=
sorry

end greatest_possible_n_l387_38779


namespace factorize_cubic_expression_l387_38731

variable (a : ℝ)

theorem factorize_cubic_expression : a^3 - a = a * (a + 1) * (a - 1) := 
sorry

end factorize_cubic_expression_l387_38731


namespace complex_number_z_l387_38794

theorem complex_number_z (z : ℂ) (i : ℂ) (hz : i^2 = -1) (h : (1 - i)^2 / z = 1 + i) : z = -1 - i :=
by
  sorry

end complex_number_z_l387_38794


namespace min_mn_value_l387_38792

theorem min_mn_value (m n : ℕ) (hmn : m > n) (hn : n ≥ 1) 
  (hdiv : 1000 ∣ 1978 ^ m - 1978 ^ n) : m + n = 106 :=
sorry

end min_mn_value_l387_38792


namespace positive_difference_sums_l387_38773

theorem positive_difference_sums : 
  let n_even := 25
  let n_odd := 20
  let sum_even_n := 2 * (n_even * (n_even + 1)) / 2
  let sum_odd_n := (1 + (2 * n_odd - 1)) * n_odd / 2
  sum_even_n - sum_odd_n = 250 :=
by
  intros
  let n_even := 25
  let n_odd := 20
  let sum_even_n := 2 * (n_even * (n_even + 1)) / 2
  let sum_odd_n := (1 + (2 * n_odd - 1)) * n_odd / 2
  show sum_even_n - sum_odd_n = 250
  sorry

end positive_difference_sums_l387_38773


namespace speed_of_stream_l387_38717

theorem speed_of_stream 
  (v : ℝ)
  (boat_speed : ℝ)
  (distance_downstream : ℝ)
  (distance_upstream : ℝ)
  (H1 : boat_speed = 12)
  (H2 : distance_downstream = 32)
  (H3 : distance_upstream = 16)
  (H4 : distance_downstream / (boat_speed + v) = distance_upstream / (boat_speed - v)) :
  v = 4 :=
by
  sorry

end speed_of_stream_l387_38717


namespace count_parallelograms_392_l387_38753

-- Define the conditions in Lean
def is_lattice_point (x y : ℕ) : Prop :=
  ∃ q : ℕ, x = q ∧ y = q

def on_line_y_eq_x (x y : ℕ) : Prop :=
  y = x ∧ is_lattice_point x y

def on_line_y_eq_mx (x y : ℕ) (m : ℕ) : Prop :=
  y = m * x ∧ is_lattice_point x y ∧ m > 1

def area_parallelogram (q s m : ℕ) : ℕ :=
  (m - 1) * q * s

-- Define the target theorem
theorem count_parallelograms_392 :
  (∀ (q s m : ℕ),
    on_line_y_eq_x q q →
    on_line_y_eq_mx s (m * s) m →
    area_parallelogram q s m = 250000) →
  (∃! n : ℕ, n = 392) :=
sorry

end count_parallelograms_392_l387_38753


namespace equivalent_statements_l387_38701

-- Definitions based on the problem
def is_not_negative (x : ℝ) : Prop := x >= 0
def is_not_positive (x : ℝ) : Prop := x <= 0
def is_positive (x : ℝ) : Prop := x > 0
def is_negative (x : ℝ) : Prop := x < 0

-- The main theorem statement
theorem equivalent_statements (x : ℝ) : 
  (is_not_negative x → is_not_positive (x^2)) ↔ (is_positive (x^2) → is_negative x) :=
by
  sorry

end equivalent_statements_l387_38701


namespace ratio_of_B_to_C_l387_38770

-- Definitions based on conditions
def A := 40
def C := A + 20
def total := 220
def B := total - A - C

-- Theorem statement
theorem ratio_of_B_to_C : B / C = 2 :=
by
  -- Placeholder for proof
  sorry

end ratio_of_B_to_C_l387_38770


namespace remainder_1234567_127_l387_38763

theorem remainder_1234567_127 : (1234567 % 127) = 51 := 
by {
  sorry
}

end remainder_1234567_127_l387_38763


namespace net_cannot_contain_2001_knots_l387_38787

theorem net_cannot_contain_2001_knots (knots : Nat) (ropes_per_knot : Nat) (total_knots : knots = 2001) (ropes_per_knot_eq : ropes_per_knot = 3) :
  false :=
by
  sorry

end net_cannot_contain_2001_knots_l387_38787


namespace evaluate_fg_sum_at_1_l387_38769

def f (x : ℚ) : ℚ := (4 * x^2 + 3 * x + 6) / (x^2 + 2 * x + 5)
def g (x : ℚ) : ℚ := x + 1

theorem evaluate_fg_sum_at_1 : f (g 1) + g (f 1) = 497 / 104 :=
by
  sorry

end evaluate_fg_sum_at_1_l387_38769


namespace smallest_delightful_integer_l387_38722

-- Definition of "delightful" integer
def is_delightful (B : ℤ) : Prop :=
  ∃ (n : ℕ), (n > 0) ∧ ((n + 1) * (2 * B + n)) / 2 = 3050

-- Proving the smallest delightful integer
theorem smallest_delightful_integer : ∃ (B : ℤ), is_delightful B ∧ ∀ (B' : ℤ), is_delightful B' → B ≤ B' :=
  sorry

end smallest_delightful_integer_l387_38722


namespace dreamCarCost_l387_38750

-- Definitions based on given conditions
def monthlyEarnings : ℕ := 4000
def monthlySavings : ℕ := 500
def totalEarnings : ℕ := 360000

-- Theorem stating the desired result
theorem dreamCarCost :
  (totalEarnings / monthlyEarnings) * monthlySavings = 45000 :=
by
  sorry

end dreamCarCost_l387_38750


namespace find_x_plus_y_squared_l387_38771

variable (x y a b : ℝ)

def condition1 := x * y = b
def condition2 := (1 / (x ^ 2)) + (1 / (y ^ 2)) = a

theorem find_x_plus_y_squared (h1 : condition1 x y b) (h2 : condition2 x y a) : 
  (x + y) ^ 2 = a * b ^ 2 + 2 * b :=
by
  sorry

end find_x_plus_y_squared_l387_38771


namespace hcf_of_numbers_is_five_l387_38777

theorem hcf_of_numbers_is_five (a b x : ℕ) (ratio : a = 3 * x) (ratio_b : b = 4 * x)
  (lcm_ab : Nat.lcm a b = 60) (hcf_ab : Nat.gcd a b = 5) : Nat.gcd a b = 5 :=
by
  sorry

end hcf_of_numbers_is_five_l387_38777


namespace gcd_m_n_15_lcm_m_n_45_l387_38745

-- Let m and n be integers greater than 0, and 3m + 2n = 225.
variables (m n : ℕ) (h1 : m > 0) (h2 : n > 0) (h3 : 3 * m + 2 * n = 225)

-- First part: If the greatest common divisor of m and n is 15, then m + n = 105.
theorem gcd_m_n_15 (h4 : Int.gcd m n = 15) : m + n = 105 :=
sorry

-- Second part: If the least common multiple of m and n is 45, then m + n = 90.
theorem lcm_m_n_45 (h5 : Int.lcm m n = 45) : m + n = 90 :=
sorry

end gcd_m_n_15_lcm_m_n_45_l387_38745


namespace area_inequality_l387_38766

theorem area_inequality 
  (α β γ : ℝ) 
  (P Q S : ℝ) 
  (h1 : P / Q = α * β * γ) 
  (h2 : S = Q * (α + 1) * (β + 1) * (γ + 1)) : 
  (S ^ (1 / 3)) ≥ (P ^ (1 / 3)) + (Q ^ (1 / 3)) :=
by
  sorry

end area_inequality_l387_38766
