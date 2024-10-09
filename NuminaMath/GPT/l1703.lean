import Mathlib

namespace memorial_visits_l1703_170303

theorem memorial_visits (x : ℕ) (total_visits : ℕ) (difference : ℕ) 
  (h1 : total_visits = 589) 
  (h2 : difference = 56) 
  (h3 : 2 * x + difference = total_visits - x) : 
  2 * x + 56 = 589 - x :=
by
  -- proof steps would go here
  sorry

end memorial_visits_l1703_170303


namespace mark_repayment_l1703_170325

noncomputable def totalDebt (days : ℕ) : ℝ :=
  if days < 3 then
    20 + (20 * 0.10 * days)
  else
    35 + (20 * 0.10 * 3) + (35 * 0.10 * (days - 3))

theorem mark_repayment :
  ∃ (x : ℕ), totalDebt x ≥ 70 ∧ x = 12 :=
by
  -- Use this theorem statement to prove the corresponding lean proof
  sorry

end mark_repayment_l1703_170325


namespace initial_volume_is_72_l1703_170305

noncomputable def initial_volume (V : ℝ) : Prop :=
  let salt_initial : ℝ := 0.10 * V
  let total_volume_new : ℝ := V + 18
  let salt_percentage_new : ℝ := 0.08 * total_volume_new
  salt_initial = salt_percentage_new

theorem initial_volume_is_72 :
  ∃ V : ℝ, initial_volume V ∧ V = 72 :=
by
  sorry

end initial_volume_is_72_l1703_170305


namespace value_of_f_minus_a_l1703_170309

noncomputable def f (x : ℝ) : ℝ := x^3 + x + 1

theorem value_of_f_minus_a (a : ℝ) (h : f a = 2) : f (-a) = 0 :=
by sorry

end value_of_f_minus_a_l1703_170309


namespace train_speed_l1703_170318

theorem train_speed (distance time : ℤ) (h_distance : distance = 500)
    (h_time : time = 3) :
    distance / time = 166 :=
by
  -- Proof steps will be filled in here
  sorry

end train_speed_l1703_170318


namespace age_difference_is_40_l1703_170375

-- Define the ages of the daughter and the mother
variables (D M : ℕ)

-- Conditions
-- 1. The mother's age is the digits of the daughter's age reversed
def mother_age_is_reversed_daughter_age : Prop :=
  M = 10 * D + D

-- 2. In thirteen years, the mother will be twice as old as the daughter
def mother_twice_as_old_in_thirteen_years : Prop :=
  M + 13 = 2 * (D + 13)

-- The theorem: The difference in their current ages is 40
theorem age_difference_is_40
  (h1 : mother_age_is_reversed_daughter_age D M)
  (h2 : mother_twice_as_old_in_thirteen_years D M) :
  M - D = 40 :=
sorry

end age_difference_is_40_l1703_170375


namespace factor_quadratic_l1703_170334

theorem factor_quadratic (y : ℝ) : 9 * y ^ 2 - 30 * y + 25 = (3 * y - 5) ^ 2 := by
  sorry

end factor_quadratic_l1703_170334


namespace find_crossed_out_digit_l1703_170330

theorem find_crossed_out_digit (n : ℕ) (h_rev : ∀ (k : ℕ), k < n → k % 9 = 0) (remaining_sum : ℕ) 
  (crossed_sum : ℕ) (h_sum : remaining_sum + crossed_sum = 27) : 
  crossed_sum = 8 :=
by
  -- We can incorporate generating the value from digit sum here.
  sorry

end find_crossed_out_digit_l1703_170330


namespace cos_seven_theta_l1703_170314

theorem cos_seven_theta (θ : ℝ) (h : Real.cos θ = 2 / 5) : Real.cos (7 * θ) = -83728 / 390625 := 
sorry

end cos_seven_theta_l1703_170314


namespace probability_of_4_vertices_in_plane_l1703_170338

-- Definition of the problem conditions
def vertices_of_cube : Nat := 8
def selecting_vertices : Nat := 4

-- Combination function
def combination (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Total ways to select 4 vertices from the 8 vertices of a cube
def total_ways : Nat := combination vertices_of_cube selecting_vertices

-- Number of favorable ways that these 4 vertices lie in the same plane
def favorable_ways : Nat := 12

-- Probability calculation
def probability : ℚ := favorable_ways / total_ways

-- The ultimate proof problem
theorem probability_of_4_vertices_in_plane :
  probability = 6 / 35 :=
by
  -- Here, the proof steps would go to verify that our setup correctly leads to the given probability.
  sorry

end probability_of_4_vertices_in_plane_l1703_170338


namespace outfit_choices_l1703_170363

theorem outfit_choices:
  let shirts := 8
  let pants := 8
  let hats := 8
  -- Each has 8 different colors
  -- No repetition of color within type of clothing
  -- Refuse to wear same color shirt and pants
  (shirts * pants * hats) - (shirts * hats) = 448 := 
sorry

end outfit_choices_l1703_170363


namespace circle_center_sum_l1703_170369

theorem circle_center_sum (x y : ℝ) (hx : (x, y) = (3, -4)) :
  (x + y) = -1 :=
by {
  -- We are given that the center of the circle is (3, -4)
  sorry -- Proof is omitted
}

end circle_center_sum_l1703_170369


namespace ellipse_hyperbola_tangent_n_values_are_62_20625_and_1_66875_l1703_170328

theorem ellipse_hyperbola_tangent_n_values_are_62_20625_and_1_66875:
  let is_ellipse (x y n : ℝ) := x^2 + n*(y - 1)^2 = n
  let is_hyperbola (x y : ℝ) := x^2 - 4*(y + 3)^2 = 4
  ∃ (n1 n2 : ℝ),
    n1 = 62.20625 ∧ n2 = 1.66875 ∧
    (∀ (x y : ℝ), is_ellipse x y n1 → is_hyperbola x y → 
       is_ellipse x y n2 → is_hyperbola x y → 
       (4 + n1)*(y^2 - 2*y + 1) = 4*(y^2 + 6*y + 9) + 4 ∧ 
       ((24 - 2*n1)^2 - 4*(4 + n1)*40 = 0) ∧
       (4 + n2)*(y^2 - 2*y + 1) = 4*(y^2 + 6*y + 9) + 4 ∧ 
       ((24 - 2*n2)^2 - 4*(4 + n2)*40 = 0))
:= sorry

end ellipse_hyperbola_tangent_n_values_are_62_20625_and_1_66875_l1703_170328


namespace find_x_minus_y_l1703_170382

theorem find_x_minus_y (x y z : ℤ) (h₁ : x - y - z = 7) (h₂ : x - y + z = 15) : x - y = 11 := by
  sorry

end find_x_minus_y_l1703_170382


namespace problem1_problem2_problem3_problem4_problem5_problem6_problem7_problem8_l1703_170379

-- Define the conditions for each problem explicitly
def cond1 : Prop := ∃ (A B C : Type), -- "A" can only be in the middle or on the sides (positions are constrainted)
  True -- (specific arrangements are abstracted here)

def cond2 : Prop := ∃ (A B C : Type), -- male students must be grouped together
  True

def cond3 : Prop := ∃ (A B C : Type), -- male students cannot be grouped together
  True

def cond4 : Prop := ∃ (A B C : Type), -- the order of "A", "B", "C" from left to right remains unchanged
  True

def cond5 : Prop := ∃ (A B C : Type), -- "A" is not on the far left and "B" is not on the far right
  True

def cond6 : Prop := ∃ (A B C D : Type), -- One more female student, males and females are not next to each other
  True

def cond7 : Prop := ∃ (A B C : Type), -- arranged in two rows, with 3 people in the front row and 2 in the back row
  True

def cond8 : Prop := ∃ (A B C : Type), -- there must be 1 person between "A" and "B"
  True

-- Prove each condition results in the specified number of arrangements

theorem problem1 : cond1 → True := by
  -- Problem (1) is to show 72 arrangements given conditions
  sorry

theorem problem2 : cond2 → True := by
  -- Problem (2) is to show 36 arrangements given conditions
  sorry

theorem problem3 : cond3 → True := by
  -- Problem (3) is to show 12 arrangements given conditions
  sorry

theorem problem4 : cond4 → True := by
  -- Problem (4) is to show 20 arrangements given conditions
  sorry

theorem problem5 : cond5 → True := by
  -- Problem (5) is to show 78 arrangements given conditions
  sorry

theorem problem6 : cond6 → True := by
  -- Problem (6) is to show 144 arrangements given conditions
  sorry

theorem problem7 : cond7 → True := by
  -- Problem (7) is to show 120 arrangements given conditions
  sorry

theorem problem8 : cond8 → True := by
  -- Problem (8) is to show 36 arrangements given conditions
  sorry

end problem1_problem2_problem3_problem4_problem5_problem6_problem7_problem8_l1703_170379


namespace dogwood_trees_after_work_l1703_170340

theorem dogwood_trees_after_work 
  (trees_part1 : ℝ) (trees_part2 : ℝ) (trees_part3 : ℝ)
  (trees_cut : ℝ) (trees_planted : ℝ)
  (h1 : trees_part1 = 5.0) (h2 : trees_part2 = 4.0) (h3 : trees_part3 = 6.0)
  (h_cut : trees_cut = 7.0) (h_planted : trees_planted = 3.0) :
  trees_part1 + trees_part2 + trees_part3 - trees_cut + trees_planted = 11.0 :=
by
  sorry

end dogwood_trees_after_work_l1703_170340


namespace evaluate_abs_expression_l1703_170353

noncomputable def approx_pi : ℝ := 3.14159 -- Defining the approximate value of pi

theorem evaluate_abs_expression : |5 * approx_pi - 16| = 0.29205 :=
by
  sorry -- Proof is skipped, as per instructions

end evaluate_abs_expression_l1703_170353


namespace part_1_solution_set_part_2_a_range_l1703_170358

-- Define the function f
def f (x a : ℝ) := |x - a^2| + |x - 2 * a + 1|

-- Part (1)
theorem part_1_solution_set (x : ℝ) : {x | x ≤ 3 / 2 ∨ x ≥ 11 / 2} = 
  {x | f x 2 ≥ 4} :=
sorry

-- Part (2)
theorem part_2_a_range (a : ℝ) : 
  {a | (a - 1)^2 ≥ 4} = {a | a ≤ -1 ∨ a ≥ 3} :=
sorry

end part_1_solution_set_part_2_a_range_l1703_170358


namespace cannot_contain_2003_0_l1703_170398

noncomputable def point_not_on_line (m b : ℝ) (h : m * b < 0) : Prop :=
  ∀ y : ℝ, ¬(0 = 2003 * m + b)

-- Prove that if m and b are real numbers and mb < 0, the line y = mx + b
-- cannot contain the point (2003, 0).
theorem cannot_contain_2003_0 (m b : ℝ) (h : m * b < 0) : point_not_on_line m b h :=
by
  sorry

end cannot_contain_2003_0_l1703_170398


namespace correct_value_calculation_l1703_170359

theorem correct_value_calculation (x : ℤ) (h : 2 * (x + 6) = 28) : 6 * x = 48 :=
by
  -- Proof steps would be here
  sorry

end correct_value_calculation_l1703_170359


namespace kimberly_store_visits_l1703_170368

def peanuts_per_visit : ℕ := 7
def total_peanuts : ℕ := 21

def visits : ℕ := total_peanuts / peanuts_per_visit

theorem kimberly_store_visits : visits = 3 :=
by
  sorry

end kimberly_store_visits_l1703_170368


namespace transformed_sum_of_coordinates_l1703_170372

theorem transformed_sum_of_coordinates (g : ℝ → ℝ) (h : g 8 = 5) :
  let x := 8 / 3
  let y := 14 / 9
  3 * y = g (3 * x) / 3 + 3 ∧ (x + y = 38 / 9) :=
by
  sorry

end transformed_sum_of_coordinates_l1703_170372


namespace second_fraction_correct_l1703_170306

theorem second_fraction_correct : 
  ∃ x : ℚ, (2 / 3) * x * (1 / 3) * (3 / 8) = 0.07142857142857142 ∧ x = 6 / 7 :=
by
  sorry

end second_fraction_correct_l1703_170306


namespace midpoint_of_symmetric_chord_on_ellipse_l1703_170310

theorem midpoint_of_symmetric_chord_on_ellipse
  (A B : ℝ × ℝ) -- coordinates of points A and B
  (hA : (A.1^2 / 16) + (A.2^2 / 4) = 1) -- A lies on the ellipse
  (hB : (B.1^2 / 16) + (B.2^2 / 4) = 1) -- B lies on the ellipse
  (symm : 2 * (A.1 + B.1) / 2 - 2 * (A.2 + B.2) / 2 - 3 = 0) -- A and B are symmetric about the line
  : ((A.1 + B.1) / 2, (A.2 + B.2) / 2) = (2, 1 / 2) :=
  sorry

end midpoint_of_symmetric_chord_on_ellipse_l1703_170310


namespace Hilt_payment_l1703_170349

def total_cost : ℝ := 2.05
def nickel_value : ℝ := 0.05
def dime_value : ℝ := 0.10

theorem Hilt_payment (n : ℕ) (h : n_n = n ∧ n_d = n) 
  (h_nickel : ℝ := n * nickel_value)
  (h_dime : ℝ := n * dime_value): 
  (n * nickel_value + n * dime_value = total_cost) 
  →  n = 14 :=
by {
  sorry
}

end Hilt_payment_l1703_170349


namespace exists_non_deg_triangle_in_sets_l1703_170350

-- Definitions used directly from conditions in a)
def non_deg_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Main theorem statement
theorem exists_non_deg_triangle_in_sets (S : Fin 100 → Set ℕ) (h_disjoint : ∀ i j : Fin 100, i ≠ j → Disjoint (S i) (S j))
  (h_union : (⋃ i, S i) = {x | 1 ≤ x ∧ x ≤ 400}) :
  ∃ i : Fin 100, ∃ a b c : ℕ, a ∈ S i ∧ b ∈ S i ∧ c ∈ S i ∧ non_deg_triangle a b c := sorry

end exists_non_deg_triangle_in_sets_l1703_170350


namespace sum_first_3n_terms_l1703_170317

-- Geometric Sequence: Sum of first n terms Sn, first 2n terms S2n, first 3n terms S3n.
variables {n : ℕ} {S : ℕ → ℕ}

-- Conditions
def sum_first_n_terms (S : ℕ → ℕ) (n : ℕ) : Prop := S n = 48
def sum_first_2n_terms (S : ℕ → ℕ) (n : ℕ) : Prop := S (2 * n) = 60

-- Theorem to Prove
theorem sum_first_3n_terms {S : ℕ → ℕ} (h1 : sum_first_n_terms S n) (h2 : sum_first_2n_terms S n) :
  S (3 * n) = 63 :=
sorry

end sum_first_3n_terms_l1703_170317


namespace max_e_of_conditions_l1703_170320

theorem max_e_of_conditions (a b c d e : ℝ) 
  (h1 : a + b + c + d + e = 8) 
  (h2 : a^2 + b^2 + c^2 + d^2 + e^2 = 16) : 
  e ≤ (16 / 5) :=
by 
  sorry

end max_e_of_conditions_l1703_170320


namespace year_2013_is_not_lucky_l1703_170343

-- Definitions based on conditions
def last_two_digits (year : ℕ) : ℕ := year % 100

def is_valid_date (month : ℕ) (day : ℕ) (year : ℕ) : Prop :=
  month * day = last_two_digits year

def is_lucky_year (year : ℕ) : Prop :=
  ∃ (month : ℕ) (day : ℕ), month <= 12 ∧ day <= 12 ∧ is_valid_date month day year

-- The main statement to prove
theorem year_2013_is_not_lucky : ¬ is_lucky_year 2013 :=
by {
  sorry
}

end year_2013_is_not_lucky_l1703_170343


namespace eight_digit_number_div_by_9_l1703_170392

theorem eight_digit_number_div_by_9 (n : ℕ) (hn : 0 ≤ n ∧ n ≤ 9)
  (h : (8 + 5 + 4 + n + 5 + 2 + 6 + 8) % 9 = 0) : n = 7 :=
by
  sorry

end eight_digit_number_div_by_9_l1703_170392


namespace number_of_club_members_l1703_170336

theorem number_of_club_members
  (num_committee : ℕ)
  (pair_of_committees_has_unique_member : ∀ (c1 c2 : Fin num_committee), c1 ≠ c2 → ∃! m : ℕ, c1 ≠ c2 ∧ c2 ≠ c1 ∧ m = m)
  (members_belong_to_two_committees : ∀ m : ℕ, ∃ (c1 c2 : Fin num_committee), c1 ≠ c2 ∧ m = m)
  : num_committee = 5 → ∃ (num_members : ℕ), num_members = 10 :=
by
  sorry

end number_of_club_members_l1703_170336


namespace remaining_dimes_l1703_170312

-- Conditions
def initial_pennies : Nat := 7
def initial_dimes : Nat := 8
def borrowed_dimes : Nat := 4

-- Define the theorem
theorem remaining_dimes : initial_dimes - borrowed_dimes = 4 := by
  -- Use the conditions to state the remaining dimes
  sorry

end remaining_dimes_l1703_170312


namespace value_of_expression_eq_34_l1703_170387

theorem value_of_expression_eq_34 : (2 - 6 + 10 - 14 + 18 - 22 + 26 - 30 + 34 - 38 + 42 - 46 + 50 - 54 + 58 - 62 + 66 - 70 + 70) = 34 :=
by
  sorry

end value_of_expression_eq_34_l1703_170387


namespace no_real_roots_of_ffx_eq_ninex_l1703_170393

variable (a : ℝ)
noncomputable def f (x : ℝ) : ℝ :=
  x^2 * Real.log (4*(a+1)/a) / Real.log 2 +
  2 * x * Real.log (2 * a / (a + 1)) / Real.log 2 +
  Real.log ((a + 1)^2 / (4 * a^2)) / Real.log 2

theorem no_real_roots_of_ffx_eq_ninex (a : ℝ) (h_pos : ∀ x, 1 ≤ x → f a x > 0) :
  ¬ ∃ x, 1 ≤ x ∧ f a (f a x) = 9 * x :=
  sorry

end no_real_roots_of_ffx_eq_ninex_l1703_170393


namespace remaining_three_digit_numbers_l1703_170346

def is_valid_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def is_invalid_number (n : ℕ) : Prop :=
  ∃ (A B : ℕ), A ≠ B ∧ B ≠ 0 ∧ n = 100 * A + 10 * B + A

def count_valid_three_digit_numbers : ℕ :=
  let total_numbers := 900
  let invalid_numbers := 10 * 9
  total_numbers - invalid_numbers

theorem remaining_three_digit_numbers : count_valid_three_digit_numbers = 810 := by
  sorry

end remaining_three_digit_numbers_l1703_170346


namespace percentage_lower_grades_have_cars_l1703_170389

-- Definitions for the conditions
def n_seniors : ℕ := 300
def p_car : ℚ := 0.50
def n_lower : ℕ := 900
def p_total : ℚ := 0.20

-- Definition for the number of students who have cars in the lower grades
def n_cars_lower : ℚ := 
  let total_students := n_seniors + n_lower
  let total_cars := p_total * total_students
  total_cars - (p_car * n_seniors)

-- Prove the percentage of freshmen, sophomores, and juniors who have cars
theorem percentage_lower_grades_have_cars : 
  (n_cars_lower / n_lower) * 100 = 10 := 
by sorry

end percentage_lower_grades_have_cars_l1703_170389


namespace solve_for_x_and_y_l1703_170329

theorem solve_for_x_and_y (x y : ℝ) 
  (h1 : 0.75 / x = 7 / 8)
  (h2 : x / y = 5 / 6) :
  x = 6 / 7 ∧ y = (6 / 7 * 6) / 5 :=
by
  sorry

end solve_for_x_and_y_l1703_170329


namespace interior_angle_sum_of_regular_polygon_l1703_170348

theorem interior_angle_sum_of_regular_polygon (h: ∀ θ, θ = 45) :
  ∃ s, s = 1080 := by
  sorry

end interior_angle_sum_of_regular_polygon_l1703_170348


namespace base_four_30121_eq_793_l1703_170344

-- Definition to convert a base-four (radix 4) number 30121_4 to its base-ten equivalent
def base_four_to_base_ten (d4 d3 d2 d1 d0 : ℕ) : ℕ :=
  d4 * 4^4 + d3 * 4^3 + d2 * 4^2 + d1 * 4^1 + d0 * 4^0

theorem base_four_30121_eq_793 : base_four_to_base_ten 3 0 1 2 1 = 793 := 
by
  sorry

end base_four_30121_eq_793_l1703_170344


namespace machine_A_produces_1_sprockets_per_hour_l1703_170307

namespace SprocketsProduction

variable {A T : ℝ} -- A: sprockets per hour of machine A, T: hours it takes for machine Q to produce 110 sprockets

-- Given conditions
axiom machine_Q_production_rate : 110 / T = 1.10 * A
axiom machine_P_production_rate : 110 / (T + 10) = A

-- The target theorem to prove
theorem machine_A_produces_1_sprockets_per_hour (h1 : 110 / T = 1.10 * A) (h2 : 110 / (T + 10) = A) : A = 1 :=
by sorry

end SprocketsProduction

end machine_A_produces_1_sprockets_per_hour_l1703_170307


namespace solve_for_w_l1703_170373

theorem solve_for_w (w : ℝ) : (2 : ℝ)^(2 * w) = (8 : ℝ)^(w - 4) → w = 12 := by
  sorry

end solve_for_w_l1703_170373


namespace sum_of_altitudes_l1703_170394

theorem sum_of_altitudes (x y : ℝ) (h : 12 * x + 5 * y = 60) :
  let a := (if y = 0 then x else 0)
  let b := (if x = 0 then y else 0)
  let c := (60 / (Real.sqrt (12^2 + 5^2)))
  a + b + c = 281 / 13 :=
sorry

end sum_of_altitudes_l1703_170394


namespace g_42_value_l1703_170384

noncomputable def g : ℕ → ℕ := sorry

axiom g_increasing (n : ℕ) (hn : n > 0) : g (n + 1) > g n
axiom g_multiplicative (m n : ℕ) (hm : m > 0) (hn : n > 0) : g (m * n) = g m * g n
axiom g_property_iii (m n : ℕ) (hm : m > 0) (hn : n > 0) : (m ≠ n ∧ m^n = n^m) → (g m = n ∨ g n = m)

theorem g_42_value : g 42 = 4410 :=
by
  sorry

end g_42_value_l1703_170384


namespace num_first_and_second_year_students_total_l1703_170380

-- Definitions based on conditions
def num_sampled_students : ℕ := 55
def num_first_year_students_sampled : ℕ := 10
def num_second_year_students_sampled : ℕ := 25
def num_third_year_students_total : ℕ := 400

-- Given that 20 students from the third year are sampled
def num_third_year_students_sampled := num_sampled_students - num_first_year_students_sampled - num_second_year_students_sampled

-- Proportion equality condition
theorem num_first_and_second_year_students_total (x : ℕ) :
  20 / 55 = 400 / (x + num_third_year_students_total) →
  x = 700 :=
by
  sorry

end num_first_and_second_year_students_total_l1703_170380


namespace cevian_sum_equals_two_l1703_170397

-- Definitions based on conditions
variables {A B C D E F O : Type*}
variables (AD BE CF : ℝ) (R : ℝ)
variables (circumcenter_O : O = circumcenter A B C)
variables (intersect_AD_O : AD = abs ((line A D).proj O))
variables (intersect_BE_O : BE = abs ((line B E).proj O))
variables (intersect_CF_O : CF = abs ((line C F).proj O))

-- Prove the main statement
theorem cevian_sum_equals_two (h : circumcenter_O ∧ intersect_AD_O ∧ intersect_BE_O ∧ intersect_CF_O) :
  1 / AD + 1 / BE + 1 / CF = 2 / R :=
sorry

end cevian_sum_equals_two_l1703_170397


namespace limit_of_sequence_l1703_170311

noncomputable def limit_problem := 
  ∀ ε > 0, ∃ N : ℕ, ∀ n > N, |((2 * n - 3) / (n + 2) : ℝ) - 2| < ε

theorem limit_of_sequence : limit_problem :=
sorry

end limit_of_sequence_l1703_170311


namespace obtuse_triangle_iff_distinct_real_roots_l1703_170385

theorem obtuse_triangle_iff_distinct_real_roots
  (A B C : ℝ)
  (h_triangle : 2 * A + B = Real.pi)
  (h_isosceles : A = C) :
  (B > Real.pi / 2) ↔ (B^2 - 4 * A * C > 0) :=
sorry

end obtuse_triangle_iff_distinct_real_roots_l1703_170385


namespace combination_count_l1703_170356

-- Definitions from conditions
def packagingPapers : Nat := 10
def ribbons : Nat := 4
def stickers : Nat := 5

-- Proof problem statement
theorem combination_count : packagingPapers * ribbons * stickers = 200 := 
by
  sorry

end combination_count_l1703_170356


namespace determine_digit_z_l1703_170381

noncomputable def ends_with_k_digits (n : ℕ) (d :ℕ) (k : ℕ) : Prop :=
  ∃ m, m ≥ 1 ∧ (10^k * m + d = n % 10^(k + 1))

noncomputable def decimal_ends_with_digits (z k n : ℕ) : Prop :=
  ends_with_k_digits (n^9) z k

theorem determine_digit_z :
  (z = 9) ↔ ∀ k ≥ 1, ∃ n ≥ 1, decimal_ends_with_digits z k n :=
by
  sorry

end determine_digit_z_l1703_170381


namespace radius_of_curvature_correct_l1703_170302

open Real

noncomputable def radius_of_curvature_squared (a b t_0 : ℝ) : ℝ :=
  (a^2 * sin t_0^2 + b^2 * cos t_0^2)^3 / (a^2 * b^2)

theorem radius_of_curvature_correct (a b t_0 : ℝ) (h : a > 0) (h₁ : b > 0) :
  radius_of_curvature_squared a b t_0 = (a^2 * sin t_0^2 + b^2 * cos t_0^2)^3 / (a^2 * b^2) :=
sorry

end radius_of_curvature_correct_l1703_170302


namespace find_t_l1703_170321

theorem find_t (t : ℝ) : (∃ y : ℝ, y = -(t - 1) ∧ 2 * y - 4 = 3 * (y - 2)) ↔ t = -1 :=
by sorry

end find_t_l1703_170321


namespace odd_number_representation_l1703_170390

theorem odd_number_representation (n : ℤ) : 
  (∃ m : ℤ, 2 * m + 1 = 2 * n + 3) ∧ (¬ ∃ m : ℤ, 2 * m + 1 = 4 * n - 1) :=
by
  -- Proof steps would go here
  sorry

end odd_number_representation_l1703_170390


namespace min_rain_fourth_day_l1703_170355

def rain_overflow_problem : Prop :=
    let holding_capacity := 6 * 12 -- in inches
    let drainage_per_day := 3 -- in inches
    let rainfall_day1 := 10 -- in inches
    let rainfall_day2 := 2 * rainfall_day1 -- 20 inches
    let rainfall_day3 := 1.5 * rainfall_day2 -- 30 inches
    let total_rain_three_days := rainfall_day1 + rainfall_day2 + rainfall_day3 -- 60 inches
    let total_drainage_three_days := 3 * drainage_per_day -- 9 inches
    let remaining_capacity := holding_capacity - (total_rain_three_days - total_drainage_three_days) -- 21 inches
    (remaining_capacity = 21)

theorem min_rain_fourth_day : rain_overflow_problem := sorry

end min_rain_fourth_day_l1703_170355


namespace triangle_relations_l1703_170395

theorem triangle_relations (A B C_1 C_2 C_3 : ℝ)
  (h1 : B > A)
  (h2 : C_2 > C_1 ∧ C_2 > C_3)
  (h3 : A + C_1 = 90) 
  (h4 : C_2 = 90)
  (h5 : B + C_3 = 90) :
  C_1 - C_3 = B - A :=
sorry

end triangle_relations_l1703_170395


namespace situation1_correct_situation2_correct_situation3_correct_l1703_170347

noncomputable def situation1 : Nat :=
  let choices_for_A := 4
  let remaining_perm := Nat.factorial 6
  choices_for_A * remaining_perm

theorem situation1_correct : situation1 = 2880 := by
  sorry

noncomputable def situation2 : Nat :=
  let permutations_A_B := Nat.factorial 2
  let remaining_perm := Nat.factorial 5
  permutations_A_B * remaining_perm

theorem situation2_correct : situation2 = 240 := by
  sorry

noncomputable def situation3 : Nat :=
  let perm_boys := Nat.factorial 3
  let perm_girls := Nat.factorial 4
  perm_boys * perm_girls

theorem situation3_correct : situation3 = 144 := by
  sorry

end situation1_correct_situation2_correct_situation3_correct_l1703_170347


namespace positive_difference_g_b_values_l1703_170308

noncomputable def g (n : ℤ) : ℤ :=
if n < 0 then n^2 + 5 * n + 6 else 3 * n - 30

theorem positive_difference_g_b_values : 
  let g_neg_3 := g (-3)
  let g_3 := g 3
  g_neg_3 = 0 → g_3 = -21 → 
  ∃ b1 b2 : ℤ, g_neg_3 + g_3 + g b1 = 0 ∧ g_neg_3 + g_3 + g b2 = 0 ∧ 
  b1 ≠ b2 ∧ b1 < b2 ∧ b1 < 0 ∧ b2 > 0 ∧ b2 - b1 = 22 :=
by
  sorry

end positive_difference_g_b_values_l1703_170308


namespace boat_speed_in_still_water_eq_16_l1703_170376

theorem boat_speed_in_still_water_eq_16 (stream_rate : ℝ) (time_downstream : ℝ) (distance_downstream : ℝ) (V_b : ℝ) 
(h1 : stream_rate = 5) (h2 : time_downstream = 6) (h3 : distance_downstream = 126) : 
  V_b = 16 :=
by sorry

end boat_speed_in_still_water_eq_16_l1703_170376


namespace vertex_difference_l1703_170366

theorem vertex_difference (n m : ℝ) : 
  ∀ x : ℝ, (∀ x, -x^2 + 2*x + n = -((x - m)^2) + 1) → m - n = 1 := 
by 
  sorry

end vertex_difference_l1703_170366


namespace sugar_solution_l1703_170339

theorem sugar_solution (V x : ℝ) (h1 : V > 0) (h2 : 0.1 * (V - x) + 0.5 * x = 0.2 * V) : x / V = 1 / 4 :=
by sorry

end sugar_solution_l1703_170339


namespace scientific_notation_280000_l1703_170326

theorem scientific_notation_280000 : (280000 : ℝ) = 2.8 * 10^5 :=
sorry

end scientific_notation_280000_l1703_170326


namespace rhombic_dodecahedron_surface_area_rhombic_dodecahedron_volume_l1703_170357

noncomputable def surface_area_rhombic_dodecahedron (a : ℝ) : ℝ :=
  6 * (a ^ 2) * Real.sqrt 2

noncomputable def volume_rhombic_dodecahedron (a : ℝ) : ℝ :=
  2 * (a ^ 3)

theorem rhombic_dodecahedron_surface_area (a : ℝ) :
  surface_area_rhombic_dodecahedron a = 6 * (a ^ 2) * Real.sqrt 2 :=
by
  sorry

theorem rhombic_dodecahedron_volume (a : ℝ) :
  volume_rhombic_dodecahedron a = 2 * (a ^ 3) :=
by
  sorry

end rhombic_dodecahedron_surface_area_rhombic_dodecahedron_volume_l1703_170357


namespace sufficient_but_not_necessary_condition_l1703_170351

theorem sufficient_but_not_necessary_condition (x y : ℝ) :
  (x = 1 ∧ y = 1 → x + y = 2) ∧ (¬(x + y = 2 → x = 1 ∧ y = 1)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l1703_170351


namespace total_amount_invested_l1703_170333

theorem total_amount_invested (x y total : ℝ) (h1 : 0.10 * x - 0.08 * y = 83) (h2 : y = 650) : total = 2000 :=
sorry

end total_amount_invested_l1703_170333


namespace similar_triangle_perimeter_l1703_170399

/-
  Given an isosceles triangle with two equal sides of 18 inches and a base of 12 inches, 
  and a similar triangle with the shortest side of 30 inches, 
  prove that the perimeter of the similar triangle is 120 inches.
-/

def is_isosceles (a b c : ℕ) : Prop :=
  (a = b ∨ b = c ∨ a = c) ∧ a + b > c ∧ b + c > a ∧ c + a > b

theorem similar_triangle_perimeter
  (a b c : ℕ) (a' b' c' : ℕ) (h1 : is_isosceles a b c)
  (h2 : a = 12) (h3 : b = 18) (h4 : c = 18)
  (h5 : a' = 30) (h6 : a' * 18 = a * b')
  (h7 : a' * 18 = a * c') :
  a' + b' + c' = 120 :=
by {
  sorry
}

end similar_triangle_perimeter_l1703_170399


namespace total_distance_l1703_170391

/--
A man completes a journey in 30 hours. He travels the first half of the journey at the rate of 20 km/hr and 
the second half at the rate of 10 km/hr. Prove that the total journey is 400 km.
-/
theorem total_distance (D : ℝ) (h : D / 40 + D / 20 = 30) :
  D = 400 :=
sorry

end total_distance_l1703_170391


namespace distance_between_first_and_last_student_l1703_170367

theorem distance_between_first_and_last_student 
  (n : ℕ) (d : ℕ)
  (students : n = 30) 
  (distance_between_students : d = 3) : 
  n - 1 * d = 87 := 
by
  sorry

end distance_between_first_and_last_student_l1703_170367


namespace functional_equation_solution_l1703_170304

-- Define the functional equation with given conditions
def func_eq (f : ℤ → ℝ) (N : ℕ) : Prop :=
  (∀ k : ℤ, f (2 * k) = 2 * f k) ∧
  (∀ k : ℤ, f (N - k) = f k)

-- State the mathematically equivalent proof problem
theorem functional_equation_solution (N : ℕ) (f : ℤ → ℝ) 
  (h1 : ∀ k : ℤ, f (2 * k) = 2 * f k)
  (h2 : ∀ k : ℤ, f (N - k) = f k) : 
  ∀ a : ℤ, f a = 0 := 
sorry

end functional_equation_solution_l1703_170304


namespace seat_adjustment_schemes_l1703_170322

theorem seat_adjustment_schemes {n k : ℕ} (h1 : n = 7) (h2 : k = 3) :
  (2 * Nat.choose n k) = 70 :=
by
  -- n is the number of people, k is the number chosen
  rw [h1, h2]
  -- the rest is skipped for the statement only
  sorry

end seat_adjustment_schemes_l1703_170322


namespace find_quotient_l1703_170300

theorem find_quotient (dividend divisor remainder quotient : ℕ) 
  (h_dividend : dividend = 171) 
  (h_divisor : divisor = 21) 
  (h_remainder : remainder = 3) 
  (h_div_eq : dividend = divisor * quotient + remainder) :
  quotient = 8 :=
by sorry

end find_quotient_l1703_170300


namespace sum_of_coeffs_is_minus_one_l1703_170319

theorem sum_of_coeffs_is_minus_one 
  (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℤ) :
  (∀ x : ℤ, (1 - x^3)^3 = a + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5 + a₆ * x^6 + a₇ * x^7 + a₈ * x^8 + a₉ * x^9)
  → a = 1 
  → a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ = -1 :=
by
  sorry

end sum_of_coeffs_is_minus_one_l1703_170319


namespace darry_total_steps_l1703_170331

def largest_ladder_steps : ℕ := 20
def largest_ladder_times : ℕ := 12

def medium_ladder_steps : ℕ := 15
def medium_ladder_times : ℕ := 8

def smaller_ladder_steps : ℕ := 10
def smaller_ladder_times : ℕ := 10

def smallest_ladder_steps : ℕ := 5
def smallest_ladder_times : ℕ := 15

theorem darry_total_steps :
  (largest_ladder_steps * largest_ladder_times)
  + (medium_ladder_steps * medium_ladder_times)
  + (smaller_ladder_steps * smaller_ladder_times)
  + (smallest_ladder_steps * smallest_ladder_times)
  = 535 := by
  sorry

end darry_total_steps_l1703_170331


namespace sample_size_is_correct_l1703_170361

-- Define the school and selection conditions
def total_classes := 40
def students_per_class := 50

-- Given condition
def selected_students := 150

-- Theorem statement
theorem sample_size_is_correct : selected_students = 150 := 
by 
  sorry

end sample_size_is_correct_l1703_170361


namespace max_ab_l1703_170301

theorem max_ab (a b : ℝ) (h1 : a + 4 * b = 8) (h2 : a > 0) (h3 : b > 0) : ab ≤ 4 := 
sorry

end max_ab_l1703_170301


namespace product_11_29_product_leq_20_squared_product_leq_half_m_squared_l1703_170332

-- Definition of natural numbers
variable (a b m : ℕ)

-- Statement 1: Prove that 11 × 29 = 20^2 - 9^2
theorem product_11_29 : 11 * 29 = 20^2 - 9^2 := sorry

-- Statement 2: Prove ∀ a, b ∈ ℕ, if a + b = 40, then ab ≤ 20^2.
theorem product_leq_20_squared (a b : ℕ) (h : a + b = 40) : a * b ≤ 20^2 := sorry

-- Statement 3: Prove ∀ a, b ∈ ℕ, if a + b = m, then ab ≤ (m/2)^2.
theorem product_leq_half_m_squared (a b : ℕ) (m : ℕ) (h : a + b = m) : a * b ≤ (m / 2)^2 := sorry

end product_11_29_product_leq_20_squared_product_leq_half_m_squared_l1703_170332


namespace equality_holds_iff_l1703_170386

theorem equality_holds_iff (k t x y z : ℤ) (h_arith_prog : x + z = 2 * y) :
  (k * y^3 = x^3 + z^3) ↔ (k = 2 * (3 * t^2 + 1)) := by
  sorry

end equality_holds_iff_l1703_170386


namespace total_puppies_adopted_l1703_170335

-- Define the number of puppies adopted each week
def first_week_puppies : ℕ := 20
def second_week_puppies : ℕ := (2 / 5) * first_week_puppies
def third_week_puppies : ℕ := 2 * second_week_puppies
def fourth_week_puppies : ℕ := 10 + first_week_puppies

-- Prove that the total number of puppies adopted over the month is 74
theorem total_puppies_adopted : 
  first_week_puppies + second_week_puppies + third_week_puppies + fourth_week_puppies = 74 := by
  sorry

end total_puppies_adopted_l1703_170335


namespace sum_roots_x_squared_minus_5x_plus_6_eq_5_l1703_170383

noncomputable def sum_of_roots (a b c : Real) : Real :=
  -b / a

theorem sum_roots_x_squared_minus_5x_plus_6_eq_5 :
  sum_of_roots 1 (-5) 6 = 5 := by
  sorry

end sum_roots_x_squared_minus_5x_plus_6_eq_5_l1703_170383


namespace value_of_x_minus_y_l1703_170370

theorem value_of_x_minus_y (x y a : ℝ) (h₁ : x + y > 0) (h₂ : a < 0) (h₃ : a * y > 0) : x - y > 0 :=
sorry

end value_of_x_minus_y_l1703_170370


namespace servings_in_one_week_l1703_170345

theorem servings_in_one_week (daily_servings : ℕ) (days_in_week : ℕ) (total_servings : ℕ)
  (h1 : daily_servings = 3)
  (h2 : days_in_week = 7)
  (h3 : total_servings = daily_servings * days_in_week) :
  total_servings = 21 := by
  sorry

end servings_in_one_week_l1703_170345


namespace right_triangle_area_l1703_170313

theorem right_triangle_area (a b c : ℝ) (h1 : c = 13) (h2 : a = 5) (h3 : a^2 + b^2 = c^2) : 0.5 * a * b = 30 := by
  sorry

end right_triangle_area_l1703_170313


namespace judy_pencil_cost_l1703_170362

theorem judy_pencil_cost 
  (pencils_per_week : ℕ)
  (days_per_week : ℕ)
  (pack_cost : ℕ)
  (pack_size : ℕ)
  (total_days : ℕ)
  (pencil_usage : pencils_per_week = 10)
  (school_days : days_per_week = 5)
  (cost_per_pack : pack_cost = 4)
  (pencils_per_pack : pack_size = 30)
  (duration : total_days = 45) : 
  ∃ (total_cost : ℕ), total_cost = 12 :=
sorry

end judy_pencil_cost_l1703_170362


namespace not_difference_of_squares_10_l1703_170396

theorem not_difference_of_squares_10 (a b : ℤ) : a^2 - b^2 ≠ 10 :=
sorry

end not_difference_of_squares_10_l1703_170396


namespace arithmetic_sequence_sum_l1703_170352

variable (a : ℕ → ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_sum 
  (h_arith : is_arithmetic_sequence a)
  (h_condition : a 2 + a 6 = 37) : 
  a 1 + a 3 + a 5 + a 7 = 74 :=
  sorry

end arithmetic_sequence_sum_l1703_170352


namespace q0_r0_eq_three_l1703_170315

variable (p q r s : Polynomial ℝ)
variable (hp_const : p.coeff 0 = 2)
variable (hs_eq : s = p * q * r)
variable (hs_const : s.coeff 0 = 6)

theorem q0_r0_eq_three : (q.coeff 0) * (r.coeff 0) = 3 := by
  sorry

end q0_r0_eq_three_l1703_170315


namespace number_property_l1703_170324

theorem number_property (n : ℕ) (h : n = 7101449275362318840579) :
  n / 7 = 101449275362318840579 :=
sorry

end number_property_l1703_170324


namespace trigonometric_identity_l1703_170377

variable (α β : Real) 

theorem trigonometric_identity (h₁ : Real.tan (α + β) = 1) 
                              (h₂ : Real.tan (α - β) = 2) 
                              : (Real.sin (2 * α)) / (Real.cos (2 * β)) = 1 := 
by 
  sorry

end trigonometric_identity_l1703_170377


namespace garden_bed_length_l1703_170354

theorem garden_bed_length (total_area : ℕ) (garden_area : ℕ) (width : ℕ) (n : ℕ)
  (total_area_eq : total_area = 42)
  (garden_area_eq : garden_area = 9)
  (num_gardens_eq : n = 2)
  (width_eq : width = 3)
  (lhs_eq : lhs = total_area - n * garden_area)
  (area_to_length_eq : length = lhs / width) :
  length = 8 := by
  sorry

end garden_bed_length_l1703_170354


namespace percentage_defective_l1703_170378

theorem percentage_defective (examined rejected : ℚ) (h1 : examined = 66.67) (h2 : rejected = 10) :
  (rejected / examined) * 100 = 15 := by
  sorry

end percentage_defective_l1703_170378


namespace complete_the_square_l1703_170337

theorem complete_the_square (x : ℝ) : x^2 - 2 * x - 1 = 0 -> (x - 1)^2 = 2 := by
  sorry

end complete_the_square_l1703_170337


namespace intersection_of_A_and_B_l1703_170327

-- Given sets A and B
def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 1}

-- Prove the intersection of A and B
theorem intersection_of_A_and_B : A ∩ B = {x | 0 ≤ x ∧ x ≤ 1} := 
by
  sorry

end intersection_of_A_and_B_l1703_170327


namespace jack_age_difference_l1703_170360

def beckett_age : ℕ := 12
def olaf_age : ℕ := beckett_age + 3
def shannen_age : ℕ := olaf_age - 2
def total_age : ℕ := 71
def jack_age : ℕ := total_age - (beckett_age + olaf_age + shannen_age)
def difference := jack_age - 2 * shannen_age

theorem jack_age_difference :
  difference = 5 :=
by
  -- Math proof goes here
  sorry

end jack_age_difference_l1703_170360


namespace certain_amount_l1703_170388

theorem certain_amount (x : ℝ) (A : ℝ) (h1: x = 900) (h2: 0.25 * x = 0.15 * 1600 - A) : A = 15 :=
by
  sorry

end certain_amount_l1703_170388


namespace line_segment_endpoint_l1703_170342

theorem line_segment_endpoint (x : ℝ) (h1 : (x - 3)^2 + 36 = 289) (h2 : x < 0) : x = 3 - Real.sqrt 253 :=
sorry

end line_segment_endpoint_l1703_170342


namespace find_m_l1703_170341

theorem find_m (a0 a1 a2 a3 a4 a5 a6 : ℝ) (m : ℝ)
  (h1 : (1 + m) * x ^ 6 = a0 + a1 * x + a2 * x ^ 2 + a3 * x ^ 3 + a4 * x ^ 4 + a5 * x ^ 5 + a6 * x ^ 6)
  (h2 : a1 - a2 + a3 - a4 + a5 - a6 = -63)
  (h3 : a0 = 1) :
  m = 3 ∨ m = -1 :=
by
  sorry

end find_m_l1703_170341


namespace geometric_sequence_a4_value_l1703_170371

variable {α : Type} [LinearOrderedField α]

noncomputable def is_geometric_sequence (a : ℕ → α) : Prop :=
∀ n m : ℕ, n < m → ∃ r : α, 0 < r ∧ a m = a n * r^(m - n)

theorem geometric_sequence_a4_value (a : ℕ → α)
  (pos : ∀ n, 0 < a n)
  (geo_seq : is_geometric_sequence a)
  (h : a 1 * a 7 = 36) :
  a 4 = 6 :=
by 
  sorry

end geometric_sequence_a4_value_l1703_170371


namespace calculate_expression_l1703_170316

theorem calculate_expression : ( (3 / 20 + 5 / 200 + 7 / 2000) * 2 = 0.357 ) :=
by
  sorry

end calculate_expression_l1703_170316


namespace smaller_angle_clock_8_10_l1703_170374

/-- The measure of the smaller angle formed by the hour and minute hands of a clock at 8:10 p.m. is 175 degrees. -/
theorem smaller_angle_clock_8_10 : 
  let full_circle := 360
  let hour_increment := 30
  let hour_angle_8 := 8 * hour_increment
  let minute_angle_increment := 6
  let hour_hand_adjustment := 10 * (hour_increment / 60)
  let hour_hand_position := hour_angle_8 + hour_hand_adjustment
  let minute_hand_position := 10 * minute_angle_increment
  let angle_difference := if hour_hand_position > minute_hand_position 
                          then hour_hand_position - minute_hand_position 
                          else minute_hand_position - hour_hand_position  
  let smaller_angle := if 2 * angle_difference > full_circle 
                       then full_circle - angle_difference 
                       else angle_difference
  smaller_angle = 175 :=
by 
  sorry

end smaller_angle_clock_8_10_l1703_170374


namespace total_fruits_l1703_170323

def num_papaya_trees : ℕ := 2
def num_mango_trees : ℕ := 3
def papayas_per_tree : ℕ := 10
def mangos_per_tree : ℕ := 20

theorem total_fruits : (num_papaya_trees * papayas_per_tree) + (num_mango_trees * mangos_per_tree) = 80 := 
by
  sorry

end total_fruits_l1703_170323


namespace algebraic_expression_is_200_l1703_170365

-- Define the condition
def satisfies_ratio (x : ℕ) : Prop :=
  x / 10 = 20

-- The proof problem statement
theorem algebraic_expression_is_200 : ∃ x : ℕ, satisfies_ratio x ∧ x = 200 :=
by
  -- Providing the necessary proof infrastructure
  use 200
  -- Assuming the proof is correct
  sorry


end algebraic_expression_is_200_l1703_170365


namespace transform_cos_function_l1703_170364

theorem transform_cos_function :
  ∀ x : ℝ, 2 * Real.cos (x + π / 3) =
           2 * Real.cos (2 * (x - π / 12) + π / 6) := 
sorry

end transform_cos_function_l1703_170364
