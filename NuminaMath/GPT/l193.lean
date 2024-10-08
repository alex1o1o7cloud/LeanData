import Mathlib

namespace value_of_p_l193_193319

theorem value_of_p (p q : ℝ) (h1 : q = (2 / 5) * p) (h2 : p * q = 90) : p = 15 :=
by
  sorry

end value_of_p_l193_193319


namespace train_speed_l193_193794

theorem train_speed (length : ℝ) (time : ℝ) (length_eq : length = 375.03) (time_eq : time = 5) :
  let speed_kmph := (length / 1000) / (time / 3600)
  speed_kmph = 270.02 :=
by
  sorry

end train_speed_l193_193794


namespace total_students_l193_193582

theorem total_students (initial_candies leftover_candies girls boys : ℕ) (h1 : initial_candies = 484)
  (h2 : leftover_candies = 4) (h3 : boys = girls + 3) (h4 : (2 * girls + boys) * (2 * girls + boys) = initial_candies - leftover_candies) :
  2 * girls + boys = 43 :=
  sorry

end total_students_l193_193582


namespace joeys_age_next_multiple_l193_193059

-- Definitions of the conditions and problem setup
def joey_age (chloe_age : ℕ) : ℕ := chloe_age + 2
def max_age : ℕ := 2
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Sum of digits function
def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

-- Main Lean statement
theorem joeys_age_next_multiple (chloe_age : ℕ) (H1 : is_prime chloe_age)
  (H2 : ∀ n : ℕ, (joey_age chloe_age + n) % (max_age + n) = 0)
  (H3 : ∀ i : ℕ, i < 11 → is_prime (chloe_age + i))
  : sum_of_digits (joey_age chloe_age + 1) = 5 :=
  sorry

end joeys_age_next_multiple_l193_193059


namespace rational_square_of_1_minus_xy_l193_193108

theorem rational_square_of_1_minus_xy (x y : ℚ) (h : x^5 + y^5 = 2 * x^2 * y^2) : ∃ (q : ℚ), 1 - x * y = q^2 :=
by
  sorry

end rational_square_of_1_minus_xy_l193_193108


namespace part1_point_A_value_of_m_part1_area_ABC_part2_max_ordinate_P_l193_193498

noncomputable def quadratic_function (m x : ℝ) : ℝ := (m - 2) * x ^ 2 - x - m ^ 2 + 6 * m - 7

theorem part1_point_A_value_of_m (m : ℝ) (h : quadratic_function m (-1) = 2) : m = 5 :=
sorry

theorem part1_area_ABC (area : ℝ) 
  (h₁ : quadratic_function 5 (1 : ℝ) = 0) 
  (h₂ : quadratic_function 5 (-2/3 : ℝ) = 0) : area = 5 / 3 :=
sorry

theorem part2_max_ordinate_P (m : ℝ) (h : - (m - 3) ^ 2 + 2 ≤ 2) : m = 3 :=
sorry

end part1_point_A_value_of_m_part1_area_ABC_part2_max_ordinate_P_l193_193498


namespace tail_count_likelihood_draw_and_rainy_l193_193994

def coin_tosses : ℕ := 25
def heads_count : ℕ := 11
def draws_when_heads : ℕ := 7
def rainy_when_tails : ℕ := 4

theorem tail_count :
  coin_tosses - heads_count = 14 :=
sorry

theorem likelihood_draw_and_rainy :
  0 = 0 :=
sorry

end tail_count_likelihood_draw_and_rainy_l193_193994


namespace sufficient_and_necessary_condition_for_positive_sum_l193_193732

variable (q : ℤ) (a1 : ℤ)

def geometric_sequence (n : ℕ) : ℤ := a1 * q ^ (n - 1)

def sum_of_first_n_terms (n : ℕ) : ℤ :=
  if q = 1 then a1 * n else (a1 * (1 - q ^ n)) / (1 - q)

theorem sufficient_and_necessary_condition_for_positive_sum :
  (a1 > 0) ↔ (sum_of_first_n_terms q a1 2017 > 0) :=
sorry

end sufficient_and_necessary_condition_for_positive_sum_l193_193732


namespace find_x2_plus_y2_l193_193671

theorem find_x2_plus_y2 (x y : ℝ) (h1 : x * y = 12) (h2 : x^2 * y + x * y^2 + x + y = 99) : 
  x^2 + y^2 = 5745 / 169 := 
sorry

end find_x2_plus_y2_l193_193671


namespace smallest_solution_eq_l193_193797

noncomputable def smallest_solution := 4 - Real.sqrt 3

theorem smallest_solution_eq (x : ℝ) : 
  (1 / (x - 3) + 1 / (x - 5) = 3 / (x - 4)) → x = smallest_solution :=
sorry

end smallest_solution_eq_l193_193797


namespace students_not_in_biology_l193_193545

theorem students_not_in_biology (total_students : ℕ) (percentage_in_biology : ℚ) 
  (h1 : total_students = 880) (h2 : percentage_in_biology = 27.5 / 100) : 
  total_students - (total_students * percentage_in_biology) = 638 := 
by
  sorry

end students_not_in_biology_l193_193545


namespace first_three_digits_of_x_are_571_l193_193249

noncomputable def x : ℝ := (10^2003 + 1)^(11/7)

theorem first_three_digits_of_x_are_571 : 
  ∃ d₁ d₂ d₃ : ℕ, 
  (d₁, d₂, d₃) = (5, 7, 1) ∧ 
  ∃ k : ℤ, 
  (x - k : ℝ) * 1000 = d₁ * 100 + d₂ * 10 + d₃ := 
by
  sorry

end first_three_digits_of_x_are_571_l193_193249


namespace quadratic_eq_solutions_l193_193055

open Real

theorem quadratic_eq_solutions (x : ℝ) :
  (2 * x + 1) ^ 2 = (2 * x + 1) * (x - 1) ↔ x = -1 / 2 ∨ x = -2 :=
by sorry

end quadratic_eq_solutions_l193_193055


namespace find_line_eq_l193_193417

-- Define the type for the line equation
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the given conditions
def given_point : ℝ × ℝ := (-3, -1)
def given_parallel_line : Line := { a := 1, b := -3, c := -1 }

-- Define what it means for two lines to be parallel
def are_parallel (L1 L2 : Line) : Prop :=
  L1.a * L2.b = L1.b * L2.a

-- Define what it means for a point to lie on the line
def lies_on_line (P : ℝ × ℝ) (L : Line) : Prop :=
  L.a * P.1 + L.b * P.2 + L.c = 0

-- Define the result line we need to prove
def result_line : Line := { a := 1, b := -3, c := 0 }

-- The final theorem statement
theorem find_line_eq : 
  ∃ (L : Line), are_parallel L given_parallel_line ∧ lies_on_line given_point L ∧ L = result_line := 
sorry

end find_line_eq_l193_193417


namespace smaller_circle_radius_l193_193213

theorem smaller_circle_radius (r R : ℝ) (hR : R = 10) (h : 2 * r = 2 * R) : r = 10 :=
by
  sorry

end smaller_circle_radius_l193_193213


namespace find_f_of_minus_five_l193_193102

theorem find_f_of_minus_five (a b : ℝ) (f : ℝ → ℝ) (h1 : f 5 = 7) (h2 : ∀ x, f x = a * x + b * Real.sin x + 1) : f (-5) = -5 :=
by
  sorry

end find_f_of_minus_five_l193_193102


namespace div_by_3_pow_101_l193_193745

theorem div_by_3_pow_101 : ∀ (n : ℕ), (∀ k : ℕ, (3^(k+1)) ∣ (2^(3^k) + 1)) → 3^101 ∣ 2^(3^100) + 1 :=
by
  sorry

end div_by_3_pow_101_l193_193745


namespace number_of_ordered_pairs_l193_193860

theorem number_of_ordered_pairs :
  ∃ (n : ℕ), n = 99 ∧
  (∀ (a b : ℕ), 1 ≤ a ∧ 1 ≤ b ∧ (Int.gcd a b) * a + b^2 = 10000
  → ∃ (k : ℕ), k = 99) :=
sorry

end number_of_ordered_pairs_l193_193860


namespace factorial_divisibility_l193_193291

theorem factorial_divisibility {n : ℕ} (h : 2011^(2011) ∣ n!) : 2011^(2012) ∣ n! :=
sorry

end factorial_divisibility_l193_193291


namespace down_payment_l193_193686

theorem down_payment {total_loan : ℕ} {monthly_payment : ℕ} {years : ℕ} (h1 : total_loan = 46000) (h2 : monthly_payment = 600) (h3 : years = 5):
  total_loan - (years * 12 * monthly_payment) = 10000 := by
  sorry

end down_payment_l193_193686


namespace instantaneous_velocity_at_3_l193_193577

noncomputable def displacement (t : ℝ) : ℝ := 
  - (1 / 3) * t^3 + 2 * t^2 - 5

theorem instantaneous_velocity_at_3 : 
  (deriv displacement 3 = 3) :=
by
  sorry

end instantaneous_velocity_at_3_l193_193577


namespace smallest_n_for_multiples_of_2015_l193_193422

theorem smallest_n_for_multiples_of_2015 (n : ℕ) (hn : 0 < n)
  (h5 : (2^n - 1) % 5 = 0)
  (h13 : (2^n - 1) % 13 = 0)
  (h31 : (2^n - 1) % 31 = 0) : n = 60 := by
  sorry

end smallest_n_for_multiples_of_2015_l193_193422


namespace no_14_non_square_rectangles_l193_193406

theorem no_14_non_square_rectangles (side_len : ℕ) 
    (h_side_len : side_len = 9) 
    (num_rectangles : ℕ) 
    (h_num_rectangles : num_rectangles = 14) 
    (min_side_len : ℕ → ℕ → Prop) 
    (h_min_side_len : ∀ l w, min_side_len l w → l ≥ 2 ∧ w ≥ 2) : 
    ¬ (∀ l w, min_side_len l w → l ≠ w) :=
by {
    sorry
}

end no_14_non_square_rectangles_l193_193406


namespace average_growth_rate_equation_l193_193109

-- Define the current and target processing capacities
def current_capacity : ℝ := 1000
def target_capacity : ℝ := 1200

-- Define the time period in months
def months : ℕ := 2

-- Define the monthly average growth rate
variable (x : ℝ)

-- The statement to be proven: current capacity increased by the growth rate over 2 months equals the target capacity 
theorem average_growth_rate_equation :
  current_capacity * (1 + x) ^ months = target_capacity :=
sorry

end average_growth_rate_equation_l193_193109


namespace adam_students_in_10_years_l193_193014

-- Define the conditions
def teaches_per_year : Nat := 50
def first_year_students : Nat := 40
def years_teaching : Nat := 10

-- Define the total number of students Adam will teach in 10 years
def total_students (first_year: Nat) (rest_years: Nat) (students_per_year: Nat) : Nat :=
  first_year + (rest_years * students_per_year)

-- State the theorem
theorem adam_students_in_10_years :
  total_students first_year_students (years_teaching - 1) teaches_per_year = 490 :=
by
  sorry

end adam_students_in_10_years_l193_193014


namespace find_m_l193_193825

theorem find_m (a b m : ℝ) (h1 : 2^a = m) (h2 : 5^b = m) (h3 : 1/a + 1/b = 2) : m = Real.sqrt 10 :=
by
  sorry

end find_m_l193_193825


namespace pieces_of_green_candy_l193_193099

theorem pieces_of_green_candy (total_pieces red_pieces blue_pieces : ℝ)
  (h_total : total_pieces = 3409.7)
  (h_red : red_pieces = 145.5)
  (h_blue : blue_pieces = 785.2) :
  total_pieces - red_pieces - blue_pieces = 2479 := by
  sorry

end pieces_of_green_candy_l193_193099


namespace village_household_count_l193_193130

theorem village_household_count
  (H : ℕ)
  (water_per_household_per_month : ℕ := 20)
  (total_water : ℕ := 2000)
  (duration_months : ℕ := 10)
  (total_consumption_condition : water_per_household_per_month * H * duration_months = total_water) :
  H = 10 :=
by
  sorry

end village_household_count_l193_193130


namespace trigonometric_value_l193_193972

theorem trigonometric_value (α : ℝ) (h : Real.tan α = 2) :
  (2 * Real.sin α ^ 2 + 1) / Real.cos (2 * (α - Real.pi / 4)) = 13 / 4 := 
sorry

end trigonometric_value_l193_193972


namespace striped_jerseys_count_l193_193199

-- Define the cost of long-sleeved jerseys
def cost_long_sleeved := 15
-- Define the cost of striped jerseys
def cost_striped := 10
-- Define the number of long-sleeved jerseys bought
def num_long_sleeved := 4
-- Define the total amount spent
def total_spent := 80

-- Define a theorem to prove the number of striped jerseys bought
theorem striped_jerseys_count : ∃ x : ℕ, x * cost_striped = total_spent - num_long_sleeved * cost_long_sleeved ∧ x = 2 := 
by 
-- TODO: The proof steps would go here, but for this exercise, we use 'sorry' to skip the proof.
sorry

end striped_jerseys_count_l193_193199


namespace onion_pieces_per_student_l193_193466

theorem onion_pieces_per_student (total_pizzas : ℕ) (slices_per_pizza : ℕ)
  (cheese_pieces_leftover : ℕ) (onion_pieces_leftover : ℕ) (students : ℕ) (cheese_per_student : ℕ)
  (h1 : total_pizzas = 6) (h2 : slices_per_pizza = 18) (h3 : cheese_pieces_leftover = 8) (h4 : onion_pieces_leftover = 4)
  (h5 : students = 32) (h6 : cheese_per_student = 2) :
  ((total_pizzas * slices_per_pizza) - cheese_pieces_leftover - onion_pieces_leftover - (students * cheese_per_student)) / students = 1 := 
by
  sorry

end onion_pieces_per_student_l193_193466


namespace short_trees_after_planting_l193_193751

-- Define the current number of short trees
def current_short_trees : ℕ := 41

-- Define the number of short trees to be planted today
def new_short_trees : ℕ := 57

-- Define the expected total number of short trees after planting
def total_short_trees_after_planting : ℕ := 98

-- The theorem to prove that the total number of short trees after planting is as expected
theorem short_trees_after_planting :
  current_short_trees + new_short_trees = total_short_trees_after_planting :=
by
  -- Proof skipped using sorry
  sorry

end short_trees_after_planting_l193_193751


namespace all_cells_equal_l193_193487

-- Define the infinite grid
def Grid := ℕ → ℕ → ℕ

-- Define the condition on the grid values
def is_min_mean_grid (g : Grid) : Prop :=
  ∀ i j : ℕ, g i j ≥ (g (i-1) j + g (i+1) j + g i (j-1) + g i (j+1)) / 4

-- Main theorem
theorem all_cells_equal (g : Grid) (h : is_min_mean_grid g) : ∃ a : ℕ, ∀ i j : ℕ, g i j = a := 
sorry

end all_cells_equal_l193_193487


namespace find_nat_pair_l193_193560

theorem find_nat_pair (a b : ℕ) (h₁ : a > 1) (h₂ : b > 1) (h₃ : a = 2^155) (h₄ : b = 3^65) : a^13 * b^31 = 6^2015 :=
by {
  sorry
}

end find_nat_pair_l193_193560


namespace number_of_intersection_points_l193_193021

noncomputable section

-- Define a type for Points in the plane
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Defining the five points
variables (A B C D E : Point)

-- Define the conditions that no three points are collinear
def no_three_collinear (A B C : Point) : Prop :=
  (B.x - A.x) * (C.y - A.y) ≠ (C.x - A.x) * (B.y - A.y)

-- Define the theorem statement
theorem number_of_intersection_points (h1 : no_three_collinear A B C)
  (h2 : no_three_collinear A B D)
  (h3 : no_three_collinear A B E)
  (h4 : no_three_collinear A C D)
  (h5 : no_three_collinear A C E)
  (h6 : no_three_collinear A D E)
  (h7 : no_three_collinear B C D)
  (h8 : no_three_collinear B C E)
  (h9 : no_three_collinear B D E)
  (h10 : no_three_collinear C D E) :
  ∃ (N : ℕ), N = 40 :=
  sorry

end number_of_intersection_points_l193_193021


namespace work_done_days_l193_193549

theorem work_done_days (a_days : ℕ) (b_days : ℕ) (together_days : ℕ) (a_work_done : ℚ) (b_work_done : ℚ) (together_work : ℚ) : 
  a_days = 12 ∧ b_days = 15 ∧ together_days = 5 ∧ 
  a_work_done = 1/12 ∧ b_work_done = 1/15 ∧ together_work = 3/4 → 
  ∃ days : ℚ, a_days > 0 ∧ b_days > 0 ∧ together_days > 0 ∧ days = 3 := 
  sorry

end work_done_days_l193_193549


namespace porter_monthly_earnings_l193_193299

def daily_rate : ℕ := 8

def regular_days : ℕ := 5

def extra_day_rate : ℕ := daily_rate * 3 / 2  -- 50% increase on the daily rate

def weekly_earnings_with_overtime : ℕ := (daily_rate * regular_days) + extra_day_rate

def weeks_in_month : ℕ := 4

theorem porter_monthly_earnings : weekly_earnings_with_overtime * weeks_in_month = 208 :=
by
  sorry

end porter_monthly_earnings_l193_193299


namespace reflection_across_y_axis_coordinates_l193_193304

def coordinates_after_reflection (x y : ℤ) : ℤ × ℤ :=
  (-x, y)

theorem reflection_across_y_axis_coordinates :
  coordinates_after_reflection (-3) 4 = (3, 4) :=
by
  sorry

end reflection_across_y_axis_coordinates_l193_193304


namespace window_side_length_is_five_l193_193179

def pane_width (x : ℝ) : ℝ := x
def pane_height (x : ℝ) : ℝ := 3 * x
def border_width : ℝ := 1
def pane_rows : ℕ := 2
def pane_columns : ℕ := 3

theorem window_side_length_is_five (x : ℝ) (h : pane_height x = 3 * pane_width x) : 
  (3 * x + 4 = 6 * x + 3) -> (3 * x + 4 = 5) :=
by
  intros h1
  sorry

end window_side_length_is_five_l193_193179


namespace smallest_positive_perfect_cube_has_divisor_l193_193412

theorem smallest_positive_perfect_cube_has_divisor (p q r s : ℕ) (hp : Prime p) (hq : Prime q)
  (hr : Prime r) (hs : Prime s) (hpqrs : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s) :
  ∃ n : ℕ, n = (p * q * r * s^2)^3 ∧ ∀ m : ℕ, (m = p^2 * q^3 * r^4 * s^5 → m ∣ n) :=
by
  sorry

end smallest_positive_perfect_cube_has_divisor_l193_193412


namespace maria_gave_towels_l193_193518

def maria_towels (green_white total_left : Nat) : Nat :=
  green_white - total_left

theorem maria_gave_towels :
  ∀ (green white left given : Nat),
    green = 35 →
    white = 21 →
    left = 22 →
    given = 34 →
    maria_towels (green + white) left = given :=
by
  intros green white left given
  intros hgreen hwhite hleft hgiven
  rw [hgreen, hwhite, hleft, hgiven]
  sorry

end maria_gave_towels_l193_193518


namespace find_angle_A_l193_193384

theorem find_angle_A (a b c : ℝ) (A : ℝ) (h : a^2 = b^2 - b * c + c^2) : A = 60 :=
sorry

end find_angle_A_l193_193384


namespace winning_percentage_is_65_l193_193178

theorem winning_percentage_is_65 
  (total_games won_games : ℕ) 
  (h1 : total_games = 280) 
  (h2 : won_games = 182) :
  ((won_games : ℚ) / (total_games : ℚ)) * 100 = 65 :=
by
  sorry

end winning_percentage_is_65_l193_193178


namespace banana_cost_is_2_l193_193672

noncomputable def bananas_cost (B : ℝ) : Prop :=
  let cost_oranges : ℝ := 10 * 1.5
  let total_cost : ℝ := 25
  let cost_bananas : ℝ := total_cost - cost_oranges
  let num_bananas : ℝ := 5
  B = cost_bananas / num_bananas

theorem banana_cost_is_2 : bananas_cost 2 :=
by
  unfold bananas_cost
  sorry

end banana_cost_is_2_l193_193672


namespace translation_result_l193_193107

-- Define the original point M
def M : ℝ × ℝ := (-10, 1)

-- Define the translation on the y-axis by 4 units
def translate_y (p : ℝ × ℝ) (dy : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + dy)

-- Define the resulting point M1 after translation
def M1 : ℝ × ℝ := translate_y M 4

-- The theorem we want to prove: the coordinates of M1 are (-10, 5)
theorem translation_result : M1 = (-10, 5) :=
by
  -- Proof goes here
  sorry

end translation_result_l193_193107


namespace weight_of_first_lift_l193_193375

-- Definitions as per conditions
variables (x y : ℝ)
def condition1 : Prop := x + y = 1800
def condition2 : Prop := 2 * x = y + 300

-- Prove that the weight of Joe's first lift is 700 pounds
theorem weight_of_first_lift (h1 : condition1 x y) (h2 : condition2 x y) : x = 700 :=
by
  sorry

end weight_of_first_lift_l193_193375


namespace xiaozhang_participates_in_martial_arts_l193_193542

theorem xiaozhang_participates_in_martial_arts
  (row : Prop) (shoot : Prop) (martial : Prop)
  (Zhang Wang Li: Prop → Prop)
  (H1 : ¬  Zhang row ∧ ¬ Wang row)
  (H2 : ∃ (n m : ℕ), Zhang (shoot ∨ martial) = (n > 0) ∧ Wang (shoot ∨ martial) = (m > 0) ∧ m = n + 1)
  (H3 : ¬ Li shoot ∧ (Li martial ∨ Li row)) :
  Zhang martial :=
by
  sorry

end xiaozhang_participates_in_martial_arts_l193_193542


namespace emma_troy_wrapping_time_l193_193289

theorem emma_troy_wrapping_time (emma_rate troy_rate total_task_time together_time emma_remaining_time : ℝ) 
  (h1 : emma_rate = 1 / 6) 
  (h2 : troy_rate = 1 / 8) 
  (h3 : total_task_time = 1) 
  (h4 : together_time = 2) 
  (h5 : emma_remaining_time = (total_task_time - (emma_rate + troy_rate) * together_time) / emma_rate) : 
  emma_remaining_time = 2.5 := 
sorry

end emma_troy_wrapping_time_l193_193289


namespace min_C_over_D_l193_193082

-- Define y + 1/y = D and y^2 + 1/y^2 = C.
theorem min_C_over_D (y C D : ℝ) (hy_pos : 0 < y) (hC : y ^ 2 + 1 / (y ^ 2) = C) (hD : y + 1 / y = D) (hC_pos : 0 < C) (hD_pos : 0 < D) :
  C / D = 2 := by
  sorry

end min_C_over_D_l193_193082


namespace simplify_and_evaluate_l193_193089

theorem simplify_and_evaluate (x : ℝ) (h : x = Real.sqrt 3 + 1) : 
  ( ( (2 * x + 1) / x - 1 ) / ( (x^2 - 1) / x ) ) = Real.sqrt 3 / 3 := by
  sorry

end simplify_and_evaluate_l193_193089


namespace parallel_line_through_P_perpendicular_line_through_P_l193_193448

-- Define point P
def P := (-4, 2)

-- Define line l
def l (x y : ℝ) := 3 * x - 2 * y - 7 = 0

-- Define the equation of the line parallel to l that passes through P
def parallel_line (x y : ℝ) := 3 * x - 2 * y + 16 = 0

-- Define the equation of the line perpendicular to l that passes through P
def perpendicular_line (x y : ℝ) := 2 * x + 3 * y + 2 = 0

-- Theorem 1: Prove that parallel_line is the equation of the line passing through P and parallel to l
theorem parallel_line_through_P :
  ∀ (x y : ℝ), 
    (parallel_line x y → x = -4 ∧ y = 2) :=
sorry

-- Theorem 2: Prove that perpendicular_line is the equation of the line passing through P and perpendicular to l
theorem perpendicular_line_through_P :
  ∀ (x y : ℝ), 
    (perpendicular_line x y → x = -4 ∧ y = 2) :=
sorry

end parallel_line_through_P_perpendicular_line_through_P_l193_193448


namespace find_f_of_1_div_8_l193_193649

noncomputable def f (x : ℝ) (a : ℝ) := (a^2 + a - 5) * Real.logb a x

theorem find_f_of_1_div_8 (a : ℝ) (hx1 : x = 1 / 8) (h1 : a > 0) (h2 : a ≠ 1) (h3 : a^2 + a - 5 = 1) :
  f x a = -3 :=
by
  sorry

end find_f_of_1_div_8_l193_193649


namespace desired_cost_per_pound_l193_193814
-- Importing the necessary library

-- Defining the candy weights and their costs per pound
def weight1 : ℝ := 20
def cost_per_pound1 : ℝ := 8
def weight2 : ℝ := 40
def cost_per_pound2 : ℝ := 5

-- Defining the proof statement
theorem desired_cost_per_pound :
  let total_cost := (weight1 * cost_per_pound1 + weight2 * cost_per_pound2)
  let total_weight := (weight1 + weight2)
  let desired_cost := total_cost / total_weight
  desired_cost = 6 := sorry

end desired_cost_per_pound_l193_193814


namespace blue_markers_count_l193_193423

-- Definitions based on given conditions
def total_markers : ℕ := 3343
def red_markers : ℕ := 2315

-- Main statement to prove
theorem blue_markers_count : total_markers - red_markers = 1028 := by
  sorry

end blue_markers_count_l193_193423


namespace cos_alpha_solution_l193_193765

theorem cos_alpha_solution (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.tan α = 1 / 2) : 
  Real.cos α = 2 * Real.sqrt 5 / 5 :=
by
  sorry

end cos_alpha_solution_l193_193765


namespace positive_difference_of_squares_l193_193410

theorem positive_difference_of_squares 
  (a b : ℕ)
  (h1 : a + b = 60)
  (h2 : a - b = 16) : a^2 - b^2 = 960 :=
by
  sorry

end positive_difference_of_squares_l193_193410


namespace hamburgers_purchased_l193_193829

theorem hamburgers_purchased (total_revenue : ℕ) (hamburger_price : ℕ) (additional_hamburgers : ℕ) 
  (target_amount : ℕ) (h1 : total_revenue = 50) (h2 : hamburger_price = 5) (h3 : additional_hamburgers = 4) 
  (h4 : target_amount = 50) :
  (target_amount - (additional_hamburgers * hamburger_price)) / hamburger_price = 6 := 
by 
  sorry

end hamburgers_purchased_l193_193829


namespace solution_l193_193899

-- Definitions
def equation1 (x y z : ℝ) : Prop := 2 * x + y + z = 17
def equation2 (x y z : ℝ) : Prop := x + 2 * y + z = 14
def equation3 (x y z : ℝ) : Prop := x + y + 2 * z = 13

-- Theorem to prove
theorem solution (x y z : ℝ) (h1 : equation1 x y z) (h2 : equation2 x y z) (h3 : equation3 x y z) : x = 6 :=
by
  sorry

end solution_l193_193899


namespace paul_sandwiches_l193_193895

theorem paul_sandwiches (sandwiches_day1 sandwiches_day2 sandwiches_day3 total_sandwiches_3days total_sandwiches_6days : ℕ) 
    (h1 : sandwiches_day1 = 2) 
    (h2 : sandwiches_day2 = 2 * sandwiches_day1) 
    (h3 : sandwiches_day3 = 2 * sandwiches_day2) 
    (h4 : total_sandwiches_3days = sandwiches_day1 + sandwiches_day2 + sandwiches_day3) 
    (h5 : total_sandwiches_6days = 2 * total_sandwiches_3days) 
    : total_sandwiches_6days = 28 := 
by 
    sorry

end paul_sandwiches_l193_193895


namespace second_caterer_cheaper_l193_193302

theorem second_caterer_cheaper (x : ℕ) :
  (150 + 18 * x > 250 + 14 * x) → x ≥ 26 :=
by
  intro h
  sorry

end second_caterer_cheaper_l193_193302


namespace sqrt_subtraction_l193_193655

theorem sqrt_subtraction : Real.sqrt (49 + 81) - Real.sqrt (36 - 9) = Real.sqrt 130 - Real.sqrt 27 := by
  sorry

end sqrt_subtraction_l193_193655


namespace pqrs_inequality_l193_193349

theorem pqrs_inequality (p q r : ℝ) (h_condition : ∀ x : ℝ, (x < -6 ∨ |x - 30| ≤ 2) ↔ ((x - p) * (x - q)) / (x - r) ≥ 0)
  (h_pq : p < q) : p = 28 ∧ q = 32 ∧ r = -6 ∧ p + 2 * q + 3 * r = 78 :=
by
  sorry

end pqrs_inequality_l193_193349


namespace votes_distribution_l193_193197

theorem votes_distribution (W : ℕ) 
  (h1 : W + (W - 53) + (W - 79) + (W - 105) = 963) 
  : W = 300 ∧ 247 = W - 53 ∧ 221 = W - 79 ∧ 195 = W - 105 :=
by
  sorry

end votes_distribution_l193_193197


namespace cistern_emptying_time_l193_193505

theorem cistern_emptying_time (R L : ℝ) (h1 : R * 8 = 1) (h2 : (R - L) * 10 = 1) : 1 / L = 40 :=
by
  -- proof omitted
  sorry

end cistern_emptying_time_l193_193505


namespace correct_calculation_l193_193628

theorem correct_calculation (x a b : ℝ) : 
  (x^4 * x^4 = x^8) ∧ ((a^3)^2 = a^6) ∧ ((a * (b^2))^3 = a^3 * b^6) → (a + 2*a = 3*a) := 
by 
  sorry

end correct_calculation_l193_193628


namespace find_second_largest_element_l193_193945

open List

theorem find_second_largest_element 
(a1 a2 a3 a4 a5 : ℕ) 
(h_pos : 0 < a1 ∧ 0 < a2 ∧ 0 < a3 ∧ 0 < a4 ∧ 0 < a5) 
(h_sorted : a1 ≤ a2 ∧ a2 ≤ a3 ∧ a3 ≤ a4 ∧ a4 ≤ a5) 
(h_mean : (a1 + a2 + a3 + a4 + a5) / 5 = 15) 
(h_range : a5 - a1 = 24) 
(h_mode : a2 = 10 ∧ a3 = 10) 
(h_median : a3 = 10) 
(h_three_diff : (a1 ≠ a2 ∨ a1 ≠ a3 ∨ a1 ≠ a4 ∨ a1 ≠ a5) ∧ (a4 ≠ a5)) :
a4 = 11 :=
sorry

end find_second_largest_element_l193_193945


namespace middle_digit_is_zero_l193_193035

noncomputable def N_in_base8 (a b c : ℕ) : ℕ := 512 * a + 64 * b + 8 * c
noncomputable def N_in_base10 (a b c : ℕ) : ℕ := 100 * b + 10 * c + a

theorem middle_digit_is_zero (a b c : ℕ) (h : N_in_base8 a b c = N_in_base10 a b c) :
  b = 0 :=
by 
  sorry

end middle_digit_is_zero_l193_193035


namespace sum_of_y_values_l193_193727

def g (x : ℚ) : ℚ := 2 * x^2 - x + 3

theorem sum_of_y_values (y1 y2 : ℚ) (hy : g (4 * y1) = 10 ∧ g (4 * y2) = 10) :
  y1 + y2 = 1 / 16 :=
sorry

end sum_of_y_values_l193_193727


namespace custom_op_4_3_equals_37_l193_193837

def custom_op (a b : ℕ) : ℕ := a^2 + a*b + b^2

theorem custom_op_4_3_equals_37 : custom_op 4 3 = 37 := by
  sorry

end custom_op_4_3_equals_37_l193_193837


namespace car_average_speed_l193_193336

theorem car_average_speed
  (d1 d2 t1 t2 : ℕ)
  (h1 : d1 = 85)
  (h2 : d2 = 45)
  (h3 : t1 = 1)
  (h4 : t2 = 1) :
  let total_distance := d1 + d2
  let total_time := t1 + t2
  (total_distance / total_time = 65) :=
by
  sorry

end car_average_speed_l193_193336


namespace problem1_solution_problem2_solution_l193_193722

def f (c a b : ℝ) (x : ℝ) : ℝ := |(c * x + a)| + |(c * x - b)|
def g (c : ℝ) (x : ℝ) : ℝ := |(x - 2)| + c

noncomputable def sol_set_eq1 := {x : ℝ | -1 / 2 ≤ x ∧ x ≤ 3 / 2}
noncomputable def range_a_eq2 := {a : ℝ | a ≤ -2 ∨ a ≥ 0}

-- Problem (1)
theorem problem1_solution : ∀ (x : ℝ), f 2 1 3 x - 4 = 0 ↔ x ∈ sol_set_eq1 := 
by
  intro x
  sorry -- Proof to be filled in

-- Problem (2)
theorem problem2_solution : 
  ∀ x_1 : ℝ, ∃ x_2 : ℝ, g 1 x_2 = f 1 0 1 x_1 ↔ a ∈ range_a_eq2 :=
by
  intro x_1
  sorry -- Proof to be filled in

end problem1_solution_problem2_solution_l193_193722


namespace total_cars_made_in_two_days_l193_193146

-- Define the number of cars made yesterday.
def cars_yesterday : ℕ := 60

-- Define the number of cars made today, which is twice the number of cars made yesterday.
def cars_today : ℕ := 2 * cars_yesterday

-- Define the total number of cars made over the two days.
def total_cars : ℕ := cars_yesterday + cars_today

-- Proof statement: Prove that the total number of cars made in these two days is 180.
theorem total_cars_made_in_two_days : total_cars = 180 := by
  -- Here, we would provide the proof, but we'll use sorry to skip it.
  sorry

end total_cars_made_in_two_days_l193_193146


namespace quadratic_root_sum_l193_193562

theorem quadratic_root_sum (k : ℝ) (h : k ≤ 1 / 2) : 
  ∃ (α β : ℝ), (α + β = 2 - 2 * k) ∧ (α^2 - 2 * (1 - k) * α + k^2 = 0) ∧ (β^2 - 2 * (1 - k) * β + k^2 = 0) ∧ (α + β ≥ 1) :=
sorry

end quadratic_root_sum_l193_193562


namespace insects_ratio_l193_193341

theorem insects_ratio (total_insects : ℕ) (geckos : ℕ) (gecko_insects : ℕ) (lizards : ℕ)
  (H1 : geckos * gecko_insects + lizards * ((total_insects - geckos * gecko_insects) / lizards) = total_insects)
  (H2 : total_insects = 66)
  (H3 : geckos = 5)
  (H4 : gecko_insects = 6)
  (H5 : lizards = 3) :
  (total_insects - geckos * gecko_insects) / lizards / gecko_insects = 2 :=
by
  sorry

end insects_ratio_l193_193341


namespace pen_cost_l193_193975

theorem pen_cost (x : ℝ) (h1 : 5 * x + x = 24) : x = 4 :=
by
  sorry

end pen_cost_l193_193975


namespace part1_part2_l193_193513

open Set

variable {R : Type} [OrderedRing R]

def U : Set R := univ
def A : Set R := {x | x^2 - 2*x - 3 > 0}
def B : Set R := {x | 4 - x^2 <= 0}

theorem part1 : A ∩ B = {x | -2 ≤ x ∧ x < -1} :=
sorry

theorem part2 : (U \ A) ∪ (U \ B) = {x | x < -2 ∨ x > -1} :=
sorry

end part1_part2_l193_193513


namespace asymptotic_minimal_eccentricity_l193_193390

noncomputable def hyperbola (m : ℝ) (x y : ℝ) : Prop :=
  x^2 / m - y^2 / (m^2 + 4) = 1

noncomputable def eccentricity (m : ℝ) : ℝ :=
  Real.sqrt (m + 4 / m + 1)

theorem asymptotic_minimal_eccentricity :
  ∃ (m : ℝ), m = 2 ∧ hyperbola m x y → ∀ x y, y = 2 * x ∨ y = -2 * x :=
by
  sorry

end asymptotic_minimal_eccentricity_l193_193390


namespace minimum_value_of_option_C_l193_193305

noncomputable def optionA (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def optionB (x : ℝ) : ℝ := |Real.sin x| + 4 / |Real.sin x|
noncomputable def optionC (x : ℝ) : ℝ := 2^x + 2^(2 - x)
noncomputable def optionD (x : ℝ) : ℝ := Real.log x + 4 / Real.log x

theorem minimum_value_of_option_C :
  ∃ x : ℝ, optionC x = 4 :=
by
  sorry

end minimum_value_of_option_C_l193_193305


namespace present_age_ratio_l193_193382

-- Define the variables and the conditions
variable (S M : ℕ)

-- Condition 1: Sandy's present age is 84 because she was 78 six years ago
def present_age_sandy := S = 84

-- Condition 2: Sixteen years from now, the ratio of their ages is 5:2
def age_ratio_16_years := (S + 16) * 2 = 5 * (M + 16)

-- The goal: The present age ratio of Sandy to Molly is 7:2
theorem present_age_ratio {S M : ℕ} (h1 : S = 84) (h2 : (S + 16) * 2 = 5 * (M + 16)) : S / M = 7 / 2 :=
by
  -- Integrating conditions
  have hS : S = 84 := h1
  have hR : (S + 16) * 2 = 5 * (M + 16) := h2
  -- We need a proof here, but we'll skip it for now
  sorry

end present_age_ratio_l193_193382


namespace husband_additional_payment_l193_193921

theorem husband_additional_payment (total_medical_cost : ℝ) (total_salary : ℝ) 
                                  (half_medical_cost : ℝ) (deduction_from_salary : ℝ) 
                                  (remaining_salary : ℝ) (total_payment : ℝ)
                                  (each_share : ℝ) (amount_paid_by_husband : ℝ) : 
                                  
                                  total_medical_cost = 128 →
                                  total_salary = 160 →
                                  half_medical_cost = total_medical_cost / 2 →
                                  deduction_from_salary = half_medical_cost →
                                  remaining_salary = total_salary - deduction_from_salary →
                                  total_payment = remaining_salary + half_medical_cost →
                                  each_share = total_payment / 2 →
                                  amount_paid_by_husband = 64 →
                                  (each_share - amount_paid_by_husband) = 16 := by
  sorry

end husband_additional_payment_l193_193921


namespace final_score_eq_l193_193887

variable (initial_score : ℝ)
def deduction_lost_answer : ℝ := 1
def deduction_error : ℝ := 0.5
def deduction_checks : ℝ := 0

def total_deduction : ℝ := deduction_lost_answer + deduction_error + deduction_checks

theorem final_score_eq : final_score = initial_score - total_deduction := by
  sorry

end final_score_eq_l193_193887


namespace sequence_formula_l193_193621

theorem sequence_formula (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : a 1 = 1)
  (h2 : a 2 = 4)
  (h3 : ∀ n : ℕ, n > 0 → a (n + 2) + 2 * a n = 3 * a (n + 1)) :
  (∀ n, a n = 3 * 2^(n-1) - 2) ∧ (S 4 > 21 - 2 * 4) :=
by
  sorry

end sequence_formula_l193_193621


namespace banana_popsicles_count_l193_193154

theorem banana_popsicles_count 
  (grape_popsicles cherry_popsicles total_popsicles : ℕ)
  (h1 : grape_popsicles = 2)
  (h2 : cherry_popsicles = 13)
  (h3 : total_popsicles = 17) :
  total_popsicles - (grape_popsicles + cherry_popsicles) = 2 := by
  sorry

end banana_popsicles_count_l193_193154


namespace MrMartinSpent_l193_193238

theorem MrMartinSpent : 
  ∀ (C B : ℝ), 
    3 * C + 2 * B = 12.75 → 
    B = 1.5 → 
    2 * C + 5 * B = 14 := 
by
  intros C B h1 h2
  sorry

end MrMartinSpent_l193_193238


namespace find_polynomials_g_l193_193615

-- Assume f(x) = x^2
def f (x : ℝ) : ℝ := x ^ 2

-- Define the condition that f(g(x)) = 9x^2 - 6x + 1
def condition (g : ℝ → ℝ) : Prop := ∀ x, f (g x) = 9 * x ^ 2 - 6 * x + 1

-- Prove that the possible polynomials for g(x) are 3x - 1 or -3x + 1
theorem find_polynomials_g (g : ℝ → ℝ) (h : condition g) :
  (∀ x, g x = 3 * x - 1) ∨ (∀ x, g x = -3 * x + 1) :=
sorry

end find_polynomials_g_l193_193615


namespace find_coefficients_l193_193008

theorem find_coefficients (A B : ℝ) (h_roots : (x^2 + A * x + B = 0 ∧ (x = A ∨ x = B))) :
  (A = 0 ∧ B = 0) ∨ (A = 1 ∧ B = -2) :=
by sorry

end find_coefficients_l193_193008


namespace cylindrical_to_rectangular_l193_193226

structure CylindricalCoord where
  r : ℝ
  θ : ℝ
  z : ℝ

structure RectangularCoord where
  x : ℝ
  y : ℝ
  z : ℝ

noncomputable def convertCylindricalToRectangular (c : CylindricalCoord) : RectangularCoord :=
  { x := c.r * Real.cos c.θ,
    y := c.r * Real.sin c.θ,
    z := c.z }

theorem cylindrical_to_rectangular :
  convertCylindricalToRectangular ⟨7, Real.pi / 3, -3⟩ = ⟨3.5, 7 * Real.sqrt 3 / 2, -3⟩ :=
by sorry

end cylindrical_to_rectangular_l193_193226


namespace arithmetic_seq_problem_l193_193730

variable {a : Nat → ℝ}  -- a_n represents the value at index n
variable {d : ℝ} -- The common difference in the arithmetic sequence

-- Define the general term of the arithmetic sequence
def arithmeticSeq (a : Nat → ℝ) (a1 : ℝ) (d : ℝ) : Prop :=
  ∀ n : Nat, a n = a1 + n * d

-- The main proof problem
theorem arithmetic_seq_problem
  (a1 : ℝ)
  (d : ℝ)
  (a : Nat → ℝ)
  (h_arithmetic: arithmeticSeq a a1 d)
  (h_sum : a 2 + a 4 + a 6 + a 8 + a 10 = 80) : 
  a 7 - (1 / 2) * a 8 = 8 := 
  by
  sorry

end arithmetic_seq_problem_l193_193730


namespace smallest_number_of_hikers_l193_193090

theorem smallest_number_of_hikers (n : ℕ) :
  (n % 6 = 1) ∧ (n % 8 = 2) ∧ (n % 9 = 4) ↔ n = 154 :=
by sorry

end smallest_number_of_hikers_l193_193090


namespace correct_area_ratio_l193_193125

noncomputable def area_ratio (P : ℝ) : ℝ :=
  let x := P / 6 
  let length := P / 3
  let diagonal := (P * Real.sqrt 5) / 6
  let r := diagonal / 2
  let A := (5 * (P^2) * Real.pi) / 144
  let s := P / 5
  let R := P / (10 * Real.sin (36 * Real.pi / 180))
  let B := (P^2 * Real.pi) / (100 * (Real.sin (36 * Real.pi / 180))^2)
  A / B

theorem correct_area_ratio (P : ℝ) : area_ratio P = 500 * (Real.sin (36 * Real.pi / 180))^2 / 144 := 
  sorry

end correct_area_ratio_l193_193125


namespace product_lcm_gcd_l193_193009

theorem product_lcm_gcd (a b : ℕ) (h_a : a = 24) (h_b : b = 36):
  Nat.lcm a b * Nat.gcd a b = 864 :=
by
  rw [h_a, h_b]
  sorry

end product_lcm_gcd_l193_193009


namespace reduced_rate_fraction_l193_193525

-- Definitions
def hours_in_a_week := 7 * 24
def hours_with_reduced_rates_on_weekdays := (12 * 5)
def hours_with_reduced_rates_on_weekends := (24 * 2)

-- Question in form of theorem
theorem reduced_rate_fraction :
  (hours_with_reduced_rates_on_weekdays + hours_with_reduced_rates_on_weekends) / hours_in_a_week = 9 / 14 := 
by
  sorry

end reduced_rate_fraction_l193_193525


namespace power_C_50_l193_193381

def matrixC : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![2, 1], ![-4, -1]]

theorem power_C_50 :
  matrixC ^ 50 = ![![4^49 + 1, 4^49], ![-4^50, -2 * 4^49 + 1]] :=
by
  sorry

end power_C_50_l193_193381


namespace area_ratio_l193_193946

theorem area_ratio
  (a b c : ℕ)
  (h1 : 2 * (a + c) = 2 * 2 * (b + c))
  (h2 : a = 2 * b)
  (h3 : c = c) :
  (a * c) = 2 * (b * c) :=
by
  sorry

end area_ratio_l193_193946


namespace number_of_balloons_Allan_bought_l193_193726

theorem number_of_balloons_Allan_bought 
  (initial_balloons final_balloons : ℕ) 
  (h1 : initial_balloons = 5) 
  (h2 : final_balloons = 8) : 
  final_balloons - initial_balloons = 3 := 
  by 
  sorry

end number_of_balloons_Allan_bought_l193_193726


namespace good_deed_done_by_C_l193_193546

def did_good (A B C : Prop) := 
  (¬A ∧ ¬B ∧ C) ∨ (¬A ∧ B ∧ ¬C) ∨ (A ∧ ¬B ∧ ¬C)

def statement_A (B : Prop) := B
def statement_B (B : Prop) := ¬B
def statement_C (C : Prop) := ¬C

theorem good_deed_done_by_C (A B C : Prop)
  (h_deed : (did_good A B C))
  (h_statement : (statement_A B ∧ ¬statement_B B ∧ ¬statement_C C) ∨ 
                      (¬statement_A B ∧ statement_B B ∧ ¬statement_C C) ∨ 
                      (¬statement_A B ∧ ¬statement_B B ∧ statement_C C)) :
  C :=
by 
  sorry

end good_deed_done_by_C_l193_193546


namespace blue_first_yellow_second_probability_l193_193247

open Classical

-- Definition of initial conditions
def total_marbles : Nat := 3 + 4 + 9
def blue_marbles : Nat := 3
def yellow_marbles : Nat := 4
def pink_marbles : Nat := 9

-- Probability functions
def probability_first_blue : ℚ := blue_marbles / total_marbles
def probability_second_yellow_given_blue : ℚ := yellow_marbles / (total_marbles - 1)

-- Combined probability
def combined_probability_first_blue_second_yellow : ℚ := 
  probability_first_blue * probability_second_yellow_given_blue

-- Theorem statement
theorem blue_first_yellow_second_probability :
  combined_probability_first_blue_second_yellow = 1 / 20 :=
by
  -- Proof will be provided here
  sorry

end blue_first_yellow_second_probability_l193_193247


namespace gcd_a_b_l193_193152

def a : ℕ := 333333333
def b : ℕ := 555555555

theorem gcd_a_b : Nat.gcd a b = 111111111 := 
by
  sorry

end gcd_a_b_l193_193152


namespace abs_inequality_solution_l193_193039

theorem abs_inequality_solution (x : ℝ) : 2 * |x - 1| - 1 < 0 ↔ (1 / 2 < x ∧ x < 3 / 2) :=
by
  sorry

end abs_inequality_solution_l193_193039


namespace upper_bound_y_l193_193116

/-- 
  Theorem:
  For any real numbers x and y such that 3 < x < 6 and 6 < y, 
  if the greatest possible positive integer difference between x and y is 6,
  then the upper bound for y is 11.
 -/
theorem upper_bound_y (x y : ℝ) (h₁ : 3 < x) (h₂ : x < 6) (h₃ : 6 < y) (h₄ : y < some_number) (h₅ : y - x = 6) : y = 11 := 
by
  sorry

end upper_bound_y_l193_193116


namespace tanya_efficiency_increase_l193_193816

theorem tanya_efficiency_increase 
  (s_efficiency : ℝ := 1 / 10) (t_efficiency : ℝ := 1 / 8) :
  (((t_efficiency - s_efficiency) / s_efficiency) * 100) = 25 := 
by
  sorry

end tanya_efficiency_increase_l193_193816


namespace abc_cubed_sum_l193_193939

theorem abc_cubed_sum (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
    (h_eq : (a^3 + 12) / a = (b^3 + 12) / b ∧ (b^3 + 12) / b = (c^3 + 12) / c) : 
    a^3 + b^3 + c^3 = -36 :=
by sorry

end abc_cubed_sum_l193_193939


namespace total_boys_school_l193_193757

variable (B : ℕ)
variables (percMuslim percHindu percSikh boysOther : ℕ)

-- Defining the conditions
def condition1 : percMuslim = 44 := by sorry
def condition2 : percHindu = 28 := by sorry
def condition3 : percSikh = 10 := by sorry
def condition4 : boysOther = 54 := by sorry

-- Main theorem statement
theorem total_boys_school (h1 : percMuslim = 44) (h2 : percHindu = 28) (h3 : percSikh = 10) (h4 : boysOther = 54) : 
  B = 300 := by sorry

end total_boys_school_l193_193757


namespace divisibility_ac_bd_l193_193883

-- Conditions definitions
variable (a b c d : ℕ)
variable (hab : a ∣ b)
variable (hcd : c ∣ d)

-- Goal
theorem divisibility_ac_bd : (a * c) ∣ (b * d) :=
  sorry

end divisibility_ac_bd_l193_193883


namespace terms_are_equal_l193_193006

theorem terms_are_equal (n : ℕ) (a b : ℕ → ℕ)
  (h_n : n ≥ 2018)
  (h_a : ∀ i j : ℕ, i ≠ j → a i ≠ a j)
  (h_b : ∀ i j : ℕ, i ≠ j → b i ≠ b j)
  (h_a_pos : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → a i > 0)
  (h_b_pos : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → b i > 0)
  (h_a_le : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → a i ≤ 5 * n)
  (h_b_le : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → b i ≤ 5 * n)
  (h_arith : ∀ i j : ℕ, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n → (a j * b i - a i * b j) * (j - i) = 0):
  ∀ i j : ℕ, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n → a i * b j = a j * b i :=
by
  sorry

end terms_are_equal_l193_193006


namespace range_of_a_l193_193488

variable (a : ℝ)

def proposition_p (a : ℝ) : Prop := 0 < a ∧ a < 1

def proposition_q (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 - x + a > 0 ∧ 1 - 4 * a^2 < 0

theorem range_of_a : (proposition_p a ∨ proposition_q a) ∧ ¬(proposition_p a ∧ proposition_q a) →
  (0 < a ∧ a ≤ 1/2 ∨ a ≥ 1) := 
by
  sorry

end range_of_a_l193_193488


namespace aunt_may_milk_left_l193_193523

theorem aunt_may_milk_left
  (morning_milk : ℕ)
  (evening_milk : ℕ)
  (sold_milk : ℕ)
  (leftover_milk : ℕ)
  (h1 : morning_milk = 365)
  (h2 : evening_milk = 380)
  (h3 : sold_milk = 612)
  (h4 : leftover_milk = 15) :
  morning_milk + evening_milk + leftover_milk - sold_milk = 148 :=
by
  sorry

end aunt_may_milk_left_l193_193523


namespace sin_cos_pi_12_eq_l193_193917

theorem sin_cos_pi_12_eq:
  (Real.sin (Real.pi / 12)) * (Real.cos (Real.pi / 12)) = 1 / 4 :=
by
  sorry

end sin_cos_pi_12_eq_l193_193917


namespace geometric_sequence_product_l193_193962

theorem geometric_sequence_product (a₁ aₙ : ℝ) (n : ℕ) (hn : n > 0) (number_of_terms : n ≥ 1) :
  -- Conditions: First term, last term, number of terms
  ∃ P : ℝ, P = (a₁ * aₙ) ^ (n / 2) :=
sorry

end geometric_sequence_product_l193_193962


namespace cos_B_eq_zero_l193_193763

variable {a b c A B C : ℝ}
variable (h1 : ∀ A B C, 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧ A + B + C = π)
variable (h2 : b * Real.cos A = c)

theorem cos_B_eq_zero (h1 : a = b) (h2 : b * Real.cos A = c) : Real.cos B = 0 :=
sorry

end cos_B_eq_zero_l193_193763


namespace volume_ratio_of_cube_and_cuboid_l193_193395

theorem volume_ratio_of_cube_and_cuboid :
  let edge_length_meter := 1
  let edge_length_cm := edge_length_meter * 100 -- Convert meter to centimeters
  let cube_volume := edge_length_cm^3
  let cuboid_width := 50
  let cuboid_length := 50
  let cuboid_height := 20
  let cuboid_volume := cuboid_width * cuboid_length * cuboid_height
  cube_volume = 20 * cuboid_volume := 
by
  sorry

end volume_ratio_of_cube_and_cuboid_l193_193395


namespace no_unsatisfactory_grades_l193_193313

theorem no_unsatisfactory_grades (total_students : ℕ)
  (top_marks : ℕ) (average_marks : ℕ) (good_marks : ℕ)
  (h1 : top_marks = total_students / 6)
  (h2 : average_marks = total_students / 3)
  (h3 : good_marks = total_students / 2) :
  total_students = top_marks + average_marks + good_marks := by
  sorry

end no_unsatisfactory_grades_l193_193313


namespace solve_inequalities_l193_193073

theorem solve_inequalities (x : ℝ) :
  (1 / x < 1 ∧ |4 * x - 1| > 2) →
  (x < -1/4 ∨ x > 1) :=
by
  sorry

end solve_inequalities_l193_193073


namespace max_k_value_l193_193750

theorem max_k_value (x y : ℝ) (k : ℝ) (hx : 0 < x) (hy : 0 < y)
(h : 5 = k^2 * ((x^2 / y^2) + (y^2 / x^2)) + 2 * k * (x / y + y / x)) :
  k ≤ Real.sqrt (5 / 6) := sorry

end max_k_value_l193_193750


namespace motorcycle_materials_cost_l193_193867

theorem motorcycle_materials_cost 
  (car_material_cost : ℕ) (cars_per_month : ℕ) (car_sale_price : ℕ)
  (motorcycles_per_month : ℕ) (motorcycle_sale_price : ℕ)
  (additional_profit : ℕ) :
  car_material_cost = 100 →
  cars_per_month = 4 →
  car_sale_price = 50 →
  motorcycles_per_month = 8 →
  motorcycle_sale_price = 50 →
  additional_profit = 50 →
  car_material_cost + additional_profit = 250 := by
  sorry

end motorcycle_materials_cost_l193_193867


namespace amount_charged_for_kids_l193_193966

theorem amount_charged_for_kids (K A: ℝ) (H1: A = 2 * K) (H2: 8 * K + 10 * A = 84) : K = 3 :=
by
  sorry

end amount_charged_for_kids_l193_193966


namespace find_m_l193_193157

theorem find_m {A B : Set ℝ} (m : ℝ) :
  (A = {x : ℝ | x^2 + x - 12 = 0}) →
  (B = {x : ℝ | mx + 1 = 0}) →
  (A ∩ B = {3}) →
  m = -1 / 3 := 
by
  intros hA hB h_inter
  sorry

end find_m_l193_193157


namespace range_of_t_circle_largest_area_eq_point_P_inside_circle_l193_193464

open Real

-- Defining the given equation representing the trajectory of a point on a circle
def circle_eq (x y t : ℝ) : Prop :=
  x^2 + y^2 - 2 * (t + 3) * x + 2 * (1 - 4 * t^2) * y + 16 * t^4 + 9 = 0

-- Problem 1: Proving the range of t
theorem range_of_t : 
  ∀ t : ℝ, (∃ x y : ℝ, circle_eq x y t) → -1/7 < t ∧ t < 1 :=
sorry

-- Problem 2: Proving the equation of the circle with the largest area
theorem circle_largest_area_eq : 
  ∃ t : ℝ, t = 3/7 ∧ (∀ x y : ℝ, circle_eq x y (3/7)) → 
  ∀ x y : ℝ, (x - 24/7)^2 + (y + 13/49)^2 = 16/7 :=
sorry

-- Problem 3: Proving the range of t for point P to be inside the circle
theorem point_P_inside_circle : 
  ∀ t : ℝ, (∃ x y : ℝ, circle_eq x y t) → 
  (0 < t ∧ t < 3/4) :=
sorry

end range_of_t_circle_largest_area_eq_point_P_inside_circle_l193_193464


namespace quadratic_real_roots_iff_l193_193926

/-- For the quadratic equation x^2 + 3x + m = 0 to have two real roots,
    the value of m must satisfy m ≤ 9/4. -/
theorem quadratic_real_roots_iff (m : ℝ) : (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 * x2 = m ∧ x1 + x2 = -3) ↔ m ≤ 9 / 4 :=
by
  sorry

end quadratic_real_roots_iff_l193_193926


namespace least_x_l193_193286

noncomputable def is_odd_prime (n : ℕ) : Prop :=
  n > 1 ∧ Prime n ∧ n % 2 = 1

theorem least_x (x p : ℕ) (hp : Prime p) (hx : x > 0) (hodd_prime : is_odd_prime (x / (12 * p))) : x = 72 := 
  sorry

end least_x_l193_193286


namespace modulo_sum_remainder_l193_193564

theorem modulo_sum_remainder (a b: ℤ) (k j: ℤ) 
  (h1 : a = 84 * k + 77) 
  (h2 : b = 120 * j + 113) :
  (a + b) % 42 = 22 := by
  sorry

end modulo_sum_remainder_l193_193564


namespace polynomial_roots_ratio_l193_193210

theorem polynomial_roots_ratio (a b c d : ℝ) (h₀ : a ≠ 0) 
    (h₁ : a * 64 + b * 16 + c * 4 + d = 0)
    (h₂ : -a + b - c + d = 0) : 
    (b + c) / a = -13 :=
by {
    sorry
}

end polynomial_roots_ratio_l193_193210


namespace route_length_l193_193635

theorem route_length (D : ℝ) (T : ℝ) 
  (hx : T = 400 / D) 
  (hy : 80 = (D / 5) * T) 
  (hz : 80 + (D / 4) * T = D) : 
  D = 180 :=
by
  sorry

end route_length_l193_193635


namespace problem_1_problem_2_problem_3_problem_4_l193_193084

theorem problem_1 : 2 * Real.sqrt 7 - 6 * Real.sqrt 7 = -4 * Real.sqrt 7 :=
by sorry

theorem problem_2 : Real.sqrt (2 / 3) / Real.sqrt (8 / 27) = (3 / 2) :=
by sorry

theorem problem_3 : Real.sqrt 18 + Real.sqrt 98 - Real.sqrt 27 = (10 * Real.sqrt 2 - 3 * Real.sqrt 3) :=
by sorry

theorem problem_4 : (Real.sqrt 0.5 + Real.sqrt 6) - (Real.sqrt (1 / 8) - Real.sqrt 24) = (Real.sqrt 2 / 4) + 3 * Real.sqrt 6 :=
by sorry

end problem_1_problem_2_problem_3_problem_4_l193_193084


namespace trigonometric_identity_l193_193689

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 2) : 
  (1 - Real.sin (2 * θ)) / (2 * (Real.cos θ)^2) = 1 / 2 :=
sorry

end trigonometric_identity_l193_193689


namespace problem1_l193_193068

theorem problem1 :
  0.064^(-1 / 3) - (-1 / 8)^0 + 16^(3 / 4) + 0.25^(1 / 2) = 10 :=
by
  sorry

end problem1_l193_193068


namespace abc_inequality_l193_193298

-- Required conditions and proof statement
theorem abc_inequality 
  {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : a * b * c = 1 / 8) : 
  a^2 + b^2 + c^2 + a^2 * b^2 + a^2 * c^2 + b^2 * c^2 ≥ 15 / 16 := 
sorry

end abc_inequality_l193_193298


namespace sum_of_squares_l193_193165

variable {x y : ℝ}

theorem sum_of_squares (h1 : x + y = 20) (h2 : x * y = 100) : x^2 + y^2 = 200 :=
sorry

end sum_of_squares_l193_193165


namespace find_z_l193_193961

/-- x and y are positive integers. When x is divided by 9, the remainder is 2, 
and when x is divided by 7, the remainder is 4. When y is divided by 13, 
the remainder is 12. The least possible value of y - x is 14. 
Prove that the number that y is divided by to get a remainder of 3 is 22. -/
theorem find_z (x y z : ℕ) (hx9 : x % 9 = 2) (hx7 : x % 7 = 4) (hy13 : y % 13 = 12) (hyx : y = x + 14) 
: y % z = 3 → z = 22 := 
by 
  sorry

end find_z_l193_193961


namespace inverse_proposition_false_l193_193937

-- Define the original proposition
def original_proposition (a b : ℝ) : Prop :=
  a = b → abs a = abs b

-- Define the inverse proposition
def inverse_proposition (a b : ℝ) : Prop :=
  abs a = abs b → a = b

-- The theorem to prove
theorem inverse_proposition_false : ∃ (a b : ℝ), abs a = abs b ∧ a ≠ b :=
sorry

end inverse_proposition_false_l193_193937


namespace probability_of_meeting_at_cafe_l193_193162

open Set

/-- Define the unit square where each side represents 1 hour (from 2:00 to 3:00 PM). -/
def unit_square : Set (ℝ × ℝ) := { p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1 }

/-- Define the overlap condition for Cara and David meeting at the café. -/
def overlap_region : Set (ℝ × ℝ) :=
  { p | max (p.1 - 0.5) 0 ≤ p.2 ∧ p.2 ≤ min (p.1 + 0.5) 1 }

/-- The area of the overlap region within the unit square. -/
noncomputable def overlap_area : ℝ :=
  ∫ x in Icc 0 1, (min (x + 0.5) 1 - max (x - 0.5) 0)

theorem probability_of_meeting_at_cafe : overlap_area / 1 = 1 / 2 :=
by
  sorry

end probability_of_meeting_at_cafe_l193_193162


namespace number_of_outfits_l193_193069

def shirts : ℕ := 5
def hats : ℕ := 3

theorem number_of_outfits : shirts * hats = 15 :=
by 
  -- This part intentionally left blank since no proof required.
  sorry

end number_of_outfits_l193_193069


namespace find_a_l193_193010

noncomputable def A : Set ℝ := {x | x^2 - x - 2 = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | a * x - 1 = 0}
def is_solution (a : ℝ) : Prop := ∀ b, b ∈ B a → b ∈ A

theorem find_a (a : ℝ) : (B a ⊆ A) → a = 0 ∨ a = -1 ∨ a = 1/2 := by
  intro h
  sorry

end find_a_l193_193010


namespace binary_111_eq_7_l193_193973

theorem binary_111_eq_7 : (1 * 2^0 + 1 * 2^1 + 1 * 2^2) = 7 :=
by
  sorry

end binary_111_eq_7_l193_193973


namespace unique_zero_function_l193_193845

theorem unique_zero_function {f : ℕ → ℕ} (h : ∀ m n, f (m + f n) = f m + f n + f (n + 1)) : ∀ n, f n = 0 :=
by {
  sorry
}

end unique_zero_function_l193_193845


namespace evaluate_f_diff_l193_193616

def f (x : ℝ) := x^5 + 2*x^3 + 7*x

theorem evaluate_f_diff : f 3 - f (-3) = 636 := by
  sorry

end evaluate_f_diff_l193_193616


namespace square_area_relation_l193_193171

variable {lA lB : ℝ}

theorem square_area_relation (h : lB = 4 * lA) : lB^2 = 16 * lA^2 :=
by sorry

end square_area_relation_l193_193171


namespace odd_solutions_eq_iff_a_le_neg3_or_a_ge3_l193_193235

theorem odd_solutions_eq_iff_a_le_neg3_or_a_ge3 (a : ℝ) :
  (∃! x : ℝ, -1 ≤ x ∧ x ≤ 5 ∧ (a - 3 * x^2 + Real.cos (9 * Real.pi * x / 2)) * Real.sqrt (3 - a * x) = 0) ↔ (a ≤ -3 ∨ a ≥ 3) := 
by
  sorry

end odd_solutions_eq_iff_a_le_neg3_or_a_ge3_l193_193235


namespace xy_value_l193_193586

theorem xy_value (x y : ℝ) (h : x * (x + y) = x ^ 2 + 12) : x * y = 12 :=
by {
  sorry
}

end xy_value_l193_193586


namespace inequality_am_gm_l193_193088

theorem inequality_am_gm 
  (a b c : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_sum : a^2 + b^2 + c^2 = 1 / 2) :
  (1 - a^2 + c^2) / (c * (a + 2 * b)) + 
  (1 - b^2 + a^2) / (a * (b + 2 * c)) + 
  (1 - c^2 + b^2) / (b * (c + 2 * a)) >= 6 := 
sorry

end inequality_am_gm_l193_193088


namespace driving_time_in_fog_is_correct_l193_193170

-- Define constants for speeds (in miles per minute)
def speed_sunny : ℚ := 35 / 60
def speed_rain : ℚ := 25 / 60
def speed_fog : ℚ := 15 / 60

-- Total distance and time
def total_distance : ℚ := 19.5
def total_time : ℚ := 45

-- Time variables for rain and fog
variables (t_r t_f : ℚ)

-- Define the driving distance equation
def distance_eq : Prop :=
  speed_sunny * (total_time - t_r - t_f) + speed_rain * t_r + speed_fog * t_f = total_distance

-- Prove the time driven in fog equals 10.25 minutes
theorem driving_time_in_fog_is_correct (h : distance_eq t_r t_f) : t_f = 10.25 :=
sorry

end driving_time_in_fog_is_correct_l193_193170


namespace find_n_of_permut_comb_eq_l193_193230

open Nat

theorem find_n_of_permut_comb_eq (n : Nat) (h : (n! / (n - 3)!) = 6 * (n! / (4! * (n - 4)!))) : n = 7 := by
  sorry

end find_n_of_permut_comb_eq_l193_193230


namespace john_days_to_lose_weight_l193_193509

noncomputable def john_calories_intake : ℕ := 1800
noncomputable def john_calories_burned : ℕ := 2300
noncomputable def calories_to_lose_1_pound : ℕ := 4000
noncomputable def pounds_to_lose : ℕ := 10

theorem john_days_to_lose_weight :
  (john_calories_burned - john_calories_intake) * (pounds_to_lose * calories_to_lose_1_pound / (john_calories_burned - john_calories_intake)) = 80 :=
by
  sorry

end john_days_to_lose_weight_l193_193509


namespace angle_bisectors_l193_193246

open Real

noncomputable def r1 : ℝ × ℝ × ℝ := (1, 1, 0)
noncomputable def r2 : ℝ × ℝ × ℝ := (0, 1, 1)

theorem angle_bisectors :
  ∃ (phi : ℝ), 0 ≤ phi ∧ phi ≤ π ∧ cos phi = 1 / 2 :=
sorry

end angle_bisectors_l193_193246


namespace even_quadratic_iff_b_zero_l193_193947

-- Define a quadratic function
def quadratic (a b c x : ℝ) : ℝ := a * x ^ 2 + b * x + c

-- State the theorem
theorem even_quadratic_iff_b_zero (a b c : ℝ) : 
  (∀ x : ℝ, quadratic a b c x = quadratic a b c (-x)) ↔ b = 0 := 
by
  sorry

end even_quadratic_iff_b_zero_l193_193947


namespace sally_balloon_count_l193_193019

theorem sally_balloon_count (n_initial : ℕ) (n_lost : ℕ) (n_final : ℕ) 
  (h_initial : n_initial = 9) 
  (h_lost : n_lost = 2) 
  (h_final : n_final = n_initial - n_lost) : 
  n_final = 7 :=
by
  sorry

end sally_balloon_count_l193_193019


namespace constant_term_expansion_l193_193106

theorem constant_term_expansion (x : ℝ) (hx : x ≠ 0) :
  ∃ k : ℝ, k = -21/2 ∧
  (∀ r : ℕ, (9 : ℕ).choose r * (x^(1/2))^(9-r) * ((-(1/(2*x)))^r) = k) :=
sorry

end constant_term_expansion_l193_193106


namespace math_problem_l193_193663

theorem math_problem (a b n r : ℕ) (h₁ : 1853 ≡ 53 [MOD 600]) (h₂ : 2101 ≡ 101 [MOD 600]) :
  (1853 * 2101) ≡ 553 [MOD 600] := by
  sorry

end math_problem_l193_193663


namespace determine_x_l193_193229

theorem determine_x (x y : ℝ) (h : x / (x - 1) = (y^3 + 2 * y^2 - 1) / (y^3 + 2 * y^2 - 2)) : 
  x = y^3 + 2 * y^2 - 1 :=
by
  sorry

end determine_x_l193_193229


namespace factorable_polynomial_l193_193985

theorem factorable_polynomial (d f e g b : ℤ) (h1 : d * f = 28) (h2 : e * g = 14)
  (h3 : d * g + e * f = b) : b = 42 :=
by sorry

end factorable_polynomial_l193_193985


namespace solve_inequality_l193_193314

theorem solve_inequality (x : ℝ) : 
  3*x^2 + 2*x - 3 > 10 - 2*x ↔ x < ( -2 - Real.sqrt 43 ) / 3 ∨ x > ( -2 + Real.sqrt 43 ) / 3 := 
by
  sorry

end solve_inequality_l193_193314


namespace blood_pressure_systolic_diastolic_l193_193789

noncomputable def blood_pressure (t : ℝ) : ℝ :=
110 + 25 * Real.sin (160 * t)

theorem blood_pressure_systolic_diastolic :
  (∀ t : ℝ, blood_pressure t ≤ 135) ∧ (∀ t : ℝ, blood_pressure t ≥ 85) :=
by
  sorry

end blood_pressure_systolic_diastolic_l193_193789


namespace rectangular_table_capacity_l193_193283

variable (R : ℕ) -- The number of pupils a rectangular table can seat

-- Conditions
variable (rectangular_tables : ℕ)
variable (square_tables : ℕ)
variable (square_table_capacity : ℕ)
variable (total_pupils : ℕ)

-- Setting the values based on the conditions
axiom h1 : rectangular_tables = 7
axiom h2 : square_tables = 5
axiom h3 : square_table_capacity = 4
axiom h4 : total_pupils = 90

-- The proof statement
theorem rectangular_table_capacity :
  7 * R + 5 * 4 = 90 → R = 10 :=
by
  intro h
  sorry

end rectangular_table_capacity_l193_193283


namespace crank_slider_motion_l193_193056

def omega : ℝ := 10
def OA : ℝ := 90
def AB : ℝ := 90
def AM : ℝ := 60
def t : ℝ := sorry -- t is a variable, no specific value required

theorem crank_slider_motion :
  (∀ t : ℝ, ((90 * Real.cos (10 * t)), (90 * Real.sin (10 * t) + 60)) = (x, y)) ∧
  (∀ t : ℝ, ((-900 * Real.sin (10 * t)), (900 * Real.cos (10 * t))) = (vx, vy)) :=
sorry

end crank_slider_motion_l193_193056


namespace sum_first_19_terms_l193_193063

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)
variable (d : ℝ)
variable (a₀ a₃ a₁₇ a₁₀ : ℝ)

noncomputable def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ a₀ d, ∀ n, a n = a₀ + n * d

noncomputable def sum_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) :=
  ∀ n, S n = (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))

theorem sum_first_19_terms (h1 : is_arithmetic_sequence a)
                          (h2 : a 3 + a 17 = 10)
                          (h3 : sum_first_n_terms S a) :
  S 19 = 95 :=
sorry

end sum_first_19_terms_l193_193063


namespace sin_double_angle_l193_193391

theorem sin_double_angle (α : ℝ) (h : Real.sin α + Real.cos α = Real.sqrt 3 / 3) : 
  Real.sin (2 * α) = -2 / 3 := 
sorry

end sin_double_angle_l193_193391


namespace total_length_infinite_sum_l193_193924

-- Define the infinite sums
noncomputable def S1 : ℝ := ∑' n : ℕ, (1 / (3^n))
noncomputable def S2 : ℝ := (∑' n : ℕ, (1 / (5^n))) * Real.sqrt 3
noncomputable def S3 : ℝ := (∑' n : ℕ, (1 / (7^n))) * Real.sqrt 5

-- Define the total length
noncomputable def total_length : ℝ := S1 + S2 + S3

-- The statement of the theorem
theorem total_length_infinite_sum : total_length = (3 / 2) + (Real.sqrt 3 / 4) + (Real.sqrt 5 / 6) :=
by
  sorry

end total_length_infinite_sum_l193_193924


namespace sum_of_reciprocals_l193_193121

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 32) :
  (1 / x) + (1 / y) = 3 / 8 := 
sorry

end sum_of_reciprocals_l193_193121


namespace remainder_is_zero_l193_193933

def f (x : ℝ) : ℝ := x^3 - 5 * x^2 + 2 * x + 8

theorem remainder_is_zero : f 2 = 0 := by
  sorry

end remainder_is_zero_l193_193933


namespace arith_seq_ratio_l193_193871

-- Definitions related to arithmetic sequence and sum
def arithmetic_seq (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_arith_seq (S a : ℕ → ℝ) := ∀ n : ℕ, S n = (n : ℝ) / 2 * (a 1 + a n)

-- Given condition
def condition (a : ℕ → ℝ) := a 8 / a 7 = 13 / 5

-- Prove statement
theorem arith_seq_ratio (a S : ℕ → ℝ)
  (h_arith : arithmetic_seq a)
  (h_sum : sum_of_arith_seq S a)
  (h_cond : condition a) :
  S 15 / S 13 = 3 := 
sorry

end arith_seq_ratio_l193_193871


namespace hcf_of_two_numbers_l193_193385

theorem hcf_of_two_numbers (H L P : ℕ) (h1 : L = 160) (h2 : P = 2560) (h3 : H * L = P) : H = 16 :=
by
  sorry

end hcf_of_two_numbers_l193_193385


namespace min_value_of_fraction_sum_l193_193058

theorem min_value_of_fraction_sum (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_sum : x^2 + y^2 + z^2 = 1) :
  (2 * (1/(1-x^2) + 1/(1-y^2) + 1/(1-z^2))) = 3 * Real.sqrt 3 :=
sorry

end min_value_of_fraction_sum_l193_193058


namespace jack_money_proof_l193_193818

variable (sock_cost : ℝ) (number_of_socks : ℕ) (shoe_cost : ℝ) (available_money : ℝ)

def jack_needs_more_money : Prop := 
  (number_of_socks * sock_cost + shoe_cost) - available_money = 71

theorem jack_money_proof (h1 : sock_cost = 9.50)
                         (h2 : number_of_socks = 2)
                         (h3 : shoe_cost = 92)
                         (h4 : available_money = 40) :
  jack_needs_more_money sock_cost number_of_socks shoe_cost available_money :=
by
  unfold jack_needs_more_money
  simp [h1, h2, h3, h4]
  norm_num
  done

end jack_money_proof_l193_193818


namespace correct_values_correct_result_l193_193070

theorem correct_values (a b : ℝ) :
  ((2 * x - a) * (3 * x + b) = 6 * x^2 + 11 * x - 10) ∧
  ((2 * x + a) * (x + b) = 2 * x^2 - 9 * x + 10) →
  (a = -5) ∧ (b = -2) :=
sorry

theorem correct_result :
  (2 * x - 5) * (3 * x - 2) = 6 * x^2 - 19 * x + 10 :=
sorry

end correct_values_correct_result_l193_193070


namespace intersection_of_squares_perimeter_l193_193662

noncomputable def perimeter_of_rectangle (side1 side2 : ℝ) : ℝ :=
2 * (side1 + side2)

theorem intersection_of_squares_perimeter
  (side_length : ℝ)
  (diagonal : ℝ)
  (distance_between_centers : ℝ)
  (h1 : 4 * side_length = 8) 
  (h2 : (side1^2 + side2^2) = diagonal^2)
  (h3 : (2 - side1)^2 + (2 - side2)^2 = distance_between_centers^2) : 
10 * (perimeter_of_rectangle side1 side2) = 25 :=
sorry

end intersection_of_squares_perimeter_l193_193662


namespace tan_passing_through_point_l193_193835

theorem tan_passing_through_point :
  (∃ ϕ : ℝ, (∀ x : ℝ, y = Real.tan (2 * x + ϕ)) ∧ (Real.tan (2 * (π / 12) + ϕ) = 0)) →
  ϕ = - (π / 6) :=
by
  sorry

end tan_passing_through_point_l193_193835


namespace thickness_relation_l193_193456

noncomputable def a : ℝ := (1/3) * Real.sin (1/2)
noncomputable def b : ℝ := (1/2) * Real.sin (1/3)
noncomputable def c : ℝ := (1/3) * Real.cos (7/8)

theorem thickness_relation : c > b ∧ b > a := by
  sorry

end thickness_relation_l193_193456


namespace price_reduction_for_1920_profit_maximum_profit_calculation_l193_193507

-- Definitions based on given conditions
def cost_price : ℝ := 12
def base_price : ℝ := 20
def base_quantity_sold : ℝ := 240
def increment_per_dollar : ℝ := 40

-- Profit function
def profit (x : ℝ) : ℝ := (base_price - cost_price - x) * (base_quantity_sold + increment_per_dollar * x)

-- Prove price reduction for $1920 profit per day
theorem price_reduction_for_1920_profit : ∃ x : ℝ, profit x = 1920 ∧ x = 8 := by
  sorry

-- Prove maximum profit calculation
theorem maximum_profit_calculation : ∃ x y : ℝ, x = 4 ∧ y = 2560 ∧ ∀ z, profit z ≤ y := by
  sorry

end price_reduction_for_1920_profit_maximum_profit_calculation_l193_193507


namespace isosceles_triangle_perimeter_l193_193998

theorem isosceles_triangle_perimeter 
  (m : ℝ) 
  (h : 2 * m + 1 = 8) : 
  (m - 2) + 2 * 8 = 17.5 := 
by 
  sorry

end isosceles_triangle_perimeter_l193_193998


namespace equilateral_triangle_perimeter_l193_193668

-- Definitions based on conditions
def equilateral_triangle_side : ℕ := 8

-- The statement we need to prove
theorem equilateral_triangle_perimeter : 3 * equilateral_triangle_side = 24 := by
  sorry

end equilateral_triangle_perimeter_l193_193668


namespace minimum_solutions_in_interval_l193_193287

open Function Real

-- Define what it means for a function to be even
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

-- Define what it means for a function to be periodic
def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x : ℝ, f (x + p) = f x

-- Main theorem statement
theorem minimum_solutions_in_interval :
  ∀ (f : ℝ → ℝ),
  is_even f → is_periodic f 3 → f 2 = 0 →
  (∃ x1 x2 x3 x4 : ℝ, 0 < x1 ∧ x1 < 6 ∧ f x1 = 0 ∧
                     0 < x2 ∧ x2 < 6 ∧ f x2 = 0 ∧
                     0 < x3 ∧ x3 < 6 ∧ f x3 = 0 ∧
                     0 < x4 ∧ x4 < 6 ∧ f x4 = 0 ∧
                     x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧
                     x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4) :=
by
  sorry

end minimum_solutions_in_interval_l193_193287


namespace ken_ride_time_l193_193703

variables (x y k t : ℝ)

-- Condition 1: It takes Ken 80 seconds to walk down an escalator when it is not moving.
def condition1 : Prop := 80 * x = y

-- Condition 2: It takes Ken 40 seconds to walk down an escalator when it is moving with a 10-second delay.
def condition2 : Prop := 50 * (x + k) = y

-- Condition 3: There is a 10-second delay before the escalator starts moving.
def condition3 : Prop := t = y / k + 10

-- Related Speed
def condition4 : Prop := k = 0.6 * x

-- Proposition: The time Ken takes to ride the escalator down without walking, including the delay, is 143 seconds.
theorem ken_ride_time {x y k t : ℝ} (h1 : condition1 x y) (h2 : condition2 x y k) (h3 : condition3 y k t) (h4 : condition4 x k) :
  t = 143 :=
by sorry

end ken_ride_time_l193_193703


namespace geometric_seq_b6_l193_193281

variable {b : ℕ → ℝ}

theorem geometric_seq_b6 (h1 : b 3 * b 9 = 9) (h2 : ∃ r, ∀ n, b (n + 1) = r * b n) : b 6 = 3 ∨ b 6 = -3 :=
by
  sorry

end geometric_seq_b6_l193_193281


namespace observation_count_l193_193609

theorem observation_count (mean_before mean_after : ℝ) 
  (wrong_value : ℝ) (correct_value : ℝ) (n : ℝ) :
  mean_before = 36 →
  correct_value = 60 →
  wrong_value = 23 →
  mean_after = 36.5 →
  n = 74 :=
by
  intros h_mean_before h_correct_value h_wrong_value h_mean_after
  sorry

end observation_count_l193_193609


namespace sum_fib_2019_eq_fib_2021_minus_1_l193_193517

def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib n + fib (n + 1)

def sum_fib : ℕ → ℕ
| 0 => 0
| n + 1 => sum_fib n + fib (n + 1)

theorem sum_fib_2019_eq_fib_2021_minus_1 : sum_fib 2019 = fib 2021 - 1 := 
by sorry -- proof here

end sum_fib_2019_eq_fib_2021_minus_1_l193_193517


namespace Aaron_final_cards_l193_193200

-- Definitions from conditions
def initial_cards_Aaron : Nat := 5
def found_cards_Aaron : Nat := 62

-- Theorem statement
theorem Aaron_final_cards : initial_cards_Aaron + found_cards_Aaron = 67 :=
by
  sorry

end Aaron_final_cards_l193_193200


namespace orange_is_faster_by_l193_193862

def forest_run_time (distance speed : ℕ) : ℕ := distance / speed
def beach_run_time (distance speed : ℕ) : ℕ := distance / speed
def mountain_run_time (distance speed : ℕ) : ℕ := distance / speed

def total_time_in_minutes (forest_distance forest_speed beach_distance beach_speed mountain_distance mountain_speed : ℕ) : ℕ :=
  (forest_run_time forest_distance forest_speed + beach_run_time beach_distance beach_speed + mountain_run_time mountain_distance mountain_speed) * 60

def apple_total_time := total_time_in_minutes 18 3 6 2 3 1
def mac_total_time := total_time_in_minutes 20 4 8 3 3 1
def orange_total_time := total_time_in_minutes 22 5 10 4 3 2

def combined_time := apple_total_time + mac_total_time
def orange_time_difference := combined_time - orange_total_time

theorem orange_is_faster_by :
  orange_time_difference = 856 := sorry

end orange_is_faster_by_l193_193862


namespace original_acid_percentage_zero_l193_193719

theorem original_acid_percentage_zero (a w : ℝ) 
  (h1 : (a + 1) / (a + w + 1) = 1 / 4) 
  (h2 : (a + 2) / (a + w + 2) = 2 / 5) : 
  a / (a + w) = 0 := 
by
  sorry

end original_acid_percentage_zero_l193_193719


namespace divisor_of_sum_of_four_consecutive_integers_l193_193808

theorem divisor_of_sum_of_four_consecutive_integers (n : ℤ) :
  2 ∣ ((n - 1) + n + (n + 1) + (n + 2)) :=
by sorry

end divisor_of_sum_of_four_consecutive_integers_l193_193808


namespace estimate_total_fish_l193_193644

theorem estimate_total_fish (marked : ℕ) (sample_size : ℕ) (marked_in_sample : ℕ) (x : ℝ) 
  (h1 : marked = 50) 
  (h2 : sample_size = 168) 
  (h3 : marked_in_sample = 8) 
  (h4 : sample_size * 50 = marked_in_sample * x) : 
  x = 1050 := 
sorry

end estimate_total_fish_l193_193644


namespace movie_sale_price_l193_193989

/-- 
Given the conditions:
- cost of actors: $1200
- number of people: 50
- cost of food per person: $3
- equipment rental costs twice as much as food and actors combined
- profit made: $5950

Prove that the selling price of the movie was $10,000.
-/
theorem movie_sale_price :
  let cost_of_actors := 1200
  let num_people := 50
  let food_cost_per_person := 3
  let total_food_cost := num_people * food_cost_per_person
  let combined_cost := total_food_cost + cost_of_actors
  let equipment_rental_cost := 2 * combined_cost
  let total_cost := cost_of_actors + total_food_cost + equipment_rental_cost
  let profit := 5950
  let sale_price := total_cost + profit
  sale_price = 10000 := 
by
  sorry

end movie_sale_price_l193_193989


namespace painted_faces_of_large_cube_l193_193324

theorem painted_faces_of_large_cube (n : ℕ) (unpainted_cubes : ℕ) :
  n = 9 ∧ unpainted_cubes = 343 → (painted_faces : ℕ) = 3 :=
by
  intros h
  let ⟨h_n, h_unpainted⟩ := h
  sorry

end painted_faces_of_large_cube_l193_193324


namespace proof_problem_l193_193522

-- Define the conditions
def a : ℤ := -3
def b : ℤ := -4
def cond1 := a^4 = 81
def cond2 := b^3 = -64

-- Define the goal in terms of the conditions
theorem proof_problem : a^4 + b^3 = 17 :=
by
  have h1 : a^4 = 81 := sorry
  have h2 : b^3 = -64 := sorry
  rw [h1, h2]
  norm_num

end proof_problem_l193_193522


namespace part1_part2_l193_193688

theorem part1 (x : ℝ) : -5 * x^2 + 3 * x + 2 > 0 ↔ -2/5 < x ∧ x < 1 :=
by sorry

theorem part2 (a x : ℝ) (h : 0 < a) :
  (a * x^2 + (a + 3) * x + 3 > 0 ↔
    (
      (0 < a ∧ a < 3 ∧ (x < -3/a ∨ x > -1)) ∨
      (a = 3 ∧ x ≠ -1) ∨
      (a > 3 ∧ (x < -1 ∨ x > -3/a))
    )
  ) :=
by sorry

end part1_part2_l193_193688


namespace third_number_in_sequence_l193_193674

theorem third_number_in_sequence (n : ℕ) (h_sum : n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 63) : n + 2 = 8 :=
by
  -- the proof would be written here
  sorry

end third_number_in_sequence_l193_193674


namespace average_after_12th_inning_revised_average_not_out_l193_193629

theorem average_after_12th_inning (A : ℝ) (H_innings : 11 * A + 92 = 12 * (A + 2)) : (A + 2) = 70 :=
by
  -- Calculation steps are skipped
  sorry

theorem revised_average_not_out (A : ℝ) (H_innings : 11 * A + 92 = 12 * (A + 2)) (H_not_out : 11 * A + 92 = 840) :
  (11 * A + 92) / 9 = 93.33 :=
by
  -- Calculation steps are skipped
  sorry

end average_after_12th_inning_revised_average_not_out_l193_193629


namespace heather_start_time_later_than_stacy_l193_193556

theorem heather_start_time_later_than_stacy :
  ∀ (distance_initial : ℝ) (H_speed : ℝ) (S_speed : ℝ) (H_distance_when_meet : ℝ),
    distance_initial = 5 ∧
    H_speed = 5 ∧
    S_speed = 6 ∧
    H_distance_when_meet = 1.1818181818181817 →
    ∃ (Δt : ℝ), Δt = 24 / 60 :=
by
  sorry

end heather_start_time_later_than_stacy_l193_193556


namespace avg_age_combined_l193_193834

-- Define the conditions
def avg_age_roomA : ℕ := 45
def avg_age_roomB : ℕ := 20
def num_people_roomA : ℕ := 8
def num_people_roomB : ℕ := 3

-- Definition of the problem statement
theorem avg_age_combined :
  (num_people_roomA * avg_age_roomA + num_people_roomB * avg_age_roomB) / (num_people_roomA + num_people_roomB) = 38 :=
by
  sorry

end avg_age_combined_l193_193834


namespace side_length_square_base_l193_193536

theorem side_length_square_base 
  (height : ℕ) (volume : ℕ) (A : ℕ) (s : ℕ) 
  (h_height : height = 8) 
  (h_volume : volume = 288) 
  (h_base_area : A = volume / height) 
  (h_square_base : A = s ^ 2) :
  s = 6 :=
by
  sorry

end side_length_square_base_l193_193536


namespace contrapositive_quadratic_roots_l193_193326

theorem contrapositive_quadratic_roots (m : ℝ) (h_discriminant : 1 + 4 * m < 0) : m ≤ 0 :=
sorry

end contrapositive_quadratic_roots_l193_193326


namespace inequality_proof_l193_193838
open Nat

theorem inequality_proof (a b : ℝ) (n : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : (1 / a) + (1 / b) = 1) (h4 : n > 0) : 
  (a + b)^n - a^n - b^n ≥ 2^(2 * n) - 2^(n + 1) :=
by
  sorry

end inequality_proof_l193_193838


namespace find_p_for_natural_roots_l193_193779

-- The polynomial is given.
def cubic_polynomial (p x : ℝ) : ℝ := 5 * x^3 - 5 * (p + 1) * x^2 + (71 * p - 1) * x + 1

-- Problem statement to prove that p = 76 is the only real number such that
-- the cubic polynomial cubic_polynomial equals 66 * p has at least two natural number roots.
theorem find_p_for_natural_roots (p : ℝ) :
  (∃ (u v : ℕ), u ≠ v ∧ cubic_polynomial p u = 66 * p ∧ cubic_polynomial p v = 66 * p) ↔ p = 76 :=
by
  sorry

end find_p_for_natural_roots_l193_193779


namespace problem1_problem2_l193_193739

variable (a : ℝ) -- Declaring a as a real number

-- Proof statement for Problem 1
theorem problem1 : (a + 2) * (a - 2) = a^2 - 4 :=
sorry

-- Proof statement for Problem 2
theorem problem2 (h : a ≠ -2) : (a^2 - 4) / (a + 2) + 2 = a :=
sorry

end problem1_problem2_l193_193739


namespace xy_product_approx_25_l193_193272

noncomputable def approx_eq (a b : ℝ) (ε : ℝ := 1e-6) : Prop :=
  |a - b| < ε

theorem xy_product_approx_25 (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) 
  (hxy : x / y = 36) (hy : y = 0.8333333333333334) : approx_eq (x * y) 25 :=
by
  sorry

end xy_product_approx_25_l193_193272


namespace monotonic_on_interval_l193_193275

theorem monotonic_on_interval (k : ℝ) :
  (∀ x y : ℝ, x ≤ y → x ≤ 8 → y ≤ 8 → (4 * x ^ 2 - k * x - 8) ≤ (4 * y ^ 2 - k * y - 8)) ↔ (64 ≤ k) :=
sorry

end monotonic_on_interval_l193_193275


namespace find_other_number_l193_193361

noncomputable def calculateB (lcm hcf a : ℕ) : ℕ :=
  (lcm * hcf) / a

theorem find_other_number :
  ∃ B : ℕ, (calculateB 76176 116 8128) = 1087 :=
by
  use 1087
  sorry

end find_other_number_l193_193361


namespace cost_of_notebook_is_12_l193_193137

/--
In a class of 36 students, a majority purchased notebooks. Each student bought the same number of notebooks (greater than 2). The price of a notebook in cents was double the number of notebooks each student bought, and the total expense was 2772 cents.
Prove that the cost of one notebook in cents is 12.
-/
theorem cost_of_notebook_is_12
  (s n c : ℕ) (total_students : ℕ := 36) 
  (h_majority : s > 18) 
  (h_notebooks : n > 2) 
  (h_cost : c = 2 * n) 
  (h_total_cost : s * c * n = 2772) 
  : c = 12 :=
by sorry

end cost_of_notebook_is_12_l193_193137


namespace mixture_weight_l193_193786

def almonds := 116.67
def walnuts := almonds / 5
def total_weight := almonds + walnuts

theorem mixture_weight : total_weight = 140.004 := by
  sorry

end mixture_weight_l193_193786


namespace evaluate_expression_l193_193640

theorem evaluate_expression : 1 - (-2) * 2 - 3 - (-4) * 2 - 5 - (-6) * 2 = 17 := 
by
  sorry

end evaluate_expression_l193_193640


namespace angle_greater_difference_l193_193499

theorem angle_greater_difference (A B C : ℕ) (h1 : B = 5 * A) (h2 : A + B + C = 180) (h3 : A = 24) 
: C - A = 12 := 
by
  -- Proof omitted
  sorry

end angle_greater_difference_l193_193499


namespace polynomial_remainder_theorem_l193_193363

open Polynomial

theorem polynomial_remainder_theorem (Q : Polynomial ℝ)
  (h1 : Q.eval 20 = 120)
  (h2 : Q.eval 100 = 40) :
  ∃ R : Polynomial ℝ, R.degree < 2 ∧ Q = (X - 20) * (X - 100) * R + (-X + 140) :=
by
  sorry

end polynomial_remainder_theorem_l193_193363


namespace ratio_AD_DC_l193_193729

-- Definitions based on conditions
variable (A B C D : Point)
variable (AB BC AD DB : ℝ)
variable (h1 : AB = 2 * BC)
variable (h2 : AD = 3 / 5 * AB)
variable (h3 : DB = 2 / 5 * AB)

-- Lean statement for the problem
theorem ratio_AD_DC (h1 : AB = 2 * BC) (h2 : AD = 3 / 5 * AB) (h3 : DB = 2 / 5 * AB) :
  AD / (DB + BC) = 2 / 3 := 
by
  sorry

end ratio_AD_DC_l193_193729


namespace geometric_mean_4_16_l193_193441

theorem geometric_mean_4_16 (x : ℝ) (h : x^2 = 4 * 16) : x = 8 ∨ x = -8 :=
sorry

end geometric_mean_4_16_l193_193441


namespace green_or_blue_marble_probability_l193_193348

theorem green_or_blue_marble_probability :
  (4 + 3 : ℝ) / (4 + 3 + 8) = 0.4667 := by
  sorry

end green_or_blue_marble_probability_l193_193348


namespace prime_ge_7_div_30_l193_193641

theorem prime_ge_7_div_30 (p : ℕ) (hp : Nat.Prime p) (h7 : p ≥ 7) : 30 ∣ (p^2 - 1) := 
sorry

end prime_ge_7_div_30_l193_193641


namespace proof_problem_l193_193516

-- Define the given condition as a constant
def condition : Prop := 213 * 16 = 3408

-- Define the statement we need to prove under the given condition
theorem proof_problem (h : condition) : 0.16 * 2.13 = 0.3408 := 
by 
  sorry

end proof_problem_l193_193516


namespace solve_floor_eq_l193_193700

theorem solve_floor_eq (x : ℝ) (hx_pos : 0 < x) (h : (⌊x⌋ : ℝ) * x = 110) : x = 11 := 
sorry

end solve_floor_eq_l193_193700


namespace part1_part2_l193_193535

open Set

variable (U : Set ℤ := {x | -3 ≤ x ∧ x ≤ 3})
variable (A : Set ℤ := {1, 2, 3})
variable (B : Set ℤ := {-1, 0, 1})
variable (C : Set ℤ := {-2, 0, 2})

theorem part1 : A ∪ (B ∩ C) = {0, 1, 2, 3} := by
  sorry

theorem part2 : A ∩ Uᶜ ∪ (B ∪ C) = {3} := by
  sorry

end part1_part2_l193_193535


namespace Mahesh_completes_in_60_days_l193_193651

noncomputable def MaheshWork (W : ℝ) : ℝ :=
    W / 60

variables (W : ℝ)
variables (M R : ℝ)
variables (daysMahesh daysRajesh daysFullRajesh : ℝ)

theorem Mahesh_completes_in_60_days
  (h1 : daysMahesh = 20)
  (h2 : daysRajesh = 30)
  (h3 : daysFullRajesh = 45)
  (hR : R = W / daysFullRajesh)
  (hM : M = (W - R * daysRajesh) / daysMahesh) :
  W / M = 60 :=
by
  sorry

end Mahesh_completes_in_60_days_l193_193651


namespace compute_alpha_powers_l193_193309

variable (α1 α2 α3 : ℂ)

open Complex

-- Given conditions
def condition1 : Prop := α1 + α2 + α3 = 2
def condition2 : Prop := α1^2 + α2^2 + α3^2 = 6
def condition3 : Prop := α1^3 + α2^3 + α3^3 = 14

-- The required proof statement
theorem compute_alpha_powers (h1 : condition1 α1 α2 α3) (h2 : condition2 α1 α2 α3) (h3 : condition3 α1 α2 α3) :
  α1^7 + α2^7 + α3^7 = 46 := by
  sorry

end compute_alpha_powers_l193_193309


namespace intersection_point_value_l193_193138

theorem intersection_point_value (c d: ℤ) (h1: d = 2 * -4 + c) (h2: -4 = 2 * d + c) : d = -4 :=
by
  sorry

end intersection_point_value_l193_193138


namespace jonas_pairs_of_pants_l193_193036

theorem jonas_pairs_of_pants (socks pairs_of_shoes t_shirts new_socks : Nat) (P : Nat) :
  socks = 20 → pairs_of_shoes = 5 → t_shirts = 10 → new_socks = 35 →
  2 * (2 * socks + 2 * pairs_of_shoes + t_shirts + P) = 2 * (2 * socks + 2 * pairs_of_shoes + t_shirts) + 70 →
  P = 5 :=
by
  intros hs hps ht hr htotal
  sorry

end jonas_pairs_of_pants_l193_193036


namespace speed_of_faster_train_l193_193589

-- Definitions based on the conditions.
def length_train_1 : ℝ := 180
def length_train_2 : ℝ := 360
def time_to_cross : ℝ := 21.598272138228943
def speed_slow_train_kmph : ℝ := 30
def speed_fast_train_kmph : ℝ := 60

-- The theorem that needs to be proven.
theorem speed_of_faster_train :
  (length_train_1 + length_train_2) / time_to_cross * 3.6 = speed_slow_train_kmph + speed_fast_train_kmph :=
sorry

end speed_of_faster_train_l193_193589


namespace express_vector_c_as_linear_combination_l193_193244

noncomputable def a : ℝ × ℝ := (1, 1)
noncomputable def b : ℝ × ℝ := (1, -1)
noncomputable def c : ℝ × ℝ := (2, 3)

theorem express_vector_c_as_linear_combination :
  ∃ x y : ℝ, c = (x * (1, 1).1 + y * (1, -1).1, x * (1, 1).2 + y * (1, -1).2) ∧
             x = 5 / 2 ∧ y = -1 / 2 :=
by
  sorry

end express_vector_c_as_linear_combination_l193_193244


namespace gcd_f_of_x_and_x_l193_193620

theorem gcd_f_of_x_and_x (x : ℕ) (hx : 7200 ∣ x) :
  Nat.gcd ((5 * x + 6) * (8 * x + 3) * (11 * x + 9) * (4 * x + 12)) x = 72 :=
sorry

end gcd_f_of_x_and_x_l193_193620


namespace joan_dimes_l193_193681

theorem joan_dimes :
  ∀ (total_dimes_jacket : ℕ) (total_money : ℝ) (value_per_dime : ℝ),
    total_dimes_jacket = 15 →
    total_money = 1.90 →
    value_per_dime = 0.10 →
    ((total_money - (total_dimes_jacket * value_per_dime)) / value_per_dime) = 4 :=
by
  intros total_dimes_jacket total_money value_per_dime h1 h2 h3
  sorry

end joan_dimes_l193_193681


namespace two_digit_multiple_condition_l193_193166

theorem two_digit_multiple_condition :
  ∃ x : ℕ, 10 ≤ x ∧ x < 100 ∧ ∃ k : ℤ, x = 30 * k + 2 :=
by
  sorry

end two_digit_multiple_condition_l193_193166


namespace compute_cubic_sum_l193_193034

theorem compute_cubic_sum (x y : ℝ) (h1 : 1 / x + 1 / y = 4) (h2 : x * y + x ^ 2 + y ^ 2 = 17) : x ^ 3 + y ^ 3 = 52 :=
sorry

end compute_cubic_sum_l193_193034


namespace ratio_of_a_to_b_l193_193527

theorem ratio_of_a_to_b (a y b : ℝ) (h1 : a = 0) (h2 : b = 2 * y) : a / b = 0 :=
by
  sorry

end ratio_of_a_to_b_l193_193527


namespace first_ant_arrives_first_l193_193693

noncomputable def time_crawling (d v : ℝ) : ℝ := d / v

noncomputable def time_riding_caterpillar (d v : ℝ) : ℝ := (d / 2) / (v / 2)

noncomputable def time_riding_grasshopper (d v : ℝ) : ℝ := (d / 2) / (10 * v)

noncomputable def time_ant1 (d v : ℝ) : ℝ := time_crawling d v

noncomputable def time_ant2 (d v : ℝ) : ℝ := time_riding_caterpillar d v + time_riding_grasshopper d v

theorem first_ant_arrives_first (d v : ℝ) (h_v_pos : 0 < v): time_ant1 d v < time_ant2 d v := by
  -- provide the justification for the theorem here
  sorry

end first_ant_arrives_first_l193_193693


namespace total_words_story_l193_193475

def words_per_line : ℕ := 10
def lines_per_page : ℕ := 20
def pages_filled : ℚ := 1.5
def words_left : ℕ := 100

theorem total_words_story : 
    words_per_line * lines_per_page * pages_filled + words_left = 400 := 
by
sorry

end total_words_story_l193_193475


namespace max_flowers_used_min_flowers_used_l193_193261

-- Part (a) Setup
def max_flowers (C M : ℕ) (flower_views : ℕ) := 2 * C + M = flower_views
def max_T (C M : ℕ) := C + M

-- Given conditions
theorem max_flowers_used :
  (∀ C M : ℕ, max_flowers C M 36 → max_T C M = 36) :=
by sorry

-- Part (b) Setup
def min_flowers (C M : ℕ) (flower_views : ℕ) := 2 * C + M = flower_views
def min_T (C M : ℕ) := C + M

-- Given conditions
theorem min_flowers_used :
  (∀ C M : ℕ, min_flowers C M 48 → min_T C M = 24) :=
by sorry

end max_flowers_used_min_flowers_used_l193_193261


namespace total_races_needed_to_determine_champion_l193_193748

-- Defining the initial conditions
def num_sprinters : ℕ := 256
def lanes : ℕ := 8
def sprinters_per_race := lanes
def eliminated_per_race := sprinters_per_race - 1

-- The statement to be proved: The number of races required to determine the champion
theorem total_races_needed_to_determine_champion :
  ∃ (races : ℕ), races = 37 ∧
  ∀ s : ℕ, s = num_sprinters → 
  ∀ l : ℕ, l = lanes → 
  ∃ e : ℕ, e = eliminated_per_race →
  s - (races * e) = 1 :=
by sorry

end total_races_needed_to_determine_champion_l193_193748


namespace taco_truck_earnings_l193_193280

/-
Question: How many dollars did the taco truck make during the lunch rush?
Conditions:
1. Soft tacos are $2 each.
2. Hard shell tacos are $5 each.
3. The family buys 4 hard shell tacos and 3 soft tacos.
4. There are ten other customers.
5. Each of the ten other customers buys 2 soft tacos.
Answer: The taco truck made $66 during the lunch rush.
-/

theorem taco_truck_earnings :
  let soft_taco_price := 2
  let hard_taco_price := 5
  let family_hard_tacos := 4
  let family_soft_tacos := 3
  let other_customers := 10
  let other_customers_soft_tacos := 2
  (family_hard_tacos * hard_taco_price + family_soft_tacos * soft_taco_price +
   other_customers * other_customers_soft_tacos * soft_taco_price) = 66 := by
  sorry

end taco_truck_earnings_l193_193280


namespace necessary_but_not_sufficient_condition_l193_193964

variables {Point Line Plane : Type} 

-- Definitions for the problem conditions
def is_subset_of (a : Line) (α : Plane) : Prop := sorry
def parallel_plane (a : Line) (β : Plane) : Prop := sorry
def parallel_lines (a b : Line) : Prop := sorry
def parallel_planes (α β : Plane) : Prop := sorry

-- The statement of the problem
theorem necessary_but_not_sufficient_condition (a b : Line) (α β : Plane) 
  (h1 : is_subset_of a α) (h2 : is_subset_of b β) :
  (parallel_plane a β ∧ parallel_plane b α) ↔ 
  (¬ parallel_planes α β ∧ sorry) :=
sorry

end necessary_but_not_sufficient_condition_l193_193964


namespace adult_ticket_cost_l193_193258

theorem adult_ticket_cost (C : ℝ) (h1 : ∀ (a : ℝ), a = C + 8)
  (h2 : ∀ (s : ℝ), s = C + 4)
  (h3 : 5 * C + 2 * (C + 8) + 2 * (C + 4) = 150) :
  ∃ (a : ℝ), a = 22 :=
by {
  sorry
}

end adult_ticket_cost_l193_193258


namespace find_a_l193_193634

theorem find_a (a b : ℝ) (h1 : 0 < a ∧ 0 < b) (h2 : a^b = b^a) (h3 : b = 4 * a) : 
  a = (4 : ℝ)^(1 / 3) :=
by
  sorry

end find_a_l193_193634


namespace P_subset_Q_l193_193593

-- Define the set P
def P := {x : ℝ | 0 < Real.log x / Real.log 2 ∧ Real.log x / Real.log 2 < 1}

-- Define the set Q
def Q := {x : ℝ | x ≤ 2}

-- Prove P ⊆ Q
theorem P_subset_Q : P ⊆ Q :=
by
  sorry

end P_subset_Q_l193_193593


namespace length_of_BC_l193_193354

theorem length_of_BC (AB AC AM : ℝ) (hAB : AB = 5) (hAC : AC = 8) (hAM : AM = 4.5) : 
  ∃ BC, BC = Real.sqrt 97 :=
by
  sorry

end length_of_BC_l193_193354


namespace donuts_distribution_l193_193754

theorem donuts_distribution (kinds total min_each : ℕ) (h_kinds : kinds = 4) (h_total : total = 7) (h_min_each : min_each = 1) :
  ∃ n : ℕ, n = 20 := by
  sorry

end donuts_distribution_l193_193754


namespace range_of_a_l193_193594

theorem range_of_a (a : ℝ) :
  let A := {x : ℝ | 2 * a ≤ x ∧ x ≤ a + 3}
  let B := {x : ℝ | 5 < x}
  (A ∩ B = ∅) ↔ a ∈ {a : ℝ | a ≤ 2 ∨ a > 3} :=
by
  sorry

end range_of_a_l193_193594


namespace carol_seq_last_three_digits_l193_193823

/-- Carol starts to make a list, in increasing order, of the positive integers that have 
    a first digit of 2. She writes 2, 20, 21, 22, ...
    Prove that the three-digit number formed by the 1198th, 1199th, 
    and 1200th digits she wrote is 218. -/
theorem carol_seq_last_three_digits : 
  (digits_1198th_1199th_1200th = 218) :=
by
  sorry

end carol_seq_last_three_digits_l193_193823


namespace total_animals_counted_l193_193370

theorem total_animals_counted :
  let antelopes := 80
  let rabbits := antelopes + 34
  let total_rabbits_antelopes := rabbits + antelopes
  let hyenas := total_rabbits_antelopes - 42
  let wild_dogs := hyenas + 50
  let leopards := rabbits / 2
  (antelopes + rabbits + hyenas + wild_dogs + leopards) = 605 :=
by
  let antelopes := 80
  let rabbits := antelopes + 34
  let total_rabbits_antelopes := rabbits + antelopes
  let hyenas := total_rabbits_antelopes - 42
  let wild_dogs := hyenas + 50
  let leopards := rabbits / 2
  show (antelopes + rabbits + hyenas + wild_dogs + leopards) = 605
  sorry

end total_animals_counted_l193_193370


namespace 1_part1_2_part2_l193_193368

/-
Define M and N sets
-/
def M : Set ℝ := {x | x ≥ 1 / 2}
def N : Set ℝ := {y | y ≤ 1}

/-
Theorem 1: Difference set M - N
-/
theorem part1 : (M \ N) = {x | x > 1} := by
  sorry

/-
Define A and B sets and the condition A - B = ∅
-/
def A (a : ℝ) : Set ℝ := {x | 0 < a * x - 1 ∧ a * x - 1 ≤ 5}
def B : Set ℝ := {y | -1 / 2 < y ∧ y ≤ 2}

/-
Theorem 2: Range of values for a
-/
theorem part2 (a : ℝ) (h : A a \ B = ∅) : a ∈ Set.Iio (-12) ∪ Set.Ici 3 := by
  sorry

end 1_part1_2_part2_l193_193368


namespace pollywog_maturation_rate_l193_193032

theorem pollywog_maturation_rate :
  ∀ (initial_pollywogs : ℕ) (melvin_rate : ℕ) (total_days : ℕ) (melvin_days : ℕ) (remaining_pollywogs : ℕ),
  initial_pollywogs = 2400 →
  melvin_rate = 10 →
  total_days = 44 →
  melvin_days = 20 →
  remaining_pollywogs = initial_pollywogs - (melvin_rate * melvin_days) →
  (total_days * (remaining_pollywogs / (total_days - melvin_days))) = remaining_pollywogs →
  (remaining_pollywogs / (total_days - melvin_days)) = 50 := 
by
  intros initial_pollywogs melvin_rate total_days melvin_days remaining_pollywogs
  intros h_initial h_melvin h_total h_melvin_days h_remaining h_eq
  sorry

end pollywog_maturation_rate_l193_193032


namespace arccos_gt_arctan_l193_193613

theorem arccos_gt_arctan (x : ℝ) (h : -1 ≤ x ∧ x < 1/2) : Real.arccos x > Real.arctan x :=
sorry

end arccos_gt_arctan_l193_193613


namespace original_price_of_cycle_l193_193916

theorem original_price_of_cycle (selling_price : ℝ) (loss_percentage : ℝ) (original_price : ℝ) 
  (h1 : selling_price = 1610)
  (h2 : loss_percentage = 30) 
  (h3 : selling_price = original_price * (1 - loss_percentage / 100)) : 
  original_price = 2300 := 
by 
  sorry

end original_price_of_cycle_l193_193916


namespace remainder_of_19_pow_60_mod_7_l193_193630

theorem remainder_of_19_pow_60_mod_7 : (19 ^ 60) % 7 = 1 := 
by {
  sorry
}

end remainder_of_19_pow_60_mod_7_l193_193630


namespace sequence_converges_to_one_l193_193196

noncomputable def u (n : ℕ) : ℝ :=
1 + (Real.sin n) / n

theorem sequence_converges_to_one :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |u n - 1| ≤ ε :=
sorry

end sequence_converges_to_one_l193_193196


namespace sin_sum_identity_l193_193461

theorem sin_sum_identity 
  (α : ℝ) 
  (h : Real.sin (2 * Real.pi / 3 - α) + Real.sin α = 4 * Real.sqrt 3 / 5) : 
  Real.sin (α + 7 * Real.pi / 6) = -4 / 5 := 
by 
  sorry

end sin_sum_identity_l193_193461


namespace salary_for_May_l193_193997

variable (J F M A May : ℕ)

axiom condition1 : (J + F + M + A) / 4 = 8000
axiom condition2 : (F + M + A + May) / 4 = 8800
axiom condition3 : J = 3300

theorem salary_for_May : May = 6500 :=
by sorry

end salary_for_May_l193_193997


namespace proof_y_minus_x_l193_193940

theorem proof_y_minus_x (x y : ℤ) (h1 : x + y = 540) (h2 : x = (4 * y) / 5) : y - x = 60 :=
sorry

end proof_y_minus_x_l193_193940


namespace rectangle_to_square_l193_193076

variable (k n : ℕ)

theorem rectangle_to_square (h1 : k > 5) (h2 : k * (k - 5) = n^2) : n = 6 := by 
  sorry

end rectangle_to_square_l193_193076


namespace number_of_companion_relation_subsets_l193_193511

def isCompanionRelationSet (A : Set ℚ) : Prop :=
  ∀ x ∈ A, (x ≠ 0 → (1 / x) ∈ A)

def M : Set ℚ := {-1, 0, 1 / 3, 1 / 2, 1, 2, 3, 4}

theorem number_of_companion_relation_subsets :
  ∃ n, n = 15 ∧
  (∀ A ⊆ M, isCompanionRelationSet A) :=
sorry

end number_of_companion_relation_subsets_l193_193511


namespace invitations_sent_out_l193_193254

-- Define the conditions
def RSVPed (I : ℝ) : ℝ := 0.9 * I
def Showed_up (I : ℝ) : ℝ := 0.8 * RSVPed I
def No_gift : ℝ := 10
def Thank_you_cards : ℝ := 134

-- Prove the number of invitations
theorem invitations_sent_out : ∃ I : ℝ, Showed_up I - No_gift = Thank_you_cards ∧ I = 200 :=
by
  sorry

end invitations_sent_out_l193_193254


namespace equivalent_after_eliminating_denominators_l193_193611

theorem equivalent_after_eliminating_denominators (x : ℝ) (h : 1 + 2 / (x - 1) = (x - 5) / (x - 3)) :
  (x - 1) * (x - 3) + 2 * (x - 3) = (x - 5) * (x - 1) :=
sorry

end equivalent_after_eliminating_denominators_l193_193611


namespace problem_inequality_l193_193078

theorem problem_inequality (a b c : ℝ) (h_pos: a > 0 ∧ b > 0 ∧ c > 0)
  (h: (a / b + b / c + c / a) + (b / a + c / b + a / c) = 9) :
  a / b + b / c + c / a = 4.5 :=
by
  sorry

end problem_inequality_l193_193078


namespace children_ticket_cost_is_8_l193_193559

-- Defining the costs of different tickets
def adult_ticket_cost : ℕ := 11
def senior_ticket_cost : ℕ := 9
def total_tickets_cost : ℕ := 64

-- Number of tickets needed
def number_of_adult_tickets : ℕ := 2
def number_of_senior_tickets : ℕ := 2
def number_of_children_tickets : ℕ := 3

-- Defining the total cost equation using the price of children's tickets (C)
def total_cost (children_ticket_cost : ℕ) : ℕ :=
  number_of_adult_tickets * adult_ticket_cost +
  number_of_senior_tickets * senior_ticket_cost +
  number_of_children_tickets * children_ticket_cost

-- Statement to prove that the children's ticket cost is $8
theorem children_ticket_cost_is_8 : (C : ℕ) → total_cost C = total_tickets_cost → C = 8 :=
by
  intro C h
  sorry

end children_ticket_cost_is_8_l193_193559


namespace range_of_values_l193_193265

noncomputable def f (x : ℝ) : ℝ := 2^(1 + x^2) - 1 / (1 + x^2)

theorem range_of_values (x : ℝ) : f (2 * x) > f (x - 3) ↔ x < -3 ∨ x > 1 := 
by
  sorry

end range_of_values_l193_193265


namespace number_of_parents_l193_193894

theorem number_of_parents (n m : ℕ) 
  (h1 : n + m = 31) 
  (h2 : 15 + m = n) 
  : n = 23 := 
by 
  sorry

end number_of_parents_l193_193894


namespace probability_of_correct_match_l193_193198

theorem probability_of_correct_match :
  let n := 3
  let total_arrangements := Nat.factorial n
  let correct_arrangements := 1
  let probability := correct_arrangements / total_arrangements
  probability = ((1: ℤ) / 6) :=
by
  sorry

end probability_of_correct_match_l193_193198


namespace fraction_pizza_covered_by_pepperoni_l193_193643

/--
Given that six pepperoni circles fit exactly across the diameter of a 12-inch pizza
and a total of 24 circles of pepperoni are placed on the pizza without overlap,
prove that the fraction of the pizza covered by pepperoni is 2/3.
-/
theorem fraction_pizza_covered_by_pepperoni : 
  (∃ d r : ℝ, 6 * r = d ∧ d = 12 ∧ (r * r * π * 24) / (6 * 6 * π) = 2 / 3) := 
sorry

end fraction_pizza_covered_by_pepperoni_l193_193643


namespace x_plus_y_equals_two_l193_193891

variable (x y : ℝ)

def condition1 : Prop := (x - 1) ^ 2017 + 2013 * (x - 1) = -1
def condition2 : Prop := (y - 1) ^ 2017 + 2013 * (y - 1) = 1

theorem x_plus_y_equals_two (h1 : condition1 x) (h2 : condition2 y) : x + y = 2 :=
  sorry

end x_plus_y_equals_two_l193_193891


namespace parabola_equation_l193_193133

theorem parabola_equation (a b c d e f: ℤ) (ha: a = 2) (hb: b = 0) (hc: c = 0) (hd: d = -16) (he: e = -1) (hf: f = 32) :
  ∃ x y : ℝ, 2 * x ^ 2 - 16 * x + 32 - y = 0 ∧ gcd (abs a) (gcd (abs b) (gcd (abs c) (gcd (abs d) (gcd (abs e) (abs f))))) = 1 :=
by
  sorry

end parabola_equation_l193_193133


namespace decimal_palindrome_multiple_l193_193095

def is_decimal_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

theorem decimal_palindrome_multiple (n : ℕ) (h : ¬ (10 ∣ n)) : 
  ∃ m : ℕ, is_decimal_palindrome m ∧ m % n = 0 :=
by sorry

end decimal_palindrome_multiple_l193_193095


namespace original_couch_price_l193_193608

def chair_price : ℝ := sorry
def table_price := 3 * chair_price
def couch_price := 5 * table_price
def bookshelf_price := 0.5 * couch_price

def discounted_chair_price := 0.8 * chair_price
def discounted_couch_price := 0.9 * couch_price
def total_price_before_tax := discounted_chair_price + table_price + discounted_couch_price + bookshelf_price
def total_price_after_tax := total_price_before_tax * 1.08

theorem original_couch_price (budget : ℝ) (h_budget : budget = 900) : 
  total_price_after_tax = budget → couch_price = 503.85 :=
by
  sorry

end original_couch_price_l193_193608


namespace transformations_map_onto_itself_l193_193183

noncomputable def recurring_pattern_map_count (s : ℝ) : ℕ := sorry

theorem transformations_map_onto_itself (s : ℝ) :
  recurring_pattern_map_count s = 2 := sorry

end transformations_map_onto_itself_l193_193183


namespace how_many_lassis_l193_193704

def lassis_per_mango : ℕ := 15 / 3

def lassis15mangos : ℕ := 15

theorem how_many_lassis (H : lassis_per_mango = 5) : lassis15mangos * lassis_per_mango = 75 :=
by
  rw [H]
  sorry

end how_many_lassis_l193_193704


namespace tom_books_read_in_may_l193_193733

def books_read_in_june := 6
def books_read_in_july := 10
def total_books_read := 18

theorem tom_books_read_in_may : total_books_read - (books_read_in_june + books_read_in_july) = 2 :=
by sorry

end tom_books_read_in_may_l193_193733


namespace ratio_first_part_l193_193706

theorem ratio_first_part (x : ℝ) (h1 : 180 / 100 * 5 = x) : x = 9 :=
by sorry

end ratio_first_part_l193_193706


namespace sum_of_tens_and_ones_digit_of_7_pow_17_l193_193800

def tens_digit (n : ℕ) : ℕ :=
(n / 10) % 10

def ones_digit (n : ℕ) : ℕ :=
n % 10

theorem sum_of_tens_and_ones_digit_of_7_pow_17 : 
  tens_digit (7^17) + ones_digit (7^17) = 7 :=
by
  sorry

end sum_of_tens_and_ones_digit_of_7_pow_17_l193_193800


namespace binomial_identity_l193_193597

theorem binomial_identity :
  (Nat.choose 16 6 = 8008) → (Nat.choose 16 7 = 11440) → (Nat.choose 16 8 = 12870) →
  Nat.choose 18 8 = 43758 :=
by
  intros h1 h2 h3
  sorry

end binomial_identity_l193_193597


namespace base_k_conversion_l193_193912

theorem base_k_conversion (k : ℕ) (hk : 4 * k + 4 = 36) : 6 * 8 + 7 = 55 :=
by
  -- Proof skipped
  sorry

end base_k_conversion_l193_193912


namespace subcommittee_count_l193_193742

theorem subcommittee_count :
  let republicans := 10
  let democrats := 8
  let subcommittee_republicans := 4
  let subcommittee_democrats := 3
  let choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  choose republicans subcommittee_republicans * choose democrats subcommittee_democrats = 11760 :=
by
  let republicans := 10
  let democrats := 8
  let subcommittee_republicans := 4
  let subcommittee_democrats := 3
  let choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  sorry

end subcommittee_count_l193_193742


namespace intersection_of_A_and_B_l193_193273

theorem intersection_of_A_and_B :
  let A := {x : ℝ | x > 0}
  let B := {x : ℝ | x^2 - 2*x - 3 < 0}
  (A ∩ B) = {x : ℝ | 0 < x ∧ x < 3} := by
  sorry

end intersection_of_A_and_B_l193_193273


namespace round_trip_time_l193_193187

noncomputable def time_to_complete_trip (speed_without_load speed_with_load distance rest_stops_in_minutes : ℝ) : ℝ :=
  let rest_stops_in_hours := rest_stops_in_minutes / 60
  let half_rest_time := 2 * rest_stops_in_hours
  let total_rest_time := 2 * half_rest_time
  let travel_time_with_load := distance / speed_with_load
  let travel_time_without_load := distance / speed_without_load
  travel_time_with_load + travel_time_without_load + total_rest_time

theorem round_trip_time :
  time_to_complete_trip 13 11 143 30 = 26 :=
sorry

end round_trip_time_l193_193187


namespace max_value_of_P_l193_193697

noncomputable def P (a b c : ℝ) : ℝ :=
  (2 / (a^2 + 1)) - (2 / (b^2 + 1)) + (3 / (c^2 + 1))

theorem max_value_of_P (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a * b * c + a + c = b) :
  ∃ x, x = 1 ∧ ∀ y, (y = P a b c) → y ≤ x :=
sorry

end max_value_of_P_l193_193697


namespace flour_already_added_l193_193439

theorem flour_already_added (sugar flour salt additional_flour : ℕ) 
  (h1 : sugar = 9) 
  (h2 : flour = 14) 
  (h3 : salt = 40)
  (h4 : additional_flour = sugar + 1) : 
  flour - additional_flour = 4 :=
by
  sorry

end flour_already_added_l193_193439


namespace mrs_choi_profit_percentage_l193_193638

theorem mrs_choi_profit_percentage :
  ∀ (original_price selling_price : ℝ) (broker_percentage : ℝ),
    original_price = 80000 →
    selling_price = 100000 →
    broker_percentage = 0.05 →
    (selling_price - (broker_percentage * original_price) - original_price) / original_price * 100 = 20 :=
by
  intros original_price selling_price broker_percentage h1 h2 h3
  sorry

end mrs_choi_profit_percentage_l193_193638


namespace sum_a6_a7_a8_is_32_l193_193489

noncomputable def geom_seq (q : ℝ) (a₁ : ℝ) (n : ℕ) : ℝ := a₁ * q^(n-1)

theorem sum_a6_a7_a8_is_32 (q a₁ : ℝ) 
  (h1 : geom_seq q a₁ 1 + geom_seq q a₁ 2 + geom_seq q a₁ 3 = 1)
  (h2 : geom_seq q a₁ 2 + geom_seq q a₁ 3 + geom_seq q a₁ 4 = 2) :
  geom_seq q a₁ 6 + geom_seq q a₁ 7 + geom_seq q a₁ 8 = 32 :=
sorry

end sum_a6_a7_a8_is_32_l193_193489


namespace trig_expression_evaluation_l193_193396

theorem trig_expression_evaluation
  (α : ℝ)
  (h_tan_α : Real.tan α = 3) :
  (Real.sin (π - α) - Real.sin (π / 2 + α)) / 
  (Real.cos (3 * π / 2 - α) + 2 * Real.cos (-π + α)) = -2 / 5 := by
  sorry

end trig_expression_evaluation_l193_193396


namespace max_cylinder_volume_l193_193393

/-- Given a rectangle with perimeter 18 cm, when rotating it around one side to form a cylinder, 
    the maximum volume of the cylinder and the corresponding side length of the rectangle. -/
theorem max_cylinder_volume (x y : ℝ) (h_perimeter : 2 * (x + y) = 18) (hx : x > 0) (hy : y > 0)
  (h_cylinder_volume : ∃ (V : ℝ), V = π * x * (y / 2)^2) :
  (x = 3 ∧ y = 6 ∧ ∀ V, V = 108 * π) := sorry

end max_cylinder_volume_l193_193393


namespace value_of_A_l193_193893

theorem value_of_A (M A T E H : ℤ) (hH : H = 8) (h1 : M + A + T + H = 31) (h2 : T + E + A + M = 40) (h3 : M + E + E + T = 44) (h4 : M + A + T + E = 39) : A = 12 :=
by
  sorry

end value_of_A_l193_193893


namespace area_of_rectangular_field_l193_193037

theorem area_of_rectangular_field (W L : ℕ) (hL : L = 10) (hFencing : 2 * W + L = 146) : W * L = 680 := by
  sorry

end area_of_rectangular_field_l193_193037


namespace eggs_per_group_l193_193815

-- Define the conditions
def num_eggs : ℕ := 18
def num_groups : ℕ := 3

-- Theorem stating number of eggs per group
theorem eggs_per_group : num_eggs / num_groups = 6 :=
by
  sorry

end eggs_per_group_l193_193815


namespace infinite_series_sum_l193_193067

theorem infinite_series_sum :
  ∑' n : ℕ, (n + 1) * (1 / 1950)^n = 3802500 / 3802601 :=
by
  sorry

end infinite_series_sum_l193_193067


namespace angle_measure_l193_193003

theorem angle_measure (A B C : ℝ) (h1 : A = B) (h2 : A + B = 110 ∨ (A = 180 - 110)) :
  A = 70 ∨ A = 55 := by
  sorry

end angle_measure_l193_193003


namespace trapezoidal_garden_solutions_l193_193292

theorem trapezoidal_garden_solutions :
  ∃ (b1 b2 : ℕ), 
    (1800 = (60 * (b1 + b2)) / 2) ∧
    (b1 % 10 = 0) ∧ (b2 % 10 = 0) ∧
    (∃ (n : ℕ), n = 4) := 
sorry

end trapezoidal_garden_solutions_l193_193292


namespace cost_of_pure_milk_l193_193140

theorem cost_of_pure_milk (C : ℝ) (total_milk : ℝ) (pure_milk : ℝ) (water : ℝ) (profit : ℝ) :
  total_milk = pure_milk + water → profit = (total_milk * C) - (pure_milk * C) → profit = 35 → C = 7 :=
by
  intros h1 h2 h3
  sorry

end cost_of_pure_milk_l193_193140


namespace linear_function_quadrant_l193_193897

theorem linear_function_quadrant (x y : ℝ) (h : y = -3 * x + 2) :
  ¬ (x > 0 ∧ y > 0) :=
by
  sorry

end linear_function_quadrant_l193_193897


namespace complex_number_problem_l193_193380

variables {a b c x y z : ℂ}

theorem complex_number_problem (h1 : a = (b + c) / (x - 2))
    (h2 : b = (c + a) / (y - 2))
    (h3 : c = (a + b) / (z - 2))
    (h4 : x * y + y * z + z * x = 67)
    (h5 : x + y + z = 2010) :
    x * y * z = -5892 :=
sorry

end complex_number_problem_l193_193380


namespace right_triangle_other_side_l193_193195

theorem right_triangle_other_side (c a : ℝ) (h_c : c = 10) (h_a : a = 6) : ∃ b : ℝ, b^2 = c^2 - a^2 ∧ b = 8 :=
by
  use 8
  rw [h_c, h_a]
  simp
  sorry

end right_triangle_other_side_l193_193195


namespace score_stability_l193_193776

theorem score_stability (mean_A mean_B : ℝ) (h_mean_eq : mean_A = mean_B)
  (variance_A variance_B : ℝ) (h_variance_A : variance_A = 0.06) (h_variance_B : variance_B = 0.35) :
  variance_A < variance_B :=
by
  -- Theorem statement and conditions sufficient to build successfully
  sorry

end score_stability_l193_193776


namespace six_digit_number_multiple_of_7_l193_193271

theorem six_digit_number_multiple_of_7 (d : ℕ) (hd : d ≤ 9) :
  (∃ k : ℤ, 56782 + d * 10 = 7 * k) ↔ (d = 0 ∨ d = 7) := by
sorry

end six_digit_number_multiple_of_7_l193_193271


namespace sequence_a_2017_l193_193356

theorem sequence_a_2017 :
  (∃ (a : ℕ → ℚ), (a 1 = 1) ∧ (∀ n : ℕ, 0 < n → a (n + 1) = 2016 * a n / (2014 * a n + 2016)) → a 2017 = 1008 / (1007 * 2017 + 1)) :=
by
  sorry

end sequence_a_2017_l193_193356


namespace trigonometric_inequality_l193_193877

theorem trigonometric_inequality (x : ℝ) : 
  -1/3 ≤ (6 * Real.cos x + Real.sin x - 5) / (2 * Real.cos x - 3 * Real.sin x - 5) ∧
  (6 * Real.cos x + Real.sin x - 5) / (2 * Real.cos x - 3 * Real.sin x - 5) ≤ 3 := 
sorry

end trigonometric_inequality_l193_193877


namespace problem_l193_193524

variable (α : ℝ)

def setA : Set ℝ := {Real.sin α, Real.cos α, 1}
def setB : Set ℝ := {Real.sin α ^ 2, Real.sin α + Real.cos α, 0}
theorem problem (h : setA α = setB α) : Real.sin α ^ 2009 + Real.cos α ^ 2009 = -1 := 
by 
  sorry

end problem_l193_193524


namespace painter_remaining_time_l193_193306

-- Define the initial conditions
def total_rooms : ℕ := 11
def hours_per_room : ℕ := 7
def painted_rooms : ℕ := 2

-- Define the remaining rooms to paint
def remaining_rooms : ℕ := total_rooms - painted_rooms

-- Define the proof problem: the remaining time to paint the rest of the rooms
def remaining_hours : ℕ := remaining_rooms * hours_per_room

theorem painter_remaining_time :
  remaining_hours = 63 :=
sorry

end painter_remaining_time_l193_193306


namespace max_apartment_size_l193_193367

/-- Define the rental rate and the maximum rent Michael can afford. -/
def rental_rate : ℝ := 1.20
def max_rent : ℝ := 720

/-- State the problem in Lean: Prove that the maximum apartment size Michael should consider is 600 square feet. -/
theorem max_apartment_size :
  ∃ s : ℝ, rental_rate * s = max_rent ∧ s = 600 := by
  sorry

end max_apartment_size_l193_193367


namespace infinite_series_sum_l193_193221

theorem infinite_series_sum :
  ∑' (n : ℕ), (n + 1) * (1 / 1000)^n = 3000000 / 998001 :=
by sorry

end infinite_series_sum_l193_193221


namespace reena_interest_paid_l193_193202

-- Definitions based on conditions
def principal : ℝ := 1200
def rate : ℝ := 0.03
def time : ℝ := 3

-- Definition of simple interest calculation based on conditions
def simple_interest (P R T : ℝ) : ℝ := P * R * T

-- Statement to prove that Reena paid $108 as interest
theorem reena_interest_paid : simple_interest principal rate time = 108 := by
  sorry

end reena_interest_paid_l193_193202


namespace equal_utilities_l193_193963

-- Conditions
def utility (juggling coding : ℕ) : ℕ := juggling * coding

def wednesday_utility (s : ℕ) : ℕ := utility s (12 - s)
def thursday_utility (s : ℕ) : ℕ := utility (6 - s) (s + 4)

-- Theorem
theorem equal_utilities (s : ℕ) (h : wednesday_utility s = thursday_utility s) : s = 12 / 5 := 
by sorry

end equal_utilities_l193_193963


namespace relationship_among_a_b_c_l193_193957

noncomputable def a := Real.log 2 / 2
noncomputable def b := Real.log 3 / 3
noncomputable def c := Real.log 5 / 5

theorem relationship_among_a_b_c : c < a ∧ a < b := by
  sorry

end relationship_among_a_b_c_l193_193957


namespace tribe_leadership_choices_l193_193563

theorem tribe_leadership_choices :
  let members := 15
  let ways_to_choose_chief := members
  let remaining_after_chief := members - 1
  let ways_to_choose_supporting_chiefs := Nat.choose remaining_after_chief 2
  let remaining_after_supporting_chiefs := remaining_after_chief - 2
  let ways_to_choose_officers_A := Nat.choose remaining_after_supporting_chiefs 2
  let remaining_for_assistants_A := remaining_after_supporting_chiefs - 2
  let ways_to_choose_assistants_A := Nat.choose remaining_for_assistants_A 2 * Nat.choose (remaining_for_assistants_A - 2) 2
  let remaining_after_A := remaining_for_assistants_A - 2
  let ways_to_choose_officers_B := Nat.choose remaining_after_A 2
  let remaining_for_assistants_B := remaining_after_A - 2
  let ways_to_choose_assistants_B := Nat.choose remaining_for_assistants_B 2 * Nat.choose (remaining_for_assistants_B - 2) 2
  (ways_to_choose_chief * ways_to_choose_supporting_chiefs *
  ways_to_choose_officers_A * ways_to_choose_assistants_A *
  ways_to_choose_officers_B * ways_to_choose_assistants_B = 400762320000) := by
  sorry

end tribe_leadership_choices_l193_193563


namespace smallest_fraction_denominator_l193_193211

theorem smallest_fraction_denominator (p q : ℕ) :
  (1:ℚ) / 2014 < p / q ∧ p / q < (1:ℚ) / 2013 → q = 4027 :=
sorry

end smallest_fraction_denominator_l193_193211


namespace smallest_integer_in_odd_set_l193_193637

theorem smallest_integer_in_odd_set (is_odd: ℤ → Prop)
  (median: ℤ) (greatest: ℤ) (smallest: ℤ) 
  (h1: median = 126)
  (h2: greatest = 153) 
  (h3: ∀ x, is_odd x ↔ ∃ k: ℤ, x = 2*k + 1)
  (h4: ∀ a b c, median = (a+b) / 2 → c = a → a ≤ b)
  : 
  smallest = 100 :=
sorry

end smallest_integer_in_odd_set_l193_193637


namespace altitude_in_scientific_notation_l193_193753

theorem altitude_in_scientific_notation : 
  (389000 : ℝ) = 3.89 * (10 : ℝ) ^ 5 :=
by
  sorry

end altitude_in_scientific_notation_l193_193753


namespace towel_length_decrease_l193_193433

theorem towel_length_decrease (L B : ℝ) (HL1: L > 0) (HB1: B > 0)
  (length_percent_decr : ℝ) (breadth_decr : B' = 0.8 * B) 
  (area_decr : (L' * B') = 0.64 * (L * B)) :
  (L' = 0.8 * L) ∧ (length_percent_decrease = 20) := by
  sorry

end towel_length_decrease_l193_193433


namespace ratio_of_place_values_l193_193297

-- Definitions based on conditions
def place_value_tens_digit : ℝ := 10
def place_value_hundredths_digit : ℝ := 0.01

-- Statement to prove
theorem ratio_of_place_values :
  (place_value_tens_digit / place_value_hundredths_digit) = 1000 :=
by
  sorry

end ratio_of_place_values_l193_193297


namespace find_a_l193_193618

-- Definitions and conditions from the problem
def M (a : ℝ) : Set ℝ := {1, 2, a^2 - 3*a - 1}
def N (a : ℝ) : Set ℝ := {-1, a, 3}
def intersection_is_three (a : ℝ) : Prop := M a ∩ N a = {3}

-- The theorem we want to prove
theorem find_a (a : ℝ) (h : intersection_is_three a) : a = 4 :=
by
  sorry

end find_a_l193_193618


namespace find_k_l193_193440

theorem find_k (k : ℝ) : 
  let a := 6
  let b := 25
  let root := (-25 - Real.sqrt 369) / 12
  6 * root^2 + 25 * root + k = 0 → k = 32 / 3 :=
sorry

end find_k_l193_193440


namespace min_expression_value_l193_193737

variable {a : ℕ → ℝ}
variable (m n : ℕ)
variable (q : ℝ)

axiom pos_seq (n : ℕ) : a n > 0
axiom geom_seq (n : ℕ) : a (n + 1) = q * a n
axiom seq_condition : a 7 = a 6 + 2 * a 5
axiom exists_terms :
  ∃ m n : ℕ, m > 0 ∧ n > 0 ∧ (Real.sqrt (a m * a n) = 4 * a 1)

theorem min_expression_value : 
  (∃m n : ℕ, m > 0 ∧ n > 0 ∧ (Real.sqrt (a m * a n) = 4 * a 1) ∧ 
  a 7 = a 6 + 2 * a 5 ∧ 
  (∀ n, a n > 0 ∧ a (n + 1) = q * a n)) → 
  (1 / m + 4 / n) ≥ 3 / 2 :=
sorry

end min_expression_value_l193_193737


namespace equip_20posts_with_5new_weapons_l193_193550

/-- 
Theorem: In a line of 20 defense posts, the number of ways to equip 5 different new weapons 
such that:
1. The first and last posts are not equipped with new weapons.
2. Each set of 5 consecutive posts has at least one post equipped with a new weapon.
3. No two adjacent posts are equipped with new weapons.
is 69600. 
-/
theorem equip_20posts_with_5new_weapons : ∃ ways : ℕ, ways = 69600 :=
by
  sorry

end equip_20posts_with_5new_weapons_l193_193550


namespace octahedron_has_constant_perimeter_cross_sections_l193_193359

structure Octahedron :=
(edge_length : ℝ)

def all_cross_sections_same_perimeter (oct : Octahedron) :=
  ∀ (face1 face2 : ℝ), (face1 = face2)

theorem octahedron_has_constant_perimeter_cross_sections (oct : Octahedron) :
  all_cross_sections_same_perimeter oct :=
  sorry

end octahedron_has_constant_perimeter_cross_sections_l193_193359


namespace math_problem_l193_193028

noncomputable def is_solution (x : ℝ) : Prop :=
  1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 12

theorem math_problem :
  (is_solution ((7 + Real.sqrt 153) / 2)) ∧ (is_solution ((7 - Real.sqrt 153) / 2)) := 
by
  sorry

end math_problem_l193_193028


namespace circle_radius_c_value_l193_193538

theorem circle_radius_c_value (x y c : ℝ) (h₁ : x^2 + 8 * x + y^2 + 10 * y + c = 0) (h₂ : (x+4)^2 + (y+5)^2 = 25) :
  c = -16 :=
by sorry

end circle_radius_c_value_l193_193538


namespace point_on_xOz_plane_l193_193284

def point : ℝ × ℝ × ℝ := (1, 0, 4)

theorem point_on_xOz_plane : point.snd = 0 :=
by 
  -- Additional definitions and conditions might be necessary,
  -- but they should come directly from the problem statement:
  -- * Define conditions for being on the xOz plane.
  -- For the purpose of this example, we skip the proof.
  sorry

end point_on_xOz_plane_l193_193284


namespace sum_of_number_and_reverse_l193_193241

def digit_representation (n m : ℕ) (a b : ℕ) :=
  n = 10 * a + b ∧
  m = 10 * b + a ∧
  n - m = 9 * (a * b) + 3

theorem sum_of_number_and_reverse :
  ∃ a b n m : ℕ, digit_representation n m a b ∧ n + m = 22 :=
by
  sorry

end sum_of_number_and_reverse_l193_193241


namespace find_value_of_fraction_l193_193614

open Real

theorem find_value_of_fraction (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 8) : 
  (x + y) / (x - y) = -sqrt (5 / 3) :=
sorry

end find_value_of_fraction_l193_193614


namespace count_success_permutations_l193_193873

theorem count_success_permutations : 
  let total_letters := 7
  let s_count := 3
  let c_count := 2
  (Nat.factorial total_letters) / ((Nat.factorial s_count) * (Nat.factorial c_count)) = 420 := 
by
  let total_letters := 7
  let s_count := 3
  let c_count := 2
  sorry

end count_success_permutations_l193_193873


namespace goose_eggs_count_l193_193049

theorem goose_eggs_count (E : ℕ)
    (hatch_fraction : ℚ := 1/3)
    (first_month_survival : ℚ := 4/5)
    (first_year_survival : ℚ := 2/5)
    (no_migration : ℚ := 3/4)
    (predator_survival : ℚ := 2/3)
    (final_survivors : ℕ := 140) :
    (predator_survival * no_migration * first_year_survival * first_month_survival * hatch_fraction * E : ℚ) = final_survivors → E = 1050 := by
  sorry

end goose_eggs_count_l193_193049


namespace man_swims_speed_l193_193315

theorem man_swims_speed (v_m v_s : ℝ) (h_downstream : 28 = (v_m + v_s) * 2) (h_upstream : 12 = (v_m - v_s) * 2) : v_m = 10 := 
by sorry

end man_swims_speed_l193_193315


namespace walkway_time_against_direction_l193_193100

theorem walkway_time_against_direction (v_p v_w t : ℝ) (h1 : 90 = (v_p + v_w) * 30)
  (h2 : v_p * 48 = 90) 
  (h3 : 90 = (v_p - v_w) * t) :
  t = 120 := by 
  sorry

end walkway_time_against_direction_l193_193100


namespace inequality_solution_l193_193520

theorem inequality_solution {x : ℝ} : (x + 1) / x > 1 ↔ x > 0 := 
sorry

end inequality_solution_l193_193520


namespace manufacturing_department_degrees_l193_193208

def percentage_of_circle (percentage : ℕ) (total_degrees : ℕ) : ℕ :=
  (percentage * total_degrees) / 100

theorem manufacturing_department_degrees :
  percentage_of_circle 30 360 = 108 :=
by
  sorry

end manufacturing_department_degrees_l193_193208


namespace tan_alpha_minus_beta_value_l193_193717

theorem tan_alpha_minus_beta_value (α β : Real) 
  (h1 : Real.sin α = 3 / 5) 
  (h2 : α ∈ Set.Ioo (π / 2) π) 
  (h3 : Real.tan (π - β) = 1 / 2) : 
  Real.tan (α - β) = -2 / 11 :=
by
  sorry

end tan_alpha_minus_beta_value_l193_193717


namespace triangle_side_lengths_l193_193481

variable {c z m : ℕ}

axiom condition1 : 3 * c + z + m = 43
axiom condition2 : c + z + 3 * m = 35
axiom condition3 : 2 * (c + z + m) = 46

theorem triangle_side_lengths : c = 10 ∧ z = 7 ∧ m = 6 := 
by 
  sorry

end triangle_side_lengths_l193_193481


namespace race_speed_ratio_l193_193339

theorem race_speed_ratio (L v_a v_b : ℝ) (h1 : v_a = v_b / 0.84375) :
  v_a / v_b = 32 / 27 :=
by sorry

end race_speed_ratio_l193_193339


namespace partition_nat_set_l193_193040

theorem partition_nat_set :
  ∃ (P : ℕ → ℕ), (∀ (n : ℕ), P n < 100) ∧ (∀ (a b c : ℕ), a + 99 * b = c → (P a = P b ∨ P b = P c ∨ P c = P a)) :=
sorry

end partition_nat_set_l193_193040


namespace binary_representation_88_l193_193579

def binary_representation (n : Nat) : String := sorry

theorem binary_representation_88 : binary_representation 88 = "1011000" := sorry

end binary_representation_88_l193_193579


namespace polynomial_simplification_l193_193431

theorem polynomial_simplification (y : ℤ) : 
  (2 * y - 1) * (4 * y ^ 10 + 2 * y ^ 9 + 4 * y ^ 8 + 2 * y ^ 7) = 8 * y ^ 11 + 6 * y ^ 9 - 2 * y ^ 7 :=
by 
  sorry

end polynomial_simplification_l193_193431


namespace find_y_l193_193357

theorem find_y 
  (h : (5 + 8 + 17) / 3 = (12 + y) / 2) : y = 8 :=
sorry

end find_y_l193_193357


namespace parkingGarageCharges_l193_193061

variable (W : ℕ)

/-- 
  Conditions:
  1. Weekly rental cost is \( W \) dollars.
  2. Monthly rental cost is $24 per month.
  3. A person saves $232 in a year by renting by the month rather than by the week.
  4. There are 52 weeks in a year.
  5. There are 12 months in a year.
-/
def garageChargesPerWeek : Prop :=
  52 * W = 12 * 24 + 232

theorem parkingGarageCharges
  (h : garageChargesPerWeek W) : W = 10 :=
by
  sorry

end parkingGarageCharges_l193_193061


namespace undefined_count_expression_l193_193710

theorem undefined_count_expression : 
  let expr (x : ℝ) := (x^2 - 16) / ((x^2 + 3*x - 10) * (x - 4))
  ∃ u v w : ℝ, (u = 2 ∨ v = -5 ∨ w = 4) ∧
  (u ≠ v ∧ u ≠ w ∧ v ≠ w) :=
by
  sorry

end undefined_count_expression_l193_193710


namespace sufficient_not_necessary_condition_l193_193604

variable (a : ℝ)

theorem sufficient_not_necessary_condition (h1 : a > 2) : (1 / a < 1 / 2) ↔ (a > 2 ∨ a < 0) :=
by
  sorry

end sufficient_not_necessary_condition_l193_193604


namespace relationship_between_a_and_b_l193_193944

open Real

theorem relationship_between_a_and_b
   (a b : ℝ)
   (ha : 0 < a ∧ a < 1)
   (hb : 0 < b ∧ b < 1)
   (hab : (1 - a) * b > 1 / 4) :
   a < b := 
sorry

end relationship_between_a_and_b_l193_193944


namespace stratified_sampling_l193_193875

theorem stratified_sampling (n : ℕ) : 100 + 600 + 500 = 1200 → 500 ≠ 0 → 40 / 500 = n / 1200 → n = 96 :=
by
  intros total_population nonzero_div divisor_eq
  sorry

end stratified_sampling_l193_193875


namespace find_M_l193_193383

theorem find_M (x y z M : ℝ) 
  (h1 : x + y + z = 120) 
  (h2 : x - 10 = M) 
  (h3 : y + 10 = M) 
  (h4 : z / 10 = M) : 
  M = 10 := 
by
  sorry

end find_M_l193_193383


namespace average_minutes_run_per_day_l193_193749

variables (f : ℕ)
def third_graders := 6 * f
def fourth_graders := 2 * f
def fifth_graders := f

def total_minutes_run := 14 * third_graders f + 18 * fourth_graders f + 8 * fifth_graders f
def total_students := third_graders f + fourth_graders f + fifth_graders f

theorem average_minutes_run_per_day : 
  (total_minutes_run f) / (total_students f) = 128 / 9 :=
by
  sorry

end average_minutes_run_per_day_l193_193749


namespace probability_same_color_correct_l193_193874

def number_of_balls : ℕ := 16
def green_balls : ℕ := 8
def red_balls : ℕ := 5
def blue_balls : ℕ := 3

def probability_two_balls_same_color : ℚ :=
  ((green_balls / number_of_balls)^2 + (red_balls / number_of_balls)^2 + (blue_balls / number_of_balls)^2)

theorem probability_same_color_correct :
  probability_two_balls_same_color = 49 / 128 := sorry

end probability_same_color_correct_l193_193874


namespace jason_fishes_on_day_12_l193_193851

def initial_fish_count : ℕ := 10

def fish_on_day (n : ℕ) : ℕ :=
  if n = 0 then initial_fish_count else
  (match n with
  | 1 => 10 * 3
  | 2 => 30 * 3
  | 3 => 90 * 3
  | 4 => 270 * 3 * 3 / 5 -- removes fish according to rule
  | 5 => (270 * 3 * 3 / 5) * 3
  | 6 => ((270 * 3 * 3 / 5) * 3 * 3) * 4 / 7 -- removes fish according to rule
  | 7 => (((270 * 3 * 3 / 5) * 3 * 3) * 4 / 7) * 3
  | 8 => ((((270 * 3 * 3 / 5) * 3 * 3) * 4 / 7) * 3) + 25
  | 9 => (((((270 * 3 * 3 / 5) * 3 * 3) * 4 / 7) * 3) + 25) * 3
  | 10 => ((((((270 * 3 * 3 / 5) * 3 * 3) * 4 / 7) * 3) + 25) * 3) - (((((((270 * 3 * 3 / 5) * 3 * 3) * 4 / 7) * 3) + 25) * 3) / 2)
  | 11 => (((((((270 * 3 * 3 / 5) * 3 * 3) * 4 / 7) * 3) + 25) * 3) - (((((((270 * 3 * 3 / 5) * 3 * 3) * 4 / 7) * 3) + 25) * 3) / 2)) * 3
  | 12 => (((((((270 * 3 * 3 / 5) * 3 * 3) * 4 / 7) * 3) + 25) * 3) - (((((((270 * 3 * 3 / 5) * 3 * 3) * 4 / 7) * 3) + 25) * 3) / 2)) * 3 + (3 * (((((((270 * 3 * 3 / 5) * 3 * 3) * 4 / 7) * 3) + 25) * 3) - (((((((270 * 3 * 3 / 5) * 3 * 3) * 4 / 7) * 3) + 25) * 3) / 2)) * 3) + 5
  | _ => 0
  )
 
theorem jason_fishes_on_day_12 : fish_on_day 12 = 1220045 := 
  by sorry

end jason_fishes_on_day_12_l193_193851


namespace sequence_general_formula_l193_193091

theorem sequence_general_formula (a : ℕ → ℝ) (h : ∀ n, a (n+2) = 2 * a (n+1) / (2 + a (n+1))) :
  (a 1 = 1) → ∀ n, a n = 2 / (n + 1) :=
by
  sorry

end sequence_general_formula_l193_193091


namespace route_comparison_l193_193228

noncomputable def t_X : ℝ := (8 / 40) * 60 -- time in minutes for Route X
noncomputable def t_Y1 : ℝ := (5.5 / 50) * 60 -- time in minutes for the normal speed segment of Route Y
noncomputable def t_Y2 : ℝ := (1 / 25) * 60 -- time in minutes for the construction zone segment of Route Y
noncomputable def t_Y3 : ℝ := (0.5 / 20) * 60 -- time in minutes for the park zone segment of Route Y
noncomputable def t_Y : ℝ := t_Y1 + t_Y2 + t_Y3 -- total time in minutes for Route Y

theorem route_comparison : t_X - t_Y = 1.5 :=
by {
  -- Proof is skipped using sorry
  sorry
}

end route_comparison_l193_193228


namespace abs_e_pi_minus_six_l193_193657

noncomputable def e : ℝ := 2.718
noncomputable def pi : ℝ := 3.14159

theorem abs_e_pi_minus_six : |e + pi - 6| = 0.14041 := by
  sorry

end abs_e_pi_minus_six_l193_193657


namespace find_lamp_cost_l193_193780

def lamp_and_bulb_costs (L B : ℝ) : Prop :=
  B = L - 4 ∧ 2 * L + 6 * B = 32

theorem find_lamp_cost : ∃ L : ℝ, ∃ B : ℝ, lamp_and_bulb_costs L B ∧ L = 7 :=
by
  sorry

end find_lamp_cost_l193_193780


namespace book_arrangements_l193_193050

theorem book_arrangements (total_books : ℕ) (at_least_in_library : ℕ) (at_least_checked_out : ℕ) 
  (h_total : total_books = 10) (h_at_least_in : at_least_in_library = 2) 
  (h_at_least_out : at_least_checked_out = 3) : 
  ∃ arrangements : ℕ, arrangements = 6 :=
by
  sorry

end book_arrangements_l193_193050


namespace no_real_solution_l193_193064

theorem no_real_solution (x : ℝ) : 
  (¬ (x^4 + 3*x^3)/(x^2 + 3*x + 1) + x = -7) :=
sorry

end no_real_solution_l193_193064


namespace arithmetic_progression_sum_l193_193079

theorem arithmetic_progression_sum (a : ℕ → ℝ) (d : ℝ)
  (h1 : ∀ n, a n = a 0 + n * d)
  (h2 : a 0 = 2)
  (h3 : a 1 + a 2 = 13) :
  a 3 + a 4 + a 5 = 42 :=
sorry

end arithmetic_progression_sum_l193_193079


namespace scientific_notation_conversion_l193_193501

theorem scientific_notation_conversion : 
  ∀ (n : ℝ), n = 1.8 * 10^8 → n = 180000000 :=
by
  intros n h
  sorry

end scientific_notation_conversion_l193_193501


namespace class_raised_initial_amount_l193_193510

/-- Miss Grayson's class raised some money for their field trip.
Each student contributed $5 each.
There are 20 students in her class.
The cost of the trip is $7 for each student.
After all the field trip costs were paid, there is $10 left in Miss Grayson's class fund.
Prove that the class initially raised $150 for the field trip. -/
theorem class_raised_initial_amount
  (students : ℕ)
  (contribution_per_student : ℕ)
  (cost_per_student : ℕ)
  (remaining_fund : ℕ)
  (total_students : students = 20)
  (per_student_contribution : contribution_per_student = 5)
  (per_student_cost : cost_per_student = 7)
  (remaining_amount : remaining_fund = 10) :
  (students * contribution_per_student + remaining_fund) = 150 := 
sorry

end class_raised_initial_amount_l193_193510


namespace cd_total_l193_193568

theorem cd_total :
  ∀ (Kristine Dawn Mark Alice : ℕ),
  Dawn = 10 →
  Kristine = Dawn + 7 →
  Mark = 2 * Kristine →
  Alice = (Kristine + Mark) - 5 →
  (Dawn + Kristine + Mark + Alice) = 107 :=
by
  intros Kristine Dawn Mark Alice hDawn hKristine hMark hAlice
  rw [hDawn, hKristine, hMark, hAlice]
  sorry

end cd_total_l193_193568


namespace current_age_l193_193052

theorem current_age (A B S Y : ℕ) 
  (h1: Y = 4) 
  (h2: S = 2 * Y) 
  (h3: B = S + 3) 
  (h4: A + 10 = 2 * (B + 10))
  (h5: A + 10 = 3 * (S + 10))
  (h6: A + 10 = 4 * (Y + 10)) 
  (h7: (A + 10) + (B + 10) + (S + 10) + (Y + 10) = 88) : 
  A = 46 :=
sorry

end current_age_l193_193052


namespace smallest_positive_integer_l193_193104

theorem smallest_positive_integer (N : ℕ) :
  (N % 2 = 1) ∧
  (N % 3 = 2) ∧
  (N % 4 = 3) ∧
  (N % 5 = 4) ∧
  (N % 6 = 5) ∧
  (N % 7 = 6) ∧
  (N % 8 = 7) ∧
  (N % 9 = 8) ∧
  (N % 10 = 9) ↔ 
  N = 2519 := by {
  sorry
}

end smallest_positive_integer_l193_193104


namespace sin_cos_sum_l193_193506

theorem sin_cos_sum (α x y r : ℝ) (h1 : x = 2) (h2 : y = -1) (h3 : r = Real.sqrt 5)
    (h4 : ∀ θ, x = r * Real.cos θ) (h5 : ∀ θ, y = r * Real.sin θ) : 
    Real.sin α + Real.cos α = (- 1 / Real.sqrt 5) + (2 / Real.sqrt 5) :=
by
  sorry

end sin_cos_sum_l193_193506


namespace sum_of_palindromes_l193_193041

/-- Definition of a three-digit palindrome -/
def is_palindrome (n : ℕ) : Prop :=
  n / 100 = n % 10

theorem sum_of_palindromes (a b : ℕ) (h1 : is_palindrome a)
  (h2 : is_palindrome b) (h3 : a * b = 334491) (h4 : 100 ≤ a)
  (h5 : a < 1000) (h6 : 100 ≤ b) (h7 : b < 1000) : a + b = 1324 :=
sorry

end sum_of_palindromes_l193_193041


namespace pattern_generalization_l193_193504

theorem pattern_generalization (n : ℕ) (h : 0 < n) : n * (n + 2) + 1 = (n + 1) ^ 2 :=
by
  -- TODO: The proof will be filled in later
  sorry

end pattern_generalization_l193_193504


namespace xiaoming_climb_stairs_five_steps_l193_193767

def count_ways_to_climb (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 1
  else count_ways_to_climb (n - 1) + count_ways_to_climb (n - 2)

theorem xiaoming_climb_stairs_five_steps :
  count_ways_to_climb 5 = 5 :=
by
  sorry

end xiaoming_climb_stairs_five_steps_l193_193767


namespace min_value_expression_l193_193902

theorem min_value_expression (α β : ℝ) : 
  ∃ a b : ℝ, 
    ((2 * Real.cos α + 5 * Real.sin β - 8) ^ 2 + 
    (2 * Real.sin α + 5 * Real.cos β - 15) ^ 2  = 100) :=
sorry

end min_value_expression_l193_193902


namespace gcd_polynomials_l193_193773

theorem gcd_polynomials (b : ℕ) (hb : 2160 ∣ b) : 
  Nat.gcd (b ^ 2 + 9 * b + 30) (b + 6) = 12 := 
  sorry

end gcd_polynomials_l193_193773


namespace num_valid_m_l193_193191

theorem num_valid_m (m : ℕ) : (∃ n : ℕ, n * (m^2 - 3) = 1722) → ∃ p : ℕ, p = 3 := 
  by
  sorry

end num_valid_m_l193_193191


namespace scrooge_no_equal_coins_l193_193784

theorem scrooge_no_equal_coins (n : ℕ → ℕ)
  (initial_state : n 1 = 1 ∧ n 2 = 0 ∧ n 3 = 0 ∧ n 4 = 0 ∧ n 5 = 0 ∧ n 6 = 0)
  (operation : ∀ x i, 1 ≤ i ∧ i ≤ 6 → (n (i + 1) = n i - x ∧ n ((i % 6) + 2) = n ((i % 6) + 2) + 6 * x) 
                      ∨ (n (i + 1) = n i + 6 * x ∧ n ((i % 6) + 2) = n ((i % 6) + 2) - x)) :
  ¬ ∃ k, n 1 = k ∧ n 2 = k ∧ n 3 = k ∧ n 4 = k ∧ n 5 = k ∧ n 6 = k :=
by {
  sorry
}

end scrooge_no_equal_coins_l193_193784


namespace decodeMINT_l193_193485

def charToDigit (c : Char) : Option Nat :=
  match c with
  | 'G' => some 0
  | 'R' => some 1
  | 'E' => some 2
  | 'A' => some 3
  | 'T' => some 4
  | 'M' => some 5
  | 'I' => some 6
  | 'N' => some 7
  | 'D' => some 8
  | 'S' => some 9
  | _   => none

def decodeWord (word : String) : Option Nat :=
  let digitsOption := word.toList.map charToDigit
  if digitsOption.all Option.isSome then
    let digits := digitsOption.map Option.get!
    some (digits.foldl (λ acc d => 10 * acc + d) 0)
  else
    none

theorem decodeMINT : decodeWord "MINT" = some 5674 := by
  sorry

end decodeMINT_l193_193485


namespace sara_lunch_total_cost_l193_193619

noncomputable def cost_hotdog : ℝ := 5.36
noncomputable def cost_salad : ℝ := 5.10
noncomputable def cost_soda : ℝ := 2.75
noncomputable def cost_fries : ℝ := 3.20
noncomputable def discount_rate : ℝ := 0.15
noncomputable def tax_rate : ℝ := 0.08

noncomputable def total_cost_before_discount_tax : ℝ :=
  cost_hotdog + cost_salad + cost_soda + cost_fries

noncomputable def discount : ℝ :=
  discount_rate * total_cost_before_discount_tax

noncomputable def discounted_total : ℝ :=
  total_cost_before_discount_tax - discount

noncomputable def tax : ℝ := 
  tax_rate * discounted_total

noncomputable def final_total : ℝ :=
  discounted_total + tax

theorem sara_lunch_total_cost : final_total = 15.07 :=
by
  sorry

end sara_lunch_total_cost_l193_193619


namespace gcd_40_56_l193_193836

theorem gcd_40_56 : Int.gcd 40 56 = 8 :=
by
  sorry

end gcd_40_56_l193_193836


namespace cylinder_height_l193_193038

theorem cylinder_height {D r : ℝ} (hD : D = 10) (hr : r = 3) : 
  ∃ h : ℝ, h = 8 :=
by
  -- hD -> Diameter of hemisphere = 10
  -- hr -> Radius of cylinder's base = 3
  sorry

end cylinder_height_l193_193038


namespace geometric_series_sum_l193_193083

  theorem geometric_series_sum :
    let a := (1 / 4 : ℚ)
    let r := (1 / 4 : ℚ)
    let n := 4
    let S_n := a * (1 - r^n) / (1 - r)
    S_n = 255 / 768 := by
  sorry
  
end geometric_series_sum_l193_193083


namespace initial_number_of_people_l193_193413

theorem initial_number_of_people (P : ℕ) : P * 10 = (P + 1) * 5 → P = 1 :=
by sorry

end initial_number_of_people_l193_193413


namespace distance_is_absolute_value_l193_193219

noncomputable def distance_to_origin (x : ℝ) : ℝ := |x|

theorem distance_is_absolute_value (x : ℝ) : distance_to_origin x = |x| :=
by
  sorry

end distance_is_absolute_value_l193_193219


namespace bill_score_l193_193673

variable {J B S : ℕ}

theorem bill_score (h1 : B = J + 20) (h2 : B = S / 2) (h3 : J + B + S = 160) : B = 45 :=
sorry

end bill_score_l193_193673


namespace increasing_function_a_range_l193_193904

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 + 4 * a * x else (2 * a + 3) * x - 4 * a + 5

theorem increasing_function_a_range (a : ℝ) :
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) ↔ (1 / 2 ≤ a ∧ a ≤ 3 / 2) :=
by
  sorry

end increasing_function_a_range_l193_193904


namespace find_difference_of_a_and_b_l193_193353

-- Define the conditions
variables (a b : ℝ)
axiom cond1 : 4 * a + 3 * b = 8
axiom cond2 : 3 * a + 4 * b = 6

-- Statement for the proof
theorem find_difference_of_a_and_b : a - b = 2 :=
by
  sorry

end find_difference_of_a_and_b_l193_193353


namespace a_minus_b_is_15_l193_193252

variables (a b c : ℝ)

-- Conditions from the problem statement
axiom cond1 : a = 1/3 * (b + c)
axiom cond2 : b = 2/7 * (a + c)
axiom cond3 : a + b + c = 540

-- The theorem we need to prove
theorem a_minus_b_is_15 : a - b = 15 :=
by
  sorry

end a_minus_b_is_15_l193_193252


namespace sum_of_ages_l193_193543

theorem sum_of_ages {l t : ℕ} (h1 : t > l) (h2 : t * t * l = 72) : t + t + l = 14 :=
sorry

end sum_of_ages_l193_193543


namespace election_winner_votes_l193_193387

theorem election_winner_votes (V : ℝ) : (0.62 * V = 806) → (0.62 * V) - (0.38 * V) = 312 → 0.62 * V = 806 :=
by
  intro hWin hDiff
  exact hWin

end election_winner_votes_l193_193387


namespace magnitude_of_T_l193_193566

theorem magnitude_of_T : 
  let i := Complex.I
  let T := 3 * ((1 + i) ^ 15 - (1 - i) ^ 15)
  Complex.abs T = 768 := by
  sorry

end magnitude_of_T_l193_193566


namespace quadratic_value_at_two_l193_193474

open Real

-- Define the conditions
variables (a b : ℝ)

def f (x : ℝ) : ℝ := x^2 + a * x + b

-- State the proof problem
theorem quadratic_value_at_two (h₀ : f a b (f a b 0) = 0) (h₁ : f a b (f a b 1) = 0) (h₂ : f a b 0 ≠ f a b 1) :
  f a b 2 = 2 := 
sorry

end quadratic_value_at_two_l193_193474


namespace cheese_cut_indefinite_l193_193617

theorem cheese_cut_indefinite (w : ℝ) (R : ℝ) (h : ℝ) :
  R = 0.5 →
  (∀ a b c d : ℝ, a > b → b > c → c > d →
    (∃ h, h < min (a - d) (d - c) ∧
     (d + h < a ∧ d - h > c))) →
  ∃ l1 l2 : ℕ → ℝ, (∀ n, l1 (n + 1) > l2 (n) ∧ l1 n > R * l2 (n)) :=
sorry

end cheese_cut_indefinite_l193_193617


namespace product_of_values_of_t_squared_eq_49_l193_193328

theorem product_of_values_of_t_squared_eq_49 :
  (∀ t, t^2 = 49 → t = 7 ∨ t = -7) →
  (7 * -7 = -49) :=
by
  intros h
  sorry

end product_of_values_of_t_squared_eq_49_l193_193328


namespace university_A_pass_one_subject_university_B_pass_one_subject_when_m_3_5_preferred_range_of_m_l193_193151

-- Part 1
def probability_A_exactly_one_subject : ℚ :=
  3 * (1/2) * (1/2)^2

def probability_B_exactly_one_subject (m : ℚ) : ℚ :=
  (1/6) * (2/5)^2 + (5/6) * (3/5) * (2/5) * 2

theorem university_A_pass_one_subject : probability_A_exactly_one_subject = 3/8 :=
sorry

theorem university_B_pass_one_subject_when_m_3_5 : probability_B_exactly_one_subject (3/5) = 32/75 :=
sorry

-- Part 2
def expected_A : ℚ :=
  3 * (1/2)

def expected_B (m : ℚ) : ℚ :=
  ((17 - 7 * m) / 30) + (2 * (3 + 14 * m) / 30) + (3 * m / 10)

theorem preferred_range_of_m : 0 < m ∧ m < 11/15 → expected_A > expected_B m :=
sorry

end university_A_pass_one_subject_university_B_pass_one_subject_when_m_3_5_preferred_range_of_m_l193_193151


namespace gardener_cabbages_increased_by_197_l193_193567

theorem gardener_cabbages_increased_by_197 (x : ℕ) (last_year_cabbages : ℕ := x^2) (increase : ℕ := 197) :
  (x + 1)^2 = x^2 + increase → (x + 1)^2 = 9801 :=
by
  intros h
  sorry

end gardener_cabbages_increased_by_197_l193_193567


namespace max_log_sum_l193_193831

noncomputable def log (x : ℝ) : ℝ := Real.log x

theorem max_log_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 4) : 
  ∃ L, (∀ x y, x > 0 → y > 0 → x + y = 4 → log x + log y ≤ L) ∧ L = log 4 :=
by
  sorry

end max_log_sum_l193_193831


namespace more_candidates_selected_l193_193605

theorem more_candidates_selected (n : ℕ) (pA pB : ℝ) 
  (hA : pA = 0.06) (hB : pB = 0.07) (hN : n = 8200) :
  (pB * n - pA * n) = 82 :=
by
  sorry

end more_candidates_selected_l193_193605


namespace part1_part2_l193_193054

-- Part (1)
theorem part1 (x y : ℚ) 
  (h1 : 2022 * x + 2020 * y = 2021)
  (h2 : 2023 * x + 2021 * y = 2022) :
  x = 1/2 ∧ y = 1/2 :=
by
  -- Placeholder for the proof
  sorry

-- Part (2)
theorem part2 (x y a b : ℚ)
  (ha : a ≠ b) 
  (h1 : (a + 1) * x + (a - 1) * y = a)
  (h2 : (b + 1) * x + (b - 1) * y = b) :
  x = 1/2 ∧ y = 1/2 :=
by
  -- Placeholder for the proof
  sorry

end part1_part2_l193_193054


namespace seeds_per_plant_l193_193132

theorem seeds_per_plant :
  let trees := 2
  let plants_per_tree := 20
  let total_plants := trees * plants_per_tree
  let planted_trees := 24
  let planting_fraction := 0.60
  exists S : ℝ, planting_fraction * (total_plants * S) = planted_trees ∧ S = 1 :=
by
  sorry

end seeds_per_plant_l193_193132


namespace find_number_l193_193734

theorem find_number :
  ∃ x : ℤ, 27 * (x + 143) = 9693 ∧ x = 216 :=
by 
  existsi 216
  sorry

end find_number_l193_193734


namespace attended_college_percentage_l193_193984

variable (total_boys : ℕ) (total_girls : ℕ) (percent_not_attend_boys : ℕ) (percent_not_attend_girls : ℕ)

def total_boys_attended_college (total_boys percent_not_attend_boys : ℕ) : ℕ :=
  total_boys - percent_not_attend_boys * total_boys / 100

def total_girls_attended_college (total_girls percent_not_attend_girls : ℕ) : ℕ :=
  total_girls - percent_not_attend_girls * total_girls / 100

noncomputable def total_student_attended_college (total_boys total_girls percent_not_attend_boys percent_not_attend_girls : ℕ) : ℕ :=
  total_boys_attended_college total_boys percent_not_attend_boys +
  total_girls_attended_college total_girls percent_not_attend_girls

noncomputable def percent_class_attended_college (total_boys total_girls percent_not_attend_boys percent_not_attend_girls : ℕ) : ℕ :=
  total_student_attended_college total_boys total_girls percent_not_attend_boys percent_not_attend_girls * 100 /
  (total_boys + total_girls)

theorem attended_college_percentage :
  total_boys = 300 → total_girls = 240 → percent_not_attend_boys = 30 → percent_not_attend_girls = 30 →
  percent_class_attended_college total_boys total_girls percent_not_attend_boys percent_not_attend_girls = 70 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end attended_college_percentage_l193_193984


namespace inequality1_solution_inequality2_solution_l193_193859

open Real

-- First problem: proving the solution set for x + |2x + 3| >= 2
theorem inequality1_solution (x : ℝ) : x + abs (2 * x + 3) >= 2 ↔ (x <= -5 ∨ x >= -1/3) := 
sorry

-- Second problem: proving the solution set for |x - 1| - |x - 5| < 2
theorem inequality2_solution (x : ℝ) : abs (x - 1) - abs (x - 5) < 2 ↔ x < 4 :=
sorry

end inequality1_solution_inequality2_solution_l193_193859


namespace problem_statement_l193_193508

theorem problem_statement :
  102^3 + 3 * 102^2 + 3 * 102 + 1 = 1092727 :=
  by sorry

end problem_statement_l193_193508


namespace count_silver_coins_l193_193136

theorem count_silver_coins 
  (gold_value : ℕ)
  (silver_value : ℕ)
  (num_gold_coins : ℕ)
  (cash : ℕ)
  (total_money : ℕ) :
  gold_value = 50 →
  silver_value = 25 →
  num_gold_coins = 3 →
  cash = 30 →
  total_money = 305 →
  ∃ S : ℕ, num_gold_coins * gold_value + S * silver_value + cash = total_money ∧ S = 5 := 
by
  sorry

end count_silver_coins_l193_193136


namespace find_n_l193_193011

variable (P : ℕ → ℝ) (n : ℕ)

def polynomialDegree (P : ℕ → ℝ) (deg : ℕ) : Prop :=
  ∀ k, k > deg → P k = 0

def zeroValues (P : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ i, i ∈ (List.range (2 * n + 1)).map (λ k => 2 * k) → P i = 0

def twoValues (P : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ i, i ∈ (List.range (2 * n)).map (λ k => 2 * k + 1) → P i = 2

def specialValue (P : ℕ → ℝ) (n : ℕ) : Prop :=
  P (2 * n + 1) = -30

theorem find_n :
  (∃ n, polynomialDegree P (2 * n) ∧ zeroValues P n ∧ twoValues P n ∧ specialValue P n) →
  n = 2 :=
by
  sorry

end find_n_l193_193011


namespace b_investment_correct_l193_193978

-- Constants for shares and investments
def a_investment : ℕ := 11000
def a_share : ℕ := 2431
def b_share : ℕ := 3315
def c_investment : ℕ := 23000

-- Goal: Prove b's investment given the conditions
theorem b_investment_correct (b_investment : ℕ) (h : 2431 * b_investment = 11000 * 3315) :
  b_investment = 15000 := by
  sorry

end b_investment_correct_l193_193978


namespace combined_basketballs_l193_193648

-- Conditions as definitions
def spursPlayers := 22
def rocketsPlayers := 18
def basketballsPerPlayer := 11

-- Math Proof Problem statement
theorem combined_basketballs : 
  (spursPlayers * basketballsPerPlayer) + (rocketsPlayers * basketballsPerPlayer) = 440 :=
by
  sorry

end combined_basketballs_l193_193648


namespace one_fourth_more_equals_thirty_percent_less_l193_193863

theorem one_fourth_more_equals_thirty_percent_less :
  ∃ n : ℝ, 80 - 0.30 * 80 = (5 / 4) * n ∧ n = 44.8 :=
by
  sorry

end one_fourth_more_equals_thirty_percent_less_l193_193863


namespace rhombus_longer_diagonal_l193_193502

theorem rhombus_longer_diagonal (d1 d2 : ℝ) (h_d1 : d1 = 11) (h_area : (d1 * d2) / 2 = 110) : d2 = 20 :=
by
  sorry

end rhombus_longer_diagonal_l193_193502


namespace sqrt_9_eq_pos_neg_3_l193_193515

theorem sqrt_9_eq_pos_neg_3 : ∀ x : ℝ, x^2 = 9 ↔ x = 3 ∨ x = -3 :=
by
  sorry

end sqrt_9_eq_pos_neg_3_l193_193515


namespace range_of_BD_l193_193539

-- Define the types of points and triangle
variables {α : Type*} [MetricSpace α]

-- Hypothesis: AD is the median of triangle ABC
-- Definition of lengths AB, AC, and that BD = CD.
def isMedianOnBC (A B C D : α) : Prop :=
  dist A B = 5 ∧ dist A C = 7 ∧ dist B D = dist C D

-- The theorem to be proven
theorem range_of_BD {A B C D : α} (h : isMedianOnBC A B C D) : 
  1 < dist B D ∧ dist B D < 6 :=
by
  sorry

end range_of_BD_l193_193539


namespace dispersion_is_variance_l193_193847

def Mean := "Mean"
def Variance := "Variance"
def Median := "Median"
def Mode := "Mode"

def dispersion_measure := Variance

theorem dispersion_is_variance (A B C D : String) (hA : A = Mean) (hB : B = Variance) (hC : C = Median) (hD : D = Mode) : 
  dispersion_measure = B :=
by
  rw [hB]
  exact sorry

end dispersion_is_variance_l193_193847


namespace minimize_z_l193_193450

theorem minimize_z (x y : ℝ) (h1 : 2 * x - y ≥ 0) (h2 : y ≥ x) (h3 : y ≥ -x + 2) :
  ∃ (x y : ℝ), (z = 2 * x + y) ∧ z = 8 / 3 :=
by
  sorry

end minimize_z_l193_193450


namespace selling_price_of_mixture_l193_193369

noncomputable def selling_price_per_pound (weight1 weight2 price1 price2 total_weight : ℝ) : ℝ :=
  (weight1 * price1 + weight2 * price2) / total_weight

theorem selling_price_of_mixture :
  selling_price_per_pound 20 10 2.95 3.10 30 = 3.00 :=
by
  -- Skipping the proof part
  sorry

end selling_price_of_mixture_l193_193369


namespace min_value_of_c_l193_193404

variable {a b c : ℝ}
variables (a_pos : a > 0) (b_pos : b > 0)
variable (hyperbola : ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1)
variable (semi_focal_dist : c = Real.sqrt (a^2 + b^2))
variable (distance_condition : ∀ (d : ℝ), d = a * b / c = 1 / 3 * c + 1)

theorem min_value_of_c : c = 6 := 
sorry

end min_value_of_c_l193_193404


namespace daily_production_l193_193310

-- Definitions based on conditions
def weekly_production : ℕ := 3400
def working_days_in_week : ℕ := 5

-- Statement to prove the number of toys produced each day
theorem daily_production : (weekly_production / working_days_in_week) = 680 :=
by
  sorry

end daily_production_l193_193310


namespace fraction_is_irreducible_l193_193570

theorem fraction_is_irreducible :
  (1 * 2 * 4 + 2 * 4 * 8 + 3 * 6 * 12 + 4 * 8 * 16 : ℚ) / 
   (1 * 3 * 9 + 2 * 6 * 18 + 3 * 9 * 27 + 4 * 12 * 36) = 8 / 27 :=
by 
  sorry

end fraction_is_irreducible_l193_193570


namespace range_of_a_l193_193807

noncomputable def f (a x : ℝ) := a * x^2 - (2 - a) * x + 1
noncomputable def g (x : ℝ) := x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a x > 0 ∨ g x > 0) ↔ (0 ≤ a ∧ a < 4 + 2 * Real.sqrt 3) :=
by
  sorry

end range_of_a_l193_193807


namespace smallest_positive_number_div_conditions_is_perfect_square_l193_193942

theorem smallest_positive_number_div_conditions_is_perfect_square :
  ∃ n : ℕ,
    (n % 11 = 10) ∧
    (n % 10 = 9) ∧
    (n % 9 = 8) ∧
    (n % 8 = 7) ∧
    (n % 7 = 6) ∧
    (n % 6 = 5) ∧
    (n % 5 = 4) ∧
    (n % 4 = 3) ∧
    (n % 3 = 2) ∧
    (n % 2 = 1) ∧
    (∃ k : ℕ, n = k * k) ∧
    n = 2782559 :=
by
  sorry

end smallest_positive_number_div_conditions_is_perfect_square_l193_193942


namespace weight_of_each_bag_of_flour_l193_193695

theorem weight_of_each_bag_of_flour
  (flour_weight_needed : ℕ)
  (cost_per_bag : ℕ)
  (salt_weight_needed : ℕ)
  (salt_cost_per_pound : ℚ)
  (promotion_cost : ℕ)
  (ticket_price : ℕ)
  (tickets_sold : ℕ)
  (total_made : ℕ)
  (profit : ℕ)
  (total_flour_cost : ℕ)
  (num_bags : ℕ)
  (weight_per_bag : ℕ)
  (calc_salt_cost : ℚ := salt_weight_needed * salt_cost_per_pound)
  (calc_total_earnings : ℕ := tickets_sold * ticket_price)
  (calc_total_cost : ℚ := calc_total_earnings - profit)
  (calc_flour_cost : ℚ := calc_total_cost - calc_salt_cost - promotion_cost)
  (calc_num_bags : ℚ := calc_flour_cost / cost_per_bag)
  (calc_weight_per_bag : ℚ := flour_weight_needed / calc_num_bags) :
  flour_weight_needed = 500 ∧
  cost_per_bag = 20 ∧
  salt_weight_needed = 10 ∧
  salt_cost_per_pound = 0.2 ∧
  promotion_cost = 1000 ∧
  ticket_price = 20 ∧
  tickets_sold = 500 ∧
  total_made = 8798 ∧
  profit = 10000 - total_made ∧
  calc_salt_cost = 2 ∧
  calc_total_earnings = 10000 ∧
  calc_total_cost = 1202 ∧
  calc_flour_cost = 200 ∧
  calc_num_bags = 10 ∧
  calc_weight_per_bag = 50 :=
by {
  sorry
}

end weight_of_each_bag_of_flour_l193_193695


namespace circle_center_radius_proof_l193_193459

noncomputable def circle_center_radius (x y : ℝ) :=
  x^2 + y^2 - 4*x + 2*y + 2 = 0

theorem circle_center_radius_proof :
  ∀ x y : ℝ, circle_center_radius x y ↔ ((x - 2)^2 + (y + 1)^2 = 3) :=
by
  sorry

end circle_center_radius_proof_l193_193459


namespace markers_multiple_of_4_l193_193606

-- Definitions corresponding to conditions
def Lisa_has_12_coloring_books := 12
def Lisa_has_36_crayons := 36
def greatest_number_baskets := 4

-- Theorem statement
theorem markers_multiple_of_4
    (h1 : Lisa_has_12_coloring_books = 12)
    (h2 : Lisa_has_36_crayons = 36)
    (h3 : greatest_number_baskets = 4) :
    ∃ (M : ℕ), M % 4 = 0 :=
by
  sorry

end markers_multiple_of_4_l193_193606


namespace decompose_max_product_l193_193180

theorem decompose_max_product (a : ℝ) (h_pos : a > 0) :
  ∃ x y : ℝ, x + y = a ∧ x * y ≤ (a / 2) * (a / 2) :=
by
  sorry

end decompose_max_product_l193_193180


namespace mixed_oil_rate_per_litre_l193_193145

variables (volume1 : ℝ) (price1 : ℝ) (volume2 : ℝ) (price2 : ℝ)

def total_cost (v p : ℝ) : ℝ := v * p
def total_volume (v1 v2 : ℝ) : ℝ := v1 + v2

theorem mixed_oil_rate_per_litre (h1 : volume1 = 10) (h2 : price1 = 55) (h3 : volume2 = 5) (h4 : price2 = 66) :
  (total_cost volume1 price1 + total_cost volume2 price2) / total_volume volume1 volume2 = 58.67 := 
by
  sorry

end mixed_oil_rate_per_litre_l193_193145


namespace xyz_value_l193_193420

theorem xyz_value (x y z : ℚ)
  (h1 : x + y + z = 1)
  (h2 : x + y - z = 2)
  (h3 : x - y - z = 3) :
  x * y * z = 1/2 :=
by
  sorry

end xyz_value_l193_193420


namespace correct_proposition_l193_193624

-- Define the propositions as Lean 4 statements.
def PropA (a : ℝ) : Prop := a^4 + a^2 = a^6
def PropB (a : ℝ) : Prop := (-2 * a^2)^3 = -6 * a^8
def PropC (a : ℝ) : Prop := 6 * a - a = 5
def PropD (a : ℝ) : Prop := a^2 * a^3 = a^5

-- The main theorem statement that only PropD is true.
theorem correct_proposition (a : ℝ) : ¬ PropA a ∧ ¬ PropB a ∧ ¬ PropC a ∧ PropD a :=
by
  sorry

end correct_proposition_l193_193624


namespace find_r_l193_193809

theorem find_r (r : ℝ) :
  (∀ (x y : ℝ), x^2 + y^2 = 4) → 
  (∀ (x y : ℝ), (x + 4)^2 + (y - 3)^2 = r^2) →
  (∀ x1 y1 x2 y2: ℝ, 
    (x2 - x1)^2 + (y2 - y1)^2 = 25) →
  (2 + |r| = 5) →
  (r = 3 ∨ r = -3) :=
by
  sorry

end find_r_l193_193809


namespace perpendicular_dot_product_zero_l193_193766

variables (a : ℝ)
def m := (a, 2)
def n := (1, 1 - a)

theorem perpendicular_dot_product_zero : (m a).1 * (n a).1 + (m a).2 * (n a).2 = 0 → a = 2 :=
by sorry

end perpendicular_dot_product_zero_l193_193766


namespace cos_beta_acos_l193_193203

theorem cos_beta_acos {α β : ℝ} (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h_cos_α : Real.cos α = 1 / 7) (h_cos_sum : Real.cos (α + β) = -11 / 14) :
  Real.cos β = 1 / 2 := by
  sorry

end cos_beta_acos_l193_193203


namespace cubic_sum_l193_193173

theorem cubic_sum (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : 
  x^3 + y^3 = 1008 := 
by 
  sorry

end cubic_sum_l193_193173


namespace find_angle_D_l193_193338

noncomputable def measure.angle_A := 80
noncomputable def measure.angle_B := 30
noncomputable def measure.angle_C := 20

def sum_angles_pentagon (A B C : ℕ) := 540 - (A + B + C)

theorem find_angle_D
  (A B C E F : ℕ)
  (hA : A = measure.angle_A)
  (hB : B = measure.angle_B)
  (hC : C = measure.angle_C)
  (h_sum_pentagon : A + B + C + D + E + F = 540)
  (h_triangle : D + E + F = 180) :
  D = 130 :=
by
  sorry

end find_angle_D_l193_193338


namespace max_triangle_side_length_l193_193850

theorem max_triangle_side_length:
  ∃ (a b c : ℕ), 
    a < b ∧ b < c ∧ a + b + c = 30 ∧
    a + b > c ∧ a + c > b ∧ b + c > a ∧ c = 14 :=
  sorry

end max_triangle_side_length_l193_193850


namespace leopards_points_l193_193403

variables (x y : ℕ)

theorem leopards_points (h₁ : x + y = 50) (h₂ : x - y = 28) : y = 11 := by
  sorry

end leopards_points_l193_193403


namespace megatek_manufacturing_percentage_l193_193645

theorem megatek_manufacturing_percentage (total_degrees manufacturing_degrees : ℝ)
    (h_proportional : total_degrees = 360)
    (h_manufacturing_degrees : manufacturing_degrees = 180) :
    (manufacturing_degrees / total_degrees) * 100 = 50 := by
  -- The proof will go here.
  sorry

end megatek_manufacturing_percentage_l193_193645


namespace loom_weaving_rate_l193_193436

theorem loom_weaving_rate (total_cloth : ℝ) (total_time : ℝ) 
    (h1 : total_cloth = 25) (h2 : total_time = 195.3125) : 
    total_cloth / total_time = 0.128 :=
sorry

end loom_weaving_rate_l193_193436


namespace annie_total_blocks_l193_193131

-- Definitions of the blocks traveled in each leg of Annie's journey
def walk_to_bus_stop := 5
def ride_bus_to_train_station := 7
def train_to_friends_house := 10
def walk_to_coffee_shop := 4
def walk_back_to_friends_house := walk_to_coffee_shop

-- The total blocks considering the round trip and additional walk to/from coffee shop
def total_blocks_traveled :=
  2 * (walk_to_bus_stop + ride_bus_to_train_station + train_to_friends_house) +
  walk_to_coffee_shop + walk_back_to_friends_house

-- Statement to prove
theorem annie_total_blocks : total_blocks_traveled = 52 :=
by
  sorry

end annie_total_blocks_l193_193131


namespace votes_for_candidate_a_l193_193602

theorem votes_for_candidate_a :
  let total_votes : ℝ := 560000
  let percentage_invalid : ℝ := 0.15
  let percentage_candidate_a : ℝ := 0.85
  let valid_votes := (1 - percentage_invalid) * total_votes
  let votes_candidate_a := percentage_candidate_a * valid_votes
  votes_candidate_a = 404600 :=
by
  sorry

end votes_for_candidate_a_l193_193602


namespace notebook_cost_l193_193580

theorem notebook_cost (s n c : ℕ) (h1 : s ≥ 19) (h2 : n > 2) (h3 : c > n) (h4 : s * c * n = 3969) : c = 27 :=
sorry

end notebook_cost_l193_193580


namespace number_multiplies_a_l193_193698

theorem number_multiplies_a (a b x : ℝ) (h₀ : x * a = 8 * b) (h₁ : a ≠ 0 ∧ b ≠ 0) (h₂ : (a / 8) / (b / 7) = 1) : x = 7 :=
by
  sorry

end number_multiplies_a_l193_193698


namespace triangle_inequality_l193_193868

theorem triangle_inequality 
  (a b c R : ℝ) 
  (h1 : a + b > c) 
  (h2 : a + c > b) 
  (h3 : b + c > a) 
  (hR : R = (a * b * c) / (4 * Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c)))) : 
  a^2 + b^2 + c^2 ≤ 9 * R^2 :=
by 
  sorry

end triangle_inequality_l193_193868


namespace problem_statement_l193_193965

noncomputable def C_points_count (A B : (ℝ × ℝ)) : ℕ :=
  if A = (0, 0) ∧ B = (12, 0) then 4 else 0

theorem problem_statement :
  let A := (0, 0)
  let B := (12, 0)
  C_points_count A B = 4 :=
by
  sorry

end problem_statement_l193_193965


namespace nonnegative_integer_solutions_l193_193764

theorem nonnegative_integer_solutions (x : ℕ) (h : 1 + x ≥ 2 * x - 1) : x = 0 ∨ x = 1 ∨ x = 2 :=
by
  sorry

end nonnegative_integer_solutions_l193_193764


namespace smallest_positive_debt_resolved_l193_193771

theorem smallest_positive_debt_resolved : ∃ (D : ℤ), D > 0 ∧ (∃ (p g : ℤ), D = 400 * p + 250 * g) ∧ D = 50 :=
by
  sorry

end smallest_positive_debt_resolved_l193_193771


namespace each_person_pays_50_97_l193_193217

noncomputable def total_bill (original_bill : ℝ) (tip_percentage : ℝ) : ℝ :=
  original_bill + original_bill * tip_percentage

noncomputable def amount_per_person (total_bill : ℝ) (num_people : ℕ) : ℝ :=
  total_bill / num_people

theorem each_person_pays_50_97 :
  let original_bill := 139.00
  let number_of_people := 3
  let tip_percentage := 0.10
  let expected_amount := 50.97
  abs (amount_per_person (total_bill original_bill tip_percentage) number_of_people - expected_amount) < 0.01
:= sorry

end each_person_pays_50_97_l193_193217


namespace height_to_top_floor_l193_193936

def total_height : ℕ := 1454
def antenna_spire_height : ℕ := 204

theorem height_to_top_floor : (total_height - antenna_spire_height) = 1250 := by
  sorry

end height_to_top_floor_l193_193936


namespace ratio_problem_l193_193494

variable (a b c d : ℝ)

theorem ratio_problem (h1 : a / b = 3) (h2 : b / c = 1 / 4) (h3 : c / d = 5) : d / a = 4 / 15 := 
sorry

end ratio_problem_l193_193494


namespace cos6_plus_sin6_equal_19_div_64_l193_193756

noncomputable def cos6_plus_sin6 (θ : ℝ) : ℝ :=
  (Real.cos θ) ^ 6 + (Real.sin θ) ^ 6

theorem cos6_plus_sin6_equal_19_div_64 (θ : ℝ) (h : Real.cos (2 * θ) = 1 / 4) :
  cos6_plus_sin6 θ = 19 / 64 := by
  sorry

end cos6_plus_sin6_equal_19_div_64_l193_193756


namespace stephanie_speed_l193_193785

noncomputable def distance : ℝ := 15
noncomputable def time : ℝ := 3

theorem stephanie_speed :
  distance / time = 5 := 
sorry

end stephanie_speed_l193_193785


namespace greatest_integer_b_l193_193858

theorem greatest_integer_b (b : ℤ) :
  (∀ x : ℝ, x^2 + (b : ℝ) * x + 5 ≠ -25) → b ≤ 10 :=
by
  intro
  sorry

end greatest_integer_b_l193_193858


namespace sum_of_arithmetic_series_l193_193551

-- Define the conditions
def first_term := 1
def last_term := 12
def number_of_terms := 12

-- Prop statement that the sum of the arithmetic series equals 78
theorem sum_of_arithmetic_series : (number_of_terms / 2) * (first_term + last_term) = 78 := 
by
  sorry

end sum_of_arithmetic_series_l193_193551


namespace golden_fish_caught_times_l193_193848

open Nat

theorem golden_fish_caught_times :
  ∃ (x y z : ℕ), (4 * x + 2 * z = 2000) ∧ (2 * y + z = 800) ∧ (x + y + z = 900) :=
sorry

end golden_fish_caught_times_l193_193848


namespace statement_A_l193_193103

theorem statement_A (x : ℝ) (h : x < -1) : x^2 > x :=
sorry

end statement_A_l193_193103


namespace neg_p_equiv_l193_193783

-- The proposition p
def p : Prop := ∀ x : ℝ, x^2 - 1 < 0

-- Equivalent Lean theorem statement
theorem neg_p_equiv : ¬ p ↔ ∃ x₀ : ℝ, x₀^2 - 1 ≥ 0 :=
by
  sorry

end neg_p_equiv_l193_193783


namespace larger_tablet_diagonal_length_l193_193015

theorem larger_tablet_diagonal_length :
  ∀ (d : ℝ), (d^2 / 2 = 25 / 2 + 5.5) → d = 6 :=
by
  intro d
  sorry

end larger_tablet_diagonal_length_l193_193015


namespace solve_for_x_l193_193161

theorem solve_for_x (x : ℝ) : 3 * x^2 + 15 = 3 * (2 * x + 20) → (x = 5 ∨ x = -3) :=
sorry

end solve_for_x_l193_193161


namespace sum_of_squares_l193_193029

theorem sum_of_squares (a b : ℝ) (h1 : (a + b)^2 = 11) (h2 : (a - b)^2 = 5) : a^2 + b^2 = 8 := 
sorry

end sum_of_squares_l193_193029


namespace f_diff_l193_193149

-- Define the function f(n)
def f (n : ℕ) : ℚ :=
  (Finset.range (3 * n + 1 + 1)).sum (λ i => 1 / (n + i + 1))

-- The theorem stating the main problem
theorem f_diff (k : ℕ) : 
  f (k + 1) - f k = (1 / (3 * k + 2)) + (1 / (3 * k + 3)) + (1 / (3 * k + 4)) - (1 / (k + 1)) :=
by
  sorry

end f_diff_l193_193149


namespace sum_series_l193_193740

theorem sum_series :
  3 * (List.sum (List.map (λ n => n - 1) (List.range' 2 14))) = 273 :=
by
  sorry

end sum_series_l193_193740


namespace price_per_yellow_stamp_l193_193113

theorem price_per_yellow_stamp 
    (num_red_stamps : ℕ) (price_red_stamp : ℝ) 
    (num_blue_stamps : ℕ) (price_blue_stamp : ℝ)
    (num_yellow_stamps : ℕ) (goal : ℝ)
    (sold_red_stamps : ℕ) (sold_red_price : ℝ)
    (sold_blue_stamps : ℕ) (sold_blue_price : ℝ):

    num_red_stamps = 20 ∧ 
    num_blue_stamps = 80 ∧ 
    num_yellow_stamps = 7 ∧ 
    sold_red_stamps = 20 ∧ 
    sold_red_price = 1.1 ∧ 
    sold_blue_stamps = 80 ∧ 
    sold_blue_price = 0.8 ∧ 
    goal = 100 → 
    (goal - (sold_red_stamps * sold_red_price + sold_blue_stamps * sold_blue_price)) / num_yellow_stamps = 2 := 
  by
  sorry

end price_per_yellow_stamp_l193_193113


namespace inequality_l193_193453

theorem inequality (a b : ℝ) (n : ℕ) (h1 : 0 < a) (h2 : 0 < b) :
  (1 + a / b) ^ n + (1 + b / a) ^ n ≥ 2 ^ (n + 1) :=
by
  sorry

end inequality_l193_193453


namespace inclination_angle_range_l193_193159

theorem inclination_angle_range :
  let Γ := fun x y : ℝ => x * abs x + y * abs y = 1
  let line (m : ℝ) := fun x y : ℝ => y = m * (x - 1)
  ∀ m : ℝ,
  (∃ p1 p2 p3 : ℝ × ℝ, 
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ 
    line m p1.1 p1.2 ∧ Γ p1.1 p1.2 ∧ 
    line m p2.1 p2.2 ∧ Γ p2.1 p2.2 ∧ 
    line m p3.1 p3.2 ∧ Γ p3.1 p3.2) →
  (∃ θ : ℝ, θ ∈ (Set.Ioo (Real.pi / 2) (3 * Real.pi / 4) ∪ 
                  Set.Ioo (3 * Real.pi / 4) (Real.pi - Real.arctan (Real.sqrt 2 / 2)))) :=
sorry

end inclination_angle_range_l193_193159


namespace sequence_infinite_pos_neg_l193_193465

theorem sequence_infinite_pos_neg (a : ℕ → ℝ)
  (h : ∀ k : ℕ, a (k + 1) = (k * a k + 1) / (k - a k)) :
  ∃ (P N : ℕ → Prop), (∀ n, P n ↔ 0 < a n) ∧ (∀ n, N n ↔ a n < 0) ∧ 
  (∀ m, ∃ n, n > m ∧ P n) ∧ (∀ m, ∃ n, n > m ∧ N n) := 
sorry

end sequence_infinite_pos_neg_l193_193465


namespace number_of_days_worked_l193_193731

-- Define the conditions
def hours_per_day := 8
def total_hours := 32

-- Define the proof statement
theorem number_of_days_worked : total_hours / hours_per_day = 4 :=
by
  -- Placeholder for the proof
  sorry

end number_of_days_worked_l193_193731


namespace impossibility_exchange_l193_193959

theorem impossibility_exchange :
  ¬ ∃ (x y z : ℕ), (x + y + z = 10) ∧ (x + 3 * y + 5 * z = 25) := 
by
  sorry

end impossibility_exchange_l193_193959


namespace gray_region_correct_b_l193_193658

-- Define the basic conditions
def square_side_length : ℝ := 3
def small_square_side_length : ℝ := 1

-- Define the triangles resulting from cutting a square
def triangle_area : ℝ := 0.5 * square_side_length * square_side_length

-- Define the gray region area for the second figure (b)
def gray_region_area_b : ℝ := 0.25

-- Lean statement to prove the area of the gray region
theorem gray_region_correct_b : gray_region_area_b = 0.25 := by
  -- Proof is omitted
  sorry

end gray_region_correct_b_l193_193658


namespace delivery_meals_l193_193329

theorem delivery_meals (M P : ℕ) 
  (h1 : P = 8 * M) 
  (h2 : M + P = 27) : 
  M = 3 := by
  sorry

end delivery_meals_l193_193329


namespace ariel_fish_l193_193854

theorem ariel_fish (total_fish : ℕ) (male_ratio : ℚ) (female_ratio : ℚ) (female_fish : ℕ) : 
  total_fish = 45 ∧ male_ratio = 2/3 ∧ female_ratio = 1/3 → female_fish = 15 :=
by
  sorry

end ariel_fish_l193_193854


namespace months_in_season_l193_193601

/-- Definitions for conditions in the problem --/
def total_games_per_month : ℝ := 323.0
def total_games_season : ℝ := 5491.0

/-- The statement to be proven: The number of months in the season --/
theorem months_in_season (x : ℝ) (h : x = total_games_season / total_games_per_month) : x = 17.0 := by
  sorry

end months_in_season_l193_193601


namespace purpose_of_LB_full_nutrient_medium_l193_193421

/--
Given the experiment "Separation of Microorganisms in Soil Using Urea as a Nitrogen Source",
which involves both experimental and control groups with the following conditions:
- The variable in the experiment is the difference in the medium used.
- The experimental group uses a medium with urea as the only nitrogen source (selective medium).
- The control group uses a full-nutrient medium.

Prove that the purpose of preparing LB full-nutrient medium is to observe the types and numbers
of soil microorganisms that can grow under full-nutrient conditions.
-/
theorem purpose_of_LB_full_nutrient_medium
  (experiment: String) (experimental_variable: String) (experimental_group: String) (control_group: String)
  (H1: experiment = "Separation of Microorganisms in Soil Using Urea as a Nitrogen Source")
  (H2: experimental_variable = "medium")
  (H3: experimental_group = "medium with urea as the only nitrogen source (selective medium)")
  (H4: control_group = "full-nutrient medium") :
  purpose_of_preparing_LB_full_nutrient_medium = "observe the types and numbers of soil microorganisms that can grow under full-nutrient conditions" :=
sorry

end purpose_of_LB_full_nutrient_medium_l193_193421


namespace distinct_positive_least_sum_seven_integers_prod_2016_l193_193004

theorem distinct_positive_least_sum_seven_integers_prod_2016 :
  ∃ (n1 n2 n3 n4 n5 n6 n7 : ℕ),
    n1 < n2 ∧ n2 < n3 ∧ n3 < n4 ∧ n4 < n5 ∧ n5 < n6 ∧ n6 < n7 ∧
    (n1 * n2 * n3 * n4 * n5 * n6 * n7) % 2016 = 0 ∧
    n1 + n2 + n3 + n4 + n5 + n6 + n7 = 31 :=
sorry

end distinct_positive_least_sum_seven_integers_prod_2016_l193_193004


namespace smallest_number_increased_by_3_divisible_l193_193478

theorem smallest_number_increased_by_3_divisible (n : ℤ) 
    (h1 : (n + 3) % 18 = 0)
    (h2 : (n + 3) % 70 = 0)
    (h3 : (n + 3) % 25 = 0)
    (h4 : (n + 3) % 21 = 0) : 
    n = 3147 :=
by
  sorry

end smallest_number_increased_by_3_divisible_l193_193478


namespace greatest_number_of_rented_trucks_l193_193702

-- Define the conditions
def total_trucks_on_monday : ℕ := 24
def trucks_returned_percentage : ℕ := 50
def trucks_on_lot_saturday (R : ℕ) (P : ℕ) : ℕ := (R * P) / 100
def min_trucks_on_lot_saturday : ℕ := 12

-- Define the theorem
theorem greatest_number_of_rented_trucks : ∃ R, R = total_trucks_on_monday ∧ trucks_returned_percentage = 50 ∧ min_trucks_on_lot_saturday = 12 → R = 24 :=
by
  sorry

end greatest_number_of_rented_trucks_l193_193702


namespace boys_in_classroom_l193_193428

-- Definitions of the conditions
def total_children := 45
def girls_fraction := 1 / 3

-- The theorem we want to prove
theorem boys_in_classroom : (2 / 3) * total_children = 30 := by
  sorry

end boys_in_classroom_l193_193428


namespace probability_at_most_one_incorrect_l193_193692

variable (p : ℝ)

theorem probability_at_most_one_incorrect (h : 0 ≤ p ∧ p ≤ 1) :
  p^9 * (10 - 9*p) = p^10 + 10 * (1 - p) * p^9 := by
  sorry

end probability_at_most_one_incorrect_l193_193692


namespace initial_cloves_l193_193026

theorem initial_cloves (used_cloves left_cloves initial_cloves : ℕ) (h1 : used_cloves = 86) (h2 : left_cloves = 7) : initial_cloves = 93 :=
by
  sorry

end initial_cloves_l193_193026


namespace probability_king_even_coords_2008_l193_193190

noncomputable def king_probability_even_coords (turns : ℕ) : ℝ :=
  let p_stay := 0.4
  let p_edge := 0.1
  let p_diag := 0.05
  if turns = 2008 then
    (5 ^ 2008 + 1) / (2 * 5 ^ 2008)
  else
    0 -- default value for other cases

theorem probability_king_even_coords_2008 :
  king_probability_even_coords 2008 = (5 ^ 2008 + 1) / (2 * 5 ^ 2008) :=
by
  sorry

end probability_king_even_coords_2008_l193_193190


namespace div64_by_expression_l193_193760

theorem div64_by_expression {n : ℕ} (h : n > 0) : ∃ k : ℤ, (3^(2 * n + 2) - 8 * ↑n - 9) = 64 * k :=
by
  sorry

end div64_by_expression_l193_193760


namespace sum_digits_10_pow_85_minus_85_l193_193480

-- Define the function that computes the sum of the digits
def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 10) + sum_of_digits (n / 10)

-- Define the specific problem for n = 10^85 - 85
theorem sum_digits_10_pow_85_minus_85 : 
  sum_of_digits (10^85 - 85) = 753 :=
by
  sorry

end sum_digits_10_pow_85_minus_85_l193_193480


namespace working_mom_work_percent_l193_193416

theorem working_mom_work_percent :
  let awake_hours := 16
  let work_hours := 8
  (work_hours / awake_hours) * 100 = 50 :=
by
  sorry

end working_mom_work_percent_l193_193416


namespace bus_seats_needed_l193_193231

def members_playing_instruments : Prop :=
  let flute := 5
  let trumpet := 3 * flute
  let trombone := trumpet - 8
  let drum := trombone + 11
  let clarinet := 2 * flute
  let french_horn := trombone + 3
  let saxophone := (trumpet + trombone) / 2
  let piano := drum + 2
  let violin := french_horn - clarinet
  let guitar := 3 * flute
  let total_members := flute + trumpet + trombone + drum + clarinet + french_horn + saxophone + piano + violin + guitar
  total_members = 111

theorem bus_seats_needed : members_playing_instruments :=
by
  sorry

end bus_seats_needed_l193_193231


namespace inequality_not_always_true_l193_193907

variables {a b c d : ℝ}

theorem inequality_not_always_true
  (h1 : a > b) (h2 : b > 0) (h3 : c > 0) (h4 : d ≠ 0) :
  ¬ ∀ (a b d : ℝ), (a > b) → (d ≠ 0) → (a + d)^2 > (b + d)^2 :=
by
  intro H
  specialize H a b d h1 h4
  sorry

end inequality_not_always_true_l193_193907


namespace treShaun_marker_ink_left_l193_193300

noncomputable def ink_left_percentage (marker_area : ℕ) (total_colored_area : ℕ) : ℕ :=
if total_colored_area >= marker_area then 0 else ((marker_area - total_colored_area) * 100) / marker_area

theorem treShaun_marker_ink_left :
  let marker_area := 3 * (4 * 4)
  let colored_area := (2 * (6 * 2) + 8 * 4)
  ink_left_percentage marker_area colored_area = 0 :=
by
  sorry

end treShaun_marker_ink_left_l193_193300


namespace fewer_cans_today_l193_193077

variable (nc_sarah_yesterday : ℕ)
variable (nc_lara_yesterday : ℕ)
variable (nc_alex_yesterday : ℕ)
variable (nc_sarah_today : ℕ)
variable (nc_lara_today : ℕ)
variable (nc_alex_today : ℕ)

-- Given conditions
def yesterday_collected_cans : Prop :=
  nc_sarah_yesterday = 50 ∧
  nc_lara_yesterday = nc_sarah_yesterday + 30 ∧
  nc_alex_yesterday = 90

def today_collected_cans : Prop :=
  nc_sarah_today = 40 ∧
  nc_lara_today = 70 ∧
  nc_alex_today = 55

theorem fewer_cans_today :
  yesterday_collected_cans nc_sarah_yesterday nc_lara_yesterday nc_alex_yesterday →
  today_collected_cans nc_sarah_today nc_lara_today nc_alex_today →
  (nc_sarah_yesterday + nc_lara_yesterday + nc_alex_yesterday) -
  (nc_sarah_today + nc_lara_today + nc_alex_today) = 55 :=
by
  intros h1 h2
  sorry

end fewer_cans_today_l193_193077


namespace calculate_seasons_l193_193005

theorem calculate_seasons :
  ∀ (episodes_per_season : ℕ) (episodes_per_day : ℕ) (days : ℕ),
  episodes_per_season = 20 →
  episodes_per_day = 2 →
  days = 30 →
  (episodes_per_day * days) / episodes_per_season = 3 :=
by
  intros episodes_per_season episodes_per_day days h_eps h_epd h_d
  sorry

end calculate_seasons_l193_193005


namespace mark_trees_total_l193_193042

def mark_trees (current_trees new_trees : Nat) : Nat :=
  current_trees + new_trees

theorem mark_trees_total (x y : Nat) (h1 : x = 13) (h2 : y = 12) :
  mark_trees x y = 25 :=
by
  rw [h1, h2]
  sorry

end mark_trees_total_l193_193042


namespace combined_marbles_l193_193886

def Rhonda_marbles : ℕ := 80
def Amon_marbles : ℕ := Rhonda_marbles + 55

theorem combined_marbles : Amon_marbles + Rhonda_marbles = 215 :=
by
  sorry

end combined_marbles_l193_193886


namespace sum_of_interior_angles_of_polygon_l193_193060

theorem sum_of_interior_angles_of_polygon (exterior_angle : ℝ) (h : exterior_angle = 36) :
  ∃ interior_sum : ℝ, interior_sum = 1440 :=
by
  sorry

end sum_of_interior_angles_of_polygon_l193_193060


namespace common_difference_arithmetic_sequence_l193_193654

theorem common_difference_arithmetic_sequence 
    (a : ℕ → ℝ) 
    (S₅ : ℝ)
    (h_arith : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
    (h₁ : a 4 + a 6 = 6)
    (h₂ : S₅ = (a 1 + a 2 + a 3 + a 4 + a 5))
    (h_S₅_val : S₅ = 10) :
  ∃ d : ℝ, d = (a 5 - a 1) / 4 ∧ d = 1/2 := 
by
  sorry

end common_difference_arithmetic_sequence_l193_193654


namespace arithmetic_progression_x_value_l193_193172

theorem arithmetic_progression_x_value (x: ℝ) (h1: 3*x - 1 - (2*x - 3) = 4*x + 1 - (3*x - 1)) : x = 3 :=
by
  sorry

end arithmetic_progression_x_value_l193_193172


namespace trigonometric_relationship_l193_193352

noncomputable def a : ℝ := Real.sin (393 * Real.pi / 180)
noncomputable def b : ℝ := Real.cos (55 * Real.pi / 180)
noncomputable def c : ℝ := Real.tan (50 * Real.pi / 180)

theorem trigonometric_relationship : a < b ∧ b < c := by
  sorry

end trigonometric_relationship_l193_193352


namespace yen_exchange_rate_l193_193012

theorem yen_exchange_rate (yen_per_dollar : ℕ) (dollars : ℕ) (y : ℕ) (h1 : yen_per_dollar = 120) (h2 : dollars = 10) : y = 1200 :=
by
  have h3 : y = yen_per_dollar * dollars := by sorry
  rw [h1, h2] at h3
  exact h3

end yen_exchange_rate_l193_193012


namespace tan_double_angle_difference_l193_193576

variable {α β : Real}

theorem tan_double_angle_difference (h1 : Real.tan α = 1 / 2) (h2 : Real.tan (α - β) = 1 / 5) :
  Real.tan (2 * α - β) = 7 / 9 := 
sorry

end tan_double_angle_difference_l193_193576


namespace probability_of_graduate_degree_l193_193164

variables (G C N : ℕ)
axiom h1 : G / N = 1 / 8
axiom h2 : C / N = 2 / 3

noncomputable def total_college_graduates (G C : ℕ) : ℕ := G + C

noncomputable def probability_graduate_degree (G C : ℕ) : ℚ := G / (total_college_graduates G C)

theorem probability_of_graduate_degree :
  probability_graduate_degree 3 16 = 3 / 19 :=
by 
  -- Here, we need to prove that the probability of picking a college graduate with a graduate degree
  -- is 3 / 19 given the conditions.
  sorry

end probability_of_graduate_degree_l193_193164


namespace find_difference_l193_193720

noncomputable def expr (a b : ℝ) : ℝ :=
  |a - b| / (|a| + |b|)

def min_val (a b : ℝ) : ℝ := 0

def max_val (a b : ℝ) : ℝ := 1

theorem find_difference (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) :
  max_val a b - min_val a b = 1 :=
by
  sorry

end find_difference_l193_193720


namespace find_a_plus_b_l193_193429

open Function

theorem find_a_plus_b (a b : ℝ) (f g h : ℝ → ℝ)
  (h_f : ∀ x, f x = a * x - b)
  (h_g : ∀ x, g x = -4 * x - 1)
  (h_h : ∀ x, h x = f (g x))
  (h_h_inv : ∀ y, h⁻¹ y = y + 9) :
  a + b = -9 := 
by
  -- Proof goes here.
  sorry

end find_a_plus_b_l193_193429


namespace certain_event_at_least_one_genuine_l193_193974

def products : Finset (Fin 12) := sorry
def genuine : Finset (Fin 12) := sorry
def defective : Finset (Fin 12) := sorry
noncomputable def draw3 : Finset (Finset (Fin 12)) := sorry

-- Condition: 12 identical products, 10 genuine, 2 defective
axiom products_condition_1 : products.card = 12
axiom products_condition_2 : genuine.card = 10
axiom products_condition_3 : defective.card = 2
axiom products_condition_4 : ∀ x ∈ genuine, x ∈ products
axiom products_condition_5 : ∀ x ∈ defective, x ∈ products
axiom products_condition_6 : genuine ∩ defective = ∅

-- The statement to be proved: when drawing 3 products randomly, it is certain that at least 1 is genuine.
theorem certain_event_at_least_one_genuine :
  ∀ s ∈ draw3, ∃ x ∈ s, x ∈ genuine :=
sorry

end certain_event_at_least_one_genuine_l193_193974


namespace intersection_A_B_l193_193532

def A : Set ℝ := {x | x < 3 * x - 1}
def B : Set ℝ := {x | -1 < x ∧ x < 3}

theorem intersection_A_B : (A ∩ B) = {x | x > 1 / 2 ∧ x < 3} :=
by sorry

end intersection_A_B_l193_193532


namespace highest_number_on_dice_l193_193977

theorem highest_number_on_dice (n : ℕ) (h1 : 0 < n)
  (h2 : ∃ p : ℝ, p = 0.1111111111111111) 
  (h3 : 1 / 9 = 4 / (n * n)) 
  : n = 6 :=
sorry

end highest_number_on_dice_l193_193977


namespace pq_square_identity_l193_193552

theorem pq_square_identity (p q : ℝ) (h1 : p - q = 4) (h2 : p * q = -2) : p^2 + q^2 = 12 :=
by
  sorry

end pq_square_identity_l193_193552


namespace quadratic_root_2020_l193_193911

theorem quadratic_root_2020 (a b : ℝ) (h₀ : a ≠ 0) (h₁ : a * 2019^2 + b * 2019 - 1 = 0) :
    ∃ x : ℝ, (a * (x - 1)^2 + b * (x - 1) = 1) ∧ x = 2020 :=
by
  sorry

end quadratic_root_2020_l193_193911


namespace problem_l193_193922

theorem problem (h : (0.00027 : ℝ) = 27 / 100000) : (10^5 - 10^3) * 0.00027 = 26.73 := by
  sorry

end problem_l193_193922


namespace basket_can_hold_40_fruits_l193_193869

-- Let us define the number of oranges as 10
def oranges : ℕ := 10

-- There are 3 times as many apples as oranges
def apples : ℕ := 3 * oranges

-- The total number of fruits in the basket
def total_fruits : ℕ := oranges + apples

theorem basket_can_hold_40_fruits (h₁ : oranges = 10) (h₂ : apples = 3 * oranges) : total_fruits = 40 :=
by
  -- We assume the conditions and derive the conclusion
  sorry

end basket_can_hold_40_fruits_l193_193869


namespace postal_code_permutations_l193_193045

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def multiplicity_permutations (n : Nat) (repetitions : List Nat) : Nat :=
  factorial n / List.foldl (λ acc k => acc * factorial k) 1 repetitions

theorem postal_code_permutations : multiplicity_permutations 4 [2, 1, 1] = 12 :=
by
  unfold multiplicity_permutations
  unfold factorial
  sorry

end postal_code_permutations_l193_193045


namespace problem_statement_l193_193296

variable (a b x : ℝ)

theorem problem_statement (h1 : x = a / b) (h2 : a ≠ b) (h3 : b ≠ 0) : 
  a / (a - b) = x / (x - 1) :=
sorry

end problem_statement_l193_193296


namespace sequence_properties_l193_193636

theorem sequence_properties (a : ℕ → ℝ)
  (h1 : a 1 = 1 / 5)
  (h2 : ∀ n : ℕ, n > 1 → a (n - 1) / a n = (2 * a (n - 1) + 1) / (1 - 2 * a n)) :
  (∀ n : ℕ, n > 0 → (1 / a n) - (1 / a (n - 1)) = 4) ∧
  (∀ m k : ℕ, m > 0 ∧ k > 0 → a m * a k = a (4 * m * k + m + k)) :=
by
  sorry

end sequence_properties_l193_193636


namespace precious_stones_l193_193555

variable (total_amount : ℕ) (price_per_stone : ℕ) (number_of_stones : ℕ)

theorem precious_stones (h1 : total_amount = 14280) (h2 : price_per_stone = 1785) : number_of_stones = 8 :=
by
  sorry

end precious_stones_l193_193555


namespace sarah_earnings_l193_193397

-- Conditions
def monday_hours : ℚ := 1 + 3 / 4
def wednesday_hours : ℚ := 65 / 60
def thursday_hours : ℚ := 2 + 45 / 60
def friday_hours : ℚ := 45 / 60
def saturday_hours : ℚ := 2

def weekday_rate : ℚ := 4
def weekend_rate : ℚ := 6

-- Definition for total earnings
def total_weekday_earnings : ℚ :=
  (monday_hours + wednesday_hours + thursday_hours + friday_hours) * weekday_rate

def total_weekend_earnings : ℚ :=
  saturday_hours * weekend_rate

def total_earnings : ℚ :=
  total_weekday_earnings + total_weekend_earnings

-- Statement to prove
theorem sarah_earnings : total_earnings = 37.3332 := by
  sorry

end sarah_earnings_l193_193397


namespace multiple_proof_l193_193828

theorem multiple_proof (n m : ℝ) (h1 : n = 25) (h2 : m * n = 3 * n - 25) : m = 2 := by
  sorry

end multiple_proof_l193_193828


namespace count_four_digit_numbers_with_digit_sum_4_l193_193401

theorem count_four_digit_numbers_with_digit_sum_4 : 
  ∃ n : ℕ, (∀ (x1 x2 x3 x4 : ℕ), 
    x1 + x2 + x3 + x4 = 4 ∧ x1 ≥ 1 ∧ x2 ≥ 0 ∧ x3 ≥ 0 ∧ x4 ≥ 0 →
    n = 20) :=
sorry

end count_four_digit_numbers_with_digit_sum_4_l193_193401


namespace max_value_l193_193276

theorem max_value (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  ∃ m ≤ 4, ∀ (z w : ℝ), z > 0 → w > 0 → (x + y = z + w) → (z^3 + w^3 ≥ x^3 + y^3 → 
  (z + w)^3 / (z^3 + w^3) ≤ m) :=
sorry

end max_value_l193_193276


namespace symmetric_line_eq_l193_193437

-- Define the original line equation
def original_line (x: ℝ) : ℝ := -2 * x - 3

-- Define the symmetric line with respect to y-axis
def symmetric_line (x: ℝ) : ℝ := 2 * x - 3

-- The theorem stating the symmetric line with respect to the y-axis
theorem symmetric_line_eq : (∀ x: ℝ, original_line (-x) = symmetric_line x) :=
by
  -- Proof goes here
  sorry

end symmetric_line_eq_l193_193437


namespace cone_sphere_volume_ratio_l193_193098

theorem cone_sphere_volume_ratio (r h : ℝ) 
  (radius_eq : r > 0)
  (volume_rel : (1 / 3 : ℝ) * π * r^2 * h = (1 / 3 : ℝ) * (4 / 3) * π * r^3) : 
  h / r = 4 / 3 :=
by
  sorry

end cone_sphere_volume_ratio_l193_193098


namespace min_sum_arth_seq_l193_193758

theorem min_sum_arth_seq (a : ℕ → ℤ) (n : ℕ)
  (h1 : ∀ k, a k = a 1 + (k - 1) * (a 2 - a 1))
  (h2 : a 1 = -3)
  (h3 : 11 * a 5 = 5 * a 8) : n = 4 := by
  sorry

end min_sum_arth_seq_l193_193758


namespace pieces_present_l193_193013

def total_pieces : ℕ := 32
def missing_pieces : ℕ := 10

theorem pieces_present : total_pieces - missing_pieces = 22 :=
by {
  sorry
}

end pieces_present_l193_193013


namespace voice_of_china_signup_ways_l193_193561

theorem voice_of_china_signup_ways : 
  (2 * 2 * 2 = 8) :=
by {
  sorry
}

end voice_of_china_signup_ways_l193_193561


namespace parities_of_E_10_11_12_l193_193240

noncomputable def E : ℕ → ℕ
| 0 => 1
| 1 => 2
| 2 => 3
| (n + 3) => 2 * (E (n + 2)) + (E n)

theorem parities_of_E_10_11_12 :
  (E 10 % 2 = 0) ∧ (E 11 % 2 = 1) ∧ (E 12 % 2 = 1) := 
  by
  sorry

end parities_of_E_10_11_12_l193_193240


namespace largest_of_five_consecutive_composite_integers_under_40_l193_193155

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

noncomputable def five_consecutive_composite_integers_under_40 : List ℕ :=
[32, 33, 34, 35, 36]

theorem largest_of_five_consecutive_composite_integers_under_40 :
  ∀ n ∈ five_consecutive_composite_integers_under_40,
  n < 40 ∧ ∀ k, (k ∈ five_consecutive_composite_integers_under_40 →
  ¬ is_prime k) →
  List.maximum five_consecutive_composite_integers_under_40 = some 36 :=
by
  sorry

end largest_of_five_consecutive_composite_integers_under_40_l193_193155


namespace coupon_probability_l193_193495

theorem coupon_probability : 
  (Nat.choose 6 6 * Nat.choose 11 3) / Nat.choose 17 9 = 3 / 442 := 
by
  sorry

end coupon_probability_l193_193495


namespace remainder_of_3_pow_99_plus_5_mod_9_l193_193537

theorem remainder_of_3_pow_99_plus_5_mod_9 : (3 ^ 99 + 5) % 9 = 5 := by
  -- Here we state the main goal
  sorry -- Proof to be filled in

end remainder_of_3_pow_99_plus_5_mod_9_l193_193537


namespace distance_from_center_of_C_to_line_l193_193129

def circle_center_distance : ℝ :=
  let line1 (x y : ℝ) := x - y - 4
  let circle1 (x y : ℝ) := x^2 + y^2 - 4 * x - 6
  let circle2 (x y : ℝ) := x^2 + y^2 - 4 * y - 6
  let line2 (x y : ℝ) := 3 * x + 4 * y + 5
  sorry

theorem distance_from_center_of_C_to_line :
  circle_center_distance = 2 := sorry

end distance_from_center_of_C_to_line_l193_193129


namespace arccos_sin_1_5_eq_pi_over_2_minus_1_5_l193_193342

-- Define the problem statement in Lean 4.
theorem arccos_sin_1_5_eq_pi_over_2_minus_1_5 : 
  Real.arccos (Real.sin 1.5) = (Real.pi / 2) - 1.5 :=
by
  sorry

end arccos_sin_1_5_eq_pi_over_2_minus_1_5_l193_193342


namespace ratio_of_circle_areas_l193_193929

variable (S L A : ℝ)

theorem ratio_of_circle_areas 
  (h1 : A = (3 / 5) * S)
  (h2 : A = (6 / 25) * L)
  : S / L = 2 / 5 :=
by
  sorry

end ratio_of_circle_areas_l193_193929


namespace pentagon_perpendicular_sums_l193_193331

noncomputable def FO := 2
noncomputable def FQ := 2
noncomputable def FR := 2

theorem pentagon_perpendicular_sums :
  FO + FQ + FR = 6 :=
by
  sorry

end pentagon_perpendicular_sums_l193_193331


namespace karen_age_is_10_l193_193528

-- Definitions for the given conditions
def ages : List ℕ := [2, 4, 6, 8, 10, 12, 14]

def to_park (a b : ℕ) : Prop := a + b = 20
def to_pool (a b : ℕ) : Prop := 3 < a ∧ a < 9 ∧ 3 < b ∧ b < 9
def stayed_home (karen_age : ℕ) : Prop := karen_age = 10

-- Theorem stating Karen's age is 10 given the conditions
theorem karen_age_is_10 :
  ∃ (a b c d e f g : ℕ),
  ages = [a, b, c, d, e, f, g] ∧
  ((to_park a b ∨ to_park a c ∨ to_park a d ∨ to_park a e ∨ to_park a f ∨ to_park a g ∨
  to_park b c ∨ to_park b d ∨ to_park b e ∨ to_park b f ∨ to_park b g ∨
  to_park c d ∨ to_park c e ∨ to_park c f ∨ to_park c g ∨
  to_park d e ∨ to_park d f ∨ to_park d g ∨
  to_park e f ∨ to_park e g ∨
  to_park f g)) ∧
  ((to_pool a b ∨ to_pool a c ∨ to_pool a d ∨ to_pool a e ∨ to_pool a f ∨ to_pool a g ∨
  to_pool b c ∨ to_pool b d ∨ to_pool b e ∨ to_pool b f ∨ to_pool b g ∨
  to_pool c d ∨ to_pool c e ∨ to_pool c f ∨
  to_pool d e ∨ to_pool d f ∨
  to_pool e f ∨
  to_pool f g)) ∧
  stayed_home 4 :=
sorry

end karen_age_is_10_l193_193528


namespace loss_per_meter_calculation_l193_193002

/-- Define the given constants and parameters. --/
def total_meters : ℕ := 600
def selling_price : ℕ := 18000
def cost_price_per_meter : ℕ := 35

/-- Now we define the total cost price, total loss and loss per meter --/
def total_cost_price : ℕ := cost_price_per_meter * total_meters
def total_loss : ℕ := total_cost_price - selling_price
def loss_per_meter : ℕ := total_loss / total_meters

/-- State the theorem we need to prove. --/
theorem loss_per_meter_calculation : loss_per_meter = 5 :=
by
  sorry

end loss_per_meter_calculation_l193_193002


namespace arithmetic_geometric_sequence_ratio_l193_193174

section
variables {a : ℕ → ℝ} {S : ℕ → ℝ}
variables {d : ℝ}

-- Definition of the arithmetic sequence and its sum
def arithmetic_sequence (a : ℕ → ℝ) (a₁ d : ℝ) : Prop :=
  ∀ n, a n = a₁ + (n - 1) * d

def sum_arithmetic_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2

-- Given conditions
-- 1. S is the sum of the first n terms of the arithmetic sequence a
axiom sn_arith_seq : sum_arithmetic_sequence S a

-- 2. a_1, a_3, and a_4 form a geometric sequence
axiom geom_seq : (a 1 + 2 * d) ^ 2 = a 1 * (a 1 + 3 * d)

-- Goal is to prove the given ratio equation
theorem arithmetic_geometric_sequence_ratio (h : ∀ n, a n = -4 * d + (n - 1) * d) :
  (S 3 - S 2) / (S 5 - S 3) = 2 :=
sorry
end

end arithmetic_geometric_sequence_ratio_l193_193174


namespace total_ages_l193_193849

variable (Frank : ℕ) (Gabriel : ℕ)
variables (h1 : Frank = 10) (h2 : Gabriel = Frank - 3)

theorem total_ages (hF : Frank = 10) (hG : Gabriel = Frank - 3) : Frank + Gabriel = 17 :=
by
  rw [hF, hG]
  norm_num
  sorry

end total_ages_l193_193849


namespace joe_first_lift_weight_l193_193175

theorem joe_first_lift_weight (x y : ℕ) (h1 : x + y = 1500) (h2 : 2 * x = y + 300) : x = 600 :=
by
  sorry

end joe_first_lift_weight_l193_193175


namespace average_remaining_numbers_l193_193184

theorem average_remaining_numbers (numbers : List ℝ) 
  (h_len : numbers.length = 50) 
  (h_avg : (numbers.sum / 50 : ℝ) = 38) 
  (h_discard : 45 ∈ numbers ∧ 55 ∈ numbers) :
  let new_sum := numbers.sum - 45 - 55
  let new_len := 50 - 2
  (new_sum / new_len : ℝ) = 37.5 :=
by
  sorry

end average_remaining_numbers_l193_193184


namespace smaller_solution_of_quadratic_eq_l193_193855

noncomputable def smaller_solution (a b c : ℝ) : ℝ :=
  if a ≠ 0 then min ((-b + Real.sqrt (b ^ 2 - 4 * a * c)) / (2 * a))
              ((-b - Real.sqrt (b ^ 2 - 4 * a * c)) / (2 * a))
  else if b ≠ 0 then -c / b else 0 

theorem smaller_solution_of_quadratic_eq :
  smaller_solution 1 (-13) (-30) = -2 := 
by
  sorry

end smaller_solution_of_quadratic_eq_l193_193855


namespace sum_of_squares_of_coefficients_l193_193811

theorem sum_of_squares_of_coefficients :
  let poly := 5 * (X^6 + 4 * X^4 + 2 * X^2 + 1)
  let coeffs := [5, 20, 10, 5]
  (coeffs.map (λ c => c * c)).sum = 550 := 
by
  sorry

end sum_of_squares_of_coefficients_l193_193811


namespace triangle_side_length_l193_193870

theorem triangle_side_length (A B : ℝ) (b : ℝ) (a : ℝ) 
  (hA : A = 60) (hB : B = 45) (hb : b = 2) 
  (h : a = b * (Real.sin A) / (Real.sin B)) :
  a = Real.sqrt 6 := by
  sorry

end triangle_side_length_l193_193870


namespace tangent_line_equation_l193_193832

theorem tangent_line_equation :
  (∃ l : ℝ → ℝ, 
   (∀ x, l x = (1 / (4 + 2 * Real.sqrt 3)) * x + (2 + Real.sqrt 3) / 2 ∨ 
         l x = (1 / (4 - 2 * Real.sqrt 3)) * x + (2 - Real.sqrt 3) / 2) ∧ 
   (l 1 = 2) ∧ 
   (∀ x, l x = Real.sqrt x)
  ) →
  (∀ x y, 
   (y = (1 / 4 + Real.sqrt 3) * x + (2 + Real.sqrt 3) / 2 ∨ 
    y = (1 / 4 - Real.sqrt 3) * x + (2 - Real.sqrt 3) / 2) ∨ 
   (x - (4 + 2 * Real.sqrt 3) * y + (7 + 4 * Real.sqrt 3) = 0 ∨ 
    x - (4 - 2 * Real.sqrt 3) * y + (7 - 4 * Real.sqrt 3) = 0)
) :=
sorry

end tangent_line_equation_l193_193832


namespace sufficient_not_necessary_condition_l193_193571

variables (a b : Line) (α β : Plane)

def Line : Type := sorry
def Plane : Type := sorry

-- Conditions: a and b are different lines, α and β are different planes
axiom diff_lines : a ≠ b
axiom diff_planes : α ≠ β

-- Perpendicular and parallel definitions
def perp (l : Line) (p : Plane) : Prop := sorry
def parallel (p1 p2 : Plane) : Prop := sorry

-- Sufficient but not necessary condition
theorem sufficient_not_necessary_condition
  (h1 : perp a β)
  (h2 : parallel α β) :
  perp a α :=
sorry

end sufficient_not_necessary_condition_l193_193571


namespace problem_l193_193701

theorem problem 
  {a1 a2 : ℝ}
  (h1 : 0 ≤ a1)
  (h2 : 0 ≤ a2)
  (h3 : a1 + a2 = 1) :
  ∃ (b1 b2 : ℝ), 0 ≤ b1 ∧ 0 ≤ b2 ∧ b1 + b2 = 1 ∧ ((5/4 - a1) * b1 + 3 * (5/4 - a2) * b2 > 1) :=
by
  sorry

end problem_l193_193701


namespace find_other_number_l193_193898

theorem find_other_number (x y : ℕ) (h1 : x + y = 10) (h2 : 2 * x = 3 * y + 5) (h3 : x = 7) : y = 3 :=
by
  sorry

end find_other_number_l193_193898


namespace nesbitt_inequality_l193_193919

variable (a b c : ℝ)
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

theorem nesbitt_inequality (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a / (b + c) + b / (c + a) + c / (a + b) ≥ 3 / 2 := 
by
  sorry

end nesbitt_inequality_l193_193919


namespace length_of_ship_l193_193282

-- Variables and conditions
variables (E L S : ℝ)
variables (W : ℝ := 0.9) -- Wind reducing factor

-- Conditions as equations
def condition1 : Prop := 150 * E = L + 150 * S
def condition2 : Prop := 70 * E = L - 63 * S

-- Theorem to prove
theorem length_of_ship (hc1 : condition1 E L S) (hc2 : condition2 E L S) : L = (19950 / 213) * E :=
sorry

end length_of_ship_l193_193282


namespace kiki_total_money_l193_193775

theorem kiki_total_money 
  (S : ℕ) (H : ℕ) (M : ℝ)
  (h1: S = 18)
  (h2: H = 2 * S)
  (h3: 0.40 * M = 36) : 
  M = 90 :=
by
  sorry

end kiki_total_money_l193_193775


namespace inequalities_always_true_l193_193048

variables {x y a b : Real}

/-- All given conditions -/
def conditions (x y a b : Real) :=
  x < a ∧ y < b ∧ x < 0 ∧ y < 0 ∧ a > 0 ∧ b > 0

theorem inequalities_always_true {x y a b : Real} (h : conditions x y a b) :
  (x + y < a + b) ∧ 
  (x - y < a - b) ∧ 
  (x * y < a * b) ∧ 
  ((x + y) / (x - y) < (a + b) / (a - b)) :=
sorry

end inequalities_always_true_l193_193048


namespace remainder_of_127_div_25_is_2_l193_193188

theorem remainder_of_127_div_25_is_2 : ∃ r, 127 = 25 * 5 + r ∧ r = 2 := by
  have h1 : 127 = 25 * 5 + (127 - 25 * 5) := by rw [mul_comm 25 5, mul_comm 5 25]
  have h2 : 127 - 25 * 5 = 2 := by norm_num
  exact ⟨127 - 25 * 5, h1, h2⟩

end remainder_of_127_div_25_is_2_l193_193188


namespace problem1_problem2_l193_193000

open Set

noncomputable def A : Set ℝ := {x | x^2 - 6*x + 8 < 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | (x - a) * (x - 3 * a) < 0}

theorem problem1 (a : ℝ) (h1 : A ⊆ (A ∩ B a)) : (4 / 3 : ℝ) ≤ a ∧ a ≤ 2 :=
sorry

theorem problem2 (a : ℝ) (h2 : A ∩ B a = ∅) : a ≤ (2 / 3 : ℝ) ∨ a ≥ 4 :=
sorry

end problem1_problem2_l193_193000


namespace xy_gt_xz_l193_193057

variable {R : Type*} [LinearOrderedField R]
variables (x y z : R)

theorem xy_gt_xz (h1 : x > y) (h2 : y > z) (h3 : x + y + z = 0) : x * y > x * z :=
by
  sorry

end xy_gt_xz_l193_193057


namespace no_infinite_seq_pos_int_l193_193419

theorem no_infinite_seq_pos_int : 
  ¬∃ (a : ℕ → ℕ), 
  (∀ n : ℕ, 0 < a n) ∧ 
  ∀ n : ℕ, a (n+1) ^ 2 ≥ 2 * a n * a (n+2) :=
by
  sorry

end no_infinite_seq_pos_int_l193_193419


namespace range_of_f1_3_l193_193411

noncomputable def f (a b : ℝ) (x y : ℝ) : ℝ :=
  a * (x^3 + 3 * x) + b * (y^2 + 2 * y + 1)

theorem range_of_f1_3 (a b : ℝ)
  (h1 : 1 ≤ f a b 1 2 ∧ f a b 1 2 ≤ 2)
  (h2 : 2 ≤ f a b 3 4 ∧ f a b 3 4 ≤ 5):
  3 / 2 ≤ f a b 1 3 ∧ f a b 1 3 ≤ 4 :=
sorry

end range_of_f1_3_l193_193411


namespace k_value_l193_193268

theorem k_value (k : ℝ) (x : ℝ) (y : ℝ) (hk : k^2 - 5 = -1) (hx : x > 0) (hy : y = (k - 1) * x^(k^2 - 5)) (h_dec : ∀ (x1 x2 : ℝ), x1 > 0 → x2 > x1 → (k - 1) * x2^(k^2 - 5) < (k - 1) * x1^(k^2 - 5)):
  k = 2 := by
  sorry

end k_value_l193_193268


namespace shells_not_red_or_green_l193_193311

theorem shells_not_red_or_green (total_shells : ℕ) (red_shells : ℕ) (green_shells : ℕ) 
  (h_total : total_shells = 291) (h_red : red_shells = 76) (h_green : green_shells = 49) :
  total_shells - (red_shells + green_shells) = 166 :=
by
  sorry

end shells_not_red_or_green_l193_193311


namespace product_is_48_l193_193362

-- Define the conditions and the target product
def problem (x y : ℝ) := 
  x ≠ y ∧ (x + y) / (x - y) = 7 ∧ (x * y) / (x - y) = 24

-- Prove that the product is 48 given the conditions
theorem product_is_48 (x y : ℝ) (h : problem x y) : x * y = 48 :=
sorry

end product_is_48_l193_193362


namespace james_net_income_correct_l193_193805

def regular_price_per_hour : ℝ := 20
def discount_percent : ℝ := 0.10
def rental_hours_per_day_monday : ℝ := 8
def rental_hours_per_day_wednesday : ℝ := 8
def rental_hours_per_day_friday : ℝ := 6
def rental_hours_per_day_sunday : ℝ := 5
def sales_tax_percent : ℝ := 0.05
def car_maintenance_cost_per_week : ℝ := 35
def insurance_fee_per_day : ℝ := 15

-- Total rental hours
def total_rental_hours : ℝ :=
  rental_hours_per_day_monday + rental_hours_per_day_wednesday + rental_hours_per_day_friday + rental_hours_per_day_sunday

-- Total rental income before discount
def total_rental_income : ℝ := total_rental_hours * regular_price_per_hour

-- Discounted rental income
def discounted_rental_income : ℝ := total_rental_income * (1 - discount_percent)

-- Total income with tax
def total_income_with_tax : ℝ := discounted_rental_income * (1 + sales_tax_percent)

-- Total expenses
def total_expenses : ℝ := car_maintenance_cost_per_week + (insurance_fee_per_day * 4)

-- Net income
def net_income : ℝ := total_income_with_tax - total_expenses

theorem james_net_income_correct : net_income = 415.30 :=
  by
    -- proof omitted
    sorry

end james_net_income_correct_l193_193805


namespace inequality_cube_of_greater_l193_193290

variable {a b : ℝ}

theorem inequality_cube_of_greater (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a > b) : a^3 > b^3 :=
sorry

end inequality_cube_of_greater_l193_193290


namespace stable_set_even_subset_count_l193_193738

open Finset

-- Definitions
def is_stable (S : Finset (ℕ × ℕ)) : Prop :=
  ∀ ⦃x y⦄, (x, y) ∈ S → ∀ x' y', x' ≤ x → y' ≤ y → (x', y') ∈ S

-- Main statement
theorem stable_set_even_subset_count (S : Finset (ℕ × ℕ)) (hS : is_stable S):
  (∃ E O : ℕ, E ≥ O ∧ E + O = 2 ^ (S.card)) :=
  sorry

end stable_set_even_subset_count_l193_193738


namespace train_has_96_cars_l193_193414

def train_cars_count (cars_in_15_seconds : Nat) (time_for_15_seconds : Nat) (total_time_seconds : Nat) : Nat :=
  total_time_seconds * cars_in_15_seconds / time_for_15_seconds

theorem train_has_96_cars :
  train_cars_count 8 15 180 = 96 :=
by
  sorry

end train_has_96_cars_l193_193414


namespace sunny_lead_l193_193321

-- Define the given conditions as hypotheses
variables (h d : ℝ) (s w : ℝ)
    (H1 : ∀ t, t = 2 * h → (s * t) = 2 * h ∧ (w * t) = 2 * h - 2 * d)
    (H2 : ∀ t, (s * t) = 2 * h + 2 * d → (w * t) = 2 * h)

-- State the theorem we want to prove
theorem sunny_lead (h d : ℝ) (s w : ℝ) 
    (H1 : ∀ t, t = 2 * h → (s * t) = 2 * h ∧ (w * t) = 2 * h - 2 * d)
    (H2 : ∀ t, (s * t) = 2 * h + 2 * d → (w * t) = 2 * h) :
    ∃ distance_ahead_Sunny : ℝ, distance_ahead_Sunny = (2 * d^2) / h :=
sorry

end sunny_lead_l193_193321


namespace suitable_sampling_method_l193_193793

theorem suitable_sampling_method 
  (seniorTeachers : ℕ)
  (intermediateTeachers : ℕ)
  (juniorTeachers : ℕ)
  (totalSample : ℕ)
  (totalTeachers : ℕ)
  (prob : ℚ)
  (seniorSample : ℕ)
  (intermediateSample : ℕ)
  (juniorSample : ℕ)
  (excludeOneSenior : ℕ) :
  seniorTeachers = 28 →
  intermediateTeachers = 54 →
  juniorTeachers = 81 →
  totalSample = 36 →
  excludeOneSenior = 27 →
  totalTeachers = excludeOneSenior + intermediateTeachers + juniorTeachers →
  prob = totalSample / totalTeachers →
  seniorSample = excludeOneSenior * prob →
  intermediateSample = intermediateTeachers * prob →
  juniorSample = juniorTeachers * prob →
  seniorSample + intermediateSample + juniorSample = totalSample :=
by
  intros hsenior hins hjunior htotal hexclude htotalTeachers hprob hseniorSample hintermediateSample hjuniorSample
  sorry

end suitable_sampling_method_l193_193793


namespace train_crosses_platform_in_34_seconds_l193_193186

theorem train_crosses_platform_in_34_seconds 
    (train_speed_kmph : ℕ) 
    (time_cross_man_sec : ℕ) 
    (platform_length_m : ℕ) 
    (h_speed : train_speed_kmph = 72) 
    (h_time : time_cross_man_sec = 18) 
    (h_platform_length : platform_length_m = 320) 
    : (platform_length_m + (train_speed_kmph * 1000 / 3600) * time_cross_man_sec) / (train_speed_kmph * 1000 / 3600) = 34 :=
by
    sorry

end train_crosses_platform_in_34_seconds_l193_193186


namespace smallest_square_area_l193_193928

theorem smallest_square_area (n : ℕ) (h : ∃ m : ℕ, 14 * n = m ^ 2) : n = 14 :=
sorry

end smallest_square_area_l193_193928


namespace area_KLMQ_l193_193144

structure Rectangle :=
(length : ℝ)
(width : ℝ)

def JR := 2
def RQ := 3
def JL := 8

def JLMR : Rectangle := {length := JL, width := JR}
def JKQR : Rectangle := {length := RQ, width := JR}

def RM : ℝ := JL
def QM : ℝ := RM - RQ
def LM : ℝ := JR

def KLMQ : Rectangle := {length := QM, width := LM}

theorem area_KLMQ : KLMQ.length * KLMQ.width = 10 :=
by
  sorry

end area_KLMQ_l193_193144


namespace hens_count_l193_193544

theorem hens_count (H C : ℕ) (h1 : H + C = 48) (h2 : 2 * H + 4 * C = 136) : H = 28 :=
by
  sorry

end hens_count_l193_193544


namespace base_five_of_156_is_1111_l193_193968

def base_five_equivalent (n : ℕ) : ℕ := sorry

theorem base_five_of_156_is_1111 :
  base_five_equivalent 156 = 1111 :=
sorry

end base_five_of_156_is_1111_l193_193968


namespace factorization_correct_l193_193486

theorem factorization_correct (x : ℝ) :
    x^2 - 3 * x - 4 = (x + 1) * (x - 4) :=
  sorry

end factorization_correct_l193_193486


namespace range_of_k_if_intersection_empty_l193_193105

open Set

variable (k : ℝ)

def M : Set ℝ := {x | -1 < x ∧ x < 2}
def N (k : ℝ) : Set ℝ := {x | x ≤ k}

theorem range_of_k_if_intersection_empty (h : M ∩ N k = ∅) : k ≤ -1 :=
by {
  sorry
}

end range_of_k_if_intersection_empty_l193_193105


namespace hyperbola_asymptotes_slope_l193_193236

open Real

theorem hyperbola_asymptotes_slope (m : ℝ) : 
  (∀ x y : ℝ, (y ^ 2 / 16) - (x ^ 2 / 9) = 1 → (y = m * x ∨ y = -m * x)) → 
  m = 4 / 3 := 
by 
  sorry

end hyperbola_asymptotes_slope_l193_193236


namespace infinite_sum_equals_two_l193_193232

theorem infinite_sum_equals_two :
  ∑' k : ℕ, (8^k / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1)))) = 2 :=
sorry

end infinite_sum_equals_two_l193_193232


namespace distance_between_vertices_of_hyperbola_l193_193065

theorem distance_between_vertices_of_hyperbola :
  ∀ (x y : ℝ), 16 * x^2 - 32 * x - y^2 + 10 * y + 19 = 0 → 
  2 * Real.sqrt (7 / 4) = Real.sqrt 7 :=
by
  intros x y h
  sorry

end distance_between_vertices_of_hyperbola_l193_193065


namespace infinite_series_eval_l193_193177

open Filter
open Real
open Topology
open BigOperators

-- Define the relevant expression for the infinite sum
noncomputable def infinite_series_sum : ℝ :=
  ∑' n : ℕ, (n / (n^4 - 4 * n^2 + 8))

-- The theorem statement
theorem infinite_series_eval : infinite_series_sum = 5 / 24 :=
by sorry

end infinite_series_eval_l193_193177


namespace geometric_sequence_ratio_l193_193633

theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h1 : 6 * a 7 = (a 8 + a 9) / 2)
  (h2 : ∀ n : ℕ, a (n + 1) = a n * q)
  (h3 : ∀ n : ℕ, S n = a 1 * (1 - q^n) / (1 - q)) :
  S 6 / S 3 = 28 :=
by
  -- The proof goes here
  sorry

end geometric_sequence_ratio_l193_193633


namespace range_of_a_for_decreasing_exponential_l193_193840

theorem range_of_a_for_decreasing_exponential :
  ∀ (a : ℝ), (∀ (x1 x2 : ℝ), x1 < x2 → (2 - a)^x1 > (2 - a)^x2) ↔ (1 < a ∧ a < 2) :=
by
  sorry

end range_of_a_for_decreasing_exponential_l193_193840


namespace combined_distance_l193_193590

-- Definitions based on the conditions
def JulienDailyDistance := 50
def SarahDailyDistance := 2 * JulienDailyDistance
def JamirDailyDistance := SarahDailyDistance + 20
def Days := 7

-- Combined weekly distances
def JulienWeeklyDistance := JulienDailyDistance * Days
def SarahWeeklyDistance := SarahDailyDistance * Days
def JamirWeeklyDistance := JamirDailyDistance * Days

-- Theorem statement with the combined distance
theorem combined_distance :
  JulienWeeklyDistance + SarahWeeklyDistance + JamirWeeklyDistance = 1890 := by
  sorry

end combined_distance_l193_193590


namespace average_difference_l193_193950

theorem average_difference (t : ℚ) (ht : t = 4) :
  let m := (13 + 16 + 10 + 15 + 11) / 5
  let n := (16 + t + 3 + 13) / 4
  m - n = 4 :=
by
  sorry

end average_difference_l193_193950


namespace problem_1_l193_193392

theorem problem_1 (α : ℝ) (k : ℤ) (n : ℕ) (hk : k > 0) (hα : α ≠ k * Real.pi) (hn : n > 0) :
  n = 1 → (0.5 + Real.cos α) = (0.5 + Real.cos α) :=
by
  sorry

end problem_1_l193_193392


namespace triangle_sides_p2_triangle_sides_p3_triangle_sides_p4_triangle_sides_p5_l193_193799

-- Definition of the sides according to Plato's rule
def triangle_sides (p : ℕ) : ℕ × ℕ × ℕ :=
  (2 * p, p^2 - 1, p^2 + 1)

-- Function to check if the given sides form a right triangle
def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

-- Theorems to verify the sides of the triangle for given p values
theorem triangle_sides_p2 : triangle_sides 2 = (4, 3, 5) ∧ is_right_triangle 4 3 5 :=
by {
  sorry -- Proof goes here
}

theorem triangle_sides_p3 : triangle_sides 3 = (6, 8, 10) ∧ is_right_triangle 6 8 10 :=
by {
  sorry -- Proof goes here
}

theorem triangle_sides_p4 : triangle_sides 4 = (8, 15, 17) ∧ is_right_triangle 8 15 17 :=
by {
  sorry -- Proof goes here
}

theorem triangle_sides_p5 : triangle_sides 5 = (10, 24, 26) ∧ is_right_triangle 10 24 26 :=
by {
  sorry -- Proof goes here
}

end triangle_sides_p2_triangle_sides_p3_triangle_sides_p4_triangle_sides_p5_l193_193799


namespace no_real_a_l193_193969

noncomputable def A (a : ℝ) : Set ℝ := {x | x^2 - a * x + a^2 - 19 = 0}
def B : Set ℝ := {x | x^2 - 5 * x + 6 = 0}

theorem no_real_a (a : ℝ) : ¬ ((A a ≠ B) ∧ (A a ∪ B = B) ∧ (∅ ⊂ (A a ∩ B))) :=
by
  intro h
  sorry

end no_real_a_l193_193969


namespace gift_bags_needed_l193_193322

/-
  Constants
  total_expected: \(\mathbb{N}\) := 90        -- 50 people who will show up + 40 more who may show up
  total_prepared: \(\mathbb{N}\) := 30        -- 10 extravagant gift bags + 20 average gift bags

  The property to be proved:
  prove that (total_expected - total_prepared = 60)
-/

def total_expected : ℕ := 50 + 40
def total_prepared : ℕ := 10 + 20
def additional_needed := total_expected - total_prepared

theorem gift_bags_needed : additional_needed = 60 := by
  sorry

end gift_bags_needed_l193_193322


namespace inverse_function_coeff_ratio_l193_193467

noncomputable def f_inv_coeff_ratio : ℝ :=
  let f (x : ℝ) := (2 * x - 1) / (x + 5)
  let a := 5
  let b := 1
  let c := -1
  let d := 2
  a / c

theorem inverse_function_coeff_ratio :
  f_inv_coeff_ratio = -5 := 
by
  sorry

end inverse_function_coeff_ratio_l193_193467


namespace length_of_DG_l193_193476

theorem length_of_DG {AB BC DG DF : ℝ} (h1 : AB = 8) (h2 : BC = 10) (h3 : DG = DF) 
  (h4 : 1/5 * (AB * BC) = 1/2 * DG^2) : DG = 4 * Real.sqrt 2 :=
by sorry

end length_of_DG_l193_193476


namespace cost_price_is_1500_l193_193956

-- Definitions for the given conditions
def selling_price : ℝ := 1200
def loss_percentage : ℝ := 20

-- Define the cost price such that the loss percentage condition is satisfied
def cost_price (C : ℝ) : Prop :=
  loss_percentage = ((C - selling_price) / C) * 100

-- The proof problem to be solved: 
-- Prove that the cost price of the radio is Rs. 1500
theorem cost_price_is_1500 : ∃ C, cost_price C ∧ C = 1500 :=
by
  sorry

end cost_price_is_1500_l193_193956


namespace find_n_for_quadratic_roots_l193_193134

noncomputable def quadratic_root_properties (d c e n : ℝ) : Prop :=
  let A := (n + 2)
  let B := -((n + 2) * d + (n - 2) * c)
  let C := e * (n - 2)
  ∃ y1 y2 : ℝ, (A * y1 * y1 + B * y1 + C = 0) ∧ (A * y2 * y2 + B * y2 + C = 0) ∧ (y1 = -y2) ∧ (y1 + y2 = 0)

theorem find_n_for_quadratic_roots (d c e : ℝ) (h : d ≠ c) : 
  (quadratic_root_properties d c e (-2)) :=
sorry

end find_n_for_quadratic_roots_l193_193134


namespace find_integers_l193_193409

theorem find_integers (x : ℤ) (h₁ : x ≠ 3) (h₂ : (x - 3) ∣ (x ^ 3 - 3)) :
  x = -21 ∨ x = -9 ∨ x = -5 ∨ x = -3 ∨ x = -1 ∨ x = 1 ∨ x = 2 ∨ x = 4 ∨ x = 5 ∨
  x = 7 ∨ x = 9 ∨ x = 11 ∨ x = 15 ∨ x = 27 :=
sorry

end find_integers_l193_193409


namespace quadratic_unique_solution_pair_l193_193445

theorem quadratic_unique_solution_pair (a c : ℝ) (h₁ : a + c = 12) (h₂ : a < c) (h₃ : a * c = 9) :
  (a, c) = (6 - 3 * Real.sqrt 3, 6 + 3 * Real.sqrt 3) :=
by
  sorry

end quadratic_unique_solution_pair_l193_193445


namespace cos_C_l193_193530

-- Define the data and conditions of the problem
variables {A B C : ℝ}
variables (triangle_ABC : Prop)
variable (h_sinA : Real.sin A = 4 / 5)
variable (h_cosB : Real.cos B = 12 / 13)

-- Statement of the theorem
theorem cos_C (h1 : triangle_ABC)
  (h2 : Real.sin A = 4 / 5)
  (h3 : Real.cos B = 12 / 13) :
  Real.cos C = -16 / 65 :=
sorry

end cos_C_l193_193530


namespace real_part_of_z_is_neg3_l193_193557

noncomputable def z : ℂ := (1 + 2 * Complex.I) ^ 2

theorem real_part_of_z_is_neg3 : z.re = -3 := by
  sorry

end real_part_of_z_is_neg3_l193_193557


namespace combined_mpg_l193_193294

theorem combined_mpg (miles_alice : ℕ) (mpg_alice : ℕ) (miles_bob : ℕ) (mpg_bob : ℕ) :
  miles_alice = 120 ∧ mpg_alice = 30 ∧ miles_bob = 180 ∧ mpg_bob = 20 →
  (miles_alice + miles_bob) / ((miles_alice / mpg_alice) + (miles_bob / mpg_bob)) = 300 / 13 :=
by
  intros h
  sorry

end combined_mpg_l193_193294


namespace used_crayons_l193_193843

open Nat

theorem used_crayons (N B T U : ℕ) (h1 : N = 2) (h2 : B = 8) (h3 : T = 14) (h4 : T = N + U + B) : U = 4 :=
by
  -- Proceed with the proof here
  sorry

end used_crayons_l193_193843


namespace train_crosses_pole_in_1_5_seconds_l193_193993

noncomputable def time_to_cross_pole (length : ℝ) (speed_km_hr : ℝ) : ℝ :=
  let speed_m_s := speed_km_hr * (1000 / 3600)
  length / speed_m_s

theorem train_crosses_pole_in_1_5_seconds :
  time_to_cross_pole 60 144 = 1.5 :=
by
  unfold time_to_cross_pole
  -- simplified proof would be here
  sorry

end train_crosses_pole_in_1_5_seconds_l193_193993


namespace find_C_plus_D_l193_193892

theorem find_C_plus_D
  (C D : ℕ)
  (h1 : D = C + 2)
  (h2 : 2 * C^2 + 5 * C + 3 - (7 * D + 5) = (C + D)^2 + 6 * (C + D) + 8)
  (hC_pos : 0 < C)
  (hD_pos : 0 < D) :
  C + D = 26 := by
  sorry

end find_C_plus_D_l193_193892


namespace contribution_per_student_l193_193112

theorem contribution_per_student (total_contribution : ℝ) (class_funds : ℝ) (num_students : ℕ) 
(h1 : total_contribution = 90) (h2 : class_funds = 14) (h3 : num_students = 19) : 
  (total_contribution - class_funds) / num_students = 4 :=
by
  sorry

end contribution_per_student_l193_193112


namespace segments_in_proportion_l193_193379

theorem segments_in_proportion (a b c d : ℝ) (ha : a = 1) (hb : b = 4) (hc : c = 2) (h : a / b = c / d) : d = 8 := 
by 
  sorry

end segments_in_proportion_l193_193379


namespace count_two_digit_integers_with_perfect_square_sum_l193_193951

def valid_pairs : List (ℕ × ℕ) :=
[(2, 9), (3, 8), (4, 7), (5, 6), (6, 5), (7, 4), (8, 3), (9, 2)]

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

def reversed_sum_is_perfect_square (n : ℕ) : Prop :=
  ∃ t u, n = 10 * t + u ∧ t + u = 11

theorem count_two_digit_integers_with_perfect_square_sum :
  Nat.card { n : ℕ // is_two_digit n ∧ reversed_sum_is_perfect_square n } = 8 := 
sorry

end count_two_digit_integers_with_perfect_square_sum_l193_193951


namespace rectangular_prism_volume_l193_193046

variables (a b c : ℝ)

theorem rectangular_prism_volume
  (h1 : a * b = 24)
  (h2 : b * c = 8)
  (h3 : c * a = 3) :
  a * b * c = 24 :=
by
  sorry

end rectangular_prism_volume_l193_193046


namespace determine_a_l193_193778

theorem determine_a (a : ℝ) (f : ℝ → ℝ) (h : f = fun x => a * x^3 - 2 * x) (pt : f (-1) = 4) : a = -2 := by
  sorry

end determine_a_l193_193778


namespace units_digit_product_odd_integers_10_to_110_l193_193293

-- Define the set of odd integer numbers between 10 and 110
def oddNumbersInRange : List ℕ := List.filter (fun n => n % 2 = 1) (List.range' 10 101)

-- Define the set of relevant odd multiples of 5 within the range
def oddMultiplesOfFive : List ℕ := List.filter (fun n => n % 5 = 0) oddNumbersInRange

-- Prove that the product of all odd positive integers between 10 and 110 has units digit 5
theorem units_digit_product_odd_integers_10_to_110 :
  let product : ℕ := List.foldl (· * ·) 1 oddNumbersInRange
  product % 10 = 5 :=
by
  sorry

end units_digit_product_odd_integers_10_to_110_l193_193293


namespace ratio_equivalence_l193_193540

theorem ratio_equivalence (x : ℕ) : 
  (10 * 60 = 600) →
  (15 : ℕ) / 5 = x / 600 →
  x = 1800 :=
by
  intros h1 h2
  sorry

end ratio_equivalence_l193_193540


namespace solve_cyclist_return_speed_l193_193650

noncomputable def cyclist_return_speed (D : ℝ) (V : ℝ) : Prop :=
  let avg_speed := 9.5
  let out_speed := 10
  let T_out := D / out_speed
  let T_back := D / V
  2 * D / (T_out + T_back) = avg_speed

theorem solve_cyclist_return_speed : ∀ (D : ℝ), cyclist_return_speed D (20 / 2.1) :=
by
  intro D
  sorry

end solve_cyclist_return_speed_l193_193650


namespace max_square_test_plots_l193_193124

theorem max_square_test_plots
    (length : ℕ)
    (width : ℕ)
    (fence : ℕ)
    (fields_measure : length = 30 ∧ width = 45)
    (fence_measure : fence = 2250) :
  ∃ (number_of_plots : ℕ),
    number_of_plots = 150 :=
by
  sorry

end max_square_test_plots_l193_193124


namespace solve_y_l193_193715

theorem solve_y (x y : ℤ) (h₁ : x = 3) (h₂ : x^3 - x - 2 = y + 2) : y = 20 :=
by
  -- Proof goes here
  sorry

end solve_y_l193_193715


namespace total_unbroken_seashells_l193_193053

/-
Given:
On the first day, Tom found 7 seashells but 4 were broken.
On the second day, he found 12 seashells but 5 were broken.
On the third day, he found 15 seashells but 8 were broken.

We need to prove that Tom found 17 unbroken seashells in total over the three days.
-/

def first_day_total := 7
def first_day_broken := 4
def first_day_unbroken := first_day_total - first_day_broken

def second_day_total := 12
def second_day_broken := 5
def second_day_unbroken := second_day_total - second_day_broken

def third_day_total := 15
def third_day_broken := 8
def third_day_unbroken := third_day_total - third_day_broken

def total_unbroken := first_day_unbroken + second_day_unbroken + third_day_unbroken

theorem total_unbroken_seashells : total_unbroken = 17 := by
  sorry

end total_unbroken_seashells_l193_193053


namespace midpoint_product_l193_193992

theorem midpoint_product (x y : ℝ) (h1 : (4 : ℝ) = (x + 10) / 2) (h2 : (-2 : ℝ) = (-6 + y) / 2) : x * y = -4 := by
  sorry

end midpoint_product_l193_193992


namespace neg_exists_equiv_forall_l193_193788

theorem neg_exists_equiv_forall (p : Prop) :
  (¬ (∃ n : ℕ, n^2 > 2^n)) ↔ (∀ n : ℕ, n^2 ≤ 2^n) := sorry

end neg_exists_equiv_forall_l193_193788


namespace number_of_teams_l193_193259

theorem number_of_teams (n : ℕ) (h : n * (n - 1) / 2 = 28) : n = 8 :=
by
  sorry

end number_of_teams_l193_193259


namespace find_a_l193_193708

theorem find_a (a : ℝ) : (∃ p : ℝ × ℝ, p = (2 - a, a - 3) ∧ p.fst = 0) → a = 2 := by
  sorry

end find_a_l193_193708


namespace x_must_be_negative_l193_193181

theorem x_must_be_negative (x y : ℝ) (h1 : y ≠ 0) (h2 : y > 0) (h3 : x / y < -3) : x < 0 :=
by 
  sorry

end x_must_be_negative_l193_193181


namespace jayden_half_of_ernesto_in_some_years_l193_193714

theorem jayden_half_of_ernesto_in_some_years :
  ∃ x : ℕ, (4 + x = (1 : ℝ) / 2 * (11 + x)) ∧ x = 3 := by
  sorry

end jayden_half_of_ernesto_in_some_years_l193_193714


namespace first_group_work_done_l193_193913

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

end first_group_work_done_l193_193913


namespace ratio_volumes_equal_ratio_areas_l193_193772

-- Defining necessary variables and functions
variables (R : ℝ) (S_sphere S_cone V_sphere V_cone : ℝ)

-- Conditions
def surface_area_sphere : Prop := S_sphere = 4 * Real.pi * R^2
def volume_sphere : Prop := V_sphere = (4 / 3) * Real.pi * R^3
def volume_polyhedron : Prop := V_cone = (S_cone * R) / 3

-- Theorem statement
theorem ratio_volumes_equal_ratio_areas
  (h1 : surface_area_sphere R S_sphere)
  (h2 : volume_sphere R V_sphere)
  (h3 : volume_polyhedron R S_cone V_cone)
  : (V_sphere / V_cone) = (S_sphere / S_cone) :=
sorry

end ratio_volumes_equal_ratio_areas_l193_193772


namespace complement_U_A_l193_193856

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3}

theorem complement_U_A : U \ A = {2, 4, 5} :=
by
  sorry

end complement_U_A_l193_193856


namespace problem1_solution_problem2_solution_l193_193472

-- Problem 1
theorem problem1_solution (x y : ℝ) : (2 * x - y = 3) ∧ (x + y = 3) ↔ (x = 2 ∧ y = 1) := by
  sorry

-- Problem 2
theorem problem2_solution (x y : ℝ) : (x / 4 + y / 3 = 3) ∧ (3 * x - 2 * (y - 1) = 11) ↔ (x = 6 ∧ y = 9 / 2) := by
  sorry

end problem1_solution_problem2_solution_l193_193472


namespace train_speed_l193_193725

theorem train_speed
    (train_length : ℕ := 800)
    (tunnel_length : ℕ := 500)
    (time_minutes : ℕ := 1)
    : (train_length + tunnel_length) * (60 / time_minutes) / 1000 = 78 := by
  sorry

end train_speed_l193_193725


namespace greatest_k_for_quadratic_roots_diff_l193_193150

theorem greatest_k_for_quadratic_roots_diff (k : ℝ)
  (H : ∀ x: ℝ, (x^2 + k * x + 8 = 0) → (∃ a b : ℝ, a ≠ b ∧ (a - b)^2 = 84)) :
  k = 2 * Real.sqrt 29 :=
by
  sorry

end greatest_k_for_quadratic_roots_diff_l193_193150


namespace wholesale_cost_l193_193890

variable (W R P : ℝ)

-- Conditions
def retail_price := R = 1.20 * W
def employee_discount := P = 0.95 * R
def employee_payment := P = 228

-- Theorem statement
theorem wholesale_cost (H1 : retail_price R W) (H2 : employee_discount P R) (H3 : employee_payment P) : W = 200 :=
by
  sorry

end wholesale_cost_l193_193890


namespace math_problem_l193_193741

theorem math_problem
  (a b c d : ℕ)
  (h1 : a = 234)
  (h2 : b = 205)
  (h3 : c = 86400)
  (h4 : d = 300) :
  (a * b = 47970) ∧ (c / d = 288) :=
by
  sorry

end math_problem_l193_193741


namespace trapezium_area_l193_193452

theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 18) (hh : h = 13) :
  (1 / 2) * (a + b) * h = 247 :=
by
  sorry

end trapezium_area_l193_193452


namespace product_of_roots_eq_neg35_l193_193583

theorem product_of_roots_eq_neg35 (x : ℝ) : 
  (x + 3) * (x - 5) = 20 → ∃ a b c : ℝ, a ≠ 0 ∧ a * x^2 + b * x + c = 0 ∧ ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1 * x2 = -35 := 
by
  sorry

end product_of_roots_eq_neg35_l193_193583


namespace judgement_only_b_correct_l193_193687

theorem judgement_only_b_correct
  (A_expr : Int := 11 + (-14) + 19 - (-6))
  (A_computed : Int := 11 + 19 + ((-14) + (-6)))
  (A_result_incorrect : A_computed ≠ 10)
  (B_expr : ℚ := -2/3 - 1/5 + (-1/3))
  (B_computed : ℚ := (-2/3 + -1/3) + -1/5)
  (B_result_correct : B_computed = -6/5) :
  (A_computed ≠ 10 ∧ B_computed = -6/5) :=
by
  sorry

end judgement_only_b_correct_l193_193687


namespace range_of_m_l193_193881

theorem range_of_m (x y m : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2 * y - x * y = 0) :
    (∀ x y : ℝ, x > 0 → y > 0 → x + 2 * y - x * y = 0 → x + 2 * y > m^2 + 2 * m) ↔ (-4 : ℝ) < m ∧ m < 2 :=
by
  sorry

end range_of_m_l193_193881


namespace one_over_nine_inv_half_eq_three_l193_193901

theorem one_over_nine_inv_half_eq_three : (1 / 9 : ℝ) ^ (-1 / 2 : ℝ) = 3 := 
by
  sorry

end one_over_nine_inv_half_eq_three_l193_193901


namespace total_profit_is_35000_l193_193225

-- Definitions based on the conditions
variables (IB TB : ℝ) -- IB: Investment of B, TB: Time period of B's investment
def IB_times_TB := IB * TB
def IA := 3 * IB
def TA := 2 * TB
def profit_share_B := IB_times_TB
def profit_share_A := 6 * IB_times_TB
variable (profit_B : ℝ)
def profit_B_val := 5000

-- Ensure these definitions are used
def total_profit := profit_share_A + profit_share_B

-- Lean 4 statement showing that the total profit is Rs 35000
theorem total_profit_is_35000 : total_profit = 35000 := by
  sorry

end total_profit_is_35000_l193_193225


namespace simplify_fraction_l193_193115

-- We state the problem as a theorem.
theorem simplify_fraction : (3^2011 + 3^2011) / (3^2010 + 3^2012) = 3 / 5 := by sorry

end simplify_fraction_l193_193115


namespace jose_age_is_26_l193_193768

def Maria_age : ℕ := 14
def Jose_age (m : ℕ) : ℕ := m + 12

theorem jose_age_is_26 (m j : ℕ) (h1 : j = m + 12) (h2 : m + j = 40) : j = 26 :=
by
  sorry

end jose_age_is_26_l193_193768


namespace wall_width_l193_193023

theorem wall_width (brick_length brick_height brick_depth : ℝ)
    (wall_length wall_height : ℝ)
    (num_bricks : ℝ)
    (total_bricks_volume : ℝ)
    (total_wall_volume : ℝ) :
    brick_length = 25 →
    brick_height = 11.25 →
    brick_depth = 6 →
    wall_length = 800 →
    wall_height = 600 →
    num_bricks = 6400 →
    total_bricks_volume = num_bricks * (brick_length * brick_height * brick_depth) →
    total_wall_volume = wall_length * wall_height * (total_bricks_volume / (brick_length * brick_height * brick_depth)) →
    (total_bricks_volume / (wall_length * wall_height) = 22.5) :=
by
  intros
  sorry -- proof not required

end wall_width_l193_193023


namespace probability_no_consecutive_tails_probability_no_consecutive_tails_in_five_tosses_l193_193497

def countWays (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 2
  else countWays (n - 1) + countWays (n - 2)

theorem probability_no_consecutive_tails : countWays 5 = 13 :=
by
  sorry

theorem probability_no_consecutive_tails_in_five_tosses : 
  (countWays 5) / (2^5 : ℕ) = 13 / 32 :=
by
  sorry

end probability_no_consecutive_tails_probability_no_consecutive_tails_in_five_tosses_l193_193497


namespace ordered_sets_equal_l193_193335

theorem ordered_sets_equal
  (n : ℕ) 
  (h_gcd : gcd n 6 = 1) 
  (a b : ℕ → ℕ) 
  (h_order_a : ∀ {i j}, i < j → a i < a j)
  (h_order_b : ∀ {i j}, i < j → b i < b j) 
  (h_sum : ∀ {j k l : ℕ}, 1 ≤ j → j < k → k < l → l ≤ n → a j + a k + a l = b j + b k + b l) : 
  ∀ (j : ℕ), 1 ≤ j → j ≤ n → a j = b j := 
sorry

end ordered_sets_equal_l193_193335


namespace three_irreducible_fractions_prod_eq_one_l193_193224

-- Define the set of numbers available for use
def available_numbers : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define a structure for an irreducible fraction
structure irreducible_fraction :=
(num : ℕ)
(denom : ℕ)
(h_coprime : Nat.gcd num denom = 1)
(h_in_set : num ∈ available_numbers ∧ denom ∈ available_numbers)

-- Definition of the main proof problem
theorem three_irreducible_fractions_prod_eq_one :
  ∃ (f1 f2 f3 : irreducible_fraction), 
    f1.num * f2.num * f3.num = f1.denom * f2.denom * f3.denom ∧ 
    f1.num ≠ f2.num ∧ f1.num ≠ f3.num ∧ f2.num ≠ f3.num ∧ 
    f1.denom ≠ f2.denom ∧ f1.denom ≠ f3.denom ∧ f2.denom ≠ f3.denom := 
by
  sorry

end three_irreducible_fractions_prod_eq_one_l193_193224


namespace forty_percent_of_thirty_percent_l193_193625

theorem forty_percent_of_thirty_percent (x : ℝ) 
  (h : 0.3 * 0.4 * x = 48) : 0.4 * 0.3 * x = 48 :=
by
  sorry

end forty_percent_of_thirty_percent_l193_193625


namespace base_angle_of_isosceles_triangle_l193_193482

-- Definitions corresponding to the conditions
def isosceles_triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  (a = b ∧ A + B + C = 180) ∧ A = 40 -- Isosceles and sum of angles is 180° with apex angle A = 40°

-- The theorem to be proven
theorem base_angle_of_isosceles_triangle (a b c : ℝ) (A B C : ℝ) :
  isosceles_triangle a b c A B C → B = 70 :=
by
  intros h
  sorry

end base_angle_of_isosceles_triangle_l193_193482


namespace smallest_candies_value_l193_193462

def smallest_valid_n := ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ n % 9 = 2 ∧ n % 7 = 5 ∧ ∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ m % 9 = 2 ∧ m % 7 = 5 → n ≤ m

theorem smallest_candies_value : ∃ n : ℕ, smallest_valid_n ∧ n = 101 := 
by {
  sorry  
}

end smallest_candies_value_l193_193462


namespace math_problem_l193_193943

theorem math_problem (a b c : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) (h₄ : a + b + c = 0) :
  (a^2 * b^2 / ((a^2 - b * c) * (b^2 - a * c)) +
  a^2 * c^2 / ((a^2 - b * c) * (c^2 - a * b)) +
  b^2 * c^2 / ((b^2 - a * c) * (c^2 - a * b))) = 1 :=
by
  sorry

end math_problem_l193_193943


namespace smallest_n_such_that_floor_eq_1989_l193_193529

theorem smallest_n_such_that_floor_eq_1989 :
  ∃ (n : ℕ), (∀ k, k < n -> ¬(∃ x : ℤ, ⌊(10^k : ℚ) / x⌋ = 1989)) ∧ (∃ x : ℤ, ⌊(10^n : ℚ) / x⌋ = 1989) :=
sorry

end smallest_n_such_that_floor_eq_1989_l193_193529


namespace intersection_eq_l193_193666

def M : Set ℝ := { x | -2 ≤ x ∧ x ≤ 0 }
def N : Set ℝ := { -1, 0, 1 }

theorem intersection_eq : M ∩ N = { -1, 0 } := by
  sorry

end intersection_eq_l193_193666


namespace sum_of_ages_l193_193455

-- Definitions based on conditions
def age_relation1 (a b c : ℕ) : Prop := a = 20 + b + c
def age_relation2 (a b c : ℕ) : Prop := a^2 = 2000 + (b + c)^2

-- The statement to be proven
theorem sum_of_ages (a b c : ℕ) (h1 : age_relation1 a b c) (h2 : age_relation2 a b c) : a + b + c = 80 :=
by
  sorry

end sum_of_ages_l193_193455


namespace no_increasing_sequence_with_unique_sum_l193_193896

theorem no_increasing_sequence_with_unique_sum :
  ¬ (∃ (a : ℕ → ℕ), (∀ n, 0 < a n) ∧ (∀ n, a n < a (n + 1)) ∧ 
  (∀ N, ∃ k ≥ N, ∀ m ≥ k, 
    (∃! (i j : ℕ), a i + a j = m))) := sorry

end no_increasing_sequence_with_unique_sum_l193_193896


namespace max_volume_small_cube_l193_193025

theorem max_volume_small_cube (a : ℝ) (h : a = 2) : (a^3 = 8) := by
  sorry

end max_volume_small_cube_l193_193025


namespace javier_initial_games_l193_193320

/--
Javier plays 2 baseball games a week. In each of his first some games, 
he averaged 2 hits. If he has 10 games left, he has to average 5 hits 
a game to bring his average for the season up to 3 hits a game. 
Prove that the number of games Javier initially played is 20.
-/
theorem javier_initial_games (x : ℕ) :
  (2 * x + 5 * 10) / (x + 10) = 3 → x = 20 :=
by
  sorry

end javier_initial_games_l193_193320


namespace real_solutions_system_l193_193627

theorem real_solutions_system (x y z : ℝ) : 
  (x = 4 * z^2 / (1 + 4 * z^2) ∧ y = 4 * x^2 / (1 + 4 * x^2) ∧ z = 4 * y^2 / (1 + 4 * y^2)) ↔ 
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1/2 ∧ y = 1/2 ∧ z = 1/2) :=
by
  sorry

end real_solutions_system_l193_193627


namespace president_and_committee_combination_l193_193209

theorem president_and_committee_combination : 
  (∃ (n : ℕ), n = 10 * (Nat.choose 9 3)) := 
by
  use 840
  sorry

end president_and_committee_combination_l193_193209


namespace spherical_to_rectangular_correct_l193_193656

noncomputable def sphericalToRectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_correct :
  let ρ := 5
  let θ := Real.pi / 4
  let φ := Real.pi / 3
  sphericalToRectangular ρ θ φ = (5 * Real.sqrt 6 / 4, 5 * Real.sqrt 6 / 4, 5 / 2) :=
by
  sorry

end spherical_to_rectangular_correct_l193_193656


namespace nested_roots_identity_l193_193017

theorem nested_roots_identity (x : ℝ) (hx : x ≥ 0) : Real.sqrt (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x))) = x ^ (15 / 16) :=
sorry

end nested_roots_identity_l193_193017


namespace right_triangle_area_l193_193639

theorem right_triangle_area (a b c: ℝ) (h1: c = 2) (h2: a + b + c = 2 + Real.sqrt 6) (h3: (a * b) / 2 = 1 / 2) :
  (1 / 2) * (a * b) = 1 / 2 :=
by
  -- Sorry is used to skip the proof
  sorry

end right_triangle_area_l193_193639


namespace intersection_on_y_axis_l193_193072

theorem intersection_on_y_axis (k : ℝ) (x y : ℝ) :
  (2 * x + 3 * y - k = 0) →
  (x - k * y + 12 = 0) →
  (x = 0) →
  k = 6 ∨ k = -6 :=
by
  sorry

end intersection_on_y_axis_l193_193072


namespace john_exactly_three_green_marbles_l193_193723

-- Definitions based on the conditions
def total_marbles : ℕ := 15
def green_marbles : ℕ := 8
def purple_marbles : ℕ := 7
def trials : ℕ := 7
def green_prob : ℚ := 8 / 15
def purple_prob : ℚ := 7 / 15
def binom_coeff : ℕ := Nat.choose 7 3 

-- Theorem Statement
theorem john_exactly_three_green_marbles :
  (binom_coeff : ℚ) * (green_prob^3 * purple_prob^4) = 8604112 / 15946875 :=
by
  sorry

end john_exactly_three_green_marbles_l193_193723


namespace average_speed_l193_193176

def total_distance : ℝ := 200
def total_time : ℝ := 40

theorem average_speed (d t : ℝ) (h₁: d = total_distance) (h₂: t = total_time) : d / t = 5 :=
by
  rw [h₁, h₂]
  norm_num
  sorry

end average_speed_l193_193176


namespace value_of_a_b_c_l193_193205

theorem value_of_a_b_c (a b c : ℚ) (h₁ : |a| = 2) (h₂ : |b| = 2) (h₃ : |c| = 3) (h₄ : b < 0) (h₅ : 0 < a) :
  a + b + c = 3 ∨ a + b + c = -3 :=
by
  sorry

end value_of_a_b_c_l193_193205


namespace min_value_x_plus_2y_l193_193303

open Real

theorem min_value_x_plus_2y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 8 / x + 1 / y = 1) :
  x + 2 * y ≥ 16 :=
sorry

end min_value_x_plus_2y_l193_193303


namespace solve_system_l193_193996

theorem solve_system :
  {p : ℝ × ℝ | p.1^3 + p.2^3 = 19 ∧ p.1^2 + p.2^2 + 5 * p.1 + 5 * p.2 + p.1 * p.2 = 12} = {(3, -2), (-2, 3)} :=
sorry

end solve_system_l193_193996


namespace nth_equation_holds_l193_193257

theorem nth_equation_holds (n : ℕ) (h : 0 < n) :
  1 / (n + 2) + 2 / (n^2 + 2 * n) = 1 / n :=
by
  sorry

end nth_equation_holds_l193_193257


namespace paula_karl_age_sum_l193_193345

theorem paula_karl_age_sum :
  ∃ (P K : ℕ), (P - 5 = 3 * (K - 5)) ∧ (P + 6 = 2 * (K + 6)) ∧ (P + K = 54) :=
by
  sorry

end paula_karl_age_sum_l193_193345


namespace storybook_pages_l193_193160

def reading_start_date := 10
def reading_end_date := 20
def pages_per_day := 11
def number_of_days := reading_end_date - reading_start_date + 1
def total_pages := pages_per_day * number_of_days

theorem storybook_pages : total_pages = 121 := by
  sorry

end storybook_pages_l193_193160


namespace sam_more_than_avg_l193_193119

def bridget_count : ℕ := 14
def reginald_count : ℕ := bridget_count - 2
def sam_count : ℕ := reginald_count + 4
def average_count : ℕ := (bridget_count + reginald_count + sam_count) / 3

theorem sam_more_than_avg 
    (h1 : bridget_count = 14) 
    (h2 : reginald_count = bridget_count - 2) 
    (h3 : sam_count = reginald_count + 4) 
    (h4 : average_count = (bridget_count + reginald_count + sam_count) / 3): 
    sam_count - average_count = 2 := 
  sorry

end sam_more_than_avg_l193_193119


namespace a_9_equals_18_l193_193991

def is_sequence_of_positive_integers (a : ℕ → ℕ) : Prop :=
∀ n : ℕ, 0 < n → 0 < a n

def satisfies_recursive_relation (a : ℕ → ℕ) : Prop :=
∀ p q : ℕ, 0 < p → 0 < q → a (p + q) = a p + a q

theorem a_9_equals_18 (a : ℕ → ℕ)
  (H1 : is_sequence_of_positive_integers a)
  (H2 : satisfies_recursive_relation a)
  (H3 : a 2 = 4) : a 9 = 18 :=
sorry

end a_9_equals_18_l193_193991


namespace construction_company_sand_weight_l193_193787

theorem construction_company_sand_weight :
  let gravel_weight := 5.91
  let total_material_weight := 14.02
  let sand_weight := total_material_weight - gravel_weight
  sand_weight = 8.11 :=
by
  let gravel_weight := 5.91
  let total_material_weight := 14.02
  let sand_weight := total_material_weight - gravel_weight
  -- Observing that 14.02 - 5.91 = 8.11
  have h : sand_weight = 8.11 := by sorry
  exact h

end construction_company_sand_weight_l193_193787


namespace fraction_given_to_emma_is_7_over_36_l193_193705

-- Define initial quantities
def stickers_noah : ℕ := sorry
def stickers_emma : ℕ := 3 * stickers_noah
def stickers_liam : ℕ := 12 * stickers_noah

-- Define required number of stickers for equal distribution
def total_stickers := stickers_noah + stickers_emma + stickers_liam
def equal_stickers := total_stickers / 3

-- Define the number of stickers to be given to Emma and the fraction of Liam's stickers he should give to Emma
def stickers_given_to_emma := equal_stickers - stickers_emma
def fraction_liams_stickers_given_to_emma := stickers_given_to_emma / stickers_liam

-- Theorem statement
theorem fraction_given_to_emma_is_7_over_36 :
  fraction_liams_stickers_given_to_emma = 7 / 36 :=
sorry

end fraction_given_to_emma_is_7_over_36_l193_193705


namespace johns_original_earnings_l193_193269

-- Definitions from conditions
variables (x : ℝ) (raise_percentage : ℝ) (new_salary : ℝ)

-- Conditions
def conditions : Prop :=
  raise_percentage = 0.25 ∧ new_salary = 75 ∧ x + raise_percentage * x = new_salary

-- Theorem statement
theorem johns_original_earnings (h : conditions x 0.25 75) : x = 60 :=
sorry

end johns_original_earnings_l193_193269


namespace g_of_f_neg_5_l193_193193

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 - 8

-- Assume g(42) = 17
axiom g_f_5_eq_17 : ∀ (g : ℝ → ℝ), g (f 5) = 17

-- State the theorem to be proven
theorem g_of_f_neg_5 (g : ℝ → ℝ) : g (f (-5)) = 17 :=
by
  sorry

end g_of_f_neg_5_l193_193193


namespace decrypt_probability_l193_193690

theorem decrypt_probability (p1 p2 p3 : ℚ) (h1 : p1 = 1/5) (h2 : p2 = 2/5) (h3 : p3 = 1/2) : 
  1 - ((1 - p1) * (1 - p2) * (1 - p3)) = 19/25 :=
by
  sorry

end decrypt_probability_l193_193690


namespace max_f_eq_find_a_l193_193143

open Real

noncomputable def f (α : ℝ) : ℝ :=
  let a := (sin α, cos α)
  let b := (6 * sin α + cos α, 7 * sin α - 2 * cos α)
  a.1 * b.1 + a.2 * b.2

theorem max_f_eq : 
  ∃ α : ℝ, f α = 4 * sqrt 2 + 2 :=
sorry

structure Triangle :=
  (A B C : ℝ)
  (a b c : ℝ)
  (sin_A : ℝ)

noncomputable def f_triangle (A : ℝ) : ℝ :=
  let a := (sin A, cos A)
  let b := (6 * sin A + cos A, 7 * sin A - 2 * cos A)
  a.1 * b.1 + a.2 * b.2

axiom f_A_eq (A : ℝ) : f_triangle A = 6

theorem find_a (A B C a b c : ℝ) (h₁ : f_triangle A = 6) (h₂ : 1 / 2 * b * c * sin A = 3) (h₃ : b + c = 2 + 3 * sqrt 2) :
  a = sqrt 10 :=
sorry

end max_f_eq_find_a_l193_193143


namespace bowls_per_minute_l193_193830

def ounces_per_bowl : ℕ := 10
def gallons_of_soup : ℕ := 6
def serving_time_minutes : ℕ := 15
def ounces_per_gallon : ℕ := 128

theorem bowls_per_minute :
  (gallons_of_soup * ounces_per_gallon / servings_time_minutes) / ounces_per_bowl = 5 :=
by
  sorry

end bowls_per_minute_l193_193830


namespace josh_marbles_l193_193163

theorem josh_marbles (original_marble : ℝ) (given_marble : ℝ)
  (h1 : original_marble = 22.5) (h2 : given_marble = 20.75) :
  original_marble + given_marble = 43.25 := by
  sorry

end josh_marbles_l193_193163


namespace correct_answer_l193_193371

def sum_squares_of_three_consecutive_even_integers (n : ℤ) : ℤ :=
  let a := 2 * n
  let b := 2 * n + 2
  let c := 2 * n + 4
  a * a + b * b + c * c

def T : Set ℤ :=
  {t | ∃ n : ℤ, t = sum_squares_of_three_consecutive_even_integers n}

theorem correct_answer : (∀ t ∈ T, t % 4 = 0) ∧ (∀ t ∈ T, t % 7 ≠ 0) :=
sorry

end correct_answer_l193_193371


namespace none_of_the_choices_sum_of_150_consecutive_integers_l193_193691

theorem none_of_the_choices_sum_of_150_consecutive_integers :
  ¬(∃ k : ℕ, 678900 = 150 * k + 11325) ∧
  ¬(∃ k : ℕ, 1136850 = 150 * k + 11325) ∧
  ¬(∃ k : ℕ, 1000000 = 150 * k + 11325) ∧
  ¬(∃ k : ℕ, 2251200 = 150 * k + 11325) ∧
  ¬(∃ k : ℕ, 1876800 = 150 * k + 11325) :=
by
  sorry

end none_of_the_choices_sum_of_150_consecutive_integers_l193_193691


namespace average_weight_of_all_children_l193_193932

theorem average_weight_of_all_children 
  (Boys: ℕ) (Girls: ℕ) (Additional: ℕ)
  (avgWeightBoys: ℚ) (avgWeightGirls: ℚ) (avgWeightAdditional: ℚ) :
  Boys = 8 ∧ Girls = 5 ∧ Additional = 3 ∧ 
  avgWeightBoys = 160 ∧ avgWeightGirls = 130 ∧ avgWeightAdditional = 145 →
  ((Boys * avgWeightBoys + Girls * avgWeightGirls + Additional * avgWeightAdditional) / (Boys + Girls + Additional) = 148) :=
by
  intros
  sorry

end average_weight_of_all_children_l193_193932


namespace dig_time_comparison_l193_193599

open Nat

theorem dig_time_comparison :
  (3 * 420 / 9) - (5 * 40 / 2) = 40 :=
by
  sorry

end dig_time_comparison_l193_193599


namespace power_mod_result_l193_193245

theorem power_mod_result :
  (47 ^ 1235 - 22 ^ 1235) % 8 = 7 := by
  sorry

end power_mod_result_l193_193245


namespace exactly_two_statements_true_l193_193194

noncomputable def f : ℝ → ℝ := sorry -- Definition of f satisfying the conditions

-- Conditions
axiom functional_eq (x : ℝ) : f (x + 3/2) + f x = 0
axiom odd_function (x : ℝ) : f (- x - 3/4) = - f (x - 3/4)

-- Proof statement
theorem exactly_two_statements_true : 
  (¬(∀ (T : ℝ), T > 0 → (∀ (x : ℝ), f (x + T) = f x) → T = 3/2) ∧
   (∀ (x : ℝ), f (-x - 3/4) = - f (x - 3/4)) ∧
   (¬(∀ (x : ℝ), f x = f (-x)))) :=
sorry

end exactly_two_statements_true_l193_193194


namespace inequality_proof_l193_193214

theorem inequality_proof (a b : ℝ) (h1 : a < 1) (h2 : b < 1) (h3 : a + b ≥ 0.5) :
  (1 - a) * (1 - b) ≤ 9 / 16 :=
sorry

end inequality_proof_l193_193214


namespace price_reduction_correct_l193_193931

noncomputable def percentage_reduction (x : ℝ) : Prop :=
  (5000 * (1 - x)^2 = 4050)

theorem price_reduction_correct {x : ℝ} (h : percentage_reduction x) : x = 0.1 :=
by
  -- proof is omitted, so we use sorry
  sorry

end price_reduction_correct_l193_193931


namespace find_original_height_l193_193970

noncomputable def original_height : ℝ := by
  let H := 102.19
  sorry

lemma ball_rebound (H : ℝ) : 
  (H + 2 * 0.8 * H + 2 * 0.56 * H + 2 * 0.336 * H + 2 * 0.168 * H + 2 * 0.0672 * H + 2 * 0.02016 * H = 500) :=
by
  sorry

theorem find_original_height : original_height = 102.19 :=
by
  have h := ball_rebound original_height
  sorry

end find_original_height_l193_193970


namespace cara_younger_than_mom_l193_193661

noncomputable def cara_grandmothers_age : ℤ := 75
noncomputable def cara_moms_age := cara_grandmothers_age - 15
noncomputable def cara_age : ℤ := 40

theorem cara_younger_than_mom :
  cara_moms_age - cara_age = 20 := by
  sorry

end cara_younger_than_mom_l193_193661


namespace number_of_pictures_in_first_coloring_book_l193_193908

-- Define the conditions
variable (X : ℕ)
variable (total_pictures_colored : ℕ := 44)
variable (pictures_left : ℕ := 11)
variable (pictures_in_second_coloring_book : ℕ := 32)
variable (total_pictures : ℕ := total_pictures_colored + pictures_left)

-- The theorem statement
theorem number_of_pictures_in_first_coloring_book :
  X + pictures_in_second_coloring_book = total_pictures → X = 23 :=
by
  intro h
  sorry

end number_of_pictures_in_first_coloring_book_l193_193908


namespace find_x_y_l193_193496

theorem find_x_y (x y : ℝ) (h : (2 * x - 3 * y + 5) ^ 2 + |x - y + 2| = 0) : x = -1 ∧ y = 1 :=
by
  sorry

end find_x_y_l193_193496


namespace pencil_cost_l193_193167

theorem pencil_cost (p e : ℝ) (h1 : p + e = 3.40) (h2 : p = 3 + e) : p = 3.20 :=
by
  sorry

end pencil_cost_l193_193167


namespace min_value_1_a_plus_2_b_l193_193683

open Real

theorem min_value_1_a_plus_2_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) :
  (∀ a b, 0 < a → 0 < b → a + b = 1 → 3 + 2 * sqrt 2 ≤ 1 / a + 2 / b) := sorry

end min_value_1_a_plus_2_b_l193_193683


namespace distinct_arithmetic_progression_roots_l193_193127

theorem distinct_arithmetic_progression_roots (a b : ℝ) : 
  (∃ (d : ℝ), d ≠ 0 ∧ ∀ x, x^3 + a * x + b = 0 ↔ x = -d ∨ x = 0 ∨ x = d) → a < 0 ∧ b = 0 :=
by
  sorry

end distinct_arithmetic_progression_roots_l193_193127


namespace total_votes_cast_l193_193323

theorem total_votes_cast (V : ℕ) (h1 : V > 0) (h2 : ∃ c r : ℕ, c = 40 * V / 100 ∧ r = 40 * V / 100 + 5000 ∧ c + r = V):
  V = 25000 :=
by
  sorry

end total_votes_cast_l193_193323


namespace fractions_lcm_l193_193267

noncomputable def lcm_of_fractions_lcm (numerators : List ℕ) (denominators : List ℕ) : ℕ :=
  let lcm_nums := numerators.foldr Nat.lcm 1
  let gcd_denom := denominators.foldr Nat.gcd (denominators.headD 1)
  lcm_nums / gcd_denom

theorem fractions_lcm (hnum : List ℕ := [4, 5, 7, 9, 13, 16, 19])
                      (hdenom : List ℕ := [9, 7, 15, 13, 21, 35, 45]) :
  lcm_of_fractions_lcm hnum hdenom = 1244880 :=
by
  sorry

end fractions_lcm_l193_193267


namespace problem1_1_problem1_2_problem2_l193_193824

open Set

/-
Given sets U, A, and B, derived from the provided conditions:
  U : Set ℝ
  A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 5}
  B (m : ℝ) : Set ℝ := {x | x < 2 * m - 3}
-/

def U : Set ℝ := univ
def A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | x < 2 * m - 3}

theorem problem1_1 (m : ℝ) (h : m = 5) : A ∩ B m = {x | -3 ≤ x ∧ x ≤ 5} :=
sorry

theorem problem1_2 (m : ℝ) (h : m = 5) : (compl A) ∪ B m = univ :=
sorry

theorem problem2 (m : ℝ) : A ⊆ B m → 4 < m :=
sorry

end problem1_1_problem1_2_problem2_l193_193824


namespace ratio_6_3_to_percent_l193_193581

theorem ratio_6_3_to_percent : (6 / 3) * 100 = 200 := by
  sorry

end ratio_6_3_to_percent_l193_193581


namespace relationship_abc_l193_193659

noncomputable def a : ℝ := (0.7 : ℝ) ^ (0.6 : ℝ)
noncomputable def b : ℝ := (0.6 : ℝ) ^ (-0.6 : ℝ)
noncomputable def c : ℝ := (0.6 : ℝ) ^ (0.7 : ℝ)

theorem relationship_abc : b > a ∧ a > c :=
by
  -- Proof will go here
  sorry

end relationship_abc_l193_193659


namespace largest_k_l193_193906

def S : Set ℕ := {x | x > 0 ∧ x ≤ 100}

def satisfies_property (A B : Set ℕ) : Prop :=
  ∃ x ∈ A ∩ B, ∀ y ∈ A ∪ B, x ≠ y

theorem largest_k (k : ℕ) : 
  (∃ subsets : Finset (Set ℕ), 
    (subsets.card = k) ∧ 
    (∀ {A B : Set ℕ}, A ∈ subsets ∧ B ∈ subsets ∧ A ≠ B → 
      ¬(A ∩ B = ∅) ∧ satisfies_property A B)) →
  k ≤ 2^99 - 1 := sorry

end largest_k_l193_193906


namespace further_flight_Gaeun_l193_193813

theorem further_flight_Gaeun :
  let nana_distance_m := 1.618
  let gaeun_distance_cm := 162.3
  let conversion_factor := 100
  let nana_distance_cm := nana_distance_m * conversion_factor
  gaeun_distance_cm > nana_distance_cm := 
  sorry

end further_flight_Gaeun_l193_193813


namespace number_smaller_than_neg3_exists_l193_193479

def numbers := [0, -1, -5, -1/2]

theorem number_smaller_than_neg3_exists : ∃ x ∈ numbers, x < -3 :=
by
  let x := -5
  have h : x ∈ numbers := by simp [numbers]
  have h_lt : x < -3 := by norm_num
  exact ⟨x, h, h_lt⟩ -- show that -5 meets the criteria

end number_smaller_than_neg3_exists_l193_193479


namespace students_in_photo_l193_193995

theorem students_in_photo (m n : ℕ) (h1 : n = m + 5) (h2 : n = m + 5 ∧ m = 3) : 
  m * n = 24 :=
by
  -- h1: n = m + 5    (new row is 4 students fewer)
  -- h2: m = 3        (all rows have the same number of students after rearrangement)
  -- Prove m * n = 24
  sorry

end students_in_photo_l193_193995


namespace complement_union_in_universe_l193_193918

variable (U : Set ℕ := {1, 2, 3, 4, 5})
variable (M : Set ℕ := {1, 3})
variable (N : Set ℕ := {1, 2})

theorem complement_union_in_universe :
  (U \ (M ∪ N)) = {4, 5} :=
by
  sorry

end complement_union_in_universe_l193_193918


namespace proof1_proof2_l193_193684

-- Definitions based on the conditions given in the problem description.

def f1 (x : ℝ) (a : ℝ) : ℝ := abs (3 * x - 1) + a * x + 3

theorem proof1 (x : ℝ) : (f1 x 1 ≤ 4) ↔ 0 ≤ x ∧ x ≤ 1 / 2 := 
by
  sorry

theorem proof2 (a : ℝ) : (-3 ≤ a ∧ a ≤ 3) ↔ 
  ∃ (x : ℝ), ∀ y : ℝ, f1 x a ≤ f1 y a := 
by
  sorry

end proof1_proof2_l193_193684


namespace line_through_points_a_plus_b_l193_193438

theorem line_through_points_a_plus_b :
  ∃ a b : ℝ, (∀ x y : ℝ, (y = a * x + b) → ((x, y) = (6, 7)) ∨ ((x, y) = (10, 23))) ∧ (a + b = -13) :=
sorry

end line_through_points_a_plus_b_l193_193438


namespace remainder_of_sum_of_integers_l193_193260

theorem remainder_of_sum_of_integers (a b c : ℕ)
  (h₁ : a % 30 = 15) (h₂ : b % 30 = 5) (h₃ : c % 30 = 10) :
  (a + b + c) % 30 = 0 := by
  sorry

end remainder_of_sum_of_integers_l193_193260


namespace find_interest_rate_l193_193156

def interest_rate_borrowed (p_borrowed: ℝ) (p_lent: ℝ) (time: ℝ) (rate_lent: ℝ) (gain: ℝ) (r: ℝ) : Prop :=
  let interest_from_ramu := p_lent * rate_lent * time / 100
  let interest_to_anwar := p_borrowed * r * time / 100
  gain = interest_from_ramu - interest_to_anwar

theorem find_interest_rate :
  interest_rate_borrowed 3900 5655 3 9 824.85 5.95 := sorry

end find_interest_rate_l193_193156


namespace hypotenuse_square_l193_193541

-- Define the right triangle property and the consecutive integer property
variables (a b c : ℤ)

-- Noncomputable definition will be used as we are proving a property related to integers
noncomputable def consecutive_integers (a b : ℤ) : Prop := b = a + 1

-- Define the statement to prove
theorem hypotenuse_square (h_consec : consecutive_integers a b) (h_right_triangle : a * a + b * b = c * c) : 
  c * c = 2 * a * a + 2 * a + 1 :=
by {
  -- We only need to state the theorem
  sorry
}

end hypotenuse_square_l193_193541


namespace friend_decks_l193_193664

-- Definitions for conditions
def price_per_deck : ℕ := 8
def victor_decks : ℕ := 6
def total_spent : ℕ := 64

-- Conclusion based on the conditions
theorem friend_decks : (64 - (6 * 8)) / 8 = 2 := by
  sorry

end friend_decks_l193_193664


namespace zilla_savings_l193_193266

theorem zilla_savings
  (monthly_earnings : ℝ)
  (h_rent : monthly_earnings * 0.07 = 133)
  (h_expenses : monthly_earnings * 0.5 = monthly_earnings / 2) :
  monthly_earnings - (133 + monthly_earnings / 2) = 817 :=
by
  sorry

end zilla_savings_l193_193266


namespace age_ratio_l193_193458
open Nat

theorem age_ratio (B A x : ℕ) (h1 : B - 4 = 2 * (A - 4)) 
                                (h2 : B - 8 = 3 * (A - 8)) 
                                (h3 : (B + x) / (A + x) = 3 / 2) : 
                                x = 4 :=
by
  sorry

end age_ratio_l193_193458


namespace books_not_sold_l193_193376

theorem books_not_sold (X : ℕ) (H1 : (2/3 : ℝ) * X * 4 = 288) : (1 / 3 : ℝ) * X = 36 :=
by
  -- Proof goes here
  sorry

end books_not_sold_l193_193376


namespace remove_candies_even_distribution_l193_193927

theorem remove_candies_even_distribution (candies friends : ℕ) (h_candies : candies = 30) (h_friends : friends = 4) :
  ∃ k, candies - k % friends = 0 ∧ k = 2 :=
by
  sorry

end remove_candies_even_distribution_l193_193927


namespace surface_area_of_large_cube_correct_l193_193301

-- Definition of the surface area problem

def edge_length_of_small_cube := 3 -- centimeters
def number_of_small_cubes := 27
def surface_area_of_large_cube (edge_length_of_small_cube : ℕ) (number_of_small_cubes : ℕ) : ℕ :=
  let edge_length_of_large_cube := edge_length_of_small_cube * (number_of_small_cubes^(1/3))
  6 * edge_length_of_large_cube^2

theorem surface_area_of_large_cube_correct :
  surface_area_of_large_cube edge_length_of_small_cube number_of_small_cubes = 486 := by
  sorry

end surface_area_of_large_cube_correct_l193_193301


namespace priya_trip_time_l193_193755

noncomputable def time_to_drive_from_X_to_Z_at_50_mph : ℝ := 5

theorem priya_trip_time :
  (∀ (distance_YZ distance_XZ : ℝ), 
    distance_YZ = 60 * 2.0833333333333335 ∧
    distance_XZ = distance_YZ * 2 →
    time_to_drive_from_X_to_Z_at_50_mph = distance_XZ / 50 ) :=
sorry

end priya_trip_time_l193_193755


namespace jordan_probability_l193_193547

-- Definitions based on conditions.
def total_students := 28
def enrolled_in_french := 20
def enrolled_in_spanish := 23
def enrolled_in_both := 17

-- Calculate students enrolled only in one language.
def only_french := enrolled_in_french - enrolled_in_both
def only_spanish := enrolled_in_spanish - enrolled_in_both

-- Calculation of combinations.
def total_combinations := Nat.choose total_students 2
def only_french_combinations := Nat.choose only_french 2
def only_spanish_combinations := Nat.choose only_spanish 2

-- Probability calculations.
def prob_both_one_language := (only_french_combinations + only_spanish_combinations) / total_combinations

def prob_both_languages : ℚ := 1 - prob_both_one_language

theorem jordan_probability :
  prob_both_languages = (20 : ℚ) / 21 := by
  sorry

end jordan_probability_l193_193547


namespace q_investment_correct_l193_193111

-- Define the conditions
def profit_ratio := (4, 6)
def p_investment := 60000
def expected_q_investment := 90000

-- Define the theorem statement
theorem q_investment_correct (p_investment: ℕ) (q_investment: ℕ) (profit_ratio : ℕ × ℕ)
  (h_ratio: profit_ratio = (4, 6)) (hp_investment: p_investment = 60000) :
  q_investment = 90000 := by
  sorry

end q_investment_correct_l193_193111


namespace greatest_whole_number_satisfying_inequality_l193_193660

theorem greatest_whole_number_satisfying_inequality :
  ∀ (x : ℤ), 3 * x + 2 < 5 - 2 * x → x <= 0 :=
by
  sorry

end greatest_whole_number_satisfying_inequality_l193_193660


namespace polygon_sides_l193_193451

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 900) : n = 7 :=
by
  sorry

end polygon_sides_l193_193451


namespace max_three_digit_sum_l193_193400

theorem max_three_digit_sum (A B C : ℕ) (hA : A < 10) (hB : B < 10) (hC : C < 10) (h1 : A ≠ B) (h2 : B ≠ C) (h3 : C ≠ A) :
  110 * A + 10 * B + 3 * C ≤ 981 :=
sorry

end max_three_digit_sum_l193_193400


namespace lcm_ac_is_420_l193_193340

theorem lcm_ac_is_420 (a b c : ℕ) (h1 : Nat.lcm a b = 20) (h2 : Nat.lcm b c = 21) :
    Nat.lcm a c = 420 :=
sorry

end lcm_ac_is_420_l193_193340


namespace unique_real_root_t_l193_193653

theorem unique_real_root_t (t : ℝ) :
  (∃ x : ℝ, 3 * x + 7 * t - 2 + (2 * t * x^2 + 7 * t^2 - 9) / (x - t) = 0 ∧ 
  ∀ y : ℝ, 3 * y + 7 * t - 2 + (2 * t * y^2 + 7 * t^2 - 9) / (y - t) = 0 ∧ x ≠ y → false) →
  t = -3 ∨ t = -7 / 2 ∨ t = 1 :=
by
  sorry

end unique_real_root_t_l193_193653


namespace bobby_initial_pieces_l193_193682

-- Definitions based on the conditions
def pieces_eaten_1 := 17
def pieces_eaten_2 := 15
def pieces_left := 4

-- Definition based on the question and answer
def initial_pieces (pieces_eaten_1 pieces_eaten_2 pieces_left : ℕ) : ℕ :=
  pieces_eaten_1 + pieces_eaten_2 + pieces_left

-- Theorem stating the problem and the expected answer
theorem bobby_initial_pieces : 
  initial_pieces pieces_eaten_1 pieces_eaten_2 pieces_left = 36 :=
by 
  sorry

end bobby_initial_pieces_l193_193682


namespace intersection_A_B_union_A_B_complement_intersection_A_B_l193_193442

def A : Set ℝ := { x | 2 ≤ x ∧ x ≤ 8 }
def B : Set ℝ := { x | 1 < x ∧ x < 6 }
def A_inter_B : Set ℝ := { x | 2 ≤ x ∧ x < 6 }
def A_union_B : Set ℝ := { x | 1 < x ∧ x ≤ 8 }
def A_compl_inter_B : Set ℝ := { x | 1 < x ∧ x < 2 }

theorem intersection_A_B :
  A ∩ B = A_inter_B := by
  sorry

theorem union_A_B :
  A ∪ B = A_union_B := by
  sorry

theorem complement_intersection_A_B :
  (Aᶜ ∩ B) = A_compl_inter_B := by
  sorry

end intersection_A_B_union_A_B_complement_intersection_A_B_l193_193442


namespace Megan_not_lead_plays_l193_193237

-- Define the problem's conditions as variables
def total_plays : ℕ := 100
def lead_play_ratio : ℤ := 80

-- Define the proposition we want to prove
theorem Megan_not_lead_plays : 
  (total_plays - (total_plays * lead_play_ratio / 100)) = 20 := 
by sorry

end Megan_not_lead_plays_l193_193237


namespace find_n_l193_193679

-- Definitions based on conditions
variables (x n y : ℕ)
variable (h1 : x / n = 3 / 2)
variable (h2 : (7 * x + n * y) / (x - n * y) = 23)

-- Proof that n is equivalent to 1 given the conditions.
theorem find_n : n = 1 :=
sorry

end find_n_l193_193679


namespace quadratic_root_a_value_l193_193872

theorem quadratic_root_a_value (a : ℝ) (h : 2^2 - 2 * a + 6 = 0) : a = 5 :=
sorry

end quadratic_root_a_value_l193_193872


namespace turtle_finishes_in_10_minutes_l193_193769

def skunk_time : ℕ := 6
def rabbit_speed_ratio : ℕ := 3
def turtle_speed_ratio : ℕ := 5
def rabbit_time := skunk_time / rabbit_speed_ratio
def turtle_time := turtle_speed_ratio * rabbit_time

theorem turtle_finishes_in_10_minutes : turtle_time = 10 := by
  sorry

end turtle_finishes_in_10_minutes_l193_193769


namespace smallest_sector_angle_l193_193842

theorem smallest_sector_angle 
  (n : ℕ) (a1 : ℕ) (d : ℕ)
  (h1 : n = 18)
  (h2 : 360 = n * ((2 * a1 + (n - 1) * d) / 2))
  (h3 : ∀ i, 0 < i ∧ i ≤ 18 → ∃ k, 360 / 18 * k = i) :
  a1 = 3 :=
by sorry

end smallest_sector_angle_l193_193842


namespace no_perfect_squares_xy_zt_l193_193632

theorem no_perfect_squares_xy_zt
    (x y z t : ℕ) 
    (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : 0 < t)
    (h_eq1 : x + y = z + t) 
    (h_eq2 : xy - zt = x + y) : ¬(∃ a b : ℕ, xy = a^2 ∧ zt = b^2) :=
by
  sorry

end no_perfect_squares_xy_zt_l193_193632


namespace fraction_decomposition_l193_193016

theorem fraction_decomposition (A B : ℚ) :
  (∀ x : ℚ, x ≠ -1 ∧ x ≠ 2  →
    7 * x - 18 = A * (3 * x + 1) + B * (x - 2))
  ↔ (A = -4 / 7 ∧ B = 61 / 7) :=
by
  sorry

end fraction_decomposition_l193_193016


namespace train_boxcar_capacity_l193_193821

theorem train_boxcar_capacity :
  let red_boxcars := 3
  let blue_boxcars := 4
  let black_boxcars := 7
  let black_boxcar_capacity := 4000
  let blue_boxcar_capacity := 2 * black_boxcar_capacity
  let red_boxcar_capacity := 3 * blue_boxcar_capacity
  (red_boxcars * red_boxcar_capacity + blue_boxcars * blue_boxcar_capacity + black_boxcars * black_boxcar_capacity) = 132000 :=
by
  sorry

end train_boxcar_capacity_l193_193821


namespace calories_burned_l193_193743

theorem calories_burned {running_minutes walking_minutes total_minutes calories_per_minute_running calories_per_minute_walking calories_total : ℕ}
    (h_run : running_minutes = 35)
    (h_total : total_minutes = 60)
    (h_calories_run : calories_per_minute_running = 10)
    (h_calories_walk : calories_per_minute_walking = 4)
    (h_walk : walking_minutes = total_minutes - running_minutes)
    (h_calories_total : calories_total = running_minutes * calories_per_minute_running + walking_minutes * calories_per_minute_walking) : 
    calories_total = 450 := by
  sorry

end calories_burned_l193_193743


namespace top_card_is_queen_probability_l193_193097

theorem top_card_is_queen_probability :
  let total_cards := 54
  let number_of_queens := 4
  (number_of_queens / total_cards) = (2 / 27) := by
    sorry

end top_card_is_queen_probability_l193_193097


namespace find_LP_l193_193941

variables (A B C K L P M : Type) 
variables {AC BC AK CK CL AM LP : ℕ}

-- Defining the given conditions
def conditions (AC BC AK CK : ℕ) (AM : ℕ) :=
  AC = 360 ∧ BC = 240 ∧ AK = CK ∧ AK = 180 ∧ AM = 144

-- The theorem statement: proving LP equals 57.6
theorem find_LP (h : conditions 360 240 180 180 144) : LP = 576 / 10 := 
by sorry

end find_LP_l193_193941


namespace triangle_is_right_l193_193233

theorem triangle_is_right (A B C a b c : ℝ) (h₁ : 0 < A) (h₂ : 0 < B) (h₃ : 0 < C) 
    (h₄ : A + B + C = π) (h_eq : a * (Real.cos C) + c * (Real.cos A) = b * (Real.sin B)) : B = π / 2 :=
by
  sorry

end triangle_is_right_l193_193233


namespace prove_s90_zero_l193_193435

variable {a : ℕ → ℕ}

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (n * (a 0) + (n * (n - 1) * (a 1 - a 0)) / 2)

theorem prove_s90_zero (a : ℕ → ℕ) (h_arith : is_arithmetic_sequence a) (h : sum_of_first_n_terms a 30 = sum_of_first_n_terms a 60) :
  sum_of_first_n_terms a 90 = 0 :=
sorry

end prove_s90_zero_l193_193435


namespace golden_ratio_expression_l193_193333

variables (R : ℝ)
noncomputable def divide_segment (R : ℝ) := R^(R^(R^2 + 1/R) + 1/R) + 1/R

theorem golden_ratio_expression :
  (R = (1 / (1 + R))) →
  divide_segment R = 2 :=
by
  sorry

end golden_ratio_expression_l193_193333


namespace range_of_a_l193_193592

theorem range_of_a 
  (a : ℝ) 
  (r : ℝ) 
  (h1 : r > 0) 
  (cos_le_zero : (3 * a - 9) / r ≤ 0) 
  (sin_gt_zero : (a + 2) / r > 0) : 
  -2 < a ∧ a ≤ 3 :=
sorry

end range_of_a_l193_193592


namespace largest_fraction_l193_193670

theorem largest_fraction (p q r s : ℕ) (hp : 0 < p) (hpq : p < q) (hqr : q < r) (hrs : r < s) : 
  max (max (max (max (↑(p + q) / ↑(r + s)) (↑(p + s) / ↑(q + r))) 
              (↑(q + r) / ↑(p + s))) 
          (↑(q + s) / ↑(p + r))) 
      (↑(r + s) / ↑(p + q)) = (↑(r + s) / ↑(p + q)) :=
sorry

end largest_fraction_l193_193670


namespace find_f_105_5_l193_193044

noncomputable def f : ℝ → ℝ := sorry

axiom even_function : ∀ x : ℝ, f x = f (-x)
axiom product_condition : ∀ x : ℝ, f x * f (x + 2) = -1
axiom specific_interval : ∀ x : ℝ, 2 ≤ x ∧ x ≤ 3 → f x = x

theorem find_f_105_5 : f 105.5 = 2.5 :=
by
  sorry

end find_f_105_5_l193_193044


namespace max_marks_l193_193110

theorem max_marks (M : ℝ) (h1 : 0.33 * M = 165): M = 500 :=
by
  sorry

end max_marks_l193_193110


namespace gcd_102_238_l193_193712

theorem gcd_102_238 : Nat.gcd 102 238 = 34 :=
by
  sorry

end gcd_102_238_l193_193712


namespace equilateral_triangle_l193_193117

theorem equilateral_triangle (a b c : ℝ) (h : a^2 + b^2 + c^2 = ab + bc + ca) : a = b ∧ b = c := 
by sorry

end equilateral_triangle_l193_193117


namespace silverware_probability_l193_193578

-- Defining the number of each type of silverware
def num_forks : ℕ := 8
def num_spoons : ℕ := 10
def num_knives : ℕ := 4
def total_silverware : ℕ := num_forks + num_spoons + num_knives
def num_remove : ℕ := 4

-- Proving the probability calculation
theorem silverware_probability :
  -- Calculation of the total number of ways to choose 4 pieces from 22
  let total_ways := Nat.choose total_silverware num_remove
  -- Calculation of ways to choose 2 forks from 8
  let ways_to_choose_forks := Nat.choose num_forks 2
  -- Calculation of ways to choose 1 spoon from 10
  let ways_to_choose_spoon := Nat.choose num_spoons 1
  -- Calculation of ways to choose 1 knife from 4
  let ways_to_choose_knife := Nat.choose num_knives 1
  -- Calculation of the number of favorable outcomes
  let favorable_outcomes := ways_to_choose_forks * ways_to_choose_spoon * ways_to_choose_knife
  -- Probability in simplified form
  let probability := (favorable_outcomes : ℚ) / total_ways
  probability = (32 : ℚ) / 209 :=
by
  sorry

end silverware_probability_l193_193578


namespace complement_of_irreducible_proper_fraction_is_irreducible_l193_193484

theorem complement_of_irreducible_proper_fraction_is_irreducible 
  (a b : ℤ) (h0 : 0 < a) (h1 : a < b) (h2 : Int.gcd a b = 1) : Int.gcd (b - a) b = 1 :=
sorry

end complement_of_irreducible_proper_fraction_is_irreducible_l193_193484


namespace smallest_B_for_divisibility_by_4_l193_193781

theorem smallest_B_for_divisibility_by_4 : 
  ∃ (B : ℕ), B < 10 ∧ (4 * 1000000 + B * 100000 + 80000 + 3961) % 4 = 0 ∧ ∀ (B' : ℕ), (B' < B ∧ B' < 10) → ¬ ((4 * 1000000 + B' * 100000 + 80000 + 3961) % 4 = 0) := 
sorry

end smallest_B_for_divisibility_by_4_l193_193781


namespace remaining_budget_is_correct_l193_193844

def budget := 750
def flasks_cost := 200
def test_tubes_cost := (2 / 3) * flasks_cost
def safety_gear_cost := (1 / 2) * test_tubes_cost
def chemicals_cost := (3 / 4) * flasks_cost
def instruments_min_cost := 50

def total_spent := flasks_cost + test_tubes_cost + safety_gear_cost + chemicals_cost
def remaining_budget_before_instruments := budget - total_spent
def remaining_budget_after_instruments := remaining_budget_before_instruments - instruments_min_cost

theorem remaining_budget_is_correct :
  remaining_budget_after_instruments = 150 := by
  unfold remaining_budget_after_instruments remaining_budget_before_instruments total_spent flasks_cost test_tubes_cost safety_gear_cost chemicals_cost budget
  sorry

end remaining_budget_is_correct_l193_193844


namespace prime_divisor_form_l193_193982

theorem prime_divisor_form {p q : ℕ} (hp : Nat.Prime p) (hpgt2 : p > 2) (hq : Nat.Prime q) (hq_dvd : q ∣ 2^p - 1) : 
  ∃ k : ℕ, q = 2 * k * p + 1 := 
sorry

end prime_divisor_form_l193_193982


namespace rectangle_area_from_square_l193_193958

theorem rectangle_area_from_square 
  (square_area : ℕ) 
  (width_rect : ℕ) 
  (length_rect : ℕ) 
  (h_square_area : square_area = 36)
  (h_width_rect : width_rect * width_rect = square_area)
  (h_length_rect : length_rect = 3 * width_rect) :
  width_rect * length_rect = 108 :=
by
  sorry

end rectangle_area_from_square_l193_193958


namespace measure_angle_A_l193_193330

-- Angles A and B are supplementary
def supplementary (A B : ℝ) : Prop :=
  A + B = 180

-- Definition of the problem conditions
def problem_conditions (A B : ℝ) : Prop :=
  supplementary A B ∧ A = 4 * B

-- The measure of angle A
def measure_of_A := 144

-- The statement to prove
theorem measure_angle_A (A B : ℝ) :
  problem_conditions A B → A = measure_of_A := 
by
  sorry

end measure_angle_A_l193_193330


namespace bricks_needed_to_build_wall_l193_193533

def volume_of_brick (length_brick height_brick thickness_brick : ℤ) : ℤ :=
  length_brick * height_brick * thickness_brick

def volume_of_wall (length_wall height_wall thickness_wall : ℤ) : ℤ :=
  length_wall * height_wall * thickness_wall

def number_of_bricks_needed (length_wall height_wall thickness_wall length_brick height_brick thickness_brick : ℤ) : ℤ :=
  (volume_of_wall length_wall height_wall thickness_wall + volume_of_brick length_brick height_brick thickness_brick - 1) / 
  volume_of_brick length_brick height_brick thickness_brick

theorem bricks_needed_to_build_wall : number_of_bricks_needed 800 100 5 25 11 6 = 243 := 
  by 
    sorry

end bricks_needed_to_build_wall_l193_193533


namespace quiz_common_difference_l193_193128

theorem quiz_common_difference 
  (x d : ℕ) 
  (h1 : x + 2 * d = 39) 
  (h2 : 8 * x + 28 * d = 360) 
  : d = 4 := 
  sorry

end quiz_common_difference_l193_193128


namespace license_plate_increase_factor_l193_193204

def old_license_plates := 26^2 * 10^3
def new_license_plates := 26^3 * 10^4

theorem license_plate_increase_factor : (new_license_plates / old_license_plates) = 260 := by
  sorry

end license_plate_increase_factor_l193_193204


namespace ratio_unchanged_l193_193424

-- Define the initial ratio
def initial_ratio (a b : ℕ) : ℚ := a / b

-- Define the new ratio after transformation
def new_ratio (a b : ℕ) : ℚ := (3 * a) / (b / (1/3))

-- The theorem stating that the ratio remains unchanged
theorem ratio_unchanged (a b : ℕ) (hb : b ≠ 0) :
  initial_ratio a b = new_ratio a b :=
by
  sorry

end ratio_unchanged_l193_193424


namespace milk_price_same_after_reductions_l193_193736

theorem milk_price_same_after_reductions (x : ℝ) (h1 : 0 < x) :
  (x - 0.4 * x) = ((x - 0.2 * x) - 0.25 * (x - 0.2 * x)) :=
by
  sorry

end milk_price_same_after_reductions_l193_193736


namespace molecular_weight_of_BaBr2_l193_193955

theorem molecular_weight_of_BaBr2 
    (atomic_weight_Ba : ℝ)
    (atomic_weight_Br : ℝ)
    (moles : ℝ)
    (hBa : atomic_weight_Ba = 137.33)
    (hBr : atomic_weight_Br = 79.90) 
    (hmol : moles = 8) :
    (atomic_weight_Ba + 2 * atomic_weight_Br) * moles = 2377.04 :=
by 
  sorry

end molecular_weight_of_BaBr2_l193_193955


namespace simplify_expression_l193_193126

theorem simplify_expression :
  (2^5 + 4^3) * (2^2 - (-2)^3)^8 = 96 * 12^8 :=
by
  sorry

end simplify_expression_l193_193126


namespace factor_tree_value_l193_193685

theorem factor_tree_value :
  let Q := 5 * 3
  let R := 11 * 2
  let Y := 2 * Q
  let Z := 7 * R
  let X := Y * Z
  X = 4620 :=
by
  sorry

end factor_tree_value_l193_193685


namespace max_value_inequality_l193_193405

theorem max_value_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (abc * (a + b + c)) / ((a + b)^2 * (b + c)^3) ≤ 1 / 4 :=
sorry

end max_value_inequality_l193_193405


namespace three_digit_number_proof_l193_193574

noncomputable def is_prime (n : ℕ) : Prop := (n > 1) ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem three_digit_number_proof (H T U : ℕ) (h1 : H = 2 * T)
  (h2 : U = 2 * T^3)
  (h3 : is_prime (H + T + U))
  (h_digits : H < 10 ∧ T < 10 ∧ U < 10)
  (h_nonzero : T > 0) : H * 100 + T * 10 + U = 212 := 
by
  sorry

end three_digit_number_proof_l193_193574


namespace solve_for_y_l193_193915

theorem solve_for_y (y : ℕ) (h1 : 40 = 2^3 * 5) (h2 : 8 = 2^3) :
  40^3 = 8^y ↔ y = 3 :=
by sorry

end solve_for_y_l193_193915


namespace blueberries_in_blue_box_l193_193020

theorem blueberries_in_blue_box (B S : ℕ) (h1 : S - B = 12) (h2 : S + B = 76) : B = 32 :=
sorry

end blueberries_in_blue_box_l193_193020


namespace layers_tall_l193_193468

def total_cards (n_d c_d : ℕ) : ℕ := n_d * c_d
def layers (total c_l : ℕ) : ℕ := total / c_l

theorem layers_tall (n_d c_d c_l : ℕ) (hn_d : n_d = 16) (hc_d : c_d = 52) (hc_l : c_l = 26) : 
  layers (total_cards n_d c_d) c_l = 32 := by
  sorry

end layers_tall_l193_193468


namespace rosa_parks_food_drive_l193_193588

theorem rosa_parks_food_drive :
  ∀ (total_students students_collected_12_cans students_collected_none students_remaining total_cans cans_collected_first_group total_cans_first_group total_cans_last_group cans_per_student_last_group : ℕ),
    total_students = 30 →
    students_collected_12_cans = 15 →
    students_collected_none = 2 →
    students_remaining = total_students - students_collected_12_cans - students_collected_none →
    total_cans = 232 →
    cans_collected_first_group = 12 →
    total_cans_first_group = students_collected_12_cans * cans_collected_first_group →
    total_cans_last_group = total_cans - total_cans_first_group →
    cans_per_student_last_group = total_cans_last_group / students_remaining →
    cans_per_student_last_group = 4 :=
by
  intros total_students students_collected_12_cans students_collected_none students_remaining total_cans cans_collected_first_group total_cans_first_group total_cans_last_group cans_per_student_last_group
  sorry

end rosa_parks_food_drive_l193_193588


namespace remainder_x_squared_l193_193791

theorem remainder_x_squared (x : ℤ) 
  (h1 : 5 * x ≡ 10 [ZMOD 20])
  (h2 : 7 * x ≡ 14 [ZMOD 20]) : 
  (x^2 ≡ 4 [ZMOD 20]) :=
sorry

end remainder_x_squared_l193_193791


namespace triangle_inequality_range_isosceles_triangle_perimeter_l193_193278

-- Define the parameters for the triangle
variables (AB BC AC a : ℝ)
variables (h_AB : AB = 8) (h_BC : BC = 2 * a + 2) (h_AC : AC = 22)

-- Define the lean proof problem for the given conditions
theorem triangle_inequality_range (h_triangle : AB = 8 ∧ BC = 2 * a + 2 ∧ AC = 22) :
  6 < a ∧ a < 14 := sorry

-- Define the isosceles condition and perimeter calculation
theorem isosceles_triangle_perimeter (h_isosceles : BC = AC) :
  perimeter = 52 := sorry

end triangle_inequality_range_isosceles_triangle_perimeter_l193_193278


namespace bacteria_eradication_time_l193_193446

noncomputable def infected_bacteria (n : ℕ) : ℕ := n

theorem bacteria_eradication_time (n : ℕ) : ∃ t : ℕ, t = n ∧ (∃ infect: ℕ → ℕ, ∀ t < n, infect t ≤ n ∧ infect n = n ∧ (∀ k < n, infect k = 2^(n-k))) :=
by sorry

end bacteria_eradication_time_l193_193446


namespace unique_handshakes_l193_193746

theorem unique_handshakes :
  let twins_sets := 12
  let triplets_sets := 3
  let twins := twins_sets * 2
  let triplets := triplets_sets * 3
  let twin_shakes_twins := twins * (twins - 2)
  let triplet_shakes_triplets := triplets * (triplets - 3)
  let twin_shakes_triplets := twins * (triplets / 3)
  (twin_shakes_twins + triplet_shakes_triplets + twin_shakes_triplets) / 2 = 327 := by
  sorry

end unique_handshakes_l193_193746


namespace range_of_a_l193_193782

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x - a| + |x - 12| < 6 → False) → (a ≤ 6 ∨ a ≥ 18) :=
by 
  intro h
  sorry

end range_of_a_l193_193782


namespace find_k_exact_one_real_solution_l193_193790

theorem find_k_exact_one_real_solution (k : ℝ) :
  (∀ x : ℝ, (3*x + 6)*(x - 4) = -33 + k*x) ↔ (k = -6 + 6*Real.sqrt 3 ∨ k = -6 - 6*Real.sqrt 3) := 
by
  sorry

end find_k_exact_one_real_solution_l193_193790


namespace boxes_needed_l193_193378

theorem boxes_needed (total_oranges boxes_capacity : ℕ) (h1 : total_oranges = 94) (h2 : boxes_capacity = 8) : 
  (total_oranges + boxes_capacity - 1) / boxes_capacity = 12 := 
by
  sorry

end boxes_needed_l193_193378


namespace selling_price_l193_193295

theorem selling_price (CP P : ℝ) (hCP : CP = 320) (hP : P = 0.25) : CP + (P * CP) = 400 :=
by
  sorry

end selling_price_l193_193295


namespace pipe_fills_tank_in_10_hours_l193_193264

variables (pipe_rate leak_rate : ℝ)

-- Conditions
def combined_rate := pipe_rate - leak_rate
def leak_time := 30
def combined_time := 15

-- Express leak_rate from leak_time
noncomputable def leak_rate_def : ℝ := 1 / leak_time

-- Express pipe_rate from combined_time with leak_rate considered
noncomputable def pipe_rate_def : ℝ := 1 / combined_time + leak_rate_def

-- Theorem to be proved
theorem pipe_fills_tank_in_10_hours :
  (1 / pipe_rate_def) = 10 :=
by
  sorry

end pipe_fills_tank_in_10_hours_l193_193264


namespace seed_grow_prob_l193_193889

theorem seed_grow_prob (P_G P_S_given_G : ℝ) (hP_G : P_G = 0.9) (hP_S_given_G : P_S_given_G = 0.8) :
  P_G * P_S_given_G = 0.72 :=
by
  rw [hP_G, hP_S_given_G]
  norm_num

end seed_grow_prob_l193_193889


namespace sheep_count_l193_193817

/-- The ratio between the number of sheep and the number of horses at the Stewart farm is 2 to 7.
    Each horse is fed 230 ounces of horse food per day, and the farm needs a total of 12,880 ounces
    of horse food per day. -/
theorem sheep_count (S H : ℕ) (h_ratio : S = (2 / 7) * H)
    (h_food : H * 230 = 12880) : S = 16 :=
sorry

end sheep_count_l193_193817


namespace smallest_possible_value_l193_193101

/-
Given:
1. m and n are positive integers.
2. gcd of m and n is (x + 5).
3. lcm of m and n is x * (x + 5).
4. m = 60.
5. x is a positive integer.

Prove:
The smallest possible value of n is 100.
-/

theorem smallest_possible_value 
  (m n x : ℕ) 
  (h1 : m = 60) 
  (h2 : x > 0) 
  (h3 : Nat.gcd m n = x + 5) 
  (h4 : Nat.lcm m n = x * (x + 5)) : 
  n = 100 := 
by 
  sorry

end smallest_possible_value_l193_193101


namespace grape_ratio_new_new_cans_from_grape_l193_193680

-- Definitions derived from the problem conditions
def apple_ratio_initial : ℚ := 1 / 6
def grape_ratio_initial : ℚ := 1 / 10
def apple_ratio_new : ℚ := 1 / 5

-- Prove the new grape_ratio
theorem grape_ratio_new : ℚ :=
  let total_volume_per_can := apple_ratio_initial + grape_ratio_initial
  let grape_ratio_new_reciprocal := (total_volume_per_can - apple_ratio_new)
  1 / grape_ratio_new_reciprocal

-- Required final quantity of cans
theorem new_cans_from_grape : 
  (1 / grape_ratio_new) = 15 :=
sorry

end grape_ratio_new_new_cans_from_grape_l193_193680


namespace solution_x_l193_193792

noncomputable def find_x (x : ℝ) : Prop :=
  (Real.log (x^4))^2 = (Real.log x)^6

theorem solution_x (x : ℝ) : find_x x ↔ (x = 1 ∨ x = Real.exp 2 ∨ x = Real.exp (-2)) :=
sorry

end solution_x_l193_193792


namespace intersection_of_P_and_Q_l193_193135

def P (x : ℝ) : Prop := 1 < x ∧ x < 4
def Q (x : ℝ) : Prop := 2 < x ∧ x < 3

theorem intersection_of_P_and_Q (x : ℝ) : P x ∧ Q x ↔ 2 < x ∧ x < 3 := by
  sorry

end intersection_of_P_and_Q_l193_193135


namespace Eric_return_time_l193_193347

theorem Eric_return_time (t1 t2 t_return : ℕ) 
  (h1 : t1 = 20) 
  (h2 : t2 = 10) 
  (h3 : t_return = 3 * (t1 + t2)) : 
  t_return = 90 := 
by 
  sorry

end Eric_return_time_l193_193347


namespace fourth_student_in_sample_l193_193364

def sample_interval (total_students : ℕ) (sample_size : ℕ) : ℕ :=
  total_students / sample_size

def in_sample (student_number : ℕ) (start : ℕ) (interval : ℕ) (n : ℕ) : Prop :=
  student_number = start + n * interval

theorem fourth_student_in_sample :
  ∀ (total_students sample_size : ℕ) (s1 s2 s3 : ℕ),
    total_students = 52 →
    sample_size = 4 →
    s1 = 7 →
    s2 = 33 →
    s3 = 46 →
    ∃ s4, in_sample s4 s1 (sample_interval total_students sample_size) 1 ∧
           in_sample s2 s1 (sample_interval total_students sample_size) 2 ∧
           in_sample s3 s1 (sample_interval total_students sample_size) 3 ∧
           s4 = 20 := 
by
  sorry

end fourth_student_in_sample_l193_193364


namespace carrots_per_bundle_l193_193744

theorem carrots_per_bundle (potatoes_total: ℕ) (potatoes_in_bundle: ℕ) (price_per_potato_bundle: ℝ) 
(carrot_total: ℕ) (price_per_carrot_bundle: ℝ) (total_revenue: ℝ) (carrots_per_bundle : ℕ) :
potatoes_total = 250 → potatoes_in_bundle = 25 → price_per_potato_bundle = 1.90 → 
carrot_total = 320 → price_per_carrot_bundle = 2 → total_revenue = 51 →
((carrots_per_bundle = carrot_total / ((total_revenue - (potatoes_total / potatoes_in_bundle) 
    * price_per_potato_bundle) / price_per_carrot_bundle))  ↔ carrots_per_bundle = 20) := by
  sorry

end carrots_per_bundle_l193_193744


namespace second_train_start_time_l193_193142

theorem second_train_start_time :
  let start_time_first_train := 14 -- 2:00 pm in 24-hour format
  let catch_up_time := 22          -- 10:00 pm in 24-hour format
  let speed_first_train := 70      -- km/h
  let speed_second_train := 80     -- km/h
  let travel_time_first_train := catch_up_time - start_time_first_train
  let distance_first_train := speed_first_train * travel_time_first_train
  let t := distance_first_train / speed_second_train
  let start_time_second_train := catch_up_time - t
  start_time_second_train = 15 := -- 3:00 pm in 24-hour format
by
  sorry

end second_train_start_time_l193_193142


namespace circle_is_axisymmetric_and_centrally_symmetric_l193_193087

structure Shape where
  isAxisymmetric : Prop
  isCentrallySymmetric : Prop

theorem circle_is_axisymmetric_and_centrally_symmetric :
  ∃ (s : Shape), s.isAxisymmetric ∧ s.isCentrallySymmetric :=
by
  sorry

end circle_is_axisymmetric_and_centrally_symmetric_l193_193087


namespace boxes_in_carton_l193_193493

theorem boxes_in_carton (cost_per_pack : ℕ) (packs_per_box : ℕ) (cost_dozen_cartons : ℕ) 
  (h1 : cost_per_pack = 1) (h2 : packs_per_box = 10) (h3 : cost_dozen_cartons = 1440) :
  (cost_dozen_cartons / 12) / (cost_per_pack * packs_per_box) = 12 :=
by
  sorry

end boxes_in_carton_l193_193493


namespace total_albums_l193_193796

-- Definitions based on given conditions
def adele_albums : ℕ := 30
def bridget_albums : ℕ := adele_albums - 15
def katrina_albums : ℕ := 6 * bridget_albums
def miriam_albums : ℕ := 5 * katrina_albums

-- The final statement to be proved
theorem total_albums : adele_albums + bridget_albums + katrina_albums + miriam_albums = 585 :=
by
  sorry

end total_albums_l193_193796


namespace n_decomposable_form_l193_193521

theorem n_decomposable_form (n : ℕ) (a : ℕ) (h₁ : a > 2) (h₂ : ∃ k, 1 < k ∧ n = 2^k) :
  (∀ d : ℕ, d ∣ n ∧ d ≠ n → (a^n - 2^n) % (a^d + 2^d) = 0) → ∃ k, 1 < k ∧ n = 2^k :=
by {
  sorry
}

end n_decomposable_form_l193_193521


namespace quadratic_roots_relationship_l193_193770

theorem quadratic_roots_relationship 
  (a b c α β : ℝ) 
  (h_eq : a * α^2 + b * α + c = 0) 
  (h_eq' : a * β^2 + b * β + c = 0)
  (h_roots : β = 3 * α) :
  3 * b^2 = 16 * a * c := 
sorry

end quadratic_roots_relationship_l193_193770


namespace range_of_real_number_m_l193_193999

open Set

variable {m : ℝ}

theorem range_of_real_number_m (U : Set ℝ) (A : Set ℝ) (B : Set ℝ) (h1 : U = univ) (h2 : A = { x | x < 1 }) (h3 : B = { x | x ≥ m }) (h4 : compl A ⊆ B) : m ≤ 1 := by
  sorry

end range_of_real_number_m_l193_193999


namespace SusanBooks_l193_193669

-- Definitions based on the conditions of the problem
def Lidia (S : ℕ) : ℕ := 4 * S
def TotalBooks (S : ℕ) : ℕ := S + Lidia S

-- The proof statement
theorem SusanBooks (S : ℕ) (h : TotalBooks S = 3000) : S = 600 :=
by
  sorry

end SusanBooks_l193_193669


namespace probability_all_choose_paper_l193_193483

-- Given conditions
def probability_choice_is_paper := 1 / 3

-- The theorem to be proved
theorem probability_all_choose_paper :
  probability_choice_is_paper ^ 3 = 1 / 27 :=
sorry

end probability_all_choose_paper_l193_193483


namespace geometric_sequence_a4_a7_l193_193344

theorem geometric_sequence_a4_a7 (a : ℕ → ℝ) (h1 : ∃ a₁ a₁₀, a₁ * a₁₀ = -6 ∧ a 1 = a₁ ∧ a 10 = a₁₀) :
  a 4 * a 7 = -6 :=
sorry

end geometric_sequence_a4_a7_l193_193344


namespace Steven_has_16_apples_l193_193022

variable (Jake_Peaches Steven_Peaches Jake_Apples Steven_Apples : ℕ)

theorem Steven_has_16_apples
  (h1 : Jake_Peaches = Steven_Peaches - 6)
  (h2 : Steven_Peaches = 17)
  (h3 : Steven_Peaches = Steven_Apples + 1)
  (h4 : Jake_Apples = Steven_Apples + 8) :
  Steven_Apples = 16 := by
  sorry

end Steven_has_16_apples_l193_193022


namespace one_of_a_b_c_is_one_l193_193554

theorem one_of_a_b_c_is_one (a b c : ℝ) (h1 : a * b * c = 1) (h2 : a + b + c = (1 / a) + (1 / b) + (1 / c)) :
  a = 1 ∨ b = 1 ∨ c = 1 :=
by
  sorry -- proof to be filled in

end one_of_a_b_c_is_one_l193_193554


namespace blue_red_difference_l193_193822

variable (B : ℕ) -- Blue crayons
variable (R : ℕ := 14) -- Red crayons
variable (Y : ℕ := 32) -- Yellow crayons
variable (H : Y = 2 * B - 6) -- Relationship between yellow and blue crayons

theorem blue_red_difference (B : ℕ) (H : (32:ℕ) = 2 * B - 6) : (B - 14 = 5) :=
by
  -- Proof steps goes here
  sorry

end blue_red_difference_l193_193822


namespace cos_2theta_plus_pi_l193_193696

-- Given condition
def tan_theta_eq_2 (θ : ℝ) : Prop := Real.tan θ = 2

-- The mathematical statement to prove
theorem cos_2theta_plus_pi (θ : ℝ) (h : tan_theta_eq_2 θ) : Real.cos (2 * θ + Real.pi) = 3 / 5 := 
sorry

end cos_2theta_plus_pi_l193_193696


namespace negation_of_every_function_has_parity_l193_193909

-- Assume the initial proposition
def every_function_has_parity := ∀ f : ℕ → ℕ, ∃ (p : ℕ), p = 0 ∨ p = 1

-- Negation of the original proposition
def exists_function_without_parity := ∃ f : ℕ → ℕ, ∀ p : ℕ, p ≠ 0 ∧ p ≠ 1

-- The theorem to prove
theorem negation_of_every_function_has_parity : 
  ¬ every_function_has_parity ↔ exists_function_without_parity := 
by
  unfold every_function_has_parity exists_function_without_parity
  sorry

end negation_of_every_function_has_parity_l193_193909


namespace speed_against_current_l193_193153

theorem speed_against_current (speed_with_current : ℝ) (current_speed : ℝ) (man_speed_against_current : ℝ) 
  (h : speed_with_current = 12) (h1 : current_speed = 2) : man_speed_against_current = 8 :=
by
  sorry

end speed_against_current_l193_193153


namespace mean_of_combined_set_l193_193607

theorem mean_of_combined_set
  (mean1 : ℕ → ℝ)
  (n1 : ℕ)
  (mean2 : ℕ → ℝ)
  (n2 : ℕ)
  (h1 : ∀ n1, mean1 n1 = 15)
  (h2 : ∀ n2, mean2 n2 = 26) :
  (n1 + n2) = 15 → 
  ((n1 * 15 + n2 * 26) / (n1 + n2)) = (313/15) :=
by
  sorry

end mean_of_combined_set_l193_193607


namespace parabola_focus_distance_l193_193027

theorem parabola_focus_distance (p : ℝ) : 
  (∀ (y : ℝ), y^2 = 2 * p * 4 → abs (4 + p / 2) = 5) → 
  p = 2 :=
by
  sorry

end parabola_focus_distance_l193_193027


namespace incorrect_statement_D_l193_193967

-- Definitions based on conditions
def length_of_spring (x : ℝ) : ℝ := 8 + 0.5 * x

-- Incorrect Statement (to be proved as incorrect)
def statement_D_incorrect : Prop :=
  ¬ (length_of_spring 30 = 23)

-- Main theorem statement
theorem incorrect_statement_D : statement_D_incorrect :=
by
  sorry

end incorrect_statement_D_l193_193967


namespace average_speed_l193_193526

theorem average_speed (x : ℝ) (h1 : x > 0) :
  let dist1 := x
  let speed1 := 40
  let dist2 := 4 * x
  let speed2 := 20
  let total_dist := dist1 + dist2
  let time1 := dist1 / speed1
  let time2 := dist2 / speed2
  let total_time := time1 + time2
  let avg_speed := total_dist / total_time
  avg_speed = 200 / 9 :=
by
  -- Definitions
  let dist1 := x
  let speed1 := 40
  let dist2 := 4 * x
  let speed2 := 20
  let total_dist := dist1 + dist2
  let time1 := dist1 / speed1
  let time2 := dist2 / speed2
  let total_time := time1 + time2
  let avg_speed := total_dist / total_time
  -- Proof structure, concluding with the correct answer.
  sorry

end average_speed_l193_193526


namespace price_difference_is_7_42_l193_193575

def total_cost : ℝ := 80.34
def shirt_price : ℝ := 36.46
def sweater_price : ℝ := total_cost - shirt_price
def price_difference : ℝ := sweater_price - shirt_price

theorem price_difference_is_7_42 : price_difference = 7.42 :=
  by
    sorry

end price_difference_is_7_42_l193_193575


namespace sin_squared_equiv_cosine_l193_193678

theorem sin_squared_equiv_cosine :
  1 - 2 * (Real.sin (Real.pi / 8))^2 = (Real.sqrt 2) / 2 :=
by sorry

end sin_squared_equiv_cosine_l193_193678


namespace area_of_square_with_diagonal_30_l193_193610

theorem area_of_square_with_diagonal_30 :
  ∀ (d : ℝ), d = 30 → (d * d / 2) = 450 := 
by
  intros d h
  rw [h]
  sorry

end area_of_square_with_diagonal_30_l193_193610


namespace sum_is_constant_l193_193866

variable (a b c d : ℚ) -- declare variables states as rational numbers

theorem sum_is_constant :
  (a + 2 = b + 3 ∧ b + 3 = c + 4 ∧ c + 4 = d + 5 ∧ d + 5 = a + b + c + d + 7) →
  a + b + c + d = -(14 / 3) :=
by
  intros h
  sorry

end sum_is_constant_l193_193866


namespace factorization_eq_l193_193709

theorem factorization_eq :
  ∀ (a : ℝ), a^2 + 4 * a - 21 = (a - 3) * (a + 7) := by
  intro a
  sorry

end factorization_eq_l193_193709


namespace find_pairs_l193_193212

theorem find_pairs (a b : ℕ) (h : a + b + a * b = 1000) : 
  (a = 6 ∧ b = 142) ∨ (a = 142 ∧ b = 6) ∨
  (a = 10 ∧ b = 90) ∨ (a = 90 ∧ b = 10) ∨
  (a = 12 ∧ b = 76) ∨ (a = 76 ∧ b = 12) :=
by sorry

end find_pairs_l193_193212


namespace no_real_ordered_triples_l193_193243

theorem no_real_ordered_triples (x y z : ℝ) (h1 : x + y = 3) (h2 : xy - z^2 = 4) : false :=
sorry

end no_real_ordered_triples_l193_193243


namespace birthday_cars_equal_12_l193_193986

namespace ToyCars

def initial_cars : Nat := 14
def bought_cars : Nat := 28
def sister_gave : Nat := 8
def friend_gave : Nat := 3
def remaining_cars : Nat := 43

def total_initial_cars := initial_cars + bought_cars
def total_given_away := sister_gave + friend_gave

theorem birthday_cars_equal_12 (B : Nat) (h : total_initial_cars + B - total_given_away = remaining_cars) : B = 12 :=
sorry

end ToyCars

end birthday_cars_equal_12_l193_193986


namespace complex_div_conjugate_l193_193394

theorem complex_div_conjugate (a b : ℂ) (h1 : a = 2 - I) (h2 : b = 1 + 2 * I) :
    a / b = -I := by
  sorry

end complex_div_conjugate_l193_193394


namespace discriminant_quadratic_eq_l193_193332

theorem discriminant_quadratic_eq : 
  let a := 1
  let b := -7
  let c := 4
  let Δ := b^2 - 4 * a * c
  Δ = 33 :=
by
  let a := 1
  let b := -7
  let c := 4
  let Δ := b^2 - 4 * a * c
  exact sorry

end discriminant_quadratic_eq_l193_193332


namespace height_of_barbed_wire_l193_193120

theorem height_of_barbed_wire (area : ℝ) (cost_per_meter : ℝ) (gate_width : ℝ) (total_cost : ℝ) (h : ℝ) :
  area = 3136 →
  cost_per_meter = 1.50 →
  gate_width = 2 →
  total_cost = 999 →
  h = 3 := 
by
  sorry

end height_of_barbed_wire_l193_193120


namespace complex_eq_z100_zReciprocal_l193_193327

theorem complex_eq_z100_zReciprocal
  (z : ℂ)
  (h : z + z⁻¹ = 2 * Real.cos (5 * Real.pi / 180)) :
  z^100 + z⁻¹^100 = -2 * Real.cos (40 * Real.pi / 180) :=
by
  sorry

end complex_eq_z100_zReciprocal_l193_193327


namespace ant_prob_bottom_vertex_l193_193762

theorem ant_prob_bottom_vertex :
  let top := 1
  let first_layer := 4
  let second_layer := 4
  let bottom := 1
  let prob_first_layer := 1 / first_layer
  let prob_second_layer := 1 / second_layer
  let prob_bottom := 1 / (second_layer + bottom)
  prob_first_layer * prob_second_layer * prob_bottom = 1 / 80 :=
by
  sorry

end ant_prob_bottom_vertex_l193_193762


namespace hexagonal_pyramid_edge_length_l193_193806

noncomputable def hexagonal_pyramid_edge_sum (s h : ℝ) : ℝ :=
  let perimeter := 6 * s
  let center_to_vertex := s * (1 / 2) * Real.sqrt 3
  let slant_height := Real.sqrt (h^2 + center_to_vertex^2)
  let edge_sum := perimeter + 6 * slant_height
  edge_sum

theorem hexagonal_pyramid_edge_length (s h : ℝ) (a : ℝ) :
  s = 8 →
  h = 15 →
  a = 48 + 6 * Real.sqrt 273 →
  hexagonal_pyramid_edge_sum s h = a :=
by
  intros
  sorry

end hexagonal_pyramid_edge_length_l193_193806


namespace k_value_if_function_not_in_first_quadrant_l193_193377

theorem k_value_if_function_not_in_first_quadrant : 
  ∀ k : ℝ, (∀ x : ℝ, x > 0 → (k - 2) * x ^ (|k|) + k ≤ 0) → k = -1 :=
by
  sorry

end k_value_if_function_not_in_first_quadrant_l193_193377


namespace cylinder_volume_options_l193_193534

theorem cylinder_volume_options (length width : ℝ) (h₀ : length = 4) (h₁ : width = 2) :
  ∃ V, (V = (4 / π) ∨ V = (8 / π)) :=
by
  sorry

end cylinder_volume_options_l193_193534


namespace min_value_expression_l193_193402

theorem min_value_expression (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2/x + 3/y = 1) :
  x/2 + y/3 = 4 :=
sorry

end min_value_expression_l193_193402


namespace negation_of_exists_l193_193934

open Set Real

theorem negation_of_exists (x : Real) :
  ¬ (∃ x ∈ Icc 0 1, x^3 + x^2 > 1) ↔ ∀ x ∈ Icc 0 1, x^3 + x^2 ≤ 1 := 
by sorry

end negation_of_exists_l193_193934


namespace students_agreed_total_l193_193600

theorem students_agreed_total :
  let third_grade_agreed : ℕ := 154
  let fourth_grade_agreed : ℕ := 237
  third_grade_agreed + fourth_grade_agreed = 391 := 
by
  let third_grade_agreed : ℕ := 154
  let fourth_grade_agreed : ℕ := 237
  show third_grade_agreed + fourth_grade_agreed = 391
  sorry

end students_agreed_total_l193_193600


namespace josh_money_remaining_l193_193980

theorem josh_money_remaining :
  let initial := 50.00
  let shirt := 7.85
  let meal := 15.49
  let magazine := 6.13
  let friends_debt := 3.27
  let cd := 11.75
  initial - shirt - meal - magazine - friends_debt - cd = 5.51 :=
by
  sorry

end josh_money_remaining_l193_193980


namespace find_daily_wage_c_l193_193430

noncomputable def daily_wage_c (total_earning : ℕ) (days_a : ℕ) (days_b : ℕ) (days_c : ℕ) (days_d : ℕ) (ratio_a : ℕ) (ratio_b : ℕ) (ratio_c : ℕ) (ratio_d : ℕ) : ℝ :=
  let total_ratio := days_a * ratio_a + days_b * ratio_b + days_c * ratio_c + days_d * ratio_d
  let x := total_earning / total_ratio
  ratio_c * x

theorem find_daily_wage_c :
  daily_wage_c 3780 6 9 4 12 3 4 5 7 = 119.60 :=
by
  sorry

end find_daily_wage_c_l193_193430


namespace distance_from_M0_to_plane_is_sqrt77_l193_193374

structure Point3D :=
  (x : ℝ) (y : ℝ) (z : ℝ)

def M1 : Point3D := ⟨1, 0, 2⟩
def M2 : Point3D := ⟨1, 2, -1⟩
def M3 : Point3D := ⟨2, -2, 1⟩
def M0 : Point3D := ⟨-5, -9, 1⟩

noncomputable def distance_to_plane (P : Point3D) (A B C : Point3D) : ℝ := sorry

theorem distance_from_M0_to_plane_is_sqrt77 : 
  distance_to_plane M0 M1 M2 M3 = Real.sqrt 77 := sorry

end distance_from_M0_to_plane_is_sqrt77_l193_193374


namespace katie_roll_probability_l193_193803

def prob_less_than_five (d : ℕ) : ℚ :=
if d < 5 then 1 else 0

def prob_even (d : ℕ) : ℚ :=
if d % 2 = 0 then 1 else 0

theorem katie_roll_probability :
  (prob_less_than_five 1 + prob_less_than_five 2 + prob_less_than_five 3 + prob_less_than_five 4 +
  prob_less_than_five 5 + prob_less_than_five 6) / 6 *
  (prob_even 1 + prob_even 2 + prob_even 3 + prob_even 4 +
  prob_even 5 + prob_even 6) / 6 = 1 / 3 :=
sorry

end katie_roll_probability_l193_193803


namespace tobys_friends_boys_count_l193_193626

theorem tobys_friends_boys_count (total_friends : ℕ) (girls : ℕ) (boys_percentage : ℕ) 
    (h1 : girls = 27) (h2 : boys_percentage = 55) (total_friends_calc : total_friends = 60) : 
    (total_friends * boys_percentage / 100) = 33 :=
by
  -- Proof is deferred
  sorry

end tobys_friends_boys_count_l193_193626


namespace find_first_number_l193_193075

noncomputable def x : ℕ := 7981
noncomputable def y : ℕ := 9409
noncomputable def mean_proportional : ℕ := 8665

theorem find_first_number (mean_is_correct : (mean_proportional^2 = x * y)) : x = 7981 := by
-- Given: mean_proportional^2 = x * y
-- Goal: x = 7981
  sorry

end find_first_number_l193_193075


namespace expression_value_l193_193365

theorem expression_value : 3 * 11 + 3 * 12 + 3 * 15 + 11 = 125 := 
by {
  sorry
}

end expression_value_l193_193365


namespace calc_1_calc_2_calc_3_calc_4_l193_193747

section
variables {m n x y z : ℕ} -- assuming all variables are natural numbers for simplicity.
-- Problem 1
theorem calc_1 : (2 * m * n) / (3 * m ^ 2) * (6 * m * n) / (5 * n) = (4 * n) / 5 :=
sorry

-- Problem 2
theorem calc_2 : (5 * x - 5 * y) / (3 * x ^ 2 * y) * (9 * x * y ^ 2) / (x ^ 2 - y ^ 2) = 
  15 * y / (x * (x + y)) :=
sorry

-- Problem 3
theorem calc_3 : ((x ^ 3 * y ^ 2) / z) ^ 2 * ((y * z) / x ^ 2) ^ 3 = y ^ 7 * z :=
sorry

-- Problem 4
theorem calc_4 : (4 * x ^ 2 * y ^ 2) / (2 * x + y) * (4 * x ^ 2 + 4 * x * y + y ^ 2) / (2 * x + y) / 
  ((2 * x * y) * (2 * x - y) / (4 * x ^ 2 - y ^ 2)) = 4 * x ^ 2 * y + 2 * x * y ^ 2 :=
sorry
end

end calc_1_calc_2_calc_3_calc_4_l193_193747


namespace convert_to_rectangular_and_find_line_l193_193033

noncomputable def circle_eq1 (x y : ℝ) : Prop := x^2 + y^2 = 4 * x
noncomputable def circle_eq2 (x y : ℝ) : Prop := x^2 + y^2 + 4 * y = 0
noncomputable def line_eq (x y : ℝ) : Prop := y = -x

theorem convert_to_rectangular_and_find_line :
  (∀ x y : ℝ, circle_eq1 x y → x^2 + y^2 = 4 * x) →
  (∀ x y : ℝ, circle_eq2 x y → x^2 + y^2 + 4 * y = 0) →
  (∀ x y : ℝ, circle_eq1 x y ∧ circle_eq2 x y → line_eq x y)
:=
sorry

end convert_to_rectangular_and_find_line_l193_193033


namespace probability_one_person_hits_probability_plane_is_hit_l193_193801
noncomputable def P_A := 0.7
noncomputable def P_B := 0.6

theorem probability_one_person_hits : P_A * (1 - P_B) + (1 - P_A) * P_B = 0.46 :=
by
  sorry

theorem probability_plane_is_hit : 1 - (1 - P_A) * (1 - P_B) = 0.88 :=
by
  sorry

end probability_one_person_hits_probability_plane_is_hit_l193_193801


namespace problem1_problem2_problem3_l193_193676

-- Problem (I)
theorem problem1 (x : ℝ) (hx : x > 1) : 2 * Real.log x < x - 1/x :=
sorry

-- Problem (II)
theorem problem2 (a : ℝ) : (∀ t : ℝ, t > 0 → (1 + a / t) * Real.log (1 + t) > a) → 0 < a ∧ a ≤ 2 :=
sorry

-- Problem (III)
theorem problem3 : (9/10 : ℝ)^19 < 1 / (Real.exp 2) :=
sorry

end problem1_problem2_problem3_l193_193676


namespace thyme_pots_count_l193_193454

theorem thyme_pots_count
  (basil_pots : ℕ := 3)
  (rosemary_pots : ℕ := 9)
  (leaves_per_basil_pot : ℕ := 4)
  (leaves_per_rosemary_pot : ℕ := 18)
  (leaves_per_thyme_pot : ℕ := 30)
  (total_leaves : ℕ := 354)
  : (total_leaves - (basil_pots * leaves_per_basil_pot + rosemary_pots * leaves_per_rosemary_pot)) / leaves_per_thyme_pot = 6 :=
by
  sorry

end thyme_pots_count_l193_193454


namespace no_x_for_rational_sin_cos_l193_193307

-- Define rational predicate
def is_rational (r : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ r = a / b

-- Define the statement of the problem
theorem no_x_for_rational_sin_cos :
  ∀ x : ℝ, ¬ (is_rational (Real.sin x + Real.sqrt 2) ∧ is_rational (Real.cos x - Real.sqrt 2)) :=
by
  -- Placeholder for proof
  sorry

end no_x_for_rational_sin_cos_l193_193307


namespace range_of_2a_plus_b_l193_193930

variable {a b c A B C : Real}
variable {sin cos : Real → Real}

theorem range_of_2a_plus_b (h1 : a^2 + b^2 + ab = 4) (h2 : c = 2) (h3 : a = c * sin A / sin C) (h4 : b = c * sin B / sin C) :
  2 < 2 * a + b ∧ 2 * a + b < 4 :=
by
  sorry

end range_of_2a_plus_b_l193_193930


namespace domain_of_f_l193_193444

def domain_f := {x : ℝ | 2 * x - 3 > 0}

theorem domain_of_f : ∀ x : ℝ, x ∈ domain_f ↔ x > 3 / 2 := 
by
  intro x
  simp [domain_f]
  sorry

end domain_of_f_l193_193444


namespace find_x_l193_193262

theorem find_x (p q r x : ℝ) (h1 : (p + q + r) / 3 = 4) (h2 : (p + q + r + x) / 4 = 5) : x = 8 :=
sorry

end find_x_l193_193262


namespace min_abc_value_l193_193189

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem min_abc_value
  (a b c : ℕ)
  (h1: is_prime a)
  (h2 : is_prime b)
  (h3 : is_prime c)
  (h4 : a^5 ∣ (b^2 - c))
  (h5 : ∃ k : ℕ, (b + c) = k^2) :
  a * b * c = 1958 := sorry

end min_abc_value_l193_193189


namespace solve_for_x_l193_193263

theorem solve_for_x (x : ℚ) (h : (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2) : x = -7 / 6 :=
sorry

end solve_for_x_l193_193263


namespace find_rate_per_kg_grapes_l193_193920

-- Define the main conditions
def rate_per_kg_mango := 55
def total_payment := 985
def kg_grapes := 7
def kg_mangoes := 9

-- Define the problem statement
theorem find_rate_per_kg_grapes (G : ℝ) : 
  (kg_grapes * G + kg_mangoes * rate_per_kg_mango = total_payment) → 
  G = 70 :=
by
  sorry

end find_rate_per_kg_grapes_l193_193920


namespace distance_in_scientific_notation_l193_193573

-- Definition for the number to be expressed in scientific notation
def distance : ℝ := 55000000

-- Expressing the number in scientific notation
def scientific_notation : ℝ := 5.5 * (10 ^ 7)

-- Theorem statement asserting the equality
theorem distance_in_scientific_notation : distance = scientific_notation :=
  by
  -- Proof not required here, so we leave it as sorry
  sorry

end distance_in_scientific_notation_l193_193573


namespace sqrt_sub_sqrt_frac_eq_l193_193222

theorem sqrt_sub_sqrt_frac_eq : (Real.sqrt 3) - (Real.sqrt (1 / 3)) = (2 * Real.sqrt 3) / 3 := 
by 
  sorry

end sqrt_sub_sqrt_frac_eq_l193_193222


namespace twelve_plus_four_times_five_minus_five_cubed_equals_twelve_l193_193337

theorem twelve_plus_four_times_five_minus_five_cubed_equals_twelve :
  12 + 4 * (5 - 10 / 2) ^ 3 = 12 := by
  sorry

end twelve_plus_four_times_five_minus_five_cubed_equals_twelve_l193_193337


namespace factorial_division_l193_193169

-- Definitions of factorial used in Lean according to math problem statement.
open Nat

-- Statement of the proof problem in Lean 4.
theorem factorial_division : (12! - 11!) / 10! = 121 := by
  sorry

end factorial_division_l193_193169


namespace additional_water_added_l193_193007

variable (M W : ℕ)

theorem additional_water_added (M W : ℕ) (initial_mix : ℕ) (initial_ratio : ℕ × ℕ) (new_ratio : ℚ) :
  initial_mix = 45 →
  initial_ratio = (4, 1) →
  new_ratio = 4 / 3 →
  (4 / 5) * initial_mix = M →
  (1 / 5) * initial_mix + W = 3 / 4 * M →
  W = 18 :=
by
  sorry

end additional_water_added_l193_193007


namespace ab_equals_five_l193_193270

variable (a m b n : ℝ)

def arithmetic_seq (x y z : ℝ) : Prop :=
  2 * y = x + z

def geometric_seq (w x y z u : ℝ) : Prop :=
  x * x = w * y ∧ y * y = x * z ∧ z * z = y * u

theorem ab_equals_five
  (h1 : arithmetic_seq (-9) a (-1))
  (h2 : geometric_seq (-9) m b n (-1)) :
  a * b = 5 := sorry

end ab_equals_five_l193_193270


namespace inequality_am_gm_l193_193888

theorem inequality_am_gm (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a / c + c / b) ≥ (4 * a / (a + b)) ∧ (a / c + c / b = 4 * a / (a + b) ↔ a = b ∧ b = c) :=
by
  -- Proof steps
  sorry

end inequality_am_gm_l193_193888


namespace simplify_expression_l193_193360

theorem simplify_expression (z : ℝ) : (3 - 5*z^2) - (4 + 3*z^2) = -1 - 8*z^2 :=
by
  sorry

end simplify_expression_l193_193360


namespace sum_of_edges_rectangular_solid_l193_193227

theorem sum_of_edges_rectangular_solid
  (a r : ℝ)
  (hr : r ≠ 0)
  (volume_eq : (a / r) * a * (a * r) = 512)
  (surface_area_eq : 2 * ((a ^ 2) / r + a ^ 2 + (a ^ 2) * r) = 384)
  (geo_progression : true) : -- This is implicitly understood in the construction
  4 * ((a / r) + a + (a * r)) = 112 :=
by
  -- The proof will be placed here
  sorry

end sum_of_edges_rectangular_solid_l193_193227


namespace find_x_coordinate_l193_193846

noncomputable def point_on_plane (x y : ℝ) :=
  (|x + y - 1| / Real.sqrt 2 = |x| ∧
   |x| = |y - 3 * x| / Real.sqrt 10)

theorem find_x_coordinate (x y : ℝ) (h : point_on_plane x y) : 
  x = 1 / (4 + Real.sqrt 10 - Real.sqrt 2) :=
sorry

end find_x_coordinate_l193_193846


namespace find_rabbits_l193_193987

theorem find_rabbits (heads rabbits chickens : ℕ) (h1 : rabbits + chickens = 40) (h2 : 4 * rabbits = 10 * 2 * chickens - 8) : rabbits = 33 :=
by
  -- We skip the proof here
  sorry

end find_rabbits_l193_193987


namespace simultaneous_equations_solution_exists_l193_193425

theorem simultaneous_equations_solution_exists (m : ℝ) : 
  (∃ (x y : ℝ), y = m * x + 2 ∧ y = (3 * m - 2) * x + 5) ↔ m ≠ 1 :=
by
  -- proof goes here
  sorry

end simultaneous_equations_solution_exists_l193_193425


namespace juwella_read_more_last_night_l193_193519

-- Definitions of the conditions
def pages_three_nights_ago : ℕ := 15
def book_pages : ℕ := 100
def pages_tonight : ℕ := 20
def pages_two_nights_ago : ℕ := 2 * pages_three_nights_ago
def total_pages_before_tonight : ℕ := book_pages - pages_tonight
def pages_last_night : ℕ := total_pages_before_tonight - pages_three_nights_ago - pages_two_nights_ago

theorem juwella_read_more_last_night :
  pages_last_night - pages_two_nights_ago = 5 :=
by
  sorry

end juwella_read_more_last_night_l193_193519


namespace smallest_five_digit_congruent_two_mod_seventeen_l193_193490

theorem smallest_five_digit_congruent_two_mod_seventeen : 
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 17 = 2 ∧ n = 10013 :=
by
  sorry

end smallest_five_digit_congruent_two_mod_seventeen_l193_193490


namespace sequences_cover_naturals_without_repetition_l193_193001

theorem sequences_cover_naturals_without_repetition
  (x y : Real) 
  (hx : Irrational x) 
  (hy : Irrational y) 
  (hxy : 1/x + 1/y = 1) :
  (∀ n : ℕ, ∃! k : ℕ, (⌊k * x⌋ = n) ∨ (⌊k * y⌋ = n)) :=
sorry

end sequences_cover_naturals_without_repetition_l193_193001


namespace min_flash_drives_needed_l193_193878

theorem min_flash_drives_needed (total_files : ℕ) (capacity_per_drive : ℝ)  
  (num_files_0_9 : ℕ) (size_0_9 : ℝ) 
  (num_files_0_8 : ℕ) (size_0_8 : ℝ) 
  (size_0_6 : ℝ) 
  (remaining_files : ℕ) :
  total_files = 40 →
  capacity_per_drive = 2.88 →
  num_files_0_9 = 5 →
  size_0_9 = 0.9 →
  num_files_0_8 = 18 →
  size_0_8 = 0.8 →
  remaining_files = total_files - (num_files_0_9 + num_files_0_8) →
  size_0_6 = 0.6 →
  (num_files_0_9 * size_0_9 + num_files_0_8 * size_0_8 + remaining_files * size_0_6) / capacity_per_drive ≤ 13 :=
by {
  sorry
}

end min_flash_drives_needed_l193_193878


namespace Ben_total_clothes_l193_193839

-- Definitions of Alex's clothing items
def Alex_shirts := 4.5
def Alex_pants := 3.0
def Alex_shoes := 2.5
def Alex_hats := 1.5
def Alex_jackets := 2.0

-- Definitions of Joe's clothing items
def Joe_shirts := Alex_shirts + 3.5
def Joe_pants := Alex_pants - 2.5
def Joe_shoes := Alex_shoes
def Joe_hats := Alex_hats + 0.3
def Joe_jackets := Alex_jackets - 1.0

-- Definitions of Ben's clothing items
def Ben_shirts := Joe_shirts + 5.3
def Ben_pants := Alex_pants + 5.5
def Ben_shoes := Joe_shoes - 1.7
def Ben_hats := Alex_hats + 0.5
def Ben_jackets := Joe_jackets + 1.5

-- Statement to prove the total number of Ben's clothing items
def total_Ben_clothing_items := Ben_shirts + Ben_pants + Ben_shoes + Ben_hats + Ben_jackets

theorem Ben_total_clothes : total_Ben_clothing_items = 27.1 :=
by
  sorry

end Ben_total_clothes_l193_193839


namespace not_age_of_child_digit_l193_193820

variable {n : Nat}

theorem not_age_of_child_digit : 
  ∀ (ages : List Nat), 
    (∀ x ∈ ages, 5 ≤ x ∧ x ≤ 13) ∧ -- condition 1
    ages.Nodup ∧                    -- condition 2: distinct ages
    ages.length = 9 ∧               -- condition 1: 9 children
    (∃ num : Nat, 
       10000 ≤ num ∧ num < 100000 ∧         -- 5-digit number
       (∀ d : Nat, d ∈ num.digits 10 →     -- condition 3 & 4: each digit appears once and follows a consecutive pattern in increasing order
          1 ≤ d ∧ d ≤ 9) ∧
       (∀ age ∈ ages, num % age = 0)       -- condition 4: number divisible by all children's ages
    ) →
    ¬(9 ∈ ages) :=                         -- question: Prove that '9' is not the age of any child
by
  intro ages h
  -- The proof would go here
  sorry

end not_age_of_child_digit_l193_193820


namespace simplify_expression_l193_193350

variables (y : ℝ)

theorem simplify_expression : 
  3 * y + 4 * y^2 - 2 - (8 - 3 * y - 4 * y^2) = 8 * y^2 + 6 * y - 10 :=
by sorry

end simplify_expression_l193_193350


namespace share_of_a_is_240_l193_193141

theorem share_of_a_is_240 (A B C : ℝ) 
  (h1 : A = (2/3) * (B + C)) 
  (h2 : B = (2/3) * (A + C)) 
  (h3 : A + B + C = 600) : 
  A = 240 := 
by sorry

end share_of_a_is_240_l193_193141


namespace solution_set_for_f_gt_0_l193_193389

noncomputable def f (x : ℝ) : ℝ := sorry

theorem solution_set_for_f_gt_0
  (odd_f : ∀ x : ℝ, f (-x) = -f x)
  (f_one_eq_zero : f 1 = 0)
  (ineq_f : ∀ x : ℝ, x > 0 → (x * (deriv^[2] f x) - f x) / x^2 > 0) :
  { x : ℝ | f x > 0 } = { x : ℝ | -1 < x ∧ x < 0 } ∪ { x : ℝ | 1 < x } :=
sorry

end solution_set_for_f_gt_0_l193_193389


namespace jeremy_remaining_money_l193_193085

-- Conditions as definitions
def computer_cost : ℝ := 3000
def accessories_cost : ℝ := 0.1 * computer_cost
def initial_money : ℝ := 2 * computer_cost

-- Theorem statement for the proof problem
theorem jeremy_remaining_money : initial_money - computer_cost - accessories_cost = 2700 := by
  -- Proof will be added here
  sorry

end jeremy_remaining_money_l193_193085


namespace total_lemonade_poured_l193_193373

def lemonade_poured (first: ℝ) (second: ℝ) (third: ℝ) := first + second + third

theorem total_lemonade_poured :
  lemonade_poured 0.25 0.4166666666666667 0.25 = 0.917 :=
by
  sorry

end total_lemonade_poured_l193_193373


namespace razors_blades_equation_l193_193463

/-- Given the number of razors sold x,
each razor sold brings a profit of 30 yuan,
each blade sold incurs a loss of 0.5 yuan,
the number of blades sold is twice the number of razors sold,
and the total profit from these two products is 5800 yuan,
prove that the linear equation is -0.5 * 2 * x + 30 * x = 5800 -/
theorem razors_blades_equation (x : ℝ) :
  -0.5 * 2 * x + 30 * x = 5800 := 
sorry

end razors_blades_equation_l193_193463


namespace inverse_h_l193_193774

def f (x : ℝ) : ℝ := 5 * x - 7
def g (x : ℝ) : ℝ := 3 * x + 2
def h (x : ℝ) : ℝ := f (g x)

theorem inverse_h : (∀ x : ℝ, h (15 * x + 3) = x) :=
by
  -- Proof would go here
  sorry

end inverse_h_l193_193774


namespace John_profit_is_correct_l193_193864

-- Definitions of conditions as necessary in Lean
variable (initial_puppies : ℕ) (given_away_puppies : ℕ) (kept_puppy : ℕ) (price_per_puppy : ℤ) (payment_to_stud_owner : ℤ)

-- Specific values from the problem
def John_initial_puppies := 8
def John_given_away_puppies := 4
def John_kept_puppy := 1
def John_price_per_puppy := 600
def John_payment_to_stud_owner := 300

-- Calculate the number of puppies left to sell
def John_remaining_puppies := John_initial_puppies - John_given_away_puppies - John_kept_puppy

-- Calculate total earnings from selling puppies
def John_earnings := John_remaining_puppies * John_price_per_puppy

-- Calculate the profit by subtracting payment to the stud owner from earnings
def John_profit := John_earnings - John_payment_to_stud_owner

-- Statement to prove
theorem John_profit_is_correct : 
  John_profit = 1500 := 
by 
  -- The proof will be here but we use sorry to skip it as requested.
  sorry

-- This ensures the definitions match the given problem conditions
#eval (John_initial_puppies, John_given_away_puppies, John_kept_puppy, John_price_per_puppy, John_payment_to_stud_owner)

end John_profit_is_correct_l193_193864


namespace min_value_expression_l193_193346

theorem min_value_expression (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 1) :
  3 ≤ (1 / (x + y)) + ((x + y) / z) :=
by
  sorry

end min_value_expression_l193_193346


namespace cost_of_paving_is_correct_l193_193612

def length : ℝ := 5.5
def width : ℝ := 3.75
def rate_per_sq_metre : ℝ := 400
def area_of_rectangle (l: ℝ) (w: ℝ) : ℝ := l * w
def cost_of_paving_floor (area: ℝ) (rate: ℝ) : ℝ := area * rate

theorem cost_of_paving_is_correct
  (h_length: length = 5.5)
  (h_width: width = 3.75)
  (h_rate: rate_per_sq_metre = 400):
  cost_of_paving_floor (area_of_rectangle length width) rate_per_sq_metre = 8250 :=
  by {
    sorry
  }

end cost_of_paving_is_correct_l193_193612


namespace area_of_paper_l193_193216

theorem area_of_paper (L W : ℕ) (h1 : L + 2 * W = 34) (h2 : 2 * L + W = 38) : L * W = 140 := by
  sorry

end area_of_paper_l193_193216


namespace candy_ratio_l193_193777

theorem candy_ratio
  (red_candies : ℕ)
  (yellow_candies : ℕ)
  (blue_candies : ℕ)
  (total_candies : ℕ)
  (remaining_candies : ℕ)
  (h1 : red_candies = 40)
  (h2 : yellow_candies = 3 * red_candies - 20)
  (h3 : remaining_candies = 90)
  (h4 : total_candies = remaining_candies + yellow_candies)
  (h5 : blue_candies = total_candies - red_candies - yellow_candies) :
  blue_candies / yellow_candies = 1 / 2 :=
sorry

end candy_ratio_l193_193777


namespace original_cost_of_each_magazine_l193_193030

-- Definitions and conditions
def magazine_cost (C : ℝ) : Prop :=
  let total_magazines := 10
  let sell_price := 3.50
  let gain := 5
  let total_revenue := total_magazines * sell_price
  let total_cost := total_revenue - gain
  C = total_cost / total_magazines

-- Goal to prove
theorem original_cost_of_each_magazine : ∃ C : ℝ, magazine_cost C ∧ C = 3 :=
by
  sorry

end original_cost_of_each_magazine_l193_193030


namespace perimeter_regular_polygon_l193_193565

-- Definitions of the conditions
def side_length : ℕ := 8
def exterior_angle : ℕ := 72
def sum_of_exterior_angles : ℕ := 360

-- Number of sides calculation
def num_sides : ℕ := sum_of_exterior_angles / exterior_angle

-- Perimeter calculation
def perimeter (n : ℕ) (l : ℕ) : ℕ := n * l

-- Theorem statement
theorem perimeter_regular_polygon : perimeter num_sides side_length = 40 :=
by
  sorry

end perimeter_regular_polygon_l193_193565


namespace no_m_for_necessary_and_sufficient_condition_m_geq_3_for_necessary_condition_l193_193572

def P (x : ℝ) : Prop := x ^ 2 - 8 * x - 20 ≤ 0
def S (x : ℝ) (m : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m

theorem no_m_for_necessary_and_sufficient_condition :
  ¬ ∃ m : ℝ, ∀ x : ℝ, P x ↔ S x m :=
by sorry

theorem m_geq_3_for_necessary_condition :
  ∃ m : ℝ, (m ≥ 3) ∧ ∀ x : ℝ, S x m → P x :=
by sorry

end no_m_for_necessary_and_sufficient_condition_m_geq_3_for_necessary_condition_l193_193572


namespace problem_statement_l193_193960

theorem problem_statement : (2^2015 + 2^2011) / (2^2015 - 2^2011) = 17 / 15 :=
by
  -- Prove the equivalence as outlined above.
  sorry

end problem_statement_l193_193960


namespace circle_y_coords_sum_l193_193752

theorem circle_y_coords_sum (x y : ℝ) (hc : (x + 3)^2 + (y - 5)^2 = 64) (hx : x = 0) : y = 5 + Real.sqrt 55 ∨ y = 5 - Real.sqrt 55 → (5 + Real.sqrt 55) + (5 - Real.sqrt 55) = 10 := 
by
  intros
  sorry

end circle_y_coords_sum_l193_193752


namespace certain_number_division_l193_193622

theorem certain_number_division (x : ℝ) (h : x / 3 + x + 3 = 63) : x = 45 :=
by
  sorry

end certain_number_division_l193_193622


namespace g_at_6_is_zero_l193_193047

def g (x : ℝ) : ℝ := 3*x^4 - 18*x^3 + 31*x^2 - 29*x - 72

theorem g_at_6_is_zero : g 6 = 0 :=
by {
  sorry
}

end g_at_6_is_zero_l193_193047


namespace point_exists_if_square_or_rhombus_l193_193470

-- Definitions to state the problem
structure Point (α : Type*) := (x : α) (y : α)
structure Rectangle (α : Type*) := (A B C D : Point α)

-- Definition of equidistant property
def isEquidistant (α : Type*) [LinearOrderedField α] (P : Point α) (R : Rectangle α) : Prop :=
  let d1 := abs (P.y - R.A.y)
  let d2 := abs (P.y - R.C.y)
  let d3 := abs (P.x - R.A.x)
  let d4 := abs (P.x - R.B.x)
  d1 = d2 ∧ d2 = d3 ∧ d3 = d4

-- Theorem stating the problem
theorem point_exists_if_square_or_rhombus {α : Type*} [LinearOrderedField α]
  (R : Rectangle α) : 
  (∃ P : Point α, isEquidistant α P R) ↔ 
  (∃ (a b : α), (a ≠ b ∧ R.A.x = a ∧ R.B.x = b ∧ R.C.y = a ∧ R.D.y = b) ∨ 
                (a = b ∧ R.A.x = a ∧ R.B.x = b ∧ R.C.y = a ∧ R.D.y = b)) :=
sorry

end point_exists_if_square_or_rhombus_l193_193470


namespace percentage_dogs_movies_l193_193457

-- Definitions from conditions
def total_students : ℕ := 30
def students_preferring_dogs_videogames : ℕ := total_students / 2
def students_preferring_dogs : ℕ := 18
def students_preferring_dogs_movies : ℕ := students_preferring_dogs - students_preferring_dogs_videogames

-- Theorem statement
theorem percentage_dogs_movies : (students_preferring_dogs_movies * 100 / total_students) = 10 := by
  sorry

end percentage_dogs_movies_l193_193457


namespace volume_of_regular_tetrahedron_with_edge_length_1_l193_193447

-- We define the concepts needed: regular tetrahedron, edge length, and volume.
open Real

noncomputable def volume_of_regular_tetrahedron (a : ℝ) : ℝ :=
  let base_area := (sqrt 3 / 4) * a^2
  let height := sqrt (a^2 - (a * (sqrt 3 / 3))^2)
  (1 / 3) * base_area * height

-- The problem statement and our goal to prove:
theorem volume_of_regular_tetrahedron_with_edge_length_1 :
  volume_of_regular_tetrahedron 1 = sqrt 2 / 12 := sorry

end volume_of_regular_tetrahedron_with_edge_length_1_l193_193447


namespace length_of_tank_l193_193434

namespace TankProblem

def field_length : ℝ := 90
def field_breadth : ℝ := 50
def field_area : ℝ := field_length * field_breadth

def tank_breadth : ℝ := 20
def tank_depth : ℝ := 4

def earth_volume (L : ℝ) : ℝ := L * tank_breadth * tank_depth

def remaining_field_area (L : ℝ) : ℝ := field_area - L * tank_breadth

def height_increase : ℝ := 0.5

theorem length_of_tank (L : ℝ) :
  earth_volume L = remaining_field_area L * height_increase →
  L = 25 :=
by
  sorry

end TankProblem

end length_of_tank_l193_193434


namespace lea_total_cost_example_l193_193952

/-- Léa bought one book for $16, three binders for $2 each, and six notebooks for $1 each. -/
def total_cost (book_cost binders_cost notebooks_cost : ℕ) : ℕ :=
  book_cost + binders_cost + notebooks_cost

/-- Given the individual costs, prove the total cost of Léa's purchases is $28. -/
theorem lea_total_cost_example : total_cost 16 (3 * 2) (6 * 1) = 28 := by
  sorry

end lea_total_cost_example_l193_193952


namespace student_ticket_price_is_2_50_l193_193024

-- Defining the given conditions
def adult_ticket_price : ℝ := 4
def total_tickets_sold : ℕ := 59
def total_revenue : ℝ := 222.50
def student_tickets_sold : ℕ := 9

-- The number of adult tickets sold
def adult_tickets_sold : ℕ := total_tickets_sold - student_tickets_sold

-- The total revenue from adult tickets
def revenue_from_adult_tickets : ℝ := adult_tickets_sold * adult_ticket_price

-- The remaining revenue must come from student tickets and defining the price of student ticket
noncomputable def student_ticket_price : ℝ :=
  (total_revenue - revenue_from_adult_tickets) / student_tickets_sold

-- The theorem to be proved
theorem student_ticket_price_is_2_50 : student_ticket_price = 2.50 :=
by
  sorry

end student_ticket_price_is_2_50_l193_193024


namespace problem1_l193_193569

theorem problem1 (a : ℝ) (m n : ℕ) (h1 : a^m = 10) (h2 : a^n = 2) : a^(m - 2 * n) = 2.5 := by
  sorry

end problem1_l193_193569


namespace solve_for_V_l193_193074

open Real

theorem solve_for_V :
  ∃ k V, 
    (U = k * (V / W) ∧ (U = 16 ∧ W = 1 / 4 ∧ V = 2) ∧ (U = 25 ∧ W = 1 / 5 ∧ V = 2.5)) :=
by {
  sorry
}

end solve_for_V_l193_193074


namespace cos_double_plus_cos_l193_193255

theorem cos_double_plus_cos (α : ℝ) (h : Real.sin (Real.pi / 2 + α) = 1 / 3) :
  Real.cos (2 * α) + Real.cos α = -4 / 9 :=
by
  sorry

end cos_double_plus_cos_l193_193255


namespace car_truck_ratio_l193_193596

theorem car_truck_ratio (total_vehicles trucks cars : ℕ)
  (h1 : total_vehicles = 300)
  (h2 : trucks = 100)
  (h3 : cars + trucks = total_vehicles)
  (h4 : ∃ (k : ℕ), cars = k * trucks) : 
  cars / trucks = 2 :=
by
  sorry

end car_truck_ratio_l193_193596


namespace henry_has_30_more_lollipops_than_alison_l193_193953

noncomputable def num_lollipops_alison : ℕ := 60
noncomputable def num_lollipops_diane : ℕ := 2 * num_lollipops_alison
noncomputable def total_num_days : ℕ := 6
noncomputable def num_lollipops_per_day : ℕ := 45
noncomputable def total_lollipops : ℕ := total_num_days * num_lollipops_per_day
noncomputable def num_lollipops_total_ad : ℕ := num_lollipops_alison + num_lollipops_diane
noncomputable def num_lollipops_henry : ℕ := total_lollipops - num_lollipops_total_ad
noncomputable def lollipops_diff_henry_alison : ℕ := num_lollipops_henry - num_lollipops_alison

theorem henry_has_30_more_lollipops_than_alison :
  lollipops_diff_henry_alison = 30 :=
by
  unfold lollipops_diff_henry_alison
  unfold num_lollipops_henry
  unfold num_lollipops_total_ad
  unfold total_lollipops
  sorry

end henry_has_30_more_lollipops_than_alison_l193_193953


namespace find_integer_x_l193_193925

theorem find_integer_x (x : ℤ) :
  (2 * x > 70 → x > 35 ∧ 4 * x > 25 → x > 6) ∧
  (x < 100 → x = 6) ∧ -- Given valid inequality relations and restrictions
  ((2 * x <= 70 ∨ x >= 100) ∨ (4 * x <= 25 ∨ x <= 5)) := -- Certain contradictions inherent

  by sorry

end find_integer_x_l193_193925


namespace find_c_l193_193724

theorem find_c (c : ℝ) 
    (h : ∀ x, (x - 4) ∣ (c * x^3 + 16 * x^2 - 5 * c * x + 40)) : 
    c = -74 / 11 :=
by
  sorry

end find_c_l193_193724


namespace derivative_at_1_l193_193827

noncomputable def f (x : ℝ) : ℝ := Real.exp x - 1 * x + (1 / 2) * x^2

theorem derivative_at_1 : deriv f 1 = Real.exp 1 := 
by 
  sorry

end derivative_at_1_l193_193827


namespace swimming_pool_width_l193_193398

theorem swimming_pool_width (L D1 D2 V : ℝ) (W : ℝ) (h : L = 12) (h1 : D1 = 1) (h2 : D2 = 4) (hV : V = 270) : W = 9 :=
  by
    -- We begin by stating the formula for the volume of 
    -- a trapezoidal prism: Volume = (1/2) * (D1 + D2) * L * W
    
    -- According to the problem, we have the following conditions:
    have hVolume : V = (1/2) * (D1 + D2) * L * W :=
      by sorry

    -- Substitute the provided values into the volume equation:
    -- 270 = (1/2) * (1 + 4) * 12 * W
    
    -- Simplify and solve for W
    simp at hVolume
    exact sorry

end swimming_pool_width_l193_193398


namespace fifteenth_term_is_143_l193_193094

noncomputable def first_term : ℕ := 3
noncomputable def second_term : ℕ := 13
noncomputable def third_term : ℕ := 23
noncomputable def common_difference : ℕ := second_term - first_term
noncomputable def nth_term (n : ℕ) : ℕ := first_term + (n - 1) * common_difference

theorem fifteenth_term_is_143 :
  nth_term 15 = 143 := by
  sorry

end fifteenth_term_is_143_l193_193094


namespace Jeffs_donuts_l193_193665

theorem Jeffs_donuts (D : ℕ) (h1 : ∀ n, n = 12 * D - 20) (h2 : n = 100) : D = 10 :=
by
  sorry

end Jeffs_donuts_l193_193665


namespace complement_of_intersection_l193_193585

-- Definitions of the sets M and N
def M : Set ℝ := { x | x ≥ 2 }
def N : Set ℝ := { x | x < 3 }

-- Definition of the intersection of M and N
def M_inter_N : Set ℝ := { x | 2 ≤ x ∧ x < 3 }

-- Definition of the complement of M ∩ N in ℝ
def complement_M_inter_N : Set ℝ := { x | x < 2 ∨ x ≥ 3 }

-- The theorem to be proved
theorem complement_of_intersection :
  (M_inter_Nᶜ) = complement_M_inter_N :=
by sorry

end complement_of_intersection_l193_193585


namespace sandra_money_left_l193_193548

def sandra_savings : ℕ := 10
def mother_gift : ℕ := 4
def father_gift : ℕ := 2 * mother_gift
def candy_cost : ℚ := 0.5
def jelly_bean_cost : ℚ := 0.2
def num_candies : ℕ := 14
def num_jelly_beans : ℕ := 20

def total_money : ℕ := sandra_savings + mother_gift + father_gift
def total_candy_cost : ℚ := num_candies * candy_cost
def total_jelly_bean_cost : ℚ := num_jelly_beans * jelly_bean_cost
def total_cost : ℚ := total_candy_cost + total_jelly_bean_cost
def money_left : ℚ := total_money - total_cost

theorem sandra_money_left : money_left = 11 := by
  sorry

end sandra_money_left_l193_193548


namespace storage_methods_l193_193168

-- Definitions for the vertices and edges of the pyramid
structure Pyramid :=
  (P A B C D : Type)
  
-- Edges of the pyramid represented by pairs of vertices
def edges (P A B C D : Type) := [(P, A), (P, B), (P, C), (P, D), (A, B), (A, C), (A, D), (B, C), (B, D), (C, D)]

-- Safe storage condition: No edges sharing a common vertex in the same warehouse
def safe (edge1 edge2 : (Type × Type)) : Prop :=
  edge1.1 ≠ edge2.1 ∧ edge1.1 ≠ edge2.2 ∧ edge1.2 ≠ edge2.1 ∧ edge1.2 ≠ edge2.2

-- The number of different methods to store the chemical products safely
def number_of_safe_storage_methods : Nat :=
  -- We should replace this part by actual calculation or combinatorial methods relevant to the problem
  48

theorem storage_methods (P A B C D : Type) : number_of_safe_storage_methods = 48 :=
  sorry

end storage_methods_l193_193168


namespace number_of_large_posters_is_5_l193_193426

theorem number_of_large_posters_is_5
  (total_posters : ℕ)
  (small_posters_ratio : ℚ)
  (medium_posters_ratio : ℚ)
  (h_total : total_posters = 50)
  (h_small_ratio : small_posters_ratio = 2 / 5)
  (h_medium_ratio : medium_posters_ratio = 1 / 2) :
  (total_posters * (1 - small_posters_ratio - medium_posters_ratio)) = 5 :=
by sorry

end number_of_large_posters_is_5_l193_193426


namespace relation_xy_l193_193694

theorem relation_xy (a c b d : ℝ) (x y p : ℝ) 
  (h1 : a^x = c^(3 * p))
  (h2 : c^(3 * p) = b^2)
  (h3 : c^y = b^p)
  (h4 : b^p = d^3) :
  y = 3 * p^2 / 2 :=
by
  sorry

end relation_xy_l193_193694


namespace average_books_per_student_l193_193250

theorem average_books_per_student
  (total_students : ℕ)
  (students_0_books : ℕ)
  (students_1_book : ℕ)
  (students_2_books : ℕ)
  (students_at_least_3_books : ℕ)
  (total_students_eq : total_students = 38)
  (students_0_books_eq : students_0_books = 2)
  (students_1_book_eq : students_1_book = 12)
  (students_2_books_eq : students_2_books = 10)
  (students_at_least_3_books_eq : students_at_least_3_books = 14)
  (students_count_consistent : total_students = students_0_books + students_1_book + students_2_books + students_at_least_3_books) :
  (students_0_books * 0 + students_1_book * 1 + students_2_books * 2 + students_at_least_3_books * 3 : ℝ) / total_students = 1.947 :=
by
  sorry

end average_books_per_student_l193_193250


namespace seats_required_l193_193983

def children := 58
def per_seat := 2
def seats_needed (children : ℕ) (per_seat : ℕ) := children / per_seat

theorem seats_required : seats_needed children per_seat = 29 := 
by
  sorry

end seats_required_l193_193983


namespace parabola_vertex_l193_193631

theorem parabola_vertex (c d : ℝ) (h : ∀ (x : ℝ), (-x^2 + c * x + d ≤ 0) ↔ (x ≤ -5 ∨ x ≥ 3)) :
  (∃ a b : ℝ, a = 4 ∧ b = 1 ∧ (-x^2 + c * x + d = -x^2 + 8 * x - 15)) :=
by
  sorry

end parabola_vertex_l193_193631


namespace stadium_seating_and_revenue_l193_193388

   def children := 52
   def adults := 29
   def seniors := 15
   def seats_A := 40
   def seats_B := 30
   def seats_C := 25
   def price_A := 10
   def price_B := 15
   def price_C := 20
   def total_seats := 95

   def revenue_A := seats_A * price_A
   def revenue_B := seats_B * price_B
   def revenue_C := seats_C * price_C
   def total_revenue := revenue_A + revenue_B + revenue_C

   theorem stadium_seating_and_revenue :
     (children <= seats_B + seats_C) ∧
     (adults + seniors <= seats_A + seats_C) ∧
     (children + adults + seniors > total_seats) →
     (revenue_A = 400) ∧
     (revenue_B = 450) ∧
     (revenue_C = 500) ∧
     (total_revenue = 1350) :=
   by
     sorry
   
end stadium_seating_and_revenue_l193_193388


namespace arithmetic_seq_common_diff_l193_193935

theorem arithmetic_seq_common_diff (a b : ℕ) (d : ℕ) (a1 a2 a8 a9 : ℕ) 
  (h1 : a1 + a8 = 10)
  (h2 : a2 + a9 = 18)
  (h3 : a2 = a1 + d)
  (h4 : a8 = a1 + 7 * d)
  (h5 : a9 = a1 + 8 * d)
  : d = 4 :=
by
  sorry

end arithmetic_seq_common_diff_l193_193935


namespace polygon_sides_l193_193399

theorem polygon_sides (h1 : 1260 - 360 = 900) (h2 : (n - 2) * 180 = 900) : n = 7 :=
by 
  sorry

end polygon_sides_l193_193399


namespace empty_cistern_time_l193_193642

variable (t_fill : ℝ) (t_empty₁ : ℝ) (t_empty₂ : ℝ) (t_empty₃ : ℝ)

theorem empty_cistern_time
  (h_fill : t_fill = 3.5)
  (h_empty₁ : t_empty₁ = 14)
  (h_empty₂ : t_empty₂ = 16)
  (h_empty₃ : t_empty₃ = 18) :
  1008 / (1/t_empty₁ + 1/t_empty₂ + 1/t_empty₃) = 1.31979 := by
  sorry

end empty_cistern_time_l193_193642


namespace total_tickets_sold_l193_193086

theorem total_tickets_sold 
(adult_ticket_price : ℕ) (child_ticket_price : ℕ) 
(total_revenue : ℕ) (adult_tickets_sold : ℕ) 
(child_tickets_sold : ℕ) (total_tickets : ℕ) : 
adult_ticket_price = 5 → 
child_ticket_price = 2 → 
total_revenue = 275 → 
adult_tickets_sold = 35 → 
(child_tickets_sold * child_ticket_price) + (adult_tickets_sold * adult_ticket_price) = total_revenue →
total_tickets = adult_tickets_sold + child_tickets_sold →
total_tickets = 85 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end total_tickets_sold_l193_193086


namespace apples_rate_per_kg_l193_193623

variable (A : ℝ)

theorem apples_rate_per_kg (h : 8 * A + 9 * 65 = 1145) : A = 70 :=
sorry

end apples_rate_per_kg_l193_193623


namespace peter_remaining_money_l193_193833

theorem peter_remaining_money (initial_money : ℕ) 
                             (potato_cost_per_kilo : ℕ) (potato_kilos : ℕ)
                             (tomato_cost_per_kilo : ℕ) (tomato_kilos : ℕ)
                             (cucumber_cost_per_kilo : ℕ) (cucumber_kilos : ℕ)
                             (banana_cost_per_kilo : ℕ) (banana_kilos : ℕ) :
  initial_money = 500 →
  potato_cost_per_kilo = 2 → potato_kilos = 6 →
  tomato_cost_per_kilo = 3 → tomato_kilos = 9 →
  cucumber_cost_per_kilo = 4 → cucumber_kilos = 5 →
  banana_cost_per_kilo = 5 → banana_kilos = 3 →
  initial_money - (potato_cost_per_kilo * potato_kilos + 
                   tomato_cost_per_kilo * tomato_kilos +
                   cucumber_cost_per_kilo * cucumber_kilos +
                   banana_cost_per_kilo * banana_kilos) = 426 := by
  sorry

end peter_remaining_money_l193_193833


namespace max_cables_cut_l193_193558

theorem max_cables_cut (computers cables clusters : ℕ) (h_computers : computers = 200) (h_cables : cables = 345) (h_clusters : clusters = 8) :
  ∃ k : ℕ, k = cables - (computers - clusters + 1) ∧ k = 153 :=
by
  sorry

end max_cables_cut_l193_193558


namespace negation_of_universal_l193_193804

theorem negation_of_universal: (¬(∀ x : ℝ, x > 1 → x^2 > 1)) ↔ (∃ x : ℝ, x > 1 ∧ x^2 ≤ 1) :=
by 
  sorry

end negation_of_universal_l193_193804


namespace expression_zero_denominator_nonzero_l193_193503

theorem expression_zero (x : ℝ) : 
  (2 * x - 6) = 0 ↔ x = 3 :=
by {
  sorry
  }

theorem denominator_nonzero (x : ℝ) : 
  x = 3 → (5 * x + 10) ≠ 0 :=
by {
  sorry
  }

end expression_zero_denominator_nonzero_l193_193503


namespace part_I_part_II_l193_193308

noncomputable def f (x : ℝ) : ℝ := abs (x - 2) + abs (x + 1)

theorem part_I (x : ℝ) : f x > 4 ↔ x < -1.5 ∨ x > 2.5 := 
sorry

theorem part_II (a : ℝ) : (∀ x, f x ≥ a) ↔ a ≤ 3 := 
sorry

end part_I_part_II_l193_193308


namespace not_hyperbola_condition_l193_193460

theorem not_hyperbola_condition (m : ℝ) (x y : ℝ) (h1 : 1 ≤ m) (h2 : m ≤ 3) :
  (m - 1) * x^2 + (3 - m) * y^2 = (m - 1) * (3 - m) :=
sorry

end not_hyperbola_condition_l193_193460


namespace max_perimeter_of_right_angled_quadrilateral_is_4rsqrt2_l193_193158

noncomputable def max_perimeter_of_right_angled_quadrilateral (r : ℝ) : ℝ :=
  4 * r * Real.sqrt 2

theorem max_perimeter_of_right_angled_quadrilateral_is_4rsqrt2
  (r : ℝ) :
  ∃ (k : ℝ), 
  (∀ (x y : ℝ), x^2 + y^2 = 4 * r^2 → 2 * (x + y) ≤ max_perimeter_of_right_angled_quadrilateral r)
  ∧ (k = max_perimeter_of_right_angled_quadrilateral r) :=
sorry

end max_perimeter_of_right_angled_quadrilateral_is_4rsqrt2_l193_193158


namespace find_k_parallel_lines_l193_193473

theorem find_k_parallel_lines (k : ℝ) : 
  (∀ x y, (k - 1) * x + y + 2 = 0 → 
            (8 * x + (k + 1) * y + k - 1 = 0 → False)) → 
  k = 3 :=
sorry

end find_k_parallel_lines_l193_193473


namespace winning_ticket_probability_l193_193449

open BigOperators

-- Calculate n choose k
def choose (n k : ℕ) : ℕ :=
  n.factorial / (k.factorial * (n - k).factorial)

-- Given conditions
def probability_PowerBall := (1 : ℚ) / 30
def probability_LuckyBalls := (1 : ℚ) / choose 49 6

-- Theorem to prove the result
theorem winning_ticket_probability :
  probability_PowerBall * probability_LuckyBalls = (1 : ℚ) / 419514480 := by
  sorry

end winning_ticket_probability_l193_193449


namespace vacation_expenses_split_l193_193407

theorem vacation_expenses_split
  (A : ℝ) (B : ℝ) (C : ℝ) (a : ℝ) (b : ℝ)
  (hA : A = 180)
  (hB : B = 240)
  (hC : C = 120)
  (ha : a = 0)
  (hb : b = 0)
  : a - b = 0 := 
by
  sorry

end vacation_expenses_split_l193_193407


namespace intersection_complement_l193_193343

def U : Set ℤ := Set.univ
def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | x^2 ≠ x}
def C_U_B : Set ℤ := {x | x ≠ 0 ∧ x ≠ 1}

theorem intersection_complement :
  A ∩ C_U_B = {-1, 2} :=
by
  sorry

end intersection_complement_l193_193343


namespace compute_expression_l193_193861

-- Definition of the expression
def expression := 5 + 4 * (4 - 9)^2

-- Statement of the theorem, asserting the expression equals 105
theorem compute_expression : expression = 105 := by
  sorry

end compute_expression_l193_193861


namespace average_of_original_set_l193_193900

theorem average_of_original_set (A : ℝ) (h1 : (35 * A) = (7 * 75)) : A = 15 := 
by sorry

end average_of_original_set_l193_193900


namespace seating_arrangements_l193_193427

-- Number of ways to arrange a block of n items
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Groups
def dodgers : ℕ := 4
def marlins : ℕ := 3
def phillies : ℕ := 2

-- Total number of players
def total_players : ℕ := dodgers + marlins + phillies

-- Number of ways to arrange the blocks
def blocks_arrangements : ℕ := factorial 3

-- Internal arrangements within each block
def dodgers_arrangements : ℕ := factorial dodgers
def marlins_arrangements : ℕ := factorial marlins
def phillies_arrangements : ℕ := factorial phillies

-- Total number of ways to seat the players
def total_arrangements : ℕ :=
  blocks_arrangements * dodgers_arrangements * marlins_arrangements * phillies_arrangements

-- Prove that the total arrangements is 1728
theorem seating_arrangements : total_arrangements = 1728 := by
  sorry

end seating_arrangements_l193_193427


namespace find_triples_l193_193415

theorem find_triples (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + b^2 = c^2) ∧ (a^3 + b^3 + 1 = (c-1)^3) ↔ (a = 6 ∧ b = 8 ∧ c = 10) ∨ (a = 8 ∧ b = 6 ∧ c = 10) :=
by
  sorry

end find_triples_l193_193415


namespace integer_solution_count_l193_193584

theorem integer_solution_count :
  ∃ n : ℕ, n = 10 ∧
  ∃ (x1 x2 x3 : ℕ), x1 + x2 + x3 = 15 ∧ (0 ≤ x1 ∧ x1 ≤ 5) ∧ (0 ≤ x2 ∧ x2 ≤ 6) ∧ (0 ≤ x3 ∧ x3 ≤ 7) := 
sorry

end integer_solution_count_l193_193584


namespace danielles_rooms_l193_193938

variable (rooms_heidi rooms_danielle : ℕ)

theorem danielles_rooms 
  (h1 : rooms_heidi = 3 * rooms_danielle)
  (h2 : 2 = 1 / 9 * rooms_heidi) :
  rooms_danielle = 6 := by
  -- Proof omitted
  sorry

end danielles_rooms_l193_193938


namespace reconstruct_quadrilateral_l193_193408

def quadrilateralVectors (W W' X X' Y Y' Z Z' : ℝ) :=
  (W - Z = W/2 + Z'/2) ∧
  (X - Y = Y'/2 + X'/2) ∧
  (Y - X = Y'/2 + X'/2) ∧
  (Z - W = W/2 + Z'/2)

theorem reconstruct_quadrilateral (W W' X X' Y Y' Z Z' : ℝ) :
  quadrilateralVectors W W' X X' Y Y' Z Z' →
  W = (1/2) * W' + 0 * X' + 0 * Y' + (1/2) * Z' :=
sorry

end reconstruct_quadrilateral_l193_193408


namespace jill_spending_on_clothing_l193_193277

theorem jill_spending_on_clothing (C : ℝ) (T : ℝ)
  (h1 : 0.2 * T = 0.2 * T)
  (h2 : 0.3 * T = 0.3 * T)
  (h3 : (C / 100) * T * 0.04 + 0.3 * T * 0.08 = 0.044 * T) :
  C = 50 :=
by
  -- This line indicates the point where the proof would typically start
  sorry

end jill_spending_on_clothing_l193_193277


namespace correct_relation_l193_193253

def A : Set ℝ := { x | x > 1 }

theorem correct_relation : 2 ∈ A := by
  -- Proof would go here
  sorry

end correct_relation_l193_193253


namespace circle_regions_division_l193_193882

theorem circle_regions_division (radii : ℕ) (con_circles : ℕ)
  (h1 : radii = 16) (h2 : con_circles = 10) :
  radii * (con_circles + 1) = 176 := 
by
  -- placeholder for proof
  sorry

end circle_regions_division_l193_193882


namespace total_is_83_l193_193218

def number_of_pirates := 45
def number_of_noodles := number_of_pirates - 7
def total_number_of_noodles_and_pirates := number_of_noodles + number_of_pirates

theorem total_is_83 : total_number_of_noodles_and_pirates = 83 := by
  sorry

end total_is_83_l193_193218


namespace find_y_for_two_thirds_l193_193256

theorem find_y_for_two_thirds (x y : ℝ) (h₁ : (2 / 3) * x + y = 10) (h₂ : x = 6) : y = 6 :=
by
  sorry

end find_y_for_two_thirds_l193_193256


namespace maximal_value_of_product_l193_193923

theorem maximal_value_of_product (m n : ℤ)
  (h1 : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (1 < x1 ∧ x1 < 3) ∧ (1 < x2 ∧ x2 < 3) ∧ 
    ∀ x : ℝ, (10 * x^2 + m * x + n) = 10 * (x - x1) * (x - x2)) :
  (∃ f1 f3 : ℝ, f1 = 10 * (1 - x1) * (1 - x2) ∧ f3 = 10 * (3 - x1) * (3 - x2) ∧ (f1 * f3 = 99)) := 
sorry

end maximal_value_of_product_l193_193923


namespace derivative_at_one_is_three_l193_193646

-- Definition of the function
def f (x : ℝ) := (x - 1)^2 + 3 * (x - 1)

-- The statement of the problem
theorem derivative_at_one_is_three : deriv f 1 = 3 := 
  sorry

end derivative_at_one_is_three_l193_193646


namespace parametric_line_eq_l193_193206

theorem parametric_line_eq (t : ℝ) :
  ∃ t : ℝ, ∃ x : ℝ, ∃ y : ℝ, 
  (x = 3 * t + 5) ∧ (y = 6 * t - 7) → y = 2 * x - 17 :=
by
  sorry

end parametric_line_eq_l193_193206


namespace game_C_more_likely_than_game_D_l193_193713

-- Definitions for the probabilities
def p_heads : ℚ := 3 / 4
def p_tails : ℚ := 1 / 4

-- Game C probability
def p_game_C : ℚ := p_heads ^ 4

-- Game D probabilities for each scenario
def p_game_D_scenario1 : ℚ := (p_heads ^ 3) * (p_heads ^ 2)
def p_game_D_scenario2 : ℚ := (p_heads ^ 3) * (p_tails ^ 2)
def p_game_D_scenario3 : ℚ := (p_tails ^ 3) * (p_heads ^ 2)
def p_game_D_scenario4 : ℚ := (p_tails ^ 3) * (p_tails ^ 2)

-- Total probability for Game D
def p_game_D : ℚ :=
  p_game_D_scenario1 + p_game_D_scenario2 + p_game_D_scenario3 + p_game_D_scenario4

-- Proof statement
theorem game_C_more_likely_than_game_D : (p_game_C - p_game_D) = 11 / 256 := by
  sorry

end game_C_more_likely_than_game_D_l193_193713


namespace find_a_l193_193215

theorem find_a (a : ℝ) : (∃ x y : ℝ, 3 * x + a * y - 5 = 0 ∧ x = 1 ∧ y = 2) → a = 1 :=
by
  intro h
  match h with
  | ⟨x, y, hx, hx1, hy2⟩ => 
    have h1 : x = 1 := hx1
    have h2 : y = 2 := hy2
    rw [h1, h2] at hx
    sorry

end find_a_l193_193215


namespace eval_expr_equals_1_l193_193071

noncomputable def eval_expr (a b : ℕ) : ℚ :=
  (a + b) / (a * b) / ((a / b) - (b / a))

theorem eval_expr_equals_1 (a b : ℕ) (h₁ : a = 3) (h₂ : b = 2) : eval_expr a b = 1 :=
by
  sorry

end eval_expr_equals_1_l193_193071


namespace area_of_triangle_formed_by_lines_l193_193285

def line1 (x : ℝ) : ℝ := 5
def line2 (x : ℝ) : ℝ := 1 + x
def line3 (x : ℝ) : ℝ := 1 - x

theorem area_of_triangle_formed_by_lines :
  let A := (4, 5)
  let B := (-4, 5)
  let C := (0, 1)
  (1 / 2) * abs (4 * 5 + (-4) * 1 + 0 * 5 - (5 * (-4) + 1 * 4 + 5 * 0)) = 16 := by
  sorry

end area_of_triangle_formed_by_lines_l193_193285


namespace two_pt_seven_five_as_fraction_l193_193910

-- Define the decimal value 2.75
def decimal_value : ℚ := 11 / 4

-- Define the question
theorem two_pt_seven_five_as_fraction : 2.75 = decimal_value := by
  sorry

end two_pt_seven_five_as_fraction_l193_193910


namespace rectangle_area_error_l193_193288

/-
  Problem: 
  Given:
  1. One side of the rectangle is taken 20% in excess.
  2. The other side of the rectangle is taken 10% in deficit.
  Prove:
  The error percentage in the calculated area is 8%.
-/

noncomputable def error_percentage (L W : ℝ) := 
  let actual_area : ℝ := L * W
  let measured_length : ℝ := 1.20 * L
  let measured_width : ℝ := 0.90 * W
  let measured_area : ℝ := measured_length * measured_width
  ((measured_area - actual_area) / actual_area) * 100

theorem rectangle_area_error
  (L W : ℝ) : error_percentage L W = 8 := 
  sorry

end rectangle_area_error_l193_193288


namespace reciprocal_sum_of_roots_l193_193139

theorem reciprocal_sum_of_roots
  (a b c : ℝ)
  (ha : a^3 - 2022 * a + 1011 = 0)
  (hb : b^3 - 2022 * b + 1011 = 0)
  (hc : c^3 - 2022 * c + 1011 = 0)
  (distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  (1 / a) + (1 / b) + (1 / c) = 2 :=
sorry

end reciprocal_sum_of_roots_l193_193139


namespace usual_time_catch_bus_l193_193469

variable (S T T' : ℝ)

theorem usual_time_catch_bus (h1 : T' = T + 6)
  (h2 : S * T = (4 / 5) * S * T') : T = 24 := by
  sorry

end usual_time_catch_bus_l193_193469


namespace min_coins_needed_l193_193492

-- Definitions for coins
def coins (pennies nickels dimes quarters : Nat) : Nat :=
  pennies + nickels + dimes + quarters

-- Condition: minimum number of coins to pay any amount less than a dollar
def can_pay_any_amount (pennies nickels dimes quarters : Nat) : Prop :=
  ∀ (amount : Nat), 1 ≤ amount ∧ amount < 100 →
  ∃ (p n d q : Nat), p ≤ pennies ∧ n ≤ nickels ∧ d ≤ dimes ∧ q ≤ quarters ∧
  p + 5 * n + 10 * d + 25 * q = amount

-- The main Lean 4 statement
theorem min_coins_needed :
  ∃ (pennies nickels dimes quarters : Nat),
    coins pennies nickels dimes quarters = 11 ∧
    can_pay_any_amount pennies nickels dimes quarters :=
sorry

end min_coins_needed_l193_193492


namespace problem_1_problem_2_l193_193795

noncomputable def f (x m : ℝ) := |x - 4 / m| + |x + m|

theorem problem_1 (m : ℝ) (hm : 0 < m) (x : ℝ) : f x m ≥ 4 := sorry

theorem problem_2 (m : ℝ) (hm : f 2 m > 5) : 
  m ∈ Set.Ioi ((1 + Real.sqrt 17) / 2) ∪ Set.Ioo 0 1 := sorry

end problem_1_problem_2_l193_193795


namespace temperature_on_Monday_l193_193062

theorem temperature_on_Monday 
  (M T W Th F : ℝ)
  (h1 : (M + T + W + Th) / 4 = 48)
  (h2 : (T + W + Th + F) / 4 = 46)
  (h3 : F = 31) : 
  M = 39 :=
by
  sorry

end temperature_on_Monday_l193_193062


namespace largest_consecutive_odd_nat_divisible_by_3_sum_72_l193_193316

theorem largest_consecutive_odd_nat_divisible_by_3_sum_72
  (a : ℕ)
  (h₁ : a % 3 = 0)
  (h₂ : (a + 6) % 3 = 0)
  (h₃ : (a + 12) % 3 = 0)
  (h₄ : a % 2 = 1)
  (h₅ : (a + 6) % 2 = 1)
  (h₆ : (a + 12) % 2 = 1)
  (h₇ : a + (a + 6) + (a + 12) = 72) :
  a + 12 = 30 :=
by
  sorry

end largest_consecutive_odd_nat_divisible_by_3_sum_72_l193_193316


namespace smallest_base10_integer_l193_193553

theorem smallest_base10_integer : 
  ∃ (a b x : ℕ), a > 2 ∧ b > 2 ∧ x = 2 * a + 1 ∧ x = b + 2 ∧ x = 7 := by
  sorry

end smallest_base10_integer_l193_193553


namespace distance_travelled_l193_193366

def actual_speed : ℝ := 50
def additional_speed : ℝ := 25
def time_difference : ℝ := 0.5

theorem distance_travelled (D : ℝ) : 0.5 = (D / actual_speed) - (D / (actual_speed + additional_speed)) → D = 75 :=
by sorry

end distance_travelled_l193_193366


namespace race_distance_A_beats_C_l193_193386

variables (race_distance1 race_distance2 race_distance3 : ℕ)
           (distance_AB distance_BC distance_AC : ℕ)

theorem race_distance_A_beats_C :
  race_distance1 = 500 →
  race_distance2 = 500 →
  distance_AB = 50 →
  distance_BC = 25 →
  distance_AC = 58 →
  race_distance3 = 400 :=
by
  sorry

end race_distance_A_beats_C_l193_193386


namespace average_lifespan_is_1013_l193_193418

noncomputable def first_factory_lifespan : ℕ := 980
noncomputable def second_factory_lifespan : ℕ := 1020
noncomputable def third_factory_lifespan : ℕ := 1032

noncomputable def total_samples : ℕ := 100

noncomputable def first_samples : ℕ := (1 * total_samples) / 4
noncomputable def second_samples : ℕ := (2 * total_samples) / 4
noncomputable def third_samples : ℕ := (1 * total_samples) / 4

noncomputable def weighted_average_lifespan : ℕ :=
  ((first_factory_lifespan * first_samples) + (second_factory_lifespan * second_samples) + (third_factory_lifespan * third_samples)) / total_samples

theorem average_lifespan_is_1013 : weighted_average_lifespan = 1013 := by
  sorry

end average_lifespan_is_1013_l193_193418


namespace find_a_b_l193_193949

theorem find_a_b (a b : ℝ) (h : (a - 2) ^ 2 + |b + 4| = 0) : a + b = -2 :=
sorry

end find_a_b_l193_193949


namespace range_of_a_l193_193914

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ≤ 1 → 1 + 2^x + 4^x * a > 0) ↔ a > -3/4 := 
sorry

end range_of_a_l193_193914


namespace carla_final_payment_l193_193876

variable (OriginalCost : ℝ) (Coupon : ℝ) (DiscountRate : ℝ)

theorem carla_final_payment
  (h1 : OriginalCost = 7.50)
  (h2 : Coupon = 2.50)
  (h3 : DiscountRate = 0.20) :
  (OriginalCost - Coupon - DiscountRate * (OriginalCost - Coupon)) = 4.00 := 
sorry

end carla_final_payment_l193_193876


namespace bruce_bank_ratio_l193_193334

noncomputable def bruce_aunt : ℝ := 75
noncomputable def bruce_grandfather : ℝ := 150
noncomputable def bruce_bank : ℝ := 45
noncomputable def bruce_total : ℝ := bruce_aunt + bruce_grandfather
noncomputable def bruce_ratio : ℝ := bruce_bank / bruce_total

theorem bruce_bank_ratio :
  bruce_ratio = 1 / 5 :=
by
  -- proof goes here
  sorry

end bruce_bank_ratio_l193_193334


namespace negation_of_proposition_l193_193114

open Classical

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x^2 + 1 > 0)) ↔ (∃ x : ℝ, x^2 + 1 ≤ 0) :=
by
  sorry

end negation_of_proposition_l193_193114


namespace max_value_m_l193_193477

noncomputable def exists_triangle_with_sides (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem max_value_m (a b c : ℝ) (m : ℝ) (h1 : 0 < m) (h2 : abc ≤ 1/4) (h3 : 1/(a^2) + 1/(b^2) + 1/(c^2) < m) :
  m ≤ 9 ↔ exists_triangle_with_sides a b c :=
sorry

end max_value_m_l193_193477


namespace vip_seat_cost_l193_193735

theorem vip_seat_cost
  (V : ℝ)
  (G V_T : ℕ)
  (h1 : 20 * G + V * V_T = 7500)
  (h2 : G + V_T = 320)
  (h3 : V_T = G - 276) :
  V = 70 := by
sorry

end vip_seat_cost_l193_193735


namespace consumer_credit_amount_l193_193182

theorem consumer_credit_amount
  (C A : ℝ)
  (h1 : A = 0.20 * C)
  (h2 : 57 = 1/3 * A) :
  C = 855 := by
  sorry

end consumer_credit_amount_l193_193182


namespace largest_number_l193_193603

theorem largest_number (a b c : ℤ) 
  (h_sum : a + b + c = 67)
  (h_diff1 : c - b = 7)
  (h_diff2 : b - a = 3)
  : c = 28 :=
sorry

end largest_number_l193_193603


namespace find_b7_l193_193318

/-- We represent the situation with twelve people in a circle, each with an integer number. The
     average announced by a person is the average of their two immediate neighbors. Given the
     person who announced the average of 7, we aim to find the number they initially chose. --/
theorem find_b7 (b : ℕ → ℕ) (announced_avg : ℕ → ℕ) :
  (announced_avg 1 = (b 12 + b 2) / 2) ∧
  (announced_avg 2 = (b 1 + b 3) / 2) ∧
  (announced_avg 3 = (b 2 + b 4) / 2) ∧
  (announced_avg 4 = (b 3 + b 5) / 2) ∧
  (announced_avg 5 = (b 4 + b 6) / 2) ∧
  (announced_avg 6 = (b 5 + b 7) / 2) ∧
  (announced_avg 7 = (b 6 + b 8) / 2) ∧
  (announced_avg 8 = (b 7 + b 9) / 2) ∧
  (announced_avg 9 = (b 8 + b 10) / 2) ∧
  (announced_avg 10 = (b 9 + b 11) / 2) ∧
  (announced_avg 11 = (b 10 + b 12) / 2) ∧
  (announced_avg 12 = (b 11 + b 1) / 2) ∧
  (announced_avg 7 = 7) →
  b 7 = 12 := 
sorry

end find_b7_l193_193318


namespace quadratic_function_value_at_neg_one_l193_193066

theorem quadratic_function_value_at_neg_one (b c : ℝ) 
  (h1 : (1:ℝ) ^ 2 + b * 1 + c = 0) 
  (h2 : (3:ℝ) ^ 2 + b * 3 + c = 0) : 
  ((-1:ℝ) ^ 2 + b * (-1) + c = 8) :=
by
  sorry

end quadratic_function_value_at_neg_one_l193_193066


namespace arithmetic_sequence_sum_l193_193051

theorem arithmetic_sequence_sum {a_n : ℕ → ℤ} (d : ℤ) (S : ℕ → ℤ) 
  (h_seq : ∀ n, a_n (n + 1) = a_n n + d)
  (h_sum : ∀ n, S n = (n * (2 * a_n 1 + (n - 1) * d)) / 2)
  (h_condition : a_n 1 = 2 * a_n 3 - 3) : 
  S 9 = 27 :=
sorry

end arithmetic_sequence_sum_l193_193051


namespace max_elements_X_l193_193954

structure GameState where
  fire : Nat
  stone : Nat
  metal : Nat

def canCreateX (state : GameState) (x : Nat) : Bool :=
  state.metal >= x ∧ state.fire >= 2 * x ∧ state.stone >= 3 * x

def maxCreateX (state : GameState) : Nat :=
  if h : canCreateX state 14 then 14 else 0 -- we would need to show how to actually maximizing the value

theorem max_elements_X : maxCreateX ⟨50, 50, 0⟩ = 14 := 
by 
  -- Proof would go here, showing via the conditions given above
  -- We would need to show no more than 14 can be created given the initial resources
  sorry

end max_elements_X_l193_193954


namespace find_total_worth_of_stock_l193_193491

theorem find_total_worth_of_stock (X : ℝ)
  (h1 : 0.20 * X * 0.10 = 0.02 * X)
  (h2 : 0.80 * X * 0.05 = 0.04 * X)
  (h3 : 0.04 * X - 0.02 * X = 200) :
  X = 10000 :=
sorry

end find_total_worth_of_stock_l193_193491


namespace max_rectangles_1x2_l193_193979

-- Define the problem conditions
def single_cell_squares : Type := sorry
def rectangles_1x2 (figure : single_cell_squares) : Prop := sorry

-- State the maximum number theorem
theorem max_rectangles_1x2 (figure : single_cell_squares) (h : rectangles_1x2 figure) :
  ∃ (n : ℕ), n ≤ 5 ∧ ∀ m : ℕ, rectangles_1x2 figure ∧ m ≤ 5 → m = 5 :=
sorry

end max_rectangles_1x2_l193_193979


namespace runner_advantage_l193_193853

theorem runner_advantage (x y z : ℝ) (hx_y: y - x = 0.1) (hy_z: z - y = 0.11111111111111111) :
  z - x = 0.21111111111111111 :=
by
  sorry

end runner_advantage_l193_193853


namespace measure_of_angle_Q_in_hexagon_l193_193591

theorem measure_of_angle_Q_in_hexagon :
  ∀ (Q : ℝ),
    (∃ (angles : List ℝ),
      angles = [134, 108, 122, 99, 87] ∧ angles.sum = 550) →
    180 * (6 - 2) - (134 + 108 + 122 + 99 + 87) = 170 → Q = 170 := by
  sorry

end measure_of_angle_Q_in_hexagon_l193_193591


namespace no_integer_pairs_satisfy_equation_l193_193647

theorem no_integer_pairs_satisfy_equation :
  ∀ (m n : ℤ), ¬(m^3 + 10 * m^2 + 11 * m + 2 = 81 * n^3 + 27 * n^2 + 3 * n - 8) :=
by
  sorry

end no_integer_pairs_satisfy_equation_l193_193647


namespace twice_perimeter_is_72_l193_193865

def twice_perimeter_of_square_field (s : ℝ) : ℝ := 2 * 4 * s

theorem twice_perimeter_is_72 (a P : ℝ) (h1 : a = s^2) (h2 : P = 36) 
    (h3 : 6 * a = 6 * (2 * P + 9)) : twice_perimeter_of_square_field s = 72 := 
by
  sorry

end twice_perimeter_is_72_l193_193865


namespace a_1000_value_l193_193317

open Nat

theorem a_1000_value :
  ∃ (a : ℕ → ℤ), (a 1 = 1010) ∧ (a 2 = 1011) ∧ 
  (∀ n ≥ 1, a n + a (n+1) + a (n+2) = 2 * n) ∧ 
  (a 1000 = 1676) :=
sorry

end a_1000_value_l193_193317


namespace projected_increase_in_attendance_l193_193905

variable (A P : ℝ)

theorem projected_increase_in_attendance :
  (0.8 * A = 0.64 * (A + (P / 100) * A)) → P = 25 :=
by
  intro h
  -- Proof omitted
  sorry

end projected_increase_in_attendance_l193_193905


namespace solve_for_y_l193_193857

theorem solve_for_y : ∃ y : ℝ, (2010 + y)^2 = y^2 ∧ y = -1005 :=
by
  sorry

end solve_for_y_l193_193857


namespace arithmetic_seq_sum_div_fifth_term_l193_193242

open Int

/-- The sequence {a_n} is an arithmetic sequence with a non-zero common difference,
    given that a₂ + a₆ = a₈, prove that S₅ / a₅ = 3. -/
theorem arithmetic_seq_sum_div_fifth_term
  (a : ℕ → ℤ)
  (d : ℤ)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_nonzero : d ≠ 0)
  (h_condition : a 2 + a 6 = a 8) :
  ((5 * a 1 + 10 * d) / (a 1 + 4 * d) : ℚ) = 3 := 
by
  sorry

end arithmetic_seq_sum_div_fifth_term_l193_193242


namespace quotient_of_polynomial_l193_193598

theorem quotient_of_polynomial (x : ℤ) :
  (x^6 + 8) = (x - 1) * (x^5 + x^4 + x^3 + x^2 + x + 1) + 9 :=
by { sorry }

end quotient_of_polynomial_l193_193598


namespace find_pairs_l193_193471

theorem find_pairs (p q : ℤ) (a b : ℤ) :
  (p^2 - 4 * q = a^2) ∧ (q^2 - 4 * p = b^2) ↔ 
    (p = 4 ∧ q = 4) ∨ (p = 9 ∧ q = 8) ∨ (p = 8 ∧ q = 9) :=
by
  sorry

end find_pairs_l193_193471


namespace repeating_decimal_fraction_equiv_l193_193279

noncomputable def repeating_decimal_to_fraction (x : ℚ) : Prop :=
  x = 0.4 + 37 / 990

theorem repeating_decimal_fraction_equiv : repeating_decimal_to_fraction (433 / 990) :=
by
  sorry

end repeating_decimal_fraction_equiv_l193_193279


namespace total_pages_read_is_785_l193_193207

-- Definitions based on the conditions in the problem
def pages_read_first_five_days : ℕ := 5 * 52
def pages_read_next_five_days : ℕ := 5 * 63
def pages_read_last_three_days : ℕ := 3 * 70

-- The main statement to prove
theorem total_pages_read_is_785 :
  pages_read_first_five_days + pages_read_next_five_days + pages_read_last_three_days = 785 :=
by
  sorry

end total_pages_read_is_785_l193_193207


namespace domain_of_c_l193_193220

theorem domain_of_c (m : ℝ) :
  (∀ x : ℝ, 7*x^2 - 6*x + m ≠ 0) ↔ (m > (9 / 7)) :=
by
  -- you would typically put the proof here, but we use sorry to skip it
  sorry

end domain_of_c_l193_193220


namespace elsa_data_remaining_l193_193531

variable (data_total : ℕ) (data_youtube : ℕ)

def data_remaining_after_youtube (data_total data_youtube : ℕ) : ℕ := data_total - data_youtube

def data_fraction_spent_on_facebook (data_left : ℕ) : ℕ := (2 * data_left) / 5

theorem elsa_data_remaining
  (h_data_total : data_total = 500)
  (h_data_youtube : data_youtube = 300) :
  data_remaining_after_youtube data_total data_youtube
  - data_fraction_spent_on_facebook (data_remaining_after_youtube data_total data_youtube) 
  = 120 :=
by
  sorry

end elsa_data_remaining_l193_193531


namespace sum_of_final_numbers_l193_193819

theorem sum_of_final_numbers (x y : ℝ) (S : ℝ) (h : x + y = S) : 
  3 * (x + 4) + 3 * (y + 4) = 3 * S + 24 := by
  sorry

end sum_of_final_numbers_l193_193819


namespace solve_for_x_y_l193_193841

theorem solve_for_x_y (x y : ℚ) 
  (h1 : (3 * x + 12 + 2 * y + 18 + 5 * x + 6 * y + (3 * x + y + 16)) / 5 = 60) 
  (h2 : x = 2 * y) : 
  x = 254 / 15 ∧ y = 127 / 15 :=
by
  -- Placeholder for the proof
  sorry

end solve_for_x_y_l193_193841


namespace sitting_break_frequency_l193_193274

theorem sitting_break_frequency (x : ℕ) (h1 : 240 % x = 0) (h2 : 240 / 20 = 12) (h3 : 240 / x + 10 = 12) : x = 120 := 
sorry

end sitting_break_frequency_l193_193274


namespace percentage_spent_on_hats_l193_193223

def total_money : ℕ := 90
def cost_per_scarf : ℕ := 2
def number_of_scarves : ℕ := 18
def cost_of_scarves : ℕ := number_of_scarves * cost_per_scarf
def money_left_for_hats : ℕ := total_money - cost_of_scarves
def number_of_hats : ℕ := 2 * number_of_scarves

theorem percentage_spent_on_hats : 
  (money_left_for_hats : ℝ) / (total_money : ℝ) * 100 = 60 :=
by
  sorry

end percentage_spent_on_hats_l193_193223


namespace cos_periodicity_even_function_property_l193_193018

theorem cos_periodicity_even_function_property (n : ℤ) (h_cos : Real.cos (n * Real.pi / 180) = Real.cos (317 * Real.pi / 180)) (h_range : -180 ≤ n ∧ n ≤ 180) : n = 43 :=
by
  sorry

end cos_periodicity_even_function_property_l193_193018


namespace slope_of_dividing_line_l193_193351

/--
Given a rectangle with vertices at (0,0), (0,4), (5,4), (5,2),
and a right triangle with vertices at (5,2), (7,2), (5,0),
prove that the slope of the line through the origin that divides the area
of this L-shaped region exactly in half is 16/11.
-/
theorem slope_of_dividing_line :
  let rectangle_area := 5 * 4
  let triangle_area := (1 / 2) * 2 * 2
  let total_area := rectangle_area + triangle_area
  let half_area := total_area / 2
  let x_division := half_area / 4
  let slope := 4 / x_division
  slope = 16 / 11 :=
by
  sorry

end slope_of_dividing_line_l193_193351


namespace terminal_side_of_610_deg_is_250_deg_l193_193652

theorem terminal_side_of_610_deg_is_250_deg:
  ∃ k : ℤ, 610 % 360 = 250 := by
  sorry

end terminal_side_of_610_deg_is_250_deg_l193_193652


namespace susie_total_savings_is_correct_l193_193201

variable (initial_amount : ℝ) (year1_addition_pct : ℝ) (year2_addition_pct : ℝ) (interest_rate : ℝ)

def susies_savings (initial_amount year1_addition_pct year2_addition_pct interest_rate : ℝ) : ℝ :=
  let end_of_first_year := initial_amount + initial_amount * year1_addition_pct
  let first_year_interest := end_of_first_year * interest_rate
  let total_after_first_year := end_of_first_year + first_year_interest
  let end_of_second_year := total_after_first_year + total_after_first_year * year2_addition_pct
  let second_year_interest := end_of_second_year * interest_rate
  end_of_second_year + second_year_interest

theorem susie_total_savings_is_correct : 
  susies_savings 200 0.20 0.30 0.05 = 343.98 := 
by
  sorry

end susie_total_savings_is_correct_l193_193201


namespace range_of_a_l193_193043

noncomputable def f (x : ℝ) : ℝ := if x >= 0 then x^2 + 2*x else -(x^2 + 2*(-x))

theorem range_of_a (a : ℝ) (h : f (3 - a^2) > f (2 * a)) : -3 < a ∧ a < 1 := sorry

end range_of_a_l193_193043


namespace cost_of_tea_l193_193948

theorem cost_of_tea (x : ℕ) (h1 : 9 * x < 1000) (h2 : 10 * x > 1100) : x = 111 :=
by
  sorry

end cost_of_tea_l193_193948


namespace total_students_exam_l193_193092

theorem total_students_exam (N T T' T'' : ℕ) (h1 : T = 88 * N) (h2 : T' = T - 8 * 50) 
  (h3 : T' = 92 * (N - 8)) (h4 : T'' = T' - 100) (h5 : T'' = 92 * (N - 9)) : N = 84 :=
by
  sorry

end total_students_exam_l193_193092


namespace complex_number_quadrant_l193_193812

theorem complex_number_quadrant 
  (i : ℂ) (hi : i.im = 1 ∧ i.re = 0)
  (x y : ℝ) 
  (h : (x + i) * i = y - i) : 
  x < 0 ∧ y < 0 := 
sorry

end complex_number_quadrant_l193_193812


namespace remaining_requests_after_7_days_l193_193325

-- Definitions based on the conditions
def dailyRequests : ℕ := 8
def dailyWork : ℕ := 4
def days : ℕ := 7

-- Theorem statement representing our final proof problem
theorem remaining_requests_after_7_days : 
  (dailyRequests * days - dailyWork * days) + dailyRequests * days = 84 := by
  sorry

end remaining_requests_after_7_days_l193_193325


namespace g_triple_composition_l193_193667

def g (n : ℕ) : ℕ :=
if n < 5 then n^2 + 1 else 2 * n + 3

theorem g_triple_composition : g (g (g 3)) = 49 :=
by
  sorry

end g_triple_composition_l193_193667


namespace blueberry_pies_correct_l193_193514

def total_pies := 36
def apple_pie_ratio := 3
def blueberry_pie_ratio := 4
def cherry_pie_ratio := 5

-- Total parts in the ratio
def total_ratio_parts := apple_pie_ratio + blueberry_pie_ratio + cherry_pie_ratio

-- Number of pies per part
noncomputable def pies_per_part := total_pies / total_ratio_parts

-- Number of blueberry pies
noncomputable def blueberry_pies := blueberry_pie_ratio * pies_per_part

theorem blueberry_pies_correct : blueberry_pies = 12 := 
by
  sorry

end blueberry_pies_correct_l193_193514


namespace orange_segments_l193_193852

noncomputable def total_segments (H S B : ℕ) : ℕ :=
  H + S + B

theorem orange_segments
  (H S B : ℕ)
  (h1 : H = 2 * S)
  (h2 : S = B / 5)
  (h3 : B = S + 8) :
  total_segments H S B = 16 := by
  -- proof goes here
  sorry

end orange_segments_l193_193852


namespace geometric_sequence_problem_l193_193761

variable (a : ℕ → ℝ)
variable (r : ℝ) (hpos : ∀ n, 0 < a n)

theorem geometric_sequence_problem
  (hgeom : ∀ n, a (n+1) = a n * r)
  (h_eq : a 1 * a 3 + 2 * a 3 * a 5 + a 5 * a 7 = 4) :
  a 2 + a 6 = 2 :=
sorry

end geometric_sequence_problem_l193_193761


namespace symmetric_trapezoid_construction_possible_l193_193903

-- Define lengths of legs and distance from intersection point
variables (a b : ℝ)

-- Symmetric trapezoid feasibility condition
theorem symmetric_trapezoid_construction_possible : 3 * b > 2 * a := sorry

end symmetric_trapezoid_construction_possible_l193_193903


namespace solve_inequality_l193_193147

-- Define the odd and monotonically decreasing function
noncomputable def f : ℝ → ℝ := sorry

-- Assume the given conditions
axiom odd_f : ∀ x, f (-x) = -f x
axiom decreasing_f : ∀ x y, x < y → y < 0 → f x > f y
axiom f_at_2 : f 2 = 0

-- The proof statement
theorem solve_inequality (x : ℝ) : (x - 1) * f (x + 1) > 0 ↔ -3 < x ∧ x < -1 :=
by
  -- Proof omitted
  sorry

end solve_inequality_l193_193147


namespace scientific_notation_of_4212000_l193_193512

theorem scientific_notation_of_4212000 :
  4212000 = 4.212 * 10^6 :=
by
  sorry

end scientific_notation_of_4212000_l193_193512


namespace arithmetic_sequence_divisible_by_2005_l193_193884

-- Problem Statement
theorem arithmetic_sequence_divisible_by_2005
  (a : ℕ → ℕ) -- Define the arithmetic sequence
  (d : ℕ) -- Common difference
  (h_arith_seq : ∀ n, a (n + 1) = a n + d) -- Arithmetic sequence condition
  (h_product_div_2005 : ∀ n, 2005 ∣ (a n) * (a (n + 31))) -- Given condition on product divisibility
  : ∀ n, 2005 ∣ a n := 
sorry

end arithmetic_sequence_divisible_by_2005_l193_193884


namespace tangency_of_abs_and_circle_l193_193355

theorem tangency_of_abs_and_circle (a : ℝ) (ha_pos : a > 0) (ha_ne_two : a ≠ 2) :
    (y = abs x ∧ ∀ x, y = abs x → x^2 + (y - a)^2 = 2 * (a - 2)^2)
    → (a = 4/3 ∨ a = 4) := sorry

end tangency_of_abs_and_circle_l193_193355


namespace find_smaller_number_l193_193810

theorem find_smaller_number (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 3) : y = 28.5 :=
by
  sorry

end find_smaller_number_l193_193810


namespace weight_of_B_l193_193879

theorem weight_of_B (A B C : ℝ)
  (h1 : (A + B + C) / 3 = 45)
  (h2 : (A + B) / 2 = 40)
  (h3 : (B + C) / 2 = 47) : B = 39 :=
by
  sorry

end weight_of_B_l193_193879


namespace partition_2004_ways_l193_193990

theorem partition_2004_ways : 
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 2004 → 
  ∃! (q r : ℕ), 2004 = q * n + r ∧ 0 ≤ r ∧ r < n :=
by
  sorry

end partition_2004_ways_l193_193990


namespace find_line_and_intersection_l193_193093

def direct_proportion_function (k : ℝ) (x : ℝ) : ℝ :=
  k * x

def shifted_function (k : ℝ) (x b : ℝ) : ℝ :=
  k * x + b

theorem find_line_and_intersection
  (k : ℝ) (b : ℝ) (h₀ : direct_proportion_function k 1 = 2) (h₁ : b = 5) :
  (shifted_function k 1 b = 7) ∧ (shifted_function k (-5/2) b = 0) :=
by
  -- This is just a placeholder to indicate where the proof would go
  sorry

end find_line_and_intersection_l193_193093


namespace atm_withdrawal_cost_l193_193372

theorem atm_withdrawal_cost (x y : ℝ)
  (h1 : 221 = x + 40000 * y)
  (h2 : 485 = x + 100000 * y) :
  (x + 85000 * y) = 419 := by
  sorry

end atm_withdrawal_cost_l193_193372


namespace sequence_periodicity_l193_193118

noncomputable def a : ℕ → ℚ
| 0       => 0
| (n + 1) => (a n - 2) / ((5/4) * a n - 2)

theorem sequence_periodicity : a 2017 = 0 := by
  sorry

end sequence_periodicity_l193_193118


namespace maximum_sum_minimum_difference_l193_193080

-- Definitions based on problem conditions
def is_least_common_multiple (m n lcm: ℕ) : Prop := Nat.lcm m n = lcm
def is_greatest_common_divisor (m n gcd: ℕ) : Prop := Nat.gcd m n = gcd

-- The target theorem to prove
theorem maximum_sum_minimum_difference (x y: ℕ) (h_lcm: is_least_common_multiple x y 2010) (h_gcd: is_greatest_common_divisor x y 2) :
  (x + y = 2012 ∧ x - y = 104 ∨ y - x = 104) :=
by
  sorry

end maximum_sum_minimum_difference_l193_193080


namespace problem_inequality_l193_193587

theorem problem_inequality 
  (a b c d : ℝ)
  (h1 : d > 0)
  (h2 : a ≥ b)
  (h3 : b ≥ c)
  (h4 : c ≥ d)
  (h5 : a * b * c * d = 1) : 
  (1 / (1 + a)) + (1 / (1 + b)) + (1 / (1 + c)) ≥ 3 / (1 + (a * b * c) ^ (1 / 3)) :=
sorry

end problem_inequality_l193_193587


namespace sin_pi_six_minus_two_alpha_l193_193976

theorem sin_pi_six_minus_two_alpha (α : ℝ) (h : Real.sin (π / 3 - α) = 1 / 3) : 
  Real.sin (π / 6 - 2 * α) = - 7 / 9 :=
by
  sorry

end sin_pi_six_minus_two_alpha_l193_193976


namespace coin_change_problem_l193_193248

theorem coin_change_problem (d q h : ℕ) (n : ℕ) 
  (h1 : 2 * d + 5 * q + 10 * h = 240)
  (h2 : d ≥ 1)
  (h3 : q ≥ 1)
  (h4 : h ≥ 1) :
  n = 275 := 
sorry

end coin_change_problem_l193_193248


namespace consecutive_integer_quadratic_l193_193239

theorem consecutive_integer_quadratic :
  ∃ (a b c : ℤ) (x₁ x₂ : ℤ),
  (a * x₁ ^ 2 + b * x₁ + c = 0 ∧ a * x₂ ^ 2 + b * x₂ + c = 0) ∧
  (a = 2 ∧ b = 0 ∧ c = -2) ∨ (a = -2 ∧ b = 0 ∧ c = 2) := sorry

end consecutive_integer_quadratic_l193_193239


namespace cube_split_with_333_l193_193798

theorem cube_split_with_333 (m : ℕ) (h1 : m > 1)
  (h2 : ∃ k : ℕ, (333 = 2 * k + 1) ∧ (333 + 2 * (k - k) + 2) * k = m^3 ) :
  m = 18 := sorry

end cube_split_with_333_l193_193798


namespace eighth_triangular_number_l193_193148

def triangular_number (n: ℕ) : ℕ := n * (n + 1) / 2

theorem eighth_triangular_number : triangular_number 8 = 36 :=
by
  -- Proof here
  sorry

end eighth_triangular_number_l193_193148


namespace who_is_next_to_Boris_l193_193595

noncomputable def arrangement := ℕ → ℕ

-- Definitions for positions
def position (n : ℕ) := n % 6 + 1 -- Modulo to ensure circular arrangement

-- Names mapped to numbers for simplicity
def Arkady := 1
def Boris := 2
def Vera := 3
def Galya := 4
def Danya := 5
def Egor := 6

-- Conditions:
-- 1. Danya stands next to Vera, on her right side.
def cond1 (a : arrangement) := ∃ n, a n = Vera ∧ a (position (n + 1)) = Danya

-- 2. Galya stands opposite Egor.
def cond2 (a : arrangement) := ∃ n, a n = Egor ∧ a (position (n + 3)) = Galya

-- 3. Egor stands next to Danya.
def cond3 (a : arrangement) := ∃ n, a n = Danya ∧ (a (position (n - 1)) = Egor ∨ a (position (n + 1)) = Egor)

-- 4. Arkady and Galya do not want to stand next to each other.
def cond4 (a : arrangement) := ∀ n, ¬(a n = Arkady ∧ (a (position (n - 1)) = Galya ∨ a (position (n + 1)) = Galya))

-- Conclusion: Arkady and Galya are standing next to Boris.
theorem who_is_next_to_Boris (a : arrangement) :
  cond1 a ∧ cond2 a ∧ cond3 a ∧ cond4 a → 
  (∃ n, a n = Boris ∧ ((a (position (n - 1)) = Arkady ∧ a (position (n + 1)) = Galya) ∨ (a (position (n + 1)) = Arkady ∧ a (position (n - 1)) = Galya))) :=
sorry

end who_is_next_to_Boris_l193_193595


namespace sum_of_six_numbers_l193_193699

theorem sum_of_six_numbers:
  ∃ (A B C D E F : ℕ), 
    A > B ∧ B > C ∧ C > D ∧ D > E ∧ E > F ∧
    E > F ∧ C > F ∧ D > F ∧ A + B + C + D + E + F = 141 := 
sorry

end sum_of_six_numbers_l193_193699


namespace problem_solution_l193_193675

noncomputable def problem_expr : ℝ :=
  (64 + 5 * 12) / (180 / 3) + Real.sqrt 49 - 2^3 * Nat.factorial 4

theorem problem_solution : problem_expr = -182.93333333 :=
by 
  sorry

end problem_solution_l193_193675


namespace squares_equal_l193_193718

theorem squares_equal (a b c : ℤ) 
  (h : (a - 5)^2 + (b - 12)^2 - (c - 13)^2 = a^2 + b^2 - c^2) 
    : ∃ (k : ℤ), a^2 + b^2 - c^2 = k^2 := 
by 
  sorry

end squares_equal_l193_193718


namespace percentage_decrease_l193_193677

noncomputable def original_fraction (N D : ℝ) : Prop := N / D = 0.75
noncomputable def new_fraction (N D x : ℝ) : Prop := (1.15 * N) / (D * (1 - x / 100)) = 15 / 16

theorem percentage_decrease (N D x : ℝ) (h1 : original_fraction N D) (h2 : new_fraction N D x) : 
  x = 22.67 := 
sorry

end percentage_decrease_l193_193677


namespace pow_mod_remainder_l193_193432

theorem pow_mod_remainder :
  (2^2013 % 11) = 8 :=
sorry

end pow_mod_remainder_l193_193432


namespace perpendicular_line_passing_point_l193_193358

theorem perpendicular_line_passing_point (x y : ℝ) (hx : 4 * x - 3 * y + 2 = 0) : 
  ∃ l : ℝ → ℝ → Prop, (∀ x y, l x y ↔ (3 * x + 4 * y + 1 = 0) → l 1 2) :=
sorry

end perpendicular_line_passing_point_l193_193358


namespace double_transmission_yellow_twice_double_transmission_less_single_l193_193711

variables {α : ℝ} (hα : 0 < α ∧ α < 1)

-- Statement B
theorem double_transmission_yellow_twice (hα : 0 < α ∧ α < 1) :
  probability_displays_yellow_twice = α^2 :=
sorry

-- Statement D
theorem double_transmission_less_single (hα : 0 < α ∧ α < 1) :
  (1 - α)^2 < (1 - α) :=
sorry

end double_transmission_yellow_twice_double_transmission_less_single_l193_193711


namespace alexander_first_gallery_pictures_l193_193707

def pictures_for_new_galleries := 5 * 2
def pencils_for_new_galleries := pictures_for_new_galleries * 4
def total_exhibitions := 1 + 5
def pencils_for_signing := total_exhibitions * 2
def total_pencils := 88
def pencils_for_first_gallery := total_pencils - pencils_for_new_galleries - pencils_for_signing
def pictures_for_first_gallery := pencils_for_first_gallery / 4

theorem alexander_first_gallery_pictures : pictures_for_first_gallery = 9 :=
by
  sorry

end alexander_first_gallery_pictures_l193_193707


namespace sum_of_digits_next_perfect_square_222_l193_193081

-- Define the condition for the perfect square that begins with "222"
def starts_with_222 (n: ℕ) : Prop :=
  n / 10^3 = 222

-- Define the sum of the digits function
def sum_of_digits (n: ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Statement for the Lean 4 statement: 
-- Prove that the sum of the digits of the next perfect square that starts with "222" is 18
theorem sum_of_digits_next_perfect_square_222 : sum_of_digits (492 ^ 2) = 18 :=
by
  sorry -- Proof omitted

end sum_of_digits_next_perfect_square_222_l193_193081


namespace bus_stops_12_minutes_per_hour_l193_193885

noncomputable def stopping_time (speed_excluding_stoppages : ℝ) (speed_including_stoppages : ℝ) : ℝ :=
  let distance_lost_per_hour := speed_excluding_stoppages - speed_including_stoppages
  let speed_per_minute := speed_excluding_stoppages / 60
  distance_lost_per_hour / speed_per_minute

theorem bus_stops_12_minutes_per_hour :
  stopping_time 50 40 = 12 :=
by
  sorry

end bus_stops_12_minutes_per_hour_l193_193885


namespace candy_count_in_third_set_l193_193192

theorem candy_count_in_third_set
  (L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ)
  (h_totals : L1 + L2 + L3 = S1 + S2 + S3 ∧ S1 + S2 + S3 = M1 + M2 + M3)
  (h1 : L1 = S1 + 7)
  (h2 : S1 = M1)
  (h3 : L2 = S2)
  (h4 : M2 = L2 - 15)
  (h5 : L3 = 0) :
  S3 + M3 + L3 = 29 :=
by
  sorry

end candy_count_in_third_set_l193_193192


namespace panda_bamboo_digestion_l193_193500

theorem panda_bamboo_digestion (h : 16 = 0.40 * x) : x = 40 :=
by sorry

end panda_bamboo_digestion_l193_193500


namespace find_a_maximize_profit_l193_193802

theorem find_a (a: ℕ) (h: 600 * (a - 110) = 160 * a) : a = 150 :=
sorry

theorem maximize_profit (x y: ℕ) (a: ℕ) 
  (ha: a = 150)
  (hx: x + 5 * x + 20 ≤ 200) 
  (profit_eq: ∀ x, y = 245 * x + 600):
  x = 30 ∧ y = 7950 :=
sorry

end find_a_maximize_profit_l193_193802


namespace water_tank_full_capacity_l193_193031

-- Define the conditions
variable {C x : ℝ}
variable (h1 : x / C = 1 / 3)
variable (h2 : (x + 6) / C = 1 / 2)

-- Prove that C = 36
theorem water_tank_full_capacity : C = 36 :=
by
  sorry

end water_tank_full_capacity_l193_193031


namespace function_range_of_roots_l193_193443

theorem function_range_of_roots (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) : a > 1 := 
sorry

end function_range_of_roots_l193_193443


namespace total_ticket_cost_l193_193123

theorem total_ticket_cost (x y : ℕ) 
  (h1 : x + y = 380) 
  (h2 : y = x + 240) 
  (cost_orchestra : ℕ := 12) 
  (cost_balcony : ℕ := 8): 
  12 * x + 8 * y = 3320 := 
by 
  sorry

end total_ticket_cost_l193_193123


namespace cone_prism_ratio_l193_193234

theorem cone_prism_ratio 
  (a b h_c h_p : ℝ) (hb_lt_a : b < a) : 
  (π * b * h_c) / (12 * a * h_p) = (1 / 3 * π * b^2 * h_c) / (4 * a * b * h_p) :=
by
  sorry

end cone_prism_ratio_l193_193234


namespace ellipse_equation_l193_193721

theorem ellipse_equation 
  (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m ≠ n)
  (h4 : ∀ A B : ℝ × ℝ, (m * A.1^2 + n * A.2^2 = 1) ∧ (m * B.1^2 + n * B.2^2 = 1) ∧ (A.1 + A.2 = 1) ∧ (B.1 + B.2 = 1) → dist A B = 2 * (2:ℝ).sqrt)
  (h5 : ∀ A B : ℝ × ℝ, (m * A.1^2 + n * A.2^2 = 1) ∧ (m * B.1^2 + n * B.2^2 = 1) ∧ (A.1 + A.2 = 1) ∧ (B.1 + B.2 = 1) → 
    (A.2 + B.2) / (A.1 + B.1) = (2:ℝ).sqrt / 2) :
  m = 1 / 3 → n = (2:ℝ).sqrt / 3 → 
  (∀ x y : ℝ, (1 / 3) * x^2 + ((2:ℝ).sqrt / 3) * y^2 = 1) :=
by
  sorry

end ellipse_equation_l193_193721


namespace squares_in_ap_l193_193971

theorem squares_in_ap (a b c : ℝ) (h : (1 / (a + b) + 1 / (b + c)) / 2 = 1 / (a + c)) : 
  a^2 + c^2 = 2 * b^2 :=
by
  sorry

end squares_in_ap_l193_193971


namespace train_speed_84_kmph_l193_193981

theorem train_speed_84_kmph (length : ℕ) (time : ℕ) (conversion_factor : ℚ)
  (h_length : length = 140) (h_time : time = 6) (h_conversion_factor : conversion_factor = 3.6) :
  (length / time) * conversion_factor = 84 :=
  sorry

end train_speed_84_kmph_l193_193981


namespace rebecca_swimming_problem_l193_193251

theorem rebecca_swimming_problem :
  ∃ D : ℕ, (D / 4 - D / 5) = 6 → D = 120 :=
sorry

end rebecca_swimming_problem_l193_193251


namespace distance_home_to_school_l193_193096

theorem distance_home_to_school :
  ∃ (D : ℝ) (T : ℝ), 
    3 * (T + 7 / 60) = D ∧
    6 * (T - 8 / 60) = D ∧
    D = 1.5 :=
by
  sorry

end distance_home_to_school_l193_193096


namespace exists_n_gt_2_divisible_by_1991_l193_193826

theorem exists_n_gt_2_divisible_by_1991 :
  ∃ n > 2, 1991 ∣ (2 * 10^(n+1) - 9) :=
by
  existsi (1799 : Nat)
  have h1 : 1799 > 2 := by decide
  have h2 : 1991 ∣ (2 * 10^(1799+1) - 9) := sorry
  constructor
  · exact h1
  · exact h2

end exists_n_gt_2_divisible_by_1991_l193_193826


namespace length_of_train_l193_193988

theorem length_of_train (speed_kmph : ℕ) (bridge_length_m : ℕ) (crossing_time_s : ℕ) 
  (h1 : speed_kmph = 45) (h2 : bridge_length_m = 220) (h3 : crossing_time_s = 30) :
  ∃ train_length_m : ℕ, train_length_m = 155 :=
by
  sorry

end length_of_train_l193_193988


namespace second_root_l193_193716

variables {a b c x : ℝ}

theorem second_root (h : a * (b + c) * x ^ 2 - b * (c + a) * x + c * (a + b) = 0)
(hroot : a * (b + c) * (-1) ^ 2 - b * (c + a) * (-1) + c * (a + b) = 0) :
  ∃ k : ℝ, k = - c * (a + b) / (a * (b + c)) ∧ a * (b + c) * k ^ 2 - b * (c + a) * k + c * (a + b) = 0 :=
sorry

end second_root_l193_193716


namespace greatest_perimeter_of_strips_l193_193759

theorem greatest_perimeter_of_strips :
  let base := 10
  let height := 12
  let half_base := base / 2
  let right_triangle_area := (base / 2 * height) / 2
  let number_of_pieces := 10
  let sub_area := right_triangle_area / (number_of_pieces / 2)
  let h1 := (2 * sub_area) / half_base
  let hypotenuse := Real.sqrt (h1^2 + (half_base / 2)^2)
  let perimeter := half_base + 2 * hypotenuse
  perimeter = 11.934 :=
by
  sorry

end greatest_perimeter_of_strips_l193_193759


namespace archer_total_fish_caught_l193_193185

noncomputable def total_fish_caught (initial : ℕ) (second_extra : ℕ) (third_round_percentage : ℕ) : ℕ :=
  let second_round := initial + second_extra
  let total_after_two_rounds := initial + second_round
  let third_round := second_round + (third_round_percentage * second_round) / 100
  total_after_two_rounds + third_round

theorem archer_total_fish_caught :
  total_fish_caught 8 12 60 = 60 :=
by
  -- Theorem statement to prove the total fish caught equals 60 given the conditions.
  sorry

end archer_total_fish_caught_l193_193185


namespace unique_positive_integers_sum_l193_193880

noncomputable def x : ℝ := Real.sqrt ((Real.sqrt 77) / 3 + 5 / 3)

theorem unique_positive_integers_sum :
  ∃ (a b c : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c),
    x^100 = 3 * x^98 + 17 * x^96 + 13 * x^94 - 2 * x^50 + (a : ℝ) * x^46 + (b : ℝ) * x^44 + (c : ℝ) * x^40
    ∧ a + b + c = 167 := by
  sorry

end unique_positive_integers_sum_l193_193880


namespace max_x1_squared_plus_x2_squared_l193_193122

theorem max_x1_squared_plus_x2_squared (k : ℝ) (x₁ x₂ : ℝ) 
  (h1 : x₁ + x₂ = k - 2)
  (h2 : x₁ * x₂ = k^2 + 3 * k + 5)
  (h3 : -4 ≤ k ∧ k ≤ -4 / 3) :
  x₁ ^ 2 + x₂ ^ 2 ≤ 18 :=
sorry

end max_x1_squared_plus_x2_squared_l193_193122


namespace power_multiplication_l193_193312

theorem power_multiplication (a : ℝ) (b : ℝ) (m : ℕ) (n : ℕ) (h1 : a = 0.25) (h2 : b = 4) (h3 : m = 2023) (h4 : n = 2024) : 
  a^m * b^n = 4 := 
by 
  sorry

end power_multiplication_l193_193312


namespace Hannah_total_spent_l193_193728

def rides_cost (total_money : ℝ) : ℝ :=
  0.35 * total_money

def games_cost (total_money : ℝ) : ℝ :=
  0.25 * total_money

def food_and_souvenirs_cost : ℝ :=
  7 + 4 + 5 + 6

def total_spent (total_money : ℝ) : ℝ :=
  rides_cost total_money + games_cost total_money + food_and_souvenirs_cost

theorem Hannah_total_spent (total_money : ℝ) (h : total_money = 80) :
  total_spent total_money = 70 :=
by
  rw [total_spent, h, rides_cost, games_cost]
  norm_num
  sorry

end Hannah_total_spent_l193_193728
