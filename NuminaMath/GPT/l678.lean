import Mathlib

namespace balanced_polygons_characterization_l678_67852

def convex_polygon (n : ℕ) (vertices : Fin n → Point) : Prop := 
  -- Definition of convex_polygon should go here
  sorry

def is_balanced (n : ℕ) (vertices : Fin n → Point) (M : Point) : Prop := 
  -- Definition of is_balanced should go here
  sorry

theorem balanced_polygons_characterization :
  ∀ (n : ℕ) (vertices : Fin n → Point) (M : Point),
  convex_polygon n vertices →
  is_balanced n vertices M →
  n = 3 ∨ n = 5 ∨ n = 7 :=
by sorry

end balanced_polygons_characterization_l678_67852


namespace smallest_positive_integer_l678_67860

theorem smallest_positive_integer {x : ℕ} (h1 : x % 6 = 3) (h2 : x % 8 = 5) : x = 21 :=
sorry

end smallest_positive_integer_l678_67860


namespace cleaner_for_cat_stain_l678_67853

theorem cleaner_for_cat_stain (c : ℕ) :
  (6 * 6) + (3 * c) + (1 * 1) = 49 → c = 4 :=
by
  sorry

end cleaner_for_cat_stain_l678_67853


namespace max_heaps_of_stones_l678_67856

noncomputable def stones : ℕ := 660
def max_heaps : ℕ := 30
def differs_less_than_twice (a b : ℕ) : Prop := a < 2 * b

theorem max_heaps_of_stones (h : ℕ) :
  (∀ i j : ℕ, i ≠ j → differs_less_than_twice (i+j) stones) → max_heaps = 30 :=
sorry

end max_heaps_of_stones_l678_67856


namespace sum_of_ages_l678_67855

theorem sum_of_ages (M S G : ℕ)
  (h1 : M = 2 * S)
  (h2 : S = 2 * G)
  (h3 : G = 20) :
  M + S + G = 140 :=
sorry

end sum_of_ages_l678_67855


namespace lattice_points_on_hyperbola_l678_67862

-- The hyperbola equation
def hyperbola_eq (x y : ℤ) : Prop :=
  x^2 - y^2 = 1800^2

-- The final number of lattice points lying on the hyperbola
theorem lattice_points_on_hyperbola : 
  ∃ (n : ℕ), n = 250 ∧ (∃ (x y : ℤ), hyperbola_eq x y) :=
sorry

end lattice_points_on_hyperbola_l678_67862


namespace fish_ranking_l678_67882

def ranks (P V K T : ℕ) : Prop :=
  P < K ∧ K < T ∧ T < V

theorem fish_ranking (P V K T : ℕ) (h1 : K < T) (h2 : P + V = K + T) (h3 : P + T < V + K) : ranks P V K T :=
by
  sorry

end fish_ranking_l678_67882


namespace rectangle_area_is_1600_l678_67834

theorem rectangle_area_is_1600 (l w : ℕ) 
  (h₁ : l = 4 * w)
  (h₂ : 2 * l + 2 * w = 200) : 
  l * w = 1600 :=
by sorry

end rectangle_area_is_1600_l678_67834


namespace calculate_expression_l678_67859

theorem calculate_expression : 
  let a := 0.82
  let b := 0.1
  a^3 - b^3 / (a^2 + 0.082 + b^2) = 0.7201 := sorry

end calculate_expression_l678_67859


namespace sample_size_is_80_l678_67831

-- Define the given conditions
variables (x : ℕ) (numA numB numC n : ℕ)

-- Conditions in Lean
def ratio_condition (x numA numB numC : ℕ) : Prop :=
  numA = 2 * x ∧ numB = 3 * x ∧ numC = 5 * x

def sample_condition (numA : ℕ) : Prop :=
  numA = 16

-- Definition of the proof problem
theorem sample_size_is_80 (x : ℕ) (numA numB numC n : ℕ)
  (h_ratio : ratio_condition x numA numB numC)
  (h_sample : sample_condition numA) : 
  n = 80 :=
by
-- The proof is omitted, just state the theorem
sorry

end sample_size_is_80_l678_67831


namespace fraction_subtraction_l678_67846

theorem fraction_subtraction (x y : ℝ) (h : x ≠ y) : (x + y) / (x - y) - (2 * y) / (x - y) = 1 := by
  sorry

end fraction_subtraction_l678_67846


namespace frequency_number_correct_l678_67826

-- Define the sample capacity and the group frequency as constants
def sample_capacity : ℕ := 100
def group_frequency : ℝ := 0.3

-- State the theorem
theorem frequency_number_correct : sample_capacity * group_frequency = 30 := by
  -- Immediate calculation
  sorry

end frequency_number_correct_l678_67826


namespace stamp_solutions_l678_67872

theorem stamp_solutions (n : ℕ) (h1 : ∀ (k : ℕ), k < 115 → ∃ (a b c : ℕ), 
  3 * a + n * b + (n + 1) * c = k) 
  (h2 : ¬ ∃ (a b c : ℕ), 3 * a + n * b + (n + 1) * c = 115) 
  (h3 : ∀ (k : ℕ), 116 ≤ k ∧ k ≤ 120 → ∃ (a b c : ℕ), 
  3 * a + n * b + (n + 1) * c = k) : 
  n = 59 :=
sorry

end stamp_solutions_l678_67872


namespace quadratic_matches_sin_values_l678_67820

noncomputable def quadratic_function (x : ℝ) : ℝ := - (4 / (Real.pi ^ 2)) * (x ^ 2) + (4 / Real.pi) * x

theorem quadratic_matches_sin_values :
  (quadratic_function 0 = Real.sin 0) ∧
  (quadratic_function (Real.pi / 2) = Real.sin (Real.pi / 2)) ∧
  (quadratic_function Real.pi = Real.sin Real.pi) :=
by
  sorry

end quadratic_matches_sin_values_l678_67820


namespace solution_set_of_inequality_l678_67814

theorem solution_set_of_inequality :
  {x : ℝ | 9 * x^2 + 6 * x + 1 ≤ 0} = {-1 / 3} :=
by {
  sorry -- Proof goes here
}

end solution_set_of_inequality_l678_67814


namespace abs_eq_of_sq_eq_l678_67818

theorem abs_eq_of_sq_eq (a b : ℝ) : a^2 = b^2 → |a| = |b| := by
  intro h
  sorry

end abs_eq_of_sq_eq_l678_67818


namespace correct_operation_l678_67857

variable (N : ℚ) -- Original number (assumed rational for simplicity)
variable (x : ℚ) -- Unknown multiplier

theorem correct_operation (h : (N / 10) = (5 / 100) * (N * x)) : x = 2 :=
by
  sorry

end correct_operation_l678_67857


namespace quadratic_trinomial_value_at_6_l678_67848

theorem quadratic_trinomial_value_at_6 {p q : ℝ} 
  (h1 : ∃ r1 r2, r1 = q ∧ r2 = 1 + p + q ∧ r1 + r2 = -p ∧ r1 * r2 = q) : 
  (6^2 + p * 6 + q) = 31 :=
by
  sorry

end quadratic_trinomial_value_at_6_l678_67848


namespace expand_product_l678_67879

theorem expand_product (x : ℝ) : 5 * (x + 2) * (x + 6) * (x - 1) = 5 * x^3 + 35 * x^2 + 20 * x - 60 := 
by
  sorry

end expand_product_l678_67879


namespace price_of_AC_l678_67802

theorem price_of_AC (x : ℝ) (price_car price_ac : ℝ)
  (h1 : price_car = 3 * x) 
  (h2 : price_ac = 2 * x) 
  (h3 : price_car = price_ac + 500) : 
  price_ac = 1000 := sorry

end price_of_AC_l678_67802


namespace triangle_condition_l678_67899

-- Definitions based on the conditions
def angle_equal (A B C : ℝ) : Prop := A = B - C
def angle_ratio123 (A B C : ℝ) : Prop := A / B = 1 / 2 ∧ A / C = 1 / 3 ∧ B / C = 2 / 3
def pythagorean (a b c : ℝ) : Prop := a * a + b * b = c * c
def side_ratio456 (a b c : ℝ) : Prop := a / b = 4 / 5 ∧ a / c = 4 / 6 ∧ b / c = 5 / 6

-- Main hypothesis with right-angle and its conditions in different options
def is_right_triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  (angle_equal A B C → A = 90 ∨ B = 90 ∨ C = 90) ∧
  (angle_ratio123 A B C → A = 30 ∧ B = 60 ∧ C = 90) ∧
  (pythagorean a b c → true) ∧
  (side_ratio456 a b c → false) -- option D cannot confirm the triangle is right

theorem triangle_condition (A B C a b c : ℝ) : is_right_triangle A B C a b c :=
sorry

end triangle_condition_l678_67899


namespace Benjamin_has_45_presents_l678_67839

-- Define the number of presents each person has
def Ethan_presents : ℝ := 31.5
def Alissa_presents : ℝ := Ethan_presents + 22
def Benjamin_presents : ℝ := Alissa_presents - 8.5

-- The statement we need to prove
theorem Benjamin_has_45_presents : Benjamin_presents = 45 :=
by
  -- on the last line, we type sorry to skip the actual proof
  sorry

end Benjamin_has_45_presents_l678_67839


namespace three_pow_2023_mod_eleven_l678_67845

theorem three_pow_2023_mod_eleven :
  (3 ^ 2023) % 11 = 5 :=
sorry

end three_pow_2023_mod_eleven_l678_67845


namespace kiyana_gives_half_l678_67829

theorem kiyana_gives_half (total_grapes : ℕ) (h : total_grapes = 24) : 
  (total_grapes / 2) = 12 :=
by
  sorry

end kiyana_gives_half_l678_67829


namespace friday_profit_l678_67822

noncomputable def total_weekly_profit : ℝ := 2000
noncomputable def profit_on_monday (total : ℝ) : ℝ := total / 3
noncomputable def profit_on_tuesday (total : ℝ) : ℝ := total / 4
noncomputable def profit_on_thursday (total : ℝ) : ℝ := 0.35 * total
noncomputable def profit_on_friday (total : ℝ) : ℝ :=
  total - (profit_on_monday total + profit_on_tuesday total + profit_on_thursday total)

theorem friday_profit (total : ℝ) : profit_on_friday total = 133.33 :=
by
  sorry

end friday_profit_l678_67822


namespace student_in_16th_group_has_number_244_l678_67815

theorem student_in_16th_group_has_number_244 :
  (∀ (n : ℕ), 1 ≤ n ∧ n ≤ 50 → ∃ k : ℕ, 1 ≤ k ∧ k ≤ 800 ∧ ((k - 36) % 16 = 0) ∧ (n = 3 + (k - 36) / 16)) →
  ∃ m : ℕ, 1 ≤ m ∧ m ≤ 800 ∧ ((m - 244) % 16 = 0) ∧ (16 = 3 + (m - 36) / 16) :=
by
  sorry

end student_in_16th_group_has_number_244_l678_67815


namespace remainder_division_1000_l678_67877

theorem remainder_division_1000 (x : ℕ) (hx : x > 0) (h : 100 % x = 10) : 1000 % x = 10 :=
  sorry

end remainder_division_1000_l678_67877


namespace Linda_journey_length_l678_67891

theorem Linda_journey_length : 
  (∃ x : ℝ, x = 30 + x * 1/4 + x * 1/7) → x = 840 / 17 :=
by
  sorry

end Linda_journey_length_l678_67891


namespace chess_tournament_games_l678_67842

-- Define the problem
def total_chess_games (n_players games_per_player : ℕ) : ℕ :=
  (n_players * games_per_player) / 2

-- Conditions: 
-- 1. There are 6 chess amateurs.
-- 2. Each amateur plays exactly 4 games.

theorem chess_tournament_games :
  total_chess_games 6 4 = 10 :=
  sorry

end chess_tournament_games_l678_67842


namespace daily_expenses_increase_l678_67850

theorem daily_expenses_increase 
  (init_students : ℕ) (new_students : ℕ) (diminish_amount : ℝ) (orig_expenditure : ℝ)
  (orig_expenditure_eq : init_students = 35)
  (new_students_eq : new_students = 42)
  (diminish_amount_eq : diminish_amount = 1)
  (orig_expenditure_val : orig_expenditure = 400)
  (orig_average_expenditure : ℝ) (increase_expenditure : ℝ)
  (orig_avg_calc : orig_average_expenditure = orig_expenditure / init_students)
  (new_total_expenditure : ℝ)
  (new_expenditure_eq : new_total_expenditure = orig_expenditure + increase_expenditure) :
  (42 * (orig_average_expenditure - diminish_amount) = new_total_expenditure) → increase_expenditure = 38 := 
by 
  sorry

end daily_expenses_increase_l678_67850


namespace min_value_x_plus_4y_l678_67894

theorem min_value_x_plus_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
(h_cond : (1 / x) + (1 / (2 * y)) = 1) : x + 4 * y = 3 + 2 * Real.sqrt 2 :=
by
  sorry

end min_value_x_plus_4y_l678_67894


namespace quadratic_inequality_false_iff_range_of_a_l678_67849

theorem quadratic_inequality_false_iff_range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + (a - 1) * x + 1 ≤ 0) ↔ (-1 < a ∧ a < 3) :=
sorry

end quadratic_inequality_false_iff_range_of_a_l678_67849


namespace pyramid_volume_l678_67884

theorem pyramid_volume (a : ℝ) (h : a > 0) : (1 / 6) * a^3 = 1 / 6 * a^3 :=
by
  sorry

end pyramid_volume_l678_67884


namespace domain_of_f_l678_67858

noncomputable def f (x : ℝ) : ℝ := 1 / Real.log (x + 1) + Real.sqrt (4 - x^2)

theorem domain_of_f :
  {x : ℝ | 4 - x^2 ≥ 0 ∧ x + 1 > 0 ∧ x + 1 ≠ 1} = {x : ℝ | -1 < x ∧ x ≤ 2 ∧ x ≠ 0} :=
by 
  sorry

end domain_of_f_l678_67858


namespace tan_theta_eq_neg_two_l678_67896

theorem tan_theta_eq_neg_two (f : ℝ → ℝ) (θ : ℝ) 
  (h₁ : ∀ x, f x = Real.sin (2 * x + θ)) 
  (h₂ : ∀ x, f x + 2 * Real.cos (2 * x + θ) = -(f (-x) + 2 * Real.cos (2 * (-x) + θ))) :
  Real.tan θ = -2 :=
by
  sorry

end tan_theta_eq_neg_two_l678_67896


namespace total_distance_traveled_l678_67892

noncomputable def total_distance (d v1 v2 v3 time_total : ℝ) : ℝ :=
  3 * d

theorem total_distance_traveled
  (d : ℝ)
  (v1 : ℝ := 3)
  (v2 : ℝ := 6)
  (v3 : ℝ := 9)
  (time_total : ℝ := 11 / 60)
  (h : d / v1 + d / v2 + d / v3 = time_total) :
  total_distance d v1 v2 v3 time_total = 0.9 :=
by
  sorry

end total_distance_traveled_l678_67892


namespace length_of_train_l678_67840

theorem length_of_train 
  (L V : ℝ) 
  (h1 : L = V * 8) 
  (h2 : L + 279 = V * 20) : 
  L = 186 :=
by
  -- solve using the given conditions
  sorry

end length_of_train_l678_67840


namespace algebraic_expression_perfect_square_l678_67817

theorem algebraic_expression_perfect_square (a : ℤ) :
  (∃ b : ℤ, ∀ x : ℤ, x^2 + (a - 1) * x + 16 = (x + b)^2) →
  (a = 9 ∨ a = -7) :=
sorry

end algebraic_expression_perfect_square_l678_67817


namespace quadratic_real_roots_l678_67871

theorem quadratic_real_roots (k : ℝ) (h : k ≠ 0) : 
  (∃ x : ℝ, k * x^2 - 2 * x - 1 = 0) ∧ (∃ y : ℝ, y ≠ x ∧ k * y^2 - 2 * y - 1 = 0) → k ≥ -1 :=
by
  sorry

end quadratic_real_roots_l678_67871


namespace solve_for_x_l678_67813

theorem solve_for_x (x : ℝ) :
  (x + 3)^3 = -64 → x = -7 :=
by
  intro h
  sorry

end solve_for_x_l678_67813


namespace point_distance_is_pm_3_l678_67890

theorem point_distance_is_pm_3 (Q : ℝ) (h : |Q - 0| = 3) : Q = 3 ∨ Q = -3 :=
sorry

end point_distance_is_pm_3_l678_67890


namespace exists_2013_distinct_numbers_l678_67803

theorem exists_2013_distinct_numbers : 
  ∃ (a : ℕ → ℕ), 
    (∀ m n, m ≠ n → m < 2013 ∧ n < 2013 → (a m + a n) % (a m - a n) = 0) ∧
    (∀ k l, k < 2013 ∧ l < 2013 → (a k) ≠ (a l)) :=
sorry

end exists_2013_distinct_numbers_l678_67803


namespace smaller_circle_radius_l678_67888

theorem smaller_circle_radius
  (R : ℝ) (r : ℝ)
  (h1 : R = 12)
  (h2 : 7 = 7) -- This is trivial and just emphasizes the arrangement of seven congruent smaller circles
  (h3 : 4 * (2 * r) = 2 * R) : r = 3 := by
  sorry

end smaller_circle_radius_l678_67888


namespace tennis_balls_per_can_is_three_l678_67830

-- Definition of the number of games in each round
def games_in_round (round: Nat) : Nat :=
  match round with
  | 1 => 8
  | 2 => 4
  | 3 => 2
  | 4 => 1
  | _ => 0

-- Definition of the average number of cans used per game
def cans_per_game : Nat := 5

-- Total number of games in the tournament
def total_games : Nat :=
  games_in_round 1 + games_in_round 2 + games_in_round 3 + games_in_round 4

-- Total number of cans used
def total_cans : Nat :=
  total_games * cans_per_game

-- Total number of tennis balls used
def total_tennis_balls : Nat := 225

-- Number of tennis balls per can
def tennis_balls_per_can : Nat :=
  total_tennis_balls / total_cans

-- Theorem to prove
theorem tennis_balls_per_can_is_three :
  tennis_balls_per_can = 3 :=
by
  -- No proof required, using sorry to skip the proof
  sorry

end tennis_balls_per_can_is_three_l678_67830


namespace small_cubes_one_face_painted_red_l678_67833

-- Definitions
def is_red_painted (cube : ℕ) : Bool := true -- representing the condition that the cube is painted red
def side_length (cube : ℕ) : ℕ := 4 -- side length of the original cube is 4 cm
def smaller_cube_side_length : ℕ := 1 -- smaller cube side length is 1 cm

-- Theorem Statement
theorem small_cubes_one_face_painted_red :
  ∀ (large_cube : ℕ), (side_length large_cube = 4) ∧ is_red_painted large_cube → 
  (∃ (number_of_cubes : ℕ), number_of_cubes = 24) :=
by
  sorry

end small_cubes_one_face_painted_red_l678_67833


namespace inverse_of_2_is_46_l678_67828

-- Given the function f(x) = 5x^3 + 6
def f (x : ℝ) : ℝ := 5 * x^3 + 6

-- Prove the statement
theorem inverse_of_2_is_46 : (∃ y, f y = x) ∧ f (2 : ℝ) = 46 → x = 46 :=
by
  sorry

end inverse_of_2_is_46_l678_67828


namespace find_range_of_a_l678_67875

def prop_p (a : ℝ) : Prop :=
∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0

def prop_q (a : ℝ) : Prop :=
(∃ x₁ x₂ : ℝ, x₁ * x₂ = 1 ∧ x₁ + x₂ = -(a - 1) ∧ (0 < x₁ ∧ x₁ < 1 ∧ 1 < x₂ ∧ x₂ < 2))

def range_a (a : ℝ) : Prop :=
(-2 < a ∧ a <= -3/2) ∨ (-1 <= a ∧ a <= 2)

theorem find_range_of_a (a : ℝ) :
  (prop_p a ∨ prop_q a) ∧ ¬ (prop_p a ∧ prop_q a) ↔ range_a a :=
sorry

end find_range_of_a_l678_67875


namespace jason_earns_88_dollars_l678_67837

theorem jason_earns_88_dollars (earn_after_school: ℝ) (earn_saturday: ℝ)
  (total_hours: ℝ) (saturday_hours: ℝ) (after_school_hours: ℝ) (total_earn: ℝ)
  (h1 : earn_after_school = 4.00)
  (h2 : earn_saturday = 6.00)
  (h3 : total_hours = 18)
  (h4 : saturday_hours = 8)
  (h5 : after_school_hours = total_hours - saturday_hours)
  (h6 : total_earn = after_school_hours * earn_after_school + saturday_hours * earn_saturday) :
  total_earn = 88.00 :=
by
  sorry

end jason_earns_88_dollars_l678_67837


namespace rectangle_length_15_l678_67838

theorem rectangle_length_15
  (w l : ℝ)
  (h_ratio : 5 * w = 2 * l + 2 * w)
  (h_area : l * w = 150) :
  l = 15 :=
sorry

end rectangle_length_15_l678_67838


namespace percentage_error_in_area_l678_67824

theorem percentage_error_in_area (s : ℝ) (h : s > 0) :
  let s' := s * (1 + 0.03)
  let A := s * s
  let A' := s' * s'
  ((A' - A) / A) * 100 = 6.09 :=
by
  sorry

end percentage_error_in_area_l678_67824


namespace funnel_paper_area_l678_67880

theorem funnel_paper_area
  (slant_height : ℝ)
  (base_circumference : ℝ)
  (h1 : slant_height = 6)
  (h2 : base_circumference = 6 * Real.pi):
  (1 / 2) * base_circumference * slant_height = 18 * Real.pi :=
by
  sorry

end funnel_paper_area_l678_67880


namespace average_visitors_in_30_day_month_l678_67866

def average_visitors_per_day (visitors_sunday visitors_other : ℕ) (days_in_month : ℕ) (starts_on_sunday : Prop) : ℕ :=
    let sundays := days_in_month / 7 + if days_in_month % 7 > 0 then 1 else 0
    let other_days := days_in_month - sundays
    let total_visitors := sundays * visitors_sunday + other_days * visitors_other
    total_visitors / days_in_month

theorem average_visitors_in_30_day_month 
    (visitors_sunday : ℕ) (visitors_other : ℕ) (days_in_month : ℕ) (starts_on_sunday : Prop) (h1 : visitors_sunday = 660) (h2 : visitors_other = 240) (h3 : days_in_month = 30) :
    average_visitors_per_day visitors_sunday visitors_other days_in_month starts_on_sunday = 296 := 
by
  sorry

end average_visitors_in_30_day_month_l678_67866


namespace sample_size_six_l678_67843

-- Definitions for the conditions
def num_senior_teachers : ℕ := 18
def num_first_level_teachers : ℕ := 12
def num_top_level_teachers : ℕ := 6
def total_teachers : ℕ := num_senior_teachers + num_first_level_teachers + num_top_level_teachers

-- The proof problem statement
theorem sample_size_six (n : ℕ) (h1 : n > 0) : 
  (∀ m : ℕ, m * n = total_teachers → 
             ((n + 1) * m - 1 = 35) → False) → n = 6 :=
sorry

end sample_size_six_l678_67843


namespace scientific_notation_example_l678_67832

theorem scientific_notation_example : 0.0000037 = 3.7 * 10^(-6) :=
by
  -- We would provide the proof here.
  sorry

end scientific_notation_example_l678_67832


namespace smallest_n_2000_divides_a_n_l678_67865

theorem smallest_n_2000_divides_a_n (a : ℕ → ℤ) 
  (h_rec : ∀ n, n ≥ 1 → (n - 1) * a (n + 1) = (n + 1) * a n - 2 * (n - 1)) 
  (h2000 : 2000 ∣ a 1999) : 
  ∃ n, n ≥ 2 ∧ 2000 ∣ a n ∧ n = 249 := 
by 
  sorry

end smallest_n_2000_divides_a_n_l678_67865


namespace proportion_sets_l678_67847

-- Define unit lengths for clarity
def length (n : ℕ) := n 

-- Define the sets of line segments
def setA := (length 4, length 5, length 6, length 7)
def setB := (length 3, length 4, length 5, length 8)
def setC := (length 5, length 15, length 3, length 9)
def setD := (length 8, length 4, length 1, length 3)

-- Define a condition for a set to form a proportion
def is_proportional (a b c d : ℕ) : Prop :=
  a * d = b * c

-- Main theorem: setC forms a proportion while others don't
theorem proportion_sets : is_proportional 5 15 3 9 ∧ 
                         ¬ is_proportional 4 5 6 7 ∧ 
                         ¬ is_proportional 3 4 5 8 ∧ 
                         ¬ is_proportional 8 4 1 3 := by
  sorry

end proportion_sets_l678_67847


namespace total_valid_arrangements_l678_67809

-- Define the students and schools
inductive Student
| G1 | G2 | B1 | B2 | B3 | BA
deriving DecidableEq

inductive School
| A | B | C
deriving DecidableEq

-- Define the condition that any two students cannot be in the same school
def is_valid_arrangement (arr : School → Student → Bool) : Bool :=
  (arr School.A Student.G1 ≠ arr School.A Student.G2) ∧ 
  (arr School.B Student.G1 ≠ arr School.B Student.G2) ∧
  (arr School.C Student.G1 ≠ arr School.C Student.G2) ∧
  ¬ arr School.C Student.G1 ∧
  ¬ arr School.C Student.G2 ∧
  ¬ arr School.A Student.BA

-- The theorem to prove the total number of different valid arrangements
theorem total_valid_arrangements : 
  ∃ n : ℕ, n = 18 ∧ ∃ arr : (School → Student → Bool), is_valid_arrangement arr := 
sorry

end total_valid_arrangements_l678_67809


namespace football_kick_distance_l678_67819

theorem football_kick_distance (a : ℕ) (avg : ℕ) (x : ℕ)
  (h1 : a = 43)
  (h2 : avg = 37)
  (h3 : 3 * avg = a + 2 * x) :
  x = 34 :=
by
  sorry

end football_kick_distance_l678_67819


namespace lines_parallel_l678_67870

theorem lines_parallel (a : ℝ) 
  (h₁ : (∀ x y : ℝ, ax + (a + 2) * y + 2 = 0)) 
  (h₂ : (∀ x y : ℝ, x + a * y + 1 = 0)) 
  : a = -1 :=
sorry

end lines_parallel_l678_67870


namespace carbonated_water_percentage_is_correct_l678_67841

-- Given percentages of lemonade and carbonated water in two solutions
def first_solution : Rat := 0.20 -- Lemonade percentage in the first solution
def second_solution : Rat := 0.45 -- Lemonade percentage in the second solution

-- Calculate percentages of carbonated water
def first_solution_carbonated_water := 1 - first_solution
def second_solution_carbonated_water := 1 - second_solution

-- Assume the mixture is 100 units, with equal parts from both solutions
def volume_mixture : Rat := 100
def volume_first_solution : Rat := volume_mixture * 0.50
def volume_second_solution : Rat := volume_mixture * 0.50

-- Calculate total carbonated water in the mixture
def carbonated_water_in_mixture :=
  (volume_first_solution * first_solution_carbonated_water) +
  (volume_second_solution * second_solution_carbonated_water)

-- Calculate the percentage of carbonated water in the mixture
def percentage_carbonated_water_in_mixture : Rat :=
  (carbonated_water_in_mixture / volume_mixture) * 100

-- Prove the percentage of carbonated water in the mixture is 67.5%
theorem carbonated_water_percentage_is_correct :
  percentage_carbonated_water_in_mixture = 67.5 := by
  sorry

end carbonated_water_percentage_is_correct_l678_67841


namespace evaluate_x_squared_minus_y_squared_l678_67835

theorem evaluate_x_squared_minus_y_squared (x y : ℝ) 
  (h1 : x + y = 12) 
  (h2 : 3*x + y = 18) 
  : x^2 - y^2 = -72 := 
by
  sorry

end evaluate_x_squared_minus_y_squared_l678_67835


namespace degrees_multiplication_proof_l678_67811

/-- Convert a measurement given in degrees and minutes to purely degrees. -/
def degrees (d : Int) (m : Int) : ℚ := d + m / 60

/-- Given conditions: -/
def lhs : ℚ := degrees 21 17
def rhs : ℚ := degrees 106 25

/-- The theorem to prove the mathematical problem. -/
theorem degrees_multiplication_proof : lhs * 5 = rhs := sorry

end degrees_multiplication_proof_l678_67811


namespace largest_multiple_of_15_who_negation_greater_than_neg_150_l678_67808

theorem largest_multiple_of_15_who_negation_greater_than_neg_150 : 
  ∃ (x : ℤ), x % 15 = 0 ∧ -x > -150 ∧ ∀ (y : ℤ), y % 15 = 0 ∧ -y > -150 → x ≥ y :=
by
  sorry

end largest_multiple_of_15_who_negation_greater_than_neg_150_l678_67808


namespace additional_teddies_per_bunny_l678_67873

theorem additional_teddies_per_bunny (teddies bunnies koala total_mascots: ℕ) 
  (h1 : teddies = 5) 
  (h2 : bunnies = 3 * teddies) 
  (h3 : koala = 1) 
  (h4 : total_mascots = 51): 
  (total_mascots - (teddies + bunnies + koala)) / bunnies = 2 := 
by 
  sorry

end additional_teddies_per_bunny_l678_67873


namespace ellipses_same_eccentricity_l678_67821

theorem ellipses_same_eccentricity 
  (a b : ℝ) (k : ℝ)
  (h1 : a > 0) 
  (h2 : b > 0)
  (h3 : k > 0)
  (e1_eq : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1 ↔ (x^2 / (a^2)) + (y^2 / (b^2)) = 1)
  (e2_eq : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = k ↔ (x^2 / (ka^2)) + (y^2 / (kb^2)) = 1) :
  1 - (b^2 / a^2) = 1 - (b^2 / (ka^2)) :=
by
  sorry

end ellipses_same_eccentricity_l678_67821


namespace cartesian_equation_of_parametric_l678_67800

variable (t : ℝ) (x y : ℝ)

open Real

theorem cartesian_equation_of_parametric 
  (h1 : x = sqrt t)
  (h2 : y = 2 * sqrt (1 - t))
  (h3 : 0 ≤ t ∧ t ≤ 1) :
  (x^2 / 1) + (y^2 / 4) = 1 := by 
  sorry

end cartesian_equation_of_parametric_l678_67800


namespace average_of_tenths_and_thousandths_l678_67895

theorem average_of_tenths_and_thousandths :
  (0.4 + 0.005) / 2 = 0.2025 :=
by
  -- We skip the proof here
  sorry

end average_of_tenths_and_thousandths_l678_67895


namespace wilfred_carrots_total_l678_67874

-- Define the number of carrots Wilfred eats each day
def tuesday_carrots := 4
def wednesday_carrots := 6
def thursday_carrots := 5

-- Define the total number of carrots eaten from Tuesday to Thursday
def total_carrots := tuesday_carrots + wednesday_carrots + thursday_carrots

-- The theorem to prove that the total number of carrots is 15
theorem wilfred_carrots_total : total_carrots = 15 := by
  sorry

end wilfred_carrots_total_l678_67874


namespace ninety_seven_squared_l678_67854

theorem ninety_seven_squared : (97 * 97 = 9409) :=
by
  sorry

end ninety_seven_squared_l678_67854


namespace inverse_function_l678_67812

variable (x : ℝ)

def f (x : ℝ) : ℝ := (x^(1 / 3)) + 1
def g (x : ℝ) : ℝ := (x - 1)^3

theorem inverse_function :
  ∀ x, f (g x) = x ∧ g (f x) = x :=
by
  -- Proof goes here
  sorry

end inverse_function_l678_67812


namespace problem1_problem2_problem3_l678_67836

-- Problem 1
def s_type_sequence (a : ℕ → ℕ) : Prop := 
∀ n ≥ 1, a (n+1) - a n > 3

theorem problem1 (a : ℕ → ℕ) (h₀ : a 1 = 4) (h₁ : a 2 = 8) 
  (h₂ : ∀ n ≥ 2, a n + a (n - 1) = 8 * n - 4) : s_type_sequence a := 
sorry

-- Problem 2
theorem problem2 (a : ℕ → ℕ) (h₀ : ∀ n m, a (n * m) = (a n) ^ m)
  (b : ℕ → ℕ) (h₁ : ∀ n, b n = (3 * a n) / 4)
  (h₂ : s_type_sequence a)
  (h₃ : ¬ s_type_sequence b) : 
  (∀ n, a n = 2^(n+1)) ∨ (∀ n, a n = 2 * 3^(n-1)) ∨ (∀ n, a n = 5^ (n-1)) :=
sorry

-- Problem 3
theorem problem3 (c : ℕ → ℕ) 
  (h₀ : c 2 = 9)
  (h₁ : ∀ n ≥ 2, (1 / n - 1 / (n + 1)) * (2 + 1 / c n) ≤ 1 / c (n - 1) + 1 / c n 
               ∧ 1 / c (n - 1) + 1 / c n ≤ (1 / n - 1 / (n + 1)) * (2 + 1 / c (n-1))) :
  ∃ f : ℕ → ℕ, (s_type_sequence c) ∧ (∀ n, c n = (n + 1)^2) := 
sorry

end problem1_problem2_problem3_l678_67836


namespace max_profit_l678_67827

noncomputable def maximum_profit : ℤ := 
  21000

theorem max_profit (x y : ℕ) 
  (h1 : 4 * x + 8 * y ≤ 8000)
  (h2 : 2 * x + y ≤ 1300)
  (h3 : 15 * x + 20 * y ≤ maximum_profit) : 
  15 * x + 20 * y = maximum_profit := 
sorry

end max_profit_l678_67827


namespace digits_exceed_10_power_15_l678_67861

noncomputable def log3 (x : ℝ) : ℝ := Real.log x / Real.log 3
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem digits_exceed_10_power_15 (x : ℝ) 
  (h : log3 (log2 (log2 x)) = 3) : log10 x > 10^15 := 
sorry

end digits_exceed_10_power_15_l678_67861


namespace total_hours_worked_l678_67885

variable (A B C D E T : ℝ)

theorem total_hours_worked (hA : A = 12)
  (hB : B = 1 / 3 * A)
  (hC : C = 2 * B)
  (hD : D = 1 / 2 * E)
  (hE : E = A + 3)
  (hT : T = A + B + C + D + E) : T = 46.5 :=
by
  sorry

end total_hours_worked_l678_67885


namespace sin_minus_cos_value_l678_67893

theorem sin_minus_cos_value
  (α : ℝ)
  (h1 : Real.tan α = (Real.sqrt 3) / 3)
  (h2 : π < α ∧ α < 3 * π / 2) :
  Real.sin α - Real.cos α = -1/2 + Real.sqrt 3 / 2 :=
by
  sorry

end sin_minus_cos_value_l678_67893


namespace segments_interior_proof_l678_67868

noncomputable def count_internal_segments (squares hexagons octagons : Nat) : Nat := 
  let vertices := (squares * 4 + hexagons * 6 + octagons * 8) / 3
  let total_segments := (vertices * (vertices - 1)) / 2
  let edges_along_faces := 3 * vertices
  (total_segments - edges_along_faces) / 2

theorem segments_interior_proof : count_internal_segments 12 8 6 = 840 := 
  by sorry

end segments_interior_proof_l678_67868


namespace tan_ratio_l678_67864

open Real

theorem tan_ratio (x y : ℝ) (h1 : sin x / cos y + sin y / cos x = 2) (h2 : cos x / sin y + cos y / sin x = 4) : 
  tan x / tan y + tan y / tan x = 2 :=
sorry

end tan_ratio_l678_67864


namespace custom_op_evaluation_l678_67804

def custom_op (a b : ℤ) : ℤ := a * b - (a + b)

theorem custom_op_evaluation : custom_op 2 (-3) = -5 :=
by
sorry

end custom_op_evaluation_l678_67804


namespace probability_function_meaningful_l678_67898

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2

def is_meaningful (x : ℝ) : Prop := 1 - x^2 > 0

def measure_interval (a b : ℝ) : ℝ := b - a

theorem probability_function_meaningful:
  let interval_a := -2
  let interval_b := 1
  let meaningful_a := -1
  let meaningful_b := 1
  let total_interval := measure_interval interval_a interval_b
  let meaningful_interval := measure_interval meaningful_a meaningful_b
  let P := meaningful_interval / total_interval
  (P = (2/3)) :=
by
  sorry

end probability_function_meaningful_l678_67898


namespace sequence_general_term_l678_67867

theorem sequence_general_term (n : ℕ) : 
  (∃ (f : ℕ → ℕ), (∀ k, f k = k^2) ∧ (∀ m, f m = m^2)) :=
by
  -- Given the sequence 1, 4, 9, 16, 25, ...
  sorry

end sequence_general_term_l678_67867


namespace circle_center_radius_l678_67823

-- Define the necessary parameters and let Lean solve the equivalent proof problem
theorem circle_center_radius:
  (∃ a b r : ℝ, (∀ x y : ℝ, x^2 + 8 * x + y^2 - 2 * y = 1 ↔ (x + 4)^2 + (y - 1)^2 = 18) 
  ∧ a = -4 
  ∧ b = 1 
  ∧ r = 3 * Real.sqrt 2
  ∧ a + b + r = -3 + 3 * Real.sqrt 2) :=
by {
  sorry
}

end circle_center_radius_l678_67823


namespace complex_b_value_l678_67878

open Complex

theorem complex_b_value (b : ℝ) (h : (2 - b * I) / (1 + 2 * I) = (2 - 2 * b) / 5 + ((-4 - b) / 5) * I) :
  b = -2 / 3 :=
sorry

end complex_b_value_l678_67878


namespace task_probabilities_l678_67805

theorem task_probabilities (P1_on_time : ℚ) (P2_on_time : ℚ) 
  (h1 : P1_on_time = 2/3) (h2 : P2_on_time = 3/5) : 
  P1_on_time * (1 - P2_on_time) = 4/15 := 
by
  -- proof is omitted
  sorry

end task_probabilities_l678_67805


namespace interest_years_calculation_l678_67869

theorem interest_years_calculation 
  (total_sum : ℝ)
  (second_sum : ℝ)
  (interest_rate_first : ℝ)
  (interest_rate_second : ℝ)
  (time_second : ℝ)
  (interest_second : ℝ)
  (x : ℝ)
  (y : ℝ)
  (h1 : total_sum = 2795)
  (h2 : second_sum = 1720)
  (h3 : interest_rate_first = 3)
  (h4 : interest_rate_second = 5)
  (h5 : time_second = 3)
  (h6 : interest_second = (second_sum * interest_rate_second * time_second) / 100)
  (h7 : interest_second = 258)
  (h8 : x = (total_sum - second_sum))
  (h9 : (interest_rate_first * x * y) / 100 = interest_second)
  : y = 8 := sorry

end interest_years_calculation_l678_67869


namespace parallel_line_eq_perpendicular_line_eq_l678_67816

-- Define the conditions: A line passing through (1, -4) and the given line equation 2x + 3y + 5 = 0
def passes_through (x y : ℝ) (a b c : ℝ) : Prop := a * x + b * y + c = 0

-- Define the theorem statements for parallel and perpendicular lines
theorem parallel_line_eq (m : ℝ) :
  passes_through 1 (-4) 2 3 m → m = 10 := 
sorry

theorem perpendicular_line_eq (n : ℝ) :
  passes_through 1 (-4) 3 (-2) (-n) → n = 11 :=
sorry

end parallel_line_eq_perpendicular_line_eq_l678_67816


namespace min_students_solved_both_l678_67810

/-- A simple mathematical proof problem to find the minimum number of students who solved both problems correctly --/
theorem min_students_solved_both (total_students first_problem second_problem : ℕ)
  (h₀ : total_students = 30)
  (h₁ : first_problem = 21)
  (h₂ : second_problem = 18) :
  ∃ (both_solved : ℕ), both_solved = 9 :=
by
  sorry

end min_students_solved_both_l678_67810


namespace zoo_animal_difference_l678_67889

theorem zoo_animal_difference :
  let parrots := 8
  let snakes := 3 * parrots
  let monkeys := 2 * snakes
  let elephants := 1 / 2 * (parrots + snakes)
  let zebras := elephants - 3
  monkeys - zebras = 35 :=
by
  sorry

end zoo_animal_difference_l678_67889


namespace base8_to_base10_l678_67886

theorem base8_to_base10 (n : ℕ) : n = 4 * 8^3 + 3 * 8^2 + 7 * 8^1 + 2 * 8^0 → n = 2298 :=
by 
  sorry

end base8_to_base10_l678_67886


namespace joe_lists_count_l678_67897

theorem joe_lists_count : ∃ (n : ℕ), n = 15 * 14 := sorry

end joe_lists_count_l678_67897


namespace inscribed_circle_radius_l678_67825

noncomputable def calculate_r (a b c : ℝ) : ℝ :=
  let term1 := 1 / a
  let term2 := 1 / b
  let term3 := 1 / c
  let term4 := 3 * Real.sqrt (1 / (a * b) + 1 / (a * c) + 1 / (b * c))
  1 / (term1 + term2 + term3 + term4)

theorem inscribed_circle_radius :
  calculate_r 6 10 15 = 30 / (10 * Real.sqrt 26 + 3) :=
by
  sorry

end inscribed_circle_radius_l678_67825


namespace area_under_abs_sin_l678_67863

noncomputable def f (x : ℝ) : ℝ := |Real.sin x|

theorem area_under_abs_sin : 
  ∫ x in -Real.pi..Real.pi, f x = 4 :=
by
  sorry

end area_under_abs_sin_l678_67863


namespace total_selling_price_of_cloth_l678_67887

theorem total_selling_price_of_cloth
  (profit_per_meter : ℕ)
  (cost_price_per_meter : ℕ)
  (total_meters : ℕ)
  (total_selling_price : ℕ) :
  profit_per_meter = 7 →
  cost_price_per_meter = 118 →
  total_meters = 80 →
  total_selling_price = (cost_price_per_meter + profit_per_meter) * total_meters →
  total_selling_price = 10000 :=
by
  intros h_profit h_cost h_total h_selling_price
  rw [h_profit, h_cost, h_total] at h_selling_price
  exact h_selling_price

end total_selling_price_of_cloth_l678_67887


namespace solve_x_squared_eq_four_x_l678_67876

theorem solve_x_squared_eq_four_x : {x : ℝ | x^2 = 4*x} = {0, 4} := 
sorry

end solve_x_squared_eq_four_x_l678_67876


namespace drink_exactly_five_bottles_last_day_l678_67801

/-- 
Robin bought 617 bottles of water and needs to purchase 4 additional bottles on the last day 
to meet her daily water intake goal. 
Prove that Robin will drink exactly 5 bottles on the last day.
-/
theorem drink_exactly_five_bottles_last_day : 
  ∀ (bottles_bought : ℕ) (extra_bottles : ℕ), bottles_bought = 617 → extra_bottles = 4 → 
  ∃ x : ℕ, 621 = x * 617 + 4 ∧ x + 4 = 5 :=
by
  intros bottles_bought extra_bottles bottles_bought_eq extra_bottles_eq
  -- The proof would follow here
  sorry

end drink_exactly_five_bottles_last_day_l678_67801


namespace boat_speed_proof_l678_67851

noncomputable def speed_in_still_water : ℝ := sorry -- Defined but proof skipped

def stream_speed : ℝ := 4
def distance_downstream : ℝ := 32
def distance_upstream : ℝ := 16

theorem boat_speed_proof (v : ℝ) :
  (distance_downstream / (v + stream_speed) = distance_upstream / (v - stream_speed)) →
  v = 12 :=
by
  sorry

end boat_speed_proof_l678_67851


namespace combined_moles_l678_67881

def balanced_reaction (NaHCO3 HC2H3O2 H2O : ℕ) : Prop :=
  NaHCO3 + HC2H3O2 = H2O

theorem combined_moles (NaHCO3 HC2H3O2 : ℕ) 
  (h : balanced_reaction NaHCO3 HC2H3O2 3) : 
  NaHCO3 + HC2H3O2 = 6 :=
sorry

end combined_moles_l678_67881


namespace diagonals_in_25_sided_polygon_l678_67807

-- Define a function to calculate the number of specific diagonals in a convex polygon with n sides
def number_of_diagonals (n : ℕ) : ℕ :=
  n * (n - 5) / 2

-- Theorem stating the number of diagonals for a convex polygon with 25 sides with the given condition
theorem diagonals_in_25_sided_polygon : number_of_diagonals 25 = 250 := 
sorry

end diagonals_in_25_sided_polygon_l678_67807


namespace ashley_cocktail_calories_l678_67844

theorem ashley_cocktail_calories:
  let mango_grams := 150
  let honey_grams := 200
  let water_grams := 300
  let vodka_grams := 100

  let mango_cal_per_100g := 60
  let honey_cal_per_100g := 640
  let vodka_cal_per_100g := 70
  let water_cal_per_100g := 0

  let total_cocktail_grams := mango_grams + honey_grams + water_grams + vodka_grams
  let total_cocktail_calories := (mango_grams * mango_cal_per_100g / 100) +
                                 (honey_grams * honey_cal_per_100g / 100) +
                                 (vodka_grams * vodka_cal_per_100g / 100) +
                                 (water_grams * water_cal_per_100g / 100)
  let caloric_density := total_cocktail_calories / total_cocktail_grams
  let result := 300 * caloric_density
  result = 576 := by
  sorry

end ashley_cocktail_calories_l678_67844


namespace determine_uv_l678_67883

theorem determine_uv :
  ∃ u v : ℝ, (u = 5 / 17) ∧ (v = -31 / 17) ∧
    ((⟨3, -2⟩ : ℝ × ℝ) + u • ⟨5, 8⟩ = (⟨-1, 4⟩ : ℝ × ℝ) + v • ⟨-3, 2⟩) :=
by
  sorry

end determine_uv_l678_67883


namespace Annie_total_cookies_l678_67806

theorem Annie_total_cookies :
  let monday_cookies := 5
  let tuesday_cookies := 2 * monday_cookies 
  let wednesday_cookies := 1.4 * tuesday_cookies
  monday_cookies + tuesday_cookies + wednesday_cookies = 29 :=
by
  sorry

end Annie_total_cookies_l678_67806
