import Mathlib

namespace NUMINAMATH_GPT_tangent_lines_to_circle_passing_through_point_l798_79855

theorem tangent_lines_to_circle_passing_through_point :
  ∀ (x y : ℝ), (x-1)^2 + (y-1)^2 = 1 → ((x = 2 ∧ y = 0) ∨ (x = 1 ∧ y = -1)) :=
by
  sorry

end NUMINAMATH_GPT_tangent_lines_to_circle_passing_through_point_l798_79855


namespace NUMINAMATH_GPT_first_term_of_geometric_sequence_l798_79847

theorem first_term_of_geometric_sequence (a r : ℚ) (h1 : a * r^2 = 12) (h2 : a * r^3 = 16) : a = 27 / 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_first_term_of_geometric_sequence_l798_79847


namespace NUMINAMATH_GPT_common_points_circle_ellipse_l798_79871

theorem common_points_circle_ellipse :
    (∃ (p1 p2: ℝ × ℝ),
        p1 ≠ p2 ∧
        (p1, p2).fst.1 ^ 2 + (p1, p2).fst.2 ^ 2 = 4 ∧
        9 * (p1, p2).fst.1 ^ 2 + 4 * (p1, p2).fst.2 ^ 2 = 36 ∧
        (p1, p2).snd.1 ^ 2 + (p1, p2).snd.2 ^ 2 = 4 ∧
        9 * (p1, p2).snd.1 ^ 2 + 4 * (p1, p2).snd.2 ^ 2 = 36) :=
sorry

end NUMINAMATH_GPT_common_points_circle_ellipse_l798_79871


namespace NUMINAMATH_GPT_non_congruent_rectangles_count_l798_79873

theorem non_congruent_rectangles_count :
  (∃ (l w : ℕ), l + w = 50 ∧ l ≠ w) ∧
  (∀ (l w : ℕ), l + w = 50 ∧ l ≠ w → l > w) →
  (∃ (n : ℕ), n = 24) :=
by
  sorry

end NUMINAMATH_GPT_non_congruent_rectangles_count_l798_79873


namespace NUMINAMATH_GPT_S_of_1_eq_8_l798_79894

variable (x : ℝ)

-- Definition of original polynomial R(x)
def R (x : ℝ) : ℝ := 3 * x^3 - 5 * x + 4

-- Definition of new polynomial S(x) created by adding 2 to each coefficient of R(x)
def S (x : ℝ) : ℝ := 5 * x^3 - 3 * x + 6

-- The theorem we want to prove
theorem S_of_1_eq_8 : S 1 = 8 := by
  sorry

end NUMINAMATH_GPT_S_of_1_eq_8_l798_79894


namespace NUMINAMATH_GPT_simplify_fraction_l798_79866

theorem simplify_fraction (x y : ℝ) (h : x ≠ y) : (x^6 - y^6) / (x^3 - y^3) = x^3 + y^3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l798_79866


namespace NUMINAMATH_GPT_ratio_trumpet_to_flute_l798_79865

-- Given conditions
def flute_players : ℕ := 5
def trumpet_players (T : ℕ) : ℕ := T
def trombone_players (T : ℕ) : ℕ := T - 8
def drummers (T : ℕ) : ℕ := T - 8 + 11
def clarinet_players : ℕ := 2 * flute_players
def french_horn_players (T : ℕ) : ℕ := T - 8 + 3
def total_seats_needed (T : ℕ) : ℕ := 
  flute_players + trumpet_players T + trombone_players T + drummers T + clarinet_players + french_horn_players T

-- Proof statement
theorem ratio_trumpet_to_flute 
  (T : ℕ) (h : total_seats_needed T = 65) : trumpet_players T / flute_players = 3 :=
sorry

end NUMINAMATH_GPT_ratio_trumpet_to_flute_l798_79865


namespace NUMINAMATH_GPT_makes_at_least_one_shot_l798_79867
noncomputable section

/-- The probability of making the free throw. -/
def free_throw_make_prob : ℚ := 4/5

/-- The probability of making the high school 3-pointer. -/
def high_school_make_prob : ℚ := 1/2

/-- The probability of making the professional 3-pointer. -/
def pro_make_prob : ℚ := 1/3

/-- The probability of making at least one of the three shots. -/
theorem makes_at_least_one_shot :
  (1 - ((1 - free_throw_make_prob) * (1 - high_school_make_prob) * (1 - pro_make_prob))) = 14 / 15 :=
by
  sorry

end NUMINAMATH_GPT_makes_at_least_one_shot_l798_79867


namespace NUMINAMATH_GPT_equivalence_condition_l798_79816

theorem equivalence_condition (a b c d : ℝ) (h : (a + b) / (b + c) = (c + d) / (d + a)) : 
  a = c ∨ a + b + c + d = 0 :=
sorry

end NUMINAMATH_GPT_equivalence_condition_l798_79816


namespace NUMINAMATH_GPT_sam_sandwich_shop_cost_l798_79893

theorem sam_sandwich_shop_cost :
  let sandwich_cost := 4
  let soda_cost := 3
  let fries_cost := 2
  let num_sandwiches := 3
  let num_sodas := 7
  let num_fries := 5
  let total_cost := num_sandwiches * sandwich_cost + num_sodas * soda_cost + num_fries * fries_cost
  total_cost = 43 :=
by
  sorry

end NUMINAMATH_GPT_sam_sandwich_shop_cost_l798_79893


namespace NUMINAMATH_GPT_compound_interest_rate_l798_79845

theorem compound_interest_rate
  (P : ℝ) (t : ℕ) (A : ℝ) (interest : ℝ)
  (hP : P = 6000)
  (ht : t = 2)
  (hA : A = 7260)
  (hInterest : interest = 1260.000000000001)
  (hA_eq : A = P + interest) :
  ∃ r : ℝ, (1 + r)^(t : ℝ) = A / P ∧ r = 0.1 :=
by
  sorry

end NUMINAMATH_GPT_compound_interest_rate_l798_79845


namespace NUMINAMATH_GPT_book_purchasing_methods_l798_79887

theorem book_purchasing_methods :
  ∃ (A B C D : ℕ),
  A + B + C + D = 10 ∧
  A > 0 ∧ B > 0 ∧ C > 0 ∧ D > 0 ∧
  3 * A + 5 * B + 7 * C + 11 * D = 70 ∧
  (∃ N : ℕ, N = 4) :=
by sorry

end NUMINAMATH_GPT_book_purchasing_methods_l798_79887


namespace NUMINAMATH_GPT_sum_of_interior_numbers_l798_79808

def sum_interior (n : ℕ) : ℕ := 2^(n-1) - 2

theorem sum_of_interior_numbers :
  sum_interior 8 + sum_interior 9 + sum_interior 10 = 890 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_interior_numbers_l798_79808


namespace NUMINAMATH_GPT_triangle_perimeter_l798_79892

def is_triangle (a b c : ℕ) := (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem triangle_perimeter {a b c : ℕ} (h : is_triangle 15 11 19) : 15 + 11 + 19 = 45 := by
  sorry

end NUMINAMATH_GPT_triangle_perimeter_l798_79892


namespace NUMINAMATH_GPT_largest_percentage_increase_is_2013_to_2014_l798_79807

-- Defining the number of students in each year as constants
def students_2010 : ℕ := 50
def students_2011 : ℕ := 56
def students_2012 : ℕ := 62
def students_2013 : ℕ := 68
def students_2014 : ℕ := 77
def students_2015 : ℕ := 81

-- Defining the percentage increase between consecutive years
def percentage_increase (a b : ℕ) : ℚ := ((b - a) : ℚ) / (a : ℚ)

-- Calculating all the percentage increases
def pi_2010_2011 := percentage_increase students_2010 students_2011
def pi_2011_2012 := percentage_increase students_2011 students_2012
def pi_2012_2013 := percentage_increase students_2012 students_2013
def pi_2013_2014 := percentage_increase students_2013 students_2014
def pi_2014_2015 := percentage_increase students_2014 students_2015

-- The theorem stating the largest percentage increase is between 2013 and 2014
theorem largest_percentage_increase_is_2013_to_2014 :
  max (pi_2010_2011) (max (pi_2011_2012) (max (pi_2012_2013) (max (pi_2013_2014) (pi_2014_2015)))) = pi_2013_2014 :=
sorry

end NUMINAMATH_GPT_largest_percentage_increase_is_2013_to_2014_l798_79807


namespace NUMINAMATH_GPT_find_x_l798_79832

theorem find_x (x : ℝ) (h : 0.35 * 400 = 0.20 * x): x = 700 :=
sorry

end NUMINAMATH_GPT_find_x_l798_79832


namespace NUMINAMATH_GPT_find_certain_number_l798_79851

theorem find_certain_number (x : ℕ) (h1 : 172 = 4 * 43) (h2 : 43 - 172 / x = 28) (h3 : 172 % x = 7) : x = 11 := by
  sorry

end NUMINAMATH_GPT_find_certain_number_l798_79851


namespace NUMINAMATH_GPT_student_competition_distribution_l798_79820

theorem student_competition_distribution :
  ∃ f : Fin 4 → Fin 3, (∀ i j : Fin 3, i ≠ j → ∃ x : Fin 4, f x = i ∧ ∃ y : Fin 4, f y = j) ∧ 
  (Finset.univ.image f).card = 3 := 
sorry

end NUMINAMATH_GPT_student_competition_distribution_l798_79820


namespace NUMINAMATH_GPT_pick_two_black_cards_l798_79864

-- Definition: conditions
def total_cards : ℕ := 52
def cards_per_suit : ℕ := 13
def black_suits : ℕ := 2
def red_suits : ℕ := 2
def total_black_cards : ℕ := black_suits * cards_per_suit

-- Theorem: number of ways to pick two different black cards
theorem pick_two_black_cards :
  (total_black_cards * (total_black_cards - 1)) = 650 :=
by
  -- proof here
  sorry

end NUMINAMATH_GPT_pick_two_black_cards_l798_79864


namespace NUMINAMATH_GPT_club_membership_l798_79860

theorem club_membership (n : ℕ) 
  (h1 : n % 10 = 6)
  (h2 : n % 11 = 6)
  (h3 : 150 ≤ n)
  (h4 : n ≤ 300) : 
  n = 226 := 
sorry

end NUMINAMATH_GPT_club_membership_l798_79860


namespace NUMINAMATH_GPT_statements_correct_l798_79824

theorem statements_correct :
  (∀ x : ℝ, (x ≠ 1 → x^2 - 3*x + 2 ≠ 0) ↔ (x^2 - 3*x + 2 = 0 → x = 1)) ∧
  (∀ x : ℝ, (∀ x, x^2 + x + 1 ≠ 0) ↔ (∃ x, x^2 + x + 1 = 0)) ∧
  (∀ p q : Prop, (p ∧ q) ↔ p ∧ q) ∧
  (∀ x : ℝ, (x > 2 → x^2 - 3*x + 2 > 0) ∧ (¬ (x^2 - 3*x + 2 > 0) → x ≤ 2)) :=
by
  sorry

end NUMINAMATH_GPT_statements_correct_l798_79824


namespace NUMINAMATH_GPT_original_square_perimeter_l798_79875

theorem original_square_perimeter (x : ℝ) 
  (h1 : ∀ r, r = x ∨ r = 4 * x) 
  (h2 : 28 * x = 56) : 
  4 * (4 * x) = 32 :=
by
  -- We don't need to consider the proof as per instructions.
  sorry

end NUMINAMATH_GPT_original_square_perimeter_l798_79875


namespace NUMINAMATH_GPT_max_square_plots_l798_79870

theorem max_square_plots (width height internal_fence_length : ℕ) 
(h_w : width = 60) (h_h : height = 30) (h_fence: internal_fence_length = 2400) : 
  ∃ n : ℕ, (60 * 30 / (n * n) = 400 ∧ 
  (30 * (60 / n - 1) + 60 * (30 / n - 1) + 60 + 30) ≤ internal_fence_length) :=
sorry

end NUMINAMATH_GPT_max_square_plots_l798_79870


namespace NUMINAMATH_GPT_seating_arrangement_ways_l798_79811

-- Define the problem conditions in Lean 4
def number_of_ways_to_seat (total_chairs : ℕ) (total_people : ℕ) := 
  Nat.factorial total_chairs / Nat.factorial (total_chairs - total_people)

-- Define the specific theorem to be proved
theorem seating_arrangement_ways : number_of_ways_to_seat 8 5 = 6720 :=
by
  sorry

end NUMINAMATH_GPT_seating_arrangement_ways_l798_79811


namespace NUMINAMATH_GPT_find_number_l798_79857

theorem find_number (N : ℝ) 
  (h1 : (5 / 6) * N = (5 / 16) * N + 200) : 
  N = 384 :=
sorry

end NUMINAMATH_GPT_find_number_l798_79857


namespace NUMINAMATH_GPT_square_root_properties_l798_79838

theorem square_root_properties (a : ℝ) (h : a^2 = 10) :
  a = Real.sqrt 10 ∧ (a^2 - 10 = 0) ∧ (3 < a ∧ a < 4) :=
by sorry

end NUMINAMATH_GPT_square_root_properties_l798_79838


namespace NUMINAMATH_GPT_contradiction_example_l798_79899

theorem contradiction_example (a b c d : ℝ) 
(h1 : a + b = 1) 
(h2 : c + d = 1) 
(h3 : ac + bd > 1) : 
¬ (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) :=
by 
  sorry

end NUMINAMATH_GPT_contradiction_example_l798_79899


namespace NUMINAMATH_GPT_total_walnut_trees_l798_79876

-- Define the conditions
def current_walnut_trees := 4
def new_walnut_trees := 6

-- State the lean proof problem
theorem total_walnut_trees : current_walnut_trees + new_walnut_trees = 10 := by
  sorry

end NUMINAMATH_GPT_total_walnut_trees_l798_79876


namespace NUMINAMATH_GPT_Doug_age_l798_79828

theorem Doug_age
  (B : ℕ) (D : ℕ) (N : ℕ)
  (h1 : 2 * B = N)
  (h2 : B + D = 90)
  (h3 : 20 * N = 2000) : 
  D = 40 := sorry

end NUMINAMATH_GPT_Doug_age_l798_79828


namespace NUMINAMATH_GPT_range_of_f_ge_1_l798_79840

noncomputable def f (x : ℝ) : ℝ :=
if x < 1 then (x + 1) ^ 2 else 4 - Real.sqrt (x - 1)

theorem range_of_f_ge_1 :
  {x : ℝ | f x ≥ 1} = {x : ℝ | x ≤ -2} ∪ {x : ℝ | 0 ≤ x ∧ x ≤ 10} :=
by
  sorry

end NUMINAMATH_GPT_range_of_f_ge_1_l798_79840


namespace NUMINAMATH_GPT_remainder_7_mul_12_pow_24_add_2_pow_24_mod_13_l798_79823

theorem remainder_7_mul_12_pow_24_add_2_pow_24_mod_13 :
  (7 * 12^24 + 2^24) % 13 = 8 := by
  sorry

end NUMINAMATH_GPT_remainder_7_mul_12_pow_24_add_2_pow_24_mod_13_l798_79823


namespace NUMINAMATH_GPT_smallest_sum_of_two_squares_l798_79895

theorem smallest_sum_of_two_squares :
  ∃ n : ℕ, (∀ m : ℕ, m < n → (¬ (∃ a b c d e f : ℕ, m = a^2 + b^2 ∧  m = c^2 + d^2 ∧ m = e^2 + f^2 ∧ ((a, b) ≠ (c, d) ∧ (a, b) ≠ (e, f) ∧ (c, d) ≠ (e, f))))) ∧
          (∃ a b c d e f : ℕ, n = a^2 + b^2 ∧  n = c^2 + d^2 ∧ n = e^2 + f^2 ∧ ((a, b) ≠ (c, d) ∧ (a, b) ≠ (e, f) ∧ (c, d) ≠ (e, f))) :=
sorry

end NUMINAMATH_GPT_smallest_sum_of_two_squares_l798_79895


namespace NUMINAMATH_GPT_part_a_l798_79882

theorem part_a (a : ℕ → ℕ) 
  (h₁ : a 1 = 1) 
  (h₂ : a 2 = 1) 
  (h₃ : ∀ n, a (n + 2) = a (n + 1) * a n + 1) :
  ∀ n, ¬ (4 ∣ a n) :=
by
  sorry

end NUMINAMATH_GPT_part_a_l798_79882


namespace NUMINAMATH_GPT_equilateral_triangle_sum_perimeters_l798_79879

theorem equilateral_triangle_sum_perimeters (s : ℝ) (h : ∑' n, 3 * s / 2 ^ n = 360) : 
  s = 60 := 
by 
  sorry

end NUMINAMATH_GPT_equilateral_triangle_sum_perimeters_l798_79879


namespace NUMINAMATH_GPT_probability_of_selection_l798_79815

-- defining necessary parameters and the systematic sampling method
def total_students : ℕ := 52
def selected_students : ℕ := 10
def exclusion_probability := 2 / total_students
def inclusion_probability_exclude := selected_students / (total_students - 2)
def final_probability := (1 - exclusion_probability) * inclusion_probability_exclude

-- the main theorem stating the probability calculation
theorem probability_of_selection :
  final_probability = 5 / 26 :=
by
  -- we skip the proof part and end with sorry since it is not required
  sorry

end NUMINAMATH_GPT_probability_of_selection_l798_79815


namespace NUMINAMATH_GPT_part1_l798_79830

def U : Set ℝ := Set.univ
def A (a : ℝ) : Set ℝ := {x | 0 < 2 * x + a ∧ 2 * x + a ≤ 3}
def B : Set ℝ := {x | -1 / 2 < x ∧ x < 2}

theorem part1 (a : ℝ) (h : a = 1) :
  (Set.compl B ∪ A a) = {x | x ≤ 1 ∨ x ≥ 2} :=
by
  sorry

end NUMINAMATH_GPT_part1_l798_79830


namespace NUMINAMATH_GPT_pizza_cost_l798_79803

theorem pizza_cost
  (initial_money_frank : ℕ)
  (initial_money_bill : ℕ)
  (final_money_bill : ℕ)
  (pizza_cost : ℕ)
  (number_of_pizzas : ℕ)
  (money_given_to_bill : ℕ) :
  initial_money_frank = 42 ∧
  initial_money_bill = 30 ∧
  final_money_bill = 39 ∧
  number_of_pizzas = 3 ∧
  money_given_to_bill = final_money_bill - initial_money_bill →
  3 * pizza_cost + money_given_to_bill = initial_money_frank →
  pizza_cost = 11 :=
by
  sorry

end NUMINAMATH_GPT_pizza_cost_l798_79803


namespace NUMINAMATH_GPT_johns_mistake_l798_79880

theorem johns_mistake (a b : ℕ) (h1 : 10000 * a + b = 11 * a * b)
  (h2 : 100 ≤ a ∧ a ≤ 999) (h3 : 1000 ≤ b ∧ b ≤ 9999) : a + b = 1093 :=
sorry

end NUMINAMATH_GPT_johns_mistake_l798_79880


namespace NUMINAMATH_GPT_volleyball_club_lineups_l798_79884
-- Import the required Lean library

-- Define the main problem
theorem volleyball_club_lineups :
  let total_players := 18
  let quadruplets := 4
  let starters := 6
  let eligible_lineups := Nat.choose 18 6 - Nat.choose 14 2 - Nat.choose 14 6
  eligible_lineups = 15470 :=
by
  sorry

end NUMINAMATH_GPT_volleyball_club_lineups_l798_79884


namespace NUMINAMATH_GPT_lunch_break_duration_l798_79833

def rate_sandra : ℝ := 0 -- Sandra's painting rate in houses per hour
def rate_helpers : ℝ := 0 -- Combined rate of the three helpers in houses per hour
def lunch_break : ℝ := 0 -- Lunch break duration in hours

axiom monday_condition : (8 - lunch_break) * (rate_sandra + rate_helpers) = 0.6
axiom tuesday_condition : (6 - lunch_break) * rate_helpers = 0.3
axiom wednesday_condition : (2 - lunch_break) * rate_sandra = 0.1

theorem lunch_break_duration : lunch_break = 0.5 :=
by {
  sorry
}

end NUMINAMATH_GPT_lunch_break_duration_l798_79833


namespace NUMINAMATH_GPT_parallel_vectors_perpendicular_vectors_obtuse_angle_vectors_l798_79814

section vector

variables {k : ℝ}
def a : ℝ × ℝ := (6, 2)
def b : ℝ × ℝ := (-2, k)

-- Parallel condition
theorem parallel_vectors : 
  (∀ c : ℝ, (6, 2) = -2 * (c * k, c)) → k = -2 / 3 :=
by 
  sorry

-- Perpendicular condition
theorem perpendicular_vectors : 
  6 * (-2) + 2 * k = 0 → k = 6 :=
by 
  sorry

-- Obtuse angle condition
theorem obtuse_angle_vectors : 
  6 * (-2) + 2 * k < 0 ∧ k ≠ -2 / 3 → k < 6 ∧ k ≠ -2 / 3 :=
by 
  sorry

end vector

end NUMINAMATH_GPT_parallel_vectors_perpendicular_vectors_obtuse_angle_vectors_l798_79814


namespace NUMINAMATH_GPT_man_walking_speed_l798_79841

theorem man_walking_speed (length_of_bridge : ℝ) (time_to_cross : ℝ) 
  (h1 : length_of_bridge = 1250) (h2 : time_to_cross = 15) : 
  (length_of_bridge / time_to_cross) * (60 / 1000) = 5 := 
sorry

end NUMINAMATH_GPT_man_walking_speed_l798_79841


namespace NUMINAMATH_GPT_problem_statement_l798_79805

theorem problem_statement 
  (h1 : 17 ≡ 3 [MOD 7])
  (h2 : 3^1 ≡ 3 [MOD 7])
  (h3 : 3^2 ≡ 2 [MOD 7])
  (h4 : 3^3 ≡ 6 [MOD 7])
  (h5 : 3^4 ≡ 4 [MOD 7])
  (h6 : 3^5 ≡ 5 [MOD 7])
  (h7 : 3^6 ≡ 1 [MOD 7])
  (h8 : 3^100 ≡ 4 [MOD 7]) :
  17^100 ≡ 4 [MOD 7] :=
by sorry

end NUMINAMATH_GPT_problem_statement_l798_79805


namespace NUMINAMATH_GPT_max_value_of_y_in_interval_l798_79821

theorem max_value_of_y_in_interval (x : ℝ) (h : 0 < x ∧ x < 1 / 3) : 
  ∃ y_max, ∀ x, 0 < x ∧ x < 1 / 3 → x * (1 - 3 * x) ≤ y_max ∧ y_max = 1 / 12 :=
by sorry

end NUMINAMATH_GPT_max_value_of_y_in_interval_l798_79821


namespace NUMINAMATH_GPT_calculate_value_l798_79818

theorem calculate_value :
  let number := 1.375
  let coef := 0.6667
  let increment := 0.75
  coef * number + increment = 1.666675 :=
by
  sorry

end NUMINAMATH_GPT_calculate_value_l798_79818


namespace NUMINAMATH_GPT_apple_cost_price_l798_79819

theorem apple_cost_price (SP : ℝ) (loss_frac : ℝ) (CP : ℝ) (h_SP : SP = 19) (h_loss_frac : loss_frac = 1 / 6) (h_loss : SP = CP - loss_frac * CP) : CP = 22.8 :=
by
  sorry

end NUMINAMATH_GPT_apple_cost_price_l798_79819


namespace NUMINAMATH_GPT_sum_geometric_series_l798_79869

theorem sum_geometric_series (x : ℂ) (h₀ : x ≠ 1) (h₁ : x^10 - 3*x + 2 = 0) : 
  x^9 + x^8 + x^7 + x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 = 3 := 
by 
  sorry

end NUMINAMATH_GPT_sum_geometric_series_l798_79869


namespace NUMINAMATH_GPT_tree_height_at_two_years_l798_79878

variable (h : ℕ → ℕ)

-- Given conditions
def condition1 := h 4 = 81
def condition2 := ∀ t : ℕ, h (t + 1) = 3 * h t

theorem tree_height_at_two_years
  (h_tripled : ∀ t : ℕ, h (t + 1) = 3 * h t)
  (h_at_four : h 4 = 81) :
  h 2 = 9 :=
by
  -- Formal proof will be provided here
  sorry

end NUMINAMATH_GPT_tree_height_at_two_years_l798_79878


namespace NUMINAMATH_GPT_final_sum_l798_79810

-- Assuming an initial condition for the values on the calculators
def initial_values : List Int := [2, 1, -1]

-- Defining the operations to be applied on the calculators
def operations (vals : List Int) : List Int :=
  match vals with
  | [a, b, c] => [a * a, b * b * b, -c]
  | _ => vals  -- This case handles unexpected input formats

-- Applying the operations for 43 participants
def final_values (vals : List Int) (n : Nat) : List Int :=
  if n = 0 then vals
  else final_values (operations vals) (n - 1)

-- Prove that the final sum of the values on the calculators equals 2 ^ 2 ^ 43
theorem final_sum : 
  final_values initial_values 43 = [2 ^ 2 ^ 43, 1, -1] → 
  List.sum (final_values initial_values 43) = 2 ^ 2 ^ 43 :=
by
  intro h -- This introduces the hypothesis that the final values list equals the expected values
  sorry   -- Provide an ultimate proof for the statement.

end NUMINAMATH_GPT_final_sum_l798_79810


namespace NUMINAMATH_GPT_sum_of_ages_is_24_l798_79861

def age_problem :=
  ∃ (x y z : ℕ), 2 * x^2 + y^2 + z^2 = 194 ∧ (x + x + y + z = 24)

theorem sum_of_ages_is_24 : age_problem :=
by
  sorry

end NUMINAMATH_GPT_sum_of_ages_is_24_l798_79861


namespace NUMINAMATH_GPT_max_ab_of_tangent_circles_l798_79868

theorem max_ab_of_tangent_circles (a b : ℝ) 
  (hC1 : ∀ x y : ℝ, (x - a)^2 + (y + 2)^2 = 4)
  (hC2 : ∀ x y : ℝ, (x + b)^2 + (y + 2)^2 = 1)
  (h_tangent : a + b = 3) :
  ab ≤ 9 / 4 :=
by
  sorry

end NUMINAMATH_GPT_max_ab_of_tangent_circles_l798_79868


namespace NUMINAMATH_GPT_findPerpendicularLine_l798_79856

-- Defining the condition: the line passes through point (-1, 2)
def pointOnLine (x y : ℝ) (a b : ℝ) (c : ℝ) : Prop :=
  a * x + b * y + c = 0

-- Defining the condition: the line is perpendicular to 2x - 3y + 4 = 0
def isPerpendicular (a1 b1 a2 b2 : ℝ) : Prop :=
  a1 * a2 + b1 * b2 = 0

-- The original line equation: 2x - 3y + 4 = 0
def originalLine (x y : ℝ) : Prop :=
  2 * x - 3 * y + 4 = 0

-- The target equation of the line: 3x + 2y - 1 = 0
def targetLine (x y : ℝ) : Prop :=
  3 * x + 2 * y - 1 = 0

theorem findPerpendicularLine :
  (pointOnLine (-1) 2 3 2 (-1)) ∧
  (isPerpendicular 3 2 2 (-3)) →
  (∀ x y, targetLine x y ↔ 3 * x + 2 * y - 1 = 0) :=
by
  sorry

end NUMINAMATH_GPT_findPerpendicularLine_l798_79856


namespace NUMINAMATH_GPT_enclosed_area_is_43pi_l798_79863

noncomputable def enclosed_area (x y : ℝ) : Prop :=
  (x^2 - 6*x + y^2 + 10*y = 9)

theorem enclosed_area_is_43pi :
  (∃ x y : ℝ, enclosed_area x y) → 
  ∃ A : ℝ, A = 43 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_enclosed_area_is_43pi_l798_79863


namespace NUMINAMATH_GPT_minimum_y_value_inequality_proof_l798_79839
-- Import necessary Lean library

-- Define a > 0, b > 0, and a + b = 1
variables {a b : ℝ}
variables (h_a_pos : a > 0) (h_b_pos : b > 0) (h_sum : a + b = 1)

-- Statement for Part (I): Prove the minimum value of y is 25/4
theorem minimum_y_value :
  (a + 1/a) * (b + 1/b) = 25/4 :=
sorry

-- Statement for Part (II): Prove the inequality
theorem inequality_proof :
  (a + 1/a)^2 + (b + 1/b)^2 ≥ 25/2 :=
sorry

end NUMINAMATH_GPT_minimum_y_value_inequality_proof_l798_79839


namespace NUMINAMATH_GPT_remaining_cookies_l798_79859

variable (total_initial_cookies : ℕ)
variable (cookies_taken_day1 : ℕ := 3)
variable (cookies_taken_day2 : ℕ := 3)
variable (cookies_eaten_day2 : ℕ := 1)
variable (cookies_put_back_day2 : ℕ := 2)
variable (cookies_taken_by_junior : ℕ := 7)

theorem remaining_cookies (total_initial_cookies cookies_taken_day1 cookies_taken_day2
                          cookies_eaten_day2 cookies_put_back_day2 cookies_taken_by_junior : ℕ) :
  (total_initial_cookies = 2 * (cookies_taken_day1 + cookies_eaten_day2 + cookies_taken_by_junior))
  → (total_initial_cookies - (cookies_taken_day1 + cookies_eaten_day2 + cookies_taken_by_junior) = 11) :=
by
  sorry

end NUMINAMATH_GPT_remaining_cookies_l798_79859


namespace NUMINAMATH_GPT_B_profit_percentage_l798_79842

theorem B_profit_percentage (cost_price_A : ℝ) (profit_A : ℝ) (selling_price_C : ℝ) 
  (h1 : cost_price_A = 154) 
  (h2 : profit_A = 0.20) 
  (h3 : selling_price_C = 231) : 
  (selling_price_C - (cost_price_A * (1 + profit_A))) / (cost_price_A * (1 + profit_A)) * 100 = 25 :=
by
  sorry

end NUMINAMATH_GPT_B_profit_percentage_l798_79842


namespace NUMINAMATH_GPT_surface_area_of_rectangular_prism_l798_79827

def SurfaceArea (length : ℕ) (width : ℕ) (height : ℕ) : ℕ :=
  2 * ((length * width) + (width * height) + (height * length))

theorem surface_area_of_rectangular_prism 
  (l w h : ℕ) 
  (hl : l = 1) 
  (hw : w = 2) 
  (hh : h = 2) : 
  SurfaceArea l w h = 16 := by
  sorry

end NUMINAMATH_GPT_surface_area_of_rectangular_prism_l798_79827


namespace NUMINAMATH_GPT_bobby_toy_cars_l798_79889

theorem bobby_toy_cars (initial_cars : ℕ) (increase_rate : ℕ → ℕ) (n : ℕ) :
  initial_cars = 16 →
  increase_rate 1 = initial_cars + (initial_cars / 2) →
  increase_rate 2 = increase_rate 1 + (increase_rate 1 / 2) →
  increase_rate 3 = increase_rate 2 + (increase_rate 2 / 2) →
  n = 3 →
  increase_rate n = 54 :=
by
  intros
  sorry

end NUMINAMATH_GPT_bobby_toy_cars_l798_79889


namespace NUMINAMATH_GPT_work_complete_in_15_days_l798_79829

theorem work_complete_in_15_days :
  let A_rate := (1 : ℚ) / 20
  let B_rate := (1 : ℚ) / 30
  let C_rate := (1 : ℚ) / 10
  let all_together_rate := A_rate + B_rate + C_rate
  let work_2_days := 2 * all_together_rate
  let B_C_rate := B_rate + C_rate
  let work_next_2_days := 2 * B_C_rate
  let total_work_4_days := work_2_days + work_next_2_days
  let remaining_work := 1 - total_work_4_days
  let B_time := remaining_work / B_rate

  2 + 2 + B_time = 15 :=
by
  sorry

end NUMINAMATH_GPT_work_complete_in_15_days_l798_79829


namespace NUMINAMATH_GPT_find_a_l798_79801

theorem find_a (z a : ℂ) (h1 : ‖z‖ = 2) (h2 : (z - a)^2 = a) : a = 2 :=
sorry

end NUMINAMATH_GPT_find_a_l798_79801


namespace NUMINAMATH_GPT_probability_B_does_not_lose_l798_79885

def prob_A_wins : ℝ := 0.3
def prob_draw : ℝ := 0.5

-- Theorem: the probability that B does not lose is 70%.
theorem probability_B_does_not_lose : prob_A_wins + prob_draw ≤ 1 → 1 - prob_A_wins - (1 - prob_draw - prob_A_wins) = 0.7 := by
  sorry

end NUMINAMATH_GPT_probability_B_does_not_lose_l798_79885


namespace NUMINAMATH_GPT_common_point_sufficient_condition_l798_79888

theorem common_point_sufficient_condition (k : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 = 1 ∧ y = k * x - 3) → k ≤ -2 * Real.sqrt 2 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_common_point_sufficient_condition_l798_79888


namespace NUMINAMATH_GPT_largest_apartment_size_l798_79848

theorem largest_apartment_size (rent_per_sqft : ℝ) (budget : ℝ) (s : ℝ) :
  rent_per_sqft = 0.9 →
  budget = 630 →
  s = budget / rent_per_sqft →
  s = 700 :=
by
  sorry

end NUMINAMATH_GPT_largest_apartment_size_l798_79848


namespace NUMINAMATH_GPT_largest_possible_n_l798_79831

theorem largest_possible_n (b g : ℕ) (n : ℕ) (h1 : g = 3 * b)
  (h2 : ∀ (boy : ℕ), boy < b → ∀ (girlfriend : ℕ), girlfriend < g → girlfriend ≤ 2013)
  (h3 : ∀ (girl : ℕ), girl < g → ∀ (boyfriend : ℕ), boyfriend < b → boyfriend ≥ n) :
  n ≤ 671 := by
    sorry

end NUMINAMATH_GPT_largest_possible_n_l798_79831


namespace NUMINAMATH_GPT_math_problem_proof_l798_79825

noncomputable def least_n : ℕ := 4

theorem math_problem_proof : 
  ∃ n ≥ 1, ((1 : ℝ) / n) - ((1 : ℝ) / (n + 1)) < 1 / 15 ∧ 
           ∀ m ≥ 1, (((1 : ℝ) / m - (1 : ℝ) / (m + 1)) < 1 / 15) → 
           least_n ≤ m := 
by
  use least_n
  sorry

end NUMINAMATH_GPT_math_problem_proof_l798_79825


namespace NUMINAMATH_GPT_Ben_shirts_is_15_l798_79862

variable (Alex_shirts Joe_shirts Ben_shirts : Nat)

def Alex_has_4 : Alex_shirts = 4 := by sorry

def Joe_has_more_than_Alex : Joe_shirts = Alex_shirts + 3 := by sorry

def Ben_has_more_than_Joe : Ben_shirts = Joe_shirts + 8 := by sorry

theorem Ben_shirts_is_15 (h1 : Alex_shirts = 4) (h2 : Joe_shirts = Alex_shirts + 3) (h3 : Ben_shirts = Joe_shirts + 8) : Ben_shirts = 15 := by
  sorry

end NUMINAMATH_GPT_Ben_shirts_is_15_l798_79862


namespace NUMINAMATH_GPT_largest_K_inequality_l798_79809

theorem largest_K_inequality :
  ∃ K : ℕ, (K < 12) ∧ (10 * K = 110) := by
  use 11
  sorry

end NUMINAMATH_GPT_largest_K_inequality_l798_79809


namespace NUMINAMATH_GPT_sum_of_averages_is_six_l798_79835

variable (a b c d e : ℕ)

def average_teacher : ℚ :=
  (5 * a + 4 * b + 3 * c + 2 * d + e) / (a + b + c + d + e)

def average_kati : ℚ :=
  (5 * e + 4 * d + 3 * c + 2 * b + a) / (a + b + c + d + e)

theorem sum_of_averages_is_six (a b c d e : ℕ) : 
    average_teacher a b c d e + average_kati a b c d e = 6 := by
  sorry

end NUMINAMATH_GPT_sum_of_averages_is_six_l798_79835


namespace NUMINAMATH_GPT_second_set_number_l798_79843

theorem second_set_number (x : ℕ) (sum1 : ℕ) (avg2 : ℕ) (total_avg : ℕ)
  (h1 : sum1 = 98) (h2 : avg2 = 11) (h3 : total_avg = 8)
  (h4 : 16 + x ≠ 0) :
  (98 + avg2 * x = total_avg * (x + 16)) → x = 10 :=
by
  sorry

end NUMINAMATH_GPT_second_set_number_l798_79843


namespace NUMINAMATH_GPT_pieces_of_gum_per_nickel_l798_79804

-- Definitions based on the given conditions
def initial_nickels : ℕ := 5
def remaining_nickels : ℕ := 2
def total_gum_pieces : ℕ := 6

-- We need to prove that Quentavious gets 2 pieces of gum per nickel.
theorem pieces_of_gum_per_nickel 
  (initial_nickels remaining_nickels total_gum_pieces : ℕ)
  (h1 : initial_nickels = 5)
  (h2 : remaining_nickels = 2)
  (h3 : total_gum_pieces = 6) :
  total_gum_pieces / (initial_nickels - remaining_nickels) = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_pieces_of_gum_per_nickel_l798_79804


namespace NUMINAMATH_GPT_probability_at_least_one_each_color_in_bag_l798_79817

open BigOperators

def num_combinations (n k : ℕ) : ℕ :=
  Nat.choose n k

def prob_at_least_one_each_color : ℚ :=
  let total_ways := num_combinations 9 5
  let favorable_ways := 27 + 27 + 27 -- 3 scenarios (2R+1B+2G, 2B+1R+2G, 2G+1R+2B)
  favorable_ways / total_ways

theorem probability_at_least_one_each_color_in_bag :
  prob_at_least_one_each_color = 9 / 14 :=
by
  sorry

end NUMINAMATH_GPT_probability_at_least_one_each_color_in_bag_l798_79817


namespace NUMINAMATH_GPT_martin_spends_30_dollars_on_berries_l798_79872

def cost_per_package : ℝ := 2.0
def cups_per_package : ℝ := 1.0
def cups_per_day : ℝ := 0.5
def days : ℝ := 30

theorem martin_spends_30_dollars_on_berries :
  (days / (cups_per_package / cups_per_day)) * cost_per_package = 30 :=
by
  sorry

end NUMINAMATH_GPT_martin_spends_30_dollars_on_berries_l798_79872


namespace NUMINAMATH_GPT_no_power_of_two_divides_3n_plus_1_l798_79883

theorem no_power_of_two_divides_3n_plus_1 (n : ℕ) (hn : n > 1) : ¬ (2^n ∣ 3^n + 1) := sorry

end NUMINAMATH_GPT_no_power_of_two_divides_3n_plus_1_l798_79883


namespace NUMINAMATH_GPT_simplify_336_to_fraction_l798_79881

theorem simplify_336_to_fraction : (336 / 100) = (84 / 25) :=
by sorry

end NUMINAMATH_GPT_simplify_336_to_fraction_l798_79881


namespace NUMINAMATH_GPT_sufficient_not_necessary_l798_79812

variable (a : ℝ)

theorem sufficient_not_necessary :
  (a > 1 → a^2 > a) ∧ (¬(a > 1) ∧ a^2 > a → a < 0) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_l798_79812


namespace NUMINAMATH_GPT_shirts_sold_correct_l798_79890

-- Define the conditions
def shoes_sold := 6
def cost_per_shoe := 3
def earnings_per_person := 27
def total_earnings := 2 * earnings_per_person
def earnings_from_shoes := shoes_sold * cost_per_shoe
def cost_per_shirt := 2
def earnings_from_shirts := total_earnings - earnings_from_shoes

-- Define the total number of shirts sold and the target value to prove
def shirts_sold : Nat := earnings_from_shirts / cost_per_shirt

-- Prove that shirts_sold is 18
theorem shirts_sold_correct : shirts_sold = 18 := by
  sorry

end NUMINAMATH_GPT_shirts_sold_correct_l798_79890


namespace NUMINAMATH_GPT_polynomial_solution_l798_79800

theorem polynomial_solution (P : ℝ → ℝ) :
  (∀ a b c : ℝ, (a * b + b * c + c * a = 0) → 
  (P (a - b) + P (b - c) + P (c - a) = 2 * P (a + b + c))) →
  ∃ α β : ℝ, ∀ x : ℝ, P x = α * x ^ 4 + β * x ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_solution_l798_79800


namespace NUMINAMATH_GPT_find_square_sum_l798_79850

theorem find_square_sum (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a + b + c = 0) (h5 : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = 2 / 7 :=
by
  sorry

end NUMINAMATH_GPT_find_square_sum_l798_79850


namespace NUMINAMATH_GPT_fraction_checked_by_worker_y_l798_79844

variables (P X Y : ℕ)
variables (defective_rate_x defective_rate_y total_defective_rate : ℚ)
variables (h1 : X + Y = P)
variables (h2 : defective_rate_x = 0.005)
variables (h3 : defective_rate_y = 0.008)
variables (h4 : total_defective_rate = 0.007)
variables (defective_x : ℚ := 0.005 * X)
variables (defective_y : ℚ := 0.008 * Y)
variables (total_defective_products : ℚ := 0.007 * P)
variables (h5 : defective_x + defective_y = total_defective_products)

theorem fraction_checked_by_worker_y : Y / P = 2 / 3 :=
by sorry

end NUMINAMATH_GPT_fraction_checked_by_worker_y_l798_79844


namespace NUMINAMATH_GPT_average_gas_mileage_round_trip_l798_79849

-- necessary definitions related to the problem conditions
def total_distance_one_way := 150
def fuel_efficiency_going := 35
def fuel_efficiency_return := 30
def round_trip_distance := total_distance_one_way + total_distance_one_way

-- calculation of gasoline used for each trip and total usage
def gasoline_used_going := total_distance_one_way / fuel_efficiency_going
def gasoline_used_return := total_distance_one_way / fuel_efficiency_return
def total_gasoline_used := gasoline_used_going + gasoline_used_return

-- calculation of average gas mileage
def average_gas_mileage := round_trip_distance / total_gasoline_used

-- the final theorem to prove the average gas mileage for the round trip 
theorem average_gas_mileage_round_trip : average_gas_mileage = 32 := 
by
  sorry

end NUMINAMATH_GPT_average_gas_mileage_round_trip_l798_79849


namespace NUMINAMATH_GPT_combined_sum_of_interior_numbers_of_eighth_and_ninth_rows_l798_79802

theorem combined_sum_of_interior_numbers_of_eighth_and_ninth_rows :
  (2 ^ (8 - 1) - 2) + (2 ^ (9 - 1) - 2) = 380 :=
by
  -- The steps of the proof would go here, but for the purpose of this task:
  sorry

end NUMINAMATH_GPT_combined_sum_of_interior_numbers_of_eighth_and_ninth_rows_l798_79802


namespace NUMINAMATH_GPT_judy_hits_percentage_l798_79822

theorem judy_hits_percentage 
  (total_hits : ℕ)
  (home_runs : ℕ)
  (triples : ℕ)
  (doubles : ℕ)
  (single_hits_percentage : ℚ) :
  total_hits = 35 →
  home_runs = 1 →
  triples = 1 →
  doubles = 5 →
  single_hits_percentage = (total_hits - (home_runs + triples + doubles)) / total_hits * 100 →
  single_hits_percentage = 80 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_judy_hits_percentage_l798_79822


namespace NUMINAMATH_GPT_jason_work_hours_l798_79874

variable (x y : ℕ)

def working_hours : Prop :=
  (4 * x + 6 * y = 88) ∧
  (x + y = 18)

theorem jason_work_hours (h : working_hours x y) : y = 8 :=
  by
    sorry

end NUMINAMATH_GPT_jason_work_hours_l798_79874


namespace NUMINAMATH_GPT_no_negative_roots_of_P_l798_79877

def P (x : ℝ) : ℝ := x^4 - 5 * x^3 + 3 * x^2 - 7 * x + 1

theorem no_negative_roots_of_P : ∀ x : ℝ, P x = 0 → x ≥ 0 := 
by 
    sorry

end NUMINAMATH_GPT_no_negative_roots_of_P_l798_79877


namespace NUMINAMATH_GPT_sum_of_numbers_l798_79898

variable (x y S : ℝ)
variable (H1 : x + y = S)
variable (H2 : x * y = 375)
variable (H3 : (1 / x) + (1 / y) = 0.10666666666666667)

theorem sum_of_numbers (H1 : x + y = S) (H2 : x * y = 375) (H3 : (1 / x) + (1 / y) = 0.10666666666666667) : S = 40 :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_numbers_l798_79898


namespace NUMINAMATH_GPT_distance_from_home_to_school_l798_79837

theorem distance_from_home_to_school
  (x y : ℝ)
  (h1 : x = y / 3)
  (h2 : x = (y + 18) / 5) : x = 9 := 
by
  sorry

end NUMINAMATH_GPT_distance_from_home_to_school_l798_79837


namespace NUMINAMATH_GPT_jason_fish_count_ninth_day_l798_79886

def fish_growth_day1 := 8 * 3
def fish_growth_day2 := fish_growth_day1 * 3
def fish_growth_day3 := fish_growth_day2 * 3
def fish_day4_removed := 2 / 5 * fish_growth_day3
def fish_after_day4 := fish_growth_day3 - fish_day4_removed
def fish_growth_day5 := fish_after_day4 * 3
def fish_growth_day6 := fish_growth_day5 * 3
def fish_day6_removed := 3 / 7 * fish_growth_day6
def fish_after_day6 := fish_growth_day6 - fish_day6_removed
def fish_growth_day7 := fish_after_day6 * 3
def fish_growth_day8 := fish_growth_day7 * 3
def fish_growth_day9 := fish_growth_day8 * 3
def fish_final := fish_growth_day9 + 20

theorem jason_fish_count_ninth_day : fish_final = 18083 :=
by
  -- proof steps will go here
  sorry

end NUMINAMATH_GPT_jason_fish_count_ninth_day_l798_79886


namespace NUMINAMATH_GPT_derivative_at_zero_l798_79858

noncomputable def f (x : ℝ) : ℝ := x * (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5) * (x - 6)

theorem derivative_at_zero : deriv f 0 = 720 :=
by
  sorry

end NUMINAMATH_GPT_derivative_at_zero_l798_79858


namespace NUMINAMATH_GPT_log_identity_l798_79836

theorem log_identity (c b : ℝ) (h1 : c = Real.log 81 / Real.log 4) (h2 : b = Real.log 3 / Real.log 2) : c = 2 * b := by
  sorry

end NUMINAMATH_GPT_log_identity_l798_79836


namespace NUMINAMATH_GPT_electric_car_travel_distance_l798_79896

theorem electric_car_travel_distance {d_electric d_diesel : ℕ} 
  (h1 : d_diesel = 120) 
  (h2 : d_electric = d_diesel + 50 * d_diesel / 100) : 
  d_electric = 180 := 
by 
  sorry

end NUMINAMATH_GPT_electric_car_travel_distance_l798_79896


namespace NUMINAMATH_GPT_find_sum_of_coefficients_l798_79854

theorem find_sum_of_coefficients
  (a b c d : ℤ)
  (h1 : (x^2 + a * x + b) * (x^2 + c * x + d) = x^4 + x^3 - 2 * x^2 + 17 * x - 5) :
  a + b + c + d = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_sum_of_coefficients_l798_79854


namespace NUMINAMATH_GPT_sin_675_eq_neg_sqrt2_div_2_l798_79852

theorem sin_675_eq_neg_sqrt2_div_2 : Real.sin (675 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_sin_675_eq_neg_sqrt2_div_2_l798_79852


namespace NUMINAMATH_GPT_sector_area_120_deg_radius_3_l798_79826

theorem sector_area_120_deg_radius_3 (r : ℝ) (theta_deg : ℝ) (theta_rad : ℝ) (A : ℝ)
  (h1 : r = 3)
  (h2 : theta_deg = 120)
  (h3 : theta_rad = (2 * Real.pi / 3))
  (h4 : A = (1 / 2) * theta_rad * r^2) :
  A = 3 * Real.pi :=
  sorry

end NUMINAMATH_GPT_sector_area_120_deg_radius_3_l798_79826


namespace NUMINAMATH_GPT_third_term_of_sequence_l798_79834

theorem third_term_of_sequence (a : ℕ → ℚ) (h₁ : a 1 = 1) (h₂ : ∀ n, a (n + 1) = (1 / 2) * a n + (1 / (2 * n))) : a 3 = 3 / 4 := by
  sorry

end NUMINAMATH_GPT_third_term_of_sequence_l798_79834


namespace NUMINAMATH_GPT_max_elements_of_valid_set_l798_79813

def valid_set (M : Finset ℤ) : Prop :=
  ∀ (a b c : ℤ), a ∈ M → b ∈ M → c ∈ M → (a ≠ b ∧ b ≠ c ∧ a ≠ c) →
  (a + b ∈ M ∨ a + c ∈ M ∨ b + c ∈ M)

theorem max_elements_of_valid_set (M : Finset ℤ) (h : valid_set M) : M.card ≤ 7 :=
sorry

end NUMINAMATH_GPT_max_elements_of_valid_set_l798_79813


namespace NUMINAMATH_GPT_moles_of_HCl_required_l798_79806

noncomputable def numberOfMolesHClRequired (moles_AgNO3 : ℕ) : ℕ :=
  if moles_AgNO3 = 3 then 3 else 0

-- Theorem statement
theorem moles_of_HCl_required : numberOfMolesHClRequired 3 = 3 := by
  sorry

end NUMINAMATH_GPT_moles_of_HCl_required_l798_79806


namespace NUMINAMATH_GPT_melanie_sale_revenue_correct_l798_79853

noncomputable def melanie_revenue : ℝ :=
let red_cost := 0.08
let green_cost := 0.10
let yellow_cost := 0.12
let red_gumballs := 15
let green_gumballs := 18
let yellow_gumballs := 22
let total_gumballs := red_gumballs + green_gumballs + yellow_gumballs
let total_cost := (red_cost * red_gumballs) + (green_cost * green_gumballs) + (yellow_cost * yellow_gumballs)
let discount := if total_gumballs >= 20 then 0.30 else if total_gumballs >= 10 then 0.20 else 0
let final_cost := total_cost * (1 - discount)
final_cost

theorem melanie_sale_revenue_correct : melanie_revenue = 3.95 :=
by
  -- All calculations and proofs omitted for brevity, as per instructions above
  sorry

end NUMINAMATH_GPT_melanie_sale_revenue_correct_l798_79853


namespace NUMINAMATH_GPT_brad_reads_26_pages_per_day_l798_79897

-- Define conditions
def greg_daily_reading : ℕ := 18
def brad_extra_pages : ℕ := 8

-- Define Brad's daily reading
def brad_daily_reading : ℕ := greg_daily_reading + brad_extra_pages

-- The theorem to be proven
theorem brad_reads_26_pages_per_day : brad_daily_reading = 26 := by
  sorry

end NUMINAMATH_GPT_brad_reads_26_pages_per_day_l798_79897


namespace NUMINAMATH_GPT_income_increase_l798_79846

-- Definitions based on conditions
def original_price := 1.0
def original_items := 100.0
def discount := 0.10
def increased_sales := 0.15

-- Calculations for new values
def new_price := original_price * (1 - discount)
def new_items := original_items * (1 + increased_sales)
def original_income := original_price * original_items
def new_income := new_price * new_items

-- The percentage increase in income
def percentage_increase := ((new_income - original_income) / original_income) * 100

-- The theorem to prove that the percentage increase in gross income is 3.5%
theorem income_increase : percentage_increase = 3.5 := 
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_income_increase_l798_79846


namespace NUMINAMATH_GPT_total_points_other_team_members_l798_79891

variable (x y : ℕ)

theorem total_points_other_team_members :
  (1 / 3 * x + 3 / 8 * x + 18 + y = x) ∧ (y ≤ 24) → y = 17 :=
by
  intro h
  have h1 : 1 / 3 * x + 3 / 8 * x + 18 + y = x := h.1
  have h2 : y ≤ 24 := h.2
  sorry

end NUMINAMATH_GPT_total_points_other_team_members_l798_79891
