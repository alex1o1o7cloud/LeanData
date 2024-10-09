import Mathlib

namespace find_x_l1974_197472

theorem find_x :
  ∀ (x y z w : ℕ), 
    x = y + 5 →
    y = z + 10 →
    z = w + 20 →
    w = 80 →
    x = 115 :=
by
  intros x y z w h1 h2 h3 h4
  sorry

end find_x_l1974_197472


namespace total_students_in_both_classrooms_l1974_197431

theorem total_students_in_both_classrooms
  (x y : ℕ)
  (hx1 : 80 * x - 250 = 90 * (x - 5))
  (hy1 : 85 * y - 480 = 95 * (y - 8)) :
  x + y = 48 := 
sorry

end total_students_in_both_classrooms_l1974_197431


namespace range_of_m_l1974_197470

theorem range_of_m (m : ℝ) : (∀ x : ℝ, m * x ^ 2 - m * x + 1 > 0) ↔ (0 ≤ m ∧ m < 4) :=
sorry

end range_of_m_l1974_197470


namespace bridge_weight_requirement_l1974_197477

def weight_soda_can : ℕ := 12
def weight_empty_soda_can : ℕ := 2
def num_soda_cans : ℕ := 6

def weight_empty_other_can : ℕ := 3
def num_other_cans : ℕ := 2

def wind_force_eq_soda_cans : ℕ := 2

def total_weight_bridge_must_hold : ℕ :=
  weight_soda_can * num_soda_cans + weight_empty_soda_can * num_soda_cans +
  weight_empty_other_can * num_other_cans +
  wind_force_eq_soda_cans * (weight_soda_can + weight_empty_soda_can)

theorem bridge_weight_requirement :
  total_weight_bridge_must_hold = 118 :=
by
  unfold total_weight_bridge_must_hold weight_soda_can weight_empty_soda_can num_soda_cans
    weight_empty_other_can num_other_cans wind_force_eq_soda_cans
  sorry

end bridge_weight_requirement_l1974_197477


namespace time_taken_by_abc_l1974_197439

-- Define the work rates for a, b, and c
def work_rate_a_b : ℚ := 1 / 15
def work_rate_c : ℚ := 1 / 41.25

-- Define the combined work rate for a, b, and c
def combined_work_rate : ℚ := work_rate_a_b + work_rate_c

-- Define the reciprocal of the combined work rate, which is the time taken
def time_taken : ℚ := 1 / combined_work_rate

-- Prove that the time taken by a, b, and c together is 11 days
theorem time_taken_by_abc : time_taken = 11 := by
  -- Substitute the values to compute the result
  sorry

end time_taken_by_abc_l1974_197439


namespace primitive_root_set_equality_l1974_197434

theorem primitive_root_set_equality 
  {p : ℕ} (hp : Nat.Prime p) (hodd: p % 2 = 1) (g : ℕ) (hg : g ^ (p - 1) % p = 1) :
  (∀ k, 1 ≤ k ∧ k ≤ (p - 1) / 2 → ∃ m, 1 ≤ m ∧ m ≤ (p - 1) / 2 ∧ (k^2 + 1) % p = g ^ m % p) ↔ p = 3 :=
by sorry

end primitive_root_set_equality_l1974_197434


namespace rope_purchases_l1974_197448

theorem rope_purchases (last_week_rope_feet : ℕ) (less_rope : ℕ) (feet_to_inches : ℕ) 
  (h1 : last_week_rope_feet = 6) 
  (h2 : less_rope = 4) 
  (h3 : feet_to_inches = 12) : 
  (last_week_rope_feet * feet_to_inches) + ((last_week_rope_feet - less_rope) * feet_to_inches) = 96 := 
by
  sorry

end rope_purchases_l1974_197448


namespace converse_true_inverse_true_contrapositive_false_sufficiency_necessity_l1974_197478

-- Define the original proposition with conditions
def prop : Prop := ∀ (m n : ℝ), m ≤ 0 ∨ n ≤ 0 → m + n ≤ 0

-- Identify converse, inverse, and contrapositive
def converse : Prop := ∀ (m n : ℝ), m + n ≤ 0 → m ≤ 0 ∨ n ≤ 0
def inverse : Prop := ∀ (m n : ℝ), m > 0 ∧ n > 0 → m + n > 0
def contrapositive : Prop := ∀ (m n : ℝ), m + n > 0 → m > 0 ∧ n > 0

-- Identifying the conditions of sufficiency and necessity
def necessary_but_not_sufficient (p q : Prop) : Prop := 
  (¬p → ¬q) ∧ (q → p) ∧ ¬(p → q)

-- Prove or provide the statements
theorem converse_true : converse := sorry
theorem inverse_true : inverse := sorry
theorem contrapositive_false : ¬contrapositive := sorry
theorem sufficiency_necessity : necessary_but_not_sufficient 
  (∀ (m n : ℝ), m ≤ 0 ∨ n ≤ 0) 
  (∀ (m n : ℝ), m + n ≤ 0) := sorry

end converse_true_inverse_true_contrapositive_false_sufficiency_necessity_l1974_197478


namespace perimeter_of_triangle_l1974_197487

namespace TrianglePerimeter

variables {a b c : ℝ}

-- Conditions translated into definitions
def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def absolute_sum_condition (a b c : ℝ) : Prop :=
  |a + b - c| + |b + c - a| + |c + a - b| = 12

-- The theorem stating the perimeter under given conditions
theorem perimeter_of_triangle (h : is_valid_triangle a b c) (h_abs_sum : absolute_sum_condition a b c) : 
  a + b + c = 12 := 
sorry

end TrianglePerimeter

end perimeter_of_triangle_l1974_197487


namespace smallest_consecutive_sum_l1974_197463

theorem smallest_consecutive_sum (n : ℤ) (h : n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 210) : 
  n = 40 := 
sorry

end smallest_consecutive_sum_l1974_197463


namespace product_of_three_consecutive_cubes_divisible_by_504_l1974_197406

theorem product_of_three_consecutive_cubes_divisible_by_504 (a : ℤ) : 
  ∃ k : ℤ, (a^3 - 1) * a^3 * (a^3 + 1) = 504 * k :=
by
  -- Proof omitted
  sorry

end product_of_three_consecutive_cubes_divisible_by_504_l1974_197406


namespace edward_spring_earnings_l1974_197496

-- Define the relevant constants and the condition
def springEarnings := 2
def summerEarnings := 27
def expenses := 5
def totalEarnings := 24

-- The condition
def edwardCondition := summerEarnings - expenses = 22

-- The statement to prove
theorem edward_spring_earnings (h : edwardCondition) : springEarnings + 22 = totalEarnings :=
by
  -- Provide the proof here, but we'll use sorry to skip it
  sorry

end edward_spring_earnings_l1974_197496


namespace Iris_pairs_of_pants_l1974_197498

theorem Iris_pairs_of_pants (jacket_cost short_cost pant_cost total_spent n_jackets n_shorts n_pants : ℕ) :
  (jacket_cost = 10) →
  (short_cost = 6) →
  (pant_cost = 12) →
  (total_spent = 90) →
  (n_jackets = 3) →
  (n_shorts = 2) →
  (n_jackets * jacket_cost + n_shorts * short_cost + n_pants * pant_cost = total_spent) →
  (n_pants = 4) := 
by
  intros h_jacket_cost h_short_cost h_pant_cost h_total_spent h_n_jackets h_n_shorts h_eq
  sorry

end Iris_pairs_of_pants_l1974_197498


namespace cos2_add_3sin2_eq_2_l1974_197462

theorem cos2_add_3sin2_eq_2 (x : ℝ) (hx : -20 < x ∧ x < 100) (h : Real.cos x ^ 2 + 3 * Real.sin x ^ 2 = 2) : 
  ∃ n : ℕ, n = 38 := 
sorry

end cos2_add_3sin2_eq_2_l1974_197462


namespace percentage_of_goals_by_two_players_l1974_197479

-- Definitions from conditions
def total_goals_league := 300
def goals_per_player := 30
def number_of_players := 2

-- Mathematically equivalent proof problem
theorem percentage_of_goals_by_two_players :
  let combined_goals := number_of_players * goals_per_player
  let percentage := (combined_goals / total_goals_league : ℝ) * 100 
  percentage = 20 :=
by
  sorry

end percentage_of_goals_by_two_players_l1974_197479


namespace greatest_two_digit_number_l1974_197430

theorem greatest_two_digit_number (x y : ℕ) (h1 : x < y) (h2 : x * y = 12) : 10 * x + y = 34 :=
sorry

end greatest_two_digit_number_l1974_197430


namespace fraction_yellow_surface_area_l1974_197497

theorem fraction_yellow_surface_area
  (cube_edge : ℕ)
  (small_cubes : ℕ)
  (yellow_cubes : ℕ)
  (total_surface_area : ℕ)
  (yellow_surface_area : ℕ)
  (fraction_yellow : ℚ) :
  cube_edge = 4 ∧
  small_cubes = 64 ∧
  yellow_cubes = 15 ∧
  total_surface_area = 6 * cube_edge * cube_edge ∧
  yellow_surface_area = 16 ∧
  fraction_yellow = yellow_surface_area / total_surface_area →
  fraction_yellow = 1/6 :=
by
  sorry

end fraction_yellow_surface_area_l1974_197497


namespace number_of_students_to_bring_donuts_l1974_197415

theorem number_of_students_to_bring_donuts (students_brownies students_cookies students_donuts : ℕ) :
  (students_brownies * 12 * 2) + (students_cookies * 24 * 2) + (students_donuts * 12 * 2) = 2040 →
  students_brownies = 30 →
  students_cookies = 20 →
  students_donuts = 15 :=
by
  -- Proof skipped
  sorry

end number_of_students_to_bring_donuts_l1974_197415


namespace area_of_rhombus_l1974_197413

-- Given values for the diagonals of a rhombus.
def d1 : ℝ := 14
def d2 : ℝ := 24

-- The target statement we want to prove.
theorem area_of_rhombus : (d1 * d2) / 2 = 168 := by
  sorry

end area_of_rhombus_l1974_197413


namespace total_weight_of_rhinos_l1974_197488

def white_rhino_weight : ℕ := 5100
def black_rhino_weight : ℕ := 2000

theorem total_weight_of_rhinos :
  7 * white_rhino_weight + 8 * black_rhino_weight = 51700 :=
by
  sorry

end total_weight_of_rhinos_l1974_197488


namespace prime_9_greater_than_perfect_square_l1974_197486

theorem prime_9_greater_than_perfect_square (p : ℕ) (hp : Nat.Prime p) :
  ∃ n m : ℕ, p - 9 = n^2 ∧ p + 2 = m^2 ∧ p = 23 :=
by
  sorry

end prime_9_greater_than_perfect_square_l1974_197486


namespace james_income_ratio_l1974_197440

theorem james_income_ratio
  (January_earnings : ℕ := 4000)
  (Total_earnings : ℕ := 18000)
  (Earnings_difference : ℕ := 2000) :
  ∃ (February_earnings : ℕ), 
    (January_earnings + February_earnings + (February_earnings - Earnings_difference) = Total_earnings) ∧
    (February_earnings / January_earnings = 2) := by
  sorry

end james_income_ratio_l1974_197440


namespace money_distribution_l1974_197414

theorem money_distribution (Maggie_share : ℝ) (fraction_Maggie : ℝ) (total_sum : ℝ) :
  Maggie_share = 7500 →
  fraction_Maggie = (1/8) →
  total_sum = Maggie_share / fraction_Maggie →
  total_sum = 60000 :=
by 
  intros h1 h2 h3
  rw [h1, h2] at h3
  linarith

end money_distribution_l1974_197414


namespace car_y_start_time_l1974_197482

theorem car_y_start_time : 
  ∀ (t m : ℝ), 
  (35 * (t + m) = 294) ∧ (40 * t = 294) → 
  t = 7.35 ∧ m = 1.05 → 
  m * 60 = 63 :=
by
  intros t m h1 h2
  sorry

end car_y_start_time_l1974_197482


namespace M_is_set_of_positive_rationals_le_one_l1974_197443

def M : Set ℚ := {x | 0 < x ∧ x ≤ 1}

axiom contains_one (M : Set ℚ) : 1 ∈ M

axiom closed_under_operations (M : Set ℚ) :
  ∀ x ∈ M, (1 / (1 + x) ∈ M) ∧ (x / (1 + x) ∈ M)

theorem M_is_set_of_positive_rationals_le_one :
  M = {x | 0 < x ∧ x ≤ 1} :=
sorry

end M_is_set_of_positive_rationals_le_one_l1974_197443


namespace maximum_term_of_sequence_l1974_197468

open Real

noncomputable def seq (n : ℕ) : ℝ := n / (n^2 + 81)

theorem maximum_term_of_sequence : ∃ n : ℕ, seq n = 1 / 18 ∧ ∀ k : ℕ, seq k ≤ 1 / 18 :=
by
  sorry

end maximum_term_of_sequence_l1974_197468


namespace locus_of_centers_l1974_197407

theorem locus_of_centers (a b : ℝ) :
  (∃ r : ℝ, a^2 + b^2 = (r + 2)^2 ∧ (a - 3)^2 + b^2 = (9 - r)^2) →
  12 * a^2 + 169 * b^2 - 36 * a - 1584 = 0 :=
by
  sorry

end locus_of_centers_l1974_197407


namespace factorize_2mn_cube_arithmetic_calculation_l1974_197400

-- Problem 1: Factorization problem
theorem factorize_2mn_cube (m n : ℝ) : 
  2 * m^3 * n - 8 * m * n^3 = 2 * m * n * (m + 2 * n) * (m - 2 * n) :=
by sorry

-- Problem 2: Arithmetic calculation problem
theorem arithmetic_calculation : 
  |1 - Real.sqrt 3| + 3 * Real.tan (Real.pi / 6) - ((Real.pi - 3)^0) + (-1/3)⁻¹ = 2 * Real.sqrt 3 - 5 :=
by sorry

end factorize_2mn_cube_arithmetic_calculation_l1974_197400


namespace evaluate_fraction_l1974_197465

theorem evaluate_fraction (a b : ℝ) (h1 : a = 5) (h2 : b = 3) : 3 / (a + b) = 3 / 8 :=
by
  rw [h1, h2]
  sorry

end evaluate_fraction_l1974_197465


namespace arithmetic_sequence_sum_l1974_197459

theorem arithmetic_sequence_sum
  (a l : ℤ) (n d : ℤ)
  (h1 : a = -5) (h2 : l = 40) (h3 : d = 5)
  (h4 : l = a + (n - 1) * d) :
  (n / 2) * (a + l) = 175 :=
by
  sorry

end arithmetic_sequence_sum_l1974_197459


namespace fraction_calculation_l1974_197453

theorem fraction_calculation :
  (3 / 4) * (1 / 2) * (2 / 5) * 5060 = 759 :=
by
  sorry

end fraction_calculation_l1974_197453


namespace total_area_is_71_l1974_197457

-- Define the lengths of the segments
def length_left : ℕ := 7
def length_top : ℕ := 6
def length_middle_1 : ℕ := 2
def length_middle_2 : ℕ := 4
def length_right : ℕ := 1
def length_right_top : ℕ := 5

-- Define the rectangles and their areas
def area_left_rect : ℕ := length_left * length_left
def area_middle_rect : ℕ := length_middle_1 * (length_top - length_left)
def area_right_rect : ℕ := length_middle_2 * length_middle_2

-- Define the total area
def total_area : ℕ := area_left_rect + area_middle_rect + area_right_rect

-- Theorem: The total area of the figure is 71 square units
theorem total_area_is_71 : total_area = 71 := by
  sorry

end total_area_is_71_l1974_197457


namespace zero_of_f_l1974_197420

noncomputable def f (x : ℝ) : ℝ := 2^x - 4

theorem zero_of_f : f 2 = 0 :=
by
  sorry

end zero_of_f_l1974_197420


namespace isosceles_triangle_l1974_197412

theorem isosceles_triangle 
  {a b : ℝ} {α β : ℝ} 
  (h : a / (Real.cos α) = b / (Real.cos β)) : 
  a = b :=
sorry

end isosceles_triangle_l1974_197412


namespace smaller_tank_capacity_l1974_197474

/-- Problem Statement:
Three-quarters of the oil from a certain tank (that was initially full) was poured into a
20000-liter capacity tanker that already had 3000 liters of oil.
To make the large tanker half-full, 4000 more liters of oil would be needed.
What is the capacity of the smaller tank?
-/

theorem smaller_tank_capacity (C : ℝ) 
  (h1 : 3 / 4 * C + 3000 + 4000 = 10000) : 
  C = 4000 :=
sorry

end smaller_tank_capacity_l1974_197474


namespace charge_difference_percentage_l1974_197424

-- Given definitions
variables (G R P : ℝ)
def hotelR := 1.80 * G
def hotelP := 0.90 * G

-- Theorem statement
theorem charge_difference_percentage (G : ℝ) (hR : R = 1.80 * G) (hP : P = 0.90 * G) :
  (R - P) / R * 100 = 50 :=
by sorry

end charge_difference_percentage_l1974_197424


namespace total_canoes_built_l1974_197489

-- Definitions of conditions
def initial_canoes : ℕ := 8
def common_ratio : ℕ := 2
def number_of_months : ℕ := 6

-- Sum of a geometric sequence formula
-- Sₙ = a * (r^n - 1) / (r - 1)
def sum_of_geometric_sequence (a r n : ℕ) : ℕ := 
  a * (r^n - 1) / (r - 1)

-- Statement to prove
theorem total_canoes_built : 504 = sum_of_geometric_sequence initial_canoes common_ratio number_of_months := 
  by
  sorry

end total_canoes_built_l1974_197489


namespace initial_music_files_l1974_197483

-- Define the conditions
def video_files : ℕ := 21
def deleted_files : ℕ := 23
def remaining_files : ℕ := 2

-- Theorem to prove the initial number of music files
theorem initial_music_files : 
  ∃ (M : ℕ), (M + video_files - deleted_files = remaining_files) → M = 4 := 
sorry

end initial_music_files_l1974_197483


namespace range_of_c_over_a_l1974_197403

theorem range_of_c_over_a (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + 2 * b + c = 0) :
    -3 < c / a ∧ c / a < -(1 / 3) := 
sorry

end range_of_c_over_a_l1974_197403


namespace eval_expr_l1974_197444

theorem eval_expr (b c : ℕ) (hb : b = 2) (hc : c = 5) : b^3 * b^4 * c^2 = 3200 :=
by {
  -- the proof is omitted
  sorry
}

end eval_expr_l1974_197444


namespace figure_Z_has_largest_shaded_area_l1974_197401

noncomputable def shaded_area_X :=
  let rectangle_area := 4 * 2
  let circle_area := Real.pi * (1)^2
  rectangle_area - circle_area

noncomputable def shaded_area_Y :=
  let rectangle_area := 4 * 2
  let semicircle_area := (1 / 2) * Real.pi * (1)^2
  rectangle_area - semicircle_area

noncomputable def shaded_area_Z :=
  let outer_square_area := 4^2
  let inner_square_area := 2^2
  outer_square_area - inner_square_area

theorem figure_Z_has_largest_shaded_area :
  shaded_area_Z > shaded_area_X ∧ shaded_area_Z > shaded_area_Y :=
by
  sorry

end figure_Z_has_largest_shaded_area_l1974_197401


namespace pencils_brought_l1974_197469

-- Given conditions
variables (A B : ℕ)

-- There are 7 people in total
def total_people : Prop := A + B = 7

-- 11 charts in total
def total_charts : Prop := A + 2 * B = 11

-- Question: Total pencils
def total_pencils : ℕ := 2 * A + B

-- Statement to be proved
theorem pencils_brought
  (h1 : total_people A B)
  (h2 : total_charts A B) :
  total_pencils A B = 10 := by
  sorry

end pencils_brought_l1974_197469


namespace range_of_a_l1974_197409

theorem range_of_a {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h1 : x + y + 4 = 2 * x * y) (h2 : ∀ (x y : ℝ), x^2 + 2*x*y + y^2 - a*x - a*y + 1 ≥ 0) :
  a ≤ 17/4 := sorry

end range_of_a_l1974_197409


namespace verify_sub_by_add_verify_sub_by_sub_verify_mul_by_div1_verify_mul_by_div2_verify_mul_by_mul_l1974_197466

variable (A B C P M N : ℝ)

-- Verification of Subtraction by Addition
theorem verify_sub_by_add (h : A - B = C) : C + B = A :=
sorry

-- Verification of Subtraction by Subtraction
theorem verify_sub_by_sub (h : A - B = C) : A - C = B :=
sorry

-- Verification of Multiplication by Division (1)
theorem verify_mul_by_div1 (h : M * N = P) : P / N = M :=
sorry

-- Verification of Multiplication by Division (2)
theorem verify_mul_by_div2 (h : M * N = P) : P / M = N :=
sorry

-- Verification of Multiplication by Multiplication
theorem verify_mul_by_mul (h : M * N = P) : M * N = P :=
sorry

end verify_sub_by_add_verify_sub_by_sub_verify_mul_by_div1_verify_mul_by_div2_verify_mul_by_mul_l1974_197466


namespace avg_height_country_l1974_197452

-- Define the parameters for the number of boys and their average heights
def num_boys_north : ℕ := 300
def num_boys_south : ℕ := 200
def avg_height_north : ℝ := 1.60
def avg_height_south : ℝ := 1.50

-- Define the total number of boys
def total_boys : ℕ := num_boys_north + num_boys_south

-- Define the total combined height
def total_height : ℝ := (num_boys_north * avg_height_north) + (num_boys_south * avg_height_south)

-- Prove that the average height of all boys combined is 1.56 meters
theorem avg_height_country : total_height / total_boys = 1.56 := by
  sorry

end avg_height_country_l1974_197452


namespace total_people_in_line_l1974_197411

theorem total_people_in_line (n_front n_behind : ℕ) (hfront : n_front = 11) (hbehind : n_behind = 12) : n_front + n_behind + 1 = 24 := by
  sorry

end total_people_in_line_l1974_197411


namespace f_at_4_l1974_197436

-- Define the conditions on the function f
variable (f : ℝ → ℝ)
variable (h_domain : true) -- All ℝ → ℝ functions have ℝ as their domain.

-- f is an odd function
axiom h_odd : ∀ x : ℝ, f (-x) = -f x

-- Given functional equation
axiom h_eqn : ∀ x : ℝ, f (2 * x - 3) - 2 * f (3 * x - 10) + f (x - 3) = 28 - 6 * x 

-- The goal is to determine the value of f(4), which should be 8.
theorem f_at_4 : f 4 = 8 :=
sorry

end f_at_4_l1974_197436


namespace probability_of_collinear_dots_l1974_197422

theorem probability_of_collinear_dots (dots : ℕ) (rows : ℕ) (columns : ℕ) (choose : ℕ → ℕ → ℕ) :
  dots = 20 ∧ rows = 5 ∧ columns = 4 ∧ choose 20 4 = 4845 → 
  (∃ sets_of_collinear_dots : ℕ, sets_of_collinear_dots = 20 ∧ 
   ∃ probability : ℚ,  probability = 4 / 969) :=
by
  sorry

end probability_of_collinear_dots_l1974_197422


namespace no_non_square_number_with_triple_product_divisors_l1974_197484

theorem no_non_square_number_with_triple_product_divisors (N : ℕ) (h_non_square : ∀ k : ℕ, k * k ≠ N) : 
  ¬ (∃ t : ℕ, ∃ d : Finset (Finset ℕ), (∀ s ∈ d, s.card = 3) ∧ (∀ s ∈ d, s.prod id = t)) := 
sorry

end no_non_square_number_with_triple_product_divisors_l1974_197484


namespace sum_of_altitudes_at_least_nine_times_inradius_l1974_197456

variables (a b c : ℝ)
variables (s : ℝ) -- semiperimeter
variables (Δ : ℝ) -- area
variables (r : ℝ) -- inradius
variables (h_A h_B h_C : ℝ) -- altitudes

-- The Lean statement of the problem
theorem sum_of_altitudes_at_least_nine_times_inradius
  (ha : s = (a + b + c) / 2)
  (hb : Δ = r * s)
  (hc : h_A = (2 * Δ) / a)
  (hd : h_B = (2 * Δ) / b)
  (he : h_C = (2 * Δ) / c) :
  h_A + h_B + h_C ≥ 9 * r :=
sorry

end sum_of_altitudes_at_least_nine_times_inradius_l1974_197456


namespace sum_ages_is_13_l1974_197410

-- Define the variables for the ages
variables (a b c : ℕ)

-- Define the conditions given in the problem
def conditions : Prop :=
  a * b * c = 72 ∧ a < b ∧ c < b

-- State the theorem to be proved
theorem sum_ages_is_13 (h : conditions a b c) : a + b + c = 13 :=
sorry

end sum_ages_is_13_l1974_197410


namespace fraction_simplify_l1974_197433

theorem fraction_simplify (x : ℝ) (hx : x ≠ 1) (hx_ne_1 : x ≠ -1) :
  (x^2 - 1) / (x^2 - 2 * x + 1) = (x + 1) / (x - 1) :=
by
  sorry

end fraction_simplify_l1974_197433


namespace q_negative_one_is_minus_one_l1974_197480

-- Define the function q and the point on the graph
def q (x : ℝ) : ℝ := sorry

-- The condition: point (-1, -1) lies on the graph of q
axiom point_on_graph : q (-1) = -1

-- The theorem to prove that q(-1) = -1
theorem q_negative_one_is_minus_one : q (-1) = -1 :=
by exact point_on_graph

end q_negative_one_is_minus_one_l1974_197480


namespace prove_divisibility_l1974_197426

-- Definitions for natural numbers m, n, k
variables (m n k : ℕ)

-- Conditions stating divisibility
def div1 := m^n ∣ n^m
def div2 := n^k ∣ k^n

-- The final theorem to prove
theorem prove_divisibility (hmn : div1 m n) (hnk : div2 n k) : m^k ∣ k^m :=
sorry

end prove_divisibility_l1974_197426


namespace sqrt_fraction_value_l1974_197428

theorem sqrt_fraction_value (a b c d : Nat) (h : a = 2 ∧ b = 0 ∧ c = 2 ∧ d = 3) : 
  Real.sqrt (2023 / (a + b + c + d)) = 17 := by
  sorry

end sqrt_fraction_value_l1974_197428


namespace total_legs_camden_dogs_l1974_197417

variable (c r j : ℕ) -- c: Camden's dogs, r: Rico's dogs, j: Justin's dogs

theorem total_legs_camden_dogs :
  (r = j + 10) ∧ (j = 14) ∧ (c = (3 * r) / 4) → 4 * c = 72 :=
by
  sorry

end total_legs_camden_dogs_l1974_197417


namespace renovation_project_total_l1974_197429

def sand : ℝ := 0.17
def dirt : ℝ := 0.33
def cement : ℝ := 0.17

theorem renovation_project_total : sand + dirt + cement = 0.67 := 
by
  sorry

end renovation_project_total_l1974_197429


namespace inequality_proof_l1974_197402

theorem inequality_proof (a b c : ℝ) (h : a + b + c = 3) : 
  (1 / (5 * a^2 - 4 * a + 11)) + (1 / (5 * b^2 - 4 * b + 11)) + (1 / (5 * c^2 - 4 * c + 11)) ≤ 1 / 4 := 
by
  -- proof steps will be here
  sorry

end inequality_proof_l1974_197402


namespace num_seven_digit_numbers_l1974_197491

theorem num_seven_digit_numbers (a b c d e f g : ℕ)
  (h1 : a * b * c = 30)
  (h2 : c * d * e = 7)
  (h3 : e * f * g = 15) :
  ∃ n : ℕ, n = 4 := 
sorry

end num_seven_digit_numbers_l1974_197491


namespace water_addition_to_achieve_concentration_l1974_197423

theorem water_addition_to_achieve_concentration :
  ∀ (w1 w2 : ℝ), 
  (60 * 0.25 = 15) →              -- initial amount of acid
  (15 / (60 + w1) = 0.15) →       -- first dilution to 15%
  (15 / (100 + w2) = 0.10) →      -- second dilution to 10%
  w1 + w2 = 90 :=                 -- total water added to achieve final concentration
by
  intros w1 w2 h_initial h_first h_second
  sorry

end water_addition_to_achieve_concentration_l1974_197423


namespace find_tan_angle_F2_F1_B_l1974_197421

-- Definitions for the points and chord lengths
def F1 : Type := ℝ × ℝ
def F2 : Type := ℝ × ℝ
def A : Type := ℝ × ℝ
def B : Type := ℝ × ℝ

-- Given distances
def F1A : ℝ := 3
def AB : ℝ := 4
def BF1 : ℝ := 5

-- The angle we want to find the tangent of
def angle_F2_F1_B (F1 F2 A B : Type) : ℝ := sorry -- Placeholder for angle calculation

-- The main theorem to prove
theorem find_tan_angle_F2_F1_B (F1 F2 A B : Type) (F1A_dist : F1A = 3) (AB_dist : AB = 4) (BF1_dist : BF1 = 5) :
  angle_F2_F1_B F1 F2 A B = 1 / 7 :=
sorry

end find_tan_angle_F2_F1_B_l1974_197421


namespace range_of_b_l1974_197460

theorem range_of_b (b : ℝ) : (¬ ∃ a < 0, a + 1/a > b) → b ≥ -2 := 
by {
  sorry
}

end range_of_b_l1974_197460


namespace percentage_increase_painting_l1974_197481

/-
Problem:
Given:
1. The original cost of jewelry is $30 each.
2. The original cost of paintings is $100 each.
3. The new cost of jewelry is $40 each.
4. The new cost of paintings is $100 + ($100 * P / 100).
5. A buyer purchased 2 pieces of jewelry and 5 paintings for $680.

Prove:
The percentage increase in the cost of each painting (P) is 20%.
-/

theorem percentage_increase_painting (P : ℝ) :
  let jewelry_price := 30
  let painting_price := 100
  let new_jewelry_price := 40
  let new_painting_price := 100 * (1 + P / 100)
  let total_cost := 2 * new_jewelry_price + 5 * new_painting_price
  total_cost = 680 → P = 20 := by
sorry

end percentage_increase_painting_l1974_197481


namespace water_depth_is_60_l1974_197432

def Ron_height : ℕ := 12
def depth_of_water (h_R : ℕ) : ℕ := 5 * h_R

theorem water_depth_is_60 : depth_of_water Ron_height = 60 :=
by
  sorry

end water_depth_is_60_l1974_197432


namespace instantaneous_velocity_at_1_2_l1974_197473

def equation_of_motion (t : ℝ) : ℝ := 2 * (1 - t^2)

def velocity_function (t : ℝ) : ℝ := -4 * t

theorem instantaneous_velocity_at_1_2 :
  velocity_function 1.2 = -4.8 :=
by sorry

end instantaneous_velocity_at_1_2_l1974_197473


namespace balance_squares_circles_l1974_197425

theorem balance_squares_circles (x y z : ℕ) (h1 : 5 * x + 2 * y = 21 * z) (h2 : 2 * x = y + 3 * z) : 
  3 * y = 9 * z :=
by 
  sorry

end balance_squares_circles_l1974_197425


namespace daily_evaporation_l1974_197419

theorem daily_evaporation (initial_water: ℝ) (days: ℝ) (evap_percentage: ℝ) : 
  initial_water = 10 → days = 50 → evap_percentage = 2 →
  (initial_water * evap_percentage / 100) / days = 0.04 := 
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end daily_evaporation_l1974_197419


namespace half_of_one_point_zero_one_l1974_197404

theorem half_of_one_point_zero_one : (1.01 / 2) = 0.505 := 
by
  sorry

end half_of_one_point_zero_one_l1974_197404


namespace proposition_truthfulness_l1974_197471

-- Definitions
def is_positive (n : ℕ) : Prop := n > 0
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

-- Original proposition
def original_prop (n : ℕ) : Prop := is_positive n ∧ is_even n → ¬ is_prime n

-- Converse proposition
def converse_prop (n : ℕ) : Prop := ¬ is_prime n → is_positive n ∧ is_even n

-- Inverse proposition
def inverse_prop (n : ℕ) : Prop := ¬ (is_positive n ∧ is_even n) → is_prime n

-- Contrapositive proposition
def contrapositive_prop (n : ℕ) : Prop := is_prime n → ¬ (is_positive n ∧ is_even n)

-- Proof problem statement
theorem proposition_truthfulness (n : ℕ) :
  (original_prop n = False) ∧
  (converse_prop n = False) ∧
  (inverse_prop n = False) ∧
  (contrapositive_prop n = True) :=
sorry

end proposition_truthfulness_l1974_197471


namespace time_ratio_school_home_l1974_197418

open Real

noncomputable def time_ratio (y x : ℝ) : ℝ :=
  let time_school := (y / (3 * x)) + (2 * y / (2 * x)) + (y / (4 * x))
  let time_home := (y / (4 * x)) + (2 * y / (2 * x)) + (y / (3 * x))
  time_school / time_home

theorem time_ratio_school_home (y x : ℝ) (hy : y ≠ 0) (hx : x ≠ 0) : time_ratio y x = 19 / 16 :=
  sorry

end time_ratio_school_home_l1974_197418


namespace travel_time_l1974_197454

-- Definitions from problem conditions
def scale := 3000000
def map_distance_cm := 6
def conversion_factor_cm_to_km := 30000 -- derived from 1 cm on the map equals 30,000 km in reality
def speed_kmh := 30

-- The travel time we want to prove
theorem travel_time : (map_distance_cm * conversion_factor_cm_to_km / speed_kmh) = 6000 := 
by
  sorry

end travel_time_l1974_197454


namespace find_n_l1974_197408

theorem find_n (n : ℕ) (h1 : 0 ≤ n ∧ n ≤ 360) (h2 : Real.cos (n * Real.pi / 180) = Real.cos (340 * Real.pi / 180)) : 
  n = 20 ∨ n = 340 := 
by
  sorry

end find_n_l1974_197408


namespace celine_library_charge_l1974_197458

variable (charge_per_day : ℝ) (days_in_may : ℕ) (books_borrowed : ℕ) (days_first_book : ℕ)
          (days_other_books : ℕ) (books_kept : ℕ)

noncomputable def total_charge (charge_per_day : ℝ) (days_first_book : ℕ) 
        (days_other_books : ℕ) (books_kept : ℕ) : ℝ :=
  charge_per_day * days_first_book + charge_per_day * days_other_books * books_kept

theorem celine_library_charge : 
  charge_per_day = 0.50 ∧ days_in_may = 31 ∧ books_borrowed = 3 ∧ days_first_book = 20 ∧
  days_other_books = 31 ∧ books_kept = 2 → 
  total_charge charge_per_day days_first_book days_other_books books_kept = 41.00 :=
by
  intros h
  sorry

end celine_library_charge_l1974_197458


namespace largest_polygon_area_l1974_197455

structure Polygon :=
(unit_squares : Nat)
(right_triangles : Nat)

def area (p : Polygon) : ℝ :=
p.unit_squares + 0.5 * p.right_triangles

def polygon_A : Polygon := { unit_squares := 6, right_triangles := 2 }
def polygon_B : Polygon := { unit_squares := 7, right_triangles := 1 }
def polygon_C : Polygon := { unit_squares := 8, right_triangles := 0 }
def polygon_D : Polygon := { unit_squares := 5, right_triangles := 4 }
def polygon_E : Polygon := { unit_squares := 6, right_triangles := 2 }

theorem largest_polygon_area :
  max (area polygon_A) (max (area polygon_B) (max (area polygon_C) (max (area polygon_D) (area polygon_E)))) = area polygon_C :=
by
  sorry

end largest_polygon_area_l1974_197455


namespace necessary_but_not_sufficient_condition_l1974_197447

variable (x y : ℝ)

theorem necessary_but_not_sufficient_condition :
  (x ≠ 1 ∨ y ≠ 1) ↔ (xy ≠ 1) :=
sorry

end necessary_but_not_sufficient_condition_l1974_197447


namespace number_of_right_handed_players_l1974_197492

/-- 
Given:
(1) There are 70 players on a football team.
(2) 34 players are throwers.
(3) One third of the non-throwers are left-handed.
(4) All throwers are right-handed.
Prove:
The total number of right-handed players is 58.
-/
theorem number_of_right_handed_players 
  (total_players : ℕ) (throwers : ℕ) (non_throwers : ℕ) (left_handed_non_throwers : ℕ) (right_handed_non_throwers : ℕ) : 
  total_players = 70 ∧ throwers = 34 ∧ non_throwers = total_players - throwers ∧ left_handed_non_throwers = non_throwers / 3 ∧ right_handed_non_throwers = non_throwers - left_handed_non_throwers ∧ right_handed_non_throwers + throwers = 58 :=
by
  sorry

end number_of_right_handed_players_l1974_197492


namespace camel_cost_l1974_197467

theorem camel_cost
  (C H O E : ℝ)
  (h1 : 10 * C = 24 * H)
  (h2 : 26 * H = 4 * O)
  (h3 : 6 * O = 4 * E)
  (h4 : 10 * E = 170000) :
  C = 4184.62 :=
by sorry

end camel_cost_l1974_197467


namespace rectangle_area_l1974_197451

theorem rectangle_area (w l : ℕ) (h_sum : w + l = 14) (h_w : w = 6) : w * l = 48 := by
  sorry

end rectangle_area_l1974_197451


namespace turtle_population_estimate_l1974_197476

theorem turtle_population_estimate :
  (tagged_in_june = 90) →
  (sample_november = 50) →
  (tagged_november = 4) →
  (natural_causes_removal = 0.30) →
  (new_hatchlings_november = 0.50) →
  estimate = 563 :=
by
  intros tagged_in_june sample_november tagged_november natural_causes_removal new_hatchlings_november
  sorry

end turtle_population_estimate_l1974_197476


namespace inequality_false_implies_range_of_a_l1974_197437

theorem inequality_false_implies_range_of_a (a : ℝ) : 
  (∀ t : ℝ, t^2 - 2 * t - a ≥ 0) ↔ a ≤ -1 :=
by
  sorry

end inequality_false_implies_range_of_a_l1974_197437


namespace infinite_series_sum_l1974_197464

theorem infinite_series_sum :
  (∑' n : ℕ, if h : n ≠ 0 then 1 / (n * (n + 1) * (n + 3)) else 0) = 5 / 36 := by
  sorry

end infinite_series_sum_l1974_197464


namespace average_speed_round_trip_l1974_197441

theorem average_speed_round_trip
  (n : ℕ)
  (distance_km : ℝ := n / 1000)
  (pace_west_min_per_km : ℝ := 2)
  (speed_east_kmh : ℝ := 3)
  (wait_time_hr : ℝ := 30 / 60) :
  (2 * distance_km) / 
  ((pace_west_min_per_km * distance_km / 60) + wait_time_hr + (distance_km / speed_east_kmh)) = 
  60 * n / (11 * n + 150000) := by
  sorry

end average_speed_round_trip_l1974_197441


namespace sufficient_not_necessary_condition_x_plus_a_div_x_geq_2_l1974_197446

open Real

theorem sufficient_not_necessary_condition_x_plus_a_div_x_geq_2 (x a : ℝ)
  (h₁ : x > 0) :
  (∀ x > 0, x + a / x ≥ 2) → (a = 1) :=
sorry

end sufficient_not_necessary_condition_x_plus_a_div_x_geq_2_l1974_197446


namespace range_of_m_hyperbola_l1974_197438

noncomputable def is_conic_hyperbola (expr : ℝ → ℝ → ℝ) : Prop :=
  ∃ f : ℝ, ∀ x y, expr x y = ((x - 2 * y + 3)^2 - f * (x^2 + y^2 + 2 * y + 1))

theorem range_of_m_hyperbola (m : ℝ) :
  is_conic_hyperbola (fun x y => m * (x^2 + y^2 + 2 * y + 1) - (x - 2 * y + 3)^2) → 5 < m :=
sorry

end range_of_m_hyperbola_l1974_197438


namespace consecutive_product_plus_one_l1974_197493

theorem consecutive_product_plus_one (n : ℕ) :
  n * (n + 1) * (n + 2) * (n + 3) + 1 = (n^2 + 3 * n + 1)^2 :=
by
  sorry

end consecutive_product_plus_one_l1974_197493


namespace fuel_reduction_16km_temperature_drop_16km_l1974_197449

-- Definition for fuel reduction condition
def fuel_reduction_rate (distance: ℕ) : ℕ := distance / 4 * 2

-- Definition for temperature drop condition
def temperature_drop_rate (distance: ℕ) : ℕ := distance / 8 * 1

-- Theorem to prove fuel reduction for 16 km
theorem fuel_reduction_16km : fuel_reduction_rate 16 = 8 := 
by
  -- proof will go here, but for now add sorry
  sorry

-- Theorem to prove temperature drop for 16 km
theorem temperature_drop_16km : temperature_drop_rate 16 = 2 := 
by
  -- proof will go here, but for now add sorry
  sorry

end fuel_reduction_16km_temperature_drop_16km_l1974_197449


namespace min_moves_is_22_l1974_197499

def casket_coins : List ℕ := [9, 17, 12, 5, 18, 10, 20]

def target_coins (total_caskets : ℕ) (total_coins : ℕ) : ℕ :=
  total_coins / total_caskets

def total_caskets : ℕ := 7

def total_coins (coins : List ℕ) : ℕ :=
  coins.foldr (· + ·) 0

noncomputable def min_moves_to_equalize (coins : List ℕ) (target : ℕ) : ℕ := sorry

theorem min_moves_is_22 :
  min_moves_to_equalize casket_coins (target_coins total_caskets (total_coins casket_coins)) = 22 :=
sorry

end min_moves_is_22_l1974_197499


namespace Maria_score_in_fourth_quarter_l1974_197450

theorem Maria_score_in_fourth_quarter (q1 q2 q3 : ℕ) 
  (hq1 : q1 = 84) 
  (hq2 : q2 = 82) 
  (hq3 : q3 = 80) 
  (average_requirement : ℕ) 
  (havg_req : average_requirement = 85) :
  ∃ q4 : ℕ, q4 ≥ 94 ∧ (q1 + q2 + q3 + q4) / 4 ≥ average_requirement := 
by 
  sorry 

end Maria_score_in_fourth_quarter_l1974_197450


namespace hotel_charge_decrease_l1974_197461

theorem hotel_charge_decrease 
  (G R P : ℝ)
  (h1 : R = 1.60 * G)
  (h2 : P = 0.50 * R) :
  (G - P) / G * 100 = 20 := by
sorry

end hotel_charge_decrease_l1974_197461


namespace find_positions_l1974_197494

def first_column (m : ℕ) : ℕ := 4 + 3*(m-1)

def table_element (m n : ℕ) : ℕ := first_column m + (n-1)*(2*m + 1)

theorem find_positions :
  (∀ m n, table_element m n ≠ 1994) ∧
  (∃ m n, table_element m n = 1995 ∧ ((m = 6 ∧ n = 153) ∨ (m = 153 ∧ n = 6))) :=
by
  sorry

end find_positions_l1974_197494


namespace randy_trip_distance_l1974_197416

theorem randy_trip_distance (x : ℝ) (h1 : x = x / 4 + 30 + x / 10 + (x - (x / 4 + 30 + x / 10))) :
  x = 60 :=
by {
  sorry -- Placeholder for the actual proof
}

end randy_trip_distance_l1974_197416


namespace largest_x_satisfies_condition_l1974_197405

theorem largest_x_satisfies_condition :
  ∃ x : ℝ, (⌊x⌋ / x = 7 / 8) ∧ (∀ y : ℝ, (⌊y⌋ / y = 7 / 8) → y ≤ 48 / 7) :=
sorry

end largest_x_satisfies_condition_l1974_197405


namespace cos_identity_l1974_197490

theorem cos_identity
  (α : ℝ)
  (h : Real.sin (π / 6 - α) = 1 / 3) :
  Real.cos (2 * π / 3 + 2 * α) = -7 / 9 :=
by
  sorry

end cos_identity_l1974_197490


namespace find_third_smallest_three_digit_palindromic_prime_l1974_197495

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def second_smallest_three_digit_palindromic_prime : ℕ :=
  131 -- Given in the problem statement

noncomputable def third_smallest_three_digit_palindromic_prime : ℕ :=
  151 -- Answer obtained from the solution

theorem find_third_smallest_three_digit_palindromic_prime :
  ∃ n, is_palindrome n ∧ is_prime n ∧ 100 ≤ n ∧ n < 1000 ∧
  (n ≠ 101) ∧ (n ≠ 131) ∧ (∀ m, is_palindrome m ∧ is_prime m ∧ 100 ≤ m ∧ m < 1000 → second_smallest_three_digit_palindromic_prime < m → m = n) :=
by
  sorry -- This is where the proof would be, but it is not needed as per instructions.

end find_third_smallest_three_digit_palindromic_prime_l1974_197495


namespace isosceles_triangle_base_length_l1974_197485

theorem isosceles_triangle_base_length (a b c : ℕ) (h_isosceles : a = b ∨ b = c ∨ c = a)
  (h_perimeter : a + b + c = 16) (h_side_length : a = 6 ∨ b = 6 ∨ c = 6) :
  (a = 4 ∨ b = 4 ∨ c = 4) ∨ (a = 6 ∨ b = 6 ∨ c = 6) :=
sorry

end isosceles_triangle_base_length_l1974_197485


namespace sqrt_range_l1974_197442

theorem sqrt_range (x : ℝ) : (1 - x ≥ 0) ↔ (x ≤ 1) := sorry

end sqrt_range_l1974_197442


namespace lily_milk_left_l1974_197445

theorem lily_milk_left : 
  let initial_milk := 5 
  let given_to_james := 18 / 7
  ∃ r : ℚ, r = 2 + 3/7 ∧ (initial_milk - given_to_james) = r :=
by
  sorry

end lily_milk_left_l1974_197445


namespace max_value_of_quadratic_l1974_197435

theorem max_value_of_quadratic (x : ℝ) (h1 : 0 < x) (h2 : x < 1/2) : 
  ∃ y, y = x * (1 - 2 * x) ∧ y ≤ 1 / 8 ∧ (y = 1 / 8 ↔ x = 1 / 4) :=
by sorry

end max_value_of_quadratic_l1974_197435


namespace barry_sotter_magic_l1974_197475

theorem barry_sotter_magic (n : ℕ) : (n + 3) / 3 = 50 → n = 147 := 
by 
  sorry

end barry_sotter_magic_l1974_197475


namespace tony_bought_10_play_doughs_l1974_197427

noncomputable def num_play_doughs 
    (lego_cost : ℕ) 
    (sword_cost : ℕ) 
    (play_dough_cost : ℕ) 
    (bought_legos : ℕ) 
    (bought_swords : ℕ) 
    (total_paid : ℕ) : ℕ :=
  let lego_total := lego_cost * bought_legos
  let sword_total := sword_cost * bought_swords
  let total_play_dough_cost := total_paid - (lego_total + sword_total)
  total_play_dough_cost / play_dough_cost

theorem tony_bought_10_play_doughs : 
  num_play_doughs 250 120 35 3 7 1940 = 10 := 
sorry

end tony_bought_10_play_doughs_l1974_197427
