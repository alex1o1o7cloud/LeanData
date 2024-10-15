import Mathlib

namespace NUMINAMATH_GPT_circle_center_and_radius_l1980_198042

theorem circle_center_and_radius (x y : ℝ) (h : x^2 + y^2 - 6*x = 0) :
  (∃ c : ℝ × ℝ, c = (3, 0)) ∧ (∃ r : ℝ, r = 3) := 
sorry

end NUMINAMATH_GPT_circle_center_and_radius_l1980_198042


namespace NUMINAMATH_GPT_time_juan_ran_l1980_198097

variable (Distance Speed : ℝ)
variable (h1 : Distance = 80)
variable (h2 : Speed = 10)

theorem time_juan_ran : (Distance / Speed) = 8 := by
  sorry

end NUMINAMATH_GPT_time_juan_ran_l1980_198097


namespace NUMINAMATH_GPT_salary_reduction_l1980_198060

theorem salary_reduction (S : ℝ) (R : ℝ) 
  (h : (S - (R / 100) * S) * (4 / 3) = S) :
  R = 25 := 
  sorry

end NUMINAMATH_GPT_salary_reduction_l1980_198060


namespace NUMINAMATH_GPT_defective_units_shipped_percentage_l1980_198033

theorem defective_units_shipped_percentage :
  let units_produced := 100
  let typeA_defective := 0.07 * units_produced
  let typeB_defective := 0.08 * units_produced
  let typeA_shipped := 0.03 * typeA_defective
  let typeB_shipped := 0.06 * typeB_defective
  let total_shipped := typeA_shipped + typeB_shipped
  let percentage_shipped := total_shipped / units_produced * 100
  percentage_shipped = 1 :=
by
  sorry

end NUMINAMATH_GPT_defective_units_shipped_percentage_l1980_198033


namespace NUMINAMATH_GPT_find_a_l1980_198077

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 4

noncomputable def line_eq (x y a : ℝ) : Prop := x + a * y + 1 = 0

theorem find_a (a : ℝ) :
  (∀ x y : ℝ, circle_eq x y → line_eq x y a → (x - 1)^2 + (y - 2)^2 = 4) →
  ∃ a, (a = -1) :=
sorry

end NUMINAMATH_GPT_find_a_l1980_198077


namespace NUMINAMATH_GPT_find_AC_l1980_198082

theorem find_AC (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (max_val : A - C = 3) (min_val : -A - C = -1) : 
  A = 2 ∧ C = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_AC_l1980_198082


namespace NUMINAMATH_GPT_homework_done_l1980_198056

theorem homework_done :
  ∃ (D E C Z M : Prop),
    -- Statements of students
    (¬ D ∧ E ∧ ¬ C ∧ ¬ Z ∧ ¬ M) ∧
    -- Truth-telling condition
    ((D → D ∧ ¬ E ∧ ¬ C ∧ ¬ Z ∧ ¬ M) ∧
    (E → ¬ D ∧ E ∧ ¬ C ∧ ¬ Z ∧ ¬ M) ∧
    (C → ¬ D ∧ ¬ E ∧ C ∧ ¬ Z ∧ ¬ M) ∧
    (Z → ¬ D ∧ ¬ E ∧ ¬ C ∧ Z ∧ ¬ M) ∧
    (M → ¬ D ∧ ¬ E ∧ ¬ C ∧ ¬ Z ∧ M)) ∧
    -- Number of students who did their homework condition
    (¬ D ∧ E ∧ ¬ C ∧ ¬ Z ∧ ¬ M) := 
sorry

end NUMINAMATH_GPT_homework_done_l1980_198056


namespace NUMINAMATH_GPT_find_k_for_parallel_lines_l1980_198000

theorem find_k_for_parallel_lines (k : ℝ) :
  (∀ x y : ℝ, (k - 2) * x + (4 - k) * y + 1 = 0) →
  (∀ x y : ℝ, 2 * (k - 2) * x - 2 * y + 3 = 0) →
  (k = 2 ∨ k = 5) :=
sorry

end NUMINAMATH_GPT_find_k_for_parallel_lines_l1980_198000


namespace NUMINAMATH_GPT_least_number_of_attendees_l1980_198005

-- Definitions based on problem conditions
inductive Person
| Anna
| Bill
| Carl
deriving DecidableEq

inductive Day
| Mon
| Tues
| Wed
| Thurs
| Fri
deriving DecidableEq

def attends : Person → Day → Prop
| Person.Anna, Day.Mon => true
| Person.Anna, Day.Tues => false
| Person.Anna, Day.Wed => true
| Person.Anna, Day.Thurs => false
| Person.Anna, Day.Fri => false
| Person.Bill, Day.Mon => false
| Person.Bill, Day.Tues => true
| Person.Bill, Day.Wed => false
| Person.Bill, Day.Thurs => true
| Person.Bill, Day.Fri => true
| Person.Carl, Day.Mon => true
| Person.Carl, Day.Tues => true
| Person.Carl, Day.Wed => false
| Person.Carl, Day.Thurs => true
| Person.Carl, Day.Fri => false

-- Proof statement
theorem least_number_of_attendees : 
  (∀ d : Day, (∀ p : Person, attends p d → p = Person.Anna ∨ p = Person.Bill ∨ p = Person.Carl) ∧
              (d = Day.Wed ∨ d = Day.Fri → (∃ n : ℕ, n = 2 ∧ (∀ p : Person, attends p d → n = 2))) ∧
              (d = Day.Mon ∨ d = Day.Tues ∨ d = Day.Thurs → (∃ n : ℕ, n = 1 ∧ (∀ p : Person, attends p d → n = 1))) ∧
              ¬ (d = Day.Wed ∨ d = Day.Fri)) :=
sorry

end NUMINAMATH_GPT_least_number_of_attendees_l1980_198005


namespace NUMINAMATH_GPT_find_total_pupils_l1980_198009

-- Define the conditions for the problem
def diff1 : ℕ := 85 - 45
def diff2 : ℕ := 79 - 49
def diff3 : ℕ := 64 - 34
def total_diff : ℕ := diff1 + diff2 + diff3
def avg_increase : ℕ := 3

-- Assert that the number of pupils n satisfies the given conditions
theorem find_total_pupils (n : ℕ) (h_diff : total_diff = 100) (h_avg_inc : avg_increase * n = total_diff) : n = 33 :=
by
  sorry

end NUMINAMATH_GPT_find_total_pupils_l1980_198009


namespace NUMINAMATH_GPT_arithmetic_sequence_x_y_sum_l1980_198070

theorem arithmetic_sequence_x_y_sum :
  ∀ (a d x y: ℕ), 
  a = 3 → d = 6 → 
  (∀ (n: ℕ), n ≥ 1 → a + (n-1) * d = 3 + (n-1) * 6) →
  (a + 5 * d = x) → (a + 6 * d = y) → 
  (y = 45 - d) → x + y = 72 :=
by
  intros a d x y h_a h_d h_seq h_x h_y h_y_equals
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_x_y_sum_l1980_198070


namespace NUMINAMATH_GPT_pool_depth_l1980_198053

theorem pool_depth 
  (length width : ℝ) 
  (chlorine_per_120_cubic_feet chlorine_cost : ℝ) 
  (total_spent volume_per_quart_of_chlorine : ℝ) 
  (H_length : length = 10) 
  (H_width : width = 8)
  (H_chlorine_per_120_cubic_feet : chlorine_per_120_cubic_feet = 1 / 120)
  (H_chlorine_cost : chlorine_cost = 3)
  (H_total_spent : total_spent = 12)
  (H_volume_per_quart_of_chlorine : volume_per_quart_of_chlorine = 120) :
  ∃ depth : ℝ, total_spent / chlorine_cost * volume_per_quart_of_chlorine = length * width * depth ∧ depth = 6 :=
by 
  sorry

end NUMINAMATH_GPT_pool_depth_l1980_198053


namespace NUMINAMATH_GPT_combined_total_score_l1980_198007

-- Define the conditions
def num_single_answer_questions : ℕ := 50
def num_multiple_answer_questions : ℕ := 20
def single_answer_score : ℕ := 2
def multiple_answer_score : ℕ := 4
def wrong_single_penalty : ℕ := 1
def wrong_multiple_penalty : ℕ := 2
def jose_wrong_single : ℕ := 10
def jose_wrong_multiple : ℕ := 5
def jose_lost_marks : ℕ := (jose_wrong_single * wrong_single_penalty) + (jose_wrong_multiple * wrong_multiple_penalty)
def jose_correct_single : ℕ := num_single_answer_questions - jose_wrong_single
def jose_correct_multiple : ℕ := num_multiple_answer_questions - jose_wrong_multiple
def jose_single_score : ℕ := jose_correct_single * single_answer_score
def jose_multiple_score : ℕ := jose_correct_multiple * multiple_answer_score
def jose_score : ℕ := (jose_single_score + jose_multiple_score) - jose_lost_marks
def alison_score : ℕ := jose_score - 50
def meghan_score : ℕ := jose_score - 30

-- Prove the combined total score
theorem combined_total_score :
  jose_score + alison_score + meghan_score = 280 :=
by
  sorry

end NUMINAMATH_GPT_combined_total_score_l1980_198007


namespace NUMINAMATH_GPT_solve_system_l1980_198030

/-- Given the system of equations:
    3 * (x + y) - 4 * (x - y) = 5
    (x + y) / 2 + (x - y) / 6 = 0
  Prove that the solution is x = -1/3 and y = 2/3 
-/
theorem solve_system (x y : ℚ) 
  (h1 : 3 * (x + y) - 4 * (x - y) = 5)
  (h2 : (x + y) / 2 + (x - y) / 6 = 0) : 
  x = -1 / 3 ∧ y = 2 / 3 := 
sorry

end NUMINAMATH_GPT_solve_system_l1980_198030


namespace NUMINAMATH_GPT_emma_age_proof_l1980_198065

def is_age_of_emma (age : Nat) : Prop := 
  let guesses := [26, 29, 31, 33, 35, 39, 42, 44, 47, 50]
  let at_least_60_percent_low := (guesses.filter (· < age)).length * 10 ≥ 6 * guesses.length
  let exactly_two_off_by_one := (guesses.filter (λ x => x = age - 1 ∨ x = age + 1)).length = 2
  let is_prime := Nat.Prime age
  at_least_60_percent_low ∧ exactly_two_off_by_one ∧ is_prime

theorem emma_age_proof : is_age_of_emma 43 := 
  by sorry

end NUMINAMATH_GPT_emma_age_proof_l1980_198065


namespace NUMINAMATH_GPT_probability_not_in_square_b_l1980_198050

theorem probability_not_in_square_b (area_A : ℝ) (perimeter_B : ℝ) 
  (area_A_eq : area_A = 30) (perimeter_B_eq : perimeter_B = 16) : 
  (14 / 30 : ℝ) = (7 / 15 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_probability_not_in_square_b_l1980_198050


namespace NUMINAMATH_GPT_find_x1_l1980_198045

theorem find_x1 (x1 x2 x3 x4 : ℝ) 
  (h1 : 0 ≤ x4) (h2 : x4 ≤ x3) (h3 : x3 ≤ x2) (h4 : x2 ≤ x1) (h5 : x1 ≤ 1)
  (h6 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + (x3 - x4)^2 + x4^2 = 1 / 3) : 
  x1 = 4 / 5 :=
  sorry

end NUMINAMATH_GPT_find_x1_l1980_198045


namespace NUMINAMATH_GPT_fixed_point_range_l1980_198006

theorem fixed_point_range (a : ℝ) : (∃ x : ℝ, x = x^2 + x + a) → a ≤ 0 :=
sorry

end NUMINAMATH_GPT_fixed_point_range_l1980_198006


namespace NUMINAMATH_GPT_pizza_sales_calculation_l1980_198049

def pizzas_sold_in_spring (total_sales : ℝ) (summer_sales : ℝ) (fall_percentage : ℝ) (winter_percentage : ℝ) : ℝ :=
  total_sales - summer_sales - (fall_percentage * total_sales) - (winter_percentage * total_sales)

theorem pizza_sales_calculation :
  let summer_sales := 5;
  let fall_percentage := 0.1;
  let winter_percentage := 0.2;
  ∃ (total_sales : ℝ), 0.4 * total_sales = summer_sales ∧
    pizzas_sold_in_spring total_sales summer_sales fall_percentage winter_percentage = 3.75 :=
by
  sorry

end NUMINAMATH_GPT_pizza_sales_calculation_l1980_198049


namespace NUMINAMATH_GPT_problem_inequality_solution_set_inequality_proof_l1980_198025

def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

theorem problem_inequality_solution_set :
  {x : ℝ | f x < 4} = {x : ℝ | -2 < x ∧ x < 2} :=
sorry

theorem inequality_proof (x y : ℝ) (hx : -2 < x ∧ x < 2) (hy : -2 < y ∧ y < 2) :
  |x + y| < |(x * y) / 2 + 2| :=
sorry

end NUMINAMATH_GPT_problem_inequality_solution_set_inequality_proof_l1980_198025


namespace NUMINAMATH_GPT_find_m_find_min_value_l1980_198020

-- Conditions
def A (m : ℤ) : Set ℝ := { x | abs (x + 1) + abs (x - m) < 5 }

-- First Problem: Prove m = 3 given 3 ∈ A
theorem find_m (m : ℤ) (h : 3 ∈ A m) : m = 3 := sorry

-- Second Problem: Prove a^2 + b^2 + c^2 ≥ 1 given a + 2b + 2c = 3
theorem find_min_value (a b c : ℝ) (h : a + 2 * b + 2 * c = 3) : (a^2 + b^2 + c^2) ≥ 1 := sorry

end NUMINAMATH_GPT_find_m_find_min_value_l1980_198020


namespace NUMINAMATH_GPT_meaningful_iff_x_ne_2_l1980_198008

theorem meaningful_iff_x_ne_2 (x : ℝ) : (x ≠ 2) ↔ (∃ y : ℝ, y = (x - 3) / (x - 2)) := 
by
  sorry

end NUMINAMATH_GPT_meaningful_iff_x_ne_2_l1980_198008


namespace NUMINAMATH_GPT_triangle_tangent_ratio_l1980_198074

variable {A B C a b c : ℝ}

theorem triangle_tangent_ratio 
  (h : a * Real.cos B - b * Real.cos A = (3 / 5) * c)
  : Real.tan A / Real.tan B = 4 :=
sorry

end NUMINAMATH_GPT_triangle_tangent_ratio_l1980_198074


namespace NUMINAMATH_GPT_remainder_sum_l1980_198034

theorem remainder_sum (n : ℤ) : ((7 - n) + (n + 3)) % 7 = 3 :=
sorry

end NUMINAMATH_GPT_remainder_sum_l1980_198034


namespace NUMINAMATH_GPT_conjecture_a_n_l1980_198094

noncomputable def a_n (n : ℕ) : ℚ := (2^n - 1) / 2^(n-1)

noncomputable def S_n (n : ℕ) : ℚ := 2 * n - a_n n

theorem conjecture_a_n (n : ℕ) (h : n > 0) : a_n n = (2^n - 1) / 2^(n-1) :=
by 
  sorry

end NUMINAMATH_GPT_conjecture_a_n_l1980_198094


namespace NUMINAMATH_GPT_modulo_11_residue_l1980_198043

theorem modulo_11_residue : 
  (341 + 6 * 50 + 4 * 156 + 3 * 12^2) % 11 = 4 := 
by
  sorry

end NUMINAMATH_GPT_modulo_11_residue_l1980_198043


namespace NUMINAMATH_GPT_number_of_ways_to_represent_1500_l1980_198037

theorem number_of_ways_to_represent_1500 :
  ∃ (count : ℕ), count = 30 ∧ ∀ (a b c : ℕ), a * b * c = 1500 :=
sorry

end NUMINAMATH_GPT_number_of_ways_to_represent_1500_l1980_198037


namespace NUMINAMATH_GPT_factor_expression_l1980_198002

theorem factor_expression (a b c : ℝ) :
  let num := (a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3
  let denom := (a - b)^3 + (b - c)^3 + (c - a)^3
  (denom ≠ 0) →
  num / denom = (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l1980_198002


namespace NUMINAMATH_GPT_smallest_multiple_1_10_is_2520_l1980_198062

noncomputable def smallest_multiple_1_10 : ℕ :=
  Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10))))))))

theorem smallest_multiple_1_10_is_2520 : smallest_multiple_1_10 = 2520 :=
  sorry

end NUMINAMATH_GPT_smallest_multiple_1_10_is_2520_l1980_198062


namespace NUMINAMATH_GPT_successful_pair_exists_another_with_same_arithmetic_mean_l1980_198023

theorem successful_pair_exists_another_with_same_arithmetic_mean
  (a b : ℕ)
  (h_distinct : a ≠ b)
  (h_arith_mean_nat : ∃ m : ℕ, 2 * m = a + b)
  (h_geom_mean_nat : ∃ g : ℕ, g * g = a * b) :
  ∃ (c d : ℕ), c ≠ d ∧ ∃ m' : ℕ, 2 * m' = c + d ∧ ∃ g' : ℕ, g' * g' = c * d ∧ m' = (a + b) / 2 :=
sorry

end NUMINAMATH_GPT_successful_pair_exists_another_with_same_arithmetic_mean_l1980_198023


namespace NUMINAMATH_GPT_max_quartets_in_5x5_max_quartets_in_mxn_l1980_198099

def quartet (c : Nat) : Bool := 
  c > 0

theorem max_quartets_in_5x5 : ∃ q, q = 5 ∧ 
  quartet q := by
  sorry

theorem max_quartets_in_mxn 
  (m n : Nat) (Hmn : m > 0 ∧ n > 0) :
  (∃ q, q = (m * (n - 1)) / 4 ∧ quartet q) ∨ 
  (∃ q, q = (m * (n - 1) - 2) / 4 ∧ quartet q) := by
  sorry

end NUMINAMATH_GPT_max_quartets_in_5x5_max_quartets_in_mxn_l1980_198099


namespace NUMINAMATH_GPT_find_pair_l1980_198011

theorem find_pair (a b : ℤ) :
  (∀ x : ℝ, (a * x^4 + b * x^3 + 20 * x^2 - 12 * x + 10) = (2 * x^2 + 3 * x - 4) * (c * x^2 + d * x + e)) → 
  (a = 2) ∧ (b = 27) :=
sorry

end NUMINAMATH_GPT_find_pair_l1980_198011


namespace NUMINAMATH_GPT_find_k_for_tangent_graph_l1980_198067

theorem find_k_for_tangent_graph (k : ℝ) (h : (∀ x : ℝ, x^2 - 6 * x + k = 0 → (x = 3))) : k = 9 :=
sorry

end NUMINAMATH_GPT_find_k_for_tangent_graph_l1980_198067


namespace NUMINAMATH_GPT_non_working_games_count_l1980_198066

def total_games : ℕ := 16
def price_each : ℕ := 7
def total_earnings : ℕ := 56

def working_games : ℕ := total_earnings / price_each
def non_working_games : ℕ := total_games - working_games

theorem non_working_games_count : non_working_games = 8 := by
  sorry

end NUMINAMATH_GPT_non_working_games_count_l1980_198066


namespace NUMINAMATH_GPT_garden_strawberry_yield_l1980_198057

-- Definitions from the conditions
def garden_length : ℝ := 10
def garden_width : ℝ := 15
def plants_per_sq_ft : ℝ := 5
def strawberries_per_plant : ℝ := 12

-- Expected total number of strawberries
def expected_strawberries : ℝ := 9000

-- Proof statement
theorem garden_strawberry_yield : 
  (garden_length * garden_width * plants_per_sq_ft * strawberries_per_plant) = expected_strawberries :=
by sorry

end NUMINAMATH_GPT_garden_strawberry_yield_l1980_198057


namespace NUMINAMATH_GPT_maximum_sum_of_numbers_in_grid_l1980_198076

theorem maximum_sum_of_numbers_in_grid :
  ∀ (grid : List (List ℕ)) (rect_cover : (ℕ × ℕ) → (ℕ × ℕ) → Prop),
  (∀ x y, rect_cover x y → x ≠ y → x.1 < 6 → x.2 < 6 → y.1 < 6 → y.2 < 6) →
  (∀ x y z w, rect_cover x y ∧ rect_cover z w → 
    (x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∨ (x.1 = z.1 ∨ x.2 = z.2) → 
    (x.1 = z.1 ∧ x.2 = y.2 ∨ x.2 = z.2 ∧ x.1 = y.1)) → False) →
  (36 = 6 * 6) →
  18 = 36 / 2 →
  342 = (18 * 19) :=
by
  intro grid rect_cover h_grid h_no_common_edge h_grid_size h_num_rectangles
  sorry

end NUMINAMATH_GPT_maximum_sum_of_numbers_in_grid_l1980_198076


namespace NUMINAMATH_GPT_largest_common_term_in_range_1_to_200_l1980_198084

theorem largest_common_term_in_range_1_to_200 :
  ∃ (a : ℕ), a < 200 ∧ (∃ (n₁ n₂ : ℕ), a = 3 + 8 * n₁ ∧ a = 5 + 9 * n₂) ∧ a = 179 :=
by
  sorry

end NUMINAMATH_GPT_largest_common_term_in_range_1_to_200_l1980_198084


namespace NUMINAMATH_GPT_weight_of_7th_person_l1980_198098

/--
There are 6 people in the elevator with an average weight of 152 lbs.
Another person enters the elevator, increasing the average weight to 151 lbs.
Prove that the weight of the 7th person is 145 lbs.
-/
theorem weight_of_7th_person
  (W : ℕ) (X : ℕ) (h1 : W / 6 = 152) (h2 : (W + X) / 7 = 151) :
  X = 145 :=
sorry

end NUMINAMATH_GPT_weight_of_7th_person_l1980_198098


namespace NUMINAMATH_GPT_john_trip_l1980_198064

theorem john_trip (t : ℝ) (h : t ≥ 0) : 
  ∀ t : ℝ, 60 * t + 90 * ((7 / 2) - t) = 300 :=
by sorry

end NUMINAMATH_GPT_john_trip_l1980_198064


namespace NUMINAMATH_GPT_point_position_after_time_l1980_198015

noncomputable def final_position (initial : ℝ × ℝ) (velocity : ℝ × ℝ) (time : ℝ) : ℝ × ℝ :=
  (initial.1 + velocity.1 * time, initial.2 + velocity.2 * time)

theorem point_position_after_time :
  final_position (-10, 10) (4, -3) 5 = (10, -5) :=
by
  sorry

end NUMINAMATH_GPT_point_position_after_time_l1980_198015


namespace NUMINAMATH_GPT_train_times_comparison_l1980_198001

-- Defining the given conditions
variables (V1 T1 T2 D : ℝ)
variables (h1 : T1 = 2) (h2 : T2 = 7/3)
variables (train1_speed : V1 = D / T1)
variables (train2_speed : V2 = (3/5) * V1)

-- The proof statement to show that T2 is 1/3 hour longer than T1
theorem train_times_comparison 
  (h1 : (6/7) * V1 = D / (T1 + 1/3))
  (h2 : (3/5) * V1 = D / (T2 + 1)) :
  T2 - T1 = 1/3 :=
sorry

end NUMINAMATH_GPT_train_times_comparison_l1980_198001


namespace NUMINAMATH_GPT_fourteen_divisible_by_7_twenty_eight_divisible_by_7_thirty_five_divisible_by_7_forty_nine_divisible_by_7_l1980_198026

def is_divisible_by_7 (n: ℕ): Prop := n % 7 = 0

theorem fourteen_divisible_by_7: is_divisible_by_7 14 :=
by
  sorry

theorem twenty_eight_divisible_by_7: is_divisible_by_7 28 :=
by
  sorry

theorem thirty_five_divisible_by_7: is_divisible_by_7 35 :=
by
  sorry

theorem forty_nine_divisible_by_7: is_divisible_by_7 49 :=
by
  sorry

end NUMINAMATH_GPT_fourteen_divisible_by_7_twenty_eight_divisible_by_7_thirty_five_divisible_by_7_forty_nine_divisible_by_7_l1980_198026


namespace NUMINAMATH_GPT_find_x_l1980_198017

theorem find_x (x : ℝ) : (1 + (1 / (1 + x)) = 2 * (1 / (1 + x))) → x = 0 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_x_l1980_198017


namespace NUMINAMATH_GPT_find_angle_C_l1980_198013

noncomputable def angle_C_value (A B : ℝ) : ℝ :=
  180 - A - B

theorem find_angle_C (A B : ℝ) 
  (h1 : 3 * Real.sin A + 4 * Real.cos B = 6)
  (h2 : 4 * Real.sin B + 3 * Real.cos A = 1) :
  angle_C_value A B = 30 :=
sorry

end NUMINAMATH_GPT_find_angle_C_l1980_198013


namespace NUMINAMATH_GPT_terminating_decimal_expansion_l1980_198059

theorem terminating_decimal_expansion : (11 / 125 : ℝ) = 0.088 := 
by
  sorry

end NUMINAMATH_GPT_terminating_decimal_expansion_l1980_198059


namespace NUMINAMATH_GPT_burger_cost_l1980_198078

theorem burger_cost 
  (b s : ℕ)
  (h1 : 4 * b + 3 * s = 440)
  (h2 : 3 * b + 2 * s = 330) : b = 110 := 
by 
  sorry

end NUMINAMATH_GPT_burger_cost_l1980_198078


namespace NUMINAMATH_GPT_batsman_average_after_11th_inning_l1980_198095

theorem batsman_average_after_11th_inning (A : ℝ) 
  (h1 : A + 5 = (10 * A + 85) / 11) : A + 5 = 35 :=
by
  sorry

end NUMINAMATH_GPT_batsman_average_after_11th_inning_l1980_198095


namespace NUMINAMATH_GPT_tiles_needed_l1980_198041

def room_area : ℝ := 2 * 4 * 2 * 6
def tile_area : ℝ := 1.5 * 2

theorem tiles_needed : room_area / tile_area = 32 := 
by
  sorry

end NUMINAMATH_GPT_tiles_needed_l1980_198041


namespace NUMINAMATH_GPT_gcd_lcm_sum_l1980_198032

variable (a b : ℕ)

-- Definition for gcd
def gcdOf (a b : ℕ) : ℕ := Nat.gcd a b

-- Definition for lcm
def lcmOf (a b : ℕ) : ℕ := Nat.lcm a b

-- Statement of the problem
theorem gcd_lcm_sum (h1 : a = 8) (h2 : b = 12) : gcdOf a b + lcmOf a b = 28 := by
  sorry

end NUMINAMATH_GPT_gcd_lcm_sum_l1980_198032


namespace NUMINAMATH_GPT_pair_D_equal_l1980_198047

theorem pair_D_equal: (-1)^3 = (-1)^2023 := by
  sorry

end NUMINAMATH_GPT_pair_D_equal_l1980_198047


namespace NUMINAMATH_GPT_peter_completes_remaining_work_in_14_days_l1980_198087

-- Define the conditions and the theorem
variable (W : ℕ) (work_done : ℕ) (remaining_work : ℕ)

theorem peter_completes_remaining_work_in_14_days
  (h1 : Matt_and_Peter_rate = (W/20))
  (h2 : Peter_rate = (W/35))
  (h3 : Work_done_in_12_days = (12 * (W/20)))
  (h4 : Remaining_work = (W - (12 * (W/20))))
  : (remaining_work / Peter_rate)  = 14 := sorry

end NUMINAMATH_GPT_peter_completes_remaining_work_in_14_days_l1980_198087


namespace NUMINAMATH_GPT_find_n_of_sum_of_evens_l1980_198088

-- Definitions based on conditions in part (a)
def is_odd (n : ℕ) : Prop := n % 2 = 1

def sum_of_evens_up_to (n : ℕ) : ℕ :=
  let k := (n - 1) / 2
  (k / 2) * (2 + (n - 1))

-- Problem statement in Lean
theorem find_n_of_sum_of_evens : 
  ∃ n : ℕ, is_odd n ∧ sum_of_evens_up_to n = 81 * 82 ∧ n = 163 :=
by
  sorry

end NUMINAMATH_GPT_find_n_of_sum_of_evens_l1980_198088


namespace NUMINAMATH_GPT_band_row_lengths_l1980_198003

theorem band_row_lengths (n : ℕ) (h1 : n = 108) (h2 : ∃ k, 10 ≤ k ∧ k ≤ 18 ∧ 108 % k = 0) : 
  (∃ count : ℕ, count = 2) :=
by 
  sorry

end NUMINAMATH_GPT_band_row_lengths_l1980_198003


namespace NUMINAMATH_GPT_ratio_of_brownies_l1980_198021

def total_brownies : ℕ := 15
def eaten_on_monday : ℕ := 5
def eaten_on_tuesday : ℕ := total_brownies - eaten_on_monday

theorem ratio_of_brownies : eaten_on_tuesday / eaten_on_monday = 2 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_brownies_l1980_198021


namespace NUMINAMATH_GPT_expected_value_full_circles_l1980_198024

-- Definition of the conditions
def num_small_triangles (n : ℕ) : ℕ :=
  n^2

def potential_full_circle_vertices (n : ℕ) : ℕ :=
  if n < 3 then 0 else (n - 2) * (n - 1) / 2

def prob_full_circle : ℚ :=
  1 / 729

-- The expected number of full circles formed
def expected_full_circles (n : ℕ) : ℚ :=
  potential_full_circle_vertices n * prob_full_circle

-- The mathematical equivalence to be proved
theorem expected_value_full_circles (n : ℕ) : expected_full_circles n = (n - 2) * (n - 1) / 1458 := 
  sorry

end NUMINAMATH_GPT_expected_value_full_circles_l1980_198024


namespace NUMINAMATH_GPT_surface_area_circumscribed_sphere_l1980_198085

-- Define the problem
theorem surface_area_circumscribed_sphere (a b c : ℝ)
    (h1 : a^2 + b^2 = 3)
    (h2 : b^2 + c^2 = 5)
    (h3 : c^2 + a^2 = 4) : 
    4 * Real.pi * (a^2 + b^2 + c^2) / 4 = 6 * Real.pi :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_surface_area_circumscribed_sphere_l1980_198085


namespace NUMINAMATH_GPT_transformation_identity_l1980_198092

theorem transformation_identity (a b : ℝ) 
    (h1 : ∃ a b : ℝ, ∀ x y : ℝ, (y, -x) = (-7, 3) → (x, y) = (3, 7))
    (h2 : ∃ a b : ℝ, ∀ c d : ℝ, (d, c) = (3, -7) → (c, d) = (-7, 3)) :
    b - a = 4 :=
by
    sorry

end NUMINAMATH_GPT_transformation_identity_l1980_198092


namespace NUMINAMATH_GPT_area_of_region_enclosed_by_parabolas_l1980_198046

-- Define the given parabolas
def parabola1 (y : ℝ) : ℝ := -3 * y^2
def parabola2 (y : ℝ) : ℝ := 1 - 4 * y^2

-- Define the integral representing the area between the parabolas
noncomputable def areaBetweenParabolas : ℝ :=
  2 * (∫ y in (0 : ℝ)..1, (parabola2 y - parabola1 y))

-- The statement to be proved
theorem area_of_region_enclosed_by_parabolas :
  areaBetweenParabolas = 4 / 3 := 
sorry

end NUMINAMATH_GPT_area_of_region_enclosed_by_parabolas_l1980_198046


namespace NUMINAMATH_GPT_ticket_cost_l1980_198014

theorem ticket_cost (total_amount_collected : ℕ) (average_tickets_per_day : ℕ) (days : ℕ) 
  (h1 : total_amount_collected = 960) 
  (h2 : average_tickets_per_day = 80) 
  (h3 : days = 3) : 
  total_amount_collected / (average_tickets_per_day * days) = 4 :=
  sorry

end NUMINAMATH_GPT_ticket_cost_l1980_198014


namespace NUMINAMATH_GPT_solve_system_l1980_198072

theorem solve_system :
  ∃ (x y z : ℝ), 7 * x + y = 19 ∧ x + 3 * y = 1 ∧ 2 * x + y - 4 * z = 10 ∧ 2 * x + y + 3 * z = 1.25 :=
by
  sorry

end NUMINAMATH_GPT_solve_system_l1980_198072


namespace NUMINAMATH_GPT_range_of_xy_l1980_198063

theorem range_of_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2 * y + x * y = 30) :
  12 < x * y ∧ x * y < 870 :=
by sorry

end NUMINAMATH_GPT_range_of_xy_l1980_198063


namespace NUMINAMATH_GPT_geometric_sequence_term_l1980_198018

theorem geometric_sequence_term (a : ℕ → ℕ) (q : ℕ) (hq : q = 2) (ha2 : a 2 = 8) :
  a 6 = 128 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_term_l1980_198018


namespace NUMINAMATH_GPT_vacuum_cleaner_cost_l1980_198068

-- Define initial amount collected
def initial_amount : ℕ := 20

-- Define amount added each week
def weekly_addition : ℕ := 10

-- Define number of weeks
def number_of_weeks : ℕ := 10

-- Define the total amount after 10 weeks
def total_amount : ℕ := initial_amount + (weekly_addition * number_of_weeks)

-- Prove that the total amount is equal to the cost of the vacuum cleaner
theorem vacuum_cleaner_cost : total_amount = 120 := by
  sorry

end NUMINAMATH_GPT_vacuum_cleaner_cost_l1980_198068


namespace NUMINAMATH_GPT_hawks_points_l1980_198036

theorem hawks_points (E H : ℕ) (h₁ : E + H = 82) (h₂ : E = H + 18) (h₃ : H ≥ 9) : H = 32 :=
sorry

end NUMINAMATH_GPT_hawks_points_l1980_198036


namespace NUMINAMATH_GPT_find_remainder_l1980_198054

theorem find_remainder (S : Finset ℕ) (h : ∀ n ∈ S, ∃ m, n^2 + 10 * n - 2010 = m^2) :
  (S.sum id) % 1000 = 304 := by
  sorry

end NUMINAMATH_GPT_find_remainder_l1980_198054


namespace NUMINAMATH_GPT_continuous_zero_point_condition_l1980_198089

theorem continuous_zero_point_condition (f : ℝ → ℝ) {a b : ℝ} (h_cont : ContinuousOn f (Set.Icc a b)) :
  (f a * f b < 0) → (∃ c ∈ Set.Ioo a b, f c = 0) ∧ ¬ (∃ c ∈ Set.Ioo a b, f c = 0 → f a * f b < 0) :=
sorry

end NUMINAMATH_GPT_continuous_zero_point_condition_l1980_198089


namespace NUMINAMATH_GPT_find_m_range_l1980_198044

/--
Given:
1. Proposition \( p \) (p): The equation \(\frac{x^2}{2} + \frac{y^2}{m} = 1\) represents an ellipse with foci on the \( y \)-axis.
2. Proposition \( q \) (q): \( f(x) = \frac{4}{3}x^3 - 2mx^2 + (4m-3)x - m \) is monotonically increasing on \((-\infty, +\infty)\).

Prove:
If \( \neg p \land q \) is true, then the range of values for \( m \) is \( [1, 2] \).
-/

def p (m : ℝ) : Prop :=
  m > 2

def q (m : ℝ) : Prop :=
  ∀ x : ℝ, (4 * x^2 - 4 * m * x + 4 * m - 3) >= 0

theorem find_m_range (m : ℝ) (hpq : ¬ p m ∧ q m) : 1 ≤ m ∧ m ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_find_m_range_l1980_198044


namespace NUMINAMATH_GPT_jack_needs_5_rocks_to_equal_weights_l1980_198052

-- Given Conditions
def WeightJack : ℕ := 60
def WeightAnna : ℕ := 40
def WeightRock : ℕ := 4

-- Theorem Statement
theorem jack_needs_5_rocks_to_equal_weights : (WeightJack - WeightAnna) / WeightRock = 5 :=
by
  sorry

end NUMINAMATH_GPT_jack_needs_5_rocks_to_equal_weights_l1980_198052


namespace NUMINAMATH_GPT_second_shipment_is_13_l1980_198029

-- Definitions based on the conditions
def first_shipment : ℕ := 7
def third_shipment : ℕ := 45
def total_couscous_used : ℕ := 13 * 5 -- 65
def total_couscous_from_three_shipments (second_shipment : ℕ) : ℕ :=
  first_shipment + second_shipment + third_shipment

-- Statement of the proof problem corresponding to the conditions and question
theorem second_shipment_is_13 (x : ℕ) 
  (h : total_couscous_used = total_couscous_from_three_shipments x) : x = 13 := 
by
  sorry

end NUMINAMATH_GPT_second_shipment_is_13_l1980_198029


namespace NUMINAMATH_GPT_correct_result_without_mistake_l1980_198028

variable {R : Type*} [CommRing R] (a b c : R)
variable (A : R)

theorem correct_result_without_mistake :
  A + 2 * (ab + 2 * bc - 4 * ac) = (3 * ab - 2 * ac + 5 * bc) → 
  A - 2 * (ab + 2 * bc - 4 * ac) = -ab + 14 * ac - 3 * bc :=
by
  sorry

end NUMINAMATH_GPT_correct_result_without_mistake_l1980_198028


namespace NUMINAMATH_GPT_problem_a_b_squared_l1980_198093

theorem problem_a_b_squared {a b : ℝ} (h1 : a + 3 = (b-1)^2) (h2 : b + 3 = (a-1)^2) (h3 : a ≠ b) : a^2 + b^2 = 10 :=
by
  sorry

end NUMINAMATH_GPT_problem_a_b_squared_l1980_198093


namespace NUMINAMATH_GPT_probability_problems_l1980_198096

theorem probability_problems (x : ℕ) :
  (0 = (if 8 + 12 > 8 then 0 else 1)) ∧
  (1 = (1 - 0)) ∧
  (3 / 5 = 12 / 20) ∧
  (4 / 5 = (8 + x) / 20 → x = 8) := by sorry

end NUMINAMATH_GPT_probability_problems_l1980_198096


namespace NUMINAMATH_GPT_line_through_intersections_l1980_198091

-- Conditions
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - x + y - 2 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Theorem statement
theorem line_through_intersections (x y : ℝ) :
  circle1 x y → circle2 x y → x - y - 3 = 0 :=
by
  sorry

end NUMINAMATH_GPT_line_through_intersections_l1980_198091


namespace NUMINAMATH_GPT_arithmetic_seq_max_sum_l1980_198035

noncomputable def max_arith_seq_sum_lemma (a1 d : ℤ) (n : ℕ) : ℤ :=
  n * a1 + (n * (n - 1) / 2) * d

theorem arithmetic_seq_max_sum :
  ∀ (a1 d : ℤ),
    (3 * a1 + 6 * d = 9) →
    (a1 + 5 * d = -9) →
    max_arith_seq_sum_lemma a1 d 3 = 21 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_max_sum_l1980_198035


namespace NUMINAMATH_GPT_find_x_l1980_198073

theorem find_x (x : ℕ) (h : 27^3 + 27^3 + 27^3 + 27^3 = 3^x) : x = 11 :=
sorry

end NUMINAMATH_GPT_find_x_l1980_198073


namespace NUMINAMATH_GPT_negation_of_exists_l1980_198004

theorem negation_of_exists (h : ∃ x : ℝ, x > 0 ∧ x^2 + 3*x + 1 < 0) : ∀ x : ℝ, x > 0 → x^2 + 3*x + 1 ≥ 0 :=
sorry

end NUMINAMATH_GPT_negation_of_exists_l1980_198004


namespace NUMINAMATH_GPT_family_member_bites_count_l1980_198058

-- Definitions based on the given conditions
def cyrus_bites_arms_and_legs : Nat := 14
def cyrus_bites_body : Nat := 10
def family_size : Nat := 6
def total_bites_cyrus : Nat := cyrus_bites_arms_and_legs + cyrus_bites_body
def total_bites_family : Nat := total_bites_cyrus / 2

-- Translation of the question to a theorem statement
theorem family_member_bites_count : (total_bites_family / family_size) = 2 := by
  -- use sorry to indicate the proof is skipped
  sorry

end NUMINAMATH_GPT_family_member_bites_count_l1980_198058


namespace NUMINAMATH_GPT_flour_needed_for_bread_l1980_198083

-- Definitions based on conditions
def flour_per_loaf : ℝ := 2.5
def number_of_loaves : ℕ := 2

-- Theorem statement
theorem flour_needed_for_bread : flour_per_loaf * number_of_loaves = 5 :=
by sorry

end NUMINAMATH_GPT_flour_needed_for_bread_l1980_198083


namespace NUMINAMATH_GPT_f_monotonic_decreasing_interval_l1980_198016

noncomputable def f (x : ℝ) : ℝ := (1/2)^(x^2 - 2*x)

theorem f_monotonic_decreasing_interval : 
  ∀ x1 x2 : ℝ, 1 ≤ x1 → x1 ≤ x2 → f x2 ≤ f x1 := 
sorry

end NUMINAMATH_GPT_f_monotonic_decreasing_interval_l1980_198016


namespace NUMINAMATH_GPT_number_of_ways_to_buy_three_items_l1980_198040

def headphones : ℕ := 9
def mice : ℕ := 13
def keyboards : ℕ := 5
def keyboard_mouse_sets : ℕ := 4
def headphone_mouse_sets : ℕ := 5

theorem number_of_ways_to_buy_three_items : 
  (keyboard_mouse_sets * headphones) + (headphone_mouse_sets * keyboards) + (headphones * mice * keyboards) = 646 := 
by 
  sorry

end NUMINAMATH_GPT_number_of_ways_to_buy_three_items_l1980_198040


namespace NUMINAMATH_GPT_theme_park_ratio_l1980_198038

theorem theme_park_ratio (a c : ℕ) (h_cost_adult : 20 * a + 15 * c = 1600) (h_eq_ratio : a * 28 = c * 59) :
  a / c = 59 / 28 :=
by
  /-
  Proof steps would go here.
  -/
  sorry

end NUMINAMATH_GPT_theme_park_ratio_l1980_198038


namespace NUMINAMATH_GPT_age_of_seventh_person_l1980_198075

theorem age_of_seventh_person (A1 A2 A3 A4 A5 A6 A7 D1 D2 D3 D4 D5 : ℕ) 
    (h1 : A1 < A2) (h2 : A2 < A3) (h3 : A3 < A4) (h4 : A4 < A5) (h5 : A5 < A6) 
    (h6 : A2 = A1 + D1) (h7 : A3 = A2 + D2) (h8 : A4 = A3 + D3) 
    (h9 : A5 = A4 + D4) (h10 : A6 = A5 + D5)
    (h11 : A1 + A2 + A3 + A4 + A5 + A6 = 246) 
    (h12 : 246 + A7 = 315) : A7 = 69 :=
by
  sorry

end NUMINAMATH_GPT_age_of_seventh_person_l1980_198075


namespace NUMINAMATH_GPT_find_f_100_l1980_198086

-- Define the function f such that it satisfies the condition f(10^x) = x
noncomputable def f : ℝ → ℝ := sorry

-- Define the main theorem to prove f(100) = 2 given the condition f(10^x) = x
theorem find_f_100 (h : ∀ x : ℝ, f (10^x) = x) : f 100 = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_f_100_l1980_198086


namespace NUMINAMATH_GPT_remaining_perimeter_of_square_with_cutouts_l1980_198079

theorem remaining_perimeter_of_square_with_cutouts 
  (square_side : ℝ) (green_square_side : ℝ) (init_perimeter : ℝ) 
  (green_square_perimeter_increase : ℝ) (final_perimeter : ℝ) :
  square_side = 10 → green_square_side = 2 →
  init_perimeter = 4 * square_side → green_square_perimeter_increase = 4 * green_square_side →
  final_perimeter = init_perimeter + green_square_perimeter_increase →
  final_perimeter = 44 :=
by
  intros hsquare_side hgreen_square_side hinit_perimeter hgreen_incr hfinal_perimeter
  -- Proof steps can be added here
  sorry

end NUMINAMATH_GPT_remaining_perimeter_of_square_with_cutouts_l1980_198079


namespace NUMINAMATH_GPT_seedlings_planted_l1980_198012

theorem seedlings_planted (x : ℕ) (h1 : 2 * x + x = 1200) : x = 400 :=
by {
  sorry
}

end NUMINAMATH_GPT_seedlings_planted_l1980_198012


namespace NUMINAMATH_GPT_senior_junior_ratio_l1980_198010

variable (S J : ℕ) (k : ℕ)

theorem senior_junior_ratio (h1 : S = k * J) 
                           (h2 : (1/8 : ℚ) * S + (3/4 : ℚ) * J = (1/3 : ℚ) * (S + J)) : 
                           k = 2 :=
by
  sorry

end NUMINAMATH_GPT_senior_junior_ratio_l1980_198010


namespace NUMINAMATH_GPT_other_factor_of_LCM_l1980_198022

-- Definitions and conditions
def A : ℕ := 624
def H : ℕ := 52 
def HCF (a b : ℕ) : ℕ := Nat.gcd a b

-- Hypotheses based on the problem statement
axiom h_hcf : HCF A 52 = 52

-- The desired statement to prove
theorem other_factor_of_LCM (B : ℕ) (y : ℕ) : HCF A B = H → (A * y = 624) → y = 1 := 
by 
  intro h1 h2
  -- Actual proof steps are omitted
  sorry

end NUMINAMATH_GPT_other_factor_of_LCM_l1980_198022


namespace NUMINAMATH_GPT_product_mod5_is_zero_l1980_198031

theorem product_mod5_is_zero :
  (2023 * 2024 * 2025 * 2026) % 5 = 0 :=
by
  sorry

end NUMINAMATH_GPT_product_mod5_is_zero_l1980_198031


namespace NUMINAMATH_GPT_fraction_denominator_l1980_198039

theorem fraction_denominator (S : ℚ) (h : S = 0.666666) : ∃ (n : ℕ), S = 2 / 3 ∧ n = 3 :=
by
  sorry

end NUMINAMATH_GPT_fraction_denominator_l1980_198039


namespace NUMINAMATH_GPT_ratio_cost_to_marked_price_l1980_198019

theorem ratio_cost_to_marked_price (x : ℝ) 
  (h_discount: ∀ y, y = marked_price → selling_price = (3/4) * y)
  (h_cost: ∀ z, z = selling_price → cost_price = (2/3) * z) :
  cost_price / marked_price = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_cost_to_marked_price_l1980_198019


namespace NUMINAMATH_GPT_profit_percent_is_correct_l1980_198069

noncomputable def profit_percent : ℝ := 
  let marked_price_per_pen := 1 
  let pens_bought := 56 
  let effective_payment := 46 
  let discount := 0.01
  let cost_price_per_pen := effective_payment / pens_bought
  let selling_price_per_pen := marked_price_per_pen * (1 - discount)
  let total_selling_price := pens_bought * selling_price_per_pen
  let profit := total_selling_price - effective_payment
  (profit / effective_payment) * 100

theorem profit_percent_is_correct : abs (profit_percent - 20.52) < 0.01 :=
by
  sorry

end NUMINAMATH_GPT_profit_percent_is_correct_l1980_198069


namespace NUMINAMATH_GPT_find_abc_l1980_198055

theorem find_abc :
  ∃ (a b c : ℤ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ 
  a + b + c = 30 ∧
  (1/a + 1/b + 1/c + 450/(a*b*c) = 1) ∧ 
  a*b*c = 1912 :=
sorry

end NUMINAMATH_GPT_find_abc_l1980_198055


namespace NUMINAMATH_GPT_y_minus_x_eq_seven_point_five_l1980_198061

theorem y_minus_x_eq_seven_point_five (x y : ℚ) (h1 : x + y = 8) (h2 : y - 3 * x = 7) :
  y - x = 7.5 :=
by sorry

end NUMINAMATH_GPT_y_minus_x_eq_seven_point_five_l1980_198061


namespace NUMINAMATH_GPT_remaining_student_number_l1980_198027

-- Definitions based on given conditions
def total_students := 48
def sample_size := 6
def sampled_students := [5, 21, 29, 37, 45]

-- Interval calculation and pattern definition based on systematic sampling
def sampling_interval := total_students / sample_size
def sampled_student_numbers (n : Nat) : Nat := 5 + sampling_interval * (n - 1)

-- Prove the student number within the sample
theorem remaining_student_number : ∃ n, n ∉ sampled_students ∧ sampled_student_numbers n = 13 :=
by
  sorry

end NUMINAMATH_GPT_remaining_student_number_l1980_198027


namespace NUMINAMATH_GPT_tan_value_of_point_on_graph_l1980_198081

theorem tan_value_of_point_on_graph (a : ℝ) (h : (4 : ℝ) ^ (1/2) = a) : 
  Real.tan ((a / 6) * Real.pi) = Real.sqrt 3 :=
by 
  sorry

end NUMINAMATH_GPT_tan_value_of_point_on_graph_l1980_198081


namespace NUMINAMATH_GPT_cylinder_volume_increase_l1980_198048

theorem cylinder_volume_increase {R H : ℕ} (x : ℚ) (C : ℝ) (π : ℝ) 
  (hR : R = 8) (hH : H = 3) (hπ : π = Real.pi)
  (hV : ∃ C > 0, π * (R + x)^2 * (H + x) = π * R^2 * H + C) :
  x = 16 / 3 :=
by
  sorry

end NUMINAMATH_GPT_cylinder_volume_increase_l1980_198048


namespace NUMINAMATH_GPT_factorize_problem_1_factorize_problem_2_l1980_198090

variables (x y : ℝ)

-- Problem 1: Prove that x^2 * y - 4 * x * y + 4 * y = y * (x - 2) ^ 2
theorem factorize_problem_1 : 
  x^2 * y - 4 * x * y + 4 * y = y * (x - 2) ^ 2 :=
sorry

-- Problem 2: Prove that x^2 - 4 * y^2 = (x + 2 * y) * (x - 2 * y)
theorem factorize_problem_2 : 
  x^2 - 4 * y^2 = (x + 2 * y) * (x - 2 * y) :=
sorry

end NUMINAMATH_GPT_factorize_problem_1_factorize_problem_2_l1980_198090


namespace NUMINAMATH_GPT_inequality_proof_l1980_198080

open Real

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (h : a * b * c * d = 1) :
  (a^4 + b^4) / (a^2 + b^2) + (b^4 + c^4) / (b^2 + c^2) + (c^4 + d^4) / (c^2 + d^2) + (d^4 + a^4) / (d^2 + a^2) ≥ 4 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1980_198080


namespace NUMINAMATH_GPT_common_root_l1980_198071

theorem common_root (p : ℝ) :
  (∃ x : ℝ, x^2 - (p+2)*x + 2*p + 6 = 0 ∧ 2*x^2 - (p+4)*x + 2*p + 3 = 0) ↔ (p = -3 ∨ p = 9) :=
by
  sorry

end NUMINAMATH_GPT_common_root_l1980_198071


namespace NUMINAMATH_GPT_no_triangle_100_sticks_yes_triangle_99_sticks_l1980_198051

-- Definitions for the sums of lengths of sticks
def sum_lengths (n : ℕ) : ℕ := (n * (n + 1)) / 2

-- Conditions and questions for the problem
def is_divisible_by_3 (x : ℕ) : Prop := x % 3 = 0

-- Proof problem for n = 100
theorem no_triangle_100_sticks : ¬ (is_divisible_by_3 (sum_lengths 100)) := by
  sorry

-- Proof problem for n = 99
theorem yes_triangle_99_sticks : is_divisible_by_3 (sum_lengths 99) := by
  sorry

end NUMINAMATH_GPT_no_triangle_100_sticks_yes_triangle_99_sticks_l1980_198051
