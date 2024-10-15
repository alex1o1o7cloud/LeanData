import Mathlib

namespace NUMINAMATH_GPT_triangle_angles_geometric_progression_l2374_237462

-- Theorem: If the sides of a triangle whose angles form an arithmetic progression are in geometric progression, then all three angles are 60°.
theorem triangle_angles_geometric_progression (A B C : ℝ) (a b c : ℝ)
  (h_arith_progression : 2 * B = A + C)
  (h_sum_angles : A + B + C = 180)
  (h_geo_progression : (a / b) = (b / c))
  (h_b_angle : B = 60) :
  A = 60 ∧ B = 60 ∧ C = 60 :=
by
  sorry

end NUMINAMATH_GPT_triangle_angles_geometric_progression_l2374_237462


namespace NUMINAMATH_GPT_lindy_distance_traveled_l2374_237498

/-- Jack and Christina are standing 240 feet apart on a level surface. 
Jack walks in a straight line toward Christina at a constant speed of 5 feet per second. 
Christina walks in a straight line toward Jack at a constant speed of 3 feet per second. 
Lindy runs at a constant speed of 9 feet per second from Christina to Jack, back to Christina, back to Jack, and so forth. 
The total distance Lindy travels when the three meet at one place is 270 feet. -/
theorem lindy_distance_traveled
    (initial_distance : ℝ)
    (jack_speed : ℝ)
    (christina_speed : ℝ)
    (lindy_speed : ℝ)
    (time_to_meet : ℝ)
    (total_distance_lindy : ℝ) :
    initial_distance = 240 ∧
    jack_speed = 5 ∧
    christina_speed = 3 ∧
    lindy_speed = 9 ∧
    time_to_meet = (initial_distance / (jack_speed + christina_speed)) ∧
    total_distance_lindy = lindy_speed * time_to_meet →
    total_distance_lindy = 270 :=
by
  sorry

end NUMINAMATH_GPT_lindy_distance_traveled_l2374_237498


namespace NUMINAMATH_GPT_gym_hours_per_week_l2374_237467

-- Definitions for conditions
def timesAtGymEachWeek : ℕ := 3
def weightliftingTimeEachDay : ℕ := 1
def warmupCardioFraction : ℚ := 1 / 3

-- The theorem to prove
theorem gym_hours_per_week : (timesAtGymEachWeek * (weightliftingTimeEachDay + weightliftingTimeEachDay * warmupCardioFraction) = 4) := 
by
  sorry

end NUMINAMATH_GPT_gym_hours_per_week_l2374_237467


namespace NUMINAMATH_GPT_min_value_of_function_l2374_237407

theorem min_value_of_function (x : ℝ) (h : x > 2) : ∃ y, y = (x^2 - 4*x + 8) / (x - 2) ∧ (∀ z, z = (x^2 - 4*x + 8) / (x - 2) → y ≤ z) :=
sorry

end NUMINAMATH_GPT_min_value_of_function_l2374_237407


namespace NUMINAMATH_GPT_student_B_speed_l2374_237473

theorem student_B_speed (x : ℝ) (h1 : 12 / x - 1 / 6 = 12 / (1.2 * x)) : x = 12 :=
by
  sorry

end NUMINAMATH_GPT_student_B_speed_l2374_237473


namespace NUMINAMATH_GPT_decimal_equiv_of_one_fourth_cubed_l2374_237499

theorem decimal_equiv_of_one_fourth_cubed : (1 / 4 : ℝ) ^ 3 = 0.015625 := 
by sorry

end NUMINAMATH_GPT_decimal_equiv_of_one_fourth_cubed_l2374_237499


namespace NUMINAMATH_GPT_candied_apple_price_l2374_237433

theorem candied_apple_price
  (x : ℝ) -- price of each candied apple in dollars
  (h1 : 15 * x + 12 * 1.5 = 48) -- total earnings equation
  : x = 2 := 
sorry

end NUMINAMATH_GPT_candied_apple_price_l2374_237433


namespace NUMINAMATH_GPT_count_ordered_pairs_l2374_237428

theorem count_ordered_pairs : 
  ∃ n, n = 719 ∧ 
    (∀ (a b : ℕ), a + b = 1100 → 
      (∀ d ∈ [a, b], 
        ¬(∃ k : ℕ, d = 10 * k ∨ d % 10 = 0 ∨ d / 10 % 10 = 0 ∨ d % 5 = 0))) -> n = 719 :=
by
  sorry

end NUMINAMATH_GPT_count_ordered_pairs_l2374_237428


namespace NUMINAMATH_GPT_brick_wall_l2374_237456

theorem brick_wall (y : ℕ) (h1 : ∀ y, 6 * ((y / 8) + (y / 12) - 12) = y) : y = 288 :=
sorry

end NUMINAMATH_GPT_brick_wall_l2374_237456


namespace NUMINAMATH_GPT_oranges_to_put_back_l2374_237420

theorem oranges_to_put_back
  (price_apple price_orange : ℕ)
  (A_all O_all : ℕ)
  (mean_initial_fruit mean_final_fruit : ℕ)
  (A O x : ℕ)
  (h_price_apple : price_apple = 40)
  (h_price_orange : price_orange = 60)
  (h_total_fruit : A_all + O_all = 10)
  (h_mean_initial : mean_initial_fruit = 54)
  (h_mean_final : mean_final_fruit = 50)
  (h_total_cost_initial : price_apple * A_all + price_orange * O_all = mean_initial_fruit * (A_all + O_all))
  (h_total_cost_final : price_apple * A + price_orange * (O - x) = mean_final_fruit * (A + (O - x)))
  : x = 4 := 
  sorry

end NUMINAMATH_GPT_oranges_to_put_back_l2374_237420


namespace NUMINAMATH_GPT_percentage_of_female_officers_on_duty_l2374_237487

theorem percentage_of_female_officers_on_duty
    (on_duty : ℕ) (half_on_duty_female : on_duty / 2 = 100)
    (total_female_officers : ℕ)
    (total_female_officers_value : total_female_officers = 1000)
    : (100 / total_female_officers : ℝ) * 100 = 10 :=
by sorry

end NUMINAMATH_GPT_percentage_of_female_officers_on_duty_l2374_237487


namespace NUMINAMATH_GPT_oldest_brother_age_ratio_l2374_237483

-- Define the ages
def rick_age : ℕ := 15
def youngest_brother_age : ℕ := 3
def smallest_brother_age : ℕ := youngest_brother_age + 2
def middle_brother_age : ℕ := smallest_brother_age * 2
def oldest_brother_age : ℕ := middle_brother_age * 3

-- Define the ratio
def expected_ratio : ℕ := oldest_brother_age / rick_age

theorem oldest_brother_age_ratio : expected_ratio = 2 := by
  sorry 

end NUMINAMATH_GPT_oldest_brother_age_ratio_l2374_237483


namespace NUMINAMATH_GPT_lionel_distance_walked_when_met_l2374_237479

theorem lionel_distance_walked_when_met (distance_between : ℕ) (lionel_speed : ℕ) (walt_speed : ℕ) (advance_time : ℕ) 
(h1 : distance_between = 48) 
(h2 : lionel_speed = 2) 
(h3 : walt_speed = 6) 
(h4 : advance_time = 2) : 
  ∃ D : ℕ, D = 15 :=
by
  sorry

end NUMINAMATH_GPT_lionel_distance_walked_when_met_l2374_237479


namespace NUMINAMATH_GPT_range_of_a_l2374_237418

-- Definitions of the conditions
def p (x : ℝ) : Prop := x^2 - 8 * x - 20 < 0
def q (x : ℝ) (a : ℝ) : Prop := x^2 - 2 * x + 1 - a^2 ≤ 0 ∧ a > 0

-- Statement of the theorem that proves the range of a
theorem range_of_a (x : ℝ) (a : ℝ) :
  (¬ (p x) → ¬ (q x a)) ∧ (¬ (q x a) → ¬ (p x)) → (a ≥ 9) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2374_237418


namespace NUMINAMATH_GPT_probability_at_least_two_same_l2374_237489

theorem probability_at_least_two_same :
  let total_outcomes := (8 ^ 4 : ℕ)
  let num_diff_outcomes := (8 * 7 * 6 * 5 : ℕ)
  let probability_diff := (num_diff_outcomes : ℝ) / total_outcomes
  let probability_at_least_two := 1 - probability_diff
  probability_at_least_two = (151 : ℝ) / 256 :=
by
  sorry

end NUMINAMATH_GPT_probability_at_least_two_same_l2374_237489


namespace NUMINAMATH_GPT_find_m_for_unique_solution_l2374_237450

theorem find_m_for_unique_solution :
  ∃ m : ℝ, (m = -8 + 2 * Real.sqrt 15 ∨ m = -8 - 2 * Real.sqrt 15) ∧ 
  ∀ x : ℝ, (mx - 2 ≠ 0 → (x + 3) / (mx - 2) = x + 1 ↔ ∃! x : ℝ, (mx - 2) * (x + 1) = (x + 3)) :=
sorry

end NUMINAMATH_GPT_find_m_for_unique_solution_l2374_237450


namespace NUMINAMATH_GPT_Integers_and_fractions_are_rational_numbers_l2374_237463

-- Definitions from conditions
def is_fraction (x : ℚ) : Prop :=
  ∃a b : ℤ, b ≠ 0 ∧ x = (a : ℚ) / (b : ℚ)

def is_integer (x : ℤ) : Prop := 
  ∃n : ℤ, x = n

def is_rational (x : ℚ) : Prop := 
  ∃a b : ℤ, b ≠ 0 ∧ x = (a : ℚ) / (b : ℚ)

-- The statement to be proven
theorem Integers_and_fractions_are_rational_numbers (x : ℚ) : 
  (∃n : ℤ, x = (n : ℚ)) ∨ is_fraction x ↔ is_rational x :=
by sorry

end NUMINAMATH_GPT_Integers_and_fractions_are_rational_numbers_l2374_237463


namespace NUMINAMATH_GPT_find_xyz_l2374_237451

theorem find_xyz (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 25)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 7) : 
  x * y * z = 6 := 
by 
  sorry

end NUMINAMATH_GPT_find_xyz_l2374_237451


namespace NUMINAMATH_GPT_num_ways_arrange_passengers_l2374_237477

theorem num_ways_arrange_passengers 
  (seats : ℕ) (passengers : ℕ) (consecutive_empty : ℕ)
  (h1 : seats = 10) (h2 : passengers = 4) (h3 : consecutive_empty = 5) :
  ∃ ways, ways = 480 := by
  sorry

end NUMINAMATH_GPT_num_ways_arrange_passengers_l2374_237477


namespace NUMINAMATH_GPT_number_line_problem_l2374_237402

theorem number_line_problem (A B C : ℤ) (hA : A = -1) (hB : B = A - 5 + 6) (hC : abs (C - B) = 5) :
  C = 5 ∨ C = -5 :=
by sorry

end NUMINAMATH_GPT_number_line_problem_l2374_237402


namespace NUMINAMATH_GPT_eq_solutions_count_l2374_237444

def f (x a : ℝ) : ℝ := abs (abs (abs (x - a) - 1) - 1)

theorem eq_solutions_count (a b : ℝ) : 
  ∃ count : ℕ, (∀ x : ℝ, f x a = abs b → true) ∧ count = 4 :=
by
  sorry

end NUMINAMATH_GPT_eq_solutions_count_l2374_237444


namespace NUMINAMATH_GPT_smallest_solution_l2374_237403

noncomputable def equation (x : ℝ) : Prop :=
  (1 / (x - 3) + 1 / (x - 5) + 1 / (x - 6) = 4 / (x - 4))

def valid_x (x : ℝ) : Prop :=
  x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 5 ∧ x ≠ 6

theorem smallest_solution (x : ℝ) (h1 : equation x) (h2 : valid_x x) : x = 16 := sorry

end NUMINAMATH_GPT_smallest_solution_l2374_237403


namespace NUMINAMATH_GPT_stratified_sampling_l2374_237438

theorem stratified_sampling
  (students_class1 : ℕ)
  (students_class2 : ℕ)
  (formation_slots : ℕ)
  (total_students : ℕ)
  (prob_selected: ℚ)
  (selected_class1 : ℕ)
  (selected_class2 : ℕ)
  (h1 : students_class1 = 54)
  (h2 : students_class2 = 42)
  (h3 : formation_slots = 16)
  (h4 : total_students = students_class1 + students_class2)
  (h5 : prob_selected = formation_slots / total_students)
  (h6 : selected_class1 = students_class1 * prob_selected)
  (h7 : selected_class2 = students_class2 * prob_selected)
  : selected_class1 = 9 ∧ selected_class2 = 7 := by
  sorry

end NUMINAMATH_GPT_stratified_sampling_l2374_237438


namespace NUMINAMATH_GPT_Adam_bought_9_cat_food_packages_l2374_237458

def num_cat_food_packages (c : ℕ) : Prop :=
  let cat_cans := 10 * c
  let dog_cans := 7 * 5
  cat_cans = dog_cans + 55

theorem Adam_bought_9_cat_food_packages : num_cat_food_packages 9 :=
by
  unfold num_cat_food_packages
  sorry

end NUMINAMATH_GPT_Adam_bought_9_cat_food_packages_l2374_237458


namespace NUMINAMATH_GPT_binom_21_10_l2374_237495

theorem binom_21_10 :
  (Nat.choose 19 9 = 92378) →
  (Nat.choose 19 10 = 92378) →
  (Nat.choose 19 11 = 75582) →
  Nat.choose 21 10 = 352716 := by
  sorry

end NUMINAMATH_GPT_binom_21_10_l2374_237495


namespace NUMINAMATH_GPT_num_possible_y_l2374_237492

theorem num_possible_y : 
  (∃ (count : ℕ), count = (54 - 26 + 1) ∧ 
  ∀ (y : ℤ), 25 < y ∧ y < 55 ↔ (26 ≤ y ∧ y ≤ 54)) :=
by {
  sorry 
}

end NUMINAMATH_GPT_num_possible_y_l2374_237492


namespace NUMINAMATH_GPT_airlines_routes_l2374_237426

open Function

theorem airlines_routes
  (n_regions m_regions : ℕ)
  (h_n_regions : n_regions = 18)
  (h_m_regions : m_regions = 10)
  (A B : Fin n_regions → Fin n_regions → Bool)
  (h_flight : ∀ r1 r2 : Fin n_regions, r1 ≠ r2 → (A r1 r2 = true ∨ B r1 r2 = true) ∧ ¬(A r1 r2 = true ∧ B r1 r2 = true)) :
  ∃ (routes_A routes_B : List (List (Fin n_regions))),
    (∀ route ∈ routes_A, 2 ∣ route.length) ∧
    (∀ route ∈ routes_B, 2 ∣ route.length) ∧
    routes_A ≠ [] ∧
    routes_B ≠ [] :=
sorry

end NUMINAMATH_GPT_airlines_routes_l2374_237426


namespace NUMINAMATH_GPT_polygon_vertices_l2374_237409

-- Define the number of diagonals from one vertex
def diagonals_from_one_vertex (n : ℕ) := n - 3

-- The main theorem stating the number of vertices is 9 given 6 diagonals from one vertex
theorem polygon_vertices (D : ℕ) (n : ℕ) (h : D = 6) (h_diagonals : diagonals_from_one_vertex n = D) :
  n = 9 := by
  sorry

end NUMINAMATH_GPT_polygon_vertices_l2374_237409


namespace NUMINAMATH_GPT_find_A_l2374_237454

theorem find_A (A B : ℚ) (h1 : B - A = 211.5) (h2 : B = 10 * A) : A = 23.5 :=
by sorry

end NUMINAMATH_GPT_find_A_l2374_237454


namespace NUMINAMATH_GPT_problem_sequence_sum_l2374_237478

theorem problem_sequence_sum (a : ℤ) (h : 14 * a^2 + 7 * a = 135) : 7 * a + (a - 1) = 23 :=
by {
  sorry
}

end NUMINAMATH_GPT_problem_sequence_sum_l2374_237478


namespace NUMINAMATH_GPT_problem1_problem2_l2374_237455

-- Problem (1)
theorem problem1 (f : ℝ → ℝ) (h : ∀ x ≠ 0, f (2 / x + 2) = x + 1) : 
  ∀ x ≠ 2, f x = x / (x - 2) :=
sorry

-- Problem (2)
theorem problem2 (f : ℝ → ℝ) (h : ∃ k b, ∀ x, f x = k * x + b ∧ k ≠ 0)
  (h' : ∀ x, 3 * f (x + 1) - 2 * f (x - 1) = 2 * x + 17) :
  ∀ x, f x = 2 * x + 7 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l2374_237455


namespace NUMINAMATH_GPT_incorrect_conclusion_C_l2374_237436

variable {a : ℕ → ℝ} {q : ℝ}

-- Conditions
def geo_seq (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n+1) = a n * q

theorem incorrect_conclusion_C 
  (h_geo: geo_seq a q)
  (h_cond: a 1 * a 2 < 0) : 
  a 1 * a 5 > 0 :=
by 
  sorry

end NUMINAMATH_GPT_incorrect_conclusion_C_l2374_237436


namespace NUMINAMATH_GPT_hexagon_perimeter_l2374_237496

-- Define the side length 's' based on the given area condition
def side_length (s : ℝ) : Prop :=
  (3 * Real.sqrt 2 + Real.sqrt 3) / 4 * s^2 = 12

-- The theorem to prove
theorem hexagon_perimeter (s : ℝ) (h : side_length s) : 
  6 * s = 6 * Real.sqrt (48 / (3 * Real.sqrt 2 + Real.sqrt 3)) :=
by
  sorry

end NUMINAMATH_GPT_hexagon_perimeter_l2374_237496


namespace NUMINAMATH_GPT_unique_a_exists_for_prime_p_l2374_237447

theorem unique_a_exists_for_prime_p (p : ℕ) [Fact p.Prime] :
  (∃! (a : ℕ), a ∈ Finset.range (p + 1) ∧ (a^3 - 3*a + 1) % p = 0) ↔ p = 3 := by
  sorry

end NUMINAMATH_GPT_unique_a_exists_for_prime_p_l2374_237447


namespace NUMINAMATH_GPT_false_statement_l2374_237419

-- Define propositions p and q
def p := ∀ x : ℝ, (|x| = x) ↔ (x ≥ 0)
def q := ∀ (f : ℝ → ℝ), (∀ x, f (-x) = -f x) → (∃ origin : ℝ, ∀ y : ℝ, f (origin + y) = f (origin - y))

-- Define the possible answers
def option_A := p ∨ q
def option_B := p ∧ q
def option_C := ¬p ∧ q
def option_D := ¬p ∨ q

-- Define the false option (the correct answer was B)
def false_proposition := option_B

-- The statement to prove
theorem false_statement : false_proposition = false :=
by sorry

end NUMINAMATH_GPT_false_statement_l2374_237419


namespace NUMINAMATH_GPT_elevator_initial_floors_down_l2374_237423

theorem elevator_initial_floors_down (x : ℕ) (h1 : 9 - x + 3 + 8 = 13) : x = 7 := 
by
  -- Proof
  sorry

end NUMINAMATH_GPT_elevator_initial_floors_down_l2374_237423


namespace NUMINAMATH_GPT_proof_problem_l2374_237446

theorem proof_problem (x : ℤ) (h : (x - 34) / 10 = 2) : (x - 5) / 7 = 7 :=
  sorry

end NUMINAMATH_GPT_proof_problem_l2374_237446


namespace NUMINAMATH_GPT_arithmetic_sequence_product_l2374_237405

theorem arithmetic_sequence_product (b : ℕ → ℤ) (d : ℤ)
  (h_arith_seq : ∀ n, b (n + 1) = b n + d)
  (h_increasing : ∀ n, b n < b (n + 1))
  (h_condition : b 5 * b 6 = 14) :
  (b 4 * b 7 = -324) ∨ (b 4 * b 7 = -36) :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_product_l2374_237405


namespace NUMINAMATH_GPT_largest_undefined_x_value_l2374_237437

theorem largest_undefined_x_value :
  ∃ x : ℝ, (6 * x^2 - 65 * x + 54 = 0) ∧ (∀ y : ℝ, (6 * y^2 - 65 * y + 54 = 0) → y ≤ x) :=
sorry

end NUMINAMATH_GPT_largest_undefined_x_value_l2374_237437


namespace NUMINAMATH_GPT_inequality_holds_if_b_greater_than_2_l2374_237484

variable (x : ℝ) (b : ℝ)

theorem inequality_holds_if_b_greater_than_2  :
  (b > 0) → (∃ x, |x-5| + |x-7| < b) ↔ (b > 2) := sorry

end NUMINAMATH_GPT_inequality_holds_if_b_greater_than_2_l2374_237484


namespace NUMINAMATH_GPT_sum_of_digits_of_d_l2374_237429

noncomputable section

def exchange_rate : ℚ := 8/5
def euros_after_spending (d : ℚ) : ℚ := exchange_rate * d - 80

theorem sum_of_digits_of_d {d : ℚ} (h : euros_after_spending d = d) : 
  d = 135 ∧ 1 + 3 + 5 = 9 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_digits_of_d_l2374_237429


namespace NUMINAMATH_GPT_smallest_integer_y_l2374_237465

theorem smallest_integer_y (y : ℤ) : (∃ (y : ℤ), (y / 4) + (3 / 7) > (4 / 7) ∧ ∀ (z : ℤ), z < y → (z / 4) + (3 / 7) ≤ (4 / 7)) := 
by
  sorry

end NUMINAMATH_GPT_smallest_integer_y_l2374_237465


namespace NUMINAMATH_GPT_misha_contributes_l2374_237412

noncomputable def misha_contribution (k l m : ℕ) : ℕ :=
  if h : k + l + m = 6 ∧ 2 * k ≤ l + m ∧ 2 * l ≤ k + m ∧ 2 * m ≤ k + l ∧ k ≤ 2 ∧ l ≤ 2 ∧ m ≤ 2 then
    2
  else
    0 -- This is a default value; the actual proof will check for exact solution.

theorem misha_contributes (k l m : ℕ) (h1 : k + l + m = 6)
    (h2 : 2 * k ≤ l + m) (h3 : 2 * l ≤ k + m) (h4 : 2 * m ≤ k + l)
    (h5 : k ≤ 2) (h6 : l ≤ 2) (h7 : m ≤ 2) : m = 2 := by
  sorry

end NUMINAMATH_GPT_misha_contributes_l2374_237412


namespace NUMINAMATH_GPT_entry_exit_ways_l2374_237434

theorem entry_exit_ways (n : ℕ) (h : n = 8) : n * (n - 1) = 56 :=
by {
  sorry
}

end NUMINAMATH_GPT_entry_exit_ways_l2374_237434


namespace NUMINAMATH_GPT_joes_total_weight_l2374_237453

theorem joes_total_weight (F S : ℕ) (h1 : F = 700) (h2 : 2 * F = S + 300) :
  F + S = 1800 :=
by
  sorry

end NUMINAMATH_GPT_joes_total_weight_l2374_237453


namespace NUMINAMATH_GPT_solution_of_inequality_l2374_237430

open Set

theorem solution_of_inequality (x : ℝ) :
  x^2 - 2 * x - 3 > 0 ↔ x < -1 ∨ x > 3 :=
by
  sorry

end NUMINAMATH_GPT_solution_of_inequality_l2374_237430


namespace NUMINAMATH_GPT_remaining_flour_needed_l2374_237445

-- Define the required total amount of flour
def total_flour : ℕ := 8

-- Define the amount of flour already added
def flour_added : ℕ := 2

-- Define the remaining amount of flour needed
def remaining_flour : ℕ := total_flour - flour_added

-- The theorem we need to prove
theorem remaining_flour_needed : remaining_flour = 6 := by
  sorry

end NUMINAMATH_GPT_remaining_flour_needed_l2374_237445


namespace NUMINAMATH_GPT_smallest_positive_period_pi_not_odd_at_theta_pi_div_4_axis_of_symmetry_at_pi_div_3_max_value_not_1_on_interval_l2374_237457

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 6)

-- Statement A: The smallest positive period of f(x) is π.
theorem smallest_positive_period_pi : 
  ∀ x : ℝ, f (x + Real.pi) = f x :=
by sorry

-- Statement B: If f(x + θ) is an odd function, then one possible value of θ is π/4.
theorem not_odd_at_theta_pi_div_4 : 
  ¬ (∀ x : ℝ, f (x + Real.pi / 4) = -f x) :=
by sorry

-- Statement C: A possible axis of symmetry for f(x) is the line x = π / 3.
theorem axis_of_symmetry_at_pi_div_3 :
  ∀ x : ℝ, f (Real.pi / 3 - x) = f (Real.pi / 3 + x) :=
by sorry

-- Statement D: The maximum value of f(x) on [0, π / 4] is 1.
theorem max_value_not_1_on_interval : 
  ¬ (∀ x ∈ Set.Icc 0 (Real.pi / 4), f x ≤ 1) :=
by sorry

end NUMINAMATH_GPT_smallest_positive_period_pi_not_odd_at_theta_pi_div_4_axis_of_symmetry_at_pi_div_3_max_value_not_1_on_interval_l2374_237457


namespace NUMINAMATH_GPT_greatest_x_for_4x_in_factorial_21_l2374_237491

-- Definition and theorem to state the problem mathematically
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem greatest_x_for_4x_in_factorial_21 : ∃ x : ℕ, (4^x ∣ factorial 21) ∧ ∀ y : ℕ, (4^y ∣ factorial 21) → y ≤ 9 :=
by
  sorry

end NUMINAMATH_GPT_greatest_x_for_4x_in_factorial_21_l2374_237491


namespace NUMINAMATH_GPT_radius_of_larger_circle_is_25_over_3_l2374_237493

noncomputable def radius_of_larger_circle (r : ℝ) : ℝ := (5 / 2) * r 

theorem radius_of_larger_circle_is_25_over_3
  (rAB rBD : ℝ)
  (h_ratio : 2 * rBD = 5 * rBD / 2)
  (h_ab : rAB = 8)
  (h_tangent : ∀ rBD, (5 * rBD / 2 - 8) ^ 2 = 64 + rBD ^ 2) :
  radius_of_larger_circle (10 / 3) = 25 / 3 :=
  by
  sorry

end NUMINAMATH_GPT_radius_of_larger_circle_is_25_over_3_l2374_237493


namespace NUMINAMATH_GPT_area_of_right_triangle_with_incircle_l2374_237481

theorem area_of_right_triangle_with_incircle (a b c r : ℝ) :
  (a = 6 + r) → 
  (b = 7 + r) → 
  (c = 13) → 
  (a^2 + b^2 = c^2) →
  (2 * r^2 + 26 * r = 84) →
  (area = 1/2 * ((6 + r) * (7 + r))) →
  area = 42 := 
by 
  sorry

end NUMINAMATH_GPT_area_of_right_triangle_with_incircle_l2374_237481


namespace NUMINAMATH_GPT_sum_of_squares_of_first_10_primes_l2374_237400

theorem sum_of_squares_of_first_10_primes :
  ((2^2) + (3^2) + (5^2) + (7^2) + (11^2) + (13^2) + (17^2) + (19^2) + (23^2) + (29^2)) = 2397 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_first_10_primes_l2374_237400


namespace NUMINAMATH_GPT_number_of_quadruplets_l2374_237472

variables (a b c : ℕ)

theorem number_of_quadruplets (h1 : 2 * a + 3 * b + 4 * c = 1200)
                             (h2 : b = 3 * c)
                             (h3 : a = 2 * b) :
  4 * c = 192 :=
by
  sorry

end NUMINAMATH_GPT_number_of_quadruplets_l2374_237472


namespace NUMINAMATH_GPT_remainder_when_101_divided_by_7_is_3_l2374_237497

theorem remainder_when_101_divided_by_7_is_3
    (A : ℤ)
    (h : 9 * A + 1 = 10 * A - 100) : A % 7 = 3 := by
  -- Mathematical steps are omitted as instructed
  sorry

end NUMINAMATH_GPT_remainder_when_101_divided_by_7_is_3_l2374_237497


namespace NUMINAMATH_GPT_total_hours_before_midterms_l2374_237404

-- Define the hours spent on each activity per week
def chess_hours_per_week : ℕ := 2
def drama_hours_per_week : ℕ := 8
def glee_hours_per_week : ℕ := 3

-- Sum up the total hours spent on extracurriculars per week
def total_hours_per_week : ℕ := chess_hours_per_week + drama_hours_per_week + glee_hours_per_week

-- Define semester information
def total_weeks_per_semester : ℕ := 12
def weeks_before_midterms : ℕ := total_weeks_per_semester / 2
def weeks_sick : ℕ := 2
def active_weeks_before_midterms : ℕ := weeks_before_midterms - weeks_sick

-- Define the theorem statement about total hours before midterms
theorem total_hours_before_midterms : total_hours_per_week * active_weeks_before_midterms = 52 := by
  -- We skip the actual proof here
  sorry

end NUMINAMATH_GPT_total_hours_before_midterms_l2374_237404


namespace NUMINAMATH_GPT_isosceles_triangle_congruent_side_length_l2374_237431

theorem isosceles_triangle_congruent_side_length 
  (base : ℝ) (area : ℝ) (a b c : ℝ) 
  (h1 : a = c)
  (h2 : a = base / 2)
  (h3 : (base * a) / 2 = area)
  : b = 5 * Real.sqrt 10 := 
by sorry

end NUMINAMATH_GPT_isosceles_triangle_congruent_side_length_l2374_237431


namespace NUMINAMATH_GPT_evaluate_polynomial_at_2_l2374_237413

theorem evaluate_polynomial_at_2 : (2^4 + 2^3 + 2^2 + 2 + 2) = 32 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_polynomial_at_2_l2374_237413


namespace NUMINAMATH_GPT_max_triangles_formed_l2374_237459

-- Define the triangles and their properties
structure EquilateralTriangle (α : Type) :=
(midpoint_segment : α) -- Each triangle has a segment connecting the midpoints of two sides

variables {α : Type} [OrderedSemiring α]

-- Define the condition of being mirrored horizontally
def areMirroredHorizontally (A B : EquilateralTriangle α) : Prop := 
  -- Placeholder for any formalization needed to specify mirrored horizontally
  sorry

-- Movement conditions and number of smaller triangles
def numberOfSmallerTrianglesAtMaxOverlap (A B : EquilateralTriangle α) (move_horizontally : α) : ℕ :=
  -- Placeholder function/modeling for counting triangles during movement
  sorry

-- Statement of our main theorem
theorem max_triangles_formed (A B : EquilateralTriangle α) (move_horizontally : α) 
  (h_mirrored : areMirroredHorizontally A B) :
  numberOfSmallerTrianglesAtMaxOverlap A B move_horizontally = 11 :=
sorry

end NUMINAMATH_GPT_max_triangles_formed_l2374_237459


namespace NUMINAMATH_GPT_car_speed_after_modifications_l2374_237468

theorem car_speed_after_modifications (s : ℕ) (p : ℝ) (w : ℕ) :
  s = 150 →
  p = 0.30 →
  w = 10 →
  s + (p * s) + w = 205 := 
by
  intros h₁ h₂ h₃
  rw [h₁, h₂, h₃]
  norm_num
  done

end NUMINAMATH_GPT_car_speed_after_modifications_l2374_237468


namespace NUMINAMATH_GPT_salary_of_A_l2374_237422

theorem salary_of_A (x y : ℝ) (h₁ : x + y = 4000) (h₂ : 0.05 * x = 0.15 * y) : x = 3000 :=
by {
    sorry
}

end NUMINAMATH_GPT_salary_of_A_l2374_237422


namespace NUMINAMATH_GPT_total_crosswalk_lines_l2374_237424

theorem total_crosswalk_lines 
  (intersections : ℕ) 
  (crosswalks_per_intersection : ℕ) 
  (lines_per_crosswalk : ℕ)
  (h1 : intersections = 10)
  (h2 : crosswalks_per_intersection = 8)
  (h3 : lines_per_crosswalk = 30) :
  intersections * crosswalks_per_intersection * lines_per_crosswalk = 2400 := 
by {
  sorry
}

end NUMINAMATH_GPT_total_crosswalk_lines_l2374_237424


namespace NUMINAMATH_GPT_greatest_possible_difference_in_rectangles_area_l2374_237441

theorem greatest_possible_difference_in_rectangles_area :
  ∃ (l1 w1 l2 w2 l3 w3 : ℤ),
    2 * l1 + 2 * w1 = 148 ∧
    2 * l2 + 2 * w2 = 150 ∧
    2 * l3 + 2 * w3 = 152 ∧
    (∃ (A1 A2 A3 : ℤ),
      A1 = l1 * w1 ∧
      A2 = l2 * w2 ∧
      A3 = l3 * w3 ∧
      (max (abs (A1 - A2)) (max (abs (A1 - A3)) (abs (A2 - A3))) = 1372)) :=
by
  sorry

end NUMINAMATH_GPT_greatest_possible_difference_in_rectangles_area_l2374_237441


namespace NUMINAMATH_GPT_thursday_to_wednesday_ratio_l2374_237421

-- Let M, T, W, Th be the number of messages sent on Monday, Tuesday, Wednesday, and Thursday respectively.
variables (M T W Th : ℕ)

-- Conditions are given as follows
axiom hM : M = 300
axiom hT : T = 200
axiom hW : W = T + 300
axiom hSum : M + T + W + Th = 2000

-- Define the function to compute the ratio
def ratio (a b : ℕ) : ℚ := a / b

-- The target is to prove that the ratio of the messages sent on Thursday to those sent on Wednesday is 2 / 1
theorem thursday_to_wednesday_ratio : ratio Th W = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_thursday_to_wednesday_ratio_l2374_237421


namespace NUMINAMATH_GPT_MissAisha_height_l2374_237427

theorem MissAisha_height (H : ℝ)
  (legs_length : ℝ := H / 3)
  (head_length : ℝ := H / 4)
  (rest_body_length : ℝ := 25) :
  H = 60 :=
by sorry

end NUMINAMATH_GPT_MissAisha_height_l2374_237427


namespace NUMINAMATH_GPT_evaluate_expression_l2374_237410

theorem evaluate_expression :
  (4 * 10^2011 - 1) / (4 * (3 * (10^2011 - 1) / 9) + 1) = 3 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2374_237410


namespace NUMINAMATH_GPT_gen_formula_arithmetic_seq_sum_maximizes_at_5_l2374_237470

def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n : ℕ, a n = a 1 + (n - 1) * d

variables (an : ℕ → ℤ) (Sn : ℕ → ℤ)
variable (d : ℤ)

theorem gen_formula_arithmetic_seq (h1 : an 3 = 5) (h2 : an 10 = -9) :
  ∀ n, an n = 11 - 2 * n :=
sorry

theorem sum_maximizes_at_5 (h_seq : ∀ n, an n = 11 - 2 * n) :
  ∀ n, Sn n = (n * 10 - n^2) → (∃ n, ∀ k, Sn n ≥ Sn k) :=
sorry

end NUMINAMATH_GPT_gen_formula_arithmetic_seq_sum_maximizes_at_5_l2374_237470


namespace NUMINAMATH_GPT_three_pow_n_minus_one_not_divisible_by_two_pow_n_minus_one_l2374_237449

theorem three_pow_n_minus_one_not_divisible_by_two_pow_n_minus_one (n : ℕ) (h : n > 1) : ¬(2^n - 1) ∣ (3^n - 1) :=
sorry

end NUMINAMATH_GPT_three_pow_n_minus_one_not_divisible_by_two_pow_n_minus_one_l2374_237449


namespace NUMINAMATH_GPT_positive_root_of_real_root_l2374_237406

theorem positive_root_of_real_root (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : b^2 - 4*a*c ≥ 0) (h2 : c^2 - 4*b*a ≥ 0) (h3 : a^2 - 4*c*b ≥ 0) : 
  ∀ (p q r : ℝ), (p = a ∧ q = b ∧ r = c) ∨ (p = b ∧ q = c ∧ r = a) ∨ (p = c ∧ q = a ∧ r = b) →
  (∃ x : ℝ, x > 0 ∧ p*x^2 + q*x + r = 0) :=
by 
  sorry

end NUMINAMATH_GPT_positive_root_of_real_root_l2374_237406


namespace NUMINAMATH_GPT_evaluate_expression_l2374_237488

noncomputable def g (x : ℝ) : ℝ := x^3 + 3*x + 2*Real.sqrt x

theorem evaluate_expression : 
  3 * g 3 - 2 * g 9 = -1416 + 6 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2374_237488


namespace NUMINAMATH_GPT_find_f_lg_lg_2_l2374_237476

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * (Real.sin x) + 4

theorem find_f_lg_lg_2 (a b : ℝ) (m : ℝ) 
  (h1 : f a b (Real.logb 10 2) = 5) 
  (h2 : m = Real.logb 10 2) : 
  f a b (Real.logb 2 m) = 3 :=
sorry

end NUMINAMATH_GPT_find_f_lg_lg_2_l2374_237476


namespace NUMINAMATH_GPT_expand_and_simplify_l2374_237414

theorem expand_and_simplify (x : ℝ) : 6 * (x - 3) * (x + 10) = 6 * x^2 + 42 * x - 180 :=
by
  sorry

end NUMINAMATH_GPT_expand_and_simplify_l2374_237414


namespace NUMINAMATH_GPT_find_values_of_a_and_b_l2374_237486

theorem find_values_of_a_and_b (a b x y : ℝ) (h1: 0 < a) (h2: 0 < b) (h3: 0 < x) (h4: 0 < y) 
  (h5 : a + b = 10) (h6 : a / x + b / y = 1) (h7 : x + y = 18) : 
  (a = 2 ∧ b = 8) ∨ (a = 8 ∧ b = 2) := 
sorry

end NUMINAMATH_GPT_find_values_of_a_and_b_l2374_237486


namespace NUMINAMATH_GPT_rickshaw_distance_l2374_237469

theorem rickshaw_distance :
  ∃ (distance : ℝ), 
  (13.5 + (distance - 1) * (2.50 / (1 / 3))) = 103.5 ∧ distance = 13 :=
by
  sorry

end NUMINAMATH_GPT_rickshaw_distance_l2374_237469


namespace NUMINAMATH_GPT_dan_stationery_spent_l2374_237416

def total_spent : ℕ := 32
def backpack_cost : ℕ := 15
def notebook_cost : ℕ := 3
def number_of_notebooks : ℕ := 5
def stationery_cost_each : ℕ := 1

theorem dan_stationery_spent : 
  (total_spent - (backpack_cost + notebook_cost * number_of_notebooks)) = 2 :=
by
  sorry

end NUMINAMATH_GPT_dan_stationery_spent_l2374_237416


namespace NUMINAMATH_GPT_find_X_plus_Y_in_base_8_l2374_237435

theorem find_X_plus_Y_in_base_8 (X Y : ℕ) (h1 : 3 * 8^2 + X * 8 + Y + 5 * 8 + 2 = 4 * 8^2 + X * 8 + 3) : X + Y = 1 :=
sorry

end NUMINAMATH_GPT_find_X_plus_Y_in_base_8_l2374_237435


namespace NUMINAMATH_GPT_simplify_expression_l2374_237411

theorem simplify_expression (w : ℝ) : 3 * w + 6 * w - 9 * w + 12 * w - 15 * w + 21 = -3 * w + 21 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2374_237411


namespace NUMINAMATH_GPT_max_possible_score_l2374_237474

theorem max_possible_score (s : ℝ) (h : 80 = s * 2) : s * 5 ≥ 100 :=
by
  -- sorry placeholder for the proof
  sorry

end NUMINAMATH_GPT_max_possible_score_l2374_237474


namespace NUMINAMATH_GPT_time_to_reach_madison_l2374_237415

-- Definitions based on the conditions
def map_distance : ℝ := 5 -- inches
def average_speed : ℝ := 60 -- miles per hour
def map_scale : ℝ := 0.016666666666666666 -- inches per mile

-- The time taken by Pete to arrive in Madison
noncomputable def time_to_madison := map_distance / map_scale / average_speed

-- The theorem to prove
theorem time_to_reach_madison : time_to_madison = 5 := 
by
  sorry

end NUMINAMATH_GPT_time_to_reach_madison_l2374_237415


namespace NUMINAMATH_GPT_postcards_per_day_l2374_237461

variable (income_per_card total_income days : ℕ)
variable (x : ℕ)

theorem postcards_per_day
  (h1 : income_per_card = 5)
  (h2 : total_income = 900)
  (h3 : days = 6)
  (h4 : total_income = income_per_card * x * days) :
  x = 30 :=
by
  rw [h1, h2, h3] at h4
  linarith

end NUMINAMATH_GPT_postcards_per_day_l2374_237461


namespace NUMINAMATH_GPT_volume_of_square_pyramid_l2374_237440

theorem volume_of_square_pyramid (a r : ℝ) : 
  a > 0 → r > 0 → volume = (1 / 3) * a^2 * r :=
by 
    sorry

end NUMINAMATH_GPT_volume_of_square_pyramid_l2374_237440


namespace NUMINAMATH_GPT_complement_of_A_in_U_is_2_l2374_237480

open Set

def U : Set ℕ := { x | x ≥ 2 }
def A : Set ℕ := { x | x^2 ≥ 5 }

theorem complement_of_A_in_U_is_2 : compl A ∩ U = {2} :=
by
  sorry

end NUMINAMATH_GPT_complement_of_A_in_U_is_2_l2374_237480


namespace NUMINAMATH_GPT_height_of_right_triangle_on_parabola_equals_one_l2374_237452

theorem height_of_right_triangle_on_parabola_equals_one 
    (x0 x1 x2 : ℝ) 
    (h0 : x0 ≠ x1)
    (h1 : x0 ≠ x2) 
    (h2 : x1 ≠ x2) 
    (h3 : x0^2 = x1^2) 
    (h4 : x0^2 < x2^2):
    x2^2 - x0^2 = 1 := by
  sorry

end NUMINAMATH_GPT_height_of_right_triangle_on_parabola_equals_one_l2374_237452


namespace NUMINAMATH_GPT_uruguayan_goals_conceded_l2374_237401

theorem uruguayan_goals_conceded (x : ℕ) (h : 14 = 9 + x) : x = 5 := by
  sorry

end NUMINAMATH_GPT_uruguayan_goals_conceded_l2374_237401


namespace NUMINAMATH_GPT_exponent_equality_l2374_237417

theorem exponent_equality (n : ℕ) : 
    5^n = 5 * (5^2)^2 * (5^3)^3 → n = 14 := by
    sorry

end NUMINAMATH_GPT_exponent_equality_l2374_237417


namespace NUMINAMATH_GPT_solve_equation_l2374_237475

noncomputable def f (x : ℝ) : ℝ :=
  2 * x + 1 + Real.arctan x * Real.sqrt (x^2 + 1)

theorem solve_equation : ∃ x : ℝ, f x + f (x + 1) = 0 ∧ x = -1/2 :=
  by
    use -1/2
    simp [f]
    sorry

end NUMINAMATH_GPT_solve_equation_l2374_237475


namespace NUMINAMATH_GPT_scientific_notation_l2374_237494

def z := 10374 * 10^9

theorem scientific_notation (a : ℝ) (n : ℤ) (h₁ : 1 ≤ |a|) (h₂ : |a| < 10) (h₃ : a * 10^n = z) : a = 1.04 ∧ n = 13 := sorry

end NUMINAMATH_GPT_scientific_notation_l2374_237494


namespace NUMINAMATH_GPT_total_movies_attended_l2374_237471

-- Defining the conditions for Timothy's movie attendance
def Timothy_2009 := 24
def Timothy_2010 := Timothy_2009 + 7

-- Defining the conditions for Theresa's movie attendance
def Theresa_2009 := Timothy_2009 / 2
def Theresa_2010 := Timothy_2010 * 2

-- Prove that the total number of movies Timothy and Theresa went to in both years is 129
theorem total_movies_attended :
  (Timothy_2009 + Timothy_2010 + Theresa_2009 + Theresa_2010) = 129 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_total_movies_attended_l2374_237471


namespace NUMINAMATH_GPT_find_f_condition_l2374_237432

theorem find_f_condition {f : ℂ → ℂ} (h : ∀ z : ℂ, f z + z * f (1 - z) = 1 + z) :
  ∀ z : ℂ, f z = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_f_condition_l2374_237432


namespace NUMINAMATH_GPT_abc_sum_l2374_237408

theorem abc_sum : ∃ a b c : ℤ, 
  (∀ x : ℤ, x^2 + 13 * x + 30 = (x + a) * (x + b)) ∧ 
  (∀ x : ℤ, x^2 + 5 * x - 50 = (x + b) * (x - c)) ∧
  a + b + c = 18 := by
  sorry

end NUMINAMATH_GPT_abc_sum_l2374_237408


namespace NUMINAMATH_GPT_song_distribution_l2374_237425

-- Let us define the necessary conditions and the result as a Lean statement.

theorem song_distribution :
    ∃ (AB BC CA A B C N : Finset ℕ),
    -- Six different songs.
    (AB ∪ BC ∪ CA ∪ A ∪ B ∪ C ∪ N) = {1, 2, 3, 4, 5, 6} ∧
    -- No song is liked by all three.
    (∀ song, ¬(song ∈ AB ∩ BC ∩ CA)) ∧
    -- Each girl dislikes at least one song.
    (N ≠ ∅) ∧
    -- For each pair of girls, at least one song liked by those two but disliked by the third.
    (AB ≠ ∅ ∧ BC ≠ ∅ ∧ CA ≠ ∅) ∧
    -- The total number of ways this can be done is 735.
    True := sorry

end NUMINAMATH_GPT_song_distribution_l2374_237425


namespace NUMINAMATH_GPT_range_of_x_l2374_237485

noncomputable def A (x : ℝ) : ℤ := Int.ceil x

theorem range_of_x (x : ℝ) (h₁ : 0 < x) (h₂ : A (2 * x * A x) = 5) : 1 < x ∧ x ≤ 5 / 4 := 
sorry

end NUMINAMATH_GPT_range_of_x_l2374_237485


namespace NUMINAMATH_GPT_cyclic_quadrilaterals_count_l2374_237490

noncomputable def num_cyclic_quadrilaterals (n : ℕ) : ℕ :=
  if n = 32 then 568 else 0 -- encapsulating the problem's answer

theorem cyclic_quadrilaterals_count :
  num_cyclic_quadrilaterals 32 = 568 :=
sorry

end NUMINAMATH_GPT_cyclic_quadrilaterals_count_l2374_237490


namespace NUMINAMATH_GPT_num_ways_product_72_l2374_237482

def num_ways_product (n : ℕ) : ℕ := sorry  -- Definition for D(n), the number of ways to write n as a product of integers greater than 1

def example_integer := 72  -- Given integer n

theorem num_ways_product_72 : num_ways_product example_integer = 67 := by 
  sorry

end NUMINAMATH_GPT_num_ways_product_72_l2374_237482


namespace NUMINAMATH_GPT_average_of_last_six_l2374_237460

theorem average_of_last_six (avg_13 : ℕ → ℝ) (avg_first_6 : ℕ → ℝ) (middle_number : ℕ → ℝ) :
  (∀ n, avg_13 n = 9) →
  (∀ n, n ≤ 6 → avg_first_6 n = 5) →
  (middle_number 7 = 45) →
  ∃ (A : ℝ), (∀ n, n > 6 → n < 13 → avg_13 n = A) ∧ A = 7 :=
by
  sorry

end NUMINAMATH_GPT_average_of_last_six_l2374_237460


namespace NUMINAMATH_GPT_strictly_increasing_interval_l2374_237464

noncomputable def f (x : ℝ) : ℝ := x - x * Real.log x

theorem strictly_increasing_interval :
  ∀ x : ℝ, 0 < x ∧ x < 1 → f x = x - x * Real.log x → ∀ y : ℝ, (0 < y ∧ y < 1 ∧ y > x) → f y > f x :=
sorry

end NUMINAMATH_GPT_strictly_increasing_interval_l2374_237464


namespace NUMINAMATH_GPT_solve_quadratic_equation_l2374_237442

theorem solve_quadratic_equation : 
  ∀ x : ℝ, 2 * x^2 = 4 ↔ x = Real.sqrt 2 ∨ x = -Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_GPT_solve_quadratic_equation_l2374_237442


namespace NUMINAMATH_GPT_greatest_root_of_gx_l2374_237466

theorem greatest_root_of_gx :
  ∃ x : ℝ, (10 * x^4 - 16 * x^2 + 3 = 0) ∧ (∀ y : ℝ, (10 * y^4 - 16 * y^2 + 3 = 0) → x ≥ y) ∧ x = Real.sqrt (3 / 5) := 
sorry

end NUMINAMATH_GPT_greatest_root_of_gx_l2374_237466


namespace NUMINAMATH_GPT_xy_nonzero_implies_iff_l2374_237443

variable {x y : ℝ}

theorem xy_nonzero_implies_iff (h : x * y ≠ 0) : (x + y = 0) ↔ (x / y + y / x = -2) :=
sorry

end NUMINAMATH_GPT_xy_nonzero_implies_iff_l2374_237443


namespace NUMINAMATH_GPT_evaluate_fraction_expression_l2374_237439

theorem evaluate_fraction_expression :
  ( (1 / 5 - 1 / 6) / (1 / 3 - 1 / 4) ) = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_fraction_expression_l2374_237439


namespace NUMINAMATH_GPT_sum_of_fourth_powers_l2374_237448

theorem sum_of_fourth_powers (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 4) : 
  a^4 + b^4 + c^4 = 8 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_fourth_powers_l2374_237448
