import Mathlib

namespace simple_interest_double_l1173_117363

theorem simple_interest_double (P : ℝ) (r : ℝ) (t : ℝ) (A : ℝ)
  (h1 : t = 50)
  (h2 : A = 2 * P) 
  (h3 : A - P = P * r * t / 100) :
  r = 2 :=
by
  -- Proof is omitted
  sorry

end simple_interest_double_l1173_117363


namespace automobile_travel_distance_l1173_117373

theorem automobile_travel_distance (b s : ℝ) (h1 : s > 0) :
  let rate := (b / 8) / s  -- rate in meters per second
  let rate_km_per_min := rate * (1 / 1000) * 60  -- convert to kilometers per minute
  let time := 5  -- time in minutes
  rate_km_per_min * time = 3 * b / 80 / s := sorry

end automobile_travel_distance_l1173_117373


namespace Jake_peach_count_l1173_117366

theorem Jake_peach_count (Steven_peaches : ℕ) (Jake_peach_difference : ℕ) (h1 : Steven_peaches = 19) (h2 : Jake_peach_difference = 12) : 
  Steven_peaches - Jake_peach_difference = 7 :=
by
  sorry

end Jake_peach_count_l1173_117366


namespace melies_meat_purchase_l1173_117357

-- Define the relevant variables and conditions
variable (initial_amount : ℕ) (amount_left : ℕ) (cost_per_kg : ℕ)

-- State the main theorem we want to prove
theorem melies_meat_purchase (h1 : initial_amount = 180) (h2 : amount_left = 16) (h3 : cost_per_kg = 82) :
  (initial_amount - amount_left) / cost_per_kg = 2 := by
  sorry

end melies_meat_purchase_l1173_117357


namespace charlie_has_32_cards_l1173_117367

variable (Chris_cards Charlie_cards : ℕ)

def chris_has_18_cards : Chris_cards = 18 := sorry
def chris_has_14_fewer_cards_than_charlie : Chris_cards + 14 = Charlie_cards := sorry

theorem charlie_has_32_cards (h18 : Chris_cards = 18) (h14 : Chris_cards + 14 = Charlie_cards) : Charlie_cards = 32 := 
sorry

end charlie_has_32_cards_l1173_117367


namespace arithmetic_progression_terms_even_l1173_117340

variable (a d : ℝ) (n : ℕ)

open Real

theorem arithmetic_progression_terms_even {n : ℕ} (hn_even : n % 2 = 0)
  (h_sum_odd : (n / 2 : ℝ) * (2 * a + (n - 2) * d) = 32)
  (h_sum_even : (n / 2 : ℝ) * (2 * a + 2 * d + (n - 2) * d) = 40)
  (h_last_exceeds_first : (a + (n - 1) * d) - a = 8) : n = 16 :=
sorry

end arithmetic_progression_terms_even_l1173_117340


namespace oranges_to_put_back_l1173_117322

variables (A O x : ℕ)

theorem oranges_to_put_back
    (h1 : 40 * A + 60 * O = 560)
    (h2 : A + O = 10)
    (h3 : (40 * A + 60 * (O - x)) / (10 - x) = 50) : x = 6 := 
sorry

end oranges_to_put_back_l1173_117322


namespace length_CD_l1173_117348

theorem length_CD (AB AC BD CD : ℝ) (hAB : AB = 2) (hAC : AC = 5) (hBD : BD = 6) :
    CD = 3 :=
by
  sorry

end length_CD_l1173_117348


namespace hypotenuse_length_l1173_117300

theorem hypotenuse_length (a b c : ℕ) (h1 : a = 12) (h2 : b = 5) (h3 : c^2 = a^2 + b^2) : c = 13 := by
  sorry

end hypotenuse_length_l1173_117300


namespace hyperbola_center_l1173_117315

theorem hyperbola_center (x1 y1 x2 y2 : ℝ) (h1 : x1 = 6) (h2 : y1 = 3) (h3 : x2 = 10) (h4 : y2 = 7) :
  ((x1 + x2) / 2, (y1 + y2) / 2) = (8, 5) :=
by
  rw [h1, h2, h3, h4]
  simp
  -- Proof steps demonstrating the calculation
  -- simplify the arithmetic expressions
  sorry

end hyperbola_center_l1173_117315


namespace line_does_not_pass_through_fourth_quadrant_l1173_117335

-- Definitions of conditions
variables {a b c x y : ℝ}

-- The mathematical statement to be proven
theorem line_does_not_pass_through_fourth_quadrant
  (h1 : a * b < 0) (h2 : b * c < 0) :
  ¬ (∃ x y, x > 0 ∧ y < 0 ∧ a * x + b * y + c = 0) :=
sorry

end line_does_not_pass_through_fourth_quadrant_l1173_117335


namespace slope_angle_vertical_line_l1173_117364

theorem slope_angle_vertical_line : 
  ∀ α : ℝ, (∀ x y : ℝ, x = 1 → y = α) → α = Real.pi / 2 := 
by 
  sorry

end slope_angle_vertical_line_l1173_117364


namespace y_increase_when_x_increases_by_9_units_l1173_117324

-- Given condition as a definition: when x increases by 3 units, y increases by 7 units.
def x_increases_y_increases (x_increase y_increase : ℕ) : Prop := 
  (x_increase = 3) → (y_increase = 7)

-- Stating the problem: when x increases by 9 units, y increases by how many units?
theorem y_increase_when_x_increases_by_9_units : 
  ∀ (x_increase y_increase : ℕ), x_increases_y_increases x_increase y_increase → ((x_increase * 3 = 9) → (y_increase * 3 = 21)) :=
by
  intros x_increase y_increase cond h
  sorry

end y_increase_when_x_increases_by_9_units_l1173_117324


namespace min_value_expression_l1173_117396

theorem min_value_expression 
  (a b c : ℝ)
  (h1 : a + b + c = -1)
  (h2 : a * b * c ≤ -3) : 
  (ab + 1) / (a + b) + (bc + 1) / (b + c) + (ca + 1) / (c + a) ≥ 3 :=
sorry

end min_value_expression_l1173_117396


namespace black_to_white_ratio_l1173_117332

theorem black_to_white_ratio (initial_black initial_white new_black new_white : ℕ) 
  (h1 : initial_black = 7) (h2 : initial_white = 18)
  (h3 : new_black = 31) (h4 : new_white = 18) :
  (new_black : ℚ) / new_white = 31 / 18 :=
by
  sorry

end black_to_white_ratio_l1173_117332


namespace gas_cost_problem_l1173_117312

theorem gas_cost_problem (x : ℝ) (h : x / 4 - 15 = x / 7) : x = 140 :=
sorry

end gas_cost_problem_l1173_117312


namespace multiple_of_5_l1173_117308

theorem multiple_of_5 (a : ℤ) (h : ¬ (5 ∣ a)) : 5 ∣ (a^12 - 1) :=
by
  sorry

end multiple_of_5_l1173_117308


namespace rectangle_width_of_square_l1173_117375

theorem rectangle_width_of_square (side_length_square : ℝ) (length_rectangle : ℝ) (width_rectangle : ℝ)
  (h1 : side_length_square = 3) (h2 : length_rectangle = 3)
  (h3 : (side_length_square ^ 2) = length_rectangle * width_rectangle) : width_rectangle = 3 :=
by
  sorry

end rectangle_width_of_square_l1173_117375


namespace exponent_multiplication_l1173_117307

theorem exponent_multiplication (m n : ℕ) (h : m + n = 3) : 2^m * 2^n = 8 := 
by
  sorry

end exponent_multiplication_l1173_117307


namespace problem_I_problem_II_l1173_117313

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := |x + (4 / x) - m| + m

-- Proof problem (I): When m = 0, find the minimum value of the function f(x).
theorem problem_I : ∀ x : ℝ, (f x 0) ≥ 4 := by
  sorry

-- Proof problem (II): If the function f(x) ≤ 5 for all x ∈ [1, 4], find the range of m.
theorem problem_II (m : ℝ) : (∀ x : ℝ, 1 ≤ x → x ≤ 4 → f x m ≤ 5) ↔ m ≤ 9 / 2 := by
  sorry

end problem_I_problem_II_l1173_117313


namespace intercepts_equal_l1173_117365

theorem intercepts_equal (a : ℝ) (ha : (a ≠ 0) ∧ (a ≠ 2)) : 
  (a = 1 ∨ a = 2) ↔ (a = 1 ∨ a = 2) := 
by 
  sorry


end intercepts_equal_l1173_117365


namespace moles_of_ammonia_combined_l1173_117356

theorem moles_of_ammonia_combined (n_CO2 n_Urea n_NH3 : ℕ) (h1 : n_CO2 = 1) (h2 : n_Urea = 1) (h3 : n_Urea = n_CO2)
  (h4 : n_Urea = 2 * n_NH3): n_NH3 = 2 := 
by
  sorry

end moles_of_ammonia_combined_l1173_117356


namespace average_weight_of_remaining_students_l1173_117370

theorem average_weight_of_remaining_students
  (M F M' F' : ℝ) (A A' : ℝ)
  (h1 : M + F = 60 * A)
  (h2 : M' + F' = 59 * A')
  (h3 : A' = A + 0.2)
  (h4 : M' = M - 45):
  A' = 57 :=
by
  sorry

end average_weight_of_remaining_students_l1173_117370


namespace dice_digit_distribution_l1173_117351

theorem dice_digit_distribution : ∃ n : ℕ, n = 10 ∧ 
  (∀ (d1 d2 : Finset ℕ), d1.card = 6 ∧ d2.card = 6 ∧
  (0 ∈ d1) ∧ (1 ∈ d1) ∧ (2 ∈ d1) ∧ 
  (0 ∈ d2) ∧ (1 ∈ d2) ∧ (2 ∈ d2) ∧
  ({3, 4, 5, 6, 7, 8} ⊆ (d1 ∪ d2)) ∧ 
  (∀ i, i ∈ d1 ∪ d2 → i ∈ (Finset.range 10))) := 
  sorry

end dice_digit_distribution_l1173_117351


namespace min_value_three_l1173_117314

noncomputable def min_value_expression (x y z : ℝ) : ℝ :=
  (1 / ((1 - x) * (1 - y) * (1 - z))) +
  (1 / ((1 + x) * (1 + y) * (1 + z))) +
  (1 / ((1 - x^2) * (1 - y^2) * (1 - z^2)))

theorem min_value_three (x y z : ℝ) (hx : 0 ≤ x) (hx1 : x ≤ 1) (hy : 0 ≤ y) (hy1 : y ≤ 1) (hz : 0 ≤ z) (hz1 : z ≤ 1) :
  min_value_expression x y z = 3 :=
by
  sorry

end min_value_three_l1173_117314


namespace arithmetic_sequence_binomial_l1173_117321

theorem arithmetic_sequence_binomial {n k u : ℕ} (h₁ : u ≥ 3)
    (h₂ : n = u^2 - 2)
    (h₃ : k = Nat.choose u 2 - 1 ∨ k = Nat.choose (u + 1) 2 - 1)
    : (Nat.choose n (k - 1)) - 2 * (Nat.choose n k) + (Nat.choose n (k + 1)) = 0 :=
by
  sorry

end arithmetic_sequence_binomial_l1173_117321


namespace reflection_y_axis_matrix_correct_l1173_117345

def reflect_y_axis_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![(-1 : ℝ), (0 : ℝ)], ![(0 : ℝ), (1 : ℝ)]]

theorem reflection_y_axis_matrix_correct :
  reflect_y_axis_matrix = ![![(-1 : ℝ), (0 : ℝ)], ![(0 : ℝ), (1 : ℝ)]] :=
by
  sorry

end reflection_y_axis_matrix_correct_l1173_117345


namespace distinct_ordered_pairs_solution_l1173_117399

theorem distinct_ordered_pairs_solution :
  (∃ n : ℕ, ∀ x y : ℕ, (x > 0 ∧ y > 0 ∧ x^4 * y^4 - 24 * x^2 * y^2 + 35 = 0) ↔ n = 1) :=
sorry

end distinct_ordered_pairs_solution_l1173_117399


namespace solve_real_equation_l1173_117362

theorem solve_real_equation (x : ℝ) :
  x^2 * (x + 1)^2 + x^2 = 3 * (x + 1)^2 ↔ x = (1 + Real.sqrt 5) / 2 ∨ x = (1 - Real.sqrt 5) / 2 :=
by sorry

end solve_real_equation_l1173_117362


namespace acute_triangle_and_angle_relations_l1173_117385

theorem acute_triangle_and_angle_relations (a b c u v w : ℝ) (A B C : ℝ)
  (h₁ : a^2 = u * (v + w - u))
  (h₂ : b^2 = v * (w + u - v))
  (h₃ : c^2 = w * (u + v - w)) :
  (a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2) ∧
  (∀ U V W : ℝ, U = 180 - 2 * A ∧ V = 180 - 2 * B ∧ W = 180 - 2 * C) :=
by sorry

end acute_triangle_and_angle_relations_l1173_117385


namespace shanghai_masters_total_matches_l1173_117368

theorem shanghai_masters_total_matches : 
  let players := 8
  let groups := 2
  let players_per_group := 4
  let round_robin_matches_per_group := (players_per_group * (players_per_group - 1)) / 2
  let round_robin_total_matches := round_robin_matches_per_group * groups
  let elimination_matches := 2 * (groups - 1)  -- semi-final matches
  let final_matches := 2  -- one final and one third-place match
  round_robin_total_matches + elimination_matches + final_matches = 16 :=
by
  sorry

end shanghai_masters_total_matches_l1173_117368


namespace function_identity_l1173_117305

theorem function_identity (f : ℕ → ℕ) 
  (h1 : f 2 = 2)
  (h2 : ∀ m n : ℕ, f (m * n) = f m * f n)
  (h3 : ∀ n : ℕ, f (n + 1) > f n) : 
  ∀ n : ℕ, f n = n :=
sorry

end function_identity_l1173_117305


namespace percentage_decrease_l1173_117336

theorem percentage_decrease (a b p : ℝ) (a_pos : 0 < a) (b_pos : 0 < b)
    (h_ratio : a / b = 4 / 5) 
    (h_x : ∃ x, x = a * 1.25)
    (h_m : ∃ m, m = b * (1 - p / 100))
    (h_mx : ∃ m x, (m / x = 0.2)) :
        (p = 80) :=
by
  sorry

end percentage_decrease_l1173_117336


namespace arithmetic_seq_sum_l1173_117303

theorem arithmetic_seq_sum {a_n : ℕ → ℤ} {d : ℤ} (S_n : ℕ → ℤ) :
  (∀ n : ℕ, S_n n = -(n * n)) →
  (∃ d, d = -2 ∧ ∀ n, a_n n = -2 * n + 1) :=
by
  -- Assuming that S_n is given as per the condition of the problem
  sorry

end arithmetic_seq_sum_l1173_117303


namespace number_of_black_bears_l1173_117350

-- Definitions of conditions
def brown_bears := 15
def white_bears := 24
def total_bears := 66

-- The proof statement
theorem number_of_black_bears : (total_bears - (brown_bears + white_bears) = 27) := by
  sorry

end number_of_black_bears_l1173_117350


namespace triangle_angles_l1173_117320

theorem triangle_angles (r_a r_b r_c R : ℝ)
    (h1 : r_a + r_b = 3 * R)
    (h2 : r_b + r_c = 2 * R) :
    ∃ (A B C : ℝ), A = 30 ∧ B = 60 ∧ C = 90 :=
sorry

end triangle_angles_l1173_117320


namespace clare_milk_cartons_l1173_117339

def money_given := 47
def cost_per_loaf := 2
def loaves_bought := 4
def cost_per_milk := 2
def money_left := 35

theorem clare_milk_cartons : (money_given - money_left - loaves_bought * cost_per_loaf) / cost_per_milk = 2 :=
by
  sorry

end clare_milk_cartons_l1173_117339


namespace subtraction_base_8_correct_l1173_117331

def sub_in_base_8 (a b : Nat) : Nat := sorry

theorem subtraction_base_8_correct : sub_in_base_8 (sub_in_base_8 0o123 0o51) 0o15 = 0o25 :=
sorry

end subtraction_base_8_correct_l1173_117331


namespace function_characterization_l1173_117391

noncomputable def f : ℝ → ℝ := sorry

theorem function_characterization :
  (∀ x y : ℝ, 0 ≤ x ∧ 0 ≤ y → f (x * f y) * f y = f (x + y)) ∧
  (f 2 = 0) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x < 2 → f x ≠ 0) →
  (∀ x : ℝ, 0 ≤ x → f x = if x < 2 then 2 / (2 - x) else 0) := sorry

end function_characterization_l1173_117391


namespace factorize_x4_y4_l1173_117317

theorem factorize_x4_y4 (x y : ℝ) : x^4 - y^4 = (x^2 + y^2) * (x^2 - y^2) :=
by
  sorry

end factorize_x4_y4_l1173_117317


namespace general_formula_for_an_l1173_117360

-- Definitions for the first few terms of the sequence
def a1 : ℚ := 1 / 7
def a2 : ℚ := 3 / 77
def a3 : ℚ := 5 / 777

-- The sequence definition as per the identified pattern
def a_n (n : ℕ) : ℚ := (18 * n - 9) / (7 * (10^n - 1))

-- The theorem to establish that the sequence definition for general n holds given the initial terms 
theorem general_formula_for_an {n : ℕ} :
  (n = 1 → a_n n = a1) ∧
  (n = 2 → a_n n = a2) ∧ 
  (n = 3 → a_n n = a3) ∧ 
  (∀ n > 3, a_n n = (18 * n - 9) / (7 * (10^n - 1))) := 
by
  sorry

end general_formula_for_an_l1173_117360


namespace total_money_spent_l1173_117347

noncomputable def total_expenditure (A : ℝ) : ℝ :=
  let person1_8_expenditure := 8 * 12
  let person9_expenditure := A + 8
  person1_8_expenditure + person9_expenditure

theorem total_money_spent :
  (∃ A : ℝ, total_expenditure A = 9 * A ∧ A = 13) →
  total_expenditure 13 = 117 :=
by
  intro h
  sorry

end total_money_spent_l1173_117347


namespace plane_hit_probability_l1173_117325

theorem plane_hit_probability :
  let P_A : ℝ := 0.3
  let P_B : ℝ := 0.5
  let P_not_A : ℝ := 1 - P_A
  let P_not_B : ℝ := 1 - P_B
  let P_both_miss : ℝ := P_not_A * P_not_B
  let P_plane_hit : ℝ := 1 - P_both_miss
  P_plane_hit = 0.65 :=
by
  sorry

end plane_hit_probability_l1173_117325


namespace total_cotton_yield_l1173_117316

variables {m n a b : ℕ}

theorem total_cotton_yield (m n a b : ℕ) : 
  m * a + n * b = m * a + n * b := by
  sorry

end total_cotton_yield_l1173_117316


namespace little_john_initial_money_l1173_117319

theorem little_john_initial_money :
  let spent := 1.05
  let given_to_each_friend := 2 * 1.00
  let total_spent := spent + given_to_each_friend
  let left := 2.05
  total_spent + left = 5.10 :=
by
  let spent := 1.05
  let given_to_each_friend := 2 * 1.00
  let total_spent := spent + given_to_each_friend
  let left := 2.05
  show total_spent + left = 5.10
  sorry

end little_john_initial_money_l1173_117319


namespace correct_statements_about_microbial_counting_l1173_117393

def hemocytometer_counts_bacteria_or_yeast : Prop :=
  true -- based on condition 1

def plate_streaking_allows_colony_counting : Prop :=
  false -- count is not done using the plate streaking method, based on the analysis

def dilution_plating_allows_colony_counting : Prop :=
  true -- based on condition 3  
  
def dilution_plating_count_is_accurate : Prop :=
  false -- colony count is often lower than the actual number, based on the analysis

theorem correct_statements_about_microbial_counting :
  (hemocytometer_counts_bacteria_or_yeast ∧ dilution_plating_allows_colony_counting)
= (plate_streaking_allows_colony_counting ∨ dilution_plating_count_is_accurate) :=
by sorry

end correct_statements_about_microbial_counting_l1173_117393


namespace inhabitant_eq_resident_l1173_117382

-- Definitions
def inhabitant : Type := String
def resident : Type := String

-- The equivalence theorem
theorem inhabitant_eq_resident :
  ∀ (x : inhabitant), x = "resident" :=
by
  sorry

end inhabitant_eq_resident_l1173_117382


namespace howard_groups_l1173_117354

theorem howard_groups :
  (18 : ℕ) / (24 / 4) = 3 := sorry

end howard_groups_l1173_117354


namespace total_people_museum_l1173_117386

-- Conditions
def first_bus_people : ℕ := 12
def second_bus_people := 2 * first_bus_people
def third_bus_people := second_bus_people - 6
def fourth_bus_people := first_bus_people + 9

-- Question to prove
theorem total_people_museum : first_bus_people + second_bus_people + third_bus_people + fourth_bus_people = 75 :=
by
  -- The proof is skipped but required to complete the theorem
  sorry

end total_people_museum_l1173_117386


namespace solve_inequality_l1173_117384

noncomputable def inequality (x : ℕ) : Prop :=
  6 * (9 : ℝ)^(1/x) - 13 * (3 : ℝ)^(1/x) * (2 : ℝ)^(1/x) + 6 * (4 : ℝ)^(1/x) ≤ 0

theorem solve_inequality (x : ℕ) (hx : 1 < x) : inequality x ↔ x ≥ 2 :=
by {
  sorry
}

end solve_inequality_l1173_117384


namespace longer_piece_length_is_20_l1173_117371

-- Define the rope length
def ropeLength : ℕ := 35

-- Define the ratio of the two pieces
def ratioA : ℕ := 3
def ratioB : ℕ := 4
def totalRatio : ℕ := ratioA + ratioB

-- Define the length of each part
def partLength : ℕ := ropeLength / totalRatio

-- Define the length of the longer piece
def longerPieceLength : ℕ := ratioB * partLength

-- Theorem to prove that the length of the longer piece is 20 inches
theorem longer_piece_length_is_20 : longerPieceLength = 20 := by 
  sorry

end longer_piece_length_is_20_l1173_117371


namespace problem_l1173_117372

noncomputable def f (x : ℝ) : ℝ :=
  sorry

theorem problem (f : ℝ → ℝ) (h1 : ∀ x y : ℝ, f (x + y) - f y = x * (x + 2 * y + 1))
                (h2 : f 1 = 0) :
  f 0 = -2 ∧ ∀ x : ℝ, f x = x^2 + x - 2 := by
  sorry

end problem_l1173_117372


namespace Danica_additional_cars_l1173_117311

theorem Danica_additional_cars (n : ℕ) (row_size : ℕ) (danica_cars : ℕ) (answer : ℕ) :
  row_size = 8 →
  danica_cars = 37 →
  answer = 3 →
  ∃ k : ℕ, (k + danica_cars) % row_size = 0 ∧ k = answer :=
by
  sorry

end Danica_additional_cars_l1173_117311


namespace final_price_of_purchases_l1173_117302

theorem final_price_of_purchases :
  let electronic_discount := 0.20
  let clothing_discount := 0.15
  let bundle_discount := 10
  let voucher_threshold := 200
  let voucher_value := 20
  let voucher_limit := 2
  let delivery_charge := 15
  let tax_rate := 0.08

  let electronic_original_price := 150
  let clothing_original_price := 80
  let num_clothing := 2

  -- Calculate discounts
  let electronic_discount_amount := electronic_original_price * electronic_discount
  let electronic_discount_price := electronic_original_price - electronic_discount_amount
  let clothing_discount_amount := clothing_original_price * clothing_discount
  let clothing_discount_price := clothing_original_price - clothing_discount_amount

  -- Sum of discounted clothing items
  let total_clothing_discount_price := clothing_discount_price * num_clothing

  -- Calculate bundle discount
  let total_before_bundle_discount := electronic_discount_price + total_clothing_discount_price
  let total_after_bundle_discount := total_before_bundle_discount - bundle_discount

  -- Calculate vouchers
  let num_vouchers := if total_after_bundle_discount >= voucher_threshold * 2 then voucher_limit else 
                      if total_after_bundle_discount >= voucher_threshold then 1 else 0
  let total_voucher_amount := num_vouchers * voucher_value
  let total_after_voucher_discount := total_after_bundle_discount - total_voucher_amount

  -- Add delivery charge
  let total_before_tax := total_after_voucher_discount + delivery_charge

  -- Calculate tax
  let tax_amount := total_before_tax * tax_rate
  let final_price := total_before_tax + tax_amount

  final_price = 260.28 :=
by
  -- the actual proof will be included here
  sorry

end final_price_of_purchases_l1173_117302


namespace fuel_tank_capacity_l1173_117333

theorem fuel_tank_capacity
  (ethanol_A_ethanol : ∀ {x : Float}, x = 0.12 * 49.99999999999999)
  (ethanol_B_ethanol : ∀ {C : Float}, x = 0.16 * (C - 49.99999999999999))
  (total_ethanol : ∀ {C : Float}, 0.12 * 49.99999999999999 + 0.16 * (C - 49.99999999999999) = 30) :
  (C = 162.5) :=
sorry

end fuel_tank_capacity_l1173_117333


namespace jezebel_total_flower_cost_l1173_117352

theorem jezebel_total_flower_cost :
  let red_rose_count := 2 * 12
  let red_rose_cost := 1.5
  let sunflower_count := 3
  let sunflower_cost := 3
  (red_rose_count * red_rose_cost + sunflower_count * sunflower_cost = 45) :=
by
  let red_rose_count := 2 * 12
  let red_rose_cost := 1.5
  let sunflower_count := 3
  let sunflower_cost := 3
  sorry

end jezebel_total_flower_cost_l1173_117352


namespace mary_can_keep_warm_l1173_117338

theorem mary_can_keep_warm :
  let chairs := 18
  let chairs_sticks := 6
  let tables := 6
  let tables_sticks := 9
  let stools := 4
  let stools_sticks := 2
  let sticks_per_hour := 5
  let total_sticks := (chairs * chairs_sticks) + (tables * tables_sticks) + (stools * stools_sticks)
  let hours := total_sticks / sticks_per_hour
  hours = 34 := by
{
  sorry
}

end mary_can_keep_warm_l1173_117338


namespace total_bananas_eq_l1173_117388

def groups_of_bananas : ℕ := 2
def bananas_per_group : ℕ := 145

theorem total_bananas_eq : groups_of_bananas * bananas_per_group = 290 :=
by
  sorry

end total_bananas_eq_l1173_117388


namespace original_grain_correct_l1173_117374

-- Define the initial quantities
def grain_spilled : ℕ := 49952
def grain_remaining : ℕ := 918

-- Define the original amount of grain expected
def original_grain : ℕ := 50870

-- Prove that the original amount of grain was correct
theorem original_grain_correct : grain_spilled + grain_remaining = original_grain := 
by
  sorry

end original_grain_correct_l1173_117374


namespace points_on_line_l1173_117379

theorem points_on_line (y1 y2 : ℝ) 
  (hA : y1 = - (1 / 2 : ℝ) * 1 - 1) 
  (hB : y2 = - (1 / 2 : ℝ) * 3 - 1) :
  y1 > y2 := 
by
  sorry

end points_on_line_l1173_117379


namespace tens_digit_11_pow_2045_l1173_117341

theorem tens_digit_11_pow_2045 : 
    ((11 ^ 2045) % 100) / 10 % 10 = 5 :=
by
    sorry

end tens_digit_11_pow_2045_l1173_117341


namespace merchant_salt_mixture_l1173_117310

theorem merchant_salt_mixture (x : ℝ) (h₀ : (0.48 * (40 + x)) = 1.20 * (14 + 0.50 * x)) : x = 0 :=
by
  sorry

end merchant_salt_mixture_l1173_117310


namespace percentage_of_125_equals_75_l1173_117387

theorem percentage_of_125_equals_75 (p : ℝ) (h : p * 125 = 75) : p = 60 / 100 :=
by
  sorry

end percentage_of_125_equals_75_l1173_117387


namespace smallest_possible_N_l1173_117381

theorem smallest_possible_N (N : ℕ) (h1 : N > 70) (h2 : 70 ∣ 21 * N) : N = 80 :=
sorry

end smallest_possible_N_l1173_117381


namespace route_difference_l1173_117343

noncomputable def time_route_A (distance_A : ℝ) (speed_A : ℝ) : ℝ :=
  (distance_A / speed_A) * 60

noncomputable def time_route_B (distance1_B distance2_B distance3_B : ℝ) (speed1_B speed2_B speed3_B : ℝ) : ℝ :=
  ((distance1_B / speed1_B) * 60) + 
  ((distance2_B / speed2_B) * 60) + 
  ((distance3_B / speed3_B) * 60)

theorem route_difference
  (distance_A : ℝ := 8)
  (speed_A : ℝ := 25)
  (distance1_B : ℝ := 2)
  (distance2_B : ℝ := 0.5)
  (speed1_B : ℝ := 50)
  (speed2_B : ℝ := 20)
  (distance_total_B : ℝ := 7)
  (speed3_B : ℝ := 35) :
  time_route_A distance_A speed_A - time_route_B distance1_B distance2_B (distance_total_B - distance1_B - distance2_B) speed1_B speed2_B speed3_B = 7.586 :=
by
  sorry

end route_difference_l1173_117343


namespace divisibility_of_powers_l1173_117377

theorem divisibility_of_powers (n : ℤ) : 65 ∣ (7^4 * n - 4^4 * n) :=
by
  sorry

end divisibility_of_powers_l1173_117377


namespace gcd_combination_l1173_117304

theorem gcd_combination (a b d : ℕ) (h : d = Nat.gcd a b) : 
  Nat.gcd (5 * a + 3 * b) (13 * a + 8 * b) = d := 
by
  sorry

end gcd_combination_l1173_117304


namespace gain_in_transaction_per_year_l1173_117359

noncomputable def compounded_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

noncomputable def gain_per_year (P : ℝ) (t : ℝ) (r1 : ℝ) (n1 : ℕ) (r2 : ℝ) (n2 : ℕ) : ℝ :=
  let amount_repaid := compounded_interest P r1 n1 t
  let amount_received := compounded_interest P r2 n2 t
  (amount_received - amount_repaid) / t

theorem gain_in_transaction_per_year :
  let P := 8000
  let t := 3
  let r1 := 0.05
  let n1 := 2
  let r2 := 0.07
  let n2 := 4
  abs (gain_per_year P t r1 n1 r2 n2 - 191.96) < 0.01 :=
by
  sorry

end gain_in_transaction_per_year_l1173_117359


namespace arithmetic_example_l1173_117397

theorem arithmetic_example : 2 + 8 * 3 - 4 + 7 * 2 / 2 = 29 :=
by
  sorry

end arithmetic_example_l1173_117397


namespace intersection_A_B_l1173_117346

def A : Set ℝ := {x | x^2 + x - 6 > 0}
def B : Set ℝ := {x | -2 < x ∧ x < 4}

theorem intersection_A_B :
  A ∩ B = {x | 2 < x ∧ x < 4} :=
sorry

end intersection_A_B_l1173_117346


namespace absent_present_probability_l1173_117392

theorem absent_present_probability : 
  ∀ (p_absent_normal p_absent_workshop p_present_workshop : ℚ), 
    p_absent_normal = 1 / 20 →
    p_absent_workshop = 2 * p_absent_normal →
    p_present_workshop = 1 - p_absent_workshop →
    p_absent_workshop = 1 / 10 →
    (p_present_workshop * p_absent_workshop + p_absent_workshop * p_present_workshop) * 100 = 18 :=
by
  intros
  sorry

end absent_present_probability_l1173_117392


namespace at_least_one_genuine_l1173_117389

/-- Given 12 products, of which 10 are genuine and 2 are defective.
    If 3 products are randomly selected, then at least one of the selected products is a genuine product. -/
theorem at_least_one_genuine : 
  ∀ (products : Fin 12 → Prop), 
  (∃ n₁ n₂ : Fin 12, (n₁ ≠ n₂) ∧ 
                   (products n₁ = true) ∧ 
                   (products n₂ = true) ∧ 
                   (∃ n₁' n₂' : Fin 12, (n₁ ≠ n₁' ∧ n₂ ≠ n₂') ∧
                                         products n₁' = products n₂' = true ∧
                                         ∀ j : Fin 3, products j = true)) → 
  (∃ m : Fin 3, products m = true) :=
sorry

end at_least_one_genuine_l1173_117389


namespace triangle_problem_l1173_117398

-- Defining the conditions as Lean constructs
variable (a c : ℝ)
variable (b : ℝ := 3)
variable (cosB : ℝ := 1 / 3)
variable (dotProductBACBC : ℝ := 2)
variable (cosB_minus_C : ℝ := 23 / 27)

-- Define the problem as a theorem in Lean 4
theorem triangle_problem
  (h1 : a > c)
  (h2 : a * c * cosB = dotProductBACBC)
  (h3 : a^2 + c^2 = 13) :
  a = 3 ∧ c = 2 ∧ cosB_minus_C = 23 / 27 := by
  sorry

end triangle_problem_l1173_117398


namespace total_legs_is_26_l1173_117349

-- Define the number of puppies and chicks
def number_of_puppies : Nat := 3
def number_of_chicks : Nat := 7

-- Define the number of legs per puppy and per chick
def legs_per_puppy : Nat := 4
def legs_per_chick : Nat := 2

-- Calculate the total number of legs
def total_legs := (number_of_puppies * legs_per_puppy) + (number_of_chicks * legs_per_chick)

-- Prove that the total number of legs is 26
theorem total_legs_is_26 : total_legs = 26 := by
  sorry

end total_legs_is_26_l1173_117349


namespace part1_l1173_117326

-- Define the vectors a and b
def a : ℝ × ℝ := (-3, 4)
def b : ℝ × ℝ := (2, -1)
-- Define the vectors a - x b and a - b
def vec1 (x : ℝ) : ℝ × ℝ := (a.1 - x * b.1, a.2 - x * b.2)
def vec2 : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
-- Define the dot product
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2 

-- Main theorem: prove that the vectors being perpendicular implies x = -7/3
theorem part1 (x : ℝ) : dot_product (vec1 x) vec2 = 0 → x = -7 / 3 :=
by
  sorry

end part1_l1173_117326


namespace carmen_candle_burn_time_l1173_117378

theorem carmen_candle_burn_time 
  (burn_time_first_scenario : ℕ)
  (nights_per_candle : ℕ)
  (total_candles_second_scenario : ℕ)
  (total_nights_second_scenario : ℕ)
  (h1 : burn_time_first_scenario = 1)
  (h2 : nights_per_candle = 8)
  (h3 : total_candles_second_scenario = 6)
  (h4 : total_nights_second_scenario = 24) :
  (total_candles_second_scenario * nights_per_candle) / total_nights_second_scenario = 2 :=
by
  sorry

end carmen_candle_burn_time_l1173_117378


namespace solution_of_equation_l1173_117355

noncomputable def integer_part (x : ℝ) : ℤ := Int.floor x
noncomputable def fractional_part (x : ℝ) : ℝ := x - integer_part x

theorem solution_of_equation (k : ℤ) (h : -1 ≤ k ∧ k ≤ 5) :
  ∃ x : ℝ, 4 * ↑(integer_part x) = 25 * fractional_part x - 4.5 ∧
           x = k + (8 * ↑k + 9) / 50 := 
sorry

end solution_of_equation_l1173_117355


namespace miles_total_instruments_l1173_117380

-- Definitions based on the conditions
def fingers : ℕ := 10
def hands : ℕ := 2
def heads : ℕ := 1
def trumpets : ℕ := fingers - 3
def guitars : ℕ := hands + 2
def trombones : ℕ := heads + 2
def french_horns : ℕ := guitars - 1
def total_instruments : ℕ := trumpets + guitars + trombones + french_horns

-- Main theorem
theorem miles_total_instruments : total_instruments = 17 := 
sorry

end miles_total_instruments_l1173_117380


namespace deposit_is_3000_l1173_117342

-- Define the constants
def cash_price : ℝ := 8000
def monthly_installment : ℝ := 300
def number_of_installments : ℕ := 30
def savings_by_paying_cash : ℝ := 4000

-- Define the total installment payments
def total_installment_payments : ℝ := number_of_installments * monthly_installment

-- Define the total price paid, which includes the deposit and installments
def total_paid : ℝ := cash_price + savings_by_paying_cash

-- Define the deposit
def deposit : ℝ := total_paid - total_installment_payments

-- Statement to be proven
theorem deposit_is_3000 : deposit = 3000 := 
by 
  sorry

end deposit_is_3000_l1173_117342


namespace slower_train_passing_time_l1173_117383

/--
Two goods trains, each 500 meters long, are running in opposite directions on parallel tracks. 
Their respective speeds are 45 kilometers per hour and 15 kilometers per hour. 
Prove that the time taken by the slower train to pass the driver of the faster train is 30 seconds.
-/
theorem slower_train_passing_time : 
  ∀ (distance length_speed : ℝ), 
    distance = 500 →
    ∃ (v1 v2 : ℝ), 
      v1 = 45 * (1000 / 3600) → 
      v2 = 15 * (1000 / 3600) →
      (distance / ((v1 + v2) * (3/50)) = 30) :=
by
  sorry

end slower_train_passing_time_l1173_117383


namespace find_number_of_girls_in_class_l1173_117323

variable (G : ℕ)

def number_of_ways_to_select_two_boys (n : ℕ) : ℕ := Nat.choose n 2

theorem find_number_of_girls_in_class 
  (boys : ℕ := 13) 
  (ways_to_select_students : ℕ := 780) 
  (ways_to_select_two_boys : ℕ := number_of_ways_to_select_two_boys boys) :
  G * ways_to_select_two_boys = ways_to_select_students → G = 10 := 
by
  sorry

end find_number_of_girls_in_class_l1173_117323


namespace entertainment_expense_percentage_l1173_117328

noncomputable def salary : ℝ := 10000
noncomputable def savings : ℝ := 2000
noncomputable def food_expense_percentage : ℝ := 0.40
noncomputable def house_rent_percentage : ℝ := 0.20
noncomputable def conveyance_percentage : ℝ := 0.10

theorem entertainment_expense_percentage :
  let E := (1 - (food_expense_percentage + house_rent_percentage + conveyance_percentage) - (savings / salary))
  E = 0.10 :=
by
  sorry

end entertainment_expense_percentage_l1173_117328


namespace average_speed_l1173_117344

theorem average_speed (D : ℝ) (h1 : 0 < D) :
  let s1 := 60   -- speed from Q to B in miles per hour
  let s2 := 20   -- speed from B to C in miles per hour
  let d1 := 2 * D  -- distance from Q to B
  let d2 := D     -- distance from B to C
  let t1 := d1 / s1  -- time to travel from Q to B
  let t2 := d2 / s2  -- time to travel from B to C
  let total_distance := d1 + d2  -- total distance
  let total_time := t1 + t2   -- total time
  let average_speed := total_distance / total_time  -- average speed
  average_speed = 36 :=
by
  sorry

end average_speed_l1173_117344


namespace min_sin_cos_sixth_power_l1173_117334

noncomputable def min_value_sin_cos_expr : ℝ :=
  (3 + 2 * Real.sqrt 2) / 12

theorem min_sin_cos_sixth_power :
  ∃ x : ℝ, (∀ y, (Real.sin y) ^ 6 + 2 * (Real.cos y) ^ 6 ≥ min_value_sin_cos_expr) ∧ 
            ((Real.sin x) ^ 6 + 2 * (Real.cos x) ^ 6 = min_value_sin_cos_expr) :=
sorry

end min_sin_cos_sixth_power_l1173_117334


namespace total_distance_traveled_l1173_117318

-- Definitions of distances in km
def ZX : ℝ := 4000
def XY : ℝ := 5000
def YZ : ℝ := (XY^2 - ZX^2)^(1/2)

-- Prove the total distance traveled
theorem total_distance_traveled :
  XY + YZ + ZX = 11500 := by
  have h1 : ZX = 4000 := rfl
  have h2 : XY = 5000 := rfl
  have h3 : YZ = (5000^2 - 4000^2)^(1/2) := rfl
  -- Continue the proof showing the calculation of each step
  sorry

end total_distance_traveled_l1173_117318


namespace peter_present_age_l1173_117376

def age_problem (P J : ℕ) : Prop :=
  J = P + 12 ∧ P - 10 = (1 / 3 : ℚ) * (J - 10)

theorem peter_present_age : ∃ (P : ℕ), ∃ (J : ℕ), age_problem P J ∧ P = 16 :=
by {
  -- Add the proof here, which is not required
  sorry
}

end peter_present_age_l1173_117376


namespace find_a100_l1173_117330

noncomputable def arithmetic_sequence (n : ℕ) (a : ℕ → ℤ) : Prop :=
∀ n : ℕ, a n - a (n + 1) = 2

theorem find_a100 (a : ℕ → ℤ) (h1 : arithmetic_sequence 3 a) (h2 : a 3 = 6) :
  a 100 = -188 :=
sorry

end find_a100_l1173_117330


namespace max_a_value_l1173_117309

noncomputable def f (x : ℝ) : ℝ := (1 + Real.log x) / x - 1

theorem max_a_value : 
  ∀ (a : ℝ), 
  (∀ (x : ℝ), x ∈ Set.Icc (1/2) 2 → (a + 1) * x - 1 - Real.log x ≤ 0) → 
  a ≤ 1 - 2 * Real.log 2 := 
by
  sorry

end max_a_value_l1173_117309


namespace robert_salary_loss_l1173_117306

theorem robert_salary_loss (S : ℝ) : 
  let decreased_salary := S - 0.3 * S
  let increased_salary := decreased_salary + 0.3 * decreased_salary
  100 * (1 - increased_salary / S) = 9 :=
by
  let decreased_salary := S - 0.3 * S
  let increased_salary := decreased_salary + 0.3 * decreased_salary
  sorry

end robert_salary_loss_l1173_117306


namespace tangent_lines_through_origin_l1173_117358

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 1

variable (a : ℝ)

theorem tangent_lines_through_origin 
  (h1 : ∃ m1 m2 : ℝ, m1 ≠ m2 ∧ (f a (-m1) + f a (m1 + 2)) / 2 = f a 1) :
  ∃ t1 t2 : ℝ, t1 ≠ t2 ∧ (f a t1 * (1 / t1) = f a 0) ∧ (f a t2 * (1 / t2) = f a 0) := 
sorry

end tangent_lines_through_origin_l1173_117358


namespace Caleb_pencils_fewer_than_twice_Candy_l1173_117337

theorem Caleb_pencils_fewer_than_twice_Candy:
  ∀ (P_Caleb P_Candy: ℕ), 
    P_Candy = 9 → 
    (∃ X, P_Caleb = 2 * P_Candy - X) → 
    P_Caleb + 5 - 10 = 10 → 
    (2 * P_Candy - P_Caleb = 3) :=
by
  intros P_Caleb P_Candy hCandy hCalebLess twCalen
  sorry

end Caleb_pencils_fewer_than_twice_Candy_l1173_117337


namespace find_x_value_l1173_117369

theorem find_x_value :
  ∃ (x : ℤ), ∀ (y z w : ℤ), (x = 2 * y + 4) → (y = z + 5) → (z = 2 * w + 3) → (w = 50) → x = 220 :=
by
  sorry

end find_x_value_l1173_117369


namespace point_in_quadrants_l1173_117394

theorem point_in_quadrants (x y : ℝ) (h1 : 4 * x + 7 * y = 28) (h2 : |x| = |y|) :
  (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) :=
by
  sorry

end point_in_quadrants_l1173_117394


namespace incorrect_statement_D_l1173_117395

theorem incorrect_statement_D :
  ¬ (abs (-1) - abs 1 = 2) :=
by
  sorry

end incorrect_statement_D_l1173_117395


namespace triangle_properties_l1173_117353

-- Define the sides of the triangle
def side1 : ℕ := 8
def side2 : ℕ := 15
def hypotenuse : ℕ := 17

-- Using the Pythagorean theorem to assert it is a right triangle
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

-- Calculate the area of the right triangle
def triangle_area (a b : ℕ) : ℕ :=
  (a * b) / 2

-- Calculate the perimeter of the triangle
def triangle_perimeter (a b c : ℕ) : ℕ :=
  a + b + c

theorem triangle_properties :
  let a := side1
  let b := side2
  let c := hypotenuse
  is_right_triangle a b c →
  triangle_area a b = 60 ∧ triangle_perimeter a b c = 40 := by
  intros h
  sorry

end triangle_properties_l1173_117353


namespace gardener_trees_problem_l1173_117390

theorem gardener_trees_problem 
  (maple_trees : ℕ) (oak_trees : ℕ) (birch_trees : ℕ) 
  (total_trees : ℕ) (valid_positions : ℕ) 
  (total_arrangements : ℕ) (probability_numerator : ℕ) (probability_denominator : ℕ) 
  (reduced_numerator : ℕ) (reduced_denominator : ℕ) (m_plus_n : ℕ) :
  (maple_trees = 5) ∧ 
  (oak_trees = 3) ∧ 
  (birch_trees = 7) ∧ 
  (total_trees = 15) ∧ 
  (valid_positions = 8) ∧ 
  (total_arrangements = 120120) ∧ 
  (probability_numerator = 40) ∧ 
  (probability_denominator = total_arrangements) ∧ 
  (reduced_numerator = 1) ∧ 
  (reduced_denominator = 3003) ∧ 
  (m_plus_n = reduced_numerator + reduced_denominator) → 
  m_plus_n = 3004 := 
by
  intros _
  sorry

end gardener_trees_problem_l1173_117390


namespace relationship_between_sets_l1173_117327

def M : Set ℤ := {x | ∃ k : ℤ, x = 5 * k - 2}
def P : Set ℤ := {x | ∃ n : ℤ, x = 5 * n + 3}
def S : Set ℤ := {x | ∃ m : ℤ, x = 10 * m + 3}

theorem relationship_between_sets : S ⊆ P ∧ P = M := by
  sorry

end relationship_between_sets_l1173_117327


namespace minimum_value_of_f_l1173_117329

def f (x : ℝ) : ℝ := |3 - x| + |x - 7|

theorem minimum_value_of_f : ∃ x : ℝ, f x = 4 ∧ ∀ y : ℝ, f y ≥ 4 :=
by {
  sorry
}

end minimum_value_of_f_l1173_117329


namespace A_share_in_profit_l1173_117301

def investment_A := 6300
def investment_B := 4200
def investment_C := 10500
def total_profit := 12500

def total_investment := investment_A + investment_B + investment_C
def A_ratio := investment_A / total_investment

theorem A_share_in_profit : (total_profit * A_ratio) = 3750 := by
  sorry

end A_share_in_profit_l1173_117301


namespace largest_four_digit_number_divisible_by_4_with_digit_sum_20_l1173_117361

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000
def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0
def digit_sum_is_20 (n : ℕ) : Prop :=
  (n / 1000) + ((n % 1000) / 100) + ((n % 100) / 10) + (n % 10) = 20

theorem largest_four_digit_number_divisible_by_4_with_digit_sum_20 :
  ∃ n : ℕ, is_four_digit n ∧ is_divisible_by_4 n ∧ digit_sum_is_20 n ∧ ∀ m : ℕ, is_four_digit m ∧ is_divisible_by_4 m ∧ digit_sum_is_20 m → m ≤ n :=
  sorry

end largest_four_digit_number_divisible_by_4_with_digit_sum_20_l1173_117361
