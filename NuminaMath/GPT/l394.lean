import Mathlib

namespace NUMINAMATH_GPT_find_digit_P_l394_39482

theorem find_digit_P (P Q R S T : ℕ) (digits : Finset ℕ) (h1 : digits = {1, 2, 3, 6, 8}) 
(h2 : P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ P ≠ T ∧ Q ≠ R ∧ Q ≠ S ∧ Q ≠ T ∧ R ≠ S ∧ R ≠ T ∧ S ≠ T)
(h3 : P ∈ digits ∧ Q ∈ digits ∧ R ∈ digits ∧ S ∈ digits ∧ T ∈ digits)
(hPQR_div_6 : (100 * P + 10 * Q + R) % 6 = 0)
(hQRS_div_8 : (100 * Q + 10 * R + S) % 8 = 0)
(hRST_div_3 : (100 * R + 10 * S + T) % 3 = 0) : 
P = 2 := 
sorry

end NUMINAMATH_GPT_find_digit_P_l394_39482


namespace NUMINAMATH_GPT_inequality_proof_equality_case_l394_39449

-- Defining that a, b, c are positive real numbers
variables (a b c : ℝ)
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

-- The main theorem statement
theorem inequality_proof :
  a^2 + b^2 + c^2 + ( (1 / a) + (1 / b) + (1 / c) )^2 >= 6 * Real.sqrt 3 :=
sorry

-- Equality case
theorem equality_case :
  a = b ∧ b = c ∧ a = Real.sqrt 3^(1/4) →
  a^2 + b^2 + c^2 + ( (1 / a) + (1 / b) + (1 / c) )^2 = 6 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_inequality_proof_equality_case_l394_39449


namespace NUMINAMATH_GPT_sum_of_digits_base10_representation_l394_39403

def digit_sum (n : ℕ) : ℕ := sorry  -- Define a function to calculate the sum of digits

noncomputable def a : ℕ := 7 * (10 ^ 1234 - 1) / 9
noncomputable def b : ℕ := 2 * (10 ^ 1234 - 1) / 9
noncomputable def product : ℕ := 7 * a * b

theorem sum_of_digits_base10_representation : digit_sum product = 11100 := 
by sorry

end NUMINAMATH_GPT_sum_of_digits_base10_representation_l394_39403


namespace NUMINAMATH_GPT_find_a_extreme_values_l394_39420

noncomputable def f (x a : ℝ) : ℝ := (x - a)^2 + 4 * Real.log (x + 1)
noncomputable def f' (x a : ℝ) : ℝ := 2 * (x - a) + 4 / (x + 1)

-- Given conditions
theorem find_a (a : ℝ) :
  f' 1 a = 0 ↔ a = 2 :=
by
  sorry

theorem extreme_values :
  ∃ x : ℝ, -1 < x ∧ f (0 : ℝ) 2 = 4 ∨ f (1 : ℝ) 2 = 1 + 4 * Real.log 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_extreme_values_l394_39420


namespace NUMINAMATH_GPT_percentage_second_question_correct_l394_39418

theorem percentage_second_question_correct (a b c : ℝ) 
  (h1 : a = 0.75) (h2 : b = 0.20) (h3 : c = 0.50) :
  (1 - b) - (a - c) + c = 0.55 :=
by
  sorry

end NUMINAMATH_GPT_percentage_second_question_correct_l394_39418


namespace NUMINAMATH_GPT_f_at_neg_2_l394_39410

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^5 + a * x^3 + x^2 + b * x + 2

-- Given the condition
def f_at_2_eq_3 (a b : ℝ) : Prop := f 2 a b = 3

-- Prove the value of f(-2)
theorem f_at_neg_2 (a b : ℝ) (h : f_at_2_eq_3 a b) : f (-2) a b = 1 :=
sorry

end NUMINAMATH_GPT_f_at_neg_2_l394_39410


namespace NUMINAMATH_GPT_speaker_is_tweedledee_l394_39409

-- Definitions
variable (Speaks : Prop) (is_tweedledum : Prop) (has_black_card : Prop)

-- Condition: If the speaker is Tweedledum, then the card in the speaker's pocket is not a black suit.
axiom A1 : is_tweedledum → ¬ has_black_card

-- Goal: Prove that the speaker is Tweedledee.
theorem speaker_is_tweedledee (h1 : Speaks) : ¬ is_tweedledum :=
by
  sorry

end NUMINAMATH_GPT_speaker_is_tweedledee_l394_39409


namespace NUMINAMATH_GPT_ratio_of_spinsters_to_cats_l394_39429

-- Defining the problem in Lean 4
theorem ratio_of_spinsters_to_cats (S C : ℕ) (h₁ : S = 22) (h₂ : C = S + 55) : S / gcd S C = 2 ∧ C / gcd S C = 7 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_spinsters_to_cats_l394_39429


namespace NUMINAMATH_GPT_probability_odd_multiple_of_5_l394_39487

theorem probability_odd_multiple_of_5 :
  (∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 100 ∧ 1 ≤ b ∧ b ≤ 100 ∧ 1 ≤ c ∧ c ≤ 100 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    (a * b * c) % 2 = 1 ∧ (a * b * c) % 5 = 0) → 
  p = 3 / 125 := 
sorry

end NUMINAMATH_GPT_probability_odd_multiple_of_5_l394_39487


namespace NUMINAMATH_GPT_division_exponent_rule_l394_39446

theorem division_exponent_rule (a : ℝ) (h : a ≠ 0) : (a^8) / (a^2) = a^6 :=
sorry

end NUMINAMATH_GPT_division_exponent_rule_l394_39446


namespace NUMINAMATH_GPT_grasshopper_jump_l394_39497

-- Definitions for the distances jumped
variables (G F M : ℕ)

-- Conditions given in the problem
def condition1 : Prop := G = F + 19
def condition2 : Prop := M = F - 12
def condition3 : Prop := M = 8

-- The theorem statement
theorem grasshopper_jump : condition1 G F ∧ condition2 F M ∧ condition3 M → G = 39 :=
by
  sorry

end NUMINAMATH_GPT_grasshopper_jump_l394_39497


namespace NUMINAMATH_GPT_exponent_property_l394_39407

theorem exponent_property (a : ℝ) (m n : ℝ) (h₁ : a^m = 4) (h₂ : a^n = 8) : a^(m + n) = 32 := 
by 
  sorry

end NUMINAMATH_GPT_exponent_property_l394_39407


namespace NUMINAMATH_GPT_smallest_angle_of_triangle_l394_39461

theorem smallest_angle_of_triangle (a b c : ℕ) 
    (h1 : a = 60) (h2 : b = 70) (h3 : a + b + c = 180) : 
    c = 50 ∧ min a (min b c) = 50 :=
by {
    sorry
}

end NUMINAMATH_GPT_smallest_angle_of_triangle_l394_39461


namespace NUMINAMATH_GPT_wheels_in_garage_l394_39462

-- Definitions of the entities within the problem
def cars : Nat := 2
def car_wheels : Nat := 4

def riding_lawnmower : Nat := 1
def lawnmower_wheels : Nat := 4

def bicycles : Nat := 3
def bicycle_wheels : Nat := 2

def tricycle : Nat := 1
def tricycle_wheels : Nat := 3

def unicycle : Nat := 1
def unicycle_wheels : Nat := 1

-- The total number of wheels in the garage
def total_wheels :=
  (cars * car_wheels) +
  (riding_lawnmower * lawnmower_wheels) +
  (bicycles * bicycle_wheels) +
  (tricycle * tricycle_wheels) +
  (unicycle * unicycle_wheels)

-- The theorem we wish to prove
theorem wheels_in_garage : total_wheels = 22 := by
  sorry

end NUMINAMATH_GPT_wheels_in_garage_l394_39462


namespace NUMINAMATH_GPT_quadratic_equal_roots_l394_39490

theorem quadratic_equal_roots (a : ℝ) : (∀ x : ℝ, x * (x + 1) + a * x = 0) → a = -1 :=
by sorry

end NUMINAMATH_GPT_quadratic_equal_roots_l394_39490


namespace NUMINAMATH_GPT_old_clock_slow_by_12_minutes_l394_39454

theorem old_clock_slow_by_12_minutes (overlap_interval: ℕ) (standard_day_minutes: ℕ)
  (h1: overlap_interval = 66) (h2: standard_day_minutes = 24 * 60):
  standard_day_minutes - 24 * 60 / 66 * 66 = 12 :=
by
  sorry

end NUMINAMATH_GPT_old_clock_slow_by_12_minutes_l394_39454


namespace NUMINAMATH_GPT_denise_crayons_l394_39466

theorem denise_crayons (c : ℕ) :
  (∀ f p : ℕ, f = 30 ∧ p = 7 → c = f * p) → c = 210 :=
by
  intro h
  specialize h 30 7 ⟨rfl, rfl⟩
  exact h

end NUMINAMATH_GPT_denise_crayons_l394_39466


namespace NUMINAMATH_GPT_rectangle_solution_l394_39412

-- Define the given conditions
variables (x y : ℚ)

-- Given equations
def condition1 := (Real.sqrt (x - y) = 2 / 5)
def condition2 := (Real.sqrt (x + y) = 2)

-- Solution
theorem rectangle_solution (x y : ℚ) (h1 : condition1 x y) (h2 : condition2 x y) : 
  x = 52 / 25 ∧ y = 48 / 25 ∧ (Real.sqrt ((52 / 25) * (48 / 25)) = 8 / 25) :=
by
  sorry

end NUMINAMATH_GPT_rectangle_solution_l394_39412


namespace NUMINAMATH_GPT_solve_7_at_8_l394_39430

theorem solve_7_at_8 : (7 * 8) / (7 + 8 + 3) = 28 / 9 := by
  sorry

end NUMINAMATH_GPT_solve_7_at_8_l394_39430


namespace NUMINAMATH_GPT_sum_xyz_l394_39427

theorem sum_xyz (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 + 2 * (y - 1) * (z - 1) = 85)
  (h2 : y^2 + 2 * (z - 1) * (x - 1) = 84)
  (h3 : z^2 + 2 * (x - 1) * (y - 1) = 89) :
  x + y + z = 18 := 
by
  sorry

end NUMINAMATH_GPT_sum_xyz_l394_39427


namespace NUMINAMATH_GPT_satisfies_equation_l394_39450

theorem satisfies_equation (a b c : ℤ) (h₁ : a = b) (h₂ : b = c + 1) :
  a * (a - b) + b * (b - c) + c * (c - a) = 3 := 
by 
  sorry

end NUMINAMATH_GPT_satisfies_equation_l394_39450


namespace NUMINAMATH_GPT_distance_between_petya_and_misha_l394_39448

theorem distance_between_petya_and_misha 
  (v1 v2 v3 : ℝ) -- Speeds of Misha, Dima, and Petya
  (t1 : ℝ) -- Time taken by Misha to finish the race
  (d : ℝ := 1000) -- Distance of the race
  (h1 : d - (v1 * (d / v1)) = 0)
  (h2 : d - 0.9 * v1 * (d / v1) = 100)
  (h3 : d - 0.81 * v1 * (d / v1) = 100) :
  (d - 0.81 * v1 * (d / v1) = 190) := 
sorry

end NUMINAMATH_GPT_distance_between_petya_and_misha_l394_39448


namespace NUMINAMATH_GPT_machines_solution_l394_39475

theorem machines_solution (x : ℝ) (h : x > 0) :
  (1 / (x + 10) + 1 / (x + 3) + 1 / (2 * x) = 1 / x) → x = 3 / 2 := 
by
  sorry

end NUMINAMATH_GPT_machines_solution_l394_39475


namespace NUMINAMATH_GPT_sequence_4951_l394_39402

theorem sequence_4951 :
  (∃ (a : ℕ → ℕ), a 1 = 1 ∧ (∀ n : ℕ, 0 < n → a (n + 1) = a n + n) ∧ a 100 = 4951) :=
sorry

end NUMINAMATH_GPT_sequence_4951_l394_39402


namespace NUMINAMATH_GPT_amount_saved_by_Dalton_l394_39413

-- Defining the costs of each item and the given conditions
def jump_rope_cost : ℕ := 7
def board_game_cost : ℕ := 12
def playground_ball_cost : ℕ := 4
def uncle_gift : ℕ := 13
def additional_needed : ℕ := 4

-- Calculated values based on the conditions
def total_cost : ℕ := jump_rope_cost + board_game_cost + playground_ball_cost
def total_money_needed : ℕ := uncle_gift + additional_needed

-- The theorem that needs to be proved
theorem amount_saved_by_Dalton : total_cost - total_money_needed = 6 :=
by
  sorry -- Proof to be filled in

end NUMINAMATH_GPT_amount_saved_by_Dalton_l394_39413


namespace NUMINAMATH_GPT_greatest_a_l394_39455

theorem greatest_a (a : ℝ) : a^2 - 14*a + 45 ≤ 0 → a ≤ 9 :=
by
  -- placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_greatest_a_l394_39455


namespace NUMINAMATH_GPT_election_votes_l394_39452

theorem election_votes (total_votes : ℕ) (h1 : (4 / 15) * total_votes = 48) : total_votes = 180 :=
sorry

end NUMINAMATH_GPT_election_votes_l394_39452


namespace NUMINAMATH_GPT_ten_year_old_dog_is_64_human_years_l394_39491

namespace DogYears

-- Definition of the conditions
def first_year_in_human_years : ℕ := 15
def second_year_in_human_years : ℕ := 9
def subsequent_year_in_human_years : ℕ := 5

-- Definition of the total human years for a 10-year-old dog.
def dog_age_in_human_years (dog_age : ℕ) : ℕ :=
  if dog_age = 1 then first_year_in_human_years
  else if dog_age = 2 then first_year_in_human_years + second_year_in_human_years
  else first_year_in_human_years + second_year_in_human_years + (dog_age - 2) * subsequent_year_in_human_years

-- The statement to prove
theorem ten_year_old_dog_is_64_human_years : dog_age_in_human_years 10 = 64 :=
  by
    sorry

end DogYears

end NUMINAMATH_GPT_ten_year_old_dog_is_64_human_years_l394_39491


namespace NUMINAMATH_GPT_maximum_quadratic_expr_l394_39435

noncomputable def quadratic_expr (x : ℝ) : ℝ :=
  -5 * x^2 + 25 * x - 7

theorem maximum_quadratic_expr : ∃ x : ℝ, quadratic_expr x = 53 / 4 :=
by
  sorry

end NUMINAMATH_GPT_maximum_quadratic_expr_l394_39435


namespace NUMINAMATH_GPT_bags_of_oranges_l394_39458

-- Define the total number of oranges in terms of bags B
def totalOranges (B : ℕ) : ℕ := 30 * B

-- Define the number of usable oranges left after considering rotten oranges
def usableOranges (B : ℕ) : ℕ := totalOranges B - 50

-- Define the oranges to be sold after keeping some for juice
def orangesToBeSold (B : ℕ) : ℕ := usableOranges B - 30

-- The theorem to state that given 220 oranges will be sold,
-- we need to find B, the number of bags of oranges
theorem bags_of_oranges (B : ℕ) : orangesToBeSold B = 220 → B = 10 :=
by
  sorry

end NUMINAMATH_GPT_bags_of_oranges_l394_39458


namespace NUMINAMATH_GPT_problem_equivalence_l394_39444

-- Define the given circles and their properties
def E (x y : ℝ) : Prop := (x + Real.sqrt 3)^2 + y^2 = 25
def F (x y : ℝ) : Prop := (x - Real.sqrt 3)^2 + y^2 = 1

-- Define the curve C as the trajectory of the center of the moving circle P
def C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the line l intersecting curve C at points A and B with midpoint M(1,1)
def M (A B : ℝ × ℝ) : Prop := (A.1 + B.1 = 2) ∧ (A.2 + B.2 = 2)
def l (x y : ℝ) : Prop := x + 4 * y - 5 = 0

theorem problem_equivalence :
  (∀ x y, E x y ∧ F x y → C x y) ∧
  (∃ A B : ℝ × ℝ, C A.1 A.2 ∧ C B.1 B.2 ∧ M A B → (∀ x y, l x y)) :=
sorry

end NUMINAMATH_GPT_problem_equivalence_l394_39444


namespace NUMINAMATH_GPT_added_number_after_doubling_l394_39463

theorem added_number_after_doubling (original_number : ℕ) (result : ℕ) (added_number : ℕ) 
  (h1 : original_number = 7)
  (h2 : 3 * (2 * original_number + added_number) = result)
  (h3 : result = 69) :
  added_number = 9 :=
by
  sorry

end NUMINAMATH_GPT_added_number_after_doubling_l394_39463


namespace NUMINAMATH_GPT_system_equations_solution_exists_l394_39474

theorem system_equations_solution_exists (m : ℝ) :
  (∃ x y : ℝ, y = 3 * m * x + 6 ∧ y = (4 * m - 3) * x + 9) ↔ m ≠ 3 :=
by
  sorry

end NUMINAMATH_GPT_system_equations_solution_exists_l394_39474


namespace NUMINAMATH_GPT_sheep_problem_system_l394_39459

theorem sheep_problem_system :
  (∃ (x y : ℝ), 5 * x - y = -90 ∧ 50 * x - y = 0) ↔ 
  (5 * x - y = -90 ∧ 50 * x - y = 0) := 
by
  sorry

end NUMINAMATH_GPT_sheep_problem_system_l394_39459


namespace NUMINAMATH_GPT_minimize_expression_l394_39439

theorem minimize_expression (x : ℝ) : 
  ∃ (m : ℝ), m = 2023 ∧ ∀ x, (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2024 ≥ m :=
sorry

end NUMINAMATH_GPT_minimize_expression_l394_39439


namespace NUMINAMATH_GPT_thor_fraction_correct_l394_39431

-- Define the initial conditions
def moes_money : ℕ := 12
def lokis_money : ℕ := 10
def nicks_money : ℕ := 8
def otts_money : ℕ := 6

def thor_received_from_each : ℕ := 2

-- Calculate total money each time
def total_initial_money : ℕ := moes_money + lokis_money + nicks_money + otts_money
def thor_total_received : ℕ := 4 * thor_received_from_each
def thor_fraction_of_total : ℚ := thor_total_received / total_initial_money

-- The theorem to prove
theorem thor_fraction_correct : thor_fraction_of_total = 2/9 :=
by
  sorry

end NUMINAMATH_GPT_thor_fraction_correct_l394_39431


namespace NUMINAMATH_GPT_triangle_side_lengths_l394_39481

theorem triangle_side_lengths {x : ℤ} (h₁ : x + 4 > 10) (h₂ : x + 10 > 4) (h₃ : 10 + 4 > x) :
  ∃ (n : ℕ), n = 7 :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_lengths_l394_39481


namespace NUMINAMATH_GPT_cube_face_coloring_l394_39476

-- Define the type of a cube's face coloring
inductive FaceColor
| black
| white

open FaceColor

def countDistinctColorings : Nat :=
  -- Function to count the number of distinct colorings considering rotational symmetry
  10

theorem cube_face_coloring :
  countDistinctColorings = 10 :=
by
  -- Skip the proof, indicating it should be proved.
  sorry

end NUMINAMATH_GPT_cube_face_coloring_l394_39476


namespace NUMINAMATH_GPT_wang_hao_not_last_l394_39417

theorem wang_hao_not_last (total_players : ℕ) (players_to_choose : ℕ) 
  (wang_hao : ℕ) (ways_to_choose_if_not_last : ℕ) : 
  total_players = 6 ∧ players_to_choose = 3 → 
  ways_to_choose_if_not_last = 100 := 
by
  sorry

end NUMINAMATH_GPT_wang_hao_not_last_l394_39417


namespace NUMINAMATH_GPT_smallest_possible_sum_l394_39479

theorem smallest_possible_sum (x y : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_diff : x ≠ y) (h_eq : 1/x + 1/y = 1/12) : x + y = 49 :=
by
  sorry

end NUMINAMATH_GPT_smallest_possible_sum_l394_39479


namespace NUMINAMATH_GPT_people_per_table_l394_39440

theorem people_per_table (initial_customers left_customers tables remaining_customers : ℕ) 
  (h1 : initial_customers = 21) 
  (h2 : left_customers = 12) 
  (h3 : tables = 3) 
  (h4 : remaining_customers = initial_customers - left_customers) 
  : remaining_customers / tables = 3 :=
by
  sorry

end NUMINAMATH_GPT_people_per_table_l394_39440


namespace NUMINAMATH_GPT_sumsquare_properties_l394_39436

theorem sumsquare_properties {a b c d e f g h i : ℕ} (hc1 : a + b + c = d + e + f) 
(hc2 : d + e + f = g + h + i) 
(hc3 : a + e + i = d + e + f) 
(hc4 : c + e + g = d + e + f) : 
∃ m : ℕ, m % 3 = 0 ∧ (a ≤ (2 * m / 3 - 1)) ∧ (b ≤ (2 * m / 3 - 1)) ∧ (c ≤ (2 * m / 3 - 1)) ∧ (d ≤ (2 * m / 3 - 1)) ∧ (e ≤ (2 * m / 3 - 1)) ∧ (f ≤ (2 * m / 3 - 1)) ∧ (g ≤ (2 * m / 3 - 1)) ∧ (h ≤ (2 * m / 3 - 1)) ∧ (i ≤ (2 * m / 3 - 1)) := 
by {
  sorry
}

end NUMINAMATH_GPT_sumsquare_properties_l394_39436


namespace NUMINAMATH_GPT_calculation_eq_990_l394_39467

theorem calculation_eq_990 : (0.0077 * 3.6) / (0.04 * 0.1 * 0.007) = 990 :=
by
  sorry

end NUMINAMATH_GPT_calculation_eq_990_l394_39467


namespace NUMINAMATH_GPT_right_triangle_hypotenuse_inequality_l394_39473

theorem right_triangle_hypotenuse_inequality
  (a b c m : ℝ)
  (h_right_triangle : c^2 = a^2 + b^2)
  (h_area_relation : a * b = c * m) :
  m + c > a + b :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_hypotenuse_inequality_l394_39473


namespace NUMINAMATH_GPT_value_of_square_sum_l394_39485

theorem value_of_square_sum (x y : ℝ) (h1 : x + 3 * y = 6) (h2 : x * y = -9) : x^2 + 9 * y^2 = 90 := 
by 
  sorry

end NUMINAMATH_GPT_value_of_square_sum_l394_39485


namespace NUMINAMATH_GPT_systematic_sampling_employee_l394_39486

theorem systematic_sampling_employee
    (n : ℕ)
    (employees : Finset ℕ)
    (sample : Finset ℕ)
    (h_n_52 : n = 52)
    (h_employees : employees = Finset.range 52)
    (h_sample_size : sample.card = 4)
    (h_systematic_sample : sample ⊆ employees)
    (h_in_sample : {6, 32, 45} ⊆ sample) :
    19 ∈ sample :=
by
  -- conditions 
  have h0 : 6 ∈ sample := Finset.mem_of_subset h_in_sample (by simp)
  have h1 : 32 ∈ sample := Finset.mem_of_subset h_in_sample (by simp)
  have h2 : 45 ∈ sample := Finset.mem_of_subset h_in_sample (by simp)
  have h_arith : 6 + 45 = 32 + 19 :=
    by linarith
  sorry

end NUMINAMATH_GPT_systematic_sampling_employee_l394_39486


namespace NUMINAMATH_GPT_range_of_a_l394_39471

noncomputable def f : ℝ → ℝ
| x => if x > 0 then Real.log x / Real.log 2 else Real.log (-x) / Real.log (1/2)

theorem range_of_a (a : ℝ) (h : f a > f (-a)) : a ∈ Set.Ioo (-1 : ℝ) 0 ∪ Set.Ioi (1 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l394_39471


namespace NUMINAMATH_GPT_first_team_more_points_l394_39484

/-
Conditions:
  - Beth scored 12 points.
  - Jan scored 10 points.
  - Judy scored 8 points.
  - Angel scored 11 points.
Question:
  - How many more points did the first team get than the second team?
Prove that the first team scored 3 points more than the second team.
-/

theorem first_team_more_points
  (Beth_score : ℕ)
  (Jan_score : ℕ)
  (Judy_score : ℕ)
  (Angel_score : ℕ)
  (First_team_total : ℕ := Beth_score + Jan_score)
  (Second_team_total : ℕ := Judy_score + Angel_score)
  (Beth_score_val : Beth_score = 12)
  (Jan_score_val : Jan_score = 10)
  (Judy_score_val : Judy_score = 8)
  (Angel_score_val : Angel_score = 11)
  : First_team_total - Second_team_total = 3 := by
  sorry

end NUMINAMATH_GPT_first_team_more_points_l394_39484


namespace NUMINAMATH_GPT_conversion_7_dms_to_cms_conversion_5_hectares_to_sms_conversion_600_hectares_to_sqkms_conversion_200_sqsmeters_to_smeters_l394_39469

theorem conversion_7_dms_to_cms :
  7 * 100 = 700 :=
by
  sorry

theorem conversion_5_hectares_to_sms :
  5 * 10000 = 50000 :=
by
  sorry

theorem conversion_600_hectares_to_sqkms :
  600 / 100 = 6 :=
by
  sorry

theorem conversion_200_sqsmeters_to_smeters :
  200 / 100 = 2 :=
by
  sorry

end NUMINAMATH_GPT_conversion_7_dms_to_cms_conversion_5_hectares_to_sms_conversion_600_hectares_to_sqkms_conversion_200_sqsmeters_to_smeters_l394_39469


namespace NUMINAMATH_GPT_smallest_circle_equation_l394_39477

-- Definitions of the points A and B
def A : ℝ × ℝ := (-3, 0)
def B : ℝ × ℝ := (3, 0)

-- The statement of the problem
theorem smallest_circle_equation : ∃ (h k r : ℝ), (x - h)^2 + (y - k)^2 = r^2 ∧ 
  A.1 = -3 ∧ A.2 = 0 ∧ B.1 = 3 ∧ B.2 = 0 ∧ ((x - 0)^2 + (y - 0)^2 = 9) :=
by
  sorry

end NUMINAMATH_GPT_smallest_circle_equation_l394_39477


namespace NUMINAMATH_GPT_wrapping_paper_cost_l394_39464

theorem wrapping_paper_cost :
  let cost_design1 := 4 * 4 -- 20 shirt boxes / 5 shirt boxes per roll * $4.00 per roll
  let cost_design2 := 3 * 8 -- 12 XL boxes / 4 XL boxes per roll * $8.00 per roll
  let cost_design3 := 3 * 12-- 6 XXL boxes / 2 XXL boxes per roll * $12.00 per roll
  cost_design1 + cost_design2 + cost_design3 = 76
:= by
  -- Definitions
  let cost_design1 := 4 * 4
  let cost_design2 := 3 * 8
  let cost_design3 := 3 * 12
  -- Proof (To be implemented)
  sorry

end NUMINAMATH_GPT_wrapping_paper_cost_l394_39464


namespace NUMINAMATH_GPT_opposite_of_neg_2_l394_39416

theorem opposite_of_neg_2 : ∃ y : ℝ, -2 + y = 0 ∧ y = 2 := by
  sorry

end NUMINAMATH_GPT_opposite_of_neg_2_l394_39416


namespace NUMINAMATH_GPT_linear_inequalities_solution_l394_39460

variable (x : ℝ)

theorem linear_inequalities_solution 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 3 < x ∧ x < 4 := 
by
  sorry

end NUMINAMATH_GPT_linear_inequalities_solution_l394_39460


namespace NUMINAMATH_GPT_smallest_q_p_l394_39451

noncomputable def q_p_difference : ℕ := 3

theorem smallest_q_p (p q : ℕ) (hp : 0 < p) (hq : 0 < q) (h1 : 5 * q < 9 * p) (h2 : 9 * p < 5 * q) : q - p = q_p_difference → q = 7 :=
by
  sorry

end NUMINAMATH_GPT_smallest_q_p_l394_39451


namespace NUMINAMATH_GPT_mulberry_sales_l394_39411

theorem mulberry_sales (x : ℝ) (p : ℝ) (h1 : 3000 = x * p)
    (h2 : 150 * (p * 1.4) + (x - 150) * (p * 0.8) - 3000 = 750) :
    x = 200 := by sorry

end NUMINAMATH_GPT_mulberry_sales_l394_39411


namespace NUMINAMATH_GPT_average_difference_l394_39447

theorem average_difference :
  let a1 := 20
  let a2 := 40
  let a3 := 60
  let b1 := 10
  let b2 := 70
  let b3 := 13
  (a1 + a2 + a3) / 3 - (b1 + b2 + b3) / 3 = 9 := by
sorry

end NUMINAMATH_GPT_average_difference_l394_39447


namespace NUMINAMATH_GPT_find_a_and_b_l394_39401

theorem find_a_and_b (a b : ℝ) :
  {-1, 3} = {x : ℝ | x^2 + a * x + b = 0} ↔ a = -2 ∧ b = -3 :=
by 
  sorry

end NUMINAMATH_GPT_find_a_and_b_l394_39401


namespace NUMINAMATH_GPT_max_pawns_l394_39498

def chessboard : Type := ℕ × ℕ -- Define a chessboard as a grid of positions (1,1) to (8,8)
def e4 : chessboard := (5, 4) -- Define the position e4
def symmetric_wrt_e4 (p1 p2 : chessboard) : Prop :=
  p1.1 + p2.1 = 10 ∧ p1.2 + p2.2 = 8 -- Symmetry condition relative to e4

def placed_on (pos : chessboard) : Prop := sorry -- placeholder for placement condition

theorem max_pawns (no_e4 : ¬ placed_on e4)
  (no_symmetric_pairs : ∀ p1 p2, symmetric_wrt_e4 p1 p2 → ¬ (placed_on p1 ∧ placed_on p2)) :
  ∃ max_pawns : ℕ, max_pawns = 39 :=
sorry

end NUMINAMATH_GPT_max_pawns_l394_39498


namespace NUMINAMATH_GPT_units_digit_42_pow_4_add_24_pow_4_l394_39441

-- Define a function to get the units digit of a number.
def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_42_pow_4_add_24_pow_4 : units_digit (42^4 + 24^4) = 2 := by
  sorry

end NUMINAMATH_GPT_units_digit_42_pow_4_add_24_pow_4_l394_39441


namespace NUMINAMATH_GPT_units_digit_7_pow_5_pow_3_l394_39456

theorem units_digit_7_pow_5_pow_3 : (7 ^ (5 ^ 3)) % 10 = 7 := by
  sorry

end NUMINAMATH_GPT_units_digit_7_pow_5_pow_3_l394_39456


namespace NUMINAMATH_GPT_sin_double_theta_eq_three_fourths_l394_39414

theorem sin_double_theta_eq_three_fourths (θ : ℝ) (hθ : 0 < θ ∧ θ < π / 2)
  (h1 : Real.sin (π * Real.cos θ) = Real.cos (π * Real.sin θ)) :
  Real.sin (2 * θ) = 3 / 4 :=
  sorry

end NUMINAMATH_GPT_sin_double_theta_eq_three_fourths_l394_39414


namespace NUMINAMATH_GPT_rabbit_can_escape_l394_39400

def RabbitEscapeExists
  (center_x : ℝ)
  (center_y : ℝ)
  (wolf_x1 wolf_y1 wolf_x2 wolf_y2 wolf_x3 wolf_y3 wolf_x4 wolf_y4 : ℝ)
  (wolf_speed rabbit_speed : ℝ)
  (condition1 : center_x = 0 ∧ center_y = 0)
  (condition2 : wolf_x1 = -1 ∧ wolf_y1 = -1 ∧ wolf_x2 = 1 ∧ wolf_y2 = -1 ∧ wolf_x3 = -1 ∧ wolf_y3 = 1 ∧ wolf_x4 = 1 ∧ wolf_y4 = 1)
  (condition3 : wolf_speed = 1.4 * rabbit_speed) : Prop :=
 ∃ (rabbit_escapes : Bool), rabbit_escapes = true

theorem rabbit_can_escape
  (center_x : ℝ)
  (center_y : ℝ)
  (wolf_x1 wolf_y1 wolf_x2 wolf_y2 wolf_x3 wolf_y3 wolf_x4 wolf_y4 : ℝ)
  (wolf_speed rabbit_speed : ℝ)
  (condition1 : center_x = 0 ∧ center_y = 0)
  (condition2 : wolf_x1 = -1 ∧ wolf_y1 = -1 ∧ wolf_x2 = 1 ∧ wolf_y2 = -1 ∧ wolf_x3 = -1 ∧ wolf_y3 = 1 ∧ wolf_x4 = 1 ∧ wolf_y4 = 1)
  (condition3 : wolf_speed = 1.4 * rabbit_speed) : RabbitEscapeExists center_x center_y wolf_x1 wolf_y1 wolf_x2 wolf_y2 wolf_x3 wolf_y3 wolf_x4 wolf_y4 wolf_speed rabbit_speed condition1 condition2 condition3 := 
sorry

end NUMINAMATH_GPT_rabbit_can_escape_l394_39400


namespace NUMINAMATH_GPT_equation_of_circle_O2_equation_of_tangent_line_l394_39465

-- Define circle O1
def circle_O1 (x y : ℝ) : Prop :=
  x^2 + (y + 1)^2 = 4

-- Define the center and radius of circle O2 given that they are externally tangent
def center_O2 : ℝ × ℝ := (3, 3)
def radius_O2 : ℝ := 3

-- Prove the equation of circle O2
theorem equation_of_circle_O2 :
  ∀ (x y : ℝ), (x - 3)^2 + (y - 3)^2 = 9 := by
  intro x y
  sorry

-- Prove the equation of the common internal tangent line to circles O1 and O2
theorem equation_of_tangent_line :
  ∀ (x y : ℝ), 3 * x + 4 * y - 21 = 0 := by
  intro x y
  sorry

end NUMINAMATH_GPT_equation_of_circle_O2_equation_of_tangent_line_l394_39465


namespace NUMINAMATH_GPT_number_of_combinations_l394_39494

-- Define the binomial coefficient (combinations) function
def C (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

-- Our main theorem statement
theorem number_of_combinations (n k m : ℕ) (h1 : 1 ≤ n) (h2 : m > 1) :
  let valid_combinations := C (n - (k - 1) * (m - 1)) k;
  let invalid_combinations := n - (k - 1) * m;
  valid_combinations - invalid_combinations = 
  C (n - (k - 1) * (m - 1)) k - (n - (k - 1) * m) := by
  let valid_combinations := C (n - (k - 1) * (m - 1)) k
  let invalid_combinations := n - (k - 1) * m
  sorry

end NUMINAMATH_GPT_number_of_combinations_l394_39494


namespace NUMINAMATH_GPT_wam_gm_gt_hm_l394_39478

noncomputable def wam (w v a b : ℝ) : ℝ := w * a + v * b
noncomputable def gm (a b : ℝ) : ℝ := Real.sqrt (a * b)
noncomputable def hm (a b : ℝ) : ℝ := (2 * a * b) / (a + b)

theorem wam_gm_gt_hm
  (a b w v : ℝ)
  (h1 : 0 < a ∧ 0 < b)
  (h2 : 0 < w ∧ 0 < v)
  (h3 : w + v = 1)
  (h4 : a ≠ b) :
  wam w v a b > gm a b ∧ gm a b > hm a b :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_wam_gm_gt_hm_l394_39478


namespace NUMINAMATH_GPT_disjoint_subsets_with_same_sum_l394_39488

theorem disjoint_subsets_with_same_sum :
  ∀ (S : Finset ℕ), S.card = 10 ∧ (∀ x ∈ S, x ∈ Finset.range 101) →
  ∃ A B : Finset ℕ, A ⊆ S ∧ B ⊆ S ∧ A ∩ B = ∅ ∧ A.sum id = B.sum id :=
by
  sorry

end NUMINAMATH_GPT_disjoint_subsets_with_same_sum_l394_39488


namespace NUMINAMATH_GPT_sum_of_squares_of_consecutive_integers_l394_39468

theorem sum_of_squares_of_consecutive_integers (a : ℕ) (h : (a - 1) * a * (a + 1) * (a + 2) = 12 * ((a - 1) + a + (a + 1) + (a + 2))) :
  (a - 1)^2 + a^2 + (a + 1)^2 + (a + 2)^2 = 86 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_consecutive_integers_l394_39468


namespace NUMINAMATH_GPT_max_groups_l394_39493

-- Define the conditions
def valid_eq (a b : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ (3 * a + b = 13)

-- The proof problem: No need for the proof body, just statement
theorem max_groups : ∃! (l : List (ℕ × ℕ)), (∀ ab ∈ l, valid_eq ab.fst ab.snd) ∧ l.length = 3 := sorry

end NUMINAMATH_GPT_max_groups_l394_39493


namespace NUMINAMATH_GPT_discount_equivalence_l394_39438

theorem discount_equivalence :
  ∀ (p d1 d2 : ℝ) (d : ℝ),
    p = 800 →
    d1 = 0.15 →
    d2 = 0.10 →
    p * (1 - d1) * (1 - d2) = p * (1 - d) →
    d = 0.235 := by
  intros p d1 d2 d hp hd1 hd2 heq
  sorry

end NUMINAMATH_GPT_discount_equivalence_l394_39438


namespace NUMINAMATH_GPT_train_speed_in_kmh_l394_39426

def train_length : ℝ := 250 -- Length of the train in meters
def station_length : ℝ := 200 -- Length of the station in meters
def time_to_pass : ℝ := 45 -- Time to pass the station in seconds

theorem train_speed_in_kmh :
  (train_length + station_length) / time_to_pass * 3.6 = 36 :=
  sorry -- Proof is skipped

end NUMINAMATH_GPT_train_speed_in_kmh_l394_39426


namespace NUMINAMATH_GPT_number_of_solutions_l394_39453

noncomputable def f (x : ℝ) : ℝ := 3 * x^4 - 4 * x^3 - 12 * x^2 + 12

theorem number_of_solutions : ∃ x1 x2, f x1 = 0 ∧ f x2 = 0 ∧ x1 ≠ x2 ∧
  ∀ x, f x = 0 → (x = x1 ∨ x = x2) :=
by
  sorry

end NUMINAMATH_GPT_number_of_solutions_l394_39453


namespace NUMINAMATH_GPT_solve_quadratic_eq_l394_39492

theorem solve_quadratic_eq (x : ℝ) (h : (x + 5) ^ 2 = 16) : x = -1 ∨ x = -9 :=
sorry

end NUMINAMATH_GPT_solve_quadratic_eq_l394_39492


namespace NUMINAMATH_GPT_games_left_is_correct_l394_39483

-- Define the initial number of DS games
def initial_games : ℕ := 98

-- Define the number of games given away
def games_given_away : ℕ := 7

-- Define the number of games left
def games_left : ℕ := initial_games - games_given_away

-- Theorem statement to prove that the number of games left is 91
theorem games_left_is_correct : games_left = 91 :=
by
  -- Currently, we use sorry to skip the actual proof part.
  sorry

end NUMINAMATH_GPT_games_left_is_correct_l394_39483


namespace NUMINAMATH_GPT_sequence_formula_min_value_Sn_min_value_Sn_completion_l394_39419

-- Define the sequence sum Sn
def Sn (n : ℕ) : ℤ := n^2 - 48 * n

-- General term of the sequence
def an (n : ℕ) : ℤ :=
  match n with
  | 0     => 0 -- Conventionally, sequences start from 1 in these problems
  | (n+1) => 2 * (n + 1) - 49

-- Prove that the general term of the sequence produces the correct sum
theorem sequence_formula (n : ℕ) (h : 0 < n) : an n = 2 * n - 49 := by
  sorry

-- Prove that the minimum value of Sn is -576 and occurs at n = 24
theorem min_value_Sn : ∃ n : ℕ, Sn n = -576 ∧ ∀ m : ℕ, Sn m ≥ -576 := by
  use 24
  sorry

-- Alternative form of the theorem using the square completion form 
theorem min_value_Sn_completion (n : ℕ) : Sn n = (n - 24)^2 - 576 := by
  sorry

end NUMINAMATH_GPT_sequence_formula_min_value_Sn_min_value_Sn_completion_l394_39419


namespace NUMINAMATH_GPT_num_of_ordered_pairs_l394_39405

theorem num_of_ordered_pairs (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : b > a)
(h4 : (a-2)*(b-2) = (ab / 2)) : (a, b) = (5, 12) ∨ (a, b) = (6, 8) :=
by
  sorry

end NUMINAMATH_GPT_num_of_ordered_pairs_l394_39405


namespace NUMINAMATH_GPT_find_value_of_expression_l394_39425

theorem find_value_of_expression (y : ℝ) (h : 6 * y^2 + 7 = 4 * y + 13) : (12 * y - 5)^2 = 161 :=
sorry

end NUMINAMATH_GPT_find_value_of_expression_l394_39425


namespace NUMINAMATH_GPT_time_to_fill_pool_l394_39457

variable (pool_volume : ℝ) (fill_rate : ℝ) (leak_rate : ℝ)

theorem time_to_fill_pool (h_pool_volume : pool_volume = 60)
    (h_fill_rate : fill_rate = 1.6)
    (h_leak_rate : leak_rate = 0.1) :
    (pool_volume / (fill_rate - leak_rate)) = 40 :=
by
  -- We skip the proof step, only the theorem statement is required
  sorry

end NUMINAMATH_GPT_time_to_fill_pool_l394_39457


namespace NUMINAMATH_GPT_unique_games_count_l394_39422

noncomputable def total_games_played (n : ℕ) (m : ℕ) : ℕ :=
  (n * m) / 2

theorem unique_games_count (students : ℕ) (games_per_student : ℕ) (h1 : students = 9) (h2 : games_per_student = 6) :
  total_games_played students games_per_student = 27 :=
by
  rw [h1, h2]
  -- This partially evaluates total_games_played using the values from h1 and h2.
  -- Performing actual proof steps is not necessary, so we'll use sorry.
  sorry

end NUMINAMATH_GPT_unique_games_count_l394_39422


namespace NUMINAMATH_GPT_number_of_fish_bought_each_year_l394_39408

-- Define the conditions
def initial_fish : ℕ := 2
def net_gain_each_year (x : ℕ) : ℕ := x - 1
def years : ℕ := 5
def final_fish : ℕ := 7

-- Define the problem statement as a Lean theorem
theorem number_of_fish_bought_each_year (x : ℕ) : 
  initial_fish + years * net_gain_each_year x = final_fish → x = 2 := 
sorry

end NUMINAMATH_GPT_number_of_fish_bought_each_year_l394_39408


namespace NUMINAMATH_GPT_coastal_city_spending_l394_39434

def beginning_of_may_spending : ℝ := 1.2
def end_of_september_spending : ℝ := 4.5

theorem coastal_city_spending :
  (end_of_september_spending - beginning_of_may_spending) = 3.3 :=
by
  -- Proof can be filled in here
  sorry

end NUMINAMATH_GPT_coastal_city_spending_l394_39434


namespace NUMINAMATH_GPT_find_coordinates_of_P_l394_39443

structure Point where
  x : Int
  y : Int

def symmetric_origin (A B : Point) : Prop :=
  B.x = -A.x ∧ B.y = -A.y

def symmetric_y_axis (A B : Point) : Prop :=
  B.x = -A.x ∧ B.y = A.y

theorem find_coordinates_of_P :
  ∀ M N P : Point, 
  M = Point.mk (-4) 3 →
  symmetric_origin M N →
  symmetric_y_axis N P →
  P = Point.mk 4 3 := 
by 
  intros M N P hM hSymN hSymP
  sorry

end NUMINAMATH_GPT_find_coordinates_of_P_l394_39443


namespace NUMINAMATH_GPT_problem_l394_39423

noncomputable def f (x a b : ℝ) := x^2 + a*x + b
noncomputable def g (x c d : ℝ) := x^2 + c*x + d

theorem problem (a b c d : ℝ) (h_min_f : f (-a/2) a b = -25) (h_min_g : g (-c/2) c d = -25)
  (h_intersection_f : f 50 a b = -50) (h_intersection_g : g 50 c d = -50)
  (h_root_f_of_g : g (-a/2) c d = 0) (h_root_g_of_f : f (-c/2) a b = 0) :
  a + c = -200 := by
  sorry

end NUMINAMATH_GPT_problem_l394_39423


namespace NUMINAMATH_GPT_perp_condition_l394_39470

def a (x : ℝ) : ℝ × ℝ := (x-1, 2)
def b : ℝ × ℝ := (2, 1)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem perp_condition (x : ℝ) : dot_product (a x) b = 0 ↔ x = 0 :=
by 
  sorry

end NUMINAMATH_GPT_perp_condition_l394_39470


namespace NUMINAMATH_GPT_drink_total_amount_l394_39489

theorem drink_total_amount (total_amount: ℝ) (grape_juice: ℝ) (grape_proportion: ℝ) 
  (h1: grape_proportion = 0.20) (h2: grape_juice = 40) : total_amount = 200 :=
by
  -- Definitions and assumptions
  let calculation := grape_juice / grape_proportion
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_drink_total_amount_l394_39489


namespace NUMINAMATH_GPT_quadratic_real_roots_and_a_value_l394_39428

-- Define the quadratic equation (a-5)x^2 - 4x - 1 = 0
def quadratic_eq (a : ℝ) (x : ℝ) := (a - 5) * x^2 - 4 * x - 1

-- Define the discriminant for the quadratic equation
def discriminant (a : ℝ) := 4 - 4 * (a - 5) * (-1)

-- Main theorem statement
theorem quadratic_real_roots_and_a_value
    (a : ℝ) (x1 x2 : ℝ) 
    (h_roots : (a - 5) ≠ 0)
    (h_eq : quadratic_eq a x1 = 0 ∧ quadratic_eq a x2 = 0)
    (h_sum_product : x1 + x2 + x1 * x2 = 3) :
    (a ≥ 1) ∧ (a = 6) :=
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_and_a_value_l394_39428


namespace NUMINAMATH_GPT_sum_of_consecutive_2022_l394_39432

theorem sum_of_consecutive_2022 (m n : ℕ) (h : m ≤ n - 1) (sum_eq : (n - m + 1) * (m + n) = 4044) :
  (m = 163 ∧ n = 174) ∨ (m = 504 ∧ n = 507) ∨ (m = 673 ∧ n = 675) :=
sorry

end NUMINAMATH_GPT_sum_of_consecutive_2022_l394_39432


namespace NUMINAMATH_GPT_roots_inequality_l394_39433

noncomputable def a : ℝ := Real.sqrt 2020

theorem roots_inequality (x1 x2 x3 : ℝ) (h_roots : ∀ x, (a * x^3 - 4040 * x^2 + 4 = 0) ↔ (x = x1 ∨ x = x2 ∨ x = x3))
  (h_inequality: x1 < x2 ∧ x2 < x3) : x2 * (x1 + x3) = 2 :=
sorry

end NUMINAMATH_GPT_roots_inequality_l394_39433


namespace NUMINAMATH_GPT_journey_speed_l394_39499

theorem journey_speed (v : ℚ) 
  (equal_distance : ∀ {d}, (d = 0.22) → ((0.66 / 3) = d))
  (total_distance : ∀ {d}, (d = 660 / 1000) → (660 / 1000 = 0.66))
  (total_time : ∀ {t} , (t = 11 / 60) → (11 / 60 = t)): 
  (0.22 / 2 + 0.22 / v + 0.22 / 6 = 11 / 60) → v = 1.2 := 
by 
  sorry

end NUMINAMATH_GPT_journey_speed_l394_39499


namespace NUMINAMATH_GPT_complex_expression_ab_l394_39472

open Complex

theorem complex_expression_ab :
  ∀ (a b : ℝ), (2 + 3 * I) / I = a + b * I → a * b = 6 :=
by
  intros a b h
  sorry

end NUMINAMATH_GPT_complex_expression_ab_l394_39472


namespace NUMINAMATH_GPT_min_B_minus_A_l394_39424

noncomputable def S_n (n : ℕ) : ℚ :=
  let a1 : ℚ := 2
  let r : ℚ := -1 / 3
  a1 * (1 - r ^ n) / (1 - r)

theorem min_B_minus_A :
  ∃ A B : ℚ, 
    (∀ n : ℕ, 1 ≤ n → A ≤ 3 * S_n n - 1 / S_n n ∧ 3 * S_n n - 1 / S_n n ≤ B) ∧
    ∀ A' B' : ℚ, 
      (∀ n : ℕ, 1 ≤ n → A' ≤ 3 * S_n n - 1 / S_n n ∧ 3 * S_n n - 1 / S_n n ≤ B') → 
      B' - A' ≥ 9 / 4 ∧ B - A = 9 / 4 :=
sorry

end NUMINAMATH_GPT_min_B_minus_A_l394_39424


namespace NUMINAMATH_GPT_matching_pairs_less_than_21_in_at_least_61_positions_l394_39406

theorem matching_pairs_less_than_21_in_at_least_61_positions :
  ∀ (disks : ℕ) (total_sectors : ℕ) (red_sectors : ℕ) (max_overlap : ℕ) (rotations : ℕ),
  disks = 2 →
  total_sectors = 1965 →
  red_sectors = 200 →
  max_overlap = 20 →
  rotations = total_sectors →
  (∃ positions, positions = total_sectors - (red_sectors * red_sectors / (max_overlap + 1)) ∧ positions ≤ rotations) →
  positions = 61 :=
by {
  -- Placeholder to provide the structure of the theorem.
  sorry
}

end NUMINAMATH_GPT_matching_pairs_less_than_21_in_at_least_61_positions_l394_39406


namespace NUMINAMATH_GPT_race_duration_l394_39496

theorem race_duration 
  (lap_distance : ℕ) (laps : ℕ)
  (award_per_hundred_meters : ℝ) (earn_rate_per_minute : ℝ)
  (total_distance : ℕ) (total_award : ℝ) (duration : ℝ) :
  lap_distance = 100 →
  laps = 24 →
  award_per_hundred_meters = 3.5 →
  earn_rate_per_minute = 7 →
  total_distance = lap_distance * laps →
  total_award = (total_distance / 100) * award_per_hundred_meters →
  duration = total_award / earn_rate_per_minute →
  duration = 12 := 
by 
  intros;
  sorry

end NUMINAMATH_GPT_race_duration_l394_39496


namespace NUMINAMATH_GPT_decimal_expansion_of_fraction_l394_39442

/-- 
Theorem: The decimal expansion of 13 / 375 is 0.034666...
-/
theorem decimal_expansion_of_fraction : 
  let numerator := 13
  let denominator := 375
  let resulting_fraction := (numerator * 2^3) / (denominator * 2^3)
  let decimal_expansion := 0.03466666666666667
  (resulting_fraction : ℝ) = decimal_expansion :=
sorry

end NUMINAMATH_GPT_decimal_expansion_of_fraction_l394_39442


namespace NUMINAMATH_GPT_john_total_beats_l394_39445

noncomputable def minutes_in_hour : ℕ := 60
noncomputable def hours_per_day : ℕ := 2
noncomputable def days_played : ℕ := 3
noncomputable def beats_per_minute : ℕ := 200

theorem john_total_beats :
  (beats_per_minute * hours_per_day * minutes_in_hour * days_played) = 72000 :=
by
  -- we will implement the proof here
  sorry

end NUMINAMATH_GPT_john_total_beats_l394_39445


namespace NUMINAMATH_GPT_initial_bushes_l394_39421

theorem initial_bushes (b : ℕ) (h1 : b + 4 = 6) : b = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_initial_bushes_l394_39421


namespace NUMINAMATH_GPT_elena_pens_l394_39415

theorem elena_pens (X Y : ℝ) 
  (h1 : X + Y = 12) 
  (h2 : 4 * X + 2.80 * Y = 40) :
  X = 5 :=
by
  sorry

end NUMINAMATH_GPT_elena_pens_l394_39415


namespace NUMINAMATH_GPT_ratio_of_a_b_to_b_c_l394_39404

theorem ratio_of_a_b_to_b_c (a b c : ℝ) (h₁ : b / a = 3) (h₂ : c / b = 2) : 
  (a + b) / (b + c) = 4 / 9 := by
  sorry

end NUMINAMATH_GPT_ratio_of_a_b_to_b_c_l394_39404


namespace NUMINAMATH_GPT_solve_investment_problem_l394_39480

def investment_problem
  (total_investment : ℝ) (etf_investment : ℝ) (mutual_funds_factor : ℝ) (mutual_funds_investment : ℝ) : Prop :=
  total_investment = etf_investment + mutual_funds_factor * etf_investment →
  mutual_funds_factor * etf_investment = mutual_funds_investment

theorem solve_investment_problem :
  investment_problem 210000 46666.67 3.5 163333.35 :=
by
  sorry

end NUMINAMATH_GPT_solve_investment_problem_l394_39480


namespace NUMINAMATH_GPT_five_n_minus_twelve_mod_nine_l394_39437

theorem five_n_minus_twelve_mod_nine (n : ℤ) (h : n % 9 = 4) : (5 * n - 12) % 9 = 8 := by
  sorry

end NUMINAMATH_GPT_five_n_minus_twelve_mod_nine_l394_39437


namespace NUMINAMATH_GPT_greatest_possible_sum_of_two_consecutive_integers_product_lt_1000_l394_39495

theorem greatest_possible_sum_of_two_consecutive_integers_product_lt_1000 : 
  ∃ n : ℤ, (n * (n + 1) < 1000) ∧ (n + (n + 1) = 63) :=
sorry

end NUMINAMATH_GPT_greatest_possible_sum_of_two_consecutive_integers_product_lt_1000_l394_39495
