import Mathlib

namespace NUMINAMATH_GPT_rectangle_tileable_iff_divisible_l643_64338

def divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, b = k * a

def tileable_with_0b_tiles (m n b : ℕ) : Prop :=
  ∃ t : ℕ, t * (2 * b) = m * n  -- This comes from the total area divided by the area of one tile

theorem rectangle_tileable_iff_divisible (m n b : ℕ) :
  tileable_with_0b_tiles m n b ↔ divisible_by (2 * b) m ∨ divisible_by (2 * b) n := 
sorry

end NUMINAMATH_GPT_rectangle_tileable_iff_divisible_l643_64338


namespace NUMINAMATH_GPT_slips_with_number_three_l643_64399

theorem slips_with_number_three : 
  ∀ (total_slips : ℕ) (number3 number8 : ℕ) (E : ℚ), 
  total_slips = 15 → 
  E = 5.6 → 
  number3 + number8 = total_slips → 
  (number3 : ℚ) / total_slips * 3 + (number8 : ℚ) / total_slips * 8 = E →
  number3 = 8 :=
by
  intros total_slips number3 number8 E h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_slips_with_number_three_l643_64399


namespace NUMINAMATH_GPT_jane_mistake_corrected_l643_64375

-- Conditions translated to Lean definitions
variables (x y z : ℤ)
variable (h1 : x - (y + z) = 15)
variable (h2 : x - y + z = 7)

-- Statement to prove
theorem jane_mistake_corrected : x - y = 11 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_jane_mistake_corrected_l643_64375


namespace NUMINAMATH_GPT_translate_parabola_l643_64382

theorem translate_parabola :
  (∀ x, y = 1/2 * x^2 + 1 → y = 1/2 * (x - 1)^2 - 2) :=
by
  sorry

end NUMINAMATH_GPT_translate_parabola_l643_64382


namespace NUMINAMATH_GPT_find_max_value_l643_64329

noncomputable def maximum_value (x y z : ℝ) : ℝ :=
  2 * x * y * Real.sqrt 3 + 3 * y * z * Real.sqrt 2 + 3 * z * x

theorem find_max_value (x y z : ℝ) (h₀ : 0 ≤ x) (h₁ : 0 ≤ y) (h₂ : 0 ≤ z)
  (h₃ : x^2 + y^2 + z^2 = 1) : 
  maximum_value x y z ≤ Real.sqrt 3 := sorry

end NUMINAMATH_GPT_find_max_value_l643_64329


namespace NUMINAMATH_GPT_johns_total_expenditure_l643_64332

-- Conditions
def treats_first_15_days : ℕ := 3 * 15
def treats_next_15_days : ℕ := 4 * 15
def total_treats : ℕ := treats_first_15_days + treats_next_15_days
def cost_per_treat : ℝ := 0.10
def discount_threshold : ℕ := 50
def discount_rate : ℝ := 0.10

-- Intermediate calculations
def total_cost_without_discount : ℝ := total_treats * cost_per_treat
def discounted_cost_per_treat : ℝ := cost_per_treat * (1 - discount_rate)
def total_cost_with_discount : ℝ := total_treats * discounted_cost_per_treat

-- Main theorem statement
theorem johns_total_expenditure : total_cost_with_discount = 9.45 :=
by
  -- Place proof here
  sorry

end NUMINAMATH_GPT_johns_total_expenditure_l643_64332


namespace NUMINAMATH_GPT_max_quarters_l643_64364

theorem max_quarters (total_value : ℝ) (n_quarters n_nickels n_dimes : ℕ) 
  (h1 : n_nickels = n_quarters) 
  (h2 : n_dimes = 2 * n_quarters)
  (h3 : 0.25 * n_quarters + 0.05 * n_nickels + 0.10 * n_dimes = total_value)
  (h4 : total_value = 3.80) : 
  n_quarters = 7 := 
by
  sorry

end NUMINAMATH_GPT_max_quarters_l643_64364


namespace NUMINAMATH_GPT_lcm_16_24_45_l643_64395

-- Define the numbers
def a : Nat := 16
def b : Nat := 24
def c : Nat := 45

-- State the theorem that the least common multiple of these numbers is 720
theorem lcm_16_24_45 : Nat.lcm (Nat.lcm 16 24) 45 = 720 := by
  sorry

end NUMINAMATH_GPT_lcm_16_24_45_l643_64395


namespace NUMINAMATH_GPT_bonnie_roark_wire_length_ratio_l643_64379

-- Define the conditions
def bonnie_wire_pieces : ℕ := 12
def bonnie_wire_length_per_piece : ℕ := 8
def roark_wire_length_per_piece : ℕ := 2
def bonnie_cube_volume : ℕ := 8 * 8 * 8
def roark_total_cube_volume : ℕ := bonnie_cube_volume
def roark_unit_cube_volume : ℕ := 1
def roark_unit_cube_wires : ℕ := 12

-- Calculate Bonnie's total wire length
noncomputable def bonnie_total_wire_length : ℕ := bonnie_wire_pieces * bonnie_wire_length_per_piece

-- Calculate the number of Roark's unit cubes
noncomputable def roark_number_of_unit_cubes : ℕ := roark_total_cube_volume / roark_unit_cube_volume

-- Calculate the total wire used by Roark
noncomputable def roark_total_wire_length : ℕ := roark_number_of_unit_cubes * roark_unit_cube_wires * roark_wire_length_per_piece

-- Calculate the ratio of Bonnie's total wire length to Roark's total wire length
noncomputable def wire_length_ratio : ℚ := bonnie_total_wire_length / roark_total_wire_length

-- State the theorem
theorem bonnie_roark_wire_length_ratio : wire_length_ratio = 1 / 128 := 
by 
  sorry

end NUMINAMATH_GPT_bonnie_roark_wire_length_ratio_l643_64379


namespace NUMINAMATH_GPT_problem_solution_l643_64345

theorem problem_solution (a b c : ℝ) 
  (h1 : ∀ x, (x < -6 ∨ |x - 30| ≤ 2) ↔ ( (x - a) * (x - b) / (x - c) ≤ 0 ))
  (h2 : a < b)
  : a + 2 * b + 3 * c = 74 := 
sorry

end NUMINAMATH_GPT_problem_solution_l643_64345


namespace NUMINAMATH_GPT_find_antonym_word_l643_64321

-- Defining the condition that the word means "rarely" or "not often."
def means_rarely_or_not_often (word : String) : Prop :=
  word = "seldom"

-- Theorem statement: There exists a word such that it meets the given condition.
theorem find_antonym_word : 
  ∃ word : String, means_rarely_or_not_often word :=
by
  use "seldom"
  unfold means_rarely_or_not_often
  rfl

end NUMINAMATH_GPT_find_antonym_word_l643_64321


namespace NUMINAMATH_GPT_relationship_a_b_l643_64330

theorem relationship_a_b (a b : ℝ) :
  (∃ (P : ℝ × ℝ), P ∈ {Q : ℝ × ℝ | Q.snd = -3 * Q.fst + b} ∧
                   ∃ (R : ℝ × ℝ), R ∈ {S : ℝ × ℝ | S.snd = -a * S.fst + 3} ∧
                   R = (-P.snd, -P.fst)) →
  a = 1 / 3 ∧ b = -9 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_relationship_a_b_l643_64330


namespace NUMINAMATH_GPT_minimum_value_proof_l643_64392

noncomputable def minimum_value_condition (a b : ℝ) : Prop :=
a > 0 ∧ b > 0 ∧ a + b = 12

theorem minimum_value_proof : ∀ (a b : ℝ), minimum_value_condition a b → (1 / a + 1 / b ≥ 1 / 3) := 
by
  intros a b h
  sorry

end NUMINAMATH_GPT_minimum_value_proof_l643_64392


namespace NUMINAMATH_GPT_area_of_triangle_intercepts_l643_64328

theorem area_of_triangle_intercepts :
  let f := fun x => (x - 4)^2 * (x + 3)
  let x_intercepts := [4, -3]
  let y_intercept := f 0
  let vertices := [(4, 0), (-3, 0), (0, y_intercept)]
  let base := 4 - (-3)
  let height := y_intercept
  let area := (1 / 2) * base * height
  area = 168 :=
by
  let f := fun x => (x - 4)^2 * (x + 3)
  let x_intercepts := [4, -3]
  let y_intercept := f 0
  let vertices := [(4, 0), (-3, 0), (0, y_intercept)]
  let base := 4 - (-3)
  let height := y_intercept
  let area := (1 / 2) * base * height
  show area = 168
  sorry

end NUMINAMATH_GPT_area_of_triangle_intercepts_l643_64328


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l643_64304

def vectors_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem sufficient_but_not_necessary (m n : ℝ) :
  vectors_parallel (m, 1) (n, 1) ↔ (m = n) := sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l643_64304


namespace NUMINAMATH_GPT_calculation_l643_64300

theorem calculation : (3 * 4 * 5) * ((1 / 3 : ℚ) + (1 / 4 : ℚ) - (1 / 5 : ℚ)) = 23 := by
  sorry

end NUMINAMATH_GPT_calculation_l643_64300


namespace NUMINAMATH_GPT_find_n_l643_64323

-- Define the arithmetic sequence
def arithmetic_sequence (n : ℕ) : ℕ :=
  11 + (n - 1) * 6

-- State the problem
theorem find_n (n : ℕ) : 
  (∀ m : ℕ, m ≥ n → arithmetic_sequence m > 2017) ↔ n = 336 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l643_64323


namespace NUMINAMATH_GPT_proof_not_necessarily_15_points_l643_64367

-- Define the number of teams
def teams := 14

-- Define a tournament where each team plays every other exactly once
def games := (teams * (teams - 1)) / 2

-- Define a function calculating the total points by summing points for each game
def total_points (wins draws : ℕ) := (3 * wins) + (1 * draws)

-- Define a statement that total points is at least 150
def scores_sum_at_least_150 (wins draws : ℕ) : Prop :=
  total_points wins draws ≥ 150

-- Define a condition that a score could be less than 15
def highest_score_not_necessarily_15 : Prop :=
  ∃ (scores : Finset ℕ), scores.card = teams ∧ ∀ score ∈ scores, score < 15

theorem proof_not_necessarily_15_points :
  ∃ (wins draws : ℕ), wins + draws = games ∧ scores_sum_at_least_150 wins draws ∧ highest_score_not_necessarily_15 :=
by
  sorry

end NUMINAMATH_GPT_proof_not_necessarily_15_points_l643_64367


namespace NUMINAMATH_GPT_find_cost_price_l643_64383

theorem find_cost_price (C : ℝ) (h1 : 1.12 * C + 18 = 1.18 * C) : C = 300 :=
by
  sorry

end NUMINAMATH_GPT_find_cost_price_l643_64383


namespace NUMINAMATH_GPT_total_dollars_l643_64326

theorem total_dollars (john emma lucas : ℝ) 
  (h_john : john = 4 / 5) 
  (h_emma : emma = 2 / 5) 
  (h_lucas : lucas = 1 / 2) : 
  john + emma + lucas = 1.7 := by
  sorry

end NUMINAMATH_GPT_total_dollars_l643_64326


namespace NUMINAMATH_GPT_no_solution_for_k_eq_4_l643_64376

theorem no_solution_for_k_eq_4 (x k : ℝ) (h₁ : x ≠ 4) (h₂ : x ≠ 8) : (k = 4) → ¬ ((x - 3) * (x - 8) = (x - k) * (x - 4)) :=
by
  sorry

end NUMINAMATH_GPT_no_solution_for_k_eq_4_l643_64376


namespace NUMINAMATH_GPT_coeff_x3y5_in_expansion_l643_64350

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem coeff_x3y5_in_expansion (x y : ℕ) :
  (binomial_coefficient 8 5) = 56 := by
  sorry

end NUMINAMATH_GPT_coeff_x3y5_in_expansion_l643_64350


namespace NUMINAMATH_GPT_molecular_weight_of_3_moles_of_Fe2_SO4_3_l643_64311

noncomputable def mol_weight_fe : ℝ := 55.845
noncomputable def mol_weight_s : ℝ := 32.065
noncomputable def mol_weight_o : ℝ := 15.999

noncomputable def mol_weight_fe2_so4_3 : ℝ :=
  (2 * mol_weight_fe) + (3 * (mol_weight_s + (4 * mol_weight_o)))

theorem molecular_weight_of_3_moles_of_Fe2_SO4_3 :
  3 * mol_weight_fe2_so4_3 = 1199.619 := by
  sorry

end NUMINAMATH_GPT_molecular_weight_of_3_moles_of_Fe2_SO4_3_l643_64311


namespace NUMINAMATH_GPT_max_integer_values_correct_l643_64301

noncomputable def max_integer_values (a b c : ℝ) : ℕ :=
  if a > 100 then 2 else 0

theorem max_integer_values_correct (a b c : ℝ) (h : a > 100) :
  max_integer_values a b c = 2 :=
by sorry

end NUMINAMATH_GPT_max_integer_values_correct_l643_64301


namespace NUMINAMATH_GPT_g_of_square_sub_one_l643_64368

variable {R : Type*} [LinearOrderedField R]

def g (x : R) : R := 3

theorem g_of_square_sub_one (x : R) : g ((x - 1)^2) = 3 := 
by sorry

end NUMINAMATH_GPT_g_of_square_sub_one_l643_64368


namespace NUMINAMATH_GPT_smallest_yummy_integer_l643_64373

theorem smallest_yummy_integer :
  ∃ (n A : ℤ), 4046 = n * (2 * A + n - 1) ∧ A ≥ 0 ∧ (∀ m, 4046 = m * (2 * A + m - 1) ∧ m ≥ 0 → A ≤ 1011) :=
sorry

end NUMINAMATH_GPT_smallest_yummy_integer_l643_64373


namespace NUMINAMATH_GPT_total_points_scored_l643_64331

def num_members : ℕ := 12
def num_absent : ℕ := 4
def points_per_member : ℕ := 8

theorem total_points_scored : 
  (num_members - num_absent) * points_per_member = 64 := by
  sorry

end NUMINAMATH_GPT_total_points_scored_l643_64331


namespace NUMINAMATH_GPT_simplify_polynomial_l643_64361

theorem simplify_polynomial (q : ℤ) :
  (4*q^4 - 2*q^3 + 3*q^2 - 7*q + 9) + (5*q^3 - 8*q^2 + 6*q - 1) =
  4*q^4 + 3*q^3 - 5*q^2 - q + 8 :=
sorry

end NUMINAMATH_GPT_simplify_polynomial_l643_64361


namespace NUMINAMATH_GPT_area_of_square_with_given_diagonal_l643_64359

-- Definition of the conditions
def diagonal := 12
def s := Real
def area (s : Real) := s^2
def diag_relation (d s : Real) := d^2 = 2 * s^2

-- The proof statement
theorem area_of_square_with_given_diagonal :
  ∃ s : Real, diag_relation diagonal s ∧ area s = 72 :=
by
  sorry

end NUMINAMATH_GPT_area_of_square_with_given_diagonal_l643_64359


namespace NUMINAMATH_GPT_solve_fractional_equation_l643_64365

theorem solve_fractional_equation (x : ℝ) (hx1 : x ≠ 0) (hx2 : x ≠ 1) :
  (3 / (x^2 - x) + 1 = x / (x - 1)) → x = 3 :=
by
  sorry -- Placeholder for the actual proof

end NUMINAMATH_GPT_solve_fractional_equation_l643_64365


namespace NUMINAMATH_GPT_yuan_to_scientific_notation_l643_64393

/-- Express 2.175 billion yuan in scientific notation,
preserving three significant figures. --/
theorem yuan_to_scientific_notation (a : ℝ) (h : a = 2.175 * 10^9) : a = 2.18 * 10^9 :=
sorry

end NUMINAMATH_GPT_yuan_to_scientific_notation_l643_64393


namespace NUMINAMATH_GPT_thirtieth_triangular_number_sum_of_thirtieth_and_twentyninth_triangular_numbers_l643_64327

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem thirtieth_triangular_number : triangular_number 30 = 465 := 
by
  sorry

theorem sum_of_thirtieth_and_twentyninth_triangular_numbers : triangular_number 30 + triangular_number 29 = 900 := 
by
  sorry

end NUMINAMATH_GPT_thirtieth_triangular_number_sum_of_thirtieth_and_twentyninth_triangular_numbers_l643_64327


namespace NUMINAMATH_GPT_proposition_false_l643_64318

theorem proposition_false : ¬ ∀ x ∈ ({1, -1, 0} : Set ℤ), 2 * x + 1 > 0 := by
  sorry

end NUMINAMATH_GPT_proposition_false_l643_64318


namespace NUMINAMATH_GPT_cubic_expression_value_l643_64302

theorem cubic_expression_value (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2 * m^2 + 2010 = 2011 :=
by sorry

end NUMINAMATH_GPT_cubic_expression_value_l643_64302


namespace NUMINAMATH_GPT_carpooling_plans_l643_64391

def last_digits (jia : ℕ) (friend1 : ℕ) (friend2 : ℕ) (friend3 : ℕ) (friend4 : ℕ) : Prop :=
  jia = 0 ∧ friend1 = 0 ∧ friend2 = 2 ∧ friend3 = 1 ∧ friend4 = 5

def total_car_plans : Prop :=
  ∀ (jia friend1 friend2 friend3 friend4 : ℕ),
    last_digits jia friend1 friend2 friend3 friend4 →
    (∃ num_ways : ℕ, num_ways = 64)

theorem carpooling_plans : total_car_plans :=
sorry

end NUMINAMATH_GPT_carpooling_plans_l643_64391


namespace NUMINAMATH_GPT_boxes_per_class_l643_64319

variable (boxes : ℕ) (classes : ℕ)

theorem boxes_per_class (h1 : boxes = 3) (h2 : classes = 4) : 
  (boxes : ℚ) / (classes : ℚ) = 3 / 4 :=
by
  rw [h1, h2]
  norm_num

end NUMINAMATH_GPT_boxes_per_class_l643_64319


namespace NUMINAMATH_GPT_system_of_equations_solution_l643_64385

theorem system_of_equations_solution (x y z : ℝ) :
  (4 * x^2 / (1 + 4 * x^2) = y ∧
   4 * y^2 / (1 + 4 * y^2) = z ∧
   4 * z^2 / (1 + 4 * z^2) = x) →
  ((x = 1/2 ∧ y = 1/2 ∧ z = 1/2) ∨ (x = 0 ∧ y = 0 ∧ z = 0)) :=
by
  sorry

end NUMINAMATH_GPT_system_of_equations_solution_l643_64385


namespace NUMINAMATH_GPT_find_x_l643_64344

theorem find_x (x : ℤ) (h : 7 * x - 18 = 66) : x = 12 :=
  sorry

end NUMINAMATH_GPT_find_x_l643_64344


namespace NUMINAMATH_GPT_solveSystem1_solveFractionalEq_l643_64343

-- Definition: system of linear equations
def system1 (x y : ℝ) : Prop :=
  x + 2 * y = 3 ∧ x - 4 * y = 9

-- Theorem: solution to the system of equations
theorem solveSystem1 : ∃ x y : ℝ, system1 x y ∧ x = 5 ∧ y = -1 :=
by
  sorry
  
-- Definition: fractional equation
def fractionalEq (x : ℝ) : Prop :=
  (x + 2) / (x^2 - 2 * x + 1) + 3 / (x - 1) = 0

-- Theorem: solution to the fractional equation
theorem solveFractionalEq : ∃ x : ℝ, fractionalEq x ∧ x = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_solveSystem1_solveFractionalEq_l643_64343


namespace NUMINAMATH_GPT_factory_output_l643_64390

variable (a : ℝ)
variable (n : ℕ)
variable (r : ℝ)

-- Initial condition: the output value increases by 10% each year for 5 years
def annual_growth (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r^n

-- Theorem statement
theorem factory_output (a : ℝ) : annual_growth a 1.1 5 = 1.1^5 * a :=
by
  sorry

end NUMINAMATH_GPT_factory_output_l643_64390


namespace NUMINAMATH_GPT_contest_sum_l643_64314

theorem contest_sum 
(A B C D E : ℕ) 
(h_sum : A + B + C + D + E = 35)
(h_right_E : B + C + D + E = 13)
(h_right_D : C + D + E = 31)
(h_right_A : B + C + D + E = 21)
(h_right_C : C + D + E = 7)
: D + B = 11 :=
sorry

end NUMINAMATH_GPT_contest_sum_l643_64314


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l643_64313

theorem isosceles_triangle_perimeter (a b c : ℕ) (h1 : a = 6) (h2 : b = 8) (h3 : a = c ∨ b = c) :
  a + b + c = 22 :=
by
  -- This part of the proof is simplified using the conditions
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l643_64313


namespace NUMINAMATH_GPT_system_of_equations_solution_l643_64340

theorem system_of_equations_solution :
  ∃ x y : ℝ, (2 * x + y = 6) ∧ (x - y = 3) ∧ (x = 3) ∧ (y = 0) :=
by
  sorry

end NUMINAMATH_GPT_system_of_equations_solution_l643_64340


namespace NUMINAMATH_GPT_find_a_l643_64346

noncomputable def log_a (a: ℝ) (x: ℝ) : ℝ := Real.log x / Real.log a

theorem find_a (a : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : log_a a 2 - log_a a 4 = 2) :
  a = Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_GPT_find_a_l643_64346


namespace NUMINAMATH_GPT_transylvanian_convinces_l643_64366

theorem transylvanian_convinces (s : Prop) (t : Prop) (h : s ↔ (¬t ∧ ¬s)) : t :=
by
  -- Leverage the existing equivalence to prove the desired result
  sorry

end NUMINAMATH_GPT_transylvanian_convinces_l643_64366


namespace NUMINAMATH_GPT_intersect_single_point_l643_64315

theorem intersect_single_point (m : ℝ) : 
  (∃ x : ℝ, (m - 3) * x^2 - 4 * x + 2 = 0) ∧ ∀ x₁ x₂ : ℝ, 
  (m - 3) * x₁^2 - 4 * x₁ + 2 = 0 → (m - 3) * x₂^2 - 4 * x₂ + 2 = 0 → x₁ = x₂ ↔ m = 3 ∨ m = 5 := 
sorry

end NUMINAMATH_GPT_intersect_single_point_l643_64315


namespace NUMINAMATH_GPT_excess_calories_l643_64322

-- Conditions
def calories_from_cheezits (bags: ℕ) (ounces_per_bag: ℕ) (calories_per_ounce: ℕ) : ℕ :=
  bags * ounces_per_bag * calories_per_ounce

def calories_from_chocolate_bars (bars: ℕ) (calories_per_bar: ℕ) : ℕ :=
  bars * calories_per_bar

def calories_from_popcorn (calories: ℕ) : ℕ :=
  calories

def calories_burned_running (minutes: ℕ) (calories_per_minute: ℕ) : ℕ :=
  minutes * calories_per_minute

def calories_burned_swimming (minutes: ℕ) (calories_per_minute: ℕ) : ℕ :=
  minutes * calories_per_minute

def calories_burned_cycling (minutes: ℕ) (calories_per_minute: ℕ) : ℕ :=
  minutes * calories_per_minute

-- Hypothesis
def total_calories_consumed : ℕ :=
  calories_from_cheezits 3 2 150 + calories_from_chocolate_bars 2 250 + calories_from_popcorn 500

def total_calories_burned : ℕ :=
  calories_burned_running 40 12 + calories_burned_swimming 30 15 + calories_burned_cycling 20 10

-- Theorem
theorem excess_calories : total_calories_consumed - total_calories_burned = 770 := by
  sorry

end NUMINAMATH_GPT_excess_calories_l643_64322


namespace NUMINAMATH_GPT_intersection_always_exists_minimum_chord_length_and_equation_l643_64333

open Real

noncomputable def circle_eq (x y : ℝ) : Prop :=
  4 * x^2 + 4 * y^2 - 4 * x - 8 * y - 11 = 0

noncomputable def line_eq (m x y : ℝ) : Prop :=
  (m - 1) * x + m * y = m + 1

theorem intersection_always_exists :
  ∀ (m : ℝ), ∃ (x y : ℝ), circle_eq x y ∧ line_eq m x y :=
by
  sorry

theorem minimum_chord_length_and_equation :
  ∃ (k : ℝ) (x y : ℝ), k = sqrt 3 ∧ (3 * x - 2 * y + 7 = 0) ∧
    ∀ m, ∃ (xp yp : ℝ), line_eq m xp yp ∧ ∃ (l1 l2 : ℝ), line_eq m l1 l2 ∧ 
    (circle_eq xp yp ∧ circle_eq l1 l2)  :=
by
  sorry

end NUMINAMATH_GPT_intersection_always_exists_minimum_chord_length_and_equation_l643_64333


namespace NUMINAMATH_GPT_maximum_distinct_numbers_l643_64355

theorem maximum_distinct_numbers (n : ℕ) (hsum : n = 250) : 
  ∃ k ≤ 21, k = 21 :=
by
  sorry

end NUMINAMATH_GPT_maximum_distinct_numbers_l643_64355


namespace NUMINAMATH_GPT_brocard_vertex_coordinates_correct_steiner_point_coordinates_correct_l643_64363

noncomputable def brocard_vertex_trilinear_coordinates (a b c : ℝ) : ℝ × ℝ × ℝ :=
(a * b * c, c^3, b^3)

theorem brocard_vertex_coordinates_correct (a b c : ℝ) :
  brocard_vertex_trilinear_coordinates a b c = (a * b * c, c^3, b^3) :=
sorry

noncomputable def steiner_point_trilinear_coordinates (a b c : ℝ) : ℝ × ℝ × ℝ :=
(1 / (a * (b^2 - c^2)),
  1 / (b * (c^2 - a^2)),
  1 / (c * (a^2 - b^2)))

theorem steiner_point_coordinates_correct (a b c : ℝ) :
  steiner_point_trilinear_coordinates a b c = 
  (1 / (a * (b^2 - c^2)),
   1 / (b * (c^2 - a^2)),
   1 / (c * (a^2 - b^2))) :=
sorry

end NUMINAMATH_GPT_brocard_vertex_coordinates_correct_steiner_point_coordinates_correct_l643_64363


namespace NUMINAMATH_GPT_area_of_OPF_eq_sqrt_2_div_2_l643_64347

noncomputable def area_of_triangle_OPF : ℝ :=
  let O : ℝ × ℝ := (0, 0)
  let F : ℝ × ℝ := (1, 0)
  let P : ℝ × ℝ := (0.5, Real.sqrt 2) -- We assume P is (1/2, sqrt(2))
  let P1 : ℝ × ℝ := (0.5, -Real.sqrt 2) -- We also define the other point P1
  if (dist O P = dist P F) ∨ (dist O P1 = dist P1 F) then
    let base := dist O F
    let height := Real.sqrt 2
    (1 / 2) * base * height
  else
    0

theorem area_of_OPF_eq_sqrt_2_div_2 : 
  let O : ℝ × ℝ := (0, 0)
  let F : ℝ × ℝ := (1, 0)
  let P : ℝ × ℝ := (0.5, Real.sqrt 2) -- We assume P is (1/2, sqrt(2))
  let P1 : ℝ × ℝ := (0.5, -Real.sqrt 2) -- We also define the other point P1
  (dist O P = dist P F) ∨ (dist O P1 = dist P1 F) →
  let base := dist O F
  let height := Real.sqrt 2
  area_of_triangle_OPF = Real.sqrt 2 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_area_of_OPF_eq_sqrt_2_div_2_l643_64347


namespace NUMINAMATH_GPT_number_of_ways_to_assign_shifts_l643_64325

def workers : List String := ["A", "B", "C"]

theorem number_of_ways_to_assign_shifts :
  let shifts := ["day", "night"]
  (workers.length * (workers.length - 1)) = 6 := by
  sorry

end NUMINAMATH_GPT_number_of_ways_to_assign_shifts_l643_64325


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l643_64339

open Real

theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (0 ≤ a ∧ a ≤ 4) → (a^2 - 4 * a < 0) := 
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l643_64339


namespace NUMINAMATH_GPT_smallest_possible_odd_b_l643_64341

theorem smallest_possible_odd_b 
    (a b : ℕ) 
    (h1 : a + b = 90) 
    (h2 : Nat.Prime a) 
    (h3 : ∃ p : ℕ, Nat.Prime p ∧ p ∣ b) 
    (h4 : a > b) 
    (h5 : b % 2 = 1) 
    : b = 85 := 
sorry

end NUMINAMATH_GPT_smallest_possible_odd_b_l643_64341


namespace NUMINAMATH_GPT_average_marks_combined_l643_64384

theorem average_marks_combined (P C M B E : ℕ) (h : P + C + M + B + E = P + 280) : 
  (C + M + B + E) / 4 = 70 :=
by 
  sorry

end NUMINAMATH_GPT_average_marks_combined_l643_64384


namespace NUMINAMATH_GPT_sum_of_solutions_l643_64316

theorem sum_of_solutions (x y : ℝ) (h₁ : y = 8) (h₂ : x^2 + y^2 = 144) : 
  ∃ x1 x2 : ℝ, (x1 = 4 * Real.sqrt 5 ∧ x2 = -4 * Real.sqrt 5) ∧ (x1 + x2 = 0) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_solutions_l643_64316


namespace NUMINAMATH_GPT_flowers_left_l643_64394

theorem flowers_left (flowers_picked_A : Nat) (flowers_picked_M : Nat) (flowers_given : Nat)
  (h_a : flowers_picked_A = 16)
  (h_m : flowers_picked_M = 16)
  (h_g : flowers_given = 18) :
  flowers_picked_A + flowers_picked_M - flowers_given = 14 :=
by
  sorry

end NUMINAMATH_GPT_flowers_left_l643_64394


namespace NUMINAMATH_GPT_part_a_part_b_l643_64362

-- Part (a)
theorem part_a (n : ℕ) (h : n > 0) :
  (2 * n ∣ n * (n + 1) / 2) ↔ ∃ k : ℕ, n = 4 * k - 1 :=
by sorry

-- Part (b)
theorem part_b (n : ℕ) (h : n > 0) :
  (2 * n + 1 ∣ n * (n + 1) / 2) ↔ (2 * n + 1 ≡ 1 [MOD 4]) ∨ (2 * n + 1 ≡ 3 [MOD 4]) :=
by sorry

end NUMINAMATH_GPT_part_a_part_b_l643_64362


namespace NUMINAMATH_GPT_lines_proportional_l643_64337

variables {x y : ℝ} {p q : ℝ}

theorem lines_proportional (h1 : p * x + 2 * y = 7) (h2 : 3 * x + q * y = 5) :
  p = 21 / 5 := 
sorry

end NUMINAMATH_GPT_lines_proportional_l643_64337


namespace NUMINAMATH_GPT_min_students_participating_l643_64348

def ratio_9th_to_10th (n9 n10 : ℕ) : Prop := n9 * 4 = n10 * 3
def ratio_10th_to_11th (n10 n11 : ℕ) : Prop := n10 * 6 = n11 * 5

theorem min_students_participating (n9 n10 n11 : ℕ) 
    (h1 : ratio_9th_to_10th n9 n10) 
    (h2 : ratio_10th_to_11th n10 n11) : 
    n9 + n10 + n11 = 59 :=
sorry

end NUMINAMATH_GPT_min_students_participating_l643_64348


namespace NUMINAMATH_GPT_number_of_8th_graders_l643_64372

variable (x y : ℕ)
variable (y_valid : 0 ≤ y)

theorem number_of_8th_graders (h : x * (x + 3 - 2 * y) = 14) :
  x = 7 :=
by 
  sorry

end NUMINAMATH_GPT_number_of_8th_graders_l643_64372


namespace NUMINAMATH_GPT_fraction_to_decimal_l643_64320

theorem fraction_to_decimal : (7 : ℚ) / 12 = 0.5833 :=
by
  sorry

end NUMINAMATH_GPT_fraction_to_decimal_l643_64320


namespace NUMINAMATH_GPT_bicycle_wheels_l643_64389

theorem bicycle_wheels :
  ∃ b : ℕ, 
  (∃ (num_bicycles : ℕ) (num_tricycles : ℕ) (wheels_per_tricycle : ℕ) (total_wheels : ℕ),
    num_bicycles = 16 ∧ 
    num_tricycles = 7 ∧ 
    wheels_per_tricycle = 3 ∧ 
    total_wheels = 53 ∧ 
    16 * b + num_tricycles * wheels_per_tricycle = total_wheels) ∧ 
  b = 2 :=
by
  sorry

end NUMINAMATH_GPT_bicycle_wheels_l643_64389


namespace NUMINAMATH_GPT_distance_between_vertices_of_hyperbola_l643_64310

def hyperbola_equation (x y : ℝ) : Prop :=
  ∃ (c₁ c₂ : ℝ), c₁ = 4 ∧ c₂ = -4 ∧
    (c₁ * x^2 + 24 * x + c₂ * y^2 + 8 * y + 44 = 0)

theorem distance_between_vertices_of_hyperbola :
  (∀ x y : ℝ, hyperbola_equation x y) → (2 : ℝ) = 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_distance_between_vertices_of_hyperbola_l643_64310


namespace NUMINAMATH_GPT_alpha_div_3_range_l643_64307

theorem alpha_div_3_range (α : ℝ) (k : ℤ) 
  (h1 : Real.sin α > 0) 
  (h2 : Real.cos α < 0) 
  (h3 : Real.sin (α / 3) > Real.cos (α / 3)) :
  ∃ k : ℤ, (2 * k * Real.pi + Real.pi / 4 < α / 3 ∧ α / 3 < 2 * k * Real.pi + Real.pi / 3) ∨ 
            (2 * k * Real.pi + 5 * Real.pi / 6 < α / 3 ∧ α / 3 < 2 * k * Real.pi + Real.pi) :=
sorry

end NUMINAMATH_GPT_alpha_div_3_range_l643_64307


namespace NUMINAMATH_GPT_solve_inequality_l643_64357

theorem solve_inequality (x : ℝ) : 
  (-1 < (x^2 - 16 * x + 15) / (x^2 - 4 * x + 5) ∧ 
  (x^2 - 16 * x + 15) / (x^2 - 4 * x + 5) < 1) ↔ (1 < x) := 
by 
  sorry

end NUMINAMATH_GPT_solve_inequality_l643_64357


namespace NUMINAMATH_GPT_binary_to_decimal_eq_l643_64309

theorem binary_to_decimal_eq :
  (1 * 2^7 + 1 * 2^6 + 0 * 2^5 + 0 * 2^4 + 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0) = 205 :=
by
  sorry

end NUMINAMATH_GPT_binary_to_decimal_eq_l643_64309


namespace NUMINAMATH_GPT_even_and_multiple_of_3_l643_64397

theorem even_and_multiple_of_3 (a b : ℤ) (h1 : ∃ k : ℤ, a = 2 * k) (h2 : ∃ n : ℤ, b = 6 * n) :
  (∃ m : ℤ, a + b = 2 * m) ∧ (∃ p : ℤ, a + b = 3 * p) :=
by
  sorry

end NUMINAMATH_GPT_even_and_multiple_of_3_l643_64397


namespace NUMINAMATH_GPT_regular_pentagons_similar_l643_64354

-- Define a regular pentagon
structure RegularPentagon :=
  (side_length : ℝ)
  (internal_angle : ℝ)
  (angle_eq : internal_angle = 108)
  (side_positive : side_length > 0)

-- The theorem stating that two regular pentagons are always similar
theorem regular_pentagons_similar (P Q : RegularPentagon) : 
  ∀ P Q : RegularPentagon, P.internal_angle = Q.internal_angle ∧ P.side_length * Q.side_length ≠ 0 := 
sorry

end NUMINAMATH_GPT_regular_pentagons_similar_l643_64354


namespace NUMINAMATH_GPT_Olivia_house_height_l643_64356

variable (h : ℕ)
variable (flagpole_height : ℕ := 35)
variable (flagpole_shadow : ℕ := 30)
variable (house_shadow : ℕ := 70)
variable (bush_height : ℕ := 14)
variable (bush_shadow : ℕ := 12)

theorem Olivia_house_height :
  (house_shadow / flagpole_shadow) * flagpole_height = 81 ∧
  (house_shadow / bush_shadow) * bush_height = 81 :=
by
  sorry

end NUMINAMATH_GPT_Olivia_house_height_l643_64356


namespace NUMINAMATH_GPT_distance_traveled_l643_64353

theorem distance_traveled (speed1 speed2 hours1 hours2 : ℝ)
  (h1 : speed1 = 45) (h2 : hours1 = 2) (h3 : speed2 = 50) (h4 : hours2 = 3) :
  speed1 * hours1 + speed2 * hours2 = 240 := by
  sorry

end NUMINAMATH_GPT_distance_traveled_l643_64353


namespace NUMINAMATH_GPT_number_of_slices_per_pizza_l643_64342

-- Given conditions as definitions in Lean 4
def total_pizzas := 2
def total_slices_per_pizza (S : ℕ) : ℕ := total_pizzas * S
def james_portion : ℚ := 2 / 3
def james_ate_slices (S : ℕ) : ℚ := james_portion * (total_slices_per_pizza S)
def james_ate_exactly := 8

-- The main theorem to prove
theorem number_of_slices_per_pizza (S : ℕ) (h : james_ate_slices S = james_ate_exactly) : S = 6 :=
sorry

end NUMINAMATH_GPT_number_of_slices_per_pizza_l643_64342


namespace NUMINAMATH_GPT_sum_of_undefined_domain_values_l643_64380

theorem sum_of_undefined_domain_values :
  ∀ (x : ℝ), (x = 0 ∨ (1 + 1/x) = 0 ∨ (1 + 1/(1 + 1/x)) = 0 ∨ (1 + 1/(1 + 1/(1 + 1/x))) = 0) →
  x = 0 ∧ x = -1 ∧ x = -1/2 ∧ x = -1/3 →
  (0 + (-1) + (-1/2) + (-1/3) = -11/6) := sorry

end NUMINAMATH_GPT_sum_of_undefined_domain_values_l643_64380


namespace NUMINAMATH_GPT_estimate_white_balls_l643_64324

-- Statements for conditions
variables (black_balls white_balls : ℕ)
variables (draws : ℕ := 40)
variables (black_draws : ℕ := 10)

-- Define total white draws
def white_draws := draws - black_draws

-- Ratio of black to white draws
def draw_ratio := black_draws / white_draws

-- Given condition on known draws
def black_ball_count := 4
def known_draw_ratio := 1 / 3

-- Lean 4 statement to prove the number of white balls
theorem estimate_white_balls (h : black_ball_count / white_balls = known_draw_ratio) : white_balls = 12 :=
sorry -- Proof omitted

end NUMINAMATH_GPT_estimate_white_balls_l643_64324


namespace NUMINAMATH_GPT_shuttle_speed_conversion_l643_64370

-- Define the speed of the space shuttle in kilometers per second
def shuttle_speed_km_per_sec : ℕ := 6

-- Define the number of seconds in an hour
def seconds_per_hour : ℕ := 3600

-- Define the expected speed in kilometers per hour
def expected_speed_km_per_hour : ℕ := 21600

-- Prove that the speed converted to kilometers per hour is equal to the expected speed
theorem shuttle_speed_conversion : shuttle_speed_km_per_sec * seconds_per_hour = expected_speed_km_per_hour :=
by
    sorry

end NUMINAMATH_GPT_shuttle_speed_conversion_l643_64370


namespace NUMINAMATH_GPT_complete_set_contains_all_rationals_l643_64303

theorem complete_set_contains_all_rationals (T : Set ℚ) (hT : ∀ (p q : ℚ), p / q ∈ T → p / (p + q) ∈ T ∧ q / (p + q) ∈ T) (r : ℚ) : 
  (r = 1 ∨ r = 1 / 2) → (∀ x : ℚ, 0 < x ∧ x < 1 → x ∈ T) :=
by
  sorry

end NUMINAMATH_GPT_complete_set_contains_all_rationals_l643_64303


namespace NUMINAMATH_GPT_eliminate_all_evil_with_at_most_one_good_l643_64369

-- Defining the problem setting
structure Wizard :=
  (is_good : Bool)

-- The main theorem
theorem eliminate_all_evil_with_at_most_one_good (wizards : List Wizard) (h_wizard_count : wizards.length = 2015) :
  ∃ (banish_sequence : List Wizard), 
    (∀ w ∈ banish_sequence, w.is_good = false) ∨ (∃ (g : Wizard), g.is_good = true ∧ g ∉ banish_sequence) :=
sorry

end NUMINAMATH_GPT_eliminate_all_evil_with_at_most_one_good_l643_64369


namespace NUMINAMATH_GPT_GCF_75_135_l643_64352

theorem GCF_75_135 : Nat.gcd 75 135 = 15 :=
by
sorry

end NUMINAMATH_GPT_GCF_75_135_l643_64352


namespace NUMINAMATH_GPT_trivia_competition_points_l643_64360

theorem trivia_competition_points 
  (total_members : ℕ := 120) 
  (absent_members : ℕ := 37) 
  (points_per_member : ℕ := 24) : 
  (total_members - absent_members) * points_per_member = 1992 := 
by
  sorry

end NUMINAMATH_GPT_trivia_competition_points_l643_64360


namespace NUMINAMATH_GPT_anthony_total_pencils_l643_64305

theorem anthony_total_pencils :
  let original_pencils := 9
  let given_pencils := 56
  original_pencils + given_pencils = 65 := by
  sorry

end NUMINAMATH_GPT_anthony_total_pencils_l643_64305


namespace NUMINAMATH_GPT_Ann_age_is_46_l643_64387

theorem Ann_age_is_46
  (a b : ℕ) 
  (h1 : a + b = 72)
  (h2 : b = (a / 3) + 2 * (a - b)) : a = 46 :=
by
  sorry

end NUMINAMATH_GPT_Ann_age_is_46_l643_64387


namespace NUMINAMATH_GPT_value_of_y_at_x8_l643_64388

theorem value_of_y_at_x8 (k : ℝ) (f : ℝ → ℝ) (h1 : ∀ x : ℝ, f x = k * x^(1 / 3)) (h2 : f 64 = 4) : f 8 = 2 :=
sorry

end NUMINAMATH_GPT_value_of_y_at_x8_l643_64388


namespace NUMINAMATH_GPT_number_of_balls_sold_l643_64317

-- Definitions from conditions
def selling_price : ℕ := 720
def cost_price_per_ball : ℕ := 120
def loss : ℕ := 5 * cost_price_per_ball

-- Mathematically equivalent proof statement
theorem number_of_balls_sold (n : ℕ) (h : n * cost_price_per_ball - selling_price = loss) : n = 11 :=
  sorry

end NUMINAMATH_GPT_number_of_balls_sold_l643_64317


namespace NUMINAMATH_GPT_range_a_mul_b_sub_three_half_l643_64377

theorem range_a_mul_b_sub_three_half (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : 1 < b) (h4 : b = (1 + Real.sqrt 5) / 2 * a) :
  (∃ l u : ℝ, ∀ f, l ≤ f ∧ f < u ↔ f = a * (b - 3 / 2)) :=
sorry

end NUMINAMATH_GPT_range_a_mul_b_sub_three_half_l643_64377


namespace NUMINAMATH_GPT_Hiram_age_l643_64371

theorem Hiram_age (H A : ℕ) (h₁ : H + 12 = 2 * A - 4) (h₂ : A = 28) : H = 40 :=
by
  sorry

end NUMINAMATH_GPT_Hiram_age_l643_64371


namespace NUMINAMATH_GPT_unique_k_satisfying_eq_l643_64398

theorem unique_k_satisfying_eq (k : ℤ) :
  (∀ a b c : ℝ, (a + b + c) * (a * b + b * c + c * a) + k * a * b * c = (a + b) * (b + c) * (c + a)) ↔ k = -1 :=
sorry

end NUMINAMATH_GPT_unique_k_satisfying_eq_l643_64398


namespace NUMINAMATH_GPT_reciprocal_neg_2023_l643_64381

theorem reciprocal_neg_2023 : 1 / (-2023) = -1 / 2023 :=
by 
  sorry

end NUMINAMATH_GPT_reciprocal_neg_2023_l643_64381


namespace NUMINAMATH_GPT_Fran_speed_l643_64308

-- Definitions needed for statements
def Joann_speed : ℝ := 15
def Joann_time : ℝ := 4
def Fran_time : ℝ := 5

-- Formalize the problem in Lean
theorem Fran_speed (Joann_distance : ℝ) (Fran_speed : ℝ) : 
  Joann_distance = Joann_speed * Joann_time →
  Fran_speed * Fran_time = Joann_distance →
  Fran_speed = 12 :=
by
  -- assume the conditions about distances
  intros h1 h2
  -- prove the goal
  sorry

end NUMINAMATH_GPT_Fran_speed_l643_64308


namespace NUMINAMATH_GPT_sum_gcd_lcm_of_4_and_10_l643_64396

theorem sum_gcd_lcm_of_4_and_10 : Nat.gcd 4 10 + Nat.lcm 4 10 = 22 :=
by
  sorry

end NUMINAMATH_GPT_sum_gcd_lcm_of_4_and_10_l643_64396


namespace NUMINAMATH_GPT_certain_number_value_l643_64312

theorem certain_number_value
  (x : ℝ)
  (y : ℝ)
  (h1 : (28 + x + 42 + 78 + y) / 5 = 62)
  (h2 : (48 + 62 + 98 + 124 + x) / 5 = 78) : 
  y = 104 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_certain_number_value_l643_64312


namespace NUMINAMATH_GPT_mod_equiv_22_l643_64334

theorem mod_equiv_22 : ∃ m : ℕ, (198 * 864) % 50 = m ∧ 0 ≤ m ∧ m < 50 ∧ m = 22 := by
  sorry

end NUMINAMATH_GPT_mod_equiv_22_l643_64334


namespace NUMINAMATH_GPT_janet_miles_per_day_l643_64306

def total_miles : ℕ := 72
def days : ℕ := 9
def miles_per_day : ℕ := 8

theorem janet_miles_per_day : total_miles / days = miles_per_day :=
by {
  sorry
}

end NUMINAMATH_GPT_janet_miles_per_day_l643_64306


namespace NUMINAMATH_GPT_cube_of_composite_as_diff_of_squares_l643_64351

theorem cube_of_composite_as_diff_of_squares (n : ℕ) (h : ∃ a b, a > 1 ∧ b > 1 ∧ n = a * b) :
  ∃ (A₁ B₁ A₂ B₂ A₃ B₃ : ℕ), 
    n^3 = A₁^2 - B₁^2 ∧ 
    n^3 = A₂^2 - B₂^2 ∧ 
    n^3 = A₃^2 - B₃^2 ∧ 
    (A₁, B₁) ≠ (A₂, B₂) ∧ 
    (A₁, B₁) ≠ (A₃, B₃) ∧ 
    (A₂, B₂) ≠ (A₃, B₃) := sorry

end NUMINAMATH_GPT_cube_of_composite_as_diff_of_squares_l643_64351


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l643_64335

-- Define the function f
noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := (b - 2^x) / (2^(x + 1) + a)

-- Problem 1
theorem problem1 (h_odd : ∀ x, f x a b = -f (-x) a b) : a = 2 ∧ b = 1 :=
sorry

-- Problem 2
theorem problem2 : (∀ x, f x 2 1 = -f (-x) 2 1) → ∀ x y, x < y → f x 2 1 > f y 2 1 :=
sorry

-- Problem 3
theorem problem3 (h_pos : ∀ x ≥ 1, f (k * 3^x) 2 1 + f (3^x - 9^x + 2) 2 1 > 0) : k < 4 / 3 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l643_64335


namespace NUMINAMATH_GPT_sum_of_cubes_l643_64378

theorem sum_of_cubes
  (x y : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y)
  (h1 : (x + y)^2 = 2500) 
  (h2 : x * y = 500) :
  x^3 + y^3 = 50000 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_cubes_l643_64378


namespace NUMINAMATH_GPT_remainder_division_l643_64374

theorem remainder_division (n r : ℕ) (k : ℤ) (h1 : n % 25 = r) (h2 : (n + 15) % 5 = r) (h3 : 0 ≤ r ∧ r < 25) : r = 5 :=
sorry

end NUMINAMATH_GPT_remainder_division_l643_64374


namespace NUMINAMATH_GPT_marcy_minimum_avg_score_l643_64358

variables (s1 s2 s3 : ℝ)
variable (qualified_avg : ℝ := 90)
variable (required_total : ℝ := 5 * qualified_avg)
variable (first_three_total : ℝ := s1 + s2 + s3)
variable (needed_points : ℝ := required_total - first_three_total)
variable (required_avg : ℝ := needed_points / 2)

/-- The admission criteria for a mathematics contest require a contestant to 
    achieve an average score of at least 90% over five rounds to qualify for the final round.
    Marcy scores 87%, 92%, and 85% in the first three rounds. 
    Prove that Marcy must average at least 93% in the next two rounds to qualify for the final. --/
theorem marcy_minimum_avg_score 
    (h1 : s1 = 87) (h2 : s2 = 92) (h3 : s3 = 85)
    : required_avg ≥ 93 :=
sorry

end NUMINAMATH_GPT_marcy_minimum_avg_score_l643_64358


namespace NUMINAMATH_GPT_Tamika_hours_l643_64349

variable (h : ℕ)

theorem Tamika_hours :
  (45 * h = 55 * 5 + 85) → h = 8 :=
by 
  sorry

end NUMINAMATH_GPT_Tamika_hours_l643_64349


namespace NUMINAMATH_GPT_weight_of_milk_l643_64386

def max_bag_capacity : ℕ := 20
def green_beans : ℕ := 4
def carrots : ℕ := 2 * green_beans
def fit_more : ℕ := 2
def current_weight : ℕ := max_bag_capacity - fit_more
def total_weight_of_green_beans_and_carrots : ℕ := green_beans + carrots

theorem weight_of_milk : (current_weight - total_weight_of_green_beans_and_carrots) = 6 := by
  -- Proof to be written here
  sorry

end NUMINAMATH_GPT_weight_of_milk_l643_64386


namespace NUMINAMATH_GPT_systematic_sampling_method_l643_64336

def num_rows : ℕ := 50
def num_seats_per_row : ℕ := 30

def is_systematic_sampling (select_interval : ℕ) : Prop :=
  ∀ n, select_interval = n * num_seats_per_row + 8

theorem systematic_sampling_method :
  is_systematic_sampling 30 :=
by
  sorry

end NUMINAMATH_GPT_systematic_sampling_method_l643_64336
