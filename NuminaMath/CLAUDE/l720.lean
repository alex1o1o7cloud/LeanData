import Mathlib

namespace inequality_proof_l720_72050

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : c = d) : a + c > b + d := by
  sorry

end inequality_proof_l720_72050


namespace same_even_number_probability_l720_72072

-- Define a standard die
def standardDie : ℕ := 6

-- Define the number of even faces on a standard die
def evenFaces : ℕ := 3

-- Define the number of dice rolled
def numDice : ℕ := 4

-- Theorem statement
theorem same_even_number_probability :
  let p : ℚ := (evenFaces / standardDie) * (1 / standardDie)^(numDice - 1)
  p = 1 / 432 := by
  sorry


end same_even_number_probability_l720_72072


namespace geometric_sequence_ratio_l720_72009

theorem geometric_sequence_ratio (a₁ : ℝ) (q : ℝ) :
  a₁ ≠ 0 →
  q ≠ 1 →
  let S₂ := a₁ * (1 - q^2) / (1 - q)
  let S₃ := a₁ * (1 - q^3) / (1 - q)
  S₃ + 3 * S₂ = 0 →
  q = -2 := by
sorry

end geometric_sequence_ratio_l720_72009


namespace intersection_A_complement_B_l720_72039

-- Define the set A
def A : Set ℝ := {a | ∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0}

-- Define the set B
def B : Set ℝ := {x | ∀ a ∈ Set.Icc (-2 : ℝ) 2, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0}

-- State the theorem
theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = {-1} := by
  sorry

end intersection_A_complement_B_l720_72039


namespace f_monotonicity_and_extrema_f_extrema_sum_max_l720_72015

noncomputable section

def f (a x : ℝ) := x^2 / 2 - 4*a*x + a * Real.log x + 3*a^2 + 2*a

def f_deriv (a x : ℝ) := x - 4*a + a/x

theorem f_monotonicity_and_extrema (a : ℝ) (ha : a > 0) :
  (∀ x > 0, f_deriv a x ≥ 0) ∨
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f_deriv a x₁ = 0 ∧ f_deriv a x₂ = 0) :=
sorry

theorem f_extrema_sum_max (a : ℝ) (ha : a > 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f_deriv a x₁ = 0 ∧ f_deriv a x₂ = 0 →
  ∀ y₁ y₂ : ℝ, y₁ ≠ y₂ → f_deriv a y₁ = 0 → f_deriv a y₂ = 0 →
  f a x₁ + f a x₂ ≥ f a y₁ + f a y₂ ∧
  f a x₁ + f a x₂ ≤ 1 :=
sorry

end

end f_monotonicity_and_extrema_f_extrema_sum_max_l720_72015


namespace betty_orange_boxes_l720_72056

/-- The minimum number of boxes needed to store oranges given specific conditions -/
def min_boxes (total_oranges : ℕ) (first_box : ℕ) (second_box : ℕ) (max_per_box : ℕ) : ℕ :=
  2 + (total_oranges - first_box - second_box + max_per_box - 1) / max_per_box

/-- Proof that Betty needs 5 boxes to store her oranges -/
theorem betty_orange_boxes : 
  min_boxes 120 30 25 30 = 5 :=
by sorry

end betty_orange_boxes_l720_72056


namespace quadratic_inequality_condition_l720_72051

theorem quadratic_inequality_condition (m : ℝ) :
  (∀ x : ℝ, m * x^2 + x + m > 0) → m > (1/4 : ℝ) ∧
  ¬(m > (1/4 : ℝ) → ∀ x : ℝ, m * x^2 + x + m > 0) :=
by sorry

end quadratic_inequality_condition_l720_72051


namespace sum_of_coefficients_l720_72019

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (1 + x)^4 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  a₁ + a₃ = 8 := by
sorry

end sum_of_coefficients_l720_72019


namespace equation_solution_l720_72020

theorem equation_solution :
  ∀ x : ℝ, 3 * (x - 3) = (x - 3)^2 ↔ x = 3 ∨ x = 6 := by
  sorry

end equation_solution_l720_72020


namespace number_sum_proof_l720_72010

theorem number_sum_proof (x : ℤ) : x + 14 = 68 → x + (x + 41) = 149 := by
  sorry

end number_sum_proof_l720_72010


namespace prob_at_least_one_red_l720_72007

/-- Probability of drawing at least one red ball in three independent draws with replacement -/
theorem prob_at_least_one_red : 
  let total_balls : ℕ := 2
  let red_balls : ℕ := 1
  let num_draws : ℕ := 3
  let prob_blue : ℚ := 1 / 2
  let prob_all_blue : ℚ := prob_blue ^ num_draws
  prob_all_blue = 1 / 8 ∧ (1 : ℚ) - prob_all_blue = 7 / 8 := by sorry

end prob_at_least_one_red_l720_72007


namespace serena_age_proof_l720_72000

/-- Serena's current age -/
def serena_age : ℕ := 9

/-- Serena's mother's current age -/
def mother_age : ℕ := 39

/-- Years into the future when the age comparison is made -/
def years_later : ℕ := 6

theorem serena_age_proof :
  serena_age = 9 ∧
  mother_age = 39 ∧
  mother_age + years_later = 3 * (serena_age + years_later) :=
by sorry

end serena_age_proof_l720_72000


namespace cube_sum_magnitude_l720_72060

theorem cube_sum_magnitude (z₁ z₂ : ℂ) 
  (h1 : Complex.abs (z₁ + z₂) = 20)
  (h2 : Complex.abs (z₁^2 + z₂^2) = 16) :
  Complex.abs (z₁^3 + z₂^3) = 3520 := by
  sorry

end cube_sum_magnitude_l720_72060


namespace imaginary_part_of_complex_fraction_l720_72092

theorem imaginary_part_of_complex_fraction : Complex.im ((1 - Complex.I) / (1 + Complex.I) + 1) = -1 := by
  sorry

end imaginary_part_of_complex_fraction_l720_72092


namespace angle_equality_l720_72068

theorem angle_equality (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.cos α + Real.cos β - Real.cos (α + β) = 3/2) :
  α = π/3 ∧ β = π/3 := by
sorry

end angle_equality_l720_72068


namespace equation_solution_l720_72038

theorem equation_solution (x : Real) :
  (|Real.cos x| + Real.cos (3 * x)) / (Real.sin x * Real.cos (2 * x)) = -2 * Real.sqrt 3 ↔
  (∃ k : ℤ, x = 2 * π / 3 + 2 * k * π ∨ x = 7 * π / 6 + 2 * k * π ∨ x = -π / 6 + 2 * k * π) :=
by sorry

end equation_solution_l720_72038


namespace product_increase_l720_72071

theorem product_increase (A B : ℝ) (h : A * B = 1.6) : (5 * A) * (5 * B) = 40 := by
  sorry

end product_increase_l720_72071


namespace tshirt_cost_l720_72017

theorem tshirt_cost (initial_amount : ℕ) (sweater_cost : ℕ) (shoes_cost : ℕ) (remaining_amount : ℕ) :
  initial_amount = 91 →
  sweater_cost = 24 →
  shoes_cost = 11 →
  remaining_amount = 50 →
  initial_amount - remaining_amount - sweater_cost - shoes_cost = 6 := by
  sorry

end tshirt_cost_l720_72017


namespace alarm_system_probability_l720_72055

theorem alarm_system_probability (p : ℝ) (h1 : p = 0.4) :
  let prob_at_least_one_alerts := 1 - (1 - p) * (1 - p)
  prob_at_least_one_alerts = 0.64 := by
sorry

end alarm_system_probability_l720_72055


namespace special_tetrahedron_equal_angle_l720_72047

/-- A tetrahedron with specific dihedral angle properties -/
structure SpecialTetrahedron where
  /-- The tetrahedron has three dihedral angles of 90° that do not belong to the same vertex -/
  three_right_angles : ℕ
  /-- All other dihedral angles are equal -/
  equal_other_angles : ℝ
  /-- The number of 90° angles is exactly 3 -/
  right_angle_count : three_right_angles = 3

/-- The theorem stating the value of the equal dihedral angles in the special tetrahedron -/
theorem special_tetrahedron_equal_angle (t : SpecialTetrahedron) :
  t.equal_other_angles = Real.arccos ((Real.sqrt 5 - 1) / 2) :=
sorry

end special_tetrahedron_equal_angle_l720_72047


namespace total_flowers_and_sticks_l720_72035

/-- The number of pots -/
def num_pots : ℕ := 466

/-- The number of flowers in each pot -/
def flowers_per_pot : ℕ := 53

/-- The number of sticks in each pot -/
def sticks_per_pot : ℕ := 181

/-- The total number of flowers and sticks in all pots -/
def total_items : ℕ := num_pots * flowers_per_pot + num_pots * sticks_per_pot

theorem total_flowers_and_sticks : total_items = 109044 := by
  sorry

end total_flowers_and_sticks_l720_72035


namespace negative_angle_quadrant_l720_72037

def is_in_third_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, k * 360 + 180 < α ∧ α < k * 360 + 270

def is_in_second_quadrant (α : Real) : Prop :=
  ∃ n : ℤ, n * 360 - 270 < α ∧ α < n * 360 - 180

theorem negative_angle_quadrant (α : Real) :
  is_in_third_quadrant α → is_in_second_quadrant (-α) := by
  sorry

end negative_angle_quadrant_l720_72037


namespace green_blue_difference_after_border_l720_72099

/-- Represents a hexagonal figure with tiles --/
structure HexagonalFigure where
  blue_tiles : ℕ
  green_tiles : ℕ

/-- Calculates the number of tiles needed for a single border layer of a hexagon --/
def single_border_tiles : ℕ := 6 * 3

/-- Calculates the number of tiles needed for a double border layer of a hexagon --/
def double_border_tiles : ℕ := single_border_tiles + 6 * 4

/-- Adds a double border of green tiles to a hexagonal figure --/
def add_double_border (figure : HexagonalFigure) : HexagonalFigure :=
  { blue_tiles := figure.blue_tiles,
    green_tiles := figure.green_tiles + double_border_tiles }

/-- The main theorem to prove --/
theorem green_blue_difference_after_border (initial_figure : HexagonalFigure)
    (h1 : initial_figure.blue_tiles = 20)
    (h2 : initial_figure.green_tiles = 10) :
    let new_figure := add_double_border initial_figure
    new_figure.green_tiles - new_figure.blue_tiles = 32 := by
  sorry

end green_blue_difference_after_border_l720_72099


namespace ab_value_for_given_equation_l720_72089

theorem ab_value_for_given_equation (a b : ℕ+) 
  (h : (2 * a + b) * (2 * b + a) = 4752) : 
  a * b = 520 := by
sorry

end ab_value_for_given_equation_l720_72089


namespace students_in_one_subject_is_32_l720_72090

/-- Represents the number of students in each class and their intersections -/
structure ClassEnrollment where
  calligraphy : ℕ
  art : ℕ
  instrumental : ℕ
  calligraphy_art : ℕ
  calligraphy_instrumental : ℕ
  art_instrumental : ℕ
  all_three : ℕ

/-- Calculates the number of students enrolled in only one subject -/
def studentsInOneSubject (e : ClassEnrollment) : ℕ :=
  e.calligraphy + e.art + e.instrumental - 2 * (e.calligraphy_art + e.calligraphy_instrumental + e.art_instrumental) + 3 * e.all_three

/-- The main theorem stating that given the enrollment conditions, 32 students are in only one subject -/
theorem students_in_one_subject_is_32 (e : ClassEnrollment)
  (h1 : e.calligraphy = 29)
  (h2 : e.art = 28)
  (h3 : e.instrumental = 27)
  (h4 : e.calligraphy_art = 13)
  (h5 : e.calligraphy_instrumental = 12)
  (h6 : e.art_instrumental = 11)
  (h7 : e.all_three = 5) :
  studentsInOneSubject e = 32 := by
  sorry


end students_in_one_subject_is_32_l720_72090


namespace polynomial_coefficient_sum_l720_72005

theorem polynomial_coefficient_sum : 
  ∀ A B C D : ℝ, 
  (∀ x : ℝ, (x + 3) * (4 * x^2 - 2 * x + 7) = A * x^3 + B * x^2 + C * x + D) → 
  A + B + C + D = 36 := by
sorry

end polynomial_coefficient_sum_l720_72005


namespace triangle_area_sine_relation_l720_72093

theorem triangle_area_sine_relation (a b c : ℝ) (A : ℝ) :
  (0 < a) → (0 < b) → (0 < c) →
  (0 < A) → (A < π) →
  (a^2 = b^2 + c^2 - 2*b*c*Real.cos A) →
  (a^2 - b^2 - c^2 + 2*b*c = (1/2) * b * c * Real.sin A) →
  Real.sin A = 8/17 := by
sorry

end triangle_area_sine_relation_l720_72093


namespace zero_neither_positive_nor_negative_l720_72044

theorem zero_neither_positive_nor_negative : ¬(0 > 0 ∨ 0 < 0) := by
  sorry

end zero_neither_positive_nor_negative_l720_72044


namespace quadratic_solution_l720_72063

theorem quadratic_solution (h : 81 * (4/9)^2 - 145 * (4/9) + 64 = 0) :
  81 * (-16/9)^2 - 145 * (-16/9) + 64 = 0 := by
  sorry

end quadratic_solution_l720_72063


namespace set_operations_l720_72076

def U : Set ℤ := {x | 0 < x ∧ x ≤ 10}
def A : Set ℤ := {1, 2, 4, 5, 9}
def B : Set ℤ := {4, 6, 7, 8, 10}

theorem set_operations :
  (A ∩ B = {4}) ∧
  (A ∪ B = {1, 2, 4, 5, 6, 7, 8, 9, 10}) ∧
  ((U \ (A ∪ B)) = {3}) ∧
  ((U \ A) ∩ (U \ B) = {3}) := by
  sorry

end set_operations_l720_72076


namespace plates_in_second_purchase_is_20_l720_72094

/-- The cost of one paper plate -/
def plate_cost : ℝ := sorry

/-- The cost of one paper cup -/
def cup_cost : ℝ := sorry

/-- The number of plates in the second purchase -/
def plates_in_second_purchase : ℕ := sorry

/-- The total cost of 100 plates and 200 cups is $7.50 -/
axiom first_purchase : 100 * plate_cost + 200 * cup_cost = 7.50

/-- The total cost of some plates and 40 cups is $1.50 -/
axiom second_purchase : plates_in_second_purchase * plate_cost + 40 * cup_cost = 1.50

theorem plates_in_second_purchase_is_20 : plates_in_second_purchase = 20 := by sorry

end plates_in_second_purchase_is_20_l720_72094


namespace sum_and_equality_implies_b_value_l720_72032

theorem sum_and_equality_implies_b_value
  (a b c : ℝ)
  (sum_eq : a + b + c = 117)
  (equality : a + 8 = b - 10 ∧ b - 10 = 4 * c) :
  b = 550 / 9 := by
sorry

end sum_and_equality_implies_b_value_l720_72032


namespace min_sum_positive_reals_min_sum_positive_reals_tight_l720_72091

theorem min_sum_positive_reals (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (3 * b) + b / (6 * c) + c / (9 * a) ≥ (1 / 2 : ℝ) :=
by sorry

theorem min_sum_positive_reals_tight (ε : ℝ) (hε : ε > 0) :
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧
  a / (3 * b) + b / (6 * c) + c / (9 * a) < (1 / 2 : ℝ) + ε :=
by sorry

end min_sum_positive_reals_min_sum_positive_reals_tight_l720_72091


namespace root_sum_fourth_power_l720_72004

theorem root_sum_fourth_power (a b c s : ℝ) : 
  (x^3 - 6*x^2 + 14*x - 6 = 0 → (x = a ∨ x = b ∨ x = c)) →
  s = Real.sqrt a + Real.sqrt b + Real.sqrt c →
  s^4 - 12*s^2 - 24*s = 20 := by
sorry

end root_sum_fourth_power_l720_72004


namespace chi_squared_relationship_confidence_l720_72067

-- Define the chi-squared statistic
def chi_squared : ℝ := 4.073

-- Define the critical values and their corresponding p-values
def critical_value_1 : ℝ := 3.841
def p_value_1 : ℝ := 0.05

def critical_value_2 : ℝ := 5.024
def p_value_2 : ℝ := 0.025

-- Define the confidence level we want to prove
def target_confidence : ℝ := 0.95

-- Theorem statement
theorem chi_squared_relationship_confidence :
  chi_squared > critical_value_1 ∧ chi_squared < critical_value_2 →
  ∃ (confidence : ℝ), confidence ≥ target_confidence ∧
    confidence ≤ 1 - p_value_1 ∧
    confidence > 1 - p_value_2 :=
by sorry

end chi_squared_relationship_confidence_l720_72067


namespace symbol_equation_solution_l720_72018

theorem symbol_equation_solution :
  ∀ (star square circle : ℕ),
    star + square = 24 →
    square + circle = 30 →
    circle + star = 36 →
    square = 9 ∧ circle = 21 ∧ star = 15 :=
by
  sorry

end symbol_equation_solution_l720_72018


namespace josiah_saved_24_days_l720_72059

/-- The number of days Josiah saved -/
def josiah_days : ℕ := sorry

/-- Josiah's daily savings in dollars -/
def josiah_daily_savings : ℚ := 1/4

/-- Leah's daily savings in dollars -/
def leah_daily_savings : ℚ := 1/2

/-- Number of days Leah saved -/
def leah_days : ℕ := 20

/-- Number of days Megan saved -/
def megan_days : ℕ := 12

/-- Total amount saved by all three children in dollars -/
def total_savings : ℚ := 28

theorem josiah_saved_24_days :
  josiah_days = 24 ∧
  josiah_daily_savings * josiah_days + 
  leah_daily_savings * leah_days + 
  (2 * leah_daily_savings) * megan_days = total_savings := by sorry

end josiah_saved_24_days_l720_72059


namespace prob_at_least_one_woman_pair_value_l720_72057

/-- The number of young men in the group -/
def num_men : ℕ := 6

/-- The number of young women in the group -/
def num_women : ℕ := 6

/-- The total number of people in the group -/
def total_people : ℕ := num_men + num_women

/-- The number of pairs formed -/
def num_pairs : ℕ := total_people / 2

/-- The total number of ways to pair up all people -/
def total_pairings : ℕ := (total_people.factorial) / (2^num_pairs * num_pairs.factorial)

/-- The number of ways to pair up without any woman-woman pairs -/
def pairings_without_woman_pairs : ℕ := num_women.factorial

/-- The probability of at least one woman-woman pair -/
def prob_at_least_one_woman_pair : ℚ :=
  (total_pairings - pairings_without_woman_pairs : ℚ) / total_pairings

theorem prob_at_least_one_woman_pair_value :
  prob_at_least_one_woman_pair = (10395 - 720) / 10395 := by sorry

end prob_at_least_one_woman_pair_value_l720_72057


namespace compound_molecular_weight_l720_72065

/-- The atomic weight of Barium in g/mol -/
def atomic_weight_Ba : ℝ := 137.33

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The atomic weight of Hydrogen in g/mol -/
def atomic_weight_H : ℝ := 1.01

/-- The number of Barium atoms in the compound -/
def num_Ba : ℕ := 1

/-- The number of Oxygen atoms in the compound -/
def num_O : ℕ := 2

/-- The number of Hydrogen atoms in the compound -/
def num_H : ℕ := 2

/-- The molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := 
  (num_Ba : ℝ) * atomic_weight_Ba + 
  (num_O : ℝ) * atomic_weight_O + 
  (num_H : ℝ) * atomic_weight_H

theorem compound_molecular_weight : 
  molecular_weight = 171.35 := by sorry

end compound_molecular_weight_l720_72065


namespace product_95_105_l720_72028

theorem product_95_105 : 95 * 105 = 9975 := by
  have h1 : 95 = 100 - 5 := by sorry
  have h2 : 105 = 100 + 5 := by sorry
  sorry

end product_95_105_l720_72028


namespace bug_population_zero_l720_72033

/-- Represents the bug population and predator actions in Bill's garden --/
structure GardenState where
  initial_bugs : ℕ
  spiders : ℕ
  ladybugs : ℕ
  mantises : ℕ
  spider_eat_rate : ℕ
  ladybug_eat_rate : ℕ
  mantis_eat_rate : ℕ
  first_spray_rate : ℚ
  second_spray_rate : ℚ

/-- Calculates the final bug population after all actions --/
def final_bug_population (state : GardenState) : ℕ :=
  sorry

/-- Theorem stating that the final bug population is 0 --/
theorem bug_population_zero (state : GardenState) 
  (h1 : state.initial_bugs = 400)
  (h2 : state.spiders = 12)
  (h3 : state.ladybugs = 5)
  (h4 : state.mantises = 8)
  (h5 : state.spider_eat_rate = 7)
  (h6 : state.ladybug_eat_rate = 6)
  (h7 : state.mantis_eat_rate = 4)
  (h8 : state.first_spray_rate = 4/5)
  (h9 : state.second_spray_rate = 7/10) :
  final_bug_population state = 0 :=
sorry

end bug_population_zero_l720_72033


namespace a_values_l720_72079

def A (a : ℝ) : Set ℝ := {0, 1, a^2 - 2*a}

theorem a_values (a : ℝ) (h : a ∈ A a) : a = 1 ∨ a = 3 := by
  sorry

end a_values_l720_72079


namespace gcd_problem_l720_72061

theorem gcd_problem : ∃! n : ℕ, 70 ≤ n ∧ n ≤ 90 ∧ Nat.gcd n 30 = 6 := by
  sorry

end gcd_problem_l720_72061


namespace positive_sum_greater_than_abs_diff_l720_72075

theorem positive_sum_greater_than_abs_diff (x y : ℝ) :
  x + y > |x - y| ↔ x > 0 ∧ y > 0 := by sorry

end positive_sum_greater_than_abs_diff_l720_72075


namespace decreasing_direct_proportion_negative_k_l720_72031

/-- A direct proportion function y = kx where y decreases as x increases -/
structure DecreasingDirectProportion where
  k : ℝ
  decreasing : ∀ (x₁ x₂ : ℝ), x₁ < x₂ → k * x₁ > k * x₂

/-- Theorem: If y = kx is a decreasing direct proportion function, then k < 0 -/
theorem decreasing_direct_proportion_negative_k (f : DecreasingDirectProportion) : f.k < 0 := by
  sorry

end decreasing_direct_proportion_negative_k_l720_72031


namespace candy_sampling_problem_l720_72046

theorem candy_sampling_problem (caught_percentage : ℝ) (total_sampling_percentage : ℝ) 
  (h1 : caught_percentage = 22)
  (h2 : total_sampling_percentage = 25) :
  total_sampling_percentage - caught_percentage = 3 := by
  sorry

end candy_sampling_problem_l720_72046


namespace business_investment_problem_l720_72096

/-- Represents the investment and profit share of a business partner -/
structure Partner where
  investment : ℕ
  profitShare : ℕ

/-- Proves that given the conditions of the business problem, partner a's investment is 16000 -/
theorem business_investment_problem 
  (a b c : Partner)
  (h1 : b.profitShare = 1800)
  (h2 : a.profitShare - c.profitShare = 720)
  (h3 : b.investment = 10000)
  (h4 : c.investment = 12000)
  (h5 : a.profitShare * b.investment = b.profitShare * a.investment)
  (h6 : b.profitShare * c.investment = c.profitShare * b.investment)
  (h7 : a.profitShare * c.investment = c.profitShare * a.investment) :
  a.investment = 16000 := by
  sorry


end business_investment_problem_l720_72096


namespace cubic_function_properties_l720_72062

-- Define the function f(x) = ax³ + bx²
def f (a b x : ℝ) : ℝ := a * x^3 + b * x^2

-- Define the derivative of f
def f_deriv (a b x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x

theorem cubic_function_properties (a b : ℝ) :
  f a b 1 = 3 ∧ f_deriv a b 1 = 0 →
  (a = -6 ∧ b = 9) ∧
  (∀ x : ℝ, f (-6) 9 x ≥ f (-6) 9 0) ∧
  f (-6) 9 0 = 0 := by
  sorry

#check cubic_function_properties

end cubic_function_properties_l720_72062


namespace unique_recurrence_sequence_l720_72036

/-- A sequence of integers satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧
  a 2 > 1 ∧
  ∀ n : ℕ, n ≥ 1 → (a (n + 1))^3 + 1 = (a n) * (a (n + 2))

/-- The theorem stating the existence and uniqueness of the sequence -/
theorem unique_recurrence_sequence :
  ∃! a : ℕ → ℤ, RecurrenceSequence a :=
sorry

end unique_recurrence_sequence_l720_72036


namespace fraction_inequality_l720_72026

theorem fraction_inequality (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : c < d) (h4 : d < 0) : 
  b / c > a / d := by
sorry

end fraction_inequality_l720_72026


namespace tulip_fraction_l720_72045

-- Define the total number of flowers (arbitrary positive real number)
variable (total : ℝ) (total_pos : 0 < total)

-- Define the number of each type of flower
variable (pink_roses : ℝ) (red_roses : ℝ) (pink_tulips : ℝ) (red_tulips : ℝ)

-- All flowers are either roses or tulips, and either pink or red
axiom flower_sum : pink_roses + red_roses + pink_tulips + red_tulips = total

-- 1/4 of pink flowers are roses
axiom pink_rose_ratio : pink_roses = (1/4) * (pink_roses + pink_tulips)

-- 1/3 of red flowers are tulips
axiom red_tulip_ratio : red_tulips = (1/3) * (red_roses + red_tulips)

-- 7/10 of all flowers are red
axiom red_flower_ratio : red_roses + red_tulips = (7/10) * total

-- Theorem: The fraction of flowers that are tulips is 11/24
theorem tulip_fraction :
  (pink_tulips + red_tulips) / total = 11/24 := by sorry

end tulip_fraction_l720_72045


namespace tree_height_l720_72049

theorem tree_height (tree_shadow : ℝ) (flagpole_shadow : ℝ) (flagpole_height : ℝ)
  (h1 : tree_shadow = 8)
  (h2 : flagpole_shadow = 100)
  (h3 : flagpole_height = 150) :
  (tree_shadow * flagpole_height) / flagpole_shadow = 12 := by
  sorry

end tree_height_l720_72049


namespace range_of_m_l720_72066

theorem range_of_m (a b m : ℝ) (h1 : 3 * a + 4 / b = 1) (h2 : a > 0) (h3 : b > 0)
  (h4 : ∀ (a b : ℝ), a > 0 → b > 0 → 3 * a + 4 / b = 1 → 1 / a + 3 * b > m) :
  m < 27 := by
  sorry

end range_of_m_l720_72066


namespace total_covid_cases_l720_72041

/-- Theorem: Total COVID-19 cases in New York, California, and Texas --/
theorem total_covid_cases (new_york california texas : ℕ) : 
  new_york = 2000 →
  california = new_york / 2 →
  california = texas + 400 →
  new_york + california + texas = 3600 := by
  sorry

end total_covid_cases_l720_72041


namespace smallest_square_containing_circle_l720_72002

theorem smallest_square_containing_circle (r : ℝ) (h : r = 7) :
  (2 * r) ^ 2 = 196 := by
  sorry

end smallest_square_containing_circle_l720_72002


namespace circle_radius_proof_l720_72043

theorem circle_radius_proof (a : ℝ) : 
  a > 0 ∧ 
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ y = 2*x^2 - 27 ∧ (x - a)^2 + (y - (2*a^2 - 27))^2 = a^2) ∧
  a^2 = (4*a - 3*(2*a^2 - 27))^2 / (4^2 + 3^2) →
  a = 9/2 :=
by sorry

end circle_radius_proof_l720_72043


namespace circle_equation_through_points_l720_72003

theorem circle_equation_through_points :
  let general_circle_eq (x y D E F : ℝ) := x^2 + y^2 + D*x + E*y + F = 0
  let specific_circle_eq (x y : ℝ) := x^2 + y^2 - 4*x - 6*y = 0
  (∀ x y, general_circle_eq x y (-4) (-6) 0 ↔ specific_circle_eq x y) ∧
  specific_circle_eq 0 0 ∧
  specific_circle_eq 4 0 ∧
  specific_circle_eq (-1) 1 := by
  sorry

end circle_equation_through_points_l720_72003


namespace seven_n_representable_l720_72023

theorem seven_n_representable (n a b : ℤ) (h : n = a^2 + a*b + b^2) :
  ∃ x y : ℤ, 7*n = x^2 + x*y + y^2 := by sorry

end seven_n_representable_l720_72023


namespace flag_distribution_l720_72034

theorem flag_distribution (total_flags : ℕ) (blue_flags red_flags : ℕ) :
  total_flags % 2 = 0 →
  blue_flags + red_flags = total_flags →
  (3 * total_flags / 10 : ℚ) = blue_flags →
  (3 * total_flags / 10 : ℚ) = red_flags →
  (total_flags / 10 : ℚ) = (blue_flags + red_flags - total_flags / 2 : ℚ) :=
by sorry

end flag_distribution_l720_72034


namespace problem_statement_l720_72083

theorem problem_statement (a b c d : ℤ) (x : ℝ) : 
  x = (a + b * Real.sqrt c) / d →
  (7 * x / 8) + 2 = 4 / x →
  (a * c * d) / b = -7 := by
sorry

end problem_statement_l720_72083


namespace farm_animals_l720_72098

theorem farm_animals (cows chickens ducks : ℕ) : 
  (4 * cows + 2 * chickens + 2 * ducks = 24 + 2 * (cows + chickens + ducks)) →
  (ducks = chickens / 2) →
  (cows = 12) := by
sorry

end farm_animals_l720_72098


namespace hexagon_toothpicks_l720_72013

/-- Represents a hexagonal pattern of small equilateral triangles -/
structure HexagonalPattern :=
  (max_row_triangles : ℕ)

/-- Calculates the total number of small triangles in the hexagonal pattern -/
def total_triangles (h : HexagonalPattern) : ℕ :=
  let half_triangles := (h.max_row_triangles * (h.max_row_triangles + 1)) / 2
  2 * half_triangles + h.max_row_triangles

/-- Calculates the number of boundary toothpicks in the hexagonal pattern -/
def boundary_toothpicks (h : HexagonalPattern) : ℕ :=
  6 * h.max_row_triangles

/-- Calculates the total number of toothpicks required to construct the hexagonal pattern -/
def total_toothpicks (h : HexagonalPattern) : ℕ :=
  3 * total_triangles h - boundary_toothpicks h

/-- Theorem stating that a hexagonal pattern with 1001 triangles in its largest row requires 3006003 toothpicks -/
theorem hexagon_toothpicks :
  let h : HexagonalPattern := ⟨1001⟩
  total_toothpicks h = 3006003 :=
by sorry

end hexagon_toothpicks_l720_72013


namespace mayor_harvey_flowers_l720_72012

/-- Represents the quantities of flowers for an institution -/
structure FlowerQuantities :=
  (roses : ℕ)
  (tulips : ℕ)
  (lilies : ℕ)

/-- Calculates the total number of flowers for given quantities -/
def totalFlowers (quantities : FlowerQuantities) : ℕ :=
  quantities.roses + quantities.tulips + quantities.lilies

/-- Theorem: The total number of flowers Mayor Harvey needs to buy is 855 -/
theorem mayor_harvey_flowers :
  let nursing_home : FlowerQuantities := ⟨90, 80, 100⟩
  let shelter : FlowerQuantities := ⟨120, 75, 95⟩
  let maternity_ward : FlowerQuantities := ⟨100, 110, 85⟩
  totalFlowers nursing_home + totalFlowers shelter + totalFlowers maternity_ward = 855 :=
by
  sorry

#eval let nursing_home : FlowerQuantities := ⟨90, 80, 100⟩
      let shelter : FlowerQuantities := ⟨120, 75, 95⟩
      let maternity_ward : FlowerQuantities := ⟨100, 110, 85⟩
      totalFlowers nursing_home + totalFlowers shelter + totalFlowers maternity_ward

end mayor_harvey_flowers_l720_72012


namespace hyperbola_min_value_l720_72086

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    with one asymptote having a slope angle of π/3 and eccentricity e,
    the minimum value of (a² + e)/b is 2√6/3 -/
theorem hyperbola_min_value (a b : ℝ) (e : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (b / a = Real.sqrt 3) →
  (∃ c : ℝ, c^2 = a^2 + b^2 ∧ e = c / a) →
  (∀ k : ℝ, k > 0 → (a^2 + e) / b ≥ 2 * Real.sqrt 6 / 3) ∧
  (∃ k : ℝ, k > 0 ∧ (a^2 + e) / b = 2 * Real.sqrt 6 / 3) := by
sorry

end hyperbola_min_value_l720_72086


namespace rectangular_field_length_l720_72080

theorem rectangular_field_length (w l : ℝ) (h1 : l = 2 * w) (h2 : 81 = (1 / 8) * (l * w)) :
  l = 36 :=
by sorry

end rectangular_field_length_l720_72080


namespace tv_show_main_characters_l720_72052

/-- Represents the TV show payment structure and calculates the number of main characters -/
def tv_show_characters : ℕ := by
  -- Define the number of minor characters
  let minor_characters : ℕ := 4
  -- Define the payment for each minor character
  let minor_payment : ℕ := 15000
  -- Define the total payment per episode
  let total_payment : ℕ := 285000
  -- Calculate the payment for each main character (3 times minor payment)
  let main_payment : ℕ := 3 * minor_payment
  -- Calculate the total payment for minor characters
  let minor_total : ℕ := minor_characters * minor_payment
  -- Calculate the remaining payment for main characters
  let main_total : ℕ := total_payment - minor_total
  -- Calculate the number of main characters
  exact main_total / main_payment

/-- Theorem stating that the number of main characters in the TV show is 5 -/
theorem tv_show_main_characters :
  tv_show_characters = 5 := by
  sorry

end tv_show_main_characters_l720_72052


namespace min_mn_tangent_line_circle_l720_72024

/-- Given positive real numbers m and n, if the line (m+1)x + (n+1)y - 2 = 0 is tangent to the circle (x-1)^2 + (y-1)^2 = 1, then the minimum value of mn is 3 + 2√2. -/
theorem min_mn_tangent_line_circle (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_tangent : ∀ x y : ℝ, (m + 1) * x + (n + 1) * y - 2 = 0 → (x - 1)^2 + (y - 1)^2 ≥ 1) :
  ∃ (min_mn : ℝ), min_mn = 3 + 2 * Real.sqrt 2 ∧ m * n ≥ min_mn := by
  sorry

end min_mn_tangent_line_circle_l720_72024


namespace square_area_proof_l720_72040

theorem square_area_proof (x : ℚ) : 
  (5 * x - 22 : ℚ) = (34 - 4 * x) → 
  (5 * x - 22 : ℚ) > 0 →
  ((5 * x - 22) ^ 2 : ℚ) = 6724 / 81 := by
sorry

end square_area_proof_l720_72040


namespace unique_quadratic_solution_l720_72082

theorem unique_quadratic_solution (p : ℝ) : 
  (p ≠ 0 ∧ ∃! x, p * x^2 - 10 * x + 2 = 0) ↔ p = 12.5 := by
  sorry

end unique_quadratic_solution_l720_72082


namespace parallel_transitive_l720_72077

-- Define the type for lines
def Line : Type := ℝ → ℝ → Prop

-- Define the parallel relation
def parallel (l1 l2 : Line) : Prop := sorry

-- Theorem statement
theorem parallel_transitive (a b c : Line) :
  parallel a b → parallel b c → parallel a c := by sorry

end parallel_transitive_l720_72077


namespace sum_of_A_and_C_l720_72073

def problem (A B C D : ℕ) : Prop :=
  A ∈ ({2, 3, 4, 5} : Set ℕ) ∧
  B ∈ ({2, 3, 4, 5} : Set ℕ) ∧
  C ∈ ({2, 3, 4, 5} : Set ℕ) ∧
  D ∈ ({2, 3, 4, 5} : Set ℕ) ∧
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  (A : ℚ) / B - (C : ℚ) / D = 1

theorem sum_of_A_and_C (A B C D : ℕ) (h : problem A B C D) : A + C = 8 := by
  sorry

end sum_of_A_and_C_l720_72073


namespace linear_function_quadrants_l720_72097

/-- A linear function passing through the first, second, and third quadrants implies positive slope and y-intercept -/
theorem linear_function_quadrants (k b : ℝ) : 
  (∀ x y : ℝ, y = k * x + b → 
    (∃ x₁ y₁, x₁ > 0 ∧ y₁ > 0 ∧ y₁ = k * x₁ + b) ∧ 
    (∃ x₂ y₂, x₂ < 0 ∧ y₂ > 0 ∧ y₂ = k * x₂ + b) ∧ 
    (∃ x₃ y₃, x₃ < 0 ∧ y₃ < 0 ∧ y₃ = k * x₃ + b)) →
  k > 0 ∧ b > 0 := by
sorry

end linear_function_quadrants_l720_72097


namespace plates_added_before_fall_l720_72025

theorem plates_added_before_fall (initial_plates : Nat) (second_addition : Nat) (total_plates : Nat)
  (h1 : initial_plates = 27)
  (h2 : second_addition = 37)
  (h3 : total_plates = 83) :
  total_plates - (initial_plates + second_addition) = 19 := by
  sorry

end plates_added_before_fall_l720_72025


namespace solution_set_quadratic_inequality_l720_72084

theorem solution_set_quadratic_inequality :
  {x : ℝ | x^2 + x - 2 ≤ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 1} := by sorry

end solution_set_quadratic_inequality_l720_72084


namespace binomial_coefficient_ratio_l720_72085

theorem binomial_coefficient_ratio (n k : ℕ) : 
  (Nat.choose n k : ℚ) / (Nat.choose n (k + 1) : ℚ) = 1 / 3 ∧
  (Nat.choose n (k + 1) : ℚ) / (Nat.choose n (k + 2) : ℚ) = 1 / 2 →
  n + k = 7 := by
  sorry

end binomial_coefficient_ratio_l720_72085


namespace sum_of_coordinates_on_h_l720_72081

def g (x : ℝ) : ℝ := x + 3

def h (x : ℝ) : ℝ := (g x)^2

theorem sum_of_coordinates_on_h : ∃ (x y : ℝ), 
  (2, 5) = (2, g 2) ∧ 
  (x, y) = (2, h 2) ∧ 
  x + y = 27 := by sorry

end sum_of_coordinates_on_h_l720_72081


namespace profit_difference_l720_72030

/-- The profit difference between selling a certain house and a standard house -/
theorem profit_difference (C : ℝ) : 
  let certain_house_cost : ℝ := C + 100000
  let standard_house_price : ℝ := 320000
  let certain_house_price : ℝ := 1.5 * standard_house_price
  let certain_house_profit : ℝ := certain_house_price - certain_house_cost
  let standard_house_profit : ℝ := standard_house_price - C
  certain_house_profit - standard_house_profit = 60000 := by
  sorry

#check profit_difference

end profit_difference_l720_72030


namespace janet_action_figures_l720_72001

theorem janet_action_figures (initial : ℕ) (sold : ℕ) (final_total : ℕ) :
  initial = 10 →
  sold = 6 →
  final_total = 24 →
  let remaining := initial - sold
  let brother_gift := 2 * remaining
  let before_new := remaining + brother_gift
  final_total - before_new = 12 :=
by sorry

end janet_action_figures_l720_72001


namespace scoops_left_is_16_l720_72069

/-- Represents the number of scoops in a carton of ice cream -/
def scoops_per_carton : ℕ := 10

/-- Represents the number of cartons Mary has -/
def marys_cartons : ℕ := 3

/-- Represents the number of scoops Ethan wants -/
def ethans_scoops : ℕ := 2

/-- Represents the number of people (Lucas, Danny, Connor) who want 2 scoops of chocolate each -/
def chocolate_lovers : ℕ := 3

/-- Represents the number of scoops each chocolate lover wants -/
def scoops_per_chocolate_lover : ℕ := 2

/-- Represents the number of scoops Olivia wants -/
def olivias_scoops : ℕ := 2

/-- Represents how many times more scoops Shannon wants compared to Olivia -/
def shannons_multiplier : ℕ := 2

/-- Theorem stating that the number of scoops left is 16 -/
theorem scoops_left_is_16 : 
  marys_cartons * scoops_per_carton - 
  (ethans_scoops + 
   chocolate_lovers * scoops_per_chocolate_lover + 
   olivias_scoops + 
   shannons_multiplier * olivias_scoops) = 16 := by
  sorry

end scoops_left_is_16_l720_72069


namespace problem_statements_l720_72029

theorem problem_statements :
  (¬ ∀ a b c : ℝ, a > b → a * c^2 > b * c^2) ∧
  (∀ a b : ℝ, a > |b| → a^2 > b^2) ∧
  (∀ a b : ℝ, a > b → a^3 > b^3) ∧
  (¬ ∀ a b : ℝ, |a| > b → a^2 > b^2) :=
by sorry

end problem_statements_l720_72029


namespace bags_sold_on_tuesday_l720_72027

theorem bags_sold_on_tuesday (total_stock : ℕ) (monday_sales wednesday_sales thursday_sales friday_sales : ℕ) 
  (h1 : total_stock = 600)
  (h2 : monday_sales = 25)
  (h3 : wednesday_sales = 100)
  (h4 : thursday_sales = 110)
  (h5 : friday_sales = 145)
  (h6 : (total_stock : ℝ) * 0.25 = total_stock - (monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales)) :
  tuesday_sales = 70 := by
  sorry

end bags_sold_on_tuesday_l720_72027


namespace min_value_of_reciprocal_sum_l720_72058

theorem min_value_of_reciprocal_sum (p q r : ℝ) (a b : ℝ) : 
  0 < p ∧ 0 < q ∧ 0 < r →
  p < q ∧ q < r →
  p^3 - a*p^2 + b*p - 48 = 0 →
  q^3 - a*q^2 + b*q - 48 = 0 →
  r^3 - a*r^2 + b*r - 48 = 0 →
  1/p + 2/q + 3/r ≥ 3/2 ∧ ∃ p' q' r' a' b', 
    0 < p' ∧ 0 < q' ∧ 0 < r' ∧
    p' < q' ∧ q' < r' ∧
    p'^3 - a'*p'^2 + b'*p' - 48 = 0 ∧
    q'^3 - a'*q'^2 + b'*q' - 48 = 0 ∧
    r'^3 - a'*r'^2 + b'*r' - 48 = 0 ∧
    1/p' + 2/q' + 3/r' = 3/2 :=
by sorry

end min_value_of_reciprocal_sum_l720_72058


namespace modulus_of_x_is_sqrt_10_l720_72014

-- Define the complex number x
def x : ℂ := sorry

-- State the theorem
theorem modulus_of_x_is_sqrt_10 :
  x + Complex.I = (2 - Complex.I) / Complex.I →
  Complex.abs x = Real.sqrt 10 := by
  sorry

end modulus_of_x_is_sqrt_10_l720_72014


namespace g_of_3_equals_64_l720_72042

/-- The function g satisfies 4g(x) - 3g(1/x) = x^2 for all nonzero x -/
def g_property (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → 4 * g x - 3 * g (1 / x) = x^2

/-- Given g satisfying the property, prove that g(3) = 64 -/
theorem g_of_3_equals_64 (g : ℝ → ℝ) (h : g_property g) : g 3 = 64 := by
  sorry

end g_of_3_equals_64_l720_72042


namespace prob_non_white_ball_l720_72021

/-- The probability of drawing a non-white ball from a bag -/
theorem prob_non_white_ball (white yellow red : ℕ) (h : white = 6 ∧ yellow = 5 ∧ red = 4) :
  (yellow + red) / (white + yellow + red) = 3 / 5 := by
  sorry

end prob_non_white_ball_l720_72021


namespace spurs_basketball_count_l720_72006

theorem spurs_basketball_count :
  let num_players : ℕ := 22
  let balls_per_player : ℕ := 11
  num_players * balls_per_player = 242 :=
by sorry

end spurs_basketball_count_l720_72006


namespace largest_number_with_digit_constraints_l720_72070

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def product_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * product_of_digits (n / 10)

theorem largest_number_with_digit_constraints : 
  ∀ n : ℕ, sum_of_digits n = 13 ∧ product_of_digits n = 36 → n ≤ 3322111 :=
by sorry

end largest_number_with_digit_constraints_l720_72070


namespace circle_tangent_to_parallel_lines_l720_72022

-- Define the parallel lines
def line1 (x y : ℝ) : Prop := x + 3 * y - 5 = 0
def line2 (x y : ℝ) : Prop := x + 3 * y - 3 = 0

-- Define the line containing the center of the circle
def centerLine (x y : ℝ) : Prop := 2 * x + y + 3 = 0

-- Define the circle equation
def circleEquation (x y : ℝ) : Prop := (x + 13/5)^2 + (y - 11/5)^2 = 1/10

-- Theorem stating the circle equation given the conditions
theorem circle_tangent_to_parallel_lines :
  ∀ (C : Set (ℝ × ℝ)),
  (∃ (x₁ y₁ : ℝ), (x₁, y₁) ∈ C ∧ line1 x₁ y₁) ∧
  (∃ (x₂ y₂ : ℝ), (x₂, y₂) ∈ C ∧ line2 x₂ y₂) ∧
  (∃ (x₀ y₀ : ℝ), (x₀, y₀) ∈ C ∧ centerLine x₀ y₀) →
  ∀ (x y : ℝ), (x, y) ∈ C ↔ circleEquation x y :=
sorry

end circle_tangent_to_parallel_lines_l720_72022


namespace total_age_problem_l720_72053

theorem total_age_problem (a b c : ℕ) : 
  b = 4 → a = b + 2 → b = 2 * c → a + b + c = 12 := by
  sorry

end total_age_problem_l720_72053


namespace work_completion_time_l720_72088

/-- The time (in days) it takes for A to complete the work alone -/
def a_time : ℝ := 30

/-- The time (in days) it takes for A and B to complete the work together -/
def ab_time : ℝ := 19.411764705882355

/-- The time (in days) it takes for B to complete the work alone -/
def b_time : ℝ := 55

/-- Theorem stating that if A can do the work in 30 days, and A and B together can do the work in 19.411764705882355 days, then B can do the work alone in 55 days -/
theorem work_completion_time : 
  (1 / a_time + 1 / b_time = 1 / ab_time) ∧ 
  (a_time > 0) ∧ (b_time > 0) ∧ (ab_time > 0) := by
  sorry

end work_completion_time_l720_72088


namespace scientific_notation_equivalence_l720_72008

theorem scientific_notation_equivalence : 
  8200000 = 8.2 * (10 : ℝ) ^ 6 := by sorry

end scientific_notation_equivalence_l720_72008


namespace snake_paint_calculation_l720_72078

theorem snake_paint_calculation (cube_paint : ℕ) (snake_length : ℕ) (segment_length : ℕ) 
  (segment_paint : ℕ) (end_paint : ℕ) : 
  cube_paint = 60 → 
  snake_length = 2016 → 
  segment_length = 6 → 
  segment_paint = 240 → 
  end_paint = 20 → 
  (snake_length / segment_length * segment_paint + end_paint : ℕ) = 80660 := by
  sorry

end snake_paint_calculation_l720_72078


namespace melanie_dimes_l720_72011

/-- The number of dimes Melanie has after receiving dimes from her parents -/
def total_dimes (initial : ℕ) (from_dad : ℕ) (from_mom : ℕ) : ℕ :=
  initial + from_dad + from_mom

/-- Proof that Melanie has 19 dimes after receiving dimes from her parents -/
theorem melanie_dimes : total_dimes 7 8 4 = 19 := by
  sorry

end melanie_dimes_l720_72011


namespace base_number_proof_l720_72054

theorem base_number_proof (x n : ℕ) (h1 : 4 * x^(2*n) = 4^22) (h2 : n = 21) : x = 2 := by
  sorry

end base_number_proof_l720_72054


namespace fraction_comparison_l720_72048

theorem fraction_comparison : (200200201 : ℚ) / 200200203 > (300300301 : ℚ) / 300300304 := by
  sorry

end fraction_comparison_l720_72048


namespace digit_with_value_difference_l720_72074

def numeral : List Nat := [6, 5, 7, 9, 3]

def local_value (digit : Nat) (place : Nat) : Nat :=
  digit * (10 ^ place)

def face_value (digit : Nat) : Nat := digit

theorem digit_with_value_difference (diff : Nat) :
  ∃ (index : Fin 5), 
    local_value (numeral[index]) (4 - index) - face_value (numeral[index]) = diff →
    numeral[index] = 7 :=
by
  sorry

end digit_with_value_difference_l720_72074


namespace lindas_savings_l720_72016

/-- Given that Linda spent 3/4 of her savings on furniture and the rest on a TV costing $500,
    prove that her original savings were $2000. -/
theorem lindas_savings (savings : ℝ) : 
  (3/4 : ℝ) * savings + 500 = savings → savings = 2000 := by
  sorry

end lindas_savings_l720_72016


namespace larger_number_problem_l720_72095

theorem larger_number_problem (L S : ℕ) (h1 : L > S) (h2 : L - S = 1365) (h3 : L = 6 * S + 35) : L = 1631 := by
  sorry

end larger_number_problem_l720_72095


namespace greatest_valid_integer_l720_72064

def is_valid (n : ℕ) : Prop :=
  n < 200 ∧ Nat.gcd n 24 = 2

theorem greatest_valid_integer : 
  is_valid 194 ∧ ∀ m : ℕ, is_valid m → m ≤ 194 :=
sorry

end greatest_valid_integer_l720_72064


namespace max_victory_margin_l720_72087

/-- Represents the vote count for a candidate in two time periods -/
structure VoteCount where
  first_period : ℕ
  second_period : ℕ

/-- The election scenario with given conditions -/
def ElectionScenario : Prop :=
  ∃ (petya vasya : VoteCount),
    -- Total votes condition
    petya.first_period + petya.second_period + vasya.first_period + vasya.second_period = 27 ∧
    -- First two hours condition
    petya.first_period = vasya.first_period + 9 ∧
    -- Last hour condition
    vasya.second_period = petya.second_period + 9 ∧
    -- Petya wins condition
    petya.first_period + petya.second_period > vasya.first_period + vasya.second_period

/-- The theorem stating the maximum possible margin of Petya's victory -/
theorem max_victory_margin (h : ElectionScenario) :
  ∃ (petya vasya : VoteCount),
    petya.first_period + petya.second_period - (vasya.first_period + vasya.second_period) ≤ 9 :=
  sorry

end max_victory_margin_l720_72087
