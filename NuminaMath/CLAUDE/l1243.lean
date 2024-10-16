import Mathlib

namespace NUMINAMATH_CALUDE_art_club_election_l1243_124307

theorem art_club_election (total_candidates : ℕ) (past_officers : ℕ) (positions : ℕ) 
  (h1 : total_candidates = 18) 
  (h2 : past_officers = 8) 
  (h3 : positions = 6) :
  (Nat.choose total_candidates positions) - 
  (Nat.choose (total_candidates - past_officers) positions) = 18354 := by
  sorry

end NUMINAMATH_CALUDE_art_club_election_l1243_124307


namespace NUMINAMATH_CALUDE_add_fractions_three_ninths_seven_twelfths_l1243_124383

theorem add_fractions_three_ninths_seven_twelfths :
  3 / 9 + 7 / 12 = 11 / 12 := by sorry

end NUMINAMATH_CALUDE_add_fractions_three_ninths_seven_twelfths_l1243_124383


namespace NUMINAMATH_CALUDE_triangle_side_length_l1243_124370

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  c = 3 →
  C = π / 3 →
  Real.sin B = 2 * Real.sin A →
  a^2 + b^2 - 2 * a * b * Real.cos C = c^2 →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  a = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1243_124370


namespace NUMINAMATH_CALUDE_john_twice_sam_age_l1243_124327

def john_age (sam_age : ℕ) : ℕ := 3 * sam_age

def sam_current_age : ℕ := 7 + 2

theorem john_twice_sam_age (years : ℕ) : 
  john_age sam_current_age + years = 2 * (sam_current_age + years) → years = 9 := by
  sorry

end NUMINAMATH_CALUDE_john_twice_sam_age_l1243_124327


namespace NUMINAMATH_CALUDE_original_number_proof_l1243_124324

theorem original_number_proof (final_number : ℝ) (increase_percentage : ℝ) 
  (h1 : final_number = 1680)
  (h2 : increase_percentage = 110) : 
  ∃ (original : ℝ), original * (1 + increase_percentage / 100) = final_number ∧ original = 800 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l1243_124324


namespace NUMINAMATH_CALUDE_correct_systematic_sample_l1243_124332

/-- Represents a systematic sampling of n items from a set of m items -/
def SystematicSample (m n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => (i + 1) * (m / n))

/-- The problem statement -/
theorem correct_systematic_sample :
  SystematicSample 50 5 = [10, 20, 30, 40, 50] := by
  sorry

end NUMINAMATH_CALUDE_correct_systematic_sample_l1243_124332


namespace NUMINAMATH_CALUDE_first_grade_boys_count_l1243_124368

theorem first_grade_boys_count (num_classrooms : ℕ) (num_girls : ℕ) (students_per_classroom : ℕ) :
  num_classrooms = 4 →
  num_girls = 44 →
  students_per_classroom = 25 →
  (∀ classroom, classroom ≤ num_classrooms →
    (num_girls / num_classrooms = students_per_classroom / 2)) →
  num_girls = 44 :=
by
  sorry

end NUMINAMATH_CALUDE_first_grade_boys_count_l1243_124368


namespace NUMINAMATH_CALUDE_eight_coin_stack_exists_fourteen_mm_stack_has_eight_coins_l1243_124336

/-- Represents the types of coins --/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter

/-- Returns the thickness of a given coin in millimeters --/
def coinThickness (c : Coin) : ℚ :=
  match c with
  | Coin.Penny => 155/100
  | Coin.Nickel => 195/100
  | Coin.Dime => 135/100
  | Coin.Quarter => 175/100

/-- Represents a stack of coins --/
def CoinStack := List Coin

/-- Calculates the height of a coin stack in millimeters --/
def stackHeight (stack : CoinStack) : ℚ :=
  stack.foldl (fun acc c => acc + coinThickness c) 0

/-- Theorem: There exists a stack of 8 coins with a height of exactly 14 mm --/
theorem eight_coin_stack_exists : ∃ (stack : CoinStack), stackHeight stack = 14 ∧ stack.length = 8 := by
  sorry

/-- Theorem: Any stack of coins with a height of exactly 14 mm must contain 8 coins --/
theorem fourteen_mm_stack_has_eight_coins (stack : CoinStack) :
  stackHeight stack = 14 → stack.length = 8 := by
  sorry

end NUMINAMATH_CALUDE_eight_coin_stack_exists_fourteen_mm_stack_has_eight_coins_l1243_124336


namespace NUMINAMATH_CALUDE_pascal_contest_average_age_l1243_124320

/-- Represents an age in years and months -/
structure Age where
  years : ℕ
  months : ℕ
  h : months < 12

/-- Converts an Age to total months -/
def ageToMonths (a : Age) : ℕ := a.years * 12 + a.months

/-- Converts total months to an Age -/
def monthsToAge (m : ℕ) : Age :=
  { years := m / 12
  , months := m % 12
  , h := by sorry }

/-- The average age of three contestants in the Pascal Contest -/
theorem pascal_contest_average_age (a1 a2 a3 : Age)
  (h1 : a1 = { years := 14, months := 9, h := by sorry })
  (h2 : a2 = { years := 15, months := 1, h := by sorry })
  (h3 : a3 = { years := 14, months := 8, h := by sorry }) :
  monthsToAge ((ageToMonths a1 + ageToMonths a2 + ageToMonths a3) / 3) =
  { years := 14, months := 10, h := by sorry } :=
by sorry

end NUMINAMATH_CALUDE_pascal_contest_average_age_l1243_124320


namespace NUMINAMATH_CALUDE_alchemerion_age_proof_l1243_124310

/-- Alchemerion's age in years -/
def alchemerion_age : ℕ := 277

/-- Alchemerion's son's age in years -/
def son_age : ℕ := alchemerion_age / 3

/-- Alchemerion's father's age in years -/
def father_age : ℕ := 2 * alchemerion_age + 40

/-- The sum of Alchemerion's, his son's, and his father's ages -/
def total_age : ℕ := alchemerion_age + son_age + father_age

theorem alchemerion_age_proof :
  alchemerion_age = 3 * son_age ∧
  father_age = 2 * alchemerion_age + 40 ∧
  total_age = 1240 →
  alchemerion_age = 277 := by
  sorry

end NUMINAMATH_CALUDE_alchemerion_age_proof_l1243_124310


namespace NUMINAMATH_CALUDE_sequence_p_bounded_l1243_124356

def isPrime (n : ℕ) : Prop := sorry

def largestPrimeDivisor (n : ℕ) : ℕ := sorry

def sequenceP : ℕ → ℕ
  | 0 => 2  -- Assuming the sequence starts with 2
  | 1 => 3  -- Assuming the second prime is 3
  | (n + 2) => largestPrimeDivisor (sequenceP (n + 1) + sequenceP n + 2008)

theorem sequence_p_bounded :
  ∃ (M : ℕ), ∀ (n : ℕ), sequenceP n ≤ M :=
sorry

end NUMINAMATH_CALUDE_sequence_p_bounded_l1243_124356


namespace NUMINAMATH_CALUDE_inequality_proof_l1243_124388

theorem inequality_proof (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_ineq : 2 * (a + b + c + d) ≥ a * b * c * d) : 
  a^2 + b^2 + c^2 + d^2 ≥ a * b * c * d := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1243_124388


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1243_124349

theorem max_value_of_expression (t : ℝ) : 
  (∃ (t_max : ℝ), ∀ (t : ℝ), ((3^t - 5*t)*t)/(9^t) ≤ ((3^t_max - 5*t_max)*t_max)/(9^t_max)) ∧ 
  (∃ (t_0 : ℝ), ((3^t_0 - 5*t_0)*t_0)/(9^t_0) = 1/20) := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1243_124349


namespace NUMINAMATH_CALUDE_block_arrangement_table_height_l1243_124359

/-- The height of the table in the block arrangement problem -/
def table_height : ℝ := 36

/-- The initial length measurement in the block arrangement -/
def initial_length : ℝ := 42

/-- The final length measurement in the block arrangement -/
def final_length : ℝ := 36

/-- The difference between block width and overlap in the first arrangement -/
def width_overlap_difference : ℝ := 6

theorem block_arrangement_table_height :
  ∃ (block_length block_width overlap : ℝ),
    block_length + table_height - overlap = initial_length ∧
    block_width + table_height - block_length = final_length ∧
    block_width = overlap + width_overlap_difference ∧
    table_height = 36 := by
  sorry

#check block_arrangement_table_height

end NUMINAMATH_CALUDE_block_arrangement_table_height_l1243_124359


namespace NUMINAMATH_CALUDE_red_balls_count_l1243_124357

/-- The number of black balls in the bag -/
def black_balls : ℕ := 3

/-- The probability of drawing a red ball -/
def red_probability : ℝ := 0.85

/-- The number of red balls in the bag -/
def red_balls : ℕ := 17

theorem red_balls_count :
  (red_balls : ℝ) / (red_balls + black_balls) = red_probability :=
by sorry

end NUMINAMATH_CALUDE_red_balls_count_l1243_124357


namespace NUMINAMATH_CALUDE_bologna_sandwich_count_l1243_124395

/-- Represents the number of sandwiches of each type -/
structure SandwichCount where
  cheese : ℕ
  bologna : ℕ
  peanutButter : ℕ

/-- The ratio of sandwiches -/
def sandwichRatio : ℕ → SandwichCount
  | x => { cheese := 1, bologna := x, peanutButter := 8 }

/-- The total number of sandwiches -/
def totalSandwiches : ℕ := 80

theorem bologna_sandwich_count :
  ∃ x : ℕ, 
    let ratio := sandwichRatio x
    (ratio.cheese + ratio.bologna + ratio.peanutButter) * y = totalSandwiches →
    ratio.bologna * y = 8 := by
  sorry

end NUMINAMATH_CALUDE_bologna_sandwich_count_l1243_124395


namespace NUMINAMATH_CALUDE_pushup_difference_l1243_124342

-- Define the number of push-ups for Zachary and the total
def zachary_pushups : ℕ := 44
def total_pushups : ℕ := 146

-- Define David's push-ups
def david_pushups : ℕ := total_pushups - zachary_pushups

-- State the theorem
theorem pushup_difference :
  david_pushups > zachary_pushups ∧
  david_pushups - zachary_pushups = 58 := by
  sorry

end NUMINAMATH_CALUDE_pushup_difference_l1243_124342


namespace NUMINAMATH_CALUDE_leggings_for_pets_l1243_124379

def number_of_leggings_needed (num_dogs : ℕ) (num_cats : ℕ) : ℕ :=
  let legs_per_animal : ℕ := 4
  let legs_per_legging : ℕ := 2
  let total_legs : ℕ := (num_dogs + num_cats) * legs_per_animal
  total_legs / legs_per_legging

theorem leggings_for_pets : number_of_leggings_needed 4 3 = 14 := by
  sorry

end NUMINAMATH_CALUDE_leggings_for_pets_l1243_124379


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1243_124345

theorem sufficient_but_not_necessary (p q : Prop) :
  (¬p → ¬q) ∧ ¬(¬q → ¬p) → (q → p) ∧ ¬(p → q) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1243_124345


namespace NUMINAMATH_CALUDE_opposite_numbers_iff_different_sign_l1243_124381

/-- Two real numbers are opposite if and only if they differ only in sign -/
theorem opposite_numbers_iff_different_sign (a b : ℝ) : 
  (a = -b) ↔ (abs a = abs b) :=
sorry

end NUMINAMATH_CALUDE_opposite_numbers_iff_different_sign_l1243_124381


namespace NUMINAMATH_CALUDE_x_plus_2y_squared_l1243_124325

theorem x_plus_2y_squared (x y : ℝ) (h1 : x * (x + 2*y) = 48) (h2 : y * (x + 2*y) = 72) : 
  (x + 2*y)^2 = 96 := by
sorry

end NUMINAMATH_CALUDE_x_plus_2y_squared_l1243_124325


namespace NUMINAMATH_CALUDE_F_3_f_4_equals_7_l1243_124339

-- Define the functions f and F
def f (a : ℝ) : ℝ := a - 2
def F (a b : ℝ) : ℝ := b^2 + a

-- State the theorem
theorem F_3_f_4_equals_7 : F 3 (f 4) = 7 := by
  sorry

end NUMINAMATH_CALUDE_F_3_f_4_equals_7_l1243_124339


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_squared_times_one_plus_i_l1243_124330

theorem imaginary_part_of_i_squared_times_one_plus_i :
  Complex.im (Complex.I^2 * (1 + Complex.I)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_squared_times_one_plus_i_l1243_124330


namespace NUMINAMATH_CALUDE_inequality_proof_l1243_124353

theorem inequality_proof (a b c : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c > 0) :
  a + b + c ≤ (a^2 + b^2)/(2*c) + (b^2 + c^2)/(2*a) + (c^2 + a^2)/(2*b) ∧
  (a^2 + b^2)/(2*c) + (b^2 + c^2)/(2*a) + (c^2 + a^2)/(2*b) ≤ a^3/(b*c) + b^3/(c*a) + c^3/(a*b) :=
by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1243_124353


namespace NUMINAMATH_CALUDE_journey_theorem_l1243_124386

/-- Represents the journey to Koschei's kingdom -/
structure Journey where
  total_distance : ℝ
  first_day_distance : ℝ
  second_day_distance : ℝ
  third_day_distance : ℝ
  fourth_day_distance : ℝ

/-- The conditions of Leshy's journey -/
def leshy_journey (j : Journey) : Prop :=
  j.first_day_distance = j.total_distance / 3 ∧
  j.second_day_distance = j.first_day_distance / 2 ∧
  j.third_day_distance = j.first_day_distance ∧
  j.fourth_day_distance = 100 ∧
  j.total_distance = j.first_day_distance + j.second_day_distance + j.third_day_distance + j.fourth_day_distance

theorem journey_theorem (j : Journey) (h : leshy_journey j) :
  j.total_distance = 600 ∧ j.fourth_day_distance = 100 := by
  sorry

#check journey_theorem

end NUMINAMATH_CALUDE_journey_theorem_l1243_124386


namespace NUMINAMATH_CALUDE_smallest_divisible_by_20_and_63_l1243_124394

theorem smallest_divisible_by_20_and_63 : ∀ n : ℕ, n > 0 ∧ 20 ∣ n ∧ 63 ∣ n → n ≥ 1260 := by
  sorry

#check smallest_divisible_by_20_and_63

end NUMINAMATH_CALUDE_smallest_divisible_by_20_and_63_l1243_124394


namespace NUMINAMATH_CALUDE_coals_per_bag_prove_coals_per_bag_l1243_124382

-- Define the constants from the problem
def coals_per_set : ℕ := 15
def minutes_per_set : ℕ := 20
def total_minutes : ℕ := 240
def num_bags : ℕ := 3

-- Define the theorem
theorem coals_per_bag : ℕ :=
  let sets_burned := total_minutes / minutes_per_set
  let total_coals_burned := sets_burned * coals_per_set
  total_coals_burned / num_bags

-- State the theorem to be proved
theorem prove_coals_per_bag : coals_per_bag = 60 := by
  sorry

end NUMINAMATH_CALUDE_coals_per_bag_prove_coals_per_bag_l1243_124382


namespace NUMINAMATH_CALUDE_ellipse_fixed_point_l1243_124333

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the line
def line (k m x y : ℝ) : Prop := y = k * x + m

-- Define the condition for the circle passing through the right vertex
def circle_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (x₁ - 2) * (x₂ - 2) + y₁ * y₂ = 0

theorem ellipse_fixed_point (k m x₁ y₁ x₂ y₂ : ℝ) :
  k ≠ 0 →
  ellipse x₁ y₁ →
  ellipse x₂ y₂ →
  line k m x₁ y₁ →
  line k m x₂ y₂ →
  (x₁ ≠ 2 ∨ y₁ ≠ 0) →
  (x₂ ≠ 2 ∨ y₂ ≠ 0) →
  (x₁ ≠ -2 ∨ y₁ ≠ 0) →
  (x₂ ≠ -2 ∨ y₂ ≠ 0) →
  circle_condition x₁ y₁ x₂ y₂ →
  ∃ (x : ℝ), x = 2/7 ∧ line k m x 0 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_fixed_point_l1243_124333


namespace NUMINAMATH_CALUDE_rs_fraction_l1243_124343

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the altitude CH
def altitude (t : Triangle) : ℝ × ℝ := sorry

-- Define the points R and S
def R (t : Triangle) : ℝ × ℝ := sorry
def S (t : Triangle) : ℝ × ℝ := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem rs_fraction (t : Triangle) :
  distance (t.A) (t.B) = 2023 →
  distance (t.A) (t.C) = 2022 →
  distance (t.B) (t.C) = 2021 →
  distance (R t) (S t) = 2021 / 2023 := by
  sorry

end NUMINAMATH_CALUDE_rs_fraction_l1243_124343


namespace NUMINAMATH_CALUDE_cosine_product_in_special_sequence_l1243_124303

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

theorem cosine_product_in_special_sequence (a₁ : ℝ) :
  let a := arithmetic_sequence a₁ (2 * Real.pi / 3)
  let S := {x | ∃ n : ℕ+, x = Real.cos (a n)}
  (∃ a b : ℝ, S = {a, b}) →
  ∃ a b : ℝ, S = {a, b} ∧ a * b = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_product_in_special_sequence_l1243_124303


namespace NUMINAMATH_CALUDE_intersection_point_l1243_124316

def f (x : ℝ) : ℝ := 4 * x - 2

theorem intersection_point :
  ∃ (x : ℝ), f x = 0 ∧ x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l1243_124316


namespace NUMINAMATH_CALUDE_reece_climbs_l1243_124351

-- Define constants
def keaton_ladder_feet : ℕ := 30
def keaton_climbs : ℕ := 20
def ladder_difference_feet : ℕ := 4
def total_climbed_inches : ℕ := 11880

-- Define functions
def feet_to_inches (feet : ℕ) : ℕ := feet * 12

def reece_ladder_feet : ℕ := keaton_ladder_feet - ladder_difference_feet

-- Theorem statement
theorem reece_climbs : 
  (feet_to_inches keaton_ladder_feet * keaton_climbs + 
   feet_to_inches reece_ladder_feet * 15 = total_climbed_inches) := by
sorry

end NUMINAMATH_CALUDE_reece_climbs_l1243_124351


namespace NUMINAMATH_CALUDE_two_thousandth_point_l1243_124393

/-- Represents a point in the first quadrant with integer coordinates -/
structure Point where
  x : Nat
  y : Nat

/-- The spiral numbering function that assigns a natural number to each point -/
def spiralNumber : Point → Nat := sorry

/-- The inverse function that finds the point corresponding to a given number -/
def spiralPoint : Nat → Point := sorry

/-- Theorem stating that the 2000th point in the spiral has coordinates (44, 24) -/
theorem two_thousandth_point : spiralPoint 2000 = Point.mk 44 24 := by sorry

end NUMINAMATH_CALUDE_two_thousandth_point_l1243_124393


namespace NUMINAMATH_CALUDE_decreasing_quadratic_implies_a_geq_two_l1243_124352

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x - 5

-- State the theorem
theorem decreasing_quadratic_implies_a_geq_two :
  ∀ a : ℝ, (∀ x ≤ 2, ∀ y ≤ 2, x < y → f a x > f a y) → a ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_decreasing_quadratic_implies_a_geq_two_l1243_124352


namespace NUMINAMATH_CALUDE_income_calculation_l1243_124305

/-- Calculates a person's income given the income to expenditure ratio and savings amount. -/
def calculate_income (income_ratio : ℕ) (expenditure_ratio : ℕ) (savings : ℕ) : ℕ :=
  (income_ratio * savings) / (income_ratio - expenditure_ratio)

/-- Theorem stating that given the specified conditions, the person's income is 10000. -/
theorem income_calculation (income_ratio : ℕ) (expenditure_ratio : ℕ) (savings : ℕ) 
  (h1 : income_ratio = 10)
  (h2 : expenditure_ratio = 7)
  (h3 : savings = 3000) :
  calculate_income income_ratio expenditure_ratio savings = 10000 := by
  sorry

end NUMINAMATH_CALUDE_income_calculation_l1243_124305


namespace NUMINAMATH_CALUDE_sqrt_fraction_sum_l1243_124380

theorem sqrt_fraction_sum : Real.sqrt (25 / 36 + 16 / 9) = Real.sqrt 89 / 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_sum_l1243_124380


namespace NUMINAMATH_CALUDE_company_size_l1243_124319

/-- Represents the number of employees in a company -/
structure Company where
  total : ℕ
  senior : ℕ
  sample_size : ℕ
  sample_senior : ℕ

/-- Given a company with 15 senior-titled employees and a stratified sample of 30 employees
    containing 3 senior-titled employees, the total number of employees is 150 -/
theorem company_size (c : Company)
  (h1 : c.senior = 15)
  (h2 : c.sample_size = 30)
  (h3 : c.sample_senior = 3)
  : c.total = 150 := by
  sorry

end NUMINAMATH_CALUDE_company_size_l1243_124319


namespace NUMINAMATH_CALUDE_largest_monochromatic_subgraph_2024_l1243_124301

/-- A 3-coloring of the edges of a complete graph -/
def ThreeColoring (n : ℕ) := Fin 3 → Sym2 (Fin n)

/-- A function that returns the size of the largest monochromatic connected subgraph -/
noncomputable def largestMonochromaticSubgraph (n : ℕ) (coloring : ThreeColoring n) : ℕ := sorry

theorem largest_monochromatic_subgraph_2024 :
  ∀ (coloring : ThreeColoring 2024),
  largestMonochromaticSubgraph 2024 coloring ≥ 1012 := by sorry

end NUMINAMATH_CALUDE_largest_monochromatic_subgraph_2024_l1243_124301


namespace NUMINAMATH_CALUDE_carl_winning_configurations_l1243_124375

def board_size : ℕ := 4

def winning_configurations : ℕ := 10

def remaining_cells_after_win : ℕ := 13

def ways_to_choose_three_from_twelve : ℕ := 220

theorem carl_winning_configurations :
  (winning_configurations * board_size * remaining_cells_after_win * ways_to_choose_three_from_twelve) = 114400 :=
by sorry

end NUMINAMATH_CALUDE_carl_winning_configurations_l1243_124375


namespace NUMINAMATH_CALUDE_relationship_abc_l1243_124318

theorem relationship_abc (a b c : ℝ) : 
  a = (1/2)^(2/3) → b = (1/5)^(2/3) → c = (1/2)^(1/3) → b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l1243_124318


namespace NUMINAMATH_CALUDE_cathy_worked_180_hours_l1243_124355

/-- Calculates the total hours worked by Cathy over 2 months, given the following conditions:
  * Normal work schedule is 20 hours per week
  * There are 4 weeks in a month
  * The job lasts for 2 months
  * Cathy covers an additional week of shifts (20 hours) due to Chris's illness
-/
def cathys_total_hours (hours_per_week : ℕ) (weeks_per_month : ℕ) (months : ℕ) (extra_week_hours : ℕ) : ℕ :=
  hours_per_week * weeks_per_month * months + extra_week_hours

/-- Proves that Cathy worked 180 hours during the 2 months -/
theorem cathy_worked_180_hours :
  cathys_total_hours 20 4 2 20 = 180 := by
  sorry

end NUMINAMATH_CALUDE_cathy_worked_180_hours_l1243_124355


namespace NUMINAMATH_CALUDE_geometric_sequence_21st_term_l1243_124329

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_21st_term
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_first_term : a 1 = 3)
  (h_common_product : ∀ n : ℕ, a n * a (n + 1) = 15) :
  a 21 = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_21st_term_l1243_124329


namespace NUMINAMATH_CALUDE_total_calories_l1243_124372

/-- The number of calories in a single candy bar -/
def calories_per_bar : ℕ := 3

/-- The number of candy bars -/
def num_bars : ℕ := 5

/-- Theorem: The total calories in 5 candy bars is 15 -/
theorem total_calories : calories_per_bar * num_bars = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_calories_l1243_124372


namespace NUMINAMATH_CALUDE_interest_difference_l1243_124373

theorem interest_difference (principal rate time : ℝ) 
  (h_principal : principal = 600)
  (h_rate : rate = 0.05)
  (h_time : time = 8) :
  principal - (principal * rate * time) = 360 := by
  sorry

end NUMINAMATH_CALUDE_interest_difference_l1243_124373


namespace NUMINAMATH_CALUDE_fraction_equivalence_l1243_124328

theorem fraction_equivalence : 
  (20 / 16 : ℚ) = 10 / 8 ∧
  (1 + 6 / 24 : ℚ) = 10 / 8 ∧
  (1 + 2 / 8 : ℚ) = 10 / 8 ∧
  (1 + 40 / 160 : ℚ) = 10 / 8 ∧
  (1 + 4 / 8 : ℚ) ≠ 10 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l1243_124328


namespace NUMINAMATH_CALUDE_unique_sums_count_l1243_124398

def bag_x : Finset ℕ := {1, 4, 7}
def bag_y : Finset ℕ := {3, 5, 8}

def possible_sums : Finset ℕ := (bag_x.product bag_y).image (λ (x, y) => x + y)

theorem unique_sums_count : possible_sums.card = 7 := by sorry

end NUMINAMATH_CALUDE_unique_sums_count_l1243_124398


namespace NUMINAMATH_CALUDE_polynomial_equality_existence_l1243_124371

theorem polynomial_equality_existence : 
  ∃ (a b c d : ℤ), ∀ (x : ℝ),
    (x^2 + a*x + b) * (x^2 + c*x + d) = x^4 - 2*x^3 + 3*x^2 - 4*x + 6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_existence_l1243_124371


namespace NUMINAMATH_CALUDE_prize_winning_condition_xiao_feng_inequality_l1243_124304

/-- Represents the school intelligence competition --/
structure Competition where
  total_questions : ℕ
  correct_points : ℤ
  incorrect_points : ℤ
  prize_threshold : ℤ

/-- Represents a student participating in the competition --/
structure Student where
  correct_answers : ℕ
  won_prize : Prop

/-- The specific competition described in the problem --/
def school_competition : Competition :=
  { total_questions := 20
  , correct_points := 5
  , incorrect_points := -2
  , prize_threshold := 75 }

/-- Theorem stating the condition for winning a prize --/
theorem prize_winning_condition (s : Student) (c : Competition) 
  (h1 : s.won_prize) 
  (h2 : s.correct_answers ≤ c.total_questions) :
  c.correct_points * s.correct_answers + 
  c.incorrect_points * (c.total_questions - s.correct_answers) > 
  c.prize_threshold := by
  sorry

/-- Theorem for Xiao Feng's specific case --/
theorem xiao_feng_inequality (x : ℕ) :
  x ≤ school_competition.total_questions →
  5 * x - 2 * (20 - x) > 75 := by
  sorry

end NUMINAMATH_CALUDE_prize_winning_condition_xiao_feng_inequality_l1243_124304


namespace NUMINAMATH_CALUDE_m_minus_n_squared_l1243_124390

theorem m_minus_n_squared (m n : ℝ) (h1 : m + n = 6) (h2 : m^2 + n^2 = 26) : 
  (m - n)^2 = 16 := by
sorry

end NUMINAMATH_CALUDE_m_minus_n_squared_l1243_124390


namespace NUMINAMATH_CALUDE_go_game_competition_l1243_124302

/-- Represents the probability of a player winning a single game -/
structure GameProbability where
  player_a : ℝ
  player_b : ℝ
  sum_to_one : player_a + player_b = 1

/-- Represents the state of the game after the first two games -/
structure GameState where
  a_wins : ℕ
  b_wins : ℕ
  total_games : a_wins + b_wins = 2

/-- The probability of the competition ending after 2 more games -/
def probability_end_in_two_more_games (p : GameProbability) : ℝ :=
  p.player_a * p.player_a + p.player_b * p.player_b

/-- The probability of player A winning the competition -/
def probability_a_wins (p : GameProbability) : ℝ :=
  p.player_a * p.player_a + 
  p.player_b * p.player_a * p.player_a + 
  p.player_a * p.player_b * p.player_a

theorem go_game_competition 
  (p : GameProbability) 
  (state : GameState) 
  (h_p : p.player_a = 0.6 ∧ p.player_b = 0.4) 
  (h_state : state.a_wins = 1 ∧ state.b_wins = 1) : 
  probability_end_in_two_more_games p = 0.52 ∧ 
  probability_a_wins p = 0.648 := by
  sorry


end NUMINAMATH_CALUDE_go_game_competition_l1243_124302


namespace NUMINAMATH_CALUDE_graph_equation_is_intersecting_lines_l1243_124354

theorem graph_equation_is_intersecting_lines :
  ∀ x y : ℝ, (x + y)^2 = x^2 + y^2 + 3*x*y ↔ x*y = 0 :=
by sorry

end NUMINAMATH_CALUDE_graph_equation_is_intersecting_lines_l1243_124354


namespace NUMINAMATH_CALUDE_opposite_of_2023_l1243_124361

theorem opposite_of_2023 : 
  (∀ x : ℤ, x + 2023 = 0 → x = -2023) ∧ (-2023 + 2023 = 0) := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l1243_124361


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1243_124346

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 - 5*x + 6 = 0

-- Define the condition x = 2
def condition (x : ℝ) : Prop := x = 2

-- Theorem statement
theorem sufficient_not_necessary :
  (∀ x : ℝ, quadratic_equation x ↔ (x = 2 ∨ x = 3)) →
  (∀ x : ℝ, condition x → quadratic_equation x) ∧
  ¬(∀ x : ℝ, quadratic_equation x → condition x) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1243_124346


namespace NUMINAMATH_CALUDE_farm_acreage_l1243_124344

theorem farm_acreage (total_acres sunflower_acres flax_acres : ℕ) : 
  total_acres = 240 →
  sunflower_acres = flax_acres + 80 →
  sunflower_acres + flax_acres = total_acres →
  flax_acres = 80 := by
  sorry

end NUMINAMATH_CALUDE_farm_acreage_l1243_124344


namespace NUMINAMATH_CALUDE_min_value_xy_over_x2_plus_y2_l1243_124366

theorem min_value_xy_over_x2_plus_y2 (x y : ℝ) 
  (hx : 1/4 ≤ x ∧ x ≤ 3/5) (hy : 1/5 ≤ y ∧ y ≤ 2/3) :
  x * y / (x^2 + y^2) ≥ 24/73 := by
  sorry

end NUMINAMATH_CALUDE_min_value_xy_over_x2_plus_y2_l1243_124366


namespace NUMINAMATH_CALUDE_subtract_negative_six_a_l1243_124377

theorem subtract_negative_six_a (a : ℝ) : (4 * a^2 - 3 * a + 7) - (-6 * a) = 4 * a^2 - 9 * a + 7 := by
  sorry

end NUMINAMATH_CALUDE_subtract_negative_six_a_l1243_124377


namespace NUMINAMATH_CALUDE_ratio_comparison_l1243_124384

theorem ratio_comparison (y : ℕ) (h : y > 4) : (3 : ℚ) / 4 < 4 / y := by
  sorry

end NUMINAMATH_CALUDE_ratio_comparison_l1243_124384


namespace NUMINAMATH_CALUDE_boat_speed_l1243_124300

/-- The speed of a boat in still water, given its downstream and upstream speeds -/
theorem boat_speed (downstream upstream : ℝ) (h1 : downstream = 11) (h2 : upstream = 5) :
  ∃ (still_speed stream_speed : ℝ),
    still_speed + stream_speed = downstream ∧
    still_speed - stream_speed = upstream ∧
    still_speed = 8 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_l1243_124300


namespace NUMINAMATH_CALUDE_profit_formula_l1243_124337

-- Define variables
variable (C S P p n : ℝ)

-- Define the conditions
def condition1 : Prop := P = p * ((C + S) / 2)
def condition2 : Prop := P = S / n - C

-- Theorem statement
theorem profit_formula 
  (h1 : condition1 C S P p)
  (h2 : condition2 C S P n)
  : P = (S * (2 * n * p + 2 * p - n)) / (n * (2 * p + n)) :=
by sorry

end NUMINAMATH_CALUDE_profit_formula_l1243_124337


namespace NUMINAMATH_CALUDE_remaining_numbers_l1243_124360

def three_digit_numbers : ℕ := 900

def numbers_with_two_identical_nonadjacent_digits : ℕ := 81

def numbers_with_three_distinct_digits : ℕ := 648

theorem remaining_numbers :
  three_digit_numbers - (numbers_with_two_identical_nonadjacent_digits + numbers_with_three_distinct_digits) = 171 := by
  sorry

end NUMINAMATH_CALUDE_remaining_numbers_l1243_124360


namespace NUMINAMATH_CALUDE_sets_satisfying_union_condition_l1243_124314

theorem sets_satisfying_union_condition :
  ∃! (S : Finset (Finset ℕ)), 
    (∀ A ∈ S, A ∪ {1, 2} = {1, 2, 3}) ∧ 
    (∀ A, A ∪ {1, 2} = {1, 2, 3} → A ∈ S) ∧
    Finset.card S = 4 := by
  sorry

end NUMINAMATH_CALUDE_sets_satisfying_union_condition_l1243_124314


namespace NUMINAMATH_CALUDE_wall_length_l1243_124313

/-- The length of a rectangular wall with a trapezoidal mirror -/
theorem wall_length (a b h w : ℝ) (ha : a > 0) (hb : b > 0) (hh : h > 0) (hw : w > 0) :
  (a + b) * h / 2 * 2 = w * (580 / 27) →
  a = 34 →
  b = 24 →
  h = 20 →
  w = 54 →
  580 / 27 = 580 / 27 :=
by sorry

end NUMINAMATH_CALUDE_wall_length_l1243_124313


namespace NUMINAMATH_CALUDE_sqrt_88200_simplification_l1243_124308

theorem sqrt_88200_simplification : Real.sqrt 88200 = 882 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_88200_simplification_l1243_124308


namespace NUMINAMATH_CALUDE_distinct_digit_sums_count_l1243_124363

/-- Calculate the digit sum of a natural number. -/
def digitSum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digitSum (n / 10)

/-- The set of all digit sums for numbers from 1 to 2021. -/
def digitSumSet : Finset ℕ :=
  Finset.image digitSum (Finset.range 2021)

/-- Theorem: The number of distinct digit sums for integers from 1 to 2021 is 28. -/
theorem distinct_digit_sums_count : digitSumSet.card = 28 := by
  sorry

end NUMINAMATH_CALUDE_distinct_digit_sums_count_l1243_124363


namespace NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l1243_124326

theorem sum_of_absolute_coefficients (a a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (1 - x)^5 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  |a| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| = 32 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l1243_124326


namespace NUMINAMATH_CALUDE_rectangle_y_value_l1243_124358

theorem rectangle_y_value (y : ℝ) (h1 : y > 0) : 
  let vertices := [(0, y), (10, y), (0, 4), (10, 4)]
  let area := 90
  let length := 10
  let height := y - 4
  (length * height = area) → y = 13 := by
sorry

end NUMINAMATH_CALUDE_rectangle_y_value_l1243_124358


namespace NUMINAMATH_CALUDE_bouncing_ball_distance_l1243_124396

/-- Calculates the total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (bounceRatio : ℝ) (numBounces : ℕ) : ℝ :=
  let rec bounceSum (height : ℝ) (n : ℕ) : ℝ :=
    if n = 0 then 0
    else height + height * bounceRatio + bounceSum (height * bounceRatio) (n - 1)
  initialHeight + 2 * bounceSum initialHeight numBounces

/-- The bouncing ball problem -/
theorem bouncing_ball_distance :
  ∃ (d : ℝ), abs (d - totalDistance 20 (2/3) 4) < 0.5 ∧ Int.floor d = 80 := by
  sorry


end NUMINAMATH_CALUDE_bouncing_ball_distance_l1243_124396


namespace NUMINAMATH_CALUDE_sum_in_base_6_l1243_124347

/-- Converts a base 6 number to base 10 --/
def to_base_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- Converts a base 10 number to base 6 --/
def to_base_6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
  aux n []

/-- The sum of 453₆, 436₆, and 42₆ in base 6 is 1415₆ --/
theorem sum_in_base_6 :
  to_base_6 (to_base_10 [3, 5, 4] + to_base_10 [6, 3, 4] + to_base_10 [2, 4]) = [5, 1, 4, 1] :=
sorry

end NUMINAMATH_CALUDE_sum_in_base_6_l1243_124347


namespace NUMINAMATH_CALUDE_range_of_a_l1243_124392

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, a * x^2 - a * x - 2 < 0) →
  a ∈ Set.Ioc (-8) 0 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1243_124392


namespace NUMINAMATH_CALUDE_unique_solution_l1243_124312

def satisfiesConditions (n : ℕ) : Prop :=
  ∃ k m p : ℕ, n = 2 * k + 1 ∧ n = 3 * m - 1 ∧ n = 5 * p + 2

theorem unique_solution : 
  satisfiesConditions 47 ∧ 
  (¬ satisfiesConditions 39) ∧ 
  (¬ satisfiesConditions 40) ∧ 
  (¬ satisfiesConditions 49) ∧ 
  (¬ satisfiesConditions 53) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l1243_124312


namespace NUMINAMATH_CALUDE_midpoint_octagon_area_l1243_124362

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : sorry

/-- The octagon formed by joining the midpoints of a regular octagon's sides -/
def midpoint_octagon (o : RegularOctagon) : RegularOctagon :=
  sorry

/-- The area of a regular octagon -/
def area (o : RegularOctagon) : ℝ :=
  sorry

theorem midpoint_octagon_area (o : RegularOctagon) :
  area (midpoint_octagon o) = (1/4) * area o := by
  sorry

end NUMINAMATH_CALUDE_midpoint_octagon_area_l1243_124362


namespace NUMINAMATH_CALUDE_equation_solution_l1243_124338

theorem equation_solution :
  let f : ℝ → ℝ := λ x => x + 30 / (x - 4)
  ∃ (x₁ x₂ : ℝ), (f x₁ = -8 ∧ f x₂ = -8) ∧ 
    x₁ = -2 + Real.sqrt 6 ∧ x₂ = -2 - Real.sqrt 6 ∧
    ∀ x : ℝ, f x = -8 → (x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1243_124338


namespace NUMINAMATH_CALUDE_floor_sqrt_80_l1243_124367

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_80_l1243_124367


namespace NUMINAMATH_CALUDE_min_cost_butter_l1243_124378

/-- The cost of a 16 oz package of butter -/
def cost_16oz : ℝ := 7

/-- The cost of an 8 oz package of butter -/
def cost_8oz : ℝ := 4

/-- The cost of a 4 oz package of butter before discount -/
def cost_4oz : ℝ := 2

/-- The discount rate applied to 4 oz packages -/
def discount_rate : ℝ := 0.5

/-- The total amount of butter needed in ounces -/
def butter_needed : ℝ := 16

/-- Theorem stating that the minimum cost of purchasing 16 oz of butter is $6.0 -/
theorem min_cost_butter : 
  min cost_16oz (cost_8oz + 2 * (cost_4oz * (1 - discount_rate))) = 6 := by sorry

end NUMINAMATH_CALUDE_min_cost_butter_l1243_124378


namespace NUMINAMATH_CALUDE_ratio_and_linear_equation_l1243_124315

theorem ratio_and_linear_equation (c d : ℝ) : 
  c / d = 4 → c = 20 - 6 * d → d = 2 := by sorry

end NUMINAMATH_CALUDE_ratio_and_linear_equation_l1243_124315


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l1243_124341

/-- A circle in the xy-plane -/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- The center of a circle -/
def Circle.center (c : Circle) : ℝ × ℝ := sorry

/-- The radius of a circle -/
def Circle.radius (c : Circle) : ℝ := sorry

/-- The given circle equation -/
def givenCircle : Circle :=
  { equation := fun x y => x^2 + y^2 - 2*x - 3 = 0 }

theorem circle_center_and_radius :
  Circle.center givenCircle = (1, 0) ∧ Circle.radius givenCircle = 2 := by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l1243_124341


namespace NUMINAMATH_CALUDE_max_profit_rate_l1243_124350

def f (x : ℕ) : ℚ :=
  if 1 ≤ x ∧ x ≤ 20 then 1
  else if 21 ≤ x ∧ x ≤ 60 then x / 10
  else 0

def g (x : ℕ) : ℚ :=
  if 1 ≤ x ∧ x ≤ 20 then 1 / (x + 80)
  else if 21 ≤ x ∧ x ≤ 60 then (2 * x) / (x^2 - x + 1600)
  else 0

theorem max_profit_rate :
  ∀ x : ℕ, 1 ≤ x ∧ x ≤ 60 → g x ≤ 2/79 ∧ g 40 = 2/79 :=
by sorry

end NUMINAMATH_CALUDE_max_profit_rate_l1243_124350


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1243_124340

theorem polynomial_factorization (m n : ℝ) : 
  (∀ x, x^2 + m*x + 6 = (x - 2)*(x + n)) → m = -5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1243_124340


namespace NUMINAMATH_CALUDE_equation_has_one_solution_l1243_124365

/-- The equation (3x^3 - 15x^2) / (x^2 - 5x) = 2x - 6 has exactly one solution -/
theorem equation_has_one_solution :
  ∃! x : ℝ, x ≠ 0 ∧ x ≠ 5 ∧ (3 * x^3 - 15 * x^2) / (x^2 - 5 * x) = 2 * x - 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_has_one_solution_l1243_124365


namespace NUMINAMATH_CALUDE_log_inequality_l1243_124369

theorem log_inequality (a b : ℝ) (ha : a = Real.log 0.3 / Real.log 0.2) (hb : b = Real.log 0.3 / Real.log 2) :
  a * b < a + b ∧ a + b < 0 := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l1243_124369


namespace NUMINAMATH_CALUDE_cos_seven_pi_six_plus_x_l1243_124389

theorem cos_seven_pi_six_plus_x (x : Real) (h : Real.sin (2 * Real.pi / 3 + x) = 3 / 5) :
  Real.cos (7 * Real.pi / 6 + x) = -3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_seven_pi_six_plus_x_l1243_124389


namespace NUMINAMATH_CALUDE_a_12_upper_bound_a_12_no_lower_bound_l1243_124364

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

/-- The upper bound of a_12 in an arithmetic sequence satisfying given conditions -/
theorem a_12_upper_bound
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a8 : a 8 ≥ 15)
  (h_a9 : a 9 ≤ 13) :
  a 12 ≤ 7 :=
sorry

/-- The non-existence of a lower bound for a_12 in an arithmetic sequence satisfying given conditions -/
theorem a_12_no_lower_bound
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a8 : a 8 ≥ 15)
  (h_a9 : a 9 ≤ 13) :
  ∀ x : ℝ, ∃ y : ℝ, y < x ∧ ∃ (a' : ℕ → ℝ), arithmetic_sequence a' ∧ a' 8 ≥ 15 ∧ a' 9 ≤ 13 ∧ a' 12 = y :=
sorry

end NUMINAMATH_CALUDE_a_12_upper_bound_a_12_no_lower_bound_l1243_124364


namespace NUMINAMATH_CALUDE_snow_leopard_arrangement_l1243_124317

theorem snow_leopard_arrangement (n : ℕ) (h : n = 9) : 
  2 * Nat.factorial (n - 2) = 10080 := by
  sorry

end NUMINAMATH_CALUDE_snow_leopard_arrangement_l1243_124317


namespace NUMINAMATH_CALUDE_points_on_line_implies_b_value_l1243_124397

/-- Three points lie on the same line if and only if the slope between any two pairs of points is equal. -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

/-- The theorem states that if the given points lie on the same line, then b = -1/2. -/
theorem points_on_line_implies_b_value (b : ℝ) :
  collinear 6 (-10) (-b + 4) 3 (3*b + 6) 3 → b = -1/2 := by
  sorry

#check points_on_line_implies_b_value

end NUMINAMATH_CALUDE_points_on_line_implies_b_value_l1243_124397


namespace NUMINAMATH_CALUDE_fibonacci_primitive_roots_l1243_124376

theorem fibonacci_primitive_roots (p : Nat) (g : Nat) (k : Nat) 
    (h1 : Nat.Prime p)
    (h2 : IsPrimitiveRoot g p)
    (h3 : g^2 % p = (g + 1) % p)
    (h4 : p = 4*k + 3) :
  IsPrimitiveRoot (g - 1) p ∧ 
  (g - 1)^(2*k + 3) % p = (g - 2) % p ∧
  IsPrimitiveRoot (g - 2) p :=
by sorry

end NUMINAMATH_CALUDE_fibonacci_primitive_roots_l1243_124376


namespace NUMINAMATH_CALUDE_line_through_origin_and_intersection_l1243_124385

/-- The equation of the line passing through the origin and the intersection point of two given lines -/
theorem line_through_origin_and_intersection (x y : ℝ) : 
  (x - 3*y + 4 = 0 ∧ 2*x + y + 5 = 0) → 
  (3*x + 19*y = 0) := by sorry

end NUMINAMATH_CALUDE_line_through_origin_and_intersection_l1243_124385


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l1243_124321

theorem parallelogram_base_length
  (area : ℝ) (base : ℝ) (altitude : ℝ)
  (h1 : area = 288)
  (h2 : altitude = 2 * base)
  (h3 : area = base * altitude) :
  base = 12 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l1243_124321


namespace NUMINAMATH_CALUDE_trig_inequality_l1243_124331

open Real

theorem trig_inequality (a b c d : ℝ) : 
  a = sin (sin (2009 * π / 180)) →
  b = sin (cos (2009 * π / 180)) →
  c = cos (sin (2009 * π / 180)) →
  d = cos (cos (2009 * π / 180)) →
  b < a ∧ a < d ∧ d < c := by sorry

end NUMINAMATH_CALUDE_trig_inequality_l1243_124331


namespace NUMINAMATH_CALUDE_optimal_selling_price_l1243_124311

/-- Represents the problem of finding the optimal selling price for a product --/
structure PricingProblem where
  costPrice : ℝ        -- Cost price per kilogram
  initialPrice : ℝ     -- Initial selling price per kilogram
  initialSales : ℝ     -- Initial monthly sales in kilograms
  salesDecrease : ℝ    -- Decrease in sales per 1 yuan price increase
  availableCapital : ℝ -- Available capital
  targetProfit : ℝ     -- Target profit

/-- Calculates the profit for a given selling price --/
def calculateProfit (p : PricingProblem) (sellingPrice : ℝ) : ℝ :=
  let salesVolume := p.initialSales - (sellingPrice - p.initialPrice) * p.salesDecrease
  (sellingPrice - p.costPrice) * salesVolume

/-- Checks if the capital required for a given selling price is within the available capital --/
def isCapitalSufficient (p : PricingProblem) (sellingPrice : ℝ) : Prop :=
  let salesVolume := p.initialSales - (sellingPrice - p.initialPrice) * p.salesDecrease
  p.costPrice * salesVolume ≤ p.availableCapital

/-- Theorem stating that the optimal selling price is 80 yuan --/
theorem optimal_selling_price (p : PricingProblem) 
  (h1 : p.costPrice = 40)
  (h2 : p.initialPrice = 50)
  (h3 : p.initialSales = 500)
  (h4 : p.salesDecrease = 10)
  (h5 : p.availableCapital = 10000)
  (h6 : p.targetProfit = 8000) :
  ∃ (x : ℝ), x = 80 ∧ 
    calculateProfit p x = p.targetProfit ∧ 
    isCapitalSufficient p x ∧
    ∀ (y : ℝ), y ≠ x → calculateProfit p y = p.targetProfit → ¬(isCapitalSufficient p y) := by
  sorry


end NUMINAMATH_CALUDE_optimal_selling_price_l1243_124311


namespace NUMINAMATH_CALUDE_quadratic_two_members_l1243_124391

-- Define the set A
def A (m : ℝ) : Set ℝ := {x | m * x^2 + 2 * x + 1 = 0}

-- Define the property that A has only two members
def has_two_members (S : Set ℝ) : Prop := ∃ (a b : ℝ), a ≠ b ∧ S = {a, b}

-- Theorem statement
theorem quadratic_two_members :
  ∀ m : ℝ, has_two_members (A m) ↔ (m = 0 ∨ m = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_members_l1243_124391


namespace NUMINAMATH_CALUDE_number_calculation_l1243_124335

theorem number_calculation (n x : ℝ) (h1 : x = 0.8999999999999999) (h2 : n / x = 0.01) :
  n = 0.008999999999999999 := by
sorry

end NUMINAMATH_CALUDE_number_calculation_l1243_124335


namespace NUMINAMATH_CALUDE_least_number_with_remainders_l1243_124334

theorem least_number_with_remainders : ∃! n : ℕ,
  n > 0 ∧
  n % 7 = 5 ∧
  n % 11 = 5 ∧
  n % 13 = 5 ∧
  n % 17 = 5 ∧
  n % 23 = 5 ∧
  n % 19 = 0 ∧
  ∀ m : ℕ, m > 0 →
    m % 7 = 5 →
    m % 11 = 5 →
    m % 13 = 5 →
    m % 17 = 5 →
    m % 23 = 5 →
    m % 19 = 0 →
    n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_number_with_remainders_l1243_124334


namespace NUMINAMATH_CALUDE_mia_speed_theorem_l1243_124348

def eugene_speed : ℚ := 5
def carlos_ratio : ℚ := 3/4
def mia_ratio : ℚ := 4/3

theorem mia_speed_theorem : 
  mia_ratio * (carlos_ratio * eugene_speed) = eugene_speed := by
  sorry

end NUMINAMATH_CALUDE_mia_speed_theorem_l1243_124348


namespace NUMINAMATH_CALUDE_granger_age_difference_l1243_124387

theorem granger_age_difference : 
  let granger_age : ℕ := 42
  let son_age : ℕ := 16
  granger_age - 2 * son_age = 10 := by sorry

end NUMINAMATH_CALUDE_granger_age_difference_l1243_124387


namespace NUMINAMATH_CALUDE_fireworks_display_total_l1243_124374

/-- Calculate the total number of fireworks in a New Year's Eve display --/
def totalFireworks : ℕ :=
  let yearDigits : ℕ := 4
  let yearFireworksPerDigit : ℕ := 6
  let happyNewYearLetters : ℕ := 12
  let regularLetterFireworks : ℕ := 5
  let helloFireworks : ℕ := 8 + 7 + 6 + 6 + 9
  let additionalBoxes : ℕ := 100
  let fireworksPerBox : ℕ := 10

  yearDigits * yearFireworksPerDigit +
  happyNewYearLetters * regularLetterFireworks +
  helloFireworks +
  additionalBoxes * fireworksPerBox

theorem fireworks_display_total :
  totalFireworks = 1120 := by sorry

end NUMINAMATH_CALUDE_fireworks_display_total_l1243_124374


namespace NUMINAMATH_CALUDE_weight_sequence_l1243_124322

theorem weight_sequence (a : ℕ → ℝ) : 
  (∀ n, a n < a (n + 1)) →  -- weights are in increasing order
  (∀ k, k ≤ 29 → a k + a (k + 3) = a (k + 1) + a (k + 2)) →  -- balancing condition
  a 3 = 9 →  -- third weight is 9 grams
  a 9 = 33 →  -- ninth weight is 33 grams
  a 33 = 257 :=  -- 33rd weight is 257 grams
by
  sorry


end NUMINAMATH_CALUDE_weight_sequence_l1243_124322


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l1243_124323

theorem quadratic_inequality_condition (a : ℝ) : 
  (∀ x, a * x^2 + 2 * a * x + 1 > 0) →
  (0 ≤ a ∧ a < 2) ∧
  ¬(0 ≤ a ∧ a < 2 → ∀ x, a * x^2 + 2 * a * x + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l1243_124323


namespace NUMINAMATH_CALUDE_max_abs_z_plus_i_l1243_124309

theorem max_abs_z_plus_i :
  ∀ (x y : ℝ), 
    x^2/4 + y^2 = 1 →
    ∀ (z : ℂ), 
      z = x + y * Complex.I →
      ∀ (w : ℂ), 
        Complex.abs w = Complex.abs (z + Complex.I) →
        Complex.abs w ≤ 4 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_max_abs_z_plus_i_l1243_124309


namespace NUMINAMATH_CALUDE_prob_even_sum_coins_and_dice_l1243_124399

/-- Represents the outcome of tossing a fair coin -/
inductive CoinOutcome
| Heads
| Tails

/-- Represents the outcome of rolling a fair die -/
def DieOutcome := Fin 6

/-- The probability of getting heads on a fair coin toss -/
def probHeads : ℚ := 1/2

/-- The probability of getting an even number on a fair die roll -/
def probEvenDie : ℚ := 1/2

/-- The number of coins tossed -/
def numCoins : ℕ := 3

/-- Calculates the probability of getting k heads in n coin tosses -/
def probKHeads (n k : ℕ) : ℚ := sorry

/-- Calculates the probability of getting an even sum when rolling k fair dice -/
def probEvenSumKDice (k : ℕ) : ℚ := sorry

theorem prob_even_sum_coins_and_dice :
  (probKHeads numCoins 0 * 1 +
   probKHeads numCoins 1 * probEvenDie +
   probKHeads numCoins 2 * probEvenSumKDice 2 +
   probKHeads numCoins 3 * probEvenSumKDice 3) = 15/16 := by sorry

end NUMINAMATH_CALUDE_prob_even_sum_coins_and_dice_l1243_124399


namespace NUMINAMATH_CALUDE_book_problem_solution_l1243_124306

def book_problem (total_cost selling_price_1 cost_1 loss_percent : ℚ) : Prop :=
  let cost_2 := total_cost - cost_1
  let selling_price_2 := selling_price_1
  let gain_percent := (selling_price_2 - cost_2) / cost_2 * 100
  (total_cost = 540) ∧
  (cost_1 = 315) ∧
  (selling_price_1 = cost_1 * (1 - loss_percent / 100)) ∧
  (loss_percent = 15) ∧
  (gain_percent = 19)

theorem book_problem_solution :
  ∃ (total_cost selling_price_1 cost_1 loss_percent : ℚ),
    book_problem total_cost selling_price_1 cost_1 loss_percent :=
sorry

end NUMINAMATH_CALUDE_book_problem_solution_l1243_124306
