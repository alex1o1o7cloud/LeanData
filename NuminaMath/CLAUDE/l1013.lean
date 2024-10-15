import Mathlib

namespace NUMINAMATH_CALUDE_second_number_is_40_l1013_101323

theorem second_number_is_40 (a b c : ℕ+) 
  (sum_eq : a + b + c = 120)
  (ratio_ab : (a : ℚ) / (b : ℚ) = 3 / 4)
  (ratio_bc : (b : ℚ) / (c : ℚ) = 7 / 9) :
  b = 40 := by
  sorry

end NUMINAMATH_CALUDE_second_number_is_40_l1013_101323


namespace NUMINAMATH_CALUDE_sign_up_ways_for_six_students_three_projects_l1013_101388

/-- The number of ways students can sign up for projects -/
def signUpWays (numStudents : ℕ) (numProjects : ℕ) : ℕ :=
  numProjects ^ numStudents

/-- Theorem: For 6 students and 3 projects, the number of ways to sign up is 3^6 -/
theorem sign_up_ways_for_six_students_three_projects :
  signUpWays 6 3 = 3^6 := by
  sorry

end NUMINAMATH_CALUDE_sign_up_ways_for_six_students_three_projects_l1013_101388


namespace NUMINAMATH_CALUDE_pizza_ingredients_calculation_l1013_101369

/-- Pizza ingredients calculation -/
theorem pizza_ingredients_calculation 
  (water : ℕ) 
  (flour : ℕ) 
  (salt : ℚ) 
  (h1 : water = 10)
  (h2 : flour = 16)
  (h3 : salt = (1/2 : ℚ) * flour) :
  (water + flour : ℕ) = 26 ∧ salt = 8 := by
  sorry

end NUMINAMATH_CALUDE_pizza_ingredients_calculation_l1013_101369


namespace NUMINAMATH_CALUDE_quadratic_fixed_point_l1013_101332

-- Define the quadratic function
def f (p q : ℝ) (x : ℝ) : ℝ := x^2 + p*x + q

-- Define the theorem
theorem quadratic_fixed_point 
  (p q : ℝ) 
  (h1 : ∀ x ∈ Set.Icc 3 5, |f p q x| ≤ 1/2)
  (h2 : f p q ((7 + Real.sqrt 15) / 2) = 0) :
  (f p q)^[2017] ((7 + Real.sqrt 15) / 2) = (7 - Real.sqrt 15) / 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_fixed_point_l1013_101332


namespace NUMINAMATH_CALUDE_son_age_proof_l1013_101328

theorem son_age_proof (son_age father_age : ℕ) : 
  father_age = son_age + 20 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 18 := by
sorry

end NUMINAMATH_CALUDE_son_age_proof_l1013_101328


namespace NUMINAMATH_CALUDE_class_size_correct_l1013_101316

/-- The number of students in class A -/
def class_size : ℕ := 30

/-- The number of students who like social studies -/
def social_studies_fans : ℕ := 25

/-- The number of students who like music -/
def music_fans : ℕ := 32

/-- The number of students who like both social studies and music -/
def both_fans : ℕ := 27

/-- Theorem stating that the class size is correct given the conditions -/
theorem class_size_correct :
  class_size = social_studies_fans + music_fans - both_fans ∧
  class_size = social_studies_fans + music_fans - both_fans :=
by sorry

end NUMINAMATH_CALUDE_class_size_correct_l1013_101316


namespace NUMINAMATH_CALUDE_ceiling_minus_x_bounds_l1013_101313

theorem ceiling_minus_x_bounds (x : ℝ) : 
  ⌈x⌉ - ⌊x⌋ = 1 → 0 < ⌈x⌉ - x ∧ ⌈x⌉ - x ≤ 1 := by sorry

end NUMINAMATH_CALUDE_ceiling_minus_x_bounds_l1013_101313


namespace NUMINAMATH_CALUDE_swimmers_pass_count_l1013_101307

/-- Represents a swimmer in the pool --/
structure Swimmer where
  speed : ℝ
  turnDelay : ℝ

/-- Calculates the number of times swimmers pass each other --/
def countPasses (poolLength : ℝ) (swimmer1 : Swimmer) (swimmer2 : Swimmer) (totalTime : ℝ) : ℕ :=
  sorry

/-- Theorem stating the number of passes for the given problem --/
theorem swimmers_pass_count :
  let poolLength : ℝ := 120
  let swimmer1 : Swimmer := ⟨4, 2⟩
  let swimmer2 : Swimmer := ⟨3, 0⟩
  let totalTime : ℝ := 900
  countPasses poolLength swimmer1 swimmer2 totalTime = 26 :=
by sorry

end NUMINAMATH_CALUDE_swimmers_pass_count_l1013_101307


namespace NUMINAMATH_CALUDE_point_distance_product_l1013_101327

theorem point_distance_product (y₁ y₂ : ℝ) : 
  ((-4 - 3)^2 + (y₁ - (-1))^2 = 13^2) →
  ((-4 - 3)^2 + (y₂ - (-1))^2 = 13^2) →
  y₁ ≠ y₂ →
  y₁ * y₂ = -119 := by
sorry

end NUMINAMATH_CALUDE_point_distance_product_l1013_101327


namespace NUMINAMATH_CALUDE_problem_statement_l1013_101380

theorem problem_statement (a b : ℝ) :
  (∀ x y : ℝ, (2 * x^2 + a * x - y + 6) - (b * x^2 - 3 * x + 5 * y - 1) = -6 * y + 7) →
  a^2 + b^2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1013_101380


namespace NUMINAMATH_CALUDE_dice_surface_sum_in_possible_sums_l1013_101349

/-- The number of dice in the arrangement -/
def num_dice : ℕ := 2012

/-- The sum of points on all faces of a standard six-sided die -/
def die_sum : ℕ := 21

/-- The sum of points on opposite faces of a die -/
def opposite_faces_sum : ℕ := 7

/-- The set of possible sums of points on the surface -/
def possible_sums : Set ℕ := {28177, 28179, 28181, 28183, 28185, 28187}

/-- Theorem: The sum of points on the surface of the arranged dice is in the set of possible sums -/
theorem dice_surface_sum_in_possible_sums :
  ∃ (x : ℕ), x ∈ possible_sums ∧
  ∃ (end_face_sum : ℕ), end_face_sum ≥ 1 ∧ end_face_sum ≤ 6 ∧
  x = num_dice * die_sum - (num_dice - 1) * opposite_faces_sum + 2 * end_face_sum :=
by
  sorry

end NUMINAMATH_CALUDE_dice_surface_sum_in_possible_sums_l1013_101349


namespace NUMINAMATH_CALUDE_sin_sum_of_complex_exponentials_l1013_101310

theorem sin_sum_of_complex_exponentials (θ φ : ℝ) :
  Complex.exp (θ * Complex.I) = 4/5 + 3/5 * Complex.I →
  Complex.exp (φ * Complex.I) = -5/13 + 12/13 * Complex.I →
  Real.sin (θ + φ) = 84/65 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_of_complex_exponentials_l1013_101310


namespace NUMINAMATH_CALUDE_student_group_equations_l1013_101305

/-- Given a number of students and groups, prove that the system of equations
    represents the given conditions. -/
theorem student_group_equations (x y : ℕ) : 
  (5 * y = x - 3 ∧ 6 * y = x + 3) ↔ 
  (x = 5 * y + 3 ∧ x = 6 * y - 3) := by
sorry

end NUMINAMATH_CALUDE_student_group_equations_l1013_101305


namespace NUMINAMATH_CALUDE_closest_fraction_is_one_fourth_l1013_101317

def total_medals : ℕ := 150
def won_medals : ℕ := 38

theorem closest_fraction_is_one_fourth :
  let fraction := won_medals / total_medals
  ∀ x ∈ ({1/3, 1/5, 1/6, 1/7} : Set ℚ),
    |fraction - (1/4 : ℚ)| ≤ |fraction - x| :=
by
  sorry

end NUMINAMATH_CALUDE_closest_fraction_is_one_fourth_l1013_101317


namespace NUMINAMATH_CALUDE_john_painting_time_l1013_101308

theorem john_painting_time (sally_time john_time combined_time : ℝ) : 
  sally_time = 4 →
  combined_time = 2.4 →
  1 / sally_time + 1 / john_time = 1 / combined_time →
  john_time = 6 := by
sorry

end NUMINAMATH_CALUDE_john_painting_time_l1013_101308


namespace NUMINAMATH_CALUDE_base7_to_base10_equality_l1013_101338

/-- Conversion from base 7 to base 10 -/
def base7to10 (n : ℕ) : ℕ := 
  7 * 7 * (n / 100) + 7 * ((n / 10) % 10) + (n % 10)

theorem base7_to_base10_equality (c d e : ℕ) : 
  (c < 10 ∧ d < 10 ∧ e < 10) → 
  (base7to10 761 = 100 * c + 10 * d + e) → 
  (d * e : ℚ) / 15 = 48 / 15 := by
sorry

end NUMINAMATH_CALUDE_base7_to_base10_equality_l1013_101338


namespace NUMINAMATH_CALUDE_paths_H_to_J_through_I_l1013_101379

/-- The number of paths from H to I -/
def paths_H_to_I : ℕ := Nat.choose 6 1

/-- The number of paths from I to J -/
def paths_I_to_J : ℕ := Nat.choose 5 2

/-- The total number of steps from H to J -/
def total_steps : ℕ := 11

/-- Theorem stating the number of paths from H to J passing through I -/
theorem paths_H_to_J_through_I : paths_H_to_I * paths_I_to_J = 60 :=
sorry

end NUMINAMATH_CALUDE_paths_H_to_J_through_I_l1013_101379


namespace NUMINAMATH_CALUDE_eighth_term_is_21_l1013_101345

/-- A Fibonacci-like sequence where each number after the second is the sum of the two preceding numbers -/
def fibonacci_like_sequence (a₁ a₂ : ℕ) : ℕ → ℕ
| 0 => a₁
| 1 => a₂
| (n + 2) => fibonacci_like_sequence a₁ a₂ n + fibonacci_like_sequence a₁ a₂ (n + 1)

/-- The theorem stating that the 8th term of the specific Fibonacci-like sequence is 21 -/
theorem eighth_term_is_21 :
  ∃ (seq : ℕ → ℕ), 
    seq = fibonacci_like_sequence 1 1 ∧
    seq 7 = 21 ∧
    seq 8 = 34 ∧
    seq 9 = 55 :=
by
  sorry

end NUMINAMATH_CALUDE_eighth_term_is_21_l1013_101345


namespace NUMINAMATH_CALUDE_debt_ratio_proof_l1013_101312

/-- Proves that the ratio of Aryan's debt to Kyro's debt is 2:1 given the problem conditions --/
theorem debt_ratio_proof (aryan_debt kyro_debt : ℝ) 
  (h1 : aryan_debt = 1200)
  (h2 : 0.6 * aryan_debt + 0.8 * kyro_debt + 300 = 1500) :
  aryan_debt / kyro_debt = 2 := by
  sorry


end NUMINAMATH_CALUDE_debt_ratio_proof_l1013_101312


namespace NUMINAMATH_CALUDE_water_displacement_squared_l1013_101374

def cube_side_length : ℝ := 12
def tank_radius : ℝ := 6
def tank_height : ℝ := 15

theorem water_displacement_squared :
  let cube_volume := cube_side_length ^ 3
  let cube_diagonal := cube_side_length * Real.sqrt 3
  cube_diagonal ≤ tank_height →
  (cube_volume ^ 2 : ℝ) = 2985984 := by sorry

end NUMINAMATH_CALUDE_water_displacement_squared_l1013_101374


namespace NUMINAMATH_CALUDE_linear_function_not_in_fourth_quadrant_l1013_101350

/-- A linear function with slope 1 and y-intercept 1 -/
def f (x : ℝ) : ℝ := x + 1

/-- The fourth quadrant of the Cartesian plane -/
def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

theorem linear_function_not_in_fourth_quadrant :
  ∀ x : ℝ, ¬(fourth_quadrant x (f x)) :=
sorry

end NUMINAMATH_CALUDE_linear_function_not_in_fourth_quadrant_l1013_101350


namespace NUMINAMATH_CALUDE_polynomial_degree_is_8_l1013_101348

def polynomial_degree (x : ℝ) : ℕ :=
  let expr1 := x^7
  let expr2 := x + 1/x
  let expr3 := 1 + 3/x + 5/(x^2)
  let result := expr1 * expr2 * expr3
  8  -- The degree of the resulting polynomial

theorem polynomial_degree_is_8 : 
  ∀ x : ℝ, x ≠ 0 → polynomial_degree x = 8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_degree_is_8_l1013_101348


namespace NUMINAMATH_CALUDE_sales_tax_percentage_l1013_101346

-- Define the prices and quantities
def tshirt_price : ℚ := 8
def sweater_price : ℚ := 18
def jacket_price : ℚ := 80
def jacket_discount : ℚ := 0.1
def tshirt_quantity : ℕ := 6
def sweater_quantity : ℕ := 4
def jacket_quantity : ℕ := 5

-- Define the total cost before tax
def total_cost_before_tax : ℚ :=
  tshirt_price * tshirt_quantity +
  sweater_price * sweater_quantity +
  jacket_price * jacket_quantity * (1 - jacket_discount)

-- Define the total cost including tax
def total_cost_with_tax : ℚ := 504

-- Theorem: The sales tax percentage is 5%
theorem sales_tax_percentage :
  ∃ (tax_rate : ℚ), 
    tax_rate = 0.05 ∧
    total_cost_with_tax = total_cost_before_tax * (1 + tax_rate) :=
sorry

end NUMINAMATH_CALUDE_sales_tax_percentage_l1013_101346


namespace NUMINAMATH_CALUDE_manuscript_cost_theorem_l1013_101355

/-- Calculates the total cost of typing and revising a manuscript --/
def manuscript_cost (total_pages : ℕ) (once_revised : ℕ) (twice_revised : ℕ) 
  (initial_rate : ℕ) (revision_rate : ℕ) : ℕ :=
  let not_revised := total_pages - once_revised - twice_revised
  let initial_cost := total_pages * initial_rate
  let once_revised_cost := once_revised * revision_rate
  let twice_revised_cost := twice_revised * (2 * revision_rate)
  initial_cost + once_revised_cost + twice_revised_cost

/-- Theorem stating the total cost of the manuscript --/
theorem manuscript_cost_theorem :
  manuscript_cost 200 80 20 5 3 = 1360 := by
  sorry

end NUMINAMATH_CALUDE_manuscript_cost_theorem_l1013_101355


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1013_101302

/-- A rectangle with integer dimensions satisfying the given condition has a perimeter of 26. -/
theorem rectangle_perimeter (a b : ℕ) : 
  a ≠ b →  -- not a square
  4 * (a + b) - a * b = 12 →  -- twice perimeter minus area equals 12
  2 * (a + b) = 26 := by  -- perimeter equals 26
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1013_101302


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1013_101335

-- Define the inequality
def inequality (x : ℝ) : Prop := (x - 2) / (x + 1) ≤ 0

-- Define the solution set
def solution_set : Set ℝ := {x | x > -1 ∧ x ≤ 2}

-- Theorem statement
theorem inequality_solution_set :
  {x : ℝ | inequality x ∧ x + 1 ≠ 0} = solution_set := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1013_101335


namespace NUMINAMATH_CALUDE_watermelon_seeds_l1013_101357

theorem watermelon_seeds (total_slices : ℕ) (black_seeds_per_slice : ℕ) (total_seeds : ℕ) :
  total_slices = 40 →
  black_seeds_per_slice = 20 →
  total_seeds = 1600 →
  (total_seeds - total_slices * black_seeds_per_slice) / total_slices = 20 :=
by sorry

end NUMINAMATH_CALUDE_watermelon_seeds_l1013_101357


namespace NUMINAMATH_CALUDE_prob_product_multiple_of_four_l1013_101396

/-- A fair 10-sided die -/
def decagonal_die := Finset.range 10

/-- A fair 12-sided die -/
def dodecagonal_die := Finset.range 12

/-- The probability of an event occurring when rolling a fair n-sided die -/
def prob (event : Finset ℕ) (die : Finset ℕ) : ℚ :=
  (event ∩ die).card / die.card

/-- The event of rolling a multiple of 4 -/
def multiple_of_four (die : Finset ℕ) : Finset ℕ :=
  die.filter (fun x => x % 4 = 0)

/-- The probability that the product of rolls from a 10-sided die and a 12-sided die is a multiple of 4 -/
theorem prob_product_multiple_of_four :
  prob (multiple_of_four decagonal_die) decagonal_die +
  prob (multiple_of_four dodecagonal_die) dodecagonal_die -
  prob (multiple_of_four decagonal_die) decagonal_die *
  prob (multiple_of_four dodecagonal_die) dodecagonal_die = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_prob_product_multiple_of_four_l1013_101396


namespace NUMINAMATH_CALUDE_characterize_function_l1013_101372

theorem characterize_function (n : ℕ) (hn : n ≥ 1) (hodd : Odd n) :
  ∃ (ε : Int) (d : ℕ) (c : Int),
    ε = 1 ∨ ε = -1 ∧
    d > 0 ∧
    d ∣ n ∧
    ∀ (f : ℤ → ℤ),
      (∀ (x y : ℤ), (f x - f y) ∣ (x^n - y^n)) →
      ∃ (ε' : Int) (d' : ℕ) (c' : Int),
        (ε' = 1 ∨ ε' = -1) ∧
        d' > 0 ∧
        d' ∣ n ∧
        ∀ (x : ℤ), f x = ε' * x^d' + c' :=
by sorry

end NUMINAMATH_CALUDE_characterize_function_l1013_101372


namespace NUMINAMATH_CALUDE_walter_exceptional_days_l1013_101370

/-- Represents Walter's chore earnings over a period of days -/
structure ChoreEarnings where
  regularPay : ℕ
  exceptionalPay : ℕ
  bonusThreshold : ℕ
  bonusAmount : ℕ
  totalDays : ℕ
  totalEarnings : ℕ

/-- Calculates the number of exceptional days given ChoreEarnings -/
def exceptionalDays (ce : ChoreEarnings) : ℕ :=
  sorry

/-- Theorem stating that Walter did chores exceptionally well for 5 days -/
theorem walter_exceptional_days (ce : ChoreEarnings) 
  (h1 : ce.regularPay = 4)
  (h2 : ce.exceptionalPay = 6)
  (h3 : ce.bonusThreshold = 5)
  (h4 : ce.bonusAmount = 10)
  (h5 : ce.totalDays = 12)
  (h6 : ce.totalEarnings = 58) :
  exceptionalDays ce = 5 :=
sorry

end NUMINAMATH_CALUDE_walter_exceptional_days_l1013_101370


namespace NUMINAMATH_CALUDE_constant_t_equality_l1013_101325

theorem constant_t_equality (x : ℝ) : 
  (5*x^2 - 6*x + 7) * (4*x^2 + (-6)*x + 10) = 20*x^4 - 54*x^3 + 114*x^2 - 102*x + 70 := by
  sorry


end NUMINAMATH_CALUDE_constant_t_equality_l1013_101325


namespace NUMINAMATH_CALUDE_local_max_derivative_condition_l1013_101389

/-- Given a function f with derivative f'(x) = a(x+1)(x-a), 
    if f attains a local maximum at x = a, then a is in the open interval (-1, 0) -/
theorem local_max_derivative_condition (f : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x, deriv f x = a * (x + 1) * (x - a))
  (h2 : IsLocalMax f a) :
  a ∈ Set.Ioo (-1 : ℝ) 0 := by
  sorry

end NUMINAMATH_CALUDE_local_max_derivative_condition_l1013_101389


namespace NUMINAMATH_CALUDE_probability_blue_between_red_and_triple_red_l1013_101330

-- Define the probability space
def Ω : Type := ℝ × ℝ

-- Define the probability measure
noncomputable def P : Set Ω → ℝ := sorry

-- Define the event where the blue point is greater than the red point but less than three times the red point
def E : Set Ω := {ω : Ω | let (x, y) := ω; 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ x < y ∧ y < 3*x}

-- State the theorem
theorem probability_blue_between_red_and_triple_red : P E = 5/6 := sorry

end NUMINAMATH_CALUDE_probability_blue_between_red_and_triple_red_l1013_101330


namespace NUMINAMATH_CALUDE_exam_maximum_marks_l1013_101368

/-- Given an exam where:
  1. The passing mark is 80% of the maximum marks.
  2. A student got 200 marks.
  3. The student failed by 200 marks (i.e., needs 200 more marks to pass).
  Prove that the maximum marks for the exam is 500. -/
theorem exam_maximum_marks :
  ∀ (max_marks : ℕ),
  (max_marks : ℚ) * (80 : ℚ) / (100 : ℚ) = (200 : ℚ) + (200 : ℚ) →
  max_marks = 500 := by
sorry

end NUMINAMATH_CALUDE_exam_maximum_marks_l1013_101368


namespace NUMINAMATH_CALUDE_oranges_thrown_away_l1013_101343

theorem oranges_thrown_away (initial : ℕ) (added : ℕ) (final : ℕ) :
  initial = 40 →
  added = 7 →
  final = 10 →
  initial - (initial - final + added) = 37 := by
sorry

end NUMINAMATH_CALUDE_oranges_thrown_away_l1013_101343


namespace NUMINAMATH_CALUDE_largest_even_digit_multiple_of_5_l1013_101333

/-- A function that checks if all digits of a natural number are even -/
def allDigitsEven (n : ℕ) : Prop := sorry

/-- A function that returns the largest positive integer less than 10000 
    with all even digits that is a multiple of 5 -/
noncomputable def largestEvenDigitMultipleOf5 : ℕ := sorry

/-- Theorem stating that 8860 is the largest positive integer less than 10000 
    with all even digits that is a multiple of 5 -/
theorem largest_even_digit_multiple_of_5 : 
  largestEvenDigitMultipleOf5 = 8860 ∧ 
  allDigitsEven 8860 ∧ 
  8860 < 10000 ∧ 
  8860 % 5 = 0 :=
by sorry

end NUMINAMATH_CALUDE_largest_even_digit_multiple_of_5_l1013_101333


namespace NUMINAMATH_CALUDE_message_pairs_l1013_101322

theorem message_pairs (n m : ℕ) (hn : n = 100) (hm : m = 50) :
  let total_messages := n * m
  let max_unique_pairs := n * (n - 1) / 2
  total_messages - max_unique_pairs = 50 :=
by sorry

end NUMINAMATH_CALUDE_message_pairs_l1013_101322


namespace NUMINAMATH_CALUDE_range_of_2a_plus_3b_l1013_101387

theorem range_of_2a_plus_3b (a b : ℝ) 
  (h1 : -1 < a + b ∧ a + b < 3) 
  (h2 : 2 < a - b ∧ a - b < 4) : 
  -9/2 < 2*a + 3*b ∧ 2*a + 3*b < 13/2 := by
sorry

end NUMINAMATH_CALUDE_range_of_2a_plus_3b_l1013_101387


namespace NUMINAMATH_CALUDE_sum_22_probability_l1013_101367

/-- Represents a 20-faced die with some numbered faces and some blank faces -/
structure Die where
  numbered_faces : Finset ℕ
  blank_faces : ℕ
  total_faces : numbered_faces.card + blank_faces = 20

/-- The first die with faces 1 through 18 and two blank faces -/
def die1 : Die where
  numbered_faces := Finset.range 18
  blank_faces := 2
  total_faces := sorry

/-- The second die with faces 2 through 9 and 11 through 20 and two blank faces -/
def die2 : Die where
  numbered_faces := (Finset.range 8).image (λ x => x + 2) ∪ (Finset.range 10).image (λ x => x + 11)
  blank_faces := 2
  total_faces := sorry

/-- The probability of an event given the number of favorable outcomes and total outcomes -/
def probability (favorable : ℕ) (total : ℕ) : ℚ := favorable / total

/-- The theorem to be proved -/
theorem sum_22_probability :
  probability (die1.numbered_faces.card * die2.numbered_faces.card) (20 * 20) = 1 / 40 := by
  sorry

end NUMINAMATH_CALUDE_sum_22_probability_l1013_101367


namespace NUMINAMATH_CALUDE_triangle_exists_from_altitudes_l1013_101304

theorem triangle_exists_from_altitudes (h₁ h₂ h₃ : ℝ) 
  (h₁_pos : h₁ > 0) (h₂_pos : h₂ > 0) (h₃_pos : h₃ > 0) :
  ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a + b > c) ∧ (b + c > a) ∧ (c + a > b) ∧
    h₁ = (2 * (a * b * c) / (a * (a + b + c))) ∧
    h₂ = (2 * (a * b * c) / (b * (a + b + c))) ∧
    h₃ = (2 * (a * b * c) / (c * (a + b + c))) :=
by sorry

end NUMINAMATH_CALUDE_triangle_exists_from_altitudes_l1013_101304


namespace NUMINAMATH_CALUDE_area_ratio_circle_ellipse_l1013_101300

/-- The ratio of the area between a circle and an ellipse to the area of the circle -/
theorem area_ratio_circle_ellipse :
  let circle_diameter : ℝ := 4
  let ellipse_major_axis : ℝ := 8
  let ellipse_minor_axis : ℝ := 6
  let circle_area := π * (circle_diameter / 2)^2
  let ellipse_area := π * (ellipse_major_axis / 2) * (ellipse_minor_axis / 2)
  (ellipse_area - circle_area) / circle_area = 2 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_circle_ellipse_l1013_101300


namespace NUMINAMATH_CALUDE_three_people_seven_steps_l1013_101384

/-- The number of ways to arrange n people on m steps with at most k people per step -/
def arrange (n m k : ℕ) : ℕ :=
  sorry

/-- The number of ways to arrange 3 people on 7 steps with at most 2 people per step -/
theorem three_people_seven_steps : arrange 3 7 2 = 336 := by
  sorry

end NUMINAMATH_CALUDE_three_people_seven_steps_l1013_101384


namespace NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l1013_101314

theorem product_of_difference_and_sum_of_squares (a b : ℝ) 
  (h1 : a - b = 5) 
  (h2 : a^2 + b^2 = 13) : 
  a * b = -6 := by sorry

end NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l1013_101314


namespace NUMINAMATH_CALUDE_bird_migration_difference_l1013_101315

/-- The number of bird families that flew to Africa -/
def africa_birds : ℕ := 42

/-- The number of bird families that flew to Asia -/
def asia_birds : ℕ := 31

/-- The number of bird families living near the mountain -/
def mountain_birds : ℕ := 8

/-- Theorem stating the difference between bird families that flew to Africa and Asia -/
theorem bird_migration_difference : africa_birds - asia_birds = 11 := by
  sorry

end NUMINAMATH_CALUDE_bird_migration_difference_l1013_101315


namespace NUMINAMATH_CALUDE_alpha_value_l1013_101371

theorem alpha_value (α : Real) (h1 : α ∈ Set.Ioo 0 Real.pi) (h2 : Real.cos α = -1/2) : 
  α = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_alpha_value_l1013_101371


namespace NUMINAMATH_CALUDE_min_value_of_a_l1013_101381

theorem min_value_of_a (x y a : ℝ) (hx : x > 0) (hy : y > 0) (ha : a > 0) (hxy : x ≠ y)
  (h : (2 * x - y / Real.exp 1) * Real.log (y / x) = x / (a * Real.exp 1)) :
  a ≥ 1 / Real.exp 1 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_a_l1013_101381


namespace NUMINAMATH_CALUDE_seokjin_floors_to_bookstore_l1013_101386

/-- The floor number of the bookstore -/
def bookstore_floor : ℕ := 4

/-- Seokjin's current floor number -/
def current_floor : ℕ := 1

/-- The number of floors Seokjin must go up -/
def floors_to_go_up : ℕ := bookstore_floor - current_floor

/-- Theorem stating that Seokjin must go up 3 floors to reach the bookstore -/
theorem seokjin_floors_to_bookstore : floors_to_go_up = 3 := by
  sorry

end NUMINAMATH_CALUDE_seokjin_floors_to_bookstore_l1013_101386


namespace NUMINAMATH_CALUDE_impossible_arrangement_l1013_101351

def numbers : List ℕ := [1, 4, 9, 16, 25, 36, 49, 64, 81]

def radial_lines : ℕ := 6

def appears_twice (n : ℕ) : Prop := ∃ (l₁ l₂ : List ℕ), l₁ ≠ l₂ ∧ n ∈ l₁ ∧ n ∈ l₂

theorem impossible_arrangement : 
  ¬∃ (arrangement : List (List ℕ)), 
    (∀ n ∈ numbers, appears_twice n) ∧ 
    (arrangement.length = radial_lines) ∧
    (∃ (s : ℕ), ∀ l ∈ arrangement, l.sum = s) :=
sorry

end NUMINAMATH_CALUDE_impossible_arrangement_l1013_101351


namespace NUMINAMATH_CALUDE_power_of_three_difference_l1013_101391

theorem power_of_three_difference : 3^(2+3+4) - (3^2 + 3^3 + 3^4) = 19566 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_difference_l1013_101391


namespace NUMINAMATH_CALUDE_inverse_f_at_3_l1013_101364

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 1

-- State the theorem
theorem inverse_f_at_3 :
  ∃ (y : ℝ), y < 0 ∧ f y = 3 ∧ ∀ (z : ℝ), z < 0 ∧ f z = 3 → z = y :=
by sorry

end NUMINAMATH_CALUDE_inverse_f_at_3_l1013_101364


namespace NUMINAMATH_CALUDE_trapezoid_side_length_l1013_101353

/-- Given a trapezoid PQRS with the following properties:
  - Area is 200 cm²
  - Altitude is 10 cm
  - PQ is 15 cm
  - RS is 20 cm
  Prove that the length of QR is 20 - 2.5√5 - 5√3 cm -/
theorem trapezoid_side_length (area : ℝ) (altitude : ℝ) (pq : ℝ) (rs : ℝ) (qr : ℝ) :
  area = 200 →
  altitude = 10 →
  pq = 15 →
  rs = 20 →
  qr = 20 - 2.5 * Real.sqrt 5 - 5 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_side_length_l1013_101353


namespace NUMINAMATH_CALUDE_new_average_weight_l1013_101366

/-- Given 6 people with an average weight of 154 lbs and a 7th person weighing 133 lbs,
    prove that the new average weight of all 7 people is 151 lbs. -/
theorem new_average_weight 
  (initial_people : Nat) 
  (initial_avg_weight : ℚ) 
  (new_person_weight : ℚ) : 
  initial_people = 6 → 
  initial_avg_weight = 154 → 
  new_person_weight = 133 → 
  ((initial_people : ℚ) * initial_avg_weight + new_person_weight) / (initial_people + 1) = 151 := by
  sorry

#check new_average_weight

end NUMINAMATH_CALUDE_new_average_weight_l1013_101366


namespace NUMINAMATH_CALUDE_equation_solution_l1013_101362

theorem equation_solution : ∃! x : ℝ, 4 * x - 8 + 3 * x = 12 + 5 * x ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1013_101362


namespace NUMINAMATH_CALUDE_negative_two_thousand_ten_plus_two_l1013_101347

theorem negative_two_thousand_ten_plus_two :
  (-2010 : ℤ) + 2 = -2008 := by sorry

end NUMINAMATH_CALUDE_negative_two_thousand_ten_plus_two_l1013_101347


namespace NUMINAMATH_CALUDE_no_bounded_function_satisfying_inequality_l1013_101392

theorem no_bounded_function_satisfying_inequality :
  ¬ ∃ (f : ℝ → ℝ), (∀ x : ℝ, ∃ M : ℝ, |f x| ≤ M) ∧ 
    (f 1 > 0) ∧ 
    (∀ x y : ℝ, f (x + y)^2 ≥ f x^2 + 2 * f (x * y) + f y^2) :=
by sorry

end NUMINAMATH_CALUDE_no_bounded_function_satisfying_inequality_l1013_101392


namespace NUMINAMATH_CALUDE_quadratic_intersection_intersection_points_l1013_101363

/-- Quadratic function f(x) = x^2 - 6x + 2m - 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 6*x + 2*m - 1

theorem quadratic_intersection (m : ℝ) :
  (∀ x, f m x ≠ 0) ↔ m > 5 :=
sorry

theorem intersection_points :
  let m : ℝ := -3
  (∃ x, f m x = 0 ∧ (x = -1 ∨ x = 7)) ∧
  (f m 0 = -7) :=
sorry

end NUMINAMATH_CALUDE_quadratic_intersection_intersection_points_l1013_101363


namespace NUMINAMATH_CALUDE_arithmetic_progression_difference_l1013_101395

theorem arithmetic_progression_difference (x y z k : ℝ) : 
  x ≠ 0 → y ≠ 0 → z ≠ 0 → k ≠ 0 → x ≠ y → y ≠ z → x ≠ z →
  ∃ d : ℝ, (y * (z - x) + k) - (x * (y - z) + k) = d ∧
           (z * (x - y) + k) - (y * (z - x) + k) = d →
  d = 0 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_difference_l1013_101395


namespace NUMINAMATH_CALUDE_unique_solution_modulo_l1013_101334

theorem unique_solution_modulo : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 14 ∧ n ≡ 16427 [ZMOD 15] := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_modulo_l1013_101334


namespace NUMINAMATH_CALUDE_rectangle_length_equals_square_side_l1013_101378

/-- The length of a rectangle with width 3 cm and area equal to a square with side length 3 cm is 3 cm. -/
theorem rectangle_length_equals_square_side : 
  ∀ (length : ℝ),
  length > 0 →
  3 * length = 3 * 3 →
  length = 3 := by
sorry

end NUMINAMATH_CALUDE_rectangle_length_equals_square_side_l1013_101378


namespace NUMINAMATH_CALUDE_scientific_notation_102200_l1013_101341

theorem scientific_notation_102200 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 102200 = a * (10 : ℝ) ^ n ∧ a = 1.022 ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_102200_l1013_101341


namespace NUMINAMATH_CALUDE_arithmetic_not_geometric_l1013_101398

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def geometric_sequence (a : ℕ → ℝ) :=
  ∃ r, ∀ n, a (n + 1) = r * a n

theorem arithmetic_not_geometric (a : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a d ∧ a 1 = 2 →
  ¬(d = 4 ↔ geometric_sequence (λ n => a n)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_not_geometric_l1013_101398


namespace NUMINAMATH_CALUDE_negation_of_universal_exponential_exponential_negation_l1013_101359

theorem negation_of_universal_exponential (P : ℝ → Prop) :
  (¬ ∀ x : ℝ, P x) ↔ ∃ x : ℝ, ¬(P x) :=
by sorry

theorem exponential_negation :
  (¬ ∀ x : ℝ, Real.exp x > 0) ↔ (∃ x : ℝ, Real.exp x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_exponential_exponential_negation_l1013_101359


namespace NUMINAMATH_CALUDE_collinear_vectors_cos_2theta_l1013_101358

/-- 
Given vectors AB and BC in 2D space and the condition that points A, B, and C are collinear,
prove that cos(2θ) = 7/9.
-/
theorem collinear_vectors_cos_2theta (θ : ℝ) :
  let AB : Fin 2 → ℝ := ![- 1, - 3]
  let BC : Fin 2 → ℝ := ![2 * Real.sin θ, 2]
  (∃ (k : ℝ), AB = k • BC) →
  Real.cos (2 * θ) = 7 / 9 := by
sorry

end NUMINAMATH_CALUDE_collinear_vectors_cos_2theta_l1013_101358


namespace NUMINAMATH_CALUDE_inequality_solution_l1013_101320

theorem inequality_solution (x : ℝ) : 
  1 / (x + 2) + 5 / (x + 4) ≥ 1 ↔ x ∈ Set.Icc (-4 : ℝ) (-3) ∪ Set.Icc (-2 : ℝ) 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1013_101320


namespace NUMINAMATH_CALUDE_math_players_count_central_park_school_math_players_l1013_101394

theorem math_players_count (total_players : ℕ) (physics_players : ℕ) (both_subjects : ℕ) : ℕ :=
  let math_players := total_players - (physics_players - both_subjects)
  math_players

theorem central_park_school_math_players : 
  math_players_count 15 10 4 = 9 := by sorry

end NUMINAMATH_CALUDE_math_players_count_central_park_school_math_players_l1013_101394


namespace NUMINAMATH_CALUDE_equation_one_solution_equation_two_solution_l1013_101377

-- Equation 1
theorem equation_one_solution (x : ℝ) : 
  9 * x^2 - 25 = 0 ↔ x = 5/3 ∨ x = -5/3 := by sorry

-- Equation 2
theorem equation_two_solution (x : ℝ) : 
  (x + 1)^3 - 27 = 0 ↔ x = 2 := by sorry

end NUMINAMATH_CALUDE_equation_one_solution_equation_two_solution_l1013_101377


namespace NUMINAMATH_CALUDE_sine_of_inverse_sum_l1013_101309

theorem sine_of_inverse_sum : 
  Real.sin (Real.arcsin (4/5) + Real.arctan (1/2) + Real.arccos (3/5)) = 41 * Real.sqrt 5 / 125 := by
  sorry

end NUMINAMATH_CALUDE_sine_of_inverse_sum_l1013_101309


namespace NUMINAMATH_CALUDE_cubic_equation_property_l1013_101319

/-- A cubic equation with coefficients a, b, c, and three non-zero real roots forming a geometric progression -/
structure CubicEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  root1 : ℝ
  root2 : ℝ
  root3 : ℝ
  nonzero_roots : root1 ≠ 0 ∧ root2 ≠ 0 ∧ root3 ≠ 0
  is_root1 : root1^3 + a*root1^2 + b*root1 + c = 0
  is_root2 : root2^3 + a*root2^2 + b*root2 + c = 0
  is_root3 : root3^3 + a*root3^2 + b*root3 + c = 0
  geometric_progression : ∃ (q : ℝ), q ≠ 0 ∧ q ≠ 1 ∧ (root2 = q * root1) ∧ (root3 = q * root2)

/-- The theorem stating that a^3c - b^3 = 0 for a cubic equation with three non-zero real roots in geometric progression -/
theorem cubic_equation_property (eq : CubicEquation) : eq.a^3 * eq.c - eq.b^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_property_l1013_101319


namespace NUMINAMATH_CALUDE_workshop_attendance_workshop_attendance_proof_l1013_101397

theorem workshop_attendance : ℕ → Prop :=
  fun total =>
    ∃ (wolf nobel wolf_nobel non_wolf_nobel non_wolf_non_nobel : ℕ),
      -- Total Wolf Prize laureates
      wolf = 31 ∧
      -- Wolf Prize laureates who are also Nobel Prize laureates
      wolf_nobel = 18 ∧
      -- Total Nobel Prize laureates
      nobel = 29 ∧
      -- Difference between Nobel (non-Wolf) and non-Nobel (non-Wolf)
      non_wolf_nobel = non_wolf_non_nobel + 3 ∧
      -- Total scientists is sum of all categories
      total = wolf + non_wolf_nobel + non_wolf_non_nobel ∧
      -- Consistency check for Nobel laureates
      nobel = wolf_nobel + non_wolf_nobel ∧
      -- The total number of scientists is 50
      total = 50

theorem workshop_attendance_proof : workshop_attendance 50 := by
  sorry

end NUMINAMATH_CALUDE_workshop_attendance_workshop_attendance_proof_l1013_101397


namespace NUMINAMATH_CALUDE_sara_bought_movie_cost_l1013_101329

/-- The cost of Sara's bought movie -/
def cost_of_bought_movie (ticket_price : ℚ) (num_tickets : ℕ) (rental_price : ℚ) (total_spent : ℚ) : ℚ :=
  total_spent - (ticket_price * num_tickets + rental_price)

/-- Theorem stating the cost of Sara's bought movie -/
theorem sara_bought_movie_cost :
  let ticket_price : ℚ := 10.62
  let num_tickets : ℕ := 2
  let rental_price : ℚ := 1.59
  let total_spent : ℚ := 36.78
  cost_of_bought_movie ticket_price num_tickets rental_price total_spent = 13.95 := by
  sorry

end NUMINAMATH_CALUDE_sara_bought_movie_cost_l1013_101329


namespace NUMINAMATH_CALUDE_speed_difference_l1013_101318

/-- The difference in average speeds between no traffic and heavy traffic conditions -/
theorem speed_difference (distance : ℝ) (time_heavy : ℝ) (time_no : ℝ)
  (h_distance : distance = 200)
  (h_time_heavy : time_heavy = 5)
  (h_time_no : time_no = 4) :
  distance / time_no - distance / time_heavy = 10 := by
  sorry

end NUMINAMATH_CALUDE_speed_difference_l1013_101318


namespace NUMINAMATH_CALUDE_first_hole_depth_l1013_101326

/-- Represents the depth of a hole dug by workers -/
structure HoleDigging where
  workers : ℕ
  hours : ℕ
  depth : ℝ

theorem first_hole_depth 
  (hole1 : HoleDigging)
  (hole2 : HoleDigging)
  (h1 : hole1.workers = 45)
  (h2 : hole1.hours = 8)
  (h3 : hole2.workers = 110)
  (h4 : hole2.hours = 6)
  (h5 : hole2.depth = 55)
  (h6 : hole1.workers * hole1.hours * hole2.depth = hole2.workers * hole2.hours * hole1.depth) :
  hole1.depth = 30 := by
sorry


end NUMINAMATH_CALUDE_first_hole_depth_l1013_101326


namespace NUMINAMATH_CALUDE_parabola_properties_l1013_101311

-- Define the parabola equation
def parabola_equation (x y : ℝ) : Prop :=
  4 * x^2 + 4 * x * y + y^2 - 10 * y - 15 = 0

-- Define the axis of symmetry
def axis_of_symmetry (x y : ℝ) : Prop :=
  2 * x + y - 1 = 0

-- Define the directrix
def directrix (x y : ℝ) : Prop :=
  x - 2 * y - 5 = 0

-- Define the tangent line
def tangent_line (y : ℝ) : Prop :=
  2 * y + 3 = 0

-- Theorem statement
theorem parabola_properties :
  ∀ (x y : ℝ),
    parabola_equation x y →
    (∃ (x₀ y₀ : ℝ), axis_of_symmetry x₀ y₀ ∧
                     directrix x₀ y₀ ∧
                     tangent_line y₀) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l1013_101311


namespace NUMINAMATH_CALUDE_plant_equation_correct_l1013_101399

/-- Represents the structure of a plant with branches and small branches. -/
structure Plant where
  branches : ℕ
  smallBranches : ℕ

/-- The total number of parts in a plant, including the main stem. -/
def totalParts (p : Plant) : ℕ := 1 + p.branches + p.smallBranches

/-- Constructs a plant where each branch grows a specific number of small branches. -/
def makePlant (x : ℕ) : Plant :=
  { branches := x, smallBranches := x * x }

/-- Theorem stating that for some natural number x, the plant structure
    results in a total of 91 parts. -/
theorem plant_equation_correct :
  ∃ x : ℕ, totalParts (makePlant x) = 91 := by
  sorry

end NUMINAMATH_CALUDE_plant_equation_correct_l1013_101399


namespace NUMINAMATH_CALUDE_athlete_weight_problem_l1013_101342

theorem athlete_weight_problem (a b c : ℕ) : 
  (a + b + c) / 3 = 42 →
  (a + b) / 2 = 40 →
  (b + c) / 2 = 43 →
  ∃ k₁ k₂ k₃ : ℕ, a = 5 * k₁ ∧ b = 5 * k₂ ∧ c = 5 * k₃ →
  b = 40 := by
sorry

end NUMINAMATH_CALUDE_athlete_weight_problem_l1013_101342


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1013_101354

theorem negation_of_proposition (p : Prop) : 
  (¬ (∀ x : ℝ, x ≥ 0 → x - 2 > 0)) ↔ (∃ x : ℝ, x ≥ 0 ∧ x - 2 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1013_101354


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1013_101383

def M : Set ℤ := {-1, 0, 1, 2}

def f (x : ℤ) : ℤ := Int.natAbs x

def N : Set ℤ := f '' M

theorem intersection_of_M_and_N : M ∩ N = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1013_101383


namespace NUMINAMATH_CALUDE_prime_sequence_l1013_101340

theorem prime_sequence (n : ℕ) (h1 : n ≥ 2) :
  (∀ k : ℕ, 0 ≤ k ∧ k ≤ Real.sqrt (n / 3) → Nat.Prime (k^2 + k + n)) →
  (∀ k : ℕ, 0 ≤ k ∧ k ≤ n - 2 → Nat.Prime (k^2 + k + n)) :=
by sorry

end NUMINAMATH_CALUDE_prime_sequence_l1013_101340


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1013_101324

/-- Given an arithmetic sequence {a_n} with sum of first n terms S_n,
    prove that if S_6 = 3S_2 + 24, then the common difference d = 2 -/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) -- a_n is the nth term of the arithmetic sequence
  (S : ℕ → ℝ) -- S_n is the sum of the first n terms
  (h_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1) -- condition for arithmetic sequence
  (h_sum : ∀ n, S n = n * (a 1 + a n) / 2) -- formula for sum of arithmetic sequence
  (h_given : S 6 = 3 * S 2 + 24) -- given condition
  : a 2 - a 1 = 2 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1013_101324


namespace NUMINAMATH_CALUDE_circle_center_line_segment_length_l1013_101303

/-- Circle C with equation x^2 + y^2 - 2x - 2y + 1 = 0 -/
def CircleC (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 2*y + 1 = 0

/-- Line l with equation x - y = 0 -/
def LineL (x y : ℝ) : Prop :=
  x - y = 0

/-- The center of circle C is at (1, 1) -/
theorem circle_center : ∃ (x y : ℝ), CircleC x y ∧ x = 1 ∧ y = 1 :=
sorry

/-- The length of line segment AB, where A and B are intersection points of circle C and line l, is 2√2 -/
theorem line_segment_length :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    CircleC x₁ y₁ ∧ CircleC x₂ y₂ ∧
    LineL x₁ y₁ ∧ LineL x₂ y₂ ∧
    x₁ ≠ x₂ ∧
    ((x₁ - x₂)^2 + (y₁ - y₂)^2) = 8 :=
sorry

end NUMINAMATH_CALUDE_circle_center_line_segment_length_l1013_101303


namespace NUMINAMATH_CALUDE_ball_max_height_l1013_101356

/-- The height function of the ball -/
def h (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 16

/-- Theorem stating that the maximum height of the ball is 141 feet -/
theorem ball_max_height :
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 141 :=
sorry

end NUMINAMATH_CALUDE_ball_max_height_l1013_101356


namespace NUMINAMATH_CALUDE_coterminal_pi_third_pi_equals_180_degrees_arc_length_pi_third_l1013_101321

-- Define the set of coterminal angles
def coterminalAngles (θ : ℝ) : Set ℝ := {α | ∃ k : ℤ, α = θ + 2 * k * Real.pi}

-- Statement 1: Coterminal angles with π/3
theorem coterminal_pi_third : 
  coterminalAngles (Real.pi / 3) = {α | ∃ k : ℤ, α = Real.pi / 3 + 2 * k * Real.pi} :=
sorry

-- Statement 2: π radians equals 180 degrees
theorem pi_equals_180_degrees : 
  Real.pi = 180 * (Real.pi / 180) :=
sorry

-- Statement 3: Arc length in a circle
theorem arc_length_pi_third : 
  let r : ℝ := 6
  let θ : ℝ := Real.pi / 3
  r * θ = 2 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_coterminal_pi_third_pi_equals_180_degrees_arc_length_pi_third_l1013_101321


namespace NUMINAMATH_CALUDE_largest_c_for_range_l1013_101337

theorem largest_c_for_range (f : ℝ → ℝ) (c : ℝ) : 
  (∀ x, f x = x^2 - 7*x + c) →
  (∃ x, f x = 3) →
  c ≤ 61/4 ∧ ∀ d > 61/4, ¬∃ x, x^2 - 7*x + d = 3 :=
by sorry

end NUMINAMATH_CALUDE_largest_c_for_range_l1013_101337


namespace NUMINAMATH_CALUDE_german_students_count_l1013_101301

theorem german_students_count (total_students : ℕ) 
                               (french_students : ℕ) 
                               (both_students : ℕ) 
                               (neither_students : ℕ) 
                               (h1 : total_students = 60)
                               (h2 : french_students = 41)
                               (h3 : both_students = 9)
                               (h4 : neither_students = 6) :
  ∃ german_students : ℕ, german_students = 22 ∧ 
    german_students + french_students - both_students + neither_students = total_students :=
by
  sorry

end NUMINAMATH_CALUDE_german_students_count_l1013_101301


namespace NUMINAMATH_CALUDE_farrah_order_proof_l1013_101360

/-- The number of matchboxes in each box -/
def matchboxes_per_box : ℕ := 20

/-- The number of match sticks in each matchbox -/
def sticks_per_matchbox : ℕ := 300

/-- The total number of match sticks ordered -/
def total_sticks : ℕ := 24000

/-- The number of boxes Farrah ordered -/
def boxes_ordered : ℕ := total_sticks / (matchboxes_per_box * sticks_per_matchbox)

theorem farrah_order_proof : boxes_ordered = 4 := by
  sorry

end NUMINAMATH_CALUDE_farrah_order_proof_l1013_101360


namespace NUMINAMATH_CALUDE_product_mod_eleven_l1013_101393

theorem product_mod_eleven : (103 * 107) % 11 = 10 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_eleven_l1013_101393


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l1013_101339

theorem arithmetic_expression_equality : 61 + 5 * 12 / (180 / 3) = 62 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l1013_101339


namespace NUMINAMATH_CALUDE_inequality_proof_l1013_101390

theorem inequality_proof (a : ℝ) : (3 * a - 6) * (2 * a^2 - a^3) ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1013_101390


namespace NUMINAMATH_CALUDE_royal_family_children_count_l1013_101376

/-- Represents the number of years that have passed -/
def n : ℕ := sorry

/-- Represents the number of daughters -/
def d : ℕ := sorry

/-- The total number of children -/
def total_children : ℕ := d + 3

/-- The initial age of the king and queen -/
def initial_royal_age : ℕ := 35

/-- The initial total age of the children -/
def initial_children_age : ℕ := 35

/-- The combined age of the king and queen after n years -/
def royal_age_after_n_years : ℕ := 2 * initial_royal_age + 2 * n

/-- The total age of the children after n years -/
def children_age_after_n_years : ℕ := initial_children_age + total_children * n

theorem royal_family_children_count :
  (royal_age_after_n_years = children_age_after_n_years) ∧
  (total_children ≤ 20) →
  (total_children = 7 ∨ total_children = 9) :=
by sorry

end NUMINAMATH_CALUDE_royal_family_children_count_l1013_101376


namespace NUMINAMATH_CALUDE_positive_solution_of_equation_l1013_101382

theorem positive_solution_of_equation (x : ℝ) :
  x > 0 ∧ x + 17 = 60 * (1/x) → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_positive_solution_of_equation_l1013_101382


namespace NUMINAMATH_CALUDE_max_both_writers_and_editors_l1013_101336

/-- Conference attendees --/
structure Conference where
  total : ℕ
  writers : ℕ
  editors : ℕ
  both : ℕ
  neither : ℕ

/-- Conference conditions --/
def ConferenceConditions (c : Conference) : Prop :=
  c.total = 100 ∧
  c.writers = 40 ∧
  c.editors > 38 ∧
  c.neither = 2 * c.both ∧
  c.writers + c.editors - c.both + c.neither = c.total

/-- Theorem: The maximum number of people who are both writers and editors is 21 --/
theorem max_both_writers_and_editors (c : Conference) 
  (h : ConferenceConditions c) : c.both ≤ 21 := by
  sorry

end NUMINAMATH_CALUDE_max_both_writers_and_editors_l1013_101336


namespace NUMINAMATH_CALUDE_area_ratio_theorem_l1013_101306

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)

-- Define point W on side XZ
def W (t : Triangle) : ℝ × ℝ := sorry

-- Define the conditions
axiom XW_length : ∀ t : Triangle, dist (t.X) (W t) = 9
axiom WZ_length : ∀ t : Triangle, dist (W t) (t.Z) = 15

-- Define the areas of triangles XYW and WYZ
def area_XYW (t : Triangle) : ℝ := sorry
def area_WYZ (t : Triangle) : ℝ := sorry

-- State the theorem
theorem area_ratio_theorem (t : Triangle) :
  (area_XYW t) / (area_WYZ t) = 3 / 5 :=
sorry

end NUMINAMATH_CALUDE_area_ratio_theorem_l1013_101306


namespace NUMINAMATH_CALUDE_original_number_calculation_l1013_101331

theorem original_number_calculation (r : ℝ) : 
  (r + 0.15 * r) - (r - 0.30 * r) = 40 → r = 40 / 0.45 := by
sorry

end NUMINAMATH_CALUDE_original_number_calculation_l1013_101331


namespace NUMINAMATH_CALUDE_sneezing_fit_proof_l1013_101373

/-- Calculates the number of sneezes given the duration of a sneezing fit in minutes
    and the interval between sneezes in seconds. -/
def number_of_sneezes (duration_minutes : ℕ) (interval_seconds : ℕ) : ℕ :=
  (duration_minutes * 60) / interval_seconds

/-- Proves that a 2-minute sneezing fit with sneezes every 3 seconds results in 40 sneezes. -/
theorem sneezing_fit_proof :
  number_of_sneezes 2 3 = 40 := by
  sorry

end NUMINAMATH_CALUDE_sneezing_fit_proof_l1013_101373


namespace NUMINAMATH_CALUDE_max_product_of_two_different_numbers_exists_max_product_l1013_101385

def S : Set Int := {-9, -5, -3, 0, 4, 5, 8}

theorem max_product_of_two_different_numbers (a b : Int) :
  a ∈ S → b ∈ S → a ≠ b → a * b ≤ 45 := by
  sorry

theorem exists_max_product :
  ∃ a b : Int, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a * b = 45 := by
  sorry

end NUMINAMATH_CALUDE_max_product_of_two_different_numbers_exists_max_product_l1013_101385


namespace NUMINAMATH_CALUDE_negation_of_p_l1013_101344

def p (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0

theorem negation_of_p (f : ℝ → ℝ) :
  ¬(p f) ↔ ∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0 :=
by sorry

end NUMINAMATH_CALUDE_negation_of_p_l1013_101344


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l1013_101365

/-- 
Given two parallel vectors a and b in ℝ², where a = (-2, 1) and b = (1, m),
prove that m = -1/2.
-/
theorem parallel_vectors_m_value (a b : ℝ × ℝ) (m : ℝ) 
  (h1 : a = (-2, 1)) 
  (h2 : b = (1, m)) 
  (h3 : ∃ (k : ℝ), a = k • b) : 
  m = -1/2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l1013_101365


namespace NUMINAMATH_CALUDE_least_days_to_double_l1013_101375

/-- The least number of days for a loan to double with daily compound interest -/
theorem least_days_to_double (principal : ℝ) (rate : ℝ) (n : ℕ) : 
  principal > 0 → 
  rate > 0 → 
  principal * (1 + rate) ^ n ≥ 2 * principal → 
  principal * (1 + rate) ^ (n - 1) < 2 * principal → 
  principal = 20 → 
  rate = 0.1 → 
  n = 8 := by
  sorry

end NUMINAMATH_CALUDE_least_days_to_double_l1013_101375


namespace NUMINAMATH_CALUDE_square_area_perimeter_ratio_l1013_101352

theorem square_area_perimeter_ratio : 
  ∀ (s₁ s₂ : ℝ), s₁ > 0 → s₂ > 0 → 
  (s₁^2 / s₂^2 = 16 / 49) → 
  ((4 * s₁) / (4 * s₂) = 4 / 7) := by
sorry

end NUMINAMATH_CALUDE_square_area_perimeter_ratio_l1013_101352


namespace NUMINAMATH_CALUDE_back_section_total_revenue_l1013_101361

/-- Calculates the total revenue from the back section of a concert arena --/
def back_section_revenue (capacity : ℕ) (regular_price : ℚ) (half_price : ℚ) : ℚ :=
  let regular_revenue := regular_price * capacity
  let half_price_tickets := capacity / 6
  let half_price_revenue := half_price * half_price_tickets
  regular_revenue + half_price_revenue

/-- Theorem stating the total revenue from the back section --/
theorem back_section_total_revenue :
  back_section_revenue 25000 55 27.5 = 1489565 := by
  sorry

#eval back_section_revenue 25000 55 27.5

end NUMINAMATH_CALUDE_back_section_total_revenue_l1013_101361
