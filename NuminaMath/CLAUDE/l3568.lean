import Mathlib

namespace NUMINAMATH_CALUDE_numerical_puzzle_solutions_l3568_356843

/-- A function that checks if a number is a two-digit number -/
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

/-- A function that extracts the tens digit of a two-digit number -/
def tens_digit (n : ℕ) : ℕ :=
  n / 10

/-- A function that extracts the ones digit of a two-digit number -/
def ones_digit (n : ℕ) : ℕ :=
  n % 10

/-- The main theorem stating the solutions to the numerical puzzle -/
theorem numerical_puzzle_solutions :
  ∀ n : ℕ, is_two_digit n →
    (∃ b v : ℕ, 
      n = b^v ∧ 
      tens_digit n ≠ ones_digit n ∧
      b = ones_digit n) ↔ 
    (n = 32 ∨ n = 36 ∨ n = 64) :=
sorry

end NUMINAMATH_CALUDE_numerical_puzzle_solutions_l3568_356843


namespace NUMINAMATH_CALUDE_c_is_power_of_two_l3568_356893

/-- Represents a string of base-ten digits -/
def DigitString : Type := List Nat

/-- Checks if a DigitString represents a number divisible by m -/
def isDivisibleBy (s : DigitString) (m : Nat) : Prop := sorry

/-- Counts the number of valid splits of a DigitString -/
def c (S : DigitString) (m : Nat) : Nat := sorry

/-- A natural number is a power of 2 -/
def isPowerOfTwo (n : Nat) : Prop := ∃ k : Nat, n = 2^k

theorem c_is_power_of_two (m : Nat) (S : DigitString) (h1 : m > 1) (h2 : S ≠ []) :
  c S m = 0 ∨ isPowerOfTwo (c S m) := by sorry

end NUMINAMATH_CALUDE_c_is_power_of_two_l3568_356893


namespace NUMINAMATH_CALUDE_work_completion_equality_second_group_size_correct_l3568_356824

/-- The number of men in the first group -/
def first_group : ℕ := 12

/-- The number of days the first group takes to complete the work -/
def first_days : ℕ := 30

/-- The number of days the second group takes to complete the work -/
def second_days : ℕ := 36

/-- The number of men in the second group -/
def second_group : ℕ := 10

theorem work_completion_equality :
  first_group * first_days = second_group * second_days :=
by sorry

/-- Proves that the number of men in the second group is correct -/
theorem second_group_size_correct :
  second_group = (first_group * first_days) / second_days :=
by sorry

end NUMINAMATH_CALUDE_work_completion_equality_second_group_size_correct_l3568_356824


namespace NUMINAMATH_CALUDE_total_lines_eq_88_l3568_356810

/-- The number of sides in a triangle -/
def triangle_sides : ℕ := 3

/-- The number of sides in a square -/
def square_sides : ℕ := 4

/-- The number of sides in a pentagon -/
def pentagon_sides : ℕ := 5

/-- The number of triangles drawn -/
def num_triangles : ℕ := 12

/-- The number of squares drawn -/
def num_squares : ℕ := 8

/-- The number of pentagons drawn -/
def num_pentagons : ℕ := 4

/-- The total number of lines drawn -/
def total_lines : ℕ := num_triangles * triangle_sides + num_squares * square_sides + num_pentagons * pentagon_sides

theorem total_lines_eq_88 : total_lines = 88 := by
  sorry

end NUMINAMATH_CALUDE_total_lines_eq_88_l3568_356810


namespace NUMINAMATH_CALUDE_salary_percentage_decrease_l3568_356832

theorem salary_percentage_decrease 
  (x : ℝ) -- Original salary
  (h1 : x * 1.15 = 575) -- 15% increase condition
  (h2 : x * (1 - y / 100) = 560) -- y% decrease condition
  : y = 12 := by
  sorry

end NUMINAMATH_CALUDE_salary_percentage_decrease_l3568_356832


namespace NUMINAMATH_CALUDE_expression_value_l3568_356805

theorem expression_value (x y : ℤ) (hx : x = -2) (hy : y = 4) :
  5 * x - 2 * y + 7 = -11 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3568_356805


namespace NUMINAMATH_CALUDE_vacation_towel_problem_l3568_356815

theorem vacation_towel_problem (families : ℕ) (days : ℕ) (towels_per_person_per_day : ℕ) 
  (towels_per_load : ℕ) (total_loads : ℕ) :
  families = 3 →
  days = 7 →
  towels_per_person_per_day = 1 →
  towels_per_load = 14 →
  total_loads = 6 →
  (total_loads * towels_per_load) / (days * families) = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_vacation_towel_problem_l3568_356815


namespace NUMINAMATH_CALUDE_geometric_sequence_logarithm_l3568_356811

/-- Given a geometric sequence {a_n} with common ratio -√2, 
    prove that ln(a_{2017})^2 - ln(a_{2016})^2 = ln(2) -/
theorem geometric_sequence_logarithm (a : ℕ → ℝ) 
  (h : ∀ n, a (n + 1) = a n * (-Real.sqrt 2)) :
  (Real.log (a 2017))^2 - (Real.log (a 2016))^2 = Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_logarithm_l3568_356811


namespace NUMINAMATH_CALUDE_bonus_distribution_solution_l3568_356891

/-- Represents the bonus distribution problem -/
def BonusDistribution (total : ℚ) (ac_sum : ℚ) (common_ratio : ℚ) (d_bonus : ℚ) : Prop :=
  let a := d_bonus / (common_ratio^3)
  let b := d_bonus / (common_ratio^2)
  let c := d_bonus / common_ratio
  (a + b + c + d_bonus = total) ∧ 
  (a + c = ac_sum) ∧
  (0 < common_ratio) ∧ 
  (common_ratio < 1)

/-- The theorem stating the correct solution to the bonus distribution problem -/
theorem bonus_distribution_solution :
  BonusDistribution 68780 36200 (9/10) 14580 := by
  sorry

#check bonus_distribution_solution

end NUMINAMATH_CALUDE_bonus_distribution_solution_l3568_356891


namespace NUMINAMATH_CALUDE_namjoon_used_seven_pencils_l3568_356889

/-- Represents the number of pencils each person has at different stages --/
structure PencilCount where
  initial : Nat
  after_taehyung_gives : Nat
  final : Nat

/-- The problem setup --/
def problem : PencilCount × PencilCount := 
  ({ initial := 10, after_taehyung_gives := 7, final := 6 },  -- Taehyung's pencils
   { initial := 10, after_taehyung_gives := 13, final := 6 }) -- Namjoon's pencils

/-- Calculates the number of pencils Namjoon used --/
def pencils_namjoon_used (p : PencilCount × PencilCount) : Nat :=
  p.2.after_taehyung_gives - p.2.final

/-- Theorem stating that Namjoon used 7 pencils --/
theorem namjoon_used_seven_pencils :
  pencils_namjoon_used problem = 7 := by
  sorry

end NUMINAMATH_CALUDE_namjoon_used_seven_pencils_l3568_356889


namespace NUMINAMATH_CALUDE_diff_same_digits_div_by_9_no_solution_to_puzzle_l3568_356841

-- Define a function to check if two numbers have the same digits
def haveSameDigits (a b : ℕ) : Prop := sorry

-- Define the property that the difference of numbers with the same digits is divisible by 9
theorem diff_same_digits_div_by_9 (a b : ℕ) (h : haveSameDigits a b) : 
  9 ∣ (a - b) := sorry

-- State the main theorem
theorem no_solution_to_puzzle : 
  ¬ ∃ (a b : ℕ), haveSameDigits a b ∧ a - b = 2018 * 2019 := by
  sorry

end NUMINAMATH_CALUDE_diff_same_digits_div_by_9_no_solution_to_puzzle_l3568_356841


namespace NUMINAMATH_CALUDE_norma_cards_l3568_356855

/-- Given that Norma has 88.0 cards initially and finds 70.0 more cards,
    prove that she will have 158.0 cards in total. -/
theorem norma_cards (initial_cards : Float) (found_cards : Float)
    (h1 : initial_cards = 88.0)
    (h2 : found_cards = 70.0) :
  initial_cards + found_cards = 158.0 := by
  sorry

end NUMINAMATH_CALUDE_norma_cards_l3568_356855


namespace NUMINAMATH_CALUDE_appropriate_word_count_l3568_356833

-- Define the presentation parameters
def min_duration : ℕ := 40
def max_duration : ℕ := 50
def speech_rate : ℕ := 160

-- Define the range of appropriate word counts
def min_words : ℕ := min_duration * speech_rate
def max_words : ℕ := max_duration * speech_rate

-- Theorem statement
theorem appropriate_word_count (word_count : ℕ) :
  (min_words ≤ word_count ∧ word_count ≤ max_words) ↔
  (word_count ≥ 6400 ∧ word_count ≤ 8000) :=
by sorry

end NUMINAMATH_CALUDE_appropriate_word_count_l3568_356833


namespace NUMINAMATH_CALUDE_sum_of_coordinates_on_inverse_graph_l3568_356844

-- Define the function f
def f : ℝ → ℝ := sorry

-- Theorem statement
theorem sum_of_coordinates_on_inverse_graph : 
  (f 2 = 6) → -- This condition is derived from (2,3) being on y=f(x)/2
  ∃ x y : ℝ, (y = 2 * (f⁻¹ x)) ∧ (x + y = 10) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_on_inverse_graph_l3568_356844


namespace NUMINAMATH_CALUDE_melissa_score_l3568_356860

/-- Calculates the total score for a player given points per game and number of games played -/
def totalScore (pointsPerGame : ℕ) (numGames : ℕ) : ℕ :=
  pointsPerGame * numGames

/-- Proves that a player scoring 7 points per game for 3 games has a total score of 21 points -/
theorem melissa_score : totalScore 7 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_melissa_score_l3568_356860


namespace NUMINAMATH_CALUDE_renovation_project_equation_l3568_356817

/-- Represents the relationship between the number of workers hired in a renovation project --/
theorem renovation_project_equation (x y : ℕ) : 
  (∀ (carpenter_wage mason_wage labor_budget : ℕ), 
    carpenter_wage = 50 ∧ 
    mason_wage = 40 ∧ 
    labor_budget = 2000 → 
    50 * x + 40 * y ≤ 2000) ↔ 
  5 * x + 4 * y ≤ 200 :=
by sorry

end NUMINAMATH_CALUDE_renovation_project_equation_l3568_356817


namespace NUMINAMATH_CALUDE_no_two_digit_reverse_sum_twice_square_l3568_356877

theorem no_two_digit_reverse_sum_twice_square : 
  ¬ ∃ (N : ℕ), 
    (10 ≤ N ∧ N ≤ 99) ∧ 
    ∃ (k : ℕ), 
      N + (10 * (N % 10) + N / 10) = 2 * k^2 := by
  sorry

end NUMINAMATH_CALUDE_no_two_digit_reverse_sum_twice_square_l3568_356877


namespace NUMINAMATH_CALUDE_lisa_additional_marbles_l3568_356872

/-- The minimum number of additional marbles Lisa needs -/
def minimum_additional_marbles (num_friends : ℕ) (current_marbles : ℕ) : ℕ :=
  let min_marbles_per_friend := 3
  let max_marbles_per_friend := min_marbles_per_friend + num_friends - 1
  let total_marbles_needed := num_friends * (min_marbles_per_friend + max_marbles_per_friend) / 2
  max (total_marbles_needed - current_marbles) 0

/-- Theorem stating the minimum number of additional marbles Lisa needs -/
theorem lisa_additional_marbles :
  minimum_additional_marbles 12 50 = 52 := by
  sorry

#eval minimum_additional_marbles 12 50

end NUMINAMATH_CALUDE_lisa_additional_marbles_l3568_356872


namespace NUMINAMATH_CALUDE_sum_of_first_n_integers_second_difference_constant_sum_formula_l3568_356835

def f (n : ℕ) : ℕ := (List.range n).sum + n

theorem sum_of_first_n_integers (n : ℕ) : 
  f n = n * (n + 1) / 2 :=
by sorry

theorem second_difference_constant (n : ℕ) : 
  f (n + 2) - 2 * f (n + 1) + f n = 1 :=
by sorry

theorem sum_formula (n : ℕ) : 
  (List.range n).sum + n = n * (n + 1) / 2 :=
by
  have h1 := sum_of_first_n_integers n
  have h2 := second_difference_constant n
  sorry

end NUMINAMATH_CALUDE_sum_of_first_n_integers_second_difference_constant_sum_formula_l3568_356835


namespace NUMINAMATH_CALUDE_gcd_120_168_l3568_356897

theorem gcd_120_168 : Nat.gcd 120 168 = 24 := by
  sorry

end NUMINAMATH_CALUDE_gcd_120_168_l3568_356897


namespace NUMINAMATH_CALUDE_quadratic_form_minimum_l3568_356894

theorem quadratic_form_minimum (x y z : ℝ) :
  x^2 + 2*x*y + 3*y^2 + 2*x*z + 3*z^2 ≥ 0 ∧
  (x^2 + 2*x*y + 3*y^2 + 2*x*z + 3*z^2 = 0 ↔ x = 0 ∧ y = 0 ∧ z = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_form_minimum_l3568_356894


namespace NUMINAMATH_CALUDE_problem_solution_l3568_356888

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 3 * (a + 1) * x^2 + 6 * a * x

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 6 * x^2 - 6 * (a + 1) * x + 6 * a

-- Define M(a) and m(a)
def M (a : ℝ) : ℝ := max (f a 1) (f a 2)
def m (a : ℝ) : ℝ := min (f a 1) (f a 2)

-- Define h(a)
def h (a : ℝ) : ℝ := M a - m a

theorem problem_solution :
  (∀ a : ℝ, f' a 0 = 3 → a = 1/2) ∧
  (∀ a : ℝ, (∀ x : ℝ, x > 0 → f a x + f a (-x) ≥ 12 * Real.log x) →
    a ≤ -1 - Real.exp (-1)) ∧
  (∀ a : ℝ, a > 1 →
    (∃ min_h : ℝ, min_h = 8/27 ∧
      ∀ a' : ℝ, a' > 1 → h a' ≥ min_h)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3568_356888


namespace NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_sine_l3568_356861

/-- An isosceles triangle with a base angle tangent of 2/3 has a vertex angle sine of 12/13 -/
theorem isosceles_triangle_vertex_angle_sine (α β : Real) : 
  -- α is a base angle of the isosceles triangle
  -- β is the vertex angle of the isosceles triangle
  -- The triangle is isosceles
  β = π - 2 * α →
  -- The tangent of the base angle is 2/3
  Real.tan α = 2 / 3 →
  -- The sine of the vertex angle is 12/13
  Real.sin β = 12 / 13 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_sine_l3568_356861


namespace NUMINAMATH_CALUDE_octagon_area_ratio_octagon_area_ratio_proof_l3568_356818

/-- The ratio of the area of a regular octagon circumscribed about a circle
    to the area of a regular octagon inscribed in the same circle is 2. -/
theorem octagon_area_ratio : ℝ → ℝ → Prop :=
  fun (area_circumscribed area_inscribed : ℝ) =>
    area_circumscribed / area_inscribed = 2

/-- Given a circle with radius r, the area of its circumscribed regular octagon
    is twice the area of its inscribed regular octagon. -/
theorem octagon_area_ratio_proof (r : ℝ) (r_pos : r > 0) :
  ∃ (area_circumscribed area_inscribed : ℝ),
    area_circumscribed > 0 ∧
    area_inscribed > 0 ∧
    octagon_area_ratio area_circumscribed area_inscribed :=
by
  sorry

end NUMINAMATH_CALUDE_octagon_area_ratio_octagon_area_ratio_proof_l3568_356818


namespace NUMINAMATH_CALUDE_linear_equation_solution_l3568_356867

theorem linear_equation_solution : 
  ∀ x : ℝ, (x + 1) / 3 = 0 ↔ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l3568_356867


namespace NUMINAMATH_CALUDE_modular_home_cost_l3568_356874

-- Define the parameters of the modular home
def total_area : ℝ := 3500
def kitchen_area : ℝ := 500
def kitchen_cost : ℝ := 35000
def bathroom_area : ℝ := 250
def bathroom_cost : ℝ := 15000
def bedroom_area : ℝ := 350
def bedroom_cost : ℝ := 21000
def living_area : ℝ := 600
def living_area_cost_per_sqft : ℝ := 100
def upgraded_cost_per_sqft : ℝ := 150

def num_kitchens : ℕ := 1
def num_bathrooms : ℕ := 3
def num_bedrooms : ℕ := 4
def num_living_areas : ℕ := 1

-- Define the theorem
theorem modular_home_cost :
  let total_module_area := kitchen_area * num_kitchens + bathroom_area * num_bathrooms +
                           bedroom_area * num_bedrooms + living_area * num_living_areas
  let remaining_area := total_area - total_module_area
  let upgraded_area := remaining_area / 2
  let total_cost := kitchen_cost * num_kitchens + bathroom_cost * num_bathrooms +
                    bedroom_cost * num_bedrooms + living_area * living_area_cost_per_sqft +
                    upgraded_area * upgraded_cost_per_sqft * 2
  total_cost = 261500 := by sorry

end NUMINAMATH_CALUDE_modular_home_cost_l3568_356874


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l3568_356808

/-- Given a triangle with two known angles of 45° and 70°, prove that the third angle is 65° and the largest angle is 70°. -/
theorem triangle_angle_proof (a b c : ℝ) : 
  a = 45 → b = 70 → a + b + c = 180 → 
  c = 65 ∧ max a (max b c) = 70 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l3568_356808


namespace NUMINAMATH_CALUDE_sweep_time_is_three_l3568_356803

/-- The time in minutes it takes to sweep one room -/
def sweep_time : ℝ := sorry

/-- The time in minutes it takes to wash one dish -/
def dish_time : ℝ := 2

/-- The time in minutes it takes to do one load of laundry -/
def laundry_time : ℝ := 9

/-- The number of rooms Anna sweeps -/
def anna_rooms : ℕ := 10

/-- The number of loads of laundry Billy does -/
def billy_laundry : ℕ := 2

/-- The number of dishes Billy washes -/
def billy_dishes : ℕ := 6

theorem sweep_time_is_three :
  sweep_time = 3 ∧
  anna_rooms * sweep_time = billy_laundry * laundry_time + billy_dishes * dish_time :=
by sorry

end NUMINAMATH_CALUDE_sweep_time_is_three_l3568_356803


namespace NUMINAMATH_CALUDE_range_of_a_l3568_356846

/-- Given that for any x ≥ 1, ln x - a(1 - 1/x) ≥ 0, prove that a ≤ 1 -/
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ≥ 1 → Real.log x - a * (1 - 1/x) ≥ 0) → 
  a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3568_356846


namespace NUMINAMATH_CALUDE_john_horses_count_l3568_356830

/-- Represents the number of horses John has -/
def num_horses : ℕ := 25

/-- Represents the number of feedings per day for each horse -/
def feedings_per_day : ℕ := 2

/-- Represents the amount of food in pounds per feeding -/
def food_per_feeding : ℕ := 20

/-- Represents the weight of a bag of food in pounds -/
def bag_weight : ℕ := 1000

/-- Represents the number of days -/
def num_days : ℕ := 60

/-- Represents the number of bags needed for the given number of days -/
def num_bags : ℕ := 60

theorem john_horses_count :
  num_horses * feedings_per_day * food_per_feeding * num_days = num_bags * bag_weight := by
  sorry


end NUMINAMATH_CALUDE_john_horses_count_l3568_356830


namespace NUMINAMATH_CALUDE_quadratic_trinomial_theorem_l3568_356858

/-- A quadratic trinomial with real coefficients -/
structure QuadraticTrinomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Condition: replacing any coefficient with 1 results in a trinomial with exactly one root -/
def has_single_root_when_replaced (q : QuadraticTrinomial) : Prop :=
  (1^2 - 4*q.b*q.c = 0) ∧ (q.b^2 - 4*1*q.c = 0) ∧ (q.b^2 - 4*q.a*1 = 0)

/-- Theorem: If a quadratic trinomial satisfies the condition, then its coefficients are a = c = 1/2 and b = ±√2 -/
theorem quadratic_trinomial_theorem (q : QuadraticTrinomial) :
  has_single_root_when_replaced q →
  (q.a = 1/2 ∧ q.c = 1/2 ∧ (q.b = Real.sqrt 2 ∨ q.b = -Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_theorem_l3568_356858


namespace NUMINAMATH_CALUDE_min_value_z_plus_inv_z_squared_l3568_356869

/-- Given a complex number z with positive real part, and a parallelogram formed by the points 0, z, 1/z, and z + 1/z with an area of 12/13, the minimum value of |z + 1/z|² is 16/13. -/
theorem min_value_z_plus_inv_z_squared (z : ℂ) (h_real_pos : 0 < z.re) 
  (h_area : abs (z.im * (1/z).re - z.re * (1/z).im) = 12/13) :
  ∃ d : ℝ, d^2 = 16/13 ∧ ∀ w : ℂ, w.re > 0 → 
    abs (w.im * (1/w).re - w.re * (1/w).im) = 12/13 → 
    d^2 ≤ Complex.normSq (w + 1/w) := by
  sorry

end NUMINAMATH_CALUDE_min_value_z_plus_inv_z_squared_l3568_356869


namespace NUMINAMATH_CALUDE_total_evaluations_is_2680_l3568_356876

/-- Represents a class with its exam components and student count -/
structure ExamClass where
  students : ℕ
  multipleChoice : ℕ
  shortAnswer : ℕ
  essay : ℕ
  otherEvaluations : ℕ

/-- Calculates the total evaluations for a single class -/
def classEvaluations (c : ExamClass) : ℕ :=
  c.students * (c.multipleChoice + c.shortAnswer + c.essay) + c.otherEvaluations

/-- The exam classes as defined in the problem -/
def examClasses : List ExamClass := [
  ⟨30, 12, 0, 3, 30⟩,  -- Class A
  ⟨25, 15, 5, 2, 5⟩,   -- Class B
  ⟨35, 10, 0, 3, 5⟩,   -- Class C
  ⟨40, 11, 4, 3, 40⟩,  -- Class D
  ⟨20, 14, 5, 2, 5⟩    -- Class E
]

/-- The theorem stating that the total evaluations equal 2680 -/
theorem total_evaluations_is_2680 :
  (examClasses.map classEvaluations).sum = 2680 := by
  sorry

end NUMINAMATH_CALUDE_total_evaluations_is_2680_l3568_356876


namespace NUMINAMATH_CALUDE_total_accidents_in_four_minutes_l3568_356870

/-- Represents the number of seconds in 4 minutes -/
def total_seconds : ℕ := 4 * 60

/-- Represents the frequency of car collisions in seconds -/
def car_collision_frequency : ℕ := 3

/-- Represents the frequency of big crashes in seconds -/
def big_crash_frequency : ℕ := 7

/-- Represents the frequency of multi-vehicle pile-ups in seconds -/
def pile_up_frequency : ℕ := 15

/-- Represents the frequency of massive accidents in seconds -/
def massive_accident_frequency : ℕ := 25

/-- Calculates the number of accidents of a given type -/
def accidents_of_type (frequency : ℕ) : ℕ :=
  total_seconds / frequency

/-- Theorem stating the total number of accidents in 4 minutes -/
theorem total_accidents_in_four_minutes :
  accidents_of_type car_collision_frequency +
  accidents_of_type big_crash_frequency +
  accidents_of_type pile_up_frequency +
  accidents_of_type massive_accident_frequency = 139 := by
  sorry


end NUMINAMATH_CALUDE_total_accidents_in_four_minutes_l3568_356870


namespace NUMINAMATH_CALUDE_inequality_proof_l3568_356854

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a - d > b - c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3568_356854


namespace NUMINAMATH_CALUDE_lcm_54_198_l3568_356881

theorem lcm_54_198 : Nat.lcm 54 198 = 594 := by
  sorry

end NUMINAMATH_CALUDE_lcm_54_198_l3568_356881


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l3568_356887

theorem cyclic_sum_inequality (x y z : ℝ) (a b : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 1) :
  (∀ x y z : ℝ, x ≥ 0 → y ≥ 0 → z ≥ 0 → x + y + z = 1 → 
    x*y + y*z + z*x ≥ a*(y^2*z^2 + z^2*x^2 + x^2*y^2) + b*x*y*z) ↔ 
  (b = 9 - a ∧ 0 < a ∧ a ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l3568_356887


namespace NUMINAMATH_CALUDE_newspaper_sale_percentage_l3568_356831

/-- Represents the problem of calculating the percentage of newspapers John sells. -/
theorem newspaper_sale_percentage
  (total_newspapers : ℕ)
  (selling_price : ℚ)
  (discount_percentage : ℚ)
  (profit : ℚ)
  (h1 : total_newspapers = 500)
  (h2 : selling_price = 2)
  (h3 : discount_percentage = 75 / 100)
  (h4 : profit = 550)
  : (selling_price * (1 - discount_percentage) * total_newspapers + profit) / (selling_price * total_newspapers) = 4 / 5 :=
sorry

end NUMINAMATH_CALUDE_newspaper_sale_percentage_l3568_356831


namespace NUMINAMATH_CALUDE_todd_sum_equals_l3568_356863

/-- Represents the counting game with Todd, Tadd, and Tucker -/
structure CountingGame where
  max_count : ℕ
  todd_turn_length : ℕ → ℕ
  todd_start_positions : ℕ → ℕ

/-- Calculates the sum of numbers Todd declares in the game -/
def todd_sum (game : CountingGame) : ℕ :=
  sorry

/-- The specific game instance described in the problem -/
def specific_game : CountingGame :=
  { max_count := 5000
  , todd_turn_length := λ n => n + 1
  , todd_start_positions := λ n => sorry }

/-- Theorem stating the sum of Todd's numbers equals a specific value -/
theorem todd_sum_equals (result : ℕ) : todd_sum specific_game = result :=
  sorry

end NUMINAMATH_CALUDE_todd_sum_equals_l3568_356863


namespace NUMINAMATH_CALUDE_square_area_tripled_l3568_356852

theorem square_area_tripled (a : ℝ) (h : a > 0) :
  (a * Real.sqrt 3) ^ 2 = 3 * a ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_square_area_tripled_l3568_356852


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3568_356812

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_a1 : a 1 = 2) 
  (h_sum : a 3 + a 5 = 10) : 
  a 7 = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3568_356812


namespace NUMINAMATH_CALUDE_total_albums_is_2835_l3568_356849

/-- The total number of albums owned by six people given certain relationships between their album counts. -/
def total_albums (adele_albums : ℕ) : ℕ :=
  let bridget_albums := adele_albums - 15
  let katrina_albums := 6 * bridget_albums
  let miriam_albums := 7 * katrina_albums
  let carlos_albums := 3 * miriam_albums
  let diane_albums := 2 * katrina_albums
  adele_albums + bridget_albums + katrina_albums + miriam_albums + carlos_albums + diane_albums

/-- Theorem stating that the total number of albums is 2835 given the conditions in the problem. -/
theorem total_albums_is_2835 : total_albums 30 = 2835 := by
  sorry

end NUMINAMATH_CALUDE_total_albums_is_2835_l3568_356849


namespace NUMINAMATH_CALUDE_rooftop_steps_l3568_356838

/-- The total number of stair steps to reach the rooftop -/
def total_steps (climbed : ℕ) (remaining : ℕ) : ℕ := climbed + remaining

/-- Theorem stating that the total number of steps is 96 -/
theorem rooftop_steps : total_steps 74 22 = 96 := by
  sorry

end NUMINAMATH_CALUDE_rooftop_steps_l3568_356838


namespace NUMINAMATH_CALUDE_oranges_per_group_l3568_356878

/-- Given the total number of oranges and the number of orange groups,
    prove that the number of oranges per group is 2. -/
theorem oranges_per_group (total_oranges : ℕ) (orange_groups : ℕ) 
  (h1 : total_oranges = 356) (h2 : orange_groups = 178) :
  total_oranges / orange_groups = 2 := by
  sorry


end NUMINAMATH_CALUDE_oranges_per_group_l3568_356878


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l3568_356873

theorem quadratic_solution_sum (x y : ℝ) : 
  x + y = 5 → 2 * x * y = 5 → 
  ∃ (a b c d : ℕ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    (x = (a + b * Real.sqrt c) / d ∨ x = (a - b * Real.sqrt c) / d) ∧
    a + b + c + d = 23 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l3568_356873


namespace NUMINAMATH_CALUDE_ballsInBoxes_correct_l3568_356807

/-- The number of ways to place four different balls into four numbered boxes with one empty box -/
def ballsInBoxes : ℕ :=
  -- Define the number of ways to place the balls
  -- We don't implement the actual calculation here
  144

/-- Theorem stating that the number of ways to place the balls is correct -/
theorem ballsInBoxes_correct : ballsInBoxes = 144 := by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_ballsInBoxes_correct_l3568_356807


namespace NUMINAMATH_CALUDE_square_inequality_for_negatives_l3568_356871

theorem square_inequality_for_negatives (a b : ℝ) (h : a < b ∧ b < 0) : a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_inequality_for_negatives_l3568_356871


namespace NUMINAMATH_CALUDE_exam_comparison_l3568_356806

/-- Proves that Lyssa has 3 fewer correct answers than Precious in an exam with 75 items,
    where Lyssa answers 20% incorrectly and Precious makes 12 mistakes. -/
theorem exam_comparison (total_items : ℕ) (lyssa_incorrect_percent : ℚ) (precious_mistakes : ℕ)
  (h1 : total_items = 75)
  (h2 : lyssa_incorrect_percent = 1/5)
  (h3 : precious_mistakes = 12) :
  (total_items - (lyssa_incorrect_percent * total_items).floor) = 
  (total_items - precious_mistakes) - 3 :=
by sorry

end NUMINAMATH_CALUDE_exam_comparison_l3568_356806


namespace NUMINAMATH_CALUDE_price_change_theorem_l3568_356809

theorem price_change_theorem (initial_price : ℝ) 
  (jan_increase : ℝ) (feb_decrease : ℝ) (mar_increase : ℝ) (apr_decrease : ℝ) : 
  initial_price = 200 ∧ 
  jan_increase = 0.3 ∧ 
  feb_decrease = 0.1 ∧ 
  mar_increase = 0.2 ∧
  initial_price * (1 + jan_increase) * (1 - feb_decrease) * (1 + mar_increase) * (1 - apr_decrease) = initial_price →
  apr_decrease = 0.29 := by
  sorry

end NUMINAMATH_CALUDE_price_change_theorem_l3568_356809


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3568_356882

theorem inequality_equivalence (x : ℝ) :
  (7 * x - 2 < 3 * (x + 2) ↔ x < 2) ∧
  ((x - 1) / 3 ≥ (x - 3) / 12 + 1 ↔ x ≥ 13 / 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3568_356882


namespace NUMINAMATH_CALUDE_negation_equivalence_l3568_356819

theorem negation_equivalence (x y : ℝ) : 
  ¬(x^2 + y^2 = 0 → x = 0 ∧ y = 0) ↔ (x^2 + y^2 = 0 → x ≠ 0 ∨ y ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3568_356819


namespace NUMINAMATH_CALUDE_f_positive_iff_triangle_l3568_356885

/-- A polynomial function representing the triangle inequality condition -/
def f (x y z : ℝ) : ℝ := (x + y + z) * (-x + y + z) * (x - y + z) * (x + y - z)

/-- Predicate to check if three real numbers can form the sides of a triangle -/
def is_triangle (x y z : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y > z ∧ x + z > y ∧ y + z > x

/-- Theorem stating that f is positive iff x, y, z can form a triangle -/
theorem f_positive_iff_triangle (x y z : ℝ) :
  f x y z > 0 ↔ is_triangle (|x|) (|y|) (|z|) := by sorry

end NUMINAMATH_CALUDE_f_positive_iff_triangle_l3568_356885


namespace NUMINAMATH_CALUDE_carrie_punch_ice_amount_l3568_356814

/-- Represents the ingredients and result of Carrie's punch recipe --/
structure PunchRecipe where
  mountain_dew_cans : Nat
  mountain_dew_oz_per_can : Nat
  fruit_juice_oz : Nat
  servings : Nat
  oz_per_serving : Nat

/-- Calculates the amount of ice added to the punch --/
def ice_added (recipe : PunchRecipe) : Nat :=
  recipe.servings * recipe.oz_per_serving - 
  (recipe.mountain_dew_cans * recipe.mountain_dew_oz_per_can + recipe.fruit_juice_oz)

/-- Theorem stating that Carrie added 28 oz of ice to her punch --/
theorem carrie_punch_ice_amount : 
  ice_added { mountain_dew_cans := 6
            , mountain_dew_oz_per_can := 12
            , fruit_juice_oz := 40
            , servings := 14
            , oz_per_serving := 10 } = 28 := by
  sorry

end NUMINAMATH_CALUDE_carrie_punch_ice_amount_l3568_356814


namespace NUMINAMATH_CALUDE_triangle_side_length_l3568_356851

theorem triangle_side_length (a b c : ℝ) (A : Real) :
  a = Real.sqrt 5 →
  b = Real.sqrt 15 →
  A = 30 * Real.pi / 180 →
  c = 2 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3568_356851


namespace NUMINAMATH_CALUDE_second_car_speed_l3568_356880

/-- Proves that given the conditions of the problem, the speed of the second car is 100 km/hr -/
theorem second_car_speed (car_a_speed : ℝ) (car_a_time : ℝ) (second_car_time : ℝ) (distance_ratio : ℝ) :
  car_a_speed = 50 →
  car_a_time = 6 →
  second_car_time = 1 →
  distance_ratio = 3 →
  (car_a_speed * car_a_time) / (distance_ratio * second_car_time) = 100 := by
  sorry

end NUMINAMATH_CALUDE_second_car_speed_l3568_356880


namespace NUMINAMATH_CALUDE_floor_of_6_8_l3568_356826

theorem floor_of_6_8 : ⌊(6.8 : ℝ)⌋ = 6 := by sorry

end NUMINAMATH_CALUDE_floor_of_6_8_l3568_356826


namespace NUMINAMATH_CALUDE_A_equals_B_l3568_356847

def A : Set ℤ := {x | ∃ a b : ℤ, x = 12 * a + 8 * b}
def B : Set ℤ := {y | ∃ c d : ℤ, y = 20 * c + 16 * d}

theorem A_equals_B : A = B := by sorry

end NUMINAMATH_CALUDE_A_equals_B_l3568_356847


namespace NUMINAMATH_CALUDE_marble_fraction_after_tripling_l3568_356850

theorem marble_fraction_after_tripling (total : ℚ) (h1 : total > 0) : 
  let blue := (4/7) * total
  let green := total - blue
  let new_green := 3 * green
  let new_total := blue + new_green
  new_green / new_total = 9/13 := by
sorry

end NUMINAMATH_CALUDE_marble_fraction_after_tripling_l3568_356850


namespace NUMINAMATH_CALUDE_function_inequality_l3568_356837

theorem function_inequality (f : ℝ → ℝ) (h : Differentiable ℝ f) 
    (h1 : ∀ x, (x - 1) * (deriv f x) ≥ 0) : 
  f 0 + f 2 ≥ 2 * f 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3568_356837


namespace NUMINAMATH_CALUDE_girls_percentage_increase_l3568_356816

theorem girls_percentage_increase (initial_boys : ℕ) (final_total : ℕ) : 
  initial_boys = 15 →
  final_total = 51 →
  ∃ (initial_girls : ℕ),
    initial_girls = initial_boys + (initial_boys * 20 / 100) ∧
    final_total = initial_boys + 2 * initial_girls :=
by sorry

end NUMINAMATH_CALUDE_girls_percentage_increase_l3568_356816


namespace NUMINAMATH_CALUDE_pool_visitors_l3568_356896

theorem pool_visitors (women : ℕ) (women_students : ℕ) (men_more : ℕ) (men_nonstudents : ℕ) 
  (h1 : women = 1518)
  (h2 : women_students = 536)
  (h3 : men_more = 525)
  (h4 : men_nonstudents = 1257) :
  women_students + ((women + men_more) - men_nonstudents) = 1322 := by
  sorry

end NUMINAMATH_CALUDE_pool_visitors_l3568_356896


namespace NUMINAMATH_CALUDE_knights_round_table_l3568_356892

theorem knights_round_table (n : ℕ) 
  (h1 : ∃ (f e : ℕ), f = e ∧ f + e = n) : 
  4 ∣ n := by
sorry

end NUMINAMATH_CALUDE_knights_round_table_l3568_356892


namespace NUMINAMATH_CALUDE_smaller_pyramid_volume_l3568_356823

/-- The volume of a smaller pyramid cut from a larger right square pyramid -/
theorem smaller_pyramid_volume
  (base_edge : ℝ)
  (total_height : ℝ)
  (cut_height : ℝ)
  (h_base : base_edge = 12)
  (h_height : total_height = 18)
  (h_cut : cut_height = 6) :
  (1/3 : ℝ) * (cut_height / total_height)^2 * base_edge^2 * cut_height = 32 := by
sorry

end NUMINAMATH_CALUDE_smaller_pyramid_volume_l3568_356823


namespace NUMINAMATH_CALUDE_odd_number_difference_difference_is_98_l3568_356834

theorem odd_number_difference : ℕ → Prop :=
  fun n => ∃ (a b : ℕ), 
    (a ≤ 100 ∧ b ≤ 100) ∧  -- Numbers are in the range 1 to 100
    (Odd a ∧ Odd b) ∧      -- Both numbers are odd
    (∀ k, k ≤ 100 → Odd k → a ≤ k ∧ k ≤ b) ∧  -- a is smallest, b is largest odd number
    b - a = n              -- Their difference is n

theorem difference_is_98 : odd_number_difference 98 := by
  sorry

end NUMINAMATH_CALUDE_odd_number_difference_difference_is_98_l3568_356834


namespace NUMINAMATH_CALUDE_sum_of_squared_coefficients_l3568_356839

/-- The polynomial resulting from simplifying 3(x^3 - x^2 + 4) - 5(x^4 - 2x^3 + x - 1) -/
def simplified_polynomial (x : ℝ) : ℝ :=
  -5 * x^4 + 13 * x^3 - 3 * x^2 - 5 * x + 17

/-- The coefficients of the simplified polynomial -/
def coefficients : List ℝ := [-5, 13, -3, -5, 17]

theorem sum_of_squared_coefficients :
  (coefficients.map (λ c => c^2)).sum = 517 := by sorry

end NUMINAMATH_CALUDE_sum_of_squared_coefficients_l3568_356839


namespace NUMINAMATH_CALUDE_league_games_count_l3568_356890

/-- The number of teams in each division -/
def teams_per_division : ℕ := 9

/-- The number of times each team plays other teams in its own division -/
def intra_division_games : ℕ := 3

/-- The number of times each team plays teams in the other division -/
def inter_division_games : ℕ := 2

/-- The number of divisions in the league -/
def num_divisions : ℕ := 2

/-- The total number of games scheduled in the league -/
def total_games : ℕ :=
  (num_divisions * (teams_per_division.choose 2 * intra_division_games)) +
  (teams_per_division * teams_per_division * inter_division_games)

theorem league_games_count : total_games = 378 := by
  sorry

end NUMINAMATH_CALUDE_league_games_count_l3568_356890


namespace NUMINAMATH_CALUDE_percentage_equation_l3568_356859

theorem percentage_equation (x : ℝ) : (0.3 / 100) * x = 0.15 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equation_l3568_356859


namespace NUMINAMATH_CALUDE_stair_step_black_squares_l3568_356899

/-- Represents the number of squares added to form a row in the stair-step pattern -/
def squaresAdded (n : ℕ) : ℕ :=
  if n ≤ 1 then 0 else n - 2

/-- Calculates the total number of squares in the nth row of the stair-step pattern -/
def totalSquares (n : ℕ) : ℕ :=
  1 + (Finset.range n).sum squaresAdded

/-- Calculates the number of black squares in a row with a given total number of squares -/
def blackSquares (total : ℕ) : ℕ :=
  (total - 1) / 2

/-- Theorem: The 20th row of the stair-step pattern contains 85 black squares -/
theorem stair_step_black_squares :
  blackSquares (totalSquares 20) = 85 := by
  sorry


end NUMINAMATH_CALUDE_stair_step_black_squares_l3568_356899


namespace NUMINAMATH_CALUDE_root_difference_is_one_l3568_356883

theorem root_difference_is_one (p : ℝ) : 
  let α := (p + 1) / 2
  let β := (p - 1) / 2
  α - β = 1 ∧ 
  α^2 - p*α + (p^2 - 1)/4 = 0 ∧ 
  β^2 - p*β + (p^2 - 1)/4 = 0 ∧
  α ≥ β := by
sorry

end NUMINAMATH_CALUDE_root_difference_is_one_l3568_356883


namespace NUMINAMATH_CALUDE_parallel_vectors_fraction_l3568_356802

theorem parallel_vectors_fraction (x : ℝ) :
  let a : ℝ × ℝ := (Real.sin x, (3 : ℝ) / 2)
  let b : ℝ × ℝ := (Real.cos x, -1)
  (∃ (k : ℝ), a.1 = k * b.1 ∧ a.2 = k * b.2) →
  (2 * Real.sin x - Real.cos x) / (4 * Real.sin x + 3 * Real.cos x) = 4 / 3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_fraction_l3568_356802


namespace NUMINAMATH_CALUDE_triangle_height_l3568_356886

/-- Given a triangle with area 615 m² and a side of 123 m, 
    the perpendicular height to that side is 10 m. -/
theorem triangle_height (A : ℝ) (b : ℝ) (h : ℝ) 
  (area_eq : A = 615) 
  (base_eq : b = 123) 
  (area_formula : A = (1/2) * b * h) : h = 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_height_l3568_356886


namespace NUMINAMATH_CALUDE_marks_reading_time_marks_reading_proof_l3568_356895

theorem marks_reading_time (increase : ℕ) (target : ℕ) (days_in_week : ℕ) : ℕ :=
  let initial_daily_hours : ℕ := (target - increase) / days_in_week
  initial_daily_hours

theorem marks_reading_proof :
  marks_reading_time 4 18 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_marks_reading_time_marks_reading_proof_l3568_356895


namespace NUMINAMATH_CALUDE_angle_around_point_l3568_356875

theorem angle_around_point (x : ℝ) : x = 110 :=
  let total_angle : ℝ := 360
  let given_angle : ℝ := 140
  have h1 : x + x + given_angle = total_angle := by sorry
  sorry

end NUMINAMATH_CALUDE_angle_around_point_l3568_356875


namespace NUMINAMATH_CALUDE_jackson_points_l3568_356842

theorem jackson_points (total_points : ℕ) (num_players : ℕ) (other_players : ℕ) (avg_points : ℕ) :
  total_points = 75 →
  num_players = 8 →
  other_players = 7 →
  avg_points = 6 →
  total_points - (other_players * avg_points) = 33 :=
by sorry

end NUMINAMATH_CALUDE_jackson_points_l3568_356842


namespace NUMINAMATH_CALUDE_max_a_inequality_max_a_is_five_l3568_356829

theorem max_a_inequality (a : ℝ) : 
  (∀ x : ℝ, x^2 + |2*x - 6| ≥ a) → a ≤ 5 :=
by sorry

theorem max_a_is_five : 
  ∃ a : ℝ, (∀ x : ℝ, x^2 + |2*x - 6| ≥ a) ∧ a = 5 :=
by sorry

end NUMINAMATH_CALUDE_max_a_inequality_max_a_is_five_l3568_356829


namespace NUMINAMATH_CALUDE_unique_four_digit_number_l3568_356836

/-- 
A four-digit number is a natural number between 1000 and 9999, inclusive.
-/
def FourDigitNumber (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- 
Given a four-digit number, this function returns the three-digit number 
obtained by removing its leftmost digit.
-/
def RemoveLeftmostDigit (n : ℕ) : ℕ := n % 1000

/-- 
Theorem: 3500 is the only four-digit number N such that the three-digit number 
obtained by removing its leftmost digit is one-seventh of N.
-/
theorem unique_four_digit_number : 
  ∀ N : ℕ, FourDigitNumber N → 
    (RemoveLeftmostDigit N = N / 7 ↔ N = 3500) :=
by sorry

end NUMINAMATH_CALUDE_unique_four_digit_number_l3568_356836


namespace NUMINAMATH_CALUDE_mMobile_first_two_lines_cost_l3568_356864

/-- The cost of a mobile phone plan for a family of 5 -/
structure MobilePlan where
  firstTwoLines : ℕ  -- Cost for first two lines
  additionalLine : ℕ  -- Cost for each additional line

/-- Calculate the total cost for 5 lines -/
def totalCost (plan : MobilePlan) : ℕ :=
  plan.firstTwoLines + 3 * plan.additionalLine

theorem mMobile_first_two_lines_cost : 
  ∃ (mMobile : MobilePlan),
    mMobile.additionalLine = 14 ∧
    ∃ (tMobile : MobilePlan),
      tMobile.firstTwoLines = 50 ∧
      tMobile.additionalLine = 16 ∧
      totalCost tMobile - totalCost mMobile = 11 ∧
      mMobile.firstTwoLines = 45 := by
  sorry

end NUMINAMATH_CALUDE_mMobile_first_two_lines_cost_l3568_356864


namespace NUMINAMATH_CALUDE_simple_random_sampling_fairness_l3568_356800

/-- Represents the probability of being selected in a simple random sample -/
def SimpleSampleProb (n : ℕ) : ℚ := 1 / n

/-- Represents a group of students -/
structure StudentGroup where
  total : ℕ
  selected : ℕ
  toEliminate : ℕ

/-- Defines fairness based on equal probability of selection -/
def isFair (g : StudentGroup) : Prop :=
  ∀ (i j : ℕ), i ≤ g.selected → j ≤ g.selected →
    SimpleSampleProb g.selected = SimpleSampleProb g.selected

theorem simple_random_sampling_fairness 
  (students : StudentGroup) 
  (h1 : students.total = 102) 
  (h2 : students.selected = 20) 
  (h3 : students.toEliminate = 2) : 
  isFair students :=
sorry

end NUMINAMATH_CALUDE_simple_random_sampling_fairness_l3568_356800


namespace NUMINAMATH_CALUDE_shortest_player_height_l3568_356862

/-- Given the heights of four players, prove the height of the shortest player. -/
theorem shortest_player_height (T S P Q : ℝ)
  (h1 : T = 77.75)
  (h2 : T = S + 9.5)
  (h3 : P = S + 5)
  (h4 : Q = P - 3) :
  S = 68.25 := by
  sorry

end NUMINAMATH_CALUDE_shortest_player_height_l3568_356862


namespace NUMINAMATH_CALUDE_pi_is_real_l3568_356845

-- Define π as a real number representing the ratio of a circle's circumference to its diameter
noncomputable def π : ℝ := Real.pi

-- Theorem stating that π is a real number
theorem pi_is_real : π ∈ Set.univ := by sorry

end NUMINAMATH_CALUDE_pi_is_real_l3568_356845


namespace NUMINAMATH_CALUDE_line_parallel_perpendicular_l3568_356822

-- Define the types for lines and planes
variable (L : Type*) [LinearOrderedField L]
variable (P : Type*) [LinearOrderedField P]

-- Define the parallel and perpendicular relations
variable (parallel : L → L → Prop)
variable (perp : L → P → Prop)

-- State the theorem
theorem line_parallel_perpendicular
  (m n : L) (α : P)
  (h1 : m ≠ n)
  (h2 : parallel m n)
  (h3 : perp m α) :
  perp n α :=
sorry

end NUMINAMATH_CALUDE_line_parallel_perpendicular_l3568_356822


namespace NUMINAMATH_CALUDE_rod_length_l3568_356827

theorem rod_length (pieces : ℕ) (piece_length : ℝ) (h1 : pieces = 50) (h2 : piece_length = 0.85) :
  pieces * piece_length = 42.5 := by
  sorry

end NUMINAMATH_CALUDE_rod_length_l3568_356827


namespace NUMINAMATH_CALUDE_binary_1111_equals_15_l3568_356856

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binaryToDecimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of the number 15 -/
def binaryFifteen : List Bool := [true, true, true, true]

/-- Theorem stating that the binary representation "1111" is equal to 15 in decimal -/
theorem binary_1111_equals_15 : binaryToDecimal binaryFifteen = 15 := by
  sorry

end NUMINAMATH_CALUDE_binary_1111_equals_15_l3568_356856


namespace NUMINAMATH_CALUDE_max_value_of_sum_products_l3568_356866

theorem max_value_of_sum_products (a b c d : ℝ) :
  a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 →
  a + b + c + d = 150 →
  a * b + b * c + c * d ≤ 5625 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_sum_products_l3568_356866


namespace NUMINAMATH_CALUDE_difference_of_greatest_values_l3568_356821

def is_valid_three_digit_number (x : ℕ) : Prop :=
  100 ≤ x ∧ x < 1000

def hundreds_digit (x : ℕ) : ℕ := (x / 100) % 10
def tens_digit (x : ℕ) : ℕ := (x / 10) % 10
def units_digit (x : ℕ) : ℕ := x % 10

def satisfies_conditions (x : ℕ) : Prop :=
  let a := hundreds_digit x
  let b := tens_digit x
  let c := units_digit x
  is_valid_three_digit_number x ∧ 2 * a = b ∧ b = 4 * c ∧ a > 0

theorem difference_of_greatest_values : 
  ∃ x₁ x₂ : ℕ, satisfies_conditions x₁ ∧ satisfies_conditions x₂ ∧
  (∀ x : ℕ, satisfies_conditions x → x ≤ x₁) ∧
  (∀ x : ℕ, satisfies_conditions x → x ≠ x₁ → x ≤ x₂) ∧
  x₁ - x₂ = 241 :=
sorry

end NUMINAMATH_CALUDE_difference_of_greatest_values_l3568_356821


namespace NUMINAMATH_CALUDE_problem_solution_l3568_356820

def star (a b : ℕ) : ℕ := a^b + a*b

theorem problem_solution (a b : ℕ) (ha : a ≥ 2) (hb : b ≥ 2) (h : star a b = 40) : a + b = 7 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3568_356820


namespace NUMINAMATH_CALUDE_two_color_theorem_l3568_356868

/-- A type representing the two colors used for coloring regions -/
inductive Color
| Blue
| Red

/-- A type representing a circle in the plane -/
structure Circle where
  -- We don't need to define the internal structure of a circle for this problem

/-- A type representing a region in the plane -/
structure Region where
  -- We don't need to define the internal structure of a region for this problem

/-- A function type for coloring regions -/
def ColoringFunction := Region → Color

/-- Predicate to check if two regions are adjacent (separated by a circle arc) -/
def are_adjacent (r1 r2 : Region) : Prop := sorry

/-- Theorem stating the existence of a valid two-color coloring for n circles -/
theorem two_color_theorem (n : ℕ) (h : n ≥ 1) :
  ∃ (circles : Finset Circle) (regions : Finset Region) (coloring : ColoringFunction),
    (circles.card = n) ∧
    (∀ r1 r2 : Region, r1 ∈ regions → r2 ∈ regions → are_adjacent r1 r2 →
      coloring r1 ≠ coloring r2) :=
sorry

end NUMINAMATH_CALUDE_two_color_theorem_l3568_356868


namespace NUMINAMATH_CALUDE_negative_number_identification_l3568_356801

theorem negative_number_identification : 
  ((-1 : ℝ) < 0) ∧ (¬(0 < 0)) ∧ (¬(2 < 0)) ∧ (¬(Real.sqrt 2 < 0)) := by
  sorry

end NUMINAMATH_CALUDE_negative_number_identification_l3568_356801


namespace NUMINAMATH_CALUDE_problem_solution_l3568_356804

theorem problem_solution (a : ℝ) (h1 : a > 0) : 
  (fun x => x^2 + 4) ((fun x => x^2 - 2) a) = 12 → 
  a = Real.sqrt (2 * (Real.sqrt 2 + 1)) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3568_356804


namespace NUMINAMATH_CALUDE_cube_to_rectangular_solid_surface_area_ratio_l3568_356825

/-- The ratio of the surface area of a cube to the surface area of a rectangular solid
    with doubled length is 3/5. -/
theorem cube_to_rectangular_solid_surface_area_ratio :
  ∀ s : ℝ, s > 0 →
  (6 * s^2) / (2 * (2*s*s + 2*s*s + s*s)) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cube_to_rectangular_solid_surface_area_ratio_l3568_356825


namespace NUMINAMATH_CALUDE_parabola_vertex_distance_l3568_356853

/-- The parabola equation -/
def parabola (x c : ℝ) : ℝ := x^2 - 6*x + c - 2

/-- The vertex of the parabola -/
def vertex (c : ℝ) : ℝ × ℝ := (3, c - 11)

/-- The distance from the vertex to the x-axis -/
def distance_to_x_axis (c : ℝ) : ℝ := |c - 11|

theorem parabola_vertex_distance (c : ℝ) :
  distance_to_x_axis c = 3 → c = 8 ∨ c = 14 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_distance_l3568_356853


namespace NUMINAMATH_CALUDE_modulus_complex_l3568_356857

theorem modulus_complex (α : Real) (h : π < α ∧ α < 2*π) :
  Complex.abs (1 + Complex.cos α + Complex.I * Complex.sin α) = 2 * Real.cos (α/2) := by
  sorry

end NUMINAMATH_CALUDE_modulus_complex_l3568_356857


namespace NUMINAMATH_CALUDE_circular_track_time_theorem_l3568_356898

/-- Represents a circular track with two points -/
structure CircularTrack :=
  (total_time : ℝ)
  (time_closer_to_point : ℝ)

/-- Theorem: If a runner on a circular track is closer to one point for half the total running time,
    then the total running time is twice the time the runner is closer to that point -/
theorem circular_track_time_theorem (track : CircularTrack) 
  (h1 : track.time_closer_to_point > 0)
  (h2 : track.time_closer_to_point = track.total_time / 2) : 
  track.total_time = 2 * track.time_closer_to_point :=
sorry

end NUMINAMATH_CALUDE_circular_track_time_theorem_l3568_356898


namespace NUMINAMATH_CALUDE_zero_neither_positive_nor_negative_l3568_356840

theorem zero_neither_positive_nor_negative :
  ¬(0 > 0) ∧ ¬(0 < 0) :=
by sorry

end NUMINAMATH_CALUDE_zero_neither_positive_nor_negative_l3568_356840


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l3568_356879

theorem quadratic_inequality_condition (a : ℝ) : 
  (∃ x : ℝ, a * x^2 + 2 * x + 1 < 0) → a < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l3568_356879


namespace NUMINAMATH_CALUDE_max_wednesday_pizzas_exists_five_pizzas_wednesday_l3568_356884

/-- Represents the number of pizzas baked on each day -/
structure PizzaSchedule where
  saturday : ℕ
  sunday : ℕ
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ

/-- Checks if the pizza schedule satisfies the given conditions -/
def isValidSchedule (schedule : PizzaSchedule) : Prop :=
  let total := 50
  schedule.saturday = (3 * total) / 5 ∧
  schedule.sunday = (3 * (total - schedule.saturday)) / 5 ∧
  schedule.monday < schedule.sunday ∧
  schedule.tuesday < schedule.monday ∧
  schedule.wednesday < schedule.tuesday ∧
  schedule.saturday + schedule.sunday + schedule.monday + schedule.tuesday + schedule.wednesday = total

/-- Theorem stating the maximum number of pizzas that could be baked on Wednesday -/
theorem max_wednesday_pizzas :
  ∀ (schedule : PizzaSchedule), isValidSchedule schedule → schedule.wednesday ≤ 5 := by
  sorry

/-- Theorem stating that there exists a valid schedule with 5 pizzas on Wednesday -/
theorem exists_five_pizzas_wednesday :
  ∃ (schedule : PizzaSchedule), isValidSchedule schedule ∧ schedule.wednesday = 5 := by
  sorry

end NUMINAMATH_CALUDE_max_wednesday_pizzas_exists_five_pizzas_wednesday_l3568_356884


namespace NUMINAMATH_CALUDE_exact_five_green_probability_l3568_356865

def total_marbles : ℕ := 12
def green_marbles : ℕ := 8
def purple_marbles : ℕ := 4
def total_draws : ℕ := 8
def green_draws : ℕ := 5

def prob_green : ℚ := green_marbles / total_marbles
def prob_purple : ℚ := purple_marbles / total_marbles

def binomial_coefficient (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem exact_five_green_probability :
  (binomial_coefficient total_draws green_draws : ℚ) * 
  (prob_green ^ green_draws) * 
  (prob_purple ^ (total_draws - green_draws)) =
  56 * (2/3)^5 * (1/3)^3 := by sorry

end NUMINAMATH_CALUDE_exact_five_green_probability_l3568_356865


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l3568_356828

/-- A point in a 2D plane represented by its x and y coordinates. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines symmetry with respect to the origin for two points. -/
def symmetricToOrigin (a b : Point) : Prop :=
  b.x = -a.x ∧ b.y = -a.y

/-- Theorem stating that if point A(5, -1) is symmetric to point B with respect to the origin,
    then the coordinates of point B are (-5, 1). -/
theorem symmetric_point_coordinates :
  let a : Point := ⟨5, -1⟩
  let b : Point := ⟨-5, 1⟩
  symmetricToOrigin a b :=
by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l3568_356828


namespace NUMINAMATH_CALUDE_range_of_a_l3568_356813

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | 2 * a ≤ x ∧ x ≤ a^2 + 1}
def B (a : ℝ) : Set ℝ := {x | x^2 - 3*(a+1)*x + 2*(3*a+1) ≤ 0}

-- Define the condition that p is sufficient for q
def p_sufficient_for_q (a : ℝ) : Prop := A a ⊆ B a

-- State the theorem
theorem range_of_a : 
  ∀ a : ℝ, p_sufficient_for_q a ↔ (1 ≤ a ∧ a ≤ 3) ∨ a = -1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3568_356813


namespace NUMINAMATH_CALUDE_donna_bananas_l3568_356848

def total_bananas : ℕ := 200
def lydia_bananas : ℕ := 60

theorem donna_bananas : 
  ∀ (dawn_bananas : ℕ) (donna_bananas : ℕ),
  dawn_bananas = lydia_bananas + 40 →
  total_bananas = dawn_bananas + lydia_bananas + donna_bananas →
  donna_bananas = 40 := by
sorry

end NUMINAMATH_CALUDE_donna_bananas_l3568_356848
