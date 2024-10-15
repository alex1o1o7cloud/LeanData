import Mathlib

namespace NUMINAMATH_CALUDE_rectangle_area_l135_13557

/-- Given a rectangle ABCD divided into four identical squares with side length s,
    prove that its area is 2500 square centimeters when three of its sides total 100 cm. -/
theorem rectangle_area (s : ℝ) : 
  s > 0 →                            -- s is positive (implied by the context)
  4 * s = 100 →                      -- three sides total 100 cm
  (2 * s) * (2 * s) = 2500 :=        -- area of ABCD is 2500 sq cm
by
  sorry

#check rectangle_area

end NUMINAMATH_CALUDE_rectangle_area_l135_13557


namespace NUMINAMATH_CALUDE_intersection_of_lines_l135_13540

theorem intersection_of_lines :
  ∃! (x y : ℚ), (8 * x - 5 * y = 40) ∧ (6 * x + 2 * y = 14) ∧ 
  (x = 75 / 23) ∧ (y = 161 / 23) := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l135_13540


namespace NUMINAMATH_CALUDE_jaydon_rachel_ratio_l135_13545

-- Define the number of cans for each person
def mark_cans : ℕ := 100
def jaydon_cans : ℕ := 25
def rachel_cans : ℕ := 10

-- Define the total number of cans
def total_cans : ℕ := 135

-- Define the conditions
axiom mark_jaydon_relation : mark_cans = 4 * jaydon_cans
axiom total_cans_sum : total_cans = mark_cans + jaydon_cans + rachel_cans
axiom jaydon_rachel_relation : ∃ k : ℕ, jaydon_cans = k * rachel_cans + 5

-- Theorem to prove
theorem jaydon_rachel_ratio : 
  (jaydon_cans : ℚ) / rachel_cans = 5 / 2 := by sorry

end NUMINAMATH_CALUDE_jaydon_rachel_ratio_l135_13545


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l135_13598

theorem necessary_not_sufficient (a b : ℝ) :
  (∀ a b : ℝ, a ≤ 1 ∧ b ≤ 1 → a + b ≤ 2) ∧
  (∃ a b : ℝ, a + b ≤ 2 ∧ ¬(a ≤ 1 ∧ b ≤ 1)) := by
sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l135_13598


namespace NUMINAMATH_CALUDE_mundane_goblet_points_difference_l135_13564

def round_robin_tournament (n : ℕ) := n * (n - 1) / 2

theorem mundane_goblet_points_difference :
  let num_teams : ℕ := 6
  let num_matches := round_robin_tournament num_teams
  let max_points := num_matches * 3
  let min_points := num_matches * 2
  max_points - min_points = 15 := by
  sorry

end NUMINAMATH_CALUDE_mundane_goblet_points_difference_l135_13564


namespace NUMINAMATH_CALUDE_no_solution_floor_equation_l135_13542

theorem no_solution_floor_equation :
  ¬ ∃ (x : ℤ), (⌊x⌋ : ℤ) + ⌊2*x⌋ + ⌊4*x⌋ + ⌊8*x⌋ + ⌊16*x⌋ + ⌊32*x⌋ = 12345 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_floor_equation_l135_13542


namespace NUMINAMATH_CALUDE_integral_ratio_theorem_l135_13519

theorem integral_ratio_theorem (a b : ℝ) (h : a < b) :
  let f (x : ℝ) := (1 / 20 + 3 / 10) * x^2
  let g (x : ℝ) := x^2
  (∫ x in a..b, f x) / (∫ x in a..b, g x) = 35 / 100 := by
  sorry

end NUMINAMATH_CALUDE_integral_ratio_theorem_l135_13519


namespace NUMINAMATH_CALUDE_painting_time_equation_l135_13593

theorem painting_time_equation (doug_time dave_time t : ℝ) :
  doug_time = 6 →
  dave_time = 8 →
  (1 / doug_time + 1 / dave_time) * (t - 1) = 1 :=
by sorry

end NUMINAMATH_CALUDE_painting_time_equation_l135_13593


namespace NUMINAMATH_CALUDE_cosine_vertical_shift_l135_13515

theorem cosine_vertical_shift 
  (a b c d : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_oscillation : ∀ x : ℝ, 0 ≤ a * Real.cos (b * x + c) + d ∧ a * Real.cos (b * x + c) + d ≤ 4) : 
  d = 2 := by
sorry

end NUMINAMATH_CALUDE_cosine_vertical_shift_l135_13515


namespace NUMINAMATH_CALUDE_hyper_box_side_sum_l135_13526

/-- The sum of side lengths of a four-dimensional rectangular hyper-box with given face volumes -/
theorem hyper_box_side_sum (W X Y Z : ℝ) 
  (h1 : W * X * Y = 60)
  (h2 : W * X * Z = 80)
  (h3 : W * Y * Z = 120)
  (h4 : X * Y * Z = 60) :
  W + X + Y + Z = 318.5 := by
  sorry

end NUMINAMATH_CALUDE_hyper_box_side_sum_l135_13526


namespace NUMINAMATH_CALUDE_line_parallel_perpendicular_implies_planes_perpendicular_l135_13554

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_parallel_perpendicular_implies_planes_perpendicular
  (l : Line) (α β : Plane) :
  parallel l α → perpendicular l β → plane_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_line_parallel_perpendicular_implies_planes_perpendicular_l135_13554


namespace NUMINAMATH_CALUDE_candy_probability_difference_l135_13525

theorem candy_probability_difference : 
  let total_candies : ℕ := 2004
  let banana_candies : ℕ := 1002
  let apple_candies : ℕ := 1002
  let different_flavor_prob : ℚ := banana_candies * apple_candies / (total_candies * (total_candies - 1))
  let same_flavor_prob : ℚ := (banana_candies * (banana_candies - 1) + apple_candies * (apple_candies - 1)) / (total_candies * (total_candies - 1))
  different_flavor_prob - same_flavor_prob = 1 / 2003 := by
sorry

end NUMINAMATH_CALUDE_candy_probability_difference_l135_13525


namespace NUMINAMATH_CALUDE_product_sum_theorem_l135_13539

theorem product_sum_theorem (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 241) 
  (h2 : a + b + c = 21) : 
  a*b + b*c + a*c = 100 := by
sorry

end NUMINAMATH_CALUDE_product_sum_theorem_l135_13539


namespace NUMINAMATH_CALUDE_arithmetic_sequence_with_geometric_mean_l135_13546

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_with_geometric_mean
  (a : ℕ → ℝ) (d : ℝ) (h1 : a 1 = 1) (h2 : d ≠ 0)
  (h3 : arithmetic_sequence a d)
  (h4 : a 2 ^ 2 = a 1 * a 4) :
  d = 1 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_with_geometric_mean_l135_13546


namespace NUMINAMATH_CALUDE_ice_cream_theorem_l135_13590

/-- The number of ways to distribute n indistinguishable objects into k distinct categories -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of ice cream flavor combinations -/
def ice_cream_combinations : ℕ := distribute 4 4

theorem ice_cream_theorem : ice_cream_combinations = 35 := by sorry

end NUMINAMATH_CALUDE_ice_cream_theorem_l135_13590


namespace NUMINAMATH_CALUDE_same_perimeter_l135_13596

-- Define the rectangle
def rectangle_length : ℝ := 10
def rectangle_width : ℝ := 8

-- Define the square side length
def square_side : ℝ := 9

-- Define perimeter functions
def rectangle_perimeter (l w : ℝ) : ℝ := 2 * (l + w)
def square_perimeter (s : ℝ) : ℝ := 4 * s

-- Theorem statement
theorem same_perimeter :
  rectangle_perimeter rectangle_length rectangle_width = square_perimeter square_side :=
by sorry

end NUMINAMATH_CALUDE_same_perimeter_l135_13596


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l135_13531

/-- A sequence of 8 positive real numbers -/
def Sequence := Fin 8 → ℝ

/-- Predicate to check if a sequence is positive -/
def is_positive (s : Sequence) : Prop :=
  ∀ i, s i > 0

/-- Predicate to check if a sequence is geometric -/
def is_geometric (s : Sequence) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ i : Fin 7, s (i + 1) = q * s i

theorem sufficient_but_not_necessary (s : Sequence) 
  (h_pos : is_positive s) :
  (s 0 + s 7 < s 3 + s 4 → ¬is_geometric s) ∧
  ∃ s' : Sequence, is_positive s' ∧ ¬is_geometric s' ∧ s' 0 + s' 7 ≥ s' 3 + s' 4 :=
sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l135_13531


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_l135_13599

theorem rectangle_area_diagonal (l w d : ℝ) (h1 : l / w = 5 / 2) (h2 : l^2 + w^2 = d^2) :
  ∃ k : ℝ, l * w = k * d^2 ∧ k = 10 / 29 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_l135_13599


namespace NUMINAMATH_CALUDE_vector_addition_result_l135_13538

theorem vector_addition_result (a b : ℝ × ℝ) :
  a = (2, 1) → b = (1, 5) → 2 • a + b = (5, 7) := by
  sorry

end NUMINAMATH_CALUDE_vector_addition_result_l135_13538


namespace NUMINAMATH_CALUDE_madan_age_is_five_l135_13510

-- Define the ages as natural numbers
def arun_age : ℕ := 60

-- Define Gokul's age as a function of Arun's age
def gokul_age (a : ℕ) : ℕ := (a - 6) / 18

-- Define Madan's age as a function of Gokul's age
def madan_age (g : ℕ) : ℕ := g + 2

-- Theorem to prove
theorem madan_age_is_five :
  madan_age (gokul_age arun_age) = 5 := by
  sorry

end NUMINAMATH_CALUDE_madan_age_is_five_l135_13510


namespace NUMINAMATH_CALUDE_inequality_order_l135_13520

theorem inequality_order (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (2 * a * b) / (a + b) ≤ Real.sqrt (a * b) ∧ Real.sqrt (a * b) ≤ (a + b) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_order_l135_13520


namespace NUMINAMATH_CALUDE_triangle_inequality_l135_13585

theorem triangle_inequality (a b c : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_sum : a + b + c = 2) : 
  a^2 + b^2 + c^2 + 2*a*b*c < 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l135_13585


namespace NUMINAMATH_CALUDE_next_divisible_by_sum_of_digits_l135_13552

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Check if a number is divisible by the sum of its digits -/
def isDivisibleBySumOfDigits (n : ℕ) : Prop :=
  n % sumOfDigits n = 0

/-- The next number after 1232 that is divisible by the sum of its digits -/
theorem next_divisible_by_sum_of_digits :
  ∃ (n : ℕ), n > 1232 ∧
    isDivisibleBySumOfDigits n ∧
    ∀ (m : ℕ), 1232 < m ∧ m < n → ¬isDivisibleBySumOfDigits m :=
by sorry

end NUMINAMATH_CALUDE_next_divisible_by_sum_of_digits_l135_13552


namespace NUMINAMATH_CALUDE_smallest_number_of_cubes_for_given_box_l135_13529

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  depth : ℕ

/-- Calculates the smallest number of cubes of different sizes needed to fill a box -/
def smallestNumberOfCubes (box : BoxDimensions) : ℕ :=
  sorry

/-- The theorem stating the smallest number of cubes needed for the given box dimensions -/
theorem smallest_number_of_cubes_for_given_box :
  let box : BoxDimensions := { length := 98, width := 77, depth := 35 }
  smallestNumberOfCubes box = 770 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_of_cubes_for_given_box_l135_13529


namespace NUMINAMATH_CALUDE_train_catch_up_time_l135_13597

/-- The problem of finding the time difference between two trains --/
theorem train_catch_up_time (goods_speed express_speed catch_up_time : ℝ) 
  (h1 : goods_speed = 36)
  (h2 : express_speed = 90)
  (h3 : catch_up_time = 4) :
  ∃ t : ℝ, t > 0 ∧ goods_speed * (t + catch_up_time) = express_speed * catch_up_time ∧ t = 6 := by
  sorry


end NUMINAMATH_CALUDE_train_catch_up_time_l135_13597


namespace NUMINAMATH_CALUDE_symmetry_x_axis_l135_13573

/-- Given two points P and Q in the Cartesian coordinate system,
    prove that if P is symmetric to Q with respect to the x-axis,
    then the sum of their x-coordinates minus 3 and the negation of Q's y-coordinate minus 1
    is equal to 3. -/
theorem symmetry_x_axis (a b : ℝ) :
  let P : ℝ × ℝ := (a - 3, 1)
  let Q : ℝ × ℝ := (2, b + 1)
  (P.1 = Q.1) →  -- x-coordinates are equal
  (P.2 = -Q.2) → -- y-coordinates are opposite
  a + b = 3 := by
sorry

end NUMINAMATH_CALUDE_symmetry_x_axis_l135_13573


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l135_13537

theorem fraction_equation_solution :
  ∃! x : ℚ, (x + 2) / (x - 3) = (x - 4) / (x + 5) :=
by
  -- The unique solution is x = 1/7
  use 1/7
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l135_13537


namespace NUMINAMATH_CALUDE_family_trip_arrangements_l135_13579

theorem family_trip_arrangements (n : Nat) (k : Nat) : 
  n = 4 ∧ k = 3 → k^n = 81 := by
  sorry

end NUMINAMATH_CALUDE_family_trip_arrangements_l135_13579


namespace NUMINAMATH_CALUDE_aluminium_count_l135_13502

/-- The number of Aluminium atoms in the compound -/
def n : ℕ := sorry

/-- Atomic weight of Aluminium in g/mol -/
def Al_weight : ℝ := 26.98

/-- Atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 16.00

/-- Atomic weight of Hydrogen in g/mol -/
def H_weight : ℝ := 1.01

/-- Molecular weight of the compound in g/mol -/
def compound_weight : ℝ := 78

/-- The number of Oxygen atoms in the compound -/
def O_count : ℕ := 3

/-- The number of Hydrogen atoms in the compound -/
def H_count : ℕ := 3

/-- Theorem stating that the number of Aluminium atoms in the compound is 1 -/
theorem aluminium_count : n = 1 := by sorry

end NUMINAMATH_CALUDE_aluminium_count_l135_13502


namespace NUMINAMATH_CALUDE_percentage_commutation_l135_13587

theorem percentage_commutation (n : ℝ) (h : 0.3 * (0.4 * n) = 24) : 0.4 * (0.3 * n) = 24 := by
  sorry

end NUMINAMATH_CALUDE_percentage_commutation_l135_13587


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l135_13543

theorem max_sum_of_factors (heart club : ℕ) : 
  heart * club = 42 → (∀ x y : ℕ, x * y = 42 → x + y ≤ heart + club) → heart + club = 43 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l135_13543


namespace NUMINAMATH_CALUDE_expected_rolls_in_year_l135_13595

/-- Represents the possible outcomes of rolling an 8-sided die -/
inductive DieOutcome
  | Prime
  | Composite
  | OddNonPrime
  | Reroll

/-- The probability of each outcome when rolling a fair 8-sided die -/
def outcomeProb (outcome : DieOutcome) : ℚ :=
  match outcome with
  | DieOutcome.Prime => 1/2
  | DieOutcome.Composite => 1/4
  | DieOutcome.OddNonPrime => 1/8
  | DieOutcome.Reroll => 1/8

/-- The expected number of rolls on a single day -/
noncomputable def expectedRollsPerDay : ℝ :=
  1

/-- The number of days in a non-leap year -/
def daysInNonLeapYear : ℕ := 365

/-- Theorem: The expected number of die rolls in a non-leap year
    is equal to the number of days in the year -/
theorem expected_rolls_in_year :
  (expectedRollsPerDay * daysInNonLeapYear : ℝ) = daysInNonLeapYear := by
  sorry

end NUMINAMATH_CALUDE_expected_rolls_in_year_l135_13595


namespace NUMINAMATH_CALUDE_algebraic_sum_equals_one_l135_13592

theorem algebraic_sum_equals_one (a b c x : ℝ) 
  (ha : a + x^2 = 2006)
  (hb : b + x^2 = 2007)
  (hc : c + x^2 = 2008)
  (habc : a * b * c = 3) :
  a / (b * c) + b / (c * a) + c / (a * b) - 1 / a - 1 / b - 1 / c = 1 :=
by sorry

end NUMINAMATH_CALUDE_algebraic_sum_equals_one_l135_13592


namespace NUMINAMATH_CALUDE_solution_problem_l135_13514

theorem solution_problem (a₁ a₂ a₃ a₄ a₅ b : ℤ) 
  (h_distinct : a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₁ ≠ a₄ ∧ a₁ ≠ a₅ ∧ 
                a₂ ≠ a₃ ∧ a₂ ≠ a₄ ∧ a₂ ≠ a₅ ∧ 
                a₃ ≠ a₄ ∧ a₃ ≠ a₅ ∧ 
                a₄ ≠ a₅)
  (h_sum : a₁ + a₂ + a₃ + a₄ + a₅ = 9)
  (h_root : (b - a₁) * (b - a₂) * (b - a₃) * (b - a₄) * (b - a₅) = 2009) :
  b = 10 := by
sorry

end NUMINAMATH_CALUDE_solution_problem_l135_13514


namespace NUMINAMATH_CALUDE_factorization_problems_l135_13550

variable (x y : ℝ)

theorem factorization_problems :
  (x^2 + 3*x = x*(x + 3)) ∧ (x^2 - 2*x*y + y^2 = (x - y)^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problems_l135_13550


namespace NUMINAMATH_CALUDE_inequality_proof_l135_13521

theorem inequality_proof (a b c : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0) 
  (sum_condition : a + b + c = 2) : 
  (1 / (1 + a * b) + 1 / (1 + b * c) + 1 / (1 + c * a) ≥ 27 / 13) ∧ 
  ((1 / (1 + a * b) + 1 / (1 + b * c) + 1 / (1 + c * a) = 27 / 13) ↔ 
   (a = 2/3 ∧ b = 2/3 ∧ c = 2/3)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l135_13521


namespace NUMINAMATH_CALUDE_equation_is_ellipse_l135_13551

def equation (x y : ℝ) : Prop :=
  x^2 + 2*y^2 - 6*x - 8*y + 9 = 0

def is_ellipse (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (h k a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
    ∀ (x y : ℝ), f x y ↔ (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1

theorem equation_is_ellipse : is_ellipse equation := by
  sorry

end NUMINAMATH_CALUDE_equation_is_ellipse_l135_13551


namespace NUMINAMATH_CALUDE_circle_reflection_translation_l135_13553

def reflect_across_y_axis (x y : ℝ) : ℝ × ℝ := (-x, y)

def translate_up (point : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (point.1, point.2 + units)

theorem circle_reflection_translation (center : ℝ × ℝ) :
  center = (3, -4) →
  (translate_up (reflect_across_y_axis center.1 center.2) 5) = (-3, 1) := by
  sorry

end NUMINAMATH_CALUDE_circle_reflection_translation_l135_13553


namespace NUMINAMATH_CALUDE_remainder_790123_div_15_l135_13583

theorem remainder_790123_div_15 : 790123 % 15 = 13 := by
  sorry

end NUMINAMATH_CALUDE_remainder_790123_div_15_l135_13583


namespace NUMINAMATH_CALUDE_mr_green_potato_yield_l135_13509

/-- Represents the dimensions of a rectangular garden in steps -/
structure GardenDimensions where
  length : ℕ
  width : ℕ

/-- Calculates the expected potato yield from a rectangular garden -/
def expected_potato_yield (garden : GardenDimensions) (step_length : ℝ) (yield_per_sqft : ℝ) : ℝ :=
  (garden.length : ℝ) * step_length * (garden.width : ℝ) * step_length * yield_per_sqft

/-- Theorem: The expected potato yield from Mr. Green's garden is 2109.375 pounds -/
theorem mr_green_potato_yield :
  let garden := GardenDimensions.mk 18 25
  let step_length := 2.5
  let yield_per_sqft := 0.75
  expected_potato_yield garden step_length yield_per_sqft = 2109.375 := by
  sorry


end NUMINAMATH_CALUDE_mr_green_potato_yield_l135_13509


namespace NUMINAMATH_CALUDE_correct_parking_methods_l135_13562

/-- Represents the number of consecutive parking spaces -/
def total_spaces : ℕ := 7

/-- Represents the number of cars to be parked -/
def cars_to_park : ℕ := 3

/-- Represents the number of consecutive empty spaces required -/
def required_empty_spaces : ℕ := 4

/-- Calculates the number of different parking methods -/
def parking_methods : ℕ := 24

/-- Theorem stating that the number of parking methods is correct -/
theorem correct_parking_methods :
  ∀ (total : ℕ) (cars : ℕ) (empty : ℕ),
    total = total_spaces →
    cars = cars_to_park →
    empty = required_empty_spaces →
    total - cars = empty →
    parking_methods = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_correct_parking_methods_l135_13562


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l135_13556

theorem product_of_three_numbers 
  (x y z : ℝ) 
  (sum_eq : x + y = 18) 
  (sum_squares_eq : x^2 + y^2 = 220) 
  (diff_eq : z = x - y) : 
  x * y * z = 104 * Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l135_13556


namespace NUMINAMATH_CALUDE_sports_competition_team_sizes_l135_13591

theorem sports_competition_team_sizes :
  ∀ (boys girls : ℕ),
  (boys + 48 : ℚ) / 6 + (girls + 50 : ℚ) / 7 = 48 - (boys : ℚ) / 6 + 50 - (girls : ℚ) / 7 →
  boys - 48 = (girls - 50) / 2 →
  boys = 72 ∧ girls = 98 :=
by
  sorry

end NUMINAMATH_CALUDE_sports_competition_team_sizes_l135_13591


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l135_13503

theorem cyclic_sum_inequality (x y z : ℝ) : 
  let a := x + y + z
  ((a - x)^4 + (a - y)^4 + (a - z)^4) + 
  2 * (x^3*y + x^3*z + y^3*x + y^3*z + z^3*x + z^3*y) + 
  4 * (x^2*y^2 + y^2*z^2 + z^2*x^2) + 
  8 * x*y*z*a ≥ 
  ((a - x)^2*(a^2 - x^2) + (a - y)^2*(a^2 - y^2) + (a - z)^2*(a^2 - z^2)) := by
sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l135_13503


namespace NUMINAMATH_CALUDE_extremum_point_implies_a_and_minimum_value_l135_13581

/-- The function f(x) = (x^2 + ax - 1)e^(x-1) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 + a*x - 1) * Real.exp (x - 1)

/-- The derivative of f(x) with respect to x -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := 
  ((2*x + a) + (x^2 + a*x - 1)) * Real.exp (x - 1)

theorem extremum_point_implies_a_and_minimum_value 
  (a : ℝ) 
  (h1 : f_derivative a (-2) = 0) :
  a = -1 ∧ ∀ x, f (-1) x ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_extremum_point_implies_a_and_minimum_value_l135_13581


namespace NUMINAMATH_CALUDE_consecutive_numbers_theorem_l135_13506

theorem consecutive_numbers_theorem 
  (a b c d e f g : ℤ) 
  (consecutive : b = a + 1 ∧ c = a + 2 ∧ d = a + 3 ∧ e = a + 4 ∧ f = a + 5 ∧ g = a + 6)
  (average_9 : (a + b + c + d + e + f + g) / 7 = 9)
  (a_half_of_g : 2 * a = g) : 
  ∃ (n : ℕ), n = 7 ∧ g - a + 1 = n :=
by sorry

end NUMINAMATH_CALUDE_consecutive_numbers_theorem_l135_13506


namespace NUMINAMATH_CALUDE_C_symmetric_origin_C_area_greater_than_pi_l135_13517

-- Define the curve C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^4 + p.2^2 = 1}

-- Symmetry with respect to the origin
theorem C_symmetric_origin : ∀ (x y : ℝ), (x, y) ∈ C ↔ (-x, -y) ∈ C := by sorry

-- Area enclosed by C is greater than π
theorem C_area_greater_than_pi : ∃ (A : ℝ), A > π ∧ (∀ (x y : ℝ), (x, y) ∈ C → x^2 + y^2 ≤ A) := by sorry

end NUMINAMATH_CALUDE_C_symmetric_origin_C_area_greater_than_pi_l135_13517


namespace NUMINAMATH_CALUDE_movie_production_cost_l135_13500

def opening_weekend_revenue : ℝ := 120000000
def total_revenue_multiplier : ℝ := 3.5
def production_company_share : ℝ := 0.60
def profit : ℝ := 192000000

theorem movie_production_cost :
  let total_revenue := opening_weekend_revenue * total_revenue_multiplier
  let production_company_revenue := total_revenue * production_company_share
  let production_cost := production_company_revenue - profit
  production_cost = 60000000 := by sorry

end NUMINAMATH_CALUDE_movie_production_cost_l135_13500


namespace NUMINAMATH_CALUDE_unique_solution_condition_l135_13572

/-- The equation has exactly one real solution if and only if b < -4 -/
theorem unique_solution_condition (b : ℝ) : 
  (∃! x : ℝ, x^3 - b*x^2 - 4*b*x + b^2 - 4 = 0) ↔ b < -4 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l135_13572


namespace NUMINAMATH_CALUDE_fraction_simplification_l135_13534

theorem fraction_simplification (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 3) :
  (x - 3) / (2 * x * (x - 3)) = 1 / (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l135_13534


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l135_13567

/-- An arithmetic sequence with first term 5 and the sum of the 6th and 8th terms equal to 58 has a common difference of 4. -/
theorem arithmetic_sequence_common_difference : ∀ (a : ℕ → ℝ),
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  a 1 = 5 →
  a 6 + a 8 = 58 →
  a 2 - a 1 = 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l135_13567


namespace NUMINAMATH_CALUDE_independent_events_probability_l135_13558

theorem independent_events_probability (a b : Set ℝ) (p : Set ℝ → ℝ) 
  (h1 : p a = 4/5)
  (h2 : p b = 2/5)
  (h3 : p (a ∩ b) = 0.32)
  (h4 : p (a ∩ b) = p a * p b) : 
  p b = 2/5 := by
sorry

end NUMINAMATH_CALUDE_independent_events_probability_l135_13558


namespace NUMINAMATH_CALUDE_special_sequence_1000th_term_l135_13518

/-- A sequence satisfying the given conditions -/
def SpecialSequence (a : ℕ → ℕ) : Prop :=
  a 1 = 2007 ∧ 
  a 2 = 2008 ∧ 
  ∀ n : ℕ, n ≥ 1 → a n + a (n + 1) + a (n + 2) = n

/-- The 1000th term of the special sequence is 2340 -/
theorem special_sequence_1000th_term (a : ℕ → ℕ) (h : SpecialSequence a) : 
  a 1000 = 2340 := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_1000th_term_l135_13518


namespace NUMINAMATH_CALUDE_find_number_l135_13580

theorem find_number (n : ℝ) : (0.47 * 1442 - 0.36 * n) + 63 = 3 → n = 2049.28 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l135_13580


namespace NUMINAMATH_CALUDE_f_is_odd_iff_a_eq_one_l135_13511

/-- A function f is odd if f(-x) = -f(x) for all x in its domain. -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = x(x-1)(x+a) -/
def f (a : ℝ) (x : ℝ) : ℝ := x * (x - 1) * (x + a)

/-- Theorem: f(x) = x(x-1)(x+a) is an odd function if and only if a = 1 -/
theorem f_is_odd_iff_a_eq_one (a : ℝ) : IsOdd (f a) ↔ a = 1 := by
  sorry


end NUMINAMATH_CALUDE_f_is_odd_iff_a_eq_one_l135_13511


namespace NUMINAMATH_CALUDE_sphere_volume_for_cube_surface_l135_13548

theorem sphere_volume_for_cube_surface (cube_side : ℝ) (L : ℝ) : 
  cube_side = 3 →
  (4 / 3 * π * (((6 * cube_side^2) / (4 * π))^(3/2))) = L * Real.sqrt 15 / Real.sqrt π →
  L = 84 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_for_cube_surface_l135_13548


namespace NUMINAMATH_CALUDE_lee_science_class_l135_13522

theorem lee_science_class (total : ℕ) (girls_ratio : ℕ) (boys_ratio : ℕ) : 
  total = 56 → girls_ratio = 4 → boys_ratio = 3 → 
  (girls_ratio + boys_ratio) * (total / (girls_ratio + boys_ratio)) * boys_ratio / girls_ratio = 24 := by
sorry

end NUMINAMATH_CALUDE_lee_science_class_l135_13522


namespace NUMINAMATH_CALUDE_sum_is_composite_l135_13586

theorem sum_is_composite (a b : ℤ) (h : 56 * a = 65 * b) : 
  ∃ (x y : ℤ), x > 1 ∧ y > 1 ∧ a + b = x * y := by
sorry

end NUMINAMATH_CALUDE_sum_is_composite_l135_13586


namespace NUMINAMATH_CALUDE_convex_figure_integer_points_l135_13582

/-- A convex figure in the plane -/
structure ConvexFigure where
  -- We don't need to define the structure fully, just declare it
  dummy : Unit

/-- The area of a convex figure -/
noncomputable def area (φ : ConvexFigure) : ℝ := sorry

/-- The semiperimeter of a convex figure -/
noncomputable def semiperimeter (φ : ConvexFigure) : ℝ := sorry

/-- The number of integer points contained in a convex figure -/
noncomputable def integerPoints (φ : ConvexFigure) : ℕ := sorry

/-- 
If the area of a convex figure is greater than n times its semiperimeter,
then it contains at least n integer points.
-/
theorem convex_figure_integer_points (φ : ConvexFigure) (n : ℕ) :
  area φ > n • (semiperimeter φ) → integerPoints φ ≥ n := by sorry

end NUMINAMATH_CALUDE_convex_figure_integer_points_l135_13582


namespace NUMINAMATH_CALUDE_max_value_theorem_l135_13576

theorem max_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  ∃ (max : ℝ), max = -9/2 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → x + y = 1 → -1/(2*x) - 2/y ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l135_13576


namespace NUMINAMATH_CALUDE_share_of_b_l135_13527

theorem share_of_b (a b c : ℕ) : 
  a = 3 * b → 
  b = c + 25 → 
  a + b + c = 645 → 
  b = 134 := by
sorry

end NUMINAMATH_CALUDE_share_of_b_l135_13527


namespace NUMINAMATH_CALUDE_product_equality_l135_13577

theorem product_equality (a : ℝ) (h : a ≠ 0 ∧ a ≠ 2 ∧ a ≠ -2) :
  (a^2 + 2*a + 4 + 8/a + 16/a^2 + 64/((a-2)*a^2)) *
  (a^2 - 2*a + 4 - 8/a + 16/a^2 - 64/((a+2)*a^2))
  =
  (a^2 + 2*a + 4 + 8/a + 16/a^2) *
  (a^2 - 2*a + 4 - 8/a + 16/a^2) :=
by sorry

end NUMINAMATH_CALUDE_product_equality_l135_13577


namespace NUMINAMATH_CALUDE_quadratic_not_equal_linear_l135_13594

theorem quadratic_not_equal_linear : ¬∃ (a b c A B : ℝ), a ≠ 0 ∧ ∀ x, a * x^2 + b * x + c = A * x + B := by
  sorry

end NUMINAMATH_CALUDE_quadratic_not_equal_linear_l135_13594


namespace NUMINAMATH_CALUDE_negation_equivalence_l135_13536

theorem negation_equivalence (m : ℝ) : 
  (¬ ∃ x : ℤ, x^2 + 2*x + m ≤ 0) ↔ (∀ x : ℤ, x^2 + 2*x + m > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l135_13536


namespace NUMINAMATH_CALUDE_x_intercept_of_line_l135_13507

/-- The x-intercept of the line 4x + 7y = 28 is the point (7, 0) -/
theorem x_intercept_of_line (x y : ℝ) :
  (4 * x + 7 * y = 28) → (x = 7 ∧ y = 0 → 4 * x + 7 * y = 28) := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_of_line_l135_13507


namespace NUMINAMATH_CALUDE_julio_mocktail_lime_juice_l135_13544

/-- Proves that Julio uses 1 tablespoon of lime juice per mocktail -/
theorem julio_mocktail_lime_juice :
  -- Define the problem parameters
  let days : ℕ := 30
  let mocktails_per_day : ℕ := 1
  let lime_juice_per_lime : ℚ := 2
  let limes_per_dollar : ℚ := 3
  let total_spent : ℚ := 5

  -- Calculate the total number of limes bought
  let total_limes : ℚ := total_spent * limes_per_dollar

  -- Calculate the total amount of lime juice
  let total_lime_juice : ℚ := total_limes * lime_juice_per_lime

  -- Calculate the amount of lime juice per mocktail
  let lime_juice_per_mocktail : ℚ := total_lime_juice / (days * mocktails_per_day)

  -- Prove that the amount of lime juice per mocktail is 1 tablespoon
  lime_juice_per_mocktail = 1 := by sorry

end NUMINAMATH_CALUDE_julio_mocktail_lime_juice_l135_13544


namespace NUMINAMATH_CALUDE_point_on_line_point_twelve_seven_on_line_l135_13533

/-- Given three points in the plane, this theorem states that if the first two points
    determine a line, then the third point lies on that line. -/
theorem point_on_line (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) :
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁) →
  ∃ (m b : ℝ), y₁ = m * x₁ + b ∧ y₂ = m * x₂ + b ∧ y₃ = m * x₃ + b :=
by sorry

/-- The point (12,7) lies on the line passing through (0,1) and (-6,-2). -/
theorem point_twelve_seven_on_line : 
  ∃ (m b : ℝ), 1 = m * 0 + b ∧ -2 = m * (-6) + b ∧ 7 = m * 12 + b :=
by
  apply point_on_line 0 1 (-6) (-2) 12 7
  -- Proof that the points are collinear
  sorry

end NUMINAMATH_CALUDE_point_on_line_point_twelve_seven_on_line_l135_13533


namespace NUMINAMATH_CALUDE_stock_change_theorem_l135_13549

theorem stock_change_theorem (initial_value : ℝ) : 
  let day1_value := initial_value * (1 - 0.15)
  let day2_value := day1_value * (1 + 0.25)
  let percent_change := (day2_value - initial_value) / initial_value * 100
  percent_change = 6.25 := by
  sorry

end NUMINAMATH_CALUDE_stock_change_theorem_l135_13549


namespace NUMINAMATH_CALUDE_parabola_reflects_to_parallel_l135_13563

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The equation of a curve in 2D space -/
def CurveEquation : Type := Point → Prop

/-- The equation of a parabola y^2 = 2Cx + C^2 -/
def ParabolaEquation (C : ℝ) : CurveEquation :=
  fun p => p.y^2 = 2*C*p.x + C^2

/-- A ray of light -/
structure Ray where
  origin : Point
  direction : Point

/-- The reflection of a ray off a curve at a point -/
def ReflectedRay (curve : CurveEquation) (incidentRay : Ray) (reflectionPoint : Point) : Ray :=
  sorry

/-- The theorem stating that a parabola reflects rays from the origin into parallel rays -/
theorem parabola_reflects_to_parallel (C : ℝ) :
  ∀ (p : Point), ParabolaEquation C p →
  ∀ (incidentRay : Ray),
    incidentRay.origin = ⟨0, 0⟩ →
    (ReflectedRay (ParabolaEquation C) incidentRay p).direction.y = 0 :=
  sorry

end NUMINAMATH_CALUDE_parabola_reflects_to_parallel_l135_13563


namespace NUMINAMATH_CALUDE_kevin_food_expenditure_l135_13532

theorem kevin_food_expenditure (total_budget : ℕ) (samuel_ticket : ℕ) (samuel_food_drinks : ℕ) (kevin_drinks : ℕ) :
  total_budget = 20 →
  samuel_ticket = 14 →
  samuel_food_drinks = 6 →
  kevin_drinks = 2 →
  total_budget = samuel_ticket + samuel_food_drinks →
  ∃ (kevin_food : ℕ), total_budget = samuel_ticket + kevin_drinks + kevin_food ∧ kevin_food = 4 :=
by sorry

end NUMINAMATH_CALUDE_kevin_food_expenditure_l135_13532


namespace NUMINAMATH_CALUDE_total_cost_nine_knives_l135_13547

/-- Calculates the total cost of sharpening knives based on a specific pricing structure. -/
def total_sharpening_cost (num_knives : ℕ) : ℚ :=
  let first_knife_cost : ℚ := 5
  let next_three_cost : ℚ := 4
  let remaining_cost : ℚ := 3
  let num_next_three : ℕ := min 3 (num_knives - 1)
  let num_remaining : ℕ := max 0 (num_knives - 4)
  first_knife_cost + 
  (next_three_cost * num_next_three) + 
  (remaining_cost * num_remaining)

/-- Theorem stating that the total cost to sharpen 9 knives is $32.00. -/
theorem total_cost_nine_knives : 
  total_sharpening_cost 9 = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_nine_knives_l135_13547


namespace NUMINAMATH_CALUDE_extremum_of_f_on_M_l135_13578

def M : Set ℝ := {x | x^2 + 4*x ≤ 0}

def f (x : ℝ) : ℝ := -x^2 - 6*x + 1

theorem extremum_of_f_on_M :
  ∃ (min max : ℝ), 
    (∀ x ∈ M, f x ≥ min) ∧ 
    (∃ x ∈ M, f x = min) ∧
    (∀ x ∈ M, f x ≤ max) ∧ 
    (∃ x ∈ M, f x = max) ∧
    min = 1 ∧ max = 10 :=
sorry

end NUMINAMATH_CALUDE_extremum_of_f_on_M_l135_13578


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sum_sixth_term_l135_13513

/-- An arithmetic-geometric sequence -/
def arithmetic_geometric_sequence (a : ℕ → ℝ) : Prop := sorry

/-- Sum of the first n terms of a sequence -/
def S (a : ℕ → ℝ) (n : ℕ) : ℝ := sorry

theorem arithmetic_geometric_sum_sixth_term 
  (a : ℕ → ℝ) 
  (h_ag : arithmetic_geometric_sequence a)
  (h_s2 : S a 2 = 1)
  (h_s4 : S a 4 = 3) : 
  S a 6 = 7 := by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sum_sixth_term_l135_13513


namespace NUMINAMATH_CALUDE_ellipse_parameter_sum_l135_13588

/-- An ellipse with foci F₁ and F₂, and constant sum of distances from any point to both foci -/
structure Ellipse where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  sum_distances : ℝ

/-- The center, semi-major axis, and semi-minor axis of an ellipse -/
structure EllipseParameters where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- Given an ellipse, compute its parameters -/
def compute_ellipse_parameters (e : Ellipse) : EllipseParameters :=
  sorry

/-- The main theorem: sum of center coordinates and axes lengths for the given ellipse -/
theorem ellipse_parameter_sum (e : Ellipse) 
    (h : e.F₁ = (0, 2) ∧ e.F₂ = (6, 2) ∧ e.sum_distances = 10) : 
    let p := compute_ellipse_parameters e
    p.h + p.k + p.a + p.b = 14 :=
  sorry

end NUMINAMATH_CALUDE_ellipse_parameter_sum_l135_13588


namespace NUMINAMATH_CALUDE_carrot_picking_l135_13584

theorem carrot_picking (carol_carrots : ℕ) (good_carrots : ℕ) (bad_carrots : ℕ) 
  (h1 : carol_carrots = 29)
  (h2 : good_carrots = 38)
  (h3 : bad_carrots = 7) :
  good_carrots + bad_carrots - carol_carrots = 16 := by
  sorry

end NUMINAMATH_CALUDE_carrot_picking_l135_13584


namespace NUMINAMATH_CALUDE_family_composition_l135_13571

/-- A family where one member has an equal number of brothers and sisters,
    and another member has twice as many brothers as sisters. -/
structure Family where
  boys : ℕ
  girls : ℕ
  tony_equal_siblings : boys - 1 = girls
  alice_double_brothers : boys = 2 * (girls - 1)

/-- The family has 4 boys and 3 girls. -/
theorem family_composition (f : Family) : f.boys = 4 ∧ f.girls = 3 := by
  sorry

end NUMINAMATH_CALUDE_family_composition_l135_13571


namespace NUMINAMATH_CALUDE_square_traffic_sign_perimeter_l135_13555

/-- A square traffic sign with sides of 4 feet has a perimeter of 16 feet. -/
theorem square_traffic_sign_perimeter : 
  ∀ (side_length : ℝ), side_length = 4 → 4 * side_length = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_traffic_sign_perimeter_l135_13555


namespace NUMINAMATH_CALUDE_exists_counterexample_to_inequality_l135_13575

theorem exists_counterexample_to_inequality (a b c : ℝ) 
  (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) :
  ∃ (a b c : ℝ), c < b ∧ b < a ∧ a * c < 0 ∧ c * b^2 ≥ a * b^2 :=
sorry

end NUMINAMATH_CALUDE_exists_counterexample_to_inequality_l135_13575


namespace NUMINAMATH_CALUDE_production_volume_equation_l135_13523

theorem production_volume_equation (x : ℝ) : 
  (200 : ℝ) + 200 * (1 + x) + 200 * (1 + x)^2 = 1400 ↔ 
  (∃ y : ℝ, y > 0 ∧ 
    200 * (1 + y + (1 + y)^2) = 1400 ∧
    (∀ n : ℕ, n ≥ 1 ∧ n ≤ 3 → 
      (200 * (1 + y)^(n - 1) = 200 * (1 + x)^(n - 1)))) :=
by sorry

end NUMINAMATH_CALUDE_production_volume_equation_l135_13523


namespace NUMINAMATH_CALUDE_ratio_chain_l135_13561

theorem ratio_chain (a b c d e f g h : ℝ) 
  (hab : a / b = 7 / 3)
  (hbc : b / c = 5 / 2)
  (hcd : c / d = 2)
  (hde : d / e = 3 / 2)
  (hef : e / f = 4 / 3)
  (hfg : f / g = 1 / 4)
  (hgh : g / h = 3 / 5) :
  a * b * c * d * e * f * g / (d * e * f * g * h * i * j) = 15.75 :=
by sorry

end NUMINAMATH_CALUDE_ratio_chain_l135_13561


namespace NUMINAMATH_CALUDE_unique_prime_triple_l135_13569

theorem unique_prime_triple : 
  ∀ p q r : ℕ+, 
    Prime p.val → Prime q.val → 
    (r.val^2 - 5*q.val^2) / (p.val^2 - 1) = 2 → 
    (p, q, r) = (3, 2, 6) := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_triple_l135_13569


namespace NUMINAMATH_CALUDE_real_part_of_i_squared_times_one_minus_two_i_l135_13530

theorem real_part_of_i_squared_times_one_minus_two_i : 
  Complex.re (Complex.I^2 * (1 - 2*Complex.I)) = -1 := by
sorry

end NUMINAMATH_CALUDE_real_part_of_i_squared_times_one_minus_two_i_l135_13530


namespace NUMINAMATH_CALUDE_zero_point_in_interval_l135_13505

noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 / x

theorem zero_point_in_interval :
  ∃ x : ℝ, 2 < x ∧ x < 3 ∧ f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_zero_point_in_interval_l135_13505


namespace NUMINAMATH_CALUDE_adjacent_sum_divisible_by_four_l135_13568

/-- A type representing a cell in the grid -/
structure Cell where
  row : Fin 22
  col : Fin 22

/-- A function representing the number in each cell -/
def gridValue : Cell → Fin (22^2) := sorry

/-- Two cells are adjacent if they share an edge or vertex -/
def adjacent (c1 c2 : Cell) : Prop := sorry

theorem adjacent_sum_divisible_by_four :
  ∃ c1 c2 : Cell, adjacent c1 c2 ∧ (gridValue c1 + gridValue c2) % 4 = 0 := by sorry

end NUMINAMATH_CALUDE_adjacent_sum_divisible_by_four_l135_13568


namespace NUMINAMATH_CALUDE_negation_of_exponential_proposition_l135_13559

theorem negation_of_exponential_proposition :
  (¬ ∀ x : ℝ, x > 0 → Real.exp x ≥ 1) ↔ (∃ x : ℝ, x > 0 ∧ Real.exp x < 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_exponential_proposition_l135_13559


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l135_13512

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) :
  (∀ x : ℝ, (1 - 2*x)^9 = a₉*x^9 + a₈*x^8 + a₇*x^7 + a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l135_13512


namespace NUMINAMATH_CALUDE_fraction_problem_l135_13516

theorem fraction_problem (N : ℝ) (x y : ℤ) :
  N = 30 →
  0.5 * N = (x / y : ℝ) * N + 10 →
  (x / y : ℝ) = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l135_13516


namespace NUMINAMATH_CALUDE_total_winter_clothing_l135_13570

/-- The number of boxes of winter clothing -/
def num_boxes : ℕ := 3

/-- The number of scarves in each box -/
def scarves_per_box : ℕ := 3

/-- The number of mittens in each box -/
def mittens_per_box : ℕ := 4

/-- Theorem: The total number of winter clothing pieces is 21 -/
theorem total_winter_clothing : 
  num_boxes * (scarves_per_box + mittens_per_box) = 21 := by
sorry

end NUMINAMATH_CALUDE_total_winter_clothing_l135_13570


namespace NUMINAMATH_CALUDE_unique_prime_triplet_l135_13566

theorem unique_prime_triplet :
  ∀ a b c : ℕ+,
    (Nat.Prime (a + b * c) ∧ 
     Nat.Prime (b + a * c) ∧ 
     Nat.Prime (c + a * b)) ∧
    ((a + b * c) ∣ ((a ^ 2 + 1) * (b ^ 2 + 1) * (c ^ 2 + 1))) ∧
    ((b + a * c) ∣ ((a ^ 2 + 1) * (b ^ 2 + 1) * (c ^ 2 + 1))) ∧
    ((c + a * b) ∣ ((a ^ 2 + 1) * (b ^ 2 + 1) * (c ^ 2 + 1))) →
    a = 1 ∧ b = 1 ∧ c = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_triplet_l135_13566


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l135_13589

theorem simplify_trig_expression :
  1 / Real.sin (10 * π / 180) - Real.sqrt 3 / Real.cos (10 * π / 180) = 4 := by sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l135_13589


namespace NUMINAMATH_CALUDE_right_triangle_vector_property_l135_13535

-- Define a right-angled triangle ABC
structure RightTriangleABC where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_right_angled : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

-- Define the theorem
theorem right_triangle_vector_property (t : RightTriangleABC) (x : ℝ) 
  (h1 : t.C.1 - t.A.1 = 2 ∧ t.C.2 - t.A.2 = 4)
  (h2 : t.C.1 - t.B.1 = -6 ∧ t.C.2 - t.B.2 = x) :
  x = 3 := by
  sorry

-- The proof is omitted as per instructions

end NUMINAMATH_CALUDE_right_triangle_vector_property_l135_13535


namespace NUMINAMATH_CALUDE_chef_initial_potatoes_l135_13504

/-- Represents the number of fries that can be made from one potato -/
def fries_per_potato : ℕ := 25

/-- Represents the total number of fries needed -/
def total_fries_needed : ℕ := 200

/-- Represents the number of potatoes leftover after making the required fries -/
def leftover_potatoes : ℕ := 7

/-- Calculates the initial number of potatoes the chef had -/
def initial_potatoes : ℕ := (total_fries_needed / fries_per_potato) + leftover_potatoes

/-- Proves that the initial number of potatoes is 15 -/
theorem chef_initial_potatoes :
  initial_potatoes = 15 :=
by sorry

end NUMINAMATH_CALUDE_chef_initial_potatoes_l135_13504


namespace NUMINAMATH_CALUDE_intersection_implies_equality_l135_13524

theorem intersection_implies_equality (k b a c : ℝ) : 
  k ≠ b → 
  (∃! p : ℝ × ℝ, (p.2 = k * p.1 + k) ∧ (p.2 = b * p.1 + b) ∧ (p.2 = a * p.1 + c)) →
  a = c := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_equality_l135_13524


namespace NUMINAMATH_CALUDE_complex_equation_solution_l135_13501

theorem complex_equation_solution (z : ℂ) : 
  z * (1 - 2 * Complex.I) = 2 + 4 * Complex.I → 
  z = -2/5 + 8/5 * Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l135_13501


namespace NUMINAMATH_CALUDE_evaluate_expression_l135_13528

theorem evaluate_expression (x : ℝ) (h : x = 2) : (3 * x^2 - 8 * x + 5) * (4 * x - 7) = 1 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l135_13528


namespace NUMINAMATH_CALUDE_square_sum_inequality_l135_13560

theorem square_sum_inequality (a b : ℝ) : a^2 + b^2 - 1 - a^2*b^2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_inequality_l135_13560


namespace NUMINAMATH_CALUDE_no_three_distinct_rational_roots_l135_13541

theorem no_three_distinct_rational_roots (a b : ℝ) : 
  ¬ (∃ (u v w : ℚ), u ≠ v ∧ v ≠ w ∧ u ≠ w ∧ 
    (u^3 + (2*a+1)*u^2 + (2*a^2+2*a-3)*u + b = 0) ∧
    (v^3 + (2*a+1)*v^2 + (2*a^2+2*a-3)*v + b = 0) ∧
    (w^3 + (2*a+1)*w^2 + (2*a^2+2*a-3)*w + b = 0)) :=
by
  sorry

end NUMINAMATH_CALUDE_no_three_distinct_rational_roots_l135_13541


namespace NUMINAMATH_CALUDE_walking_speed_equation_l135_13508

theorem walking_speed_equation (x : ℝ) 
  (h1 : x > 0) -- Xiao Wang's speed is positive
  (h2 : x + 1 > 0) -- Xiao Zhang's speed is positive
  : 15 / x - 15 / (x + 1) = 1 / 2 ↔ 
    (15 / x = 15 / (x + 1) + 1 / 2 ∧ 
     15 / (x + 1) < 15 / x) :=
by sorry

end NUMINAMATH_CALUDE_walking_speed_equation_l135_13508


namespace NUMINAMATH_CALUDE_product_of_squares_and_prime_l135_13565

theorem product_of_squares_and_prime : 2^2 * 3^2 * 5^2 * 7 = 6300 := by
  sorry

end NUMINAMATH_CALUDE_product_of_squares_and_prime_l135_13565


namespace NUMINAMATH_CALUDE_book_purchase_problem_l135_13574

/-- Represents the number of books purchased -/
def num_books : ℕ := 8

/-- Represents the number of albums purchased -/
def num_albums : ℕ := num_books - 6

/-- Represents the price of a book in kopecks -/
def price_book : ℕ := 1056 / num_books

/-- Represents the price of an album in kopecks -/
def price_album : ℕ := 56 / num_albums

/-- Theorem stating that the given conditions are satisfied by the defined values -/
theorem book_purchase_problem :
  (num_books : ℤ) = (num_albums : ℤ) + 6 ∧
  num_books * price_book = 1056 ∧
  num_albums * price_album = 56 ∧
  price_book > price_album + 100 :=
by sorry

end NUMINAMATH_CALUDE_book_purchase_problem_l135_13574
