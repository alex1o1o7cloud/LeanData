import Mathlib

namespace NUMINAMATH_CALUDE_base_is_seven_l1450_145079

/-- Converts a number from base s to base 10 -/
def to_base_10 (digits : List Nat) (s : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * s^i) 0

/-- The transaction equation holds for the given base -/
def transaction_holds (s : Nat) : Prop :=
  to_base_10 [3, 2, 5] s + to_base_10 [3, 5, 4] s = to_base_10 [0, 0, 1, 1] s

theorem base_is_seven :
  ∃ s : Nat, s > 1 ∧ transaction_holds s ∧ s = 7 := by
  sorry

end NUMINAMATH_CALUDE_base_is_seven_l1450_145079


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1450_145086

/-- Given a cube with surface area 54 square centimeters, its volume is 27 cubic centimeters. -/
theorem cube_volume_from_surface_area :
  ∀ (side_length : ℝ),
  (6 * side_length^2 = 54) →
  side_length^3 = 27 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1450_145086


namespace NUMINAMATH_CALUDE_brownie_pieces_l1450_145063

theorem brownie_pieces (pan_length pan_width piece_length piece_width : ℕ) 
  (h1 : pan_length = 24)
  (h2 : pan_width = 15)
  (h3 : piece_length = 3)
  (h4 : piece_width = 2) :
  (pan_length * pan_width) / (piece_length * piece_width) = 60 := by
  sorry

end NUMINAMATH_CALUDE_brownie_pieces_l1450_145063


namespace NUMINAMATH_CALUDE_mean_median_difference_l1450_145083

-- Define the frequency distribution of sick days
def sick_days_freq : List (Nat × Nat) := [(0, 4), (1, 2), (2, 5), (3, 2), (4, 1), (5, 1)]

-- Total number of students
def total_students : Nat := 15

-- Function to calculate the median
def median (freq : List (Nat × Nat)) (total : Nat) : Nat :=
  sorry

-- Function to calculate the mean
def mean (freq : List (Nat × Nat)) (total : Nat) : Rat :=
  sorry

-- Theorem statement
theorem mean_median_difference :
  mean sick_days_freq total_students = (median sick_days_freq total_students : Rat) - 1/5 :=
sorry

end NUMINAMATH_CALUDE_mean_median_difference_l1450_145083


namespace NUMINAMATH_CALUDE_white_square_area_l1450_145075

-- Define the cube's edge length
def cube_edge : ℝ := 15

-- Define the total area of blue paint
def blue_paint_area : ℝ := 500

-- Define the number of faces of a cube
def cube_faces : ℕ := 6

-- Theorem statement
theorem white_square_area :
  let face_area := cube_edge ^ 2
  let blue_area_per_face := blue_paint_area / cube_faces
  let white_area_per_face := face_area - blue_area_per_face
  white_area_per_face = 425 / 3 := by
  sorry

end NUMINAMATH_CALUDE_white_square_area_l1450_145075


namespace NUMINAMATH_CALUDE_waiter_customers_l1450_145043

theorem waiter_customers (initial_customers : ℕ) : 
  (initial_customers - 3 + 39 = 50) → initial_customers = 14 := by
  sorry

end NUMINAMATH_CALUDE_waiter_customers_l1450_145043


namespace NUMINAMATH_CALUDE_product_squared_l1450_145057

theorem product_squared (a b : ℝ) : (a * b)^2 = a^2 * b^2 := by sorry

end NUMINAMATH_CALUDE_product_squared_l1450_145057


namespace NUMINAMATH_CALUDE_alligators_not_hiding_l1450_145033

/-- The number of alligators not hiding in a zoo cage -/
theorem alligators_not_hiding (total_alligators hiding_alligators : ℕ) :
  total_alligators = 75 → hiding_alligators = 19 →
  total_alligators - hiding_alligators = 56 := by
  sorry

#check alligators_not_hiding

end NUMINAMATH_CALUDE_alligators_not_hiding_l1450_145033


namespace NUMINAMATH_CALUDE_minimizing_integral_minimizing_function_achieves_minimum_minimizing_function_integral_one_l1450_145061

noncomputable def minimizing_function (x : ℝ) : ℝ := 6 / (Real.pi * (x^2 + x + 1))

theorem minimizing_integral 
  (f : ℝ → ℝ) 
  (hf_continuous : Continuous f) 
  (hf_integral : ∫ x in (0:ℝ)..1, f x = 1) :
  ∫ x in (0:ℝ)..1, (x^2 + x + 1) * (f x)^2 ≥ 6 / Real.pi :=
sorry

theorem minimizing_function_achieves_minimum :
  ∫ x in (0:ℝ)..1, (x^2 + x + 1) * (minimizing_function x)^2 = 6 / Real.pi :=
sorry

theorem minimizing_function_integral_one :
  ∫ x in (0:ℝ)..1, minimizing_function x = 1 :=
sorry

end NUMINAMATH_CALUDE_minimizing_integral_minimizing_function_achieves_minimum_minimizing_function_integral_one_l1450_145061


namespace NUMINAMATH_CALUDE_rationalization_sqrt_five_l1450_145089

/-- Rationalization of (2+√5)/(2-√5) -/
theorem rationalization_sqrt_five : ∃ (A B C : ℤ), 
  (2 + Real.sqrt 5) / (2 - Real.sqrt 5) = A + B * Real.sqrt C ∧ 
  A = -9 ∧ B = -4 ∧ C = 5 := by
  sorry

end NUMINAMATH_CALUDE_rationalization_sqrt_five_l1450_145089


namespace NUMINAMATH_CALUDE_january_has_greatest_difference_l1450_145017

-- Define the sales data for each month
def january_sales : (Nat × Nat) := (5, 2)
def february_sales : (Nat × Nat) := (6, 4)
def march_sales : (Nat × Nat) := (5, 5)
def april_sales : (Nat × Nat) := (4, 6)
def may_sales : (Nat × Nat) := (3, 5)

-- Define the percentage difference function
def percentage_difference (sales : Nat × Nat) : ℚ :=
  let (drummers, buglers) := sales
  (↑(max drummers buglers - min drummers buglers) / ↑(min drummers buglers)) * 100

-- Theorem statement
theorem january_has_greatest_difference :
  percentage_difference january_sales >
  max (percentage_difference february_sales)
    (max (percentage_difference march_sales)
      (max (percentage_difference april_sales)
        (percentage_difference may_sales))) :=
by sorry

end NUMINAMATH_CALUDE_january_has_greatest_difference_l1450_145017


namespace NUMINAMATH_CALUDE_arithmetic_mean_relation_l1450_145052

theorem arithmetic_mean_relation (a b x : ℝ) : 
  (2 * x = a + b) →  -- x is the arithmetic mean of a and b
  (2 * x^2 = a^2 - b^2) →  -- x² is the arithmetic mean of a² and -b²
  (a = -b ∨ a = 3*b) :=  -- The relationship between a and b
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_relation_l1450_145052


namespace NUMINAMATH_CALUDE_benjamin_egg_collection_l1450_145009

/-- Proves that Benjamin collects 6 dozen eggs a day given the conditions of the problem -/
theorem benjamin_egg_collection :
  ∀ (benjamin_eggs : ℕ),
  (∃ (carla_eggs trisha_eggs : ℕ),
    carla_eggs = 3 * benjamin_eggs ∧
    trisha_eggs = benjamin_eggs - 4 ∧
    benjamin_eggs + carla_eggs + trisha_eggs = 26) →
  benjamin_eggs = 6 := by
sorry

end NUMINAMATH_CALUDE_benjamin_egg_collection_l1450_145009


namespace NUMINAMATH_CALUDE_not_q_necessary_not_sufficient_for_not_p_l1450_145069

def p (x : ℝ) : Prop := |x + 1| ≤ 4

def q (x : ℝ) : Prop := 2 < x ∧ x < 3

theorem not_q_necessary_not_sufficient_for_not_p :
  (∀ x, ¬(p x) → ¬(q x)) ∧ (∃ x, ¬(q x) ∧ p x) :=
sorry

end NUMINAMATH_CALUDE_not_q_necessary_not_sufficient_for_not_p_l1450_145069


namespace NUMINAMATH_CALUDE_milk_conversion_rate_l1450_145074

/-- The number of ounces in a gallon of milk -/
def ounces_per_gallon : ℕ := sorry

/-- The initial amount of milk in gallons -/
def initial_gallons : ℕ := 3

/-- The amount of milk consumed in ounces -/
def consumed_ounces : ℕ := 13

/-- The remaining amount of milk in ounces -/
def remaining_ounces : ℕ := 371

theorem milk_conversion_rate :
  ounces_per_gallon = 128 :=
by sorry

end NUMINAMATH_CALUDE_milk_conversion_rate_l1450_145074


namespace NUMINAMATH_CALUDE_sin_cos_range_l1450_145037

open Real

theorem sin_cos_range :
  ∀ y : ℝ, (∃ x : ℝ, sin x + cos x = y) ↔ -Real.sqrt 2 ≤ y ∧ y ≤ Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_sin_cos_range_l1450_145037


namespace NUMINAMATH_CALUDE_fraction_simplification_l1450_145016

theorem fraction_simplification : (25 : ℚ) / 24 * 18 / 35 * 56 / 45 = 50 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1450_145016


namespace NUMINAMATH_CALUDE_unique_valid_stamp_set_l1450_145050

/-- Given unlimited supply of stamps of denominations 7, n, and n+1 cents,
    101 cents is the greatest postage that cannot be formed -/
def is_valid_stamp_set (n : ℕ+) : Prop :=
  ∀ k : ℕ, k > 101 → ∃ a b c : ℕ, k = 7 * a + n * b + (n + 1) * c ∧
  ¬∃ a b c : ℕ, 101 = 7 * a + n * b + (n + 1) * c

theorem unique_valid_stamp_set :
  ∃! n : ℕ+, is_valid_stamp_set n ∧ n = 18 := by sorry

end NUMINAMATH_CALUDE_unique_valid_stamp_set_l1450_145050


namespace NUMINAMATH_CALUDE_unique_triangle_side_l1450_145091

/-- A function that checks if a triangle with sides a, b, and c can exist -/
def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem stating that 3 is the only positive integer value of x for which
    a triangle with sides 5, x + 1, and x^3 can exist -/
theorem unique_triangle_side : ∀ x : ℕ+, 
  (is_valid_triangle 5 (x + 1) (x^3) ↔ x = 3) := by sorry

end NUMINAMATH_CALUDE_unique_triangle_side_l1450_145091


namespace NUMINAMATH_CALUDE_anthony_final_pet_count_l1450_145023

/-- The number of pets Anthony has after a series of events --/
def final_pet_count (initial_pets : ℕ) : ℕ :=
  let pets_after_loss := initial_pets - (initial_pets * 12 / 100)
  let pets_after_contest := pets_after_loss + 7
  let pets_giving_birth := pets_after_contest / 4
  let new_offspring := pets_giving_birth * 2
  let pets_before_deaths := pets_after_contest + new_offspring
  pets_before_deaths - (pets_before_deaths / 10)

/-- Theorem stating that Anthony ends up with 62 pets --/
theorem anthony_final_pet_count :
  final_pet_count 45 = 62 := by
  sorry

end NUMINAMATH_CALUDE_anthony_final_pet_count_l1450_145023


namespace NUMINAMATH_CALUDE_triangle_side_length_l1450_145006

theorem triangle_side_length (a c : ℝ) (B : ℝ) (h1 : a = 5) (h2 : c = 8) (h3 : B = π / 3) :
  ∃ b : ℝ, b ^ 2 = a ^ 2 + c ^ 2 - 2 * a * c * Real.cos B ∧ b > 0 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1450_145006


namespace NUMINAMATH_CALUDE_equality_and_inequality_of_expressions_l1450_145090

variable (a : ℝ)

def f (n : ℕ) (x : ℝ) : ℝ := x ^ n

theorem equality_and_inequality_of_expressions (h : a ≠ 1) :
  (∀ n : ℕ, f n a = a ^ n) →
  ((f 11 (f 13 a)) ^ 14 = f 2002 a) ∧
  (f 11 (f 13 (f 14 a)) = f 2002 a) ∧
  ((f 11 a * f 13 a) ^ 14 ≠ f 2002 a) ∧
  (f 11 a * f 13 a * f 14 a ≠ f 2002 a) := by
  sorry

end NUMINAMATH_CALUDE_equality_and_inequality_of_expressions_l1450_145090


namespace NUMINAMATH_CALUDE_possible_sets_B_l1450_145034

theorem possible_sets_B (A B : Set Int) : 
  A = {-1} → A ∪ B = {-1, 3} → (B = {3} ∨ B = {-1, 3}) := by
  sorry

end NUMINAMATH_CALUDE_possible_sets_B_l1450_145034


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1450_145062

-- Problem 1
theorem problem_1 (a : ℝ) : (-2*a)^3 + 2*a^2 * 5*a = 2*a^3 := by
  sorry

-- Problem 2
theorem problem_2 (x y : ℝ) : (3*x*y^2)^2 + (-4*x*y^3)*(-x*y) = 13*x^2*y^4 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1450_145062


namespace NUMINAMATH_CALUDE_difference_of_values_l1450_145053

theorem difference_of_values (x y : ℝ) 
  (sum_eq : x + y = 10) 
  (diff_squares_eq : x^2 - y^2 = 40) : 
  x - y = 4 := by
sorry

end NUMINAMATH_CALUDE_difference_of_values_l1450_145053


namespace NUMINAMATH_CALUDE_no_real_solutions_l1450_145067

theorem no_real_solutions :
  ¬∃ y : ℝ, (3 * y - 4)^2 + 4 = -(y + 3) := by
sorry

end NUMINAMATH_CALUDE_no_real_solutions_l1450_145067


namespace NUMINAMATH_CALUDE_cos_equation_solution_l1450_145028

theorem cos_equation_solution (θ : Real) :
  2 * (Real.cos θ)^2 - 5 * Real.cos θ + 2 = 0 → θ = Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_equation_solution_l1450_145028


namespace NUMINAMATH_CALUDE_cube_root_of_64_l1450_145056

theorem cube_root_of_64 : (64 : ℝ) ^ (1/3) = 4 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_64_l1450_145056


namespace NUMINAMATH_CALUDE_second_cube_volume_is_64_l1450_145059

-- Define the volume of the first cube
def first_cube_volume : ℝ := 8

-- Define the relationship between the surface areas of the two cubes
def surface_area_ratio : ℝ := 4

-- Theorem statement
theorem second_cube_volume_is_64 :
  let first_side := first_cube_volume ^ (1/3 : ℝ)
  let first_surface_area := 6 * first_side^2
  let second_surface_area := surface_area_ratio * first_surface_area
  let second_side := (second_surface_area / 6) ^ (1/2 : ℝ)
  second_side^3 = 64 := by sorry

end NUMINAMATH_CALUDE_second_cube_volume_is_64_l1450_145059


namespace NUMINAMATH_CALUDE_difference_of_hypotenuse_numbers_l1450_145047

/-- A hypotenuse number is a natural number that can be represented as the sum of two squares of non-negative integers. -/
def is_hypotenuse (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = a^2 + b^2

/-- Any natural number greater than 10 can be represented as the difference of two hypotenuse numbers. -/
theorem difference_of_hypotenuse_numbers (n : ℕ) (h : n > 10) :
  ∃ m₁ m₂ : ℕ, is_hypotenuse m₁ ∧ is_hypotenuse m₂ ∧ n = m₁ - m₂ :=
sorry

end NUMINAMATH_CALUDE_difference_of_hypotenuse_numbers_l1450_145047


namespace NUMINAMATH_CALUDE_right_triangle_among_sets_l1450_145046

def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

theorem right_triangle_among_sets :
  ¬ is_right_triangle 1 2 3 ∧
  ¬ is_right_triangle 2 3 4 ∧
  is_right_triangle 3 4 5 ∧
  ¬ is_right_triangle 4 5 6 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_among_sets_l1450_145046


namespace NUMINAMATH_CALUDE_system_solutions_l1450_145032

def is_solution (x y : ℤ) : Prop :=
  |x^2 - 2*x| < y + (1/2) ∧ y + |x - 1| < 2

theorem system_solutions :
  ∀ x y : ℤ, is_solution x y ↔ (x = 0 ∧ y = 0) ∨ (x = 2 ∧ y = 0) ∨ (x = 1 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l1450_145032


namespace NUMINAMATH_CALUDE_job_fair_problem_l1450_145027

/-- The probability of individual A being hired -/
def prob_A : ℚ := 4/9

/-- The probability of individuals B and C being hired -/
def prob_BC (t : ℚ) : ℚ := t/3

/-- The condition that t is between 0 and 3 -/
def t_condition (t : ℚ) : Prop := 0 < t ∧ t < 3

/-- The probability of all three individuals being hired -/
def prob_all (t : ℚ) : ℚ := prob_A * prob_BC t * prob_BC t

/-- The number of people hired from A and B -/
def ξ : Fin 3 → ℚ
| 0 => 0
| 1 => 1
| 2 => 2

/-- The probability distribution of ξ -/
def prob_ξ (t : ℚ) : Fin 3 → ℚ
| 0 => (1 - prob_A) * (1 - prob_BC t)
| 1 => prob_A * (1 - prob_BC t) + (1 - prob_A) * prob_BC t
| 2 => prob_A * prob_BC t

/-- The mathematical expectation of ξ -/
def expectation_ξ (t : ℚ) : ℚ :=
  (ξ 0) * (prob_ξ t 0) + (ξ 1) * (prob_ξ t 1) + (ξ 2) * (prob_ξ t 2)

theorem job_fair_problem (t : ℚ) (h : t_condition t) (h_prob : prob_all t = 16/81) :
  t = 2 ∧ expectation_ξ t = 10/9 := by
  sorry

end NUMINAMATH_CALUDE_job_fair_problem_l1450_145027


namespace NUMINAMATH_CALUDE_rectangle_length_proof_l1450_145054

theorem rectangle_length_proof (width : ℝ) (small_area : ℝ) : 
  width = 20 → small_area = 200 → ∃ (length : ℝ), 
    length = 40 ∧ 
    (length / 2) * (width / 2) = small_area := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_proof_l1450_145054


namespace NUMINAMATH_CALUDE_fifth_selected_number_l1450_145005

def random_number_table : List Nat :=
  [16, 22, 77, 94, 39, 49, 54, 43, 54, 82, 17, 37, 93, 23, 78, 87, 35, 20, 96, 43]

def class_size : Nat := 50

def is_valid_number (n : Nat) : Bool :=
  n < class_size

def select_valid_numbers (numbers : List Nat) (count : Nat) : List Nat :=
  (numbers.filter is_valid_number).take count

theorem fifth_selected_number :
  (select_valid_numbers random_number_table 5).reverse.head? = some 43 := by
  sorry

end NUMINAMATH_CALUDE_fifth_selected_number_l1450_145005


namespace NUMINAMATH_CALUDE_coefficient_of_x_l1450_145038

theorem coefficient_of_x (x y : ℚ) :
  (x + 3 * y = 1) →
  (2 * x + y = 5) →
  ∃ (a : ℚ), a * x + y = 19 ∧ a = 7 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x_l1450_145038


namespace NUMINAMATH_CALUDE_math_contest_schools_count_l1450_145094

/-- Represents a participant in the math contest -/
structure Participant where
  score : ℕ
  rank : ℕ

/-- Represents a school team in the math contest -/
structure School where
  team : Fin 4 → Participant

/-- The math contest -/
structure MathContest where
  schools : List School
  andrea : Participant
  beth : Participant
  carla : Participant

/-- The conditions of the math contest -/
def ContestConditions (contest : MathContest) : Prop :=
  ∀ s₁ s₂ : School, ∀ p₁ p₂ : Fin 4, 
    (s₁ ≠ s₂ ∨ p₁ ≠ p₂) → (s₁.team p₁).score ≠ (s₂.team p₂).score
  ∧ contest.andrea.rank < contest.beth.rank
  ∧ contest.beth.rank = 46
  ∧ contest.carla.rank = 79
  ∧ contest.andrea.rank = (contest.schools.length * 4 + 1) / 2
  ∧ ∀ s : School, ∀ p : Fin 4, contest.andrea.score ≥ (s.team p).score

theorem math_contest_schools_count 
  (contest : MathContest) 
  (h : ContestConditions contest) : 
  contest.schools.length = 19 := by
  sorry


end NUMINAMATH_CALUDE_math_contest_schools_count_l1450_145094


namespace NUMINAMATH_CALUDE_two_colored_cubes_count_l1450_145011

/-- Represents a cube with its side length -/
structure Cube where
  side : ℕ

/-- Represents a hollow cube with outer and inner dimensions -/
structure HollowCube where
  outer : Cube
  inner : Cube

/-- Calculates the number of smaller cubes with paint on exactly two sides -/
def cubesWithTwoColoredSides (hc : HollowCube) (smallCubeSide : ℕ) : ℕ :=
  12 * (hc.outer.side / smallCubeSide - 2)

theorem two_colored_cubes_count 
  (bigCube : Cube)
  (smallCube : Cube)
  (tinyCube : Cube)
  (hc : HollowCube) :
  bigCube.side = 27 →
  smallCube.side = 9 →
  tinyCube.side = 3 →
  hc.outer = bigCube →
  hc.inner = smallCube →
  cubesWithTwoColoredSides hc tinyCube.side = 84 := by
  sorry

#check two_colored_cubes_count

end NUMINAMATH_CALUDE_two_colored_cubes_count_l1450_145011


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l1450_145020

def f (x : ℝ) : ℝ := 4*x^4 + 17*x^3 - 37*x^2 + 6*x

theorem roots_of_polynomial :
  ∃ (a b c d : ℝ),
    (a = 0) ∧
    (b = 1/2) ∧
    (c = (-9 + Real.sqrt 129) / 4) ∧
    (d = (-9 - Real.sqrt 129) / 4) ∧
    (∀ x : ℝ, f x = 0 ↔ (x = a ∨ x = b ∨ x = c ∨ x = d)) :=
by sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l1450_145020


namespace NUMINAMATH_CALUDE_sequence_formula_l1450_145025

/-- Given a sequence {a_n} defined by a₁ = 2 and a_{n+1} = a_n + ln(1 + 1/n) for n ≥ 1,
    prove that a_n = 2 + ln(n) for all n ≥ 1 -/
theorem sequence_formula (a : ℕ → ℝ) (h1 : a 1 = 2) 
    (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + Real.log (1 + 1 / n)) :
    ∀ n : ℕ, n ≥ 1 → a n = 2 + Real.log n := by
  sorry

end NUMINAMATH_CALUDE_sequence_formula_l1450_145025


namespace NUMINAMATH_CALUDE_sin_product_seventh_pi_l1450_145007

theorem sin_product_seventh_pi : 
  Real.sin (π / 7) * Real.sin (2 * π / 7) * Real.sin (3 * π / 7) = Real.sqrt 13 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sin_product_seventh_pi_l1450_145007


namespace NUMINAMATH_CALUDE_initial_girls_count_l1450_145010

theorem initial_girls_count (initial_boys : ℕ) (new_girls : ℕ) (total_pupils : ℕ) 
  (h1 : initial_boys = 222)
  (h2 : new_girls = 418)
  (h3 : total_pupils = 1346)
  : ∃ initial_girls : ℕ, initial_girls + initial_boys + new_girls = total_pupils ∧ initial_girls = 706 := by
  sorry

end NUMINAMATH_CALUDE_initial_girls_count_l1450_145010


namespace NUMINAMATH_CALUDE_quadratic_minimum_l1450_145018

/-- 
Given a quadratic function y = ax² + px + q where a ≠ 0,
if the minimum value of y is m, then q = m + p²/(4a)
-/
theorem quadratic_minimum (a p q m : ℝ) (ha : a ≠ 0) :
  (∀ x, a * x^2 + p * x + q ≥ m) →
  (∃ x₀, a * x₀^2 + p * x₀ + q = m) →
  q = m + p^2 / (4 * a) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l1450_145018


namespace NUMINAMATH_CALUDE_mollys_age_l1450_145058

theorem mollys_age (sandy_age molly_age : ℕ) : 
  sandy_age = 42 → 
  sandy_age * 9 = molly_age * 7 → 
  molly_age = 54 := by
sorry

end NUMINAMATH_CALUDE_mollys_age_l1450_145058


namespace NUMINAMATH_CALUDE_speed_conversion_l1450_145080

/-- Converts meters per second to kilometers per hour -/
def mps_to_kmph (speed_mps : ℝ) : ℝ := speed_mps * 3.6

/-- Theorem: A speed of 70.0056 meters per second is equivalent to 252.02016 kilometers per hour -/
theorem speed_conversion : mps_to_kmph 70.0056 = 252.02016 := by
  sorry

end NUMINAMATH_CALUDE_speed_conversion_l1450_145080


namespace NUMINAMATH_CALUDE_cube_surface_area_l1450_145064

theorem cube_surface_area (x : ℝ) (h : x > 0) :
  let volume := x^3
  let side_length := x
  let surface_area := 6 * side_length^2
  volume = x^3 → surface_area = 6 * x^2 := by
sorry

end NUMINAMATH_CALUDE_cube_surface_area_l1450_145064


namespace NUMINAMATH_CALUDE_evaluate_expression_l1450_145041

theorem evaluate_expression : Real.sqrt ((4 / 3) * (1 / 15 + 1 / 25)) = 4 * Real.sqrt 2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1450_145041


namespace NUMINAMATH_CALUDE_jamie_marbles_l1450_145035

theorem jamie_marbles (n : ℕ) : 
  n > 0 ∧ 
  (2 * n) % 5 = 0 ∧ 
  n % 3 = 0 ∧ 
  n ≥ 15 → 
  ∃ (blue red green yellow : ℕ), 
    blue = 2 * n / 5 ∧
    red = n / 3 ∧
    green = 4 ∧
    yellow = n - (blue + red + green) ∧
    yellow ≥ 0 ∧
    ∀ (m : ℕ), m < n → 
      (2 * m) % 5 = 0 → 
      m % 3 = 0 → 
      m - (2 * m / 5 + m / 3 + 4) < 0 :=
by sorry

end NUMINAMATH_CALUDE_jamie_marbles_l1450_145035


namespace NUMINAMATH_CALUDE_f_max_at_neg_two_l1450_145042

-- Define the function
def f (x : ℝ) : ℝ := -2 * x^2 - 8 * x + 16

-- State the theorem
theorem f_max_at_neg_two :
  ∀ x : ℝ, f x ≤ f (-2) :=
by
  sorry

end NUMINAMATH_CALUDE_f_max_at_neg_two_l1450_145042


namespace NUMINAMATH_CALUDE_sequence_periodicity_l1450_145013

def is_periodic (a : ℕ → ℤ) : Prop :=
  ∃ p : ℕ, p > 0 ∧ ∀ n : ℕ, a (n + p) = a n

theorem sequence_periodicity (a : ℕ → ℤ) 
  (h : ∀ n : ℕ, n ≥ 2 → (0 : ℝ) ≤ a (n - 1) + ((1 - Real.sqrt 5) / 2) * (a n) + a (n + 1) ∧
                       a (n - 1) + ((1 - Real.sqrt 5) / 2) * (a n) + a (n + 1) < 1) :
  is_periodic a :=
sorry

end NUMINAMATH_CALUDE_sequence_periodicity_l1450_145013


namespace NUMINAMATH_CALUDE_monotone_decreasing_implies_a_bound_l1450_145008

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

theorem monotone_decreasing_implies_a_bound (a : ℝ) :
  (∀ x y : ℝ, x < y ∧ y ≤ 4 → f a x > f a y) → a ≤ -3 := by
  sorry

end NUMINAMATH_CALUDE_monotone_decreasing_implies_a_bound_l1450_145008


namespace NUMINAMATH_CALUDE_triangle_side_length_l1450_145099

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : Real)
  (x y z : Real)

-- State the theorem
theorem triangle_side_length 
  (t : Triangle) 
  (h1 : t.y = 7) 
  (h2 : t.z = 6) 
  (h3 : Real.cos (t.Y - t.Z) = 15/16) : 
  t.x = Real.sqrt 22 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1450_145099


namespace NUMINAMATH_CALUDE_same_combination_probability_is_correct_l1450_145065

def jar_candies : ℕ × ℕ := (12, 8)

def total_candies : ℕ := jar_candies.1 + jar_candies.2

def same_combination_probability : ℚ :=
  let terry_picks := Nat.choose total_candies 2
  let mary_picks := Nat.choose (total_candies - 2) 2
  let both_red := (Nat.choose jar_candies.1 2 * Nat.choose (jar_candies.1 - 2) 2) / (terry_picks * mary_picks)
  let both_blue := (Nat.choose jar_candies.2 2 * Nat.choose (jar_candies.2 - 2) 2) / (terry_picks * mary_picks)
  let mixed := (Nat.choose jar_candies.1 1 * Nat.choose jar_candies.2 1 * 
                Nat.choose (jar_candies.1 - 1) 1 * Nat.choose (jar_candies.2 - 1) 1) / 
               (terry_picks * mary_picks)
  both_red + both_blue + mixed

theorem same_combination_probability_is_correct : 
  same_combination_probability = 143 / 269 := by
  sorry

end NUMINAMATH_CALUDE_same_combination_probability_is_correct_l1450_145065


namespace NUMINAMATH_CALUDE_cryptarithm_solution_l1450_145039

theorem cryptarithm_solution (A B C : ℕ) : 
  A ≠ 0 ∧ 
  A ≠ B ∧ A ≠ C ∧ B ≠ C ∧
  A < 10 ∧ B < 10 ∧ C < 10 ∧
  100 * A + 10 * B + C - (10 * B + C) = 100 * A + A → 
  C = 9 := by
sorry

end NUMINAMATH_CALUDE_cryptarithm_solution_l1450_145039


namespace NUMINAMATH_CALUDE_modular_inverse_of_7_mod_26_l1450_145015

theorem modular_inverse_of_7_mod_26 : ∃ x : ℕ, x ∈ Finset.range 26 ∧ (7 * x) % 26 = 1 := by
  use 15
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_7_mod_26_l1450_145015


namespace NUMINAMATH_CALUDE_flea_market_spending_l1450_145087

/-- Given that Jayda spent $400 and Aitana spent 2/5 times more than Jayda,
    prove that the total amount they spent together is $960. -/
theorem flea_market_spending (jayda_spent : ℝ) (aitana_ratio : ℝ) : 
  jayda_spent = 400 → 
  aitana_ratio = 2/5 → 
  jayda_spent + (jayda_spent + aitana_ratio * jayda_spent) = 960 := by
sorry

end NUMINAMATH_CALUDE_flea_market_spending_l1450_145087


namespace NUMINAMATH_CALUDE_manipulation_function_l1450_145093

theorem manipulation_function (f : ℤ → ℤ) (h : 3 * (f 19 + 5) = 129) :
  ∀ x : ℤ, f x = 2 * x := by
  sorry

end NUMINAMATH_CALUDE_manipulation_function_l1450_145093


namespace NUMINAMATH_CALUDE_divisibility_by_100_l1450_145021

theorem divisibility_by_100 (a : ℕ) (h : ¬(5 ∣ a)) : 100 ∣ (a^8 + 3*a^4 - 4) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_100_l1450_145021


namespace NUMINAMATH_CALUDE_exists_special_sequence_l1450_145024

/-- A sequence of natural numbers satisfying specific conditions -/
def SpecialSequence (F : ℕ → ℕ) : Prop :=
  (∀ k, ∃ n, F n = k) ∧
  (∀ k, Set.Infinite {n | F n = k}) ∧
  (∀ n ≥ 2, F (F (n^163)) = F (F n) + F (F 361))

/-- There exists a sequence satisfying the SpecialSequence conditions -/
theorem exists_special_sequence : ∃ F, SpecialSequence F := by
  sorry

end NUMINAMATH_CALUDE_exists_special_sequence_l1450_145024


namespace NUMINAMATH_CALUDE_units_digit_problem_l1450_145082

theorem units_digit_problem : ∃ n : ℕ, (8 * 13 * 1989 - 8^3) % 10 = 4 ∧ n * 10 ≤ 8 * 13 * 1989 - 8^3 ∧ 8 * 13 * 1989 - 8^3 < (n + 1) * 10 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_problem_l1450_145082


namespace NUMINAMATH_CALUDE_smallest_value_for_y_between_zero_and_one_l1450_145092

theorem smallest_value_for_y_between_zero_and_one
  (y : ℝ) (h : 0 < y ∧ y < 1) :
  y^3 < y ∧ y^3 < 3*y ∧ y^3 < y^(1/3) ∧ y^3 < 1/y :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_for_y_between_zero_and_one_l1450_145092


namespace NUMINAMATH_CALUDE_exists_periodic_nonconstant_sequence_l1450_145060

def isPeriodicSequence (x : ℕ → ℤ) : Prop :=
  ∃ T : ℕ, T > 0 ∧ ∀ n, x (n + T) = x n

def satisfiesRecurrence (x : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → x (n + 1) = 2 * x n + 3 * x (n - 1)

def isConstant (x : ℕ → ℤ) : Prop :=
  ∀ m n : ℕ, x m = x n

theorem exists_periodic_nonconstant_sequence :
  ∃ x : ℕ → ℤ, satisfiesRecurrence x ∧ isPeriodicSequence x ∧ ¬isConstant x := by
  sorry

end NUMINAMATH_CALUDE_exists_periodic_nonconstant_sequence_l1450_145060


namespace NUMINAMATH_CALUDE_cow_count_theorem_l1450_145081

/-- The number of cows on a dairy farm -/
def number_of_cows : ℕ := 20

/-- The number of bags of husk eaten by some cows in 20 days -/
def total_bags_eaten : ℕ := 20

/-- The number of bags of husk eaten by one cow in 20 days -/
def bags_per_cow : ℕ := 1

/-- Theorem stating that the number of cows is equal to the total bags eaten divided by the bags eaten per cow -/
theorem cow_count_theorem : number_of_cows = total_bags_eaten / bags_per_cow := by
  sorry

end NUMINAMATH_CALUDE_cow_count_theorem_l1450_145081


namespace NUMINAMATH_CALUDE_range_of_f_l1450_145049

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2*x + 3

-- State the theorem
theorem range_of_f :
  {y : ℝ | ∃ x ≥ 0, f x = y} = {y : ℝ | y ≥ 3} := by
  sorry

end NUMINAMATH_CALUDE_range_of_f_l1450_145049


namespace NUMINAMATH_CALUDE_max_true_statements_l1450_145084

theorem max_true_statements (x : ℝ) : 
  let statements := [
    (0 < x^2 ∧ x^2 < 1),
    (x^2 > 1),
    (-1 < x ∧ x < 0),
    (0 < x ∧ x < 1),
    (0 < x - x^3 ∧ x - x^3 < 1)
  ]
  ¬∃ (s : Finset (Fin 5)), s.card > 3 ∧ (∀ i ∈ s, statements[i]) :=
by sorry

end NUMINAMATH_CALUDE_max_true_statements_l1450_145084


namespace NUMINAMATH_CALUDE_blue_highlighters_count_l1450_145029

theorem blue_highlighters_count (pink : ℕ) (yellow : ℕ) (total : ℕ) (blue : ℕ) :
  pink = 6 → yellow = 2 → total = 12 → blue = total - (pink + yellow) → blue = 4 := by
  sorry

end NUMINAMATH_CALUDE_blue_highlighters_count_l1450_145029


namespace NUMINAMATH_CALUDE_only_seven_satisfies_inequality_l1450_145071

theorem only_seven_satisfies_inequality :
  ∃! (n : ℤ), (3 : ℚ) / 10 < (n : ℚ) / 20 ∧ (n : ℚ) / 20 < 2 / 5 :=
by
  sorry

end NUMINAMATH_CALUDE_only_seven_satisfies_inequality_l1450_145071


namespace NUMINAMATH_CALUDE_good_apples_count_l1450_145000

theorem good_apples_count (total_apples unripe_apples : ℕ) 
  (h1 : total_apples = 14) 
  (h2 : unripe_apples = 6) : 
  total_apples - unripe_apples = 8 := by
sorry

end NUMINAMATH_CALUDE_good_apples_count_l1450_145000


namespace NUMINAMATH_CALUDE_leo_marbles_l1450_145095

theorem leo_marbles (total_marbles : ℕ) (marbles_per_pack : ℕ) 
  (manny_fraction : ℚ) (neil_fraction : ℚ) : 
  total_marbles = 400 →
  marbles_per_pack = 10 →
  manny_fraction = 1/4 →
  neil_fraction = 1/8 →
  (total_marbles / marbles_per_pack : ℚ) * (1 - manny_fraction - neil_fraction) = 25 := by
  sorry

end NUMINAMATH_CALUDE_leo_marbles_l1450_145095


namespace NUMINAMATH_CALUDE_train_crossing_time_l1450_145088

/-- Proves that a train with given length and speed takes the calculated time to cross a pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 300 →
  train_speed_kmh = 120 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 9 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l1450_145088


namespace NUMINAMATH_CALUDE_expand_expression_l1450_145014

theorem expand_expression (x y z : ℝ) : 
  (x + 10) * (3 * y + 5 * z + 15) = 3 * x * y + 5 * x * z + 15 * x + 30 * y + 50 * z + 150 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1450_145014


namespace NUMINAMATH_CALUDE_min_value_sqrt_sum_squares_l1450_145012

theorem min_value_sqrt_sum_squares (a b m n : ℝ) 
  (h1 : a^2 + b^2 = 3) 
  (h2 : m*a + n*b = 3) : 
  Real.sqrt (m^2 + n^2) ≥ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sqrt_sum_squares_l1450_145012


namespace NUMINAMATH_CALUDE_undefined_slopes_parallel_l1450_145077

-- Define a type for lines
structure Line where
  slope : Option ℝ
  -- Other properties of a line could be added here if needed

-- Define what it means for two lines to be parallel
def parallel (l1 l2 : Line) : Prop :=
  (l1.slope = none ∧ l2.slope = none) ∨ (l1.slope ≠ none ∧ l2.slope ≠ none ∧ l1.slope = l2.slope)

-- Define what it means for two lines to be distinct
def distinct (l1 l2 : Line) : Prop :=
  l1 ≠ l2

-- Theorem statement
theorem undefined_slopes_parallel (l1 l2 : Line) :
  distinct l1 l2 → l1.slope = none → l2.slope = none → parallel l1 l2 :=
by
  sorry


end NUMINAMATH_CALUDE_undefined_slopes_parallel_l1450_145077


namespace NUMINAMATH_CALUDE_oil_bill_difference_l1450_145036

/-- Given the oil bills for January and February, calculate the difference
    between February's bill in two scenarios. -/
theorem oil_bill_difference (jan_bill : ℝ) (feb_ratio1 feb_ratio2 jan_ratio1 jan_ratio2 : ℚ) :
  jan_bill = 120 →
  feb_ratio1 / jan_ratio1 = 5 / 4 →
  feb_ratio2 / jan_ratio2 = 3 / 2 →
  ∃ (feb_bill1 feb_bill2 : ℝ),
    feb_bill1 / jan_bill = feb_ratio1 / jan_ratio1 ∧
    feb_bill2 / jan_bill = feb_ratio2 / jan_ratio2 ∧
    feb_bill2 - feb_bill1 = 30 :=
by sorry

end NUMINAMATH_CALUDE_oil_bill_difference_l1450_145036


namespace NUMINAMATH_CALUDE_time_to_finish_book_l1450_145066

/-- Calculates the time needed to finish a book given the current reading progress and reading speed. -/
theorem time_to_finish_book (total_pages reading_speed current_page : ℕ) 
  (h1 : total_pages = 210)
  (h2 : current_page = 90)
  (h3 : reading_speed = 30) : 
  (total_pages - current_page) / reading_speed = 4 :=
by sorry

end NUMINAMATH_CALUDE_time_to_finish_book_l1450_145066


namespace NUMINAMATH_CALUDE_cubic_is_odd_l1450_145022

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def f (x : ℝ) : ℝ := x^3

theorem cubic_is_odd : is_odd_function f := by
  sorry

end NUMINAMATH_CALUDE_cubic_is_odd_l1450_145022


namespace NUMINAMATH_CALUDE_prob_one_common_is_two_thirds_l1450_145068

/-- The number of elective courses available -/
def num_courses : ℕ := 4

/-- The number of courses each student selects -/
def courses_per_student : ℕ := 2

/-- The total number of ways two students can select their courses -/
def total_selections : ℕ := (num_courses.choose courses_per_student) ^ 2

/-- The number of ways two students can select courses with exactly one in common -/
def one_common_selection : ℕ := num_courses * (num_courses - 1) * (num_courses - 2)

/-- The probability of two students sharing exactly one course in common -/
def prob_one_common : ℚ := one_common_selection / total_selections

theorem prob_one_common_is_two_thirds : prob_one_common = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_prob_one_common_is_two_thirds_l1450_145068


namespace NUMINAMATH_CALUDE_distinct_triangles_in_2x4_grid_l1450_145085

/-- Represents a 2x4 grid of points -/
def Grid := Fin 2 × Fin 4

/-- Checks if three points in the grid are collinear -/
def collinear (p q r : Grid) : Prop := sorry

/-- The number of ways to choose 3 points from 8 points -/
def total_combinations : ℕ := Nat.choose 8 3

/-- The number of collinear triples in a 2x4 grid -/
def collinear_triples : ℕ := sorry

/-- The number of distinct triangles in a 2x4 grid -/
def distinct_triangles : ℕ := total_combinations - collinear_triples

theorem distinct_triangles_in_2x4_grid :
  distinct_triangles = 44 := by sorry

end NUMINAMATH_CALUDE_distinct_triangles_in_2x4_grid_l1450_145085


namespace NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l1450_145073

-- Define the condition "m < 1/4"
def condition (m : ℝ) : Prop := m < (1/4 : ℝ)

-- Define when a quadratic equation has real solutions
def has_real_solutions (a b c : ℝ) : Prop := b^2 - 4*a*c ≥ 0

-- State the theorem
theorem condition_sufficient_not_necessary :
  (∀ m : ℝ, condition m → has_real_solutions 1 1 m) ∧
  (∃ m : ℝ, ¬(condition m) ∧ has_real_solutions 1 1 m) :=
sorry

end NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l1450_145073


namespace NUMINAMATH_CALUDE_measles_cases_1995_l1450_145044

/-- Represents the number of measles cases in a given year -/
def measles_cases (year : ℕ) : ℝ :=
  if year ≤ 1990 then
    300000 - 14950 * (year - 1970)
  else
    -8 * (year - 1990)^2 + 1000

/-- The theorem stating that the number of measles cases in 1995 is 800 -/
theorem measles_cases_1995 : measles_cases 1995 = 800 := by
  sorry

end NUMINAMATH_CALUDE_measles_cases_1995_l1450_145044


namespace NUMINAMATH_CALUDE_power_of_power_l1450_145048

theorem power_of_power : (3^2)^4 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l1450_145048


namespace NUMINAMATH_CALUDE_negative_f_m_plus_one_l1450_145019

theorem negative_f_m_plus_one 
  (f : ℝ → ℝ) 
  (a m : ℝ) 
  (h1 : ∀ x, f x = x^2 - x + a) 
  (h2 : f (-m) < 0) : 
  f (m + 1) < 0 := by
sorry

end NUMINAMATH_CALUDE_negative_f_m_plus_one_l1450_145019


namespace NUMINAMATH_CALUDE_parallelogram_side_length_l1450_145078

-- Define a parallelogram
structure Parallelogram :=
  (A B C D : ℝ × ℝ)

-- Define the length function
def length (p q : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem parallelogram_side_length 
  (ABCD : Parallelogram) 
  (x : ℝ) 
  (h1 : length ABCD.A ABCD.B = x + 3)
  (h2 : length ABCD.B ABCD.C = x - 4)
  (h3 : length ABCD.C ABCD.D = 16) :
  length ABCD.A ABCD.D = 9 := by sorry

end NUMINAMATH_CALUDE_parallelogram_side_length_l1450_145078


namespace NUMINAMATH_CALUDE_at_least_one_half_l1450_145030

theorem at_least_one_half (x y z : ℝ) 
  (h : x + y + z - 2 * (x * y + y * z + x * z) + 4 * x * y * z = (1 : ℝ) / 2) : 
  x = (1 : ℝ) / 2 ∨ y = (1 : ℝ) / 2 ∨ z = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_half_l1450_145030


namespace NUMINAMATH_CALUDE_inequality_proof_l1450_145002

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  let P := Real.sqrt ((a^2 + b^2)/2) - (a + b)/2
  let Q := (a + b)/2 - Real.sqrt (a*b)
  let R := Real.sqrt (a*b) - (2*a*b)/(a + b)
  Q ≥ P ∧ P ≥ R := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1450_145002


namespace NUMINAMATH_CALUDE_equation1_solutions_equation2_solutions_l1450_145040

-- Define the quadratic equations
def equation1 (x : ℝ) : Prop := x^2 + 4*x - 1 = 0
def equation2 (x : ℝ) : Prop := (x-2)^2 - 3*x*(x-2) = 0

-- Theorem for the first equation
theorem equation1_solutions :
  ∃ x1 x2 : ℝ, x1 = -2 + Real.sqrt 5 ∧ x2 = -2 - Real.sqrt 5 ∧
  equation1 x1 ∧ equation1 x2 ∧
  ∀ x : ℝ, equation1 x → x = x1 ∨ x = x2 :=
sorry

-- Theorem for the second equation
theorem equation2_solutions :
  ∃ x1 x2 : ℝ, x1 = 2 ∧ x2 = -1 ∧
  equation2 x1 ∧ equation2 x2 ∧
  ∀ x : ℝ, equation2 x → x = x1 ∨ x = x2 :=
sorry

end NUMINAMATH_CALUDE_equation1_solutions_equation2_solutions_l1450_145040


namespace NUMINAMATH_CALUDE_circle_area_increase_l1450_145031

theorem circle_area_increase (r : ℝ) (h : r > 0) :
  let new_radius := 1.5 * r
  let original_area := π * r^2
  let new_area := π * new_radius^2
  (new_area - original_area) / original_area = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_increase_l1450_145031


namespace NUMINAMATH_CALUDE_power_division_nineteen_l1450_145001

theorem power_division_nineteen : (19 : ℕ)^11 / (19 : ℕ)^8 = 6859 := by sorry

end NUMINAMATH_CALUDE_power_division_nineteen_l1450_145001


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1450_145003

/-- Given that z₁ = -1 + i and z₁z₂ = -2, prove that |z₂ + 2i| = √10 -/
theorem complex_modulus_problem (z₁ z₂ : ℂ) : 
  z₁ = -1 + Complex.I → z₁ * z₂ = -2 → Complex.abs (z₂ + 2 * Complex.I) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1450_145003


namespace NUMINAMATH_CALUDE_cube_minus_reciprocal_cube_l1450_145096

theorem cube_minus_reciprocal_cube (x : ℝ) (h : x - 1/x = 5) : x^3 - 1/x^3 = 140 := by
  sorry

end NUMINAMATH_CALUDE_cube_minus_reciprocal_cube_l1450_145096


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l1450_145076

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (x : ℕ), x < d ∧ (n - x) % d = 0 ∧ ∀ (y : ℕ), y < x → (n - y) % d ≠ 0 :=
by
  sorry

theorem problem_solution :
  ∃ (x : ℕ), x = 2 ∧ (427398 - x) % 13 = 0 ∧ ∀ (y : ℕ), y < x → (427398 - y) % 13 ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l1450_145076


namespace NUMINAMATH_CALUDE_student_ticket_price_l1450_145055

theorem student_ticket_price 
  (total_tickets : ℕ) 
  (student_tickets : ℕ) 
  (non_student_tickets : ℕ) 
  (non_student_price : ℚ) 
  (total_revenue : ℚ) :
  total_tickets = 150 →
  student_tickets = 90 →
  non_student_tickets = 60 →
  non_student_price = 8 →
  total_revenue = 930 →
  ∃ (student_price : ℚ), 
    student_price * student_tickets + non_student_price * non_student_tickets = total_revenue ∧
    student_price = 5 := by
  sorry

end NUMINAMATH_CALUDE_student_ticket_price_l1450_145055


namespace NUMINAMATH_CALUDE_garden_length_is_50_l1450_145072

def garden_length (width : ℝ) : ℝ := 2 * width

def garden_perimeter (width : ℝ) : ℝ := 2 * garden_length width + 2 * width

theorem garden_length_is_50 :
  ∃ (width : ℝ), garden_perimeter width = 150 ∧ garden_length width = 50 :=
by sorry

end NUMINAMATH_CALUDE_garden_length_is_50_l1450_145072


namespace NUMINAMATH_CALUDE_salaria_trees_count_l1450_145045

/-- Represents the total number of trees Salaria has -/
def total_trees : ℕ := sorry

/-- Represents the number of oranges tree A produces per month -/
def tree_A_oranges : ℕ := 10

/-- Represents the number of oranges tree B produces per month -/
def tree_B_oranges : ℕ := 15

/-- Represents the fraction of good oranges from tree A -/
def tree_A_good_fraction : ℚ := 3/5

/-- Represents the fraction of good oranges from tree B -/
def tree_B_good_fraction : ℚ := 1/3

/-- Represents the total number of good oranges Salaria gets per month -/
def total_good_oranges : ℕ := 55

theorem salaria_trees_count :
  total_trees = 10 ∧
  (total_trees / 2 : ℚ) * tree_A_oranges * tree_A_good_fraction +
  (total_trees / 2 : ℚ) * tree_B_oranges * tree_B_good_fraction = total_good_oranges := by
  sorry

end NUMINAMATH_CALUDE_salaria_trees_count_l1450_145045


namespace NUMINAMATH_CALUDE_u_v_cube_sum_l1450_145070

theorem u_v_cube_sum (u v : ℝ) (hu : u > 1) (hv : v > 1)
  (h : Real.log u / Real.log 4 ^ 3 + Real.log v / Real.log 5 ^ 3 + 9 = 
       9 * (Real.log u / Real.log 4) * (Real.log v / Real.log 5)) :
  u^3 + v^3 = 4^(9/2) + 5^(9/2) := by
sorry

end NUMINAMATH_CALUDE_u_v_cube_sum_l1450_145070


namespace NUMINAMATH_CALUDE_tom_teaching_years_l1450_145097

theorem tom_teaching_years (tom devin : ℕ) 
  (h1 : tom + devin = 70)
  (h2 : devin = tom / 2 - 5) :
  tom = 50 := by
  sorry

end NUMINAMATH_CALUDE_tom_teaching_years_l1450_145097


namespace NUMINAMATH_CALUDE_dans_remaining_money_l1450_145051

def dans_money_left (initial_amount spent_on_candy spent_on_chocolate : ℕ) : ℕ :=
  initial_amount - (spent_on_candy + spent_on_chocolate)

theorem dans_remaining_money :
  dans_money_left 7 2 3 = 2 := by sorry

end NUMINAMATH_CALUDE_dans_remaining_money_l1450_145051


namespace NUMINAMATH_CALUDE_analects_reasoning_is_common_sense_l1450_145098

-- Define the types of reasoning
inductive ReasoningType
  | CommonSense
  | Inductive
  | Analogical
  | Deductive

-- Define the structure of a statement in the reasoning chain
structure Statement :=
  (premise : String)
  (conclusion : String)

-- Define the passage from The Analects as a list of statements
def analectsPassage : List Statement :=
  [⟨"Names are not correct", "Language will not be in accordance with the truth of things"⟩,
   ⟨"Language is not in accordance with the truth of things", "Affairs cannot be carried out successfully"⟩,
   ⟨"Affairs cannot be carried out successfully", "Rites and music will not flourish"⟩,
   ⟨"Rites and music do not flourish", "Punishments will not be properly executed"⟩,
   ⟨"Punishments are not properly executed", "The people will have nowhere to put their hands and feet"⟩]

-- Define a function to determine the type of reasoning
def determineReasoningType (passage : List Statement) : ReasoningType := sorry

-- Theorem stating that the reasoning in the Analects passage is common sense reasoning
theorem analects_reasoning_is_common_sense :
  determineReasoningType analectsPassage = ReasoningType.CommonSense := by sorry

end NUMINAMATH_CALUDE_analects_reasoning_is_common_sense_l1450_145098


namespace NUMINAMATH_CALUDE_min_sum_squares_exists_min_sum_squares_l1450_145026

def S : Finset ℤ := {3, -5, 0, 9, -2}

theorem min_sum_squares (a b c : ℤ) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  13 ≤ a^2 + b^2 + c^2 :=
by sorry

theorem exists_min_sum_squares :
  ∃ (a b c : ℤ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a^2 + b^2 + c^2 = 13 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_exists_min_sum_squares_l1450_145026


namespace NUMINAMATH_CALUDE_least_number_with_remainder_l1450_145004

theorem least_number_with_remainder (n : ℕ) : n = 282 ↔ 
  (n > 0 ∧ 
   n % 31 = 3 ∧ 
   n % 9 = 3 ∧ 
   ∀ m : ℕ, m > 0 → m % 31 = 3 → m % 9 = 3 → m ≥ n) :=
by sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_l1450_145004
