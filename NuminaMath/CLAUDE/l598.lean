import Mathlib

namespace NUMINAMATH_CALUDE_fraction_states_1800_1809_l598_59833

/-- The number of states that joined the union from 1800 to 1809 -/
def states_1800_1809 : ℕ := 5

/-- The total number of states considered (first 30 states) -/
def total_states : ℕ := 30

/-- The fraction of states that joined from 1800 to 1809 -/
def fraction_1800_1809 : ℚ := states_1800_1809 / total_states

theorem fraction_states_1800_1809 : fraction_1800_1809 = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_states_1800_1809_l598_59833


namespace NUMINAMATH_CALUDE_parabola_properties_l598_59853

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 - 2*x - 3

-- Define the conditions
theorem parabola_properties :
  (parabola (-1) = 0) ∧
  (parabola 3 = 0) ∧
  (parabola 0 = -3) ∧
  (∃ (a b c : ℝ), ∀ x, parabola x = a * x^2 + b * x + c) ∧
  (let vertex := (1, -4);
   parabola vertex.1 = vertex.2 ∧
   ∀ x, parabola x ≥ parabola vertex.1) ∧
  (∀ x₁ x₂ y₁ y₂, 
    x₁ < x₂ → x₂ < 1 → 
    parabola x₁ = y₁ → parabola x₂ = y₂ → 
    y₁ < y₂) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l598_59853


namespace NUMINAMATH_CALUDE_triangle_properties_l598_59814

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The perimeter of the triangle is √2 + 1 -/
def perimeter_condition (t : Triangle) : Prop :=
  t.a + t.b + t.c = Real.sqrt 2 + 1

/-- The sum of sines condition -/
def sine_sum_condition (t : Triangle) : Prop :=
  Real.sin t.A + Real.sin t.B = Real.sqrt 2 * Real.sin t.C

/-- The area of the triangle is (1/6) * sin C -/
def area_condition (t : Triangle) : Prop :=
  (1/2) * t.a * t.b * Real.sin t.C = (1/6) * Real.sin t.C

theorem triangle_properties (t : Triangle) 
  (h_perimeter : perimeter_condition t)
  (h_sine_sum : sine_sum_condition t)
  (h_area : area_condition t) :
  t.c = 1 ∧ t.C = π/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l598_59814


namespace NUMINAMATH_CALUDE_ella_sold_200_apples_l598_59872

/-- The number of apples Ella sold -/
def apples_sold (bags_of_20 bags_of_25 apples_per_bag_20 apples_per_bag_25 apples_left : ℕ) : ℕ :=
  bags_of_20 * apples_per_bag_20 + bags_of_25 * apples_per_bag_25 - apples_left

/-- Theorem stating that Ella sold 200 apples -/
theorem ella_sold_200_apples :
  apples_sold 4 6 20 25 30 = 200 := by
  sorry

end NUMINAMATH_CALUDE_ella_sold_200_apples_l598_59872


namespace NUMINAMATH_CALUDE_train_speed_calculation_l598_59866

/-- Given a train of length 140 meters passing a platform of length 260 meters in 23.998080153587715 seconds,
    prove that the speed of the train is 60.0048 kilometers per hour. -/
theorem train_speed_calculation (train_length platform_length time_to_pass : ℝ)
    (h1 : train_length = 140)
    (h2 : platform_length = 260)
    (h3 : time_to_pass = 23.998080153587715) :
    (train_length + platform_length) / time_to_pass * 3.6 = 60.0048 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l598_59866


namespace NUMINAMATH_CALUDE_functional_equation_solution_l598_59812

-- Define the property that the function must satisfy
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * f x + f y) = f x ^ 2 + y

-- State the theorem
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, SatisfiesFunctionalEquation f →
    (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l598_59812


namespace NUMINAMATH_CALUDE_fraction_simplification_l598_59840

theorem fraction_simplification :
  (15 : ℚ) / 35 * 28 / 45 * 75 / 28 = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l598_59840


namespace NUMINAMATH_CALUDE_binary_110101_equals_53_l598_59877

/-- Converts a binary number represented as a list of bits (least significant bit first) to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 110101 -/
def binary_110101 : List Bool := [true, false, true, false, true, true]

theorem binary_110101_equals_53 :
  binary_to_decimal binary_110101 = 53 := by
  sorry

end NUMINAMATH_CALUDE_binary_110101_equals_53_l598_59877


namespace NUMINAMATH_CALUDE_angle_complement_l598_59875

/-- 
Given an angle x and its complement y, prove that y equals 90° minus x.
-/
theorem angle_complement (x y : ℝ) (h : x + y = 90) : y = 90 - x := by
  sorry

end NUMINAMATH_CALUDE_angle_complement_l598_59875


namespace NUMINAMATH_CALUDE_shaded_area_square_with_quarter_circles_l598_59816

/-- The area of the shaded region inside a square with quarter circles at its corners -/
theorem shaded_area_square_with_quarter_circles 
  (square_side : ℝ) 
  (circle_radius : ℝ) 
  (h1 : square_side = 15) 
  (h2 : circle_radius = square_side / 3) :
  square_side ^ 2 - π * circle_radius ^ 2 = 225 - 25 * π := by
sorry

end NUMINAMATH_CALUDE_shaded_area_square_with_quarter_circles_l598_59816


namespace NUMINAMATH_CALUDE_triangle_side_length_l598_59819

/-- Theorem: In a triangle ABC, if c + b = 12, A = 60°, and B = 30°, then c = 8 -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  c + b = 12 → A = 60 → B = 30 → c = 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l598_59819


namespace NUMINAMATH_CALUDE_parallel_lines_count_parallel_lines_problem_l598_59857

/-- Given two sets of intersecting parallel lines, the number of parallelograms formed is the product of the spaces between the lines in each set. -/
def parallelogram_count (lines_set1 lines_set2 : ℕ) : ℕ := (lines_set1 - 1) * (lines_set2 - 1)

/-- The problem statement -/
theorem parallel_lines_count (lines_set1 : ℕ) (parallelograms : ℕ) : ℕ :=
  let lines_set2 := (parallelograms / (lines_set1 - 1)) + 1
  lines_set2

/-- The main theorem to prove -/
theorem parallel_lines_problem :
  parallel_lines_count 6 420 = 85 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_count_parallel_lines_problem_l598_59857


namespace NUMINAMATH_CALUDE_sum_always_six_digits_l598_59850

def first_number : Nat := 98765

def second_number (C : Nat) : Nat := C * 1000 + 433

def third_number (D : Nat) : Nat := D * 100 + 22

def is_nonzero_digit (n : Nat) : Prop := 1 ≤ n ∧ n ≤ 9

theorem sum_always_six_digits (C D : Nat) 
  (hC : is_nonzero_digit C) (hD : is_nonzero_digit D) : 
  ∃ (n : Nat), 100000 ≤ first_number + second_number C + third_number D ∧ 
               first_number + second_number C + third_number D < 1000000 :=
sorry

end NUMINAMATH_CALUDE_sum_always_six_digits_l598_59850


namespace NUMINAMATH_CALUDE_right_side_exponent_l598_59834

theorem right_side_exponent (s : ℝ) : 
  (2^16 : ℝ) * (25^s) = 5 * (10^16) → 16 = 16 := by sorry

end NUMINAMATH_CALUDE_right_side_exponent_l598_59834


namespace NUMINAMATH_CALUDE_profit_maximum_at_five_l598_59841

/-- Profit function parameters -/
def a : ℝ := -10
def b : ℝ := 100
def c : ℝ := 2000

/-- Profit function -/
def profit_function (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The point where the maximum profit occurs -/
def max_profit_point : ℝ := 5

theorem profit_maximum_at_five :
  ∀ x : ℝ, profit_function x ≤ profit_function max_profit_point :=
by sorry


end NUMINAMATH_CALUDE_profit_maximum_at_five_l598_59841


namespace NUMINAMATH_CALUDE_books_returned_count_l598_59818

/-- Represents the number of books Mary has at different stages --/
structure BookCount where
  initial : Nat
  after_first_return : Nat
  after_second_checkout : Nat
  final : Nat

/-- Represents Mary's library transactions --/
def library_transactions (x : Nat) : BookCount :=
  { initial := 5,
    after_first_return := 5 - x + 5,
    after_second_checkout := 5 - x + 5 - 2 + 7,
    final := 12 }

/-- Theorem stating the number of books Mary returned --/
theorem books_returned_count : ∃ x : Nat, 
  (library_transactions x).final = 12 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_books_returned_count_l598_59818


namespace NUMINAMATH_CALUDE_line_slope_l598_59880

theorem line_slope (x y : ℝ) (h : x / 4 + y / 3 = 1) : 
  ∃ m b : ℝ, y = m * x + b ∧ m = -3/4 := by
sorry

end NUMINAMATH_CALUDE_line_slope_l598_59880


namespace NUMINAMATH_CALUDE_absolute_value_equation_solutions_l598_59863

theorem absolute_value_equation_solutions :
  ∃! (s : Finset ℝ), s.card = 2 ∧ 
  (∀ x : ℝ, x ∈ s ↔ |x - 1| = |x - 2| + |x - 3| + |x - 4|) ∧
  (3 ∈ s ∧ 4 ∈ s) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solutions_l598_59863


namespace NUMINAMATH_CALUDE_divisor_problem_l598_59822

theorem divisor_problem (n d : ℕ) (h1 : n % d = 3) (h2 : (n^2) % d = 3) : d = 3 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l598_59822


namespace NUMINAMATH_CALUDE_shoe_price_calculation_l598_59825

theorem shoe_price_calculation (initial_price : ℝ) 
  (price_increase_percent : ℝ) (discount_percent : ℝ) (tax_percent : ℝ) : 
  initial_price = 50 ∧ 
  price_increase_percent = 20 ∧ 
  discount_percent = 15 ∧ 
  tax_percent = 5 → 
  initial_price * (1 + price_increase_percent / 100) * 
  (1 - discount_percent / 100) * (1 + tax_percent / 100) = 53.55 := by
sorry

end NUMINAMATH_CALUDE_shoe_price_calculation_l598_59825


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l598_59813

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 2/5) 
  (h2 : x - y = 1/10) : 
  x^2 - y^2 = 1/25 := by
sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l598_59813


namespace NUMINAMATH_CALUDE_shane_current_age_l598_59836

/-- Given that twenty years ago Shane was 2 times older than Garret is now,
    and Garret is currently 12 years old, prove that Shane is 44 years old now. -/
theorem shane_current_age :
  (∀ (shane_age_now garret_age_now : ℕ),
    garret_age_now = 12 →
    shane_age_now - 20 = 2 * garret_age_now →
    shane_age_now = 44) :=
by sorry

end NUMINAMATH_CALUDE_shane_current_age_l598_59836


namespace NUMINAMATH_CALUDE_inequality_abc_at_least_one_positive_l598_59817

-- Problem 1
theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a + b + c ≥ Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a) := by
  sorry

-- Problem 2
theorem at_least_one_positive (x y z : ℝ) :
  let a := x^2 - 2*y + π/2
  let b := y^2 - 2*z + π/3
  let c := z^2 - 2*x + π/6
  0 < a ∨ 0 < b ∨ 0 < c := by
  sorry

end NUMINAMATH_CALUDE_inequality_abc_at_least_one_positive_l598_59817


namespace NUMINAMATH_CALUDE_olivias_wallet_l598_59839

/-- Calculates the remaining money in Olivia's wallet after shopping --/
def remaining_money (initial : ℕ) (supermarket : ℕ) (showroom : ℕ) : ℕ :=
  initial - supermarket - showroom

/-- Theorem stating that Olivia has 26 dollars left after shopping --/
theorem olivias_wallet : remaining_money 106 31 49 = 26 := by
  sorry

end NUMINAMATH_CALUDE_olivias_wallet_l598_59839


namespace NUMINAMATH_CALUDE_initial_children_on_bus_proof_initial_children_l598_59831

theorem initial_children_on_bus : ℕ → Prop :=
  fun initial_children =>
    initial_children + 7 = 25

theorem proof_initial_children : 
  ∃ initial_children : ℕ, initial_children_on_bus initial_children ∧ initial_children = 18 := by
  sorry

end NUMINAMATH_CALUDE_initial_children_on_bus_proof_initial_children_l598_59831


namespace NUMINAMATH_CALUDE_red_balls_in_stratified_sample_l598_59852

/-- Calculates the number of red balls to be sampled in a stratified sampling by color -/
def stratifiedSampleRedBalls (totalPopulation : ℕ) (totalRedBalls : ℕ) (sampleSize : ℕ) : ℕ :=
  (totalRedBalls * sampleSize) / totalPopulation

/-- Theorem: The number of red balls in a stratified sample of 100 from 1000 balls with 50 red balls is 5 -/
theorem red_balls_in_stratified_sample :
  stratifiedSampleRedBalls 1000 50 100 = 5 := by
  sorry

end NUMINAMATH_CALUDE_red_balls_in_stratified_sample_l598_59852


namespace NUMINAMATH_CALUDE_specialNumberCount_is_70_l598_59862

/-- The count of numbers between 200 and 899 (inclusive) with three different digits 
    that can be arranged in either strictly increasing or strictly decreasing order -/
def specialNumberCount : ℕ :=
  let lowerBound := 200
  let upperBound := 899
  let digitSet := {2, 3, 4, 5, 6, 7, 8}
  2 * (Finset.card digitSet).choose 3

theorem specialNumberCount_is_70 : specialNumberCount = 70 := by
  sorry

end NUMINAMATH_CALUDE_specialNumberCount_is_70_l598_59862


namespace NUMINAMATH_CALUDE_caiden_roofing_problem_l598_59849

/-- Calculates the number of feet of free metal roofing given the total required roofing,
    cost per foot, and amount paid for the remaining roofing. -/
def free_roofing (total_required : ℕ) (cost_per_foot : ℕ) (amount_paid : ℕ) : ℕ :=
  total_required - (amount_paid / cost_per_foot)

/-- Theorem stating that given the specific conditions of Mr. Caiden's roofing problem,
    the amount of free roofing is 250 feet. -/
theorem caiden_roofing_problem :
  free_roofing 300 8 400 = 250 := by
  sorry

end NUMINAMATH_CALUDE_caiden_roofing_problem_l598_59849


namespace NUMINAMATH_CALUDE_triangle_perimeter_l598_59830

theorem triangle_perimeter (a b c : ℝ) (h_right_angle : a^2 + b^2 = c^2) 
  (h_hypotenuse : c = 15) (h_side_ratio : a = b / 2) : 
  a + b + c = 15 + 9 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l598_59830


namespace NUMINAMATH_CALUDE_sum_of_coefficients_excluding_constant_l598_59810

/-- The sum of the coefficients of the terms, excluding the constant term, 
    in the expansion of (x^2 - 2/x)^6 is -239 -/
theorem sum_of_coefficients_excluding_constant (x : ℝ) : 
  let f := (x^2 - 2/x)^6
  let all_coeff_sum := (1 - 2)^6
  let constant_term := 240
  all_coeff_sum - constant_term = -239 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_excluding_constant_l598_59810


namespace NUMINAMATH_CALUDE_cameron_questions_total_l598_59838

/-- Represents a tour group with a number of people and an optional inquisitive person. -/
structure TourGroup where
  people : Nat
  inquisitivePerson : Option Nat

/-- Calculates the number of questions answered for a given tour group. -/
def questionsAnswered (group : TourGroup) (questionsPerPerson : Nat) : Nat :=
  match group.inquisitivePerson with
  | none => group.people * questionsPerPerson
  | some n => (group.people - 1) * questionsPerPerson + n * questionsPerPerson

/-- Represents Cameron's tour day. -/
def cameronsTourDay : List TourGroup := [
  { people := 6, inquisitivePerson := none },
  { people := 11, inquisitivePerson := none },
  { people := 8, inquisitivePerson := some 3 },
  { people := 7, inquisitivePerson := none }
]

/-- The theorem stating the total number of questions Cameron answered. -/
theorem cameron_questions_total :
  (cameronsTourDay.map (questionsAnswered · 2)).sum = 68 := by
  sorry

end NUMINAMATH_CALUDE_cameron_questions_total_l598_59838


namespace NUMINAMATH_CALUDE_max_value_inequality_l598_59864

theorem max_value_inequality (x y : ℝ) (h : x * y > 0) :
  (x / (x + y)) + (2 * y / (x + 2 * y)) ≤ 4 - 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_inequality_l598_59864


namespace NUMINAMATH_CALUDE_positive_difference_of_solutions_l598_59803

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop :=
  x^2 - 5*x + 20 = x + 51

-- Define the solutions of the quadratic equation
def solutions : Set ℝ :=
  {x : ℝ | quadratic_equation x}

-- State the theorem
theorem positive_difference_of_solutions :
  ∃ (x y : ℝ), x ∈ solutions ∧ y ∈ solutions ∧ x ≠ y ∧ |x - y| = 4 * Real.sqrt 10 :=
sorry

end NUMINAMATH_CALUDE_positive_difference_of_solutions_l598_59803


namespace NUMINAMATH_CALUDE_circle_A_tangent_to_x_axis_l598_59882

def circle_A_center : ℝ × ℝ := (-4, -3)
def circle_A_radius : ℝ := 3

theorem circle_A_tangent_to_x_axis :
  let (x, y) := circle_A_center
  abs y = circle_A_radius := by sorry

end NUMINAMATH_CALUDE_circle_A_tangent_to_x_axis_l598_59882


namespace NUMINAMATH_CALUDE_fourth_root_of_207360000_l598_59801

theorem fourth_root_of_207360000 : Real.sqrt (Real.sqrt 207360000) = 120 := by sorry

end NUMINAMATH_CALUDE_fourth_root_of_207360000_l598_59801


namespace NUMINAMATH_CALUDE_family_savings_l598_59802

def income : ℕ := 509600
def expenses : ℕ := 276000
def initial_savings : ℕ := 1147240

theorem family_savings : initial_savings + income - expenses = 1340840 := by
  sorry

end NUMINAMATH_CALUDE_family_savings_l598_59802


namespace NUMINAMATH_CALUDE_largest_c_for_five_in_range_l598_59807

/-- The quadratic function f(x) = 2x^2 - 4x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := 2 * x^2 - 4 * x + c

/-- Theorem: The largest value of c such that 5 is in the range of f(x) = 2x^2 - 4x + c is 7 -/
theorem largest_c_for_five_in_range : 
  (∃ (x : ℝ), f 7 x = 5) ∧ 
  (∀ (c : ℝ), c > 7 → ¬∃ (x : ℝ), f c x = 5) := by
  sorry

end NUMINAMATH_CALUDE_largest_c_for_five_in_range_l598_59807


namespace NUMINAMATH_CALUDE_initial_speed_is_three_l598_59858

/-- Represents the scenario of two pedestrians walking towards each other --/
structure PedestrianScenario where
  totalDistance : ℝ
  delayDistance : ℝ
  delayTime : ℝ
  meetingDistanceAfterDelay : ℝ
  speedIncrease : ℝ

/-- Calculates the initial speed of the pedestrians --/
def initialSpeed (scenario : PedestrianScenario) : ℝ :=
  sorry

/-- Theorem stating that the initial speed is 3 km/h for the given scenario --/
theorem initial_speed_is_three 
  (scenario : PedestrianScenario) 
  (h1 : scenario.totalDistance = 28)
  (h2 : scenario.delayDistance = 9)
  (h3 : scenario.delayTime = 1)
  (h4 : scenario.meetingDistanceAfterDelay = 4)
  (h5 : scenario.speedIncrease = 1) :
  initialSpeed scenario = 3 := by
  sorry

end NUMINAMATH_CALUDE_initial_speed_is_three_l598_59858


namespace NUMINAMATH_CALUDE_square_triangulation_100_points_l598_59867

/-- Represents a triangulation of a square with additional points -/
structure SquareTriangulation where
  n : ℕ  -- number of additional points inside the square
  triangles : ℕ  -- number of triangles in the triangulation

/-- Theorem: A square triangulation with 100 additional points has 202 triangles -/
theorem square_triangulation_100_points :
  ∀ (st : SquareTriangulation), st.n = 100 → st.triangles = 202 := by
  sorry

end NUMINAMATH_CALUDE_square_triangulation_100_points_l598_59867


namespace NUMINAMATH_CALUDE_absolute_value_sum_zero_l598_59860

theorem absolute_value_sum_zero (a b : ℝ) :
  |a - 2| + |b + 3| = 0 → b^a = 9 := by sorry

end NUMINAMATH_CALUDE_absolute_value_sum_zero_l598_59860


namespace NUMINAMATH_CALUDE_fraction_simplification_l598_59886

theorem fraction_simplification (a b : ℝ) (h : a ≠ b) : (1/2 * a) / (1/2 * b) = a / b := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l598_59886


namespace NUMINAMATH_CALUDE_jim_skips_proof_l598_59842

/-- The number of times Bob can skip a rock. -/
def bob_skips : ℕ := 12

/-- The number of rocks Bob and Jim each skipped. -/
def rocks_skipped : ℕ := 10

/-- The total number of skips for both Bob and Jim. -/
def total_skips : ℕ := 270

/-- The number of times Jim can skip a rock. -/
def jim_skips : ℕ := 15

theorem jim_skips_proof : 
  bob_skips * rocks_skipped + jim_skips * rocks_skipped = total_skips :=
by sorry

end NUMINAMATH_CALUDE_jim_skips_proof_l598_59842


namespace NUMINAMATH_CALUDE_complex_addition_l598_59832

theorem complex_addition : ∃ z : ℂ, (5 - 3*I + z = -2 + 9*I) ∧ (z = -7 + 12*I) := by
  sorry

end NUMINAMATH_CALUDE_complex_addition_l598_59832


namespace NUMINAMATH_CALUDE_sampled_bag_number_61st_group_l598_59804

/-- Given a total number of bags, sample size, first sampled bag number, and group number,
    calculate the bag number for that group. -/
def sampledBagNumber (totalBags : ℕ) (sampleSize : ℕ) (firstSampledBag : ℕ) (groupNumber : ℕ) : ℕ :=
  firstSampledBag + (groupNumber - 1) * (totalBags / sampleSize)

/-- Theorem stating that for the given conditions, the 61st group's sampled bag number is 1211. -/
theorem sampled_bag_number_61st_group :
  sampledBagNumber 3000 150 11 61 = 1211 := by
  sorry


end NUMINAMATH_CALUDE_sampled_bag_number_61st_group_l598_59804


namespace NUMINAMATH_CALUDE_garrison_provisions_theorem_l598_59809

/-- Represents the number of days provisions last for a garrison -/
def provisionDays (initialMen : ℕ) (reinforcementMen : ℕ) (daysBeforeReinforcement : ℕ) (daysAfterReinforcement : ℕ) : ℕ :=
  let totalProvisions := initialMen * (daysBeforeReinforcement + daysAfterReinforcement)
  let remainingProvisions := totalProvisions - initialMen * daysBeforeReinforcement
  let totalMenAfterReinforcement := initialMen + reinforcementMen
  (totalProvisions / initialMen : ℕ)

theorem garrison_provisions_theorem (initialMen reinforcementMen daysBeforeReinforcement daysAfterReinforcement : ℕ) :
  initialMen = 2000 →
  reinforcementMen = 1300 →
  daysBeforeReinforcement = 21 →
  daysAfterReinforcement = 20 →
  provisionDays initialMen reinforcementMen daysBeforeReinforcement daysAfterReinforcement = 54 := by
  sorry

#eval provisionDays 2000 1300 21 20

end NUMINAMATH_CALUDE_garrison_provisions_theorem_l598_59809


namespace NUMINAMATH_CALUDE_max_sum_and_reciprocal_l598_59865

theorem max_sum_and_reciprocal (nums : Finset ℝ) (x : ℝ) :
  (Finset.card nums = 2023) →
  (∀ y ∈ nums, y > 0) →
  (x ∈ nums) →
  (Finset.sum nums id = 2024) →
  (Finset.sum nums (λ y => 1 / y) = 2024) →
  (x + 1 / x ≤ 4096094 / 2024) :=
by sorry

end NUMINAMATH_CALUDE_max_sum_and_reciprocal_l598_59865


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l598_59861

theorem quadratic_equations_solutions :
  (∃ x1 x2 : ℝ, (2 * (x1 - 1)^2 = 18 ∧ x1 = 4) ∧ (2 * (x2 - 1)^2 = 18 ∧ x2 = -2)) ∧
  (∃ y1 y2 : ℝ, (y1^2 - 4*y1 - 3 = 0 ∧ y1 = 2 + Real.sqrt 7) ∧ (y2^2 - 4*y2 - 3 = 0 ∧ y2 = 2 - Real.sqrt 7)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l598_59861


namespace NUMINAMATH_CALUDE_election_majority_l598_59896

theorem election_majority (total_votes : ℕ) (winning_percentage : ℚ) : 
  total_votes = 800 →
  winning_percentage = 70 / 100 →
  (winning_percentage * total_votes : ℚ) - ((1 - winning_percentage) * total_votes : ℚ) = 320 := by
  sorry

end NUMINAMATH_CALUDE_election_majority_l598_59896


namespace NUMINAMATH_CALUDE_honor_students_count_l598_59828

theorem honor_students_count 
  (total_students : ℕ) 
  (girls : ℕ) 
  (boys : ℕ) 
  (honor_girls : ℕ) 
  (honor_boys : ℕ) :
  total_students < 30 →
  total_students = girls + boys →
  (honor_girls : ℚ) / girls = 3 / 13 →
  (honor_boys : ℚ) / boys = 4 / 11 →
  honor_girls + honor_boys = 7 :=
by sorry

end NUMINAMATH_CALUDE_honor_students_count_l598_59828


namespace NUMINAMATH_CALUDE_total_spent_is_122_80_l598_59869

-- Define the cost per deck
def cost_per_deck : ℚ := 8

-- Define the number of decks bought by each person
def victor_decks : ℕ := 6
def friend_a_decks : ℕ := 4
def friend_b_decks : ℕ := 5
def friend_c_decks : ℕ := 3

-- Define the discount rates
def discount_rate (n : ℕ) : ℚ :=
  if n ≥ 6 then 0.20
  else if n = 5 then 0.15
  else if n ≥ 3 then 0.10
  else 0

-- Define the function to calculate the total cost for a person
def total_cost (decks : ℕ) : ℚ :=
  let base_cost := cost_per_deck * decks
  base_cost - (base_cost * discount_rate decks)

-- Theorem statement
theorem total_spent_is_122_80 :
  total_cost victor_decks +
  total_cost friend_a_decks +
  total_cost friend_b_decks +
  total_cost friend_c_decks = 122.80 :=
by sorry

end NUMINAMATH_CALUDE_total_spent_is_122_80_l598_59869


namespace NUMINAMATH_CALUDE_salary_solution_l598_59844

def salary_problem (salary : ℝ) : Prop :=
  let food_expense := (1 / 5 : ℝ) * salary
  let rent_expense := (1 / 10 : ℝ) * salary
  let clothes_expense := (3 / 5 : ℝ) * salary
  let remaining := salary - (food_expense + rent_expense + clothes_expense)
  remaining = 19000

theorem salary_solution :
  ∃ (salary : ℝ), salary_problem salary ∧ salary = 190000 := by
  sorry

end NUMINAMATH_CALUDE_salary_solution_l598_59844


namespace NUMINAMATH_CALUDE_problem1_problem2_l598_59846

-- Problem 1
def M : ℝ × ℝ := (3, 2)
def N : ℝ × ℝ := (4, -1)

def is_right_angle (A B C : ℝ × ℝ) : Prop :=
  let AB := (B.1 - A.1, B.2 - A.2)
  let AC := (C.1 - A.1, C.2 - A.2)
  AB.1 * AC.1 + AB.2 * AC.2 = 0

def on_x_axis (P : ℝ × ℝ) : Prop := P.2 = 0

theorem problem1 (P : ℝ × ℝ) :
  on_x_axis P ∧ is_right_angle M P N → P = (2, 0) ∨ P = (5, 0) :=
sorry

-- Problem 2
def A : ℝ × ℝ := (7, -4)
def B : ℝ × ℝ := (-5, 6)

def perpendicular_bisector (A B : ℝ × ℝ) (x y : ℝ) : Prop :=
  6 * x - 5 * y - 1 = 0

theorem problem2 :
  perpendicular_bisector A B = λ x y => 6 * x - 5 * y - 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_problem1_problem2_l598_59846


namespace NUMINAMATH_CALUDE_no_solutions_equation_l598_59895

theorem no_solutions_equation (x y : ℕ+) : x * (x + 1) ≠ 4 * y * (y + 1) := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_equation_l598_59895


namespace NUMINAMATH_CALUDE_yellow_flags_in_200_l598_59823

/-- Represents the number of flags in one complete pattern -/
def pattern_length : ℕ := 9

/-- Represents the number of yellow flags in one complete pattern -/
def yellow_per_pattern : ℕ := 3

/-- Represents the total number of flags we're considering -/
def total_flags : ℕ := 200

/-- Calculates the number of yellow flags in the given sequence -/
def yellow_flags (n : ℕ) : ℕ :=
  (n / pattern_length) * yellow_per_pattern + min yellow_per_pattern (n % pattern_length)

theorem yellow_flags_in_200 : yellow_flags total_flags = 67 := by
  sorry

end NUMINAMATH_CALUDE_yellow_flags_in_200_l598_59823


namespace NUMINAMATH_CALUDE_log_inequality_l598_59881

theorem log_inequality (a b : ℝ) (ha : a = Real.log 2 / Real.log 3) (hb : b = Real.log 3 / Real.log 2) :
  Real.log a < (1/2) ^ b := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l598_59881


namespace NUMINAMATH_CALUDE_work_rate_ab_together_days_for_ab_together_l598_59856

-- Define the work rates for workers a, b, and c
variable (A B C : ℝ)

-- Define the conditions
variable (h1 : A + B + C = 1 / 5)  -- a, b, and c together finish in 5 days
variable (h2 : C = 1 / 7.5)        -- c alone finishes in 7.5 days

-- Theorem to prove
theorem work_rate_ab_together : A + B = 1 / 15 := by
  sorry

-- Theorem to prove the final result
theorem days_for_ab_together : 1 / (A + B) = 15 := by
  sorry

end NUMINAMATH_CALUDE_work_rate_ab_together_days_for_ab_together_l598_59856


namespace NUMINAMATH_CALUDE_complex_power_eight_l598_59874

theorem complex_power_eight (a b : ℝ) (h : (a : ℂ) + Complex.I = 1 - b * Complex.I) : 
  (a + b * Complex.I : ℂ) ^ 8 = 16 := by sorry

end NUMINAMATH_CALUDE_complex_power_eight_l598_59874


namespace NUMINAMATH_CALUDE_difference_divisible_by_99_l598_59848

/-- Represents a three-digit number formed by digits a, b, and c -/
def three_digit_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

/-- The largest three-digit number formed by digits a, b, and c where a > b > c -/
def largest_number (a b c : ℕ) : ℕ := three_digit_number a b c

/-- The smallest three-digit number formed by digits a, b, and c where a > b > c -/
def smallest_number (a b c : ℕ) : ℕ := three_digit_number c b a

theorem difference_divisible_by_99 (a b c : ℕ) 
  (h1 : a > b) (h2 : b > c) (h3 : c > 0) (h4 : a < 10) (h5 : b < 10) (h6 : c < 10) :
  ∃ k : ℕ, largest_number a b c - smallest_number a b c = 99 * k :=
sorry

end NUMINAMATH_CALUDE_difference_divisible_by_99_l598_59848


namespace NUMINAMATH_CALUDE_problem_solving_probability_l598_59859

theorem problem_solving_probability : 
  let p_arthur : ℚ := 1/4
  let p_bella : ℚ := 3/10
  let p_xavier : ℚ := 1/6
  let p_yvonne : ℚ := 1/2
  let p_zelda : ℚ := 5/8
  let p_not_zelda : ℚ := 1 - p_zelda
  let p_four_solve : ℚ := p_arthur * p_yvonne * p_bella * p_xavier * p_not_zelda
  p_four_solve = 9/3840 := by sorry

end NUMINAMATH_CALUDE_problem_solving_probability_l598_59859


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l598_59854

theorem gcd_of_three_numbers : Nat.gcd 8650 (Nat.gcd 11570 28980) = 10 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l598_59854


namespace NUMINAMATH_CALUDE_negative_128_squared_div_64_l598_59888

theorem negative_128_squared_div_64 : ((-128)^2) / 64 = 256 := by sorry

end NUMINAMATH_CALUDE_negative_128_squared_div_64_l598_59888


namespace NUMINAMATH_CALUDE_initial_average_height_calculation_l598_59883

theorem initial_average_height_calculation (n : ℕ) (error : ℝ) (actual_avg : ℝ) :
  n = 35 ∧ error = 60 ∧ actual_avg = 178 →
  (n * actual_avg + error) / n = 179.71 := by
  sorry

end NUMINAMATH_CALUDE_initial_average_height_calculation_l598_59883


namespace NUMINAMATH_CALUDE_smith_family_laundry_l598_59894

/-- The number of bath towels that can fit in one load of laundry for the Smith family. -/
def towels_per_load (kylie_towels : ℕ) (daughters_towels : ℕ) (husband_towels : ℕ) (total_loads : ℕ) : ℕ :=
  (kylie_towels + daughters_towels + husband_towels) / total_loads

/-- Theorem stating that the washing machine can fit 4 bath towels in one load of laundry. -/
theorem smith_family_laundry :
  towels_per_load 3 6 3 3 = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_smith_family_laundry_l598_59894


namespace NUMINAMATH_CALUDE_binary_1101_to_base5_l598_59821

-- Define a function to convert binary to decimal
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

-- Define a function to convert decimal to base-5
def decimal_to_base5 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) :=
    if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
  aux n []

-- Theorem statement
theorem binary_1101_to_base5 :
  decimal_to_base5 (binary_to_decimal [true, false, true, true]) = [2, 3] := by
  sorry

end NUMINAMATH_CALUDE_binary_1101_to_base5_l598_59821


namespace NUMINAMATH_CALUDE_original_denominator_problem_l598_59870

theorem original_denominator_problem (d : ℚ) : 
  (3 : ℚ) / d ≠ 0 →
  (3 + 7 : ℚ) / (d + 7) = 1 / 3 →
  d = 23 := by
sorry

end NUMINAMATH_CALUDE_original_denominator_problem_l598_59870


namespace NUMINAMATH_CALUDE_solve_equation_l598_59837

theorem solve_equation (y : ℝ) (x : ℝ) (h1 : 9^y = x^12) (h2 : y = 6) : x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l598_59837


namespace NUMINAMATH_CALUDE_f_min_value_l598_59824

def f (x : ℝ) : ℝ := |2*x - 1| + |3*x - 2| + |4*x - 3| + |5*x - 4|

theorem f_min_value : ∀ x : ℝ, f x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_f_min_value_l598_59824


namespace NUMINAMATH_CALUDE_prime_odd_sum_l598_59873

theorem prime_odd_sum (a b : ℕ) : 
  Prime a → Odd b → a^2 + b = 2001 → a + b = 1999 := by sorry

end NUMINAMATH_CALUDE_prime_odd_sum_l598_59873


namespace NUMINAMATH_CALUDE_allocation_methods_3_6_3_l598_59892

/-- The number of ways to allocate doctors and nurses to schools -/
def allocation_methods (num_doctors num_nurses num_schools : ℕ) : ℕ :=
  (num_doctors.choose 1 * num_nurses.choose 2) *
  ((num_doctors - 1).choose 1 * (num_nurses - 2).choose 2) *
  ((num_doctors - 2).choose 1 * (num_nurses - 4).choose 2)

/-- Theorem stating that the number of allocation methods for 3 doctors and 6 nurses to 3 schools is 540 -/
theorem allocation_methods_3_6_3 :
  allocation_methods 3 6 3 = 540 := by
  sorry

end NUMINAMATH_CALUDE_allocation_methods_3_6_3_l598_59892


namespace NUMINAMATH_CALUDE_remainder_97_power_51_mod_100_l598_59897

theorem remainder_97_power_51_mod_100 : 97^51 % 100 = 39 := by
  sorry

end NUMINAMATH_CALUDE_remainder_97_power_51_mod_100_l598_59897


namespace NUMINAMATH_CALUDE_probability_one_defective_is_half_l598_59876

/-- Represents the total number of items -/
def total_items : Nat := 4

/-- Represents the number of genuine items -/
def genuine_items : Nat := 3

/-- Represents the number of defective items -/
def defective_items : Nat := 1

/-- Represents the number of items to be selected -/
def items_to_select : Nat := 2

/-- Calculates the number of ways to select k items from n items -/
def combinations (n k : Nat) : Nat := sorry

/-- Calculates the probability of selecting exactly one defective item -/
def probability_one_defective : Rat :=
  (combinations defective_items 1 * combinations genuine_items (items_to_select - 1)) /
  (combinations total_items items_to_select)

/-- Theorem stating that the probability of selecting exactly one defective item is 1/2 -/
theorem probability_one_defective_is_half :
  probability_one_defective = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_probability_one_defective_is_half_l598_59876


namespace NUMINAMATH_CALUDE_simplify_fraction_division_l598_59878

theorem simplify_fraction_division (x : ℝ) 
  (h1 : x ≠ 4) (h2 : x ≠ 2) (h3 : x ≠ 5) (h4 : x ≠ 3) (h5 : x ≠ 1) : 
  (x^2 - 4*x + 3) / (x^2 - 6*x + 8) / ((x^2 - 6*x + 5) / (x^2 - 8*x + 15)) = 1 / ((x - 4) * (x - 2)) :=
by
  sorry

#check simplify_fraction_division

end NUMINAMATH_CALUDE_simplify_fraction_division_l598_59878


namespace NUMINAMATH_CALUDE_no_valid_numbers_l598_59879

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  first : Nat
  middle : Nat
  last : Nat
  first_digit : first < 10
  middle_digit : middle < 10
  last_digit : last < 10
  three_digits : first ≠ 0

/-- Checks if a number is not divisible by 3 -/
def notDivisibleByThree (n : ThreeDigitNumber) : Prop :=
  (100 * n.first + 10 * n.middle + n.last) % 3 ≠ 0

/-- Checks if the sum of digits is less than 22 -/
def sumLessThan22 (n : ThreeDigitNumber) : Prop :=
  n.first + n.middle + n.last < 22

/-- Checks if the middle digit is twice the first digit -/
def middleTwiceFirst (n : ThreeDigitNumber) : Prop :=
  n.middle = 2 * n.first

theorem no_valid_numbers :
  ¬ ∃ (n : ThreeDigitNumber),
    notDivisibleByThree n ∧
    sumLessThan22 n ∧
    middleTwiceFirst n :=
sorry

end NUMINAMATH_CALUDE_no_valid_numbers_l598_59879


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l598_59890

theorem regular_polygon_sides (n : ℕ) (h : n > 2) :
  (180 * (n - 2) : ℝ) / n = 156 → n = 15 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l598_59890


namespace NUMINAMATH_CALUDE_log_relationship_depends_on_base_l598_59805

noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem log_relationship_depends_on_base (a : ℝ) 
  (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ (a1 a2 : ℝ), 
    (a1 > 0 ∧ a1 ≠ 1 ∧ log a1 2 + log a1 10 > 2 * log a1 6) ∧
    (a2 > 0 ∧ a2 ≠ 1 ∧ log a2 2 + log a2 10 < 2 * log a2 6) :=
by sorry

end NUMINAMATH_CALUDE_log_relationship_depends_on_base_l598_59805


namespace NUMINAMATH_CALUDE_fraction_simplification_l598_59891

theorem fraction_simplification :
  (1 / 2 + 1 / 5) / (3 / 7 - 1 / 14) = 49 / 25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l598_59891


namespace NUMINAMATH_CALUDE_tan_eq_two_solution_set_l598_59815

theorem tan_eq_two_solution_set :
  {x : ℝ | ∃ k : ℤ, x = k * Real.pi + Real.arctan 2} = {x : ℝ | Real.tan x = 2} := by
sorry

end NUMINAMATH_CALUDE_tan_eq_two_solution_set_l598_59815


namespace NUMINAMATH_CALUDE_roots_and_coefficients_l598_59820

theorem roots_and_coefficients (θ : Real) (m : Real) :
  0 < θ ∧ θ < 2 * Real.pi →
  (2 * Real.sin θ ^ 2 - (Real.sqrt 3 + 1) * Real.sin θ + m = 0) ∧
  (2 * Real.cos θ ^ 2 - (Real.sqrt 3 + 1) * Real.cos θ + m = 0) →
  (Real.sin θ ^ 2 / (Real.sin θ - Real.cos θ) + Real.cos θ ^ 2 / (Real.cos θ - Real.sin θ) = (Real.sqrt 3 + 1) / 2) ∧
  (m = Real.sqrt 3 / 2) := by
sorry

end NUMINAMATH_CALUDE_roots_and_coefficients_l598_59820


namespace NUMINAMATH_CALUDE_opposite_numbers_l598_59851

theorem opposite_numbers : ∀ x : ℚ, |x| = -x → x = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_l598_59851


namespace NUMINAMATH_CALUDE_tetrahedron_side_sum_squares_l598_59889

/-- A tetrahedron with side lengths a, b, c and circumradius 1 -/
structure Tetrahedron where
  a : ℝ
  b : ℝ
  c : ℝ
  circumradius : ℝ
  circumradius_eq_one : circumradius = 1

/-- The sum of squares of the side lengths of the tetrahedron is 8 -/
theorem tetrahedron_side_sum_squares (t : Tetrahedron) : t.a^2 + t.b^2 + t.c^2 = 8 := by
  sorry


end NUMINAMATH_CALUDE_tetrahedron_side_sum_squares_l598_59889


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l598_59847

theorem other_root_of_quadratic (m : ℝ) : 
  (1 : ℝ) ^ 2 - 3 * (1 : ℝ) + m = 0 → 
  ∃ (x : ℝ), x ≠ 1 ∧ x ^ 2 - 3 * x + m = 0 ∧ x = 2 := by
sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l598_59847


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l598_59871

/-- Given a geometric sequence {a_n} with a_1 = 3 and a_1 + a_3 + a_5 = 21,
    prove that a_3 + a_5 + a_7 = 42 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) : 
  (∀ n, a (n + 1) = a n * q) →  -- Geometric sequence definition
  a 1 = 3 →                    -- First condition
  a 1 + a 3 + a 5 = 21 →       -- Second condition
  a 3 + a 5 + a 7 = 42 :=      -- Conclusion to prove
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l598_59871


namespace NUMINAMATH_CALUDE_functional_equation_properties_l598_59827

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * f y

theorem functional_equation_properties (f : ℝ → ℝ) 
  (h_eq : FunctionalEquation f) (h_nonzero : f 0 ≠ 0) : 
  (f 0 = 1) ∧ (∀ x : ℝ, f (-x) = f x) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_properties_l598_59827


namespace NUMINAMATH_CALUDE_line_not_in_third_quadrant_line_through_two_points_l598_59829

-- Define a line with coefficients A, B, and C
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

-- Define a point in 2D plane
structure Point where
  x : ℝ
  y : ℝ

-- Theorem 1: Line does not pass through third quadrant
theorem line_not_in_third_quadrant (l : Line) 
  (h1 : l.A * l.B < 0) (h2 : l.B * l.C < 0) : 
  ∀ (p : Point), p.x < 0 ∧ p.y < 0 → l.A * p.x - l.B * p.y - l.C ≠ 0 :=
sorry

-- Theorem 2: Line equation through two distinct points
theorem line_through_two_points (p1 p2 : Point) (h : p1 ≠ p2) :
  ∀ (p : Point), (p2.x - p1.x) * (p.y - p1.y) = (p2.y - p1.y) * (p.x - p1.x) ↔
  ∃ (t : ℝ), p.x = p1.x + t * (p2.x - p1.x) ∧ p.y = p1.y + t * (p2.y - p1.y) :=
sorry

end NUMINAMATH_CALUDE_line_not_in_third_quadrant_line_through_two_points_l598_59829


namespace NUMINAMATH_CALUDE_models_after_price_increase_l598_59806

-- Define the original price, price increase percentage, and initial number of models
def original_price : ℚ := 45/100
def price_increase_percent : ℚ := 15/100
def initial_models : ℕ := 30

-- Calculate the new price after the increase
def new_price : ℚ := original_price * (1 + price_increase_percent)

-- Calculate the total savings
def total_savings : ℚ := original_price * initial_models

-- Define the theorem
theorem models_after_price_increase :
  ⌊total_savings / new_price⌋ = 26 := by
  sorry

#eval ⌊total_savings / new_price⌋

end NUMINAMATH_CALUDE_models_after_price_increase_l598_59806


namespace NUMINAMATH_CALUDE_profit_margin_in_terms_of_selling_price_l598_59826

/-- Given a selling price S, cost C, and profit margin M, prove that
    if S = 3C and M = (1/2n)C + (1/3n)S, then M = S/(2n) -/
theorem profit_margin_in_terms_of_selling_price
  (S C : ℝ) (n : ℝ) (hn : n ≠ 0) (M : ℝ) 
  (h_selling_price : S = 3 * C)
  (h_profit_margin : M = (1 / (2 * n)) * C + (1 / (3 * n)) * S) :
  M = S / (2 * n) := by
sorry

end NUMINAMATH_CALUDE_profit_margin_in_terms_of_selling_price_l598_59826


namespace NUMINAMATH_CALUDE_high_school_enrollment_l598_59811

/-- The number of students in a high school with given enrollment in music and art classes -/
def total_students (music : ℕ) (art : ℕ) (both : ℕ) (neither : ℕ) : ℕ :=
  (music - both) + (art - both) + both + neither

/-- Theorem stating that the total number of students is 500 given the specific enrollment numbers -/
theorem high_school_enrollment : total_students 30 20 10 460 = 500 := by
  sorry

end NUMINAMATH_CALUDE_high_school_enrollment_l598_59811


namespace NUMINAMATH_CALUDE_village_population_l598_59843

theorem village_population (initial_population : ℕ) 
  (death_rate : ℚ) (leaving_rate : ℚ) (final_population : ℕ) : 
  initial_population = 4400 →
  death_rate = 5 / 100 →
  leaving_rate = 15 / 100 →
  final_population = 
    (initial_population - 
      (initial_population * death_rate).floor - 
      ((initial_population - (initial_population * death_rate).floor) * leaving_rate).floor) →
  final_population = 3553 := by
sorry

end NUMINAMATH_CALUDE_village_population_l598_59843


namespace NUMINAMATH_CALUDE_ellipse_min_sum_l598_59898

theorem ellipse_min_sum (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 1/m + 4/n = 1) :
  m + n ≥ 9 ∧ ∃ m₀ n₀ : ℝ, m₀ > 0 ∧ n₀ > 0 ∧ 1/m₀ + 4/n₀ = 1 ∧ m₀ + n₀ = 9 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_min_sum_l598_59898


namespace NUMINAMATH_CALUDE_distribute_books_count_l598_59885

/-- The number of ways to distribute books among students -/
def distribute_books : ℕ :=
  let num_students : ℕ := 4
  let num_novels : ℕ := 4
  let num_anthologies : ℕ := 1
  -- Category 1: Each student gets 1 novel, anthology to any student
  let category1 : ℕ := num_students
  -- Category 2: Anthology to one student, novels distributed to others
  let category2 : ℕ := num_students * (num_students - 1)
  category1 + category2

/-- Theorem stating that the number of distribution methods is 16 -/
theorem distribute_books_count : distribute_books = 16 := by
  sorry

end NUMINAMATH_CALUDE_distribute_books_count_l598_59885


namespace NUMINAMATH_CALUDE_power_of_two_triples_l598_59887

def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

def valid_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  is_power_of_two (a * b - c) ∧
  is_power_of_two (b * c - a) ∧
  is_power_of_two (c * a - b)

theorem power_of_two_triples :
  ∀ a b c : ℕ, valid_triple a b c ↔
    (a = 2 ∧ b = 2 ∧ c = 2) ∨
    (a = 2 ∧ b = 2 ∧ c = 3) ∨
    (a = 3 ∧ b = 5 ∧ c = 7) ∨
    (a = 2 ∧ b = 6 ∧ c = 11) ∨
    (a = 2 ∧ b = 3 ∧ c = 2) ∨
    (a = 2 ∧ b = 11 ∧ c = 6) ∨
    (a = 3 ∧ b = 7 ∧ c = 5) ∨
    (a = 5 ∧ b = 7 ∧ c = 3) ∨
    (a = 5 ∧ b = 3 ∧ c = 7) ∨
    (a = 6 ∧ b = 11 ∧ c = 2) ∨
    (a = 7 ∧ b = 3 ∧ c = 5) ∨
    (a = 7 ∧ b = 5 ∧ c = 3) ∨
    (a = 11 ∧ b = 2 ∧ c = 6) :=
by sorry

end NUMINAMATH_CALUDE_power_of_two_triples_l598_59887


namespace NUMINAMATH_CALUDE_problem_statement_l598_59899

noncomputable def f (x : ℝ) : ℝ := (1 - 2^x) / (1 + 2^x)

noncomputable def g (x : ℝ) : ℝ := Real.log (Real.sqrt (x^2 + 1) - x)

noncomputable def F (x : ℝ) : ℝ := f x + g x

theorem problem_statement :
  (∀ x, f (-x) = -f x) ∧
  (∀ x, g (-x) = -g x) ∧
  (∃ M m, (∀ x ∈ Set.Icc (-1) 1, F x ≤ M ∧ m ≤ F x) ∧ M + m = 0) ∧
  (Set.Ioi 1 = {a | F (2*a) + F (-1-a) < 0}) :=
sorry

end NUMINAMATH_CALUDE_problem_statement_l598_59899


namespace NUMINAMATH_CALUDE_smallest_number_l598_59893

theorem smallest_number (a b c d : ℝ) (ha : a = -2) (hb : b = 0) (hc : c = 1/2) (hd : d = 2) :
  a ≤ b ∧ a ≤ c ∧ a ≤ d :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l598_59893


namespace NUMINAMATH_CALUDE_f_properties_l598_59868

noncomputable def f (x : ℝ) : ℝ := 1/2 - 1/(2^x + 1)

theorem f_properties :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂) ∧
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → f x ∈ Set.Icc (1/6) (3/10)) ∧
  f 1 = 1/6 ∧ f 2 = 3/10 :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l598_59868


namespace NUMINAMATH_CALUDE_cylinder_height_l598_59835

theorem cylinder_height (r h : ℝ) : 
  r = 4 →
  2 * Real.pi * r^2 + 2 * Real.pi * r * h = 40 * Real.pi →
  h = 1 := by
sorry

end NUMINAMATH_CALUDE_cylinder_height_l598_59835


namespace NUMINAMATH_CALUDE_article_cost_l598_59884

/-- The cost of an article, given selling conditions -/
theorem article_cost : ∃ (cost : ℝ), 
  (580 - cost) = 1.08 * (520 - cost) ∧ 
  cost = 230 := by
  sorry

end NUMINAMATH_CALUDE_article_cost_l598_59884


namespace NUMINAMATH_CALUDE_contrapositive_sum_irrational_l598_59845

/-- The contrapositive of "If a + b is irrational, then at least one of a or b is irrational" -/
theorem contrapositive_sum_irrational (a b : ℝ) :
  (¬(∃ q : ℚ, (a : ℝ) = q) ∨ ¬(∃ q : ℚ, (b : ℝ) = q) → ¬(∃ q : ℚ, (a + b : ℝ) = q)) ↔
  ((∃ q : ℚ, (a : ℝ) = q) ∧ (∃ q : ℚ, (b : ℝ) = q) → (∃ q : ℚ, (a + b : ℝ) = q)) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_sum_irrational_l598_59845


namespace NUMINAMATH_CALUDE_lcm_factor_problem_l598_59800

theorem lcm_factor_problem (A B : ℕ+) : 
  Nat.gcd A B = 10 →
  A = 150 →
  11 ∣ Nat.lcm A B →
  Nat.lcm A B = 10 * 11 * 15 := by
sorry

end NUMINAMATH_CALUDE_lcm_factor_problem_l598_59800


namespace NUMINAMATH_CALUDE_race_pace_cristina_pace_l598_59855

/-- The race between Nicky and Cristina -/
theorem race_pace (nicky_pace : ℝ) (race_time : ℝ) (head_start : ℝ) : ℝ :=
  let nicky_distance := nicky_pace * race_time
  let cristina_distance := nicky_distance + head_start
  cristina_distance / race_time

/-- Cristina's pace in the race -/
theorem cristina_pace : race_pace 3 36 36 = 4 := by
  sorry

end NUMINAMATH_CALUDE_race_pace_cristina_pace_l598_59855


namespace NUMINAMATH_CALUDE_inequality_proof_l598_59808

theorem inequality_proof (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : 0 > c) (h4 : c > d) :
  (a + c > b + d) ∧ (a * d^2 > b * c^2) ∧ (1 / (b * c) < 1 / (a * d)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l598_59808
