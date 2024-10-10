import Mathlib

namespace f_at_2_l1219_121983

def f (x : ℝ) : ℝ := 2 * x^5 + 3 * x^4 + 2 * x^3 - 4 * x + 5

theorem f_at_2 : f 2 = 125 := by
  sorry

end f_at_2_l1219_121983


namespace unique_n_satisfying_conditions_l1219_121949

/-- Greatest prime factor of a positive integer -/
def greatest_prime_factor (n : ℕ) : ℕ := sorry

/-- The theorem states that there exists exactly one positive integer n > 1
    satisfying both conditions simultaneously -/
theorem unique_n_satisfying_conditions : ∃! n : ℕ, n > 1 ∧ 
  (greatest_prime_factor n = n.sqrt) ∧ 
  (greatest_prime_factor (n + 72) = (n + 72).sqrt) := by sorry

end unique_n_satisfying_conditions_l1219_121949


namespace min_abs_sum_l1219_121930

theorem min_abs_sum (a b c : ℝ) (h1 : a + b + c = -2) (h2 : a * b * c = -4) :
  ∀ x y z : ℝ, x + y + z = -2 → x * y * z = -4 → |a| + |b| + |c| ≤ |x| + |y| + |z| ∧ 
  ∃ a₀ b₀ c₀ : ℝ, a₀ + b₀ + c₀ = -2 ∧ a₀ * b₀ * c₀ = -4 ∧ |a₀| + |b₀| + |c₀| = 6 := by
sorry

end min_abs_sum_l1219_121930


namespace net_gain_calculation_l1219_121931

def initial_value : ℝ := 15000
def profit_percentage : ℝ := 0.20
def loss_percentage : ℝ := 0.15
def transaction_fee : ℝ := 300

def first_sale_price : ℝ := initial_value * (1 + profit_percentage)
def second_sale_price : ℝ := first_sale_price * (1 - loss_percentage)
def total_cost : ℝ := second_sale_price + transaction_fee

theorem net_gain_calculation :
  first_sale_price - total_cost = 2400 := by sorry

end net_gain_calculation_l1219_121931


namespace least_subtraction_for_divisibility_problem_solution_l1219_121921

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (k : ℕ), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : ℕ), m < k → (n - m) % d ≠ 0 :=
by
  sorry

theorem problem_solution :
  ∃ (k : ℕ), k = 3 ∧ (5474827 - k) % 12 = 0 ∧ ∀ (m : ℕ), m < k → (5474827 - m) % 12 ≠ 0 :=
by
  sorry

end least_subtraction_for_divisibility_problem_solution_l1219_121921


namespace extreme_points_cubic_l1219_121993

/-- A function f(x) = x^3 + ax has exactly two extreme points on R if and only if a < 0 -/
theorem extreme_points_cubic (a : ℝ) :
  (∃! (p q : ℝ), p ≠ q ∧ 
    (∀ x : ℝ, (3 * x^2 + a = 0) ↔ (x = p ∨ x = q))) ↔ 
  a < 0 := by
  sorry

end extreme_points_cubic_l1219_121993


namespace carey_gumballs_difference_l1219_121938

/-- The number of gumballs Carolyn bought -/
def carolyn_gumballs : ℕ := 17

/-- The number of gumballs Lew bought -/
def lew_gumballs : ℕ := 12

/-- The average number of gumballs bought by the three people -/
def average_gumballs : ℚ → ℚ := λ c => (carolyn_gumballs + lew_gumballs + c) / 3

/-- The theorem stating the difference between max and min gumballs Carey could have bought -/
theorem carey_gumballs_difference :
  ∃ (min_c max_c : ℕ),
    (∀ c : ℚ, 19 ≤ average_gumballs c → average_gumballs c ≤ 25 → ↑min_c ≤ c ∧ c ≤ ↑max_c) ∧
    max_c - min_c = 18 := by
  sorry

end carey_gumballs_difference_l1219_121938


namespace lagrange_interpolation_polynomial_l1219_121953

/-- Lagrange interpolation polynomial for the given points -/
def P₂ (x : ℝ) : ℝ := 2 * x^2 + 5 * x - 8

/-- The x-coordinates of the interpolation points -/
def x₀ : ℝ := -3
def x₁ : ℝ := -1
def x₂ : ℝ := 2

/-- The y-coordinates of the interpolation points -/
def y₀ : ℝ := -5
def y₁ : ℝ := -11
def y₂ : ℝ := 10

/-- Theorem stating that P₂ is the Lagrange interpolation polynomial for the given points -/
theorem lagrange_interpolation_polynomial :
  P₂ x₀ = y₀ ∧ P₂ x₁ = y₁ ∧ P₂ x₂ = y₂ := by
  sorry

end lagrange_interpolation_polynomial_l1219_121953


namespace largest_root_of_g_l1219_121907

def g (x : ℝ) : ℝ := 24 * x^4 - 34 * x^2 + 6

theorem largest_root_of_g :
  ∃ (r : ℝ), r = 1/2 ∧ g r = 0 ∧ ∀ (x : ℝ), g x = 0 → x ≤ r :=
sorry

end largest_root_of_g_l1219_121907


namespace linear_regression_change_specific_regression_change_l1219_121916

/-- Given a linear regression equation y = a + bx, this theorem proves
    that when x increases by 1 unit, y changes by b units. -/
theorem linear_regression_change (a b : ℝ) :
  let y : ℝ → ℝ := λ x ↦ a + b * x
  ∀ x : ℝ, y (x + 1) - y x = b := by
  sorry

/-- For the specific linear regression equation y = 2 - 3.5x,
    this theorem proves that when x increases by 1 unit, y decreases by 3.5 units. -/
theorem specific_regression_change :
  let y : ℝ → ℝ := λ x ↦ 2 - 3.5 * x
  ∀ x : ℝ, y (x + 1) - y x = -3.5 := by
  sorry

end linear_regression_change_specific_regression_change_l1219_121916


namespace last_digit_of_one_over_three_to_fifteen_l1219_121959

theorem last_digit_of_one_over_three_to_fifteen (n : ℕ) :
  n = 15 →
  ∃ k : ℕ, (1 : ℚ) / 3^n = k / 10 + (0 : ℚ) / 10 := by sorry

end last_digit_of_one_over_three_to_fifteen_l1219_121959


namespace inverse_sum_modulo_eleven_l1219_121955

theorem inverse_sum_modulo_eleven :
  (((3⁻¹ : ZMod 11) + (5⁻¹ : ZMod 11) + (7⁻¹ : ZMod 11))⁻¹ : ZMod 11) = 10 := by
  sorry

end inverse_sum_modulo_eleven_l1219_121955


namespace polynomial_root_relation_l1219_121974

theorem polynomial_root_relation (p q r s : ℝ) (h_p : p ≠ 0) :
  (p * (4 : ℝ)^3 + q * (4 : ℝ)^2 + r * (4 : ℝ) + s = 0) →
  (p * (-3 : ℝ)^3 + q * (-3 : ℝ)^2 + r * (-3 : ℝ) + s = 0) →
  (q + r) / p = -13 := by
sorry

end polynomial_root_relation_l1219_121974


namespace gcd_12345_6789_l1219_121989

theorem gcd_12345_6789 : Nat.gcd 12345 6789 = 3 := by
  sorry

end gcd_12345_6789_l1219_121989


namespace town_growth_is_21_percent_l1219_121922

/-- Represents the population of a town over a 20-year period -/
structure TownPopulation where
  pop1991 : Nat
  pop2001 : Nat
  pop2011 : Nat

/-- Conditions for the town population -/
def ValidPopulation (t : TownPopulation) : Prop :=
  ∃ p q : Nat,
    t.pop1991 = p^2 ∧
    t.pop2001 = t.pop1991 + 180 ∧
    t.pop2001 = q^2 + 16 ∧
    t.pop2011 = t.pop2001 + 180

/-- The percent growth of the population over 20 years -/
def PercentGrowth (t : TownPopulation) : ℚ :=
  (t.pop2011 - t.pop1991 : ℚ) / t.pop1991 * 100

/-- Theorem stating that the percent growth is 21% -/
theorem town_growth_is_21_percent (t : TownPopulation) 
  (h : ValidPopulation t) : PercentGrowth t = 21 := by
  sorry

#check town_growth_is_21_percent

end town_growth_is_21_percent_l1219_121922


namespace ratio_w_to_y_l1219_121991

theorem ratio_w_to_y (w x y z : ℚ) 
  (hw : w / x = 3 / 2)
  (hy : y / z = 4 / 3)
  (hz : z / x = 1 / 5) :
  w / y = 45 / 8 := by
sorry

end ratio_w_to_y_l1219_121991


namespace triangle_side_length_l1219_121941

theorem triangle_side_length (a b c : ℝ) (C : ℝ) :
  a = 2 → b = 1 → C = Real.pi / 3 →
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) →
  c = Real.sqrt 3 := by
sorry

end triangle_side_length_l1219_121941


namespace line_translation_l1219_121936

/-- Given a line with equation y = -2x, prove that translating it upward by 1 unit results in the equation y = -2x + 1 -/
theorem line_translation (x y : ℝ) :
  (y = -2 * x) →  -- Original line equation
  (∃ (y' : ℝ), y' = y + 1 ∧ y' = -2 * x + 1) -- Translated line equation
  := by sorry

end line_translation_l1219_121936


namespace triangle_inradius_l1219_121908

/-- Given a triangle with perimeter 24 cm and area 30 cm², prove that its inradius is 2.5 cm. -/
theorem triangle_inradius (perimeter : ℝ) (area : ℝ) (inradius : ℝ) : 
  perimeter = 24 → area = 30 → inradius * (perimeter / 2) = area → inradius = 2.5 := by
  sorry

end triangle_inradius_l1219_121908


namespace example_implicit_function_l1219_121997

-- Define the concept of an implicit function
def is_implicit_function (F : ℝ → ℝ → ℝ) : Prop :=
  ∃ (x y : ℝ), F x y = 0

-- Define our specific function
def F (x y : ℝ) : ℝ := 2 * x - 3 * y - 1

-- Theorem statement
theorem example_implicit_function :
  is_implicit_function F :=
sorry

end example_implicit_function_l1219_121997


namespace hyperbola_asymptotes_l1219_121948

/-- Given a hyperbola with equation 9x^2 - 4y^2 = -36, 
    its asymptotes are y = ±(3/2)(-ix) -/
theorem hyperbola_asymptotes :
  ∀ (x y : ℂ), 9 * x^2 - 4 * y^2 = -36 →
  ∃ (k : ℂ), k = (3 / 2) * Complex.I ∧
  (y = k * x ∨ y = -k * x) :=
by sorry

end hyperbola_asymptotes_l1219_121948


namespace scaled_box_capacity_l1219_121939

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  height : ℝ
  width : ℝ
  length : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ :=
  d.height * d.width * d.length

/-- Theorem: A box with 3 times the height, 2 times the width, and 1/2 times the length of a box
    that can hold 60 grams of clay can hold 180 grams of clay -/
theorem scaled_box_capacity
  (first_box : BoxDimensions)
  (first_box_capacity : ℝ)
  (h_first_box_capacity : first_box_capacity = 60)
  (second_box : BoxDimensions)
  (h_second_box_height : second_box.height = 3 * first_box.height)
  (h_second_box_width : second_box.width = 2 * first_box.width)
  (h_second_box_length : second_box.length = 1/2 * first_box.length) :
  (boxVolume second_box / boxVolume first_box) * first_box_capacity = 180 := by
  sorry

end scaled_box_capacity_l1219_121939


namespace set_difference_M_N_l1219_121903

def M : Set ℕ := {1, 3, 5, 7, 9}
def N : Set ℕ := {2, 3, 5}

def setDifference (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

theorem set_difference_M_N : setDifference M N = {1, 7, 9} := by
  sorry

end set_difference_M_N_l1219_121903


namespace student_selection_and_advancement_probability_l1219_121947

-- Define the scores for students A and B
def scores_A : List ℕ := [100, 90, 120, 130, 105, 115]
def scores_B : List ℕ := [95, 125, 110, 95, 100, 135]

-- Define a function to calculate the average score
def average (scores : List ℕ) : ℚ :=
  (scores.sum : ℚ) / scores.length

-- Define a function to calculate the variance
def variance (scores : List ℕ) : ℚ :=
  let avg := average scores
  (scores.map (λ x => ((x : ℚ) - avg) ^ 2)).sum / scores.length

-- Define a function to select the student with lower variance
def select_student (scores_A scores_B : List ℕ) : Bool :=
  variance scores_A < variance scores_B

-- Define the probability of advancing to the final round
def probability_advance : ℚ := 7 / 10

-- Theorem statement
theorem student_selection_and_advancement_probability 
  (scores_A scores_B : List ℕ) 
  (h_scores_A : scores_A = [100, 90, 120, 130, 105, 115])
  (h_scores_B : scores_B = [95, 125, 110, 95, 100, 135]) :
  select_student scores_A scores_B = true ∧ 
  probability_advance = 7 / 10 := by
  sorry


end student_selection_and_advancement_probability_l1219_121947


namespace cubic_coefficient_in_product_l1219_121943

/-- The coefficient of x^3 in the expansion of (3x^3 + 2x^2 + 4x + 5)(4x^3 + 3x^2 + 5x + 6) -/
def cubic_coefficient : ℤ := 40

/-- The first polynomial in the product -/
def polynomial1 (x : ℚ) : ℚ := 3 * x^3 + 2 * x^2 + 4 * x + 5

/-- The second polynomial in the product -/
def polynomial2 (x : ℚ) : ℚ := 4 * x^3 + 3 * x^2 + 5 * x + 6

/-- The theorem stating that the coefficient of x^3 in the expansion of the product of polynomial1 and polynomial2 is equal to cubic_coefficient -/
theorem cubic_coefficient_in_product : 
  ∃ (a b c d e f g : ℚ), 
    polynomial1 x * polynomial2 x = a * x^6 + b * x^5 + c * x^4 + cubic_coefficient * x^3 + d * x^2 + e * x + f :=
by sorry

end cubic_coefficient_in_product_l1219_121943


namespace common_difference_is_one_l1219_121906

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (a b c : ℝ) : Prop :=
  b^2 = a * c

theorem common_difference_is_one
  (a : ℕ → ℝ)
  (d : ℝ)
  (h1 : d ≠ 0)
  (h2 : a 1 = 1)
  (h3 : arithmetic_sequence a d)
  (h4 : geometric_sequence (a 1) (a 3) (a 9)) :
  d = 1 :=
sorry

end common_difference_is_one_l1219_121906


namespace number_above_210_l1219_121998

/-- The number of elements in the k-th row of the triangular array -/
def row_size (k : ℕ) : ℕ := k * (k + 1) / 2

/-- The sum of elements up to and including the k-th row -/
def sum_up_to_row (k : ℕ) : ℕ := k * (k + 1) * (k + 2) / 6

/-- The first element in the k-th row -/
def first_in_row (k : ℕ) : ℕ := sum_up_to_row (k - 1) + 1

/-- The last element in the k-th row -/
def last_in_row (k : ℕ) : ℕ := sum_up_to_row k

theorem number_above_210 :
  ∃ (k : ℕ), 
    (first_in_row k ≤ 210) ∧ 
    (210 ≤ last_in_row k) ∧ 
    (210 - first_in_row k + 1 = row_size k) ∧
    (last_in_row (k - 1) = 165) := by
  sorry

#eval sum_up_to_row 9  -- Expected: 165
#eval sum_up_to_row 10 -- Expected: 220
#eval first_in_row 10  -- Expected: 166
#eval last_in_row 9    -- Expected: 165

end number_above_210_l1219_121998


namespace sum_of_base3_digits_333_l1219_121990

/-- Converts a natural number to its base-3 representation -/
def toBase3 (n : ℕ) : List ℕ :=
  sorry

/-- Sums the digits in a list -/
def sumDigits (digits : List ℕ) : ℕ :=
  sorry

/-- The sum of the digits in the base-3 representation of 333 is 3 -/
theorem sum_of_base3_digits_333 : sumDigits (toBase3 333) = 3 := by
  sorry

end sum_of_base3_digits_333_l1219_121990


namespace solutions_for_20_l1219_121988

/-- The number of integer solutions for |x| + |y| = n -/
def num_solutions (n : ℕ) : ℕ :=
  4 * n

/-- The property that |x| + |y| = 1 has 4 solutions -/
axiom base_case : num_solutions 1 = 4

/-- The property that the number of solutions increases by 4 for each unit increase -/
axiom induction_step : ∀ n : ℕ, num_solutions (n + 1) = num_solutions n + 4

/-- The theorem to be proved -/
theorem solutions_for_20 : num_solutions 20 = 80 := by
  sorry

end solutions_for_20_l1219_121988


namespace jerome_contacts_l1219_121944

/-- Calculates the total number of contacts on Jerome's list --/
def total_contacts (classmates : ℕ) (family_members : ℕ) : ℕ :=
  classmates + (classmates / 2) + family_members

/-- Theorem stating that Jerome's contact list has 33 people --/
theorem jerome_contacts : total_contacts 20 3 = 33 := by
  sorry

end jerome_contacts_l1219_121944


namespace unique_rational_pair_l1219_121982

theorem unique_rational_pair : 
  ∀ (a b r s : ℚ), 
    a ≠ b → 
    r ≠ s → 
    (∀ (z : ℚ), (z - r) * (z - s) = (z - a*r) * (z - b*s)) → 
    ∃! (p : ℚ × ℚ), p.1 ≠ p.2 ∧ 
      ∀ (z : ℚ), (z - r) * (z - s) = (z - p.1*r) * (z - p.2*s) :=
by sorry

end unique_rational_pair_l1219_121982


namespace at_op_difference_l1219_121987

-- Define the @ operation
def at_op (x y : ℤ) : ℤ := x * y - 3 * x

-- Theorem statement
theorem at_op_difference : at_op 9 6 - at_op 6 9 = -9 := by
  sorry

end at_op_difference_l1219_121987


namespace unique_perfect_square_polynomial_l1219_121933

theorem unique_perfect_square_polynomial : 
  ∃! y : ℤ, ∃ n : ℤ, y^4 + 4*y^3 + 9*y^2 + 2*y + 17 = n^2 := by
  sorry

end unique_perfect_square_polynomial_l1219_121933


namespace geometric_sequence_solution_l1219_121992

theorem geometric_sequence_solution :
  ∃! (x : ℝ), x > 0 ∧ (∃ (r : ℝ), 12 * r = x ∧ x * r = 2/3) :=
by
  -- The proof goes here
  sorry

end geometric_sequence_solution_l1219_121992


namespace interest_rate_problem_l1219_121986

/-- The interest rate problem --/
theorem interest_rate_problem
  (principal : ℝ)
  (rate_a : ℝ)
  (time : ℝ)
  (gain_b : ℝ)
  (h1 : principal = 3500)
  (h2 : rate_a = 10)
  (h3 : time = 3)
  (h4 : gain_b = 157.5)
  : ∃ (rate_c : ℝ), rate_c = 11.5 ∧
    gain_b = (principal * rate_c / 100 * time) - (principal * rate_a / 100 * time) :=
by
  sorry

end interest_rate_problem_l1219_121986


namespace stream_speed_l1219_121945

/-- Proves that given a boat with a speed of 13 km/hr in still water,
    traveling 68 km downstream in 4 hours, the speed of the stream is 4 km/hr. -/
theorem stream_speed (boat_speed : ℝ) (distance : ℝ) (time : ℝ) (stream_speed : ℝ) : 
  boat_speed = 13 →
  distance = 68 →
  time = 4 →
  distance = (boat_speed + stream_speed) * time →
  stream_speed = 4 := by
sorry

end stream_speed_l1219_121945


namespace solve_for_y_l1219_121971

theorem solve_for_y (x y : ℝ) (h1 : x^(3*y) = 8) (h2 : x = 2) : y = 1 := by
  sorry

end solve_for_y_l1219_121971


namespace intersection_union_when_m_3_intersection_equals_B_implies_m_range_l1219_121956

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | m - 2 ≤ x ∧ x ≤ m + 1}

-- Part 1
theorem intersection_union_when_m_3 :
  (A ∩ B 3) = {x | 1 ≤ x ∧ x ≤ 3} ∧
  (A ∪ B 3) = {x | -1 ≤ x ∧ x ≤ 4} := by sorry

-- Part 2
theorem intersection_equals_B_implies_m_range (m : ℝ) :
  A ∩ B m = B m → 1 ≤ m ∧ m ≤ 2 := by sorry

end intersection_union_when_m_3_intersection_equals_B_implies_m_range_l1219_121956


namespace sum_57_68_rounded_l1219_121932

/-- Rounds a number to the nearest ten -/
def roundToNearestTen (x : ℤ) : ℤ :=
  10 * ((x + 5) / 10)

/-- The sum of 57 and 68, when rounded to the nearest ten, equals 130 -/
theorem sum_57_68_rounded : roundToNearestTen (57 + 68) = 130 := by
  sorry

end sum_57_68_rounded_l1219_121932


namespace max_k_for_quadratic_roots_difference_l1219_121900

theorem max_k_for_quadratic_roots_difference (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
   x^2 + k*x - 3 = 0 ∧ 
   y^2 + k*y - 3 = 0 ∧ 
   |x - y| = 10) →
  k ≤ Real.sqrt 88 :=
sorry

end max_k_for_quadratic_roots_difference_l1219_121900


namespace marbles_left_theorem_l1219_121954

def initial_marbles : ℕ := 87
def marbles_given_away : ℕ := 8

theorem marbles_left_theorem : 
  initial_marbles - marbles_given_away = 79 := by
  sorry

end marbles_left_theorem_l1219_121954


namespace min_value_of_fraction_min_value_achieved_l1219_121951

theorem min_value_of_fraction (x : ℝ) (h : x > 9) : 
  (x^2) / (x - 9) ≥ 36 := by
sorry

theorem min_value_achieved (x : ℝ) (h : x > 9) : 
  (x^2) / (x - 9) = 36 ↔ x = 18 := by
sorry

end min_value_of_fraction_min_value_achieved_l1219_121951


namespace english_only_enrollment_l1219_121968

/-- Represents the number of students in different enrollment categories -/
structure EnrollmentCount where
  total : ℕ
  bothSubjects : ℕ
  germanTotal : ℕ

/-- Calculates the number of students enrolled only in English -/
def studentsOnlyEnglish (e : EnrollmentCount) : ℕ :=
  e.total - e.germanTotal

/-- Theorem: Given the enrollment conditions, 28 students are enrolled only in English -/
theorem english_only_enrollment (e : EnrollmentCount) 
  (h1 : e.total = 50)
  (h2 : e.bothSubjects = 12)
  (h3 : e.germanTotal = 22)
  (h4 : e.total ≥ e.germanTotal) : 
  studentsOnlyEnglish e = 28 := by
  sorry

#check english_only_enrollment

end english_only_enrollment_l1219_121968


namespace zhang_or_beibei_probability_l1219_121925

/-- The number of singers in total -/
def total_singers : ℕ := 5

/-- The number of singers to be signed -/
def singers_to_sign : ℕ := 3

/-- The probability of signing a specific combination of singers -/
def prob_combination : ℚ := 1 / (total_singers.choose singers_to_sign)

/-- The probability that either Zhang Lei or Beibei will be signed -/
def prob_zhang_or_beibei : ℚ := 1 - ((total_singers - 2).choose singers_to_sign) * prob_combination

theorem zhang_or_beibei_probability :
  prob_zhang_or_beibei = 9 / 10 := by sorry

end zhang_or_beibei_probability_l1219_121925


namespace smallest_multiple_45_div_3_l1219_121985

theorem smallest_multiple_45_div_3 : 
  ∀ n : ℕ, n > 0 ∧ 45 ∣ n ∧ 3 ∣ n → n ≥ 45 := by
sorry

end smallest_multiple_45_div_3_l1219_121985


namespace yonder_license_plates_l1219_121996

/-- The number of possible letters in a license plate position -/
def num_letters : ℕ := 26

/-- The number of possible digits in a license plate position -/
def num_digits : ℕ := 10

/-- The total number of possible license plates in Yonder -/
def total_license_plates : ℕ := num_letters ^ 3 * num_digits ^ 3

theorem yonder_license_plates :
  total_license_plates = 17576000 := by sorry

end yonder_license_plates_l1219_121996


namespace f_increasing_and_not_in_second_quadrant_l1219_121980

-- Define the function
def f (x : ℝ) : ℝ := 2 * x - 5

-- State the theorem
theorem f_increasing_and_not_in_second_quadrant :
  (∀ x y : ℝ, x < y → f x < f y) ∧ 
  (∀ x y : ℝ, x < 0 ∧ y > 0 → ¬(f x = y)) :=
sorry

end f_increasing_and_not_in_second_quadrant_l1219_121980


namespace tia_walking_time_l1219_121957

/-- Represents a person's walking characteristics and time to destination -/
structure Walker where
  steps_per_minute : ℝ
  step_length : ℝ
  time_to_destination : ℝ

/-- Calculates the distance walked based on walking characteristics and time -/
def distance (w : Walker) : ℝ :=
  w.steps_per_minute * w.step_length * w.time_to_destination

theorem tia_walking_time (ella tia : Walker)
  (h1 : ella.steps_per_minute = 80)
  (h2 : ella.step_length = 80)
  (h3 : ella.time_to_destination = 20)
  (h4 : tia.steps_per_minute = 120)
  (h5 : tia.step_length = 70)
  (h6 : distance ella = distance tia) :
  tia.time_to_destination = 15.24 := by
  sorry

end tia_walking_time_l1219_121957


namespace range_of_a_l1219_121905

theorem range_of_a (x a : ℝ) : 
  (∀ x, -x^2 + 5*x - 6 > 0 → |x - a| < 4) ∧ 
  (∃ x, |x - a| < 4 ∧ -x^2 + 5*x - 6 ≤ 0) →
  a ∈ Set.Icc (-1 : ℝ) 6 :=
sorry

end range_of_a_l1219_121905


namespace integral_equals_ln5_over_8_l1219_121927

/-- The definite integral of the given function from 0 to 1 is equal to (1/8) * ln(5) -/
theorem integral_equals_ln5_over_8 :
  ∫ x in (0 : ℝ)..1, (4 * Real.sqrt (1 - x) - Real.sqrt (x + 1)) /
    ((Real.sqrt (x + 1) + 4 * Real.sqrt (1 - x)) * (x + 1)^2) = (1/8) * Real.log 5 := by
  sorry

end integral_equals_ln5_over_8_l1219_121927


namespace preimage_of_neg_one_plus_two_i_l1219_121976

/-- The complex transformation f(Z) = (1+i)Z -/
def f (Z : ℂ) : ℂ := (1 + Complex.I) * Z

/-- Theorem: The pre-image of -1+2i under f is (1+3i)/2 -/
theorem preimage_of_neg_one_plus_two_i :
  f ((1 + 3 * Complex.I) / 2) = -1 + 2 * Complex.I := by
  sorry

end preimage_of_neg_one_plus_two_i_l1219_121976


namespace function_equality_implies_sum_l1219_121935

-- Define the function f
def f (x : ℝ) : ℝ := sorry

-- Define the constants a, b, and c
def a : ℝ := sorry
def b : ℝ := sorry
def c : ℝ := sorry

-- State the theorem
theorem function_equality_implies_sum (x : ℝ) :
  (∀ x, f (x + 4) = 2 * x^2 + 8 * x + 10) ∧
  (∀ x, f x = a * x^2 + b * x + c) →
  a + b + c = 4 := by sorry

end function_equality_implies_sum_l1219_121935


namespace pet_shop_grooming_time_l1219_121918

/-- The time it takes to groom all dogs in a pet shop -/
theorem pet_shop_grooming_time 
  (poodle_time : ℝ) 
  (terrier_time : ℝ) 
  (num_poodles : ℕ) 
  (num_terriers : ℕ) 
  (num_employees : ℕ) 
  (h1 : poodle_time = 30) 
  (h2 : terrier_time = poodle_time / 2) 
  (h3 : num_poodles = 3) 
  (h4 : num_terriers = 8) 
  (h5 : num_employees = 4) 
  (h6 : num_employees > 0) :
  (num_poodles * poodle_time + num_terriers * terrier_time) / num_employees = 52.5 := by
  sorry


end pet_shop_grooming_time_l1219_121918


namespace total_cost_l1219_121972

/-- The cost of an enchilada -/
def e : ℝ := sorry

/-- The cost of a taco -/
def t : ℝ := sorry

/-- The cost of a burrito -/
def b : ℝ := sorry

/-- The first condition: 4 enchiladas, 5 tacos, and 2 burritos cost $8.20 -/
axiom condition1 : 4 * e + 5 * t + 2 * b = 8.20

/-- The second condition: 6 enchiladas, 3 tacos, and 4 burritos cost $9.40 -/
axiom condition2 : 6 * e + 3 * t + 4 * b = 9.40

/-- Theorem stating that the total cost of 5 enchiladas, 6 tacos, and 3 burritos is $12.20 -/
theorem total_cost : 5 * e + 6 * t + 3 * b = 12.20 := by
  sorry

end total_cost_l1219_121972


namespace original_number_exists_l1219_121940

theorem original_number_exists : ∃ x : ℝ, 3 * (2 * x + 5) = 123 := by
  sorry

end original_number_exists_l1219_121940


namespace area_of_region_l1219_121966

-- Define the circle and chord properties
def circle_radius : ℝ := 50
def chord_length : ℝ := 84
def intersection_distance : ℝ := 24

-- Define the area calculation function
def area_calculation (r : ℝ) (c : ℝ) (d : ℝ) : ℝ := sorry

-- Theorem statement
theorem area_of_region :
  area_calculation circle_radius chord_length intersection_distance = 1250 * Real.sqrt 3 + (1250 / 3) * Real.pi :=
sorry

end area_of_region_l1219_121966


namespace gravel_cost_theorem_l1219_121952

/-- The cost of gravel in dollars per cubic foot -/
def gravel_cost_per_cubic_foot : ℝ := 4

/-- The conversion factor from cubic yards to cubic feet -/
def cubic_yards_to_cubic_feet : ℝ := 27

/-- The volume of gravel in cubic yards -/
def gravel_volume_cubic_yards : ℝ := 8

/-- The total cost of gravel for a given volume in cubic yards -/
def total_cost (volume_cubic_yards : ℝ) : ℝ :=
  volume_cubic_yards * cubic_yards_to_cubic_feet * gravel_cost_per_cubic_foot

theorem gravel_cost_theorem : total_cost gravel_volume_cubic_yards = 864 := by
  sorry

end gravel_cost_theorem_l1219_121952


namespace track_meet_seating_l1219_121942

theorem track_meet_seating (children adults seniors pets seats : ℕ) : 
  children = 52 → 
  adults = 29 → 
  seniors = 15 → 
  pets = 3 → 
  seats = 95 → 
  children + adults + seniors + pets - seats = 4 := by
  sorry

end track_meet_seating_l1219_121942


namespace shaded_area_problem_l1219_121911

/-- The area of the shaded region in a figure where a 4-inch by 4-inch square 
    adjoins a 12-inch by 12-inch square. -/
theorem shaded_area_problem : 
  let small_square_side : ℝ := 4
  let large_square_side : ℝ := 12
  let small_square_area := small_square_side ^ 2
  let triangle_base := small_square_side
  let triangle_height := small_square_side * large_square_side / (large_square_side + small_square_side)
  let triangle_area := (1 / 2) * triangle_base * triangle_height
  let shaded_area := small_square_area - triangle_area
  shaded_area = 10
  := by sorry

end shaded_area_problem_l1219_121911


namespace z_absolute_value_range_l1219_121946

open Complex

theorem z_absolute_value_range (t : ℝ) :
  let z : ℂ := (sin t / Real.sqrt 2 + I * cos t) / (sin t - I * cos t / Real.sqrt 2)
  1 / Real.sqrt 2 ≤ abs z ∧ abs z ≤ Real.sqrt 2 := by
  sorry

end z_absolute_value_range_l1219_121946


namespace exponential_function_fixed_point_l1219_121994

theorem exponential_function_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^x + 1
  f 0 = 2 := by
sorry

end exponential_function_fixed_point_l1219_121994


namespace complex_fraction_simplification_l1219_121978

theorem complex_fraction_simplification :
  (2 - Complex.I) / (1 + 2 * Complex.I) = -Complex.I := by sorry

end complex_fraction_simplification_l1219_121978


namespace line_tangent_to_parabola_l1219_121901

theorem line_tangent_to_parabola :
  ∃ (m : ℝ), m = 49 ∧
  ∀ (x y : ℝ),
    (4 * x + 7 * y + m = 0) →
    (y^2 = 16 * x) →
    ∃! (x₀ y₀ : ℝ), 4 * x₀ + 7 * y₀ + m = 0 ∧ y₀^2 = 16 * x₀ :=
by sorry

end line_tangent_to_parabola_l1219_121901


namespace prank_combinations_l1219_121964

theorem prank_combinations (choices : List Nat) : 
  choices = [1, 3, 5, 6, 2] → choices.prod = 180 := by
  sorry

end prank_combinations_l1219_121964


namespace union_of_A_and_B_l1219_121961

def A : Set Int := {0, 1, 2}
def B : Set Int := {-1, 0, 1}

theorem union_of_A_and_B : A ∪ B = {-1, 0, 1, 2} := by
  sorry

end union_of_A_and_B_l1219_121961


namespace variance_of_doubled_data_l1219_121969

-- Define a set of data as a list of real numbers
def DataSet := List ℝ

-- Define the standard deviation of a data set
noncomputable def standardDeviation (data : DataSet) : ℝ := sorry

-- Define the variance of a data set
noncomputable def variance (data : DataSet) : ℝ := sorry

-- Define a function to double each element in a data set
def doubleData (data : DataSet) : DataSet := data.map (· * 2)

-- Theorem statement
theorem variance_of_doubled_data (data : DataSet) :
  let s := standardDeviation data
  variance (doubleData data) = 4 * (s ^ 2) := by sorry

end variance_of_doubled_data_l1219_121969


namespace smallest_n_congruence_l1219_121915

theorem smallest_n_congruence :
  ∃ (n : ℕ), n > 0 ∧ 5 * n ≡ 1723 [MOD 26] ∧
  ∀ (m : ℕ), m > 0 ∧ 5 * m ≡ 1723 [MOD 26] → n ≤ m :=
by sorry

end smallest_n_congruence_l1219_121915


namespace seating_probability_l1219_121912

/-- Represents a seating arrangement of 6 students in a 2x3 grid -/
def SeatingArrangement := Fin 6 → Fin 6

/-- The total number of possible seating arrangements -/
def totalArrangements : ℕ := 720

/-- Checks if three students are seated next to each other and adjacent in the same row or column -/
def isAdjacentArrangement (arr : SeatingArrangement) (a b c : Fin 6) : Prop :=
  sorry

/-- The number of arrangements where Abby, Bridget, and Chris are seated next to each other and adjacent in the same row or column -/
def favorableArrangements : ℕ := 114

/-- The probability of Abby, Bridget, and Chris being seated in a specific arrangement -/
def probability : ℚ := 19 / 120

theorem seating_probability :
  (favorableArrangements : ℚ) / totalArrangements = probability :=
sorry

end seating_probability_l1219_121912


namespace expected_sum_of_rook_positions_l1219_121919

/-- Represents a chessboard with 64 fields -/
def ChessboardSize : ℕ := 64

/-- Number of rooks placed on the board -/
def NumRooks : ℕ := 6

/-- Expected value of a single randomly chosen position -/
def ExpectedSinglePosition : ℚ := (ChessboardSize + 1) / 2

/-- Theorem: The expected value of the sum of positions of NumRooks rooks 
    on a chessboard of size ChessboardSize is NumRooks * ExpectedSinglePosition -/
theorem expected_sum_of_rook_positions :
  NumRooks * ExpectedSinglePosition = 195 := by sorry

end expected_sum_of_rook_positions_l1219_121919


namespace log_equality_l1219_121984

theorem log_equality (y : ℝ) : y = (Real.log 3 / Real.log 9) ^ (Real.log 9 / Real.log 3) → Real.log y / Real.log 4 = -1 := by
  sorry

end log_equality_l1219_121984


namespace triangle_area_l1219_121970

/-- Given a triangle with perimeter 32 cm and inradius 3.5 cm, its area is 56 cm². -/
theorem triangle_area (p r A : ℝ) (h1 : p = 32) (h2 : r = 3.5) (h3 : A = r * p / 2) : A = 56 := by
  sorry

end triangle_area_l1219_121970


namespace kitchen_length_l1219_121920

theorem kitchen_length (tile_area : ℝ) (kitchen_width : ℝ) (num_tiles : ℕ) :
  tile_area = 6 →
  kitchen_width = 48 →
  num_tiles = 96 →
  (kitchen_width * (num_tiles * tile_area / kitchen_width) : ℝ) = 12 * kitchen_width :=
by sorry

end kitchen_length_l1219_121920


namespace rubies_in_chest_l1219_121950

theorem rubies_in_chest (diamonds : ℕ) (difference : ℕ) (rubies : ℕ) : 
  diamonds = 421 → difference = 44 → diamonds = rubies + difference → rubies = 377 := by
  sorry

end rubies_in_chest_l1219_121950


namespace tree_spacing_l1219_121914

/-- Given a yard of length 400 meters with 26 equally spaced trees, including one at each end,
    the distance between consecutive trees is 16 meters. -/
theorem tree_spacing (yard_length : ℝ) (num_trees : ℕ) (h1 : yard_length = 400) (h2 : num_trees = 26) :
  yard_length / (num_trees - 1) = 16 :=
sorry

end tree_spacing_l1219_121914


namespace norris_savings_l1219_121981

/-- The amount of money Norris saved in September -/
def september_savings : ℕ := 29

/-- The amount of money Norris saved in October -/
def october_savings : ℕ := 25

/-- The amount of money Norris saved in November -/
def november_savings : ℕ := 31

/-- The amount of money Norris saved in December -/
def december_savings : ℕ := 35

/-- The amount of money Norris saved in January -/
def january_savings : ℕ := 40

/-- The total amount of money Norris saved from September to January -/
def total_savings : ℕ := september_savings + october_savings + november_savings + december_savings + january_savings

theorem norris_savings : total_savings = 160 := by
  sorry

end norris_savings_l1219_121981


namespace arithmetic_sequence_geometric_mean_l1219_121909

/-- An arithmetic sequence with common difference d ≠ 0 and first term a₁ = 2d -/
def arithmetic_sequence (d : ℝ) (n : ℕ) : ℝ :=
  2 * d + (n - 1) * d

theorem arithmetic_sequence_geometric_mean (d : ℝ) (k : ℕ) (h : d ≠ 0) :
  (arithmetic_sequence d k) ^ 2 = (arithmetic_sequence d 1) * (arithmetic_sequence d (2 * k + 7)) →
  k = 5 := by
sorry

end arithmetic_sequence_geometric_mean_l1219_121909


namespace certain_number_subtraction_l1219_121928

theorem certain_number_subtraction (x : ℤ) : x + 468 = 954 → x - 3 = 483 := by
  sorry

end certain_number_subtraction_l1219_121928


namespace corresponding_angles_relationships_l1219_121958

/-- Two angles are corresponding if they occupy the same relative position at each intersection where a straight line crosses two others. -/
def corresponding_angles (α β : Real) : Prop := sorry

/-- The statement that all relationships (equal, greater than, less than) are possible for corresponding angles. -/
theorem corresponding_angles_relationships (α β : Real) (h : corresponding_angles α β) :
  (∃ (α₁ β₁ : Real), corresponding_angles α₁ β₁ ∧ α₁ = β₁) ∧
  (∃ (α₂ β₂ : Real), corresponding_angles α₂ β₂ ∧ α₂ > β₂) ∧
  (∃ (α₃ β₃ : Real), corresponding_angles α₃ β₃ ∧ α₃ < β₃) :=
sorry

end corresponding_angles_relationships_l1219_121958


namespace sqrt_63_minus_7_sqrt_one_seventh_l1219_121962

theorem sqrt_63_minus_7_sqrt_one_seventh (x : ℝ) : 
  Real.sqrt 63 - 7 * Real.sqrt (1 / 7) = 2 * Real.sqrt 7 := by
  sorry

end sqrt_63_minus_7_sqrt_one_seventh_l1219_121962


namespace polynomial_division_remainder_l1219_121960

theorem polynomial_division_remainder : 
  ∃ q : Polynomial ℚ, x^4 = (x^3 + 3*x^2 + 2*x + 1) * q + (-x^2 - x - 1) := by
  sorry

end polynomial_division_remainder_l1219_121960


namespace exam_pass_probability_l1219_121929

/-- The probability of passing an exam given the following conditions:
  - There are 5 total questions
  - The candidate is familiar with 3 questions
  - The candidate randomly selects 3 questions to answer
  - The candidate needs to answer 2 questions correctly to pass
-/
theorem exam_pass_probability :
  let total_questions : ℕ := 5
  let familiar_questions : ℕ := 3
  let selected_questions : ℕ := 3
  let required_correct : ℕ := 2
  let pass_probability : ℚ := 7 / 10
  (Nat.choose familiar_questions selected_questions +
   Nat.choose familiar_questions (selected_questions - 1) * Nat.choose (total_questions - familiar_questions) 1) /
  Nat.choose total_questions selected_questions = pass_probability :=
by sorry

end exam_pass_probability_l1219_121929


namespace exponential_function_property_l1219_121926

theorem exponential_function_property (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  (∀ x ∈ Set.Icc 0 2, a^x ≤ 1) ∧
  (∀ x ∈ Set.Icc 0 2, a^x ≥ a^2) ∧
  (1 - a^2 = 3/4) →
  a = 1/2 := by
sorry

end exponential_function_property_l1219_121926


namespace company_picnic_attendance_l1219_121999

theorem company_picnic_attendance 
  (total_employees : ℕ) 
  (men_percentage : ℝ) 
  (women_percentage : ℝ) 
  (men_attendance_rate : ℝ) 
  (women_attendance_rate : ℝ) 
  (h1 : men_percentage = 0.35) 
  (h2 : women_percentage = 1 - men_percentage) 
  (h3 : men_attendance_rate = 0.2) 
  (h4 : women_attendance_rate = 0.4) : 
  (men_percentage * men_attendance_rate + women_percentage * women_attendance_rate) * 100 = 33 := by
  sorry

#check company_picnic_attendance

end company_picnic_attendance_l1219_121999


namespace fraction_sum_equality_l1219_121937

theorem fraction_sum_equality (a b c : ℝ) 
  (h : a / (30 - a) + b / (70 - b) + c / (75 - c) = 9) :
  6 / (30 - a) + 14 / (70 - b) + 15 / (75 - c) = 2.4 := by
  sorry

end fraction_sum_equality_l1219_121937


namespace x_squared_is_quadratic_l1219_121904

/-- A quadratic equation in one variable is of the form ax² + bx + c = 0, where a ≠ 0 -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function f(x) = x² is a quadratic equation in one variable -/
theorem x_squared_is_quadratic : is_quadratic_equation (λ x : ℝ => x^2) := by
  sorry

#check x_squared_is_quadratic

end x_squared_is_quadratic_l1219_121904


namespace necessary_not_sufficient_l1219_121973

theorem necessary_not_sufficient (p q : Prop) :
  (¬p → ¬(p ∨ q)) ∧ ¬(¬p → ¬(p ∨ q)) :=
by sorry

end necessary_not_sufficient_l1219_121973


namespace tetrahedron_cube_volume_ratio_l1219_121975

theorem tetrahedron_cube_volume_ratio :
  let cube_side : ℝ := x
  let cube_volume := cube_side ^ 3
  let tetrahedron_side := cube_side * Real.sqrt 3 / 2
  let tetrahedron_volume := tetrahedron_side ^ 3 * Real.sqrt 2 / 12
  tetrahedron_volume / cube_volume = Real.sqrt 6 / 32 := by
  sorry

end tetrahedron_cube_volume_ratio_l1219_121975


namespace digits_for_369_pages_l1219_121910

/-- The total number of digits used to number pages in a book -/
def totalDigits (n : ℕ) : ℕ :=
  (min n 9) + 
  (max (min n 99 - 9) 0) * 2 + 
  (max (n - 99) 0) * 3

/-- Theorem: The total number of digits used in numbering the pages of a book with 369 pages is 999 -/
theorem digits_for_369_pages : totalDigits 369 = 999 := by
  sorry

end digits_for_369_pages_l1219_121910


namespace sam_and_mary_balloons_l1219_121963

/-- The total number of yellow balloons Sam and Mary have together -/
def total_balloons (sam_initial : ℝ) (sam_given : ℝ) (mary : ℝ) : ℝ :=
  (sam_initial - sam_given) + mary

/-- Proof that Sam and Mary have 8.0 yellow balloons in total -/
theorem sam_and_mary_balloons :
  total_balloons 6.0 5.0 7.0 = 8.0 := by
  sorry

end sam_and_mary_balloons_l1219_121963


namespace divisibility_proof_l1219_121967

theorem divisibility_proof (k : ℕ) (p : ℕ) (m : ℕ) 
  (h1 : k > 1) 
  (h2 : p = 6 * k + 1) 
  (h3 : Nat.Prime p) 
  (h4 : m = 2^p - 1) : 
  (127 * m) ∣ (2^(m-1) - 1) := by
  sorry

end divisibility_proof_l1219_121967


namespace girls_to_boys_ratio_l1219_121965

theorem girls_to_boys_ratio (girls boys : ℕ) (h1 : girls = 10) (h2 : boys = 20) :
  (girls : ℚ) / (boys : ℚ) = 1 / 2 := by
  sorry

end girls_to_boys_ratio_l1219_121965


namespace sum_of_three_consecutive_integers_l1219_121979

theorem sum_of_three_consecutive_integers (a b c : ℕ) : 
  (a + 1 = b ∧ b + 1 = c) → c = 7 → a + b + c = 18 := by
  sorry

end sum_of_three_consecutive_integers_l1219_121979


namespace max_min_f_on_interval_l1219_121923

def f (x : ℝ) : ℝ := 3 * x^4 + 4 * x^3 + 34

theorem max_min_f_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (-2 : ℝ) 1, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-2 : ℝ) 1, f x = max) ∧
    (∀ x ∈ Set.Icc (-2 : ℝ) 1, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (-2 : ℝ) 1, f x = min) ∧
    max = 50 ∧ min = 33 :=
by sorry

end max_min_f_on_interval_l1219_121923


namespace unique_triples_l1219_121917

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem unique_triples : 
  ∀ a b c : ℕ,
    (is_prime (a^2 - 23)) →
    (is_prime (b^2 - 23)) →
    ((a^2 - 23) * (b^2 - 23) = c^2 - 23) →
    ((a = 5 ∧ b = 6 ∧ c = 7) ∨ (a = 6 ∧ b = 5 ∧ c = 7)) :=
by sorry

end unique_triples_l1219_121917


namespace no_three_common_tangents_l1219_121977

/-- Two circles in the same plane with different radii -/
structure TwoCircles where
  plane : Type*
  circle1 : Set plane
  circle2 : Set plane
  radius1 : ℝ
  radius2 : ℝ
  different_radii : radius1 ≠ radius2

/-- A common tangent to two circles -/
def CommonTangent (tc : TwoCircles) (line : Set tc.plane) : Prop := sorry

/-- The number of common tangents to two circles -/
def NumCommonTangents (tc : TwoCircles) : ℕ := sorry

/-- Theorem: Two circles with different radii cannot have exactly 3 common tangents -/
theorem no_three_common_tangents (tc : TwoCircles) : 
  NumCommonTangents tc ≠ 3 := by sorry

end no_three_common_tangents_l1219_121977


namespace sum_of_squares_l1219_121913

theorem sum_of_squares (a b : ℝ) (h1 : a - b = 10) (h2 : a * b = 25) : a^2 + b^2 = 150 := by
  sorry

end sum_of_squares_l1219_121913


namespace line_system_properties_l1219_121934

-- Define the line system M
def line_system (θ : ℝ) (x y : ℝ) : Prop :=
  x * Real.cos θ + y * Real.sin θ = 1

-- Define the region enclosed by the lines
def enclosed_region (p : ℝ × ℝ) : Prop :=
  ∃ θ, line_system θ p.1 p.2

-- Theorem statement
theorem line_system_properties :
  -- 1. The area of the region enclosed by the lines is π
  (∃ A : Set (ℝ × ℝ), (∀ p, p ∈ A ↔ enclosed_region p) ∧ MeasureTheory.volume A = π) ∧
  -- 2. Not all lines in the system are parallel
  (∃ θ₁ θ₂, θ₁ ≠ θ₂ ∧ ¬ (∀ x y, line_system θ₁ x y ↔ line_system θ₂ x y)) ∧
  -- 3. Not all lines in the system pass through a fixed point
  (¬ ∃ p : ℝ × ℝ, ∀ θ, line_system θ p.1 p.2) ∧
  -- 4. For any integer n ≥ 3, there exists a regular n-gon with edges on the lines of the system
  (∀ n : ℕ, n ≥ 3 → ∃ vertices : Fin n → ℝ × ℝ,
    (∀ i : Fin n, ∃ θ, line_system θ (vertices i).1 (vertices i).2) ∧
    (∀ i j : Fin n, (vertices i).1^2 + (vertices i).2^2 = (vertices j).1^2 + (vertices j).2^2) ∧
    (∀ i j : Fin n, i ≠ j → (vertices i).1 ≠ (vertices j).1 ∨ (vertices i).2 ≠ (vertices j).2)) :=
by sorry


end line_system_properties_l1219_121934


namespace sin_linear_dependence_l1219_121995

theorem sin_linear_dependence :
  ∃ (α₁ α₂ α₃ : ℝ), (α₁ ≠ 0 ∨ α₂ ≠ 0 ∨ α₃ ≠ 0) ∧
  ∀ x : ℝ, α₁ * Real.sin x + α₂ * Real.sin (x + π/8) + α₃ * Real.sin (x - π/8) = 0 := by
sorry

end sin_linear_dependence_l1219_121995


namespace dog_tricks_conversion_l1219_121902

def base5_to_base10 (a b c d : ℕ) : ℕ :=
  d * 5^0 + c * 5^1 + b * 5^2 + a * 5^3

theorem dog_tricks_conversion :
  base5_to_base10 1 2 3 4 = 194 := by
  sorry

end dog_tricks_conversion_l1219_121902


namespace estimate_smaller_than_exact_l1219_121924

theorem estimate_smaller_than_exact (a b c d a' b' c' d' : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (ha' : 0 < a' ∧ a' ≤ a) (hb' : 0 < b' ∧ b ≤ b')
  (hc' : 0 < c' ∧ c' ≤ c) (hd' : 0 < d' ∧ d ≤ d') :
  d' * (a' / b') + c' < d * (a / b) + c := by
  sorry

end estimate_smaller_than_exact_l1219_121924
