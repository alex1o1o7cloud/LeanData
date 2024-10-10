import Mathlib

namespace parametric_to_standard_equation_l342_34241

/-- Given parametric equations x = 1 + (1/2)t and y = 5 + (√3/2)t,
    prove they are equivalent to the standard equation √3x - y + 5 - √3 = 0 -/
theorem parametric_to_standard_equation 
  (t x y : ℝ) 
  (h1 : x = 1 + (1/2) * t) 
  (h2 : y = 5 + (Real.sqrt 3 / 2) * t) :
  Real.sqrt 3 * x - y + 5 - Real.sqrt 3 = 0 :=
sorry

end parametric_to_standard_equation_l342_34241


namespace employee_bonuses_l342_34258

theorem employee_bonuses :
  ∃ (x y z : ℝ), 
    x > 0 ∧ y > 0 ∧ z > 0 ∧
    x + y + z = 2970 ∧
    y = (1/3) * x + 180 ∧
    z = (1/3) * y + 130 ∧
    x = 1800 ∧ y = 780 ∧ z = 390 := by
  sorry

end employee_bonuses_l342_34258


namespace f_log_4_9_l342_34235

/-- A function that is even and equals 2^x for negative x -/
def f (x : ℝ) : ℝ := sorry

/-- f is an even function -/
axiom f_even : ∀ x : ℝ, f x = f (-x)

/-- f(x) = 2^x for x < 0 -/
axiom f_neg : ∀ x : ℝ, x < 0 → f x = 2^x

/-- The main theorem: f(log_4(9)) = 1/3 -/
theorem f_log_4_9 : f (Real.log 9 / Real.log 4) = 1/3 := by sorry

end f_log_4_9_l342_34235


namespace expression_value_l342_34233

theorem expression_value : 
  ∃ (m : ℕ), (3^1005 + 7^1006)^2 - (3^1005 - 7^1006)^2 = m * 10^1006 ∧ m = 280 := by
  sorry

end expression_value_l342_34233


namespace fraction_simplification_l342_34232

theorem fraction_simplification (m : ℝ) (h : m ≠ 0) : 
  (3 * m^3) / (6 * m^2) = m / 2 := by sorry

end fraction_simplification_l342_34232


namespace female_listeners_count_l342_34215

/-- Represents the survey results from radio station KMAT -/
structure SurveyResults where
  total_listeners : Nat
  total_non_listeners : Nat
  male_listeners : Nat
  male_non_listeners : Nat
  female_non_listeners : Nat
  undeclared_listeners : Nat
  undeclared_non_listeners : Nat

/-- Calculates the number of female listeners based on the survey results -/
def female_listeners (results : SurveyResults) : Nat :=
  results.total_listeners - results.male_listeners - results.undeclared_listeners

/-- Theorem stating that the number of female listeners is 65 -/
theorem female_listeners_count (results : SurveyResults)
  (h1 : results.total_listeners = 160)
  (h2 : results.total_non_listeners = 235)
  (h3 : results.male_listeners = 75)
  (h4 : results.male_non_listeners = 85)
  (h5 : results.female_non_listeners = 135)
  (h6 : results.undeclared_listeners = 20)
  (h7 : results.undeclared_non_listeners = 15) :
  female_listeners results = 65 := by
  sorry

#check female_listeners_count

end female_listeners_count_l342_34215


namespace nabla_four_seven_l342_34225

-- Define the nabla operation
def nabla (a b : ℚ) : ℚ := (a + b) / (1 + a * b)

-- Theorem statement
theorem nabla_four_seven : nabla 4 7 = 11 / 29 := by
  sorry

end nabla_four_seven_l342_34225


namespace division_of_decimals_l342_34291

theorem division_of_decimals : (0.45 : ℝ) / 0.005 = 90 := by sorry

end division_of_decimals_l342_34291


namespace circle_y_axis_intersection_sum_l342_34207

theorem circle_y_axis_intersection_sum (h k r : ℝ) : 
  h = -3 → k = 5 → r = 8 → 
  (k + (r^2 - h^2).sqrt) + (k - (r^2 - h^2).sqrt) = 10 := by sorry

end circle_y_axis_intersection_sum_l342_34207


namespace simplify_expression_l342_34252

theorem simplify_expression (x : ℝ) : 3*x + 4 - x + 8 = 2*x + 12 := by
  sorry

end simplify_expression_l342_34252


namespace symmetric_scanning_codes_count_l342_34201

/-- Represents a color of a square in the grid -/
inductive Color
| Black
| White

/-- Represents a square in the 8x8 grid -/
structure Square where
  row : Fin 8
  col : Fin 8
  color : Color

/-- Represents the 8x8 grid -/
def Grid := Array (Array Square)

/-- Checks if a square has at least one adjacent square of each color -/
def hasAdjacentColors (grid : Grid) (square : Square) : Prop :=
  sorry

/-- Checks if the grid is symmetric under 90 degree rotation -/
def isSymmetricUnder90Rotation (grid : Grid) : Prop :=
  sorry

/-- Checks if the grid is symmetric under 180 degree rotation -/
def isSymmetricUnder180Rotation (grid : Grid) : Prop :=
  sorry

/-- Checks if the grid is symmetric under 270 degree rotation -/
def isSymmetricUnder270Rotation (grid : Grid) : Prop :=
  sorry

/-- Checks if the grid is symmetric under reflection across midpoint lines -/
def isSymmetricUnderMidpointReflection (grid : Grid) : Prop :=
  sorry

/-- Checks if the grid is symmetric under reflection across diagonals -/
def isSymmetricUnderDiagonalReflection (grid : Grid) : Prop :=
  sorry

/-- Checks if the grid satisfies all symmetry conditions -/
def isSymmetric (grid : Grid) : Prop :=
  isSymmetricUnder90Rotation grid ∧
  isSymmetricUnder180Rotation grid ∧
  isSymmetricUnder270Rotation grid ∧
  isSymmetricUnderMidpointReflection grid ∧
  isSymmetricUnderDiagonalReflection grid

/-- Counts the number of symmetric scanning codes -/
def countSymmetricCodes : Nat :=
  sorry

/-- The main theorem stating that the number of symmetric scanning codes is 254 -/
theorem symmetric_scanning_codes_count :
  countSymmetricCodes = 254 :=
sorry

end symmetric_scanning_codes_count_l342_34201


namespace camp_men_count_l342_34219

/-- The number of days the food lasts initially -/
def initial_days : ℕ := 50

/-- The number of days the food lasts after more men join -/
def final_days : ℕ := 25

/-- The number of additional men who join -/
def additional_men : ℕ := 10

/-- The initial number of men in the camp -/
def initial_men : ℕ := 10

theorem camp_men_count :
  ∀ (food : ℕ),
  food = initial_men * initial_days ∧
  food = (initial_men + additional_men) * final_days →
  initial_men = 10 := by
sorry

end camp_men_count_l342_34219


namespace probability_of_being_leader_l342_34205

theorem probability_of_being_leader (total_people : ℕ) (num_groups : ℕ) 
  (h1 : total_people = 12) 
  (h2 : num_groups = 2) 
  (h3 : total_people % num_groups = 0) : 
  (1 : ℚ) / (total_people / num_groups) = 1/6 :=
sorry

end probability_of_being_leader_l342_34205


namespace triangle_ratio_l342_34220

theorem triangle_ratio (A B C : ℝ) (a b c : ℝ) :
  A = 2 * π / 3 →  -- 120° in radians
  b = 1 →
  (1 / 2) * c * b * Real.sin A = Real.sqrt 3 →
  (b + c) / (Real.sin B + Real.sin C) = 2 * Real.sqrt 7 := by
  sorry

end triangle_ratio_l342_34220


namespace factorization_equality_l342_34247

theorem factorization_equality (x y : ℝ) :
  (2*x - y) * (x + 3*y) - (2*x + 3*y) * (y - 2*x) = 3 * (2*x - y) * (x + 2*y) := by
  sorry

end factorization_equality_l342_34247


namespace polynomial_multiplication_equality_l342_34248

-- Define the polynomials
def p (y : ℝ) : ℝ := 2*y - 1
def q (y : ℝ) : ℝ := 5*y^12 - 3*y^11 + y^9 - 4*y^8
def r (y : ℝ) : ℝ := 10*y^13 - 11*y^12 + 3*y^11 + y^10 - 9*y^9 + 4*y^8

-- Theorem statement
theorem polynomial_multiplication_equality :
  ∀ y : ℝ, p y * q y = r y :=
by sorry

end polynomial_multiplication_equality_l342_34248


namespace petya_larger_than_vasya_l342_34266

theorem petya_larger_than_vasya : 2^25 > 4^12 := by
  sorry

end petya_larger_than_vasya_l342_34266


namespace circle_center_and_radius_l342_34264

theorem circle_center_and_radius :
  ∀ (x y : ℝ), (x - 1)^2 + (y + 5)^2 = 3 →
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (1, -5) ∧ radius = Real.sqrt 3 ∧
    (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end circle_center_and_radius_l342_34264


namespace chair_price_proof_l342_34299

/-- The normal price of a chair -/
def normal_price : ℝ := 20

/-- The discounted price for the first 5 chairs -/
def discounted_price_first_5 : ℝ := 0.75 * normal_price

/-- The discounted price for chairs after the first 5 -/
def discounted_price_after_5 : ℝ := 0.5 * normal_price

/-- The number of chairs bought -/
def chairs_bought : ℕ := 8

/-- The total cost of all chairs bought -/
def total_cost : ℝ := 105

theorem chair_price_proof :
  5 * discounted_price_first_5 + (chairs_bought - 5) * discounted_price_after_5 = total_cost :=
sorry

end chair_price_proof_l342_34299


namespace simplify_expression_l342_34218

theorem simplify_expression : (2^8 + 7^3) * (2^2 - (-2)^3)^5 = 149062368 := by
  sorry

end simplify_expression_l342_34218


namespace smallest_number_l342_34230

-- Define a function to convert a number from any base to decimal
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

-- Define the numbers in their respective bases
def num1 : List Nat := [8, 5]
def base1 : Nat := 9

def num2 : List Nat := [2, 1, 0]
def base2 : Nat := 6

def num3 : List Nat := [1, 0, 0, 0]
def base3 : Nat := 4

def num4 : List Nat := [1, 1, 1, 1, 1, 1, 1]
def base4 : Nat := 2

-- Theorem statement
theorem smallest_number :
  to_decimal num3 base3 < to_decimal num1 base1 ∧
  to_decimal num3 base3 < to_decimal num2 base2 ∧
  to_decimal num3 base3 < to_decimal num4 base4 := by
  sorry

end smallest_number_l342_34230


namespace room_width_is_correct_l342_34245

/-- The width of a room satisfying given conditions -/
def room_width : ℝ :=
  let length : ℝ := 25
  let height : ℝ := 12
  let door_area : ℝ := 6 * 3
  let window_area : ℝ := 4 * 3
  let num_windows : ℕ := 3
  let whitewash_cost_per_sqft : ℝ := 3
  let total_whitewash_cost : ℝ := 2718
  15

/-- Theorem stating that the room width is correct given the conditions -/
theorem room_width_is_correct :
  let length : ℝ := 25
  let height : ℝ := 12
  let door_area : ℝ := 6 * 3
  let window_area : ℝ := 4 * 3
  let num_windows : ℕ := 3
  let whitewash_cost_per_sqft : ℝ := 3
  let total_whitewash_cost : ℝ := 2718
  let width := room_width
  whitewash_cost_per_sqft * (2 * (length * height) + 2 * (width * height) - door_area - num_windows * window_area) = total_whitewash_cost :=
by
  sorry


end room_width_is_correct_l342_34245


namespace amount_to_hand_in_l342_34229

/-- Represents the contents of Jack's till -/
structure TillContents where
  usd_100: Nat
  usd_50: Nat
  usd_20: Nat
  usd_10: Nat
  usd_5: Nat
  usd_1: Nat
  quarters: Nat
  dimes: Nat
  nickels: Nat
  pennies: Nat
  euro_5: Nat
  gbp_10: Nat

/-- Exchange rates -/
def euro_to_usd : Rat := 118/100
def gbp_to_usd : Rat := 139/100

/-- The amount to be left in the till -/
def amount_to_leave : Rat := 300

/-- Calculate the total amount in USD -/
def total_amount (contents : TillContents) : Rat :=
  contents.usd_100 * 100 +
  contents.usd_50 * 50 +
  contents.usd_20 * 20 +
  contents.usd_10 * 10 +
  contents.usd_5 * 5 +
  contents.usd_1 +
  contents.quarters * (1/4) +
  contents.dimes * (1/10) +
  contents.nickels * (1/20) +
  contents.pennies * (1/100) +
  contents.euro_5 * 5 * euro_to_usd +
  contents.gbp_10 * 10 * gbp_to_usd

/-- Calculate the total amount of coins -/
def total_coins (contents : TillContents) : Rat :=
  contents.quarters * (1/4) +
  contents.dimes * (1/10) +
  contents.nickels * (1/20) +
  contents.pennies * (1/100)

/-- Jack's till contents -/
def jacks_till : TillContents := {
  usd_100 := 2,
  usd_50 := 1,
  usd_20 := 5,
  usd_10 := 3,
  usd_5 := 7,
  usd_1 := 27,
  quarters := 42,
  dimes := 19,
  nickels := 36,
  pennies := 47,
  euro_5 := 20,
  gbp_10 := 25
}

theorem amount_to_hand_in :
  total_amount jacks_till - (amount_to_leave + total_coins jacks_till) = 607.5 := by
  sorry

end amount_to_hand_in_l342_34229


namespace sin_increases_with_angle_sum_of_cosines_positive_l342_34283

-- Define a triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  -- Ensure the angles form a triangle
  angle_sum : A + B + C = π
  -- Ensure all sides and angles are positive
  positive_sides : a > 0 ∧ b > 0 ∧ c > 0
  positive_angles : A > 0 ∧ B > 0 ∧ C > 0

-- Theorem 1: If angle A is greater than angle B, then sin A is greater than sin B
theorem sin_increases_with_angle (t : Triangle) :
  t.A > t.B → Real.sin t.A > Real.sin t.B := by
  sorry

-- Theorem 2: The sum of cosines of all three angles is always positive
theorem sum_of_cosines_positive (t : Triangle) :
  Real.cos t.A + Real.cos t.B + Real.cos t.C > 0 := by
  sorry

end sin_increases_with_angle_sum_of_cosines_positive_l342_34283


namespace scientific_notation_equality_l342_34265

theorem scientific_notation_equality : 3790000 = 3.79 * (10 ^ 6) := by
  sorry

end scientific_notation_equality_l342_34265


namespace simplify_expression_l342_34270

theorem simplify_expression : 
  (Real.sqrt 450 / Real.sqrt 200) + (Real.sqrt 98 / Real.sqrt 49) = (3 + 2 * Real.sqrt 2) / 2 := by
sorry

end simplify_expression_l342_34270


namespace ice_distribution_proof_l342_34216

/-- Calculates the number of ice cubes per ice chest after melting --/
def ice_cubes_per_chest (initial_cubes : ℕ) (num_chests : ℕ) (melt_rate : ℕ) (hours : ℕ) : ℕ :=
  let remaining_cubes := initial_cubes - melt_rate * hours
  (remaining_cubes / num_chests : ℕ)

/-- Theorem: Given the initial conditions, each ice chest will contain 39 ice cubes --/
theorem ice_distribution_proof :
  ice_cubes_per_chest 294 7 5 3 = 39 := by
  sorry

end ice_distribution_proof_l342_34216


namespace least_three_digit_with_digit_product_8_l342_34226

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_product (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * tens * ones

theorem least_three_digit_with_digit_product_8 :
  ∀ n : ℕ, is_three_digit n → digit_product n = 8 → 181 ≤ n :=
sorry

end least_three_digit_with_digit_product_8_l342_34226


namespace multiply_and_simplify_l342_34222

theorem multiply_and_simplify (x y : ℝ) :
  (3 * x^4 - 7 * y^3) * (9 * x^8 + 21 * x^4 * y^3 + 49 * y^6) = 27 * x^12 - 343 * y^9 := by
  sorry

end multiply_and_simplify_l342_34222


namespace intersection_of_M_and_S_l342_34259

def M : Set ℕ := {x | 0 < x ∧ x < 4}
def S : Set ℕ := {2, 3, 5}

theorem intersection_of_M_and_S : M ∩ S = {2, 3} := by sorry

end intersection_of_M_and_S_l342_34259


namespace stratified_sampling_theorem_l342_34238

/-- Represents the composition of teachers in a school -/
structure TeacherComposition where
  total : ℕ
  senior : ℕ
  intermediate : ℕ
  junior : ℕ
  total_sum : total = senior + intermediate + junior

/-- Represents the sample of teachers -/
structure TeacherSample where
  size : ℕ
  senior : ℕ
  intermediate : ℕ
  junior : ℕ
  size_sum : size = senior + intermediate + junior

/-- Theorem stating the correct stratified sampling for the given teacher composition -/
theorem stratified_sampling_theorem 
  (school : TeacherComposition) 
  (sample : TeacherSample) 
  (h1 : school.total = 300) 
  (h2 : school.senior = 90) 
  (h3 : school.intermediate = 150) 
  (h4 : school.junior = 60) 
  (h5 : sample.size = 40) : 
  sample.senior = 12 ∧ sample.intermediate = 20 ∧ sample.junior = 8 := by
  sorry

end stratified_sampling_theorem_l342_34238


namespace domain_of_function_l342_34262

/-- The domain of the function f(x) = √(2x-1) / (x^2 + x - 2) -/
theorem domain_of_function (x : ℝ) : 
  x ∈ {y : ℝ | y ≥ (1/2 : ℝ) ∧ y ≠ 1} ↔ 
    (2*x - 1 ≥ 0 ∧ x^2 + x - 2 ≠ 0) :=
by sorry

end domain_of_function_l342_34262


namespace vector_sum_parallel_l342_34253

def a : ℝ × ℝ := (-1, 2)
def b (m : ℝ) : ℝ × ℝ := (m, 1)

theorem vector_sum_parallel (m : ℝ) : 
  (∃ k : ℝ, k ≠ 0 ∧ (a + 2 • b m) = k • (2 • a - b m)) → 
  (a + b m) = (-3/2, 3) := by
  sorry

end vector_sum_parallel_l342_34253


namespace distinct_groups_eq_seven_l342_34274

/-- The number of distinct groups of 3 marbles Tom can choose -/
def distinct_groups : ℕ :=
  let red_marbles : ℕ := 1
  let green_marbles : ℕ := 1
  let blue_marbles : ℕ := 1
  let yellow_marbles : ℕ := 4
  let non_yellow_marbles : ℕ := red_marbles + green_marbles + blue_marbles
  let all_yellow_groups : ℕ := 1
  let two_yellow_groups : ℕ := non_yellow_marbles
  let one_yellow_groups : ℕ := Nat.choose non_yellow_marbles 2
  all_yellow_groups + two_yellow_groups + one_yellow_groups

theorem distinct_groups_eq_seven : distinct_groups = 7 := by
  sorry

end distinct_groups_eq_seven_l342_34274


namespace homes_numbering_twos_l342_34277

/-- In a city with 100 homes numbered from 1 to 100, 
    the number of 2's used in the numbering is 20. -/
theorem homes_numbering_twos (homes : Nat) (twos_used : Nat) : 
  homes = 100 → twos_used = 20 := by
  sorry

#check homes_numbering_twos

end homes_numbering_twos_l342_34277


namespace chair_probability_l342_34273

/- Define the number of chairs -/
def total_chairs : ℕ := 10

/- Define the number of broken chairs -/
def broken_chairs : ℕ := 2

/- Define the number of usable chairs -/
def usable_chairs : ℕ := total_chairs - broken_chairs

/- Define the number of adjacent pairs in usable chairs -/
def adjacent_pairs : ℕ := usable_chairs - 1 - 1  -- Subtract 1 for the gap between 4 and 7

/- Define the probability of not sitting next to each other -/
def prob_not_adjacent : ℚ := 11 / 14

theorem chair_probability : 
  prob_not_adjacent = 1 - (adjacent_pairs : ℚ) / (usable_chairs.choose 2) :=
by sorry

end chair_probability_l342_34273


namespace f_nonnegative_when_a_is_one_f_has_two_zeros_iff_a_in_open_unit_interval_l342_34243

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 1 - 2 * log x

theorem f_nonnegative_when_a_is_one (x : ℝ) (h : x > 0) :
  f 1 x ≥ 0 := by sorry

theorem f_has_two_zeros_iff_a_in_open_unit_interval (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ f a x₁ = 0 ∧ f a x₂ = 0) ↔
  0 < a ∧ a < 1 := by sorry

end f_nonnegative_when_a_is_one_f_has_two_zeros_iff_a_in_open_unit_interval_l342_34243


namespace y_relationship_l342_34284

/-- A quadratic function of the form y = -x² + 2x + c -/
def quadratic (c : ℝ) (x : ℝ) : ℝ := -x^2 + 2*x + c

/-- Theorem stating the relationship between y-coordinates of three points on the quadratic function -/
theorem y_relationship (c : ℝ) (y₁ y₂ y₃ : ℝ) 
  (h₁ : y₁ = quadratic c (-1))
  (h₂ : y₂ = quadratic c 2)
  (h₃ : y₃ = quadratic c 5) :
  y₂ > y₁ ∧ y₁ > y₃ := by
  sorry


end y_relationship_l342_34284


namespace lottery_expected_months_l342_34210

/-- Represents the lottery system for car permits -/
structure LotterySystem where
  initial_participants : ℕ
  permits_per_month : ℕ
  new_participants_per_month : ℕ

/-- Calculates the expected number of months to win a permit with constant probability -/
def expected_months_constant (system : LotterySystem) : ℝ :=
  10 -- The actual calculation is omitted

/-- Calculates the expected number of months to win a permit with quarterly variable probabilities -/
def expected_months_variable (system : LotterySystem) : ℝ :=
  10 -- The actual calculation is omitted

/-- The main theorem stating that both lottery systems result in an expected 10 months wait -/
theorem lottery_expected_months (system : LotterySystem) 
    (h1 : system.initial_participants = 300000)
    (h2 : system.permits_per_month = 30000)
    (h3 : system.new_participants_per_month = 30000) :
    expected_months_constant system = 10 ∧ expected_months_variable system = 10 := by
  sorry

#check lottery_expected_months

end lottery_expected_months_l342_34210


namespace roots_transformation_l342_34261

-- Define the original polynomial
def original_poly (x : ℝ) : ℝ := x^3 - 4*x^2 + 5

-- Define the roots of the original polynomial
def roots_original : Set ℝ := {r | original_poly r = 0}

-- Define the new polynomial
def new_poly (x : ℝ) : ℝ := x^3 - 12*x^2 + 135

-- Define the roots of the new polynomial
def roots_new : Set ℝ := {r | new_poly r = 0}

-- State the theorem
theorem roots_transformation :
  ∃ (r₁ r₂ r₃ : ℝ), roots_original = {r₁, r₂, r₃} →
    roots_new = {3*r₁, 3*r₂, 3*r₃} :=
sorry

end roots_transformation_l342_34261


namespace functional_polynomial_characterization_l342_34285

/-- A polynomial that satisfies the given functional equation -/
def FunctionalPolynomial (p : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, 1 + p x = (p (x - 1) + p (x + 1)) / 2

theorem functional_polynomial_characterization :
  ∀ p : ℝ → ℝ, FunctionalPolynomial p →
  ∃ b c : ℝ, ∀ x : ℝ, p x = x^2 + b*x + c :=
sorry

end functional_polynomial_characterization_l342_34285


namespace cookies_remaining_l342_34276

/-- Represents the baked goods scenario --/
structure BakedGoods where
  cookies : ℕ
  brownies : ℕ
  cookie_price : ℚ
  brownie_price : ℚ

/-- Calculates the total value of baked goods --/
def total_value (bg : BakedGoods) : ℚ :=
  bg.cookies * bg.cookie_price + bg.brownies * bg.brownie_price

/-- Theorem stating the number of cookies remaining --/
theorem cookies_remaining (bg : BakedGoods) 
  (h1 : bg.brownies = 32)
  (h2 : bg.cookie_price = 1)
  (h3 : bg.brownie_price = 3/2)
  (h4 : total_value bg = 99) :
  bg.cookies = 51 := by
sorry


end cookies_remaining_l342_34276


namespace sum_of_ages_l342_34257

-- Define the ages of George, Christopher, and Ford
def christopher_age : ℕ := 18
def george_age : ℕ := christopher_age + 8
def ford_age : ℕ := christopher_age - 2

-- Theorem to prove
theorem sum_of_ages : george_age + christopher_age + ford_age = 60 := by
  sorry

end sum_of_ages_l342_34257


namespace catch_up_distance_l342_34206

/-- Proves that B catches up with A 100 km from the start given the specified conditions -/
theorem catch_up_distance (speed_a speed_b : ℝ) (delay : ℝ) (catch_up_dist : ℝ) : 
  speed_a = 10 →
  speed_b = 20 →
  delay = 5 →
  catch_up_dist = speed_b * (catch_up_dist / (speed_b - speed_a)) →
  catch_up_dist = speed_a * (delay + catch_up_dist / (speed_b - speed_a)) →
  catch_up_dist = 100 := by
  sorry

#check catch_up_distance

end catch_up_distance_l342_34206


namespace twenty_men_handshakes_l342_34269

/-- The number of handshakes in a complete graph with n vertices -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The theorem stating that 20 men result in 190 handshakes -/
theorem twenty_men_handshakes :
  ∃ n : ℕ, n > 0 ∧ handshakes n = 190 ∧ n = 20 := by
  sorry

#check twenty_men_handshakes

end twenty_men_handshakes_l342_34269


namespace video_recorder_markup_percentage_l342_34254

/-- Proves that the percentage markup on a video recorder's wholesale cost is 20%,
    given the wholesale cost, employee discount, and final price paid by the employee. -/
theorem video_recorder_markup_percentage
  (wholesale_cost : ℝ)
  (employee_discount_percent : ℝ)
  (employee_paid_price : ℝ)
  (h1 : wholesale_cost = 200)
  (h2 : employee_discount_percent = 10)
  (h3 : employee_paid_price = 216) :
  let retail_price := wholesale_cost * (1 + markup_percentage / 100)
  let discounted_price := retail_price * (1 - employee_discount_percent / 100)
  markup_percentage = 20 :=
sorry

end video_recorder_markup_percentage_l342_34254


namespace rhombus_perimeter_l342_34250

/-- The perimeter of a rhombus with diagonals 8 and 30 inches is 4√241 inches. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 8) (h2 : d2 = 30) :
  let s := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  4 * s = 4 * Real.sqrt 241 := by
  sorry

end rhombus_perimeter_l342_34250


namespace fixed_points_exist_l342_34236

-- Define the fixed point F and the line l
def F : ℝ × ℝ := (1, 0)
def l : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 4}

-- Define the trajectory E
def E : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 / 3 = 1}

-- Define the point A where E intersects the negative x-axis
def A : ℝ × ℝ := (-2, 0)

-- Define a function to represent a line through F that intersects E at two points
def line_through_F (m : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = m * p.2 + 1}

-- Define the theorem to be proved
theorem fixed_points_exist (m : ℝ) (hm : m ≠ 0) : 
  ∃ (B C M N : ℝ × ℝ) (Q : ℝ × ℝ),
    B ∈ E ∩ line_through_F m ∧
    C ∈ E ∩ line_through_F m ∧
    M ∈ l ∧
    N ∈ l ∧
    (Q = (1, 0) ∨ Q = (7, 0)) ∧
    ((Q.1 - M.1) * (Q.1 - N.1) + (Q.2 - M.2) * (Q.2 - N.2) = 0) :=
sorry

end fixed_points_exist_l342_34236


namespace arbitrarily_large_special_numbers_l342_34272

/-- A function that checks if all digits of a natural number are 2 or more -/
def all_digits_two_or_more (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d ≥ 2

/-- A function that checks if the product of any four digits of a number divides the number -/
def product_of_four_divides (n : ℕ) : Prop :=
  ∀ a b c d, a ∈ n.digits 10 → b ∈ n.digits 10 → c ∈ n.digits 10 → d ∈ n.digits 10 →
    (a * b * c * d) ∣ n

/-- The main theorem stating that for any k, there exists a number n > k satisfying the conditions -/
theorem arbitrarily_large_special_numbers :
  ∀ k : ℕ, ∃ n : ℕ, n > k ∧ all_digits_two_or_more n ∧ product_of_four_divides n :=
sorry

end arbitrarily_large_special_numbers_l342_34272


namespace probability_is_31_145_l342_34268

-- Define the shoe collection
def total_pairs : ℕ := 15
def black_pairs : ℕ := 8
def brown_pairs : ℕ := 5
def red_pairs : ℕ := 2

-- Define the probability function
def probability_same_color_different_foot : ℚ :=
  -- Black shoes probability
  (2 * black_pairs : ℚ) / (2 * total_pairs) * (black_pairs : ℚ) / (2 * total_pairs - 1) +
  -- Brown shoes probability
  (2 * brown_pairs : ℚ) / (2 * total_pairs) * (brown_pairs : ℚ) / (2 * total_pairs - 1) +
  -- Red shoes probability
  (2 * red_pairs : ℚ) / (2 * total_pairs) * (red_pairs : ℚ) / (2 * total_pairs - 1)

-- Theorem statement
theorem probability_is_31_145 : probability_same_color_different_foot = 31 / 145 := by
  sorry

end probability_is_31_145_l342_34268


namespace set_inclusion_l342_34208

-- Define the sets A, B, and C
def A : Set ℝ := {x | ∃ k : ℤ, x = k / 6 + 1}
def B : Set ℝ := {x | ∃ k : ℤ, x = k / 3 + 1 / 2}
def C : Set ℝ := {x | ∃ k : ℤ, x = 2 * k / 3 + 1 / 2}

-- State the theorem
theorem set_inclusion : C ⊆ B ∧ B ⊆ A := by sorry

end set_inclusion_l342_34208


namespace square_area_with_circles_8_l342_34267

/-- The area of a square containing four circles of radius r, with two circles touching each side of the square. -/
def square_area_with_circles (r : ℝ) : ℝ :=
  (4 * r) ^ 2

/-- Theorem: The area of a square containing four circles of radius 8 inches, 
    with two circles touching each side of the square, is 1024 square inches. -/
theorem square_area_with_circles_8 : 
  square_area_with_circles 8 = 1024 :=
by sorry

end square_area_with_circles_8_l342_34267


namespace ab_inequality_l342_34234

theorem ab_inequality (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hab : a + b = 2) : a * b ≤ 1 := by
  sorry

end ab_inequality_l342_34234


namespace ellipse_eccentricity_range_l342_34204

/-- An ellipse with foci F₁ and F₂ -/
structure Ellipse where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- A point on the ellipse -/
def PointOnEllipse (C : Ellipse) := ℝ × ℝ

/-- The angle F₁MF₂ for a point M on the ellipse -/
def angle (C : Ellipse) (M : PointOnEllipse C) : ℝ := sorry

/-- The eccentricity of an ellipse -/
def eccentricity (C : Ellipse) : ℝ := sorry

/-- Theorem: If there exists a point M on ellipse C such that ∠F₁MF₂ = π/3,
    then the eccentricity e of C satisfies 1/2 ≤ e < 1 -/
theorem ellipse_eccentricity_range (C : Ellipse) :
  (∃ M : PointOnEllipse C, angle C M = π / 3) →
  let e := eccentricity C
  1 / 2 ≤ e ∧ e < 1 := by
  sorry

end ellipse_eccentricity_range_l342_34204


namespace solve_for_q_l342_34217

theorem solve_for_q (m n q : ℚ) : 
  (7/8 : ℚ) = m/96 ∧ 
  (7/8 : ℚ) = (n + m)/112 ∧ 
  (7/8 : ℚ) = (q - m)/144 → 
  q = 210 := by
sorry

end solve_for_q_l342_34217


namespace triangle_with_120_degree_angle_divisible_into_isosceles_l342_34263

-- Define a triangle type
structure Triangle :=
  (a b c : ℝ)
  (sum_to_180 : a + b + c = 180)
  (all_positive : 0 < a ∧ 0 < b ∧ 0 < c)

-- Define an isosceles triangle
def IsIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

-- Define the property of being divisible into two isosceles triangles
def DivisibleIntoTwoIsosceles (t : Triangle) : Prop :=
  ∃ (t1 t2 : Triangle), IsIsosceles t1 ∧ IsIsosceles t2

-- The main theorem
theorem triangle_with_120_degree_angle_divisible_into_isosceles
  (t : Triangle)
  (has_120_degree : t.a = 120 ∨ t.b = 120 ∨ t.c = 120)
  (divisible : DivisibleIntoTwoIsosceles t) :
  (t.b = 30 ∧ t.c = 15) ∨ (t.b = 45 ∧ t.c = 15) ∨
  (t.b = 15 ∧ t.c = 30) ∨ (t.b = 15 ∧ t.c = 45) :=
by sorry


end triangle_with_120_degree_angle_divisible_into_isosceles_l342_34263


namespace range_of_2a_minus_b_l342_34203

theorem range_of_2a_minus_b (a b : ℝ) (ha : 1 < a ∧ a < 3) (hb : 2 < b ∧ b < 4) :
  -2 < 2*a - b ∧ 2*a - b < 4 := by
  sorry

end range_of_2a_minus_b_l342_34203


namespace middle_carriages_passengers_l342_34292

/-- Represents a train with carriages and passengers -/
structure Train where
  num_carriages : Nat
  total_passengers : Nat
  block_passengers : Nat
  block_size : Nat

/-- Calculates the number of passengers in the middle two carriages -/
def middle_two_passengers (t : Train) : Nat :=
  t.total_passengers - (4 * t.block_passengers - 3 * t.total_passengers)

/-- Theorem stating that for a train with given specifications, 
    the middle two carriages contain 96 passengers -/
theorem middle_carriages_passengers 
  (t : Train) 
  (h1 : t.num_carriages = 18) 
  (h2 : t.total_passengers = 700) 
  (h3 : t.block_passengers = 199) 
  (h4 : t.block_size = 5) : 
  middle_two_passengers t = 96 := by
  sorry

end middle_carriages_passengers_l342_34292


namespace triangle_angle_value_l342_34278

theorem triangle_angle_value (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  (a^2 + c^2 - b^2) * Real.tan B = Real.sqrt 3 * a * c →
  B = π / 3 ∨ B = 2 * π / 3 := by
  sorry

end triangle_angle_value_l342_34278


namespace horner_v1_at_negative_two_l342_34255

def horner_polynomial (x : ℝ) : ℝ := x^6 - 5*x^5 + 6*x^4 + x^2 + 0.3*x + 2

def horner_v0 : ℝ := 1

def horner_v1 (x : ℝ) : ℝ := horner_v0 * x - 5

theorem horner_v1_at_negative_two :
  horner_v1 (-2) = -7 :=
sorry

end horner_v1_at_negative_two_l342_34255


namespace eugene_pencils_left_l342_34214

/-- The number of pencils Eugene has left after giving some away -/
def pencils_left (initial : Real) (given_away : Real) : Real :=
  initial - given_away

/-- Theorem: Eugene has 199.0 pencils left after giving away 35.0 pencils from his initial 234.0 pencils -/
theorem eugene_pencils_left : pencils_left 234.0 35.0 = 199.0 := by
  sorry

end eugene_pencils_left_l342_34214


namespace function_properties_l342_34200

def IsAdditive (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y

theorem function_properties
    (f : ℝ → ℝ)
    (h_additive : IsAdditive f)
    (h_neg : ∀ x : ℝ, x > 0 → f x < 0)
    (h_f_neg_one : f (-1) = 2) :
    (f 0 = 0 ∧ ∀ x : ℝ, f (-x) = -f x) ∧
    (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂) ∧
    Set.range (fun x => f x) ∩ Set.Icc (-2 : ℝ) 4 = Set.Icc (-8 : ℝ) 4 := by
  sorry

end function_properties_l342_34200


namespace root_in_interval_l342_34280

-- Define the function f(x) = x^3 - x - 1
def f (x : ℝ) : ℝ := x^3 - x - 1

-- Theorem statement
theorem root_in_interval :
  ∃ x ∈ Set.Ioo 1 2, f x = 0 :=
by
  -- The proof goes here
  sorry

end root_in_interval_l342_34280


namespace remainder_equality_l342_34221

theorem remainder_equality (P P' Q D R R' s s' : ℕ) : 
  P > P' → 
  Q > 0 → 
  P < D → P' < D → Q < D →
  R = P % D →
  R' = P' % D →
  s = (P + P') % D →
  s' = (R + R') % D →
  s = s' :=
by sorry

end remainder_equality_l342_34221


namespace knife_value_l342_34223

theorem knife_value (n : ℕ) (k : ℕ) (m : ℕ) :
  (n * n = 20 * k + 10 + m) →
  (1 ≤ m) →
  (m ≤ 9) →
  (∃ b : ℕ, 10 - b = m + b) →
  (∃ b : ℕ, b = 2) :=
by sorry

end knife_value_l342_34223


namespace tan_alpha_value_l342_34287

theorem tan_alpha_value (α : Real) 
  (h1 : Real.sin (π - α) = Real.sqrt 5 / 5)
  (h2 : π / 2 < α ∧ α < π) : 
  Real.tan α = -1/2 := by
  sorry

end tan_alpha_value_l342_34287


namespace number_equals_two_l342_34298

theorem number_equals_two : ∃ x : ℝ, 0.4 * x = 0.8 ∧ x = 2 := by sorry

end number_equals_two_l342_34298


namespace second_term_of_geometric_sequence_l342_34231

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

/-- Given a geometric sequence where the third term is 12 and the fourth term is 18, the second term is 8. -/
theorem second_term_of_geometric_sequence
    (a : ℕ → ℚ)
    (h_geometric : IsGeometricSequence a)
    (h_third_term : a 3 = 12)
    (h_fourth_term : a 4 = 18) :
    a 2 = 8 := by
  sorry

#check second_term_of_geometric_sequence

end second_term_of_geometric_sequence_l342_34231


namespace expression_evaluation_l342_34296

theorem expression_evaluation (a b c : ℚ) (h1 : a = 5) (h2 : b = -3) (h3 : c = 2) :
  3 / (a + b + c) = 3 / 4 := by
sorry

end expression_evaluation_l342_34296


namespace students_history_not_statistics_l342_34227

/-- Given a group of students with the following properties:
  * There are 89 students in total
  * 36 students are taking history
  * 32 students are taking statistics
  * 59 students are taking history or statistics or both
  This theorem proves that 27 students are taking history but not statistics. -/
theorem students_history_not_statistics 
  (total : ℕ) (history : ℕ) (statistics : ℕ) (history_or_statistics : ℕ)
  (h_total : total = 89)
  (h_history : history = 36)
  (h_statistics : statistics = 32)
  (h_history_or_statistics : history_or_statistics = 59) :
  history - (history + statistics - history_or_statistics) = 27 := by
  sorry

end students_history_not_statistics_l342_34227


namespace solution_implies_k_value_l342_34246

theorem solution_implies_k_value (x y k : ℝ) :
  x = -3 → y = 2 → 2 * x + k * y = 0 → k = 3 := by sorry

end solution_implies_k_value_l342_34246


namespace sum_110_terms_l342_34286

-- Define an arithmetic sequence type
def ArithmeticSequence := ℕ → ℤ

-- Define the sum of the first n terms of an arithmetic sequence
def sum_n_terms (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  (List.range n).map seq |>.sum

-- Define the properties of our specific arithmetic sequence
def special_arithmetic_sequence (seq : ArithmeticSequence) : Prop :=
  sum_n_terms seq 10 = 100 ∧ sum_n_terms seq 100 = 10

-- State the theorem
theorem sum_110_terms (seq : ArithmeticSequence) 
  (h : special_arithmetic_sequence seq) : 
  sum_n_terms seq 110 = -110 := by
  sorry

end sum_110_terms_l342_34286


namespace unique_solution_square_equation_l342_34228

theorem unique_solution_square_equation :
  ∃! x : ℝ, (10 - x)^2 = x^2 ∧ x = 5 := by sorry

end unique_solution_square_equation_l342_34228


namespace y_coordinate_range_l342_34271

-- Define the circle C
def CircleC (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 2

-- Define the condition MA^2 + MO^2 ≤ 10
def Condition (x y : ℝ) : Prop := (x - 2)^2 + y^2 + x^2 + y^2 ≤ 10

-- Theorem statement
theorem y_coordinate_range :
  ∀ x y : ℝ, CircleC x y → Condition x y →
  -Real.sqrt 7 / 2 ≤ y ∧ y ≤ Real.sqrt 7 / 2 :=
by sorry

end y_coordinate_range_l342_34271


namespace negation_of_forall_gt_sin_l342_34212

theorem negation_of_forall_gt_sin (P : ℝ → Prop) : 
  (¬ ∀ x > 0, 2 * x > Real.sin x) ↔ (∃ x₀ > 0, 2 * x₀ ≤ Real.sin x₀) := by
  sorry

end negation_of_forall_gt_sin_l342_34212


namespace expand_and_simplify_l342_34237

theorem expand_and_simplify (y : ℝ) : 5 * (6 * y^2 - 3 * y + 2) = 30 * y^2 - 15 * y + 10 := by
  sorry

end expand_and_simplify_l342_34237


namespace parabola_rotation_l342_34295

/-- Represents a parabola in the form y = a(x-h)^2 + k --/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Rotates a parabola by 180° around its vertex --/
def rotate180 (p : Parabola) : Parabola :=
  { a := -p.a, h := p.h, k := p.k }

theorem parabola_rotation (p : Parabola) (hp : p = { a := 2, h := 3, k := -2 }) :
  rotate180 p = { a := -2, h := 3, k := -2 } := by
  sorry

#check parabola_rotation

end parabola_rotation_l342_34295


namespace rabbit_speed_l342_34279

/-- Proves that given a dog running at 24 miles per hour chasing a rabbit with a 0.6-mile head start,
    if it takes the dog 4 minutes to catch up to the rabbit, then the rabbit's speed is 15 miles per hour. -/
theorem rabbit_speed (dog_speed : ℝ) (head_start : ℝ) (catch_up_time : ℝ) :
  dog_speed = 24 →
  head_start = 0.6 →
  catch_up_time = 4 / 60 →
  ∃ (rabbit_speed : ℝ),
    rabbit_speed * catch_up_time = dog_speed * catch_up_time - head_start ∧
    rabbit_speed = 15 := by
  sorry

end rabbit_speed_l342_34279


namespace unique_integer_divisible_by_18_with_cube_root_between_8_and_8_1_l342_34224

theorem unique_integer_divisible_by_18_with_cube_root_between_8_and_8_1 :
  ∃! n : ℕ+, 18 ∣ n ∧ (8 : ℝ) < (n : ℝ)^(1/3) ∧ (n : ℝ)^(1/3) < (8.1 : ℝ) ∧ n = 522 := by
  sorry

end unique_integer_divisible_by_18_with_cube_root_between_8_and_8_1_l342_34224


namespace sequence_property_l342_34202

theorem sequence_property (a : ℕ → ℤ) (S : ℕ → ℤ) : 
  (∀ n, S n = n * (n - 40)) →
  (∀ n, a n = S n - S (n - 1)) →
  a 19 < 0 ∧ a 21 > 0 := by
  sorry

end sequence_property_l342_34202


namespace beta_interval_l342_34242

theorem beta_interval (β : ℝ) : 
  (∃ k : ℤ, β = π/6 + 2*k*π) ∧ -2*π < β ∧ β < 2*π ↔ β = π/6 ∨ β = -11*π/6 := by
  sorry

end beta_interval_l342_34242


namespace whole_number_between_l342_34239

theorem whole_number_between : 
  ∀ N : ℕ, (6 < (N : ℚ) / 4 ∧ (N : ℚ) / 4 < 7) → (N = 25 ∨ N = 26 ∨ N = 27) := by
  sorry

end whole_number_between_l342_34239


namespace difference_of_three_times_number_and_five_l342_34288

theorem difference_of_three_times_number_and_five (x : ℝ) : 3 * x - 5 = 15 → 3 * x - 5 = 15 := by
  sorry

end difference_of_three_times_number_and_five_l342_34288


namespace gp_common_ratio_l342_34281

theorem gp_common_ratio (a : ℝ) (r : ℝ) (h : r ≠ 1) :
  (a * (1 - r^6) / (1 - r)) / (a * (1 - r^3) / (1 - r)) = 28 →
  r = 3 := by
sorry

end gp_common_ratio_l342_34281


namespace muffins_per_box_l342_34290

theorem muffins_per_box (total_muffins : ℕ) (available_boxes : ℕ) (additional_boxes : ℕ) :
  total_muffins = 95 →
  available_boxes = 10 →
  additional_boxes = 9 →
  (total_muffins / (available_boxes + additional_boxes) : ℚ) = 5 := by
sorry

end muffins_per_box_l342_34290


namespace x27x_divisible_by_36_l342_34211

def is_single_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def four_digit_number (x : ℕ) : ℕ := x * 1000 + 270 + x

theorem x27x_divisible_by_36 : 
  ∃! x : ℕ, is_single_digit x ∧ (four_digit_number x) % 36 = 0 :=
sorry

end x27x_divisible_by_36_l342_34211


namespace engine_system_theorems_l342_34282

/-- Engine connecting rod and crank system -/
structure EngineSystem where
  a : ℝ  -- length of crank OA
  b : ℝ  -- length of connecting rod AP
  α : ℝ  -- angle AOP
  β : ℝ  -- angle APO
  h : 0 < a ∧ 0 < b  -- positive lengths

/-- Theorems about the engine connecting rod and crank system -/
theorem engine_system_theorems (sys : EngineSystem) :
  -- Part 1
  sys.a * Real.sin sys.α = sys.b * Real.sin sys.β ∧
  -- Part 2
  (∀ β', Real.sin β' ≤ sys.a / sys.b) ∧
  -- Part 3
  ∀ x, x = sys.a * (1 - Real.cos sys.α) + sys.b * (1 - Real.cos sys.β) := by
  sorry

end engine_system_theorems_l342_34282


namespace incorrect_conclusion_l342_34293

theorem incorrect_conclusion (a b c : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : b > c) (h4 : c > 0) :
  ¬((a / b) > (a / c)) :=
sorry

end incorrect_conclusion_l342_34293


namespace line_through_points_l342_34260

/-- Given a line with slope 3 passing through points (3, 4) and (x, 7), prove that x = 4 -/
theorem line_through_points (x : ℝ) : 
  (7 - 4) / (x - 3) = 3 → x = 4 := by
  sorry

end line_through_points_l342_34260


namespace circle_symmetry_l342_34251

-- Define the original circle
def original_circle (x y : ℝ) : Prop :=
  (x - 1/2)^2 + (y + 1)^2 = 5/4

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop :=
  x - y + 1 = 0

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop :=
  (x + 2)^2 + (y - 3/2)^2 = 5/4

-- Theorem statement
theorem circle_symmetry :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    original_circle x₁ y₁ →
    symmetric_circle x₂ y₂ →
    ∃ (x y : ℝ),
      symmetry_line x y ∧
      (x₁ + x₂ = 2*x) ∧
      (y₁ + y₂ = 2*y) :=
by sorry

end circle_symmetry_l342_34251


namespace box_weight_difference_l342_34240

theorem box_weight_difference (first_box_weight third_box_weight : ℕ) 
  (h1 : first_box_weight = 2)
  (h2 : third_box_weight = 13) : 
  third_box_weight - first_box_weight = 11 := by
  sorry

end box_weight_difference_l342_34240


namespace percentage_of_number_l342_34244

theorem percentage_of_number (x : ℝ) (h : (1/4) * (1/3) * (2/5) * x = 16) : 
  (40/100) * x = 192 := by sorry

end percentage_of_number_l342_34244


namespace distance_between_trees_l342_34297

/-- Given a yard of length 500 metres with 105 trees planted at equal distances,
    including one at each end, prove that the distance between two consecutive
    trees is 500/104 metres. -/
theorem distance_between_trees (yard_length : ℝ) (num_trees : ℕ) 
    (h1 : yard_length = 500)
    (h2 : num_trees = 105) :
  let num_segments := num_trees - 1
  yard_length / num_segments = 500 / 104 := by
  sorry

end distance_between_trees_l342_34297


namespace choose_four_from_ten_l342_34209

theorem choose_four_from_ten : Nat.choose 10 4 = 210 := by
  sorry

end choose_four_from_ten_l342_34209


namespace range_of_m_l342_34275

/-- The function f(x) as defined in the problem -/
def f (a x : ℝ) : ℝ := 2 * x^3 - 3 * (a + 1) * x^2 + 6 * a * x

/-- The theorem statement -/
theorem range_of_m (m : ℝ) :
  (∀ a ∈ Set.Icc (-3 : ℝ) 0,
    ∀ x₁ ∈ Set.Icc 0 2,
    ∀ x₂ ∈ Set.Icc 0 2,
    m - a * m^2 ≥ |f a x₁ - f a x₂|) →
  m ∈ Set.Ici 5 := by
  sorry

end range_of_m_l342_34275


namespace worker_net_income_proof_l342_34249

/-- Calculates the net income after tax for a tax resident worker --/
def netIncomeAfterTax (grossIncome : ℝ) (taxRate : ℝ) : ℝ :=
  grossIncome * (1 - taxRate)

/-- Proves that the net income after tax for a worker credited with 45000 and a 13% tax rate is 39150 --/
theorem worker_net_income_proof :
  let grossIncome : ℝ := 45000
  let taxRate : ℝ := 0.13
  netIncomeAfterTax grossIncome taxRate = 39150 := by
sorry

#eval netIncomeAfterTax 45000 0.13

end worker_net_income_proof_l342_34249


namespace bus_speed_excluding_stoppages_l342_34294

/-- Given a bus that stops for 45 minutes per hour and has an average speed of 15 km/hr including stoppages,
    prove that its average speed excluding stoppages is 60 km/hr. -/
theorem bus_speed_excluding_stoppages (stop_time : ℝ) (avg_speed_with_stops : ℝ) 
  (h1 : stop_time = 45) 
  (h2 : avg_speed_with_stops = 15) :
  let moving_time : ℝ := 60 - stop_time
  let speed_excluding_stops : ℝ := (avg_speed_with_stops * 60) / moving_time
  speed_excluding_stops = 60 := by
sorry

end bus_speed_excluding_stoppages_l342_34294


namespace alternating_squares_sum_l342_34256

theorem alternating_squares_sum : 
  23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 288 := by
  sorry

end alternating_squares_sum_l342_34256


namespace rate_squares_sum_l342_34213

theorem rate_squares_sum : ∃ (b j s : ℕ), b + j + s = 34 ∧ b^2 + j^2 + s^2 = 406 := by
  sorry

end rate_squares_sum_l342_34213


namespace parabola_properties_l342_34289

/-- Parabola structure -/
structure Parabola where
  focus : ℝ × ℝ
  equation : ℝ → ℝ → Prop

/-- Line structure -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Two points on a parabola -/
structure ParabolaPoints where
  A : ℝ × ℝ
  B : ℝ × ℝ

/-- Theorem about properties of points on a parabola -/
theorem parabola_properties
  (E : Parabola)
  (pts : ParabolaPoints)
  (symmetry_line : Line)
  (h1 : E.equation = fun x y ↦ y^2 = 4*x)
  (h2 : pts.A.1 ≠ pts.B.1 ∨ pts.A.2 ≠ pts.B.2)
  (h3 : E.equation pts.A.1 pts.A.2 ∧ E.equation pts.B.1 pts.B.2)
  (h4 : symmetry_line.slope = k)
  (h5 : symmetry_line.intercept = 4)
  (h6 : ∃ x₀, pts.A.2 - pts.B.2 = -k * (pts.A.1 - pts.B.1) ∧ 
                pts.A.2 / (pts.A.1 - x₀) = pts.B.2 / (pts.B.1 - x₀)) :
  E.focus = (1, 0) ∧ 
  pts.A.1 + pts.B.1 = 4 ∧ 
  ∃ x₀ : ℝ, -2 < x₀ ∧ x₀ < 2 := by
  sorry

end parabola_properties_l342_34289
