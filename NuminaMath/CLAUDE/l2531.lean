import Mathlib

namespace NUMINAMATH_CALUDE_power_problem_l2531_253192

theorem power_problem (a m n : ℕ) (h1 : a ^ m = 3) (h2 : a ^ n = 2) : 
  a ^ (2 * m + 3 * n) = 72 := by
sorry

end NUMINAMATH_CALUDE_power_problem_l2531_253192


namespace NUMINAMATH_CALUDE_right_triangle_in_segment_sets_l2531_253163

/-- Check if three line segments can form a right-angled triangle -/
def is_right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

/-- The given sets of line segments -/
def segment_sets : List (ℝ × ℝ × ℝ) :=
  [(1, 2, 4), (3, 4, 5), (4, 6, 8), (5, 7, 11)]

theorem right_triangle_in_segment_sets :
  ∃! (a b c : ℝ), (a, b, c) ∈ segment_sets ∧ is_right_triangle a b c :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_in_segment_sets_l2531_253163


namespace NUMINAMATH_CALUDE_projectile_max_height_l2531_253183

/-- The height function of the projectile -/
def h (t : ℝ) : ℝ := -9 * t^2 + 36 * t + 24

/-- Theorem stating that the maximum height of the projectile is 60 meters -/
theorem projectile_max_height : 
  ∃ (t : ℝ), ∀ (s : ℝ), h s ≤ h t ∧ h t = 60 := by
  sorry

end NUMINAMATH_CALUDE_projectile_max_height_l2531_253183


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2531_253111

theorem rationalize_denominator :
  (Real.sqrt 12 + Real.sqrt 5) / (Real.sqrt 3 + Real.sqrt 5) = (Real.sqrt 15 - 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2531_253111


namespace NUMINAMATH_CALUDE_amusement_park_average_cost_l2531_253125

theorem amusement_park_average_cost
  (num_people : ℕ)
  (transportation_fee : ℚ)
  (admission_fee : ℚ)
  (h_num_people : num_people = 5)
  (h_transportation_fee : transportation_fee = 9.5)
  (h_admission_fee : admission_fee = 32.5) :
  (transportation_fee + admission_fee) / num_people = 8.2 :=
by sorry

end NUMINAMATH_CALUDE_amusement_park_average_cost_l2531_253125


namespace NUMINAMATH_CALUDE_arithmetic_progression_polynomial_j_value_l2531_253168

/-- A polynomial of degree 4 with four distinct real zeros in arithmetic progression -/
structure ArithmeticProgressionPolynomial where
  j : ℝ
  k : ℝ
  zeros : Fin 4 → ℝ
  distinct : ∀ i j, i ≠ j → zeros i ≠ zeros j
  arithmetic_progression : ∃ (b d : ℝ), ∀ i, zeros i = b + d * i.val
  is_zero : ∀ x, x^4 + j * x^2 + k * x + 256 = (x - zeros 0) * (x - zeros 1) * (x - zeros 2) * (x - zeros 3)

/-- The value of j in an ArithmeticProgressionPolynomial is -40 -/
theorem arithmetic_progression_polynomial_j_value (p : ArithmeticProgressionPolynomial) : p.j = -40 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_polynomial_j_value_l2531_253168


namespace NUMINAMATH_CALUDE_cement_bags_calculation_l2531_253149

theorem cement_bags_calculation (cement_cost : ℕ) (sand_lorries : ℕ) (sand_tons_per_lorry : ℕ)
  (sand_cost_per_ton : ℕ) (total_payment : ℕ) :
  cement_cost = 10 →
  sand_lorries = 20 →
  sand_tons_per_lorry = 10 →
  sand_cost_per_ton = 40 →
  total_payment = 13000 →
  (total_payment - sand_lorries * sand_tons_per_lorry * sand_cost_per_ton) / cement_cost = 500 := by
  sorry

end NUMINAMATH_CALUDE_cement_bags_calculation_l2531_253149


namespace NUMINAMATH_CALUDE_vowel_word_count_l2531_253191

/-- The number of vowels available (including Y) -/
def num_vowels : ℕ := 6

/-- The number of times each vowel appears, except A -/
def vowel_count : ℕ := 5

/-- The number of times A appears -/
def a_count : ℕ := 3

/-- The length of each word -/
def word_length : ℕ := 5

/-- The number of five-letter words that can be formed using vowels A, E, I, O, U, Y,
    where each vowel appears 5 times except A which appears 3 times -/
def num_words : ℕ := 7750

theorem vowel_word_count : 
  (vowel_count ^ word_length) + 
  (word_length.choose 1 * vowel_count ^ (word_length - 1)) +
  (word_length.choose 2 * vowel_count ^ (word_length - 2)) +
  (word_length.choose 3 * vowel_count ^ (word_length - 3)) = num_words :=
sorry

end NUMINAMATH_CALUDE_vowel_word_count_l2531_253191


namespace NUMINAMATH_CALUDE_rotated_square_height_l2531_253161

theorem rotated_square_height :
  let square_side : ℝ := 1
  let rotation_angle : ℝ := 60 * (π / 180)  -- 60 degrees in radians
  let diagonal : ℝ := square_side * Real.sqrt 2
  let height_above_center : ℝ := (diagonal / 2) * Real.sin rotation_angle
  let original_center_height : ℝ := square_side / 2
  let total_height : ℝ := original_center_height + height_above_center
  total_height = (2 + Real.sqrt 6) / 4 := by sorry

end NUMINAMATH_CALUDE_rotated_square_height_l2531_253161


namespace NUMINAMATH_CALUDE_percentage_difference_l2531_253142

theorem percentage_difference (x y : ℝ) (h1 : y = 125 * (1 + 0.1)) (h2 : x = 123.75) :
  (y - x) / y * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l2531_253142


namespace NUMINAMATH_CALUDE_cos_75_degrees_l2531_253199

theorem cos_75_degrees :
  Real.cos (75 * π / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_75_degrees_l2531_253199


namespace NUMINAMATH_CALUDE_cylinder_lateral_area_l2531_253126

/-- The lateral area of a cylinder with volume π and base radius 1 is 2π -/
theorem cylinder_lateral_area (V : ℝ) (r : ℝ) (h : ℝ) : 
  V = π → r = 1 → V = π * r^2 * h → 2 * π * r * h = 2 * π := by
  sorry

end NUMINAMATH_CALUDE_cylinder_lateral_area_l2531_253126


namespace NUMINAMATH_CALUDE_prop_p_and_q_false_l2531_253103

theorem prop_p_and_q_false : 
  (¬(∀ a b : ℝ, a > b → a^2 > b^2)) ∧ 
  (¬(∃ x : ℝ, x^2 + 2 > 3*x)) := by
  sorry

end NUMINAMATH_CALUDE_prop_p_and_q_false_l2531_253103


namespace NUMINAMATH_CALUDE_missing_fraction_sum_l2531_253157

theorem missing_fraction_sum (x : ℚ) :
  1/3 + 1/2 + 1/5 + 1/4 + (-9/20) + (-9/20) + x = 9/20 →
  x = 1/15 := by
sorry

end NUMINAMATH_CALUDE_missing_fraction_sum_l2531_253157


namespace NUMINAMATH_CALUDE_square_sum_equals_three_times_product_l2531_253144

theorem square_sum_equals_three_times_product
  (x y : ℝ) 
  (h1 : 1/x + 1/y = 5) 
  (h2 : x + y = 5*x*y) : 
  x^2 + y^2 = 3*x*y :=
by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_three_times_product_l2531_253144


namespace NUMINAMATH_CALUDE_decimal_to_binary_51_l2531_253175

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
    let rec toBinaryAux (m : ℕ) : List Bool :=
      if m = 0 then [] else (m % 2 = 1) :: toBinaryAux (m / 2)
    toBinaryAux n

/-- The decimal number to be converted -/
def decimalNumber : ℕ := 51

/-- The expected binary representation -/
def expectedBinary : List Bool := [true, true, false, false, true, true]

/-- Theorem stating that the binary representation of 51 is [1,1,0,0,1,1] -/
theorem decimal_to_binary_51 : toBinary decimalNumber = expectedBinary := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_binary_51_l2531_253175


namespace NUMINAMATH_CALUDE_smallest_three_digit_mod_congruence_l2531_253116

theorem smallest_three_digit_mod_congruence :
  ∃ n : ℕ, 
    n ≥ 100 ∧ 
    n < 1000 ∧ 
    45 * n % 315 = 90 ∧ 
    ∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ 45 * m % 315 = 90 → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_mod_congruence_l2531_253116


namespace NUMINAMATH_CALUDE_puppies_sold_calculation_l2531_253181

-- Define the given conditions
def initial_puppies : ℕ := 18
def puppies_per_cage : ℕ := 5
def cages_used : ℕ := 3

-- Define the theorem
theorem puppies_sold_calculation :
  initial_puppies - (cages_used * puppies_per_cage) = 3 := by
  sorry

end NUMINAMATH_CALUDE_puppies_sold_calculation_l2531_253181


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l2531_253120

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5}

-- Define set M
def M : Finset Nat := {1, 4}

-- Define set N
def N : Finset Nat := {1, 3, 5}

-- Theorem statement
theorem intersection_complement_equality :
  N ∩ (U \ M) = {3, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l2531_253120


namespace NUMINAMATH_CALUDE_andy_location_after_10_turns_l2531_253122

/-- Represents a direction on the coordinate plane -/
inductive Direction
  | North
  | East
  | South
  | West

/-- Represents Andy's position and facing direction -/
structure State where
  x : Int
  y : Int
  dir : Direction
  moveCount : Nat

/-- Turns the current direction 90 degrees right -/
def turnRight (d : Direction) : Direction :=
  match d with
  | Direction.North => Direction.East
  | Direction.East => Direction.South
  | Direction.South => Direction.West
  | Direction.West => Direction.North

/-- Moves Andy according to his current state -/
def move (s : State) : State :=
  let newMoveCount := s.moveCount + 1
  match s.dir with
  | Direction.North => { s with y := s.y + newMoveCount, dir := turnRight s.dir, moveCount := newMoveCount }
  | Direction.East => { s with x := s.x + newMoveCount, dir := turnRight s.dir, moveCount := newMoveCount }
  | Direction.South => { s with y := s.y - newMoveCount, dir := turnRight s.dir, moveCount := newMoveCount }
  | Direction.West => { s with x := s.x - newMoveCount, dir := turnRight s.dir, moveCount := newMoveCount }

/-- Applies the move function n times to the initial state -/
def applyMoves (n : Nat) : State :=
  match n with
  | 0 => { x := 0, y := 0, dir := Direction.North, moveCount := 0 }
  | n + 1 => move (applyMoves n)

theorem andy_location_after_10_turns :
  let finalState := applyMoves 10
  finalState.x = 6 ∧ finalState.y = 5 :=
sorry

end NUMINAMATH_CALUDE_andy_location_after_10_turns_l2531_253122


namespace NUMINAMATH_CALUDE_paintbrush_cost_l2531_253145

/-- The cost of each paintbrush given Marc's purchases -/
theorem paintbrush_cost (model_cars : ℕ) (car_cost : ℕ) (paint_bottles : ℕ) (paint_cost : ℕ) 
  (paintbrushes : ℕ) (total_spent : ℕ) : 
  model_cars = 5 → 
  car_cost = 20 → 
  paint_bottles = 5 → 
  paint_cost = 10 → 
  paintbrushes = 5 → 
  total_spent = 160 → 
  (total_spent - (model_cars * car_cost + paint_bottles * paint_cost)) / paintbrushes = 2 := by
  sorry

#check paintbrush_cost

end NUMINAMATH_CALUDE_paintbrush_cost_l2531_253145


namespace NUMINAMATH_CALUDE_committee_selection_l2531_253180

theorem committee_selection (n m : ℕ) (h1 : n = 15) (h2 : m = 5) : 
  Nat.choose n m = 3003 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_l2531_253180


namespace NUMINAMATH_CALUDE_subtracted_number_l2531_253108

theorem subtracted_number (x : ℕ) : 10000 - x = 9001 → x = 999 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_number_l2531_253108


namespace NUMINAMATH_CALUDE_replaced_tomatoes_cost_is_2_20_l2531_253110

/-- Represents the grocery order with item prices and total costs -/
structure GroceryOrder where
  original_total : ℝ
  original_tomatoes : ℝ
  original_lettuce : ℝ
  original_celery : ℝ
  new_lettuce : ℝ
  new_celery : ℝ
  delivery_tip : ℝ
  new_total : ℝ

/-- Calculates the cost of the replaced can of tomatoes -/
def replaced_tomatoes_cost (order : GroceryOrder) : ℝ :=
  order.new_total - order.original_total - order.delivery_tip -
  (order.new_lettuce - order.original_lettuce) -
  (order.new_celery - order.original_celery) +
  order.original_tomatoes

/-- Theorem stating that the cost of the replaced can of tomatoes is $2.20 -/
theorem replaced_tomatoes_cost_is_2_20 (order : GroceryOrder)
  (h1 : order.original_total = 25)
  (h2 : order.original_tomatoes = 0.99)
  (h3 : order.original_lettuce = 1)
  (h4 : order.original_celery = 1.96)
  (h5 : order.new_lettuce = 1.75)
  (h6 : order.new_celery = 2)
  (h7 : order.delivery_tip = 8)
  (h8 : order.new_total = 35) :
  replaced_tomatoes_cost order = 2.20 := by
  sorry


end NUMINAMATH_CALUDE_replaced_tomatoes_cost_is_2_20_l2531_253110


namespace NUMINAMATH_CALUDE_seating_probability_l2531_253174

/-- Represents the number of delegates -/
def num_delegates : ℕ := 9

/-- Represents the number of countries -/
def num_countries : ℕ := 3

/-- Represents the number of delegates per country -/
def delegates_per_country : ℕ := 3

/-- Calculates the total number of seating arrangements -/
def total_arrangements : ℕ := (num_delegates.factorial) / ((delegates_per_country.factorial) ^ num_countries)

/-- Calculates the number of unwanted arrangements (where at least one country's delegates sit together) -/
def unwanted_arrangements : ℕ := 
  num_countries * num_delegates * ((num_delegates - delegates_per_country).factorial / ((delegates_per_country.factorial) ^ (num_countries - 1))) -
  (num_countries.choose 2) * num_delegates * (num_delegates - 2 * delegates_per_country + 1) +
  num_delegates * 2

/-- The probability that each delegate sits next to at least one delegate from another country -/
def probability : ℚ := (total_arrangements - unwanted_arrangements : ℚ) / total_arrangements

theorem seating_probability : probability = 41 / 56 := by
  sorry

end NUMINAMATH_CALUDE_seating_probability_l2531_253174


namespace NUMINAMATH_CALUDE_intersection_implies_determinant_one_l2531_253129

/-- Given three lines that intersect at one point, prove that the determinant is 1 -/
theorem intersection_implies_determinant_one 
  (a : ℝ) 
  (h1 : ∃ (x y : ℝ), ax + y + 3 = 0 ∧ x + y + 2 = 0 ∧ 2*x - y + 1 = 0) :
  Matrix.det ![![a, 1], ![1, 1]] = 1 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_determinant_one_l2531_253129


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l2531_253165

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 138) 
  (h2 : a*b + b*c + c*a = 131) : 
  a + b + c = 20 := by sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l2531_253165


namespace NUMINAMATH_CALUDE_max_measurements_exact_measurements_l2531_253131

/-- The number of ways to measure a weight P using weights up to 2^n -/
def K (n : ℕ) (P : ℕ) : ℕ := sorry

/-- The maximum number of ways any weight can be measured using weights up to 2^n -/
def K_max (n : ℕ) : ℕ := sorry

/-- The set of available weights -/
def weights : Set ℕ := {w : ℕ | ∃ k : ℕ, w = 2^k ∧ k ≤ 9}

theorem max_measurements (P : ℕ) : K 9 P ≤ 89 := sorry

theorem exact_measurements : K 9 171 = 89 := sorry

#check max_measurements
#check exact_measurements

end NUMINAMATH_CALUDE_max_measurements_exact_measurements_l2531_253131


namespace NUMINAMATH_CALUDE_gcd_204_85_l2531_253155

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := by
  sorry

end NUMINAMATH_CALUDE_gcd_204_85_l2531_253155


namespace NUMINAMATH_CALUDE_ten_men_and_boys_complete_in_ten_days_l2531_253136

/-- The number of days it takes for a group of men and boys to complete a work -/
def daysToComplete (numMen numBoys : ℕ) : ℚ :=
  10 / ((2 * numMen : ℚ) / 3 + (numBoys : ℚ) / 3)

/-- Theorem stating that 10 men and 10 boys will complete the work in 10 days -/
theorem ten_men_and_boys_complete_in_ten_days :
  daysToComplete 10 10 = 10 := by sorry

end NUMINAMATH_CALUDE_ten_men_and_boys_complete_in_ten_days_l2531_253136


namespace NUMINAMATH_CALUDE_germination_probability_l2531_253162

/-- The probability of exactly k successes in n independent Bernoulli trials -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The germination rate of seeds -/
def germination_rate : ℝ := 0.9

/-- The number of seeds sown -/
def total_seeds : ℕ := 7

/-- The number of seeds expected to germinate -/
def germinated_seeds : ℕ := 5

theorem germination_probability :
  binomial_probability total_seeds germinated_seeds germination_rate =
  21 * (germination_rate^5) * ((1 - germination_rate)^2) :=
by sorry

end NUMINAMATH_CALUDE_germination_probability_l2531_253162


namespace NUMINAMATH_CALUDE_two_pants_three_tops_six_looks_l2531_253166

/-- The number of possible looks given a number of pants and tops -/
def number_of_looks (pants : ℕ) (tops : ℕ) : ℕ := pants * tops

/-- Theorem stating that 2 pairs of pants and 3 pairs of tops result in 6 looks -/
theorem two_pants_three_tops_six_looks : 
  number_of_looks 2 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_two_pants_three_tops_six_looks_l2531_253166


namespace NUMINAMATH_CALUDE_cd_cost_fraction_l2531_253123

theorem cd_cost_fraction (m : ℝ) (n : ℕ) (h : n > 0) : 
  let total_cd_cost : ℝ := 2 * (1/3 * m)
  let cd_cost : ℝ := total_cd_cost / n
  let savings : ℝ := m - total_cd_cost
  (1/3 * m = (1/2 * n) * (cd_cost)) ∧ 
  (savings ≥ 1/4 * m) →
  cd_cost = 1/3 * m := by
sorry

end NUMINAMATH_CALUDE_cd_cost_fraction_l2531_253123


namespace NUMINAMATH_CALUDE_circle_radius_is_two_l2531_253130

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 - 4*y + 16 = 0

-- Define the center of the circle
def circle_center : ℝ × ℝ := (4, 2)

-- Define the radius of the circle
def circle_radius : ℝ := 2

-- Theorem statement
theorem circle_radius_is_two :
  ∀ x y : ℝ, circle_equation x y →
  (x - circle_center.1)^2 + (y - circle_center.2)^2 = circle_radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_is_two_l2531_253130


namespace NUMINAMATH_CALUDE_first_discount_percentage_l2531_253153

/-- Given an initial price and a final price after two discounts, 
    where the second discount is known, calculate the first discount percentage. -/
theorem first_discount_percentage 
  (initial_price : ℝ) 
  (final_price : ℝ) 
  (second_discount : ℝ) 
  (h1 : initial_price = 528)
  (h2 : final_price = 380.16)
  (h3 : second_discount = 0.1)
  : ∃ (first_discount : ℝ),
    first_discount = 0.2 ∧ 
    final_price = initial_price * (1 - first_discount) * (1 - second_discount) :=
by
  sorry

end NUMINAMATH_CALUDE_first_discount_percentage_l2531_253153


namespace NUMINAMATH_CALUDE_find_divisor_l2531_253104

theorem find_divisor (n s : ℕ) (hn : n = 5264) (hs : s = 11) :
  let d := n - s
  (d ∣ d) ∧ (∀ m : ℕ, m < s → ¬(d ∣ (n - m))) → d = 5253 :=
by sorry

end NUMINAMATH_CALUDE_find_divisor_l2531_253104


namespace NUMINAMATH_CALUDE_line_inclination_gt_45_deg_l2531_253140

/-- The angle of inclination of a line ax + (a + 1)y + 2 = 0 is greater than 45° if and only if a < -1/2 or a > 0 -/
theorem line_inclination_gt_45_deg (a : ℝ) :
  let line := {(x, y) : ℝ × ℝ | a * x + (a + 1) * y + 2 = 0}
  let angle_of_inclination := Real.arctan (abs (a / (a + 1)))
  angle_of_inclination > Real.pi / 4 ↔ a < -1/2 ∨ a > 0 :=
sorry

end NUMINAMATH_CALUDE_line_inclination_gt_45_deg_l2531_253140


namespace NUMINAMATH_CALUDE_black_area_after_three_changes_l2531_253124

def black_area_fraction (n : ℕ) : ℚ :=
  (1 / 2) ^ n

theorem black_area_after_three_changes :
  black_area_fraction 3 = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_black_area_after_three_changes_l2531_253124


namespace NUMINAMATH_CALUDE_finite_solutions_factorial_difference_l2531_253147

theorem finite_solutions_factorial_difference (u : ℕ+) :
  ∃ (S : Finset (ℕ × ℕ × ℕ)), ∀ (n a b : ℕ), 
    n! = u^a - u^b → (n, a, b) ∈ S :=
sorry

end NUMINAMATH_CALUDE_finite_solutions_factorial_difference_l2531_253147


namespace NUMINAMATH_CALUDE_jimmy_father_emails_l2531_253154

/-- The number of emails Jimmy's father receives per day before subscribing to the news channel -/
def initial_emails_per_day : ℕ := 20

/-- The number of additional emails per day after subscribing to the news channel -/
def additional_emails_per_day : ℕ := 5

/-- The total number of days in April -/
def days_in_april : ℕ := 30

/-- The day in April when Jimmy's father subscribed to the news channel -/
def subscription_day : ℕ := days_in_april / 2

theorem jimmy_father_emails :
  (subscription_day * initial_emails_per_day) +
  ((days_in_april - subscription_day) * (initial_emails_per_day + additional_emails_per_day)) = 675 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_father_emails_l2531_253154


namespace NUMINAMATH_CALUDE_coordinate_change_l2531_253134

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the vectors a, b, c
variable (a b c : V)

-- Define that {a, b, c} is a basis
variable (h₁ : LinearIndependent ℝ ![a, b, c])
variable (h₂ : Submodule.span ℝ {a, b, c} = ⊤)

-- Define that {a+b, a-b, c} is also a basis
variable (h₃ : LinearIndependent ℝ ![a + b, a - b, c])
variable (h₄ : Submodule.span ℝ {a + b, a - b, c} = ⊤)

-- Define the vector p
variable (p : V)

-- State the theorem
theorem coordinate_change (hp : p = a - 2 • b + 3 • c) :
  p = (-1/2 : ℝ) • (a + b) + (3/2 : ℝ) • (a - b) + 3 • c := by sorry

end NUMINAMATH_CALUDE_coordinate_change_l2531_253134


namespace NUMINAMATH_CALUDE_thabo_hardcover_nonfiction_count_l2531_253127

/-- Represents the number of books Thabo owns of each type -/
structure BookCollection where
  hardcover_nonfiction : ℕ
  paperback_nonfiction : ℕ
  paperback_fiction : ℕ

/-- Thabo's book collection satisfies the given conditions -/
def is_valid_collection (bc : BookCollection) : Prop :=
  bc.hardcover_nonfiction + bc.paperback_nonfiction + bc.paperback_fiction = 180 ∧
  bc.paperback_nonfiction = bc.hardcover_nonfiction + 20 ∧
  bc.paperback_fiction = 2 * bc.paperback_nonfiction

theorem thabo_hardcover_nonfiction_count :
  ∀ bc : BookCollection, is_valid_collection bc → bc.hardcover_nonfiction = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_thabo_hardcover_nonfiction_count_l2531_253127


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_l2531_253109

theorem smallest_four_digit_multiple : ∃ n : ℕ, 
  (n ≥ 1000 ∧ n < 10000) ∧ 
  5 ∣ n ∧ 6 ∣ n ∧ 2 ∣ n ∧
  (∀ m : ℕ, (m ≥ 1000 ∧ m < 10000) → 5 ∣ m → 6 ∣ m → 2 ∣ m → m ≥ n) ∧
  n = 1020 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_l2531_253109


namespace NUMINAMATH_CALUDE_ellipse_intersection_k_values_l2531_253133

noncomputable section

def Ellipse (a b : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

def Line (k m : ℝ) := {p : ℝ × ℝ | p.2 = k * p.1 + m}

def Eccentricity (a b : ℝ) := Real.sqrt (1 - (b^2 / a^2))

def Parallelogram (A B C D : ℝ × ℝ) :=
  (B.1 - A.1 = D.1 - C.1 ∧ B.2 - A.2 = D.2 - C.2) ∧
  (C.1 - A.1 = D.1 - B.1 ∧ C.2 - A.2 = D.2 - B.2)

theorem ellipse_intersection_k_values
  (a b : ℝ)
  (h_ab : a > b ∧ b > 0)
  (h_A : (2, 0) ∈ Ellipse a b)
  (h_e : Eccentricity a b = Real.sqrt 3 / 2)
  (k : ℝ)
  (M N : ℝ × ℝ)
  (h_MN : M ∈ Ellipse a b ∧ N ∈ Ellipse a b)
  (h_MN_line : M ∈ Line k (Real.sqrt 3) ∧ N ∈ Line k (Real.sqrt 3))
  (P : ℝ × ℝ)
  (h_P : P.1 = 3)
  (h_parallelogram : Parallelogram (2, 0) P M N) :
  k = Real.sqrt 3 / 2 ∨ k = Real.sqrt 11 / 2 ∨ k = -Real.sqrt 11 / 2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_k_values_l2531_253133


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l2531_253185

theorem arithmetic_expression_equality : 3 + 5 * 2^3 - 4 / 2 + 7 * 3 = 62 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l2531_253185


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l2531_253152

/-- A geometric sequence with a_1 = 1 and a_5 = 4 has a_3 = 2 -/
theorem geometric_sequence_third_term (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) = a n * (a 2 / a 1)) →  -- geometric sequence condition
  a 1 = 1 →
  a 5 = 4 →
  a 3 = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l2531_253152


namespace NUMINAMATH_CALUDE_denominator_divisor_not_zero_l2531_253195

theorem denominator_divisor_not_zero :
  ∀ (a : ℝ), a ≠ 0 → (∃ (b : ℝ), b / a = b / a) ∧ (∃ (c d : ℝ), c / d = c / d) :=
by sorry

end NUMINAMATH_CALUDE_denominator_divisor_not_zero_l2531_253195


namespace NUMINAMATH_CALUDE_circles_diameter_sum_l2531_253193

theorem circles_diameter_sum (D d : ℝ) (h1 : D > d) (h2 : D - d = 9) (h3 : D / 2 - 5 > 0) :
  let TO := D / 2 - 5
  let OC := (D - d) / 2
  let CT := d / 2
  TO ^ 2 + OC ^ 2 = CT ^ 2 → d + D = 91 := by
sorry

end NUMINAMATH_CALUDE_circles_diameter_sum_l2531_253193


namespace NUMINAMATH_CALUDE_interest_rate_problem_l2531_253176

/-- Given a principal amount and an interest rate, if increasing the interest rate by 3%
    results in 210 more interest over 10 years, then the principal amount must be 700. -/
theorem interest_rate_problem (P R : ℝ) (h : P * (R + 3) * 10 / 100 = P * R * 10 / 100 + 210) :
  P = 700 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_problem_l2531_253176


namespace NUMINAMATH_CALUDE_tv_installment_plan_duration_l2531_253186

theorem tv_installment_plan_duration (cash_price down_payment monthly_payment cash_savings : ℕ) : 
  cash_price = 400 →
  down_payment = 120 →
  monthly_payment = 30 →
  cash_savings = 80 →
  (cash_price + cash_savings - down_payment) / monthly_payment = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_tv_installment_plan_duration_l2531_253186


namespace NUMINAMATH_CALUDE_rocket_ascent_time_l2531_253194

theorem rocket_ascent_time (n : ℕ) (a₁ d : ℝ) (h₁ : a₁ = 2) (h₂ : d = 2) :
  n * a₁ + (n * (n - 1) * d) / 2 = 240 → n = 15 :=
by sorry

end NUMINAMATH_CALUDE_rocket_ascent_time_l2531_253194


namespace NUMINAMATH_CALUDE_chord_length_l2531_253102

/-- The length of the chord formed by the intersection of a circle and a line --/
theorem chord_length (x y : ℝ) : 
  let circle := (fun (x y : ℝ) ↦ (x - 1)^2 + y^2 = 4)
  let line := (fun (x y : ℝ) ↦ x + y + 1 = 0)
  let chord_length := 
    Real.sqrt (8 - 2 * ((1 * 1 + 1 * 0 + 1) / Real.sqrt (1^2 + 1^2))^2)
  (∃ (a b : ℝ × ℝ), circle a.1 a.2 ∧ circle b.1 b.2 ∧ 
                     line a.1 a.2 ∧ line b.1 b.2 ∧ 
                     a ≠ b) →
  chord_length = 2 * Real.sqrt 2 := by
sorry


end NUMINAMATH_CALUDE_chord_length_l2531_253102


namespace NUMINAMATH_CALUDE_solution_replacement_fraction_l2531_253158

/-- Proves that the fraction of a 45% solution replaced by a 25% solution
    to achieve a 35% solution is 0.5 -/
theorem solution_replacement_fraction 
  (initial_concentration : Real)
  (replacement_concentration : Real)
  (final_concentration : Real)
  (h1 : initial_concentration = 0.45)
  (h2 : replacement_concentration = 0.25)
  (h3 : final_concentration = 0.35) :
  ∃ (x : Real), 
    initial_concentration * (1 - x) + replacement_concentration * x = final_concentration ∧ 
    x = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_solution_replacement_fraction_l2531_253158


namespace NUMINAMATH_CALUDE_function_domain_constraint_l2531_253170

theorem function_domain_constraint (f : ℝ → ℝ) (h : ∀ x, x ∈ (Set.Icc 0 1) → f x ≠ 0) :
  ∀ a : ℝ, (∀ x, x ∈ (Set.Icc 0 1) → (f (x - a) + f (x + a)) ≠ 0) ↔ a ∈ (Set.Icc (-1/2) (1/2)) :=
by sorry

end NUMINAMATH_CALUDE_function_domain_constraint_l2531_253170


namespace NUMINAMATH_CALUDE_equation_solution_l2531_253132

theorem equation_solution : 
  ∃ x : ℚ, x ≠ -4 ∧ (7 * x / (x + 4) - 4 / (x + 4) = 2 / (x + 4)) ∧ x = 6 / 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2531_253132


namespace NUMINAMATH_CALUDE_circle_center_l2531_253150

def circle_equation (x y : ℝ) : Prop :=
  4 * x^2 - 8 * x + 4 * y^2 - 24 * y - 36 = 0

def is_center (h k : ℝ) : Prop :=
  ∀ x y : ℝ, circle_equation x y → (x - h)^2 + (y - k)^2 = 1

theorem circle_center : is_center 1 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_l2531_253150


namespace NUMINAMATH_CALUDE_specific_box_volume_l2531_253156

/-- The volume of an open box created from a rectangular sheet --/
def box_volume (sheet_length sheet_width x : ℝ) : ℝ :=
  (sheet_length - 2*x) * (sheet_width - 2*x) * x

/-- Theorem: The volume of the specific box is 4x^3 - 60x^2 + 216x --/
theorem specific_box_volume :
  ∀ x : ℝ, box_volume 18 12 x = 4*x^3 - 60*x^2 + 216*x :=
by
  sorry

end NUMINAMATH_CALUDE_specific_box_volume_l2531_253156


namespace NUMINAMATH_CALUDE_tan_seventeen_pi_fourths_l2531_253137

theorem tan_seventeen_pi_fourths : Real.tan (17 * π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_seventeen_pi_fourths_l2531_253137


namespace NUMINAMATH_CALUDE_apple_difference_l2531_253151

/-- The number of apples Adam has -/
def adam_apples : ℕ := 10

/-- The number of apples Jackie has -/
def jackie_apples : ℕ := 2

/-- Theorem stating the difference in apples between Adam and Jackie -/
theorem apple_difference : adam_apples - jackie_apples = 8 := by
  sorry

end NUMINAMATH_CALUDE_apple_difference_l2531_253151


namespace NUMINAMATH_CALUDE_special_rectangle_difference_l2531_253105

/-- A rectangle with perimeter 4r and diagonal k times the length of one side -/
structure SpecialRectangle (r k : ℝ) where
  length : ℝ
  width : ℝ
  perimeter_eq : length + width = 2 * r
  diagonal_eq : length ^ 2 + width ^ 2 = (k * length) ^ 2

/-- The absolute difference between length and width is k times the length -/
theorem special_rectangle_difference (r k : ℝ) (rect : SpecialRectangle r k) :
  |rect.length - rect.width| = k * rect.length :=
sorry

end NUMINAMATH_CALUDE_special_rectangle_difference_l2531_253105


namespace NUMINAMATH_CALUDE_sqrt_eight_and_three_ninths_simplification_l2531_253164

theorem sqrt_eight_and_three_ninths_simplification :
  Real.sqrt (8 + 3 / 9) = 5 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_and_three_ninths_simplification_l2531_253164


namespace NUMINAMATH_CALUDE_second_term_of_geometric_series_l2531_253135

/-- Given an infinite geometric series with common ratio 1/4 and sum 16,
    the second term of the sequence is 3. -/
theorem second_term_of_geometric_series (a : ℝ) :
  (∑' n, a * (1/4)^n : ℝ) = 16 →
  a * (1/4) = 3 := by sorry

end NUMINAMATH_CALUDE_second_term_of_geometric_series_l2531_253135


namespace NUMINAMATH_CALUDE_gcd_1234_2047_l2531_253107

theorem gcd_1234_2047 : Nat.gcd 1234 2047 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1234_2047_l2531_253107


namespace NUMINAMATH_CALUDE_division_reduction_l2531_253160

theorem division_reduction (x : ℝ) : 
  (63 / x = 63 - 42) → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_division_reduction_l2531_253160


namespace NUMINAMATH_CALUDE_complex_number_location_l2531_253114

theorem complex_number_location (z : ℂ) (h : z / (4 + 2*I) = I) :
  (z.re < 0) ∧ (z.im > 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_location_l2531_253114


namespace NUMINAMATH_CALUDE_total_weight_is_seven_pounds_l2531_253179

-- Define the weights of items
def brie_cheese : ℚ := 8 / 16  -- in pounds
def bread : ℚ := 1
def tomatoes : ℚ := 1
def zucchini : ℚ := 2
def chicken_breasts : ℚ := 3 / 2
def raspberries : ℚ := 8 / 16  -- in pounds
def blueberries : ℚ := 8 / 16  -- in pounds

-- Define the conversion factor
def ounces_per_pound : ℚ := 16

-- Theorem statement
theorem total_weight_is_seven_pounds :
  brie_cheese + bread + tomatoes + zucchini + chicken_breasts + raspberries + blueberries = 7 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_is_seven_pounds_l2531_253179


namespace NUMINAMATH_CALUDE_banana_arrangements_l2531_253117

def word_length : ℕ := 6
def occurrences : List ℕ := [1, 2, 3]

theorem banana_arrangements :
  (word_length.factorial) / (occurrences.prod) = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_arrangements_l2531_253117


namespace NUMINAMATH_CALUDE_jenny_distance_difference_l2531_253148

theorem jenny_distance_difference (run_distance walk_distance : ℝ) 
  (h1 : run_distance = 0.6)
  (h2 : walk_distance = 0.4) : 
  run_distance - walk_distance = 0.2 := by sorry

end NUMINAMATH_CALUDE_jenny_distance_difference_l2531_253148


namespace NUMINAMATH_CALUDE_greater_number_problem_l2531_253143

theorem greater_number_problem (x y : ℝ) 
  (sum_eq : x + y = 40)
  (diff_eq : x - y = 10) :
  max x y = 25 := by sorry

end NUMINAMATH_CALUDE_greater_number_problem_l2531_253143


namespace NUMINAMATH_CALUDE_randolph_age_l2531_253139

/-- Proves that Randolph's age is 55 given the conditions of the problem -/
theorem randolph_age :
  (∀ (sherry sydney randolph : ℕ),
    randolph = sydney + 5 →
    sydney = 2 * sherry →
    sherry = 25 →
    randolph = 55) :=
by sorry

end NUMINAMATH_CALUDE_randolph_age_l2531_253139


namespace NUMINAMATH_CALUDE_sum_of_bases_l2531_253167

-- Define the fractions F₁ and F₂
def F₁ (R : ℕ) : ℚ := (4 * R + 5) / (R^2 - 1)
def F₂ (R : ℕ) : ℚ := (5 * R + 4) / (R^2 - 1)

-- Define the conditions
def condition1 (R₁ : ℕ) : Prop := F₁ R₁ = 5 / 11
def condition2 (R₁ : ℕ) : Prop := F₂ R₁ = 6 / 11
def condition3 (R₂ : ℕ) : Prop := F₁ R₂ = 3 / 7
def condition4 (R₂ : ℕ) : Prop := F₂ R₂ = 4 / 7

-- State the theorem
theorem sum_of_bases (R₁ R₂ : ℕ) :
  condition1 R₁ → condition2 R₁ → condition3 R₂ → condition4 R₂ → R₁ + R₂ = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_bases_l2531_253167


namespace NUMINAMATH_CALUDE_f_f_has_four_distinct_roots_l2531_253106

-- Define the function f
def f (d : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + d

-- State the theorem
theorem f_f_has_four_distinct_roots :
  ∃! d : ℝ, ∃ (r₁ r₂ r₃ r₄ : ℝ), 
    (r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₁ ≠ r₄ ∧ r₂ ≠ r₃ ∧ r₂ ≠ r₄ ∧ r₃ ≠ r₄) ∧
    (∀ x : ℝ, f (f d x) = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃ ∨ x = r₄) ∧
    d = 2 :=
sorry

end NUMINAMATH_CALUDE_f_f_has_four_distinct_roots_l2531_253106


namespace NUMINAMATH_CALUDE_number_difference_l2531_253173

theorem number_difference (a b : ℕ) (h1 : a + b = 56) (h2 : a < b) (h3 : a = 22) (h4 : b = 34) :
  b - a = 12 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l2531_253173


namespace NUMINAMATH_CALUDE_complex_absolute_value_sum_l2531_253182

theorem complex_absolute_value_sum : Complex.abs (3 - 5*I) + Complex.abs (3 + 5*I) = 2 * Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_sum_l2531_253182


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2531_253178

theorem sum_of_roots_quadratic (b : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 - b*x + 20
  (∃ x y : ℝ, f x = 0 ∧ f y = 0 ∧ x * y = 20) →
  (∃ x y : ℝ, f x = 0 ∧ f y = 0 ∧ x + y = b) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2531_253178


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2531_253138

theorem sum_of_roots_quadratic (x : ℝ) : 
  (2 * x^2 - 5 * x + 3 = 9) → 
  (∃ y : ℝ, 2 * y^2 - 5 * y + 3 = 9 ∧ x + y = 5/2) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2531_253138


namespace NUMINAMATH_CALUDE_not_necessarily_similar_remaining_parts_l2531_253190

-- Define a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define similarity between triangles
def similar (t1 t2 : Triangle) : Prop := sorry

-- Define a function to split a triangle into two smaller triangles
def split (t : Triangle) : Triangle × Triangle := sorry

theorem not_necessarily_similar_remaining_parts 
  (T1 T2 : Triangle) 
  (h_similar : similar T1 T2) 
  (T1_split : Triangle × Triangle) 
  (T2_split : Triangle × Triangle)
  (h_T1_split : T1_split = split T1)
  (h_T2_split : T2_split = split T2)
  (h_part_similar : similar T1_split.1 T2_split.1) :
  ¬ (∀ (T1 T2 : Triangle) (h_similar : similar T1 T2) 
      (T1_split T2_split : Triangle × Triangle)
      (h_T1_split : T1_split = split T1)
      (h_T2_split : T2_split = split T2)
      (h_part_similar : similar T1_split.1 T2_split.1),
    similar T1_split.2 T2_split.2) :=
by sorry

end NUMINAMATH_CALUDE_not_necessarily_similar_remaining_parts_l2531_253190


namespace NUMINAMATH_CALUDE_percentage_reduction_price_increase_for_target_profit_price_increase_for_max_profit_maximum_daily_profit_l2531_253119

-- Define the original price and final price
def original_price : ℝ := 50
def final_price : ℝ := 32

-- Define the profit per kilogram and initial daily sales
def profit_per_kg : ℝ := 10
def initial_daily_sales : ℝ := 500

-- Define the reduction in sales per yuan increase
def sales_reduction_per_yuan : ℝ := 20

-- Define the target daily profit
def target_daily_profit : ℝ := 6000

-- Part 1: Percentage reduction
theorem percentage_reduction : 
  ∃ x : ℝ, x > 0 ∧ x < 1 ∧ original_price * (1 - x)^2 = final_price ∧ x = 0.2 := by sorry

-- Part 2: Price increase for target profit
theorem price_increase_for_target_profit :
  ∃ x : ℝ, x > 0 ∧ (profit_per_kg + x) * (initial_daily_sales - sales_reduction_per_yuan * x) = target_daily_profit ∧
  (∀ y : ℝ, y > 0 ∧ (profit_per_kg + y) * (initial_daily_sales - sales_reduction_per_yuan * y) = target_daily_profit → x ≤ y) ∧
  x = 5 := by sorry

-- Part 3: Price increase for maximum profit
def profit_function (x : ℝ) : ℝ := (profit_per_kg + x) * (initial_daily_sales - sales_reduction_per_yuan * x)

theorem price_increase_for_max_profit :
  ∃ x : ℝ, x > 0 ∧ (∀ y : ℝ, profit_function y ≤ profit_function x) ∧ x = 7.5 := by sorry

-- Part 4: Maximum daily profit
theorem maximum_daily_profit :
  ∃ x : ℝ, x > 0 ∧ (∀ y : ℝ, profit_function y ≤ profit_function x) ∧ profit_function x = 6125 := by sorry

end NUMINAMATH_CALUDE_percentage_reduction_price_increase_for_target_profit_price_increase_for_max_profit_maximum_daily_profit_l2531_253119


namespace NUMINAMATH_CALUDE_gnome_with_shoes_weighs_34_l2531_253128

/-- The weight of a gnome without shoes -/
def gnome_weight : ℝ := sorry

/-- The weight of a gnome with shoes -/
def gnome_with_shoes_weight : ℝ := sorry

/-- The difference in weight between a gnome with shoes and without shoes -/
def shoe_weight_difference : ℝ := 2

/-- The total weight of five gnomes with shoes and five gnomes without shoes -/
def total_weight : ℝ := 330

/-- Theorem stating that a gnome with shoes weighs 34 kg -/
theorem gnome_with_shoes_weighs_34 :
  gnome_with_shoes_weight = 34 :=
by
  sorry

/-- Axiom: A gnome with shoes weighs 2 kg more than a gnome without shoes -/
axiom shoe_weight_relation :
  gnome_with_shoes_weight = gnome_weight + shoe_weight_difference

/-- Axiom: The total weight of five gnomes with shoes and five gnomes without shoes is 330 kg -/
axiom total_weight_relation :
  5 * gnome_with_shoes_weight + 5 * gnome_weight = total_weight

end NUMINAMATH_CALUDE_gnome_with_shoes_weighs_34_l2531_253128


namespace NUMINAMATH_CALUDE_prime_sum_divisibility_l2531_253159

theorem prime_sum_divisibility (p q : ℕ) : 
  Prime p → Prime q → q = p + 2 → (p + q) ∣ (p^q + q^p) := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_divisibility_l2531_253159


namespace NUMINAMATH_CALUDE_trig_identity_l2531_253115

theorem trig_identity (θ a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (Real.sin θ)^4 / a + (Real.cos θ)^4 / b = 1 / (2 * (a + b)) →
  (Real.sin θ)^6 / a^2 + (Real.cos θ)^6 / b^2 = 1 / (a + b)^2 :=
by sorry

end NUMINAMATH_CALUDE_trig_identity_l2531_253115


namespace NUMINAMATH_CALUDE_students_interested_in_all_subjects_prove_students_interested_in_all_subjects_l2531_253177

/-- Represents the number of students interested in a combination of subjects -/
structure InterestCounts where
  total : ℕ
  biology : ℕ
  chemistry : ℕ
  physics : ℕ
  none : ℕ
  onlyBiology : ℕ
  onlyPhysics : ℕ
  biologyAndChemistry : ℕ

/-- The theorem stating the number of students interested in all three subjects -/
theorem students_interested_in_all_subjects (counts : InterestCounts) : ℕ :=
  let all_three := counts.biology + counts.chemistry + counts.physics -
    (counts.onlyBiology + counts.biologyAndChemistry + counts.onlyPhysics) - 
    (counts.total - counts.none)
  2

/-- The main theorem proving the number of students interested in all subjects -/
theorem prove_students_interested_in_all_subjects : 
  ∃ (counts : InterestCounts), 
    counts.total = 40 ∧ 
    counts.biology = 20 ∧ 
    counts.chemistry = 10 ∧ 
    counts.physics = 8 ∧ 
    counts.none = 7 ∧ 
    counts.onlyBiology = 12 ∧ 
    counts.onlyPhysics = 4 ∧ 
    counts.biologyAndChemistry = 6 ∧ 
    students_interested_in_all_subjects counts = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_students_interested_in_all_subjects_prove_students_interested_in_all_subjects_l2531_253177


namespace NUMINAMATH_CALUDE_expression_value_at_three_l2531_253121

theorem expression_value_at_three : 
  let x : ℝ := 3
  x^6 - x^3 - 6*x = 684 := by sorry

end NUMINAMATH_CALUDE_expression_value_at_three_l2531_253121


namespace NUMINAMATH_CALUDE_average_weight_b_c_l2531_253146

/-- Given the weights of three people a, b, and c, prove that the average weight of b and c is 42 kg -/
theorem average_weight_b_c (a b c : ℝ) : 
  (a + b + c) / 3 = 43 →  -- The average weight of a, b, and c is 43 kg
  (a + b) / 2 = 48 →      -- The average weight of a and b is 48 kg
  b = 51 →                -- The weight of b is 51 kg
  (b + c) / 2 = 42 :=     -- The average weight of b and c is 42 kg
by
  sorry

end NUMINAMATH_CALUDE_average_weight_b_c_l2531_253146


namespace NUMINAMATH_CALUDE_stable_performance_lower_variance_athlete_a_more_stable_l2531_253188

-- Define the structure for an athlete's performance
structure AthletePerformance where
  average_score : ℝ
  variance : ℝ
  variance_positive : variance > 0

-- Define the notion of stability
def more_stable (a b : AthletePerformance) : Prop :=
  a.variance < b.variance

-- Theorem statement
theorem stable_performance_lower_variance 
  (a b : AthletePerformance) 
  (h_equal_avg : a.average_score = b.average_score) :
  more_stable a b ↔ a.variance < b.variance :=
sorry

-- Specific instance for the given problem
def athlete_a : AthletePerformance := {
  average_score := 9
  variance := 1.2
  variance_positive := by norm_num
}

def athlete_b : AthletePerformance := {
  average_score := 9
  variance := 2.4
  variance_positive := by norm_num
}

-- Theorem application to the specific instance
theorem athlete_a_more_stable : more_stable athlete_a athlete_b :=
sorry

end NUMINAMATH_CALUDE_stable_performance_lower_variance_athlete_a_more_stable_l2531_253188


namespace NUMINAMATH_CALUDE_smallest_block_volume_l2531_253197

theorem smallest_block_volume (l m n : ℕ) : 
  (l - 1) * (m - 1) * (n - 1) = 120 → 
  l * m * n ≥ 216 :=
by sorry

end NUMINAMATH_CALUDE_smallest_block_volume_l2531_253197


namespace NUMINAMATH_CALUDE_bake_sale_total_l2531_253141

/-- Represents the number of cookies sold at a bake sale -/
structure CookieSale where
  raisin : ℕ
  oatmeal : ℕ
  chocolate_chip : ℕ

/-- Theorem stating the total number of cookies sold given the conditions -/
theorem bake_sale_total (sale : CookieSale) : 
  sale.raisin = 42 ∧ 
  sale.raisin = 6 * sale.oatmeal ∧ 
  sale.raisin = 2 * sale.chocolate_chip → 
  sale.raisin + sale.oatmeal + sale.chocolate_chip = 70 := by
  sorry

#check bake_sale_total

end NUMINAMATH_CALUDE_bake_sale_total_l2531_253141


namespace NUMINAMATH_CALUDE_sum_of_digits_of_number_l2531_253112

/-- The sum of the digits of 10^100 - 57 -/
def sum_of_digits : ℕ := 889

/-- The number we're considering -/
def number : ℕ := 10^100 - 57

/-- Theorem stating that the sum of the digits of our number is equal to sum_of_digits -/
theorem sum_of_digits_of_number : 
  (number.digits 10).sum = sum_of_digits := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_number_l2531_253112


namespace NUMINAMATH_CALUDE_cos_2x_minus_pi_4_graph_translation_l2531_253100

open Real

theorem cos_2x_minus_pi_4_graph_translation (x : ℝ) : 
  cos (2*x - π/4) = sin (2*(x + π/8)) := by sorry

end NUMINAMATH_CALUDE_cos_2x_minus_pi_4_graph_translation_l2531_253100


namespace NUMINAMATH_CALUDE_unique_solution_for_power_sum_l2531_253113

theorem unique_solution_for_power_sum : 
  ∃! (x y z : ℕ), x < y ∧ y < z ∧ 3^x + 3^y + 3^z = 179415 ∧ x = 4 ∧ y = 7 ∧ z = 11 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_power_sum_l2531_253113


namespace NUMINAMATH_CALUDE_clerk_salary_l2531_253184

theorem clerk_salary (manager_salary : ℝ) (num_managers : ℕ) (num_clerks : ℕ) (total_salary : ℝ) :
  manager_salary = 5 →
  num_managers = 2 →
  num_clerks = 3 →
  total_salary = 16 →
  ∃ (clerk_salary : ℝ), clerk_salary = 2 ∧ total_salary = num_managers * manager_salary + num_clerks * clerk_salary :=
by
  sorry

end NUMINAMATH_CALUDE_clerk_salary_l2531_253184


namespace NUMINAMATH_CALUDE_line_inclination_angle_l2531_253169

/-- The angle of inclination of a line passing through (0, 0) and (1, -1) is 135°. -/
theorem line_inclination_angle : 
  let l : Set (ℝ × ℝ) := {p | ∃ t : ℝ, p = (t, -t)}
  let angle : ℝ := Real.arctan (-1) * (180 / Real.pi)
  angle = 135 := by sorry

end NUMINAMATH_CALUDE_line_inclination_angle_l2531_253169


namespace NUMINAMATH_CALUDE_function_properties_l2531_253198

open Real

noncomputable def f (a b x : ℝ) : ℝ := a * log x - b * x^2 + 1

theorem function_properties (a b : ℝ) :
  (∀ x > 0, deriv (f a b) x = 3 → f a b 1 = 1/2) →
  (a = 4 ∧ b = 1/2) ∧
  (∀ x ∈ Set.Icc (1/ℯ) (ℯ^2), f 4 (1/2) x ≤ 4 * log 2 - 1) ∧
  (∃ x ∈ Set.Icc (1/ℯ) (ℯ^2), f 4 (1/2) x = 4 * log 2 - 1) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2531_253198


namespace NUMINAMATH_CALUDE_f_expression_for_x_less_than_2_l2531_253171

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem f_expression_for_x_less_than_2
  (f : ℝ → ℝ)
  (h1 : is_even_function (λ x ↦ f (x + 2)))
  (h2 : ∀ x ≥ 2, f x = 3^x - 1) :
  ∀ x < 2, f x = 3^(4 - x) - 1 := by
  sorry

end NUMINAMATH_CALUDE_f_expression_for_x_less_than_2_l2531_253171


namespace NUMINAMATH_CALUDE_competition_results_l2531_253118

def team_a_scores : List ℝ := [7, 8, 9, 7, 10, 10, 9, 10, 10, 10]
def team_b_scores : List ℝ := [10, 8, 7, 9, 8, 10, 10, 9, 10, 9]
def team_a_variance : ℝ := 1.4

def median (l : List ℝ) : ℝ := sorry
def mode (l : List ℝ) : ℝ := sorry
def average (l : List ℝ) : ℝ := sorry
def variance (l : List ℝ) : ℝ := sorry

theorem competition_results :
  (median team_a_scores = 9.5) ∧
  (mode team_b_scores = 10) ∧
  (average team_b_scores = 9) ∧
  (variance team_b_scores = 1) ∧
  (variance team_b_scores < team_a_variance) :=
by sorry

end NUMINAMATH_CALUDE_competition_results_l2531_253118


namespace NUMINAMATH_CALUDE_obtuse_triangle_side_range_l2531_253196

theorem obtuse_triangle_side_range (x : ℝ) : 
  (x > 0 ∧ x + 1 > 0 ∧ x + 2 > 0) →  -- Positive side lengths
  (x + (x + 1) > (x + 2) ∧ (x + 2) + x > (x + 1) ∧ (x + 2) + (x + 1) > x) →  -- Triangle inequality
  ((x + 2)^2 > x^2 + (x + 1)^2) →  -- Obtuse triangle condition
  (1 < x ∧ x < 3) :=
by sorry

end NUMINAMATH_CALUDE_obtuse_triangle_side_range_l2531_253196


namespace NUMINAMATH_CALUDE_strictly_decreasing_implies_inequality_odd_function_property_l2531_253172

-- Define a function f from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Statement 1
theorem strictly_decreasing_implies_inequality (h : ∀ x y, x < y → f x > f y) : f (-4) > f 4 := by
  sorry

-- Statement 2
theorem odd_function_property (h : ∀ x, f (-x) = -f x) : f (-4) + f 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_strictly_decreasing_implies_inequality_odd_function_property_l2531_253172


namespace NUMINAMATH_CALUDE_point_inside_circle_l2531_253189

theorem point_inside_circle (a b : ℝ) : 
  a ≠ b → 
  a^2 - a - Real.sqrt 2 = 0 → 
  b^2 - b - Real.sqrt 2 = 0 → 
  a^2 + b^2 < 8 := by
sorry

end NUMINAMATH_CALUDE_point_inside_circle_l2531_253189


namespace NUMINAMATH_CALUDE_square_root_equal_self_l2531_253187

theorem square_root_equal_self (a : ℝ) : 
  (Real.sqrt a = a) → (a^2 + 1 = 1 ∨ a^2 + 1 = 2) := by
  sorry

end NUMINAMATH_CALUDE_square_root_equal_self_l2531_253187


namespace NUMINAMATH_CALUDE_unruly_max_sum_squares_l2531_253101

/-- A quadratic polynomial q(x) with real coefficients a and b -/
def q (a b x : ℝ) : ℝ := x^2 - (a+b)*x + a*b - 1

/-- The condition for q to be unruly -/
def is_unruly (a b : ℝ) : Prop :=
  ∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    ∀ w, q a b (q a b w) = 0 ↔ w = x ∨ w = y ∨ w = z

/-- The sum of squares of roots of q(x) -/
def sum_of_squares (a b : ℝ) : ℝ := (a+b)^2 + 2*(a*b - 1)

/-- Theorem stating that the unruly polynomial maximizing the sum of squares of its roots satisfies q(1) = -3 -/
theorem unruly_max_sum_squares :
  ∃ (a b : ℝ), is_unruly a b ∧
    (∀ (c d : ℝ), is_unruly c d → sum_of_squares c d ≤ sum_of_squares a b) ∧
    q a b 1 = -3 :=
sorry

end NUMINAMATH_CALUDE_unruly_max_sum_squares_l2531_253101
