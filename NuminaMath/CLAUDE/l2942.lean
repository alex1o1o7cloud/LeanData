import Mathlib

namespace NUMINAMATH_CALUDE_cousin_payment_l2942_294285

def friend_payment : ℕ := 5
def brother_payment : ℕ := 8
def total_days : ℕ := 7
def total_amount : ℕ := 119

theorem cousin_payment (cousin_pay : ℕ) : 
  (friend_payment * total_days + brother_payment * total_days + cousin_pay * total_days = total_amount) →
  cousin_pay = 4 := by
sorry

end NUMINAMATH_CALUDE_cousin_payment_l2942_294285


namespace NUMINAMATH_CALUDE_composite_divisor_of_product_l2942_294236

def product_up_to (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem composite_divisor_of_product (m : ℕ) :
  m > 1 →
  (m ∣ product_up_to m) ↔ (¬ Nat.Prime m ∧ m ≠ 4) :=
by sorry

end NUMINAMATH_CALUDE_composite_divisor_of_product_l2942_294236


namespace NUMINAMATH_CALUDE_cans_display_rows_l2942_294211

def triangular_display (n : ℕ) : ℕ := (3 * n * (n + 1)) / 2

theorem cans_display_rows :
  ∃ (n : ℕ), triangular_display n = 225 ∧ n = 11 := by
sorry

end NUMINAMATH_CALUDE_cans_display_rows_l2942_294211


namespace NUMINAMATH_CALUDE_no_quadratic_composition_with_given_zeros_l2942_294212

theorem no_quadratic_composition_with_given_zeros :
  ¬∃ (P Q : ℝ → ℝ),
    (∃ a b c : ℝ, ∀ x, P x = a * x^2 + b * x + c) ∧
    (∃ d e f : ℝ, ∀ x, Q x = d * x^2 + e * x + f) ∧
    (∀ x, (P ∘ Q) x = 0 ↔ x = 2 ∨ x = 3 ∨ x = 5 ∨ x = 7) :=
by sorry

end NUMINAMATH_CALUDE_no_quadratic_composition_with_given_zeros_l2942_294212


namespace NUMINAMATH_CALUDE_mikes_bills_l2942_294233

theorem mikes_bills (total_amount : ℕ) (bill_value : ℕ) (num_bills : ℕ) :
  total_amount = 45 →
  bill_value = 5 →
  total_amount = bill_value * num_bills →
  num_bills = 9 := by
sorry

end NUMINAMATH_CALUDE_mikes_bills_l2942_294233


namespace NUMINAMATH_CALUDE_multiply_to_325027405_l2942_294281

theorem multiply_to_325027405 (m : ℕ) : m * 32519 = 325027405 → m = 9995 := by
  sorry

end NUMINAMATH_CALUDE_multiply_to_325027405_l2942_294281


namespace NUMINAMATH_CALUDE_defective_pencils_count_l2942_294227

/-- The probability of selecting 3 non-defective pencils out of N non-defective pencils from a total of 6 pencils. -/
def probability (N : ℕ) : ℚ :=
  (Nat.choose N 3 : ℚ) / (Nat.choose 6 3 : ℚ)

/-- The number of defective pencils in a box of 6 pencils. -/
def num_defective (N : ℕ) : ℕ := 6 - N

theorem defective_pencils_count :
  ∃ N : ℕ, N ≤ 6 ∧ probability N = 1/5 ∧ num_defective N = 2 := by
  sorry

#check defective_pencils_count

end NUMINAMATH_CALUDE_defective_pencils_count_l2942_294227


namespace NUMINAMATH_CALUDE_cannonball_max_height_l2942_294268

/-- The height function of the cannonball -/
def h (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 36

/-- The maximum height reached by the cannonball -/
def max_height : ℝ := 161

/-- Theorem stating that the maximum height reached by the cannonball is 161 meters -/
theorem cannonball_max_height :
  ∀ t : ℝ, h t ≤ max_height :=
sorry

end NUMINAMATH_CALUDE_cannonball_max_height_l2942_294268


namespace NUMINAMATH_CALUDE_sum_divisible_by_11211_l2942_294255

-- Define the type for digits (0-9)
def Digit := Fin 10

-- Define a structure for five consecutive digits
structure ConsecutiveDigits where
  a : Digit
  h1 : a.val + 1 < 10  -- Ensure there are 4 consecutive digits after a
  b : Digit
  h2 : b.val = a.val + 1
  c : Digit
  h3 : c.val = a.val + 2
  d : Digit
  h4 : d.val = a.val + 3
  e : Digit
  h5 : e.val = a.val + 4

-- Define the function to create the number abcde
def makeNumber (digits : ConsecutiveDigits) : ℕ :=
  10000 * digits.a.val + 1000 * digits.b.val + 100 * digits.c.val + 10 * digits.d.val + digits.e.val

-- Define the function to create the reversed number edcba
def makeReversedNumber (digits : ConsecutiveDigits) : ℕ :=
  10000 * digits.e.val + 1000 * digits.d.val + 100 * digits.c.val + 10 * digits.b.val + digits.a.val

-- Theorem statement
theorem sum_divisible_by_11211 (digits : ConsecutiveDigits) :
  11211 ∣ (makeNumber digits + makeReversedNumber digits) := by
  sorry

end NUMINAMATH_CALUDE_sum_divisible_by_11211_l2942_294255


namespace NUMINAMATH_CALUDE_rhonda_marbles_l2942_294297

/-- Given that Amon and Rhonda have a total of 215 marbles, and Amon has 55 more marbles than Rhonda,
    prove that Rhonda has 80 marbles. -/
theorem rhonda_marbles (total : ℕ) (difference : ℕ) (rhonda : ℕ) : 
  total = 215 → difference = 55 → total = rhonda + (rhonda + difference) → rhonda = 80 := by
  sorry

end NUMINAMATH_CALUDE_rhonda_marbles_l2942_294297


namespace NUMINAMATH_CALUDE_tank_plastering_cost_l2942_294284

/-- Calculates the cost of plastering a rectangular tank's walls and bottom -/
def plasteringCost (length width depth rate : ℝ) : ℝ :=
  let wallArea := 2 * (length * depth + width * depth)
  let bottomArea := length * width
  let totalArea := wallArea + bottomArea
  totalArea * rate

/-- Theorem stating the cost of plastering the given tank -/
theorem tank_plastering_cost :
  plasteringCost 25 12 6 0.75 = 558 := by
  sorry

#eval plasteringCost 25 12 6 0.75

end NUMINAMATH_CALUDE_tank_plastering_cost_l2942_294284


namespace NUMINAMATH_CALUDE_thomas_leftover_money_l2942_294201

theorem thomas_leftover_money (num_books : ℕ) (book_price : ℚ) (record_price : ℚ) (num_records : ℕ) :
  num_books = 200 →
  book_price = 3/2 →
  record_price = 3 →
  num_records = 75 →
  (num_books : ℚ) * book_price - (num_records : ℚ) * record_price = 75 :=
by sorry

end NUMINAMATH_CALUDE_thomas_leftover_money_l2942_294201


namespace NUMINAMATH_CALUDE_danny_collection_difference_l2942_294219

/-- Represents Danny's collection of bottle caps and wrappers --/
structure Collection where
  park_caps : ℕ
  park_wrappers : ℕ
  beach_caps : ℕ
  beach_wrappers : ℕ
  forest_caps : ℕ
  forest_wrappers : ℕ
  previous_caps : ℕ
  previous_wrappers : ℕ

/-- Calculates the total number of bottle caps in the collection --/
def total_caps (c : Collection) : ℕ :=
  c.park_caps + c.beach_caps + c.forest_caps + c.previous_caps

/-- Calculates the total number of wrappers in the collection --/
def total_wrappers (c : Collection) : ℕ :=
  c.park_wrappers + c.beach_wrappers + c.forest_wrappers + c.previous_wrappers

/-- Theorem stating the difference between bottle caps and wrappers in Danny's collection --/
theorem danny_collection_difference :
  ∀ (c : Collection),
  c.park_caps = 58 →
  c.park_wrappers = 25 →
  c.beach_caps = 34 →
  c.beach_wrappers = 15 →
  c.forest_caps = 21 →
  c.forest_wrappers = 32 →
  c.previous_caps = 12 →
  c.previous_wrappers = 11 →
  total_caps c - total_wrappers c = 42 := by
  sorry

end NUMINAMATH_CALUDE_danny_collection_difference_l2942_294219


namespace NUMINAMATH_CALUDE_kyle_car_payment_l2942_294229

def monthly_income : ℝ := 3200

def rent : ℝ := 1250
def utilities : ℝ := 150
def retirement_savings : ℝ := 400
def groceries : ℝ := 300
def insurance : ℝ := 200
def miscellaneous : ℝ := 200
def gas_maintenance : ℝ := 350

def other_expenses : ℝ := rent + utilities + retirement_savings + groceries + insurance + miscellaneous + gas_maintenance

def car_payment : ℝ := monthly_income - other_expenses

theorem kyle_car_payment :
  car_payment = 350 := by sorry

end NUMINAMATH_CALUDE_kyle_car_payment_l2942_294229


namespace NUMINAMATH_CALUDE_compute_expression_l2942_294221

theorem compute_expression : 10 + 4 * (5 + 3)^3 = 2058 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l2942_294221


namespace NUMINAMATH_CALUDE_function_relation_characterization_l2942_294277

theorem function_relation_characterization 
  (f g : ℕ → ℕ) 
  (h : ∀ m n : ℕ, f m - f n = (m - n) * (g m + g n)) :
  ∃ a b c : ℕ, 
    (∀ n : ℕ, f n = a * n^2 + 2 * b * n + c) ∧ 
    (∀ n : ℕ, g n = a * n + b) :=
by sorry

end NUMINAMATH_CALUDE_function_relation_characterization_l2942_294277


namespace NUMINAMATH_CALUDE_exchange_10_dollars_equals_1200_yen_l2942_294228

/-- The exchange rate from US dollars to Japanese yen -/
def exchange_rate : ℝ := 120

/-- The amount of US dollars to be exchanged -/
def dollars_to_exchange : ℝ := 10

/-- The function that calculates the amount of yen received for a given amount of dollars -/
def exchange (dollars : ℝ) : ℝ := dollars * exchange_rate

theorem exchange_10_dollars_equals_1200_yen :
  exchange dollars_to_exchange = 1200 := by
  sorry

end NUMINAMATH_CALUDE_exchange_10_dollars_equals_1200_yen_l2942_294228


namespace NUMINAMATH_CALUDE_triangle_side_b_value_l2942_294256

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if the area is 2√3, B = π/3, and a² + c² = 3ac, then b = 4. -/
theorem triangle_side_b_value (a b c : ℝ) (A B C : ℝ) :
  (1/2 * a * c * Real.sin B = 2 * Real.sqrt 3) →  -- Area condition
  (B = π/3) →                                     -- Angle B condition
  (a^2 + c^2 = 3*a*c) →                           -- Relation between a, c
  (b^2 = a^2 + c^2 - 2*a*c*Real.cos B) →          -- Law of cosines
  (b = 4) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_b_value_l2942_294256


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l2942_294208

theorem arithmetic_mean_problem (y : ℝ) : 
  (8 + 15 + 20 + 5 + y) / 5 = 12 → y = 12 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l2942_294208


namespace NUMINAMATH_CALUDE_crayon_selection_combinations_l2942_294280

theorem crayon_selection_combinations : Nat.choose 15 5 = 3003 := by
  sorry

end NUMINAMATH_CALUDE_crayon_selection_combinations_l2942_294280


namespace NUMINAMATH_CALUDE_perfect_square_binomial_l2942_294273

/-- If 9x^2 - 18x + a is the square of a binomial, then a = 9 -/
theorem perfect_square_binomial (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 9*x^2 - 18*x + a = (3*x + b)^2) → a = 9 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_binomial_l2942_294273


namespace NUMINAMATH_CALUDE_constant_sum_of_powers_l2942_294218

theorem constant_sum_of_powers (n : ℕ) : 
  (∀ x y z : ℝ, x + y + z = 0 → x * y * z = 1 → 
    ∃ c : ℝ, ∀ x' y' z' : ℝ, x' + y' + z' = 0 → x' * y' * z' = 1 → 
      x'^n + y'^n + z'^n = c) ↔ 
  n = 1 ∨ n = 3 := by sorry

end NUMINAMATH_CALUDE_constant_sum_of_powers_l2942_294218


namespace NUMINAMATH_CALUDE_parabola_intersection_condition_l2942_294291

theorem parabola_intersection_condition (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 - 2*(m+2)*x₁ + m^2 - 1 = 0 ∧ 
    x₂^2 - 2*(m+2)*x₂ + m^2 - 1 = 0) 
  ↔ m > -5/4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_condition_l2942_294291


namespace NUMINAMATH_CALUDE_fruits_remaining_l2942_294250

/-- Calculates the number of fruits remaining after picking from multiple trees -/
theorem fruits_remaining
  (num_trees : ℕ)
  (fruits_per_tree : ℕ)
  (fraction_picked : ℚ)
  (h1 : num_trees = 8)
  (h2 : fruits_per_tree = 200)
  (h3 : fraction_picked = 2/5) :
  num_trees * fruits_per_tree - num_trees * (fruits_per_tree * fraction_picked) = 960 :=
by
  sorry

#check fruits_remaining

end NUMINAMATH_CALUDE_fruits_remaining_l2942_294250


namespace NUMINAMATH_CALUDE_sum_abc_equals_16_l2942_294261

theorem sum_abc_equals_16 (a b c : ℕ+) 
  (h1 : a * b + 2 * c + 3 = 47)
  (h2 : b * c + 2 * a + 3 = 47)
  (h3 : a * c + 2 * b + 3 = 47) :
  a + b + c = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_abc_equals_16_l2942_294261


namespace NUMINAMATH_CALUDE_Z_in_third_quadrant_l2942_294244

-- Define the complex number Z
def Z : ℂ := -1 + (1 - Complex.I)^2

-- Theorem stating that Z is in the third quadrant
theorem Z_in_third_quadrant :
  Real.sign (Z.re) = -1 ∧ Real.sign (Z.im) = -1 :=
sorry


end NUMINAMATH_CALUDE_Z_in_third_quadrant_l2942_294244


namespace NUMINAMATH_CALUDE_construct_line_segment_l2942_294270

/-- A straight edge tool -/
structure StraightEdge where
  length : ℝ

/-- A right-angled triangle tool -/
structure RightTriangle where
  hypotenuse : ℝ

/-- A construction using given tools -/
structure Construction where
  straightEdge : StraightEdge
  rightTriangle : RightTriangle

/-- Theorem stating that a line segment of 37 cm can be constructed
    with a 20 cm straight edge and a right triangle with 15 cm hypotenuse -/
theorem construct_line_segment
  (c : Construction)
  (h1 : c.straightEdge.length = 20)
  (h2 : c.rightTriangle.hypotenuse = 15) :
  ∃ (segment_length : ℝ), segment_length = 37 ∧ 
  (∃ (constructed_segment : ℝ → ℝ → Prop), 
    constructed_segment 0 segment_length) :=
sorry

end NUMINAMATH_CALUDE_construct_line_segment_l2942_294270


namespace NUMINAMATH_CALUDE_max_expr_value_l2942_294204

def S : Finset ℕ := {1, 2, 3, 4}

def expr (e f g h : ℕ) : ℕ := e * f^g - h

theorem max_expr_value :
  ∃ (e f g h : ℕ), e ∈ S ∧ f ∈ S ∧ g ∈ S ∧ h ∈ S ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ f ≠ g ∧ f ≠ h ∧ g ≠ h ∧
  expr e f g h = 161 ∧
  ∀ (a b c d : ℕ), a ∈ S → b ∈ S → c ∈ S → d ∈ S →
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
  expr a b c d ≤ 161 :=
by sorry

end NUMINAMATH_CALUDE_max_expr_value_l2942_294204


namespace NUMINAMATH_CALUDE_points_in_quadrants_I_and_II_l2942_294266

/-- A point in the 2D plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- The set of points satisfying the given inequalities -/
def SatisfyingPoints : Set Point :=
  {p : Point | p.y > 3 * p.x ∧ p.y > 5 - 2 * p.x}

/-- A point is in Quadrant I if both x and y are positive -/
def InQuadrantI (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- A point is in Quadrant II if x is negative and y is positive -/
def InQuadrantII (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Theorem: All points satisfying the inequalities are in Quadrants I or II -/
theorem points_in_quadrants_I_and_II :
  ∀ p ∈ SatisfyingPoints, InQuadrantI p ∨ InQuadrantII p :=
by sorry

end NUMINAMATH_CALUDE_points_in_quadrants_I_and_II_l2942_294266


namespace NUMINAMATH_CALUDE_function_property_l2942_294279

/-- Strictly increasing function from ℕ+ to ℕ+ -/
def StrictlyIncreasing (f : ℕ+ → ℕ+) : Prop :=
  ∀ x y : ℕ+, x < y → f x < f y

/-- The image of a function f : ℕ+ → ℕ+ -/
def Image (f : ℕ+ → ℕ+) : Set ℕ+ :=
  {y : ℕ+ | ∃ x : ℕ+, f x = y}

theorem function_property (f g : ℕ+ → ℕ+) 
  (h1 : StrictlyIncreasing f)
  (h2 : StrictlyIncreasing g)
  (h3 : Image f ∪ Image g = Set.univ)
  (h4 : Image f ∩ Image g = ∅)
  (h5 : ∀ n : ℕ+, g n = f (f n) + 1) :
  f 240 = 388 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l2942_294279


namespace NUMINAMATH_CALUDE_matrix_equation_l2942_294243

def A : Matrix (Fin 2) (Fin 2) ℚ := !![2, -5; 4, -3]
def B : Matrix (Fin 2) (Fin 2) ℚ := !![-20, -8; 12, 4]
def N : Matrix (Fin 2) (Fin 2) ℚ := !![46/7, -58/7; -26/7, 34/7]

theorem matrix_equation : N * A = B := by sorry

end NUMINAMATH_CALUDE_matrix_equation_l2942_294243


namespace NUMINAMATH_CALUDE_constant_function_equals_derivative_l2942_294213

theorem constant_function_equals_derivative :
  ∀ (f : ℝ → ℝ), (∀ x, f x = 0) → ∀ x, f x = deriv f x := by sorry

end NUMINAMATH_CALUDE_constant_function_equals_derivative_l2942_294213


namespace NUMINAMATH_CALUDE_pie_eating_contest_l2942_294260

theorem pie_eating_contest (first_student second_student : ℚ) :
  first_student = 7/8 ∧ second_student = 5/6 →
  first_student - second_student = 1/24 := by
  sorry

end NUMINAMATH_CALUDE_pie_eating_contest_l2942_294260


namespace NUMINAMATH_CALUDE_coffee_consumption_theorem_l2942_294249

/-- Represents the relationship between sleep and coffee consumption -/
def coffee_sleep_relation (sleep : ℝ) (coffee : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ sleep * coffee = k

theorem coffee_consumption_theorem (sleep_monday sleep_tuesday coffee_monday : ℝ) 
  (h1 : sleep_monday > 0)
  (h2 : sleep_tuesday > 0)
  (h3 : coffee_monday > 0)
  (h4 : coffee_sleep_relation sleep_monday coffee_monday)
  (h5 : coffee_sleep_relation sleep_tuesday (sleep_monday * coffee_monday / sleep_tuesday))
  (h6 : sleep_monday = 9)
  (h7 : sleep_tuesday = 6)
  (h8 : coffee_monday = 2) :
  sleep_monday * coffee_monday / sleep_tuesday = 3 := by
  sorry

#check coffee_consumption_theorem

end NUMINAMATH_CALUDE_coffee_consumption_theorem_l2942_294249


namespace NUMINAMATH_CALUDE_length_BI_isosceles_triangle_l2942_294231

/-- An isosceles triangle with given side lengths -/
structure IsoscelesTriangle where
  -- Side lengths
  ab : ℝ
  bc : ℝ
  -- Isosceles condition
  isIsosceles : ab > 0 ∧ bc > 0

/-- The incenter of a triangle -/
def incenter (t : IsoscelesTriangle) : ℝ × ℝ := sorry

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Theorem: Length of BI in isosceles triangle ABC -/
theorem length_BI_isosceles_triangle (t : IsoscelesTriangle) 
  (h1 : t.ab = 6) 
  (h2 : t.bc = 8) : 
  ∃ (ε : ℝ), abs (distance (0, 0) (incenter t) - 4.4 * Real.sqrt 1.1) < ε :=
sorry

end NUMINAMATH_CALUDE_length_BI_isosceles_triangle_l2942_294231


namespace NUMINAMATH_CALUDE_new_class_mean_l2942_294275

theorem new_class_mean (total_students : ℕ) (first_group : ℕ) (second_group : ℕ) 
  (first_mean : ℚ) (second_mean : ℚ) :
  total_students = first_group + second_group →
  first_group = 45 →
  second_group = 5 →
  first_mean = 80 / 100 →
  second_mean = 90 / 100 →
  (first_group * first_mean + second_group * second_mean) / total_students = 81 / 100 := by
sorry

end NUMINAMATH_CALUDE_new_class_mean_l2942_294275


namespace NUMINAMATH_CALUDE_wheels_in_garage_is_39_l2942_294274

/-- The number of wheels in a garage with various vehicles and items -/
def total_wheels_in_garage (
  num_cars : ℕ)
  (num_lawnmowers : ℕ)
  (num_bicycles : ℕ)
  (num_tricycles : ℕ)
  (num_unicycles : ℕ)
  (num_skateboards : ℕ)
  (num_wheelbarrows : ℕ)
  (num_four_wheeled_wagons : ℕ)
  (num_two_wheeled_dollies : ℕ)
  (num_four_wheeled_shopping_carts : ℕ)
  (num_two_wheeled_scooters : ℕ) : ℕ :=
  num_cars * 4 +
  num_lawnmowers * 4 +
  num_bicycles * 2 +
  num_tricycles * 3 +
  num_unicycles * 1 +
  num_skateboards * 4 +
  num_wheelbarrows * 1 +
  num_four_wheeled_wagons * 4 +
  num_two_wheeled_dollies * 2 +
  num_four_wheeled_shopping_carts * 4 +
  num_two_wheeled_scooters * 2

/-- Theorem stating that the total number of wheels in the garage is 39 -/
theorem wheels_in_garage_is_39 :
  total_wheels_in_garage 2 1 3 1 1 1 1 1 1 1 1 = 39 := by
  sorry

end NUMINAMATH_CALUDE_wheels_in_garage_is_39_l2942_294274


namespace NUMINAMATH_CALUDE_find_a_l2942_294239

def U : Finset ℕ := {1, 3, 5, 7}

theorem find_a (a : ℕ) : 
  let M : Finset ℕ := {1, a - 5}
  M ⊆ U ∧ 
  (U \ M : Finset ℕ) = {5, 7} →
  a = 8 := by
sorry

end NUMINAMATH_CALUDE_find_a_l2942_294239


namespace NUMINAMATH_CALUDE_inequality_solution_range_l2942_294237

theorem inequality_solution_range (d : ℝ) :
  (∃ x : ℝ, |2017 - x| + |2018 - x| ≤ d) → d ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l2942_294237


namespace NUMINAMATH_CALUDE_complex_sum_equal_negative_three_l2942_294232

theorem complex_sum_equal_negative_three (w : ℂ) 
  (h1 : w = Complex.exp (6 * Real.pi * Complex.I / 11))
  (h2 : w^11 = 1) :
  w / (1 + w^2) + w^2 / (1 + w^4) + w^3 / (1 + w^6) + w^4 / (1 + w^8) = -3 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_equal_negative_three_l2942_294232


namespace NUMINAMATH_CALUDE_total_cost_for_group_stay_l2942_294282

-- Define the rates and conditions
def weekdayRateFirstWeek : ℚ := 18
def weekendRateFirstWeek : ℚ := 20
def weekdayRateAdditionalWeeks : ℚ := 11
def weekendRateAdditionalWeeks : ℚ := 13
def securityDeposit : ℚ := 50
def groupDiscountRate : ℚ := 0.1
def groupSize : ℕ := 5
def stayDuration : ℕ := 23

-- Define the function to calculate the total cost
def calculateTotalCost : ℚ := sorry

-- Theorem statement
theorem total_cost_for_group_stay :
  calculateTotalCost = 327.6 := by sorry

end NUMINAMATH_CALUDE_total_cost_for_group_stay_l2942_294282


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2942_294240

/-- A geometric sequence with first term 1/3 and the property that 2a_2 = a_4 -/
def geometric_sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 1/3 ∧
  (∃ q : ℚ, ∀ n : ℕ, a n = 1/3 * q^(n-1)) ∧
  2 * (a 2) = a 4

theorem geometric_sequence_fifth_term
  (a : ℕ → ℚ)
  (h : geometric_sequence a) :
  a 5 = 4/3 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2942_294240


namespace NUMINAMATH_CALUDE_unit_circle_sector_angle_l2942_294287

/-- In a unit circle, a sector with area 1 has a central angle of 2 radians -/
theorem unit_circle_sector_angle (r : ℝ) (area : ℝ) (angle : ℝ) :
  r = 1 → area = 1 → angle = 2 * area / r → angle = 2 :=
by sorry

end NUMINAMATH_CALUDE_unit_circle_sector_angle_l2942_294287


namespace NUMINAMATH_CALUDE_f_properties_l2942_294215

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  Real.sqrt ((1 - x^2) / (1 + x^2)) + a * Real.sqrt ((1 + x^2) / (1 - x^2))

theorem f_properties (a : ℝ) (h : a > 0) :
  -- Function domain
  ∀ x : ℝ, -1 < x ∧ x < 1 →
  -- 1. Minimum value when a = 1
  (a = 1 → ∀ x : ℝ, -1 < x ∧ x < 1 → f 1 x ≥ 2) ∧
  -- 2. Monotonicity when a = 1
  (a = 1 → ∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y < 1 → f 1 x < f 1 y) ∧
  -- 3. Range of a for triangle formation
  (∀ r s t : ℝ, -2*Real.sqrt 5/5 ≤ r ∧ r ≤ 2*Real.sqrt 5/5 ∧
                -2*Real.sqrt 5/5 ≤ s ∧ s ≤ 2*Real.sqrt 5/5 ∧
                -2*Real.sqrt 5/5 ≤ t ∧ t ≤ 2*Real.sqrt 5/5 →
    f a r + f a s > f a t ∧ f a s + f a t > f a r ∧ f a t + f a r > f a s) ↔
  (1/15 < a ∧ a < 5/3) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l2942_294215


namespace NUMINAMATH_CALUDE_horner_v3_calculation_l2942_294290

/-- Horner's method V₃ calculation for a specific polynomial -/
theorem horner_v3_calculation (x : ℝ) (h : x = 4) : 
  let f := fun (x : ℝ) => 4*x^6 + 3*x^5 + 4*x^4 + 2*x^3 + 5*x^2 - 7*x + 9
  let v3 := (4*x + 3)*x + 4
  v3 = 80 := by
  sorry

end NUMINAMATH_CALUDE_horner_v3_calculation_l2942_294290


namespace NUMINAMATH_CALUDE_john_ate_ten_chips_l2942_294298

/-- The number of potato chips John ate -/
def num_potato_chips : ℕ := 10

/-- The number of calories in one potato chip -/
def calories_per_chip : ℕ := 6

/-- The number of cheezits John ate -/
def num_cheezits : ℕ := 6

/-- The total number of calories John ate -/
def total_calories : ℕ := 108

/-- The total number of calories from potato chips -/
def calories_from_chips : ℕ := 60

theorem john_ate_ten_chips :
  (num_potato_chips * calories_per_chip = calories_from_chips) ∧
  (num_cheezits * (calories_per_chip + calories_per_chip / 3) + calories_from_chips = total_calories) →
  num_potato_chips = 10 := by
  sorry

end NUMINAMATH_CALUDE_john_ate_ten_chips_l2942_294298


namespace NUMINAMATH_CALUDE_jacob_needs_18_marshmallows_l2942_294292

/-- Calculates the number of additional marshmallows needed for s'mores -/
def additional_marshmallows_needed (graham_crackers : ℕ) (marshmallows : ℕ) : ℕ :=
  let max_smores := graham_crackers / 2
  max_smores - marshmallows

/-- Proves that Jacob needs 18 more marshmallows -/
theorem jacob_needs_18_marshmallows :
  additional_marshmallows_needed 48 6 = 18 := by
  sorry

#eval additional_marshmallows_needed 48 6

end NUMINAMATH_CALUDE_jacob_needs_18_marshmallows_l2942_294292


namespace NUMINAMATH_CALUDE_pizza_toppings_l2942_294225

theorem pizza_toppings (total_slices : ℕ) (pepperoni_slices : ℕ) (mushroom_slices : ℕ) 
  (h1 : total_slices = 24)
  (h2 : pepperoni_slices = 14)
  (h3 : mushroom_slices = 16)
  (h4 : ∀ s, s ≤ total_slices → (s ≤ pepperoni_slices ∨ s ≤ mushroom_slices)) :
  ∃ both_toppings : ℕ, 
    both_toppings = pepperoni_slices + mushroom_slices - total_slices ∧ 
    both_toppings = 6 :=
by sorry

end NUMINAMATH_CALUDE_pizza_toppings_l2942_294225


namespace NUMINAMATH_CALUDE_simplified_fraction_sum_l2942_294222

theorem simplified_fraction_sum (a b : ℕ) (h : a = 75 ∧ b = 180) :
  let g := Nat.gcd a b
  (a / g) + (b / g) = 17 := by
  sorry

end NUMINAMATH_CALUDE_simplified_fraction_sum_l2942_294222


namespace NUMINAMATH_CALUDE_evaluate_expression_l2942_294252

theorem evaluate_expression (x : ℤ) (h : x = -2) : 5 * x + 7 = -3 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2942_294252


namespace NUMINAMATH_CALUDE_computer_profit_percentage_l2942_294226

theorem computer_profit_percentage (C : ℝ) (P : ℝ) : 
  2560 = C + 0.6 * C →
  2240 = C + P / 100 * C →
  P = 40 := by
sorry

end NUMINAMATH_CALUDE_computer_profit_percentage_l2942_294226


namespace NUMINAMATH_CALUDE_extreme_values_of_f_l2942_294200

/-- Given a positive constant a, prove the extreme values of f(x) = sin(2x) + √(sin²(2x) + a cos²(x)) -/
theorem extreme_values_of_f (a : ℝ) (ha : a > 0) :
  let f := fun (x : ℝ) => Real.sin (2 * x) + Real.sqrt ((Real.sin (2 * x))^2 + a * (Real.cos x)^2)
  (∀ x, f x ≥ 0) ∧ 
  (∃ x, f x = 0) ∧
  (∀ x, f x ≤ Real.sqrt (a + 4)) ∧
  (∃ x, f x = Real.sqrt (a + 4)) := by
  sorry

end NUMINAMATH_CALUDE_extreme_values_of_f_l2942_294200


namespace NUMINAMATH_CALUDE_value_of_a_l2942_294283

/-- Given a function f(x) = ax³ + 3x² - 6 where f'(-1) = 4, prove that a = 10/3 -/
theorem value_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x, f x = a * x^3 + 3 * x^2 - 6)
  (h2 : deriv f (-1) = 4) : 
  a = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l2942_294283


namespace NUMINAMATH_CALUDE_lawn_area_proof_l2942_294253

theorem lawn_area_proof (total_posts : ℕ) (post_spacing : ℕ) 
  (h_total_posts : total_posts = 24)
  (h_post_spacing : post_spacing = 5)
  (h_longer_side_posts : ∀ s l : ℕ, s + l = total_posts / 2 → l + 1 = 3 * (s + 1)) :
  ∃ short_side long_side : ℕ,
    short_side * long_side = 500 ∧
    short_side + 1 + long_side + 1 = total_posts ∧
    (long_side + 1) * post_spacing = (short_side + 1) * post_spacing * 3 :=
by sorry

end NUMINAMATH_CALUDE_lawn_area_proof_l2942_294253


namespace NUMINAMATH_CALUDE_min_value_theorem_l2942_294299

def f (x : ℝ) : ℝ := |2*x - 4| + |x + 1|

def g (x : ℝ) : ℝ := f x - |x - 2|

theorem min_value_theorem (m : ℝ) (hm : ∀ x, g x ≥ m) (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = m) :
  1/a + 1/b + 1/c ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2942_294299


namespace NUMINAMATH_CALUDE_blue_tshirts_count_l2942_294288

/-- Calculates the number of blue t-shirts in each pack -/
def blue_tshirts_per_pack (white_packs : ℕ) (blue_packs : ℕ) (white_per_pack : ℕ) (total_tshirts : ℕ) : ℕ :=
  let white_total := white_packs * white_per_pack
  let blue_total := total_tshirts - white_total
  blue_total / blue_packs

/-- Proves that the number of blue t-shirts in each pack is 9 -/
theorem blue_tshirts_count : blue_tshirts_per_pack 5 3 6 57 = 9 := by
  sorry

end NUMINAMATH_CALUDE_blue_tshirts_count_l2942_294288


namespace NUMINAMATH_CALUDE_heartsuit_three_four_l2942_294259

-- Define the heartsuit operation
def heartsuit (x y : ℝ) : ℝ := 4 * x + 6 * y

-- Theorem statement
theorem heartsuit_three_four : heartsuit 3 4 = 36 := by
  sorry

end NUMINAMATH_CALUDE_heartsuit_three_four_l2942_294259


namespace NUMINAMATH_CALUDE_sunday_to_friday_spending_ratio_l2942_294230

def friday_spending : ℝ := 20

theorem sunday_to_friday_spending_ratio :
  ∀ (sunday_multiple : ℝ),
  friday_spending + 2 * friday_spending + sunday_multiple * friday_spending = 120 →
  sunday_multiple * friday_spending / friday_spending = 3 := by
  sorry

end NUMINAMATH_CALUDE_sunday_to_friday_spending_ratio_l2942_294230


namespace NUMINAMATH_CALUDE_value_of_y_l2942_294217

theorem value_of_y (x y : ℝ) (h1 : x^2 - 3*x + 2 = y + 2) (h2 : x = -5) : y = 40 := by
  sorry

end NUMINAMATH_CALUDE_value_of_y_l2942_294217


namespace NUMINAMATH_CALUDE_rainfall_difference_l2942_294271

/-- Rainfall data for Thomas's science project in May --/
def rainfall_problem (day1 day2 day3 : ℝ) : Prop :=
  let normal_average := 140
  let this_year_total := normal_average - 58
  day1 = 26 ∧
  day2 = 34 ∧
  day3 < day2 ∧
  day1 + day2 + day3 = this_year_total

/-- The difference between the second and third day's rainfall is 12 cm --/
theorem rainfall_difference (day1 day2 day3 : ℝ) 
  (h : rainfall_problem day1 day2 day3) : day2 - day3 = 12 := by
  sorry


end NUMINAMATH_CALUDE_rainfall_difference_l2942_294271


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l2942_294214

theorem line_segment_endpoint (x : ℝ) : 
  x > 0 →
  Real.sqrt ((x - 2)^2 + (10 - 5)^2) = 13 →
  x = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l2942_294214


namespace NUMINAMATH_CALUDE_solution_of_linear_system_l2942_294210

theorem solution_of_linear_system :
  ∃ (x y : ℝ), x + 3 * y = 7 ∧ y = 2 * x ∧ x = 1 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_of_linear_system_l2942_294210


namespace NUMINAMATH_CALUDE_eighth_grade_students_l2942_294209

/-- The number of students in eighth grade -/
def total_students (num_girls : ℕ) (num_boys : ℕ) : ℕ :=
  num_girls + num_boys

/-- The relationship between the number of boys and girls -/
def boys_girls_relation (num_girls : ℕ) (num_boys : ℕ) : Prop :=
  num_boys = 2 * num_girls - 16

theorem eighth_grade_students :
  ∃ (num_boys : ℕ),
    boys_girls_relation 28 num_boys ∧
    total_students 28 num_boys = 68 :=
by sorry

end NUMINAMATH_CALUDE_eighth_grade_students_l2942_294209


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l2942_294269

theorem simplify_fraction_product : 8 * (15 / 9) * (-45 / 40) = -1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l2942_294269


namespace NUMINAMATH_CALUDE_milk_pour_problem_l2942_294207

theorem milk_pour_problem (initial_milk : ℚ) (pour_fraction : ℚ) :
  initial_milk = 3/8 →
  pour_fraction = 5/6 →
  pour_fraction * initial_milk = 5/16 := by
sorry

end NUMINAMATH_CALUDE_milk_pour_problem_l2942_294207


namespace NUMINAMATH_CALUDE_min_value_problem_l2942_294241

theorem min_value_problem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (sum_constraint : a + b + c = 12) (product_constraint : a * b * c = 27) :
  (a^2 + b^2) / (a + b) + (a^2 + c^2) / (a + c) + (b^2 + c^2) / (b + c) ≥ 12 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l2942_294241


namespace NUMINAMATH_CALUDE_tom_gave_balloons_to_fred_l2942_294216

/-- The number of balloons Tom gave to Fred -/
def balloons_given (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

theorem tom_gave_balloons_to_fred (initial : ℕ) (remaining : ℕ) 
  (h1 : initial = 30) (h2 : remaining = 14) :
  balloons_given initial remaining = 16 := by
  sorry

end NUMINAMATH_CALUDE_tom_gave_balloons_to_fred_l2942_294216


namespace NUMINAMATH_CALUDE_intersection_M_N_l2942_294267

-- Define the sets M and N
def M : Set ℝ := {x | x > -1}
def N : Set ℝ := {x | x^2 - x - 6 < 0}

-- State the theorem
theorem intersection_M_N :
  M ∩ N = {x : ℝ | -1 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2942_294267


namespace NUMINAMATH_CALUDE_sequence_ratio_l2942_294202

/-- Given an arithmetic sequence and a geometric sequence with specific properties, 
    prove that (b-a)/d = 1/2 --/
theorem sequence_ratio (a b c d e : ℝ) : 
  ((-1 : ℝ) - a = a - b) ∧ (b - (-4 : ℝ) = a - b) ∧  -- arithmetic sequence condition
  (c = (-1 : ℝ) * d / c) ∧ (d = c * e / d) ∧ (e = d * (-4 : ℝ) / e) →  -- geometric sequence condition
  (b - a) / d = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_ratio_l2942_294202


namespace NUMINAMATH_CALUDE_ellipse_foci_on_y_axis_iff_l2942_294251

/-- Represents an ellipse with the equation mx^2 + ny^2 = 1 -/
structure Ellipse (m n : ℝ) where
  equation : ∀ (x y : ℝ), m * x^2 + n * y^2 = 1

/-- Determines if an ellipse has foci on the y-axis -/
def hasFociOnYAxis (e : Ellipse m n) : Prop := sorry

/-- The main theorem stating that m > n > 0 is necessary and sufficient for
    the equation mx^2 + ny^2 = 1 to represent an ellipse with foci on the y-axis -/
theorem ellipse_foci_on_y_axis_iff (m n : ℝ) :
  (∃ e : Ellipse m n, hasFociOnYAxis e) ↔ m > n ∧ n > 0 := by sorry

end NUMINAMATH_CALUDE_ellipse_foci_on_y_axis_iff_l2942_294251


namespace NUMINAMATH_CALUDE_sixth_power_sum_l2942_294205

theorem sixth_power_sum (x : ℝ) (h : x + 1/x = 5) : x^6 + 1/x^6 = 12098 := by
  sorry

end NUMINAMATH_CALUDE_sixth_power_sum_l2942_294205


namespace NUMINAMATH_CALUDE_bryson_shoes_pairs_l2942_294264

/-- Given that Bryson has a total of 4 new shoes and a pair of shoes consists of 2 shoes,
    prove that the number of pairs of shoes he bought is 2. -/
theorem bryson_shoes_pairs : 
  ∀ (total_shoes : ℕ) (shoes_per_pair : ℕ),
    total_shoes = 4 →
    shoes_per_pair = 2 →
    total_shoes / shoes_per_pair = 2 := by
  sorry

end NUMINAMATH_CALUDE_bryson_shoes_pairs_l2942_294264


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2942_294276

/-- A quadratic function with a positive leading coefficient and symmetry about x = 2 -/
def symmetric_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a > 0 ∧ (∀ x, f x = a * x^2 + b * x + c) ∧ (∀ x, f x = f (4 - x))

theorem quadratic_inequality (f : ℝ → ℝ) (a : ℝ) 
  (h1 : symmetric_quadratic f) 
  (h2 : f (2 - a^2) < f (1 + a - a^2)) : 
  a < 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2942_294276


namespace NUMINAMATH_CALUDE_mans_age_to_sons_age_ratio_l2942_294293

/-- Proves that the ratio of a man's age to his son's age in two years is 2:1,
    given that the man is 26 years older than his son and the son's current age is 24 years. -/
theorem mans_age_to_sons_age_ratio (sons_current_age : ℕ) (age_difference : ℕ) : 
  sons_current_age = 24 →
  age_difference = 26 →
  (sons_current_age + age_difference + 2) / (sons_current_age + 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_mans_age_to_sons_age_ratio_l2942_294293


namespace NUMINAMATH_CALUDE_inequality_range_l2942_294242

theorem inequality_range (a : ℝ) : 
  (∀ (x θ : ℝ), θ ∈ Set.Icc 0 (Real.pi / 2) → 
    (x + 3 + 2 * Real.sin θ * Real.cos θ)^2 + (x + a * Real.sin θ + a * Real.cos θ)^2 ≥ 1/8) ↔ 
  (a ≤ Real.sqrt 6 ∨ a ≥ 7/2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l2942_294242


namespace NUMINAMATH_CALUDE_absolute_value_equation_l2942_294286

theorem absolute_value_equation (x z : ℝ) : 
  |3*x - 2*Real.log z| = 3*x + 2*Real.log z → x = 0 ∧ z = 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_l2942_294286


namespace NUMINAMATH_CALUDE_profit_maximum_l2942_294245

/-- Profit function for a product -/
def profit (m : ℝ) : ℝ := (m - 8) * (900 - 15 * m)

/-- Maximum profit expression -/
def max_profit_expr (m : ℝ) : ℝ := -15 * (m - 34)^2 + 10140

theorem profit_maximum :
  ∃ (m : ℝ), 
    (∀ (x : ℝ), profit x ≤ profit m) ∧
    (profit m = max_profit_expr m) ∧
    (m = 34) :=
sorry

end NUMINAMATH_CALUDE_profit_maximum_l2942_294245


namespace NUMINAMATH_CALUDE_parity_of_D_2024_2025_2026_l2942_294246

def D : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | 2 => 1
  | 3 => 2
  | n + 4 => D (n + 3) + D (n + 1)

theorem parity_of_D_2024_2025_2026 :
  Odd (D 2024) ∧ Even (D 2025) ∧ Even (D 2026) := by
  sorry

end NUMINAMATH_CALUDE_parity_of_D_2024_2025_2026_l2942_294246


namespace NUMINAMATH_CALUDE_function_property_l2942_294238

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- State the theorem
theorem function_property (h : ∀ x : ℝ, f (Real.exp x) = x + 2) :
  (f 1 = 2) ∧ (∀ x : ℝ, x > 0 → f x = Real.log x + 2) := by sorry

end NUMINAMATH_CALUDE_function_property_l2942_294238


namespace NUMINAMATH_CALUDE_neil_cookies_l2942_294257

theorem neil_cookies (total : ℕ) (first_fraction second_fraction third_fraction : ℚ) : 
  total = 60 ∧ 
  first_fraction = 1/3 ∧ 
  second_fraction = 1/4 ∧ 
  third_fraction = 2/5 →
  total - 
    (total * first_fraction).floor - 
    ((total - (total * first_fraction).floor) * second_fraction).floor - 
    ((total - (total * first_fraction).floor - ((total - (total * first_fraction).floor) * second_fraction).floor) * third_fraction).floor = 18 :=
by sorry

end NUMINAMATH_CALUDE_neil_cookies_l2942_294257


namespace NUMINAMATH_CALUDE_area_of_pentagon_l2942_294247

-- Define the points and lengths
structure Triangle :=
  (A B C : ℝ × ℝ)

def AB : ℝ := 5
def BC : ℝ := 3
def BD : ℝ := 3
def EC : ℝ := 1
def FD : ℝ := 2

-- Define the triangles
def triangleABC : Triangle := sorry
def triangleABD : Triangle := sorry

-- Define that ABC and ABD are right triangles
axiom ABC_right : triangleABC.C.1^2 + triangleABC.C.2^2 = AB^2
axiom ABD_right : triangleABD.C.1^2 + triangleABD.C.2^2 = AB^2

-- Define that C and D are on opposite sides of AB
axiom C_D_opposite : triangleABC.C.2 * triangleABD.C.2 < 0

-- Define points E and F
def E : ℝ × ℝ := sorry
def F : ℝ × ℝ := sorry

-- Define that E is on AC and F is on AD
axiom E_on_AC : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ E = (t * triangleABC.A.1 + (1 - t) * triangleABC.C.1, t * triangleABC.A.2 + (1 - t) * triangleABC.C.2)
axiom F_on_AD : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ F = (t * triangleABD.A.1 + (1 - t) * triangleABD.C.1, t * triangleABD.A.2 + (1 - t) * triangleABD.C.2)

-- Define the area of a polygon
def area (polygon : List (ℝ × ℝ)) : ℝ := sorry

-- Theorem to prove
theorem area_of_pentagon :
  area [E, C, B, D, F] = 303 / 25 := sorry

end NUMINAMATH_CALUDE_area_of_pentagon_l2942_294247


namespace NUMINAMATH_CALUDE_river_boat_capacity_l2942_294203

theorem river_boat_capacity (river_width : ℕ) (boat_width : ℕ) (space_required : ℕ) : 
  river_width = 42 ∧ boat_width = 3 ∧ space_required = 2 →
  (river_width / (boat_width + 2 * space_required) : ℕ) = 6 :=
by sorry

end NUMINAMATH_CALUDE_river_boat_capacity_l2942_294203


namespace NUMINAMATH_CALUDE_square_root_problem_l2942_294296

theorem square_root_problem (a b : ℝ) : 
  (∀ x : ℝ, x^2 = a + 11 → x = 1 ∨ x = -1) → 
  ((1 - b).sqrt = 4) → 
  (a = -10 ∧ b = -15 ∧ (2*a + 7*b)^(1/3 : ℝ) = -5) := by
sorry

end NUMINAMATH_CALUDE_square_root_problem_l2942_294296


namespace NUMINAMATH_CALUDE_third_circle_radius_l2942_294272

theorem third_circle_radius (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 19) (h₂ : r₂ = 29) :
  (r₂^2 - r₁^2) * π = π * r₃^2 → r₃ = 4 * Real.sqrt 30 := by
  sorry

end NUMINAMATH_CALUDE_third_circle_radius_l2942_294272


namespace NUMINAMATH_CALUDE_smallest_number_satisfying_congruences_l2942_294278

theorem smallest_number_satisfying_congruences : ∃ n : ℕ, 
  n > 0 ∧
  n % 4 = 1 ∧
  n % 5 = 1 ∧
  n % 6 = 1 ∧
  n % 7 = 0 ∧
  (∀ m : ℕ, m > 0 ∧ m % 4 = 1 ∧ m % 5 = 1 ∧ m % 6 = 1 ∧ m % 7 = 0 → m ≥ n) ∧
  n = 301 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_satisfying_congruences_l2942_294278


namespace NUMINAMATH_CALUDE_distance_ratio_bound_l2942_294235

/-- Manhattan distance between two points -/
def manhattan_distance (p q : ℝ × ℝ) : ℝ :=
  |p.1 - q.1| + |p.2 - q.2|

/-- The theorem to be proved -/
theorem distance_ratio_bound (points : Finset (ℝ × ℝ)) (h : points.card = 2023) :
  let distances := {d | ∃ (p q : ℝ × ℝ), p ∈ points ∧ q ∈ points ∧ p ≠ q ∧ d = manhattan_distance p q}
  (⨆ d ∈ distances, d) / (⨅ d ∈ distances, d) ≥ 44 :=
sorry

end NUMINAMATH_CALUDE_distance_ratio_bound_l2942_294235


namespace NUMINAMATH_CALUDE_gcd_50400_37800_l2942_294220

theorem gcd_50400_37800 : Nat.gcd 50400 37800 = 12600 := by
  sorry

end NUMINAMATH_CALUDE_gcd_50400_37800_l2942_294220


namespace NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l2942_294206

theorem complex_number_in_third_quadrant : 
  let z : ℂ := (2 - I) / I
  (z.re < 0) ∧ (z.im < 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l2942_294206


namespace NUMINAMATH_CALUDE_cylinder_volume_ratio_l2942_294224

/-- Theorem: For two cylinders with given properties, the ratio of their volumes is 3/2 -/
theorem cylinder_volume_ratio (S₁ S₂ V₁ V₂ : ℝ) (h₁ : S₁ / S₂ = 9 / 4) (h₂ : S₁ > 0) (h₃ : S₂ > 0) : 
  ∃ (R r H h : ℝ), 
    R > 0 ∧ r > 0 ∧ H > 0 ∧ h > 0 ∧
    S₁ = π * R^2 ∧
    S₂ = π * r^2 ∧
    V₁ = π * R^2 * H ∧
    V₂ = π * r^2 * h ∧
    2 * π * R * H = 2 * π * r * h →
    V₁ / V₂ = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_ratio_l2942_294224


namespace NUMINAMATH_CALUDE_equation_two_solutions_l2942_294223

/-- The equation has exactly two distinct solutions when k < -3/8 -/
theorem equation_two_solutions (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (x₁ - 3) / (k * x₁ + 2) = 2 * x₁ ∧ 
    (x₂ - 3) / (k * x₂ + 2) = 2 * x₂ ∧
    (∀ x : ℝ, (x - 3) / (k * x + 2) = 2 * x → x = x₁ ∨ x = x₂)) ↔ 
  k < -3/8 :=
sorry

end NUMINAMATH_CALUDE_equation_two_solutions_l2942_294223


namespace NUMINAMATH_CALUDE_prob_two_tails_three_coins_l2942_294254

/-- A fair coin is a coin with equal probability of heads and tails. -/
def FairCoin : Type := Unit

/-- The outcome of tossing a coin. -/
inductive CoinOutcome
| Heads
| Tails

/-- The outcome of tossing multiple coins. -/
def MultiCoinOutcome (n : ℕ) := Fin n → CoinOutcome

/-- The number of coins being tossed. -/
def numCoins : ℕ := 3

/-- The total number of possible outcomes when tossing n fair coins. -/
def totalOutcomes (n : ℕ) : ℕ := 2^n

/-- The number of ways to get exactly k tails when tossing n coins. -/
def waysToGetKTails (n k : ℕ) : ℕ := Nat.choose n k

/-- The probability of an event is the number of favorable outcomes divided by the total number of possible outcomes. -/
def probability (favorableOutcomes totalOutcomes : ℕ) : ℚ := favorableOutcomes / totalOutcomes

/-- The main theorem: the probability of getting exactly 2 tails when tossing 3 fair coins is 3/8. -/
theorem prob_two_tails_three_coins : 
  probability (waysToGetKTails numCoins 2) (totalOutcomes numCoins) = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_tails_three_coins_l2942_294254


namespace NUMINAMATH_CALUDE_min_value_ab_l2942_294262

theorem min_value_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b = a + b + 3) :
  a * b ≥ 9 := by
sorry

end NUMINAMATH_CALUDE_min_value_ab_l2942_294262


namespace NUMINAMATH_CALUDE_equation_one_solutions_l2942_294265

theorem equation_one_solutions (x : ℝ) :
  3 * (x - 1)^2 = 12 ↔ x = 3 ∨ x = -1 := by sorry

end NUMINAMATH_CALUDE_equation_one_solutions_l2942_294265


namespace NUMINAMATH_CALUDE_expression_simplification_l2942_294258

theorem expression_simplification (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  ((x^3 + 1) / x * (y^3 + 1) / y) - ((x^3 - 1) / y * (y^3 - 1) / x) = 2*x^2 + 2*y^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2942_294258


namespace NUMINAMATH_CALUDE_line_intersection_l2942_294263

theorem line_intersection :
  ∃! p : ℝ × ℝ, 
    (p.2 = -3 * p.1 + 1) ∧ 
    (p.2 + 1 = 15 * p.1) ∧ 
    p = (1/9, 2/3) := by
  sorry

end NUMINAMATH_CALUDE_line_intersection_l2942_294263


namespace NUMINAMATH_CALUDE_complex_arithmetic_equality_l2942_294234

theorem complex_arithmetic_equality : 9 - (8 + 7) * 6 + 5^2 - (4 * 3) + 2 - 1 = -67 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equality_l2942_294234


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l2942_294294

theorem gcd_of_three_numbers : Nat.gcd 4557 (Nat.gcd 1953 5115) = 93 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l2942_294294


namespace NUMINAMATH_CALUDE_polynomial_inequality_implies_upper_bound_l2942_294289

theorem polynomial_inequality_implies_upper_bound (a : ℝ) :
  (∀ x ∈ Set.Icc 0 2, x^3 + x^2 + a < 0) → a < -12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_inequality_implies_upper_bound_l2942_294289


namespace NUMINAMATH_CALUDE_inequality_proof_l2942_294295

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 < (a / Real.sqrt (a^2 + b^2)) + (b / Real.sqrt (b^2 + c^2)) + (c / Real.sqrt (c^2 + a^2)) ∧
  (a / Real.sqrt (a^2 + b^2)) + (b / Real.sqrt (b^2 + c^2)) + (c / Real.sqrt (c^2 + a^2)) ≤ 3 * Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2942_294295


namespace NUMINAMATH_CALUDE_linear_system_ratio_l2942_294248

/-- Given a system of linear equations with a nontrivial solution, prove that xz/y^2 = 26/9 -/
theorem linear_system_ratio (x y z k : ℝ) : 
  x ≠ 0 → y ≠ 0 → z ≠ 0 →
  x + k * y + 4 * z = 0 →
  4 * x + k * y - 3 * z = 0 →
  x + 3 * y - 2 * z = 0 →
  x * z / (y ^ 2) = 26 / 9 := by
  sorry

end NUMINAMATH_CALUDE_linear_system_ratio_l2942_294248
