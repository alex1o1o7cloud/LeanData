import Mathlib

namespace sin_cos_sum_equals_half_l3572_357221

theorem sin_cos_sum_equals_half :
  Real.sin (13 * π / 180) * Real.cos (343 * π / 180) +
  Real.cos (13 * π / 180) * Real.sin (17 * π / 180) = 1 / 2 := by
  sorry

end sin_cos_sum_equals_half_l3572_357221


namespace oil_depth_conversion_l3572_357298

/-- Represents a right cylindrical tank with oil -/
structure OilTank where
  height : ℝ
  baseDiameter : ℝ
  sideOilDepth : ℝ

/-- Calculates the upright oil depth given a tank configuration -/
noncomputable def uprightOilDepth (tank : OilTank) : ℝ :=
  sorry

/-- Theorem stating the relationship between side oil depth and upright oil depth -/
theorem oil_depth_conversion (tank : OilTank) 
  (h1 : tank.height = 12)
  (h2 : tank.baseDiameter = 6)
  (h3 : tank.sideOilDepth = 2) :
  ∃ (ε : ℝ), abs (uprightOilDepth tank - 2.4) < ε ∧ ε < 0.1 :=
sorry

end oil_depth_conversion_l3572_357298


namespace shortest_ant_path_l3572_357241

/-- Represents a grid of square tiles -/
structure TileGrid where
  rows : ℕ
  columns : ℕ
  tileSize : ℝ

/-- Represents the path of an ant on a tile grid -/
def antPath (grid : TileGrid) : ℝ :=
  grid.tileSize * (grid.rows + grid.columns - 2)

/-- Theorem stating the shortest path for an ant on a 5x3 grid with tile size 10 -/
theorem shortest_ant_path :
  let grid : TileGrid := ⟨5, 3, 10⟩
  antPath grid = 80 := by
  sorry

#check shortest_ant_path

end shortest_ant_path_l3572_357241


namespace count_fives_in_S_l3572_357212

/-- The sum of an arithmetic sequence with first term 1, common difference 9, and last term 10^2013 -/
def S : ℕ := (1 + 10^2013) * ((10^2013 + 8) / 18)

/-- Counts the occurrences of a digit in a natural number -/
def countDigit (n : ℕ) (d : ℕ) : ℕ := sorry

theorem count_fives_in_S : countDigit S 5 = 4022 := by sorry

end count_fives_in_S_l3572_357212


namespace greatest_K_inequality_l3572_357251

theorem greatest_K_inequality : 
  ∃ (K : ℝ), K = 16 ∧ 
  (∀ (u v w : ℝ), u > 0 → v > 0 → w > 0 → u^2 > 4*v*w → 
    (u^2 - 4*v*w)^2 > K*(2*v^2 - u*w)*(2*w^2 - u*v)) ∧
  (∀ (K' : ℝ), K' > K → 
    ∃ (u v w : ℝ), u > 0 ∧ v > 0 ∧ w > 0 ∧ u^2 > 4*v*w ∧ 
      (u^2 - 4*v*w)^2 ≤ K'*(2*v^2 - u*w)*(2*w^2 - u*v)) :=
by sorry

end greatest_K_inequality_l3572_357251


namespace ceiling_sqrt_156_l3572_357214

theorem ceiling_sqrt_156 : ⌈Real.sqrt 156⌉ = 13 := by
  sorry

end ceiling_sqrt_156_l3572_357214


namespace sqrt_two_squared_times_three_to_fourth_l3572_357254

theorem sqrt_two_squared_times_three_to_fourth : Real.sqrt (2^2 * 3^4) = 18 := by
  sorry

end sqrt_two_squared_times_three_to_fourth_l3572_357254


namespace carly_backstroke_practice_days_l3572_357229

/-- Represents the number of days in a week -/
def daysInWeek : ℕ := 7

/-- Represents the number of weeks in a month -/
def weeksInMonth : ℕ := 4

/-- Represents the total hours Carly practices swimming in a month -/
def totalPracticeHours : ℕ := 96

/-- Represents the hours Carly practices butterfly stroke per day -/
def butterflyHoursPerDay : ℕ := 3

/-- Represents the days Carly practices butterfly stroke per week -/
def butterflyDaysPerWeek : ℕ := 4

/-- Represents the hours Carly practices backstroke per day -/
def backstrokeHoursPerDay : ℕ := 2

/-- Theorem stating that Carly practices backstroke 6 days a week -/
theorem carly_backstroke_practice_days :
  ∃ (backstrokeDaysPerWeek : ℕ),
    backstrokeDaysPerWeek * backstrokeHoursPerDay * weeksInMonth +
    butterflyDaysPerWeek * butterflyHoursPerDay * weeksInMonth = totalPracticeHours ∧
    backstrokeDaysPerWeek = 6 :=
by sorry

end carly_backstroke_practice_days_l3572_357229


namespace kids_difference_l3572_357292

theorem kids_difference (monday tuesday : ℕ) 
  (h1 : monday = 11) 
  (h2 : tuesday = 12) : 
  tuesday - monday = 1 := by
  sorry

end kids_difference_l3572_357292


namespace parabola_vertex_l3572_357227

/-- A parabola is defined by the equation y^2 + 10y + 4x + 9 = 0 -/
def parabola_equation (x y : ℝ) : Prop :=
  y^2 + 10*y + 4*x + 9 = 0

/-- The vertex of a parabola is the point where it turns -/
def is_vertex (x y : ℝ) : Prop :=
  ∀ t : ℝ, parabola_equation (x + t) (y + t) → t = 0

/-- The vertex of the parabola y^2 + 10y + 4x + 9 = 0 is the point (4, -5) -/
theorem parabola_vertex : is_vertex 4 (-5) := by
  sorry

end parabola_vertex_l3572_357227


namespace fifty_second_digit_of_1_17_l3572_357279

-- Define the decimal representation of 1/17
def decimal_rep_1_17 : ℚ := 1 / 17

-- Define the length of the repeating sequence
def repeat_length : ℕ := 16

-- Define the position we're interested in
def target_position : ℕ := 52

-- Define the function to get the nth digit after the decimal point
noncomputable def nth_digit (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem fifty_second_digit_of_1_17 : 
  nth_digit target_position = 8 := by sorry

end fifty_second_digit_of_1_17_l3572_357279


namespace ratio_inequality_l3572_357230

theorem ratio_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 2*b + 3*c)^2 / (a^2 + 2*b^2 + 3*c^2) ≤ 6 := by
  sorry

end ratio_inequality_l3572_357230


namespace expression_evaluation_l3572_357253

theorem expression_evaluation : 
  1 / 2^2 + ((2 / 3^3 * (3 / 2)^2) + 4^(1/2)) - 8 / (4^2 - 3^2) = 107/84 := by
  sorry

end expression_evaluation_l3572_357253


namespace inequality_solution_set_l3572_357226

theorem inequality_solution_set (x : ℝ) : (x - 2) * (3 - x) > 0 ↔ 2 < x ∧ x < 3 := by
  sorry

end inequality_solution_set_l3572_357226


namespace mashed_potatoes_tomatoes_difference_l3572_357284

/-- The number of students who suggested mashed potatoes -/
def mashed_potatoes : ℕ := 144

/-- The number of students who suggested bacon -/
def bacon : ℕ := 467

/-- The number of students who suggested tomatoes -/
def tomatoes : ℕ := 79

/-- The theorem stating the difference between the number of students who suggested
    mashed potatoes and those who suggested tomatoes -/
theorem mashed_potatoes_tomatoes_difference :
  mashed_potatoes - tomatoes = 65 := by sorry

end mashed_potatoes_tomatoes_difference_l3572_357284


namespace abc_sum_l3572_357297

theorem abc_sum (A B C : Nat) : 
  A < 10 → B < 10 → C < 10 →  -- A, B, C are single digits
  A ≠ B → B ≠ C → A ≠ C →     -- A, B, C are different
  (100 * A + 10 * B + C) * 4 = 1436 →  -- ABC + ABC + ABC + ABC = 1436
  A + B + C = 17 := by
sorry

end abc_sum_l3572_357297


namespace racket_sales_total_l3572_357280

/-- The total amount earned from selling rackets given the average price per pair and the number of pairs sold -/
theorem racket_sales_total (avg_price : ℝ) (num_pairs : ℕ) : 
  avg_price = 9.8 → num_pairs = 55 → avg_price * (num_pairs : ℝ) = 539 := by
  sorry

end racket_sales_total_l3572_357280


namespace postal_code_arrangements_l3572_357205

/-- The number of possible arrangements of four distinct digits -/
def fourDigitArrangements : ℕ := 24

/-- The set of digits used in the postal code -/
def postalCodeDigits : Finset ℕ := {2, 3, 5, 8}

/-- Theorem: The number of arrangements of four distinct digits equals 24 -/
theorem postal_code_arrangements :
  Finset.card (Finset.powersetCard 4 postalCodeDigits) = fourDigitArrangements :=
by sorry

end postal_code_arrangements_l3572_357205


namespace kevin_distance_after_six_hops_l3572_357224

/-- Kevin's hopping journey on a number line -/
def kevin_hop (total_distance : ℚ) (first_hop_fraction : ℚ) (subsequent_hop_fraction : ℚ) (num_hops : ℕ) : ℚ :=
  let first_hop := first_hop_fraction * total_distance
  let remaining_distance := total_distance - first_hop
  let subsequent_hops := remaining_distance * (1 - (1 - subsequent_hop_fraction) ^ (num_hops - 1))
  first_hop + subsequent_hops

/-- The theorem stating the distance Kevin has hopped after six hops -/
theorem kevin_distance_after_six_hops :
  kevin_hop 2 (1/4) (2/3) 6 = 1071/243 := by
  sorry

end kevin_distance_after_six_hops_l3572_357224


namespace range_of_a_l3572_357206

theorem range_of_a (x a : ℝ) : 
  x > 2 → a ≤ x + 2 / (x - 2) → ∃ s : ℝ, s = 2 + 2 * Real.sqrt 2 ∧ IsLUB {a | ∃ x > 2, a ≤ x + 2 / (x - 2)} s :=
sorry

end range_of_a_l3572_357206


namespace equal_roots_quadratic_l3572_357246

theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - m * x + 2 * x + 12 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 - m * y + 2 * y + 12 = 0 → y = x) ↔ 
  (m = -10 ∨ m = 14) :=
by sorry

end equal_roots_quadratic_l3572_357246


namespace electronic_shop_price_l3572_357228

def smartphone_price : ℝ := 300

def personal_computer_price (smartphone_price : ℝ) : ℝ :=
  smartphone_price + 500

def advanced_tablet_price (smartphone_price personal_computer_price : ℝ) : ℝ :=
  smartphone_price + personal_computer_price

def total_price (smartphone_price personal_computer_price advanced_tablet_price : ℝ) : ℝ :=
  smartphone_price + personal_computer_price + advanced_tablet_price

def discounted_price (total_price : ℝ) : ℝ :=
  total_price * 0.9

def final_price (discounted_price : ℝ) : ℝ :=
  discounted_price * 1.05

theorem electronic_shop_price :
  final_price (discounted_price (total_price smartphone_price 
    (personal_computer_price smartphone_price) 
    (advanced_tablet_price smartphone_price (personal_computer_price smartphone_price)))) = 2079 := by
  sorry

end electronic_shop_price_l3572_357228


namespace smallest_area_of_P_l3572_357294

/-- Represents a point on the grid --/
structure GridPoint where
  x : Nat
  y : Nat
  label : Nat
  deriving Repr

/-- Defines the properties of the grid --/
def grid : List GridPoint := sorry

/-- Checks if a label is divisible by 7 --/
def isDivisibleBySeven (n : Nat) : Bool :=
  n % 7 == 0

/-- Defines the convex polygon P --/
def P : Set GridPoint := sorry

/-- Calculates the area of a convex polygon --/
noncomputable def areaOfConvexPolygon (polygon : Set GridPoint) : Real := sorry

/-- States that P contains all points with labels divisible by 7 --/
axiom P_contains_divisible_by_seven :
  ∀ p : GridPoint, p ∈ grid → isDivisibleBySeven p.label → p ∈ P

/-- Theorem: The smallest possible area of P is 60.5 square units --/
theorem smallest_area_of_P :
  ∀ Q : Set GridPoint,
    (∀ p : GridPoint, p ∈ grid → isDivisibleBySeven p.label → p ∈ Q) →
    areaOfConvexPolygon P ≤ areaOfConvexPolygon Q ∧
    areaOfConvexPolygon P = 60.5 := by
  sorry

end smallest_area_of_P_l3572_357294


namespace sum_of_polygon_sides_l3572_357277

/-- The number of sides in a hexagon -/
def hexagon_sides : ℕ := 6

/-- The number of sides in a triangle -/
def triangle_sides : ℕ := 3

/-- The number of sides in a quadrilateral -/
def quadrilateral_sides : ℕ := 4

/-- Theorem: The sum of the sides of a hexagon, triangle, and quadrilateral is 13 -/
theorem sum_of_polygon_sides : 
  hexagon_sides + triangle_sides + quadrilateral_sides = 13 := by
  sorry

end sum_of_polygon_sides_l3572_357277


namespace direction_vector_of_line_l3572_357269

/-- Given a line with equation y = -1/2 * x + 1, prove that (2, -1) is a valid direction vector. -/
theorem direction_vector_of_line (x y : ℝ) :
  y = -1/2 * x + 1 →
  ∃ (t : ℝ), (x + 2*t, y - t) = (x, y) :=
by sorry

end direction_vector_of_line_l3572_357269


namespace tangent_sum_simplification_l3572_357258

theorem tangent_sum_simplification :
  (Real.tan (20 * π / 180) + Real.tan (30 * π / 180) + Real.tan (60 * π / 180) + Real.tan (70 * π / 180)) / Real.cos (10 * π / 180) =
  (2 * Real.cos (40 * π / 180)) / (Real.cos (10 * π / 180) ^ 2 * Real.cos (30 * π / 180) * Real.cos (60 * π / 180) * Real.cos (70 * π / 180)) :=
by sorry

end tangent_sum_simplification_l3572_357258


namespace most_likely_outcome_l3572_357285

-- Define the number of children
def n : ℕ := 5

-- Define the probability of a child being a boy or a girl
def p : ℚ := 1/2

-- Define the probability of all children being the same gender
def prob_all_same : ℚ := 2 * p^n

-- Define the probability of having 2 of one gender and 3 of the other
def prob_2_3 : ℚ := (n.choose 2) * p^n

-- Define the probability of having 4 of one gender and 1 of the other
def prob_4_1 : ℚ := 2 * (n.choose 1) * p^n

-- Theorem statement
theorem most_likely_outcome :
  prob_2_3 + prob_4_1 > prob_all_same :=
by sorry

end most_likely_outcome_l3572_357285


namespace social_gathering_handshakes_l3572_357266

theorem social_gathering_handshakes (n : ℕ) (h : n = 8) : 
  let total_people := 2 * n
  let handshakes_per_person := total_people - 2
  (total_people * handshakes_per_person) / 2 = 112 := by
sorry

end social_gathering_handshakes_l3572_357266


namespace right_angled_triangle_l3572_357222

theorem right_angled_triangle (A B C : ℝ) (h : A + B + C = π) 
  (eq : (Real.cos A) / 20 + (Real.cos B) / 21 + (Real.cos C) / 29 = 29 / 420) : 
  C = π / 2 := by
  sorry

end right_angled_triangle_l3572_357222


namespace arithmetic_sequence_common_difference_l3572_357238

def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem arithmetic_sequence_common_difference 
  (a₁ d : ℝ) 
  (h1 : arithmetic_sequence a₁ d 5 = 8)
  (h2 : arithmetic_sequence a₁ d 1 + arithmetic_sequence a₁ d 2 + arithmetic_sequence a₁ d 3 = 6) :
  d = 2 := by
  sorry

end arithmetic_sequence_common_difference_l3572_357238


namespace complex_equation_sum_l3572_357240

theorem complex_equation_sum (a b : ℝ) :
  (Complex.I : ℂ)^2 = -1 →
  (2 + b * Complex.I) / (1 - Complex.I) = a * Complex.I →
  a + b = 4 := by sorry

end complex_equation_sum_l3572_357240


namespace circle_properties_l3572_357264

/-- Given a circle with equation x^2 + y^2 - 4x + 2y + 4 = 0, 
    its radius is 1 and its center coordinates are (2, -1) -/
theorem circle_properties : 
  ∃ (r : ℝ) (x₀ y₀ : ℝ), 
    (∀ x y : ℝ, x^2 + y^2 - 4*x + 2*y + 4 = 0 ↔ (x - x₀)^2 + (y - y₀)^2 = r^2) ∧
    r = 1 ∧ x₀ = 2 ∧ y₀ = -1 :=
sorry

end circle_properties_l3572_357264


namespace complex_fraction_equals_one_minus_i_l3572_357202

theorem complex_fraction_equals_one_minus_i : 
  let i : ℂ := Complex.I
  2 / (1 + i) = 1 - i :=
by sorry

end complex_fraction_equals_one_minus_i_l3572_357202


namespace mike_picked_64_peaches_l3572_357260

/-- Calculates the number of peaches Mike picked from the orchard -/
def peaches_picked (initial : ℕ) (given_away : ℕ) (final : ℕ) : ℕ :=
  final - (initial - given_away)

/-- Theorem: Given the initial conditions, Mike picked 64 peaches from the orchard -/
theorem mike_picked_64_peaches (initial : ℕ) (given_away : ℕ) (final : ℕ)
    (h1 : initial = 34)
    (h2 : given_away = 12)
    (h3 : final = 86) :
  peaches_picked initial given_away final = 64 := by
  sorry

#eval peaches_picked 34 12 86

end mike_picked_64_peaches_l3572_357260


namespace final_number_bound_l3572_357268

/-- A function that represents the process of replacing two numbers with their arithmetic mean. -/
def replace (numbers : List ℝ) : List ℝ :=
  sorry

/-- The theorem stating that the final number is not less than 1/n. -/
theorem final_number_bound (n : ℕ) (h : n > 0) :
  ∃ (process : ℕ → List ℝ), 
    (process 0 = List.replicate n 1) ∧ 
    (∀ k, process (k + 1) = replace (process k)) ∧
    (∃ m, (process m).length = 1 ∧ 
      ∀ x ∈ process m, x ≥ 1 / n) :=
  sorry

end final_number_bound_l3572_357268


namespace all_propositions_imply_target_l3572_357262

theorem all_propositions_imply_target : ∀ (p q r : Prop),
  (p ∧ q ∧ r → (p → q) ∨ r) ∧
  (¬p ∧ q ∧ ¬r → (p → q) ∨ r) ∧
  (p ∧ ¬q ∧ r → (p → q) ∨ r) ∧
  (¬p ∧ ¬q ∧ ¬r → (p → q) ∨ r) :=
by sorry

#check all_propositions_imply_target

end all_propositions_imply_target_l3572_357262


namespace function_value_inequality_l3572_357243

theorem function_value_inequality (f : ℝ → ℝ) (h1 : Differentiable ℝ f) (h2 : ∀ x, deriv f x > 1) :
  f 3 > f 1 + 2 := by
  sorry

end function_value_inequality_l3572_357243


namespace negation_of_universal_proposition_l3572_357287

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, 2 * x^2 - 1 > 0)) ↔ (∃ x₀ : ℝ, 2 * x₀^2 - 1 ≤ 0) := by
  sorry

end negation_of_universal_proposition_l3572_357287


namespace isosceles_triangle_vertex_angle_l3572_357220

-- Define an isosceles triangle
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isIsosceles : (a = b) ∨ (b = c) ∨ (a = c)
  sumOfAngles : a + b + c = 180

-- Define the condition of angle ratio
def angleRatio (t : IsoscelesTriangle) : Prop :=
  (t.a = 2 * t.b) ∨ (t.b = 2 * t.c) ∨ (t.c = 2 * t.a)

-- Theorem statement
theorem isosceles_triangle_vertex_angle 
  (t : IsoscelesTriangle) 
  (h : angleRatio t) : 
  (t.a = 90 ∨ t.b = 90 ∨ t.c = 90) ∨ 
  (t.a = 36 ∨ t.b = 36 ∨ t.c = 36) := by
  sorry

end isosceles_triangle_vertex_angle_l3572_357220


namespace new_savings_approx_400_l3572_357278

/-- Represents the monthly salary in rupees -/
def monthly_salary : ℝ := 7272.727272727273

/-- Represents the initial savings rate as a decimal -/
def initial_savings_rate : ℝ := 0.10

/-- Represents the expense increase rate as a decimal -/
def expense_increase_rate : ℝ := 0.05

/-- Calculates the new monthly savings after the expense increase -/
def new_monthly_savings : ℝ :=
  monthly_salary * (1 - (1 - initial_savings_rate) * (1 + expense_increase_rate))

/-- Theorem stating that the new monthly savings is approximately 400 rupees -/
theorem new_savings_approx_400 :
  ∃ ε > 0, |new_monthly_savings - 400| < ε :=
sorry

end new_savings_approx_400_l3572_357278


namespace ratio_of_numbers_l3572_357234

theorem ratio_of_numbers (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) (h : x + y = 7 * (x - y)) : x / y = 4 / 3 := by
  sorry

end ratio_of_numbers_l3572_357234


namespace toms_calculation_l3572_357236

theorem toms_calculation (x y z : ℝ) 
  (h1 : (x + y) - z = 8) 
  (h2 : (x + y) + z = 20) : 
  x + y = 14 := by
sorry

end toms_calculation_l3572_357236


namespace product_of_three_consecutive_integers_divisibility_l3572_357281

theorem product_of_three_consecutive_integers_divisibility :
  ∀ n : ℕ, n > 0 →
  ∃ k : ℕ, (n - 1) * n * (n + 1) = 6 * k ∧
  ∀ m : ℕ, m > 6 → ∃ n : ℕ, n > 0 ∧ ¬(∃ k : ℕ, (n - 1) * n * (n + 1) = m * k) :=
by sorry

end product_of_three_consecutive_integers_divisibility_l3572_357281


namespace complex_modulus_problem_l3572_357293

theorem complex_modulus_problem (z : ℂ) (h : z * (1 - Complex.I) = 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_modulus_problem_l3572_357293


namespace cubic_sum_from_linear_and_quadratic_sum_l3572_357270

theorem cubic_sum_from_linear_and_quadratic_sum (x y : ℝ) 
  (h1 : x + y = 5) 
  (h2 : x^2 + y^2 = 17) : 
  x^3 + y^3 = 65 := by
  sorry

end cubic_sum_from_linear_and_quadratic_sum_l3572_357270


namespace solve_equations_l3572_357204

theorem solve_equations :
  (∃ x : ℝ, 4 * x = 2 * x + 6 ∧ x = 3) ∧
  (∃ x : ℝ, 3 * x + 5 = 6 * x - 1 ∧ x = 2) ∧
  (∃ x : ℝ, 3 * x - 2 * (x - 1) = 2 + 3 * (4 - x) ∧ x = 3) ∧
  (∃ x : ℝ, (x - 3) / 5 - (x + 4) / 2 = -2 ∧ x = -2) :=
by sorry

end solve_equations_l3572_357204


namespace inverse_variation_problem_l3572_357250

-- Define the inverse relationship between y and x^2
def inverse_relation (k : ℝ) (x y : ℝ) : Prop := y = k / (x^2)

-- Theorem statement
theorem inverse_variation_problem (k : ℝ) :
  (inverse_relation k 1 8) →
  (inverse_relation k 4 0.5) :=
by
  sorry

end inverse_variation_problem_l3572_357250


namespace shelter_cats_l3572_357210

theorem shelter_cats (total : ℕ) (tuna : ℕ) (chicken : ℕ) (both : ℕ) 
  (h1 : total = 75)
  (h2 : tuna = 18)
  (h3 : chicken = 55)
  (h4 : both = 10) :
  total - (tuna + chicken - both) = 12 :=
by sorry

end shelter_cats_l3572_357210


namespace ladder_problem_l3572_357231

theorem ladder_problem (ladder_length height : ℝ) 
  (h1 : ladder_length = 8.5)
  (h2 : height = 7.5) :
  ∃ base : ℝ, base = 4 ∧ base^2 + height^2 = ladder_length^2 :=
sorry

end ladder_problem_l3572_357231


namespace percentage_passed_both_subjects_l3572_357208

theorem percentage_passed_both_subjects 
  (failed_hindi : ℝ) 
  (failed_english : ℝ) 
  (failed_both : ℝ) 
  (h1 : failed_hindi = 25) 
  (h2 : failed_english = 35) 
  (h3 : failed_both = 40) : 
  100 - (failed_hindi + failed_english - failed_both) = 80 := by
  sorry

end percentage_passed_both_subjects_l3572_357208


namespace smallest_section_area_l3572_357245

/-- The area of the smallest circular section of a sphere circumscribed around a cube --/
theorem smallest_section_area (cube_edge : ℝ) (h : cube_edge = 4) : 
  let sphere_radius : ℝ := cube_edge * Real.sqrt 3 / 2
  let midpoint_to_center : ℝ := cube_edge * Real.sqrt 2 / 2
  let section_radius : ℝ := Real.sqrt (sphere_radius^2 - midpoint_to_center^2)
  π * section_radius^2 = 4 * π :=
by sorry

end smallest_section_area_l3572_357245


namespace shopping_cost_other_goods_l3572_357242

def tuna_packs : ℕ := 5
def tuna_price : ℚ := 2
def water_bottles : ℕ := 4
def water_price : ℚ := 3/2
def discount_rate : ℚ := 1/10
def paid_after_discount : ℚ := 56
def conversion_rate : ℚ := 3/2

theorem shopping_cost_other_goods :
  let total_cost := paid_after_discount / (1 - discount_rate)
  let tuna_water_cost := tuna_packs * tuna_price + water_bottles * water_price
  let other_goods_local := total_cost - tuna_water_cost
  let other_goods_home := other_goods_local / conversion_rate
  other_goods_home = 30.81 := by sorry

end shopping_cost_other_goods_l3572_357242


namespace bike_price_l3572_357261

theorem bike_price (upfront_payment : ℝ) (upfront_percentage : ℝ) (total_price : ℝ) : 
  upfront_payment = 240 ∧ upfront_percentage = 20 ∧ upfront_payment = (upfront_percentage / 100) * total_price →
  total_price = 1200 :=
by sorry

end bike_price_l3572_357261


namespace initial_orange_balloons_l3572_357257

theorem initial_orange_balloons (blue_balloons : ℕ) (lost_orange_balloons : ℕ) (remaining_orange_balloons : ℕ) : 
  blue_balloons = 4 → 
  lost_orange_balloons = 2 → 
  remaining_orange_balloons = 7 → 
  remaining_orange_balloons + lost_orange_balloons = 9 :=
by sorry

end initial_orange_balloons_l3572_357257


namespace high_school_ten_season_games_l3572_357274

/-- Represents a basketball conference -/
structure BasketballConference where
  teamCount : ℕ
  intraConferenceGamesPerPair : ℕ
  nonConferenceGamesPerTeam : ℕ

/-- Calculates the total number of games in a season for a given basketball conference -/
def totalSeasonGames (conf : BasketballConference) : ℕ :=
  let intraConferenceGames := conf.teamCount.choose 2 * conf.intraConferenceGamesPerPair
  let nonConferenceGames := conf.teamCount * conf.nonConferenceGamesPerTeam
  intraConferenceGames + nonConferenceGames

/-- The High School Ten basketball conference -/
def highSchoolTen : BasketballConference :=
  { teamCount := 10
  , intraConferenceGamesPerPair := 2
  , nonConferenceGamesPerTeam := 6 }

theorem high_school_ten_season_games :
  totalSeasonGames highSchoolTen = 150 := by
  sorry

end high_school_ten_season_games_l3572_357274


namespace percent_decrease_l3572_357215

theorem percent_decrease (original_price sale_price : ℝ) :
  original_price = 100 ∧ sale_price = 20 →
  (original_price - sale_price) / original_price * 100 = 80 := by
sorry

end percent_decrease_l3572_357215


namespace square_on_hypotenuse_l3572_357203

theorem square_on_hypotenuse (a b : ℝ) (ha : a = 9) (hb : b = 12) :
  let c := Real.sqrt (a^2 + b^2)
  let s := (a * b) / (a + b)
  s = 120 / 37 := by sorry

end square_on_hypotenuse_l3572_357203


namespace root_exists_in_interval_l3572_357237

def f (x : ℝ) := x^3 - x - 1

theorem root_exists_in_interval :
  ∃ c ∈ Set.Ioo 1 2, f c = 0 :=
by
  sorry

end root_exists_in_interval_l3572_357237


namespace quadratic_solution_l3572_357286

theorem quadratic_solution (m : ℝ) : 
  (2^2 - m*2 + 8 = 0) → m = 6 := by
  sorry

end quadratic_solution_l3572_357286


namespace problem_solution_l3572_357282

def A : Set ℤ := {-2, 3, 4, 6}
def B (a : ℤ) : Set ℤ := {3, a, a^2}

theorem problem_solution (a : ℤ) : 
  (B a ⊆ A → a = 2) ∧ 
  (A ∩ B a = {3, 4} → a = 2 ∨ a = 4) := by
  sorry

end problem_solution_l3572_357282


namespace constant_term_expansion_l3572_357259

theorem constant_term_expansion (x : ℝ) : 
  ∃ (f : ℝ → ℝ), (∀ x ≠ 0, f x = (x - 2 + 1/x)^4) ∧ 
  (∃ c : ℝ, ∀ x ≠ 0, f x = c + x * (f x - c) / x) ∧ 
  (∃ c : ℝ, ∀ x ≠ 0, f x = c + x * (f x - c) / x) ∧ c = 70 := by
  sorry

end constant_term_expansion_l3572_357259


namespace quadratic_inequality_l3572_357289

theorem quadratic_inequality (x : ℝ) : x^2 - x - 30 < 0 ↔ -5 < x ∧ x < 6 := by
  sorry

end quadratic_inequality_l3572_357289


namespace no_integer_solutions_for_hyperbola_l3572_357225

theorem no_integer_solutions_for_hyperbola : 
  ¬∃ (x y : ℤ), x^2 - y^2 = 2022 := by
  sorry

end no_integer_solutions_for_hyperbola_l3572_357225


namespace martin_crayon_boxes_l3572_357265

theorem martin_crayon_boxes
  (crayons_per_box : ℕ)
  (total_crayons : ℕ)
  (h1 : crayons_per_box = 7)
  (h2 : total_crayons = 56) :
  total_crayons / crayons_per_box = 8 := by
  sorry

end martin_crayon_boxes_l3572_357265


namespace nancy_purchase_cost_l3572_357235

/-- The cost of a set of crystal beads in dollars -/
def crystal_cost : ℕ := 9

/-- The cost of a set of metal beads in dollars -/
def metal_cost : ℕ := 10

/-- The number of crystal bead sets purchased -/
def crystal_sets : ℕ := 1

/-- The number of metal bead sets purchased -/
def metal_sets : ℕ := 2

/-- The total cost of the purchase in dollars -/
def total_cost : ℕ := crystal_cost * crystal_sets + metal_cost * metal_sets

theorem nancy_purchase_cost : total_cost = 29 := by
  sorry

end nancy_purchase_cost_l3572_357235


namespace complex_number_quadrant_l3572_357244

theorem complex_number_quadrant (z : ℂ) (h : z * (2 + Complex.I) = 3 - Complex.I) :
  z.re > 0 ∧ z.im < 0 := by
  sorry

end complex_number_quadrant_l3572_357244


namespace quadratic_function_properties_l3572_357232

theorem quadratic_function_properties (a c : ℕ+) (m : ℝ) :
  let f : ℝ → ℝ := fun x ↦ (a : ℝ) * x^2 + 2 * x + c
  (f 1 = 5) →
  (6 < f 2 ∧ f 2 < 11) →
  (∀ x ∈ Set.Icc (1/2 : ℝ) (3/2 : ℝ), f x - 2 * m * x ≤ 1) →
  (a = 1 ∧ c = 2 ∧ m ≥ 9/4) := by
  sorry

end quadratic_function_properties_l3572_357232


namespace shaded_region_area_l3572_357271

/-- The area of a shaded region formed by the intersection of two circles -/
theorem shaded_region_area (r : ℝ) (h : r = 5) : 
  (2 * (π * r^2 / 4) - r^2) = (50 * π - 100) / 4 := by
  sorry

#check shaded_region_area

end shaded_region_area_l3572_357271


namespace f_neg_five_l3572_357239

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^5 + b * x^3 + 1

-- State the theorem
theorem f_neg_five (a b : ℝ) (h : f a b 5 = 7) : f a b (-5) = -5 := by
  sorry

end f_neg_five_l3572_357239


namespace monotonicity_condition_positivity_condition_l3572_357218

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ := 4 * x^2 - k * x - 8

-- Define the interval [5, 20]
def I : Set ℝ := Set.Icc 5 20

-- Part I: Monotonicity condition
theorem monotonicity_condition (k : ℝ) :
  (∀ x ∈ I, ∀ y ∈ I, x ≤ y → f k x ≤ f k y) ∨
  (∀ x ∈ I, ∀ y ∈ I, x ≤ y → f k x ≥ f k y) ↔
  k ∈ Set.Iic 40 ∪ Set.Ici 160 :=
sorry

-- Part II: Positivity condition
theorem positivity_condition (k : ℝ) :
  (∀ x ∈ I, f k x > 0) ↔ k < 92/5 :=
sorry

end monotonicity_condition_positivity_condition_l3572_357218


namespace joint_purchase_popularity_l3572_357247

/-- Represents the benefits of joint purchases -/
structure JointPurchaseBenefits where
  cost_savings : ℝ
  information_sharing : ℝ

/-- Represents the drawbacks of joint purchases -/
structure JointPurchaseDrawbacks where
  risks : ℝ
  transactional_costs : ℝ

/-- Represents factors affecting joint purchases -/
structure JointPurchaseFactors where
  benefits : JointPurchaseBenefits
  drawbacks : JointPurchaseDrawbacks
  proximity_to_stores : ℝ
  delivery_cost_savings : ℝ

/-- Determines if joint purchases are popular based on given factors -/
def joint_purchases_popular (factors : JointPurchaseFactors) : Prop :=
  factors.benefits.cost_savings + factors.benefits.information_sharing >
  factors.drawbacks.risks

/-- Determines if joint purchases are popular among neighbors based on given factors -/
def joint_purchases_popular_neighbors (factors : JointPurchaseFactors) : Prop :=
  factors.delivery_cost_savings >
  factors.drawbacks.transactional_costs + factors.proximity_to_stores

theorem joint_purchase_popularity
  (factors_countries factors_neighbors : JointPurchaseFactors)
  (h1 : joint_purchases_popular factors_countries)
  (h2 : ¬joint_purchases_popular_neighbors factors_neighbors) :
  (factors_countries.benefits.cost_savings + factors_countries.benefits.information_sharing >
   factors_countries.drawbacks.risks) ∧
  (factors_neighbors.drawbacks.transactional_costs + factors_neighbors.proximity_to_stores ≥
   factors_neighbors.delivery_cost_savings) :=
by sorry

end joint_purchase_popularity_l3572_357247


namespace cutting_tool_distance_l3572_357291

-- Define the circle and points
def Circle (center : ℝ × ℝ) (radius : ℝ) := {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

def is_right_angle (A B C : ℝ × ℝ) : Prop :=
  (C.1 - B.1) * (A.1 - B.1) + (C.2 - B.2) * (A.2 - B.2) = 0

def distance_squared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.1 - p1.1)^2 + (p2.2 - p1.2)^2

-- State the theorem
theorem cutting_tool_distance (O A B C : ℝ × ℝ) :
  O = (0, 0) →
  A ∈ Circle O (Real.sqrt 72) →
  C ∈ Circle O (Real.sqrt 72) →
  distance_squared A B = 64 →
  distance_squared B C = 9 →
  is_right_angle A B C →
  distance_squared O B = 50 := by
  sorry

end cutting_tool_distance_l3572_357291


namespace largest_two_digit_divisible_by_six_ending_in_four_l3572_357233

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def ends_in_four (n : ℕ) : Prop := n % 10 = 4

theorem largest_two_digit_divisible_by_six_ending_in_four :
  ∃ (max : ℕ), 
    is_two_digit max ∧ 
    max % 6 = 0 ∧ 
    ends_in_four max ∧
    ∀ (n : ℕ), is_two_digit n → n % 6 = 0 → ends_in_four n → n ≤ max :=
by
  -- Proof goes here
  sorry

end largest_two_digit_divisible_by_six_ending_in_four_l3572_357233


namespace edge_enlargement_equals_graph_scale_l3572_357283

/-- A graph is represented by a set of edges, where each edge has a length. -/
structure Graph where
  edges : Set (ℝ)

/-- Enlarging a graph by multiplying each edge length by a factor. -/
def enlarge (g : Graph) (factor : ℝ) : Graph :=
  { edges := g.edges.image (· * factor) }

/-- The scale factor of a transformation that multiplies each edge by 4. -/
def scale_factor : ℝ := 4

theorem edge_enlargement_equals_graph_scale (g : Graph) :
  enlarge g scale_factor = enlarge g scale_factor :=
by
  sorry

end edge_enlargement_equals_graph_scale_l3572_357283


namespace fraction_of_c_grades_l3572_357216

theorem fraction_of_c_grades 
  (total_students : ℕ) 
  (a_fraction : ℚ) 
  (b_fraction : ℚ) 
  (d_count : ℕ) 
  (h_total : total_students = 800)
  (h_a : a_fraction = 1 / 5)
  (h_b : b_fraction = 1 / 4)
  (h_d : d_count = 40) :
  (total_students - (a_fraction * total_students + b_fraction * total_students + d_count)) / total_students = 1 / 2 := by
  sorry

end fraction_of_c_grades_l3572_357216


namespace first_child_born_1982_l3572_357211

/-- Represents the year the first child was born -/
def first_child_birth_year : ℕ := sorry

/-- The year the couple got married -/
def marriage_year : ℕ := 1980

/-- The year the second child was born -/
def second_child_birth_year : ℕ := 1984

/-- The year when the combined ages of children equal the years of marriage -/
def reference_year : ℕ := 1986

theorem first_child_born_1982 :
  (reference_year - first_child_birth_year) + (reference_year - second_child_birth_year) = reference_year - marriage_year →
  first_child_birth_year = 1982 :=
by sorry

end first_child_born_1982_l3572_357211


namespace correct_commutative_transformation_l3572_357200

-- Define the commutative property of addition
axiom commutative_add (a b : ℝ) : a + b = b + a

-- Define the associative property of addition
axiom associative_add (a b c : ℝ) : (a + b) + c = a + (b + c)

-- State the theorem
theorem correct_commutative_transformation :
  4 + (-6) + 3 = (-6) + 4 + 3 :=
by
  sorry

end correct_commutative_transformation_l3572_357200


namespace min_value_fraction_min_value_achieved_l3572_357219

theorem min_value_fraction (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) 
  (h_sum : a + b + 2 * c = 2) : 
  (a + b) / (a * b * c) ≥ 8 := by
  sorry

theorem min_value_achieved : ∃ a b c : ℝ, 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a + b + 2 * c = 2 ∧
  (a + b) / (a * b * c) = 8 := by
  sorry

end min_value_fraction_min_value_achieved_l3572_357219


namespace line_intersects_circle_l3572_357276

/-- Given a point (x₀, y₀) outside the circle x² + y² = r², 
    prove that the line x₀x + y₀y = r² intersects the circle. -/
theorem line_intersects_circle (x₀ y₀ r : ℝ) (h : x₀^2 + y₀^2 > r^2) :
  ∃ (x y : ℝ), x^2 + y^2 = r^2 ∧ x₀*x + y₀*y = r^2 := by
  sorry

end line_intersects_circle_l3572_357276


namespace skew_lines_sufficient_not_necessary_l3572_357255

-- Define the concept of a line in 3D space
structure Line3D where
  -- Add appropriate fields to represent a line in 3D space
  -- This is a simplified representation
  dummy : Unit

-- Define what it means for two lines to be skew
def are_skew (l1 l2 : Line3D) : Prop :=
  -- Add appropriate definition
  sorry

-- Define what it means for two lines to not intersect
def do_not_intersect (l1 l2 : Line3D) : Prop :=
  -- Add appropriate definition
  sorry

-- Theorem statement
theorem skew_lines_sufficient_not_necessary :
  (∀ l1 l2 : Line3D, are_skew l1 l2 → do_not_intersect l1 l2) ∧
  (∃ l1 l2 : Line3D, do_not_intersect l1 l2 ∧ ¬are_skew l1 l2) :=
sorry

end skew_lines_sufficient_not_necessary_l3572_357255


namespace g_range_l3572_357249

noncomputable def g (x : ℝ) : ℝ := 
  (Real.cos x ^ 3 + 7 * Real.cos x ^ 2 + 2 * Real.cos x + 3 * Real.sin x ^ 2 - 14) / (Real.cos x - 2)

theorem g_range : 
  ∀ x : ℝ, Real.cos x ≠ 2 → 
  (∃ y ∈ Set.Icc (1/2 : ℝ) (25/2 : ℝ), g x = y) ∧ 
  (∀ y : ℝ, g x = y → y ∈ Set.Icc (1/2 : ℝ) (25/2 : ℝ)) :=
by sorry

end g_range_l3572_357249


namespace oranges_per_box_l3572_357275

/-- Given a fruit farm that packs 2650 oranges into 265 boxes,
    prove that each box contains 10 oranges. -/
theorem oranges_per_box :
  let total_oranges : ℕ := 2650
  let total_boxes : ℕ := 265
  total_oranges / total_boxes = 10 := by
sorry

end oranges_per_box_l3572_357275


namespace dave_apps_left_l3572_357296

/-- The number of files Dave has left on his phone -/
def files_left : ℕ := 4

/-- The number of apps Dave has left on his phone -/
def apps_left : ℕ := files_left + 17

/-- Theorem: Dave has 21 apps left on his phone -/
theorem dave_apps_left : apps_left = 21 := by
  sorry

end dave_apps_left_l3572_357296


namespace sqrt_102_between_consecutive_integers_l3572_357209

theorem sqrt_102_between_consecutive_integers : ∃ n : ℕ, 
  (n : ℝ) < Real.sqrt 102 ∧ 
  Real.sqrt 102 < (n + 1 : ℝ) ∧ 
  n * (n + 1) = 110 := by
sorry

end sqrt_102_between_consecutive_integers_l3572_357209


namespace video_games_expense_is_11_l3572_357248

def total_allowance : ℚ := 60

def books_fraction : ℚ := 1/4
def snacks_fraction : ℚ := 1/6
def toys_fraction : ℚ := 2/5

def video_games_expense : ℚ := total_allowance - (books_fraction * total_allowance + snacks_fraction * total_allowance + toys_fraction * total_allowance)

theorem video_games_expense_is_11 : video_games_expense = 11 := by
  sorry

end video_games_expense_is_11_l3572_357248


namespace square_root_of_four_l3572_357267

theorem square_root_of_four :
  ∀ x : ℝ, x^2 = 4 ↔ x = 2 ∨ x = -2 := by sorry

end square_root_of_four_l3572_357267


namespace total_wheels_four_wheelers_l3572_357290

theorem total_wheels_four_wheelers (num_four_wheelers : ℕ) (wheels_per_four_wheeler : ℕ) :
  num_four_wheelers = 11 →
  wheels_per_four_wheeler = 4 →
  num_four_wheelers * wheels_per_four_wheeler = 44 :=
by sorry

end total_wheels_four_wheelers_l3572_357290


namespace math_reading_difference_l3572_357252

def reading_homework : ℕ := 4
def math_homework : ℕ := 7

theorem math_reading_difference : math_homework - reading_homework = 3 := by
  sorry

end math_reading_difference_l3572_357252


namespace inequality_range_l3572_357256

-- Define the inequality
def inequality (x a : ℝ) : Prop :=
  x^2 - (a + 1) * x + a ≤ 0

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ :=
  {x : ℝ | inequality x a}

-- Define the interval [-4, 3]
def interval : Set ℝ :=
  {x : ℝ | -4 ≤ x ∧ x ≤ 3}

-- Statement of the theorem
theorem inequality_range :
  (∀ a : ℝ, solution_set a ⊆ interval) →
  ∀ a : ℝ, -4 ≤ a ∧ a ≤ 3 :=
by sorry

end inequality_range_l3572_357256


namespace modulo_eleven_residue_l3572_357217

theorem modulo_eleven_residue : (310 + 6 * 45 + 8 * 154 + 3 * 23) % 11 = 0 := by
  sorry

end modulo_eleven_residue_l3572_357217


namespace mothers_age_five_times_daughters_l3572_357288

/-- 
Given:
- The mother's current age is 43 years.
- The daughter's current age is 11 years.

Prove that 3 years ago, the mother's age was five times her daughter's age.
-/
theorem mothers_age_five_times_daughters (mother_age : ℕ) (daughter_age : ℕ) :
  mother_age = 43 → daughter_age = 11 → 
  ∃ (x : ℕ), x = 3 ∧ (mother_age - x) = 5 * (daughter_age - x) :=
by sorry

end mothers_age_five_times_daughters_l3572_357288


namespace only_set_D_is_right_triangle_l3572_357213

-- Define a function to check if three numbers form a right triangle
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

-- Define the sets of line segments
def set_A : (ℝ × ℝ × ℝ) := (3, 5, 7)
def set_B : (ℝ × ℝ × ℝ) := (4, 6, 8)
def set_C : (ℝ × ℝ × ℝ) := (5, 7, 9)
def set_D : (ℝ × ℝ × ℝ) := (6, 8, 10)

-- Theorem stating that only set D forms a right triangle
theorem only_set_D_is_right_triangle :
  ¬(is_right_triangle set_A.1 set_A.2.1 set_A.2.2) ∧
  ¬(is_right_triangle set_B.1 set_B.2.1 set_B.2.2) ∧
  ¬(is_right_triangle set_C.1 set_C.2.1 set_C.2.2) ∧
  (is_right_triangle set_D.1 set_D.2.1 set_D.2.2) :=
by sorry


end only_set_D_is_right_triangle_l3572_357213


namespace nadia_mistakes_l3572_357263

/-- Calculates the number of mistakes made by a piano player given their error rate, playing speed, and duration of play. -/
def calculate_mistakes (mistakes_per_block : ℕ) (notes_per_block : ℕ) (notes_per_minute : ℕ) (minutes_played : ℕ) : ℕ :=
  let total_notes := notes_per_minute * minutes_played
  let num_blocks := total_notes / notes_per_block
  num_blocks * mistakes_per_block

/-- Theorem stating that under the given conditions, Nadia will make 36 mistakes on average when playing for 8 minutes. -/
theorem nadia_mistakes :
  calculate_mistakes 3 40 60 8 = 36 := by
  sorry

end nadia_mistakes_l3572_357263


namespace milk_sharing_problem_l3572_357273

/-- Given a total amount of milk and a difference between two people's consumption,
    calculate the amount consumed by the person drinking more. -/
def calculate_larger_share (total : ℕ) (difference : ℕ) : ℕ :=
  (total + difference) / 2

/-- Proof that given 2100 ml of milk shared between two people,
    where one drinks 200 ml more than the other,
    the person drinking more consumes 1150 ml. -/
theorem milk_sharing_problem :
  calculate_larger_share 2100 200 = 1150 := by
  sorry

end milk_sharing_problem_l3572_357273


namespace binomial_coefficient_equality_l3572_357295

theorem binomial_coefficient_equality (x : ℕ) : 
  (Nat.choose 5 x = Nat.choose 5 2) → (x = 2 ∨ x = 3) := by
  sorry

end binomial_coefficient_equality_l3572_357295


namespace arithmetic_geometric_general_term_l3572_357201

-- Define the arithmetic-geometric sequence
def arithmetic_geometric_seq (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ n : ℕ, a (n + 1) = r * a n

-- Define the conditions
def conditions (a : ℕ → ℝ) : Prop :=
  a 2 = 6 ∧ 6 * a 1 + a 3 = 30

-- Theorem statement
theorem arithmetic_geometric_general_term (a : ℕ → ℝ) :
  arithmetic_geometric_seq a → conditions a →
  (∀ n : ℕ, a n = 3 * 3^(n - 1)) ∨ (∀ n : ℕ, a n = 2 * 2^(n - 1)) :=
sorry

end arithmetic_geometric_general_term_l3572_357201


namespace sum_equals_product_integer_pairs_l3572_357223

theorem sum_equals_product_integer_pairs :
  ∀ x y : ℤ, x + y = x * y ↔ (x = 2 ∧ y = 2) ∨ (x = 0 ∧ y = 0) := by
sorry

end sum_equals_product_integer_pairs_l3572_357223


namespace polynomial_simplification_l3572_357299

theorem polynomial_simplification (x : ℝ) : 
  x * (4 * x^2 - 2) - 5 * (x^2 - 3*x + 5) = 4 * x^3 - 5 * x^2 + 13 * x - 25 := by
  sorry

end polynomial_simplification_l3572_357299


namespace prob_equal_prob_first_value_prob_second_value_l3572_357272

/-- Represents the number of classes -/
def total_classes : ℕ := 10

/-- Represents the specific class we're interested in (Class 5) -/
def target_class : ℕ := 5

/-- The probability of drawing the target class first -/
def prob_first : ℚ := 1 / total_classes

/-- The probability of drawing the target class second -/
def prob_second : ℚ := 1 / total_classes

/-- Theorem stating that the probabilities of drawing the target class first and second are equal -/
theorem prob_equal : prob_first = prob_second := by sorry

/-- Theorem stating that the probability of drawing the target class first is 1/10 -/
theorem prob_first_value : prob_first = 1 / 10 := by sorry

/-- Theorem stating that the probability of drawing the target class second is 1/10 -/
theorem prob_second_value : prob_second = 1 / 10 := by sorry

end prob_equal_prob_first_value_prob_second_value_l3572_357272


namespace sundae_price_l3572_357207

theorem sundae_price 
  (ice_cream_bars : ℕ) 
  (sundaes : ℕ) 
  (total_price : ℚ) 
  (ice_cream_price : ℚ) :
  ice_cream_bars = 225 →
  sundaes = 125 →
  total_price = 200 →
  ice_cream_price = 0.6 →
  (total_price - ice_cream_bars * ice_cream_price) / sundaes = 0.52 :=
by sorry

end sundae_price_l3572_357207
