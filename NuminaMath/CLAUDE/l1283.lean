import Mathlib

namespace NUMINAMATH_CALUDE_bisection_uses_all_structures_l1283_128358

/-- Represents the basic algorithm structures -/
inductive AlgorithmStructure
  | Sequential
  | Conditional
  | Loop

/-- Represents an algorithm -/
structure Algorithm where
  structures : List AlgorithmStructure

/-- The bisection method algorithm -/
def bisectionMethod : Algorithm := sorry

/-- Every algorithm has a sequential structure -/
axiom sequential_in_all (a : Algorithm) : 
  AlgorithmStructure.Sequential ∈ a.structures

/-- Loop structure implies conditional structure -/
axiom loop_implies_conditional (a : Algorithm) :
  AlgorithmStructure.Loop ∈ a.structures → 
  AlgorithmStructure.Conditional ∈ a.structures

/-- Bisection method involves a loop structure -/
axiom bisection_has_loop : 
  AlgorithmStructure.Loop ∈ bisectionMethod.structures

/-- Theorem: The bisection method algorithm requires all three basic structures -/
theorem bisection_uses_all_structures : 
  AlgorithmStructure.Sequential ∈ bisectionMethod.structures ∧
  AlgorithmStructure.Conditional ∈ bisectionMethod.structures ∧
  AlgorithmStructure.Loop ∈ bisectionMethod.structures := by
  sorry


end NUMINAMATH_CALUDE_bisection_uses_all_structures_l1283_128358


namespace NUMINAMATH_CALUDE_share_price_calculation_l1283_128393

/-- Proves the price of shares given dividend rate, face value, and return on investment -/
theorem share_price_calculation (dividend_rate : ℝ) (face_value : ℝ) (roi : ℝ) : 
  dividend_rate = 0.125 →
  face_value = 60 →
  roi = 0.25 →
  dividend_rate * face_value = roi * (face_value * dividend_rate / roi) := by
  sorry

#check share_price_calculation

end NUMINAMATH_CALUDE_share_price_calculation_l1283_128393


namespace NUMINAMATH_CALUDE_panda_babies_born_l1283_128351

/-- The number of panda babies born in a zoo with given conditions -/
theorem panda_babies_born (total_pandas : ℕ) (pregnancy_rate : ℚ) : 
  total_pandas = 16 →
  pregnancy_rate = 1/4 →
  (total_pandas / 2 : ℚ) * pregnancy_rate * 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_panda_babies_born_l1283_128351


namespace NUMINAMATH_CALUDE_salary_increase_after_three_years_l1283_128395

theorem salary_increase_after_three_years (annual_raise : Real) (years : Nat) : 
  annual_raise = 0.15 → years = 3 → 
  ((1 + annual_raise) ^ years - 1) * 100 = 52.0875 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_after_three_years_l1283_128395


namespace NUMINAMATH_CALUDE_n2o_molecular_weight_l1283_128399

/-- The atomic weight of nitrogen in g/mol -/
def nitrogen_weight : ℝ := 14.01

/-- The atomic weight of oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The number of nitrogen atoms in N2O -/
def n_nitrogen : ℕ := 2

/-- The number of oxygen atoms in N2O -/
def n_oxygen : ℕ := 1

/-- The number of moles of N2O -/
def n_moles : ℝ := 8

theorem n2o_molecular_weight :
  n_moles * (n_nitrogen * nitrogen_weight + n_oxygen * oxygen_weight) = 352.16 := by
  sorry

end NUMINAMATH_CALUDE_n2o_molecular_weight_l1283_128399


namespace NUMINAMATH_CALUDE_vegetable_planting_methods_l1283_128350

theorem vegetable_planting_methods :
  let total_vegetables : ℕ := 4
  let vegetables_to_choose : ℕ := 3
  let remaining_choices : ℕ := total_vegetables - 1  -- Cucumber is always chosen
  let remaining_to_choose : ℕ := vegetables_to_choose - 1
  let soil_types : ℕ := 3
  
  (remaining_choices.choose remaining_to_choose) * (vegetables_to_choose.factorial) = 18 :=
by sorry

end NUMINAMATH_CALUDE_vegetable_planting_methods_l1283_128350


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_l1283_128314

theorem absolute_value_equation_solution_difference : ∃ (x₁ x₂ : ℝ),
  (|x₁ - 4| = 15 ∧ |x₂ - 4| = 15 ∧ x₁ ≠ x₂) ∧ |x₁ - x₂| = 30 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_l1283_128314


namespace NUMINAMATH_CALUDE_similar_triangle_perimeter_l1283_128313

/-- Represents a triangle with side lengths -/
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ

/-- Checks if a triangle is isosceles -/
def Triangle.isIsosceles (t : Triangle) : Prop :=
  t.side1 = t.side2 ∨ t.side1 = t.side3 ∨ t.side2 = t.side3

/-- Calculates the perimeter of a triangle -/
def Triangle.perimeter (t : Triangle) : ℝ :=
  t.side1 + t.side2 + t.side3

/-- Checks if two triangles are similar -/
def areTrianglesSimilar (t1 t2 : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧
    t2.side1 = k * t1.side1 ∧
    t2.side2 = k * t1.side2 ∧
    t2.side3 = k * t1.side3

theorem similar_triangle_perimeter
  (small large : Triangle)
  (h_small_isosceles : small.isIsosceles)
  (h_small_sides : small.side1 = 12 ∧ small.side2 = 24)
  (h_similar : areTrianglesSimilar small large)
  (h_large_shortest : min large.side1 (min large.side2 large.side3) = 30) :
  large.perimeter = 150 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangle_perimeter_l1283_128313


namespace NUMINAMATH_CALUDE_square_difference_formula_l1283_128329

theorem square_difference_formula (x y A : ℝ) : 
  (3*x + 2*y)^2 = (3*x - 2*y)^2 + A → A = 24*x*y := by sorry

end NUMINAMATH_CALUDE_square_difference_formula_l1283_128329


namespace NUMINAMATH_CALUDE_fourth_term_coefficient_binomial_expansion_l1283_128312

theorem fourth_term_coefficient_binomial_expansion :
  let n : ℕ := 5
  let a : ℤ := 2
  let b : ℤ := -3
  let k : ℕ := 3  -- For the fourth term, we choose 3 from 5
  (n.choose k) * a^(n - k) * b^k = 720 :=
by sorry

end NUMINAMATH_CALUDE_fourth_term_coefficient_binomial_expansion_l1283_128312


namespace NUMINAMATH_CALUDE_solutions_rearrangements_l1283_128383

def word := "SOLUTIONS"

def vowels := ['O', 'I', 'U', 'O']
def consonants := ['S', 'L', 'T', 'N', 'S', 'S']

def vowel_arrangements := Nat.factorial 4 / Nat.factorial 2
def consonant_arrangements := Nat.factorial 6 / Nat.factorial 3

theorem solutions_rearrangements : 
  vowel_arrangements * consonant_arrangements = 1440 := by
  sorry

end NUMINAMATH_CALUDE_solutions_rearrangements_l1283_128383


namespace NUMINAMATH_CALUDE_square_area_13m_l1283_128390

/-- The area of a square with side length 13 meters is 169 square meters. -/
theorem square_area_13m (side_length : ℝ) (h : side_length = 13) :
  side_length * side_length = 169 := by
  sorry

end NUMINAMATH_CALUDE_square_area_13m_l1283_128390


namespace NUMINAMATH_CALUDE_box_depth_calculation_l1283_128301

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  depth : ℕ

/-- Represents a cube -/
structure Cube where
  sideLength : ℕ

def BoxDimensions.volume (b : BoxDimensions) : ℕ :=
  b.length * b.width * b.depth

def Cube.volume (c : Cube) : ℕ :=
  c.sideLength ^ 3

/-- The theorem to be proved -/
theorem box_depth_calculation (box : BoxDimensions) (cube : Cube) :
  box.length = 30 →
  box.width = 48 →
  (80 * cube.volume = box.volume) →
  (box.length % cube.sideLength = 0) →
  (box.width % cube.sideLength = 0) →
  (box.depth % cube.sideLength = 0) →
  box.depth = 12 := by
  sorry

end NUMINAMATH_CALUDE_box_depth_calculation_l1283_128301


namespace NUMINAMATH_CALUDE_unique_solution_g_equals_g_inv_l1283_128346

-- Define the function g
def g (x : ℝ) : ℝ := 5 * x - 12

-- Define the inverse function g⁻¹
noncomputable def g_inv (x : ℝ) : ℝ := (x + 12) / 5

-- Theorem statement
theorem unique_solution_g_equals_g_inv :
  ∃! x : ℝ, g x = g_inv x :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_g_equals_g_inv_l1283_128346


namespace NUMINAMATH_CALUDE_ab_value_l1283_128345

theorem ab_value (a b : ℝ) (h : (a + 3)^2 + (b - 3)^2 = 0) : a^b = -27 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l1283_128345


namespace NUMINAMATH_CALUDE_sports_club_non_athletic_parents_l1283_128318

/-- Represents a sports club with members and their parents' athletic status -/
structure SportsClub where
  total_members : ℕ
  athletic_dads : ℕ
  athletic_moms : ℕ
  both_athletic : ℕ
  no_dads : ℕ

/-- Calculates the number of members with non-athletic parents in a sports club -/
def members_with_non_athletic_parents (club : SportsClub) : ℕ :=
  club.total_members - (club.athletic_dads + club.athletic_moms - club.both_athletic - club.no_dads)

/-- Theorem stating the number of members with non-athletic parents in the given sports club -/
theorem sports_club_non_athletic_parents :
  let club : SportsClub := {
    total_members := 50,
    athletic_dads := 25,
    athletic_moms := 30,
    both_athletic := 10,
    no_dads := 5
  }
  members_with_non_athletic_parents club = 10 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_non_athletic_parents_l1283_128318


namespace NUMINAMATH_CALUDE_rhombus_sides_equal_l1283_128354

/-- A rhombus is a quadrilateral with all sides equal -/
structure Rhombus where
  sides : Fin 4 → ℝ
  is_quadrilateral : True
  all_sides_equal : ∀ (i j : Fin 4), sides i = sides j

/-- All four sides of a rhombus are equal -/
theorem rhombus_sides_equal (r : Rhombus) : 
  ∀ (i j : Fin 4), r.sides i = r.sides j := by
  sorry

end NUMINAMATH_CALUDE_rhombus_sides_equal_l1283_128354


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1283_128378

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ) (m n p : ℕ) 
  (h_arith : ArithmeticSequence a)
  (h_distinct : m ≠ n ∧ n ≠ p ∧ m ≠ p)
  (h_positive : 0 < m ∧ 0 < n ∧ 0 < p) :
  m * (a p - a n) + n * (a m - a p) + p * (a n - a m) = 0 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1283_128378


namespace NUMINAMATH_CALUDE_combine_like_terms_l1283_128315

theorem combine_like_terms (x y : ℝ) :
  -x^2 * y + 3/4 * x^2 * y = -(1/4) * x^2 * y := by
  sorry

end NUMINAMATH_CALUDE_combine_like_terms_l1283_128315


namespace NUMINAMATH_CALUDE_only_negative_two_less_than_negative_one_l1283_128317

theorem only_negative_two_less_than_negative_one : ∀ x : ℚ, 
  (x = 0 ∨ x = -1/2 ∨ x = 1 ∨ x = -2) → (x < -1 ↔ x = -2) :=
by
  sorry

end NUMINAMATH_CALUDE_only_negative_two_less_than_negative_one_l1283_128317


namespace NUMINAMATH_CALUDE_sum_of_two_before_last_l1283_128323

def arithmetic_sequence (a : ℕ → ℕ) (n : ℕ) : Prop :=
  ∀ i j : ℕ, i < j → j < n → a (j + 1) - a j = a (i + 1) - a i

theorem sum_of_two_before_last (a : ℕ → ℕ) :
  arithmetic_sequence a 7 →
  a 0 = 3 →
  a 1 = 8 →
  a 2 = 13 →
  a 6 = 33 →
  a 4 + a 5 = 51 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_before_last_l1283_128323


namespace NUMINAMATH_CALUDE_evaluate_expression_l1283_128352

theorem evaluate_expression : (827 * 827) - (826 * 828) + 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1283_128352


namespace NUMINAMATH_CALUDE_ones_digit_product_seven_consecutive_l1283_128385

theorem ones_digit_product_seven_consecutive (k : ℕ) (h1 : k > 0) (h2 : k % 5 = 1) : 
  (((k * (k + 1) * (k + 2) * (k + 3) * (k + 4) * (k + 5) * (k + 6)) % 10) = 0) := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_product_seven_consecutive_l1283_128385


namespace NUMINAMATH_CALUDE_student_marks_l1283_128353

theorem student_marks (M P C : ℕ) : 
  C = P + 20 →
  (M + C) / 2 = 25 →
  M + P = 30 :=
by sorry

end NUMINAMATH_CALUDE_student_marks_l1283_128353


namespace NUMINAMATH_CALUDE_intersection_x_sum_l1283_128336

theorem intersection_x_sum (x y : ℕ) : 
  (y ≡ 7 * x + 3 [ZMOD 20] ∧ y ≡ 13 * x + 18 [ZMOD 20]) → 
  x ≡ 15 [ZMOD 20] := by
  sorry

#check intersection_x_sum

end NUMINAMATH_CALUDE_intersection_x_sum_l1283_128336


namespace NUMINAMATH_CALUDE_inequality_proof_l1283_128334

theorem inequality_proof (x y : ℝ) (hx : x > Real.sqrt 2) (hy : y > Real.sqrt 2) :
  x^4 - x^3*y + x^2*y^2 - x*y^3 + y^4 > x^2 + y^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1283_128334


namespace NUMINAMATH_CALUDE_fescue_percentage_in_y_l1283_128340

/-- Represents the composition of a seed mixture -/
structure SeedMixture where
  ryegrass : ℝ
  bluegrass : ℝ
  fescue : ℝ

/-- The combined mixture of X and Y -/
def combinedMixture (x y : SeedMixture) (xProportion : ℝ) : SeedMixture :=
  { ryegrass := x.ryegrass * xProportion + y.ryegrass * (1 - xProportion)
  , bluegrass := x.bluegrass * xProportion + y.bluegrass * (1 - xProportion)
  , fescue := x.fescue * xProportion + y.fescue * (1 - xProportion) }

/-- The theorem stating the percentage of fescue in mixture Y -/
theorem fescue_percentage_in_y
  (x : SeedMixture)
  (y : SeedMixture)
  (h1 : x.ryegrass = 0.4)
  (h2 : x.bluegrass = 0.6)
  (h3 : x.fescue = 0)
  (h4 : y.ryegrass = 0.25)
  (h5 : x.ryegrass + x.bluegrass + x.fescue = 1)
  (h6 : y.ryegrass + y.bluegrass + y.fescue = 1)
  (h7 : (combinedMixture x y 0.4667).ryegrass = 0.32) :
  y.fescue = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_fescue_percentage_in_y_l1283_128340


namespace NUMINAMATH_CALUDE_liters_to_pints_conversion_l1283_128338

/-- Given that 0.75 liters is approximately 1.575 pints, prove that 3 liters is equal to 6.3 pints. -/
theorem liters_to_pints_conversion (liter_to_pint : ℝ → ℝ) 
  (h : liter_to_pint 0.75 = 1.575) : liter_to_pint 3 = 6.3 := by
  sorry

end NUMINAMATH_CALUDE_liters_to_pints_conversion_l1283_128338


namespace NUMINAMATH_CALUDE_square_of_binomial_constant_l1283_128347

theorem square_of_binomial_constant (m : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, 4*x^2 + 16*x + m = (a*x + b)^2) → m = 16 := by
sorry

end NUMINAMATH_CALUDE_square_of_binomial_constant_l1283_128347


namespace NUMINAMATH_CALUDE_prove_my_current_age_l1283_128368

/-- The age at which my dog was born -/
def age_when_dog_born : ℕ := 15

/-- The age my dog will be in two years -/
def dog_age_in_two_years : ℕ := 4

/-- My current age -/
def my_current_age : ℕ := age_when_dog_born + (dog_age_in_two_years - 2)

theorem prove_my_current_age : my_current_age = 17 := by
  sorry

end NUMINAMATH_CALUDE_prove_my_current_age_l1283_128368


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1283_128342

theorem complex_equation_solution (x : ℝ) : 
  (x - 2 * Complex.I) * Complex.I = 2 + Complex.I → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1283_128342


namespace NUMINAMATH_CALUDE_intersection_implies_a_equals_one_l1283_128309

theorem intersection_implies_a_equals_one (a : ℝ) : 
  let A : Set ℝ := {-1, 1, 3}
  let B : Set ℝ := {a + 2, a^2 + 4}
  A ∩ B = {3} → a = 1 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_a_equals_one_l1283_128309


namespace NUMINAMATH_CALUDE_amanda_ticket_sales_l1283_128303

/-- The number of tickets Amanda needs to sell on the third day -/
def tickets_to_sell_on_third_day (total_tickets : ℕ) (friends : ℕ) (tickets_per_friend : ℕ) (second_day_sales : ℕ) : ℕ :=
  total_tickets - (friends * tickets_per_friend + second_day_sales)

/-- Theorem stating the number of tickets Amanda needs to sell on the third day -/
theorem amanda_ticket_sales : tickets_to_sell_on_third_day 80 5 4 32 = 28 := by
  sorry

end NUMINAMATH_CALUDE_amanda_ticket_sales_l1283_128303


namespace NUMINAMATH_CALUDE_inverse_of_proposition_l1283_128302

-- Define the original proposition
def original_proposition (x : ℝ) : Prop := x < 0 → x^2 > 0

-- Define the inverse proposition
def inverse_proposition (x : ℝ) : Prop := x^2 > 0 → x < 0

-- Theorem stating that inverse_proposition is the inverse of original_proposition
theorem inverse_of_proposition :
  (∀ x : ℝ, original_proposition x) ↔ (∀ x : ℝ, inverse_proposition x) :=
sorry

end NUMINAMATH_CALUDE_inverse_of_proposition_l1283_128302


namespace NUMINAMATH_CALUDE_three_squares_representation_l1283_128335

theorem three_squares_representation (N : ℕ) :
  (∃ a b c : ℤ, N = (3*a)^2 + (3*b)^2 + (3*c)^2) →
  (∃ x y z : ℤ, N = x^2 + y^2 + z^2 ∧ ¬(3 ∣ x) ∧ ¬(3 ∣ y) ∧ ¬(3 ∣ z)) :=
by sorry

end NUMINAMATH_CALUDE_three_squares_representation_l1283_128335


namespace NUMINAMATH_CALUDE_exam_maximum_marks_l1283_128344

theorem exam_maximum_marks 
  (passing_percentage : ℝ)
  (student_score : ℕ)
  (failing_margin : ℕ)
  (h1 : passing_percentage = 0.45)
  (h2 : student_score = 40)
  (h3 : failing_margin = 40) :
  ∃ (max_marks : ℕ), max_marks = 180 ∧ 
    (passing_percentage * max_marks : ℝ) = (student_score + failing_margin) :=
by sorry

end NUMINAMATH_CALUDE_exam_maximum_marks_l1283_128344


namespace NUMINAMATH_CALUDE_f_no_extreme_points_l1283_128381

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*x^2 + 3*x - a

-- Theorem stating that f has no extreme points for any real a
theorem f_no_extreme_points (a : ℝ) : 
  ∀ x : ℝ, ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), f a y ≠ f a x ∨ (f a y < f a x ∧ y < x) ∨ (f a y > f a x ∧ y > x) :=
sorry

end NUMINAMATH_CALUDE_f_no_extreme_points_l1283_128381


namespace NUMINAMATH_CALUDE_algebraic_expression_simplification_l1283_128386

theorem algebraic_expression_simplification :
  let x : ℝ := (Real.sqrt 3) / 2 + 1 / 2
  ((1 / x + (x + 1) / x) / ((x + 2) / (x^2 + x))) = ((Real.sqrt 3) + 3) / 2 :=
by sorry

end NUMINAMATH_CALUDE_algebraic_expression_simplification_l1283_128386


namespace NUMINAMATH_CALUDE_factor_expression_l1283_128363

theorem factor_expression (y : ℝ) : 3 * y^3 - 75 * y = 3 * y * (y + 5) * (y - 5) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1283_128363


namespace NUMINAMATH_CALUDE_percentage_increase_l1283_128311

theorem percentage_increase (original : ℝ) (new : ℝ) :
  original = 30 ∧ new = 40 →
  (new - original) / original * 100 = 100 / 3 :=
by sorry

end NUMINAMATH_CALUDE_percentage_increase_l1283_128311


namespace NUMINAMATH_CALUDE_down_payment_calculation_l1283_128376

/-- Proves that the down payment is $4 given the specified conditions -/
theorem down_payment_calculation (purchase_price : ℝ) (monthly_payment : ℝ) 
  (num_payments : ℕ) (interest_rate : ℝ) (down_payment : ℝ) : 
  purchase_price = 112 →
  monthly_payment = 10 →
  num_payments = 12 →
  interest_rate = 0.10714285714285714 →
  down_payment + num_payments * monthly_payment = purchase_price * (1 + interest_rate) →
  down_payment = 4 := by
sorry

end NUMINAMATH_CALUDE_down_payment_calculation_l1283_128376


namespace NUMINAMATH_CALUDE_unique_positive_integer_solution_l1283_128321

theorem unique_positive_integer_solution :
  ∃! (x y z : ℕ+), 
    (3 * x.val - 4 * y.val + 5 * z.val = 10) ∧
    (7 * y.val + 8 * x.val - 3 * z.val = 13) ∧
    (x.val = 1 ∧ y.val = 2 ∧ z.val = 3) :=
by sorry

end NUMINAMATH_CALUDE_unique_positive_integer_solution_l1283_128321


namespace NUMINAMATH_CALUDE_circle_area_l1283_128320

theorem circle_area (x y : ℝ) : 
  (4 * x^2 + 4 * y^2 - 8 * x + 24 * y + 60 = 0) → 
  (∃ (center_x center_y radius : ℝ), 
    ((x - center_x)^2 + (y - center_y)^2 = radius^2) ∧ 
    (π * radius^2 = 5 * π)) := by
  sorry

end NUMINAMATH_CALUDE_circle_area_l1283_128320


namespace NUMINAMATH_CALUDE_perimeter_ratio_specific_triangle_l1283_128369

/-- Right triangle DEF with altitude FG and external point J -/
structure RightTriangleWithAltitude where
  /-- Length of side DF -/
  df : ℝ
  /-- Length of side EF -/
  ef : ℝ
  /-- Length of hypotenuse DE -/
  de : ℝ
  /-- Length of altitude FG -/
  fg : ℝ
  /-- Length of tangent from J to circle with diameter FG -/
  tj : ℝ
  /-- de² = df² + ef² (Pythagorean theorem) -/
  pythagoras : de^2 = df^2 + ef^2
  /-- fg² = df * ef (geometric mean property of altitude) -/
  altitude_property : fg^2 = df * ef
  /-- tj² = df * (de - df) (tangent-secant theorem) -/
  tangent_secant : tj^2 = df * (de - df)

/-- Theorem: Perimeter ratio for specific right triangle -/
theorem perimeter_ratio_specific_triangle :
  ∀ t : RightTriangleWithAltitude,
  t.df = 9 →
  t.ef = 40 →
  (t.de + 2 * t.tj) / t.de = 49 / 41 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_ratio_specific_triangle_l1283_128369


namespace NUMINAMATH_CALUDE_intersection_count_l1283_128372

/-- The number of distinct intersection points between two algebraic curves -/
def num_intersections (f g : ℝ → ℝ → ℝ) : ℕ :=
  sorry

/-- First curve equation -/
def curve1 (x y : ℝ) : ℝ :=
  (x - y + 3) * (3 * x + y - 7)

/-- Second curve equation -/
def curve2 (x y : ℝ) : ℝ :=
  (x + y - 3) * (2 * x - 5 * y + 12)

theorem intersection_count :
  num_intersections curve1 curve2 = 4 := by sorry

end NUMINAMATH_CALUDE_intersection_count_l1283_128372


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l1283_128379

theorem max_sum_of_squares (a b c d : ℝ) : 
  a + b = 18 →
  a * b + c + d = 91 →
  a * d + b * c = 187 →
  c * d = 105 →
  a^2 + b^2 + c^2 + d^2 ≤ 107 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l1283_128379


namespace NUMINAMATH_CALUDE_expression_simplification_l1283_128377

theorem expression_simplification (b : ℝ) (h : b ≠ -1) :
  1 - (1 / (1 - b / (1 + b))) = -b :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l1283_128377


namespace NUMINAMATH_CALUDE_fly_path_distance_l1283_128308

theorem fly_path_distance (radius : ℝ) (third_segment : ℝ) :
  radius = 60 ∧ third_segment = 90 →
  ∃ (second_segment : ℝ),
    second_segment^2 + third_segment^2 = (2 * radius)^2 ∧
    (2 * radius) + third_segment + second_segment = 120 + 90 + 30 * Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_fly_path_distance_l1283_128308


namespace NUMINAMATH_CALUDE_remainder_equality_l1283_128356

theorem remainder_equality (a b k : ℤ) (h : k ∣ (a - b)) : a % k = b % k := by
  sorry

end NUMINAMATH_CALUDE_remainder_equality_l1283_128356


namespace NUMINAMATH_CALUDE_least_common_period_l1283_128339

-- Define the property for function f
def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 5) + f (x - 5) = f x

-- Define the concept of a period for a function
def is_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

-- State the theorem
theorem least_common_period :
  ∃ p : ℝ, p > 0 ∧
    (∀ f : ℝ → ℝ, satisfies_condition f → is_period f p) ∧
    (∀ q : ℝ, q > 0 → (∀ f : ℝ → ℝ, satisfies_condition f → is_period f q) → p ≤ q) ∧
    p = 30 :=
  sorry

end NUMINAMATH_CALUDE_least_common_period_l1283_128339


namespace NUMINAMATH_CALUDE_circle_equation_l1283_128341

/-- A circle with center on the x-axis, radius √2, passing through (-2, 1) -/
structure CircleOnXAxis where
  center : ℝ × ℝ
  radius : ℝ
  passes_through : ℝ × ℝ
  center_on_x_axis : center.2 = 0
  radius_is_sqrt2 : radius = Real.sqrt 2
  point_on_circle : (center.1 + 2)^2 + (center.2 - 1)^2 = radius^2

/-- The equation of the circle is either (x+1)^2 + y^2 = 2 or (x+3)^2 + y^2 = 2 -/
theorem circle_equation (c : CircleOnXAxis) :
  (∀ x y : ℝ, (x + 1)^2 + y^2 = 2 ↔ (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) ∨
  (∀ x y : ℝ, (x + 3)^2 + y^2 = 2 ↔ (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l1283_128341


namespace NUMINAMATH_CALUDE_area_of_arcsin_cos_l1283_128300

open Set
open MeasureTheory
open Interval

noncomputable def f (x : ℝ) := Real.arcsin (Real.cos x)

theorem area_of_arcsin_cos (a b : ℝ) (h : 0 ≤ a ∧ b = 2 * Real.pi) :
  (∫ x in a..b, |f x| ) = Real.pi^2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_area_of_arcsin_cos_l1283_128300


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_3_seconds_l1283_128333

-- Define the displacement function
def s (t : ℝ) : ℝ := 4 - 2*t + t^2

-- Define the velocity function (derivative of s)
def v (t : ℝ) : ℝ := 2*t - 2

-- Theorem statement
theorem instantaneous_velocity_at_3_seconds :
  v 3 = 4 := by sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_3_seconds_l1283_128333


namespace NUMINAMATH_CALUDE_problem_solution_l1283_128307

theorem problem_solution (x y z a b : ℝ) 
  (h1 : (x + y) / 2 = (z + x) / 3)
  (h2 : (x + y) / 2 = (y + z) / 4)
  (h3 : x + y + z = 36 * a)
  (h4 : b = x + y) :
  b = 16 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1283_128307


namespace NUMINAMATH_CALUDE_probability_two_specific_people_obtain_items_l1283_128359

-- Define the number of people and items
def num_people : ℕ := 4
def num_items : ℕ := 3

-- Define the probability function
noncomputable def probability_both_obtain (n_people n_items : ℕ) : ℚ :=
  (n_items.choose 2 * (n_people - 2).choose 1) / n_people.choose n_items

-- State the theorem
theorem probability_two_specific_people_obtain_items :
  probability_both_obtain num_people num_items = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_specific_people_obtain_items_l1283_128359


namespace NUMINAMATH_CALUDE_num_tangent_lines_specific_case_l1283_128306

/-- Two circles are internally tangent if the distance between their centers
    equals the absolute difference of their radii. -/
def internally_tangent (r₁ r₂ d : ℝ) : Prop :=
  d = |r₁ - r₂|

/-- The number of common tangent lines for two internally tangent circles is 1. -/
def num_common_tangents_internal : ℕ := 1

/-- Theorem: For two circles with radii 4 and 5, and distance between centers 3,
    the number of lines simultaneously tangent to both circles is 1. -/
theorem num_tangent_lines_specific_case :
  let r₁ : ℝ := 4
  let r₂ : ℝ := 5
  let d : ℝ := 3
  internally_tangent r₁ r₂ d →
  (num_common_tangents_internal : ℕ) = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_num_tangent_lines_specific_case_l1283_128306


namespace NUMINAMATH_CALUDE_loss_per_metre_is_five_l1283_128337

/-- Calculates the loss per metre of cloth given the total metres sold, 
    total selling price, and cost price per metre. -/
def loss_per_metre (total_metres : ℕ) (total_selling_price : ℕ) (cost_price_per_metre : ℕ) : ℕ :=
  let total_cost_price := total_metres * cost_price_per_metre
  let total_loss := total_cost_price - total_selling_price
  total_loss / total_metres

/-- Proves that the loss per metre is 5 given the specified conditions. -/
theorem loss_per_metre_is_five : 
  loss_per_metre 500 18000 41 = 5 := by
  sorry

#eval loss_per_metre 500 18000 41

end NUMINAMATH_CALUDE_loss_per_metre_is_five_l1283_128337


namespace NUMINAMATH_CALUDE_sylvester_theorem_l1283_128326

-- Define coprimality
def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Define the theorem
theorem sylvester_theorem (a b : ℕ) (h : coprime a b) :
  -- Part 1: Unique solution in the strip
  (∀ c : ℕ, ∃! p : ℕ × ℕ, p.1 < b ∧ a * p.1 + b * p.2 = c) ∧
  -- Part 2: Largest value without non-negative solutions
  (∀ c : ℕ, c > a * b - a - b → ∃ x y : ℕ, a * x + b * y = c) ∧
  (¬∃ x y : ℕ, a * x + b * y = a * b - a - b) := by
  sorry

end NUMINAMATH_CALUDE_sylvester_theorem_l1283_128326


namespace NUMINAMATH_CALUDE_factor_expression_l1283_128392

theorem factor_expression (z : ℝ) :
  75 * z^24 + 225 * z^48 = 75 * z^24 * (1 + 3 * z^24) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1283_128392


namespace NUMINAMATH_CALUDE_line_equation_l1283_128387

/-- A line with slope -2 and sum of x and y intercepts equal to 12 has the general equation 2x + y - 8 = 0 -/
theorem line_equation (l : Set (ℝ × ℝ)) (slope : ℝ) (intercept_sum : ℝ) : 
  slope = -2 →
  intercept_sum = 12 →
  ∃ (a b c : ℝ), a = 2 ∧ b = 1 ∧ c = -8 ∧
  l = {(x, y) | a * x + b * y + c = 0} :=
by sorry

end NUMINAMATH_CALUDE_line_equation_l1283_128387


namespace NUMINAMATH_CALUDE_right_triangle_legs_l1283_128397

/-- A right-angled triangle with an inscribed circle -/
structure RightTriangleWithInscribedCircle where
  /-- Length of the first segment of the hypotenuse -/
  a : ℝ
  /-- Length of the second segment of the hypotenuse -/
  b : ℝ
  /-- The first leg of the triangle -/
  leg1 : ℝ
  /-- The second leg of the triangle -/
  leg2 : ℝ
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The first segment plus radius equals the first leg -/
  h1 : a + r = leg1
  /-- The second segment plus radius equals the second leg -/
  h2 : b + r = leg2
  /-- The Pythagorean theorem holds -/
  pythagoras : leg1^2 + leg2^2 = (a + b)^2

/-- The main theorem -/
theorem right_triangle_legs (t : RightTriangleWithInscribedCircle)
  (ha : t.a = 5) (hb : t.b = 12) : t.leg1 = 8 ∧ t.leg2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_legs_l1283_128397


namespace NUMINAMATH_CALUDE_jenny_ate_65_squares_l1283_128367

/-- The number of chocolate squares Mike ate -/
def mike_squares : ℕ := 20

/-- The number of chocolate squares Jenny ate -/
def jenny_squares : ℕ := 3 * mike_squares + 5

/-- Theorem stating that Jenny ate 65 chocolate squares -/
theorem jenny_ate_65_squares : jenny_squares = 65 := by
  sorry

end NUMINAMATH_CALUDE_jenny_ate_65_squares_l1283_128367


namespace NUMINAMATH_CALUDE_total_dreams_calculation_l1283_128382

/-- The number of days in a year -/
def daysInYear : ℕ := 365

/-- The number of dreams per day in the current year -/
def dreamsPerDay : ℕ := 4

/-- The number of dreams in the current year -/
def dreamsThisYear : ℕ := dreamsPerDay * daysInYear

/-- The number of dreams in the previous year -/
def dreamsLastYear : ℕ := 2 * dreamsThisYear

/-- The total number of dreams over two years -/
def totalDreams : ℕ := dreamsLastYear + dreamsThisYear

theorem total_dreams_calculation :
  totalDreams = 4380 := by sorry

end NUMINAMATH_CALUDE_total_dreams_calculation_l1283_128382


namespace NUMINAMATH_CALUDE_sum_of_products_nonzero_l1283_128362

/-- A 25x25 matrix with entries either 1 or -1 -/
def SignMatrix := Matrix (Fin 25) (Fin 25) Int

/-- Predicate to check if a matrix is a valid SignMatrix -/
def isValidSignMatrix (M : SignMatrix) : Prop :=
  ∀ i j, M i j = 1 ∨ M i j = -1

/-- Product of elements in a row -/
def rowProduct (M : SignMatrix) (i : Fin 25) : Int :=
  (List.range 25).foldl (fun acc j => acc * M i j) 1

/-- Product of elements in a column -/
def colProduct (M : SignMatrix) (j : Fin 25) : Int :=
  (List.range 25).foldl (fun acc i => acc * M i j) 1

/-- Sum of all row and column products -/
def sumOfProducts (M : SignMatrix) : Int :=
  (List.range 25).foldl (fun acc i => acc + rowProduct M i) 0 +
  (List.range 25).foldl (fun acc j => acc + colProduct M j) 0

theorem sum_of_products_nonzero (M : SignMatrix) (h : isValidSignMatrix M) :
  sumOfProducts M ≠ 0 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_products_nonzero_l1283_128362


namespace NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l1283_128319

theorem polar_to_rectangular_conversion :
  let r : ℝ := 3
  let θ : ℝ := 3 * π / 4
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x, y) = (-3 / Real.sqrt 2, 3 / Real.sqrt 2) := by sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l1283_128319


namespace NUMINAMATH_CALUDE_shirts_washed_l1283_128380

theorem shirts_washed (short_sleeve : ℕ) (long_sleeve : ℕ) (unwashed : ℕ) : 
  short_sleeve = 9 → long_sleeve = 21 → unwashed = 1 →
  short_sleeve + long_sleeve - unwashed = 29 := by
sorry

end NUMINAMATH_CALUDE_shirts_washed_l1283_128380


namespace NUMINAMATH_CALUDE_coin_flip_probability_l1283_128371

theorem coin_flip_probability :
  let n : ℕ := 12  -- number of coin flips
  let k : ℕ := 9   -- number of heads we want
  let p : ℚ := 1/2 -- probability of heads for a fair coin
  Nat.choose n k * p^k * (1-p)^(n-k) = 55/1024 := by
sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l1283_128371


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l1283_128384

/-- A line passing through a point and parallel to another line -/
theorem line_through_point_parallel_to_line :
  let point : ℝ × ℝ := (2, 1)
  let parallel_line (x y : ℝ) := 2 * x - 3 * y + 1 = 0
  let target_line (x y : ℝ) := 2 * x - 3 * y - 1 = 0
  (∀ x y : ℝ, parallel_line x y ↔ y = 2/3 * x + 1/3) →
  (target_line point.1 point.2) ∧
  (∀ x y : ℝ, target_line x y ↔ y = 2/3 * x + 1/3) :=
by sorry

end NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l1283_128384


namespace NUMINAMATH_CALUDE_prob_black_second_draw_l1283_128373

/-- Represents the color of a ball -/
inductive Color
| Red
| Black

/-- Represents the state of the box -/
structure Box :=
  (red : ℕ)
  (black : ℕ)

/-- Calculates the probability of drawing a black ball -/
def prob_black (b : Box) : ℚ :=
  b.black / (b.red + b.black)

/-- Adds balls to the box based on the color drawn -/
def add_balls (b : Box) (c : Color) : Box :=
  match c with
  | Color.Red => Box.mk (b.red + 3) b.black
  | Color.Black => Box.mk b.red (b.black + 3)

/-- The main theorem to prove -/
theorem prob_black_second_draw (initial_box : Box) 
  (h1 : initial_box.red = 4)
  (h2 : initial_box.black = 5) : 
  (prob_black initial_box * prob_black (add_balls initial_box Color.Black) +
   (1 - prob_black initial_box) * prob_black (add_balls initial_box Color.Red)) = 5/9 :=
by sorry

end NUMINAMATH_CALUDE_prob_black_second_draw_l1283_128373


namespace NUMINAMATH_CALUDE_task_completion_time_l1283_128375

/-- The time required for Sumin and Junwoo to complete a task together, given their individual work rates -/
theorem task_completion_time (sumin_rate junwoo_rate : ℚ) 
  (h_sumin : sumin_rate = 1 / 10)
  (h_junwoo : junwoo_rate = 1 / 15) :
  (1 : ℚ) / (sumin_rate + junwoo_rate) = 6 := by
  sorry

end NUMINAMATH_CALUDE_task_completion_time_l1283_128375


namespace NUMINAMATH_CALUDE_feeding_sequences_count_l1283_128357

/-- Represents the number of distinct pairs of animals -/
def num_pairs : ℕ := 6

/-- Calculates the number of ways to feed animals given the conditions -/
def feeding_sequences : ℕ :=
  1 * num_pairs.factorial * (num_pairs - 1) * (num_pairs - 2) * (num_pairs - 3) * (num_pairs - 4) * (num_pairs - 5)

/-- Theorem stating that the number of feeding sequences is 17280 -/
theorem feeding_sequences_count : feeding_sequences = 17280 := by
  sorry

end NUMINAMATH_CALUDE_feeding_sequences_count_l1283_128357


namespace NUMINAMATH_CALUDE_solution_Y_initial_weight_l1283_128310

/-- Represents the composition and transformation of a solution --/
structure Solution where
  initialWeight : ℝ
  liquidXPercentage : ℝ
  waterPercentage : ℝ
  evaporatedWater : ℝ
  addedSolutionWeight : ℝ
  newLiquidXPercentage : ℝ

/-- Theorem stating the initial weight of solution Y given the conditions --/
theorem solution_Y_initial_weight (s : Solution) 
  (h1 : s.liquidXPercentage = 0.30)
  (h2 : s.waterPercentage = 0.70)
  (h3 : s.liquidXPercentage + s.waterPercentage = 1)
  (h4 : s.evaporatedWater = 2)
  (h5 : s.addedSolutionWeight = 2)
  (h6 : s.newLiquidXPercentage = 0.36)
  (h7 : s.newLiquidXPercentage * s.initialWeight = 
        s.liquidXPercentage * s.initialWeight + 
        s.liquidXPercentage * s.addedSolutionWeight) :
  s.initialWeight = 10 := by
  sorry


end NUMINAMATH_CALUDE_solution_Y_initial_weight_l1283_128310


namespace NUMINAMATH_CALUDE_clara_climbs_96_blocks_l1283_128388

/-- The number of stone blocks Clara climbs past in the historical tower -/
def total_blocks (levels : ℕ) (steps_per_level : ℕ) (blocks_per_step : ℕ) : ℕ :=
  levels * steps_per_level * blocks_per_step

/-- Theorem stating that Clara climbs past 96 blocks of stone -/
theorem clara_climbs_96_blocks :
  total_blocks 4 8 3 = 96 := by
  sorry

end NUMINAMATH_CALUDE_clara_climbs_96_blocks_l1283_128388


namespace NUMINAMATH_CALUDE_complex_arithmetic_result_l1283_128396

def A : ℂ := 3 - 4 * Complex.I
def M : ℂ := -3 + 2 * Complex.I
def S : ℂ := 2 * Complex.I
def P : ℂ := -1

theorem complex_arithmetic_result : A - M + S + P = 5 - 4 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_result_l1283_128396


namespace NUMINAMATH_CALUDE_inequality_proof_l1283_128324

theorem inequality_proof (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  Real.exp a - 1 > a ∧ a > a ^ Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1283_128324


namespace NUMINAMATH_CALUDE_gold_coin_percentage_is_48_percent_l1283_128365

/-- Represents the composition of objects in an urn --/
structure UrnComposition where
  bead_percentage : ℝ
  silver_coin_percentage : ℝ
  gold_coin_percentage : ℝ

/-- Calculates the percentage of gold coins in the urn --/
def gold_coin_percentage (u : UrnComposition) : ℝ :=
  u.gold_coin_percentage

/-- The urn composition satisfies the given conditions --/
def valid_urn_composition (u : UrnComposition) : Prop :=
  u.bead_percentage = 0.2 ∧
  u.silver_coin_percentage + u.gold_coin_percentage = 0.8 ∧
  u.silver_coin_percentage = 0.4 * (u.silver_coin_percentage + u.gold_coin_percentage)

theorem gold_coin_percentage_is_48_percent (u : UrnComposition) 
  (h : valid_urn_composition u) : gold_coin_percentage u = 0.48 := by
  sorry

end NUMINAMATH_CALUDE_gold_coin_percentage_is_48_percent_l1283_128365


namespace NUMINAMATH_CALUDE_vanessa_score_is_40_5_l1283_128398

/-- Calculates Vanessa's score in a basketball game. -/
def vanessaScore (totalTeamScore : ℝ) (numPlayers : ℕ) (otherPlayersAverage : ℝ) : ℝ :=
  totalTeamScore - (otherPlayersAverage * (numPlayers - 1 : ℝ))

/-- Proves that Vanessa's score is 40.5 points given the conditions of the game. -/
theorem vanessa_score_is_40_5 :
  vanessaScore 72 8 4.5 = 40.5 := by
  sorry

end NUMINAMATH_CALUDE_vanessa_score_is_40_5_l1283_128398


namespace NUMINAMATH_CALUDE_shawnas_workout_goal_l1283_128316

/-- Shawna's workout goal in situps -/
def workout_goal : ℕ := sorry

/-- Number of situps Shawna did on Monday -/
def monday_situps : ℕ := 12

/-- Number of situps Shawna did on Tuesday -/
def tuesday_situps : ℕ := 19

/-- Number of situps Shawna needs to do on Wednesday to meet her goal -/
def wednesday_situps : ℕ := 59

/-- Theorem stating that Shawna's workout goal is 90 situps -/
theorem shawnas_workout_goal :
  workout_goal = monday_situps + tuesday_situps + wednesday_situps ∧
  workout_goal = 90 := by sorry

end NUMINAMATH_CALUDE_shawnas_workout_goal_l1283_128316


namespace NUMINAMATH_CALUDE_volunteer_group_selection_l1283_128360

def class_size : ℕ := 4
def total_classes : ℕ := 4
def group_size : ℕ := class_size * total_classes
def selection_size : ℕ := 3

def select_committee (n k : ℕ) : ℕ := Nat.choose n k

theorem volunteer_group_selection :
  let with_class3 := select_committee class_size 1 * select_committee (group_size - class_size) (selection_size - 1)
  let without_class3 := select_committee (group_size - class_size) selection_size - 
                        (total_classes - 1) * select_committee class_size selection_size
  with_class3 + without_class3 = 472 := by sorry

end NUMINAMATH_CALUDE_volunteer_group_selection_l1283_128360


namespace NUMINAMATH_CALUDE_distance_to_circle_center_l1283_128355

/-- The distance from a point on y = 2x to the center of (x-8)^2 + (y-1)^2 = 2,
    given symmetric tangents -/
theorem distance_to_circle_center (P : ℝ × ℝ) : 
  (∃ t : ℝ, P.1 = t ∧ P.2 = 2*t) →  -- P is on the line y = 2x
  (∃ l₁ l₂ : ℝ × ℝ → Prop,  -- l₁ and l₂ are tangent lines
    (∀ Q : ℝ × ℝ, l₁ Q → (Q.1 - 8)^2 + (Q.2 - 1)^2 = 2) ∧
    (∀ Q : ℝ × ℝ, l₂ Q → (Q.1 - 8)^2 + (Q.2 - 1)^2 = 2) ∧
    l₁ P ∧ l₂ P ∧
    (∀ Q : ℝ × ℝ, l₁ Q ↔ l₂ (2*P.1 - Q.1, 2*P.2 - Q.2))) →  -- l₁ and l₂ are symmetric about y = 2x
  Real.sqrt ((P.1 - 8)^2 + (P.2 - 1)^2) = 3 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_distance_to_circle_center_l1283_128355


namespace NUMINAMATH_CALUDE_tractor_finance_l1283_128349

/-- Calculates the total amount financed given monthly payment and number of years -/
def total_financed (monthly_payment : ℚ) (years : ℕ) : ℚ :=
  monthly_payment * (years * 12)

/-- Proves that financing $150 per month for 5 years results in a total of $9000 -/
theorem tractor_finance : total_financed 150 5 = 9000 := by
  sorry

end NUMINAMATH_CALUDE_tractor_finance_l1283_128349


namespace NUMINAMATH_CALUDE_point_C_x_value_l1283_128348

def A : ℝ × ℝ := (-1, -2)
def B : ℝ × ℝ := (4, 8)
def C (x : ℝ) : ℝ × ℝ := (5, x)

def collinear (p q r : ℝ × ℝ) : Prop :=
  (r.2 - q.2) * (q.1 - p.1) = (q.2 - p.2) * (r.1 - q.1)

theorem point_C_x_value :
  ∀ x : ℝ, collinear A B (C x) → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_point_C_x_value_l1283_128348


namespace NUMINAMATH_CALUDE_max_altitude_triangle_ABC_l1283_128343

/-- Given a triangle ABC with the specified conditions, the maximum altitude on side BC is √3 + 1 -/
theorem max_altitude_triangle_ABC (A B C : Real) (h1 : 3 * (Real.sin B ^ 2 + Real.sin C ^ 2 - Real.sin A ^ 2) = 2 * Real.sqrt 3 * Real.sin B * Real.sin C) 
  (h2 : (1 / 2) * Real.sin A * (Real.sin B / Real.sin A) * (Real.sin C / Real.sin A) = Real.sqrt 6 + Real.sqrt 2) :
  ∃ (h : Real), h ≤ Real.sqrt 3 + 1 ∧ 
    ∀ (h' : Real), (∃ (a b c : Real), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
      3 * (Real.sin (b / a) ^ 2 + Real.sin (c / a) ^ 2 - Real.sin 1 ^ 2) = 2 * Real.sqrt 3 * Real.sin (b / a) * Real.sin (c / a) ∧
      (1 / 2) * a * b * Real.sin (c / a) = Real.sqrt 6 + Real.sqrt 2 ∧
      h' = (2 * (Real.sqrt 6 + Real.sqrt 2)) / c) → 
    h' ≤ h :=
by sorry

end NUMINAMATH_CALUDE_max_altitude_triangle_ABC_l1283_128343


namespace NUMINAMATH_CALUDE_nth_ring_area_l1283_128366

/-- Represents the area of a ring in a square garden system -/
def ring_area (n : ℕ) : ℝ :=
  36 * n

/-- Theorem stating the area of the nth ring in a square garden system -/
theorem nth_ring_area (n : ℕ) :
  ring_area n = 36 * n :=
by
  -- The proof goes here
  sorry

/-- The area of the 50th ring -/
def area_50th_ring : ℝ := ring_area 50

#eval area_50th_ring  -- Should evaluate to 1800

end NUMINAMATH_CALUDE_nth_ring_area_l1283_128366


namespace NUMINAMATH_CALUDE_hostel_cost_calculation_hostel_cost_23_days_l1283_128305

/-- Cost calculation for student youth hostel stay --/
theorem hostel_cost_calculation (first_week_rate : ℝ) (additional_week_rate : ℝ) (total_days : ℕ) : 
  first_week_rate = 18 →
  additional_week_rate = 11 →
  total_days = 23 →
  (7 * first_week_rate + (total_days - 7) * additional_week_rate : ℝ) = 302 := by
  sorry

/-- Main theorem: Cost of 23-day stay is $302.00 --/
theorem hostel_cost_23_days : 
  ∃ (first_week_rate additional_week_rate : ℝ),
    first_week_rate = 18 ∧
    additional_week_rate = 11 ∧
    (7 * first_week_rate + 16 * additional_week_rate : ℝ) = 302 := by
  sorry

end NUMINAMATH_CALUDE_hostel_cost_calculation_hostel_cost_23_days_l1283_128305


namespace NUMINAMATH_CALUDE_smallest_number_with_remainders_l1283_128331

theorem smallest_number_with_remainders : ∃ n : ℕ, 
  (n > 0) ∧ 
  (n % 3 = 2) ∧ 
  (n % 5 = 3) ∧ 
  (∀ m : ℕ, m > 0 → m % 3 = 2 → m % 5 = 3 → n ≤ m) ∧
  n = 8 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainders_l1283_128331


namespace NUMINAMATH_CALUDE_max_dot_product_l1283_128332

/-- Given plane vectors a, b, c satisfying the conditions, 
    the maximum value of the dot product of a and b is 1/3 -/
theorem max_dot_product (a b c : ℝ × ℝ) : 
  (a.1 * (a.1 + c.1) + a.2 * (a.2 + c.2) = 0) →
  ((a.1 + b.1 - 2*c.1)^2 + (a.2 + b.2 - 2*c.2)^2 = 4) →
  (∃ (k : ℝ), ∀ (a' b' c' : ℝ × ℝ), 
    (a'.1 * (a'.1 + c'.1) + a'.2 * (a'.2 + c'.2) = 0) →
    ((a'.1 + b'.1 - 2*c'.1)^2 + (a'.2 + b'.2 - 2*c'.2)^2 = 4) →
    (a'.1 * b'.1 + a'.2 * b'.2 ≤ k)) ∧
  (a.1 * b.1 + a.2 * b.2 ≤ 1/3) ∧
  (∃ (a'' b'' c'' : ℝ × ℝ), 
    (a''.1 * (a''.1 + c''.1) + a''.2 * (a''.2 + c''.2) = 0) ∧
    ((a''.1 + b''.1 - 2*c''.1)^2 + (a''.2 + b''.2 - 2*c''.2)^2 = 4) ∧
    (a''.1 * b''.1 + a''.2 * b''.2 = 1/3)) :=
by sorry

end NUMINAMATH_CALUDE_max_dot_product_l1283_128332


namespace NUMINAMATH_CALUDE_triangle_equation_no_real_roots_l1283_128322

theorem triangle_equation_no_real_roots 
  (a b c : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  ∀ x : ℝ, a^2 * x^2 - (c^2 - a^2 - b^2) * x + b^2 ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_triangle_equation_no_real_roots_l1283_128322


namespace NUMINAMATH_CALUDE_circle_sum_radii_geq_rectangle_sides_l1283_128325

/-- Given a rectangle ABCD with sides a and b, and two circles k₁ and k₂ where:
    - k₁ passes through A and B and is tangent to CD
    - k₂ passes through A and D and is tangent to BC
    - r₁ and r₂ are the radii of k₁ and k₂ respectively
    Prove that r₁ + r₂ ≥ 5/8 * (a + b) -/
theorem circle_sum_radii_geq_rectangle_sides 
  (a b r₁ r₂ : ℝ) 
  (ha : a > 0) (hb : b > 0)
  (hr₁ : r₁ = (a^2 + 4*b^2) / (8*b))
  (hr₂ : r₂ = (b^2 + 4*a^2) / (8*a)) :
  r₁ + r₂ ≥ 5/8 * (a + b) := by
  sorry

end NUMINAMATH_CALUDE_circle_sum_radii_geq_rectangle_sides_l1283_128325


namespace NUMINAMATH_CALUDE_solution_set_is_closed_interval_l1283_128394

-- Define the function representing the left side of the inequality
def f (x : ℝ) : ℝ := -x^2 + 2*x + 3

-- Define the solution set
def solution_set : Set ℝ := {x | f x ≥ 0}

-- Theorem statement
theorem solution_set_is_closed_interval :
  solution_set = Set.Icc (-1) 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_is_closed_interval_l1283_128394


namespace NUMINAMATH_CALUDE_quadratic_equation_with_zero_root_l1283_128304

theorem quadratic_equation_with_zero_root (a : ℝ) : 
  (∃ x : ℝ, (a - 1) * x^2 + x + (a - 2) = 0) ∧ 
  ((a - 1) * 0^2 + 0 + (a - 2) = 0) → 
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_with_zero_root_l1283_128304


namespace NUMINAMATH_CALUDE_number_of_people_l1283_128327

/-- Given a group of people, prove that there are 5 people based on the given conditions. -/
theorem number_of_people (n : ℕ) (total_age : ℕ) : n = 5 :=
  by
  /- Define the average age of all people -/
  have avg_age : total_age = n * 30 := by sorry
  
  /- Define the total age when the youngest was born -/
  have prev_total_age : total_age - 6 = (n - 1) * 24 := by sorry
  
  /- The main proof -/
  sorry

end NUMINAMATH_CALUDE_number_of_people_l1283_128327


namespace NUMINAMATH_CALUDE_trivia_team_size_l1283_128391

theorem trivia_team_size (absent_members : ℕ) (points_per_member : ℕ) (total_points : ℕ) :
  absent_members = 7 →
  points_per_member = 5 →
  total_points = 35 →
  absent_members + (total_points / points_per_member) = 14 := by
sorry

end NUMINAMATH_CALUDE_trivia_team_size_l1283_128391


namespace NUMINAMATH_CALUDE_polynomial_uniqueness_l1283_128364

-- Define the polynomial Q
def Q (a b c d : ℝ) (x : ℝ) : ℝ := a + b * x + c * x^2 + d * x^3

-- State the theorem
theorem polynomial_uniqueness (a b c d : ℝ) :
  Q a b c d (-1) = 2 →
  Q a b c d = (fun x => x^3 - x^2 + x + 2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_uniqueness_l1283_128364


namespace NUMINAMATH_CALUDE_hiltons_marbles_l1283_128328

theorem hiltons_marbles (initial_marbles : ℕ) : 
  (initial_marbles + 6 - 10 + 2 * 10 = 42) → initial_marbles = 26 := by
  sorry

end NUMINAMATH_CALUDE_hiltons_marbles_l1283_128328


namespace NUMINAMATH_CALUDE_exists_integer_fifth_power_less_than_one_l1283_128374

theorem exists_integer_fifth_power_less_than_one :
  ∃ x : ℤ, x^5 < 1 := by sorry

end NUMINAMATH_CALUDE_exists_integer_fifth_power_less_than_one_l1283_128374


namespace NUMINAMATH_CALUDE_floor_ceiling_evaluation_l1283_128361

theorem floor_ceiling_evaluation : ⌊(1.999 : ℝ)⌋ + ⌈(3.001 : ℝ)⌉ - ⌊(0.001 : ℝ)⌋ = 5 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceiling_evaluation_l1283_128361


namespace NUMINAMATH_CALUDE_currency_notes_total_l1283_128330

theorem currency_notes_total (total_notes : ℕ) (denom_1 denom_2 : ℕ) (amount_denom_2 : ℕ) : 
  total_notes = 100 → 
  denom_1 = 70 → 
  denom_2 = 50 → 
  amount_denom_2 = 100 →
  ∃ (notes_denom_1 notes_denom_2 : ℕ),
    notes_denom_1 + notes_denom_2 = total_notes ∧
    notes_denom_2 * denom_2 = amount_denom_2 ∧
    notes_denom_1 * denom_1 + notes_denom_2 * denom_2 = 6960 :=
by sorry

end NUMINAMATH_CALUDE_currency_notes_total_l1283_128330


namespace NUMINAMATH_CALUDE_complex_expression_equals_two_l1283_128370

theorem complex_expression_equals_two :
  (Complex.I * (1 - Complex.I)^2 : ℂ) = 2 := by sorry

end NUMINAMATH_CALUDE_complex_expression_equals_two_l1283_128370


namespace NUMINAMATH_CALUDE_quadratic_real_root_condition_l1283_128389

theorem quadratic_real_root_condition (b : ℝ) :
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≤ -10 ∨ b ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_root_condition_l1283_128389
