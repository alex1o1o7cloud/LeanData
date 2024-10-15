import Mathlib

namespace NUMINAMATH_CALUDE_train_speed_l3272_327291

/-- The speed of a train given its length, time to cross a man, and the man's speed -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (man_speed : ℝ) :
  train_length = 900 →
  crossing_time = 53.99568034557235 →
  man_speed = 3 →
  ∃ (train_speed : ℝ), abs (train_speed - 63.0036) < 0.0001 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l3272_327291


namespace NUMINAMATH_CALUDE_line_point_sum_l3272_327287

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := y = -5/3 * x + 15

/-- Point P is on the x-axis -/
def P : ℝ × ℝ := (9, 0)

/-- Point Q is on the y-axis -/
def Q : ℝ × ℝ := (0, 15)

/-- Point T is on the line segment PQ -/
def T_on_PQ (r s : ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ r = t * P.1 + (1 - t) * Q.1 ∧ s = t * P.2 + (1 - t) * Q.2

/-- The area of triangle POQ is four times the area of triangle TOP -/
def area_condition (r s : ℝ) : Prop :=
  abs ((P.1 - 0) * (Q.2 - 0) - (Q.1 - 0) * (P.2 - 0)) / 2 =
  4 * abs ((P.1 - 0) * (s - 0) - (r - 0) * (P.2 - 0)) / 2

/-- The main theorem -/
theorem line_point_sum (r s : ℝ) :
  line_equation r s →
  T_on_PQ r s →
  area_condition r s →
  r + s = 8.75 := by sorry

end NUMINAMATH_CALUDE_line_point_sum_l3272_327287


namespace NUMINAMATH_CALUDE_b_over_a_range_l3272_327201

/-- A cubic equation with real coefficients -/
structure CubicEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to check if the roots represent eccentricities of conic sections -/
def has_conic_eccentricities (eq : CubicEquation) : Prop :=
  ∃ (e₁ e₂ e₃ : ℝ), 
    e₁^3 + eq.a * e₁^2 + eq.b * e₁ + eq.c = 0 ∧
    e₂^3 + eq.a * e₂^2 + eq.b * e₂ + eq.c = 0 ∧
    e₃^3 + eq.a * e₃^2 + eq.b * e₃ + eq.c = 0 ∧
    (0 ≤ e₁ ∧ e₁ < 1) ∧  -- ellipse eccentricity
    (e₂ > 1) ∧           -- hyperbola eccentricity
    (e₃ = 1)             -- parabola eccentricity

/-- The main theorem stating the range of b/a -/
theorem b_over_a_range (eq : CubicEquation) 
  (h : has_conic_eccentricities eq) : 
  -2 < eq.b / eq.a ∧ eq.b / eq.a < -1/2 :=
by sorry

end NUMINAMATH_CALUDE_b_over_a_range_l3272_327201


namespace NUMINAMATH_CALUDE_special_multiplication_pattern_l3272_327244

theorem special_multiplication_pattern (a b : ℕ) (ha : a ≤ 9) (hb : b ≤ 9) :
  (10 * a + b) * (10 * a + (10 - b)) = 100 * a * (a + 1) + b * (10 - b) := by
  sorry

end NUMINAMATH_CALUDE_special_multiplication_pattern_l3272_327244


namespace NUMINAMATH_CALUDE_expression_value_l3272_327280

theorem expression_value : (3^2 - 3) - (5^2 - 5) * 2 + (6^2 - 6) = -4 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3272_327280


namespace NUMINAMATH_CALUDE_negation_equivalence_l3272_327219

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀^2 + 2*x₀ > 0) ↔ (∀ x : ℝ, x^2 + 2*x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3272_327219


namespace NUMINAMATH_CALUDE_lynne_book_cost_l3272_327204

/-- Proves that the cost of each book is $7 given the conditions of Lynne's purchase -/
theorem lynne_book_cost (num_books : ℕ) (num_magazines : ℕ) (magazine_cost : ℚ) (total_spent : ℚ) :
  num_books = 9 →
  num_magazines = 3 →
  magazine_cost = 4 →
  total_spent = 75 →
  (num_books * (total_spent - num_magazines * magazine_cost) / num_books : ℚ) = 7 := by
  sorry

end NUMINAMATH_CALUDE_lynne_book_cost_l3272_327204


namespace NUMINAMATH_CALUDE_box_minus_two_zero_three_l3272_327208

def box (a b c : ℤ) : ℚ := a^b - b^c + c^a

theorem box_minus_two_zero_three : box (-2) 0 3 = 10/9 := by
  sorry

end NUMINAMATH_CALUDE_box_minus_two_zero_three_l3272_327208


namespace NUMINAMATH_CALUDE_length_of_PQ_l3272_327274

-- Define the points
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (-1, 0)

-- Define the locus E
def E : Set (ℝ × ℝ) := {p | p.1^2 - p.2^2/4 = 1 ∧ p.1 ≠ 1 ∧ p.1 ≠ -1}

-- Define the slope product condition
def slope_product (M : ℝ × ℝ) : Prop :=
  (M.2 / (M.1 - 1)) * (M.2 / (M.1 + 1)) = 4

-- Define line l
def l : Set (ℝ × ℝ) := {p | ∃ (k : ℝ), p.2 = k * p.1 - 2}

-- Define the midpoint condition
def midpoint_condition (P Q : ℝ × ℝ) : Prop :=
  let D := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
  D.1 > 0 ∧ D.2 = 2

-- Main theorem
theorem length_of_PQ :
  ∀ (P Q : ℝ × ℝ),
  P ∈ E ∧ Q ∈ E ∧ P ∈ l ∧ Q ∈ l ∧
  midpoint_condition P Q ∧
  (∀ M ∈ E, slope_product M) →
  ‖P - Q‖ = 2 * Real.sqrt 14 :=
sorry

end NUMINAMATH_CALUDE_length_of_PQ_l3272_327274


namespace NUMINAMATH_CALUDE_exists_twelve_digit_non_cube_l3272_327278

theorem exists_twelve_digit_non_cube : ∃ n : ℕ, (10^11 ≤ n ∧ n < 10^12) ∧ ¬∃ k : ℕ, n = k^3 := by
  sorry

end NUMINAMATH_CALUDE_exists_twelve_digit_non_cube_l3272_327278


namespace NUMINAMATH_CALUDE_factor_difference_of_squares_l3272_327202

theorem factor_difference_of_squares (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := by
  sorry

end NUMINAMATH_CALUDE_factor_difference_of_squares_l3272_327202


namespace NUMINAMATH_CALUDE_total_cost_is_correct_l3272_327253

def calculate_total_cost (type_a_count : ℕ) (type_b_count : ℕ) (type_c_count : ℕ)
  (type_a_price : ℚ) (type_b_price : ℚ) (type_c_price : ℚ)
  (type_a_discount : ℚ) (type_b_discount : ℚ) (type_c_discount : ℚ)
  (type_a_discount_threshold : ℕ) (type_b_discount_threshold : ℕ) (type_c_discount_threshold : ℕ) : ℚ :=
  let type_a_cost := type_a_count * type_a_price
  let type_b_cost := type_b_count * type_b_price
  let type_c_cost := type_c_count * type_c_price
  let type_a_discounted_cost := if type_a_count > type_a_discount_threshold then type_a_cost * (1 - type_a_discount) else type_a_cost
  let type_b_discounted_cost := if type_b_count > type_b_discount_threshold then type_b_cost * (1 - type_b_discount) else type_b_cost
  let type_c_discounted_cost := if type_c_count > type_c_discount_threshold then type_c_cost * (1 - type_c_discount) else type_c_cost
  type_a_discounted_cost + type_b_discounted_cost + type_c_discounted_cost

theorem total_cost_is_correct :
  calculate_total_cost 150 90 60 2 3 5 0.2 0.15 0.1 100 50 30 = 739.5 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_correct_l3272_327253


namespace NUMINAMATH_CALUDE_milk_production_days_l3272_327211

/-- Given that x cows produce x+1 cans of milk in x+2 days, 
    this theorem proves the number of days it takes x+3 cows to produce x+5 cans of milk. -/
theorem milk_production_days (x : ℝ) (h : x > 0) : 
  let initial_cows := x
  let initial_milk := x + 1
  let initial_days := x + 2
  let new_cows := x + 3
  let new_milk := x + 5
  let daily_production_per_cow := initial_milk / (initial_cows * initial_days)
  let days_for_new_production := new_milk / (new_cows * daily_production_per_cow)
  days_for_new_production = x * (x + 2) * (x + 5) / ((x + 1) * (x + 3)) :=
by sorry

end NUMINAMATH_CALUDE_milk_production_days_l3272_327211


namespace NUMINAMATH_CALUDE_card_number_sum_l3272_327292

theorem card_number_sum (a b c d e f g h : ℕ) :
  (a + b) * (c + d) * (e + f) * (g + h) = 330 →
  a + b + c + d + e + f + g + h = 21 := by
  sorry

end NUMINAMATH_CALUDE_card_number_sum_l3272_327292


namespace NUMINAMATH_CALUDE_lines_per_page_l3272_327249

theorem lines_per_page (total_words : ℕ) (words_per_line : ℕ) (pages_filled : ℚ) (words_left : ℕ) : 
  total_words = 400 →
  words_per_line = 10 →
  pages_filled = 3/2 →
  words_left = 100 →
  (total_words - words_left) / words_per_line / pages_filled = 20 := by
sorry

end NUMINAMATH_CALUDE_lines_per_page_l3272_327249


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3272_327213

theorem complex_modulus_problem (z : ℂ) : (1 - Complex.I) * z = 1 + Complex.I → Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3272_327213


namespace NUMINAMATH_CALUDE_unique_p_l3272_327235

/-- The cubic equation with parameter p -/
def cubic_equation (p : ℝ) (x : ℝ) : Prop :=
  5 * x^3 - 5*(p+1)*x^2 + (71*p-1)*x + 1 = 66*p

/-- A number is a natural root of the cubic equation -/
def is_natural_root (p : ℝ) (x : ℕ) : Prop :=
  cubic_equation p (x : ℝ)

/-- The cubic equation has exactly three natural roots -/
def has_three_natural_roots (p : ℝ) : Prop :=
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    is_natural_root p a ∧ is_natural_root p b ∧ is_natural_root p c ∧
    ∀ (x : ℕ), is_natural_root p x → (x = a ∨ x = b ∨ x = c)

/-- The main theorem: 76 is the only real number satisfying the conditions -/
theorem unique_p : ∀ (p : ℝ), has_three_natural_roots p ↔ p = 76 := by
  sorry

end NUMINAMATH_CALUDE_unique_p_l3272_327235


namespace NUMINAMATH_CALUDE_complement_of_B_l3272_327252

-- Define the sets A and B
def A (x : ℝ) : Set ℝ := {1, 3, x}
def B (x : ℝ) : Set ℝ := {1, x^2}

-- Define the universal set U
def U (x : ℝ) : Set ℝ := A x ∪ B x

-- State the theorem
theorem complement_of_B (x : ℝ) :
  (B x ∪ (U x \ B x) = A x) →
  ((x = 0 → U x \ B x = {3}) ∧
   (x = Real.sqrt 3 → U x \ B x = {Real.sqrt 3}) ∧
   (x = -Real.sqrt 3 → U x \ B x = {-Real.sqrt 3})) :=
by sorry

end NUMINAMATH_CALUDE_complement_of_B_l3272_327252


namespace NUMINAMATH_CALUDE_solution_sets_l3272_327230

def solution_set_1 (a b : ℝ) : Set ℝ := {x | a * x - b > 0}
def solution_set_2 (a b : ℝ) : Set ℝ := {x | (a * x + b) / (x - 2) > 0}

theorem solution_sets (a b : ℝ) :
  solution_set_1 a b = Set.Ioi 1 →
  solution_set_2 a b = Set.Iic (-1) ∪ Set.Ioi 2 :=
by sorry

end NUMINAMATH_CALUDE_solution_sets_l3272_327230


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2015th_term_l3272_327269

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  a₁_eq_1 : a 1 = 1
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1
  d_neq_0 : a 2 - a 1 ≠ 0
  geometric_subset : (a 2 / a 1) = (a 5 / a 2)

/-- The 2015th term of the arithmetic sequence is 4029 -/
theorem arithmetic_sequence_2015th_term (seq : ArithmeticSequence) : seq.a 2015 = 4029 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2015th_term_l3272_327269


namespace NUMINAMATH_CALUDE_absolute_value_equality_l3272_327295

theorem absolute_value_equality (a b : ℝ) : |a| = |b| → a = b ∨ a = -b := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l3272_327295


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l3272_327273

theorem nested_fraction_equality : 
  (1 / (1 + 1 / (2 + 1 / 5))) = 11 / 16 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l3272_327273


namespace NUMINAMATH_CALUDE_sara_remaining_marbles_l3272_327248

def initial_black_marbles : ℕ := 792
def marbles_taken : ℕ := 233

theorem sara_remaining_marbles :
  initial_black_marbles - marbles_taken = 559 :=
by sorry

end NUMINAMATH_CALUDE_sara_remaining_marbles_l3272_327248


namespace NUMINAMATH_CALUDE_darren_fergie_equal_debt_l3272_327217

/-- Represents the amount owed after t days with simple interest -/
def amountOwed (principal : ℝ) (rate : ℝ) (days : ℝ) : ℝ :=
  principal * (1 + rate * days)

/-- The problem statement -/
theorem darren_fergie_equal_debt : ∃ t : ℝ, 
  t = 20 ∧ 
  amountOwed 100 0.10 t = amountOwed 150 0.05 t :=
sorry

end NUMINAMATH_CALUDE_darren_fergie_equal_debt_l3272_327217


namespace NUMINAMATH_CALUDE_area_curve_C_m_1_intersection_points_l3272_327241

-- Define the curve C
def curve_C (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |p.1| + |p.2| = m}

-- Define the ellipse
def ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 9 + p.2^2 / 4 = 1}

-- Theorem 1: Area enclosed by curve C when m = 1
theorem area_curve_C_m_1 :
  MeasureTheory.volume (curve_C 1) = 2 := by sorry

-- Theorem 2: Intersection points of curve C and ellipse
theorem intersection_points (m : ℝ) :
  (∃ (a b c d : ℝ × ℝ), a ∈ curve_C m ∩ ellipse ∧
                         b ∈ curve_C m ∩ ellipse ∧
                         c ∈ curve_C m ∩ ellipse ∧
                         d ∈ curve_C m ∩ ellipse ∧
                         a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ↔
  (2 < m ∧ m < 3) ∨ m = Real.sqrt 13 := by sorry

end NUMINAMATH_CALUDE_area_curve_C_m_1_intersection_points_l3272_327241


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3272_327207

theorem quadratic_inequality_solution_set :
  {x : ℝ | -2 * x^2 - x + 6 ≥ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 3/2} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3272_327207


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3272_327242

-- Define the complex number z as a function of a
def z (a : ℝ) : ℂ := (a^2 - 4 : ℝ) + (a + 1 : ℝ) * Complex.I

-- Define what it means for a complex number to be purely imaginary
def is_purely_imaginary (w : ℂ) : Prop := w.re = 0 ∧ w.im ≠ 0

-- State the theorem
theorem sufficient_not_necessary :
  (∃ (a : ℝ), a ≠ -2 ∧ is_purely_imaginary (z a)) ∧
  (∀ (a : ℝ), a = -2 → is_purely_imaginary (z a)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3272_327242


namespace NUMINAMATH_CALUDE_pharmacy_tubs_l3272_327298

theorem pharmacy_tubs (total_needed : ℕ) (in_storage : ℕ) : 
  total_needed = 100 →
  in_storage = 20 →
  let to_buy := total_needed - in_storage
  let from_new_vendor := to_buy / 4
  let from_usual_vendor := to_buy - from_new_vendor
  from_usual_vendor = 60 := by
  sorry

end NUMINAMATH_CALUDE_pharmacy_tubs_l3272_327298


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l3272_327228

/-- Given an ellipse with semi-major axis a and semi-minor axis b,
    and a line l passing through a vertex (0, b) and a focus (c, 0),
    if the distance from the center to l is b/4,
    then the eccentricity of the ellipse is 1/2. -/
theorem ellipse_eccentricity (a b c : ℝ) (h1 : a > b) (h2 : b > 0) :
  (1 / Real.sqrt ((1 / c^2) + (1 / b^2)) = b / 4) →
  c / a = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l3272_327228


namespace NUMINAMATH_CALUDE_intersection_sum_l3272_327212

/-- 
Given two lines y = 2x + c and y = 4x + d that intersect at the point (8, 12),
prove that c + d = -24
-/
theorem intersection_sum (c d : ℝ) : 
  (∀ x y : ℝ, y = 2*x + c) →
  (∀ x y : ℝ, y = 4*x + d) →
  12 = 2*8 + c →
  12 = 4*8 + d →
  c + d = -24 := by
sorry

end NUMINAMATH_CALUDE_intersection_sum_l3272_327212


namespace NUMINAMATH_CALUDE_anthony_pencils_l3272_327290

theorem anthony_pencils (x : ℕ) : x + 56 = 65 → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_anthony_pencils_l3272_327290


namespace NUMINAMATH_CALUDE_equal_book_distribution_l3272_327266

theorem equal_book_distribution (total_students : ℕ) (girls : ℕ) (boys : ℕ) 
  (total_books : ℕ) (girls_books : ℕ) :
  total_students = girls + boys →
  total_books = 375 →
  girls = 15 →
  boys = 10 →
  girls_books = 225 →
  ∃ (books_per_student : ℕ), 
    books_per_student = 15 ∧
    girls_books = girls * books_per_student ∧
    total_books = total_students * books_per_student :=
by sorry

end NUMINAMATH_CALUDE_equal_book_distribution_l3272_327266


namespace NUMINAMATH_CALUDE_cylinder_cross_section_area_coefficient_sum_l3272_327232

/-- The area of a cross-section in a cylinder --/
theorem cylinder_cross_section_area :
  ∀ (r : ℝ) (θ : ℝ),
  r = 8 →
  θ = π / 2 →
  ∃ (A : ℝ),
  A = 16 * Real.sqrt 3 * π + 16 * Real.sqrt 6 ∧
  A = (r^2 * θ / 4 + r^2 * Real.sin (θ / 2) * Real.cos (θ / 2)) * Real.sqrt 3 :=
by sorry

/-- The sum of coefficients in the area expression --/
theorem coefficient_sum :
  ∃ (d e : ℝ) (f : ℕ),
  16 * Real.sqrt 3 * π + 16 * Real.sqrt 6 = d * π + e * Real.sqrt f ∧
  d + e + f = 38 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_cross_section_area_coefficient_sum_l3272_327232


namespace NUMINAMATH_CALUDE_complex_cube_equals_negative_one_l3272_327246

theorem complex_cube_equals_negative_one : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (1/2 - Complex.I * (Real.sqrt 3)/2) ^ 3 = -1 :=
by sorry

end NUMINAMATH_CALUDE_complex_cube_equals_negative_one_l3272_327246


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l3272_327240

/-- An arithmetic sequence with given conditions -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  a 2 = -1 ∧ a 4 = 3 ∧ ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

/-- The general formula for the arithmetic sequence -/
def GeneralFormula (n : ℕ) : ℤ := 2 * n - 5

theorem arithmetic_sequence_formula (a : ℕ → ℤ) :
  ArithmeticSequence a → ∀ n : ℕ, a n = GeneralFormula n := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l3272_327240


namespace NUMINAMATH_CALUDE_total_cost_of_shed_is_818_25_l3272_327200

/-- Calculate the total cost of constructing a shed given the following conditions:
  * 1000 bricks are needed
  * 30% of bricks are at 50% discount off $0.50 each
  * 40% of bricks are at 20% discount off $0.50 each
  * 30% of bricks are at full price of $0.50 each
  * 5% tax on total cost of bricks
  * Additional building materials cost $200
  * 7% tax on additional building materials
  * Labor fees are $20 per hour for 10 hours
-/
def total_cost_of_shed : ℝ :=
  let total_bricks : ℝ := 1000
  let brick_full_price : ℝ := 0.50
  let discounted_bricks_1 : ℝ := 0.30 * total_bricks
  let discounted_bricks_2 : ℝ := 0.40 * total_bricks
  let full_price_bricks : ℝ := 0.30 * total_bricks
  let discount_1 : ℝ := 0.50
  let discount_2 : ℝ := 0.20
  let brick_tax_rate : ℝ := 0.05
  let additional_materials_cost : ℝ := 200
  let materials_tax_rate : ℝ := 0.07
  let labor_rate : ℝ := 20
  let labor_hours : ℝ := 10

  let discounted_price_1 : ℝ := brick_full_price * (1 - discount_1)
  let discounted_price_2 : ℝ := brick_full_price * (1 - discount_2)
  
  let brick_cost : ℝ := 
    discounted_bricks_1 * discounted_price_1 +
    discounted_bricks_2 * discounted_price_2 +
    full_price_bricks * brick_full_price
  
  let brick_tax : ℝ := brick_cost * brick_tax_rate
  let materials_tax : ℝ := additional_materials_cost * materials_tax_rate
  let labor_cost : ℝ := labor_rate * labor_hours

  brick_cost + brick_tax + additional_materials_cost + materials_tax + labor_cost

theorem total_cost_of_shed_is_818_25 : 
  total_cost_of_shed = 818.25 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_of_shed_is_818_25_l3272_327200


namespace NUMINAMATH_CALUDE_number_transformation_l3272_327268

theorem number_transformation (x : ℝ) : 
  x + 0.40 * x = 1680 → x * 0.80 * 1.15 = 1104 := by
  sorry

end NUMINAMATH_CALUDE_number_transformation_l3272_327268


namespace NUMINAMATH_CALUDE_max_profit_theorem_l3272_327215

/-- Represents the profit function for a product sale scenario. -/
def profit_function (x : ℝ) : ℝ := -160 * x^2 + 560 * x + 3120

/-- Represents the factory price of the product. -/
def factory_price : ℝ := 3

/-- Represents the initial retail price. -/
def initial_retail_price : ℝ := 4

/-- Represents the initial monthly sales volume. -/
def initial_sales_volume : ℝ := 400

/-- Represents the change in sales volume for every 0.5 CNY price change. -/
def sales_volume_change : ℝ := 40

/-- Theorem stating the maximum profit and the corresponding selling prices. -/
theorem max_profit_theorem :
  (∃ (x : ℝ), x = 1.5 ∨ x = 2) ∧
  (∀ (y : ℝ), y ≤ 3600 → ∃ (x : ℝ), profit_function x = y) ∧
  profit_function 1.5 = 3600 ∧
  profit_function 2 = 3600 := by sorry

end NUMINAMATH_CALUDE_max_profit_theorem_l3272_327215


namespace NUMINAMATH_CALUDE_crayons_in_drawer_l3272_327233

/-- The number of crayons initially in the drawer -/
def initial_crayons : ℕ := 9

/-- The number of crayons Benny added to the drawer -/
def added_crayons : ℕ := 3

/-- The total number of crayons in the drawer after Benny's addition -/
def total_crayons : ℕ := initial_crayons + added_crayons

theorem crayons_in_drawer : total_crayons = 12 := by
  sorry

end NUMINAMATH_CALUDE_crayons_in_drawer_l3272_327233


namespace NUMINAMATH_CALUDE_airplane_rows_l3272_327275

/-- 
Given an airplane with the following conditions:
- Each row has 8 seats
- Only 3/4 of the seats in each row can be occupied
- There are 24 unoccupied seats on the plane

This theorem proves that the number of rows on the airplane is 12.
-/
theorem airplane_rows : 
  ∀ (rows : ℕ), 
  (8 : ℚ) * rows - (3 / 4 : ℚ) * 8 * rows = 24 → 
  rows = 12 := by
sorry

end NUMINAMATH_CALUDE_airplane_rows_l3272_327275


namespace NUMINAMATH_CALUDE_pirate_count_l3272_327210

theorem pirate_count : ∃ p : ℕ, 
  p > 0 ∧ 
  (∃ (participants : ℕ), participants = p - 10 ∧ 
    (54 : ℚ) / 100 * participants = (↑⌊(54 : ℚ) / 100 * participants⌋ : ℚ) ∧ 
    (34 : ℚ) / 100 * participants = (↑⌊(34 : ℚ) / 100 * participants⌋ : ℚ) ∧ 
    (2 : ℚ) / 3 * p = (↑⌊(2 : ℚ) / 3 * p⌋ : ℚ)) ∧ 
  p = 60 := by
  sorry

end NUMINAMATH_CALUDE_pirate_count_l3272_327210


namespace NUMINAMATH_CALUDE_tangent_line_problem_range_problem_l3272_327272

noncomputable section

-- Define the function f(x) = x - ln x
def f (x : ℝ) : ℝ := x - Real.log x

-- Define the function g(x) = (e-1)x
def g (x : ℝ) : ℝ := (Real.exp 1 - 1) * x

-- Define the piecewise function F(x)
def F (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ a then f x else g x

-- Theorem for the tangent line problem
theorem tangent_line_problem (x₀ : ℝ) :
  (∃ k : ℝ, ∀ x : ℝ, k * x = f x + (x - x₀) * (1 - 1 / x₀)) →
  (x₀ = Real.exp 1 ∧ ∃ k : ℝ, k = 1 - 1 / Real.exp 1) :=
sorry

-- Theorem for the range problem
theorem range_problem (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, F a x = y) →
  a ≥ 1 / (Real.exp 1 - 1) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_problem_range_problem_l3272_327272


namespace NUMINAMATH_CALUDE_deaf_to_blind_ratio_l3272_327257

theorem deaf_to_blind_ratio (total_students blind_students : ℕ) 
  (h1 : total_students = 180)
  (h2 : blind_students = 45) :
  (total_students - blind_students) / blind_students = 3 := by
  sorry

end NUMINAMATH_CALUDE_deaf_to_blind_ratio_l3272_327257


namespace NUMINAMATH_CALUDE_archer_probabilities_l3272_327265

/-- Represents the probability of an archer hitting a target -/
def hit_probability : ℚ := 2/3

/-- Represents the number of shots taken -/
def num_shots : ℕ := 5

/-- Calculates the probability of hitting the target exactly k times in n shots -/
def prob_exact_hits (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * hit_probability ^ k * (1 - hit_probability) ^ (n - k)

/-- Calculates the probability of hitting the target k times in a row and missing n-k times in n shots -/
def prob_consecutive_hits (n k : ℕ) : ℚ :=
  (n - k + 1 : ℚ) * hit_probability ^ k * (1 - hit_probability) ^ (n - k)

theorem archer_probabilities :
  (prob_exact_hits num_shots 2 = 40/243) ∧
  (prob_consecutive_hits num_shots 3 = 8/81) := by
  sorry

end NUMINAMATH_CALUDE_archer_probabilities_l3272_327265


namespace NUMINAMATH_CALUDE_tuesday_children_count_l3272_327286

/-- Represents the number of children who went to the zoo on Tuesday -/
def tuesday_children : ℕ := sorry

/-- Theorem stating that the number of children who went to the zoo on Tuesday is 4 -/
theorem tuesday_children_count : tuesday_children = 4 := by
  have monday_revenue : ℕ := 7 * 3 + 5 * 4
  have tuesday_revenue : ℕ := tuesday_children * 3 + 2 * 4
  have total_revenue : ℕ := 61
  have revenue_equation : monday_revenue + tuesday_revenue = total_revenue := sorry
  sorry

end NUMINAMATH_CALUDE_tuesday_children_count_l3272_327286


namespace NUMINAMATH_CALUDE_digit_sum_property_l3272_327231

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

theorem digit_sum_property (n : ℕ) 
  (h1 : sum_of_digits n = 100)
  (h2 : sum_of_digits (44 * n) = 800) :
  sum_of_digits (3 * n) = 300 := by sorry

end NUMINAMATH_CALUDE_digit_sum_property_l3272_327231


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_triangle_l3272_327245

/-- Given a quadrilateral inscribed in a circle of radius R, with points P, Q, and M as described,
    and distances a, b, and c from these points to the circle's center,
    prove that the sides of triangle PQM have the given lengths. -/
theorem inscribed_quadrilateral_triangle (R a b c : ℝ) (h_pos : R > 0) :
  ∃ (PQ QM PM : ℝ),
    PQ = Real.sqrt (a^2 + b^2 - 2*R^2) ∧
    QM = Real.sqrt (b^2 + c^2 - 2*R^2) ∧
    PM = Real.sqrt (c^2 + a^2 - 2*R^2) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_triangle_l3272_327245


namespace NUMINAMATH_CALUDE_min_value_problem_l3272_327222

theorem min_value_problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 3 * y = 5 * x * y) :
  3 * x + 4 * y ≥ 5 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ x + 3 * y = 5 * x * y ∧ 3 * x + 4 * y = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l3272_327222


namespace NUMINAMATH_CALUDE_polynomial_remainder_l3272_327250

theorem polynomial_remainder (x : ℝ) : 
  let f : ℝ → ℝ := λ x => x^4 - 8*x^3 + 12*x^2 + 20*x - 10
  let g : ℝ → ℝ := λ x => x - 2
  ∃ q : ℝ → ℝ, f x = g x * q x + 30 := by
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l3272_327250


namespace NUMINAMATH_CALUDE_bug_probability_l3272_327282

/-- Probability of the bug being at vertex A after n meters -/
def P : ℕ → ℚ
  | 0 => 1
  | n + 1 => 1/3 * (1 - P n)

/-- The probability of the bug being at vertex A after 8 meters is 1823/6561 -/
theorem bug_probability : P 8 = 1823 / 6561 := by
  sorry

end NUMINAMATH_CALUDE_bug_probability_l3272_327282


namespace NUMINAMATH_CALUDE_james_heavy_lifting_days_l3272_327224

/-- Calculates the number of days until James can lift heavy again after an injury. -/
def daysUntilHeavyLifting (painSubsideDays : ℕ) (healingMultiplier : ℕ) (waitAfterHealingDays : ℕ) (waitBeforeHeavyLiftingWeeks : ℕ) : ℕ :=
  let healingDays := painSubsideDays * healingMultiplier
  let totalDaysBeforeWorkout := healingDays + waitAfterHealingDays
  let waitBeforeHeavyLiftingDays := waitBeforeHeavyLiftingWeeks * 7
  totalDaysBeforeWorkout + waitBeforeHeavyLiftingDays

/-- Theorem stating that James can lift heavy again after 39 days given the specific conditions. -/
theorem james_heavy_lifting_days :
  daysUntilHeavyLifting 3 5 3 3 = 39 := by
  sorry

#eval daysUntilHeavyLifting 3 5 3 3

end NUMINAMATH_CALUDE_james_heavy_lifting_days_l3272_327224


namespace NUMINAMATH_CALUDE_zhang_san_not_losing_probability_l3272_327247

theorem zhang_san_not_losing_probability 
  (p_win : ℚ) (p_draw : ℚ) 
  (h_win : p_win = 1/3) 
  (h_draw : p_draw = 1/4) : 
  p_win + p_draw = 7/12 := by
  sorry

end NUMINAMATH_CALUDE_zhang_san_not_losing_probability_l3272_327247


namespace NUMINAMATH_CALUDE_polygon_sides_count_l3272_327279

/-- Given a polygon where the sum of its interior angles is 180° less than three times
    the sum of its exterior angles, prove that it has 5 sides. -/
theorem polygon_sides_count (n : ℕ) : n > 2 →
  (n - 2) * 180 = 3 * 360 - 180 →
  n = 5 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_count_l3272_327279


namespace NUMINAMATH_CALUDE_total_amount_after_stock_sale_l3272_327223

def initial_wallet_amount : ℝ := 300
def initial_investment : ℝ := 2000
def stock_price_increase_percentage : ℝ := 0.30

theorem total_amount_after_stock_sale :
  initial_wallet_amount +
  initial_investment * (1 + stock_price_increase_percentage) =
  2900 :=
by sorry

end NUMINAMATH_CALUDE_total_amount_after_stock_sale_l3272_327223


namespace NUMINAMATH_CALUDE_dice_roll_sum_l3272_327284

theorem dice_roll_sum (a b c d : ℕ) : 
  1 ≤ a ∧ a ≤ 6 →
  1 ≤ b ∧ b ≤ 6 →
  1 ≤ c ∧ c ≤ 6 →
  1 ≤ d ∧ d ≤ 6 →
  a * b * c * d = 216 →
  a + b + c + d ≠ 18 := by
sorry

end NUMINAMATH_CALUDE_dice_roll_sum_l3272_327284


namespace NUMINAMATH_CALUDE_largest_sum_of_digits_for_reciprocal_fraction_l3272_327220

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the decimal 0.abc -/
def DecimalABC (a b c : Digit) : ℚ := (a.val * 100 + b.val * 10 + c.val : ℕ) / 1000

/-- The theorem statement -/
theorem largest_sum_of_digits_for_reciprocal_fraction :
  ∀ (a b c : Digit) (y : ℕ+),
    (0 < y.val) → (y.val ≤ 16) →
    (DecimalABC a b c = 1 / y) →
    (∀ (a' b' c' : Digit) (y' : ℕ+),
      (0 < y'.val) → (y'.val ≤ 16) →
      (DecimalABC a' b' c' = 1 / y') →
      (a.val + b.val + c.val ≥ a'.val + b'.val + c'.val)) →
    (a.val + b.val + c.val = 8) :=
by sorry

end NUMINAMATH_CALUDE_largest_sum_of_digits_for_reciprocal_fraction_l3272_327220


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3272_327267

-- Define the function f
def f : ℝ → ℝ := λ x ↦ x^2

-- Define set B
def B : Set ℝ := {1, 2}

-- Theorem statement
theorem intersection_of_A_and_B 
  (A : Set ℝ) 
  (h : f '' A ⊆ B) :
  (A ∩ B = ∅) ∨ (A ∩ B = {1}) :=
sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3272_327267


namespace NUMINAMATH_CALUDE_proposition_1_proposition_3_l3272_327297

-- Define the set S
def S (m l : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ l}

-- Define the property that for all x in S, x² is also in S
def square_closed (m l : ℝ) : Prop :=
  ∀ x, x ∈ S m l → x^2 ∈ S m l

-- Theorem 1: If m = 1, then S = {1}
theorem proposition_1 (l : ℝ) (h : square_closed 1 l) :
  S 1 l = {1} :=
sorry

-- Theorem 2: If m = -1/3, then l ∈ [1/9, 1]
theorem proposition_3 (l : ℝ) (h : square_closed (-1/3) l) :
  1/9 ≤ l ∧ l ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_proposition_1_proposition_3_l3272_327297


namespace NUMINAMATH_CALUDE_solve_system_l3272_327261

-- Define the system of equations and the condition
def system_equations (x y m : ℝ) : Prop :=
  (4 * x + 2 * y = 3 * m) ∧ (3 * x + y = m + 2)

def opposite_sign (x y : ℝ) : Prop :=
  y = -x

-- Theorem statement
theorem solve_system :
  ∀ x y m : ℝ, system_equations x y m → opposite_sign x y → m = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_system_l3272_327261


namespace NUMINAMATH_CALUDE_unique_integer_satisfying_conditions_l3272_327229

theorem unique_integer_satisfying_conditions (x : ℤ) 
  (h1 : 5 < x ∧ x < 21)
  (h2 : 7 < x ∧ x < 18)
  (h3 : 13 > x ∧ x > 2)
  (h4 : 12 > x ∧ x > 9)
  (h5 : x + 1 < 13) : 
  x = 10 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_satisfying_conditions_l3272_327229


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3272_327289

theorem rationalize_denominator : 
  (1 : ℝ) / (Real.rpow 3 (1/3) + Real.rpow 81 (1/3)) = (1/4 : ℝ) := by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3272_327289


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l3272_327271

theorem smallest_n_congruence :
  ∃ (n : ℕ), n > 0 ∧ (3 * n) % 26 = 1367 % 26 ∧
  ∀ (m : ℕ), m > 0 ∧ (3 * m) % 26 = 1367 % 26 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l3272_327271


namespace NUMINAMATH_CALUDE_m_range_theorem_l3272_327238

-- Define the statements p and q
def p (x : ℝ) : Prop := -2 ≤ 1 - (x - 1) / 3 ∧ 1 - (x - 1) / 3 ≤ 2

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0 ∧ m > 0

-- Define the set A (where p is true)
def A : Set ℝ := {x | p x}

-- Define the set B (where q is true)
def B (m : ℝ) : Set ℝ := {x | q x m}

-- State the theorem
theorem m_range_theorem :
  ∀ m : ℝ, 
    (∀ x : ℝ, x ∈ A → x ∈ B m) ∧  -- p implies q
    (∃ x : ℝ, x ∈ B m ∧ x ∉ A) ∧  -- q does not imply p
    m ≥ 40 ∧ m < 50               -- m is in [40, 50)
  ↔ m ∈ Set.Icc 40 50 := by sorry

end NUMINAMATH_CALUDE_m_range_theorem_l3272_327238


namespace NUMINAMATH_CALUDE_not_perfect_squares_l3272_327296

theorem not_perfect_squares : 
  (∃ x : ℝ, (6 : ℝ)^3032 = x^2) ∧ 
  (∀ x : ℝ, (7 : ℝ)^3033 ≠ x^2) ∧ 
  (∃ x : ℝ, (8 : ℝ)^3034 = x^2) ∧ 
  (∀ x : ℝ, (9 : ℝ)^3035 ≠ x^2) ∧ 
  (∃ x : ℝ, (10 : ℝ)^3036 = x^2) := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_squares_l3272_327296


namespace NUMINAMATH_CALUDE_lindas_coins_l3272_327299

/-- Represents the number of coins Linda has initially and receives from her mother --/
structure CoinCounts where
  initial_dimes : Nat
  initial_quarters : Nat
  initial_nickels : Nat
  mother_dimes : Nat
  mother_quarters : Nat
  mother_nickels : Nat

/-- The theorem statement --/
theorem lindas_coins (c : CoinCounts) 
  (h1 : c.initial_dimes = 2)
  (h2 : c.initial_quarters = 6)
  (h3 : c.initial_nickels = 5)
  (h4 : c.mother_dimes = 2)
  (h5 : c.mother_nickels = 2 * c.initial_nickels)
  (h6 : c.initial_dimes + c.initial_quarters + c.initial_nickels + 
        c.mother_dimes + c.mother_quarters + c.mother_nickels = 35) :
  c.mother_quarters = 10 := by
  sorry

end NUMINAMATH_CALUDE_lindas_coins_l3272_327299


namespace NUMINAMATH_CALUDE_remaining_fuel_fraction_l3272_327203

def tank_capacity : ℚ := 12
def round_trip_distance : ℚ := 20
def miles_per_gallon : ℚ := 5

theorem remaining_fuel_fraction :
  (tank_capacity - round_trip_distance / miles_per_gallon) / tank_capacity = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_remaining_fuel_fraction_l3272_327203


namespace NUMINAMATH_CALUDE_no_given_factors_l3272_327243

def f (x : ℝ) : ℝ := x^5 + 3*x^3 - 4*x^2 + 12*x + 8

theorem no_given_factors :
  (∀ x, f x ≠ 0 → x + 1 ≠ 0) ∧
  (∀ x, f x ≠ 0 → x^2 + 1 ≠ 0) ∧
  (∀ x, f x ≠ 0 → x^2 - 2 ≠ 0) ∧
  (∀ x, f x ≠ 0 → x^2 + 3 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_no_given_factors_l3272_327243


namespace NUMINAMATH_CALUDE_dave_apps_unchanged_l3272_327254

theorem dave_apps_unchanged (initial_files final_files deleted_files final_apps : ℕ) :
  initial_files = final_files + deleted_files →
  initial_files = 24 →
  final_files = 21 →
  deleted_files = 3 →
  final_apps = 17 →
  initial_apps = final_apps :=
by
  sorry

#check dave_apps_unchanged

end NUMINAMATH_CALUDE_dave_apps_unchanged_l3272_327254


namespace NUMINAMATH_CALUDE_triangle_increase_l3272_327283

theorem triangle_increase (AB BC : ℝ) (h1 : AB = 24) (h2 : BC = 10) :
  let AC := Real.sqrt (AB^2 + BC^2)
  let AB' := AB + 6
  let BC' := BC + 6
  let AC' := Real.sqrt (AB'^2 + BC'^2)
  AC' - AC = 8 := by sorry

end NUMINAMATH_CALUDE_triangle_increase_l3272_327283


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3272_327270

theorem quadratic_equation_roots (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 - 2*x + 1 = 0 ∧ a * y^2 - 2*y + 1 = 0) → 
  (a < 1 ∧ a ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3272_327270


namespace NUMINAMATH_CALUDE_inequality_property_l3272_327206

theorem inequality_property (a b : ℝ) : a > b → -5 * a < -5 * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_property_l3272_327206


namespace NUMINAMATH_CALUDE_rhombus_and_rectangle_diagonals_bisect_l3272_327255

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define a rhombus
def is_rhombus (q : Quadrilateral) : Prop :=
  sorry

-- Define a rectangle
def is_rectangle (q : Quadrilateral) : Prop :=
  sorry

-- Define the property of diagonals bisecting each other
def diagonals_bisect_each_other (q : Quadrilateral) : Prop :=
  sorry

-- Theorem statement
theorem rhombus_and_rectangle_diagonals_bisect :
  ∀ q : Quadrilateral, 
    (is_rhombus q ∨ is_rectangle q) → diagonals_bisect_each_other q :=
by sorry

end NUMINAMATH_CALUDE_rhombus_and_rectangle_diagonals_bisect_l3272_327255


namespace NUMINAMATH_CALUDE_alloy_mixture_l3272_327260

/-- Proves that the amount of the first alloy used is 15 kg given the specified conditions -/
theorem alloy_mixture (x : ℝ) 
  (h1 : 0.12 * x + 0.08 * 35 = 0.092 * (x + 35)) : x = 15 := by
  sorry

#check alloy_mixture

end NUMINAMATH_CALUDE_alloy_mixture_l3272_327260


namespace NUMINAMATH_CALUDE_half_circle_is_300_clerts_l3272_327277

-- Define the number of clerts in a full circle
def full_circle_clerts : ℕ := 600

-- Define a half-circle as half of a full circle
def half_circle_clerts : ℕ := full_circle_clerts / 2

-- Theorem to prove
theorem half_circle_is_300_clerts : half_circle_clerts = 300 := by
  sorry

end NUMINAMATH_CALUDE_half_circle_is_300_clerts_l3272_327277


namespace NUMINAMATH_CALUDE_jellybean_problem_l3272_327262

theorem jellybean_problem :
  ∃ (n : ℕ), n = 151 ∧ 
  (∀ m : ℕ, m ≥ 150 ∧ m % 17 = 15 → m ≥ n) ∧ 
  n ≥ 150 ∧ 
  n % 17 = 15 := by
sorry

end NUMINAMATH_CALUDE_jellybean_problem_l3272_327262


namespace NUMINAMATH_CALUDE_focus_of_hyperbola_l3272_327293

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := y^2 / 3 - x^2 / 6 = 1

/-- Definition of a focus for this hyperbola -/
def is_focus (x y : ℝ) : Prop :=
  x^2 + y^2 = (3 : ℝ)^2 ∧ hyperbola x y

/-- Theorem: (0, 3) is a focus of the given hyperbola -/
theorem focus_of_hyperbola : is_focus 0 3 := by sorry

end NUMINAMATH_CALUDE_focus_of_hyperbola_l3272_327293


namespace NUMINAMATH_CALUDE_pretzel_ratio_l3272_327285

/-- The number of pretzels bought by Angie -/
def angie_pretzels : ℕ := 18

/-- The number of pretzels bought by Barry -/
def barry_pretzels : ℕ := 12

/-- The number of pretzels bought by Shelly -/
def shelly_pretzels : ℕ := angie_pretzels / 3

/-- Theorem stating the ratio of pretzels Shelly bought to pretzels Barry bought -/
theorem pretzel_ratio : 
  (shelly_pretzels : ℚ) / barry_pretzels = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_pretzel_ratio_l3272_327285


namespace NUMINAMATH_CALUDE_square_area_not_covered_by_circles_l3272_327214

/-- The area of a square not covered by circles -/
theorem square_area_not_covered_by_circles (side_length : ℝ) (num_circles : ℕ) : 
  side_length = 16 → num_circles = 9 → 
  side_length^2 - (num_circles : ℝ) * (side_length / 3)^2 * Real.pi = 256 - 64 * Real.pi := by
  sorry

#check square_area_not_covered_by_circles

end NUMINAMATH_CALUDE_square_area_not_covered_by_circles_l3272_327214


namespace NUMINAMATH_CALUDE_dataset_manipulation_result_l3272_327288

def calculate_final_dataset_size (initial_size : ℕ) : ℕ :=
  let size_after_increase := initial_size + (initial_size * 15 / 100)
  let size_after_addition := size_after_increase + 40
  let size_after_removal := size_after_addition - (size_after_addition / 6)
  let final_size := size_after_removal - (size_after_removal * 10 / 100)
  final_size

theorem dataset_manipulation_result :
  calculate_final_dataset_size 300 = 289 := by
  sorry

end NUMINAMATH_CALUDE_dataset_manipulation_result_l3272_327288


namespace NUMINAMATH_CALUDE_star_value_l3272_327239

theorem star_value (star : ℝ) : star * 12^2 = 12^7 → star = 12^5 := by
  sorry

end NUMINAMATH_CALUDE_star_value_l3272_327239


namespace NUMINAMATH_CALUDE_triangle_is_equilateral_l3272_327225

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def condition1 (t : Triangle) : Prop :=
  (t.a + t.b + t.c) * (t.a + t.b - t.c) = 3 * t.a * t.b

def condition2 (t : Triangle) : Prop :=
  2 * Real.cos t.A * Real.sin t.B = Real.sin t.C

-- Theorem statement
theorem triangle_is_equilateral (t : Triangle) 
  (h1 : condition1 t) (h2 : condition2 t) : 
  t.a = t.b ∧ t.b = t.c ∧ t.A = t.B ∧ t.B = t.C ∧ t.C = Real.pi / 3 :=
sorry

end NUMINAMATH_CALUDE_triangle_is_equilateral_l3272_327225


namespace NUMINAMATH_CALUDE_sphere_surface_area_ratio_l3272_327209

theorem sphere_surface_area_ratio (r1 r2 : ℝ) (h1 : r1 = 40) (h2 : r2 = 10) :
  (4 * π * r1^2) / (4 * π * r2^2) = 16 := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_ratio_l3272_327209


namespace NUMINAMATH_CALUDE_sufficient_condition_for_existence_l3272_327251

theorem sufficient_condition_for_existence (a : ℝ) :
  (a ≥ 2) →
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc 1 2 ∧ x₀^2 - a ≤ 0) ∧
  ¬((∃ x₀ : ℝ, x₀ ∈ Set.Icc 1 2 ∧ x₀^2 - a ≤ 0) → (a ≥ 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_existence_l3272_327251


namespace NUMINAMATH_CALUDE_dormitory_to_city_distance_l3272_327281

theorem dormitory_to_city_distance :
  ∀ (D : ℝ),
  (1 / 3 : ℝ) * D + (3 / 5 : ℝ) * D + 2 = D →
  D = 30 := by
sorry

end NUMINAMATH_CALUDE_dormitory_to_city_distance_l3272_327281


namespace NUMINAMATH_CALUDE_garden_bugs_l3272_327236

theorem garden_bugs (B : ℕ) : 0.8 * (B : ℝ) - 12 * 7 = 236 → B = 400 := by
  sorry

end NUMINAMATH_CALUDE_garden_bugs_l3272_327236


namespace NUMINAMATH_CALUDE_find_A_l3272_327263

/-- Represents a three-digit number of the form 2A3 where A is a single digit -/
def threeDigitNumber (A : Nat) : Nat :=
  200 + 10 * A + 3

/-- Condition that A is a single digit -/
def isSingleDigit (A : Nat) : Prop :=
  A ≥ 0 ∧ A ≤ 9

theorem find_A :
  ∀ A : Nat,
    isSingleDigit A →
    (threeDigitNumber A).mod 11 = 0 →
    A = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_find_A_l3272_327263


namespace NUMINAMATH_CALUDE_find_N_l3272_327276

theorem find_N : ∀ N : ℕ, (1 + 2 + 3) / 6 = (1988 + 1989 + 1990) / N → N = 5967 := by
  sorry

end NUMINAMATH_CALUDE_find_N_l3272_327276


namespace NUMINAMATH_CALUDE_page_number_added_twice_l3272_327221

theorem page_number_added_twice (n : ℕ) (h1 : n > 0) : 
  (∃ p : ℕ, p ≤ n ∧ n * (n + 1) / 2 + p = 2630) → 
  (∃ p : ℕ, p ≤ n ∧ n * (n + 1) / 2 + p = 2630 ∧ p = 2) :=
by sorry

end NUMINAMATH_CALUDE_page_number_added_twice_l3272_327221


namespace NUMINAMATH_CALUDE_monotonic_f_implies_a_range_l3272_327294

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

-- Define the property of being monotonic in an interval
def isMonotonicIn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → (f x < f y ∨ f y < f x)

-- Theorem statement
theorem monotonic_f_implies_a_range (a : ℝ) :
  isMonotonicIn (f a) 1 2 → a ≤ -1 ∨ a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_f_implies_a_range_l3272_327294


namespace NUMINAMATH_CALUDE_S_is_empty_l3272_327216

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the set S
def S : Set ℂ := {z : ℂ | ∃ r : ℝ, (2 + 5*i)*z = r ∧ z.re = 2*z.im}

-- Theorem statement
theorem S_is_empty : S = ∅ := by sorry

end NUMINAMATH_CALUDE_S_is_empty_l3272_327216


namespace NUMINAMATH_CALUDE_hyperbola_imaginary_axis_length_l3272_327259

/-- A hyperbola with semi-major axis a, semi-minor axis b, and eccentricity e -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  e : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  e_def : e = (a^2 + b^2).sqrt / a

theorem hyperbola_imaginary_axis_length 
  (h : Hyperbola) 
  (dist_foci : ∃ (p : ℝ × ℝ), p.1^2 / h.a^2 - p.2^2 / h.b^2 = 1 ∧ 
    ∃ (f₁ f₂ : ℝ × ℝ), (p.1 - f₁.1)^2 + (p.2 - f₁.2)^2 = 100 ∧ 
                        (p.1 - f₂.1)^2 + (p.2 - f₂.2)^2 = 16) 
  (h_e : h.e = 2) : 
  2 * h.b = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_imaginary_axis_length_l3272_327259


namespace NUMINAMATH_CALUDE_tennis_ball_order_l3272_327264

/-- The number of tennis balls originally ordered by a sports retailer -/
def original_order : ℕ := 288

/-- The number of extra yellow balls sent by mistake -/
def extra_yellow_balls : ℕ := 90

/-- The ratio of white balls to yellow balls after the error -/
def final_ratio : Rat := 8 / 13

theorem tennis_ball_order :
  ∃ (white yellow : ℕ),
    -- The retailer ordered equal numbers of white and yellow tennis balls
    white = yellow ∧
    -- The total original order
    white + yellow = original_order ∧
    -- After the error, the ratio of white to yellow balls is 8/13
    (white : Rat) / ((yellow : Rat) + extra_yellow_balls) = final_ratio :=
by sorry

end NUMINAMATH_CALUDE_tennis_ball_order_l3272_327264


namespace NUMINAMATH_CALUDE_spider_legs_proof_l3272_327234

/-- The number of legs a single spider has -/
def spider_legs : ℕ := 8

/-- The number of spiders in the group -/
def group_size (L : ℕ) : ℕ := L / 2 + 10

theorem spider_legs_proof :
  (∀ L : ℕ, group_size L * L = 112) → spider_legs = 8 := by
  sorry

end NUMINAMATH_CALUDE_spider_legs_proof_l3272_327234


namespace NUMINAMATH_CALUDE_car_payment_months_l3272_327218

def car_price : ℕ := 13380
def initial_payment : ℕ := 5400
def monthly_payment : ℕ := 420

theorem car_payment_months : 
  (car_price - initial_payment) / monthly_payment = 19 := by
  sorry

end NUMINAMATH_CALUDE_car_payment_months_l3272_327218


namespace NUMINAMATH_CALUDE_sum_of_digits_1_to_5000_l3272_327258

def sum_of_digits (n : ℕ) : ℕ := sorry

def sequence_sum (n : ℕ) : ℕ := sorry

theorem sum_of_digits_1_to_5000 : 
  sequence_sum 5000 = 194450 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_1_to_5000_l3272_327258


namespace NUMINAMATH_CALUDE_proof_by_contradiction_assumption_l3272_327237

theorem proof_by_contradiction_assumption (a b : ℤ) : 
  (5 ∣ a * b) → (5 ∣ a ∨ 5 ∣ b) ↔ 
  (¬(5 ∣ a) ∧ ¬(5 ∣ b)) → False :=
by sorry

end NUMINAMATH_CALUDE_proof_by_contradiction_assumption_l3272_327237


namespace NUMINAMATH_CALUDE_sum_of_cube_edges_l3272_327227

-- Define a cube with edge length 15
def cube_edge_length : ℝ := 15

-- Define the number of edges in a cube
def cube_num_edges : ℕ := 12

-- Theorem: The sum of all edge lengths in the cube is 180
theorem sum_of_cube_edges :
  cube_edge_length * cube_num_edges = 180 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cube_edges_l3272_327227


namespace NUMINAMATH_CALUDE_preston_received_correct_amount_l3272_327226

/-- Calculates the total amount Preston received from Abra Company's order --/
def prestonReceived (
  sandwichPrice : ℚ)
  (sideDishPrice : ℚ)
  (drinkPrice : ℚ)
  (deliveryFee : ℚ)
  (sandwichCount : ℕ)
  (sideDishCount : ℕ)
  (drinkCount : ℕ)
  (tipPercentage : ℚ)
  (discountPercentage : ℚ) : ℚ :=
  let foodCost := sandwichPrice * sandwichCount + sideDishPrice * sideDishCount
  let drinkCost := drinkPrice * drinkCount
  let discountAmount := discountPercentage * foodCost
  let subtotal := foodCost + drinkCost - discountAmount + deliveryFee
  let tipAmount := tipPercentage * subtotal
  subtotal + tipAmount

/-- Theorem stating that Preston received $158.95 from Abra Company's order --/
theorem preston_received_correct_amount :
  prestonReceived 5 3 (3/2) 20 18 10 15 (1/10) (15/100) = 15895/100 := by
  sorry

end NUMINAMATH_CALUDE_preston_received_correct_amount_l3272_327226


namespace NUMINAMATH_CALUDE_sequence_property_l3272_327205

/-- A sequence satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℚ) : Prop :=
  ∀ n, a n - a (n + 1) = (a n * a (n + 1)) / (2^(n - 1))

theorem sequence_property (a : ℕ → ℚ) (k : ℕ) 
  (h1 : RecurrenceSequence a) 
  (h2 : a 2 = -1)
  (h3 : a k = 16 * a 8)
  (h4 : k > 0) : 
  k = 12 := by
  sorry

end NUMINAMATH_CALUDE_sequence_property_l3272_327205


namespace NUMINAMATH_CALUDE_certain_event_at_least_one_genuine_l3272_327256

theorem certain_event_at_least_one_genuine :
  ∀ (total : ℕ) (genuine : ℕ) (defective : ℕ) (selected : ℕ),
    total = 12 →
    genuine = 10 →
    defective = 2 →
    total = genuine + defective →
    selected = 3 →
    (∀ outcome : Finset (Fin total),
      outcome.card = selected →
      ∃ i ∈ outcome, i.val < genuine) :=
by sorry

end NUMINAMATH_CALUDE_certain_event_at_least_one_genuine_l3272_327256
