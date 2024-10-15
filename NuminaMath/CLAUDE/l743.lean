import Mathlib

namespace NUMINAMATH_CALUDE_power_zero_minus_pi_l743_74356

theorem power_zero_minus_pi (x : ℝ) : (x - Real.pi) ^ (0 : ℕ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_zero_minus_pi_l743_74356


namespace NUMINAMATH_CALUDE_unique_egyptian_fraction_representation_l743_74335

theorem unique_egyptian_fraction_representation (p : ℕ) (h_prime : Nat.Prime p) (h_p_gt_2 : p > 2) :
  ∃! (x y : ℕ), x ≠ y ∧ x > 0 ∧ y > 0 ∧ (2 : ℚ) / p = 1 / x + 1 / y := by
  sorry

end NUMINAMATH_CALUDE_unique_egyptian_fraction_representation_l743_74335


namespace NUMINAMATH_CALUDE_ellipse_properties_hyperbola_properties_l743_74322

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/2 = 1

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/4 = 1

-- Theorem for the ellipse
theorem ellipse_properties :
  ∀ x y : ℝ, ellipse x y →
  (∃ c : ℝ, c = Real.sqrt 2 ∧ 
   ((x - c)^2 + y^2 = 4 ∨ (x + c)^2 + y^2 = 4)) ∧
  (x = -2 * Real.sqrt 2 ∨ x = 2 * Real.sqrt 2) :=
sorry

-- Theorem for the hyperbola
theorem hyperbola_properties :
  ∀ x y : ℝ, hyperbola x y →
  (hyperbola (Real.sqrt 2) 2) ∧
  (∃ k : ℝ, k = 2 ∧ (y = k*x ∨ y = -k*x)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_hyperbola_properties_l743_74322


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l743_74399

-- Define the function f
def f (a x : ℝ) : ℝ := |x + a| + |x - 1|

-- Part 1: Solution set when a = 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x < 3} = Set.Ioo (-3/2) (3/2) := by sorry

-- Part 2: Range of a for which f(x) ≥ 3 for all x
theorem range_of_a_part2 :
  ∀ a : ℝ, (∀ x : ℝ, f a x ≥ 3) ↔ (a ≥ 2 ∨ a ≤ -4) := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l743_74399


namespace NUMINAMATH_CALUDE_total_payment_calculation_l743_74364

def worker_count : ℝ := 2.5
def hourly_rate : ℝ := 15
def daily_hours : List ℝ := [12, 10, 8, 6, 14]

theorem total_payment_calculation :
  worker_count * hourly_rate * (daily_hours.sum) = 1875 := by sorry

end NUMINAMATH_CALUDE_total_payment_calculation_l743_74364


namespace NUMINAMATH_CALUDE_min_sum_squares_l743_74368

theorem min_sum_squares (x y : ℝ) :
  x^2 - y^2 + 6*x + 4*y + 5 = 0 →
  ∃ (min : ℝ), min = 0.5 ∧ ∀ (a b : ℝ), a^2 - b^2 + 6*a + 4*b + 5 = 0 → a^2 + b^2 ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l743_74368


namespace NUMINAMATH_CALUDE_rectangular_field_perimeter_l743_74310

def rectangle_perimeter (length width : ℝ) : ℝ :=
  2 * (length + width)

theorem rectangular_field_perimeter :
  let length : ℝ := 15
  let width : ℝ := 20
  rectangle_perimeter length width = 70 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_perimeter_l743_74310


namespace NUMINAMATH_CALUDE_comparison_of_special_points_l743_74361

theorem comparison_of_special_points (a b c : Real) 
  (ha : 0 < a ∧ a < Real.pi / 2)
  (hb : 0 < b ∧ b < Real.pi / 2)
  (hc : 0 < c ∧ c < Real.pi / 2)
  (eq_a : a = Real.cos a)
  (eq_b : b = Real.sin (Real.cos b))
  (eq_c : c = Real.cos (Real.sin c)) :
  b < a ∧ a < c :=
by sorry

end NUMINAMATH_CALUDE_comparison_of_special_points_l743_74361


namespace NUMINAMATH_CALUDE_vector_expression_inequality_l743_74353

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Given arbitrary points P, A, B, Q in a real vector space V, 
    the expression PA + AB - BQ is not always equal to PQ. -/
theorem vector_expression_inequality (P A B Q : V) :
  ¬ (∀ (P A B Q : V), (A - P) + (B - A) - (Q - B) = Q - P) :=
sorry

end NUMINAMATH_CALUDE_vector_expression_inequality_l743_74353


namespace NUMINAMATH_CALUDE_intersection_point_of_perpendicular_chords_l743_74343

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define a line
def line (m b x y : ℝ) : Prop := x = m*y + b

-- Define perpendicularity of two points with respect to the origin
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁*x₂ + y₁*y₂ = 0

theorem intersection_point_of_perpendicular_chords :
  ∀ (m b x₁ y₁ x₂ y₂ : ℝ),
  parabola x₁ y₁ →
  parabola x₂ y₂ →
  line m b x₁ y₁ →
  line m b x₂ y₂ →
  perpendicular x₁ y₁ x₂ y₂ →
  ∃ (x y : ℝ), line m b x y ∧ x = 2 ∧ y = 0 :=
sorry

end NUMINAMATH_CALUDE_intersection_point_of_perpendicular_chords_l743_74343


namespace NUMINAMATH_CALUDE_nicky_dmv_wait_l743_74327

/-- The time Nicky spent waiting to take a number, in minutes. -/
def initial_wait : ℕ := 20

/-- The time Nicky spent waiting for his number to be called, in minutes. -/
def number_wait : ℕ := 4 * initial_wait + 14

/-- The total time Nicky spent waiting at the DMV, in minutes. -/
def total_wait : ℕ := initial_wait + number_wait

theorem nicky_dmv_wait : total_wait = 114 := by
  sorry

end NUMINAMATH_CALUDE_nicky_dmv_wait_l743_74327


namespace NUMINAMATH_CALUDE_sin_translation_l743_74381

/-- Given a function f obtained by translating the graph of y = sin 2x
    1 unit left and 1 unit upward, prove that f(x) = sin(2x+2)+1 for all real x. -/
theorem sin_translation (f : ℝ → ℝ) 
  (h : ∀ x, f x = (fun y ↦ Real.sin (2 * y)) (x + 1) + 1) :
  ∀ x, f x = Real.sin (2 * x + 2) + 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_translation_l743_74381


namespace NUMINAMATH_CALUDE_father_double_son_age_l743_74373

/-- Represents the ages of a father and son, and the time until the father's age is twice the son's. -/
structure FatherSonAges where
  sonAge : ℕ
  fatherAge : ℕ
  yearsUntilDouble : ℕ

/-- The condition that the father is 25 years older than the son -/
def ageDifference (ages : FatherSonAges) : Prop :=
  ages.fatherAge = ages.sonAge + 25

/-- The condition that after a certain number of years, the father's age will be twice the son's -/
def doubleAgeCondition (ages : FatherSonAges) : Prop :=
  ages.fatherAge + ages.yearsUntilDouble = 2 * (ages.sonAge + ages.yearsUntilDouble)

/-- The main theorem stating that given the initial conditions, it will take 2 years for the father's age to be twice the son's -/
theorem father_double_son_age :
  ∀ (ages : FatherSonAges),
  ages.sonAge = 23 →
  ageDifference ages →
  doubleAgeCondition ages →
  ages.yearsUntilDouble = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_father_double_son_age_l743_74373


namespace NUMINAMATH_CALUDE_total_tiles_on_floor_l743_74371

/-- Represents a square floor with a border of tiles -/
structure BorderedFloor where
  /-- The side length of the square floor -/
  side_length : ℕ
  /-- The width of the border in tiles -/
  border_width : ℕ
  /-- The number of tiles in the border -/
  border_tiles : ℕ

/-- Theorem: Given a square floor with a 1-tile wide black border containing 204 tiles, 
    the total number of tiles on the floor is 2704 -/
theorem total_tiles_on_floor (floor : BorderedFloor) 
  (h1 : floor.border_width = 1)
  (h2 : floor.border_tiles = 204) : 
  floor.side_length^2 = 2704 := by
  sorry

#check total_tiles_on_floor

end NUMINAMATH_CALUDE_total_tiles_on_floor_l743_74371


namespace NUMINAMATH_CALUDE_equation_solution_l743_74306

theorem equation_solution (x : ℝ) : 
  (Real.sqrt ((3 + Real.sqrt 5) ^ x)) ^ 2 + (Real.sqrt ((3 - Real.sqrt 5) ^ x)) ^ 2 = 18 ↔ 
  x = 2 ∨ x = -2 :=
sorry

end NUMINAMATH_CALUDE_equation_solution_l743_74306


namespace NUMINAMATH_CALUDE_correct_calculation_l743_74384

theorem correct_calculation (x : ℝ) (h : 21 * x = 63) : x + 40 = 43 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l743_74384


namespace NUMINAMATH_CALUDE_polynomial_simplification_l743_74377

theorem polynomial_simplification (y : ℝ) :
  (3 * y - 2) * (5 * y^12 + 3 * y^11 + y^10 + 2 * y^9) =
  15 * y^13 - y^12 - 3 * y^11 + 4 * y^10 - 4 * y^9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l743_74377


namespace NUMINAMATH_CALUDE_exactly_one_negative_l743_74325

theorem exactly_one_negative 
  (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ) 
  (hx₁ : x₁ ≠ 0) (hx₂ : x₂ ≠ 0) (hx₃ : x₃ ≠ 0) 
  (hy₁ : y₁ ≠ 0) (hy₂ : y₂ ≠ 0) (hy₃ : y₃ ≠ 0) 
  (v₁ : ℝ) (hv₁ : v₁ = x₁ + y₁)
  (v₂ : ℝ) (hv₂ : v₂ = x₂ + y₂)
  (v₃ : ℝ) (hv₃ : v₃ = x₃ + y₃)
  (h_prod : x₁ * x₂ * x₃ = -(y₁ * y₂ * y₃))
  (h_sum_squares : x₁^2 + x₂^2 + x₃^2 = y₁^2 + y₂^2 + y₃^2)
  (h_triangle : v₁ + v₂ ≥ v₃ ∧ v₂ + v₃ ≥ v₁ ∧ v₃ + v₁ ≥ v₂)
  (h_triangle_squares : v₁^2 + v₂^2 ≥ v₃^2 ∧ v₂^2 + v₃^2 ≥ v₁^2 ∧ v₃^2 + v₁^2 ≥ v₂^2) :
  (x₁ < 0 ∨ x₂ < 0 ∨ x₃ < 0 ∨ y₁ < 0 ∨ y₂ < 0 ∨ y₃ < 0) ∧
  ¬(x₁ < 0 ∧ x₂ < 0) ∧ ¬(x₁ < 0 ∧ x₃ < 0) ∧ ¬(x₂ < 0 ∧ x₃ < 0) ∧
  ¬(y₁ < 0 ∧ y₂ < 0) ∧ ¬(y₁ < 0 ∧ y₃ < 0) ∧ ¬(y₂ < 0 ∧ y₃ < 0) ∧
  ¬(x₁ < 0 ∧ y₁ < 0) ∧ ¬(x₁ < 0 ∧ y₂ < 0) ∧ ¬(x₁ < 0 ∧ y₃ < 0) ∧
  ¬(x₂ < 0 ∧ y₁ < 0) ∧ ¬(x₂ < 0 ∧ y₂ < 0) ∧ ¬(x₂ < 0 ∧ y₃ < 0) ∧
  ¬(x₃ < 0 ∧ y₁ < 0) ∧ ¬(x₃ < 0 ∧ y₂ < 0) ∧ ¬(x₃ < 0 ∧ y₃ < 0) := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_negative_l743_74325


namespace NUMINAMATH_CALUDE_election_ratio_l743_74378

theorem election_ratio (X Y : ℝ) 
  (h1 : 0.74 * X + 0.5000000000000002 * Y = 0.66 * (X + Y)) 
  (h2 : X > 0) 
  (h3 : Y > 0) : 
  X / Y = 2 := by
sorry

end NUMINAMATH_CALUDE_election_ratio_l743_74378


namespace NUMINAMATH_CALUDE_ticket_distribution_count_l743_74389

/-- The number of ways to distribute 5 consecutive movie tickets to 5 people. -/
def distribute_tickets : ℕ :=
  /- Number of ways to group tickets -/ 4 *
  /- Number of ways to order A and B -/ 2 *
  /- Number of ways to permute remaining tickets -/ 6

/-- Theorem stating that there are 48 ways to distribute the tickets. -/
theorem ticket_distribution_count :
  distribute_tickets = 48 := by
  sorry

end NUMINAMATH_CALUDE_ticket_distribution_count_l743_74389


namespace NUMINAMATH_CALUDE_yonderland_license_plates_l743_74341

/-- The number of possible letters in each position of a license plate. -/
def numLetters : ℕ := 26

/-- The number of possible digits in each position of a license plate. -/
def numDigits : ℕ := 10

/-- The number of letter positions in a license plate. -/
def numLetterPositions : ℕ := 3

/-- The number of digit positions in a license plate. -/
def numDigitPositions : ℕ := 4

/-- The total number of valid license plates in Yonderland. -/
def totalLicensePlates : ℕ := numLetters ^ numLetterPositions * numDigits ^ numDigitPositions

theorem yonderland_license_plates : totalLicensePlates = 175760000 := by
  sorry

end NUMINAMATH_CALUDE_yonderland_license_plates_l743_74341


namespace NUMINAMATH_CALUDE_unique_y_for_diamond_eq_21_l743_74350

def diamond (x y : ℝ) : ℝ := 5 * x - 4 * y + 2 * x * y + 1

theorem unique_y_for_diamond_eq_21 : ∃! y : ℝ, diamond 4 y = 21 := by
  sorry

end NUMINAMATH_CALUDE_unique_y_for_diamond_eq_21_l743_74350


namespace NUMINAMATH_CALUDE_evaluate_expression_l743_74397

theorem evaluate_expression : 6 - 8 * (9 - 4^2) * 5 = 286 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l743_74397


namespace NUMINAMATH_CALUDE_sum_squares_formula_l743_74313

theorem sum_squares_formula (m n : ℝ) (h : m + n = 3) : 
  2*m^2 + 4*m*n + 2*n^2 - 6 = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_squares_formula_l743_74313


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l743_74344

theorem p_sufficient_not_necessary_for_q :
  ∃ (a : ℝ), (a = 1 → abs a = 1) ∧ (abs a = 1 → a = 1 → False) := by
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l743_74344


namespace NUMINAMATH_CALUDE_sum_congruent_to_6_mod_9_l743_74391

def sum : ℕ := 2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999

theorem sum_congruent_to_6_mod_9 : sum % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_congruent_to_6_mod_9_l743_74391


namespace NUMINAMATH_CALUDE_art_class_problem_l743_74317

theorem art_class_problem (total_students : ℕ) (total_artworks : ℕ) 
  (first_half_artworks_per_student : ℕ) :
  total_students = 10 →
  total_artworks = 35 →
  first_half_artworks_per_student = 3 →
  ∃ (second_half_artworks_per_student : ℕ),
    (total_students / 2 * first_half_artworks_per_student) + 
    (total_students / 2 * second_half_artworks_per_student) = total_artworks ∧
    second_half_artworks_per_student = 4 :=
by sorry

end NUMINAMATH_CALUDE_art_class_problem_l743_74317


namespace NUMINAMATH_CALUDE_mindy_income_multiple_l743_74379

/-- Proves that Mindy earned 3 times more than Mork given their tax rates and combined tax rate -/
theorem mindy_income_multiple (mork_rate mindy_rate combined_rate : ℚ) : 
  mork_rate = 40/100 →
  mindy_rate = 30/100 →
  combined_rate = 325/1000 →
  ∃ k : ℚ, k = 3 ∧ 
    (mork_rate + k * mindy_rate) / (1 + k) = combined_rate :=
by sorry

end NUMINAMATH_CALUDE_mindy_income_multiple_l743_74379


namespace NUMINAMATH_CALUDE_math_score_difference_l743_74383

def regression_equation (x : ℝ) : ℝ := 6 + 0.4 * x

theorem math_score_difference (x₁ x₂ : ℝ) (h : x₂ - x₁ = 50) :
  regression_equation x₂ - regression_equation x₁ = 20 := by
  sorry

end NUMINAMATH_CALUDE_math_score_difference_l743_74383


namespace NUMINAMATH_CALUDE_parallelogram_roots_theorem_l743_74302

/-- The polynomial in question -/
def polynomial (b : ℝ) (z : ℂ) : ℂ :=
  z^4 - 8*z^3 + 13*b*z^2 - 5*(2*b^2 + 4*b - 4)*z + 4

/-- Predicate to check if four complex numbers form a parallelogram -/
def form_parallelogram (z₁ z₂ z₃ z₄ : ℂ) : Prop :=
  (z₁ + z₃ = z₂ + z₄) ∧ (z₁ - z₂ = z₄ - z₃)

/-- The main theorem -/
theorem parallelogram_roots_theorem :
  ∃! (b : ℝ), b = (3/2) ∧
  ∃ (z₁ z₂ z₃ z₄ : ℂ),
    (polynomial b z₁ = 0) ∧
    (polynomial b z₂ = 0) ∧
    (polynomial b z₃ = 0) ∧
    (polynomial b z₄ = 0) ∧
    form_parallelogram z₁ z₂ z₃ z₄ :=
by
  sorry

end NUMINAMATH_CALUDE_parallelogram_roots_theorem_l743_74302


namespace NUMINAMATH_CALUDE_multiply_and_simplify_l743_74387

theorem multiply_and_simplify (x : ℝ) : (x^4 + 12*x^2 + 144) * (x^2 - 12) = x^6 - 1728 := by
  sorry

end NUMINAMATH_CALUDE_multiply_and_simplify_l743_74387


namespace NUMINAMATH_CALUDE_equation_solutions_l743_74318

theorem equation_solutions :
  (∃ x : ℝ, 0.4 * x = -1.2 * x + 1.6 ∧ x = 1) ∧
  (∃ y : ℝ, (1/3) * (y + 2) = 1 - (1/6) * (2 * y - 1) ∧ y = 3/4) := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l743_74318


namespace NUMINAMATH_CALUDE_count_five_digit_numbers_with_one_odd_l743_74320

/-- The count of five-digit numbers with exactly one odd digit -/
def five_digit_numbers_with_one_odd : ℕ :=
  let odd_digits := 5  -- Count of odd digits (1, 3, 5, 7, 9)
  let even_digits := 5  -- Count of even digits (0, 2, 4, 6, 8)
  let first_digit_odd := odd_digits * even_digits^4
  let other_digit_odd := 4 * odd_digits * (even_digits - 1) * even_digits^3
  first_digit_odd + other_digit_odd

theorem count_five_digit_numbers_with_one_odd :
  five_digit_numbers_with_one_odd = 10625 := by
  sorry

end NUMINAMATH_CALUDE_count_five_digit_numbers_with_one_odd_l743_74320


namespace NUMINAMATH_CALUDE_deer_distribution_l743_74326

theorem deer_distribution (a₁ : ℚ) (d : ℚ) :
  a₁ = 5/3 ∧ 
  5 * a₁ + (5 * 4)/2 * d = 5 →
  a₁ + 2*d = 1 :=
by sorry

end NUMINAMATH_CALUDE_deer_distribution_l743_74326


namespace NUMINAMATH_CALUDE_sum_of_digits_odd_numbers_to_10000_l743_74398

/-- Sum of digits function for natural numbers -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Predicate to check if a natural number is odd -/
def isOdd (n : ℕ) : Prop := n % 2 = 1

/-- The sum of digits of all odd numbers from 1 to 10000 -/
def sumOfDigitsOddNumbers : ℕ := sorry

/-- Theorem stating that the sum of digits of all odd numbers from 1 to 10000 is 97500 -/
theorem sum_of_digits_odd_numbers_to_10000 :
  sumOfDigitsOddNumbers = 97500 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_odd_numbers_to_10000_l743_74398


namespace NUMINAMATH_CALUDE_peach_tart_fraction_l743_74385

theorem peach_tart_fraction (total : ℝ) (cherry : ℝ) (blueberry : ℝ) 
  (h1 : total = 0.91)
  (h2 : cherry = 0.08)
  (h3 : blueberry = 0.75) :
  total - (cherry + blueberry) = 0.08 := by
  sorry

end NUMINAMATH_CALUDE_peach_tart_fraction_l743_74385


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l743_74331

theorem polynomial_division_theorem (x : ℝ) :
  x^5 - 22*x^3 + 12*x^2 - 16*x + 8 = (x - 3) * (x^4 + 3*x^3 - 13*x^2 - 27*x - 97) + (-211) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l743_74331


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l743_74328

theorem polynomial_divisibility (m n : ℤ) :
  (∀ (x y : ℤ), (107 ∣ (x^3 + m*x + n) - (y^3 + m*y + n)) → (107 ∣ (x - y))) →
  (107 ∣ m) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l743_74328


namespace NUMINAMATH_CALUDE_amusement_park_tickets_l743_74369

/-- Proves that the number of children in the group is 15 given the specified conditions -/
theorem amusement_park_tickets (total_cost adult_price child_price adult_child_difference : ℕ) 
  (h1 : total_cost = 720)
  (h2 : adult_price = 15)
  (h3 : child_price = 8)
  (h4 : adult_child_difference = 25) :
  ∃ (num_children : ℕ), 
    (num_children + adult_child_difference) * adult_price + num_children * child_price = total_cost ∧ 
    num_children = 15 := by
  sorry

end NUMINAMATH_CALUDE_amusement_park_tickets_l743_74369


namespace NUMINAMATH_CALUDE_pizza_price_problem_l743_74375

/-- Proves the price of large pizza slices given the conditions of the problem -/
theorem pizza_price_problem (small_price : ℕ) (total_slices : ℕ) (total_revenue : ℕ) (small_slices : ℕ) :
  small_price = 150 →
  total_slices = 5000 →
  total_revenue = 1050000 →
  small_slices = 2000 →
  (total_revenue - small_price * small_slices) / (total_slices - small_slices) = 250 := by
sorry

end NUMINAMATH_CALUDE_pizza_price_problem_l743_74375


namespace NUMINAMATH_CALUDE_count_sevens_up_to_2017_l743_74394

/-- Count of digit 7 in a natural number -/
def count_sevens (n : ℕ) : ℕ := sorry

/-- Sum of count_sevens for all numbers from 1 to n -/
def sum_count_sevens (n : ℕ) : ℕ := sorry

/-- The main theorem stating the count of digit 7 in numbers from 1 to 2017 -/
theorem count_sevens_up_to_2017 : sum_count_sevens 2017 = 602 := by sorry

end NUMINAMATH_CALUDE_count_sevens_up_to_2017_l743_74394


namespace NUMINAMATH_CALUDE_jennas_tanning_schedule_l743_74342

/-- Jenna's tanning schedule problem -/
theorem jennas_tanning_schedule :
  ∀ (x : ℝ),
  (x ≥ 0) →  -- Non-negative tanning time
  (4 * x + 80 ≤ 200) →  -- Total tanning time constraint
  (x = 30) :=  -- Prove that x is 30 minutes
by
  sorry

end NUMINAMATH_CALUDE_jennas_tanning_schedule_l743_74342


namespace NUMINAMATH_CALUDE_rectangle_max_area_l743_74396

theorem rectangle_max_area (length width : ℝ) :
  length > 0 → width > 0 → length + width = 18 →
  length * width ≤ 81 ∧
  (length * width = 81 ↔ length = 9 ∧ width = 9) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l743_74396


namespace NUMINAMATH_CALUDE_tetromino_tiling_divisibility_l743_74346

/-- Represents a T-tetromino tile -/
structure TTetromino :=
  (size : Nat)
  (shape : Unit)
  (h_size : size = 4)

/-- Represents a rectangle that can be tiled with T-tetrominoes -/
structure TileableRectangle :=
  (m n : Nat)
  (tiles : List TTetromino)
  (h_tiling : tiles.length * 4 = m * n)  -- Complete tiling without gaps or overlaps

/-- 
If a rectangle can be tiled with T-tetrominoes, then its dimensions are divisible by 4 
-/
theorem tetromino_tiling_divisibility (rect : TileableRectangle) : 
  4 ∣ rect.m ∧ 4 ∣ rect.n :=
sorry

end NUMINAMATH_CALUDE_tetromino_tiling_divisibility_l743_74346


namespace NUMINAMATH_CALUDE_units_digit_G_1000_l743_74366

/-- The function G_n defined as 3^(3^n) + 1 -/
def G (n : ℕ) : ℕ := 3^(3^n) + 1

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- Theorem stating that the units digit of G(1000) is 2 -/
theorem units_digit_G_1000 : unitsDigit (G 1000) = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_G_1000_l743_74366


namespace NUMINAMATH_CALUDE_stairs_fibonacci_equivalence_nine_steps_ways_l743_74324

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fibonacci n + fibonacci (n + 1)

def climbStairs : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => climbStairs n + climbStairs (n + 1)

theorem stairs_fibonacci_equivalence (n : ℕ) : climbStairs n = fibonacci (n + 1) := by
  sorry

theorem nine_steps_ways : climbStairs 9 = 55 := by
  sorry

end NUMINAMATH_CALUDE_stairs_fibonacci_equivalence_nine_steps_ways_l743_74324


namespace NUMINAMATH_CALUDE_absolute_difference_bound_l743_74388

theorem absolute_difference_bound (x y s t : ℝ) 
  (hx : |x - s| < t) (hy : |y - s| < t) : |x - y| < 2*t := by
  sorry

end NUMINAMATH_CALUDE_absolute_difference_bound_l743_74388


namespace NUMINAMATH_CALUDE_probability_two_red_cards_modified_deck_l743_74300

/-- A modified deck of cards -/
structure ModifiedDeck :=
  (total_cards : ℕ)
  (red_cards : ℕ)
  (suits : ℕ)
  (cards_per_suit : ℕ)

/-- The probability of drawing two red cards in succession from the modified deck -/
def probability_two_red_cards (deck : ModifiedDeck) : ℚ :=
  (deck.red_cards * (deck.red_cards - 1)) / (deck.total_cards * (deck.total_cards - 1))

/-- Theorem stating the probability of drawing two red cards from the modified deck -/
theorem probability_two_red_cards_modified_deck :
  ∃ (deck : ModifiedDeck),
    deck.total_cards = 60 ∧
    deck.red_cards = 24 ∧
    deck.suits = 5 ∧
    deck.cards_per_suit = 12 ∧
    probability_two_red_cards deck = 92 / 590 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_red_cards_modified_deck_l743_74300


namespace NUMINAMATH_CALUDE_root_sum_theorem_l743_74376

theorem root_sum_theorem (a b : ℝ) (ha : a ≠ 0) (h : a^2 + b*a - 2*a = 0) : a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l743_74376


namespace NUMINAMATH_CALUDE_bus_car_speed_problem_l743_74382

/-- Proves that given the conditions of the problem, the bus speed is 50 km/h and the car speed is 75 km/h -/
theorem bus_car_speed_problem (distance : ℝ) (delay : ℝ) (speed_ratio : ℝ) 
  (h1 : distance = 50)
  (h2 : delay = 1/3)
  (h3 : speed_ratio = 1.5)
  (h4 : ∀ (bus_speed : ℝ), bus_speed > 0 → 
    distance / bus_speed - distance / (speed_ratio * bus_speed) = delay) :
  ∃ (bus_speed car_speed : ℝ),
    bus_speed = 50 ∧ 
    car_speed = 75 ∧
    car_speed = speed_ratio * bus_speed :=
by sorry

end NUMINAMATH_CALUDE_bus_car_speed_problem_l743_74382


namespace NUMINAMATH_CALUDE_square_side_increase_percentage_l743_74340

theorem square_side_increase_percentage (a : ℝ) (x : ℝ) :
  (a > 0) →
  (x > 0) →
  (a * (1 + x / 100) * 1.8)^2 = 2.592 * (a^2 + (a * (1 + x / 100))^2) →
  x = 100 := by sorry

end NUMINAMATH_CALUDE_square_side_increase_percentage_l743_74340


namespace NUMINAMATH_CALUDE_candy_problem_l743_74349

theorem candy_problem (total_candies : ℕ) : 
  (∃ (n : ℕ), n > 10 ∧ total_candies = 3 * (n - 1) + 2) ∧ 
  (∃ (m : ℕ), m < 10 ∧ total_candies = 4 * (m - 1) + 3) →
  total_candies = 35 := by
sorry

end NUMINAMATH_CALUDE_candy_problem_l743_74349


namespace NUMINAMATH_CALUDE_sequence_2003_l743_74314

theorem sequence_2003 (a : ℕ → ℕ) (h1 : a 1 = 0) (h2 : ∀ n : ℕ, a (n + 1) = a n + 2 * n) : 
  a 2003 = 2003 * 2002 := by
sorry

end NUMINAMATH_CALUDE_sequence_2003_l743_74314


namespace NUMINAMATH_CALUDE_xyz_sum_product_bounds_l743_74315

theorem xyz_sum_product_bounds (x y z : ℝ) : 
  5 * (x + y + z) = x^2 + y^2 + z^2 → 
  ∃ (M m : ℝ), 
    (∀ a b c : ℝ, 5 * (a + b + c) = a^2 + b^2 + c^2 → 
      a * b + a * c + b * c ≤ M) ∧
    (∀ a b c : ℝ, 5 * (a + b + c) = a^2 + b^2 + c^2 → 
      m ≤ a * b + a * c + b * c) ∧
    M + 10 * m = 31 := by
  sorry

end NUMINAMATH_CALUDE_xyz_sum_product_bounds_l743_74315


namespace NUMINAMATH_CALUDE_parallel_lines_m_l743_74358

/-- Two lines are parallel if their slopes are equal -/
def parallel (a b c d e f : ℝ) : Prop :=
  a * e = b * d ∧ a * f ≠ b * c

/-- The problem statement -/
theorem parallel_lines_m (m : ℝ) :
  parallel m 1 (-1) 9 m (-(2 * m + 3)) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_m_l743_74358


namespace NUMINAMATH_CALUDE_square_side_length_l743_74393

/-- Given a rectangle with width 2 and a square placed next to it,
    if the total length of the bottom side is 7,
    then the side length of the square is 5. -/
theorem square_side_length (rectangle_width square_side total_length : ℝ) : 
  rectangle_width = 2 →
  total_length = 7 →
  total_length = rectangle_width + square_side →
  square_side = 5 := by
sorry


end NUMINAMATH_CALUDE_square_side_length_l743_74393


namespace NUMINAMATH_CALUDE_vector_calculation_l743_74395

/-- Given vectors a, b, c, and e in a vector space, 
    where a = 5e, b = -3e, and c = 4e,
    prove that 2a - 3b + c = 23e -/
theorem vector_calculation 
  (V : Type*) [AddCommGroup V] [Module ℝ V] 
  (e : V) 
  (a b c : V) 
  (ha : a = 5 • e) 
  (hb : b = -3 • e) 
  (hc : c = 4 • e) : 
  2 • a - 3 • b + c = 23 • e := by
  sorry

end NUMINAMATH_CALUDE_vector_calculation_l743_74395


namespace NUMINAMATH_CALUDE_right_angled_triangle_l743_74316

theorem right_angled_triangle (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a / (Real.sin A) = b / (Real.sin B) ∧
  b / (Real.sin B) = c / (Real.sin C) ∧
  2 * c * (Real.sin (A / 2))^2 = c - b →
  C = π / 2 := by
sorry

end NUMINAMATH_CALUDE_right_angled_triangle_l743_74316


namespace NUMINAMATH_CALUDE_birds_ate_one_third_of_tomatoes_l743_74329

theorem birds_ate_one_third_of_tomatoes
  (initial_tomatoes : ℕ)
  (remaining_tomatoes : ℕ)
  (h1 : initial_tomatoes = 21)
  (h2 : remaining_tomatoes = 14) :
  (initial_tomatoes - remaining_tomatoes : ℚ) / initial_tomatoes = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_birds_ate_one_third_of_tomatoes_l743_74329


namespace NUMINAMATH_CALUDE_triangle_angle_c_60_degrees_l743_74390

theorem triangle_angle_c_60_degrees
  (A B C : Real)
  (triangle_sum : A + B + C = Real.pi)
  (tan_condition : Real.tan A + Real.tan B + Real.sqrt 3 = Real.sqrt 3 * Real.tan A * Real.tan B) :
  C = Real.pi / 3 :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_c_60_degrees_l743_74390


namespace NUMINAMATH_CALUDE_no_solution_condition_l743_74374

theorem no_solution_condition (k : ℝ) : 
  (∀ x : ℝ, x ≠ 3 ∧ x ≠ 4 → (x - 1) / (x - 3) ≠ (x - k) / (x - 4)) ↔ k = 2 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_condition_l743_74374


namespace NUMINAMATH_CALUDE_prob_at_least_one_head_in_five_tosses_l743_74360

/-- The probability of getting at least one head in five coin tosses -/
theorem prob_at_least_one_head_in_five_tosses : 
  let p_head : ℚ := 1/2  -- probability of getting heads on a single toss
  let n : ℕ := 5        -- number of coin tosses
  1 - (1 - p_head)^n = 31/32 :=
by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_head_in_five_tosses_l743_74360


namespace NUMINAMATH_CALUDE_prob_both_selected_l743_74308

/-- The probability of both brothers being selected in an exam -/
theorem prob_both_selected (p_x p_y : ℚ) (h_x : p_x = 1/5) (h_y : p_y = 2/3) :
  p_x * p_y = 2/15 := by
  sorry

end NUMINAMATH_CALUDE_prob_both_selected_l743_74308


namespace NUMINAMATH_CALUDE_greatest_root_of_f_l743_74370

def f (x : ℝ) : ℝ := 21 * x^4 - 20 * x^2 + 3

theorem greatest_root_of_f :
  ∃ (r : ℝ), r = Real.sqrt 21 / 7 ∧
  f r = 0 ∧
  ∀ (x : ℝ), f x = 0 → x ≤ r :=
sorry

end NUMINAMATH_CALUDE_greatest_root_of_f_l743_74370


namespace NUMINAMATH_CALUDE_sum_of_four_numbers_l743_74357

/-- Given two-digit numbers EH, OY, AY, and OH, where EH is four times OY 
    and AY is four times OH, prove that their sum is 150. -/
theorem sum_of_four_numbers (EH OY AY OH : ℕ) : 
  (10 ≤ EH) ∧ (EH < 100) ∧
  (10 ≤ OY) ∧ (OY < 100) ∧
  (10 ≤ AY) ∧ (AY < 100) ∧
  (10 ≤ OH) ∧ (OH < 100) ∧
  (EH = 4 * OY) ∧
  (AY = 4 * OH) →
  EH + OY + AY + OH = 150 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_numbers_l743_74357


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l743_74332

-- Define the universal set I
def I : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define set A
def A : Set Nat := {1, 3, 5}

-- Define set B
def B : Set Nat := {2, 3, 6}

-- Theorem statement
theorem complement_intersection_theorem :
  (I \ A) ∩ B = {2, 6} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l743_74332


namespace NUMINAMATH_CALUDE_equation_proof_l743_74336

/-- Given a > 0 and -∛(√a) ≤ b < ∛(a³ - √a), prove that A = 1 when
    2.334 A = √(a³-b³+√a) · (√(a³/² + √(b³+√a)) · √(a³/² - √(b³+√a))) / √((a³+b³)² - a(4a²b³+1)) -/
theorem equation_proof (a b A : ℝ) 
  (ha : a > 0) 
  (hb : -Real.rpow a (1/6) ≤ b ∧ b < Real.rpow (a^3 - Real.sqrt a) (1/3)) 
  (heq : 2.334 * A = Real.sqrt (a^3 - b^3 + Real.sqrt a) * 
    (Real.sqrt (Real.sqrt (a^3) + Real.sqrt (b^3 + Real.sqrt a)) * 
     Real.sqrt (Real.sqrt (a^3) - Real.sqrt (b^3 + Real.sqrt a))) / 
    Real.sqrt ((a^3 + b^3)^2 - a * (4 * a^2 * b^3 + 1))) : 
  A = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l743_74336


namespace NUMINAMATH_CALUDE_walking_distance_approx_2_9_l743_74307

/-- Represents a journey with cycling and walking portions -/
structure Journey where
  total_time : ℝ
  bike_speed : ℝ
  walk_speed : ℝ
  bike_fraction : ℝ
  walk_fraction : ℝ

/-- Calculates the walking distance for a given journey -/
def walking_distance (j : Journey) : ℝ :=
  let total_distance := (j.bike_speed * j.bike_fraction + j.walk_speed * j.walk_fraction) * j.total_time
  total_distance * j.walk_fraction

/-- Theorem stating that for the given journey parameters, the walking distance is approximately 2.9 km -/
theorem walking_distance_approx_2_9 :
  let j : Journey := {
    total_time := 1,
    bike_speed := 20,
    walk_speed := 4,
    bike_fraction := 2/3,
    walk_fraction := 1/3
  }
  ∃ ε > 0, |walking_distance j - 2.9| < ε :=
sorry

end NUMINAMATH_CALUDE_walking_distance_approx_2_9_l743_74307


namespace NUMINAMATH_CALUDE_parabola_intersection_l743_74355

theorem parabola_intersection (k α β : ℝ) : 
  (∀ x, x^2 - (k-1)*x - 3*k - 2 = 0 ↔ x = α ∨ x = β) →
  α^2 + β^2 = 17 →
  k = 2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_l743_74355


namespace NUMINAMATH_CALUDE_nursery_school_students_l743_74345

theorem nursery_school_students (T : ℕ) 
  (h1 : T / 8 + T / 4 + T / 3 + 40 + 60 = T) 
  (h2 : T / 8 + T / 4 + T / 3 = 100) : T = 142 := by
  sorry

end NUMINAMATH_CALUDE_nursery_school_students_l743_74345


namespace NUMINAMATH_CALUDE_quadratic_real_solutions_range_l743_74348

-- Define the quadratic equation
def quadratic_equation (m : ℝ) (x : ℝ) : ℝ := (m - 3) * x^2 + 4 * x + 1

-- Define the condition for real solutions
def has_real_solutions (m : ℝ) : Prop :=
  ∃ x : ℝ, quadratic_equation m x = 0

-- Theorem statement
theorem quadratic_real_solutions_range :
  ∀ m : ℝ, has_real_solutions m ↔ m ≤ 7 ∧ m ≠ 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_real_solutions_range_l743_74348


namespace NUMINAMATH_CALUDE_triangle_existence_and_perimeter_l743_74305

/-- A triangle with sides a, b, and c is valid if it satisfies the triangle inequality theorem -/
def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

/-- The perimeter of a triangle with sides a, b, and c -/
def triangle_perimeter (a b c : ℝ) : ℝ :=
  a + b + c

/-- Theorem: The given lengths form a valid triangle with perimeter 44 -/
theorem triangle_existence_and_perimeter :
  let a := 15
  let b := 11
  let c := 18
  is_valid_triangle a b c ∧ triangle_perimeter a b c = 44 := by sorry

end NUMINAMATH_CALUDE_triangle_existence_and_perimeter_l743_74305


namespace NUMINAMATH_CALUDE_sixth_term_equals_five_l743_74372

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The 6th term of the sequence equals 5 given the conditions -/
theorem sixth_term_equals_five (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_sum : a 2 + a 6 + a 10 = 15) : 
  a 6 = 5 := by
sorry

end NUMINAMATH_CALUDE_sixth_term_equals_five_l743_74372


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l743_74321

theorem sufficient_not_necessary (a b : ℝ) :
  (∀ a b : ℝ, a > 1 ∧ b > 1 → a + b > 2 ∧ a * b > 1) ∧
  (∃ a b : ℝ, a + b > 2 ∧ a * b > 1 ∧ ¬(a > 1 ∧ b > 1)) := by
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l743_74321


namespace NUMINAMATH_CALUDE_inequality_solution_range_l743_74352

theorem inequality_solution_range (k : ℝ) :
  (∃ x : ℝ, |x + 1| + k < x) ↔ k < -1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l743_74352


namespace NUMINAMATH_CALUDE_basketball_game_scores_l743_74338

/-- Represents the scores of a team in a basketball game -/
structure TeamScores where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- Checks if a sequence of four numbers forms a geometric sequence -/
def isGeometricSequence (a b c d : ℕ) : Prop :=
  ∃ r : ℚ, r > 1 ∧ b = a * r ∧ c = b * r ∧ d = c * r

/-- Checks if a sequence of four numbers forms an arithmetic sequence -/
def isArithmeticSequence (a b c d : ℕ) : Prop :=
  ∃ diff : ℕ, diff > 0 ∧ b = a + diff ∧ c = b + diff ∧ d = c + diff

/-- The main theorem about the basketball game -/
theorem basketball_game_scores
  (alpha : TeamScores)
  (beta : TeamScores)
  (h1 : alpha.first = beta.first)  -- Tied at the end of first quarter
  (h2 : isGeometricSequence alpha.first alpha.second alpha.third alpha.fourth)
  (h3 : isArithmeticSequence beta.first beta.second beta.third beta.fourth)
  (h4 : alpha.first + alpha.second + alpha.third + alpha.fourth =
        beta.first + beta.second + beta.third + beta.fourth + 2)  -- Alpha won by 2 points
  (h5 : alpha.first + alpha.second + alpha.third + alpha.fourth +
        beta.first + beta.second + beta.third + beta.fourth < 200)  -- Total score under 200
  : alpha.first + alpha.second + beta.first + beta.second = 30 :=
by sorry


end NUMINAMATH_CALUDE_basketball_game_scores_l743_74338


namespace NUMINAMATH_CALUDE_school_walk_time_difference_l743_74351

/-- Proves that a child walking to school is 6 minutes late when walking at 5 m/min,
    given the conditions of the problem. -/
theorem school_walk_time_difference (distance : ℝ) (slow_rate fast_rate : ℝ) (early_time : ℝ) :
  distance = 630 →
  slow_rate = 5 →
  fast_rate = 7 →
  early_time = 30 →
  distance / fast_rate + early_time = distance / slow_rate →
  distance / slow_rate - distance / fast_rate = 6 :=
by sorry

end NUMINAMATH_CALUDE_school_walk_time_difference_l743_74351


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l743_74365

open Real

theorem trigonometric_equation_solution (x : ℝ) :
  (2 * cos (π + x) - 5 * cos ((3/2) * π - x)) / (cos ((3/2) * π + x) - cos (π - x)) = 3/2 ↔
  ∃ k : ℤ, x = (π/4) * (4 * k + 1) := by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l743_74365


namespace NUMINAMATH_CALUDE_pyramid_volume_l743_74304

/-- A cube ABCDEFGH with volume 8 -/
structure Cube :=
  (volume : ℝ)
  (is_cube : volume = 8)

/-- Pyramid ACDH within the cube ABCDEFGH -/
def pyramid (c : Cube) : ℝ := sorry

/-- Theorem: The volume of pyramid ACDH is 4/3 -/
theorem pyramid_volume (c : Cube) : pyramid c = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_volume_l743_74304


namespace NUMINAMATH_CALUDE_chemistry_marks_proof_l743_74311

def english_marks : ℕ := 76
def math_marks : ℕ := 60
def physics_marks : ℕ := 82
def biology_marks : ℕ := 85
def average_marks : ℕ := 74
def total_subjects : ℕ := 5

def chemistry_marks : ℕ := 67

theorem chemistry_marks_proof :
  chemistry_marks = total_subjects * average_marks - (english_marks + math_marks + physics_marks + biology_marks) :=
by sorry

end NUMINAMATH_CALUDE_chemistry_marks_proof_l743_74311


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l743_74354

theorem fraction_equation_solution (a b : ℕ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (2 : ℚ) / 7 = 1 / (a : ℚ) + 1 / (b : ℚ) → a = 28 ∧ b = 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l743_74354


namespace NUMINAMATH_CALUDE_square_area_proof_l743_74334

theorem square_area_proof (x : ℝ) : 
  (5 * x - 20 : ℝ) = (25 - 4 * x : ℝ) → 
  (5 * x - 20 : ℝ) ^ 2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_area_proof_l743_74334


namespace NUMINAMATH_CALUDE_books_loaned_out_special_collection_loaned_books_l743_74363

/-- Proves that the number of books loaned out during the month is 20 --/
theorem books_loaned_out 
  (initial_books : ℕ) 
  (final_books : ℕ) 
  (return_rate : ℚ) : ℕ :=
  let loaned_books := (initial_books - final_books) / (1 - return_rate)
  20

/-- Given conditions --/
def initial_books : ℕ := 75
def final_books : ℕ := 68
def return_rate : ℚ := 65 / 100

/-- Main theorem --/
theorem special_collection_loaned_books : 
  books_loaned_out initial_books final_books return_rate = 20 := by
  sorry

end NUMINAMATH_CALUDE_books_loaned_out_special_collection_loaned_books_l743_74363


namespace NUMINAMATH_CALUDE_product_pricing_and_profit_maximization_l743_74337

/-- Represents the purchase and selling prices of products A and B -/
structure ProductPrices where
  purchase_price_A : ℝ
  purchase_price_B : ℝ
  selling_price_A : ℝ
  selling_price_B : ℝ

/-- Represents the number of units purchased for products A and B -/
structure PurchaseUnits where
  units_A : ℕ
  units_B : ℕ

/-- Calculates the total cost of purchasing given units of products A and B -/
def total_cost (prices : ProductPrices) (units : PurchaseUnits) : ℝ :=
  prices.purchase_price_A * units.units_A + prices.purchase_price_B * units.units_B

/-- Calculates the total profit from selling given units of products A and B -/
def total_profit (prices : ProductPrices) (units : PurchaseUnits) : ℝ :=
  (prices.selling_price_A - prices.purchase_price_A) * units.units_A +
  (prices.selling_price_B - prices.purchase_price_B) * units.units_B

theorem product_pricing_and_profit_maximization
  (prices : ProductPrices)
  (h1 : prices.purchase_price_B = 80)
  (h2 : prices.selling_price_A = 300)
  (h3 : prices.selling_price_B = 100)
  (h4 : total_cost prices { units_A := 50, units_B := 25 } = 15000)
  (h5 : ∀ units : PurchaseUnits, units.units_A + units.units_B = 300 → units.units_B ≥ 2 * units.units_A) :
  prices.purchase_price_A = 260 ∧
  ∃ (max_units : PurchaseUnits),
    max_units.units_A + max_units.units_B = 300 ∧
    max_units.units_B ≥ 2 * max_units.units_A ∧
    max_units.units_A = 100 ∧
    max_units.units_B = 200 ∧
    total_profit prices max_units = 8000 ∧
    ∀ (units : PurchaseUnits),
      units.units_A + units.units_B = 300 →
      units.units_B ≥ 2 * units.units_A →
      total_profit prices units ≤ total_profit prices max_units := by
  sorry

end NUMINAMATH_CALUDE_product_pricing_and_profit_maximization_l743_74337


namespace NUMINAMATH_CALUDE_no_hammers_loaded_l743_74359

theorem no_hammers_loaded (crate_capacity : ℕ) (num_crates : ℕ) (nail_bags : ℕ) (nail_weight : ℕ)
  (plank_bags : ℕ) (plank_weight : ℕ) (leave_out : ℕ) (hammer_weight : ℕ) :
  crate_capacity = 20 →
  num_crates = 15 →
  nail_bags = 4 →
  nail_weight = 5 →
  plank_bags = 10 →
  plank_weight = 30 →
  leave_out = 80 →
  hammer_weight = 5 →
  (∃ (loaded_planks : ℕ), 
    loaded_planks ≤ plank_bags * plank_weight ∧
    crate_capacity * num_crates - leave_out = nail_bags * nail_weight + loaded_planks) →
  (∀ (hammer_bags : ℕ), 
    crate_capacity * num_crates - leave_out < 
      nail_bags * nail_weight + plank_bags * plank_weight - leave_out + hammer_bags * hammer_weight) :=
by sorry

end NUMINAMATH_CALUDE_no_hammers_loaded_l743_74359


namespace NUMINAMATH_CALUDE_max_value_product_l743_74319

theorem max_value_product (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hsum : 5 * a + 3 * b < 90) :
  a * b * (90 - 5 * a - 3 * b) ≤ 1800 := by
sorry

end NUMINAMATH_CALUDE_max_value_product_l743_74319


namespace NUMINAMATH_CALUDE_f_properties_l743_74380

noncomputable def f (x : ℝ) := (1/3) * x^3 - 2 * x^2 + 3 * x + 2/3

theorem f_properties :
  (∀ x : ℝ, f x ≤ 2) ∧
  (∀ b : ℝ, 
    (b ≤ 0 ∨ b > (9 + Real.sqrt 33) / 6 → 
      (∀ x ∈ Set.Icc b (b + 1), f x ≤ (b^3 / 3) - b^2 + 2) ∧
      ∃ x ∈ Set.Icc b (b + 1), f x = (b^3 / 3) - b^2 + 2) ∧
    (0 < b ∧ b ≤ 1 → 
      (∀ x ∈ Set.Icc b (b + 1), f x ≤ 2) ∧
      ∃ x ∈ Set.Icc b (b + 1), f x = 2) ∧
    (1 < b ∧ b ≤ (9 + Real.sqrt 33) / 6 → 
      (∀ x ∈ Set.Icc b (b + 1), f x ≤ (b^3 / 3) - 2 * b^2 + 3 * b + 2/3) ∧
      ∃ x ∈ Set.Icc b (b + 1), f x = (b^3 / 3) - 2 * b^2 + 3 * b + 2/3)) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l743_74380


namespace NUMINAMATH_CALUDE_min_distance_to_line_l743_74362

open Real

theorem min_distance_to_line :
  let line := {(x, y) : ℝ × ℝ | 4 * x - 3 * y - 5 * sqrt 2 = 0}
  ∃ (m n : ℝ), (m, n) ∈ line ∧ ∀ (x y : ℝ), (x, y) ∈ line → m^2 + n^2 ≤ x^2 + y^2 ∧ m^2 + n^2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l743_74362


namespace NUMINAMATH_CALUDE_remainder_sum_l743_74333

theorem remainder_sum (c d : ℤ) 
  (hc : c % 90 = 84) 
  (hd : d % 120 = 117) : 
  (c + d) % 30 = 21 := by
sorry

end NUMINAMATH_CALUDE_remainder_sum_l743_74333


namespace NUMINAMATH_CALUDE_largest_factorial_as_consecutive_product_l743_74347

theorem largest_factorial_as_consecutive_product : 
  ∀ n : ℕ, n > 0 → 
  (∃ k : ℕ, k > 0 ∧ n.factorial = (List.range (n - 5)).prod.succ) → 
  n ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_largest_factorial_as_consecutive_product_l743_74347


namespace NUMINAMATH_CALUDE_initial_alcohol_percentage_l743_74386

/-- Proves that the initial alcohol percentage is 5% given the problem conditions -/
theorem initial_alcohol_percentage
  (initial_volume : ℝ)
  (added_alcohol : ℝ)
  (added_water : ℝ)
  (final_percentage : ℝ)
  (h1 : initial_volume = 40)
  (h2 : added_alcohol = 5.5)
  (h3 : added_water = 4.5)
  (h4 : final_percentage = 15)
  (h5 : final_percentage / 100 * (initial_volume + added_alcohol + added_water) =
        initial_percentage / 100 * initial_volume + added_alcohol) :
  initial_percentage = 5 :=
by sorry


end NUMINAMATH_CALUDE_initial_alcohol_percentage_l743_74386


namespace NUMINAMATH_CALUDE_min_value_of_exponential_sum_l743_74339

theorem min_value_of_exponential_sum (a b : ℝ) (h : a + b = 2) :
  ∃ (min : ℝ), min = 6 ∧ ∀ x y : ℝ, x + y = 2 → 3^x + 3^y ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_of_exponential_sum_l743_74339


namespace NUMINAMATH_CALUDE_average_speed_three_lap_run_l743_74330

/-- Calculates the average speed of a three-lap run given the track length and lap times -/
theorem average_speed_three_lap_run (track_length : ℝ) (first_lap_time second_lap_time third_lap_time : ℝ) :
  track_length = 400 →
  first_lap_time = 70 →
  second_lap_time = 85 →
  third_lap_time = 85 →
  (3 * track_length) / (first_lap_time + second_lap_time + third_lap_time) = 5 := by
  sorry

#check average_speed_three_lap_run

end NUMINAMATH_CALUDE_average_speed_three_lap_run_l743_74330


namespace NUMINAMATH_CALUDE_xyz_acronym_length_l743_74312

theorem xyz_acronym_length :
  let straight_segments : ℕ := 6
  let slanted_segments : ℕ := 6
  let straight_length : ℝ := 1
  let slanted_length : ℝ := Real.sqrt 2
  (straight_segments : ℝ) * straight_length + (slanted_segments : ℝ) * slanted_length = 6 + 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_xyz_acronym_length_l743_74312


namespace NUMINAMATH_CALUDE_power_division_result_l743_74309

theorem power_division_result : (3 : ℕ)^12 / 27^2 = 729 := by
  sorry

end NUMINAMATH_CALUDE_power_division_result_l743_74309


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l743_74323

-- Define the sets U and A
def U : Set ℝ := {x | ∃ y, y = Real.sqrt x}
def A : Set ℝ := {x | 3 ≤ 2*x - 1 ∧ 2*x - 1 < 5}

-- State the theorem
theorem complement_of_A_in_U : 
  (U \ A) = Set.Icc 0 2 ∪ Set.Ici 3 := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l743_74323


namespace NUMINAMATH_CALUDE_part_one_part_two_l743_74392

/-- The function f(x) = |a-4x| + |2a+x| -/
def f (a : ℝ) (x : ℝ) : ℝ := |a - 4*x| + |2*a + x|

/-- Part I: When a = 1, f(x) ≥ 3 if and only if x ≤ 0 or x ≥ 2/5 -/
theorem part_one : 
  ∀ x : ℝ, f 1 x ≥ 3 ↔ x ≤ 0 ∨ x ≥ 2/5 := by sorry

/-- Part II: For all x ≠ 0 and all a, f(x) + f(-1/x) ≥ 10 -/
theorem part_two : 
  ∀ a x : ℝ, x ≠ 0 → f a x + f a (-1/x) ≥ 10 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l743_74392


namespace NUMINAMATH_CALUDE_expression_evaluation_l743_74301

theorem expression_evaluation :
  let x : ℝ := 2 - Real.sqrt 3
  (7 + 4 * Real.sqrt 3) * x^2 - (2 + Real.sqrt 3) * x + Real.sqrt 3 = 2 + Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l743_74301


namespace NUMINAMATH_CALUDE_jeff_rental_duration_l743_74367

/-- Represents the rental scenario for Jeff's apartment. -/
structure RentalScenario where
  initialRent : ℕ  -- Monthly rent for the first 3 years
  raisedRent : ℕ   -- Monthly rent after the raise
  initialYears : ℕ -- Number of years at the initial rent
  totalPaid : ℕ    -- Total amount paid over the entire rental period

/-- Calculates the total number of years Jeff rented the apartment. -/
def totalRentalYears (scenario : RentalScenario) : ℕ :=
  scenario.initialYears + 
  ((scenario.totalPaid - scenario.initialRent * scenario.initialYears * 12) / (scenario.raisedRent * 12))

/-- Theorem stating that Jeff rented the apartment for 5 years. -/
theorem jeff_rental_duration (scenario : RentalScenario) 
  (h1 : scenario.initialRent = 300)
  (h2 : scenario.raisedRent = 350)
  (h3 : scenario.initialYears = 3)
  (h4 : scenario.totalPaid = 19200) :
  totalRentalYears scenario = 5 := by
  sorry

end NUMINAMATH_CALUDE_jeff_rental_duration_l743_74367


namespace NUMINAMATH_CALUDE_billys_age_l743_74303

theorem billys_age (billy joe : ℕ) 
  (h1 : billy = 3 * joe) 
  (h2 : billy + joe = 60) : 
  billy = 45 := by
sorry

end NUMINAMATH_CALUDE_billys_age_l743_74303
