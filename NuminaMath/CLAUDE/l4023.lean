import Mathlib

namespace NUMINAMATH_CALUDE_odd_function_property_l4023_402304

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem odd_function_property (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_even : is_even_function (fun x ↦ f (x + 1)))
  (h_f_neg_one : f (-1) = -1) :
  f 2018 + f 2019 = -1 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l4023_402304


namespace NUMINAMATH_CALUDE_cube_minus_cylinder_volume_l4023_402360

/-- The remaining volume of a cube after removing a cylindrical section -/
theorem cube_minus_cylinder_volume (cube_side : ℝ) (cylinder_radius : ℝ) (cylinder_height : ℝ)
  (h1 : cube_side = 5)
  (h2 : cylinder_radius = 1)
  (h3 : cylinder_height = 5) :
  cube_side ^ 3 - π * cylinder_radius ^ 2 * cylinder_height = 125 - 5 * π := by
  sorry

#check cube_minus_cylinder_volume

end NUMINAMATH_CALUDE_cube_minus_cylinder_volume_l4023_402360


namespace NUMINAMATH_CALUDE_find_A_l4023_402385

theorem find_A (A B : ℕ) (h : 15 = 3 * A ∧ 15 = 5 * B) : A = 5 := by
  sorry

end NUMINAMATH_CALUDE_find_A_l4023_402385


namespace NUMINAMATH_CALUDE_lunch_cost_theorem_l4023_402311

/-- Calculates the amount each paying student contributes for lunch -/
def lunch_cost_per_paying_student (total_students : ℕ) (free_lunch_percentage : ℚ) (total_cost : ℚ) : ℚ :=
  let paying_students := total_students * (1 - free_lunch_percentage)
  total_cost / paying_students

theorem lunch_cost_theorem (total_students : ℕ) (free_lunch_percentage : ℚ) (total_cost : ℚ) 
  (h1 : total_students = 50)
  (h2 : free_lunch_percentage = 2/5)
  (h3 : total_cost = 210) :
  lunch_cost_per_paying_student total_students free_lunch_percentage total_cost = 7 := by
  sorry

#eval lunch_cost_per_paying_student 50 (2/5) 210

end NUMINAMATH_CALUDE_lunch_cost_theorem_l4023_402311


namespace NUMINAMATH_CALUDE_exists_unique_function_satisfying_equation_l4023_402342

/-- A functional equation that uniquely determines a function f: ℝ → ℤ --/
def functional_equation (f : ℝ → ℤ) (x₁ x₂ : ℝ) : Prop :=
  0 = (f (-x₁^2 - (x₁ * x₂ - 1)^2))^2 +
      ((f (-x₁^2 - (x₁ * x₂ - 1)^2 + 1) - 1/2)^2 - 1/4)^2 +
      (f (x₁^2 + 2) - 2 * f (x₁^2) + f (x₁^2 - 2))^2 +
      ((f (x₁^2) - f (x₁^2 - 2))^2 - 1)^2 +
      ((f (x₁^2) + f (x₁^2 + 1) - 1/2)^2 - 1/4)^2

/-- The theorem stating the existence of a unique function satisfying the functional equation --/
theorem exists_unique_function_satisfying_equation :
  ∃! f : ℝ → ℤ, (∀ x₁ x₂ : ℝ, functional_equation f x₁ x₂) ∧ Set.range f = Set.univ :=
sorry

end NUMINAMATH_CALUDE_exists_unique_function_satisfying_equation_l4023_402342


namespace NUMINAMATH_CALUDE_rental_van_cost_increase_l4023_402320

theorem rental_van_cost_increase (C : ℝ) : 
  C / 8 - C / 9 = C / 72 := by sorry

end NUMINAMATH_CALUDE_rental_van_cost_increase_l4023_402320


namespace NUMINAMATH_CALUDE_number_division_problem_l4023_402301

theorem number_division_problem (x y : ℚ) 
  (h1 : (x - 5) / 7 = 7) 
  (h2 : (x - 34) / y = 2) : 
  y = 10 := by sorry

end NUMINAMATH_CALUDE_number_division_problem_l4023_402301


namespace NUMINAMATH_CALUDE_pirate_count_correct_l4023_402341

/-- The number of pirates on the schooner satisfying the given conditions -/
def pirate_count : ℕ := 60

/-- The fraction of pirates who lost a leg -/
def leg_loss_fraction : ℚ := 2/3

/-- The fraction of fight participants who lost an arm -/
def arm_loss_fraction : ℚ := 54/100

/-- The fraction of fight participants who lost both an arm and a leg -/
def both_loss_fraction : ℚ := 34/100

/-- The number of pirates who did not participate in the fight -/
def non_participants : ℕ := 10

theorem pirate_count_correct : 
  ∃ (p : ℕ), p = pirate_count ∧ 
  (leg_loss_fraction : ℚ) * p = (p - non_participants) * both_loss_fraction + 
    ((p - non_participants) * arm_loss_fraction - (p - non_participants) * both_loss_fraction) +
    (leg_loss_fraction * p - (p - non_participants) * both_loss_fraction) :=
sorry

end NUMINAMATH_CALUDE_pirate_count_correct_l4023_402341


namespace NUMINAMATH_CALUDE_trader_gain_percentage_l4023_402306

/-- The gain percentage of a trader selling pens -/
def gain_percentage (num_sold : ℕ) (num_gained : ℕ) : ℚ :=
  (num_gained : ℚ) / (num_sold : ℚ) * 100

/-- Theorem: The gain percentage is 25% when selling 80 pens and gaining the cost of 20 pens -/
theorem trader_gain_percentage : gain_percentage 80 20 = 25 := by
  sorry

end NUMINAMATH_CALUDE_trader_gain_percentage_l4023_402306


namespace NUMINAMATH_CALUDE_percentage_men_undeclared_l4023_402339

/-- Represents the percentages of students in different majors and categories -/
structure ClassComposition where
  men_science : ℝ
  men_humanities : ℝ
  men_business : ℝ
  men_double_science_humanities : ℝ
  men_double_science_business : ℝ
  men_double_humanities_business : ℝ

/-- Theorem stating the percentage of men with undeclared majors -/
theorem percentage_men_undeclared (c : ClassComposition) : 
  c.men_science = 24 ∧ 
  c.men_humanities = 13 ∧ 
  c.men_business = 18 ∧
  c.men_double_science_humanities = 13.5 ∧
  c.men_double_science_business = 9 ∧
  c.men_double_humanities_business = 6.75 →
  100 - (c.men_science + c.men_humanities + c.men_business + 
         c.men_double_science_humanities + c.men_double_science_business + 
         c.men_double_humanities_business) = 15.75 := by
  sorry

#check percentage_men_undeclared

end NUMINAMATH_CALUDE_percentage_men_undeclared_l4023_402339


namespace NUMINAMATH_CALUDE_tan_alpha_value_l4023_402392

theorem tan_alpha_value (α : Real) 
  (h1 : π/2 < α) (h2 : α < π) 
  (h3 : Real.sin α + Real.cos α = Real.sqrt 10 / 5) : 
  Real.tan α = -3 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l4023_402392


namespace NUMINAMATH_CALUDE_permutation_combination_equality_l4023_402321

/-- Given that A_n^2 = C_n^(n-3), prove that n = 8 --/
theorem permutation_combination_equality (n : ℕ) : 
  (n.factorial / (n - 2).factorial) = (n.factorial / ((3).factorial * (n - 3).factorial)) → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_permutation_combination_equality_l4023_402321


namespace NUMINAMATH_CALUDE_line_mn_properties_l4023_402355

-- Define the parabola
def parabola (p : ℝ × ℝ) : Prop := p.2^2 = 4 * p.1

-- Define the condition for the sum of vertical coordinates
def sum_of_verticals (m n : ℝ × ℝ) : Prop := m.2 + n.2 = 4

-- Define the angle condition
def angle_condition (m n : ℝ × ℝ) : Prop :=
  (m.2 / (m.1 + 2)) + (n.2 / (n.1 + 2)) = 0

-- Main theorem
theorem line_mn_properties (m n : ℝ × ℝ) :
  parabola m → parabola n → sum_of_verticals m n → angle_condition m n →
  ∃ k b : ℝ, k = 1 ∧ b = -2 ∧ ∀ x y : ℝ, y = k * x + b ↔ (x = m.1 ∧ y = m.2) ∨ (x = n.1 ∧ y = n.2) :=
sorry

end NUMINAMATH_CALUDE_line_mn_properties_l4023_402355


namespace NUMINAMATH_CALUDE_multiple_of_nine_in_range_l4023_402333

theorem multiple_of_nine_in_range (y : ℕ) :
  y > 0 ∧ 
  ∃ k : ℕ, y = 9 * k ∧ 
  y^2 > 225 ∧ 
  y < 30 →
  y = 18 ∨ y = 27 := by
sorry

end NUMINAMATH_CALUDE_multiple_of_nine_in_range_l4023_402333


namespace NUMINAMATH_CALUDE_card_distribution_events_l4023_402382

-- Define the set of colors
inductive Color
| Red
| Yellow
| Blue
| White

-- Define the set of people
inductive Person
| A
| B
| C
| D

-- Define the distribution of cards
def Distribution := Person → Color

-- Define the event "A receives the red card"
def A_red (d : Distribution) : Prop := d Person.A = Color.Red

-- Define the event "D receives the red card"
def D_red (d : Distribution) : Prop := d Person.D = Color.Red

-- State the theorem
theorem card_distribution_events :
  -- Each person receives one card
  (∀ p : Person, ∃! c : Color, ∀ d : Distribution, d p = c) →
  -- The events are mutually exclusive
  (∀ d : Distribution, ¬(A_red d ∧ D_red d)) ∧
  -- The events are not complementary
  ¬(∀ d : Distribution, A_red d ↔ ¬(D_red d)) :=
by sorry

end NUMINAMATH_CALUDE_card_distribution_events_l4023_402382


namespace NUMINAMATH_CALUDE_min_value_sum_squares_l4023_402363

theorem min_value_sum_squares (x y z k : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = k) (hk : k ≥ -1) :
  ∃ (m : ℝ), m = 1 ∧ ∀ (a b c : ℝ), a^3 + b^3 + c^3 - 3*a*b*c = k → a^2 + b^2 + c^2 ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_l4023_402363


namespace NUMINAMATH_CALUDE_polygon_sides_l4023_402313

theorem polygon_sides (n : ℕ) (h : n ≥ 3) : 
  (n - 2) * 180 + 360 = 1260 → n = 7 := by
sorry

end NUMINAMATH_CALUDE_polygon_sides_l4023_402313


namespace NUMINAMATH_CALUDE_equation_solutions_l4023_402373

/-- The set of solutions to the equation (3x+6)/(x^2+5x-14) = (3-x)/(x-2) -/
def solutions : Set ℝ := {x | x = 3 ∨ x = -5}

/-- The original equation -/
def equation (x : ℝ) : Prop :=
  x ≠ 2 ∧ x ≠ -7 ∧ (3*x + 6) / (x^2 + 5*x - 14) = (3 - x) / (x - 2)

theorem equation_solutions :
  ∀ x : ℝ, equation x ↔ x ∈ solutions := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l4023_402373


namespace NUMINAMATH_CALUDE_holiday_ticket_cost_theorem_l4023_402354

def holiday_ticket_cost (regular_adult_price : ℝ) : ℝ :=
  let holiday_adult_price := 1.1 * regular_adult_price
  let child_price := 0.5 * regular_adult_price
  6 * holiday_adult_price + 5 * child_price

theorem holiday_ticket_cost_theorem (regular_adult_price : ℝ) :
  4 * (1.1 * regular_adult_price) + 3 * (0.5 * regular_adult_price) = 28.80 →
  holiday_ticket_cost regular_adult_price = 44.41 := by
  sorry

end NUMINAMATH_CALUDE_holiday_ticket_cost_theorem_l4023_402354


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l4023_402349

/-- A quadratic function f(x) = ax^2 + bx + c with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  h1 : a > 0
  h2 : a + b + c = -a/2
  h3 : 3 * a > 2 * c
  h4 : 2 * c > 2 * b

/-- The main theorem about the properties of the quadratic function -/
theorem quadratic_function_properties (f : QuadraticFunction) :
  (-3 < f.b / f.a ∧ f.b / f.a < -3/4) ∧
  (∃ x : ℝ, 0 < x ∧ x < 2 ∧ f.a * x^2 + f.b * x + f.c = 0) ∧
  (∀ x₁ x₂ : ℝ, f.a * x₁^2 + f.b * x₁ + f.c = 0 → f.a * x₂^2 + f.b * x₂ + f.c = 0 →
    Real.sqrt 2 ≤ |x₁ - x₂| ∧ |x₁ - x₂| < Real.sqrt 57 / 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l4023_402349


namespace NUMINAMATH_CALUDE_smallest_t_for_70_degrees_l4023_402352

-- Define the temperature function
def T (t : ℝ) : ℝ := -t^2 + 10*t + 60

-- Define the atmospheric pressure function (not used in the proof, but included for completeness)
def P (t : ℝ) : ℝ := 800 - 2*t

-- Theorem statement
theorem smallest_t_for_70_degrees :
  ∃ (t : ℝ), t > 0 ∧ T t = 70 ∧ ∀ (s : ℝ), s > 0 ∧ T s = 70 → t ≤ s :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_smallest_t_for_70_degrees_l4023_402352


namespace NUMINAMATH_CALUDE_expand_and_simplify_l4023_402343

theorem expand_and_simplify (x : ℝ) (hx : x ≠ 0) :
  (3 / 7) * ((14 / x^3) + 15*x - 6*x^5) = 6 / x^3 + (45*x) / 7 - (18*x^5) / 7 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l4023_402343


namespace NUMINAMATH_CALUDE_sign_language_size_l4023_402381

theorem sign_language_size :
  ∀ n : ℕ,
  (n ≥ 2) →
  (n^2 - (n-2)^2 = 888) →
  n = 223 := by
sorry

end NUMINAMATH_CALUDE_sign_language_size_l4023_402381


namespace NUMINAMATH_CALUDE_owen_final_count_l4023_402348

/-- The number of turtles Owen has after all transformations and donations -/
def final_owen_turtles (initial_owen : ℕ) (johanna_difference : ℕ) : ℕ :=
  let initial_johanna := initial_owen - johanna_difference
  let owen_after_month := initial_owen * 2
  let johanna_after_month := initial_johanna / 2
  owen_after_month + johanna_after_month

/-- Theorem stating that Owen ends up with 50 turtles -/
theorem owen_final_count :
  final_owen_turtles 21 5 = 50 := by
  sorry

end NUMINAMATH_CALUDE_owen_final_count_l4023_402348


namespace NUMINAMATH_CALUDE_tan_addition_special_case_l4023_402303

theorem tan_addition_special_case (x : Real) (h : Real.tan x = 3) :
  Real.tan (x + π/3) = (12 * Real.sqrt 3 + 3) / 26 := by
  sorry

end NUMINAMATH_CALUDE_tan_addition_special_case_l4023_402303


namespace NUMINAMATH_CALUDE_antiderivative_increment_l4023_402309

-- Define the function f(x) = 2x + 4
def f (x : ℝ) : ℝ := 2 * x + 4

-- Define what it means for F to be an antiderivative of f on the interval [-2, 0]
def is_antiderivative (F : ℝ → ℝ) : Prop :=
  ∀ x ∈ Set.Icc (-2) 0, (deriv F) x = f x

-- Theorem statement
theorem antiderivative_increment (F : ℝ → ℝ) (h : is_antiderivative F) :
  F 0 - F (-2) = 4 := by sorry

end NUMINAMATH_CALUDE_antiderivative_increment_l4023_402309


namespace NUMINAMATH_CALUDE_ellipse_chord_ratio_theorem_l4023_402317

noncomputable section

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0

/-- The eccentricity of an ellipse -/
def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Theorem: For any ellipse satisfying the given conditions, 
    the ratio of the square of the chord length passing through the origin 
    to the chord length passing through the left focus is always 4, 
    when the slope angles of these chords sum to π -/
theorem ellipse_chord_ratio_theorem (e : Ellipse) 
    (h_focus : e.eccentricity * e.a = 1)
    (h_b_mean : e.b^2 = 3 * e.eccentricity * e.a)
    (α β : ℝ)
    (h_angle_sum : α + β = π)
    (A B D E : Point)
    (h_AB_on_ellipse : A.x^2 / e.a^2 + A.y^2 / e.b^2 = 1 ∧ 
                       B.x^2 / e.a^2 + B.y^2 / e.b^2 = 1)
    (h_DE_on_ellipse : D.x^2 / e.a^2 + D.y^2 / e.b^2 = 1 ∧ 
                       E.x^2 / e.a^2 + E.y^2 / e.b^2 = 1)
    (h_AB_through_origin : ∃ (k : ℝ), A.y = k * A.x ∧ B.y = k * B.x)
    (h_DE_through_focus : ∃ (m : ℝ), D.y = m * (D.x + 1) ∧ E.y = m * (E.x + 1))
    (h_AB_slope : ∃ (k : ℝ), k = Real.tan α)
    (h_DE_slope : ∃ (m : ℝ), m = Real.tan β) :
    (distance A B)^2 / (distance D E) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_chord_ratio_theorem_l4023_402317


namespace NUMINAMATH_CALUDE_factory_month_days_l4023_402345

/-- The number of days in a month for a computer factory -/
def days_in_month (computers_per_month : ℕ) (computers_per_half_hour : ℕ) : ℕ :=
  computers_per_month * 30 / (computers_per_half_hour * 24 * 2)

/-- Theorem: Given the production rate, the number of days in the month is 28 -/
theorem factory_month_days :
  days_in_month 4032 3 = 28 := by
  sorry

end NUMINAMATH_CALUDE_factory_month_days_l4023_402345


namespace NUMINAMATH_CALUDE_greatest_integer_less_than_negative_fraction_l4023_402308

theorem greatest_integer_less_than_negative_fraction :
  ⌊-22 / 5⌋ = -5 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_less_than_negative_fraction_l4023_402308


namespace NUMINAMATH_CALUDE_income_comparison_l4023_402327

theorem income_comparison (tim mary juan : ℝ) 
  (h1 : mary = 1.6 * tim) 
  (h2 : mary = 0.8 * juan) : 
  tim = 0.5 * juan := by
  sorry

end NUMINAMATH_CALUDE_income_comparison_l4023_402327


namespace NUMINAMATH_CALUDE_addition_problem_solution_l4023_402366

/-- Represents a digit in the addition problem -/
structure Digit :=
  (value : Nat)
  (property : value < 10)

/-- Represents the addition problem -/
structure AdditionProblem :=
  (E : Digit)
  (I : Digit)
  (G : Digit)
  (H : Digit)
  (T : Digit)
  (F : Digit)
  (V : Digit)
  (R : Digit)
  (N : Digit)
  (all_different : ∀ d1 d2 : Digit, d1.value = d2.value → d1 = d2)
  (E_is_nine : E.value = 9)
  (G_is_odd : G.value % 2 = 1)
  (equation_holds : 
    10000 * E.value + 1000 * I.value + 100 * G.value + 10 * H.value + T.value +
    10000 * F.value + 1000 * I.value + 100 * V.value + 10 * E.value =
    10000000 * T.value + 1000000 * H.value + 100000 * I.value + 10000 * R.value +
    1000 * T.value + 100 * E.value + 10 * E.value + N.value)

theorem addition_problem_solution (problem : AdditionProblem) : problem.I.value = 4 := by
  sorry

end NUMINAMATH_CALUDE_addition_problem_solution_l4023_402366


namespace NUMINAMATH_CALUDE_volume_of_specific_tetrahedron_l4023_402332

/-- The volume of a tetrahedron given its edge lengths -/
def tetrahedron_volume (pq pr ps qr qs rs : ℝ) : ℝ := sorry

/-- Theorem: The volume of tetrahedron PQRS with given edge lengths is 140/9 -/
theorem volume_of_specific_tetrahedron :
  let pq : ℝ := 6
  let pr : ℝ := 5
  let ps : ℝ := 4 * Real.sqrt 2
  let qr : ℝ := 3 * Real.sqrt 2
  let qs : ℝ := 5
  let rs : ℝ := 4
  tetrahedron_volume pq pr ps qr qs rs = 140 / 9 := by sorry

end NUMINAMATH_CALUDE_volume_of_specific_tetrahedron_l4023_402332


namespace NUMINAMATH_CALUDE_expansion_terms_count_l4023_402337

/-- The number of dissimilar terms in the expansion of (a + b + c + d)^12 -/
def dissimilarTerms : ℕ :=
  Nat.choose 15 3

/-- The number of ways to distribute 12 indistinguishable objects into 4 distinguishable boxes -/
def distributionWays : ℕ :=
  Nat.choose (12 + 4 - 1) (4 - 1)

theorem expansion_terms_count :
  dissimilarTerms = distributionWays ∧ dissimilarTerms = 455 := by
  sorry

end NUMINAMATH_CALUDE_expansion_terms_count_l4023_402337


namespace NUMINAMATH_CALUDE_S_infinite_l4023_402340

/-- Sum of divisors function -/
def sigma (n : ℕ) : ℕ := sorry

/-- The set of natural numbers n such that σ(n)/n > σ(k)/k for all k < n -/
def S : Set ℕ :=
  {n : ℕ | ∀ k < n, (sigma n : ℚ) / n > (sigma k : ℚ) / k}

/-- Theorem stating that S is infinite -/
theorem S_infinite : Set.Infinite S := by sorry

end NUMINAMATH_CALUDE_S_infinite_l4023_402340


namespace NUMINAMATH_CALUDE_license_plate_palindrome_probability_l4023_402329

/-- The probability of a license plate containing at least one palindrome -/
theorem license_plate_palindrome_probability :
  let num_letters : ℕ := 26
  let num_digits : ℕ := 10
  let total_arrangements : ℕ := num_letters^4 * num_digits^4
  let letter_palindromes : ℕ := num_letters^2
  let digit_palindromes : ℕ := num_digits^2
  let prob_letter_palindrome : ℚ := letter_palindromes / (num_letters^4 : ℚ)
  let prob_digit_palindrome : ℚ := digit_palindromes / (num_digits^4 : ℚ)
  let prob_both_palindromes : ℚ := (letter_palindromes * digit_palindromes) / (total_arrangements : ℚ)
  let prob_at_least_one_palindrome : ℚ := prob_letter_palindrome + prob_digit_palindrome - prob_both_palindromes
  prob_at_least_one_palindrome = 131 / 1142 :=
by sorry

end NUMINAMATH_CALUDE_license_plate_palindrome_probability_l4023_402329


namespace NUMINAMATH_CALUDE_discount_saves_money_savings_amount_l4023_402365

/-- Represents the ticket pricing strategy for a park -/
structure TicketStrategy where
  regular_price : ℕ  -- Regular price per ticket
  discount_rate : ℚ  -- Discount rate for group tickets
  discount_threshold : ℕ  -- Minimum number of people for group discount

/-- Calculates the total cost for a given number of tickets -/
def total_cost (strategy : TicketStrategy) (num_tickets : ℕ) : ℚ :=
  if num_tickets ≥ strategy.discount_threshold
  then (strategy.regular_price * num_tickets * (1 - strategy.discount_rate))
  else (strategy.regular_price * num_tickets)

/-- Theorem: Purchasing 25 tickets with discount is cheaper than 23 without discount -/
theorem discount_saves_money (strategy : TicketStrategy) 
  (h1 : strategy.regular_price = 10)
  (h2 : strategy.discount_rate = 1/5)
  (h3 : strategy.discount_threshold = 25) :
  total_cost strategy 25 < total_cost strategy 23 ∧ 
  total_cost strategy 23 - total_cost strategy 25 = 30 :=
by sorry

/-- Corollary: The savings amount to exactly 30 yuan -/
theorem savings_amount (strategy : TicketStrategy)
  (h1 : strategy.regular_price = 10)
  (h2 : strategy.discount_rate = 1/5)
  (h3 : strategy.discount_threshold = 25) :
  total_cost strategy 23 - total_cost strategy 25 = 30 :=
by sorry

end NUMINAMATH_CALUDE_discount_saves_money_savings_amount_l4023_402365


namespace NUMINAMATH_CALUDE_investment_problem_l4023_402353

/-- Proves that given the conditions of the investment problem, the initial investment in the 2.5% account was $290 -/
theorem investment_problem (total_investment : ℝ) (interest_rate1 : ℝ) (interest_rate2 : ℝ) 
  (final_amount : ℝ) (investment1 : ℝ) :
  total_investment = 1500 →
  interest_rate1 = 0.025 →
  interest_rate2 = 0.045 →
  final_amount = 1650 →
  investment1 * (1 + interest_rate1)^2 + (total_investment - investment1) * (1 + interest_rate2)^2 = final_amount →
  investment1 = 290 :=
by sorry

end NUMINAMATH_CALUDE_investment_problem_l4023_402353


namespace NUMINAMATH_CALUDE_sum_reciprocals_and_powers_l4023_402387

theorem sum_reciprocals_and_powers (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a + 1 / b ≥ 4) ∧ (1 / a^2016 + 1 / b^2016 ≥ 2^2017) := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_and_powers_l4023_402387


namespace NUMINAMATH_CALUDE_hotel_expenditure_l4023_402316

/-- The total expenditure of 9 persons in a hotel, given specific spending conditions. -/
theorem hotel_expenditure (n : ℕ) (individual_cost : ℕ) (extra_cost : ℕ) : 
  n = 9 → 
  individual_cost = 12 → 
  extra_cost = 8 → 
  (n - 1) * individual_cost + 
  (individual_cost + ((n - 1) * individual_cost + (individual_cost + extra_cost)) / n) = 117 :=
by sorry

end NUMINAMATH_CALUDE_hotel_expenditure_l4023_402316


namespace NUMINAMATH_CALUDE_systematic_sampling_60_5_l4023_402371

/-- Systematic sampling function that returns a list of sample numbers -/
def systematicSample (totalPopulation : ℕ) (sampleSize : ℕ) : List ℕ :=
  let interval := totalPopulation / sampleSize
  List.range sampleSize |>.map (fun i => i * interval + interval)

/-- Theorem: The systematic sampling of 5 students from a class of 60 yields [6, 18, 30, 42, 54] -/
theorem systematic_sampling_60_5 :
  systematicSample 60 5 = [6, 18, 30, 42, 54] := by
  sorry

#eval systematicSample 60 5

end NUMINAMATH_CALUDE_systematic_sampling_60_5_l4023_402371


namespace NUMINAMATH_CALUDE_unique_coprime_pair_l4023_402374

def divides (a b : ℕ) : Prop := ∃ k, b = a * k

theorem unique_coprime_pair :
  ∀ a b : ℕ,
    a > 0 ∧ b > 0 →
    a < b →
    (∀ n : ℕ, n > 0 → divides b ((n+2)*a^(n+1002) - (n+1)*a^(n+1001) - n*a^(n+1000))) →
    (∀ d : ℕ, d > 1 → (divides d a ∧ divides d b) → d = 1) →
    a = 3 ∧ b = 5 :=
by sorry

end NUMINAMATH_CALUDE_unique_coprime_pair_l4023_402374


namespace NUMINAMATH_CALUDE_symmetric_point_y_axis_l4023_402344

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Symmetry with respect to the y-axis -/
def symmetricYAxis (p : Point3D) : Point3D :=
  { x := -p.x, y := p.y, z := -p.z }

/-- The theorem stating that the point symmetric to (2, -3, 5) with respect to the y-axis is (-2, -3, -5) -/
theorem symmetric_point_y_axis :
  let original := Point3D.mk 2 (-3) 5
  symmetricYAxis original = Point3D.mk (-2) (-3) (-5) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_y_axis_l4023_402344


namespace NUMINAMATH_CALUDE_apartment_count_l4023_402395

theorem apartment_count : 
  ∀ (total : ℕ) 
    (at_least_one : ℕ) 
    (at_least_two : ℕ) 
    (only_one : ℕ),
  at_least_one = (85 * total) / 100 →
  at_least_two = (60 * total) / 100 →
  only_one = 30 →
  only_one = at_least_one - at_least_two →
  total = 75 := by
sorry

end NUMINAMATH_CALUDE_apartment_count_l4023_402395


namespace NUMINAMATH_CALUDE_ngon_recovery_l4023_402305

/-- Represents a point in the plane with an associated number -/
structure MarkedPoint where
  x : ℝ
  y : ℝ
  number : ℕ

/-- Represents a regular n-gon with its center -/
structure RegularNGon where
  n : ℕ
  center : MarkedPoint
  vertices : Fin n → MarkedPoint

/-- Represents a triangle formed by two adjacent vertices and the center -/
structure Triangle where
  a : MarkedPoint
  b : MarkedPoint
  c : MarkedPoint

/-- Function to generate the list of triangles from a regular n-gon -/
def generateTriangles (ngon : RegularNGon) : List Triangle := sorry

/-- Function to get the multiset of numbers from a triangle -/
def getTriangleNumbers (triangle : Triangle) : Multiset ℕ := sorry

/-- Predicate to check if the original numbers can be uniquely recovered -/
def canRecover (ngon : RegularNGon) : Prop := sorry

theorem ngon_recovery (n : ℕ) :
  ∀ (ngon : RegularNGon),
    ngon.n = n →
    canRecover ngon ↔ Odd n :=
  sorry

end NUMINAMATH_CALUDE_ngon_recovery_l4023_402305


namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l4023_402367

theorem hemisphere_surface_area (r : ℝ) (h : r > 0) : 
  π * r^2 = 100 * π → 2 * π * r^2 + π * r^2 = 300 * π := by
  sorry

end NUMINAMATH_CALUDE_hemisphere_surface_area_l4023_402367


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l4023_402386

theorem sufficient_but_not_necessary_condition :
  ∃ (a b : ℝ), (a > 1 ∧ b > 1 → a * b > 1) ∧
  ¬(a * b > 1 → a > 1 ∧ b > 1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l4023_402386


namespace NUMINAMATH_CALUDE_max_value_fraction_l4023_402326

theorem max_value_fraction (x y : ℝ) (hx : -6 ≤ x ∧ x ≤ -3) (hy : 1 ≤ y ∧ y ≤ 5) :
  (∀ x' y', -6 ≤ x' ∧ x' ≤ -3 → 1 ≤ y' ∧ y' ≤ 5 → (x' + y') / x' ≤ (x + y) / x) →
  (x + y) / x = 2/3 :=
sorry

end NUMINAMATH_CALUDE_max_value_fraction_l4023_402326


namespace NUMINAMATH_CALUDE_fraction_equality_l4023_402307

theorem fraction_equality (a b : ℝ) (h : a / 4 = b / 3) : (a - b) / b = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l4023_402307


namespace NUMINAMATH_CALUDE_regression_line_equation_l4023_402318

/-- Given a regression line with slope 1.23 and a point (4, 5) on the line,
    prove that the equation of the line is y = 1.23x + 0.08 -/
theorem regression_line_equation (x y : ℝ) :
  let slope : ℝ := 1.23
  let point : ℝ × ℝ := (4, 5)
  (y - point.2 = slope * (x - point.1)) → (y = slope * x + 0.08) :=
by
  sorry

end NUMINAMATH_CALUDE_regression_line_equation_l4023_402318


namespace NUMINAMATH_CALUDE_words_with_consonants_count_l4023_402397

def alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E', 'I'}
def vowels : Finset Char := {'A', 'E', 'I'}
def consonants : Finset Char := {'B', 'C', 'D'}

def word_length : Nat := 5

theorem words_with_consonants_count :
  (alphabet.card ^ word_length) - (vowels.card ^ word_length) = 7533 := by
  sorry

end NUMINAMATH_CALUDE_words_with_consonants_count_l4023_402397


namespace NUMINAMATH_CALUDE_problem_solution_l4023_402347

theorem problem_solution : 
  (-(3^2) / 3 + |(-7)| + 3 * (-1/3) = 3) ∧
  ((-1)^2022 - (-1/4 - (-1/3)) / (-1/12) = 2) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l4023_402347


namespace NUMINAMATH_CALUDE_quadratic_functions_coincidence_l4023_402312

/-- Represents a quadratic function of the form ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Returns true if two quadratic functions can coincide through parallel translation -/
def can_coincide (f g : QuadraticFunction) : Prop :=
  f.a = g.a ∧ f.a ≠ 0

/-- The three given quadratic functions -/
def A : QuadraticFunction := ⟨1, 0, -1⟩
def B : QuadraticFunction := ⟨-1, 0, 1⟩
def C : QuadraticFunction := ⟨1, 2, -1⟩

theorem quadratic_functions_coincidence :
  can_coincide A C ∧ ¬can_coincide A B ∧ ¬can_coincide B C := by sorry

end NUMINAMATH_CALUDE_quadratic_functions_coincidence_l4023_402312


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l4023_402369

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h : ∀ x : ℝ, -1/2 < x ∧ x < 2 ↔ f a b c x > 0) : 
  a < 0 ∧ c > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l4023_402369


namespace NUMINAMATH_CALUDE_root_ratio_sum_l4023_402361

theorem root_ratio_sum (m₁ m₂ : ℝ) : 
  (∃ a b : ℝ, (m₁ * (a^2 - 2*a) + 3*a + 7 = 0) ∧ 
               (m₁ * (b^2 - 2*b) + 3*b + 7 = 0) ∧ 
               (a / b + b / a = 7 / 10)) →
  (∃ a b : ℝ, (m₂ * (a^2 - 2*a) + 3*a + 7 = 0) ∧ 
               (m₂ * (b^2 - 2*b) + 3*b + 7 = 0) ∧ 
               (a / b + b / a = 7 / 10)) →
  m₁ / m₂ + m₂ / m₁ = 253 / 36 := by
sorry

end NUMINAMATH_CALUDE_root_ratio_sum_l4023_402361


namespace NUMINAMATH_CALUDE_solution_set_characterization_l4023_402351

theorem solution_set_characterization (k : ℝ) :
  (∀ x : ℝ, (|x - 2007| + |x + 2007| = k) ↔ (x < -2007 ∨ x > 2007)) ↔ k > 4014 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_characterization_l4023_402351


namespace NUMINAMATH_CALUDE_range_of_f_minus_x_l4023_402310

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x =>
  if x < -3 then -4
  else if x < -2 then -2
  else if x < -1 then -1
  else if x < 0 then 0
  else if x < 1 then 1
  else if x < 2 then 2
  else if x < 3 then 3
  else 4

-- Define the domain
def domain : Set ℝ := Set.Icc (-4) 4

-- State the theorem
theorem range_of_f_minus_x :
  Set.range (fun x => f x - x) ∩ (Set.Icc 0 1) = Set.Icc 0 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_f_minus_x_l4023_402310


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l4023_402389

theorem imaginary_part_of_complex_fraction : 
  let i : ℂ := Complex.I
  let z : ℂ := (2 * i) / (1 + i)
  Complex.im z = 1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l4023_402389


namespace NUMINAMATH_CALUDE_disaster_relief_team_selection_l4023_402338

def internal_medicine_doctors : ℕ := 5
def surgeons : ℕ := 6
def total_doctors : ℕ := internal_medicine_doctors + surgeons
def team_size : ℕ := 4

theorem disaster_relief_team_selection :
  (Nat.choose total_doctors team_size) -
  (Nat.choose internal_medicine_doctors team_size) -
  (Nat.choose surgeons team_size) = 310 := by
  sorry

end NUMINAMATH_CALUDE_disaster_relief_team_selection_l4023_402338


namespace NUMINAMATH_CALUDE_ice_floe_mass_l4023_402378

/-- The mass of the ice floe given the polar bear's mass and trajectory diameters -/
theorem ice_floe_mass (m : ℝ) (D d : ℝ) (hm : m = 600) (hD : D = 10) (hd : d = 9.5) :
  (m * d) / (D - d) = 11400 := by
  sorry

end NUMINAMATH_CALUDE_ice_floe_mass_l4023_402378


namespace NUMINAMATH_CALUDE_cups_count_l4023_402322

-- Define the cost of a single paper plate and cup
variable (plate_cost cup_cost : ℝ)

-- Define the number of cups in the second purchase
variable (cups_in_second_purchase : ℕ)

-- First condition: 100 plates and 200 cups cost $7.50
axiom first_purchase : 100 * plate_cost + 200 * cup_cost = 7.50

-- Second condition: 20 plates and cups_in_second_purchase cups cost $1.50
axiom second_purchase : 20 * plate_cost + cups_in_second_purchase * cup_cost = 1.50

-- Theorem to prove
theorem cups_count : cups_in_second_purchase = 40 := by
  sorry

end NUMINAMATH_CALUDE_cups_count_l4023_402322


namespace NUMINAMATH_CALUDE_trim_100_edge_polyhedron_l4023_402396

/-- Represents a polyhedron before and after vertex trimming --/
structure TrimmedPolyhedron where
  initial_edges : ℕ
  is_convex : Bool
  trimmed_vertices : ℕ
  trimmed_edges : ℕ

/-- Represents the process of trimming vertices of a polyhedron --/
def trim_vertices (p : TrimmedPolyhedron) : TrimmedPolyhedron :=
  { p with
    trimmed_vertices := 2 * p.initial_edges,
    trimmed_edges := 3 * p.initial_edges
  }

/-- Theorem stating the result of trimming vertices of a specific polyhedron --/
theorem trim_100_edge_polyhedron :
  ∀ p : TrimmedPolyhedron,
    p.initial_edges = 100 →
    p.is_convex = true →
    (trim_vertices p).trimmed_vertices = 200 ∧
    (trim_vertices p).trimmed_edges = 300 := by
  sorry


end NUMINAMATH_CALUDE_trim_100_edge_polyhedron_l4023_402396


namespace NUMINAMATH_CALUDE_max_red_socks_l4023_402391

/-- The maximum number of red socks in a dresser with specific conditions -/
theorem max_red_socks (t : ℕ) (h1 : t ≤ 2500) :
  let p := 12 / 23
  ∃ r : ℕ, r ≤ t ∧
    (r * (r - 1) + (t - r) * (t - r - 1)) / (t * (t - 1)) = p ∧
    (∀ r' : ℕ, r' ≤ t →
      (r' * (r' - 1) + (t - r') * (t - r' - 1)) / (t * (t - 1)) = p →
      r' ≤ r) ∧
    r = 1225 :=
sorry

end NUMINAMATH_CALUDE_max_red_socks_l4023_402391


namespace NUMINAMATH_CALUDE_rectangle_count_l4023_402357

/-- The number of rows and columns in the square grid -/
def gridSize : ℕ := 5

/-- The number of ways to choose 2 items from n items -/
def choose2 (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of different rectangles in a gridSize x gridSize square array of dots -/
def numRectangles : ℕ := (choose2 gridSize) * (choose2 gridSize)

theorem rectangle_count : numRectangles = 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_count_l4023_402357


namespace NUMINAMATH_CALUDE_vessel_mixture_problem_l4023_402331

theorem vessel_mixture_problem (x : ℝ) : 
  (0 < x) ∧ (x < 8) →
  (((8 * 0.16 - (8 * 0.16) * (x / 8)) - ((8 * 0.16 - (8 * 0.16) * (x / 8)) * (x / 8))) / 8 = 0.09) →
  x = 2 := by sorry

end NUMINAMATH_CALUDE_vessel_mixture_problem_l4023_402331


namespace NUMINAMATH_CALUDE_masha_sasha_numbers_l4023_402394

theorem masha_sasha_numbers : ∃ (a b : ℕ), 
  a ≠ b ∧ 
  a > 11 ∧ 
  b > 11 ∧ 
  (∀ (x y : ℕ), x ≠ y ∧ x > 11 ∧ y > 11 ∧ x + y < a + b → 
    ∃! (p q : ℕ), p ≠ q ∧ p > 11 ∧ q > 11 ∧ p + q = x + y) ∧
  (Even a ∨ Even b) ∧
  (∀ (x y : ℕ), x ≠ y ∧ x > 11 ∧ y > 11 ∧ x + y = a + b → (x = 12 ∧ y = 16) ∨ (x = 16 ∧ y = 12)) :=
by
  sorry

end NUMINAMATH_CALUDE_masha_sasha_numbers_l4023_402394


namespace NUMINAMATH_CALUDE_max_teams_advancing_l4023_402399

/-- The number of teams in the tournament -/
def num_teams : ℕ := 7

/-- The minimum number of points required to advance -/
def min_points_to_advance : ℕ := 13

/-- The number of points awarded for a win -/
def win_points : ℕ := 3

/-- The number of points awarded for a draw -/
def draw_points : ℕ := 1

/-- The number of points awarded for a loss -/
def loss_points : ℕ := 0

/-- The total number of games played in the tournament -/
def total_games : ℕ := (num_teams * (num_teams - 1)) / 2

/-- The maximum total points that can be awarded in the tournament -/
def max_total_points : ℕ := total_games * win_points

/-- Theorem stating the maximum number of teams that can advance -/
theorem max_teams_advancing :
  ∀ n : ℕ, (n * min_points_to_advance ≤ max_total_points) →
  (∀ m : ℕ, m > n → m * min_points_to_advance > max_total_points) →
  n = 4 := by sorry

end NUMINAMATH_CALUDE_max_teams_advancing_l4023_402399


namespace NUMINAMATH_CALUDE_continuous_piecewise_function_l4023_402323

noncomputable def f (c d : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then c * x + 2
  else if x ≥ -1 then 2 * x - 4
  else 3 * x - d

theorem continuous_piecewise_function (c d : ℝ) :
  Continuous (f c d) → c + d = -7 := by
  sorry

end NUMINAMATH_CALUDE_continuous_piecewise_function_l4023_402323


namespace NUMINAMATH_CALUDE_workshop_average_salary_l4023_402350

/-- Represents the average salary of all workers in a workshop -/
def average_salary (total_workers : ℕ) (technicians : ℕ) (technician_salary : ℕ) (other_salary : ℕ) : ℚ :=
  ((technicians * technician_salary + (total_workers - technicians) * other_salary) : ℚ) / total_workers

/-- Theorem stating the average salary of all workers in the workshop -/
theorem workshop_average_salary :
  average_salary 24 8 12000 6000 = 8000 := by
  sorry

end NUMINAMATH_CALUDE_workshop_average_salary_l4023_402350


namespace NUMINAMATH_CALUDE_expression_factorization_l4023_402336

theorem expression_factorization (x : ℝ) : 
  4*x*(x-5) + 5*(x-5) + 6*x*(x-2) = (4*x+5)*(x-5) + 6*x*(x-2) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l4023_402336


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l4023_402335

theorem absolute_value_inequality (x : ℝ) : 
  x ≠ 0 → (|((x - 2) / x)| > ((x - 2) / x) ↔ 0 < x ∧ x < 2) := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l4023_402335


namespace NUMINAMATH_CALUDE_lamp_arrangements_count_l4023_402398

/-- The number of ways to select k items from n items. -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to turn off 3 lamps in a row of 10 lamps,
    where the end lamps must remain on and no two consecutive lamps can be off. -/
def lamp_arrangements : ℕ := choose 6 3

theorem lamp_arrangements_count : lamp_arrangements = 20 := by sorry

end NUMINAMATH_CALUDE_lamp_arrangements_count_l4023_402398


namespace NUMINAMATH_CALUDE_complement_of_A_l4023_402302

def U : Set ℕ := {n | 1 ≤ n ∧ n ≤ 10}
def A : Set ℕ := {1, 2, 3, 5, 8}

theorem complement_of_A : (Aᶜ : Set ℕ) = {4, 6, 7, 9, 10} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l4023_402302


namespace NUMINAMATH_CALUDE_quadratic_roots_and_triangle_perimeter_l4023_402372

/-- The quadratic equation in terms of x and k -/
def quadratic (x k : ℝ) : Prop :=
  x^2 - (3*k + 1)*x + 2*k^2 + 2*k = 0

/-- An isosceles triangle with side lengths a, b, and c -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isIsosceles : (a = b ∧ a ≠ c) ∨ (a = c ∧ a ≠ b) ∨ (b = c ∧ b ≠ a)

/-- The theorem to be proved -/
theorem quadratic_roots_and_triangle_perimeter :
  (∀ k : ℝ, ∃ x : ℝ, quadratic x k) ∧
  (∃ t : IsoscelesTriangle, 
    t.a = 6 ∧
    quadratic t.b (t.b/2) ∧
    quadratic t.c ((t.c - 1)/2) ∧
    (t.a + t.b + t.c = 16 ∨ t.a + t.b + t.c = 22)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_and_triangle_perimeter_l4023_402372


namespace NUMINAMATH_CALUDE_volume_of_region_l4023_402330

-- Define the region in space
def Region : Set (ℝ × ℝ × ℝ) :=
  {p : ℝ × ℝ × ℝ | let (x, y, z) := p
                   (|x + y + 2*z| + |x + y - 2*z| ≤ 12) ∧
                   (x ≥ 0) ∧ (y ≥ 0) ∧ (z ≥ 0)}

-- State the theorem
theorem volume_of_region : MeasureTheory.volume Region = 54 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_region_l4023_402330


namespace NUMINAMATH_CALUDE_stephanie_store_visits_l4023_402359

/-- Represents the number of oranges Stephanie buys per store visit -/
def oranges_per_visit : ℕ := 2

/-- Represents the total number of oranges Stephanie bought last month -/
def total_oranges : ℕ := 16

/-- Represents the number of times Stephanie went to the store -/
def store_visits : ℕ := total_oranges / oranges_per_visit

theorem stephanie_store_visits : store_visits = 8 := by
  sorry

end NUMINAMATH_CALUDE_stephanie_store_visits_l4023_402359


namespace NUMINAMATH_CALUDE_sum_not_arithmetic_l4023_402346

/-- An infinite arithmetic progression -/
def arithmetic_progression (a d : ℝ) : ℕ → ℝ := λ n => a + (n - 1) * d

/-- An infinite geometric progression -/
def geometric_progression (b q : ℝ) : ℕ → ℝ := λ n => b * q^(n - 1)

/-- The sum of an arithmetic and a geometric progression -/
def sum_progression (a d b q : ℝ) : ℕ → ℝ :=
  λ n => arithmetic_progression a d n + geometric_progression b q n

theorem sum_not_arithmetic (a d b q : ℝ) (hq : q ≠ 1) :
  ¬ ∃ (A D : ℝ), ∀ n : ℕ, sum_progression a d b q n = A + (n - 1) * D :=
sorry

end NUMINAMATH_CALUDE_sum_not_arithmetic_l4023_402346


namespace NUMINAMATH_CALUDE_door_opening_proofs_l4023_402383

/-- The number of buttons on the lock -/
def num_buttons : Nat := 10

/-- The number of buttons that need to be pressed simultaneously -/
def buttons_to_press : Nat := 3

/-- Time taken for each attempt in seconds -/
def time_per_attempt : Nat := 2

/-- The total number of possible combinations -/
def total_combinations : Nat := (num_buttons.choose buttons_to_press)

/-- The maximum time needed to try all combinations in seconds -/
def max_time : Nat := total_combinations * time_per_attempt

/-- The average number of attempts needed -/
def avg_attempts : Rat := (1 + total_combinations) / 2

/-- The average time needed in seconds -/
def avg_time : Rat := avg_attempts * time_per_attempt

/-- The maximum number of attempts possible in 60 seconds -/
def max_attempts_in_minute : Nat := 60 / time_per_attempt

theorem door_opening_proofs :
  (max_time = 240) ∧
  (avg_time = 121) ∧
  (max_attempts_in_minute = 30) ∧
  ((max_attempts_in_minute - 1 : Rat) / total_combinations = 29 / 120) := by
  sorry

end NUMINAMATH_CALUDE_door_opening_proofs_l4023_402383


namespace NUMINAMATH_CALUDE_sheridan_cats_proof_l4023_402362

/-- The number of cats Mrs. Sheridan bought -/
def cats_bought (initial final : ℝ) : ℝ := final - initial

theorem sheridan_cats_proof (initial final : ℝ) 
  (h_initial : initial = 11.0) 
  (h_final : final = 54) : 
  cats_bought initial final = 43 := by
  sorry

end NUMINAMATH_CALUDE_sheridan_cats_proof_l4023_402362


namespace NUMINAMATH_CALUDE_ceiling_negative_fraction_cube_l4023_402377

theorem ceiling_negative_fraction_cube : ⌈(-7/4)^3⌉ = -5 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_negative_fraction_cube_l4023_402377


namespace NUMINAMATH_CALUDE_triangle_side_length_l4023_402300

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  -- Conditions
  (2 * b = a + c) →  -- Arithmetic sequence
  (B = π / 6) →      -- Angle B = 30°
  (1 / 2 * a * c * Real.sin B = 3 / 2) →  -- Area of triangle
  -- Conclusion
  b = 1 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l4023_402300


namespace NUMINAMATH_CALUDE_unqualified_weight_l4023_402314

def flour_label_center : ℝ := 25
def flour_label_tolerance : ℝ := 0.25

def is_qualified (weight : ℝ) : Prop :=
  flour_label_center - flour_label_tolerance ≤ weight ∧ 
  weight ≤ flour_label_center + flour_label_tolerance

theorem unqualified_weight : ¬ (is_qualified 25.26) := by
  sorry

end NUMINAMATH_CALUDE_unqualified_weight_l4023_402314


namespace NUMINAMATH_CALUDE_lines_1_and_4_are_perpendicular_l4023_402393

-- Define the slopes of the lines
def slope1 : ℚ := 3 / 4
def slope4 : ℚ := -4 / 3

-- Define the condition for perpendicularity
def are_perpendicular (m1 m2 : ℚ) : Prop := m1 * m2 = -1

-- Theorem statement
theorem lines_1_and_4_are_perpendicular : 
  are_perpendicular slope1 slope4 := by sorry

end NUMINAMATH_CALUDE_lines_1_and_4_are_perpendicular_l4023_402393


namespace NUMINAMATH_CALUDE_different_pairs_eq_48_l4023_402368

/-- The number of distinct mystery novels -/
def mystery_novels : ℕ := 4

/-- The number of distinct fantasy novels -/
def fantasy_novels : ℕ := 4

/-- The number of distinct biographies -/
def biographies : ℕ := 4

/-- The number of genres -/
def num_genres : ℕ := 3

/-- The number of different pairs of books that can be chosen -/
def different_pairs : ℕ := num_genres * mystery_novels * fantasy_novels

theorem different_pairs_eq_48 : different_pairs = 48 := by
  sorry

end NUMINAMATH_CALUDE_different_pairs_eq_48_l4023_402368


namespace NUMINAMATH_CALUDE_division_multiplication_equality_l4023_402379

theorem division_multiplication_equality : -3 / (1/2) * 2 = -12 := by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_equality_l4023_402379


namespace NUMINAMATH_CALUDE_inverse_square_theorem_l4023_402370

/-- A function representing the inverse square relationship between x and y -/
noncomputable def inverse_square_relation (k : ℝ) (y : ℝ) : ℝ := k / y^2

/-- Theorem stating that given the inverse square relationship and a known point,
    we can determine the value of x for y = 3 -/
theorem inverse_square_theorem (k : ℝ) :
  (inverse_square_relation k 4 = 0.5625) →
  (inverse_square_relation k 3 = 1) :=
by
  sorry

#check inverse_square_theorem

end NUMINAMATH_CALUDE_inverse_square_theorem_l4023_402370


namespace NUMINAMATH_CALUDE_morse_code_symbols_l4023_402324

theorem morse_code_symbols : 
  (Finset.range 5).sum (fun i => 2^(i+1)) = 62 :=
sorry

end NUMINAMATH_CALUDE_morse_code_symbols_l4023_402324


namespace NUMINAMATH_CALUDE_square_circle_union_area_l4023_402384

theorem square_circle_union_area (s : Real) (r : Real) :
  s = 12 ∧ r = 12 →
  (s^2) + (π * r^2) - (π * r^2 / 4) = 144 + 108 * π := by
  sorry

end NUMINAMATH_CALUDE_square_circle_union_area_l4023_402384


namespace NUMINAMATH_CALUDE_milan_phone_bill_l4023_402388

/-- Calculates the number of minutes billed given the total bill, monthly fee, and cost per minute. -/
def minutes_billed (total_bill monthly_fee cost_per_minute : ℚ) : ℚ :=
  (total_bill - monthly_fee) / cost_per_minute

/-- Proves that given the specified conditions, the number of minutes billed is 178. -/
theorem milan_phone_bill : minutes_billed 23.36 2 0.12 = 178 := by
  sorry

end NUMINAMATH_CALUDE_milan_phone_bill_l4023_402388


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l4023_402334

/-- The coefficient of x^2 in the expansion of (1+x)^7(1-x) -/
def coefficient_x_squared : ℤ := 14

/-- The expansion of (1+x)^7(1-x) -/
def expansion (x : ℝ) : ℝ := (1 + x)^7 * (1 - x)

theorem coefficient_x_squared_in_expansion :
  (∃ f : ℝ → ℝ, ∃ g : ℝ → ℝ, expansion = λ x => coefficient_x_squared * x^2 + x * f x + g x) :=
sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l4023_402334


namespace NUMINAMATH_CALUDE_polygon_sides_from_interior_angle_sum_l4023_402325

-- Define a convex polygon
structure ConvexPolygon where
  sides : ℕ
  is_convex : sides ≥ 3

-- Define the sum of interior angles for a polygon
def sum_interior_angles (p : ConvexPolygon) : ℝ :=
  180 * (p.sides - 2 : ℝ)

-- Theorem statement
theorem polygon_sides_from_interior_angle_sum (p : ConvexPolygon) 
  (h : sum_interior_angles p - x = 2190)
  (hx : 0 < x ∧ x < 180) : p.sides = 15 := by
  sorry

#check polygon_sides_from_interior_angle_sum

end NUMINAMATH_CALUDE_polygon_sides_from_interior_angle_sum_l4023_402325


namespace NUMINAMATH_CALUDE_division_problem_l4023_402315

theorem division_problem (N x : ℕ) : 
  (N / x = 500) → 
  (N % x = 20) → 
  (4 * 500 + 20 = 2020) → 
  x = 4 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l4023_402315


namespace NUMINAMATH_CALUDE_expand_product_l4023_402376

theorem expand_product (x : ℝ) : (x + 3) * (x^2 + 4*x + 6) = x^3 + 7*x^2 + 18*x + 18 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l4023_402376


namespace NUMINAMATH_CALUDE_same_color_probability_l4023_402328

theorem same_color_probability (total_balls : ℕ) (green_balls : ℕ) (white_balls : ℕ) 
  (h1 : total_balls = green_balls + white_balls)
  (h2 : green_balls = 5)
  (h3 : white_balls = 8) :
  (green_balls * (green_balls - 1) + white_balls * (white_balls - 1)) / (total_balls * (total_balls - 1)) = 19 / 39 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l4023_402328


namespace NUMINAMATH_CALUDE_symmetrical_circles_sin_cos_theta_l4023_402375

/-- Given two circles C₁ and C₂ defined by their equations and a line of symmetry,
    prove that sin θ cos θ = -2/5 --/
theorem symmetrical_circles_sin_cos_theta (a θ : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + a*x = 0 → 2*x - y - 1 = 0) →  -- C₁ is symmetrical about the line
  (∀ x y : ℝ, x^2 + y^2 + 2*a*x + y*(Real.tan θ) = 0 → 2*x - y - 1 = 0) →  -- C₂ is symmetrical about the line
  Real.sin θ * Real.cos θ = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_symmetrical_circles_sin_cos_theta_l4023_402375


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l4023_402380

theorem unique_solution_for_equation : ∃! (x y : ℕ), 1983 = 1982 * x - 1981 * y ∧ x = 11888 ∧ y = 11893 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l4023_402380


namespace NUMINAMATH_CALUDE_larger_number_problem_l4023_402319

theorem larger_number_problem (L S : ℕ) (hL : L > S) :
  L - S = 1365 →
  L = 6 * S + 15 →
  L = 1635 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l4023_402319


namespace NUMINAMATH_CALUDE_computer_operations_l4023_402356

/-- Represents the performance of a computer --/
structure ComputerPerformance where
  additions_per_second : ℕ
  multiplications_per_second : ℕ
  hours : ℕ

/-- Calculates the total number of operations a computer can perform --/
def total_operations (cp : ComputerPerformance) : ℕ :=
  (cp.additions_per_second + cp.multiplications_per_second) * (cp.hours * 3600)

/-- Theorem: A computer with given specifications performs 388,800,000 operations in 3 hours --/
theorem computer_operations :
  ∃ (cp : ComputerPerformance),
    cp.additions_per_second = 12000 ∧
    cp.multiplications_per_second = 2 * cp.additions_per_second ∧
    cp.hours = 3 ∧
    total_operations cp = 388800000 := by
  sorry


end NUMINAMATH_CALUDE_computer_operations_l4023_402356


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l4023_402364

/-- A point in the Cartesian plane lies in the fourth quadrant if and only if
    its x-coordinate is positive and its y-coordinate is negative. -/
def is_in_fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

/-- The point (8, -3) lies in the fourth quadrant of the Cartesian coordinate system. -/
theorem point_in_fourth_quadrant :
  is_in_fourth_quadrant 8 (-3) := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l4023_402364


namespace NUMINAMATH_CALUDE_constant_e_value_l4023_402390

theorem constant_e_value (x y e : ℝ) 
  (h1 : x / (2 * y) = 5 / e)
  (h2 : (7 * x + 4 * y) / (x - 2 * y) = 13) :
  e = 2 := by sorry

end NUMINAMATH_CALUDE_constant_e_value_l4023_402390


namespace NUMINAMATH_CALUDE_tangent_line_polar_equation_l4023_402358

/-- Given a circle in polar form ρ = 4sinθ and a point (2√2, π/4),
    the polar equation of the tangent line passing through this point is ρcosθ = 2 -/
theorem tangent_line_polar_equation
  (ρ θ : ℝ) 
  (circle_eq : ρ = 4 * Real.sin θ) 
  (point : (ρ, θ) = (2 * Real.sqrt 2, Real.pi / 4)) :
  ∃ (k : ℝ), ρ * Real.cos θ = k ∧ k = 2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_polar_equation_l4023_402358
