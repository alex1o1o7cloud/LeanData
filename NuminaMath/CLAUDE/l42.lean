import Mathlib

namespace NUMINAMATH_CALUDE_highlighters_count_l42_4288

/-- The number of pink highlighters -/
def pink_highlighters : Nat := 47

/-- The number of yellow highlighters -/
def yellow_highlighters : Nat := 36

/-- The number of blue highlighters -/
def blue_highlighters : Nat := 21

/-- The number of orange highlighters -/
def orange_highlighters : Nat := 15

/-- The number of green highlighters -/
def green_highlighters : Nat := 27

/-- The total number of highlighters -/
def total_highlighters : Nat :=
  pink_highlighters + yellow_highlighters + blue_highlighters + orange_highlighters + green_highlighters

theorem highlighters_count : total_highlighters = 146 := by
  sorry

end NUMINAMATH_CALUDE_highlighters_count_l42_4288


namespace NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l42_4209

theorem sum_of_solutions_is_zero (x₁ x₂ : ℝ) :
  (8 : ℝ) = 8 →
  x₁^2 + 8^2 = 144 →
  x₂^2 + 8^2 = 144 →
  x₁ + x₂ = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l42_4209


namespace NUMINAMATH_CALUDE_photos_after_bali_trip_l42_4259

/-- Calculates the total number of photos after a trip -/
def total_photos_after_trip (initial_photos : ℕ) (first_week : ℕ) (third_fourth_weeks : ℕ) : ℕ :=
  initial_photos + first_week + 2 * first_week + third_fourth_weeks

/-- Theorem stating the total number of photos after the trip -/
theorem photos_after_bali_trip :
  total_photos_after_trip 100 50 80 = 330 := by
  sorry

end NUMINAMATH_CALUDE_photos_after_bali_trip_l42_4259


namespace NUMINAMATH_CALUDE_equation_solution_l42_4284

theorem equation_solution (x : ℝ) : 
  (3 * x + 6 = |(-20 + 2 * x - 3)|) ↔ (x = -29 ∨ x = 17/5) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l42_4284


namespace NUMINAMATH_CALUDE_count_lattice_points_on_hyperbola_l42_4201

def lattice_points_on_hyperbola : ℕ :=
  let a := 3000
  2 * (((2 + 1) * (2 + 1) * (6 + 1)) : ℕ)

theorem count_lattice_points_on_hyperbola :
  lattice_points_on_hyperbola = 126 := by sorry

end NUMINAMATH_CALUDE_count_lattice_points_on_hyperbola_l42_4201


namespace NUMINAMATH_CALUDE_no_integer_roots_l42_4235

theorem no_integer_roots : ∀ x : ℤ, x^3 - 5*x^2 - 11*x + 35 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_roots_l42_4235


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l42_4205

theorem sqrt_equation_solution :
  ∃ x : ℝ, (Real.sqrt x) / 19 = 4 ∧ x = 5776 := by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l42_4205


namespace NUMINAMATH_CALUDE_ricas_prize_fraction_l42_4255

theorem ricas_prize_fraction (total_prize : ℚ) (rica_remaining : ℚ) :
  total_prize = 1000 →
  rica_remaining = 300 →
  ∃ (rica_fraction : ℚ),
    rica_fraction * total_prize * (4/5) = rica_remaining ∧
    rica_fraction = 3/8 :=
by sorry

end NUMINAMATH_CALUDE_ricas_prize_fraction_l42_4255


namespace NUMINAMATH_CALUDE_tiles_difference_l42_4287

/-- Represents the number of tiles in the nth square of the progression -/
def tiles (n : ℕ) : ℕ := n^2

/-- The difference in the number of tiles between the 8th and 6th squares is 28 -/
theorem tiles_difference : tiles 8 - tiles 6 = 28 := by
  sorry

end NUMINAMATH_CALUDE_tiles_difference_l42_4287


namespace NUMINAMATH_CALUDE_beef_weight_before_processing_l42_4237

theorem beef_weight_before_processing 
  (weight_after : ℝ) 
  (percent_lost : ℝ) 
  (h1 : weight_after = 750)
  (h2 : percent_lost = 50) : 
  weight_after / (1 - percent_lost / 100) = 1500 :=
by sorry

end NUMINAMATH_CALUDE_beef_weight_before_processing_l42_4237


namespace NUMINAMATH_CALUDE_smallest_B_for_divisibility_l42_4250

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def number (B C : ℕ) : ℕ := 4000000 + 100000 * B + 80000 + 3000 + 900 + 90 + C

theorem smallest_B_for_divisibility :
  ∃ (C : ℕ), is_digit C ∧ number 0 C % 3 = 0 ∧
  ∀ (B : ℕ), is_digit B → (∃ (C : ℕ), is_digit C ∧ number B C % 3 = 0) → B ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_smallest_B_for_divisibility_l42_4250


namespace NUMINAMATH_CALUDE_angle_point_m_value_l42_4221

theorem angle_point_m_value (θ : Real) (m : Real) :
  let P : Prod Real Real := (-Real.sqrt 3, m)
  (∃ (r : Real), r > 0 ∧ P.1^2 + P.2^2 = r^2) →  -- P is on the terminal side of θ
  Real.sin θ = Real.sqrt 13 / 13 →
  m = 1/2 := by
sorry

end NUMINAMATH_CALUDE_angle_point_m_value_l42_4221


namespace NUMINAMATH_CALUDE_simplify_inverse_sum_l42_4294

theorem simplify_inverse_sum (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = x⁻¹ * y⁻¹ * z⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_simplify_inverse_sum_l42_4294


namespace NUMINAMATH_CALUDE_ratio_x_y_l42_4219

theorem ratio_x_y (x y : ℝ) (h : (0.6 * 500 : ℝ) = 0.5 * x ∧ (0.6 * 500 : ℝ) = 0.4 * y) : 
  x / y = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_x_y_l42_4219


namespace NUMINAMATH_CALUDE_surface_area_of_revolution_l42_4298

theorem surface_area_of_revolution (S α : ℝ) (h1 : S > 0) (h2 : 0 < α ∧ α < 2 * π) :
  let surface_area := (8 * π * S * Real.sin (α / 4)^2 * (1 + Real.cos (α / 4)^2)) / (α - Real.sin α)
  ∃ (R : ℝ), R > 0 ∧
    S = R^2 / 2 * (α - Real.sin α) ∧
    surface_area = 2 * π * R * (R * (1 - Real.cos (α / 2))) + π * (R * Real.sin (α / 2))^2 :=
by sorry

end NUMINAMATH_CALUDE_surface_area_of_revolution_l42_4298


namespace NUMINAMATH_CALUDE_thirteen_power_division_l42_4218

theorem thirteen_power_division : (13 : ℕ) ^ 8 / (13 : ℕ) ^ 5 = 2197 := by sorry

end NUMINAMATH_CALUDE_thirteen_power_division_l42_4218


namespace NUMINAMATH_CALUDE_final_value_of_A_l42_4256

/-- Given an initial value of A and an operation, prove the final value of A -/
theorem final_value_of_A (initial_A : Int) : 
  let A₁ := initial_A
  let A₂ := -A₁ + 10
  A₂ = -10 :=
by sorry

end NUMINAMATH_CALUDE_final_value_of_A_l42_4256


namespace NUMINAMATH_CALUDE_total_dolls_l42_4244

/-- The number of dolls each person has -/
structure DollCounts where
  lisa : ℕ
  vera : ℕ
  sophie : ℕ
  aida : ℕ

/-- The conditions of the doll ownership -/
def validDollCounts (d : DollCounts) : Prop :=
  d.lisa = 20 ∧
  d.vera = 2 * d.lisa ∧
  d.sophie = 2 * d.vera ∧
  d.aida = 2 * d.sophie

/-- The theorem stating the total number of dolls -/
theorem total_dolls (d : DollCounts) (h : validDollCounts d) :
  d.lisa + d.vera + d.sophie + d.aida = 300 :=
by sorry

end NUMINAMATH_CALUDE_total_dolls_l42_4244


namespace NUMINAMATH_CALUDE_sphere_only_all_circular_views_l42_4200

-- Define the geometric shapes
inductive GeometricShape
  | Cuboid
  | Cylinder
  | Cone
  | Sphere

-- Define the view types
inductive ViewType
  | Front
  | Left
  | Top

-- Define a function to check if a view is circular
def isCircularView (shape : GeometricShape) (view : ViewType) : Prop :=
  match shape, view with
  | GeometricShape.Sphere, _ => true
  | GeometricShape.Cylinder, ViewType.Top => true
  | GeometricShape.Cone, ViewType.Top => true
  | _, _ => false

-- Define a function to check if all three views are circular
def hasAllCircularViews (shape : GeometricShape) : Prop :=
  isCircularView shape ViewType.Front ∧
  isCircularView shape ViewType.Left ∧
  isCircularView shape ViewType.Top

-- Theorem: Only the sphere has circular views in all three perspectives
theorem sphere_only_all_circular_views :
  ∀ (shape : GeometricShape),
    hasAllCircularViews shape ↔ shape = GeometricShape.Sphere :=
by sorry

end NUMINAMATH_CALUDE_sphere_only_all_circular_views_l42_4200


namespace NUMINAMATH_CALUDE_expression_simplification_l42_4222

theorem expression_simplification (a b : ℝ) 
  (ha : a = 2 + Real.sqrt 3) 
  (hb : b = 2 - Real.sqrt 3) : 
  (a^2 - b^2) / a / (a - (2*a*b - b^2) / a) = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l42_4222


namespace NUMINAMATH_CALUDE_square_area_rational_l42_4238

theorem square_area_rational (s : ℚ) : ∃ (a : ℚ), a = s^2 :=
  sorry

end NUMINAMATH_CALUDE_square_area_rational_l42_4238


namespace NUMINAMATH_CALUDE_quiz_score_theorem_l42_4207

/-- Represents a quiz with scoring rules and results -/
structure Quiz where
  totalQuestions : ℕ
  correctPoints : ℕ
  incorrectPoints : ℕ
  totalScore : ℤ

/-- Calculates the score based on the number of correct answers -/
def calculateScore (q : Quiz) (correctAnswers : ℕ) : ℤ :=
  (correctAnswers : ℤ) * q.correctPoints - (q.totalQuestions - correctAnswers : ℤ) * q.incorrectPoints

/-- Theorem: Given the quiz parameters, 15 correct answers result in a score of 70 -/
theorem quiz_score_theorem (q : Quiz) 
    (h1 : q.totalQuestions = 20)
    (h2 : q.correctPoints = 5)
    (h3 : q.incorrectPoints = 1)
    (h4 : q.totalScore = 70) :
  calculateScore q 15 = q.totalScore := by
  sorry


end NUMINAMATH_CALUDE_quiz_score_theorem_l42_4207


namespace NUMINAMATH_CALUDE_smallest_constant_inequality_l42_4280

theorem smallest_constant_inequality (a b c d e : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) :
  Real.sqrt (a / (b + c + d + e)) +
  Real.sqrt (b / (a + c + d + e)) +
  Real.sqrt (c / (a + b + d + e)) +
  Real.sqrt (d / (a + b + c + e)) +
  Real.sqrt (e / (a + b + c + d)) ≥ 2 ∧
  ∀ m : ℝ, m < 2 → ∃ a' b' c' d' e' : ℝ, a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ d' > 0 ∧ e' > 0 ∧
    Real.sqrt (a' / (b' + c' + d' + e')) +
    Real.sqrt (b' / (a' + c' + d' + e')) +
    Real.sqrt (c' / (a' + b' + d' + e')) +
    Real.sqrt (d' / (a' + b' + c' + e')) +
    Real.sqrt (e' / (a' + b' + c' + d')) < m :=
by sorry

end NUMINAMATH_CALUDE_smallest_constant_inequality_l42_4280


namespace NUMINAMATH_CALUDE_find_A_in_three_digit_sum_l42_4275

theorem find_A_in_three_digit_sum (A B : ℕ) : 
  (100 ≤ A * 100 + 70 + B) ∧ 
  (A * 100 + 70 + B < 1000) ∧ 
  (32 + A * 100 + 70 + B = 705) → 
  A = 6 := by
sorry

end NUMINAMATH_CALUDE_find_A_in_three_digit_sum_l42_4275


namespace NUMINAMATH_CALUDE_sqrt_31_minus_2_range_l42_4263

theorem sqrt_31_minus_2_range : 
  (∃ x : ℝ, x = Real.sqrt 31 ∧ 5 < x ∧ x < 6) → 
  3 < Real.sqrt 31 - 2 ∧ Real.sqrt 31 - 2 < 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_31_minus_2_range_l42_4263


namespace NUMINAMATH_CALUDE_least_possible_third_side_l42_4257

theorem least_possible_third_side (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a = 8 → b = 15 →
  (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) →
  c ≥ 8 :=
by sorry

end NUMINAMATH_CALUDE_least_possible_third_side_l42_4257


namespace NUMINAMATH_CALUDE_expression_equality_l42_4292

theorem expression_equality : 
  Real.sqrt 27 / (Real.sqrt 3 / 2) * (2 * Real.sqrt 2) - 6 * Real.sqrt 2 = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l42_4292


namespace NUMINAMATH_CALUDE_only_3_and_4_propositional_l42_4293

-- Define a type for statements
inductive Statement
| Question : Statement
| Imperative : Statement
| Declarative : Statement

-- Define a function to check if a statement is propositional
def isPropositional (s : Statement) : Prop :=
  match s with
  | Statement.Declarative => True
  | _ => False

-- Define our four statements
def statement1 : Statement := Statement.Question
def statement2 : Statement := Statement.Imperative
def statement3 : Statement := Statement.Declarative
def statement4 : Statement := Statement.Declarative

-- Theorem to prove
theorem only_3_and_4_propositional :
  isPropositional statement1 = False ∧
  isPropositional statement2 = False ∧
  isPropositional statement3 = True ∧
  isPropositional statement4 = True := by
  sorry


end NUMINAMATH_CALUDE_only_3_and_4_propositional_l42_4293


namespace NUMINAMATH_CALUDE_keaton_orange_earnings_l42_4234

/-- Represents Keaton's farm earnings -/
structure FarmEarnings where
  months_between_orange_harvests : ℕ
  months_between_apple_harvests : ℕ
  apple_harvest_earnings : ℕ
  total_annual_earnings : ℕ

/-- Calculates the earnings from each orange harvest -/
def orange_harvest_earnings (farm : FarmEarnings) : ℕ :=
  let orange_harvests_per_year := 12 / farm.months_between_orange_harvests
  let apple_harvests_per_year := 12 / farm.months_between_apple_harvests
  let annual_apple_earnings := apple_harvests_per_year * farm.apple_harvest_earnings
  let annual_orange_earnings := farm.total_annual_earnings - annual_apple_earnings
  annual_orange_earnings / orange_harvests_per_year

/-- Theorem stating that Keaton's orange harvest earnings are $50 -/
theorem keaton_orange_earnings :
  let farm := FarmEarnings.mk 2 3 30 420
  orange_harvest_earnings farm = 50 := by
  sorry

end NUMINAMATH_CALUDE_keaton_orange_earnings_l42_4234


namespace NUMINAMATH_CALUDE_h_range_l42_4249

/-- A quadratic function passing through three points with specific y-value relationships -/
structure QuadraticFunction where
  h : ℝ
  k : ℝ
  y₁ : ℝ
  y₂ : ℝ
  y₃ : ℝ
  eq_p₁ : y₁ = ((-3) - h)^2 + k
  eq_p₂ : y₂ = ((-1) - h)^2 + k
  eq_p₃ : y₃ = (1 - h)^2 + k
  y_order : y₂ < y₁ ∧ y₁ < y₃

/-- The range of h for the quadratic function -/
theorem h_range (f : QuadraticFunction) : -2 < f.h ∧ f.h < -1 := by
  sorry

end NUMINAMATH_CALUDE_h_range_l42_4249


namespace NUMINAMATH_CALUDE_max_true_statements_l42_4260

theorem max_true_statements (x : ℝ) : ∃ x : ℝ,
  (0 < x^2 ∧ x^2 < 1) ∧ (0 < x ∧ x < 1) ∧ (0 < 2*x - x^2 ∧ 2*x - x^2 < 2) ∧
  ¬∃ y : ℝ, (((0 < y^2 ∧ y^2 < 1) ∧ (y^2 > 1) ∧ (-1 < y ∧ y < 0) ∧ (0 < y ∧ y < 1) ∧ (0 < 2*y - y^2 ∧ 2*y - y^2 < 2)) ∨
            ((0 < y^2 ∧ y^2 < 1) ∧ (y^2 > 1) ∧ (-1 < y ∧ y < 0) ∧ (0 < y ∧ y < 1)) ∨
            ((0 < y^2 ∧ y^2 < 1) ∧ (y^2 > 1) ∧ (-1 < y ∧ y < 0) ∧ (0 < 2*y - y^2 ∧ 2*y - y^2 < 2)) ∨
            ((0 < y^2 ∧ y^2 < 1) ∧ (y^2 > 1) ∧ (0 < y ∧ y < 1) ∧ (0 < 2*y - y^2 ∧ 2*y - y^2 < 2)) ∨
            ((0 < y^2 ∧ y^2 < 1) ∧ (-1 < y ∧ y < 0) ∧ (0 < y ∧ y < 1) ∧ (0 < 2*y - y^2 ∧ 2*y - y^2 < 2)) ∨
            ((y^2 > 1) ∧ (-1 < y ∧ y < 0) ∧ (0 < y ∧ y < 1) ∧ (0 < 2*y - y^2 ∧ 2*y - y^2 < 2))) :=
by sorry

end NUMINAMATH_CALUDE_max_true_statements_l42_4260


namespace NUMINAMATH_CALUDE_same_color_probability_l42_4281

/-- Represents the number of jelly beans of each color for a person -/
structure JellyBeans where
  green : ℕ
  red : ℕ
  yellow : ℕ

/-- Calculates the total number of jelly beans a person has -/
def JellyBeans.total (jb : JellyBeans) : ℕ := jb.green + jb.red + jb.yellow

/-- Abe's jelly bean distribution -/
def abe : JellyBeans := { green := 2, red := 2, yellow := 0 }

/-- Bob's jelly bean distribution -/
def bob : JellyBeans := { green := 2, red := 3, yellow := 1 }

/-- Calculates the probability of picking the same color jelly bean -/
def probSameColor (jb1 jb2 : JellyBeans) : ℚ :=
  (jb1.green * jb2.green + jb1.red * jb2.red) / (jb1.total * jb2.total)

theorem same_color_probability :
  probSameColor abe bob = 5/12 := by sorry

end NUMINAMATH_CALUDE_same_color_probability_l42_4281


namespace NUMINAMATH_CALUDE_steps_in_flight_l42_4282

/-- The number of steps in each flight of stairs -/
def steps_per_flight : ℕ := sorry

/-- The height of each step in inches -/
def step_height : ℕ := 8

/-- The number of flights Jack goes up -/
def flights_up : ℕ := 3

/-- The number of flights Jack goes down -/
def flights_down : ℕ := 6

/-- The total vertical distance traveled in inches -/
def total_distance : ℕ := 24 * 12

theorem steps_in_flight :
  steps_per_flight * step_height * (flights_down - flights_up) = total_distance ∧
  steps_per_flight = 108 := by sorry

end NUMINAMATH_CALUDE_steps_in_flight_l42_4282


namespace NUMINAMATH_CALUDE_original_salary_is_twenty_thousand_l42_4297

/-- Calculates the original salary of employees given the conditions of Emily's salary change --/
def calculate_original_salary (emily_original_salary : ℕ) (emily_new_salary : ℕ) (num_employees : ℕ) (new_employee_salary : ℕ) : ℕ :=
  let salary_difference := emily_original_salary - emily_new_salary
  let salary_increase_per_employee := salary_difference / num_employees
  new_employee_salary - salary_increase_per_employee

/-- Theorem stating that given the problem conditions, the original salary of each employee was $20,000 --/
theorem original_salary_is_twenty_thousand :
  calculate_original_salary 1000000 850000 10 35000 = 20000 := by
  sorry

end NUMINAMATH_CALUDE_original_salary_is_twenty_thousand_l42_4297


namespace NUMINAMATH_CALUDE_f_negative_solution_l42_4245

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain : Set ℝ := {x : ℝ | -5 ≤ x ∧ x ≤ 5}

-- State that f is odd
axiom f_odd : ∀ x ∈ domain, f (-x) = -f x

-- Define the set where f(x) < 0 for x ∈ [0, 5]
def negative_range : Set ℝ := {x : ℝ | ((-2 < x ∧ x < 0) ∨ (2 < x ∧ x ≤ 5)) ∧ f x < 0}

-- Theorem to prove
theorem f_negative_solution :
  {x ∈ domain | f x < 0} = {x : ℝ | (-5 < x ∧ x < -2) ∨ (-2 < x ∧ x < 0) ∨ (2 < x ∧ x ≤ 5)} :=
sorry

end NUMINAMATH_CALUDE_f_negative_solution_l42_4245


namespace NUMINAMATH_CALUDE_smallest_prime_perimeter_triangle_l42_4258

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that checks if three numbers are consecutive primes -/
def areConsecutivePrimes (a b c : ℕ) : Prop :=
  isPrime a ∧ isPrime b ∧ isPrime c ∧ b = a + 2 ∧ c = b + 2

/-- The theorem stating the smallest perimeter of a scalene triangle with consecutive prime side lengths and prime perimeter -/
theorem smallest_prime_perimeter_triangle :
  ∀ a b c : ℕ,
    areConsecutivePrimes a b c →
    isPrime (a + b + c) →
    a + b + c ≥ 23 :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_perimeter_triangle_l42_4258


namespace NUMINAMATH_CALUDE_rectangular_box_surface_area_l42_4212

theorem rectangular_box_surface_area 
  (a b c : ℝ) 
  (h1 : 4 * (a + b + c) = 160) 
  (h2 : Real.sqrt (a^2 + b^2 + c^2) = 25) : 
  2 * (a * b + b * c + c * a) = 975 := by
sorry

end NUMINAMATH_CALUDE_rectangular_box_surface_area_l42_4212


namespace NUMINAMATH_CALUDE_restaurant_bill_calculation_l42_4261

theorem restaurant_bill_calculation (num_people : ℕ) (cost_per_person : ℝ) (gratuity_percentage : ℝ) :
  num_people = 6 →
  cost_per_person = 100 →
  gratuity_percentage = 0.20 →
  num_people * cost_per_person * (1 + gratuity_percentage) = 720 := by
sorry

end NUMINAMATH_CALUDE_restaurant_bill_calculation_l42_4261


namespace NUMINAMATH_CALUDE_exactly_two_primes_in_ten_consecutive_l42_4223

/-- A function that determines if a number is prime --/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that counts the number of primes in a list of natural numbers --/
def countPrimes (list : List ℕ) : ℕ := sorry

/-- A function that generates a list of 10 consecutive numbers starting from a given number --/
def consecutiveNumbers (start : ℕ) : List ℕ := sorry

/-- The theorem to be proved --/
theorem exactly_two_primes_in_ten_consecutive : 
  (Finset.filter (fun k => countPrimes (consecutiveNumbers k) = 2) (Finset.range 21)).card = 4 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_primes_in_ten_consecutive_l42_4223


namespace NUMINAMATH_CALUDE_point_on_curve_l42_4283

theorem point_on_curve : 
  let x : ℝ := Real.sqrt 2
  let y : ℝ := Real.sqrt 2
  x^2 + y^2 - 3*x*y + 2 = 0 := by
sorry

end NUMINAMATH_CALUDE_point_on_curve_l42_4283


namespace NUMINAMATH_CALUDE_cube_edge_length_in_water_l42_4295

/-- Theorem: Edge length of a cube immersed in water --/
theorem cube_edge_length_in_water 
  (base_length : ℝ) (base_width : ℝ) (water_rise : ℝ) (a : ℝ) :
  base_length = 20 →
  base_width = 15 →
  water_rise = 11.25 →
  a^3 = base_length * base_width * water_rise →
  a = 15 :=
by sorry

end NUMINAMATH_CALUDE_cube_edge_length_in_water_l42_4295


namespace NUMINAMATH_CALUDE_unique_base_property_l42_4278

theorem unique_base_property (a : ℕ) (b : ℕ) (h1 : a > 0) (h2 : b > 1) :
  (a * b + (a + 1) = (a + 2) * (a + 3)) → b = 10 :=
by sorry

end NUMINAMATH_CALUDE_unique_base_property_l42_4278


namespace NUMINAMATH_CALUDE_square_all_digits_odd_iff_one_or_three_l42_4210

/-- A function that returns true if all digits in the decimal representation of a natural number are odd -/
def allDigitsOdd (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d % 2 = 1

/-- Theorem stating that for a positive integer n, all digits in n^2 are odd if and only if n = 1 or n = 3 -/
theorem square_all_digits_odd_iff_one_or_three (n : ℕ) (hn : n > 0) :
  allDigitsOdd (n^2) ↔ n = 1 ∨ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_all_digits_odd_iff_one_or_three_l42_4210


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_one_l42_4265

theorem sin_cos_sum_equals_one : 
  Real.sin (47 * π / 180) * Real.cos (43 * π / 180) + 
  Real.sin (137 * π / 180) * Real.sin (43 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_one_l42_4265


namespace NUMINAMATH_CALUDE_geometric_sequence_inequality_l42_4267

theorem geometric_sequence_inequality (a : Fin 8 → ℝ) (q : ℝ) :
  (∀ i : Fin 8, a i > 0) →
  (∀ i : Fin 7, a (i + 1) = a i * q) →
  q ≠ 1 →
  a 0 + a 7 > a 3 + a 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_inequality_l42_4267


namespace NUMINAMATH_CALUDE_inequality_of_powers_l42_4228

theorem inequality_of_powers (a b c : ℝ) 
  (ha : a ≠ 1) (hb : b ≠ 1) (hc : c ≠ 1) 
  (h_order : a > b ∧ b > c ∧ c > 0) : a^b > c^b := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_powers_l42_4228


namespace NUMINAMATH_CALUDE_book_price_proof_l42_4272

theorem book_price_proof (P V T : ℝ) 
  (vasya_short : V + 150 = P)
  (tolya_short : T + 200 = P)
  (exchange_scenario : V + T / 2 - P = 100) : 
  P = 700 := by
  sorry

end NUMINAMATH_CALUDE_book_price_proof_l42_4272


namespace NUMINAMATH_CALUDE_election_votes_theorem_l42_4274

/-- Represents an election with two candidates -/
structure Election where
  total_votes : ℕ
  winning_percentage : ℚ
  vote_majority : ℕ

/-- Theorem: If the winning candidate receives 70% of the votes and wins by a 320 vote majority,
    then the total number of votes polled is 800. -/
theorem election_votes_theorem (e : Election) 
  (h1 : e.winning_percentage = 70 / 100)
  (h2 : e.vote_majority = 320) :
  e.total_votes = 800 := by
sorry

end NUMINAMATH_CALUDE_election_votes_theorem_l42_4274


namespace NUMINAMATH_CALUDE_prob_three_red_before_green_l42_4273

/-- A hat containing red and green chips -/
structure ChipHat :=
  (red_chips : ℕ)
  (green_chips : ℕ)

/-- The probability of drawing all red chips before all green chips -/
def prob_all_red_before_green (hat : ChipHat) : ℚ :=
  sorry

/-- The theorem to prove -/
theorem prob_three_red_before_green :
  let hat := ChipHat.mk 3 3
  prob_all_red_before_green hat = 1/2 :=
sorry

end NUMINAMATH_CALUDE_prob_three_red_before_green_l42_4273


namespace NUMINAMATH_CALUDE_tan_alpha_point_one_two_l42_4243

/-- For an angle α whose terminal side passes through the point (1,2), tan α = 2 -/
theorem tan_alpha_point_one_two (α : Real) :
  (∃ (P : ℝ × ℝ), P.1 = 1 ∧ P.2 = 2 ∧ (∃ (r : ℝ), r > 0 ∧ P = (r * Real.cos α, r * Real.sin α))) →
  Real.tan α = 2 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_point_one_two_l42_4243


namespace NUMINAMATH_CALUDE_expression_value_l42_4215

theorem expression_value (x : ℝ) (h : x^2 + 3*x = 3) : -3*x^2 - 9*x - 2 = -11 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l42_4215


namespace NUMINAMATH_CALUDE_cube_with_holes_surface_area_l42_4241

/-- Represents a cube with holes cut through each face. -/
structure CubeWithHoles where
  cubeSideLength : ℝ
  holeSideLength : ℝ

/-- Calculates the total surface area of a cube with holes. -/
def totalSurfaceArea (c : CubeWithHoles) : ℝ :=
  let originalSurfaceArea := 6 * c.cubeSideLength ^ 2
  let holeArea := 6 * c.holeSideLength ^ 2
  let remainingExteriorArea := originalSurfaceArea - holeArea
  let interiorSurfaceArea := 6 * 4 * c.holeSideLength * c.cubeSideLength
  remainingExteriorArea + interiorSurfaceArea

/-- Theorem stating that the total surface area of the given cube with holes is 72 square meters. -/
theorem cube_with_holes_surface_area :
  totalSurfaceArea { cubeSideLength := 3, holeSideLength := 1 } = 72 := by
  sorry

end NUMINAMATH_CALUDE_cube_with_holes_surface_area_l42_4241


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_meaningful_l42_4277

theorem sqrt_x_minus_one_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 1) ↔ x ≥ 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_meaningful_l42_4277


namespace NUMINAMATH_CALUDE_five_cube_grid_toothpicks_l42_4231

/-- Calculates the number of toothpicks needed for a cube-shaped grid --/
def toothpicks_for_cube_grid (n : ℕ) : ℕ :=
  let vertical_toothpicks := (n + 1)^2 * n
  let horizontal_toothpicks := 2 * (n + 1) * (n + 1) * n
  vertical_toothpicks + horizontal_toothpicks

/-- Theorem stating that a 5x5x5 cube grid requires 2340 toothpicks --/
theorem five_cube_grid_toothpicks :
  toothpicks_for_cube_grid 5 = 2340 := by
  sorry


end NUMINAMATH_CALUDE_five_cube_grid_toothpicks_l42_4231


namespace NUMINAMATH_CALUDE_compare_fractions_l42_4211

theorem compare_fractions (a b c d e : ℝ) 
  (h1 : a > b) (h2 : b > 0) 
  (h3 : c < d) (h4 : d < 0) 
  (h5 : e < 0) : 
  e / (a - c) > e / (b - d) := by
  sorry

end NUMINAMATH_CALUDE_compare_fractions_l42_4211


namespace NUMINAMATH_CALUDE_age_ratio_sydney_sherry_l42_4279

/-- Given the ages of Randolph, Sydney, and Sherry, prove the ratio of Sydney's age to Sherry's age -/
theorem age_ratio_sydney_sherry :
  ∀ (randolph sydney sherry : ℕ),
    randolph = sydney + 5 →
    randolph = 55 →
    sherry = 25 →
    sydney / sherry = 2 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_sydney_sherry_l42_4279


namespace NUMINAMATH_CALUDE_max_sum_of_roots_l42_4214

theorem max_sum_of_roots (x y z : ℝ) 
  (sum_eq : x + y + z = 0)
  (x_ge : x ≥ -1/2)
  (y_ge : y ≥ -1)
  (z_ge : z ≥ -3/2) :
  (∀ a b c : ℝ, a + b + c = 0 → a ≥ -1/2 → b ≥ -1 → c ≥ -3/2 → 
    Real.sqrt (4*a + 2) + Real.sqrt (4*b + 4) + Real.sqrt (4*c + 6) ≤ 
    Real.sqrt (4*x + 2) + Real.sqrt (4*y + 4) + Real.sqrt (4*z + 6)) ∧
  Real.sqrt (4*x + 2) + Real.sqrt (4*y + 4) + Real.sqrt (4*z + 6) = 6 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_roots_l42_4214


namespace NUMINAMATH_CALUDE_average_marks_chemistry_mathematics_l42_4224

theorem average_marks_chemistry_mathematics 
  (P C M : ℕ) -- Marks in Physics, Chemistry, and Mathematics
  (h1 : P + C + M = P + 150) -- Total marks condition
  : (C + M) / 2 = 75 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_chemistry_mathematics_l42_4224


namespace NUMINAMATH_CALUDE_rudolph_encountered_two_stop_signs_per_mile_l42_4236

/-- Rudolph's car trip across town -/
def rudolph_trip (miles : ℕ) (stop_signs : ℕ) : Prop :=
  miles = 5 + 2 ∧ stop_signs = 17 - 3

/-- The number of stop signs per mile -/
def stop_signs_per_mile (miles : ℕ) (stop_signs : ℕ) : ℚ :=
  stop_signs / miles

/-- Theorem: Rudolph encountered 2 stop signs per mile -/
theorem rudolph_encountered_two_stop_signs_per_mile :
  ∀ (miles : ℕ) (stop_signs : ℕ),
  rudolph_trip miles stop_signs →
  stop_signs_per_mile miles stop_signs = 2 := by
  sorry

end NUMINAMATH_CALUDE_rudolph_encountered_two_stop_signs_per_mile_l42_4236


namespace NUMINAMATH_CALUDE_complex_number_properties_l42_4269

theorem complex_number_properties (z : ℂ) (h : (Complex.I - 1) * z = 2 * Complex.I) : 
  (Complex.abs z = Real.sqrt 2) ∧ (z^2 - 2*z + 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_properties_l42_4269


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l42_4266

theorem complex_modulus_problem (m n : ℝ) : 
  (m / (1 + Complex.I)) = (1 - n * Complex.I) → 
  Complex.abs (m + n * Complex.I) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l42_4266


namespace NUMINAMATH_CALUDE_apple_prices_l42_4206

/-- The unit price of Zhao Tong apples in yuan per jin -/
def zhao_tong_price : ℚ := 5

/-- The unit price of Tianma from Xiaocaoba in yuan per jin -/
def tianma_price : ℚ := 50

/-- The original purchase price of apples in yuan per jin -/
def original_purchase_price : ℚ := 4

/-- The cost of 1 jin of Zhao Tong apples and 2 jin of Tianma from Xiaocaoba -/
def cost1 : ℚ := 105

/-- The cost of 3 jin of Zhao Tong apples and 5 jin of Tianma from Xiaocaoba -/
def cost2 : ℚ := 265

/-- The original cost for transporting apples -/
def original_transport_cost : ℚ := 240

/-- The new cost for transporting apples -/
def new_transport_cost : ℚ := 300

/-- The increase in purchase price -/
def price_increase : ℚ := 1

theorem apple_prices :
  (zhao_tong_price + 2 * tianma_price = cost1) ∧
  (3 * zhao_tong_price + 5 * tianma_price = cost2) ∧
  (original_transport_cost / original_purchase_price = new_transport_cost / (original_purchase_price + price_increase)) := by
  sorry

end NUMINAMATH_CALUDE_apple_prices_l42_4206


namespace NUMINAMATH_CALUDE_lidia_remaining_money_l42_4296

/-- Calculates the remaining money after buying apps -/
def remaining_money (app_cost : ℕ) (num_apps : ℕ) (initial_money : ℕ) : ℕ :=
  initial_money - app_cost * num_apps

/-- Proves that Lidia will have $6 left after buying apps -/
theorem lidia_remaining_money :
  let app_cost : ℕ := 4
  let num_apps : ℕ := 15
  let initial_money : ℕ := 66
  remaining_money app_cost num_apps initial_money = 6 := by
sorry

end NUMINAMATH_CALUDE_lidia_remaining_money_l42_4296


namespace NUMINAMATH_CALUDE_codecracker_combinations_l42_4299

/-- The number of different colors of pegs available in CodeCracker -/
def num_colors : ℕ := 6

/-- The number of slots in a CodeCracker code -/
def num_slots : ℕ := 5

/-- The number of possible secret codes in CodeCracker -/
def num_codes : ℕ := num_colors ^ num_slots

theorem codecracker_combinations : num_codes = 7776 := by
  sorry

end NUMINAMATH_CALUDE_codecracker_combinations_l42_4299


namespace NUMINAMATH_CALUDE_sector_to_cone_area_ratio_l42_4233

/-- Given a sector with central angle 135° and area S₁, when formed into a cone
    with total surface area S₂, prove that S₁/S₂ = 8/11 -/
theorem sector_to_cone_area_ratio :
  ∀ (S₁ S₂ : ℝ),
  S₁ > 0 → S₂ > 0 →
  (∃ (r : ℝ), r > 0 ∧
    S₁ = (135 / 360) * π * r^2 ∧
    S₂ = S₁ + π * ((3/8) * r)^2) →
  S₁ / S₂ = 8 / 11 := by
sorry


end NUMINAMATH_CALUDE_sector_to_cone_area_ratio_l42_4233


namespace NUMINAMATH_CALUDE_divisibility_problem_l42_4246

theorem divisibility_problem (n m k : ℕ) (h1 : n = 425897) (h2 : m = 456) (h3 : k = 247) :
  (n + k) % m = 0 :=
by sorry

end NUMINAMATH_CALUDE_divisibility_problem_l42_4246


namespace NUMINAMATH_CALUDE_absolute_value_plus_exponent_l42_4230

theorem absolute_value_plus_exponent : |(-8 : ℤ)| + 3^(0 : ℕ) = 9 := by sorry

end NUMINAMATH_CALUDE_absolute_value_plus_exponent_l42_4230


namespace NUMINAMATH_CALUDE_banquet_arrangement_theorem_l42_4240

/-- Represents the problem of arranging tables for a banquet. -/
structure BanquetArrangement where
  guests : Nat
  initial_tables : Nat
  seats_per_table : Nat

/-- Calculates the minimum number of tables needed for the banquet arrangement. -/
def min_tables_needed (arrangement : BanquetArrangement) : Nat :=
  sorry

/-- Theorem stating that for the given banquet arrangement, 11 tables are needed. -/
theorem banquet_arrangement_theorem :
  let arrangement : BanquetArrangement := {
    guests := 44,
    initial_tables := 15,
    seats_per_table := 4
  }
  min_tables_needed arrangement = 11 := by sorry

end NUMINAMATH_CALUDE_banquet_arrangement_theorem_l42_4240


namespace NUMINAMATH_CALUDE_sine_cosine_ratio_simplification_l42_4253

theorem sine_cosine_ratio_simplification :
  (Real.sin (30 * π / 180) + Real.sin (60 * π / 180)) /
  (Real.cos (30 * π / 180) + Real.cos (60 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_ratio_simplification_l42_4253


namespace NUMINAMATH_CALUDE_mans_speed_against_current_l42_4220

/-- Given a man's speed with the current and the speed of the current, 
    calculate the man's speed against the current. -/
def speed_against_current (speed_with_current : ℝ) (current_speed : ℝ) : ℝ :=
  speed_with_current - 2 * current_speed

/-- Theorem stating that given the specific speeds in the problem, 
    the man's speed against the current is 12 km/hr. -/
theorem mans_speed_against_current :
  speed_against_current 22 5 = 12 := by
  sorry

#eval speed_against_current 22 5

end NUMINAMATH_CALUDE_mans_speed_against_current_l42_4220


namespace NUMINAMATH_CALUDE_complex_equation_solution_l42_4227

theorem complex_equation_solution : ∃ (a b : ℝ) (z : ℂ), 
  a > 0 ∧ b > 0 ∧ 
  z = a + b * I ∧
  z * (z + I) * (z + 3 * I) * (z - 2) = 180 * I ∧
  a = Real.sqrt 180 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l42_4227


namespace NUMINAMATH_CALUDE_existence_of_m_n_l42_4270

theorem existence_of_m_n (p : ℕ) (hp : p.Prime) (hp_gt_10 : p > 10) :
  ∃ m n : ℕ, m > 0 ∧ n > 0 ∧ m + n < p ∧ (5^m * 7^n - 1) % p = 0 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_m_n_l42_4270


namespace NUMINAMATH_CALUDE_game_terminates_l42_4239

/-- Represents the state of knowledge for a player -/
structure PlayerKnowledge where
  lower : ℕ
  upper : ℕ

/-- Represents the game state -/
structure GameState where
  r₁ : ℕ
  r₂ : ℕ
  a_knowledge : PlayerKnowledge
  b_knowledge : PlayerKnowledge

/-- Updates a player's knowledge based on the game state -/
def update_knowledge (state : GameState) (is_player_a : Bool) : PlayerKnowledge :=
  sorry

/-- Checks if a player can determine the other's number -/
def can_determine (knowledge : PlayerKnowledge) : Bool :=
  sorry

/-- The main theorem stating that the game terminates -/
theorem game_terminates (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  ∃ (n : ℕ), ∃ (final_state : GameState),
    (final_state.r₁ = a + b ∨ final_state.r₂ = a + b) ∧
    (can_determine final_state.a_knowledge ∨ can_determine final_state.b_knowledge) :=
  sorry

end NUMINAMATH_CALUDE_game_terminates_l42_4239


namespace NUMINAMATH_CALUDE_triangle_inequality_l42_4286

theorem triangle_inequality (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  (1/2) * c^2 = (1/2) * a * b * Real.sin C →
  a * b = Real.sqrt 2 →
  a^2 + b^2 + c^2 ≤ 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l42_4286


namespace NUMINAMATH_CALUDE_product_digit_sum_equals_999_l42_4290

/-- The number of digits in the second factor -/
def n : ℕ := 111

/-- The second factor in the product -/
def second_factor (n : ℕ) : ℕ := (10^n - 1) / 9

/-- The product of 9 and the second factor -/
def product (n : ℕ) : ℕ := 9 * second_factor n

/-- Sum of digits function -/
def sum_of_digits (x : ℕ) : ℕ := sorry

theorem product_digit_sum_equals_999 :
  sum_of_digits (product n) = 999 :=
sorry

end NUMINAMATH_CALUDE_product_digit_sum_equals_999_l42_4290


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_l42_4264

-- Define the types for plane and line
variable (Plane : Type) (Line : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem line_perpendicular_to_plane 
  (α : Plane) (a b : Line) (ha : a ≠ b) :
  perpendicular a α → parallel a b → perpendicular b α :=
by sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_l42_4264


namespace NUMINAMATH_CALUDE_total_games_attended_l42_4217

def games_this_year : ℕ := 15
def games_last_year : ℕ := 39

theorem total_games_attended : games_this_year + games_last_year = 54 := by
  sorry

end NUMINAMATH_CALUDE_total_games_attended_l42_4217


namespace NUMINAMATH_CALUDE_problem_statement_l42_4276

theorem problem_statement (a b : ℝ) 
  (h1 : a - b = 1) 
  (h2 : a^2 - b^2 = -1) : 
  a^4 - b^4 = -1 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l42_4276


namespace NUMINAMATH_CALUDE_handshaking_theorem_l42_4242

/-- Represents a handshaking arrangement for 11 people -/
def HandshakingArrangement := Fin 11 → Finset (Fin 11)

/-- The number of people in the group -/
def group_size : Nat := 11

/-- The number of handshakes per person -/
def handshakes_per_person : Nat := 3

/-- Predicate for a valid handshaking arrangement -/
def is_valid_arrangement (a : HandshakingArrangement) : Prop :=
  ∀ i : Fin group_size, (a i).card = handshakes_per_person ∧ i ∉ a i

/-- The number of valid handshaking arrangements -/
def num_arrangements : Nat := 1814400

/-- The theorem to be proved -/
theorem handshaking_theorem :
  (∃ (S : Finset HandshakingArrangement),
    (∀ a ∈ S, is_valid_arrangement a) ∧
    S.card = num_arrangements) ∧
  num_arrangements % 1000 = 400 := by sorry

end NUMINAMATH_CALUDE_handshaking_theorem_l42_4242


namespace NUMINAMATH_CALUDE_first_player_strategy_guarantees_six_no_root_equations_l42_4262

/-- Represents a quadratic equation ax² + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  b_nonzero : b ≠ 0
  c_nonzero : c ≠ 0

/-- Represents the game state -/
structure GameState where
  equations : List QuadraticEquation
  num_equations : Nat
  num_equations_eq_11 : num_equations = 11

/-- Represents a player's strategy -/
def Strategy := GameState → QuadraticEquation

/-- Determines if a quadratic equation has no real roots -/
def has_no_real_roots (eq : QuadraticEquation) : Prop :=
  eq.b * eq.b - 4 * eq.a * eq.c < 0

/-- The maximum number of equations without real roots that the first player can guarantee -/
def max_no_root_equations : Nat := 6

/-- The main theorem to be proved -/
theorem first_player_strategy_guarantees_six_no_root_equations
  (initial_state : GameState)
  (first_player_strategy : Strategy)
  (second_player_strategy : Strategy) :
  ∃ (final_state : GameState),
    (final_state.num_equations = initial_state.num_equations) ∧
    (∀ eq ∈ final_state.equations, has_no_real_roots eq) ∧
    (final_state.equations.length ≥ max_no_root_equations) :=
  sorry

#check first_player_strategy_guarantees_six_no_root_equations

end NUMINAMATH_CALUDE_first_player_strategy_guarantees_six_no_root_equations_l42_4262


namespace NUMINAMATH_CALUDE_max_profit_week3_l42_4289

/-- Represents the sales and profit data for a bicycle store. -/
structure BicycleStore where
  profit_a : ℕ  -- Profit per type A bicycle
  profit_b : ℕ  -- Profit per type B bicycle
  week1_a : ℕ   -- Week 1 sales of type A
  week1_b : ℕ   -- Week 1 sales of type B
  week1_profit : ℕ  -- Week 1 total profit
  week2_a : ℕ   -- Week 2 sales of type A
  week2_b : ℕ   -- Week 2 sales of type B
  week2_profit : ℕ  -- Week 2 total profit
  week3_total : ℕ  -- Week 3 total sales

/-- Determines if the given sales distribution for Week 3 is valid. -/
def validWeek3Sales (store : BicycleStore) (a : ℕ) (b : ℕ) : Prop :=
  a + b = store.week3_total ∧ b > a ∧ b ≤ 2 * a

/-- Calculates the profit for a given sales distribution. -/
def calculateProfit (store : BicycleStore) (a : ℕ) (b : ℕ) : ℕ :=
  a * store.profit_a + b * store.profit_b

/-- Theorem stating that the maximum profit in Week 3 is achieved by selling 9 type A and 16 type B bicycles. -/
theorem max_profit_week3 (store : BicycleStore) 
    (h1 : store.week1_a * store.profit_a + store.week1_b * store.profit_b = store.week1_profit)
    (h2 : store.week2_a * store.profit_a + store.week2_b * store.profit_b = store.week2_profit)
    (h3 : store.week3_total = 25)
    (h4 : store.profit_a = 80)
    (h5 : store.profit_b = 100) :
    (∀ a b, validWeek3Sales store a b → calculateProfit store a b ≤ 2320) ∧
    validWeek3Sales store 9 16 ∧
    calculateProfit store 9 16 = 2320 := by
  sorry


end NUMINAMATH_CALUDE_max_profit_week3_l42_4289


namespace NUMINAMATH_CALUDE_reciprocal_equal_self_is_set_l42_4247

def reciprocal_equal_self (x : ℝ) : Prop := x ≠ 0 ∧ 1 / x = x

def reciprocal_equal_self_set : Set ℝ := {x : ℝ | reciprocal_equal_self x}

theorem reciprocal_equal_self_is_set : 
  ∃ (S : Set ℝ), ∀ x : ℝ, x ∈ S ↔ reciprocal_equal_self x :=
sorry

end NUMINAMATH_CALUDE_reciprocal_equal_self_is_set_l42_4247


namespace NUMINAMATH_CALUDE_student_arrangement_count_l42_4248

def num_students : Nat := 6

def leftmost_students : Finset Char := {'A', 'B'}

def all_students : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}

theorem student_arrangement_count :
  (leftmost_students.card * (num_students - 1).factorial) +
  ((all_students.card - 2) * (num_students - 2).factorial) = 216 := by
  sorry

end NUMINAMATH_CALUDE_student_arrangement_count_l42_4248


namespace NUMINAMATH_CALUDE_y_derivative_f_monotonicity_l42_4226

-- Part 1
noncomputable def y (x : ℝ) : ℝ := (2 * x^2 - 3) * Real.sqrt (1 + x^2)

theorem y_derivative (x : ℝ) :
  deriv y x = 4 * x * Real.sqrt (1 + x^2) + (2 * x^3 - 3 * x) / Real.sqrt (1 + x^2) :=
sorry

-- Part 2
noncomputable def f (x : ℝ) : ℝ := (x * Real.log x)⁻¹

theorem f_monotonicity (x : ℝ) (hx : x > 0 ∧ x ≠ 1) :
  (StrictMonoOn f (Set.Ioo 0 (Real.exp (-1)))) ∧
  (StrictAntiOn f (Set.Ioi (Real.exp (-1)))) :=
sorry

end NUMINAMATH_CALUDE_y_derivative_f_monotonicity_l42_4226


namespace NUMINAMATH_CALUDE_constant_expression_l42_4251

theorem constant_expression (x : ℝ) (h : x ≥ 4/7) :
  -4*x + |4 - 7*x| - |1 - 3*x| + 4 = 1 := by
sorry

end NUMINAMATH_CALUDE_constant_expression_l42_4251


namespace NUMINAMATH_CALUDE_city_distance_proof_l42_4268

def is_valid_gcd (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 13

theorem city_distance_proof :
  ∃ (S : ℕ), S > 0 ∧
  (∀ (x : ℕ), x ≤ S → is_valid_gcd (Nat.gcd x (S - x))) ∧
  (∀ (T : ℕ), T > 0 →
    (∀ (y : ℕ), y ≤ T → is_valid_gcd (Nat.gcd y (T - y))) →
    S ≤ T) ∧
  S = 39 :=
sorry

end NUMINAMATH_CALUDE_city_distance_proof_l42_4268


namespace NUMINAMATH_CALUDE_equation_has_real_roots_l42_4271

theorem equation_has_real_roots (K : ℝ) : ∃ x : ℝ, x = K^2 * (x - 1) * (x - 3) :=
sorry

end NUMINAMATH_CALUDE_equation_has_real_roots_l42_4271


namespace NUMINAMATH_CALUDE_club_ranking_l42_4285

def Chess : ℚ := 9/28
def Drama : ℚ := 11/28
def Art : ℚ := 1/7
def Science : ℚ := 5/14

theorem club_ranking :
  Drama > Science ∧ Science > Chess ∧ Chess > Art := by
  sorry

end NUMINAMATH_CALUDE_club_ranking_l42_4285


namespace NUMINAMATH_CALUDE_new_shoe_cost_calculation_l42_4213

/-- The cost of repairing used shoes -/
def repair_cost : ℝ := 11.50

/-- The duration that repaired shoes last (in years) -/
def repaired_duration : ℝ := 1

/-- The duration that new shoes last (in years) -/
def new_duration : ℝ := 2

/-- The percentage increase in average yearly cost of new shoes compared to repaired shoes -/
def cost_increase_percentage : ℝ := 0.2173913043478261

/-- The cost of purchasing new shoes -/
def new_shoe_cost : ℝ := 2 * (repair_cost + cost_increase_percentage * repair_cost)

theorem new_shoe_cost_calculation :
  new_shoe_cost = 28 :=
sorry

end NUMINAMATH_CALUDE_new_shoe_cost_calculation_l42_4213


namespace NUMINAMATH_CALUDE_balloons_left_l42_4291

theorem balloons_left (round_bags : ℕ) (round_per_bag : ℕ) (long_bags : ℕ) (long_per_bag : ℕ) (burst : ℕ) : 
  round_bags = 5 → 
  round_per_bag = 20 → 
  long_bags = 4 → 
  long_per_bag = 30 → 
  burst = 5 → 
  round_bags * round_per_bag + long_bags * long_per_bag - burst = 215 := by
sorry

end NUMINAMATH_CALUDE_balloons_left_l42_4291


namespace NUMINAMATH_CALUDE_multiply_by_twelve_l42_4202

theorem multiply_by_twelve (x : ℝ) : x / 14 = 42 → 12 * x = 7056 := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_twelve_l42_4202


namespace NUMINAMATH_CALUDE_line_y_coordinate_l42_4203

/-- Given a line that passes through points (-6, y1) and (x2, 3), 
    with an x-intercept at (4, 0), prove that y1 = 7.5 -/
theorem line_y_coordinate (y1 x2 : ℝ) : 
  (∃ m b : ℝ, 
    (y1 = m * (-6) + b) ∧ 
    (3 = m * x2 + b) ∧ 
    (0 = m * 4 + b)) →
  y1 = 7.5 := by
sorry

end NUMINAMATH_CALUDE_line_y_coordinate_l42_4203


namespace NUMINAMATH_CALUDE_new_room_area_l42_4216

/-- Given a bedroom and bathroom area, calculate the area of a new room that is twice as large as their combined area -/
theorem new_room_area (bedroom_area bathroom_area : ℕ) : 
  bedroom_area = 309 → bathroom_area = 150 → 
  2 * (bedroom_area + bathroom_area) = 918 := by
  sorry

end NUMINAMATH_CALUDE_new_room_area_l42_4216


namespace NUMINAMATH_CALUDE_symmetry_coordinates_l42_4204

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the origin -/
def symmetricToOrigin (p q : Point) : Prop :=
  q.x = -p.x ∧ q.y = -p.y

theorem symmetry_coordinates :
  let A : Point := ⟨1, -2⟩
  let A' : Point := ⟨-1, 2⟩
  symmetricToOrigin A A' :=
by
  sorry

end NUMINAMATH_CALUDE_symmetry_coordinates_l42_4204


namespace NUMINAMATH_CALUDE_power_difference_equality_l42_4225

theorem power_difference_equality : 5^(7+2) - 2^(5+3) = 1952869 := by sorry

end NUMINAMATH_CALUDE_power_difference_equality_l42_4225


namespace NUMINAMATH_CALUDE_problem_solution_l42_4252

theorem problem_solution (x y : ℝ) (h1 : 1.5 * x = 0.3 * y) (h2 : x = 20) : y = 100 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l42_4252


namespace NUMINAMATH_CALUDE_total_amount_collected_l42_4254

/-- Represents the ratio of passengers in I and II class -/
def passenger_ratio : ℚ := 1 / 50

/-- Represents the ratio of fares for I and II class -/
def fare_ratio : ℚ := 3 / 1

/-- Amount collected from II class passengers in rupees -/
def amount_II : ℕ := 1250

/-- Theorem stating the total amount collected from all passengers -/
theorem total_amount_collected :
  let amount_I := (amount_II : ℚ) * passenger_ratio * fare_ratio
  (amount_I + amount_II : ℚ) = 1325 := by sorry

end NUMINAMATH_CALUDE_total_amount_collected_l42_4254


namespace NUMINAMATH_CALUDE_roller_derby_laces_l42_4229

theorem roller_derby_laces (num_teams : ℕ) (members_per_team : ℕ) (skates_per_member : ℕ) (total_laces : ℕ) :
  num_teams = 4 →
  members_per_team = 10 →
  skates_per_member = 2 →
  total_laces = 240 →
  total_laces / (num_teams * members_per_team * skates_per_member) = 3 :=
by sorry

end NUMINAMATH_CALUDE_roller_derby_laces_l42_4229


namespace NUMINAMATH_CALUDE_solve_for_x_l42_4208

theorem solve_for_x (y : ℚ) (h1 : y = -3/2) (h2 : -2*x - y^2 = 1/4) : x = -5/4 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l42_4208


namespace NUMINAMATH_CALUDE_banana_cost_is_three_l42_4232

/-- The cost of a bunch of bananas -/
def banana_cost : ℝ := 3

/-- The cost of a dozen apples -/
def apple_dozen_cost : ℝ := 2

/-- Arnold's purchase: 1 dozen apples and 1 bunch of bananas -/
def arnold_purchase : ℝ := apple_dozen_cost + banana_cost

/-- Tony's purchase: 2 dozen apples and 1 bunch of bananas -/
def tony_purchase : ℝ := 2 * apple_dozen_cost + banana_cost

theorem banana_cost_is_three :
  arnold_purchase = 5 ∧ tony_purchase = 7 → banana_cost = 3 := by
  sorry

end NUMINAMATH_CALUDE_banana_cost_is_three_l42_4232
