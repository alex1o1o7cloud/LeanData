import Mathlib

namespace NUMINAMATH_CALUDE_vasya_mistake_l2086_208669

-- Define the function to calculate the number of digits used for page numbering
def digits_used (n : ℕ) : ℕ :=
  if n < 10 then n
  else if n < 100 then 9 + 2 * (n - 9)
  else 189 + 3 * (n - 99)

-- Theorem statement
theorem vasya_mistake :
  ¬ ∃ (n : ℕ), digits_used n = 301 :=
sorry

end NUMINAMATH_CALUDE_vasya_mistake_l2086_208669


namespace NUMINAMATH_CALUDE_distance_circle_center_to_point_l2086_208657

/-- The distance between the center of a circle with polar equation ρ = 4sin θ 
    and a point with polar coordinates (2√2, π/4) is 2 -/
theorem distance_circle_center_to_point : 
  let circle_equation : ℝ → ℝ := λ θ => 4 * Real.sin θ
  let point_A : ℝ × ℝ := (2 * Real.sqrt 2, Real.pi / 4)
  ∃ center : ℝ × ℝ, 
    Real.sqrt ((center.1 - (point_A.1 * Real.cos point_A.2))^2 + 
               (center.2 - (point_A.1 * Real.sin point_A.2))^2) = 2 :=
by sorry

end NUMINAMATH_CALUDE_distance_circle_center_to_point_l2086_208657


namespace NUMINAMATH_CALUDE_diagonals_150_sided_polygon_l2086_208601

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: The number of diagonals in a polygon with 150 sides is 11025 -/
theorem diagonals_150_sided_polygon : num_diagonals 150 = 11025 := by
  sorry

end NUMINAMATH_CALUDE_diagonals_150_sided_polygon_l2086_208601


namespace NUMINAMATH_CALUDE_stating_arithmetic_sequence_iff_60_degree_l2086_208697

/-- A triangle with interior angles A, B, and C. -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_180 : A + B + C = 180
  positive : A > 0 ∧ B > 0 ∧ C > 0

/-- The interior angles of a triangle form an arithmetic sequence. -/
def arithmetic_sequence (t : Triangle) : Prop :=
  t.A + t.C = 2 * t.B ∨ t.A + t.B = 2 * t.C ∨ t.B + t.C = 2 * t.A

/-- One of the interior angles of a triangle is 60 degrees. -/
def has_60_degree (t : Triangle) : Prop :=
  t.A = 60 ∨ t.B = 60 ∨ t.C = 60

/-- 
Theorem stating that a triangle's interior angles form an arithmetic sequence 
if and only if one of its interior angles is 60 degrees.
-/
theorem arithmetic_sequence_iff_60_degree (t : Triangle) :
  arithmetic_sequence t ↔ has_60_degree t :=
sorry

end NUMINAMATH_CALUDE_stating_arithmetic_sequence_iff_60_degree_l2086_208697


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l2086_208603

theorem polynomial_remainder_theorem (x : ℝ) : 
  let f : ℝ → ℝ := λ x => 8*x^4 - 18*x^3 + 5*x^2 - 3*x - 30
  let g : ℝ → ℝ := λ x => 2*x - 4
  f 2 = -32 ∧ (∃ q : ℝ → ℝ, ∀ x, f x = g x * q x + f 2) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l2086_208603


namespace NUMINAMATH_CALUDE_certain_expression_l2086_208600

theorem certain_expression (a b X : ℝ) 
  (h1 : a / b = 4 / 3) 
  (h2 : (3 * a + 2 * b) / X = 3) : 
  X = 2 * b := by
sorry

end NUMINAMATH_CALUDE_certain_expression_l2086_208600


namespace NUMINAMATH_CALUDE_opposite_of_one_minus_cube_root_three_l2086_208602

theorem opposite_of_one_minus_cube_root_three :
  -(1 - Real.rpow 3 (1/3)) = Real.rpow 3 (1/3) - 1 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_one_minus_cube_root_three_l2086_208602


namespace NUMINAMATH_CALUDE_propositions_truth_l2086_208628

-- Definition of correlation coefficient
def correlation_strength (r : ℝ) : ℝ := 1 - |r|

-- Definition of perpendicular lines
def perpendicular (A B C A' B' C' : ℝ) : Prop := A * A' + B * B' = 0

theorem propositions_truth : 
  -- Proposition 1 (false)
  (∃ x : ℝ, x^2 < 0 ↔ ¬ ∀ x : ℝ, x^2 ≥ 0) ∧
  -- Proposition 2 (true)
  (∀ r : ℝ, |r| ≤ 1 → correlation_strength r ≤ correlation_strength 0) ∧
  -- Proposition 3 (false, not included)
  -- Proposition 4 (true)
  perpendicular 2 10 6 3 (-3/5) (13/5) :=
by sorry

end NUMINAMATH_CALUDE_propositions_truth_l2086_208628


namespace NUMINAMATH_CALUDE_garrison_problem_l2086_208673

/-- Represents the number of men in a garrison and their provisions --/
structure Garrison where
  initialMen : ℕ
  initialDays : ℕ
  reinforcementMen : ℕ
  remainingDays : ℕ
  reinforcementArrivalDay : ℕ

/-- Calculates the initial number of men in the garrison --/
def calculateInitialMen (g : Garrison) : ℕ :=
  (g.initialDays - g.reinforcementArrivalDay) * g.initialMen / 
  (g.initialDays - g.reinforcementArrivalDay - g.remainingDays)

/-- Theorem stating that given the conditions, the initial number of men is 2000 --/
theorem garrison_problem (g : Garrison) 
  (h1 : g.initialDays = 65)
  (h2 : g.reinforcementMen = 3000)
  (h3 : g.remainingDays = 20)
  (h4 : g.reinforcementArrivalDay = 15) :
  calculateInitialMen g = 2000 := by
  sorry

#eval calculateInitialMen { initialMen := 2000, initialDays := 65, reinforcementMen := 3000, remainingDays := 20, reinforcementArrivalDay := 15 }

end NUMINAMATH_CALUDE_garrison_problem_l2086_208673


namespace NUMINAMATH_CALUDE_tangent_line_properties_l2086_208652

def curve_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 1 = 0

def line_l (α : ℝ) (x y : ℝ) : Prop := 
  ∃ t : ℝ, x = 4 + t * Real.sin α ∧ y = t * Real.cos α

def is_tangent (α : ℝ) : Prop :=
  ∃ x y : ℝ, curve_C x y ∧ line_l α x y ∧
  ∀ x' y' : ℝ, curve_C x' y' ∧ line_l α x' y' → (x', y') = (x, y)

theorem tangent_line_properties :
  ∀ α : ℝ, 0 ≤ α ∧ α < Real.pi → is_tangent α →
    α = Real.pi / 6 ∧
    ∃ x y : ℝ, curve_C x y ∧ line_l α x y ∧ x = 7/2 ∧ y = -Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_properties_l2086_208652


namespace NUMINAMATH_CALUDE_unique_solution_to_equation_l2086_208636

theorem unique_solution_to_equation : ∃! (x : ℝ), x ≠ 0 ∧ (7 * x)^5 = (14 * x)^4 ∧ x = 16/7 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_to_equation_l2086_208636


namespace NUMINAMATH_CALUDE_parallel_line_slope_l2086_208667

/-- The slope of a line parallel to 3x + 6y = 15 is -1/2 -/
theorem parallel_line_slope :
  ∀ (m : ℚ), (∃ (b : ℚ), ∀ (x y : ℚ), y = m * x + b ↔ 3 * x + 6 * y = 15) →
  m = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_slope_l2086_208667


namespace NUMINAMATH_CALUDE_polynomial_division_quotient_l2086_208685

theorem polynomial_division_quotient : ∀ x : ℝ,
  (9 * x^3 - 5 * x^2 + 8 * x - 12) = (x - 3) * (9 * x^2 + 22 * x + 74) + 210 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_quotient_l2086_208685


namespace NUMINAMATH_CALUDE_opposite_number_on_line_l2086_208665

theorem opposite_number_on_line (a : ℝ) : (a + (a - 6) = 0) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_number_on_line_l2086_208665


namespace NUMINAMATH_CALUDE_cube_cutting_problem_l2086_208630

theorem cube_cutting_problem :
  ∃! n : ℕ, ∃ s : ℕ, n > s ∧ n^3 - s^3 = 152 :=
by sorry

end NUMINAMATH_CALUDE_cube_cutting_problem_l2086_208630


namespace NUMINAMATH_CALUDE_max_initial_states_l2086_208604

/-- Represents the state of friendships between sheep and the wolf -/
structure SheepcoteState (n : ℕ) :=
  (wolf_friends : Finset (Fin n))
  (sheep_friendships : Finset (Fin n × Fin n))

/-- Represents the process of the wolf eating sheep -/
def eat_sheep (state : SheepcoteState n) : Option (SheepcoteState n) :=
  sorry

/-- Checks if all sheep can be eaten given an initial state -/
def can_eat_all_sheep (initial_state : SheepcoteState n) : Prop :=
  sorry

/-- The number of valid initial states -/
def num_valid_initial_states (n : ℕ) : ℕ :=
  sorry

theorem max_initial_states (n : ℕ) :
  num_valid_initial_states n = 2^(n-1) :=
sorry

end NUMINAMATH_CALUDE_max_initial_states_l2086_208604


namespace NUMINAMATH_CALUDE_book_distribution_ways_l2086_208620

/-- The number of ways to distribute identical books between two states --/
def distribute_books (n : ℕ) : ℕ := n - 1

/-- The number of books --/
def total_books : ℕ := 8

/-- Theorem: The number of ways to distribute eight identical books between
    the library and being checked out, with at least one book in each state,
    is equal to 7. --/
theorem book_distribution_ways :
  distribute_books total_books = 7 := by
  sorry

end NUMINAMATH_CALUDE_book_distribution_ways_l2086_208620


namespace NUMINAMATH_CALUDE_scientific_notation_748_million_l2086_208647

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ

/-- Converts a real number to scientific notation with given significant figures -/
def toScientificNotation (x : ℝ) (sigFigs : ℕ) : ScientificNotation :=
  sorry

/-- Rounds a real number to a given number of significant figures -/
def roundToSigFigs (x : ℝ) (sigFigs : ℕ) : ℝ :=
  sorry

theorem scientific_notation_748_million :
  let original := (748 : ℝ) * 1000000
  let scientificForm := toScientificNotation original 2
  scientificForm = ScientificNotation.mk 7.5 8 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_748_million_l2086_208647


namespace NUMINAMATH_CALUDE_strings_per_normal_guitar_is_6_l2086_208660

/-- Calculates the number of strings on each normal guitar given the following conditions:
  * There are 3 basses with 4 strings each
  * There are twice as many normal guitars as basses
  * There are 3 fewer 8-string guitars than normal guitars
  * The total number of strings needed is 72
-/
def strings_per_normal_guitar : ℕ :=
  let num_basses : ℕ := 3
  let strings_per_bass : ℕ := 4
  let num_normal_guitars : ℕ := 2 * num_basses
  let num_8string_guitars : ℕ := num_normal_guitars - 3
  let total_strings : ℕ := 72
  (total_strings - num_basses * strings_per_bass - num_8string_guitars * 8) / num_normal_guitars

theorem strings_per_normal_guitar_is_6 : strings_per_normal_guitar = 6 := by
  sorry

end NUMINAMATH_CALUDE_strings_per_normal_guitar_is_6_l2086_208660


namespace NUMINAMATH_CALUDE_mencius_reading_problem_l2086_208689

theorem mencius_reading_problem (total_chars : ℕ) (days : ℕ) (first_day_chars : ℕ) : 
  total_chars = 34685 →
  days = 3 →
  first_day_chars + 2 * first_day_chars + 4 * first_day_chars = total_chars →
  first_day_chars = 4955 := by
sorry

end NUMINAMATH_CALUDE_mencius_reading_problem_l2086_208689


namespace NUMINAMATH_CALUDE_extended_quadrilateral_area_l2086_208687

-- Define the quadrilateral ABCD
structure Quadrilateral (V : Type*) [AddCommGroup V] [Module ℝ V] :=
  (A B C D : V)

-- Define the extended quadrilateral A₁B₁C₁D₁
structure ExtendedQuadrilateral (V : Type*) [AddCommGroup V] [Module ℝ V] extends Quadrilateral V :=
  (A₁ B₁ C₁ D₁ : V)
  (hDA₁ : A₁ - D = 2 • (A - D))
  (hAB₁ : B₁ - A = 2 • (B - A))
  (hBC₁ : C₁ - B = 2 • (C - B))
  (hCD₁ : D₁ - C = 2 • (D - C))

-- Define the area function
noncomputable def area {V : Type*} [AddCommGroup V] [Module ℝ V] (q : Quadrilateral V) : ℝ := sorry

-- State the theorem
theorem extended_quadrilateral_area {V : Type*} [AddCommGroup V] [Module ℝ V] 
  (q : ExtendedQuadrilateral V) :
  area {A := q.A₁, B := q.B₁, C := q.C₁, D := q.D₁} = 5 * area {A := q.A, B := q.B, C := q.C, D := q.D} :=
sorry

end NUMINAMATH_CALUDE_extended_quadrilateral_area_l2086_208687


namespace NUMINAMATH_CALUDE_infinite_sum_evaluation_l2086_208698

theorem infinite_sum_evaluation :
  (∑' n : ℕ, (3 : ℝ)^n / (1 + 3^n + 3^(n + 1) + 3^(2*n + 1))) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_infinite_sum_evaluation_l2086_208698


namespace NUMINAMATH_CALUDE_negation_of_existential_proposition_l2086_208676

theorem negation_of_existential_proposition :
  (¬∃ x₀ : ℝ, x₀ ∈ Set.Icc (-3) 3 ∧ x₀^2 + 2*x₀ + 1 ≤ 0) ↔
  (∀ x₀ : ℝ, x₀ ∈ Set.Icc (-3) 3 → x₀^2 + 2*x₀ + 1 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existential_proposition_l2086_208676


namespace NUMINAMATH_CALUDE_sector_radius_l2086_208624

/-- Given a circular sector with area 10 cm² and arc length 4 cm, prove that the radius is 5 cm -/
theorem sector_radius (area : ℝ) (arc_length : ℝ) (radius : ℝ) 
  (h_area : area = 10) 
  (h_arc : arc_length = 4) 
  (h_sector : area = (arc_length * radius) / 2) : radius = 5 := by
  sorry

end NUMINAMATH_CALUDE_sector_radius_l2086_208624


namespace NUMINAMATH_CALUDE_AB_value_l2086_208627

-- Define the points
variable (A B C D E : ℝ × ℝ)

-- Define the conditions
axiom on_line : ∃ (y : ℝ), A.2 = y ∧ B.2 = y ∧ C.2 = y ∧ D.2 = y
axiom order : A.1 < B.1 ∧ B.1 < C.1 ∧ C.1 < D.1
axiom AB_eq_CD : dist A B = dist C D
axiom BC_eq_16 : dist B C = 16
axiom E_not_on_line : E.2 ≠ A.2
axiom BE_eq_CE : dist B E = dist C E
axiom BE_eq_13 : dist B E = 13

-- Define perimeter function
def perimeter (X Y Z : ℝ × ℝ) : ℝ := dist X Y + dist Y Z + dist Z X

-- State the theorem
theorem AB_value : 
  perimeter A E D = 3 * perimeter B E C → 
  dist A B = 34/3 :=
sorry

end NUMINAMATH_CALUDE_AB_value_l2086_208627


namespace NUMINAMATH_CALUDE_two_digit_powers_of_three_l2086_208663

theorem two_digit_powers_of_three : 
  ∃! (count : ℕ), ∃ (S : Finset ℕ), 
    (∀ n ∈ S, 10 ≤ 3^n ∧ 3^n ≤ 99) ∧ 
    (∀ n ∉ S, 3^n < 10 ∨ 99 < 3^n) ∧ 
    Finset.card S = count ∧
    count = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_powers_of_three_l2086_208663


namespace NUMINAMATH_CALUDE_zero_point_in_interval_l2086_208637

noncomputable def f (x : ℝ) : ℝ := x^3 - (1/2)^(x-2)

theorem zero_point_in_interval :
  ∃ x₀ : ℝ, f x₀ = 0 ∧ 1 < x₀ ∧ x₀ < 2 :=
sorry

end NUMINAMATH_CALUDE_zero_point_in_interval_l2086_208637


namespace NUMINAMATH_CALUDE_integral_evaluation_l2086_208682

theorem integral_evaluation : ∫ (x : ℝ) in (0)..(1), (8 / Real.pi) * Real.sqrt (1 - x^2) + 6 * x^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_integral_evaluation_l2086_208682


namespace NUMINAMATH_CALUDE_linear_function_composition_l2086_208616

theorem linear_function_composition (a b : ℝ) :
  (∀ x y : ℝ, x < y → (a * x + b) < (a * y + b)) →
  (∀ x : ℝ, a * (a * x + b) + b = 4 * x - 1) →
  a = -2 ∧ b = 1 := by
sorry

end NUMINAMATH_CALUDE_linear_function_composition_l2086_208616


namespace NUMINAMATH_CALUDE_apples_difference_l2086_208640

/-- The number of apples Yanna bought -/
def total_apples : ℕ := 60

/-- The number of apples Yanna gave to Zenny -/
def apples_to_zenny : ℕ := 18

/-- The number of apples Yanna kept for herself -/
def apples_kept : ℕ := 36

/-- The number of apples Yanna gave to Andrea -/
def apples_to_andrea : ℕ := total_apples - apples_to_zenny - apples_kept

theorem apples_difference :
  apples_to_zenny - apples_to_andrea = 12 :=
by sorry

end NUMINAMATH_CALUDE_apples_difference_l2086_208640


namespace NUMINAMATH_CALUDE_bakery_profit_is_175_l2086_208622

/-- Calculates the total profit for Uki's bakery over five days -/
def bakery_profit : ℝ :=
  let cupcake_price : ℝ := 1.50
  let cookie_price : ℝ := 2.00
  let biscuit_price : ℝ := 1.00
  let cupcake_cost : ℝ := 0.75
  let cookie_cost : ℝ := 1.00
  let biscuit_cost : ℝ := 0.50
  let daily_cupcakes : ℝ := 20
  let daily_cookies : ℝ := 10
  let daily_biscuits : ℝ := 20
  let days : ℝ := 5
  let daily_profit : ℝ := 
    (cupcake_price - cupcake_cost) * daily_cupcakes +
    (cookie_price - cookie_cost) * daily_cookies +
    (biscuit_price - biscuit_cost) * daily_biscuits
  daily_profit * days

theorem bakery_profit_is_175 : bakery_profit = 175 := by
  sorry

end NUMINAMATH_CALUDE_bakery_profit_is_175_l2086_208622


namespace NUMINAMATH_CALUDE_sarah_christmas_shopping_l2086_208693

/-- The amount of money Sarah started with for Christmas shopping. -/
def initial_amount : ℕ := 100

/-- The cost of each toy car. -/
def toy_car_cost : ℕ := 11

/-- The number of toy cars Sarah bought. -/
def num_toy_cars : ℕ := 2

/-- The cost of the scarf. -/
def scarf_cost : ℕ := 10

/-- The cost of the beanie. -/
def beanie_cost : ℕ := 14

/-- The cost of the necklace. -/
def necklace_cost : ℕ := 20

/-- The cost of the gloves. -/
def gloves_cost : ℕ := 12

/-- The cost of the book. -/
def book_cost : ℕ := 15

/-- The amount of money Sarah has remaining after purchasing all gifts. -/
def remaining_amount : ℕ := 7

/-- Theorem stating that the initial amount is equal to the sum of all gift costs plus the remaining amount. -/
theorem sarah_christmas_shopping :
  initial_amount = 
    num_toy_cars * toy_car_cost + 
    scarf_cost + 
    beanie_cost + 
    necklace_cost + 
    gloves_cost + 
    book_cost + 
    remaining_amount :=
by sorry

end NUMINAMATH_CALUDE_sarah_christmas_shopping_l2086_208693


namespace NUMINAMATH_CALUDE_max_length_sum_l2086_208653

/-- Length of an integer is the number of positive prime factors (not necessarily distinct) whose product equals the integer -/
def length (n : ℕ) : ℕ := sorry

theorem max_length_sum (x y : ℕ) (hx : x > 1) (hy : y > 1) (hsum : x + 3 * y < 920) :
  ∃ (a b : ℕ), length x + length y ≤ a + b ∧ a + b = 16 := by
  sorry

end NUMINAMATH_CALUDE_max_length_sum_l2086_208653


namespace NUMINAMATH_CALUDE_max_ab_value_l2086_208672

theorem max_ab_value (a b : ℝ) (h : ∀ x : ℝ, Real.exp (x + 1) ≥ a * x + b) :
  (∀ c d : ℝ, (∀ x : ℝ, Real.exp (x + 1) ≥ c * x + d) → a * b ≥ c * d) ∧
  a * b = Real.exp 3 / 2 :=
sorry

end NUMINAMATH_CALUDE_max_ab_value_l2086_208672


namespace NUMINAMATH_CALUDE_total_spent_correct_l2086_208668

def calculate_total_spent (sandwich_price : Float) (sandwich_discount : Float)
                          (salad_price : Float) (salad_tax : Float)
                          (soda_price : Float) (soda_tax : Float)
                          (tip_percentage : Float) : Float :=
  let discounted_sandwich := sandwich_price * (1 - sandwich_discount)
  let taxed_salad := salad_price * (1 + salad_tax)
  let taxed_soda := soda_price * (1 + soda_tax)
  let subtotal := discounted_sandwich + taxed_salad + taxed_soda
  let total_with_tip := subtotal * (1 + tip_percentage)
  (total_with_tip * 100).round / 100

theorem total_spent_correct :
  calculate_total_spent 10.50 0.15 5.25 0.07 1.75 0.05 0.20 = 19.66 := by
  sorry


end NUMINAMATH_CALUDE_total_spent_correct_l2086_208668


namespace NUMINAMATH_CALUDE_x_value_l2086_208617

theorem x_value : ∃ x : ℚ, (3 * x - 2) / 7 = 15 ∧ x = 107 / 3 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l2086_208617


namespace NUMINAMATH_CALUDE_no_intersection_l2086_208674

-- Define the two functions
def f (x : ℝ) : ℝ := |2 * x + 5|
def g (x : ℝ) : ℝ := -|3 * x - 2|

-- Theorem statement
theorem no_intersection :
  ∀ x : ℝ, f x ≠ g x :=
sorry

end NUMINAMATH_CALUDE_no_intersection_l2086_208674


namespace NUMINAMATH_CALUDE_sqrt_of_three_times_two_five_cubed_l2086_208679

theorem sqrt_of_three_times_two_five_cubed (x : ℝ) : 
  x = Real.sqrt (2 * (5^3) + 2 * (5^3) + 2 * (5^3)) → x = 5 * Real.sqrt 30 :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_three_times_two_five_cubed_l2086_208679


namespace NUMINAMATH_CALUDE_x_value_in_set_l2086_208659

theorem x_value_in_set (x : ℝ) : 1 ∈ ({x, x^2} : Set ℝ) → x ≠ x^2 → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_x_value_in_set_l2086_208659


namespace NUMINAMATH_CALUDE_expression_equality_l2086_208614

theorem expression_equality : (5^1003 + 6^1004)^2 - (5^1003 - 6^1004)^2 = 24 * 30^1003 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2086_208614


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2086_208686

theorem triangle_perimeter (a b c : ℝ) : 
  (a ≥ 0) → (b ≥ 0) → (c ≥ 0) →
  (a^2 + 5*b^2 + c^2 - 4*a*b - 6*b - 10*c + 34 = 0) →
  (a + b + c = 14) := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l2086_208686


namespace NUMINAMATH_CALUDE_tetrahedron_division_l2086_208646

/-- A regular tetrahedron with unit edge length -/
structure RegularTetrahedron where
  edge_length : ℝ
  is_unit : edge_length = 1

/-- Perpendicular bisector plane of a tetrahedron -/
structure PerpendicularBisectorPlane (t : RegularTetrahedron) where

/-- The number of parts the perpendicular bisector planes divide the tetrahedron into -/
def num_parts (t : RegularTetrahedron) : ℕ := sorry

/-- The volume of each part after division -/
def part_volume (t : RegularTetrahedron) : ℝ := sorry

/-- Theorem stating the number of parts and their volumes -/
theorem tetrahedron_division (t : RegularTetrahedron) :
  num_parts t = 24 ∧ part_volume t = Real.sqrt 2 / 288 := by sorry

end NUMINAMATH_CALUDE_tetrahedron_division_l2086_208646


namespace NUMINAMATH_CALUDE_yoga_studio_total_people_l2086_208683

theorem yoga_studio_total_people :
  let num_men : ℕ := 8
  let num_women : ℕ := 6
  let avg_weight_men : ℝ := 190
  let avg_weight_women : ℝ := 120
  let avg_weight_all : ℝ := 160
  num_men + num_women = 14 := by
  sorry

end NUMINAMATH_CALUDE_yoga_studio_total_people_l2086_208683


namespace NUMINAMATH_CALUDE_sin_225_degrees_l2086_208635

theorem sin_225_degrees : Real.sin (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_225_degrees_l2086_208635


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2086_208650

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1)*x + 1 > 0) ↔ -1 < a ∧ a < 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2086_208650


namespace NUMINAMATH_CALUDE_c_condition_l2086_208607

theorem c_condition (a b c : ℝ) (h1 : a < b) (h2 : a * c > b * c) : c < 0 := by
  sorry

end NUMINAMATH_CALUDE_c_condition_l2086_208607


namespace NUMINAMATH_CALUDE_one_thirds_in_eleven_halves_l2086_208688

theorem one_thirds_in_eleven_halves : (11 / 2) / (1 / 3) = 33 / 2 := by
  sorry

end NUMINAMATH_CALUDE_one_thirds_in_eleven_halves_l2086_208688


namespace NUMINAMATH_CALUDE_old_edition_pages_l2086_208639

theorem old_edition_pages (new_edition : ℕ) (h1 : new_edition = 450) 
  (h2 : new_edition = 2 * old_edition - 230) : old_edition = 340 :=
by
  sorry

end NUMINAMATH_CALUDE_old_edition_pages_l2086_208639


namespace NUMINAMATH_CALUDE_unique_krakozyabr_count_l2086_208680

def Krakozyabr : Type := Unit

structure KrakozyabrPopulation where
  total : ℕ
  horns : ℕ
  wings : ℕ
  both : ℕ
  all_have_horns_or_wings : total = horns + wings - both
  horns_with_wings_ratio : both = horns / 5
  wings_with_horns_ratio : both = wings / 4
  total_range : 25 < total ∧ total < 35

theorem unique_krakozyabr_count : 
  ∀ (pop : KrakozyabrPopulation), pop.total = 32 := by
  sorry

end NUMINAMATH_CALUDE_unique_krakozyabr_count_l2086_208680


namespace NUMINAMATH_CALUDE_slope_is_negative_one_l2086_208666

/-- The slope of a line through two points is -1 -/
theorem slope_is_negative_one (P Q : ℝ × ℝ) : 
  P = (-3, 8) → 
  Q.1 = 5 → 
  Q.2 = 0 → 
  (Q.2 - P.2) / (Q.1 - P.1) = -1 := by
sorry

end NUMINAMATH_CALUDE_slope_is_negative_one_l2086_208666


namespace NUMINAMATH_CALUDE_expression_evaluation_l2086_208645

theorem expression_evaluation : (-6)^6 / 6^4 + 4^5 - 7^2 * 2 = 890 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2086_208645


namespace NUMINAMATH_CALUDE_adams_apples_l2086_208605

/-- 
Given:
- Jackie has 10 apples
- Jackie has 2 more apples than Adam
Prove that Adam has 8 apples
-/
theorem adams_apples (jackie_apples : ℕ) (adam_apples : ℕ) 
  (h1 : jackie_apples = 10)
  (h2 : jackie_apples = adam_apples + 2) : 
  adam_apples = 8 := by
  sorry

end NUMINAMATH_CALUDE_adams_apples_l2086_208605


namespace NUMINAMATH_CALUDE_garden_roller_length_l2086_208694

theorem garden_roller_length :
  let diameter : ℝ := 1.4
  let area_covered : ℝ := 66
  let revolutions : ℝ := 5
  let π : ℝ := 22 / 7
  let radius : ℝ := diameter / 2
  let length : ℝ := (area_covered / revolutions) / (2 * π * radius)
  length = 2.1 := by sorry

end NUMINAMATH_CALUDE_garden_roller_length_l2086_208694


namespace NUMINAMATH_CALUDE_systematic_sample_result_l2086_208644

def systematic_sample (total : ℕ) (sample_size : ℕ) (interval_start : ℕ) (interval_end : ℕ) : ℕ :=
  let sample_interval := total / sample_size
  let interval_size := interval_end - interval_start + 1
  interval_size / sample_interval

theorem systematic_sample_result :
  systematic_sample 360 20 181 288 = 6 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sample_result_l2086_208644


namespace NUMINAMATH_CALUDE_odd_square_minus_one_div_eight_l2086_208610

theorem odd_square_minus_one_div_eight (n : ℤ) : 
  ∃ k : ℤ, (2*n + 1)^2 - 1 = 8*k := by sorry

end NUMINAMATH_CALUDE_odd_square_minus_one_div_eight_l2086_208610


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2086_208651

theorem absolute_value_inequality (x : ℝ) : 
  (|x - 2| + |x - 3| < 9) ↔ (-2 < x ∧ x < 7) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2086_208651


namespace NUMINAMATH_CALUDE_eli_age_difference_l2086_208613

/-- Given the ages and relationships between Kaylin, Sarah, Eli, and Freyja, prove that Eli is 9 years older than Freyja. -/
theorem eli_age_difference (kaylin sarah eli freyja : ℕ) : 
  kaylin = 33 →
  freyja = 10 →
  sarah = kaylin + 5 →
  sarah = 2 * eli →
  eli > freyja →
  eli - freyja = 9 := by
  sorry

end NUMINAMATH_CALUDE_eli_age_difference_l2086_208613


namespace NUMINAMATH_CALUDE_zero_points_sum_inequality_l2086_208642

theorem zero_points_sum_inequality (a : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₁ ≠ x₂)
  (h₄ : Real.log x₁ - a * x₁ = 0) (h₅ : Real.log x₂ - a * x₂ = 0) :
  x₁ + x₂ > 2 / a :=
by sorry

end NUMINAMATH_CALUDE_zero_points_sum_inequality_l2086_208642


namespace NUMINAMATH_CALUDE_quadratic_minimum_l2086_208678

theorem quadratic_minimum (x : ℝ) : 
  (∀ x, 2 * x^2 - 8 * x + 15 ≥ 7) ∧ (∃ x, 2 * x^2 - 8 * x + 15 = 7) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l2086_208678


namespace NUMINAMATH_CALUDE_square_perimeter_47_20_l2086_208638

-- Define the side length of the square
def side_length : ℚ := 47 / 20

-- Define the perimeter of a square
def square_perimeter (s : ℚ) : ℚ := 4 * s

-- Theorem statement
theorem square_perimeter_47_20 : 
  square_perimeter side_length = 47 / 5 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_47_20_l2086_208638


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l2086_208691

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 and eccentricity e,
    prove that its asymptotes are √3x ± y = 0 when e = 2 -/
theorem hyperbola_asymptotes (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) :
  let e := Real.sqrt (a^2 + b^2) / a
  e = 2 →
  ∃ (k : ℝ), k = Real.sqrt 3 ∧
    (∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 →
      (y = k * x ∨ y = -k * x)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l2086_208691


namespace NUMINAMATH_CALUDE_parabola_line_theorem_l2086_208661

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (m : ℝ) (x y : ℝ) : Prop := x = m*y + 1

-- Define the perpendicular bisector of a line with slope m
def perpendicular_bisector (m : ℝ) (x y : ℝ) : Prop := x = -1/m * y + (2*m^2 + 3)

-- Define the condition for points to lie on the same circle
def on_same_circle (A B M N : ℝ × ℝ) : Prop := 
  let midpoint_AB := ((A.1 + B.1)/2, (A.2 + B.2)/2)
  let midpoint_MN := ((M.1 + N.1)/2, (M.2 + N.2)/2)
  (A.1 - midpoint_MN.1)^2 + (A.2 - midpoint_MN.2)^2 = 
  (M.1 - midpoint_AB.1)^2 + (M.2 - midpoint_AB.2)^2

theorem parabola_line_theorem (m : ℝ) :
  ∃ A B M N : ℝ × ℝ,
    parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ parabola M.1 M.2 ∧ parabola N.1 N.2 ∧
    line_through_focus m A.1 A.2 ∧ line_through_focus m B.1 B.2 ∧
    perpendicular_bisector m M.1 M.2 ∧ perpendicular_bisector m N.1 N.2 ∧
    on_same_circle A B M N →
  m = 1 ∨ m = -1 := by sorry

end NUMINAMATH_CALUDE_parabola_line_theorem_l2086_208661


namespace NUMINAMATH_CALUDE_all_equilateral_triangles_similar_l2086_208681

/-- An equilateral triangle -/
structure EquilateralTriangle where
  side : ℝ
  side_positive : side > 0

/-- Similarity relation between two equilateral triangles -/
def similar (t1 t2 : EquilateralTriangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ t1.side = k * t2.side

/-- All angles in an equilateral triangle are 60° -/
axiom equilateral_angle (t : EquilateralTriangle) : 
  ∀ angle, angle = 60

/-- Theorem: Any two equilateral triangles are similar -/
theorem all_equilateral_triangles_similar (t1 t2 : EquilateralTriangle) :
  similar t1 t2 := by
  sorry

end NUMINAMATH_CALUDE_all_equilateral_triangles_similar_l2086_208681


namespace NUMINAMATH_CALUDE_bakers_sales_l2086_208656

/-- Baker's cake and pastry problem -/
theorem bakers_sales (cakes_made pastries_made pastries_sold : ℕ) 
  (h1 : cakes_made = 157)
  (h2 : pastries_made = 169)
  (h3 : pastries_sold = 147)
  (h4 : ∃ cakes_sold : ℕ, cakes_sold = pastries_sold + 11) :
  ∃ cakes_sold : ℕ, cakes_sold = 158 := by
  sorry

end NUMINAMATH_CALUDE_bakers_sales_l2086_208656


namespace NUMINAMATH_CALUDE_divisor_problem_l2086_208695

theorem divisor_problem (d : ℕ) : d > 0 ∧ 109 = 9 * d + 1 → d = 12 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l2086_208695


namespace NUMINAMATH_CALUDE_last_digit_power_sum_l2086_208655

theorem last_digit_power_sum (m : ℕ+) : (2^(m.val + 2007) + 2^(m.val + 1)) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_power_sum_l2086_208655


namespace NUMINAMATH_CALUDE_perpendicular_lines_m_values_l2086_208634

-- Define the coefficients of the two lines as functions of m
def line1_coeff (m : ℝ) : ℝ × ℝ := (m + 2, m)
def line2_coeff (m : ℝ) : ℝ × ℝ := (m - 1, m - 4)

-- Define the perpendicularity condition
def perpendicular (m : ℝ) : Prop :=
  (line1_coeff m).1 * (line2_coeff m).1 + (line1_coeff m).2 * (line2_coeff m).2 = 0

-- State the theorem
theorem perpendicular_lines_m_values :
  ∀ m : ℝ, perpendicular m → m = -1/2 ∨ m = 2 := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_m_values_l2086_208634


namespace NUMINAMATH_CALUDE_cost_of_three_pencils_two_pens_l2086_208684

/-- The cost of a single pencil -/
def pencil_cost : ℝ := sorry

/-- The cost of a single pen -/
def pen_cost : ℝ := sorry

/-- The total cost of three pencils and two pens is $4.15 -/
axiom three_pencils_two_pens : 3 * pencil_cost + 2 * pen_cost = 4.15

/-- The cost of two pencils and three pens is $3.70 -/
axiom two_pencils_three_pens : 2 * pencil_cost + 3 * pen_cost = 3.70

/-- The cost of three pencils and two pens is $4.15 -/
theorem cost_of_three_pencils_two_pens : 3 * pencil_cost + 2 * pen_cost = 4.15 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_three_pencils_two_pens_l2086_208684


namespace NUMINAMATH_CALUDE_aaron_guitar_loan_l2086_208677

/-- Calculates the total amount owed for a loan with monthly payments and interest. -/
def totalAmountOwed (monthlyPayment : ℝ) (numberOfMonths : ℕ) (interestRate : ℝ) : ℝ :=
  let totalWithoutInterest := monthlyPayment * numberOfMonths
  let interestAmount := totalWithoutInterest * interestRate
  totalWithoutInterest + interestAmount

/-- Theorem stating that given the specific conditions of Aaron's guitar purchase,
    the total amount owed is $1320. -/
theorem aaron_guitar_loan :
  totalAmountOwed 100 12 0.1 = 1320 := by
  sorry

end NUMINAMATH_CALUDE_aaron_guitar_loan_l2086_208677


namespace NUMINAMATH_CALUDE_sequence_pattern_l2086_208649

def S : ℕ → ℕ
  | n => if n % 2 = 1 then 1 + 2 * ((n + 1) / 2 - 1) else 2^(n / 2)

theorem sequence_pattern (n : ℕ) : 
  S n = if n % 2 = 1 then 1 + 2 * ((n + 1) / 2 - 1) else 2^(n / 2) := by
  sorry

#eval [S 1, S 2, S 3, S 4, S 5, S 6, S 7, S 8, S 9, S 10, S 11, S 12, S 13, S 14]

end NUMINAMATH_CALUDE_sequence_pattern_l2086_208649


namespace NUMINAMATH_CALUDE_parity_of_A_15_16_17_l2086_208615

def A : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | 2 => 0
  | n + 3 => A (n + 2) + A n

theorem parity_of_A_15_16_17 : 
  Odd (A 15) ∧ Even (A 16) ∧ Odd (A 17) := by sorry

end NUMINAMATH_CALUDE_parity_of_A_15_16_17_l2086_208615


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2086_208621

theorem arithmetic_sequence_problem (a : ℕ → ℚ) (S : ℕ → ℚ) : 
  (∀ n, a (n + 1) - a n = 1) →  -- Common difference is 1
  (∀ n, S n = (n * (a 1 + a n)) / 2) →  -- Sum formula for arithmetic sequence
  (S 6 = 4 * S 3) →  -- Given condition
  a 10 = 19 / 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2086_208621


namespace NUMINAMATH_CALUDE_smallest_box_volume_l2086_208658

theorem smallest_box_volume (l w h : ℕ) (h1 : l > 0) (h2 : w = 3 * l) (h3 : h = 4 * l) :
  l * w * h = 96 ∨ l * w * h > 96 :=
sorry

end NUMINAMATH_CALUDE_smallest_box_volume_l2086_208658


namespace NUMINAMATH_CALUDE_assignment_methods_eq_eight_l2086_208632

/-- Represents the number of schools --/
def num_schools : ℕ := 2

/-- Represents the number of student teachers --/
def num_teachers : ℕ := 4

/-- Calculates the number of assignment methods --/
def assignment_methods : ℕ := 
  let a_assignments := num_schools -- A can be assigned to either school
  let b_assignments := num_schools - 1 -- B must be assigned to the other school
  let remaining_assignments := num_schools ^ (num_teachers - 2) -- Remaining 2 teachers can be assigned freely
  a_assignments * b_assignments * remaining_assignments

/-- Theorem stating that the number of assignment methods is 8 --/
theorem assignment_methods_eq_eight : assignment_methods = 8 := by
  sorry

end NUMINAMATH_CALUDE_assignment_methods_eq_eight_l2086_208632


namespace NUMINAMATH_CALUDE_square_polynomial_k_values_l2086_208631

theorem square_polynomial_k_values (k : ℝ) : 
  (∃ p : ℝ → ℝ, ∀ x, x^2 + 2*(k-1)*x + 64 = (p x)^2) → 
  (k = 9 ∨ k = -7) := by
  sorry

end NUMINAMATH_CALUDE_square_polynomial_k_values_l2086_208631


namespace NUMINAMATH_CALUDE_zeros_of_f_l2086_208648

def f (x : ℝ) := x^2 - 3*x + 2

theorem zeros_of_f :
  ∀ x : ℝ, f x = 0 ↔ x = 1 ∨ x = 2 := by
sorry

end NUMINAMATH_CALUDE_zeros_of_f_l2086_208648


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2086_208608

theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 2) + 3
  f 2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2086_208608


namespace NUMINAMATH_CALUDE_complex_equation_implies_sum_of_squares_l2086_208626

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- Given complex equation -/
def complex_equation (a b : ℝ) : Prop :=
  (4 - 3 * i) * (a + b * i) = 25 * i

/-- Theorem stating that the given complex equation implies a² + b² = 25 -/
theorem complex_equation_implies_sum_of_squares (a b : ℝ) :
  complex_equation a b → a^2 + b^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_implies_sum_of_squares_l2086_208626


namespace NUMINAMATH_CALUDE_right_triangle_sets_l2086_208654

/-- A function that checks if three numbers can form a right-angled triangle -/
def isRightTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a + b > c ∧ b + c > a ∧ c + a > b ∧
  a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2

/-- The theorem stating that among the given sets, only {3, 4, 5} forms a right-angled triangle -/
theorem right_triangle_sets : 
  ¬ isRightTriangle 1 2 3 ∧
  isRightTriangle 3 4 5 ∧
  ¬ isRightTriangle 7 8 9 ∧
  ¬ isRightTriangle 5 10 20 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l2086_208654


namespace NUMINAMATH_CALUDE_complex_sum_direction_l2086_208609

theorem complex_sum_direction (r : ℝ) (h : r > 0) :
  ∃ (r : ℝ), r > 0 ∧ 
  Complex.exp (11 * Real.pi * Complex.I / 60) +
  Complex.exp (21 * Real.pi * Complex.I / 60) +
  Complex.exp (31 * Real.pi * Complex.I / 60) +
  Complex.exp (41 * Real.pi * Complex.I / 60) +
  Complex.exp (51 * Real.pi * Complex.I / 60) =
  r * Complex.exp (31 * Real.pi * Complex.I / 60) :=
by sorry

end NUMINAMATH_CALUDE_complex_sum_direction_l2086_208609


namespace NUMINAMATH_CALUDE_total_tomatoes_l2086_208611

/-- The number of cucumber rows for each tomato row -/
def cucumber_rows_per_tomato_row : ℕ := 2

/-- The total number of rows in the garden -/
def total_rows : ℕ := 15

/-- The number of tomato plants in each row -/
def plants_per_row : ℕ := 8

/-- The number of tomatoes produced by each plant -/
def tomatoes_per_plant : ℕ := 3

/-- The theorem stating the total number of tomatoes Aubrey will have -/
theorem total_tomatoes : 
  (total_rows / (cucumber_rows_per_tomato_row + 1)) * plants_per_row * tomatoes_per_plant = 120 := by
  sorry

end NUMINAMATH_CALUDE_total_tomatoes_l2086_208611


namespace NUMINAMATH_CALUDE_five_in_C_l2086_208692

def C : Set ℕ := {x | 1 ≤ x ∧ x < 10}

theorem five_in_C : 5 ∈ C := by sorry

end NUMINAMATH_CALUDE_five_in_C_l2086_208692


namespace NUMINAMATH_CALUDE_kate_bouncy_balls_l2086_208643

/-- The number of packs of yellow bouncy balls Kate bought -/
def yellow_packs : ℕ := 6

/-- The number of packs of red bouncy balls Kate bought -/
def red_packs : ℕ := 7

/-- The number of bouncy balls in each pack -/
def balls_per_pack : ℕ := 18

theorem kate_bouncy_balls :
  yellow_packs * balls_per_pack + 18 = red_packs * balls_per_pack :=
by sorry

end NUMINAMATH_CALUDE_kate_bouncy_balls_l2086_208643


namespace NUMINAMATH_CALUDE_ellipse_condition_l2086_208670

-- Define the equation
def equation (x y z m : ℝ) : Prop :=
  3 * x^2 + 9 * y^2 - 12 * x + 18 * y + 6 * z = m

-- Define what it means for the equation to represent a non-degenerate ellipse when projected onto the xy-plane
def is_nondegenerate_ellipse_projection (m : ℝ) : Prop :=
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ ∀ (x y : ℝ), 
    ∃ (z : ℝ), equation x y z m ↔ (x - c)^2 / a + (y - c)^2 / b = 1

-- State the theorem
theorem ellipse_condition (m : ℝ) : 
  is_nondegenerate_ellipse_projection m ↔ m > -21 :=
sorry

end NUMINAMATH_CALUDE_ellipse_condition_l2086_208670


namespace NUMINAMATH_CALUDE_supermarket_sales_problem_l2086_208619

/-- Represents the monthly sales volume as a function of selling price -/
def sales_volume (x : ℝ) : ℝ := -5 * x + 550

/-- Represents the monthly profit as a function of selling price -/
def monthly_profit (x : ℝ) : ℝ := sales_volume x * (x - 50)

/-- The cost per item -/
def cost : ℝ := 50

/-- The initial selling price -/
def initial_price : ℝ := 100

/-- The initial monthly sales -/
def initial_sales : ℝ := 50

/-- The change in sales for every 2 yuan decrease in price -/
def sales_change : ℝ := 10

theorem supermarket_sales_problem :
  (∀ x : ℝ, x ≥ cost → sales_volume x = -5 * x + 550) ∧
  (∃ x : ℝ, x ≥ cost ∧ monthly_profit x = 4000 ∧ x = 70) ∧
  (∃ x : ℝ, x ≥ cost ∧ ∀ y : ℝ, y ≥ cost → monthly_profit x ≥ monthly_profit y ∧ x = 80) :=
by sorry

end NUMINAMATH_CALUDE_supermarket_sales_problem_l2086_208619


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l2086_208675

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 10| + |x - 14| = |2*x - 24| :=
by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l2086_208675


namespace NUMINAMATH_CALUDE_fraction_with_buddies_l2086_208690

/-- Represents the number of students in each grade --/
structure StudentCounts where
  ninth : ℚ
  sixth : ℚ
  seventh : ℚ

/-- Represents the pairing ratios --/
structure PairingRatios where
  ninth : ℚ
  sixth : ℚ

/-- Represents the school mentoring program --/
structure MentoringProgram where
  counts : StudentCounts
  ratios : PairingRatios

/-- The main theorem about the fraction of students with buddies --/
theorem fraction_with_buddies (program : MentoringProgram) 
  (h1 : program.counts.ninth = 5 * program.counts.sixth / 4)
  (h2 : program.counts.seventh = 3 * program.counts.sixth / 4)
  (h3 : program.ratios.ninth = 1/4)
  (h4 : program.ratios.sixth = 1/3)
  (h5 : program.ratios.ninth * program.counts.ninth = program.ratios.sixth * program.counts.sixth) :
  (program.ratios.ninth * program.counts.ninth) / 
  (program.counts.ninth + program.counts.sixth + program.counts.seventh) = 5/48 := by
  sorry

end NUMINAMATH_CALUDE_fraction_with_buddies_l2086_208690


namespace NUMINAMATH_CALUDE_no_three_primes_sum_squares_l2086_208633

theorem no_three_primes_sum_squares : ¬∃ (p q r : ℕ), 
  p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ 
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ 
  ∃ (a b c : ℕ), p + q = a^2 ∧ p + r = b^2 ∧ q + r = c^2 :=
by sorry

end NUMINAMATH_CALUDE_no_three_primes_sum_squares_l2086_208633


namespace NUMINAMATH_CALUDE_shaded_area_of_carpet_l2086_208618

/-- Given a square carpet with side length 12 feet, one large shaded square,
    and twelve smaller congruent shaded squares, where the ratios of side lengths
    are as specified, the total shaded area is 15.75 square feet. -/
theorem shaded_area_of_carpet (S T : ℝ) : 
  (12 : ℝ) / S = 4 →
  S / T = 4 →
  S^2 + 12 * T^2 = 15.75 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_of_carpet_l2086_208618


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l2086_208625

/-- The slope of the asymptotes for the hyperbola (x^2 / 144) - (y^2 / 81) = 1 is 3/4 -/
theorem hyperbola_asymptote_slope :
  let hyperbola := fun (x y : ℝ) => x^2 / 144 - y^2 / 81 = 1
  ∃ m : ℝ, m = 3/4 ∧ 
    ∀ (x y : ℝ), hyperbola x y → (y = m * x ∨ y = -m * x) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l2086_208625


namespace NUMINAMATH_CALUDE_solution_range_l2086_208699

theorem solution_range (b : ℝ) :
  (∀ x : ℝ, x^2 - b*x - 5 = 5 → (x = -2 ∨ x = 5)) →
  (∀ x : ℝ, x^2 - b*x - 5 = -1 → (x = -1 ∨ x = 4)) →
  ∃ x₁ x₂ : ℝ, 
    (x₁^2 - b*x₁ - 5 = 0 ∧ -2 < x₁ ∧ x₁ < -1) ∧
    (x₂^2 - b*x₂ - 5 = 0 ∧ 4 < x₂ ∧ x₂ < 5) ∧
    (∀ x : ℝ, x^2 - b*x - 5 = 0 → ((-2 < x ∧ x < -1) ∨ (4 < x ∧ x < 5))) := by
  sorry

end NUMINAMATH_CALUDE_solution_range_l2086_208699


namespace NUMINAMATH_CALUDE_cos_two_beta_l2086_208696

theorem cos_two_beta (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.tan α = 7) (h4 : Real.sin (α - β) = Real.sqrt 10 / 10) :
  Real.cos (2 * β) = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_two_beta_l2086_208696


namespace NUMINAMATH_CALUDE_gas_pressure_volume_relationship_l2086_208623

/-- Given a gas with initial pressure P1, initial volume V1, and final volume V2,
    where pressure and volume are inversely proportional at constant temperature,
    prove that the final pressure P2 is equal to (P1 * V1) / V2. -/
theorem gas_pressure_volume_relationship (P1 V1 V2 : ℝ) (h1 : P1 > 0) (h2 : V1 > 0) (h3 : V2 > 0) :
  let P2 := (P1 * V1) / V2
  ∀ k : ℝ, (P1 * V1 = k ∧ P2 * V2 = k) → P2 = (P1 * V1) / V2 := by
sorry

end NUMINAMATH_CALUDE_gas_pressure_volume_relationship_l2086_208623


namespace NUMINAMATH_CALUDE_quadratic_positive_combination_l2086_208612

/-- A quadratic function is a function of the form ax² + bx + c where a ≠ 0 -/
def IsQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- Two intervals are disjoint if they have no common points -/
def DisjointIntervals (I J : Set ℝ) : Prop :=
  I ∩ J = ∅

/-- A function is negative on an interval if it takes negative values for all points in that interval -/
def NegativeOnInterval (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x ∈ I, f x < 0

theorem quadratic_positive_combination
  (f g : ℝ → ℝ)
  (hf : IsQuadratic f)
  (hg : IsQuadratic g)
  (hfI : ∃ I : Set ℝ, NegativeOnInterval f I)
  (hgJ : ∃ J : Set ℝ, NegativeOnInterval g J)
  (hIJ : ∀ I J, (NegativeOnInterval f I ∧ NegativeOnInterval g J) → DisjointIntervals I J) :
  ∃ α β : ℝ, α > 0 ∧ β > 0 ∧ ∀ x, α * f x + β * g x > 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_positive_combination_l2086_208612


namespace NUMINAMATH_CALUDE_brick_height_is_6cm_l2086_208664

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular object given its dimensions -/
def volume (d : Dimensions) : ℝ := d.length * d.width * d.height

/-- The dimensions of the wall in centimeters -/
def wall_dimensions : Dimensions := ⟨800, 600, 22.5⟩

/-- The known dimensions of a brick in centimeters -/
def brick_dimensions (height : ℝ) : Dimensions := ⟨80, 11.25, height⟩

/-- The number of bricks needed to build the wall -/
def num_bricks : ℕ := 2000

/-- Theorem stating that the height of each brick is 6 cm -/
theorem brick_height_is_6cm :
  ∃ (h : ℝ), h = 6 ∧ 
  volume wall_dimensions = ↑num_bricks * volume (brick_dimensions h) := by
  sorry

end NUMINAMATH_CALUDE_brick_height_is_6cm_l2086_208664


namespace NUMINAMATH_CALUDE_no_rectangular_parallelepiped_sum_866_l2086_208606

theorem no_rectangular_parallelepiped_sum_866 :
  ¬∃ (x y z : ℕ+), x * y * z + 2 * (x * y + x * z + y * z) + 4 * (x + y + z) = 866 := by
sorry

end NUMINAMATH_CALUDE_no_rectangular_parallelepiped_sum_866_l2086_208606


namespace NUMINAMATH_CALUDE_factorial_ratio_50_48_l2086_208629

theorem factorial_ratio_50_48 : Nat.factorial 50 / Nat.factorial 48 = 2450 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_50_48_l2086_208629


namespace NUMINAMATH_CALUDE_q_div_p_eq_275_l2086_208662

/-- The number of cards in the box -/
def total_cards : ℕ := 60

/-- The number of distinct numbers on the cards -/
def distinct_numbers : ℕ := 12

/-- The number of cards for each number -/
def cards_per_number : ℕ := 5

/-- The number of cards drawn -/
def cards_drawn : ℕ := 5

/-- The probability of drawing all cards with the same number -/
def p : ℚ := (distinct_numbers : ℚ) / Nat.choose total_cards cards_drawn

/-- The probability of drawing four cards with one number and one card with a different number -/
def q : ℚ := (3300 : ℚ) / Nat.choose total_cards cards_drawn

/-- The main theorem stating that q/p = 275 -/
theorem q_div_p_eq_275 : q / p = 275 := by sorry

end NUMINAMATH_CALUDE_q_div_p_eq_275_l2086_208662


namespace NUMINAMATH_CALUDE_john_average_bottle_price_l2086_208641

/-- The average price of bottles purchased by John -/
def average_price (large_quantity : ℕ) (large_price : ℚ) (small_quantity : ℕ) (small_price : ℚ) : ℚ :=
  (large_quantity * large_price + small_quantity * small_price) / (large_quantity + small_quantity)

/-- The average price of bottles purchased by John is approximately $1.70 -/
theorem john_average_bottle_price :
  let large_quantity : ℕ := 1300
  let large_price : ℚ := 189/100
  let small_quantity : ℕ := 750
  let small_price : ℚ := 138/100
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/100 ∧ 
    |average_price large_quantity large_price small_quantity small_price - 17/10| < ε :=
sorry

end NUMINAMATH_CALUDE_john_average_bottle_price_l2086_208641


namespace NUMINAMATH_CALUDE_agathas_bike_purchase_l2086_208671

/-- Agatha's bike purchase problem -/
theorem agathas_bike_purchase (frame_cost seat_handlebar_cost front_wheel_cost remaining_money : ℕ) 
  (h1 : frame_cost = 15)
  (h2 : front_wheel_cost = 25)
  (h3 : remaining_money = 20) :
  frame_cost + front_wheel_cost + remaining_money = 60 := by
  sorry

#check agathas_bike_purchase

end NUMINAMATH_CALUDE_agathas_bike_purchase_l2086_208671
