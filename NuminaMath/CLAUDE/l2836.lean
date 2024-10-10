import Mathlib

namespace count_integers_satisfying_inequality_l2836_283640

theorem count_integers_satisfying_inequality :
  ∃! (S : Finset ℤ), (∀ n ∈ S, Real.sqrt (3 * n) ≤ Real.sqrt (5 * n - 8) ∧
                                Real.sqrt (5 * n - 8) < Real.sqrt (3 * n + 7)) ∧
                     S.card = 4 := by
  sorry

end count_integers_satisfying_inequality_l2836_283640


namespace divide_eight_by_repeating_third_l2836_283643

-- Define the repeating decimal 0.overline{3}
def repeating_third : ℚ := 1/3

-- State the theorem
theorem divide_eight_by_repeating_third : 8 / repeating_third = 24 := by
  sorry

end divide_eight_by_repeating_third_l2836_283643


namespace tan_sum_special_l2836_283611

theorem tan_sum_special : Real.tan (17 * π / 180) + Real.tan (28 * π / 180) + Real.tan (17 * π / 180) * Real.tan (28 * π / 180) = 1 := by
  sorry

end tan_sum_special_l2836_283611


namespace fruit_drink_composition_l2836_283642

/-- Represents a fruit drink mixture -/
structure FruitDrink where
  total_volume : ℝ
  grape_volume : ℝ
  orange_percent : ℝ
  watermelon_percent : ℝ
  grape_percent : ℝ

/-- The theorem statement -/
theorem fruit_drink_composition (drink : FruitDrink)
  (h1 : drink.total_volume = 150)
  (h2 : drink.grape_volume = 45)
  (h3 : drink.orange_percent = drink.watermelon_percent)
  (h4 : drink.orange_percent + drink.watermelon_percent + drink.grape_percent = 100)
  (h5 : drink.grape_volume / drink.total_volume * 100 = drink.grape_percent) :
  drink.orange_percent = 35 ∧ drink.watermelon_percent = 35 := by
  sorry

end fruit_drink_composition_l2836_283642


namespace complex_equation_solution_l2836_283676

theorem complex_equation_solution (a : ℝ) (i : ℂ) (h1 : i * i = -1) :
  (Complex.im ((1 - i^2023) / (a * i)) = 3) → a = -1/3 := by
  sorry

end complex_equation_solution_l2836_283676


namespace max_pairs_sum_l2836_283682

theorem max_pairs_sum (n : ℕ) (h : n = 2023) :
  ∃ (k : ℕ) (pairs : List (ℕ × ℕ)),
    k = 813 ∧
    pairs.length = k ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 < p.2 ∧ p.1 ∈ Finset.range n ∧ p.2 ∈ Finset.range n) ∧
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) ∧
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 + p.2 ≠ q.1 + q.2) ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 + p.2 ≤ 2033) ∧
    (∀ (m : ℕ) (pairs' : List (ℕ × ℕ)),
      m > k →
      (∀ (p : ℕ × ℕ), p ∈ pairs' → p.1 < p.2 ∧ p.1 ∈ Finset.range n ∧ p.2 ∈ Finset.range n) →
      (∀ (p q : ℕ × ℕ), p ∈ pairs' → q ∈ pairs' → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) →
      (∀ (p q : ℕ × ℕ), p ∈ pairs' → q ∈ pairs' → p ≠ q → p.1 + p.2 ≠ q.1 + q.2) →
      (∀ (p : ℕ × ℕ), p ∈ pairs' → p.1 + p.2 ≤ 2033) →
      False) :=
by
  sorry

end max_pairs_sum_l2836_283682


namespace count_numbers_satisfying_condition_l2836_283637

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def satisfies_condition (n : ℕ) : Prop :=
  n < 2000 ∧ n = 9 * sum_of_digits n

theorem count_numbers_satisfying_condition :
  ∃ (S : Finset ℕ), (∀ n ∈ S, satisfies_condition n) ∧ S.card = 4 :=
sorry

end count_numbers_satisfying_condition_l2836_283637


namespace star_three_two_l2836_283672

/-- The star operation defined as a * b = a * b^3 - b^2 + 2 -/
def star (a b : ℝ) : ℝ := a * b^3 - b^2 + 2

/-- Theorem stating that 3 star 2 equals 22 -/
theorem star_three_two : star 3 2 = 22 := by
  sorry

end star_three_two_l2836_283672


namespace divisibility_by_five_l2836_283618

theorem divisibility_by_five (a b c d e f g : ℕ) 
  (h1 : (a + b + c + d + e + f) % 5 = 0)
  (h2 : (a + b + c + d + e + g) % 5 = 0)
  (h3 : (a + b + c + d + f + g) % 5 = 0)
  (h4 : (a + b + c + e + f + g) % 5 = 0)
  (h5 : (a + b + d + e + f + g) % 5 = 0)
  (h6 : (a + c + d + e + f + g) % 5 = 0)
  (h7 : (b + c + d + e + f + g) % 5 = 0) :
  (a % 5 = 0) ∧ (b % 5 = 0) ∧ (c % 5 = 0) ∧ (d % 5 = 0) ∧ 
  (e % 5 = 0) ∧ (f % 5 = 0) ∧ (g % 5 = 0) := by
  sorry

#check divisibility_by_five

end divisibility_by_five_l2836_283618


namespace march_pancake_expense_l2836_283675

/-- Given the total expense on pancakes in March and the number of days,
    calculate the daily expense assuming equal consumption each day. -/
def daily_pancake_expense (total_expense : ℕ) (days : ℕ) : ℕ :=
  total_expense / days

/-- Theorem stating that the daily pancake expense in March is 11 dollars -/
theorem march_pancake_expense :
  daily_pancake_expense 341 31 = 11 := by
  sorry

end march_pancake_expense_l2836_283675


namespace complex_symmetry_product_l2836_283684

theorem complex_symmetry_product (z₁ z₂ : ℂ) : 
  z₁ = 2 + I → 
  z₂.re = -z₁.re ∧ z₂.im = z₁.im → 
  z₁ * z₂ = -5 := by sorry

end complex_symmetry_product_l2836_283684


namespace quadratic_roots_l2836_283695

theorem quadratic_roots (k : ℝ) (C D : ℝ) : 
  (k * C^2 + 2 * C + 5 = 0) →
  (k * D^2 + 2 * D + 5 = 0) →
  (C = 10) →
  (D = -2) :=
by
  sorry

end quadratic_roots_l2836_283695


namespace prism_pyramid_sum_l2836_283632

/-- Represents a shape formed by adding a pyramid to one rectangular face of a right rectangular prism -/
structure PrismPyramid where
  prism_faces : Nat
  prism_edges : Nat
  prism_vertices : Nat
  pyramid_new_faces : Nat
  pyramid_new_edges : Nat
  pyramid_new_vertex : Nat

/-- The total number of exterior faces, edges, and vertices of the combined shape -/
def total_elements (shape : PrismPyramid) : Nat :=
  (shape.prism_faces + shape.pyramid_new_faces - 1) +
  (shape.prism_edges + shape.pyramid_new_edges) +
  (shape.prism_vertices + shape.pyramid_new_vertex)

/-- Theorem stating that the sum of exterior faces, edges, and vertices is 34 -/
theorem prism_pyramid_sum :
  ∀ (shape : PrismPyramid),
    shape.prism_faces = 6 ∧
    shape.prism_edges = 12 ∧
    shape.prism_vertices = 8 ∧
    shape.pyramid_new_faces = 4 ∧
    shape.pyramid_new_edges = 4 ∧
    shape.pyramid_new_vertex = 1 →
    total_elements shape = 34 := by
  sorry

end prism_pyramid_sum_l2836_283632


namespace arithmetic_progression_rth_term_l2836_283667

/-- The sum of the first n terms of an arithmetic progression -/
def S (n : ℕ) : ℝ := 3 * n + 4 * n^2

/-- The rth term of the arithmetic progression -/
def a (r : ℕ) : ℝ := 8 * r - 1

theorem arithmetic_progression_rth_term (r : ℕ) :
  a r = S r - S (r - 1) := by sorry

end arithmetic_progression_rth_term_l2836_283667


namespace sum_of_reciprocals_l2836_283650

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 16) (h2 : x * y = 55) :
  1 / x + 1 / y = 16 / 55 := by
  sorry

end sum_of_reciprocals_l2836_283650


namespace f_divisible_by_13_l2836_283629

def f : ℕ → ℤ
  | 0 => 0
  | 1 => 0
  | (n + 2) => (4 * (n + 2) * f (n + 1) - 16 * (n + 1) * f n + n^2 * n^2) / n

theorem f_divisible_by_13 :
  13 ∣ f 1989 ∧ 13 ∣ f 1990 ∧ 13 ∣ f 1991 := by sorry

end f_divisible_by_13_l2836_283629


namespace house_transaction_problem_l2836_283621

/-- Represents a person's assets -/
structure Assets where
  cash : Int
  has_house : Bool

/-- Represents a transaction between two people -/
def transaction (seller buyer : Assets) (price : Int) : Assets × Assets :=
  ({ cash := seller.cash + price, has_house := false },
   { cash := buyer.cash - price, has_house := true })

/-- The problem statement -/
theorem house_transaction_problem :
  let initial_a : Assets := { cash := 15000, has_house := true }
  let initial_b : Assets := { cash := 20000, has_house := false }
  let house_value := 15000

  let (a1, b1) := transaction initial_a initial_b 18000
  let (b2, a2) := transaction b1 a1 12000
  let (a3, b3) := transaction a2 b2 16000

  (a3.cash - initial_a.cash = 22000) ∧
  (b3.cash + house_value - initial_b.cash = -7000) := by
  sorry

end house_transaction_problem_l2836_283621


namespace reciprocal_of_four_l2836_283685

-- Define the reciprocal function
def reciprocal (x : ℚ) : ℚ := 1 / x

-- Theorem statement
theorem reciprocal_of_four : reciprocal 4 = 1 / 4 := by
  sorry

end reciprocal_of_four_l2836_283685


namespace first_knife_cost_is_five_l2836_283652

/-- The cost structure for knife sharpening -/
structure KnifeSharpening where
  first_knife_cost : ℝ
  next_three_cost : ℝ
  remaining_cost : ℝ
  total_knives : ℕ
  total_cost : ℝ

/-- The theorem stating the cost of sharpening the first knife -/
theorem first_knife_cost_is_five (ks : KnifeSharpening)
  (h1 : ks.next_three_cost = 4)
  (h2 : ks.remaining_cost = 3)
  (h3 : ks.total_knives = 9)
  (h4 : ks.total_cost = 32)
  (h5 : ks.total_cost = ks.first_knife_cost + 3 * ks.next_three_cost + 5 * ks.remaining_cost) :
  ks.first_knife_cost = 5 := by
  sorry

end first_knife_cost_is_five_l2836_283652


namespace composite_plus_four_prime_l2836_283649

/-- A number is composite if it has a factor between 1 and itself -/
def IsComposite (n : ℕ) : Prop :=
  ∃ m : ℕ, 1 < m ∧ m < n ∧ n % m = 0

/-- A number is prime if it's greater than 1 and its only factors are 1 and itself -/
def IsPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 0 ∧ m < n → n % m ≠ 0

theorem composite_plus_four_prime :
  ∃ n : ℕ, IsComposite n ∧ IsPrime (n + 4) :=
sorry

end composite_plus_four_prime_l2836_283649


namespace triangle_properties_l2836_283668

/-- Given a triangle ABC with the following properties:
    1. (1 - tan A)(1 - tan B) = 2
    2. b = 2√2
    3. c = 4
    Prove that:
    1. Angle C = π/4
    2. Area of triangle ABC = 2√3 + 2 -/
theorem triangle_properties (A B C : Real) (a b c : Real) :
  (1 - Real.tan A) * (1 - Real.tan B) = 2 →
  b = 2 * Real.sqrt 2 →
  c = 4 →
  C = π / 4 ∧
  (1/2) * b * c * Real.sin A = 2 * Real.sqrt 3 + 2 := by
  sorry

end triangle_properties_l2836_283668


namespace vector_equation_l2836_283609

def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (1, -1)
def c : ℝ × ℝ := (-2, 4)

theorem vector_equation : c = a - 3 • b := by sorry

end vector_equation_l2836_283609


namespace base_13_conversion_l2836_283660

/-- Represents a digit in base 13 -/
inductive Base13Digit
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | A | B | C

/-- Converts a Base13Digit to its decimal value -/
def toDecimal (d : Base13Digit) : Nat :=
  match d with
  | Base13Digit.D0 => 0
  | Base13Digit.D1 => 1
  | Base13Digit.D2 => 2
  | Base13Digit.D3 => 3
  | Base13Digit.D4 => 4
  | Base13Digit.D5 => 5
  | Base13Digit.D6 => 6
  | Base13Digit.D7 => 7
  | Base13Digit.D8 => 8
  | Base13Digit.D9 => 9
  | Base13Digit.A => 10
  | Base13Digit.B => 11
  | Base13Digit.C => 12

/-- Converts a three-digit base 13 number to its decimal equivalent -/
def base13ToDecimal (d1 d2 d3 : Base13Digit) : Nat :=
  (toDecimal d1) * 169 + (toDecimal d2) * 13 + (toDecimal d3)

theorem base_13_conversion :
  base13ToDecimal Base13Digit.D1 Base13Digit.D2 Base13Digit.D1 = 196 := by
  sorry

end base_13_conversion_l2836_283660


namespace gcd_8_factorial_10_factorial_l2836_283663

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem gcd_8_factorial_10_factorial : 
  Nat.gcd (factorial 8) (factorial 10) = factorial 8 := by sorry

end gcd_8_factorial_10_factorial_l2836_283663


namespace two_incorrect_statements_l2836_283651

/-- Represents the coefficients of a quadratic equation ax^2 + bx + c = 0 -/
structure QuadraticCoefficients where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents the roots of a quadratic equation -/
structure QuadraticRoots where
  x₁ : ℝ
  x₂ : ℝ

/-- Predicate to check if both roots are positive -/
def both_roots_positive (roots : QuadraticRoots) : Prop :=
  roots.x₁ > 0 ∧ roots.x₂ > 0

/-- Predicate to check if the given coefficients satisfy Vieta's formulas for the given roots -/
def satisfies_vieta (coeff : QuadraticCoefficients) (roots : QuadraticRoots) : Prop :=
  roots.x₁ + roots.x₂ = -coeff.b / coeff.a ∧ roots.x₁ * roots.x₂ = coeff.c / coeff.a

/-- The four statements about the signs of coefficients -/
def statement_1 (coeff : QuadraticCoefficients) : Prop := coeff.a > 0 ∧ coeff.b > 0 ∧ coeff.c > 0
def statement_2 (coeff : QuadraticCoefficients) : Prop := coeff.a < 0 ∧ coeff.b < 0 ∧ coeff.c < 0
def statement_3 (coeff : QuadraticCoefficients) : Prop := coeff.a > 0 ∧ coeff.b < 0 ∧ coeff.c < 0
def statement_4 (coeff : QuadraticCoefficients) : Prop := coeff.a < 0 ∧ coeff.b > 0 ∧ coeff.c > 0

/-- Main theorem: Exactly 2 out of 4 statements are incorrect for a quadratic equation with two positive roots -/
theorem two_incorrect_statements
  (coeff : QuadraticCoefficients)
  (roots : QuadraticRoots)
  (h_positive : both_roots_positive roots)
  (h_vieta : satisfies_vieta coeff roots) :
  (¬statement_1 coeff ∧ ¬statement_2 coeff ∧ statement_3 coeff ∧ statement_4 coeff) ∨
  (¬statement_1 coeff ∧ ¬statement_2 coeff ∧ statement_3 coeff ∧ ¬statement_4 coeff) ∨
  (¬statement_1 coeff ∧ ¬statement_2 coeff ∧ ¬statement_3 coeff ∧ statement_4 coeff) :=
by sorry

end two_incorrect_statements_l2836_283651


namespace berry_temperature_proof_l2836_283607

theorem berry_temperature_proof (temps : List Float) (avg : Float) : 
  temps = [99.1, 98.2, 98.7, 99.8, 99, 98.9] →
  avg = 99 →
  ∃ (wed_temp : Float), 
    wed_temp = 99.3 ∧ 
    (temps.sum + wed_temp) / 7 = avg :=
by sorry

end berry_temperature_proof_l2836_283607


namespace three_cones_apex_angle_l2836_283683

/-- Represents a cone with vertex at point A -/
structure Cone where
  apexAngle : ℝ

/-- Represents the configuration of three cones -/
structure ConeConfiguration where
  cone1 : Cone
  cone2 : Cone
  cone3 : Cone
  touchingPlane : Bool
  sameSide : Bool

/-- The theorem statement -/
theorem three_cones_apex_angle 
  (config : ConeConfiguration)
  (h1 : config.cone1 = config.cone2)
  (h2 : config.cone3.apexAngle = π / 2)
  (h3 : config.touchingPlane)
  (h4 : config.sameSide) :
  config.cone1.apexAngle = 2 * Real.arctan (4 / 5) := by
  sorry


end three_cones_apex_angle_l2836_283683


namespace range_of_a_l2836_283692

-- Define the set A
def A (a : ℝ) : Set ℝ := {x : ℝ | (x - a) / (x + a) < 0}

-- State the theorem
theorem range_of_a (a : ℝ) : 1 ∉ A a ↔ -1 ≤ a ∧ a ≤ 1 := by sorry

end range_of_a_l2836_283692


namespace messages_cleared_in_29_days_l2836_283686

/-- The number of days required to clear all unread messages -/
def days_to_clear_messages (initial_messages : ℕ) (read_per_day : ℕ) (new_per_day : ℕ) : ℕ :=
  (initial_messages + (read_per_day - new_per_day) - 1) / (read_per_day - new_per_day)

/-- Proof that it takes 29 days to clear all unread messages -/
theorem messages_cleared_in_29_days :
  days_to_clear_messages 198 15 8 = 29 := by
  sorry

#eval days_to_clear_messages 198 15 8

end messages_cleared_in_29_days_l2836_283686


namespace ball_radius_from_hole_dimensions_l2836_283655

/-- Given a spherical ball partially submerged in a frozen surface,
    where the hole left by the ball has a diameter of 30 cm and a depth of 8 cm,
    prove that the radius of the ball is 18.0625 cm. -/
theorem ball_radius_from_hole_dimensions (diameter : ℝ) (depth : ℝ) (radius : ℝ) :
  diameter = 30 →
  depth = 8 →
  radius = (((diameter / 2) ^ 2 + depth ^ 2) / (2 * depth)).sqrt →
  radius = 18.0625 := by
sorry

end ball_radius_from_hole_dimensions_l2836_283655


namespace factory_earnings_l2836_283605

/-- Represents a factory with machines producing material --/
structure Factory where
  original_machines : ℕ
  original_hours : ℕ
  new_machines : ℕ
  new_hours : ℕ
  production_rate : ℕ
  price_per_kg : ℕ

/-- Calculates the daily earnings of the factory --/
def daily_earnings (f : Factory) : ℕ :=
  ((f.original_machines * f.original_hours + f.new_machines * f.new_hours) * f.production_rate) * f.price_per_kg

/-- Theorem stating that the factory's daily earnings are $8100 --/
theorem factory_earnings :
  ∃ (f : Factory), 
    f.original_machines = 3 ∧
    f.original_hours = 23 ∧
    f.new_machines = 1 ∧
    f.new_hours = 12 ∧
    f.production_rate = 2 ∧
    f.price_per_kg = 50 ∧
    daily_earnings f = 8100 := by
  sorry


end factory_earnings_l2836_283605


namespace volume_submerged_object_iron_block_volume_l2836_283656

/-- The volume of a submerged object in a cylindrical container --/
theorem volume_submerged_object
  (r h₁ h₂ : ℝ)
  (hr : r > 0)
  (hh : h₂ > h₁) :
  let V := π * r^2 * (h₂ - h₁)
  V = π * r^2 * h₂ - π * r^2 * h₁ :=
by sorry

/-- The volume of the irregular iron block --/
theorem iron_block_volume
  (r h₁ h₂ : ℝ)
  (hr : r = 5)
  (hh₁ : h₁ = 6)
  (hh₂ : h₂ = 8) :
  π * r^2 * (h₂ - h₁) = 50 * π :=
by sorry

end volume_submerged_object_iron_block_volume_l2836_283656


namespace least_integer_satisfying_inequality_negative_two_satisfies_inequality_least_integer_is_negative_two_l2836_283670

theorem least_integer_satisfying_inequality :
  ∀ y : ℤ, (2 * y^2 + 2 * |y| + 7 < 25) → y ≥ -2 :=
by
  sorry

theorem negative_two_satisfies_inequality :
  2 * (-2)^2 + 2 * |-2| + 7 < 25 :=
by
  sorry

theorem least_integer_is_negative_two :
  ∃ x : ℤ, (2 * x^2 + 2 * |x| + 7 < 25) ∧ 
    (∀ y : ℤ, (2 * y^2 + 2 * |y| + 7 < 25) → y ≥ x) ∧
    x = -2 :=
by
  sorry

end least_integer_satisfying_inequality_negative_two_satisfies_inequality_least_integer_is_negative_two_l2836_283670


namespace arrangements_count_l2836_283644

/-- Represents the number of students -/
def total_students : ℕ := 6

/-- Represents the number of male students -/
def male_students : ℕ := 3

/-- Represents the number of female students -/
def female_students : ℕ := 3

/-- Represents that exactly two female students stand next to each other -/
def adjacent_female_students : ℕ := 2

/-- Calculates the number of arrangements satisfying the given conditions -/
def num_arrangements : ℕ := 288

/-- Theorem stating that the number of arrangements satisfying the given conditions is 288 -/
theorem arrangements_count :
  (total_students = male_students + female_students) →
  (male_students = 3) →
  (female_students = 3) →
  (adjacent_female_students = 2) →
  num_arrangements = 288 := by
  sorry

end arrangements_count_l2836_283644


namespace sphere_in_cube_intersection_l2836_283696

-- Define the cube and sphere
structure Cube where
  vertices : Fin 8 → ℝ × ℝ × ℝ

structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

-- Define the intersection of the sphere with a face
def intersectionRadius (s : Sphere) (face : List (ℝ × ℝ × ℝ)) : ℝ := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem sphere_in_cube_intersection (c : Cube) (s : Sphere) :
  s.radius = 10 →
  intersectionRadius s [c.vertices 0, c.vertices 1, c.vertices 4, c.vertices 5] = 1 →
  intersectionRadius s [c.vertices 4, c.vertices 5, c.vertices 6, c.vertices 7] = 1 →
  intersectionRadius s [c.vertices 2, c.vertices 3, c.vertices 6, c.vertices 7] = 3 →
  distance s.center (c.vertices 7) = 17 := by
  sorry

end sphere_in_cube_intersection_l2836_283696


namespace scientific_notation_35_million_l2836_283669

theorem scientific_notation_35_million :
  35000000 = 3.5 * (10 ^ 7) :=
by sorry

end scientific_notation_35_million_l2836_283669


namespace second_platform_length_l2836_283628

/-- Calculates the length of a second platform given train and first platform details -/
theorem second_platform_length
  (train_length : ℝ)
  (first_platform_length : ℝ)
  (first_crossing_time : ℝ)
  (second_crossing_time : ℝ)
  (h1 : train_length = 310)
  (h2 : first_platform_length = 110)
  (h3 : first_crossing_time = 15)
  (h4 : second_crossing_time = 20) :
  (second_crossing_time * (train_length + first_platform_length) / first_crossing_time) - train_length = 250 :=
by sorry

end second_platform_length_l2836_283628


namespace stratified_sample_size_l2836_283633

/-- Represents the total number of employees -/
def total_employees : ℕ := 750

/-- Represents the number of young employees -/
def young_employees : ℕ := 350

/-- Represents the number of middle-aged employees -/
def middle_aged_employees : ℕ := 250

/-- Represents the number of elderly employees -/
def elderly_employees : ℕ := 150

/-- Represents the number of young employees in the sample -/
def young_in_sample : ℕ := 7

/-- Theorem stating that the sample size is 15 given the conditions -/
theorem stratified_sample_size :
  (total_employees = young_employees + middle_aged_employees + elderly_employees) →
  (young_in_sample * total_employees = 15 * young_employees) :=
by
  sorry

end stratified_sample_size_l2836_283633


namespace vector_scalar_addition_l2836_283636

theorem vector_scalar_addition (v₁ v₂ : Fin 3 → ℝ) (c : ℝ) :
  v₁ = ![2, -3, 4] →
  v₂ = ![-4, 7, -1] →
  c = 3 →
  c • v₁ + v₂ = ![2, -2, 11] := by sorry

end vector_scalar_addition_l2836_283636


namespace transformation_result_l2836_283623

-- Define the original curve
def original_curve (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the transformation φ
def φ (x y : ℝ) : ℝ × ℝ := (3*x, 4*y)

-- Define the new curve
def new_curve (x y : ℝ) : Prop := (x^2 / 9) + (y^2 / 16) = 1

-- Theorem statement
theorem transformation_result :
  ∀ (x y : ℝ), original_curve x y → new_curve (φ x y).1 (φ x y).2 :=
by
  sorry

end transformation_result_l2836_283623


namespace another_divisor_of_increased_number_l2836_283658

theorem another_divisor_of_increased_number : ∃ (n : ℕ), n ≠ 12 ∧ n ≠ 30 ∧ n ≠ 74 ∧ n ≠ 100 ∧ (44402 + 2) % n = 0 :=
by
  -- The proof goes here
  sorry

end another_divisor_of_increased_number_l2836_283658


namespace negation_of_universal_proposition_l2836_283671

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 > 0) ↔ (∃ x : ℝ, x^2 ≤ 0) := by
  sorry

end negation_of_universal_proposition_l2836_283671


namespace greatest_three_digit_multiple_of_17_l2836_283626

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, n ≤ 999 ∧ n ≥ 100 ∧ 17 ∣ n → n ≤ 986 := by
  sorry

end greatest_three_digit_multiple_of_17_l2836_283626


namespace complete_square_formula_l2836_283627

theorem complete_square_formula (x : ℝ) : x^2 + 4*x + 4 = (x + 2)^2 := by
  sorry

end complete_square_formula_l2836_283627


namespace money_made_washing_cars_l2836_283608

def initial_amount : ℕ := 74
def current_amount : ℕ := 86

theorem money_made_washing_cars :
  current_amount - initial_amount = 12 :=
by sorry

end money_made_washing_cars_l2836_283608


namespace remaining_money_is_16000_l2836_283666

def salary : ℚ := 160000

def food_fraction : ℚ := 1/5
def rent_fraction : ℚ := 1/10
def clothes_fraction : ℚ := 3/5

def remaining_money : ℚ := salary - (food_fraction * salary + rent_fraction * salary + clothes_fraction * salary)

theorem remaining_money_is_16000 : remaining_money = 16000 := by
  sorry

end remaining_money_is_16000_l2836_283666


namespace special_collection_loans_l2836_283601

theorem special_collection_loans (initial_books final_books : ℕ) 
  (return_rate : ℚ) (loaned_books : ℕ) : 
  initial_books = 75 → 
  final_books = 57 → 
  return_rate = 7/10 →
  initial_books - final_books = (1 - return_rate) * loaned_books →
  loaned_books = 60 := by
  sorry

end special_collection_loans_l2836_283601


namespace rows_per_wall_is_fifty_l2836_283687

/-- The number of bricks in a single row of each wall -/
def bricks_per_row : ℕ := 30

/-- The total number of bricks used for both walls -/
def total_bricks : ℕ := 3000

/-- The number of rows in each wall -/
def rows_per_wall : ℕ := total_bricks / (2 * bricks_per_row)

theorem rows_per_wall_is_fifty : rows_per_wall = 50 := by
  sorry

end rows_per_wall_is_fifty_l2836_283687


namespace binomial_expansion_sum_l2836_283606

theorem binomial_expansion_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (1 - 2*x)^5 = a₀ + a₁*(1+x) + a₂*(1+x)^2 + a₃*(1+x)^3 + a₄*(1+x)^4 + a₅*(1+x)^5) →
  a₃ + a₄ = -480 := by
sorry

end binomial_expansion_sum_l2836_283606


namespace compound_interest_problem_l2836_283677

/-- Compound interest calculation --/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * ((1 + rate) ^ time - 1)

/-- Theorem statement --/
theorem compound_interest_problem :
  let principal : ℝ := 500
  let rate : ℝ := 0.05
  let time : ℕ := 5
  let interest : ℝ := compound_interest principal rate time
  ∃ ε > 0, |interest - 138.14| < ε :=
by sorry

end compound_interest_problem_l2836_283677


namespace mart_income_percentage_of_juan_l2836_283602

/-- Represents the income relationships between Tim, Mart, Juan, and Alex -/
structure IncomeRelationships where
  tim : ℝ
  mart : ℝ
  juan : ℝ
  alex : ℝ
  mart_tim_ratio : mart = 1.6 * tim
  tim_juan_ratio : tim = 0.6 * juan
  alex_mart_ratio : alex = 1.25 * mart
  juan_alex_ratio : juan = 1.2 * alex

/-- Theorem stating that Mart's income is 96% of Juan's income -/
theorem mart_income_percentage_of_juan (ir : IncomeRelationships) :
  ir.mart = 0.96 * ir.juan := by
  sorry

end mart_income_percentage_of_juan_l2836_283602


namespace polynomial_division_theorem_l2836_283688

theorem polynomial_division_theorem :
  let f (x : ℝ) := x^4 - 8*x^3 + 18*x^2 - 22*x + 8
  let g (x : ℝ) := x^2 - 3*x + k
  let r (x : ℝ) := x + a
  ∀ (k a : ℝ),
  (∃ q : ℝ → ℝ, ∀ x, f x = g x * q x + r x) →
  (k = 8/3 ∧ a = 64/9) :=
by sorry

end polynomial_division_theorem_l2836_283688


namespace unique_triple_gcd_sum_square_l2836_283678

theorem unique_triple_gcd_sum_square : 
  ∃! (m n l : ℕ), 
    m + n = (Nat.gcd m n)^2 ∧
    m + l = (Nat.gcd m l)^2 ∧
    n + l = (Nat.gcd n l)^2 ∧
    m = 2 ∧ n = 2 ∧ l = 2 := by
  sorry

end unique_triple_gcd_sum_square_l2836_283678


namespace distance_to_gym_l2836_283673

theorem distance_to_gym (home_to_grocery : ℝ) (grocery_to_gym_speed : ℝ) 
  (time_difference : ℝ) :
  home_to_grocery = 200 →
  grocery_to_gym_speed = 2 →
  time_difference = 50 →
  grocery_to_gym_speed = 2 * (home_to_grocery / 200) →
  (200 / (home_to_grocery / 200)) - (200 / grocery_to_gym_speed) = time_difference →
  200 / grocery_to_gym_speed = 300 := by
  sorry

end distance_to_gym_l2836_283673


namespace philip_monthly_mileage_l2836_283619

/-- Represents Philip's driving routine and calculates the total monthly mileage -/
def philipDrivingMileage : ℕ :=
  let schoolRoundTrip : ℕ := 5 /- 2.5 * 2, rounded to nearest integer -/
  let workOneWay : ℕ := 8
  let marketRoundTrip : ℕ := 2
  let gymRoundTrip : ℕ := 4
  let friendRoundTrip : ℕ := 6
  let weekdayMileage : ℕ := (schoolRoundTrip * 2 + workOneWay * 2) * 5
  let saturdayMileage : ℕ := marketRoundTrip + gymRoundTrip + friendRoundTrip
  let weeklyMileage : ℕ := weekdayMileage + saturdayMileage
  let weeksInMonth : ℕ := 4
  weeklyMileage * weeksInMonth

/-- Theorem stating that Philip's total monthly mileage is 468 miles -/
theorem philip_monthly_mileage : philipDrivingMileage = 468 := by
  sorry

end philip_monthly_mileage_l2836_283619


namespace camping_matches_l2836_283697

def matches_left (initial : ℕ) (dropped : ℕ) : ℕ :=
  initial - dropped - 2 * dropped

theorem camping_matches (initial : ℕ) (dropped : ℕ) 
  (h1 : initial ≥ dropped) 
  (h2 : initial ≥ dropped + 2 * dropped) :
  matches_left initial dropped = initial - dropped - 2 * dropped :=
by sorry

end camping_matches_l2836_283697


namespace mod_inverse_two_mod_221_l2836_283604

theorem mod_inverse_two_mod_221 : ∃ x : ℕ, x < 221 ∧ (2 * x) % 221 = 1 :=
by
  use 111
  sorry

end mod_inverse_two_mod_221_l2836_283604


namespace sqrt_720_simplification_l2836_283689

theorem sqrt_720_simplification : Real.sqrt 720 = 6 * Real.sqrt 5 := by
  sorry

end sqrt_720_simplification_l2836_283689


namespace special_triangle_side_length_l2836_283648

/-- An equilateral triangle with a special interior point -/
structure SpecialTriangle where
  -- The side length of the equilateral triangle
  s : ℝ
  -- The coordinates of the special point P
  p : ℝ × ℝ
  -- Condition that the triangle is equilateral
  equilateral : s > 0
  -- Condition that P is inside the triangle
  p_inside : p.1 > 0 ∧ p.2 > 0 ∧ p.1 + p.2 < s
  -- Conditions for distances from P to vertices
  dist_ap : Real.sqrt ((0 - p.1)^2 + (0 - p.2)^2) = 1
  dist_bp : Real.sqrt ((s - p.1)^2 + (0 - p.2)^2) = Real.sqrt 3
  dist_cp : Real.sqrt ((s/2 - p.1)^2 + (s*Real.sqrt 3/2 - p.2)^2) = 2

theorem special_triangle_side_length (t : SpecialTriangle) : t.s = Real.sqrt 7 := by
  sorry

end special_triangle_side_length_l2836_283648


namespace largest_n_with_1992_divisors_and_phi_divides_l2836_283624

/-- The number of positive divisors of n -/
def divisor_count (n : ℕ) : ℕ := sorry

/-- Euler's totient function -/
def phi (n : ℕ) : ℕ := sorry

theorem largest_n_with_1992_divisors_and_phi_divides (n : ℕ) :
  (phi n ∣ n) →
  (divisor_count n = 1992) →
  n ≤ 2^1991 :=
sorry

end largest_n_with_1992_divisors_and_phi_divides_l2836_283624


namespace probability_inequalities_l2836_283631

open ProbabilityTheory MeasureTheory

variable {Ω : Type*} [MeasurableSpace Ω] [Fintype Ω]
variable (P : Measure Ω) [IsProbabilityMeasure P]
variable (A B : Set Ω)

theorem probability_inequalities
  (h1 : P A = P (Aᶜ))
  (h2 : P (Bᶜ ∩ A) / P A > P (B ∩ Aᶜ) / P Aᶜ) :
  (P (A ∩ Bᶜ) > P (Aᶜ ∩ B)) ∧ (P (A ∩ B) < P (Aᶜ ∩ Bᶜ)) := by
  sorry

end probability_inequalities_l2836_283631


namespace quadratic_equation_solution_l2836_283635

theorem quadratic_equation_solution :
  {x : ℝ | x^2 = -2*x} = {0, -2} := by sorry

end quadratic_equation_solution_l2836_283635


namespace jennifer_remaining_money_l2836_283625

def initial_amount : ℚ := 150.75

def sandwich_fraction : ℚ := 3/10
def museum_fraction : ℚ := 1/4
def book_fraction : ℚ := 1/8
def coffee_percentage : ℚ := 2.5/100

def remaining_amount : ℚ := initial_amount - (
  initial_amount * sandwich_fraction +
  initial_amount * museum_fraction +
  initial_amount * book_fraction +
  initial_amount * coffee_percentage
)

theorem jennifer_remaining_money :
  remaining_amount = 45.225 := by sorry

end jennifer_remaining_money_l2836_283625


namespace honda_cars_sold_l2836_283610

/-- Represents the total number of cars sold -/
def total_cars : ℕ := 300

/-- Represents the percentage of Audi cars sold -/
def audi_percent : ℚ := 10 / 100

/-- Represents the percentage of Toyota cars sold -/
def toyota_percent : ℚ := 20 / 100

/-- Represents the percentage of Acura cars sold -/
def acura_percent : ℚ := 30 / 100

/-- Represents the percentage of BMW cars sold -/
def bmw_percent : ℚ := 15 / 100

/-- Theorem stating that the number of Honda cars sold is 75 -/
theorem honda_cars_sold : 
  (total_cars : ℚ) * (1 - (audi_percent + toyota_percent + acura_percent + bmw_percent)) = 75 := by
  sorry

end honda_cars_sold_l2836_283610


namespace unique_solution_l2836_283654

def base7_to_decimal (a b : Nat) : Nat := 7 * a + b

theorem unique_solution :
  ∀ P Q R : Nat,
    P ≠ 0 ∧ Q ≠ 0 ∧ R ≠ 0 →
    P < 7 ∧ Q < 7 ∧ R < 7 →
    P ≠ Q ∧ P ≠ R ∧ Q ≠ R →
    base7_to_decimal P Q + R = base7_to_decimal R 0 →
    base7_to_decimal P Q + base7_to_decimal Q P = base7_to_decimal R R →
    P = 4 ∧ Q = 3 ∧ R = 4 := by
  sorry

end unique_solution_l2836_283654


namespace opposite_of_negative_eight_l2836_283616

theorem opposite_of_negative_eight :
  (∃ x : ℤ, -8 + x = 0) ∧ (∀ y : ℤ, -8 + y = 0 → y = 8) :=
by sorry

end opposite_of_negative_eight_l2836_283616


namespace divisible_by_three_after_rotation_l2836_283680

theorem divisible_by_three_after_rotation (n : ℕ) : 
  n = 857142 → 
  (n % 3 = 0) ∧ 
  ((285714 : ℕ) % 3 = 0) := by
sorry

end divisible_by_three_after_rotation_l2836_283680


namespace hot_dog_stand_sales_l2836_283698

/-- A hot dog stand problem -/
theorem hot_dog_stand_sales 
  (price : ℝ) 
  (hours : ℝ) 
  (total_sales : ℝ) 
  (h1 : price = 2)
  (h2 : hours = 10)
  (h3 : total_sales = 200) :
  total_sales / (hours * price) = 10 :=
sorry

end hot_dog_stand_sales_l2836_283698


namespace hyperbola_asymptotes_l2836_283674

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 3 = 1

-- Define the asymptotes
def asymptotes (x y : ℝ) : Prop := Real.sqrt 3 * x = 2 * y ∨ Real.sqrt 3 * x = -2 * y

-- Theorem statement
theorem hyperbola_asymptotes : 
  ∀ x y : ℝ, hyperbola x y → asymptotes x y :=
sorry

end hyperbola_asymptotes_l2836_283674


namespace letter_digit_problem_l2836_283653

/-- Represents a mapping from letters to digits -/
def LetterDigitMap := Char → Nat

/-- Checks if a LetterDigitMap is valid according to the problem conditions -/
def is_valid_map (f : LetterDigitMap) : Prop :=
  (f 'E' ≠ f 'H') ∧ (f 'E' ≠ f 'M') ∧ (f 'E' ≠ f 'O') ∧ (f 'E' ≠ f 'P') ∧
  (f 'H' ≠ f 'M') ∧ (f 'H' ≠ f 'O') ∧ (f 'H' ≠ f 'P') ∧
  (f 'M' ≠ f 'O') ∧ (f 'M' ≠ f 'P') ∧
  (f 'O' ≠ f 'P') ∧
  (∀ c, c ∈ ['E', 'H', 'M', 'O', 'P'] → f c ∈ [1, 2, 3, 4, 6, 8, 9])

theorem letter_digit_problem (f : LetterDigitMap) 
  (h1 : is_valid_map f)
  (h2 : f 'E' * f 'H' = f 'M' * f 'O' * f 'P' * f 'O' * 3)
  (h3 : f 'E' + f 'H' = f 'M' + f 'O' + f 'P' + f 'O' + 3) :
  f 'E' * f 'H' + f 'M' * f 'O' * f 'P' * f 'O' * 3 = 72 := by
  sorry

end letter_digit_problem_l2836_283653


namespace jasmine_milk_purchase_jasmine_milk_purchase_holds_l2836_283620

/-- Proves that Jasmine bought 2 gallons of milk given the problem conditions -/
theorem jasmine_milk_purchase : ℝ → Prop :=
  fun gallons_of_milk =>
    let coffee_pounds : ℝ := 4
    let coffee_price_per_pound : ℝ := 2.5
    let milk_price_per_gallon : ℝ := 3.5
    let total_cost : ℝ := 17
    coffee_pounds * coffee_price_per_pound + gallons_of_milk * milk_price_per_gallon = total_cost →
    gallons_of_milk = 2

/-- The theorem holds -/
theorem jasmine_milk_purchase_holds : jasmine_milk_purchase 2 := by
  sorry

end jasmine_milk_purchase_jasmine_milk_purchase_holds_l2836_283620


namespace factorization_equality_l2836_283691

theorem factorization_equality (a b : ℝ) : 2 * a^2 - 8 * b^2 = 2 * (a + 2*b) * (a - 2*b) := by
  sorry

end factorization_equality_l2836_283691


namespace product_equation_l2836_283634

theorem product_equation : 935420 * 625 = 584638125 := by
  sorry

end product_equation_l2836_283634


namespace prob_different_suits_is_78_103_l2836_283630

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Finset (Nat × Nat))
  (size : cards.card = 52)

/-- Represents two mixed standard 52-card decks -/
def MixedDecks := Deck × Deck

/-- The probability of picking two different cards of different suits from mixed decks -/
def prob_different_suits (decks : MixedDecks) : ℚ :=
  78 / 103

/-- Theorem stating the probability of picking two different cards of different suits -/
theorem prob_different_suits_is_78_103 (decks : MixedDecks) :
  prob_different_suits decks = 78 / 103 := by
  sorry

end prob_different_suits_is_78_103_l2836_283630


namespace product_of_sum_of_squares_l2836_283617

theorem product_of_sum_of_squares (p q r s : ℤ) :
  ∃ (x y : ℤ), (p^2 + q^2) * (r^2 + s^2) = x^2 + y^2 := by
  sorry

end product_of_sum_of_squares_l2836_283617


namespace coat_price_problem_l2836_283612

theorem coat_price_problem (price_reduction : ℝ) (percentage_reduction : ℝ) :
  price_reduction = 200 →
  percentage_reduction = 0.40 →
  ∃ original_price : ℝ, 
    original_price * percentage_reduction = price_reduction ∧
    original_price = 500 := by
  sorry

end coat_price_problem_l2836_283612


namespace sequence_consecutive_product_l2836_283603

/-- The nth term of the sequence, represented as n 1's followed by n 2's -/
def sequence_term (n : ℕ) : ℕ := 
  (10^n - 1) * (10^n + 2)

/-- The first factor of the product -/
def factor1 (n : ℕ) : ℕ := 
  (10^n - 1) / 3

/-- The second factor of the product -/
def factor2 (n : ℕ) : ℕ := 
  (10^n + 2) / 3

theorem sequence_consecutive_product (n : ℕ) : 
  sequence_term n = factor1 n * factor2 n ∧ factor2 n = factor1 n + 1 :=
sorry

end sequence_consecutive_product_l2836_283603


namespace two_self_inverse_matrices_l2836_283679

def is_self_inverse (a d : ℝ) : Prop :=
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![a, 3; -8, d]
  M * M = 1

theorem two_self_inverse_matrices :
  ∃! (n : ℕ), ∃ (S : Finset (ℝ × ℝ)),
    S.card = n ∧
    (∀ (p : ℝ × ℝ), p ∈ S ↔ is_self_inverse p.1 p.2) :=
  sorry

end two_self_inverse_matrices_l2836_283679


namespace efficiency_comparison_l2836_283657

theorem efficiency_comparison (p q : ℝ) (work : ℝ) 
  (h_p_time : p * 25 = work)
  (h_combined_time : (p + q) * 15 = work)
  (h_p_more_efficient : p > q) :
  (p - q) / q = 1/2 := by
sorry

end efficiency_comparison_l2836_283657


namespace tan_2a_values_l2836_283600

theorem tan_2a_values (a : ℝ) (h : 2 * Real.sin (2 * a) = 1 + Real.cos (2 * a)) :
  Real.tan (2 * a) = 4 / 3 ∨ Real.tan (2 * a) = 0 := by
  sorry

end tan_2a_values_l2836_283600


namespace scout_saturday_customers_l2836_283639

/-- Scout's delivery earnings over a weekend --/
def scout_earnings (base_pay hourly_rate tip_rate : ℚ) 
                   (saturday_hours sunday_hours : ℚ) 
                   (sunday_customers : ℕ) 
                   (total_earnings : ℚ) : Prop :=
  let saturday_base := base_pay * saturday_hours
  let sunday_base := base_pay * sunday_hours
  let sunday_tips := tip_rate * sunday_customers
  ∃ saturday_customers : ℕ,
    saturday_base + sunday_base + sunday_tips + (tip_rate * saturday_customers) = total_earnings

theorem scout_saturday_customers :
  scout_earnings 10 10 5 4 5 8 155 →
  ∃ saturday_customers : ℕ, saturday_customers = 5 :=
by sorry

end scout_saturday_customers_l2836_283639


namespace log_relation_l2836_283615

theorem log_relation (a b : ℝ) (ha : a = Real.log 128 / Real.log 4) (hb : b = Real.log 16 / Real.log 2) :
  a = (7 * b) / 8 := by sorry

end log_relation_l2836_283615


namespace least_positive_integer_with_congruences_l2836_283645

theorem least_positive_integer_with_congruences : ∃ b : ℕ+, 
  (b : ℤ) ≡ 2 [ZMOD 3] ∧ 
  (b : ℤ) ≡ 3 [ZMOD 4] ∧ 
  (b : ℤ) ≡ 4 [ZMOD 5] ∧ 
  (b : ℤ) ≡ 6 [ZMOD 7] ∧ 
  ∀ c : ℕ+, 
    ((c : ℤ) ≡ 2 [ZMOD 3] ∧ 
     (c : ℤ) ≡ 3 [ZMOD 4] ∧ 
     (c : ℤ) ≡ 4 [ZMOD 5] ∧ 
     (c : ℤ) ≡ 6 [ZMOD 7]) → 
    b ≤ c :=
by sorry

end least_positive_integer_with_congruences_l2836_283645


namespace parallel_vectors_l2836_283661

/-- Given two 2D vectors a and b, find the value of k such that 
    (2a + b) is parallel to (1/2a + kb) -/
theorem parallel_vectors (a b : ℝ × ℝ) (h1 : a = (2, 1)) (h2 : b = (1, 2)) :
  ∃ k : ℝ, k = (1 : ℝ) / 4 ∧ 
  ∃ c : ℝ, c ≠ 0 ∧ c • (2 • a + b) = (1 / 2 : ℝ) • a + k • b :=
by sorry

end parallel_vectors_l2836_283661


namespace some_mythical_creatures_are_winged_animals_l2836_283622

-- Define the sets
variable (D : Type) -- Dragons
variable (M : Type) -- Mythical creatures
variable (W : Type) -- Winged animals

-- Define the relations
variable (isDragon : D → Prop)
variable (isMythical : M → Prop)
variable (isWinged : W → Prop)

-- Define the conditions
variable (h1 : ∀ d : D, ∃ m : M, isMythical m)
variable (h2 : ∃ w : W, ∃ d : D, isDragon d ∧ isWinged w)

-- Theorem to prove
theorem some_mythical_creatures_are_winged_animals :
  ∃ m : M, ∃ w : W, isMythical m ∧ isWinged w :=
sorry

end some_mythical_creatures_are_winged_animals_l2836_283622


namespace sum_of_digits_of_large_number_l2836_283699

/-- The sum of the digits of 10^91 + 100 is 2 -/
theorem sum_of_digits_of_large_number : ∃ (n : ℕ), n = 10^91 + 100 ∧ (n.digits 10).sum = 2 := by
  sorry

end sum_of_digits_of_large_number_l2836_283699


namespace lcm_15_18_20_l2836_283646

theorem lcm_15_18_20 : Nat.lcm (Nat.lcm 15 18) 20 = 180 := by
  sorry

end lcm_15_18_20_l2836_283646


namespace sqrt_sum_squares_zero_implies_zero_l2836_283694

theorem sqrt_sum_squares_zero_implies_zero (a b : ℂ) : 
  Real.sqrt (Complex.abs a ^ 2 + Complex.abs b ^ 2) = 0 → a = 0 ∧ b = 0 := by
  sorry

end sqrt_sum_squares_zero_implies_zero_l2836_283694


namespace parallelogram_area_l2836_283681

/-- The area of a parallelogram with base 48 cm and height 36 cm is 1728 square centimeters. -/
theorem parallelogram_area : 
  ∀ (base height area : ℝ), 
  base = 48 → 
  height = 36 → 
  area = base * height → 
  area = 1728 :=
by sorry

end parallelogram_area_l2836_283681


namespace set_equality_l2836_283638

def I : Set Char := {'a', 'b', 'c', 'd', 'e', 'f', 'g'}
def M : Set Char := {'c', 'd', 'e'}
def N : Set Char := {'a', 'c', 'f'}

theorem set_equality : (I \ M) ∩ (I \ N) = {'b', 'g'} := by sorry

end set_equality_l2836_283638


namespace eugene_sunday_swim_time_l2836_283614

/-- Represents the swim times for Eugene over three days -/
structure SwimTimes where
  sunday : ℕ
  monday : ℕ
  tuesday : ℕ

/-- Calculates the average swim time over three days -/
def averageSwimTime (times : SwimTimes) : ℚ :=
  (times.sunday + times.monday + times.tuesday : ℚ) / 3

theorem eugene_sunday_swim_time :
  ∃ (times : SwimTimes),
    times.monday = 30 ∧
    times.tuesday = 45 ∧
    averageSwimTime times = 34 ∧
    times.sunday = 27 :=
  sorry

end eugene_sunday_swim_time_l2836_283614


namespace exam_score_per_correct_answer_l2836_283613

/-- Proves the number of marks scored for each correct answer in an exam -/
theorem exam_score_per_correct_answer 
  (total_questions : ℕ) 
  (total_marks : ℤ) 
  (correct_answers : ℕ) 
  (wrong_penalty : ℤ) 
  (h1 : total_questions = 60)
  (h2 : total_marks = 110)
  (h3 : correct_answers = 34)
  (h4 : wrong_penalty = -1)
  (h5 : total_questions = correct_answers + (total_questions - correct_answers)) :
  ∃ (score_per_correct : ℤ), 
    score_per_correct * correct_answers + wrong_penalty * (total_questions - correct_answers) = total_marks ∧ 
    score_per_correct = 4 := by
  sorry

end exam_score_per_correct_answer_l2836_283613


namespace square_root_calculation_l2836_283665

theorem square_root_calculation : (Real.sqrt 2 + 1)^2 - Real.sqrt (9/2) = 3 + Real.sqrt 2 / 2 := by
  sorry

end square_root_calculation_l2836_283665


namespace f_has_one_root_l2836_283662

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 3*x - 2

theorem f_has_one_root : ∃! x : ℝ, f x = 0 := by
  sorry

end f_has_one_root_l2836_283662


namespace no_consecutive_triples_sum_squares_equal_repeating_digit_l2836_283647

theorem no_consecutive_triples_sum_squares_equal_repeating_digit : 
  ¬ ∃ (n a : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ (n-1)^2 + n^2 + (n+1)^2 = 1111 * a := by
  sorry

end no_consecutive_triples_sum_squares_equal_repeating_digit_l2836_283647


namespace tenth_power_sum_of_roots_l2836_283664

theorem tenth_power_sum_of_roots (u v : ℝ) : 
  u^2 - 2*u*Real.sqrt 3 + 1 = 0 ∧ 
  v^2 - 2*v*Real.sqrt 3 + 1 = 0 → 
  u^10 + v^10 = 93884 := by
  sorry

end tenth_power_sum_of_roots_l2836_283664


namespace problem_solution_l2836_283659

theorem problem_solution : ∃ x : ℚ, x + (1/4 * x) = 90 - (30/100 * 90) ∧ x = 50 := by
  sorry

end problem_solution_l2836_283659


namespace tangent_line_perpendicular_l2836_283690

/-- Given a > 0 and f(x) = x³ + ax² - 9x - 1, if the tangent line with the smallest slope
    on the curve y = f(x) is perpendicular to the line x - 12y = 0, then a = 3 -/
theorem tangent_line_perpendicular (a : ℝ) (h1 : a > 0) : 
  let f : ℝ → ℝ := λ x ↦ x^3 + a*x^2 - 9*x - 1
  (∃ x₀ : ℝ, ∀ x : ℝ, (deriv f x₀ ≤ deriv f x) ∧ 
    (deriv f x₀ * (1 / 12) = -1)) → a = 3 := by
  sorry

end tangent_line_perpendicular_l2836_283690


namespace trisomicCrossRatio_l2836_283693

-- Define the basic types
inductive Genotype
| BB
| Bb
| bb
| Bbb
| bbb

inductive Gamete
| B
| b
| Bb
| bb

-- Define the meiosis process for trisomic cells
def trisomicMeiosis (g : Genotype) : List Gamete := sorry

-- Define the fertilization process
def fertilize (female : Gamete) (male : Gamete) : Option Genotype := sorry

-- Define the phenotype (disease resistance) based on genotype
def isResistant (g : Genotype) : Bool := sorry

-- Define the cross between two plants
def cross (female : Genotype) (male : Genotype) : List Genotype := sorry

-- Define the ratio calculation function
def ratioResistantToSusceptible (offspring : List Genotype) : Rat := sorry

-- Theorem statement
theorem trisomicCrossRatio :
  let femaleParent : Genotype := Genotype.bbb
  let maleParent : Genotype := Genotype.BB
  let f1 : List Genotype := cross femaleParent maleParent
  let f1Trisomic : Genotype := Genotype.Bbb
  let susceptibleNormal : Genotype := Genotype.bb
  let f2 : List Genotype := cross f1Trisomic susceptibleNormal
  ratioResistantToSusceptible f2 = 1 / 2 := by sorry

end trisomicCrossRatio_l2836_283693


namespace selection_problem_l2836_283641

theorem selection_problem (n_boys : ℕ) (n_girls : ℕ) : 
  n_boys = 4 → n_girls = 3 → 
  (Nat.choose n_boys 2 * Nat.choose n_girls 1) + 
  (Nat.choose n_girls 2 * Nat.choose n_boys 1) = 30 := by
sorry

end selection_problem_l2836_283641
