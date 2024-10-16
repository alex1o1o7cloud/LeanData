import Mathlib

namespace NUMINAMATH_CALUDE_least_value_theorem_l4010_401020

theorem least_value_theorem (x y z w : ℕ+) 
  (h : (5 : ℕ) * w.val = (3 : ℕ) * x.val ∧ 
       (3 : ℕ) * x.val = (4 : ℕ) * y.val ∧ 
       (4 : ℕ) * y.val = (7 : ℕ) * z.val) : 
  (∀ a b c d : ℕ+, 
    ((5 : ℕ) * d.val = (3 : ℕ) * a.val ∧ 
     (3 : ℕ) * a.val = (4 : ℕ) * b.val ∧ 
     (4 : ℕ) * b.val = (7 : ℕ) * c.val) → 
    (x.val - y.val + z.val - w.val : ℤ) ≤ (a.val - b.val + c.val - d.val : ℤ)) ∧
  (x.val - y.val + z.val - w.val : ℤ) = 11 := by
sorry

end NUMINAMATH_CALUDE_least_value_theorem_l4010_401020


namespace NUMINAMATH_CALUDE_coin_toss_probability_l4010_401053

theorem coin_toss_probability : 
  let n : ℕ := 5
  let p_tail : ℚ := 1 / 2
  let p_all_tails : ℚ := p_tail ^ n
  let p_at_least_one_head : ℚ := 1 - p_all_tails
  p_at_least_one_head = 31 / 32 := by
sorry

end NUMINAMATH_CALUDE_coin_toss_probability_l4010_401053


namespace NUMINAMATH_CALUDE_reflected_line_tangent_to_circle_l4010_401041

/-- The line reflected off the x-axis that is tangent to a given circle -/
theorem reflected_line_tangent_to_circle 
  (A : ℝ × ℝ) 
  (circle : ℝ × ℝ → Prop) 
  (l : ℝ × ℝ → Prop) : 
  A = (-3, 3) →
  (∀ x y, circle (x, y) ↔ x^2 + y^2 - 4*x - 4*y + 7 = 0) →
  (∀ x y, l (x, y) ↔ (4*x + 3*y + 3 = 0 ∨ 3*x + 4*y - 3 = 0)) →
  (∃ P : ℝ × ℝ, P.2 = 0 ∧ l A ∧ l P) →
  (∃ Q : ℝ × ℝ, circle Q ∧ l Q ∧ 
    ∀ R : ℝ × ℝ, R ≠ Q → circle R → ¬(l R)) := by
sorry

end NUMINAMATH_CALUDE_reflected_line_tangent_to_circle_l4010_401041


namespace NUMINAMATH_CALUDE_polynomial_parity_l4010_401015

/-- Represents a polynomial with integer coefficients -/
def IntPolynomial := List Int

/-- Multiplies two polynomials -/
def polyMul (p q : IntPolynomial) : IntPolynomial := sorry

/-- Checks if all coefficients of a polynomial are even -/
def allCoeffsEven (p : IntPolynomial) : Prop := sorry

/-- Checks if all coefficients of a polynomial are divisible by 4 -/
def allCoeffsDivBy4 (p : IntPolynomial) : Prop := sorry

/-- Checks if a polynomial has at least one odd coefficient -/
def hasOddCoeff (p : IntPolynomial) : Prop := sorry

theorem polynomial_parity (p q : IntPolynomial) :
  (allCoeffsEven (polyMul p q)) ∧ ¬(allCoeffsDivBy4 (polyMul p q)) →
  (allCoeffsEven p ∧ hasOddCoeff q) ∨ (allCoeffsEven q ∧ hasOddCoeff p) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_parity_l4010_401015


namespace NUMINAMATH_CALUDE_correct_sampling_methods_l4010_401027

-- Define the type for sampling methods
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic
  | DrawingLots
  | RandomNumber

-- Define the set of correct sampling methods
def correctSamplingMethods : Set SamplingMethod :=
  {SamplingMethod.SimpleRandom, SamplingMethod.Stratified, SamplingMethod.Systematic}

-- Define the property of being a valid sampling method
def isValidSamplingMethod (method : SamplingMethod) : Prop :=
  method ∈ correctSamplingMethods

-- State the conditions
axiom simple_random_valid : isValidSamplingMethod SamplingMethod.SimpleRandom
axiom stratified_valid : isValidSamplingMethod SamplingMethod.Stratified
axiom systematic_valid : isValidSamplingMethod SamplingMethod.Systematic
axiom drawing_lots_is_simple_random : SamplingMethod.DrawingLots = SamplingMethod.SimpleRandom
axiom random_number_is_simple_random : SamplingMethod.RandomNumber = SamplingMethod.SimpleRandom

-- State the theorem
theorem correct_sampling_methods :
  correctSamplingMethods = {SamplingMethod.SimpleRandom, SamplingMethod.Stratified, SamplingMethod.Systematic} :=
by sorry

end NUMINAMATH_CALUDE_correct_sampling_methods_l4010_401027


namespace NUMINAMATH_CALUDE_sequence_properties_l4010_401028

def geometric_sequence (a : ℕ → ℕ) : Prop :=
  ∃ q : ℕ, ∀ n : ℕ, a (n + 1) = a n * q

theorem sequence_properties (a b : ℕ → ℕ) (k : ℝ) :
  geometric_sequence a →
  (a 1 = 3) →
  (2 * a 3 = a 2 + (3/4) * a 4) →
  (b 1 = 1) →
  (∀ n : ℕ, b (n + 1) = 2 * b n + 1) →
  (∀ n : ℕ, k * ((b n + 5) / 2) - a n ≥ 8 * n + 2 * k - 24) →
  (∀ n : ℕ, a n = 3 * 2^(n - 1)) ∧
  (∀ n : ℕ, b n = 2^n - 1) ∧
  (k ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l4010_401028


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l4010_401071

theorem fraction_equals_zero (x : ℝ) (h : x ≠ 0) :
  (x - 5) / (6 * x) = 0 ↔ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l4010_401071


namespace NUMINAMATH_CALUDE_smallest_prime_perimeter_scalene_triangle_l4010_401039

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that checks if three numbers are consecutive primes -/
def areConsecutivePrimes (a b c : ℕ) : Prop := sorry

/-- A function that checks if three side lengths can form a triangle -/
def canFormTriangle (a b c : ℕ) : Prop := sorry

/-- The smallest perimeter of a scalene triangle with consecutive prime side lengths and a prime perimeter -/
theorem smallest_prime_perimeter_scalene_triangle :
  ∃ (a b c : ℕ),
    a < b ∧ b < c ∧
    areConsecutivePrimes a b c ∧
    canFormTriangle a b c ∧
    isPrime (a + b + c) ∧
    (a + b + c = 23) ∧
    (∀ (x y z : ℕ),
      x < y ∧ y < z ∧
      areConsecutivePrimes x y z ∧
      canFormTriangle x y z ∧
      isPrime (x + y + z) →
      (x + y + z ≥ 23)) :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_perimeter_scalene_triangle_l4010_401039


namespace NUMINAMATH_CALUDE_geometric_sequence_incorrect_statement_l4010_401080

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_incorrect_statement
  (a : ℕ → ℝ) (q : ℝ) (h1 : geometric_sequence a q) (h2 : q ≠ 1) :
  ¬(a 2 > a 1 → ∀ n : ℕ, a (n + 1) > a n) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_incorrect_statement_l4010_401080


namespace NUMINAMATH_CALUDE_homework_problems_per_page_l4010_401097

theorem homework_problems_per_page 
  (total_problems : ℕ) 
  (solved_percentage : ℚ) 
  (remaining_pages : ℕ) 
  (h1 : total_problems = 550) 
  (h2 : solved_percentage = 65 / 100) 
  (h3 : remaining_pages = 3) : 
  (total_problems - Int.floor (solved_percentage * total_problems)) / remaining_pages = 64 := by
  sorry

end NUMINAMATH_CALUDE_homework_problems_per_page_l4010_401097


namespace NUMINAMATH_CALUDE_min_trios_l4010_401069

/-- Represents a group of people in a meeting -/
structure Meeting :=
  (people : Finset Nat)
  (handshakes : Set (Nat × Nat))
  (size_eq : people.card = 5)

/-- Defines a trio in the meeting -/
def is_trio (m : Meeting) (a b c : Nat) : Prop :=
  (a ∈ m.people ∧ b ∈ m.people ∧ c ∈ m.people) ∧
  ((⟨a, b⟩ ∈ m.handshakes ∧ ⟨b, c⟩ ∈ m.handshakes) ∨
   (⟨a, b⟩ ∉ m.handshakes ∧ ⟨b, c⟩ ∉ m.handshakes))

/-- Counts the number of unique trios in the meeting -/
def count_trios (m : Meeting) : Nat :=
  (m.people.powerset.filter (fun s => s.card = 3)).card

/-- The main theorem stating the minimum number of trios -/
theorem min_trios (m : Meeting) : 
  ∃ (handshakes : Set (Nat × Nat)), count_trios { people := m.people, handshakes := handshakes, size_eq := m.size_eq } = 10 ∧ 
  ∀ (other_handshakes : Set (Nat × Nat)), count_trios { people := m.people, handshakes := other_handshakes, size_eq := m.size_eq } ≥ 10 :=
sorry

end NUMINAMATH_CALUDE_min_trios_l4010_401069


namespace NUMINAMATH_CALUDE_shifted_quadratic_function_l4010_401090

/-- A quadratic function -/
def f (x : ℝ) : ℝ := -x^2

/-- Horizontal shift of a function -/
def horizontalShift (f : ℝ → ℝ) (h : ℝ) : ℝ → ℝ := fun x ↦ f (x + h)

/-- Vertical shift of a function -/
def verticalShift (f : ℝ → ℝ) (v : ℝ) : ℝ → ℝ := fun x ↦ f x + v

/-- The shifted function -/
def g : ℝ → ℝ := verticalShift (horizontalShift f 1) 3

theorem shifted_quadratic_function :
  ∀ x : ℝ, g x = -(x + 1)^2 + 3 :=
by sorry

end NUMINAMATH_CALUDE_shifted_quadratic_function_l4010_401090


namespace NUMINAMATH_CALUDE_steven_extra_seeds_l4010_401093

/-- Represents the number of seeds in different fruits -/
structure FruitSeeds where
  apple : Nat
  pear : Nat
  grape : Nat
  orange : Nat
  watermelon : Nat

/-- Represents the number of each fruit Steven has -/
structure StevenFruits where
  apples : Nat
  pears : Nat
  grapes : Nat
  oranges : Nat
  watermelons : Nat

def required_seeds : Nat := 420

def average_seeds : FruitSeeds := {
  apple := 6,
  pear := 2,
  grape := 3,
  orange := 10,
  watermelon := 300
}

def steven_fruits : StevenFruits := {
  apples := 2,
  pears := 3,
  grapes := 5,
  oranges := 1,
  watermelons := 2
}

/-- Calculates the total number of seeds Steven has -/
def total_seeds (avg : FruitSeeds) (fruits : StevenFruits) : Nat :=
  avg.apple * fruits.apples +
  avg.pear * fruits.pears +
  avg.grape * fruits.grapes +
  avg.orange * fruits.oranges +
  avg.watermelon * fruits.watermelons

/-- Theorem stating that Steven has 223 more seeds than required -/
theorem steven_extra_seeds :
  total_seeds average_seeds steven_fruits - required_seeds = 223 := by
  sorry

end NUMINAMATH_CALUDE_steven_extra_seeds_l4010_401093


namespace NUMINAMATH_CALUDE_joan_has_three_marbles_l4010_401046

/-- The number of yellow marbles Mary has -/
def mary_marbles : ℕ := 9

/-- The total number of yellow marbles Mary and Joan have together -/
def total_marbles : ℕ := 12

/-- The number of yellow marbles Joan has -/
def joan_marbles : ℕ := total_marbles - mary_marbles

theorem joan_has_three_marbles : joan_marbles = 3 := by
  sorry

end NUMINAMATH_CALUDE_joan_has_three_marbles_l4010_401046


namespace NUMINAMATH_CALUDE_total_salary_formula_l4010_401094

/-- Represents the total annual salary (in ten thousand yuan) paid by the enterprise in the nth year -/
def total_salary (n : ℕ) : ℝ :=
  (3 * n + 5) * (1.2 : ℝ)^n + 2.4

/-- The initial number of workers -/
def initial_workers : ℕ := 8

/-- The initial annual salary per worker (in yuan) -/
def initial_salary : ℝ := 10000

/-- The annual salary increase rate -/
def salary_increase_rate : ℝ := 0.2

/-- The number of new workers added each year -/
def new_workers_per_year : ℕ := 3

/-- The first-year salary of new workers (in yuan) -/
def new_worker_salary : ℝ := 8000

theorem total_salary_formula (n : ℕ) :
  total_salary n = (3 * n + initial_workers - 3) * (1 + salary_increase_rate)^n +
    (new_workers_per_year * new_worker_salary / 10000) := by
  sorry

end NUMINAMATH_CALUDE_total_salary_formula_l4010_401094


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4010_401004

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Theorem: In an arithmetic sequence, if a_1 + a_2 = 5 and a_3 + a_4 = 7, then a_5 + a_6 = 9 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
    (h_arith : arithmetic_sequence a)
    (h_sum_12 : a 1 + a 2 = 5)
    (h_sum_34 : a 3 + a 4 = 7) :
  a 5 + a 6 = 9 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4010_401004


namespace NUMINAMATH_CALUDE_polynomial_expansion_l4010_401060

theorem polynomial_expansion (x : ℝ) : 
  (5 * x^2 + 7 * x - 3) * (3 * x^3 + 4) = 
  15 * x^5 + 21 * x^4 - 9 * x^3 + 20 * x^2 + 28 * x - 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l4010_401060


namespace NUMINAMATH_CALUDE_lyft_taxi_cost_difference_l4010_401021

def uber_cost : ℝ := 22
def lyft_cost : ℝ := uber_cost - 3
def taxi_cost_with_tip : ℝ := 18
def tip_percentage : ℝ := 0.2

theorem lyft_taxi_cost_difference : 
  lyft_cost - (taxi_cost_with_tip / (1 + tip_percentage)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_lyft_taxi_cost_difference_l4010_401021


namespace NUMINAMATH_CALUDE_train_sequence_count_l4010_401014

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The factorial of a natural number -/
def factorial (n : ℕ) : ℕ := sorry

/-- The total number of departure sequences for 6 trains under given conditions -/
def train_sequences : ℕ :=
  let total_trains : ℕ := 6
  let trains_per_group : ℕ := 3
  let remaining_trains : ℕ := total_trains - 2  -- excluding A and B
  let ways_to_group : ℕ := choose remaining_trains (trains_per_group - 1)
  let ways_to_arrange_group : ℕ := factorial trains_per_group
  ways_to_group * ways_to_arrange_group * ways_to_arrange_group

theorem train_sequence_count : train_sequences = 216 := by sorry

end NUMINAMATH_CALUDE_train_sequence_count_l4010_401014


namespace NUMINAMATH_CALUDE_lemonade_sales_difference_l4010_401087

/-- 
Given:
- x: number of glasses of plain lemonade sold
- y: number of glasses of strawberry lemonade sold
- p: price of each glass of plain lemonade
- s: price of each glass of strawberry lemonade
- The total amount from plain lemonade is 1.5 times the total amount from strawberry lemonade

Prove that the difference between the total amount made from plain lemonade and 
strawberry lemonade is equal to 0.5 * (y * s)
-/
theorem lemonade_sales_difference 
  (x y p s : ℝ) 
  (h : x * p = 1.5 * (y * s)) : 
  x * p - y * s = 0.5 * (y * s) := by
  sorry


end NUMINAMATH_CALUDE_lemonade_sales_difference_l4010_401087


namespace NUMINAMATH_CALUDE_unique_intersection_l4010_401040

-- Define the function type
def RealFunction := ℝ → ℝ

-- Define the property of having domain [-1, 5]
def HasDomain (f : RealFunction) : Prop :=
  ∀ x, x ∈ Set.Icc (-1) 5 → ∃ y, f x = y

-- Define the intersection of f with the line x=1
def Intersection (f : RealFunction) : Set ℝ :=
  {y : ℝ | f 1 = y}

-- Theorem statement
theorem unique_intersection
  (f : RealFunction) (h : HasDomain f) :
  ∃! y, y ∈ Intersection f :=
sorry

end NUMINAMATH_CALUDE_unique_intersection_l4010_401040


namespace NUMINAMATH_CALUDE_ram_exam_result_l4010_401058

/-- The percentage of marks Ram got in his exam -/
def ram_percentage (marks_obtained : ℕ) (total_marks : ℕ) : ℚ :=
  (marks_obtained : ℚ) / (total_marks : ℚ) * 100

/-- Theorem stating that Ram's percentage is 90% -/
theorem ram_exam_result : ram_percentage 450 500 = 90 := by
  sorry

end NUMINAMATH_CALUDE_ram_exam_result_l4010_401058


namespace NUMINAMATH_CALUDE_plane_through_skew_perp_existence_l4010_401072

-- Define the concept of skew lines
def are_skew (a b : Line3D) : Prop := sorry

-- Define the concept of perpendicular lines
def are_perpendicular (l1 l2 : Line3D) : Prop := sorry

-- Define a plane passing through a line and perpendicular to another line
def plane_through_perp_to (a b : Line3D) : Set Point3D := sorry

theorem plane_through_skew_perp_existence (a b : Line3D) 
  (h_skew : are_skew a b) : 
  (∃! p : Set Point3D, p = plane_through_perp_to a b) ↔ are_perpendicular a b :=
sorry

end NUMINAMATH_CALUDE_plane_through_skew_perp_existence_l4010_401072


namespace NUMINAMATH_CALUDE_max_value_product_sum_l4010_401095

theorem max_value_product_sum (A M C : ℕ) (sum_constraint : A + M + C = 15) :
  A * M * C + A * M + M * C + C * A ≤ 200 ∧
  ∃ (A' M' C' : ℕ), A' + M' + C' = 15 ∧ A' * M' * C' + A' * M' + M' * C' + C' * A' = 200 :=
by sorry

end NUMINAMATH_CALUDE_max_value_product_sum_l4010_401095


namespace NUMINAMATH_CALUDE_min_distance_to_line_l4010_401012

/-- Given a right triangle with sides a, b, and hypotenuse c, and a point M(m, n) on the line ax+by+3c=0,
    the minimum value of m^2+n^2 is 9. -/
theorem min_distance_to_line (a b c : ℝ) (m n : ℝ → ℝ) :
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  (∀ t, a * (m t) + b * (n t) + 3 * c = 0) →
  (∃ t₀, ∀ t, (m t)^2 + (n t)^2 ≥ (m t₀)^2 + (n t₀)^2) →
  ∃ t₀, (m t₀)^2 + (n t₀)^2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l4010_401012


namespace NUMINAMATH_CALUDE_remainder_theorem_l4010_401031

theorem remainder_theorem : (1 - 90) ^ 10 ≡ 1 [MOD 88] := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l4010_401031


namespace NUMINAMATH_CALUDE_square_of_negation_l4010_401099

theorem square_of_negation (a : ℝ) : (-a)^2 = a^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negation_l4010_401099


namespace NUMINAMATH_CALUDE_cubic_roots_property_l4010_401032

theorem cubic_roots_property (a b c : ℂ) : 
  (a^3 - a^2 - a - 1 = 0) → 
  (b^3 - b^2 - b - 1 = 0) → 
  (c^3 - c^2 - c - 1 = 0) → 
  (a ≠ b ∧ b ≠ c ∧ c ≠ a) ∧ 
  (∃ n : ℤ, (a^1982 - b^1982) / (a - b) + (b^1982 - c^1982) / (b - c) + (c^1982 - a^1982) / (c - a) = n) := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_property_l4010_401032


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l4010_401081

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 2*x + 5 > 0) ↔ (∃ x : ℝ, x^2 + 2*x + 5 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l4010_401081


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l4010_401007

/-- An isosceles triangle with one angle of 94 degrees has a base angle of 43 degrees. -/
theorem isosceles_triangle_base_angle : ∀ (a b c : ℝ),
  a + b + c = 180 →  -- Sum of angles in a triangle is 180°
  a = b →            -- Two angles are equal (isosceles property)
  c = 94 →           -- One angle is 94°
  a = 43 :=          -- One of the base angles is 43°
by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l4010_401007


namespace NUMINAMATH_CALUDE_multiply_negatives_l4010_401079

theorem multiply_negatives : (-4 : ℚ) * (-(-(1/2))) = -2 := by
  sorry

end NUMINAMATH_CALUDE_multiply_negatives_l4010_401079


namespace NUMINAMATH_CALUDE_gcd_multiple_relation_l4010_401085

theorem gcd_multiple_relation (a b : ℕ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a = 7 * b) :
  Nat.gcd a b = b :=
by sorry

end NUMINAMATH_CALUDE_gcd_multiple_relation_l4010_401085


namespace NUMINAMATH_CALUDE_tangent_circle_height_difference_l4010_401049

/-- A circle tangent to the parabola y = x^2 + 1 at two points and inside the parabola -/
structure TangentCircle where
  /-- x-coordinate of one tangent point -/
  a : ℝ
  /-- y-coordinate of the circle's center -/
  b : ℝ
  /-- radius of the circle -/
  r : ℝ
  /-- The circle is tangent to the parabola at (a, a^2 + 1) and (-a, a^2 + 1) -/
  tangent_points : (a^2 + (a^2 + 1 - b)^2 = r^2) ∧ (a^2 + (a^2 + 1 - b)^2 = r^2)
  /-- The circle lies inside the parabola -/
  inside_parabola : ∀ x y, x^2 + (y - b)^2 = r^2 → y ≤ x^2 + 1

/-- The height difference between the center of the circle and the points of tangency is 1/2 -/
theorem tangent_circle_height_difference (c : TangentCircle) : 
  c.b - (c.a^2 + 1) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circle_height_difference_l4010_401049


namespace NUMINAMATH_CALUDE_add_9999_seconds_to_5_45_00_l4010_401074

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds seconds to a given time -/
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

/-- The initial time -/
def initialTime : Time :=
  { hours := 5, minutes := 45, seconds := 0 }

/-- The number of seconds to add -/
def secondsToAdd : Nat := 9999

/-- The expected final time -/
def expectedFinalTime : Time :=
  { hours := 8, minutes := 31, seconds := 39 }

theorem add_9999_seconds_to_5_45_00 :
  addSeconds initialTime secondsToAdd = expectedFinalTime := by
  sorry

end NUMINAMATH_CALUDE_add_9999_seconds_to_5_45_00_l4010_401074


namespace NUMINAMATH_CALUDE_triangular_coin_array_l4010_401000

theorem triangular_coin_array (N : ℕ) : (N * (N + 1)) / 2 = 3003 → N = 77 := by
  sorry

end NUMINAMATH_CALUDE_triangular_coin_array_l4010_401000


namespace NUMINAMATH_CALUDE_gcd_of_seven_digit_set_l4010_401054

/-- A function that generates a seven-digit number from a three-digit number -/
def seven_digit_from_three (n : ℕ) : ℕ := 1001 * n

/-- The set of all seven-digit numbers formed by repeating three-digit numbers -/
def seven_digit_set : Set ℕ := {m | ∃ n, 100 ≤ n ∧ n < 1000 ∧ m = seven_digit_from_three n}

/-- The theorem stating that 1001 is the greatest common divisor of all numbers in the set -/
theorem gcd_of_seven_digit_set :
  ∃ d, d > 0 ∧ (∀ m ∈ seven_digit_set, d ∣ m) ∧
  (∀ d' > 0, (∀ m ∈ seven_digit_set, d' ∣ m) → d' ≤ d) ∧
  d = 1001 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_seven_digit_set_l4010_401054


namespace NUMINAMATH_CALUDE_largest_x_value_l4010_401036

-- Define the probability function
def prob (x y : ℕ) : ℚ :=
  (Nat.choose x 2 + Nat.choose y 2) / Nat.choose (x + y) 2

-- State the theorem
theorem largest_x_value :
  ∀ x y : ℕ,
    x > y →
    x + y ≤ 2008 →
    prob x y = 1/2 →
    x ≤ 990 ∧ (∃ x' y' : ℕ, x' = 990 ∧ y' = 946 ∧ x' > y' ∧ x' + y' ≤ 2008 ∧ prob x' y' = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_largest_x_value_l4010_401036


namespace NUMINAMATH_CALUDE_linear_decreasing_iff_negative_slope_l4010_401025

/-- A linear function y = mx + b is decreasing on ℝ if and only if m < 0 -/
theorem linear_decreasing_iff_negative_slope (m b : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → m * x₁ + b > m * x₂ + b) ↔ m < 0 :=
by sorry

end NUMINAMATH_CALUDE_linear_decreasing_iff_negative_slope_l4010_401025


namespace NUMINAMATH_CALUDE_job_completion_time_l4010_401052

theorem job_completion_time (job : ℝ) (a_time : ℝ) (b_efficiency : ℝ) :
  job > 0 ∧ a_time = 15 ∧ b_efficiency = 1.8 →
  (job / (job / a_time * b_efficiency)) = 25 / 6 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l4010_401052


namespace NUMINAMATH_CALUDE_x4_plus_y4_l4010_401033

theorem x4_plus_y4 (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^4 + y^4 = 257 := by
  sorry

end NUMINAMATH_CALUDE_x4_plus_y4_l4010_401033


namespace NUMINAMATH_CALUDE_williams_land_percentage_l4010_401011

/-- Given a village with farm tax and Mr. William's tax payment, calculate the percentage of
    Mr. William's taxable land over the total taxable land of the village. -/
theorem williams_land_percentage 
  (total_tax : ℝ) 
  (williams_tax : ℝ) 
  (h1 : total_tax = 3840) 
  (h2 : williams_tax = 480) :
  williams_tax / total_tax = 0.125 := by
  sorry

#check williams_land_percentage

end NUMINAMATH_CALUDE_williams_land_percentage_l4010_401011


namespace NUMINAMATH_CALUDE_sum_of_divisors_30_l4010_401067

def sum_of_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem sum_of_divisors_30 : sum_of_divisors 30 = 72 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_30_l4010_401067


namespace NUMINAMATH_CALUDE_balloon_permutations_l4010_401078

def balloon_letters : Nat := 7
def l_count : Nat := 2
def o_count : Nat := 3

theorem balloon_permutations :
  (balloon_letters.factorial) / (l_count.factorial * o_count.factorial) = 420 := by
  sorry

end NUMINAMATH_CALUDE_balloon_permutations_l4010_401078


namespace NUMINAMATH_CALUDE_unpainted_cubes_count_l4010_401042

/-- Represents a cuboid with integer dimensions -/
structure Cuboid where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of unpainted cubes when a cuboid is cut into unit cubes -/
def unpaintedCubes (c : Cuboid) : ℕ :=
  (c.length - 2) * (c.width - 2) * (c.height - 2)

theorem unpainted_cubes_count :
  let c : Cuboid := { length := 6, width := 5, height := 4 }
  unpaintedCubes c = 24 := by sorry

end NUMINAMATH_CALUDE_unpainted_cubes_count_l4010_401042


namespace NUMINAMATH_CALUDE_prime_power_sum_l4010_401038

theorem prime_power_sum (a b c d e : ℕ) :
  2^a * 3^b * 5^c * 7^d * 11^e = 27720 →
  2*a + 3*b + 5*c + 7*d + 11*e = 35 := by
  sorry

end NUMINAMATH_CALUDE_prime_power_sum_l4010_401038


namespace NUMINAMATH_CALUDE_rabbit_carrots_l4010_401083

/-- Represents the number of carrots in each burrow -/
def carrots_per_burrow : ℕ := 2

/-- Represents the number of apples in each tree -/
def apples_per_tree : ℕ := 3

/-- Represents the difference between the number of burrows and trees -/
def burrow_tree_difference : ℕ := 3

theorem rabbit_carrots (burrows trees : ℕ) : 
  burrows = trees + burrow_tree_difference →
  carrots_per_burrow * burrows = apples_per_tree * trees →
  carrots_per_burrow * burrows = 18 := by
  sorry

end NUMINAMATH_CALUDE_rabbit_carrots_l4010_401083


namespace NUMINAMATH_CALUDE_geometric_sequences_theorem_l4010_401050

/-- Two geometric sequences satisfying given conditions -/
structure GeometricSequences where
  a : ℝ
  q : ℝ
  r : ℝ
  a_pos : a > 0
  b1_minus_a1 : a * r - a = 1
  b2_minus_a2 : a * r * r - a * q = 2
  b3_minus_a3 : a * r^3 - a * q^2 = 3

/-- The general term of the sequence a_n -/
def a_n (gs : GeometricSequences) (n : ℕ) : ℝ := gs.a * gs.q^(n-1)

/-- The general term of the sequence b_n -/
def b_n (gs : GeometricSequences) (n : ℕ) : ℝ := gs.a * gs.r^(n-1)

theorem geometric_sequences_theorem (gs : GeometricSequences) :
  (gs.a = 1 → (∀ n : ℕ, a_n gs n = (2 + Real.sqrt 2)^(n-1) ∨ a_n gs n = (2 - Real.sqrt 2)^(n-1))) ∧
  ((∃! q : ℝ, ∀ n : ℕ, a_n gs n = gs.a * q^(n-1)) → gs.a = 1/3) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequences_theorem_l4010_401050


namespace NUMINAMATH_CALUDE_possible_values_of_e_l4010_401051

theorem possible_values_of_e :
  ∀ e : ℝ, |2 - e| = 5 → (e = 7 ∨ e = -3) :=
sorry

end NUMINAMATH_CALUDE_possible_values_of_e_l4010_401051


namespace NUMINAMATH_CALUDE_ivan_nails_purchase_l4010_401091

-- Define the cost of nails per 100 grams in each store
def cost_store1 : ℝ := 180
def cost_store2 : ℝ := 120

-- Define the amount Ivan was short in the first store
def short_amount : ℝ := 1430

-- Define the change Ivan received in the second store
def change_amount : ℝ := 490

-- Define the function to calculate the cost of nails in kilograms
def cost_per_kg (cost_per_100g : ℝ) : ℝ := cost_per_100g * 10

-- Define the amount of nails Ivan bought in kilograms
def nails_bought : ℝ := 3.2

-- Theorem statement
theorem ivan_nails_purchase :
  (cost_per_kg cost_store1 * nails_bought - (cost_per_kg cost_store2 * nails_bought + change_amount) = short_amount) ∧
  (nails_bought = 3.2) :=
by sorry

end NUMINAMATH_CALUDE_ivan_nails_purchase_l4010_401091


namespace NUMINAMATH_CALUDE_car_speed_problem_l4010_401001

/-- The speed of Car A in km/h -/
def speed_A : ℝ := 80

/-- The time taken by Car A in hours -/
def time_A : ℝ := 5

/-- The speed of Car B in km/h -/
def speed_B : ℝ := 100

/-- The time taken by Car B in hours -/
def time_B : ℝ := 2

/-- The ratio of distances covered by Car A and Car B -/
def distance_ratio : ℝ := 2

theorem car_speed_problem :
  speed_A * time_A = distance_ratio * speed_B * time_B :=
sorry

end NUMINAMATH_CALUDE_car_speed_problem_l4010_401001


namespace NUMINAMATH_CALUDE_least_value_x_minus_y_minus_z_l4010_401035

theorem least_value_x_minus_y_minus_z :
  ∀ (x y z : ℕ+), x = 4 → y = 7 → (x : ℤ) - y - z ≥ -4 ∧ ∃ (z : ℕ+), (x : ℤ) - y - z = -4 :=
by sorry

end NUMINAMATH_CALUDE_least_value_x_minus_y_minus_z_l4010_401035


namespace NUMINAMATH_CALUDE_num_valid_selections_l4010_401068

/-- Represents the set of volunteers --/
inductive Volunteer
| A
| B
| C
| D
| E

/-- Represents the set of roles --/
inductive Role
| Translator
| TourGuide
| Etiquette
| Driver

/-- Predicate to check if a volunteer can take on a role --/
def canTakeRole (v : Volunteer) (r : Role) : Prop :=
  match v, r with
  | Volunteer.A, Role.Driver => False
  | Volunteer.B, Role.Driver => False
  | _, _ => True

/-- A selection is a function from Role to Volunteer --/
def Selection := Role → Volunteer

/-- Predicate to check if a selection is valid --/
def validSelection (s : Selection) : Prop :=
  (∀ r : Role, canTakeRole (s r) r) ∧
  (∀ v : Volunteer, ∃! r : Role, s r = v)

/-- The number of valid selections --/
def numValidSelections : ℕ := sorry

theorem num_valid_selections :
  numValidSelections = 72 := by sorry

end NUMINAMATH_CALUDE_num_valid_selections_l4010_401068


namespace NUMINAMATH_CALUDE_quadratic_roots_property_integer_values_k_l4010_401059

theorem quadratic_roots_property (k : ℝ) (x₁ x₂ : ℝ) : 
  (4 * k * x₁^2 - 4 * k * x₁ + k + 1 = 0 ∧ 
   4 * k * x₂^2 - 4 * k * x₂ + k + 1 = 0) → 
  (2 * x₁ - x₂) * (x₁ - 2 * x₂) ≠ -3/2 :=
sorry

theorem integer_values_k (k : ℤ) :
  (∃ x₁ x₂ : ℝ, 4 * (k : ℝ) * x₁^2 - 4 * (k : ℝ) * x₁ + (k : ℝ) + 1 = 0 ∧
                4 * (k : ℝ) * x₂^2 - 4 * (k : ℝ) * x₂ + (k : ℝ) + 1 = 0 ∧
                ∃ n : ℤ, (x₁ / x₂ + x₂ / x₁ - 2 : ℝ) = n) ↔
  k = -2 ∨ k = -3 ∨ k = -5 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_integer_values_k_l4010_401059


namespace NUMINAMATH_CALUDE_coefficient_x_term_expansion_l4010_401034

theorem coefficient_x_term_expansion (x : ℝ) : 
  ∃ a b c d : ℝ, (x^2 - 3*x + 3)^3 = a*x^3 + b*x^2 + (-81)*x + d :=
sorry

end NUMINAMATH_CALUDE_coefficient_x_term_expansion_l4010_401034


namespace NUMINAMATH_CALUDE_certain_number_is_100_l4010_401022

theorem certain_number_is_100 : ∃! x : ℝ, ((x / 4) + 25) * 3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_is_100_l4010_401022


namespace NUMINAMATH_CALUDE_sum_of_factors_l4010_401003

theorem sum_of_factors (d e f : ℤ) : 
  (∀ x : ℝ, x^2 + 21*x + 110 = (x + d)*(x + e)) → 
  (∀ x : ℝ, x^2 - 19*x + 88 = (x - e)*(x - f)) → 
  d + e + f = 30 := by
sorry

end NUMINAMATH_CALUDE_sum_of_factors_l4010_401003


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l4010_401055

/-- Given two positive real numbers with sum 55, HCF 5, and LCM 120, 
    prove that the sum of their reciprocals is 11/120 -/
theorem sum_of_reciprocals (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (sum : a + b = 55) (hcf : Int.gcd (Int.floor a) (Int.floor b) = 5) 
  (lcm : Int.lcm (Int.floor a) (Int.floor b) = 120) : 
  1 / a + 1 / b = 11 / 120 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l4010_401055


namespace NUMINAMATH_CALUDE_infinitely_many_linear_combinations_l4010_401005

/-- An infinite sequence of positive integers with strictly increasing terms. -/
def StrictlyIncreasingSequence (a : ℕ → ℕ) : Prop :=
  ∀ k, 0 < a k ∧ a k < a (k + 1)

/-- The property that infinitely many terms can be expressed as a linear combination of two earlier terms. -/
def InfinitelyManyLinearCombinations (a : ℕ → ℕ) : Prop :=
  ∀ N, ∃ m p q x y, N < m ∧ p ≠ q ∧ 0 < x ∧ 0 < y ∧ a m = x * a p + y * a q

/-- The main theorem: any strictly increasing sequence of positive integers has infinitely many terms
    that can be expressed as a linear combination of two earlier terms. -/
theorem infinitely_many_linear_combinations
  (a : ℕ → ℕ) (h : StrictlyIncreasingSequence a) :
  InfinitelyManyLinearCombinations a :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_linear_combinations_l4010_401005


namespace NUMINAMATH_CALUDE_expand_expression_l4010_401043

theorem expand_expression (x y : ℝ) : (x + 15) * (3 * y + 20) = 3 * x * y + 20 * x + 45 * y + 300 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l4010_401043


namespace NUMINAMATH_CALUDE_robot_position_difference_l4010_401045

-- Define the robot's position function
def robot_position (n : ℕ) : ℤ :=
  let full_cycles := n / 7
  let remainder := n % 7
  let cycle_progress := if remainder ≤ 4 then remainder else 4 - (remainder - 4)
  full_cycles + cycle_progress

-- State the theorem
theorem robot_position_difference : robot_position 2007 - robot_position 2011 = 0 := by
  sorry

end NUMINAMATH_CALUDE_robot_position_difference_l4010_401045


namespace NUMINAMATH_CALUDE_simple_interest_principal_l4010_401024

/-- Simple interest calculation -/
theorem simple_interest_principal (interest : ℚ) (time : ℚ) (rate : ℚ) (principal : ℚ) :
  interest = principal * rate * time ∧
  interest = 10.92 ∧
  time = 6 ∧
  rate = 7 / 100 / 12 →
  principal = 26 := by sorry

end NUMINAMATH_CALUDE_simple_interest_principal_l4010_401024


namespace NUMINAMATH_CALUDE_line_intersects_circle_twice_tangent_line_m_value_l4010_401096

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 25

-- Define the line L
def line_L (m x y : ℝ) : Prop := (2*m + 1)*x + (m + 1)*y - 7*m - 4 = 0

-- Define the circle D
def circle_D (R x y : ℝ) : Prop := (x + 1)^2 + (y - 5)^2 = R^2

theorem line_intersects_circle_twice (m : ℝ) :
  ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    line_L m x₁ y₁ ∧ line_L m x₂ y₂ :=
sorry

theorem tangent_line_m_value :
  ∃ (R : ℝ), R > 0 ∧
    (∀ (R' : ℝ), R' > 0 →
      (∃ (x y : ℝ), circle_D R' x y ∧ line_L (-2/3) x y) →
      R' ≤ R) ∧
    (∃ (x y : ℝ), circle_D R x y ∧ line_L (-2/3) x y) :=
sorry

end NUMINAMATH_CALUDE_line_intersects_circle_twice_tangent_line_m_value_l4010_401096


namespace NUMINAMATH_CALUDE_leahs_coins_value_l4010_401066

/-- Represents the number of coins Leah has -/
def total_coins : ℕ := 15

/-- Represents the value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Represents the value of a penny in cents -/
def penny_value : ℕ := 1

/-- Represents Leah's coin collection -/
structure CoinCollection where
  nickels : ℕ
  pennies : ℕ

/-- The conditions of Leah's coin collection -/
def valid_collection (c : CoinCollection) : Prop :=
  c.nickels + c.pennies = total_coins ∧
  c.nickels + 1 = c.pennies

/-- The total value of a coin collection in cents -/
def collection_value (c : CoinCollection) : ℕ :=
  c.nickels * nickel_value + c.pennies * penny_value

/-- The main theorem stating that Leah's coins are worth 43 cents -/
theorem leahs_coins_value (c : CoinCollection) :
  valid_collection c → collection_value c = 43 := by
  sorry

end NUMINAMATH_CALUDE_leahs_coins_value_l4010_401066


namespace NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l4010_401018

theorem square_sum_zero_implies_both_zero (a b : ℝ) : a^2 + b^2 = 0 → a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l4010_401018


namespace NUMINAMATH_CALUDE_destiny_snack_bags_l4010_401088

theorem destiny_snack_bags (chocolate_bars : Nat) (cookies : Nat) 
  (h1 : chocolate_bars = 18) (h2 : cookies = 12) :
  Nat.gcd chocolate_bars cookies = 6 := by
  sorry

end NUMINAMATH_CALUDE_destiny_snack_bags_l4010_401088


namespace NUMINAMATH_CALUDE_two_digit_number_problem_l4010_401026

theorem two_digit_number_problem (A B : ℕ) : 
  (A ≥ 1 ∧ A ≤ 9) →  -- A is a digit from 1 to 9 (tens digit)
  (B ≥ 0 ∧ B ≤ 9) →  -- B is a digit from 0 to 9 (ones digit)
  (10 * A + B) - 21 = 14 →  -- The equation AB - 21 = 14
  B = 5 := by  -- We want to prove B = 5
sorry

end NUMINAMATH_CALUDE_two_digit_number_problem_l4010_401026


namespace NUMINAMATH_CALUDE_gcd_lcm_product_360_l4010_401077

theorem gcd_lcm_product_360 (x y : ℕ+) : 
  (Nat.gcd x y * Nat.lcm x y = 360) → 
  (∃ (s : Finset ℕ), s.card = 8 ∧ ∀ (d : ℕ), d ∈ s ↔ ∃ (a b : ℕ+), Nat.gcd a b * Nat.lcm a b = 360 ∧ Nat.gcd a b = d) :=
sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_360_l4010_401077


namespace NUMINAMATH_CALUDE_share_price_increase_l4010_401017

theorem share_price_increase (initial_price : ℝ) : 
  let first_quarter_price := initial_price * (1 + 0.2)
  let second_quarter_price := first_quarter_price * (1 + 1/3)
  second_quarter_price = initial_price * (1 + 0.6) := by
sorry

end NUMINAMATH_CALUDE_share_price_increase_l4010_401017


namespace NUMINAMATH_CALUDE_altitude_equation_correct_l4010_401057

-- Define the triangle vertices
def A : ℝ × ℝ := (-5, 3)
def B : ℝ × ℝ := (3, 7)
def C : ℝ × ℝ := (4, -1)

-- Define the vector BC
def BC : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)

-- Define the altitude equation
def altitudeEquation (x y : ℝ) : Prop := x - 8*y + 29 = 0

-- Theorem statement
theorem altitude_equation_correct :
  ∀ x y : ℝ, altitudeEquation x y ↔
  ((x - A.1, y - A.2) • BC = 0 ∧ ∃ t : ℝ, (x, y) = (B.1 + t * (C.1 - B.1), B.2 + t * (C.2 - B.2))) :=
by sorry

end NUMINAMATH_CALUDE_altitude_equation_correct_l4010_401057


namespace NUMINAMATH_CALUDE_parallel_line_equation_l4010_401075

/-- Given a line with slope 2/3 and y-intercept 5, 
    prove that a parallel line 5 units away has the equation 
    y = (2/3)x + (5 ± (5√13)/3) -/
theorem parallel_line_equation (x y : ℝ) : 
  let given_line := λ x : ℝ => (2/3) * x + 5
  let distance := 5
  let parallel_line := λ x : ℝ => (2/3) * x + c
  let c_diff := |c - 5|
  (∀ x, |parallel_line x - given_line x| = distance) →
  (c = 5 + (5 * Real.sqrt 13) / 3 ∨ c = 5 - (5 * Real.sqrt 13) / 3) :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_equation_l4010_401075


namespace NUMINAMATH_CALUDE_quadratic_equation_equivalence_l4010_401002

theorem quadratic_equation_equivalence (m : ℝ) : 
  (∀ x, x^2 - m*x + 6 = 0 ↔ (x - 3)^2 = 3) → m = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_equivalence_l4010_401002


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l4010_401006

-- Define the diamond operation
noncomputable def diamond (b c : ℝ) : ℝ := b + Real.sqrt (c + Real.sqrt (c + Real.sqrt c))

-- State the theorem
theorem diamond_equation_solution (k : ℝ) :
  diamond 10 k = 13 → k = 6 := by
  sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l4010_401006


namespace NUMINAMATH_CALUDE_cos_72_degrees_l4010_401029

theorem cos_72_degrees : Real.cos (72 * π / 180) = (-1 + Real.sqrt 5) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_72_degrees_l4010_401029


namespace NUMINAMATH_CALUDE_sum_of_coordinates_D_l4010_401009

-- Define the points
def C : ℝ × ℝ := (-6, 1)
def M : ℝ × ℝ := (-2, 3)

-- Define the midpoint formula
def is_midpoint (m x y : ℝ × ℝ) : Prop :=
  m.1 = (x.1 + y.1) / 2 ∧ m.2 = (x.2 + y.2) / 2

-- Theorem statement
theorem sum_of_coordinates_D :
  ∃ D : ℝ × ℝ, is_midpoint M C D ∧ D.1 + D.2 = 7 :=
sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_D_l4010_401009


namespace NUMINAMATH_CALUDE_smallest_marble_count_l4010_401063

/-- Represents the number of marbles of each color in the urn -/
structure MarbleCount where
  red : ℕ
  white : ℕ
  blue : ℕ

/-- Calculates the probability of drawing a specific combination of marbles -/
def probability (m : MarbleCount) (r w b : ℕ) : ℚ :=
  (m.red.choose r) * (m.white.choose w) * (m.blue.choose b) /
  ((m.red + m.white + m.blue).choose 4)

/-- Checks if the three specified events are equally likely -/
def events_equally_likely (m : MarbleCount) : Prop :=
  probability m 3 1 0 = probability m 2 1 1 ∧
  probability m 3 1 0 = probability m 2 1 1

/-- The theorem stating that 8 is the smallest number of marbles satisfying the conditions -/
theorem smallest_marble_count :
  ∃ (m : MarbleCount),
    m.red + m.white + m.blue = 8 ∧
    events_equally_likely m ∧
    ∀ (n : MarbleCount),
      n.red + n.white + n.blue < 8 →
      ¬(events_equally_likely n) :=
sorry

end NUMINAMATH_CALUDE_smallest_marble_count_l4010_401063


namespace NUMINAMATH_CALUDE_no_two_digit_divisible_by_reverse_l4010_401048

theorem no_two_digit_divisible_by_reverse : ¬ ∃ (a b : ℕ), 
  1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ a ≠ b ∧ 
  ∃ (k : ℕ), k > 1 ∧ (10 * a + b) = k * (10 * b + a) :=
by sorry

end NUMINAMATH_CALUDE_no_two_digit_divisible_by_reverse_l4010_401048


namespace NUMINAMATH_CALUDE_diego_yearly_savings_l4010_401062

/-- Calculates the yearly savings given monthly deposit, monthly expenses, and number of months in a year. -/
def yearly_savings (monthly_deposit : ℕ) (monthly_expenses : ℕ) (months_in_year : ℕ) : ℕ :=
  (monthly_deposit - monthly_expenses) * months_in_year

/-- Theorem stating that Diego's yearly savings is $4,800 -/
theorem diego_yearly_savings :
  yearly_savings 5000 4600 12 = 4800 := by
  sorry

end NUMINAMATH_CALUDE_diego_yearly_savings_l4010_401062


namespace NUMINAMATH_CALUDE_area_of_specific_trapezoid_l4010_401030

/-- An isosceles trapezoid inscribed in a circle -/
structure InscribedTrapezoid where
  /-- The radius of the circumscribed circle -/
  radius : ℝ
  /-- The length of the shorter base of the trapezoid -/
  shorter_base : ℝ
  /-- The center of the circle lies on the longer base of the trapezoid -/
  center_on_longer_base : Bool

/-- The area of an inscribed isosceles trapezoid -/
def area (t : InscribedTrapezoid) : ℝ := sorry

/-- Theorem stating that the area of the specific inscribed trapezoid is 32 -/
theorem area_of_specific_trapezoid :
  let t : InscribedTrapezoid := ⟨5, 6, true⟩
  area t = 32 := by sorry

end NUMINAMATH_CALUDE_area_of_specific_trapezoid_l4010_401030


namespace NUMINAMATH_CALUDE_lisa_flight_time_l4010_401064

/-- 
Given that Lisa flew 500 miles at a speed of 45 miles per hour, 
prove that the time Lisa flew is equal to 500 miles divided by 45 miles per hour.
-/
theorem lisa_flight_time : 
  let distance : ℝ := 500  -- Distance in miles
  let speed : ℝ := 45      -- Speed in miles per hour
  let time : ℝ := distance / speed
  time = 500 / 45 := by sorry

end NUMINAMATH_CALUDE_lisa_flight_time_l4010_401064


namespace NUMINAMATH_CALUDE_solution_set_theorem_a_range_theorem_l4010_401073

/-- The function f(x) defined as |x| + 2|x-a| where a > 0 -/
def f (a : ℝ) (x : ℝ) : ℝ := |x| + 2 * |x - a|

/-- The solution set of f(x) ≤ 4 when a = 1 -/
def solution_set : Set ℝ := {x : ℝ | x ∈ Set.Icc (-2/3) 2}

/-- The range of a for which f(x) ≥ 4 always holds -/
def a_range : Set ℝ := {a : ℝ | a ∈ Set.Ici 4}

/-- Theorem stating the solution set of f(x) ≤ 4 when a = 1 -/
theorem solution_set_theorem :
  ∀ x : ℝ, f 1 x ≤ 4 ↔ x ∈ solution_set := by sorry

/-- Theorem stating the range of a for which f(x) ≥ 4 always holds -/
theorem a_range_theorem :
  ∀ a : ℝ, (∀ x : ℝ, f a x ≥ 4) ↔ a ∈ a_range := by sorry

end NUMINAMATH_CALUDE_solution_set_theorem_a_range_theorem_l4010_401073


namespace NUMINAMATH_CALUDE_squares_in_figure_50_l4010_401092

/-- The number of squares in figure n -/
def f (n : ℕ) : ℕ := 2 * n^2 + 4 * n + 2

/-- The sequence satisfies the given initial conditions -/
axiom initial_conditions :
  f 0 = 2 ∧ f 1 = 8 ∧ f 2 = 18 ∧ f 3 = 32

/-- The number of squares in figure 50 is 5202 -/
theorem squares_in_figure_50 : f 50 = 5202 := by
  sorry

end NUMINAMATH_CALUDE_squares_in_figure_50_l4010_401092


namespace NUMINAMATH_CALUDE_final_tree_count_l4010_401070

/-- 
Given:
- T: The initial number of trees
- P: The percentage of trees cut (as a whole number, e.g., 20 for 20%)
- R: The number of new trees planted for each tree cut

Prove that the final number of trees F is equal to T - (P/100 * T) + (P/100 * T * R)
-/
theorem final_tree_count (T P R : ℕ) (h1 : P ≤ 100) : 
  ∃ F : ℕ, F = T - (P * T / 100) + (P * T * R / 100) :=
sorry

end NUMINAMATH_CALUDE_final_tree_count_l4010_401070


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l4010_401047

theorem hyperbola_focal_length (x y : ℝ) :
  x^2 / 7 - y^2 / 3 = 1 → 2 * Real.sqrt 10 = 2 * Real.sqrt (7 + 3) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l4010_401047


namespace NUMINAMATH_CALUDE_ethan_hourly_wage_l4010_401037

/-- Represents Ethan's work schedule and earnings --/
structure WorkSchedule where
  hours_per_day : ℕ
  days_per_week : ℕ
  weeks_worked : ℕ
  total_earnings : ℕ

/-- Calculates the hourly wage given a work schedule --/
def hourly_wage (schedule : WorkSchedule) : ℚ :=
  schedule.total_earnings / (schedule.hours_per_day * schedule.days_per_week * schedule.weeks_worked)

/-- Theorem stating that Ethan's hourly wage is $18 --/
theorem ethan_hourly_wage :
  let ethan_schedule : WorkSchedule := {
    hours_per_day := 8,
    days_per_week := 5,
    weeks_worked := 5,
    total_earnings := 3600
  }
  hourly_wage ethan_schedule = 18 := by
  sorry

end NUMINAMATH_CALUDE_ethan_hourly_wage_l4010_401037


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l4010_401008

theorem line_passes_through_fixed_point 
  (a b c : ℝ) 
  (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) 
  (h_sum : 1/a + 1/b = 1/c) : 
  ∃ (x y : ℝ), x/a + y/b = 1 ∧ x = c ∧ y = c :=
by sorry


end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l4010_401008


namespace NUMINAMATH_CALUDE_crabapple_recipients_count_l4010_401019

/-- The number of students in Mrs. Crabapple's class -/
def num_students : ℕ := 12

/-- The number of times the class meets in a week -/
def class_meetings_per_week : ℕ := 5

/-- The number of possible sequences of crabapple recipients in a week -/
def crabapple_sequences : ℕ := num_students ^ class_meetings_per_week

/-- Theorem stating the number of possible sequences of crabapple recipients -/
theorem crabapple_recipients_count : crabapple_sequences = 248832 := by
  sorry

end NUMINAMATH_CALUDE_crabapple_recipients_count_l4010_401019


namespace NUMINAMATH_CALUDE_max_students_distribution_l4010_401061

theorem max_students_distribution (pens pencils : ℕ) 
  (h1 : pens = 1001) (h2 : pencils = 910) : ℕ :=
  Nat.gcd pens pencils

#check max_students_distribution

end NUMINAMATH_CALUDE_max_students_distribution_l4010_401061


namespace NUMINAMATH_CALUDE_two_over_a_lt_one_necessary_not_sufficient_for_a_squared_gt_four_l4010_401098

theorem two_over_a_lt_one_necessary_not_sufficient_for_a_squared_gt_four :
  (∀ a : ℝ, a^2 > 4 → 2/a < 1) ∧
  (∃ a : ℝ, 2/a < 1 ∧ a^2 ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_two_over_a_lt_one_necessary_not_sufficient_for_a_squared_gt_four_l4010_401098


namespace NUMINAMATH_CALUDE_volume_cylinder_from_square_rotation_l4010_401016

/-- The volume of a cylinder formed by rotating a square about its horizontal line of symmetry -/
theorem volume_cylinder_from_square_rotation (side_length : ℝ) (volume : ℝ) :
  side_length = 16 →
  volume = π * side_length^3 / 4 →
  volume = 1024 * π :=
by sorry

end NUMINAMATH_CALUDE_volume_cylinder_from_square_rotation_l4010_401016


namespace NUMINAMATH_CALUDE_erica_safari_lions_erica_saw_three_lions_l4010_401084

/-- Prove that Erica saw 3 lions on Saturday during her safari -/
theorem erica_safari_lions : ℕ → Prop := fun n =>
  let total_animals : ℕ := 20
  let saturday_elephants : ℕ := 2
  let sunday_animals : ℕ := 2 + 5  -- 2 buffaloes and 5 leopards
  let monday_animals : ℕ := 5 + 3  -- 5 rhinos and 3 warthogs
  n = total_animals - (saturday_elephants + sunday_animals + monday_animals)

/-- The number of lions Erica saw on Saturday is 3 -/
theorem erica_saw_three_lions : erica_safari_lions 3 := by
  sorry

end NUMINAMATH_CALUDE_erica_safari_lions_erica_saw_three_lions_l4010_401084


namespace NUMINAMATH_CALUDE_correct_travel_distance_l4010_401065

/-- The distance traveled by Gavril on the electric train -/
def travel_distance : ℝ := 257

/-- The time it takes for the smartphone to fully discharge while watching videos -/
def video_discharge_time : ℝ := 3

/-- The time it takes for the smartphone to fully discharge while playing Tetris -/
def tetris_discharge_time : ℝ := 5

/-- The speed of the train for the first half of the journey -/
def speed_first_half : ℝ := 80

/-- The speed of the train for the second half of the journey -/
def speed_second_half : ℝ := 60

/-- Theorem stating that given the conditions, the travel distance is correct -/
theorem correct_travel_distance :
  let total_time := (video_discharge_time * tetris_discharge_time) / (video_discharge_time / 2 + tetris_discharge_time / 2)
  travel_distance = total_time * (speed_first_half / 2 + speed_second_half / 2) :=
by sorry

end NUMINAMATH_CALUDE_correct_travel_distance_l4010_401065


namespace NUMINAMATH_CALUDE_triangle_side_length_l4010_401056

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  2 * b = a + c →
  B = π / 6 →
  (1 / 2) * a * c * Real.sin B = 3 / 2 →
  b = 1 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l4010_401056


namespace NUMINAMATH_CALUDE_cricket_team_age_difference_l4010_401013

theorem cricket_team_age_difference (team_size : ℕ) (captain_age : ℕ) (keeper_age_diff : ℕ) (team_avg_age : ℚ) :
  team_size = 11 →
  captain_age = 26 →
  keeper_age_diff = 3 →
  team_avg_age = 23 →
  let keeper_age := captain_age + keeper_age_diff
  let total_team_age := team_avg_age * team_size
  let remaining_players := team_size - 2
  let remaining_age := total_team_age - (captain_age + keeper_age)
  let remaining_avg_age := remaining_age / remaining_players
  (team_avg_age - remaining_avg_age) = 1 := by
sorry

end NUMINAMATH_CALUDE_cricket_team_age_difference_l4010_401013


namespace NUMINAMATH_CALUDE_f_of_2_equals_5_l4010_401086

def f (x : ℝ) : ℝ := x^2 + 2*x - 3

theorem f_of_2_equals_5 : f 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_f_of_2_equals_5_l4010_401086


namespace NUMINAMATH_CALUDE_convex_polygon_sides_l4010_401010

/-- The number of sides in a convex polygon where the sum of n-1 internal angles is 2009 degrees -/
def polygon_sides : ℕ := 14

theorem convex_polygon_sides :
  ∀ n : ℕ,
  n > 2 →
  (n - 1) * 180 < 2009 →
  n * 180 > 2009 →
  n = polygon_sides :=
by sorry

end NUMINAMATH_CALUDE_convex_polygon_sides_l4010_401010


namespace NUMINAMATH_CALUDE_capital_payment_theorem_l4010_401023

def remaining_capital (m : ℕ) (d : ℚ) : ℚ :=
  (3/2)^(m-1) * (3000 - 3*d) + 2*d

theorem capital_payment_theorem (m : ℕ) (h : m ≥ 3) :
  ∃ d : ℚ, remaining_capital m d = 4000 ∧ 
    d = (1000 * (3^m - 2^(m+1))) / (3^m - 2^m) := by
  sorry

end NUMINAMATH_CALUDE_capital_payment_theorem_l4010_401023


namespace NUMINAMATH_CALUDE_exponential_equation_solution_l4010_401082

theorem exponential_equation_solution :
  ∃! x : ℝ, 3^(2*x + 2) = (1 : ℝ) / 9 :=
by
  use -2
  sorry

end NUMINAMATH_CALUDE_exponential_equation_solution_l4010_401082


namespace NUMINAMATH_CALUDE_expression_factorization_l4010_401044

theorem expression_factorization (a b c x : ℝ) :
  (x - a)^2 * (b - c) + (x - b)^2 * (c - a) + (x - c)^2 * (a - b) = -(a - b) * (b - c) * (c - a) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l4010_401044


namespace NUMINAMATH_CALUDE_store_transaction_result_l4010_401076

theorem store_transaction_result : 
  let selling_price : ℝ := 960
  let profit_margin : ℝ := 0.2
  let cost_profit_item : ℝ := selling_price / (1 + profit_margin)
  let cost_loss_item : ℝ := selling_price / (1 - profit_margin)
  let total_cost : ℝ := cost_profit_item + cost_loss_item
  let total_revenue : ℝ := 2 * selling_price
  total_cost - total_revenue = 80
  := by sorry

end NUMINAMATH_CALUDE_store_transaction_result_l4010_401076


namespace NUMINAMATH_CALUDE_alcohol_dilution_l4010_401089

theorem alcohol_dilution (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hab : b < a) (hbc : c < b) :
  let initial_volume : ℝ := 1
  let first_dilution_volume : ℝ := a / b
  let second_dilution_volume : ℝ := a / (b + c)
  let total_water_used : ℝ := (first_dilution_volume - initial_volume) + (2 * second_dilution_volume - first_dilution_volume)
  total_water_used = 2 * a / (b + c) - 1 := by
sorry

end NUMINAMATH_CALUDE_alcohol_dilution_l4010_401089
