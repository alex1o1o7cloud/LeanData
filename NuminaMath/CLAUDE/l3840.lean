import Mathlib

namespace NUMINAMATH_CALUDE_square_perimeter_l3840_384038

theorem square_perimeter (rectangle_length : ℝ) (rectangle_width : ℝ) 
  (h1 : rectangle_length = 125)
  (h2 : rectangle_width = 64)
  (h3 : ∃ square_side : ℝ, square_side^2 = 5 * rectangle_length * rectangle_width) :
  ∃ square_perimeter : ℝ, square_perimeter = 800 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_l3840_384038


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l3840_384037

/-- An arithmetic sequence with common difference d -/
def ArithmeticSequence (d : ℝ) (a : ℕ → ℝ) : Prop :=
  d ≠ 0 ∧ ∀ n, a (n + 1) = a n + d

/-- The property that the sum of any two distinct terms is a term in the sequence -/
def SumPropertyHolds (a : ℕ → ℝ) : Prop :=
  ∀ s t, s ≠ t → ∃ k, a s + a t = a k

/-- The theorem stating the equivalence of the sum property and the existence of m -/
theorem arithmetic_sequence_sum_property (d : ℝ) (a : ℕ → ℝ) :
  ArithmeticSequence d a →
  (SumPropertyHolds a ↔ ∃ m : ℤ, m ≥ -1 ∧ a 1 = m * d) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l3840_384037


namespace NUMINAMATH_CALUDE_logarithm_expression_equality_l3840_384010

-- Define the logarithm base 2
noncomputable def log2 (x : ℝ) := Real.log x / Real.log 2

-- Define the logarithm base 10
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

theorem logarithm_expression_equality : 
  2^(log2 3) + lg (Real.sqrt 5) + lg (Real.sqrt 20) = 4 := by sorry

end NUMINAMATH_CALUDE_logarithm_expression_equality_l3840_384010


namespace NUMINAMATH_CALUDE_sum_of_prime_factors_2550_l3840_384065

def sum_of_prime_factors (n : ℕ) : ℕ := sorry

theorem sum_of_prime_factors_2550 : sum_of_prime_factors 2550 = 27 := by sorry

end NUMINAMATH_CALUDE_sum_of_prime_factors_2550_l3840_384065


namespace NUMINAMATH_CALUDE_smallest_factor_for_perfect_cube_l3840_384020

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

theorem smallest_factor_for_perfect_cube (x y : ℕ) : 
  x = 5 * 30 * 60 →
  y > 0 →
  is_perfect_cube (x * y) →
  (∀ z : ℕ, z > 0 → z < y → ¬ is_perfect_cube (x * z)) →
  y = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_factor_for_perfect_cube_l3840_384020


namespace NUMINAMATH_CALUDE_alpha_value_l3840_384083

theorem alpha_value (α β : ℂ) 
  (h1 : (α + β).re > 0)
  (h2 : (2 * (α - 3 * β)).re > 0)
  (h3 : β = 5 + 4 * Complex.I) :
  α = 16 - 4 * Complex.I := by
sorry

end NUMINAMATH_CALUDE_alpha_value_l3840_384083


namespace NUMINAMATH_CALUDE_simple_interest_principal_l3840_384043

/-- Simple interest calculation --/
theorem simple_interest_principal (amount : ℝ) (rate : ℝ) (time : ℝ) (principal : ℝ) :
  amount = 1456 ∧ rate = 0.05 ∧ time = 2.4 →
  principal = 1300 ∧ amount = principal * (1 + rate * time) := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_principal_l3840_384043


namespace NUMINAMATH_CALUDE_percent_calculation_l3840_384051

theorem percent_calculation (x : ℝ) (h : 0.2 * x = 60) : 0.8 * x = 240 := by
  sorry

end NUMINAMATH_CALUDE_percent_calculation_l3840_384051


namespace NUMINAMATH_CALUDE_double_quarter_four_percent_l3840_384007

theorem double_quarter_four_percent : 
  (4 / 100 / 4 * 2 : ℝ) = 0.02 := by sorry

end NUMINAMATH_CALUDE_double_quarter_four_percent_l3840_384007


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3840_384097

def M : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}

def N : Set (ℝ × ℝ) := {p | p.1 = 1}

theorem intersection_of_M_and_N : M ∩ N = {(1, 0)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3840_384097


namespace NUMINAMATH_CALUDE_donovans_test_incorrect_answers_l3840_384060

theorem donovans_test_incorrect_answers :
  ∀ (total : ℕ) (correct : ℕ) (percentage : ℚ),
    correct = 35 →
    percentage = 7292 / 10000 →
    (correct : ℚ) / (total : ℚ) = percentage →
    total - correct = 13 :=
  by sorry

end NUMINAMATH_CALUDE_donovans_test_incorrect_answers_l3840_384060


namespace NUMINAMATH_CALUDE_x_bijective_l3840_384045

def x : ℕ → ℤ
  | 0 => 0
  | n + 1 => 
    let r := (n + 1).log 3 + 1
    let k := (n + 1) / (3^(r-1)) - 1
    if (n + 1) = 3^(r-1) * (3*k + 1) then
      x n + (3^r - 1) / 2
    else if (n + 1) = 3^(r-1) * (3*k + 2) then
      x n - (3^r + 1) / 2
    else
      x n

theorem x_bijective : Function.Bijective x := by sorry

end NUMINAMATH_CALUDE_x_bijective_l3840_384045


namespace NUMINAMATH_CALUDE_base5_98_to_base9_l3840_384034

/-- Converts a base-5 number to decimal --/
def base5ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a decimal number to base-9 --/
def decimalToBase9 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else aux (m / 9) ((m % 9) :: acc)
    aux n []

/-- Theorem: The base-9 representation of 98₍₅₎ is 58₍₉₎ --/
theorem base5_98_to_base9 :
  decimalToBase9 (base5ToDecimal [8, 9]) = [5, 8] :=
sorry

end NUMINAMATH_CALUDE_base5_98_to_base9_l3840_384034


namespace NUMINAMATH_CALUDE_jung_min_wire_purchase_l3840_384019

/-- The length of wire needed to make a regular pentagon with given side length -/
def pentagonWireLength (sideLength : ℝ) : ℝ := 5 * sideLength

/-- The total length of wire bought given the side length of the pentagon and the leftover wire -/
def totalWireBought (sideLength leftover : ℝ) : ℝ := pentagonWireLength sideLength + leftover

theorem jung_min_wire_purchase :
  totalWireBought 13 8 = 73 := by
  sorry

end NUMINAMATH_CALUDE_jung_min_wire_purchase_l3840_384019


namespace NUMINAMATH_CALUDE_pair_sequence_existence_l3840_384017

theorem pair_sequence_existence (n q : ℕ) (h : n > 0) (h2 : q > 0) :
  ∃ (m : ℕ) (seq : List (Fin n × Fin n)),
    m = ⌈(2 * q : ℚ) / n⌉ ∧
    seq.length = m ∧
    seq.Nodup ∧
    (∀ i < m - 1, ∃ x, (seq.get ⟨i, by sorry⟩).1 = x ∨ (seq.get ⟨i, by sorry⟩).2 = x) ∧
    (∀ i < m - 1, (seq.get ⟨i, by sorry⟩).1.val < (seq.get ⟨i + 1, by sorry⟩).1.val) :=
by sorry

end NUMINAMATH_CALUDE_pair_sequence_existence_l3840_384017


namespace NUMINAMATH_CALUDE_completing_square_form_l3840_384079

theorem completing_square_form (x : ℝ) : 
  (x^2 - 6*x - 3 = 0) ↔ ((x - 3)^2 = 12) := by sorry

end NUMINAMATH_CALUDE_completing_square_form_l3840_384079


namespace NUMINAMATH_CALUDE_bees_12_feet_apart_l3840_384015

/-- Represents the position of a bee in 3D space -/
structure Position where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the movement cycle of a bee -/
structure MovementCycle where
  steps : List Position

/-- Calculates the position of a bee after a given number of steps -/
def beePosition (start : Position) (cycle : MovementCycle) (steps : ℕ) : Position :=
  sorry

/-- Calculates the distance between two positions -/
def distance (p1 p2 : Position) : ℝ :=
  sorry

/-- Determines the direction of movement for a bee given its current and next position -/
def movementDirection (current next : Position) : String :=
  sorry

/-- The theorem to be proved -/
theorem bees_12_feet_apart :
  ∀ (steps : ℕ),
  let start := Position.mk 0 0 0
  let cycleA := MovementCycle.mk [Position.mk 2 0 0, Position.mk 0 2 0]
  let cycleB := MovementCycle.mk [Position.mk 0 (-2) 1, Position.mk (-1) 0 0]
  let posA := beePosition start cycleA steps
  let posB := beePosition start cycleB steps
  let nextA := beePosition start cycleA (steps + 1)
  let nextB := beePosition start cycleB (steps + 1)
  distance posA posB = 12 →
  (∀ (s : ℕ), s < steps → distance (beePosition start cycleA s) (beePosition start cycleB s) < 12) →
  movementDirection posA nextA = "east" ∧ movementDirection posB nextB = "upwards" :=
sorry

end NUMINAMATH_CALUDE_bees_12_feet_apart_l3840_384015


namespace NUMINAMATH_CALUDE_parents_per_child_l3840_384076

-- Define the number of girls and boys
def num_girls : ℕ := 6
def num_boys : ℕ := 8

-- Define the total number of parents attending
def total_parents : ℕ := 28

-- Theorem statement
theorem parents_per_child (parents_per_child : ℕ) :
  parents_per_child * num_girls + parents_per_child * num_boys = total_parents →
  parents_per_child = 2 := by
sorry

end NUMINAMATH_CALUDE_parents_per_child_l3840_384076


namespace NUMINAMATH_CALUDE_eighth_term_equals_general_term_l3840_384071

/-- The general term of the sequence -/
def generalTerm (n : ℕ) (a : ℝ) : ℝ := (-1)^n * n^2 * a^(n+1)

/-- The 8th term of the sequence -/
def eighthTerm (a : ℝ) : ℝ := 64 * a^9

theorem eighth_term_equals_general_term : 
  ∀ a : ℝ, generalTerm 8 a = eighthTerm a := by sorry

end NUMINAMATH_CALUDE_eighth_term_equals_general_term_l3840_384071


namespace NUMINAMATH_CALUDE_maple_trees_equation_l3840_384026

/-- The number of maple trees initially in the park -/
def initial_maple_trees : ℕ := 2

/-- The number of maple trees planted -/
def planted_maple_trees : ℕ := 9

/-- The final number of maple trees after planting -/
def final_maple_trees : ℕ := 11

/-- Theorem stating that the initial number of maple trees plus the planted ones equals the final number -/
theorem maple_trees_equation : 
  initial_maple_trees + planted_maple_trees = final_maple_trees := by
  sorry

end NUMINAMATH_CALUDE_maple_trees_equation_l3840_384026


namespace NUMINAMATH_CALUDE_consecutive_integers_problem_l3840_384025

/-- Given consecutive integers x, y, and z where x > y > z, 
    2x + 3y + 3z = 5y + 11, and z = 3, prove that x = 5 -/
theorem consecutive_integers_problem (x y z : ℤ) 
  (consecutive : (x = y + 1) ∧ (y = z + 1))
  (order : x > y ∧ y > z)
  (equation : 2*x + 3*y + 3*z = 5*y + 11)
  (z_value : z = 3) :
  x = 5 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_problem_l3840_384025


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l3840_384041

theorem inscribed_circle_radius (a b c : ℝ) (ha : a = 3) (hb : b = 6) (hc : c = 18) :
  let r := (1 / a + 1 / b + 1 / c + 4 * Real.sqrt (1 / (a * b) + 1 / (a * c) + 1 / (b * c)))⁻¹
  r = 9 / (5 + 6 * Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l3840_384041


namespace NUMINAMATH_CALUDE_common_value_proof_l3840_384036

theorem common_value_proof (a b : ℝ) (h1 : 4 * a = 5 * b) (h2 : 40 * a * b = 1800) :
  4 * a = 60 ∧ 5 * b = 60 := by
sorry

end NUMINAMATH_CALUDE_common_value_proof_l3840_384036


namespace NUMINAMATH_CALUDE_multiplicative_inverse_143_mod_391_l3840_384032

theorem multiplicative_inverse_143_mod_391 :
  ∃ a : ℕ, a < 391 ∧ (143 * a) % 391 = 1 :=
by
  use 28
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_143_mod_391_l3840_384032


namespace NUMINAMATH_CALUDE_inequality_problem_l3840_384096

theorem inequality_problem (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) 
  (h3 : c < d) (h4 : d < 0) : 
  a * c < b * d := by
  sorry

end NUMINAMATH_CALUDE_inequality_problem_l3840_384096


namespace NUMINAMATH_CALUDE_no_perfect_square_in_range_l3840_384052

theorem no_perfect_square_in_range : 
  ∀ m : ℤ, 4 ≤ m ∧ m ≤ 12 → ¬∃ k : ℤ, 2 * m^2 + 3 * m + 2 = k^2 := by
  sorry

#check no_perfect_square_in_range

end NUMINAMATH_CALUDE_no_perfect_square_in_range_l3840_384052


namespace NUMINAMATH_CALUDE_paperboy_delivery_ways_l3840_384006

/-- Represents the number of ways to deliver newspapers to n houses without missing four consecutive houses. -/
def delivery_ways : ℕ → ℕ
  | 0 => 1  -- base case: one way to deliver to zero houses
  | 1 => 2  -- base case: two ways to deliver to one house
  | 2 => 4  -- base case: four ways to deliver to two houses
  | 3 => 8  -- base case: eight ways to deliver to three houses
  | n + 4 => delivery_ways (n + 3) + delivery_ways (n + 2) + delivery_ways (n + 1) + delivery_ways n

/-- Theorem stating that there are 2872 ways for a paperboy to deliver newspapers to 12 houses without missing four consecutive houses. -/
theorem paperboy_delivery_ways :
  delivery_ways 12 = 2872 := by
  sorry

end NUMINAMATH_CALUDE_paperboy_delivery_ways_l3840_384006


namespace NUMINAMATH_CALUDE_equal_numbers_exist_l3840_384062

/-- A quadratic polynomial function -/
def QuadraticPolynomial (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- Theorem: Given a quadratic polynomial and real numbers l, t, v satisfying certain conditions,
    there exist at least two equal numbers among l, t, and v. -/
theorem equal_numbers_exist (a b c l t v : ℝ) (ha : a ≠ 0)
    (h1 : QuadraticPolynomial a b c l = t + v)
    (h2 : QuadraticPolynomial a b c t = l + v)
    (h3 : QuadraticPolynomial a b c v = l + t) :
    (l = t ∨ l = v ∨ t = v) := by
  sorry

end NUMINAMATH_CALUDE_equal_numbers_exist_l3840_384062


namespace NUMINAMATH_CALUDE_sam_exchange_probability_l3840_384082

/-- The number of toys in the vending machine -/
def num_toys : ℕ := 10

/-- The price of the first toy in cents -/
def first_toy_price : ℕ := 50

/-- The price increment between consecutive toys in cents -/
def price_increment : ℕ := 25

/-- The number of quarters Sam has -/
def sam_quarters : ℕ := 10

/-- The price of Sam's favorite toy in cents -/
def favorite_toy_price : ℕ := 225

/-- The total number of possible toy arrangements -/
def total_arrangements : ℕ := Nat.factorial num_toys

/-- The number of favorable arrangements where Sam can buy his favorite toy without exchanging his bill -/
def favorable_arrangements : ℕ := Nat.factorial 9 + Nat.factorial 8 + Nat.factorial 7 + Nat.factorial 6 + Nat.factorial 5

/-- The probability that Sam needs to exchange his bill -/
def exchange_probability : ℚ := 1 - (favorable_arrangements : ℚ) / (total_arrangements : ℚ)

theorem sam_exchange_probability :
  exchange_probability = 8 / 9 := by sorry

end NUMINAMATH_CALUDE_sam_exchange_probability_l3840_384082


namespace NUMINAMATH_CALUDE_f_of_10_l3840_384046

/-- Given a function f(x) = 2x^2 + y where f(2) = 30, prove that f(10) = 222 -/
theorem f_of_10 (f : ℝ → ℝ) (y : ℝ) 
    (h1 : ∀ x, f x = 2 * x^2 + y) 
    (h2 : f 2 = 30) : 
  f 10 = 222 := by
sorry

end NUMINAMATH_CALUDE_f_of_10_l3840_384046


namespace NUMINAMATH_CALUDE_odd_function_property_l3840_384011

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_property (f : ℝ → ℝ) (h : is_odd_function f) :
  ∀ x, f x * f (-x) ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l3840_384011


namespace NUMINAMATH_CALUDE_unique_solution_for_radical_equation_l3840_384063

theorem unique_solution_for_radical_equation (x : ℝ) (h_pos : x > 0) 
  (h_eq : Real.sqrt (12 * x) * Real.sqrt (5 * x) * Real.sqrt (6 * x) * Real.sqrt (10 * x) = 10) : 
  x = 1 / 6 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_for_radical_equation_l3840_384063


namespace NUMINAMATH_CALUDE_part_one_calculation_part_two_calculation_l3840_384066

-- Part I
theorem part_one_calculation : -(-1)^1000 - 2.45 * 8 + 2.55 * (-8) = -41 := by
  sorry

-- Part II
theorem part_two_calculation : (1/6 - 1/3 + 0.25) / (-1/12) = -1 := by
  sorry

end NUMINAMATH_CALUDE_part_one_calculation_part_two_calculation_l3840_384066


namespace NUMINAMATH_CALUDE_sum_of_divisors_900_prime_factors_l3840_384000

def sum_of_divisors (n : ℕ) : ℕ := sorry

theorem sum_of_divisors_900_prime_factors :
  ∃ (p q r : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
  sum_of_divisors 900 = p * q * r ∧
  ∀ (s : ℕ), Nat.Prime s → s ∣ sum_of_divisors 900 → (s = p ∨ s = q ∨ s = r) :=
sorry

end NUMINAMATH_CALUDE_sum_of_divisors_900_prime_factors_l3840_384000


namespace NUMINAMATH_CALUDE_hyperbola_right_angle_triangle_area_l3840_384087

/-- Hyperbola type representing the equation x²/9 - y²/16 = 1 -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  equation : (x : ℝ) → (y : ℝ) → Prop

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Triangle formed by three points -/
structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

/-- The area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- The angle between three points -/
def angle (p1 p2 p3 : Point) : ℝ := sorry

/-- Distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Check if a point is on the hyperbola -/
def isOnHyperbola (h : Hyperbola) (p : Point) : Prop := h.equation p.x p.y

/-- The foci of a hyperbola -/
def foci (h : Hyperbola) : (Point × Point) := sorry

theorem hyperbola_right_angle_triangle_area 
  (h : Hyperbola) 
  (p : Point) 
  (hP : isOnHyperbola h p) 
  (f1 f2 : Point) 
  (hFoci : foci h = (f1, f2)) 
  (hAngle : angle f1 p f2 = 90) : 
  triangleArea (Triangle.mk f1 p f2) = 16 := by sorry

end NUMINAMATH_CALUDE_hyperbola_right_angle_triangle_area_l3840_384087


namespace NUMINAMATH_CALUDE_min_value_theorem_l3840_384091

theorem min_value_theorem (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (h : x + 2*y = 1) :
  (2/x + 3/y) ≥ 8 + 4*Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3840_384091


namespace NUMINAMATH_CALUDE_zeros_in_concatenated_number_l3840_384018

/-- Counts the number of zeros in a given positive integer -/
def countZeros (n : ℕ) : ℕ := sorry

/-- Counts the total number of zeros in all integers from 1 to n -/
def totalZeros (n : ℕ) : ℕ := sorry

/-- The concatenated number formed by all integers from 1 to 2007 -/
def concatenatedNumber : ℕ := sorry

theorem zeros_in_concatenated_number :
  countZeros concatenatedNumber = 506 := by sorry

end NUMINAMATH_CALUDE_zeros_in_concatenated_number_l3840_384018


namespace NUMINAMATH_CALUDE_simplify_expression_l3840_384093

theorem simplify_expression : 5 * (18 / 6) * (21 / -63) = -5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3840_384093


namespace NUMINAMATH_CALUDE_certain_number_proof_l3840_384059

theorem certain_number_proof : ∃ x : ℕ, 865 * 48 = 173 * x ∧ x = 240 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3840_384059


namespace NUMINAMATH_CALUDE_cube_face_diagonal_edge_angle_l3840_384099

/-- Represents a cube in 3D space -/
structure Cube where
  -- Define necessary properties of a cube

/-- Represents a line segment in 3D space -/
structure LineSegment where
  -- Define necessary properties of a line segment

/-- Represents an angle between two line segments -/
def angle (l1 l2 : LineSegment) : ℝ := sorry

/-- Predicate to check if a line segment is an edge of the cube -/
def is_edge (c : Cube) (l : LineSegment) : Prop := sorry

/-- Predicate to check if a line segment is a face diagonal of the cube -/
def is_face_diagonal (c : Cube) (l : LineSegment) : Prop := sorry

/-- Predicate to check if two line segments are incident to the same vertex -/
def incident_to_same_vertex (l1 l2 : LineSegment) : Prop := sorry

/-- Theorem: In a cube, the angle between a face diagonal and an edge 
    incident to the same vertex is 60 degrees -/
theorem cube_face_diagonal_edge_angle (c : Cube) (d e : LineSegment) :
  is_face_diagonal c d → is_edge c e → incident_to_same_vertex d e →
  angle d e = 60 := by sorry

end NUMINAMATH_CALUDE_cube_face_diagonal_edge_angle_l3840_384099


namespace NUMINAMATH_CALUDE_four_points_reciprocal_sum_l3840_384027

theorem four_points_reciprocal_sum (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) :
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 
    1 / |x - a| + 1 / |x - b| + 1 / |x - c| + 1 / |x - d| ≤ 40 := by
  sorry

end NUMINAMATH_CALUDE_four_points_reciprocal_sum_l3840_384027


namespace NUMINAMATH_CALUDE_cara_seating_arrangements_l3840_384012

theorem cara_seating_arrangements (n : ℕ) (h : n = 8) :
  (n - 2 : ℕ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_cara_seating_arrangements_l3840_384012


namespace NUMINAMATH_CALUDE_sum_of_roots_squared_equation_l3840_384088

theorem sum_of_roots_squared_equation (x : ℝ) :
  (∀ x, (x - 4)^2 = 16 ↔ x = 8 ∨ x = 0) →
  (∃ a b : ℝ, (a - 4)^2 = 16 ∧ (b - 4)^2 = 16 ∧ a + b = 8) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_squared_equation_l3840_384088


namespace NUMINAMATH_CALUDE_last_term_is_1344_l3840_384016

/-- Defines the nth term of the sequence -/
def sequenceTerm (n : ℕ) : ℕ :=
  if n % 3 = 1 then (n + 2) / 3 else (n + 1) / 3

/-- The last term of the sequence with 2015 elements -/
def lastTerm : ℕ := sequenceTerm 2015

theorem last_term_is_1344 : lastTerm = 1344 := by
  sorry

end NUMINAMATH_CALUDE_last_term_is_1344_l3840_384016


namespace NUMINAMATH_CALUDE_coaches_next_meeting_l3840_384086

theorem coaches_next_meeting (ella_schedule : Nat) (felix_schedule : Nat) (greta_schedule : Nat) (harry_schedule : Nat)
  (h_ella : ella_schedule = 5)
  (h_felix : felix_schedule = 9)
  (h_greta : greta_schedule = 8)
  (h_harry : harry_schedule = 11) :
  Nat.lcm (Nat.lcm (Nat.lcm ella_schedule felix_schedule) greta_schedule) harry_schedule = 3960 := by
sorry

end NUMINAMATH_CALUDE_coaches_next_meeting_l3840_384086


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3840_384013

/-- A function f is odd if f(-x) = -f(x) for all x -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function f is increasing on a set S if f(x) ≤ f(y) for all x, y in S with x ≤ y -/
def IncreasingOn (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → x ≤ y → f x ≤ f y

theorem inequality_solution_set
  (f : ℝ → ℝ)
  (h_odd : OddFunction f)
  (h_incr : IncreasingOn f (Set.Iic 0))
  (h_f2 : f 2 = 4) :
  {x : ℝ | 4 + f (x^2 - x) > 0} = Set.univ :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3840_384013


namespace NUMINAMATH_CALUDE_intersection_line_through_origin_l3840_384033

/-- Given two lines in the plane, this theorem proves that a specific line
    passes through their intersection point and the origin. -/
theorem intersection_line_through_origin :
  let line1 : ℝ → ℝ → Prop := λ x y => 2023 * x - 2022 * y - 1 = 0
  let line2 : ℝ → ℝ → Prop := λ x y => 2022 * x + 2023 * y + 1 = 0
  let intersection_line : ℝ → ℝ → Prop := λ x y => 4045 * x + y = 0
  (∃ x y, line1 x y ∧ line2 x y) →  -- Assumption: The two lines intersect
  (∀ x y, line1 x y ∧ line2 x y → intersection_line x y) ∧  -- The line passes through the intersection
  intersection_line 0 0  -- The line passes through the origin
  := by sorry

end NUMINAMATH_CALUDE_intersection_line_through_origin_l3840_384033


namespace NUMINAMATH_CALUDE_price_decrease_unit_increase_ratio_l3840_384080

theorem price_decrease_unit_increase_ratio (P U V : ℝ) 
  (h1 : P > 0) 
  (h2 : U > 0) 
  (h3 : V > U) 
  (h4 : P * U = 0.25 * P * V) : 
  ((V - U) / U) / 0.75 = 4 := by
sorry

end NUMINAMATH_CALUDE_price_decrease_unit_increase_ratio_l3840_384080


namespace NUMINAMATH_CALUDE_tooth_fairy_total_l3840_384039

/-- The total number of baby teeth a child has. -/
def totalTeeth : ℕ := 20

/-- The number of teeth lost or swallowed. -/
def lostTeeth : ℕ := 2

/-- The amount received for the first tooth. -/
def firstToothAmount : ℕ := 20

/-- The amount received for each subsequent tooth. -/
def regularToothAmount : ℕ := 2

/-- The total amount received from the tooth fairy. -/
def totalAmount : ℕ := firstToothAmount + regularToothAmount * (totalTeeth - lostTeeth - 1)

theorem tooth_fairy_total : totalAmount = 54 := by
  sorry

end NUMINAMATH_CALUDE_tooth_fairy_total_l3840_384039


namespace NUMINAMATH_CALUDE_three_digit_distinct_sum_remainder_l3840_384067

def S : ℕ := sorry

theorem three_digit_distinct_sum_remainder : S % 1000 = 680 := by sorry

end NUMINAMATH_CALUDE_three_digit_distinct_sum_remainder_l3840_384067


namespace NUMINAMATH_CALUDE_wine_price_increase_l3840_384031

/-- Proves that the percentage increase in wine price is 25% given the initial and future prices -/
theorem wine_price_increase (initial_price : ℝ) (future_price_increase : ℝ) : 
  initial_price = 20 →
  future_price_increase = 25 →
  (((initial_price + future_price_increase / 5) - initial_price) / initial_price) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_wine_price_increase_l3840_384031


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l3840_384098

theorem sum_of_two_numbers (x y : ℝ) : 
  (0.45 * x = 2700) → (y = 2 * x) → (x + y = 18000) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l3840_384098


namespace NUMINAMATH_CALUDE_car_speed_problem_l3840_384028

theorem car_speed_problem (v : ℝ) : v > 0 →
  (1 / v * 3600 = 1 / 120 * 3600 + 2) ↔ v = 112.5 :=
by sorry

end NUMINAMATH_CALUDE_car_speed_problem_l3840_384028


namespace NUMINAMATH_CALUDE_odometer_sum_l3840_384061

theorem odometer_sum (a b c : ℕ) : 
  a ≥ 1 → 
  a + b + c ≤ 9 → 
  (100 * c + 10 * a + b) - (100 * a + 10 * b + c) % 45 = 0 →
  100 * a + 10 * b + c + 100 * b + 10 * c + a + 100 * c + 10 * a + b = 999 :=
by sorry

end NUMINAMATH_CALUDE_odometer_sum_l3840_384061


namespace NUMINAMATH_CALUDE_percentage_problem_l3840_384081

theorem percentage_problem (P : ℝ) : 
  (0.15 * P * (0.5 * 4800) = 108) → P = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3840_384081


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l3840_384022

/-- Represents the number of volunteers in each grade --/
structure GradeVolunteers where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Represents the number of volunteers selected in the sample from each grade --/
structure SampleVolunteers where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Calculates the probability of selecting two volunteers from the same grade --/
def probability_same_grade (sample : SampleVolunteers) : ℚ :=
  let total_pairs := sample.first.choose 2 + sample.second.choose 2
  let all_pairs := (sample.first + sample.second).choose 2
  total_pairs / all_pairs

theorem stratified_sampling_theorem (volunteers : GradeVolunteers) (sample : SampleVolunteers) :
  volunteers.first = 36 →
  volunteers.second = 72 →
  volunteers.third = 54 →
  sample.third = 3 →
  sample.first = 2 ∧
  sample.second = 4 ∧
  probability_same_grade sample = 7/15 := by
  sorry

#check stratified_sampling_theorem

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l3840_384022


namespace NUMINAMATH_CALUDE_tetrahedron_volume_in_cube_l3840_384023

/-- The volume of a tetrahedron formed by alternately colored vertices of a cube -/
theorem tetrahedron_volume_in_cube (cube_side_length : ℝ) (h : cube_side_length = 8) :
  let cube_volume : ℝ := cube_side_length ^ 3
  let clear_tetrahedron_volume : ℝ := (1 / 6) * cube_side_length ^ 3
  let colored_tetrahedron_volume : ℝ := cube_volume - 4 * clear_tetrahedron_volume
  colored_tetrahedron_volume = 172 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_in_cube_l3840_384023


namespace NUMINAMATH_CALUDE_alices_favorite_number_l3840_384009

def is_multiple (a b : ℕ) : Prop := ∃ k, a = b * k

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

theorem alices_favorite_number : ∃! n : ℕ, 
  90 < n ∧ n < 150 ∧ 
  is_multiple n 13 ∧ 
  ¬is_multiple n 3 ∧ 
  is_multiple (digit_sum n) 4 ∧
  n = 130 := by sorry

end NUMINAMATH_CALUDE_alices_favorite_number_l3840_384009


namespace NUMINAMATH_CALUDE_sat_score_improvement_l3840_384064

theorem sat_score_improvement (first_score second_score : ℝ) : 
  (second_score = first_score * 1.1) → 
  (second_score = 1100) → 
  (first_score = 1000) := by
sorry

end NUMINAMATH_CALUDE_sat_score_improvement_l3840_384064


namespace NUMINAMATH_CALUDE_distance_between_points_l3840_384048

def point_A : ℝ × ℝ := (2, 3)
def point_B : ℝ × ℝ := (5, 10)

theorem distance_between_points :
  Real.sqrt ((point_B.1 - point_A.1)^2 + (point_B.2 - point_A.2)^2) = Real.sqrt 58 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l3840_384048


namespace NUMINAMATH_CALUDE_al_mass_percentage_l3840_384042

theorem al_mass_percentage (mass_percentage : ℝ) (h : mass_percentage = 20.45) :
  mass_percentage = 20.45 := by
sorry

end NUMINAMATH_CALUDE_al_mass_percentage_l3840_384042


namespace NUMINAMATH_CALUDE_smaller_number_proof_l3840_384024

theorem smaller_number_proof (x y : ℝ) : 
  x + y = 70 ∧ y = 3 * x + 10 → x = 15 ∧ x ≤ y := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l3840_384024


namespace NUMINAMATH_CALUDE_circumradius_right_triangle_l3840_384002

/-- The radius of the circumscribed circle of a right triangle with sides 10, 8, and 6 is 5 -/
theorem circumradius_right_triangle : 
  ∀ (a b c : ℝ), 
    a = 10 → b = 8 → c = 6 →
    a^2 = b^2 + c^2 →
    (a / 2 : ℝ) = 5 :=
by sorry

end NUMINAMATH_CALUDE_circumradius_right_triangle_l3840_384002


namespace NUMINAMATH_CALUDE_favorite_movies_sum_l3840_384056

/-- Given the movie lengths of Joyce, Michael, Nikki, and Ryn, prove their sum is 76 hours -/
theorem favorite_movies_sum (michael nikki joyce ryn : ℝ) : 
  nikki = 30 ∧ 
  michael = nikki / 3 ∧ 
  joyce = michael + 2 ∧ 
  ryn = 4 / 5 * nikki → 
  joyce + michael + nikki + ryn = 76 := by
  sorry

end NUMINAMATH_CALUDE_favorite_movies_sum_l3840_384056


namespace NUMINAMATH_CALUDE_product_pure_imaginary_l3840_384089

theorem product_pure_imaginary (x : ℝ) :
  (∃ b : ℝ, (x + 1 + Complex.I) * ((x + 2) + Complex.I) * ((x + 3) + Complex.I) = b * Complex.I) ↔ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_product_pure_imaginary_l3840_384089


namespace NUMINAMATH_CALUDE_abs_z_equals_sqrt_5_l3840_384047

-- Define the complex number z
def z : ℂ := -Complex.I * (1 + 2 * Complex.I)

-- Theorem stating that the absolute value of z is √5
theorem abs_z_equals_sqrt_5 : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_abs_z_equals_sqrt_5_l3840_384047


namespace NUMINAMATH_CALUDE_cauchy_schwarz_inequality_l3840_384072

theorem cauchy_schwarz_inequality (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  a * c + b * d ≤ Real.sqrt ((a^2 + b^2) * (c^2 + d^2)) := by
  sorry

end NUMINAMATH_CALUDE_cauchy_schwarz_inequality_l3840_384072


namespace NUMINAMATH_CALUDE_allowance_spent_on_books_l3840_384090

theorem allowance_spent_on_books (total : ℚ) (games snacks toys books : ℚ) : 
  total = 45 → 
  games = 2/9 * total → 
  snacks = 1/3 * total → 
  toys = 1/5 * total → 
  books = total - (games + snacks + toys) → 
  books = 11 := by
sorry

end NUMINAMATH_CALUDE_allowance_spent_on_books_l3840_384090


namespace NUMINAMATH_CALUDE_power_of_two_geq_n_plus_one_l3840_384035

theorem power_of_two_geq_n_plus_one (n : ℕ) (h : n ≥ 1) : 2^n ≥ n + 1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_geq_n_plus_one_l3840_384035


namespace NUMINAMATH_CALUDE_equation_solution_l3840_384050

theorem equation_solution : ∃ x : ℝ, (4 / 7) * (1 / 8) * x = 12 ∧ x = 168 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3840_384050


namespace NUMINAMATH_CALUDE_remainder_4_100_div_9_l3840_384005

theorem remainder_4_100_div_9 : (4^100) % 9 = 4 := by sorry

end NUMINAMATH_CALUDE_remainder_4_100_div_9_l3840_384005


namespace NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l3840_384095

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Function to check if a point lies on a line
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to check if a line has equal intercepts on x and y axes
def equalIntercepts (l : Line2D) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ l.c / l.a = -l.c / l.b

-- The main theorem
theorem line_through_point_with_equal_intercepts :
  ∃ (l1 l2 : Line2D),
    pointOnLine ⟨1, 2⟩ l1 ∧
    pointOnLine ⟨1, 2⟩ l2 ∧
    equalIntercepts l1 ∧
    equalIntercepts l2 ∧
    ((l1.a = 1 ∧ l1.b = 1 ∧ l1.c = -3) ∨ (l2.a = 2 ∧ l2.b = -1 ∧ l2.c = 0)) :=
by sorry

end NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l3840_384095


namespace NUMINAMATH_CALUDE_solution_approximation_l3840_384029

def equation (x : ℝ) : Prop :=
  (0.66^3 - x^3) = 0.5599999999999999 * ((0.66^2) + 0.066 + x^2)

theorem solution_approximation : ∃ x : ℝ, equation x ∧ abs (x - 0.1) < 1e-6 := by
  sorry

end NUMINAMATH_CALUDE_solution_approximation_l3840_384029


namespace NUMINAMATH_CALUDE_expression_evaluation_l3840_384057

theorem expression_evaluation :
  let a : ℤ := -3
  let b : ℤ := -2
  (3 * a^2 * b + 2 * a * b^2) - (2 * (a^2 * b - 1) + 3 * a * b^2 + 2) = -6 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3840_384057


namespace NUMINAMATH_CALUDE_divisibility_implies_equality_l3840_384074

theorem divisibility_implies_equality (a b n : ℕ) :
  (∀ k : ℕ, k ≠ b → (b - k) ∣ (a - k^n)) →
  a = b^n :=
by sorry

end NUMINAMATH_CALUDE_divisibility_implies_equality_l3840_384074


namespace NUMINAMATH_CALUDE_theater_pricing_l3840_384053

/-- The price of orchestra seats in dollars -/
def orchestra_price : ℝ := 12

/-- The total number of tickets sold -/
def total_tickets : ℕ := 380

/-- The total revenue in dollars -/
def total_revenue : ℝ := 3320

/-- The difference between balcony and orchestra tickets sold -/
def ticket_difference : ℕ := 240

/-- The price of balcony seats in dollars -/
def balcony_price : ℝ := 8

theorem theater_pricing :
  ∃ (orchestra_tickets : ℕ) (balcony_tickets : ℕ),
    orchestra_tickets + balcony_tickets = total_tickets ∧
    balcony_tickets = orchestra_tickets + ticket_difference ∧
    orchestra_price * orchestra_tickets + balcony_price * balcony_tickets = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_theater_pricing_l3840_384053


namespace NUMINAMATH_CALUDE_randy_biscuits_l3840_384070

/-- The number of biscuits Randy has after receiving and losing some -/
def final_biscuits (initial : ℕ) (from_father : ℕ) (from_mother : ℕ) (eaten_by_brother : ℕ) : ℕ :=
  initial + from_father + from_mother - eaten_by_brother

/-- Theorem stating that Randy ends up with 40 biscuits -/
theorem randy_biscuits : final_biscuits 32 13 15 20 = 40 := by
  sorry

end NUMINAMATH_CALUDE_randy_biscuits_l3840_384070


namespace NUMINAMATH_CALUDE_rectangular_prism_dimensions_l3840_384073

theorem rectangular_prism_dimensions :
  ∀ (l b h : ℝ),
    l = 3 * b →
    l = 2 * h →
    l * b * h = 12168 →
    l = 42 ∧ b = 14 ∧ h = 21 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_dimensions_l3840_384073


namespace NUMINAMATH_CALUDE_airline_route_theorem_l3840_384030

/-- Represents a city in the country -/
structure City where
  id : Nat
  republic : Nat
  routes : Finset Nat

/-- The country with its cities and airline routes -/
structure Country where
  cities : Finset City
  total_cities : Nat
  num_republics : Nat

/-- A country satisfies the problem conditions -/
def satisfies_conditions (country : Country) : Prop :=
  country.total_cities = 100 ∧
  country.num_republics = 3 ∧
  (country.cities.filter (λ c => c.routes.card ≥ 70)).card ≥ 70

/-- There exists an airline route within the same republic -/
def exists_intra_republic_route (country : Country) : Prop :=
  ∃ c1 c2 : City, c1 ∈ country.cities ∧ c2 ∈ country.cities ∧
    c1.id ≠ c2.id ∧ c1.republic = c2.republic ∧ c2.id ∈ c1.routes

/-- The main theorem -/
theorem airline_route_theorem (country : Country) :
  satisfies_conditions country → exists_intra_republic_route country :=
by
  sorry


end NUMINAMATH_CALUDE_airline_route_theorem_l3840_384030


namespace NUMINAMATH_CALUDE_balloon_count_l3840_384085

/-- The number of violet balloons Dan has -/
def dans_balloons : ℕ := 29

/-- The number of times more balloons Tim has compared to Dan -/
def tims_multiplier : ℕ := 7

/-- The number of times more balloons Molly has compared to Dan -/
def mollys_multiplier : ℕ := 5

/-- The total number of violet balloons Dan, Tim, and Molly have -/
def total_balloons : ℕ := dans_balloons + tims_multiplier * dans_balloons + mollys_multiplier * dans_balloons

theorem balloon_count : total_balloons = 377 := by sorry

end NUMINAMATH_CALUDE_balloon_count_l3840_384085


namespace NUMINAMATH_CALUDE_triangle_decomposition_l3840_384014

theorem triangle_decomposition (a b c : ℝ) 
  (h1 : b + c > a) (h2 : a + c > b) (h3 : a + b > c) :
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ 
    a = y + z ∧ b = x + z ∧ c = x + y :=
by sorry

end NUMINAMATH_CALUDE_triangle_decomposition_l3840_384014


namespace NUMINAMATH_CALUDE_moving_circle_trajectory_l3840_384049

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x + 5)^2 + y^2 = 16
def C₂ (x y : ℝ) : Prop := (x - 5)^2 + y^2 = 16

-- Define the moving circle M
structure MovingCircle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the tangency conditions
def externally_tangent (M : MovingCircle) : Prop :=
  C₁ (M.center.1 + M.radius) M.center.2

def internally_tangent (M : MovingCircle) : Prop :=
  C₂ (M.center.1 - M.radius) M.center.2

-- State the theorem
theorem moving_circle_trajectory
  (M : MovingCircle)
  (h1 : externally_tangent M)
  (h2 : internally_tangent M) :
  ∃ x y : ℝ, x > 0 ∧ x^2 / 16 - y^2 / 9 = 1 ∧ M.center = (x, y) :=
sorry

end NUMINAMATH_CALUDE_moving_circle_trajectory_l3840_384049


namespace NUMINAMATH_CALUDE_sqrt_sum_difference_equals_four_sqrt_three_plus_one_l3840_384078

theorem sqrt_sum_difference_equals_four_sqrt_three_plus_one :
  Real.sqrt 12 + Real.sqrt 27 - |1 - Real.sqrt 3| = 4 * Real.sqrt 3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_difference_equals_four_sqrt_three_plus_one_l3840_384078


namespace NUMINAMATH_CALUDE_base_conversion_512_to_octal_l3840_384092

theorem base_conversion_512_to_octal :
  (512 : ℕ) = 1 * 8^3 + 0 * 8^2 + 0 * 8^1 + 0 * 8^0 :=
by sorry

end NUMINAMATH_CALUDE_base_conversion_512_to_octal_l3840_384092


namespace NUMINAMATH_CALUDE_compound_interest_rate_l3840_384054

theorem compound_interest_rate (P : ℝ) (r : ℝ) 
  (h1 : P * (1 + r)^2 = 2420)
  (h2 : P * (1 + r)^3 = 3025) : 
  r = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_rate_l3840_384054


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3840_384001

-- Problem 1
theorem problem_1 : 
  (3 * Real.sqrt 18 + (1/6) * Real.sqrt 72 - 4 * Real.sqrt (1/8)) / (4 * Real.sqrt 2) = 9/4 := by
  sorry

-- Problem 2
theorem problem_2 : 
  let x : ℝ := Real.sqrt 2 + 1
  ((x + 2) / (x * (x - 1)) - 1 / (x - 1)) * (x / (x - 1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3840_384001


namespace NUMINAMATH_CALUDE_predicted_holiday_shoppers_l3840_384077

theorem predicted_holiday_shoppers 
  (packages_per_box : ℕ) 
  (boxes_ordered : ℕ) 
  (shopper_ratio : ℕ) 
  (h1 : packages_per_box = 25)
  (h2 : boxes_ordered = 5)
  (h3 : shopper_ratio = 3) :
  boxes_ordered * packages_per_box * shopper_ratio = 375 :=
by sorry

end NUMINAMATH_CALUDE_predicted_holiday_shoppers_l3840_384077


namespace NUMINAMATH_CALUDE_tray_height_proof_l3840_384021

/-- Given a square with side length 150 and cuts starting 8 units from each corner
    meeting at a 45° angle on the diagonal, the height of the resulting tray when folded
    is equal to the fourth root of 4096. -/
theorem tray_height_proof (side_length : ℝ) (cut_distance : ℝ) (cut_angle : ℝ) :
  side_length = 150 →
  cut_distance = 8 →
  cut_angle = 45 →
  ∃ (h : ℝ), h = (8 * Real.sqrt 2 - 8) ∧ h^4 = 4096 :=
by sorry

end NUMINAMATH_CALUDE_tray_height_proof_l3840_384021


namespace NUMINAMATH_CALUDE_boys_in_row_l3840_384069

theorem boys_in_row (left_position right_position between : ℕ) : 
  left_position = 6 →
  right_position = 10 →
  between = 8 →
  left_position - 1 + between + right_position = 24 :=
by sorry

end NUMINAMATH_CALUDE_boys_in_row_l3840_384069


namespace NUMINAMATH_CALUDE_angle_W_measure_l3840_384075

-- Define the quadrilateral WXYZ
structure Quadrilateral :=
  (W X Y Z : ℝ)

-- Define the properties of the quadrilateral
def is_valid_quadrilateral (q : Quadrilateral) : Prop :=
  q.W = 3 * q.X ∧ q.W = 2 * q.Y ∧ q.W = 4 * q.Z ∧
  q.W + q.X + q.Y + q.Z = 360

-- Theorem statement
theorem angle_W_measure (q : Quadrilateral) 
  (h : is_valid_quadrilateral q) : q.W = 172.8 := by
  sorry

end NUMINAMATH_CALUDE_angle_W_measure_l3840_384075


namespace NUMINAMATH_CALUDE_stratified_sample_probability_l3840_384044

/-- Represents the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The probability of an event -/
def probability (favorable_outcomes total_outcomes : ℕ) : ℚ := sorry

theorem stratified_sample_probability : 
  let total_sample_size := 6
  let elementary_teachers_in_sample := 3
  let further_selection_size := 2
  probability (choose elementary_teachers_in_sample further_selection_size) 
              (choose total_sample_size further_selection_size) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_probability_l3840_384044


namespace NUMINAMATH_CALUDE_opposite_property_opposite_of_neg_two_l3840_384008

/-- The opposite of a real number -/
def opposite (x : ℝ) : ℝ := -x

/-- The property that defines the opposite of a number -/
theorem opposite_property (x : ℝ) : x + opposite x = 0 := by sorry

/-- Proof that the opposite of -2 is 2 -/
theorem opposite_of_neg_two :
  opposite (-2 : ℝ) = 2 := by sorry

end NUMINAMATH_CALUDE_opposite_property_opposite_of_neg_two_l3840_384008


namespace NUMINAMATH_CALUDE_polygon_diagonals_l3840_384084

theorem polygon_diagonals (n : ℕ) (h : (n - 2) * 180 + 360 = 2160) : 
  n * (n - 3) / 2 = 54 := by sorry

end NUMINAMATH_CALUDE_polygon_diagonals_l3840_384084


namespace NUMINAMATH_CALUDE_twirly_tea_cups_capacity_l3840_384004

/-- The 'Twirly Tea Cups' ride capacity problem -/
theorem twirly_tea_cups_capacity 
  (people_per_teacup : ℕ) 
  (number_of_teacups : ℕ) 
  (h1 : people_per_teacup = 9)
  (h2 : number_of_teacups = 7) : 
  people_per_teacup * number_of_teacups = 63 := by
  sorry

end NUMINAMATH_CALUDE_twirly_tea_cups_capacity_l3840_384004


namespace NUMINAMATH_CALUDE_lighthouse_distance_l3840_384094

/-- Proves that in a triangle ABS with given side length and angles, BS = 72 km -/
theorem lighthouse_distance (AB : ℝ) (angle_A angle_B : ℝ) :
  AB = 36 * Real.sqrt 6 →
  angle_A = 45 * π / 180 →
  angle_B = 75 * π / 180 →
  let angle_S := π - (angle_A + angle_B)
  let BS := AB * Real.sin angle_A / Real.sin angle_S
  BS = 72 := by sorry

end NUMINAMATH_CALUDE_lighthouse_distance_l3840_384094


namespace NUMINAMATH_CALUDE_sin_two_x_value_l3840_384068

theorem sin_two_x_value (x : ℝ) (h : Real.sin (π / 4 - x) = 1 / 3) : 
  Real.sin (2 * x) = 7 / 9 := by
sorry

end NUMINAMATH_CALUDE_sin_two_x_value_l3840_384068


namespace NUMINAMATH_CALUDE_normal_distribution_two_std_below_mean_l3840_384055

theorem normal_distribution_two_std_below_mean :
  let μ : ℝ := 16.2  -- mean
  let σ : ℝ := 2.3   -- standard deviation
  let x : ℝ := μ - 2 * σ  -- value 2 standard deviations below mean
  x = 11.6 := by
  sorry

end NUMINAMATH_CALUDE_normal_distribution_two_std_below_mean_l3840_384055


namespace NUMINAMATH_CALUDE_f_decreasing_iff_a_in_range_l3840_384058

/-- The function f(x) defined as 2ax² + 4(a-3)x + 5 -/
def f (a : ℝ) (x : ℝ) : ℝ := 2*a*x^2 + 4*(a-3)*x + 5

/-- The property of f(x) being decreasing on the interval (-∞, 3) -/
def is_decreasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, x < y → x < 3 → y < 3 → f a x > f a y

/-- The theorem stating the range of a for which f(x) is decreasing on (-∞, 3) -/
theorem f_decreasing_iff_a_in_range :
  ∀ a, is_decreasing_on_interval a ↔ a ∈ Set.Icc 0 (3/4) :=
sorry

end NUMINAMATH_CALUDE_f_decreasing_iff_a_in_range_l3840_384058


namespace NUMINAMATH_CALUDE_power_calculation_l3840_384040

theorem power_calculation : 3^2022 * (1/3)^2023 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l3840_384040


namespace NUMINAMATH_CALUDE_certain_number_problem_l3840_384003

theorem certain_number_problem : ∃! x : ℝ, ((x - 50) / 4) * 3 + 28 = 73 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l3840_384003
