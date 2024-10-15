import Mathlib

namespace NUMINAMATH_CALUDE_golf_rounds_l380_38083

theorem golf_rounds (n : ℕ) (average_score : ℚ) (new_score : ℚ) (drop : ℚ) : 
  average_score = 78 →
  new_score = 68 →
  drop = 2 →
  (n * average_score + new_score) / (n + 1) = average_score - drop →
  n = 4 := by
sorry

end NUMINAMATH_CALUDE_golf_rounds_l380_38083


namespace NUMINAMATH_CALUDE_equation_solution_l380_38004

theorem equation_solution : ∃! (x : ℝ), x ≠ 0 ∧ (6 * x)^18 = (12 * x)^9 ∧ x = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l380_38004


namespace NUMINAMATH_CALUDE_base5_arithmetic_sequence_implies_xyz_decimal_l380_38078

/-- Converts a base-5 number to decimal -/
def toDecimal (a b c : Nat) : Nat :=
  a * 25 + b * 5 + c

/-- Checks if a number is a valid base-5 digit -/
def isBase5Digit (n : Nat) : Prop :=
  n ≥ 0 ∧ n < 5

theorem base5_arithmetic_sequence_implies_xyz_decimal (V W X Y Z : Nat) :
  isBase5Digit V ∧ isBase5Digit W ∧ isBase5Digit X ∧ isBase5Digit Y ∧ isBase5Digit Z →
  toDecimal V Y X = toDecimal V Y Z + 1 →
  toDecimal V V W = toDecimal V Y X + 1 →
  toDecimal X Y Z = 108 := by
  sorry

end NUMINAMATH_CALUDE_base5_arithmetic_sequence_implies_xyz_decimal_l380_38078


namespace NUMINAMATH_CALUDE_hyperbola_focus_to_parabola_l380_38066

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

/-- The right focus of the hyperbola -/
def right_focus : ℝ × ℝ := (2, 0)

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y^2 = 16*x

/-- 
Given a hyperbola with equation x^2 - y^2/3 = 1 and right focus F(2,0),
the standard equation of the parabola with focus F is y^2 = 16x.
-/
theorem hyperbola_focus_to_parabola :
  ∀ x y : ℝ, hyperbola x y → parabola x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_focus_to_parabola_l380_38066


namespace NUMINAMATH_CALUDE_A_intersect_B_l380_38015

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := {x : ℝ | x^2 - x ≤ 0}

theorem A_intersect_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l380_38015


namespace NUMINAMATH_CALUDE_one_non_negative_root_l380_38046

theorem one_non_negative_root (a : ℝ) : 
  (∃ x : ℝ, x ≥ 0 ∧ (x = a + Real.sqrt (a^2 - 4*a + 3) ∨ x = a - Real.sqrt (a^2 - 4*a + 3)) ∧
   ¬∃ y : ℝ, y ≠ x ∧ y ≥ 0 ∧ (y = a + Real.sqrt (a^2 - 4*a + 3) ∨ y = a - Real.sqrt (a^2 - 4*a + 3))) ↔ 
  ((3/4 ≤ a ∧ a < 1) ∨ (a > 3) ∨ (0 < a ∧ a < 3/4)) :=
by sorry

end NUMINAMATH_CALUDE_one_non_negative_root_l380_38046


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_length_l380_38013

theorem isosceles_triangle_side_length 
  (equilateral_side : ℝ) 
  (isosceles_base : ℝ) 
  (equilateral_area : ℝ) 
  (isosceles_area : ℝ) :
  equilateral_side = 1 →
  isosceles_base = 1/3 →
  equilateral_area = Real.sqrt 3 / 4 →
  isosceles_area = equilateral_area / 3 →
  ∃ (isosceles_side : ℝ), 
    isosceles_side = Real.sqrt 3 / 3 ∧ 
    isosceles_side^2 = (isosceles_base/2)^2 + (2 * isosceles_area / isosceles_base)^2 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_side_length_l380_38013


namespace NUMINAMATH_CALUDE_triangle_perimeter_l380_38014

theorem triangle_perimeter (a b c : ℕ) : 
  a = 2 → b = 3 → Odd c → a + b > c → b + c > a → c + a > b → a + b + c = 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l380_38014


namespace NUMINAMATH_CALUDE_base8_253_to_base10_l380_38043

-- Define a function to convert a base 8 number to base 10
def base8ToBase10 (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let units := n % 10
  hundreds * 8^2 + tens * 8^1 + units * 8^0

-- Theorem statement
theorem base8_253_to_base10 : base8ToBase10 253 = 171 := by
  sorry

end NUMINAMATH_CALUDE_base8_253_to_base10_l380_38043


namespace NUMINAMATH_CALUDE_triangle_side_length_l380_38064

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  A = 45 * π / 180 →
  B = 60 * π / 180 →
  a = 10 →
  b = 5 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l380_38064


namespace NUMINAMATH_CALUDE_cube_difference_equals_negative_875_l380_38050

theorem cube_difference_equals_negative_875 (x y : ℝ) 
  (h1 : x + y = 15) 
  (h2 : 2 * x + y = 20) : 
  x^3 - y^3 = -875 := by sorry

end NUMINAMATH_CALUDE_cube_difference_equals_negative_875_l380_38050


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l380_38099

theorem fraction_to_decimal : (17 : ℚ) / (2^2 * 5^4) = (68 : ℚ) / 10000 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l380_38099


namespace NUMINAMATH_CALUDE_marbles_difference_l380_38086

def initial_marbles : ℕ := 7
def lost_marbles : ℕ := 8
def found_marbles : ℕ := 10

theorem marbles_difference : found_marbles - lost_marbles = 2 := by
  sorry

end NUMINAMATH_CALUDE_marbles_difference_l380_38086


namespace NUMINAMATH_CALUDE_complex_multiplication_complex_division_l380_38047

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Theorem 1
theorem complex_multiplication :
  (4 - i) * (6 + 2 * i^3) = 22 - 14 * i :=
by sorry

-- Theorem 2
theorem complex_division :
  (5 * (4 + i)^2) / (i * (2 + i)) = 1 - 38 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_multiplication_complex_division_l380_38047


namespace NUMINAMATH_CALUDE_somus_age_l380_38069

theorem somus_age (somu father : ℕ) : 
  somu = father / 3 → 
  (somu - 7) = (father - 7) / 5 → 
  somu = 14 := by
sorry

end NUMINAMATH_CALUDE_somus_age_l380_38069


namespace NUMINAMATH_CALUDE_l_shaped_figure_perimeter_l380_38093

/-- Represents an L-shaped figure formed by a 3x3 square with a 2x2 square attached to one side -/
structure LShapedFigure :=
  (base_side : ℕ)
  (extension_side : ℕ)
  (unit_length : ℝ)
  (h_base : base_side = 3)
  (h_extension : extension_side = 2)
  (h_unit : unit_length = 1)

/-- Calculates the perimeter of the L-shaped figure -/
def perimeter (figure : LShapedFigure) : ℝ :=
  2 * (figure.base_side + figure.extension_side + figure.base_side) * figure.unit_length

/-- Theorem stating that the perimeter of the L-shaped figure is 15 units -/
theorem l_shaped_figure_perimeter :
  ∀ (figure : LShapedFigure), perimeter figure = 15 := by
  sorry

end NUMINAMATH_CALUDE_l_shaped_figure_perimeter_l380_38093


namespace NUMINAMATH_CALUDE_m_range_l380_38036

def p (x m : ℝ) : Prop := x^2 + 2*x - m > 0

theorem m_range :
  (∀ m : ℝ, ¬(p 1 m) ∧ (p 2 m)) ↔ (∀ m : ℝ, 3 ≤ m ∧ m < 8) :=
by sorry

end NUMINAMATH_CALUDE_m_range_l380_38036


namespace NUMINAMATH_CALUDE_sequence_fixed_points_l380_38021

theorem sequence_fixed_points 
  (a b c d : ℝ) 
  (h1 : c ≠ 0) 
  (h2 : a * d - b * c ≠ 0) 
  (a_n : ℕ → ℝ) 
  (h_seq : ∀ n, a_n (n + 1) = (a * a_n n + b) / (c * a_n n + d)) :
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ 
    (a * x₁ + b) / (c * x₁ + d) = x₁ ∧ 
    (a * x₂ + b) / (c * x₂ + d) = x₂ →
    ∀ n, (a_n (n + 1) - x₁) / (a_n (n + 1) - x₂) = 
         ((a - c * x₁) / (a - c * x₂)) * ((a_n n - x₁) / (a_n n - x₂))) ∧
  (∃ x₀, (a * x₀ + b) / (c * x₀ + d) = x₀ ∧ a ≠ -d →
    ∀ n, 1 / (a_n (n + 1) - x₀) = (2 * c) / (a + d) + 1 / (a_n n - x₀)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_fixed_points_l380_38021


namespace NUMINAMATH_CALUDE_field_trip_attendance_l380_38000

theorem field_trip_attendance :
  let num_vans : ℕ := 6
  let num_buses : ℕ := 8
  let people_per_van : ℕ := 6
  let people_per_bus : ℕ := 18
  let total_people : ℕ := num_vans * people_per_van + num_buses * people_per_bus
  total_people = 180 := by
sorry

end NUMINAMATH_CALUDE_field_trip_attendance_l380_38000


namespace NUMINAMATH_CALUDE_f_is_quadratic_l380_38037

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The specific equation we want to prove is quadratic -/
def f (x : ℝ) : ℝ := x^2 + x - 5

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_f_is_quadratic_l380_38037


namespace NUMINAMATH_CALUDE_problem_solution_l380_38060

theorem problem_solution :
  ∀ (a b c : ℕ+) (x y z : ℤ),
    x = -2272 →
    y = 1000 + 100 * c.val + 10 * b.val + a.val →
    z = 1 →
    a.val * x + b.val * y + c.val * z = 1 →
    a < b →
    b < c →
    y = 1987 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l380_38060


namespace NUMINAMATH_CALUDE_djibo_age_problem_l380_38042

/-- Represents the problem of finding when Djibo and his sister's ages summed to 35 --/
theorem djibo_age_problem (djibo_current_age sister_current_age past_sum : ℕ) 
  (h1 : djibo_current_age = 17)
  (h2 : sister_current_age = 28)
  (h3 : past_sum = 35) :
  ∃ (years_ago : ℕ), 
    (djibo_current_age - years_ago) + (sister_current_age - years_ago) = past_sum ∧ 
    years_ago = 5 := by
  sorry


end NUMINAMATH_CALUDE_djibo_age_problem_l380_38042


namespace NUMINAMATH_CALUDE_prob_one_boy_correct_dist_X_correct_dist_X_sum_to_one_l380_38074

/-- Represents the probability distribution of a discrete random variable -/
def ProbabilityDistribution (α : Type*) := α → ℚ

/-- The total number of students in the group -/
def total_students : ℕ := 5

/-- The number of boys in the group -/
def num_boys : ℕ := 3

/-- The number of girls in the group -/
def num_girls : ℕ := 2

/-- The number of students selected -/
def num_selected : ℕ := 2

/-- Calculates the probability of selecting exactly one boy when choosing two students -/
def prob_one_boy : ℚ := 3/5

/-- Represents the number of boys selected -/
inductive X where
  | zero : X
  | one : X
  | two : X

/-- The probability distribution of X (number of boys selected) -/
def dist_X : ProbabilityDistribution X :=
  fun x => match x with
    | X.zero => 1/10
    | X.one  => 3/5
    | X.two  => 3/10

/-- Theorem stating the probability of selecting exactly one boy is correct -/
theorem prob_one_boy_correct :
  prob_one_boy = 3/5 := by sorry

/-- Theorem stating the probability distribution of X is correct -/
theorem dist_X_correct :
  dist_X X.zero = 1/10 ∧
  dist_X X.one  = 3/5  ∧
  dist_X X.two  = 3/10 := by sorry

/-- Theorem stating the sum of probabilities in the distribution equals 1 -/
theorem dist_X_sum_to_one :
  dist_X X.zero + dist_X X.one + dist_X X.two = 1 := by sorry

end NUMINAMATH_CALUDE_prob_one_boy_correct_dist_X_correct_dist_X_sum_to_one_l380_38074


namespace NUMINAMATH_CALUDE_cindy_marbles_l380_38063

theorem cindy_marbles (initial_marbles : ℕ) (friends : ℕ) (marbles_per_friend : ℕ) :
  initial_marbles = 500 →
  friends = 4 →
  marbles_per_friend = 80 →
  4 * (initial_marbles - friends * marbles_per_friend) = 720 := by
  sorry

end NUMINAMATH_CALUDE_cindy_marbles_l380_38063


namespace NUMINAMATH_CALUDE_max_degree_polynomial_l380_38002

theorem max_degree_polynomial (p : ℕ) (hp : Nat.Prime p) :
  ∃ (d : ℕ), d = p - 2 ∧
  (∃ (T : Polynomial ℤ), (Polynomial.degree T = d) ∧
    (∀ (m n : ℤ), T.eval m ≡ T.eval n [ZMOD p] → m ≡ n [ZMOD p])) ∧
  (∀ (d' : ℕ), d' > d →
    ¬∃ (T : Polynomial ℤ), (Polynomial.degree T = d') ∧
      (∀ (m n : ℤ), T.eval m ≡ T.eval n [ZMOD p] → m ≡ n [ZMOD p])) := by
  sorry

end NUMINAMATH_CALUDE_max_degree_polynomial_l380_38002


namespace NUMINAMATH_CALUDE_max_value_of_s_l380_38058

-- Define the function s
def s (x y : ℝ) : ℝ := x + y

-- State the theorem
theorem max_value_of_s :
  ∃ (M : ℝ), M = 9 ∧ ∀ (x y : ℝ), s x y ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_s_l380_38058


namespace NUMINAMATH_CALUDE_creature_perimeter_l380_38061

/-- The perimeter of a circular creature with an open mouth -/
theorem creature_perimeter (r : ℝ) (central_angle : ℝ) : 
  r = 2 → central_angle = 270 → 
  (central_angle / 360) * (2 * π * r) + 2 * r = 3 * π + 4 :=
by sorry

end NUMINAMATH_CALUDE_creature_perimeter_l380_38061


namespace NUMINAMATH_CALUDE_binary_representation_of_41_l380_38081

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec go (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else go (m / 2) ((m % 2) :: acc)
    go n []

/-- The binary representation of 41 -/
def binary41 : List ℕ := [1, 0, 1, 0, 0, 1]

/-- Theorem stating that the binary representation of 41 is [1, 0, 1, 0, 0, 1] -/
theorem binary_representation_of_41 : toBinary 41 = binary41 := by
  sorry

end NUMINAMATH_CALUDE_binary_representation_of_41_l380_38081


namespace NUMINAMATH_CALUDE_expression_equals_one_l380_38080

def numerator : ℕ → ℚ
  | 0 => 1
  | n + 1 => numerator n * (1 + 18 / (n + 1))

def denominator : ℕ → ℚ
  | 0 => 1
  | n + 1 => denominator n * (1 + 20 / (n + 1))

theorem expression_equals_one :
  (numerator 20) / (denominator 18) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l380_38080


namespace NUMINAMATH_CALUDE_greatest_difference_l380_38062

/-- A type representing a chessboard arrangement of numbers 1 to 400 -/
def Arrangement := Fin 20 → Fin 20 → Fin 400

/-- The property that an arrangement has two numbers in the same row or column differing by at least N -/
def HasDifference (arr : Arrangement) (N : ℕ) : Prop :=
  ∃ (i j k : Fin 20), (arr i j).val + N ≤ (arr i k).val ∨ (arr j i).val + N ≤ (arr k i).val

/-- The theorem stating that 209 is the greatest natural number satisfying the given condition -/
theorem greatest_difference : 
  (∀ (arr : Arrangement), HasDifference arr 209) ∧ 
  ¬(∀ (arr : Arrangement), HasDifference arr 210) :=
sorry

end NUMINAMATH_CALUDE_greatest_difference_l380_38062


namespace NUMINAMATH_CALUDE_distance_to_focus_l380_38091

-- Define the parabola C: y^2 = 4x
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define point P
def P : ℝ × ℝ := (-2, 0)

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define points A and B as intersection points
axiom A : ℝ × ℝ
axiom B : ℝ × ℝ

-- A and B are on the parabola
axiom A_on_parabola : parabola A.1 A.2
axiom B_on_parabola : parabola B.1 B.2

-- A, B, and P are collinear
axiom collinear : ∃ (t : ℝ), B.1 - P.1 = t * (A.1 - P.1) ∧ B.2 - P.2 = t * (A.2 - P.2)

-- |PA| = 1/2 |AB|
axiom distance_relation : Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2) = 1/2 * Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

-- Theorem to prove
theorem distance_to_focus :
  Real.sqrt ((A.1 - focus.1)^2 + (A.2 - focus.2)^2) = 5/3 := by sorry

end NUMINAMATH_CALUDE_distance_to_focus_l380_38091


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l380_38068

theorem perpendicular_vectors_x_value :
  let a : Fin 2 → ℝ := ![(-3), 1]
  let b : Fin 2 → ℝ := ![x, 6]
  (∀ (i j : Fin 2), i.val + j.val = 1 → a i * b j = 0) →
  x = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l380_38068


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_implies_a_range_l380_38017

/-- Piecewise function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then x^2 + (4*a - 3)*x + 3*a
  else if 0 ≤ x ∧ x < Real.pi/2 then -Real.sin x
  else 0  -- undefined for x ≥ π/2

/-- The domain of f(x) -/
def dom (x : ℝ) : Prop := x < Real.pi/2

/-- f(x) is monotonically decreasing in its domain -/
def monotone_decreasing (a : ℝ) : Prop :=
  ∀ x y, dom x → dom y → x < y → f a x > f a y

/-- Theorem: If f(x) is monotonically decreasing, then a ∈ [0, 4/3] -/
theorem f_monotone_decreasing_implies_a_range (a : ℝ) :
  monotone_decreasing a → 0 ≤ a ∧ a ≤ 4/3 := by sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_implies_a_range_l380_38017


namespace NUMINAMATH_CALUDE_prob_same_length_is_11_35_l380_38040

/-- The set of all sides and diagonals of a regular hexagon -/
def T : Finset ℝ := sorry

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of shorter diagonals in a regular hexagon -/
def num_short_diagonals : ℕ := 6

/-- The number of longer diagonals in a regular hexagon -/
def num_long_diagonals : ℕ := 3

/-- The total number of segments in a regular hexagon -/
def total_segments : ℕ := num_sides + num_short_diagonals + num_long_diagonals

/-- The probability of selecting two segments of the same length -/
def prob_same_length : ℚ :=
  (num_sides * (num_sides - 1) + num_short_diagonals * (num_short_diagonals - 1) + num_long_diagonals * (num_long_diagonals - 1)) /
  (total_segments * (total_segments - 1))

theorem prob_same_length_is_11_35 : prob_same_length = 11 / 35 := by sorry

end NUMINAMATH_CALUDE_prob_same_length_is_11_35_l380_38040


namespace NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_8_l380_38018

theorem factorization_of_2x_squared_minus_8 (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_8_l380_38018


namespace NUMINAMATH_CALUDE_continuous_stripe_probability_l380_38019

/-- Represents the orientation of a stripe on a cube face -/
inductive StripeOrientation
  | EdgeToEdge1
  | EdgeToEdge2
  | Diagonal1
  | Diagonal2

/-- Represents a cube with stripes on its faces -/
structure StripedCube :=
  (faces : Fin 6 → StripeOrientation)

/-- Checks if a given StripedCube has a continuous stripe encircling it -/
def hasContinuousStripe (cube : StripedCube) : Bool :=
  sorry

/-- The total number of possible stripe combinations -/
def totalCombinations : Nat :=
  4^6

/-- The number of stripe combinations that result in a continuous stripe -/
def favorableCombinations : Nat :=
  3 * 4

/-- The probability of a continuous stripe encircling the cube -/
def probabilityOfContinuousStripe : Rat :=
  favorableCombinations / totalCombinations

theorem continuous_stripe_probability :
  probabilityOfContinuousStripe = 3 / 1024 :=
sorry

end NUMINAMATH_CALUDE_continuous_stripe_probability_l380_38019


namespace NUMINAMATH_CALUDE_sum_of_f_l380_38026

noncomputable def f (x : ℝ) : ℝ := 1 / (2^x + Real.sqrt 2)

theorem sum_of_f (x : ℝ) : f (-x) + f (1 + x) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_f_l380_38026


namespace NUMINAMATH_CALUDE_sin_510_degrees_l380_38011

theorem sin_510_degrees : Real.sin (510 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_510_degrees_l380_38011


namespace NUMINAMATH_CALUDE_min_sqrt_equality_l380_38032

theorem min_sqrt_equality {a b c : ℝ} (ha : 0 < a ∧ a ≤ 1) (hb : 0 < b ∧ b ≤ 1) (hc : 0 < c ∧ c ≤ 1) :
  (min (Real.sqrt ((a*b + 1)/(a*b*c))) (min (Real.sqrt ((b*c + 1)/(a*b*c))) (Real.sqrt ((c*a + 1)/(a*b*c)))) =
    Real.sqrt ((1-a)/a) + Real.sqrt ((1-b)/b) + Real.sqrt ((1-c)/c)) ↔
  ∃ w : ℝ, w > 0 ∧ a = w^2/(1+(w^2+1)^2) ∧ b = w^2/(1+w^2) ∧ c = 1/(1+w^2) :=
by sorry

end NUMINAMATH_CALUDE_min_sqrt_equality_l380_38032


namespace NUMINAMATH_CALUDE_unique_multiple_of_72_l380_38039

def is_multiple_of_72 (n : ℕ) : Prop := ∃ k : ℕ, n = 72 * k

def is_form_a679b (n : ℕ) : Prop :=
  ∃ a b : ℕ, a < 10 ∧ b < 10 ∧ n = a * 10000 + 6 * 1000 + 7 * 100 + 9 * 10 + b

theorem unique_multiple_of_72 :
  ∀ n : ℕ, is_form_a679b n ∧ is_multiple_of_72 n ↔ n = 36792 :=
by sorry

end NUMINAMATH_CALUDE_unique_multiple_of_72_l380_38039


namespace NUMINAMATH_CALUDE_window_area_ratio_l380_38033

theorem window_area_ratio :
  ∀ (ad ab : ℝ),
  ad / ab = 4 / 3 →
  ab = 36 →
  let r := ab / 2
  let rectangle_area := ad * ab
  let semicircles_area := π * r^2
  rectangle_area / semicircles_area = 16 / (3 * π) := by
sorry

end NUMINAMATH_CALUDE_window_area_ratio_l380_38033


namespace NUMINAMATH_CALUDE_student_count_pedro_grade_count_l380_38051

/-- If a student is ranked both n-th best and n-th worst in a grade,
    then the total number of students in that grade is 2n - 1. -/
theorem student_count (n : ℕ) (h : n > 0) :
  ∃ (total : ℕ), total = 2 * n - 1 := by sorry

/-- There are 59 students in Pedro's grade. -/
theorem pedro_grade_count :
  ∃ (total : ℕ), total = 59 := by
  apply student_count 30
  norm_num

end NUMINAMATH_CALUDE_student_count_pedro_grade_count_l380_38051


namespace NUMINAMATH_CALUDE_expression_equality_l380_38045

theorem expression_equality (x y : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hsum : 2*x + y/2 ≠ 0) : 
  (2*x + y/2)⁻¹ * ((2*x)⁻¹ + (y/2)⁻¹) = (x*y)⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l380_38045


namespace NUMINAMATH_CALUDE_unique_solution_sqrt_equation_l380_38097

theorem unique_solution_sqrt_equation :
  ∀ x y : ℕ,
    x ≥ 1 →
    y ≥ 1 →
    y ≥ x →
    (Real.sqrt (2 * x) - 1) * (Real.sqrt (2 * y) - 1) = 1 →
    x = 2 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_sqrt_equation_l380_38097


namespace NUMINAMATH_CALUDE_speed_increase_ratio_l380_38023

theorem speed_increase_ratio (v : ℝ) (h : (v + 2) / v = 2.5) : (v + 4) / v = 4 := by
  sorry

end NUMINAMATH_CALUDE_speed_increase_ratio_l380_38023


namespace NUMINAMATH_CALUDE_inequality_proof_l380_38038

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a + b + c ≤ b*c/(b+c) + c*a/(c+a) + a*b/(a+b) + (1/2) * (b*c/a + c*a/b + a*b/c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l380_38038


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l380_38055

/-- An isosceles triangle with side lengths 4 and 8 has a perimeter of 20. -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a > 0 ∧ b > 0 ∧ c > 0 →  -- positive side lengths
  (a = 4 ∧ b = 8 ∧ c = 8) ∨ (a = 8 ∧ b = 4 ∧ c = 8) ∨ (a = 8 ∧ b = 8 ∧ c = 4) →  -- possible configurations
  a + b > c ∧ b + c > a ∧ a + c > b →  -- triangle inequality
  a + b + c = 20 :=
by sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l380_38055


namespace NUMINAMATH_CALUDE_sequence_a_property_l380_38009

def sequence_a : ℕ → ℚ
  | 0 => 1/2
  | (n+1) => sequence_a n + (sequence_a n)^2 / 2023

theorem sequence_a_property : sequence_a 2023 < 1 ∧ 1 < sequence_a 2024 := by
  sorry

end NUMINAMATH_CALUDE_sequence_a_property_l380_38009


namespace NUMINAMATH_CALUDE_convergence_trap_equivalence_l380_38049

open Set Filter Topology Metric

variable {X : Type*} [MetricSpace X]
variable (x : ℕ → X) (a : X)

def is_trap (s : Set X) (x : ℕ → X) : Prop :=
  ∃ N, ∀ n ≥ N, x n ∈ s

theorem convergence_trap_equivalence :
  (Tendsto x atTop (𝓝 a)) ↔
  (∀ ε > 0, is_trap (ball a ε) x) :=
sorry

end NUMINAMATH_CALUDE_convergence_trap_equivalence_l380_38049


namespace NUMINAMATH_CALUDE_area_of_square_on_hypotenuse_l380_38001

/-- Represents a right-angled isosceles triangle with squares on its sides -/
structure IsoscelesRightTriangle where
  /-- Length of the equal sides -/
  side : ℝ
  /-- Sum of the areas of squares on all sides -/
  squaresSum : ℝ
  /-- The sum of squares is 450 -/
  sum_eq_450 : squaresSum = 450

/-- The area of the square on the hypotenuse of an isosceles right triangle -/
def squareOnHypotenuse (t : IsoscelesRightTriangle) : ℝ := 2 * t.side^2

theorem area_of_square_on_hypotenuse (t : IsoscelesRightTriangle) :
  squareOnHypotenuse t = 225 := by
  sorry

end NUMINAMATH_CALUDE_area_of_square_on_hypotenuse_l380_38001


namespace NUMINAMATH_CALUDE_largest_divisor_of_expression_l380_38028

theorem largest_divisor_of_expression (x : ℤ) (h : Even x) :
  ∃ (k : ℤ), (8*x + 2) * (8*x + 4) * (4*x + 2) = 240 * k ∧
  ∀ (m : ℤ), m > 240 → ∃ (y : ℤ), Even y ∧ ¬∃ (l : ℤ), (8*y + 2) * (8*y + 4) * (4*y + 2) = m * l :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_expression_l380_38028


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_implies_m_range_l380_38075

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 12*x

-- Define the property of monotonically decreasing
def monotone_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f y < f x

-- State the theorem
theorem f_monotone_decreasing_implies_m_range (m : ℝ) :
  monotone_decreasing f (2*m) (m+1) → m ∈ Set.Icc (-1) 1 :=
sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_implies_m_range_l380_38075


namespace NUMINAMATH_CALUDE_subtract_inequality_preserves_order_l380_38054

theorem subtract_inequality_preserves_order (a b c : ℝ) : a > b → a - c > b - c := by
  sorry

end NUMINAMATH_CALUDE_subtract_inequality_preserves_order_l380_38054


namespace NUMINAMATH_CALUDE_vector_magnitude_l380_38024

theorem vector_magnitude (a b : ℝ × ℝ) :
  ‖a‖ = 1 →
  ‖b‖ = 2 →
  a - b = (Real.sqrt 3, Real.sqrt 2) →
  ‖a + 2 • b‖ = Real.sqrt 17 := by
sorry

end NUMINAMATH_CALUDE_vector_magnitude_l380_38024


namespace NUMINAMATH_CALUDE_min_max_f_l380_38027

def a : ℕ := 2001

def A : Set (ℕ × ℕ) :=
  {p | let m := p.1
       let n := p.2
       m < 2 * a ∧
       (2 * n) ∣ (2 * a * m - m^2 + n^2) ∧
       n^2 - m^2 + 2 * m * n ≤ 2 * a * (n - m)}

def f (p : ℕ × ℕ) : ℚ :=
  let m := p.1
  let n := p.2
  (2 * a * m - m^2 - m * n) / n

theorem min_max_f :
  ∃ (min max : ℚ), min = 2 ∧ max = 3750 ∧
  (∀ p ∈ A, min ≤ f p ∧ f p ≤ max) ∧
  (∃ p₁ ∈ A, f p₁ = min) ∧
  (∃ p₂ ∈ A, f p₂ = max) :=
sorry

end NUMINAMATH_CALUDE_min_max_f_l380_38027


namespace NUMINAMATH_CALUDE_f_has_root_in_interval_l380_38006

-- Define the function f(x) = x³ - 2x² + 2
def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 2

-- Theorem statement
theorem f_has_root_in_interval :
  ∃ x ∈ Set.Ioo (-1 : ℝ) (-1/2 : ℝ), f x = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_f_has_root_in_interval_l380_38006


namespace NUMINAMATH_CALUDE_power_of_five_mod_eighteen_l380_38092

theorem power_of_five_mod_eighteen (x : ℕ) : ∃ x, x > 0 ∧ (5^x : ℤ) % 18 = 13 ∧ ∀ y, 0 < y ∧ y < x → (5^y : ℤ) % 18 ≠ 13 := by
  sorry

end NUMINAMATH_CALUDE_power_of_five_mod_eighteen_l380_38092


namespace NUMINAMATH_CALUDE_x_positive_sufficient_not_necessary_for_abs_x_positive_l380_38016

theorem x_positive_sufficient_not_necessary_for_abs_x_positive :
  (∃ x : ℝ, x > 0 → |x| > 0) ∧
  (∃ x : ℝ, |x| > 0 ∧ ¬(x > 0)) := by
  sorry

end NUMINAMATH_CALUDE_x_positive_sufficient_not_necessary_for_abs_x_positive_l380_38016


namespace NUMINAMATH_CALUDE_compute_expression_l380_38048

theorem compute_expression : 9 + 7 * (5 - Real.sqrt 16) ^ 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l380_38048


namespace NUMINAMATH_CALUDE_symmetric_point_y_axis_of_2_1_l380_38022

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The operation of finding the symmetric point with respect to the y-axis -/
def symmetricPointYAxis (p : Point) : Point :=
  { x := -p.x, y := p.y }

/-- Theorem stating that the symmetric point of (2,1) with respect to the y-axis is (-2,1) -/
theorem symmetric_point_y_axis_of_2_1 :
  let P : Point := { x := 2, y := 1 }
  symmetricPointYAxis P = { x := -2, y := 1 } := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_y_axis_of_2_1_l380_38022


namespace NUMINAMATH_CALUDE_sum_seven_probability_l380_38067

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The total number of possible outcomes when tossing two dice -/
def totalOutcomes : ℕ := numFaces * numFaces

/-- The number of ways to get a sum of 7 when tossing two dice -/
def favorableOutcomes : ℕ := 6

/-- The probability of getting a sum of 7 when tossing two dice -/
def probabilitySumSeven : ℚ := favorableOutcomes / totalOutcomes

theorem sum_seven_probability :
  probabilitySumSeven = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_seven_probability_l380_38067


namespace NUMINAMATH_CALUDE_pedros_plums_l380_38044

theorem pedros_plums (total_fruits : ℕ) (total_cost : ℕ) (plum_cost : ℕ) (peach_cost : ℕ) 
  (h1 : total_fruits = 32)
  (h2 : total_cost = 52)
  (h3 : plum_cost = 2)
  (h4 : peach_cost = 1) :
  ∃ (plums peaches : ℕ), 
    plums + peaches = total_fruits ∧ 
    plum_cost * plums + peach_cost * peaches = total_cost ∧
    plums = 20 :=
by sorry

end NUMINAMATH_CALUDE_pedros_plums_l380_38044


namespace NUMINAMATH_CALUDE_quadratic_equation_completion_square_l380_38052

theorem quadratic_equation_completion_square :
  ∃ (d e : ℤ), (∀ x : ℝ, x^2 - 10*x + 15 = 0 ↔ (x + d)^2 = e) ∧ d + e = 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_completion_square_l380_38052


namespace NUMINAMATH_CALUDE_largest_number_proof_l380_38087

theorem largest_number_proof (a b : ℕ+) 
  (hcf_cond : Nat.gcd a b = 42)
  (lcm_cond : Nat.lcm a b = 42 * 11 * 12) :
  max a b = 504 := by
sorry

end NUMINAMATH_CALUDE_largest_number_proof_l380_38087


namespace NUMINAMATH_CALUDE_alpha_value_l380_38053

theorem alpha_value (α : Real) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.sin (α - Real.pi / 18) = Real.sqrt 3 / 2) : 
  α = Real.pi * 7 / 18 := by
  sorry

end NUMINAMATH_CALUDE_alpha_value_l380_38053


namespace NUMINAMATH_CALUDE_section_plane_angle_cosine_l380_38071

/-- Regular hexagonal pyramid with given properties -/
structure HexagonalPyramid where
  -- Base side length
  a : ℝ
  -- Distance from apex to section plane
  d : ℝ
  -- Base is a regular hexagon
  is_regular_hexagon : a > 0
  -- Section plane properties
  section_plane_properties : True
  -- Given distance
  distance_constraint : d = 1
  -- Given base side length
  base_side_length : a = 2

/-- The angle between the section plane and the base plane -/
def section_angle (pyramid : HexagonalPyramid) : ℝ := sorry

/-- Theorem stating the cosine of the angle between the section plane and base plane -/
theorem section_plane_angle_cosine (pyramid : HexagonalPyramid) : 
  Real.cos (section_angle pyramid) = 3/4 := by sorry

end NUMINAMATH_CALUDE_section_plane_angle_cosine_l380_38071


namespace NUMINAMATH_CALUDE_jean_is_cyclist_l380_38070

/-- Represents a traveler's journey --/
structure Traveler where
  distanceTraveled : ℝ
  distanceRemaining : ℝ

/-- Jean's travel condition --/
def jeanCondition (j : Traveler) : Prop :=
  3 * j.distanceTraveled + 2 * j.distanceRemaining = j.distanceTraveled + j.distanceRemaining

/-- Jules' travel condition --/
def julesCondition (j : Traveler) : Prop :=
  (1/2) * j.distanceTraveled + 3 * j.distanceRemaining = j.distanceTraveled + j.distanceRemaining

/-- The theorem to prove --/
theorem jean_is_cyclist (jean jules : Traveler) 
  (hj : jeanCondition jean) (hk : julesCondition jules) : 
  jean.distanceTraveled / (jean.distanceTraveled + jean.distanceRemaining) < 
  jules.distanceTraveled / (jules.distanceTraveled + jules.distanceRemaining) :=
sorry

end NUMINAMATH_CALUDE_jean_is_cyclist_l380_38070


namespace NUMINAMATH_CALUDE_units_digit_sum_powers_l380_38029

theorem units_digit_sum_powers : (2^20 + 3^21 + 7^20) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_powers_l380_38029


namespace NUMINAMATH_CALUDE_action_movies_rented_l380_38098

theorem action_movies_rented (a : ℝ) : 
  let total_movies := 10 * a / 0.64
  let comedy_movies := 10 * a
  let non_comedy_movies := total_movies - comedy_movies
  let drama_movies := 5 * (non_comedy_movies / 6)
  let action_movies := non_comedy_movies / 6
  action_movies = 0.9375 * a := by
sorry

end NUMINAMATH_CALUDE_action_movies_rented_l380_38098


namespace NUMINAMATH_CALUDE_probability_nine_matches_zero_l380_38007

/-- A matching problem with n pairs -/
structure MatchingProblem (n : ℕ) where
  /-- The number of pairs to match -/
  pairs : ℕ
  /-- Assertion that the number of pairs is n -/
  pairs_eq : pairs = n

/-- The probability of correctly matching exactly k pairs in a matching problem with n pairs by random selection -/
noncomputable def probability_exact_matches (n k : ℕ) (problem : MatchingProblem n) : ℝ :=
  sorry

/-- Theorem: In a matching problem with 10 pairs, the probability of correctly matching exactly 9 pairs by random selection is 0 -/
theorem probability_nine_matches_zero :
  ∀ (problem : MatchingProblem 10), probability_exact_matches 10 9 problem = 0 :=
sorry

end NUMINAMATH_CALUDE_probability_nine_matches_zero_l380_38007


namespace NUMINAMATH_CALUDE_jimmy_stair_climbing_l380_38010

/-- The sum of the first n terms of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- Jimmy's stair climbing problem -/
theorem jimmy_stair_climbing : arithmetic_sum 25 7 6 = 255 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_stair_climbing_l380_38010


namespace NUMINAMATH_CALUDE_larger_part_of_sum_and_product_l380_38077

theorem larger_part_of_sum_and_product (x y : ℝ) : 
  x > 0 ∧ y > 0 ∧ x + y = 20 ∧ x * y = 96 → max x y = 12 := by
  sorry

end NUMINAMATH_CALUDE_larger_part_of_sum_and_product_l380_38077


namespace NUMINAMATH_CALUDE_distance_to_focus_is_three_l380_38056

/-- Parabola structure -/
structure Parabola where
  a : ℝ
  h : a > 0

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between a point and a vertical line -/
def distanceToVerticalLine (P : Point) (x₀ : ℝ) : ℝ :=
  |P.x - x₀|

/-- Check if a point is on the parabola -/
def isOnParabola (P : Point) (p : Parabola) : Prop :=
  P.y^2 = 4 * p.a * P.x

/-- Distance from a point to the focus of the parabola -/
noncomputable def distanceToFocus (P : Point) (p : Parabola) : ℝ :=
  sorry

/-- Main theorem -/
theorem distance_to_focus_is_three
  (p : Parabola)
  (P : Point)
  (h_on_parabola : isOnParabola P p)
  (h_distance : distanceToVerticalLine P (-3) = 5)
  : distanceToFocus P p = 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_focus_is_three_l380_38056


namespace NUMINAMATH_CALUDE_adam_apple_purchase_l380_38079

/-- The total quantity of apples Adam bought over three days -/
def total_apples (monday_apples : ℝ) : ℝ :=
  let tuesday_apples := monday_apples * 3.2
  let wednesday_apples := tuesday_apples * 1.05
  monday_apples + tuesday_apples + wednesday_apples

/-- Theorem stating the total quantity of apples Adam bought -/
theorem adam_apple_purchase :
  total_apples 15.5 = 117.18 := by
  sorry

end NUMINAMATH_CALUDE_adam_apple_purchase_l380_38079


namespace NUMINAMATH_CALUDE_hanks_route_length_l380_38008

theorem hanks_route_length :
  ∀ (route_length : ℝ) (monday_speed tuesday_speed : ℝ) (time_diff : ℝ),
    monday_speed = 70 →
    tuesday_speed = 75 →
    time_diff = 1/30 →
    route_length / monday_speed - route_length / tuesday_speed = time_diff →
    route_length = 35 := by
  sorry

end NUMINAMATH_CALUDE_hanks_route_length_l380_38008


namespace NUMINAMATH_CALUDE_exists_overlap_at_least_one_fifth_l380_38096

/-- Represents a patch on the coat -/
structure Patch where
  area : ℝ
  area_nonneg : area ≥ 0

/-- Represents a coat with patches -/
structure Coat where
  total_area : ℝ
  patches : Finset Patch
  total_area_is_one : total_area = 1
  five_patches : patches.card = 5
  patch_area_at_least_half : ∀ p ∈ patches, p.area ≥ 1/2

/-- The theorem to be proved -/
theorem exists_overlap_at_least_one_fifth (coat : Coat) : 
  ∃ p1 p2 : Patch, p1 ∈ coat.patches ∧ p2 ∈ coat.patches ∧ p1 ≠ p2 ∧ 
    ∃ overlap_area : ℝ, overlap_area ≥ 1/5 ∧ 
      overlap_area ≤ min p1.area p2.area := by
  sorry

end NUMINAMATH_CALUDE_exists_overlap_at_least_one_fifth_l380_38096


namespace NUMINAMATH_CALUDE_katy_brownies_l380_38003

/-- The number of brownies Katy eats on Monday -/
def monday_brownies : ℕ := 5

/-- The number of brownies Katy makes in total -/
def total_brownies : ℕ := monday_brownies + 2 * monday_brownies

theorem katy_brownies : 
  total_brownies = 15 := by sorry

end NUMINAMATH_CALUDE_katy_brownies_l380_38003


namespace NUMINAMATH_CALUDE_larger_number_proof_l380_38025

theorem larger_number_proof (a b : ℕ+) : 
  (Nat.gcd a b = 23) →
  (Nat.lcm a b = 5382) →
  (∃ (x y : ℕ+), x * y = 234 ∧ (x = 13 ∨ x = 18) ∧ (y = 13 ∨ y = 18)) →
  (max a b = 414) := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l380_38025


namespace NUMINAMATH_CALUDE_intersection_point_l380_38034

/-- The slope of the first line -/
def m : ℚ := 3

/-- The first line equation -/
def line1 (x y : ℚ) : Prop := y = m * x + 2

/-- The point through which the perpendicular line passes -/
def point : ℚ × ℚ := (3, 4)

/-- The slope of the perpendicular line -/
def m_perp : ℚ := -1 / m

/-- The perpendicular line equation -/
def line2 (x y : ℚ) : Prop := y - point.2 = m_perp * (x - point.1)

/-- The intersection point -/
def intersection : ℚ × ℚ := (9/10, 47/10)

theorem intersection_point : 
  line1 intersection.1 intersection.2 ∧ 
  line2 intersection.1 intersection.2 := by sorry

end NUMINAMATH_CALUDE_intersection_point_l380_38034


namespace NUMINAMATH_CALUDE_every_algorithm_relies_on_sequential_structure_l380_38073

/-- Represents the basic structures used in algorithms -/
inductive AlgorithmStructure
  | Logical
  | Conditional
  | Loop
  | Sequential

/-- Represents an algorithm with its characteristics -/
structure Algorithm where
  input : Nat
  output : Nat
  steps : List AlgorithmStructure
  isDefinite : Bool
  isFinite : Bool
  isEffective : Bool

/-- Theorem stating that every algorithm relies on the Sequential structure -/
theorem every_algorithm_relies_on_sequential_structure (a : Algorithm) :
  AlgorithmStructure.Sequential ∈ a.steps :=
sorry

end NUMINAMATH_CALUDE_every_algorithm_relies_on_sequential_structure_l380_38073


namespace NUMINAMATH_CALUDE_triangle_properties_l380_38041

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  b = c →
  2 * Real.sin B = Real.sqrt 3 * Real.sin A →
  0 < B →
  B < π / 2 →
  A + B + C = π →
  a * Real.sin B = b * Real.sin A →
  b * Real.sin C = c * Real.sin B →
  c * Real.sin A = a * Real.sin C →
  (Real.sin B = Real.sqrt 6 / 3) ∧
  (Real.cos (2 * B + π / 3) = -(1 + 2 * Real.sqrt 6) / 6) ∧
  (b = 2 → (1 / 2) * b * c * Real.sin A = 4 * Real.sqrt 2 / 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l380_38041


namespace NUMINAMATH_CALUDE_divisor_sum_360_l380_38005

/-- Sum of positive divisors function -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- Main theorem -/
theorem divisor_sum_360 (i j k : ℕ) : 
  sum_of_divisors (2^i * 3^j * 5^k) = 360 → i + j + k = 6 := by
  sorry

end NUMINAMATH_CALUDE_divisor_sum_360_l380_38005


namespace NUMINAMATH_CALUDE_family_savings_l380_38020

def income : ℕ := 509600
def expenses : ℕ := 276000
def initial_savings : ℕ := 1147240

theorem family_savings : initial_savings + income - expenses = 1340840 := by
  sorry

end NUMINAMATH_CALUDE_family_savings_l380_38020


namespace NUMINAMATH_CALUDE_lunks_needed_for_dozen_apples_l380_38084

/-- Exchange rate between lunks and kunks -/
def lunks_to_kunks : ℚ := 4 / 7

/-- Exchange rate between kunks and apples -/
def kunks_to_apples : ℚ := 5 / 3

/-- Number of apples to purchase -/
def apples_to_buy : ℕ := 12

/-- Theorem stating the number of lunks needed to buy 12 apples -/
theorem lunks_needed_for_dozen_apples :
  ⌈(apples_to_buy : ℚ) / kunks_to_apples / lunks_to_kunks⌉ = 14 := by sorry

end NUMINAMATH_CALUDE_lunks_needed_for_dozen_apples_l380_38084


namespace NUMINAMATH_CALUDE_sin_minus_cos_105_deg_l380_38057

theorem sin_minus_cos_105_deg : 
  Real.sin (105 * π / 180) - Real.cos (105 * π / 180) = Real.sqrt 6 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_minus_cos_105_deg_l380_38057


namespace NUMINAMATH_CALUDE_total_goals_is_fifteen_l380_38085

def soccer_match_goals : ℕ := by
  -- Define the goals scored by The Kickers in the first period
  let kickers_first_period : ℕ := 2

  -- Define the goals scored by The Kickers in the second period
  let kickers_second_period : ℕ := 2 * kickers_first_period

  -- Define the goals scored by The Spiders in the first period
  let spiders_first_period : ℕ := kickers_first_period / 2

  -- Define the goals scored by The Spiders in the second period
  let spiders_second_period : ℕ := 2 * kickers_second_period

  -- Calculate the total goals
  let total_goals : ℕ := kickers_first_period + kickers_second_period + 
                         spiders_first_period + spiders_second_period

  -- Prove that the total goals equal 15
  have : total_goals = 15 := by sorry

  exact total_goals

-- Theorem stating that the total number of goals is 15
theorem total_goals_is_fifteen : soccer_match_goals = 15 := by sorry

end NUMINAMATH_CALUDE_total_goals_is_fifteen_l380_38085


namespace NUMINAMATH_CALUDE_platform_length_l380_38082

/-- Given a train and platform with specific properties, prove the length of the platform -/
theorem platform_length 
  (train_length : ℝ)
  (time_cross_platform : ℝ)
  (time_cross_pole : ℝ)
  (h1 : train_length = 300)
  (h2 : time_cross_platform = 36)
  (h3 : time_cross_pole = 18) :
  let train_speed := train_length / time_cross_pole
  let platform_length := train_speed * time_cross_platform - train_length
  platform_length = 300 := by
sorry


end NUMINAMATH_CALUDE_platform_length_l380_38082


namespace NUMINAMATH_CALUDE_solve_for_a_l380_38094

theorem solve_for_a (a b d : ℤ) 
  (h1 : a + b = d) 
  (h2 : b + d = 7) 
  (h3 : d = 4) : 
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_solve_for_a_l380_38094


namespace NUMINAMATH_CALUDE_apple_ratio_l380_38030

/-- Prove that the ratio of Harry's apples to Tim's apples is 1:2 -/
theorem apple_ratio :
  ∀ (martha_apples tim_apples harry_apples : ℕ),
    martha_apples = 68 →
    tim_apples = martha_apples - 30 →
    harry_apples = 19 →
    (harry_apples : ℚ) / tim_apples = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_apple_ratio_l380_38030


namespace NUMINAMATH_CALUDE_bar_chart_suitable_for_rope_skipping_l380_38031

/-- Represents different types of statistical charts -/
inductive StatisticalChart
  | BarChart
  | LineChart
  | PieChart

/-- Represents a dataset of rope skipping scores -/
structure RopeSkippingData where
  scores : List Nat

/-- Defines the property of a chart being suitable for representing discrete data points -/
def suitableForDiscreteData (chart : StatisticalChart) : Prop :=
  match chart with
  | StatisticalChart.BarChart => True
  | _ => False

/-- Theorem stating that a bar chart is suitable for representing rope skipping scores -/
theorem bar_chart_suitable_for_rope_skipping (data : RopeSkippingData) :
  suitableForDiscreteData StatisticalChart.BarChart :=
by sorry

end NUMINAMATH_CALUDE_bar_chart_suitable_for_rope_skipping_l380_38031


namespace NUMINAMATH_CALUDE_inequality_solution_set_l380_38012

theorem inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, (a^2 - 1) * x^2 - (a - 1) * x - 1 < 0) ↔ 
  a ∈ Set.Ioc (-3/5) 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l380_38012


namespace NUMINAMATH_CALUDE_ratio_of_shares_l380_38059

/-- Given a total amount divided among three persons, prove the ratio of the first person's share to the second person's share. -/
theorem ratio_of_shares (total : ℕ) (r_share : ℕ) (q_to_r_ratio : Rat) :
  total = 1210 →
  r_share = 400 →
  q_to_r_ratio = 9 / 10 →
  ∃ (p_share q_share : ℕ),
    p_share + q_share + r_share = total ∧
    q_share = (q_to_r_ratio * r_share).num ∧
    p_share * 4 = q_share * 5 :=
by sorry

end NUMINAMATH_CALUDE_ratio_of_shares_l380_38059


namespace NUMINAMATH_CALUDE_ones_count_l380_38035

theorem ones_count (hundreds tens total : ℕ) (h1 : hundreds = 3) (h2 : tens = 8) (h3 : total = 383) :
  total - (hundreds * 100 + tens * 10) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ones_count_l380_38035


namespace NUMINAMATH_CALUDE_average_of_data_l380_38076

def data : List ℝ := [2, 5, 5, 6, 7]

theorem average_of_data : (data.sum / data.length : ℝ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_average_of_data_l380_38076


namespace NUMINAMATH_CALUDE_unique_zero_of_exp_plus_linear_l380_38072

/-- The function f(x) = e^x + 3x has exactly one zero. -/
theorem unique_zero_of_exp_plus_linear : ∃! x : ℝ, Real.exp x + 3 * x = 0 := by sorry

end NUMINAMATH_CALUDE_unique_zero_of_exp_plus_linear_l380_38072


namespace NUMINAMATH_CALUDE_hair_reaches_floor_simultaneously_l380_38090

/-- Represents the growth rate of a person or their hair -/
structure GrowthRate where
  rate : ℝ

/-- Represents a person with their growth rate and hair growth rate -/
structure Person where
  growth : GrowthRate
  hairGrowth : GrowthRate

/-- The rate at which the distance from hair to floor decreases -/
def hairToFloorRate (p : Person) : ℝ :=
  p.hairGrowth.rate - p.growth.rate

theorem hair_reaches_floor_simultaneously
  (katya alena : Person)
  (h1 : katya.hairGrowth.rate = 2 * katya.growth.rate)
  (h2 : alena.growth.rate = katya.hairGrowth.rate)
  (h3 : alena.hairGrowth.rate = 1.5 * alena.growth.rate) :
  hairToFloorRate katya = hairToFloorRate alena :=
sorry

end NUMINAMATH_CALUDE_hair_reaches_floor_simultaneously_l380_38090


namespace NUMINAMATH_CALUDE_nigels_winnings_l380_38089

/-- The amount of money Nigel won initially -/
def initial_winnings : ℝ := sorry

/-- The amount Nigel gave away -/
def amount_given_away : ℝ := 25

/-- The amount Nigel's mother gave him -/
def amount_from_mother : ℝ := 80

/-- The extra amount Nigel has compared to twice his initial winnings -/
def extra_amount : ℝ := 10

theorem nigels_winnings :
  initial_winnings - amount_given_away + amount_from_mother = 
  2 * initial_winnings + extra_amount ∧ initial_winnings = 45 := by
  sorry

end NUMINAMATH_CALUDE_nigels_winnings_l380_38089


namespace NUMINAMATH_CALUDE_line_no_dot_count_l380_38095

/-- Represents the properties of an alphabet with dots and lines -/
structure Alphabet where
  total_letters : ℕ
  dot_and_line : ℕ
  dot_no_line : ℕ
  has_dot_or_line : Prop

/-- The number of letters with a straight line but no dot -/
def line_no_dot (α : Alphabet) : ℕ :=
  α.total_letters - (α.dot_and_line + α.dot_no_line)

/-- Theorem stating the number of letters with a line but no dot in the given alphabet -/
theorem line_no_dot_count (α : Alphabet) 
  (h1 : α.total_letters = 40)
  (h2 : α.dot_and_line = 11)
  (h3 : α.dot_no_line = 5)
  (h4 : α.has_dot_or_line) :
  line_no_dot α = 24 := by
  sorry

end NUMINAMATH_CALUDE_line_no_dot_count_l380_38095


namespace NUMINAMATH_CALUDE_expense_equalization_l380_38065

/-- Given three people's expenses A, B, and C, where A < B < C, 
    prove that the amount the person who paid A needs to give to each of the others 
    to equalize the costs is (B + C - 2A) / 3 -/
theorem expense_equalization (A B C : ℝ) (h1 : A < B) (h2 : B < C) :
  let total := A + B + C
  let equal_share := total / 3
  let amount_to_give := equal_share - A
  amount_to_give = (B + C - 2 * A) / 3 := by
  sorry

end NUMINAMATH_CALUDE_expense_equalization_l380_38065


namespace NUMINAMATH_CALUDE_tommy_balloons_l380_38088

/-- The number of balloons Tommy has after receiving more from his mom -/
def total_balloons (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem stating that Tommy's total balloons is the sum of his initial balloons and additional balloons -/
theorem tommy_balloons (initial : ℕ) (additional : ℕ) :
  total_balloons initial additional = initial + additional := by
  sorry

end NUMINAMATH_CALUDE_tommy_balloons_l380_38088
