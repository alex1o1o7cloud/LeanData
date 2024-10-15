import Mathlib

namespace NUMINAMATH_CALUDE_pirate_catch_caravel_l3979_397978

/-- Represents the velocity of a ship in nautical miles per hour -/
structure Velocity where
  speed : ℝ
  angle : ℝ

/-- Represents the position of a ship -/
structure Position where
  x : ℝ
  y : ℝ

/-- Calculate the minimum speed required for the pirate ship to catch the caravel -/
def min_pirate_speed (initial_distance : ℝ) (caravel_velocity : Velocity) : ℝ :=
  sorry

theorem pirate_catch_caravel (initial_distance : ℝ) (caravel_velocity : Velocity) :
  initial_distance = 10 ∧
  caravel_velocity.speed = 12 ∧
  caravel_velocity.angle = -5 * π / 6 →
  min_pirate_speed initial_distance caravel_velocity = 6 * Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_pirate_catch_caravel_l3979_397978


namespace NUMINAMATH_CALUDE_complex_fraction_pure_imaginary_l3979_397990

def complex_is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem complex_fraction_pure_imaginary (a : ℝ) : 
  complex_is_pure_imaginary ((a + 6 * Complex.I) / (3 - Complex.I)) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_pure_imaginary_l3979_397990


namespace NUMINAMATH_CALUDE_gcd_630_945_l3979_397903

theorem gcd_630_945 : Nat.gcd 630 945 = 315 := by
  sorry

end NUMINAMATH_CALUDE_gcd_630_945_l3979_397903


namespace NUMINAMATH_CALUDE_arithmetic_sum_formula_main_theorem_l3979_397968

-- Define the sum of an arithmetic sequence from 1 to n
def arithmeticSum (n : ℕ) : ℕ := n * (1 + n) / 2

-- Define the sum of the odd numbers from 1 to 69
def oddSum : ℕ := (1 + 3 + 5 + 7 + 9 + 11 + 13 + 15 + 17 + 19 + 21 + 23 + 25 + 27 + 29 + 31 + 33 + 35 + 37 + 39 + 41 + 43 + 45 + 47 + 49 + 51 + 53 + 55 + 57 + 59 + 61 + 63 + 65 + 67 + 69)

-- Theorem stating the correctness of the arithmetic sum formula
theorem arithmetic_sum_formula (n : ℕ) : 
  (List.range n).sum = arithmeticSum n :=
by sorry

-- Given condition
axiom odd_sum_condition : 3 * oddSum = 3675

-- Main theorem to prove
theorem main_theorem (n : ℕ) :
  (List.range n).sum = n * (1 + n) / 2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sum_formula_main_theorem_l3979_397968


namespace NUMINAMATH_CALUDE_joe_cookies_sold_l3979_397966

/-- The number of cookies Joe sold -/
def cookies : ℕ := sorry

/-- The cost to make each cookie in dollars -/
def cost : ℚ := 1

/-- The markup percentage -/
def markup : ℚ := 20 / 100

/-- The selling price of each cookie -/
def selling_price : ℚ := cost * (1 + markup)

/-- The total revenue in dollars -/
def revenue : ℚ := 60

theorem joe_cookies_sold :
  cookies = 50 ∧
  selling_price * cookies = revenue :=
sorry

end NUMINAMATH_CALUDE_joe_cookies_sold_l3979_397966


namespace NUMINAMATH_CALUDE_comic_book_stacking_order_l3979_397988

theorem comic_book_stacking_order : 
  let spiderman := 6
  let archie := 5
  let garfield := 4
  let batman := 3
  let superman := 2
  let total_groups := 5
  (spiderman.factorial * archie.factorial * garfield.factorial * batman.factorial * superman.factorial * total_groups.factorial : ℕ) = 1492992000 := by
  sorry

end NUMINAMATH_CALUDE_comic_book_stacking_order_l3979_397988


namespace NUMINAMATH_CALUDE_honest_person_different_answers_possible_l3979_397933

-- Define a person who always tells the truth
structure HonestPerson where
  name : String
  always_truthful : Bool

-- Define a question with its context
structure Question where
  text : String
  context : String

-- Define an answer
structure Answer where
  text : String

-- Define a function to represent a person answering a question
def answer (person : HonestPerson) (q : Question) : Answer :=
  sorry

-- Theorem: It's possible for an honest person to give different answers to the same question asked twice
theorem honest_person_different_answers_possible 
  (person : HonestPerson) 
  (q : Question) 
  (q_repeated : Question) 
  (different_context : q.context ≠ q_repeated.context) :
  ∃ (a1 a2 : Answer), 
    person.always_truthful = true ∧ 
    q.text = q_repeated.text ∧ 
    answer person q = a1 ∧ 
    answer person q_repeated = a2 ∧ 
    a1 ≠ a2 :=
  sorry

end NUMINAMATH_CALUDE_honest_person_different_answers_possible_l3979_397933


namespace NUMINAMATH_CALUDE_probability_of_pair_l3979_397940

/-- Represents a standard deck of cards -/
def StandardDeck := 52

/-- Represents the number of cards of each rank in a standard deck -/
def CardsPerRank := 4

/-- Represents the number of ranks in a standard deck -/
def NumRanks := 13

/-- Represents the number of cards remaining after removing a pair -/
def RemainingCards := StandardDeck - 2

/-- Represents the number of ways to choose 2 cards from the remaining deck -/
def TotalChoices := (RemainingCards.choose 2)

/-- Represents the number of ranks with 4 cards after removing a pair -/
def FullRanks := NumRanks - 1

/-- Represents the number of ways to form a pair from ranks with 4 cards -/
def PairsFromFullRanks := FullRanks * (CardsPerRank.choose 2)

/-- Represents the number of ways to form a pair from the rank with 2 cards -/
def PairsFromReducedRank := 1

/-- Represents the total number of ways to form a pair -/
def TotalPairs := PairsFromFullRanks + PairsFromReducedRank

/-- The main theorem stating the probability of forming a pair -/
theorem probability_of_pair : 
  (TotalPairs : ℚ) / TotalChoices = 73 / 1225 := by sorry

end NUMINAMATH_CALUDE_probability_of_pair_l3979_397940


namespace NUMINAMATH_CALUDE_correct_operation_l3979_397928

theorem correct_operation : ∀ x : ℝ, 
  (∃ y : ℝ, y ^ 2 = 4 ∧ y > 0) ∧ 
  (3 * x^3 + 2 * x^3 ≠ 5 * x^6) ∧ 
  ((x + 1)^2 ≠ x^2 + 1) ∧ 
  (x^8 / x^4 ≠ x^2) :=
by sorry

end NUMINAMATH_CALUDE_correct_operation_l3979_397928


namespace NUMINAMATH_CALUDE_power_division_equality_l3979_397908

theorem power_division_equality (a b : ℝ) : (a^2 * b)^3 / ((-a * b)^2) = a^4 * b := by sorry

end NUMINAMATH_CALUDE_power_division_equality_l3979_397908


namespace NUMINAMATH_CALUDE_contrapositive_odd_product_l3979_397970

theorem contrapositive_odd_product (a b : ℤ) : 
  (((a % 2 = 1 ∧ b % 2 = 1) → (a * b) % 2 = 1) ↔ 
   ((a * b) % 2 ≠ 1 → (a % 2 ≠ 1 ∨ b % 2 ≠ 1))) ∧
  (∀ a b : ℤ, (a * b) % 2 ≠ 1 → (a % 2 ≠ 1 ∨ b % 2 ≠ 1)) := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_odd_product_l3979_397970


namespace NUMINAMATH_CALUDE_fish_ratio_l3979_397907

theorem fish_ratio (jerk_fish : ℕ) (total_fish : ℕ) : 
  jerk_fish = 144 → total_fish = 432 → 
  (total_fish - jerk_fish) / jerk_fish = 2 := by
sorry

end NUMINAMATH_CALUDE_fish_ratio_l3979_397907


namespace NUMINAMATH_CALUDE_bryan_continents_l3979_397992

/-- The number of books Bryan collected per continent -/
def books_per_continent : ℕ := 122

/-- The total number of books Bryan collected from all continents -/
def total_books : ℕ := 488

/-- The number of continents Bryan collected books from -/
def num_continents : ℕ := total_books / books_per_continent

theorem bryan_continents :
  num_continents = 4 := by sorry

end NUMINAMATH_CALUDE_bryan_continents_l3979_397992


namespace NUMINAMATH_CALUDE_quadratic_root_theorem_l3979_397932

theorem quadratic_root_theorem (a b : ℝ) (ha : a ≠ 0) :
  (∃ x : ℝ, x^2 + b*x + a = 0 ∧ x = -a) → b - a = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_theorem_l3979_397932


namespace NUMINAMATH_CALUDE_polly_breakfast_time_l3979_397904

/-- The number of minutes Polly spends cooking breakfast every day -/
def breakfast_time : ℕ := sorry

/-- The number of minutes Polly spends cooking lunch every day -/
def lunch_time : ℕ := 5

/-- The number of days in a week Polly spends 10 minutes cooking dinner -/
def short_dinner_days : ℕ := 4

/-- The number of minutes Polly spends cooking dinner on short dinner days -/
def short_dinner_time : ℕ := 10

/-- The number of minutes Polly spends cooking dinner on long dinner days -/
def long_dinner_time : ℕ := 30

/-- The total number of minutes Polly spends cooking in a week -/
def total_cooking_time : ℕ := 305

/-- The number of days in a week -/
def days_in_week : ℕ := 7

theorem polly_breakfast_time :
  breakfast_time * days_in_week +
  lunch_time * days_in_week +
  short_dinner_time * short_dinner_days +
  long_dinner_time * (days_in_week - short_dinner_days) =
  total_cooking_time ∧
  breakfast_time = 20 := by sorry

end NUMINAMATH_CALUDE_polly_breakfast_time_l3979_397904


namespace NUMINAMATH_CALUDE_largest_digit_sum_l3979_397973

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def isDigit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

theorem largest_digit_sum (a b c : ℕ) (y : ℕ) :
  isDigit a → isDigit b → isDigit c →
  isPrime y →
  0 ≤ y ∧ y ≤ 7 →
  (a * 100 + b * 10 + c : ℚ) / 1000 = 1 / y →
  a + b + c ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_largest_digit_sum_l3979_397973


namespace NUMINAMATH_CALUDE_complex_abs_sum_l3979_397946

theorem complex_abs_sum : Complex.abs (3 - 5*I) + Complex.abs (3 + 8*I) = Real.sqrt 34 + Real.sqrt 73 := by
  sorry

end NUMINAMATH_CALUDE_complex_abs_sum_l3979_397946


namespace NUMINAMATH_CALUDE_interest_rate_difference_l3979_397975

/-- Proves that given a principal of $600 invested for 6 years, if the difference in interest earned between two rates is $144, then the difference between these two rates is 4%. -/
theorem interest_rate_difference (principal : ℝ) (time : ℝ) (interest_diff : ℝ) 
  (h1 : principal = 600)
  (h2 : time = 6)
  (h3 : interest_diff = 144) :
  ∃ (original_rate higher_rate : ℝ),
    (principal * time * higher_rate / 100 - principal * time * original_rate / 100 = interest_diff) ∧
    (higher_rate - original_rate = 4) := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_difference_l3979_397975


namespace NUMINAMATH_CALUDE_expansion_equality_l3979_397913

theorem expansion_equality (m : ℝ) : (m + 2) * (m - 3) = m^2 - m - 6 := by
  sorry

end NUMINAMATH_CALUDE_expansion_equality_l3979_397913


namespace NUMINAMATH_CALUDE_inequality_implication_l3979_397906

theorem inequality_implication (a b c : ℝ) (h : a < b) : -a * c^2 ≥ -b * c^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l3979_397906


namespace NUMINAMATH_CALUDE_cube_equation_solution_l3979_397986

theorem cube_equation_solution (x : ℝ) : (x + 1)^3 = -27 → x = -4 := by
  sorry

end NUMINAMATH_CALUDE_cube_equation_solution_l3979_397986


namespace NUMINAMATH_CALUDE_florist_roses_problem_l3979_397984

theorem florist_roses_problem (x : ℕ) : 
  x - 2 + 32 = 41 → x = 11 := by
  sorry

end NUMINAMATH_CALUDE_florist_roses_problem_l3979_397984


namespace NUMINAMATH_CALUDE_square_difference_identity_l3979_397919

theorem square_difference_identity (x : ℝ) : (x + 1)^2 - x^2 = 2*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_identity_l3979_397919


namespace NUMINAMATH_CALUDE_compound_interest_existence_l3979_397989

/-- Proves the existence of a principal amount and interest rate satisfying the compound interest conditions --/
theorem compound_interest_existence : ∃ (P r : ℝ), 
  P * (1 + r)^2 = 8840 ∧ P * (1 + r)^3 = 9261 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_existence_l3979_397989


namespace NUMINAMATH_CALUDE_square_perimeter_l3979_397982

theorem square_perimeter (area : ℝ) (side : ℝ) (h1 : area = 625) (h2 : side * side = area) :
  4 * side = 100 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l3979_397982


namespace NUMINAMATH_CALUDE_exists_n_for_all_k_l3979_397983

theorem exists_n_for_all_k (k : ℕ) : ∃ n : ℕ, 
  Real.sqrt (n + 1981^k) + Real.sqrt n = (Real.sqrt 1982 + 1)^k := by
  sorry

end NUMINAMATH_CALUDE_exists_n_for_all_k_l3979_397983


namespace NUMINAMATH_CALUDE_absolute_value_of_w_l3979_397979

theorem absolute_value_of_w (s : ℝ) (w : ℂ) (h1 : |s| < 3) (h2 : w + 2 / w = s) : Complex.abs w = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_of_w_l3979_397979


namespace NUMINAMATH_CALUDE_direction_vector_l3979_397930

-- Define the line equation
def line_eq (x y : ℝ) : Prop := y = (4 * x - 6) / 5

-- Define the parameterization
def parameterization (t : ℝ) (d : ℝ × ℝ) : ℝ × ℝ :=
  (4 + t * d.1, 2 + t * d.2)

-- Define the distance condition
def distance_condition (x y : ℝ) (t : ℝ) : Prop :=
  x ≥ 4 → (x - 4)^2 + (y - 2)^2 = t^2

-- Theorem statement
theorem direction_vector :
  ∃ (d : ℝ × ℝ),
    (∀ x y t, line_eq x y →
      (x, y) = parameterization t d →
      distance_condition x y t) →
    d = (5 / Real.sqrt 41, 4 / Real.sqrt 41) :=
sorry

end NUMINAMATH_CALUDE_direction_vector_l3979_397930


namespace NUMINAMATH_CALUDE_trapezoid_determines_plane_l3979_397921

/-- A point in 3D space --/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A trapezoid in 3D space --/
structure Trapezoid where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  parallel_sides : (A.x - B.x) * (C.y - D.y) = (A.y - B.y) * (C.x - D.x) ∧
                   (A.x - D.x) * (B.y - C.y) = (A.y - D.y) * (B.x - C.x)

/-- A plane in 3D space --/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Function to determine if a point lies on a plane --/
def point_on_plane (p : Point3D) (plane : Plane) : Prop :=
  plane.a * p.x + plane.b * p.y + plane.c * p.z + plane.d = 0

/-- Theorem: A trapezoid uniquely determines a plane --/
theorem trapezoid_determines_plane (t : Trapezoid) :
  ∃! plane : Plane, point_on_plane t.A plane ∧
                    point_on_plane t.B plane ∧
                    point_on_plane t.C plane ∧
                    point_on_plane t.D plane :=
sorry

end NUMINAMATH_CALUDE_trapezoid_determines_plane_l3979_397921


namespace NUMINAMATH_CALUDE_hollow_cube_5x5x5_l3979_397955

/-- The number of cubes needed for a hollow cube -/
def hollow_cube_cubes (n : ℕ) : ℕ :=
  6 * (n^2 - (n-2)^2) - 12 * (n-2)

/-- Theorem: A hollow cube with outer dimensions 5 * 5 * 5 requires 60 cubes -/
theorem hollow_cube_5x5x5 : hollow_cube_cubes 5 = 60 := by
  sorry

end NUMINAMATH_CALUDE_hollow_cube_5x5x5_l3979_397955


namespace NUMINAMATH_CALUDE_pine_seedlings_in_sample_l3979_397942

/-- Represents a forest with seedlings -/
structure Forest where
  total_seedlings : ℕ
  pine_seedlings : ℕ
  sample_size : ℕ

/-- Calculates the expected number of pine seedlings in a sample -/
def expected_pine_seedlings (f : Forest) : ℚ :=
  (f.pine_seedlings : ℚ) * (f.sample_size : ℚ) / (f.total_seedlings : ℚ)

/-- Theorem stating the expected number of pine seedlings in the sample -/
theorem pine_seedlings_in_sample (f : Forest) 
  (h1 : f.total_seedlings = 30000)
  (h2 : f.pine_seedlings = 4000)
  (h3 : f.sample_size = 150) :
  expected_pine_seedlings f = 20 := by
  sorry

end NUMINAMATH_CALUDE_pine_seedlings_in_sample_l3979_397942


namespace NUMINAMATH_CALUDE_add_fractions_three_fourths_five_ninths_l3979_397941

theorem add_fractions_three_fourths_five_ninths :
  (3 : ℚ) / 4 + (5 : ℚ) / 9 = (47 : ℚ) / 36 := by
  sorry

end NUMINAMATH_CALUDE_add_fractions_three_fourths_five_ninths_l3979_397941


namespace NUMINAMATH_CALUDE_tree_height_difference_l3979_397945

theorem tree_height_difference :
  let pine_height : ℚ := 53/4
  let maple_height : ℚ := 41/2
  maple_height - pine_height = 29/4 := by sorry

end NUMINAMATH_CALUDE_tree_height_difference_l3979_397945


namespace NUMINAMATH_CALUDE_max_value_theorem_l3979_397953

theorem max_value_theorem (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x^2 + y^2 + z^2 = 1) :
  2 * x * y * Real.sqrt 2 + 6 * y * z ≤ Real.sqrt 11 :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3979_397953


namespace NUMINAMATH_CALUDE_max_shapes_from_7x7_grid_l3979_397943

/-- Represents a grid with dimensions n x n -/
structure Grid (n : ℕ) where
  size : ℕ := n * n

/-- Represents a shape that can be cut from the grid -/
inductive Shape
  | Square : Shape  -- 2x2 square
  | Rectangle : Shape  -- 1x4 rectangle

/-- The size of a shape in terms of grid cells -/
def shapeSize : Shape → ℕ
  | Shape.Square => 4
  | Shape.Rectangle => 4

/-- The maximum number of shapes that can be cut from a grid -/
def maxShapes (g : Grid 7) : ℕ :=
  g.size / shapeSize Shape.Square

/-- Checks if a number of shapes can be equally divided between squares and rectangles -/
def isEquallyDivisible (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2 * k

theorem max_shapes_from_7x7_grid :
  ∃ (n : ℕ), maxShapes (Grid.mk 7) = n ∧ isEquallyDivisible n ∧ n = 12 := by
  sorry

end NUMINAMATH_CALUDE_max_shapes_from_7x7_grid_l3979_397943


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3979_397900

def vector_a : ℝ × ℝ := (3, 1)
def vector_b (x : ℝ) : ℝ × ℝ := (x, -1)

theorem parallel_vectors_x_value :
  ∀ x : ℝ, (vector_a.1 * (vector_b x).2 = vector_a.2 * (vector_b x).1) → x = -3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3979_397900


namespace NUMINAMATH_CALUDE_problem_solution_l3979_397954

theorem problem_solution (x y : ℝ) : 
  x = 201 → x^3 * y - 2 * x^2 * y + x * y = 804000 → y = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3979_397954


namespace NUMINAMATH_CALUDE_rebecca_work_hours_l3979_397902

/-- Given the working hours of Thomas, Toby, and Rebecca, prove that Rebecca worked 56 hours. -/
theorem rebecca_work_hours :
  ∀ x : ℕ,
  let thomas_hours := x
  let toby_hours := 2 * x - 10
  let rebecca_hours := toby_hours - 8
  (thomas_hours + toby_hours + rebecca_hours = 157) →
  rebecca_hours = 56 :=
by
  sorry

end NUMINAMATH_CALUDE_rebecca_work_hours_l3979_397902


namespace NUMINAMATH_CALUDE_no_ab_term_in_polynomial_l3979_397999

theorem no_ab_term_in_polynomial (m : ℝ) : 
  (∀ a b : ℝ, (a^2 + 2*a*b - b^2) - (a^2 + m*a*b + 2*b^2) = (-3:ℝ)*b^2) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_no_ab_term_in_polynomial_l3979_397999


namespace NUMINAMATH_CALUDE_choose_product_equals_8400_l3979_397926

theorem choose_product_equals_8400 : Nat.choose 10 3 * Nat.choose 8 4 = 8400 := by
  sorry

end NUMINAMATH_CALUDE_choose_product_equals_8400_l3979_397926


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l3979_397905

theorem coefficient_x_squared_in_expansion (x : ℝ) :
  (Finset.range 6).sum (fun k => (Nat.choose 5 k : ℝ) * x^k * (1:ℝ)^(5-k)) =
  10 * x^2 + (Finset.range 6).sum (fun k => if k ≠ 2 then (Nat.choose 5 k : ℝ) * x^k * (1:ℝ)^(5-k) else 0) :=
by sorry


end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l3979_397905


namespace NUMINAMATH_CALUDE_quadrilateral_perimeter_l3979_397977

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the perimeter function
def perimeter (q : Quadrilateral) : ℝ := sorry

-- Define the length function
def length (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define perpendicularity
def perpendicular (p1 p2 p3 p4 : ℝ × ℝ) : Prop := sorry

theorem quadrilateral_perimeter (ABCD : Quadrilateral) :
  perpendicular ABCD.A ABCD.B ABCD.B ABCD.C →
  perpendicular ABCD.D ABCD.C ABCD.B ABCD.C →
  length ABCD.A ABCD.B = 7 →
  length ABCD.D ABCD.C = 3 →
  length ABCD.B ABCD.C = 10 →
  length ABCD.A ABCD.C = 15 →
  perimeter ABCD = 20 + 6 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_perimeter_l3979_397977


namespace NUMINAMATH_CALUDE_origin_inside_ellipse_iff_k_range_l3979_397939

/-- The ellipse equation -/
def ellipse_equation (k x y : ℝ) : ℝ := k^2 * x^2 + y^2 - 4*k*x + 2*k*y + k^2 - 1

/-- The origin is inside the ellipse -/
def origin_inside_ellipse (k : ℝ) : Prop := ellipse_equation k 0 0 < 0

/-- Theorem: The origin is inside the ellipse if and only if 0 < |k| < 1 -/
theorem origin_inside_ellipse_iff_k_range (k : ℝ) : 
  origin_inside_ellipse k ↔ 0 < |k| ∧ |k| < 1 := by sorry

end NUMINAMATH_CALUDE_origin_inside_ellipse_iff_k_range_l3979_397939


namespace NUMINAMATH_CALUDE_mindmaster_codes_l3979_397947

/-- The number of available colors in the Mindmaster game -/
def num_colors : ℕ := 8

/-- The number of slots in each code -/
def num_slots : ℕ := 4

/-- The total number of possible secret codes in the Mindmaster game -/
def total_codes : ℕ := num_colors ^ num_slots

/-- Theorem stating that the total number of secret codes is 4096 -/
theorem mindmaster_codes : total_codes = 4096 := by
  sorry

end NUMINAMATH_CALUDE_mindmaster_codes_l3979_397947


namespace NUMINAMATH_CALUDE_infinite_log_3_64_equals_4_l3979_397957

noncomputable def log_3 (x : ℝ) : ℝ := Real.log x / Real.log 3

theorem infinite_log_3_64_equals_4 :
  ∃! x : ℝ, x > 0 ∧ x = log_3 (64 + x) ∧ x = 4 := by sorry

end NUMINAMATH_CALUDE_infinite_log_3_64_equals_4_l3979_397957


namespace NUMINAMATH_CALUDE_bridesmaids_dresses_completion_time_l3979_397910

def dress_hours : List Nat := [15, 18, 20, 22, 24, 26, 28]
def hours_per_week : Nat := 5

theorem bridesmaids_dresses_completion_time : 
  ∃ (total_hours : Nat),
    total_hours = dress_hours.sum ∧
    (total_hours / hours_per_week : ℚ) ≤ 31 ∧
    31 < (total_hours / hours_per_week : ℚ) + 1 :=
by sorry

end NUMINAMATH_CALUDE_bridesmaids_dresses_completion_time_l3979_397910


namespace NUMINAMATH_CALUDE_symmetry_properties_l3979_397969

def Point := ℝ × ℝ

def symmetricAboutXAxis (p : Point) : Point :=
  (p.1, -p.2)

def symmetricAboutYAxis (p : Point) : Point :=
  (-p.1, p.2)

theorem symmetry_properties (x y : ℝ) :
  let A : Point := (x, y)
  (symmetricAboutXAxis A = (x, -y)) ∧
  (symmetricAboutYAxis A = (-x, y)) := by
  sorry

end NUMINAMATH_CALUDE_symmetry_properties_l3979_397969


namespace NUMINAMATH_CALUDE_laura_walk_distance_l3979_397959

/-- Calculates the total distance walked in miles given the number of blocks walked east and north, and the length of each block in miles. -/
def total_distance (blocks_east blocks_north : ℕ) (block_length : ℚ) : ℚ :=
  (blocks_east + blocks_north : ℚ) * block_length

/-- Proves that walking 8 blocks east and 14 blocks north, with each block being 1/4 mile, results in a total distance of 5.5 miles. -/
theorem laura_walk_distance : total_distance 8 14 (1/4) = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_laura_walk_distance_l3979_397959


namespace NUMINAMATH_CALUDE_distinct_two_mark_grids_l3979_397994

/-- Represents a 4x4 grid --/
def Grid := Fin 4 → Fin 4 → Bool

/-- Represents a rotation of the grid --/
inductive Rotation
| r0 | r90 | r180 | r270

/-- Applies a rotation to a grid --/
def applyRotation (r : Rotation) (g : Grid) : Grid :=
  sorry

/-- Checks if two grids are equivalent under rotation --/
def areEquivalent (g1 g2 : Grid) : Bool :=
  sorry

/-- Counts the number of marked cells in a grid --/
def countMarked (g : Grid) : Nat :=
  sorry

/-- Generates all possible grids with exactly two marked cells --/
def allGridsWithTwoMarked : List Grid :=
  sorry

/-- Counts the number of distinct grids under rotation --/
def countDistinctGrids (grids : List Grid) : Nat :=
  sorry

/-- The main theorem to be proved --/
theorem distinct_two_mark_grids :
  countDistinctGrids allGridsWithTwoMarked = 32 :=
sorry

end NUMINAMATH_CALUDE_distinct_two_mark_grids_l3979_397994


namespace NUMINAMATH_CALUDE_range_of_m_for_real_roots_l3979_397918

theorem range_of_m_for_real_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 - x + m = 0 ∨ (m-1)*x^2 + 2*x + 1 = 0 ∨ (m-2)*x^2 + 2*x - 1 = 0) →
  (∃ y : ℝ, y^2 - y + m = 0 ∨ (m-1)*y^2 + 2*y + 1 = 0 ∨ (m-2)*y^2 + 2*y - 1 = 0) →
  (x ≠ y) →
  (m ≤ 1/4 ∨ (1 ≤ m ∧ m ≤ 2)) := by
sorry

end NUMINAMATH_CALUDE_range_of_m_for_real_roots_l3979_397918


namespace NUMINAMATH_CALUDE_mod_nineteen_problem_l3979_397981

theorem mod_nineteen_problem : ∃! n : ℤ, 0 ≤ n ∧ n < 19 ∧ 38574 ≡ n [ZMOD 19] := by sorry

end NUMINAMATH_CALUDE_mod_nineteen_problem_l3979_397981


namespace NUMINAMATH_CALUDE_a_minus_b_value_l3979_397931

theorem a_minus_b_value (a b : ℝ) 
  (h1 : a^2 * b - a * b^2 = -6) 
  (h2 : a * b = 3) : 
  a - b = -2 := by
sorry

end NUMINAMATH_CALUDE_a_minus_b_value_l3979_397931


namespace NUMINAMATH_CALUDE_function_properties_l3979_397917

open Real

theorem function_properties (f : ℝ → ℝ) (a b : ℝ) (h_cont : ContinuousOn f (Set.Icc a b))
    (h_diff : DifferentiableOn ℝ f (Set.Icc a b)) (h_a_lt_b : a < b)
    (h_f'_a : deriv f a > 0) (h_f'_b : deriv f b < 0) :
  (∃ x₀ ∈ Set.Icc a b, f x₀ > f b) ∧
  (∃ x₀ ∈ Set.Icc a b, f a - f b = (deriv (deriv f)) x₀ * (a - b)) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3979_397917


namespace NUMINAMATH_CALUDE_log_product_less_than_one_l3979_397911

theorem log_product_less_than_one : Real.log 9 * Real.log 11 < 1 := by
  sorry

end NUMINAMATH_CALUDE_log_product_less_than_one_l3979_397911


namespace NUMINAMATH_CALUDE_max_value_at_negative_one_l3979_397920

-- Define the function f
def f (a b x : ℝ) : ℝ := x^3 + 3*a*x^2 + b*x + a^2

-- State the theorem
theorem max_value_at_negative_one (a b : ℝ) :
  (∀ x, f a b x ≤ 0) ∧
  (f a b (-1) = 0) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x, |x - (-1)| < δ → f a b x < f a b (-1) + ε) →
  a + b = 11 :=
by sorry

end NUMINAMATH_CALUDE_max_value_at_negative_one_l3979_397920


namespace NUMINAMATH_CALUDE_oranges_left_l3979_397980

def initial_oranges : ℕ := 55
def oranges_taken : ℕ := 35

theorem oranges_left : initial_oranges - oranges_taken = 20 := by
  sorry

end NUMINAMATH_CALUDE_oranges_left_l3979_397980


namespace NUMINAMATH_CALUDE_sequence_2023rd_term_l3979_397974

theorem sequence_2023rd_term (a : ℕ → ℝ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, (a n / 2) - (1 / (2 * a (n + 1))) = a (n + 1) - (1 / a n)) :
  a 2023 = 1 ∨ a 2023 = (1 / 2) ^ 2022 := by
  sorry

end NUMINAMATH_CALUDE_sequence_2023rd_term_l3979_397974


namespace NUMINAMATH_CALUDE_bacon_calorie_percentage_example_l3979_397962

/-- The percentage of calories from bacon in a sandwich -/
def bacon_calorie_percentage (total_calories : ℕ) (bacon_strips : ℕ) (calories_per_strip : ℕ) : ℚ :=
  (bacon_strips * calories_per_strip : ℚ) / total_calories * 100

/-- Theorem stating that the percentage of calories from bacon in a 1250-calorie sandwich with two 125-calorie bacon strips is 20% -/
theorem bacon_calorie_percentage_example :
  bacon_calorie_percentage 1250 2 125 = 20 := by
  sorry

end NUMINAMATH_CALUDE_bacon_calorie_percentage_example_l3979_397962


namespace NUMINAMATH_CALUDE_zoo_animals_legs_count_l3979_397914

theorem zoo_animals_legs_count : 
  ∀ (total_heads : ℕ) (rabbit_count : ℕ) (peacock_count : ℕ),
    total_heads = 60 →
    rabbit_count = 36 →
    peacock_count = total_heads - rabbit_count →
    4 * rabbit_count + 2 * peacock_count = 192 :=
by
  sorry

end NUMINAMATH_CALUDE_zoo_animals_legs_count_l3979_397914


namespace NUMINAMATH_CALUDE_odd_composite_sum_representation_l3979_397937

/-- A number is composite if it has a factor other than 1 and itself -/
def IsComposite (n : ℕ) : Prop :=
  ∃ k m : ℕ, k > 1 ∧ m > 1 ∧ n = k * m

/-- An odd number can be represented as the sum of two composite numbers -/
def CanBeRepresentedAsCompositeSum (n : ℕ) : Prop :=
  ∃ a b : ℕ, IsComposite a ∧ IsComposite b ∧ n = a + b

theorem odd_composite_sum_representation :
  ∀ n : ℕ, n ≥ 13 → Odd n → CanBeRepresentedAsCompositeSum n := by
  sorry

#check odd_composite_sum_representation

end NUMINAMATH_CALUDE_odd_composite_sum_representation_l3979_397937


namespace NUMINAMATH_CALUDE_rahul_deepak_age_ratio_l3979_397997

theorem rahul_deepak_age_ratio :
  ∀ (rahul_age deepak_age : ℕ),
    deepak_age = 12 →
    rahul_age + 6 = 22 →
    rahul_age / deepak_age = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_rahul_deepak_age_ratio_l3979_397997


namespace NUMINAMATH_CALUDE_total_books_l3979_397950

theorem total_books (tim_books sam_books : ℕ) 
  (h1 : tim_books = 44) 
  (h2 : sam_books = 52) : 
  tim_books + sam_books = 96 := by
sorry

end NUMINAMATH_CALUDE_total_books_l3979_397950


namespace NUMINAMATH_CALUDE_difference_of_squares_l3979_397912

theorem difference_of_squares (x : ℝ) : 1 - x^2 = (1 - x) * (1 + x) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3979_397912


namespace NUMINAMATH_CALUDE_complex_equality_l3979_397972

theorem complex_equality (ω : ℂ) :
  Complex.abs (ω - 2) = Complex.abs (ω - 2 * Complex.I) →
  ω.re = ω.im :=
by sorry

end NUMINAMATH_CALUDE_complex_equality_l3979_397972


namespace NUMINAMATH_CALUDE_rectangular_box_surface_area_l3979_397934

/-- 
Given a rectangular box with dimensions a, b, and c, 
if the sum of the lengths of its twelve edges is 172 
and the distance from one corner to the farthest corner is 21, 
then its total surface area is 1408.
-/
theorem rectangular_box_surface_area 
  (a b c : ℝ) 
  (h1 : 4 * a + 4 * b + 4 * c = 172) 
  (h2 : Real.sqrt (a^2 + b^2 + c^2) = 21) : 
  2 * (a * b + b * c + c * a) = 1408 := by
sorry

end NUMINAMATH_CALUDE_rectangular_box_surface_area_l3979_397934


namespace NUMINAMATH_CALUDE_seven_boys_handshakes_l3979_397985

/-- The number of handshakes between n boys, where each boy shakes hands exactly once with each of the others -/
def num_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that the number of handshakes between 7 boys is 21 -/
theorem seven_boys_handshakes : num_handshakes 7 = 21 := by
  sorry

end NUMINAMATH_CALUDE_seven_boys_handshakes_l3979_397985


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l3979_397925

theorem min_value_of_sum_of_squares (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) :
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 1 → (a + 2)^2 + (b + 2)^2 ≤ (x + 2)^2 + (y + 2)^2) ∧
  (a + 2)^2 + (b + 2)^2 = 25/2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l3979_397925


namespace NUMINAMATH_CALUDE_smallest_product_l3979_397976

def S : Finset Int := {-9, -7, -4, 2, 5, 7}

theorem smallest_product (a b : Int) :
  a ∈ S → b ∈ S → a * b ≥ -63 ∧ ∃ (x y : Int), x ∈ S ∧ y ∈ S ∧ x * y = -63 := by
  sorry

end NUMINAMATH_CALUDE_smallest_product_l3979_397976


namespace NUMINAMATH_CALUDE_saree_price_after_discounts_l3979_397998

def original_price : ℝ := 1000

def discount1 : ℝ := 0.30
def discount2 : ℝ := 0.15
def discount3 : ℝ := 0.10
def discount4 : ℝ := 0.05

def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

def final_price : ℝ :=
  apply_discount (apply_discount (apply_discount (apply_discount original_price discount1) discount2) discount3) discount4

theorem saree_price_after_discounts :
  ⌊final_price⌋ = 509 := by sorry

end NUMINAMATH_CALUDE_saree_price_after_discounts_l3979_397998


namespace NUMINAMATH_CALUDE_sum_with_radical_conjugate_l3979_397996

theorem sum_with_radical_conjugate : 
  let x : ℝ := 12 - Real.sqrt 5000
  let y : ℝ := 12 + Real.sqrt 5000  -- radical conjugate
  x + y = 24 := by
sorry

end NUMINAMATH_CALUDE_sum_with_radical_conjugate_l3979_397996


namespace NUMINAMATH_CALUDE_bob_probability_after_three_turns_l3979_397929

/-- Represents the player who has the ball -/
inductive Player : Type
| Alice : Player
| Bob : Player

/-- The game state after a certain number of turns -/
structure GameState :=
  (current_player : Player)
  (turn : ℕ)

/-- The probability of a player having the ball after a certain number of turns -/
def probability_has_ball (player : Player) (turns : ℕ) : ℚ :=
  sorry

theorem bob_probability_after_three_turns :
  probability_has_ball Player.Bob 3 = 11/16 := by
  sorry

end NUMINAMATH_CALUDE_bob_probability_after_three_turns_l3979_397929


namespace NUMINAMATH_CALUDE_root_of_polynomial_l3979_397922

theorem root_of_polynomial (a₁ a₂ a₃ a₄ a₅ b : ℤ) : 
  a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₁ ≠ a₄ ∧ a₁ ≠ a₅ ∧ 
  a₂ ≠ a₃ ∧ a₂ ≠ a₄ ∧ a₂ ≠ a₅ ∧ 
  a₃ ≠ a₄ ∧ a₃ ≠ a₅ ∧ 
  a₄ ≠ a₅ →
  a₁ + a₂ + a₃ + a₄ + a₅ = 9 →
  (b - a₁) * (b - a₂) * (b - a₃) * (b - a₄) * (b - a₅) = 2009 →
  b = 10 := by
sorry

end NUMINAMATH_CALUDE_root_of_polynomial_l3979_397922


namespace NUMINAMATH_CALUDE_coefficient_x2y4_in_expansion_l3979_397961

/-- The coefficient of x^2y^4 in the expansion of (1+x+y^2)^5 is 30 -/
theorem coefficient_x2y4_in_expansion : ℕ := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x2y4_in_expansion_l3979_397961


namespace NUMINAMATH_CALUDE_cos_theta_value_l3979_397923

theorem cos_theta_value (θ : Real) 
  (h : (1 + Real.sin θ + Real.cos θ) / (1 + Real.sin θ - Real.cos θ) = 1/2) : 
  Real.cos θ = -3/5 := by
sorry

end NUMINAMATH_CALUDE_cos_theta_value_l3979_397923


namespace NUMINAMATH_CALUDE_soccer_ball_weight_l3979_397927

theorem soccer_ball_weight (soccer_ball_weight bicycle_weight : ℝ) : 
  5 * soccer_ball_weight = 3 * bicycle_weight →
  2 * bicycle_weight = 60 →
  soccer_ball_weight = 18 := by
  sorry

end NUMINAMATH_CALUDE_soccer_ball_weight_l3979_397927


namespace NUMINAMATH_CALUDE_car_travel_time_difference_l3979_397958

/-- Proves that the time difference between two cars traveling 150 miles is 2 hours,
    given their speeds differ by 10 mph and one car's speed is 22.83882181415011 mph. -/
theorem car_travel_time_difference 
  (distance : ℝ) 
  (speed_R : ℝ) 
  (speed_P : ℝ) : 
  distance = 150 →
  speed_R = 22.83882181415011 →
  speed_P = speed_R + 10 →
  distance / speed_R - distance / speed_P = 2 := by
sorry

end NUMINAMATH_CALUDE_car_travel_time_difference_l3979_397958


namespace NUMINAMATH_CALUDE_largest_pot_cost_l3979_397995

/-- The cost of the largest pot in a set of 6 pots with specific pricing rules -/
theorem largest_pot_cost (total_cost : ℚ) (num_pots : ℕ) (price_diff : ℚ) :
  total_cost = 33/4 ∧ num_pots = 6 ∧ price_diff = 1/10 →
  ∃ (smallest_cost : ℚ),
    smallest_cost > 0 ∧
    (smallest_cost + (num_pots - 1) * price_diff) = 13/8 := by
  sorry

end NUMINAMATH_CALUDE_largest_pot_cost_l3979_397995


namespace NUMINAMATH_CALUDE_total_rectangles_3x3_grid_l3979_397944

/-- Represents a grid of points -/
structure Grid where
  rows : Nat
  cols : Nat

/-- Represents a rectangle on the grid -/
structure Rectangle where
  width : Nat
  height : Nat

/-- Counts the number of rectangles of a given size on the grid -/
def countRectangles (g : Grid) (r : Rectangle) : Nat :=
  sorry

/-- The total number of rectangles on a 3x3 grid -/
def totalRectangles : Nat :=
  let g : Grid := { rows := 3, cols := 3 }
  (countRectangles g { width := 1, height := 1 }) +
  (countRectangles g { width := 1, height := 2 }) +
  (countRectangles g { width := 1, height := 3 }) +
  (countRectangles g { width := 2, height := 1 }) +
  (countRectangles g { width := 2, height := 2 }) +
  (countRectangles g { width := 2, height := 3 }) +
  (countRectangles g { width := 3, height := 1 }) +
  (countRectangles g { width := 3, height := 2 })

/-- Theorem stating that the total number of rectangles on a 3x3 grid is 124 -/
theorem total_rectangles_3x3_grid :
  totalRectangles = 124 := by
  sorry

end NUMINAMATH_CALUDE_total_rectangles_3x3_grid_l3979_397944


namespace NUMINAMATH_CALUDE_probability_AC_less_than_11_l3979_397993

-- Define the given lengths
def AB : ℝ := 10
def BC : ℝ := 6

-- Define the maximum length of AC
def AC_max : ℝ := 11

-- Define the angle α
def α : ℝ → Prop := λ x => 0 < x ∧ x < Real.pi / 2

-- Define the probability function
noncomputable def P : ℝ := (2 / Real.pi) * Real.arctan (4 / (3 * Real.sqrt 63))

-- State the theorem
theorem probability_AC_less_than_11 :
  ∀ x, α x → P = (2 / Real.pi) * Real.arctan (4 / (3 * Real.sqrt 63)) :=
by sorry

end NUMINAMATH_CALUDE_probability_AC_less_than_11_l3979_397993


namespace NUMINAMATH_CALUDE_inscribed_hexagon_area_l3979_397901

/-- A regular hexagon inscribed in a square with specific properties -/
structure InscribedHexagon where
  square_perimeter : ℝ
  square_side_length : ℝ
  hexagon_side_length : ℝ
  hexagon_area : ℝ
  perimeter_constraint : square_perimeter = 160
  side_length_relation : square_side_length = square_perimeter / 4
  hexagon_side_relation : hexagon_side_length = square_side_length / 2
  area_formula : hexagon_area = 3 * Real.sqrt 3 / 2 * hexagon_side_length ^ 2

/-- The theorem stating the area of the inscribed hexagon -/
theorem inscribed_hexagon_area (h : InscribedHexagon) : h.hexagon_area = 600 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_hexagon_area_l3979_397901


namespace NUMINAMATH_CALUDE_solution_to_equation_l3979_397949

theorem solution_to_equation : ∃ x : ℚ, (1/3 - 1/2) * x = 1 ∧ x = -6 := by sorry

end NUMINAMATH_CALUDE_solution_to_equation_l3979_397949


namespace NUMINAMATH_CALUDE_sam_washing_pennies_l3979_397938

/-- The number of pennies Sam earned from washing clothes -/
def pennies_from_washing (total_cents : ℕ) (num_quarters : ℕ) : ℕ :=
  total_cents - (num_quarters * 25)

/-- Theorem: Given 7 quarters and a total of $1.84, Sam earned 9 pennies from washing clothes -/
theorem sam_washing_pennies :
  pennies_from_washing 184 7 = 9 := by
  sorry

end NUMINAMATH_CALUDE_sam_washing_pennies_l3979_397938


namespace NUMINAMATH_CALUDE_problem_statement_l3979_397948

theorem problem_statement (n : ℝ) (h : n + 1/n = 6) : n^2 + 1/n^2 + 9 = 43 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3979_397948


namespace NUMINAMATH_CALUDE_solve_flower_problem_l3979_397965

def flower_problem (yoojung_flowers namjoon_flowers : ℕ) : Prop :=
  (yoojung_flowers = 32) ∧
  (yoojung_flowers = 4 * namjoon_flowers) ∧
  (yoojung_flowers + namjoon_flowers = 40)

theorem solve_flower_problem :
  ∃ (yoojung_flowers namjoon_flowers : ℕ),
    flower_problem yoojung_flowers namjoon_flowers :=
by
  sorry

end NUMINAMATH_CALUDE_solve_flower_problem_l3979_397965


namespace NUMINAMATH_CALUDE_moores_law_transistor_growth_l3979_397963

/-- Moore's Law Transistor Growth --/
theorem moores_law_transistor_growth
  (initial_year : Nat)
  (final_year : Nat)
  (initial_transistors : Nat)
  (doubling_period : Nat)
  (h1 : initial_year = 1985)
  (h2 : final_year = 2005)
  (h3 : initial_transistors = 500000)
  (h4 : doubling_period = 2) :
  initial_transistors * 2^((final_year - initial_year) / doubling_period) = 512000000 :=
by sorry

end NUMINAMATH_CALUDE_moores_law_transistor_growth_l3979_397963


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l3979_397987

theorem other_root_of_quadratic (m : ℝ) : 
  (2^2 - 5*2 - m = 0) → 
  ∃ (t : ℝ), t ≠ 2 ∧ t^2 - 5*t - m = 0 ∧ t = 3 :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l3979_397987


namespace NUMINAMATH_CALUDE_tangent_line_slope_l3979_397971

/-- Given a curve y = ax + e^x - 1 and its tangent line y = 3x at (0,0), prove a = 2 -/
theorem tangent_line_slope (a : ℝ) : 
  (∀ x y : ℝ, y = a * x + Real.exp x - 1) →  -- Curve equation
  (∃ m : ℝ, m = 3 ∧ ∀ x y : ℝ, y = m * x) →  -- Tangent line equation
  (∀ ε > 0, ∃ δ > 0, ∀ x, |x| < δ → |((a * x + Real.exp x - 1) - 0) / (x - 0) - 3| < ε) →  -- Tangent condition
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_slope_l3979_397971


namespace NUMINAMATH_CALUDE_pentagon_rectangle_ratio_l3979_397924

theorem pentagon_rectangle_ratio : 
  let pentagon_perimeter : ℝ := 100
  let rectangle_perimeter : ℝ := 100
  let pentagon_side := pentagon_perimeter / 5
  let rectangle_width := rectangle_perimeter / 6
  pentagon_side / rectangle_width = 6 / 5 := by sorry

end NUMINAMATH_CALUDE_pentagon_rectangle_ratio_l3979_397924


namespace NUMINAMATH_CALUDE_committee_selection_l3979_397916

theorem committee_selection (n : ℕ) (h : Nat.choose n 2 = 15) : Nat.choose n 4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_l3979_397916


namespace NUMINAMATH_CALUDE_smallest_cuboid_face_area_l3979_397952

/-- Given a cuboid with integer volume and face areas 7, 27, and L, 
    prove that the smallest possible integer value for L is 21 -/
theorem smallest_cuboid_face_area (a b c : ℕ+) (L : ℕ) : 
  (a * b : ℕ) = 7 →
  (a * c : ℕ) = 27 →
  (b * c : ℕ) = L →
  (∃ (v : ℕ), v = a * b * c) →
  L ≥ 21 ∧ 
  (∀ L' : ℕ, L' ≥ 21 → ∃ (a' b' c' : ℕ+), 
    (a' * b' : ℕ) = 7 ∧ 
    (a' * c' : ℕ) = 27 ∧ 
    (b' * c' : ℕ) = L') :=
by sorry

#check smallest_cuboid_face_area

end NUMINAMATH_CALUDE_smallest_cuboid_face_area_l3979_397952


namespace NUMINAMATH_CALUDE_largest_of_five_consecutive_even_integers_l3979_397964

/-- The sum of the first n positive even integers -/
def sumFirstNEvenIntegers (n : ℕ) : ℕ := n * (n + 1)

/-- The sum of five consecutive even integers starting from k -/
def sumFiveConsecutiveEvenIntegers (k : ℕ) : ℕ := 5 * k - 10

theorem largest_of_five_consecutive_even_integers :
  ∃ k : ℕ, 
    sumFirstNEvenIntegers 30 = sumFiveConsecutiveEvenIntegers k ∧ 
    k = 190 := by
  sorry

#eval sumFirstNEvenIntegers 30  -- Should output 930
#eval sumFiveConsecutiveEvenIntegers 190  -- Should also output 930

end NUMINAMATH_CALUDE_largest_of_five_consecutive_even_integers_l3979_397964


namespace NUMINAMATH_CALUDE_hex_9A3_to_base_4_l3979_397915

/-- Converts a single hexadecimal digit to its decimal representation -/
def hex_to_dec (h : Char) : ℕ :=
  match h with
  | '0' => 0
  | '1' => 1
  | '2' => 2
  | '3' => 3
  | '4' => 4
  | '5' => 5
  | '6' => 6
  | '7' => 7
  | '8' => 8
  | '9' => 9
  | 'A' => 10
  | 'B' => 11
  | 'C' => 12
  | 'D' => 13
  | 'E' => 14
  | 'F' => 15
  | _ => 0  -- Default case, though it should never be reached for valid hex digits

/-- Converts a hexadecimal number (as a string) to its decimal representation -/
def hex_to_dec_num (s : String) : ℕ :=
  s.foldl (fun acc d => 16 * acc + hex_to_dec d) 0

/-- Converts a natural number to its base 4 representation (as a list of digits) -/
def to_base_4 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

/-- The main theorem: 9A3₁₆ is equal to 212203₄ -/
theorem hex_9A3_to_base_4 :
  to_base_4 (hex_to_dec_num "9A3") = [2, 1, 2, 2, 0, 3] := by
  sorry

end NUMINAMATH_CALUDE_hex_9A3_to_base_4_l3979_397915


namespace NUMINAMATH_CALUDE_compound_oxygen_atoms_l3979_397991

/-- Proves that a compound with 6 C atoms, 8 H atoms, and a molecular weight of 192
    contains 7 O atoms, given the atomic weights of C, H, and O. -/
theorem compound_oxygen_atoms 
  (atomic_weight_C : ℝ) 
  (atomic_weight_H : ℝ) 
  (atomic_weight_O : ℝ) 
  (h1 : atomic_weight_C = 12.01)
  (h2 : atomic_weight_H = 1.008)
  (h3 : atomic_weight_O = 16.00)
  (h4 : (6 * atomic_weight_C + 8 * atomic_weight_H + 7 * atomic_weight_O) = 192) :
  ∃ n : ℕ, n = 7 ∧ (6 * atomic_weight_C + 8 * atomic_weight_H + n * atomic_weight_O) = 192 :=
by
  sorry


end NUMINAMATH_CALUDE_compound_oxygen_atoms_l3979_397991


namespace NUMINAMATH_CALUDE_pirate_coins_l3979_397909

/-- Represents the number of pirates --/
def num_pirates : ℕ := 15

/-- Calculates the number of coins remaining after the k-th pirate takes their share --/
def coins_after (k : ℕ) (initial_coins : ℕ) : ℚ :=
  (num_pirates - k : ℚ) / num_pirates * initial_coins

/-- Checks if a given number of initial coins results in each pirate receiving a whole number of coins --/
def valid_distribution (initial_coins : ℕ) : Prop :=
  ∀ k : ℕ, k ≤ num_pirates → (coins_after k initial_coins - coins_after (k+1) initial_coins).isInt

/-- The statement to be proved --/
theorem pirate_coins :
  ∃ initial_coins : ℕ,
    valid_distribution initial_coins ∧
    (∀ n : ℕ, n < initial_coins → ¬valid_distribution n) ∧
    coins_after (num_pirates - 1) initial_coins = 1001 := by
  sorry


end NUMINAMATH_CALUDE_pirate_coins_l3979_397909


namespace NUMINAMATH_CALUDE_favorite_fruit_pears_l3979_397960

theorem favorite_fruit_pears (total students_oranges students_apples students_strawberries : ℕ) 
  (h1 : total = 450)
  (h2 : students_oranges = 70)
  (h3 : students_apples = 147)
  (h4 : students_strawberries = 113) :
  total - (students_oranges + students_apples + students_strawberries) = 120 := by
  sorry

end NUMINAMATH_CALUDE_favorite_fruit_pears_l3979_397960


namespace NUMINAMATH_CALUDE_candy_distribution_l3979_397967

/-- Given 200 candies distributed among A, B, and C, where A has more than twice as many candies as B,
    and B has more than three times as many candies as C, prove that the minimum number of candies A
    can have is 121, and the maximum number of candies C can have is 19. -/
theorem candy_distribution (a b c : ℕ) : 
  a + b + c = 200 →
  a > 2 * b →
  b > 3 * c →
  (∀ a' b' c' : ℕ, a' + b' + c' = 200 → a' > 2 * b' → b' > 3 * c' → a' ≥ a) →
  (∀ a' b' c' : ℕ, a' + b' + c' = 200 → a' > 2 * b' → b' > 3 * c' → c' ≤ c) →
  a = 121 ∧ c = 19 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l3979_397967


namespace NUMINAMATH_CALUDE_jerrys_action_figures_l3979_397936

theorem jerrys_action_figures (initial_figures : ℕ) : 
  (10 : ℕ) = initial_figures + 4 + 4 → initial_figures = 2 := by
sorry

end NUMINAMATH_CALUDE_jerrys_action_figures_l3979_397936


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3979_397956

theorem hyperbola_eccentricity (a b c : ℝ) (h1 : b^2 / a^2 = 1) (h2 : c^2 = a^2 + b^2) :
  c / a = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3979_397956


namespace NUMINAMATH_CALUDE_even_function_c_value_f_increasing_on_interval_l3979_397951

def f (x : ℝ) : ℝ := x^2 + 4*x + 3

def g (x c : ℝ) : ℝ := f x + c*x

theorem even_function_c_value :
  (∀ x, g x (-4) = g (-x) (-4)) ∧ 
  (∀ c, (∀ x, g x c = g (-x) c) → c = -4) :=
sorry

theorem f_increasing_on_interval :
  ∀ x₁ x₂, -2 ≤ x₁ → x₁ < x₂ → f x₁ < f x₂ :=
sorry

end NUMINAMATH_CALUDE_even_function_c_value_f_increasing_on_interval_l3979_397951


namespace NUMINAMATH_CALUDE_solution_equality_l3979_397935

theorem solution_equality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (Real.sqrt 2 * x + 1 / (Real.sqrt 2 * x) +
   Real.sqrt 2 * y + 1 / (Real.sqrt 2 * y) +
   Real.sqrt 2 * z + 1 / (Real.sqrt 2 * z) =
   6 - 2 * Real.sqrt (2 * x) * |y - z| -
   Real.sqrt (2 * y) * (x - z)^2 -
   Real.sqrt (2 * z) * Real.sqrt |x - y|) ↔
  (x = 1 / Real.sqrt 2 ∧ y = 1 / Real.sqrt 2 ∧ z = 1 / Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_solution_equality_l3979_397935
