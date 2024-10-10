import Mathlib

namespace no_solution_in_range_l1669_166977

theorem no_solution_in_range (x y : ℕ+) (h : 3 * x^2 + x = 4 * y^2 + y) :
  x - y ≠ 2013 ∧ x - y ≠ 2014 ∧ x - y ≠ 2015 ∧ x - y ≠ 2016 := by
  sorry

end no_solution_in_range_l1669_166977


namespace equilateral_triangle_x_value_l1669_166976

/-- An equilateral triangle with side lengths expressed in terms of x -/
structure EquilateralTriangle where
  x : ℝ
  side_length : ℝ
  eq_sides : side_length = 4 * x ∧ side_length = x + 12

theorem equilateral_triangle_x_value (t : EquilateralTriangle) : t.x = 4 := by
  sorry

end equilateral_triangle_x_value_l1669_166976


namespace dhoni_leftover_percentage_l1669_166968

/-- The percentage of earnings Dhoni had left over after spending on rent and a dishwasher -/
theorem dhoni_leftover_percentage : ℝ := by
  -- Define the percentage spent on rent
  let rent_percentage : ℝ := 20
  -- Define the percentage spent on dishwasher (5% less than rent)
  let dishwasher_percentage : ℝ := rent_percentage - 5
  -- Define the total percentage spent
  let total_spent_percentage : ℝ := rent_percentage + dishwasher_percentage
  -- Define the leftover percentage
  let leftover_percentage : ℝ := 100 - total_spent_percentage
  -- Prove that the leftover percentage is 65%
  have : leftover_percentage = 65 := by sorry
  -- Return the result
  exact leftover_percentage

end dhoni_leftover_percentage_l1669_166968


namespace shoe_donation_percentage_l1669_166946

theorem shoe_donation_percentage (initial_shoes : ℕ) (final_shoes : ℕ) (purchased_shoes : ℕ) : 
  initial_shoes = 80 → 
  final_shoes = 62 → 
  purchased_shoes = 6 → 
  (initial_shoes - (final_shoes - purchased_shoes)) / initial_shoes * 100 = 30 := by
  sorry

end shoe_donation_percentage_l1669_166946


namespace quadratic_polynomial_remainder_l1669_166914

theorem quadratic_polynomial_remainder (m n : ℚ) : 
  let P : ℚ → ℚ := λ x => x^2 + m*x + n
  (P m = m ∧ P n = n) → 
  ((m = 0 ∧ n = 0) ∨ (m = 1/2 ∧ n = 0) ∨ (m = 1 ∧ n = -1)) := by
  sorry

end quadratic_polynomial_remainder_l1669_166914


namespace percentage_problem_l1669_166931

theorem percentage_problem (N : ℝ) (P : ℝ) (h1 : N = 140) 
  (h2 : (P / 100) * N = (4 / 5) * N - 21) : P = 65 := by
  sorry

end percentage_problem_l1669_166931


namespace degrees_90_to_radians_l1669_166937

/-- Conversion of 90 degrees to radians -/
theorem degrees_90_to_radians : 
  (90 : ℝ) * (Real.pi / 180) = Real.pi / 2 := by sorry

end degrees_90_to_radians_l1669_166937


namespace coin_sum_impossibility_l1669_166928

def nickel : ℕ := 5
def dime : ℕ := 10
def quarter : ℕ := 25

def is_valid_sum (sum : ℕ) : Prop :=
  ∃ (n d q : ℕ), n + d + q = 6 ∧ n * nickel + d * dime + q * quarter = sum

theorem coin_sum_impossibility :
  is_valid_sum 40 ∧
  is_valid_sum 50 ∧
  is_valid_sum 60 ∧
  is_valid_sum 70 ∧
  ¬ is_valid_sum 30 :=
sorry

end coin_sum_impossibility_l1669_166928


namespace ellipse_sum_theorem_l1669_166978

/-- Represents an ellipse with center (h, k), semi-major axis a, and semi-minor axis b -/
structure Ellipse where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- The sum of h, k, a, and b for a specific ellipse -/
def ellipse_sum (e : Ellipse) : ℝ :=
  e.h + e.k + e.a + e.b

/-- Theorem: For an ellipse centered at (3, -5) with vertical semi-major axis 8 and semi-minor axis 4,
    the sum of h, k, a, and b equals 10 -/
theorem ellipse_sum_theorem (e : Ellipse)
    (h_center : e.h = 3 ∧ e.k = -5)
    (h_axes : e.a = 8 ∧ e.b = 4)
    (h_vertical : e.a > e.b) :
    ellipse_sum e = 10 := by
  sorry

end ellipse_sum_theorem_l1669_166978


namespace arccos_sqrt3_over_2_l1669_166945

theorem arccos_sqrt3_over_2 : Real.arccos (Real.sqrt 3 / 2) = π / 6 := by sorry

end arccos_sqrt3_over_2_l1669_166945


namespace ab_value_l1669_166989

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) : a * b = 10 := by
  sorry

end ab_value_l1669_166989


namespace door_height_problem_l1669_166966

theorem door_height_problem (pole_length width height diagonal : ℝ) : 
  pole_length > 0 ∧
  width > 0 ∧
  height > 0 ∧
  pole_length = width + 4 ∧
  pole_length = height + 2 ∧
  pole_length = diagonal ∧
  diagonal^2 = width^2 + height^2
  → height = 8 := by
  sorry

end door_height_problem_l1669_166966


namespace daphnes_collection_height_l1669_166988

/-- Represents the height of a book collection in inches and pages -/
structure BookCollection where
  inches : ℝ
  pages : ℝ
  pages_per_inch : ℝ

/-- The problem statement -/
theorem daphnes_collection_height 
  (miles : BookCollection)
  (daphne : BookCollection)
  (longest_collection_pages : ℝ)
  (h1 : miles.pages_per_inch = 5)
  (h2 : daphne.pages_per_inch = 50)
  (h3 : miles.inches = 240)
  (h4 : longest_collection_pages = 1250)
  (h5 : longest_collection_pages ≥ miles.pages)
  (h6 : longest_collection_pages ≥ daphne.pages)
  (h7 : daphne.pages = longest_collection_pages) :
  daphne.inches = 25 := by
sorry

end daphnes_collection_height_l1669_166988


namespace book_pages_count_l1669_166934

theorem book_pages_count :
  ∀ (P : ℕ),
  (P / 2 : ℕ) = P / 2 →  -- Half of the pages are filled with images
  (P - (P / 2 + 11)) / 2 = 19 →  -- Remaining pages after images and intro, half of which are text
  P = 98 :=
by sorry

end book_pages_count_l1669_166934


namespace quadratic_prime_square_solution_l1669_166986

/-- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. -/
def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

/-- The theorem states that the only integer solution to the equation 2x^2 - x - 36 = p^2,
    where p is a prime number, is x = 13. -/
theorem quadratic_prime_square_solution :
  ∀ x : ℤ, (∃ p : ℕ, is_prime p ∧ (2 * x^2 - x - 36 : ℤ) = p^2) ↔ x = 13 := by
  sorry

end quadratic_prime_square_solution_l1669_166986


namespace complement_intersection_theorem_l1669_166902

def U : Set ℝ := Set.univ
def A : Set ℝ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x | x ≤ -1 ∨ x > 2}

theorem complement_intersection_theorem :
  (Set.compl B ∩ A) = {0, 1, 2} := by sorry

end complement_intersection_theorem_l1669_166902


namespace cubic_divisibility_l1669_166960

theorem cubic_divisibility (t : ℤ) : (((125 * t - 12) ^ 3 + 2 * (125 * t - 12) + 2) % 125 = 0) := by
  sorry

end cubic_divisibility_l1669_166960


namespace smallest_common_pet_count_l1669_166908

theorem smallest_common_pet_count : ∃ n : ℕ, n > 0 ∧ n % 3 = 0 ∧ n % 15 = 0 ∧ ∀ m : ℕ, m > 0 → m % 3 = 0 → m % 15 = 0 → n ≤ m := by
  sorry

end smallest_common_pet_count_l1669_166908


namespace boat_distance_difference_l1669_166929

/-- The difference in distance traveled between two boats, one traveling downstream
    and one upstream, is 30 km. -/
theorem boat_distance_difference
  (a : ℝ)  -- Speed of both boats in still water (km/h)
  (h : a > 5)  -- Assumption that the boat speed is greater than the water flow speed
  : (3 * (a + 5)) - (3 * (a - 5)) = 30 :=
by sorry

end boat_distance_difference_l1669_166929


namespace point_coordinates_on_terminal_side_l1669_166949

/-- Given a point P on the terminal side of angle 4π/3 with |OP| = 4,
    prove that the coordinates of P are (-2, -2√3) -/
theorem point_coordinates_on_terminal_side (P : ℝ × ℝ) :
  (P.1 = 4 * Real.cos (4 * Real.pi / 3) ∧ P.2 = 4 * Real.sin (4 * Real.pi / 3)) →
  P = (-2, -2 * Real.sqrt 3) := by
  sorry

end point_coordinates_on_terminal_side_l1669_166949


namespace binomial_expectation_and_variance_l1669_166922

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The expected value of a binomial distribution -/
def expectedValue (b : BinomialDistribution) : ℝ := b.n * b.p

/-- The variance of a binomial distribution -/
def variance (b : BinomialDistribution) : ℝ := b.n * b.p * (1 - b.p)

/-- Theorem: For a binomial distribution with n=200 and p=0.01, 
    the expected value is 2 and the variance is 1.98 -/
theorem binomial_expectation_and_variance :
  ∃ b : BinomialDistribution, 
    b.n = 200 ∧ 
    b.p = 0.01 ∧ 
    expectedValue b = 2 ∧ 
    variance b = 1.98 := by
  sorry


end binomial_expectation_and_variance_l1669_166922


namespace sum_edges_pyramid_prism_l1669_166912

/-- A triangular pyramid (tetrahedron) -/
structure TriangularPyramid where
  edges : ℕ
  edge_count : edges = 6

/-- A triangular prism -/
structure TriangularPrism where
  edges : ℕ
  edge_count : edges = 9

/-- The sum of edges of a triangular pyramid and a triangular prism is 15 -/
theorem sum_edges_pyramid_prism (p : TriangularPyramid) (q : TriangularPrism) :
  p.edges + q.edges = 15 := by
  sorry

end sum_edges_pyramid_prism_l1669_166912


namespace quadratic_equation_from_means_l1669_166905

theorem quadratic_equation_from_means (a b : ℝ) : 
  (a + b) / 2 = 8 → 
  Real.sqrt (a * b) = 12 → 
  ∀ x, x^2 - 16*x + 144 = 0 ↔ (x = a ∨ x = b) :=
by sorry

end quadratic_equation_from_means_l1669_166905


namespace no_valid_operation_l1669_166903

-- Define the set of basic arithmetic operations
inductive BasicOperation
  | Add
  | Subtract
  | Multiply
  | Divide

-- Define a function to apply a basic operation
def applyOperation (op : BasicOperation) (a b : ℤ) : ℤ :=
  match op with
  | BasicOperation.Add => a + b
  | BasicOperation.Subtract => a - b
  | BasicOperation.Multiply => a * b
  | BasicOperation.Divide => a / b

-- Theorem statement
theorem no_valid_operation :
  ¬ ∃ (op : BasicOperation), (applyOperation op 8 2) + 5 - (3 - 2) = 12 :=
by sorry

end no_valid_operation_l1669_166903


namespace summer_course_duration_l1669_166927

/-- The number of days required for a summer course with the given conditions. -/
def summer_course_days (n k : ℕ) : ℕ :=
  (n.choose 2) / (k.choose 2)

/-- Theorem stating the number of days for the summer course. -/
theorem summer_course_duration :
  summer_course_days 15 3 = 35 := by
  sorry

end summer_course_duration_l1669_166927


namespace inequality_solution_set_l1669_166916

theorem inequality_solution_set (x : ℝ) : 
  (((2 * x - 3) / (x + 4) > (4 * x + 1) / (3 * x - 2)) ↔ 
  (x > -4 ∧ x < (17 - Real.sqrt 201) / 4) ∨ 
  (x > (17 + Real.sqrt 201) / 4 ∧ x < 2 / 3)) := by
  sorry

end inequality_solution_set_l1669_166916


namespace adam_total_score_l1669_166904

/-- Calculates the total points scored in a game given points per round and number of rounds -/
def totalPoints (pointsPerRound : ℕ) (numRounds : ℕ) : ℕ :=
  pointsPerRound * numRounds

/-- Theorem stating that given 71 points per round and 4 rounds, the total points is 284 -/
theorem adam_total_score : totalPoints 71 4 = 284 := by
  sorry

end adam_total_score_l1669_166904


namespace penny_throwing_ratio_l1669_166955

/-- Given the conditions of the penny-throwing problem, prove that the ratio of Rocky's pennies to Gretchen's is 1:3 -/
theorem penny_throwing_ratio (rachelle gretchen rocky : ℕ) : 
  rachelle = 180 →
  gretchen = rachelle / 2 →
  rachelle + gretchen + rocky = 300 →
  rocky / gretchen = 1 / 3 := by
sorry

end penny_throwing_ratio_l1669_166955


namespace lewis_speed_l1669_166990

/-- Proves that Lewis's speed is 80 mph given the problem conditions -/
theorem lewis_speed (john_speed : ℝ) (total_distance : ℝ) (meeting_distance : ℝ) :
  john_speed = 40 ∧ 
  total_distance = 240 ∧ 
  meeting_distance = 160 →
  (total_distance + (total_distance - meeting_distance)) / (meeting_distance / john_speed) = 80 :=
by
  sorry

end lewis_speed_l1669_166990


namespace infinite_fraction_value_l1669_166975

theorem infinite_fraction_value : 
  ∃ x : ℝ, x = 3 + 3 / (1 + 5 / x) ∧ x = (1 + Real.sqrt 61) / 2 := by
  sorry

end infinite_fraction_value_l1669_166975


namespace line_slope_hyperbola_intersection_l1669_166993

/-- A line intersecting a hyperbola x^2 - y^2 = 1 at two points has a slope of 2 
    if the midpoint of the line segment between these points is (2,1) -/
theorem line_slope_hyperbola_intersection (A B : ℝ × ℝ) : 
  (A.1^2 - A.2^2 = 1) →  -- A is on the hyperbola
  (B.1^2 - B.2^2 = 1) →  -- B is on the hyperbola
  ((A.1 + B.1) / 2 = 2 ∧ (A.2 + B.2) / 2 = 1) →  -- Midpoint is (2,1)
  (B.2 - A.2) / (B.1 - A.1) = 2 :=  -- Slope is 2
by sorry

end line_slope_hyperbola_intersection_l1669_166993


namespace percentage_difference_l1669_166947

theorem percentage_difference (x y z : ℝ) (h1 : y = 1.2 * z) (h2 : z = 150) (h3 : x + y + z = 555) :
  (x - y) / y * 100 = 25 := by
  sorry

end percentage_difference_l1669_166947


namespace three_digit_cube_divisible_by_16_l1669_166982

theorem three_digit_cube_divisible_by_16 : 
  ∃! n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ ∃ m : ℕ, n = m^3 ∧ 16 ∣ n :=
by sorry

end three_digit_cube_divisible_by_16_l1669_166982


namespace solution_characterization_l1669_166950

def system_equations (n : ℕ) (x : ℕ → ℝ) : Prop :=
  (∀ i ∈ Finset.range n, 1 - x i * x ((i + 1) % n) = 0)

theorem solution_characterization (n : ℕ) (x : ℕ → ℝ) (hn : n > 0) :
  system_equations n x →
  (n % 2 = 1 ∧ (∀ i ∈ Finset.range n, x i = 1 ∨ x i = -1)) ∨
  (n % 2 = 0 ∧ ∃ a : ℝ, a ≠ 0 ∧
    x 0 = a ∧ x 1 = 1 / a ∧
    ∀ i ∈ Finset.range (n - 2), x (i + 2) = x i) :=
by sorry

end solution_characterization_l1669_166950


namespace slope_at_negative_five_l1669_166974

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem slope_at_negative_five
  (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_diff : Differentiable ℝ f)
  (h_der_one : deriv f 1 = 1)
  (h_period : has_period f 4) :
  deriv f (-5) = -1 := by
  sorry

end slope_at_negative_five_l1669_166974


namespace bryans_bookshelves_l1669_166979

theorem bryans_bookshelves (total_books : ℕ) (books_per_shelf : ℕ) (h1 : total_books = 38) (h2 : books_per_shelf = 2) :
  total_books / books_per_shelf = 19 := by
  sorry

end bryans_bookshelves_l1669_166979


namespace power_of_two_100_l1669_166956

theorem power_of_two_100 :
  (10^30 : ℕ) ≤ 2^100 ∧ 2^100 < (10^31 : ℕ) ∧ 2^100 % 1000 = 376 := by
  sorry

end power_of_two_100_l1669_166956


namespace alice_prob_is_nine_twentyfifths_l1669_166994

/-- Represents the person holding the ball -/
inductive Person : Type
| Alice : Person
| Bob : Person

/-- The probability of tossing the ball to the other person -/
def toss_prob (p : Person) : ℚ :=
  match p with
  | Person.Alice => 3/5
  | Person.Bob => 1/3

/-- The probability of keeping the ball -/
def keep_prob (p : Person) : ℚ :=
  1 - toss_prob p

/-- The probability that Alice has the ball after two turns, given she starts with it -/
def prob_alice_after_two_turns : ℚ :=
  toss_prob Person.Alice * toss_prob Person.Bob +
  keep_prob Person.Alice * keep_prob Person.Alice

theorem alice_prob_is_nine_twentyfifths :
  prob_alice_after_two_turns = 9/25 := by
  sorry

end alice_prob_is_nine_twentyfifths_l1669_166994


namespace final_quantity_of_B_l1669_166910

/-- Represents the quantity of each product type -/
structure ProductQuantities where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Calculates the total cost of the products -/
def totalCost (q : ProductQuantities) : ℕ :=
  2 * q.a + 3 * q.b + 5 * q.c

/-- Represents the problem constraints -/
structure ProblemConstraints where
  initial : ProductQuantities
  final : ProductQuantities
  initialCost : totalCost initial = 20
  finalCost : totalCost final = 20
  returnedTwoItems : initial.a + initial.b + initial.c = final.a + final.b + final.c + 2
  atLeastOne : final.a ≥ 1 ∧ final.b ≥ 1 ∧ final.c ≥ 1

theorem final_quantity_of_B (constraints : ProblemConstraints) : constraints.final.b = 1 := by
  sorry


end final_quantity_of_B_l1669_166910


namespace triangle_properties_l1669_166957

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : Real.cos t.A * Real.cos t.C + Real.sin t.A * Real.sin t.C + Real.cos t.B = 3/2)
  (h2 : t.b^2 = t.a * t.c)  -- Geometric progression condition
  (h3 : t.a / Real.tan t.A + t.c / Real.tan t.C = 2 * t.b / Real.tan t.B) :
  t.B = π/3 ∧ t.A = π/3 ∧ t.C = π/3 := by
  sorry

end triangle_properties_l1669_166957


namespace fraction_to_decimal_l1669_166915

theorem fraction_to_decimal (n d : ℕ) (h : d ≠ 0) :
  (n : ℚ) / d = 0.650 :=
by
  sorry

#check fraction_to_decimal 13 320

end fraction_to_decimal_l1669_166915


namespace ratio_a_to_b_l1669_166930

/-- An arithmetic sequence with specific terms -/
structure ArithmeticSequence where
  a : ℝ
  b : ℝ
  y : ℝ
  first_term : ℝ := 2 * a
  second_term : ℝ := y
  third_term : ℝ := 3 * b
  fourth_term : ℝ := 4 * y
  is_arithmetic : ∃ (d : ℝ), second_term - first_term = d ∧ 
                              third_term - second_term = d ∧ 
                              fourth_term - third_term = d

/-- The ratio of a to b in the arithmetic sequence is -1/5 -/
theorem ratio_a_to_b (seq : ArithmeticSequence) : seq.a / seq.b = -1 / 5 := by
  sorry

end ratio_a_to_b_l1669_166930


namespace password_identification_l1669_166940

def is_valid_password (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ n % 9 = 0 ∧ n / 1000 = 5

def alice_knows (n : ℕ) : Prop :=
  ∃ a b : ℕ, n / 100 % 10 = a ∧ n / 10 % 10 = b

def bob_knows (n : ℕ) : Prop :=
  ∃ b c : ℕ, n / 10 % 10 = b ∧ n % 10 = c

def initially_unknown (n : ℕ) : Prop :=
  ∃ m : ℕ, m ≠ n ∧ is_valid_password m ∧ alice_knows m ∧ bob_knows m

theorem password_identification :
  ∃ n : ℕ,
    is_valid_password n ∧
    alice_knows n ∧
    bob_knows n ∧
    initially_unknown n ∧
    (∀ m : ℕ, is_valid_password m ∧ alice_knows m ∧ bob_knows m ∧ initially_unknown m → m ≤ n) ∧
    n = 5940 :=
  sorry

end password_identification_l1669_166940


namespace perfect_game_score_l1669_166941

/-- Given that a perfect score is 21 points, prove that the total points after 3 perfect games is 63. -/
theorem perfect_game_score (perfect_score : ℕ) (h : perfect_score = 21) :
  3 * perfect_score = 63 := by
  sorry

end perfect_game_score_l1669_166941


namespace total_amount_raised_l1669_166909

/-- Represents the sizes of rubber ducks --/
inductive DuckSize
  | Small
  | Medium
  | Large

/-- Calculates the price of a duck given its size --/
def price (s : DuckSize) : ℚ :=
  match s with
  | DuckSize.Small => 2
  | DuckSize.Medium => 3
  | DuckSize.Large => 5

/-- Calculates the bulk discount rate for a given size and quantity --/
def bulkDiscountRate (s : DuckSize) (quantity : ℕ) : ℚ :=
  match s with
  | DuckSize.Small => if quantity ≥ 10 then 0.1 else 0
  | DuckSize.Medium => if quantity ≥ 15 then 0.15 else 0
  | DuckSize.Large => if quantity ≥ 20 then 0.2 else 0

/-- Returns the sales tax rate for a given duck size --/
def salesTaxRate (s : DuckSize) : ℚ :=
  match s with
  | DuckSize.Small => 0.05
  | DuckSize.Medium => 0.07
  | DuckSize.Large => 0.09

/-- Calculates the total amount raised for a given duck size and quantity --/
def amountRaised (s : DuckSize) (quantity : ℕ) : ℚ :=
  let basePrice := price s * quantity
  let discountedPrice := basePrice * (1 - bulkDiscountRate s quantity)
  discountedPrice * (1 + salesTaxRate s)

/-- Theorem stating the total amount raised for charity --/
theorem total_amount_raised :
  amountRaised DuckSize.Small 150 +
  amountRaised DuckSize.Medium 221 +
  amountRaised DuckSize.Large 185 = 1693.1 := by
  sorry


end total_amount_raised_l1669_166909


namespace geometric_sequence_a3_l1669_166924

/-- Given a geometric sequence {a_n} with common ratio q = 3,
    if S_3 + S_4 = 53/3, then a_3 = 3 -/
theorem geometric_sequence_a3 (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) = 3 * a n) →  -- common ratio q = 3
  (∀ n, S n = (a 1) * (3^n - 1) / 2) →  -- sum formula for geometric sequence
  S 3 + S 4 = 53 / 3 →  -- given condition
  a 3 = 3 := by
sorry


end geometric_sequence_a3_l1669_166924


namespace factor_expression_l1669_166921

theorem factor_expression (y z : ℝ) : 64 - 16 * y^2 * z^2 = 16 * (2 - y*z) * (2 + y*z) := by
  sorry

end factor_expression_l1669_166921


namespace max_lateral_area_triangular_prism_l1669_166973

/-- The maximum lateral area of a triangular prism inscribed in a sphere -/
theorem max_lateral_area_triangular_prism (r : ℝ) (h : r = 2) :
  ∃ (a h : ℝ),
    -- Condition: prism inscribed in sphere
    4 * a^2 + 3 * h^2 = 48 ∧
    -- Condition: lateral area
    (3 : ℝ) * a * h ≤ 12 * Real.sqrt 3 ∧
    -- Condition: maximum value
    ∀ (a' h' : ℝ), 4 * a'^2 + 3 * h'^2 = 48 → (3 : ℝ) * a' * h' ≤ 12 * Real.sqrt 3 :=
by
  sorry


end max_lateral_area_triangular_prism_l1669_166973


namespace log_xy_value_l1669_166997

open Real

theorem log_xy_value (x y : ℝ) (h1 : log (x * y^4) = 1) (h2 : log (x^3 * y) = 1) : 
  log (x * y) = 5 / 11 := by
  sorry

end log_xy_value_l1669_166997


namespace complex_absolute_value_l1669_166953

theorem complex_absolute_value (ω : ℂ) : ω = 7 + 3*I → Complex.abs (ω^2 + 8*ω + 98) = Real.sqrt 41605 := by
  sorry

end complex_absolute_value_l1669_166953


namespace least_addition_for_divisibility_l1669_166920

theorem least_addition_for_divisibility (n : ℕ) : 
  (∀ m : ℕ, m < 4 → ¬(5 ∣ (2496 + m))) ∧ (5 ∣ (2496 + 4)) → n = 4 := by
  sorry

end least_addition_for_divisibility_l1669_166920


namespace henry_bought_two_fireworks_l1669_166999

/-- The number of fireworks Henry bought -/
def henrys_fireworks (total : ℕ) (last_year : ℕ) (friends : ℕ) : ℕ :=
  total - last_year - friends

/-- Proof that Henry bought 2 fireworks -/
theorem henry_bought_two_fireworks :
  henrys_fireworks 11 6 3 = 2 := by
  sorry

end henry_bought_two_fireworks_l1669_166999


namespace equilateral_triangle_intersection_l1669_166971

theorem equilateral_triangle_intersection (a : ℝ) : 
  (∃ (A B : ℝ × ℝ), 
    (A.1 + A.2 = Real.sqrt 3 * a ∧ A.1^2 + A.2^2 = a^2 + (a-1)^2) ∧
    (B.1 + B.2 = Real.sqrt 3 * a ∧ B.1^2 + B.2^2 = a^2 + (a-1)^2) ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = A.1^2 + A.2^2 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = B.1^2 + B.2^2) →
  a = 1/2 := by
sorry

end equilateral_triangle_intersection_l1669_166971


namespace survey_result_l1669_166981

theorem survey_result (total : ℕ) (migraines insomnia anxiety : ℕ)
  (migraines_insomnia migraines_anxiety insomnia_anxiety : ℕ)
  (all_three : ℕ) :
  total = 150 →
  migraines = 90 →
  insomnia = 60 →
  anxiety = 30 →
  migraines_insomnia = 20 →
  migraines_anxiety = 10 →
  insomnia_anxiety = 15 →
  all_three = 5 →
  total - (migraines + insomnia + anxiety - migraines_insomnia - migraines_anxiety - insomnia_anxiety + all_three) = 40 := by
  sorry

#check survey_result

end survey_result_l1669_166981


namespace point_on_line_l1669_166952

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem point_on_line (p1 p2 p3 p4 : Point) :
  p1 = Point.mk 2 5 →
  p2 = Point.mk 4 11 →
  p3 = Point.mk 6 17 →
  p4 = Point.mk 15 44 →
  collinear p1 p2 p3 →
  collinear p1 p2 p4 :=
by
  sorry


end point_on_line_l1669_166952


namespace cookie_jar_problem_l1669_166984

theorem cookie_jar_problem (C : ℕ) : (C - 1 = (C + 5) / 2) → C = 7 := by
  sorry

end cookie_jar_problem_l1669_166984


namespace fraction_less_than_one_l1669_166959

theorem fraction_less_than_one (a b : ℝ) (h1 : a > b) (h2 : b > 0) : b / a < 1 := by
  sorry

end fraction_less_than_one_l1669_166959


namespace largest_positive_root_l1669_166948

theorem largest_positive_root (a₀ a₁ a₂ a₃ : ℝ) 
  (h₀ : |a₀| ≤ 3) (h₁ : |a₁| ≤ 3) (h₂ : |a₂| ≤ 3) (h₃ : |a₃| ≤ 3) :
  ∃ (r : ℝ), r = 3 ∧ 
  (∀ (x : ℝ), x > r → ∀ (b₀ b₁ b₂ b₃ : ℝ), 
    |b₀| ≤ 3 → |b₁| ≤ 3 → |b₂| ≤ 3 → |b₃| ≤ 3 → 
    x^4 + b₃*x^3 + b₂*x^2 + b₁*x + b₀ ≠ 0) ∧
  (∃ (c₀ c₁ c₂ c₃ : ℝ), 
    |c₀| ≤ 3 ∧ |c₁| ≤ 3 ∧ |c₂| ≤ 3 ∧ |c₃| ≤ 3 ∧ 
    r^4 + c₃*r^3 + c₂*r^2 + c₁*r + c₀ = 0) :=
by sorry

end largest_positive_root_l1669_166948


namespace triangle_property_l1669_166900

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if acosB = (1/2)b + c, then A = 2π/3 and (b^2 + c^2 + bc) / (4R^2) = 3/4,
    where R is the radius of the circumcircle of triangle ABC -/
theorem triangle_property (a b c : ℝ) (A B C : ℝ) (R : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  a * Real.cos B = (1/2) * b + c →
  R > 0 →
  A = 2 * π / 3 ∧ (b^2 + c^2 + b*c) / (4 * R^2) = 3/4 := by sorry

end triangle_property_l1669_166900


namespace ab_and_c_values_l1669_166992

theorem ab_and_c_values (a b c : ℤ) 
  (h1 : a - b = 3) 
  (h2 : a^2 + b^2 = 29) 
  (h3 : a + b + c = 10) : 
  a * b = 10 ∧ (c = 3 ∨ c = 17) := by
sorry

end ab_and_c_values_l1669_166992


namespace amy_muffin_problem_l1669_166923

/-- Represents the number of muffins Amy brings to school each day -/
def muffins_sequence (first_day : ℕ) : ℕ → ℕ
| 0 => first_day
| n + 1 => muffins_sequence first_day n + 1

/-- Calculates the total number of muffins brought to school over 5 days -/
def total_muffins_brought (first_day : ℕ) : ℕ :=
  (List.range 5).map (muffins_sequence first_day) |>.sum

/-- Theorem stating the solution to Amy's muffin problem -/
theorem amy_muffin_problem :
  ∃ (first_day : ℕ),
    total_muffins_brought first_day = 22 - 7 ∧
    first_day = 1 := by
  sorry

end amy_muffin_problem_l1669_166923


namespace earphone_cost_l1669_166932

def mean_expenditure : ℝ := 500
def mon_expenditure : ℝ := 450
def tue_expenditure : ℝ := 600
def wed_expenditure : ℝ := 400
def thu_expenditure : ℝ := 500
def sat_expenditure : ℝ := 550
def sun_expenditure : ℝ := 300
def pen_cost : ℝ := 30
def notebook_cost : ℝ := 50
def num_days : ℕ := 7

theorem earphone_cost :
  let total_expenditure := mean_expenditure * num_days
  let known_expenditures := mon_expenditure + tue_expenditure + wed_expenditure + 
                            thu_expenditure + sat_expenditure + sun_expenditure
  let friday_expenditure := total_expenditure - known_expenditures
  let other_items_cost := pen_cost + notebook_cost
  friday_expenditure - other_items_cost = 620 := by
sorry

end earphone_cost_l1669_166932


namespace implication_equivalences_l1669_166935

variable (p q : Prop)

theorem implication_equivalences (h : p → q) :
  (∃ (f : p → q), True) ∧
  (p → q) ∧
  (¬q → ¬p) ∧
  ((p → q) ∧ (¬p ∨ q)) :=
by sorry

end implication_equivalences_l1669_166935


namespace even_blue_faces_count_l1669_166998

/-- Represents a cube with a certain number of blue faces -/
structure PaintedCube where
  blueFaces : Nat

/-- Represents the wooden block -/
structure WoodenBlock where
  length : Nat
  width : Nat
  height : Nat
  paintedSides : Nat

/-- Function to generate the list of cubes from a wooden block -/
def generateCubes (block : WoodenBlock) : List PaintedCube :=
  sorry

/-- Function to count cubes with even number of blue faces -/
def countEvenBlueFaces (cubes : List PaintedCube) : Nat :=
  sorry

/-- Main theorem -/
theorem even_blue_faces_count (block : WoodenBlock) 
    (h1 : block.length = 5)
    (h2 : block.width = 3)
    (h3 : block.height = 1)
    (h4 : block.paintedSides = 5) :
  countEvenBlueFaces (generateCubes block) = 5 := by
  sorry

end even_blue_faces_count_l1669_166998


namespace solve_equation_l1669_166995

theorem solve_equation (x : ℝ) (h : 3 * x + 15 = (1 / 3) * (6 * x + 45)) : x = 0 := by
  sorry

end solve_equation_l1669_166995


namespace absolute_difference_of_U_coordinates_l1669_166917

/-- Triangle PQR with vertices P(0,10), Q(5,0), and R(10,0) -/
def P : ℝ × ℝ := (0, 10)
def Q : ℝ × ℝ := (5, 0)
def R : ℝ × ℝ := (10, 0)

/-- V is on QR and 3 units away from Q -/
def V : ℝ × ℝ := (2, 0)

/-- U is on PR and has the same x-coordinate as V -/
def U : ℝ × ℝ := (2, 8)

/-- The theorem to be proved -/
theorem absolute_difference_of_U_coordinates : 
  |U.2 - U.1| = 6 := by sorry

end absolute_difference_of_U_coordinates_l1669_166917


namespace complex_number_range_angle_between_vectors_l1669_166958

-- Problem 1
theorem complex_number_range (Z : ℂ) (a : ℝ) 
  (h1 : (Z + 2*I).im = 0)
  (h2 : ((Z / (2 - I)).im = 0))
  (h3 : ((Z + a*I)^2).re > 0)
  (h4 : ((Z + a*I)^2).im > 0) :
  2 < a ∧ a < 6 := by sorry

-- Problem 2
theorem angle_between_vectors (z₁ z₂ : ℂ) 
  (h1 : z₁ = 3)
  (h2 : z₂ = -5 + 5*I) :
  Real.arccos ((z₁.re * z₂.re + z₁.im * z₂.im) / (Complex.abs z₁ * Complex.abs z₂)) = 3 * Real.pi / 4 := by sorry

end complex_number_range_angle_between_vectors_l1669_166958


namespace computer_upgrade_cost_l1669_166987

/-- Calculates the total amount spent on a computer after upgrading the video card -/
def totalSpent (initialCost salePrice newCardCost : ℕ) : ℕ :=
  initialCost + newCardCost - salePrice

/-- Theorem stating the total amount spent on the computer -/
theorem computer_upgrade_cost :
  ∀ (initialCost salePrice newCardCost : ℕ),
    initialCost = 1200 →
    salePrice = 300 →
    newCardCost = 500 →
    totalSpent initialCost salePrice newCardCost = 1400 :=
by
  sorry

end computer_upgrade_cost_l1669_166987


namespace complex_magnitude_equation_l1669_166961

theorem complex_magnitude_equation (x : ℝ) :
  x > 0 → Complex.abs (3 + x * Complex.I) = 7 → x = 2 * Real.sqrt 10 := by
  sorry

end complex_magnitude_equation_l1669_166961


namespace partner_contribution_correct_l1669_166919

/-- Calculates the partner's contribution given the investment details and profit ratio -/
def calculate_partner_contribution (a_investment : ℚ) (a_months : ℚ) (b_months : ℚ) (a_ratio : ℚ) (b_ratio : ℚ) : ℚ :=
  (a_investment * a_months * b_ratio) / (a_ratio * b_months)

theorem partner_contribution_correct :
  let a_investment : ℚ := 3500
  let a_months : ℚ := 12
  let b_months : ℚ := 3
  let a_ratio : ℚ := 2
  let b_ratio : ℚ := 3
  calculate_partner_contribution a_investment a_months b_months a_ratio b_ratio = 21000 := by
  sorry

end partner_contribution_correct_l1669_166919


namespace sum_and_double_l1669_166926

theorem sum_and_double : (142 + 29 + 26 + 14) * 2 = 422 := by
  sorry

end sum_and_double_l1669_166926


namespace inequalities_and_minimum_l1669_166962

theorem inequalities_and_minimum (a b : ℝ) :
  (a > b ∧ b > 0 → a - 1/a > b - 1/b) ∧
  (a > 0 ∧ b > 0 ∧ 2*a + b = 1 → 2/a + 1/b ≥ 9) :=
by sorry

end inequalities_and_minimum_l1669_166962


namespace no_solution_for_all_a_b_l1669_166938

theorem no_solution_for_all_a_b : ∃ (a b : ℤ), a ≠ 0 ∧ b ≠ 0 ∧
  ¬∃ (x y : ℝ), (Real.tan (13 * x) * Real.tan (a * y) = 1) ∧
                (Real.tan (21 * x) * Real.tan (b * y) = 1) :=
by sorry

end no_solution_for_all_a_b_l1669_166938


namespace trigonometric_identity_l1669_166907

theorem trigonometric_identity (θ : Real) 
  (h : Real.sin (π / 3 - θ) = 1 / 2) : 
  Real.cos (π / 6 + θ) = 1 / 2 := by
  sorry

end trigonometric_identity_l1669_166907


namespace total_bugs_eaten_l1669_166951

def gecko_bugs : ℕ := 12

def lizard_bugs : ℕ := gecko_bugs / 2

def frog_bugs : ℕ := 3 * lizard_bugs

def toad_bugs : ℕ := frog_bugs + frog_bugs / 2

theorem total_bugs_eaten :
  gecko_bugs + lizard_bugs + frog_bugs + toad_bugs = 63 := by
  sorry

end total_bugs_eaten_l1669_166951


namespace expected_votes_for_candidate_a_l1669_166901

-- Define the percentage of Democrat and Republican voters
def democrat_percentage : ℝ := 0.60
def republican_percentage : ℝ := 1 - democrat_percentage

-- Define the percentage of Democrats and Republicans voting for candidate A
def democrat_vote_for_a : ℝ := 0.85
def republican_vote_for_a : ℝ := 0.20

-- Define the theorem
theorem expected_votes_for_candidate_a :
  democrat_percentage * democrat_vote_for_a + republican_percentage * republican_vote_for_a = 0.59 := by
  sorry

end expected_votes_for_candidate_a_l1669_166901


namespace tan_alpha_plus_pi_fourth_l1669_166991

theorem tan_alpha_plus_pi_fourth (α : Real) 
  (h1 : α > 0) (h2 : α < Real.pi / 2) 
  (h3 : Real.cos (2 * α) + Real.cos α ^ 2 = 0) : 
  Real.tan (α + Real.pi / 4) = -3 - 2 * Real.sqrt 2 := by
  sorry

end tan_alpha_plus_pi_fourth_l1669_166991


namespace multiply_by_97_preserves_form_l1669_166983

theorem multiply_by_97_preserves_form (a b : ℕ) :
  ∃ (a' b' : ℕ), 97 * (3 * a^2 + 32 * b^2) = 3 * a'^2 + 32 * b'^2 := by
  sorry

end multiply_by_97_preserves_form_l1669_166983


namespace maggie_bouncy_balls_l1669_166969

/-- The number of packs of red bouncy balls -/
def red_packs : ℕ := 4

/-- The number of balls in each pack of red bouncy balls -/
def red_per_pack : ℕ := 12

/-- The number of packs of yellow bouncy balls -/
def yellow_packs : ℕ := 8

/-- The number of balls in each pack of yellow bouncy balls -/
def yellow_per_pack : ℕ := 10

/-- The number of packs of green bouncy balls -/
def green_packs : ℕ := 4

/-- The number of balls in each pack of green bouncy balls -/
def green_per_pack : ℕ := 14

/-- The number of packs of blue bouncy balls -/
def blue_packs : ℕ := 6

/-- The number of balls in each pack of blue bouncy balls -/
def blue_per_pack : ℕ := 8

/-- The total number of bouncy balls Maggie bought -/
def total_balls : ℕ := red_packs * red_per_pack + yellow_packs * yellow_per_pack + 
                        green_packs * green_per_pack + blue_packs * blue_per_pack

theorem maggie_bouncy_balls : total_balls = 232 := by
  sorry

end maggie_bouncy_balls_l1669_166969


namespace system_of_equations_solution_l1669_166939

theorem system_of_equations_solution (x y m : ℝ) : 
  (3 * x + 5 * y = m + 2) → 
  (2 * x + 3 * y = m) → 
  (x + y = -10) → 
  (m^2 - 2*m + 1 = 81) := by
sorry

end system_of_equations_solution_l1669_166939


namespace coefficient_sum_l1669_166985

theorem coefficient_sum (x : ℝ) (a₀ a₁ a₂ a₃ : ℝ) 
  (h : (1 - 2/x)^3 = a₀ + a₁*(1/x) + a₂*(1/x)^2 + a₃*(1/x)^3) :
  a₁ + a₂ = 6 := by
  sorry

end coefficient_sum_l1669_166985


namespace glycol_concentration_mixture_l1669_166936

/-- Proves that mixing 16 gallons of 40% glycol concentration with 8 gallons of 10% glycol concentration 
    results in a 30% glycol concentration in the final 24-gallon mixture. -/
theorem glycol_concentration_mixture 
  (total_volume : ℝ) 
  (volume_40_percent : ℝ)
  (volume_10_percent : ℝ)
  (concentration_40_percent : ℝ)
  (concentration_10_percent : ℝ)
  (h1 : total_volume = 24)
  (h2 : volume_40_percent = 16)
  (h3 : volume_10_percent = 8)
  (h4 : concentration_40_percent = 0.4)
  (h5 : concentration_10_percent = 0.1)
  (h6 : volume_40_percent + volume_10_percent = total_volume) :
  (volume_40_percent * concentration_40_percent + volume_10_percent * concentration_10_percent) / total_volume = 0.3 := by
  sorry


end glycol_concentration_mixture_l1669_166936


namespace perimeter_calculation_l1669_166996

theorem perimeter_calculation : 
  let segments : List ℕ := [2, 3, 2, 6, 2, 4, 3]
  segments.sum = 22 := by sorry

end perimeter_calculation_l1669_166996


namespace sweet_potatoes_sold_l1669_166954

theorem sweet_potatoes_sold (total harvested : ℕ) (sold_to_lenon : ℕ) (unsold : ℕ) 
  (h1 : total = 80)
  (h2 : sold_to_lenon = 15)
  (h3 : unsold = 45) :
  total - sold_to_lenon - unsold = 20 :=
by sorry

end sweet_potatoes_sold_l1669_166954


namespace sum_of_digits_of_even_numbers_up_to_12000_l1669_166911

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Check if a natural number is even -/
def isEven (n : ℕ) : Prop := sorry

/-- Sum of digits of all even numbers in a sequence from 1 to n -/
def sumOfDigitsOfEvenNumbers (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of digits of all even numbers from 1 to 12000 is 129348 -/
theorem sum_of_digits_of_even_numbers_up_to_12000 :
  sumOfDigitsOfEvenNumbers 12000 = 129348 := by sorry

end sum_of_digits_of_even_numbers_up_to_12000_l1669_166911


namespace solution_set_trig_equation_l1669_166925

theorem solution_set_trig_equation :
  {x : ℝ | 3 * Real.sin x = 1 + Real.cos (2 * x)} =
  {x : ℝ | ∃ k : ℤ, x = k * Real.pi + (-1)^k * Real.pi / 6} := by
  sorry

end solution_set_trig_equation_l1669_166925


namespace lemonade_pitcher_capacity_l1669_166933

theorem lemonade_pitcher_capacity (total_glasses : ℕ) (total_pitchers : ℕ) 
  (h1 : total_glasses = 30) (h2 : total_pitchers = 6) :
  total_glasses / total_pitchers = 5 := by
  sorry

end lemonade_pitcher_capacity_l1669_166933


namespace min_value_of_f_l1669_166972

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + a

-- Define the interval
def I : Set ℝ := Set.Icc (-1) 2

-- State the theorem
theorem min_value_of_f (a : ℝ) :
  (∃ x ∈ I, ∀ y ∈ I, f a y ≤ f a x) ∧ (f a 2 = 20) →
  (∃ x ∈ I, ∀ y ∈ I, f a x ≤ f a y) ∧ (f a (-1) = -7) :=
by sorry

end min_value_of_f_l1669_166972


namespace quadratic_equation_roots_l1669_166964

theorem quadratic_equation_roots : ∃ (p q : ℤ),
  (∃ (x₁ x₂ : ℤ), 
    x₁ > 0 ∧ x₂ > 0 ∧
    x₁^2 + p*x₁ + q = 0 ∧
    x₂^2 + p*x₂ + q = 0 ∧
    p + q = 28) →
  (∃ (x₁ x₂ : ℤ), 
    (x₁ = 30 ∧ x₂ = 2) ∨ (x₁ = 2 ∧ x₂ = 30)) :=
by sorry

end quadratic_equation_roots_l1669_166964


namespace geometric_series_sum_l1669_166943

/-- Sum of a geometric series -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The first term of the geometric series -/
def a : ℚ := 2

/-- The common ratio of the geometric series -/
def r : ℚ := -2

/-- The number of terms in the geometric series -/
def n : ℕ := 10

theorem geometric_series_sum :
  geometric_sum a r n = 2050 / 3 := by sorry

end geometric_series_sum_l1669_166943


namespace mod_31_equivalence_l1669_166963

theorem mod_31_equivalence : ∃! n : ℤ, 0 ≤ n ∧ n < 31 ∧ 78256 ≡ n [ZMOD 31] ∧ n = 19 := by
  sorry

end mod_31_equivalence_l1669_166963


namespace part_one_part_two_l1669_166965

-- Define the inequality
def inequality (a b x : ℝ) : Prop := a * x^2 - b ≥ 2 * x - a * x

-- Define the solution set
def solution_set (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ -1

-- Theorem for part (1)
theorem part_one (a b : ℝ) : 
  (∀ x, inequality a b x ↔ solution_set x) → a = -1 ∧ b = 2 := by sorry

-- Define the second inequality
def inequality_two (a x : ℝ) : Prop := (a * x - 2) * (x + 1) ≥ 0

-- Define the solution sets for part (2)
def solution_set_one (a x : ℝ) : Prop := 2 / a ≤ x ∧ x ≤ -1
def solution_set_two (x : ℝ) : Prop := x = -1
def solution_set_three (a x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 2 / a

-- Theorem for part (2)
theorem part_two (a : ℝ) (h : a < 0) :
  (∀ x, inequality_two a x ↔ 
    ((-2 < a ∧ a < 0 ∧ solution_set_one a x) ∨
     (a = -2 ∧ solution_set_two x) ∨
     (a < -2 ∧ solution_set_three a x))) := by sorry

end part_one_part_two_l1669_166965


namespace operation_on_original_number_l1669_166918

theorem operation_on_original_number : ∃ (f : ℝ → ℝ), 
  (3 * (f 4 + 9) = 51) ∧ (f 4 = 2 * 4) := by
  sorry

end operation_on_original_number_l1669_166918


namespace tv_ad_sequences_l1669_166942

/-- Represents the number of different broadcast sequences for advertisements -/
def num_broadcast_sequences (total_ads : ℕ) (commercial_ads : ℕ) (public_service_ads : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of different broadcast sequences for the given conditions -/
theorem tv_ad_sequences :
  let total_ads := 5
  let commercial_ads := 3
  let public_service_ads := 2
  num_broadcast_sequences total_ads commercial_ads public_service_ads = 36 :=
by
  sorry

end tv_ad_sequences_l1669_166942


namespace race_inequality_l1669_166980

theorem race_inequality (x : ℝ) : 
  (∀ (race_length : ℝ) (initial_speed : ℝ) (ming_speed : ℝ) (li_speed : ℝ) (distance_ahead : ℝ),
    race_length = 10000 ∧ 
    initial_speed = 200 ∧ 
    ming_speed = 250 ∧ 
    li_speed = 300 ∧ 
    distance_ahead = 200 ∧ 
    x > 0 ∧ 
    x < 50 ∧  -- This ensures Xiao Ming doesn't finish before encountering Xiao Li
    (race_length - initial_speed * x - distance_ahead) / ming_speed < 
      (race_length - initial_speed * x) / li_speed) →
  (10000 - 200 * x - 200) / 250 > (10000 - 200 * x) / 300 :=
by sorry

end race_inequality_l1669_166980


namespace smallest_n_for_Q_less_than_threshold_l1669_166944

def Q (n : ℕ) : ℚ := 2 / (n * (n + 1) * (n + 2))

theorem smallest_n_for_Q_less_than_threshold : 
  ∀ k : ℕ, k > 0 → k < 19 → Q (5 * k) ≥ 1 / 2500 ∧ Q (5 * 19) < 1 / 2500 :=
sorry

end smallest_n_for_Q_less_than_threshold_l1669_166944


namespace data_transformation_theorem_l1669_166906

variable {α : Type*} [LinearOrderedField α]

def average (data : Finset α) (f : α → α) : α :=
  (data.sum f) / data.card

def variance (data : Finset α) (f : α → α) (μ : α) : α :=
  (data.sum (fun x => (f x - μ) ^ 2)) / data.card

theorem data_transformation_theorem (data : Finset α) (f : α → α) :
  (average data (fun x => f x - 80) = 1.2) →
  (variance data (fun x => f x - 80) 1.2 = 4.4) →
  (average data f = 81.2) ∧ (variance data f 81.2 = 4.4) := by
  sorry

end data_transformation_theorem_l1669_166906


namespace chocolate_problem_l1669_166913

theorem chocolate_problem :
  ∃ n : ℕ, n ≥ 150 ∧ n % 17 = 15 ∧ ∀ m : ℕ, m ≥ 150 ∧ m % 17 = 15 → m ≥ n :=
by
  -- The proof goes here
  sorry

end chocolate_problem_l1669_166913


namespace batsman_matches_l1669_166970

theorem batsman_matches (total_matches : ℕ) (last_matches : ℕ) (last_avg : ℚ) (overall_avg : ℚ) :
  total_matches = 35 →
  last_matches = 13 →
  last_avg = 15 →
  overall_avg = 23.17142857142857 →
  total_matches - last_matches = 22 :=
by sorry

end batsman_matches_l1669_166970


namespace consecutive_integers_product_sum_l1669_166967

theorem consecutive_integers_product_sum (x : ℕ) : 
  x > 0 ∧ x * (x + 1) = 812 → x + (x + 1) = 57 := by
  sorry

end consecutive_integers_product_sum_l1669_166967
