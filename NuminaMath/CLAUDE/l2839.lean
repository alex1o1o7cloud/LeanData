import Mathlib

namespace cubic_equation_solution_l2839_283989

theorem cubic_equation_solution :
  ∃! x : ℝ, x^3 - 3*x^2 - 8*x + 40 - 8*(4*x + 4)^(1/4) = 0 :=
by
  sorry

end cubic_equation_solution_l2839_283989


namespace equal_representations_l2839_283969

/-- Represents the number of ways to write a positive integer as a product of powers of primes,
    where each factor is greater than or equal to the previous one. -/
def primeRepresentations (n : ℕ+) : ℕ := sorry

/-- Represents the number of ways to write a positive integer as a product of integers greater than 1,
    where each factor is divisible by all previous factors. -/
def divisibilityRepresentations (n : ℕ+) : ℕ := sorry

/-- Theorem stating that for any positive integer n, the number of prime representations
    is equal to the number of divisibility representations. -/
theorem equal_representations (n : ℕ+) : primeRepresentations n = divisibilityRepresentations n := by
  sorry

end equal_representations_l2839_283969


namespace combination_sum_theorem_l2839_283923

theorem combination_sum_theorem : 
  ∃ (n : ℕ+), 
    (0 ≤ 38 - n.val ∧ 38 - n.val ≤ 3 * n.val) ∧ 
    (n.val + 21 ≥ 3 * n.val) ∧ 
    (Nat.choose (3 * n.val) (38 - n.val) + Nat.choose (n.val + 21) (3 * n.val) = 466) := by
  sorry

end combination_sum_theorem_l2839_283923


namespace arithmetic_sequence_sum_l2839_283927

/-- Given an arithmetic sequence {a_n} where a_2 + a_3 + a_10 + a_11 = 48, prove that a_6 + a_7 = 24 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h1 : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h2 : a 2 + a 3 + a 10 + a 11 = 48) : a 6 + a 7 = 24 := by
  sorry

end arithmetic_sequence_sum_l2839_283927


namespace digit_table_size_l2839_283910

/-- A table with digits -/
structure DigitTable where
  rows : ℕ
  cols : ℕ
  digits : Fin rows → Fin cols → Fin 10

/-- The property that for any row and any two columns, there exists another row
    that differs only in those two columns -/
def hasTwoColumnDifference (t : DigitTable) : Prop :=
  ∀ (r : Fin t.rows) (c₁ c₂ : Fin t.cols),
    c₁ ≠ c₂ →
    ∃ (r' : Fin t.rows),
      r' ≠ r ∧
      (∀ (c : Fin t.cols), c ≠ c₁ ∧ c ≠ c₂ → t.digits r c = t.digits r' c) ∧
      (t.digits r c₁ ≠ t.digits r' c₁ ∨ t.digits r c₂ ≠ t.digits r' c₂)

/-- The main theorem -/
theorem digit_table_size (t : DigitTable) (h : t.cols = 10) (p : hasTwoColumnDifference t) :
  t.rows ≥ 512 := by
  sorry

end digit_table_size_l2839_283910


namespace f_property_l2839_283955

def f (a b x : ℝ) : ℝ := a * x^3 + b * x + 1

theorem f_property (a b : ℝ) : f a b (-2) = 3 → f a b 2 = -1 := by
  sorry

end f_property_l2839_283955


namespace floor_length_proof_l2839_283947

/-- Given two rectangular floors X and Y with equal areas, 
    where X is 10 feet by 18 feet and Y is 9 feet wide, 
    prove that the length of floor Y is 20 feet. -/
theorem floor_length_proof (area_x area_y length_x width_x width_y : ℝ) : 
  area_x = area_y → 
  length_x = 10 → 
  width_x = 18 → 
  width_y = 9 → 
  area_x = length_x * width_x → 
  area_y = width_y * (area_y / width_y) → 
  area_y / width_y = 20 := by
  sorry

#check floor_length_proof

end floor_length_proof_l2839_283947


namespace product_of_functions_l2839_283977

theorem product_of_functions (x : ℝ) (hx : x ≠ 0) : 
  let f : ℝ → ℝ := λ x => 2 * x
  let g : ℝ → ℝ := λ x => -(3 * x - 1) / x
  (f x) * (g x) = -6 * x + 2 := by
  sorry

end product_of_functions_l2839_283977


namespace parabola_shift_l2839_283935

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := x^2

-- Define the shifted parabola
def shifted_parabola (x : ℝ) : ℝ := (x + 3)^2 + 2

-- Theorem stating that the shifted parabola is correct
theorem parabola_shift :
  ∀ x : ℝ, shifted_parabola x = original_parabola (x + 3) + 2 :=
by
  sorry


end parabola_shift_l2839_283935


namespace area_outside_small_inside_large_l2839_283968

/-- Represents a circle with a center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Calculates the area of the region outside the smaller circle and inside the two larger circles -/
def areaOutsideSmallInsideLarge (smallCircle : Circle) (largeCircle1 largeCircle2 : Circle) : ℝ :=
  sorry

/-- Theorem stating the area of the specific configuration -/
theorem area_outside_small_inside_large :
  let smallCircle : Circle := { center := (0, 0), radius := 2 }
  let largeCircle1 : Circle := { center := (0, -2), radius := 3 }
  let largeCircle2 : Circle := { center := (0, 2), radius := 3 }
  areaOutsideSmallInsideLarge smallCircle largeCircle1 largeCircle2 = (5 * Real.pi / 2) - 4 * Real.sqrt 5 := by
  sorry

end area_outside_small_inside_large_l2839_283968


namespace expected_coincidences_value_l2839_283993

/-- The number of questions in the test -/
def num_questions : ℕ := 20

/-- Vasya's probability of guessing correctly -/
def p_vasya : ℚ := 6 / 20

/-- Misha's probability of guessing correctly -/
def p_misha : ℚ := 8 / 20

/-- The probability of a coincidence (both correct or both incorrect) for a single question -/
def p_coincidence : ℚ := p_vasya * p_misha + (1 - p_vasya) * (1 - p_misha)

/-- The expected number of coincidences -/
def expected_coincidences : ℚ := num_questions * p_coincidence

theorem expected_coincidences_value :
  expected_coincidences = 54 / 5 := by sorry

end expected_coincidences_value_l2839_283993


namespace circle_polygons_l2839_283959

/-- The number of points marked on the circle -/
def n : ℕ := 12

/-- The number of distinct convex polygons with 3 or more sides -/
def num_polygons : ℕ := 2^n - (n.choose 0 + n.choose 1 + n.choose 2)

theorem circle_polygons :
  num_polygons = 4017 :=
sorry

end circle_polygons_l2839_283959


namespace pasta_preference_ratio_l2839_283933

theorem pasta_preference_ratio (total_students : ℕ) (spaghetti_pref : ℕ) (fettuccine_pref : ℕ)
  (h_total : total_students = 800)
  (h_spaghetti : spaghetti_pref = 300)
  (h_fettuccine : fettuccine_pref = 80) :
  (spaghetti_pref : ℚ) / fettuccine_pref = 15 / 4 := by
  sorry

end pasta_preference_ratio_l2839_283933


namespace no_real_solutions_l2839_283960

theorem no_real_solutions :
  ∀ y : ℝ, (8 * y^2 + 155 * y + 3) / (4 * y + 45) ≠ 4 * y + 3 :=
by
  sorry

end no_real_solutions_l2839_283960


namespace smallest_k_for_two_roots_l2839_283978

/-- A quadratic trinomial with natural number coefficients -/
structure QuadraticTrinomial where
  k : ℕ
  p : ℕ
  q : ℕ

/-- Predicate to check if a quadratic trinomial has two distinct positive roots less than 1 -/
def has_two_distinct_positive_roots_less_than_one (qt : QuadraticTrinomial) : Prop :=
  ∃ (x₁ x₂ : ℝ), 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 ∧
    qt.k * x₁^2 - qt.p * x₁ + qt.q = 0 ∧
    qt.k * x₂^2 - qt.p * x₂ + qt.q = 0

/-- The main theorem stating that 5 is the smallest natural number k satisfying the condition -/
theorem smallest_k_for_two_roots : 
  (∀ k < 5, ¬∃ (p q : ℕ), has_two_distinct_positive_roots_less_than_one ⟨k, p, q⟩) ∧
  (∃ (p q : ℕ), has_two_distinct_positive_roots_less_than_one ⟨5, p, q⟩) :=
sorry

end smallest_k_for_two_roots_l2839_283978


namespace quadratic_function_value_l2839_283919

/-- A quadratic function f(x) = ax^2 + bx + c satisfying specific conditions -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ,
    (∀ x, f x = a * x^2 + b * x + c) ∧
    f 1 = 7 ∧
    f 2 = 12 ∧
    f 3 = 19

/-- Theorem stating that if f is a QuadraticFunction, then f(4) = 28 -/
theorem quadratic_function_value (f : ℝ → ℝ) (hf : QuadraticFunction f) : f 4 = 28 := by
  sorry

end quadratic_function_value_l2839_283919


namespace average_score_is_42_l2839_283999

/-- Intelligence contest game setup and results -/
structure ContestData where
  q1_points : ℕ := 20
  q2_points : ℕ := 25
  q3_points : ℕ := 25
  q1_correct : ℕ
  q2_correct : ℕ
  q3_correct : ℕ
  all_correct : ℕ := 1
  two_correct : ℕ := 15
  q1q2_sum : ℕ := 29
  q2q3_sum : ℕ := 20
  q1q3_sum : ℕ := 25

/-- Calculate the average score of the contest -/
def average_score (data : ContestData) : ℚ :=
  let total_participants := data.q1_correct + data.q2_correct + data.q3_correct - 2 * data.all_correct - data.two_correct
  let total_score := data.q1_correct * data.q1_points + (data.q2_correct + data.q3_correct) * data.q2_points
  (total_score : ℚ) / total_participants

/-- Theorem stating that the average score is 42 points -/
theorem average_score_is_42 (data : ContestData) 
  (h1 : data.q1_correct + data.q2_correct = data.q1q2_sum)
  (h2 : data.q2_correct + data.q3_correct = data.q2q3_sum)
  (h3 : data.q1_correct + data.q3_correct = data.q1q3_sum) :
  average_score data = 42 := by
  sorry


end average_score_is_42_l2839_283999


namespace only_D_greater_than_one_l2839_283971

theorem only_D_greater_than_one : 
  (0 / 0.16 ≤ 1) ∧ (1 * 0.16 ≤ 1) ∧ (1 / 1.6 ≤ 1) ∧ (1 * 1.6 > 1) := by
  sorry

end only_D_greater_than_one_l2839_283971


namespace quadratic_inequality_solution_l2839_283981

theorem quadratic_inequality_solution (x : ℝ) :
  -4 * x^2 + 7 * x + 2 < 0 ↔ x < -1/4 ∨ x > 2 :=
by sorry

end quadratic_inequality_solution_l2839_283981


namespace coefficient_of_x_squared_l2839_283988

def p (x : ℝ) : ℝ := x^5 - 2*x^4 + 4*x^3 - 5*x + 2
def q (x : ℝ) : ℝ := 3*x^4 - x^3 + x^2 + 4*x - 1

theorem coefficient_of_x_squared :
  ∃ (a b c d e f : ℝ),
    p x * q x = a*x^9 + b*x^8 + c*x^7 + d*x^6 + e*x^5 + (-18)*x^2 + f :=
by
  sorry

end coefficient_of_x_squared_l2839_283988


namespace coordinates_of_point_A_l2839_283914

-- Define a Point type for 2D coordinates
structure Point where
  x : ℝ
  y : ℝ

-- Define the problem statement
theorem coordinates_of_point_A (A B : Point) : 
  (B.x = 2 ∧ B.y = 4) →  -- Coordinates of point B
  (A.y = B.y) →  -- AB is parallel to x-axis
  ((A.x - B.x)^2 + (A.y - B.y)^2 = 3^2) →  -- Length of AB is 3
  ((A.x = 5 ∧ A.y = 4) ∨ (A.x = -1 ∧ A.y = 4)) := by
  sorry

end coordinates_of_point_A_l2839_283914


namespace mango_rate_calculation_l2839_283973

def grape_quantity : ℝ := 7
def grape_rate : ℝ := 70
def mango_quantity : ℝ := 9
def total_paid : ℝ := 985

theorem mango_rate_calculation :
  (total_paid - grape_quantity * grape_rate) / mango_quantity = 55 :=
by sorry

end mango_rate_calculation_l2839_283973


namespace dining_bill_share_l2839_283953

theorem dining_bill_share (total_bill : ℝ) (num_people : ℕ) (tip_percentage : ℝ) :
  total_bill = 139 ∧ num_people = 5 ∧ tip_percentage = 0.1 →
  (total_bill * (1 + tip_percentage)) / num_people = 30.58 := by
  sorry

end dining_bill_share_l2839_283953


namespace hexagon_angle_measure_l2839_283958

theorem hexagon_angle_measure (A N G L E S : ℝ) : 
  -- ANGLES is a hexagon
  A + N + G + L + E + S = 720 →
  -- ∠A ≅ ∠G ≅ ∠E
  A = G ∧ G = E →
  -- ∠N is supplementary to ∠S
  N + S = 180 →
  -- ∠L is a right angle
  L = 90 →
  -- The measure of ∠E is 150°
  E = 150 := by sorry

end hexagon_angle_measure_l2839_283958


namespace merchant_profit_problem_l2839_283903

theorem merchant_profit_problem (X : ℕ) (C S : ℝ) : 
  X * C = 25 * S → -- Cost price of X articles equals selling price of 25 articles
  S = 1.6 * C →    -- 60% profit, selling price is 160% of cost price
  X = 40           -- Number of articles bought at cost price is 40
  := by sorry

end merchant_profit_problem_l2839_283903


namespace specific_quadrilateral_area_l2839_283996

/-- Represents a convex quadrilateral ABCD with given side lengths and a right angle -/
structure ConvexQuadrilateral :=
  (AB : ℝ)
  (BC : ℝ)
  (CD : ℝ)
  (DA : ℝ)
  (angle_CDA : ℝ)
  (convex : AB > 0 ∧ BC > 0 ∧ CD > 0 ∧ DA > 0)
  (right_angle : angle_CDA = Real.pi / 2)

/-- Calculates the area of the convex quadrilateral ABCD -/
def area (q : ConvexQuadrilateral) : ℝ :=
  sorry

/-- Theorem stating that the area of the specific quadrilateral is 72 + 24√7 -/
theorem specific_quadrilateral_area :
  ∀ (q : ConvexQuadrilateral),
    q.AB = 10 ∧ q.BC = 6 ∧ q.CD = 12 ∧ q.DA = 12 →
    area q = 72 + 24 * Real.sqrt 7 :=
  sorry

end specific_quadrilateral_area_l2839_283996


namespace circular_table_dice_probability_l2839_283985

def num_people : ℕ := 5
def die_sides : ℕ := 6

def probability_no_adjacent_same : ℚ :=
  375 / 2592

theorem circular_table_dice_probability :
  let total_outcomes := die_sides ^ num_people
  let favorable_outcomes := 
    (die_sides * (die_sides - 1)^(num_people - 1) * (die_sides - 2)) +
    (die_sides * (die_sides - 1)^(num_people - 1) * (die_sides - 1) / die_sides)
  favorable_outcomes / total_outcomes = probability_no_adjacent_same := by
  sorry

end circular_table_dice_probability_l2839_283985


namespace three_digit_sum_magic_l2839_283944

/-- Given a three-digit number abc where a, b, and c are digits in base 10,
    if the sum of (acb), (bca), (bac), (cab), and (cba) is 3333,
    then abc = 555. -/
theorem three_digit_sum_magic (a b c : Nat) : 
  a < 10 → b < 10 → c < 10 →
  (100 * a + 10 * c + b) + 
  (100 * b + 10 * c + a) + 
  (100 * b + 10 * a + c) + 
  (100 * c + 10 * a + b) + 
  (100 * c + 10 * b + a) = 3333 →
  100 * a + 10 * b + c = 555 := by
  sorry


end three_digit_sum_magic_l2839_283944


namespace consecutive_even_sum_l2839_283940

theorem consecutive_even_sum (a b c : ℤ) : 
  (∃ n : ℤ, a = 2*n ∧ b = 2*n + 2 ∧ c = 2*n + 4) →  -- a, b, c are consecutive even integers
  (a + b + c = 246) →                              -- their sum is 246
  (c = 84) :=                                      -- the third number is 84
by
  sorry

end consecutive_even_sum_l2839_283940


namespace max_abs_z_cubed_minus_3z_minus_2_l2839_283934

/-- The maximum absolute value of z³ - 3z - 2 for complex z on the unit circle -/
theorem max_abs_z_cubed_minus_3z_minus_2 (z : ℂ) (h : Complex.abs z = 1) :
  ∃ (w : ℂ), Complex.abs w = 1 ∧ 
    ∀ (u : ℂ), Complex.abs u = 1 → 
      Complex.abs (w^3 - 3*w - 2) ≥ Complex.abs (u^3 - 3*u - 2) ∧
      Complex.abs (w^3 - 3*w - 2) = 3 * Real.sqrt 3 :=
by sorry

end max_abs_z_cubed_minus_3z_minus_2_l2839_283934


namespace inscribed_rectangle_sides_l2839_283951

theorem inscribed_rectangle_sides (a b c : ℝ) (x y : ℝ) : 
  a = 10 ∧ b = 17 ∧ c = 21 →  -- Triangle sides
  c > a ∧ c > b →  -- c is the longest side
  x + y = 12 →  -- Half of rectangle's perimeter
  y < 8 →  -- Rectangle's height is less than triangle's height
  (8 - y) / 8 = (c - x) / c →  -- Similarity of triangles
  x = 72 / 13 ∧ y = 84 / 13 := by
sorry

end inscribed_rectangle_sides_l2839_283951


namespace triangular_prism_float_l2839_283995

theorem triangular_prism_float (x : ℝ) : ¬ (0 < x ∧ x < Real.sqrt 3 / 2 ∧ (Real.sqrt 3 / 4) * x = x * (1 - x / Real.sqrt 3)) := by
  sorry

end triangular_prism_float_l2839_283995


namespace decimal_to_fraction_simplest_l2839_283943

theorem decimal_to_fraction_simplest : 
  ∃ (a b : ℕ), 
    a > 0 ∧ b > 0 ∧ 
    (a : ℚ) / (b : ℚ) = 0.84375 ∧
    ∀ (c d : ℕ), c > 0 ∧ d > 0 ∧ (c : ℚ) / (d : ℚ) = 0.84375 → b ≤ d ∧
    a + b = 59 := by
  sorry

end decimal_to_fraction_simplest_l2839_283943


namespace derivative_at_one_l2839_283982

-- Define the function f(x) = x^2 + 1
def f (x : ℝ) : ℝ := x^2 + 1

-- State the theorem
theorem derivative_at_one (x : ℝ) :
  ∀ ε > 0, ∃ δ > 0, ∀ h ≠ 0, |h| < δ → |((f (1 + h) - f 1) / h) - 2| < ε := by
  sorry

end derivative_at_one_l2839_283982


namespace merchant_pricing_strategy_l2839_283992

theorem merchant_pricing_strategy (L : ℝ) (h : L > 0) :
  let purchase_price := L * 0.7
  let marked_price := L * 1.25
  let selling_price := marked_price * 0.8
  selling_price = purchase_price * 1.3 := by
sorry

end merchant_pricing_strategy_l2839_283992


namespace vector_c_satisfies_conditions_l2839_283930

/-- Given vectors a and b in ℝ², prove that vector c satisfies the required conditions -/
theorem vector_c_satisfies_conditions (a b c : ℝ × ℝ) : 
  a = (1, 2) → b = (2, -3) → c = (7/2, -7/4) → 
  (c.1 * a.1 + c.2 * a.2 = 0) ∧ 
  (∃ k : ℝ, b.1 = k * (a.1 - c.1) ∧ b.2 = k * (a.2 - c.2)) := by
sorry

end vector_c_satisfies_conditions_l2839_283930


namespace gcd_of_repeated_numbers_l2839_283921

/-- A fifteen-digit integer formed by repeating a five-digit integer three times -/
def repeatedNumber (n : ℕ) : ℕ := n * 10000100001

/-- The set of all such fifteen-digit numbers -/
def S : Set ℕ := {m : ℕ | ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ m = repeatedNumber n}

theorem gcd_of_repeated_numbers : 
  ∃ d : ℕ, d > 0 ∧ ∀ m ∈ S, d ∣ m ∧ ∀ k : ℕ, (∀ m ∈ S, k ∣ m) → k ∣ d :=
by sorry

end gcd_of_repeated_numbers_l2839_283921


namespace x_squared_coefficient_l2839_283980

/-- The coefficient of x^2 in the expansion of (x^2+x+1)(1-x)^6 is 10 -/
theorem x_squared_coefficient : Int := by
  sorry

end x_squared_coefficient_l2839_283980


namespace perimeter_ratio_not_integer_l2839_283906

theorem perimeter_ratio_not_integer (a k l : ℕ+) (h : a^2 = k * l) :
  ¬ ∃ (n : ℕ), (k + l : ℚ) / (2 * a) = n := by
  sorry

end perimeter_ratio_not_integer_l2839_283906


namespace horner_polynomial_eval_l2839_283942

def horner_eval (coeffs : List ℤ) (x : ℤ) : ℤ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

theorem horner_polynomial_eval :
  let coeffs := [7, 3, -5, 11]
  let x := 23
  horner_eval coeffs x = 86652 := by
sorry

end horner_polynomial_eval_l2839_283942


namespace solution_values_l2839_283932

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

def is_valid_p (a b p : E) : Prop :=
  ‖p - b‖ = 3 * ‖p - a‖

def fixed_distance (a b p : E) (t u : ℝ) : Prop :=
  ∃ (k : ℝ), ‖p - (t • a + u • b)‖ = k

theorem solution_values (a b : E) :
  ∃ (p : E), is_valid_p a b p ∧
  fixed_distance a b p (9/8) (-1/8) :=
sorry

end solution_values_l2839_283932


namespace jason_shelves_needed_l2839_283994

/-- Calculates the number of shelves needed to store books -/
def shelves_needed (regular_books : ℕ) (large_books : ℕ) : ℕ :=
  let regular_shelves := (regular_books + 44) / 45
  let large_shelves := (large_books + 29) / 30
  regular_shelves + large_shelves

/-- Theorem stating that Jason needs 9 shelves to store all his books -/
theorem jason_shelves_needed : shelves_needed 240 75 = 9 := by
  sorry

end jason_shelves_needed_l2839_283994


namespace fibonacci_arithmetic_sequence_l2839_283979

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Define the property of forming an increasing arithmetic sequence
def is_increasing_arithmetic_seq (a b c : ℕ) : Prop :=
  fib b - fib a = fib c - fib b ∧ fib a < fib b ∧ fib b < fib c

-- State the theorem
theorem fibonacci_arithmetic_sequence :
  ∃ (a b c : ℕ),
    is_increasing_arithmetic_seq a b c ∧
    a + b + c = 2000 ∧
    a = 665 := by
  sorry

end fibonacci_arithmetic_sequence_l2839_283979


namespace class_average_calculation_l2839_283905

/-- Proves that the overall average of a class is 63.4 marks given specific score distributions -/
theorem class_average_calculation (total_students : ℕ) 
  (high_scorers : ℕ) (high_score : ℝ)
  (zero_scorers : ℕ) 
  (mid_scorers : ℕ) (mid_score : ℝ)
  (remaining_scorers : ℕ) (remaining_score : ℝ) :
  total_students = 50 ∧ 
  high_scorers = 6 ∧ 
  high_score = 95 ∧
  zero_scorers = 4 ∧
  mid_scorers = 10 ∧
  mid_score = 80 ∧
  remaining_scorers = total_students - (high_scorers + zero_scorers + mid_scorers) ∧
  remaining_score = 60 →
  (high_scorers * high_score + zero_scorers * 0 + mid_scorers * mid_score + remaining_scorers * remaining_score) / total_students = 63.4 := by
  sorry

#eval (6 * 95 + 4 * 0 + 10 * 80 + 30 * 60) / 50

end class_average_calculation_l2839_283905


namespace red_and_large_toys_l2839_283976

/-- Represents the color of a toy -/
inductive Color
| Red
| Green
| Blue
| Yellow
| Orange

/-- Represents the size of a toy -/
inductive Size
| Small
| Medium
| Large
| ExtraLarge

/-- Represents the distribution of toys by color and size -/
structure ToyDistribution where
  red_small : Rat
  red_medium : Rat
  red_large : Rat
  red_extra_large : Rat
  green_small : Rat
  green_medium : Rat
  green_large : Rat
  green_extra_large : Rat
  blue_small : Rat
  blue_medium : Rat
  blue_large : Rat
  blue_extra_large : Rat
  yellow_small : Rat
  yellow_medium : Rat
  yellow_large : Rat
  yellow_extra_large : Rat
  orange_small : Rat
  orange_medium : Rat
  orange_large : Rat
  orange_extra_large : Rat

/-- The given distribution of toys -/
def given_distribution : ToyDistribution :=
  { red_small := 6/100, red_medium := 8/100, red_large := 7/100, red_extra_large := 4/100,
    green_small := 4/100, green_medium := 7/100, green_large := 5/100, green_extra_large := 4/100,
    blue_small := 6/100, blue_medium := 3/100, blue_large := 4/100, blue_extra_large := 2/100,
    yellow_small := 8/100, yellow_medium := 10/100, yellow_large := 5/100, yellow_extra_large := 2/100,
    orange_small := 9/100, orange_medium := 6/100, orange_large := 5/100, orange_extra_large := 5/100 }

/-- Theorem stating the number of red and large toys -/
theorem red_and_large_toys (total_toys : ℕ) (h : total_toys * given_distribution.green_large = 47) :
  total_toys * given_distribution.red_large = 329 := by
  sorry

end red_and_large_toys_l2839_283976


namespace prob_greater_than_four_l2839_283931

-- Define a fair 6-sided die
def fair_die : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define the probability of an event on a fair die
def prob (event : Finset ℕ) : ℚ :=
  (event ∩ fair_die).card / fair_die.card

-- Define the event of rolling a number greater than 4
def greater_than_four : Finset ℕ := {5, 6}

-- Theorem statement
theorem prob_greater_than_four :
  prob greater_than_four = 1/3 := by sorry

end prob_greater_than_four_l2839_283931


namespace optimal_allocation_l2839_283949

/-- Represents the allocation of workers in a furniture factory -/
structure WorkerAllocation where
  total_workers : ℕ
  tabletop_workers : ℕ
  tableleg_workers : ℕ
  tabletops_per_worker : ℕ
  tablelegs_per_worker : ℕ
  legs_per_table : ℕ

/-- Checks if the allocation produces matching numbers of tabletops and table legs -/
def is_matching_production (w : WorkerAllocation) : Prop :=
  w.tabletop_workers * w.tabletops_per_worker * w.legs_per_table = 
  w.tableleg_workers * w.tablelegs_per_worker

/-- The theorem stating the optimal worker allocation -/
theorem optimal_allocation :
  ∀ w : WorkerAllocation,
    w.total_workers = 60 ∧
    w.tabletops_per_worker = 3 ∧
    w.tablelegs_per_worker = 6 ∧
    w.legs_per_table = 4 ∧
    w.tabletop_workers + w.tableleg_workers = w.total_workers →
    (w.tabletop_workers = 20 ∧ w.tableleg_workers = 40) ↔ 
    is_matching_production w :=
by sorry

end optimal_allocation_l2839_283949


namespace intersection_of_A_and_B_l2839_283917

-- Define the sets A and B
def A : Set ℝ := {x | (x + 1) / (x - 1) ≤ 0}
def B : Set ℝ := {x | Real.log x ≤ 0}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = Set.Ioo 0 1 := by sorry

end intersection_of_A_and_B_l2839_283917


namespace rectangular_prism_width_zero_l2839_283963

/-- A rectangular prism with given dimensions --/
structure RectangularPrism where
  l : ℝ  -- length
  h : ℝ  -- height
  d : ℝ  -- diagonal
  w : ℝ  -- width

/-- The theorem stating that a rectangular prism with length 6, height 8, and diagonal 10 has width 0 --/
theorem rectangular_prism_width_zero (p : RectangularPrism) 
  (hl : p.l = 6) 
  (hh : p.h = 8) 
  (hd : p.d = 10) : 
  p.w = 0 := by
  sorry

end rectangular_prism_width_zero_l2839_283963


namespace extended_euclidean_algorithm_l2839_283962

theorem extended_euclidean_algorithm (m₀ m₁ : ℤ) (h : 0 < m₁ ∧ m₁ ≤ m₀) :
  ∃ u v : ℤ, m₀ * u + m₁ * v = Int.gcd m₀ m₁ := by
  sorry

end extended_euclidean_algorithm_l2839_283962


namespace chris_age_l2839_283974

theorem chris_age (a b c : ℚ) : 
  (a + b + c) / 3 = 12 →
  c - 5 = 2 * a →
  b + 2 = (a + 2) / 2 →
  c = 163 / 7 := by
sorry

end chris_age_l2839_283974


namespace triangle_centroid_length_l2839_283909

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  -- Right triangle condition
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0 ∧
  -- BC = 6
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = 36 ∧
  -- AC = 8
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 64

-- Define the centroid
def Centroid (O A B C : ℝ × ℝ) : Prop :=
  O.1 = (A.1 + B.1 + C.1) / 3 ∧ O.2 = (A.2 + B.2 + C.2) / 3

-- Define the midpoint
def Midpoint (M A B : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

-- Main theorem
theorem triangle_centroid_length (A B C O P Q : ℝ × ℝ) :
  Triangle A B C →
  Centroid O A B C →
  Midpoint Q A B →
  Midpoint P B C →
  ((O.1 - P.1)^2 + (O.2 - P.2)^2) = (4/9) * 73 :=
by sorry

end triangle_centroid_length_l2839_283909


namespace fraction_sum_and_product_l2839_283970

theorem fraction_sum_and_product (x y : ℚ) :
  x + y = 13/14 ∧ x * y = 3/28 →
  (x = 3/7 ∧ y = 1/4) ∨ (x = 1/4 ∧ y = 3/7) := by
sorry

end fraction_sum_and_product_l2839_283970


namespace curve_is_parabola_l2839_283964

/-- The curve defined by √X + √Y = 1 is a parabola -/
theorem curve_is_parabola :
  ∃ (a b c : ℝ) (h : a ≠ 0),
    ∀ (x y : ℝ),
      (Real.sqrt x + Real.sqrt y = 1) ↔ (y = a * x^2 + b * x + c) :=
sorry

end curve_is_parabola_l2839_283964


namespace find_divisor_l2839_283972

theorem find_divisor (divisor : ℕ) : 
  (∃ k : ℕ, (228712 + 5) = divisor * k) ∧ 
  (∀ n < 5, ¬∃ m : ℕ, (228712 + n) = divisor * m) →
  divisor = 3 := by
sorry

end find_divisor_l2839_283972


namespace max_rock_value_is_58_l2839_283912

/-- Represents a type of rock with its weight and value -/
structure Rock where
  weight : ℕ
  value : ℕ

/-- Calculates the maximum value of rocks that can be carried given the constraints -/
def maxRockValue (rocks : List Rock) (maxWeight : ℕ) (maxSixPoundRocks : ℕ) : ℕ :=
  sorry

/-- The main theorem stating the maximum value of rocks Carl can carry -/
theorem max_rock_value_is_58 :
  let rocks : List Rock := [
    { weight := 3, value := 9 },
    { weight := 6, value := 20 },
    { weight := 2, value := 5 }
  ]
  let maxWeight : ℕ := 20
  let maxSixPoundRocks : ℕ := 2
  maxRockValue rocks maxWeight maxSixPoundRocks = 58 := by
  sorry

end max_rock_value_is_58_l2839_283912


namespace ln_f_greater_than_one_max_a_value_l2839_283900

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 2| + |x - a|

-- Theorem for part (I)
theorem ln_f_greater_than_one :
  ∀ x : ℝ, Real.log (f (-1) x) > 1 := by sorry

-- Theorem for part (II)
theorem max_a_value :
  (∃ a : ℝ, ∀ x : ℝ, f a x ≥ a) ∧
  (∀ b : ℝ, (∀ x : ℝ, f b x ≥ b) → b ≤ 1) := by sorry

end ln_f_greater_than_one_max_a_value_l2839_283900


namespace total_games_in_league_l2839_283936

theorem total_games_in_league (n : ℕ) (h : n = 12) : 
  (n * (n - 1)) / 2 = 66 := by
  sorry

end total_games_in_league_l2839_283936


namespace elsa_token_count_l2839_283908

/-- The number of tokens Angus has -/
def angus_tokens : ℕ := 55

/-- The value of each token in dollars -/
def token_value : ℕ := 4

/-- The difference in dollar value between Elsa's and Angus's tokens -/
def value_difference : ℕ := 20

/-- The number of tokens Elsa has -/
def elsa_tokens : ℕ := 60

theorem elsa_token_count : elsa_tokens = 60 := by
  sorry

end elsa_token_count_l2839_283908


namespace car_rental_cost_l2839_283990

/-- The daily rental cost of a car, given specific conditions. -/
theorem car_rental_cost (daily_rate : ℝ) (cost_per_mile : ℝ) (budget : ℝ) (miles : ℝ) : 
  cost_per_mile = 0.23 →
  budget = 76 →
  miles = 200 →
  daily_rate + cost_per_mile * miles = budget →
  daily_rate = 30 := by
sorry

end car_rental_cost_l2839_283990


namespace intersection_nonempty_iff_l2839_283967

/-- Given sets A and B, prove the condition for their non-empty intersection -/
theorem intersection_nonempty_iff (a : ℝ) : 
  (∃ x : ℝ, x ∈ {x | 1 ≤ x ∧ x ≤ 2} ∩ {x | x ≤ a}) ↔ a ≥ 1 := by
  sorry

end intersection_nonempty_iff_l2839_283967


namespace rectangle_longer_side_l2839_283939

theorem rectangle_longer_side (a : ℝ) (h1 : a > 0) : 
  (a * (0.8 * a) = 81/20) → a = 2.25 := by
  sorry

end rectangle_longer_side_l2839_283939


namespace sara_second_book_cost_l2839_283948

/-- The cost of Sara's second book -/
def second_book_cost (first_book_cost bill_given change_received : ℝ) : ℝ :=
  bill_given - change_received - first_book_cost

/-- Theorem stating the cost of Sara's second book -/
theorem sara_second_book_cost :
  second_book_cost 5.5 20 8 = 6.5 := by
  sorry

end sara_second_book_cost_l2839_283948


namespace isosceles_triangle_area_l2839_283997

/-- An isosceles triangle with side lengths 13, 13, and 10 has an area of 60 square units. -/
theorem isosceles_triangle_area (A B C : ℝ × ℝ) : 
  let d := (fun p q : ℝ × ℝ => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2))
  (d A B = 13 ∧ d A C = 13 ∧ d B C = 10) →
  (A.1 - B.1) * (C.2 - B.2) - (C.1 - B.1) * (A.2 - B.2) = 120 :=
by sorry


end isosceles_triangle_area_l2839_283997


namespace impossible_coloring_l2839_283916

theorem impossible_coloring (R G B : Set ℤ) : 
  (∀ (x y : ℤ), (x ∈ G ∧ y ∈ B) ∨ (x ∈ R ∧ y ∈ B) ∨ (x ∈ R ∧ y ∈ G) → x + y ∈ R) →
  (R ∪ G ∪ B = Set.univ) →
  (R ∩ G = ∅ ∧ R ∩ B = ∅ ∧ G ∩ B = ∅) →
  (R ≠ ∅ ∧ G ≠ ∅ ∧ B ≠ ∅) →
  False :=
by sorry

end impossible_coloring_l2839_283916


namespace circle_equation_l2839_283902

-- Define the circle C
def Circle (center : ℝ × ℝ) (radius : ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define points A and B
def A : ℝ × ℝ := (0, -6)
def B : ℝ × ℝ := (1, -5)

-- Define the line l
def line_l (p : ℝ × ℝ) : Prop := p.1 - p.2 + 1 = 0

-- Theorem statement
theorem circle_equation :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    A ∈ Circle center radius ∧
    B ∈ Circle center radius ∧
    line_l center ∧
    center = (-3, -2) ∧
    radius = 5 :=
  sorry

end circle_equation_l2839_283902


namespace sqrt_x_minus_2_real_l2839_283998

theorem sqrt_x_minus_2_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 2) ↔ x ≥ 2 := by
  sorry

end sqrt_x_minus_2_real_l2839_283998


namespace gcf_of_3150_and_7350_l2839_283946

theorem gcf_of_3150_and_7350 : Nat.gcd 3150 7350 = 525 := by
  sorry

end gcf_of_3150_and_7350_l2839_283946


namespace women_at_dance_event_l2839_283907

/-- Represents a dance event with men and women -/
structure DanceEvent where
  men : ℕ
  women : ℕ
  men_dances : ℕ
  women_dances : ℕ

/-- Calculates the total number of dance pairs in the event -/
def total_dance_pairs (event : DanceEvent) : ℕ :=
  event.men * event.men_dances

/-- Theorem: Given the conditions of the dance event, prove that 24 women attended -/
theorem women_at_dance_event (event : DanceEvent) 
  (h1 : event.men_dances = 4)
  (h2 : event.women_dances = 3)
  (h3 : event.men = 18) :
  event.women = 24 := by
  sorry

#check women_at_dance_event

end women_at_dance_event_l2839_283907


namespace equation_solution_l2839_283928

theorem equation_solution :
  ∃ (x : ℝ), x^2 - 4 ≠ 0 ∧ (x - 2) / (x + 2) + 4 / (x^2 - 4) = 1 ∧ x = 3 :=
by sorry

end equation_solution_l2839_283928


namespace crayons_lost_or_given_away_l2839_283925

theorem crayons_lost_or_given_away (start_crayons end_crayons : ℕ) 
  (h1 : start_crayons = 253)
  (h2 : end_crayons = 183) :
  start_crayons - end_crayons = 70 := by
  sorry

end crayons_lost_or_given_away_l2839_283925


namespace negation_of_conditional_l2839_283938

theorem negation_of_conditional (x : ℝ) :
  ¬(x > 1 → x^2 > x) ↔ (x ≤ 1 → x^2 ≤ x) :=
by sorry

end negation_of_conditional_l2839_283938


namespace hair_length_calculation_l2839_283983

theorem hair_length_calculation (initial_length cut_length growth_length : ℕ) 
  (h1 : initial_length = 16)
  (h2 : cut_length = 11)
  (h3 : growth_length = 12) :
  initial_length - cut_length + growth_length = 17 := by
  sorry

end hair_length_calculation_l2839_283983


namespace sqrt_two_squared_inverse_l2839_283956

theorem sqrt_two_squared_inverse : ((-Real.sqrt 2)^2)⁻¹ = (1/2 : ℝ) := by
  sorry

end sqrt_two_squared_inverse_l2839_283956


namespace train_length_calculation_l2839_283984

/-- Proves that given a train and platform of equal length, if the train crosses the platform
    in one minute at a speed of 216 km/hr, then the length of the train is 1800 meters. -/
theorem train_length_calculation (train_length platform_length : ℝ) 
    (speed : ℝ) (time : ℝ) :
  train_length = platform_length →
  speed = 216 →
  time = 1 / 60 →
  train_length = 1800 := by
  sorry

end train_length_calculation_l2839_283984


namespace hoseok_additional_jumps_theorem_l2839_283911

/-- The number of additional jumps Hoseok needs to match Minyoung's total -/
def additional_jumps (hoseok_jumps minyoung_jumps : ℕ) : ℕ :=
  minyoung_jumps - hoseok_jumps

theorem hoseok_additional_jumps_theorem (hoseok_jumps minyoung_jumps : ℕ) 
    (h : minyoung_jumps > hoseok_jumps) :
  additional_jumps hoseok_jumps minyoung_jumps = 17 :=
by
  sorry

#eval additional_jumps 34 51

end hoseok_additional_jumps_theorem_l2839_283911


namespace melanie_picked_zero_pears_l2839_283904

/-- The number of pears Melanie picked -/
def melanie_pears : ℕ := 0

/-- The number of plums Alyssa picked -/
def alyssa_plums : ℕ := 17

/-- The number of plums Jason picked -/
def jason_plums : ℕ := 10

/-- The total number of plums picked -/
def total_plums : ℕ := 27

theorem melanie_picked_zero_pears :
  alyssa_plums + jason_plums = total_plums → melanie_pears = 0 := by
  sorry

end melanie_picked_zero_pears_l2839_283904


namespace modulus_of_z_is_sqrt_2_l2839_283945

theorem modulus_of_z_is_sqrt_2 :
  let z : ℂ := 1 - 1 / Complex.I
  Complex.abs z = Real.sqrt 2 := by sorry

end modulus_of_z_is_sqrt_2_l2839_283945


namespace max_abs_z_l2839_283937

theorem max_abs_z (z : ℂ) (h : Complex.abs (z + 3 + 4 * I) ≤ 2) :
  ∃ (max_val : ℝ), max_val = 7 ∧ ∀ w : ℂ, Complex.abs (w + 3 + 4 * I) ≤ 2 → Complex.abs w ≤ max_val :=
by sorry

end max_abs_z_l2839_283937


namespace complement_intersection_theorem_l2839_283952

def U : Set ℤ := {-1, 0, 1, 2}
def A : Set ℤ := {1, 2}
def B : Set ℤ := {0, 2}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {0} := by sorry

end complement_intersection_theorem_l2839_283952


namespace sum_of_x_and_y_l2839_283926

theorem sum_of_x_and_y (x y : ℤ) (h1 : x - y = 200) (h2 : y = 230) : x + y = 660 := by
  sorry

end sum_of_x_and_y_l2839_283926


namespace nth_equation_and_specific_case_l2839_283924

theorem nth_equation_and_specific_case :
  (∀ n : ℕ, n > 0 → Real.sqrt (1 - (2 * n - 1) / (n * n)) = (n - 1) / n) ∧
  Real.sqrt (1 - 199 / 10000) = 99 / 100 :=
by sorry

end nth_equation_and_specific_case_l2839_283924


namespace sum_of_four_consecutive_integers_l2839_283913

theorem sum_of_four_consecutive_integers (a b c d : ℤ) : 
  (a + 1 = b) ∧ (b + 1 = c) ∧ (c + 1 = d) ∧ (d = 27) → a + b + c + d = 102 := by
  sorry

end sum_of_four_consecutive_integers_l2839_283913


namespace initial_time_theorem_l2839_283961

/-- Given a distance of 720 km, if increasing the initial time by 3/2 results in a speed of 80 kmph,
    then the initial time taken to cover the distance was 6 hours. -/
theorem initial_time_theorem (t : ℝ) (h1 : t > 0) : 
  (720 : ℝ) / ((3/2) * t) = 80 → t = 6 := by
  sorry

end initial_time_theorem_l2839_283961


namespace line_intersects_parabola_once_l2839_283987

/-- The value of k for which the line x = k intersects the parabola x = 3y² - 7y + 2 at exactly one point -/
def k : ℚ := -25/12

/-- The parabola equation -/
def parabola (y : ℝ) : ℝ := 3*y^2 - 7*y + 2

/-- Theorem stating that k is the unique value for which the line x = k intersects the parabola at exactly one point -/
theorem line_intersects_parabola_once :
  ∀ y : ℝ, (∃! y, parabola y = k) ∧ 
  (∀ k' : ℚ, k' ≠ k → ¬(∃! y, parabola y = k')) :=
sorry

end line_intersects_parabola_once_l2839_283987


namespace marcella_shoes_l2839_283986

theorem marcella_shoes (initial_pairs : ℕ) : 
  (initial_pairs * 2 - 9 ≥ 21 * 2) ∧ 
  (∀ n : ℕ, n > initial_pairs → n * 2 - 9 < 21 * 2) → 
  initial_pairs = 25 := by
sorry

end marcella_shoes_l2839_283986


namespace total_feathers_is_11638_l2839_283901

/-- The total number of feathers needed for all animals in the circus performance --/
def total_feathers : ℕ :=
  let group1_animals : ℕ := 934
  let group1_feathers_per_crown : ℕ := 7
  let group2_animals : ℕ := 425
  let group2_feathers_per_crown : ℕ := 12
  (group1_animals * group1_feathers_per_crown) + (group2_animals * group2_feathers_per_crown)

/-- Theorem stating that the total number of feathers needed is 11638 --/
theorem total_feathers_is_11638 : total_feathers = 11638 := by
  sorry

end total_feathers_is_11638_l2839_283901


namespace alyssas_total_spent_l2839_283957

/-- The amount Alyssa paid for grapes in dollars -/
def grapes_cost : ℚ := 12.08

/-- The amount Alyssa was refunded for cherries in dollars -/
def cherries_refund : ℚ := 9.85

/-- The total amount Alyssa spent in dollars -/
def total_spent : ℚ := grapes_cost - cherries_refund

/-- Theorem stating that the total amount Alyssa spent is $2.23 -/
theorem alyssas_total_spent : total_spent = 2.23 := by
  sorry

end alyssas_total_spent_l2839_283957


namespace min_value_complex_expression_l2839_283915

theorem min_value_complex_expression (a b c : ℤ) (ξ : ℂ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_fourth_root : ξ^4 = 1)
  (h_not_one : ξ ≠ 1) :
  ∃ (m : ℝ), m = 1 ∧ ∀ (x y z : ℤ), x ≠ y ∧ y ≠ z ∧ x ≠ z → 
    Complex.abs (↑x + ↑y * ξ + ↑z * ξ^3) ≥ m :=
sorry

end min_value_complex_expression_l2839_283915


namespace max_dimes_and_nickels_l2839_283965

def total_amount : ℚ := 485 / 100
def dime_value : ℚ := 10 / 100
def nickel_value : ℚ := 5 / 100

theorem max_dimes_and_nickels :
  ∃ (d : ℕ), d * dime_value + d * nickel_value ≤ total_amount ∧
  ∀ (n : ℕ), n * dime_value + n * nickel_value ≤ total_amount → n ≤ d :=
by sorry

end max_dimes_and_nickels_l2839_283965


namespace train_speed_l2839_283929

theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 450) (h2 : time = 8) :
  length / time = 56.25 := by
  sorry

end train_speed_l2839_283929


namespace distance_traveled_l2839_283922

theorem distance_traveled (original_speed : ℝ) (increased_speed : ℝ) (additional_distance : ℝ) 
  (h1 : original_speed = 8)
  (h2 : increased_speed = 12)
  (h3 : additional_distance = 20)
  : ∃ (actual_distance : ℝ) (time : ℝ),
    actual_distance = original_speed * time ∧
    actual_distance + additional_distance = increased_speed * time ∧
    actual_distance = 40 := by
  sorry

end distance_traveled_l2839_283922


namespace salary_after_four_months_l2839_283950

def salary_calculation (initial_salary : ℝ) (initial_increase_rate : ℝ) (initial_bonus : ℝ) (bonus_increase_rate : ℝ) (months : ℕ) : ℝ :=
  let rec helper (current_salary : ℝ) (current_bonus : ℝ) (current_increase_rate : ℝ) (month : ℕ) : ℝ :=
    if month = 0 then
      current_salary + current_bonus
    else
      let new_salary := current_salary * (1 + current_increase_rate)
      let new_bonus := current_bonus * (1 + bonus_increase_rate)
      let new_increase_rate := current_increase_rate * 2
      helper new_salary new_bonus new_increase_rate (month - 1)
  helper initial_salary initial_bonus initial_increase_rate months

theorem salary_after_four_months :
  salary_calculation 2000 0.05 150 0.1 4 = 4080.45 := by
  sorry

end salary_after_four_months_l2839_283950


namespace sum_of_powers_of_three_l2839_283941

theorem sum_of_powers_of_three : (-3)^4 + (-3)^2 + (-3)^0 + 3^0 + 3^2 + 3^4 = 182 := by
  sorry

end sum_of_powers_of_three_l2839_283941


namespace sum_of_squares_of_roots_l2839_283918

theorem sum_of_squares_of_roots (p q r : ℂ) : 
  (3 * p^3 - 3 * p^2 + 6 * p - 9 = 0) →
  (3 * q^3 - 3 * q^2 + 6 * q - 9 = 0) →
  (3 * r^3 - 3 * r^2 + 6 * r - 9 = 0) →
  p^2 + q^2 + r^2 = -3 := by
  sorry

end sum_of_squares_of_roots_l2839_283918


namespace cube_root_cube_equality_l2839_283920

theorem cube_root_cube_equality (x : ℝ) : x = (x^3)^(1/3) := by
  sorry

end cube_root_cube_equality_l2839_283920


namespace sports_meeting_medals_l2839_283991

/-- The number of medals awarded on day x -/
def f (x m : ℕ) : ℚ :=
  if x = 1 then
    1 + (m - 1) / 7
  else
    (6 / 7) ^ (x - 1) * ((m - 36) / 7) + 6

/-- The total number of medals awarded over n days -/
def total_medals (n : ℕ) : ℕ := 36

/-- The number of days the sports meeting lasted -/
def meeting_duration : ℕ := 6

theorem sports_meeting_medals :
  ∀ n m : ℕ,
  (∀ i : ℕ, i < n → f i m = i + (f (i+1) m - i) / 7) →
  f n m = n →
  n = meeting_duration ∧ m = total_medals n :=
sorry

end sports_meeting_medals_l2839_283991


namespace solution_is_ray_iff_a_is_pm1_l2839_283954

/-- The polynomial function in x parameterized by a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - (a^2 + a + 1)*x^2 + (a^3 + a^2 + a)*x - a^3

/-- The set of solutions for the inequality -/
def solution_set (a : ℝ) : Set ℝ := {x : ℝ | f a x ≥ 0}

/-- Definition of a ray (half-line) in ℝ -/
def is_ray (S : Set ℝ) : Prop :=
  ∃ (x₀ : ℝ), S = {x : ℝ | x ≥ x₀} ∨ S = {x : ℝ | x ≤ x₀}

/-- The main theorem -/
theorem solution_is_ray_iff_a_is_pm1 :
  ∀ a : ℝ, is_ray (solution_set a) ↔ (a = 1 ∨ a = -1) :=
sorry

end solution_is_ray_iff_a_is_pm1_l2839_283954


namespace equal_intercept_line_equation_l2839_283966

/-- A line passing through point (1,1) with equal intercepts on both axes -/
structure EqualInterceptLine where
  /-- The slope of the line -/
  slope : ℝ
  /-- The y-intercept of the line -/
  y_intercept : ℝ
  /-- The line passes through point (1,1) -/
  passes_through_point : slope + y_intercept = 2
  /-- The line has equal intercepts on both axes -/
  equal_intercepts : y_intercept = slope * y_intercept ∨ y_intercept = 0

/-- The equation of an EqualInterceptLine is x + y = 2 or y = x -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (l.slope = 1 ∧ l.y_intercept = 1) ∨ (l.slope = 1 ∧ l.y_intercept = 0) :=
sorry

end equal_intercept_line_equation_l2839_283966


namespace linear_function_negative_slope_l2839_283975

/-- Given a linear function y = kx + b passing through points A(1, m) and B(-1, n), 
    where m < n and k ≠ 0, prove that k < 0. -/
theorem linear_function_negative_slope (k b m n : ℝ) 
  (h1 : k ≠ 0)
  (h2 : m < n)
  (h3 : m = k + b)  -- Point A(1, m) satisfies the equation
  (h4 : n = -k + b) -- Point B(-1, n) satisfies the equation
  : k < 0 := by
  sorry

end linear_function_negative_slope_l2839_283975
