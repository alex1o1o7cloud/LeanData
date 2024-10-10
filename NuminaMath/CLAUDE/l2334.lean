import Mathlib

namespace sin_graph_shift_l2334_233493

/-- Shifting the graph of y = sin(1/2x - π/6) to the left by π/3 units results in the graph of y = sin(1/2x) -/
theorem sin_graph_shift (x : ℝ) : 
  Real.sin (1/2 * (x + π/3) - π/6) = Real.sin (1/2 * x) := by sorry

end sin_graph_shift_l2334_233493


namespace gcd_problem_l2334_233423

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 2027 * k) :
  Nat.gcd (Int.natAbs (b^2 + 7*b + 18)) (Int.natAbs (b + 6)) = 12 := by
  sorry

end gcd_problem_l2334_233423


namespace bd_range_l2334_233431

/-- Represents a quadrilateral ABCD with side lengths and diagonal BD --/
structure Quadrilateral :=
  (AB : ℝ)
  (BC : ℝ)
  (CD : ℝ)
  (DA : ℝ)
  (BD : ℤ)

/-- The specific quadrilateral from the problem --/
def problem_quadrilateral : Quadrilateral :=
  { AB := 7
  , BC := 15
  , CD := 7
  , DA := 11
  , BD := 0 }  -- BD is initially set to 0, but will be constrained later

theorem bd_range (q : Quadrilateral) (h : q = problem_quadrilateral) :
  9 ≤ q.BD ∧ q.BD ≤ 17 := by
  sorry

#check bd_range

end bd_range_l2334_233431


namespace polynomial_division_theorem_l2334_233437

/-- A polynomial of degree 103 with real coefficients -/
def poly (C D : ℝ) (x : ℂ) : ℂ := x^103 + C*x + D

/-- The quadratic polynomial x^2 - x + 1 -/
def quad (x : ℂ) : ℂ := x^2 - x + 1

theorem polynomial_division_theorem (C D : ℝ) :
  (∀ x : ℂ, quad x = 0 → poly C D x = 0) →
  C + D = -1 := by sorry

end polynomial_division_theorem_l2334_233437


namespace necessary_but_not_sufficient_l2334_233471

theorem necessary_but_not_sufficient :
  (∀ x : ℝ, x > 1/2 → 1/x < 2) ∧
  (∃ x : ℝ, 1/x < 2 ∧ x ≤ 1/2) :=
by sorry

end necessary_but_not_sufficient_l2334_233471


namespace sequence_inequality_l2334_233462

/-- A sequence satisfying the given conditions -/
def SequenceSatisfyingConditions (a : ℕ → ℝ) : Prop :=
  (∀ n, a n ≥ 0) ∧ 
  (∀ m n : ℕ, a (m + n) ≤ a m + a n)

/-- The main theorem to be proved -/
theorem sequence_inequality (a : ℕ → ℝ) (h : SequenceSatisfyingConditions a) :
    ∀ m n : ℕ, n ≥ m → a n ≤ m * a 1 + (n / m - 1) * a m := by
  sorry


end sequence_inequality_l2334_233462


namespace possible_x_values_l2334_233444

theorem possible_x_values (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1 / y = 10) (h2 : y + 1 / x = 5 / 12) :
  x = 4 ∨ x = 6 := by
sorry

end possible_x_values_l2334_233444


namespace alpo4_molecular_weight_l2334_233452

/-- The atomic weight of Aluminum in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of Phosphorus in g/mol -/
def atomic_weight_P : ℝ := 30.97

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The molecular weight of AlPO4 in g/mol -/
def molecular_weight_AlPO4 : ℝ :=
  atomic_weight_Al + atomic_weight_P + 4 * atomic_weight_O

/-- Theorem stating that the molecular weight of AlPO4 is 121.95 g/mol -/
theorem alpo4_molecular_weight :
  molecular_weight_AlPO4 = 121.95 := by sorry

end alpo4_molecular_weight_l2334_233452


namespace unique_ten_digit_square_match_l2334_233455

theorem unique_ten_digit_square_match : 
  ∃! (N : ℕ), 
    (10^9 ≤ N) ∧ (N < 10^10) ∧ 
    (∃ (K : ℕ), N^2 = 10^10 * K + N) ∧
    N = 10^9 := by
  sorry

end unique_ten_digit_square_match_l2334_233455


namespace magnitude_of_vector_l2334_233481

/-- Given two unit vectors e₁ and e₂ on a plane with an angle of 60° between them,
    and a vector OP = 3e₁ + 2e₂, prove that the magnitude of OP is √19. -/
theorem magnitude_of_vector (e₁ e₂ : ℝ × ℝ) : 
  (e₁.1^2 + e₁.2^2 = 1) →  -- e₁ is a unit vector
  (e₂.1^2 + e₂.2^2 = 1) →  -- e₂ is a unit vector
  (e₁.1 * e₂.1 + e₁.2 * e₂.2 = 1/2) →  -- angle between e₁ and e₂ is 60°
  let OP := (3 * e₁.1 + 2 * e₂.1, 3 * e₁.2 + 2 * e₂.2)
  (OP.1^2 + OP.2^2 = 19) :=
by sorry

end magnitude_of_vector_l2334_233481


namespace least_b_proof_l2334_233464

/-- The number of factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- The least possible value of b given the conditions -/
def least_b : ℕ := 12

theorem least_b_proof (a b : ℕ+) 
  (ha : num_factors a = 4) 
  (hb : num_factors b = a.val) 
  (hdiv : a ∣ b) : 
  ∀ c : ℕ+, 
    (num_factors c = a.val) → 
    (a ∣ c) → 
    least_b ≤ c.val :=
by sorry

end least_b_proof_l2334_233464


namespace cruise_ship_theorem_l2334_233409

def cruise_ship_problem (min_capacity max_capacity current_passengers : ℕ) : ℕ :=
  if min_capacity ≤ current_passengers then 0
  else min_capacity - current_passengers

theorem cruise_ship_theorem :
  cruise_ship_problem 16 30 9 = 7 := by
  sorry

end cruise_ship_theorem_l2334_233409


namespace folded_paper_distance_l2334_233404

/-- Given a square sheet of paper with area 12 cm², prove that when folded so
    that a corner point B rests on the diagonal and the visible red area equals
    the visible blue area, the distance from B to its original position is 4 cm. -/
theorem folded_paper_distance (sheet_area : ℝ) (fold_length : ℝ) :
  sheet_area = 12 →
  fold_length^2 / 2 = sheet_area - fold_length^2 →
  Real.sqrt (2 * fold_length^2) = 4 :=
by sorry

end folded_paper_distance_l2334_233404


namespace parallel_vectors_m_value_l2334_233459

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (m, -1)
  parallel a b → m = -2 := by
  sorry

end parallel_vectors_m_value_l2334_233459


namespace distinct_sequences_is_256_l2334_233486

/-- Represents the number of coin flips -/
def total_flips : ℕ := 10

/-- Represents the number of fixed heads at the start -/
def fixed_heads : ℕ := 2

/-- Represents the number of possible outcomes for each flip -/
def outcomes_per_flip : ℕ := 2

/-- Calculates the number of distinct sequences -/
def distinct_sequences : ℕ := outcomes_per_flip ^ (total_flips - fixed_heads)

/-- Theorem stating that the number of distinct sequences is 256 -/
theorem distinct_sequences_is_256 : distinct_sequences = 256 := by
  sorry

end distinct_sequences_is_256_l2334_233486


namespace sin_product_equality_l2334_233419

theorem sin_product_equality : 
  Real.sin (18 * π / 180) * Real.sin (30 * π / 180) * Real.sin (60 * π / 180) * Real.sin (72 * π / 180) = 
  (Real.sqrt 3 / 8) * Real.sin (36 * π / 180) := by
  sorry

end sin_product_equality_l2334_233419


namespace unique_solution_l2334_233469

theorem unique_solution (a b c d : ℝ) 
  (sum_eq : a + b + c + d = 20)
  (sum_prod_eq : a*b + a*c + a*d + b*c + b*d + c*d = 150) :
  a = 5 ∧ b = 5 ∧ c = 5 ∧ d = 5 := by
  sorry

end unique_solution_l2334_233469


namespace largest_non_sum_of_composites_l2334_233451

/-- A natural number is composite if it has more than two divisors -/
def IsComposite (n : ℕ) : Prop :=
  ∃ k m : ℕ, k > 1 ∧ m > 1 ∧ k * m = n

/-- A natural number can be expressed as the sum of two composite numbers -/
def IsSumOfTwoComposites (n : ℕ) : Prop :=
  ∃ a b : ℕ, IsComposite a ∧ IsComposite b ∧ a + b = n

/-- 11 is the largest natural number that cannot be expressed as the sum of two composite numbers -/
theorem largest_non_sum_of_composites :
  (∀ n : ℕ, n > 11 → IsSumOfTwoComposites n) ∧
  ¬IsSumOfTwoComposites 11 :=
sorry

end largest_non_sum_of_composites_l2334_233451


namespace ian_roses_problem_l2334_233408

theorem ian_roses_problem (initial_roses : ℕ) : 
  initial_roses = 6 + 9 + 4 + 1 → initial_roses = 20 :=
by
  sorry

end ian_roses_problem_l2334_233408


namespace bus_problem_solution_l2334_233447

def bus_problem (initial : ℕ) (stop1_off : ℕ) (stop2_off stop2_on : ℕ) (stop3_off stop3_on : ℕ) : ℕ :=
  initial - stop1_off - stop2_off + stop2_on - stop3_off + stop3_on

theorem bus_problem_solution :
  bus_problem 50 15 8 2 4 3 = 28 := by
  sorry

end bus_problem_solution_l2334_233447


namespace circle_equation_l2334_233432

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the line
def Line (a b c : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}

-- Define tangency
def IsTangent (c : Circle) (l : Set (ℝ × ℝ)) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ l ∧ (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Theorem statement
theorem circle_equation (c : Circle) 
  (h1 : c.radius = 2)
  (h2 : c.center.2 = 0 ∧ c.center.1 > 0)
  (h3 : IsTangent c (Line 3 4 4)) :
  ∀ (x y : ℝ), (x - c.center.1)^2 + y^2 = 4 ↔ (x, y) ∈ {p : ℝ × ℝ | (p.1 - 2)^2 + p.2^2 = 4} :=
by sorry

end circle_equation_l2334_233432


namespace racket_purchase_cost_l2334_233412

/-- Calculates the total cost of two rackets with given discounts and sales tax -/
def totalCost (originalPrice : ℝ) (discount1 : ℝ) (discount2 : ℝ) (salesTax : ℝ) : ℝ :=
  let price1 := originalPrice * (1 - discount1)
  let price2 := originalPrice * (1 - discount2)
  let subtotal := price1 + price2
  subtotal * (1 + salesTax)

/-- Theorem stating the total cost of two rackets under specific conditions -/
theorem racket_purchase_cost :
  totalCost 60 0.2 0.5 0.05 = 81.90 := by
  sorry

end racket_purchase_cost_l2334_233412


namespace gnuff_tutoring_time_l2334_233484

/-- Calculates the number of minutes tutored given the flat rate, per-minute rate, and total amount paid. -/
def minutes_tutored (flat_rate : ℚ) (per_minute_rate : ℚ) (total_paid : ℚ) : ℚ :=
  (total_paid - flat_rate) / per_minute_rate

/-- Proves that Gnuff tutored for 18 minutes given the specified rates and total amount paid. -/
theorem gnuff_tutoring_time :
  let flat_rate : ℚ := 20
  let per_minute_rate : ℚ := 7
  let total_paid : ℚ := 146
  minutes_tutored flat_rate per_minute_rate total_paid = 18 := by
  sorry

end gnuff_tutoring_time_l2334_233484


namespace zero_point_implies_a_range_l2334_233496

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 1

-- State the theorem
theorem zero_point_implies_a_range (a : ℝ) :
  (∃ x : ℝ, 1/2 < x ∧ x < 4 ∧ f a x = 0) → 2 ≤ a ∧ a < 17/4 := by
  sorry

end zero_point_implies_a_range_l2334_233496


namespace library_visitors_average_l2334_233428

def average_visitors (total_visitors : ℕ) (days : ℕ) : ℚ :=
  (total_visitors : ℚ) / (days : ℚ)

theorem library_visitors_average (
  sunday_visitors : ℕ)
  (weekday_visitors : ℕ)
  (weekend_visitors : ℕ)
  (special_event_visitors : ℕ)
  (h1 : sunday_visitors = 660)
  (h2 : weekday_visitors = 280)
  (h3 : weekend_visitors = 350)
  (h4 : special_event_visitors = 120)
  : average_visitors (
    4 * sunday_visitors +
    17 * weekday_visitors +
    8 * weekend_visitors +
    special_event_visitors
  ) 30 = 344 := by
  sorry

#eval average_visitors (
  4 * 660 +
  17 * 280 +
  8 * 350 +
  120
) 30

end library_visitors_average_l2334_233428


namespace count_two_digits_eq_655_l2334_233475

/-- A function that checks if a positive integer contains the digit '2' in its base-ten representation. -/
def containsTwo (n : ℕ+) : Bool :=
  sorry

/-- The count of positive integers less than or equal to 2537 that contain the digit '2'. -/
def countTwoDigits : ℕ :=
  sorry

/-- Theorem stating that the count of positive integers less than or equal to 2537
    containing the digit '2' is equal to 655. -/
theorem count_two_digits_eq_655 : countTwoDigits = 655 := by
  sorry

end count_two_digits_eq_655_l2334_233475


namespace bagel_shop_benches_l2334_233497

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (n : ℕ) : ℕ := sorry

/-- The problem statement -/
theorem bagel_shop_benches :
  let seating_capacity_base7 : ℕ := 321
  let people_per_bench : ℕ := 3
  let num_benches : ℕ := (base7ToBase10 seating_capacity_base7) / people_per_bench
  num_benches = 54 := by sorry

end bagel_shop_benches_l2334_233497


namespace digit2List_998_999_1000_l2334_233483

/-- A list of increasing positive integers starting with 2 and containing all numbers with a first digit of 2 -/
def digit2List : List ℕ := sorry

/-- The function that extracts the nth digit from the digit2List -/
def nthDigit (n : ℕ) : ℕ := sorry

/-- Theorem stating that the 998th, 999th, and 1000th digits in digit2List form the number 216 -/
theorem digit2List_998_999_1000 : 
  nthDigit 998 = 2 ∧ nthDigit 999 = 1 ∧ nthDigit 1000 = 6 := by sorry

end digit2List_998_999_1000_l2334_233483


namespace kids_difference_l2334_233476

/-- The number of kids Julia played with on Monday -/
def monday_kids : ℕ := 18

/-- The number of kids Julia played with on Tuesday -/
def tuesday_kids : ℕ := 10

/-- The difference in the number of kids Julia played with between Monday and Tuesday -/
def difference : ℕ := monday_kids - tuesday_kids

theorem kids_difference : difference = 8 := by
  sorry

end kids_difference_l2334_233476


namespace smallest_three_digit_multiple_of_17_l2334_233445

theorem smallest_three_digit_multiple_of_17 :
  (∀ n : ℕ, n ≥ 100 ∧ n < 102 → ¬(17 ∣ n)) ∧ (17 ∣ 102) := by
  sorry

end smallest_three_digit_multiple_of_17_l2334_233445


namespace max_value_of_a_l2334_233429

theorem max_value_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc (1/2 : ℝ) 2, a ≤ (1 - x) / x + Real.log x) → 
  a ≤ 0 := by
  sorry

end max_value_of_a_l2334_233429


namespace theresa_shared_crayons_l2334_233400

/-- Represents the number of crayons Theresa has initially -/
def initial_crayons : ℕ := 32

/-- Represents the number of crayons Theresa has after sharing -/
def final_crayons : ℕ := 19

/-- Represents the number of crayons Theresa shared -/
def shared_crayons : ℕ := initial_crayons - final_crayons

theorem theresa_shared_crayons : 
  shared_crayons = initial_crayons - final_crayons :=
by sorry

end theresa_shared_crayons_l2334_233400


namespace total_distance_traveled_l2334_233470

/-- Proves that the total distance traveled is 900 kilometers given the specified conditions -/
theorem total_distance_traveled (D : ℝ) : 
  (D / 3 : ℝ) + (2 / 3 * 360 : ℝ) + 360 = D → D = 900 := by
  sorry

end total_distance_traveled_l2334_233470


namespace polynomial_evaluation_l2334_233430

theorem polynomial_evaluation (x : ℝ) (h1 : x > 0) (h2 : x^2 - 3*x - 9 = 0) :
  x^4 - 3*x^3 - 9*x^2 + 27*x - 8 = (65 + 81*Real.sqrt 5) / 2 := by
  sorry

end polynomial_evaluation_l2334_233430


namespace largest_x_value_l2334_233413

theorem largest_x_value : ∃ (x : ℝ), 
  (∀ (z : ℝ), (|z - 3| = 8 ∧ 2*z + 1 ≤ 25) → z ≤ x) ∧ 
  |x - 3| = 8 ∧ 
  2*x + 1 ≤ 25 ∧
  x = 11 := by
sorry

end largest_x_value_l2334_233413


namespace function_properties_l2334_233478

/-- Given real numbers b and c, and a function f(x) = x^2 + bx + c that satisfies
    f(sin α) ≥ 0 and f(2 + cos β) ≤ 0 for any α, β ∈ ℝ, prove that f(1) = 0 and c ≥ 3 -/
theorem function_properties (b c : ℝ) 
    (f : ℝ → ℝ) 
    (f_def : ∀ x, f x = x^2 + b*x + c)
    (f_sin_nonneg : ∀ α, f (Real.sin α) ≥ 0)
    (f_cos_nonpos : ∀ β, f (2 + Real.cos β) ≤ 0) : 
  f 1 = 0 ∧ c ≥ 3 := by
  sorry

end function_properties_l2334_233478


namespace correct_arrangement_count_l2334_233425

/-- The number of ways to arrange 3 teachers and 3 students in a row, such that no two students are adjacent -/
def arrangementCount : ℕ := 144

/-- The number of teachers -/
def teacherCount : ℕ := 3

/-- The number of students -/
def studentCount : ℕ := 3

/-- The number of spots available for students (always one more than the number of teachers) -/
def studentSpots : ℕ := teacherCount + 1

theorem correct_arrangement_count :
  arrangementCount = 
    (Nat.factorial teacherCount) * 
    (Nat.choose studentSpots studentCount) * 
    (Nat.factorial studentCount) :=
by sorry

end correct_arrangement_count_l2334_233425


namespace divisibility_by_five_l2334_233427

theorem divisibility_by_five (a b : ℕ) : 
  (5 ∣ a ∨ 5 ∣ b) → (5 ∣ a * b) := by
  sorry

end divisibility_by_five_l2334_233427


namespace inequality_solution_l2334_233454

theorem inequality_solution (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (a > 1 → (∀ x : ℝ, a^(2*x - 1) > (1/a)^(x - 2) ↔ x > 1)) ∧
  (0 < a ∧ a < 1 → (∀ x : ℝ, a^(2*x - 1) > (1/a)^(x - 2) ↔ x < 1)) := by
  sorry

end inequality_solution_l2334_233454


namespace max_sum_xyz_l2334_233489

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorial (x : ℕ) : ℕ := (List.range x).map factorial |>.sum

theorem max_sum_xyz (x y z : ℕ) (hx : x > 1) (hy : y > 1) (hz : z > 1)
  (h_eq : sum_factorial x = y^2) :
  x + y + z ≤ 8 := by sorry

end max_sum_xyz_l2334_233489


namespace point_b_coordinates_l2334_233473

/-- A line segment parallel to the x-axis -/
structure ParallelSegment where
  A : ℝ × ℝ  -- Coordinates of point A
  B : ℝ × ℝ  -- Coordinates of point B
  length : ℝ  -- Length of the segment
  parallel_to_x : B.2 = A.2  -- y-coordinates are equal

/-- The theorem statement -/
theorem point_b_coordinates (seg : ParallelSegment) 
  (h1 : seg.A = (3, 2))  -- Point A is at (3, 2)
  (h2 : seg.length = 3)  -- Length of AB is 3
  : seg.B = (0, 2) ∨ seg.B = (6, 2) := by
  sorry

end point_b_coordinates_l2334_233473


namespace min_sum_squares_l2334_233424

theorem min_sum_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 2) :
  ∃ (m : ℝ), m = 3 ∧ ∀ (a b c : ℝ), a^3 + b^3 + c^3 - 3*a*b*c = 2 → x^2 + y^2 + z^2 ≥ m ∧ a^2 + b^2 + c^2 ≥ m :=
by sorry

end min_sum_squares_l2334_233424


namespace brian_oranges_l2334_233418

theorem brian_oranges (someone_oranges : ℕ) (brian_difference : ℕ) 
  (h1 : someone_oranges = 12)
  (h2 : brian_difference = 0) : 
  someone_oranges - brian_difference = 12 := by
  sorry

end brian_oranges_l2334_233418


namespace angle_terminal_side_point_l2334_233463

theorem angle_terminal_side_point (α : Real) (a : Real) :
  (∃ (x y : Real), x = a ∧ y = -1 ∧ (Real.tan α) * x = y) →
  Real.tan α = -1/2 →
  a = 2 := by
sorry

end angle_terminal_side_point_l2334_233463


namespace cone_base_radius_l2334_233439

/-- A right circular cone with given volume and height has a specific base radius -/
theorem cone_base_radius (V : ℝ) (h : ℝ) (r : ℝ) : 
  V = 24 * Real.pi ∧ h = 6 → V = (1/3) * Real.pi * r^2 * h → r = 2 * Real.sqrt 3 := by
  sorry

end cone_base_radius_l2334_233439


namespace factors_multiple_of_300_eq_1320_l2334_233491

/-- The number of natural-number factors of 2^12 * 3^15 * 5^9 that are multiples of 300 -/
def factors_multiple_of_300 : ℕ :=
  (12 - 2 + 1) * (15 - 1 + 1) * (9 - 2 + 1)

/-- Theorem stating that the number of natural-number factors of 2^12 * 3^15 * 5^9
    that are multiples of 300 is equal to 1320 -/
theorem factors_multiple_of_300_eq_1320 :
  factors_multiple_of_300 = 1320 := by
  sorry

end factors_multiple_of_300_eq_1320_l2334_233491


namespace min_t_value_l2334_233401

def f (x : ℝ) := x^3 - 3*x - 1

theorem min_t_value (t : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc (-3) 2 → x₂ ∈ Set.Icc (-3) 2 → |f x₁ - f x₂| ≤ t) ↔ t ≥ 20 :=
by sorry

end min_t_value_l2334_233401


namespace sum_distinct_f_values_l2334_233440

def f (x : ℤ) : ℤ := x^2 - 4*x + 100

def sum_distinct_values : ℕ → ℤ
  | 0 => 0
  | n + 1 => sum_distinct_values n + f (n + 1)

theorem sum_distinct_f_values : 
  sum_distinct_values 100 - f 1 = 328053 := by sorry

end sum_distinct_f_values_l2334_233440


namespace remainder_problem_l2334_233460

theorem remainder_problem (N : ℤ) (k : ℤ) (h : N = 35 * k + 25) :
  N % 15 = 10 := by
  sorry

end remainder_problem_l2334_233460


namespace calvin_chips_days_l2334_233482

/-- The number of days per week Calvin buys chips -/
def days_per_week : ℕ := sorry

/-- The cost of one pack of chips in dollars -/
def cost_per_pack : ℚ := 1/2

/-- The number of weeks Calvin has been buying chips -/
def num_weeks : ℕ := 4

/-- The total amount Calvin has spent on chips in dollars -/
def total_spent : ℚ := 10

theorem calvin_chips_days :
  days_per_week * num_weeks * cost_per_pack = total_spent ∧
  days_per_week = 5 := by sorry

end calvin_chips_days_l2334_233482


namespace unique_operation_assignment_l2334_233442

-- Define the possible operations
inductive Operation
| Division
| Equal
| Multiplication
| Addition
| Subtraction

-- Define a function to apply an operation
def apply_operation (op : Operation) (x y : ℕ) : Prop :=
  match op with
  | Operation.Division => x / y = 2
  | Operation.Equal => x = y
  | Operation.Multiplication => x * y = 8
  | Operation.Addition => x + y = 5
  | Operation.Subtraction => x - y = 4

-- Define the theorem
theorem unique_operation_assignment :
  ∃! (A B C D E : Operation),
    (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧
     B ≠ C ∧ B ≠ D ∧ B ≠ E ∧
     C ≠ D ∧ C ≠ E ∧
     D ≠ E) ∧
    apply_operation A 4 2 ∧
    apply_operation B 2 2 ∧
    apply_operation B 8 (4 * 2) ∧
    apply_operation C 4 2 ∧
    apply_operation D 2 3 ∧
    apply_operation B 5 5 ∧
    apply_operation B 4 (5 - 1) ∧
    apply_operation E 5 1 :=
sorry

end unique_operation_assignment_l2334_233442


namespace basic_computer_price_l2334_233495

/-- The price of the basic computer and printer total $2,500, and the printer costs 1/6 of the total when paired with an enhanced computer $500 more expensive than the basic one. -/
theorem basic_computer_price (basic_price printer_price : ℝ) 
  (h1 : basic_price + printer_price = 2500)
  (h2 : printer_price = (1 / 6) * ((basic_price + 500) + printer_price)) :
  basic_price = 2000 := by sorry

end basic_computer_price_l2334_233495


namespace complex_equation_solution_l2334_233468

theorem complex_equation_solution (z : ℂ) : (z - 2*I = 3 + 7*I) → z = 3 + 9*I := by
  sorry

end complex_equation_solution_l2334_233468


namespace arithmetic_sequence_sixth_term_l2334_233438

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sixth_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 3 + a 5 = 12)
  (h_second : a 2 = 3) :
  a 6 = 9 := by
sorry

end arithmetic_sequence_sixth_term_l2334_233438


namespace brick_width_calculation_l2334_233450

/-- Represents the dimensions of a brick in centimeters -/
structure BrickDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the dimensions of a wall in centimeters -/
structure WallDimensions where
  length : ℝ
  height : ℝ
  thickness : ℝ

/-- Calculates the volume of a brick given its dimensions -/
def brickVolume (b : BrickDimensions) : ℝ :=
  b.length * b.width * b.height

/-- Calculates the volume of a wall given its dimensions -/
def wallVolume (w : WallDimensions) : ℝ :=
  w.length * w.height * w.thickness

theorem brick_width_calculation (brick : BrickDimensions) (wall : WallDimensions) 
    (h1 : brick.length = 25)
    (h2 : brick.height = 6)
    (h3 : wall.length = 800)
    (h4 : wall.height = 600)
    (h5 : wall.thickness = 22.5)
    (h6 : (6400 : ℝ) * brickVolume brick = wallVolume wall) :
    brick.width = 11.25 := by
  sorry

end brick_width_calculation_l2334_233450


namespace min_workers_for_profit_l2334_233479

/-- Proves the minimum number of workers needed for profit --/
theorem min_workers_for_profit :
  let daily_maintenance : ℝ := 600
  let hourly_wage : ℝ := 20
  let widgets_per_hour : ℝ := 6
  let price_per_widget : ℝ := 3.50
  let work_hours : ℝ := 9

  let cost (n : ℝ) := daily_maintenance + hourly_wage * work_hours * n
  let revenue (n : ℝ) := price_per_widget * widgets_per_hour * work_hours * n

  ∀ n : ℕ, (n ≥ 67 ↔ revenue n > cost n) :=
by
  sorry

#check min_workers_for_profit

end min_workers_for_profit_l2334_233479


namespace group_total_sum_l2334_233410

/-- The total sum spent by a group of friends on dinner and a gift -/
def total_sum (num_friends : ℕ) (additional_payment : ℚ) (gift_cost : ℚ) : ℚ :=
  let dinner_cost := (num_friends - 1) * additional_payment * 10
  dinner_cost + gift_cost

/-- Proof of the total sum spent by the group -/
theorem group_total_sum :
  total_sum 10 3 15 = 285 := by
  sorry

end group_total_sum_l2334_233410


namespace spoon_set_count_l2334_233415

/-- 
Given a set of spoons costing $21, where 5 spoons would cost $15 if sold separately,
prove that the number of spoons in the set is 7.
-/
theorem spoon_set_count (total_cost : ℚ) (five_spoon_cost : ℚ) (spoon_count : ℕ) : 
  total_cost = 21 →
  five_spoon_cost = 15 →
  (5 : ℚ) * (total_cost / (spoon_count : ℚ)) = five_spoon_cost →
  spoon_count = 7 := by
  sorry

#check spoon_set_count

end spoon_set_count_l2334_233415


namespace probability_at_least_two_red_is_half_l2334_233405

def total_balls : ℕ := 6
def red_balls : ℕ := 3
def white_balls : ℕ := 2
def black_balls : ℕ := 1
def drawn_balls : ℕ := 3

def probability_at_least_two_red : ℚ :=
  (Nat.choose red_balls drawn_balls + 
   Nat.choose red_balls (drawn_balls - 1) * Nat.choose (white_balls + black_balls) 1) / 
  Nat.choose total_balls drawn_balls

theorem probability_at_least_two_red_is_half :
  probability_at_least_two_red = 1 / 2 := by
  sorry

end probability_at_least_two_red_is_half_l2334_233405


namespace ellipse_and_circle_properties_l2334_233461

/-- An ellipse with the given conditions -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h_ab : a > b
  h_b_pos : b > 0
  h_chord : 2 * b^2 / a = 3
  h_foci : 2 * c = a
  h_arithmetic : ∀ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1 →
    ∃ (k : ℝ), (x + c)^2 + y^2 = k^2 ∧
               4 * c^2 = (k + 1)^2 ∧
               (x - c)^2 + y^2 = (k + 2)^2

/-- The theorem to be proved -/
theorem ellipse_and_circle_properties (E : Ellipse) :
  (E.a = 2 ∧ E.b = Real.sqrt 3) ∧
  ∃ (r : ℝ), r^2 = 12/7 ∧
    ∀ (k m : ℝ),
      (∀ (x y : ℝ), y = k*x + m → x^2 + y^2 = r^2) →
      ∃ (x₁ y₁ x₂ y₂ : ℝ),
        x₁^2/E.a^2 + y₁^2/E.b^2 = 1 ∧
        x₂^2/E.a^2 + y₂^2/E.b^2 = 1 ∧
        y₁ = k*x₁ + m ∧
        y₂ = k*x₂ + m ∧
        x₁*x₂ + y₁*y₂ = 0 :=
by sorry

end ellipse_and_circle_properties_l2334_233461


namespace find_a1_l2334_233435

def recurrence (a : ℕ → ℚ) : Prop :=
  ∀ n, a (n + 1) = a n / (2 * a n + 1)

theorem find_a1 (a : ℕ → ℚ) (h1 : recurrence a) (h2 : a 3 = 1/5) :
  a 1 = 1 := by
  sorry

end find_a1_l2334_233435


namespace money_division_l2334_233490

theorem money_division (a b c : ℝ) 
  (h1 : a = (1/3) * (b + c))
  (h2 : b = (2/7) * (a + c))
  (h3 : a = b + 15) :
  a + b + c = 540 := by
sorry

end money_division_l2334_233490


namespace complement_of_M_in_U_l2334_233441

def U : Set ℕ := {1, 3, 5, 7}
def M : Set ℕ := {1, 3}

theorem complement_of_M_in_U :
  (U \ M) = {5, 7} := by sorry

end complement_of_M_in_U_l2334_233441


namespace heracles_age_l2334_233422

theorem heracles_age (heracles_age audrey_age : ℕ) : 
  audrey_age = heracles_age + 7 →
  audrey_age + 3 = 2 * heracles_age →
  heracles_age = 10 := by
sorry

end heracles_age_l2334_233422


namespace limit_f_zero_l2334_233433

-- Define the function f
noncomputable def f (x y : ℝ) : ℝ :=
  if x^2 + y^2 ≠ 0 then x * Real.sin (1 / y) + y * Real.sin (1 / x)
  else 0

-- State the theorem
theorem limit_f_zero :
  ∀ ε > 0, ∃ δ > 0, ∀ x y : ℝ, Real.sqrt (x^2 + y^2) < δ → |f x y| < ε :=
sorry

end limit_f_zero_l2334_233433


namespace perfect_square_with_conditions_l2334_233457

theorem perfect_square_with_conditions : ∃ (N : ℕ), ∃ (K : ℕ), ∃ (X : ℕ), 
  N = K^2 ∧ 
  K % 20 = 5 ∧ 
  K % 21 = 3 ∧ 
  1000 ≤ X ∧ X < 10000 ∧
  N = X - (X / 1000 + (X / 100 % 10) + (X / 10 % 10) + (X % 10)) ∧
  N = 2025 := by
sorry

end perfect_square_with_conditions_l2334_233457


namespace unique_m_exists_l2334_233498

/-- Given a positive integer m, converts the hexadecimal number Im05 to decimal --/
def hex_to_decimal (m : ℕ+) : ℕ :=
  16 * 16 * 16 * m.val + 16 * 16 * 13 + 16 * 0 + 5

/-- Theorem stating that there exists a unique positive integer m 
    such that Im05 in hexadecimal equals 293 in decimal --/
theorem unique_m_exists : ∃! (m : ℕ+), hex_to_decimal m = 293 := by
  sorry

end unique_m_exists_l2334_233498


namespace prom_dancers_l2334_233472

theorem prom_dancers (total_kids : ℕ) (slow_dancers : ℕ) 
  (h_total : total_kids = 140)
  (h_dancers : (total_kids : ℚ) / 4 = total_kids / 4)
  (h_slow : slow_dancers = 25)
  (h_ratio : ∃ (x : ℕ), x > 0 ∧ slow_dancers = 5 * x ∧ (total_kids / 4 : ℚ) = 10 * x) :
  (total_kids / 4 : ℚ) - slow_dancers = 0 :=
sorry

end prom_dancers_l2334_233472


namespace company_growth_rate_inequality_l2334_233494

theorem company_growth_rate_inequality (p q x : ℝ) : 
  (1 + p) * (1 + q) = (1 + x)^2 → x ≤ (p + q) / 2 := by
  sorry

end company_growth_rate_inequality_l2334_233494


namespace binary_multiplication_correct_l2334_233443

/-- Represents a binary number as a list of bits (0 or 1) -/
def BinaryNumber := List Bool

/-- Converts a natural number to its binary representation -/
def toBinary (n : ℕ) : BinaryNumber :=
  sorry

/-- Converts a binary number to its decimal representation -/
def toDecimal (b : BinaryNumber) : ℕ :=
  sorry

/-- Multiplies two binary numbers -/
def binaryMultiply (a b : BinaryNumber) : BinaryNumber :=
  sorry

theorem binary_multiplication_correct :
  binaryMultiply (toBinary 13) (toBinary 7) = toBinary 91 :=
by sorry

end binary_multiplication_correct_l2334_233443


namespace harry_apples_l2334_233407

/-- Harry's apple problem -/
theorem harry_apples : ∀ (initial_apples bought_apples friends apples_per_friend : ℕ),
  initial_apples = 79 →
  bought_apples = 5 →
  friends = 7 →
  apples_per_friend = 3 →
  initial_apples + bought_apples - friends * apples_per_friend = 63 := by
  sorry

end harry_apples_l2334_233407


namespace tates_total_education_duration_l2334_233402

def normal_high_school_duration : ℕ := 4

def tates_high_school_duration : ℕ := normal_high_school_duration - 1

def tates_college_duration : ℕ := 3 * tates_high_school_duration

theorem tates_total_education_duration :
  tates_high_school_duration + tates_college_duration = 12 := by
  sorry

end tates_total_education_duration_l2334_233402


namespace train_passing_time_l2334_233456

/-- The time it takes for a train to pass a man running in the opposite direction -/
theorem train_passing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : 
  train_length = 110 →
  train_speed = 90 * (1000 / 3600) →
  man_speed = 9 * (1000 / 3600) →
  (train_length / (train_speed + man_speed)) = 4 := by
  sorry

#check train_passing_time

end train_passing_time_l2334_233456


namespace percent_of_a_l2334_233406

theorem percent_of_a (a b c : ℝ) (h1 : b = 0.35 * a) (h2 : c = 0.4 * b) : c = 0.14 * a := by
  sorry

end percent_of_a_l2334_233406


namespace gcd_12012_18018_l2334_233414

theorem gcd_12012_18018 : Nat.gcd 12012 18018 = 6006 := by
  sorry

end gcd_12012_18018_l2334_233414


namespace arithmetic_sequence_sum_l2334_233465

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  (a 1 + a 4 + a 7 = 39) →
  (a 2 + a 5 + a 8 = 33) →
  (a 3 + a 6 + a 9 = 27) :=
by sorry

end arithmetic_sequence_sum_l2334_233465


namespace angle_value_l2334_233487

theorem angle_value (θ : Real) (h1 : θ ∈ Set.Ioo 0 (2 * Real.pi)) 
  (h2 : (Real.sin 2, Real.cos 2) ∈ Set.range (λ t => (Real.sin t, Real.cos t))) :
  θ = 5 * Real.pi / 2 - 2 := by sorry

end angle_value_l2334_233487


namespace sum_of_digits_315_base_2_l2334_233436

/-- The sum of the digits in the base-2 expression of 315₁₀ is equal to 6. -/
theorem sum_of_digits_315_base_2 : 
  (Nat.digits 2 315).sum = 6 := by sorry

end sum_of_digits_315_base_2_l2334_233436


namespace equation_solution_l2334_233426

theorem equation_solution : ∃ x : ℝ, 20 * 14 + x = 20 + 14 * x ∧ x = 20 := by
  sorry

end equation_solution_l2334_233426


namespace mean_less_than_median_l2334_233466

/-- Represents the data for days missed and number of students -/
def days_missed_data : List (Nat × Nat) := [(0, 5), (1, 3), (2, 8), (3, 2), (4, 1), (5, 1)]

/-- Total number of students in the classroom -/
def total_students : Nat := 20

/-- Calculates the median number of days missed -/
def median (data : List (Nat × Nat)) (total : Nat) : ℚ :=
  sorry

/-- Calculates the mean number of days missed -/
def mean (data : List (Nat × Nat)) (total : Nat) : ℚ :=
  sorry

theorem mean_less_than_median :
  mean days_missed_data total_students = median days_missed_data total_students - 3/10 := by
  sorry

end mean_less_than_median_l2334_233466


namespace quadratic_equation_equivalence_l2334_233417

theorem quadratic_equation_equivalence :
  ∃ (m n : ℝ), (∀ x, 4 * x^2 + 8 * x - 448 = 0 ↔ (x + m)^2 = n) ∧ m = 1 ∧ n = 113 := by
  sorry

end quadratic_equation_equivalence_l2334_233417


namespace division_problem_l2334_233434

theorem division_problem (dividend quotient remainder divisor : ℕ) 
  (h1 : dividend = 271)
  (h2 : quotient = 9)
  (h3 : remainder = 1)
  (h4 : dividend = divisor * quotient + remainder) :
  divisor = 30 := by
sorry

end division_problem_l2334_233434


namespace partnership_investment_ratio_l2334_233448

/-- Partnership investment problem -/
theorem partnership_investment_ratio 
  (a b c : ℝ) -- Investments of A, B, and C
  (total_profit b_share : ℝ) -- Total profit and B's share
  (h1 : a = 3 * b) -- A invests 3 times as much as B
  (h2 : b < c) -- B invests some fraction of what C invests
  (h3 : total_profit = 8800) -- Total profit is 8800
  (h4 : b_share = 1600) -- B's share is 1600
  (h5 : b_share / total_profit = b / (a + b + c)) -- Profit distribution ratio
  : b / c = 2 / 3 := by
  sorry

end partnership_investment_ratio_l2334_233448


namespace smallest_constant_inequality_l2334_233488

theorem smallest_constant_inequality (x y z : ℝ) (h : x + y + z = -1) :
  ∃ C : ℝ, (∀ x y z : ℝ, x + y + z = -1 → |x^3 + y^3 + z^3 + 1| ≤ C * |x^5 + y^5 + z^5 + 1|) ∧
  C = 9/10 ∧
  ∀ C' : ℝ, (∀ x y z : ℝ, x + y + z = -1 → |x^3 + y^3 + z^3 + 1| ≤ C' * |x^5 + y^5 + z^5 + 1|) → C' ≥ 9/10 :=
by sorry

end smallest_constant_inequality_l2334_233488


namespace unique_point_distance_to_line_l2334_233477

/-- Given a circle C: (x - √a)² + (y - a)² = 1 where a ≥ 0, if there exists only one point P on C
    such that the distance from P to the line l: y = 2x - 6 equals √5 - 1, then a = 1. -/
theorem unique_point_distance_to_line (a : ℝ) (h1 : a ≥ 0) :
  (∃! P : ℝ × ℝ, (P.1 - Real.sqrt a)^2 + (P.2 - a)^2 = 1 ∧
    |2 * P.1 - P.2 - 6| / Real.sqrt 5 = Real.sqrt 5 - 1) →
  a = 1 := by
sorry

end unique_point_distance_to_line_l2334_233477


namespace fib_sum_equality_l2334_233480

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Sum of first n terms of Fibonacci sequence -/
def fibSum (n : ℕ) : ℕ :=
  (List.range n).map fib |>.sum

/-- Theorem: S_2016 + S_2015 - S_2014 - S_2013 = a_2018 for Fibonacci sequence -/
theorem fib_sum_equality :
  fibSum 2016 + fibSum 2015 - fibSum 2014 - fibSum 2013 = fib 2018 := by
  sorry

end fib_sum_equality_l2334_233480


namespace permutation_combination_relation_l2334_233474

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def permutations (n r : ℕ) : ℕ := factorial n / factorial (n - r)

def combinations (n r : ℕ) : ℕ := factorial n / (factorial r * factorial (n - r))

theorem permutation_combination_relation :
  ∃ k : ℕ, permutations 32 6 = k * combinations 32 6 ∧ k = 720 := by
sorry

end permutation_combination_relation_l2334_233474


namespace prism_with_27_edges_has_11_faces_l2334_233492

/-- A prism is a polyhedron with two congruent parallel faces (bases) and faces that connect the bases (lateral faces). -/
structure Prism where
  edges : ℕ

/-- The number of faces in a prism given its number of edges -/
def num_faces (p : Prism) : ℕ :=
  let base_edges := p.edges / 3
  2 + base_edges

theorem prism_with_27_edges_has_11_faces (p : Prism) (h : p.edges = 27) :
  num_faces p = 11 := by
  sorry

end prism_with_27_edges_has_11_faces_l2334_233492


namespace expected_defectives_in_sample_l2334_233453

/-- Given a population of products with some defectives, calculate the expected number of defectives in a random sample. -/
def expected_defectives (total : ℕ) (defectives : ℕ) (sample_size : ℕ) : ℚ :=
  (sample_size : ℚ) * (defectives : ℚ) / (total : ℚ)

/-- Theorem stating that the expected number of defectives in the given scenario is 10. -/
theorem expected_defectives_in_sample :
  expected_defectives 15000 1000 150 = 10 := by
  sorry

end expected_defectives_in_sample_l2334_233453


namespace ski_boat_rental_cost_l2334_233467

/-- The cost per hour to rent a ski boat -/
def ski_boat_cost_per_hour : ℝ := 40

/-- The cost to rent a sailboat per day -/
def sailboat_cost_per_day : ℝ := 60

/-- The number of hours per day the boats were rented -/
def hours_per_day : ℝ := 3

/-- The number of days the boats were rented -/
def days_rented : ℝ := 2

/-- The additional cost Aldrich paid compared to Ken -/
def additional_cost : ℝ := 120

theorem ski_boat_rental_cost : 
  ski_boat_cost_per_hour * hours_per_day * days_rented = 
  sailboat_cost_per_day * days_rented + additional_cost := by
  sorry

end ski_boat_rental_cost_l2334_233467


namespace function_range_l2334_233416

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 2*x + 2

-- Define the domain
def domain : Set ℝ := { x | -3 ≤ x ∧ x ≤ 2 }

-- State the theorem
theorem function_range :
  { y | ∃ x ∈ domain, f x = y } = { y | 1 ≤ y ∧ y ≤ 17 } := by sorry

end function_range_l2334_233416


namespace sqrt_eight_and_nine_sixteenths_l2334_233458

theorem sqrt_eight_and_nine_sixteenths (x : ℝ) : 
  x = Real.sqrt (8 + 9 / 16) → x = Real.sqrt 137 / 4 :=
by sorry

end sqrt_eight_and_nine_sixteenths_l2334_233458


namespace necessary_but_not_sufficient_condition_l2334_233499

theorem necessary_but_not_sufficient_condition (a b : ℝ) :
  (∀ a b, a + b ≠ 3 → (a ≠ 1 ∨ b ≠ 2)) ∧
  (∃ a b, (a ≠ 1 ∨ b ≠ 2) ∧ a + b = 3) := by sorry

end necessary_but_not_sufficient_condition_l2334_233499


namespace cone_volume_l2334_233411

/-- The volume of a cone with lateral surface forming a sector of radius √31 and arc length 4π -/
theorem cone_volume (r h : ℝ) : 
  r > 0 → h > 0 → 
  (h^2 + r^2 : ℝ) = 31 → 
  2 * π * r = 4 * π → 
  (1/3 : ℝ) * π * r^2 * h = 4 * Real.sqrt 3 * π := by
  sorry

end cone_volume_l2334_233411


namespace unique_number_exists_l2334_233485

theorem unique_number_exists : ∃! N : ℚ, (1 / 3) * N = 8 ∧ (1 / 8) * N = 3 := by
  sorry

end unique_number_exists_l2334_233485


namespace smallest_n_square_and_cube_l2334_233421

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3

theorem smallest_n_square_and_cube : 
  (∀ m : ℕ, m > 0 ∧ m < 100 → ¬(is_perfect_square (4*m) ∧ is_perfect_cube (5*m))) ∧ 
  (is_perfect_square (4*100) ∧ is_perfect_cube (5*100)) := by
  sorry

end smallest_n_square_and_cube_l2334_233421


namespace intersection_equals_interval_l2334_233403

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | |x - 3| < 2}
def B : Set ℝ := {x : ℝ | x ≠ 0 ∧ (x - 4) / x ≥ 0}

-- Define the interval [4, 5)
def interval : Set ℝ := {x : ℝ | 4 ≤ x ∧ x < 5}

-- Theorem statement
theorem intersection_equals_interval : A ∩ B = interval := by sorry

end intersection_equals_interval_l2334_233403


namespace normal_distribution_probability_l2334_233449

-- Define the standard normal distribution
def standard_normal (ξ : ℝ → ℝ) : Prop :=
  ∃ (μ σ : ℝ), σ > 0 ∧ ∀ x, ξ x = (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-(1/2) * ((x - μ) / σ)^2)

-- Define the probability measure
noncomputable def P (A : Set ℝ) : ℝ := sorry

-- Theorem statement
theorem normal_distribution_probability 
  (ξ : ℝ → ℝ) 
  (h1 : standard_normal ξ) 
  (h2 : P {x | ξ x > 1} = 1/4) : 
  P {x | -1 < ξ x ∧ ξ x < 1} = 1/2 := by
  sorry

end normal_distribution_probability_l2334_233449


namespace laundry_bill_calculation_l2334_233420

/-- Given a laundry bill with trousers and shirts, calculate the charge per pair of trousers. -/
theorem laundry_bill_calculation 
  (total_bill : ℝ) 
  (shirt_charge : ℝ) 
  (num_trousers : ℕ) 
  (num_shirts : ℕ) 
  (h1 : total_bill = 140)
  (h2 : shirt_charge = 5)
  (h3 : num_trousers = 10)
  (h4 : num_shirts = 10) :
  (total_bill - shirt_charge * num_shirts) / num_trousers = 9 := by
sorry

end laundry_bill_calculation_l2334_233420


namespace cone_volume_l2334_233446

/-- 
Given a cone with surface area 4π and whose unfolded side view is a sector 
with a central angle of 2π/3, prove that its volume is 2√2π/3.
-/
theorem cone_volume (r l h : ℝ) : 
  (π * r * l + π * r^2 = 4 * π) →  -- Surface area condition
  ((2 * π / 3) * l = 2 * π * r) →  -- Sector condition
  (h^2 + r^2 = l^2) →              -- Pythagorean theorem
  ((1/3) * π * r^2 * h = (2 * Real.sqrt 2 * π) / 3) := by
sorry

end cone_volume_l2334_233446
