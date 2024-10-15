import Mathlib

namespace NUMINAMATH_CALUDE_parabola_tangent_theorem_l764_76489

/-- Parabola C₁ with equation x² = 2py -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Point on the parabola -/
structure ParabolaPoint (C : Parabola) where
  x : ℝ
  y : ℝ
  hy : x^2 = 2 * C.p * y

/-- External point M -/
structure ExternalPoint (C : Parabola) where
  a : ℝ
  y : ℝ
  hy : y = -2 * C.p

/-- Theorem stating the main results -/
theorem parabola_tangent_theorem (C : Parabola) (M : ExternalPoint C) :
  -- Part 1: If a line through focus with x-intercept 2 intersects C₁ at Q and N 
  -- such that |Q'N'| = 2√5, then p = 2
  (∃ (Q N : ParabolaPoint C), 
    (Q.x / 2 + 2 * Q.y / C.p = 1) ∧ 
    (N.x / 2 + 2 * N.y / C.p = 1) ∧ 
    ((Q.x - N.x)^2 = 20)) →
  C.p = 2 ∧
  -- Part 2: If A and B are tangent points, then k₁ · k₂ = -4
  (∀ (A B : ParabolaPoint C),
    (A.y - M.y = (A.x / C.p) * (A.x - M.a)) →
    (B.y - M.y = (B.x / C.p) * (B.x - M.a)) →
    ((A.x / C.p) * (B.x / C.p) = -4)) := by
  sorry

end NUMINAMATH_CALUDE_parabola_tangent_theorem_l764_76489


namespace NUMINAMATH_CALUDE_calculate_upstream_speed_l764_76487

/-- The speed of a man rowing in a river -/
structure RowerSpeed where
  stillWater : ℝ
  downstream : ℝ
  upstream : ℝ

/-- Theorem: Given a man's speed in still water and downstream, calculate his upstream speed -/
theorem calculate_upstream_speed (s : RowerSpeed) 
  (h1 : s.stillWater = 40)
  (h2 : s.downstream = 45) :
  s.upstream = 35 := by
  sorry

end NUMINAMATH_CALUDE_calculate_upstream_speed_l764_76487


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l764_76438

theorem quadratic_distinct_roots (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (k - 2) * x₁^2 + 2 * x₁ - 1 = 0 ∧ (k - 2) * x₂^2 + 2 * x₂ - 1 = 0) ↔
  (k > 1 ∧ k ≠ 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l764_76438


namespace NUMINAMATH_CALUDE_linear_equation_rewrite_l764_76412

theorem linear_equation_rewrite (k m : ℚ) : 
  (∀ x y : ℚ, 2 * x + 3 * y - 4 = 0 ↔ y = k * x + m) → 
  k + m = 2/3 := by
sorry

end NUMINAMATH_CALUDE_linear_equation_rewrite_l764_76412


namespace NUMINAMATH_CALUDE_farmer_profit_l764_76460

/-- Calculate the profit for a group of piglets -/
def profit_for_group (num_piglets : ℕ) (months : ℕ) (price : ℕ) : ℕ :=
  num_piglets * price - num_piglets * 12 * months

/-- Calculate the total profit for all piglet groups -/
def total_profit : ℕ :=
  profit_for_group 2 12 350 +
  profit_for_group 3 15 400 +
  profit_for_group 2 18 450 +
  profit_for_group 1 21 500

/-- The farmer's profit from selling 8 piglets is $1788 -/
theorem farmer_profit : total_profit = 1788 := by
  sorry

end NUMINAMATH_CALUDE_farmer_profit_l764_76460


namespace NUMINAMATH_CALUDE_no_pairs_50_75_six_pairs_50_600_l764_76472

-- Define the function to count pairs satisfying the conditions
def countPairs (gcd : Nat) (lcm : Nat) : Nat :=
  (Finset.filter (fun p : Nat × Nat => 
    p.1.gcd p.2 = gcd ∧ p.1.lcm p.2 = lcm) (Finset.product (Finset.range (lcm + 1)) (Finset.range (lcm + 1)))).card

-- Theorem for the first part
theorem no_pairs_50_75 : countPairs 50 75 = 0 := by sorry

-- Theorem for the second part
theorem six_pairs_50_600 : countPairs 50 600 = 6 := by sorry

end NUMINAMATH_CALUDE_no_pairs_50_75_six_pairs_50_600_l764_76472


namespace NUMINAMATH_CALUDE_cubic_function_properties_l764_76484

/-- A cubic function with specific properties -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 - b * x + 4

/-- The function reaches an extreme value at x = 2 -/
def extreme_at_2 (a b : ℝ) : Prop := f a b 2 = -4/3

/-- The derivative of f is zero at x = 2 -/
def derivative_zero_at_2 (a b : ℝ) : Prop := 3 * a * 2^2 - b = 0

theorem cubic_function_properties (a b : ℝ) 
  (h1 : extreme_at_2 a b) 
  (h2 : derivative_zero_at_2 a b) :
  (∀ x, f a b x = (1/3) * x^3 - 4 * x + 4) ∧
  (∃ x ∈ Set.Icc (-3 : ℝ) 3, ∀ y ∈ Set.Icc (-3 : ℝ) 3, f a b x ≥ f a b y) ∧
  (∃ x ∈ Set.Icc (-3 : ℝ) 3, ∀ y ∈ Set.Icc (-3 : ℝ) 3, f a b x ≤ f a b y) ∧
  (∃ x ∈ Set.Icc (-3 : ℝ) 3, f a b x = 28/3) ∧
  (∃ x ∈ Set.Icc (-3 : ℝ) 3, f a b x = -4/3) :=
sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l764_76484


namespace NUMINAMATH_CALUDE_existence_of_good_subset_l764_76473

def M : Finset ℕ := Finset.range 20

def is_valid_function (f : Finset ℕ → ℕ) : Prop :=
  ∀ S : Finset ℕ, S ⊆ M → S.card = 9 → f S ∈ M

theorem existence_of_good_subset (f : Finset ℕ → ℕ) (h : is_valid_function f) :
  ∃ T : Finset ℕ, T ⊆ M ∧ T.card = 10 ∧ ∀ k ∈ T, f (T \ {k}) ≠ k := by
  sorry

#check existence_of_good_subset

end NUMINAMATH_CALUDE_existence_of_good_subset_l764_76473


namespace NUMINAMATH_CALUDE_chocolate_box_problem_l764_76483

theorem chocolate_box_problem (total : ℕ) (caramels : ℕ) (nougats : ℕ) (truffles : ℕ) (peanut_clusters : ℕ) :
  total = 50 →
  caramels = 3 →
  nougats = 2 * caramels →
  truffles = caramels + (truffles - caramels) →
  peanut_clusters = (64 * total) / 100 →
  total = caramels + nougats + truffles + peanut_clusters →
  truffles - caramels = 6 := by
sorry

end NUMINAMATH_CALUDE_chocolate_box_problem_l764_76483


namespace NUMINAMATH_CALUDE_longer_train_length_l764_76413

/-- Calculates the length of the longer train given the speeds of two trains,
    the time they take to cross each other, and the length of the shorter train. -/
theorem longer_train_length
  (speed1 speed2 : ℝ)
  (crossing_time : ℝ)
  (shorter_train_length : ℝ)
  (h1 : speed1 = 68)
  (h2 : speed2 = 40)
  (h3 : crossing_time = 11.999040076793857)
  (h4 : shorter_train_length = 160)
  (h5 : speed1 > 0)
  (h6 : speed2 > 0)
  (h7 : crossing_time > 0)
  (h8 : shorter_train_length > 0) :
  let relative_speed := (speed1 + speed2) * (1000 / 3600)
  let total_distance := relative_speed * crossing_time
  total_distance - shorter_train_length = 200 := by
  sorry

end NUMINAMATH_CALUDE_longer_train_length_l764_76413


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l764_76453

/-- Given an arithmetic sequence with first term a₁ = 3, second term a₂ = 10,
    third term a₃ = 17, and sixth term a₆ = 38, prove that a₄ + a₅ = 55. -/
theorem arithmetic_sequence_sum (a : ℕ → ℕ) :
  a 1 = 3 →
  a 2 = 10 →
  a 3 = 17 →
  a 6 = 38 →
  (∀ n : ℕ, n ≥ 1 → a (n + 1) - a n = a 2 - a 1) →
  a 4 + a 5 = 55 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l764_76453


namespace NUMINAMATH_CALUDE_consecutive_sum_theorem_l764_76477

theorem consecutive_sum_theorem (n : ℕ) (h : n ≥ 6) :
  ∃ (k a : ℕ), k ≥ 3 ∧ n = k * a + k * (k - 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_sum_theorem_l764_76477


namespace NUMINAMATH_CALUDE_factorial_ratio_equals_seven_and_half_l764_76433

theorem factorial_ratio_equals_seven_and_half :
  (Nat.factorial 10 * Nat.factorial 7 * Nat.factorial 3) / 
  (Nat.factorial 9 * Nat.factorial 8 : ℚ) = 15 / 2 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_equals_seven_and_half_l764_76433


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l764_76499

-- Problem 1
theorem problem_1 (x : ℝ) (h : x^2 + x - 2 = 0) :
  x^2 + x + 2023 = 2025 := by sorry

-- Problem 2
theorem problem_2 (a b : ℝ) (h : a + b = 5) :
  2*(a + b) - 4*a - 4*b + 21 = 11 := by sorry

-- Problem 3
theorem problem_3 (a b : ℝ) (h1 : a^2 + 3*a*b = 20) (h2 : b^2 + 5*a*b = 8) :
  2*a^2 - b^2 + a*b = 32 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l764_76499


namespace NUMINAMATH_CALUDE_car_truck_difference_l764_76480

theorem car_truck_difference (total_vehicles trucks : ℕ) 
  (h1 : total_vehicles = 69)
  (h2 : trucks = 21)
  (h3 : total_vehicles > 2 * trucks) : 
  total_vehicles - 2 * trucks = 27 := by
  sorry

end NUMINAMATH_CALUDE_car_truck_difference_l764_76480


namespace NUMINAMATH_CALUDE_loss_percentage_calculation_l764_76455

theorem loss_percentage_calculation (purchase_price selling_price : ℚ) : 
  purchase_price = 490 → 
  selling_price = 465.5 → 
  (purchase_price - selling_price) / purchase_price * 100 = 5 := by
sorry

end NUMINAMATH_CALUDE_loss_percentage_calculation_l764_76455


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_23_l764_76404

theorem greatest_three_digit_multiple_of_23 : 
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ 23 ∣ n → n ≤ 989 := by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_23_l764_76404


namespace NUMINAMATH_CALUDE_luncheon_cost_theorem_l764_76437

/-- Represents the cost of a luncheon item -/
structure LuncheonItem where
  price : ℚ

/-- Represents a luncheon order -/
structure Luncheon where
  sandwiches : ℕ
  coffee : ℕ
  pie : ℕ
  total : ℚ

/-- The theorem to be proved -/
theorem luncheon_cost_theorem (s : LuncheonItem) (c : LuncheonItem) (p : LuncheonItem) 
  (l1 : Luncheon) (l2 : Luncheon) : 
  l1.sandwiches = 2 ∧ l1.coffee = 5 ∧ l1.pie = 2 ∧ l1.total = 25/4 ∧
  l2.sandwiches = 5 ∧ l2.coffee = 8 ∧ l2.pie = 3 ∧ l2.total = 121/10 →
  s.price + c.price + p.price = 31/20 := by
  sorry

#eval 31/20  -- This should evaluate to 1.55

end NUMINAMATH_CALUDE_luncheon_cost_theorem_l764_76437


namespace NUMINAMATH_CALUDE_theresa_julia_multiple_l764_76463

/-- The number of video games Tory has -/
def tory_games : ℕ := 6

/-- The number of video games Julia has -/
def julia_games : ℕ := tory_games / 3

/-- The number of video games Theresa has -/
def theresa_games : ℕ := 11

/-- The multiple of video games Theresa has compared to Julia -/
def multiple : ℕ := (theresa_games - 5) / julia_games

theorem theresa_julia_multiple :
  multiple = 3 :=
sorry

end NUMINAMATH_CALUDE_theresa_julia_multiple_l764_76463


namespace NUMINAMATH_CALUDE_inverse_inequality_l764_76408

theorem inverse_inequality (x y : ℝ) (h1 : x > y) (h2 : y > 0) : 1/x < 1/y := by
  sorry

end NUMINAMATH_CALUDE_inverse_inequality_l764_76408


namespace NUMINAMATH_CALUDE_range_of_k_for_quadratic_inequality_l764_76417

theorem range_of_k_for_quadratic_inequality :
  {k : ℝ | ∀ x : ℝ, k * x^2 - k * x - 1 < 0} = {k : ℝ | -4 < k ∧ k ≤ 0} :=
by sorry

end NUMINAMATH_CALUDE_range_of_k_for_quadratic_inequality_l764_76417


namespace NUMINAMATH_CALUDE_number_exceeds_value_l764_76450

theorem number_exceeds_value (n : ℕ) (v : ℕ) (h : n = 69) : 
  n = v + 3 * (86 - n) → v = 18 := by
sorry

end NUMINAMATH_CALUDE_number_exceeds_value_l764_76450


namespace NUMINAMATH_CALUDE_paul_bought_45_cookies_l764_76475

/-- The number of cookies Paul bought -/
def paul_cookies : ℕ := 45

/-- The number of cookies Paula bought -/
def paula_cookies : ℕ := paul_cookies - 3

/-- The total number of cookies bought by Paul and Paula -/
def total_cookies : ℕ := 87

/-- Theorem stating that Paul bought 45 cookies given the conditions -/
theorem paul_bought_45_cookies : 
  paul_cookies = 45 ∧ 
  paula_cookies = paul_cookies - 3 ∧ 
  paul_cookies + paula_cookies = total_cookies :=
sorry

end NUMINAMATH_CALUDE_paul_bought_45_cookies_l764_76475


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l764_76426

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ, 
  3 * X^2 - 22 * X + 58 = (X - 3) * q + 19 := by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l764_76426


namespace NUMINAMATH_CALUDE_max_value_abc_l764_76419

theorem max_value_abc (a b c : ℝ) (h : a + 3*b + c = 6) :
  ∃ m : ℝ, m = 8 ∧ ∀ x y z : ℝ, x + 3*y + z = 6 → x*y + x*z + y*z ≤ m :=
sorry

end NUMINAMATH_CALUDE_max_value_abc_l764_76419


namespace NUMINAMATH_CALUDE_inequality_range_l764_76479

theorem inequality_range (m : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → x^2 - 4*x ≥ m) ↔ m ∈ Set.Iic (-3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l764_76479


namespace NUMINAMATH_CALUDE_triangle_properties_l764_76496

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  (3 * Real.cos B * Real.cos C + 1 = 3 * Real.sin B * Real.sin C + Real.cos (2 * A)) →
  (A = π / 3) ∧
  (a = 2 * Real.sqrt 3 → ∃ (max_value : ℝ), max_value = 4 * Real.sqrt 7 ∧
    ∀ (b' c' : ℝ), b' + 2 * c' ≤ max_value) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l764_76496


namespace NUMINAMATH_CALUDE_inner_square_is_square_l764_76424

/-- A point on a line segment -/
structure PointOnSegment (A B : ℝ × ℝ) where
  point : ℝ × ℝ
  on_segment : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ point = (1 - t) • A + t • B

/-- A square defined by its four vertices -/
structure Square (A B C D : ℝ × ℝ) where
  is_square : sorry  -- Definition of a square

/-- Points dividing sides of a square in the same ratio -/
def divide_sides_equally (ABCD : Square A B C D) 
  (N : PointOnSegment A B) (K : PointOnSegment B C) 
  (L : PointOnSegment C D) (M : PointOnSegment D A) : Prop :=
  sorry  -- Definition of dividing sides equally

theorem inner_square_is_square 
  (ABCD : Square A B C D)
  (N : PointOnSegment A B) (K : PointOnSegment B C) 
  (L : PointOnSegment C D) (M : PointOnSegment D A)
  (h : divide_sides_equally ABCD N K L M) :
  Square N.point K.point L.point M.point :=
sorry

end NUMINAMATH_CALUDE_inner_square_is_square_l764_76424


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l764_76439

theorem quadratic_equation_solutions :
  (∀ x : ℝ, 3 * x^2 - 6 * x - 2 = 0 ↔ x = (3 + Real.sqrt 15) / 3 ∨ x = (3 - Real.sqrt 15) / 3) ∧
  (∀ x : ℝ, x^2 + 6 * x + 8 = 0 ↔ x = -2 ∨ x = -4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l764_76439


namespace NUMINAMATH_CALUDE_negation_equivalence_l764_76486

theorem negation_equivalence : 
  (¬ ∃ (x : ℝ), x^2 - x + 1 ≤ 0) ↔ (∀ (x : ℝ), x^2 - x + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l764_76486


namespace NUMINAMATH_CALUDE_expression_equality_l764_76435

theorem expression_equality : (-1)^2023 - Real.sqrt 9 + |1 - Real.sqrt 2| - ((-8) ^ (1/3 : ℝ)) = Real.sqrt 2 - 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l764_76435


namespace NUMINAMATH_CALUDE_failed_students_l764_76456

theorem failed_students (total : ℕ) (passed_percentage : ℚ) 
  (h1 : total = 804)
  (h2 : passed_percentage = 75 / 100) :
  ↑total * (1 - passed_percentage) = 201 := by
  sorry

end NUMINAMATH_CALUDE_failed_students_l764_76456


namespace NUMINAMATH_CALUDE_ellipse_properties_l764_76451

/-- Represents an ellipse with center at the origin -/
structure Ellipse where
  /-- The equation of the ellipse in the form x²/a² + y²/b² = 1 -/
  equation : ℝ → ℝ → Prop

/-- Checks if a point (x, y) lies on the ellipse -/
def Ellipse.contains (e : Ellipse) (x y : ℝ) : Prop :=
  e.equation x y

/-- The focal distance of the ellipse -/
def Ellipse.focalDistance (e : Ellipse) : ℝ := 2

/-- Theorem about the properties of a specific ellipse -/
theorem ellipse_properties (e : Ellipse) 
    (h1 : e.focalDistance = 2)
    (h2 : e.contains (-1) (3/2)) : 
  (∃ a b : ℝ, e.equation = fun x y ↦ x^2/a^2 + y^2/b^2 = 1 ∧ a = 2 ∧ b^2 = 3) ∧
  (∀ x y : ℝ, e.contains x y ↔ x^2/4 + y^2/3 = 1) ∧
  (e.contains 2 0 ∧ e.contains (-2) 0 ∧ e.contains 0 (Real.sqrt 3) ∧ e.contains 0 (-Real.sqrt 3)) ∧
  (∃ majorAxis : ℝ, majorAxis = 4) ∧
  (∃ minorAxis : ℝ, minorAxis = 2 * Real.sqrt 3) ∧
  (∃ eccentricity : ℝ, eccentricity = 1/2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l764_76451


namespace NUMINAMATH_CALUDE_members_playing_both_l764_76495

/-- The number of members who play both badminton and tennis in a sports club -/
theorem members_playing_both (total : ℕ) (badminton : ℕ) (tennis : ℕ) (neither : ℕ)
  (h_total : total = 30)
  (h_badminton : badminton = 17)
  (h_tennis : tennis = 19)
  (h_neither : neither = 3) :
  badminton + tennis - (total - neither) = 9 := by
  sorry

end NUMINAMATH_CALUDE_members_playing_both_l764_76495


namespace NUMINAMATH_CALUDE_tourist_distribution_count_l764_76444

def num_guides : ℕ := 3
def num_tourists : ℕ := 8

theorem tourist_distribution_count :
  (3^8 : ℕ) - num_guides * (2^8 : ℕ) + (num_guides.choose 2) * (1^8 : ℕ) = 5796 :=
sorry

end NUMINAMATH_CALUDE_tourist_distribution_count_l764_76444


namespace NUMINAMATH_CALUDE_percentage_repeated_approx_l764_76400

/-- The count of five-digit numbers -/
def total_five_digit_numbers : ℕ := 90000

/-- The count of five-digit numbers without repeated digits -/
def unique_digit_numbers : ℕ := 9 * 9 * 8 * 7 * 6

/-- The count of five-digit numbers with at least one repeated digit -/
def repeated_digit_numbers : ℕ := total_five_digit_numbers - unique_digit_numbers

/-- The percentage of five-digit numbers with at least one repeated digit -/
def percentage_repeated : ℚ := (repeated_digit_numbers : ℚ) / (total_five_digit_numbers : ℚ) * 100

theorem percentage_repeated_approx :
  ∃ ε > 0, ε < 0.1 ∧ |percentage_repeated - 69.8| < ε :=
sorry

end NUMINAMATH_CALUDE_percentage_repeated_approx_l764_76400


namespace NUMINAMATH_CALUDE_unique_solution_condition_l764_76432

theorem unique_solution_condition (k : ℝ) : 
  (∃! x : ℝ, (3*x + 6)*(x - 4) = -33 + k*x) ↔ (k = -6 + 6*Real.sqrt 3 ∨ k = -6 - 6*Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l764_76432


namespace NUMINAMATH_CALUDE_mountain_distances_l764_76458

/-- Given a mountainous region with points A, B, and C, where:
    - The horizontal projection of BC is 2400 m
    - Peak B is 800 m higher than C
    - The elevation angle of AB is 20°
    - The elevation angle of AC is 2°
    - The angle between AB' and AC' (where B' and C' are horizontal projections) is 60°
    
    This theorem states that:
    - The horizontal projection of AB is approximately 2426 m
    - The horizontal projection of AC is approximately 2374 m
    - The height difference between B and A is approximately 883.2 m
-/
theorem mountain_distances (BC_proj : ℝ) (B_height_diff : ℝ) (AB_angle : ℝ) (AC_angle : ℝ) (ABC_angle : ℝ)
  (h_BC_proj : BC_proj = 2400)
  (h_B_height_diff : B_height_diff = 800)
  (h_AB_angle : AB_angle = 20 * π / 180)
  (h_AC_angle : AC_angle = 2 * π / 180)
  (h_ABC_angle : ABC_angle = 60 * π / 180) :
  ∃ (AB_proj AC_proj BA_height : ℝ),
    (abs (AB_proj - 2426) < 1) ∧
    (abs (AC_proj - 2374) < 1) ∧
    (abs (BA_height - 883.2) < 0.1) := by
  sorry

end NUMINAMATH_CALUDE_mountain_distances_l764_76458


namespace NUMINAMATH_CALUDE_binomial_divisibility_l764_76485

theorem binomial_divisibility (k : ℕ) (h : k ≥ 2) :
  ∃ m : ℤ, (Nat.choose (2^(k+1)) (2^k) - Nat.choose (2^k) (2^(k-1))) = m * 2^(3*k) ∧
  ¬∃ n : ℤ, (Nat.choose (2^(k+1)) (2^k) - Nat.choose (2^k) (2^(k-1))) = n * 2^(3*k+1) :=
sorry

end NUMINAMATH_CALUDE_binomial_divisibility_l764_76485


namespace NUMINAMATH_CALUDE_initial_bushes_count_l764_76481

/-- The number of new bushes that grow between each pair of neighboring bushes every hour. -/
def new_bushes_per_hour : ℕ := 2

/-- The total number of hours of growth. -/
def total_hours : ℕ := 3

/-- The total number of bushes after the growth period. -/
def final_bush_count : ℕ := 190

/-- Calculate the number of bushes after one hour of growth. -/
def bushes_after_one_hour (initial_bushes : ℕ) : ℕ :=
  initial_bushes + new_bushes_per_hour * (initial_bushes - 1)

/-- Calculate the number of bushes after the total growth period. -/
def bushes_after_growth (initial_bushes : ℕ) : ℕ :=
  (bushes_after_one_hour^[total_hours]) initial_bushes

/-- The theorem stating that 8 is the correct initial number of bushes. -/
theorem initial_bushes_count : 
  ∃ (n : ℕ), n > 0 ∧ bushes_after_growth n = final_bush_count ∧ 
  ∀ (m : ℕ), m ≠ n → bushes_after_growth m ≠ final_bush_count :=
sorry

end NUMINAMATH_CALUDE_initial_bushes_count_l764_76481


namespace NUMINAMATH_CALUDE_digit_sum_proof_l764_76466

theorem digit_sum_proof (P Q R : ℕ) : 
  P ∈ Finset.range 9 → 
  Q ∈ Finset.range 9 → 
  R ∈ Finset.range 9 → 
  P + P + P = 2022 → 
  P + Q + R = 15 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_proof_l764_76466


namespace NUMINAMATH_CALUDE_ball_probabilities_l764_76403

/-- Represents the color of a ball -/
inductive BallColor
  | Yellow
  | Green
  | Red

/-- Represents the box of balls -/
structure BallBox where
  total : Nat
  yellow : Nat
  green : Nat
  red : Nat

/-- Calculates the probability of drawing a ball of a specific color -/
def probability (box : BallBox) (color : BallColor) : Rat :=
  match color with
  | BallColor.Yellow => box.yellow / box.total
  | BallColor.Green => box.green / box.total
  | BallColor.Red => box.red / box.total

/-- The main theorem to prove -/
theorem ball_probabilities (box : BallBox) : 
  box.total = 10 ∧ 
  box.yellow = 1 ∧ 
  box.green = 3 ∧ 
  box.red = box.total - box.yellow - box.green →
  probability box BallColor.Green > probability box BallColor.Yellow ∧
  probability box BallColor.Red = 3/5 := by
  sorry


end NUMINAMATH_CALUDE_ball_probabilities_l764_76403


namespace NUMINAMATH_CALUDE_bill_purchase_percentage_bill_specific_problem_l764_76422

/-- The problem of determining the percentage by which Bill could have purchased a product for less -/
theorem bill_purchase_percentage (original_profit_rate : ℝ) (new_profit_rate : ℝ) 
  (original_selling_price : ℝ) (additional_profit : ℝ) : ℝ :=
  let original_cost := original_selling_price / (1 + original_profit_rate)
  let new_selling_price := original_selling_price + additional_profit
  let percentage_less := 1 - (new_selling_price / ((1 + new_profit_rate) * original_cost))
  percentage_less * 100

/-- Proof of the specific problem instance -/
theorem bill_specific_problem : 
  bill_purchase_percentage 0.1 0.3 549.9999999999995 35 = 10 := by
  sorry

end NUMINAMATH_CALUDE_bill_purchase_percentage_bill_specific_problem_l764_76422


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l764_76411

theorem simplify_trig_expression (θ : Real) (h : θ = 160 * π / 180) :
  Real.sqrt (1 - Real.sin θ ^ 2) = -Real.cos θ := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l764_76411


namespace NUMINAMATH_CALUDE_trigonometric_fraction_simplification_l764_76461

theorem trigonometric_fraction_simplification (x : ℝ) :
  (2 + 3 * Real.sin x - 4 * Real.cos x) / (2 + 3 * Real.sin x + 2 * Real.cos x) 
  = (-1 + 3 * Real.sin (x/2) * Real.cos (x/2) + 4 * (Real.sin (x/2))^2) / 
    (2 + 3 * Real.sin (x/2) * Real.cos (x/2) - 2 * (Real.sin (x/2))^2) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_fraction_simplification_l764_76461


namespace NUMINAMATH_CALUDE_rectangle_longer_side_length_l764_76497

/-- Given a circle of radius 6 cm tangent to three sides of a rectangle,
    and the area of the rectangle is three times the area of the circle,
    prove that the length of the longer side of the rectangle is 9π cm. -/
theorem rectangle_longer_side_length (r : ℝ) (circle_area rectangle_area : ℝ) :
  r = 6 →
  circle_area = π * r^2 →
  rectangle_area = 3 * circle_area →
  ∃ (shorter_side longer_side : ℝ),
    shorter_side = 2 * r ∧
    rectangle_area = shorter_side * longer_side ∧
    longer_side = 9 * π :=
by sorry

end NUMINAMATH_CALUDE_rectangle_longer_side_length_l764_76497


namespace NUMINAMATH_CALUDE_cube_remainder_mod_nine_l764_76446

theorem cube_remainder_mod_nine (n : ℤ) :
  (n % 9 = 1 ∨ n % 9 = 4 ∨ n % 9 = 7) → n^3 % 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_remainder_mod_nine_l764_76446


namespace NUMINAMATH_CALUDE_inequality_generalization_l764_76401

theorem inequality_generalization (x : ℝ) (n : ℕ) (h : x > 0) :
  x^n + n / x > n + 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_generalization_l764_76401


namespace NUMINAMATH_CALUDE_certain_number_proof_l764_76493

theorem certain_number_proof : ∃ x : ℝ, 0.60 * x = 0.45 * 30 + 16.5 ∧ x = 50 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l764_76493


namespace NUMINAMATH_CALUDE_angle_is_15_degrees_l764_76430

-- Define the triangle MIT
structure Triangle :=
  (M I T : ℝ × ℝ)

-- Define the points X, Y, O, P
structure Points :=
  (X Y O P : ℝ × ℝ)

def angle_MOP (t : Triangle) (p : Points) : ℝ :=
  sorry  -- The actual calculation of the angle

-- Main theorem
theorem angle_is_15_degrees 
  (t : Triangle) 
  (p : Points) 
  (h1 : t.M.1 = 0 ∧ t.M.2 = 0)  -- Assume M is at (0,0)
  (h2 : t.I.1 = 12 ∧ t.I.2 = 0)  -- MI = 12
  (h3 : (t.M.1 - p.X.1)^2 + (t.M.2 - p.X.2)^2 = 4)  -- MX = 2
  (h4 : (t.I.1 - p.Y.1)^2 + (t.I.2 - p.Y.2)^2 = 4)  -- YI = 2
  (h5 : p.O = ((t.M.1 + t.I.1)/2, (t.M.2 + t.I.2)/2))  -- O is midpoint of MI
  (h6 : p.P = ((p.X.1 + p.Y.1)/2, (p.X.2 + p.Y.2)/2))  -- P is midpoint of XY
  : angle_MOP t p = 15 * π / 180 :=
sorry

end NUMINAMATH_CALUDE_angle_is_15_degrees_l764_76430


namespace NUMINAMATH_CALUDE_pet_store_dogs_l764_76468

/-- Given a ratio of cats to dogs and the number of cats, calculate the number of dogs -/
def calculate_dogs (cat_ratio : ℕ) (dog_ratio : ℕ) (num_cats : ℕ) : ℕ :=
  (num_cats / cat_ratio) * dog_ratio

/-- Theorem: Given the ratio of cats to dogs as 3:5 and 18 cats, there are 30 dogs -/
theorem pet_store_dogs : calculate_dogs 3 5 18 = 30 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_dogs_l764_76468


namespace NUMINAMATH_CALUDE_price_increase_percentage_l764_76492

def original_price : ℝ := 300
def new_price : ℝ := 390

theorem price_increase_percentage :
  (new_price - original_price) / original_price * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_price_increase_percentage_l764_76492


namespace NUMINAMATH_CALUDE_siblings_age_sum_l764_76434

theorem siblings_age_sum (R D S J : ℕ) : 
  R = D + 6 →
  D = S + 8 →
  J = R - 5 →
  R + 8 = 2 * (S + 8) →
  J + 10 = (D + 10) / 2 + 4 →
  (R - 3) + (D - 3) + (S - 3) + (J - 3) = 43 :=
by sorry

end NUMINAMATH_CALUDE_siblings_age_sum_l764_76434


namespace NUMINAMATH_CALUDE_no_valid_numbers_l764_76464

def isEven (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

def digitSum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem no_valid_numbers : ¬ ∃ n : ℕ, 
  100 ≤ n ∧ n < 1000 ∧ 
  digitSum n = 27 ∧
  isEven ((n / 10) % 10) ∧
  isEven n :=
sorry

end NUMINAMATH_CALUDE_no_valid_numbers_l764_76464


namespace NUMINAMATH_CALUDE_hyperbola_condition_ellipse_x_foci_condition_l764_76476

-- Define the curve C
def C (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / (4 - k) + p.2^2 / (k - 1) = 1}

-- Define what it means for C to be a hyperbola
def is_hyperbola (k : ℝ) : Prop :=
  (4 - k) * (k - 1) < 0

-- Define what it means for C to be an ellipse with foci on the x-axis
def is_ellipse_x_foci (k : ℝ) : Prop :=
  k - 1 > 0 ∧ 4 - k > 0 ∧ 4 - k > k - 1

-- Theorem 1: If C is a hyperbola, then k < 1 or k > 4
theorem hyperbola_condition (k : ℝ) :
  is_hyperbola k → k < 1 ∨ k > 4 :=
by sorry

-- Theorem 2: If C is an ellipse with foci on the x-axis, then 1 < k < 2.5
theorem ellipse_x_foci_condition (k : ℝ) :
  is_ellipse_x_foci k → 1 < k ∧ k < 2.5 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_condition_ellipse_x_foci_condition_l764_76476


namespace NUMINAMATH_CALUDE_largest_house_number_l764_76465

def phone_number : List Nat := [2, 7, 1, 3, 1, 4, 7]

def sum_digits (digits : List Nat) : Nat :=
  digits.sum

def is_distinct (digits : List Nat) : Prop :=
  digits.length = digits.eraseDups.length

def is_valid_house_number (house : List Nat) : Prop :=
  house.length = 4 ∧ 
  is_distinct house ∧
  sum_digits house = sum_digits phone_number

theorem largest_house_number : 
  ∀ house : List Nat, is_valid_house_number house → 
  house.foldl (fun acc d => acc * 10 + d) 0 ≤ 9871 :=
by sorry

end NUMINAMATH_CALUDE_largest_house_number_l764_76465


namespace NUMINAMATH_CALUDE_johns_annual_savings_l764_76452

/-- Calculates the annual savings for John's new apartment situation -/
theorem johns_annual_savings
  (former_rent_per_sqft : ℝ)
  (former_apartment_size : ℝ)
  (new_apartment_cost : ℝ)
  (h1 : former_rent_per_sqft = 2)
  (h2 : former_apartment_size = 750)
  (h3 : new_apartment_cost = 2800)
  : (former_rent_per_sqft * former_apartment_size - new_apartment_cost / 2) * 12 = 1200 := by
  sorry

#check johns_annual_savings

end NUMINAMATH_CALUDE_johns_annual_savings_l764_76452


namespace NUMINAMATH_CALUDE_min_value_of_expression_l764_76442

theorem min_value_of_expression (x : ℝ) :
  (x^2 - 4*x + 3) * (x^2 + 4*x + 3) ≥ -16 ∧
  ∃ y : ℝ, (y^2 - 4*y + 3) * (y^2 + 4*y + 3) = -16 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l764_76442


namespace NUMINAMATH_CALUDE_find_twelfth_number_l764_76482

/-- Given a set of 12 numbers where the sum of the first 11 is known and the arithmetic mean of all 12 is known, find the 12th number. -/
theorem find_twelfth_number (sum_first_eleven : ℕ) (arithmetic_mean : ℚ) (h1 : sum_first_eleven = 137) (h2 : arithmetic_mean = 12) :
  ∃ x : ℕ, (sum_first_eleven + x : ℚ) / 12 = arithmetic_mean ∧ x = 7 :=
by sorry

end NUMINAMATH_CALUDE_find_twelfth_number_l764_76482


namespace NUMINAMATH_CALUDE_range_of_quadratic_expression_l764_76431

theorem range_of_quadratic_expression (x y : ℝ) :
  (4 * x^2 - 2 * Real.sqrt 3 * x * y + 4 * y^2 = 13) →
  (10 - 4 * Real.sqrt 3 ≤ x^2 + 4 * y^2) ∧ (x^2 + 4 * y^2 ≤ 10 + 4 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_range_of_quadratic_expression_l764_76431


namespace NUMINAMATH_CALUDE_extra_legs_count_l764_76440

/-- Represents the number of legs for a cow -/
def cow_legs : ℕ := 4

/-- Represents the number of legs for a chicken -/
def chicken_legs : ℕ := 2

/-- Represents the number of cows in the group -/
def num_cows : ℕ := 9

theorem extra_legs_count (num_chickens : ℕ) : 
  cow_legs * num_cows + chicken_legs * num_chickens = 
  2 * (num_cows + num_chickens) + 18 := by
  sorry

end NUMINAMATH_CALUDE_extra_legs_count_l764_76440


namespace NUMINAMATH_CALUDE_existence_of_irrational_sum_l764_76491

theorem existence_of_irrational_sum (n : ℕ) (a : Fin n → ℝ) :
  ∃ (x : ℝ), ∀ (i : Fin n), Irrational (x + a i) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_irrational_sum_l764_76491


namespace NUMINAMATH_CALUDE_amusement_park_probabilities_l764_76415

/-- Amusement park problem -/
theorem amusement_park_probabilities
  (p_A1 : ℝ)
  (p_B1 : ℝ)
  (p_A2_given_A1 : ℝ)
  (p_A2_given_B1 : ℝ)
  (h1 : p_A1 = 0.3)
  (h2 : p_B1 = 0.7)
  (h3 : p_A2_given_A1 = 0.7)
  (h4 : p_A2_given_B1 = 0.6)
  (h5 : p_A1 + p_B1 = 1) :
  let p_A2 := p_A1 * p_A2_given_A1 + p_B1 * p_A2_given_B1
  let p_B1_given_A2 := (p_B1 * p_A2_given_B1) / p_A2
  ∃ (ε : ℝ), abs (p_A2 - 0.63) < ε ∧ abs (p_B1_given_A2 - (2/3)) < ε :=
by
  sorry


end NUMINAMATH_CALUDE_amusement_park_probabilities_l764_76415


namespace NUMINAMATH_CALUDE_exist_integers_with_gcd_property_l764_76436

theorem exist_integers_with_gcd_property :
  ∃ (a : Fin 2011 → ℕ+), (∀ i j, i < j → a i < a j) ∧
    (∀ i j, i < j → Nat.gcd (a i) (a j) = (a j) - (a i)) := by
  sorry

end NUMINAMATH_CALUDE_exist_integers_with_gcd_property_l764_76436


namespace NUMINAMATH_CALUDE_coat_drive_total_l764_76405

theorem coat_drive_total (high_school_coats elementary_school_coats : ℕ) 
  (h1 : high_school_coats = 6922)
  (h2 : elementary_school_coats = 2515) :
  high_school_coats + elementary_school_coats = 9437 :=
by sorry

end NUMINAMATH_CALUDE_coat_drive_total_l764_76405


namespace NUMINAMATH_CALUDE_puppies_left_l764_76494

theorem puppies_left (initial : ℕ) (given_away : ℕ) (h1 : initial = 7) (h2 : given_away = 5) :
  initial - given_away = 2 := by
  sorry

end NUMINAMATH_CALUDE_puppies_left_l764_76494


namespace NUMINAMATH_CALUDE_space_filling_crystalline_structure_exists_l764_76416

/-- A cell in a crystalline structure -/
inductive Cell
| Octahedron : Cell
| Tetrahedron : Cell

/-- A crystalline structure is a periodic arrangement of cells in space -/
structure CrystallineStructure :=
(cells : Set Cell)
(periodic : Bool)
(fillsSpace : Bool)

/-- The existence of a space-filling crystalline structure with octahedrons and tetrahedrons -/
theorem space_filling_crystalline_structure_exists :
  ∃ (c : CrystallineStructure), 
    c.cells = {Cell.Octahedron, Cell.Tetrahedron} ∧ 
    c.periodic = true ∧ 
    c.fillsSpace = true :=
sorry

end NUMINAMATH_CALUDE_space_filling_crystalline_structure_exists_l764_76416


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l764_76498

/-- Two lines are parallel if their slopes are equal but not equal to the ratio of their constants -/
def parallel (m₁ n₁ c₁ m₂ n₂ c₂ : ℝ) : Prop :=
  m₁ / n₁ = m₂ / n₂ ∧ m₁ / n₁ ≠ c₁ / c₂

theorem parallel_lines_a_value (a : ℝ) :
  parallel (3 + a) 4 (5 - 3*a) 2 (5 + a) 8 → a = -7 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l764_76498


namespace NUMINAMATH_CALUDE_A_difference_max_l764_76429

def A (a b : ℕ+) : ℚ := (a + 3) / 12

theorem A_difference_max :
  (∃ a₁ b₁ a₂ b₂ : ℕ+, 
    A a₁ b₁ = 15 / (26 - b₁) ∧
    A a₂ b₂ = 15 / (26 - b₂) ∧
    ∀ a b : ℕ+, A a b = 15 / (26 - b) → 
      A a₁ b₁ ≤ A a b ∧ A a b ≤ A a₂ b₂) →
  A a₂ b₂ - A a₁ b₁ = 57 / 4 :=
sorry

end NUMINAMATH_CALUDE_A_difference_max_l764_76429


namespace NUMINAMATH_CALUDE_discount_difference_l764_76406

def original_price : ℝ := 12000

def single_discount_rate : ℝ := 0.45
def successive_discount_rate1 : ℝ := 0.35
def successive_discount_rate2 : ℝ := 0.10

def price_after_single_discount : ℝ := original_price * (1 - single_discount_rate)
def price_after_successive_discounts : ℝ := original_price * (1 - successive_discount_rate1) * (1 - successive_discount_rate2)

theorem discount_difference :
  price_after_successive_discounts - price_after_single_discount = 420 := by
  sorry

end NUMINAMATH_CALUDE_discount_difference_l764_76406


namespace NUMINAMATH_CALUDE_negation_of_implication_l764_76427

theorem negation_of_implication (a b c : ℝ) :
  ¬(a + b + c = 3 → a^2 + b^2 + c^2 ≥ 3) ↔ (a + b + c = 3 ∧ a^2 + b^2 + c^2 < 3) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l764_76427


namespace NUMINAMATH_CALUDE_tenth_row_fifth_column_l764_76469

/-- Calculates the sum of the first n natural numbers -/
def triangularSum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Represents the triangular arrangement of natural numbers -/
def triangularArrangement (row : ℕ) (col : ℕ) : ℕ :=
  triangularSum (row - 1) + col

/-- The number in the 10th row and 5th column of the triangular arrangement -/
theorem tenth_row_fifth_column :
  triangularArrangement 10 5 = 101 := by
  sorry

end NUMINAMATH_CALUDE_tenth_row_fifth_column_l764_76469


namespace NUMINAMATH_CALUDE_equation_solution_l764_76443

theorem equation_solution (m n : ℚ) : 
  (m * 1 + n * 1 = 6) → 
  (m * 2 + n * (-1) = 6) → 
  (m = 4 ∧ n = 2) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l764_76443


namespace NUMINAMATH_CALUDE_part_I_part_II_l764_76488

-- Define the ellipse and line
def ellipse (x y : ℝ) : Prop := x^2 + 3*y^2 = 4
def line_l (x y : ℝ) : Prop := y = x + 2

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  ellipse t.A.1 t.A.2 ∧ 
  ellipse t.B.1 t.B.2 ∧
  line_l t.C.1 t.C.2 ∧
  (t.B.2 - t.A.2) / (t.B.1 - t.A.1) = 1

-- Theorem for part I
theorem part_I (t : Triangle) (h : triangle_conditions t) 
  (h_origin : t.A.1 = 0 ∧ t.A.2 = 0) :
  (∃ (AB_length area : ℝ), 
    AB_length = 2 * Real.sqrt 2 ∧ 
    area = 2) :=
sorry

-- Theorem for part II
theorem part_II (t : Triangle) (h : triangle_conditions t) 
  (h_right_angle : (t.B.1 - t.A.1) * (t.C.1 - t.A.1) + (t.B.2 - t.A.2) * (t.C.2 - t.A.2) = 0)
  (h_max_AC : ∀ (t' : Triangle), triangle_conditions t' → 
    (t'.C.1 - t'.A.1)^2 + (t'.C.2 - t'.A.2)^2 ≤ (t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2) :
  (∃ (m : ℝ), m = -1 ∧ t.B.2 - t.A.2 = t.B.1 - t.A.1 ∧ t.A.2 = t.A.1 + m) :=
sorry

end NUMINAMATH_CALUDE_part_I_part_II_l764_76488


namespace NUMINAMATH_CALUDE_right_triangle_tan_G_l764_76467

theorem right_triangle_tan_G (FG HG FH : ℝ) (h1 : FG = 17) (h2 : HG = 15) 
  (h3 : FG^2 = FH^2 + HG^2) : 
  FH / HG = 8 / 15 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_tan_G_l764_76467


namespace NUMINAMATH_CALUDE_cube_root_negative_27_l764_76409

theorem cube_root_negative_27 :
  (∃ x : ℝ, x^3 = -27 ∧ x = -3) ∧
  (¬ (∀ x : ℝ, x^2 = 64 → x = 8 ∨ x = -8)) ∧
  (¬ ((-Real.sqrt 2)^2 = 4)) ∧
  (¬ (Real.sqrt ((-5)^2) = -5)) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_negative_27_l764_76409


namespace NUMINAMATH_CALUDE_square_difference_divided_l764_76462

theorem square_difference_divided : (111^2 - 102^2) / 9 = 213 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_divided_l764_76462


namespace NUMINAMATH_CALUDE_y_minus_x_value_l764_76407

theorem y_minus_x_value (x y z : ℚ) 
  (eq1 : x + y + z = 12)
  (eq2 : x + y = 8)
  (eq3 : y - 3*x + z = 9) :
  y - x = 13/2 := by
sorry

end NUMINAMATH_CALUDE_y_minus_x_value_l764_76407


namespace NUMINAMATH_CALUDE_non_monotonic_range_l764_76478

/-- A function f is not monotonic on an interval if there exists a point in the interval where f' is zero --/
def NotMonotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x ∈ Set.Ioo a b, deriv f x = 0

/-- The main theorem --/
theorem non_monotonic_range (k : ℝ) :
  NotMonotonic (fun x => x^3 - 12*x) (k - 1) (k + 1) →
  k ∈ Set.union (Set.Ioo (-3) (-1)) (Set.Ioo 1 3) := by
  sorry

end NUMINAMATH_CALUDE_non_monotonic_range_l764_76478


namespace NUMINAMATH_CALUDE_complex_square_simplification_l764_76474

theorem complex_square_simplification :
  let i : ℂ := Complex.I
  (4 - 3 * i)^2 = 25 - 24 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_square_simplification_l764_76474


namespace NUMINAMATH_CALUDE_sum_of_powers_l764_76459

theorem sum_of_powers (a b : ℝ) 
  (h1 : (1 / (a + b)) ^ 2003 = 1)
  (h2 : (-a + b) ^ 2005 = 1) :
  a ^ 2003 + b ^ 2004 = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_powers_l764_76459


namespace NUMINAMATH_CALUDE_race_heartbeats_l764_76418

/-- Calculates the total number of heartbeats during a race given the heart rate, race distance, and pace. -/
def total_heartbeats (heart_rate : ℕ) (race_distance : ℕ) (pace : ℕ) : ℕ :=
  heart_rate * race_distance * pace

/-- Theorem stating that given specific conditions, the total number of heartbeats during a race is 28800. -/
theorem race_heartbeats :
  let heart_rate : ℕ := 160  -- beats per minute
  let race_distance : ℕ := 30  -- miles
  let pace : ℕ := 6  -- minutes per mile
  total_heartbeats heart_rate race_distance pace = 28800 :=
by sorry

end NUMINAMATH_CALUDE_race_heartbeats_l764_76418


namespace NUMINAMATH_CALUDE_infinite_squares_l764_76414

theorem infinite_squares (k : ℕ) (hk : k ≥ 2) :
  ∃ (S : Set ℕ), Set.Infinite S ∧
    ∀ n ∈ S, ∃ (u v : ℕ), k * n + 1 = u^2 ∧ (k + 1) * n + 1 = v^2 := by
  sorry

end NUMINAMATH_CALUDE_infinite_squares_l764_76414


namespace NUMINAMATH_CALUDE_no_120_cents_combination_l764_76420

/-- Represents the types of coins available --/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter

/-- Returns the value of a coin in cents --/
def coinValue (c : Coin) : ℕ :=
  match c with
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25

/-- A selection of coins --/
def CoinSelection := List Coin

/-- Calculates the total value of a coin selection in cents --/
def totalValue (selection : CoinSelection) : ℕ :=
  selection.map coinValue |>.sum

/-- Theorem: It's impossible to select 6 coins with a total value of 120 cents --/
theorem no_120_cents_combination :
  ¬ ∃ (selection : CoinSelection), selection.length = 6 ∧ totalValue selection = 120 := by
  sorry

end NUMINAMATH_CALUDE_no_120_cents_combination_l764_76420


namespace NUMINAMATH_CALUDE_range_of_m_l764_76425

-- Define the propositions
def p (x : ℝ) : Prop := x^2 + x - 2 > 0
def q (x m : ℝ) : Prop := x > m

-- Define the theorem
theorem range_of_m :
  (∀ x m : ℝ, (¬(q x m) → ¬(p x)) ∧ ¬(¬(p x) → ¬(q x m))) →
  (∀ m : ℝ, m ≥ 1 ↔ ∃ x : ℝ, p x ∧ q x m) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l764_76425


namespace NUMINAMATH_CALUDE_units_digit_of_seven_to_six_to_five_l764_76428

theorem units_digit_of_seven_to_six_to_five (n : ℕ) : n = 7^(6^5) → n % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_seven_to_six_to_five_l764_76428


namespace NUMINAMATH_CALUDE_first_book_cost_l764_76471

/-- The cost of Shelby's first book given her spending at the book fair -/
theorem first_book_cost (initial_amount : ℕ) (second_book_cost : ℕ) (poster_cost : ℕ) (num_posters : ℕ) :
  initial_amount = 20 →
  second_book_cost = 4 →
  poster_cost = 4 →
  num_posters = 2 →
  ∃ (first_book_cost : ℕ),
    first_book_cost + second_book_cost + (num_posters * poster_cost) = initial_amount ∧
    first_book_cost = 8 :=
by sorry

end NUMINAMATH_CALUDE_first_book_cost_l764_76471


namespace NUMINAMATH_CALUDE_common_tangents_exist_curves_intersect_at_angles_l764_76470

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines an ellipse with the equation 16x^2 + 25y^2 = 400 -/
def is_on_ellipse (p : Point) : Prop :=
  16 * p.x^2 + 25 * p.y^2 = 400

/-- Defines a circle with the equation x^2 + y^2 = 20 -/
def is_on_circle (p : Point) : Prop :=
  p.x^2 + p.y^2 = 20

/-- Represents a line in 2D space using the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a line is tangent to the ellipse -/
def is_tangent_to_ellipse (l : Line) : Prop :=
  ∃ p : Point, is_on_ellipse p ∧ l.a * p.x + l.b * p.y + l.c = 0

/-- Checks if a line is tangent to the circle -/
def is_tangent_to_circle (l : Line) : Prop :=
  ∃ p : Point, is_on_circle p ∧ l.a * p.x + l.b * p.y + l.c = 0

/-- Theorem stating that there exist common tangents to the ellipse and circle -/
theorem common_tangents_exist : 
  ∃ l : Line, is_tangent_to_ellipse l ∧ is_tangent_to_circle l :=
sorry

/-- Calculates the angle between two curves at an intersection point -/
noncomputable def angle_between_curves (p : Point) : ℝ :=
sorry

/-- Theorem stating that the ellipse and circle intersect at certain angles -/
theorem curves_intersect_at_angles : 
  ∃ p : Point, is_on_ellipse p ∧ is_on_circle p ∧ angle_between_curves p ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_common_tangents_exist_curves_intersect_at_angles_l764_76470


namespace NUMINAMATH_CALUDE_sin_cos_fourth_power_nonnegative_with_zero_l764_76457

theorem sin_cos_fourth_power_nonnegative_with_zero (x : ℝ) :
  (∀ x, (Real.sin x + Real.cos x)^4 ≥ 0) ∧
  (∃ x, (Real.sin x + Real.cos x)^4 = 0) := by
sorry

end NUMINAMATH_CALUDE_sin_cos_fourth_power_nonnegative_with_zero_l764_76457


namespace NUMINAMATH_CALUDE_min_sum_distances_min_sum_distances_equality_l764_76454

theorem min_sum_distances (u v : ℝ) : 
  Real.sqrt (u^2 + v^2) + Real.sqrt ((u - 1)^2 + v^2) + 
  Real.sqrt (u^2 + (v - 1)^2) + Real.sqrt ((u - 1)^2 + (v - 1)^2) ≥ 2 * Real.sqrt 2 :=
by sorry

theorem min_sum_distances_equality : 
  Real.sqrt ((1/2)^2 + (1/2)^2) + Real.sqrt ((1/2 - 1)^2 + (1/2)^2) + 
  Real.sqrt ((1/2)^2 + (1/2 - 1)^2) + Real.sqrt ((1/2 - 1)^2 + (1/2 - 1)^2) = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_distances_min_sum_distances_equality_l764_76454


namespace NUMINAMATH_CALUDE_table_runner_coverage_l764_76490

theorem table_runner_coverage (total_runner_area : ℝ) (table_area : ℝ) (coverage_percentage : ℝ) (three_layer_area : ℝ) :
  total_runner_area = 212 →
  table_area = 175 →
  coverage_percentage = 0.80 →
  three_layer_area = 24 →
  ∃ (two_layer_area : ℝ),
    two_layer_area = 48 ∧
    two_layer_area + three_layer_area + (coverage_percentage * table_area - two_layer_area - three_layer_area) = coverage_percentage * table_area ∧
    two_layer_area + three_layer_area = total_runner_area - (coverage_percentage * table_area - two_layer_area - three_layer_area) :=
by sorry

end NUMINAMATH_CALUDE_table_runner_coverage_l764_76490


namespace NUMINAMATH_CALUDE_wire_cut_ratio_l764_76447

theorem wire_cut_ratio (x y : ℝ) : 
  x > 0 → y > 0 → -- Ensure positive lengths
  (4 * (x / 4) = 5 * (y / 5)) → -- Equal perimeters condition
  x / y = 1 := by
sorry

end NUMINAMATH_CALUDE_wire_cut_ratio_l764_76447


namespace NUMINAMATH_CALUDE_hexagon_area_ratio_l764_76410

theorem hexagon_area_ratio (a b : ℝ) (ha : a > 0) (hb : b = 2 * a) :
  (3 * Real.sqrt 3 / 2 * a^2) / (3 * Real.sqrt 3 / 2 * b^2) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_area_ratio_l764_76410


namespace NUMINAMATH_CALUDE_common_course_probability_l764_76402

/-- Represents the set of all possible course selections for a student -/
def CourseSelection : Type := Fin 10

/-- The total number of possible course selections for three students -/
def totalCombinations : ℕ := 1000

/-- The number of favorable combinations where at least two students share two courses -/
def favorableCombinations : ℕ := 280

/-- The probability that any one student will have at least two elective courses in common with the other two students -/
def commonCourseProbability : ℚ := 79 / 250

theorem common_course_probability :
  (favorableCombinations : ℚ) / totalCombinations = commonCourseProbability := by
  sorry

end NUMINAMATH_CALUDE_common_course_probability_l764_76402


namespace NUMINAMATH_CALUDE_x_axis_segment_range_l764_76449

/-- Definition of a quadratic function -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ -a * x^2 + 2 * b * x - c

/-- Definition of the centrally symmetric function with respect to (0,0) -/
def CentrallySymmetricFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + 2 * b * x + c

/-- Theorem about the range of x-axis segment length for the centrally symmetric function -/
theorem x_axis_segment_range
  (a b c : ℝ)
  (ha : a ≠ 0)
  (h1 : a + b + c = 0)
  (h2 : (2*c + b - a) * (2*c + b + 3*a) < 0) :
  ∃ (x₁ x₂ : ℝ), CentrallySymmetricFunction a b c x₁ = 0 ∧
                 CentrallySymmetricFunction a b c x₂ = 0 ∧
                 Real.sqrt 3 < |x₁ - x₂| ∧
                 |x₁ - x₂| < 2 * Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_x_axis_segment_range_l764_76449


namespace NUMINAMATH_CALUDE_subset_implies_m_leq_3_l764_76441

/-- Set A definition -/
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 5}

/-- Set B definition -/
def B (m : ℝ) : Set ℝ := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

/-- Theorem stating that if B is a subset of A, then m ≤ 3 -/
theorem subset_implies_m_leq_3 (m : ℝ) (h : B m ⊆ A) : m ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_m_leq_3_l764_76441


namespace NUMINAMATH_CALUDE_book_cost_is_300_divided_by_num_books_l764_76423

/-- Represents the cost of lawn mowing and video games -/
structure Costs where
  lawn_price : ℕ
  video_game_price : ℕ

/-- Represents Kenny's lawn mowing and purchasing activities -/
structure KennyActivities where
  costs : Costs
  lawns_mowed : ℕ
  video_games_bought : ℕ

/-- Calculates the cost of each book based on Kenny's activities -/
def book_cost (activities : KennyActivities) (num_books : ℕ) : ℚ :=
  let total_earned := activities.costs.lawn_price * activities.lawns_mowed
  let spent_on_games := activities.costs.video_game_price * activities.video_games_bought
  let remaining_for_books := total_earned - spent_on_games
  (remaining_for_books : ℚ) / num_books

/-- Theorem stating that the cost of each book is $300 divided by the number of books -/
theorem book_cost_is_300_divided_by_num_books 
  (activities : KennyActivities) 
  (num_books : ℕ) 
  (h1 : activities.costs.lawn_price = 15)
  (h2 : activities.costs.video_game_price = 45)
  (h3 : activities.lawns_mowed = 35)
  (h4 : activities.video_games_bought = 5)
  (h5 : num_books > 0) :
  book_cost activities num_books = 300 / num_books :=
by
  sorry

#check book_cost_is_300_divided_by_num_books

end NUMINAMATH_CALUDE_book_cost_is_300_divided_by_num_books_l764_76423


namespace NUMINAMATH_CALUDE_decimal_255_to_octal_l764_76421

-- Define a function to convert decimal to octal
def decimal_to_octal (n : ℕ) : List ℕ :=
  sorry

-- Theorem statement
theorem decimal_255_to_octal :
  decimal_to_octal 255 = [3, 7, 7] := by
  sorry

end NUMINAMATH_CALUDE_decimal_255_to_octal_l764_76421


namespace NUMINAMATH_CALUDE_remainder_divisibility_l764_76445

theorem remainder_divisibility (N : ℤ) : N % 17 = 2 → N % 357 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l764_76445


namespace NUMINAMATH_CALUDE_sandwich_bread_packs_l764_76448

theorem sandwich_bread_packs (total_sandwiches : ℕ) (slices_per_sandwich : ℕ) (packs_bought : ℕ) :
  total_sandwiches = 8 →
  slices_per_sandwich = 2 →
  packs_bought = 4 →
  (total_sandwiches * slices_per_sandwich) / packs_bought = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_sandwich_bread_packs_l764_76448
