import Mathlib

namespace NUMINAMATH_CALUDE_largest_prime_factor_of_pythagorean_triplet_number_l2699_269902

/-- Given a three-digit number abc where a, b, and c are nonzero digits
    satisfying a^2 + b^2 = c^2, the largest possible prime factor of abc is 29. -/
theorem largest_prime_factor_of_pythagorean_triplet_number : ∃ (a b c : ℕ),
  (1 ≤ a ∧ a ≤ 9) ∧ 
  (1 ≤ b ∧ b ≤ 9) ∧ 
  (1 ≤ c ∧ c ≤ 9) ∧ 
  a^2 + b^2 = c^2 ∧
  (∀ p : ℕ, p.Prime → p ∣ (100*a + 10*b + c) → p ≤ 29) ∧
  29 ∣ (100*a + 10*b + c) :=
sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_pythagorean_triplet_number_l2699_269902


namespace NUMINAMATH_CALUDE_three_card_selection_count_l2699_269987

/-- The number of cards in a standard deck -/
def standard_deck_size : ℕ := 52

/-- The number of suits in a standard deck -/
def number_of_suits : ℕ := 4

/-- The number of cards in each suit -/
def cards_per_suit : ℕ := 13

/-- 
  Theorem: The number of ways to select three different cards in sequence 
  from a standard deck is 132600.
-/
theorem three_card_selection_count : 
  standard_deck_size * (standard_deck_size - 1) * (standard_deck_size - 2) = 132600 := by
  sorry


end NUMINAMATH_CALUDE_three_card_selection_count_l2699_269987


namespace NUMINAMATH_CALUDE_train_speed_l2699_269977

/-- The speed of a train crossing a bridge -/
theorem train_speed (train_length bridge_length : ℝ) (crossing_time : ℝ) 
  (h1 : train_length = 140)
  (h2 : bridge_length = 235)
  (h3 : crossing_time = 30) :
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l2699_269977


namespace NUMINAMATH_CALUDE_crayons_left_correct_l2699_269971

/-- Given an initial number of crayons and a number of crayons lost or given away,
    calculate the number of crayons left. -/
def crayons_left (initial : ℕ) (lost_or_given : ℕ) : ℕ :=
  initial - lost_or_given

/-- Theorem: The number of crayons left is equal to the initial number minus
    the number lost or given away. -/
theorem crayons_left_correct (initial : ℕ) (lost_or_given : ℕ) 
  (h : lost_or_given ≤ initial) : 
  crayons_left initial lost_or_given = initial - lost_or_given :=
by sorry

end NUMINAMATH_CALUDE_crayons_left_correct_l2699_269971


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l2699_269926

/-- Given a geometric sequence {a_n} with sum S_n of first n terms, prove the general formula. -/
theorem geometric_sequence_formula (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- geometric sequence condition
  a 3 = 3/2 →                   -- given a_3
  S 3 = 9/2 →                   -- given S_3
  (∀ n, S n = (a 1) * (1 - q^n) / (1 - q)) →  -- sum formula for geometric sequence
  (q = 1 ∨ q = -1/2) ∧
  (∀ n, (q = 1 → a n = 3/2) ∧ 
        (q = -1/2 → a n = 6 * (-1/2)^(n-1))) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l2699_269926


namespace NUMINAMATH_CALUDE_nails_needed_l2699_269905

theorem nails_needed (nails_per_plank : ℕ) (num_planks : ℕ) : 
  nails_per_plank = 2 → num_planks = 2 → nails_per_plank * num_planks = 4 := by
  sorry

#check nails_needed

end NUMINAMATH_CALUDE_nails_needed_l2699_269905


namespace NUMINAMATH_CALUDE_team_a_games_won_lost_team_b_minimum_wins_l2699_269991

/-- Represents the number of games a team plays in the tournament -/
def total_games : ℕ := 10

/-- Represents the points earned for a win -/
def win_points : ℕ := 2

/-- Represents the points earned for a loss -/
def loss_points : ℕ := 1

/-- Represents the minimum points needed to qualify for the next round -/
def qualification_points : ℕ := 15

theorem team_a_games_won_lost (points : ℕ) (h : points = 18) :
  ∃ (wins losses : ℕ), wins + losses = total_games ∧
                        wins * win_points + losses * loss_points = points ∧
                        wins = 8 ∧ losses = 2 := by sorry

theorem team_b_minimum_wins :
  ∃ (min_wins : ℕ), ∀ (wins : ℕ),
    wins * win_points + (total_games - wins) * loss_points > qualification_points →
    wins ≥ min_wins ∧
    min_wins = 6 := by sorry

end NUMINAMATH_CALUDE_team_a_games_won_lost_team_b_minimum_wins_l2699_269991


namespace NUMINAMATH_CALUDE_rod_cutting_l2699_269966

theorem rod_cutting (rod_length piece_length : ℝ) (h1 : rod_length = 42.5) (h2 : piece_length = 0.85) :
  ⌊rod_length / piece_length⌋ = 50 := by
sorry

end NUMINAMATH_CALUDE_rod_cutting_l2699_269966


namespace NUMINAMATH_CALUDE_circles_intersect_l2699_269922

/-- Circle C₁ with equation x² + y² + 2x + 2y - 2 = 0 -/
def C₁ (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x + 2*y - 2 = 0

/-- Circle C₂ with equation x² + y² - 4x - 2y + 1 = 0 -/
def C₂ (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 2*y + 1 = 0

/-- The circles C₁ and C₂ are intersecting -/
theorem circles_intersect : ∃ (x y : ℝ), C₁ x y ∧ C₂ x y := by
  sorry

end NUMINAMATH_CALUDE_circles_intersect_l2699_269922


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2699_269921

theorem necessary_but_not_sufficient 
  (a b : ℝ) : 
  (((b + 2) / (a + 2) > b / a) ↔ (a > b ∧ b > 0)) → False :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2699_269921


namespace NUMINAMATH_CALUDE_min_value_expression_l2699_269915

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + 1/y) * (x + 1/y - 2018) + (y + 1/x) * (y + 1/x - 2018) ≥ -2036162 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2699_269915


namespace NUMINAMATH_CALUDE_solve_for_a_l2699_269956

theorem solve_for_a (x a : ℝ) (h1 : 2 * x - a + 5 = 0) (h2 : x = -2) : a = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l2699_269956


namespace NUMINAMATH_CALUDE_max_intersections_six_paths_l2699_269945

/-- The number of intersection points for a given number of paths -/
def intersection_points (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: With 6 paths, where each path intersects with every other path
    exactly once, the maximum number of intersection points is 15 -/
theorem max_intersections_six_paths :
  intersection_points 6 = 15 := by
  sorry

#eval intersection_points 6  -- This will output 15

end NUMINAMATH_CALUDE_max_intersections_six_paths_l2699_269945


namespace NUMINAMATH_CALUDE_gym_membership_cost_theorem_l2699_269950

/-- Calculates the total cost of a gym membership for a given number of years -/
def gymMembershipCost (monthlyFee : ℕ) (downPayment : ℕ) (years : ℕ) : ℕ :=
  monthlyFee * 12 * years + downPayment

/-- Theorem: The total cost for a 3-year gym membership with a $12 monthly fee and $50 down payment is $482 -/
theorem gym_membership_cost_theorem :
  gymMembershipCost 12 50 3 = 482 := by
  sorry

end NUMINAMATH_CALUDE_gym_membership_cost_theorem_l2699_269950


namespace NUMINAMATH_CALUDE_misread_number_correction_l2699_269923

theorem misread_number_correction (n : ℕ) (incorrect_avg correct_avg incorrect_number : ℚ) 
  (h1 : n = 10)
  (h2 : incorrect_avg = 19)
  (h3 : correct_avg = 24)
  (h4 : incorrect_number = 26) :
  ∃ (correct_number : ℚ), 
    (n : ℚ) * correct_avg - (n : ℚ) * incorrect_avg = correct_number - incorrect_number ∧
    correct_number = 76 := by
  sorry

end NUMINAMATH_CALUDE_misread_number_correction_l2699_269923


namespace NUMINAMATH_CALUDE_equation_solution_l2699_269919

theorem equation_solution : ∃ x : ℝ, (((1 + x) / (2 - x)) - 1 = 1 / (x - 2)) ∧ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2699_269919


namespace NUMINAMATH_CALUDE_minimum_sum_geometric_mean_l2699_269907

theorem minimum_sum_geometric_mean (a b : ℝ) (ha : a > 0) (hb : b > 0) (hgm : Real.sqrt (a * b) = 1) :
  2 * (a + b) ≥ 4 ∧ (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ Real.sqrt (x * y) = 1 ∧ 2 * (x + y) = 4) := by
  sorry

end NUMINAMATH_CALUDE_minimum_sum_geometric_mean_l2699_269907


namespace NUMINAMATH_CALUDE_altitude_inradius_sum_implies_equilateral_l2699_269901

/-- A triangle with side lengths a, b, c, altitudes h₁, h₂, h₃, and inradius r. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h₁ : ℝ
  h₂ : ℝ
  h₃ : ℝ
  r : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  pos_h₁ : 0 < h₁
  pos_h₂ : 0 < h₂
  pos_h₃ : 0 < h₃
  pos_r : 0 < r
  altitude_sum : h₁ + h₂ + h₃ = 9 * r

/-- A triangle is equilateral if all its sides are equal. -/
def Triangle.isEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

/-- If the altitudes and the radius of the inscribed circle of a triangle satisfy
    h₁ + h₂ + h₃ = 9r, then the triangle is equilateral. -/
theorem altitude_inradius_sum_implies_equilateral (t : Triangle) :
  t.isEquilateral :=
sorry

end NUMINAMATH_CALUDE_altitude_inradius_sum_implies_equilateral_l2699_269901


namespace NUMINAMATH_CALUDE_complex_number_value_l2699_269969

theorem complex_number_value : Complex.I ^ 2 * (1 + Complex.I) = -1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_value_l2699_269969


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_m_greater_than_one_sufficient_m_greater_than_one_not_necessary_l2699_269953

theorem sufficient_not_necessary_condition (m : ℝ) : 
  (∀ x ≥ 1, 3^(x + m) - 3 * Real.sqrt 3 > 0) ↔ m > 1/2 :=
by sorry

theorem m_greater_than_one_sufficient (m : ℝ) :
  m > 1 → ∀ x ≥ 1, 3^(x + m) - 3 * Real.sqrt 3 > 0 :=
by sorry

theorem m_greater_than_one_not_necessary :
  ∃ m, m ≤ 1 ∧ (∀ x ≥ 1, 3^(x + m) - 3 * Real.sqrt 3 > 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_m_greater_than_one_sufficient_m_greater_than_one_not_necessary_l2699_269953


namespace NUMINAMATH_CALUDE_complex_product_in_first_quadrant_l2699_269908

/-- The point corresponding to (1+3i)(3-i) is located in the first quadrant. -/
theorem complex_product_in_first_quadrant :
  let z : ℂ := (1 + 3*I) * (3 - I)
  (z.re > 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_product_in_first_quadrant_l2699_269908


namespace NUMINAMATH_CALUDE_cosine_inequality_solution_l2699_269918

theorem cosine_inequality_solution (y : Real) : 
  (y ∈ Set.Icc 0 (Real.pi / 2)) → 
  (∀ x ∈ Set.Icc 0 (2 * Real.pi), Real.cos (x + y) ≥ Real.cos x - Real.cos y) → 
  y = 0 :=
sorry

end NUMINAMATH_CALUDE_cosine_inequality_solution_l2699_269918


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2699_269955

/-- The eccentricity of a hyperbola with equation x^2/a^2 - y^2/b^2 = 1,
    where one of its asymptotes passes through the point (3, -4), is 5/3. -/
theorem hyperbola_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → c^2 = a^2 + b^2) →
  (∃ k : ℝ, k * 3 = a ∧ k * (-4) = b) →
  c / a = 5 / 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2699_269955


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2699_269943

theorem inequality_solution_set :
  {x : ℝ | 3 - 2*x > 7} = {x : ℝ | x < -2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2699_269943


namespace NUMINAMATH_CALUDE_paint_time_per_room_l2699_269928

theorem paint_time_per_room 
  (total_rooms : ℕ) 
  (painted_rooms : ℕ) 
  (remaining_time : ℕ) 
  (h1 : total_rooms = 12) 
  (h2 : painted_rooms = 5) 
  (h3 : remaining_time = 49) : 
  remaining_time / (total_rooms - painted_rooms) = 7 := by
  sorry

end NUMINAMATH_CALUDE_paint_time_per_room_l2699_269928


namespace NUMINAMATH_CALUDE_factorization_1_factorization_2_l2699_269903

-- Part 1
theorem factorization_1 (a : ℝ) : (a^2 - 4*a + 4) - 4*(a - 2) + 4 = (a - 4)^2 := by
  sorry

-- Part 2
theorem factorization_2 (x y : ℝ) : 16*x^4 - 81*y^4 = (4*x^2 + 9*y^2)*(2*x + 3*y)*(2*x - 3*y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_1_factorization_2_l2699_269903


namespace NUMINAMATH_CALUDE_horse_purchase_problem_l2699_269959

/-- The problem of three people buying a horse -/
theorem horse_purchase_problem (x y z : ℚ) : 
  (x + 1/2 * y + 1/2 * z = 12) →
  (y + 1/3 * x + 1/3 * z = 12) →
  (z + 1/4 * x + 1/4 * y = 12) →
  (x = 60/17 ∧ y = 136/17 ∧ z = 156/17) := by
  sorry

end NUMINAMATH_CALUDE_horse_purchase_problem_l2699_269959


namespace NUMINAMATH_CALUDE_direct_square_variation_problem_l2699_269912

/-- A function representing direct variation with the square of x -/
def direct_square_variation (k : ℝ) (x : ℝ) : ℝ := k * x^2

theorem direct_square_variation_problem (y : ℝ → ℝ) :
  (∃ k : ℝ, ∀ x, y x = direct_square_variation k x) →  -- y varies directly as the square of x
  y 3 = 18 →  -- y = 18 when x = 3
  y 6 = 72 :=  -- y = 72 when x = 6
by
  sorry

end NUMINAMATH_CALUDE_direct_square_variation_problem_l2699_269912


namespace NUMINAMATH_CALUDE_expected_deviation_10_gt_100_l2699_269995

/-- Represents the outcome of a coin toss experiment -/
structure CoinTossExperiment where
  n : ℕ  -- number of tosses
  m : ℕ  -- number of heads
  h_m_le_n : m ≤ n  -- ensure m is not greater than n

/-- The frequency of heads in a coin toss experiment -/
def frequency (e : CoinTossExperiment) : ℚ :=
  e.m / e.n

/-- The deviation of the frequency from the probability of a fair coin (0.5) -/
def deviation (e : CoinTossExperiment) : ℚ :=
  frequency e - 1/2

/-- The absolute deviation of the frequency from the probability of a fair coin (0.5) -/
def absoluteDeviation (e : CoinTossExperiment) : ℚ :=
  |deviation e|

/-- The expected value of the absolute deviation for n coin tosses -/
noncomputable def expectedAbsoluteDeviation (n : ℕ) : ℝ :=
  sorry  -- Definition not provided in the problem, so we leave it as sorry

/-- Theorem stating that the expected absolute deviation for 10 tosses
    is greater than for 100 tosses -/
theorem expected_deviation_10_gt_100 :
  expectedAbsoluteDeviation 10 > expectedAbsoluteDeviation 100 :=
by sorry

end NUMINAMATH_CALUDE_expected_deviation_10_gt_100_l2699_269995


namespace NUMINAMATH_CALUDE_angle_A_value_perimeter_range_l2699_269962

-- Define the triangle
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Sides

-- Define the conditions
axiom triangle_condition : a * Real.cos C + Real.sqrt 3 * Real.sin C - b - c = 0
axiom positive_sides : a > 0 ∧ b > 0 ∧ c > 0
axiom angle_sum : A + B + C = Real.pi
axiom law_of_sines : a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C

-- Part 1: Prove that A = π/3
theorem angle_A_value : A = Real.pi / 3 := by sorry

-- Part 2: Prove the perimeter range
theorem perimeter_range (h_acute : A < Real.pi/2 ∧ B < Real.pi/2 ∧ C < Real.pi/2) (h_c : c = 3) :
  (3 * Real.sqrt 3 + 9) / 2 < a + b + c ∧ a + b + c < 9 + 3 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_angle_A_value_perimeter_range_l2699_269962


namespace NUMINAMATH_CALUDE_trapezoidal_fence_poles_l2699_269994

/-- Calculates the number of poles needed for a trapezoidal fence --/
theorem trapezoidal_fence_poles
  (parallel_side1 parallel_side2 non_parallel_side : ℕ)
  (parallel_pole_interval non_parallel_pole_interval : ℕ)
  (h1 : parallel_side1 = 60)
  (h2 : parallel_side2 = 80)
  (h3 : non_parallel_side = 50)
  (h4 : parallel_pole_interval = 5)
  (h5 : non_parallel_pole_interval = 7) :
  (parallel_side1 / parallel_pole_interval + 1) +
  (parallel_side2 / parallel_pole_interval + 1) +
  2 * (⌈(non_parallel_side : ℝ) / non_parallel_pole_interval⌉ + 1) - 4 = 44 := by
  sorry

#check trapezoidal_fence_poles

end NUMINAMATH_CALUDE_trapezoidal_fence_poles_l2699_269994


namespace NUMINAMATH_CALUDE_quadratic_roots_distance_bounds_l2699_269929

theorem quadratic_roots_distance_bounds (z₁ z₂ m : ℂ) (α β : ℂ) :
  (∀ x : ℂ, x^2 + z₁*x + z₂ + m = 0 ↔ x = α ∨ x = β) →
  z₁^2 - 4*z₂ = 16 + 20*I →
  Complex.abs (α - β) = 2 * Real.sqrt 7 →
  (Complex.abs m ≤ 7 + Real.sqrt 41 ∧ Complex.abs m ≥ 7 - Real.sqrt 41) ∧
  (∃ m₁ m₂ : ℂ, Complex.abs m₁ = 7 + Real.sqrt 41 ∧ Complex.abs m₂ = 7 - Real.sqrt 41) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_distance_bounds_l2699_269929


namespace NUMINAMATH_CALUDE_triangle_area_l2699_269900

/-- Given a triangle with sides in ratio 5:12:13, perimeter 300 m, and angle 45° between shortest and middle sides, its area is 1500 * √2 m² -/
theorem triangle_area (a b c : ℝ) (h_ratio : (a, b, c) = (5, 12, 13)) 
  (h_perimeter : a + b + c = 300) (h_angle : Real.cos (45 * π / 180) = b / (2 * a)) : 
  (1/2) * a * b * Real.sin (45 * π / 180) = 1500 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2699_269900


namespace NUMINAMATH_CALUDE_tangent_line_circle_product_l2699_269917

/-- Given a line ax + by - 3 = 0 tangent to the circle x^2 + y^2 + 4x - 1 = 0 at point P(-1, 2),
    the product ab equals 2. -/
theorem tangent_line_circle_product (a b : ℝ) : 
  (∀ x y, a * x + b * y - 3 = 0 → x^2 + y^2 + 4*x - 1 = 0 → (x + 1)^2 + (y - 2)^2 ≠ 0) →
  a * (-1) + b * 2 - 3 = 0 →
  (-1)^2 + 2^2 + 4*(-1) - 1 = 0 →
  a * b = 2 := by
sorry


end NUMINAMATH_CALUDE_tangent_line_circle_product_l2699_269917


namespace NUMINAMATH_CALUDE_probability_point_in_ellipsoid_l2699_269993

/-- The probability of a point in a rectangular prism satisfying an ellipsoid equation -/
theorem probability_point_in_ellipsoid : 
  let prism_volume := (2 - (-2)) * (1 - (-1)) * (1 - (-1))
  let ellipsoid_volume := (4 * Real.pi / 3) * 1 * 2 * 2
  let probability := ellipsoid_volume / prism_volume
  probability = Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_point_in_ellipsoid_l2699_269993


namespace NUMINAMATH_CALUDE_line_circle_intersection_l2699_269999

/-- Given a line and a circle that intersect, prove the value of the line's slope --/
theorem line_circle_intersection (a : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    (a * A.1 - A.2 + 3 = 0) ∧ 
    (a * B.1 - B.2 + 3 = 0) ∧ 
    ((A.1 - 1)^2 + (A.2 - 2)^2 = 4) ∧ 
    ((B.1 - 1)^2 + (B.2 - 2)^2 = 4) ∧ 
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 12)) →
  a = 0 := by
sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l2699_269999


namespace NUMINAMATH_CALUDE_isosceles_triangle_smallest_angle_isosceles_triangle_smallest_angle_proof_l2699_269980

/-- An isosceles triangle with one angle 40% larger than a right angle has two smallest angles measuring 27°. -/
theorem isosceles_triangle_smallest_angle : ℝ → Prop :=
  fun x =>
    let right_angle := 90
    let large_angle := 1.4 * right_angle
    let sum_of_angles := 180
    x > 0 ∧ 
    x < large_angle ∧ 
    2 * x + large_angle = sum_of_angles →
    x = 27

/-- Proof of the theorem -/
theorem isosceles_triangle_smallest_angle_proof : isosceles_triangle_smallest_angle 27 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_smallest_angle_isosceles_triangle_smallest_angle_proof_l2699_269980


namespace NUMINAMATH_CALUDE_tea_bags_count_l2699_269948

/-- The number of tea bags in a box -/
def n : ℕ+ := sorry

/-- The number of cups of tea made from Natasha's box -/
def natasha_cups : ℕ := 41

/-- The number of cups of tea made from Inna's box -/
def inna_cups : ℕ := 58

/-- Theorem stating that the number of tea bags in the box is 20 -/
theorem tea_bags_count :
  (2 * n ≤ natasha_cups ∧ natasha_cups ≤ 3 * n) ∧
  (2 * n ≤ inna_cups ∧ inna_cups ≤ 3 * n) →
  n = 20 := by sorry

end NUMINAMATH_CALUDE_tea_bags_count_l2699_269948


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2699_269930

/-- The eccentricity of a hyperbola with equation y² - x²/4 = 1 is √5 -/
theorem hyperbola_eccentricity : 
  let hyperbola := fun (x y : ℝ) => y^2 - x^2/4 = 1
  ∃ e : ℝ, e = Real.sqrt 5 ∧ 
    ∀ x y : ℝ, hyperbola x y → 
      e = Real.sqrt ((1 + 4) / 1) := by
        sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2699_269930


namespace NUMINAMATH_CALUDE_prob_all_even_before_odd_prob_all_even_before_odd_proof_l2699_269944

/-- Represents an 8-sided die with numbers from 1 to 8 -/
inductive Die
| one | two | three | four | five | six | seven | eight

/-- Defines whether a number on the die is even or odd -/
def Die.isEven : Die → Bool
| Die.two => true
| Die.four => true
| Die.six => true
| Die.eight => true
| _ => false

/-- The probability of rolling an even number -/
def probEven : ℚ := 1/2

/-- The probability of rolling an odd number -/
def probOdd : ℚ := 1/2

/-- The set of even numbers on the die -/
def evenNumbers : Set Die := {Die.two, Die.four, Die.six, Die.eight}

/-- Theorem: The probability of rolling every even number at least once
    before rolling any odd number on an 8-sided die is 1/384 -/
theorem prob_all_even_before_odd : ℚ :=
  1/384

/-- Proof of the theorem -/
theorem prob_all_even_before_odd_proof :
  prob_all_even_before_odd = 1/384 := by
  sorry

end NUMINAMATH_CALUDE_prob_all_even_before_odd_prob_all_even_before_odd_proof_l2699_269944


namespace NUMINAMATH_CALUDE_point_movement_l2699_269970

/-- Represents a 2D vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Calculates the new position of a point after moving with a given velocity for a certain time -/
def move (p : Point2D) (v : Vector2D) (t : ℝ) : Point2D :=
  { x := p.x + v.x * t,
    y := p.y + v.y * t }

theorem point_movement :
  let initialPoint : Point2D := { x := -10, y := 10 }
  let velocity : Vector2D := { x := 4, y := -3 }
  let time : ℝ := 5
  let finalPoint : Point2D := move initialPoint velocity time
  finalPoint = { x := 10, y := -5 } := by sorry

end NUMINAMATH_CALUDE_point_movement_l2699_269970


namespace NUMINAMATH_CALUDE_ab_length_is_eleven_l2699_269935

-- Define the triangle structures
structure Triangle :=
  (a b c : ℝ)

-- Define isosceles property
def isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.a = t.c

-- Define perimeter
def perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

theorem ab_length_is_eleven 
  (ABC CBD : Triangle)
  (ABC_isosceles : isIsosceles ABC)
  (CBD_isosceles : isIsosceles CBD)
  (CBD_perimeter : perimeter CBD = 24)
  (ABC_perimeter : perimeter ABC = 25)
  (BD_length : CBD.c = 10) :
  ABC.c = 11 := by
  sorry

end NUMINAMATH_CALUDE_ab_length_is_eleven_l2699_269935


namespace NUMINAMATH_CALUDE_f_properties_l2699_269906

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Theorem for monotonicity intervals and extreme values
theorem f_properties :
  (∀ x < -1, HasDerivAt f (f x) x ∧ 0 < (deriv f x)) ∧
  (∀ x ∈ Set.Ioo (-1) 1, HasDerivAt f (f x) x ∧ (deriv f x) < 0) ∧
  (∀ x > 1, HasDerivAt f (f x) x ∧ 0 < (deriv f x)) ∧
  (∀ x ∈ Set.Icc (-3) 2, f x ≥ -18) ∧
  (∀ x ∈ Set.Icc (-3) 2, f x ≤ 2) ∧
  (f (-3) = -18) ∧
  (f (-1) = 2 ∨ f 2 = 2) := by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_f_properties_l2699_269906


namespace NUMINAMATH_CALUDE_homework_percentage_l2699_269951

theorem homework_percentage (total_angle : ℝ) (less_than_one_hour_angle : ℝ) :
  total_angle = 360 →
  less_than_one_hour_angle = 90 →
  (1 - less_than_one_hour_angle / total_angle) * 100 = 75 := by
  sorry

end NUMINAMATH_CALUDE_homework_percentage_l2699_269951


namespace NUMINAMATH_CALUDE_remainder_div_six_l2699_269967

theorem remainder_div_six (n : ℕ) (h : n % 12 = 8) : n % 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_div_six_l2699_269967


namespace NUMINAMATH_CALUDE_brochure_printing_problem_l2699_269986

/-- Represents the number of pages printed for the spreads for which the press prints a block of 4 ads -/
def pages_per_ad_block : ℕ := by sorry

theorem brochure_printing_problem :
  let single_page_spreads : ℕ := 20
  let double_page_spreads : ℕ := 2 * single_page_spreads
  let pages_per_brochure : ℕ := 5
  let total_brochures : ℕ := 25
  let total_pages : ℕ := total_brochures * pages_per_brochure
  let pages_from_double_spreads : ℕ := double_page_spreads * 2
  let remaining_pages : ℕ := total_pages - pages_from_double_spreads
  let unused_single_spreads : ℕ := single_page_spreads - remaining_pages
  pages_per_ad_block = unused_single_spreads := by sorry

end NUMINAMATH_CALUDE_brochure_printing_problem_l2699_269986


namespace NUMINAMATH_CALUDE_square_ratio_l2699_269963

theorem square_ratio (n m : ℝ) :
  (∃ a : ℝ, 9 * x^2 + n * x + 1 = (3 * x + a)^2) →
  (∃ b : ℝ, 4 * y^2 + 12 * y + m = (2 * y + b)^2) →
  n > 0 →
  n / m = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_square_ratio_l2699_269963


namespace NUMINAMATH_CALUDE_degree_of_our_monomial_l2699_269982

/-- The degree of a monomial is the sum of the exponents of its variables. -/
def degree_of_monomial (m : String) : ℕ :=
  sorry

/-- The monomial -2/5 * x^2 * y -/
def our_monomial : String := "-2/5x^2y"

theorem degree_of_our_monomial :
  degree_of_monomial our_monomial = 3 := by
  sorry

end NUMINAMATH_CALUDE_degree_of_our_monomial_l2699_269982


namespace NUMINAMATH_CALUDE_copper_ion_test_l2699_269904

theorem copper_ion_test (total_beakers : ℕ) (copper_beakers : ℕ) (total_drops : ℕ) (non_copper_tested : ℕ) :
  total_beakers = 22 →
  copper_beakers = 8 →
  total_drops = 45 →
  non_copper_tested = 7 →
  (copper_beakers + non_copper_tested) * 3 = total_drops :=
by sorry

end NUMINAMATH_CALUDE_copper_ion_test_l2699_269904


namespace NUMINAMATH_CALUDE_soccer_lineup_combinations_l2699_269985

def team_size : ℕ := 16
def non_goalkeeper : ℕ := 1
def lineup_positions : ℕ := 4

theorem soccer_lineup_combinations :
  (team_size - non_goalkeeper) *
  (team_size - 1) *
  (team_size - 2) *
  (team_size - 3) = 42210 :=
by sorry

end NUMINAMATH_CALUDE_soccer_lineup_combinations_l2699_269985


namespace NUMINAMATH_CALUDE_probability_theorem_l2699_269911

def family_A_size : ℕ := 5
def family_B_size : ℕ := 3
def total_girls : ℕ := 5
def total_boys : ℕ := 3

def probability_at_least_one_family_all_girls : ℚ :=
  11 / 56

theorem probability_theorem :
  let total_children := family_A_size + family_B_size
  probability_at_least_one_family_all_girls = 11 / 56 :=
by sorry

end NUMINAMATH_CALUDE_probability_theorem_l2699_269911


namespace NUMINAMATH_CALUDE_domain_transformation_l2699_269924

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x+1)
def domain_f_plus_one : Set ℝ := Set.Icc (-1) 0

-- Define the domain of f(2x)
def domain_f_double : Set ℝ := Set.Ico 0 (1/2)

-- Theorem statement
theorem domain_transformation (h : ∀ x ∈ domain_f_plus_one, f (x + 1) = f (x + 1)) :
  ∀ x ∈ domain_f_double, f (2 * x) = f (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_domain_transformation_l2699_269924


namespace NUMINAMATH_CALUDE_subset_divisibility_l2699_269998

theorem subset_divisibility (n : ℕ) (k : ℕ) (h1 : n = 1000) (h2 : k = 500) :
  ¬(11 ∣ Nat.choose n k) ∧ 
  (3 ∣ Nat.choose n k) ∧ 
  (5 ∣ Nat.choose n k) ∧ 
  (13 ∣ Nat.choose n k) ∧ 
  (17 ∣ Nat.choose n k) := by
  sorry

end NUMINAMATH_CALUDE_subset_divisibility_l2699_269998


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l2699_269972

/-- The area of a square with diagonal length 20 is 200 -/
theorem square_area_from_diagonal : 
  ∀ s : ℝ, s > 0 → s * s * 2 = 20 * 20 → s * s = 200 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l2699_269972


namespace NUMINAMATH_CALUDE_mrs_awesome_class_size_l2699_269989

theorem mrs_awesome_class_size :
  ∀ (total_jelly_beans : ℕ) (leftover_jelly_beans : ℕ) (boy_girl_difference : ℕ),
    total_jelly_beans = 480 →
    leftover_jelly_beans = 5 →
    boy_girl_difference = 3 →
    ∃ (girls : ℕ) (boys : ℕ),
      girls + boys = 31 ∧
      boys = girls + boy_girl_difference ∧
      girls * girls + boys * boys = total_jelly_beans - leftover_jelly_beans :=
by sorry

end NUMINAMATH_CALUDE_mrs_awesome_class_size_l2699_269989


namespace NUMINAMATH_CALUDE_bike_wheel_rotations_l2699_269946

theorem bike_wheel_rotations 
  (rotations_per_block : ℕ) 
  (min_blocks : ℕ) 
  (remaining_rotations : ℕ) 
  (h1 : rotations_per_block = 200)
  (h2 : min_blocks = 8)
  (h3 : remaining_rotations = 1000) :
  min_blocks * rotations_per_block - remaining_rotations = 600 :=
by
  sorry

end NUMINAMATH_CALUDE_bike_wheel_rotations_l2699_269946


namespace NUMINAMATH_CALUDE_total_marbles_eq_4_9r_l2699_269949

/-- The total number of marbles in a bag given the number of red marbles -/
def total_marbles (r : ℝ) : ℝ :=
  let blue := 1.3 * r
  let green := 2 * blue
  r + blue + green

/-- Theorem stating that the total number of marbles is 4.9 times the number of red marbles -/
theorem total_marbles_eq_4_9r (r : ℝ) : total_marbles r = 4.9 * r := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_eq_4_9r_l2699_269949


namespace NUMINAMATH_CALUDE_fraction_product_theorem_l2699_269913

theorem fraction_product_theorem : 
  (7 / 4 : ℚ) * (8 / 16 : ℚ) * (21 / 14 : ℚ) * (15 / 25 : ℚ) * 
  (28 / 21 : ℚ) * (20 / 40 : ℚ) * (49 / 28 : ℚ) * (25 / 50 : ℚ) = 147 / 320 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_theorem_l2699_269913


namespace NUMINAMATH_CALUDE_interesting_2018_gon_after_marked_removal_l2699_269931

/-- A convex polygon with n vertices --/
structure ConvexPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  is_convex : sorry

/-- A coloring of vertices in two colors --/
def Coloring (n : ℕ) := Fin n → Bool

/-- The sum of angles at vertices of a given color in a polygon --/
def sumAngles (p : ConvexPolygon n) (c : Coloring n) (color : Bool) : ℝ := sorry

/-- A polygon is interesting if the sum of angles of one color equals the sum of angles of the other color --/
def isInteresting (p : ConvexPolygon n) (c : Coloring n) : Prop :=
  sumAngles p c true = sumAngles p c false

/-- Remove a vertex from a polygon --/
def removeVertex (p : ConvexPolygon (n + 1)) (i : Fin (n + 1)) : ConvexPolygon n := sorry

/-- The theorem to be proved --/
theorem interesting_2018_gon_after_marked_removal
  (p : ConvexPolygon 2019)
  (marked : Fin 2019)
  (h : ∀ (i : Fin 2019), i ≠ marked → ∃ (c : Coloring 2018), isInteresting (removeVertex p i) c) :
  ∃ (c : Coloring 2018), isInteresting (removeVertex p marked) c :=
sorry

end NUMINAMATH_CALUDE_interesting_2018_gon_after_marked_removal_l2699_269931


namespace NUMINAMATH_CALUDE_program_output_l2699_269933

def S : ℕ → ℕ
  | 0 => 1
  | (n + 1) => S n + (2 * (n + 1) - 1)

theorem program_output :
  (S 1 = 2) ∧ (S 2 = 5) ∧ (S 3 = 10) := by
  sorry

end NUMINAMATH_CALUDE_program_output_l2699_269933


namespace NUMINAMATH_CALUDE_food_drive_problem_l2699_269974

theorem food_drive_problem (total_students : ℕ) (cans_per_first_group : ℕ) (non_collecting_students : ℕ) (last_group_students : ℕ) (total_cans : ℕ) :
  total_students = 30 →
  cans_per_first_group = 12 →
  non_collecting_students = 2 →
  last_group_students = 13 →
  total_cans = 232 →
  (total_students / 2) * cans_per_first_group + 0 * non_collecting_students + last_group_students * ((total_cans - (total_students / 2) * cans_per_first_group) / last_group_students) = total_cans →
  (total_cans - (total_students / 2) * cans_per_first_group) / last_group_students = 4 :=
by sorry

end NUMINAMATH_CALUDE_food_drive_problem_l2699_269974


namespace NUMINAMATH_CALUDE_permutations_of_sees_l2699_269941

theorem permutations_of_sees (n : ℕ) (a b : ℕ) (h1 : n = 4) (h2 : a = 2) (h3 : b = 2) :
  (n.factorial) / (a.factorial * b.factorial) = 6 :=
sorry

end NUMINAMATH_CALUDE_permutations_of_sees_l2699_269941


namespace NUMINAMATH_CALUDE_wizard_hat_theorem_l2699_269961

/-- Represents a strategy for the wizard hat problem -/
def Strategy : Type := Unit

/-- Represents the outcome of applying a strategy -/
def Outcome (n : ℕ) : Type := Fin n → Bool

/-- A wizard can see hats in front but not their own -/
axiom can_see_forward (n : ℕ) (i : Fin n) : ∀ j : Fin n, i < j → Prop

/-- Each wizard says a unique number between 1 and 1001 -/
axiom unique_numbers (n : ℕ) (outcome : Outcome n) : 
  ∀ i j : Fin n, i ≠ j → outcome i ≠ outcome j

/-- Wizards speak from back to front -/
axiom speak_order (n : ℕ) (i j : Fin n) : i < j → Prop

/-- Applying a strategy produces an outcome -/
def apply_strategy (n : ℕ) (s : Strategy) : Outcome n := sorry

/-- Counts the number of correct identifications in an outcome -/
def count_correct (n : ℕ) (outcome : Outcome n) : ℕ := sorry

theorem wizard_hat_theorem (n : ℕ) (h : n > 1000) :
  ∃ (s : Strategy), 
    (count_correct n (apply_strategy n s) > 500) ∧ 
    (count_correct n (apply_strategy n s) ≥ 999) := by
  sorry

end NUMINAMATH_CALUDE_wizard_hat_theorem_l2699_269961


namespace NUMINAMATH_CALUDE_inequality_solution_l2699_269965

theorem inequality_solution (x : ℝ) : 
  (1 / (x * (x + 1)) - 1 / ((x + 1) * (x + 2)) < 1 / 4) ↔ 
  (x < -2 ∨ (-1 < x ∧ x < 0) ∨ 1 < x) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2699_269965


namespace NUMINAMATH_CALUDE_min_value_theorem_l2699_269996

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2/x + 1/y = 1) :
  x + 2*y ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2699_269996


namespace NUMINAMATH_CALUDE_statements_are_false_l2699_269960

theorem statements_are_false : 
  (¬ ∀ (x : ℚ), ∃ (y : ℚ), (x < y ∧ y < -x) ∨ (-x < y ∧ y < x)) ∧ 
  (¬ ∀ (x : ℚ), x ≠ 0 → ∃ (y : ℚ), (x < y ∧ y < x⁻¹) ∨ (x⁻¹ < y ∧ y < x)) :=
by sorry

end NUMINAMATH_CALUDE_statements_are_false_l2699_269960


namespace NUMINAMATH_CALUDE_sum_of_50th_row_l2699_269978

/-- Represents the sum of numbers in the nth row of the triangular array -/
def f (n : ℕ) : ℕ :=
  2^n - 2 * (n * (n + 1) / 2)

/-- The triangular array property -/
axiom triangular_array_property (n : ℕ) :
  f n = 2 * f (n - 1) + n * (n + 1)

/-- Theorem: The sum of numbers in the 50th row is 2^50 - 2550 -/
theorem sum_of_50th_row :
  f 50 = 2^50 - 2550 := by sorry

end NUMINAMATH_CALUDE_sum_of_50th_row_l2699_269978


namespace NUMINAMATH_CALUDE_sum_of_differences_equals_6999993_l2699_269927

/-- Calculates the local value of a digit in a number based on its position --/
def localValue (digit : ℕ) (position : ℕ) : ℕ :=
  digit * (10 ^ position)

/-- Calculates the difference between local value and face value for a digit --/
def valueDifference (digit : ℕ) (position : ℕ) : ℕ :=
  localValue digit position - digit

/-- The numeral we're working with --/
def numeral : ℕ := 657932657

/-- Positions of 7 in the numeral (0-indexed from right) --/
def sevenPositions : List ℕ := [0, 6]

/-- Sum of differences between local and face values for all 7s in the numeral --/
def sumOfDifferences : ℕ :=
  (sevenPositions.map (valueDifference 7)).sum

theorem sum_of_differences_equals_6999993 :
  sumOfDifferences = 6999993 := by sorry

end NUMINAMATH_CALUDE_sum_of_differences_equals_6999993_l2699_269927


namespace NUMINAMATH_CALUDE_emily_jumps_in_75_seconds_l2699_269975

/-- Emily's jumping rate in jumps per second -/
def jumping_rate : ℚ := 52 / 60

/-- The number of jumps Emily makes in a given time -/
def jumps (time : ℚ) : ℚ := jumping_rate * time

theorem emily_jumps_in_75_seconds : 
  jumps 75 = 65 := by sorry

end NUMINAMATH_CALUDE_emily_jumps_in_75_seconds_l2699_269975


namespace NUMINAMATH_CALUDE_jack_christina_lindy_meeting_l2699_269925

/-- The problem of Jack, Christina, and Lindy meeting --/
theorem jack_christina_lindy_meeting
  (initial_distance : ℝ)
  (jack_speed : ℝ)
  (christina_speed : ℝ)
  (lindy_speed : ℝ)
  (h1 : initial_distance = 360)
  (h2 : jack_speed = 5)
  (h3 : christina_speed = 7)
  (h4 : lindy_speed = 12)
  (h5 : jack_speed > 0)
  (h6 : christina_speed > 0)
  (h7 : lindy_speed > jack_speed + christina_speed) :
  let meeting_time := initial_distance / (jack_speed + christina_speed)
  lindy_speed * meeting_time = initial_distance :=
sorry

end NUMINAMATH_CALUDE_jack_christina_lindy_meeting_l2699_269925


namespace NUMINAMATH_CALUDE_allen_blocks_count_l2699_269942

/-- The number of blocks for each color -/
def blocks_per_color : ℕ := 7

/-- The number of colors used -/
def number_of_colors : ℕ := 7

/-- The total number of blocks -/
def total_blocks : ℕ := blocks_per_color * number_of_colors

theorem allen_blocks_count : total_blocks = 49 := by
  sorry

end NUMINAMATH_CALUDE_allen_blocks_count_l2699_269942


namespace NUMINAMATH_CALUDE_knight_count_l2699_269909

def is_correct_arrangement (knights liars : ℕ) : Prop :=
  knights + liars = 2019 ∧
  knights > 2 * liars ∧
  knights ≤ 2 * liars + 1

theorem knight_count : ∃ (knights liars : ℕ), 
  is_correct_arrangement knights liars ∧ knights = 1346 := by
  sorry

end NUMINAMATH_CALUDE_knight_count_l2699_269909


namespace NUMINAMATH_CALUDE_marys_blueberries_l2699_269983

theorem marys_blueberries (apples oranges total_left : ℕ) (h1 : apples = 14) (h2 : oranges = 9) (h3 : total_left = 26) :
  ∃ blueberries : ℕ, blueberries = 5 ∧ total_left = (apples - 1) + (oranges - 1) + (blueberries - 1) :=
by
  sorry

end NUMINAMATH_CALUDE_marys_blueberries_l2699_269983


namespace NUMINAMATH_CALUDE_prob_A_value_l2699_269910

/-- The probability that person A speaks the truth -/
def prob_A : ℝ := sorry

/-- The probability that person B speaks the truth -/
def prob_B : ℝ := 0.6

/-- The probability that both A and B speak the truth simultaneously -/
def prob_A_and_B : ℝ := 0.48

/-- The events of A and B speaking the truth are independent -/
axiom independence : prob_A_and_B = prob_A * prob_B

theorem prob_A_value : prob_A = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_prob_A_value_l2699_269910


namespace NUMINAMATH_CALUDE_system_solution_l2699_269984

theorem system_solution (x y : ℝ) : 
  (4 * (x - y) = 8 - 3 * y) ∧ 
  (x / 2 + y / 3 = 1) ↔ 
  (x = 2 ∧ y = 0) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2699_269984


namespace NUMINAMATH_CALUDE_sum_of_radii_is_14_l2699_269936

-- Define the circle with center C
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define the condition of being tangent to positive x- and y-axes
def tangentToAxes (c : Circle) : Prop :=
  c.center.1 = c.radius ∧ c.center.2 = c.radius

-- Define the condition of being externally tangent to another circle
def externallyTangent (c1 c2 : Circle) : Prop :=
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 = (c1.radius + c2.radius)^2

-- Theorem statement
theorem sum_of_radii_is_14 :
  ∃ (c1 c2 : Circle),
    tangentToAxes c1 ∧
    tangentToAxes c2 ∧
    c1.center ≠ c2.center ∧
    externallyTangent c1 { center := (5, 0), radius := 2 } ∧
    externallyTangent c2 { center := (5, 0), radius := 2 } ∧
    c1.radius + c2.radius = 14 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_radii_is_14_l2699_269936


namespace NUMINAMATH_CALUDE_inequality_proof_l2699_269920

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 1) :
  a * Real.sqrt b + b * Real.sqrt c + c * Real.sqrt a ≤ 1 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2699_269920


namespace NUMINAMATH_CALUDE_max_programs_max_programs_achievable_max_programs_optimal_l2699_269968

theorem max_programs (n : ℕ) : n ≤ 4 :=
  sorry

theorem max_programs_achievable : ∃ (P : Fin 4 → Finset (Fin 12)),
  (∀ i : Fin 4, (P i).card = 6) ∧
  (∀ i j : Fin 4, i ≠ j → (P i ∩ P j).card ≤ 2) :=
  sorry

theorem max_programs_optimal :
  ¬∃ (P : Fin 5 → Finset (Fin 12)),
    (∀ i : Fin 5, (P i).card = 6) ∧
    (∀ i j : Fin 5, i ≠ j → (P i ∩ P j).card ≤ 2) :=
  sorry

end NUMINAMATH_CALUDE_max_programs_max_programs_achievable_max_programs_optimal_l2699_269968


namespace NUMINAMATH_CALUDE_decrease_in_profit_for_given_scenario_l2699_269954

/-- Represents the financial data of a textile manufacturing firm -/
structure TextileFirm where
  total_looms : ℕ
  sales_value : ℕ
  manufacturing_expenses : ℕ
  establishment_charges : ℕ

/-- Calculates the decrease in profit when one loom is idle for a month -/
def decrease_in_profit (firm : TextileFirm) : ℕ :=
  let sales_per_loom := firm.sales_value / firm.total_looms
  let expenses_per_loom := firm.manufacturing_expenses / firm.total_looms
  sales_per_loom - expenses_per_loom

/-- Theorem stating the decrease in profit for the given scenario -/
theorem decrease_in_profit_for_given_scenario :
  let firm := TextileFirm.mk 125 500000 150000 75000
  decrease_in_profit firm = 2800 := by
  sorry

#eval decrease_in_profit (TextileFirm.mk 125 500000 150000 75000)

end NUMINAMATH_CALUDE_decrease_in_profit_for_given_scenario_l2699_269954


namespace NUMINAMATH_CALUDE_backpack_store_theorem_l2699_269979

/-- Represents the backpack types --/
inductive BackpackType
| A
| B

/-- Represents a purchasing plan --/
structure PurchasePlan where
  typeA : ℕ
  typeB : ℕ

/-- Represents the backpack pricing and inventory --/
structure BackpackStore where
  sellingPriceA : ℕ
  sellingPriceB : ℕ
  costPriceA : ℕ
  costPriceB : ℕ
  inventory : PurchasePlan
  givenAwayA : ℕ
  givenAwayB : ℕ

/-- The main theorem to prove --/
theorem backpack_store_theorem (store : BackpackStore) : 
  (store.sellingPriceA = store.sellingPriceB + 12) →
  (2 * store.sellingPriceA + 3 * store.sellingPriceB = 264) →
  (store.inventory.typeA + store.inventory.typeB = 100) →
  (store.costPriceA * store.inventory.typeA + store.costPriceB * store.inventory.typeB ≤ 4550) →
  (store.inventory.typeA > 52) →
  (store.costPriceA = 50) →
  (store.costPriceB = 40) →
  (store.givenAwayA + store.givenAwayB = 5) →
  (store.sellingPriceA * (store.inventory.typeA - store.givenAwayA) + 
   store.sellingPriceB * (store.inventory.typeB - store.givenAwayB) - 
   store.costPriceA * store.inventory.typeA - 
   store.costPriceB * store.inventory.typeB = 658) →
  (store.sellingPriceA = 60 ∧ store.sellingPriceB = 48) ∧
  ((store.inventory.typeA = 53 ∧ store.inventory.typeB = 47) ∨
   (store.inventory.typeA = 54 ∧ store.inventory.typeB = 46) ∨
   (store.inventory.typeA = 55 ∧ store.inventory.typeB = 45)) ∧
  (store.givenAwayA = 1 ∧ store.givenAwayB = 4) :=
by sorry


end NUMINAMATH_CALUDE_backpack_store_theorem_l2699_269979


namespace NUMINAMATH_CALUDE_infinitely_many_not_n_attainable_all_except_seven_3_attainable_l2699_269957

/-- Definition of an n-admissible sequence -/
def IsNAdmissibleSequence (n : ℕ) (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧
  (∀ k, k > 0 →
    ((a (2*k) = a (2*k-1) + 2 ∨ a (2*k) = a (2*k-1) + n) ∧
     (a (2*k+1) = 2 * a (2*k) ∨ a (2*k+1) = n * a (2*k))) ∨
    ((a (2*k) = 2 * a (2*k-1) ∨ a (2*k) = n * a (2*k-1)) ∧
     (a (2*k+1) = a (2*k) + 2 ∨ a (2*k+1) = a (2*k) + n)))

/-- Definition of n-attainable number -/
def IsNAttainable (n : ℕ) (m : ℕ) : Prop :=
  m > 1 ∧ ∃ a, IsNAdmissibleSequence n a ∧ ∃ k, a k = m

/-- There are infinitely many positive integers not n-attainable for n > 8 -/
theorem infinitely_many_not_n_attainable (n : ℕ) (hn : n > 8) :
  ∃ S : Set ℕ, Set.Infinite S ∧ ∀ m ∈ S, ¬IsNAttainable n m :=
sorry

/-- All positive integers except 7 are 3-attainable -/
theorem all_except_seven_3_attainable :
  ∀ m : ℕ, m > 0 ∧ m ≠ 7 → IsNAttainable 3 m :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_not_n_attainable_all_except_seven_3_attainable_l2699_269957


namespace NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_set_l2699_269992

/-- Given a quadratic equation ax² + bx + c = 0 with a > 0 and no real roots,
    the solution set of ax² + bx + c < 0 is empty. -/
theorem quadratic_inequality_empty_solution_set
  (a b c : ℝ) 
  (h_a_pos : a > 0)
  (h_no_roots : ∀ x : ℝ, a * x^2 + b * x + c ≠ 0) :
  {x : ℝ | a * x^2 + b * x + c < 0} = ∅ :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_set_l2699_269992


namespace NUMINAMATH_CALUDE_height_of_d_l2699_269932

theorem height_of_d (h_abc : ℝ) (h_abcd : ℝ) 
  (avg_abc : (h_abc + h_abc + h_abc) / 3 = 130)
  (avg_abcd : (h_abc + h_abc + h_abc + h_abcd) / 4 = 126) :
  h_abcd = 114 := by
  sorry

end NUMINAMATH_CALUDE_height_of_d_l2699_269932


namespace NUMINAMATH_CALUDE_oranges_picked_total_l2699_269973

/-- The number of oranges Mary picked -/
def mary_oranges : ℕ := 14

/-- The number of oranges Jason picked -/
def jason_oranges : ℕ := 41

/-- The total number of oranges picked -/
def total_oranges : ℕ := mary_oranges + jason_oranges

theorem oranges_picked_total :
  total_oranges = 55 := by sorry

end NUMINAMATH_CALUDE_oranges_picked_total_l2699_269973


namespace NUMINAMATH_CALUDE_polynomial_expansion_l2699_269940

theorem polynomial_expansion (x : ℝ) :
  (5 * x - 3) * (2 * x^2 + 4 * x + 1) = 10 * x^3 + 14 * x^2 - 7 * x - 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l2699_269940


namespace NUMINAMATH_CALUDE_function_value_at_two_l2699_269990

/-- Given a function f(x) = ax^5 + bx^3 - x + 2 where a and b are constants,
    and f(-2) = 5, prove that f(2) = -1 -/
theorem function_value_at_two (a b : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^5 + b * x^3 - x + 2)
  (h2 : f (-2) = 5) : f 2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_two_l2699_269990


namespace NUMINAMATH_CALUDE_sphere_radius_is_60_37_l2699_269939

/-- A triangular pyramid with perpendicular lateral edges and a sphere touching all lateral faces -/
structure PerpendicularPyramid where
  /-- The side lengths of the triangular base -/
  base_side_1 : ℝ
  base_side_2 : ℝ
  base_side_3 : ℝ
  /-- The radius of the sphere touching all lateral faces -/
  sphere_radius : ℝ
  /-- The lateral edges are pairwise perpendicular -/
  lateral_edges_perpendicular : True
  /-- The center of the sphere lies on the base -/
  sphere_center_on_base : True
  /-- The base side lengths satisfy the given conditions -/
  base_side_1_sq : base_side_1^2 = 61
  base_side_2_sq : base_side_2^2 = 52
  base_side_3_sq : base_side_3^2 = 41

/-- The theorem stating that the radius of the sphere is 60/37 -/
theorem sphere_radius_is_60_37 (p : PerpendicularPyramid) : p.sphere_radius = 60 / 37 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_is_60_37_l2699_269939


namespace NUMINAMATH_CALUDE_beatrix_pages_l2699_269952

theorem beatrix_pages (beatrix cristobal : ℕ) 
  (h1 : cristobal = 3 * beatrix + 15)
  (h2 : cristobal = beatrix + 1423) : 
  beatrix = 704 := by
sorry

end NUMINAMATH_CALUDE_beatrix_pages_l2699_269952


namespace NUMINAMATH_CALUDE_fourth_power_equals_sixteenth_l2699_269976

theorem fourth_power_equals_sixteenth (n : ℝ) : (1/4 : ℝ)^n = 0.0625 → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_equals_sixteenth_l2699_269976


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l2699_269934

theorem vector_sum_magnitude 
  (a b : ℝ × ℝ) 
  (h1 : a • b = 0) 
  (h2 : ‖a‖ = 2) 
  (h3 : ‖b‖ = 1) : 
  ‖a + 2 • b‖ = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l2699_269934


namespace NUMINAMATH_CALUDE_polar_to_cartesian_line_l2699_269988

/-- The polar equation r = 1 / (sin θ + cos θ) represents a line in Cartesian coordinates -/
theorem polar_to_cartesian_line :
  ∀ (θ : ℝ) (r : ℝ), r = 1 / (Real.sin θ + Real.cos θ) →
  ∃ (x y : ℝ), x = r * Real.cos θ ∧ y = r * Real.sin θ ∧ x + y = 1 :=
by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_line_l2699_269988


namespace NUMINAMATH_CALUDE_max_enclosed_area_l2699_269981

/-- Represents an infinite chessboard -/
structure InfiniteChessboard where

/-- Represents a closed non-self-intersecting polygonal line on the chessboard -/
structure PolygonalLine where
  chessboard : InfiniteChessboard
  is_closed : Bool
  is_non_self_intersecting : Bool
  along_cell_sides : Bool

/-- Represents the area enclosed by a polygonal line -/
def EnclosedArea (line : PolygonalLine) : ℕ := sorry

/-- Counts the number of black cells inside a polygonal line -/
def BlackCellsCount (line : PolygonalLine) : ℕ := sorry

/-- Theorem stating the maximum area enclosed by a polygonal line -/
theorem max_enclosed_area (line : PolygonalLine) (k : ℕ) 
  (h1 : line.is_closed = true)
  (h2 : line.is_non_self_intersecting = true)
  (h3 : line.along_cell_sides = true)
  (h4 : BlackCellsCount line = k) :
  EnclosedArea line ≤ 4 * k + 1 := by sorry

end NUMINAMATH_CALUDE_max_enclosed_area_l2699_269981


namespace NUMINAMATH_CALUDE_solution_set_l2699_269937

theorem solution_set (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let S := {(x, y, z) : ℝ × ℝ × ℝ | 
    a * x + b * y = (x - y)^2 ∧
    b * y + c * z = (y - z)^2 ∧
    c * z + a * x = (z - x)^2}
  S = {(0, 0, 0), (a, 0, 0), (0, b, 0), (0, 0, c)} := by
sorry

end NUMINAMATH_CALUDE_solution_set_l2699_269937


namespace NUMINAMATH_CALUDE_largest_solution_is_four_l2699_269997

theorem largest_solution_is_four : 
  ∃ c : ℝ, (3*c + 4)*(c - 2) = 9*c ∧ 
  (∀ x : ℝ, (3*x + 4)*(x - 2) = 9*x → x ≤ c) ∧ 
  c = 4 := by
sorry

end NUMINAMATH_CALUDE_largest_solution_is_four_l2699_269997


namespace NUMINAMATH_CALUDE_union_equal_M_l2699_269964

def M : Set Char := {'a', 'b', 'c', 'd', 'e'}
def N : Set Char := {'b', 'd', 'e'}

theorem union_equal_M : M ∪ N = M := by sorry

end NUMINAMATH_CALUDE_union_equal_M_l2699_269964


namespace NUMINAMATH_CALUDE_bob_remaining_corn_l2699_269916

/-- Represents the amount of corn in bushels and ears -/
structure CornAmount where
  bushels : ℚ
  ears : ℕ

/-- Calculates the remaining corn after giving some away -/
def remaining_corn (initial : CornAmount) (given_away : List CornAmount) : ℕ :=
  sorry

/-- Theorem stating that Bob has 357 ears of corn left -/
theorem bob_remaining_corn :
  let initial := CornAmount.mk 50 0
  let given_away := [
    CornAmount.mk 8 0,    -- Terry
    CornAmount.mk 3 0,    -- Jerry
    CornAmount.mk 12 0,   -- Linda
    CornAmount.mk 0 21    -- Stacy
  ]
  let ears_per_bushel := 14
  remaining_corn initial given_away = 357 := by
  sorry

end NUMINAMATH_CALUDE_bob_remaining_corn_l2699_269916


namespace NUMINAMATH_CALUDE_second_year_percentage_approx_l2699_269947

def numeric_methods_students : ℕ := 240
def automatic_control_students : ℕ := 423
def both_subjects_students : ℕ := 134
def total_faculty_students : ℕ := 663

def second_year_students : ℕ := numeric_methods_students + automatic_control_students - both_subjects_students

def percentage_second_year : ℚ := (second_year_students : ℚ) / (total_faculty_students : ℚ) * 100

theorem second_year_percentage_approx :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ |percentage_second_year - 79.79| < ε :=
sorry

end NUMINAMATH_CALUDE_second_year_percentage_approx_l2699_269947


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2699_269938

theorem quadratic_equation_roots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 + 4*x₁ = 0 ∧ x₂^2 + 4*x₂ = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2699_269938


namespace NUMINAMATH_CALUDE_cos_a_sin_b_value_l2699_269958

theorem cos_a_sin_b_value (A B : Real) (hA : 0 < A ∧ A < Real.pi / 2) (hB : 0 < B ∧ B < Real.pi / 2)
  (h : (4 + Real.tan A ^ 2) * (5 + Real.tan B ^ 2) = Real.sqrt 320 * Real.tan A * Real.tan B) :
  Real.cos A * Real.sin B = Real.sqrt 6 / 6 := by
  sorry

end NUMINAMATH_CALUDE_cos_a_sin_b_value_l2699_269958


namespace NUMINAMATH_CALUDE_fraction_difference_equals_one_minus_two_over_x_l2699_269914

theorem fraction_difference_equals_one_minus_two_over_x 
  (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = x * y) : 
  1 / x - 1 / y = 1 - 2 / x :=
by sorry

end NUMINAMATH_CALUDE_fraction_difference_equals_one_minus_two_over_x_l2699_269914
