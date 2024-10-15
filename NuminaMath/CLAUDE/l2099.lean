import Mathlib

namespace NUMINAMATH_CALUDE_monomial_sum_condition_l2099_209995

/-- 
If the sum of the monomials x^2 * y^(m+2) and x^n * y is still a monomial, 
then m + n = 1.
-/
theorem monomial_sum_condition (m n : ℤ) : 
  (∃ (x y : ℚ), x ≠ 0 ∧ y ≠ 0 ∧ ∃ (k : ℚ), x^2 * y^(m+2) + x^n * y = k * (x^2 * y^(m+2))) → 
  m + n = 1 := by
sorry

end NUMINAMATH_CALUDE_monomial_sum_condition_l2099_209995


namespace NUMINAMATH_CALUDE_arabella_dance_steps_l2099_209928

/-- Arabella's dance step learning problem -/
theorem arabella_dance_steps (T₁ T₂ T₃ : ℚ) 
  (h1 : T₁ = 30)
  (h2 : T₃ = T₁ + T₂)
  (h3 : T₁ + T₂ + T₃ = 90) :
  T₂ / T₁ = 1/2 := by
  sorry

#check arabella_dance_steps

end NUMINAMATH_CALUDE_arabella_dance_steps_l2099_209928


namespace NUMINAMATH_CALUDE_chickens_and_rabbits_equation_l2099_209982

/-- Represents the number of chickens in the cage -/
def chickens : ℕ := sorry

/-- Represents the number of rabbits in the cage -/
def rabbits : ℕ := sorry

/-- The total number of heads in the cage -/
def total_heads : ℕ := 16

/-- The total number of feet in the cage -/
def total_feet : ℕ := 44

/-- Theorem stating that the system of equations correctly represents the problem -/
theorem chickens_and_rabbits_equation :
  (chickens + rabbits = total_heads) ∧
  (2 * chickens + 4 * rabbits = total_feet) :=
sorry

end NUMINAMATH_CALUDE_chickens_and_rabbits_equation_l2099_209982


namespace NUMINAMATH_CALUDE_triangle_equivalence_l2099_209976

/-- A triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The angles of a triangle -/
def Triangle.angles (t : Triangle) : ℝ × ℝ × ℝ := sorry

/-- The Nine-point circle of a triangle -/
def Triangle.ninePointCircle (t : Triangle) : Circle := sorry

/-- The Incircle of a triangle -/
def Triangle.incircle (t : Triangle) : Circle := sorry

/-- The Euler Line of a triangle -/
def Triangle.eulerLine (t : Triangle) : Line := sorry

/-- Check if a triangle is equilateral -/
def Triangle.isEquilateral (t : Triangle) : Prop := sorry

/-- Check if one of the angles is 60° -/
def Triangle.hasAngle60 (t : Triangle) : Prop := sorry

/-- Check if the angles are in arithmetic progression -/
def Triangle.anglesInArithmeticProgression (t : Triangle) : Prop := sorry

/-- Check if the common tangent to the Nine-point circle and Incircle is parallel to the Euler Line -/
def Triangle.commonTangentParallelToEulerLine (t : Triangle) : Prop := sorry

/-- Main theorem -/
theorem triangle_equivalence (t : Triangle) (h : ¬ t.isEquilateral) :
  t.hasAngle60 ↔ t.anglesInArithmeticProgression ∧ t.commonTangentParallelToEulerLine :=
sorry

end NUMINAMATH_CALUDE_triangle_equivalence_l2099_209976


namespace NUMINAMATH_CALUDE_factorize_difference_of_squares_l2099_209967

theorem factorize_difference_of_squares (a b : ℝ) : a^2 - 4*b^2 = (a + 2*b) * (a - 2*b) := by
  sorry

end NUMINAMATH_CALUDE_factorize_difference_of_squares_l2099_209967


namespace NUMINAMATH_CALUDE_suresh_work_hours_l2099_209946

theorem suresh_work_hours (suresh_rate ashutosh_rate : ℚ) 
  (ashutosh_remaining_time : ℚ) : 
  suresh_rate = 1 / 15 →
  ashutosh_rate = 1 / 25 →
  ashutosh_remaining_time = 10 →
  ∃ (suresh_time : ℚ), 
    suresh_time * suresh_rate + ashutosh_remaining_time * ashutosh_rate = 1 ∧
    suresh_time = 9 := by
sorry

end NUMINAMATH_CALUDE_suresh_work_hours_l2099_209946


namespace NUMINAMATH_CALUDE_trapezoid_side_length_l2099_209969

/-- Represents a trapezoid with an inscribed circle -/
structure TrapezoidWithInscribedCircle where
  /-- Distance from the center of the inscribed circle to one end of a non-parallel side -/
  distance1 : ℝ
  /-- Distance from the center of the inscribed circle to the other end of the same non-parallel side -/
  distance2 : ℝ

/-- Theorem: If the center of the inscribed circle in a trapezoid is at distances 5 and 12
    from the ends of one non-parallel side, then the length of that side is 13. -/
theorem trapezoid_side_length (t : TrapezoidWithInscribedCircle)
    (h1 : t.distance1 = 5)
    (h2 : t.distance2 = 12) :
    Real.sqrt (t.distance1^2 + t.distance2^2) = 13 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_side_length_l2099_209969


namespace NUMINAMATH_CALUDE_house_painting_cost_l2099_209994

/-- The total cost of painting a house -/
def total_cost (area : ℝ) (price_per_sqft : ℝ) : ℝ :=
  area * price_per_sqft

/-- Theorem: The total cost of painting a house with an area of 484 sq ft
    at a price of Rs. 20 per sq ft is Rs. 9680 -/
theorem house_painting_cost :
  total_cost 484 20 = 9680 := by
  sorry

end NUMINAMATH_CALUDE_house_painting_cost_l2099_209994


namespace NUMINAMATH_CALUDE_proposition_3_proposition_4_l2099_209908

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (belongs_to : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (plane_parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Line → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)
variable (line_perpendicular_plane : Line → Plane → Prop)

-- Proposition 3
theorem proposition_3 
  (α β : Plane) (a b : Line) :
  plane_perpendicular α β →
  intersect α β a →
  belongs_to b β →
  perpendicular a b →
  line_perpendicular_plane b α :=
sorry

-- Proposition 4
theorem proposition_4
  (α : Plane) (a b l : Line) :
  belongs_to a α →
  belongs_to b α →
  perpendicular l a →
  perpendicular l b →
  line_perpendicular_plane l α :=
sorry

end NUMINAMATH_CALUDE_proposition_3_proposition_4_l2099_209908


namespace NUMINAMATH_CALUDE_log2_derivative_l2099_209978

theorem log2_derivative (x : ℝ) (h : x > 0) : 
  deriv (λ x => Real.log x / Real.log 2) x = 1 / (x * Real.log 2) := by
  sorry

end NUMINAMATH_CALUDE_log2_derivative_l2099_209978


namespace NUMINAMATH_CALUDE_range_of_inequality_l2099_209979

/-- An even function that is monotonically decreasing on (-∞,0] -/
class EvenDecreasingFunction (f : ℝ → ℝ) : Prop where
  even : ∀ x, f x = f (-x)
  decreasing : ∀ {x y}, x ≤ y → y ≤ 0 → f y ≤ f x

/-- The theorem statement -/
theorem range_of_inequality (f : ℝ → ℝ) [EvenDecreasingFunction f] :
  {x : ℝ | f (2*x + 1) < f 3} = Set.Ioo (-2) 1 := by sorry

end NUMINAMATH_CALUDE_range_of_inequality_l2099_209979


namespace NUMINAMATH_CALUDE_preimage_of_one_two_l2099_209922

/-- The mapping f from R² to R² -/
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 - p.2)

/-- The theorem stating that (3/2, -1/2) is the preimage of (1, 2) under f -/
theorem preimage_of_one_two :
  f (3/2, -1/2) = (1, 2) := by sorry

end NUMINAMATH_CALUDE_preimage_of_one_two_l2099_209922


namespace NUMINAMATH_CALUDE_regular_polygon_with_120_degree_angles_has_6_sides_l2099_209953

theorem regular_polygon_with_120_degree_angles_has_6_sides :
  ∀ n : ℕ, n > 2 →
  (∀ θ : ℝ, θ = 120 → θ * n = 180 * (n - 2)) →
  n = 6 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_with_120_degree_angles_has_6_sides_l2099_209953


namespace NUMINAMATH_CALUDE_reversed_segment_appears_in_powers_of_two_l2099_209910

/-- The sequence of first digits of powers of 5 -/
def firstDigitsPowersOf5 : ℕ → ℕ :=
  λ n => (5^n : ℕ) % 10

/-- The sequence of first digits of powers of 2 -/
def firstDigitsPowersOf2 : ℕ → ℕ :=
  λ n => (2^n : ℕ) % 10

/-- Check if a list is a subsequence of another list -/
def isSubsequence {α : Type} [DecidableEq α] : List α → List α → Bool :=
  λ subseq seq => sorry

/-- Theorem: Any reversed segment of firstDigitsPowersOf5 appears in firstDigitsPowersOf2 -/
theorem reversed_segment_appears_in_powers_of_two :
  ∀ (start finish : ℕ),
    start ≤ finish →
    ∃ (n m : ℕ),
      isSubsequence
        ((List.range (finish - start + 1)).map (λ i => firstDigitsPowersOf5 (start + i))).reverse
        ((List.range (m - n + 1)).map (λ i => firstDigitsPowersOf2 (n + i))) = true :=
by
  sorry

end NUMINAMATH_CALUDE_reversed_segment_appears_in_powers_of_two_l2099_209910


namespace NUMINAMATH_CALUDE_coefficient_of_x_squared_l2099_209991

theorem coefficient_of_x_squared (x : ℝ) : 
  let expr := 2 * (x^2 - 5) + 6 * (3*x^2 - 2*x + 4) - 4 * (x^2 - 3*x)
  ∃ (a b c : ℝ), expr = 16 * x^2 + a * x + b * x + c :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_squared_l2099_209991


namespace NUMINAMATH_CALUDE_completing_square_l2099_209912

theorem completing_square (x : ℝ) : 
  (x^2 - 4*x - 3 = 0) ↔ ((x - 2)^2 = 7) := by
  sorry

end NUMINAMATH_CALUDE_completing_square_l2099_209912


namespace NUMINAMATH_CALUDE_positive_integer_solution_exists_l2099_209914

theorem positive_integer_solution_exists (a : ℤ) (h : a > 2) :
  ∃ x y : ℤ, x > 0 ∧ y > 0 ∧ x^2 - y^2 = a^2 := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_solution_exists_l2099_209914


namespace NUMINAMATH_CALUDE_homework_points_l2099_209913

theorem homework_points (total_points : ℕ) (test_quiz_ratio : ℕ) (quiz_homework_diff : ℕ)
  (h1 : total_points = 265)
  (h2 : test_quiz_ratio = 4)
  (h3 : quiz_homework_diff = 5) :
  ∃ (homework : ℕ), 
    homework + (homework + quiz_homework_diff) + test_quiz_ratio * (homework + quiz_homework_diff) = total_points ∧ 
    homework = 40 := by
  sorry

end NUMINAMATH_CALUDE_homework_points_l2099_209913


namespace NUMINAMATH_CALUDE_f_inequality_l2099_209964

/-- An odd function f: ℝ → ℝ with specific properties -/
def f_properties (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x, f (x - 4) = -f x) ∧
  (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 2 → f x < f y)

/-- Theorem stating the inequality for the given function -/
theorem f_inequality (f : ℝ → ℝ) (h : f_properties f) : 
  f (-1) < f 4 ∧ f 4 < f 3 := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l2099_209964


namespace NUMINAMATH_CALUDE_unique_polynomial_solution_l2099_209965

/-- A polynomial P(x) that satisfies P(P(x)) = (x^2 + x + 1) P(x) -/
def P (x : ℝ) : ℝ := x^2 + x

/-- Theorem stating that P(x) = x^2 + x is the unique nonconstant polynomial solution 
    to the equation P(P(x)) = (x^2 + x + 1) P(x) -/
theorem unique_polynomial_solution :
  (∀ x, P (P x) = (x^2 + x + 1) * P x) ∧
  (∀ Q : ℝ → ℝ, (∀ x, Q (Q x) = (x^2 + x + 1) * Q x) → 
    (∃ a b c, ∀ x, Q x = a * x^2 + b * x + c) →
    (∃ x y, Q x ≠ Q y) →
    (∀ x, Q x = P x)) :=
by sorry


end NUMINAMATH_CALUDE_unique_polynomial_solution_l2099_209965


namespace NUMINAMATH_CALUDE_lottery_prizes_approx_10_l2099_209980

-- Define the number of blanks
def num_blanks : ℕ := 25

-- Define the probability of drawing a blank
def blank_probability : ℚ := 5000000000000000/7000000000000000

-- Define the function to calculate the number of prizes
def calculate_prizes (blanks : ℕ) (prob : ℚ) : ℚ :=
  (blanks : ℚ) / prob - blanks

-- Theorem statement
theorem lottery_prizes_approx_10 :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ 
  |calculate_prizes num_blanks blank_probability - 10| < ε :=
sorry

end NUMINAMATH_CALUDE_lottery_prizes_approx_10_l2099_209980


namespace NUMINAMATH_CALUDE_max_profit_at_12_ships_l2099_209911

-- Define the output function
def R (x : ℕ) : ℤ := 3700 * x + 45 * x^2 - 10 * x^3

-- Define the cost function
def C (x : ℕ) : ℤ := 460 * x + 5000

-- Define the profit function
def P (x : ℕ) : ℤ := R x - C x

-- Define the marginal profit function
def MP (x : ℕ) : ℤ := P (x + 1) - P x

-- Theorem statement
theorem max_profit_at_12_ships :
  ∀ x : ℕ, 1 ≤ x → x ≤ 20 → P x ≤ P 12 :=
sorry

end NUMINAMATH_CALUDE_max_profit_at_12_ships_l2099_209911


namespace NUMINAMATH_CALUDE_overlap_time_theorem_l2099_209972

structure MovingSegment where
  length : ℝ
  initialPosition : ℝ
  speed : ℝ

def positionAt (s : MovingSegment) (t : ℝ) : ℝ :=
  s.initialPosition + s.speed * t

theorem overlap_time_theorem (ab mn : MovingSegment)
  (hab : ab.length = 100)
  (hmn : mn.length = 40)
  (hab_init : ab.initialPosition = 120)
  (hab_speed : ab.speed = -50)
  (hmn_init : mn.initialPosition = -30)
  (hmn_speed : mn.speed = 30)
  (overlap : ℝ) (hoverlap : overlap = 32) :
  ∃ t : ℝ, (t = 71/40 ∨ t = 109/40) ∧
    (positionAt ab t + ab.length - positionAt mn t = overlap ∨
     positionAt mn t + mn.length - positionAt ab t = overlap) :=
sorry

end NUMINAMATH_CALUDE_overlap_time_theorem_l2099_209972


namespace NUMINAMATH_CALUDE_perfect_squares_condition_l2099_209940

theorem perfect_squares_condition (n : ℤ) : 
  (∃ a : ℤ, 4 * n + 1 = a ^ 2) ∧ (∃ b : ℤ, 9 * n + 1 = b ^ 2) → n = 0 :=
by sorry

end NUMINAMATH_CALUDE_perfect_squares_condition_l2099_209940


namespace NUMINAMATH_CALUDE_relationship_abc_l2099_209949

theorem relationship_abc (a b c : Real) 
  (ha : a = 3^(0.3 : Real))
  (hb : b = Real.log 3 / Real.log π)
  (hc : c = Real.log 2 / Real.log 0.3) :
  c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l2099_209949


namespace NUMINAMATH_CALUDE_remainder_4523_div_32_l2099_209981

theorem remainder_4523_div_32 : 4523 % 32 = 11 := by
  sorry

end NUMINAMATH_CALUDE_remainder_4523_div_32_l2099_209981


namespace NUMINAMATH_CALUDE_gina_money_problem_l2099_209996

theorem gina_money_problem (initial_amount : ℚ) : 
  initial_amount = 400 → 
  initial_amount - (initial_amount * (1/4 + 1/8 + 1/5)) = 170 := by
  sorry

end NUMINAMATH_CALUDE_gina_money_problem_l2099_209996


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2099_209921

theorem isosceles_triangle_perimeter (a b c : ℝ) : 
  a = 6 → b = 3 → c = 3 → 
  (a + b > c ∧ b + c > a ∧ c + a > b) →  -- Triangle inequality
  (b = c) →  -- Isosceles condition
  a + b + c = 15 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2099_209921


namespace NUMINAMATH_CALUDE_expand_expression_l2099_209906

theorem expand_expression (x y : ℝ) : 
  (6*x + 8 - 3*y) * (4*x - 5*y) = 24*x^2 - 42*x*y + 32*x - 40*y + 15*y^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2099_209906


namespace NUMINAMATH_CALUDE_proposition_b_l2099_209961

theorem proposition_b (a b c : ℝ) : a < b → a * c^2 ≤ b * c^2 := by
  sorry

end NUMINAMATH_CALUDE_proposition_b_l2099_209961


namespace NUMINAMATH_CALUDE_inner_triangle_perimeter_l2099_209929

/-- Given a triangle ABC with side lengths, prove that the perimeter of the inner triangle
    formed by lines parallel to each side is equal to the length of side AB. -/
theorem inner_triangle_perimeter (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h_x_lt_c : x < c) (h_y_lt_a : y < a) (h_z_lt_b : z < b)
  (h_prop_x : x / c = (c - x) / a) (h_prop_y : y / a = (a - y) / b) (h_prop_z : z / b = (b - z) / c) :
  x / c * a + y / a * b + z / b * c = a := by
  sorry

end NUMINAMATH_CALUDE_inner_triangle_perimeter_l2099_209929


namespace NUMINAMATH_CALUDE_matrix_not_invertible_iff_y_eq_two_fifths_l2099_209993

def matrix (y : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![2 + y, 6; 4 - y, 9]

theorem matrix_not_invertible_iff_y_eq_two_fifths :
  ∀ y : ℝ, ¬(Matrix.det (matrix y) ≠ 0) ↔ y = 2/5 := by sorry

end NUMINAMATH_CALUDE_matrix_not_invertible_iff_y_eq_two_fifths_l2099_209993


namespace NUMINAMATH_CALUDE_box_sales_ratio_l2099_209950

/-- Proof of the ratio of boxes sold on Saturday to Friday -/
theorem box_sales_ratio :
  ∀ (friday saturday sunday : ℕ),
  friday = 30 →
  sunday = saturday - 15 →
  friday + saturday + sunday = 135 →
  saturday / friday = 2 := by
sorry

end NUMINAMATH_CALUDE_box_sales_ratio_l2099_209950


namespace NUMINAMATH_CALUDE_candy_distribution_l2099_209957

theorem candy_distribution (num_students : ℕ) (pieces_per_student : ℕ) 
  (h1 : num_students = 43) 
  (h2 : pieces_per_student = 8) : 
  num_students * pieces_per_student = 344 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l2099_209957


namespace NUMINAMATH_CALUDE_alexey_dowel_cost_l2099_209923

theorem alexey_dowel_cost (screw_cost dowel_cost : ℚ) : 
  screw_cost = 7 →
  (0.85 * (screw_cost + dowel_cost) = screw_cost + 0.5 * dowel_cost) →
  dowel_cost = 3 := by
sorry

end NUMINAMATH_CALUDE_alexey_dowel_cost_l2099_209923


namespace NUMINAMATH_CALUDE_inverse_proportion_point_l2099_209900

/-- Given an inverse proportion function y = 14/x passing through the point (a, 7), prove that a = 2 -/
theorem inverse_proportion_point (a : ℝ) : 
  (∃ f : ℝ → ℝ, (∀ x, f x = 14 / x) ∧ f a = 7) → a = 2 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_point_l2099_209900


namespace NUMINAMATH_CALUDE_chicken_count_l2099_209935

/-- The number of chickens in the coop -/
def coop_chickens : ℕ := 14

/-- The number of chickens in the run -/
def run_chickens : ℕ := 2 * coop_chickens

/-- The total number of chickens in the coop and run -/
def total_coop_run : ℕ := coop_chickens + run_chickens

/-- The number of free-ranging chickens -/
def free_range_chickens : ℕ := 105

theorem chicken_count : 
  (2 : ℚ) / 5 = (total_coop_run : ℚ) / free_range_chickens := by
  sorry

end NUMINAMATH_CALUDE_chicken_count_l2099_209935


namespace NUMINAMATH_CALUDE_sum_of_squared_coefficients_l2099_209915

def original_expression (x : ℝ) : ℝ := 3 * (x^3 - 4*x^2 + x) - 5 * (x^3 + 2*x^2 - 5*x + 3)

def simplified_expression (x : ℝ) : ℝ := -2*x^3 - 22*x^2 + 28*x - 15

def coefficients : List ℤ := [-2, -22, 28, -15]

theorem sum_of_squared_coefficients :
  (coefficients.map (λ c => c^2)).sum = 1497 :=
sorry

end NUMINAMATH_CALUDE_sum_of_squared_coefficients_l2099_209915


namespace NUMINAMATH_CALUDE_basketball_game_probability_formula_l2099_209977

/-- Basketball shooting game between Student A and Student B -/
structure BasketballGame where
  /-- Probability of Student A making a basket -/
  prob_a : ℚ
  /-- Probability of Student B making a basket -/
  prob_b : ℚ
  /-- Each shot is independent -/
  independent_shots : Bool

/-- Score of Student A after one round -/
inductive Score where
  | lose : Score  -- Student A loses (-1)
  | draw : Score  -- Draw (0)
  | win  : Score  -- Student A wins (+1)

/-- Probability distribution of Student A's score after one round -/
def score_distribution (game : BasketballGame) : Score → ℚ
  | Score.lose => (1 - game.prob_a) * game.prob_b
  | Score.draw => game.prob_a * game.prob_b + (1 - game.prob_a) * (1 - game.prob_b)
  | Score.win  => game.prob_a * (1 - game.prob_b)

/-- Expected value of Student A's score after one round -/
def expected_score (game : BasketballGame) : ℚ :=
  -1 * score_distribution game Score.lose +
   0 * score_distribution game Score.draw +
   1 * score_distribution game Score.win

/-- Probability that Student A's cumulative score is lower than Student B's after n rounds -/
def p (n : ℕ) : ℚ :=
  (1 / 5) * (1 - (1 / 6)^n)

/-- Main theorem: Probability formula for Student A's score being lower after n rounds -/
theorem basketball_game_probability_formula (game : BasketballGame) (n : ℕ) 
    (h1 : game.prob_a = 2/3) (h2 : game.prob_b = 1/2) (h3 : game.independent_shots = true) :
    p n = (1 / 5) * (1 - (1 / 6)^n) := by
  sorry

end NUMINAMATH_CALUDE_basketball_game_probability_formula_l2099_209977


namespace NUMINAMATH_CALUDE_steve_orange_count_l2099_209936

/-- The number of oranges each person has -/
structure OrangeCount where
  marcie : ℝ
  brian : ℝ
  shawn : ℝ
  steve : ℝ

/-- The conditions of the orange distribution problem -/
def orange_problem (o : OrangeCount) : Prop :=
  o.marcie = 12 ∧
  o.brian = o.marcie ∧
  o.shawn = (o.marcie + o.brian) * 1.075 ∧
  o.steve = 3 * (o.marcie + o.brian + o.shawn)

/-- The theorem stating Steve's orange count -/
theorem steve_orange_count (o : OrangeCount) (h : orange_problem o) : o.steve = 149.4 := by
  sorry

end NUMINAMATH_CALUDE_steve_orange_count_l2099_209936


namespace NUMINAMATH_CALUDE_larger_solution_of_quadratic_l2099_209938

theorem larger_solution_of_quadratic (x : ℝ) : 
  x^2 - 13*x + 36 = 0 → x ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_larger_solution_of_quadratic_l2099_209938


namespace NUMINAMATH_CALUDE_cab_driver_income_l2099_209943

/-- Cab driver's income problem -/
theorem cab_driver_income 
  (income : Fin 5 → ℕ) 
  (h1 : income 0 = 600)
  (h2 : income 1 = 250)
  (h3 : income 2 = 450)
  (h4 : income 3 = 400)
  (h_avg : (income 0 + income 1 + income 2 + income 3 + income 4) / 5 = 500) :
  income 4 = 800 := by
sorry


end NUMINAMATH_CALUDE_cab_driver_income_l2099_209943


namespace NUMINAMATH_CALUDE_thirtieth_term_of_sequence_l2099_209924

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

theorem thirtieth_term_of_sequence (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 3) (h₂ : a₂ = 17) (h₃ : a₃ = 31) :
  arithmetic_sequence a₁ (a₂ - a₁) 30 = 409 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_term_of_sequence_l2099_209924


namespace NUMINAMATH_CALUDE_room_width_calculation_l2099_209952

/-- Given a rectangular room with known length, paving cost per square meter, and total paving cost,
    calculate the width of the room. -/
theorem room_width_calculation (length : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) 
    (h1 : length = 5.5)
    (h2 : cost_per_sqm = 300)
    (h3 : total_cost = 6187.5) :
    total_cost / cost_per_sqm / length = 3.75 := by
  sorry

end NUMINAMATH_CALUDE_room_width_calculation_l2099_209952


namespace NUMINAMATH_CALUDE_range_of_a_l2099_209985

def sequence_a (a : ℝ) (n : ℕ) : ℝ :=
  if n ≤ 5 then (5 - a) * n - 11 else a^(n - 4)

theorem range_of_a (a : ℝ) :
  (∀ n m : ℕ, n < m → sequence_a a n < sequence_a a m) →
  2 < a ∧ a < 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2099_209985


namespace NUMINAMATH_CALUDE_lollipop_count_l2099_209947

theorem lollipop_count (total_cost : ℝ) (single_cost : ℝ) (h1 : total_cost = 90) (h2 : single_cost = 0.75) :
  total_cost / single_cost = 120 := by
  sorry

end NUMINAMATH_CALUDE_lollipop_count_l2099_209947


namespace NUMINAMATH_CALUDE_fullPriceRevenue_l2099_209956

/-- Represents the fundraising event ticket sales -/
structure FundraisingEvent where
  totalTickets : ℕ
  totalRevenue : ℕ
  fullPriceTickets : ℕ
  halfPriceTickets : ℕ
  fullPrice : ℕ

/-- The conditions of the fundraising event -/
def eventConditions (e : FundraisingEvent) : Prop :=
  e.totalTickets = 180 ∧
  e.totalRevenue = 2709 ∧
  e.totalTickets = e.fullPriceTickets + e.halfPriceTickets ∧
  e.totalRevenue = e.fullPriceTickets * e.fullPrice + e.halfPriceTickets * (e.fullPrice / 2)

/-- The theorem to prove -/
theorem fullPriceRevenue (e : FundraisingEvent) :
  eventConditions e → e.fullPriceTickets * e.fullPrice = 2142 := by
  sorry


end NUMINAMATH_CALUDE_fullPriceRevenue_l2099_209956


namespace NUMINAMATH_CALUDE_distance_between_complex_points_l2099_209920

def complex_to_point (z : ℂ) : ℝ × ℝ := (z.re, z.im)

theorem distance_between_complex_points :
  let z1 : ℂ := 7 - 4*I
  let z2 : ℂ := 2 + 8*I
  let A : ℝ × ℝ := complex_to_point z1
  let B : ℝ × ℝ := complex_to_point z2
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 13 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_complex_points_l2099_209920


namespace NUMINAMATH_CALUDE_prop_truth_values_l2099_209951

theorem prop_truth_values (p q : Prop) :
  ¬(p ∨ (¬q)) → (¬p ∧ q) := by
  sorry

end NUMINAMATH_CALUDE_prop_truth_values_l2099_209951


namespace NUMINAMATH_CALUDE_frog_jump_distance_l2099_209925

/-- The jumping contest between a grasshopper and a frog -/
theorem frog_jump_distance (grasshopper_jump : ℕ) (frog_extra_distance : ℕ) : 
  grasshopper_jump = 9 → frog_extra_distance = 3 → 
  grasshopper_jump + frog_extra_distance = 12 := by
  sorry

end NUMINAMATH_CALUDE_frog_jump_distance_l2099_209925


namespace NUMINAMATH_CALUDE_no_valid_numbers_l2099_209919

def digits : List Nat := [2, 3, 5, 6, 9]

def is_valid_number (n : Nat) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  (n % 15 = 0) ∧
  (∀ d : Nat, d ∈ digits → (n / 100 = d ∨ (n / 10) % 10 = d ∨ n % 10 = d)) ∧
  (n / 100 ≠ (n / 10) % 10) ∧
  (n / 100 ≠ n % 10) ∧
  ((n / 10) % 10 ≠ n % 10)

theorem no_valid_numbers : ¬∃ n : Nat, is_valid_number n :=
sorry

end NUMINAMATH_CALUDE_no_valid_numbers_l2099_209919


namespace NUMINAMATH_CALUDE_possible_values_of_expression_l2099_209992

theorem possible_values_of_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a / abs a + b / abs b + (a * b) / abs (a * b)) = 3 ∨
  (a / abs a + b / abs b + (a * b) / abs (a * b)) = -1 :=
sorry

end NUMINAMATH_CALUDE_possible_values_of_expression_l2099_209992


namespace NUMINAMATH_CALUDE_sum_of_four_digit_odd_and_multiples_of_five_l2099_209944

/-- The number of four-digit odd numbers -/
def A : ℕ := 4500

/-- The number of four-digit multiples of 5 -/
def B : ℕ := 1800

/-- The sum of four-digit odd numbers and four-digit multiples of 5 is 6300 -/
theorem sum_of_four_digit_odd_and_multiples_of_five : A + B = 6300 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_digit_odd_and_multiples_of_five_l2099_209944


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2099_209927

theorem polynomial_division_remainder :
  let f (x : ℝ) := x^6 - 2*x^5 + x^4 - x^2 + 3*x - 1
  let g (x : ℝ) := (x^2 - 1)*(x + 2)
  let r (x : ℝ) := 7/3*x^2 + x - 7/3
  ∃ q : ℝ → ℝ, ∀ x, f x = g x * q x + r x :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2099_209927


namespace NUMINAMATH_CALUDE_f_continuous_iff_b_eq_12_l2099_209933

-- Define the piecewise function f(x)
noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  if x > -2 then 3 * x + b else -x + 4

-- Theorem statement
theorem f_continuous_iff_b_eq_12 (b : ℝ) :
  Continuous (f b) ↔ b = 12 := by
  sorry

end NUMINAMATH_CALUDE_f_continuous_iff_b_eq_12_l2099_209933


namespace NUMINAMATH_CALUDE_base_ratio_in_special_isosceles_trapezoid_l2099_209968

/-- Represents an isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  smaller_base : ℝ
  larger_base : ℝ
  diagonal : ℝ
  altitude : ℝ
  sum_of_bases : smaller_base + larger_base = 10
  larger_base_prop : larger_base = 2 * diagonal
  smaller_base_prop : smaller_base = 2 * altitude

/-- Theorem stating the ratio of bases in the specific isosceles trapezoid -/
theorem base_ratio_in_special_isosceles_trapezoid (t : IsoscelesTrapezoid) :
  t.smaller_base / t.larger_base = (2 * Real.sqrt 2 - 1) / 2 := by
  sorry


end NUMINAMATH_CALUDE_base_ratio_in_special_isosceles_trapezoid_l2099_209968


namespace NUMINAMATH_CALUDE_hyperbola_properties_l2099_209966

-- Define the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop := 16 * x^2 - 9 * y^2 = 144

-- Define the parabola C
def parabola_C (x y : ℝ) : Prop := y^2 = -12 * x

-- Theorem statement
theorem hyperbola_properties :
  -- Length of real axis
  (∃ a : ℝ, a = 3 ∧ 2 * a = 6) ∧
  -- Length of imaginary axis
  (∃ b : ℝ, b = 4 ∧ 2 * b = 8) ∧
  -- Eccentricity
  (∃ e : ℝ, e = 5/3) ∧
  -- Parabola C equation
  (∀ x y : ℝ, hyperbola_eq x y → 
    (x = -3 → parabola_C x y) ∧ 
    (x = 0 ∧ y = 0 → parabola_C x y)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l2099_209966


namespace NUMINAMATH_CALUDE_problem_solution_l2099_209918

def has_at_least_four_divisors (n : ℕ) : Prop :=
  (Nat.divisors n).card ≥ 4

def divisor_differences_divide (n : ℕ) : Prop :=
  ∀ a b : ℕ, a ∣ n → b ∣ n → 1 < a → a < b → b < n → (b - a) ∣ n

def satisfies_conditions (n : ℕ) : Prop :=
  has_at_least_four_divisors n ∧ divisor_differences_divide n

theorem problem_solution : 
  {n : ℕ | satisfies_conditions n} = {6, 8, 12} := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2099_209918


namespace NUMINAMATH_CALUDE_coconut_trips_l2099_209907

def total_coconuts : ℕ := 144
def barbie_capacity : ℕ := 4
def bruno_capacity : ℕ := 8

theorem coconut_trips : 
  (total_coconuts / (barbie_capacity + bruno_capacity) : ℕ) = 12 := by
  sorry

end NUMINAMATH_CALUDE_coconut_trips_l2099_209907


namespace NUMINAMATH_CALUDE_haley_trees_died_l2099_209987

/-- The number of trees that died in a typhoon given the initial number of trees and the number of trees remaining. -/
def trees_died (initial_trees remaining_trees : ℕ) : ℕ :=
  initial_trees - remaining_trees

/-- Proof that 5 trees died in the typhoon given the conditions in Haley's problem. -/
theorem haley_trees_died : trees_died 17 12 = 5 := by
  sorry

end NUMINAMATH_CALUDE_haley_trees_died_l2099_209987


namespace NUMINAMATH_CALUDE_min_value_of_f_l2099_209983

/-- The function f(x) = e^x - e^(2x) has a minimum value of -e^2 -/
theorem min_value_of_f (x : ℝ) : 
  let f : ℝ → ℝ := λ x => Real.exp x - Real.exp (2 * x)
  ∃ m : ℝ, (∀ x, f x ≥ m) ∧ (∃ x₀, f x₀ = m) ∧ m = -Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2099_209983


namespace NUMINAMATH_CALUDE_shopkeeper_cloth_sale_l2099_209902

/-- Represents the sale of cloth by a shopkeeper -/
structure ClothSale where
  totalSellingPrice : ℕ
  lossPerMetre : ℕ
  costPricePerMetre : ℕ

/-- Calculates the number of metres of cloth sold given the sale details -/
def metresSold (sale : ClothSale) : ℕ :=
  sale.totalSellingPrice / (sale.costPricePerMetre - sale.lossPerMetre)

/-- Theorem stating that for the given conditions, the shopkeeper sold 200 metres of cloth -/
theorem shopkeeper_cloth_sale :
  let sale : ClothSale := {
    totalSellingPrice := 12000,
    lossPerMetre := 6,
    costPricePerMetre := 66
  }
  metresSold sale = 200 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_cloth_sale_l2099_209902


namespace NUMINAMATH_CALUDE_four_digit_sum_l2099_209954

theorem four_digit_sum (a b c d : ℕ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a * b * c * d = 810 →
  1000 ≤ a * 1000 + b * 100 + c * 10 + d →
  a * 1000 + b * 100 + c * 10 + d < 10000 →
  a + b + c + d = 23 := by
sorry

end NUMINAMATH_CALUDE_four_digit_sum_l2099_209954


namespace NUMINAMATH_CALUDE_parabola_equation_l2099_209934

/-- A parabola with vertex at the origin, axis of symmetry along the x-axis,
    and focus on the line 3x - 4y - 12 = 0 has the equation y² = 16x -/
theorem parabola_equation (x y : ℝ) :
  (∀ a b : ℝ, 3 * a - 4 * b - 12 = 0 → (a = 4 ∧ b = 0)) →
  y^2 = 16 * x := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l2099_209934


namespace NUMINAMATH_CALUDE_race_result_l2099_209959

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  (speed_pos : 0 < speed)

/-- The race setup -/
structure Race where
  anton : Runner
  seryozha : Runner
  tolya : Runner
  (different_speeds : anton.speed ≠ seryozha.speed ∧ seryozha.speed ≠ tolya.speed ∧ anton.speed ≠ tolya.speed)

/-- The race conditions -/
def race_conditions (r : Race) : Prop :=
  let t_anton := 100 / r.anton.speed
  let d_seryozha := r.seryozha.speed * t_anton
  let t_seryozha := 100 / r.seryozha.speed
  let d_tolya := r.tolya.speed * t_seryozha
  d_seryozha = 90 ∧ d_tolya = 90

theorem race_result (r : Race) (h : race_conditions r) :
  r.tolya.speed * (100 / r.anton.speed) = 81 := by
  sorry

#check race_result

end NUMINAMATH_CALUDE_race_result_l2099_209959


namespace NUMINAMATH_CALUDE_divisible_polynomial_sum_l2099_209909

-- Define the polynomial
def p (A B : ℝ) (x : ℂ) := x^101 + A*x + B

-- Define the condition of divisibility
def is_divisible (A B : ℝ) : Prop :=
  ∀ x : ℂ, x^2 + x + 1 = 0 → p A B x = 0

-- Theorem statement
theorem divisible_polynomial_sum (A B : ℝ) (h : is_divisible A B) : A + B = 2 := by
  sorry

end NUMINAMATH_CALUDE_divisible_polynomial_sum_l2099_209909


namespace NUMINAMATH_CALUDE_number_operation_result_l2099_209963

theorem number_operation_result : ∃ (x : ℝ), x = 295 ∧ (x / 5 + 6 = 65) := by sorry

end NUMINAMATH_CALUDE_number_operation_result_l2099_209963


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2099_209958

/-- The perimeter of a rectangular field with area 800 square meters and width 20 meters is 120 meters. -/
theorem rectangle_perimeter (area width : ℝ) (h_area : area = 800) (h_width : width = 20) :
  2 * (area / width + width) = 120 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2099_209958


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2099_209917

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  3 * X^2 - 11 * X + 18 = (X - 3) * q + 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2099_209917


namespace NUMINAMATH_CALUDE_divisibility_problem_l2099_209971

theorem divisibility_problem :
  ∃ k : ℕ, (2^286 - 1) * (3^500 - 1) * (1978^100 - 1) = k * (2^4 * 5^7 * 2003) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l2099_209971


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l2099_209945

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  x + y = -b / a :=
sorry

theorem sum_of_roots_specific_equation :
  let f : ℝ → ℝ := λ x => x^2 - 2003 * x - 2004
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  x + y = 2003 :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l2099_209945


namespace NUMINAMATH_CALUDE_selling_price_calculation_l2099_209990

/-- Given a sale where the gain is $20 and the gain percentage is 25%, 
    prove that the selling price is $100. -/
theorem selling_price_calculation (gain : ℝ) (gain_percentage : ℝ) :
  gain = 20 →
  gain_percentage = 25 →
  ∃ (cost_price selling_price : ℝ),
    gain = gain_percentage / 100 * cost_price ∧
    selling_price = cost_price + gain ∧
    selling_price = 100 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_calculation_l2099_209990


namespace NUMINAMATH_CALUDE_chocolate_bar_cost_is_two_l2099_209937

/-- The cost of a chocolate bar given Frank's purchase information -/
def chocolate_bar_cost (num_bars : ℕ) (num_chips : ℕ) (chips_cost : ℕ) (total_paid : ℕ) (change : ℕ) : ℕ :=
  (total_paid - change - num_chips * chips_cost) / num_bars

/-- Theorem stating that the cost of each chocolate bar is $2 -/
theorem chocolate_bar_cost_is_two :
  chocolate_bar_cost 5 2 3 20 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bar_cost_is_two_l2099_209937


namespace NUMINAMATH_CALUDE_binomial_coefficient_third_term_2x_minus_y_power_8_l2099_209926

/-- The binomial coefficient of the 3rd term in the expansion of (2x-y)^8 is 28 -/
theorem binomial_coefficient_third_term_2x_minus_y_power_8 :
  Nat.choose 8 2 = 28 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_third_term_2x_minus_y_power_8_l2099_209926


namespace NUMINAMATH_CALUDE_star_interior_angle_sum_l2099_209962

/-- An n-pointed star constructed from an n-sided convex polygon -/
structure StarPolygon where
  n : ℕ
  n_ge_6 : n ≥ 6

/-- The sum of interior angles at the vertices of the star -/
def interior_angle_sum (star : StarPolygon) : ℝ :=
  180 * (star.n - 2)

/-- Theorem: The sum of interior angles at the vertices of an n-pointed star
    constructed by extending every third side of an n-sided convex polygon (n ≥ 6)
    is equal to 180° * (n - 2) -/
theorem star_interior_angle_sum (star : StarPolygon) :
  interior_angle_sum star = 180 * (star.n - 2) := by
  sorry

end NUMINAMATH_CALUDE_star_interior_angle_sum_l2099_209962


namespace NUMINAMATH_CALUDE_fraction_equality_l2099_209939

theorem fraction_equality (w z : ℝ) (h : (1/w + 1/z)/(1/w - 1/z) = 1008) :
  (w + z)/(w - z) = 1008 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2099_209939


namespace NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l2099_209942

theorem sum_of_cubes_of_roots (x₁ x₂ x₃ : ℝ) : 
  x₁^3 + x₂^3 + x₃^3 = 11 ∧ 
  x₁ + x₂ + x₃ = 2 ∧
  x₁ * x₂ + x₂ * x₃ + x₃ * x₁ = -1 ∧
  x₁ * x₂ * x₃ = -1 ∧
  x₁^3 - 2*x₁^2 - x₁ + 1 = 0 ∧
  x₂^3 - 2*x₂^2 - x₂ + 1 = 0 ∧
  x₃^3 - 2*x₃^2 - x₃ + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l2099_209942


namespace NUMINAMATH_CALUDE_twenty_three_to_binary_l2099_209999

-- Define a function to convert a natural number to its binary representation
def to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false]
  else
    let rec aux (m : ℕ) : List Bool :=
      if m = 0 then []
      else (m % 2 = 1) :: aux (m / 2)
    aux n

-- Define the decimal number we want to convert
def decimal_number : ℕ := 23

-- Define the expected binary representation
def expected_binary : List Bool := [true, true, true, false, true]

-- Theorem statement
theorem twenty_three_to_binary :
  to_binary decimal_number = expected_binary := by sorry

end NUMINAMATH_CALUDE_twenty_three_to_binary_l2099_209999


namespace NUMINAMATH_CALUDE_radio_loss_percentage_l2099_209970

/-- Calculates the loss percentage given the cost price and selling price. -/
def loss_percentage (cost_price selling_price : ℚ) : ℚ :=
  (cost_price - selling_price) / cost_price * 100

/-- Theorem stating that for a radio with cost price 1500 and selling price 1275,
    the loss percentage is 15%. -/
theorem radio_loss_percentage :
  loss_percentage 1500 1275 = 15 := by sorry

end NUMINAMATH_CALUDE_radio_loss_percentage_l2099_209970


namespace NUMINAMATH_CALUDE_turtle_speed_specific_turtle_speed_l2099_209984

/-- Given a race with a hare and a turtle, calculate the turtle's speed -/
theorem turtle_speed (race_distance : ℝ) (hare_speed : ℝ) (head_start : ℝ) : ℝ :=
  let turtle_speed := race_distance / (race_distance / hare_speed + head_start)
  turtle_speed

/-- The turtle's speed in the specific race scenario -/
theorem specific_turtle_speed : 
  turtle_speed 20 10 18 = 1 := by sorry

end NUMINAMATH_CALUDE_turtle_speed_specific_turtle_speed_l2099_209984


namespace NUMINAMATH_CALUDE_min_value_of_polynomial_l2099_209973

theorem min_value_of_polynomial (x : ℝ) : 
  x * (x + 4) * (x + 8) * (x + 12) ≥ -256 ∧ 
  ∃ y : ℝ, y * (y + 4) * (y + 8) * (y + 12) = -256 := by sorry

end NUMINAMATH_CALUDE_min_value_of_polynomial_l2099_209973


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2099_209903

theorem complex_modulus_problem (z : ℂ) (h : z⁻¹ = 1 + I) : Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2099_209903


namespace NUMINAMATH_CALUDE_female_fraction_l2099_209974

theorem female_fraction (total_students : ℕ) (non_foreign_males : ℕ) :
  total_students = 300 →
  non_foreign_males = 90 →
  (2 : ℚ) / 3 = (total_students - (non_foreign_males / (9 : ℚ) / 10)) / total_students :=
by sorry

end NUMINAMATH_CALUDE_female_fraction_l2099_209974


namespace NUMINAMATH_CALUDE_horner_method_v2_l2099_209989

/-- Horner's method for polynomial evaluation -/
def horner_v2 (x : ℤ) : ℤ := x^2 + 6

/-- The polynomial f(x) = x^6 + 6x^4 + 9x^2 + 208 -/
def f (x : ℤ) : ℤ := x^6 + 6*x^4 + 9*x^2 + 208

theorem horner_method_v2 :
  horner_v2 (-4) = 22 :=
by sorry

end NUMINAMATH_CALUDE_horner_method_v2_l2099_209989


namespace NUMINAMATH_CALUDE_train_length_l2099_209986

/-- The length of a train given its speed, time to cross a bridge, and the bridge length -/
theorem train_length (speed : ℝ) (time : ℝ) (bridge_length : ℝ) :
  speed = 45 * 1000 / 3600 →
  time = 30 →
  bridge_length = 215 →
  speed * time - bridge_length = 160 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l2099_209986


namespace NUMINAMATH_CALUDE_solution_l2099_209931

/-- The set of points satisfying the given equation -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2 * p.1^2 + p.2^2 + 3 * p.1 * p.2 + 3 * p.1 + p.2 = 2}

/-- The first line -/
def L₁ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = -p.1 - 2}

/-- The second line -/
def L₂ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = -2 * p.1 + 1}

/-- The union of the two lines -/
def U : Set (ℝ × ℝ) :=
  L₁ ∪ L₂

theorem solution : S = U := by
  sorry

end NUMINAMATH_CALUDE_solution_l2099_209931


namespace NUMINAMATH_CALUDE_translation_teams_count_l2099_209916

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The total number of translators -/
def total_translators : ℕ := 11

/-- The number of English-only translators -/
def english_only : ℕ := 5

/-- The number of Japanese-only translators -/
def japanese_only : ℕ := 4

/-- The number of bilingual translators -/
def bilingual : ℕ := 2

/-- The size of each translation team -/
def team_size : ℕ := 4

/-- The total number of ways to form two translation teams -/
def total_ways : ℕ :=
  choose english_only team_size * choose japanese_only team_size +
  choose bilingual 1 * (choose english_only (team_size - 1) * choose japanese_only team_size +
                        choose english_only team_size * choose japanese_only (team_size - 1)) +
  choose bilingual 2 * (choose english_only (team_size - 2) * choose japanese_only team_size +
                        choose english_only team_size * choose japanese_only (team_size - 2) +
                        choose english_only (team_size - 1) * choose japanese_only (team_size - 1))

theorem translation_teams_count : total_ways = 185 := by sorry

end NUMINAMATH_CALUDE_translation_teams_count_l2099_209916


namespace NUMINAMATH_CALUDE_joeys_reading_assignment_l2099_209975

/-- The number of pages Joey must read after his break -/
def pages_after_break : ℕ := 9

/-- The percentage of pages Joey reads before taking a break -/
def percentage_before_break : ℚ := 70 / 100

theorem joeys_reading_assignment :
  ∃ (total_pages : ℕ),
    (1 - percentage_before_break) * total_pages = pages_after_break ∧
    total_pages = 30 := by
  sorry

end NUMINAMATH_CALUDE_joeys_reading_assignment_l2099_209975


namespace NUMINAMATH_CALUDE_gross_revenue_increase_l2099_209948

theorem gross_revenue_increase
  (original_price : ℝ)
  (original_quantity : ℝ)
  (price_decrease_percentage : ℝ)
  (quantity_increase_percentage : ℝ)
  (h1 : price_decrease_percentage = 20)
  (h2 : quantity_increase_percentage = 70)
  : let new_price := original_price * (1 - price_decrease_percentage / 100)
    let new_quantity := original_quantity * (1 + quantity_increase_percentage / 100)
    let original_revenue := original_price * original_quantity
    let new_revenue := new_price * new_quantity
    (new_revenue - original_revenue) / original_revenue * 100 = 36 := by
  sorry

end NUMINAMATH_CALUDE_gross_revenue_increase_l2099_209948


namespace NUMINAMATH_CALUDE_system_solution_l2099_209905

theorem system_solution (a b x y : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  x^5 * y^17 = a ∧ x^2 * y^7 = b → x = a^7 / b^17 ∧ y = b^5 / a^2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2099_209905


namespace NUMINAMATH_CALUDE_ellipse_chord_slope_range_l2099_209988

/-- The slope of a chord of the ellipse x^2 + y^2/4 = 1 whose midpoint lies on the line segment
    between (1/2, 1/2) and (1/2, 1) is between -4 and -2. -/
theorem ellipse_chord_slope_range :
  ∀ (x₁ y₁ x₂ y₂ x₀ y₀ : ℝ),
  (x₁^2 + y₁^2/4 = 1) →  -- P(x₁, y₁) is on the ellipse
  (x₂^2 + y₂^2/4 = 1) →  -- Q(x₂, y₂) is on the ellipse
  (x₀ = (x₁ + x₂)/2) →   -- x-coordinate of midpoint
  (y₀ = (y₁ + y₂)/2) →   -- y-coordinate of midpoint
  (x₀ = 1/2) →           -- midpoint x-coordinate is on AB
  (1/2 ≤ y₀ ∧ y₀ ≤ 1) →  -- midpoint y-coordinate is between A and B
  (-4 ≤ -(y₁ - y₂)/(x₁ - x₂) ∧ -(y₁ - y₂)/(x₁ - x₂) ≤ -2) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_chord_slope_range_l2099_209988


namespace NUMINAMATH_CALUDE_sqrt_four_equals_two_l2099_209955

theorem sqrt_four_equals_two : Real.sqrt 4 = 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_four_equals_two_l2099_209955


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_l2099_209941

theorem least_three_digit_multiple : ∃ n : ℕ, 
  (100 ≤ n ∧ n < 1000) ∧ 
  (2 ∣ n) ∧ (3 ∣ n) ∧ (5 ∣ n) ∧ (7 ∣ n) ∧
  (∀ m : ℕ, (100 ≤ m ∧ m < n) → ¬((2 ∣ m) ∧ (3 ∣ m) ∧ (5 ∣ m) ∧ (7 ∣ m))) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_l2099_209941


namespace NUMINAMATH_CALUDE_prob_both_truth_l2099_209932

/-- The probability that A speaks the truth -/
def prob_A_truth : ℝ := 0.75

/-- The probability that B speaks the truth -/
def prob_B_truth : ℝ := 0.60

/-- The theorem stating the probability of A and B both telling the truth simultaneously -/
theorem prob_both_truth : prob_A_truth * prob_B_truth = 0.45 := by sorry

end NUMINAMATH_CALUDE_prob_both_truth_l2099_209932


namespace NUMINAMATH_CALUDE_decompose_4_705_l2099_209901

theorem decompose_4_705 : 
  ∃ (units hundredths thousandths : ℕ),
    4.705 = (units : ℝ) + (7 : ℝ) / 10 + (thousandths : ℝ) / 1000 ∧
    units = 4 ∧
    thousandths = 5 := by
  sorry

end NUMINAMATH_CALUDE_decompose_4_705_l2099_209901


namespace NUMINAMATH_CALUDE_parallelogram_side_length_l2099_209960

/-- Given a parallelogram with adjacent sides of lengths 3s and s units, 
    forming a 60-degree angle, and having an area of 9√3 square units, 
    prove that s = √6. -/
theorem parallelogram_side_length (s : ℝ) : 
  s > 0 →
  let adjacent_side1 := 3 * s
  let adjacent_side2 := s
  let angle := Real.pi / 3  -- 60 degrees in radians
  let area := 9 * Real.sqrt 3
  area = adjacent_side1 * adjacent_side2 * Real.sin angle →
  s = Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_side_length_l2099_209960


namespace NUMINAMATH_CALUDE_inequality_proof_l2099_209997

theorem inequality_proof (a b c d : ℝ) 
  (non_neg_a : 0 ≤ a) (non_neg_b : 0 ≤ b) (non_neg_c : 0 ≤ c) (non_neg_d : 0 ≤ d)
  (sum_condition : a * b + b * c + c * d + d * a = 1) : 
  a^3 / (b + c + d) + b^3 / (c + d + a) + c^3 / (a + b + d) + d^3 / (a + b + c) ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2099_209997


namespace NUMINAMATH_CALUDE_problem_statement_l2099_209930

theorem problem_statement (a b c m : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : a + b + c = m) (h2 : a^2 + b^2 + c^2 = m^2 / 2) :
  (a * (m - 2*a)^2 + b * (m - 2*b)^2 + c * (m - 2*c)^2) / (a * b * c) = 12 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2099_209930


namespace NUMINAMATH_CALUDE_regular_10gon_triangle_probability_l2099_209904

/-- Regular 10-gon -/
def regular_10gon : Set (ℝ × ℝ) := sorry

/-- Set of all segments in the 10-gon -/
def segments (polygon : Set (ℝ × ℝ)) : Set (Set (ℝ × ℝ)) := sorry

/-- Predicate to check if three segments form a triangle with positive area -/
def forms_triangle (s1 s2 s3 : Set (ℝ × ℝ)) : Prop := sorry

/-- The probability of forming a triangle with positive area from three randomly chosen segments -/
def triangle_probability (polygon : Set (ℝ × ℝ)) : ℚ := sorry

/-- Main theorem: The probability of forming a triangle with positive area 
    from three distinct segments chosen randomly from a regular 10-gon is 343/715 -/
theorem regular_10gon_triangle_probability : 
  triangle_probability regular_10gon = 343 / 715 := by sorry

end NUMINAMATH_CALUDE_regular_10gon_triangle_probability_l2099_209904


namespace NUMINAMATH_CALUDE_tabitha_current_age_l2099_209998

/-- Tabitha's hair color tradition --/
def tabitha_age : ℕ → Prop :=
  fun current_age =>
    ∃ (colors : ℕ),
      colors = current_age - 15 + 2 ∧
      colors + 3 = 8

theorem tabitha_current_age :
  ∃ (age : ℕ), tabitha_age age ∧ age = 20 := by
  sorry

end NUMINAMATH_CALUDE_tabitha_current_age_l2099_209998
