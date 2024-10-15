import Mathlib

namespace NUMINAMATH_CALUDE_extra_bananas_l3083_308372

/-- Given the total number of children, the number of absent children, and the planned distribution,
    prove that each present child received 2 extra bananas. -/
theorem extra_bananas (total_children absent_children planned_per_child : ℕ) 
  (h1 : total_children = 660)
  (h2 : absent_children = 330)
  (h3 : planned_per_child = 2) :
  let present_children := total_children - absent_children
  let total_bananas := total_children * planned_per_child
  let actual_per_child := total_bananas / present_children
  actual_per_child - planned_per_child = 2 := by
  sorry

end NUMINAMATH_CALUDE_extra_bananas_l3083_308372


namespace NUMINAMATH_CALUDE_election_winner_votes_l3083_308362

theorem election_winner_votes (total_votes : ℕ) (candidates : ℕ) 
  (difference1 : ℕ) (difference2 : ℕ) (difference3 : ℕ) :
  total_votes = 963 →
  candidates = 4 →
  difference1 = 53 →
  difference2 = 79 →
  difference3 = 105 →
  ∃ (winner_votes : ℕ),
    winner_votes + (winner_votes - difference1) + 
    (winner_votes - difference2) + (winner_votes - difference3) = total_votes ∧
    winner_votes = 300 :=
by sorry

end NUMINAMATH_CALUDE_election_winner_votes_l3083_308362


namespace NUMINAMATH_CALUDE_solve_cassette_problem_l3083_308349

structure AudioVideoCassettes where
  audioCost : ℝ
  videoCost : ℝ
  firstSetAudioCount : ℝ
  secondSetAudioCount : ℝ

def cassetteProblem (c : AudioVideoCassettes) : Prop :=
  c.videoCost = 300 ∧
  c.firstSetAudioCount * c.audioCost + 4 * c.videoCost = 1350 ∧
  7 * c.audioCost + 3 * c.videoCost = 1110 ∧
  c.secondSetAudioCount = 7

theorem solve_cassette_problem :
  ∃ c : AudioVideoCassettes, cassetteProblem c :=
by
  sorry

end NUMINAMATH_CALUDE_solve_cassette_problem_l3083_308349


namespace NUMINAMATH_CALUDE_fourth_place_points_value_l3083_308396

def first_place_points : ℕ := 11
def second_place_points : ℕ := 7
def third_place_points : ℕ := 5
def total_participations : ℕ := 7
def total_points_product : ℕ := 38500

def is_valid_fourth_place_points (fourth_place_points : ℕ) : Prop :=
  ∃ (a b c d : ℕ),
    a + b + c + d = total_participations ∧
    first_place_points ^ a * second_place_points ^ b * third_place_points ^ c * fourth_place_points ^ d = total_points_product

theorem fourth_place_points_value :
  ∃! (x : ℕ), is_valid_fourth_place_points x ∧ x = 4 := by sorry

end NUMINAMATH_CALUDE_fourth_place_points_value_l3083_308396


namespace NUMINAMATH_CALUDE_expansion_coefficient_l3083_308310

theorem expansion_coefficient (a : ℝ) : 
  (∃ k : ℝ, k = 21 ∧ k = a^2 * 15 - 6 * a) ↔ (a = -1 ∨ a = 7/5) := by
sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l3083_308310


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3083_308370

theorem absolute_value_inequality (a : ℝ) :
  (∀ x : ℝ, |x + 1| + |x - 2| > a) → a < 3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3083_308370


namespace NUMINAMATH_CALUDE_hexagon_diagonal_area_bound_l3083_308335

/-- A convex hexagon is a six-sided polygon where all interior angles are less than or equal to 180 degrees. -/
structure ConvexHexagon where
  -- We assume the existence of a convex hexagon without explicitly defining its properties
  -- as the specific geometric representation is not crucial for this theorem.

/-- The theorem states that for any convex hexagon, there exists a diagonal that cuts off a triangle
    with an area less than or equal to one-sixth of the total area of the hexagon. -/
theorem hexagon_diagonal_area_bound (h : ConvexHexagon) (S : ℝ) (h_area : S > 0) :
  ∃ (triangle_area : ℝ), triangle_area ≤ S / 6 ∧ triangle_area > 0 := by
  sorry


end NUMINAMATH_CALUDE_hexagon_diagonal_area_bound_l3083_308335


namespace NUMINAMATH_CALUDE_largest_prime_factor_l3083_308308

def expression : ℤ := 16^4 + 3 * 16^2 + 2 - 15^4

theorem largest_prime_factor (p : ℕ) : 
  Nat.Prime p ∧ p ∣ expression.natAbs ∧ 
  ∀ q : ℕ, Nat.Prime q ∧ q ∣ expression.natAbs → q ≤ p ↔ p = 241 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_l3083_308308


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l3083_308397

def p (x : ℝ) : ℝ := 4 * x^3 - 8 * x^2 + 8 * x - 16

theorem polynomial_divisibility :
  (∃ q : ℝ → ℝ, ∀ x, p x = (x - 2) * q x) ∧
  (∃ r : ℝ → ℝ, ∀ x, p x = (x^2 + 1) * r x) :=
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l3083_308397


namespace NUMINAMATH_CALUDE_fraction_equality_l3083_308355

theorem fraction_equality (a b c : ℝ) (h1 : c ≠ 0) :
  a / c = b / c → a = b := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l3083_308355


namespace NUMINAMATH_CALUDE_inequality_proof_l3083_308317

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^3 * (y^2 + z^2)^2 + y^3 * (z^2 + x^2)^2 + z^3 * (x^2 + y^2)^2 ≥
  x * y * z * (x * y * (x + y)^2 + y * z * (y + z)^2 + z * x * (z + x)^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3083_308317


namespace NUMINAMATH_CALUDE_triangle_intersection_invariance_l3083_308345

/-- Represents a right triangle in a plane -/
structure RightTriangle where
  leg1 : Real
  leg2 : Real

/-- Represents a line in a plane -/
structure Line where
  slope : Real
  intercept : Real

/-- Represents the configuration of three right triangles relative to a line -/
structure TriangleConfiguration where
  triangles : Fin 3 → RightTriangle
  base_line : Line
  intersecting_line : Line

/-- Checks if a line intersects three triangles into equal segments -/
def intersects_equally (config : TriangleConfiguration) : Prop :=
  sorry

/-- The main theorem -/
theorem triangle_intersection_invariance 
  (initial_config : TriangleConfiguration)
  (rotated_config : TriangleConfiguration)
  (h1 : intersects_equally initial_config)
  (h2 : ∀ i : Fin 3, 
    (initial_config.triangles i).leg1 = (rotated_config.triangles i).leg2 ∧
    (initial_config.triangles i).leg2 = (rotated_config.triangles i).leg1)
  (h3 : initial_config.base_line = rotated_config.base_line) :
  ∃ new_line : Line, 
    new_line.slope = initial_config.intersecting_line.slope ∧
    intersects_equally { triangles := rotated_config.triangles,
                         base_line := rotated_config.base_line,
                         intersecting_line := new_line } :=
sorry

end NUMINAMATH_CALUDE_triangle_intersection_invariance_l3083_308345


namespace NUMINAMATH_CALUDE_f_is_quadratic_l3083_308346

-- Define a quadratic function
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- Define the specific function
def f (x : ℝ) : ℝ := 11 * x^2 + 29 * x

-- Theorem statement
theorem f_is_quadratic : is_quadratic f := by
  sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l3083_308346


namespace NUMINAMATH_CALUDE_fibonacci_identity_l3083_308364

-- Define Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

theorem fibonacci_identity (θ : ℝ) (x : ℝ) (n : ℕ) 
  (h1 : 0 < θ ∧ θ < π)
  (h2 : x + 1/x = 2 * Real.cos (2 * θ)) :
  x^(fib n) + 1/(x^(fib n)) = 2 * Real.cos (2 * (fib n) * θ) := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_identity_l3083_308364


namespace NUMINAMATH_CALUDE_transformed_system_solution_l3083_308320

/-- Given a system of equations with solution, prove that a transformed system has a specific solution -/
theorem transformed_system_solution (a b m n : ℝ) :
  (∃ x y : ℝ, a * x + b * y = 10 ∧ m * x - n * y = 8 ∧ x = 1 ∧ y = 2) →
  (∃ x y : ℝ, (1/2) * a * (x + y) + (1/3) * b * (x - y) = 10 ∧
              (1/2) * m * (x + y) - (1/3) * n * (x - y) = 8 ∧
              x = 4 ∧ y = -2) :=
by sorry

end NUMINAMATH_CALUDE_transformed_system_solution_l3083_308320


namespace NUMINAMATH_CALUDE_segment_sum_midpoint_inequality_l3083_308339

theorem segment_sum_midpoint_inequality
  (f : ℚ → ℤ) :
  ∃ (x y : ℚ), f x + f y ≤ 2 * f ((x + y) / 2) :=
sorry

end NUMINAMATH_CALUDE_segment_sum_midpoint_inequality_l3083_308339


namespace NUMINAMATH_CALUDE_sphere_surface_area_circumscribing_cube_l3083_308371

theorem sphere_surface_area_circumscribing_cube (edge_length : ℝ) (surface_area : ℝ) :
  edge_length = 2 →
  surface_area = 4 * Real.pi * (((edge_length ^ 2 + edge_length ^ 2 + edge_length ^ 2) / 4) : ℝ) →
  surface_area = 12 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_circumscribing_cube_l3083_308371


namespace NUMINAMATH_CALUDE_existence_of_nine_digit_combination_l3083_308348

theorem existence_of_nine_digit_combination : ∃ (a b c d e f g h i : ℕ),
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧
  f ≠ g ∧ f ≠ h ∧ f ≠ i ∧
  g ≠ h ∧ g ≠ i ∧
  h ≠ i ∧
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10 ∧ g < 10 ∧ h < 10 ∧ i < 100 ∧
  (a + b + c + d + e + f + g + h * i = 100 ∨
   a + b + c + d + e * f + g + h = 100 ∨
   a + b + c + d + e - f - g + h + i = 100) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_nine_digit_combination_l3083_308348


namespace NUMINAMATH_CALUDE_F_and_I_mutually_exclusive_and_complementary_l3083_308311

structure TouristChoice where
  goesToA : Bool
  goesToB : Bool

def E (choice : TouristChoice) : Prop := choice.goesToA ∧ ¬choice.goesToB
def F (choice : TouristChoice) : Prop := choice.goesToA ∨ choice.goesToB
def G (choice : TouristChoice) : Prop := (choice.goesToA ∧ ¬choice.goesToB) ∨ (¬choice.goesToA ∧ choice.goesToB) ∨ (¬choice.goesToA ∧ ¬choice.goesToB)
def H (choice : TouristChoice) : Prop := ¬choice.goesToA
def I (choice : TouristChoice) : Prop := ¬choice.goesToA ∧ ¬choice.goesToB

theorem F_and_I_mutually_exclusive_and_complementary :
  ∀ (choice : TouristChoice),
    (F choice ∧ I choice → False) ∧
    (F choice ∨ I choice) :=
sorry

end NUMINAMATH_CALUDE_F_and_I_mutually_exclusive_and_complementary_l3083_308311


namespace NUMINAMATH_CALUDE_job_arrangements_l3083_308334

/-- The number of ways to arrange n distinct candidates into k distinct jobs,
    where each job requires exactly one person and each person can take only one job. -/
def arrangements (n k : ℕ) : ℕ :=
  Nat.factorial n / Nat.factorial (n - k)

/-- There are 3 different jobs, each requiring only one person,
    and each person taking on only one job.
    There are 4 candidates available for selection. -/
theorem job_arrangements : arrangements 4 3 = 24 := by
  sorry

end NUMINAMATH_CALUDE_job_arrangements_l3083_308334


namespace NUMINAMATH_CALUDE_integer_roots_of_f_l3083_308306

def f (x : ℤ) : ℤ := 4*x^4 - 16*x^3 + 11*x^2 + 4*x - 3

theorem integer_roots_of_f :
  {x : ℤ | f x = 0} = {1, 3} := by sorry

end NUMINAMATH_CALUDE_integer_roots_of_f_l3083_308306


namespace NUMINAMATH_CALUDE_least_integer_with_digit_sum_property_l3083_308303

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: 2999999999999 is the least positive integer N such that
    the sum of its digits is 100 and the sum of the digits of 2N is 110 -/
theorem least_integer_with_digit_sum_property : 
  (∀ m : ℕ, m > 0 ∧ m < 2999999999999 → 
    (sum_of_digits m = 100 ∧ sum_of_digits (2 * m) = 110) → False) ∧ 
  sum_of_digits 2999999999999 = 100 ∧ 
  sum_of_digits (2 * 2999999999999) = 110 :=
sorry

end NUMINAMATH_CALUDE_least_integer_with_digit_sum_property_l3083_308303


namespace NUMINAMATH_CALUDE_sine_difference_inequality_l3083_308318

theorem sine_difference_inequality (A B : Real) (hA : 0 ≤ A ∧ A ≤ π) (hB : 0 ≤ B ∧ B ≤ π) :
  |Real.sin A - Real.sin B| ≤ |Real.sin (A - B)| := by
  sorry

end NUMINAMATH_CALUDE_sine_difference_inequality_l3083_308318


namespace NUMINAMATH_CALUDE_car_discount_proof_l3083_308389

/-- Proves that the initial discount on a car's original price was 30%, given specific selling conditions --/
theorem car_discount_proof (P : ℝ) (D : ℝ) : 
  P > 0 →  -- Original price is positive
  0 ≤ D ∧ D < 1 →  -- Discount is between 0 and 1 (exclusive)
  P * (1 - D) * 1.7 = P * 1.18999999999999993 →  -- Selling price equation
  D = 0.3 := by
sorry

end NUMINAMATH_CALUDE_car_discount_proof_l3083_308389


namespace NUMINAMATH_CALUDE_sqrt_two_plus_abs_diff_solve_quadratic_equation_l3083_308325

-- Part 1
theorem sqrt_two_plus_abs_diff : 
  Real.sqrt 2 * (Real.sqrt 2 + 1) + |Real.sqrt 2 - Real.sqrt 3| = 2 + Real.sqrt 3 := by
  sorry

-- Part 2
theorem solve_quadratic_equation : 
  ∀ x : ℝ, 4 * x^2 = 25 ↔ x = 5/2 ∨ x = -5/2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_plus_abs_diff_solve_quadratic_equation_l3083_308325


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3083_308344

theorem sufficient_not_necessary_condition (a b : ℝ) : 
  (((a > 0 ∧ b > 0) → (a * b > 0)) ∧
   (∃ a b : ℝ, (a * b > 0) ∧ ¬(a > 0 ∧ b > 0))) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3083_308344


namespace NUMINAMATH_CALUDE_inequality_proof_l3083_308383

theorem inequality_proof (a b c d e : ℝ) 
  (h1 : 0 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) (h5 : d ≤ e)
  (h6 : a + b + c + d + e = 1) : 
  a * d + d * c + c * b + b * e + e * a ≤ 1/5 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3083_308383


namespace NUMINAMATH_CALUDE_remainder_x15_plus_1_div_x_plus_1_l3083_308314

theorem remainder_x15_plus_1_div_x_plus_1 (x : ℝ) : (x^15 + 1) % (x + 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_x15_plus_1_div_x_plus_1_l3083_308314


namespace NUMINAMATH_CALUDE_cube_surface_area_l3083_308398

/-- The surface area of a cube with edge length 11 cm is 726 cm². -/
theorem cube_surface_area (edge_length : ℝ) (h : edge_length = 11) :
  6 * edge_length^2 = 726 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l3083_308398


namespace NUMINAMATH_CALUDE_josie_remaining_money_l3083_308354

/-- Calculates the remaining money after Josie's grocery shopping --/
def remaining_money (initial_amount : ℚ) 
  (milk_price : ℚ) (milk_discount : ℚ) 
  (bread_price : ℚ) 
  (detergent_price : ℚ) (detergent_coupon : ℚ) 
  (banana_price_per_pound : ℚ) (banana_pounds : ℚ) : ℚ :=
  let milk_cost := milk_price * (1 - milk_discount)
  let detergent_cost := detergent_price - detergent_coupon
  let banana_cost := banana_price_per_pound * banana_pounds
  let total_cost := milk_cost + bread_price + detergent_cost + banana_cost
  initial_amount - total_cost

/-- Theorem stating that Josie has $4.00 left after shopping --/
theorem josie_remaining_money :
  remaining_money 20 4 (1/2) 3.5 10.25 1.25 0.75 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_josie_remaining_money_l3083_308354


namespace NUMINAMATH_CALUDE_number_of_piglets_born_l3083_308343

def sellPrice : ℕ := 300
def feedCost : ℕ := 10
def profitEarned : ℕ := 960

def pigsSoldAt12Months : ℕ := 3
def pigsSoldAt16Months : ℕ := 3

def totalPigsSold : ℕ := pigsSoldAt12Months + pigsSoldAt16Months

theorem number_of_piglets_born (sellPrice feedCost profitEarned 
  pigsSoldAt12Months pigsSoldAt16Months totalPigsSold : ℕ) :
  sellPrice = 300 →
  feedCost = 10 →
  profitEarned = 960 →
  pigsSoldAt12Months = 3 →
  pigsSoldAt16Months = 3 →
  totalPigsSold = pigsSoldAt12Months + pigsSoldAt16Months →
  totalPigsSold = 6 :=
by sorry

end NUMINAMATH_CALUDE_number_of_piglets_born_l3083_308343


namespace NUMINAMATH_CALUDE_league_score_range_l3083_308313

/-- Represents a sports league -/
structure League where
  numTeams : ℕ
  pointsForWin : ℕ
  pointsForDraw : ℕ

/-- Calculate the total number of games in a double round-robin tournament -/
def totalGames (league : League) : ℕ :=
  league.numTeams * (league.numTeams - 1)

/-- Calculate the minimum possible total score for the league -/
def minTotalScore (league : League) : ℕ :=
  (totalGames league) * (2 * league.pointsForDraw)

/-- Calculate the maximum possible total score for the league -/
def maxTotalScore (league : League) : ℕ :=
  (totalGames league) * league.pointsForWin

/-- Theorem stating that the total score for a 15-team league with 3 points for a win
    and 1 point for a draw is between 420 and 630, inclusive -/
theorem league_score_range :
  let league := League.mk 15 3 1
  420 ≤ minTotalScore league ∧ maxTotalScore league ≤ 630 := by
  sorry

#eval minTotalScore (League.mk 15 3 1)
#eval maxTotalScore (League.mk 15 3 1)

end NUMINAMATH_CALUDE_league_score_range_l3083_308313


namespace NUMINAMATH_CALUDE_projection_matrix_condition_l3083_308378

/-- A projection matrix is idempotent (P^2 = P) -/
def is_projection_matrix (P : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  P * P = P

/-- The specific matrix form given in the problem -/
def P (a c : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![a, 20/49], ![c, 29/49]]

/-- The theorem stating the conditions for the given matrix to be a projection matrix -/
theorem projection_matrix_condition (a c : ℚ) :
  is_projection_matrix (P a c) ↔ a = 1 ∧ c = 0 := by
  sorry

#check projection_matrix_condition

end NUMINAMATH_CALUDE_projection_matrix_condition_l3083_308378


namespace NUMINAMATH_CALUDE_correct_prediction_probability_l3083_308351

theorem correct_prediction_probability :
  let n_monday : ℕ := 5
  let n_tuesday : ℕ := 6
  let n_total : ℕ := n_monday + n_tuesday
  let n_correct : ℕ := 7
  let n_correct_monday : ℕ := 3
  let n_correct_tuesday : ℕ := n_correct - n_correct_monday
  let p : ℝ := 1 / 2

  (Nat.choose n_monday n_correct_monday * p^n_monday * (1-p)^(n_monday - n_correct_monday)) *
  (Nat.choose n_tuesday n_correct_tuesday * p^n_tuesday * (1-p)^(n_tuesday - n_correct_tuesday)) /
  (Nat.choose n_total n_correct * p^n_correct * (1-p)^(n_total - n_correct)) = 5 / 11 :=
by
  sorry

end NUMINAMATH_CALUDE_correct_prediction_probability_l3083_308351


namespace NUMINAMATH_CALUDE_median_equations_l3083_308327

/-- Triangle ABC with given coordinates -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Equation of a line in general form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given triangle ABC -/
def givenTriangle : Triangle :=
  { A := (1, -4)
  , B := (6, 6)
  , C := (-2, 0) }

/-- Theorem stating the equations of the two medians -/
theorem median_equations (t : Triangle) 
  (h : t = givenTriangle) : 
  ∃ (l1 l2 : LineEquation),
    (l1.a = 6 ∧ l1.b = -8 ∧ l1.c = -13) ∧
    (l2.a = 7 ∧ l2.b = -1 ∧ l2.c = -11) :=
  sorry

end NUMINAMATH_CALUDE_median_equations_l3083_308327


namespace NUMINAMATH_CALUDE_point_symmetry_l3083_308385

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = -q.2

theorem point_symmetry (a b : ℝ) :
  symmetric_wrt_origin (3, a - 2) (b, a) → a + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_point_symmetry_l3083_308385


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l3083_308387

theorem binomial_expansion_coefficient (n : ℕ) : 
  (8 * (Nat.choose n 3) * 2^3 = 16 * n) → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l3083_308387


namespace NUMINAMATH_CALUDE_multitive_function_thirtysix_l3083_308379

/-- A function satisfying f(a · b) = f(a) + f(b) -/
def MultitiveFunction (f : ℝ → ℝ) : Prop :=
  ∀ a b, f (a * b) = f a + f b

/-- Theorem: Given a multitive function f with f(2) = p and f(3) = q, prove f(36) = 2(p + q) -/
theorem multitive_function_thirtysix
  (f : ℝ → ℝ) (p q : ℝ)
  (hf : MultitiveFunction f)
  (h2 : f 2 = p)
  (h3 : f 3 = q) :
  f 36 = 2 * (p + q) := by
  sorry

end NUMINAMATH_CALUDE_multitive_function_thirtysix_l3083_308379


namespace NUMINAMATH_CALUDE_acute_angle_condition_x_plus_y_value_l3083_308388

-- Define the vectors
def a : Fin 2 → ℝ := ![2, -1]
def b (x : ℝ) : Fin 2 → ℝ := ![x, 1]

-- Define the dot product
def dot_product (u v : Fin 2 → ℝ) : ℝ := (u 0) * (v 0) + (u 1) * (v 1)

-- Theorem 1: Acute angle condition
theorem acute_angle_condition (x : ℝ) :
  (dot_product a (b x) > 0) ↔ (x > 1/2) := by sorry

-- Theorem 2: Value of x + y
theorem x_plus_y_value (x y : ℝ) :
  (3 • a - 2 • (b x) = ![4, y]) → x + y = -4 := by sorry

end NUMINAMATH_CALUDE_acute_angle_condition_x_plus_y_value_l3083_308388


namespace NUMINAMATH_CALUDE_flag_distribution_theorem_l3083_308363

/-- Represents the box of flags -/
structure FlagBox where
  total : ℕ
  blue : ℕ
  red : ℕ

/-- Represents the distribution of flags among children -/
structure FlagDistribution where
  total : ℕ
  blue : ℕ
  red : ℕ
  both : ℕ

def is_valid_box (box : FlagBox) : Prop :=
  box.total = box.blue + box.red ∧ box.total % 2 = 0

def is_valid_distribution (box : FlagBox) (dist : FlagDistribution) : Prop :=
  dist.total = box.total / 2 ∧
  dist.blue = (6 * dist.total) / 10 ∧
  dist.red = (6 * dist.total) / 10 ∧
  dist.total = dist.blue + dist.red - dist.both

theorem flag_distribution_theorem (box : FlagBox) (dist : FlagDistribution) :
  is_valid_box box → is_valid_distribution box dist →
  dist.both = dist.total / 5 :=
sorry

end NUMINAMATH_CALUDE_flag_distribution_theorem_l3083_308363


namespace NUMINAMATH_CALUDE_marilyn_bottle_caps_l3083_308369

/-- The number of bottle caps Marilyn shared -/
def shared_caps : ℕ := 36

/-- The number of bottle caps Marilyn ended up with -/
def remaining_caps : ℕ := 15

/-- The initial number of bottle caps Marilyn had -/
def initial_caps : ℕ := shared_caps + remaining_caps

theorem marilyn_bottle_caps : initial_caps = 51 := by
  sorry

end NUMINAMATH_CALUDE_marilyn_bottle_caps_l3083_308369


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3083_308323

theorem rationalize_denominator : 
  1 / (2 - Real.sqrt 2) = (2 + Real.sqrt 2) / 2 := by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3083_308323


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_mod_15_l3083_308394

theorem arithmetic_sequence_sum_mod_15 (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) (n : ℕ) :
  a₁ = 2 →
  d = 5 →
  aₙ = 102 →
  n * (a₁ + aₙ) / 2 ≡ 12 [MOD 15] :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_mod_15_l3083_308394


namespace NUMINAMATH_CALUDE_problem_2009_2007_2008_l3083_308324

theorem problem_2009_2007_2008 : 2009 * (2007 / 2008) + 1 / 2008 = 2008 := by
  sorry

end NUMINAMATH_CALUDE_problem_2009_2007_2008_l3083_308324


namespace NUMINAMATH_CALUDE_existence_of_divisible_m_l3083_308341

theorem existence_of_divisible_m : ∃ m : ℕ+, (3^100 * m.val + 3^100 - 1) % 1988 = 0 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_divisible_m_l3083_308341


namespace NUMINAMATH_CALUDE_composite_divisor_bound_l3083_308360

/-- A number is composite if it's a natural number greater than 1 and not prime -/
def IsComposite (n : ℕ) : Prop :=
  n > 1 ∧ ¬ Nat.Prime n

/-- Theorem: Every composite number has a divisor greater than 1 but not greater than its square root -/
theorem composite_divisor_bound {n : ℕ} (h : IsComposite n) :
  ∃ d : ℕ, d ∣ n ∧ 1 < d ∧ d ≤ Real.sqrt (n : ℝ) :=
sorry

end NUMINAMATH_CALUDE_composite_divisor_bound_l3083_308360


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3083_308390

theorem quadratic_equation_roots (m : ℚ) :
  (∃ x : ℚ, x^2 + 2*x + 3*m - 4 = 0) ∧ 
  (2^2 + 2*2 + 3*m - 4 = 0) →
  ((-4)^2 + 2*(-4) + 3*m - 4 = 0) ∧ 
  m = -4/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3083_308390


namespace NUMINAMATH_CALUDE_binomial_square_constant_l3083_308326

/-- If x^2 + 80x + c is equal to the square of a binomial, then c = 1600 -/
theorem binomial_square_constant (c : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 80*x + c = (x + a)^2) → c = 1600 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_constant_l3083_308326


namespace NUMINAMATH_CALUDE_factorial_divisor_sum_l3083_308302

theorem factorial_divisor_sum (n : ℕ) :
  ∀ k : ℕ, k ≤ n.factorial → ∃ (s : Finset ℕ),
    (∀ x ∈ s, x ∣ n.factorial) ∧
    s.card ≤ n ∧
    k = s.sum id :=
sorry

end NUMINAMATH_CALUDE_factorial_divisor_sum_l3083_308302


namespace NUMINAMATH_CALUDE_divisible_by_eleven_l3083_308377

theorem divisible_by_eleven (n : ℕ) : 
  11 ∣ (6^(2*n) + 3^(n+2) + 3^n) := by
sorry

end NUMINAMATH_CALUDE_divisible_by_eleven_l3083_308377


namespace NUMINAMATH_CALUDE_percentage_men_undeclared_l3083_308332

/-- Represents the percentages of students in different majors and categories -/
structure ClassComposition where
  men_science : ℝ
  men_humanities : ℝ
  men_business : ℝ
  men_double_science_humanities : ℝ
  men_double_science_business : ℝ
  men_double_humanities_business : ℝ

/-- Theorem stating the percentage of men with undeclared majors -/
theorem percentage_men_undeclared (c : ClassComposition) : 
  c.men_science = 24 ∧ 
  c.men_humanities = 13 ∧ 
  c.men_business = 18 ∧
  c.men_double_science_humanities = 13.5 ∧
  c.men_double_science_business = 9 ∧
  c.men_double_humanities_business = 6.75 →
  100 - (c.men_science + c.men_humanities + c.men_business + 
         c.men_double_science_humanities + c.men_double_science_business + 
         c.men_double_humanities_business) = 15.75 := by
  sorry

#check percentage_men_undeclared

end NUMINAMATH_CALUDE_percentage_men_undeclared_l3083_308332


namespace NUMINAMATH_CALUDE_cost_of_20_pencils_12_notebooks_l3083_308393

/-- The cost of a pencil in dollars -/
def pencil_cost : ℚ := sorry

/-- The cost of a notebook in dollars -/
def notebook_cost : ℚ := sorry

/-- The first condition: 8 pencils and 10 notebooks cost $5.20 -/
axiom condition1 : 8 * pencil_cost + 10 * notebook_cost = 5.20

/-- The second condition: 6 pencils and 4 notebooks cost $2.24 -/
axiom condition2 : 6 * pencil_cost + 4 * notebook_cost = 2.24

/-- The theorem to prove -/
theorem cost_of_20_pencils_12_notebooks : 
  20 * pencil_cost + 12 * notebook_cost = 6.84 := by sorry

end NUMINAMATH_CALUDE_cost_of_20_pencils_12_notebooks_l3083_308393


namespace NUMINAMATH_CALUDE_staples_left_l3083_308353

def initial_staples : ℕ := 50
def dozen : ℕ := 12
def reports_stapled : ℕ := 3 * dozen

theorem staples_left : initial_staples - reports_stapled = 14 := by
  sorry

end NUMINAMATH_CALUDE_staples_left_l3083_308353


namespace NUMINAMATH_CALUDE_larger_number_problem_l3083_308328

theorem larger_number_problem (L S : ℕ) (hL : L > S) :
  L - S = 1365 →
  L = 6 * S + 15 →
  L = 1635 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l3083_308328


namespace NUMINAMATH_CALUDE_finleys_age_l3083_308380

/-- Proves Finley's age given the conditions in the problem -/
theorem finleys_age (jill_age : ℕ) (roger_age : ℕ) (finley_age : ℕ) : 
  jill_age = 20 →
  roger_age = 2 * jill_age + 5 →
  (roger_age + 15) - (jill_age + 15) = finley_age - 30 →
  finley_age = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_finleys_age_l3083_308380


namespace NUMINAMATH_CALUDE_first_player_can_avoid_losing_l3083_308399

/-- A strategy for selecting vectors -/
def Strategy := List (ℝ × ℝ) → ℝ × ℝ

/-- The game state, including all vectors and the current player's turn -/
structure GameState where
  vectors : List (ℝ × ℝ)
  player_turn : ℕ

/-- The result of the game -/
inductive GameResult
  | FirstPlayerWins
  | SecondPlayerWins
  | Draw

/-- Play the game with given strategies -/
def play_game (initial_vectors : List (ℝ × ℝ)) (strategy1 strategy2 : Strategy) : GameResult :=
  sorry

/-- Theorem stating that the first player can always avoid losing -/
theorem first_player_can_avoid_losing (vectors : List (ℝ × ℝ)) 
  (h : vectors.length = 1992) : 
  ∃ (strategy1 : Strategy), ∀ (strategy2 : Strategy),
    play_game vectors strategy1 strategy2 ≠ GameResult.SecondPlayerWins :=
  sorry

end NUMINAMATH_CALUDE_first_player_can_avoid_losing_l3083_308399


namespace NUMINAMATH_CALUDE_abs_neg_one_wrt_one_abs_wrt_one_eq_2023_l3083_308367

-- Define the absolute value with respect to 1
def abs_wrt_one (a : ℝ) : ℝ := |a - 1|

-- Theorem 1
theorem abs_neg_one_wrt_one : abs_wrt_one (-1) = 2 := by sorry

-- Theorem 2
theorem abs_wrt_one_eq_2023 (a : ℝ) : 
  abs_wrt_one a = 2023 ↔ a = 2024 ∨ a = -2022 := by sorry

end NUMINAMATH_CALUDE_abs_neg_one_wrt_one_abs_wrt_one_eq_2023_l3083_308367


namespace NUMINAMATH_CALUDE_S_infinite_l3083_308333

/-- Sum of divisors function -/
def sigma (n : ℕ) : ℕ := sorry

/-- The set of natural numbers n such that σ(n)/n > σ(k)/k for all k < n -/
def S : Set ℕ :=
  {n : ℕ | ∀ k < n, (sigma n : ℚ) / n > (sigma k : ℚ) / k}

/-- Theorem stating that S is infinite -/
theorem S_infinite : Set.Infinite S := by sorry

end NUMINAMATH_CALUDE_S_infinite_l3083_308333


namespace NUMINAMATH_CALUDE_number_of_divisors_of_fourth_power_l3083_308384

/-- Given a positive integer n where n = p₁ * p₂² * p₃⁵ and p₁, p₂, and p₃ are different prime numbers,
    the number of positive divisors of x = n⁴ is 945. -/
theorem number_of_divisors_of_fourth_power (p₁ p₂ p₃ : Nat) (h_prime₁ : Prime p₁) (h_prime₂ : Prime p₂)
    (h_prime₃ : Prime p₃) (h_distinct : p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃) :
    let n := p₁ * p₂^2 * p₃^5
    let x := n^4
    (Nat.divisors x).card = 945 := by
  sorry

#check number_of_divisors_of_fourth_power

end NUMINAMATH_CALUDE_number_of_divisors_of_fourth_power_l3083_308384


namespace NUMINAMATH_CALUDE_scavenger_hunt_difference_l3083_308338

theorem scavenger_hunt_difference (lewis_items samantha_items tanya_items : ℕ) : 
  lewis_items = 20 →
  samantha_items = 4 * tanya_items →
  tanya_items = 4 →
  lewis_items - samantha_items = 4 := by
sorry

end NUMINAMATH_CALUDE_scavenger_hunt_difference_l3083_308338


namespace NUMINAMATH_CALUDE_other_coin_denomination_l3083_308376

/-- Given the following conditions:
    - There are 344 coins in total
    - The total value of all coins is 7100 paise (Rs. 71)
    - There are 300 coins of 20 paise each
    - There are two types of coins: 20 paise and another unknown denomination
    Prove that the denomination of the other type of coin is 25 paise -/
theorem other_coin_denomination
  (total_coins : ℕ)
  (total_value : ℕ)
  (twenty_paise_coins : ℕ)
  (h_total_coins : total_coins = 344)
  (h_total_value : total_value = 7100)
  (h_twenty_paise_coins : twenty_paise_coins = 300)
  : (total_value - twenty_paise_coins * 20) / (total_coins - twenty_paise_coins) = 25 := by
  sorry

end NUMINAMATH_CALUDE_other_coin_denomination_l3083_308376


namespace NUMINAMATH_CALUDE_hyperbola_properties_l3083_308307

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop :=
  y^2 / 12 - x^2 / 27 = 1

-- Define the asymptote equation
def asymptote_equation (x y : ℝ) : Prop :=
  y = (2/3) * x ∨ y = -(2/3) * x

-- Theorem statement
theorem hyperbola_properties :
  (∀ x y : ℝ, asymptote_equation x y ↔ (y^2 / x^2 = 4/9)) ∧
  hyperbola_equation 3 4 :=
sorry


end NUMINAMATH_CALUDE_hyperbola_properties_l3083_308307


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3083_308395

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The main theorem -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_sum : a 3 + a 5 + a 7 + a 9 + a 11 = 100) :
  3 * a 9 - a 13 = 40 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3083_308395


namespace NUMINAMATH_CALUDE_unique_solution_l3083_308331

theorem unique_solution : 
  ∀ (a b : ℕ), 
    a > 1 → 
    b > 0 → 
    b ∣ (a - 1) → 
    (2 * a + 1) ∣ (5 * b - 3) → 
    a = 10 ∧ b = 9 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3083_308331


namespace NUMINAMATH_CALUDE_opposite_unit_vector_l3083_308359

def vec_a : ℝ × ℝ := (4, 2)

theorem opposite_unit_vector :
  let opposite_unit := (-vec_a.1 / Real.sqrt (vec_a.1^2 + vec_a.2^2),
                        -vec_a.2 / Real.sqrt (vec_a.1^2 + vec_a.2^2))
  opposite_unit = (-2 * Real.sqrt 5 / 5, -Real.sqrt 5 / 5) :=
by sorry

end NUMINAMATH_CALUDE_opposite_unit_vector_l3083_308359


namespace NUMINAMATH_CALUDE_alok_ice_cream_order_l3083_308336

/-- The number of ice-cream cups ordered by Alok -/
def ice_cream_cups (chapatis rice mixed_veg : ℕ) 
  (chapati_cost rice_cost mixed_veg_cost ice_cream_cost total_paid : ℕ) : ℕ :=
  (total_paid - (chapatis * chapati_cost + rice * rice_cost + mixed_veg * mixed_veg_cost)) / ice_cream_cost

/-- Theorem stating that Alok ordered 6 ice-cream cups -/
theorem alok_ice_cream_order : 
  ice_cream_cups 16 5 7 6 45 70 40 1051 = 6 := by
  sorry

end NUMINAMATH_CALUDE_alok_ice_cream_order_l3083_308336


namespace NUMINAMATH_CALUDE_polynomial_factorization_and_range_l3083_308309

-- Define the polynomial and factored form
def P (x : ℝ) := x^3 - 2*x^2 - x + 2
def Q (a b c x : ℝ) := (x + a) * (x + b) * (x + c)

-- State the theorem
theorem polynomial_factorization_and_range :
  ∃ (a b c : ℝ),
    (∀ x, P x = Q a b c x) ∧
    (a > b) ∧ (b > c) ∧
    (a = 1) ∧ (b = -1) ∧ (c = -2) ∧
    (∀ x ∈ Set.Icc 0 3, a*x^2 + 2*b*x + c ∈ Set.Icc (-3) 1) ∧
    (∃ x₁ ∈ Set.Icc 0 3, a*x₁^2 + 2*b*x₁ + c = -3) ∧
    (∃ x₂ ∈ Set.Icc 0 3, a*x₂^2 + 2*b*x₂ + c = 1) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_and_range_l3083_308309


namespace NUMINAMATH_CALUDE_crystal_cupcake_sales_l3083_308304

def crystal_sales (original_cupcake_price original_cookie_price : ℚ)
  (price_reduction_factor : ℚ) (total_revenue : ℚ) (cookies_sold : ℕ) : Prop :=
  let reduced_cupcake_price := original_cupcake_price * price_reduction_factor
  let reduced_cookie_price := original_cookie_price * price_reduction_factor
  let cookie_revenue := reduced_cookie_price * cookies_sold
  let cupcake_revenue := total_revenue - cookie_revenue
  let cupcakes_sold := cupcake_revenue / reduced_cupcake_price
  cupcakes_sold = 16

theorem crystal_cupcake_sales :
  crystal_sales 3 2 (1/2) 32 8 := by sorry

end NUMINAMATH_CALUDE_crystal_cupcake_sales_l3083_308304


namespace NUMINAMATH_CALUDE_jeans_business_weekly_hours_l3083_308375

/-- Represents the operating hours of a business for a single day -/
structure DailyHours where
  open_time : Nat
  close_time : Nat

/-- Calculates the number of hours a business is open in a day -/
def hours_open (dh : DailyHours) : Nat :=
  dh.close_time - dh.open_time

/-- Represents the operating hours of Jean's business for a week -/
structure WeeklyHours where
  weekday_hours : DailyHours
  weekend_hours : DailyHours

/-- Calculates the total number of hours Jean's business is open in a week -/
def total_weekly_hours (wh : WeeklyHours) : Nat :=
  (hours_open wh.weekday_hours * 5) + (hours_open wh.weekend_hours * 2)

/-- Jean's business hours -/
def jeans_business : WeeklyHours :=
  { weekday_hours := { open_time := 16, close_time := 22 }
  , weekend_hours := { open_time := 18, close_time := 22 } }

theorem jeans_business_weekly_hours :
  total_weekly_hours jeans_business = 38 := by
  sorry

end NUMINAMATH_CALUDE_jeans_business_weekly_hours_l3083_308375


namespace NUMINAMATH_CALUDE_soda_price_calculation_l3083_308392

/-- Proves that the original price of each soda is $20/9 given the conditions of the problem -/
theorem soda_price_calculation (num_sodas : ℕ) (discount_rate : ℚ) (total_paid : ℚ) :
  num_sodas = 3 →
  discount_rate = 1/10 →
  total_paid = 6 →
  ∃ (original_price : ℚ), 
    original_price = 20/9 ∧ 
    num_sodas * (original_price * (1 - discount_rate)) = total_paid :=
by sorry

end NUMINAMATH_CALUDE_soda_price_calculation_l3083_308392


namespace NUMINAMATH_CALUDE_square_and_arithmetic_computation_l3083_308386

theorem square_and_arithmetic_computation : 7^2 - (4 * 6) / 2 + 6^2 = 73 := by
  sorry

end NUMINAMATH_CALUDE_square_and_arithmetic_computation_l3083_308386


namespace NUMINAMATH_CALUDE_negative_two_a_cubed_l3083_308391

theorem negative_two_a_cubed (a : ℝ) : (-2 * a)^3 = -8 * a^3 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_a_cubed_l3083_308391


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3083_308374

theorem geometric_sequence_sum (n : ℕ) (S : ℕ → ℝ) :
  S n = 48 ∧ S (2 * n) = 60 →
  S (3 * n) = 63 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3083_308374


namespace NUMINAMATH_CALUDE_jack_and_jill_speed_l3083_308312

/-- Jack and Jill's Mountain Climb Theorem -/
theorem jack_and_jill_speed (x : ℝ) : 
  (x^2 - 13*x - 26 = (x^2 - 5*x - 66) / (x + 8)) → 
  (x^2 - 13*x - 26 = 4) := by
  sorry

#check jack_and_jill_speed

end NUMINAMATH_CALUDE_jack_and_jill_speed_l3083_308312


namespace NUMINAMATH_CALUDE_sandy_scooter_price_l3083_308347

/-- The initial price of Sandy's scooter -/
def initial_price : ℝ := 800

/-- The cost of repairs -/
def repair_cost : ℝ := 200

/-- The selling price of the scooter -/
def selling_price : ℝ := 1200

/-- The gain percentage -/
def gain_percent : ℝ := 20

theorem sandy_scooter_price :
  ∃ (P : ℝ),
    P = initial_price ∧
    selling_price = (1 + gain_percent / 100) * (P + repair_cost) :=
by sorry

end NUMINAMATH_CALUDE_sandy_scooter_price_l3083_308347


namespace NUMINAMATH_CALUDE_walter_zoo_time_l3083_308321

theorem walter_zoo_time (S : ℝ) : 
  S > 0 ∧ 
  S + 8*S + 13 + S/2 = 185 → 
  S = 16 :=
by sorry

end NUMINAMATH_CALUDE_walter_zoo_time_l3083_308321


namespace NUMINAMATH_CALUDE_perfect_squares_mod_six_l3083_308316

theorem perfect_squares_mod_six :
  (∀ n : ℤ, n^2 % 6 ≠ 2) ∧
  (∃ K : Set ℤ, Set.Infinite K ∧ ∀ k ∈ K, ((6 * k + 3)^2) % 6 = 3) :=
by sorry

end NUMINAMATH_CALUDE_perfect_squares_mod_six_l3083_308316


namespace NUMINAMATH_CALUDE_melted_ice_cream_depth_l3083_308340

/-- Given a sphere of ice cream that melts into a cylinder, calculate the height of the cylinder -/
theorem melted_ice_cream_depth (initial_radius final_radius : ℝ) 
  (initial_radius_pos : 0 < initial_radius)
  (final_radius_pos : 0 < final_radius)
  (h_initial_radius : initial_radius = 3)
  (h_final_radius : final_radius = 12) :
  (4 / 3 * Real.pi * initial_radius ^ 3) / (Real.pi * final_radius ^ 2) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_melted_ice_cream_depth_l3083_308340


namespace NUMINAMATH_CALUDE_hyperbola_vertices_distance_l3083_308365

/-- The distance between the vertices of a hyperbola -/
def distance_between_vertices (a : ℝ) : ℝ := 2 * a

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 36 - y^2 / 16 = 1

theorem hyperbola_vertices_distance :
  ∃ (a : ℝ), a^2 = 36 ∧ distance_between_vertices a = 12 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_vertices_distance_l3083_308365


namespace NUMINAMATH_CALUDE_one_mile_equals_500_rods_l3083_308319

/-- Conversion factor from miles to furlongs -/
def mile_to_furlong : ℚ := 10

/-- Conversion factor from furlongs to rods -/
def furlong_to_rod : ℚ := 50

/-- The number of rods in one mile -/
def rods_in_mile : ℚ := mile_to_furlong * furlong_to_rod

/-- Theorem stating that one mile is equal to 500 rods -/
theorem one_mile_equals_500_rods : rods_in_mile = 500 := by
  sorry

end NUMINAMATH_CALUDE_one_mile_equals_500_rods_l3083_308319


namespace NUMINAMATH_CALUDE_magazine_boxes_l3083_308329

theorem magazine_boxes (total_magazines : ℕ) (magazines_per_box : ℕ) (h1 : total_magazines = 63) (h2 : magazines_per_box = 9) :
  total_magazines / magazines_per_box = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_magazine_boxes_l3083_308329


namespace NUMINAMATH_CALUDE_softball_team_savings_l3083_308366

/-- Calculates the savings for a softball team when buying uniforms with a group discount -/
theorem softball_team_savings 
  (regular_shirt_price regular_pants_price regular_socks_price : ℚ)
  (discounted_shirt_price discounted_pants_price discounted_socks_price : ℚ)
  (team_size : ℕ)
  (h_regular_shirt : regular_shirt_price = 7.5)
  (h_regular_pants : regular_pants_price = 15)
  (h_regular_socks : regular_socks_price = 4.5)
  (h_discounted_shirt : discounted_shirt_price = 6.75)
  (h_discounted_pants : discounted_pants_price = 13.5)
  (h_discounted_socks : discounted_socks_price = 3.75)
  (h_team_size : team_size = 12) :
  let regular_uniform_cost := regular_shirt_price + regular_pants_price + regular_socks_price
  let discounted_uniform_cost := discounted_shirt_price + discounted_pants_price + discounted_socks_price
  let savings_per_uniform := regular_uniform_cost - discounted_uniform_cost
  let total_savings := savings_per_uniform * team_size
  total_savings = 36 := by sorry


end NUMINAMATH_CALUDE_softball_team_savings_l3083_308366


namespace NUMINAMATH_CALUDE_infinitely_many_consecutive_epsilon_squarish_l3083_308350

/-- A positive integer is ε-squarish if it's the product of two integers a and b
    where 1 < a < b < (1 + ε)a -/
def IsEpsilonSquarish (ε : ℝ) (k : ℕ) : Prop :=
  ∃ (a b : ℕ), k = a * b ∧ 1 < a ∧ a < b ∧ b < (1 + ε) * a

/-- There exist infinitely many positive integers n such that
    n², n² - 1, n² - 2, n² - 3, n² - 4, and n² - 5 are all ε-squarish -/
theorem infinitely_many_consecutive_epsilon_squarish (ε : ℝ) (hε : ε > 0) :
  ∀ N : ℕ, ∃ n : ℕ, n > N ∧
    IsEpsilonSquarish ε (n^2) ∧
    IsEpsilonSquarish ε (n^2 - 1) ∧
    IsEpsilonSquarish ε (n^2 - 2) ∧
    IsEpsilonSquarish ε (n^2 - 3) ∧
    IsEpsilonSquarish ε (n^2 - 4) ∧
    IsEpsilonSquarish ε (n^2 - 5) :=
by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_consecutive_epsilon_squarish_l3083_308350


namespace NUMINAMATH_CALUDE_only_sqrt_8_not_simplest_l3083_308337

-- Define a function to check if a number is a perfect square
def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

-- Define a function to check if a radical is in its simplest form
def isSimplestForm (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 → m * m ∣ n → ¬isPerfectSquare m

-- Theorem statement
theorem only_sqrt_8_not_simplest : 
  isSimplestForm 10 ∧ 
  isSimplestForm 6 ∧ 
  isSimplestForm 2 ∧ 
  ¬isSimplestForm 8 :=
sorry

end NUMINAMATH_CALUDE_only_sqrt_8_not_simplest_l3083_308337


namespace NUMINAMATH_CALUDE_total_bowling_balls_l3083_308305

def red_balls : ℕ := 30
def green_balls : ℕ := red_balls + 6

theorem total_bowling_balls : red_balls + green_balls = 66 := by
  sorry

end NUMINAMATH_CALUDE_total_bowling_balls_l3083_308305


namespace NUMINAMATH_CALUDE_max_profit_at_60_l3083_308342

/-- The profit function for a travel agency chartering a plane -/
def profit (x : ℕ) : ℝ :=
  if x ≤ 30 then
    900 * x - 15000
  else if x ≤ 75 then
    (-10 * x + 1200) * x - 15000
  else
    0

/-- The maximum number of people allowed in the tour group -/
def max_people : ℕ := 75

/-- The charter fee for the travel agency -/
def charter_fee : ℝ := 15000

theorem max_profit_at_60 :
  ∀ x : ℕ, x ≤ max_people → profit x ≤ profit 60 ∧ profit 60 = 21000 :=
sorry

end NUMINAMATH_CALUDE_max_profit_at_60_l3083_308342


namespace NUMINAMATH_CALUDE_intersection_with_complement_l3083_308373

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}
def A : Set Nat := {3, 4, 5}
def B : Set Nat := {1, 3, 6}

theorem intersection_with_complement : A ∩ (U \ B) = {4, 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l3083_308373


namespace NUMINAMATH_CALUDE_simplify_fraction_l3083_308300

theorem simplify_fraction (x : ℝ) (h : x ≠ 1) : 
  (x^2 + 1) / (x - 1) - 2 * x / (x - 1) = x - 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3083_308300


namespace NUMINAMATH_CALUDE_function_inequality_l3083_308358

theorem function_inequality (f : ℝ → ℝ) (h_diff : Differentiable ℝ f) 
  (h_cond : ∀ x, (x^2 - 3*x + 2) * deriv f x ≤ 0) :
  ∀ x ∈ Set.Icc 1 2, f 1 ≤ f x ∧ f x ≤ f 2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3083_308358


namespace NUMINAMATH_CALUDE_max_necklaces_proof_l3083_308382

def necklace_green_beads : ℕ := 9
def necklace_white_beads : ℕ := 6
def necklace_orange_beads : ℕ := 3
def available_beads : ℕ := 45

def max_necklaces : ℕ := 5

theorem max_necklaces_proof :
  min (available_beads / necklace_green_beads)
      (min (available_beads / necklace_white_beads)
           (available_beads / necklace_orange_beads)) = max_necklaces := by
  sorry

end NUMINAMATH_CALUDE_max_necklaces_proof_l3083_308382


namespace NUMINAMATH_CALUDE_cube_side_length_when_volume_equals_surface_area_l3083_308352

/-- For a cube where the numerical value of its volume equals the numerical value of its surface area, the side length is 6 units. -/
theorem cube_side_length_when_volume_equals_surface_area :
  ∀ s : ℝ, s > 0 → s^3 = 6 * s^2 → s = 6 := by
  sorry

end NUMINAMATH_CALUDE_cube_side_length_when_volume_equals_surface_area_l3083_308352


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l3083_308322

theorem cubic_equation_solutions :
  let f : ℝ → ℝ := λ x => x^3 - 8
  let g : ℝ → ℝ := λ x => 16 * (x + 1)^(1/3)
  ∀ x : ℝ, f x = g x ↔ x = -2 ∨ x = 1 + Real.sqrt 5 ∨ x = 1 - Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l3083_308322


namespace NUMINAMATH_CALUDE_twelve_returning_sequences_l3083_308301

-- Define the triangle T'
structure Triangle :=
  (v1 v2 v3 : ℝ × ℝ)

-- Define the set of transformations
inductive Transformation
  | Rotate60 : Transformation
  | Rotate120 : Transformation
  | Rotate240 : Transformation
  | ReflectYeqX : Transformation
  | ReflectYeqNegX : Transformation

-- Define a sequence of three transformations
def TransformationSequence := (Transformation × Transformation × Transformation)

-- Define the original triangle T'
def T' : Triangle :=
  { v1 := (1, 1), v2 := (5, 1), v3 := (1, 4) }

-- Function to check if a sequence of transformations returns T' to its original position
def returnsToOriginal (seq : TransformationSequence) : Prop :=
  sorry

-- Theorem stating that exactly 12 sequences return T' to its original position
theorem twelve_returning_sequences :
  ∃ (S : Finset TransformationSequence),
    (∀ seq ∈ S, returnsToOriginal seq) ∧
    (∀ seq, returnsToOriginal seq → seq ∈ S) ∧
    Finset.card S = 12 :=
  sorry

end NUMINAMATH_CALUDE_twelve_returning_sequences_l3083_308301


namespace NUMINAMATH_CALUDE_balloon_radius_increase_l3083_308315

theorem balloon_radius_increase (c₁ c₂ : ℝ) (h₁ : c₁ = 24) (h₂ : c₂ = 30) :
  (c₂ / (2 * π)) - (c₁ / (2 * π)) = 3 / π := by sorry

end NUMINAMATH_CALUDE_balloon_radius_increase_l3083_308315


namespace NUMINAMATH_CALUDE_apples_picked_total_l3083_308330

/-- The number of apples Benny picked -/
def benny_apples : ℕ := 2

/-- The number of apples Dan picked -/
def dan_apples : ℕ := 9

/-- The total number of apples picked -/
def total_apples : ℕ := benny_apples + dan_apples

theorem apples_picked_total :
  total_apples = 11 := by sorry

end NUMINAMATH_CALUDE_apples_picked_total_l3083_308330


namespace NUMINAMATH_CALUDE_sum_cube_value_l3083_308381

theorem sum_cube_value (x y : ℝ) (h1 : x * (x + y) = 49) (h2 : y * (x + y) = 63) :
  (x + y)^3 = 448 * Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_sum_cube_value_l3083_308381


namespace NUMINAMATH_CALUDE_billy_basketball_points_difference_l3083_308356

theorem billy_basketball_points_difference : 
  ∀ (billy_points friend_points : ℕ),
    billy_points = 7 →
    friend_points = 9 →
    friend_points - billy_points = 2 := by
  sorry

end NUMINAMATH_CALUDE_billy_basketball_points_difference_l3083_308356


namespace NUMINAMATH_CALUDE_find_b_value_l3083_308361

/-- Given two functions p and q, where p(x) = 2x - 11 and q(x) = 5x - b,
    prove that b = 8 when p(q(3)) = 3. -/
theorem find_b_value (b : ℝ) : 
  let p : ℝ → ℝ := λ x ↦ 2 * x - 11
  let q : ℝ → ℝ := λ x ↦ 5 * x - b
  p (q 3) = 3 → b = 8 := by
sorry

end NUMINAMATH_CALUDE_find_b_value_l3083_308361


namespace NUMINAMATH_CALUDE_ellipse_properties_l3083_308357

noncomputable section

-- Define the ellipse C
def ellipse (a b : ℝ) (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

-- Define eccentricity
def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x - y + Real.sqrt 2 = 0

-- Define the theorem
theorem ellipse_properties
  (a b : ℝ) 
  (h_ab : a > b ∧ b > 0)
  (h_ecc : eccentricity a b = Real.sqrt 3 / 2)
  (h_tangent : ∃ (x y : ℝ), x^2 + y^2 = b^2 ∧ tangent_line x y) :
  -- 1. Equation of C
  (∀ x y, ellipse a b x y ↔ x^2/4 + y^2 = 1) ∧
  -- 2. Range of slope k
  (∀ M N E : ℝ × ℝ,
    ellipse a b M.1 M.2 →
    ellipse a b N.1 N.2 →
    M.1 = N.1 ∧ M.2 = -N.2 →
    M ≠ N →
    ellipse a b E.1 E.2 →
    (∃ k : ℝ, k ≠ 0 ∧ 
      N.2 - 0 = k * (N.1 - 4) ∧
      E.2 - 0 = k * (E.1 - 4)) →
    -Real.sqrt 3 / 6 < k ∧ k < Real.sqrt 3 / 6) ∧
  -- 3. Fixed intersection point
  (∀ M N E : ℝ × ℝ,
    ellipse a b M.1 M.2 →
    ellipse a b N.1 N.2 →
    M.1 = N.1 ∧ M.2 = -N.2 →
    M ≠ N →
    ellipse a b E.1 E.2 →
    (∃ k : ℝ, k ≠ 0 ∧ 
      N.2 - 0 = k * (N.1 - 4) ∧
      E.2 - 0 = k * (E.1 - 4)) →
    ∃ t : ℝ, M.2 - E.2 = ((E.2 + M.2) / (E.1 - M.1)) * (t - E.1) ∧ t = 1) :=
by sorry

end

end NUMINAMATH_CALUDE_ellipse_properties_l3083_308357


namespace NUMINAMATH_CALUDE_sum_of_roots_equation_l3083_308368

theorem sum_of_roots_equation (x : ℝ) :
  (x ≠ 3 ∧ x ≠ -3) →
  ((-6 * x) / (x^2 - 9) = (3 * x) / (x + 3) - 2 / (x - 3) + 1) →
  ∃ (y : ℝ), (y ≠ 3 ∧ y ≠ -3) ∧
             ((-6 * y) / (y^2 - 9) = (3 * y) / (y + 3) - 2 / (y - 3) + 1) ∧
             x + y = 5/4 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_equation_l3083_308368
