import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l403_40383

theorem quadratic_inequality_range :
  {a : ℝ | ∃ x : ℝ, a * x^2 + 2 * x + a < 0} = {a : ℝ | a < 1} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l403_40383


namespace NUMINAMATH_CALUDE_feathers_per_crown_l403_40327

theorem feathers_per_crown (total_feathers : ℕ) (total_crowns : ℕ) 
  (h1 : total_feathers = 6538) 
  (h2 : total_crowns = 934) : 
  total_feathers / total_crowns = 7 := by
  sorry

end NUMINAMATH_CALUDE_feathers_per_crown_l403_40327


namespace NUMINAMATH_CALUDE_real_solution_implies_a_eq_one_no_purely_imaginary_roots_l403_40349

variable (a : ℝ)

/-- The complex polynomial z^2 - (a+i)z - (i+2) = 0 -/
def f (z : ℂ) : ℂ := z^2 - (a + Complex.I) * z - (Complex.I + 2)

theorem real_solution_implies_a_eq_one :
  (∃ x : ℝ, f a x = 0) → a = 1 := by sorry

theorem no_purely_imaginary_roots :
  ¬∃ y : ℝ, y ≠ 0 ∧ f a (Complex.I * y) = 0 := by sorry

end NUMINAMATH_CALUDE_real_solution_implies_a_eq_one_no_purely_imaginary_roots_l403_40349


namespace NUMINAMATH_CALUDE_unique_solution_implies_a_equals_two_l403_40304

theorem unique_solution_implies_a_equals_two (a : ℝ) :
  (∃! x : ℝ, 0 ≤ x^2 - a*x + a ∧ x^2 - a*x + a ≤ 1) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_implies_a_equals_two_l403_40304


namespace NUMINAMATH_CALUDE_parallel_planes_line_sufficiency_l403_40323

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallelism relations
variable (plane_parallel : Plane → Plane → Prop)
variable (line_parallel_plane : Line → Plane → Prop)

-- Define the subset relation for a line and a plane
variable (line_subset_plane : Line → Plane → Prop)

-- State the theorem
theorem parallel_planes_line_sufficiency 
  (α β : Plane) (m : Line) 
  (h_subset : line_subset_plane m α) 
  (h_distinct : α ≠ β) :
  (∀ α β m, plane_parallel α β → line_parallel_plane m β) ∧ 
  (∃ α β m, line_parallel_plane m β ∧ ¬plane_parallel α β) :=
by sorry

end NUMINAMATH_CALUDE_parallel_planes_line_sufficiency_l403_40323


namespace NUMINAMATH_CALUDE_average_temperature_l403_40373

theorem average_temperature (temperatures : List ℝ) (h1 : temperatures = [18, 21, 19, 22, 20]) :
  temperatures.sum / temperatures.length = 20 := by
sorry

end NUMINAMATH_CALUDE_average_temperature_l403_40373


namespace NUMINAMATH_CALUDE_hannah_spending_l403_40385

-- Define the quantities and prices
def num_sweatshirts : ℕ := 3
def price_sweatshirt : ℚ := 15
def num_tshirts : ℕ := 2
def price_tshirt : ℚ := 10
def num_socks : ℕ := 4
def price_socks : ℚ := 5
def price_jacket : ℚ := 50
def discount_rate : ℚ := 0.1

-- Define the total cost before discount
def total_cost_before_discount : ℚ :=
  num_sweatshirts * price_sweatshirt +
  num_tshirts * price_tshirt +
  num_socks * price_socks +
  price_jacket

-- Define the discount amount
def discount_amount : ℚ := discount_rate * total_cost_before_discount

-- Define the final cost after discount
def final_cost : ℚ := total_cost_before_discount - discount_amount

-- Theorem statement
theorem hannah_spending :
  final_cost = 121.5 := by sorry

end NUMINAMATH_CALUDE_hannah_spending_l403_40385


namespace NUMINAMATH_CALUDE_last_ten_digits_periodicity_l403_40318

theorem last_ten_digits_periodicity (n : ℕ) (h : n ≥ 10) :
  2^n % 10^10 = 2^(n + 4 * 10^9) % 10^10 := by
  sorry

end NUMINAMATH_CALUDE_last_ten_digits_periodicity_l403_40318


namespace NUMINAMATH_CALUDE_cubic_root_sum_l403_40326

/-- Given that p, q, and r are the roots of x³ - 3x - 2 = 0,
    prove that p(q - r)² + q(r - p)² + r(p - q)² = -18 -/
theorem cubic_root_sum (p q r : ℝ) : 
  (p^3 = 3*p + 2) → 
  (q^3 = 3*q + 2) → 
  (r^3 = 3*r + 2) → 
  p*(q - r)^2 + q*(r - p)^2 + r*(p - q)^2 = -18 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l403_40326


namespace NUMINAMATH_CALUDE_indigo_restaurant_rating_l403_40329

/-- The average star rating for a restaurant given the number of reviews for each star rating. -/
def averageStarRating (fiveStars fourStars threeStars twoStars : ℕ) : ℚ :=
  let totalStars := 5 * fiveStars + 4 * fourStars + 3 * threeStars + 2 * twoStars
  let totalReviews := fiveStars + fourStars + threeStars + twoStars
  (totalStars : ℚ) / totalReviews

/-- Theorem stating that the average star rating for Indigo Restaurant is 4 stars. -/
theorem indigo_restaurant_rating :
  averageStarRating 6 7 4 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_indigo_restaurant_rating_l403_40329


namespace NUMINAMATH_CALUDE_smallest_a_for_sum_of_squares_l403_40362

theorem smallest_a_for_sum_of_squares (a : ℝ) : 
  (∀ x : ℝ, x^2 - 3*a*x + a^2 = 0 → 
   ∃ x1 x2 : ℝ, x1^2 + x2^2 = 0.28 ∧ x1 ≠ x2 ∧ 
   (∀ y : ℝ, y^2 - 3*a*y + a^2 = 0 → y = x1 ∨ y = x2)) →
  a = -0.2 ∧ 
  (∀ b : ℝ, b < -0.2 → 
   ¬(∀ x : ℝ, x^2 - 3*b*x + b^2 = 0 → 
     ∃ x1 x2 : ℝ, x1^2 + x2^2 = 0.28 ∧ x1 ≠ x2 ∧ 
     (∀ y : ℝ, y^2 - 3*b*y + b^2 = 0 → y = x1 ∨ y = x2))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_a_for_sum_of_squares_l403_40362


namespace NUMINAMATH_CALUDE_cotangent_half_angle_identity_l403_40396

theorem cotangent_half_angle_identity (α : Real) (m : Real) :
  (Real.tan (α / 2))⁻¹ = m → (1 - Real.sin α) / Real.cos α = (m - 1) / (m + 1) := by
  sorry

end NUMINAMATH_CALUDE_cotangent_half_angle_identity_l403_40396


namespace NUMINAMATH_CALUDE_simplify_expression_l403_40374

theorem simplify_expression :
  -2^2005 + (-2)^2006 + 3^2007 - 2^2008 = -7 * 2^2005 + 3^2007 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l403_40374


namespace NUMINAMATH_CALUDE_complex_division_problem_l403_40346

theorem complex_division_problem (a : ℝ) (h : (a^2 - 9 : ℂ) + (a + 3 : ℂ) * I = (0 : ℂ) + b * I) :
  (a + I^19) / (1 + I) = 1 - 2*I :=
sorry

end NUMINAMATH_CALUDE_complex_division_problem_l403_40346


namespace NUMINAMATH_CALUDE_probability_three_white_two_black_l403_40306

/-- The probability of drawing exactly 3 white and 2 black balls from a box
    containing 8 white and 7 black balls, when 5 balls are drawn at random. -/
theorem probability_three_white_two_black : 
  let total_balls : ℕ := 8 + 7
  let white_balls : ℕ := 8
  let black_balls : ℕ := 7
  let drawn_balls : ℕ := 5
  let white_drawn : ℕ := 3
  let black_drawn : ℕ := 2
  let favorable_outcomes : ℕ := (Nat.choose white_balls white_drawn) * (Nat.choose black_balls black_drawn)
  let total_outcomes : ℕ := Nat.choose total_balls drawn_balls
  (favorable_outcomes : ℚ) / total_outcomes = 8 / 17 := by
sorry

end NUMINAMATH_CALUDE_probability_three_white_two_black_l403_40306


namespace NUMINAMATH_CALUDE_solve_equations_l403_40368

theorem solve_equations :
  (∃ x1 x2 : ℝ, 2 * x1 * (x1 - 1) = 1 ∧ 2 * x2 * (x2 - 1) = 1 ∧
    x1 = (1 + Real.sqrt 3) / 2 ∧ x2 = (1 - Real.sqrt 3) / 2) ∧
  (∃ y1 y2 : ℝ, y1^2 + 8*y1 + 7 = 0 ∧ y2^2 + 8*y2 + 7 = 0 ∧
    y1 = -7 ∧ y2 = -1) :=
by sorry

end NUMINAMATH_CALUDE_solve_equations_l403_40368


namespace NUMINAMATH_CALUDE_alternating_series_sum_l403_40322

def alternating_series (n : ℕ) : ℤ := 
  if n % 2 = 0 then (n + 1) else -(n + 1)

def series_sum (n : ℕ) : ℤ := 
  (Finset.range n).sum (λ i => alternating_series i)

theorem alternating_series_sum : series_sum 10001 = -5001 := by
  sorry

end NUMINAMATH_CALUDE_alternating_series_sum_l403_40322


namespace NUMINAMATH_CALUDE_fourth_number_proof_l403_40365

theorem fourth_number_proof (x : ℝ) : 
  3 + 33 + 333 + x = 369.63 → x = 0.63 := by sorry

end NUMINAMATH_CALUDE_fourth_number_proof_l403_40365


namespace NUMINAMATH_CALUDE_min_moves_for_equal_stones_l403_40316

/-- Operation that can be performed on boxes -/
structure Operation where
  stones_per_box : Nat
  num_boxes : Nat
  extra_stones : Nat

/-- Problem setup -/
def total_boxes : Nat := 2019
def operation : Operation := { stones_per_box := 100, num_boxes := 100, extra_stones := 0 }

/-- Function to calculate the minimum number of moves -/
def min_moves (n : Nat) (op : Operation) : Nat :=
  let d := Nat.gcd n op.num_boxes
  Nat.ceil ((n^2 : Rat) / (d * op.num_boxes))

/-- Theorem statement -/
theorem min_moves_for_equal_stones :
  min_moves total_boxes operation = 40762 := by
  sorry

end NUMINAMATH_CALUDE_min_moves_for_equal_stones_l403_40316


namespace NUMINAMATH_CALUDE_conference_attendees_l403_40363

theorem conference_attendees :
  ∃ n : ℕ,
    n < 50 ∧
    n % 8 = 5 ∧
    n % 6 = 3 ∧
    n = 45 := by
  sorry

end NUMINAMATH_CALUDE_conference_attendees_l403_40363


namespace NUMINAMATH_CALUDE_system_solutions_l403_40390

def has_solution (a : ℝ) (x y : ℝ) : Prop :=
  x > 0 ∧ y ≥ 0 ∧ 2*y - 2 = a*(x - 2) ∧ 4*y / (|x| + x) = Real.sqrt y

theorem system_solutions :
  ∀ a : ℝ,
    (a < 0 ∨ a > 1 → 
      has_solution a (2 - 2/a) 0 ∧ has_solution a 2 1) ∧
    (0 ≤ a ∧ a ≤ 1 → 
      has_solution a 2 1) ∧
    ((1 < a ∧ a < 2) ∨ a > 2 → 
      has_solution a (2 - 2/a) 0 ∧ 
      has_solution a 2 1 ∧ 
      has_solution a (2*a - 2) ((a-1)^2)) :=
sorry

end NUMINAMATH_CALUDE_system_solutions_l403_40390


namespace NUMINAMATH_CALUDE_diana_grace_age_ratio_l403_40300

/-- The ratio of Diana's age to Grace's age -/
def age_ratio (diana_age : ℕ) (grace_age : ℕ) : ℚ :=
  diana_age / grace_age

/-- Grace's current age -/
def grace_current_age (grace_last_year : ℕ) : ℕ :=
  grace_last_year + 1

theorem diana_grace_age_ratio :
  let diana_age : ℕ := 8
  let grace_last_year : ℕ := 3
  age_ratio diana_age (grace_current_age grace_last_year) = 2 := by
sorry

end NUMINAMATH_CALUDE_diana_grace_age_ratio_l403_40300


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l403_40397

/-- Given vectors a and b in ℝ², if a is perpendicular to b, then the y-coordinate of b is -1 -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (h : a.1 * b.1 + a.2 * b.2 = 0) :
  a = (1, 3) → b.1 = 3 → b.2 = -1 := by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l403_40397


namespace NUMINAMATH_CALUDE_quadratic_square_plus_constant_l403_40338

theorem quadratic_square_plus_constant :
  ∃ k : ℤ, ∀ z : ℂ, z^2 - 6*z + 17 = (z - 3)^2 + k := by
  sorry

end NUMINAMATH_CALUDE_quadratic_square_plus_constant_l403_40338


namespace NUMINAMATH_CALUDE_surface_area_of_cuboid_from_cubes_l403_40335

/-- The surface area of a cuboid formed by three cubes in a row -/
theorem surface_area_of_cuboid_from_cubes (cube_side_length : ℝ) (h : cube_side_length = 8) : 
  let cuboid_length : ℝ := 3 * cube_side_length
  let cuboid_width : ℝ := cube_side_length
  let cuboid_height : ℝ := cube_side_length
  2 * (cuboid_length * cuboid_width + cuboid_length * cuboid_height + cuboid_width * cuboid_height) = 896 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_of_cuboid_from_cubes_l403_40335


namespace NUMINAMATH_CALUDE_intersection_k_value_l403_40392

/-- Given two lines that intersect at a point, find the value of k -/
theorem intersection_k_value (m n : ℝ → ℝ) (k : ℝ) :
  (∀ x, m x = 4 * x + 2) →  -- Line m equation
  (∀ x, n x = k * x - 8) →  -- Line n equation
  m (-2) = -6 →             -- Lines intersect at (-2, -6)
  n (-2) = -6 →             -- Lines intersect at (-2, -6)
  k = -1 := by
sorry

end NUMINAMATH_CALUDE_intersection_k_value_l403_40392


namespace NUMINAMATH_CALUDE_plane_division_l403_40394

/-- The maximum number of regions that can be created by n lines in a plane -/
def max_regions (n : ℕ) : ℕ := sorry

/-- The number of additional regions created by adding a line that intersects all existing lines -/
def additional_regions (n : ℕ) : ℕ := sorry

theorem plane_division (total_lines : ℕ) (parallel_lines : ℕ) 
  (h1 : total_lines = 10) 
  (h2 : parallel_lines = 4) 
  (h3 : parallel_lines ≤ total_lines) :
  max_regions (total_lines - parallel_lines) + 
  (parallel_lines * additional_regions (total_lines - parallel_lines)) = 50 := by
  sorry

end NUMINAMATH_CALUDE_plane_division_l403_40394


namespace NUMINAMATH_CALUDE_triangle_side_length_l403_40341

/-- Given a triangle ABC where:
    - a, b, c are sides opposite to angles A, B, C respectively
    - A = 60°
    - b = 4
    - Area of triangle ABC = 4√3
    Prove that a = 4 -/
theorem triangle_side_length (a b c : ℝ) (A : Real) (S : ℝ) : 
  A = π / 3 → 
  b = 4 → 
  S = 4 * Real.sqrt 3 → 
  S = (1 / 2) * b * c * Real.sin A → 
  a = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l403_40341


namespace NUMINAMATH_CALUDE_intersection_nonempty_iff_a_in_range_l403_40308

/-- Given sets A and B, prove that their intersection is non-empty if and only if -1 < a < 3 -/
theorem intersection_nonempty_iff_a_in_range (a : ℝ) :
  let A := {x : ℝ | x - 1 > a^2}
  let B := {x : ℝ | x - 4 < 2*a}
  (∃ x, x ∈ A ∩ B) ↔ -1 < a ∧ a < 3 :=
by sorry

end NUMINAMATH_CALUDE_intersection_nonempty_iff_a_in_range_l403_40308


namespace NUMINAMATH_CALUDE_tangent_line_intercept_l403_40352

/-- Given a curve y = x³ + ax + 1 and a line y = kx + b tangent to the curve at (2, 3), prove b = -15 -/
theorem tangent_line_intercept (a k b : ℝ) : 
  (3 = 2^3 + a*2 + 1) →  -- The curve passes through (2, 3)
  (k = 3*2^2 + a) →      -- The slope of the tangent line equals the derivative at x = 2
  (3 = k*2 + b) →        -- The line passes through (2, 3)
  b = -15 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_intercept_l403_40352


namespace NUMINAMATH_CALUDE_normal_dist_probability_l403_40369

-- Define the normal distribution
def normal_dist (μ σ : ℝ) (X : ℝ → ℝ) : Prop :=
  ∃ (f : ℝ → ℝ), ∀ x, X x = f ((x - μ) / σ)

-- Define the probability function
noncomputable def P (a b : ℝ) (X : ℝ → ℝ) : ℝ := sorry

-- State the theorem
theorem normal_dist_probability 
  (X : ℝ → ℝ) (μ σ : ℝ) 
  (h1 : normal_dist μ σ X)
  (h2 : P (μ - 2*σ) (μ + 2*σ) X = 0.9544)
  (h3 : P (μ - σ) (μ + σ) X = 0.682)
  (h4 : μ = 4)
  (h5 : σ = 1) :
  P 5 6 X = 0.1359 := by sorry

end NUMINAMATH_CALUDE_normal_dist_probability_l403_40369


namespace NUMINAMATH_CALUDE_inequality_solution_l403_40347

theorem inequality_solution (x : ℝ) : (x - 2) / (x - 4) ≥ 3 ↔ x < 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l403_40347


namespace NUMINAMATH_CALUDE_A_intersect_B_l403_40377

def A : Set ℤ := {1, 2, 3, 4}

def B : Set ℤ := {y | ∃ x ∈ A, y = 3 * x - 2}

theorem A_intersect_B : A ∩ B = {1, 4} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l403_40377


namespace NUMINAMATH_CALUDE_bowling_team_weight_problem_l403_40384

theorem bowling_team_weight_problem (initial_players : ℕ) (initial_avg_weight : ℝ)
  (new_player2_weight : ℝ) (new_avg_weight : ℝ) :
  initial_players = 7 →
  initial_avg_weight = 121 →
  new_player2_weight = 60 →
  new_avg_weight = 113 →
  ∃ new_player1_weight : ℝ,
    new_player1_weight = 110 ∧
    (initial_players : ℝ) * initial_avg_weight + new_player1_weight + new_player2_weight =
      ((initial_players : ℝ) + 2) * new_avg_weight :=
by sorry

end NUMINAMATH_CALUDE_bowling_team_weight_problem_l403_40384


namespace NUMINAMATH_CALUDE_elizabeth_study_time_l403_40337

/-- The total study time for Elizabeth given her time spent on science and math tests -/
def total_study_time (science_time math_time : ℕ) : ℕ :=
  science_time + math_time

/-- Theorem stating that Elizabeth's total study time is 60 minutes -/
theorem elizabeth_study_time :
  total_study_time 25 35 = 60 := by
  sorry

end NUMINAMATH_CALUDE_elizabeth_study_time_l403_40337


namespace NUMINAMATH_CALUDE_four_team_hierarchy_exists_l403_40319

/-- Represents a volleyball team -/
structure Team :=
  (id : Nat)

/-- Represents the result of a match between two teams -/
inductive MatchResult
  | Win
  | Loss

/-- Represents a tournament with n teams -/
structure Tournament (n : Nat) :=
  (teams : Fin n → Team)
  (results : Fin n → Fin n → MatchResult)
  (results_valid : ∀ i j, i ≠ j → results i j ≠ results j i)

/-- Theorem stating the existence of four teams with the specified winning relationships -/
theorem four_team_hierarchy_exists (t : Tournament 8) :
  ∃ (a b c d : Fin 8),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    t.results a b = MatchResult.Win ∧
    t.results a c = MatchResult.Win ∧
    t.results a d = MatchResult.Win ∧
    t.results b c = MatchResult.Win ∧
    t.results b d = MatchResult.Win ∧
    t.results c d = MatchResult.Win :=
  sorry

end NUMINAMATH_CALUDE_four_team_hierarchy_exists_l403_40319


namespace NUMINAMATH_CALUDE_sufficient_condition_B_proper_subset_A_l403_40354

def A : Set ℝ := {x | x^2 + x - 6 = 0}
def B (m : ℝ) : Set ℝ := {x | x * m + 1 = 0}

theorem sufficient_condition_B_proper_subset_A :
  ∃ S : Set ℝ, (S = {0, 1/3}) ∧ 
  (∀ m : ℝ, m ∈ S → B m ⊂ A) ∧
  (∃ m : ℝ, m ∉ S ∧ B m ⊂ A) :=
sorry

end NUMINAMATH_CALUDE_sufficient_condition_B_proper_subset_A_l403_40354


namespace NUMINAMATH_CALUDE_board_and_sum_properties_l403_40331

/-- The number of squares in a square board -/
def boardSquares (n : ℕ) : ℕ := n * n

/-- The number of squares in each region separated by the diagonal -/
def regionSquares (n : ℕ) : ℕ := (n * n - n) / 2

/-- The sum of consecutive integers from 1 to n -/
def sumIntegers (n : ℕ) : ℕ := n * (n + 1) / 2

theorem board_and_sum_properties :
  (boardSquares 11 = 121) ∧
  (regionSquares 11 = 55) ∧
  (sumIntegers 10 = 55) ∧
  (sumIntegers 100 = 5050) :=
sorry

end NUMINAMATH_CALUDE_board_and_sum_properties_l403_40331


namespace NUMINAMATH_CALUDE_plane_contains_points_plane_uniqueness_l403_40309

/-- A plane in 3D space defined by its equation coefficients -/
structure Plane where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ
  A_pos : A > 0
  gcd_one : Nat.gcd (Nat.gcd (Int.natAbs A) (Int.natAbs B)) (Nat.gcd (Int.natAbs C) (Int.natAbs D)) = 1

/-- A point in 3D space with integer coordinates -/
structure Point3D where
  x : ℤ
  y : ℤ
  z : ℤ

/-- Check if a point lies on a plane -/
def Point3D.liesOn (p : Point3D) (plane : Plane) : Prop :=
  plane.A * p.x + plane.B * p.y + plane.C * p.z + plane.D = 0

/-- The specific plane we're proving about -/
def targetPlane : Plane := {
  A := 1
  B := 3
  C := -2
  D := -11
  A_pos := by simp
  gcd_one := by sorry
}

/-- The three points given in the problem -/
def p1 : Point3D := ⟨0, 3, -1⟩
def p2 : Point3D := ⟨2, 3, 1⟩
def p3 : Point3D := ⟨4, 1, 0⟩

/-- The main theorem stating that the target plane contains the three given points -/
theorem plane_contains_points : 
  p1.liesOn targetPlane ∧ p2.liesOn targetPlane ∧ p3.liesOn targetPlane :=
by sorry

/-- The theorem stating that the target plane is unique -/
theorem plane_uniqueness (plane : Plane) :
  p1.liesOn plane ∧ p2.liesOn plane ∧ p3.liesOn plane → plane = targetPlane :=
by sorry

end NUMINAMATH_CALUDE_plane_contains_points_plane_uniqueness_l403_40309


namespace NUMINAMATH_CALUDE_root_implies_m_value_l403_40307

theorem root_implies_m_value (m : ℝ) : 
  (2^2 - m*2 + 2 = 0) → m = 3 := by
sorry

end NUMINAMATH_CALUDE_root_implies_m_value_l403_40307


namespace NUMINAMATH_CALUDE_vendor_apples_thrown_away_l403_40303

/-- Represents the percentage of apples remaining after each operation -/
def apples_remaining (initial_percentage : ℚ) (sell_percentage : ℚ) : ℚ :=
  initial_percentage * (1 - sell_percentage)

/-- Represents the percentage of apples thrown away -/
def apples_thrown (initial_percentage : ℚ) (throw_percentage : ℚ) : ℚ :=
  initial_percentage * throw_percentage

theorem vendor_apples_thrown_away :
  let initial_stock := 1
  let first_day_remaining := apples_remaining initial_stock (60 / 100)
  let first_day_thrown := apples_thrown first_day_remaining (40 / 100)
  let second_day_remaining := apples_remaining (first_day_remaining - first_day_thrown) (50 / 100)
  let second_day_thrown := second_day_remaining
  first_day_thrown + second_day_thrown = 28 / 100 := by
  sorry

end NUMINAMATH_CALUDE_vendor_apples_thrown_away_l403_40303


namespace NUMINAMATH_CALUDE_solution_set_inequality_l403_40348

/-- Given that the solution set of ax^2 + bx + c < 0 is (-∞, -1) ∪ (1/2, +∞),
    prove that the solution set of cx^2 - bx + a < 0 is (-2, 1) -/
theorem solution_set_inequality (a b c : ℝ) :
  (∀ x : ℝ, ax^2 + b*x + c < 0 ↔ x < -1 ∨ x > 1/2) →
  (∀ x : ℝ, c*x^2 - b*x + a < 0 ↔ -2 < x ∧ x < 1) :=
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l403_40348


namespace NUMINAMATH_CALUDE_exponent_operations_l403_40355

theorem exponent_operations (a : ℝ) : 
  (a^4 * a^3 = a^7) ∧ 
  ((a^2)^3 ≠ a^5) ∧ 
  (3*a^2 - a^2 ≠ 2) ∧ 
  ((a - b)^2 ≠ a^2 - b^2) :=
sorry

end NUMINAMATH_CALUDE_exponent_operations_l403_40355


namespace NUMINAMATH_CALUDE_three_hundredth_term_of_specific_sequence_l403_40364

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r ^ (n - 1)

theorem three_hundredth_term_of_specific_sequence :
  let a₁ := 8
  let a₂ := -8
  let r := a₂ / a₁
  geometric_sequence a₁ r 300 = -8 := by
sorry

end NUMINAMATH_CALUDE_three_hundredth_term_of_specific_sequence_l403_40364


namespace NUMINAMATH_CALUDE_apple_heavier_than_kiwi_l403_40386

-- Define a type for fruits
inductive Fruit
  | Apple
  | Banana
  | Kiwi

-- Define a weight relation between fruits
def heavier_than (a b : Fruit) : Prop := sorry

-- State the theorem
theorem apple_heavier_than_kiwi 
  (h1 : heavier_than Fruit.Apple Fruit.Banana) 
  (h2 : heavier_than Fruit.Banana Fruit.Kiwi) : 
  heavier_than Fruit.Apple Fruit.Kiwi := by
  sorry

end NUMINAMATH_CALUDE_apple_heavier_than_kiwi_l403_40386


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l403_40314

/-- Represents an isosceles triangle with base 4 and leg length x -/
structure IsoscelesTriangle where
  x : ℝ
  is_root : x^2 - 5*x + 6 = 0
  is_valid : x + x > 4

/-- The perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ := 2 * t.x + 4

theorem isosceles_triangle_perimeter :
  ∀ t : IsoscelesTriangle, perimeter t = 10 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l403_40314


namespace NUMINAMATH_CALUDE_circle_and_tangent_lines_l403_40340

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define points A, B, and M
def A : ℝ × ℝ := (0, -6)
def B : ℝ × ℝ := (1, -5)
def M : ℝ × ℝ := (2, 8)

-- Define the line l: x - y + 1 = 0
def l : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 - p.2 + 1 = 0}

-- Theorem statement
theorem circle_and_tangent_lines 
  (C : ℝ × ℝ) -- Center of the circle
  (h1 : C ∈ l) -- Center lies on line l
  (h2 : A ∈ Circle C (|C.1 - A.1|)) -- A is on the circle
  (h3 : B ∈ Circle C (|C.1 - A.1|)) -- B is on the circle
  : 
  -- 1. Standard equation of the circle
  (∀ p : ℝ × ℝ, p ∈ Circle C (|C.1 - A.1|) ↔ (p.1 + 3)^2 + (p.2 + 2)^2 = 25) ∧
  -- 2. Equations of tangent lines
  (∀ p : ℝ × ℝ, (p.1 = 2 ∨ 3*p.1 - 4*p.2 + 26 = 0) ↔ 
    (p ∈ {q : ℝ × ℝ | (q.1 - M.1) * (C.1 - q.1) + (q.2 - M.2) * (C.2 - q.2) = 0} ∧
     p ∈ Circle C (|C.1 - A.1|))) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_tangent_lines_l403_40340


namespace NUMINAMATH_CALUDE_remaining_balance_l403_40370

def house_price : ℕ := 100000
def down_payment_percentage : ℚ := 20 / 100
def parents_payment_percentage : ℚ := 30 / 100

theorem remaining_balance (hp : ℕ) (dp : ℚ) (pp : ℚ) : 
  hp - hp * dp - (hp - hp * dp) * pp = 56000 :=
by sorry

end NUMINAMATH_CALUDE_remaining_balance_l403_40370


namespace NUMINAMATH_CALUDE_sample_size_comparison_l403_40360

/-- Given two samples with different means, prove that the number of elements in the first sample
    is less than or equal to the number of elements in the second sample, based on the combined mean. -/
theorem sample_size_comparison (m n : ℕ) (x_bar y_bar z_bar : ℝ) (a : ℝ) :
  x_bar ≠ y_bar →
  z_bar = a * x_bar + (1 - a) * y_bar →
  0 < a →
  a ≤ 1/2 →
  z_bar = (m * x_bar + n * y_bar) / (m + n : ℝ) →
  m ≤ n := by
  sorry


end NUMINAMATH_CALUDE_sample_size_comparison_l403_40360


namespace NUMINAMATH_CALUDE_ladybug_count_l403_40334

theorem ladybug_count (num_leaves : ℕ) (ladybugs_per_leaf : ℕ) 
  (h1 : num_leaves = 84) 
  (h2 : ladybugs_per_leaf = 139) : 
  num_leaves * ladybugs_per_leaf = 11676 := by
  sorry

end NUMINAMATH_CALUDE_ladybug_count_l403_40334


namespace NUMINAMATH_CALUDE_proportion_equation_proof_l403_40372

theorem proportion_equation_proof (x y : ℝ) (h1 : y ≠ 0) (h2 : 3 * x = 5 * y) :
  x / 5 = y / 3 := by sorry

end NUMINAMATH_CALUDE_proportion_equation_proof_l403_40372


namespace NUMINAMATH_CALUDE_min_value_geometric_sequence_l403_40389

theorem min_value_geometric_sequence (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) 
    (h_geometric : ∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = a n * q)
    (h_condition : a 2 * a 3 * a 4 = a 2 + a 3 + a 4) : 
  a 3 ≥ Real.sqrt 3 ∧ ∃ a' : ℕ → ℝ, (∀ n, a' n > 0) ∧ 
    (∃ q' : ℝ, q' > 0 ∧ ∀ n, a' (n + 1) = a' n * q') ∧
    (a' 2 * a' 3 * a' 4 = a' 2 + a' 3 + a' 4) ∧
    (a' 3 = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_geometric_sequence_l403_40389


namespace NUMINAMATH_CALUDE_f_not_monotonic_iff_m_in_open_zero_one_l403_40371

open Real

-- Define the function f(x) = |log₂(x)|
noncomputable def f (x : ℝ) : ℝ := abs (log x / log 2)

-- Define the property of not being monotonic in an interval
def not_monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x y z, a < x ∧ x < y ∧ y < z ∧ z < b ∧
  ((f x < f y ∧ f y > f z) ∨ (f x > f y ∧ f y < f z))

-- State the theorem
theorem f_not_monotonic_iff_m_in_open_zero_one (m : ℝ) :
  m > 0 → (not_monotonic f m (2*m + 1) ↔ 0 < m ∧ m < 1) :=
sorry

end NUMINAMATH_CALUDE_f_not_monotonic_iff_m_in_open_zero_one_l403_40371


namespace NUMINAMATH_CALUDE_factor_problem_l403_40388

theorem factor_problem (n : ℤ) (f : ℚ) (h1 : n = 9) (h2 : (n + 2) * f = 24 + n) : f = 3 := by
  sorry

end NUMINAMATH_CALUDE_factor_problem_l403_40388


namespace NUMINAMATH_CALUDE_no_quadratic_term_implies_k_equals_three_l403_40301

/-- 
Given an algebraic expression in x and y: (-3kxy+3y)+(9xy-8x+1),
prove that if there is no quadratic term, then k = 3.
-/
theorem no_quadratic_term_implies_k_equals_three (k : ℚ) : 
  (∀ x y : ℚ, (-3*k*x*y + 3*y) + (9*x*y - 8*x + 1) = (-3*k + 9)*x*y + 3*y - 8*x + 1) →
  (∀ x y : ℚ, (-3*k + 9)*x*y + 3*y - 8*x + 1 = 3*y - 8*x + 1) →
  k = 3 := by
sorry

end NUMINAMATH_CALUDE_no_quadratic_term_implies_k_equals_three_l403_40301


namespace NUMINAMATH_CALUDE_count_primes_with_squares_between_5000_and_9000_l403_40345

theorem count_primes_with_squares_between_5000_and_9000 :
  ∃ (S : Finset Nat),
    (∀ p ∈ S, Nat.Prime p ∧ 5000 ≤ p^2 ∧ p^2 ≤ 9000) ∧
    (∀ p : Nat, Nat.Prime p → 5000 ≤ p^2 → p^2 ≤ 9000 → p ∈ S) ∧
    Finset.card S = 6 := by
  sorry

end NUMINAMATH_CALUDE_count_primes_with_squares_between_5000_and_9000_l403_40345


namespace NUMINAMATH_CALUDE_expand_product_l403_40359

theorem expand_product (x : ℝ) : (x + 4) * (x - 9) = x^2 - 5*x - 36 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l403_40359


namespace NUMINAMATH_CALUDE_swimmer_speed_l403_40376

/-- The speed of a swimmer in still water, given his downstream and upstream speeds and distances. -/
theorem swimmer_speed (downstream_distance : ℝ) (downstream_time : ℝ) 
  (upstream_distance : ℝ) (upstream_time : ℝ) :
  downstream_distance = 62 →
  downstream_time = 10 →
  upstream_distance = 84 →
  upstream_time = 14 →
  ∃ (v_m v_s : ℝ),
    v_m + v_s = downstream_distance / downstream_time ∧
    v_m - v_s = upstream_distance / upstream_time ∧
    v_m = 6.1 :=
by sorry

end NUMINAMATH_CALUDE_swimmer_speed_l403_40376


namespace NUMINAMATH_CALUDE_expected_ones_is_half_l403_40381

/-- The probability of rolling a 1 on a standard die -/
def prob_one : ℚ := 1/6

/-- The probability of not rolling a 1 on a standard die -/
def prob_not_one : ℚ := 1 - prob_one

/-- The number of dice rolled -/
def num_dice : ℕ := 3

/-- The expected number of 1's when rolling three standard dice -/
def expected_ones : ℚ := 
  0 * (prob_not_one ^ num_dice) +
  1 * (num_dice.choose 1 * prob_one * (prob_not_one ^ 2)) +
  2 * (num_dice.choose 2 * (prob_one ^ 2) * prob_not_one) +
  3 * (prob_one ^ num_dice)

theorem expected_ones_is_half : expected_ones = 1/2 := by sorry

end NUMINAMATH_CALUDE_expected_ones_is_half_l403_40381


namespace NUMINAMATH_CALUDE_bottle_caps_count_l403_40339

theorem bottle_caps_count (initial_caps : ℕ) (added_caps : ℕ) : 
  initial_caps = 7 → added_caps = 7 → initial_caps + added_caps = 14 := by
  sorry

end NUMINAMATH_CALUDE_bottle_caps_count_l403_40339


namespace NUMINAMATH_CALUDE_purely_imaginary_condition_l403_40315

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem purely_imaginary_condition (a : ℝ) :
  is_purely_imaginary ((a^2 - 1 : ℝ) + (2 * (a + 1) : ℝ) * I) ↔ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_condition_l403_40315


namespace NUMINAMATH_CALUDE_expression_equivalence_l403_40395

theorem expression_equivalence :
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * 
  (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) = 5^128 - 4^128 := by
  sorry

end NUMINAMATH_CALUDE_expression_equivalence_l403_40395


namespace NUMINAMATH_CALUDE_smallest_n_for_cube_sum_inequality_l403_40324

theorem smallest_n_for_cube_sum_inequality : 
  ∃ n : ℕ, (∀ x y z : ℝ, (x^3 + y^3 + z^3)^2 ≤ n * (x^6 + y^6 + z^6)) ∧ 
  (∀ m : ℕ, m < n → ∃ x y z : ℝ, (x^3 + y^3 + z^3)^2 > m * (x^6 + y^6 + z^6)) ∧
  n = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_cube_sum_inequality_l403_40324


namespace NUMINAMATH_CALUDE_tree_heights_theorem_l403_40342

/-- Represents the heights of 5 trees -/
structure TreeHeights where
  h1 : ℤ
  h2 : ℤ
  h3 : ℤ
  h4 : ℤ
  h5 : ℤ

/-- The condition that each tree is either twice as tall or half as tall as the one to its right -/
def validHeights (h : TreeHeights) : Prop :=
  (h.h1 = 2 * h.h2 ∨ 2 * h.h1 = h.h2) ∧
  (h.h2 = 2 * h.h3 ∨ 2 * h.h2 = h.h3) ∧
  (h.h3 = 2 * h.h4 ∨ 2 * h.h3 = h.h4) ∧
  (h.h4 = 2 * h.h5 ∨ 2 * h.h4 = h.h5)

/-- The average height of the trees -/
def averageHeight (h : TreeHeights) : ℚ :=
  (h.h1 + h.h2 + h.h3 + h.h4 + h.h5) / 5

/-- The main theorem -/
theorem tree_heights_theorem (h : TreeHeights) 
  (h_valid : validHeights h) 
  (h_second : h.h2 = 11) : 
  averageHeight h = 121 / 5 := by
  sorry

end NUMINAMATH_CALUDE_tree_heights_theorem_l403_40342


namespace NUMINAMATH_CALUDE_min_sum_of_six_l403_40311

def consecutive_numbers (start : ℕ) : List ℕ :=
  List.range 11 |>.map (λ i => start + i)

theorem min_sum_of_six (start : ℕ) :
  (consecutive_numbers start).length = 11 →
  start + (start + 10) = 90 →
  ∃ (subset : List ℕ), subset.length = 6 ∧ 
    subset.all (λ x => x ∈ consecutive_numbers start) ∧
    subset.sum = 90 ∧
    ∀ (other_subset : List ℕ), other_subset.length = 6 →
      other_subset.all (λ x => x ∈ consecutive_numbers start) →
      other_subset.sum ≥ 90 :=
by
  sorry

#check min_sum_of_six

end NUMINAMATH_CALUDE_min_sum_of_six_l403_40311


namespace NUMINAMATH_CALUDE_distinct_elements_in_union_of_progressions_l403_40399

def arithmetic_progression (a₀ : ℕ) (d : ℕ) (n : ℕ) : Finset ℕ :=
  Finset.image (λ k => a₀ + k * d) (Finset.range n)

theorem distinct_elements_in_union_of_progressions :
  let progression1 := arithmetic_progression 2 3 2023
  let progression2 := arithmetic_progression 10 7 2023
  (progression1 ∪ progression2).card = 3756 := by
  sorry

end NUMINAMATH_CALUDE_distinct_elements_in_union_of_progressions_l403_40399


namespace NUMINAMATH_CALUDE_smallest_square_divisible_by_2016_l403_40320

theorem smallest_square_divisible_by_2016 :
  ∀ n : ℕ, n > 0 → n^2 % 2016 = 0 → n ≥ 168 :=
by sorry

end NUMINAMATH_CALUDE_smallest_square_divisible_by_2016_l403_40320


namespace NUMINAMATH_CALUDE_two_digit_repeating_decimal_l403_40325

theorem two_digit_repeating_decimal (ab : ℕ) (h1 : ab ≥ 10 ∧ ab < 100) :
  66 * (1 + ab / 100 : ℚ) + 1/2 = 66 * (1 + ab / 99 : ℚ) → ab = 75 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_repeating_decimal_l403_40325


namespace NUMINAMATH_CALUDE_right_triangle_area_l403_40310

theorem right_triangle_area (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : c = 5) (h3 : a = 4) :
  (1/2) * a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l403_40310


namespace NUMINAMATH_CALUDE_diagonals_concurrent_l403_40358

-- Define a regular 12-gon
def Regular12gon (P : Fin 12 → ℝ × ℝ) : Prop :=
  ∀ i j : Fin 12, dist (P i) (P ((i + 1) % 12)) = dist (P j) (P ((j + 1) % 12))

-- Define a diagonal in the 12-gon
def Diagonal (P : Fin 12 → ℝ × ℝ) (i j : Fin 12) : Set (ℝ × ℝ) :=
  {x | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ x = (1 - t) • (P i) + t • (P j)}

-- Define concurrency of three lines
def Concurrent (L₁ L₂ L₃ : Set (ℝ × ℝ)) : Prop :=
  ∃ x : ℝ × ℝ, x ∈ L₁ ∧ x ∈ L₂ ∧ x ∈ L₃

-- Theorem statement
theorem diagonals_concurrent (P : Fin 12 → ℝ × ℝ) (h : Regular12gon P) :
  Concurrent (Diagonal P 0 8) (Diagonal P 11 3) (Diagonal P 1 10) :=
sorry

end NUMINAMATH_CALUDE_diagonals_concurrent_l403_40358


namespace NUMINAMATH_CALUDE_sum_perfect_square_l403_40328

theorem sum_perfect_square (K M : ℕ) : 
  K > 0 → M < 100 → K * (K + 1) = M^2 → (K = 8 ∨ K = 35) := by
  sorry

end NUMINAMATH_CALUDE_sum_perfect_square_l403_40328


namespace NUMINAMATH_CALUDE_cyclist_round_time_l403_40356

/-- Given a rectangular park with length L and breadth B, prove that a cyclist
    traveling at 12 km/hr along the park's boundary will complete one round in 8 minutes
    when the length to breadth ratio is 1:3 and the area is 120,000 sq. m. -/
theorem cyclist_round_time (L B : ℝ) (h_ratio : B = 3 * L) (h_area : L * B = 120000) :
  (2 * L + 2 * B) / (12000 / 60) = 8 := by
  sorry

end NUMINAMATH_CALUDE_cyclist_round_time_l403_40356


namespace NUMINAMATH_CALUDE_monotonic_cubic_function_l403_40344

/-- A function f(x) = x^3 - 3x^2 + ax - 5 is monotonically increasing on ℝ if and only if a ≥ 3 -/
theorem monotonic_cubic_function (a : ℝ) :
  (∀ x : ℝ, Monotone (fun x => x^3 - 3*x^2 + a*x - 5)) ↔ a ≥ 3 := by
sorry

end NUMINAMATH_CALUDE_monotonic_cubic_function_l403_40344


namespace NUMINAMATH_CALUDE_race_head_start_l403_40336

theorem race_head_start (L : ℝ) (va vb : ℝ) (h : va = (15 / 13) * vb) :
  let H := (L - (13 * L / 15) + (1 / 4 * L))
  H = (23 / 60) * L := by sorry

end NUMINAMATH_CALUDE_race_head_start_l403_40336


namespace NUMINAMATH_CALUDE_new_average_after_grace_marks_l403_40350

theorem new_average_after_grace_marks 
  (num_students : ℕ) 
  (original_average : ℚ) 
  (grace_marks : ℚ) 
  (h1 : num_students = 35) 
  (h2 : original_average = 37) 
  (h3 : grace_marks = 3) : 
  (num_students * original_average + num_students * grace_marks) / num_students = 40 := by
sorry

end NUMINAMATH_CALUDE_new_average_after_grace_marks_l403_40350


namespace NUMINAMATH_CALUDE_no_five_cent_combination_l403_40391

/-- Represents the types of coins available -/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | HalfDollar

/-- Returns the value of a coin in cents -/
def coinValue (c : Coin) : ℕ :=
  match c with
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.HalfDollar => 50

/-- A function that takes a list of 5 coins and returns their total value in cents -/
def totalValue (coins : List Coin) : ℕ :=
  coins.map coinValue |>.sum

/-- Theorem stating that it's impossible to select 5 coins with a total value of 5 cents -/
theorem no_five_cent_combination :
  ¬ ∃ (coins : List Coin), coins.length = 5 ∧ totalValue coins = 5 := by
  sorry


end NUMINAMATH_CALUDE_no_five_cent_combination_l403_40391


namespace NUMINAMATH_CALUDE_halfway_fraction_l403_40379

theorem halfway_fraction (a b c d : ℕ) (h1 : a = 3 ∧ b = 4) (h2 : c = 5 ∧ d = 7) :
  (a / b + c / d) / 2 = 41 / 56 :=
sorry

end NUMINAMATH_CALUDE_halfway_fraction_l403_40379


namespace NUMINAMATH_CALUDE_puzzle_solution_l403_40382

/-- Represents a single digit (0-9) -/
def Digit := Fin 10

/-- Represents a two-digit number -/
def TwoDigitNumber := Fin 100

theorem puzzle_solution (P Q R S : Digit) 
  (h1 : (P.val * 10 + Q.val) * R.val = S.val * 10 + P.val)
  (h2 : (P.val * 10 + Q.val) + (R.val * 10 + P.val) = S.val * 10 + Q.val) :
  S.val = 1 := by
  sorry

end NUMINAMATH_CALUDE_puzzle_solution_l403_40382


namespace NUMINAMATH_CALUDE_city_population_ratio_l403_40305

theorem city_population_ratio :
  ∀ (pop_X pop_Y pop_Z : ℕ),
    pop_X = 3 * pop_Y →
    pop_Y = 2 * pop_Z →
    pop_X / pop_Z = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_city_population_ratio_l403_40305


namespace NUMINAMATH_CALUDE_shelby_driving_time_l403_40330

/-- Represents the driving scenario for Shelby --/
structure DrivingScenario where
  sunnySpeed : ℝ  -- Speed in miles per hour when not raining
  rainySpeed : ℝ  -- Speed in miles per hour when raining
  totalDistance : ℝ  -- Total distance traveled in miles
  totalTime : ℝ  -- Total time traveled in hours

/-- Calculates the time spent driving in the rain --/
def timeInRain (scenario : DrivingScenario) : ℝ :=
  -- The actual calculation is not implemented here
  sorry

/-- Theorem stating that the time spent driving in the rain is 40 minutes --/
theorem shelby_driving_time (scenario : DrivingScenario) 
  (h1 : scenario.sunnySpeed = 40)
  (h2 : scenario.rainySpeed = 25)
  (h3 : scenario.totalDistance = 20)
  (h4 : scenario.totalTime = 0.75)  -- 45 minutes in hours
  : timeInRain scenario = 40 / 60 := by
  sorry


end NUMINAMATH_CALUDE_shelby_driving_time_l403_40330


namespace NUMINAMATH_CALUDE_not_all_probabilities_equal_l403_40343

/-- Represents a student in the sampling process -/
structure Student :=
  (id : Nat)

/-- Represents the sampling process -/
structure SamplingProcess :=
  (totalStudents : Nat)
  (selectedStudents : Nat)
  (excludedStudents : Nat)

/-- Represents the probability of a student being selected -/
def selectionProbability (student : Student) (process : SamplingProcess) : ℝ :=
  sorry

/-- The main theorem stating that not all probabilities are equal -/
theorem not_all_probabilities_equal
  (process : SamplingProcess)
  (h1 : process.totalStudents = 2010)
  (h2 : process.selectedStudents = 50)
  (h3 : process.excludedStudents = 10) :
  ∃ (s1 s2 : Student), selectionProbability s1 process ≠ selectionProbability s2 process :=
sorry

end NUMINAMATH_CALUDE_not_all_probabilities_equal_l403_40343


namespace NUMINAMATH_CALUDE_second_parentheses_zero_l403_40367

-- Define the custom operation
def diamond (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

-- State the theorem
theorem second_parentheses_zero : 
  let x : ℝ := Real.sqrt 6
  (diamond x x = (x + x)^2 - (x - x)^2) ∧ (x - x = 0) := by sorry

end NUMINAMATH_CALUDE_second_parentheses_zero_l403_40367


namespace NUMINAMATH_CALUDE_symmetric_points_determine_a_l403_40398

/-- Two points are symmetric about the x-axis if their x-coordinates are equal and their y-coordinates are negatives of each other -/
def symmetric_about_x_axis (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = p2.1 ∧ p1.2 = -p2.2

theorem symmetric_points_determine_a :
  ∀ a b : ℝ,
  let M : ℝ × ℝ := (2*a + b, a - 2*b)
  let N : ℝ × ℝ := (1 - 2*b, -2*a - b - 1)
  symmetric_about_x_axis M N → a = 2 := by
sorry

end NUMINAMATH_CALUDE_symmetric_points_determine_a_l403_40398


namespace NUMINAMATH_CALUDE_perpendicular_foot_is_circumcenter_l403_40317

-- Define the plane
variable (π : Set (Fin 3 → ℝ))

-- Define points
variable (P A B C : Fin 3 → ℝ)

-- Define the foot of the perpendicular
variable (F : Fin 3 → ℝ)

-- P is outside the plane
axiom P_outside : P ∉ π

-- A, B, C are in the plane
axiom A_in_plane : A ∈ π
axiom B_in_plane : B ∈ π
axiom C_in_plane : C ∈ π

-- F is in the plane
axiom F_in_plane : F ∈ π

-- PA, PB, PC are equal
axiom equal_distances : norm (P - A) = norm (P - B) ∧ norm (P - B) = norm (P - C)

-- F is on the perpendicular from P to the plane
axiom F_on_perpendicular : ∀ X ∈ π, norm (P - F) ≤ norm (P - X)

-- Define what it means for F to be the circumcenter of triangle ABC
def is_circumcenter (F A B C : Fin 3 → ℝ) : Prop :=
  norm (F - A) = norm (F - B) ∧ norm (F - B) = norm (F - C)

-- The theorem to prove
theorem perpendicular_foot_is_circumcenter :
  is_circumcenter F A B C :=
sorry

end NUMINAMATH_CALUDE_perpendicular_foot_is_circumcenter_l403_40317


namespace NUMINAMATH_CALUDE_grants_yearly_expense_l403_40366

/-- Grant's yearly newspaper delivery expense --/
def grants_expense : ℝ := 200

/-- Juanita's daily expense from Monday to Saturday --/
def juanita_weekday_expense : ℝ := 0.5

/-- Juanita's Sunday expense --/
def juanita_sunday_expense : ℝ := 2

/-- Number of weeks in a year --/
def weeks_per_year : ℕ := 52

/-- Difference between Juanita's and Grant's yearly expenses --/
def expense_difference : ℝ := 60

/-- Theorem stating Grant's yearly newspaper delivery expense --/
theorem grants_yearly_expense : 
  grants_expense = 
    weeks_per_year * (6 * juanita_weekday_expense + juanita_sunday_expense) - expense_difference :=
by sorry

end NUMINAMATH_CALUDE_grants_yearly_expense_l403_40366


namespace NUMINAMATH_CALUDE_probability_at_least_10_rubles_l403_40333

-- Define the total number of tickets
def total_tickets : ℕ := 100

-- Define the number of tickets for each prize category
def tickets_20_rubles : ℕ := 5
def tickets_15_rubles : ℕ := 10
def tickets_10_rubles : ℕ := 15
def tickets_2_rubles : ℕ := 25

-- Define the probability of winning at least 10 rubles
def prob_at_least_10_rubles : ℚ :=
  (tickets_20_rubles + tickets_15_rubles + tickets_10_rubles) / total_tickets

-- Theorem statement
theorem probability_at_least_10_rubles :
  prob_at_least_10_rubles = 30 / 100 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_10_rubles_l403_40333


namespace NUMINAMATH_CALUDE_simplify_expression_l403_40378

theorem simplify_expression (x : ℝ) : (3*x)^4 - (2*x)*(x^3) = 79*x^4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l403_40378


namespace NUMINAMATH_CALUDE_point_on_600_degree_angle_l403_40393

theorem point_on_600_degree_angle (a : ℝ) : 
  (∃ θ : ℝ, θ = 600 * Real.pi / 180 ∧ 
   (-1 : ℝ) = Real.cos θ ∧ 
   a = Real.sin θ) → 
  a = -Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_point_on_600_degree_angle_l403_40393


namespace NUMINAMATH_CALUDE_unique_solution_cubic_system_l403_40361

theorem unique_solution_cubic_system (x y z : ℝ) 
  (h1 : x + y + z = 3)
  (h2 : x^2 + y^2 + z^2 = 3)
  (h3 : x^3 + y^3 + z^3 = 3) :
  x = 1 ∧ y = 1 ∧ z = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_cubic_system_l403_40361


namespace NUMINAMATH_CALUDE_terminal_side_quadrant_l403_40332

-- Define the quadrants
inductive Quadrant
  | I
  | II
  | III
  | IV

-- Define a function to determine the quadrant of an angle
def angle_quadrant (α : Real) : Quadrant := sorry

-- Define the theorem
theorem terminal_side_quadrant (α : Real) 
  (h1 : Real.sin α * Real.cos α < 0) 
  (h2 : Real.sin α * Real.tan α > 0) : 
  (angle_quadrant (α / 2) = Quadrant.II) ∨ (angle_quadrant (α / 2) = Quadrant.IV) := by
  sorry

end NUMINAMATH_CALUDE_terminal_side_quadrant_l403_40332


namespace NUMINAMATH_CALUDE_marias_minimum_score_l403_40313

/-- The minimum score needed in the fifth term to achieve a given average -/
def minimum_fifth_score (score1 score2 score3 score4 : ℝ) (required_average : ℝ) : ℝ :=
  5 * required_average - (score1 + score2 + score3 + score4)

/-- Theorem: Maria's minimum required score for the 5th term is 101% -/
theorem marias_minimum_score :
  minimum_fifth_score 84 80 82 78 85 = 101 := by
  sorry

end NUMINAMATH_CALUDE_marias_minimum_score_l403_40313


namespace NUMINAMATH_CALUDE_line_equation_through_midpoint_l403_40351

/-- Given an ellipse and a point M, prove the equation of a line passing through M and intersecting the ellipse at two points where M is their midpoint. -/
theorem line_equation_through_midpoint (x y : ℝ) : 
  let M : ℝ × ℝ := (2, 1)
  let ellipse (x y : ℝ) := x^2 / 16 + y^2 / 4 = 1
  ∃ A B : ℝ × ℝ, 
    ellipse A.1 A.2 ∧ 
    ellipse B.1 B.2 ∧ 
    M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
    ∃ l : ℝ → ℝ → Prop, 
      (∀ x y, l x y ↔ x + 2*y - 4 = 0) ∧
      l A.1 A.2 ∧ 
      l B.1 B.2 ∧ 
      l M.1 M.2 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_through_midpoint_l403_40351


namespace NUMINAMATH_CALUDE_weight_loss_difference_l403_40353

theorem weight_loss_difference (total_loss weight_first weight_third weight_fourth : ℕ) :
  total_loss = weight_first + weight_third + weight_fourth + (weight_first - 7) →
  weight_third = weight_fourth →
  total_loss = 103 →
  weight_first = 27 →
  weight_third = 28 →
  7 = weight_first - (total_loss - weight_first - weight_third - weight_fourth) :=
by sorry

end NUMINAMATH_CALUDE_weight_loss_difference_l403_40353


namespace NUMINAMATH_CALUDE_like_terms_imply_n_eq_one_l403_40375

/-- Two terms are considered like terms if they have the same variables with the same exponents -/
def like_terms (term1 term2 : ℤ → ℤ → ℤ) : Prop :=
  ∃ (a b : ℤ), ∀ (x y : ℤ), term1 x y = a * x^2 * y ∧ term2 x y = b * x^2 * y

/-- If -x^2y^n and 3yx^2 are like terms, then n = 1 -/
theorem like_terms_imply_n_eq_one :
  ∀ n : ℕ, like_terms (λ x y => -x^2 * y^n) (λ x y => 3 * y * x^2) → n = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_like_terms_imply_n_eq_one_l403_40375


namespace NUMINAMATH_CALUDE_factor_and_multiple_greatest_factor_smallest_multiple_smallest_multiple_one_prime_sum_10_product_21_prime_sum_20_product_91_l403_40312

-- (1)
theorem factor_and_multiple (n : ℕ) : 
  n ∣ 42 ∧ 7 ∣ n ∧ 2 ∣ n ∧ 3 ∣ n → n = 42 := by sorry

-- (2)
theorem greatest_factor_smallest_multiple (n : ℕ) :
  (∀ m : ℕ, m ∣ n → m ≤ 18) ∧ (∀ k : ℕ, n ∣ k → k ≥ 18) → n = 18 := by sorry

-- (3)
theorem smallest_multiple_one (n : ℕ) :
  (∀ k : ℕ, n ∣ k → k ≥ 1) → n = 1 := by sorry

-- (4)
theorem prime_sum_10_product_21 (p q : ℕ) :
  Prime p ∧ Prime q ∧ p + q = 10 ∧ p * q = 21 → (p = 3 ∧ q = 7) ∨ (p = 7 ∧ q = 3) := by sorry

-- (5)
theorem prime_sum_20_product_91 (p q : ℕ) :
  Prime p ∧ Prime q ∧ p + q = 20 ∧ p * q = 91 → (p = 13 ∧ q = 7) ∨ (p = 7 ∧ q = 13) := by sorry

end NUMINAMATH_CALUDE_factor_and_multiple_greatest_factor_smallest_multiple_smallest_multiple_one_prime_sum_10_product_21_prime_sum_20_product_91_l403_40312


namespace NUMINAMATH_CALUDE_max_M_value_l403_40357

/-- Given a system of equations and conditions, prove the maximum value of M --/
theorem max_M_value (x y z u : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hu : u > 0)
  (heq1 : x - 2*y = z - 2*u) (heq2 : 2*y*z = u*x) (hyz : z ≥ y) :
  ∃ (M : ℝ), M > 0 ∧ M ≤ z/y ∧ ∀ (N : ℝ), (N > 0 ∧ N ≤ z/y → N ≤ 6 + 4*Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_max_M_value_l403_40357


namespace NUMINAMATH_CALUDE_angle_equality_l403_40302

-- Define the problem statement
theorem angle_equality (θ : Real) (h1 : 0 < θ ∧ θ < π / 2) 
  (h2 : Real.sqrt 3 * Real.sin (15 * π / 180) = Real.cos θ + Real.sin θ) : 
  θ = 15 * π / 180 :=
sorry

end NUMINAMATH_CALUDE_angle_equality_l403_40302


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l403_40380

theorem inscribed_circle_radius (d : ℝ) (h : d = Real.sqrt 12) : 
  let R := d / 2
  let s := R * Real.sqrt 3
  let h := (Real.sqrt 3 / 2) * s
  let a := Real.sqrt (h^2 - (h/2)^2)
  let r := (a * Real.sqrt 3) / 6
  r = 9/8 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l403_40380


namespace NUMINAMATH_CALUDE_kabadi_players_count_l403_40387

/-- The number of people who play kabadi -/
def kabadi_players : ℕ := 15

/-- The number of people who play kho kho only -/
def kho_kho_only : ℕ := 15

/-- The number of people who play both games -/
def both_games : ℕ := 5

/-- The total number of players -/
def total_players : ℕ := 25

/-- Theorem stating that the number of kabadi players is correct given the conditions -/
theorem kabadi_players_count : 
  kabadi_players = total_players - kho_kho_only + both_games :=
by sorry

end NUMINAMATH_CALUDE_kabadi_players_count_l403_40387


namespace NUMINAMATH_CALUDE_power_of_product_l403_40321

theorem power_of_product (a : ℝ) : (-2 * a^4)^3 = -8 * a^12 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l403_40321
