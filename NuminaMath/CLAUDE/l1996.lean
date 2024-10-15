import Mathlib

namespace NUMINAMATH_CALUDE_triangle_area_fraction_l1996_199677

-- Define the grid dimensions
def grid_width : ℕ := 8
def grid_height : ℕ := 6

-- Define the triangle vertices
def point_A : ℚ × ℚ := (2, 5)
def point_B : ℚ × ℚ := (7, 2)
def point_C : ℚ × ℚ := (6, 6)

-- Function to calculate the area of a triangle using the Shoelace formula
def triangle_area (p1 p2 p3 : ℚ × ℚ) : ℚ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs (x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))

-- Theorem statement
theorem triangle_area_fraction :
  (triangle_area point_A point_B point_C) / (grid_width * grid_height : ℚ) = 17/96 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_fraction_l1996_199677


namespace NUMINAMATH_CALUDE_gaming_chair_price_proof_l1996_199628

/-- The price of a set of toy organizers -/
def toy_organizer_price : ℝ := 78

/-- The number of toy organizer sets ordered -/
def toy_organizer_sets : ℕ := 3

/-- The number of gaming chairs ordered -/
def gaming_chairs : ℕ := 2

/-- The delivery fee percentage -/
def delivery_fee_percent : ℝ := 0.05

/-- The total amount Leon paid -/
def total_paid : ℝ := 420

/-- The price of a gaming chair -/
def gaming_chair_price : ℝ := 83

theorem gaming_chair_price_proof :
  gaming_chair_price * gaming_chairs + toy_organizer_price * toy_organizer_sets +
  (gaming_chair_price * gaming_chairs + toy_organizer_price * toy_organizer_sets) * delivery_fee_percent =
  total_paid := by sorry

end NUMINAMATH_CALUDE_gaming_chair_price_proof_l1996_199628


namespace NUMINAMATH_CALUDE_line_plane_relationship_l1996_199662

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism relation between a line and a plane
variable (parallel : Line → Plane → Prop)

-- Define the containment relation between a line and a plane
variable (contains : Plane → Line → Prop)

-- Define the parallelism relation between two lines
variable (parallel_lines : Line → Line → Prop)

-- Define the "on different planes" relation between two lines
variable (different_planes : Line → Line → Prop)

-- State the theorem
theorem line_plane_relationship (m n : Line) (α : Plane) 
  (h1 : parallel m α) (h2 : contains α n) :
  parallel_lines m n ∨ different_planes m n :=
sorry

end NUMINAMATH_CALUDE_line_plane_relationship_l1996_199662


namespace NUMINAMATH_CALUDE_power_function_through_point_l1996_199617

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, f x = x ^ a

-- Theorem statement
theorem power_function_through_point :
  ∀ f : ℝ → ℝ,
  isPowerFunction f →
  f 2 = 32 →
  ∀ x : ℝ, f x = x^5 := by
sorry

end NUMINAMATH_CALUDE_power_function_through_point_l1996_199617


namespace NUMINAMATH_CALUDE_seeds_per_flower_bed_l1996_199616

theorem seeds_per_flower_bed 
  (total_seeds : ℕ) 
  (num_flower_beds : ℕ) 
  (h1 : total_seeds = 54) 
  (h2 : num_flower_beds = 9) 
  (h3 : num_flower_beds ≠ 0) : 
  total_seeds / num_flower_beds = 6 := by
sorry

end NUMINAMATH_CALUDE_seeds_per_flower_bed_l1996_199616


namespace NUMINAMATH_CALUDE_n_squared_plus_inverse_squared_plus_six_l1996_199609

theorem n_squared_plus_inverse_squared_plus_six (n : ℝ) (h : n + 1/n = 10) :
  n^2 + 1/n^2 + 6 = 104 := by
  sorry

end NUMINAMATH_CALUDE_n_squared_plus_inverse_squared_plus_six_l1996_199609


namespace NUMINAMATH_CALUDE_treasure_chest_problem_l1996_199645

theorem treasure_chest_problem (n : ℕ) : 
  (n > 0 ∧ n % 8 = 6 ∧ n % 9 = 5) → 
  (∀ m : ℕ, m > 0 ∧ m % 8 = 6 ∧ m % 9 = 5 → n ≤ m) → 
  (n = 14 ∧ n % 11 = 3) := by
sorry

end NUMINAMATH_CALUDE_treasure_chest_problem_l1996_199645


namespace NUMINAMATH_CALUDE_tangent_line_and_monotonicity_l1996_199620

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 - 9*x - 1

-- Define the derivative of f(x)
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x - 9

-- Theorem statement
theorem tangent_line_and_monotonicity 
  (a : ℝ) 
  (h1 : a < 0) 
  (h2 : ∃ x₀, ∀ x, f' a x₀ ≤ f' a x ∧ f' a x₀ = -12) :
  a = -3 ∧ 
  (∀ x₁ x₂, x₁ < x₂ → 
    ((x₂ < -1 → f a x₁ < f a x₂) ∧
     (x₁ > 3 → f a x₁ < f a x₂) ∧
     (-1 < x₁ ∧ x₂ < 3 → f a x₁ > f a x₂))) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_and_monotonicity_l1996_199620


namespace NUMINAMATH_CALUDE_equal_variance_implies_arithmetic_square_alternating_sequence_is_equal_variance_equal_variance_subsequence_l1996_199689

-- Define the equal variance sequence property
def is_equal_variance_sequence (a : ℕ+ → ℝ) : Prop :=
  ∃ p : ℝ, ∀ n : ℕ+, a n ^ 2 - a (n + 1) ^ 2 = p

-- Define arithmetic sequence property
def is_arithmetic_sequence (b : ℕ+ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ+, b (n + 1) - b n = d

-- Statement 1
theorem equal_variance_implies_arithmetic_square (a : ℕ+ → ℝ) :
  is_equal_variance_sequence a → is_arithmetic_sequence (λ n => a n ^ 2) := by sorry

-- Statement 2
theorem alternating_sequence_is_equal_variance :
  is_equal_variance_sequence (λ n => (-1) ^ (n : ℕ)) := by sorry

-- Statement 3
theorem equal_variance_subsequence (a : ℕ+ → ℝ) (k : ℕ+) :
  is_equal_variance_sequence a → is_equal_variance_sequence (λ n => a (k * n)) := by sorry

end NUMINAMATH_CALUDE_equal_variance_implies_arithmetic_square_alternating_sequence_is_equal_variance_equal_variance_subsequence_l1996_199689


namespace NUMINAMATH_CALUDE_least_possible_value_z_minus_x_l1996_199631

theorem least_possible_value_z_minus_x 
  (x y z : ℤ) 
  (h1 : x < y ∧ y < z) 
  (h2 : y - x > 11) 
  (h3 : Even x) 
  (h4 : Odd y ∧ Odd z) :
  ∀ w, w = z - x → w ≥ 15 ∧ ∃ (x' y' z' : ℤ), 
    x' < y' ∧ y' < z' ∧ 
    y' - x' > 11 ∧ 
    Even x' ∧ Odd y' ∧ Odd z' ∧ 
    z' - x' = 15 :=
by sorry

end NUMINAMATH_CALUDE_least_possible_value_z_minus_x_l1996_199631


namespace NUMINAMATH_CALUDE_max_area_rectangle_perimeter_36_l1996_199682

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.width * r.height

/-- Theorem: The maximum area of a rectangle with perimeter 36 is 81 -/
theorem max_area_rectangle_perimeter_36 :
  (∃ (r : Rectangle), perimeter r = 36 ∧ 
    ∀ (s : Rectangle), perimeter s = 36 → area s ≤ area r) ∧
  (∀ (r : Rectangle), perimeter r = 36 → area r ≤ 81) := by
  sorry

end NUMINAMATH_CALUDE_max_area_rectangle_perimeter_36_l1996_199682


namespace NUMINAMATH_CALUDE_midpoint_exists_but_no_centroid_l1996_199613

/-- A triangle in 2D space -/
structure Triangle :=
  (v1 v2 v3 : ℝ × ℝ)

/-- Check if a point is inside a triangle -/
def isInsideTriangle (t : Triangle) (p : ℝ × ℝ) : Prop :=
  sorry

/-- Check if a point is on the perimeter of a triangle -/
def isOnPerimeter (t : Triangle) (p : ℝ × ℝ) : Prop :=
  sorry

/-- Check if a point is the midpoint of a line segment -/
def isMidpoint (a b m : ℝ × ℝ) : Prop :=
  sorry

/-- Check if a point is the centroid of a triangle -/
def isCentroid (t : Triangle) (c : ℝ × ℝ) : Prop :=
  sorry

theorem midpoint_exists_but_no_centroid (t : Triangle) (p : ℝ × ℝ) 
  (h : isInsideTriangle t p) :
  (∃ a b : ℝ × ℝ, isOnPerimeter t a ∧ isOnPerimeter t b ∧ isMidpoint a b p) ∧
  (¬ ∃ a b c : ℝ × ℝ, isOnPerimeter t a ∧ isOnPerimeter t b ∧ isOnPerimeter t c ∧
                      isCentroid (Triangle.mk a b c) p) :=
by sorry

end NUMINAMATH_CALUDE_midpoint_exists_but_no_centroid_l1996_199613


namespace NUMINAMATH_CALUDE_shifted_quadratic_sum_l1996_199608

/-- Given a quadratic function f(x) = 3x^2 + 2x + 5, shifting it 5 units to the left
    results in a new quadratic function g(x) = ax^2 + bx + c.
    This theorem proves that a + b + c = 125. -/
theorem shifted_quadratic_sum (a b c : ℝ) : 
  (∀ x, 3 * (x + 5)^2 + 2 * (x + 5) + 5 = a * x^2 + b * x + c) →
  a + b + c = 125 := by
sorry

end NUMINAMATH_CALUDE_shifted_quadratic_sum_l1996_199608


namespace NUMINAMATH_CALUDE_same_color_pair_count_l1996_199660

/-- The number of ways to choose a pair of socks of the same color -/
def choose_same_color_pair (white : Nat) (brown : Nat) (blue : Nat) : Nat :=
  Nat.choose white 2 + Nat.choose brown 2 + Nat.choose blue 2

/-- Theorem: The number of ways to choose a pair of socks of the same color
    from 4 white, 4 brown, and 2 blue socks is 13 -/
theorem same_color_pair_count :
  choose_same_color_pair 4 4 2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_same_color_pair_count_l1996_199660


namespace NUMINAMATH_CALUDE_collinear_vectors_x_value_l1996_199637

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2)

/-- Given that vector a = (2, 4) is collinear with vector b = (x, 6), prove that x = 3 -/
theorem collinear_vectors_x_value :
  ∀ x : ℝ, collinear (2, 4) (x, 6) → x = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_x_value_l1996_199637


namespace NUMINAMATH_CALUDE_line_parallel_to_plane_l1996_199641

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (perp_planes : Plane → Plane → Prop)

-- State the theorem
theorem line_parallel_to_plane
  (m : Line) (α β : Plane)
  (h1 : perp_planes α β)
  (h2 : perpendicular m β)
  (h3 : ¬ contains α m) :
  parallel m α :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_plane_l1996_199641


namespace NUMINAMATH_CALUDE_train_speed_l1996_199658

/-- The speed of a train given its length and time to cross a fixed point. -/
theorem train_speed (length time : ℝ) (h1 : length = 400) (h2 : time = 16) :
  length / time = 25 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l1996_199658


namespace NUMINAMATH_CALUDE_sequence_sum_formula_l1996_199634

/-- Given a sequence a with a₁ = 1 and Sₙ = n² * aₙ for all positive integers n,
    prove that the sum of the first n terms Sₙ is equal to 2n / (n+1). -/
theorem sequence_sum_formula (a : ℕ → ℚ) (S : ℕ → ℚ) :
  a 1 = 1 →
  (∀ n : ℕ, n > 0 → S n = n^2 * a n) →
  ∀ n : ℕ, n > 0 → S n = 2 * n / (n + 1) :=
by sorry

end NUMINAMATH_CALUDE_sequence_sum_formula_l1996_199634


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1996_199663

theorem partial_fraction_decomposition :
  ∀ x : ℝ, x ≠ 0 → x ≠ 1 → x ≠ -1 →
  (-x^2 + 5*x - 6) / (x^3 - x) = 6 / x + (-7*x + 5) / (x^2 - 1) :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1996_199663


namespace NUMINAMATH_CALUDE_A_times_B_equals_result_l1996_199636

-- Define the sets A and B
def A : Set ℝ := {x | |x - 1/2| < 1}
def B : Set ℝ := {x | 1/x ≥ 1}

-- Define the operation ×
def times (X Y : Set ℝ) : Set ℝ := {x | x ∈ X ∪ Y ∧ x ∉ X ∩ Y}

-- State the theorem
theorem A_times_B_equals_result : 
  times A B = {x | -1/2 < x ∧ x ≤ 0 ∨ 1 < x ∧ x < 3/2} := by sorry

end NUMINAMATH_CALUDE_A_times_B_equals_result_l1996_199636


namespace NUMINAMATH_CALUDE_treasure_in_blown_out_dunes_l1996_199665

/-- The probability that a sand dune remains after being formed -/
def prob_remain : ℚ := 1 / 3

/-- The probability that a sand dune has a lucky coupon -/
def prob_lucky_coupon : ℚ := 2 / 3

/-- The probability that a blown-out sand dune contains both treasure and lucky coupon -/
def prob_both : ℚ := 8888888888888889 / 100000000000000000

/-- The number of blown-out sand dunes considered to find the one with treasure -/
def num_blown_out_dunes : ℕ := 8

theorem treasure_in_blown_out_dunes :
  ∃ (n : ℕ), n = num_blown_out_dunes ∧ 
  (1 : ℚ) / n * prob_lucky_coupon = prob_both ∧
  n = ⌈(1 : ℚ) / (prob_both / prob_lucky_coupon)⌉ :=
sorry

end NUMINAMATH_CALUDE_treasure_in_blown_out_dunes_l1996_199665


namespace NUMINAMATH_CALUDE_unique_a_value_l1996_199627

-- Define the inequality function
def inequality (a x : ℝ) : Prop :=
  (a * x - 20) * Real.log (2 * a / x) ≤ 0

-- State the theorem
theorem unique_a_value : 
  ∃! a : ℝ, ∀ x : ℝ, x > 0 → inequality a x :=
by
  -- The unique value of a is √10
  use Real.sqrt 10
  sorry -- Proof omitted

end NUMINAMATH_CALUDE_unique_a_value_l1996_199627


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1996_199606

-- Define an arithmetic sequence
def isArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  isArithmeticSequence a →
  a 1 + 2 * a 8 + a 15 = 96 →
  2 * a 9 - a 10 = 24 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1996_199606


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l1996_199656

theorem trigonometric_simplification (x y : ℝ) :
  Real.sin x ^ 2 + Real.sin (x + y) ^ 2 - 2 * Real.sin x * Real.sin y * Real.sin (x + y) = Real.sin y ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l1996_199656


namespace NUMINAMATH_CALUDE_simplify_expression_l1996_199633

variable (x y : ℝ)

theorem simplify_expression (hx : x ≠ 0) (hy : y ≠ 0) : 
  (6 * x^2 * y - 2 * x * y^2) / (2 * x * y) = 3 * x - y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1996_199633


namespace NUMINAMATH_CALUDE_first_term_of_geometric_series_l1996_199671

/-- Given an infinite geometric series with first term a and common ratio r -/
def InfiniteGeometricSeries (a : ℝ) (r : ℝ) : Prop :=
  |r| < 1

theorem first_term_of_geometric_series
  (a : ℝ) (r : ℝ)
  (h_series : InfiniteGeometricSeries a r)
  (h_sum : a / (1 - r) = 30)
  (h_sum_squares : a^2 / (1 - r^2) = 180) :
  a = 10 := by
sorry

end NUMINAMATH_CALUDE_first_term_of_geometric_series_l1996_199671


namespace NUMINAMATH_CALUDE_secretary_donuts_donut_problem_l1996_199646

theorem secretary_donuts (initial : ℕ) (bill_eaten : ℕ) (final : ℕ) : ℕ :=
  let remaining_after_bill := initial - bill_eaten
  let remaining_after_coworkers := final * 2
  let secretary_taken := remaining_after_bill - remaining_after_coworkers
  secretary_taken

theorem donut_problem :
  secretary_donuts 50 2 22 = 4 := by sorry

end NUMINAMATH_CALUDE_secretary_donuts_donut_problem_l1996_199646


namespace NUMINAMATH_CALUDE_jerry_tickets_l1996_199681

theorem jerry_tickets (initial_tickets spent_tickets later_won_tickets current_tickets : ℕ) :
  spent_tickets = 2 →
  later_won_tickets = 47 →
  current_tickets = 49 →
  initial_tickets = current_tickets - later_won_tickets + spent_tickets →
  initial_tickets = 4 := by
sorry

end NUMINAMATH_CALUDE_jerry_tickets_l1996_199681


namespace NUMINAMATH_CALUDE_limit_rational_function_l1996_199603

/-- The limit of (2x³ - 3x² + 5x + 7) / (3x³ + 4x² - x + 2) as x approaches infinity is 2/3 -/
theorem limit_rational_function : 
  ∀ ε > 0, ∃ N : ℝ, ∀ x : ℝ, x > N → 
    |((2 * x^3 - 3 * x^2 + 5 * x + 7) / (3 * x^3 + 4 * x^2 - x + 2)) - 2/3| < ε := by
  sorry

end NUMINAMATH_CALUDE_limit_rational_function_l1996_199603


namespace NUMINAMATH_CALUDE_typing_job_solution_l1996_199614

/-- Represents the time taken by two typists to complete a typing job -/
structure TypingJob where
  combined_time : ℝ  -- Time taken when working together
  sequential_time : ℝ  -- Time taken when working sequentially (half each)
  first_typist_time : ℝ  -- Time for first typist to complete job alone
  second_typist_time : ℝ  -- Time for second typist to complete job alone

/-- Theorem stating the solution to the typing job problem -/
theorem typing_job_solution (job : TypingJob) 
  (h1 : job.combined_time = 12)
  (h2 : job.sequential_time = 25)
  (h3 : job.first_typist_time + job.second_typist_time = 50)
  (h4 : job.first_typist_time * job.second_typist_time = 600) :
  job.first_typist_time = 20 ∧ job.second_typist_time = 30 := by
  sorry

#check typing_job_solution

end NUMINAMATH_CALUDE_typing_job_solution_l1996_199614


namespace NUMINAMATH_CALUDE_toy_store_problem_l1996_199639

/-- Toy store problem -/
theorem toy_store_problem 
  (cost_sum : ℝ) 
  (budget_A budget_B : ℝ) 
  (total_toys : ℕ) 
  (max_A : ℕ) 
  (total_budget : ℝ) 
  (sell_price_A sell_price_B : ℝ) :
  cost_sum = 40 →
  budget_A = 90 →
  budget_B = 150 →
  total_toys = 48 →
  max_A = 23 →
  total_budget = 1000 →
  sell_price_A = 30 →
  sell_price_B = 45 →
  ∃ (cost_A cost_B : ℝ) (num_plans : ℕ) (profit_function : ℕ → ℝ) (max_profit : ℝ),
    cost_A + cost_B = cost_sum ∧
    budget_A / cost_A = budget_B / cost_B ∧
    cost_A = 15 ∧
    cost_B = 25 ∧
    num_plans = 4 ∧
    (∀ m : ℕ, profit_function m = -5 * m + 960) ∧
    max_profit = 860 :=
by sorry

end NUMINAMATH_CALUDE_toy_store_problem_l1996_199639


namespace NUMINAMATH_CALUDE_contradiction_proof_l1996_199666

theorem contradiction_proof (a b c d : ℝ) 
  (sum1 : a + b = 1) 
  (sum2 : c + d = 1) 
  (prod_sum : a * c + b * d > 1) 
  (all_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) : 
  False :=
sorry

end NUMINAMATH_CALUDE_contradiction_proof_l1996_199666


namespace NUMINAMATH_CALUDE_final_painting_width_l1996_199612

theorem final_painting_width :
  let total_paintings : ℕ := 5
  let total_area : ℝ := 200
  let small_painting_count : ℕ := 3
  let small_painting_side : ℝ := 5
  let large_painting_width : ℝ := 10
  let large_painting_height : ℝ := 8
  let final_painting_height : ℝ := 5

  let small_paintings_area : ℝ := small_painting_count * small_painting_side * small_painting_side
  let large_painting_area : ℝ := large_painting_width * large_painting_height
  let known_paintings_area : ℝ := small_paintings_area + large_painting_area
  let final_painting_area : ℝ := total_area - known_paintings_area
  let final_painting_width : ℝ := final_painting_area / final_painting_height

  final_painting_width = 9 :=
by sorry

end NUMINAMATH_CALUDE_final_painting_width_l1996_199612


namespace NUMINAMATH_CALUDE_smallest_solution_of_quartic_l1996_199672

theorem smallest_solution_of_quartic (x : ℝ) :
  x^4 - 50*x^2 + 576 = 0 →
  x ≥ -Real.sqrt 26 ∧ 
  ∃ y : ℝ, y^4 - 50*y^2 + 576 = 0 ∧ y = -Real.sqrt 26 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_of_quartic_l1996_199672


namespace NUMINAMATH_CALUDE_trigonometric_identities_l1996_199688

theorem trigonometric_identities (α : Real) (h : Real.tan (π / 4 + α) = 3) :
  (Real.tan α = 1 / 2) ∧ 
  (Real.tan (2 * α) = 4 / 3) ∧ 
  ((2 * Real.sin α * Real.cos α + 3 * Real.cos (2 * α)) / 
   (5 * Real.cos (2 * α) - 3 * Real.sin (2 * α)) = 13 / 3) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l1996_199688


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l1996_199647

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (∃ r s : ℝ, (25 - 10*r - r^2 = 0) ∧ (25 - 10*s - s^2 = 0) ∧ (r + s = -10)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l1996_199647


namespace NUMINAMATH_CALUDE_john_initial_payment_l1996_199678

def soda_cost : ℕ := 2
def num_sodas : ℕ := 3
def change_received : ℕ := 14

theorem john_initial_payment :
  num_sodas * soda_cost + change_received = 20 := by
  sorry

end NUMINAMATH_CALUDE_john_initial_payment_l1996_199678


namespace NUMINAMATH_CALUDE_fraction_inequality_l1996_199642

theorem fraction_inequality (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : 0 > c) (h4 : c > d) : 
  c / a - d / b > 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l1996_199642


namespace NUMINAMATH_CALUDE_sin_3phi_value_l1996_199654

theorem sin_3phi_value (φ : ℝ) (h : Complex.exp (Complex.I * φ) = (3 + Complex.I * Real.sqrt 8) / 5) :
  Real.sin (3 * φ) = 19 * Real.sqrt 8 / 125 := by
  sorry

end NUMINAMATH_CALUDE_sin_3phi_value_l1996_199654


namespace NUMINAMATH_CALUDE_brian_shoe_count_l1996_199604

theorem brian_shoe_count :
  ∀ (b e j : ℕ),
    j = e / 2 →
    e = 3 * b →
    b + e + j = 121 →
    b = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_brian_shoe_count_l1996_199604


namespace NUMINAMATH_CALUDE_pig_count_l1996_199685

theorem pig_count (P H : ℕ) : 
  4 * P + 2 * H = 2 * (P + H) + 22 → P = 11 := by
sorry

end NUMINAMATH_CALUDE_pig_count_l1996_199685


namespace NUMINAMATH_CALUDE_vector_magnitude_l1996_199693

def a (m : ℝ) : ℝ × ℝ := (2, m)
def b (m : ℝ) : ℝ × ℝ := (-1, m)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v = k • w

theorem vector_magnitude (m : ℝ) :
  parallel (2 • (a m) + b m) (b m) → ‖a m‖ = 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l1996_199693


namespace NUMINAMATH_CALUDE_division_problem_l1996_199697

theorem division_problem : (((120 / 5) / 2) / 3) = 4 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1996_199697


namespace NUMINAMATH_CALUDE_simplify_expression_l1996_199699

theorem simplify_expression (y : ℝ) : 3*y + 5*y + 2*y + 7*y = 17*y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1996_199699


namespace NUMINAMATH_CALUDE_drum_fill_time_l1996_199607

/-- The time to fill a cylindrical drum with varying rain rate -/
theorem drum_fill_time (initial_rate : ℝ) (area : ℝ) (depth : ℝ) :
  let rate := fun t : ℝ => initial_rate * t^2
  let volume := area * depth
  let fill_time := (volume * 3 / (5 * initial_rate))^(1/3)
  fill_time^3 = volume * 3 / (5 * initial_rate) :=
by sorry

end NUMINAMATH_CALUDE_drum_fill_time_l1996_199607


namespace NUMINAMATH_CALUDE_triangle_area_ratio_l1996_199655

/-- A rectangle on a coordinate grid with vertices at (0,0), (x,0), (0,y), and (x,y) -/
structure Rectangle where
  x : ℝ
  y : ℝ

/-- The number of parts the diagonals are divided into -/
structure DiagonalDivisions where
  n : ℕ  -- number of parts for diagonal from (0,0) to (x,y)
  m : ℕ  -- number of parts for diagonal from (x,0) to (0,y)

/-- Triangle formed by joining a point on a diagonal to the rectangle's center -/
inductive Triangle
  | A  -- formed from diagonal (0,0) to (x,y)
  | B  -- formed from diagonal (x,0) to (0,y)

/-- The area of a triangle -/
def triangleArea (t : Triangle) (r : Rectangle) (d : DiagonalDivisions) : ℝ :=
  sorry  -- definition omitted as it's not directly given in the problem conditions

/-- The theorem to be proved -/
theorem triangle_area_ratio (r : Rectangle) (d : DiagonalDivisions) :
  triangleArea Triangle.A r d / triangleArea Triangle.B r d = d.m / d.n :=
sorry

end NUMINAMATH_CALUDE_triangle_area_ratio_l1996_199655


namespace NUMINAMATH_CALUDE_price_difference_year_l1996_199684

/-- 
Given:
- The price of commodity X increases by 45 cents every year
- The price of commodity Y increases by 20 cents every year
- In 2001, the price of commodity X was $4.20
- The price of commodity Y in 2001 is Y dollars

Prove that the number of years n after 2001 when the price of X is 65 cents more than 
the price of Y is given by n = (Y - 3.55) / 0.25
-/
theorem price_difference_year (Y : ℝ) : 
  let n : ℝ := (Y - 3.55) / 0.25
  let price_X (t : ℝ) : ℝ := 4.20 + 0.45 * t
  let price_Y (t : ℝ) : ℝ := Y + 0.20 * t
  price_X n = price_Y n + 0.65 :=
by sorry

end NUMINAMATH_CALUDE_price_difference_year_l1996_199684


namespace NUMINAMATH_CALUDE_chocolate_chip_calculation_l1996_199605

/-- Represents the number of cups of chocolate chips per batch in the recipe -/
def cups_per_batch : ℝ := 2.0

/-- Represents the number of batches that can be made with the available chocolate chips -/
def number_of_batches : ℝ := 11.5

/-- Calculates the total number of cups of chocolate chips -/
def total_chocolate_chips : ℝ := cups_per_batch * number_of_batches

/-- Proves that the total number of cups of chocolate chips is 23 -/
theorem chocolate_chip_calculation : total_chocolate_chips = 23 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_chip_calculation_l1996_199605


namespace NUMINAMATH_CALUDE_nested_fourth_root_l1996_199659

theorem nested_fourth_root (M : ℝ) (h : M > 1) :
  (M * (M * (M^(1/4))^(1/4))^(1/4))^(1/4) = M^(21/64) := by
  sorry

end NUMINAMATH_CALUDE_nested_fourth_root_l1996_199659


namespace NUMINAMATH_CALUDE_sin_330_degrees_l1996_199669

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l1996_199669


namespace NUMINAMATH_CALUDE_solutions_absolute_value_equation_l1996_199640

theorem solutions_absolute_value_equation :
  (∀ x : ℝ, |x| = 1 ↔ x = 1 ∨ x = -1) :=
by sorry

end NUMINAMATH_CALUDE_solutions_absolute_value_equation_l1996_199640


namespace NUMINAMATH_CALUDE_carmen_earnings_l1996_199650

-- Define the sales for each house
def green_house_sales : ℕ := 3
def green_house_price : ℚ := 4

def yellow_house_thin_mints : ℕ := 2
def yellow_house_thin_mints_price : ℚ := 3.5
def yellow_house_fudge_delights : ℕ := 1
def yellow_house_fudge_delights_price : ℚ := 5

def brown_house_sales : ℕ := 9
def brown_house_price : ℚ := 2

-- Define the total earnings
def total_earnings : ℚ := 
  green_house_sales * green_house_price +
  yellow_house_thin_mints * yellow_house_thin_mints_price +
  yellow_house_fudge_delights * yellow_house_fudge_delights_price +
  brown_house_sales * brown_house_price

-- Theorem statement
theorem carmen_earnings : total_earnings = 42 := by
  sorry

end NUMINAMATH_CALUDE_carmen_earnings_l1996_199650


namespace NUMINAMATH_CALUDE_train_speed_in_km_hr_l1996_199625

-- Define the given parameters
def train_length : ℝ := 50
def platform_length : ℝ := 250
def crossing_time : ℝ := 15

-- Define the conversion factor from m/s to km/hr
def m_s_to_km_hr : ℝ := 3.6

-- Theorem statement
theorem train_speed_in_km_hr :
  let total_distance := train_length + platform_length
  let speed_m_s := total_distance / crossing_time
  speed_m_s * m_s_to_km_hr = 72 := by
  sorry


end NUMINAMATH_CALUDE_train_speed_in_km_hr_l1996_199625


namespace NUMINAMATH_CALUDE_sum_of_zeros_greater_than_one_l1996_199638

open Real

theorem sum_of_zeros_greater_than_one (a : ℝ) :
  let f := fun x : ℝ => log x - a * x + 1 / (2 * x)
  let g := fun x : ℝ => f x + a * (x - 1)
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → g x₁ = 0 → g x₂ = 0 → x₁ + x₂ > 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_zeros_greater_than_one_l1996_199638


namespace NUMINAMATH_CALUDE_distance_to_origin_l1996_199687

theorem distance_to_origin (z : ℂ) (h : z = 1 - 2*I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_origin_l1996_199687


namespace NUMINAMATH_CALUDE_residue_of_negative_thousand_mod_33_l1996_199680

theorem residue_of_negative_thousand_mod_33 :
  ∃ (k : ℤ), -1000 = 33 * k + 23 ∧ (0 ≤ 23 ∧ 23 < 33) := by
  sorry

end NUMINAMATH_CALUDE_residue_of_negative_thousand_mod_33_l1996_199680


namespace NUMINAMATH_CALUDE_allans_balloons_l1996_199698

theorem allans_balloons (total : ℕ) (jakes_balloons : ℕ) (h1 : total = 3) (h2 : jakes_balloons = 1) :
  total - jakes_balloons = 2 :=
by sorry

end NUMINAMATH_CALUDE_allans_balloons_l1996_199698


namespace NUMINAMATH_CALUDE_triangle_side_length_l1996_199624

/-- Given a triangle ABC with angle A = 30°, angle B = 105°, and side a = 4,
    prove that the length of side c is 4√2. -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  A = π/6 → B = 7*π/12 → a = 4 → 
  A + B + C = π → 
  a / Real.sin A = b / Real.sin B → 
  b / Real.sin B = c / Real.sin C → 
  c = 4 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1996_199624


namespace NUMINAMATH_CALUDE_division_problem_l1996_199670

theorem division_problem : (144 / 6) / 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1996_199670


namespace NUMINAMATH_CALUDE_linear_equations_compatibility_l1996_199629

theorem linear_equations_compatibility (a b c d : ℝ) :
  (∃ x : ℝ, a * x + b = 0 ∧ c * x + d = 0) ↔ a * d - b * c = 0 := by
  sorry

end NUMINAMATH_CALUDE_linear_equations_compatibility_l1996_199629


namespace NUMINAMATH_CALUDE_paperclip_capacity_l1996_199673

theorem paperclip_capacity (box_volume : ℝ) (box_capacity : ℕ) (cube_volume : ℝ) : 
  box_volume = 24 → 
  box_capacity = 75 → 
  cube_volume = 64 → 
  (cube_volume / box_volume * box_capacity : ℝ) = 200 := by
  sorry

end NUMINAMATH_CALUDE_paperclip_capacity_l1996_199673


namespace NUMINAMATH_CALUDE_inverse_proportion_ratio_l1996_199619

/-- Given that x is inversely proportional to y, this theorem proves that
    if x₁/x₂ = 3/4, then y₁/y₂ = 4/3, where y₁ and y₂ are the corresponding y values. -/
theorem inverse_proportion_ratio (x₁ x₂ y₁ y₂ : ℝ) (hx : x₁ ≠ 0 ∧ x₂ ≠ 0) (hy : y₁ ≠ 0 ∧ y₂ ≠ 0)
    (h_prop : ∃ k : ℝ, ∀ x y, x * y = k) (h_ratio : x₁ / x₂ = 3 / 4) :
    y₁ / y₂ = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_ratio_l1996_199619


namespace NUMINAMATH_CALUDE_mandy_school_ratio_l1996_199626

theorem mandy_school_ratio : 
  ∀ (researched applied accepted : ℕ),
    researched = 42 →
    accepted = 7 →
    2 * accepted = applied →
    (applied : ℚ) / researched = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_mandy_school_ratio_l1996_199626


namespace NUMINAMATH_CALUDE_all_lines_have_inclination_angle_not_necessarily_slope_l1996_199602

-- Define what a line is (this is a simplified representation)
structure Line where
  -- You might add more properties here in a real implementation
  dummy : Unit

-- Define the concept of an inclination angle
def has_inclination_angle (l : Line) : Prop := sorry

-- Define the concept of a slope
def has_slope (l : Line) : Prop := sorry

-- The theorem to prove
theorem all_lines_have_inclination_angle_not_necessarily_slope :
  (∀ l : Line, has_inclination_angle l) ∧
  (∃ l : Line, ¬ has_slope l) := by sorry

end NUMINAMATH_CALUDE_all_lines_have_inclination_angle_not_necessarily_slope_l1996_199602


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1996_199664

theorem absolute_value_inequality (x a : ℝ) (h1 : |x - 4| + |x - 3| < a) (h2 : a > 0) : a > 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1996_199664


namespace NUMINAMATH_CALUDE_no_integer_sqrt_representation_l1996_199674

theorem no_integer_sqrt_representation : ¬ ∃ (A B : ℤ), 99999 + 111111 * Real.sqrt 3 = (A + B * Real.sqrt 3) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_sqrt_representation_l1996_199674


namespace NUMINAMATH_CALUDE_largest_triangle_perimeter_l1996_199601

theorem largest_triangle_perimeter : 
  ∀ y : ℤ, 
  (y > 0) → 
  (7 + 9 > y) → 
  (7 + y > 9) → 
  (9 + y > 7) → 
  (∀ z : ℤ, (z > 0) → (7 + 9 > z) → (7 + z > 9) → (9 + z > 7) → (7 + 9 + y ≥ 7 + 9 + z)) →
  (7 + 9 + y = 31) :=
by sorry

end NUMINAMATH_CALUDE_largest_triangle_perimeter_l1996_199601


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1996_199668

-- Problem 1
theorem problem_1 (α : Real) (h : Real.tan (π/4 + α) = 2) :
  Real.sin (2*α) + Real.cos α ^ 2 = 3/2 := by sorry

-- Problem 2
theorem problem_2 (x₁ y₁ x₂ y₂ α : Real) 
  (h1 : x₁^2 + y₁^2 = 1) 
  (h2 : x₂^2 + y₂^2 = 1) 
  (h3 : Real.sin α + Real.cos α = 7/17) 
  (h4 : 0 < α) (h5 : α < π) :
  x₁*x₂ + y₁*y₂ = -8/17 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1996_199668


namespace NUMINAMATH_CALUDE_constant_remainder_implies_b_value_l1996_199675

/-- The dividend polynomial -/
def dividend (b x : ℚ) : ℚ := 12 * x^4 - 14 * x^3 + b * x^2 + 7 * x + 9

/-- The divisor polynomial -/
def divisor (x : ℚ) : ℚ := 3 * x^2 - 4 * x + 2

/-- The remainder polynomial -/
def remainder (b x : ℚ) : ℚ := dividend b x - divisor x * (4 * x^2 + 2/3 * x)

theorem constant_remainder_implies_b_value :
  (∃ (r : ℚ), ∀ (x : ℚ), remainder b x = r) ↔ b = 16/3 := by sorry

end NUMINAMATH_CALUDE_constant_remainder_implies_b_value_l1996_199675


namespace NUMINAMATH_CALUDE_subset_range_m_l1996_199686

theorem subset_range_m (m : ℝ) : 
  let A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 7}
  let B : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}
  B ⊆ A → m ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_subset_range_m_l1996_199686


namespace NUMINAMATH_CALUDE_upstream_speed_calculation_l1996_199623

/-- Calculates the upstream speed of a person given their downstream speed and the stream speed. -/
def upstreamSpeed (downstreamSpeed streamSpeed : ℝ) : ℝ :=
  downstreamSpeed - 2 * streamSpeed

/-- Theorem: Given a downstream speed of 12 km/h and a stream speed of 2 km/h, the upstream speed is 8 km/h. -/
theorem upstream_speed_calculation :
  upstreamSpeed 12 2 = 8 := by
  sorry

#eval upstreamSpeed 12 2

end NUMINAMATH_CALUDE_upstream_speed_calculation_l1996_199623


namespace NUMINAMATH_CALUDE_sin_600_plus_tan_240_l1996_199621

theorem sin_600_plus_tan_240 : Real.sin (600 * π / 180) + Real.tan (240 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_600_plus_tan_240_l1996_199621


namespace NUMINAMATH_CALUDE_motorcycles_sold_is_eight_l1996_199651

/-- Represents the monthly production and sales data for a vehicle factory -/
structure VehicleProduction where
  car_material_cost : ℕ
  cars_produced : ℕ
  car_price : ℕ
  motorcycle_material_cost : ℕ
  motorcycle_price : ℕ
  profit_increase : ℕ

/-- Calculates the number of motorcycles sold per month -/
def motorcycles_sold (data : VehicleProduction) : ℕ :=
  sorry

/-- Theorem stating that the number of motorcycles sold is 8 -/
theorem motorcycles_sold_is_eight (data : VehicleProduction) 
  (h1 : data.car_material_cost = 100)
  (h2 : data.cars_produced = 4)
  (h3 : data.car_price = 50)
  (h4 : data.motorcycle_material_cost = 250)
  (h5 : data.motorcycle_price = 50)
  (h6 : data.profit_increase = 50) :
  motorcycles_sold data = 8 := by
  sorry

end NUMINAMATH_CALUDE_motorcycles_sold_is_eight_l1996_199651


namespace NUMINAMATH_CALUDE_sum_of_f_values_l1996_199615

noncomputable def f (x : ℝ) : ℝ := x^2 / (1 + x^2)

theorem sum_of_f_values : 
  f 1 + f 2 + f (1/2) + f 3 + f (1/3) + f 4 + f (1/4) = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_f_values_l1996_199615


namespace NUMINAMATH_CALUDE_vectors_parallel_opposite_l1996_199653

/-- Given vectors a = (-1, 2) and b = (2, -4), prove that they are parallel and in opposite directions. -/
theorem vectors_parallel_opposite (a b : ℝ × ℝ) : 
  a = (-1, 2) → b = (2, -4) → ∃ k : ℝ, k < 0 ∧ b = k • a := by sorry

end NUMINAMATH_CALUDE_vectors_parallel_opposite_l1996_199653


namespace NUMINAMATH_CALUDE_fifth_term_ratio_l1996_199696

/-- Two arithmetic sequences and their sum ratios -/
structure ArithmeticSequences where
  a : ℕ → ℝ  -- First arithmetic sequence
  b : ℕ → ℝ  -- Second arithmetic sequence
  S : ℕ → ℝ  -- Sum of first n terms of sequence a
  T : ℕ → ℝ  -- Sum of first n terms of sequence b
  sum_ratio : ∀ n : ℕ, S n / T n = (2 * n - 3) / (3 * n - 2)

/-- The ratio of the 5th terms of the sequences is 3/5 -/
theorem fifth_term_ratio (seq : ArithmeticSequences) : seq.a 5 / seq.b 5 = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_ratio_l1996_199696


namespace NUMINAMATH_CALUDE_bud_uncle_age_ratio_l1996_199648

/-- The ratio of Bud's age to his uncle's age -/
def age_ratio (bud_age uncle_age : ℕ) : ℚ :=
  bud_age / uncle_age

/-- Bud's age -/
def bud_age : ℕ := 8

/-- Bud's uncle's age -/
def uncle_age : ℕ := 24

theorem bud_uncle_age_ratio :
  age_ratio bud_age uncle_age = 1 / 3 := by
  sorry


end NUMINAMATH_CALUDE_bud_uncle_age_ratio_l1996_199648


namespace NUMINAMATH_CALUDE_smallest_number_with_properties_l1996_199610

/-- Given a prime number p where 2p+1 is a cube of a natural number,
    find the smallest natural number N that is divisible by p,
    ends with p, and has a digit sum equal to p. -/
theorem smallest_number_with_properties (p : ℕ) (h_prime : Nat.Prime p)
  (h_cube : ∃ n : ℕ, 2 * p + 1 = n^3) :
  let N := 11713
  (N % p = 0) ∧
  (N % 100 = p) ∧
  (Nat.digits 10 N).sum = p ∧
  (∀ m : ℕ, m < N →
    (m % p = 0) ∧ (m % 100 = p) ∧ (Nat.digits 10 m).sum = p → False) ∧
  (p = 13) := by
sorry

end NUMINAMATH_CALUDE_smallest_number_with_properties_l1996_199610


namespace NUMINAMATH_CALUDE_square_of_1023_l1996_199600

theorem square_of_1023 : (1023 : ℕ)^2 = 1046529 := by
  sorry

end NUMINAMATH_CALUDE_square_of_1023_l1996_199600


namespace NUMINAMATH_CALUDE_min_value_expression_l1996_199643

theorem min_value_expression (a b c : ℝ) (h1 : 2 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ 3) :
  (a - 2)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (3/c - 1)^2 ≥ 4 * (9^(1/4) - 5/4)^2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1996_199643


namespace NUMINAMATH_CALUDE_jan_skips_proof_l1996_199676

def initial_speed : ℕ := 70
def time_period : ℕ := 5

theorem jan_skips_proof (doubled_speed : ℕ) (total_skips : ℕ) 
  (h1 : doubled_speed = 2 * initial_speed) 
  (h2 : total_skips = doubled_speed * time_period) : 
  total_skips = 700 := by sorry

end NUMINAMATH_CALUDE_jan_skips_proof_l1996_199676


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l1996_199611

theorem inequality_system_solution_set :
  ∀ a : ℝ, (2 * a - 3 < 0 ∧ 1 - a < 0) ↔ (1 < a ∧ a < 3/2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l1996_199611


namespace NUMINAMATH_CALUDE_first_project_depth_l1996_199652

-- Define the parameters for the first digging project
def length1 : ℝ := 25
def breadth1 : ℝ := 30
def days1 : ℝ := 12

-- Define the parameters for the second digging project
def length2 : ℝ := 20
def breadth2 : ℝ := 50
def depth2 : ℝ := 75
def days2 : ℝ := 12

-- Define the function to calculate volume
def volume (length : ℝ) (breadth : ℝ) (depth : ℝ) : ℝ :=
  length * breadth * depth

-- Theorem statement
theorem first_project_depth :
  ∃ (depth1 : ℝ),
    volume length1 breadth1 depth1 = volume length2 breadth2 depth2 ∧
    depth1 = 100 := by
  sorry

end NUMINAMATH_CALUDE_first_project_depth_l1996_199652


namespace NUMINAMATH_CALUDE_railway_ticket_types_l1996_199661

/-- The number of stations on the railway --/
def num_stations : ℕ := 25

/-- The number of different types of tickets needed for a railway with n stations --/
def num_ticket_types (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: The number of different types of tickets needed for a railway with 25 stations is 300 --/
theorem railway_ticket_types : num_ticket_types num_stations = 300 := by
  sorry

end NUMINAMATH_CALUDE_railway_ticket_types_l1996_199661


namespace NUMINAMATH_CALUDE_stating_circle_symmetry_l1996_199694

/-- Given circle -/
def given_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 6*y + 9 = 0

/-- Line of symmetry -/
def symmetry_line (x y : ℝ) : Prop :=
  2*x + y + 5 = 0

/-- Symmetric circle -/
def symmetric_circle (x y : ℝ) : Prop :=
  (x + 7)^2 + (y + 1)^2 = 1

/-- 
Theorem stating that the symmetric_circle is indeed symmetric to the given_circle
with respect to the symmetry_line
-/
theorem circle_symmetry :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
  given_circle x₁ y₁ →
  symmetric_circle x₂ y₂ →
  (∃ (x_mid y_mid : ℝ),
    symmetry_line x_mid y_mid ∧
    x_mid = (x₁ + x₂) / 2 ∧
    y_mid = (y₁ + y₂) / 2) :=
sorry

end NUMINAMATH_CALUDE_stating_circle_symmetry_l1996_199694


namespace NUMINAMATH_CALUDE_train_length_problem_l1996_199679

/-- The length of two trains passing each other --/
theorem train_length_problem (speed1 speed2 : ℝ) (passing_time : ℝ) (h1 : speed1 = 65) (h2 : speed2 = 50) (h3 : passing_time = 11.895652173913044) :
  let relative_speed := (speed1 + speed2) * (1000 / 3600)
  let total_distance := relative_speed * passing_time
  let train_length := total_distance / 2
  train_length = 190 := by sorry

end NUMINAMATH_CALUDE_train_length_problem_l1996_199679


namespace NUMINAMATH_CALUDE_cafeteria_apples_l1996_199630

/-- Calculates the number of apples bought by the cafeteria -/
def apples_bought (initial : ℕ) (used : ℕ) (final : ℕ) : ℕ :=
  final - (initial - used)

/-- Proves that the cafeteria bought 6 apples -/
theorem cafeteria_apples : apples_bought 23 20 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_apples_l1996_199630


namespace NUMINAMATH_CALUDE_correct_number_of_choices_l1996_199632

/-- Represents a team in the club -/
inductive Team
| A
| B

/-- Represents the gender of a club member -/
inductive Gender
| Boy
| Girl

/-- Represents the composition of a team -/
structure TeamComposition :=
  (boys : ℕ)
  (girls : ℕ)

/-- The total number of members in the club -/
def totalMembers : ℕ := 24

/-- The number of boys in the club -/
def totalBoys : ℕ := 14

/-- The number of girls in the club -/
def totalGirls : ℕ := 10

/-- The composition of Team A -/
def teamA : TeamComposition := ⟨8, 6⟩

/-- The composition of Team B -/
def teamB : TeamComposition := ⟨6, 4⟩

/-- Returns the number of ways to choose a president and vice-president -/
def chooseLeaders : ℕ := sorry

/-- Theorem stating that the number of ways to choose a president and vice-president
    of different genders and from different teams is 136 -/
theorem correct_number_of_choices :
  chooseLeaders = 136 := by sorry

end NUMINAMATH_CALUDE_correct_number_of_choices_l1996_199632


namespace NUMINAMATH_CALUDE_bottom_price_is_3350_l1996_199622

/-- The price of a bottom pajama in won -/
def bottom_price : ℕ := sorry

/-- The price of a top pajama in won -/
def top_price : ℕ := sorry

/-- The number of pajama sets bought -/
def num_sets : ℕ := 3

/-- The total amount paid in won -/
def total_paid : ℕ := 21000

/-- The price difference between top and bottom in won -/
def price_difference : ℕ := 300

theorem bottom_price_is_3350 : 
  bottom_price = 3350 ∧ 
  top_price = bottom_price + price_difference ∧
  num_sets * (bottom_price + top_price) = total_paid := by
  sorry

end NUMINAMATH_CALUDE_bottom_price_is_3350_l1996_199622


namespace NUMINAMATH_CALUDE_symmetry_implies_m_sqrt3_l1996_199618

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line mx - y + 1 = 0 -/
def line_of_symmetry (m : ℝ) (p : Point) : Prop :=
  m * p.x - p.y + 1 = 0

/-- The line x + y = 0 -/
def line_xy (p : Point) : Prop :=
  p.x + p.y = 0

/-- Two points are symmetric with respect to a line -/
def symmetric_points (m : ℝ) (p q : Point) : Prop :=
  ∃ (mid : Point), line_of_symmetry m mid ∧
    mid.x = (p.x + q.x) / 2 ∧
    mid.y = (p.y + q.y) / 2

theorem symmetry_implies_m_sqrt3 :
  ∀ (m : ℝ) (N : Point),
    symmetric_points m (Point.mk 1 0) N →
    line_xy N →
    m = Real.sqrt 3 ∨ m = -Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_symmetry_implies_m_sqrt3_l1996_199618


namespace NUMINAMATH_CALUDE_polygon_triangulation_l1996_199683

/-- Given an n-sided polygon divided into triangles by non-intersecting diagonals,
    this theorem states that the number of triangles with exactly two sides
    as edges of the original polygon is at least 2. -/
theorem polygon_triangulation (n : ℕ) (h : n ≥ 3) :
  ∃ (k₀ k₁ k₂ : ℕ),
    k₀ + k₁ + k₂ = n - 2 ∧
    k₁ + 2 * k₂ = n ∧
    k₂ ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_polygon_triangulation_l1996_199683


namespace NUMINAMATH_CALUDE_unique_number_property_l1996_199657

theorem unique_number_property : ∃! x : ℝ, x / 3 = x - 5 := by sorry

end NUMINAMATH_CALUDE_unique_number_property_l1996_199657


namespace NUMINAMATH_CALUDE_geometric_sequence_reciprocal_sum_l1996_199695

/-- Given a geometric sequence {a_n} with a₁ = 2 and a₁ + a₃ + a₅ = 14,
    prove that 1/a₁ + 1/a₃ + 1/a₅ = 7/8 -/
theorem geometric_sequence_reciprocal_sum (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  a 1 = 2 →
  a 1 + a 3 + a 5 = 14 →
  1 / a 1 + 1 / a 3 + 1 / a 5 = 7 / 8 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_reciprocal_sum_l1996_199695


namespace NUMINAMATH_CALUDE_deposit_ratio_is_one_fifth_l1996_199649

/-- Represents the financial transaction of Lulu --/
structure LuluFinances where
  initial_amount : ℕ
  ice_cream_cost : ℕ
  cash_left : ℕ

/-- Calculates the ratio of deposited money to money left after buying the t-shirt --/
def deposit_ratio (finances : LuluFinances) : Rat :=
  let after_ice_cream := finances.initial_amount - finances.ice_cream_cost
  let after_tshirt := after_ice_cream / 2
  let deposited := after_tshirt - finances.cash_left
  deposited / after_tshirt

/-- Theorem stating that the deposit ratio is 1:5 given the initial conditions --/
theorem deposit_ratio_is_one_fifth (finances : LuluFinances) 
  (h1 : finances.initial_amount = 65)
  (h2 : finances.ice_cream_cost = 5)
  (h3 : finances.cash_left = 24) : 
  deposit_ratio finances = 1 / 5 := by
  sorry

#eval deposit_ratio ⟨65, 5, 24⟩

end NUMINAMATH_CALUDE_deposit_ratio_is_one_fifth_l1996_199649


namespace NUMINAMATH_CALUDE_unique_paintable_number_l1996_199644

def isPaintable (s b a : ℕ+) : Prop :=
  -- Sarah's sequence doesn't overlap with Bob's or Alice's
  ∀ k l : ℕ, k * s.val ≠ l * b.val ∧ k * s.val ≠ 4 + l * a.val
  -- Bob's sequence doesn't overlap with Sarah's or Alice's
  ∧ ∀ k l : ℕ, 2 + k * b.val ≠ l * s.val ∧ 2 + k * b.val ≠ 4 + l * a.val
  -- Alice's sequence doesn't overlap with Sarah's or Bob's
  ∧ ∀ k l : ℕ, 4 + k * a.val ≠ l * s.val ∧ 4 + k * a.val ≠ 2 + l * b.val
  -- Every picket is painted
  ∧ ∀ n : ℕ, n > 0 → (∃ k : ℕ, n = k * s.val ∨ n = 2 + k * b.val ∨ n = 4 + k * a.val)

theorem unique_paintable_number :
  ∃! n : ℕ, ∃ s b a : ℕ+, isPaintable s b a ∧ n = 1000 * s.val + 100 * b.val + 10 * a.val :=
by sorry

end NUMINAMATH_CALUDE_unique_paintable_number_l1996_199644


namespace NUMINAMATH_CALUDE_set_equality_implies_y_zero_l1996_199690

theorem set_equality_implies_y_zero (x y : ℝ) :
  ({0, 1, x} : Set ℝ) = {x^2, y, -1} → y = 0 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_y_zero_l1996_199690


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2017_l1996_199635

/-- An arithmetic sequence satisfying the given condition -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, a n + 2 * a (n + 1) + 3 * a (n + 2) = 6 * n + 22

/-- The 2017th term of the arithmetic sequence is 6058/3 -/
theorem arithmetic_sequence_2017 (a : ℕ → ℚ) (h : ArithmeticSequence a) :
  a 2017 = 6058 / 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2017_l1996_199635


namespace NUMINAMATH_CALUDE_function_zero_point_l1996_199691

theorem function_zero_point
  (f : ℝ → ℝ)
  (h_mono : Monotone f)
  (h_prop : ∀ x : ℝ, f (f x - 2^x) = -1/2) :
  ∃! x : ℝ, f x = 0 ∧ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_zero_point_l1996_199691


namespace NUMINAMATH_CALUDE_compute_m_3v_minus_2w_l1996_199667

variable (M : Matrix (Fin 2) (Fin 2) ℝ)
variable (v w : Fin 2 → ℝ)

def Mv : Fin 2 → ℝ := ![3, -1]
def Mw : Fin 2 → ℝ := ![4, 3]

axiom mv_eq : M.mulVec v = Mv
axiom mw_eq : M.mulVec w = Mw

theorem compute_m_3v_minus_2w : M.mulVec (3 • v - 2 • w) = ![1, -9] := by sorry

end NUMINAMATH_CALUDE_compute_m_3v_minus_2w_l1996_199667


namespace NUMINAMATH_CALUDE_pet_store_cages_l1996_199692

/-- Given a pet store with an initial number of puppies, some sold, and a fixed number per cage,
    calculate the number of cages needed for the remaining puppies. -/
theorem pet_store_cages (initial_puppies : ℕ) (sold_puppies : ℕ) (puppies_per_cage : ℕ) 
    (h1 : initial_puppies = 102)
    (h2 : sold_puppies = 21)
    (h3 : puppies_per_cage = 9)
    (h4 : sold_puppies < initial_puppies) :
  (initial_puppies - sold_puppies) / puppies_per_cage = 9 :=
by sorry

end NUMINAMATH_CALUDE_pet_store_cages_l1996_199692
