import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_squares_residuals_l2253_225315

/-- Linear Regression Sum of Squares -/
structure LinearRegressionSS where
  SST : ℝ  -- Total sum of squares
  SSR : ℝ  -- Sum of squares due to regression
  SSE : ℝ  -- Sum of squares for residuals

/-- Theorem: Sum of Squares for Residuals in Linear Regression -/
theorem sum_of_squares_residuals 
  (lr : LinearRegressionSS) 
  (h1 : lr.SST = 13) 
  (h2 : lr.SSR = 10) 
  (h3 : lr.SST = lr.SSR + lr.SSE) : 
  lr.SSE = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_residuals_l2253_225315


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l2253_225354

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  d_ne_zero : d ≠ 0
  is_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d
  is_geometric : (a 3) ^ 2 = a 1 * a 9

/-- The main theorem -/
theorem arithmetic_geometric_ratio (seq : ArithmeticSequence) :
  (seq.a 2 + seq.a 4 + seq.a 10) / (seq.a 1 + seq.a 3 + seq.a 9) = 16 / 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l2253_225354


namespace NUMINAMATH_CALUDE_max_pogs_purchase_l2253_225304

theorem max_pogs_purchase (x y z : ℕ) : 
  x ≥ 1 → y ≥ 1 → z ≥ 1 →
  3 * x + 4 * y + 9 * z = 75 →
  z ≤ 7 :=
sorry

end NUMINAMATH_CALUDE_max_pogs_purchase_l2253_225304


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l2253_225310

/-- The surface area of a cylinder given its unfolded lateral surface dimensions -/
theorem cylinder_surface_area (h w : ℝ) (h_pos : h > 0) (w_pos : w > 0) 
  (h_eq : h = 6 * Real.pi) (w_eq : w = 4 * Real.pi) :
  ∃ (r : ℝ), (r = 3 ∨ r = 2) ∧ 
    (2 * Real.pi * r * h + 2 * Real.pi * r^2 = 24 * Real.pi^2 + 18 * Real.pi ∨
     2 * Real.pi * r * h + 2 * Real.pi * r^2 = 24 * Real.pi^2 + 8 * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l2253_225310


namespace NUMINAMATH_CALUDE_equation_solutions_l2253_225360

theorem equation_solutions : 
  let f (x : ℂ) := (x - 2)^4 + (x - 6)^4 + 16
  ∀ x : ℂ, f x = 0 ↔ 
    x = 4 + 2*I*Real.sqrt 3 ∨ 
    x = 4 - 2*I*Real.sqrt 3 ∨ 
    x = 4 + I*Real.sqrt 2 ∨ 
    x = 4 - I*Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2253_225360


namespace NUMINAMATH_CALUDE_square_roots_of_nine_l2253_225320

theorem square_roots_of_nine :
  ∀ y : ℝ, y^2 = 9 ↔ y = 3 ∨ y = -3 :=
by sorry

end NUMINAMATH_CALUDE_square_roots_of_nine_l2253_225320


namespace NUMINAMATH_CALUDE_f_f_eq_f_solution_l2253_225399

def f (x : ℝ) : ℝ := x^2 - 5*x + 6

theorem f_f_eq_f_solution :
  ∀ x : ℝ, f (f x) = f x ↔ x = 0 ∨ x = 2 ∨ x = 3 ∨ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_f_f_eq_f_solution_l2253_225399


namespace NUMINAMATH_CALUDE_expression_equals_36_75_l2253_225322

-- Define a function to convert a number from any base to base 10
def toBase10 (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base ^ i) 0

-- Define the expression
def expression : ℚ :=
  (toBase10 [3, 4, 6] 8 : ℚ) / (toBase10 [1, 5] 3) +
  (toBase10 [2, 0, 4] 5 : ℚ) / (toBase10 [1, 2] 4) - 1

-- State the theorem
theorem expression_equals_36_75 : expression = 36.75 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_36_75_l2253_225322


namespace NUMINAMATH_CALUDE_fabric_sales_fraction_l2253_225306

theorem fabric_sales_fraction (total_sales stationery_sales : ℕ) 
  (h1 : total_sales = 36)
  (h2 : stationery_sales = 15)
  (h3 : ∃ jewelry_sales : ℕ, jewelry_sales = total_sales / 4)
  (h4 : ∃ fabric_sales : ℕ, fabric_sales + total_sales / 4 + stationery_sales = total_sales) :
  ∃ fabric_sales : ℕ, (fabric_sales : ℚ) / total_sales = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fabric_sales_fraction_l2253_225306


namespace NUMINAMATH_CALUDE_min_value_theorem_l2253_225349

theorem min_value_theorem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x > y) (h4 : x + 2*y = 3) :
  ∃ (min_val : ℝ), min_val = 8/3 ∧ 
  ∀ (z : ℝ), z = (1 / (x - y)) + (9 / (x + 5*y)) → z ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2253_225349


namespace NUMINAMATH_CALUDE_binary_253_ones_minus_zeros_l2253_225326

/-- The binary representation of a natural number -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec toBinaryAux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinaryAux (m / 2)
  toBinaryAux n

/-- Count the number of true values in a list of booleans -/
def countTrue (l : List Bool) : ℕ :=
  l.foldl (fun acc b => if b then acc + 1 else acc) 0

/-- Count the number of false values in a list of booleans -/
def countFalse (l : List Bool) : ℕ :=
  l.length - countTrue l

theorem binary_253_ones_minus_zeros : 
  let binary := toBinary 253
  let ones := countTrue binary
  let zeros := countFalse binary
  ones - zeros = 6 := by sorry

end NUMINAMATH_CALUDE_binary_253_ones_minus_zeros_l2253_225326


namespace NUMINAMATH_CALUDE_island_not_mayya_l2253_225346

structure Inhabitant where
  name : String
  isKnight : Bool

def statement (a b : Inhabitant) (islandName : String) : Prop :=
  (a.name = "A" ∧ b.name = "B" ∧ b.isKnight ∧ islandName = "Mayya") ∨
  (a.name = "B" ∧ b.name = "A" ∧ ¬b.isKnight ∧ islandName = "Mayya")

theorem island_not_mayya (a b : Inhabitant) (islandName : String) :
  statement a b islandName → islandName ≠ "Mayya" :=
by
  sorry


end NUMINAMATH_CALUDE_island_not_mayya_l2253_225346


namespace NUMINAMATH_CALUDE_shooting_sequences_l2253_225330

theorem shooting_sequences (n : Nat) (c₁ c₂ c₃ : Nat) 
  (h₁ : n = c₁ + c₂ + c₃) 
  (h₂ : c₁ = 3) 
  (h₃ : c₂ = 2) 
  (h₄ : c₃ = 3) :
  (Nat.factorial n) / (Nat.factorial c₁ * Nat.factorial c₂ * Nat.factorial c₃) = 560 :=
by sorry

end NUMINAMATH_CALUDE_shooting_sequences_l2253_225330


namespace NUMINAMATH_CALUDE_intersection_empty_union_equals_B_l2253_225334

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2*a + 1}
def B : Set ℝ := {x | 0 < x ∧ x < 1}

-- Theorem for the first part
theorem intersection_empty (a : ℝ) :
  A a ∩ B = ∅ ↔ a ≥ 2 ∨ a ≤ -1/2 :=
sorry

-- Theorem for the second part
theorem union_equals_B (a : ℝ) :
  A a ∪ B = B ↔ a ≤ -2 :=
sorry

end NUMINAMATH_CALUDE_intersection_empty_union_equals_B_l2253_225334


namespace NUMINAMATH_CALUDE_not_necessary_not_sufficient_l2253_225317

theorem not_necessary_not_sufficient (a : ℝ) : 
  ¬(∀ a, a < 2 → a^2 < 2*a) ∧ ¬(∀ a, a^2 < 2*a → a < 2) := by
  sorry

end NUMINAMATH_CALUDE_not_necessary_not_sufficient_l2253_225317


namespace NUMINAMATH_CALUDE_solution_characterization_l2253_225312

def system_equations (x₁ x₂ x₃ x₄ x₅ y : ℝ) : Prop :=
  x₅ + x₂ = y * x₁ ∧
  x₁ + x₃ = y * x₂ ∧
  x₂ + x₄ = y * x₃ ∧
  x₃ + x₅ = y * x₄ ∧
  x₄ + x₁ = y * x₅

theorem solution_characterization :
  ∀ y : ℝ,
  (∀ x₁ x₂ x₃ x₄ x₅ : ℝ, system_equations x₁ x₂ x₃ x₄ x₅ y →
    ((y ≠ 2 ∧ y^2 + y - 1 ≠ 0 → x₁ = 0 ∧ x₂ = 0 ∧ x₃ = 0 ∧ x₄ = 0 ∧ x₅ = 0) ∧
     (y = 2 → ∃ u : ℝ, x₁ = u ∧ x₂ = u ∧ x₃ = u ∧ x₄ = u ∧ x₅ = u) ∧
     (y^2 + y - 1 = 0 →
       ∃ u v : ℝ, x₁ = u ∧ x₂ = v ∧ x₃ = -u + y*v ∧ x₄ = -y*(u + v) ∧ x₅ = y*u - v))) :=
by sorry


end NUMINAMATH_CALUDE_solution_characterization_l2253_225312


namespace NUMINAMATH_CALUDE_gcd_30_and_number_l2253_225380

theorem gcd_30_and_number (n : ℕ) : 
  70 ≤ n ∧ n ≤ 90 ∧ Nat.gcd 30 n = 10 → n = 70 ∨ n = 80 ∨ n = 90 := by
  sorry

end NUMINAMATH_CALUDE_gcd_30_and_number_l2253_225380


namespace NUMINAMATH_CALUDE_inscribed_circle_in_tangent_quadrilateral_l2253_225394

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a rectangle with four vertices -/
structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- Represents a quadrilateral formed by tangent lines -/
structure TangentQuadrilateral where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  S : ℝ × ℝ

/-- Function to check if two circles intersect -/
def circlesIntersect (c1 c2 : Circle) : Prop := sorry

/-- Function to check if a quadrilateral is tangential (can have an inscribed circle) -/
def isTangentialQuadrilateral (quad : TangentQuadrilateral) : Prop := sorry

/-- Main theorem statement -/
theorem inscribed_circle_in_tangent_quadrilateral 
  (rect : Rectangle) 
  (circleA circleB circleC circleD : Circle)
  (quad : TangentQuadrilateral) :
  circleA.center = rect.A ∧
  circleB.center = rect.B ∧
  circleC.center = rect.C ∧
  circleD.center = rect.D ∧
  ¬(circlesIntersect circleA circleB) ∧
  ¬(circlesIntersect circleA circleC) ∧
  ¬(circlesIntersect circleA circleD) ∧
  ¬(circlesIntersect circleB circleC) ∧
  ¬(circlesIntersect circleB circleD) ∧
  ¬(circlesIntersect circleC circleD) ∧
  circleA.radius + circleC.radius = circleB.radius + circleD.radius →
  isTangentialQuadrilateral quad :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_in_tangent_quadrilateral_l2253_225394


namespace NUMINAMATH_CALUDE_bob_initial_nickels_l2253_225363

theorem bob_initial_nickels (a b : ℕ) 
  (h1 : b + 1 = 4 * (a - 1)) 
  (h2 : b - 1 = 3 * (a + 1)) : 
  b = 31 := by sorry

end NUMINAMATH_CALUDE_bob_initial_nickels_l2253_225363


namespace NUMINAMATH_CALUDE_eleventh_tenth_square_difference_l2253_225344

/-- The side length of the nth square in the sequence -/
def squareSideLength (n : ℕ) : ℕ := 3 + 2 * (n - 1)

/-- The number of tiles in the nth square -/
def squareTiles (n : ℕ) : ℕ := (squareSideLength n) ^ 2

/-- The difference in tiles between the nth and (n-1)th squares -/
def tileDifference (n : ℕ) : ℕ := squareTiles n - squareTiles (n - 1)

theorem eleventh_tenth_square_difference :
  tileDifference 11 = 88 := by sorry

end NUMINAMATH_CALUDE_eleventh_tenth_square_difference_l2253_225344


namespace NUMINAMATH_CALUDE_james_dance_duration_l2253_225376

/-- Represents the number of calories burned per hour while walking -/
def calories_walking : ℕ := 300

/-- Represents the number of calories burned per week from dancing -/
def calories_dancing_weekly : ℕ := 2400

/-- Represents the number of times James dances per week -/
def dance_sessions_per_week : ℕ := 4

/-- Represents the ratio of calories burned dancing compared to walking -/
def dancing_to_walking_ratio : ℕ := 2

/-- Proves that James dances for 1 hour each time given the conditions -/
theorem james_dance_duration :
  (calories_dancing_weekly / (dancing_to_walking_ratio * calories_walking)) / dance_sessions_per_week = 1 :=
by sorry

end NUMINAMATH_CALUDE_james_dance_duration_l2253_225376


namespace NUMINAMATH_CALUDE_no_solution_iff_n_eq_neg_one_l2253_225388

/-- A system of equations parameterized by n -/
def system (n : ℝ) (x y z : ℝ) : Prop :=
  n * x + y = 2 ∧ n * y + z = 2 ∧ x + n^2 * z = 2

/-- The system has no solution if and only if n = -1 -/
theorem no_solution_iff_n_eq_neg_one :
  ∀ n : ℝ, (∀ x y z : ℝ, ¬system n x y z) ↔ n = -1 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_n_eq_neg_one_l2253_225388


namespace NUMINAMATH_CALUDE_hyperbola_equation_from_properties_l2253_225337

/-- Represents a hyperbola -/
structure Hyperbola where
  center : ℝ × ℝ
  focal_length : ℝ
  directrix : ℝ

/-- The equation of a hyperbola given its properties -/
def hyperbola_equation (h : Hyperbola) : ℝ → ℝ → Prop :=
  fun x y => 2 * x^2 - 2 * y^2 = 1

/-- Theorem: Given a hyperbola with center at the origin, focal length 2, 
    and one directrix at x = -1/2, its equation is 2x^2 - 2y^2 = 1 -/
theorem hyperbola_equation_from_properties 
  (h : Hyperbola) 
  (h_center : h.center = (0, 0))
  (h_focal_length : h.focal_length = 2)
  (h_directrix : h.directrix = -1/2) :
  ∀ x y, hyperbola_equation h x y ↔ 2 * x^2 - 2 * y^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_from_properties_l2253_225337


namespace NUMINAMATH_CALUDE_min_value_expression_l2253_225341

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_xyz : x * y * z = 2/3) : 
  x^2 + 6*x*y + 18*y^2 + 12*y*z + 4*z^2 ≥ 18 := by
sorry

end NUMINAMATH_CALUDE_min_value_expression_l2253_225341


namespace NUMINAMATH_CALUDE_sum_of_digits_2003n_l2253_225384

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem statement -/
theorem sum_of_digits_2003n (n : ℕ) 
  (h_pos : n > 0)
  (h_sum_n : sum_of_digits n = 111)
  (h_sum_7002n : sum_of_digits (7002 * n) = 990) : 
  sum_of_digits (2003 * n) = 555 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_2003n_l2253_225384


namespace NUMINAMATH_CALUDE_a_eq_one_sufficient_not_necessary_l2253_225324

def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

theorem a_eq_one_sufficient_not_necessary :
  (∀ a : ℝ, (a = 1 → ∀ x y : ℝ, 1 ≤ x → x ≤ y → f a x ≤ f a y)) ∧
  (∃ a : ℝ, a ≠ 1 ∧ ∀ x y : ℝ, 1 ≤ x → x ≤ y → f a x ≤ f a y) :=
sorry

end NUMINAMATH_CALUDE_a_eq_one_sufficient_not_necessary_l2253_225324


namespace NUMINAMATH_CALUDE_sum_at_two_and_neg_two_l2253_225377

/-- A cubic polynomial Q with specific properties -/
structure CubicPolynomial (p : ℝ) where
  Q : ℝ → ℝ
  is_cubic : ∃ (a b c : ℝ), ∀ x, Q x = a * x^3 + b * x^2 + c * x + p
  at_zero : Q 0 = p
  at_one : Q 1 = 3 * p
  at_neg_one : Q (-1) = 4 * p

/-- The sum of Q(2) and Q(-2) for a specific cubic polynomial Q -/
theorem sum_at_two_and_neg_two (p : ℝ) (Q : CubicPolynomial p) :
  Q.Q 2 + Q.Q (-2) = 22 * p := by
  sorry

end NUMINAMATH_CALUDE_sum_at_two_and_neg_two_l2253_225377


namespace NUMINAMATH_CALUDE_sum_10_is_negative_15_l2253_225343

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_def : ∀ n, S n = (n : ℝ) * a 1 + (n * (n - 1) / 2 : ℝ) * (a 2 - a 1)
  S_3 : S 3 = 6
  S_6 : S 6 = 3

/-- The sum of the first 10 terms is -15 -/
theorem sum_10_is_negative_15 (seq : ArithmeticSequence) : seq.S 10 = -15 := by
  sorry

end NUMINAMATH_CALUDE_sum_10_is_negative_15_l2253_225343


namespace NUMINAMATH_CALUDE_expression_evaluation_l2253_225351

theorem expression_evaluation (a : ℝ) 
  (h1 : a ≥ 0) 
  (h2 : a ≠ 1) 
  (h3 : a ≠ 1 + Real.sqrt 2) 
  (h4 : a ≠ 1 - Real.sqrt 2) :
  (1 + 2 * a^(1/4) - a^(1/2)) / (1 - a + 4 * a^(3/4) - 4 * a^(1/2)) +
  (a^(1/4) - 2) / ((a^(1/4) - 1)^2) = 1 / (a^(1/4) - 1) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2253_225351


namespace NUMINAMATH_CALUDE_point_product_l2253_225395

theorem point_product (y₁ y₂ : ℝ) : 
  ((-4 - 7)^2 + (y₁ - 3)^2 = 13^2) →
  ((-4 - 7)^2 + (y₂ - 3)^2 = 13^2) →
  y₁ ≠ y₂ →
  y₁ * y₂ = -39 := by
sorry

end NUMINAMATH_CALUDE_point_product_l2253_225395


namespace NUMINAMATH_CALUDE_partition_contains_perfect_square_sum_l2253_225367

theorem partition_contains_perfect_square_sum (n : ℕ) (h : n ≥ 15) :
  ∀ (A B : Set ℕ), (A ∪ B = Finset.range n.succ) → (A ∩ B = ∅) →
  (∃ (x y : ℕ), x ≠ y ∧ ((x ∈ A ∧ y ∈ A) ∨ (x ∈ B ∧ y ∈ B)) ∧ ∃ (z : ℕ), x + y = z^2) :=
by sorry

end NUMINAMATH_CALUDE_partition_contains_perfect_square_sum_l2253_225367


namespace NUMINAMATH_CALUDE_max_profit_allocation_l2253_225307

/-- Profit function for product A -/
def profit_A (x : ℝ) : ℝ := -x^2 + 4*x

/-- Profit function for product B -/
def profit_B (x : ℝ) : ℝ := 2*x

/-- Total profit function -/
def total_profit (x : ℝ) : ℝ := profit_A x + profit_B (3 - x)

/-- Theorem stating the maximum profit and optimal investment allocation -/
theorem max_profit_allocation :
  ∃ (x : ℝ), x ∈ Set.Icc 0 3 ∧
  (∀ y ∈ Set.Icc 0 3, total_profit x ≥ total_profit y) ∧
  x = 1 ∧ total_profit x = 7 := by
  sorry

end NUMINAMATH_CALUDE_max_profit_allocation_l2253_225307


namespace NUMINAMATH_CALUDE_composite_shape_area_l2253_225379

/-- The area of a composite shape consisting of three rectangles --/
def composite_area (rect1_width rect1_height rect2_width rect2_height rect3_width rect3_height : ℕ) : ℕ :=
  rect1_width * rect1_height + rect2_width * rect2_height + rect3_width * rect3_height

/-- Theorem stating that the area of the given composite shape is 77 square units --/
theorem composite_shape_area : composite_area 10 4 4 7 3 3 = 77 := by
  sorry

end NUMINAMATH_CALUDE_composite_shape_area_l2253_225379


namespace NUMINAMATH_CALUDE_mans_age_l2253_225369

theorem mans_age (P : ℝ) 
  (h1 : P = 1.25 * (P - 10)) 
  (h2 : P = (250 / 300) * (P + 10)) : 
  P = 50 := by
  sorry

end NUMINAMATH_CALUDE_mans_age_l2253_225369


namespace NUMINAMATH_CALUDE_president_secretary_selection_l2253_225332

/-- The number of ways to select 2 people from n people and assign them to 2 distinct roles -/
def permutation_two_roles (n : ℕ) : ℕ := n * (n - 1)

/-- There are 6 people to choose from -/
def number_of_people : ℕ := 6

theorem president_secretary_selection :
  permutation_two_roles number_of_people = 30 := by
  sorry

end NUMINAMATH_CALUDE_president_secretary_selection_l2253_225332


namespace NUMINAMATH_CALUDE_circle_radius_problem_l2253_225389

-- Define the points
def A : ℝ × ℝ := (-1, 2)
def B : ℝ × ℝ := (1, 2)
def C : ℝ × ℝ := (5, -2)

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem statement
theorem circle_radius_problem (M N : Circle) : 
  -- Conditions
  (M.center.1 = 0) →  -- Center of M is on y-axis
  (N.center.1 = 2) →  -- x-coordinate of N's center is 2
  (N.center.2 = 4 - M.center.2) →  -- y-coordinate of N's center
  (M.radius = N.radius) →  -- Equal radii
  (M.radius^2 = (B.1 - M.center.1)^2 + (B.2 - M.center.2)^2) →  -- M passes through B
  (N.radius^2 = (B.1 - N.center.1)^2 + (B.2 - N.center.2)^2) →  -- N passes through B
  (N.radius^2 = (C.1 - N.center.1)^2 + (C.2 - N.center.2)^2) →  -- N passes through C
  -- Conclusion
  M.radius = Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_problem_l2253_225389


namespace NUMINAMATH_CALUDE_edwards_initial_money_l2253_225365

theorem edwards_initial_money :
  ∀ (initial_book_cost : ℝ) (discount_rate : ℝ) (num_books : ℕ) (pen_cost : ℝ) (num_pens : ℕ) (money_left : ℝ),
    initial_book_cost = 40 →
    discount_rate = 0.25 →
    num_books = 100 →
    pen_cost = 2 →
    num_pens = 3 →
    money_left = 6 →
    ∃ (initial_money : ℝ),
      initial_money = initial_book_cost * (1 - discount_rate) + (pen_cost * num_pens) + money_left ∧
      initial_money = 42 :=
by sorry

end NUMINAMATH_CALUDE_edwards_initial_money_l2253_225365


namespace NUMINAMATH_CALUDE_point_p_final_position_l2253_225314

def final_position (initial : ℤ) (right_move : ℤ) (left_move : ℤ) : ℤ :=
  initial + right_move - left_move

theorem point_p_final_position :
  final_position (-2) 5 4 = -1 := by
  sorry

end NUMINAMATH_CALUDE_point_p_final_position_l2253_225314


namespace NUMINAMATH_CALUDE_maruti_car_sales_decrease_l2253_225308

theorem maruti_car_sales_decrease (initial_price initial_sales : ℝ) 
  (price_increase : ℝ) (revenue_increase : ℝ) (sales_decrease : ℝ) :
  price_increase = 0.3 →
  revenue_increase = 0.04 →
  (initial_price * (1 + price_increase)) * (initial_sales * (1 - sales_decrease)) = 
    initial_price * initial_sales * (1 + revenue_increase) →
  sales_decrease = 0.2 := by
sorry

end NUMINAMATH_CALUDE_maruti_car_sales_decrease_l2253_225308


namespace NUMINAMATH_CALUDE_manuscript_cost_is_860_l2253_225368

/-- Calculates the total cost of typing a manuscript with given parameters. -/
def manuscriptTypingCost (totalPages : ℕ) (revisedOnce : ℕ) (revisedTwice : ℕ) 
  (firstTypeCost : ℕ) (revisionCost : ℕ) : ℕ :=
  totalPages * firstTypeCost + revisedOnce * revisionCost + revisedTwice * 2 * revisionCost

/-- Proves that the total cost of typing a 100-page manuscript with given revision parameters is $860. -/
theorem manuscript_cost_is_860 : 
  manuscriptTypingCost 100 35 15 6 4 = 860 := by
  sorry

#eval manuscriptTypingCost 100 35 15 6 4

end NUMINAMATH_CALUDE_manuscript_cost_is_860_l2253_225368


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2253_225364

theorem quadratic_equation_roots (x : ℝ) : 
  (∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ x * (x - 2) = x - 2 → x = r₁ ∨ x = r₂) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2253_225364


namespace NUMINAMATH_CALUDE_circle_tangent_slope_range_l2253_225396

theorem circle_tangent_slope_range (x y : ℝ) (h : x^2 + y^2 = 1) :
  ∃ (k : ℝ), k = 3/4 ∧ ∀ (z : ℝ), z ≥ k → ∃ (a b : ℝ), a^2 + b^2 = 1 ∧ z = (b + 2) / (a + 1) :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_slope_range_l2253_225396


namespace NUMINAMATH_CALUDE_coefficient_c_nonzero_l2253_225328

def Q (a' b' c' d' x : ℝ) : ℝ := x^4 + a'*x^3 + b'*x^2 + c'*x + d'

theorem coefficient_c_nonzero 
  (a' b' c' d' : ℝ) 
  (h1 : ∃ u v w : ℝ, u ≠ v ∧ u ≠ w ∧ v ≠ w ∧ u ≠ 0 ∧ v ≠ 0 ∧ w ≠ 0)
  (h2 : ∀ x : ℝ, Q a' b' c' d' x = x * (x - u) * (x - v) * (x - w))
  (h3 : d' = 0) :
  c' ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_coefficient_c_nonzero_l2253_225328


namespace NUMINAMATH_CALUDE_nested_sqrt_solution_l2253_225397

/-- The positive solution to the nested square root equation -/
theorem nested_sqrt_solution : 
  ∃! (x : ℝ), x > 0 ∧ 
  (∃ (z : ℝ), z > 0 ∧ z = Real.sqrt (x + z)) ∧
  (∃ (y : ℝ), y > 0 ∧ y = Real.sqrt (x * y)) ∧
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_nested_sqrt_solution_l2253_225397


namespace NUMINAMATH_CALUDE_factory_sampling_is_systematic_l2253_225347

/-- Represents a sampling method -/
inductive SamplingMethod
| Simple
| Stratified
| Systematic

/-- Represents a sampling scenario -/
structure SamplingScenario where
  totalItems : Nat
  sampleSize : Nat
  method : SamplingMethod

/-- Determines if a sampling scenario is suitable for systematic sampling -/
def isSuitableForSystematic (scenario : SamplingScenario) : Prop :=
  scenario.method = SamplingMethod.Systematic ∧
  scenario.totalItems ≥ scenario.sampleSize ∧
  scenario.totalItems % scenario.sampleSize = 0

/-- The given sampling scenario -/
def factorySampling : SamplingScenario :=
  { totalItems := 2000
    sampleSize := 200
    method := SamplingMethod.Systematic }

/-- Theorem stating that the factory sampling scenario is suitable for systematic sampling -/
theorem factory_sampling_is_systematic :
  isSuitableForSystematic factorySampling :=
by
  sorry


end NUMINAMATH_CALUDE_factory_sampling_is_systematic_l2253_225347


namespace NUMINAMATH_CALUDE_injective_function_equality_l2253_225359

def injective (f : ℕ → ℝ) : Prop := ∀ n m : ℕ, f n = f m → n = m

theorem injective_function_equality (f : ℕ → ℝ) (n m : ℕ) 
  (h_inj : injective f) 
  (h_eq : 1 / f n + 1 / f m = 4 / (f n + f m)) : 
  n = m := by
  sorry

end NUMINAMATH_CALUDE_injective_function_equality_l2253_225359


namespace NUMINAMATH_CALUDE_jack_morning_emails_l2253_225301

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := sorry

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 10

/-- The number of emails Jack received in the evening -/
def evening_emails : ℕ := 7

/-- Theorem stating that Jack received 9 emails in the morning -/
theorem jack_morning_emails : 
  morning_emails = evening_emails + 2 → morning_emails = 9 :=
by sorry

end NUMINAMATH_CALUDE_jack_morning_emails_l2253_225301


namespace NUMINAMATH_CALUDE_exponent_equality_and_inequalities_l2253_225358

theorem exponent_equality_and_inequalities : 
  ((-2 : ℤ)^3 = -2^3) ∧ 
  ((-2 : ℤ)^2 ≠ -2^2) ∧ 
  (|(-2 : ℤ)|^2 ≠ -2^2) ∧ 
  (|(-2 : ℤ)|^3 ≠ -2^3) :=
by sorry

end NUMINAMATH_CALUDE_exponent_equality_and_inequalities_l2253_225358


namespace NUMINAMATH_CALUDE_cookies_with_four_cups_l2253_225366

/-- Represents the number of cookies that can be made with a given amount of flour,
    maintaining a constant ratio of flour to sugar. -/
def cookies_made (flour : ℚ) : ℚ :=
  24 * flour / 3

/-- The ratio of flour to sugar remains constant. -/
axiom constant_ratio : ∀ (f : ℚ), cookies_made f / f = 24 / 3

theorem cookies_with_four_cups :
  cookies_made 4 = 128 :=
sorry

end NUMINAMATH_CALUDE_cookies_with_four_cups_l2253_225366


namespace NUMINAMATH_CALUDE_fourth_power_of_nested_root_l2253_225361

theorem fourth_power_of_nested_root : 
  let x := Real.sqrt (2 + Real.sqrt (3 + Real.sqrt 4))
  x^4 = 9 + 4 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_fourth_power_of_nested_root_l2253_225361


namespace NUMINAMATH_CALUDE_bus_count_l2253_225353

theorem bus_count (total_students : ℕ) (students_per_bus : ℕ) (h1 : total_students = 360) (h2 : students_per_bus = 45) :
  total_students / students_per_bus = 8 :=
by sorry

end NUMINAMATH_CALUDE_bus_count_l2253_225353


namespace NUMINAMATH_CALUDE_number_operations_problem_l2253_225321

theorem number_operations_problem (x : ℚ) : 
  (((11 * x + 6) / 5) - 42 = 12) ↔ (x = 24) :=
by sorry

end NUMINAMATH_CALUDE_number_operations_problem_l2253_225321


namespace NUMINAMATH_CALUDE_hot_dogs_remainder_l2253_225331

theorem hot_dogs_remainder : 35252983 % 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hot_dogs_remainder_l2253_225331


namespace NUMINAMATH_CALUDE_triangle_angle_difference_l2253_225350

/-- In a triangle with angles a, b, and c, where b = 2a and c = a - 15, 
    prove that a - c = 15 --/
theorem triangle_angle_difference (a b c : ℝ) : 
  a + b + c = 180 → b = 2 * a → c = a - 15 → a - c = 15 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_difference_l2253_225350


namespace NUMINAMATH_CALUDE_danny_jane_age_difference_l2253_225319

/-- Proves that 22 years ago, Danny was four and a half times as old as Jane. -/
theorem danny_jane_age_difference : ∃ (x : ℕ), 
  (40 - x : ℚ) = (4.5 : ℚ) * (26 - x : ℚ) ∧ x = 22 := by
  sorry

end NUMINAMATH_CALUDE_danny_jane_age_difference_l2253_225319


namespace NUMINAMATH_CALUDE_lucy_paid_correct_l2253_225356

/-- Calculate the total amount Lucy paid for fruits with discounts applied -/
def total_paid (grapes_kg : ℝ) (grapes_price : ℝ) (mangoes_kg : ℝ) (mangoes_price : ℝ)
                (apples_kg : ℝ) (apples_price : ℝ) (oranges_kg : ℝ) (oranges_price : ℝ)
                (grapes_apples_discount : ℝ) (mangoes_oranges_discount : ℝ) : ℝ :=
  let grapes_cost := grapes_kg * grapes_price
  let mangoes_cost := mangoes_kg * mangoes_price
  let apples_cost := apples_kg * apples_price
  let oranges_cost := oranges_kg * oranges_price
  let grapes_apples_total := grapes_cost + apples_cost
  let mangoes_oranges_total := mangoes_cost + oranges_cost
  let grapes_apples_discounted := grapes_apples_total * (1 - grapes_apples_discount)
  let mangoes_oranges_discounted := mangoes_oranges_total * (1 - mangoes_oranges_discount)
  grapes_apples_discounted + mangoes_oranges_discounted

theorem lucy_paid_correct :
  total_paid 6 74 9 59 4 45 12 32 0.07 0.05 = 1449.57 := by
  sorry

end NUMINAMATH_CALUDE_lucy_paid_correct_l2253_225356


namespace NUMINAMATH_CALUDE_correct_probability_l2253_225370

-- Define the set of balls
inductive Ball : Type
| Red1 : Ball
| Red2 : Ball
| Red3 : Ball
| White2 : Ball
| White3 : Ball

-- Define a function to check if two balls have different colors and numbers
def differentColorAndNumber (b1 b2 : Ball) : Prop :=
  match b1, b2 with
  | Ball.Red1, Ball.White2 => True
  | Ball.Red1, Ball.White3 => True
  | Ball.Red2, Ball.White3 => True
  | Ball.Red3, Ball.White2 => True
  | _, _ => False

-- Define the probability of drawing two balls with different colors and numbers
def probabilityDifferentColorAndNumber : ℚ :=
  2 / 5

-- State the theorem
theorem correct_probability :
  probabilityDifferentColorAndNumber = 2 / 5 := by
  sorry


end NUMINAMATH_CALUDE_correct_probability_l2253_225370


namespace NUMINAMATH_CALUDE_log_equation_solution_l2253_225386

theorem log_equation_solution :
  ∃! x : ℝ, x > 0 ∧ Real.log x + Real.log (x + 1) = 2 ∧ x = (-1 + Real.sqrt 401) / 2 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2253_225386


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l2253_225392

theorem polynomial_division_theorem (x : ℝ) :
  ∃ (q r : ℝ), 5*x^4 - 3*x^3 + 7*x^2 - 9*x + 12 = (x - 3) * (5*x^3 + 12*x^2 + 43*x + 120) + 372 :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l2253_225392


namespace NUMINAMATH_CALUDE_money_sharing_l2253_225318

theorem money_sharing (amanda ben carlos total : ℕ) : 
  amanda + ben + carlos = total → 
  amanda = 24 → 
  2 * ben = 3 * amanda → 
  8 * amanda = 2 * carlos → 
  total = 156 := by
  sorry

end NUMINAMATH_CALUDE_money_sharing_l2253_225318


namespace NUMINAMATH_CALUDE_no_square_divisible_by_six_between_50_and_120_l2253_225378

theorem no_square_divisible_by_six_between_50_and_120 : 
  ¬ ∃ x : ℕ, x^2 = x ∧ x % 6 = 0 ∧ 50 < x ∧ x < 120 := by
sorry

end NUMINAMATH_CALUDE_no_square_divisible_by_six_between_50_and_120_l2253_225378


namespace NUMINAMATH_CALUDE_gcd_15_70_l2253_225305

theorem gcd_15_70 : Nat.gcd 15 70 = 5 := by
  sorry

end NUMINAMATH_CALUDE_gcd_15_70_l2253_225305


namespace NUMINAMATH_CALUDE_coplanar_condition_l2253_225316

open Vector

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]
variable (O P Q R S : V)

-- Define the coplanarity condition
def coplanar (P Q R S : V) : Prop :=
  ∃ (a b c d : ℝ), a • (P - O) + b • (Q - O) + c • (R - O) + d • (S - O) = 0 ∧ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∨ d ≠ 0)

-- State the theorem
theorem coplanar_condition (O P Q R S : V) :
  4 • (P - O) - 3 • (Q - O) + 6 • (R - O) + (-7) • (S - O) = 0 →
  coplanar O P Q R S :=
by sorry

end NUMINAMATH_CALUDE_coplanar_condition_l2253_225316


namespace NUMINAMATH_CALUDE_tan_arcsec_25_24_l2253_225374

theorem tan_arcsec_25_24 : Real.tan (Real.arccos (24 / 25)) = 7 / 24 := by
  sorry

end NUMINAMATH_CALUDE_tan_arcsec_25_24_l2253_225374


namespace NUMINAMATH_CALUDE_zinc_copper_ratio_theorem_l2253_225372

/-- Represents the ratio of two quantities -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Represents a mixture of zinc and copper -/
structure Mixture where
  total_weight : ℝ
  zinc_weight : ℝ

/-- Calculates the ratio of zinc to copper in a mixture -/
def zinc_copper_ratio (m : Mixture) : Ratio :=
  sorry

/-- The given mixture of zinc and copper -/
def given_mixture : Mixture :=
  { total_weight := 74
    zinc_weight := 33.3 }

/-- Theorem stating the correct ratio of zinc to copper in the given mixture -/
theorem zinc_copper_ratio_theorem :
  zinc_copper_ratio given_mixture = Ratio.mk 333 407 :=
  sorry

end NUMINAMATH_CALUDE_zinc_copper_ratio_theorem_l2253_225372


namespace NUMINAMATH_CALUDE_sum_of_f_negative_l2253_225323

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom f_property : ∀ x, f (-x) = -f (x + 4)
axiom f_increasing : ∀ x y, x > 2 → y > x → f y > f x

-- Define the theorem
theorem sum_of_f_negative (x₁ x₂ : ℝ) 
  (h1 : x₁ + x₂ < 4) 
  (h2 : (x₁ - 2) * (x₂ - 2) < 0) : 
  f x₁ + f x₂ < 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_f_negative_l2253_225323


namespace NUMINAMATH_CALUDE_A_alone_days_l2253_225333

-- Define work rates for A, B, and C
def work_rate_A : ℝ := sorry
def work_rate_B : ℝ := sorry
def work_rate_C : ℝ := sorry

-- Define conditions
axiom cond1 : work_rate_A + work_rate_B = 1 / 3
axiom cond2 : work_rate_B + work_rate_C = 1 / 6
axiom cond3 : work_rate_A + work_rate_C = 5 / 18
axiom cond4 : work_rate_A + work_rate_B + work_rate_C = 1 / 2

-- Theorem to prove
theorem A_alone_days : 1 / work_rate_A = 36 / 7 := by sorry

end NUMINAMATH_CALUDE_A_alone_days_l2253_225333


namespace NUMINAMATH_CALUDE_combined_average_mark_l2253_225382

/-- Given two classes with specified number of students and average marks,
    calculate the combined average mark of all students. -/
theorem combined_average_mark (n1 n2 : ℕ) (avg1 avg2 : ℚ) :
  n1 > 0 → n2 > 0 →
  (n1 : ℚ) * avg1 + (n2 : ℚ) * avg2 / ((n1 : ℚ) + (n2 : ℚ)) =
  ((55 : ℚ) * 60 + (48 : ℚ) * 58) / ((55 : ℚ) + (48 : ℚ)) := by
  sorry

#eval ((55 : ℚ) * 60 + (48 : ℚ) * 58) / ((55 : ℚ) + (48 : ℚ))

end NUMINAMATH_CALUDE_combined_average_mark_l2253_225382


namespace NUMINAMATH_CALUDE_triangle_properties_l2253_225311

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π ∧
  b * Real.sin A = 2 * Real.sqrt 3 * a * (Real.cos (B / 2))^2 - Real.sqrt 3 * a ∧
  b = 4 * Real.sqrt 3 ∧
  Real.sin A * Real.cos B + Real.cos A * Real.sin B = 2 * Real.sin A →
  B = π / 3 ∧ 
  (1/2) * a * c * Real.sin B = 8 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l2253_225311


namespace NUMINAMATH_CALUDE_largest_integer_less_than_x_l2253_225371

theorem largest_integer_less_than_x (x : ℤ) 
  (h1 : 5 < x ∧ x < 21)
  (h2 : 7 < x ∧ x < 18)
  (h3 : x < 13)
  (h4 : 12 > x ∧ x > 9)
  (h5 : x + 1 < 13) :
  ∃ (y : ℤ), x > y ∧ ∀ (z : ℤ), x > z → z ≤ y ∧ y = 9 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_less_than_x_l2253_225371


namespace NUMINAMATH_CALUDE_cost_price_calculation_l2253_225325

/-- Proves that the cost price is 17500 given the selling price, discount rate, and profit rate --/
theorem cost_price_calculation (selling_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) : 
  selling_price = 21000 →
  discount_rate = 0.1 →
  profit_rate = 0.08 →
  (selling_price * (1 - discount_rate)) / (1 + profit_rate) = 17500 := by
  sorry

#check cost_price_calculation

end NUMINAMATH_CALUDE_cost_price_calculation_l2253_225325


namespace NUMINAMATH_CALUDE_not_all_acquainted_l2253_225338

/-- Represents a person in the company -/
inductive Person : Type
  | host : Person
  | son1 : Person
  | son2 : Person
  | son3 : Person
  | guest1 : Person
  | guest2 : Person
  | guest3 : Person

/-- Represents the acquaintance relation between two people -/
def acquainted : Person → Person → Prop := sorry

/-- The host is acquainted with all his sons -/
axiom host_knows_sons :
  acquainted Person.host Person.son1 ∧
  acquainted Person.host Person.son2 ∧
  acquainted Person.host Person.son3

/-- Each son is acquainted with exactly one guest -/
axiom sons_know_guests :
  acquainted Person.son1 Person.guest1 ∧
  acquainted Person.son2 Person.guest2 ∧
  acquainted Person.son3 Person.guest3

/-- No guest knows another guest -/
axiom guests_dont_know_each_other :
  ¬acquainted Person.guest1 Person.guest2 ∧
  ¬acquainted Person.guest2 Person.guest3 ∧
  ¬acquainted Person.guest3 Person.guest1

/-- The acquaintance relation is symmetric -/
axiom acquainted_symmetric :
  ∀ (p q : Person), acquainted p q → acquainted q p

/-- Theorem: There exists a pair of people who are not acquainted -/
theorem not_all_acquainted : ∃ (p q : Person), p ≠ q ∧ ¬acquainted p q := by
  sorry

end NUMINAMATH_CALUDE_not_all_acquainted_l2253_225338


namespace NUMINAMATH_CALUDE_inequality_solution_existence_l2253_225329

theorem inequality_solution_existence (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ x * Real.log x - a < 0) ↔ a > -1 / Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_existence_l2253_225329


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l2253_225335

theorem simplify_and_evaluate_expression :
  let x : ℝ := Real.sqrt 3 + 1
  let y : ℝ := Real.sqrt 3
  ((3 * x + y) / (x^2 - y^2) + (2 * x) / (y^2 - x^2)) / (2 / (x^2 * y - x * y^2)) = (3 + Real.sqrt 3) / 2 :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l2253_225335


namespace NUMINAMATH_CALUDE_walter_age_theorem_l2253_225393

def walter_age_1999 (walter_age_1994 grandmother_age_1994 birth_year_sum : ℕ) : Prop :=
  walter_age_1994 * 2 = grandmother_age_1994 ∧
  (1994 - walter_age_1994) + (1994 - grandmother_age_1994) = birth_year_sum ∧
  walter_age_1994 + (1999 - 1994) = 55

theorem walter_age_theorem : 
  ∃ (walter_age_1994 grandmother_age_1994 : ℕ), 
    walter_age_1999 walter_age_1994 grandmother_age_1994 3838 :=
by
  sorry

end NUMINAMATH_CALUDE_walter_age_theorem_l2253_225393


namespace NUMINAMATH_CALUDE_johnson_potatoes_problem_l2253_225373

theorem johnson_potatoes_problem (initial_potatoes : ℕ) (remaining_potatoes : ℕ) 
  (h1 : initial_potatoes = 300)
  (h2 : remaining_potatoes = 47) :
  ∃ (gina_potatoes : ℕ),
    gina_potatoes = 69 ∧
    initial_potatoes - remaining_potatoes = 
      gina_potatoes + 2 * gina_potatoes + 2 * gina_potatoes / 3 :=
by sorry

end NUMINAMATH_CALUDE_johnson_potatoes_problem_l2253_225373


namespace NUMINAMATH_CALUDE_smallest_five_digit_congruent_to_3_mod_17_l2253_225352

theorem smallest_five_digit_congruent_to_3_mod_17 : ∃ (n : ℕ), 
  (n ≥ 10000 ∧ n < 100000) ∧  -- five-digit positive integer
  (n % 17 = 3) ∧              -- congruent to 3 modulo 17
  (∀ m : ℕ, (m ≥ 10000 ∧ m < 100000) ∧ (m % 17 = 3) → m ≥ n) ∧ 
  n = 10012 :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_congruent_to_3_mod_17_l2253_225352


namespace NUMINAMATH_CALUDE_complement_hit_at_least_once_l2253_225302

-- Define the sample space
def Ω : Type := Bool × Bool

-- Define the event of hitting the target at least once
def hit_at_least_once (ω : Ω) : Prop :=
  ω.1 ∨ ω.2

-- Define the event of missing the target both times
def miss_both_times (ω : Ω) : Prop :=
  ¬ω.1 ∧ ¬ω.2

-- Theorem stating that the complement of hitting at least once
-- is equivalent to missing both times
theorem complement_hit_at_least_once (ω : Ω) :
  ¬(hit_at_least_once ω) ↔ miss_both_times ω :=
sorry

end NUMINAMATH_CALUDE_complement_hit_at_least_once_l2253_225302


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2253_225381

theorem quadratic_equation_roots (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 + (2*k - 1)*x₁ + k^2 - 1 = 0 ∧
    x₂^2 + (2*k - 1)*x₂ + k^2 - 1 = 0 ∧
    x₁^2 + x₂^2 = 16 + x₁*x₂) →
  k = -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2253_225381


namespace NUMINAMATH_CALUDE_cycling_speed_rectangular_park_l2253_225340

/-- Calculates the cycling speed around a rectangular park -/
theorem cycling_speed_rectangular_park 
  (L B : ℝ) 
  (h1 : B = 3 * L) 
  (h2 : L * B = 120000) 
  (h3 : (2 * L + 2 * B) / 8 = 200) : 
  (200 : ℝ) * 60 / 1000 = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cycling_speed_rectangular_park_l2253_225340


namespace NUMINAMATH_CALUDE_climb_10_stairs_l2253_225339

/-- The number of ways to climb n stairs -/
def climbWays : ℕ → ℕ
  | 0 => 1  -- base case for 0 stairs
  | 1 => 1  -- given condition
  | 2 => 2  -- given condition
  | (n + 3) => climbWays (n + 2) + climbWays (n + 1)

/-- Theorem stating that there are 89 ways to climb 10 stairs -/
theorem climb_10_stairs : climbWays 10 = 89 := by
  sorry

/-- Lemma: The number of ways to climb n stairs is the sum of ways to climb (n-1) and (n-2) stairs -/
lemma climb_recursive (n : ℕ) (h : n ≥ 3) : climbWays n = climbWays (n - 1) + climbWays (n - 2) := by
  sorry

end NUMINAMATH_CALUDE_climb_10_stairs_l2253_225339


namespace NUMINAMATH_CALUDE_unique_extremum_implies_a_range_l2253_225390

noncomputable def f (a x : ℝ) : ℝ := a * (x - 2) * Real.exp x + Real.log x + 1 / x

theorem unique_extremum_implies_a_range (a : ℝ) :
  (∃! x, ∀ y, f a y ≤ f a x) →
  (∃ x, ∀ y, f a y ≤ f a x ∧ f a x > 0) →
  0 ≤ a ∧ a < 1 / Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_unique_extremum_implies_a_range_l2253_225390


namespace NUMINAMATH_CALUDE_parallel_vectors_condition_l2253_225327

/-- Two vectors are parallel if and only if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- The statement that m = 1 is a necessary but not sufficient condition for
    vectors (m, 1) and (1, m) to be parallel -/
theorem parallel_vectors_condition :
  ∃ m : ℝ, (m = 1 → are_parallel (m, 1) (1, m)) ∧
           ¬(are_parallel (m, 1) (1, m) → m = 1) :=
sorry

end NUMINAMATH_CALUDE_parallel_vectors_condition_l2253_225327


namespace NUMINAMATH_CALUDE_maya_has_largest_answer_l2253_225345

def sara_calculation (x : ℕ) : ℕ := x^2 - 3 + 4

def liam_calculation (x : ℕ) : ℕ := (x - 2)^2 + 4

def maya_calculation (x : ℕ) : ℕ := (x - 3 + 4)^2

theorem maya_has_largest_answer :
  let initial_number := 15
  maya_calculation initial_number > sara_calculation initial_number ∧
  maya_calculation initial_number > liam_calculation initial_number :=
by sorry

end NUMINAMATH_CALUDE_maya_has_largest_answer_l2253_225345


namespace NUMINAMATH_CALUDE_fraction_closest_to_longest_side_l2253_225355

-- Define the quadrilateral
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)
  (angleA angleD : Real)
  (angleB angleC : Real)
  (lengthAB lengthBC lengthCD lengthDA : Real)

-- Define the function to calculate the area closest to DA
def areaClosestToDA (q : Quadrilateral) : Real := sorry

-- Define the function to calculate the total area of the quadrilateral
def totalArea (q : Quadrilateral) : Real := sorry

-- Theorem statement
theorem fraction_closest_to_longest_side 
  (q : Quadrilateral)
  (h1 : q.A = (0, 0))
  (h2 : q.B = (1, 2))
  (h3 : q.C = (3, 2))
  (h4 : q.D = (4, 0))
  (h5 : q.angleA = 75)
  (h6 : q.angleD = 75)
  (h7 : q.angleB = 105)
  (h8 : q.angleC = 105)
  (h9 : q.lengthAB = 100)
  (h10 : q.lengthBC = 150)
  (h11 : q.lengthCD = 100)
  (h12 : q.lengthDA = 150)
  : areaClosestToDA q / totalArea q = areaClosestToDA q / totalArea q := by
  sorry

end NUMINAMATH_CALUDE_fraction_closest_to_longest_side_l2253_225355


namespace NUMINAMATH_CALUDE_first_player_winning_strategy_l2253_225362

/-- Represents the state of the game with two piles of candies -/
structure GameState where
  p : Nat
  q : Nat

/-- Determines if a given number is a winning number (congruent to 0, 1, or 4 mod 5) -/
def isWinningNumber (n : Nat) : Prop :=
  n % 5 = 0 ∨ n % 5 = 1 ∨ n % 5 = 4

/-- Determines if a given game state is a winning state for the first player -/
def isWinningState (state : GameState) : Prop :=
  isWinningNumber state.p ∨ isWinningNumber state.q

/-- Theorem stating the winning condition for the first player -/
theorem first_player_winning_strategy (state : GameState) :
  (∃ (strategy : GameState → GameState), 
    (∀ (opponent_move : GameState → GameState), 
      strategy (opponent_move (strategy state)) = state)) ↔ 
  isWinningState state :=
sorry

end NUMINAMATH_CALUDE_first_player_winning_strategy_l2253_225362


namespace NUMINAMATH_CALUDE_remainder_problem_l2253_225391

theorem remainder_problem (n : ℕ) (h1 : n = 349) (h2 : n % 17 = 9) : n % 13 = 11 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2253_225391


namespace NUMINAMATH_CALUDE_product_remainder_l2253_225383

theorem product_remainder (a b c : ℕ) (h : a = 1625 ∧ b = 1627 ∧ c = 1629) : 
  (a * b * c) % 12 = 3 := by
sorry

end NUMINAMATH_CALUDE_product_remainder_l2253_225383


namespace NUMINAMATH_CALUDE_hyperbola_midpoint_existence_l2253_225300

theorem hyperbola_midpoint_existence : ∃ (x₁ y₁ x₂ y₂ : ℝ),
  (x₁^2 - y₁^2/9 = 1) ∧
  (x₂^2 - y₂^2/9 = 1) ∧
  ((x₁ + x₂)/2 = -1) ∧
  ((y₁ + y₂)/2 = -4) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_midpoint_existence_l2253_225300


namespace NUMINAMATH_CALUDE_player1_wins_l2253_225375

/-- Represents a position on the rectangular table -/
structure Position :=
  (x : ℝ) (y : ℝ)

/-- Represents a coin placed on the table -/
structure Coin :=
  (center : Position)

/-- Represents the game state -/
structure GameState :=
  (table_width : ℝ)
  (table_height : ℝ)
  (coins : List Coin)

/-- Checks if a coin placement is valid -/
def is_valid_placement (state : GameState) (new_coin : Coin) : Prop :=
  ∀ c ∈ state.coins, 
    (c.center.x - new_coin.center.x)^2 + (c.center.y - new_coin.center.y)^2 > 4

/-- Represents a player's strategy -/
def Strategy := GameState → Option Coin

/-- Represents the winning strategy for Player 1 -/
def winning_strategy : Strategy := sorry

/-- Theorem stating that Player 1 has a winning strategy -/
theorem player1_wins (width height : ℝ) (h_positive : width > 0 ∧ height > 0) :
  ∃ (s : Strategy), ∀ (game : GameState),
    game.table_width = width ∧ 
    game.table_height = height → 
    (∃ (c : Coin), is_valid_placement game c) → 
    ∃ (c : Coin), is_valid_placement (GameState.mk width height (c :: game.coins)) c :=
sorry

end NUMINAMATH_CALUDE_player1_wins_l2253_225375


namespace NUMINAMATH_CALUDE_retailer_loss_percentage_l2253_225303

-- Define the initial conditions
def initial_cost_price_A : ℝ := 800
def initial_retail_price_B : ℝ := 900
def initial_exchange_rate : ℝ := 1.1
def first_discount : ℝ := 0.1
def second_discount : ℝ := 0.15
def sales_tax : ℝ := 0.1
def final_exchange_rate : ℝ := 1.5

-- Define the theorem
theorem retailer_loss_percentage :
  let price_after_first_discount := initial_retail_price_B * (1 - first_discount)
  let price_after_second_discount := price_after_first_discount * (1 - second_discount)
  let price_with_tax := price_after_second_discount * (1 + sales_tax)
  let final_price_A := price_with_tax / final_exchange_rate
  let loss := initial_cost_price_A - final_price_A
  let percentage_loss := loss / initial_cost_price_A * 100
  ∃ ε > 0, abs (percentage_loss - 36.89) < ε :=
by sorry

end NUMINAMATH_CALUDE_retailer_loss_percentage_l2253_225303


namespace NUMINAMATH_CALUDE_expected_games_specific_l2253_225357

/-- Represents a game with given win probabilities -/
structure Game where
  p_frank : ℝ  -- Probability of Frank winning a game
  p_joe : ℝ    -- Probability of Joe winning a game
  games_to_win : ℕ  -- Number of games needed to win the match

/-- Expected number of games in a match -/
def expected_games (g : Game) : ℝ := sorry

/-- Theorem stating the expected number of games in the specific scenario -/
theorem expected_games_specific :
  let g : Game := {
    p_frank := 0.3,
    p_joe := 0.7,
    games_to_win := 21
  }
  expected_games g = 30 := by sorry

end NUMINAMATH_CALUDE_expected_games_specific_l2253_225357


namespace NUMINAMATH_CALUDE_total_miles_walked_l2253_225385

def monday_miles : ℕ := 9
def tuesday_miles : ℕ := 9

theorem total_miles_walked : monday_miles + tuesday_miles = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_miles_walked_l2253_225385


namespace NUMINAMATH_CALUDE_school_journey_problem_l2253_225342

/-- Represents the time taken for John's journey to and from school -/
structure SchoolJourney where
  road_one_way : ℕ        -- Time taken to walk one way by road
  shortcut_one_way : ℕ    -- Time taken to walk one way by shortcut

/-- The theorem representing John's school journey problem -/
theorem school_journey_problem (j : SchoolJourney) 
  (h1 : j.road_one_way + j.shortcut_one_way = 50)  -- Road + Shortcut = 50 minutes
  (h2 : 2 * j.shortcut_one_way = 30)               -- Shortcut both ways = 30 minutes
  : 2 * j.road_one_way = 70 := by                  -- Road both ways = 70 minutes
  sorry

#check school_journey_problem

end NUMINAMATH_CALUDE_school_journey_problem_l2253_225342


namespace NUMINAMATH_CALUDE_f_properties_l2253_225309

noncomputable def f (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1)

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧
  (∀ x y, x < y → f x < f y) ∧
  (∀ x, f (2*x + 1) + f x < 0 ↔ x < -1/3) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2253_225309


namespace NUMINAMATH_CALUDE_smaller_number_value_l2253_225336

theorem smaller_number_value (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 3) : 
  min x y = 28.5 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_value_l2253_225336


namespace NUMINAMATH_CALUDE_geometric_sequence_solution_l2253_225313

theorem geometric_sequence_solution (x : ℝ) : 
  (1 : ℝ) < x ∧ x < 9 ∧ x^2 = 9 → x = 3 ∨ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_solution_l2253_225313


namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l2253_225387

theorem hemisphere_surface_area (r : ℝ) (h1 : r > 0) (h2 : π * r^2 = 225 * π) : 
  3 * π * r^2 + π * r^2 = 900 * π := by
  sorry

end NUMINAMATH_CALUDE_hemisphere_surface_area_l2253_225387


namespace NUMINAMATH_CALUDE_job_completion_relationship_l2253_225398

/-- Represents the relationship between number of machines and time to finish a job -/
theorem job_completion_relationship (D : ℝ) : 
  D > 0 → -- D is positive (time can't be negative or zero)
  (15 : ℝ) / 20 = (3 / 4 * D) / D := by
  sorry

#check job_completion_relationship

end NUMINAMATH_CALUDE_job_completion_relationship_l2253_225398


namespace NUMINAMATH_CALUDE_max_cookies_andy_l2253_225348

/-- Represents the number of cookies eaten by each sibling -/
structure CookieDistribution where
  andy : Nat
  alexa : Nat
  john : Nat

/-- Checks if the distribution satisfies the problem conditions -/
def isValidDistribution (d : CookieDistribution) : Prop :=
  d.andy + d.alexa + d.john = 36 ∧
  d.andy % d.alexa = 0 ∧
  d.andy % d.john = 0 ∧
  d.alexa > 0 ∧
  d.john > 0

/-- Theorem stating the maximum number of cookies Andy could have eaten -/
theorem max_cookies_andy :
  ∀ d : CookieDistribution, isValidDistribution d → d.andy ≤ 30 :=
sorry

end NUMINAMATH_CALUDE_max_cookies_andy_l2253_225348
