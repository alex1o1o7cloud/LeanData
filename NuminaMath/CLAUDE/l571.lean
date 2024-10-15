import Mathlib

namespace NUMINAMATH_CALUDE_acid_mixture_theorem_l571_57110

/-- Represents an acid solution with a given concentration and volume -/
structure AcidSolution where
  concentration : ℝ
  volume : ℝ

/-- Calculates the amount of pure acid in a solution -/
def pureAcid (solution : AcidSolution) : ℝ :=
  solution.concentration * solution.volume

/-- Theorem: Mixing 4L of 60% acid with 16L of 75% acid yields 20L of 72% acid -/
theorem acid_mixture_theorem :
  let solution1 : AcidSolution := { concentration := 0.60, volume := 4 }
  let solution2 : AcidSolution := { concentration := 0.75, volume := 16 }
  let finalSolution : AcidSolution := { concentration := 0.72, volume := 20 }
  pureAcid solution1 + pureAcid solution2 = pureAcid finalSolution :=
by sorry

end NUMINAMATH_CALUDE_acid_mixture_theorem_l571_57110


namespace NUMINAMATH_CALUDE_triangle_theorem_l571_57187

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_theorem (t : Triangle) 
  (h1 : Real.cos t.C = 1/4)
  (h2 : t.a^2 = t.b^2 + (1/2) * t.c^2) :
  Real.sin (t.A - t.B) = Real.sqrt 15 / 8 ∧
  (t.c = Real.sqrt 10 → t.a = 3 ∧ t.b = 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l571_57187


namespace NUMINAMATH_CALUDE_f_is_odd_and_increasing_l571_57196

def f (x : ℝ) : ℝ := x * |x|

theorem f_is_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_f_is_odd_and_increasing_l571_57196


namespace NUMINAMATH_CALUDE_brothers_ages_l571_57165

theorem brothers_ages (a b : ℕ) (h1 : a > b) (h2 : a / b = 3 / 2) (h3 : a - b = 24) :
  a + b = 120 := by
  sorry

end NUMINAMATH_CALUDE_brothers_ages_l571_57165


namespace NUMINAMATH_CALUDE_arithmetic_sequence_special_case_l571_57136

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the theorem
theorem arithmetic_sequence_special_case (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (a 2)^2 + 12 * (a 2) - 8 = 0 →
  (a 10)^2 + 12 * (a 10) - 8 = 0 →
  a 6 = -6 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_special_case_l571_57136


namespace NUMINAMATH_CALUDE_trig_expression_value_l571_57102

theorem trig_expression_value : 
  (Real.sqrt 3) / (Real.cos (10 * π / 180)) - 1 / (Real.sin (170 * π / 180)) = -4 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_value_l571_57102


namespace NUMINAMATH_CALUDE_benny_state_tax_l571_57156

/-- Calculates the total state tax in cents per hour given an hourly wage in dollars, a tax rate percentage, and a fixed tax in cents. -/
def total_state_tax (hourly_wage : ℚ) (tax_rate_percent : ℚ) (fixed_tax_cents : ℕ) : ℕ :=
  sorry

/-- Proves that given Benny's hourly wage of $25, a 2% state tax rate, and a fixed tax of 50 cents per hour, the total amount of state taxes paid per hour is 100 cents. -/
theorem benny_state_tax :
  total_state_tax 25 2 50 = 100 := by
  sorry

end NUMINAMATH_CALUDE_benny_state_tax_l571_57156


namespace NUMINAMATH_CALUDE_specific_triangle_perimeter_l571_57186

/-- Triangle with parallel lines intersecting its interior --/
structure TriangleWithParallelLines where
  -- Side lengths of the original triangle
  PQ : ℝ
  QR : ℝ
  PR : ℝ
  -- Lengths of segments formed by parallel lines intersecting the triangle
  m_P_length : ℝ
  m_Q_length : ℝ
  m_R_length : ℝ

/-- Calculate the perimeter of the inner triangle formed by parallel lines --/
def innerTrianglePerimeter (t : TriangleWithParallelLines) : ℝ :=
  sorry

/-- Theorem statement for the specific triangle problem --/
theorem specific_triangle_perimeter :
  let t : TriangleWithParallelLines := {
    PQ := 150,
    QR := 275,
    PR := 225,
    m_P_length := 65,
    m_Q_length := 55,
    m_R_length := 25
  }
  innerTrianglePerimeter t = 755 := by
    sorry

end NUMINAMATH_CALUDE_specific_triangle_perimeter_l571_57186


namespace NUMINAMATH_CALUDE_tenth_term_of_arithmetic_sequence_l571_57162

/-- Given an arithmetic sequence of 20 terms with first term 7 and last term 67,
    the 10th term is equal to 673 / 19. -/
theorem tenth_term_of_arithmetic_sequence : 
  ∀ (a : ℕ → ℚ), 
    (∀ n : ℕ, n < 19 → a (n + 1) - a n = a 1 - a 0) →  -- arithmetic sequence condition
    a 0 = 7 →                                         -- first term
    a 19 = 67 →                                       -- last term
    a 9 = 673 / 19 :=                                 -- 10th term (index 9)
by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_arithmetic_sequence_l571_57162


namespace NUMINAMATH_CALUDE_product_of_powers_l571_57145

theorem product_of_powers (m : ℝ) : 2 * m^3 * (3 * m^4) = 6 * m^7 := by
  sorry

end NUMINAMATH_CALUDE_product_of_powers_l571_57145


namespace NUMINAMATH_CALUDE_impossible_belief_l571_57119

-- Define the characters
inductive Character : Type
| King : Character
| Queen : Character

-- Define the state of mind
inductive MindState : Type
| Sane : MindState
| NotSane : MindState

-- Define a belief
structure Belief where
  subject : Character
  object : Character
  state : MindState

-- Define a nested belief
structure NestedBelief where
  level1 : Character
  level2 : Character
  level3 : Character
  finalBelief : Belief

-- Define logical consistency
def logicallyConsistent (c : Character) : Prop :=
  ∀ (b : Belief), b.subject = c → (b.state = MindState.Sane ↔ c = b.object)

-- Define the problematic belief
def problematicBelief : NestedBelief :=
  { level1 := Character.King
  , level2 := Character.Queen
  , level3 := Character.King
  , finalBelief := { subject := Character.King
                   , object := Character.Queen
                   , state := MindState.NotSane } }

-- Theorem statement
theorem impossible_belief
  (h1 : logicallyConsistent Character.King)
  (h2 : logicallyConsistent Character.Queen) :
  ¬ (∃ (b : NestedBelief), b = problematicBelief) :=
sorry

end NUMINAMATH_CALUDE_impossible_belief_l571_57119


namespace NUMINAMATH_CALUDE_symmetry_implies_a_pow_b_eq_four_l571_57133

/-- Two points are symmetric with respect to the x-axis if their x-coordinates are equal
    and their y-coordinates are opposite. -/
def symmetric_x_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = q.1 ∧ p.2 = -q.2

theorem symmetry_implies_a_pow_b_eq_four (a b : ℝ) :
  symmetric_x_axis (a + 2, -2) (4, b) → a^b = 4 := by
  sorry

#check symmetry_implies_a_pow_b_eq_four

end NUMINAMATH_CALUDE_symmetry_implies_a_pow_b_eq_four_l571_57133


namespace NUMINAMATH_CALUDE_salary_increase_proof_l571_57149

theorem salary_increase_proof (num_employees : ℕ) (initial_avg : ℚ) (manager_salary : ℚ) :
  num_employees = 20 ∧ initial_avg = 1500 ∧ manager_salary = 4650 →
  (((num_employees : ℚ) * initial_avg + manager_salary) / (num_employees + 1 : ℚ)) - initial_avg = 150 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_proof_l571_57149


namespace NUMINAMATH_CALUDE_range_of_k_l571_57183

theorem range_of_k (k : ℝ) : 
  (∀ x : ℝ, |x - 2| + |x - 3| > |k - 1|) ↔ k ∈ Set.Ioo 0 2 :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_k_l571_57183


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l571_57181

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the necessary relations
variable (subset : Line → Plane → Prop)
variable (perp : Line → Plane → Prop)
variable (plane_perp : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (a : Line) (α β : Plane) 
  (h : subset a α) : 
  (∀ (b : Line), subset b α → (perp b β → plane_perp α β)) ∧ 
  (∃ (c : Line), subset c α ∧ plane_perp α β ∧ ¬perp c β) :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l571_57181


namespace NUMINAMATH_CALUDE_steven_pears_count_l571_57171

/-- The number of seeds Steven needs to collect -/
def total_seeds : ℕ := 60

/-- The average number of seeds in an apple -/
def apple_seeds : ℕ := 6

/-- The average number of seeds in a pear -/
def pear_seeds : ℕ := 2

/-- The average number of seeds in a grape -/
def grape_seeds : ℕ := 3

/-- The number of apples Steven has set aside -/
def apples_set_aside : ℕ := 4

/-- The number of grapes Steven has set aside -/
def grapes_set_aside : ℕ := 9

/-- The number of additional seeds Steven needs -/
def additional_seeds_needed : ℕ := 3

/-- The number of pears Steven has set aside -/
def pears_set_aside : ℕ := 3

theorem steven_pears_count :
  pears_set_aside * pear_seeds + 
  apples_set_aside * apple_seeds + 
  grapes_set_aside * grape_seeds = 
  total_seeds - additional_seeds_needed :=
by sorry

end NUMINAMATH_CALUDE_steven_pears_count_l571_57171


namespace NUMINAMATH_CALUDE_range_of_expressions_l571_57137

theorem range_of_expressions (a b : ℝ) 
  (ha : 1 < a ∧ a < 4) 
  (hb : 2 < b ∧ b < 8) : 
  (8 < 2*a + 3*b ∧ 2*a + 3*b < 32) ∧ 
  (-7 < a - b ∧ a - b < 2) := by
sorry

end NUMINAMATH_CALUDE_range_of_expressions_l571_57137


namespace NUMINAMATH_CALUDE_bmw_sales_l571_57143

theorem bmw_sales (total : ℕ) (ford_percent : ℚ) (nissan_percent : ℚ) (chevrolet_percent : ℚ)
  (h_total : total = 300)
  (h_ford : ford_percent = 20 / 100)
  (h_nissan : nissan_percent = 25 / 100)
  (h_chevrolet : chevrolet_percent = 10 / 100)
  (h_sum : ford_percent + nissan_percent + chevrolet_percent < 1) :
  ↑total * (1 - (ford_percent + nissan_percent + chevrolet_percent)) = 135 := by
  sorry

end NUMINAMATH_CALUDE_bmw_sales_l571_57143


namespace NUMINAMATH_CALUDE_tangent_sum_difference_l571_57100

theorem tangent_sum_difference (α β : Real) 
  (h1 : Real.tan (α + β) = 2 / 5)
  (h2 : Real.tan (β - Real.pi / 4) = 1 / 4) :
  Real.tan (α + Real.pi / 4) = 3 / 22 := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_difference_l571_57100


namespace NUMINAMATH_CALUDE_convex_polygon_as_intersection_of_halfplanes_l571_57154

-- Define a convex polygon
def ConvexPolygon (P : Set (ℝ × ℝ)) : Prop := sorry

-- Define a half-plane
def HalfPlane (H : Set (ℝ × ℝ)) : Prop := sorry

-- Theorem statement
theorem convex_polygon_as_intersection_of_halfplanes 
  (P : Set (ℝ × ℝ)) (h : ConvexPolygon P) :
  ∃ (n : ℕ) (H : Fin n → Set (ℝ × ℝ)), 
    (∀ i, HalfPlane (H i)) ∧ 
    P = ⋂ i, H i :=
sorry

end NUMINAMATH_CALUDE_convex_polygon_as_intersection_of_halfplanes_l571_57154


namespace NUMINAMATH_CALUDE_sin_2x_value_l571_57169

theorem sin_2x_value (x : ℝ) (h : Real.sin (π + x) + Real.cos (π + x) = 1/2) : 
  Real.sin (2 * x) = -3/4 := by
sorry

end NUMINAMATH_CALUDE_sin_2x_value_l571_57169


namespace NUMINAMATH_CALUDE_cos_theta_equals_three_fifths_l571_57114

/-- Given that the terminal side of angle θ passes through the point (3, -4), prove that cos θ = 3/5 -/
theorem cos_theta_equals_three_fifths (θ : Real) (h : ∃ (r : Real), r > 0 ∧ r * Real.cos θ = 3 ∧ r * Real.sin θ = -4) : 
  Real.cos θ = 3/5 := by
sorry

end NUMINAMATH_CALUDE_cos_theta_equals_three_fifths_l571_57114


namespace NUMINAMATH_CALUDE_unpainted_cubes_4x4x4_l571_57141

/-- Represents a cube with painted faces -/
structure PaintedCube where
  size : ℕ
  total_cubes : ℕ
  painted_per_face : ℕ

/-- Calculates the number of unpainted unit cubes in a painted cube -/
def unpainted_cubes (cube : PaintedCube) : ℕ :=
  cube.total_cubes - (6 * cube.painted_per_face - 12)

/-- Theorem stating that a 4x4x4 cube with 6 painted squares per face has 40 unpainted unit cubes -/
theorem unpainted_cubes_4x4x4 :
  let cube : PaintedCube := ⟨4, 64, 6⟩
  unpainted_cubes cube = 40 := by sorry

end NUMINAMATH_CALUDE_unpainted_cubes_4x4x4_l571_57141


namespace NUMINAMATH_CALUDE_min_value_sum_l571_57191

theorem min_value_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1 / (x + 3) + 1 / (y + 3) = 1 / 4) : 
  x + 3 * y ≥ 4 + 8 * Real.sqrt 3 ∧ 
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 
    1 / (x₀ + 3) + 1 / (y₀ + 3) = 1 / 4 ∧ 
    x₀ + 3 * y₀ = 4 + 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_l571_57191


namespace NUMINAMATH_CALUDE_sibling_product_l571_57138

/-- Represents a family with a specific structure -/
structure Family :=
  (sisters : ℕ)
  (brothers : ℕ)

/-- Calculates the number of sisters and brothers for a sibling in the family -/
def sibling_count (f : Family) : ℕ × ℕ :=
  (f.sisters - 1, f.brothers)

/-- The main theorem stating the product of sisters and brothers for a sibling -/
theorem sibling_product (f : Family) (h1 : f.sisters = 6) (h2 : f.brothers = 3) :
  let (s, b) := sibling_count f
  s * b = 15 := by
  sorry

#check sibling_product

end NUMINAMATH_CALUDE_sibling_product_l571_57138


namespace NUMINAMATH_CALUDE_line_through_points_l571_57129

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem line_through_points (a b : V) (h : a ≠ b) :
  ∃ t : ℝ, (1/2 : ℝ) • a + (1/2 : ℝ) • b = a + t • (b - a) :=
sorry

end NUMINAMATH_CALUDE_line_through_points_l571_57129


namespace NUMINAMATH_CALUDE_floor_sqrt_50_l571_57151

theorem floor_sqrt_50 : ⌊Real.sqrt 50⌋ = 7 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_50_l571_57151


namespace NUMINAMATH_CALUDE_triangle_problem_l571_57172

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  a > b →
  a = 5 →
  c = 6 →
  Real.sin B = 3/5 →
  b = Real.sqrt 13 ∧
  Real.sin A = 3 * Real.sqrt 13 / 13 ∧
  Real.sin (2 * A + π/4) = 7 * Real.sqrt 2 / 26 :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l571_57172


namespace NUMINAMATH_CALUDE_athul_rowing_problem_l571_57185

/-- Athul's rowing problem -/
theorem athul_rowing_problem 
  (v : ℝ) -- Athul's speed in still water (km/h)
  (d : ℝ) -- Distance rowed upstream (km)
  (h1 : v + 1 = 24 / 4) -- Downstream speed equation
  (h2 : v - 1 = d / 4) -- Upstream speed equation
  : d = 16 := by
  sorry

end NUMINAMATH_CALUDE_athul_rowing_problem_l571_57185


namespace NUMINAMATH_CALUDE_quadratic_root_implies_a_value_l571_57197

theorem quadratic_root_implies_a_value (a : ℝ) : 
  (4^2 - 3*4 = a^2) → (a = 2 ∨ a = -2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_a_value_l571_57197


namespace NUMINAMATH_CALUDE_max_pens_purchased_l571_57198

/-- Represents the prices and quantities of pens and mechanical pencils -/
structure PriceQuantity where
  pen_price : ℕ
  pencil_price : ℕ
  pen_quantity : ℕ
  pencil_quantity : ℕ

/-- Represents the pricing conditions given in the problem -/
def pricing_conditions (p : PriceQuantity) : Prop :=
  2 * p.pen_price + 5 * p.pencil_price = 75 ∧
  3 * p.pen_price + 2 * p.pencil_price = 85

/-- Represents the promotion and quantity conditions -/
def promotion_conditions (p : PriceQuantity) : Prop :=
  p.pencil_quantity = 2 * p.pen_quantity + 8 ∧
  p.pen_price * p.pen_quantity + p.pencil_price * (p.pencil_quantity - p.pen_quantity) < 670

/-- Theorem stating the maximum number of pens that can be purchased -/
theorem max_pens_purchased (p : PriceQuantity) 
  (h1 : pricing_conditions p) 
  (h2 : promotion_conditions p) : 
  p.pen_quantity ≤ 20 :=
sorry

end NUMINAMATH_CALUDE_max_pens_purchased_l571_57198


namespace NUMINAMATH_CALUDE_quadratic_polynomial_satisfies_conditions_l571_57155

-- Define the quadratic polynomial
def q (x : ℝ) : ℝ := 2.1 * x^2 - 3.1 * x - 1.2

-- State the theorem
theorem quadratic_polynomial_satisfies_conditions :
  q (-1) = 4 ∧ q 2 = 1 ∧ q 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_satisfies_conditions_l571_57155


namespace NUMINAMATH_CALUDE_ball_bounce_count_l571_57124

/-- The number of bounces required for a ball dropped from 16 feet to reach a height less than 2 feet,
    when it bounces back up two-thirds the distance it just fell. -/
theorem ball_bounce_count : ∃ k : ℕ, 
  (∀ n < k, 16 * (2/3)^n ≥ 2) ∧ 
  16 * (2/3)^k < 2 ∧
  k = 6 := by
  sorry

end NUMINAMATH_CALUDE_ball_bounce_count_l571_57124


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l571_57152

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_sum : a + 3 * b = 1) :
  (1 / a + 1 / b) ≥ 4 + 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l571_57152


namespace NUMINAMATH_CALUDE_bridge_length_l571_57150

/-- The length of a bridge given train parameters -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (time_to_pass : ℝ) :
  train_length = 300 →
  train_speed_kmh = 35 →
  time_to_pass = 42.68571428571429 →
  ∃ (bridge_length : ℝ),
    bridge_length = 115 ∧
    bridge_length + train_length = (train_speed_kmh * 1000 / 3600) * time_to_pass :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_l571_57150


namespace NUMINAMATH_CALUDE_binomial_coefficient_20_10_l571_57189

theorem binomial_coefficient_20_10 
  (h1 : Nat.choose 18 8 = 31824)
  (h2 : Nat.choose 18 9 = 48620)
  (h3 : Nat.choose 18 10 = 43758) :
  Nat.choose 20 10 = 172822 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_20_10_l571_57189


namespace NUMINAMATH_CALUDE_expression_value_l571_57134

theorem expression_value : 2^4 - 4 * 2^3 + 6 * 2^2 - 4 * 2 + 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l571_57134


namespace NUMINAMATH_CALUDE_three_digit_swap_solution_l571_57194

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def swap_digits (n : ℕ) : ℕ :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  100 * c + 10 * b + a

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem three_digit_swap_solution :
  ∀ A B : ℕ,
    is_three_digit A →
    is_three_digit B →
    B = swap_digits A →
    A / B = 3 →
    A % B = 7 * sum_of_digits A →
    ((A = 421 ∧ B = 124) ∨ (A = 842 ∧ B = 248)) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_swap_solution_l571_57194


namespace NUMINAMATH_CALUDE_max_storage_period_is_56_days_l571_57112

/-- Represents the financial parameters for a wholesale product --/
structure WholesaleProduct where
  wholesalePrice : ℝ
  grossProfitMargin : ℝ
  borrowedCapitalRatio : ℝ
  monthlyInterestRate : ℝ
  dailyStorageCost : ℝ

/-- Calculates the maximum storage period without incurring a loss --/
def maxStoragePeriod (p : WholesaleProduct) : ℕ :=
  sorry

/-- Theorem stating the maximum storage period for the given product --/
theorem max_storage_period_is_56_days (p : WholesaleProduct)
  (h1 : p.wholesalePrice = 500)
  (h2 : p.grossProfitMargin = 0.04)
  (h3 : p.borrowedCapitalRatio = 0.8)
  (h4 : p.monthlyInterestRate = 0.0042)
  (h5 : p.dailyStorageCost = 0.30) :
  maxStoragePeriod p = 56 :=
sorry

end NUMINAMATH_CALUDE_max_storage_period_is_56_days_l571_57112


namespace NUMINAMATH_CALUDE_base_b_perfect_square_l571_57164

-- Define the representation of a number in base b
def base_representation (b : ℕ) : ℕ := b^2 + 4*b + 1

-- Theorem statement
theorem base_b_perfect_square (b : ℕ) (h : b > 4) :
  ∃ n : ℕ, base_representation b = n^2 :=
sorry

end NUMINAMATH_CALUDE_base_b_perfect_square_l571_57164


namespace NUMINAMATH_CALUDE_x_three_times_y_l571_57158

theorem x_three_times_y (q : ℚ) (x y : ℚ) 
  (hx : x = 5 - q) 
  (hy : y = 3 * q - 1) : 
  q = 4/5 ↔ x = 3 * y := by
sorry

end NUMINAMATH_CALUDE_x_three_times_y_l571_57158


namespace NUMINAMATH_CALUDE_complex_multiplication_example_l571_57111

theorem complex_multiplication_example :
  let z₁ : ℂ := 2 + 2*I
  let z₂ : ℂ := 1 - 2*I
  z₁ * z₂ = 6 - 2*I :=
by sorry

end NUMINAMATH_CALUDE_complex_multiplication_example_l571_57111


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l571_57123

/-- A line is tangent to a parabola if and only if the value of k satisfies the tangency condition -/
theorem line_tangent_to_parabola (x y k : ℝ) : 
  (∃ (x₀ y₀ : ℝ), (4 * x₀ + 3 * y₀ + k = 0) ∧ (y₀^2 = 12 * x₀) ∧
    (∀ (x' y' : ℝ), (4 * x' + 3 * y' + k = 0) ∧ (y'^2 = 12 * x') → (x' = x₀ ∧ y' = y₀))) ↔
  (k = 27 / 4) := by
sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l571_57123


namespace NUMINAMATH_CALUDE_myrtle_egg_count_l571_57178

/-- The number of eggs Myrtle has after her trip -/
def myrtle_eggs (num_hens : ℕ) (eggs_per_hen : ℕ) (days_gone : ℕ) (neighbor_took : ℕ) (eggs_dropped : ℕ) : ℕ :=
  num_hens * eggs_per_hen * days_gone - neighbor_took - eggs_dropped

/-- Proof that Myrtle has 46 eggs given the conditions -/
theorem myrtle_egg_count : myrtle_eggs 3 3 7 12 5 = 46 := by
  sorry

end NUMINAMATH_CALUDE_myrtle_egg_count_l571_57178


namespace NUMINAMATH_CALUDE_intersection_theorem_l571_57132

/-- The line equation y = 2√2(x-1) -/
def line (x y : ℝ) : Prop := y = 2 * Real.sqrt 2 * (x - 1)

/-- The parabola equation y² = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

/-- Point M with coordinates (-1, m) -/
def point_M (m : ℝ) : ℝ × ℝ := (-1, m)

/-- Function to calculate the dot product of two 2D vectors -/
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  (v1.1 * v2.1) + (v1.2 * v2.2)

/-- Theorem stating that m = √2/2 given the conditions -/
theorem intersection_theorem (A B : ℝ × ℝ) (m : ℝ) :
  line A.1 A.2 →
  line B.1 B.2 →
  parabola A.1 A.2 →
  parabola B.1 B.2 →
  let M := point_M m
  let MA := (A.1 - M.1, A.2 - M.2)
  let MB := (B.1 - M.1, B.2 - M.2)
  dot_product MA MB = 0 →
  m = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_theorem_l571_57132


namespace NUMINAMATH_CALUDE_rays_dog_walks_66_blocks_per_day_l571_57125

/-- Represents the number of blocks in each segment of Ray's walk --/
structure WalkSegments where
  toPark : ℕ
  toHighSchool : ℕ
  toHome : ℕ

/-- Calculates the total blocks walked in one trip --/
def totalBlocksPerTrip (w : WalkSegments) : ℕ :=
  w.toPark + w.toHighSchool + w.toHome

/-- Represents Ray's daily dog walking routine --/
structure DailyWalk where
  segments : WalkSegments
  tripsPerDay : ℕ

/-- Calculates the total blocks walked per day --/
def totalBlocksPerDay (d : DailyWalk) : ℕ :=
  (totalBlocksPerTrip d.segments) * d.tripsPerDay

/-- Theorem stating that Ray's dog walks 66 blocks each day --/
theorem rays_dog_walks_66_blocks_per_day :
  ∀ (d : DailyWalk),
    d.segments.toPark = 4 →
    d.segments.toHighSchool = 7 →
    d.segments.toHome = 11 →
    d.tripsPerDay = 3 →
    totalBlocksPerDay d = 66 :=
by
  sorry


end NUMINAMATH_CALUDE_rays_dog_walks_66_blocks_per_day_l571_57125


namespace NUMINAMATH_CALUDE_angle_calculations_l571_57130

/-- Given points P and Q on the terminal sides of angles α and β, 
    prove the values of sin(α - β) and cos(α + β) -/
theorem angle_calculations (P Q : ℝ × ℝ) (α β : ℝ) : 
  P = (-3, 4) → Q = (-1, -2) → 
  (P.1 = (Real.cos α) * Real.sqrt (P.1^2 + P.2^2)) →
  (P.2 = (Real.sin α) * Real.sqrt (P.1^2 + P.2^2)) →
  (Q.1 = (Real.cos β) * Real.sqrt (Q.1^2 + Q.2^2)) →
  (Q.2 = (Real.sin β) * Real.sqrt (Q.1^2 + Q.2^2)) →
  Real.sin (α - β) = -2 * Real.sqrt 5 / 5 ∧ 
  Real.cos (α + β) = 11 * Real.sqrt 5 / 25 := by
  sorry


end NUMINAMATH_CALUDE_angle_calculations_l571_57130


namespace NUMINAMATH_CALUDE_will_picked_up_38_sticks_l571_57163

/-- The number of sticks originally in the yard -/
def original_sticks : ℕ := 99

/-- The number of sticks left after Will picked some up -/
def remaining_sticks : ℕ := 61

/-- The number of sticks Will picked up -/
def picked_up_sticks : ℕ := original_sticks - remaining_sticks

theorem will_picked_up_38_sticks : picked_up_sticks = 38 := by
  sorry

end NUMINAMATH_CALUDE_will_picked_up_38_sticks_l571_57163


namespace NUMINAMATH_CALUDE_star_equation_solution_l571_57184

-- Define the * operation
def star (a b : ℝ) : ℝ := 4 * a + 2 * b

-- State the theorem
theorem star_equation_solution :
  ∃ y : ℝ, star 3 (star 4 y) = -2 ∧ y = -11.5 := by
  sorry

end NUMINAMATH_CALUDE_star_equation_solution_l571_57184


namespace NUMINAMATH_CALUDE_average_of_solutions_l571_57121

variable (b : ℝ)

def quadratic_equation (x : ℝ) : Prop :=
  3 * x^2 - 6 * b * x + 2 * b = 0

def has_two_real_solutions : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation b x₁ ∧ quadratic_equation b x₂

theorem average_of_solutions :
  has_two_real_solutions b →
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    quadratic_equation b x₁ ∧ 
    quadratic_equation b x₂ ∧
    (x₁ + x₂) / 2 = b :=
by sorry

end NUMINAMATH_CALUDE_average_of_solutions_l571_57121


namespace NUMINAMATH_CALUDE_doughnuts_given_away_l571_57168

/-- Represents the bakery's doughnut sales and production --/
structure BakeryData where
  total_doughnuts : ℕ
  small_box_capacity : ℕ
  large_box_capacity : ℕ
  small_box_price : ℚ
  large_box_price : ℚ
  discount_rate : ℚ
  small_boxes_sold : ℕ
  large_boxes_sold : ℕ
  large_boxes_discounted : ℕ

/-- Theorem stating the number of doughnuts given away --/
theorem doughnuts_given_away (data : BakeryData) : 
  data.total_doughnuts = 300 ∧
  data.small_box_capacity = 6 ∧
  data.large_box_capacity = 12 ∧
  data.small_box_price = 5 ∧
  data.large_box_price = 9 ∧
  data.discount_rate = 1/10 ∧
  data.small_boxes_sold = 20 ∧
  data.large_boxes_sold = 10 ∧
  data.large_boxes_discounted = 5 →
  data.total_doughnuts - 
    (data.small_boxes_sold * data.small_box_capacity + 
     data.large_boxes_sold * data.large_box_capacity) = 60 := by
  sorry

end NUMINAMATH_CALUDE_doughnuts_given_away_l571_57168


namespace NUMINAMATH_CALUDE_marbles_given_l571_57105

theorem marbles_given (drew_initial : ℕ) (marcus_initial : ℕ) (marbles_given : ℕ) : 
  drew_initial = marcus_initial + 24 →
  drew_initial - marbles_given = 25 →
  marcus_initial + marbles_given = 25 →
  marbles_given = 12 := by
sorry

end NUMINAMATH_CALUDE_marbles_given_l571_57105


namespace NUMINAMATH_CALUDE_earnings_difference_is_250_l571_57188

/-- Represents the investment and return ratios for three investors -/
structure InvestmentData where
  a_invest : ℕ
  b_invest : ℕ
  c_invest : ℕ
  a_return : ℕ
  b_return : ℕ
  c_return : ℕ

/-- Calculates the earnings difference between investors B and A -/
def earningsDifference (data : InvestmentData) (total_earnings : ℕ) : ℕ :=
  sorry

/-- Theorem stating the earnings difference between B and A -/
theorem earnings_difference_is_250 :
  let data : InvestmentData := {
    a_invest := 3, b_invest := 4, c_invest := 5,
    a_return := 6, b_return := 5, c_return := 4
  }
  earningsDifference data 7250 = 250 := by sorry

end NUMINAMATH_CALUDE_earnings_difference_is_250_l571_57188


namespace NUMINAMATH_CALUDE_quadratic_equation_condition_l571_57104

theorem quadratic_equation_condition (m : ℝ) : 
  (∀ x, (m - 1) * x^2 + 3 * x - 1 = 0 → (m - 1) ≠ 0) → m ≠ 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_condition_l571_57104


namespace NUMINAMATH_CALUDE_rate_per_axle_above_two_l571_57131

/-- The toll formula for a truck crossing a bridge -/
def toll_formula (rate : ℝ) (x : ℕ) : ℝ :=
  3.50 + rate * (x - 2 : ℝ)

/-- The number of axles on the 18-wheel truck -/
def truck_axles : ℕ := 5

/-- The toll for the 18-wheel truck -/
def truck_toll : ℝ := 5

/-- The rate per axle above 2 axles is $0.50 -/
theorem rate_per_axle_above_two (rate : ℝ) :
  toll_formula rate truck_axles = truck_toll →
  rate = 0.50 := by
  sorry

end NUMINAMATH_CALUDE_rate_per_axle_above_two_l571_57131


namespace NUMINAMATH_CALUDE_opposite_of_negative_fraction_opposite_of_negative_one_over_2023_l571_57199

theorem opposite_of_negative_fraction (n : ℕ) (hn : n ≠ 0) :
  (-(1 : ℚ) / n) + (1 : ℚ) / n = 0 :=
by sorry

theorem opposite_of_negative_one_over_2023 :
  (-(1 : ℚ) / 2023) + (1 : ℚ) / 2023 = 0 :=
by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_fraction_opposite_of_negative_one_over_2023_l571_57199


namespace NUMINAMATH_CALUDE_binomial_cube_plus_one_l571_57140

theorem binomial_cube_plus_one : 7^3 + 3*(7^2) + 3*7 + 2 = 513 := by
  sorry

end NUMINAMATH_CALUDE_binomial_cube_plus_one_l571_57140


namespace NUMINAMATH_CALUDE_system_solution_l571_57126

theorem system_solution (a b c A B C x y z : ℝ) : 
  (x + a * y + a^2 * z = A) ∧ 
  (x + b * y + b^2 * z = B) ∧ 
  (x + c * y + c^2 * z = C) →
  ((A = b + c ∧ B = c + a ∧ C = a + b) → 
    (z = 0 ∧ y = -1 ∧ x = A + b)) ∧
  ((A = b * c ∧ B = c * a ∧ C = a * b) → 
    (z = 1 ∧ y = -(a + b + c) ∧ x = a * b * c)) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l571_57126


namespace NUMINAMATH_CALUDE_inequality_system_solution_l571_57167

theorem inequality_system_solution (x : ℝ) :
  (3 * (x - 1) < 5 * x + 11 ∧ 2 * x > (9 - x) / 4) ↔ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l571_57167


namespace NUMINAMATH_CALUDE_tangent_line_equation_l571_57195

/-- The equation of a cubic curve -/
def f (x : ℝ) : ℝ := x^3 - 2*x + 3

/-- The derivative of the cubic curve -/
def f' (x : ℝ) : ℝ := 3*x^2 - 2

/-- The point on the curve where we want to find the tangent line -/
def x₀ : ℝ := 1

/-- The y-coordinate of the point on the curve -/
def y₀ : ℝ := f x₀

/-- The slope of the tangent line at the point (x₀, y₀) -/
def m : ℝ := f' x₀

theorem tangent_line_equation :
  ∀ x y : ℝ, (x - x₀) = m * (y - y₀) ↔ x - y + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l571_57195


namespace NUMINAMATH_CALUDE_cubic_three_distinct_roots_l571_57157

/-- The cubic equation x^3 - 3x^2 - a = 0 has three distinct real roots if and only if a is in the open interval (-4, 0) -/
theorem cubic_three_distinct_roots (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    x^3 - 3*x^2 - a = 0 ∧
    y^3 - 3*y^2 - a = 0 ∧
    z^3 - 3*z^2 - a = 0) ↔
  -4 < a ∧ a < 0 :=
sorry

end NUMINAMATH_CALUDE_cubic_three_distinct_roots_l571_57157


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l571_57101

theorem quadratic_equation_solution : 
  ∃ x₁ x₂ : ℝ, x₁ = -1 ∧ x₂ = -3/2 ∧ 
  (∀ x : ℝ, 2*x^2 + 5*x + 3 = 0 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l571_57101


namespace NUMINAMATH_CALUDE_function_upper_bound_l571_57193

theorem function_upper_bound 
  (a r : ℝ) 
  (ha : a > 1) 
  (hr : r > 1) 
  (f : ℝ → ℝ) 
  (hf : ∀ x > 0, f x ^ 2 ≤ a * x * f (x / a))
  (hf_small : ∀ x, 0 < x → x < 1 / 2^2005 → f x < 2^2005) :
  ∀ x > 0, f x ≤ a^(1 - r) * x^r := by
sorry

end NUMINAMATH_CALUDE_function_upper_bound_l571_57193


namespace NUMINAMATH_CALUDE_fence_painting_l571_57159

theorem fence_painting (total_length : ℝ) (percentage_difference : ℝ) : 
  total_length = 792 → percentage_difference = 0.2 → 
  ∃ (x : ℝ), x + (1 + percentage_difference) * x = total_length ∧ 
  (1 + percentage_difference) * x = 432 :=
by
  sorry

end NUMINAMATH_CALUDE_fence_painting_l571_57159


namespace NUMINAMATH_CALUDE_hiking_equipment_cost_l571_57144

/-- Calculates the total cost of hiking equipment for Celina --/
theorem hiking_equipment_cost :
  let hoodie_cost : ℚ := 80
  let flashlight_cost : ℚ := 0.2 * hoodie_cost
  let boots_original : ℚ := 110
  let boots_discount : ℚ := 0.1
  let water_filter_original : ℚ := 65
  let water_filter_discount : ℚ := 0.25
  let camping_mat_original : ℚ := 45
  let camping_mat_discount : ℚ := 0.15
  let total_cost : ℚ := 
    hoodie_cost + 
    flashlight_cost + 
    (boots_original * (1 - boots_discount)) + 
    (water_filter_original * (1 - water_filter_discount)) + 
    (camping_mat_original * (1 - camping_mat_discount))
  total_cost = 282 := by
  sorry

end NUMINAMATH_CALUDE_hiking_equipment_cost_l571_57144


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l571_57107

theorem simplify_fraction_product : (144 / 12) * (5 / 90) * (9 / 3) * 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l571_57107


namespace NUMINAMATH_CALUDE_max_value_x3_minus_y3_l571_57182

theorem max_value_x3_minus_y3 (x y : ℝ) 
  (h1 : 3 * (x^3 + y^3) = x + y) 
  (h2 : x + y = 1) : 
  ∃ (max : ℝ), max = 7/27 ∧ ∀ (a b : ℝ), 3 * (a^3 + b^3) = a + b → a + b = 1 → a^3 - b^3 ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_x3_minus_y3_l571_57182


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l571_57153

theorem complex_modulus_problem (z : ℂ) (x : ℝ) 
  (h1 : z * Complex.I = 2 * Complex.I + x)
  (h2 : z.im = 2) : 
  Complex.abs z = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l571_57153


namespace NUMINAMATH_CALUDE_min_alpha_gamma_sum_l571_57139

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the function f
def f (α γ : ℂ) (z : ℂ) : ℂ := (5 + 2*i)*z^3 + (4 + i)*z^2 + α*z + γ

-- State the theorem
theorem min_alpha_gamma_sum (α γ : ℂ) : 
  (f α γ 1).im = 0 → (f α γ i).im = 0 → (f α γ (-1)).im = 0 → 
  Complex.abs α + Complex.abs γ ≥ 7 :=
sorry

end NUMINAMATH_CALUDE_min_alpha_gamma_sum_l571_57139


namespace NUMINAMATH_CALUDE_curve_is_line_l571_57173

/-- The curve represented by the equation (x+2y-1)√(x²+y²-2x+2)=0 is a line. -/
theorem curve_is_line : 
  ∀ (x y : ℝ), (x + 2*y - 1) * Real.sqrt (x^2 + y^2 - 2*x + 2) = 0 ↔ x + 2*y - 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_curve_is_line_l571_57173


namespace NUMINAMATH_CALUDE_outfit_count_l571_57174

/-- The number of shirts available. -/
def num_shirts : ℕ := 8

/-- The number of pairs of pants available. -/
def num_pants : ℕ := 5

/-- The number of ties available. -/
def num_ties : ℕ := 4

/-- The number of belts available. -/
def num_belts : ℕ := 2

/-- An outfit consists of one shirt, one pair of pants, and optionally one tie and/or one belt. -/
def outfit := ℕ × ℕ × Option ℕ × Option ℕ

/-- The total number of possible outfits. -/
def total_outfits : ℕ := num_shirts * num_pants * (num_ties + 1) * (num_belts + 1)

/-- Theorem stating that the total number of possible outfits is 600. -/
theorem outfit_count : total_outfits = 600 := by sorry

end NUMINAMATH_CALUDE_outfit_count_l571_57174


namespace NUMINAMATH_CALUDE_zoo_visitors_l571_57176

theorem zoo_visitors (visitors_saturday : ℕ) (visitors_that_day : ℕ) : 
  visitors_saturday = 3750 →
  visitors_saturday = 3 * visitors_that_day →
  visitors_that_day = 1250 := by
sorry

end NUMINAMATH_CALUDE_zoo_visitors_l571_57176


namespace NUMINAMATH_CALUDE_trig_product_equals_one_l571_57135

theorem trig_product_equals_one :
  let sin30 : ℝ := 1/2
  let cos30 : ℝ := Real.sqrt 3 / 2
  let sin60 : ℝ := Real.sqrt 3 / 2
  let cos60 : ℝ := 1/2
  (1 - 1/sin30) * (1 + 1/cos60) * (1 - 1/cos30) * (1 + 1/sin60) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_product_equals_one_l571_57135


namespace NUMINAMATH_CALUDE_nested_radical_solution_l571_57161

theorem nested_radical_solution :
  ∃ x : ℝ, x > 0 ∧ x^2 = 6 + x ∧ x = 3 := by sorry

end NUMINAMATH_CALUDE_nested_radical_solution_l571_57161


namespace NUMINAMATH_CALUDE_sin_2x_derivative_l571_57177

theorem sin_2x_derivative (x : ℝ) : 
  deriv (fun x => Real.sin (2 * x)) x = 2 * Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_sin_2x_derivative_l571_57177


namespace NUMINAMATH_CALUDE_salt_solution_concentration_l571_57113

/-- Proves that the concentration of a salt solution is 75% when mixed with pure water to form a 15% solution -/
theorem salt_solution_concentration 
  (water_volume : ℝ) 
  (salt_solution_volume : ℝ) 
  (mixture_concentration : ℝ) 
  (h1 : water_volume = 1) 
  (h2 : salt_solution_volume = 0.25) 
  (h3 : mixture_concentration = 15) : 
  (mixture_concentration * (water_volume + salt_solution_volume)) / salt_solution_volume = 75 := by
  sorry

end NUMINAMATH_CALUDE_salt_solution_concentration_l571_57113


namespace NUMINAMATH_CALUDE_quadratic_function_unique_l571_57115

/-- A quadratic function that intersects the x-axis at (-1, 0) and (2, 0), and the y-axis at (0, -2) -/
def f (x : ℝ) : ℝ := x^2 - x - 2

/-- The theorem stating that f is the unique quadratic function satisfying the given conditions -/
theorem quadratic_function_unique :
  (f (-1) = 0) ∧ 
  (f 2 = 0) ∧ 
  (f 0 = -2) ∧ 
  (∀ x : ℝ, ∃ a b c : ℝ, f x = a * x^2 + b * x + c) ∧
  (∀ g : ℝ → ℝ, (g (-1) = 0) → (g 2 = 0) → (g 0 = -2) → 
    (∀ x : ℝ, ∃ a b c : ℝ, g x = a * x^2 + b * x + c) → 
    (∀ x : ℝ, g x = f x)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_unique_l571_57115


namespace NUMINAMATH_CALUDE_product_of_cubes_equality_l571_57108

theorem product_of_cubes_equality : 
  (8 / 9 : ℚ)^3 * (-1 / 3 : ℚ)^3 * (3 / 4 : ℚ)^3 = -8 / 729 := by sorry

end NUMINAMATH_CALUDE_product_of_cubes_equality_l571_57108


namespace NUMINAMATH_CALUDE_parallelogram_bisecting_line_l571_57175

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram -/
structure Parallelogram where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- Represents a line passing through the origin -/
structure Line where
  slope : ℝ

def parallelogram : Parallelogram := {
  v1 := ⟨12, 50⟩,
  v2 := ⟨12, 120⟩,
  v3 := ⟨30, 160⟩,
  v4 := ⟨30, 90⟩
}

/-- Function to check if a line cuts a parallelogram into two congruent polygons -/
def cutsIntoCongruentPolygons (p : Parallelogram) (l : Line) : Prop := sorry

/-- Function to express a real number as a fraction of relatively prime positive integers -/
def asRelativelyPrimeFraction (x : ℝ) : (ℕ × ℕ) := sorry

theorem parallelogram_bisecting_line :
  ∃ (l : Line),
    cutsIntoCongruentPolygons parallelogram l ∧
    l.slope = 5 ∧
    let (m, n) := asRelativelyPrimeFraction l.slope
    m + n = 6 := by sorry

end NUMINAMATH_CALUDE_parallelogram_bisecting_line_l571_57175


namespace NUMINAMATH_CALUDE_expression_equals_one_l571_57190

theorem expression_equals_one (a b c : ℝ) (h : b^2 = a*c) :
  (a^2 * b^2 * c^2) / (a^3 + b^3 + c^3) * (1/a^3 + 1/b^3 + 1/c^3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l571_57190


namespace NUMINAMATH_CALUDE_set_equality_implies_m_values_l571_57192

def A : Set ℝ := {x | x^2 - 3*x - 10 = 0}
def B (m : ℝ) : Set ℝ := {x | m*x - 1 = 0}

theorem set_equality_implies_m_values (m : ℝ) : A ∪ B m = A → m = 0 ∨ m = -1/2 ∨ m = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_m_values_l571_57192


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l571_57109

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  ((1 + i)^10) / (1 - i) = -16 + 16*i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l571_57109


namespace NUMINAMATH_CALUDE_divisibility_proof_l571_57116

def is_valid_number (r b c : Nat) : Prop :=
  r < 10 ∧ b < 10 ∧ c < 10

def number_value (r b c : Nat) : Nat :=
  523000 + r * 100 + b * 10 + c

theorem divisibility_proof (r b c : Nat) 
  (h1 : is_valid_number r b c) 
  (h2 : r * b * c = 180) 
  (h3 : (number_value r b c) % 89 = 0) : 
  (number_value r b c) % 5886 = 0 := by
sorry

end NUMINAMATH_CALUDE_divisibility_proof_l571_57116


namespace NUMINAMATH_CALUDE_bridge_length_l571_57103

/-- The length of a bridge that a train can cross, given the train's length, speed, and crossing time. -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time_s : ℝ) :
  train_length = 140 ∧ 
  train_speed_kmh = 45 ∧ 
  crossing_time_s = 30 →
  (train_speed_kmh * 1000 / 3600 * crossing_time_s) - train_length = 235 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l571_57103


namespace NUMINAMATH_CALUDE_line_AB_equation_line_l_equations_l571_57160

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the midpoint of AB
def midpoint_AB (x y : ℝ) : Prop := x = 3 ∧ y = 2

-- Define the point (2, 0) on line l
def point_on_l (x y : ℝ) : Prop := x = 2 ∧ y = 0

-- Define the area of triangle OMN
def area_OMN : ℝ := 6

-- Theorem for the equation of line AB
theorem line_AB_equation (A B : ℝ × ℝ) :
  parabola A.1 A.2 → parabola B.1 B.2 → midpoint_AB ((A.1 + B.1)/2) ((A.2 + B.2)/2) →
  ∃ (k : ℝ), A.1 - A.2 - k = 0 ∧ B.1 - B.2 - k = 0 :=
sorry

-- Theorem for the equations of line l
theorem line_l_equations :
  ∃ (M N : ℝ × ℝ), parabola M.1 M.2 ∧ parabola N.1 N.2 ∧
  point_on_l 2 0 ∧
  (∃ (m : ℝ), (M.2 = m*M.1 - 2 ∧ N.2 = m*N.1 - 2) ∨
              (M.2 = -m*M.1 - 2 ∧ N.2 = -m*N.1 - 2)) ∧
  area_OMN = 6 :=
sorry

end NUMINAMATH_CALUDE_line_AB_equation_line_l_equations_l571_57160


namespace NUMINAMATH_CALUDE_fifteenth_digit_of_sum_one_seventh_one_eleventh_l571_57180

/-- The decimal representation of a rational number -/
def decimal_representation (q : ℚ) : ℕ → ℕ := sorry

/-- The sum of decimal representations of two rational numbers -/
def sum_decimal_representations (q₁ q₂ : ℚ) : ℕ → ℕ := sorry

/-- The nth digit after the decimal point in a decimal representation -/
def nth_digit_after_decimal (rep : ℕ → ℕ) (n : ℕ) : ℕ := sorry

theorem fifteenth_digit_of_sum_one_seventh_one_eleventh :
  nth_digit_after_decimal (sum_decimal_representations (1/7) (1/11)) 15 = 2 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_digit_of_sum_one_seventh_one_eleventh_l571_57180


namespace NUMINAMATH_CALUDE_bakers_purchase_cost_l571_57118

/-- Calculate the total cost in dollars after discount for a baker's purchase -/
theorem bakers_purchase_cost (flour_price : ℝ) (egg_price : ℝ) (milk_price : ℝ) (soda_price : ℝ)
  (discount_rate : ℝ) (exchange_rate : ℝ) :
  flour_price = 6 →
  egg_price = 12 →
  milk_price = 3 →
  soda_price = 1.5 →
  discount_rate = 0.15 →
  exchange_rate = 1.2 →
  let total_euro := 5 * flour_price + 6 * egg_price + 8 * milk_price + 4 * soda_price
  let discounted_euro := total_euro * (1 - discount_rate)
  let total_dollar := discounted_euro * exchange_rate
  total_dollar = 134.64 := by
sorry

end NUMINAMATH_CALUDE_bakers_purchase_cost_l571_57118


namespace NUMINAMATH_CALUDE_problem_solution_l571_57117

theorem problem_solution (a b c d : ℝ) 
  (h : a^2 + b^2 + c^2 + d^2 - a*b - b*c - c*d - d + 2/5 = 0) : 
  a = 1/5 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l571_57117


namespace NUMINAMATH_CALUDE_sock_selection_l571_57142

theorem sock_selection (n k : ℕ) (h1 : n = 7) (h2 : k = 4) : 
  Nat.choose n k = 35 := by
sorry

end NUMINAMATH_CALUDE_sock_selection_l571_57142


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l571_57128

theorem quadratic_inequality_solution_set (x : ℝ) :
  x^2 - 3*x + 2 > 0 ↔ x < 1 ∨ x > 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l571_57128


namespace NUMINAMATH_CALUDE_expected_rolls_in_year_l571_57170

/-- Represents the outcome of rolling an 8-sided die -/
inductive DieOutcome
  | composite
  | prime
  | reroll

/-- Represents the rules for Bob's breakfast die -/
def breakfastDie : Fin 8 → DieOutcome
  | 1 => DieOutcome.reroll
  | 2 => DieOutcome.prime
  | 3 => DieOutcome.prime
  | 4 => DieOutcome.composite
  | 5 => DieOutcome.prime
  | 6 => DieOutcome.composite
  | 7 => DieOutcome.reroll
  | 8 => DieOutcome.reroll

/-- The probability of getting each outcome -/
def outcomeProb : DieOutcome → Rat
  | DieOutcome.composite => 2/8
  | DieOutcome.prime => 3/8
  | DieOutcome.reroll => 3/8

/-- The number of days in a non-leap year -/
def daysInYear : Nat := 365

/-- Theorem stating the expected number of rolls in a non-leap year -/
theorem expected_rolls_in_year :
  let expectedRollsPerDay := 8/5
  (expectedRollsPerDay * daysInYear : Rat) = 584 := by
  sorry

end NUMINAMATH_CALUDE_expected_rolls_in_year_l571_57170


namespace NUMINAMATH_CALUDE_billionth_term_is_16_l571_57166

def sequence_term (n : ℕ) : ℕ :=
  if n = 0 then 112002
  else
    let prev := sequence_term (n - 1)
    prev + 5 * (prev % 10) - 10 * ((prev % 10) / 10)

def is_cyclic (seq : ℕ → ℕ) (cycle_length : ℕ) : Prop :=
  ∀ n : ℕ, seq (n + cycle_length) = seq n

theorem billionth_term_is_16 :
  is_cyclic sequence_term 42 →
  sequence_term (10^9 % 42) = 16 →
  sequence_term (10^9) = 16 :=
by sorry

end NUMINAMATH_CALUDE_billionth_term_is_16_l571_57166


namespace NUMINAMATH_CALUDE_linear_function_m_greater_than_one_l571_57106

/-- A linear function y = (m+1)x + (m-1) whose graph passes through the first, second, and third quadrants -/
structure LinearFunction (m : ℝ) :=
  (passes_through_quadrants : ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    (x₁ > 0 ∧ y₁ > 0) ∧  -- First quadrant
    (x₂ < 0 ∧ y₂ > 0) ∧  -- Second quadrant
    (x₃ < 0 ∧ y₃ < 0) ∧  -- Third quadrant
    y₁ = (m + 1) * x₁ + (m - 1) ∧
    y₂ = (m + 1) * x₂ + (m - 1) ∧
    y₃ = (m + 1) * x₃ + (m - 1))

/-- Theorem: If a linear function y = (m+1)x + (m-1) has a graph that passes through
    the first, second, and third quadrants, then m > 1 -/
theorem linear_function_m_greater_than_one (m : ℝ) (f : LinearFunction m) : m > 1 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_m_greater_than_one_l571_57106


namespace NUMINAMATH_CALUDE_city_mpg_equals_highway_mpg_l571_57148

/-- The average miles per gallon (mpg) for an SUV on the highway -/
def highway_mpg : ℝ := 12.2

/-- The maximum distance in miles that the SUV can travel on 21 gallons of gasoline -/
def max_distance : ℝ := 256.2

/-- The amount of gasoline in gallons used for the maximum distance -/
def gasoline_amount : ℝ := 21

/-- Theorem: The average mpg in the city is equal to the average mpg on the highway -/
theorem city_mpg_equals_highway_mpg :
  max_distance / gasoline_amount = highway_mpg := by
  sorry


end NUMINAMATH_CALUDE_city_mpg_equals_highway_mpg_l571_57148


namespace NUMINAMATH_CALUDE_chip_notebook_usage_l571_57120

/-- Calculates the number of packs of notebook paper used by Chip over a given number of weeks. -/
def notebook_packs_used (pages_per_day_per_class : ℕ) (classes : ℕ) (days_per_week : ℕ) 
  (weeks : ℕ) (sheets_per_pack : ℕ) : ℕ :=
  let total_pages := pages_per_day_per_class * classes * days_per_week * weeks
  (total_pages + sheets_per_pack - 1) / sheets_per_pack

/-- Proves that Chip uses 3 packs of notebook paper after 6 weeks. -/
theorem chip_notebook_usage : 
  notebook_packs_used 2 5 5 6 100 = 3 := by
  sorry

end NUMINAMATH_CALUDE_chip_notebook_usage_l571_57120


namespace NUMINAMATH_CALUDE_calculate_expression_l571_57179

theorem calculate_expression : 2003^3 - 2001^3 - 6 * 2003^2 + 24 * 1001 = -4 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l571_57179


namespace NUMINAMATH_CALUDE_gcd_56_63_l571_57122

theorem gcd_56_63 : Nat.gcd 56 63 = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_56_63_l571_57122


namespace NUMINAMATH_CALUDE_binomial_1000_500_not_divisible_by_7_l571_57127

theorem binomial_1000_500_not_divisible_by_7 : ¬ (7 ∣ Nat.choose 1000 500) := by
  sorry

end NUMINAMATH_CALUDE_binomial_1000_500_not_divisible_by_7_l571_57127


namespace NUMINAMATH_CALUDE_additional_workers_needed_additional_workers_theorem_l571_57147

theorem additional_workers_needed (initial_workers : ℕ) (initial_parts : ℕ) (initial_hours : ℕ)
  (target_parts : ℕ) (target_hours : ℕ) : ℕ :=
  let production_rate := initial_parts / (initial_workers * initial_hours)
  let required_workers := (target_parts / target_hours) / production_rate
  required_workers - initial_workers

theorem additional_workers_theorem :
  additional_workers_needed 4 108 3 504 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_additional_workers_needed_additional_workers_theorem_l571_57147


namespace NUMINAMATH_CALUDE_normal_distribution_symmetry_regression_coefficient_quadratic_inequality_condition_l571_57146

-- Normal distribution properties
def normal_distribution (μ σ : ℝ) (σ_pos : σ > 0) : ℝ → ℝ := sorry

-- Probability measure
def probability (P : Set ℝ → ℝ) : Set ℝ → ℝ := sorry

-- Statement 1
theorem normal_distribution_symmetry 
  (σ : ℝ) (σ_pos : σ > 0) (P : Set ℝ → ℝ) :
  probability P {x | 0 < x ∧ x < 1} = 0.35 →
  probability P {x | 0 < x ∧ x < 2} = 0.7 := sorry

-- Statement 2
theorem regression_coefficient 
  (c k : ℝ) (y x : ℝ → ℝ) :
  (∀ x, y x = c * Real.exp (k * x)) →
  (∀ x, Real.log (y x) = 0.3 * x + 4) →
  c = Real.exp 4 := sorry

-- Statement 3
theorem quadratic_inequality_condition
  (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) :
  (∀ x > 1, a * x^2 - (a + b - 1) * x + b > 0) ↔ a ≥ b - 1 := sorry

end NUMINAMATH_CALUDE_normal_distribution_symmetry_regression_coefficient_quadratic_inequality_condition_l571_57146
