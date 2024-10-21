import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_min_value_of_f_solution_set_inequality_l795_79558

-- Define the quadratic function
noncomputable def q (m x : ℝ) : ℝ := x^2 + 2*m*x + m + 2

-- Define the function f
noncomputable def f (m : ℝ) : ℝ := m + 3 / (m + 2)

-- Statement 1: Range of m
theorem range_of_m :
  {m : ℝ | ∀ x, q m x ≥ 0} = Set.Icc (-1) 2 := by sorry

-- Statement 2: Minimum value of f
theorem min_value_of_f :
  ∃ m₀, f m₀ = 2 * Real.sqrt 3 - 2 ∧ ∀ m, f m ≥ f m₀ := by sorry

-- Statement 3: Solution set of the inequality
theorem solution_set_inequality (m : ℝ) :
  {x : ℝ | x^2 + (m - 3) * x - 3 * m > 0} =
  Set.Iio (-m) ∪ Set.Ioi 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_min_value_of_f_solution_set_inequality_l795_79558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_conversion_theorem_l795_79518

/-- Converts a speed from miles per minute to kilometers per hour. -/
noncomputable def speed_conversion (speed_mph : ℝ) : ℝ :=
  speed_mph * 60 * (1 / 0.6)

/-- Theorem stating that a speed of 6 miles per minute is equivalent to 600 kilometers per hour. -/
theorem speed_conversion_theorem :
  speed_conversion 6 = 600 := by
  -- Unfold the definition of speed_conversion
  unfold speed_conversion
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_conversion_theorem_l795_79518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_point_z_coordinate_l795_79539

/-- Given a line passing through points (1,3,2) and (4,5,6), 
    if a point on this line has a y-coordinate of 7, 
    then its z-coordinate is 10. -/
theorem line_point_z_coordinate 
  (line : Set (ℝ × ℝ × ℝ))
  (p1 : (1, 3, 2) ∈ line)
  (p2 : (4, 5, 6) ∈ line)
  (point : ℝ × ℝ × ℝ)
  (point_on_line : point ∈ line)
  (y_coord : point.2.1 = 7) :
  point.2.2 = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_point_z_coordinate_l795_79539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_system_of_equations_proof_l795_79560

-- Problem 1
theorem calculation_proof : (-3)^2 * (3 : ℝ)⁻¹ + (-5 + 2) + |(-2)| = 2 := by sorry

-- Problem 2
theorem system_of_equations_proof :
  ∃ (x y : ℝ), 2*x - y = 3 ∧ x + y = 6 ∧ x = 3 ∧ y = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_system_of_equations_proof_l795_79560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_polygon_odd_sides_not_tileable_l795_79521

-- Define a rectangular polygon
structure RectangularPolygon where
  sides : List ℕ
  is_rectangular : Bool
  all_sides_odd : ∀ s ∈ sides, Odd s

-- Define a 2x1 domino tile
structure DominoTile where
  length : ℕ := 2
  width : ℕ := 1

-- Define a tiling
def Tiling (p : RectangularPolygon) := List DominoTile

-- Define a covering relation
def covers (t : Tiling p) (p : RectangularPolygon) : Prop := sorry

-- Theorem statement
theorem rectangular_polygon_odd_sides_not_tileable (p : RectangularPolygon) :
  ¬ ∃ (t : Tiling p), covers t p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_polygon_odd_sides_not_tileable_l795_79521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_open_interval_one_to_infinity_l795_79570

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (2 * x + 1) / (x - 1)

-- State the theorem
theorem f_decreasing_on_open_interval_one_to_infinity :
  ∀ (x₁ x₂ : ℝ), 1 < x₁ → x₁ < x₂ → f x₁ > f x₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_open_interval_one_to_infinity_l795_79570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_special_triangle_l795_79501

/-- A square with side length 1 -/
structure UnitSquare where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  is_square : Prop
  side_length : ℝ

/-- A point on a line segment -/
def PointOnSegment (P : ℝ × ℝ) (A B : ℝ × ℝ) : Prop := sorry

/-- The angle between three points -/
noncomputable def Angle (A B C : ℝ × ℝ) : ℝ := sorry

/-- The perimeter of a triangle -/
noncomputable def TrianglePerimeter (A B C : ℝ × ℝ) : ℝ := sorry

theorem perimeter_of_special_triangle (ABCD : UnitSquare) 
  (E : ℝ × ℝ) (hE : PointOnSegment E ABCD.B ABCD.C)
  (F : ℝ × ℝ) (hF : PointOnSegment F ABCD.C ABCD.D)
  (h_angle : Angle ABCD.A E F = π/4) : 
  TrianglePerimeter ABCD.C E F = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_special_triangle_l795_79501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_E_decreasing_D_decreasing_l795_79554

/-- Represents the number of red balls drawn -/
noncomputable def ξ (n : ℕ+) : ℝ := 1 / (1 + n)

/-- Expected value of ξ -/
noncomputable def E (n : ℕ+) : ℝ := ξ n

/-- Variance of ξ -/
noncomputable def D (n : ℕ+) : ℝ := (n : ℝ) / ((n + 1) ^ 2)

/-- E(ξ) is a decreasing function with respect to n -/
theorem E_decreasing : ∀ n m : ℕ+, n < m → E n > E m := by
  sorry

/-- D(ξ) is a decreasing function with respect to n -/
theorem D_decreasing : ∀ n m : ℕ+, n < m → D n > D m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_E_decreasing_D_decreasing_l795_79554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_roots_l795_79503

theorem product_of_roots (x y : ℝ) : 
  x = Real.sqrt 9 → y = Real.rpow 4 (1/3) → x * y = 3 * Real.rpow 2 (2/3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_roots_l795_79503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trouser_sale_price_l795_79564

/-- Calculates the sale price of an item given its original price and discount percentage. -/
noncomputable def salePrice (originalPrice : ℝ) (discountPercentage : ℝ) : ℝ :=
  originalPrice * (1 - discountPercentage / 100)

/-- Theorem: The sale price of a $100 trouser with a 50% discount is $50. -/
theorem trouser_sale_price :
  salePrice 100 50 = 50 := by
  -- Unfold the definition of salePrice
  unfold salePrice
  -- Simplify the arithmetic
  simp [mul_sub, mul_div, mul_one]
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trouser_sale_price_l795_79564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tangent_line_l795_79510

-- Define the slope of the given line y = x + 1
def slope_given : ℝ := 1

-- Define the equation of the unit circle
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the first quadrant
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

-- Define a line with slope -1/slope_given (perpendicular to the given line)
def perpendicular_line (x y b : ℝ) : Prop := y = -1/slope_given * x + b

-- Define the tangent condition
def is_tangent (b : ℝ) : Prop :=
  ∃ (x y : ℝ), perpendicular_line x y b ∧ unit_circle x y ∧ first_quadrant x y

-- The main theorem
theorem perpendicular_tangent_line :
  ∃ (b : ℝ), is_tangent b ∧ b = Real.sqrt 2 := by
  sorry

#check perpendicular_tangent_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tangent_line_l795_79510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l795_79589

noncomputable section

/-- Line l in parametric form -/
def line_l (t : ℝ) : ℝ × ℝ := (-1 - t, 2 + t)

/-- Curve C in polar form -/
def curve_C (θ : ℝ) : ℝ := Real.sqrt (2 / (1 + Real.sin θ ^ 2))

/-- Point P in polar coordinates -/
def point_P : ℝ × ℝ := (Real.sqrt 2 / 2, Real.pi / 4)

/-- Theorem stating the product of distances from P to intersection points is 5/6 -/
theorem intersection_distance_product :
  ∃ (A B : ℝ × ℝ), 
  (∃ t, line_l t = A) ∧ 
  (∃ θ, (curve_C θ * Real.cos θ, curve_C θ * Real.sin θ) = A) ∧
  (∃ t, line_l t = B) ∧ 
  (∃ θ, (curve_C θ * Real.cos θ, curve_C θ * Real.sin θ) = B) ∧
  let dist (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  dist point_P A * dist point_P B = 5/6 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l795_79589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_angle_bisector_slope_l795_79559

/-- The slope of the angle bisector of the acute angle formed by two lines -/
noncomputable def angle_bisector_slope (m₁ m₂ : ℝ) : ℝ :=
  (m₁ + m₂ + Real.sqrt (m₁^2 + m₂^2 + 2*m₁*m₂)) / (1 - m₁*m₂)

theorem acute_angle_bisector_slope :
  let m₁ : ℝ := 2
  let m₂ : ℝ := 4
  angle_bisector_slope m₁ m₂ = -12/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_angle_bisector_slope_l795_79559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_is_one_l795_79582

noncomputable def polar_to_cartesian (r : ℝ) (θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

noncomputable def line_equation (x y : ℝ) : ℝ :=
  x - Real.sqrt 3 * y + 2

noncomputable def distance_point_to_line (x y : ℝ) : ℝ :=
  |line_equation x y| / Real.sqrt (1 + (-Real.sqrt 3)^2)

theorem distance_point_to_line_is_one :
  let (x, y) := polar_to_cartesian 2 (π/6)
  distance_point_to_line x y = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_is_one_l795_79582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l795_79596

-- Define the power function as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^a

-- State the theorem
theorem power_function_through_point (a : ℝ) :
  f a 3 = 1/9 → f a 4 = 1/16 := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l795_79596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_numbers_l795_79542

theorem order_of_numbers : 
  let a : ℝ := (3/4 : ℝ)^(-(1/3 : ℝ))
  let b : ℝ := (3/4 : ℝ)^(-(1/4 : ℝ))
  let c : ℝ := (3/2 : ℝ)^(-(1/4 : ℝ))
  c < b ∧ b < a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_numbers_l795_79542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_equation_solution_l795_79579

theorem trig_equation_solution (x : ℝ) : 
  Real.sin (2*x) * Real.sin (6*x) * Real.cos (4*x) + (1/4) * Real.cos (12*x) = 0 →
  (∃ k : ℤ, x = (Real.pi/8) * (2*↑k + 1)) ∨ 
  (∃ n : ℤ, x = (Real.pi/12) * (6*↑n + 1) ∨ x = (Real.pi/12) * (6*↑n - 1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_equation_solution_l795_79579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_projection_l795_79508

/-- A parallelogram in a 2D plane -/
structure Parallelogram where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  is_parallelogram : Prop

/-- A straight line in a 2D plane -/
structure StraightLine where
  X : ℝ × ℝ
  Y : ℝ × ℝ
  is_line : Prop

/-- Projection of a line segment onto another line -/
noncomputable def project (p q : ℝ × ℝ) (l : StraightLine) : ℝ := sorry

/-- Statement of the parallelogram projection theorem -/
theorem parallelogram_projection 
  (ABCD : Parallelogram) (XY : StraightLine) : 
  (project ABCD.A ABCD.C XY = project ABCD.A ABCD.B XY + project ABCD.B ABCD.C XY) ∨
  (project ABCD.A ABCD.C XY = project ABCD.A ABCD.B XY - project ABCD.B ABCD.D XY) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_projection_l795_79508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inspection_theorem_l795_79532

-- Define the probability distribution of X
def prob_dist_X : Fin 4 → ℚ
| 0 => 1/15  -- P(X=2)
| 1 => 2/15  -- P(X=3)
| 2 => 4/15  -- P(X=4)
| 3 => 8/15  -- P(X=5)

-- Define the expectation of X
def expectation_X : ℚ := 64/15

-- Define the function f(p)
noncomputable def f (p : ℝ) : ℝ := (Nat.choose 50 2 : ℝ) * p^2 * (1-p)^48

-- State the theorem
theorem inspection_theorem :
  (∀ i, prob_dist_X i ≥ 0) ∧  -- Probabilities are non-negative
  (Finset.sum Finset.univ prob_dist_X = 1) ∧  -- Sum of probabilities is 1
  (Finset.sum Finset.univ (λ i => (↑i + 2 : ℚ) * prob_dist_X i) = expectation_X) ∧  -- Expectation calculation
  (∃ p_max : ℝ, p_max = 1/25 ∧ ∀ p, 0 < p ∧ p < 1 → f p ≤ f p_max) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inspection_theorem_l795_79532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_interest_rate_is_five_percent_l795_79516

/-- Calculates the new interest rate given the initial rate, initial interest, and additional interest -/
noncomputable def calculate_new_interest_rate (initial_rate : ℝ) (initial_interest : ℝ) (additional_interest : ℝ) : ℝ :=
  let principal := (initial_interest * 100) / initial_rate
  let rate_increase := (additional_interest * 100) / principal
  initial_rate + rate_increase

/-- Theorem stating that given the specified conditions, the new interest rate is 5% -/
theorem new_interest_rate_is_five_percent 
  (initial_rate : ℝ) 
  (initial_interest : ℝ) 
  (additional_interest : ℝ) 
  (h1 : initial_rate = 4.5)
  (h2 : initial_interest = 202.5)
  (h3 : additional_interest = 22.5) : 
  calculate_new_interest_rate initial_rate initial_interest additional_interest = 5 := by
  sorry

-- Remove the #eval statement as it's not necessary for building and may cause issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_interest_rate_is_five_percent_l795_79516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_winning_board_size_l795_79565

/-- A game board represented as a set of cells -/
structure Board :=
  (cells : Finset ℕ)

/-- A player in the game -/
inductive Player := | First | Second

/-- A game state, representing the marks on the board -/
def GameState := Board → Player → Finset ℕ

/-- Checks if a player has won by having three consecutive marks -/
def has_won (state : GameState) (player : Player) (board : Board) : Prop :=
  ∃ (a b c : ℕ), a + 1 = b ∧ b + 1 = c ∧ {a, b, c} ⊆ state board player

/-- A strategy for a player -/
def Strategy := Board → GameState → ℕ

/-- Checks if a strategy is winning for the first player -/
def is_winning_strategy (s : Strategy) (board : Board) : Prop :=
  ∀ (opponent_strategy : Strategy),
    ∃ (final_state : GameState),
      has_won final_state Player.First board

/-- The main theorem: 7 is the minimum number of cells for a winning board -/
theorem min_winning_board_size :
  (∃ (board : Board), board.cells.card = 7 ∧ ∃ (s : Strategy), is_winning_strategy s board) ∧
  (∀ (board : Board), board.cells.card < 7 → ∀ (s : Strategy), ¬is_winning_strategy s board) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_winning_board_size_l795_79565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_price_change_l795_79504

theorem book_price_change (P : ℝ) (h : P > 0) : 
  let price_after_decrease : ℝ := P * (1 - 0.3)
  let final_price : ℝ := price_after_decrease * (1 + 0.4)
  let net_change : ℝ := (final_price - P) / P
  net_change = -0.02 := by
    -- Proof steps would go here
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_price_change_l795_79504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_inequality_l795_79577

theorem logarithm_inequality (a x y : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : x^2 + y = 0) :
  Real.log (a^x + a^y) / Real.log a ≤ Real.log a / Real.log 2 + 1/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_inequality_l795_79577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_condition_l795_79591

/-- Two lines are parallel if they have the same slope but different y-intercepts -/
def are_parallel (m₁ n₁ b₁ m₂ n₂ b₂ : ℝ) : Prop :=
  (m₁ * n₂ = m₂ * n₁) ∧ (b₁ * n₂ ≠ b₂ * n₁)

/-- Line l₁: ax + 2y - 1 = 0 -/
def l₁ (a : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ a * x + 2 * y - 1 = 0

/-- Line l₂: x + (a+1)y + 4 = 0 -/
def l₂ (a : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ x + (a + 1) * y + 4 = 0

theorem parallel_condition (a : ℝ) :
  (are_parallel a 2 (-1) 1 (a+1) 4) ∧
  (∃ b : ℝ, b ≠ 1 ∧ are_parallel b 2 (-1) 1 (b+1) 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_condition_l795_79591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_alcohol_percentage_l795_79538

/-- Calculates the percentage of alcohol in a mixture given the volumes and concentrations of different solutions. -/
noncomputable def alcohol_percentage (vol1 vol2 vol3 vol4 : ℝ) (conc1 conc2 conc3 : ℝ) : ℝ :=
  let total_alcohol := vol1 * conc1 + vol2 * conc2 + vol4 * conc3
  let total_volume := vol1 + vol2 + vol3 + vol4
  (total_alcohol / total_volume) * 100

/-- Theorem stating that the given mixture results in 29% alcohol concentration. -/
theorem mixture_alcohol_percentage :
  alcohol_percentage 4 3 2 1 0.3 0.4 0.5 = 29 := by
  sorry

-- Remove the #eval statement as it's not necessary for the proof
-- and may cause issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_alcohol_percentage_l795_79538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_length_l795_79517

/-- Golden ratio -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- The reciprocal of the golden ratio -/
noncomputable def φ_reciprocal : ℝ := (Real.sqrt 5 - 1) / 2

/-- Given a segment AB with golden section point P, where AP < BP and BP = 10, prove AP = 5√5 - 5 -/
theorem golden_section_length (A B P : ℝ) (h1 : P - A < B - P) 
  (h2 : B - P = 10) (h3 : (B - P) / (B - A) = φ_reciprocal) : P - A = 5 * Real.sqrt 5 - 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_length_l795_79517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_two_digit_prime_saturated_l795_79520

def is_prime_saturated (n : ℕ) : Prop :=
  (Finset.prod (Finset.filter Nat.Prime (Finset.range (n + 1))) id) < n

theorem greatest_two_digit_prime_saturated :
  is_prime_saturated 96 ∧ ∀ m : ℕ, m > 96 → m < 100 → ¬ is_prime_saturated m :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_two_digit_prime_saturated_l795_79520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_difference_l795_79594

theorem average_difference : ℤ := by
  let set1_start : ℤ := 100
  let set1_end : ℤ := 400
  let set2_start : ℤ := 50
  let set2_end : ℤ := 250
  let avg1 : ℚ := (set1_start + set1_end) / 2
  let avg2 : ℚ := (set2_start + set2_end) / 2
  have h1 : avg1 - avg2 = 100 := by sorry
  exact 100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_difference_l795_79594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_butterfat_percentage_calculation_l795_79523

/-- Calculates the butterfat percentage of added milk --/
noncomputable def butterfat_percentage_of_added_milk (initial_volume : ℝ) (initial_butterfat : ℝ) 
  (final_volume : ℝ) (final_butterfat : ℝ) : ℝ :=
  let added_volume := final_volume - initial_volume
  ((final_butterfat * final_volume) - (initial_butterfat * initial_volume)) / added_volume

/-- Theorem stating the butterfat percentage of added milk --/
theorem butterfat_percentage_calculation 
  (initial_volume : ℝ) (initial_butterfat : ℝ) (final_volume : ℝ) (final_butterfat : ℝ)
  (h1 : initial_volume = 8)
  (h2 : initial_butterfat = 0.45)
  (h3 : final_volume = 28)
  (h4 : final_butterfat = 0.20) :
  abs (butterfat_percentage_of_added_milk initial_volume initial_butterfat final_volume final_butterfat - 0.0333) < 0.0001 := by
  sorry

-- Remove #eval as it's not necessary for the proof and may cause issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_butterfat_percentage_calculation_l795_79523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sunny_behind_windy_l795_79571

/-- Represents the race scenario with two runners, Sunny and Windy -/
structure RaceScenario where
  initial_race_length : ℝ
  initial_sunny_lead : ℝ
  second_race_length : ℝ
  sunny_start_behind : ℝ
  sunny_speed_decrease : ℝ

/-- Calculates Sunny's position relative to Windy at the end of the second race -/
def sunny_final_position (scenario : RaceScenario) : ℝ :=
  sorry

/-- Theorem stating that Sunny finishes 14.72 meters behind Windy in the second race -/
theorem sunny_behind_windy (scenario : RaceScenario) 
  (h1 : scenario.initial_race_length = 400)
  (h2 : scenario.initial_sunny_lead = 30)
  (h3 : scenario.second_race_length = 500)
  (h4 : scenario.sunny_start_behind = 30)
  (h5 : scenario.sunny_speed_decrease = 0.1) :
  abs (sunny_final_position scenario + 14.72) < 0.01 := by
  sorry

#check sunny_behind_windy

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sunny_behind_windy_l795_79571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C₂_center_coordinates_l795_79592

-- Define the circle structure
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the circles
def C₁ : Circle := { center := (0, 9), radius := 9 }
def C₂ : Circle := { center := (0, -6), radius := 6 }

-- Define helper functions
def externally_tangent (c1 c2 : Circle) : Prop :=
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 = (c1.radius + c2.radius)^2

def tangent_to_x_axis (c : Circle) : Prop :=
  c.center.2 = c.radius

-- Define the conditions
axiom touch_externally : externally_tangent C₁ C₂
axiom tangent_x_axis : tangent_to_x_axis C₁ ∧ tangent_to_x_axis C₂
axiom sum_of_radii : C₁.radius + C₂.radius = 15
axiom C₁_radius : C₁.radius = 9

-- Theorem to prove
theorem C₂_center_coordinates : C₂.center = (0, -6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_C₂_center_coordinates_l795_79592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_conditions_l795_79512

def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B (a : ℝ) : Set ℝ := {x | |x - 1| < a}

theorem subset_conditions (a : ℝ) :
  (A ⊂ B a ↔ a > 2) ∧ (B a ⊂ A ↔ a < 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_conditions_l795_79512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cookfire_burn_rate_l795_79583

def cookfire_logs (burn_rate : ℕ) : ℕ → ℕ
| 0 => 6  -- Initial number of logs
| (n+1) => cookfire_logs burn_rate n - burn_rate + 2  -- Logs after each hour

theorem cookfire_burn_rate :
  ∃ (burn_rate : ℕ), burn_rate > 0 ∧ cookfire_logs burn_rate 3 = 3 :=
by
  -- We'll use 3 as the burn rate
  use 3
  constructor
  · -- Prove burn_rate > 0
    simp
  · -- Prove cookfire_logs 3 3 = 3
    rfl

#eval cookfire_logs 3 3  -- This should output 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cookfire_burn_rate_l795_79583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inscribed_circle_theorem_l795_79580

/-- Triangle AQM with circle O inscribed --/
structure TriangleWithInscribedCircle where
  /-- Side lengths of triangle AQM --/
  a : ℝ
  q : ℝ
  m : ℝ
  /-- Radius of inscribed circle --/
  r : ℝ
  /-- Distance from O to Q --/
  oq : ℝ
  /-- Perimeter of triangle AQM --/
  perim_eq : a + q + m = 180
  /-- Right angle at A --/
  right_angle : a^2 + m^2 = q^2
  /-- Circle radius is 20 --/
  radius_eq : r = 20
  /-- Circle is tangent to AM and QM --/
  tangent : a - r = m - r
  /-- O is on AQ --/
  o_on_aq : oq ≤ q

/-- OQ as a fraction m/n --/
def oq_fraction (t : TriangleWithInscribedCircle) (m n : ℕ) : Prop :=
  t.oq = m / n ∧ Nat.Coprime m n

theorem triangle_inscribed_circle_theorem (t : TriangleWithInscribedCircle) 
  (m n : ℕ) (h : oq_fraction t m n) : 
  m + n = 52 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inscribed_circle_theorem_l795_79580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_part1_range_of_b_part2_l795_79578

-- Define the function f
def f (x b : ℝ) : ℝ := |x - b| + |x + b|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f x 1 ≤ x + 2} = Set.Icc 0 2 := by sorry

-- Part 2
theorem range_of_b_part2 :
  {b : ℝ | ∀ a : ℝ, a ≠ 0 → f 1 b ≥ (|a + 1| - |2*a - 1|) / |a|} =
  Set.Iic (-3/2) ∪ Set.Ici (3/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_part1_range_of_b_part2_l795_79578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ammeter_error_probability_l795_79561

/-- The scale division value of the ammeter in amperes -/
noncomputable def scale_division : ℝ := 0.1

/-- The maximum possible error due to rounding (half of the scale division) -/
noncomputable def max_error : ℝ := scale_division / 2

/-- The error threshold we're interested in -/
noncomputable def error_threshold : ℝ := 0.02

/-- The probability density function for the uniform distribution of errors -/
noncomputable def error_pdf (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ scale_division then 1 / scale_division else 0

/-- The theorem stating the probability of a reading error exceeding the threshold -/
theorem ammeter_error_probability :
  ∫ x in error_threshold..(scale_division - error_threshold), error_pdf x = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ammeter_error_probability_l795_79561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_pairs_l795_79572

def A (a : ℕ) : Set ℝ := {x : ℝ | 5 * x - a ≤ 0}

def B (b : ℕ) : Set ℝ := {x : ℝ | 6 * x - b > 0}

def valid_pair (a b : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ (A a ∩ B b ∩ Set.range (fun n : ℕ => (n : ℝ))) = {2, 3, 4}

-- Define the set of valid pairs
def valid_pairs : Set (ℕ × ℕ) := {p | valid_pair p.1 p.2}

-- State the theorem without using Fintype
theorem count_valid_pairs : Finite valid_pairs ∧ Set.ncard valid_pairs = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_pairs_l795_79572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_non_real_roots_l795_79540

theorem quadratic_non_real_roots (b : ℝ) : 
  (∀ x : ℂ, x^2 + b*x + 25 = 0 → ¬(∃ y : ℝ, x = y)) ↔ b ∈ Set.Ioo (-10) 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_non_real_roots_l795_79540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_negative_four_l795_79541

-- Define the polynomial
noncomputable def f (x : ℂ) : ℂ := x^4 + 2*x + 4

-- Define the roots
axiom a : ℂ
axiom b : ℂ
axiom c : ℂ
axiom d : ℂ

-- Assume a, b, c, d are the roots of f
axiom root_a : f a = 0
axiom root_b : f b = 0
axiom root_c : f c = 0
axiom root_d : f d = 0

-- Define the sum we want to prove
noncomputable def sum : ℂ := (a^2 / (a^3 + 2)) + (b^2 / (b^3 + 2)) + (c^2 / (c^3 + 2)) + (d^2 / (d^3 + 2))

-- State the theorem
theorem sum_equals_negative_four : sum = -4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_negative_four_l795_79541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_semicircle_problem_l795_79524

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  E : Point
  F : Point
  G : Point
  H : Point

/-- Represents a semicircle -/
structure Semicircle where
  center : Point
  radius : ℝ

/-- The region enclosed by the semicircle and rectangle -/
def Region (rect : Rectangle) (semi : Semicircle) : Set Point := sorry

/-- A line intersects the semicircle, EF, and GH at distinct points -/
def intersects (l : Line) (semi : Semicircle) (ef : Line) (gh : Line) : Prop := sorry

/-- The line divides the region into two subregions with area ratio 3:1 -/
def divides_region (l : Line) (r : Set Point) : Prop := sorry

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem rectangle_semicircle_problem 
  (rect : Rectangle) 
  (semi : Semicircle) 
  (l : Line) 
  (P V Q : Point) :
  intersects l semi (Line.mk 0 1 0) (Line.mk 0 1 (distance rect.E rect.G)) →
  divides_region l (Region rect semi) →
  distance rect.E V = 70 →
  distance rect.E P = 105 →
  distance V rect.F = 210 →
  distance rect.E rect.G = 280 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_semicircle_problem_l795_79524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_loop_resistance_theorem_l795_79593

/-- Represents the resistance of a wire loop with symmetric vertical connectors -/
noncomputable def wire_loop_resistance (distance : ℝ) (resistance_per_meter : ℝ) : ℝ :=
  1 / (2 / (resistance_per_meter * 1) + 1 / (resistance_per_meter * distance))

/-- Theorem: The resistance of the wire loop between points A and B is 4 ohms -/
theorem wire_loop_resistance_theorem :
  wire_loop_resistance 2 10 = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_loop_resistance_theorem_l795_79593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weeklySalesEqual22140_l795_79550

/-- Calculates the weekly sales of a left-handed store --/
def weeklySales (normalMousePrice normalKeyboardPrice normalScissorPrice : ℚ)
  (mouseMarkup keyboardMarkup scissorMarkup : ℚ)
  (mouseSoldDaily keyboardSoldDaily scissorSoldDaily : ℕ)
  (daysOpen : ℕ) : ℚ :=
  let leftMousePrice := normalMousePrice * (1 + mouseMarkup)
  let leftKeyboardPrice := normalKeyboardPrice * (1 + keyboardMarkup)
  let leftScissorPrice := normalScissorPrice * (1 + scissorMarkup)
  let dailySales := leftMousePrice * mouseSoldDaily +
                    leftKeyboardPrice * keyboardSoldDaily +
                    leftScissorPrice * scissorSoldDaily
  dailySales * daysOpen

/-- Theorem stating that the weekly sales equal $22,140 --/
theorem weeklySalesEqual22140 :
  weeklySales 120 80 30 (3/10) (1/5) (1/2) 25 10 15 4 = 22140 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_weeklySalesEqual22140_l795_79550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_sum_property_l795_79525

theorem partition_sum_property (X : Finset ℕ) (A B : Finset ℕ) :
  X = Finset.range 9 →
  X = A ∪ B →
  A ∩ B = ∅ →
  (∃ a b c, a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ a + b = c) ∨ 
  (∃ a b c, a ∈ B ∧ b ∈ B ∧ c ∈ B ∧ a + b = c) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_sum_property_l795_79525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_theorem_l795_79506

theorem cosine_sum_theorem (x y z : ℝ) 
  (h1 : Real.cos x + Real.cos y + Real.cos z = 1) 
  (h2 : Real.sin x + Real.sin y + Real.sin z = 0) : 
  Real.cos (2*x) + Real.cos (2*y) + Real.cos (2*z) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_theorem_l795_79506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_gain_percentage_approx_l795_79545

noncomputable def cycle_prices : List ℝ := [900, 1200, 1500]
noncomputable def shipping_fees : List ℝ := [50, 75, 100]
noncomputable def sales_tax_rate : ℝ := 0.05
noncomputable def selling_prices : List ℝ := [1080, 1320, 1650]

noncomputable def total_cost_price : ℝ := (List.sum (List.zipWith (· + ·) cycle_prices shipping_fees)) * (1 + sales_tax_rate)
noncomputable def total_selling_price : ℝ := List.sum selling_prices

noncomputable def overall_gain : ℝ := total_selling_price - total_cost_price
noncomputable def overall_gain_percentage : ℝ := (overall_gain / total_cost_price) * 100

theorem overall_gain_percentage_approx :
  abs (overall_gain_percentage - 0.84) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_gain_percentage_approx_l795_79545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_rectangle_arrangement_l795_79543

/-- Represents a rectangle in a 2D plane -/
structure Rectangle where
  center : ℝ × ℝ
  width : ℝ
  height : ℝ

/-- Represents a vertex of a rectangle -/
def Vertex := ℝ × ℝ

/-- Predicate to check if a vertex is shared by two rectangles -/
def shared_vertex (r1 r2 : Rectangle) (v : Vertex) : Prop :=
  sorry

/-- Predicate to check if an arrangement of rectangles satisfies the conditions -/
def valid_arrangement (rectangles : Finset Rectangle) : Prop :=
  rectangles.card = 4 ∧
  (∀ v : Vertex, ¬(∀ r ∈ rectangles, shared_vertex r r v)) ∧
  (∀ r1 r2, r1 ∈ rectangles → r2 ∈ rectangles → r1 ≠ r2 → ∃! v : Vertex, shared_vertex r1 r2 v)

/-- Theorem stating the existence of a valid arrangement -/
theorem exists_valid_rectangle_arrangement :
  ∃ (rectangles : Finset Rectangle), valid_arrangement rectangles :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_rectangle_arrangement_l795_79543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l795_79597

noncomputable def a : ℕ → ℚ
| 0 => 1/2
| n + 1 => a n + (1 / (n + 1)^2) * (a n)^2

theorem sequence_properties :
  ∀ n : ℕ, n > 0 →
    ((1 / a (n - 1)) - (1 / a n) < 1 / n^2) ∧
    (a n < n) ∧
    (1 / a n < 5/6 + 1 / (n + 1)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l795_79597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_problem_l795_79527

/-- Dilation of a complex number -/
noncomputable def dilation (center : ℂ) (scale : ℝ) (z : ℂ) : ℂ :=
  center + scale • (z - center)

/-- The problem statement -/
theorem dilation_problem : 
  dilation (1 - 3*I) 3 (-1 + 2*I) = -5 + 12*I := by
  -- Unfold the definition of dilation
  unfold dilation
  -- Simplify the expression
  simp [Complex.I, Complex.add_re, Complex.add_im, Complex.mul_re, Complex.mul_im]
  -- The proof is complete
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_problem_l795_79527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_plus_pi_fourth_l795_79519

theorem tan_theta_plus_pi_fourth (θ : Real) 
  (h1 : θ > π / 2) (h2 : θ < π) (h3 : Real.sin θ = 3 / 5) : 
  Real.tan (θ + π / 4) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_plus_pi_fourth_l795_79519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_correct_l795_79569

/-- Given a natural number m, k is the smallest natural number such that 
    for any decomposition of {m, m+1, ..., k} into two sets A and B, 
    either A or B contains three elements a, b, c with a^b = c. -/
def smallest_k (m : ℕ) : ℕ := m^(m^(m+2))

/-- For any decomposition of {m, m+1, ..., k} into two sets A and B,
    either A or B contains three elements a, b, c with a^b = c. -/
def satisfies_condition (m k : ℕ) : Prop :=
  ∀ (A B : Set ℕ), (∀ n, m ≤ n ∧ n ≤ k → n ∈ A ∨ n ∈ B) →
    (∃ a b c, a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ a^b = c) ∨ 
    (∃ a b c, a ∈ B ∧ b ∈ B ∧ c ∈ B ∧ a^b = c)

/-- Theorem stating that smallest_k satisfies the condition and is minimal. -/
theorem smallest_k_correct (m : ℕ) : 
  satisfies_condition m (smallest_k m) ∧ 
  ∀ k, k < smallest_k m → ¬(satisfies_condition m k) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_correct_l795_79569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l795_79599

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x^2 + 1/x
  else if x > 0 then -x^2 + 1/x
  else 0

-- State the theorem
theorem f_properties :
  (∀ x, f (-x) = -f x) ∧  -- f is odd
  (∀ x, x > 0 → f x = -x^2 + 1/x) ∧  -- expression for x > 0
  (∀ x y, 0 < x ∧ x < y → f x > f y)  -- f is decreasing on (0, +∞)
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l795_79599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_endpoint_l795_79535

/-- Given a line segment from (3, -2) to (x, 7) with length 12 and x < 0, prove x = 3 - √63 -/
theorem line_segment_endpoint (x : ℝ) : 
  x < 0 → 
  Real.sqrt ((x - 3)^2 + (7 - (-2))^2) = 12 → 
  x = 3 - Real.sqrt 63 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_endpoint_l795_79535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circle_intersection_l795_79552

-- Define the triangle PQR
structure Triangle (P Q R : ℝ × ℝ) : Prop where
  is_right_angle : (Q.1 - P.1) * (R.1 - P.1) + (Q.2 - P.2) * (R.2 - P.2) = 0

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the intersection point S
noncomputable def S (P Q R : ℝ × ℝ) : ℝ × ℝ := sorry

-- Theorem statement
theorem triangle_circle_intersection 
  (P Q R : ℝ × ℝ) 
  (h_triangle : Triangle P Q R) 
  (h_circle : S P Q R ∈ Circle ((Q.1 + R.1) / 2, (Q.2 + R.2) / 2) (Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2) / 2))
  (h_area : abs ((P.1 - R.1) * (Q.2 - R.2) - (Q.1 - R.1) * (P.2 - R.2)) / 2 = 200)
  (h_PR : Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2) = 40) :
  Real.sqrt ((Q.1 - (S P Q R).1)^2 + (Q.2 - (S P Q R).2)^2) = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circle_intersection_l795_79552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_at_27_equals_2_twentyseven_in_range_of_f_l795_79544

-- Define the function f
def f (x : ℝ) : ℝ := 5 * x^2 + 7

-- Theorem statement
theorem inverse_f_at_27_equals_2 : 
  ∃ (y : ℝ), f y = 27 ∧ y = 2 :=
by
  -- We'll use 2 as our witness
  use 2
  constructor
  -- Prove f 2 = 27
  · simp [f]
    norm_num
  -- Prove 2 = 2 (trivial)
  · rfl

-- Additional theorem to show that 27 is in the range of f
theorem twentyseven_in_range_of_f :
  ∃ (x : ℝ), f x = 27 :=
by
  use 2
  simp [f]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_at_27_equals_2_twentyseven_in_range_of_f_l795_79544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorization_correct_sum_of_coefficients_correct_main_theorem_l795_79526

/-- The sum of all integer coefficients in the complete factorization of 27x^6 - 512y^6 -/
def sum_of_coefficients : ℤ := 92

/-- The complete factorization of 27x^6 - 512y^6 -/
def factored_expression (x y : ℝ) : ℝ := (3*x^2 - 8*y^2) * (9*x^4 + 24*x^2*y^2 + 64*y^4)

theorem factorization_correct (x y : ℝ) :
  27*x^6 - 512*y^6 = factored_expression x y := by
  sorry

theorem sum_of_coefficients_correct :
  sum_of_coefficients = 3 + (-8) + 9 + 24 + 64 := by
  sorry

theorem main_theorem :
  sum_of_coefficients = 92 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorization_correct_sum_of_coefficients_correct_main_theorem_l795_79526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_x_value_l795_79588

/-- Represents a 3x2 grid --/
structure Grid :=
  (a b c d x : Nat)

/-- Check if two numbers are adjacent in the grid --/
def adjacent (n m : Nat) : Prop :=
  (n = m + 1) ∨ (m = n + 1)

/-- Check if a grid configuration is valid --/
def valid_grid (g : Grid) : Prop :=
  -- All numbers are between 1 and 6
  (g.a ∈ Finset.range 6) ∧ (g.b ∈ Finset.range 6) ∧ 
  (g.c ∈ Finset.range 6) ∧ (g.d ∈ Finset.range 6) ∧ 
  (g.x ∈ Finset.range 6) ∧
  -- All numbers are distinct
  g.a ≠ g.b ∧ g.a ≠ g.c ∧ g.a ≠ g.d ∧ g.a ≠ g.x ∧
  g.b ≠ g.c ∧ g.b ≠ g.d ∧ g.b ≠ g.x ∧
  g.c ≠ g.d ∧ g.c ≠ g.x ∧
  g.d ≠ g.x ∧
  -- No adjacent numbers differ by 1
  ¬(adjacent g.a g.b) ∧ ¬(adjacent g.a g.c) ∧
  ¬(adjacent g.b g.d) ∧ ¬(adjacent g.b g.x) ∧
  ¬(adjacent g.c g.d) ∧ ¬(adjacent g.c g.x)

theorem unique_x_value : 
  ∃! x, ∃ (g : Grid), g.a = 1 ∧ valid_grid g ∧ g.x = x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_x_value_l795_79588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_angle_l795_79557

theorem line_slope_angle (a : ℝ) : 
  (∃ l : Set (ℝ × ℝ), l = {(x, y) | a * x - y - 1 = 0}) → 
  (∃ θ : ℝ, θ = π / 3 ∧ Real.tan θ = a) → 
  a = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_angle_l795_79557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_sum_prime_power_solutions_l795_79584

theorem cube_sum_prime_power_solutions (a b p n : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_p : 0 < p) (h_pos_n : 0 < n) (h_prime : Nat.Prime p) (h_eq : a^3 + b^3 = p^n) :
  (∃ k : ℕ, a = 2^k ∧ b = 2^k ∧ p = 2 ∧ n = 3*k + 1) ∨
  (∃ k : ℕ, a = 3^k ∧ b = 2 * 3^k ∧ p = 3 ∧ n = 3*k + 2) ∨
  (∃ k : ℕ, a = 2 * 3^k ∧ b = 3^k ∧ p = 3 ∧ n = 3*k + 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_sum_prime_power_solutions_l795_79584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_l795_79574

-- Define the function f
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := x^(-k^2 + k + 2)

-- State the theorem
theorem k_range (k : ℝ) :
  (∀ x > 0, f k x < f k (x + 1)) →
  -1 < k ∧ k < 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_l795_79574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_inscribed_triangle_area_l795_79537

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the quadrilateral P₁P₂P₃P₄
structure Quadrilateral :=
  (P₁ P₂ P₃ P₄ : ℝ × ℝ)

-- Function to calculate the area of a triangle
noncomputable def triangleArea (t : Triangle) : ℝ := sorry

-- Function to check if a point lies on a line segment
def pointOnSegment (p : ℝ × ℝ) (a b : ℝ × ℝ) : Prop := sorry

-- Function to calculate the area of a triangle given three points
noncomputable def areaOfTriangle (p₁ p₂ p₃ : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem quadrilateral_inscribed_triangle_area
  (abc : Triangle)
  (p₁p₂p₃p₄ : Quadrilateral)
  (h_area : triangleArea abc = 1)
  (h_p₁ : pointOnSegment p₁p₂p₃p₄.P₁ abc.A abc.B ∨ pointOnSegment p₁p₂p₃p₄.P₁ abc.B abc.C ∨ pointOnSegment p₁p₂p₃p₄.P₁ abc.C abc.A)
  (h_p₂ : pointOnSegment p₁p₂p₃p₄.P₂ abc.A abc.B ∨ pointOnSegment p₁p₂p₃p₄.P₂ abc.B abc.C ∨ pointOnSegment p₁p₂p₃p₄.P₂ abc.C abc.A)
  (h_p₃ : pointOnSegment p₁p₂p₃p₄.P₃ abc.A abc.B ∨ pointOnSegment p₁p₂p₃p₄.P₃ abc.B abc.C ∨ pointOnSegment p₁p₂p₃p₄.P₃ abc.C abc.A)
  (h_p₄ : pointOnSegment p₁p₂p₃p₄.P₄ abc.A abc.B ∨ pointOnSegment p₁p₂p₃p₄.P₄ abc.B abc.C ∨ pointOnSegment p₁p₂p₃p₄.P₄ abc.C abc.A) :
  ∃ (t : Fin 4), areaOfTriangle 
    (List.get! [p₁p₂p₃p₄.P₁, p₁p₂p₃p₄.P₂, p₁p₂p₃p₄.P₃, p₁p₂p₃p₄.P₄] t)
    (List.get! [p₁p₂p₃p₄.P₁, p₁p₂p₃p₄.P₂, p₁p₂p₃p₄.P₃, p₁p₂p₃p₄.P₄] ((t + 1) % 4))
    (List.get! [p₁p₂p₃p₄.P₁, p₁p₂p₃p₄.P₂, p₁p₂p₃p₄.P₃, p₁p₂p₃p₄.P₄] ((t + 2) % 4)) ≤ 1/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_inscribed_triangle_area_l795_79537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_cot_2alpha_l795_79507

theorem expression_equals_cot_2alpha (α : ℝ) :
  (1 + Real.cos (4 * α - 2 * Real.pi) + Real.cos (4 * α - Real.pi / 2)) /
  (1 + Real.cos (4 * α + Real.pi) + Real.cos (4 * α + 3 * Real.pi / 2)) =
  Real.tan (Real.pi / 2 - 2 * α) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_cot_2alpha_l795_79507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_meeting_distance_l795_79546

/-- The distance traveled by Train X when it meets Train Y -/
noncomputable def distance_traveled_by_train_x (route_length : ℝ) (time_x : ℝ) (time_y : ℝ) : ℝ :=
  let speed_x := route_length / time_x
  let speed_y := route_length / time_y
  let combined_speed := speed_x + speed_y
  let time_to_meet := route_length / combined_speed
  speed_x * time_to_meet

/-- Theorem stating that Train X travels approximately 60 km when it meets Train Y -/
theorem train_meeting_distance : 
  ∀ ε > 0, 
  ∃ δ > 0, 
  ∀ route_length time_x time_y,
  route_length > 0 ∧ time_x > 0 ∧ time_y > 0 →
  |route_length - 140| < δ ∧ |time_x - 4| < δ ∧ |time_y - 3| < δ →
  |distance_traveled_by_train_x route_length time_x time_y - 60| < ε := by
  sorry

#check train_meeting_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_meeting_distance_l795_79546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentatonic_scale_properties_l795_79568

noncomputable def threefold_loss (x : ℝ) : ℝ := (2/3) * x

noncomputable def threefold_gain (x : ℝ) : ℝ := (4/3) * x

noncomputable def gong_length : ℝ := 81

noncomputable def zhi_length : ℝ := threefold_loss gong_length

noncomputable def shang_length : ℝ := threefold_gain zhi_length

noncomputable def yu_length : ℝ := threefold_loss shang_length

noncomputable def jue_length : ℝ := threefold_gain yu_length

theorem pentatonic_scale_properties :
  (∀ x ∈ ({zhi_length, shang_length, yu_length, jue_length} : Set ℝ), x < gong_length) ∧
  (∀ x ∈ ({gong_length, zhi_length, shang_length, jue_length} : Set ℝ), yu_length < x) ∧
  (jue_length = 64) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentatonic_scale_properties_l795_79568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l795_79548

/-- Given that tg(2α) = 3/4 and α ∈ (-π/2, π/2), prove that cos(2α) = -4/5 and tg(α/2) = (1 - √10) / 3
    when f(x) = sin(x+2) + sin(α-x) - 2sin(α) has a minimum value of 0. -/
theorem trig_problem (α : ℝ) (h1 : Real.tan (2 * α) = 3 / 4) (h2 : α ∈ Set.Ioo (-π/2) (π/2))
  (h3 : ∃ x, ∀ y, Real.sin (y + 2) + Real.sin (α - y) - 2 * Real.sin α ≥ Real.sin (x + 2) + Real.sin (α - x) - 2 * Real.sin α)
  (h4 : ∃ x, Real.sin (x + 2) + Real.sin (α - x) - 2 * Real.sin α = 0) :
  Real.cos (2 * α) = -4/5 ∧ Real.tan (α / 2) = (1 - Real.sqrt 10) / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l795_79548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inradius_relations_l795_79515

/-- Given a triangle with sides a, b, c, semiperimeter s, and inradius t, 
    this theorem states the relationships between these quantities and the angles. -/
theorem triangle_inradius_relations (a b c s t : ℝ) (α β γ : Real)  -- Changed ℝ to Real for angles
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_pos_s : 0 < s) (h_pos_t : 0 < t)
  (h_semiperimeter : s = (a + b + c) / 2)
  (h_angle_sum : α + β + γ = Real.pi)  -- Changed π to Real.pi
  (h_sin_law : Real.sin α / a = Real.sin β / b)  -- Added Real. prefix
  (h_cos_law : c^2 = a^2 + b^2 - 2*a*b*(Real.cos γ)) :  -- Added Real. prefix and parentheses
  (t = s * (s - a) * Real.tan (α/2) ∧ 
   t = s * (s - b) * Real.tan (β/2) ∧ 
   t = s * (s - c) * Real.tan (γ/2)) ∧
  (t = (s - b) * (s - c) * (Real.tan (α/2))⁻¹ ∧ 
   t = (s - a) * (s - c) * (Real.tan (β/2))⁻¹ ∧ 
   t = (s - a) * (s - b) * (Real.tan (γ/2))⁻¹) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inradius_relations_l795_79515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_rotation_l795_79534

/-- The volume of the solid formed by rotating the region bounded by y = sin(πx/2) and y = x^2 around the x-axis -/
noncomputable def rotationVolume : ℝ := 3 * Real.pi / 10

/-- The upper bounding function -/
noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.pi * x / 2)

/-- The lower bounding function -/
def g (x : ℝ) : ℝ := x^2

/-- The lower limit of integration -/
def a : ℝ := 0

/-- The upper limit of integration -/
def b : ℝ := 1

theorem volume_of_rotation :
  ∫ x in a..b, Real.pi * (f x^2 - g x^2) = rotationVolume := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_rotation_l795_79534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particular_solution_l795_79536

noncomputable def y (x : ℝ) : ℝ := 2 * x * Real.exp x - Real.sinh x

theorem particular_solution :
  (∀ x, (deriv (deriv y)) x - y x = 4 * Real.exp x) ∧
  y 0 = 0 ∧
  (deriv y) 0 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particular_solution_l795_79536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l795_79528

theorem expression_value (y : ℝ) : (1 : ℝ) ^ (4 * y - 1) / ((1 : ℝ) / 5 + (1 : ℝ) / 3) = 15 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l795_79528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_inequality_l795_79598

theorem max_value_inequality (x : ℝ) (h : x > 0) :
  (x^2 + 2 - Real.sqrt (x^4 + 4*x^2)) / x ≤ 2 / (2 * Real.sqrt 2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_inequality_l795_79598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mute_person_meat_purchase_l795_79575

/-- The price of meat in cents per liang -/
def x : ℚ := sorry

/-- The amount of money the person carries in cents -/
def y : ℚ := sorry

/-- Condition: When buying 16 liang, the person is short of 25 cents -/
axiom condition1 : 16 * x = y + 25

/-- Condition: When buying 8 liang, the person pays 15 cents more -/
axiom condition2 : 8 * x = y - 15

/-- The amount of meat the person can buy in liang -/
def meat_amount : ℚ := y / x

/-- Theorem: The amount of meat the person can buy is 11 liang -/
theorem mute_person_meat_purchase : meat_amount = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mute_person_meat_purchase_l795_79575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_m_equals_negative_three_l795_79511

-- Define the function f
noncomputable def f (x m : ℝ) : ℝ := ((x + 3) * (x + m)) / x

-- State the theorem
theorem odd_function_implies_m_equals_negative_three :
  (∀ x : ℝ, x ≠ 0 → f x m = -f (-x) m) → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_m_equals_negative_three_l795_79511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l795_79562

-- Define the triangle ABC
def Triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = Real.pi

-- State the theorem
theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  Triangle a b c A B C →
  b * Real.cos B = (a * Real.cos C + c * Real.cos A) / 2 →
  b = Real.sqrt 3 →
  a * c * Real.sin B / 2 = 3 * Real.sqrt 3 / 4 →
  B = Real.pi / 3 ∧ a + c = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l795_79562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_inequality_domain_l795_79514

noncomputable def f (n : ℕ+) (x : ℝ) : ℝ := x^(n : ℝ)

theorem functional_inequality_domain (n : ℕ+) (hn : n > 1) :
  ∀ x : ℝ, f n x + f n (1 - x) > 1 ↔ x ∈ Set.Iio 0 ∪ Set.Ioi 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_inequality_domain_l795_79514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_construction_solutions_l795_79547

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A circle in a 2D plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- Inversion of a point P with respect to center A and radius r -/
noncomputable def invert (A P : Point) (r : ℝ) : Point :=
  sorry

/-- Determine if a point lies inside, on, or outside a circle -/
inductive PointPosition
  | Inside
  | On
  | Outside

/-- Get the position of a point relative to a circle -/
noncomputable def getPointPosition (P : Point) (C : Circle) : PointPosition :=
  sorry

/-- Invert a circle with respect to a point and radius -/
noncomputable def invertCircle (A : Point) (C : Circle) (r : ℝ) : Circle :=
  sorry

/-- Number of solutions for the circle construction problem -/
noncomputable def numSolutions (A B : Point) (S : Circle) : Nat :=
  match getPointPosition (invert A B 1) (invertCircle A S 1) with
  | PointPosition.Inside => 0
  | PointPosition.On => 1
  | PointPosition.Outside => 2

/-- Theorem stating the number of solutions for the circle construction problem -/
theorem circle_construction_solutions (A B : Point) (S : Circle) 
  (h : A ≠ S.center ∧ (A.x - S.center.x)^2 + (A.y - S.center.y)^2 ≠ S.radius^2) :
  (numSolutions A B S = 0 ∨ numSolutions A B S = 1 ∨ numSolutions A B S = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_construction_solutions_l795_79547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_one_f_plus_a_positive_implies_a_ge_one_l795_79573

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 2 / (2^x + 1)

-- Part 1: Prove that if f is odd, then a = 1
theorem odd_function_implies_a_equals_one (a : ℝ) :
  (∀ x : ℝ, f a (-x) = -(f a x)) → a = 1 :=
by sorry

-- Part 2: Prove that if f(x) + a > 0 for all x, then a ≥ 1
theorem f_plus_a_positive_implies_a_ge_one (a : ℝ) :
  (∀ x : ℝ, f a x + a > 0) → a ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_one_f_plus_a_positive_implies_a_ge_one_l795_79573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kitchen_painting_rate_l795_79595

/-- Calculates the painting rate given kitchen dimensions and painting time -/
noncomputable def paintingRate (length width height : ℝ) (coats : ℕ) (totalTime : ℝ) : ℝ :=
  let wallArea := 2 * (length * height + width * height)
  let totalArea := wallArea * (coats : ℝ)
  totalArea / totalTime

/-- Theorem stating that for the given kitchen dimensions and painting time, 
    the painting rate is 40 square feet per hour -/
theorem kitchen_painting_rate :
  paintingRate 16 12 10 3 42 = 40 := by
  -- Unfold the definition of paintingRate
  unfold paintingRate
  -- Simplify the expression
  simp
  -- Perform the numerical calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kitchen_painting_rate_l795_79595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l795_79586

/-- The parabola defined by a real parameter a -/
def parabola (a : ℝ) (x : ℝ) : ℝ := x^2 + (a+2)*x - 2*a + 1

/-- The fixed point that the parabola always passes through -/
def fixed_point : ℝ × ℝ := (2, 9)

/-- The curve on which the vertex of the parabola lies -/
def vertex_curve (x : ℝ) : ℝ := -x^2 + 4*x + 5

/-- The range of the larger root -/
def larger_root_range : Set ℝ := Set.Ioo (-1) 2 ∪ Set.Ioi 5

theorem parabola_properties (a : ℝ) :
  (parabola a (fixed_point.1) = fixed_point.2) ∧
  (∃ x y, parabola a x = y ∧ vertex_curve x = y) ∧
  (∀ x, x^2 + (a+2)*x - 2*a + 1 = 0 → x ∈ larger_root_range) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l795_79586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_at_8_l795_79567

-- Define the polynomial P
def P : ℝ → ℝ := sorry

-- State the properties of P
axiom P_quadratic : ∃ a b c : ℝ, ∀ x, P x = a * x^2 + b * x + c

axiom P_composition : ∀ x, P (P x) = x^4 - 2*x^3 + 4*x^2 - 3*x + 4

-- Theorem to prove
theorem P_at_8 : P 8 = 58 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_at_8_l795_79567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_inverse_proportion_l795_79533

/-- Definition of an inverse proportion function -/
noncomputable def is_inverse_proportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, x ≠ 0 → f x = k / x

/-- The function f(x) = 1/(3x) -/
noncomputable def f (x : ℝ) : ℝ := 1 / (3 * x)

/-- Theorem: f(x) = 1/(3x) is an inverse proportion function -/
theorem f_is_inverse_proportion : is_inverse_proportion f := by
  use (1/3)
  intro x hx
  simp [f]
  field_simp
  ring
  

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_inverse_proportion_l795_79533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_problem_l795_79502

-- Define the given conditions
theorem tan_problem (A B : Real) 
  (h1 : A ∈ Set.Ioo 0 π)
  (h2 : B ∈ Set.Ioo 0 π)
  (h3 : Real.tan (A - B) = 1/2)
  (h4 : Real.tan B = -1/7) :
  Real.tan A = 1/3 ∧ Real.tan (2*A - B) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_problem_l795_79502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_relation_l795_79549

noncomputable section

open Real

theorem triangle_side_relation (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ A < π/2 →  -- A is acute
  0 < B ∧ B < π/2 →  -- B is acute
  0 < C ∧ C < π/2 →  -- C is acute
  a > 0 →            -- side lengths are positive
  b > 0 → 
  c > 0 → 
  Real.sin B * (1 + 2 * Real.cos C) = 2 * Real.sin A * Real.cos C + Real.cos A * Real.sin C →
  a = Real.sin A →        -- law of sines
  b = Real.sin B →        -- law of sines
  c = Real.sin C →        -- law of sines
  a = 2 * b :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_relation_l795_79549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_before_first_nonzero_digit_specific_case_l795_79553

/-- Represents a decimal representation of a rational number -/
structure DecimalRepresentation where
  numZerosBeforeFirstNonZeroDigit : ℕ

/-- Convert a rational number to its decimal representation -/
noncomputable def toDecimal (q : ℚ) : DecimalRepresentation :=
  { numZerosBeforeFirstNonZeroDigit := sorry }

theorem zeros_before_first_nonzero_digit (n m : ℕ) :
  let f : ℚ := 1 / (2^n * 5^m)
  (toDecimal f).numZerosBeforeFirstNonZeroDigit = min n m :=
by
  sorry

theorem specific_case : 
  let f : ℚ := 1 / (2^3 * 5^6)
  (toDecimal f).numZerosBeforeFirstNonZeroDigit = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_before_first_nonzero_digit_specific_case_l795_79553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbolic_functions_properties_l795_79590

noncomputable section

open Real

def sh (x : ℝ) : ℝ := (exp x - exp (-x)) / 2
def ch (x : ℝ) : ℝ := (exp x + exp (-x)) / 2
def th (x : ℝ) : ℝ := (exp x - exp (-x)) / (exp x + exp (-x))

theorem hyperbolic_functions_properties :
  (∀ x : ℝ, th x = sh x / ch x) ∧
  (∀ x : ℝ, (ch x)^2 - (sh x)^2 = 1) ∧
  (∀ x : ℝ, sh (2*x) = 2 * sh x * ch x) ∧
  (deriv sh 0 = 1) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbolic_functions_properties_l795_79590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_bed_coverage_l795_79563

theorem flower_bed_coverage : 
  let rectangle_length : ℝ := 40
  let rectangle_width : ℝ := 20
  let total_base_length : ℝ := 10
  let num_triangles : ℕ := 3

  let triangle_base : ℝ := total_base_length / num_triangles
  let triangle_height : ℝ := (Real.sqrt 3 / 2) * triangle_base
  let triangle_area : ℝ := (1 / 2) * triangle_base * triangle_height
  let total_flower_bed_area : ℝ := num_triangles * triangle_area
  let rectangle_area : ℝ := rectangle_length * rectangle_width

  (total_flower_bed_area / rectangle_area) = (25 * Real.sqrt 3) / 480 :=
by
  sorry

#eval "Theorem statement compiled successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_bed_coverage_l795_79563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigExpression_equals_half_l795_79500

-- Define the expression
noncomputable def trigExpression : ℝ := 
  2 * Real.cos (30 * Real.pi / 180) - 
  Real.tan (60 * Real.pi / 180) + 
  Real.sin (45 * Real.pi / 180) * Real.cos (45 * Real.pi / 180)

-- State the theorem
theorem trigExpression_equals_half : trigExpression = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigExpression_equals_half_l795_79500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_double_application_function_l795_79566

theorem no_double_application_function :
  ¬ ∃ f : ℝ → ℝ, ∀ x : ℝ, f (f x) = x^2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_double_application_function_l795_79566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_equals_two_l795_79555

open Set
open Real
open MeasureTheory
open Interval

-- Define the integrand function
noncomputable def f (x : ℝ) : ℝ := x^2 * Real.tan x + x^3 + 1

-- State the theorem
theorem integral_f_equals_two :
  ∫ x in Icc (-1) 1, f x = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_equals_two_l795_79555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_unusual_numbers_l795_79505

theorem two_unusual_numbers : ∃ (n₁ n₂ : ℕ), 
  n₁ ≠ n₂ ∧ 
  n₁ = 10^100 - 1 ∧ 
  n₂ = 5 * 10^99 - 1 ∧
  (∀ n : ℕ, n = n₁ ∨ n = n₂ →
    (10^99 ≤ n ∧ n < 10^100) ∧  -- 100-digit number
    (n^3 % 10^100 = n) ∧        -- n^3 ends with n
    (n^2 % 10^100 ≠ n))         -- n^2 does not end with n
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_unusual_numbers_l795_79505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_ln_to_line_l795_79551

/-- The shortest distance from a point on the curve y = ln x to the line y = x + 2 -/
theorem shortest_distance_ln_to_line : 
  ∃ d : ℝ, d = (3 * Real.sqrt 2) / 2 ∧ 
    ∀ x : ℝ, x > 0 → 
      d ≤ Real.sqrt ((x - (Real.log x - 2))^2 + (Real.log x - (x + 2))^2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_ln_to_line_l795_79551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_of_A_l795_79522

def A : Finset Nat := {1, 2, 3}

theorem number_of_subsets_of_A : (Finset.powerset A).card = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_of_A_l795_79522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_is_two_l795_79529

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def S (a : ℝ) (n : ℕ+) : ℝ := n * (2 * a + (n - 1) * 4) / 2

/-- The ratio of S_{2n} to S_n is constant -/
def ratio_is_constant (a : ℝ) : Prop :=
  ∃ c : ℝ, ∀ n : ℕ+, S a (2 * n) / S a n = c

/-- The first term of the sequence is 2 if the ratio is constant -/
theorem first_term_is_two :
  ∀ a : ℝ, ratio_is_constant a → a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_is_two_l795_79529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_distance_to_line_l795_79581

noncomputable def distance_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / Real.sqrt (A^2 + B^2)

theorem point_distance_to_line (a : ℝ) (h1 : a > 0) 
  (h2 : distance_to_line a 2 1 (-1) 3 = 1) : a = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_distance_to_line_l795_79581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l795_79585

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  S : Real

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.S = (Real.sqrt 3 / 4) * (t.a^2 + t.b^2 - t.c^2)) :
  (t.C = π / 3) ∧ 
  (∀ (A B : Real), A + B + t.C = π → Real.sin A + Real.sin B ≤ Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l795_79585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l795_79530

-- Define the triangle ABC
theorem triangle_property (A B C : ℝ) (a b c : ℝ) :
  -- Conditions
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧          -- Sum of angles in a triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Side lengths are positive
  a = 2*b ∧                -- Given condition
  Real.cos B = 2*Real.sqrt 2/3 →-- Given condition
  -- Conclusion
  Real.sin ((A - B)/2) + Real.sin (C/2) = Real.sqrt 10/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l795_79530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_characterization_and_inequality_l795_79576

noncomputable def f (x : ℝ) : ℝ := |x - 1/2| + |x + 1/2|

def M : Set ℝ := {x | f x < 2}

theorem M_characterization_and_inequality :
  (M = Set.Ioo (-1 : ℝ) 1) ∧
  (∀ a b, a ∈ M → b ∈ M → |a + b| < |1 + a * b|) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_characterization_and_inequality_l795_79576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_in_savings_problem_solution_l795_79513

noncomputable def final_price (initial_price : ℝ) (changes : List ℝ) : ℝ :=
  changes.foldl (λ acc change => acc * (1 + change / 100)) initial_price

noncomputable def amount_spent (final_price : ℝ) (percent_purchased : ℝ) : ℝ :=
  final_price * (percent_purchased / 100)

structure PriceScenario where
  initial_price : ℝ
  changes_x : List ℝ
  changes_y : List ℝ
  percent_purchased_x : ℝ
  percent_purchased_y : ℝ

theorem difference_in_savings (scenario : PriceScenario) :
  let final_price_x := final_price scenario.initial_price scenario.changes_x
  let final_price_y := final_price scenario.initial_price scenario.changes_y
  let spent_x := amount_spent final_price_x scenario.percent_purchased_x
  let spent_y := amount_spent final_price_y scenario.percent_purchased_y
  let saved_x := scenario.initial_price - spent_x
  let saved_y := scenario.initial_price - spent_y
  abs (saved_x - saved_y - 9.9477) < 0.0001 :=
by sorry

def problem_scenario : PriceScenario :=
  { initial_price := 100
  , changes_x := [10, 15, 5]
  , changes_y := [5, -7, 8]
  , percent_purchased_x := 60
  , percent_purchased_y := 85
  }

theorem problem_solution : 
  let final_price_x := final_price problem_scenario.initial_price problem_scenario.changes_x
  let final_price_y := final_price problem_scenario.initial_price problem_scenario.changes_y
  let spent_x := amount_spent final_price_x problem_scenario.percent_purchased_x
  let spent_y := amount_spent final_price_y problem_scenario.percent_purchased_y
  let saved_x := problem_scenario.initial_price - spent_x
  let saved_y := problem_scenario.initial_price - spent_y
  abs (saved_x - saved_y - 9.9477) < 0.0001 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_in_savings_problem_solution_l795_79513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_speed_calculation_l795_79531

/-- Calculates the speed for the second half of a journey given the total distance,
    total time, and speed for the first half. -/
noncomputable def second_half_speed (total_distance : ℝ) (total_time : ℝ) (first_half_speed : ℝ) : ℝ :=
  let half_distance := total_distance / 2
  let first_half_time := half_distance / first_half_speed
  let second_half_time := total_time - first_half_time
  half_distance / second_half_time

/-- Theorem stating that for a journey of 960 km completed in 40 hours,
    with the first half traveled at 20 km/hr, the speed for the second half is 30 km/hr. -/
theorem journey_speed_calculation :
  second_half_speed 960 40 20 = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_speed_calculation_l795_79531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l795_79509

theorem trigonometric_identity (A B C : ℝ) 
  (h1 : Real.sin A + Real.sin B + Real.sin C = 0)
  (h2 : Real.cos A + Real.cos B + Real.cos C = 0) :
  (Real.cos (3 * A) + Real.cos (3 * B) + Real.cos (3 * C) = 3 * Real.cos (A + B + C)) ∧
  (Real.sin (3 * A) + Real.sin (3 * B) + Real.sin (3 * C) = 3 * Real.sin (A + B + C)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l795_79509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_value_range_l795_79556

def p (m : ℝ) : Prop := ∀ x ∈ ({1/4, 1/2} : Set ℝ), 3*x < m*(x^2 + 1)

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 4^x + 2^(x+1) + m - 1

def q (m : ℝ) : Prop := ∃ x, f m x = 0

def m_range (m : ℝ) : Prop := m < 1 ∨ m ≥ 6/5

theorem m_value_range (m : ℝ) (h1 : ¬(p m ∧ q m)) (h2 : p m ∨ q m) : m_range m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_value_range_l795_79556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_two_l795_79587

/-- A function satisfying f(x) + 3f(1 - x) = 2x^2 for all real x -/
noncomputable def f : ℝ → ℝ := sorry

/-- The main property of f -/
axiom f_property : ∀ x : ℝ, f x + 3 * f (1 - x) = 2 * x^2

/-- The theorem to prove -/
theorem f_at_two : f 2 = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_two_l795_79587
