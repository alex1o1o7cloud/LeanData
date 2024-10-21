import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_m_value_l1256_125690

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = k • b

/-- The theorem states that if a = (1, -1, 2) and b = (-2, 2, m) are parallel vectors, then m = -4 -/
theorem parallel_vectors_m_value (m : ℝ) :
  let a : ℝ × ℝ × ℝ := (1, -1, 2)
  let b : ℝ × ℝ × ℝ := (-2, 2, m)
  parallel a b → m = -4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_m_value_l1256_125690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_work_hours_l1256_125662

/-- Calculates the number of hours James needs to work to pay for wasted food and janitorial costs --/
noncomputable def calculate_work_hours (minimum_wage : ℚ) (meat_price meat_weight : ℚ) 
  (veg_price veg_weight : ℚ) (bread_price bread_weight : ℚ) 
  (janitor_wage janitor_hours : ℚ) : ℚ :=
  let meat_cost := meat_price * meat_weight
  let veg_cost := veg_price * veg_weight
  let bread_cost := bread_price * bread_weight
  let janitor_overtime_rate := janitor_wage * (3/2)
  let janitor_cost := janitor_overtime_rate * janitor_hours
  let total_cost := meat_cost + veg_cost + bread_cost + janitor_cost
  total_cost / minimum_wage

/-- Theorem stating that James needs to work 50 hours --/
theorem james_work_hours : 
  calculate_work_hours 8 5 20 4 15 (3/2) 60 10 10 = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_work_hours_l1256_125662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l1256_125673

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6)

theorem f_monotone_increasing :
  StrictMonoOn f (Set.Ioo (-Real.pi/3) (Real.pi/6)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l1256_125673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1256_125616

/-- The eccentricity of an ellipse with specific conditions -/
theorem ellipse_eccentricity (a b c : ℝ) (h1 : a > b) (h2 : b > 0) : 
  let F : ℝ × ℝ := (-c, 0)
  let B : ℝ × ℝ := (0, b)
  let A : ℝ × ℝ := (-4*c/3, -b/3)
  (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 → 
    (x = A.1 ∧ y = A.2) ∨ (x = B.1 ∧ y = B.2)) →
  (B.1 - F.1) / (A.1 - F.1) = 3 →
  (B.2 - F.2) / (A.2 - F.2) = 3 →
  c / a = Real.sqrt 2 / 2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1256_125616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l1256_125617

noncomputable def slope_angle : Real := 135 * Real.pi / 180

def y_intercept : Real := -1

theorem line_equation (x y : Real) : 
  (Real.tan slope_angle = -1) → 
  (y = Real.tan slope_angle * x + y_intercept) → 
  (x + y + 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l1256_125617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lucia_better_deal_l1256_125646

/-- Represents the pricing structure of a store selling kale -/
structure KalePricing where
  initialPrice : ℚ → ℚ  -- Price per pound for initial quantity
  subsequentPrice : ℚ → ℚ  -- Price per pound for subsequent quantity
  initialQuantity : ℚ  -- Quantity threshold for initial price

/-- Lucia's pricing structure -/
def luciaPricing : KalePricing := {
  initialPrice := λ x => x,
  subsequentPrice := λ x => 4/5 * x,
  initialQuantity := 20
}

/-- Amby's pricing structure -/
def ambyPricing : KalePricing := {
  initialPrice := λ x => x,
  subsequentPrice := λ x => 9/10 * x,
  initialQuantity := 14
}

/-- Calculate the total price for a given quantity and pricing structure -/
def totalPrice (pricing : KalePricing) (x : ℚ) (quantity : ℚ) : ℚ :=
  if quantity ≤ pricing.initialQuantity then
    quantity * pricing.initialPrice x
  else
    pricing.initialQuantity * pricing.initialPrice x +
    (quantity - pricing.initialQuantity) * pricing.subsequentPrice x

/-- The theorem stating that Lucia's pricing becomes equal or better than Amby's
    at 11 pounds over 15 pounds -/
theorem lucia_better_deal (x : ℚ) (h : x > 0) :
  ∀ n : ℚ, n ≥ 11 →
    totalPrice luciaPricing x (15 + n) ≤ totalPrice ambyPricing x (15 + n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lucia_better_deal_l1256_125646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_backgammon_match_schedules_l1256_125632

/-- Represents a backgammon match schedule between two schools -/
structure BackgammonMatch where
  /-- Number of players per school -/
  players_per_school : Nat
  /-- Number of games each player plays against each opponent -/
  games_per_opponent : Nat
  /-- Number of rounds in the match -/
  total_rounds : Nat
  /-- Number of games played simultaneously in each round -/
  games_per_round : Nat

/-- Calculates the number of ways to schedule a backgammon match -/
def count_schedules (m : BackgammonMatch) : Nat :=
  sorry

/-- Theorem stating that the number of ways to schedule the specific backgammon match is 900 -/
theorem backgammon_match_schedules :
  let m : BackgammonMatch := {
    players_per_school := 3,
    games_per_opponent := 2,
    total_rounds := 6,
    games_per_round := 3
  }
  count_schedules m = 900 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_backgammon_match_schedules_l1256_125632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_power_of_five_l1256_125682

theorem factorial_power_of_five (n : ℕ+) 
  (h1 : (5 : ℕ) ^ n.val ∣ Nat.factorial 20)
  (h2 : ¬((5 : ℕ) ^ (n.val + 1) ∣ Nat.factorial 20)) :
  n = 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_power_of_five_l1256_125682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l1256_125600

/-- The initial investment that grows to $610 after 8 years at 5% interest compounded annually -/
noncomputable def initial_investment : ℝ :=
  610 / (1 + 0.05) ^ 8

/-- The final balance after 8 years of growth -/
noncomputable def final_balance (initial : ℝ) : ℝ :=
  initial * (1 + 0.05) ^ 8

theorem investment_growth :
  final_balance initial_investment = 610 := by
  sorry

/-- Approximate value of the initial investment -/
def initial_investment_approx : ℚ :=
  413.16

#eval initial_investment_approx

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l1256_125600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1256_125620

noncomputable section

open Real

def f (x a : ℝ) : ℝ := 4 * cos x * sin (x - π/3) + a

theorem function_properties :
  ∃ (a : ℝ), 
    (∀ x, f x a ≤ 2) ∧ 
    (∃ x, f x a = 2) ∧
    (∀ x, f (x + π) a = f x a) ∧
    (∀ p, 0 < p → p < π → ∃ x, f (x + p) a ≠ f x a) ∧
    (a = sqrt 3) ∧
    (∃ (A B C : ℝ), 
      A < B ∧ 
      f A a = 1 ∧ 
      f B a = 1 ∧ 
      sin C / sin A = sqrt 2) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1256_125620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bobbie_name_reduction_l1256_125677

/-- Proves the number of letters Bobbie needs to remove from her last name -/
theorem bobbie_name_reduction (samantha_name_length : ℕ) (jamie_last_name : String) : 
  samantha_name_length = 7 →
  jamie_last_name = "Grey" →
  2 = (samantha_name_length + 3) - (2 * jamie_last_name.length) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bobbie_name_reduction_l1256_125677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_trapezoidal_prism_volume_l1256_125627

/-- Represents the volume of a trapezoidal prism -/
noncomputable def trapezoidalPrismVolume (a b h d : ℝ) : ℝ := ((a + b) / 2) * d * h

/-- Theorem: Volume of a specific trapezoidal prism -/
theorem specific_trapezoidal_prism_volume :
  ∀ h : ℝ, h > 0 → trapezoidalPrismVolume 24 18 h 15 = 315 * h :=
by
  intros h h_pos
  unfold trapezoidalPrismVolume
  ring
  
#check specific_trapezoidal_prism_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_trapezoidal_prism_volume_l1256_125627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_and_function_l1256_125696

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- The given conditions on the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  (Real.sin t.A)^2 + (Real.sin t.B)^2 + 4 * (Real.sin t.A) * (Real.sin t.B) * (Real.cos t.C) = 0 ∧
  t.c^2 = 3 * t.a * t.b

/-- The function f(x) with its properties -/
structure FunctionF where
  ω : ℕ+
  φ : ℝ
  monotonic : ∀ x y, π/7 < x → x < y → y < π/2 → 
    (Real.sin (ω * x + φ) < Real.sin (ω * y + φ)) ∨ (Real.sin (ω * x + φ) > Real.sin (ω * y + φ))

/-- The theorem statement -/
theorem triangle_angle_and_function (t : Triangle) (f : FunctionF) :
  TriangleConditions t →
  |f.φ| < π/2 →
  Real.sin (f.ω * t.C + f.φ) = -1/2 →
  t.C = 2*π/3 ∧ False := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_and_function_l1256_125696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geq_g_condition_l1256_125656

/-- The function f(x) = ax³ - x² + x + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - x^2 + x + 2

/-- The function g(x) = (e ln x) / x -/
noncomputable def g (x : ℝ) : ℝ := (Real.exp 1 * Real.log x) / x

/-- The theorem stating the condition for f(x₁) ≥ g(x₂) to hold -/
theorem f_geq_g_condition (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Ioc 0 1 → x₂ ∈ Set.Ioc 0 1 → f a x₁ ≥ g x₂) ↔ a ≥ -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geq_g_condition_l1256_125656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_ratio_equality_l1256_125628

-- Define the ellipse
def Ellipse (x y : ℝ) : Prop := x^2 + 4*y^2 = 4

-- Define points
def M : ℝ × ℝ := (-2, 0)
def N : ℝ × ℝ := (2, 0)
def P : ℝ × ℝ := (-2, 2)

-- Define A and B as points on the ellipse in the first quadrant
axiom A : ℝ × ℝ
axiom B : ℝ × ℝ

-- Define C as the intersection of OP and NA
axiom C : ℝ × ℝ

-- Define slopes
axiom k_AM : ℝ
axiom k_AC : ℝ
axiom k_MB : ℝ
axiom k_MC : ℝ

-- Assumptions
axiom A_on_ellipse : Ellipse A.1 A.2
axiom B_on_ellipse : Ellipse B.1 B.2
axiom A_first_quadrant : A.1 > 0 ∧ A.2 > 0
axiom B_first_quadrant : B.1 > 0 ∧ B.2 > 0
axiom A_on_BP : ∃ t : ℝ, 0 < t ∧ t < 1 ∧ A = (1 - t) • B + t • P

-- Theorem to prove
theorem slope_ratio_equality : k_MB / k_AM = k_AC / k_MC := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_ratio_equality_l1256_125628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trouser_sale_percent_decrease_l1256_125658

/-- Calculate the percent decrease between two prices -/
noncomputable def percentDecrease (originalPrice salePrice : ℝ) : ℝ :=
  ((originalPrice - salePrice) / originalPrice) * 100

/-- Prove that the percent decrease from $100 to $60 is 40% -/
theorem trouser_sale_percent_decrease :
  percentDecrease 100 60 = 40 := by
  -- Unfold the definition of percentDecrease
  unfold percentDecrease
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trouser_sale_percent_decrease_l1256_125658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l1256_125686

/-- Point in 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in 2D plane -/
structure Line where
  slope : ℝ
  passesThrough : Point

/-- Triangle formed by three lines -/
structure Triangle where
  l1 : Line
  l2 : Line
  l3 : Line

/-- Rotation of a line around a point -/
noncomputable def rotate (l : Line) (center : Point) (angle : ℝ) : Line :=
  sorry

/-- Area of a triangle -/
noncomputable def triangleArea (t : Triangle) : ℝ :=
  sorry

/-- The three points as defined in the problem -/
def A : Point := ⟨0, 0⟩
def B : Point := ⟨12, 0⟩
def C : Point := ⟨20, 0⟩

/-- The three initial lines as defined in the problem -/
def ℓA : Line := ⟨1, A⟩
def ℓB : Line := ⟨0, B⟩  -- Vertical line, using 0 instead of ∞
def ℓC : Line := ⟨-1, C⟩

/-- The triangle formed by the three rotating lines -/
noncomputable def rotatingTriangle (angle : ℝ) : Triangle :=
  ⟨rotate ℓA A angle, rotate ℓB B angle, rotate ℓC C angle⟩

/-- Theorem: The maximum area of the triangle formed by the rotating lines is 104 -/
theorem max_triangle_area :
  ∃ (maxArea : ℝ), maxArea = 104 ∧ ∀ (angle : ℝ), triangleArea (rotatingTriangle angle) ≤ maxArea := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l1256_125686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_no_adjacent_green_hats_l1256_125615

/-- The number of children in the group -/
def total_children : ℕ := 9

/-- The number of children wearing green hats -/
def green_hats : ℕ := 3

/-- The probability that no two children wearing green hats are standing next to each other -/
def prob_no_adjacent_green : ℚ := 5/14

theorem prob_no_adjacent_green_hats :
  prob_no_adjacent_green = 1 - (Nat.choose (total_children - green_hats + 1) 1 + Nat.choose (total_children - 1) 2) / Nat.choose total_children green_hats :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_no_adjacent_green_hats_l1256_125615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_50_mod_4_l1256_125613

def T : ℕ → ℕ
  | 0 => 3  -- Add this case to handle Nat.zero
  | 1 => 3
  | n + 1 => 3^(T n)

theorem t_50_mod_4 : T 50 % 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_50_mod_4_l1256_125613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l1256_125691

/-- The area of a trapezium with given parallel sides and height -/
noncomputable def trapeziumArea (a b h : ℝ) : ℝ := (a + b) * h / 2

/-- Theorem: The area of a trapezium with parallel sides of 20 cm and 18 cm, 
    and a height of 5 cm, is 95 square centimeters -/
theorem trapezium_area_example : 
  trapeziumArea 20 18 5 = 95 := by
  -- Unfold the definition of trapeziumArea
  unfold trapeziumArea
  -- Simplify the arithmetic
  simp [add_mul, mul_div_assoc]
  -- Check that the result is equal to 95
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l1256_125691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rook_paths_eq_catalan_l1256_125609

/-- The number of paths for a rook on an n × n chessboard from (1,1) to (n,n),
    moving only right and up, avoiding the main diagonal except at start and end. -/
def rook_paths (n : ℕ) : ℕ :=
  sorry

/-- The nth Catalan number -/
def catalan_number (n : ℕ) : ℕ :=
  sorry

/-- The chessboard is at least 2x2 -/
theorem rook_paths_eq_catalan (n : ℕ) (h : n ≥ 2) :
  rook_paths n = catalan_number (n - 2) := by
  sorry

#check rook_paths_eq_catalan

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rook_paths_eq_catalan_l1256_125609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_always_prime_polynomial_l1256_125659

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def polynomial (α : Type) := α → ℕ

def degree (p : polynomial ℕ) : ℕ := sorry

def non_negative_integer_coefficients (p : polynomial ℕ) : Prop := sorry

theorem no_always_prime_polynomial :
  ∀ (Q : polynomial ℕ),
    degree Q ≥ 2 →
    non_negative_integer_coefficients Q →
    ∃ (p : ℕ), is_prime p ∧ ¬(is_prime (Q p)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_always_prime_polynomial_l1256_125659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1256_125676

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + 2 else 4*x * Real.cos x + 1

-- Define the proposition
theorem range_of_m :
  ∃ (S : Set ℝ), S = Set.Ioo (-4) 2 ∪ {4} ∧
  (∀ m : ℝ, m ∈ S ↔
    ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ ∈ Set.Icc (-2*Real.pi) Real.pi ∧
      x₂ ∈ Set.Icc (-2*Real.pi) Real.pi ∧
      f x₁ = m * x₁ + 1 ∧ f x₂ = m * x₂ + 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1256_125676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_unique_solution_parameter_l1256_125610

/-- The equation has a unique solution when the parameter a equals this value -/
def unique_solution_parameter : ℝ := 7

/-- The equation in question -/
def equation (a x : ℝ) : Prop :=
  ((abs ((a*x^2 - a*x - 12*a + x^2 + x + 12) / (a*x + 3*a - x - 3)) - a) * 
   abs (4*a - 3*x - 19) = 0) ∧ (a ≠ 1) ∧ (x ≠ -3)

/-- The theorem stating that 7 is the largest value of a for which the equation has a unique solution -/
theorem largest_unique_solution_parameter :
  ∀ a : ℝ, (∃! x : ℝ, equation a x) → a ≤ unique_solution_parameter :=
by
  sorry

#check largest_unique_solution_parameter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_unique_solution_parameter_l1256_125610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_into_rectangles_l1256_125683

/-- Represents a cell on the chessboard -/
structure Cell where
  x : ℕ
  y : ℕ

/-- Represents a rectangle on the chessboard -/
structure Rectangle where
  topLeft : Cell
  bottomRight : Cell

/-- Represents a chessboard with marked cells -/
structure Chessboard where
  size : ℕ
  markedCells : Finset Cell

/-- Checks if a cell is within a rectangle -/
def cellInRectangle (c : Cell) (r : Rectangle) : Prop :=
  r.topLeft.x ≤ c.x ∧ c.x ≤ r.bottomRight.x ∧
  r.topLeft.y ≤ c.y ∧ c.y ≤ r.bottomRight.y

/-- Checks if a rook can move through all marked cells without jumping over unmarked cells -/
def validRookPath (board : Chessboard) : Prop :=
  sorry

/-- Theorem: Given 2n marked cells on a chessboard with a valid rook path,
    it's possible to partition the figure into n rectangles -/
theorem partition_into_rectangles (n : ℕ) (board : Chessboard) :
  (board.markedCells.card = 2 * n) →
  validRookPath board →
  ∃ (partition : Finset Rectangle), 
    partition.card = n ∧ 
    (∀ c : Cell, c ∈ board.markedCells ↔ ∃ r ∈ partition, cellInRectangle c r) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_into_rectangles_l1256_125683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_histogram_height_represents_frequency_density_l1256_125687

/-- Represents a frequency histogram --/
structure FrequencyHistogram where
  bars : List (ℝ × ℝ)  -- List of (height, width) pairs for each bar

/-- Represents a group within a sample --/
structure SampleGroup where
  frequency : ℕ  -- Number of individuals in the group
  interval : ℝ   -- Class interval width

/-- The height of a bar in a frequency histogram represents the ratio of the group's frequency to its class interval --/
theorem histogram_height_represents_frequency_density (h : FrequencyHistogram) (g : SampleGroup) :
  ∃ (bar : ℝ × ℝ), bar ∈ h.bars ∧ bar.1 = (g.frequency : ℝ) / g.interval := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_histogram_height_represents_frequency_density_l1256_125687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_general_term_l1256_125695

/-- 
For an arithmetic sequence with first term a₁ = 1 and second term a₂ = -1,
the general term formula is aₙ = -2n + 3.
-/
theorem arithmetic_sequence_general_term :
  ∀ (n : ℕ), n ≥ 1 →
  let a : ℕ → ℤ := λ k => 
    if k = 1 then 1
    else if k = 2 then -1
    else 1 + (k - 1) * ((-1) - 1)
  a n = -2 * (n : ℤ) + 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_general_term_l1256_125695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_neg_one_range_of_x_when_f_equals_abs_x_minus_a_f_equals_abs_x_minus_a_condition_l1256_125688

-- Define the function f
def f (x a : ℝ) : ℝ := |2*x| + |x + a|

-- Part I
theorem solution_set_when_a_is_neg_one :
  {x : ℝ | f x (-1) ≤ 4} = Set.Icc (-1) (5/3) := by sorry

-- Part II
theorem range_of_x_when_f_equals_abs_x_minus_a (a : ℝ) :
  {x : ℝ | f x a = |x - a|} = 
    if a > 0 then Set.Icc (-a) 0
    else if a < 0 then Set.Icc 0 (-a)
    else {0} := by sorry

-- Additional helper theorem to connect the two parts
theorem f_equals_abs_x_minus_a_condition (x a : ℝ) :
  f x a = |x - a| ↔ 2*x*(x + a) ≤ 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_neg_one_range_of_x_when_f_equals_abs_x_minus_a_f_equals_abs_x_minus_a_condition_l1256_125688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_properties_l1256_125689

/-- A cubic function with specific properties -/
def cubic_function (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^3 + b * x + c

/-- The derivative of the cubic function -/
def cubic_derivative (a b : ℝ) : ℝ → ℝ := λ x ↦ 3 * a * x^2 + b

theorem cubic_function_properties (a b c : ℝ) (ha : a > 0) :
  (∀ x, cubic_function a b c (-x) = -(cubic_function a b c x)) →
  ((cubic_derivative a b 1) = -6) →
  (∃ x, ∀ y, cubic_derivative a b y ≥ cubic_derivative a b x) ∧ 
  (∃ x, cubic_derivative a b x = -12) →
  a = 2 ∧ b = -12 ∧ c = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_properties_l1256_125689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brian_homework_theorem_l1256_125679

open BigOperators

def assignments_needed (n : ℕ) : ℕ := (n + 6) / 7

def total_assignments (total_points : ℕ) : ℕ :=
  ∑ i in Finset.range total_points, assignments_needed (i + 1)

theorem brian_homework_theorem :
  total_assignments 28 = 70 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brian_homework_theorem_l1256_125679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_PA_distance_bounds_l1256_125680

/-- The curve C -/
def curve_C (x y : ℝ) : Prop := x^2/4 + y^2/9 = 1

/-- The line l -/
def line_l (x y t : ℝ) : Prop := x = 2 + t ∧ y = 2 - 2*t

/-- The angle between PA and line l -/
noncomputable def angle_PA_l : ℝ := 30 * Real.pi / 180

/-- The distance |PA| -/
noncomputable def distance_PA (x y : ℝ) : ℝ := 
  (2 * Real.sqrt 5 / 5) * abs (5 * Real.sin (Real.arccos (x/2) + Real.arcsin (y/3) + Real.pi/6) - 6)

theorem PA_distance_bounds :
  ∀ x y : ℝ, curve_C x y →
  (∀ t : ℝ, ∃ a : ℝ, line_l (x + a * Real.cos angle_PA_l) (y + a * Real.sin angle_PA_l) t) →
  distance_PA x y ≤ 22 * Real.sqrt 5 / 5 ∧
  distance_PA x y ≥ 2 * Real.sqrt 5 / 5 := by
  sorry

#check PA_distance_bounds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_PA_distance_bounds_l1256_125680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_vowel_initials_l1256_125651

/-- Represents the alphabet --/
def Alphabet : Finset Char := sorry

/-- Represents the set of vowels --/
def Vowels : Finset Char := sorry

/-- Represents the class of students --/
structure Student where
  initial : Char

/-- The class of students --/
def ClassOfStudents : Finset Student := sorry

theorem probability_of_vowel_initials :
  let n := Finset.card ClassOfStudents
  let v := Finset.card (ClassOfStudents.filter (fun s => s.initial ∈ Vowels))
  n = 30 →
  Finset.card Alphabet = 26 →
  Finset.card Vowels = 8 →
  (∀ s₁ s₂, s₁ ∈ ClassOfStudents → s₂ ∈ ClassOfStudents → s₁ ≠ s₂ → s₁.initial ≠ s₂.initial) →
  (∀ s, s ∈ ClassOfStudents → s.initial ∉ Vowels) →
  (v : ℚ) / n = 4 / 15 := by
  sorry

#check probability_of_vowel_initials

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_vowel_initials_l1256_125651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l1256_125694

/-- Calculates the average speed given distance and time -/
noncomputable def average_speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

/-- The problem statement -/
theorem car_average_speed :
  let distance : ℝ := 160
  let time : ℝ := 6
  average_speed distance time = 80 / 3 := by
  -- Unfold the definition of average_speed
  unfold average_speed
  -- Simplify the expression
  simp
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l1256_125694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1256_125681

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (|x + 1| + |x + 2| - 5)

-- Define the domain A
def A : Set ℝ := {x | x ≤ -4 ∨ x ≥ 1}

-- Define the set B
def B : Set ℝ := {x | -1 < x ∧ x < 2}

-- Theorem statement
theorem inequality_proof (a b : ℝ) (ha : a ∈ (B ∩ (Aᶜ))) (hb : b ∈ (B ∩ (Aᶜ))) :
  |a + b| / 2 < |1 + a * b / 4| := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1256_125681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l1256_125607

noncomputable def f (a : ℝ) (x : ℝ) := Real.log (a * x)

def monotonically_increasing (g : ℝ → ℝ) :=
  ∀ x y, 0 < x → x < y → g x < g y

theorem sufficient_not_necessary :
  (∀ x y, 0 < x → x < y → f 1 x < f 1 y) ∧
  (∃ a, a ≠ 1 ∧ ∀ x y, 0 < x → x < y → f a x < f a y) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l1256_125607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_three_zeros_l1256_125655

/-- A cubic function f(x) = x³ - 3x + m has three distinct real zeros if and only if -2 < m < 2. -/
theorem cubic_three_zeros (m : ℝ) : 
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    (fun t : ℝ ↦ t^3 - 3*t + m) x = 0 ∧ 
    (fun t : ℝ ↦ t^3 - 3*t + m) y = 0 ∧ 
    (fun t : ℝ ↦ t^3 - 3*t + m) z = 0) ↔ 
  -2 < m ∧ m < 2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_three_zeros_l1256_125655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_earnings_value_l1256_125678

/-- The time (in days) it takes x to complete the job alone -/
noncomputable def x_time : ℝ := 2

/-- The time (in days) it takes y to complete the job alone -/
noncomputable def y_time : ℝ := 4

/-- The time (in days) it takes z to complete the job alone -/
noncomputable def z_time : ℝ := 6

/-- The total earnings when all three work together -/
noncomputable def total_earnings : ℝ := 2000

/-- Z's work rate (fraction of job completed per day) -/
noncomputable def z_rate : ℝ := 1 / z_time

/-- Combined work rate of x, y, and z (fraction of job completed per day) -/
noncomputable def combined_rate : ℝ := 1 / x_time + 1 / y_time + 1 / z_time

/-- Z's share of the work -/
noncomputable def z_share : ℝ := z_rate / combined_rate

/-- Z's earnings -/
noncomputable def z_earnings : ℝ := total_earnings * z_share

theorem z_earnings_value : z_earnings = 4000 / 11 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_earnings_value_l1256_125678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_vector_ratio_l1256_125624

/-- Helper function to define the incenter of a triangle -/
def is_incenter (O A B C : ℝ × ℝ) : Prop :=
  ∃ (a b c : ℝ),
    (a > 0) ∧ (b > 0) ∧ (c > 0) ∧
    (O.1 = (a * A.1 + b * B.1 + c * C.1) / (a + b + c)) ∧
    (O.2 = (a * A.2 + b * B.2 + c * C.2) / (a + b + c))

/-- Given a triangle ABC with AB = BC = 2 and AC = 3, if O is the incenter 
    and AO = p * AB + q * AC, then p/q = 2/3 -/
theorem incenter_vector_ratio (A B C O : ℝ × ℝ) (p q : ℝ) :
  let AB := B - A
  let AC := C - A
  let AO := O - A
  (norm AB = 2) →
  (norm (C - B) = 2) →
  (norm AC = 3) →
  (is_incenter O A B C) →
  (AO = p • AB + q • AC) →
  (p / q = 2 / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_vector_ratio_l1256_125624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_minus_theta_l1256_125643

theorem tan_pi_minus_theta (θ : Real) 
  (h1 : π / 2 < θ) (h2 : θ < π) 
  (h3 : Real.sin (π / 2 + θ) = -3 / 5) : 
  Real.tan (π - θ) = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_minus_theta_l1256_125643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_integers_sum_l1256_125642

theorem two_integers_sum (a b : ℕ) : 
  a > 0 →
  b > 0 →
  a * b + a + b = 95 →
  Nat.gcd a b = 1 →
  a < 20 ∧ b < 20 →
  a + b = 18 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_integers_sum_l1256_125642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_from_volume_l1256_125671

noncomputable section

open Real

-- Define the region around a line segment
noncomputable def region_volume (segment_length : ℝ) : ℝ :=
  (2 / 3) * π * 5^3 + π * 5^2 * segment_length

-- Theorem statement
theorem segment_length_from_volume :
  ∃ (segment_length : ℝ), region_volume segment_length = 660 * π ∧ segment_length = 20 := by
  -- Proof goes here
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_from_volume_l1256_125671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_ratio_sum_l1256_125664

theorem trigonometric_ratio_sum (x y : ℝ) 
  (h1 : Real.sin x / Real.sin y = 4)
  (h2 : Real.cos x / Real.cos y = 1/3) :
  Real.sin (2*x) / Real.sin (2*y) + Real.cos (2*x) / Real.cos (2*y) = 
    4/3 + (2/9 * (Real.cos y)^2 - 1) / (2 * (Real.cos y)^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_ratio_sum_l1256_125664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concyclic_points_l1256_125633

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the basic geometric relations
variable (InCircle : Point → Circle → Prop)
variable (Intersect : Circle → Circle → Point → Point → Prop)
variable (IntersectLine : Circle → Point → Point → Point → Point → Prop)
variable (IntersectLines : Point → Point → Point → Point → Point → Point → Prop)

-- Define the theorem
theorem concyclic_points
  (ω₁ ω₂ : Circle)
  (A B C D X Y P Q R S U V W Z : Point)
  (h1 : InCircle A ω₁ ∧ InCircle B ω₁ ∧ InCircle C ω₁ ∧ InCircle D ω₁)
  (h2 : Intersect ω₁ ω₂ X Y)
  (h3 : IntersectLine ω₂ A B P Q)
  (h4 : IntersectLine ω₂ C D R S)
  (h5 : IntersectLines Q R P S U V)
  (h6 : IntersectLines Q R P S W Z) :
  ∃ (ω₃ : Circle), InCircle X ω₃ ∧ InCircle Y ω₃ ∧ InCircle U ω₃ ∧ 
                   InCircle V ω₃ ∧ InCircle W ω₃ ∧ InCircle Z ω₃ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_concyclic_points_l1256_125633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_red_points_l1256_125692

/-- Represents a point on the grid -/
structure Point where
  x : Int
  y : Int

/-- Represents the 6x6 grid -/
def Grid := Finset Point

/-- Checks if a point is on the boundary of a k×k sub-grid -/
def isOnBoundary (p : Point) (k : Nat) : Bool :=
  sorry

/-- Checks if a set of points satisfies the condition for all sub-grids -/
def satisfiesCondition (redPoints : Finset Point) : Bool :=
  sorry

/-- The main theorem to prove -/
theorem min_red_points :
  ∃ (redPoints : Finset Point),
    redPoints.card = 12 ∧
    satisfiesCondition redPoints ∧
    ∀ (smallerSet : Finset Point),
      smallerSet.card < 12 →
      ¬satisfiesCondition smallerSet :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_red_points_l1256_125692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sliced_cone_volume_ratio_l1256_125603

/-- Represents a right circular cone -/
structure RightCircularCone where
  height : ℝ
  baseRadius : ℝ

/-- Calculates the volume of a right circular cone -/
noncomputable def coneVolume (c : RightCircularCone) : ℝ :=
  (1/3) * Real.pi * c.baseRadius^2 * c.height

/-- Theorem: The ratio of volumes in a sliced cone -/
theorem sliced_cone_volume_ratio :
  ∀ (c : RightCircularCone),
  let pieceHeight := c.height / 5
  let largestPieceVolume := coneVolume { height := c.height, baseRadius := c.baseRadius } -
                            coneVolume { height := c.height * 4/5, baseRadius := c.baseRadius * 4/5 }
  let secondLargestPieceVolume := coneVolume { height := c.height * 4/5, baseRadius := c.baseRadius * 4/5 } -
                                  coneVolume { height := c.height * 3/5, baseRadius := c.baseRadius * 3/5 }
  secondLargestPieceVolume / largestPieceVolume = 37 / 61 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sliced_cone_volume_ratio_l1256_125603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spectral_density_correlation_function_relation_l1256_125629

/-- Spectral density function -/
noncomputable def spectral_density (s₀ ω₀ ω : ℝ) : ℝ :=
  if -ω₀ ≤ ω ∧ ω ≤ ω₀ then s₀ else 0

/-- Correlation function -/
noncomputable def correlation_function (s₀ ω₀ τ : ℝ) : ℝ :=
  2 * s₀ * Real.sin (ω₀ * τ) / τ

/-- Theorem stating the relationship between spectral density and correlation function -/
theorem spectral_density_correlation_function_relation
  (X : ℝ → ℝ) -- Stationary random function
  (s₀ ω₀ : ℝ) -- Parameters of the spectral density
  (h₁ : s₀ > 0)
  (h₂ : ω₀ > 0) :
  ∀ τ : ℝ, τ ≠ 0 →
  (∫ (ω : ℝ), spectral_density s₀ ω₀ ω * Complex.exp (Complex.I * ω * τ)) =
  correlation_function s₀ ω₀ τ := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spectral_density_correlation_function_relation_l1256_125629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_a_b_is_neg_two_l1256_125699

noncomputable def line_l_slope_angle : ℝ := 3 * Real.pi / 4

def point_A : ℝ × ℝ := (3, 2)
def point_B (a : ℝ) : ℝ × ℝ := (a, -1)

def line_l₁ (a : ℝ) : Set (ℝ × ℝ) := {p | ∃ t : ℝ, p = (1 - t) • point_A + t • (point_B a)}

def line_l₂ (b : ℝ) : Set (ℝ × ℝ) := {p | 2 * p.1 + b * p.2 + 1 = 0}

def line_l : Set (ℝ × ℝ) := {p | ∃ t : ℝ, p.2 = Real.tan line_l_slope_angle * p.1 + t}

def perpendicular (l₁ l₂ : Set (ℝ × ℝ)) : Prop := sorry
def parallel (l₁ l₂ : Set (ℝ × ℝ)) : Prop := sorry

theorem sum_a_b_is_neg_two (a b : ℝ) : 
  perpendicular (line_l₁ a) line_l →
  parallel (line_l₁ a) (line_l₂ b) →
  a + b = -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_a_b_is_neg_two_l1256_125699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_de_plus_fg_equals_two_l1256_125648

open Geometry

-- Define the equilateral triangle ABC
def Triangle (A B C : EuclideanSpace ℝ (Fin 2)) : Prop :=
  dist A B = 2 ∧ 
  dist B C = 2 ∧ 
  dist C A = 2

-- Define points D, F on AB and E, G on AC
def PointsOnSides (A B C D E F G : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ (t₁ t₂ t₃ t₄ : ℝ), 0 ≤ t₁ ∧ t₁ ≤ 1 ∧ 0 ≤ t₂ ∧ t₂ ≤ 1 ∧ 
                        0 ≤ t₃ ∧ t₃ ≤ 1 ∧ 0 ≤ t₄ ∧ t₄ ≤ 1 ∧
                        D = A + t₁ • (B - A) ∧
                        F = A + t₂ • (B - A) ∧
                        E = A + t₃ • (C - A) ∧
                        G = A + t₄ • (C - A)

-- Define DE and FG parallel to BC
def ParallelToBC (A B C D E F G : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ (k₁ k₂ : ℝ), E - D = k₁ • (C - B) ∧ G - F = k₂ • (C - B)

-- Define perimeter equality
def EqualPerimeters (A B C D E F G : EuclideanSpace ℝ (Fin 2)) : Prop :=
  dist A D + dist D E + dist E A = 
  dist D F + dist F G + dist G E + dist E D ∧
  dist A D + dist D E + dist E A = 
  dist F B + dist B C + dist C G + dist G F

theorem de_plus_fg_equals_two 
  (A B C D E F G : EuclideanSpace ℝ (Fin 2)) 
  (h₁ : Triangle A B C) 
  (h₂ : PointsOnSides A B C D E F G) 
  (h₃ : ParallelToBC A B C D E F G) 
  (h₄ : EqualPerimeters A B C D E F G) : 
  dist D E + dist F G = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_de_plus_fg_equals_two_l1256_125648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l1256_125605

noncomputable section

-- Define the vectors a and b
def a (α : Real) : Fin 2 → Real
  | 0 => 3 * Real.sin α
  | 1 => Real.cos α

def b (α : Real) : Fin 2 → Real
  | 0 => 2 * Real.sin α
  | 1 => 5 * Real.sin α - 4 * Real.cos α

-- Define the dot product
def dot_product (v w : Fin 2 → Real) : Real :=
  (v 0) * (w 0) + (v 1) * (w 1)

-- State the theorem
theorem vector_problem (α : Real) 
  (h1 : α > 3 * Real.pi / 2) 
  (h2 : α < 2 * Real.pi) 
  (h3 : dot_product (a α) (b α) = 0) : 
  Real.tan α = -4/3 ∧ 
  Real.cos (2*α + Real.pi/3) = (24 * Real.sqrt 3 - 7) / 50 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l1256_125605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_perimeter_l1256_125612

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- Calculates the eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- Calculates the perimeter of triangle ABF₂ for an ellipse -/
def triangle_perimeter (e : Ellipse) : ℝ :=
  4 * e.a

theorem ellipse_triangle_perimeter :
  ∀ (e : Ellipse), e.b = 4 → eccentricity e = 3/5 → triangle_perimeter e = 20 := by
  intro e h_b h_ecc
  have h_a : e.a = 5 := by
    sorry -- Proof that a = 5 given b = 4 and eccentricity = 3/5
  calc
    triangle_perimeter e = 4 * e.a := rfl
    _ = 4 * 5 := by rw [h_a]
    _ = 20 := by norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_perimeter_l1256_125612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1256_125640

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

noncomputable def f (x : ℝ) : ℝ := x * floor (2 / x) + x / 2 - 2

def domain (x : ℝ) : Prop := (1/3 < |x|) ∧ (|x| < 2)

theorem f_properties :
  ∀ x : ℝ, domain x →
  (∃ a b : ℝ, ∃ I : Set ℝ, x ∈ I ∧ IsOpen I ∧ ∀ y ∈ I, f y = a * y + b) ∧
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 6 → ¬ContinuousAt f (2/n) ∧ ¬ContinuousAt f (-2/n)) ∧
  (∀ y : ℝ, domain y → f (-y) = f y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1256_125640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_identity_l1256_125697

theorem sin_cos_identity (α : Real) 
  (h : (Real.sin α + 3 * Real.cos α) / (3 * Real.cos α - Real.sin α) = 5) : 
  Real.sin α ^ 2 - Real.sin α * Real.cos α = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_identity_l1256_125697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1256_125601

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h : t.b * Real.sin t.A / (t.a * Real.cos t.B) = Real.sqrt 3) : 
  t.B = π / 3 ∧ 
  (t.b = 3 ↔ t.a * t.c * Real.sin t.B / 2 = 9 * Real.sqrt 3 / 4) ∧
  (t.b = 3 ↔ t.a + t.c = 6) ∧
  (t.a * t.c * Real.sin t.B / 2 = 9 * Real.sqrt 3 / 4 ↔ t.a + t.c = 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1256_125601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_plus_alpha_l1256_125672

theorem cos_pi_plus_alpha (α : ℝ) (h : Real.sin (π / 2 + α) = 1 / 3) :
  Real.cos (π + α) = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_plus_alpha_l1256_125672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_gcd_values_l1256_125636

theorem sum_of_gcd_values : ∃ (S : Finset ℕ), 
  (∀ n : ℕ+, ∃ m ∈ S, m = Nat.gcd (5 * n + 6) n) ∧ 
  (S.sum id = 12) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_gcd_values_l1256_125636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_origin_l1256_125634

theorem min_distance_to_origin (a b : ℝ) : 
  (3 * a - 4 * b = 10) → 
  (∃ (m : ℝ), ∀ (x y : ℝ), 3 * x - 4 * y = 10 → Real.sqrt (x^2 + y^2) ≥ m) ∧ 
  Real.sqrt (a^2 + b^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_origin_l1256_125634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_linear_combination_l1256_125639

open Matrix

theorem matrix_linear_combination (M : Matrix (Fin 2) (Fin 2) ℝ) 
  (u v w : Fin 2 → ℝ) :
  M.mulVec u = ![3, -1] →
  M.mulVec v = ![-2, 4] →
  M.mulVec w = ![5, -3] →
  M.mulVec (3 • u - v + 2 • w) = ![21, -13] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_linear_combination_l1256_125639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_width_is_15_l1256_125638

/-- Represents a rectangular garden with given perimeter and maximum length -/
structure Garden where
  perimeter : ℝ
  maxLength : ℝ

/-- Calculates the width of the garden given its length -/
noncomputable def Garden.width (g : Garden) (length : ℝ) : ℝ :=
  (g.perimeter - 2 * length) / 2

/-- Calculates the area of the garden given its length -/
noncomputable def Garden.area (g : Garden) (length : ℝ) : ℝ :=
  length * g.width length

/-- Theorem stating that the width maximizing the sum of triangular areas is 15 meters -/
theorem optimal_width_is_15 (g : Garden) (h1 : g.perimeter = 80) (h2 : g.maxLength = 25) :
  g.width 25 = 15 ∧ ∀ l, 0 < l ∧ l ≤ g.maxLength → g.area l ≤ g.area 25 := by
  sorry

#check optimal_width_is_15

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_width_is_15_l1256_125638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_added_to_change_ratio_l1256_125684

/-- Proves that 9 litres of water need to be added to a 45-litre mixture
    with an initial milk-to-water ratio of 4:1 to achieve a new ratio of 2:1 -/
theorem water_added_to_change_ratio : 
  ∀ (initial_volume : ℝ) 
    (initial_milk_ratio : ℝ) 
    (initial_water_ratio : ℝ) 
    (final_milk_ratio : ℝ) 
    (final_water_ratio : ℝ),
  initial_volume = 45 →
  initial_milk_ratio = 4 →
  initial_water_ratio = 1 →
  final_milk_ratio = 2 →
  final_water_ratio = 1 →
  let initial_milk := initial_volume * initial_milk_ratio / (initial_milk_ratio + initial_water_ratio)
  let initial_water := initial_volume * initial_water_ratio / (initial_milk_ratio + initial_water_ratio)
  let final_water := initial_milk * final_water_ratio / final_milk_ratio
  let water_added := final_water - initial_water
  water_added = 9 := by
  intro initial_volume initial_milk_ratio initial_water_ratio final_milk_ratio final_water_ratio
  intro h1 h2 h3 h4 h5
  -- The proof steps would go here
  sorry

#check water_added_to_change_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_added_to_change_ratio_l1256_125684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_bound_l1256_125621

theorem quadratic_function_bound (a b c : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → |a * x^2 + b * x + c| ≤ 1) →
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → |2 * a * x + b| ≤ 4) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_bound_l1256_125621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_congruence_solutions_count_l1256_125650

theorem congruence_solutions_count : 
  ∃! n : ℕ, n = (Finset.filter 
    (λ x : ℕ ↦ x > 0 ∧ x < 100 ∧ (x + 17) % 29 = 63 % 29) 
    (Finset.range 100)).card ∧ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_congruence_solutions_count_l1256_125650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_min_max_values_l1256_125606

noncomputable def y (p q : ℝ) (x : ℝ) : ℝ := (Real.cos x) ^ p * (Real.sin x) ^ q

theorem product_min_max_values (p q : ℝ) (hp : 0 < p) (hq : 0 < q) :
  ∃ (x_min x_max : ℝ),
    (0 ≤ x_min ∧ x_min ≤ Real.pi / 2) ∧
    (0 ≤ x_max ∧ x_max ≤ Real.pi / 2) ∧
    (∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → y p q x_min ≤ y p q x ∧ y p q x ≤ y p q x_max) ∧
    y p q x_min = 0 ∧
    x_max = Real.arctan (Real.sqrt (q / p)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_min_max_values_l1256_125606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1256_125619

/-- Given function f(x) with parameters ω and φ -/
noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ - Real.pi / 6)

/-- Theorem stating the properties of the function f -/
theorem function_properties (ω φ : ℝ) 
  (h_ω : ω > 0) 
  (h_φ : 0 < φ ∧ φ < Real.pi) 
  (h_even : ∀ x, f ω φ x = f ω φ (-x))
  (h_symmetry : ∀ x, f ω φ (x + Real.pi / (2 * ω)) = f ω φ x) :
  (∀ x, f ω φ x = 2 * Real.cos (2 * x) + 1) ∧
  (Set.Icc (-Real.sqrt 2 + 1) 3 = 
    Set.image (f ω φ) (Set.Icc (-Real.pi / 8) (3 * Real.pi / 8))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1256_125619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_inequality_l1256_125698

theorem sqrt_inequality : Real.sqrt 6 + Real.sqrt 7 > 2 * Real.sqrt 2 + Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_inequality_l1256_125698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lolas_number_proof_l1256_125645

/-- Lola's number in the game -/
noncomputable def lolas_number : ℂ := 1.68 + 5.76 * Complex.I

/-- Mia's number in the game -/
def mias_number : ℂ := 4 - 3 * Complex.I

/-- The product of Lola's and Mia's numbers -/
def product : ℂ := 24 + 18 * Complex.I

theorem lolas_number_proof : lolas_number * mias_number = product := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lolas_number_proof_l1256_125645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_term_l1256_125665

noncomputable def geometric_sequence (a₁ : ℝ) (q : ℝ) : ℕ → ℝ
  | 0 => a₁
  | n + 1 => q * geometric_sequence a₁ q n

noncomputable def S (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

noncomputable def T (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a₁^n * q^((n * (n - 1)) / 2)

theorem max_product_term (a₁ q : ℝ) (h₁ : a₁ > 1) 
    (h₂ : geometric_sequence a₁ q 2016 * geometric_sequence a₁ q 2017 > 1)
    (h₃ : (geometric_sequence a₁ q 2016 - 1) / (geometric_sequence a₁ q 2017 - 1) < 0) :
  ∀ n : ℕ, T a₁ q 2016 ≥ T a₁ q n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_term_l1256_125665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chopped_cube_height_is_correct_l1256_125626

/-- The height of a unit cube with a corner chopped off, when placed on the cut face -/
noncomputable def chopped_cube_height : ℝ := 2 * Real.sqrt 3 / 3

/-- Theorem stating the height of a unit cube with a corner chopped off -/
theorem chopped_cube_height_is_correct :
  let cube_side_length : ℝ := 1
  let cube_diagonal : ℝ := Real.sqrt 3
  let cut_face_side_length : ℝ := Real.sqrt 2
  let cut_face_area : ℝ := Real.sqrt 3 / 2
  let chopped_pyramid_volume : ℝ := 1 / 6
  let chopped_pyramid_height : ℝ := Real.sqrt 3 / 3
  chopped_cube_height = cube_side_length - chopped_pyramid_height :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chopped_cube_height_is_correct_l1256_125626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1256_125635

/-- The function we want to maximize -/
noncomputable def f (x : ℝ) := Real.sqrt (x + 64) + Real.sqrt (25 - x) + 2 * Real.sqrt x

/-- The theorem stating the maximum value of the function -/
theorem max_value_of_f :
  ∃ (max_x : ℝ), 0 ≤ max_x ∧ max_x ≤ 25 ∧
  (∀ x, 0 ≤ x → x ≤ 25 → f x ≤ f max_x) ∧
  max_x = 64 / 15 ∧
  f max_x = Real.sqrt (64 / 15 + 64) + Real.sqrt (25 - 64 / 15) + 2 * Real.sqrt (64 / 15) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1256_125635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_point_sum_l1256_125623

noncomputable def f (x : ℝ) : ℝ := (3/2) * Real.sin (2*x) + (Real.sqrt 3 / 2) * Real.cos (2*x) + Real.pi/12

theorem symmetry_point_sum (a b : ℝ) :
  a ∈ Set.Ioo (-Real.pi/2) 0 →
  (∀ x : ℝ, f (a + (a - x)) = f x) →
  a + b = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_point_sum_l1256_125623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_vertices_l1256_125685

/-- Predicate to check if a set of complex numbers forms an equilateral triangle -/
def IsEquilateralTriangle (s : Set ℂ) : Prop :=
  ∃ (a b c : ℂ), s = {a, b, c} ∧ 
    Complex.abs (b - a) = Complex.abs (c - b) ∧ 
    Complex.abs (c - b) = Complex.abs (a - c)

/-- Given a complex number ω such that ω³ = 1 and ω ≠ 1, 
    the points z₁, z₂, -ωz₁, -ω²z₂ form the vertices of an equilateral triangle
    for any complex numbers z₁ and z₂. -/
theorem equilateral_triangle_vertices 
  (ω : ℂ) (h1 : ω^3 = 1) (h2 : ω ≠ 1) (z₁ z₂ : ℂ) : 
  IsEquilateralTriangle {z₁, z₂, -ω * z₁, -ω^2 * z₂} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_vertices_l1256_125685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_inequality_solution_l1256_125668

theorem sin_inequality_solution (x y : ℝ) :
  |Real.sin x - Real.sin y| + Real.sin x * Real.sin y ≤ 0 ↔ ∃ (n m : ℤ), x = n * Real.pi ∧ y = m * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_inequality_solution_l1256_125668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_limit_is_infinity_l1256_125637

noncomputable def sequence_limit (n : ℝ) : ℝ :=
  (n * n^(1/6) + (32*n^10 + 1)^(1/3)) / ((n + n^(1/4)) * (n^3 - 1)^(1/3))

theorem sequence_limit_is_infinity :
  Filter.Tendsto sequence_limit Filter.atTop Filter.atTop := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_limit_is_infinity_l1256_125637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_no_obtuse_triangle_is_correct_l1256_125669

/-- The probability that no set of three points among four randomly chosen points 
    on a circle forms an obtuse triangle with the circle's center -/
noncomputable def probability_no_obtuse_triangle : ℝ := (3 / 8) ^ 4

/-- Four points chosen uniformly at random on a circle -/
def random_points_on_circle : ℕ := 4

theorem probability_no_obtuse_triangle_is_correct :
  probability_no_obtuse_triangle = (3 / 8) ^ 4 :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_no_obtuse_triangle_is_correct_l1256_125669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1256_125675

-- Define the function f as noncomputable
noncomputable def f (a b x : ℝ) : ℝ := Real.log (a^x - b^x)

-- State the theorem
theorem function_properties (a b : ℝ) (h1 : a > 1) (h2 : 1 > b) (h3 : b > 0) :
  -- 1. Domain of f is (0, +∞)
  (∀ x : ℝ, x > 0 → a^x - b^x > 0) ∧
  -- 2. f is strictly increasing on its domain
  (∀ x y : ℝ, 0 < x ∧ x < y → f a b x < f a b y) ∧
  -- 3. For a ≥ b + 1, f(x) > 0 for all x > 1
  (a ≥ b + 1 → ∀ x : ℝ, x > 1 → f a b x > 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1256_125675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_y_coordinate_l1256_125622

noncomputable def r (θ : ℝ) : ℝ := Real.sin (3 * θ)

noncomputable def y (θ : ℝ) : ℝ := r θ * Real.sin θ

theorem max_y_coordinate :
  ∃ (y_max : ℝ), y_max = 1 ∧ ∀ (θ : ℝ), y θ ≤ y_max := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_y_coordinate_l1256_125622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1256_125663

noncomputable def A : Set ℝ := {x | (2 : ℝ)^(2*x + 1) ≥ 4}
def B : Set ℝ := {x | x < 2}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 1/2 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1256_125663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1256_125611

noncomputable def f (x : ℝ) := Real.sqrt (8 * x^2 + 20 * x - 9)

theorem domain_of_f :
  {x : ℝ | ∃ y : ℝ, f x = y} = {x : ℝ | x ≤ -3/2 ∨ x ≥ 3/4} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1256_125611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_line_intersection_l1256_125667

-- Define the curve C in polar coordinates
noncomputable def curve_C (θ : ℝ) : ℝ := 6 * Real.cos θ + 2 * Real.sin θ

-- Define the line l in parametric form
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (1 - Real.sqrt 2 * t, 2 + Real.sqrt 2 * t)

-- Point Q
def point_Q : ℝ × ℝ := (1, 2)

-- Theorem statement
theorem curve_and_line_intersection :
  -- The Cartesian equation of curve C
  (∀ x y : ℝ, (x - 3)^2 + (y - 1)^2 = 10 ↔ 
    ∃ θ : ℝ, x = curve_C θ * Real.cos θ ∧ y = curve_C θ * Real.sin θ) ∧
  -- The general equation of line l
  (∀ x y : ℝ, x + y = 3 ↔ ∃ t : ℝ, (x, y) = line_l t) ∧
  -- The product of distances |QA| * |QB|
  (∃ A B : ℝ × ℝ, 
    (∃ t₁ : ℝ, A = line_l t₁) ∧ 
    (∃ t₂ : ℝ, B = line_l t₂) ∧
    (∃ θ₁ : ℝ, A.1 = curve_C θ₁ * Real.cos θ₁ ∧ A.2 = curve_C θ₁ * Real.sin θ₁) ∧
    (∃ θ₂ : ℝ, B.1 = curve_C θ₂ * Real.cos θ₂ ∧ B.2 = curve_C θ₂ * Real.sin θ₂) ∧
    ((A.1 - point_Q.1)^2 + (A.2 - point_Q.2)^2) * 
    ((B.1 - point_Q.1)^2 + (B.2 - point_Q.2)^2) = 25) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_line_intersection_l1256_125667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_problem_l1256_125654

/-- The speed of a train traveling between two stations -/
noncomputable def train_speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

/-- The distance traveled by a train given its speed and time -/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

theorem train_speed_problem (total_distance : ℝ) (time_A : ℝ) (time_B : ℝ) (speed_B : ℝ) :
  total_distance = 155 →
  time_A = 4 →
  time_B = 3 →
  speed_B = 25 →
  ∃ (speed_A : ℝ), 
    speed_A = 20 ∧ 
    total_distance = distance_traveled speed_A time_A + distance_traveled speed_B time_B :=
by
  intro h1 h2 h3 h4
  use 20
  constructor
  · rfl
  · rw [h1, h2, h3, h4]
    simp [distance_traveled]
    norm_num

#check train_speed_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_problem_l1256_125654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l1256_125604

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (2 * Real.pi * x)^2

-- State the theorem
theorem derivative_of_f :
  deriv f = fun x => 8 * Real.pi^2 * x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l1256_125604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_view_angles_l1256_125614

theorem inscribed_circle_view_angles (A B C : ℝ) (h1 : A = 50) (h2 : B = 100) (h3 : A + B + C = 180) :
  let O1 := 180 - (A / 2 + B / 2)
  let O2 := 180 - (B / 2 + C / 2)
  let O3 := 180 - (C / 2 + A / 2)
  O1 = 105 ∧ O2 = 115 ∧ O3 = 140 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_view_angles_l1256_125614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dubois_car_payment_l1256_125602

/-- Calculates the number of months required to fully pay for a car -/
def months_to_pay (total_price initial_payment monthly_payment : ℚ) : ℕ :=
  (((total_price - initial_payment) / monthly_payment).ceil).toNat

/-- Theorem stating that it takes 19 months to fully pay for Mr. Dubois' car -/
theorem dubois_car_payment : 
  months_to_pay 13380 5400 420 = 19 := by
  sorry

#eval months_to_pay 13380 5400 420

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dubois_car_payment_l1256_125602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1256_125666

noncomputable def f (a ω b x : ℝ) : ℝ := a * Real.sin (2 * ω * x + Real.pi / 6) + a / 2 + b

theorem function_properties :
  ∀ (a ω b : ℝ),
  a > 0 →
  ω > 0 →
  (∀ x : ℝ, f a ω b (x + Real.pi / (2 * ω)) = f a ω b x) →
  (∀ x : ℝ, f a ω b x ≤ 7) →
  (∀ x : ℝ, f a ω b x ≥ 3) →
  (∃ x : ℝ, f a ω b x = 7) →
  (∃ x : ℝ, f a ω b x = 3) →
  (ω = 1 ∧ a = 2 ∧ b = 4) ∧
  (∀ x : ℝ, x ∈ Set.Icc (-Real.pi/3) (Real.pi/3) →
    f 2 1 4 (x + Real.pi/12) ∈ Set.Icc (5 - Real.sqrt 3) 7) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1256_125666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_cost_is_1_80_l1256_125608

/-- Represents the fundraising scenario for a school selling ice-pops to buy pencils -/
structure FundraisingScenario where
  selling_price : ℚ  -- Selling price of one ice-pop
  production_cost : ℚ  -- Cost to make one ice-pop
  pops_to_sell : ℕ  -- Number of ice-pops that need to be sold
  pencils_to_buy : ℕ  -- Number of pencils to be bought

/-- Calculates the cost of each pencil in the given fundraising scenario -/
def pencil_cost (scenario : FundraisingScenario) : ℚ :=
  ((scenario.selling_price - scenario.production_cost) * scenario.pops_to_sell) / scenario.pencils_to_buy

/-- Theorem stating that in the given scenario, the cost of each pencil is $1.80 -/
theorem pencil_cost_is_1_80 (scenario : FundraisingScenario) 
    (h1 : scenario.selling_price = 3/2)
    (h2 : scenario.production_cost = 9/10)
    (h3 : scenario.pops_to_sell = 300)
    (h4 : scenario.pencils_to_buy = 100) : 
  pencil_cost scenario = 9/5 := by
  sorry

#eval pencil_cost { selling_price := 3/2, production_cost := 9/10, pops_to_sell := 300, pencils_to_buy := 100 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_cost_is_1_80_l1256_125608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_sum_implies_sin_2x_l1256_125657

theorem trig_sum_implies_sin_2x (x : Real) 
  (h : Real.sin x + Real.cos x + Real.tan x + (Real.cos x / Real.sin x) + (1 / Real.cos x) + (1 / Real.sin x) = 9) :
  Real.sin (2 * x) = 40 - 10 * Real.sqrt 39 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_sum_implies_sin_2x_l1256_125657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discrete_rv_expectation_variance_l1256_125670

/-- A discrete random variable X with distribution P(X=k) = p^k * q^(1-k) where k = 0 or 1 -/
structure DiscreteRV (p q : ℝ) where
  prob : ℕ → ℝ
  property : ∀ k, prob k = p^k * q^(1-k) ∧ k ≤ 1

/-- The expectation of the discrete random variable X -/
def expectation (p q : ℝ) (X : DiscreteRV p q) : ℝ := 
  X.prob 0 * 0 + X.prob 1 * 1

/-- The variance of the discrete random variable X -/
def variance (p q : ℝ) (X : DiscreteRV p q) : ℝ := 
  X.prob 0 * (0 - expectation p q X)^2 + X.prob 1 * (1 - expectation p q X)^2

theorem discrete_rv_expectation_variance 
  (p q : ℝ) (hpq : p + q = 1) (X : DiscreteRV p q) : 
  expectation p q X = p ∧ variance p q X = p * (1 - p) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discrete_rv_expectation_variance_l1256_125670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_angle_condition_l1256_125653

/-- Given vectors a and b in ℝ², prove that the angle between a and a + λb is acute
    if and only if λ > -5/3 and λ ≠ 0 -/
theorem acute_angle_condition (a b : ℝ × ℝ) (lambda : ℝ) :
  a = (1, 2) →
  b = (1, 1) →
  (0 < a.1 * (a.1 + lambda * b.1) + a.2 * (a.2 + lambda * b.2) ↔ lambda > -5/3 ∧ lambda ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_angle_condition_l1256_125653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_time_calculation_l1256_125652

/-- The total time for a round trip between two points -/
noncomputable def round_trip_time (distance : ℝ) (speed_downhill speed_uphill : ℝ) : ℝ :=
  distance / speed_downhill + distance / speed_uphill

/-- Theorem: The round trip time for the given conditions is 5.25 hours -/
theorem round_trip_time_calculation : 
  round_trip_time 75.6 33.6 25.2 = 5.25 := by
  -- Unfold the definition of round_trip_time
  unfold round_trip_time
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_time_calculation_l1256_125652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ounces_per_gallon_l1256_125693

/-- Proves that there are 125 ounces in a gallon based on the given conditions -/
theorem ounces_per_gallon :
  ∃ (ounces_per_gallon : ℚ),
  let ounces_per_bowl : ℕ := 10
  let bowls_per_minute : ℕ := 5
  let total_gallons : ℕ := 6
  let total_minutes : ℕ := 15
  (total_gallons : ℚ) * ounces_per_gallon = 
    (ounces_per_bowl : ℚ) * (bowls_per_minute : ℚ) * (total_minutes : ℚ) ∧
  ounces_per_gallon = 125 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ounces_per_gallon_l1256_125693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_payment_is_31_l1256_125674

/-- Represents a four-digit number of the form 20** -/
def FourDigitNumber : Type := { n : ℕ // 2000 ≤ n ∧ n ≤ 2099 }

/-- Payments for divisibility by 1, 3, 5, 7, 9, 11 respectively -/
def Payments : List ℕ := [1, 3, 5, 7, 9, 11]

/-- Calculates the payment for a given number -/
def calculatePayment (n : FourDigitNumber) : ℕ :=
  (Payments.zip [1, 3, 5, 7, 9, 11]).foldl
    (fun acc (payment, divisor) => 
      if n.val % divisor = 0 then acc + payment else acc)
    0

/-- The maximum possible payment -/
def maxPayment : ℕ := 31

theorem max_payment_is_31 : 
  ∃ (n : FourDigitNumber), calculatePayment n = maxPayment ∧ 
  ∀ (m : FourDigitNumber), calculatePayment m ≤ maxPayment := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_payment_is_31_l1256_125674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_route_comparison_theorem_l1256_125625

/-- Represents a route with distance and speed information -/
structure Route where
  distance : ℝ
  speed : ℝ

/-- Represents a school zone with distance and speed information -/
structure SchoolZone where
  distance : ℝ
  speed : ℝ

/-- Calculates the time taken to travel a route -/
noncomputable def travelTime (r : Route) : ℝ :=
  r.distance / r.speed

/-- Calculates the time taken to travel through a school zone -/
noncomputable def schoolZoneTime (sz : SchoolZone) : ℝ :=
  sz.distance / sz.speed

/-- Theorem statement for the route comparison problem -/
theorem route_comparison_theorem (routeX : Route) (routeY : Route) 
  (schoolZone1 : SchoolZone) (schoolZone2 : SchoolZone) : 
  routeX.distance = 8 ∧ 
  routeX.speed = 32 ∧ 
  routeY.distance = 7 ∧ 
  routeY.speed = 45 ∧ 
  schoolZone1.distance = 1 ∧ 
  schoolZone1.speed = 25 ∧ 
  schoolZone2.distance = 0.5 ∧ 
  schoolZone2.speed = 15 → 
  abs ((travelTime routeX * 60) - 
    ((travelTime { distance := routeY.distance - (schoolZone1.distance + schoolZone2.distance), 
                   speed := routeY.speed } +
      schoolZoneTime schoolZone1 + 
      schoolZoneTime schoolZone2) * 60) - 3.27) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_route_comparison_theorem_l1256_125625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_square_height_l1256_125644

/-- The side length of each square -/
noncomputable def side_length : ℝ := 2

/-- The rotation angle in radians -/
noncomputable def rotation_angle : ℝ := Real.pi / 6

/-- The distance between centers of adjacent squares after rotation -/
noncomputable def distance_between_centers : ℝ := side_length * Real.sqrt 3

theorem rotated_square_height (B : ℝ × ℝ) :
  (∃ (A C : ℝ × ℝ),
    -- A, B, C form the rotated square
    dist A B = side_length ∧
    dist B C = side_length ∧
    dist A C = side_length * Real.sqrt 2 ∧
    -- The square is rotated by rotation_angle
    Real.cos rotation_angle = (C.1 - A.1) / (side_length * Real.sqrt 2) ∧
    -- The rotated square touches both adjacent squares
    C.1 - A.1 = distance_between_centers) →
  -- The height of point B is equal to the side length
  B.2 = side_length := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_square_height_l1256_125644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sqrt2_over_2_condition_l1256_125618

theorem cos_sqrt2_over_2_condition (k : ℤ) (α : ℝ) :
  (α = 2 * k * Real.pi - Real.pi / 4 → Real.cos α = Real.sqrt 2 / 2) ∧
  ∃ β, Real.cos β = Real.sqrt 2 / 2 ∧ ∀ m : ℤ, β ≠ 2 * m * Real.pi - Real.pi / 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sqrt2_over_2_condition_l1256_125618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequalities_proof_l1256_125649

theorem inequalities_proof (a b : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : a ≠ b) (h4 : a + b = 2) :
  (1 / a + 1 / b > 2) ∧ (1 / a^2 + 1 / b^2 > 2) ∧ ((2:ℝ)^a + (2:ℝ)^b > 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequalities_proof_l1256_125649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_magnitude_l1256_125631

theorem order_of_magnitude (a b c d : ℝ) 
  (ha : a = Real.rpow 1.7 0.3)
  (hb : b = Real.rpow 0.9 0.1)
  (hc : c = Real.log 5 / Real.log 2)
  (hd : d = Real.log 1.8 / Real.log 0.3) :
  c > a ∧ a > b ∧ b > d := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_magnitude_l1256_125631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l1256_125660

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y = (1/4)x^2 -/
def Parabola : Point → Prop :=
  fun p => p.y = (1/4) * p.x^2

/-- The focus of the parabola y = (1/4)x^2 -/
def Focus : Point :=
  { x := 0, y := 1 }

/-- Angle of the line passing through the focus -/
noncomputable def Angle : ℝ := 30 * Real.pi / 180

/-- Slope of the line passing through the focus -/
noncomputable def Slope : ℝ := Real.tan Angle

/-- Line passing through the focus at the given angle -/
def Line : Point → Prop :=
  fun p => p.y - Focus.y = Slope * (p.x - Focus.x)

/-- Theorem: The length of the chord AB is 16/3 -/
theorem chord_length :
  ∃ (A B : Point),
    Parabola A ∧ Parabola B ∧
    Line A ∧ Line B ∧
    A ≠ B ∧
    ((A.x - B.x)^2 + (A.y - B.y)^2)^(1/2) = 16/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l1256_125660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_time_proof_l1256_125630

/-- Calculates the travel time given distance and speed -/
noncomputable def travel_time (distance : ℝ) (speed : ℝ) : ℝ :=
  distance / speed

/-- Proves that the travel time is 5 hours given the conditions -/
theorem travel_time_proof (distance speed : ℝ) 
  (h1 : distance = 300) 
  (h2 : speed = 60) : 
  travel_time distance speed = 5 := by
  -- Unfold the definition of travel_time
  unfold travel_time
  -- Rewrite using the given hypotheses
  rw [h1, h2]
  -- Simplify the division
  norm_num

#check travel_time_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_travel_time_proof_l1256_125630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_in_interval_l1256_125661

noncomputable def f (x : ℝ) := Real.log x - 2 / x

theorem zero_point_in_interval :
  ∃! x : ℝ, 2 < x ∧ x < Real.exp 1 ∧ f x = 0 :=
by
  have h1 : Continuous f := by sorry
  have h2 : StrictMono f := by sorry
  have h3 : f 2 < 0 := by sorry
  have h4 : f (Real.exp 1) > 0 := by sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_in_interval_l1256_125661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_one_fifth_l1256_125641

-- Define the lines
noncomputable def line1 (x : ℝ) : ℝ := 2 * x + 1
noncomputable def line2 (x : ℝ) : ℝ := -1/2 * x + 4
noncomputable def line3 : ℝ := 3

-- Define the vertices of the triangle
noncomputable def vertex1 : ℝ × ℝ := (1, 3)
noncomputable def vertex2 : ℝ × ℝ := (2, 3)
noncomputable def vertex3 : ℝ × ℝ := (6/5, 17/5)

-- Define the triangle area
noncomputable def triangle_area (v1 v2 v3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := v1
  let (x2, y2) := v2
  let (x3, y3) := v3
  (1/2) * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

-- Theorem statement
theorem triangle_area_is_one_fifth :
  triangle_area vertex1 vertex2 vertex3 = 1/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_one_fifth_l1256_125641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2014_equals_3_l1256_125647

-- Define the function f
noncomputable def f (a b α β x : ℝ) : ℝ := a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β) + 4

-- State the theorem
theorem f_2014_equals_3 
  (a b α β : ℝ) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hα : α ≠ 0) 
  (hβ : β ≠ 0) 
  (h_2013 : f a b α β 2013 = 5) : 
  f a b α β 2014 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2014_equals_3_l1256_125647
