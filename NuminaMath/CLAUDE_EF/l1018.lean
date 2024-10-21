import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perp_plane_theorem_necessary_not_sufficient_condition_l1018_101887

-- Define the necessary structures
structure Plane where

structure Line where

structure Point where

-- Define the perpendicular relation between planes
def perpendicular (α β : Plane) : Prop := sorry

-- Define the intersection of planes
def intersection (α β : Plane) : Line := sorry

-- Define a point being in a plane
def in_plane (P : Point) (α : Plane) : Prop := sorry

-- Define a line being perpendicular to another line
def line_perp_line (l₁ l₂ : Line) : Prop := sorry

-- Define a line being perpendicular to a plane
def line_perp_plane (l : Line) (α : Plane) : Prop := sorry

-- Theorem for proposition ②
theorem perp_plane_theorem (α β : Plane) (l : Line) (P : Point) :
  perpendicular α β →
  intersection α β = l →
  in_plane P α →
  (∃ m : Line, line_perp_line m l ∧ line_perp_plane m β) := by
  sorry

-- Theorem for proposition ④
theorem necessary_not_sufficient_condition (a : ℝ) :
  (a^2 < 2*a → a < 2) ∧ ¬(a < 2 → a^2 < 2*a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perp_plane_theorem_necessary_not_sufficient_condition_l1018_101887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mn_length_formula_l1018_101858

/-- Represents a trapezoid ABCD with points M on AC and N on BD -/
structure Trapezoid where
  -- Bases of the trapezoid
  a : ℝ
  b : ℝ
  -- Condition that a > b
  h_ab : a > b
  -- Ratio condition for points M and N
  h_ratio : ∃ (k : ℝ), k > 0 ∧ k = 1/4

/-- The length of MN in the trapezoid -/
noncomputable def mn_length (t : Trapezoid) : ℝ := (1/5) * (4 * t.a - t.b)

/-- Theorem stating that MN = (1/5)(4a - b) in the described trapezoid -/
theorem mn_length_formula (t : Trapezoid) : 
  ∃ (m n : ℝ), m - n = mn_length t := by
  sorry

#check mn_length_formula

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mn_length_formula_l1018_101858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_analysis_l1018_101872

noncomputable section

-- Define the function f and its derivative f'
def f (m : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + m * x^2 + 1
def f' (m : ℝ) (x : ℝ) : ℝ := x^2 + 2 * m * x

-- State the theorem
theorem function_analysis (m : ℝ) :
  (f' m 1 = 3) →
  (m = 1) ∧
  (∃ A B C : ℝ, A * 1 + B * (f 1 1) + C = 0 ∧ 
              A = 3 ∧ B = -3 ∧ C = 4) ∧
  (∀ x : ℝ, x < -2 → (f' 1 x > 0)) ∧
  (∀ x : ℝ, -2 < x ∧ x < 0 → (f' 1 x < 0)) ∧
  (∀ x : ℝ, 0 < x → (f' 1 x > 0)) ∧
  (f' 1 (-2) = 0) ∧
  (f' 1 0 = 0) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_analysis_l1018_101872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1018_101852

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem f_properties :
  -- 1. Line of symmetry
  (∀ k : ℤ, ∀ x : ℝ, f (Real.pi/12 + k*Real.pi/2 + x) = f (Real.pi/12 + k*Real.pi/2 - x)) ∧
  -- 2. Point symmetry
  (∀ k : ℤ, ∀ x : ℝ, f (-Real.pi/6 + k*Real.pi/2 + x) = -f (-Real.pi/6 + k*Real.pi/2 - x)) ∧
  -- 3. Shifting results in cosine
  (∀ x : ℝ, f (x + Real.pi/12) = Real.cos (2*x)) ∧
  -- 4. Smallest positive period
  (∀ x : ℝ, f (x + Real.pi) = f x) ∧
  (∀ T : ℝ, 0 < T → T < Real.pi → ∃ x : ℝ, f (x + T) ≠ f x) ∧
  -- 5. Increasing interval
  (∀ k : ℤ, ∀ x y : ℝ, -5*Real.pi/12 + k*Real.pi/2 ≤ x → x < y → y ≤ Real.pi/12 + k*Real.pi/2 → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1018_101852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_derivative_f_l1018_101825

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt x - Real.log x

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := 1 / (2 * Real.sqrt x) - 1 / x

-- Theorem statement
theorem max_derivative_f :
  ∃ (c : ℝ), c > 0 ∧ ∀ (x : ℝ), x > 0 → f' x ≤ 1/16 ∧ f' c = 1/16 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_derivative_f_l1018_101825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_uniqueness_l1018_101889

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * a * x^3 + (3-a) * x^2 - 7*x + 5

noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2*(3-a)*x - 7

theorem f_uniqueness (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ x ∈ Set.Icc (-2) 2, |f_derivative a x| ≤ 7) :
  f a = λ x => x^3 - 7*x + 5 := by
  sorry

#check f_uniqueness

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_uniqueness_l1018_101889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wolf_goat_cabbage_problem_l1018_101880

-- Define the items
inductive Item
| Wolf
| Goat
| Cabbage

-- Define the bank of the river
inductive Bank
| Left
| Right

-- Define the state of the system
structure State where
  farmerBank : Bank
  wolfBank : Bank
  goatBank : Bank
  cabbageBank : Bank

-- Define a valid move
def validMove (s1 s2 : State) : Prop :=
  -- The farmer always moves
  s1.farmerBank ≠ s2.farmerBank ∧
  -- Only one item can move with the farmer
  ((s1.wolfBank ≠ s2.wolfBank ∧ s1.goatBank = s2.goatBank ∧ s1.cabbageBank = s2.cabbageBank) ∨
   (s1.wolfBank = s2.wolfBank ∧ s1.goatBank ≠ s2.goatBank ∧ s1.cabbageBank = s2.cabbageBank) ∨
   (s1.wolfBank = s2.wolfBank ∧ s1.goatBank = s2.goatBank ∧ s1.cabbageBank ≠ s2.cabbageBank) ∨
   (s1.wolfBank = s2.wolfBank ∧ s1.goatBank = s2.goatBank ∧ s1.cabbageBank = s2.cabbageBank))

-- Define a safe state
def safeState (s : State) : Prop :=
  (s.wolfBank = s.goatBank → s.farmerBank = s.goatBank) ∧
  (s.goatBank = s.cabbageBank → s.farmerBank = s.goatBank)

-- Define the initial and goal states
def initialState : State :=
  { farmerBank := Bank.Left, wolfBank := Bank.Left, goatBank := Bank.Left, cabbageBank := Bank.Left }

def goalState : State :=
  { farmerBank := Bank.Right, wolfBank := Bank.Right, goatBank := Bank.Right, cabbageBank := Bank.Right }

-- Theorem: There exists a sequence of valid moves from the initial state to the goal state
-- while maintaining safe states throughout the sequence
theorem wolf_goat_cabbage_problem :
  ∃ (n : ℕ) (sequence : Fin (n + 1) → State),
    sequence 0 = initialState ∧
    sequence (Fin.last n) = goalState ∧
    (∀ i : Fin n, validMove (sequence i) (sequence i.succ)) ∧
    (∀ i : Fin (n + 1), safeState (sequence i)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wolf_goat_cabbage_problem_l1018_101880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_part1_range_of_b_part2_l1018_101879

-- Define the function f
def f (x b : ℝ) : ℝ := |x - b| + |x + b|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f x 1 ≤ x + 2} = Set.Icc 0 2 := by sorry

-- Part 2
theorem range_of_b_part2 :
  {b : ℝ | ∀ a : ℝ, a ≠ 0 → f 1 b ≥ (|a + 1| - |2*a - 1|) / |a|} =
  Set.Iic (-3/2) ∪ Set.Ici (3/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_part1_range_of_b_part2_l1018_101879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_poet_birthday_l1018_101809

/-- The number of days in a week -/
def daysInWeek : Nat := 7

/-- The number of years between the birth and the anniversary -/
def yearsBetween : Nat := 300

/-- The day of the week of the 300th anniversary (0 = Sunday, 1 = Monday, ..., 6 = Saturday) -/
def anniversaryDay : Fin 7 := 3  -- Wednesday

/-- The number of leap years in a 300-year period, excluding century years not divisible by 400 -/
def leapYears : Nat := (yearsBetween / 4) - 2

/-- The number of regular years in a 300-year period -/
def regularYears : Nat := yearsBetween - leapYears

/-- The total number of days to move back -/
def totalDaysBack : Nat := regularYears + 2 * leapYears

theorem poet_birthday :
  (anniversaryDay - (totalDaysBack % daysInWeek) : Fin 7) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_poet_birthday_l1018_101809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_is_four_percent_l1018_101895

/- Define the profit percentages -/
def profit_with_discount : ℚ := 44
def profit_without_discount : ℚ := 50

/- Define the function to calculate the selling price given a cost price and profit percentage -/
def selling_price (cost_price : ℚ) (profit_percentage : ℚ) : ℚ :=
  cost_price * (1 + profit_percentage / 100)

/- Define the function to calculate the discount percentage -/
def discount_percentage (cost_price : ℚ) : ℚ :=
  let sp_with_discount := selling_price cost_price profit_with_discount
  let sp_without_discount := selling_price cost_price profit_without_discount
  ((sp_without_discount - sp_with_discount) / sp_without_discount) * 100

/- Theorem statement -/
theorem discount_is_four_percent (cost_price : ℚ) (h : cost_price > 0) :
  discount_percentage cost_price = 4 := by
  sorry

#eval discount_percentage 100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_is_four_percent_l1018_101895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1018_101800

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * 2^x + 1) / (2^x + 1)

-- Theorem stating the properties of f
theorem f_properties (a : ℝ) :
  (∀ x, f a x = -f a (-x)) →  -- f is an odd function
  (a = -1 ∧
   (∀ x y, x < y → f a x > f a y) ∧
   (∀ t k, t ∈ Set.Icc 1 2 → f a (t^2 - 2*t) + f a (2*t^2 - k) > 0 → k > 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1018_101800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_29_l1018_101875

theorem sum_of_divisors_29 : 
  (Finset.filter (· ∣ 29) (Finset.range 30)).sum id = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_29_l1018_101875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_numbers_count_l1018_101851

/-- The set of digits used to form the numbers -/
def digits : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- The number of digits in each formed number -/
def number_length : ℕ := 3

/-- The theorem stating that the number of three-digit numbers
    formed using digits 1 to 6 (with repetition) is 216 -/
theorem three_digit_numbers_count :
  Fintype.card (Fin number_length → digits) = 216 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_numbers_count_l1018_101851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_firetruck_reachable_area_l1018_101843

/-- Represents the speed of the firetruck on highways in miles per hour -/
noncomputable def highway_speed : ℝ := 50

/-- Represents the speed of the firetruck across the prairie in miles per hour -/
noncomputable def prairie_speed : ℝ := 14

/-- Represents the time limit in hours -/
noncomputable def time_limit : ℝ := 1/10

/-- Represents the area of the region reachable by the firetruck within the time limit -/
noncomputable def reachable_area : ℝ := 700/31

theorem firetruck_reachable_area :
  let max_highway_distance := highway_speed * time_limit
  let max_prairie_distance := prairie_speed * time_limit
  ∃ (area_function : ℝ → ℝ),
    (∀ x, 0 ≤ x → x ≤ max_highway_distance →
      area_function x = 4 * (x^2 + x * (max_prairie_distance - x))) ∧
    (∃ x, 0 < x ∧ x < max_highway_distance ∧ area_function x = reachable_area) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_firetruck_reachable_area_l1018_101843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_shift_even_condition_l1018_101826

-- Define the function f
noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := Real.cos (x + φ)

-- Define what it means for a function to be even
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

-- State the theorem
theorem cos_shift_even_condition (φ : ℝ) :
  (φ = 0 → is_even (f φ)) ∧ 
  ¬(is_even (f φ) → φ = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_shift_even_condition_l1018_101826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_properties_l1018_101884

/-- An isosceles right triangle with side length 8 -/
structure IsoscelesRightTriangle where
  /-- The length of one of the equal sides -/
  side : ℝ
  /-- The side length is positive -/
  side_pos : side > 0

/-- Properties of an isosceles right triangle -/
def IsoscelesRightTriangle.properties (t : IsoscelesRightTriangle) : Prop :=
  t.side = 8

/-- The hypotenuse of an isosceles right triangle -/
noncomputable def hypotenuse (t : IsoscelesRightTriangle) : ℝ :=
  t.side * Real.sqrt 2

/-- The area of an isosceles right triangle -/
noncomputable def area (t : IsoscelesRightTriangle) : ℝ :=
  (1 / 2) * t.side * t.side

/-- Theorem about the hypotenuse and area of an isosceles right triangle -/
theorem isosceles_right_triangle_properties (t : IsoscelesRightTriangle) 
  (h : t.properties) : 
  hypotenuse t = 8 * Real.sqrt 2 ∧ area t = 32 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_properties_l1018_101884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_is_long_jump_results_l1018_101839

/-- Represents a long jump result -/
structure LongJumpResult where
  distance : ℝ

/-- Represents a sample in a statistical analysis -/
structure Sample where
  data : List LongJumpResult
  size : ℕ

/-- Represents the context of the statistical analysis -/
structure AnalysisContext where
  city : String
  schoolLevel : String
  healthAspect : String

/-- The specific analysis context for this problem -/
def problemContext : AnalysisContext := {
  city := "a certain city",
  schoolLevel := "middle school",
  healthAspect := "physical health status"
}

/-- The sample size in this problem -/
def sampleSize : ℕ := 12000

/-- Theorem stating that the sample in this analysis is the set of long jump results -/
theorem sample_is_long_jump_results (results : List LongJumpResult) 
  (h1 : results.length = sampleSize) 
  (h2 : ∀ r ∈ results, r.distance > 0) :
  ∃ (s : Sample), s.data = results ∧ s.size = sampleSize := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_is_long_jump_results_l1018_101839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_of_geometric_sequence_l1018_101886

-- Define a geometric sequence
def geometric_sequence (a r : ℝ) (n : ℕ) : ℝ := a * r ^ (n - 1)

-- State the theorem
theorem first_term_of_geometric_sequence
  (a r : ℝ) -- a is the first term, r is the common ratio
  (h1 : geometric_sequence a r 4 = Nat.factorial 6)
  (h2 : geometric_sequence a r 7 = Nat.factorial 7) :
  a = 720 / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_of_geometric_sequence_l1018_101886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_rational_l1018_101890

theorem all_rational : 
  ∀ (a b c d : ℝ),
    (a = ((9/16 : ℝ)^2).sqrt) →
    (b = (0.125 : ℝ) ^ (1/3 : ℝ)) →
    (c = (0.004096 : ℝ) ^ (1/4 : ℝ)) →
    (d = (-8 : ℝ) ^ (1/3 : ℝ) * (0.25⁻¹ : ℝ).sqrt) →
    (∃ (w x y z : ℚ), a = w ∧ b = x ∧ c = y ∧ d = z) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_rational_l1018_101890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_variation_problem_l1018_101803

/-- Given that x² varies inversely with y⁴, and x = 5 when y = 2, prove that x² = 25/16 when y = 4 -/
theorem inverse_variation_problem (x y : ℝ) (k : ℝ) (h1 : x^2 * y^4 = k) (h2 : 5^2 * 2^4 = k) :
  y = 4 → x^2 = 25/16 := by
  intro h3
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_variation_problem_l1018_101803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_constant_l1018_101813

/-- Calculates the difference in interest earned between two interest rates over a fixed period --/
noncomputable def interestDifference (principal : ℝ) (time : ℝ) (baseRate : ℝ) (rateIncrease : ℝ) : ℝ :=
  principal * time * (baseRate + rateIncrease) / 100 - principal * time * baseRate / 100

theorem interest_difference_constant 
  (principal : ℝ) (time : ℝ) (baseRate : ℝ) (rateIncrease : ℝ) :
  principal = 300 ∧ time = 10 ∧ rateIncrease = 5 →
  interestDifference principal time baseRate rateIncrease = 150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_constant_l1018_101813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_product_l1018_101821

theorem polynomial_roots_product (a b c : ℝ) : 
  (∀ x, x^3 + a*x^2 + b*x + c = 0 ↔ 
    x = Real.cos (2*Real.pi/7) ∨ 
    x = Real.cos (4*Real.pi/7) ∨ 
    x = Real.cos (6*Real.pi/7)) →
  a * b * c = 1/32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_product_l1018_101821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equations_l1018_101862

-- Define the line l
def line_l (x y : ℝ) : Prop := 5 * x - 3 * y + 15 = 0

-- Define the point that line l passes through
def point_l : ℝ × ℝ := (0, 5)

-- Define the sum of intercepts
def sum_of_intercepts : ℝ := 2

-- Define line l1
def line_l1 (x y : ℝ) : Prop := 3 * x + 5 * y - 3 = 0

-- Define the point that line l1 passes through
noncomputable def point_l1 : ℝ × ℝ := (8/3, -1)

-- Define line l2
def line_l2 (x y : ℝ) : Prop := 3 * x - 5 * y - 3 = 0

-- Theorem statement
theorem line_equations :
  (∀ x y : ℝ, line_l x y ↔ 5 * x - 3 * y + 15 = 0) ∧
  (∀ x y : ℝ, line_l2 x y ↔ 3 * x - 5 * y - 3 = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equations_l1018_101862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_range_l1018_101806

/-- Given a function f(x) = 2sin(ωx + φ), prove that the range of φ is (π/4, π/3) under certain conditions. -/
theorem phi_range (ω φ : ℝ) : 
  ω > 0 → 
  |φ| < π / 2 →
  (∀ x : ℝ, -π/6 < x → x < π → 2 * Real.sin (ω * x + φ) > 1) →
  (∀ x : ℝ, 2 * Real.sin (ω * x + φ) = 2 * Real.sin (ω * (x + 2*π) + φ)) →
  π/4 < φ ∧ φ < π/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_range_l1018_101806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_equals_negative_four_sqrt_three_fifths_l1018_101894

theorem sin_sum_equals_negative_four_sqrt_three_fifths (α : ℝ) 
  (h1 : Real.cos (α + 2/3 * Real.pi) = 4/5) 
  (h2 : -Real.pi/2 < α) 
  (h3 : α < 0) : 
  Real.sin (α + Real.pi/3) + Real.sin α = -(4 * Real.sqrt 3)/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_equals_negative_four_sqrt_three_fifths_l1018_101894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_of_sum_l1018_101837

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the circle (renamed to avoid conflict)
def my_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1/4

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1)

-- Define the intersection points
def intersection_points (A B C D : ℝ × ℝ) (k : ℝ) : Prop :=
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ parabola C.1 C.2 ∧ parabola D.1 D.2 ∧
  my_circle A.1 A.2 ∧ my_circle B.1 B.2 ∧ my_circle C.1 C.2 ∧ my_circle D.1 D.2 ∧
  line_through_focus k A.1 A.2 ∧ line_through_focus k B.1 B.2 ∧
  line_through_focus k C.1 C.2 ∧ line_through_focus k D.1 D.2 ∧
  A.2 ≥ B.2 ∧ B.2 ≥ C.2 ∧ C.2 ≥ D.2

-- Theorem statement
theorem minimum_value_of_sum (A B C D : ℝ × ℝ) (k : ℝ) :
  intersection_points A B C D k →
  (A.1 + 1/2) + 4*(D.1 + 1/2) ≥ 13/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_of_sum_l1018_101837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_harvest_and_earnings_l1018_101869

/-- Represents a triangular vegetable plot -/
structure TriangularPlot where
  base : ℝ
  height : ℝ

/-- Represents the harvest yield and price information -/
structure HarvestInfo where
  yieldPerSquareMeter : ℝ
  pricePerKilogram : ℝ

/-- Calculate the area of a triangular plot -/
noncomputable def plotArea (plot : TriangularPlot) : ℝ :=
  plot.base * plot.height / 2

/-- Calculate the total harvest from a plot -/
noncomputable def totalHarvest (plot : TriangularPlot) (info : HarvestInfo) : ℝ :=
  plotArea plot * info.yieldPerSquareMeter

/-- Calculate the total earnings from selling the harvest -/
noncomputable def totalEarnings (plot : TriangularPlot) (info : HarvestInfo) : ℝ :=
  totalHarvest plot info * info.pricePerKilogram

/-- Theorem stating the correct total harvest and earnings for the given plot and harvest info -/
theorem correct_harvest_and_earnings (plot : TriangularPlot) (info : HarvestInfo)
  (h1 : plot.base = 50)
  (h2 : plot.height = 28)
  (h3 : info.yieldPerSquareMeter = 15)
  (h4 : info.pricePerKilogram = 0.5) :
  totalHarvest plot info = 10500 ∧ totalEarnings plot info = 5250 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_harvest_and_earnings_l1018_101869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tin_weight_loss_l1018_101896

/-- Represents the properties of a metal bar made of tin and silver -/
structure MetalBar where
  total_weight : ℝ
  weight_loss_in_water : ℝ
  tin_silver_ratio : ℝ
  silver_weight_loss : ℝ
  silver_reference_weight : ℝ

/-- Calculates the amount of tin that loses a specific weight in water -/
noncomputable def tin_weight_for_loss (bar : MetalBar) (tin_loss : ℝ) : ℝ :=
  let total_weight_in_water := bar.total_weight - bar.weight_loss_in_water
  let silver_weight := bar.total_weight / (1 + bar.tin_silver_ratio)
  let tin_weight := bar.total_weight - silver_weight
  tin_weight

/-- Theorem stating the amount of tin that loses 1.375 kg in water for the given metal bar -/
theorem tin_weight_loss (bar : MetalBar) :
  bar.total_weight = 40 ∧
  bar.weight_loss_in_water = 4 ∧
  bar.tin_silver_ratio = 0.6666666666666664 ∧
  bar.silver_weight_loss = 0.375 ∧
  bar.silver_reference_weight = 5 →
  tin_weight_for_loss bar 1.375 = 15.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tin_weight_loss_l1018_101896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_and_function_properties_l1018_101882

theorem triangle_and_function_properties :
  ∀ (A B C : ℝ) (a b c : ℝ) (lambda omega : ℝ),
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a * Real.sin B + Real.sqrt 3 * a * Real.cos B = Real.sqrt 3 * c ∧
  lambda > 0 ∧ omega > 0 ∧
  (lambda * (Real.cos (omega * 0 + A / 2))^2 - 3 ≤ 2) ∧
  (∀ x, lambda * (Real.cos (omega * x + A / 2))^2 - 3 ≤ 2) ∧
  (∃ k, k > 0 ∧ ∀ x, lambda * (Real.cos (omega * (x / 1.5) + A / 2))^2 - 3 = lambda * (Real.cos (omega * x + A / 2))^2 - 3) ∧
  (∀ x, lambda * (Real.cos (omega * (x / 1.5) + A / 2))^2 - 3 = lambda * (Real.cos (omega * (x + π) + A / 2))^2 - 3) →
  A = π / 3 ∧
  (∀ x, 0 ≤ x ∧ x ≤ π / 2 →
    -3 ≤ lambda * (Real.cos (omega * x + π / 6))^2 - 3 ∧
    lambda * (Real.cos (omega * x + π / 6))^2 - 3 ≤ (5 * Real.sqrt 3 - 2) / 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_and_function_properties_l1018_101882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_n_ubiquitous_l1018_101833

/-- An infinite periodic word consisting of letters 'a' and 'b' -/
structure InfiniteWord where
  sequence : ℕ → Char
  period : ℕ
  is_periodic : ∀ i : ℕ, sequence i = sequence (i % period)
  alphabet : ∀ i : ℕ, sequence i = 'a' ∨ sequence i = 'b'

/-- A finite word appearing in an infinite word -/
def appears (W : InfiniteWord) (U : List Char) : Prop :=
  ∃ k : ℕ, ∀ i : Fin U.length, W.sequence (k + i) = U.get i

/-- A ubiquitous word in an infinite word -/
def ubiquitous (W : InfiniteWord) (U : List Char) : Prop :=
  appears W (U ++ ['a']) ∧
  appears W (U ++ ['b']) ∧
  appears W ('a' :: U) ∧
  appears W ('b' :: U)

/-- The number of ubiquitous words in an infinite word -/
noncomputable def num_ubiquitous (W : InfiniteWord) : ℕ :=
  Nat.card { U : List Char | ubiquitous W U ∧ U.length > 0 }

/-- Main theorem -/
theorem at_least_n_ubiquitous (n : ℕ) (W : InfiniteWord) 
    (h_n : n > 0) (h_period : W.period > 2^n) :
    num_ubiquitous W ≥ n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_n_ubiquitous_l1018_101833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l1018_101842

-- Define the points
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (1, 2)
def B (m : ℝ) : ℝ × ℝ := (m, 6)

-- Define the vectors
def OA : ℝ × ℝ := A
def AB (m : ℝ) : ℝ × ℝ := (B m - A)

-- Define the dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem perpendicular_vectors (m : ℝ) :
  dot_product OA (AB m) = 0 → m = -7 := by
  intro h
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l1018_101842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_l1018_101838

theorem trigonometric_expression (α : ℝ) (h : Real.tan α = 2) :
  (4 * (Real.sin α)^3 - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_l1018_101838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1018_101840

/-- The function f(x) = x - 4 * ln(x) -/
noncomputable def f (x : ℝ) : ℝ := x - 4 * Real.log x

/-- The derivative of f(x) -/
noncomputable def f_derivative (x : ℝ) : ℝ := 1 - 4 / x

theorem tangent_line_equation :
  ∀ x y : ℝ, y = f x → (x = 1 → 3 * x + y - 4 = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1018_101840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_factorial_sum_l1018_101823

theorem greatest_prime_factor_factorial_sum : 
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ (Nat.factorial 15 + Nat.factorial 18) ∧ 
  ∀ q : ℕ, Nat.Prime q → q ∣ (Nat.factorial 15 + Nat.factorial 18) → q ≤ p :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_factorial_sum_l1018_101823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_value_l1018_101863

/-- The equation of an ellipse with parameter m and focal length 2 -/
def ellipse_equation (x y m : ℝ) : Prop :=
  x^2 + m * y^2 = 1

/-- The focal length of an ellipse -/
noncomputable def focal_length (a b : ℝ) : ℝ :=
  2 * Real.sqrt (a^2 - b^2)

/-- Theorem: For an ellipse with equation x^2 + my^2 = 1 and focal length 2, m = 1/2 -/
theorem ellipse_m_value :
  ∀ m : ℝ, (∃ a b : ℝ, a > b ∧ b > 0 ∧
    (∀ x y : ℝ, ellipse_equation x y m ↔ (x/a)^2 + (y/b)^2 = 1) ∧
    focal_length a b = 2) →
  m = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_value_l1018_101863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_parabolas_l1018_101854

/-- A parabola with vertex on the x-axis and directrix as the y-axis -/
structure Parabola where
  a : ℝ  -- x-coordinate of the vertex
  p : ℝ  -- parameter of the parabola

/-- The squared distance from a point (x, y) to A(4, 0) -/
def squared_distance (x y : ℝ) : ℝ :=
  (x - 4)^2 + y^2

/-- The point on the parabola closest to A(4, 0) -/
noncomputable def closest_point (para : Parabola) : ℝ × ℝ :=
  let x := 4 - para.p / 2
  let y := Real.sqrt (2 * para.p * (x - para.a))
  (x, y)

/-- The statement to be proved -/
theorem count_parabolas :
  ∃ (s : Finset Parabola),
    s.card = 3 ∧
    (∀ para : Parabola, para ∈ s ↔
      (para.a = para.p / 2 ∧
       squared_distance (closest_point para).1 (closest_point para).2 = 4 ∧
       ∀ x y : ℝ, y^2 = 2 * para.p * (x - para.a) →
         squared_distance x y ≥ 4)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_parabolas_l1018_101854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_storage_cost_is_120_l1018_101849

/-- Represents the dimensions of a storage box in inches -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ := d.length * d.width * d.height

/-- Represents the storage problem parameters -/
structure StorageProblem where
  boxDim : BoxDimensions
  totalVolume : ℕ
  costPerBox : ℚ

/-- Calculates the number of boxes given the total volume and box volume -/
def numberOfBoxes (p : StorageProblem) : ℕ :=
  p.totalVolume / boxVolume p.boxDim

/-- Calculates the total monthly cost for storage -/
def totalMonthlyCost (p : StorageProblem) : ℚ :=
  (numberOfBoxes p : ℚ) * p.costPerBox

/-- Theorem stating that the total monthly cost for the given problem is $120 -/
theorem storage_cost_is_120 (p : StorageProblem) 
  (h1 : p.boxDim = { length := 15, width := 12, height := 10 })
  (h2 : p.totalVolume = 1080000)
  (h3 : p.costPerBox = 1/5) : 
  totalMonthlyCost p = 120 := by
  sorry

#eval totalMonthlyCost 
  { boxDim := { length := 15, width := 12, height := 10 },
    totalVolume := 1080000,
    costPerBox := 1/5 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_storage_cost_is_120_l1018_101849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_with_MN_diameter_l1018_101817

-- Define the points M and N
def M : ℝ × ℝ := (0, 2)
def N : ℝ × ℝ := (2, -2)

-- Define the midpoint O
noncomputable def O : ℝ × ℝ := ((M.1 + N.1) / 2, (M.2 + N.2) / 2)

-- Define the radius
noncomputable def r : ℝ := Real.sqrt ((N.1 - M.1)^2 + (N.2 - M.2)^2) / 2

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  (x - O.1)^2 + (y - O.2)^2 = r^2

-- Theorem statement
theorem circle_with_MN_diameter :
  ∀ x y : ℝ, circle_equation x y ↔ (x - 1)^2 + y^2 = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_with_MN_diameter_l1018_101817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_relations_l1018_101824

-- Define the points
variable (A B C D M N P Q O : ℝ × ℝ)

-- Define the quadrilateral ABCD
def is_convex_quadrilateral (A B C D : ℝ × ℝ) : Prop := sorry

-- Define the intersection point O
def intersect_at (O A B C D : ℝ × ℝ) : Prop := sorry

-- Define midpoints
def is_midpoint (M A B : ℝ × ℝ) : Prop := sorry

-- Define area calculation functions
noncomputable def area_parallelogram (P M Q N : ℝ × ℝ) : ℝ := sorry
noncomputable def area_triangle (X Y Z : ℝ × ℝ) : ℝ := sorry
noncomputable def area_quadrilateral (A B C D : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem quadrilateral_area_relations
  (h_quad : is_convex_quadrilateral A B C D)
  (h_intersect : intersect_at O A D B C)
  (h_M : is_midpoint M A B)
  (h_N : is_midpoint N C D)
  (h_P : is_midpoint P A C)
  (h_Q : is_midpoint Q B D) :
  area_parallelogram P M Q N = |area_triangle A B D - area_triangle A C D| / 2 ∧
  area_triangle O P Q = area_quadrilateral A B C D / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_relations_l1018_101824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_decreasing_l1018_101831

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log (5 - 4*x - x^2) / Real.log (1/3)

-- Define the domain of f(x)
def domain (x : ℝ) : Prop := -5 < x ∧ x < 1

-- Theorem statement
theorem f_strictly_decreasing :
  ∀ x y, domain x ∧ domain y ∧ -5 < x ∧ x < y ∧ y < -2 → f x > f y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_decreasing_l1018_101831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_function_sum_l1018_101855

/-- A rational function with specific properties -/
structure RationalFunction where
  p : ℝ → ℝ
  q : ℝ → ℝ
  horizontal_asymptote : (λ y ↦ ∃ M, ∀ x, |x| > M → |p x / q x - y| < 1) 0
  vertical_asymptote_neg_one : ∀ ε > 0, ∃ δ > 0, ∀ x, |x + 1| < δ → |p x / q x| > 1/ε
  vertical_asymptote_two : ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 2| < δ → |p x / q x| > 1/ε
  hole_neg_two : p (-2) = 0 ∧ q (-2) = 0
  p_three : p 3 = 2
  q_three : q 3 = 4

/-- The main theorem stating the sum of p and q -/
theorem rational_function_sum (f : RationalFunction) : 
  ∀ x, f.p x + f.q x = (1/5) * x^3 + x + 4/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_function_sum_l1018_101855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_CDB_is_six_l1018_101822

/-- The measure of an interior angle of a regular pentagon in degrees -/
noncomputable def regular_pentagon_angle : ℝ := (5 - 2) * 180 / 5

/-- The measure of an interior angle of an equilateral triangle in degrees -/
def equilateral_triangle_angle : ℝ := 60

/-- The configuration of the problem: an equilateral triangle sharing a common side with a regular pentagon -/
structure TrianglePentagonConfig where
  /-- The measure of angle BCD in degrees -/
  angle_BCD : ℝ
  /-- The measure of angle CDB in degrees -/
  angle_CDB : ℝ
  /-- The measure of angle CBD in degrees -/
  angle_CBD : ℝ

/-- The properties of the configuration -/
axiom config_properties (c : TrianglePentagonConfig) :
  c.angle_BCD = regular_pentagon_angle + equilateral_triangle_angle ∧
  c.angle_CDB = c.angle_CBD ∧
  c.angle_CDB + c.angle_CBD + c.angle_BCD = 180

/-- The theorem stating that the measure of angle CDB is 6 degrees -/
theorem angle_CDB_is_six (c : TrianglePentagonConfig) : c.angle_CDB = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_CDB_is_six_l1018_101822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_tangent_line_sum_l1018_101870

-- Define the circles and their properties
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the problem parameters
noncomputable def C₁ : Set (ℝ × ℝ) := sorry
noncomputable def C₂ : Set (ℝ × ℝ) := sorry
def intersection_point : ℝ × ℝ := (7, 4)
def radii_product : ℝ := 50

-- Define the tangent line properties
noncomputable def m : ℝ := sorry
def a : ℕ := sorry
def b : ℕ := sorry
def c : ℕ := sorry

-- State the theorem
theorem circles_tangent_line_sum :
  (∃ (center₁ center₂ : ℝ × ℝ) (r₁ r₂ : ℝ),
    C₁ = Circle center₁ r₁ ∧
    C₂ = Circle center₂ r₂ ∧
    intersection_point ∈ C₁ ∧
    intersection_point ∈ C₂ ∧
    r₁ * r₂ = radii_product ∧
    m > 0 ∧
    m = a * Real.sqrt b / c ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    (∀ (p : ℕ), Nat.Prime p → ¬(p^2 ∣ b)) ∧
    Nat.Coprime a c ∧
    (∀ (x y : ℝ), y = m * x → (x, y) ∉ C₁ ∧ (x, y) ∉ C₂) ∧
    (∀ (x : ℝ), (x, 0) ∉ C₁ ∧ (x, 0) ∉ C₂)) →
  a + b + c = 135 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_tangent_line_sum_l1018_101870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identities_l1018_101868

open Real

theorem trig_identities :
  (∀ α : ℝ, (Real.cos (180 * π / 180 + α) * Real.sin (α + 360 * π / 180)) /
            (Real.sin (-α - 180 * π / 180) * Real.cos (-180 * π / 180 - α)) = 1) ∧
  (∀ α : ℝ, Real.tan α = -3/4 →
    (Real.cos (π/2 + α) * Real.sin (-π - α)) /
    (Real.cos (11*π/2 - α) * Real.sin (11*π/2 + α)) = 3/4) :=
by
  constructor
  · intro α
    sorry -- Proof for the first part
  · intro α h
    sorry -- Proof for the second part

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identities_l1018_101868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_triangle_area_l1018_101883

/-- Given a curve y = a/x (a ≠ 0), the area of the triangle formed by its tangent line
    and the coordinate axes is equal to 2|a|. -/
theorem tangent_line_triangle_area (a : ℝ) (ha : a ≠ 0) :
  ∃ (t : ℝ), t ≠ 0 ∧
  let f : ℝ → ℝ := λ x ↦ a / x
  let f' : ℝ → ℝ := λ x ↦ -a / x^2
  let tangent_line : ℝ → ℝ := λ x ↦ f' t * (x - t) + f t
  let y_intercept : ℝ := tangent_line 0
  let x_intercept : ℝ := (Function.invFun tangent_line) 0
  (1/2) * x_intercept * y_intercept = 2 * |a| :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_triangle_area_l1018_101883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_angle_range_l1018_101897

open Real

theorem ellipse_chord_angle_range (a b : ℝ) (h : a > b) (hb : b > 0) :
  ∀ θ₁ θ₂ : ℝ, 0 ≤ |cos (θ₁ - θ₂)| ∧ |cos (θ₁ - θ₂)| ≤ (a^2 - b^2) / (a^2 + b^2) := by
  sorry

#check ellipse_chord_angle_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_angle_range_l1018_101897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100th_term_bounds_l1018_101873

noncomputable def a : ℕ → ℝ
| 0 => 1
| n + 1 => a n + 1 / a n

theorem a_100th_term_bounds :
  14 < a 99 ∧ a 99 < 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100th_term_bounds_l1018_101873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chameleon_convergence_l1018_101850

/-- Represents the color of a chameleon -/
inductive Color : Type where
  | Red : Color
  | Green : Color
  | Blue : Color
  | Yellow : Color
  | Purple : Color
deriving Repr, DecidableEq

/-- Represents the result of a bite between two chameleons -/
def bite_result : Color → Color → Color := sorry

/-- The total number of chameleons -/
def total_chameleons : Nat := 2023

/-- The initial number of red chameleons -/
def initial_red_chameleons : Nat := 2023

/-- Represents the state of all chameleons on the island -/
def ChameleonState : Type := Color → Nat

/-- Represents a single bite action -/
structure BiteAction where
  biter : Color
  bitten : Color

/-- Represents a sequence of bite actions -/
def BiteSequence : Type := List BiteAction

/-- Applies a bite action to the chameleon state -/
def apply_bite (state : ChameleonState) (action : BiteAction) : ChameleonState := sorry

/-- Applies a sequence of bites to the chameleon state -/
def apply_bite_sequence (state : ChameleonState) (sequence : BiteSequence) : ChameleonState := sorry

/-- Checks if all chameleons in the state have the same color -/
def all_same_color (state : ChameleonState) : Prop := sorry

/-- The main theorem to be proved -/
theorem chameleon_convergence :
  ∃ (final_color : Color) (sequence : BiteSequence),
    let initial_state : ChameleonState := fun c => if c = Color.Red then initial_red_chameleons else 0
    let final_state := apply_bite_sequence initial_state sequence
    all_same_color final_state ∧
    (∀ c, final_state c > 0 → c = final_color) := by
  sorry

#eval Color.Red

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chameleon_convergence_l1018_101850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_city_miles_per_tankful_l1018_101802

/-- Represents the fuel efficiency and travel distance of a car in different conditions -/
structure CarData where
  highway_miles_per_tankful : ℚ
  city_mpg : ℚ
  mpg_difference : ℚ

/-- Calculates the miles per tankful in the city given car data -/
def city_miles_per_tankful (data : CarData) : ℚ :=
  let highway_mpg := data.city_mpg + data.mpg_difference
  let tank_size := data.highway_miles_per_tankful / highway_mpg
  tank_size * data.city_mpg

/-- Theorem stating that given the problem conditions, the car travels 336 miles per tankful in the city -/
theorem car_city_miles_per_tankful :
  let data : CarData := {
    highway_miles_per_tankful := 462,
    city_mpg := 48,
    mpg_difference := 18
  }
  city_miles_per_tankful data = 336 := by
  -- Proof goes here
  sorry

#eval city_miles_per_tankful {
  highway_miles_per_tankful := 462,
  city_mpg := 48,
  mpg_difference := 18
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_city_miles_per_tankful_l1018_101802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_l1018_101847

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.cos x

theorem derivative_f : 
  deriv f = fun x => 2 * x * Real.cos x - x^2 * Real.sin x := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_l1018_101847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_walk_distance_l1018_101808

/-- The total distance an ant walks on a 7x10 grid -/
theorem ant_walk_distance (A B C : ℝ × ℝ) : 
  A.2 - B.2 = 5 →
  C.1 - B.1 = 8 →
  (A.1 - B.1 = 0 ∧ B.1 - C.1 = 0) →
  (A.2 - C.2)^2 + (A.1 - C.1)^2 = 89 →
  |A.2 - B.2| + |C.1 - B.1| + Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) = 13 + Real.sqrt 89 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_walk_distance_l1018_101808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_distinguishable_triangles_l1018_101820

/-- A color represents one of the six available colors for the triangles. -/
inductive Color
| Red | Blue | Green | Yellow | Orange | Purple

/-- A LargeTriangle represents the configuration of colors in the large equilateral triangle. -/
structure LargeTriangle where
  corner1 : Color
  corner2 : Color
  corner3 : Color
  center : Color

/-- Two LargeTriangles are considered equivalent if they can be transformed into each other
    by rotations or reflections. -/
def equivalent : LargeTriangle → LargeTriangle → Prop := sorry

/-- The set of all possible LargeTriangles. -/
def allLargeTriangles : Set LargeTriangle := sorry

/-- We need to define a Setoid for LargeTriangle based on our equivalence relation -/
def largeTriangleSetoid : Setoid LargeTriangle where
  r := equivalent
  iseqv := sorry

/-- The set of distinguishable LargeTriangles is the quotient of allLargeTriangles
    by the equivalence relation. -/
def distinguishableTriangles : Type := Quotient largeTriangleSetoid

/-- We assume that distinguishableTriangles is finite -/
instance : Fintype distinguishableTriangles := sorry

theorem count_distinguishable_triangles :
  Fintype.card distinguishableTriangles = 336 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_distinguishable_triangles_l1018_101820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1018_101859

/-- The eccentricity of a hyperbola with given properties -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let C : ℝ → ℝ → Prop := λ x y => x^2 / a^2 - y^2 / b^2 = 1
  let asymptote : ℝ → ℝ := λ x => (b / a) * x
  let vertex : ℝ × ℝ := (a, 0)
  let distance := abs (b * a - a * 0) / Real.sqrt (b^2 + a^2)
  distance = b / 2 →
  Real.sqrt (a^2 + b^2) / a = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1018_101859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_conversion_l1018_101881

noncomputable section

/-- Converts spherical coordinates to rectangular coordinates -/
def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ,
   ρ * Real.sin φ * Real.sin θ,
   ρ * Real.cos φ)

/-- The given point in spherical coordinates -/
def given_point : ℝ × ℝ × ℝ := (5, Real.pi / 4, Real.pi / 3)

/-- The expected point in rectangular coordinates -/
def expected_point : ℝ × ℝ × ℝ := (5 * Real.sqrt 6 / 4, 5 * Real.sqrt 6 / 4, 2.5)

theorem spherical_to_rectangular_conversion :
  spherical_to_rectangular given_point.1 given_point.2.1 given_point.2.2 = expected_point := by
  sorry

#check spherical_to_rectangular_conversion

end


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_conversion_l1018_101881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ramu_profit_percentage_approx_l1018_101841

/-- Calculates the profit percentage for Ramu's car transaction --/
noncomputable def calculate_profit_percentage (
  initial_car_cost : ℝ)
  (repair_cost : ℝ)
  (shipping_fee_usd : ℝ)
  (initial_conversion_rate : ℝ)
  (import_tax_rate : ℝ)
  (tax_time_conversion_rate : ℝ)
  (selling_price_usd : ℝ)
  (final_conversion_rate : ℝ) : ℝ :=
  let initial_total_cost := initial_car_cost + repair_cost
  let shipping_fee_inr := shipping_fee_usd * initial_conversion_rate
  let pre_tax_total_cost := initial_total_cost + shipping_fee_inr
  let import_tax := pre_tax_total_cost * import_tax_rate
  let total_cost := pre_tax_total_cost + import_tax
  let selling_price_inr := selling_price_usd * final_conversion_rate
  let profit := selling_price_inr - total_cost
  (profit / total_cost) * 100

/-- Theorem stating that Ramu's profit percentage is approximately 0.225% --/
theorem ramu_profit_percentage_approx :
  ∃ ε > 0, |calculate_profit_percentage 42000 10000 250 75 0.1 70 1200 65 - 0.225| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ramu_profit_percentage_approx_l1018_101841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithmic_equation_solution_l1018_101816

def logarithmic_equation (x : ℝ) : Prop :=
  20 * (Real.log (Real.sqrt x) / Real.log (4 * x)) +
  7 * (Real.log (x^3) / Real.log (16 * x)) -
  3 * (Real.log (x^2) / Real.log (x / 2)) = 0

noncomputable def solution_set : Set ℝ := {1, 1 / (4 * Real.rpow 8 (1/5)), 4}

theorem logarithmic_equation_solution :
  ∀ x : ℝ, x > 0 ∧ x ≠ 1/4 ∧ x ≠ 1/16 ∧ x ≠ 2 →
  (logarithmic_equation x ↔ x ∈ solution_set) :=
by
  sorry

#check logarithmic_equation_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithmic_equation_solution_l1018_101816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_option_B_is_correct_l1018_101891

/-- Represents the options for a multiple-choice question -/
inductive MCOption
| A
| B
| C
| D

/-- Data for a multiple-choice question -/
structure QuestionData where
  total_students : ℕ
  points_per_question : ℕ
  difficulty_coefficient : ℝ
  percentage_A : ℝ
  percentage_B : ℝ
  percentage_C : ℝ
  percentage_D : ℝ

/-- Determines if the given option is the correct answer based on the question data -/
def is_correct_answer (data : QuestionData) (option : MCOption) : Prop :=
  let expected_average_score := data.difficulty_coefficient * data.points_per_question
  match option with
  | MCOption.A => data.percentage_A / 100 * data.points_per_question = expected_average_score
  | MCOption.B => data.percentage_B / 100 * data.points_per_question = expected_average_score
  | MCOption.C => data.percentage_C / 100 * data.points_per_question = expected_average_score
  | MCOption.D => data.percentage_D / 100 * data.points_per_question = expected_average_score

/-- The main theorem stating that Option B is the correct answer -/
theorem option_B_is_correct (data : QuestionData)
  (h_total : data.total_students = 11623)
  (h_points : data.points_per_question = 5)
  (h_difficulty : data.difficulty_coefficient = 0.34)
  (h_A : data.percentage_A = 36.21)
  (h_B : data.percentage_B = 33.85)
  (h_C : data.percentage_C = 17.7)
  (h_D : data.percentage_D = 11.96) :
  is_correct_answer data MCOption.B := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_option_B_is_correct_l1018_101891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1018_101898

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 3)

theorem f_properties :
  (∀ x, f x ≤ 2) ∧
  (∃ x, f x = 2) ∧
  (∀ x, f (x + Real.pi) = f x) ∧
  (f (-Real.pi / 6) = 0) ∧
  (∀ k : ℤ, ∀ x ∈ Set.Icc (k * Real.pi - 5 * Real.pi / 12) (k * Real.pi + Real.pi / 12),
    ∀ y ∈ Set.Icc (k * Real.pi - 5 * Real.pi / 12) (k * Real.pi + Real.pi / 12),
    x ≤ y → f x ≤ f y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1018_101898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1018_101860

/-- Triangle ABC with angles A, B, C and sides a, b, c opposite to them respectively -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- The main theorem -/
theorem triangle_properties (t : Triangle) : 
  (2 * Real.cos t.C * (t.a * Real.cos t.B + t.b * Real.cos t.A) = t.c) →
  (t.C = π / 3) ∧
  (t.c = Real.sqrt 7 ∧ 
   (1/2 * t.a * t.b * Real.sin t.C = 3 * Real.sqrt 3 / 2) →
   t.a + t.b + t.c = 5 + Real.sqrt 7) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1018_101860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goods_train_speed_approx_l1018_101818

/-- Calculates the speed of a goods train given the following conditions:
  * A train A is traveling at 50 km/h
  * A goods train B of length 280 m is passing train A in the opposite direction
  * It takes 9 seconds for train B to completely pass train A
-/
noncomputable def goodsTrainSpeed (speedA : ℝ) (lengthB : ℝ) (passingTime : ℝ) : ℝ :=
  let speedAMps := speedA * 1000 / 3600  -- Convert km/h to m/s
  let relativeSpeed := lengthB / passingTime
  let speedBMps := relativeSpeed - speedAMps
  speedBMps * 3600 / 1000  -- Convert m/s back to km/h

/-- Theorem stating that under the given conditions, the speed of the goods train
    is approximately 61.99 km/h -/
theorem goods_train_speed_approx :
  let speedA : ℝ := 50  -- km/h
  let lengthB : ℝ := 280  -- m
  let passingTime : ℝ := 9  -- s
  ∃ ε > 0, |goodsTrainSpeed speedA lengthB passingTime - 61.99| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_goods_train_speed_approx_l1018_101818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_difference_l1018_101885

/-- The product of all positive divisors of n -/
noncomputable def π (n : ℕ+) : ℕ+ := sorry

/-- The function g(n) as defined in the problem -/
noncomputable def g (n : ℕ+) : ℝ := (π n : ℝ) / n

/-- The number of positive divisors of n -/
def d (n : ℕ+) : ℕ := sorry

theorem g_difference : g 180 - g 90 = 180^8 - 90^5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_difference_l1018_101885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_equation_solution_l1018_101807

theorem sine_equation_solution (x : Real) :
  Real.sin x = 1/3 ∧ x ∈ Set.Icc (π/2) (3*π/2) → x = π - Real.arcsin (1/3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_equation_solution_l1018_101807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_history_to_biology_ratio_main_theorem_l1018_101830

/-- Represents the time spent on different subjects --/
structure HomeworkTime where
  biology : ℕ
  history : ℕ
  geography : ℕ

/-- The conditions of Max's homework time --/
def maxHomework (h : ℕ) : HomeworkTime where
  biology := 20
  history := h
  geography := 3 * h

/-- The total time Max spent on homework --/
def totalTime : ℕ := 180

/-- Theorem stating the ratio of history to biology time --/
theorem history_to_biology_ratio (h : ℕ) :
  (maxHomework h).history = 2 * (maxHomework h).biology :=
by
  sorry

/-- Main theorem proving the ratio of history to biology time is 2:1 --/
theorem main_theorem (h : ℕ) :
  (maxHomework h).biology + (maxHomework h).history + (maxHomework h).geography = totalTime →
  (maxHomework h).history = 2 * (maxHomework h).biology :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_history_to_biology_ratio_main_theorem_l1018_101830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_value_g_range_cos_value_l1018_101814

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sin x + a * Real.cos x

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (f a x)^2 - 2

-- Theorem 1
theorem a_value (a : ℝ) : f a (Real.pi / 3) = 0 → a = -Real.sqrt 3 := by sorry

-- Theorem 2
theorem g_range (x : ℝ) : 
  x ∈ Set.Ioo (Real.pi / 4) ((2 * Real.pi) / 3) → 
  g (-Real.sqrt 3) x ∈ Set.Icc (-2) 1 ∧ 
  g (-Real.sqrt 3) x ≠ 1 := by sorry

-- Theorem 3
theorem cos_value (a : ℝ) :
  g (-Real.sqrt 3) (a / 2) = -Real.sqrt 3 / 4 →
  a ∈ Set.Ioo (Real.pi / 6) ((2 * Real.pi) / 3) →
  Real.cos (a + (3 * Real.pi) / 2) = (3 + Real.sqrt 61) / 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_value_g_range_cos_value_l1018_101814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x₀_minus_pi_12_l1018_101805

noncomputable def a (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, Real.sqrt 3 / 2)

noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sin (x - Real.pi / 3), 1)

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

noncomputable def x₀ : ℝ := Real.pi * 5 / 12

theorem cos_2x₀_minus_pi_12 :
  x₀ ∈ Set.Icc (Real.pi * 5 / 12) (2 * Real.pi / 3) →
  f x₀ = 4 / 5 →
  Real.cos (2 * x₀ - Real.pi / 12) = -(7 * Real.sqrt 2) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x₀_minus_pi_12_l1018_101805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1018_101899

open Set

theorem inequality_solution (x : ℝ) : 
  (5 * x^2 + 20 * x - 56) / ((3 * x - 4) * (x + 5)) < 2 ↔ 
  x ∈ Ioo (-5 : ℝ) (-4 : ℝ) ∪ Ioo (4/3 : ℝ) (4 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1018_101899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_theorem_l1018_101853

-- Define the line l: y = k(x + 1)
def line (k : ℝ) (x y : ℝ) : Prop := y = k * (x + 1)

-- Define the circle C: (x - 1)^2 + y^2 = 1
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the condition that the line always intersects the circle
def always_intersects (k : ℝ) : Prop := ∀ x y : ℝ, line k x y → circle_eq x y

-- Define the range of k
def k_range (k : ℝ) : Prop := -Real.sqrt 3 / 3 ≤ k ∧ k ≤ Real.sqrt 3 / 3

-- Define the angle of inclination θ
noncomputable def angle_of_inclination (k : ℝ) : ℝ := Real.arctan (abs k)

-- Define the range of the angle of inclination θ
def theta_range (θ : ℝ) : Prop := 
  (0 ≤ θ ∧ θ ≤ Real.pi / 6) ∨ (5 * Real.pi / 6 ≤ θ ∧ θ < Real.pi)

-- State the theorem
theorem line_circle_intersection_theorem (k : ℝ) :
  always_intersects k → k_range k ∧ theta_range (angle_of_inclination k) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_theorem_l1018_101853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pi_minus_five_floor_l1018_101836

-- Define the greatest integer function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- State the theorem
theorem pi_minus_five_floor : floor (Real.pi - 5) = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pi_minus_five_floor_l1018_101836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_outside_circle_l1018_101828

-- Define the circle and point
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Point where
  coordinates : ℝ × ℝ

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define what it means for a point to be outside a circle
def isOutside (p : Point) (c : Circle) : Prop :=
  distance p.coordinates c.center > c.radius

-- Theorem statement
theorem point_outside_circle (O : Circle) (A : Point) :
  O.radius = 6 → distance A.coordinates O.center = 8 → isOutside A O :=
by
  intros h1 h2
  unfold isOutside
  rw [h1, h2]
  exact lt_of_lt_of_le (by norm_num : (8 : ℝ) > 6) (le_refl 8)


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_outside_circle_l1018_101828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reduced_price_is_fifteen_l1018_101832

/-- Represents the price of oil per kg before and after reduction -/
structure OilPrice where
  original : ℝ
  reduced : ℝ

/-- Represents the quantity of oil that can be bought for a fixed amount -/
structure OilQuantity where
  original : ℝ
  increased : ℝ

def price_reduction_percentage : ℝ := 0.1
def fixed_amount : ℝ := 900
def quantity_increase : ℝ := 6

/-- The reduced price is 10% less than the original price -/
def reduced_price (p : OilPrice) : Prop :=
  p.reduced = p.original * (1 - price_reduction_percentage)

/-- The increased quantity is 6 kgs more than the original quantity -/
def increased_quantity (q : OilQuantity) : Prop :=
  q.increased = q.original + quantity_increase

/-- The fixed amount (900) buys the original quantity at the original price -/
def original_purchase (p : OilPrice) (q : OilQuantity) : Prop :=
  fixed_amount = q.original * p.original

/-- The fixed amount (900) buys the increased quantity at the reduced price -/
def reduced_purchase (p : OilPrice) (q : OilQuantity) : Prop :=
  fixed_amount = q.increased * p.reduced

/-- The main theorem: given the conditions, the reduced price is approximately 15 -/
theorem reduced_price_is_fifteen (p : OilPrice) (q : OilQuantity) :
  reduced_price p →
  increased_quantity q →
  original_purchase p q →
  reduced_purchase p q →
  ∃ ε > 0, |p.reduced - 15| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reduced_price_is_fifteen_l1018_101832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_of_chord_l1018_101834

-- Define the line
def line_eq (x y : ℝ) : Prop := y = -1/2 * x - 2

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 15

-- Define the perpendicular bisector
def perp_bisector_eq (x y : ℝ) : Prop := y = 2*x - 2

-- Theorem statement
theorem perpendicular_bisector_of_chord :
  ∀ A B : ℝ × ℝ,
  line_eq A.1 A.2 ∧ line_eq B.1 B.2 ∧
  circle_eq A.1 A.2 ∧ circle_eq B.1 B.2 ∧
  A ≠ B →
  ∃ C : ℝ × ℝ, perp_bisector_eq C.1 C.2 ∧
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = (C.1 - B.1)^2 + (C.2 - B.2)^2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_of_chord_l1018_101834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_area_proof_l1018_101819

/-- Given two lines in the xy-plane -/
def line1 : ℝ → ℝ → Prop := λ x y ↦ 3*x + 4*y - 2 = 0
def line2 : ℝ → ℝ → Prop := λ x y ↦ 2*x + y + 2 = 0

/-- The perpendicular line to l -/
def perp_line : ℝ → ℝ → Prop := λ x y ↦ x - 2*y - 1 = 0

/-- The intersection point of line1 and line2 -/
def P : ℝ × ℝ := (-2, 2)

/-- The equation of line l -/
def line_l : ℝ → ℝ → Prop := λ x y ↦ 2*x + y + 2 = 0

/-- The area of the triangle formed by line l and the coordinate axes -/
def S : ℝ := 1

theorem line_and_area_proof :
  (∀ x y, line1 x y ∧ line2 x y → (x, y) = P) →
  (∀ x y, line_l x y → perp_line x y) →
  (line_l P.1 P.2) →
  (∀ x y, line_l x y ↔ 2*x + y + 2 = 0) ∧
  S = 1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_area_proof_l1018_101819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_cube_roots_l1018_101815

-- Define the expression
noncomputable def expression : ℝ := (1 + 27) ^ (1/3) * (1 + 27 ^ (1/3)) ^ (1/3)

-- Theorem statement
theorem simplify_cube_roots : expression = 112 ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_cube_roots_l1018_101815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_remainder_implies_b_value_l1018_101865

noncomputable section

/-- The dividend polynomial -/
def dividend (b x : ℝ) : ℝ := 12 * x^3 - 9 * x^2 + b * x + 8

/-- The divisor polynomial -/
def divisor (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2

/-- The remainder polynomial -/
def remainder (b x : ℝ) : ℝ := (b - 8 + 28/3) * x + 10/3

theorem constant_remainder_implies_b_value :
  (∃ (c : ℝ), ∀ (x : ℝ), dividend b x = divisor x * (4 * x + 7/3) + c) →
  b = -4/3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_remainder_implies_b_value_l1018_101865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_negative_23π_over_6_cos_sin_relation_l1018_101871

-- Part 1
noncomputable def f (α : ℝ) : ℝ :=
  (2 * Real.sin (Real.pi + α) * Real.cos (Real.pi - α) - Real.cos (Real.pi + α)) /
  (1 + Real.sin α ^ 2 + Real.cos ((3 * Real.pi) / 2 + α) - Real.sin ((Real.pi / 2) + α) ^ 2)

theorem f_value_at_negative_23π_over_6 :
  1 + 2 * Real.sin (-23 * Real.pi / 6) ≠ 0 →
  f (-23 * Real.pi / 6) = Real.sqrt 3 := by
  sorry

-- Part 2
theorem cos_sin_relation (α : ℝ) :
  Real.cos (Real.pi / 6 - α) = Real.sqrt 3 / 3 →
  Real.cos (5 * Real.pi / 6 + α) - Real.sin (α - Real.pi / 6) ^ 2 = -(Real.sqrt 3 + 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_negative_23π_over_6_cos_sin_relation_l1018_101871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_l1018_101835

/-- A parabola with equation y = 1/4 * x^2 -/
structure Parabola where
  equation : ℝ → ℝ
  eq_def : equation = fun x => (1/4) * x^2

/-- A point on the parabola -/
structure PointOnParabola (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y = p.equation x

/-- The focus of a parabola -/
noncomputable def focus (p : Parabola) : ℝ × ℝ := (0, 1/4)

/-- Distance between two points in 2D space -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem: For a parabola y = 1/4 * x^2, if a point A has an ordinate of 4,
    then the distance from A to the focus is 5 -/
theorem distance_to_focus (p : Parabola) (A : PointOnParabola p)
    (h : A.y = 4) :
    distance (A.x, A.y) (focus p) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_l1018_101835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_half_angle_l1018_101845

theorem sin_half_angle (α : ℝ) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.cos α = (1 + Real.sqrt 5) / 4) : 
  Real.sin (α / 2) = (-1 + Real.sqrt 5) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_half_angle_l1018_101845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_cost_12_hours_is_2_77_l1018_101811

/-- Parking cost structure -/
structure ParkingCost where
  base_cost : ℚ  -- Cost for up to 2 hours
  rate_2_to_5 : ℚ  -- Rate per hour for hours 3 to 5
  rate_5_to_8 : ℚ  -- Rate per hour for hours 6 to 8
  rate_over_8 : ℚ  -- Rate per hour for hours over 8

/-- Calculate total parking cost for given hours -/
def calculate_cost (pc : ParkingCost) (hours : ℕ) : ℚ :=
  if hours ≤ 2 then pc.base_cost
  else if hours ≤ 5 then pc.base_cost + (hours - 2 : ℚ) * pc.rate_2_to_5
  else if hours ≤ 8 then pc.base_cost + 3 * pc.rate_2_to_5 + (hours - 5 : ℚ) * pc.rate_5_to_8
  else pc.base_cost + 3 * pc.rate_2_to_5 + 3 * pc.rate_5_to_8 + (hours - 8 : ℚ) * pc.rate_over_8

/-- Average cost per hour -/
def average_cost_per_hour (pc : ParkingCost) (hours : ℕ) : ℚ :=
  calculate_cost pc hours / hours

/-- Theorem: The average cost per hour for 12 hours of parking is $2.77 (rounded to two decimal places) -/
theorem average_cost_12_hours_is_2_77 (pc : ParkingCost)
    (h1 : pc.base_cost = 10)
    (h2 : pc.rate_2_to_5 = 7/4)
    (h3 : pc.rate_5_to_8 = 2)
    (h4 : pc.rate_over_8 = 3) :
    ∃ (x : ℚ), x ≥ 277/100 ∧ x < 2775/1000 ∧ average_cost_per_hour pc 12 = x := by
  sorry

#eval average_cost_per_hour { base_cost := 10, rate_2_to_5 := 7/4, rate_5_to_8 := 2, rate_over_8 := 3 } 12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_cost_12_hours_is_2_77_l1018_101811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1018_101856

open Set
open Real

noncomputable def f (x : ℝ) := Real.sin (π * x)

theorem f_range : 
  {y | ∃ x ∈ Icc (1/3 : ℝ) (5/6 : ℝ), f x = y} = Icc (1/2 : ℝ) 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1018_101856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_athlete_long_jump_l1018_101829

/-- Represents an athlete's jump distances -/
structure AthleteJumps where
  long_jump : ℝ
  triple_jump : ℝ
  high_jump : ℝ

/-- Calculates the average jump distance for an athlete -/
noncomputable def average_jump (jumps : AthleteJumps) : ℝ :=
  (jumps.long_jump + jumps.triple_jump + jumps.high_jump) / 3

/-- The main theorem stating the second athlete's long jump distance -/
theorem second_athlete_long_jump (x : ℝ) :
  let first_athlete : AthleteJumps := ⟨26, 30, 7⟩
  let second_athlete : AthleteJumps := ⟨x, 34, 8⟩
  let winner_average := 22
  (average_jump first_athlete < winner_average) →
  (average_jump second_athlete = winner_average) →
  x = 24 := by
  sorry

#check second_athlete_long_jump

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_athlete_long_jump_l1018_101829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pen_cost_is_12_cents_l1018_101877

/-- The average cost per pen rounded to the nearest cent -/
def average_cost_per_pen (num_pens : ℕ) (catalog_price shipping_cost : ℚ) (discount_rate : ℚ) : ℕ :=
  let total_cost := catalog_price + shipping_cost
  let discounted_cost := total_cost * (1 - discount_rate)
  let cost_in_cents := (discounted_cost * 100).floor
  (cost_in_cents / num_pens).natAbs

/-- Theorem stating that the average cost per pen is 12 cents under the given conditions -/
theorem pen_cost_is_12_cents :
  average_cost_per_pen 150 15 5.5 (1/10) = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pen_cost_is_12_cents_l1018_101877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_inside_rectangle_outside_circles_l1018_101844

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a circle with center coordinates and radius -/
structure Circle where
  center_x : ℝ
  center_y : ℝ
  radius : ℝ

/-- Calculates the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℝ := r.width * r.height

/-- Calculates the area of a circle -/
noncomputable def circleArea (c : Circle) : ℝ := Real.pi * c.radius^2

/-- Calculates the total area of multiple circles -/
noncomputable def totalCircleArea (circles : List Circle) : ℝ :=
  circles.map circleArea |> List.sum

/-- Represents the problem setup -/
def problemSetup : Rectangle × List Circle := 
  ({ width := 4, height := 6 },
   [{ center_x := 0, center_y := 0, radius := 2 },
    { center_x := 4, center_y := 0, radius := 1.5 },
    { center_x := 4, center_y := 6, radius := 2.5 }])

/-- The main theorem to be proved -/
theorem area_inside_rectangle_outside_circles :
  let (rect, circles) := problemSetup
  let rectArea := rectangleArea rect
  let circlesArea := totalCircleArea circles
  let overlapEstimate := 6 * Real.pi
  abs ((rectArea - (circlesArea - overlapEstimate)) - 3.6) < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_inside_rectangle_outside_circles_l1018_101844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_martha_profit_is_85_l1018_101867

/-- Calculates Martha's profit from baking and selling bread --/
def marthas_profit (total_loaves : ℕ) (cost_per_loaf : ℚ) 
  (morning_price : ℚ) (afternoon_discount : ℚ) (evening_price : ℚ) : ℚ :=
  let morning_sales := total_loaves / 3
  let morning_revenue := morning_sales * morning_price
  let remaining_after_morning := total_loaves - morning_sales
  let afternoon_sales := remaining_after_morning / 2
  let afternoon_price := morning_price * (1 - afternoon_discount)
  let afternoon_revenue := afternoon_sales * afternoon_price
  let evening_sales := remaining_after_morning - afternoon_sales
  let evening_revenue := evening_sales * evening_price
  let total_revenue := morning_revenue + afternoon_revenue + evening_revenue
  let total_cost := total_loaves * cost_per_loaf
  total_revenue - total_cost

/-- Martha's profit is $85 --/
theorem martha_profit_is_85 : 
  marthas_profit 60 1 3 (1/4) 2 = 85 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_martha_profit_is_85_l1018_101867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grocery_distance_is_eight_l1018_101864

/-- Represents the distances and costs of Carla's trip -/
structure CarlaTrip where
  grocery_distance : ℚ
  school_distance : ℚ
  soccer_distance : ℚ
  mpg : ℚ
  gas_price : ℚ
  gas_spent : ℚ

/-- Calculates the total distance of Carla's trip -/
def total_distance (trip : CarlaTrip) : ℚ :=
  trip.grocery_distance + trip.school_distance + trip.soccer_distance + 2 * trip.soccer_distance

/-- Calculates the total distance possible with the gas spent -/
def distance_from_gas (trip : CarlaTrip) : ℚ :=
  (trip.gas_spent / trip.gas_price) * trip.mpg

/-- Theorem stating that the distance to the grocery store is 8 miles -/
theorem grocery_distance_is_eight (trip : CarlaTrip) 
  (h1 : trip.school_distance = 6)
  (h2 : trip.soccer_distance = 12)
  (h3 : trip.mpg = 25)
  (h4 : trip.gas_price = (5/2))
  (h5 : trip.gas_spent = 5)
  : trip.grocery_distance = 8 := by
  sorry

#check grocery_distance_is_eight

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grocery_distance_is_eight_l1018_101864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_changes_in_gene_frequency_lead_to_evolution_l1018_101804

-- Define a population
structure Population where
  species : Type
  location : Type
  individuals : Set species

-- Define gene frequency
def GeneFrequency := Float

-- Define biological evolution
def BiologicalEvolution := Bool

-- Define the relationship between gene frequency changes and biological evolution
axiom gene_frequency_evolution_equivalence :
  ∀ (p : Population) (initial_freq final_freq : GeneFrequency),
    initial_freq ≠ final_freq → BiologicalEvolution

-- Theorem statement
theorem changes_in_gene_frequency_lead_to_evolution
  (p : Population) (initial_freq final_freq : GeneFrequency) :
  initial_freq ≠ final_freq → BiologicalEvolution :=
by
  intro h
  exact gene_frequency_evolution_equivalence p initial_freq final_freq h


end NUMINAMATH_CALUDE_ERRORFEEDBACK_changes_in_gene_frequency_lead_to_evolution_l1018_101804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quartic_equation_roots_l1018_101861

theorem quartic_equation_roots : 
  ∀ x : ℝ, x^4 - 16*x^3 + 91*x^2 - 216*x + 180 = 0 ↔ x = 2 ∨ x = 3 ∨ x = 5 ∨ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quartic_equation_roots_l1018_101861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_difference_is_zero_l1018_101892

/-- The distance to the beach in miles -/
noncomputable def distance_to_beach : ℝ := 15

/-- Maya's travel time in hours -/
noncomputable def maya_time : ℝ := 45 / 60

/-- Naomi's total travel time in hours -/
noncomputable def naomi_time : ℝ := (15 + 15 + 15) / 60

/-- Maya's average speed in miles per hour -/
noncomputable def maya_speed : ℝ := distance_to_beach / maya_time

/-- Naomi's average speed in miles per hour -/
noncomputable def naomi_speed : ℝ := distance_to_beach / naomi_time

/-- Theorem stating that the difference between Naomi's and Maya's average speeds is 0 mph -/
theorem speed_difference_is_zero : naomi_speed - maya_speed = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_difference_is_zero_l1018_101892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_area_difference_l1018_101888

/-- Regular octagon with side length 1 -/
structure RegularOctagon :=
  (side_length : ℝ)
  (is_regular : side_length = 1)

/-- Right-angled isosceles triangle formed by diagonals in the octagon -/
structure DiagonalTriangle :=
  (hypotenuse : ℝ)
  (is_right_angled_isosceles : hypotenuse = 1)

/-- The central square formed by diagonals in the octagon -/
structure CentralSquare :=
  (side_length : ℝ)
  (is_unit_square : side_length = 1)

/-- The area of a right-angled isosceles triangle with hypotenuse 1 -/
noncomputable def area_diagonal_triangle (t : DiagonalTriangle) : ℝ := 1 / 4

/-- The area of the central square -/
noncomputable def area_central_square (s : CentralSquare) : ℝ := 1

/-- Theorem: The difference between the area of the central square and 
    the area of three diagonal triangles in a regular octagon is 1/4 -/
theorem octagon_area_difference 
  (oct : RegularOctagon) 
  (sq : CentralSquare) 
  (t1 t2 t3 : DiagonalTriangle) : 
  area_central_square sq - (area_diagonal_triangle t1 + area_diagonal_triangle t2 + area_diagonal_triangle t3) = 1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_area_difference_l1018_101888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertices_form_semicircles_l1018_101878

noncomputable section

-- Define the circle k
def Circle (O : ℝ × ℝ) (r : ℝ) := {P : ℝ × ℝ | (P.1 - O.1)^2 + (P.2 - O.2)^2 = r^2}

-- Define a point on the circle
def PointOnCircle (k : Set (ℝ × ℝ)) (P : ℝ × ℝ) := P ∈ k

-- Define a chord passing through a point
def Chord (k : Set (ℝ × ℝ)) (P Q : ℝ × ℝ) := 
  PointOnCircle k P ∧ PointOnCircle k Q ∧ P ≠ Q

-- Define an isosceles right triangle
def IsoscelesRightTriangle (P Q R : ℝ × ℝ) :=
  (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = (R.1 - P.1)^2 + (R.2 - P.2)^2 ∧
  (Q.1 - P.1) * (R.1 - P.1) + (Q.2 - P.2) * (R.2 - P.2) = 0

-- Define the set of vertices outside the circle
def VerticesOutsideCircle (k : Set (ℝ × ℝ)) (P : ℝ × ℝ) :=
  {Q : ℝ × ℝ | ∃ R, Chord k P R ∧ IsoscelesRightTriangle P Q R ∧ Q ∉ k}

-- Define the midpoint of two points
def Midpoint (P S : ℝ × ℝ) : ℝ × ℝ := ((P.1 + S.1) / 2, (P.2 + S.2) / 2)

-- Define reflection of a point over another point
def Reflect (P O : ℝ × ℝ) : ℝ × ℝ := (2 * O.1 - P.1, 2 * O.2 - P.2)

-- Define a semicircle
def Semicircle (P M : ℝ × ℝ) := 
  {Q : ℝ × ℝ | (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = (M.1 - P.1)^2 + (M.2 - P.2)^2 ∧
                ((Q.1 - P.1) * (M.2 - P.2) - (Q.2 - P.2) * (M.1 - P.1) ≥ 0)}

-- Theorem statement
theorem vertices_form_semicircles (k : Set (ℝ × ℝ)) (O : ℝ × ℝ) (r : ℝ) (P : ℝ × ℝ) :
  let S := Reflect P O
  let M := Midpoint P S
  let M' := Reflect M P
  VerticesOutsideCircle k P = 
    (Semicircle P M ∪ Semicircle P M') \ {P, M, M'} :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertices_form_semicircles_l1018_101878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_cylinder_base_area_relation_l1018_101848

/-- Represents a cylinder with base area and height -/
structure Cylinder where
  baseArea : ℝ
  height : ℝ

/-- Represents a cone with base area and height -/
structure Cone where
  baseArea : ℝ
  height : ℝ

/-- Volume of a cylinder -/
noncomputable def cylinderVolume (c : Cylinder) : ℝ := c.baseArea * c.height

/-- Volume of a cone -/
noncomputable def coneVolume (c : Cone) : ℝ := (1/3) * c.baseArea * c.height

theorem cone_cylinder_base_area_relation 
  (cyl : Cylinder) (con : Cone) 
  (h_vol_eq : cylinderVolume cyl = coneVolume con)
  (h_height_eq : cyl.height = con.height)
  (h_cyl_base : cyl.baseArea = 36) :
  con.baseArea = 108 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_cylinder_base_area_relation_l1018_101848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_directrix_of_standard_parabola_eq_l1018_101874

/-- The directrix of a parabola y = x^2 --/
noncomputable def directrix_of_standard_parabola : ℝ := -1/4

/-- A parabola with equation y = x^2 --/
def standard_parabola (x y : ℝ) : Prop := y = x^2

theorem directrix_of_standard_parabola_eq :
  ∀ x y : ℝ, standard_parabola x y →
  ∃ d : ℝ, d = directrix_of_standard_parabola ∧
  (x = 0 → y = -d) ∧
  (∀ p q : ℝ, standard_parabola p q → (p - x)^2 + (q - y)^2 = (q + d)^2) :=
by
  sorry

#check directrix_of_standard_parabola_eq

end NUMINAMATH_CALUDE_ERRORFEEDBACK_directrix_of_standard_parabola_eq_l1018_101874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_negative_six_to_seven_l1018_101857

/-- The arithmetic mean of integers from -6 through 7, inclusive, is 0.5 -/
theorem arithmetic_mean_negative_six_to_seven : 
  let s := Finset.range 14
  let f : ℕ → ℚ := λ i => (i : ℚ) - 6
  (s.sum f) / 14 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_negative_six_to_seven_l1018_101857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meal_total_calculation_l1018_101827

/-- Calculates the total amount paid for a meal with given conditions --/
theorem meal_total_calculation (initial_cost : ℚ) (discount_rate : ℚ) 
  (service_tax_rate : ℚ) (vat_rate : ℚ) (tip_min_rate : ℚ) (tip_max_rate : ℚ) :
  initial_cost = 35.50 ∧ 
  discount_rate = 0.10 ∧ 
  service_tax_rate = 0.10 ∧ 
  vat_rate = 0.05 ∧ 
  tip_min_rate = 0.15 ∧ 
  tip_max_rate = 0.25 →
  ∃ (total_paid : ℚ), 
    total_paid = (initial_cost * (1 - discount_rate) * (1 + service_tax_rate + vat_rate)) + 
      ((tip_min_rate + tip_max_rate) / 2 * initial_cost) ∧
    abs (total_paid - 43.8425) < 0.0001 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meal_total_calculation_l1018_101827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_questionnaires_for_survey_l1018_101846

/-- The minimum number of questionnaires to mail given a response rate and required responses. -/
def min_questionnaires_to_mail (response_rate : ℚ) (required_responses : ℕ) : ℕ :=
  Nat.ceil ((required_responses : ℚ) / response_rate)

/-- Theorem stating the minimum number of questionnaires to mail for the given problem. -/
theorem min_questionnaires_for_survey : 
  min_questionnaires_to_mail (70/100) 300 = 429 := by
  sorry

#eval min_questionnaires_to_mail (70/100) 300

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_questionnaires_for_survey_l1018_101846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_inequality_l1018_101893

-- Define a, b, and c as noncomputable
noncomputable def a : ℝ := (3 : ℝ) ^ (1/10)
noncomputable def b : ℝ := Real.log 2 / Real.log 3
noncomputable def c : ℝ := Real.log (1/3) / Real.log 2

-- Theorem statement
theorem abc_inequality : c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_inequality_l1018_101893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_general_term_l1018_101876

variable (p q c : ℝ)
variable (n : ℕ)

def a : ℕ → ℝ
  | 0 => c  -- Add this case for 0
  | 1 => c
  | m + 1 => p * a m + q * m

theorem general_term (hp : p ≠ 0) :
  a p q c n = p^(n-1) * c + (q*(n-1))/(1-p) + (q*(p^n - p))/((1-p)^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_general_term_l1018_101876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_ratio_theorem_l1018_101812

/-- Point type for geometric objects -/
structure Point where

/-- Distance between two points -/
noncomputable def distance (P Q : Point) : ℝ := sorry

/-- Given two chords EF and GH meeting at point Q inside a circle, 
    this theorem proves the ratio of FQ to HQ given specific segment lengths. -/
theorem chord_ratio_theorem (x : ℝ) (hx : x > 0) : 
  ∀ (E F G H Q : Point), 
  (distance E Q = x + 1) → 
  (distance G Q = 2*x) → 
  (distance H Q = 3*x) → 
  (distance F Q * distance E Q = distance G Q * distance H Q) →  -- Power of a Point theorem
  (distance F Q / distance H Q = 2*x / (x + 1)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_ratio_theorem_l1018_101812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l1018_101866

-- Define e as the base of the natural logarithm
noncomputable def e : ℝ := Real.exp 1

-- Define the logarithm functions
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10
noncomputable def ln : ℝ → ℝ := Real.log

-- Define x and y as given in the problem
noncomputable def x : ℝ := lg e
noncomputable def y : ℝ := ln 10

-- State the theorem to be proved
theorem log_inequality : y > 1 ∧ 1 > x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l1018_101866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extra_bananas_distribution_l1018_101801

theorem extra_bananas_distribution (total_children absent_children : ℕ) 
  (total_bananas : ℝ) (h1 : total_children > 0) (h2 : absent_children < total_children) :
  let present_children := total_children - absent_children
  let bananas_per_child_all := total_bananas / total_children
  let bananas_per_child_present := total_bananas / present_children
  bananas_per_child_present - bananas_per_child_all = bananas_per_child_all :=
by
  -- Introduce the local definitions
  let present_children := total_children - absent_children
  let bananas_per_child_all := total_bananas / total_children
  let bananas_per_child_present := total_bananas / present_children

  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extra_bananas_distribution_l1018_101801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_triangle_area_l1018_101810

-- Define the parabola C
def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2*p*y ∧ p > 0

-- Define the circle M
def circleM (x y : ℝ) : Prop := x^2 + (y+3)^2 = 1

-- Define the focus of the parabola
noncomputable def focus (p : ℝ) : ℝ × ℝ := (0, p/2)

-- Define the maximum distance condition
def max_distance_condition (p : ℝ) : Prop :=
  ∃ (x y : ℝ), circleM x y ∧ 
  Real.sqrt ((x - (focus p).1)^2 + (y - (focus p).2)^2) ≤ 5 ∧
  ∀ (x' y' : ℝ), circleM x' y' → 
    Real.sqrt ((x' - (focus p).1)^2 + (y' - (focus p).2)^2) ≤ 
    Real.sqrt ((x - (focus p).1)^2 + (y - (focus p).2)^2)

-- Define the tangent points A and B
def tangent_points (p : ℝ) (x y : ℝ) : Prop :=
  circleM x y ∧ ∃ (x1 y1 x2 y2 : ℝ),
    parabola p x1 y1 ∧ parabola p x2 y2 ∧
    (y - y1) = (x - x1) * x1 / (2*p) ∧
    (y - y2) = (x - x2) * x2 / (2*p)

-- Theorem statement
theorem parabola_and_triangle_area :
  ∀ p : ℝ, parabola p 2 1 ∧ max_distance_condition p →
  (∀ x y : ℝ, parabola p x y ↔ x^2 = 4*y) ∧
  (∃ (max_area : ℝ), max_area = 32 ∧
    ∀ x y : ℝ, tangent_points p x y →
    ∃ (area : ℝ), area ≤ max_area ∧
    ∀ (area' : ℝ), (∃ x' y' : ℝ, tangent_points p x' y' ∧ area' ≤ area)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_triangle_area_l1018_101810
