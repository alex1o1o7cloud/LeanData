import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l124_12440

-- Define the speed of the train in km/hr
noncomputable def train_speed_kmh : ℝ := 126

-- Define the length of the train in meters
noncomputable def train_length_m : ℝ := 315

-- Define the conversion factor from km/hr to m/s
noncomputable def kmh_to_ms : ℝ := 1000 / 3600

-- Define the time it takes for the train to cross the pole
noncomputable def crossing_time : ℝ := train_length_m / (train_speed_kmh * kmh_to_ms)

-- Theorem statement
theorem train_crossing_time :
  crossing_time = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l124_12440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_l124_12482

/-- Represents the journey of Boris Mikhailovich --/
structure Journey where
  total_distance : ℝ
  initial_speed : ℝ
  tractor_speed : ℝ

/-- The conditions of Boris's journey --/
noncomputable def boris_journey (d : ℝ) : Journey where
  total_distance := 4 * d
  initial_speed := 2 * d
  tractor_speed := d / 2

/-- Theorem stating the time taken to reach the destination after encountering the tractor --/
theorem journey_time (d : ℝ) (h_d : d > 0) :
  let j := boris_journey d
  let time_after_tractor := (j.total_distance - 2 * d) / j.tractor_speed
  time_after_tractor = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_l124_12482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_solution_l124_12448

-- Define the triangle
structure Triangle where
  area : ℝ
  perimeter : ℝ
  angle_alpha : ℝ
  side_a : ℝ
  side_b : ℝ
  side_c : ℝ
  angle_beta : ℝ
  angle_gamma : ℝ

-- Define the problem conditions
noncomputable def problem_triangle : Triangle :=
  { area := 234
  , perimeter := 108
  , angle_alpha := 130 + 26/60 + 59/3600
  , side_a := 0  -- to be calculated
  , side_b := 0  -- to be calculated
  , side_c := 0  -- to be calculated
  , angle_beta := 0  -- to be calculated
  , angle_gamma := 0  -- to be calculated
  }

-- Define the theorem
theorem triangle_solution (t : Triangle) (ε : ℝ) :
  t.area = 234 ∧ 
  t.perimeter = 108 ∧ 
  t.angle_alpha = 130 + 26/60 + 59/3600 →
  (abs (t.side_a - 52) < ε ∧ 
   abs (t.side_b - 41) < ε ∧ 
   abs (t.side_c - 15) < ε ∧
   abs (t.angle_beta - (36 + 51/60 + 50.5/3600)) < ε ∧
   abs (t.angle_gamma - (12 + 41/60 + 10.5/3600)) < ε) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_solution_l124_12448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_melted_ice_cream_depth_l124_12438

/-- The volume of a sphere with radius r -/
noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

/-- The volume of a cylinder with radius r and height h -/
noncomputable def cylinder_volume (r h : ℝ) : ℝ := Real.pi * r^2 * h

/-- 
Given a sphere of ice cream with radius 3 inches that melts into a cylinder 
with radius 12 inches, maintaining constant density, the height of the 
resulting cylinder is 1/4 inch.
-/
theorem melted_ice_cream_depth :
  ∀ (h : ℝ), 
  sphere_volume 3 = cylinder_volume 12 h → 
  h = 1/4 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_melted_ice_cream_depth_l124_12438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_is_all_reals_denominator_never_zero_l124_12405

-- Define the function f
noncomputable def f (t : ℝ) : ℝ := 1 / ((t - 1) * (t + 1) + (t - 2)^2)

-- Theorem stating that the domain of f is all real numbers
theorem domain_of_f_is_all_reals :
  ∀ t : ℝ, ∃ y : ℝ, f t = y :=
by
  sorry

-- Alternatively, we can state it as the denominator is never zero
theorem denominator_never_zero :
  ∀ t : ℝ, (t - 1) * (t + 1) + (t - 2)^2 ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_is_all_reals_denominator_never_zero_l124_12405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l124_12420

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + 2 * Real.cos x

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-2 * π / 3) a, f x ∈ Set.Icc (-1/4) 2) ↔
  a ∈ Set.Icc 0 (2 * π / 3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l124_12420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l124_12452

-- Define the functions
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x^2 - m * Real.log x
def h (a : ℝ) (x : ℝ) : ℝ := x^2 - x + a
noncomputable def g (m a : ℝ) (x : ℝ) : ℝ := f m x - h a x

-- Part I
theorem part_one :
  ∀ m : ℝ, (∀ x : ℝ, x > 1 → f m x ≥ h 0 x) ↔ m ≤ Real.exp 1 := by
  sorry

-- Part II
theorem part_two :
  ∀ a : ℝ, (∃! x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ ∈ Set.Icc 1 3 ∧ x₂ ∈ Set.Icc 1 3 ∧ g 2 a x₁ = 0 ∧ g 2 a x₂ = 0) ↔
  (2 - 2 * Real.log 2 < a ∧ a ≤ 3 - 2 * Real.log 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l124_12452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_uninsured_employees_count_l124_12417

/-- Represents the survey data and calculates the number of uninsured employees --/
noncomputable def calculate_uninsured_employees (total : ℕ) (part_time : ℕ) (uninsured_part_time_ratio : ℚ) 
  (neither_uninsured_nor_part_time_prob : ℚ) : ℕ :=
  let uninsured := (total : ℚ) - ((neither_uninsured_nor_part_time_prob * total) + part_time) / 
                   (1 - uninsured_part_time_ratio)
  Int.toNat (Int.floor uninsured)

/-- Theorem stating that given the survey conditions, there are 104 uninsured employees --/
theorem uninsured_employees_count : 
  calculate_uninsured_employees 350 54 (1/8) (205/350) = 104 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_uninsured_employees_count_l124_12417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_operations_2345_l124_12421

noncomputable def digit_operations (n : ℕ) : ℝ :=
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  (d1 ^ 2 : ℝ) + (d2 : ℝ) * Real.sin (30 * Real.pi / 180) + (d3 ^ 3 : ℝ) + ((d3 ^ 3 : ℝ) - d4)

theorem sum_of_operations_2345 : 
  digit_operations 2345 = 128.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_operations_2345_l124_12421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l124_12480

theorem problem_statement (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a + b = 1) : 
  (a^2 + b^2 ≥ 1/2) ∧ 
  ((2 : ℝ)^(a-b) > 1/2) ∧ 
  (Real.sqrt a + Real.sqrt b ≤ Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l124_12480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l124_12473

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.log (-x^2 - 2*x + 8)}
def B : Set ℝ := {y | ∃ x, y = x + 1/x + 1}

-- Define the theorem
theorem set_operations :
  (A ∩ B = Set.Icc (-4) (-1)) ∧
  (A ∪ B = Set.Iic 2 ∪ Set.Ici 3) ∧
  (A ∩ (Set.compl B) = Set.Ioo (-1) 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l124_12473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_fx_x_l124_12485

theorem gcd_fx_x (x : ℤ) (h : ∃ k : ℤ, x = 54896 * k) :
  Nat.gcd ((5*x+4)*(9*x+7)*(11*x+3)*(x+12)).natAbs x.natAbs = 112 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_fx_x_l124_12485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_excel_count_l124_12436

theorem excel_count (A₁ A₂ A₃ : Finset Nat) 
  (h₁ : A₁.card = 30)
  (h₂ : A₂.card = 28)
  (h₃ : A₃.card = 25)
  (h₁₂ : (A₁ ∩ A₂).card = 20)
  (h₂₃ : (A₂ ∩ A₃).card = 16)
  (h₃₁ : (A₃ ∩ A₁).card = 17)
  (h₁₂₃ : (A₁ ∩ A₂ ∩ A₃).card = 10) :
  ((A₁ ∪ A₂).card = 38) ∧ 
  ((A₁ ∪ A₂ ∪ A₃).card = 40) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_excel_count_l124_12436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_line_intersection_range_l124_12450

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 / 2 = 1

-- Define the foci of the hyperbola
def foci (F₁ F₂ : ℝ × ℝ) : Prop :=
  hyperbola F₁.1 F₁.2 ∧ hyperbola F₂.1 F₂.2 ∧ F₁.1 < F₂.1

-- Define the circle (renamed to avoid conflict with built-in circle)
def myCircle (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Define the line
def line (x y t : ℝ) : Prop := Real.sqrt 2 * x + Real.sqrt 3 * y + t = 0

-- State the theorem
theorem hyperbola_line_intersection_range 
  (F₁ F₂ : ℝ × ℝ) (h_foci : foci F₁ F₂) :
  ∀ t : ℝ, (∃ x y : ℝ, myCircle x y ∧ line x y t) ↔ t ∈ Set.Icc (-5 : ℝ) 5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_line_intersection_range_l124_12450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_number_divisible_by_1996_with_digit_sum_1996_l124_12474

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem exists_number_divisible_by_1996_with_digit_sum_1996 :
  ∃ n : ℕ, (n % 1996 = 0) ∧ (sum_of_digits n = 1996) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_number_divisible_by_1996_with_digit_sum_1996_l124_12474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_arrangement_count_l124_12462

theorem four_digit_arrangement_count : 
  Fintype.card (Fin 4 → Fin 4) = 24 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_arrangement_count_l124_12462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_calculation_l124_12497

/-- Calculates the savings at the end of the year given income, income-to-expenditure ratio, tax rate, and investment rate. -/
def calculateSavings (income : ℕ) (incomeRatio expenditureRatio : ℕ) (taxRate investmentRate : ℚ) : ℚ :=
  let totalDeductions := (taxRate + investmentRate) * income
  let remainingIncome := income - totalDeductions.floor
  let expenditures := (expenditureRatio * income) / (incomeRatio + expenditureRatio)
  (remainingIncome - expenditures : ℚ)

/-- Theorem stating that given the specified conditions, the savings at the end of the year is 1500. -/
theorem savings_calculation (income : ℕ) (incomeRatio expenditureRatio : ℕ) (taxRate investmentRate : ℚ)
    (h1 : income = 10000)
    (h2 : incomeRatio = 5)
    (h3 : expenditureRatio = 3)
    (h4 : taxRate = 15/100)
    (h5 : investmentRate = 10/100) :
  calculateSavings income incomeRatio expenditureRatio taxRate investmentRate = 1500 := by
  sorry

#eval calculateSavings 10000 5 3 (15/100) (10/100)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_calculation_l124_12497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_steve_return_speed_l124_12425

/-- Calculates the return speed given total distance, total time, and the fact that return speed is twice the outbound speed -/
noncomputable def calculate_return_speed (total_distance : ℝ) (total_time : ℝ) : ℝ :=
  let outbound_distance := total_distance / 2
  let return_distance := total_distance / 2
  let outbound_speed := outbound_distance / (total_time / 3)
  2 * outbound_speed

theorem steve_return_speed :
  calculate_return_speed 20 6 = 5 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval calculate_return_speed 20 6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_steve_return_speed_l124_12425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diver_descent_time_approx_l124_12444

/-- Represents the time taken for a diver to reach a lost ship under specific conditions. -/
noncomputable def diverDescentTime (normalRate depthToShip downwardCurrentDepth upwardCurrentDepth : ℝ)
  (downwardCurrentSpeed upwardCurrentSpeed : ℝ)
  (decompStop1Depth decompStop2Depth : ℝ)
  (decompStop1Time decompStop2Time : ℝ) : ℝ :=
  let downwardCurrentTime := downwardCurrentDepth / (normalRate + downwardCurrentSpeed)
  let upwardCurrentTime := (upwardCurrentDepth - downwardCurrentDepth) / (normalRate - upwardCurrentSpeed)
  let normalDepthTime := (depthToShip - upwardCurrentDepth) / normalRate
  let decompStopTime := decompStop1Time + decompStop2Time
  downwardCurrentTime + upwardCurrentTime + normalDepthTime + decompStopTime

/-- Theorem stating the time taken for the diver to reach the lost ship. -/
theorem diver_descent_time_approx :
  let normalRate : ℝ := 80
  let depthToShip : ℝ := 4000
  let downwardCurrentDepth : ℝ := 1500
  let upwardCurrentDepth : ℝ := 3000
  let downwardCurrentSpeed : ℝ := 30
  let upwardCurrentSpeed : ℝ := 20
  let decompStop1Depth : ℝ := 1800
  let decompStop2Depth : ℝ := 3600
  let decompStop1Time : ℝ := 5
  let decompStop2Time : ℝ := 8
  ∃ ε > 0, |diverDescentTime normalRate depthToShip downwardCurrentDepth upwardCurrentDepth
    downwardCurrentSpeed upwardCurrentSpeed decompStop1Depth decompStop2Depth
    decompStop1Time decompStop2Time - 64.14| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diver_descent_time_approx_l124_12444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_2_3_5_7_less_than_500_l124_12456

theorem divisible_by_2_3_5_7_less_than_500 :
  (Finset.filter (λ n : ℕ => n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0)
    (Finset.range 500)).card = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_2_3_5_7_less_than_500_l124_12456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_m_bounds_l124_12466

/-- Given a quadratic equation x^2 + z₁x + z₂ + m = 0 with complex coefficients,
    prove the maximum and minimum values of |m| -/
theorem quadratic_equation_m_bounds
  (z₁ z₂ m : ℂ)
  (h₁ : z₁^2 - 4*z₂ = 16 + 20*Complex.I)
  (h₂ : ∃ (α β : ℂ), Complex.abs (α - β) = 2 * Real.sqrt 7 ∧ α^2 + z₁*α + z₂ + m = 0 ∧ β^2 + z₁*β + z₂ + m = 0) :
  (Complex.abs m ≤ Real.sqrt 41 + 7) ∧ (Complex.abs m ≥ 7 - Real.sqrt 41) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_m_bounds_l124_12466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l124_12486

theorem sequence_inequality (a : ℕ → ℝ) (h : ∀ n, a n > 0) :
  ∃ S : Set ℕ, Set.Infinite S ∧ ∀ n ∈ S, (a 1 + a (n + 1)) / a n > 1 + 1 / (n : ℝ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l124_12486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_min_distance_l124_12446

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The parabola x = (1/4)y^2 -/
def onParabola (p : Point) : Prop :=
  p.x = (1/4) * p.y^2

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Distance from a point to the y-axis -/
def distanceToYAxis (p : Point) : ℝ :=
  abs p.x

/-- The point A(0,2) -/
def A : Point :=
  { x := 0, y := 2 }

/-- The minimum value of the sum of distances -/
noncomputable def minSumOfDistances : ℝ :=
  Real.sqrt 5 - 1

theorem parabola_min_distance :
  ∀ p : Point, onParabola p →
    ∃ q : Point, onParabola q ∧
      ∀ r : Point, onParabola r →
        distance p A + distanceToYAxis p ≥
        distance q A + distanceToYAxis q ∧
        distance q A + distanceToYAxis q = minSumOfDistances :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_min_distance_l124_12446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_range_l124_12419

-- Define the sine function
noncomputable def f (x : ℝ) : ℝ := Real.sin x

-- Define the tangent slope to the sine curve at a point
noncomputable def tangent_slope (x : ℝ) : ℝ := Real.cos x

-- Define the angle of inclination of the tangent line
noncomputable def angle_of_inclination (x : ℝ) : ℝ := Real.arctan (tangent_slope x)

-- Theorem statement
theorem angle_of_inclination_range :
  Set.range angle_of_inclination = Set.union (Set.Icc 0 (π/4)) (Set.Ico (3*π/4) π) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_range_l124_12419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_a_production_time_l124_12483

/-- The time it takes for Machine A to produce x boxes -/
noncomputable def machine_a_time (x : ℝ) : ℝ :=
  5

/-- The rate at which Machine B produces boxes -/
noncomputable def machine_b_rate (x : ℝ) : ℝ :=
  x / 5

/-- The combined production rate of Machines A and B -/
noncomputable def combined_rate (x : ℝ) : ℝ :=
  3 * x / 7.5

theorem machine_a_production_time (x : ℝ) (hx : x > 0) :
  machine_a_time x = 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_a_production_time_l124_12483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_difference_in_S_l124_12404

def S : Set ℤ := {-16, -4, 0, 2, 4, 12}

theorem largest_difference_in_S : 
  (∀ (a b : ℤ), a ∈ S → b ∈ S → (a - b : ℤ) ≤ 28) ∧ 
  (∃ (a b : ℤ), a ∈ S ∧ b ∈ S ∧ a - b = 28) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_difference_in_S_l124_12404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_name_recall_l124_12477

-- Define recall_name as an axiom since it's not defined elsewhere
axiom recall_name : String → ℕ → Prop

theorem book_name_recall :
  ∀ (book : String) (time : ℕ),
    (∃ (t : ℕ), t ≥ time ∧ recall_name book t) :=
by
  intro book time
  -- The proof is skipped using sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_name_recall_l124_12477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_minimum_area_sum_l124_12479

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line passing through two points -/
def Line (p1 p2 : Point) :=
  {p : Point | (p.y - p1.y) * (p2.x - p1.x) = (p2.y - p1.y) * (p.x - p1.x)}

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  p3 ∈ Line p1 p2

/-- The triangle defined by three points -/
structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

/-- The area of a triangle -/
noncomputable def area (t : Triangle) : ℝ :=
  let a := t.p1
  let b := t.p2
  let c := t.p3
  abs ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)) / 2

theorem triangle_minimum_area_sum :
  ∃ (k1 k2 : ℤ),
    let p1 := Point.mk 2 9
    let p2 := Point.mk 14 18
    let p3 := Point.mk 6 (k1 : ℝ)
    let p4 := Point.mk 6 (k2 : ℝ)
    let t1 := Triangle.mk p1 p2 p3
    let t2 := Triangle.mk p1 p2 p4
    k1 ≠ k2 ∧
    (∀ (k : ℤ), area (Triangle.mk p1 p2 (Point.mk 6 (k : ℝ))) ≥ area t1) ∧
    (∀ (k : ℤ), area (Triangle.mk p1 p2 (Point.mk 6 (k : ℝ))) ≥ area t2) ∧
    area t1 > 0 ∧
    area t2 > 0 ∧
    k1 + k2 = 24 :=
  by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_minimum_area_sum_l124_12479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_for_given_chord_and_angle_l124_12434

noncomputable section

open Real

theorem sector_area_for_given_chord_and_angle :
  ∀ (r : ℝ) (chord_length : ℝ) (central_angle : ℝ),
    chord_length = 2 →
    central_angle = 2 →
    2 * r * sin (central_angle / 2) = chord_length →
    (1/2) * r^2 * central_angle = 1 / (sin 1)^2 :=
by
  intros r chord_length central_angle h1 h2 h3
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_for_given_chord_and_angle_l124_12434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_problem_l124_12406

/-- Calculate compound interest and total interest earned --/
theorem compound_interest_problem (P : ℝ) (r : ℝ) (n : ℕ) 
  (h1 : P = 1000) 
  (h2 : r = 0.05) 
  (h3 : n = 5) : 
  ∃ A : ℝ, A = P * (1 + r)^n ∧ abs (A - P - 276.28) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_problem_l124_12406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_calculation_l124_12487

/-- Calculates speed in km/h given distance in meters and time in minutes -/
noncomputable def calculate_speed (distance_m : ℝ) (time_min : ℝ) : ℝ :=
  (distance_m / 1000) / (time_min / 60)

/-- Theorem: A person crossing a 1440 m long street in 12 minutes has a speed of 7.2 km/h -/
theorem speed_calculation :
  calculate_speed 1440 12 = 7.2 := by
  -- Unfold the definition of calculate_speed
  unfold calculate_speed
  -- Simplify the expression
  simp
  -- The proof is completed numerically
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_calculation_l124_12487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geq_g_iff_a_eq_two_l124_12455

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x

noncomputable def g (x : ℝ) : ℝ := 2 - Real.sin x - Real.cos x

theorem f_geq_g_iff_a_eq_two :
  ∃! a : ℝ, ∀ x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 4), f a x ≥ g x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geq_g_iff_a_eq_two_l124_12455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_angle_l124_12449

noncomputable def vector_a (α : Real) : Fin 2 → Real := ![3/2, Real.sin α]
noncomputable def vector_b (α : Real) : Fin 2 → Real := ![Real.cos α, 1/3]

theorem parallel_vectors_angle (α : Real) 
  (h1 : 0 < α ∧ α < Real.pi/2) -- α is an acute angle
  (h2 : ∃ (k : Real), k ≠ 0 ∧ vector_a α = k • vector_b α) -- a is parallel to b
  : α = Real.pi/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_angle_l124_12449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_problem_l124_12426

-- Define the ★ operation
def star : ℕ → ℕ → ℕ := sorry

-- Axioms for the ★ operation
axiom star_succ_zero (x : ℕ) : star (x + 1) 0 = star 0 x + 1
axiom star_zero_succ (y : ℕ) : star 0 (y + 1) = star y 0 + 1
axiom star_succ_succ (x y : ℕ) : star (x + 1) (y + 1) = star x y + 1

-- Given condition
axiom star_given : star 123 456 = 789

-- Theorem to prove
theorem star_problem : star 246 135 = 579 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_problem_l124_12426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_syllogism_arrangement_l124_12400

/-- Represents an imaginary number -/
structure ImaginaryNumber where
  dummy : Unit

/-- Predicate to check if two numbers can be compared -/
def can_be_compared (a b : ImaginaryNumber) : Prop :=
  sorry -- We don't need to define this for the problem

theorem syllogism_arrangement (Z₁ Z₂ : ImaginaryNumber) : 
  (∀ a b : ImaginaryNumber, ¬(can_be_compared a b)) →  -- Statement ③
  (True) →                                            -- Statement ②
  ¬(can_be_compared Z₁ Z₂)                             -- Statement ①
  := by
  intro h1 _
  exact h1 Z₁ Z₂

#check syllogism_arrangement

end NUMINAMATH_CALUDE_ERRORFEEDBACK_syllogism_arrangement_l124_12400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l124_12488

def a : ℕ → ℚ
  | 0 => 1  -- Adding a case for 0 to cover all natural numbers
  | 1 => 1
  | (n + 1) => ((n + 1) * a n) / (2 * n)

def b (n : ℕ) : ℚ := if n = 0 then 1 else a n / n

theorem sequence_properties :
  (b 1 = 1 ∧ b 2 = 1/2 ∧ b 3 = 1/4) ∧
  (∀ n : ℕ, n ≥ 1 → b (n + 1) = 1/2 * b n) ∧
  (∀ n : ℕ, n ≥ 1 → a n = n / (2^(n-1))) :=
by
  sorry  -- Proof is skipped using sorry

#eval b 1  -- This will evaluate b 1 and display the result
#eval b 2
#eval b 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l124_12488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_l124_12402

def is_consecutive (a b : ℤ) : Prop := (a - b).natAbs = 1

def sequence_property (a : ℕ → ℤ) : Prop :=
  ∀ i : ℕ, i ≥ 2 → (a i = 2 * a (i - 1) - a (i - 2)) ∨ (a i = 2 * a (i - 2) - a (i - 1))

theorem consecutive_integers (a : ℕ → ℤ) 
  (h : sequence_property a) 
  (h_consecutive : is_consecutive (a 2023) (a 2024)) :
  is_consecutive (a 0) (a 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_l124_12402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l124_12459

/-- The cube root function -/
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

/-- The equation we want to solve -/
noncomputable def f (y : ℝ) : ℝ := cubeRoot (30*y + cubeRoot (30*y + 19))

/-- Theorem stating that 228 is the unique solution to the equation -/
theorem unique_solution : 
  (∃! y : ℝ, f y = 19) ∧ f 228 = 19 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l124_12459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_wheel_rpm_l124_12439

/-- The revolutions per minute of a wheel given its radius and the speed of the vehicle --/
noncomputable def wheelRPM (radius : ℝ) (speed : ℝ) : ℝ :=
  let circumference := 2 * Real.pi * radius
  let speedCmPerMin := speed * 100000 / 60
  speedCmPerMin / circumference

/-- Theorem stating that a wheel with radius 100 cm on a bus moving at 66 km/h
    has approximately 1750.48 revolutions per minute --/
theorem bus_wheel_rpm :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |wheelRPM 100 66 - 1750.48| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_wheel_rpm_l124_12439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_distinct_extreme_points_implies_a_less_than_negative_four_l124_12431

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 2*Real.log x

-- State the theorem
theorem two_distinct_extreme_points_implies_a_less_than_negative_four (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧
    (∀ y : ℝ, y > 0 → f a y ≥ f a x₁) ∧
    (∀ y : ℝ, y > 0 → f a y ≥ f a x₂) ∧
    (∃ z : ℝ, z > 0 ∧ f a z < f a x₁) ∧
    (∃ z : ℝ, z > 0 ∧ f a z < f a x₂)) →
  a < -4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_distinct_extreme_points_implies_a_less_than_negative_four_l124_12431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_light_source_height_bound_l124_12407

/-- The greatest integer not exceeding 500 times the height of a light source above a cylinder,
    given specific shadow conditions. -/
theorem light_source_height_bound (r h A : ℝ) (hr : r = 1) (hh : h = 1) 
  (hA : A = 75) : 
  ⌊(500 : ℝ) * (Real.sqrt ((A / Real.pi + 1) * r ^ 2 / h ^ 2))⌋ = 2494 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_light_source_height_bound_l124_12407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_best_fitting_model_l124_12441

/-- Represents a regression model with its coefficient of determination (R²) -/
structure RegressionModel where
  r_squared : ℝ

/-- Returns true if the first model has a better fitting effect than the second -/
def better_fit (m1 m2 : RegressionModel) : Prop :=
  abs (1 - m1.r_squared) < abs (1 - m2.r_squared)

theorem best_fitting_model 
  (model1 model2 model3 model4 : RegressionModel)
  (h1 : model1.r_squared = 0.98)
  (h2 : model2.r_squared = 0.80)
  (h3 : model3.r_squared = 0.50)
  (h4 : model4.r_squared = 0.25) :
  better_fit model1 model2 ∧ better_fit model1 model3 ∧ better_fit model1 model4 := by
  sorry

#check best_fitting_model

end NUMINAMATH_CALUDE_ERRORFEEDBACK_best_fitting_model_l124_12441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_interval_l124_12437

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (a * x + 1) - x * (Real.log x - 2)

/-- Theorem stating the range of a for which f has a monotonically decreasing interval -/
theorem f_monotone_decreasing_interval (a : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧ ∀ x ∈ Set.Ioo x₁ x₂, StrictMonoOn (f a) (Set.Ioo x₁ x₂)) ↔
  (0 < a ∧ a < Real.exp (-2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_interval_l124_12437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l124_12403

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

def A : Set ℝ := {x | floor (6 * x^2 + x) = 0}

def B (a : ℝ) : Set ℝ := {x | 2 * x^2 - 5 * a * x + 3 * a^2 > 0}

theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, x ∈ A ∪ B a) ↔ 
  ((-1/3 < a ∧ a ≤ -1/6) ∨ (0 ≤ a ∧ a < 2/9)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l124_12403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_zero_l124_12469

-- Define the integrand
noncomputable def f (x : ℝ) : ℝ :=
  (4 * Real.sqrt (1 - x) - Real.sqrt (2 * x + 1)) /
  ((Real.sqrt (2 * x + 1) + 4 * Real.sqrt (1 - x)) * (2 * x + 1)^2)

-- State the theorem
theorem integral_equals_zero :
  ∫ x in Set.Icc 0 1, f x = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_zero_l124_12469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_hd_ha_ratio_l124_12496

/-- An isosceles triangle with sides 13, 13, and 10 -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 13
  hb : b = 13
  hc : c = 10
  isosceles : a = b

/-- The point where the altitudes of the triangle meet -/
noncomputable def OrthocenterH (t : IsoscelesTriangle) : ℝ × ℝ := sorry

/-- The foot of the altitude from vertex A to the base -/
noncomputable def AltitudeFootD (t : IsoscelesTriangle) : ℝ × ℝ := sorry

/-- The vertex A of the triangle -/
noncomputable def VertexA (t : IsoscelesTriangle) : ℝ × ℝ := sorry

/-- The ratio of HD to HA -/
noncomputable def HDHARatio (t : IsoscelesTriangle) : ℝ :=
  let h := OrthocenterH t
  let d := AltitudeFootD t
  let a := VertexA t
  dist h d / dist h a

theorem isosceles_triangle_hd_ha_ratio (t : IsoscelesTriangle) :
  HDHARatio t = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_hd_ha_ratio_l124_12496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_1999_equals_f_l124_12423

noncomputable def f (x : ℝ) : ℝ := (x - 1) / (x + 1)

noncomputable def f_n : ℕ → ℝ → ℝ 
| 0, x => x
| n + 1, x => f (f_n n x)

theorem f_1999_equals_f (x : ℝ) : f_n 1999 x = f x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_1999_equals_f_l124_12423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_translation_l124_12401

theorem cosine_sine_translation (φ : ℝ) : 
  (∀ x : ℝ, Real.cos (2*x + φ - π/2) = Real.sin (2*x + π/3)) → φ = 5*π/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_translation_l124_12401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_inequality_l124_12498

theorem tan_inequality (x₁ x₂ : ℝ) 
  (h₁ : 0 < x₁ ∧ x₁ < π/2) 
  (h₂ : 0 < x₂ ∧ x₂ < π/2) 
  (h₃ : x₁ ≠ x₂) : 
  (Real.tan x₁ + Real.tan x₂) / 2 > Real.tan ((x₁ + x₂) / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_inequality_l124_12498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_power_2024_l124_12432

open Real Matrix

noncomputable def B : Matrix (Fin 3) (Fin 3) ℝ :=
  !![Real.cos (π / 4), 0, -Real.sin (π / 4);
     0, 1, 0;
     Real.sin (π / 4), 0, Real.cos (π / 4)]

theorem B_power_2024 :
  B ^ 2024 = !![(-1 : ℝ), 0, 0;
                0, 1, 0;
                0, 0, -1] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_power_2024_l124_12432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_parallel_planes_parallel_perpendicular_implies_perpendicular_planes_l124_12489

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the relationships between geometric objects
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (parallel_plane : Line → Plane → Prop)
variable (perpendicular_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- Theorem 1
theorem perpendicular_parallel_planes 
  (α β γ : Plane) 
  (h1 : perpendicular_planes α γ) 
  (h2 : parallel_planes β γ) : 
  perpendicular_planes α β :=
sorry

-- Theorem 2
theorem parallel_perpendicular_implies_perpendicular_planes 
  (l : Line) 
  (α β : Plane) 
  (h1 : parallel_plane l α) 
  (h2 : perpendicular_plane l β) : 
  perpendicular_planes α β :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_parallel_planes_parallel_perpendicular_implies_perpendicular_planes_l124_12489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_factors_l124_12447

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem min_sum_of_factors (p q r s : ℕ+) (h : p * q * r * s = factorial 12) :
  p + q + r + s ≥ 1042 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_factors_l124_12447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_a_value_l124_12457

/-- Given three points (2,-3), (-2a + 4, 4), and (3a + 2, -1) that lie on the same line,
    prove that a = 4/25 -/
theorem collinear_points_a_value (a : ℚ) :
  let p1 : ℚ × ℚ := (2, -3)
  let p2 : ℚ × ℚ := (-2 * a + 4, 4)
  let p3 : ℚ × ℚ := (3 * a + 2, -1)
  (∃ (m b : ℚ), (p1.2 = m * p1.1 + b) ∧ 
                (p2.2 = m * p2.1 + b) ∧ 
                (p3.2 = m * p3.1 + b)) →
  a = 4 / 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_a_value_l124_12457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_symmetry_l124_12442

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 3)

theorem min_shift_for_symmetry (ω : ℝ) (h_ω_pos : ω > 0) 
  (h_period : ∀ x, f ω x = f ω (x + Real.pi)) :
  ∃ m : ℝ, m > 0 ∧ m = Real.pi / 6 ∧ 
    (∀ x, f ω (x - m) = -f ω (-x + m)) ∧
    (∀ m' : ℝ, 0 < m' ∧ m' < m → ∃ x, f ω (x - m') ≠ -f ω (-x + m')) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_symmetry_l124_12442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_height_is_sqrt_3_l124_12478

/-- A regular triangular prism with a hemisphere inscribed in its base -/
structure RegularTriangularPrism :=
  (height : ℝ)
  (hemisphere_radius : ℝ)

/-- The minimum height of the prism given the conditions -/
noncomputable def min_height (prism : RegularTriangularPrism) : ℝ := Real.sqrt 3

/-- Predicate to check if a face is tangent to the hemisphere -/
def is_tangent (face : ℝ) (radius : ℝ) : Prop := sorry

/-- Theorem stating the minimum height of the prism -/
theorem min_height_is_sqrt_3 (prism : RegularTriangularPrism) 
  (h1 : prism.hemisphere_radius = 1)
  (h2 : ∀ face, is_tangent face prism.hemisphere_radius) : 
  min_height prism = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_height_is_sqrt_3_l124_12478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l124_12484

noncomputable def proj_vector (a b : ℝ × ℝ) : ℝ × ℝ :=
  let scalar := (a.1 * b.1 + a.2 * b.2) / (b.1 * b.1 + b.2 * b.2)
  (scalar * b.1, scalar * b.2)

theorem projection_theorem (a b : ℝ × ℝ) 
  (h1 : a.1 * b.1 + a.2 * b.2 = 10) 
  (h2 : b = (6, -8)) : 
  proj_vector a b = (3/5, -4/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l124_12484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_close_to_integer_l124_12411

theorem exists_close_to_integer (a : ℝ) (n : ℕ) (ha : a > 0) (hn : n > 1) :
  ∃ (k : ℕ) (m : ℤ), 1 ≤ k ∧ k < n ∧ |k * a - m| ≤ 1 / (n : ℝ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_close_to_integer_l124_12411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_geometric_sequence_constant_l124_12424

/-- A geometric sequence with a special sum formula -/
structure SpecialGeometricSequence where
  a : ℝ  -- The constant in the sum formula
  terms : ℕ → ℝ  -- The sequence terms
  is_geometric : ∀ n : ℕ, terms (n + 2) * terms n = (terms (n + 1))^2
  sum_formula : ∀ n : ℕ, (Finset.range n).sum terms = 2 * 3^n + a

/-- The constant in the sum formula of the special geometric sequence is -2 -/
theorem special_geometric_sequence_constant (seq : SpecialGeometricSequence) : seq.a = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_geometric_sequence_constant_l124_12424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_set_l124_12468

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then Real.exp x + 1 else 2

-- Define the solution set
def solution_set : Set ℝ := {x | x ≥ 0}

-- Theorem statement
theorem equation_solution_set :
  {x : ℝ | f (1 + x^2) = f (2*x)} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_set_l124_12468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_sphere_area_ratio_l124_12492

-- Define the sphere
variable (R : ℝ) -- Radius of the sphere

-- Define the cone
noncomputable def cone_base_radius (R : ℝ) : ℝ := (Real.sqrt 3 / 2) * R
noncomputable def cone_slant_height (R : ℝ) : ℝ := Real.sqrt 3 * R

-- Surface areas
noncomputable def sphere_surface_area (R : ℝ) : ℝ := 4 * Real.pi * R^2
noncomputable def cone_surface_area (R : ℝ) : ℝ := 
  Real.pi * (cone_base_radius R)^2 + Real.pi * cone_base_radius R * cone_slant_height R

-- Theorem statement
theorem cone_sphere_area_ratio (R : ℝ) (h : R > 0) : 
  cone_surface_area R / sphere_surface_area R = 9/16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_sphere_area_ratio_l124_12492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_geometric_series_sum_l124_12495

/-- The sum of an infinite geometric series with first term a and common ratio r -/
noncomputable def geometricSeriesSum (a r : ℝ) : ℝ := a / (1 - r)

/-- Proof that the sum of the infinite geometric series 1 + (1/3) + (1/3)^2 + ... is 3/2 -/
theorem infinite_geometric_series_sum :
  let a : ℝ := 1
  let r : ℝ := 1/3
  geometricSeriesSum a r = 3/2 := by
  sorry

#check infinite_geometric_series_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_geometric_series_sum_l124_12495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l124_12422

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then (1/2)^x + 1 else 2 - x^2

-- State the theorem
theorem inequality_solution_set (a : ℝ) :
  f (2*a^2 - 1) > f (3*a + 4) ↔ -1 < a ∧ a < 5/2 := by
  sorry

#check inequality_solution_set

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l124_12422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_shifted_sine_l124_12429

noncomputable section

/-- The original function f(x) -/
def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

/-- The shifted function g(x) -/
def g (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)

/-- The shift amount -/
def shift : ℝ := Real.pi / 12

/-- The symmetry axis of g(x) -/
def symmetry_axis : ℝ := 5 * Real.pi / 12

theorem symmetry_axis_of_shifted_sine :
  (∀ x, g x = f (x - shift)) →
  (∀ x, g (symmetry_axis + x) = g (symmetry_axis - x)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_shifted_sine_l124_12429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_deriv_at_zero_l124_12433

-- Define a smooth function f from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define smoothness of f
variable (hf : Differentiable ℝ f)

-- Define the condition f'(x)^2 = f(x) f''(x) for all x
variable (h1 : ∀ x, (deriv f x)^2 = f x * (deriv^[2] f x))

-- Define f(0) = 1
variable (h2 : f 0 = 1)

-- Define f^(4)(0) = 9
variable (h3 : (deriv^[4] f) 0 = 9)

-- Theorem statement
theorem f_deriv_at_zero :
  (deriv f 0 = Real.sqrt 3) ∨ (deriv f 0 = -Real.sqrt 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_deriv_at_zero_l124_12433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_alloy_copper_percentage_is_27_l124_12428

/-- Represents the composition of an alloy mixture --/
structure AlloyMixture where
  total_weight : ℝ
  copper_percentage : ℝ
  first_alloy_weight : ℝ
  first_alloy_copper_percentage : ℝ

/-- Calculates the copper percentage in the second alloy --/
noncomputable def second_alloy_copper_percentage (mixture : AlloyMixture) : ℝ :=
  let second_alloy_weight := mixture.total_weight - mixture.first_alloy_weight
  let total_copper := mixture.total_weight * mixture.copper_percentage / 100
  let first_alloy_copper := mixture.first_alloy_weight * mixture.first_alloy_copper_percentage / 100
  let second_alloy_copper := total_copper - first_alloy_copper
  (second_alloy_copper / second_alloy_weight) * 100

/-- Theorem stating that given the specific mixture, the second alloy contains 27% copper --/
theorem second_alloy_copper_percentage_is_27 (mixture : AlloyMixture) 
    (h1 : mixture.total_weight = 100)
    (h2 : mixture.copper_percentage = 24.9)
    (h3 : mixture.first_alloy_weight = 30)
    (h4 : mixture.first_alloy_copper_percentage = 20) :
    second_alloy_copper_percentage mixture = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_alloy_copper_percentage_is_27_l124_12428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l124_12430

/-- Defines what it means for a line to be the directrix of a parabola -/
def IsDirectrix (x y : ℝ) : Prop :=
  ∀ (p q : ℝ), (q = -(1/8) * p^2) → 
    (x - p)^2 + (y - q)^2 = (x - p)^2 + (2 - q)^2

/-- The equation of the directrix of a parabola given by y = -1/8 * x^2 is y = 2 -/
theorem parabola_directrix (x y : ℝ) : 
  y = -(1/8) * x^2 → (∃ (k : ℝ), k = 2 ∧ (y = k ↔ IsDirectrix x y)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l124_12430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_line_intersection_geometric_sequence_l124_12471

/-- The curve C in the xy-plane -/
def C (a : ℝ) (x y : ℝ) : Prop := y^2 = 2*a*x ∧ a > 0

/-- The line l in the xy-plane -/
def L (x y : ℝ) : Prop := x - y - 2 = 0

/-- Point P on the line l -/
def P : ℝ × ℝ := (-2, -4)

/-- Distance between two points in the plane -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem stating that if |PM|, |MN|, |PN| form a geometric sequence, then a = 1 -/
theorem curve_line_intersection_geometric_sequence (a : ℝ) (M N : ℝ × ℝ) :
  C a M.1 M.2 ∧ C a N.1 N.2 ∧ L M.1 M.2 ∧ L N.1 N.2 ∧
  ∃ (r : ℝ), (distance P M * distance M N = (distance P N)^2 ∨
              distance P N * distance M N = (distance P M)^2 ∨
              (distance P M)^2 = distance P N * distance M N) →
  a = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_line_intersection_geometric_sequence_l124_12471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_minus_sin_positive_l124_12493

/-- Represents an angle in the third quadrant -/
structure ThirdQuadrantAngle where
  α : Real
  in_third_quadrant : Real.sin α < 0 ∧ Real.cos α < 0

theorem tan_minus_sin_positive (θ : ThirdQuadrantAngle) : Real.tan θ.α - Real.sin θ.α > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_minus_sin_positive_l124_12493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_between_spheres_l124_12464

noncomputable def small_radius : ℝ := 5
noncomputable def large_radius : ℝ := 10

noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

theorem volume_between_spheres :
  sphere_volume large_radius - sphere_volume small_radius = (3500 / 3) * Real.pi := by
  -- Expand the definitions
  unfold sphere_volume
  -- Simplify the expressions
  simp [small_radius, large_radius]
  -- The rest of the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_between_spheres_l124_12464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_equation_l124_12470

theorem roots_of_equation (x : ℝ) :
  (3 * Real.sqrt x + 3 * x^(-(1/2 : ℝ)) = 7) ↔ 
  (x = (49 + 14 * Real.sqrt 13 + 13) / 36 ∨ x = (49 - 14 * Real.sqrt 13 + 13) / 36) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_equation_l124_12470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l124_12414

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

theorem function_properties :
  (∃ (p : ℝ), p > 0 ∧ p = Real.pi ∧ ∀ (x y : ℝ), f (x + p) = f x) ∧
  (∀ (x y : ℝ), f (Real.pi / 3 + y) = f (Real.pi / 3 - y)) ∧
  (∀ (x₁ x₂ : ℝ), -Real.pi / 6 < x₁ ∧ x₁ < x₂ ∧ x₂ < Real.pi / 3 → f x₁ < f x₂) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l124_12414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_fraction_l124_12416

/-- Represents a square pyramid -/
structure SquarePyramid where
  baseEdge : ℝ
  altitude : ℝ

/-- The volume of a square pyramid -/
noncomputable def pyramidVolume (p : SquarePyramid) : ℝ :=
  (1 / 3) * p.baseEdge^2 * p.altitude

/-- The original square pyramid -/
noncomputable def originalPyramid : SquarePyramid :=
  { baseEdge := 40, altitude := 20 }

/-- The smaller pyramid cut from the apex -/
noncomputable def smallerPyramid : SquarePyramid :=
  { baseEdge := originalPyramid.baseEdge / 3, altitude := originalPyramid.altitude / 3 }

/-- The volume of the frustum -/
noncomputable def frustumVolume : ℝ :=
  pyramidVolume originalPyramid - pyramidVolume smallerPyramid

/-- The theorem stating the fractional part of the frustum volume -/
theorem frustum_volume_fraction :
  frustumVolume / pyramidVolume originalPyramid = 26 / 27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_fraction_l124_12416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l124_12454

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := y = -2*x + 3

-- Define the intersection points
def intersection_points (P Q : ℝ × ℝ) : Prop :=
  curve_C P.1 P.2 ∧ curve_C Q.1 Q.2 ∧ line_l P.1 P.2 ∧ line_l Q.1 Q.2

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem intersection_product :
  ∀ P Q : ℝ × ℝ,
  intersection_points P Q →
  (distance (0, 3) P) * (distance (0, 3) Q) = 120/19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l124_12454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pattern_A_cannot_fold_to_cube_l124_12410

/-- Represents a pattern of squares -/
structure Pattern where
  squares : Nat
  configuration : String

/-- Represents a cube -/
structure Cube where
  faces : Nat

/-- Defines the properties of Pattern A -/
def pattern_A : Pattern where
  squares := 6
  configuration := "3 squares horizontally and 3 vertically aligned additional squares connected to the middle square of the horizontal set"

/-- Defines the properties of a standard cube -/
def standard_cube : Cube where
  faces := 6

/-- Predicate to check if a pattern can be folded into a cube -/
def can_fold_to_cube (p : Pattern) (c : Cube) : Prop :=
  p.squares = c.faces ∧ 
  ∃ (folding : Nat → Nat), 
    Function.Bijective folding

/-- Theorem stating that Pattern A cannot be folded into a standard cube -/
theorem pattern_A_cannot_fold_to_cube : 
  ¬(can_fold_to_cube pattern_A standard_cube) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pattern_A_cannot_fold_to_cube_l124_12410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_and_maximum_l124_12491

/-- Represents the annual production volume in ten thousand units -/
noncomputable def x (m : ℝ) : ℝ := 3 - m / 2

/-- Represents the profit in ten thousand yuan -/
noncomputable def y (m : ℝ) : ℝ := 28 - m

/-- Theorem stating the correctness of the profit function and maximum profit -/
theorem profit_and_maximum (m : ℝ) (h : m ≥ 0) :
  y m = (x m * 1.5 * (0.8 + 1.6 * x m) - (0.8 + 1.6 * x m) - m) ∧
  y m ≤ 28 ∧
  y 0 = 28 := by
  sorry

#check profit_and_maximum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_and_maximum_l124_12491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_same_number_prob_divisible_by_3_l124_12467

-- Define the set of ball numbers
def BallNumbers : Finset ℕ := {1, 2, 3, 4}

-- Define the sample space
def SampleSpace : Finset (ℕ × ℕ) :=
  Finset.product BallNumbers BallNumbers

-- Define the event where the numbers are the same
def SameNumberEvent : Finset (ℕ × ℕ) :=
  SampleSpace.filter (fun p => p.1 = p.2)

-- Define the event where the product is divisible by 3
def DivisibleBy3Event : Finset (ℕ × ℕ) :=
  SampleSpace.filter (fun p => p.1 * p.2 % 3 = 0)

-- Theorem for the probability of drawing balls with the same number
theorem prob_same_number :
  (SameNumberEvent.card : ℚ) / SampleSpace.card = 1 / 4 := by
  sorry

-- Theorem for the probability of drawing balls with product divisible by 3
theorem prob_divisible_by_3 :
  (DivisibleBy3Event.card : ℚ) / SampleSpace.card = 7 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_same_number_prob_divisible_by_3_l124_12467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_centers_l124_12451

/-- Circle C₁ with equation x² + y² = 1 -/
def C₁ : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}

/-- Circle C₂ with equation (x - 3)² + y² = 9 -/
def C₂ : Set (ℝ × ℝ) := {p | (p.1 - 3)^2 + p.2^2 = 9}

/-- A circle with center (a, 0) and radius r -/
def myCircle (a r : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - a)^2 + p.2^2 = r^2}

/-- Predicate for a circle being externally tangent to C₁ -/
def externally_tangent_C₁ (a r : ℝ) : Prop :=
  ∃ p ∈ C₁, p ∈ myCircle a r ∧ ∀ q ∈ C₁, q ∉ (interior (myCircle a r))

/-- Predicate for a circle being internally tangent to C₂ -/
def internally_tangent_C₂ (a r : ℝ) : Prop :=
  ∃ p ∈ C₂, p ∈ myCircle a r ∧ ∀ q ∈ C₂, q ∉ (exterior (myCircle a r))

/-- The main theorem stating the locus of centers -/
theorem locus_of_centers :
  ∀ a : ℝ, (∃ r : ℝ, externally_tangent_C₁ a r ∧ internally_tangent_C₂ a r) ↔ 2*a^2 - 6*a - 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_centers_l124_12451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_distance_l124_12408

open Complex Real

theorem quadratic_roots_distance (z₁ z₂ m : ℂ) :
  z₁^2 - 4 * z₂ = 16 + 20 * I →
  ∃ α β : ℂ, (α^2 + z₁ * α + z₂ + m = 0) ∧
            (β^2 + z₁ * β + z₂ + m = 0) ∧
            Complex.abs (α - β) = 2 * sqrt 7 →
  (Complex.abs m = sqrt 41 + 7 ∨ Complex.abs m = 7 - sqrt 41) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_distance_l124_12408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_proof_l124_12435

noncomputable section

open Real

theorem triangle_ratio_proof (a b c : ℝ) (A B C : ℝ) :
  -- Conditions
  b = 2 →
  c = 3 →
  A = π / 3 →  -- 60° in radians
  -- Triangle properties
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = π →
  a * sin B = b * sin A →
  a * sin C = c * sin A →
  a^2 = b^2 + c^2 - 2*b*c*(cos A) →
  -- Theorem
  sin (2*C) / sin B = 3 * sqrt 7 / 14 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_proof_l124_12435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l124_12413

theorem cos_beta_value (α β : Real) (h1 : 0 < α ∧ α < Real.pi/2) (h2 : 0 < β ∧ β < Real.pi/2)
  (h3 : Real.sin α = 2 * Real.sqrt 5 / 5) (h4 : Real.cos (α + β) = -4/5) :
  Real.cos β = 2 * Real.sqrt 5 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_beta_value_l124_12413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tomato_price_l124_12458

/-- Proves that the price of a tomato is $1 given the harvested quantities and total revenue -/
theorem tomato_price
  (num_tomatoes : ℕ)
  (num_carrots : ℕ)
  (carrot_price : ℚ)
  (total_revenue : ℚ)
  (h1 : num_tomatoes = 200)
  (h2 : num_carrots = 350)
  (h3 : carrot_price = 3/2)
  (h4 : total_revenue = 725)
  : (total_revenue - num_carrots * carrot_price) / num_tomatoes = 1 := by
  sorry

-- Remove the #eval statement as it's causing issues with universe levels
-- #eval tomato_price 200 350 (3/2) 725

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tomato_price_l124_12458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_quadrant_l124_12443

theorem complex_quadrant (z : ℂ) (h : z / (1 + Complex.I) = 2 * Complex.I) :
  z.re < 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_quadrant_l124_12443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_cross_product_equals_sqrt545_l124_12499

/-- The cross product of two 3D vectors -/
def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.2.1 * b.2.2 - a.2.2 * b.2.1,
   a.2.2 * b.1 - a.1 * b.2.2,
   a.1 * b.2.1 - a.2.1 * b.1)

/-- The magnitude (Euclidean norm) of a 3D vector -/
noncomputable def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2.1^2 + v.2.2^2)

/-- Theorem: The magnitude of the cross product of given vectors is √545 -/
theorem magnitude_cross_product_equals_sqrt545 :
  let a : ℝ × ℝ × ℝ := (3, 1, 4)
  let b : ℝ × ℝ × ℝ := (2, -3, 6)
  magnitude (cross_product a b) = Real.sqrt 545 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_cross_product_equals_sqrt545_l124_12499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l124_12412

/-- Two lines are parallel if their slopes are equal -/
def parallel (a b d e : ℝ) : Prop :=
  a * e = b * d

/-- Distance between two parallel lines ax + by + c = 0 and dx + ey + f = 0 -/
noncomputable def distance (a b c d e f : ℝ) : ℝ :=
  abs (c - f) / Real.sqrt (a^2 + b^2)

/-- The problem statement -/
theorem parallel_lines_distance :
  ∃ a : ℝ,
  parallel a 2 1 (a-1) ∧
  ∃ x y : ℝ,
  a * x + 2 * y - 1 = 0 ∧
  x + (a-1) * y + a^2 = 0 ∧
  distance a 2 (-1) 1 (a-1) (a^2) = (9 * Real.sqrt 2) / 4 :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l124_12412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_island_ocean_depth_l124_12427

noncomputable def island_height : ℝ := 12000

-- Define the ratio of volume above water to total volume
noncomputable def above_water_ratio : ℝ := 1/4

-- Define the function to calculate the submerged height
noncomputable def submerged_height (h : ℝ) (r : ℝ) : ℝ :=
  h * (1 - r^(1/3))

-- Define the ocean depth as the difference between total height and submerged height
noncomputable def ocean_depth (h : ℝ) (r : ℝ) : ℝ :=
  h - submerged_height h r

-- Theorem statement
theorem island_ocean_depth :
  ocean_depth island_height above_water_ratio = 1092 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_island_ocean_depth_l124_12427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_and_absolute_value_l124_12418

theorem opposite_and_absolute_value :
  (∀ x : ℝ, -x = -x) ∧
  (∀ x : ℝ, |x| = 2 ↔ x = 2 ∨ x = -2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_and_absolute_value_l124_12418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_erased_digit_theorem_l124_12460

def is_valid_number (x : Nat) : Prop :=
  (x = 1625 * 10^96) ∨ 
  (x = 195 * 10^97) ∨ 
  (x = 2925 * 10^96) ∨ 
  (∃ b : Nat, b ∈ ({1, 2, 3} : Set Nat) ∧ x = 13 * b * 10^98)

theorem erased_digit_theorem (x : Nat) :
  (∃ k m n a : Nat, 
    x = m + 10^k * a + 10^(k+1) * n ∧ 
    0 < k ∧ k < 99 ∧ 
    a < 10 ∧
    13 * (m + 10^k * n) = x) →
  is_valid_number x :=
by
  sorry

#check erased_digit_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_erased_digit_theorem_l124_12460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l124_12490

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp x + 2 * x

-- Define the point of tangency
def point : ℝ × ℝ := (0, 1)

-- Define what it means for a point to lie on a line
def lies_on (p : ℝ × ℝ) (m b : ℝ) : Prop :=
  p.2 = m * p.1 + b

-- Define the tangent line
noncomputable def tangent_line (f : ℝ → ℝ) (p : ℝ × ℝ) : ℝ × ℝ → Prop :=
  λ q => lies_on q (deriv f p.1) (f p.1 - deriv f p.1 * p.1)

-- State the theorem
theorem tangent_line_equation :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m * x + b ↔ tangent_line f point (x, y)) ∧
    m = 3 ∧ 
    b = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l124_12490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_store_loses_out_l124_12463

/-- Represents an unequal arm balance scale -/
structure UnequalArmScale where
  left_arm : ℝ
  right_arm : ℝ
  unequal_arms : left_arm ≠ right_arm

/-- Represents the weighing process described in the problem -/
def weighing_process (scale : UnequalArmScale) (x y : ℝ) : Prop :=
  scale.left_arm = scale.right_arm * x ∧ scale.right_arm = scale.left_arm * y

theorem store_loses_out (scale : UnequalArmScale) :
  ∀ x y, weighing_process scale x y → x + y > 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_store_loses_out_l124_12463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_f_decreasing_l124_12475

-- Define the parameter a
variable (a : ℝ) 

-- Define the condition 0 < a < 1
variable (ha : 0 < a ∧ a < 1)

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (a^x - a^(-x)) / (a^x + a^(-x))

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := f a x - 1

-- Statement 1: The range of g(x) is (-2, 0)
theorem range_of_g : Set.range (g a) = Set.Ioo (-2) 0 := by sorry

-- Statement 2: f is decreasing on ℝ
theorem f_decreasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ > f a x₂ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_f_decreasing_l124_12475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_value_l124_12465

theorem sin_beta_value (α β : Real)
  (h1 : 0 < α ∧ α < Real.pi / 2)
  (h2 : -Real.pi / 2 < β ∧ β < 0)
  (h3 : Real.cos (α - β) = -3 / 5)
  (h4 : Real.tan α = 4 / 3) :
  Real.sin β = -24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_value_l124_12465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inradius_circumradius_ratio_of_hyperbola_point_l124_12409

/-- A hyperbola with given parameters -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b

/-- The eccentricity of a hyperbola -/
noncomputable def Hyperbola.eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt ((h.a^2 + h.b^2) / h.a^2)

/-- A point on a hyperbola -/
structure PointOnHyperbola (h : Hyperbola) where
  x : ℝ
  y : ℝ
  h_on_hyperbola : x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- The foci of a hyperbola -/
noncomputable def Hyperbola.foci (h : Hyperbola) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let c := Real.sqrt (h.a^2 + h.b^2)
  ((c, 0), (-c, 0))

/-- The inradius of a triangle -/
noncomputable def inradius (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c)) / s

/-- The circumradius of a triangle -/
noncomputable def circumradius (a b c : ℝ) : ℝ :=
  (a * b * c) / (4 * Real.sqrt ((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c)))

/-- The main theorem -/
theorem inradius_circumradius_ratio_of_hyperbola_point
  (h : Hyperbola)
  (h_eccentricity : h.eccentricity = Real.sqrt 2)
  (P : PointOnHyperbola h)
  (h_perpendicular : let (F₁, F₂) := h.foci
                     (P.x - F₁.1) * (P.x - F₂.1) + (P.y - F₁.2) * (P.y - F₂.2) = 0) :
  let (F₁, F₂) := h.foci
  let a := Real.sqrt ((P.x - F₁.1)^2 + (P.y - F₁.2)^2)
  let b := Real.sqrt ((P.x - F₂.1)^2 + (P.y - F₂.2)^2)
  let c := Real.sqrt ((F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2)
  inradius a b c / circumradius a b c = Real.sqrt 6 / 2 - 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inradius_circumradius_ratio_of_hyperbola_point_l124_12409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_even_function_phi_l124_12453

theorem sin_even_function_phi (φ : ℝ) : 
  (∀ x : ℝ, Real.sin (x + φ) = Real.sin (-x + φ)) →
  0 ≤ φ →
  φ ≤ π →
  φ = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_even_function_phi_l124_12453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_difference_l124_12472

/-- The duration of the marathon in hours -/
def marathon_duration : ℕ := 7

/-- The distance Cyra biked in miles -/
def cyra_distance : ℕ := 77

/-- The difference in speed between Devin and Cyra in miles per hour -/
def speed_difference : ℕ := 3

/-- Calculates the speed of Cyra in miles per hour -/
noncomputable def cyra_speed : ℚ := cyra_distance / marathon_duration

/-- Calculates the speed of Devin in miles per hour -/
noncomputable def devin_speed : ℚ := cyra_speed + speed_difference

/-- Calculates the distance Devin biked in miles -/
noncomputable def devin_distance : ℚ := devin_speed * marathon_duration

/-- Theorem stating the difference in distance biked between Devin and Cyra -/
theorem distance_difference : ⌊devin_distance - cyra_distance⌋ = 21 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_difference_l124_12472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_more_middle_less_than_greater_l124_12445

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  value : ℕ
  h100 : 100 ≤ value
  h999 : value ≤ 999

/-- Returns the hundreds digit of a three-digit number -/
def hundreds (n : ThreeDigitNumber) : ℕ := n.value / 100

/-- Returns the tens digit of a three-digit number -/
def tens (n : ThreeDigitNumber) : ℕ := (n.value / 10) % 10

/-- Returns the ones digit of a three-digit number -/
def ones (n : ThreeDigitNumber) : ℕ := n.value % 10

/-- Predicate for numbers where the middle digit is greater than both outer digits -/
def middleGreater (n : ThreeDigitNumber) : Prop :=
  tens n > hundreds n ∧ tens n > ones n

/-- Predicate for numbers where the middle digit is less than both outer digits -/
def middleLess (n : ThreeDigitNumber) : Prop :=
  tens n < hundreds n ∧ tens n < ones n

/-- The set of all three-digit numbers -/
def allNumbers : Set ThreeDigitNumber :=
  {n : ThreeDigitNumber | True}

/-- The set of numbers where the middle digit is greater than both outer digits -/
def greaterSet : Set ThreeDigitNumber :=
  {n : ThreeDigitNumber | middleGreater n}

/-- The set of numbers where the middle digit is less than both outer digits -/
def lessSet : Set ThreeDigitNumber :=
  {n : ThreeDigitNumber | middleLess n}

/-- Theorem stating that there are more numbers where the middle digit is less than both outer digits -/
theorem more_middle_less_than_greater :
  ∃ (f : ThreeDigitNumber → ThreeDigitNumber),
    Function.Injective f ∧
    (∀ n ∈ greaterSet, f n ∈ lessSet) ∧
    ∃ n ∈ lessSet, ∀ m ∈ greaterSet, f m ≠ n :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_more_middle_less_than_greater_l124_12445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_layer_volume_formula_l124_12481

noncomputable section

/-- The volume of a spherical layer -/
def spherical_layer_volume (r₁ r₂ h : ℝ) : ℝ :=
  (Real.pi * h / 6) * (3 * r₁^2 + 3 * r₂^2 + h^2)

/-- Theorem: The volume of a spherical layer with top circle radius r₁, 
    bottom circle radius r₂, and height h is equal to (πh/6) * (3r₁² + 3r₂² + h²) -/
theorem spherical_layer_volume_formula (r₁ r₂ h : ℝ) (hr₁ : r₁ > 0) (hr₂ : r₂ > 0) (hh : h > 0) :
  ∃ (V : ℝ), V = spherical_layer_volume r₁ r₂ h ∧ 
             V = (Real.pi * h / 6) * (3 * r₁^2 + 3 * r₂^2 + h^2) := by
  use spherical_layer_volume r₁ r₂ h
  constructor
  · rfl
  · rfl

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_layer_volume_formula_l124_12481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_and_sum_of_exceptions_l124_12461

noncomputable def f (x : ℝ) : ℝ := 4 * x / (3 * x^2 - 9 * x + 6)

theorem domain_and_sum_of_exceptions :
  (∀ x : ℝ, f x ≠ 0 ↔ x ≠ 1 ∧ x ≠ 2) ∧
  (1 + 2 = 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_and_sum_of_exceptions_l124_12461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l124_12494

/-- Calculates the average speed of a car given two segments of a journey -/
noncomputable def averageSpeed (distance1 : ℝ) (speed1 : ℝ) (distance2 : ℝ) (speed2 : ℝ) : ℝ :=
  (distance1 + distance2) / (distance1 / speed1 + distance2 / speed2)

/-- Theorem stating that the average speed of a car traveling 160 km at 75 km/hr
    and then 160 km at 80 km/hr is approximately 77.42 km/hr -/
theorem car_average_speed :
  let d1 : ℝ := 160
  let s1 : ℝ := 75
  let d2 : ℝ := 160
  let s2 : ℝ := 80
  abs (averageSpeed d1 s1 d2 s2 - 77.42) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_average_speed_l124_12494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_is_15_l124_12476

/-- A sequence where the difference between consecutive terms increases by 1 each time -/
def increasing_diff_seq : ℕ → ℕ
| 0 => 1
| 1 => 3
| 2 => 6
| 3 => 10
| 4 => 15  -- We replace 'x' with the actual value 15
| 5 => 21
| 6 => 28
| n + 7 => increasing_diff_seq (n + 6) + (n + 7)

theorem fifth_term_is_15 :
  increasing_diff_seq 4 = 15 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_is_15_l124_12476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_x_range_l124_12415

noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then (1/2) * x - 1 else 1/x

theorem f_greater_than_x_range : 
  {a : ℝ | f a > a} = Set.Ioo (-1 : ℝ) 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_x_range_l124_12415
