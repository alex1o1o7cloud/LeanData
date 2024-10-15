import Mathlib

namespace NUMINAMATH_CALUDE_contest_scores_l1893_189381

theorem contest_scores (x y : ℝ) : 
  (9 + 8.7 + 9.3 + x + y) / 5 = 9 →
  ((9 - 9)^2 + (8.7 - 9)^2 + (9.3 - 9)^2 + (x - 9)^2 + (y - 9)^2) / 5 = 0.1 →
  |x - y| = 0.8 := by
sorry

end NUMINAMATH_CALUDE_contest_scores_l1893_189381


namespace NUMINAMATH_CALUDE_honey_harvest_calculation_l1893_189324

/-- The amount of honey harvested last year -/
def last_year_harvest : ℕ := 8564 - 6085

/-- The increase in honey harvest this year -/
def harvest_increase : ℕ := 6085

/-- The total amount of honey harvested this year -/
def this_year_harvest : ℕ := 8564

theorem honey_harvest_calculation :
  last_year_harvest = 2479 :=
by sorry

end NUMINAMATH_CALUDE_honey_harvest_calculation_l1893_189324


namespace NUMINAMATH_CALUDE_no_zeros_in_interval_l1893_189376

open Real

theorem no_zeros_in_interval (ω : ℝ) (h_ω_pos : ω > 0) :
  (∀ x ∈ Set.Ioo (π / 2) (3 * π / 2), cos (ω * x - 5 * π / 6) ≠ 0) →
  ω ∈ Set.Ioc 0 (2 / 9) ∪ Set.Icc (2 / 3) (8 / 9) :=
by sorry

end NUMINAMATH_CALUDE_no_zeros_in_interval_l1893_189376


namespace NUMINAMATH_CALUDE_joan_initial_balloons_l1893_189303

/-- The number of balloons Joan lost -/
def lost_balloons : ℕ := 2

/-- The number of balloons Joan currently has -/
def current_balloons : ℕ := 6

/-- The initial number of balloons Joan had -/
def initial_balloons : ℕ := current_balloons + lost_balloons

theorem joan_initial_balloons : initial_balloons = 8 := by
  sorry

end NUMINAMATH_CALUDE_joan_initial_balloons_l1893_189303


namespace NUMINAMATH_CALUDE_problem_solution_l1893_189362

theorem problem_solution (x y : ℚ) : 
  x = 51 → x^3*y - 2*x^2*y + x*y = 51000 → y = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1893_189362


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1893_189341

theorem inequality_solution_set (x : ℝ) : 
  8 * x^2 + 6 * x > 10 ↔ x < -1 ∨ x > 5/4 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1893_189341


namespace NUMINAMATH_CALUDE_binary_product_l1893_189347

-- Define the binary numbers
def binary1 : Nat := 0b11011
def binary2 : Nat := 0b111
def binary3 : Nat := 0b101

-- Define the result
def result : Nat := 0b1110110001

-- Theorem statement
theorem binary_product :
  binary1 * binary2 * binary3 = result := by
  sorry

end NUMINAMATH_CALUDE_binary_product_l1893_189347


namespace NUMINAMATH_CALUDE_P_equals_F_l1893_189326

-- Define the sets P and F
def P : Set ℝ := {y | ∃ x, y = x^2 + 1}
def F : Set ℝ := {x | x ≥ 1}

-- Theorem statement
theorem P_equals_F : P = F := by sorry

end NUMINAMATH_CALUDE_P_equals_F_l1893_189326


namespace NUMINAMATH_CALUDE_inverse_function_sum_l1893_189351

/-- Given a function g and constants a, b, c, d, k satisfying certain conditions,
    prove that a + d = 0 -/
theorem inverse_function_sum (a b c d k : ℝ) :
  (∀ x, (k * (a * x + b)) / (k * (c * x + d)) = 
        ((k * (a * ((k * (a * x + b)) / (k * (c * x + d))) + b)) / 
         (k * (c * ((k * (a * x + b)) / (k * (c * x + d))) + d)))) →
  (a * b * c * d * k ≠ 0) →
  (a + k * c = 0) →
  (a + d = 0) := by
sorry


end NUMINAMATH_CALUDE_inverse_function_sum_l1893_189351


namespace NUMINAMATH_CALUDE_total_seashells_l1893_189395

/-- The number of seashells found by Mary -/
def x : ℝ := 2

/-- The number of seashells found by Keith -/
def y : ℝ := 5

/-- The percentage of cracked seashells found by Mary -/
def m : ℝ := 0.5

/-- The percentage of cracked seashells found by Keith -/
def k : ℝ := 0.6

/-- The total number of seashells found by Mary and Keith -/
def T : ℝ := x + y

/-- The total number of cracked seashells -/
def z : ℝ := m * x + k * y

theorem total_seashells : T = 7 := by
  sorry

end NUMINAMATH_CALUDE_total_seashells_l1893_189395


namespace NUMINAMATH_CALUDE_smallest_base_for_145_l1893_189327

theorem smallest_base_for_145 :
  ∃ (b : ℕ), b = 12 ∧ 
  (∀ (n : ℕ), n^2 ≤ 145 ∧ 145 < n^3 → b ≤ n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_for_145_l1893_189327


namespace NUMINAMATH_CALUDE_log_2_base_10_bounds_l1893_189322

theorem log_2_base_10_bounds : ∃ (log_2_base_10 : ℝ),
  (10 : ℝ) ^ 3 = 1000 ∧
  (10 : ℝ) ^ 4 = 10000 ∧
  (2 : ℝ) ^ 10 = 1024 ∧
  (2 : ℝ) ^ 11 = 2048 ∧
  (2 : ℝ) ^ 12 = 4096 ∧
  (2 : ℝ) ^ 13 = 8192 ∧
  (∀ x > 0, (10 : ℝ) ^ (log_2_base_10 * Real.log x) = x) ∧
  3 / 10 < log_2_base_10 ∧
  log_2_base_10 < 4 / 13 :=
by sorry

end NUMINAMATH_CALUDE_log_2_base_10_bounds_l1893_189322


namespace NUMINAMATH_CALUDE_dave_car_count_l1893_189330

theorem dave_car_count (store1 store2 store3 store4 store5 : ℕ) 
  (h1 : store2 = 14)
  (h2 : store3 = 14)
  (h3 : store4 = 21)
  (h4 : store5 = 25)
  (h5 : (store1 + store2 + store3 + store4 + store5) / 5 = 208/10) :
  store1 = 30 := by
sorry

end NUMINAMATH_CALUDE_dave_car_count_l1893_189330


namespace NUMINAMATH_CALUDE_three_lines_intersection_angles_l1893_189334

-- Define a structure for a line
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

-- Define a structure for an intersection point
structure IntersectionPoint where
  point : ℝ × ℝ

-- Define a function to calculate the angle between two lines
def angleBetweenLines (l1 l2 : Line) : ℝ := sorry

-- Theorem statement
theorem three_lines_intersection_angles 
  (l1 l2 l3 : Line) 
  (p : IntersectionPoint) 
  (h1 : l1.point1 = p.point ∨ l1.point2 = p.point)
  (h2 : l2.point1 = p.point ∨ l2.point2 = p.point)
  (h3 : l3.point1 = p.point ∨ l3.point2 = p.point) :
  angleBetweenLines l1 l2 = 120 ∧ 
  angleBetweenLines l2 l3 = 120 ∧ 
  angleBetweenLines l3 l1 = 120 := by sorry

end NUMINAMATH_CALUDE_three_lines_intersection_angles_l1893_189334


namespace NUMINAMATH_CALUDE_largest_subset_sine_inequality_l1893_189336

theorem largest_subset_sine_inequality :
  ∀ y ∈ Set.Icc 0 Real.pi, ∀ x ∈ Set.Icc 0 Real.pi,
  Real.sin (x + y) ≤ Real.sin x + Real.sin y :=
by sorry

end NUMINAMATH_CALUDE_largest_subset_sine_inequality_l1893_189336


namespace NUMINAMATH_CALUDE_product_of_real_parts_of_complex_solutions_l1893_189343

theorem product_of_real_parts_of_complex_solutions : ∃ (z₁ z₂ : ℂ),
  (z₁^2 + 2*z₁ = Complex.I) ∧
  (z₂^2 + 2*z₂ = Complex.I) ∧
  (z₁ ≠ z₂) ∧
  (Complex.re z₁ * Complex.re z₂ = (1 - Real.sqrt 2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_product_of_real_parts_of_complex_solutions_l1893_189343


namespace NUMINAMATH_CALUDE_parabola_circle_tangent_l1893_189321

/-- Given a parabola and a circle, if the parabola's axis is tangent to the circle, then p = 2 -/
theorem parabola_circle_tangent (p : ℝ) (h1 : p > 0) : 
  (∀ x y : ℝ, y^2 = 2*p*x) →  -- Parabola equation
  (∀ x y : ℝ, x^2 + y^2 - 6*x - 7 = 0) →  -- Circle equation
  (∀ x y : ℝ, x = -p/2) →  -- Parabola's axis equation
  (abs (-p/2 + 3) = 4) →  -- Tangency condition (distance from circle center to axis equals radius)
  p = 2 := by
sorry

end NUMINAMATH_CALUDE_parabola_circle_tangent_l1893_189321


namespace NUMINAMATH_CALUDE_cos_double_angle_problem_l1893_189384

theorem cos_double_angle_problem (α : Real) : 
  (π/2 < α ∧ α < π) →  -- α is in the second quadrant
  (Real.sin α + Real.cos α = Real.sqrt 3 / 3) → 
  Real.cos (2 * α) = -(Real.sqrt 5 / 3) := by
sorry

end NUMINAMATH_CALUDE_cos_double_angle_problem_l1893_189384


namespace NUMINAMATH_CALUDE_largest_primary_divisor_l1893_189332

/-- A positive integer is prime if it has exactly two positive divisors. -/
def IsPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

/-- A positive integer is a primary divisor if for every positive divisor d,
    at least one of d - 1 or d + 1 is prime. -/
def IsPrimaryDivisor (n : ℕ) : Prop :=
  n > 0 ∧ ∀ d : ℕ, d ∣ n → IsPrime (d - 1) ∨ IsPrime (d + 1)

/-- 48 is the largest primary divisor number. -/
theorem largest_primary_divisor : ∀ n : ℕ, IsPrimaryDivisor n → n ≤ 48 :=
  sorry

#check largest_primary_divisor

end NUMINAMATH_CALUDE_largest_primary_divisor_l1893_189332


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1893_189365

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℚ) :
  arithmetic_sequence a →
  a 2 = 3 →
  a 5 + a 7 = 10 →
  a 1 + a 10 = 9.5 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1893_189365


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1893_189333

def f (a b x : ℝ) : ℝ := (x + a) * abs (x + b)

theorem necessary_not_sufficient_condition :
  (∀ x, f a b x = -f a b (-x)) → a = b ∧
  ∃ a b, a = b ∧ ∃ x, f a b x ≠ -f a b (-x) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1893_189333


namespace NUMINAMATH_CALUDE_red_points_centroid_theorem_l1893_189383

/-- Represents a point in a 2D grid -/
structure GridPoint where
  x : Int
  y : Int

/-- Represents a line in a 2D grid -/
inductive GridLine
  | Horizontal (y : Int)
  | Vertical (x : Int)

/-- Definition of a grid -/
structure Grid where
  size : Nat
  horizontal_lines : List GridLine
  vertical_lines : List GridLine

/-- Definition of a triangle -/
structure Triangle where
  a : GridPoint
  b : GridPoint
  c : GridPoint

/-- Calculates the centroid of a triangle -/
def centroid (t : Triangle) : GridPoint :=
  { x := (t.a.x + t.b.x + t.c.x) / 3,
    y := (t.a.y + t.b.y + t.c.y) / 3 }

/-- Theorem statement -/
theorem red_points_centroid_theorem (m : Nat) (grid : Grid)
  (h1 : grid.size = 4 * m + 2)
  (h2 : grid.horizontal_lines.length = 2 * m + 1)
  (h3 : grid.vertical_lines.length = 2 * m + 1) :
  ∃ (A B C D E F : GridPoint),
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
    C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
    D ≠ E ∧ D ≠ F ∧
    E ≠ F ∧
    centroid {a := A, b := B, c := C} = {x := 0, y := 0} ∧
    centroid {a := D, b := E, c := F} = {x := 0, y := 0} :=
  sorry

end NUMINAMATH_CALUDE_red_points_centroid_theorem_l1893_189383


namespace NUMINAMATH_CALUDE_angle_CAG_measure_l1893_189323

-- Define the points
variable (A B C F G : ℝ × ℝ)

-- Define the properties of the configuration
def is_equilateral (A B C : ℝ × ℝ) : Prop := sorry

def is_rectangle (B C F G : ℝ × ℝ) : Prop := sorry

def shared_side (A B C F G : ℝ × ℝ) : Prop := sorry

def longer_side (B C F G : ℝ × ℝ) : Prop := sorry

-- Define the angle measure function
def angle_measure (P Q R : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem angle_CAG_measure 
  (h1 : is_equilateral A B C)
  (h2 : is_rectangle B C F G)
  (h3 : shared_side A B C F G)
  (h4 : longer_side B C F G) :
  angle_measure C A G = 15 := by sorry

end NUMINAMATH_CALUDE_angle_CAG_measure_l1893_189323


namespace NUMINAMATH_CALUDE_john_paid_21_dollars_l1893_189371

/-- Calculates the amount John paid for candy bars -/
def john_payment (total_bars : ℕ) (dave_bars : ℕ) (cost_per_bar : ℚ) : ℚ :=
  (total_bars - dave_bars) * cost_per_bar

/-- Proves that John paid $21 for the candy bars -/
theorem john_paid_21_dollars (total_bars : ℕ) (dave_bars : ℕ) (cost_per_bar : ℚ)
  (h1 : total_bars = 20)
  (h2 : dave_bars = 6)
  (h3 : cost_per_bar = 3/2) :
  john_payment total_bars dave_bars cost_per_bar = 21 := by
  sorry

end NUMINAMATH_CALUDE_john_paid_21_dollars_l1893_189371


namespace NUMINAMATH_CALUDE_quadrilateral_area_theorem_l1893_189364

-- Define the quadrilateral PQRS
def P (k a : ℤ) : ℤ × ℤ := (k, a)
def Q (k a : ℤ) : ℤ × ℤ := (a, k)
def R (k a : ℤ) : ℤ × ℤ := (-k, -a)
def S (k a : ℤ) : ℤ × ℤ := (-a, -k)

-- Define the area function for PQRS
def area_PQRS (k a : ℤ) : ℤ := 2 * |k - a| * |k + a|

-- Theorem statement
theorem quadrilateral_area_theorem (k a : ℤ) 
  (h1 : k > a) (h2 : a > 0) (h3 : area_PQRS k a = 32) : 
  k + a = 8 := by sorry

end NUMINAMATH_CALUDE_quadrilateral_area_theorem_l1893_189364


namespace NUMINAMATH_CALUDE_triangle_area_proof_l1893_189358

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop := sorry

-- Define the side lengths
def SideLength (A B C : ℝ × ℝ) (a b c : ℝ) : Prop := 
  Triangle A B C ∧ 
  (a = Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)) ∧
  (b = Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)) ∧
  (c = Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2))

-- Define the angle C
def AngleC (A B C : ℝ × ℝ) : ℝ := sorry

-- Define the area of the triangle
def TriangleArea (A B C : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_area_proof 
  (A B C : ℝ × ℝ) (a b c : ℝ) 
  (h1 : SideLength A B C a b c)
  (h2 : b = 1)
  (h3 : c = Real.sqrt 3)
  (h4 : AngleC A B C = 2 * Real.pi / 3) :
  TriangleArea A B C = Real.sqrt 3 / 4 := by sorry

end NUMINAMATH_CALUDE_triangle_area_proof_l1893_189358


namespace NUMINAMATH_CALUDE_sally_peaches_theorem_l1893_189382

/-- Represents the number of peaches Sally picked at the orchard -/
def peaches_picked (initial total : ℕ) : ℕ := total - initial

/-- Theorem stating that the number of peaches Sally picked is the difference between her total and initial peaches -/
theorem sally_peaches_theorem (initial total : ℕ) (h : initial ≤ total) :
  peaches_picked initial total = total - initial :=
by sorry

end NUMINAMATH_CALUDE_sally_peaches_theorem_l1893_189382


namespace NUMINAMATH_CALUDE_circle_equation_l1893_189304

theorem circle_equation (x y : ℝ) : 
  (∃ (C : Set (ℝ × ℝ)), 
    (∀ (p : ℝ × ℝ), p ∈ C ↔ (p.1^2 + p.2^2 = 16)) ∧
    ((-4, 0) ∈ C) ∧
    ((x, y) ∈ C)) →
  x^2 + y^2 = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_equation_l1893_189304


namespace NUMINAMATH_CALUDE_cos_300_degrees_l1893_189372

theorem cos_300_degrees : Real.cos (300 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_300_degrees_l1893_189372


namespace NUMINAMATH_CALUDE_inequality_group_C_equality_group_A_equality_group_B_equality_group_D_l1893_189340

theorem inequality_group_C (a b : ℝ) : ∃ a b, 3 * (a + b) ≠ 3 * a + b :=
sorry

theorem equality_group_A (a b : ℝ) : a + b = b + a :=
sorry

theorem equality_group_B (a : ℝ) : 3 * a = a + a + a :=
sorry

theorem equality_group_D (a : ℝ) : a ^ 3 = a * a * a :=
sorry

end NUMINAMATH_CALUDE_inequality_group_C_equality_group_A_equality_group_B_equality_group_D_l1893_189340


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1893_189339

def M : Set ℤ := {0, 1, 2}
def N : Set ℤ := {x | -1 ≤ x ∧ x ≤ 1}

theorem intersection_of_M_and_N : M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1893_189339


namespace NUMINAMATH_CALUDE_two_queens_or_at_least_one_jack_probability_l1893_189306

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of Jacks in a standard deck -/
def num_jacks : ℕ := 4

/-- The number of Queens in a standard deck -/
def num_queens : ℕ := 4

/-- The probability of drawing either two Queens or at least 1 Jack from a standard deck when selecting 2 cards randomly -/
def prob_two_queens_or_at_least_one_jack : ℚ := 2 / 13

theorem two_queens_or_at_least_one_jack_probability :
  prob_two_queens_or_at_least_one_jack = 2 / 13 := by
  sorry

end NUMINAMATH_CALUDE_two_queens_or_at_least_one_jack_probability_l1893_189306


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l1893_189328

theorem arithmetic_mean_of_fractions :
  (1 / 2 : ℚ) * ((3 / 8 : ℚ) + (5 / 9 : ℚ)) = 67 / 144 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l1893_189328


namespace NUMINAMATH_CALUDE_total_questions_in_three_hours_l1893_189301

/-- The number of questions Bob creates in the first hour -/
def first_hour_questions : ℕ := 13

/-- Calculates the number of questions created in the second hour -/
def second_hour_questions : ℕ := 2 * first_hour_questions

/-- Calculates the number of questions created in the third hour -/
def third_hour_questions : ℕ := 2 * second_hour_questions

/-- Theorem: The total number of questions Bob creates in three hours is 91 -/
theorem total_questions_in_three_hours :
  first_hour_questions + second_hour_questions + third_hour_questions = 91 := by
  sorry

end NUMINAMATH_CALUDE_total_questions_in_three_hours_l1893_189301


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l1893_189398

/-- The time taken for a train to cross a bridge -/
theorem train_bridge_crossing_time 
  (train_length : ℝ) 
  (bridge_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (h1 : train_length = 100) 
  (h2 : bridge_length = 170) 
  (h3 : train_speed_kmph = 36) : 
  (train_length + bridge_length) / (train_speed_kmph * 1000 / 3600) = 27 := by
  sorry

#check train_bridge_crossing_time

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l1893_189398


namespace NUMINAMATH_CALUDE_intersection_properties_l1893_189318

/-- Given a line y = a intersecting two curves, prove properties of intersection points -/
theorem intersection_properties (a : ℝ) (x₁ x₂ x₃ : ℝ) : 
  (∀ x, x/Real.exp x = a ↔ x = x₁ ∨ x = x₂) →  -- y = x/e^x intersects y = a at x₁ and x₂
  (∀ x, Real.log x/x = a ↔ x = x₂ ∨ x = x₃) →  -- y = ln(x)/x intersects y = a at x₂ and x₃
  0 < x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ →                 -- order of x₁, x₂, x₃
  (x₂ = a * Real.exp x₂ ∧                      -- Statement A
   x₃ = Real.exp x₂ ∧                          -- Statement C
   x₁ + x₃ > 2 * x₂)                           -- Statement D
:= by sorry

end NUMINAMATH_CALUDE_intersection_properties_l1893_189318


namespace NUMINAMATH_CALUDE_l_shaped_area_l1893_189375

/-- The area of an L-shaped region formed by subtracting three squares from a larger square -/
theorem l_shaped_area (a b c d : ℕ) (h1 : a = 7) (h2 : b = 2) (h3 : c = 2) (h4 : d = 3) : 
  a^2 - (b^2 + c^2 + d^2) = 32 := by
  sorry

end NUMINAMATH_CALUDE_l_shaped_area_l1893_189375


namespace NUMINAMATH_CALUDE_novel_contest_first_prize_l1893_189350

/-- The first place prize in a novel contest --/
def first_place_prize (total_prize : ℕ) (num_winners : ℕ) (second_prize : ℕ) (third_prize : ℕ) (other_prize : ℕ) : ℕ :=
  total_prize - (second_prize + third_prize + (num_winners - 3) * other_prize)

/-- Theorem stating the first place prize is $200 given the contest conditions --/
theorem novel_contest_first_prize :
  first_place_prize 800 18 150 120 22 = 200 := by
  sorry

end NUMINAMATH_CALUDE_novel_contest_first_prize_l1893_189350


namespace NUMINAMATH_CALUDE_foci_coordinates_l1893_189307

/-- Definition of a hyperbola with equation x^2 - y^2/3 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

/-- Definition of the distance from center to focus for this hyperbola -/
def c : ℝ := 2

/-- The coordinates of the foci of the hyperbola x^2 - y^2/3 = 1 are (±2, 0) -/
theorem foci_coordinates :
  ∀ x y : ℝ, hyperbola x y → (x = c ∨ x = -c) ∧ y = 0 :=
sorry

end NUMINAMATH_CALUDE_foci_coordinates_l1893_189307


namespace NUMINAMATH_CALUDE_y_at_64_l1893_189392

-- Define the function y in terms of k and x
def y (k : ℝ) (x : ℝ) : ℝ := k * x^(1/3)

-- State the theorem
theorem y_at_64 (k : ℝ) :
  y k 8 = 4 → y k 64 = 8 := by
  sorry

end NUMINAMATH_CALUDE_y_at_64_l1893_189392


namespace NUMINAMATH_CALUDE_white_paint_calculation_l1893_189315

theorem white_paint_calculation (total_paint blue_paint : ℕ) 
  (h1 : total_paint = 6689)
  (h2 : blue_paint = 6029) :
  total_paint - blue_paint = 660 := by
  sorry

end NUMINAMATH_CALUDE_white_paint_calculation_l1893_189315


namespace NUMINAMATH_CALUDE_digit_sum_is_seventeen_l1893_189360

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a two-digit number -/
def TwoDigitNumber := Fin 100

/-- Represents a three-digit number -/
def ThreeDigitNumber := Fin 1000

/-- The equation (AB) * (CD) = GGG -/
def satisfiesEquation (A B C D G : Digit) : Prop :=
  ∃ (AB CD : TwoDigitNumber) (GGG : ThreeDigitNumber),
    AB.val = 10 * A.val + B.val ∧
    CD.val = 10 * C.val + D.val ∧
    GGG.val = 100 * G.val + 10 * G.val + G.val ∧
    AB.val * CD.val = GGG.val

/-- All digits are distinct -/
def allDistinct (A B C D G : Digit) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ G ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ G ∧
  C ≠ D ∧ C ≠ G ∧
  D ≠ G

theorem digit_sum_is_seventeen :
  ∃ (A B C D G : Digit),
    satisfiesEquation A B C D G ∧
    allDistinct A B C D G ∧
    A.val + B.val + C.val + D.val + G.val = 17 :=
by sorry

end NUMINAMATH_CALUDE_digit_sum_is_seventeen_l1893_189360


namespace NUMINAMATH_CALUDE_dog_does_not_catch_hare_l1893_189309

/-- Represents the chase scenario between a dog and a hare -/
structure ChaseScenario where
  dog_speed : ℝ
  hare_speed : ℝ
  initial_distance : ℝ
  bushes_distance : ℝ

/-- Determines if the dog catches the hare before it reaches the bushes -/
def dog_catches_hare (scenario : ChaseScenario) : Prop :=
  let relative_speed := scenario.dog_speed - scenario.hare_speed
  let catch_time := scenario.initial_distance / relative_speed
  let hare_distance := scenario.hare_speed * catch_time
  hare_distance < scenario.bushes_distance

/-- The theorem stating that the dog does not catch the hare -/
theorem dog_does_not_catch_hare (scenario : ChaseScenario)
  (h1 : scenario.dog_speed = 17)
  (h2 : scenario.hare_speed = 14)
  (h3 : scenario.initial_distance = 150)
  (h4 : scenario.bushes_distance = 520) :
  ¬(dog_catches_hare scenario) := by
  sorry

#check dog_does_not_catch_hare

end NUMINAMATH_CALUDE_dog_does_not_catch_hare_l1893_189309


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l1893_189374

/-- 
Given a quadratic equation 6x² - 1 = 3x, when converted to the general form ax² + bx + c = 0,
the coefficient of x² (a) is 6 and the coefficient of x (b) is -3.
-/
theorem quadratic_equation_coefficients :
  ∃ (a b c : ℝ), 
    (∀ x, 6 * x^2 - 1 = 3 * x ↔ a * x^2 + b * x + c = 0) ∧
    a = 6 ∧ 
    b = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l1893_189374


namespace NUMINAMATH_CALUDE_count_zeros_up_to_2500_l1893_189311

/-- A function that returns true if a natural number contains the digit 0 in its decimal representation -/
def containsZero (n : ℕ) : Bool :=
  sorry

/-- The count of numbers less than or equal to 2500 that contain the digit 0 -/
def countZeros : ℕ := (List.range 2501).filter containsZero |>.length

/-- Theorem stating that the count of numbers less than or equal to 2500 containing 0 is 591 -/
theorem count_zeros_up_to_2500 : countZeros = 591 := by
  sorry

end NUMINAMATH_CALUDE_count_zeros_up_to_2500_l1893_189311


namespace NUMINAMATH_CALUDE_length_OB_is_sqrt_13_l1893_189305

-- Define the point A
def A : ℝ × ℝ × ℝ := (1, 2, 3)

-- Define the projection B of A onto the yOz plane
def B : ℝ × ℝ × ℝ := (0, A.2.1, A.2.2)

-- Define the origin O
def O : ℝ × ℝ × ℝ := (0, 0, 0)

-- Theorem to prove
theorem length_OB_is_sqrt_13 : 
  Real.sqrt ((B.1 - O.1)^2 + (B.2.1 - O.2.1)^2 + (B.2.2 - O.2.2)^2) = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_length_OB_is_sqrt_13_l1893_189305


namespace NUMINAMATH_CALUDE_alice_work_problem_l1893_189342

/-- Alice's work problem -/
theorem alice_work_problem (total_days : ℕ) (daily_wage : ℕ) (daily_loss : ℕ) (total_earnings : ℤ) :
  total_days = 20 →
  daily_wage = 80 →
  daily_loss = 40 →
  total_earnings = 880 →
  ∃ (days_not_worked : ℕ),
    days_not_worked = 6 ∧
    days_not_worked ≤ total_days ∧
    (daily_wage * (total_days - days_not_worked) : ℤ) - (daily_loss * days_not_worked : ℤ) = total_earnings :=
by sorry

end NUMINAMATH_CALUDE_alice_work_problem_l1893_189342


namespace NUMINAMATH_CALUDE_boys_age_l1893_189345

theorem boys_age (boy daughter wife father : ℕ) : 
  boy = 5 * daughter →
  wife = 5 * boy →
  father = 2 * wife →
  boy + daughter + wife + father = 81 →
  boy = 5 :=
by sorry

end NUMINAMATH_CALUDE_boys_age_l1893_189345


namespace NUMINAMATH_CALUDE_two_digit_number_problem_l1893_189308

theorem two_digit_number_problem (n m : ℕ) : 
  10 ≤ m ∧ m < n ∧ n ≤ 99 →  -- n and m are 2-digit numbers, n > m
  n - m = 58 →  -- difference is 58
  n^2 % 100 = m^2 % 100 →  -- last two digits of squares are the same
  m = 21 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_problem_l1893_189308


namespace NUMINAMATH_CALUDE_f_composition_equals_constant_l1893_189399

-- Define the complex function f
noncomputable def f (z : ℂ) : ℂ :=
  if z.im ≠ 0 then 2 * z ^ 2 else -3 * z ^ 2

-- State the theorem
theorem f_composition_equals_constant : f (f (f (f (1 + I)))) = (-28311552 : ℂ) := by
  sorry

end NUMINAMATH_CALUDE_f_composition_equals_constant_l1893_189399


namespace NUMINAMATH_CALUDE_marble_theorem_l1893_189312

def marble_problem (wolfgang ludo michael shania gabriel : ℕ) : Prop :=
  wolfgang = 16 ∧
  ludo = wolfgang + wolfgang / 4 ∧
  michael = 2 * (wolfgang + ludo) / 3 ∧
  shania = 2 * ludo ∧
  gabriel = wolfgang + ludo + michael + shania - 1 ∧
  (wolfgang + ludo + michael + shania + gabriel) / 5 = 39

theorem marble_theorem : ∃ wolfgang ludo michael shania gabriel : ℕ,
  marble_problem wolfgang ludo michael shania gabriel := by
  sorry

end NUMINAMATH_CALUDE_marble_theorem_l1893_189312


namespace NUMINAMATH_CALUDE_fraction_to_decimal_equiv_l1893_189386

theorem fraction_to_decimal_equiv : (5 : ℚ) / 8 = 0.625 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_equiv_l1893_189386


namespace NUMINAMATH_CALUDE_tetrahedralContactsFormula_l1893_189388

/-- The number of contact points in a tetrahedral stack of spheres -/
def tetrahedralContacts (n : ℕ) : ℕ := n^3 - n

/-- Theorem: The number of contact points in a tetrahedral stack of spheres
    with n spheres along each edge is n³ - n -/
theorem tetrahedralContactsFormula (n : ℕ) :
  tetrahedralContacts n = n^3 - n := by
  sorry

end NUMINAMATH_CALUDE_tetrahedralContactsFormula_l1893_189388


namespace NUMINAMATH_CALUDE_correct_hourly_wage_l1893_189355

/-- The hourly wage for a manufacturing plant worker --/
def hourly_wage : ℝ :=
  12.50

/-- The piece rate per widget --/
def piece_rate : ℝ :=
  0.16

/-- The number of widgets produced in a week --/
def widgets_per_week : ℕ :=
  1000

/-- The number of hours worked in a week --/
def hours_per_week : ℕ :=
  40

/-- The total earnings for a week --/
def total_earnings : ℝ :=
  660

theorem correct_hourly_wage :
  hourly_wage * hours_per_week + piece_rate * widgets_per_week = total_earnings :=
by sorry

end NUMINAMATH_CALUDE_correct_hourly_wage_l1893_189355


namespace NUMINAMATH_CALUDE_sin_40_minus_sin_80_l1893_189391

theorem sin_40_minus_sin_80 : 
  Real.sin (40 * π / 180) - Real.sin (80 * π / 180) = 
    Real.sin (40 * π / 180) * (1 - 2 * Real.sqrt (1 - Real.sin (40 * π / 180) ^ 2)) := by
  sorry

end NUMINAMATH_CALUDE_sin_40_minus_sin_80_l1893_189391


namespace NUMINAMATH_CALUDE_water_tank_fill_time_l1893_189387

/-- Represents the time (in hours) it takes to fill a water tank -/
def fill_time : ℝ → ℝ → ℝ → Prop :=
  λ T leak_empty_time leak_fill_time =>
    (1 / T - 1 / leak_empty_time = 1 / leak_fill_time) ∧
    (leak_fill_time = T + 1)

theorem water_tank_fill_time :
  ∃ (T : ℝ), T > 0 ∧ fill_time T 30 (T + 1) ∧ T = 5 := by
  sorry

end NUMINAMATH_CALUDE_water_tank_fill_time_l1893_189387


namespace NUMINAMATH_CALUDE_joan_seashells_problem_l1893_189377

theorem joan_seashells_problem (initial : ℕ) (remaining : ℕ) (sam_to_lily_ratio : ℕ) :
  initial = 70 →
  remaining = 27 →
  sam_to_lily_ratio = 2 →
  ∃ (sam lily : ℕ),
    initial = remaining + sam + lily ∧
    sam = sam_to_lily_ratio * lily ∧
    sam = 28 :=
by sorry

end NUMINAMATH_CALUDE_joan_seashells_problem_l1893_189377


namespace NUMINAMATH_CALUDE_total_legs_of_daniels_animals_l1893_189397

/-- The number of legs an animal has -/
def legs (animal : String) : ℕ :=
  match animal with
  | "horse" => 4
  | "dog" => 4
  | "cat" => 4
  | "turtle" => 4
  | "goat" => 4
  | _ => 0

/-- Daniel's collection of animals -/
def daniels_animals : List (String × ℕ) :=
  [("horse", 2), ("dog", 5), ("cat", 7), ("turtle", 3), ("goat", 1)]

/-- Theorem: The total number of legs of Daniel's animals is 72 -/
theorem total_legs_of_daniels_animals :
  (daniels_animals.map (fun (animal, count) => count * legs animal)).sum = 72 := by
  sorry

end NUMINAMATH_CALUDE_total_legs_of_daniels_animals_l1893_189397


namespace NUMINAMATH_CALUDE_line_passes_through_quadrants_l1893_189354

-- Define the line ax + by + c = 0
def line (a b c : ℝ) (x y : ℝ) : Prop := a * x + b * y + c = 0

-- Define the quadrants
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0
def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

-- State the theorem
theorem line_passes_through_quadrants (a b c : ℝ) 
  (h1 : a * c < 0) (h2 : b * c < 0) :
  ∃ (x1 y1 x2 y2 x4 y4 : ℝ),
    line a b c x1 y1 ∧ first_quadrant x1 y1 ∧
    line a b c x2 y2 ∧ second_quadrant x2 y2 ∧
    line a b c x4 y4 ∧ fourth_quadrant x4 y4 :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_quadrants_l1893_189354


namespace NUMINAMATH_CALUDE_complex_sum_theorem_l1893_189331

theorem complex_sum_theorem (a b c d : ℝ) (ω : ℂ) : 
  a ≠ -1 → b ≠ -1 → c ≠ -1 → d ≠ -1 →
  ω^3 = 1 →
  ω ≠ 1 →
  (1 / (a + ω)) + (1 / (b + ω)) + (1 / (c + ω)) + (1 / (d + ω)) = 3 / ω →
  (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) + (1 / (d + 1)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_theorem_l1893_189331


namespace NUMINAMATH_CALUDE_square_formation_theorem_l1893_189366

def sum_of_natural_numbers (n : ℕ) : ℕ := n * (n + 1) / 2

def can_form_square (n : ℕ) : Prop :=
  sum_of_natural_numbers n % 4 = 0

def min_breaks_for_square (n : ℕ) : ℕ :=
  let total := sum_of_natural_numbers n
  let remainder := total % 4
  if remainder = 0 then 0
  else if remainder = 1 || remainder = 3 then 1
  else 2

theorem square_formation_theorem :
  (min_breaks_for_square 12 = 2) ∧
  (can_form_square 15 = true) := by sorry

end NUMINAMATH_CALUDE_square_formation_theorem_l1893_189366


namespace NUMINAMATH_CALUDE_final_salary_ratio_l1893_189373

/-- Represents the sequence of salary adjustments throughout the year -/
def salary_adjustments : List (ℝ → ℝ) := [
  (· * 1.20),       -- 20% increase after 2 months
  (· * 0.90),       -- 10% decrease in 3rd month
  (· * 1.12),       -- 12% increase in 4th month
  (· * 0.92),       -- 8% decrease in 5th month
  (· * 1.12),       -- 12% increase in 6th month
  (· * 0.92),       -- 8% decrease in 7th month
  (· * 1.08),       -- 8% bonus in 8th month
  (· * 0.50),       -- 50% decrease due to financial crisis
  (· * 0.90),       -- 10% decrease in 9th month
  (· * 1.15),       -- 15% increase in 10th month
  (· * 0.90),       -- 10% decrease in 11th month
  (· * 1.50)        -- 50% increase in last month
]

/-- Applies a list of functions sequentially to an initial value -/
def apply_adjustments (adjustments : List (ℝ → ℝ)) (initial : ℝ) : ℝ :=
  adjustments.foldl (λ acc f => f acc) initial

/-- Theorem stating the final salary ratio after adjustments -/
theorem final_salary_ratio (S : ℝ) (hS : S > 0) :
  let initial_after_tax := 0.70 * S
  let final_salary := apply_adjustments salary_adjustments initial_after_tax
  ∃ ε > 0, abs (final_salary / initial_after_tax - 0.8657) < ε :=
sorry

end NUMINAMATH_CALUDE_final_salary_ratio_l1893_189373


namespace NUMINAMATH_CALUDE_smallest_sum_X_plus_c_l1893_189353

theorem smallest_sum_X_plus_c : ∀ (X c : ℕ),
  X < 5 → 
  X > 0 →
  c > 6 →
  (31 * X = 4 * c + 4) →
  ∀ (Y d : ℕ), Y < 5 → Y > 0 → d > 6 → (31 * Y = 4 * d + 4) →
  X + c ≤ Y + d :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_X_plus_c_l1893_189353


namespace NUMINAMATH_CALUDE_fourth_power_sum_l1893_189361

theorem fourth_power_sum (α β γ : ℂ) 
  (h1 : α + β + γ = 1)
  (h2 : α^2 + β^2 + γ^2 = 5)
  (h3 : α^3 + β^3 + γ^3 = 9) :
  α^4 + β^4 + γ^4 = 56 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_sum_l1893_189361


namespace NUMINAMATH_CALUDE_product_cde_value_l1893_189359

theorem product_cde_value (a b c d e f : ℝ) 
  (h1 : a * b * c = 130)
  (h2 : b * c * d = 65)
  (h3 : d * e * f = 250)
  (h4 : (a * f) / (c * d) = 0.6666666666666666) :
  c * d * e = 750 := by
  sorry

end NUMINAMATH_CALUDE_product_cde_value_l1893_189359


namespace NUMINAMATH_CALUDE_largest_factorial_with_100_zeros_l1893_189313

/-- Count the number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625)

/-- The largest positive integer n such that n! ends with exactly 100 zeros -/
theorem largest_factorial_with_100_zeros : 
  (∀ m : ℕ, m > 409 → trailingZeros m > 100) ∧ 
  trailingZeros 409 = 100 :=
sorry

end NUMINAMATH_CALUDE_largest_factorial_with_100_zeros_l1893_189313


namespace NUMINAMATH_CALUDE_tony_running_speed_l1893_189356

/-- The distance to the store in miles -/
def distance : ℝ := 4

/-- Tony's walking speed in miles per hour -/
def walking_speed : ℝ := 2

/-- The average time Tony spends to get to the store in minutes -/
def average_time : ℝ := 56

/-- Tony's running speed in miles per hour -/
def running_speed : ℝ := 10

theorem tony_running_speed :
  let time_walking := (distance / walking_speed) * 60
  let time_running := (distance / running_speed) * 60
  (time_walking + 2 * time_running) / 3 = average_time :=
by sorry

end NUMINAMATH_CALUDE_tony_running_speed_l1893_189356


namespace NUMINAMATH_CALUDE_perfect_matching_exists_l1893_189352

/-- Represents a polygon with unit area -/
structure UnitPolygon where
  -- Add necessary fields here
  area : ℝ
  area_eq_one : area = 1

/-- Represents a square sheet of side length 2019 cut into 2019² unit polygons -/
structure Sheet where
  side_length : ℕ
  side_length_eq_2019 : side_length = 2019
  polygons : Finset UnitPolygon
  polygon_count : polygons.card = side_length * side_length

/-- Represents the intersection between two polygons from different sheets -/
def intersects (p1 p2 : UnitPolygon) : Prop :=
  sorry

/-- The main theorem -/
theorem perfect_matching_exists (sheet1 sheet2 : Sheet) : ∃ (matching : Finset (UnitPolygon × UnitPolygon)), 
  matching.card = 2019 * 2019 ∧ 
  (∀ (p1 p2 : UnitPolygon), (p1, p2) ∈ matching → p1 ∈ sheet1.polygons ∧ p2 ∈ sheet2.polygons ∧ intersects p1 p2) ∧
  (∀ p1 ∈ sheet1.polygons, ∃! p2, (p1, p2) ∈ matching) ∧
  (∀ p2 ∈ sheet2.polygons, ∃! p1, (p1, p2) ∈ matching) :=
sorry

end NUMINAMATH_CALUDE_perfect_matching_exists_l1893_189352


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1893_189325

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 5 * a 11 = 3 →
  a 3 + a 13 = 4 →
  a 15 / a 5 = 1/3 ∨ a 15 / a 5 = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1893_189325


namespace NUMINAMATH_CALUDE_intersection_point_sum_l1893_189379

theorem intersection_point_sum (c d : ℝ) :
  (∃ x y : ℝ, x = (1/3) * y + c ∧ y = (1/3) * x + d) →
  (3 = (1/3) * 6 + c ∧ 6 = (1/3) * 3 + d) →
  c + d = 6 := by
sorry

end NUMINAMATH_CALUDE_intersection_point_sum_l1893_189379


namespace NUMINAMATH_CALUDE_round_table_seating_l1893_189300

/-- The number of unique circular arrangements of n distinct objects -/
def circularArrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- The number of people to be seated around the round table -/
def numberOfPeople : ℕ := 8

theorem round_table_seating :
  circularArrangements numberOfPeople = 5040 := by
  sorry

end NUMINAMATH_CALUDE_round_table_seating_l1893_189300


namespace NUMINAMATH_CALUDE_sticker_ratio_l1893_189389

/-- Proves that the ratio of silver stickers to gold stickers is 2:1 --/
theorem sticker_ratio :
  ∀ (gold silver bronze : ℕ),
  gold = 50 →
  bronze = silver - 20 →
  gold + silver + bronze = 5 * 46 →
  silver / gold = 2 := by
  sorry

end NUMINAMATH_CALUDE_sticker_ratio_l1893_189389


namespace NUMINAMATH_CALUDE_mike_weekly_pullups_l1893_189310

/-- The number of pull-ups Mike does each time he enters his office -/
def pullups_per_entry : ℕ := 2

/-- The number of times Mike enters his office per day -/
def office_entries_per_day : ℕ := 5

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The total number of pull-ups Mike does in a week -/
def total_pullups_per_week : ℕ := pullups_per_entry * office_entries_per_day * days_in_week

/-- Theorem stating that Mike does 70 pull-ups in a week -/
theorem mike_weekly_pullups : total_pullups_per_week = 70 := by
  sorry

end NUMINAMATH_CALUDE_mike_weekly_pullups_l1893_189310


namespace NUMINAMATH_CALUDE_read_book_series_l1893_189370

/-- The number of weeks required to read a book series -/
def weeks_to_read (total_books : ℕ) (first_week : ℕ) (second_week : ℕ) (subsequent_weeks : ℕ) : ℕ :=
  let remaining_books := total_books - first_week - second_week
  let additional_weeks := (remaining_books + subsequent_weeks - 1) / subsequent_weeks
  2 + additional_weeks

/-- Theorem: It takes 7 weeks to read the book series under given conditions -/
theorem read_book_series : weeks_to_read 54 6 3 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_read_book_series_l1893_189370


namespace NUMINAMATH_CALUDE_perpendicular_vector_scalar_l1893_189320

/-- Given vectors a and b in ℝ², if a + t*b is perpendicular to a, then t = -5/8 -/
theorem perpendicular_vector_scalar (a b : ℝ × ℝ) (t : ℝ) 
  (h1 : a = (1, 2))
  (h2 : b = (2, 3))
  (h3 : (a.1 + t * b.1, a.2 + t * b.2) • a = 0) :
  t = -5/8 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vector_scalar_l1893_189320


namespace NUMINAMATH_CALUDE_tan_2alpha_proof_l1893_189396

theorem tan_2alpha_proof (α : Real) (h : Real.sin α + 2 * Real.cos α = Real.sqrt 10 / 2) :
  Real.tan (2 * α) = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_2alpha_proof_l1893_189396


namespace NUMINAMATH_CALUDE_uncoverable_iff_odd_specified_boards_uncoverable_l1893_189346

/-- Represents a board configuration -/
structure Board :=
  (rows : ℕ)
  (cols : ℕ)
  (missing : ℕ)

/-- Calculates the number of coverable squares on a board -/
def coverableSquares (b : Board) : ℕ :=
  b.rows * b.cols - b.missing

/-- Determines if a board can be completely covered by dominoes -/
def canBeCovered (b : Board) : Prop :=
  coverableSquares b % 2 = 0

/-- Theorem: A board cannot be covered iff the number of coverable squares is odd -/
theorem uncoverable_iff_odd (b : Board) :
  ¬(canBeCovered b) ↔ coverableSquares b % 2 = 1 :=
sorry

/-- Examples of board configurations -/
def board_7x3 : Board := ⟨7, 3, 0⟩
def board_6x4_unpainted : Board := ⟨6, 4, 1⟩
def board_5x7 : Board := ⟨5, 7, 0⟩
def board_8x8_missing : Board := ⟨8, 8, 1⟩

/-- Theorem: The specified boards cannot be covered -/
theorem specified_boards_uncoverable :
  (¬(canBeCovered board_7x3)) ∧
  (¬(canBeCovered board_6x4_unpainted)) ∧
  (¬(canBeCovered board_5x7)) ∧
  (¬(canBeCovered board_8x8_missing)) :=
sorry

end NUMINAMATH_CALUDE_uncoverable_iff_odd_specified_boards_uncoverable_l1893_189346


namespace NUMINAMATH_CALUDE_not_integer_fraction_l1893_189368

theorem not_integer_fraction (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a ≠ b) :
  ¬ ∃ (n : ℤ), (a^2 + b^2) / (a^2 - b^2) = n := by
  sorry

end NUMINAMATH_CALUDE_not_integer_fraction_l1893_189368


namespace NUMINAMATH_CALUDE_blood_cell_count_l1893_189378

theorem blood_cell_count (total : ℕ) (first_sample : ℕ) (second_sample : ℕ) 
  (h1 : total = 7341)
  (h2 : first_sample = 4221)
  (h3 : total = first_sample + second_sample) : 
  second_sample = 3120 := by
  sorry

end NUMINAMATH_CALUDE_blood_cell_count_l1893_189378


namespace NUMINAMATH_CALUDE_circle_contains_at_least_250_points_l1893_189394

/-- A circle on a grid --/
structure GridCircle where
  radius : ℝ
  gridSize : ℝ

/-- The number of grid points inside a circle --/
def gridPointsInside (c : GridCircle) : ℕ :=
  sorry

/-- Theorem: A circle with radius 10 on a unit grid contains at least 250 grid points --/
theorem circle_contains_at_least_250_points (c : GridCircle) 
  (h1 : c.radius = 10)
  (h2 : c.gridSize = 1) : 
  gridPointsInside c ≥ 250 := by
  sorry

end NUMINAMATH_CALUDE_circle_contains_at_least_250_points_l1893_189394


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1893_189348

theorem simplify_and_evaluate (x : ℝ) (h : x ≠ 1) : 
  (2 / (x - 1) + 1) / ((x + 1) / (x^2 - 2*x + 1)) = x - 1 :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1893_189348


namespace NUMINAMATH_CALUDE_cubic_root_product_l1893_189367

theorem cubic_root_product : ∃ (z₁ z₂ : ℂ),
  z₁^3 = -27 ∧ z₂^3 = -27 ∧ 
  (∃ (a₁ b₁ a₂ b₂ : ℝ), z₁ = a₁ + b₁ * I ∧ z₂ = a₂ + b₂ * I ∧ a₁ > 0 ∧ a₂ > 0) ∧
  z₁ * z₂ = 9 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_product_l1893_189367


namespace NUMINAMATH_CALUDE_father_son_age_relationship_l1893_189380

/-- Represents the age relationship between a father and his son Ronit -/
structure AgeRelationship where
  ronit_age : ℕ
  father_age : ℕ
  years_passed : ℕ

/-- The conditions of the problem -/
def age_conditions (ar : AgeRelationship) : Prop :=
  (ar.father_age = 4 * ar.ronit_age) ∧
  (ar.father_age + ar.years_passed = (5/2) * (ar.ronit_age + ar.years_passed)) ∧
  (ar.father_age + ar.years_passed + 8 = 2 * (ar.ronit_age + ar.years_passed + 8))

theorem father_son_age_relationship :
  ∃ ar : AgeRelationship, age_conditions ar ∧ ar.years_passed = 8 := by
  sorry

end NUMINAMATH_CALUDE_father_son_age_relationship_l1893_189380


namespace NUMINAMATH_CALUDE_triangle_altitude_circumradius_l1893_189302

/-- For any triangle with sides a, b, c, altitude ha from vertex A to side a,
    and circumradius R, the equation ha = bc / (2R) holds. -/
theorem triangle_altitude_circumradius (a b c ha R : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ ha > 0 ∧ R > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_altitude : ha = (2 * (a.sqrt * b.sqrt * c.sqrt + (a + b + c) / 2)) / a)
  (h_circumradius : R = (a * b * c) / (4 * (a.sqrt * b.sqrt * c.sqrt + (a + b + c) / 2))) :
  ha = b * c / (2 * R) := by
sorry

end NUMINAMATH_CALUDE_triangle_altitude_circumradius_l1893_189302


namespace NUMINAMATH_CALUDE_base_10_to_base_4_123_l1893_189357

/-- Converts a natural number to its base 4 representation -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Checks if a list of digits is a valid base 4 representation -/
def isValidBase4 (digits : List ℕ) : Prop :=
  sorry

theorem base_10_to_base_4_123 :
  let base4Repr := toBase4 123
  isValidBase4 base4Repr ∧ base4Repr = [1, 3, 2, 3] :=
by sorry

end NUMINAMATH_CALUDE_base_10_to_base_4_123_l1893_189357


namespace NUMINAMATH_CALUDE_jeff_running_schedule_l1893_189369

/-- Jeff's running schedule problem -/
theorem jeff_running_schedule 
  (weekday_run : ℕ) -- Planned running time per weekday in minutes
  (thursday_cut : ℕ) -- Minutes cut from Thursday's run
  (total_time : ℕ) -- Total running time for the week in minutes
  (h1 : weekday_run = 60)
  (h2 : thursday_cut = 20)
  (h3 : total_time = 290) :
  total_time - (4 * weekday_run + (weekday_run - thursday_cut)) = 10 :=
by sorry

end NUMINAMATH_CALUDE_jeff_running_schedule_l1893_189369


namespace NUMINAMATH_CALUDE_complex_product_real_l1893_189393

theorem complex_product_real (a : ℝ) :
  (Complex.I : ℂ) * (Complex.I : ℂ) = -1 →
  (((a : ℂ) - 2 * Complex.I) * (3 + Complex.I)).im = 0 ↔ a = 6 :=
sorry

end NUMINAMATH_CALUDE_complex_product_real_l1893_189393


namespace NUMINAMATH_CALUDE_play_attendance_l1893_189316

theorem play_attendance (total_people : ℕ) (adult_price child_price : ℕ) (total_receipts : ℕ) :
  total_people = 610 →
  adult_price = 2 →
  child_price = 1 →
  total_receipts = 960 →
  ∃ (adults children : ℕ),
    adults + children = total_people ∧
    adult_price * adults + child_price * children = total_receipts ∧
    adults = 350 :=
by sorry

end NUMINAMATH_CALUDE_play_attendance_l1893_189316


namespace NUMINAMATH_CALUDE_eight_point_five_million_scientific_notation_l1893_189317

theorem eight_point_five_million_scientific_notation :
  (8.5 * 1000000 : ℝ) = 8.5 * (10 ^ 6) :=
by sorry

end NUMINAMATH_CALUDE_eight_point_five_million_scientific_notation_l1893_189317


namespace NUMINAMATH_CALUDE_unique_positive_solution_l1893_189390

theorem unique_positive_solution (x y z : ℝ) : 
  x > 0 →
  x * y + 3 * x + 2 * y = 12 →
  y * z + 5 * y + 3 * z = 18 →
  x * z + 2 * x + 3 * z = 18 →
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l1893_189390


namespace NUMINAMATH_CALUDE_perfect_square_factors_count_l1893_189337

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def count_perfect_square_factors (a b c : ℕ) : ℕ :=
  (a + 1) * (b + 1) * (c + 1)

theorem perfect_square_factors_count :
  count_perfect_square_factors 6 7 9 = 560 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_factors_count_l1893_189337


namespace NUMINAMATH_CALUDE_cos_alpha_plus_pi_third_l1893_189363

theorem cos_alpha_plus_pi_third (α β : Real) 
  (h1 : 0 < α ∧ α < π/2)
  (h2 : 2 * Real.sin β - Real.cos α = 1)
  (h3 : Real.sin α + 2 * Real.cos β = Real.sqrt 3) :
  Real.cos (α + π/3) = -1/4 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_plus_pi_third_l1893_189363


namespace NUMINAMATH_CALUDE_new_shoes_duration_l1893_189338

/-- The duration of new shoes given repair and purchase costs -/
theorem new_shoes_duration (repair_cost : ℝ) (repair_duration : ℝ) (new_cost : ℝ) (cost_increase_percentage : ℝ) :
  repair_cost = 11.50 →
  repair_duration = 1 →
  new_cost = 28.00 →
  cost_increase_percentage = 0.2173913043478261 →
  ∃ (new_duration : ℝ),
    new_duration = 2 ∧
    (new_cost / new_duration) = (repair_cost / repair_duration) * (1 + cost_increase_percentage) :=
by
  sorry

end NUMINAMATH_CALUDE_new_shoes_duration_l1893_189338


namespace NUMINAMATH_CALUDE_max_value_of_operation_l1893_189335

theorem max_value_of_operation : ∃ (n : ℤ), 
  10 ≤ n ∧ n ≤ 99 ∧ 
  (250 - 3*n)^2 = 4 ∧
  ∀ (m : ℤ), 10 ≤ m ∧ m ≤ 99 → (250 - 3*m)^2 ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_operation_l1893_189335


namespace NUMINAMATH_CALUDE_village_population_l1893_189385

/-- If 40% of a population is 23040, then the total population is 57600. -/
theorem village_population (population : ℕ) : (40 : ℕ) * population / 100 = 23040 → population = 57600 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l1893_189385


namespace NUMINAMATH_CALUDE_between_negative_two_and_zero_l1893_189314

def numbers : Set ℝ := {3, 1, -3, -1}

theorem between_negative_two_and_zero :
  ∃ x ∈ numbers, -2 < x ∧ x < 0 :=
by
  sorry

end NUMINAMATH_CALUDE_between_negative_two_and_zero_l1893_189314


namespace NUMINAMATH_CALUDE_problem_solution_l1893_189319

theorem problem_solution : (1 / ((-8^2)^4)) * (-8)^9 = -8 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l1893_189319


namespace NUMINAMATH_CALUDE_play_attendance_l1893_189349

theorem play_attendance (total_people : ℕ) (adult_price child_price : ℚ) (total_receipts : ℚ) :
  total_people = 610 →
  adult_price = 2 →
  child_price = 1 →
  total_receipts = 960 →
  ∃ (adults children : ℕ),
    adults + children = total_people ∧
    adult_price * adults + child_price * children = total_receipts ∧
    children = 260 :=
by sorry

end NUMINAMATH_CALUDE_play_attendance_l1893_189349


namespace NUMINAMATH_CALUDE_solve_equation_l1893_189344

theorem solve_equation (n : ℚ) : 
  (1/(n+2)) + (3/(n+2)) + (2*n/(n+2)) = 4 → n = -2 := by
sorry

end NUMINAMATH_CALUDE_solve_equation_l1893_189344


namespace NUMINAMATH_CALUDE_charity_event_arrangements_l1893_189329

/-- The number of ways to arrange volunteers for a 3-day charity event -/
def charity_arrangements (total_volunteers : ℕ) (day1_needed : ℕ) (day2_needed : ℕ) (day3_needed : ℕ) : ℕ :=
  Nat.choose total_volunteers day1_needed *
  Nat.choose (total_volunteers - day1_needed) day2_needed *
  Nat.choose (total_volunteers - day1_needed - day2_needed) day3_needed

/-- Theorem stating that the number of arrangements for the given conditions is 60 -/
theorem charity_event_arrangements :
  charity_arrangements 6 1 2 3 = 60 := by
  sorry

end NUMINAMATH_CALUDE_charity_event_arrangements_l1893_189329
