import Mathlib

namespace NUMINAMATH_CALUDE_remainder_product_mod_twelve_l196_19605

theorem remainder_product_mod_twelve : (1425 * 1427 * 1429) % 12 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_product_mod_twelve_l196_19605


namespace NUMINAMATH_CALUDE_smaller_number_proof_l196_19652

theorem smaller_number_proof (x y : ℝ) 
  (sum_eq : x + y = 18)
  (diff_eq : x - y = 4)
  (prod_eq : x * y = 77) :
  y = 7 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l196_19652


namespace NUMINAMATH_CALUDE_infinite_product_equals_nine_l196_19677

def infinite_product : ℕ → ℝ
  | 0 => 3^(1/2)
  | n + 1 => infinite_product n * (3^(n+1))^(1 / 2^(n+1))

theorem infinite_product_equals_nine :
  ∃ (limit : ℝ), (∀ ε > 0, ∃ N, ∀ n ≥ N, |infinite_product n - limit| < ε) ∧ limit = 9 := by
  sorry

end NUMINAMATH_CALUDE_infinite_product_equals_nine_l196_19677


namespace NUMINAMATH_CALUDE_minimum_value_f_l196_19617

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + x^2

theorem minimum_value_f (a : ℝ) :
  (∀ x ∈ Set.Icc 1 (Real.exp 1), f a x ≥ f a 1 ∧ f a 1 = 1) ∨
  (∀ x ∈ Set.Icc 1 (Real.exp 1), f a x ≥ f a (Real.sqrt (-a/2)) ∧
    f a (Real.sqrt (-a/2)) = a/2 * Real.log (-a/2) - a/2 ∧
    -2*(Real.exp 1)^2 < a ∧ a < -2) ∨
  (∀ x ∈ Set.Icc 1 (Real.exp 1), f a x ≥ f a (Real.exp 1) ∧
    f a (Real.exp 1) = a + (Real.exp 1)^2) := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_f_l196_19617


namespace NUMINAMATH_CALUDE_base5_calculation_l196_19612

/-- Converts a base 5 number to base 10 --/
def base5ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 5 --/
def base10ToBase5 (n : ℕ) : ℕ := sorry

/-- Theorem: 231₅ × 24₅ - 12₅ = 12132₅ in base 5 --/
theorem base5_calculation : 
  base10ToBase5 (base5ToBase10 231 * base5ToBase10 24 - base5ToBase10 12) = 12132 := by sorry

end NUMINAMATH_CALUDE_base5_calculation_l196_19612


namespace NUMINAMATH_CALUDE_pump_out_time_for_specific_basement_l196_19613

/-- Represents the dimensions and flooding of a basement -/
structure Basement :=
  (length : ℝ)
  (width : ℝ)
  (depth_inches : ℝ)

/-- Represents a water pump -/
structure Pump :=
  (rate : ℝ)  -- gallons per minute

/-- Calculates the time required to pump out a flooded basement -/
def pump_out_time (b : Basement) (pumps : List Pump) (cubic_foot_to_gallon : ℝ) : ℝ :=
  sorry

/-- Theorem stating the time required to pump out the specific basement -/
theorem pump_out_time_for_specific_basement :
  let basement := Basement.mk 40 20 24
  let pumps := [Pump.mk 10, Pump.mk 10, Pump.mk 10]
  pump_out_time basement pumps 7.5 = 400 := by
  sorry

end NUMINAMATH_CALUDE_pump_out_time_for_specific_basement_l196_19613


namespace NUMINAMATH_CALUDE_sandcastle_height_difference_l196_19657

theorem sandcastle_height_difference (miki_height sister_height : ℝ) 
  (h1 : miki_height = 0.83)
  (h2 : sister_height = 0.5) :
  miki_height - sister_height = 0.33 := by
sorry

end NUMINAMATH_CALUDE_sandcastle_height_difference_l196_19657


namespace NUMINAMATH_CALUDE_nina_total_spent_l196_19675

/-- The total amount Nina spends on her children's presents -/
def total_spent (toy_price toy_quantity card_price card_quantity shirt_price shirt_quantity : ℕ) : ℕ :=
  toy_price * toy_quantity + card_price * card_quantity + shirt_price * shirt_quantity

/-- Theorem stating that Nina spends $70 in total -/
theorem nina_total_spent :
  total_spent 10 3 5 2 6 5 = 70 := by
  sorry

end NUMINAMATH_CALUDE_nina_total_spent_l196_19675


namespace NUMINAMATH_CALUDE_max_difference_is_61_l196_19631

def digits : List Nat := [2, 4, 5, 8]

def two_digit_number (d1 d2 : Nat) : Nat := 10 * d1 + d2

def valid_two_digit_number (n : Nat) : Prop :=
  ∃ d1 d2, d1 ∈ digits ∧ d2 ∈ digits ∧ d1 ≠ d2 ∧ n = two_digit_number d1 d2

theorem max_difference_is_61 :
  ∃ a b, valid_two_digit_number a ∧ valid_two_digit_number b ∧
    (∀ x y, valid_two_digit_number x → valid_two_digit_number y →
      x - y ≤ a - b) ∧
    a - b = 61 := by sorry

end NUMINAMATH_CALUDE_max_difference_is_61_l196_19631


namespace NUMINAMATH_CALUDE_tangent_sine_equality_l196_19609

open Real

theorem tangent_sine_equality (α : ℝ) :
  (∃ k : ℤ, -π/2 + 2*π*(k : ℝ) < α ∧ α < π/2 + 2*π*(k : ℝ)) ↔
  Real.sqrt ((tan α)^2 - (sin α)^2) = tan α * sin α :=
sorry

end NUMINAMATH_CALUDE_tangent_sine_equality_l196_19609


namespace NUMINAMATH_CALUDE_parallelogram_points_l196_19664

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if four points form a parallelogram -/
def is_parallelogram (a b c d : Point) : Prop :=
  (b.x - a.x = d.x - c.x ∧ b.y - a.y = d.y - c.y) ∨
  (c.x - a.x = d.x - b.x ∧ c.y - a.y = d.y - b.y) ∨
  (b.x - a.x = c.x - d.x ∧ b.y - a.y = c.y - d.y)

/-- The main theorem -/
theorem parallelogram_points :
  let a : Point := ⟨3, 7⟩
  let b : Point := ⟨4, 6⟩
  let c : Point := ⟨1, -2⟩
  ∀ d : Point, is_parallelogram a b c d ↔ d = ⟨0, -1⟩ ∨ d = ⟨2, -3⟩ ∨ d = ⟨6, 15⟩ :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_points_l196_19664


namespace NUMINAMATH_CALUDE_enchilada_cost_l196_19644

theorem enchilada_cost (T E : ℝ) 
  (h1 : 2 * T + 3 * E = 7.80)
  (h2 : 3 * T + 5 * E = 12.70) : 
  E = 2.00 := by
sorry

end NUMINAMATH_CALUDE_enchilada_cost_l196_19644


namespace NUMINAMATH_CALUDE_hypotenuse_product_squared_l196_19634

-- Define the triangles
structure RightTriangle where
  leg1 : ℝ
  leg2 : ℝ
  hypotenuse : ℝ
  right_angle : leg1^2 + leg2^2 = hypotenuse^2

-- Define the problem
def triangle_problem (T1 T2 : RightTriangle) : Prop :=
  -- Areas of the triangles
  T1.leg1 * T1.leg2 / 2 = 2 ∧
  T2.leg1 * T2.leg2 / 2 = 3 ∧
  -- Congruent sides
  (T1.leg1 = T2.leg1 ∨ T1.leg1 = T2.leg2) ∧
  (T1.leg2 = T2.leg1 ∨ T1.leg2 = T2.leg2) ∧
  -- Similar triangles
  T1.leg1 / T2.leg1 = T1.leg2 / T2.leg2

-- Theorem statement
theorem hypotenuse_product_squared (T1 T2 : RightTriangle) 
  (h : triangle_problem T1 T2) : 
  (T1.hypotenuse * T2.hypotenuse)^2 = 9216 / 25 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_product_squared_l196_19634


namespace NUMINAMATH_CALUDE_sin_60_abs_5_pi_sqrt2_equality_l196_19604

theorem sin_60_abs_5_pi_sqrt2_equality : 
  2 * Real.sin (π / 3) + |-5| - (π - Real.sqrt 2) ^ 0 = Real.sqrt 3 + 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_60_abs_5_pi_sqrt2_equality_l196_19604


namespace NUMINAMATH_CALUDE_min_values_xy_and_x_plus_2y_l196_19659

theorem min_values_xy_and_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1 / x + 9 / y = 1) : 
  (∀ a b : ℝ, a > 0 → b > 0 → 1 / a + 9 / b = 1 → x * y ≤ a * b) ∧ 
  (∀ a b : ℝ, a > 0 → b > 0 → 1 / a + 9 / b = 1 → x + 2 * y ≤ a + 2 * b) ∧
  x * y = 36 ∧ 
  x + 2 * y = 20 + 6 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_min_values_xy_and_x_plus_2y_l196_19659


namespace NUMINAMATH_CALUDE_inequalities_proof_l196_19620

theorem inequalities_proof :
  (∃ a b c : ℝ, a > b ∧ a * c^2 ≤ b * c^2) ∧
  (∀ a b : ℝ, a > 0 ∧ 0 > b → a * b < a^2) ∧
  (∃ a b : ℝ, a * b = 4 ∧ a + b < 4) ∧
  (∀ a b c d : ℝ, a > b ∧ c > d → a - d > b - c) :=
sorry

end NUMINAMATH_CALUDE_inequalities_proof_l196_19620


namespace NUMINAMATH_CALUDE_sin_sum_less_than_sum_of_sins_l196_19676

theorem sin_sum_less_than_sum_of_sins (x y : Real) :
  x ∈ Set.Icc 0 (Real.pi / 2) →
  y ∈ Set.Icc 0 (Real.pi / 2) →
  Real.sin (x + y) < Real.sin x + Real.sin y :=
by sorry

end NUMINAMATH_CALUDE_sin_sum_less_than_sum_of_sins_l196_19676


namespace NUMINAMATH_CALUDE_permanent_non_technicians_percentage_l196_19632

structure Factory where
  total_workers : ℕ
  technicians : ℕ
  non_technicians : ℕ
  permanent_technicians : ℕ
  temporary_workers : ℕ

def Factory.valid (f : Factory) : Prop :=
  f.technicians + f.non_technicians = f.total_workers ∧
  f.technicians = f.non_technicians ∧
  f.permanent_technicians = f.technicians / 2 ∧
  f.temporary_workers = f.total_workers / 2

theorem permanent_non_technicians_percentage (f : Factory) 
  (h : f.valid) : 
  (f.non_technicians - (f.temporary_workers - f.permanent_technicians)) / f.non_technicians = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_permanent_non_technicians_percentage_l196_19632


namespace NUMINAMATH_CALUDE_complex_equality_implies_a_value_l196_19622

theorem complex_equality_implies_a_value (a : ℝ) : 
  let z : ℂ := (1 + a * Complex.I) / (2 + Complex.I)
  Complex.re z = Complex.im z → a = 3 := by
sorry

end NUMINAMATH_CALUDE_complex_equality_implies_a_value_l196_19622


namespace NUMINAMATH_CALUDE_dance_team_recruitment_l196_19653

theorem dance_team_recruitment (total : ℕ) (track : ℕ) (choir : ℕ) (dance : ℕ) : 
  total = 100 ∧ 
  choir = 2 * track ∧ 
  dance = choir + 10 ∧ 
  total = track + choir + dance → 
  dance = 46 := by
sorry

end NUMINAMATH_CALUDE_dance_team_recruitment_l196_19653


namespace NUMINAMATH_CALUDE_even_painted_faces_count_l196_19699

/-- Represents a rectangular block with given dimensions -/
structure Block where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Counts the number of cubes with an even number of painted faces -/
def countEvenPaintedFaces (b : Block) : ℕ :=
  -- We don't implement the actual counting logic here
  sorry

/-- Theorem stating that a 3x4x2 block has 12 cubes with even number of painted faces -/
theorem even_painted_faces_count (b : Block) 
  (h1 : b.length = 3) 
  (h2 : b.width = 4) 
  (h3 : b.height = 2) : 
  countEvenPaintedFaces b = 12 := by
  sorry

end NUMINAMATH_CALUDE_even_painted_faces_count_l196_19699


namespace NUMINAMATH_CALUDE_average_speed_calculation_l196_19603

theorem average_speed_calculation (total_distance : ℝ) (first_half_distance : ℝ) (second_half_distance : ℝ) 
  (first_half_speed : ℝ) (second_half_speed : ℝ) 
  (h1 : total_distance = 50)
  (h2 : first_half_distance = 25)
  (h3 : second_half_distance = 25)
  (h4 : first_half_speed = 60)
  (h5 : second_half_speed = 30)
  (h6 : total_distance = first_half_distance + second_half_distance) :
  (total_distance) / ((first_half_distance / first_half_speed) + (second_half_distance / second_half_speed)) = 40 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l196_19603


namespace NUMINAMATH_CALUDE_pizza_order_l196_19695

theorem pizza_order (slices_per_pizza : ℕ) (total_slices : ℕ) (num_people : ℕ)
  (h1 : slices_per_pizza = 4)
  (h2 : total_slices = 68)
  (h3 : num_people = 25) :
  total_slices / slices_per_pizza = 17 := by
sorry

end NUMINAMATH_CALUDE_pizza_order_l196_19695


namespace NUMINAMATH_CALUDE_average_marks_combined_classes_l196_19650

theorem average_marks_combined_classes (n1 n2 : ℕ) (avg1 avg2 : ℚ) :
  n1 = 22 →
  n2 = 28 →
  avg1 = 40 →
  avg2 = 60 →
  (n1 * avg1 + n2 * avg2) / (n1 + n2 : ℚ) = 51.2 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_combined_classes_l196_19650


namespace NUMINAMATH_CALUDE_power_function_sum_l196_19680

/-- A function f is a power function if it has the form f(x) = cx^n + k, where c ≠ 0 and n is a real number -/
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ (c k n : ℝ), c ≠ 0 ∧ ∀ x, f x = c * x^n + k

/-- Given that f(x) = ax^(2a+1) - b + 1 is a power function, prove that a + b = 2 -/
theorem power_function_sum (a b : ℝ) 
    (h : isPowerFunction (fun x => a * x^(2*a+1) - b + 1)) : 
  a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_sum_l196_19680


namespace NUMINAMATH_CALUDE_sweets_neither_red_nor_green_l196_19624

theorem sweets_neither_red_nor_green 
  (total : ℕ) (red : ℕ) (green : ℕ) 
  (h1 : total = 285) 
  (h2 : red = 49) 
  (h3 : green = 59) : 
  total - (red + green) = 177 := by
  sorry

end NUMINAMATH_CALUDE_sweets_neither_red_nor_green_l196_19624


namespace NUMINAMATH_CALUDE_triangle_cosine_relation_l196_19671

/-- Given a triangle ABC with angles A, B, C and sides a, b, c opposite to these angles respectively.
    If 2sin²A + 2sin²B = 2sin²(A+B) + 3sinAsinB, then cos C = 3/4. -/
theorem triangle_cosine_relation (A B C a b c : Real) : 
  A + B + C = Real.pi →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  2 * (Real.sin A)^2 + 2 * (Real.sin B)^2 = 2 * (Real.sin (A + B))^2 + 3 * Real.sin A * Real.sin B →
  Real.cos C = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_relation_l196_19671


namespace NUMINAMATH_CALUDE_max_value_and_min_side_l196_19654

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.sin x - m * Real.cos x

theorem max_value_and_min_side (m : ℝ) (A B C : ℝ) (a b c : ℝ) :
  (∀ x, f m x ≤ f m (π/3)) →  -- f achieves maximum at π/3
  f m (A - π/2) = 0 →         -- condition on angle A
  2 * b + c = 3 →             -- condition on sides b and c
  0 < A ∧ A < π →             -- A is a valid angle
  0 < B ∧ B < π →             -- B is a valid angle
  0 < C ∧ C < π →             -- C is a valid angle
  a > 0 ∧ b > 0 ∧ c > 0 →     -- sides are positive
  A + B + C = π →             -- sum of angles in a triangle
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A →  -- cosine rule
  m = -Real.sqrt 3 / 3 ∧ a ≥ 3 * Real.sqrt 21 / 14 :=
by sorry

end NUMINAMATH_CALUDE_max_value_and_min_side_l196_19654


namespace NUMINAMATH_CALUDE_quadrilateral_diagonal_length_l196_19674

theorem quadrilateral_diagonal_length 
  (A B C D O : ℝ × ℝ) 
  (h1 : dist O A = 5)
  (h2 : dist O C = 12)
  (h3 : dist O D = 5)
  (h4 : dist O B = 7)
  (h5 : dist B D = 9) :
  dist A C = 13 := by sorry

end NUMINAMATH_CALUDE_quadrilateral_diagonal_length_l196_19674


namespace NUMINAMATH_CALUDE_f_properties_l196_19682

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log x - m * x + m

theorem f_properties (m : ℝ) :
  (∀ x > 0, f m x ≤ 0) →
  (m = 1 ∧
   ∀ a b, 0 < a → a < b →
     (f m b - f m a) / (b - a) < 1 / (a * (a + 1))) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l196_19682


namespace NUMINAMATH_CALUDE_range_of_a_l196_19689

open Set

/-- The statement p: √(2x-1) ≤ 1 -/
def p (x : ℝ) : Prop := Real.sqrt (2 * x - 1) ≤ 1

/-- The statement q: (x-a)(x-(a+1)) ≤ 0 -/
def q (x a : ℝ) : Prop := (x - a) * (x - (a + 1)) ≤ 0

/-- The set of x satisfying statement p -/
def P : Set ℝ := {x | p x}

/-- The set of x satisfying statement q -/
def Q (a : ℝ) : Set ℝ := {x | q x a}

/-- p is a sufficient but not necessary condition for q -/
def sufficient_not_necessary (a : ℝ) : Prop := P ⊂ Q a ∧ P ≠ Q a

theorem range_of_a : 
  ∀ a : ℝ, sufficient_not_necessary a ↔ a ∈ Icc 0 (1/2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l196_19689


namespace NUMINAMATH_CALUDE_folding_coincidence_implies_rhombus_l196_19656

/-- A quadrilateral on a piece of paper. -/
structure PaperQuadrilateral where
  /-- The four vertices of the quadrilateral -/
  vertices : Fin 4 → ℝ × ℝ

/-- Represents the result of folding a paper quadrilateral along a diagonal -/
def foldAlongDiagonal (q : PaperQuadrilateral) (d : Fin 2) : Prop :=
  -- This is a placeholder for the actual folding operation
  sorry

/-- A quadrilateral is a rhombus if it satisfies certain properties -/
def isRhombus (q : PaperQuadrilateral) : Prop :=
  -- This is a placeholder for the actual definition of a rhombus
  sorry

/-- 
If folding a quadrilateral along both diagonals results in coinciding parts each time, 
then the quadrilateral is a rhombus.
-/
theorem folding_coincidence_implies_rhombus (q : PaperQuadrilateral) :
  (∀ d : Fin 2, foldAlongDiagonal q d) → isRhombus q :=
by
  sorry

end NUMINAMATH_CALUDE_folding_coincidence_implies_rhombus_l196_19656


namespace NUMINAMATH_CALUDE_x_squared_value_l196_19673

theorem x_squared_value (x : ℝ) (hx : x > 0) (h : Real.sin (Real.arctan x) = 1 / x) : 
  x^2 = (1 + Real.sqrt 5) / 2 := by
sorry

end NUMINAMATH_CALUDE_x_squared_value_l196_19673


namespace NUMINAMATH_CALUDE_line_x_axis_intersection_l196_19615

/-- The line equation: 5y + 3x = 15 -/
def line_equation (x y : ℝ) : Prop := 5 * y + 3 * x = 15

/-- A point on the x-axis has y-coordinate equal to 0 -/
def on_x_axis (x y : ℝ) : Prop := y = 0

/-- The intersection point of the line and the x-axis -/
def intersection_point : ℝ × ℝ := (5, 0)

theorem line_x_axis_intersection :
  let (x, y) := intersection_point
  line_equation x y ∧ on_x_axis x y :=
by sorry

end NUMINAMATH_CALUDE_line_x_axis_intersection_l196_19615


namespace NUMINAMATH_CALUDE_division_remainder_problem_l196_19600

theorem division_remainder_problem (L S : ℕ) (h1 : L - S = 1370) (h2 : L = 1626) 
  (h3 : L / S = 6) : L % S = 90 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l196_19600


namespace NUMINAMATH_CALUDE_sequence_theorem_l196_19669

def arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, Real.sqrt (a (n + 1)) - Real.sqrt (a n) = d * n + (Real.sqrt (a 1) - Real.sqrt (a 0))

theorem sequence_theorem (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (∀ n : ℕ, Real.sqrt (a (n + 1)) - Real.sqrt (a n) = 2 * n - 2) →
  a 1 = 1 →
  a 3 = 9 →
  ∀ n : ℕ, a n = (n^2 - 3*n + 3)^2 :=
by sorry

end NUMINAMATH_CALUDE_sequence_theorem_l196_19669


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_properties_l196_19627

/-- Represents a sequence of seven consecutive even numbers -/
structure ConsecutiveEvenNumbers where
  middle : ℤ
  sum : ℤ
  sum_eq : sum = 7 * middle

/-- Properties of the sequence of consecutive even numbers -/
theorem consecutive_even_numbers_properties (seq : ConsecutiveEvenNumbers)
  (h : seq.sum = 686) :
  let smallest := seq.middle - 6
  let median := seq.middle
  let mean := seq.sum / 7
  (smallest = 92) ∧ (median = 98) ∧ (mean = 98) := by
  sorry

#check consecutive_even_numbers_properties

end NUMINAMATH_CALUDE_consecutive_even_numbers_properties_l196_19627


namespace NUMINAMATH_CALUDE_stating_rowing_speed_calculation_l196_19667

/-- Represents the speed of the river current in km/h -/
def stream_speed : ℝ := 12

/-- Represents the man's rowing speed in still water in km/h -/
def rowing_speed : ℝ := 24

/-- 
Theorem stating that if it takes thrice as long to row up as to row down the river,
given the stream speed, then the rowing speed in still water is 24 km/h
-/
theorem rowing_speed_calculation (distance : ℝ) (h : distance > 0) :
  (distance / (rowing_speed - stream_speed)) = 3 * (distance / (rowing_speed + stream_speed)) →
  rowing_speed = 24 := by
sorry

end NUMINAMATH_CALUDE_stating_rowing_speed_calculation_l196_19667


namespace NUMINAMATH_CALUDE_perimeter_of_figure_c_l196_19665

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ :=
  2 * (r.width + r.height)

/-- Represents the large rectangle composed of small rectangles -/
structure LargeRectangle where
  small_rectangle : Rectangle
  total_count : ℕ

/-- Theorem: Given the conditions, the perimeter of figure C is 40 cm -/
theorem perimeter_of_figure_c (large_rect : LargeRectangle)
    (h1 : large_rect.total_count = 20)
    (h2 : Rectangle.perimeter { width := 6 * large_rect.small_rectangle.width,
                                height := large_rect.small_rectangle.height } = 56)
    (h3 : Rectangle.perimeter { width := 2 * large_rect.small_rectangle.width,
                                height := 3 * large_rect.small_rectangle.height } = 56) :
  Rectangle.perimeter { width := large_rect.small_rectangle.width,
                        height := 3 * large_rect.small_rectangle.height } = 40 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_figure_c_l196_19665


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l196_19608

theorem polynomial_evaluation (x : ℝ) (h1 : x > 0) (h2 : x^2 - 3*x - 9 = 0) :
  x^3 - 3*x^2 - 9*x + 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l196_19608


namespace NUMINAMATH_CALUDE_pet_store_cages_l196_19618

theorem pet_store_cages (initial_puppies : ℕ) (sold_puppies : ℕ) (puppies_per_cage : ℕ) 
  (h1 : initial_puppies = 45)
  (h2 : sold_puppies = 39)
  (h3 : puppies_per_cage = 2) :
  (initial_puppies - sold_puppies) / puppies_per_cage = 3 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_cages_l196_19618


namespace NUMINAMATH_CALUDE_parabolas_intersection_l196_19668

/-- The x-coordinates of the intersection points of two parabolas -/
def intersection_x : Set ℝ :=
  {x : ℝ | 3 * x^2 + 6 * x - 4 = x^2 + 2 * x + 1}

/-- The y-coordinates of the intersection points of two parabolas -/
def intersection_y (x : ℝ) : ℝ :=
  x^2 + 2 * x + 1

/-- The set of intersection points of two parabolas -/
def intersection_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 ∈ intersection_x ∧ p.2 = intersection_y p.1}

theorem parabolas_intersection :
  intersection_points = {(-5, 16), (1/2, 9/4)} :=
by sorry

end NUMINAMATH_CALUDE_parabolas_intersection_l196_19668


namespace NUMINAMATH_CALUDE_product_of_roots_l196_19661

/-- The polynomial coefficients -/
def a : ℝ := 2
def b : ℝ := -5
def c : ℝ := -10
def d : ℝ := 22

/-- The polynomial equation -/
def polynomial (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

theorem product_of_roots :
  ∃ (r₁ r₂ r₃ : ℝ), (∀ x, polynomial x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) ∧ r₁ * r₂ * r₃ = -11 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l196_19661


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l196_19633

theorem smallest_dual_base_representation :
  ∃ (a b : ℕ), a > 2 ∧ b > 2 ∧
  (1 * a + 2 * 1 = 7) ∧
  (2 * b + 1 * 1 = 7) ∧
  (∀ (x y : ℕ), x > 2 → y > 2 → 1 * x + 2 * 1 = 2 * y + 1 * 1 → 1 * x + 2 * 1 ≥ 7) :=
by sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l196_19633


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l196_19687

theorem least_subtraction_for_divisibility : ∃! x : ℕ, 
  (∀ y : ℕ, y < x → ¬((50248 - y) % 20 = 0 ∧ (50248 - y) % 37 = 0)) ∧ 
  (50248 - x) % 20 = 0 ∧ 
  (50248 - x) % 37 = 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l196_19687


namespace NUMINAMATH_CALUDE_square_triangle_equal_area_l196_19691

theorem square_triangle_equal_area (square_perimeter : ℝ) (triangle_height : ℝ) (triangle_base : ℝ) :
  square_perimeter = 48 →
  triangle_height = 48 →
  (square_perimeter / 4) ^ 2 = (1 / 2) * triangle_base * triangle_height →
  triangle_base = 6 := by
  sorry

end NUMINAMATH_CALUDE_square_triangle_equal_area_l196_19691


namespace NUMINAMATH_CALUDE_overtake_twice_implies_double_speed_l196_19663

/-- Represents a runner with a constant speed -/
structure Runner where
  speed : ℝ
  speed_pos : speed > 0

/-- Represents the race course -/
structure Course where
  distance_to_stadium : ℝ
  lap_length : ℝ
  total_laps : ℕ
  distance_to_stadium_pos : distance_to_stadium > 0
  lap_length_pos : lap_length > 0
  total_laps_pos : total_laps > 0

/-- Theorem: If a runner overtakes another runner twice in a race with three laps,
    then the faster runner's speed is at least twice the slower runner's speed -/
theorem overtake_twice_implies_double_speed
  (runner1 runner2 : Runner) (course : Course) :
  course.total_laps = 3 →
  (∃ (t1 t2 : ℝ), 0 < t1 ∧ t1 < t2 ∧
    runner1.speed * t1 = runner2.speed * t1 + course.lap_length ∧
    runner1.speed * t2 = runner2.speed * t2 + 2 * course.lap_length) →
  runner1.speed ≥ 2 * runner2.speed :=
by sorry

end NUMINAMATH_CALUDE_overtake_twice_implies_double_speed_l196_19663


namespace NUMINAMATH_CALUDE_sandy_work_hours_l196_19606

/-- Sandy's work schedule -/
structure WorkSchedule where
  total_hours : ℕ
  num_days : ℕ
  hours_per_day : ℕ
  equal_hours : total_hours = num_days * hours_per_day

/-- Theorem: Sandy worked 9 hours per day -/
theorem sandy_work_hours (schedule : WorkSchedule)
  (h1 : schedule.total_hours = 45)
  (h2 : schedule.num_days = 5) :
  schedule.hours_per_day = 9 := by
  sorry

end NUMINAMATH_CALUDE_sandy_work_hours_l196_19606


namespace NUMINAMATH_CALUDE_sarah_finished_problems_l196_19655

/-- Calculates the number of problems Sarah finished given the initial number of problems,
    remaining pages, and problems per page. -/
def problems_finished (initial_problems : ℕ) (remaining_pages : ℕ) (problems_per_page : ℕ) : ℕ :=
  initial_problems - (remaining_pages * problems_per_page)

/-- Proves that Sarah finished 20 problems given the initial conditions. -/
theorem sarah_finished_problems :
  problems_finished 60 5 8 = 20 := by
  sorry

end NUMINAMATH_CALUDE_sarah_finished_problems_l196_19655


namespace NUMINAMATH_CALUDE_minimum_distance_point_l196_19666

/-- The point that minimizes the sum of distances to two fixed points lies on the line connecting those points -/
theorem minimum_distance_point (P Q R : ℝ × ℝ) :
  P.1 = -2 ∧ P.2 = -3 ∧
  Q.1 = 5 ∧ Q.2 = 3 ∧
  R.1 = 2 →
  (∀ m : ℝ, (Real.sqrt ((R.1 - P.1)^2 + (R.2 - P.2)^2) + 
              Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2)) ≥
             (Real.sqrt ((R.1 - P.1)^2 + ((3/7) - P.2)^2) + 
              Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - (3/7))^2))) →
  R.2 = 3/7 := by
  sorry

end NUMINAMATH_CALUDE_minimum_distance_point_l196_19666


namespace NUMINAMATH_CALUDE_house_number_painting_cost_l196_19616

/-- Represents a side of the street with houses -/
structure StreetSide where
  start : ℕ
  diff : ℕ
  count : ℕ

/-- Calculates the cost of painting numbers for a given street side -/
def paintCost (side : StreetSide) : ℕ := sorry

/-- The problem statement -/
theorem house_number_painting_cost :
  let southSide : StreetSide := { start := 5, diff := 7, count := 25 }
  let northSide : StreetSide := { start := 2, diff := 8, count := 25 }
  paintCost southSide + paintCost northSide = 123 := by sorry

end NUMINAMATH_CALUDE_house_number_painting_cost_l196_19616


namespace NUMINAMATH_CALUDE_original_number_proof_l196_19645

theorem original_number_proof (h : 204 / 12.75 = 16) : 
  ∃ x : ℝ, x / 1.275 = 1.6 ∧ x = 2.04 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l196_19645


namespace NUMINAMATH_CALUDE_square_sum_equals_product_implies_zero_l196_19662

theorem square_sum_equals_product_implies_zero (x y z : ℤ) :
  x^2 + y^2 + z^2 = 2*x*y*z → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_product_implies_zero_l196_19662


namespace NUMINAMATH_CALUDE_multiply_to_all_ones_l196_19694

theorem multiply_to_all_ones : 
  ∃ (A : ℕ) (n : ℕ), (10^9 - 1) * A = (10^n - 1) / 9 :=
sorry

end NUMINAMATH_CALUDE_multiply_to_all_ones_l196_19694


namespace NUMINAMATH_CALUDE_quadratic_inequality_l196_19651

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 2

theorem quadratic_inequality (a : ℝ) :
  (∀ x ∈ Set.Icc 1 5, f a x > 3*a*x) ↔ a < 2*Real.sqrt 2 ∧
  (∀ x : ℝ, (a + 1)*x^2 + x > f a x ↔
    (a = 0 ∧ x > 2) ∨
    (a > 0 ∧ (x < -1/a ∨ x > 2)) ∨
    (-1/2 < a ∧ a < 0 ∧ 2 < x ∧ x < -1/a) ∨
    (a < -1/2 ∧ -1/a < x ∧ x < 2)) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l196_19651


namespace NUMINAMATH_CALUDE_frank_reading_time_l196_19610

/-- Calculates the number of days needed to read a book -/
def days_to_read (total_pages : ℕ) (pages_per_day : ℕ) : ℕ :=
  total_pages / pages_per_day

/-- Proves that it takes 569 days to read a book with 12518 pages at 22 pages per day -/
theorem frank_reading_time : days_to_read 12518 22 = 569 := by
  sorry

end NUMINAMATH_CALUDE_frank_reading_time_l196_19610


namespace NUMINAMATH_CALUDE_inscribed_rectangle_max_area_l196_19648

theorem inscribed_rectangle_max_area :
  ∀ (x : ℝ) (r l b : ℝ),
  x > 0 ∧
  x^2 - 25*x + 144 = 0 ∧
  r^2 = x ∧
  l = (2/5) * r ∧
  ∃ (ratio : ℝ), ratio^2 - 3*ratio - 10 = 0 ∧ ratio > 0 ∧ l / b = ratio →
  l * b ≤ 0.512 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_max_area_l196_19648


namespace NUMINAMATH_CALUDE_sum_percentage_l196_19635

theorem sum_percentage (A B : ℝ) : 
  (0.4 * A = 160) → 
  (160 = (2/3) * B) → 
  (0.6 * (A + B) = 384) := by
sorry

end NUMINAMATH_CALUDE_sum_percentage_l196_19635


namespace NUMINAMATH_CALUDE_prime_power_divisors_l196_19696

theorem prime_power_divisors (p q : ℕ) (x : ℕ) (hp : Prime p) (hq : Prime q) :
  (∀ d : ℕ, d ∣ p^4 * q^x ↔ d ∈ Finset.range 51) → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_prime_power_divisors_l196_19696


namespace NUMINAMATH_CALUDE_fraction_value_implies_x_l196_19647

theorem fraction_value_implies_x (x : ℝ) : 2 / (x - 3) = 2 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_implies_x_l196_19647


namespace NUMINAMATH_CALUDE_cube_sum_of_symmetric_polynomials_l196_19646

theorem cube_sum_of_symmetric_polynomials (a b c : ℝ) 
  (h1 : a + b + c = 2) 
  (h2 : a * b + a * c + b * c = -3) 
  (h3 : a * b * c = 9) : 
  a^3 + b^3 + c^3 = 22 := by sorry

end NUMINAMATH_CALUDE_cube_sum_of_symmetric_polynomials_l196_19646


namespace NUMINAMATH_CALUDE_melissa_bananas_l196_19692

/-- Calculates the remaining bananas after sharing -/
def remaining_bananas (initial : ℕ) (shared : ℕ) : ℕ :=
  initial - shared

theorem melissa_bananas : remaining_bananas 88 4 = 84 := by
  sorry

end NUMINAMATH_CALUDE_melissa_bananas_l196_19692


namespace NUMINAMATH_CALUDE_triangle_problem_l196_19672

theorem triangle_problem (A B C : Real) (a b c : Real) :
  -- Triangle conditions
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Side lengths are positive
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Law of sines
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  -- Law of cosines
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A →
  b^2 = a^2 + c^2 - 2*a*c*Real.cos B →
  c^2 = a^2 + b^2 - 2*a*b*Real.cos C →
  -- Given conditions
  a = 2 * Real.sqrt 6 →
  b = 3 →
  Real.sin (B + C)^2 + Real.sqrt 2 * Real.sin (2 * A) = 0 →
  -- Conclusion
  c = 3 ∧ Real.cos B = Real.sqrt 6 / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l196_19672


namespace NUMINAMATH_CALUDE_events_complementary_l196_19611

-- Define the sample space for a fair die
def DieOutcome := Fin 6

-- Define Event 1: odd numbers
def Event1 (outcome : DieOutcome) : Prop :=
  outcome.val % 2 = 1

-- Define Event 2: even numbers
def Event2 (outcome : DieOutcome) : Prop :=
  outcome.val % 2 = 0

-- Theorem stating that Event1 and Event2 are complementary
theorem events_complementary :
  ∀ (outcome : DieOutcome), Event1 outcome ↔ ¬Event2 outcome :=
sorry

end NUMINAMATH_CALUDE_events_complementary_l196_19611


namespace NUMINAMATH_CALUDE_f_is_increasing_l196_19684

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 6*x

-- Theorem statement
theorem f_is_increasing : ∀ x : ℝ, Monotone f := by sorry

end NUMINAMATH_CALUDE_f_is_increasing_l196_19684


namespace NUMINAMATH_CALUDE_james_fish_catch_l196_19642

/-- The total weight of fish James caught -/
def total_fish_weight (trout salmon tuna bass catfish : ℝ) : ℝ :=
  trout + salmon + tuna + bass + catfish

/-- Theorem stating the total weight of fish James caught -/
theorem james_fish_catch :
  ∃ (trout salmon tuna bass catfish : ℝ),
    trout = 200 ∧
    salmon = trout * 1.6 ∧
    tuna = trout * 2 ∧
    bass = salmon * 3 ∧
    catfish = tuna / 3 ∧
    total_fish_weight trout salmon tuna bass catfish = 2013.33 :=
by
  sorry

end NUMINAMATH_CALUDE_james_fish_catch_l196_19642


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l196_19693

theorem sin_2alpha_value (α : Real) 
  (h1 : α > -π/2 ∧ α < 0) 
  (h2 : Real.tan (π/4 - α) = 3 * Real.cos (2 * α)) : 
  Real.sin (2 * α) = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l196_19693


namespace NUMINAMATH_CALUDE_mike_picked_52_peaches_l196_19628

/-- The number of peaches Mike initially had -/
def initial_peaches : ℕ := 34

/-- The total number of peaches Mike has after picking more -/
def total_peaches : ℕ := 86

/-- The number of peaches Mike picked -/
def picked_peaches : ℕ := total_peaches - initial_peaches

theorem mike_picked_52_peaches : picked_peaches = 52 := by
  sorry

end NUMINAMATH_CALUDE_mike_picked_52_peaches_l196_19628


namespace NUMINAMATH_CALUDE_equation_solution_l196_19698

theorem equation_solution : 
  ∃ x : ℝ, (6 * x^2 + 111 * x + 1) / (2 * x + 37) = 3 * x + 1 ∧ x = -18 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l196_19698


namespace NUMINAMATH_CALUDE_solution_exists_l196_19636

theorem solution_exists : ∃ (x y z : ℝ), 
  (15 + (1/4) * x = 27) ∧ 
  ((1/2) * x - y^2 = 37) ∧ 
  (y^3 + z = 50) ∧ 
  (x = 48) ∧ 
  ((y = Real.sqrt 13 ∧ z = 50 - 13 * Real.sqrt 13) ∨ 
   (y = -Real.sqrt 13 ∧ z = 50 + 13 * Real.sqrt 13)) := by
  sorry

end NUMINAMATH_CALUDE_solution_exists_l196_19636


namespace NUMINAMATH_CALUDE_optionC_is_most_suitable_l196_19621

structure SamplingMethod where
  method : String
  representativeOfAllStudents : Bool
  includesAllGrades : Bool
  unbiased : Bool

def cityJuniorHighSchools : Set String := sorry

def isMostSuitableSamplingMethod (m : SamplingMethod) : Prop :=
  m.representativeOfAllStudents ∧ m.includesAllGrades ∧ m.unbiased

def optionC : SamplingMethod := {
  method := "Randomly select 1000 students from each of the three grades in junior high schools in the city",
  representativeOfAllStudents := true,
  includesAllGrades := true,
  unbiased := true
}

theorem optionC_is_most_suitable :
  isMostSuitableSamplingMethod optionC :=
sorry

end NUMINAMATH_CALUDE_optionC_is_most_suitable_l196_19621


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l196_19690

theorem decimal_to_fraction : 
  (2.36 : ℚ) = 59 / 25 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l196_19690


namespace NUMINAMATH_CALUDE_fraction_problem_l196_19670

theorem fraction_problem (n d : ℚ) : 
  n / (d + 1) = 1 / 2 → (n + 1) / d = 1 → n / d = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l196_19670


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l196_19637

theorem nested_fraction_evaluation : 
  (1 : ℚ) / (2 + 1 / (3 + 1 / 4)) = 13 / 30 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l196_19637


namespace NUMINAMATH_CALUDE_rita_backstroke_hours_l196_19630

/-- Calculates the number of backstroke hours completed by Rita --/
def backstroke_hours (total_required : ℕ) (breaststroke : ℕ) (butterfly : ℕ) 
  (freestyle_sidestroke_per_month : ℕ) (months : ℕ) : ℕ :=
  total_required - (breaststroke + butterfly + freestyle_sidestroke_per_month * months)

/-- Theorem stating that Rita completed 50 hours of backstroke --/
theorem rita_backstroke_hours : 
  backstroke_hours 1500 9 121 220 6 = 50 := by
  sorry

end NUMINAMATH_CALUDE_rita_backstroke_hours_l196_19630


namespace NUMINAMATH_CALUDE_right_triangles_common_hypotenuse_l196_19639

theorem right_triangles_common_hypotenuse (AC AD CD : ℝ) (hAC : AC = 16) (hAD : AD = 32) (hCD : CD = 14) :
  let AB := Real.sqrt (AD^2 - (AC + CD)^2)
  let BC := Real.sqrt (AB^2 + AC^2)
  BC = Real.sqrt 380 := by
sorry

end NUMINAMATH_CALUDE_right_triangles_common_hypotenuse_l196_19639


namespace NUMINAMATH_CALUDE_quadratic_properties_l196_19683

-- Define the quadratic function
def f (x : ℝ) : ℝ := (x - 2)^2 + 6

-- State the theorem
theorem quadratic_properties :
  (∀ x y : ℝ, x < y → f x > f y → f y > f ((y - x) + y)) ∧ 
  (∀ x : ℝ, f (2 + x) = f (2 - x)) ∧
  (f 0 = 10) :=
sorry

end NUMINAMATH_CALUDE_quadratic_properties_l196_19683


namespace NUMINAMATH_CALUDE_estimation_greater_than_exact_l196_19638

theorem estimation_greater_than_exact 
  (a b d : ℕ+) 
  (a' b' d' : ℝ)
  (h_a : a' > a ∧ a' < a + 1)
  (h_b : b' < b ∧ b' > b - 1)
  (h_d : d' < d ∧ d' > d - 1) :
  Real.sqrt (a' / b') - Real.sqrt d' > Real.sqrt (a / b) - Real.sqrt d :=
sorry

end NUMINAMATH_CALUDE_estimation_greater_than_exact_l196_19638


namespace NUMINAMATH_CALUDE_intersection_segment_length_l196_19660

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/25 + y^2/16 = 1

-- Define the right focus of the ellipse
def right_focus : ℝ × ℝ := (3, 0)

-- Define the line perpendicular to x-axis passing through the right focus
def perpendicular_line (x y : ℝ) : Prop := x = 3

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ellipse p.1 p.2 ∧ perpendicular_line p.1 p.2}

-- Statement to prove
theorem intersection_segment_length :
  let A := (3, 16/5)
  let B := (3, -16/5)
  (A ∈ intersection_points) ∧ 
  (B ∈ intersection_points) ∧
  (∀ p ∈ intersection_points, p = A ∨ p = B) ∧
  (dist A B = 32/5) := by sorry


end NUMINAMATH_CALUDE_intersection_segment_length_l196_19660


namespace NUMINAMATH_CALUDE_x_less_neg_one_sufficient_not_necessary_for_abs_x_greater_x_l196_19697

theorem x_less_neg_one_sufficient_not_necessary_for_abs_x_greater_x :
  (∃ x : ℝ, x < -1 → abs x > x) ∧ 
  (∃ x : ℝ, abs x > x ∧ x ≥ -1) :=
by sorry

end NUMINAMATH_CALUDE_x_less_neg_one_sufficient_not_necessary_for_abs_x_greater_x_l196_19697


namespace NUMINAMATH_CALUDE_triangle_expression_negative_l196_19626

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0
  triangle_inequality_ab : a + b > c
  triangle_inequality_bc : b + c > a
  triangle_inequality_ca : c + a > b

-- Theorem statement
theorem triangle_expression_negative (t : Triangle) : (t.a - t.c)^2 - t.b^2 < 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_expression_negative_l196_19626


namespace NUMINAMATH_CALUDE_intersection_property_l196_19686

noncomputable section

-- Define the line l
def line_l (α t : ℝ) : ℝ × ℝ := (2 + t * Real.cos α, 1 + t * Real.sin α)

-- Define the curve C in polar coordinates
def curve_C_polar (θ : ℝ) : ℝ := 4 * Real.cos θ

-- Define the curve C in Cartesian coordinates
def curve_C_cartesian (x y : ℝ) : Prop := x^2 + y^2 = 4*x

-- Define the point P
def point_P : ℝ × ℝ := (2, 1)

-- Theorem statement
theorem intersection_property (α : ℝ) (h_α : 0 ≤ α ∧ α < Real.pi) :
  ∃ t₁ t₂ : ℝ, 
    let A := line_l α t₁
    let B := line_l α t₂
    let P := point_P
    curve_C_cartesian A.1 A.2 ∧ 
    curve_C_cartesian B.1 B.2 ∧
    (A.1 - P.1, A.2 - P.2) = 2 • (P.1 - B.1, P.2 - B.2) →
    Real.tan α = Real.sqrt (3/5) ∨ Real.tan α = -Real.sqrt (3/5) := by
  sorry

end

end NUMINAMATH_CALUDE_intersection_property_l196_19686


namespace NUMINAMATH_CALUDE_cost_of_tax_free_items_l196_19619

/-- Given a total cost, sales tax, and tax rate, calculate the cost of tax-free items -/
theorem cost_of_tax_free_items
  (total_cost : ℝ)
  (sales_tax : ℝ)
  (tax_rate : ℝ)
  (h1 : total_cost = 25)
  (h2 : sales_tax = 0.30)
  (h3 : tax_rate = 0.05)
  : ∃ (tax_free_cost : ℝ), tax_free_cost = 19 :=
by sorry

end NUMINAMATH_CALUDE_cost_of_tax_free_items_l196_19619


namespace NUMINAMATH_CALUDE_deepak_age_l196_19625

theorem deepak_age (rahul_age deepak_age : ℕ) : 
  (rahul_age : ℚ) / deepak_age = 4 / 3 →
  rahul_age + 6 = 18 →
  deepak_age = 9 := by
sorry

end NUMINAMATH_CALUDE_deepak_age_l196_19625


namespace NUMINAMATH_CALUDE_train_crossing_time_l196_19658

/-- Proves that a train with given length and speed takes a specific time to cross an electric pole -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) (crossing_time : Real) :
  train_length = 1500 ∧ 
  train_speed_kmh = 108 →
  crossing_time = 50 :=
by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l196_19658


namespace NUMINAMATH_CALUDE_school_workbook_cost_l196_19641

/-- The total cost for purchasing workbooks -/
def total_cost (num_workbooks : ℕ) (cost_per_workbook : ℚ) : ℚ :=
  num_workbooks * cost_per_workbook

/-- Theorem: The total cost for the school to purchase 400 workbooks, each costing x yuan, is equal to 400x yuan -/
theorem school_workbook_cost (x : ℚ) : 
  total_cost 400 x = 400 * x := by
  sorry

end NUMINAMATH_CALUDE_school_workbook_cost_l196_19641


namespace NUMINAMATH_CALUDE_Ca_concentration_after_mixing_l196_19629

-- Define the constants
def K_sp : ℝ := 4.96e-9
def c_Na2CO3 : ℝ := 0.40
def c_CaCl2 : ℝ := 0.20

-- Define the theorem
theorem Ca_concentration_after_mixing :
  let c_CO3_remaining : ℝ := (c_Na2CO3 - c_CaCl2) / 2
  let c_Ca : ℝ := K_sp / c_CO3_remaining
  c_Ca = 4.96e-8 := by sorry

end NUMINAMATH_CALUDE_Ca_concentration_after_mixing_l196_19629


namespace NUMINAMATH_CALUDE_teaching_team_formation_l196_19640

def chinese_teachers : ℕ := 2
def math_teachers : ℕ := 2
def english_teachers : ℕ := 4
def team_size : ℕ := 5

def ways_to_form_team : ℕ := 
  Nat.choose english_teachers 1 + 
  (Nat.choose chinese_teachers 1 * Nat.choose english_teachers 2) +
  (Nat.choose math_teachers 1 * Nat.choose english_teachers 2) +
  (Nat.choose chinese_teachers 1 * Nat.choose math_teachers 1 * Nat.choose english_teachers 3)

theorem teaching_team_formation :
  ways_to_form_team = 44 :=
by sorry

end NUMINAMATH_CALUDE_teaching_team_formation_l196_19640


namespace NUMINAMATH_CALUDE_exists_a_with_full_domain_and_range_l196_19678

/-- Given a real number a, f is a function from ℝ to ℝ defined as f(x) = ax^2 + x + 1 -/
def f (a : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + x + 1

/-- Theorem stating that there exists a real number a such that f(a) has domain and range ℝ -/
theorem exists_a_with_full_domain_and_range :
  ∃ a : ℝ, Function.Surjective (f a) ∧ Function.Injective (f a) := by
  sorry

end NUMINAMATH_CALUDE_exists_a_with_full_domain_and_range_l196_19678


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l196_19614

theorem rectangle_area_increase (x y : ℝ) 
  (area_eq : x * y = 180)
  (perimeter_eq : 2 * x + 2 * y = 54) :
  (x + 6) * (y + 6) = 378 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_l196_19614


namespace NUMINAMATH_CALUDE_fraction_problem_l196_19679

theorem fraction_problem :
  ∃ (x y : ℚ), x / y = 2 / 3 ∧ (x / y) * 6 + 6 = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l196_19679


namespace NUMINAMATH_CALUDE_king_game_winner_l196_19602

/-- Represents the result of the game -/
inductive GameResult
  | PlayerAWins
  | PlayerBWins

/-- Represents a chessboard of size m × n -/
structure Chessboard where
  m : Nat
  n : Nat

/-- Determines the winner of the game based on the chessboard size -/
def determineWinner (board : Chessboard) : GameResult :=
  if board.m * board.n % 2 == 0 then
    GameResult.PlayerAWins
  else
    GameResult.PlayerBWins

/-- Theorem stating the winning condition for the game -/
theorem king_game_winner (board : Chessboard) :
  determineWinner board = GameResult.PlayerAWins ↔ board.m * board.n % 2 == 0 := by
  sorry

end NUMINAMATH_CALUDE_king_game_winner_l196_19602


namespace NUMINAMATH_CALUDE_probability_theorem_l196_19685

/-- The number of days the performance lasts -/
def total_days : ℕ := 8

/-- The number of consecutive days Resident A watches -/
def watch_days : ℕ := 3

/-- The number of days we're interested in (first to fourth day) -/
def interest_days : ℕ := 4

/-- The total number of ways to choose 3 consecutive days out of 8 days -/
def total_choices : ℕ := total_days - watch_days + 1

/-- The number of ways to choose 3 consecutive days within the first 4 days -/
def interest_choices : ℕ := interest_days - watch_days + 1

/-- The probability of choosing 3 consecutive days within the first 4 days out of 8 total days -/
theorem probability_theorem : 
  (interest_choices : ℚ) / total_choices = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_probability_theorem_l196_19685


namespace NUMINAMATH_CALUDE_periodic_function_roots_l196_19688

theorem periodic_function_roots (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f (2 + x) = f (2 - x))
  (h2 : ∀ x : ℝ, f (7 + x) = f (7 - x))
  (h3 : f 0 = 0) :
  ∃ (roots : Finset ℝ), (∀ x ∈ roots, f x = 0 ∧ x ∈ Set.Icc (-1000) 1000) ∧ roots.card ≥ 201 :=
sorry

end NUMINAMATH_CALUDE_periodic_function_roots_l196_19688


namespace NUMINAMATH_CALUDE_chocolate_bars_distribution_l196_19643

theorem chocolate_bars_distribution (total_bars : ℕ) (num_people : ℕ) (num_combined : ℕ) :
  total_bars = 12 →
  num_people = 3 →
  num_combined = 2 →
  (total_bars / num_people) * num_combined = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_distribution_l196_19643


namespace NUMINAMATH_CALUDE_regular_polygon_angle_characterization_l196_19601

def is_regular_polygon_angle (angle : ℕ) : Prop :=
  ∃ n : ℕ, n ≥ 3 ∧ angle = 180 - 360 / n

def regular_polygon_angles : Set ℕ :=
  {60, 90, 108, 120, 135, 140, 144, 150, 156, 160, 162, 165, 168, 170, 171, 172, 174, 175, 176, 177, 178, 179}

theorem regular_polygon_angle_characterization :
  ∀ angle : ℕ, is_regular_polygon_angle angle ↔ angle ∈ regular_polygon_angles :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_angle_characterization_l196_19601


namespace NUMINAMATH_CALUDE_rectangular_field_length_l196_19623

/-- Proves the length of a rectangular field given specific conditions -/
theorem rectangular_field_length : ∀ w : ℝ,
  w > 0 →  -- width is positive
  (w + 10) * w = 171 →  -- area equation
  w + 10 = 19 :=  -- length equation
by
  sorry

#check rectangular_field_length

end NUMINAMATH_CALUDE_rectangular_field_length_l196_19623


namespace NUMINAMATH_CALUDE_grasshopper_jump_distance_l196_19607

theorem grasshopper_jump_distance (frog_jump : ℕ) (difference : ℕ) : 
  frog_jump = 40 → difference = 15 → frog_jump - difference = 25 := by
  sorry

end NUMINAMATH_CALUDE_grasshopper_jump_distance_l196_19607


namespace NUMINAMATH_CALUDE_mikey_has_56_jelly_beans_l196_19681

def napoleon_jelly_beans : ℕ := 34

def sedrich_jelly_beans (napoleon : ℕ) : ℕ := napoleon + 7

def daphne_jelly_beans (sedrich : ℕ) : ℕ := sedrich - 4

def mikey_jelly_beans (napoleon sedrich daphne : ℕ) : ℕ :=
  (3 * (napoleon + sedrich + daphne)) / 6

theorem mikey_has_56_jelly_beans :
  mikey_jelly_beans napoleon_jelly_beans 
    (sedrich_jelly_beans napoleon_jelly_beans) 
    (daphne_jelly_beans (sedrich_jelly_beans napoleon_jelly_beans)) = 56 := by
  sorry

end NUMINAMATH_CALUDE_mikey_has_56_jelly_beans_l196_19681


namespace NUMINAMATH_CALUDE_typists_calculation_l196_19649

/-- The number of typists in the initial group -/
def initial_typists : ℕ := 10

/-- The number of letters typed by the initial group in 20 minutes -/
def initial_letters : ℕ := 20

/-- The time taken by the initial group to type the initial letters (in minutes) -/
def initial_time : ℕ := 20

/-- The number of typists in the second group -/
def second_typists : ℕ := 40

/-- The number of letters typed by the second group in 1 hour -/
def second_letters : ℕ := 240

/-- The time taken by the second group to type the second letters (in minutes) -/
def second_time : ℕ := 60

theorem typists_calculation :
  initial_typists * second_typists * second_time * initial_letters =
  initial_time * second_typists * second_letters * initial_typists :=
by sorry

end NUMINAMATH_CALUDE_typists_calculation_l196_19649
