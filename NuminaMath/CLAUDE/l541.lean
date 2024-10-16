import Mathlib

namespace NUMINAMATH_CALUDE_total_pieces_is_11403_l541_54124

/-- Calculates the total number of pieces in John's puzzles -/
def totalPuzzlePieces : ℕ :=
  let puzzle1 : ℕ := 1000
  let puzzle2 : ℕ := puzzle1 + (puzzle1 * 20 / 100)
  let puzzle3 : ℕ := puzzle2 + (puzzle2 * 50 / 100)
  let puzzle4 : ℕ := puzzle3 + (puzzle3 * 75 / 100)
  let puzzle5 : ℕ := puzzle4 + (puzzle4 * 35 / 100)
  puzzle1 + puzzle2 + puzzle3 + puzzle4 + puzzle5

theorem total_pieces_is_11403 : totalPuzzlePieces = 11403 := by
  sorry

end NUMINAMATH_CALUDE_total_pieces_is_11403_l541_54124


namespace NUMINAMATH_CALUDE_equation_solution_difference_l541_54119

theorem equation_solution_difference : ∃ (r s : ℝ),
  (∀ x, (6 * x - 18) / (x^2 + 4 * x - 21) = x + 3 ↔ (x = r ∨ x = s)) ∧
  r > s ∧
  r - s = 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_difference_l541_54119


namespace NUMINAMATH_CALUDE_square_of_1031_l541_54125

theorem square_of_1031 : (1031 : ℕ)^2 = 1062961 := by
  sorry

end NUMINAMATH_CALUDE_square_of_1031_l541_54125


namespace NUMINAMATH_CALUDE_point_movement_power_l541_54135

/-- 
Given a point (-1, 1) in the Cartesian coordinate system,
if it is moved up 1 unit and then left 2 units to reach a point (x, y),
then x^y = 9.
-/
theorem point_movement_power (x y : ℝ) : 
  ((-1 : ℝ) + -2 = x) → ((1 : ℝ) + 1 = y) → x^y = 9 := by
  sorry

end NUMINAMATH_CALUDE_point_movement_power_l541_54135


namespace NUMINAMATH_CALUDE_correct_propositions_l541_54127

theorem correct_propositions :
  (∀ m : ℝ, m > 0 → ∃ x : ℝ, x^2 + x - m = 0) ∧
  (∀ a b : ℝ, ab ≠ 0 → a ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_correct_propositions_l541_54127


namespace NUMINAMATH_CALUDE_complex_fraction_real_l541_54157

theorem complex_fraction_real (t : ℝ) : 
  (Complex.I * (2 * t + Complex.I) / (1 - 2 * Complex.I)).im = 0 → t = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_real_l541_54157


namespace NUMINAMATH_CALUDE_partition_into_three_exists_partition_into_four_not_exists_l541_54108

-- Define a partition of positive integers into three sets
def PartitionIntoThree : (ℕ → Fin 3) → Prop :=
  λ f => ∀ n, n > 0 → ∃ i, f n = i

-- Define a partition of positive integers into four sets
def PartitionIntoFour : (ℕ → Fin 4) → Prop :=
  λ f => ∀ n, n > 0 → ∃ i, f n = i

-- Statement 1
theorem partition_into_three_exists :
  ∃ f : ℕ → Fin 3, PartitionIntoThree f ∧
    ∀ n ≥ 15, ∀ i : Fin 3,
      ∃ a b : ℕ, a ≠ b ∧ f a = i ∧ f b = i ∧ a + b = n :=
sorry

-- Statement 2
theorem partition_into_four_not_exists :
  ∀ f : ℕ → Fin 4, PartitionIntoFour f →
    ∃ n ≥ 15, ∃ i : Fin 4,
      ∀ a b : ℕ, a ≠ b → f a = i → f b = i → a + b ≠ n :=
sorry

end NUMINAMATH_CALUDE_partition_into_three_exists_partition_into_four_not_exists_l541_54108


namespace NUMINAMATH_CALUDE_binomial_expansion_property_l541_54177

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the condition for the sum of the first three binomial coefficients
def first_three_sum_condition (n : ℕ) : Prop :=
  binomial n 0 + binomial n 1 + binomial n 2 = 79

-- Define the coefficient of the k-th term in the expansion
def coefficient (n k : ℕ) : ℚ := sorry

-- Define the property of having maximum coefficient
def has_max_coefficient (n k : ℕ) : Prop :=
  ∀ j, j ≠ k → coefficient n k ≥ coefficient n j

theorem binomial_expansion_property (n : ℕ) 
  (h : n > 0) 
  (h_sum : first_three_sum_condition n) :
  n = 12 ∧ has_max_coefficient n 10 := by sorry

end NUMINAMATH_CALUDE_binomial_expansion_property_l541_54177


namespace NUMINAMATH_CALUDE_asha_granny_gift_l541_54109

/-- The amount of money Asha was gifted by her granny --/
def granny_gift (brother_loan mother_loan father_loan savings spent_fraction remaining : ℚ) : ℚ :=
  (remaining / (1 - spent_fraction)) - (brother_loan + mother_loan + father_loan + savings)

/-- Theorem stating the amount gifted by Asha's granny --/
theorem asha_granny_gift :
  granny_gift 20 30 40 100 (3/4) 65 = 70 := by sorry

end NUMINAMATH_CALUDE_asha_granny_gift_l541_54109


namespace NUMINAMATH_CALUDE_m_range_l541_54190

-- Define propositions p and q
def p (x : ℝ) : Prop := x + 2 ≥ 0 ∧ x - 10 ≤ 0

def q (x m : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m

-- Define the necessary but not sufficient condition
def necessary_not_sufficient (p q : ℝ → Prop) : Prop :=
  (∀ x, q x → p x) ∧ ∃ x, p x ∧ ¬q x

-- State the theorem
theorem m_range (m : ℝ) :
  m > 0 ∧
  necessary_not_sufficient (p) (q m) →
  0 < m ∧ m ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_m_range_l541_54190


namespace NUMINAMATH_CALUDE_minimum_loss_for_1997_pills_l541_54107

/-- Represents a bottle of medicine --/
structure Bottle where
  capacity : ℕ
  pills : ℕ

/-- Represents the state of all bottles --/
structure State where
  a : Bottle
  b : Bottle
  c : Bottle
  loss : ℕ

/-- Calculates the minimum total loss of active ingredient --/
def minimumTotalLoss (initialPills : ℕ) : ℕ :=
  sorry

/-- Theorem stating the minimum total loss for the given problem --/
theorem minimum_loss_for_1997_pills :
  minimumTotalLoss 1997 = 32401 := by
  sorry

#check minimum_loss_for_1997_pills

end NUMINAMATH_CALUDE_minimum_loss_for_1997_pills_l541_54107


namespace NUMINAMATH_CALUDE_original_price_correct_l541_54137

/-- The original selling price of a shirt before discount -/
def original_price : ℝ := 700

/-- The discount percentage offered by the shop -/
def discount_percentage : ℝ := 20

/-- The price Smith paid for the shirt after discount -/
def discounted_price : ℝ := 560

/-- Theorem stating that the original price is correct given the discount and final price -/
theorem original_price_correct : 
  original_price * (1 - discount_percentage / 100) = discounted_price :=
by sorry

end NUMINAMATH_CALUDE_original_price_correct_l541_54137


namespace NUMINAMATH_CALUDE_tangent_slope_implies_a_value_l541_54115

-- Define the curve
def f (a : ℝ) (x : ℝ) : ℝ := x^4 + a*x^2 + 1

-- Define the derivative of the curve
def f' (a : ℝ) (x : ℝ) : ℝ := 4*x^3 + 2*a*x

theorem tangent_slope_implies_a_value :
  ∀ a : ℝ, f' a (-1) = 8 → a = -6 :=
by
  sorry

#check tangent_slope_implies_a_value

end NUMINAMATH_CALUDE_tangent_slope_implies_a_value_l541_54115


namespace NUMINAMATH_CALUDE_unique_rectangle_exists_restore_coordinate_system_l541_54103

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle in 2D space -/
structure Rectangle where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Checks if four points form a rectangle -/
def isRectangle (r : Rectangle) : Prop :=
  let AB := (r.B.x - r.A.x)^2 + (r.B.y - r.A.y)^2
  let BC := (r.C.x - r.B.x)^2 + (r.C.y - r.B.y)^2
  let CD := (r.D.x - r.C.x)^2 + (r.D.y - r.C.y)^2
  let DA := (r.A.x - r.D.x)^2 + (r.A.y - r.D.y)^2
  AB = CD ∧ BC = DA ∧ 
  (r.B.x - r.A.x) * (r.C.x - r.B.x) + (r.B.y - r.A.y) * (r.C.y - r.B.y) = 0

/-- Theorem: Given two points A and B, there exists a unique rectangle with A and B as diagonal endpoints -/
theorem unique_rectangle_exists (A B : Point) : 
  ∃! (r : Rectangle), r.A = A ∧ r.B = B ∧ isRectangle r := by
  sorry

/-- Main theorem: Given points A(1,2) and B(3,1), a unique rectangle can be constructed 
    with A and B as diagonal endpoints, which is sufficient to restore the coordinate system -/
theorem restore_coordinate_system : 
  let A : Point := ⟨1, 2⟩
  let B : Point := ⟨3, 1⟩
  ∃! (r : Rectangle), r.A = A ∧ r.B = B ∧ isRectangle r := by
  sorry

end NUMINAMATH_CALUDE_unique_rectangle_exists_restore_coordinate_system_l541_54103


namespace NUMINAMATH_CALUDE_line_through_parabola_vertex_count_l541_54183

/-- The number of values of a for which the line y = 2x + a passes through
    the vertex of the parabola y = x^2 + 2a^2 -/
theorem line_through_parabola_vertex_count : 
  ∃! (s : Finset ℝ), 
    (∀ a ∈ s, ∃ x y : ℝ, 
      y = 2 * x + a ∧ 
      y = x^2 + 2 * a^2 ∧ 
      ∀ x' : ℝ, x'^2 + 2 * a^2 ≤ x^2 + 2 * a^2) ∧ 
    s.card = 2 :=
sorry

end NUMINAMATH_CALUDE_line_through_parabola_vertex_count_l541_54183


namespace NUMINAMATH_CALUDE_circle_center_coordinates_l541_54112

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in the form ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y = l.c

/-- Check if a circle with center p is tangent to two parallel lines -/
def circleTangentToParallelLines (p : Point) (l1 l2 : Line) : Prop :=
  l1.a = l2.a ∧ l1.b = l2.b ∧ 
  abs (l1.a * p.x + l1.b * p.y - l1.c) = abs (l2.a * p.x + l2.b * p.y - l2.c)

theorem circle_center_coordinates : 
  ∃ (p : Point),
    circleTangentToParallelLines p (Line.mk 3 4 40) (Line.mk 3 4 (-20)) ∧
    pointOnLine p (Line.mk 1 (-2) 0) ∧
    p.x = 2 ∧ p.y = 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_coordinates_l541_54112


namespace NUMINAMATH_CALUDE_min_value_and_nonexistence_l541_54134

theorem min_value_and_nonexistence (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 4 * b = (a * b) ^ (3/2)) :
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → a' + 4 * b' = (a' * b') ^ (3/2) → a' ^ 2 + 16 * b' ^ 2 ≥ 32) ∧
  ¬∃ a' b' : ℝ, a' > 0 ∧ b' > 0 ∧ a' + 4 * b' = (a' * b') ^ (3/2) ∧ a' + 3 * b' = 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_nonexistence_l541_54134


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l541_54148

theorem contrapositive_equivalence (x : ℝ) : 
  (x^2 = 1 → x = 1 ∨ x = -1) ↔ (x ≠ 1 ∧ x ≠ -1 → x^2 ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l541_54148


namespace NUMINAMATH_CALUDE_jack_remaining_notebooks_l541_54168

-- Define the initial number of notebooks for Gerald
def gerald_notebooks : ℕ := 8

-- Define Jack's initial number of notebooks relative to Gerald's
def jack_initial_notebooks : ℕ := gerald_notebooks + 13

-- Define the number of notebooks Jack gives to Paula
def notebooks_to_paula : ℕ := 5

-- Define the number of notebooks Jack gives to Mike
def notebooks_to_mike : ℕ := 6

-- Theorem: Jack has 10 notebooks left
theorem jack_remaining_notebooks :
  jack_initial_notebooks - (notebooks_to_paula + notebooks_to_mike) = 10 := by
  sorry

end NUMINAMATH_CALUDE_jack_remaining_notebooks_l541_54168


namespace NUMINAMATH_CALUDE_circle_partition_exists_l541_54167

/-- Represents a person with their country and position -/
structure Person where
  country : Fin 25
  position : Fin 100

/-- Defines the arrangement of people in a circle -/
def arrangement : Fin 100 → Person :=
  sorry

/-- Checks if two people are adjacent in the circle -/
def are_adjacent (p1 p2 : Person) : Prop :=
  sorry

/-- Represents a partition of people into 4 groups -/
def Partition := Fin 100 → Fin 4

/-- Checks if a partition is valid according to the problem conditions -/
def is_valid_partition (p : Partition) : Prop :=
  ∀ i j : Fin 100,
    i ≠ j →
    (arrangement i).country = (arrangement j).country ∨ are_adjacent (arrangement i) (arrangement j) →
    p i ≠ p j

theorem circle_partition_exists :
  ∃ p : Partition, is_valid_partition p :=
sorry

end NUMINAMATH_CALUDE_circle_partition_exists_l541_54167


namespace NUMINAMATH_CALUDE_apples_eaten_l541_54142

theorem apples_eaten (x : ℚ) 
  (h1 : x > 0)  -- Assumption that x is positive (number of apples can't be negative)
  (h2 : x + 2*x + x/2 = 14) : x = 4 := by
  sorry

end NUMINAMATH_CALUDE_apples_eaten_l541_54142


namespace NUMINAMATH_CALUDE_lcm_48_180_l541_54149

theorem lcm_48_180 : Nat.lcm 48 180 = 720 := by
  sorry

end NUMINAMATH_CALUDE_lcm_48_180_l541_54149


namespace NUMINAMATH_CALUDE_max_profit_l541_54188

noncomputable section

def fixed_cost : ℝ := 2.5

def variable_cost (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 40 then 10*x^2 + 100*x
  else if x ≥ 40 then 701*x + 10000/x - 9450
  else 0

def selling_price : ℝ := 0.7

def profit (x : ℝ) : ℝ :=
  selling_price * x - (fixed_cost + variable_cost x)

def production_quantity : ℝ := 100

theorem max_profit :
  profit production_quantity = 9000 ∧
  ∀ x > 0, profit x ≤ profit production_quantity :=
sorry

end

end NUMINAMATH_CALUDE_max_profit_l541_54188


namespace NUMINAMATH_CALUDE_janet_ticket_count_l541_54102

/-- The number of tickets needed for Janet's amusement park rides -/
def total_tickets (roller_coaster_tickets : ℕ) (giant_slide_tickets : ℕ) 
                  (roller_coaster_rides : ℕ) (giant_slide_rides : ℕ) : ℕ :=
  roller_coaster_tickets * roller_coaster_rides + giant_slide_tickets * giant_slide_rides

/-- Proof that Janet needs 47 tickets for her planned rides -/
theorem janet_ticket_count : 
  total_tickets 5 3 7 4 = 47 := by
  sorry

end NUMINAMATH_CALUDE_janet_ticket_count_l541_54102


namespace NUMINAMATH_CALUDE_only_valid_rectangles_l541_54133

/-- A rectangle that can be divided into 13 equal squares -/
structure Rectangle13Squares where
  width : ℕ
  height : ℕ
  is_valid : width * height = 13

/-- The set of all valid rectangles that can be divided into 13 equal squares -/
def valid_rectangles : Set Rectangle13Squares :=
  {r : Rectangle13Squares | r.width = 1 ∧ r.height = 13 ∨ r.width = 13 ∧ r.height = 1}

/-- Theorem stating that the only valid rectangles are 1x13 or 13x1 -/
theorem only_valid_rectangles :
  ∀ r : Rectangle13Squares, r ∈ valid_rectangles :=
by
  sorry

end NUMINAMATH_CALUDE_only_valid_rectangles_l541_54133


namespace NUMINAMATH_CALUDE_expected_sufferers_l541_54123

theorem expected_sufferers (sample_size : ℕ) (probability : ℚ) (h1 : sample_size = 400) (h2 : probability = 1/4) :
  ↑sample_size * probability = 100 := by
  sorry

end NUMINAMATH_CALUDE_expected_sufferers_l541_54123


namespace NUMINAMATH_CALUDE_total_shoes_count_l541_54195

theorem total_shoes_count (bonny becky bobby cherry diane : ℕ) : 
  bonny = 13 ∧
  bonny = 2 * becky - 5 ∧
  bobby = 3 * becky ∧
  cherry = bonny + becky + 4 ∧
  diane = 2 * cherry - 2 →
  bonny + becky + bobby + cherry + diane = 125 := by
  sorry

end NUMINAMATH_CALUDE_total_shoes_count_l541_54195


namespace NUMINAMATH_CALUDE_triangle_sum_bounds_l541_54132

theorem triangle_sum_bounds (A B C : Real) (hsum : A + B + C = Real.pi) (hpos : 0 < A ∧ 0 < B ∧ 0 < C) :
  let S := Real.sqrt (3 * Real.tan (A/2) * Real.tan (B/2) + 1) +
           Real.sqrt (3 * Real.tan (B/2) * Real.tan (C/2) + 1) +
           Real.sqrt (3 * Real.tan (C/2) * Real.tan (A/2) + 1)
  4 ≤ S ∧ S < 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sum_bounds_l541_54132


namespace NUMINAMATH_CALUDE_box_dimensions_l541_54194

theorem box_dimensions (a h : ℝ) (b : ℝ) : 
  h = a / 2 →
  6 * a + b = 156 →
  7 * a + b = 178 →
  a = 22 ∧ h = 11 :=
by sorry

end NUMINAMATH_CALUDE_box_dimensions_l541_54194


namespace NUMINAMATH_CALUDE_other_endpoint_coordinate_sum_l541_54189

/-- Given a line segment with midpoint (5, -8) and one endpoint at (9, -6),
    the sum of the coordinates of the other endpoint is -9. -/
theorem other_endpoint_coordinate_sum :
  ∀ (x y : ℝ),
  (5 = (9 + x) / 2) →
  (-8 = (-6 + y) / 2) →
  x + y = -9 :=
by sorry

end NUMINAMATH_CALUDE_other_endpoint_coordinate_sum_l541_54189


namespace NUMINAMATH_CALUDE_omega_range_l541_54173

theorem omega_range (f : ℝ → ℝ) (ω : ℝ) :
  (∀ x, f x = Real.cos (ω * x + π / 6)) →
  ω > 0 →
  (∀ x ∈ Set.Icc 0 π, f x ∈ Set.Icc (-1) (Real.sqrt 3 / 2)) →
  ω ∈ Set.Icc (5 / 6) (5 / 3) :=
sorry

end NUMINAMATH_CALUDE_omega_range_l541_54173


namespace NUMINAMATH_CALUDE_area_triangle_AOC_l541_54181

/-- Circle C with equation x^2 + y^2 - 4x - 6y + 12 = 0 -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 6*y + 12 = 0

/-- Point A with coordinates (3, 5) -/
def point_A : ℝ × ℝ := (3, 5)

/-- Origin O -/
def point_O : ℝ × ℝ := (0, 0)

/-- Point C is the center of the circle -/
def point_C : ℝ × ℝ := (2, 3)

/-- The area of triangle AOC is 1/2 -/
theorem area_triangle_AOC :
  let A := point_A
  let O := point_O
  let C := point_C
  (1/2 : ℝ) * ‖(A.1 - O.1, A.2 - O.2)‖ * ‖(C.1 - O.1, C.2 - O.2)‖ * 
    Real.sin (Real.arccos ((A.1 - O.1) * (C.1 - O.1) + (A.2 - O.2) * (C.2 - O.2)) / 
    (‖(A.1 - O.1, A.2 - O.2)‖ * ‖(C.1 - O.1, C.2 - O.2)‖)) = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_area_triangle_AOC_l541_54181


namespace NUMINAMATH_CALUDE_value_of_expression_l541_54106

theorem value_of_expression (x : ℝ) (h : x = -2) : (3 * x + 4)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l541_54106


namespace NUMINAMATH_CALUDE_ceiling_sum_of_roots_l541_54138

theorem ceiling_sum_of_roots : ⌈Real.sqrt 3⌉ + ⌈Real.sqrt 33⌉ + ⌈Real.sqrt 333⌉ = 27 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sum_of_roots_l541_54138


namespace NUMINAMATH_CALUDE_same_gender_probability_same_school_probability_l541_54144

/-- Represents a school with a certain number of male and female teachers --/
structure School where
  male_teachers : ℕ
  female_teachers : ℕ

/-- The total number of teachers in a school --/
def School.total_teachers (s : School) : ℕ := s.male_teachers + s.female_teachers

/-- The schools in the problem --/
def school_A : School := { male_teachers := 2, female_teachers := 1 }
def school_B : School := { male_teachers := 1, female_teachers := 2 }

/-- The total number of teachers in both schools --/
def total_teachers : ℕ := school_A.total_teachers + school_B.total_teachers

/-- Theorem for the probability of selecting two teachers of the same gender --/
theorem same_gender_probability :
  (school_A.male_teachers * school_B.male_teachers + school_A.female_teachers * school_B.female_teachers) / 
  (school_A.total_teachers * school_B.total_teachers) = 4 / 9 := by sorry

/-- Theorem for the probability of selecting two teachers from the same school --/
theorem same_school_probability :
  (school_A.total_teachers * (school_A.total_teachers - 1) + school_B.total_teachers * (school_B.total_teachers - 1)) / 
  (total_teachers * (total_teachers - 1)) = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_same_gender_probability_same_school_probability_l541_54144


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l541_54152

theorem cubic_equation_roots (a b : ℝ) : 
  (∀ x : ℝ, 2*x^3 + a*x^2 - 13*x + b = 0 ↔ x = 2 ∨ x = -3 ∨ (∃ r : ℝ, x = r ∧ 2*(2-r)*(3+r) = 0)) →
  a = 1 ∧ b = 6 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l541_54152


namespace NUMINAMATH_CALUDE_quadrilateral_bd_length_l541_54139

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the length function
def length (p q : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem quadrilateral_bd_length (ABCD : Quadrilateral) : 
  length ABCD.A ABCD.B = 4 →
  length ABCD.B ABCD.C = 14 →
  length ABCD.C ABCD.D = 4 →
  length ABCD.D ABCD.A = 7 →
  ∃ (n : ℕ), length ABCD.B ABCD.D = n →
  length ABCD.B ABCD.D = 11 := by
sorry

end NUMINAMATH_CALUDE_quadrilateral_bd_length_l541_54139


namespace NUMINAMATH_CALUDE_car_travel_time_l541_54129

/-- Proves that a car traveling 715 kilometers at an average speed of 65.0 km/h takes 11 hours -/
theorem car_travel_time (distance : ℝ) (speed : ℝ) (time : ℝ) :
  distance = 715 →
  speed = 65 →
  time = distance / speed →
  time = 11 :=
by sorry

end NUMINAMATH_CALUDE_car_travel_time_l541_54129


namespace NUMINAMATH_CALUDE_exists_k_for_special_sequence_l541_54191

/-- A sequence of non-negative integers satisfying certain conditions -/
def SpecialSequence (c : Fin 1997 → ℕ) : Prop :=
  (c 1 ≥ 0) ∧
  (∀ m n : Fin 1997, m > 0 → n > 0 → m + n < 1998 →
    c m + c n ≤ c (m + n) ∧ c (m + n) ≤ c m + c n + 1)

/-- Theorem stating the existence of k for the special sequence -/
theorem exists_k_for_special_sequence (c : Fin 1997 → ℕ) (h : SpecialSequence c) :
  ∃ k : ℝ, ∀ n : Fin 1997, c n = ⌊n * k⌋ :=
sorry

end NUMINAMATH_CALUDE_exists_k_for_special_sequence_l541_54191


namespace NUMINAMATH_CALUDE_singing_competition_ratio_l541_54184

/-- Proves that the ratio of female contestants to the total number of contestants is 1/3 -/
theorem singing_competition_ratio :
  let total_contestants : ℕ := 18
  let male_contestants : ℕ := 12
  let female_contestants : ℕ := total_contestants - male_contestants
  (female_contestants : ℚ) / total_contestants = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_singing_competition_ratio_l541_54184


namespace NUMINAMATH_CALUDE_coles_return_speed_coles_return_speed_is_90_l541_54193

/-- Calculates the average speed on the return trip given the conditions of Cole's journey. -/
theorem coles_return_speed (speed_to_work : ℝ) (total_time : ℝ) (time_to_work : ℝ) : ℝ :=
  let distance_to_work := speed_to_work * (time_to_work / 60)
  let time_to_return := total_time - (time_to_work / 60)
  distance_to_work / time_to_return

/-- Proves that Cole's average speed on the return trip is 90 km/h given the problem conditions. -/
theorem coles_return_speed_is_90 :
  coles_return_speed 30 2 90 = 90 := by
  sorry

end NUMINAMATH_CALUDE_coles_return_speed_coles_return_speed_is_90_l541_54193


namespace NUMINAMATH_CALUDE_product_of_solutions_is_negative_162_l541_54185

-- Define the function f
def f (x : ℝ) : ℝ := 18 * x + 4

-- Define the inverse function of f
noncomputable def f_inv (x : ℝ) : ℝ := (x - 4) / 18

-- Theorem statement
theorem product_of_solutions_is_negative_162 :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧
  (∀ x : ℝ, f_inv x = f ((2*x)⁻¹) ↔ (x = x₁ ∨ x = x₂)) ∧
  x₁ * x₂ = -162 := by
sorry

end NUMINAMATH_CALUDE_product_of_solutions_is_negative_162_l541_54185


namespace NUMINAMATH_CALUDE_divisibility_in_ones_sequence_l541_54174

theorem divisibility_in_ones_sequence (k : ℕ) (hprime : Nat.Prime k) (h2 : k ≠ 2) (h5 : k ≠ 5) :
  ∃ n : ℕ, n ≤ k ∧ k ∣ ((10^n - 1) / 9) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_in_ones_sequence_l541_54174


namespace NUMINAMATH_CALUDE_perfect_squares_divisibility_l541_54196

theorem perfect_squares_divisibility (a b : ℕ+) 
  (h : ∃ S : Set (ℕ+ × ℕ+), Set.Infinite S ∧ 
    ∀ (m n : ℕ+), (m, n) ∈ S → 
      ∃ (k l : ℕ+), (m : ℕ)^2 + (a : ℕ) * (n : ℕ) + (b : ℕ) = (k : ℕ)^2 ∧
                    (n : ℕ)^2 + (a : ℕ) * (m : ℕ) + (b : ℕ) = (l : ℕ)^2) : 
  (a : ℕ) ∣ 2 * (b : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_perfect_squares_divisibility_l541_54196


namespace NUMINAMATH_CALUDE_honey_balance_l541_54104

/-- The initial amount of honey produced by a bee colony -/
def initial_honey : ℝ := 0.36

/-- The amount of honey eaten by bears -/
def eaten_honey : ℝ := 0.05

/-- The amount of honey that remains -/
def remaining_honey : ℝ := 0.31

/-- Theorem stating that the initial amount of honey is equal to the sum of eaten and remaining honey -/
theorem honey_balance : initial_honey = eaten_honey + remaining_honey := by
  sorry

end NUMINAMATH_CALUDE_honey_balance_l541_54104


namespace NUMINAMATH_CALUDE_add_base6_35_14_l541_54131

/-- Converts a base 6 number to base 10 --/
def base6_to_base10 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 6 --/
def base10_to_base6 (n : ℕ) : ℕ := sorry

/-- Addition in base 6 --/
def add_base6 (a b : ℕ) : ℕ :=
  base10_to_base6 (base6_to_base10 a + base6_to_base10 b)

theorem add_base6_35_14 : add_base6 35 14 = 53 := by sorry

end NUMINAMATH_CALUDE_add_base6_35_14_l541_54131


namespace NUMINAMATH_CALUDE_solution_set_theorem_range_of_k_theorem_l541_54118

-- Define the function f
def f (x k : ℝ) : ℝ := |x + 1| + |2 - x| - k

-- Theorem 1: Solution set of f(x) < 0 when k = 4
theorem solution_set_theorem :
  {x : ℝ | f x 4 < 0} = Set.Ioo (-3/2) (5/2) := by sorry

-- Theorem 2: Range of k for f(x) ≥ √(k+3) for all x ∈ ℝ
theorem range_of_k_theorem :
  ∀ k : ℝ, (∀ x : ℝ, f x k ≥ Real.sqrt (k + 3)) ↔ k ∈ Set.Iic 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_theorem_range_of_k_theorem_l541_54118


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l541_54130

theorem quadratic_inequality_condition (x : ℝ) :
  (∀ x, 0 < x ∧ x < 2 → x^2 - x - 6 < 0) ∧
  (∃ x, x^2 - x - 6 < 0 ∧ ¬(0 < x ∧ x < 2)) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l541_54130


namespace NUMINAMATH_CALUDE_remainder_divisibility_l541_54198

theorem remainder_divisibility (N : ℤ) : 
  N % 342 = 47 → N % 19 = 9 := by
  sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l541_54198


namespace NUMINAMATH_CALUDE_intersection_not_empty_implies_a_value_l541_54110

theorem intersection_not_empty_implies_a_value (a : ℤ) : 
  let M : Set ℤ := {a, 0}
  let N : Set ℤ := {x | 2 * x^2 - 5 * x < 0}
  (M ∩ N).Nonempty → a = 1 ∨ a = 2 := by
sorry

end NUMINAMATH_CALUDE_intersection_not_empty_implies_a_value_l541_54110


namespace NUMINAMATH_CALUDE_f_min_value_l541_54178

/-- The function f(x) = |2x-1| + |3x-2| + |4x-3| + |5x-4| -/
def f (x : ℝ) : ℝ := |2*x - 1| + |3*x - 2| + |4*x - 3| + |5*x - 4|

/-- Theorem: The minimum value of f(x) is 1 -/
theorem f_min_value :
  (∀ x : ℝ, f x ≥ 1) ∧ (∃ x : ℝ, f x = 1) := by
  sorry

end NUMINAMATH_CALUDE_f_min_value_l541_54178


namespace NUMINAMATH_CALUDE_expression_value_l541_54155

theorem expression_value : 
  let x : ℕ := 3
  x + x * (x ^ (x ^ 2)) = 59052 := by sorry

end NUMINAMATH_CALUDE_expression_value_l541_54155


namespace NUMINAMATH_CALUDE_max_value_x_plus_2cos_x_l541_54180

open Real

theorem max_value_x_plus_2cos_x (x : ℝ) :
  let f : ℝ → ℝ := λ x => x + 2 * cos x
  (∀ y ∈ Set.Icc 0 (π / 2), f (π / 6) ≥ f y) ∧
  (π / 6 ∈ Set.Icc 0 (π / 2)) :=
by sorry

end NUMINAMATH_CALUDE_max_value_x_plus_2cos_x_l541_54180


namespace NUMINAMATH_CALUDE_num_pencils_is_75_l541_54154

/-- The number of pencils purchased given the conditions of the problem -/
def num_pencils : ℕ :=
  let num_pens : ℕ := 30
  let total_cost : ℕ := 570
  let pencil_price : ℕ := 2
  let pen_price : ℕ := 14
  let pen_cost : ℕ := num_pens * pen_price
  let pencil_cost : ℕ := total_cost - pen_cost
  pencil_cost / pencil_price

theorem num_pencils_is_75 : num_pencils = 75 := by
  sorry

end NUMINAMATH_CALUDE_num_pencils_is_75_l541_54154


namespace NUMINAMATH_CALUDE_john_replacement_cost_l541_54159

/-- Represents the genre of a movie --/
inductive Genre
  | Action
  | Comedy
  | Drama

/-- Represents the popularity of a movie --/
inductive Popularity
  | Popular
  | ModeratelyPopular
  | Unpopular

/-- Represents a movie with its genre and popularity --/
structure Movie where
  genre : Genre
  popularity : Popularity

/-- The trade-in value for a VHS movie based on its genre --/
def tradeInValue (g : Genre) : ℕ :=
  match g with
  | Genre.Action => 3
  | Genre.Comedy => 2
  | Genre.Drama => 1

/-- The purchase price for a DVD based on its popularity --/
def purchasePrice (p : Popularity) : ℕ :=
  match p with
  | Popularity.Popular => 12
  | Popularity.ModeratelyPopular => 8
  | Popularity.Unpopular => 5

/-- The collection of movies John has --/
def johnMovies : List Movie :=
  (List.replicate 20 ⟨Genre.Action, Popularity.Popular⟩) ++
  (List.replicate 30 ⟨Genre.Comedy, Popularity.ModeratelyPopular⟩) ++
  (List.replicate 10 ⟨Genre.Drama, Popularity.Unpopular⟩) ++
  (List.replicate 15 ⟨Genre.Comedy, Popularity.Popular⟩) ++
  (List.replicate 25 ⟨Genre.Action, Popularity.ModeratelyPopular⟩)

/-- The total cost to replace all movies --/
def replacementCost (movies : List Movie) : ℕ :=
  (movies.map (fun m => purchasePrice m.popularity)).sum -
  (movies.map (fun m => tradeInValue m.genre)).sum

/-- Theorem stating the cost to replace all of John's movies --/
theorem john_replacement_cost :
  replacementCost johnMovies = 675 := by
  sorry

end NUMINAMATH_CALUDE_john_replacement_cost_l541_54159


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l541_54136

/-- An arithmetic sequence with specified properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum sequence
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_property : ∀ n : ℕ, S n = (n : ℝ) * (a 1 + a n) / 2
  a4_eq_10 : a 4 = 10
  S6_eq_S3_plus_39 : S 6 = S 3 + 39

/-- The theorem stating the properties of the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  seq.a 1 = 1 ∧ ∀ n : ℕ, seq.a n = 3 * n - 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l541_54136


namespace NUMINAMATH_CALUDE_arccos_zero_equals_pi_half_l541_54146

theorem arccos_zero_equals_pi_half : Real.arccos 0 = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arccos_zero_equals_pi_half_l541_54146


namespace NUMINAMATH_CALUDE_book_arrangement_theorem_l541_54147

/-- Represents the number of ways to arrange books of two subjects -/
def arrange_books (n : ℕ) : ℕ :=
  2 * (n.factorial * n.factorial)

/-- Theorem: The number of ways to arrange 3 math books and 3 Chinese books
    on a shelf, such that no two books of the same subject are adjacent,
    is equal to 72 -/
theorem book_arrangement_theorem :
  arrange_books 3 = 72 := by
  sorry

#eval arrange_books 3  -- This should output 72

end NUMINAMATH_CALUDE_book_arrangement_theorem_l541_54147


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l541_54182

def A : Set ℝ := {x | x < 2}
def B : Set ℝ := {y | ∃ x, y = 2^x - 1}

theorem intersection_of_A_and_B : A ∩ B = {m | -1 < m ∧ m < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l541_54182


namespace NUMINAMATH_CALUDE_plan_y_cheaper_at_601_l541_54114

/-- Represents an internet service plan with a flat fee and per-gigabyte charge -/
structure InternetPlan where
  flatFee : ℕ
  perGBCharge : ℕ

/-- Calculates the total cost in cents for a given plan and number of gigabytes -/
def totalCost (plan : InternetPlan) (gigabytes : ℕ) : ℕ :=
  plan.flatFee * 100 + plan.perGBCharge * gigabytes

theorem plan_y_cheaper_at_601 :
  let planX : InternetPlan := ⟨50, 15⟩
  let planY : InternetPlan := ⟨80, 10⟩
  ∀ g : ℕ, g ≥ 601 → totalCost planY g < totalCost planX g ∧
  ∀ g : ℕ, g < 601 → totalCost planX g ≤ totalCost planY g :=
by sorry

end NUMINAMATH_CALUDE_plan_y_cheaper_at_601_l541_54114


namespace NUMINAMATH_CALUDE_equation_solutions_l541_54128

theorem equation_solutions :
  (∀ x : ℝ, 9 * x^2 - 25 = 0 ↔ x = 5/3 ∨ x = -5/3) ∧
  (∀ x : ℝ, (x + 1)^3 - 27 = 0 ↔ x = 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l541_54128


namespace NUMINAMATH_CALUDE_folded_rectangle_length_l541_54116

/-- Given a rectangular strip of paper with dimensions 4 × 13, folded to form two rectangles
    with areas P and Q such that P = 2Q, prove that the length of one of the resulting rectangles is 6. -/
theorem folded_rectangle_length (x y : ℝ) (P Q : ℝ) : 
  x + y = 9 →  -- Sum of lengths of the two rectangles
  x + 4 + y = 13 →  -- Total length of the original rectangle
  P = 4 * x →  -- Area of rectangle P
  Q = 4 * y →  -- Area of rectangle Q
  P = 2 * Q →  -- Relationship between areas P and Q
  x = 6 := by sorry

end NUMINAMATH_CALUDE_folded_rectangle_length_l541_54116


namespace NUMINAMATH_CALUDE_f₁_eq_f₂_l541_54126

/-- Function f₁ that always returns 1 -/
def f₁ : ℝ → ℝ := λ _ => 1

/-- Function f₂ that returns x^0 -/
def f₂ : ℝ → ℝ := λ x => x^0

/-- Theorem stating that f₁ and f₂ are the same function -/
theorem f₁_eq_f₂ : f₁ = f₂ := by sorry

end NUMINAMATH_CALUDE_f₁_eq_f₂_l541_54126


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_property_l541_54169

/-- A positive arithmetic geometric sequence -/
def ArithmeticGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n > 0 ∧ (a (n + 1) - a n) = (a (n + 2) - a (n + 1))
    ∧ (a (n + 1))^2 = (a n) * (a (n + 2))

theorem arithmetic_geometric_sequence_property
  (a : ℕ → ℝ) (h : ArithmeticGeometricSequence a)
  (h_eq : a 1 * a 5 + 2 * a 3 * a 6 + a 1 * a 11 = 16) :
  a 3 + a 6 = 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_property_l541_54169


namespace NUMINAMATH_CALUDE_max_distance_difference_l541_54122

/-- The hyperbola E with equation x²/m - y²/3 = 1 where m > 0 -/
structure Hyperbola where
  m : ℝ
  h_m_pos : m > 0

/-- The eccentricity of the hyperbola -/
def eccentricity (E : Hyperbola) : ℝ := 2

/-- The right focus F of the hyperbola -/
def right_focus (E : Hyperbola) : ℝ × ℝ := sorry

/-- Point A -/
def point_A : ℝ × ℝ := (0, 1)

/-- A point P on the right branch of the hyperbola -/
def point_P (E : Hyperbola) : ℝ × ℝ := sorry

/-- The distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- The theorem stating the maximum value of |PF| - |PA| -/
theorem max_distance_difference (E : Hyperbola) :
  ∃ (max : ℝ), ∀ (P : ℝ × ℝ), P = point_P E →
    distance P (right_focus E) - distance P point_A ≤ max ∧
    max = Real.sqrt 5 - 2 :=
sorry

end NUMINAMATH_CALUDE_max_distance_difference_l541_54122


namespace NUMINAMATH_CALUDE_hyperbola_equation_tangent_line_perpendicular_intersection_l541_54117

-- Define the hyperbola C
def hyperbola_C (x y : ℝ) : Prop :=
  abs (((x + 2)^2 + y^2).sqrt - ((x - 2)^2 + y^2).sqrt) = 2 * Real.sqrt 3

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * x + 1

-- Define the modified hyperbola C' for part 3
def hyperbola_C' (x y : ℝ) : Prop :=
  abs (((x + 2)^2 + y^2).sqrt - ((x - 2)^2 + y^2).sqrt) = 2

-- Define the line for part 3
def line_part3 (k : ℝ) (x y : ℝ) : Prop := y = k * x + 2

-- Theorem statements
theorem hyperbola_equation :
  ∀ x y : ℝ, hyperbola_C x y ↔ x^2 / 3 - y^2 = 1 := by sorry

theorem tangent_line :
  ∀ k : ℝ, (∃! p : ℝ × ℝ, hyperbola_C p.1 p.2 ∧ line_l k p.1 p.2) ↔
    k = Real.sqrt 3 / 3 ∨ k = -Real.sqrt 3 / 3 ∨ k = 2 ∨ k = -2 := by sorry

theorem perpendicular_intersection :
  ∀ k : ℝ, (∃ A B : ℝ × ℝ,
    hyperbola_C' A.1 A.2 ∧ hyperbola_C' B.1 B.2 ∧
    line_part3 k A.1 A.2 ∧ line_part3 k B.1 B.2 ∧
    A.1 * B.1 + A.2 * B.2 = 0) ↔
  k = Real.sqrt 2 ∨ k = -Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_tangent_line_perpendicular_intersection_l541_54117


namespace NUMINAMATH_CALUDE_no_real_solution_for_sqrt_equation_l541_54145

theorem no_real_solution_for_sqrt_equation :
  ¬∃ (x : ℝ), Real.sqrt (4 - 5*x) = 9 - x := by
sorry

end NUMINAMATH_CALUDE_no_real_solution_for_sqrt_equation_l541_54145


namespace NUMINAMATH_CALUDE_adrian_cards_l541_54163

theorem adrian_cards (n : ℕ) : 
  (∃ k : ℕ, 
    k ≥ 1 ∧ 
    k + n - 1 ≤ 2 * n ∧ 
    (2 * n * (2 * n + 1)) / 2 - (n * k + (n * (n - 1)) / 2) = 1615) →
  (n = 34 ∨ n = 38) :=
by sorry

end NUMINAMATH_CALUDE_adrian_cards_l541_54163


namespace NUMINAMATH_CALUDE_chicken_egg_production_l541_54143

/-- Calculates the number of eggs laid per day by each chicken given the total revenue, 
    price per dozen eggs, number of chickens, and number of days. -/
def eggs_per_chicken_per_day (total_revenue : ℚ) (price_per_dozen : ℚ) 
                              (num_chickens : ℕ) (num_days : ℕ) : ℚ :=
  (total_revenue / price_per_dozen * 12) / (num_chickens * num_days)

/-- Proves that given the specified conditions, each chicken lays 3 eggs per day. -/
theorem chicken_egg_production : 
  eggs_per_chicken_per_day 280 5 8 (4 * 7) = 3 := by
  sorry

end NUMINAMATH_CALUDE_chicken_egg_production_l541_54143


namespace NUMINAMATH_CALUDE_average_hours_worked_per_month_l541_54172

def hours_per_day_april : ℕ := 6
def hours_per_day_june : ℕ := 5
def hours_per_day_september : ℕ := 8
def days_per_month : ℕ := 30
def num_months : ℕ := 3

theorem average_hours_worked_per_month :
  (hours_per_day_april * days_per_month +
   hours_per_day_june * days_per_month +
   hours_per_day_september * days_per_month) / num_months = 190 := by
  sorry

end NUMINAMATH_CALUDE_average_hours_worked_per_month_l541_54172


namespace NUMINAMATH_CALUDE_two_train_problem_l541_54160

/-- Prove that given the conditions of the two-train problem, the speed of the second train is 40 km/hr -/
theorem two_train_problem (v : ℝ) : 
  (∀ t : ℝ, 50 * t = v * t + 100) →  -- First train travels 100 km more
  (∀ t : ℝ, 50 * t + v * t = 900) →  -- Total distance is 900 km
  v = 40 := by
  sorry

end NUMINAMATH_CALUDE_two_train_problem_l541_54160


namespace NUMINAMATH_CALUDE_express_y_in_terms_of_x_l541_54158

/-- Given the equation 2x + y = 6, prove that y can be expressed as 6 - 2x. -/
theorem express_y_in_terms_of_x (x y : ℝ) (h : 2 * x + y = 6) : y = 6 - 2 * x := by
  sorry

end NUMINAMATH_CALUDE_express_y_in_terms_of_x_l541_54158


namespace NUMINAMATH_CALUDE_f_min_at_three_l541_54100

/-- The quadratic function to be minimized -/
def f (c : ℝ) : ℝ := 3 * c^2 - 18 * c + 20

/-- Theorem stating that f is minimized at c = 3 -/
theorem f_min_at_three : 
  ∀ x : ℝ, f 3 ≤ f x := by sorry

end NUMINAMATH_CALUDE_f_min_at_three_l541_54100


namespace NUMINAMATH_CALUDE_grunters_win_probabilities_l541_54121

/-- The number of games played -/
def num_games : ℕ := 6

/-- The probability of winning a single game -/
def win_prob : ℚ := 7/10

/-- The probability of winning all games -/
def prob_win_all : ℚ := 117649/1000000

/-- The probability of winning exactly 5 out of 6 games -/
def prob_win_five : ℚ := 302526/1000000

/-- Theorem stating the probabilities for winning all games and winning exactly 5 out of 6 games -/
theorem grunters_win_probabilities :
  (win_prob ^ num_games = prob_win_all) ∧
  (Nat.choose num_games 5 * win_prob ^ 5 * (1 - win_prob) ^ 1 = prob_win_five) := by
  sorry

end NUMINAMATH_CALUDE_grunters_win_probabilities_l541_54121


namespace NUMINAMATH_CALUDE_scalene_triangle_not_divisible_into_two_congruent_pieces_l541_54165

-- Define a scalene triangle
structure ScaleneTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0
  hab : a ≠ b
  hbc : b ≠ c
  hac : a ≠ c

-- Define congruence for triangles
def CongruentTriangles (t1 t2 : ScaleneTriangle) : Prop :=
  t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c

-- Theorem statement
theorem scalene_triangle_not_divisible_into_two_congruent_pieces 
  (t : ScaleneTriangle) : 
  ¬∃ (t1 t2 : ScaleneTriangle), 
    CongruentTriangles t1 t2 ∧ 
    t.a = t1.a + t2.a ∧ 
    t.b = t1.b + t2.b ∧ 
    t.c = t1.c + t2.c :=
by sorry

end NUMINAMATH_CALUDE_scalene_triangle_not_divisible_into_two_congruent_pieces_l541_54165


namespace NUMINAMATH_CALUDE_board_covering_l541_54197

def can_cover (m n : ℕ) : Prop :=
  ∃ (a b : ℕ), m * n = 3 * a + 10 * b

def excluded_pairs : Set (ℕ × ℕ) :=
  {(4,4), (2,2), (2,4), (2,7)}

def excluded_1xn (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = 3*k + 1 ∨ n = 3*k + 2

theorem board_covering (m n : ℕ) :
  can_cover m n ↔ (m, n) ∉ excluded_pairs ∧ (m ≠ 1 ∨ ¬excluded_1xn n) :=
sorry

end NUMINAMATH_CALUDE_board_covering_l541_54197


namespace NUMINAMATH_CALUDE_tips_fraction_is_three_sevenths_l541_54176

/-- Represents the waiter's income structure -/
structure WaiterIncome where
  salary : ℚ
  tips : ℚ

/-- Calculates the fraction of income from tips -/
def fractionFromTips (income : WaiterIncome) : ℚ :=
  income.tips / (income.salary + income.tips)

/-- Theorem: The fraction of income from tips is 3/7 when tips are 3/4 of the salary -/
theorem tips_fraction_is_three_sevenths (income : WaiterIncome) 
    (h : income.tips = 3/4 * income.salary) : 
    fractionFromTips income = 3/7 := by
  sorry

end NUMINAMATH_CALUDE_tips_fraction_is_three_sevenths_l541_54176


namespace NUMINAMATH_CALUDE_pattern_36_l541_54141

-- Define a function that represents the pattern
def f (n : ℕ) : ℕ :=
  if n = 1 then 6
  else if n ≤ 5 then 360 + n
  else 3600 + n

-- State the theorem
theorem pattern_36 : f 36 = 3636 := by
  sorry

end NUMINAMATH_CALUDE_pattern_36_l541_54141


namespace NUMINAMATH_CALUDE_solve_for_b_l541_54164

theorem solve_for_b (a b : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 315 * b) : b = 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_b_l541_54164


namespace NUMINAMATH_CALUDE_insertion_methods_l541_54161

theorem insertion_methods (n : ℕ) (k : ℕ) : n = 5 ∧ k = 2 → (n + 1) * (n + 2) = 42 := by
  sorry

end NUMINAMATH_CALUDE_insertion_methods_l541_54161


namespace NUMINAMATH_CALUDE_some_number_value_l541_54162

theorem some_number_value (x : ℝ) (some_number : ℝ) 
  (h1 : 5 + 7 / x = some_number - 5 / x)
  (h2 : x = 12) : 
  some_number = 6 := by
sorry

end NUMINAMATH_CALUDE_some_number_value_l541_54162


namespace NUMINAMATH_CALUDE_toys_in_box_time_l541_54150

/-- The time in minutes required to put all toys in the box -/
def time_to_put_toys_in_box (total_toys : ℕ) (mom_puts_in : ℕ) (mia_takes_out : ℕ) (cycle_time : ℕ) : ℕ :=
  let net_gain_per_cycle := mom_puts_in - mia_takes_out
  let cycles_after_first_minute := (total_toys - 2 * mom_puts_in) / net_gain_per_cycle
  let total_seconds := (cycles_after_first_minute + 2) * cycle_time
  total_seconds / 60

/-- Theorem stating that under the given conditions, it takes 22 minutes to put all toys in the box -/
theorem toys_in_box_time : time_to_put_toys_in_box 45 4 3 30 = 22 := by
  sorry

end NUMINAMATH_CALUDE_toys_in_box_time_l541_54150


namespace NUMINAMATH_CALUDE_square_plus_fifteen_perfect_square_l541_54140

theorem square_plus_fifteen_perfect_square (n : ℤ) : 
  (∃ m : ℤ, n^2 + 15 = m^2) ↔ n = -7 ∨ n = -1 ∨ n = 1 ∨ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_fifteen_perfect_square_l541_54140


namespace NUMINAMATH_CALUDE_areas_product_eq_volume_squared_l541_54105

/-- A rectangular box with specific proportions -/
structure Box where
  width : ℝ
  length : ℝ
  height : ℝ
  length_eq : length = 2 * width
  height_eq : height = 3 * width

/-- The volume of the box -/
def volume (b : Box) : ℝ := b.length * b.width * b.height

/-- The area of the bottom of the box -/
def bottomArea (b : Box) : ℝ := b.length * b.width

/-- The area of the side of the box -/
def sideArea (b : Box) : ℝ := b.width * b.height

/-- The area of the front of the box -/
def frontArea (b : Box) : ℝ := b.length * b.height

/-- Theorem: The product of the areas equals the square of the volume -/
theorem areas_product_eq_volume_squared (b : Box) :
  bottomArea b * sideArea b * frontArea b = (volume b) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_areas_product_eq_volume_squared_l541_54105


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_inequality_negation_of_proposition_l541_54170

theorem negation_of_existence (p : ℝ → Prop) : 
  (¬∃ x, p x) ↔ (∀ x, ¬p x) :=
by sorry

theorem negation_of_inequality : 
  (¬(x + 2 ≤ 0)) ↔ (x + 2 > 0) :=
by sorry

theorem negation_of_proposition : 
  (¬∃ x : ℝ, x + 2 ≤ 0) ↔ (∀ x : ℝ, x + 2 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_inequality_negation_of_proposition_l541_54170


namespace NUMINAMATH_CALUDE_dog_bone_collection_l541_54192

/-- Calculates the final number of bones in a dog's collection after finding and giving away some bones. -/
def final_bone_count (initial_bones : ℕ) (found_multiplier : ℕ) (bones_given_away : ℕ) : ℕ :=
  initial_bones + (initial_bones * found_multiplier) - bones_given_away

/-- Theorem stating that given the specific conditions, the dog ends up with 3380 bones. -/
theorem dog_bone_collection : final_bone_count 350 9 120 = 3380 := by
  sorry

end NUMINAMATH_CALUDE_dog_bone_collection_l541_54192


namespace NUMINAMATH_CALUDE_pigeon_hole_theorem_l541_54111

/-- The number of pigeons -/
def num_pigeons : ℕ := 160

/-- The function that determines which hole a pigeon flies to -/
def pigeon_hole (i n : ℕ) : ℕ := i^2 % n

/-- Predicate to check if all pigeons fly to unique holes -/
def all_unique_holes (n : ℕ) : Prop :=
  ∀ i j, i ≤ num_pigeons → j ≤ num_pigeons → i ≠ j → pigeon_hole i n ≠ pigeon_hole j n

/-- The minimum number of holes needed -/
def min_holes : ℕ := 326

theorem pigeon_hole_theorem :
  (∀ k, k < min_holes → ¬(all_unique_holes k)) ∧ all_unique_holes min_holes :=
by sorry

end NUMINAMATH_CALUDE_pigeon_hole_theorem_l541_54111


namespace NUMINAMATH_CALUDE_faculty_reduction_percentage_l541_54186

theorem faculty_reduction_percentage (original : ℕ) (reduced : ℕ) : 
  original = 260 → reduced = 195 → 
  (original - reduced : ℚ) / original * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_faculty_reduction_percentage_l541_54186


namespace NUMINAMATH_CALUDE_complex_square_equality_l541_54113

theorem complex_square_equality (a b : ℝ) : 
  (Complex.I : ℂ)^2 = -1 →
  (a + b * Complex.I)^2 = (3 : ℂ) + 4 * Complex.I →
  a^2 + b^2 = 5 ∧ a * b = 2 := by
sorry

end NUMINAMATH_CALUDE_complex_square_equality_l541_54113


namespace NUMINAMATH_CALUDE_min_distance_B_to_M_l541_54153

-- Define the rectilinear distance function
def rectilinearDistance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  |x₁ - x₂| + |y₁ - y₂|

-- Define the point B
def B : ℝ × ℝ := (1, 1)

-- Define the line on which M moves
def lineM (x y : ℝ) : Prop :=
  x - y + 4 = 0

-- Theorem statement
theorem min_distance_B_to_M :
  ∃ (min : ℝ), min = 4 ∧
  ∀ (x y : ℝ), lineM x y →
    rectilinearDistance B.1 B.2 x y ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_distance_B_to_M_l541_54153


namespace NUMINAMATH_CALUDE_star_calculation_l541_54175

-- Define the star operation
def star (x y : ℝ) : ℝ := x^2 - 2*y

-- State the theorem
theorem star_calculation :
  let a := star 5 14
  let b := star 4 6
  star (2^a) (4^b) = -512.421875 := by sorry

end NUMINAMATH_CALUDE_star_calculation_l541_54175


namespace NUMINAMATH_CALUDE_simplify_expressions_l541_54166

variable (a b : ℝ)

theorem simplify_expressions :
  (4 * a^3 - a^2 + 1 - a^2 - 2 * a^3 = 2 * a^3 - 2 * a^2 + 1) ∧
  (2 * a - 3 * (5 * a - b) + 7 * (a + 2 * b) = -6 * a + 17 * b) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expressions_l541_54166


namespace NUMINAMATH_CALUDE_copper_percentage_in_alloy_l541_54151

/-- Given the following conditions:
    - 30 ounces of 20% alloy is used
    - 70 ounces of 27% alloy is used
    - Total amount of the desired alloy is 100 ounces
    Prove that the percentage of copper in the desired alloy is 24.9% -/
theorem copper_percentage_in_alloy : 
  let alloy_20_amount : ℝ := 30
  let alloy_27_amount : ℝ := 70
  let total_alloy : ℝ := 100
  let alloy_20_copper_percentage : ℝ := 20
  let alloy_27_copper_percentage : ℝ := 27
  let copper_amount : ℝ := (alloy_20_amount * alloy_20_copper_percentage / 100) + 
                           (alloy_27_amount * alloy_27_copper_percentage / 100)
  copper_amount / total_alloy * 100 = 24.9 := by
  sorry

end NUMINAMATH_CALUDE_copper_percentage_in_alloy_l541_54151


namespace NUMINAMATH_CALUDE_hidden_cannonball_label_l541_54179

structure CannonballPyramid where
  total_cannonballs : Nat
  labels : Finset Char
  label_count : Char → Nat
  visible_count : Char → Nat

def is_valid_pyramid (p : CannonballPyramid) : Prop :=
  p.total_cannonballs = 20 ∧
  p.labels = {'A', 'B', 'C', 'D', 'E'} ∧
  ∀ l ∈ p.labels, p.label_count l = 4 ∧
  ∀ l ∈ p.labels, p.visible_count l ≤ p.label_count l

theorem hidden_cannonball_label (p : CannonballPyramid) 
  (h_valid : is_valid_pyramid p)
  (h_visible : ∀ l ∈ p.labels, l ≠ 'D' → p.visible_count l = 4)
  (h_d_visible : p.visible_count 'D' = 3) :
  p.label_count 'D' - p.visible_count 'D' = 1 := by
sorry

end NUMINAMATH_CALUDE_hidden_cannonball_label_l541_54179


namespace NUMINAMATH_CALUDE_unique_solution_l541_54171

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the addition problem -/
def AdditionProblem (A B C : Digit) : Prop :=
  (C.val * 100 + C.val * 10 + A.val) + (B.val * 100 + 2 * 10 + B.val) = A.val * 100 + 8 * 10 + 8

theorem unique_solution :
  ∃! (A B C : Digit), AdditionProblem A B C ∧ A.val * B.val * C.val = 42 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l541_54171


namespace NUMINAMATH_CALUDE_simplify_expression_l541_54120

theorem simplify_expression : (256 : ℝ) ^ (1/4 : ℝ) * (125 : ℝ) ^ (1/3 : ℝ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l541_54120


namespace NUMINAMATH_CALUDE_divisibility_of_factorial_l541_54156

theorem divisibility_of_factorial (n : ℕ+) :
  (2011^2011 ∣ n!) → (2011^2012 ∣ n!) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_of_factorial_l541_54156


namespace NUMINAMATH_CALUDE_squares_below_line_count_l541_54101

/-- The number of squares below the line 5x + 45y = 225 in the first quadrant -/
def squares_below_line : ℕ :=
  let x_intercept : ℕ := 45
  let y_intercept : ℕ := 5
  let total_squares : ℕ := x_intercept * y_intercept
  let diagonal_squares : ℕ := x_intercept + y_intercept - 1
  let non_diagonal_squares : ℕ := total_squares - diagonal_squares
  non_diagonal_squares / 2

theorem squares_below_line_count : squares_below_line = 88 := by
  sorry

end NUMINAMATH_CALUDE_squares_below_line_count_l541_54101


namespace NUMINAMATH_CALUDE_common_roots_product_l541_54187

/-- Given two cubic equations with two common roots, prove their product is 8 -/
theorem common_roots_product (C D : ℝ) : 
  ∃ (p q r s : ℝ), 
    (p^3 + C*p + 20 = 0) ∧ 
    (q^3 + C*q + 20 = 0) ∧ 
    (r^3 + C*r + 20 = 0) ∧
    (p^3 + D*p^2 + 80 = 0) ∧ 
    (q^3 + D*q^2 + 80 = 0) ∧ 
    (s^3 + D*s^2 + 80 = 0) ∧
    (p ≠ q) ∧ (p ≠ r) ∧ (q ≠ r) ∧
    (p ≠ s) ∧ (q ≠ s) →
    p * q = 8 := by
  sorry

end NUMINAMATH_CALUDE_common_roots_product_l541_54187


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l541_54199

theorem sum_of_reciprocals_of_roots (p q : ℝ) : 
  p^2 - 17*p + 8 = 0 → q^2 - 17*q + 8 = 0 → 1/p + 1/q = 17/8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l541_54199
