import Mathlib

namespace NUMINAMATH_CALUDE_intersection_with_complement_l1306_130623

def U : Set Nat := {1,2,3,4,5,6,7}
def A : Set Nat := {2,4,5}
def B : Set Nat := {1,3,5,7}

theorem intersection_with_complement : A ∩ (U \ B) = {2,4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l1306_130623


namespace NUMINAMATH_CALUDE_gold_families_count_l1306_130695

def fundraiser (bronze_families : ℕ) (silver_families : ℕ) (gold_families : ℕ) : Prop :=
  let bronze_donation := 25
  let silver_donation := 50
  let gold_donation := 100
  let total_goal := 750
  let final_day_goal := 50
  bronze_families * bronze_donation + 
  silver_families * silver_donation + 
  gold_families * gold_donation = 
  total_goal - final_day_goal

theorem gold_families_count : 
  ∃! gold_families : ℕ, fundraiser 10 7 gold_families :=
sorry

end NUMINAMATH_CALUDE_gold_families_count_l1306_130695


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1306_130608

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℕ),
  a > 0 ∧ b > 0 ∧ c > 0 →
  a^2 + b^2 = c^2 →
  b = c - 1575 →
  a < 1991 →
  c = 1800 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1306_130608


namespace NUMINAMATH_CALUDE_region_location_l1306_130645

-- Define the line
def line (x y : ℝ) : Prop := x - 2*y + 6 = 0

-- Define the region
def region (x y : ℝ) : Prop := x - 2*y + 6 < 0

-- Theorem statement
theorem region_location :
  ∀ (x y : ℝ), region x y → 
  ∃ (x₀ y₀ : ℝ), line x₀ y₀ ∧ x < x₀ ∧ y > y₀ :=
sorry

end NUMINAMATH_CALUDE_region_location_l1306_130645


namespace NUMINAMATH_CALUDE_outside_trash_count_l1306_130681

def total_trash : ℕ := 1576
def classroom_trash : ℕ := 344

theorem outside_trash_count : total_trash - classroom_trash = 1232 := by
  sorry

end NUMINAMATH_CALUDE_outside_trash_count_l1306_130681


namespace NUMINAMATH_CALUDE_non_opaque_arrangements_l1306_130632

/-- Represents the number of glasses in the stack -/
def num_glasses : ℕ := 5

/-- Represents the number of possible rotations for each glass -/
def num_rotations : ℕ := 3

/-- Calculates the total number of possible arrangements -/
def total_arrangements : ℕ := num_glasses.factorial * num_rotations ^ (num_glasses - 1)

/-- Calculates the number of opaque arrangements -/
def opaque_arrangements : ℕ := 50 * num_glasses.factorial

/-- Theorem stating the number of non-opaque arrangements -/
theorem non_opaque_arrangements :
  total_arrangements - opaque_arrangements = 3720 :=
sorry

end NUMINAMATH_CALUDE_non_opaque_arrangements_l1306_130632


namespace NUMINAMATH_CALUDE_r_to_s_conversion_l1306_130612

/-- Given a linear relationship between r and s scales, prove that r = 48 corresponds to s = 100 -/
theorem r_to_s_conversion (r s : ℝ → ℝ) : 
  (∃ a b : ℝ, ∀ x, s x = a * x + b) →  -- Linear relationship
  s 6 = 30 →                          -- First given point
  s 24 = 60 →                         -- Second given point
  s 48 = 100 :=                       -- Conclusion to prove
by sorry

end NUMINAMATH_CALUDE_r_to_s_conversion_l1306_130612


namespace NUMINAMATH_CALUDE_set_operation_example_l1306_130657

def set_operation (M N : Set ℕ) : Set ℕ :=
  {x | x ∈ M ∪ N ∧ x ∉ M ∩ N}

theorem set_operation_example :
  let M : Set ℕ := {1, 2, 3}
  let N : Set ℕ := {2, 3, 4}
  set_operation M N = {1, 4} := by
  sorry

end NUMINAMATH_CALUDE_set_operation_example_l1306_130657


namespace NUMINAMATH_CALUDE_intersection_A_B_l1306_130617

-- Define set A
def A : Set ℝ := {x | -2 < x ∧ x < 4}

-- Define set B
def B : Set ℝ := {2, 3, 4, 5}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1306_130617


namespace NUMINAMATH_CALUDE_exponential_greater_than_trig_squared_l1306_130638

theorem exponential_greater_than_trig_squared (x : ℝ) : 
  Real.exp x + Real.exp (-x) ≥ (Real.sin x + Real.cos x)^2 := by
  sorry

end NUMINAMATH_CALUDE_exponential_greater_than_trig_squared_l1306_130638


namespace NUMINAMATH_CALUDE_initial_number_of_girls_l1306_130647

/-- The initial number of girls -/
def n : ℕ := sorry

/-- The initial average weight of the girls -/
def A : ℝ := sorry

/-- The weight of the replaced girl -/
def replaced_weight : ℝ := 40

/-- The weight of the new girl -/
def new_weight : ℝ := 80

/-- The increase in average weight -/
def avg_increase : ℝ := 2

theorem initial_number_of_girls :
  (n : ℝ) * (A + avg_increase) - n * A = new_weight - replaced_weight →
  n = 20 := by sorry

end NUMINAMATH_CALUDE_initial_number_of_girls_l1306_130647


namespace NUMINAMATH_CALUDE_triangle_side_length_expression_l1306_130698

/-- For any triangle with side lengths a, b, and c, |a+b-c|-|a-b-c| = 2a-2c -/
theorem triangle_side_length_expression (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  |a + b - c| - |a - b - c| = 2*a - 2*c := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_expression_l1306_130698


namespace NUMINAMATH_CALUDE_solution_set_f_positive_range_of_m_l1306_130689

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 2|

-- Define the theorem for part I
theorem solution_set_f_positive :
  {x : ℝ | f x > 0} = {x : ℝ | x < -3 ∨ x > 1/3} := by sorry

-- Define the theorem for part II
theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, |m + 1| ≥ f x + 3*|x - 2|) ↔ m ≤ -6 ∨ m ≥ 4 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_positive_range_of_m_l1306_130689


namespace NUMINAMATH_CALUDE_power_of_power_negative_l1306_130606

theorem power_of_power_negative (a : ℝ) : -(a^3)^4 = -a^12 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_negative_l1306_130606


namespace NUMINAMATH_CALUDE_sale_price_calculation_l1306_130628

def ticket_price : ℝ := 25
def discount_rate : ℝ := 0.25

theorem sale_price_calculation :
  ticket_price * (1 - discount_rate) = 18.75 := by
  sorry

end NUMINAMATH_CALUDE_sale_price_calculation_l1306_130628


namespace NUMINAMATH_CALUDE_root_nature_depends_on_k_l1306_130637

theorem root_nature_depends_on_k :
  ∀ k : ℝ, ∃ Δ : ℝ, 
    (Δ = 1 + 4*k) ∧ 
    (Δ < 0 → (∀ x : ℝ, (x - 1) * (x - 2) ≠ k)) ∧
    (Δ = 0 → (∃! x : ℝ, (x - 1) * (x - 2) = k)) ∧
    (Δ > 0 → (∃ x y : ℝ, x ≠ y ∧ (x - 1) * (x - 2) = k ∧ (y - 1) * (y - 2) = k)) :=
by sorry


end NUMINAMATH_CALUDE_root_nature_depends_on_k_l1306_130637


namespace NUMINAMATH_CALUDE_intersection_distance_l1306_130690

/-- Two lines intersecting at 60 degrees --/
structure IntersectingLines :=
  (angle : ℝ)
  (h_angle : angle = 60)

/-- Points on the intersecting lines --/
structure PointsOnLines (l : IntersectingLines) :=
  (A B : ℝ × ℝ)
  (dist_initial : ℝ)
  (dist_after_move : ℝ)
  (move_distance : ℝ)
  (h_initial_dist : dist_initial = 31)
  (h_after_move_dist : dist_after_move = 21)
  (h_move_distance : move_distance = 20)

/-- The theorem to be proved --/
theorem intersection_distance (l : IntersectingLines) (p : PointsOnLines l) :
  ∃ (dist_A dist_B : ℝ),
    dist_A = 35 ∧ dist_B = 24 ∧
    (dist_A - p.move_distance)^2 + dist_B^2 = p.dist_initial^2 ∧
    dist_A^2 + dist_B^2 = p.dist_after_move^2 + p.move_distance^2 :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_l1306_130690


namespace NUMINAMATH_CALUDE_mushroom_collection_l1306_130656

/-- A function that returns the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem mushroom_collection :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ sum_of_digits n = 14 ∧ n % 50 = 0 :=
by
  -- The proof would go here
  sorry

#eval sum_of_digits 950  -- Should output 14
#eval 950 % 50           -- Should output 0

end NUMINAMATH_CALUDE_mushroom_collection_l1306_130656


namespace NUMINAMATH_CALUDE_dropped_class_hours_l1306_130635

/-- Calculates the remaining class hours after dropping a class -/
def remaining_class_hours (initial_classes : ℕ) (hours_per_class : ℕ) (dropped_classes : ℕ) : ℕ :=
  (initial_classes - dropped_classes) * hours_per_class

/-- Theorem: Given 4 classes of 2 hours each, dropping 1 class results in 6 hours of classes -/
theorem dropped_class_hours : remaining_class_hours 4 2 1 = 6 := by
  sorry

end NUMINAMATH_CALUDE_dropped_class_hours_l1306_130635


namespace NUMINAMATH_CALUDE_hexagon_area_is_19444_l1306_130671

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (positive_a : a > 0)
  (positive_b : b > 0)
  (positive_c : c > 0)
  (triangle_inequality_ab : a + b > c)
  (triangle_inequality_bc : b + c > a)
  (triangle_inequality_ca : c + a > b)

-- Define the specific triangle with sides 13, 14, and 15
def specific_triangle : Triangle :=
  { a := 13
  , b := 14
  , c := 15
  , positive_a := by norm_num
  , positive_b := by norm_num
  , positive_c := by norm_num
  , triangle_inequality_ab := by norm_num
  , triangle_inequality_bc := by norm_num
  , triangle_inequality_ca := by norm_num }

-- Define the area of the hexagon A₅A₆B₅B₆C₅C₆
def hexagon_area (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem hexagon_area_is_19444 :
  hexagon_area specific_triangle = 19444 := by sorry

end NUMINAMATH_CALUDE_hexagon_area_is_19444_l1306_130671


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1306_130675

theorem polynomial_divisibility (n : ℤ) : 
  ∃ k : ℤ, (n + 7)^2 - n^2 = 7 * k := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1306_130675


namespace NUMINAMATH_CALUDE_negative_fraction_multiplication_l1306_130643

theorem negative_fraction_multiplication :
  ((-144 : ℤ) / (-36 : ℤ)) * 3 = 12 := by sorry

end NUMINAMATH_CALUDE_negative_fraction_multiplication_l1306_130643


namespace NUMINAMATH_CALUDE_kolya_cannot_descend_l1306_130693

/-- Represents the possible jump sizes Kolya can make. -/
inductive JumpSize
  | six
  | seven
  | eight

/-- Represents a sequence of jumps Kolya makes. -/
def JumpSequence := List JumpSize

/-- The total number of steps on the ladder. -/
def totalSteps : Nat := 100

/-- Converts a JumpSize to its corresponding natural number. -/
def jumpSizeToNat (j : JumpSize) : Nat :=
  match j with
  | JumpSize.six => 6
  | JumpSize.seven => 7
  | JumpSize.eight => 8

/-- Calculates the position after a sequence of jumps. -/
def finalPosition (jumps : JumpSequence) : Int :=
  totalSteps - (jumps.map jumpSizeToNat).sum

/-- Checks if a sequence of jumps results in unique positions. -/
def hasUniquePositions (jumps : JumpSequence) : Prop :=
  let positions := List.scanl (fun pos jump => pos - jumpSizeToNat jump) totalSteps jumps
  positions.Nodup

/-- Theorem stating that Kolya cannot descend the ladder under the given conditions. -/
theorem kolya_cannot_descend :
  ¬∃ (jumps : JumpSequence), finalPosition jumps = 0 ∧ hasUniquePositions jumps :=
sorry


end NUMINAMATH_CALUDE_kolya_cannot_descend_l1306_130693


namespace NUMINAMATH_CALUDE_supplier_A_lower_variance_l1306_130680

-- Define the purity data for Supplier A
def purity_A : List Nat := [72, 73, 74, 74, 74, 74, 74, 75, 75, 75, 76, 76, 76, 78, 79]

-- Define the purity data for Supplier B
def purity_B : List Nat := [72, 75, 72, 75, 78, 77, 73, 75, 76, 77, 71, 78, 79, 72, 75]

-- Define the statistical measures for Supplier A
def mean_A : Nat := 75
def median_A : Nat := 75
def mode_A : Nat := 74
def variance_A : Float := 3.7

-- Define the statistical measures for Supplier B
def mean_B : Nat := 75
def median_B : Nat := 75
def mode_B : Nat := 75
def variance_B : Float := 6.0

-- Theorem statement
theorem supplier_A_lower_variance :
  variance_A < variance_B ∧
  List.length purity_A = 15 ∧
  List.length purity_B = 15 ∧
  mean_A = mean_B ∧
  median_A = median_B :=
sorry

end NUMINAMATH_CALUDE_supplier_A_lower_variance_l1306_130680


namespace NUMINAMATH_CALUDE_circle_definition_l1306_130665

/-- Definition of a circle in a plane -/
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

/-- Theorem: The set of all points in a plane at a fixed distance from a given point forms a circle -/
theorem circle_definition (center : ℝ × ℝ) (radius : ℝ) :
  {p : ℝ × ℝ | Real.sqrt ((p.1 - center.1)^2 + (p.2 - center.2)^2) = radius} = Circle center radius :=
by sorry

end NUMINAMATH_CALUDE_circle_definition_l1306_130665


namespace NUMINAMATH_CALUDE_shannons_to_olivias_scoops_ratio_l1306_130697

/-- Represents the number of scoops in a carton of ice cream -/
def scoops_per_carton : ℕ := 10

/-- Represents the number of cartons Mary has -/
def marys_cartons : ℕ := 3

/-- Represents the number of scoops Ethan wants -/
def ethans_scoops : ℕ := 2

/-- Represents the number of scoops Lucas, Danny, and Connor want in total -/
def lucas_danny_connor_scoops : ℕ := 6

/-- Represents the number of scoops Olivia wants -/
def olivias_scoops : ℕ := 2

/-- Represents the number of scoops left -/
def scoops_left : ℕ := 16

/-- Theorem stating that the ratio of Shannon's scoops to Olivia's scoops is 2:1 -/
theorem shannons_to_olivias_scoops_ratio : 
  ∃ (shannons_scoops : ℕ), 
    shannons_scoops = marys_cartons * scoops_per_carton - 
      (ethans_scoops + lucas_danny_connor_scoops + olivias_scoops + scoops_left) ∧
    shannons_scoops = 2 * olivias_scoops :=
by sorry

end NUMINAMATH_CALUDE_shannons_to_olivias_scoops_ratio_l1306_130697


namespace NUMINAMATH_CALUDE_max_value_of_f_on_I_l1306_130669

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 2*x - 1

-- Define the closed interval [0, 3]
def I : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}

-- Statement of the theorem
theorem max_value_of_f_on_I :
  ∃ (c : ℝ), c ∈ I ∧ ∀ (x : ℝ), x ∈ I → f x ≤ f c ∧ f c = 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_on_I_l1306_130669


namespace NUMINAMATH_CALUDE_subtracted_amount_l1306_130639

theorem subtracted_amount (N : ℝ) (A : ℝ) : 
  N = 100 → 0.7 * N - A = 30 → A = 40 := by sorry

end NUMINAMATH_CALUDE_subtracted_amount_l1306_130639


namespace NUMINAMATH_CALUDE_greater_than_implies_half_greater_than_l1306_130605

theorem greater_than_implies_half_greater_than (a b : ℝ) (h : a > b) : a / 2 > b / 2 := by
  sorry

end NUMINAMATH_CALUDE_greater_than_implies_half_greater_than_l1306_130605


namespace NUMINAMATH_CALUDE_ben_whitewashed_length_l1306_130629

theorem ben_whitewashed_length (total_length : ℝ) (remaining_length : ℝ)
  (h1 : total_length = 100)
  (h2 : remaining_length = 48)
  (h3 : ∃ x : ℝ, 
    remaining_length = total_length - x - 
    (1/5) * (total_length - x) - 
    (1/3) * (total_length - x - (1/5) * (total_length - x))) :
  ∃ x : ℝ, x = 10 ∧ 
    remaining_length = total_length - x - 
    (1/5) * (total_length - x) - 
    (1/3) * (total_length - x - (1/5) * (total_length - x)) :=
by sorry

end NUMINAMATH_CALUDE_ben_whitewashed_length_l1306_130629


namespace NUMINAMATH_CALUDE_equation_represents_hyperbola_l1306_130660

/-- Represents a conic section -/
inductive ConicSection
  | Parabola
  | Circle
  | Ellipse
  | Hyperbola
  | Point
  | Line
  | TwoLines
  | Empty

/-- Determines the type of conic section given by the equation ax² + bxy + cy² + dx + ey + f = 0 -/
def determineConicSection (a b c d e f : ℝ) : ConicSection :=
  sorry

/-- The equation x² - 4y² + 6x - 8 = 0 represents a hyperbola -/
theorem equation_represents_hyperbola :
  determineConicSection 1 0 (-4) 6 0 (-8) = ConicSection.Hyperbola :=
sorry

end NUMINAMATH_CALUDE_equation_represents_hyperbola_l1306_130660


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1306_130630

/-- An arithmetic sequence with its partial sums. -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  S : ℕ → ℚ  -- Partial sums
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- The main theorem. -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence)
    (h1 : seq.S 4 = 8)
    (h2 : seq.S 8 = 20) :
    seq.a 11 + seq.a 12 + seq.a 13 + seq.a 14 = 18 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1306_130630


namespace NUMINAMATH_CALUDE_sum_product_equality_l1306_130692

theorem sum_product_equality : (153 + 39 + 27 + 21) * 2 = 480 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_equality_l1306_130692


namespace NUMINAMATH_CALUDE_average_weight_after_student_left_l1306_130641

theorem average_weight_after_student_left (initial_count : ℕ) (left_weight : ℝ) 
  (remaining_count : ℕ) (weight_increase : ℝ) (final_average : ℝ) : 
  initial_count = 60 →
  left_weight = 45 →
  remaining_count = 59 →
  weight_increase = 0.2 →
  final_average = 57 →
  (initial_count : ℝ) * (final_average - weight_increase) = 
    (remaining_count : ℝ) * final_average + left_weight := by
  sorry

end NUMINAMATH_CALUDE_average_weight_after_student_left_l1306_130641


namespace NUMINAMATH_CALUDE_perimeter_circumference_ratio_l1306_130640

/-- An isosceles trapezoid with an inscribed circle -/
structure IsoscelesTrapezoidWithInscribedCircle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The distance from the center of the circle to the intersection of diagonals -/
  d : ℝ
  /-- Condition that d is 3/5 of r -/
  h_d_ratio : d = 3/5 * r

/-- The perimeter of the trapezoid -/
def perimeter (t : IsoscelesTrapezoidWithInscribedCircle) : ℝ := sorry

/-- The circumference of the inscribed circle -/
def circumference (t : IsoscelesTrapezoidWithInscribedCircle) : ℝ := sorry

theorem perimeter_circumference_ratio 
  (t : IsoscelesTrapezoidWithInscribedCircle) : 
  perimeter t / circumference t = 5 / Real.pi := by sorry

end NUMINAMATH_CALUDE_perimeter_circumference_ratio_l1306_130640


namespace NUMINAMATH_CALUDE_inverse_of_100_mod_101_l1306_130651

theorem inverse_of_100_mod_101 : ∃ x : ℕ, 0 ≤ x ∧ x ≤ 100 ∧ (100 * x) % 101 = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_100_mod_101_l1306_130651


namespace NUMINAMATH_CALUDE_geometric_sequence_shift_l1306_130691

/-- 
Given a geometric sequence {a_n} with common ratio q ≠ 1, 
if {a_n + c} is also a geometric sequence, then c = 0.
-/
theorem geometric_sequence_shift (a : ℕ → ℝ) (q c : ℝ) : 
  (∀ n, a (n + 1) = q * a n) →  -- {a_n} is a geometric sequence
  q ≠ 1 →  -- common ratio q ≠ 1
  (∃ r, ∀ n, (a (n + 1) + c) = r * (a n + c)) →  -- {a_n + c} is also a geometric sequence
  c = 0 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_shift_l1306_130691


namespace NUMINAMATH_CALUDE_miguels_wall_paint_area_l1306_130634

/-- The area to be painted on a wall with given dimensions and a window -/
def area_to_paint (wall_height wall_length window_side : ℝ) : ℝ :=
  wall_height * wall_length - window_side * window_side

/-- Theorem stating the area to be painted for Miguel's wall -/
theorem miguels_wall_paint_area :
  area_to_paint 10 15 3 = 141 := by
  sorry

end NUMINAMATH_CALUDE_miguels_wall_paint_area_l1306_130634


namespace NUMINAMATH_CALUDE_litter_patrol_theorem_l1306_130667

/-- The total number of litter items picked up by the Litter Patrol -/
def total_litter : ℕ := 40

/-- The number of non-miscellaneous items (glass bottles + aluminum cans + plastic bags) -/
def non_misc_items : ℕ := 30

/-- The percentage of non-miscellaneous items in the total litter -/
def non_misc_percentage : ℚ := 3/4

theorem litter_patrol_theorem :
  (non_misc_items : ℚ) / non_misc_percentage = total_litter := by sorry

end NUMINAMATH_CALUDE_litter_patrol_theorem_l1306_130667


namespace NUMINAMATH_CALUDE_students_in_neither_course_l1306_130684

theorem students_in_neither_course (total : ℕ) (coding : ℕ) (robotics : ℕ) (both : ℕ)
  (h1 : total = 150)
  (h2 : coding = 90)
  (h3 : robotics = 70)
  (h4 : both = 25) :
  total - (coding + robotics - both) = 15 := by
  sorry

end NUMINAMATH_CALUDE_students_in_neither_course_l1306_130684


namespace NUMINAMATH_CALUDE_fraction_value_l1306_130607

theorem fraction_value (x y : ℚ) (hx : x = 7/9) (hy : y = 3/5) :
  (7*x + 5*y) / (63*x*y) = 20/69 := by sorry

end NUMINAMATH_CALUDE_fraction_value_l1306_130607


namespace NUMINAMATH_CALUDE_smallest_representable_integer_l1306_130618

theorem smallest_representable_integer :
  ∃ (m n : ℕ+), 11 = 36 * m - 5 * n ∧
  ∀ (k : ℕ+) (m' n' : ℕ+), k < 11 → k ≠ 36 * m' - 5 * n' :=
sorry

end NUMINAMATH_CALUDE_smallest_representable_integer_l1306_130618


namespace NUMINAMATH_CALUDE_human_habitable_fraction_l1306_130659

theorem human_habitable_fraction (water_fraction : ℚ) (inhabitable_fraction : ℚ) (agriculture_fraction : ℚ)
  (h1 : water_fraction = 3/5)
  (h2 : inhabitable_fraction = 2/3)
  (h3 : agriculture_fraction = 1/2) :
  (1 - water_fraction) * inhabitable_fraction * (1 - agriculture_fraction) = 2/15 := by
  sorry

end NUMINAMATH_CALUDE_human_habitable_fraction_l1306_130659


namespace NUMINAMATH_CALUDE_political_science_majors_l1306_130661

/-- Represents the number of applicants who majored in political science -/
def P : ℕ := sorry

theorem political_science_majors :
  let total_applicants : ℕ := 40
  let high_gpa : ℕ := 20
  let not_ps_low_gpa : ℕ := 10
  let ps_high_gpa : ℕ := 5
  P = 15 := by sorry

end NUMINAMATH_CALUDE_political_science_majors_l1306_130661


namespace NUMINAMATH_CALUDE_smallest_special_integer_l1306_130696

theorem smallest_special_integer (N : ℕ) : N = 793 ↔ 
  N > 1 ∧
  (∀ M : ℕ, M > 1 → 
    (M ≡ 1 [ZMOD 8] ∧
     M ≡ 1 [ZMOD 9] ∧
     (∃ k : ℕ, 8^k ≤ M ∧ M < 2 * 8^k) ∧
     (∃ m : ℕ, 9^m ≤ M ∧ M < 2 * 9^m)) →
    N ≤ M) ∧
  N ≡ 1 [ZMOD 8] ∧
  N ≡ 1 [ZMOD 9] ∧
  (∃ k : ℕ, 8^k ≤ N ∧ N < 2 * 8^k) ∧
  (∃ m : ℕ, 9^m ≤ N ∧ N < 2 * 9^m) :=
by sorry

end NUMINAMATH_CALUDE_smallest_special_integer_l1306_130696


namespace NUMINAMATH_CALUDE_air_conditioner_power_consumption_l1306_130687

/-- Power consumption of three air conditioners over specified periods -/
theorem air_conditioner_power_consumption 
  (power_A : Real) (hours_A : Real) (days_A : Real)
  (power_B : Real) (hours_B : Real) (days_B : Real)
  (power_C : Real) (hours_C : Real) (days_C : Real) :
  power_A = 7.2 →
  power_B = 9.6 →
  power_C = 12 →
  hours_A = 6 →
  hours_B = 4 →
  hours_C = 3 →
  days_A = 5 →
  days_B = 7 →
  days_C = 10 →
  (power_A / 8 * hours_A * days_A) +
  (power_B / 10 * hours_B * days_B) +
  (power_C / 12 * hours_C * days_C) = 83.88 := by
  sorry

#eval (7.2 / 8 * 6 * 5) + (9.6 / 10 * 4 * 7) + (12 / 12 * 3 * 10)

end NUMINAMATH_CALUDE_air_conditioner_power_consumption_l1306_130687


namespace NUMINAMATH_CALUDE_sum_9000_eq_1355_l1306_130650

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  /-- The sum of the first 3000 terms -/
  sum_3000 : ℝ
  /-- The sum of the first 6000 terms -/
  sum_6000 : ℝ
  /-- The sum of the first 3000 terms is 500 -/
  sum_3000_eq : sum_3000 = 500
  /-- The sum of the first 6000 terms is 950 -/
  sum_6000_eq : sum_6000 = 950

/-- The sum of the first 9000 terms of the geometric sequence is 1355 -/
theorem sum_9000_eq_1355 (seq : GeometricSequence) : ℝ := by
  sorry

end NUMINAMATH_CALUDE_sum_9000_eq_1355_l1306_130650


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l1306_130644

/-- The discriminant of a quadratic equation ax^2 + bx + c -/
def discriminant (a b c : ℚ) : ℚ := b^2 - 4*a*c

/-- The coefficients of the quadratic equation 5x^2 + (5 + 1/2)x + 1/2 -/
def a : ℚ := 5
def b : ℚ := 5 + 1/2
def c : ℚ := 1/2

theorem quadratic_discriminant :
  discriminant a b c = 81/4 := by sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l1306_130644


namespace NUMINAMATH_CALUDE_log_equation_solution_l1306_130609

theorem log_equation_solution (x : ℝ) :
  Real.log x / Real.log 9 = 2.4 → x = (81 ^ (1/5)) ^ 6 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1306_130609


namespace NUMINAMATH_CALUDE_milk_calculation_l1306_130668

/-- The initial amount of milk in quarts -/
def initial_milk : ℝ := 1000

/-- The percentage of butterfat in the initial milk -/
def initial_butterfat_percent : ℝ := 4

/-- The percentage of butterfat in the final milk -/
def final_butterfat_percent : ℝ := 3

/-- The amount of cream separated in quarts -/
def separated_cream : ℝ := 50

/-- The percentage of butterfat in the separated cream -/
def cream_butterfat_percent : ℝ := 23

theorem milk_calculation :
  initial_milk = 1000 ∧
  initial_butterfat_percent / 100 * initial_milk =
    final_butterfat_percent / 100 * (initial_milk - separated_cream) +
    cream_butterfat_percent / 100 * separated_cream :=
by sorry

end NUMINAMATH_CALUDE_milk_calculation_l1306_130668


namespace NUMINAMATH_CALUDE_billy_coins_problem_l1306_130654

theorem billy_coins_problem (total_coins : Nat) (quarter_piles : Nat) (dime_piles : Nat) 
  (h1 : total_coins = 20)
  (h2 : quarter_piles = 2)
  (h3 : dime_piles = 3) :
  ∃! coins_per_pile : Nat, 
    coins_per_pile > 0 ∧ 
    quarter_piles * coins_per_pile + dime_piles * coins_per_pile = total_coins ∧
    coins_per_pile = 4 := by
  sorry

end NUMINAMATH_CALUDE_billy_coins_problem_l1306_130654


namespace NUMINAMATH_CALUDE_complex_cube_sum_ratio_l1306_130613

theorem complex_cube_sum_ratio (x y z : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 30)
  (h_eq : (x - y)^2 + (x - z)^2 + (y - z)^2 = 3*x*y*z) :
  (x^3 + y^3 + z^3) / (x*y*z) = 48 := by
sorry

end NUMINAMATH_CALUDE_complex_cube_sum_ratio_l1306_130613


namespace NUMINAMATH_CALUDE_systematic_sampling_example_l1306_130662

/-- Systematic sampling function -/
def systematicSample (totalItems : ℕ) (numGroups : ℕ) (startGroup : ℕ) (startNum : ℕ) (targetGroup : ℕ) : ℕ :=
  startNum + (targetGroup - startGroup) * (totalItems / numGroups)

/-- Theorem: In systematic sampling of 200 items into 40 groups, 
    if the 5th group draws 24, then the 9th group draws 44 -/
theorem systematic_sampling_example :
  systematicSample 200 40 5 24 9 = 44 := by
  sorry

#eval systematicSample 200 40 5 24 9

end NUMINAMATH_CALUDE_systematic_sampling_example_l1306_130662


namespace NUMINAMATH_CALUDE_gcd_13013_15015_l1306_130621

theorem gcd_13013_15015 : Nat.gcd 13013 15015 = 1001 := by
  sorry

end NUMINAMATH_CALUDE_gcd_13013_15015_l1306_130621


namespace NUMINAMATH_CALUDE_solution_valid_l1306_130624

open Real

variables (t : ℝ) (x y : ℝ → ℝ) (C₁ C₂ : ℝ)

def diff_eq_system (x y : ℝ → ℝ) : Prop :=
  (∀ t, deriv x t = (y t + exp (x t)) / (y t + exp t)) ∧
  (∀ t, deriv y t = (y t ^ 2 - exp (x t + t)) / (y t + exp t))

def general_solution (x y : ℝ → ℝ) (C₁ C₂ : ℝ) : Prop :=
  (∀ t, exp (-t) * y t + x t = C₁) ∧
  (∀ t, exp (-(x t)) * y t + t = C₂)

theorem solution_valid :
  diff_eq_system x y → general_solution x y C₁ C₂ → diff_eq_system x y :=
sorry

end NUMINAMATH_CALUDE_solution_valid_l1306_130624


namespace NUMINAMATH_CALUDE_system_of_equations_l1306_130631

theorem system_of_equations (p t j x y : ℝ) : 
  j = 0.75 * p →
  j = 0.8 * t →
  t = p * (1 - t / 100) →
  x = 0.1 * t →
  y = 0.5 * j →
  x + y = 12 →
  t = 24 := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_l1306_130631


namespace NUMINAMATH_CALUDE_x_squared_is_quadratic_l1306_130685

/-- Definition of a quadratic equation in one variable -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x^2 = 0 -/
def f (x : ℝ) : ℝ := x^2

/-- Theorem stating that x^2 = 0 is a quadratic equation in one variable -/
theorem x_squared_is_quadratic : is_quadratic_equation f :=
sorry

end NUMINAMATH_CALUDE_x_squared_is_quadratic_l1306_130685


namespace NUMINAMATH_CALUDE_comparison_inequalities_l1306_130673

theorem comparison_inequalities :
  (Real.sqrt 37 > 6) ∧ ((Real.sqrt 5 - 1) / 2 > 1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_comparison_inequalities_l1306_130673


namespace NUMINAMATH_CALUDE_renovation_project_materials_l1306_130626

theorem renovation_project_materials (sand dirt cement : ℝ) 
  (h_sand : sand = 0.17)
  (h_dirt : dirt = 0.33)
  (h_cement : cement = 0.17) :
  sand + dirt + cement = 0.67 := by
  sorry

end NUMINAMATH_CALUDE_renovation_project_materials_l1306_130626


namespace NUMINAMATH_CALUDE_cupboard_has_35_slots_l1306_130682

/-- Represents a cupboard with shelves and slots -/
structure Cupboard where
  shelves : ℕ
  slots_per_shelf : ℕ

/-- Represents the position of a plate in the cupboard -/
structure PlatePosition where
  shelf_from_top : ℕ
  shelf_from_bottom : ℕ
  slot_from_left : ℕ
  slot_from_right : ℕ

/-- Calculates the total number of slots in a cupboard -/
def total_slots (c : Cupboard) : ℕ := c.shelves * c.slots_per_shelf

/-- Theorem: Given the position of a plate, the cupboard has 35 slots -/
theorem cupboard_has_35_slots (pos : PlatePosition) 
  (h1 : pos.shelf_from_top = 2)
  (h2 : pos.shelf_from_bottom = 4)
  (h3 : pos.slot_from_left = 1)
  (h4 : pos.slot_from_right = 7) :
  ∃ c : Cupboard, total_slots c = 35 := by
  sorry

end NUMINAMATH_CALUDE_cupboard_has_35_slots_l1306_130682


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1306_130601

theorem trigonometric_identity (α : Real) (m : Real) (h : Real.tan α = m) :
  Real.sin (π/4 + α)^2 - Real.sin (π/6 - α)^2 - Real.cos (5*π/12) * Real.sin (5*π/12 - 2*α) = 2*m / (1 + m^2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1306_130601


namespace NUMINAMATH_CALUDE_line_not_in_second_quadrant_iff_l1306_130676

/-- A line that does not pass through the second quadrant -/
def LineNotInSecondQuadrant (a : ℝ) : Prop :=
  ∀ x y : ℝ, (a - 2) * y = (3 * a - 1) * x - 1 → (x ≤ 0 → y ≤ 0)

/-- The main theorem: characterization of a for which the line doesn't pass through the second quadrant -/
theorem line_not_in_second_quadrant_iff (a : ℝ) :
  LineNotInSecondQuadrant a ↔ a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_line_not_in_second_quadrant_iff_l1306_130676


namespace NUMINAMATH_CALUDE_logarithm_expression_equality_l1306_130633

-- Define the logarithm function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Define the logarithm with arbitrary base
noncomputable def log (b x : ℝ) : ℝ := Real.log x / Real.log b

-- State the theorem
theorem logarithm_expression_equality :
  (lg 5)^2 + lg 2 * lg 50 - log 8 9 * log 27 32 = -1/9 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_expression_equality_l1306_130633


namespace NUMINAMATH_CALUDE_coefficient_sum_is_five_sixths_l1306_130658

/-- A polynomial function from ℝ to ℝ -/
def PolynomialFunction := ℝ → ℝ

/-- The property that f(x) - f(x-2) = (2x-1)^2 for all x -/
def SatisfiesEquation (f : PolynomialFunction) : Prop :=
  ∀ x, f x - f (x - 2) = (2 * x - 1)^2

/-- The coefficient of x^2 in a polynomial function -/
def CoefficientOfXSquared (f : PolynomialFunction) : ℝ := sorry

/-- The coefficient of x in a polynomial function -/
def CoefficientOfX (f : PolynomialFunction) : ℝ := sorry

theorem coefficient_sum_is_five_sixths (f : PolynomialFunction) 
  (h : SatisfiesEquation f) : 
  CoefficientOfXSquared f + CoefficientOfX f = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_sum_is_five_sixths_l1306_130658


namespace NUMINAMATH_CALUDE_q_div_p_equals_225_l1306_130649

def total_cards : ℕ := 50
def num_range : Set ℕ := Finset.range 10
def cards_per_num : ℕ := 5
def drawn_cards : ℕ := 5

def p : ℚ := 10 / Nat.choose total_cards drawn_cards
def q : ℚ := (10 * 9 * cards_per_num * cards_per_num) / Nat.choose total_cards drawn_cards

theorem q_div_p_equals_225 : q / p = 225 := by sorry

end NUMINAMATH_CALUDE_q_div_p_equals_225_l1306_130649


namespace NUMINAMATH_CALUDE_biased_coin_expected_value_l1306_130686

/-- The expected value of winnings for a biased coin flip -/
theorem biased_coin_expected_value :
  let p_head : ℚ := 1/4  -- Probability of getting a head
  let p_tail : ℚ := 3/4  -- Probability of getting a tail
  let win_head : ℚ := 4  -- Amount won for flipping a head
  let lose_tail : ℚ := 3 -- Amount lost for flipping a tail
  p_head * win_head - p_tail * lose_tail = -5/4 := by
sorry

end NUMINAMATH_CALUDE_biased_coin_expected_value_l1306_130686


namespace NUMINAMATH_CALUDE_intersection_point_is_solution_l1306_130655

/-- Given two linear functions that intersect at a specific point,
    prove that this point is the solution to the system of equations. -/
theorem intersection_point_is_solution (b : ℝ) :
  (∃ (x y : ℝ), y = 3 * x - 5 ∧ y = 2 * x + b) →
  (1 : ℝ) = 3 * (1 : ℝ) - 5 →
  (-2 : ℝ) = 2 * (1 : ℝ) + b →
  (∀ (x y : ℝ), y = 3 * x - 5 ∧ y = 2 * x + b → x = 1 ∧ y = -2) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_is_solution_l1306_130655


namespace NUMINAMATH_CALUDE_highlighter_spend_l1306_130620

def total_money : ℕ := 100
def heaven_spend : ℕ := 30
def eraser_price : ℕ := 4
def eraser_count : ℕ := 10

theorem highlighter_spend :
  total_money - heaven_spend - (eraser_price * eraser_count) = 30 := by
  sorry

end NUMINAMATH_CALUDE_highlighter_spend_l1306_130620


namespace NUMINAMATH_CALUDE_log_sin_in_terms_of_m_n_l1306_130625

theorem log_sin_in_terms_of_m_n (α m n : Real) 
  (h1 : 0 < α) (h2 : α < π/2)
  (h3 : Real.log (1 + Real.cos α) = m)
  (h4 : Real.log (1 / (1 - Real.cos α)) = n) :
  Real.log (Real.sin α) = (1/2) * (m - n) := by
  sorry

end NUMINAMATH_CALUDE_log_sin_in_terms_of_m_n_l1306_130625


namespace NUMINAMATH_CALUDE_vertex_difference_hexagonal_pentagonal_prism_l1306_130652

/-- The number of vertices in a regular polygon. -/
def verticesInPolygon (sides : ℕ) : ℕ := sides

/-- The number of vertices in a prism with regular polygonal bases. -/
def verticesInPrism (baseSides : ℕ) : ℕ := 2 * (verticesInPolygon baseSides)

/-- The difference between the number of vertices of a hexagonal prism and a pentagonal prism. -/
theorem vertex_difference_hexagonal_pentagonal_prism : 
  verticesInPrism 6 - verticesInPrism 5 = 2 := by
  sorry


end NUMINAMATH_CALUDE_vertex_difference_hexagonal_pentagonal_prism_l1306_130652


namespace NUMINAMATH_CALUDE_five_nine_difference_l1306_130678

/-- Count of a specific digit in page numbers from 1 to n -/
def digitCount (digit : Nat) (n : Nat) : Nat :=
  sorry

/-- The difference between the count of 5's and 9's in page numbers from 1 to 600 -/
theorem five_nine_difference : digitCount 5 600 - digitCount 9 600 = 100 := by
  sorry

end NUMINAMATH_CALUDE_five_nine_difference_l1306_130678


namespace NUMINAMATH_CALUDE_range_of_m_for_inequality_l1306_130683

theorem range_of_m_for_inequality (m : ℝ) : 
  (∃ x : ℝ, Real.sqrt ((x + m) ^ 2) + Real.sqrt ((x - 1) ^ 2) ≤ 3) ↔ 
  -4 ≤ m ∧ m ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_for_inequality_l1306_130683


namespace NUMINAMATH_CALUDE_max_area_rectangle_perimeter_100_l1306_130627

/-- The maximum area of a rectangle with perimeter 100 and integer side lengths --/
theorem max_area_rectangle_perimeter_100 :
  ∃ (w h : ℕ), w + h = 50 ∧ w * h = 625 ∧ 
  ∀ (x y : ℕ), x + y = 50 → x * y ≤ 625 := by
  sorry

end NUMINAMATH_CALUDE_max_area_rectangle_perimeter_100_l1306_130627


namespace NUMINAMATH_CALUDE_discount_profit_theorem_l1306_130610

/-- Given a discount percentage and a profit percentage without discount,
    calculate the profit percentage with the discount. -/
def profit_with_discount (discount : ℝ) (profit_without_discount : ℝ) : ℝ :=
  (1 + profit_without_discount) * (1 - discount) - 1

/-- Theorem stating that with a 4% discount and 25% profit without discount,
    the profit percentage with discount is 20%. -/
theorem discount_profit_theorem :
  profit_with_discount 0.04 0.25 = 0.20 := by
  sorry

#eval profit_with_discount 0.04 0.25

end NUMINAMATH_CALUDE_discount_profit_theorem_l1306_130610


namespace NUMINAMATH_CALUDE_lucy_money_problem_l1306_130642

theorem lucy_money_problem (initial_money : ℚ) : 
  (initial_money * (2/3) * (3/4) = 15) → initial_money = 30 := by
  sorry

end NUMINAMATH_CALUDE_lucy_money_problem_l1306_130642


namespace NUMINAMATH_CALUDE_area_not_above_y_axis_equals_total_area_l1306_130622

/-- Parallelogram PQRS with given vertices -/
structure Parallelogram where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  S : ℝ × ℝ

/-- The specific parallelogram from the problem -/
def PQRS : Parallelogram :=
  { P := (-1, 5)
    Q := (2, -3)
    R := (-5, -3)
    S := (-8, 5) }

/-- Area of a parallelogram -/
def parallelogramArea (p : Parallelogram) : ℝ :=
  sorry

/-- Area of the part of the parallelogram not above the y-axis -/
def areaNotAboveYAxis (p : Parallelogram) : ℝ :=
  sorry

/-- Theorem stating that the area not above the y-axis equals the total area -/
theorem area_not_above_y_axis_equals_total_area :
  areaNotAboveYAxis PQRS = parallelogramArea PQRS :=
sorry

end NUMINAMATH_CALUDE_area_not_above_y_axis_equals_total_area_l1306_130622


namespace NUMINAMATH_CALUDE_least_positive_t_for_geometric_progression_l1306_130694

open Real

theorem least_positive_t_for_geometric_progression :
  ∃ (t : ℝ), t > 0 ∧
  (∀ (α : ℝ), 0 < α → α < π / 3 →
    ∃ (r : ℝ), r > 0 ∧
    (arcsin (sin (3 * α)) * r = arcsin (sin (6 * α))) ∧
    (arcsin (sin (6 * α)) * r = arccos (cos (10 * α))) ∧
    (arccos (cos (10 * α)) * r = arcsin (sin (t * α)))) ∧
  (∀ (t' : ℝ), t' > 0 →
    (∀ (α : ℝ), 0 < α → α < π / 3 →
      ∃ (r : ℝ), r > 0 ∧
      (arcsin (sin (3 * α)) * r = arcsin (sin (6 * α))) ∧
      (arcsin (sin (6 * α)) * r = arccos (cos (10 * α))) ∧
      (arccos (cos (10 * α)) * r = arcsin (sin (t' * α)))) →
    t ≤ t') ∧
  t = 10 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_t_for_geometric_progression_l1306_130694


namespace NUMINAMATH_CALUDE_concentric_circles_area_l1306_130603

theorem concentric_circles_area (r : ℝ) (h : r > 0) : 
  2 * π * r + 2 * π * (2 * r) = 36 * π → 
  π * (2 * r)^2 - π * r^2 = 108 * π := by
sorry

end NUMINAMATH_CALUDE_concentric_circles_area_l1306_130603


namespace NUMINAMATH_CALUDE_two_different_color_chips_probability_l1306_130663

/-- Represents the colors of chips in the bag -/
inductive ChipColor
  | Blue
  | Red
  | Yellow

/-- Represents the state of the bag of chips -/
structure ChipBag where
  blue : Nat
  red : Nat
  yellow : Nat

/-- Calculates the total number of chips in the bag -/
def ChipBag.total (bag : ChipBag) : Nat :=
  bag.blue + bag.red + bag.yellow

/-- Calculates the probability of drawing a specific color -/
def drawProbability (bag : ChipBag) (color : ChipColor) : Rat :=
  match color with
  | ChipColor.Blue => bag.blue / bag.total
  | ChipColor.Red => bag.red / bag.total
  | ChipColor.Yellow => bag.yellow / bag.total

/-- Calculates the probability of drawing two different colored chips -/
def differentColorProbability (bag : ChipBag) : Rat :=
  let blueFirst := drawProbability bag ChipColor.Blue * (1 - drawProbability bag ChipColor.Blue / 2)
  let redFirst := drawProbability bag ChipColor.Red * (1 - drawProbability bag ChipColor.Red)
  let yellowFirst := drawProbability bag ChipColor.Yellow * (1 - drawProbability bag ChipColor.Yellow)
  blueFirst + redFirst + yellowFirst

theorem two_different_color_chips_probability :
  let initialBag : ChipBag := { blue := 7, red := 5, yellow := 4 }
  differentColorProbability initialBag = 381 / 512 := by
  sorry


end NUMINAMATH_CALUDE_two_different_color_chips_probability_l1306_130663


namespace NUMINAMATH_CALUDE_alcohol_solution_percentage_l1306_130670

theorem alcohol_solution_percentage (initial_volume : ℝ) (initial_percentage : ℝ) 
  (added_alcohol : ℝ) (added_water : ℝ) : 
  initial_volume = 40 →
  initial_percentage = 5 →
  added_alcohol = 6.5 →
  added_water = 3.5 →
  let initial_alcohol := initial_volume * (initial_percentage / 100)
  let final_alcohol := initial_alcohol + added_alcohol
  let final_volume := initial_volume + added_alcohol + added_water
  let final_percentage := (final_alcohol / final_volume) * 100
  final_percentage = 17 := by
sorry

end NUMINAMATH_CALUDE_alcohol_solution_percentage_l1306_130670


namespace NUMINAMATH_CALUDE_prob_king_queen_standard_deck_l1306_130614

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (ranks : Nat)
  (suits : Nat)
  (cards_per_rank_suit : Nat)

/-- A standard deck has 52 cards, 13 ranks, 4 suits, and 1 card per rank per suit -/
def standard_deck : Deck :=
  { cards := 52
  , ranks := 13
  , suits := 4
  , cards_per_rank_suit := 1
  }

/-- The probability of drawing a King first and a Queen second from a standard deck -/
def prob_king_queen (d : Deck) : ℚ :=
  (d.suits : ℚ) / d.cards * (d.suits : ℚ) / (d.cards - 1)

/-- Theorem: The probability of drawing a King first and a Queen second from a standard deck is 4/663 -/
theorem prob_king_queen_standard_deck : 
  prob_king_queen standard_deck = 4 / 663 := by
  sorry

end NUMINAMATH_CALUDE_prob_king_queen_standard_deck_l1306_130614


namespace NUMINAMATH_CALUDE_text_ratio_is_five_to_one_l1306_130602

/-- Represents the number of texts in each category --/
structure TextCounts where
  grocery : ℕ
  notResponding : ℕ
  police : ℕ

/-- The conditions of the problem --/
def textProblemConditions (t : TextCounts) : Prop :=
  t.grocery = 5 ∧
  t.police = (t.grocery + t.notResponding) / 10 ∧
  t.grocery + t.notResponding + t.police = 33

/-- The theorem to be proved --/
theorem text_ratio_is_five_to_one (t : TextCounts) :
  textProblemConditions t →
  t.notResponding / t.grocery = 5 := by
sorry

end NUMINAMATH_CALUDE_text_ratio_is_five_to_one_l1306_130602


namespace NUMINAMATH_CALUDE_parabola_perpendicular_line_passes_through_point_l1306_130664

/-- The parabola y = x^2 -/
def parabola (p : ℝ × ℝ) : Prop := p.2 = p.1^2

/-- Two points are different -/
def different (p q : ℝ × ℝ) : Prop := p ≠ q

/-- A point is not the origin -/
def not_origin (p : ℝ × ℝ) : Prop := p ≠ (0, 0)

/-- Two vectors are perpendicular -/
def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

/-- A point lies on a line defined by two other points -/
def on_line (p q r : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, r = (1 - t) • p + t • q

theorem parabola_perpendicular_line_passes_through_point
  (A B : ℝ × ℝ)
  (h_parabola_A : parabola A)
  (h_parabola_B : parabola B)
  (h_different : different A B)
  (h_not_origin_A : not_origin A)
  (h_not_origin_B : not_origin B)
  (h_perpendicular : perpendicular A B) :
  on_line A B (0, 1) :=
sorry

end NUMINAMATH_CALUDE_parabola_perpendicular_line_passes_through_point_l1306_130664


namespace NUMINAMATH_CALUDE_four_point_lines_l1306_130666

/-- A point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in a plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) = (p3.x - p1.x) * (p2.y - p1.y)

/-- Count the number of distinct lines through four points -/
def count_lines (p1 p2 p3 p4 : Point) : ℕ :=
  sorry

/-- Theorem: The number of distinct lines through four points is either 1, 4, or 6 -/
theorem four_point_lines (p1 p2 p3 p4 : Point) :
  count_lines p1 p2 p3 p4 = 1 ∨ count_lines p1 p2 p3 p4 = 4 ∨ count_lines p1 p2 p3 p4 = 6 :=
by sorry

end NUMINAMATH_CALUDE_four_point_lines_l1306_130666


namespace NUMINAMATH_CALUDE_mary_sticker_problem_l1306_130679

/-- Given the conditions about Mary's stickers, prove the total number of students in the class. -/
theorem mary_sticker_problem (total_stickers : ℕ) (friends : ℕ) (stickers_per_friend : ℕ) 
  (stickers_per_other : ℕ) (leftover_stickers : ℕ) :
  total_stickers = 250 →
  friends = 10 →
  stickers_per_friend = 15 →
  stickers_per_other = 5 →
  leftover_stickers = 25 →
  ∃ (total_students : ℕ), total_students = 26 ∧ 
    total_stickers = friends * stickers_per_friend + 
    (total_students - friends - 1) * stickers_per_other + leftover_stickers :=
by sorry

end NUMINAMATH_CALUDE_mary_sticker_problem_l1306_130679


namespace NUMINAMATH_CALUDE_survey_response_rate_change_l1306_130653

theorem survey_response_rate_change 
  (original_customers : Nat) 
  (original_responses : Nat)
  (final_customers : Nat)
  (final_responses : Nat)
  (h1 : original_customers = 100)
  (h2 : original_responses = 10)
  (h3 : final_customers = 90)
  (h4 : final_responses = 27) :
  ((final_responses : ℝ) / final_customers - (original_responses : ℝ) / original_customers) / 
  ((original_responses : ℝ) / original_customers) * 100 = 200 := by
sorry

end NUMINAMATH_CALUDE_survey_response_rate_change_l1306_130653


namespace NUMINAMATH_CALUDE_picnic_watermelon_slices_l1306_130616

/-- The number of watermelons Danny brings -/
def danny_watermelons : ℕ := 3

/-- The number of slices Danny cuts each watermelon into -/
def danny_slices_per_watermelon : ℕ := 10

/-- The number of watermelons Danny's sister brings -/
def sister_watermelons : ℕ := 1

/-- The number of slices Danny's sister cuts her watermelon into -/
def sister_slices_per_watermelon : ℕ := 15

/-- The total number of watermelon slices at the picnic -/
def total_slices : ℕ := danny_watermelons * danny_slices_per_watermelon + sister_watermelons * sister_slices_per_watermelon

theorem picnic_watermelon_slices : total_slices = 45 := by
  sorry

end NUMINAMATH_CALUDE_picnic_watermelon_slices_l1306_130616


namespace NUMINAMATH_CALUDE_solution_set_properties_l1306_130648

def M : Set ℝ := {x : ℝ | 3 - 2*x < 0}

theorem solution_set_properties : (0 ∉ M) ∧ (2 ∈ M) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_properties_l1306_130648


namespace NUMINAMATH_CALUDE_f_monotonicity_and_g_zeros_l1306_130604

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (x^3 - a*x)

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x - x^2

theorem f_monotonicity_and_g_zeros (a : ℝ) :
  (a ≤ 0 → ∀ x y : ℝ, x < y → f a x < f a y) ∧
  (a > 0 → ∀ x y : ℝ, 
    ((x < y ∧ y < -Real.sqrt (a/3)) ∨ (x > Real.sqrt (a/3) ∧ y > x)) → f a x < f a y) ∧
  (a > 0 → ∀ x y : ℝ, 
    (-Real.sqrt (a/3) < x ∧ x < y ∧ y < Real.sqrt (a/3)) → f a x > f a y) ∧
  (∃ x y : ℝ, x < y ∧ g a x = 0 ∧ g a y = 0 ∧ ∀ z : ℝ, z ≠ x ∧ z ≠ y → g a z ≠ 0) →
  a > 1 :=
sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_g_zeros_l1306_130604


namespace NUMINAMATH_CALUDE_good_2013_implies_good_20_l1306_130699

/-- A sequence of positive integers is non-decreasing -/
def IsNonDecreasingSeq (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a n ≤ a (n + 1)

/-- A number is good if it can be expressed as i/a_i for some index i -/
def IsGood (n : ℕ) (a : ℕ → ℕ) : Prop :=
  ∃ i : ℕ, n = i / a i

theorem good_2013_implies_good_20 (a : ℕ → ℕ) 
  (h_nondec : IsNonDecreasingSeq a) 
  (h_2013 : IsGood 2013 a) : 
  IsGood 20 a :=
sorry

end NUMINAMATH_CALUDE_good_2013_implies_good_20_l1306_130699


namespace NUMINAMATH_CALUDE_exists_fibonacci_divisible_by_1000_l1306_130677

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem exists_fibonacci_divisible_by_1000 :
  ∃ n : ℕ, n ≤ 1000001 ∧ fibonacci n % 1000 = 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_fibonacci_divisible_by_1000_l1306_130677


namespace NUMINAMATH_CALUDE_sin_150_degrees_l1306_130600

theorem sin_150_degrees : Real.sin (150 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_150_degrees_l1306_130600


namespace NUMINAMATH_CALUDE_cube_root_and_square_root_l1306_130688

-- Define the cube root function
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- Define the square root function
noncomputable def squareRoot (x : ℝ) : Set ℝ := {y : ℝ | y^2 = x ∧ (y ≥ 0 ∨ y ≤ 0)}

theorem cube_root_and_square_root :
  (cubeRoot (1/8) = 1/2) ∧
  (squareRoot ((-6)^2) = {6, -6}) :=
sorry

end NUMINAMATH_CALUDE_cube_root_and_square_root_l1306_130688


namespace NUMINAMATH_CALUDE_currency_exchange_problem_l1306_130611

/-- 
Proves the existence of a positive integer d that satisfies the conditions
of the currency exchange problem and has a digit sum of 3.
-/
theorem currency_exchange_problem : ∃ d : ℕ+, 
  (8 : ℚ) / 5 * d.val - 72 = d.val ∧ 
  (d.val.repr.toList.map (λ c => c.toString.toNat!)).sum = 3 := by
  sorry

end NUMINAMATH_CALUDE_currency_exchange_problem_l1306_130611


namespace NUMINAMATH_CALUDE_monomial_sum_condition_l1306_130674

/-- If the sum of two monomials 2a^5*b^(2m+4) and a^(2n-3)*b^8 is still a monomial,
    then m = 2 and n = 4 -/
theorem monomial_sum_condition (a b : ℝ) (m n : ℕ) :
  (∃ (k : ℝ) (p q : ℕ), 2 * a^5 * b^(2*m+4) + a^(2*n-3) * b^8 = k * a^p * b^q) →
  m = 2 ∧ n = 4 := by
sorry


end NUMINAMATH_CALUDE_monomial_sum_condition_l1306_130674


namespace NUMINAMATH_CALUDE_day_15_net_income_l1306_130619

/-- Calculate the net income on a given day of business -/
def net_income (initial_income : ℝ) (daily_multiplier : ℝ) (daily_expenses : ℝ) (tax_rate : ℝ) (day : ℕ) : ℝ :=
  let gross_income := initial_income * daily_multiplier^(day - 1)
  let tax := tax_rate * gross_income
  let after_tax := gross_income - tax
  after_tax - daily_expenses

/-- The net income on the 15th day of business is $12,913,916.3 -/
theorem day_15_net_income :
  net_income 3 3 100 0.1 15 = 12913916.3 := by
  sorry

end NUMINAMATH_CALUDE_day_15_net_income_l1306_130619


namespace NUMINAMATH_CALUDE_arcsin_sqrt3_over_2_l1306_130615

theorem arcsin_sqrt3_over_2 : Real.arcsin (Real.sqrt 3 / 2) = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_sqrt3_over_2_l1306_130615


namespace NUMINAMATH_CALUDE_complex_calculation_l1306_130646

theorem complex_calculation (c d : ℂ) (hc : c = 3 + 2*I) (hd : d = 2 - 3*I) :
  3*c + 4*d = 17 - 6*I :=
by sorry

end NUMINAMATH_CALUDE_complex_calculation_l1306_130646


namespace NUMINAMATH_CALUDE_fourth_root_simplification_l1306_130672

theorem fourth_root_simplification :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ 
  (2^8 * 3^5)^(1/4 : ℝ) = a * (b : ℝ)^(1/4 : ℝ) ∧
  a + b = 15 :=
by sorry

end NUMINAMATH_CALUDE_fourth_root_simplification_l1306_130672


namespace NUMINAMATH_CALUDE_consecutive_ranks_probability_l1306_130636

/-- Represents a standard deck of 52 playing cards -/
def StandardDeck : ℕ := 52

/-- Number of cards drawn -/
def CardsDrawn : ℕ := 3

/-- Number of possible consecutive rank sets (A-2-3 to J-Q-K) -/
def ConsecutiveRankSets : ℕ := 10

/-- Number of suits in a standard deck -/
def Suits : ℕ := 4

/-- The probability of drawing three cards of consecutive ranks from a standard deck -/
theorem consecutive_ranks_probability :
  (ConsecutiveRankSets * Suits^CardsDrawn) / (StandardDeck.choose CardsDrawn) = 32 / 1105 := by
  sorry

#check consecutive_ranks_probability

end NUMINAMATH_CALUDE_consecutive_ranks_probability_l1306_130636
