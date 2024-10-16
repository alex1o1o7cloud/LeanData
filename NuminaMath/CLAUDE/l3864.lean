import Mathlib

namespace NUMINAMATH_CALUDE_equation_solution_l3864_386464

theorem equation_solution (x : ℝ) : 
  (8 / (Real.sqrt (x - 5) - 10) + 2 / (Real.sqrt (x - 5) - 5) + 
   9 / (Real.sqrt (x - 5) + 5) + 16 / (Real.sqrt (x - 5) + 10) = 0) ↔ 
  (x = 145 / 9 ∨ x = 1200 / 121) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3864_386464


namespace NUMINAMATH_CALUDE_inequality_proof_l3864_386442

theorem inequality_proof (x y z : ℝ) 
  (h1 : x ≥ 1) (h2 : y ≥ 1) (h3 : z ≥ 1)
  (h4 : 1 / (x^2 - 1) + 1 / (y^2 - 1) + 1 / (z^2 - 1) = 1) : 
  1 / (x + 1) + 1 / (y + 1) + 1 / (z + 1) ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3864_386442


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l3864_386439

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def M : Set ℕ := {2, 4, 7}

theorem complement_of_M_in_U :
  U \ M = {1, 3, 5, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l3864_386439


namespace NUMINAMATH_CALUDE_integral_x_over_sqrt_x_squared_plus_one_l3864_386467

theorem integral_x_over_sqrt_x_squared_plus_one (x : ℝ) :
  deriv (λ x => Real.sqrt (x^2 + 1)) x = x / Real.sqrt (x^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_integral_x_over_sqrt_x_squared_plus_one_l3864_386467


namespace NUMINAMATH_CALUDE_jane_rejection_calculation_l3864_386443

/-- The percentage of products John rejected -/
def john_rejection_rate : ℝ := 0.007

/-- The total percentage of products rejected -/
def total_rejection_rate : ℝ := 0.0075

/-- The fraction of products Jane inspected -/
def jane_inspection_fraction : ℝ := 0.5

/-- The percentage of products Jane rejected -/
def jane_rejection_rate : ℝ := 0.001

theorem jane_rejection_calculation :
  john_rejection_rate + jane_rejection_rate * jane_inspection_fraction = total_rejection_rate :=
sorry

end NUMINAMATH_CALUDE_jane_rejection_calculation_l3864_386443


namespace NUMINAMATH_CALUDE_largest_undefined_value_l3864_386475

theorem largest_undefined_value (x : ℝ) :
  let f (x : ℝ) := (x + 2) / (9 * x^2 - 74 * x + 9)
  let roots := { x | 9 * x^2 - 74 * x + 9 = 0 }
  ∃ (max_root : ℝ), max_root ∈ roots ∧ ∀ (y : ℝ), y ∈ roots → y ≤ max_root ∧
  ∀ (z : ℝ), z > max_root → f z ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_largest_undefined_value_l3864_386475


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l3864_386482

theorem absolute_value_equation_solution : 
  ∀ x : ℝ, (|x + 2| = 3*x - 6) ↔ (x = 4) := by
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l3864_386482


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l3864_386415

theorem triangle_angle_measure (a b c : ℝ) (A C : ℝ) (h : b = c * Real.cos A + Real.sqrt 3 * a * Real.sin C) :
  C = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l3864_386415


namespace NUMINAMATH_CALUDE_ellipse_triangle_area_implies_segment_length_l3864_386469

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define point B
def B : ℝ × ℝ := (0, 3)

-- Define the triangle area function
noncomputable def triangleArea (P A B : ℝ × ℝ) : ℝ := sorry

-- Define the length function
noncomputable def segmentLength (A B : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem ellipse_triangle_area_implies_segment_length :
  ∀ P A : ℝ × ℝ,
  ellipse P.1 P.2 →
  (∀ Q : ℝ × ℝ, ellipse Q.1 Q.2 → triangleArea Q A B ≥ 1) →
  (∃ R : ℝ × ℝ, ellipse R.1 R.2 ∧ triangleArea R A B = 5) →
  segmentLength A B = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_triangle_area_implies_segment_length_l3864_386469


namespace NUMINAMATH_CALUDE_tenth_term_is_123_a_plus_b_power_10_is_123_l3864_386499

-- Define the sequence
def seq : ℕ → ℕ
| 0 => 1  -- a + b
| 1 => 3  -- a² + b²
| 2 => 4  -- a³ + b³
| n + 3 => seq (n + 1) + seq (n + 2)

-- State the theorem
theorem tenth_term_is_123 : seq 9 = 123 := by
  sorry

-- Define a and b
noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

-- State the given conditions
axiom sum_1 : a + b = 1
axiom sum_2 : a^2 + b^2 = 3
axiom sum_3 : a^3 + b^3 = 4
axiom sum_4 : a^4 + b^4 = 7
axiom sum_5 : a^5 + b^5 = 11

-- State the main theorem
theorem a_plus_b_power_10_is_123 : a^10 + b^10 = 123 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_is_123_a_plus_b_power_10_is_123_l3864_386499


namespace NUMINAMATH_CALUDE_inequality_proof_l3864_386488

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x * y * z + x * y + y * z + z * x = x + y + z + 1) :
  (1 / 3 : ℝ) * (Real.sqrt ((1 + x^2) / (1 + x)) + Real.sqrt ((1 + y^2) / (1 + y)) + Real.sqrt ((1 + z^2) / (1 + z)))
  ≤ ((x + y + z) / 3) ^ (5 / 8) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3864_386488


namespace NUMINAMATH_CALUDE_car_speed_problem_l3864_386491

/-- Proves that the speed of Car A is 50 km/hr given the problem conditions -/
theorem car_speed_problem (speed_B time_B time_A ratio : ℝ) 
  (h1 : speed_B = 25)
  (h2 : time_B = 4)
  (h3 : time_A = 8)
  (h4 : ratio = 4)
  (h5 : ratio = (speed_A * time_A) / (speed_B * time_B)) :
  speed_A = 50 :=
by
  sorry

#check car_speed_problem

end NUMINAMATH_CALUDE_car_speed_problem_l3864_386491


namespace NUMINAMATH_CALUDE_least_seven_ternary_correct_l3864_386430

/-- Converts a base 10 number to its ternary (base 3) representation --/
def to_ternary (n : ℕ) : List ℕ :=
  sorry

/-- Checks if a number has exactly 7 digits in its ternary representation --/
def has_seven_ternary_digits (n : ℕ) : Prop :=
  (to_ternary n).length = 7

/-- The least positive base ten number with seven ternary digits --/
def least_seven_ternary : ℕ := 729

theorem least_seven_ternary_correct :
  (has_seven_ternary_digits least_seven_ternary) ∧
  (∀ m : ℕ, m > 0 ∧ m < least_seven_ternary → ¬(has_seven_ternary_digits m)) :=
sorry

end NUMINAMATH_CALUDE_least_seven_ternary_correct_l3864_386430


namespace NUMINAMATH_CALUDE_rectangle_width_length_ratio_l3864_386486

theorem rectangle_width_length_ratio 
  (w : ℝ) 
  (h1 : w > 0) 
  (h2 : 2 * (w + 10) = 30) : 
  w / 10 = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_rectangle_width_length_ratio_l3864_386486


namespace NUMINAMATH_CALUDE_avg_people_moving_rounded_l3864_386420

/-- The number of people moving to California -/
def people_moving : ℕ := 4500

/-- The time period in days -/
def days : ℕ := 5

/-- The additional hours beyond full days -/
def extra_hours : ℕ := 12

/-- Function to calculate the average people per minute -/
def avg_people_per_minute (people : ℕ) (days : ℕ) (hours : ℕ) : ℚ :=
  people / (((days * 24 + hours) * 60) : ℚ)

/-- Function to round a rational number to the nearest integer -/
def round_to_nearest (q : ℚ) : ℤ :=
  ⌊q + 1/2⌋

/-- Theorem stating that the average number of people moving per minute, 
    when rounded to the nearest whole number, is 1 -/
theorem avg_people_moving_rounded : 
  round_to_nearest (avg_people_per_minute people_moving days extra_hours) = 1 := by
  sorry


end NUMINAMATH_CALUDE_avg_people_moving_rounded_l3864_386420


namespace NUMINAMATH_CALUDE_divisibility_congruence_l3864_386417

theorem divisibility_congruence (n : ℤ) :
  (6 ∣ (n - 4)) → (10 ∣ (n - 8)) → n ≡ -2 [ZMOD 30] := by
  sorry

end NUMINAMATH_CALUDE_divisibility_congruence_l3864_386417


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3864_386400

def arithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def increasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) > a n

theorem arithmetic_sequence_properties
  (a : ℕ → ℝ) (d : ℝ) (h : d > 0) (h_arith : arithmeticSequence a d) :
  increasingSequence a ∧
  increasingSequence (fun n ↦ a n + 3 * n * d) ∧
  (¬ ∀ d, arithmeticSequence a d → increasingSequence (fun n ↦ n * a n)) ∧
  (¬ ∀ d, arithmeticSequence a d → increasingSequence (fun n ↦ a n / n)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3864_386400


namespace NUMINAMATH_CALUDE_sum_of_zeros_l3864_386424

/-- The parabola after transformations -/
def transformed_parabola (x : ℝ) : ℝ := -(x - 7)^2 + 7

/-- The zeros of the transformed parabola -/
def zeros : Set ℝ := {x | transformed_parabola x = 0}

theorem sum_of_zeros : ∃ (a b : ℝ), a ∈ zeros ∧ b ∈ zeros ∧ a + b = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_zeros_l3864_386424


namespace NUMINAMATH_CALUDE_inequality_proof_equality_condition_l3864_386435

theorem inequality_proof (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hab : a + b < 2) :
  (1 / (1 + a^2)) + (1 / (1 + b^2)) ≤ 2 / (1 + a*b) :=
by sorry

theorem equality_condition (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hab : a + b < 2) :
  (1 / (1 + a^2)) + (1 / (1 + b^2)) = 2 / (1 + a*b) ↔ a = b :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_equality_condition_l3864_386435


namespace NUMINAMATH_CALUDE_a_8_equals_3_l3864_386403

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- An arithmetic sequence -/
def IsArithmeticSequence (b : Sequence) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) - b n = d

theorem a_8_equals_3
  (a b : Sequence)
  (h1 : a 1 = 3)
  (h2 : IsArithmeticSequence b)
  (h3 : ∀ n : ℕ, n > 0 → b n = a (n + 1) - a n)
  (h4 : b 3 = -2)
  (h5 : b 10 = 12) :
  a 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_a_8_equals_3_l3864_386403


namespace NUMINAMATH_CALUDE_turnip_bag_options_l3864_386451

def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]

def is_valid_turnip_weight (t : ℕ) : Prop :=
  t ∈ bag_weights ∧
  ∃ (o c : ℕ),
    o + c = (bag_weights.sum - t) ∧
    c = 2 * o

theorem turnip_bag_options :
  ∀ t : ℕ, is_valid_turnip_weight t ↔ t = 13 ∨ t = 16 :=
by sorry

end NUMINAMATH_CALUDE_turnip_bag_options_l3864_386451


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3864_386489

/-- Given an arithmetic sequence {a_n} where a_5 = 3 and a_9 = 6, prove that a_13 = 9 -/
theorem arithmetic_sequence_problem (a : ℕ → ℤ) 
  (h_arithmetic : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) 
  (h_a5 : a 5 = 3) 
  (h_a9 : a 9 = 6) : 
  a 13 = 9 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3864_386489


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l3864_386474

theorem square_area_from_diagonal (d : ℝ) (h : d = 10) : 
  (d^2 / 2 : ℝ) = 50 := by sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l3864_386474


namespace NUMINAMATH_CALUDE_piston_experiment_l3864_386450

variable (l d P q π : ℝ)
variable (x y : ℝ)

-- Conditions
variable (h1 : l > 0)
variable (h2 : d > 0)
variable (h3 : P > 0)
variable (h4 : q > 0)
variable (h5 : π > 0)

-- Theorem statement
theorem piston_experiment :
  -- First experiment
  (P * x^2 + 2*q*l*π*x - P*l^2 = 0) ∧
  -- Pressure in AC region
  (l*π / (l + x) = P * (l - x) / q) ∧
  -- Second experiment
  (y = l*P / (q*π - P)) ∧
  -- Condition for piston not falling to bottom
  (P < q*π/2) :=
by sorry

end NUMINAMATH_CALUDE_piston_experiment_l3864_386450


namespace NUMINAMATH_CALUDE_complex_modulus_l3864_386401

theorem complex_modulus (z : ℂ) : z + 2*I = (3 - I^3) / (1 + I) → Complex.abs z = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l3864_386401


namespace NUMINAMATH_CALUDE_min_value_sum_and_reciprocals_l3864_386459

theorem min_value_sum_and_reciprocals (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a + b + 1/a + 1/b ≥ 4 ∧ (a + b + 1/a + 1/b = 4 ↔ a = 1 ∧ b = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_and_reciprocals_l3864_386459


namespace NUMINAMATH_CALUDE_right_triangle_tangent_midpoint_l3864_386423

theorem right_triangle_tangent_midpoint (n : ℕ) (a h : ℝ) (α : ℝ) :
  n > 1 →
  Odd n →
  0 < a →
  0 < h →
  0 < α →
  α < π / 2 →
  Real.tan α = (4 * n * h) / ((n^2 - 1) * a) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_tangent_midpoint_l3864_386423


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l3864_386432

-- Define the polynomial p
def p (x : ℝ) : ℝ := sorry

-- State the theorem
theorem polynomial_evaluation (y : ℝ) :
  (p (y^2 + 1) = 6 * y^4 - y^2 + 5) →
  (p (y^2 - 1) = 6 * y^4 - 25 * y^2 + 31) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l3864_386432


namespace NUMINAMATH_CALUDE_possible_values_of_a_l3864_386454

theorem possible_values_of_a (a b c d : ℕ+) 
  (h1 : a > b ∧ b > c ∧ c > d)
  (h2 : a + b + c + d = 2020)
  (h3 : a^2 - b^2 + c^2 - d^2 = 2020) :
  ∃! s : Finset ℕ+, s.card = 501 ∧ ∀ x, x ∈ s ↔ ∃ b' c' d' : ℕ+, 
    x > b' ∧ b' > c' ∧ c' > d' ∧
    x + b' + c' + d' = 2020 ∧
    x^2 - b'^2 + c'^2 - d'^2 = 2020 :=
by sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l3864_386454


namespace NUMINAMATH_CALUDE_motel_rent_theorem_l3864_386480

/-- Represents the total rent charged by a motel on a specific night -/
def TotalRent : ℕ → ℕ → ℕ 
  | r40, r60 => 40 * r40 + 60 * r60

/-- Represents the reduced total rent after changing 10 rooms from $60 to $40 -/
def ReducedRent : ℕ → ℕ → ℕ 
  | r40, r60 => 40 * (r40 + 10) + 60 * (r60 - 10)

theorem motel_rent_theorem (r40 r60 : ℕ) :
  (TotalRent r40 r60 - ReducedRent r40 r60 = 200) → 
  (ReducedRent r40 r60 = (9 * TotalRent r40 r60) / 10) → 
  TotalRent r40 r60 = 2000 := by
  sorry

#check motel_rent_theorem

end NUMINAMATH_CALUDE_motel_rent_theorem_l3864_386480


namespace NUMINAMATH_CALUDE_line_points_l3864_386493

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem line_points : 
  let p1 : Point := ⟨4, 10⟩
  let p2 : Point := ⟨-2, -8⟩
  let p3 : Point := ⟨1, 1⟩
  let p4 : Point := ⟨-1, -5⟩
  let p5 : Point := ⟨3, 7⟩
  let p6 : Point := ⟨0, -1⟩
  let p7 : Point := ⟨2, 3⟩
  collinear p1 p2 p3 ∧ 
  collinear p1 p2 p4 ∧ 
  collinear p1 p2 p5 ∧ 
  ¬collinear p1 p2 p6 ∧ 
  ¬collinear p1 p2 p7 := by sorry

end NUMINAMATH_CALUDE_line_points_l3864_386493


namespace NUMINAMATH_CALUDE_jared_car_count_l3864_386419

theorem jared_car_count : ∀ (j a f : ℕ),
  (j : ℝ) = 0.85 * a →
  a = f + 7 →
  j + a + f = 983 →
  j = 295 :=
by sorry

end NUMINAMATH_CALUDE_jared_car_count_l3864_386419


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l3864_386410

/-- An arithmetic sequence with its sum function and properties -/
structure ArithmeticSequence where
  /-- The general term of the sequence -/
  a : ℕ → ℝ
  /-- The sum of the first n terms -/
  S : ℕ → ℝ
  /-- The sum of the first 4 terms is 0 -/
  sum_4 : S 4 = 0
  /-- The 5th term is 5 -/
  term_5 : a 5 = 5
  /-- The sequence is arithmetic -/
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)

/-- The general term of the arithmetic sequence is 2n - 5 -/
theorem arithmetic_sequence_general_term (seq : ArithmeticSequence) :
  ∀ n : ℕ, seq.a n = 2 * n - 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l3864_386410


namespace NUMINAMATH_CALUDE_penumbra_ring_area_l3864_386405

/-- The area of a ring formed between two concentric circles --/
theorem penumbra_ring_area (r_umbra : ℝ) (r_penumbra : ℝ) (h1 : r_umbra = 40) (h2 : r_penumbra = 3 * r_umbra) :
  let a_ring := π * r_penumbra^2 - π * r_umbra^2
  a_ring = 12800 * π := by sorry

end NUMINAMATH_CALUDE_penumbra_ring_area_l3864_386405


namespace NUMINAMATH_CALUDE_complete_square_with_integer_l3864_386481

theorem complete_square_with_integer (y : ℝ) : ∃ (a b : ℤ), y^2 + 12*y + 50 = (y + ↑a)^2 + ↑b ∧ b = 14 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_with_integer_l3864_386481


namespace NUMINAMATH_CALUDE_trig_identity_l3864_386479

theorem trig_identity (α : ℝ) (h1 : α ∈ Set.Ioo (-π/2) 0) 
  (h2 : Real.sin (α + π/4) = -1/3) : 
  Real.sin (2*α) / Real.cos (π/4 - α) = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3864_386479


namespace NUMINAMATH_CALUDE_min_cards_36_4suits_l3864_386406

/-- Represents a deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (num_suits : ℕ)
  (cards_per_suit : ℕ)
  (h_total : total_cards = num_suits * cards_per_suit)

/-- The minimum number of cards to draw to guarantee all suits are represented -/
def min_cards_to_draw (d : Deck) : ℕ :=
  (d.num_suits - 1) * d.cards_per_suit + 1

/-- Theorem stating the minimum number of cards to draw for a 36-card deck with 4 suits -/
theorem min_cards_36_4suits :
  ∃ (d : Deck), d.total_cards = 36 ∧ d.num_suits = 4 ∧ min_cards_to_draw d = 28 :=
sorry

end NUMINAMATH_CALUDE_min_cards_36_4suits_l3864_386406


namespace NUMINAMATH_CALUDE_probability_two_primary_schools_l3864_386418

/-- Represents the types of schools in the region -/
inductive SchoolType
| Primary
| Middle
| University

/-- Represents the total number of schools of each type -/
def totalSchools : SchoolType → ℕ
| SchoolType.Primary => 21
| SchoolType.Middle => 14
| SchoolType.University => 7

/-- Represents the number of schools selected in the stratified sample -/
def selectedSchools : SchoolType → ℕ
| SchoolType.Primary => 3
| SchoolType.Middle => 2
| SchoolType.University => 1

/-- The total number of schools in the stratified sample -/
def totalSampleSize : ℕ := 6

/-- The number of schools to be randomly selected from the sample -/
def selectionSize : ℕ := 2

/-- Theorem stating that the probability of selecting two primary schools
    from the stratified sample is 1/5 -/
theorem probability_two_primary_schools :
  (selectedSchools SchoolType.Primary).choose selectionSize /
  (totalSampleSize.choose selectionSize) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_primary_schools_l3864_386418


namespace NUMINAMATH_CALUDE_quadratic_roots_equal_and_real_l3864_386466

theorem quadratic_roots_equal_and_real (a c : ℝ) (h : a ≠ 0) :
  (∃ x : ℝ, a * x^2 - 2 * x * Real.sqrt 2 + c = 0) ∧
  ((-2 * Real.sqrt 2)^2 - 4 * a * c = 0) →
  ∃! x : ℝ, a * x^2 - 2 * x * Real.sqrt 2 + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_equal_and_real_l3864_386466


namespace NUMINAMATH_CALUDE_farmer_water_capacity_l3864_386498

/-- Calculates the total water capacity for a farmer's trucks -/
def total_water_capacity (num_trucks : ℕ) (tanks_per_truck : ℕ) (tank_capacity : ℕ) : ℕ :=
  num_trucks * tanks_per_truck * tank_capacity

/-- Theorem: The farmer can carry 1350 liters of water -/
theorem farmer_water_capacity :
  total_water_capacity 3 3 150 = 1350 := by
  sorry

end NUMINAMATH_CALUDE_farmer_water_capacity_l3864_386498


namespace NUMINAMATH_CALUDE_all_sections_clearance_l3864_386473

/-- Represents the percentage of candidates who cleared a specific number of sections -/
structure SectionClearance where
  zero : ℝ
  one : ℝ
  two : ℝ
  three : ℝ
  four : ℝ
  five : ℝ

/-- Theorem stating the percentage of candidates who cleared all 5 sections -/
theorem all_sections_clearance 
  (total_candidates : ℕ) 
  (three_section_candidates : ℕ) 
  (clearance : SectionClearance) :
  total_candidates = 1200 →
  three_section_candidates = 300 →
  clearance.zero = 5 →
  clearance.one = 25 →
  clearance.two = 24.5 →
  clearance.four = 20 →
  clearance.three = (three_section_candidates : ℝ) / (total_candidates : ℝ) * 100 →
  clearance.five = 0.5 :=
by sorry

end NUMINAMATH_CALUDE_all_sections_clearance_l3864_386473


namespace NUMINAMATH_CALUDE_diagonal_intersection_y_value_l3864_386455

/-- A square in the coordinate plane with specific properties -/
structure Square where
  vertex : ℝ × ℝ
  diagonal_intersection_x : ℝ
  area : ℝ

/-- The y-coordinate of the diagonal intersection point of the square -/
def diagonal_intersection_y (s : Square) : ℝ :=
  s.vertex.2 + (s.diagonal_intersection_x - s.vertex.1)

/-- Theorem stating the y-coordinate of the diagonal intersection point -/
theorem diagonal_intersection_y_value (s : Square) 
  (h1 : s.vertex = (-6, -4))
  (h2 : s.diagonal_intersection_x = 3)
  (h3 : s.area = 324) :
  diagonal_intersection_y s = 5 := by
  sorry


end NUMINAMATH_CALUDE_diagonal_intersection_y_value_l3864_386455


namespace NUMINAMATH_CALUDE_rectangle_dimension_increase_l3864_386492

theorem rectangle_dimension_increase (L B : ℝ) (L' B' : ℝ) (h1 : L' = 1.1 * L) (h2 : L' * B' = 1.43 * (L * B)) : B' = 1.3 * B :=
sorry

end NUMINAMATH_CALUDE_rectangle_dimension_increase_l3864_386492


namespace NUMINAMATH_CALUDE_data_analysis_l3864_386421

def dataset : List ℕ := [10, 8, 6, 9, 8, 7, 8]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℚ := sorry

def mean (l : List ℕ) : ℚ := sorry

def variance (l : List ℕ) : ℚ := sorry

theorem data_analysis (l : List ℕ) (h : l = dataset) : 
  (mode l = 8) ∧ 
  (median l = 8) ∧ 
  (mean l = 8) ∧ 
  (variance l ≠ 8) := by sorry

end NUMINAMATH_CALUDE_data_analysis_l3864_386421


namespace NUMINAMATH_CALUDE_horner_method_v3_l3864_386461

def f (x : ℝ) : ℝ := 2*x^5 - x + 3*x^2 + x + 1

def horner_v3 (x : ℝ) : ℝ := 
  let v0 := 2
  let v1 := v0 * x + 0
  let v2 := v1 * x - 1
  v2 * x + 3

theorem horner_method_v3 : horner_v3 3 = 54 := by sorry

end NUMINAMATH_CALUDE_horner_method_v3_l3864_386461


namespace NUMINAMATH_CALUDE_triangle_lines_theorem_l3864_386433

/-- Triangle ABC with vertices A(-3,5), B(5,7), and C(5,1) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The given triangle -/
def ABC : Triangle := { A := (-3, 5), B := (5, 7), C := (5, 1) }

/-- Equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The median line on side AB -/
def medianAB : LineEquation := { a := 5, b := 4, c := -29 }

/-- The line through A with equal x-axis and y-axis intercepts -/
def lineA : LineEquation := { a := 1, b := 1, c := -2 }

theorem triangle_lines_theorem (t : Triangle) (m : LineEquation) (l : LineEquation) : 
  t = ABC → m = medianAB → l = lineA → True := by sorry

end NUMINAMATH_CALUDE_triangle_lines_theorem_l3864_386433


namespace NUMINAMATH_CALUDE_vacation_days_calculation_l3864_386477

theorem vacation_days_calculation (families : Nat) (people_per_family : Nat) 
  (towels_per_person_per_day : Nat) (towels_per_load : Nat) (total_loads : Nat) :
  families = 3 →
  people_per_family = 4 →
  towels_per_person_per_day = 1 →
  towels_per_load = 14 →
  total_loads = 6 →
  (total_loads * towels_per_load) / (families * people_per_family * towels_per_person_per_day) = 7 := by
  sorry

end NUMINAMATH_CALUDE_vacation_days_calculation_l3864_386477


namespace NUMINAMATH_CALUDE_rafael_weekly_earnings_l3864_386452

/-- Rafael's weekly earnings calculation --/
theorem rafael_weekly_earnings :
  let monday_hours : ℕ := 10
  let tuesday_hours : ℕ := 8
  let remaining_hours : ℕ := 20
  let hourly_rate : ℕ := 20

  let total_hours : ℕ := monday_hours + tuesday_hours + remaining_hours
  let weekly_earnings : ℕ := total_hours * hourly_rate

  weekly_earnings = 760 := by sorry

end NUMINAMATH_CALUDE_rafael_weekly_earnings_l3864_386452


namespace NUMINAMATH_CALUDE_largest_element_of_A_l3864_386470

def A : Set ℝ := {x | ∃ n : ℕ, 1 ≤ n ∧ n ≤ 2020 ∧ x = n ^ (1 / n : ℝ)}

theorem largest_element_of_A : ∀ x ∈ A, x ≤ 3 ^ (1 / 3 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_largest_element_of_A_l3864_386470


namespace NUMINAMATH_CALUDE_budgets_equal_after_6_years_l3864_386422

def initial_budget_Q : ℕ := 540000
def initial_budget_V : ℕ := 780000
def annual_increase_Q : ℕ := 30000
def annual_decrease_V : ℕ := 10000

def budget_Q (years : ℕ) : ℕ := initial_budget_Q + annual_increase_Q * years
def budget_V (years : ℕ) : ℕ := initial_budget_V - annual_decrease_V * years

theorem budgets_equal_after_6_years :
  ∃ (years : ℕ), years = 6 ∧ budget_Q years = budget_V years :=
sorry

end NUMINAMATH_CALUDE_budgets_equal_after_6_years_l3864_386422


namespace NUMINAMATH_CALUDE_range_of_m_for_false_proposition_l3864_386412

theorem range_of_m_for_false_proposition : 
  (∃ m : ℝ, ¬(∀ x : ℝ, x^2 - 2*x - m ≥ 0)) ↔ 
  (∃ m : ℝ, m > -1) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_for_false_proposition_l3864_386412


namespace NUMINAMATH_CALUDE_remainder_problem_l3864_386494

theorem remainder_problem (x : ℕ+) (h : 7 * x.val ≡ 1 [MOD 31]) : (20 + x.val) % 31 = 29 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3864_386494


namespace NUMINAMATH_CALUDE_student_marks_proof_l3864_386462

/-- Given a student's marks in mathematics, physics, and chemistry,
    prove that the total marks in mathematics and physics is 50. -/
theorem student_marks_proof (M P C : ℕ) : 
  C = P + 20 →
  (M + C) / 2 = 35 →
  M + P = 50 := by
sorry

end NUMINAMATH_CALUDE_student_marks_proof_l3864_386462


namespace NUMINAMATH_CALUDE_sector_central_angle_l3864_386408

/-- Given a circle sector with radius 10 cm and perimeter 45 cm, 
    the central angle of the sector is 2.5 radians. -/
theorem sector_central_angle (r : ℝ) (p : ℝ) (h1 : r = 10) (h2 : p = 45) :
  (p - 2 * r) / r = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l3864_386408


namespace NUMINAMATH_CALUDE_perpendicular_line_plane_implies_perpendicular_lines_l3864_386438

structure Plane where
  -- Define plane structure

structure Line where
  -- Define line structure

-- Define perpendicularity between a line and a plane
def perpendicular_line_plane (l : Line) (p : Plane) : Prop :=
  sorry

-- Define a line being contained in a plane
def line_in_plane (l : Line) (p : Plane) : Prop :=
  sorry

-- Define perpendicularity between two lines
def perpendicular_lines (l1 l2 : Line) : Prop :=
  sorry

-- Theorem statement
theorem perpendicular_line_plane_implies_perpendicular_lines
  (m n : Line) (α : Plane) 
  (h1 : m ≠ n) 
  (h2 : perpendicular_line_plane m α) 
  (h3 : line_in_plane n α) : 
  perpendicular_lines m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_plane_implies_perpendicular_lines_l3864_386438


namespace NUMINAMATH_CALUDE_matrix_equation_l3864_386497

def A : Matrix (Fin 2) (Fin 2) ℤ := !![3, -5; 2, -3]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![21, -34; 13, -21]
def N : Matrix (Fin 2) (Fin 2) ℤ := !![5, 3; 3, 2]

theorem matrix_equation : N * A = B := by sorry

end NUMINAMATH_CALUDE_matrix_equation_l3864_386497


namespace NUMINAMATH_CALUDE_cousins_ages_sum_l3864_386431

theorem cousins_ages_sum (ages : Fin 5 → ℕ) 
  (mean_condition : (ages 0 + ages 1 + ages 2 + ages 3 + ages 4) / 5 = 10)
  (median_condition : ages 2 = 12)
  (sorted : ∀ i j, i ≤ j → ages i ≤ ages j) :
  ages 0 + ages 1 = 14 := by
  sorry

end NUMINAMATH_CALUDE_cousins_ages_sum_l3864_386431


namespace NUMINAMATH_CALUDE_largest_integer_cube_less_than_triple_square_l3864_386428

theorem largest_integer_cube_less_than_triple_square :
  ∀ n : ℤ, n > 2 → n^3 ≥ 3*n^2 ∧ 2^3 < 3*2^2 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_cube_less_than_triple_square_l3864_386428


namespace NUMINAMATH_CALUDE_negative_roots_range_l3864_386485

theorem negative_roots_range (a : ℝ) : 
  (∃ x : ℝ, x < 0 ∧ (1/2)^x = 3*a + 2) → a > -1/3 :=
by sorry

end NUMINAMATH_CALUDE_negative_roots_range_l3864_386485


namespace NUMINAMATH_CALUDE_class_average_l3864_386448

theorem class_average (total_students : ℕ) 
                      (top_scorers : ℕ) 
                      (zero_scorers : ℕ) 
                      (top_score : ℝ) 
                      (rest_average : ℝ) :
  total_students = 25 →
  top_scorers = 5 →
  zero_scorers = 3 →
  top_score = 95 →
  rest_average = 45 →
  let rest_students := total_students - top_scorers - zero_scorers
  let total_score := top_scorers * top_score + zero_scorers * 0 + rest_students * rest_average
  total_score / total_students = 49.6 := by
  sorry

end NUMINAMATH_CALUDE_class_average_l3864_386448


namespace NUMINAMATH_CALUDE_min_c_value_l3864_386472

theorem min_c_value (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_order : a < b ∧ b < c)
  (h_unique : ∃! (x y : ℝ), 2 * x + y = 2023 ∧ y = |x - a| + |x - b| + |x - c|) :
  c ≥ 1012 ∧ ∃ (a' b' : ℕ), 0 < a' ∧ 0 < b' ∧ a' < b' ∧ b' < 1012 ∧
    ∃! (x y : ℝ), 2 * x + y = 2023 ∧ y = |x - a'| + |x - b'| + |x - 1012| :=
by sorry

end NUMINAMATH_CALUDE_min_c_value_l3864_386472


namespace NUMINAMATH_CALUDE_find_M_l3864_386427

theorem find_M : ∃ M : ℚ, (10 + 11 + 12) / 3 = (2024 + 2025 + 2026) / M ∧ M = 552 := by
  sorry

end NUMINAMATH_CALUDE_find_M_l3864_386427


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_l3864_386429

-- Define the concept of opposite
def opposite (a : ℝ) : ℝ := -a

-- Theorem statement
theorem opposite_of_negative_two : opposite (-2) = 2 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_l3864_386429


namespace NUMINAMATH_CALUDE_shooting_competition_solution_l3864_386468

/-- Represents the number of shots for each score (8, 9, 10) -/
structure ScoreCounts where
  eight : ℕ
  nine : ℕ
  ten : ℕ

/-- Checks if a ScoreCounts satisfies the competition conditions -/
def is_valid_score (s : ScoreCounts) : Prop :=
  s.eight + s.nine + s.ten > 11 ∧
  8 * s.eight + 9 * s.nine + 10 * s.ten = 100

/-- The set of all valid score combinations -/
def valid_scores : Set ScoreCounts :=
  { s | is_valid_score s }

/-- The theorem stating the unique solution to the shooting competition problem -/
theorem shooting_competition_solution :
  valid_scores = { ⟨10, 0, 2⟩, ⟨9, 2, 1⟩, ⟨8, 4, 0⟩ } :=
sorry

end NUMINAMATH_CALUDE_shooting_competition_solution_l3864_386468


namespace NUMINAMATH_CALUDE_division_remainder_l3864_386411

theorem division_remainder (x y u v : ℕ) (hx : x > 0) (hy : y > 0) 
  (h_div : x = u * y + v) (h_rem : v < y) : 
  ∃ (q : ℕ), 2 * x + 3 * u * y = q * y + (if 2 * v < y then 2 * v else 2 * v - y) := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l3864_386411


namespace NUMINAMATH_CALUDE_rectangle_area_theorem_l3864_386446

theorem rectangle_area_theorem (m : ℕ) (hm : m > 12) :
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧
  (x * y > m) ∧
  ((x - 1) * y < m) ∧
  (x * (y - 1) < m) ∧
  (∀ (a b : ℕ), a > 0 → b > 0 → a * b ≠ m) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_theorem_l3864_386446


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3864_386476

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 2 + a 4 + a 9 + a 11 = 32) →
  (a 6 + a 7 = 16) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3864_386476


namespace NUMINAMATH_CALUDE_chessboard_not_fully_covered_l3864_386434

/-- Represents the dimensions of a square chessboard -/
def BoardSize : ℕ := 10

/-- Represents the number of squares covered by one L-shaped tromino piece -/
def SquaresPerPiece : ℕ := 3

/-- Represents the number of L-shaped tromino pieces available -/
def NumberOfPieces : ℕ := 25

/-- Theorem stating that the chessboard cannot be fully covered by the given pieces -/
theorem chessboard_not_fully_covered :
  NumberOfPieces * SquaresPerPiece < BoardSize * BoardSize := by
  sorry

end NUMINAMATH_CALUDE_chessboard_not_fully_covered_l3864_386434


namespace NUMINAMATH_CALUDE_solution_set_l3864_386453

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : Differentiable ℝ f)
variable (h2 : ∀ x > 0, x * (deriv (deriv f) x) < f x)
variable (h3 : f 1 = 0)

-- Define the theorem
theorem solution_set (x : ℝ) :
  {x : ℝ | x > 0 ∧ f x / x < 0} = {x : ℝ | x > 1} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_l3864_386453


namespace NUMINAMATH_CALUDE_greatest_possible_median_l3864_386496

theorem greatest_possible_median (k m r s t : ℕ) : 
  k > 0 → m > 0 → r > 0 → s > 0 → t > 0 →
  (k + m + r + s + t) / 5 = 16 →
  k < m → m < r → r < s → s < t →
  t = 42 →
  r ≤ 32 ∧ ∃ (k' m' r' s' : ℕ), 
    k' > 0 ∧ m' > 0 ∧ r' > 0 ∧ s' > 0 ∧
    (k' + m' + r' + s' + 42) / 5 = 16 ∧
    k' < m' ∧ m' < r' ∧ r' < s' ∧ s' < 42 ∧
    r' = 32 :=
by sorry

end NUMINAMATH_CALUDE_greatest_possible_median_l3864_386496


namespace NUMINAMATH_CALUDE_unique_solution_l3864_386478

/-- The exponent function for our problem -/
def f (m : ℕ+) : ℤ := m^2 - 2*m - 3

/-- The condition that the exponent is negative -/
def condition1 (m : ℕ+) : Prop := f m < 0

/-- The condition that the exponent is odd -/
def condition2 (m : ℕ+) : Prop := ∃ k : ℤ, f m = 2*k + 1

/-- The theorem stating that 2 is the only positive integer satisfying all conditions -/
theorem unique_solution :
  ∃! m : ℕ+, condition1 m ∧ condition2 m ∧ m = 2 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l3864_386478


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3864_386458

-- Define set A
def A : Set ℝ := {y | ∃ x, y = Real.cos x}

-- Define set B
def B : Set ℝ := {x | x^2 + x ≥ 0}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = Set.Icc 0 1 ∪ {-1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3864_386458


namespace NUMINAMATH_CALUDE_survey_respondents_l3864_386426

/-- Represents the number of people preferring each brand in a survey. -/
structure SurveyPreferences where
  x : ℕ
  y : ℕ
  z : ℕ

/-- Calculates the total number of respondents in a survey. -/
def totalRespondents (prefs : SurveyPreferences) : ℕ :=
  prefs.x + prefs.y + prefs.z

/-- Theorem stating the total number of respondents in the survey. -/
theorem survey_respondents : ∃ (prefs : SurveyPreferences), 
  prefs.x = 360 ∧ 
  prefs.x * 4 = prefs.y * 9 ∧ 
  prefs.x * 3 = prefs.z * 9 ∧ 
  totalRespondents prefs = 640 := by
  sorry


end NUMINAMATH_CALUDE_survey_respondents_l3864_386426


namespace NUMINAMATH_CALUDE_min_K_is_two_l3864_386483

-- Define the function f
def f (x : ℝ) : ℝ := 2 - x - x^2

-- Define the property that f_K(x) = f(x) for all x ≥ 0
def f_K_equals_f (K : ℝ) : Prop :=
  ∀ x : ℝ, x ≥ 0 → f x ≤ K

-- Theorem statement
theorem min_K_is_two :
  ∃ K : ℝ, (f_K_equals_f K ∧ ∀ K' : ℝ, K' < K → ¬f_K_equals_f K') ∧ K = 2 :=
sorry

end NUMINAMATH_CALUDE_min_K_is_two_l3864_386483


namespace NUMINAMATH_CALUDE_inscribed_circumscribed_ratio_l3864_386449

/-- Given a right-angled triangle with perpendicular sides of 6 and 8,
    prove that the ratio of the radius of the inscribed circle
    to the radius of the circumscribed circle is 2:5 -/
theorem inscribed_circumscribed_ratio (a b c r R : ℝ) : 
  a = 6 → b = 8 → c^2 = a^2 + b^2 → 
  r = (a + b - c) / 2 → R = c / 2 → 
  r / R = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circumscribed_ratio_l3864_386449


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_increase_l3864_386409

/-- The area increase of an equilateral triangle -/
theorem equilateral_triangle_area_increase :
  ∀ (s : ℝ),
  s > 0 →
  s^2 * Real.sqrt 3 / 4 = 100 * Real.sqrt 3 →
  let new_s := s + 3
  let new_area := new_s^2 * Real.sqrt 3 / 4
  let initial_area := 100 * Real.sqrt 3
  new_area - initial_area = 32.25 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_increase_l3864_386409


namespace NUMINAMATH_CALUDE_cubic_root_sum_theorem_l3864_386436

theorem cubic_root_sum_theorem (x : ℝ) (p q : ℤ) : 
  (Real.rpow x (1/3 : ℝ) + Real.rpow (30 - x) (1/3 : ℝ) = 2) →
  (∃ (p q : ℤ), x = p - Real.sqrt q) →
  (p + q = 48) :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_sum_theorem_l3864_386436


namespace NUMINAMATH_CALUDE_solution_set_f_range_of_a_l3864_386445

-- Part 1
def f (x : ℝ) : ℝ := |2 * x + 1| + |2 * x - 3|

theorem solution_set_f (x : ℝ) : f x = 4 ↔ -1/2 ≤ x ∧ x ≤ 3/2 := by sorry

-- Part 2
def f' (a x : ℝ) : ℝ := |x + a| + |x - 1|
def g (x : ℝ) : ℝ := |x - 2| + 1

theorem range_of_a (a : ℝ) :
  (∀ x₁, ∃ x₂, g x₂ = f' a x₁) → a ≤ -2 ∨ a ≥ 0 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_range_of_a_l3864_386445


namespace NUMINAMATH_CALUDE_line_equation_with_parallel_intersections_l3864_386441

/-- The equation of a line passing through point P(1,2) and intersecting two parallel lines,
    forming a line segment of length √2. --/
theorem line_equation_with_parallel_intersections
  (l : Set (ℝ × ℝ))  -- The line we're looking for
  (P : ℝ × ℝ)        -- Point P
  (l₁ l₂ : Set (ℝ × ℝ))  -- The two parallel lines
  (A B : ℝ × ℝ)      -- Points of intersection
  (h_P : P = (1, 2))
  (h_l₁ : l₁ = {(x, y) : ℝ × ℝ | 4*x + 3*y + 1 = 0})
  (h_l₂ : l₂ = {(x, y) : ℝ × ℝ | 4*x + 3*y + 6 = 0})
  (h_A : A ∈ l ∩ l₁)
  (h_B : B ∈ l ∩ l₂)
  (h_dist : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 2)
  (h_P_on_l : P ∈ l) :
  l = {(x, y) : ℝ × ℝ | 7*x - y - 5 = 0} ∨
  l = {(x, y) : ℝ × ℝ | x + 7*y - 15 = 0} :=
sorry

end NUMINAMATH_CALUDE_line_equation_with_parallel_intersections_l3864_386441


namespace NUMINAMATH_CALUDE_trig_equation_solution_l3864_386437

theorem trig_equation_solution (z : ℝ) :
  (1 - Real.sin z ^ 6 - Real.cos z ^ 6) / (1 - Real.sin z ^ 4 - Real.cos z ^ 4) = 2 * (Real.cos (3 * z)) ^ 2 →
  ∃ k : ℤ, z = π / 18 * (6 * ↑k + 1) ∨ z = π / 18 * (6 * ↑k - 1) :=
by sorry

end NUMINAMATH_CALUDE_trig_equation_solution_l3864_386437


namespace NUMINAMATH_CALUDE_xy_reciprocal_inequality_l3864_386471

theorem xy_reciprocal_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (1 + x) * (1 + y) = 2) : x * y + 1 / (x * y) ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_xy_reciprocal_inequality_l3864_386471


namespace NUMINAMATH_CALUDE_cubic_factorization_l3864_386490

theorem cubic_factorization (x : ℝ) : x^3 - 4*x = x*(x+2)*(x-2) := by sorry

end NUMINAMATH_CALUDE_cubic_factorization_l3864_386490


namespace NUMINAMATH_CALUDE_circle_diameter_ratio_l3864_386413

theorem circle_diameter_ratio (R S : Real) (harea : R^2 = 0.36 * S^2) :
  R = 0.6 * S := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_ratio_l3864_386413


namespace NUMINAMATH_CALUDE_fractional_part_equality_l3864_386487

/-- Given k = 2 + √3, prove that k^n - ⌊k^n⌋ = 1 - 1/k^n for any natural number n. -/
theorem fractional_part_equality (n : ℕ) : 
  let k : ℝ := 2 + Real.sqrt 3
  (k^n : ℝ) - ⌊k^n⌋ = 1 - 1 / (k^n) := by
  sorry

end NUMINAMATH_CALUDE_fractional_part_equality_l3864_386487


namespace NUMINAMATH_CALUDE_parallel_lines_a_eq_neg_one_l3864_386416

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, m₁ * x + y = b₁ ↔ m₂ * x + y = b₂) ↔ m₁ = m₂

/-- The slope of a line ax + by + c = 0 is -a/b when b ≠ 0 -/
axiom line_slope {a b c : ℝ} (hb : b ≠ 0) :
  ∀ x y : ℝ, a * x + b * y + c = 0 ↔ y = -a/b * x - c/b

theorem parallel_lines_a_eq_neg_one (a : ℝ) :
  (∀ x y : ℝ, a * x + 2 * y + 6 = 0 ↔ x + (a - 1) * y + a^2 - 1 = 0) →
  a = -1 :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_a_eq_neg_one_l3864_386416


namespace NUMINAMATH_CALUDE_fraction_simplification_l3864_386447

theorem fraction_simplification (a b m n : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hm : m ≠ 0) (hn : n ≠ 0) :
  (a^2 * b) / (m * n^2) / ((a * b) / (3 * m * n)) = 3 * a / n :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3864_386447


namespace NUMINAMATH_CALUDE_unique_positive_integer_pair_l3864_386495

theorem unique_positive_integer_pair : 
  ∃! (a b : ℕ+), 
    (b ^ 2 + b + 1 : ℤ) ≡ 0 [ZMOD a] ∧ 
    (a ^ 2 + a + 1 : ℤ) ≡ 0 [ZMOD b] ∧
    a = 1 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_integer_pair_l3864_386495


namespace NUMINAMATH_CALUDE_soda_bottle_difference_l3864_386402

/-- The number of regular soda bottles in the grocery store. -/
def regular_soda : ℕ := 67

/-- The number of diet soda bottles in the grocery store. -/
def diet_soda : ℕ := 9

/-- The difference between the number of regular soda bottles and diet soda bottles. -/
def soda_difference : ℕ := regular_soda - diet_soda

theorem soda_bottle_difference : soda_difference = 58 := by
  sorry

end NUMINAMATH_CALUDE_soda_bottle_difference_l3864_386402


namespace NUMINAMATH_CALUDE_margin_expression_l3864_386414

/-- Given a selling price S, a ratio m, and a cost C, prove that the margin M
    can be expressed as (1/m)S. -/
theorem margin_expression (S m : ℝ) (h_m : m ≠ 0) :
  let M := (1 / m) * S
  let C := S - M
  M = (1 / m) * S := by sorry

end NUMINAMATH_CALUDE_margin_expression_l3864_386414


namespace NUMINAMATH_CALUDE_original_number_l3864_386456

theorem original_number (N : ℤ) : 
  (∃ k : ℤ, N + 4 = 25 * k) ∧ 
  (∀ m : ℤ, m < 4 → ¬(∃ j : ℤ, N + m = 25 * j)) →
  N = 21 := by
sorry

end NUMINAMATH_CALUDE_original_number_l3864_386456


namespace NUMINAMATH_CALUDE_fraction_equality_l3864_386465

theorem fraction_equality : (3 * 4 + 5) / 7 = 17 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3864_386465


namespace NUMINAMATH_CALUDE_triangle_properties_l3864_386460

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π / 2 →
  0 < B ∧ B < π →
  0 < C ∧ C < π / 2 →
  A + B + C = π →
  4 * a = Real.sqrt 5 * c →
  Real.cos C = 3 / 5 →
  b = 11 →
  Real.sin A = Real.sqrt 5 / 5 ∧
  1 / 2 * a * b * Real.sin C = 22 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3864_386460


namespace NUMINAMATH_CALUDE_smallest_n_for_f_greater_than_15_l3864_386444

-- Define the function f
def f (n : ℕ+) : ℕ := sorry

-- Theorem statement
theorem smallest_n_for_f_greater_than_15 :
  (∀ k : ℕ+, k < 4 → f k ≤ 15) ∧ f 4 > 15 := by sorry

end NUMINAMATH_CALUDE_smallest_n_for_f_greater_than_15_l3864_386444


namespace NUMINAMATH_CALUDE_equation_solutions_l3864_386425

theorem equation_solutions :
  (∀ x : ℚ, (1/2 * x - 2 = 4 + 1/3 * x) ↔ (x = 36)) ∧
  (∀ x : ℚ, ((x - 1) / 4 - 2 = (2 * x - 3) / 6) ↔ (x = -21)) ∧
  (∀ x : ℚ, (1/3 * (x - 1/2 * (x - 1)) = 2/3 * (x - 1/2)) ↔ (x = 1)) ∧
  (∀ x : ℚ, (x / (7/10) - (17/100 - 1/5 * x) / (3/100) = 1) ↔ (x = 14/17)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3864_386425


namespace NUMINAMATH_CALUDE_noahs_lights_l3864_386440

theorem noahs_lights (W : ℝ) 
  (h1 : W > 0)  -- Assuming W is positive
  (h2 : 2 * W + 2 * (3 * W) + 2 * (4 * W) = 96) : W = 6 := by
  sorry

end NUMINAMATH_CALUDE_noahs_lights_l3864_386440


namespace NUMINAMATH_CALUDE_average_speed_calculation_l3864_386407

/-- Proves that given the average speed from y to x and the average speed for the whole journey,
    we can determine the average speed from x to y. -/
theorem average_speed_calculation (speed_y_to_x : ℝ) (speed_round_trip : ℝ) (speed_x_to_y : ℝ) :
  speed_y_to_x = 36 →
  speed_round_trip = 39.6 →
  speed_x_to_y = 44 :=
by sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l3864_386407


namespace NUMINAMATH_CALUDE_equilateral_iff_complex_equation_l3864_386484

/-- A primitive cube root of unity -/
noncomputable def w : ℂ := Complex.exp (Complex.I * (2 * Real.pi / 3))

/-- Definition of an equilateral triangle in the complex plane -/
def is_equilateral (z₁ z₂ z₃ : ℂ) : Prop :=
  Complex.abs (z₂ - z₁) = Complex.abs (z₃ - z₂) ∧
  Complex.abs (z₃ - z₂) = Complex.abs (z₁ - z₃)

/-- Definition of counterclockwise orientation -/
def is_counterclockwise (z₁ z₂ z₃ : ℂ) : Prop :=
  (z₂ - z₁).arg < (z₃ - z₁).arg ∧ (z₃ - z₁).arg < (z₂ - z₁).arg + Real.pi

/-- Theorem: A triangle is equilateral iff it satisfies the given complex equation -/
theorem equilateral_iff_complex_equation (z₁ z₂ z₃ : ℂ) :
  is_counterclockwise z₁ z₂ z₃ →
  is_equilateral z₁ z₂ z₃ ↔ z₁ + w * z₂ + w^2 * z₃ = 0 :=
by sorry

end NUMINAMATH_CALUDE_equilateral_iff_complex_equation_l3864_386484


namespace NUMINAMATH_CALUDE_journey_distance_l3864_386463

theorem journey_distance (total_time : ℝ) (speed1 speed2 speed3 : ℝ) :
  total_time = 36 ∧ 
  speed1 = 21 ∧ 
  speed2 = 45 ∧ 
  speed3 = 24 → 
  ∃ (distance : ℝ),
    distance = 972 ∧
    total_time = distance / (3 * speed1) + distance / (3 * speed2) + distance / (3 * speed3) :=
by sorry

end NUMINAMATH_CALUDE_journey_distance_l3864_386463


namespace NUMINAMATH_CALUDE_f_properties_l3864_386404

noncomputable def f (x : ℝ) : ℝ := x - Real.log x - 1

theorem f_properties :
  (∀ x > 0, f x ≥ 0) ∧
  (∀ p : ℝ, (∀ x ≥ 1, f (1/x) ≥ (Real.log x)^2 / (p + Real.log x)) ↔ p ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l3864_386404


namespace NUMINAMATH_CALUDE_abs_greater_necessary_not_sufficient_l3864_386457

theorem abs_greater_necessary_not_sufficient (a b : ℝ) :
  (∀ a b : ℝ, a > b → |a| > b) ∧
  (∃ a b : ℝ, |a| > b ∧ ¬(a > b)) :=
by sorry

end NUMINAMATH_CALUDE_abs_greater_necessary_not_sufficient_l3864_386457
