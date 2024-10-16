import Mathlib

namespace NUMINAMATH_CALUDE_sin_cos_330_degrees_l3853_385349

theorem sin_cos_330_degrees :
  Real.sin (330 * π / 180) = -1/2 ∧ Real.cos (330 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_330_degrees_l3853_385349


namespace NUMINAMATH_CALUDE_larger_number_problem_l3853_385360

theorem larger_number_problem (s l : ℝ) : 
  s = 48 → l - s = (1 / 3) * l → l = 72 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l3853_385360


namespace NUMINAMATH_CALUDE_adolfo_blocks_l3853_385359

theorem adolfo_blocks (initial_blocks added_blocks : ℕ) 
  (h1 : initial_blocks = 35)
  (h2 : added_blocks = 30) :
  initial_blocks + added_blocks = 65 := by
  sorry

end NUMINAMATH_CALUDE_adolfo_blocks_l3853_385359


namespace NUMINAMATH_CALUDE_parabola_properties_l3853_385326

/-- A parabola passing through two points with a specific translation and minimum value -/
theorem parabola_properties (a b : ℝ) (m : ℝ) :
  (a * 1^2 + b * 1 + 1 = -2) →
  (a * (-2)^2 + b * (-2) + 1 = 13) →
  (m > 0) →
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 3 → 
    (x - m)^2 - 4*(x - m) + 1 ≥ 6) →
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 3 ∧ 
    (x - m)^2 - 4*(x - m) + 1 = 6) →
  ((a = 1 ∧ b = -4) ∧ (m = 6 ∨ m = 4)) := by
sorry

end NUMINAMATH_CALUDE_parabola_properties_l3853_385326


namespace NUMINAMATH_CALUDE_solution_set_equality_l3853_385364

/-- The solution set of the inequality (a^2 - 1)x^2 - (a - 1)x - 1 < 0 is equal to ℝ if and only if -3/5 < a < 1 -/
theorem solution_set_equality (a : ℝ) : 
  (∀ x : ℝ, (a^2 - 1) * x^2 - (a - 1) * x - 1 < 0) ↔ (-3/5 < a ∧ a < 1) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equality_l3853_385364


namespace NUMINAMATH_CALUDE_square_perimeter_from_area_l3853_385363

theorem square_perimeter_from_area (area : ℝ) (side : ℝ) (perimeter : ℝ) : 
  area = 450 → 
  side * side = area → 
  perimeter = 4 * side → 
  perimeter = 60 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_from_area_l3853_385363


namespace NUMINAMATH_CALUDE_percent_relation_l3853_385307

theorem percent_relation (a b c : ℝ) 
  (h1 : c = 0.1 * b) 
  (h2 : b = 2 * a) : 
  c = 0.2 * a := by
sorry

end NUMINAMATH_CALUDE_percent_relation_l3853_385307


namespace NUMINAMATH_CALUDE_paul_spent_three_tickets_l3853_385372

/-- Represents the number of tickets Paul spent on the Ferris wheel -/
def tickets_spent (initial : ℕ) (left : ℕ) : ℕ := initial - left

/-- Theorem stating that Paul spent 3 tickets on the Ferris wheel -/
theorem paul_spent_three_tickets :
  tickets_spent 11 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_paul_spent_three_tickets_l3853_385372


namespace NUMINAMATH_CALUDE_smallest_three_digit_square_base_seven_l3853_385361

/-- The smallest integer whose square has exactly 3 digits in base 7 -/
def M : ℕ := 7

/-- Converts a natural number to its base 7 representation -/
def to_base_seven (n : ℕ) : List ℕ := sorry

/-- Checks if a number has exactly 3 digits when written in base 7 -/
def has_three_digits_base_seven (n : ℕ) : Prop :=
  (to_base_seven n).length = 3

theorem smallest_three_digit_square_base_seven :
  (M ^ 2 ≥ 7^2) ∧
  (M ^ 2 < 7^3) ∧
  (∀ k : ℕ, k < M → ¬(has_three_digits_base_seven (k^2))) ∧
  (to_base_seven M = [1, 0]) := by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_square_base_seven_l3853_385361


namespace NUMINAMATH_CALUDE_product_inequality_l3853_385392

theorem product_inequality (x₁ x₂ x₃ x₄ y₁ y₂ : ℝ) 
  (h1 : y₂ ≥ y₁ ∧ y₁ ≥ x₁ ∧ x₁ ≥ x₃ ∧ x₃ ≥ x₂ ∧ x₂ ≥ x₁ ∧ x₁ ≥ 2)
  (h2 : x₁ + x₂ + x₃ + x₄ ≥ y₁ + y₂) :
  x₁ * x₂ * x₃ * x₄ ≥ y₁ * y₂ := by
sorry

end NUMINAMATH_CALUDE_product_inequality_l3853_385392


namespace NUMINAMATH_CALUDE_cycle_cut_orthogonality_l3853_385333

-- Define a graph
structure Graph where
  V : Type
  E : Type
  incident : E → V → Prop

-- Define cycle space and cut space
def CycleSpace (G : Graph) : Type := sorry
def CutSpace (G : Graph) : Type := sorry

-- Define orthogonal complement
def OrthogonalComplement (S : Type) : Type := sorry

-- State the theorem
theorem cycle_cut_orthogonality (G : Graph) :
  (CycleSpace G = OrthogonalComplement (CutSpace G)) ∧
  (CutSpace G = OrthogonalComplement (CycleSpace G)) := by
  sorry

end NUMINAMATH_CALUDE_cycle_cut_orthogonality_l3853_385333


namespace NUMINAMATH_CALUDE_fourth_term_of_arithmetic_sequence_l3853_385350

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem fourth_term_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_first : a 1 = 13)
  (h_last : a 6 = 49) :
  a 4 = 31 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_of_arithmetic_sequence_l3853_385350


namespace NUMINAMATH_CALUDE_henrys_initial_book_count_l3853_385398

/-- Calculates the initial number of books in Henry's collection --/
def initialBookCount (boxCount : ℕ) (booksPerBox : ℕ) (roomBooks : ℕ) (tableBooks : ℕ) (kitchenBooks : ℕ) (pickedUpBooks : ℕ) (remainingBooks : ℕ) : ℕ :=
  boxCount * booksPerBox + roomBooks + tableBooks + kitchenBooks - pickedUpBooks + remainingBooks

/-- Theorem stating that Henry's initial book count is 99 --/
theorem henrys_initial_book_count :
  initialBookCount 3 15 21 4 18 12 23 = 99 := by
  sorry

end NUMINAMATH_CALUDE_henrys_initial_book_count_l3853_385398


namespace NUMINAMATH_CALUDE_maria_total_money_l3853_385351

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- The number of dimes Maria has initially -/
def initial_dimes : ℕ := 4

/-- The number of quarters Maria has initially -/
def initial_quarters : ℕ := 4

/-- The number of nickels Maria has initially -/
def initial_nickels : ℕ := 7

/-- The number of quarters Maria's mom gives her -/
def additional_quarters : ℕ := 5

/-- The total amount of money Maria has after receiving the additional quarters -/
theorem maria_total_money :
  (initial_dimes * dime_value +
   initial_quarters * quarter_value +
   initial_nickels * nickel_value +
   additional_quarters * quarter_value) = 3 :=
by sorry

end NUMINAMATH_CALUDE_maria_total_money_l3853_385351


namespace NUMINAMATH_CALUDE_f_inequality_solution_range_l3853_385377

-- Define the function f
def f (x m : ℝ) : ℝ := -x^2 + x + m + 2

-- Define the property of having exactly one integer solution
def has_exactly_one_integer_solution (m : ℝ) : Prop :=
  ∃! (n : ℤ), f n m ≥ |n|

-- State the theorem
theorem f_inequality_solution_range :
  ∀ m : ℝ, has_exactly_one_integer_solution m ↔ m ∈ Set.Icc (-2) (-1) := by sorry

end NUMINAMATH_CALUDE_f_inequality_solution_range_l3853_385377


namespace NUMINAMATH_CALUDE_temperature_difference_l3853_385303

theorem temperature_difference (lowest highest : ℤ) 
  (h_lowest : lowest = -11)
  (h_highest : highest = -3) :
  highest - lowest = 8 := by
sorry

end NUMINAMATH_CALUDE_temperature_difference_l3853_385303


namespace NUMINAMATH_CALUDE_valid_parts_characterization_valid_parts_complete_l3853_385394

/-- A type representing the possible numbers of equal parts. -/
inductive ValidParts : Nat → Prop where
  | two : ValidParts 2
  | three : ValidParts 3
  | four : ValidParts 4
  | six : ValidParts 6
  | eight : ValidParts 8
  | twelve : ValidParts 12
  | twentyfour : ValidParts 24

/-- The total number of cells in the figure. -/
def totalCells : Nat := 24

/-- A function that checks if a number divides the total number of cells evenly. -/
def isDivisor (n : Nat) : Prop := totalCells % n = 0

/-- The main theorem stating that the valid numbers of parts are exactly those that divide the total number of cells evenly. -/
theorem valid_parts_characterization (n : Nat) : 
  ValidParts n ↔ (isDivisor n ∧ n > 1) :=
sorry

/-- The theorem stating that the list of valid parts is complete. -/
theorem valid_parts_complete : 
  ∀ n, isDivisor n ∧ n > 1 → ValidParts n :=
sorry

end NUMINAMATH_CALUDE_valid_parts_characterization_valid_parts_complete_l3853_385394


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l3853_385324

theorem unique_solution_for_equation (x : ℝ) :
  x ≥ 0 →
  (2021 * (x^2020)^(1/202) - 1 = 2020 * x) ↔
  x = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l3853_385324


namespace NUMINAMATH_CALUDE_slope_negative_one_implies_y_coordinate_l3853_385302

/-- Given two points P and Q in a coordinate plane, if the slope of the line through P and Q is -1, then the y-coordinate of Q is -3. -/
theorem slope_negative_one_implies_y_coordinate (x1 y1 x2 y2 : ℝ) :
  x1 = -3 →
  y1 = 5 →
  x2 = 5 →
  (y2 - y1) / (x2 - x1) = -1 →
  y2 = -3 := by
sorry

end NUMINAMATH_CALUDE_slope_negative_one_implies_y_coordinate_l3853_385302


namespace NUMINAMATH_CALUDE_scientific_notation_equivalence_l3853_385342

theorem scientific_notation_equivalence : 0.0000036 = 3.6 * 10^(-6) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equivalence_l3853_385342


namespace NUMINAMATH_CALUDE_smallest_solution_quartic_l3853_385387

theorem smallest_solution_quartic (x : ℝ) :
  x^4 - 50*x^2 + 576 = 0 → x ≥ -Real.sqrt 26 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_quartic_l3853_385387


namespace NUMINAMATH_CALUDE_sqrt_five_lt_sqrt_two_plus_one_l3853_385334

theorem sqrt_five_lt_sqrt_two_plus_one : Real.sqrt 5 < Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_five_lt_sqrt_two_plus_one_l3853_385334


namespace NUMINAMATH_CALUDE_horner_v3_value_l3853_385329

/-- The polynomial f(x) = 7x^7 + 6x^6 + 5x^5 + 4x^4 + 3x^3 + 2x^2 + x -/
def f (x : ℝ) : ℝ := 7*x^7 + 6*x^6 + 5*x^5 + 4*x^4 + 3*x^3 + 2*x^2 + x

/-- The value of v_3 in Horner's method -/
def v_3 (x : ℝ) : ℝ := (((7*x + 6)*x + 5)*x + 4)

/-- Theorem: The value of v_3 is 262 when x = 3 -/
theorem horner_v3_value : v_3 3 = 262 := by
  sorry

end NUMINAMATH_CALUDE_horner_v3_value_l3853_385329


namespace NUMINAMATH_CALUDE_gcd_of_390_455_546_l3853_385389

theorem gcd_of_390_455_546 : Nat.gcd 390 (Nat.gcd 455 546) = 13 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_390_455_546_l3853_385389


namespace NUMINAMATH_CALUDE_benches_around_circular_track_l3853_385391

/-- The radius of the circular walking track in feet -/
def radius : ℝ := 15

/-- The spacing between benches in feet -/
def bench_spacing : ℝ := 3

/-- The number of benches needed for the circular track -/
def num_benches : ℕ := 31

/-- Theorem stating that the number of benches needed is approximately 31 -/
theorem benches_around_circular_track :
  Int.floor ((2 * Real.pi * radius) / bench_spacing) = num_benches := by
  sorry

end NUMINAMATH_CALUDE_benches_around_circular_track_l3853_385391


namespace NUMINAMATH_CALUDE_price_increase_over_two_years_l3853_385385

theorem price_increase_over_two_years (a b : ℝ) :
  let first_year_increase := 1 + a / 100
  let second_year_increase := 1 + b / 100
  let total_increase := first_year_increase * second_year_increase - 1
  total_increase = (a + b + a * b / 100) / 100 := by
sorry

end NUMINAMATH_CALUDE_price_increase_over_two_years_l3853_385385


namespace NUMINAMATH_CALUDE_product_inequality_l3853_385300

theorem product_inequality (a a' b b' c c' : ℝ) 
  (h1 : a * a' > 0) 
  (h2 : a * c ≥ b^2) 
  (h3 : a' * c' ≥ b'^2) : 
  (a + a') * (c + c') ≥ (b + b')^2 := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l3853_385300


namespace NUMINAMATH_CALUDE_infinite_impossible_d_l3853_385378

theorem infinite_impossible_d : ∃ (S : Set ℕ), Set.Infinite S ∧
  ∀ (d : ℕ), d ∈ S →
    ¬∃ (t r : ℝ), t > 0 ∧ 3 * t - 2 * Real.pi * r = 500 ∧ t = 2 * r + d :=
sorry

end NUMINAMATH_CALUDE_infinite_impossible_d_l3853_385378


namespace NUMINAMATH_CALUDE_min_value_ab_l3853_385321

theorem min_value_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 4/b = Real.sqrt (a*b)) :
  a * b ≥ 4 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 1/a₀ + 4/b₀ = Real.sqrt (a₀*b₀) ∧ a₀ * b₀ = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_ab_l3853_385321


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3853_385331

-- Define a geometric sequence with positive terms
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∃ r : ℝ, r > 0 ∧ ∀ n, a (n + 1) = r * a n)

-- State the theorem
theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 1) * (a 19) = 16 →
  (a 1) + (a 19) = 10 →
  (a 8) * (a 10) * (a 12) = 64 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l3853_385331


namespace NUMINAMATH_CALUDE_x_intercept_ratio_l3853_385327

/-- Two lines with the same non-zero y-intercept -/
structure TwoLines where
  b : ℝ
  s : ℝ
  t : ℝ
  b_nonzero : b ≠ 0
  line1_equation : 0 = 8 * s + b
  line2_equation : 0 = 4 * t + b

/-- The ratio of x-intercepts is 1/2 -/
theorem x_intercept_ratio (lines : TwoLines) : lines.s / lines.t = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_ratio_l3853_385327


namespace NUMINAMATH_CALUDE_odd_sum_probability_l3853_385315

theorem odd_sum_probability (even_sum_prob odd_sum_prob : ℚ) : 
  (even_sum_prob = 2 * odd_sum_prob) →
  (even_sum_prob + odd_sum_prob = 1) →
  (odd_sum_prob = 4/9) := by
sorry

end NUMINAMATH_CALUDE_odd_sum_probability_l3853_385315


namespace NUMINAMATH_CALUDE_specificPolygonArea_l3853_385367

/-- A point on a 2D grid -/
structure GridPoint where
  x : ℕ
  y : ℕ

/-- A polygon defined by a list of grid points -/
def Polygon := List GridPoint

/-- The polygon formed by connecting specific points on a 4x4 grid -/
def specificPolygon : Polygon :=
  [⟨0,0⟩, ⟨1,0⟩, ⟨1,1⟩, ⟨0,1⟩, ⟨1,2⟩, ⟨0,2⟩, ⟨1,3⟩, ⟨0,3⟩, 
   ⟨3,3⟩, ⟨2,3⟩, ⟨3,2⟩, ⟨2,2⟩, ⟨2,1⟩, ⟨3,1⟩, ⟨3,0⟩, ⟨2,0⟩]

/-- Function to calculate the area of a polygon -/
def calculateArea (p : Polygon) : ℕ := sorry

/-- Theorem stating that the area of the specific polygon is 16 square units -/
theorem specificPolygonArea : calculateArea specificPolygon = 16 := by sorry

end NUMINAMATH_CALUDE_specificPolygonArea_l3853_385367


namespace NUMINAMATH_CALUDE_tin_in_new_alloy_l3853_385365

/-- Calculate the amount of tin in a new alloy formed by mixing two alloys -/
theorem tin_in_new_alloy (alloy_a_mass : ℝ) (alloy_b_mass : ℝ)
  (lead_tin_ratio_a : ℚ) (tin_copper_ratio_b : ℚ) :
  alloy_a_mass = 135 →
  alloy_b_mass = 145 →
  lead_tin_ratio_a = 3 / 5 →
  tin_copper_ratio_b = 2 / 3 →
  let tin_in_a := alloy_a_mass * (5 / 8 : ℝ)
  let tin_in_b := alloy_b_mass * (2 / 5 : ℝ)
  tin_in_a + tin_in_b = 142.375 := by
  sorry

end NUMINAMATH_CALUDE_tin_in_new_alloy_l3853_385365


namespace NUMINAMATH_CALUDE_complement_intersection_equals_set_l3853_385380

def U : Set Nat := {0, 1, 2, 3}
def M : Set Nat := {0, 1, 2}
def N : Set Nat := {1, 2, 3}

theorem complement_intersection_equals_set : (U \ (M ∩ N)) = {0, 3} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_set_l3853_385380


namespace NUMINAMATH_CALUDE_randys_pig_feed_l3853_385314

/-- Calculates the total amount of pig feed for a month -/
def total_pig_feed_per_month (feed_per_pig_per_day : ℕ) (num_pigs : ℕ) (days_in_month : ℕ) : ℕ :=
  feed_per_pig_per_day * num_pigs * days_in_month

/-- Proves that Randy's pigs will be fed 1800 pounds of pig feed per month -/
theorem randys_pig_feed :
  total_pig_feed_per_month 15 4 30 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_randys_pig_feed_l3853_385314


namespace NUMINAMATH_CALUDE_mixed_feed_cost_per_pound_l3853_385376

theorem mixed_feed_cost_per_pound
  (total_weight : ℝ)
  (cheap_cost_per_pound : ℝ)
  (expensive_cost_per_pound : ℝ)
  (cheap_weight : ℝ)
  (h1 : total_weight = 35)
  (h2 : cheap_cost_per_pound = 0.18)
  (h3 : expensive_cost_per_pound = 0.53)
  (h4 : cheap_weight = 17)
  : (cheap_weight * cheap_cost_per_pound + (total_weight - cheap_weight) * expensive_cost_per_pound) / total_weight = 0.36 := by
  sorry

end NUMINAMATH_CALUDE_mixed_feed_cost_per_pound_l3853_385376


namespace NUMINAMATH_CALUDE_complex_z_imaginary_part_l3853_385399

theorem complex_z_imaginary_part (z : ℂ) (h : (3 + 4 * Complex.I) * z = Complex.abs (3 - 4 * Complex.I)) : 
  z.im = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_complex_z_imaginary_part_l3853_385399


namespace NUMINAMATH_CALUDE_triangular_prism_volume_l3853_385308

/-- The volume of a triangular prism given the area of a lateral face and the distance to the opposite edge -/
theorem triangular_prism_volume (A_face : ℝ) (d : ℝ) (h_pos_A : A_face > 0) (h_pos_d : d > 0) :
  ∃ (V : ℝ), V = (1 / 2) * A_face * d ∧ V > 0 := by
  sorry

end NUMINAMATH_CALUDE_triangular_prism_volume_l3853_385308


namespace NUMINAMATH_CALUDE_triangle_sine_property_l3853_385354

theorem triangle_sine_property (A B C : ℝ) (h : 3 * Real.sin B ^ 2 + 7 * Real.sin C ^ 2 = 2 * Real.sin A * Real.sin B * Real.sin C + 2 * Real.sin A ^ 2) :
  Real.sin (A + π / 4) = -Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_property_l3853_385354


namespace NUMINAMATH_CALUDE_sine_product_ratio_equals_one_l3853_385373

theorem sine_product_ratio_equals_one :
  let d : ℝ := 2 * Real.pi / 15
  (Real.sin (4 * d) * Real.sin (6 * d) * Real.sin (8 * d) * Real.sin (10 * d) * Real.sin (12 * d)) /
  (Real.sin (2 * d) * Real.sin (3 * d) * Real.sin (4 * d) * Real.sin (5 * d) * Real.sin (6 * d)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_sine_product_ratio_equals_one_l3853_385373


namespace NUMINAMATH_CALUDE_min_distance_ellipse_line_l3853_385366

/-- The minimum distance between a point on the ellipse x²/3 + y² = 1 and 
    a point on the line x + y = 4, along with the coordinates of the point 
    on the ellipse at this minimum distance. -/
theorem min_distance_ellipse_line :
  let ellipse := {p : ℝ × ℝ | p.1^2 / 3 + p.2^2 = 1}
  let line := {q : ℝ × ℝ | q.1 + q.2 = 4}
  ∃ (p : ℝ × ℝ), p ∈ ellipse ∧ 
    (∀ (p' : ℝ × ℝ) (q : ℝ × ℝ), p' ∈ ellipse → q ∈ line → 
      Real.sqrt 2 ≤ Real.sqrt ((p'.1 - q.1)^2 + (p'.2 - q.2)^2)) ∧
    p = (3/2, 1/2) := by
  sorry

end NUMINAMATH_CALUDE_min_distance_ellipse_line_l3853_385366


namespace NUMINAMATH_CALUDE_sum_of_squares_perfect_square_two_even_l3853_385301

theorem sum_of_squares_perfect_square_two_even (x y z : ℤ) :
  ∃ (u : ℤ), x^2 + y^2 + z^2 = u^2 →
  (Even x ∧ Even y) ∨ (Even x ∧ Even z) ∨ (Even y ∧ Even z) :=
sorry

end NUMINAMATH_CALUDE_sum_of_squares_perfect_square_two_even_l3853_385301


namespace NUMINAMATH_CALUDE_dress_price_discount_l3853_385390

theorem dress_price_discount (P : ℝ) : P > 0 → 
  (1 - 0.35) * (1 - 0.30) * P = 0.455 * P :=
by
  sorry

end NUMINAMATH_CALUDE_dress_price_discount_l3853_385390


namespace NUMINAMATH_CALUDE_complex_simplification_and_multiplication_l3853_385358

theorem complex_simplification_and_multiplication :
  -2 * ((5 - 3 * Complex.I) - (2 + 5 * Complex.I)) = -6 + 16 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_simplification_and_multiplication_l3853_385358


namespace NUMINAMATH_CALUDE_min_value_of_function_l3853_385306

theorem min_value_of_function (x : ℝ) (h : x ∈ Set.Ico 1 2) : 
  (1 / x + 1 / (2 - x)) ≥ 2 ∧ 
  (1 / x + 1 / (2 - x) = 2 ↔ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l3853_385306


namespace NUMINAMATH_CALUDE_euler_family_mean_age_l3853_385371

theorem euler_family_mean_age :
  let ages : List ℕ := [6, 6, 6, 6, 10, 10, 16]
  (List.sum ages) / (List.length ages) = 60 / 7 := by
  sorry

end NUMINAMATH_CALUDE_euler_family_mean_age_l3853_385371


namespace NUMINAMATH_CALUDE_square_area_ratio_l3853_385346

theorem square_area_ratio (side_c side_d : ℝ) (hc : side_c = 24) (hd : side_d = 54) :
  (side_c^2) / (side_d^2) = 16 / 81 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l3853_385346


namespace NUMINAMATH_CALUDE_image_of_A_under_f_l3853_385320

def A : Set ℕ := {1, 2}

def f (x : ℕ) : ℕ := x^2

theorem image_of_A_under_f : Set.image f A = {1, 4} := by sorry

end NUMINAMATH_CALUDE_image_of_A_under_f_l3853_385320


namespace NUMINAMATH_CALUDE_odd_function_property_l3853_385370

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_property (f : ℝ → ℝ) (h1 : is_odd_function f)
    (h2 : ∀ x > 0, f x = x^2 + x - 1) :
    ∀ x < 0, f x = -x^2 + x + 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l3853_385370


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3853_385368

/-- The function f(x) = a^(x-1) + 7 always passes through the point (1, 8) for any a > 0 and a ≠ 1 -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := λ x => a^(x-1) + 7
  f 1 = 8 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3853_385368


namespace NUMINAMATH_CALUDE_squirrel_survey_l3853_385305

theorem squirrel_survey (total : ℕ) 
  (harmful_belief_rate : ℚ) 
  (attack_belief_rate : ℚ) 
  (wrong_believers : ℕ) 
  (h1 : harmful_belief_rate = 883 / 1000) 
  (h2 : attack_belief_rate = 538 / 1000) 
  (h3 : wrong_believers = 28) :
  (↑wrong_believers / (harmful_belief_rate * attack_belief_rate) : ℚ).ceil = total → 
  total = 59 := by
sorry

end NUMINAMATH_CALUDE_squirrel_survey_l3853_385305


namespace NUMINAMATH_CALUDE_small_plate_diameter_l3853_385310

theorem small_plate_diameter
  (big_plate_diameter : ℝ)
  (uncovered_fraction : ℝ)
  (h1 : big_plate_diameter = 12)
  (h2 : uncovered_fraction = 0.3055555555555555) :
  ∃ (small_plate_diameter : ℝ),
    small_plate_diameter = 10 ∧
    (1 - uncovered_fraction) * (π * big_plate_diameter^2 / 4) = π * small_plate_diameter^2 / 4 :=
by sorry

end NUMINAMATH_CALUDE_small_plate_diameter_l3853_385310


namespace NUMINAMATH_CALUDE_negation_equivalence_l3853_385352

theorem negation_equivalence : 
  (¬∃ x : ℝ, (2 / x) + Real.log x ≤ 0) ↔ (∀ x : ℝ, (2 / x) + Real.log x > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3853_385352


namespace NUMINAMATH_CALUDE_truncated_cone_volume_l3853_385325

/-- The volume of a truncated cone formed by cutting a smaller cone from a larger cone -/
theorem truncated_cone_volume (d_large h_large d_small h_small : ℝ) : 
  d_large = 8 → h_large = 10 → d_small = 4 → h_small = 4 →
  (1/3 * π * (d_large/2)^2 * h_large) - (1/3 * π * (d_small/2)^2 * h_small) = 48 * π := by
sorry

end NUMINAMATH_CALUDE_truncated_cone_volume_l3853_385325


namespace NUMINAMATH_CALUDE_smallest_five_digit_negative_congruent_to_one_mod_seventeen_l3853_385344

theorem smallest_five_digit_negative_congruent_to_one_mod_seventeen :
  ∀ n : ℤ, -99999 ≤ n ∧ n < -9999 ∧ n ≡ 1 [ZMOD 17] → n ≥ -10011 :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_negative_congruent_to_one_mod_seventeen_l3853_385344


namespace NUMINAMATH_CALUDE_base_13_conversion_l3853_385313

/-- Represents a digit in base 13 -/
inductive Base13Digit
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | A | B | C

/-- Converts a Base13Digit to its numerical value -/
def base13DigitToNat (d : Base13Digit) : ℕ :=
  match d with
  | Base13Digit.D0 => 0
  | Base13Digit.D1 => 1
  | Base13Digit.D2 => 2
  | Base13Digit.D3 => 3
  | Base13Digit.D4 => 4
  | Base13Digit.D5 => 5
  | Base13Digit.D6 => 6
  | Base13Digit.D7 => 7
  | Base13Digit.D8 => 8
  | Base13Digit.D9 => 9
  | Base13Digit.A => 10
  | Base13Digit.B => 11
  | Base13Digit.C => 12

/-- Converts a two-digit number in base 13 to its decimal (base 10) equivalent -/
def base13ToDecimal (d1 d2 : Base13Digit) : ℕ :=
  13 * (base13DigitToNat d1) + (base13DigitToNat d2)

theorem base_13_conversion :
  base13ToDecimal Base13Digit.C Base13Digit.D1 = 157 := by
  sorry

end NUMINAMATH_CALUDE_base_13_conversion_l3853_385313


namespace NUMINAMATH_CALUDE_counterfeit_identification_possible_l3853_385332

/-- Represents the result of weighing two coins on a balance scale -/
inductive WeighResult
  | Equal : WeighResult
  | LeftLighter : WeighResult
  | RightLighter : WeighResult

/-- Represents a coin, which can be either real or counterfeit -/
inductive Coin
  | Real : Coin
  | Counterfeit : Coin

/-- A function that simulates weighing two coins on a balance scale -/
def weigh (a b : Coin) : WeighResult :=
  match a, b with
  | Coin.Real, Coin.Real => WeighResult.Equal
  | Coin.Counterfeit, Coin.Real => WeighResult.LeftLighter
  | Coin.Real, Coin.Counterfeit => WeighResult.RightLighter
  | Coin.Counterfeit, Coin.Counterfeit => WeighResult.Equal

/-- A function that identifies the counterfeit coin based on one weighing -/
def identifyCounterfeit (coins : Fin 3 → Coin) : Fin 3 :=
  match weigh (coins 0) (coins 1) with
  | WeighResult.Equal => 2
  | WeighResult.LeftLighter => 0
  | WeighResult.RightLighter => 1

theorem counterfeit_identification_possible :
  ∀ (coins : Fin 3 → Coin),
  (∃! i, coins i = Coin.Counterfeit) →
  coins (identifyCounterfeit coins) = Coin.Counterfeit :=
sorry


end NUMINAMATH_CALUDE_counterfeit_identification_possible_l3853_385332


namespace NUMINAMATH_CALUDE_min_balls_for_fifteen_in_box_l3853_385357

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  black : Nat

/-- The minimum number of balls needed to guarantee at least 15 of a single color -/
def minBallsForFifteen (counts : BallCounts) : Nat :=
  sorry

/-- Theorem stating the minimum number of balls to draw for the given problem -/
theorem min_balls_for_fifteen_in_box :
  let counts : BallCounts := {
    red := 28, green := 20, yellow := 19,
    blue := 13, white := 11, black := 9
  }
  minBallsForFifteen counts = 76 := by sorry

end NUMINAMATH_CALUDE_min_balls_for_fifteen_in_box_l3853_385357


namespace NUMINAMATH_CALUDE_lines_parallel_to_same_line_are_parallel_l3853_385379

-- Define a type for lines
def Line := Type

-- Define a relation for parallel lines
def Parallel (l1 l2 : Line) : Prop := sorry

-- Theorem statement
theorem lines_parallel_to_same_line_are_parallel 
  (l1 l2 l3 : Line) : 
  Parallel l1 l3 → Parallel l2 l3 → Parallel l1 l2 := by sorry

end NUMINAMATH_CALUDE_lines_parallel_to_same_line_are_parallel_l3853_385379


namespace NUMINAMATH_CALUDE_investment_ratio_l3853_385356

theorem investment_ratio (p q : ℝ) (h1 : p > 0) (h2 : q > 0) : 
  (p * 10) / (q * 20) = 7 / 10 → p / q = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_investment_ratio_l3853_385356


namespace NUMINAMATH_CALUDE_line_intersection_y_axis_l3853_385343

/-- The line passing through points (2, 10) and (5, 16) intersects the y-axis at (0, 6) -/
theorem line_intersection_y_axis :
  let p₁ : ℝ × ℝ := (2, 10)
  let p₂ : ℝ × ℝ := (5, 16)
  let m : ℝ := (p₂.2 - p₁.2) / (p₂.1 - p₁.1)
  let b : ℝ := p₁.2 - m * p₁.1
  let line (x : ℝ) : ℝ := m * x + b
  (0, line 0) = (0, 6) :=
by sorry

end NUMINAMATH_CALUDE_line_intersection_y_axis_l3853_385343


namespace NUMINAMATH_CALUDE_johnny_wage_l3853_385374

/-- Given a total earning and hours worked, calculates the hourly wage -/
def hourly_wage (total_earning : ℚ) (hours_worked : ℚ) : ℚ :=
  total_earning / hours_worked

theorem johnny_wage :
  let total_earning : ℚ := 33/2  -- $16.5 represented as a rational number
  let hours_worked : ℚ := 2
  hourly_wage total_earning hours_worked = 33/4  -- $8.25 represented as a rational number
:= by sorry

end NUMINAMATH_CALUDE_johnny_wage_l3853_385374


namespace NUMINAMATH_CALUDE_simple_interest_rate_l3853_385309

/-- Proves that given a principal of $600 lent at simple interest for 8 years,
    if the total interest is $360 less than the principal,
    then the annual interest rate is 5%. -/
theorem simple_interest_rate (principal : ℝ) (time : ℝ) (interest : ℝ) (rate : ℝ) :
  principal = 600 →
  time = 8 →
  interest = principal - 360 →
  interest = principal * rate * time →
  rate = 0.05 := by sorry

end NUMINAMATH_CALUDE_simple_interest_rate_l3853_385309


namespace NUMINAMATH_CALUDE_largest_n_for_product_l3853_385328

/-- Arithmetic sequence (a_n) -/
def a (n : ℕ) (d : ℤ) : ℤ := 2 + (n - 1 : ℤ) * d

/-- Arithmetic sequence (b_n) -/
def b (n : ℕ) (e : ℤ) : ℤ := 3 + (n - 1 : ℤ) * e

theorem largest_n_for_product (d e : ℤ) (h1 : a 2 d ≤ b 2 e) :
  (∃ n : ℕ, a n d * b n e = 2728) →
  (∀ m : ℕ, a m d * b m e = 2728 → m ≤ 52) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_product_l3853_385328


namespace NUMINAMATH_CALUDE_greatest_integer_for_fraction_twenty_nine_satisfies_twenty_nine_is_greatest_l3853_385335

def is_integer (x : ℝ) : Prop := ∃ n : ℤ, x = n

theorem greatest_integer_for_fraction : 
  ∀ x : ℤ, (is_integer ((x^2 + 3*x + 8) / (x - 3))) → x ≤ 29 :=
by sorry

theorem twenty_nine_satisfies :
  is_integer ((29^2 + 3*29 + 8) / (29 - 3)) :=
by sorry

theorem twenty_nine_is_greatest :
  ∀ x : ℤ, x > 29 → ¬(is_integer ((x^2 + 3*x + 8) / (x - 3))) :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_for_fraction_twenty_nine_satisfies_twenty_nine_is_greatest_l3853_385335


namespace NUMINAMATH_CALUDE_square_49_equals_square_50_minus_99_l3853_385381

theorem square_49_equals_square_50_minus_99 : 49^2 = 50^2 - 99 := by
  sorry

end NUMINAMATH_CALUDE_square_49_equals_square_50_minus_99_l3853_385381


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l3853_385353

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 17 = 0 → n ≤ 986 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l3853_385353


namespace NUMINAMATH_CALUDE_coin_collection_problem_l3853_385340

/-- Represents the state of a coin collection --/
structure CoinCollection where
  gold : ℕ
  silver : ℕ

/-- Calculates the ratio of gold to silver coins --/
def goldSilverRatio (c : CoinCollection) : ℚ :=
  c.gold / c.silver

/-- Represents the coin collection problem --/
theorem coin_collection_problem 
  (initial : CoinCollection)
  (final : CoinCollection)
  (added_gold : ℕ) :
  goldSilverRatio initial = 1 / 3 →
  goldSilverRatio final = 1 / 2 →
  final.gold + final.silver = 135 →
  final.gold = initial.gold + added_gold →
  final.silver = initial.silver →
  added_gold = 15 := by
  sorry


end NUMINAMATH_CALUDE_coin_collection_problem_l3853_385340


namespace NUMINAMATH_CALUDE_quadradois_theorem_l3853_385339

/-- A number is quadradois if a square can be divided into that many squares of at most two different sizes. -/
def IsQuadradois (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a * a + b * b = n ∧ (a = 0 ∨ b = 0 ∨ a ≠ b)

theorem quadradois_theorem :
  IsQuadradois 6 ∧ 
  IsQuadradois 2015 ∧ 
  ∀ n : ℕ, n > 5 → IsQuadradois n :=
by sorry

end NUMINAMATH_CALUDE_quadradois_theorem_l3853_385339


namespace NUMINAMATH_CALUDE_lemonade_water_amount_solution_is_correct_l3853_385355

/-- Represents the recipe for lemonade --/
structure LemonadeRecipe where
  water : ℝ
  sugar : ℝ
  lemon_juice : ℝ

/-- Checks if the recipe satisfies the given ratios --/
def is_valid_recipe (r : LemonadeRecipe) : Prop :=
  r.water = 5 * r.sugar ∧ r.sugar = 3 * r.lemon_juice

/-- The main theorem: given the ratios and lemon juice amount, prove the water amount --/
theorem lemonade_water_amount (r : LemonadeRecipe) 
  (h1 : is_valid_recipe r) (h2 : r.lemon_juice = 5) : r.water = 75 := by
  sorry

/-- Proof that our solution is correct --/
theorem solution_is_correct : ∃ r : LemonadeRecipe, 
  is_valid_recipe r ∧ r.lemon_juice = 5 ∧ r.water = 75 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_water_amount_solution_is_correct_l3853_385355


namespace NUMINAMATH_CALUDE_limit_expression_equals_six_l3853_385312

theorem limit_expression_equals_six :
  ∀ ε > 0, ∃ δ > 0, ∀ h : ℝ, 0 < |h| ∧ |h| < δ →
    |((3 + h)^2 - 3^2) / h - 6| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_expression_equals_six_l3853_385312


namespace NUMINAMATH_CALUDE_cucumber_equivalent_to_16_apples_l3853_385395

/-- The cost of fruits in an arbitrary unit -/
structure FruitCost where
  apple : ℕ → ℚ
  banana : ℕ → ℚ
  cucumber : ℕ → ℚ

/-- The given conditions about fruit costs -/
def fruit_cost_conditions (c : FruitCost) : Prop :=
  c.apple 8 = c.banana 4 ∧ c.banana 2 = c.cucumber 3

/-- The theorem to prove -/
theorem cucumber_equivalent_to_16_apples (c : FruitCost) 
  (h : fruit_cost_conditions c) : 
  ∃ n : ℕ, c.apple 16 = c.cucumber n ∧ n = 12 := by
  sorry

end NUMINAMATH_CALUDE_cucumber_equivalent_to_16_apples_l3853_385395


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l3853_385386

theorem fraction_sum_equality : (2 / 10 : ℚ) + (7 / 100 : ℚ) + (3 / 1000 : ℚ) + (8 / 10000 : ℚ) = 2738 / 10000 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l3853_385386


namespace NUMINAMATH_CALUDE_A_intersect_B_eq_open_interval_l3853_385345

-- Define set A
def A : Set ℝ := {x | (x + 2) / (x - 3) < 0}

-- Define set B
def B : Set ℝ := {x | x > 0}

-- Theorem statement
theorem A_intersect_B_eq_open_interval : A ∩ B = Set.Ioo 0 3 := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_eq_open_interval_l3853_385345


namespace NUMINAMATH_CALUDE_det_A_eq_neg94_l3853_385341

def A : Matrix (Fin 3) (Fin 3) ℤ := !![2, 4, -2; 3, -1, 5; -1, 3, 2]

theorem det_A_eq_neg94 : Matrix.det A = -94 := by
  sorry

end NUMINAMATH_CALUDE_det_A_eq_neg94_l3853_385341


namespace NUMINAMATH_CALUDE_unique_solution_system_l3853_385393

/-- The system of equations:
    1. 2(x-1) - 3(y+1) = 12
    2. x/2 + y/3 = 1
    has a unique solution (x, y) = (4, -3) -/
theorem unique_solution_system :
  ∃! (x y : ℝ), (2*(x-1) - 3*(y+1) = 12) ∧ (x/2 + y/3 = 1) ∧ x = 4 ∧ y = -3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_system_l3853_385393


namespace NUMINAMATH_CALUDE_article_cost_price_l3853_385317

/-- Given an article with a marked price and a cost price, prove that the cost price is 50 
    when the selling price after a 5% deduction is 70 and represents a 40% profit. -/
theorem article_cost_price (M C : ℝ) : 
  0.95 * M = 1.40 * C → -- Condition 1 and 2 combined
  0.95 * M = 70 →       -- Condition 3
  C = 50 := by           
  sorry                 -- Proof omitted

end NUMINAMATH_CALUDE_article_cost_price_l3853_385317


namespace NUMINAMATH_CALUDE_distance_from_A_to_x_axis_l3853_385383

/-- The distance from a point to the x-axis in a Cartesian coordinate system -/
def distance_to_x_axis (y : ℝ) : ℝ := |y|

/-- Point A in the Cartesian coordinate system -/
def point_A : ℝ × ℝ := (-5, -9)

theorem distance_from_A_to_x_axis :
  distance_to_x_axis (point_A.2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_A_to_x_axis_l3853_385383


namespace NUMINAMATH_CALUDE_expression_not_prime_l3853_385397

def expression (x y : ℕ) : ℕ :=
  x^8 - x^7*y + x^6*y^2 - x^5*y^3 + x^4*y^4 - x^3*y^5 + x^2*y^6 - x*y^7 + y^8

theorem expression_not_prime (x y : ℕ) :
  ¬(Nat.Prime (expression x y)) :=
sorry

end NUMINAMATH_CALUDE_expression_not_prime_l3853_385397


namespace NUMINAMATH_CALUDE_cos_angle_between_vectors_l3853_385382

/-- Given two vectors in R², prove that the cosine of the angle between them is 4/5 -/
theorem cos_angle_between_vectors (a b : ℝ × ℝ) : 
  a = (1, 2) → b = (4, 2) → 
  let θ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  Real.cos θ = 4/5 := by
sorry

end NUMINAMATH_CALUDE_cos_angle_between_vectors_l3853_385382


namespace NUMINAMATH_CALUDE_min_m_value_x_range_l3853_385375

-- Define the conditions
def conditions (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a + b = 1

-- Part 1: Minimum value of m
theorem min_m_value (a b : ℝ) (h : conditions a b) :
  ∀ m : ℝ, (∀ a b : ℝ, conditions a b → a * b ≤ m) → m ≥ 1/4 :=
sorry

-- Part 2: Range of x
theorem x_range (a b : ℝ) (h : conditions a b) :
  ∀ x : ℝ, (4/a + 1/b ≥ |2*x - 1| - |x + 2|) ↔ -6 ≤ x ∧ x ≤ 12 :=
sorry

end NUMINAMATH_CALUDE_min_m_value_x_range_l3853_385375


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l3853_385311

/-- Two planes are mutually perpendicular -/
def mutually_perpendicular (α β : Plane) : Prop := sorry

/-- A line is parallel to a plane -/
def line_parallel_to_plane (m : Line) (α : Plane) : Prop := sorry

/-- A line is perpendicular to a plane -/
def line_perpendicular_to_plane (n : Line) (β : Plane) : Prop := sorry

/-- Two planes intersect at a line -/
def planes_intersect_at_line (α β : Plane) (l : Line) : Prop := sorry

/-- A line is perpendicular to another line -/
def line_perpendicular_to_line (n l : Line) : Prop := sorry

theorem perpendicular_lines_from_perpendicular_planes
  (α β : Plane) (l m n : Line)
  (h1 : mutually_perpendicular α β)
  (h2 : planes_intersect_at_line α β l)
  (h3 : line_parallel_to_plane m α)
  (h4 : line_perpendicular_to_plane n β) :
  line_perpendicular_to_line n l :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l3853_385311


namespace NUMINAMATH_CALUDE_fred_earnings_l3853_385347

/-- Fred's initial amount of money in dollars -/
def initial_amount : ℕ := 23

/-- Fred's final amount of money in dollars after washing cars -/
def final_amount : ℕ := 86

/-- The amount Fred made washing cars -/
def earnings : ℕ := final_amount - initial_amount

theorem fred_earnings : earnings = 63 := by
  sorry

end NUMINAMATH_CALUDE_fred_earnings_l3853_385347


namespace NUMINAMATH_CALUDE_rachel_budget_l3853_385330

/-- Given Sara's expenses and Rachel's spending intention, calculate Rachel's budget. -/
theorem rachel_budget (sara_shoes : ℕ) (sara_dress : ℕ) (rachel_multiplier : ℕ) : 
  sara_shoes = 50 → sara_dress = 200 → rachel_multiplier = 2 →
  (rachel_multiplier * sara_shoes + rachel_multiplier * sara_dress) = 500 := by
  sorry

#check rachel_budget

end NUMINAMATH_CALUDE_rachel_budget_l3853_385330


namespace NUMINAMATH_CALUDE_maintenance_check_time_l3853_385338

theorem maintenance_check_time (initial_time : ℝ) : 
  (initial_time * 1.2 = 30) → initial_time = 25 := by
  sorry

end NUMINAMATH_CALUDE_maintenance_check_time_l3853_385338


namespace NUMINAMATH_CALUDE_remainder_invariance_l3853_385319

theorem remainder_invariance (n : ℤ) (h : n % 7 = 2) : (n + 5040) % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_invariance_l3853_385319


namespace NUMINAMATH_CALUDE_carlotta_time_theorem_l3853_385396

theorem carlotta_time_theorem (n : ℝ) :
  let s : ℝ := 6
  let p : ℝ := 2 * n * s
  let t : ℝ := 3 * n * s + s
  let C : ℝ := p + t + s
  C = 30 * n + 12 := by sorry

end NUMINAMATH_CALUDE_carlotta_time_theorem_l3853_385396


namespace NUMINAMATH_CALUDE_larger_number_proof_l3853_385316

theorem larger_number_proof (x y : ℝ) (h1 : x - y = 1670) (h2 : 0.075 * x = 0.125 * y) (h3 : x > 0) (h4 : y > 0) : x = 4175 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l3853_385316


namespace NUMINAMATH_CALUDE_opposite_to_light_green_is_red_l3853_385384

-- Define the colors
inductive Color
| Red
| White
| Green
| Brown
| LightGreen
| Purple

-- Define a cube
structure Cube where
  faces : Fin 6 → Color
  different_colors : ∀ (i j : Fin 6), i ≠ j → faces i ≠ faces j

-- Define the concept of opposite faces
def opposite (c : Cube) (color1 color2 : Color) : Prop :=
  ∃ (i j : Fin 6), i ≠ j ∧ c.faces i = color1 ∧ c.faces j = color2 ∧
  ∀ (k : Fin 6), k ≠ i ∧ k ≠ j → (c.faces k = Color.White ∨ c.faces k = Color.Brown ∨ 
                                  c.faces k = Color.Purple ∨ c.faces k = Color.Green)

-- Theorem statement
theorem opposite_to_light_green_is_red (c : Cube) :
  opposite c Color.LightGreen Color.Red :=
sorry

end NUMINAMATH_CALUDE_opposite_to_light_green_is_red_l3853_385384


namespace NUMINAMATH_CALUDE_sin_five_pi_thirds_l3853_385318

theorem sin_five_pi_thirds : Real.sin (5 * π / 3) = - (Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_sin_five_pi_thirds_l3853_385318


namespace NUMINAMATH_CALUDE_doritos_ratio_l3853_385369

theorem doritos_ratio (total_bags : ℕ) (doritos_piles : ℕ) (bags_per_pile : ℕ) 
  (h1 : total_bags = 80)
  (h2 : doritos_piles = 4)
  (h3 : bags_per_pile = 5) : 
  (doritos_piles * bags_per_pile : ℚ) / total_bags = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_doritos_ratio_l3853_385369


namespace NUMINAMATH_CALUDE_data_set_median_and_variance_l3853_385336

def data_set : List ℝ := [5, 9, 8, 8, 10]

def median (l : List ℝ) : ℝ := sorry

def mean (l : List ℝ) : ℝ := sorry

def variance (l : List ℝ) : ℝ := sorry

theorem data_set_median_and_variance :
  median data_set = 8 ∧ variance data_set = 2.8 := by sorry

end NUMINAMATH_CALUDE_data_set_median_and_variance_l3853_385336


namespace NUMINAMATH_CALUDE_circle_circumference_l3853_385337

theorem circle_circumference (r : ℝ) (h : r > 0) : 
  (2 * r^2 = π * r^2) → (2 * π * r = 4 * r) :=
by sorry

end NUMINAMATH_CALUDE_circle_circumference_l3853_385337


namespace NUMINAMATH_CALUDE_felix_tree_chopping_l3853_385323

/-- Given that Felix needs to resharpen his axe every 13 trees, each sharpening costs $5,
    and he has spent $35 on sharpening, prove that he has chopped down at least 91 trees. -/
theorem felix_tree_chopping (trees_per_sharpening : ℕ) (cost_per_sharpening : ℕ) (total_spent : ℕ) 
    (h1 : trees_per_sharpening = 13)
    (h2 : cost_per_sharpening = 5)
    (h3 : total_spent = 35) :
  trees_per_sharpening * (total_spent / cost_per_sharpening) ≥ 91 := by
  sorry

#check felix_tree_chopping

end NUMINAMATH_CALUDE_felix_tree_chopping_l3853_385323


namespace NUMINAMATH_CALUDE_alternating_squares_sum_l3853_385322

theorem alternating_squares_sum : 
  23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 288 := by
  sorry

end NUMINAMATH_CALUDE_alternating_squares_sum_l3853_385322


namespace NUMINAMATH_CALUDE_complex_number_properties_l3853_385362

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Theorem to prove the four statements
theorem complex_number_properties :
  (i^2017 = i) ∧
  ((i + 1) * i = -1 + i) ∧
  ((1 - i) / (1 + i) = -i) ∧
  (Complex.abs (2 + i) = Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_properties_l3853_385362


namespace NUMINAMATH_CALUDE_f_range_l3853_385304

noncomputable def f (x : ℝ) : ℝ := x^2 / (Real.log x + x)

noncomputable def g (x : ℝ) : ℝ := Real.log x + x

theorem f_range :
  ∃ (a : ℝ), 0 < a ∧ a < 1 ∧ g a = 0 →
  Set.range f = {y | y < 0 ∨ y ≥ 1} :=
sorry

end NUMINAMATH_CALUDE_f_range_l3853_385304


namespace NUMINAMATH_CALUDE_ages_sum_l3853_385348

theorem ages_sum (a b c : ℕ) : 
  a = 20 + b + c → 
  a^2 = 2120 + (b + c)^2 → 
  a + b + c = 82 := by
sorry

end NUMINAMATH_CALUDE_ages_sum_l3853_385348


namespace NUMINAMATH_CALUDE_same_color_probability_l3853_385388

theorem same_color_probability (blue_balls yellow_balls : ℕ) 
  (h_blue : blue_balls = 8) (h_yellow : yellow_balls = 5) : 
  let total_balls := blue_balls + yellow_balls
  let prob_blue := blue_balls / total_balls
  let prob_yellow := yellow_balls / total_balls
  prob_blue ^ 2 + prob_yellow ^ 2 = 89 / 169 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l3853_385388
