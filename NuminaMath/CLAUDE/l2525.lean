import Mathlib

namespace NUMINAMATH_CALUDE_parallelogram_area_theorem_l2525_252532

/-- Represents a point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram defined by four vertices -/
structure Parallelogram where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- Calculates the area of a parallelogram given its vertices -/
def parallelogramArea (p : Parallelogram) : ℝ :=
  sorry

/-- Theorem stating that the area of a parallelogram with specific vertex coordinates is 4ap -/
theorem parallelogram_area_theorem (x a b p : ℝ) :
  let para := Parallelogram.mk
    (Point.mk x p)
    (Point.mk a b)
    (Point.mk x (-p))
    (Point.mk (-a) (-b))
  parallelogramArea para = 4 * a * p := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_theorem_l2525_252532


namespace NUMINAMATH_CALUDE_triangle_translation_l2525_252585

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a translation in 2D space -/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- Applies a translation to a point -/
def applyTranslation (p : Point) (t : Translation) : Point :=
  { x := p.x + t.dx, y := p.y + t.dy }

theorem triangle_translation (a b : ℝ) :
  let A : Point := { x := -1, y := 2 }
  let B : Point := { x := 1, y := -1 }
  let C : Point := { x := 2, y := 1 }
  let A' : Point := { x := -3, y := a }
  let B' : Point := { x := b, y := 3 }
  let t : Translation := { dx := A'.x - A.x, dy := B'.y - B.y }
  applyTranslation C t = { x := 0, y := 5 } := by
  sorry

end NUMINAMATH_CALUDE_triangle_translation_l2525_252585


namespace NUMINAMATH_CALUDE_ellipse_m_range_l2525_252511

/-- Represents an ellipse with the given equation and foci on the y-axis -/
structure Ellipse where
  m : ℝ
  eq : ∀ (x y : ℝ), x^2 / (25 - m) + y^2 / (m + 9) = 1
  foci_on_y_axis : True  -- This is a placeholder for the foci condition

/-- The range of valid m values for the given ellipse -/
theorem ellipse_m_range (e : Ellipse) : 8 < e.m ∧ e.m < 25 := by
  sorry

#check ellipse_m_range

end NUMINAMATH_CALUDE_ellipse_m_range_l2525_252511


namespace NUMINAMATH_CALUDE_probability_five_odd_in_six_rolls_l2525_252598

theorem probability_five_odd_in_six_rolls : 
  let n : ℕ := 6  -- number of rolls
  let k : ℕ := 5  -- number of desired odd rolls
  let p : ℚ := 1/2  -- probability of rolling an odd number on a single roll
  Nat.choose n k * p^k * (1-p)^(n-k) = 3/32 := by
sorry

end NUMINAMATH_CALUDE_probability_five_odd_in_six_rolls_l2525_252598


namespace NUMINAMATH_CALUDE_log_sum_equality_fraction_sum_equality_l2525_252574

-- Part 1
theorem log_sum_equality : 2 * (Real.log 10 / Real.log 5) + (Real.log 0.25 / Real.log 5) + 2^(Real.log 3 / Real.log 2) = 5 := by sorry

-- Part 2
theorem fraction_sum_equality : (5 + 1/16)^(1/2) + (-1)^(-1) / 0.75^(-2) + (2 + 10/27)^(-2/3) = 9/4 := by sorry

end NUMINAMATH_CALUDE_log_sum_equality_fraction_sum_equality_l2525_252574


namespace NUMINAMATH_CALUDE_diagonal_crosses_24_tiles_l2525_252593

/-- The number of tiles crossed by a diagonal line on a rectangular grid --/
def tiles_crossed (width length : ℕ) : ℕ :=
  width + length - Nat.gcd width length

/-- Proof that a diagonal on a 12x15 rectangle crosses 24 tiles --/
theorem diagonal_crosses_24_tiles :
  tiles_crossed 12 15 = 24 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_crosses_24_tiles_l2525_252593


namespace NUMINAMATH_CALUDE_completing_square_quadratic_l2525_252554

theorem completing_square_quadratic : 
  ∀ x : ℝ, 2 * x^2 - 3 * x - 1 = 0 ↔ (x - 3/4)^2 = 17/16 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_quadratic_l2525_252554


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2525_252578

/-- A geometric sequence is defined by its first term and common ratio. -/
def GeometricSequence (a : ℚ) (r : ℚ) : ℕ → ℚ := fun n => a * r^(n - 1)

/-- The common ratio of a geometric sequence. -/
def CommonRatio (seq : ℕ → ℚ) : ℚ := seq 2 / seq 1

theorem geometric_sequence_common_ratio :
  let seq := GeometricSequence 16 (-3/2)
  (seq 1 = 16) ∧ (seq 2 = -24) ∧ (seq 3 = 36) ∧ (seq 4 = -54) →
  CommonRatio seq = -3/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2525_252578


namespace NUMINAMATH_CALUDE_pages_left_to_read_l2525_252523

/-- Given a book with a total number of pages and a number of pages already read,
    calculate the number of pages left to read. -/
theorem pages_left_to_read (total_pages pages_read : ℕ) 
    (h1 : total_pages = 563)
    (h2 : pages_read = 147) :
    total_pages - pages_read = 416 := by
  sorry

end NUMINAMATH_CALUDE_pages_left_to_read_l2525_252523


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2525_252538

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 4 * x + 2

-- State the theorem
theorem quadratic_inequality_solution (a : ℝ) :
  (∀ x : ℝ, -1/3 < x ∧ x < 1 ↔ f a x > 0) → a = -6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2525_252538


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_three_l2525_252510

theorem reciprocal_of_negative_three :
  ∃ x : ℚ, x * (-3) = 1 ∧ x = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_three_l2525_252510


namespace NUMINAMATH_CALUDE_total_new_games_is_92_l2525_252570

/-- The number of new games Katie has -/
def katie_new_games : ℕ := 84

/-- The number of new games Katie's friends have -/
def friends_new_games : ℕ := 8

/-- The total number of new games Katie and her friends have together -/
def total_new_games : ℕ := katie_new_games + friends_new_games

/-- Theorem stating that the total number of new games is 92 -/
theorem total_new_games_is_92 : total_new_games = 92 := by sorry

end NUMINAMATH_CALUDE_total_new_games_is_92_l2525_252570


namespace NUMINAMATH_CALUDE_special_numbers_l2525_252502

def is_special (n : ℕ+) : Prop :=
  ∃ k : ℕ+, ∀ d : ℕ+, d ∣ n → (d - k : ℤ) ∣ n

theorem special_numbers (n : ℕ+) :
  is_special n ↔ n = 3 ∨ n = 4 ∨ n = 6 ∨ Nat.Prime n.val :=
sorry

end NUMINAMATH_CALUDE_special_numbers_l2525_252502


namespace NUMINAMATH_CALUDE_equation_solution_l2525_252548

theorem equation_solution : ∃ y : ℝ, (125 : ℝ)^(3*y) = 25^(4*y - 5) ∧ y = -10 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2525_252548


namespace NUMINAMATH_CALUDE_pencil_cost_is_25_l2525_252597

/-- The cost of a pencil in cents -/
def pencil_cost : ℕ := sorry

/-- The cost of a pen in cents -/
def pen_cost : ℕ := 80

/-- The total number of items (pens and pencils) bought -/
def total_items : ℕ := 36

/-- The number of pencils bought -/
def pencils_bought : ℕ := 16

/-- The total amount spent in cents -/
def total_spent : ℕ := 2000  -- 20 dollars = 2000 cents

theorem pencil_cost_is_25 : 
  pencil_cost = 25 ∧ 
  pencil_cost * pencils_bought + pen_cost * (total_items - pencils_bought) = total_spent :=
sorry

end NUMINAMATH_CALUDE_pencil_cost_is_25_l2525_252597


namespace NUMINAMATH_CALUDE_new_vessel_capacity_l2525_252576

/-- Given two vessels with different alcohol concentrations, prove the capacity of a new vessel -/
theorem new_vessel_capacity
  (vessel1_capacity : ℝ) (vessel1_alcohol_percentage : ℝ)
  (vessel2_capacity : ℝ) (vessel2_alcohol_percentage : ℝ)
  (total_liquid : ℝ) (new_concentration : ℝ)
  (h1 : vessel1_capacity = 2)
  (h2 : vessel1_alcohol_percentage = 0.3)
  (h3 : vessel2_capacity = 6)
  (h4 : vessel2_alcohol_percentage = 0.45)
  (h5 : total_liquid = 8)
  (h6 : new_concentration = 0.33) :
  (vessel1_capacity * vessel1_alcohol_percentage + vessel2_capacity * vessel2_alcohol_percentage) / new_concentration = 10 := by
sorry

end NUMINAMATH_CALUDE_new_vessel_capacity_l2525_252576


namespace NUMINAMATH_CALUDE_min_value_fraction_l2525_252563

theorem min_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 4 * x + y = 1) :
  (x + y) / (x * y) ≥ 9 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 4 * x + y = 1 ∧ (x + y) / (x * y) = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_l2525_252563


namespace NUMINAMATH_CALUDE_correct_operation_l2525_252520

theorem correct_operation (a b : ℝ) : 3*a + 2*b - 2*(a - b) = a + 4*b := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l2525_252520


namespace NUMINAMATH_CALUDE_equation_d_is_quadratic_l2525_252552

-- Define a quadratic equation
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- Define the specific equation from Option D
def equation_d (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 2

-- Theorem stating that equation_d is a quadratic equation
theorem equation_d_is_quadratic : is_quadratic_equation equation_d :=
sorry

end NUMINAMATH_CALUDE_equation_d_is_quadratic_l2525_252552


namespace NUMINAMATH_CALUDE_x_plus_reciprocal_squared_l2525_252522

theorem x_plus_reciprocal_squared (x : ℝ) (h : x^2 + 1/x^2 = 7) : (x + 1/x)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_reciprocal_squared_l2525_252522


namespace NUMINAMATH_CALUDE_ball_box_probability_l2525_252537

/-- The number of ways to place 5 balls in 4 boxes with no box left empty -/
def total_placements : ℕ := 240

/-- The number of ways to place 5 balls in 4 boxes with no box left empty and no ball in a box with the same label -/
def valid_placements : ℕ := 84

/-- The probability of placing 5 balls in 4 boxes with no box left empty and no ball in a box with the same label -/
def probability : ℚ := valid_placements / total_placements

theorem ball_box_probability : probability = 7 / 20 := by
  sorry

end NUMINAMATH_CALUDE_ball_box_probability_l2525_252537


namespace NUMINAMATH_CALUDE_pq_length_in_30_60_90_triangle_l2525_252515

/-- Represents a 30-60-90 triangle -/
structure Triangle30_60_90 where
  /-- The length of the hypotenuse -/
  hypotenuse : ℝ
  /-- The length of the side opposite to the 30° angle -/
  short_side : ℝ
  /-- The length of the side opposite to the 60° angle -/
  long_side : ℝ
  /-- The hypotenuse is twice the short side -/
  hypotenuse_twice_short : hypotenuse = 2 * short_side
  /-- The long side is √3 times the short side -/
  long_side_sqrt3_short : long_side = Real.sqrt 3 * short_side

/-- Theorem: In a 30-60-90 triangle PQR where PR = 6√3 and angle QPR = 30°, PQ = 6√3 -/
theorem pq_length_in_30_60_90_triangle (t : Triangle30_60_90) 
  (h : t.hypotenuse = 6 * Real.sqrt 3) : t.long_side = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_pq_length_in_30_60_90_triangle_l2525_252515


namespace NUMINAMATH_CALUDE_painter_workdays_l2525_252556

theorem painter_workdays (job_size : ℝ) (rate : ℝ) (h : job_size = 6 * 1.5 * rate) :
  job_size = 4 * 2.25 * rate := by
sorry

end NUMINAMATH_CALUDE_painter_workdays_l2525_252556


namespace NUMINAMATH_CALUDE_chord_line_equation_l2525_252599

/-- Given a circle with equation x^2 + y^2 = 10 and a chord with midpoint P(1, 1),
    the equation of the line containing this chord is x + y - 2 = 0 -/
theorem chord_line_equation (x y : ℝ) :
  (x^2 + y^2 = 10) →
  (∃ (t : ℝ), x = 1 + t ∧ y = 1 - t) →
  (x + y - 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_chord_line_equation_l2525_252599


namespace NUMINAMATH_CALUDE_stirling_bounds_l2525_252590

-- Define e as the limit of (1 + 1/n)^n as n approaches infinity
noncomputable def e : ℝ := Real.exp 1

-- State the theorem
theorem stirling_bounds (n : ℕ) (h : n > 6) :
  (n / e : ℝ)^n < n! ∧ (n! : ℝ) < n * (n / e)^n :=
sorry

end NUMINAMATH_CALUDE_stirling_bounds_l2525_252590


namespace NUMINAMATH_CALUDE_molecular_weight_AlPO4_correct_l2525_252596

/-- The molecular weight of AlPO4 in grams per mole -/
def molecular_weight_AlPO4 : ℝ := 122

/-- The number of moles given in the problem -/
def moles : ℝ := 4

/-- The total weight of the given moles of AlPO4 in grams -/
def total_weight : ℝ := 488

/-- Theorem: The molecular weight of AlPO4 is correct given the total weight of 4 moles -/
theorem molecular_weight_AlPO4_correct : 
  molecular_weight_AlPO4 * moles = total_weight :=
sorry

end NUMINAMATH_CALUDE_molecular_weight_AlPO4_correct_l2525_252596


namespace NUMINAMATH_CALUDE_wire_cutting_l2525_252561

theorem wire_cutting (total_length : ℝ) (ratio : ℝ) (shorter_piece : ℝ) : 
  total_length = 21 →
  ratio = 2 / 5 →
  shorter_piece + (shorter_piece / ratio) = total_length →
  shorter_piece = 6 := by
sorry

end NUMINAMATH_CALUDE_wire_cutting_l2525_252561


namespace NUMINAMATH_CALUDE_probability_of_winning_l2525_252595

/-- The probability of player A winning a match in a game with 2n rounds -/
def P (n : ℕ) : ℚ :=
  1/2 * (1 - (Nat.choose (2*n) n : ℚ) / (2^(2*n)))

/-- Theorem stating the probability of player A winning the match -/
theorem probability_of_winning (n : ℕ) (h : n > 0) : 
  P n = 1/2 * (1 - (Nat.choose (2*n) n : ℚ) / (2^(2*n))) :=
by sorry

end NUMINAMATH_CALUDE_probability_of_winning_l2525_252595


namespace NUMINAMATH_CALUDE_intersection_sum_l2525_252586

/-- Given two lines y = 2x + c and y = 4x + d intersecting at (3, 12), prove that c + d = 6 -/
theorem intersection_sum (c d : ℝ) : 
  (2 * 3 + c = 12) → (4 * 3 + d = 12) → c + d = 6 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l2525_252586


namespace NUMINAMATH_CALUDE_glove_sequences_l2525_252565

/-- Represents the number of hands. -/
def num_hands : ℕ := 2

/-- Represents the number of layers of gloves. -/
def num_layers : ℕ := 2

/-- Represents whether the inner gloves are identical. -/
def inner_gloves_identical : Prop := True

/-- Represents whether the outer gloves are distinct for left and right hands. -/
def outer_gloves_distinct : Prop := True

/-- The number of different sequences for wearing the gloves. -/
def num_sequences : ℕ := 6

/-- Theorem stating that the number of different sequences for wearing the gloves is 6. -/
theorem glove_sequences :
  num_hands = 2 ∧ 
  num_layers = 2 ∧ 
  inner_gloves_identical ∧ 
  outer_gloves_distinct →
  num_sequences = 6 :=
by sorry


end NUMINAMATH_CALUDE_glove_sequences_l2525_252565


namespace NUMINAMATH_CALUDE_system2_solution_l2525_252529

variable (a b : ℝ)

-- Define the first system of equations and its solution
def system1_eq1 (x y : ℝ) : Prop := a * x - b * y = 3
def system1_eq2 (x y : ℝ) : Prop := a * x + b * y = 5
def system1_solution (x y : ℝ) : Prop := x = 2 ∧ y = 1

-- Define the second system of equations
def system2_eq1 (m n : ℝ) : Prop := a * (m + 2 * n) - 2 * b * n = 6
def system2_eq2 (m n : ℝ) : Prop := a * (m + 2 * n) + 2 * b * n = 10

-- State the theorem
theorem system2_solution :
  (∃ x y, system1_eq1 a b x y ∧ system1_eq2 a b x y ∧ system1_solution x y) →
  (∃ m n, system2_eq1 a b m n ∧ system2_eq2 a b m n ∧ m = 2 ∧ n = 1) :=
by sorry

end NUMINAMATH_CALUDE_system2_solution_l2525_252529


namespace NUMINAMATH_CALUDE_pillowcase_material_proof_l2525_252547

/-- The amount of material needed for one pillowcase -/
def pillowcase_material : ℝ := 1.25

theorem pillowcase_material_proof :
  let total_material : ℝ := 5000
  let third_bale_ratio : ℝ := 0.22
  let sheet_pillowcase_diff : ℝ := 3.25
  let sheets_sewn : ℕ := 150
  let pillowcases_sewn : ℕ := 240
  ∃ (first_bale second_bale third_bale : ℝ),
    first_bale + second_bale + third_bale = total_material ∧
    3 * first_bale = second_bale ∧
    third_bale = third_bale_ratio * total_material ∧
    sheets_sewn * (pillowcase_material + sheet_pillowcase_diff) + pillowcases_sewn * pillowcase_material = first_bale :=
by sorry

end NUMINAMATH_CALUDE_pillowcase_material_proof_l2525_252547


namespace NUMINAMATH_CALUDE_triangle_side_inequality_l2525_252555

theorem triangle_side_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  a / (b + c) + b / (c + a) + c / (a + b) < 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_inequality_l2525_252555


namespace NUMINAMATH_CALUDE_imaginary_part_of_2_plus_i_l2525_252566

theorem imaginary_part_of_2_plus_i : Complex.im (2 + Complex.I) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_2_plus_i_l2525_252566


namespace NUMINAMATH_CALUDE_cos_two_alpha_equals_zero_l2525_252513

theorem cos_two_alpha_equals_zero (α : ℝ) 
  (h : Real.sin (π / 6 - α) = Real.cos (π / 6 + α)) : 
  Real.cos (2 * α) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_two_alpha_equals_zero_l2525_252513


namespace NUMINAMATH_CALUDE_geometric_area_ratios_l2525_252544

theorem geometric_area_ratios (s : ℝ) (h : s > 0) :
  let square_area := s^2
  let triangle_area := s^2 / 2
  let circle_area := π * (s/2)^2
  let small_square_area := (s/2)^2
  (triangle_area / square_area = 1/2) ∧
  (circle_area / square_area = π/4) ∧
  (small_square_area / square_area = 1/4) :=
by sorry

end NUMINAMATH_CALUDE_geometric_area_ratios_l2525_252544


namespace NUMINAMATH_CALUDE_prime_absolute_value_quadratic_l2525_252531

theorem prime_absolute_value_quadratic (a : ℤ) : 
  Nat.Prime (Int.natAbs (a^2 - 3*a - 6)) ↔ a = -1 ∨ a = 4 := by
sorry

end NUMINAMATH_CALUDE_prime_absolute_value_quadratic_l2525_252531


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2525_252551

/-- An isosceles triangle with two sides measuring 5 and 6 -/
structure IsoscelesTriangle :=
  (side1 : ℝ)
  (side2 : ℝ)
  (is_isosceles : side1 = side2 ∨ side1 = 6 ∨ side2 = 6)
  (has_sides_5_6 : (side1 = 5 ∧ side2 = 6) ∨ (side1 = 6 ∧ side2 = 5))

/-- The perimeter of an isosceles triangle with sides 5 and 6 is either 16 or 17 -/
theorem isosceles_triangle_perimeter (t : IsoscelesTriangle) :
  ∃ (p : ℝ), (p = 16 ∨ p = 17) ∧ p = t.side1 + t.side2 + (if t.side1 = t.side2 then 5 else 6) :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2525_252551


namespace NUMINAMATH_CALUDE_range_of_m_l2525_252507

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x^2 + 2*x + m > 0) ↔ m > 1 := by sorry

end NUMINAMATH_CALUDE_range_of_m_l2525_252507


namespace NUMINAMATH_CALUDE_max_m_for_monotonic_f_l2525_252500

/-- Given a function f(x) = x^4 - (1/3)mx^3 + (1/2)x^2 + 1, 
    if f is monotonically increasing on (0,1), 
    then the maximum value of m is 4 -/
theorem max_m_for_monotonic_f (m : ℝ) : 
  let f := fun (x : ℝ) ↦ x^4 - (1/3)*m*x^3 + (1/2)*x^2 + 1
  (∀ x ∈ Set.Ioo 0 1, Monotone f) → m ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_max_m_for_monotonic_f_l2525_252500


namespace NUMINAMATH_CALUDE_problem_solution_l2525_252591

theorem problem_solution (x : ℝ) 
  (h : x * Real.sqrt (x^2 - 1) + 1 / (x + Real.sqrt (x^2 - 1)) = 21) :
  x^2 * Real.sqrt (x^4 - 1) + 1 / (x^2 + Real.sqrt (x^4 - 1)) = 2 * x^2 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2525_252591


namespace NUMINAMATH_CALUDE_sandbox_area_l2525_252572

-- Define the sandbox dimensions
def sandbox_length : ℝ := 312
def sandbox_width : ℝ := 146

-- State the theorem
theorem sandbox_area : sandbox_length * sandbox_width = 45552 := by
  sorry

end NUMINAMATH_CALUDE_sandbox_area_l2525_252572


namespace NUMINAMATH_CALUDE_greatest_k_inequality_l2525_252583

theorem greatest_k_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a^2 > b*c) :
  (a^2 - b*c)^2 ≥ 4*(b^2 - c*a)*(c^2 - a*b) := by
  sorry

end NUMINAMATH_CALUDE_greatest_k_inequality_l2525_252583


namespace NUMINAMATH_CALUDE_Only_Statement3_Is_Correct_l2525_252508

-- Define the basic properties of functions
def Monotonic_Increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

def Odd_Function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def Even_Function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def Symmetric_About_Y_Axis (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Define the four statements
def Statement1 : Prop :=
  Monotonic_Increasing (fun x => -1/x)

def Statement2 : Prop :=
  ∀ f : ℝ → ℝ, Odd_Function f → f 0 = 0

def Statement3 : Prop :=
  ∀ f : ℝ → ℝ, Even_Function f → Symmetric_About_Y_Axis f

def Statement4 : Prop :=
  ∀ f : ℝ → ℝ, Odd_Function f → Even_Function f → ∀ x, f x = 0

-- Theorem stating that only Statement3 is correct
theorem Only_Statement3_Is_Correct :
  ¬Statement1 ∧ ¬Statement2 ∧ Statement3 ∧ ¬Statement4 :=
sorry

end NUMINAMATH_CALUDE_Only_Statement3_Is_Correct_l2525_252508


namespace NUMINAMATH_CALUDE_students_facing_teacher_l2525_252587

theorem students_facing_teacher (n : ℕ) (h : n = 50) : 
  n - (n / 3 + n / 7 - n / 21) = 31 :=
sorry

end NUMINAMATH_CALUDE_students_facing_teacher_l2525_252587


namespace NUMINAMATH_CALUDE_negation_equivalence_l2525_252592

theorem negation_equivalence :
  (¬ ∃ x₀ > 2, x₀^3 - 2*x₀^2 < 0) ↔ (∀ x > 2, x^3 - 2*x^2 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2525_252592


namespace NUMINAMATH_CALUDE_order_of_expressions_l2525_252588

theorem order_of_expressions :
  let x : ℝ := Real.exp (-1/2)
  let y : ℝ := (Real.log 2) / (Real.log 5)
  let z : ℝ := Real.log 3
  y < x ∧ x < z := by sorry

end NUMINAMATH_CALUDE_order_of_expressions_l2525_252588


namespace NUMINAMATH_CALUDE_stock_value_change_l2525_252535

theorem stock_value_change (initial_value : ℝ) (h : initial_value > 0) : 
  let day1_value := initial_value * (1 - 0.25)
  let day2_value := day1_value * (1 + 0.35)
  (day2_value - initial_value) / initial_value * 100 = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_stock_value_change_l2525_252535


namespace NUMINAMATH_CALUDE_matrix_row_replacement_determinant_l2525_252504

theorem matrix_row_replacement_determinant :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![2, 5; 3, 7]
  let B : Matrix (Fin 2) (Fin 2) ℤ := 
    Matrix.updateRow A 1 (fun j => 2 * A 0 j + A 1 j)
  Matrix.det B = -1 := by
  sorry

end NUMINAMATH_CALUDE_matrix_row_replacement_determinant_l2525_252504


namespace NUMINAMATH_CALUDE_cube_volume_from_space_diagonal_l2525_252582

theorem cube_volume_from_space_diagonal (d : ℝ) (h : d = 6 * Real.sqrt 3) :
  let s := d / Real.sqrt 3
  s ^ 3 = 216 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_space_diagonal_l2525_252582


namespace NUMINAMATH_CALUDE_walking_distance_l2525_252549

/-- 
Given a person who walks at 10 km/hr, if increasing their speed to 16 km/hr 
would allow them to walk 36 km more in the same time, then the actual distance 
traveled is 60 km.
-/
theorem walking_distance (actual_speed : ℝ) (faster_speed : ℝ) (extra_distance : ℝ) 
  (h1 : actual_speed = 10)
  (h2 : faster_speed = 16)
  (h3 : extra_distance = 36)
  (h4 : (actual_distance / actual_speed) = ((actual_distance + extra_distance) / faster_speed)) :
  actual_distance = 60 :=
by
  sorry

#check walking_distance

end NUMINAMATH_CALUDE_walking_distance_l2525_252549


namespace NUMINAMATH_CALUDE_power_sum_inequality_l2525_252575

theorem power_sum_inequality (a b c : ℝ) (m : ℕ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hm : 0 < m) :
  (a + b)^m + (b + c)^m + (c + a)^m ≤ 2^m * (a^m + b^m + c^m) := by
  sorry

end NUMINAMATH_CALUDE_power_sum_inequality_l2525_252575


namespace NUMINAMATH_CALUDE_movie_production_l2525_252562

theorem movie_production (x : ℝ) : 
  (∃ y : ℝ, y = 1.25 * x ∧ 5 * (x + y) = 2475) → x = 220 :=
by sorry

end NUMINAMATH_CALUDE_movie_production_l2525_252562


namespace NUMINAMATH_CALUDE_boys_playing_neither_l2525_252581

/-- Given a group of boys with information about their sports participation,
    calculate the number of boys who play neither basketball nor football. -/
theorem boys_playing_neither (total : ℕ) (basketball : ℕ) (football : ℕ) (both : ℕ)
    (h_total : total = 22)
    (h_basketball : basketball = 13)
    (h_football : football = 15)
    (h_both : both = 18) :
    total - (basketball + football - both) = 12 := by
  sorry

end NUMINAMATH_CALUDE_boys_playing_neither_l2525_252581


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2525_252559

theorem pure_imaginary_complex_number (a : ℝ) : 
  (∃ b : ℝ, (a + 2 * Complex.I) / (2 - Complex.I) = b * Complex.I) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2525_252559


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l2525_252557

/-- The lateral surface area of a cone with base radius 3 and slant height 5 is 15π. -/
theorem cone_lateral_surface_area : 
  ∀ (r l : ℝ), r = 3 → l = 5 → π * r * l = 15 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l2525_252557


namespace NUMINAMATH_CALUDE_simplify_expression_l2525_252546

theorem simplify_expression (n : ℕ) : 
  (3^(n+4) - 3*(3^n) - 3^(n+2)) / (3*(3^(n+3))) = 23/9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2525_252546


namespace NUMINAMATH_CALUDE_correct_sum_and_digit_change_l2525_252512

theorem correct_sum_and_digit_change : ∃ (d e : ℕ), 
  (d ≤ 9 ∧ e ≤ 9) ∧ 
  (553672 + 637528 = 1511200) ∧ 
  (d + e = 14) ∧
  (953672 + 637528 ≠ 1511200) := by
sorry

end NUMINAMATH_CALUDE_correct_sum_and_digit_change_l2525_252512


namespace NUMINAMATH_CALUDE_tetrahedron_volume_specific_l2525_252509

def tetrahedron_volume (AB AC AD BC BD CD : ℝ) : ℝ := sorry

theorem tetrahedron_volume_specific : 
  tetrahedron_volume 2 4 3 (Real.sqrt 17) (Real.sqrt 13) 5 = 6 * Real.sqrt 247 / 64 := by sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_specific_l2525_252509


namespace NUMINAMATH_CALUDE_wilsons_theorem_l2525_252571

theorem wilsons_theorem (n : ℕ) (h : n ≥ 2) :
  Nat.Prime n ↔ (Nat.factorial (n - 1) % n = n - 1) := by
  sorry

end NUMINAMATH_CALUDE_wilsons_theorem_l2525_252571


namespace NUMINAMATH_CALUDE_max_value_expression_max_value_attained_l2525_252514

theorem max_value_expression (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) 
  (h_sum : a^2 + b^2 + c^2 = 1) : 
  3 * a * b * Real.sqrt 3 + 9 * b * c ≤ 3 :=
by sorry

theorem max_value_attained (ε : ℝ) (hε : ε > 0) : 
  ∃ a b c : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a^2 + b^2 + c^2 = 1 ∧ 
  3 * a * b * Real.sqrt 3 + 9 * b * c > 3 - ε :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_max_value_attained_l2525_252514


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_range_l2525_252501

theorem quadratic_equation_roots_range (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x > 0 ∧ y > 0 ∧ 2 * x^2 - (m + 1) * x + m = 0 ∧ 2 * y^2 - (m + 1) * y + m = 0) 
  ↔ 
  (0 < m ∧ m < 3 - 2 * Real.sqrt 2) ∨ (m > 3 + 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_range_l2525_252501


namespace NUMINAMATH_CALUDE_password_length_l2525_252517

-- Define the structure of the password
structure PasswordStructure where
  lowercase_letters : Nat
  uppercase_and_numbers : Nat
  digits : Nat
  symbols : Nat

-- Define Pat's password structure
def pats_password : PasswordStructure :=
  { lowercase_letters := 12
  , uppercase_and_numbers := 6
  , digits := 4
  , symbols := 2 }

-- Theorem to prove the total number of characters
theorem password_length :
  (pats_password.lowercase_letters +
   pats_password.uppercase_and_numbers +
   pats_password.digits +
   pats_password.symbols) = 24 := by
  sorry

end NUMINAMATH_CALUDE_password_length_l2525_252517


namespace NUMINAMATH_CALUDE_hotel_cost_proof_l2525_252530

theorem hotel_cost_proof (initial_share : ℝ) (final_share : ℝ) : 
  (∃ (total_cost : ℝ),
    (initial_share = total_cost / 4) ∧ 
    (final_share = total_cost / 7) ∧
    (initial_share - 15 = final_share)) →
  ∃ (total_cost : ℝ), total_cost = 140 := by
sorry

end NUMINAMATH_CALUDE_hotel_cost_proof_l2525_252530


namespace NUMINAMATH_CALUDE_greatest_integer_satisfying_inequality_l2525_252594

theorem greatest_integer_satisfying_inequality :
  ∃ (x : ℕ), x > 0 ∧ (x^4 : ℚ) / (x^2 : ℚ) < 12 ∧
  ∀ (y : ℕ), y > x → (y^4 : ℚ) / (y^2 : ℚ) ≥ 12 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_greatest_integer_satisfying_inequality_l2525_252594


namespace NUMINAMATH_CALUDE_apartments_per_floor_l2525_252569

theorem apartments_per_floor (num_buildings : ℕ) (floors_per_building : ℕ) 
  (doors_per_apartment : ℕ) (total_doors : ℕ) :
  num_buildings = 2 →
  floors_per_building = 12 →
  doors_per_apartment = 7 →
  total_doors = 1008 →
  (total_doors / doors_per_apartment) / (num_buildings * floors_per_building) = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_apartments_per_floor_l2525_252569


namespace NUMINAMATH_CALUDE_camel_cost_calculation_l2525_252579

/-- The cost of a camel in rupees -/
def camel_cost : ℝ := 4184.62

/-- The cost of a horse in rupees -/
def horse_cost : ℝ := 1743.59

/-- The cost of an ox in rupees -/
def ox_cost : ℝ := 11333.33

/-- The cost of an elephant in rupees -/
def elephant_cost : ℝ := 17000

theorem camel_cost_calculation :
  (10 * camel_cost = 24 * horse_cost) ∧
  (26 * horse_cost = 4 * ox_cost) ∧
  (6 * ox_cost = 4 * elephant_cost) ∧
  (10 * elephant_cost = 170000) →
  camel_cost = 4184.62 := by
sorry

#eval camel_cost

end NUMINAMATH_CALUDE_camel_cost_calculation_l2525_252579


namespace NUMINAMATH_CALUDE_givenEquationIsParabola_l2525_252525

/-- Represents a conic section type -/
inductive ConicType
  | Circle
  | Parabola
  | Ellipse
  | Hyperbola
  | None

/-- Determines if an equation represents a parabola -/
def isParabola (f : ℝ → ℝ → Prop) : Prop :=
  ∃ a b c d e : ℝ, a ≠ 0 ∧ 
    ∀ x y : ℝ, f x y ↔ (a * y^2 + b * y + c * x + d = 0 ∨ a * x^2 + b * x + c * y + d = 0)

/-- The given equation -/
def givenEquation (x y : ℝ) : Prop :=
  |x - 3| = Real.sqrt ((y + 4)^2 + x^2)

/-- Theorem stating that the given equation represents a parabola -/
theorem givenEquationIsParabola : isParabola givenEquation := by
  sorry

/-- The conic type of the given equation is a parabola -/
def conicTypeOfGivenEquation : ConicType := ConicType.Parabola

end NUMINAMATH_CALUDE_givenEquationIsParabola_l2525_252525


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2525_252542

theorem sqrt_equation_solution (a b c : ℝ) 
  (h1 : Real.sqrt a = Real.sqrt b + Real.sqrt c)
  (h2 : b = 52 - 30 * Real.sqrt 3)
  (h3 : c = a - 2) : 
  a = 27 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2525_252542


namespace NUMINAMATH_CALUDE_sum_of_numbers_ge_04_l2525_252541

theorem sum_of_numbers_ge_04 : 
  let numbers := [0.8, 1/2, 0.3]
  let sum_ge_04 := (numbers.filter (λ x => x ≥ 0.4)).sum
  sum_ge_04 = 1.3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_numbers_ge_04_l2525_252541


namespace NUMINAMATH_CALUDE_last_letter_151st_permutation_l2525_252558

-- Define the word and its permutations
def word : String := "JOKING"
def num_permutations : Nat := 720

-- Define the dictionary order function (not implemented, just declared)
def dictionary_order (s1 s2 : String) : Bool :=
  sorry

-- Define a function to get the nth permutation in dictionary order
def nth_permutation (n : Nat) : String :=
  sorry

-- Define a function to get the last letter of a string
def last_letter (s : String) : Char :=
  sorry

-- The theorem to prove
theorem last_letter_151st_permutation :
  last_letter (nth_permutation 151) = 'O' :=
sorry

end NUMINAMATH_CALUDE_last_letter_151st_permutation_l2525_252558


namespace NUMINAMATH_CALUDE_folded_rectangle_BC_l2525_252584

/-- A rectangle ABCD with the following properties:
  - AB = 10
  - AD is folded onto AB, creating crease AE
  - Triangle AED is folded along DE
  - AE intersects BC at point F
  - Area of triangle ABF is 2 -/
structure FoldedRectangle where
  AB : ℝ
  BC : ℝ
  area_ABF : ℝ
  AB_eq_10 : AB = 10
  area_ABF_eq_2 : area_ABF = 2

/-- Theorem: In a FoldedRectangle, BC = 5.2 -/
theorem folded_rectangle_BC (r : FoldedRectangle) : r.BC = 5.2 := by
  sorry

end NUMINAMATH_CALUDE_folded_rectangle_BC_l2525_252584


namespace NUMINAMATH_CALUDE_sphere_ratio_l2525_252518

theorem sphere_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (4 * Real.pi * r₁^2) / (4 * Real.pi * r₂^2) = 1 / 16 →
  ((4 / 3) * Real.pi * r₁^3) / ((4 / 3) * Real.pi * r₂^3) = 1 / 64 := by
sorry

end NUMINAMATH_CALUDE_sphere_ratio_l2525_252518


namespace NUMINAMATH_CALUDE_dividend_calculation_l2525_252580

theorem dividend_calculation (quotient divisor remainder : ℕ) 
  (h_quotient : quotient = 65)
  (h_divisor : divisor = 24)
  (h_remainder : remainder = 5) :
  (divisor * quotient) + remainder = 1565 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l2525_252580


namespace NUMINAMATH_CALUDE_degree_of_composed_product_l2525_252521

-- Define polynomials f and g with their respective degrees
def f : Polynomial ℝ := sorry
def g : Polynomial ℝ := sorry

-- State the theorem
theorem degree_of_composed_product :
  (Polynomial.degree f = 4) →
  (Polynomial.degree g = 5) →
  Polynomial.degree (f.comp (Polynomial.X ^ 2) * g.comp (Polynomial.X ^ 4)) = 28 := by
  sorry

end NUMINAMATH_CALUDE_degree_of_composed_product_l2525_252521


namespace NUMINAMATH_CALUDE_badger_walnuts_l2525_252528

theorem badger_walnuts (badger_walnuts_per_hole fox_walnuts_per_hole : ℕ)
  (h_badger_walnuts : badger_walnuts_per_hole = 5)
  (h_fox_walnuts : fox_walnuts_per_hole = 7)
  (h_hole_diff : ℕ)
  (h_hole_diff_value : h_hole_diff = 2)
  (badger_holes fox_holes : ℕ)
  (h_holes_relation : badger_holes = fox_holes + h_hole_diff)
  (total_walnuts : ℕ)
  (h_total_equality : badger_walnuts_per_hole * badger_holes = fox_walnuts_per_hole * fox_holes)
  (h_total_walnuts : total_walnuts = badger_walnuts_per_hole * badger_holes) :
  total_walnuts = 35 :=
by sorry

end NUMINAMATH_CALUDE_badger_walnuts_l2525_252528


namespace NUMINAMATH_CALUDE_book_distribution_theorem_l2525_252505

/-- The number of different books to be distributed -/
def num_books : ℕ := 6

/-- The number of students -/
def num_students : ℕ := 6

/-- The number of students who receive books -/
def num_receiving_students : ℕ := num_students - 1

/-- The number of ways to distribute the books -/
def distribution_ways : ℕ := num_students * (num_receiving_students ^ num_books)

theorem book_distribution_theorem : distribution_ways = 93750 := by
  sorry

end NUMINAMATH_CALUDE_book_distribution_theorem_l2525_252505


namespace NUMINAMATH_CALUDE_sara_remaining_pears_l2525_252516

-- Define the initial number of pears Sara picked
def initial_pears : ℕ := 35

-- Define the number of pears Sara gave to Dan
def pears_given : ℕ := 28

-- Theorem to prove
theorem sara_remaining_pears :
  initial_pears - pears_given = 7 := by
  sorry

end NUMINAMATH_CALUDE_sara_remaining_pears_l2525_252516


namespace NUMINAMATH_CALUDE_fred_baseball_cards_l2525_252550

theorem fred_baseball_cards (initial_cards : ℕ) (cards_bought : ℕ) (remaining_cards : ℕ) : 
  initial_cards = 40 → cards_bought = 22 → remaining_cards = initial_cards - cards_bought → 
  remaining_cards = 18 := by
sorry

end NUMINAMATH_CALUDE_fred_baseball_cards_l2525_252550


namespace NUMINAMATH_CALUDE_triangle_side_length_l2525_252526

/-- Given a triangle ABC with specific properties, prove that the length of side b is √7 -/
theorem triangle_side_length (A B C : ℝ) (α l a b c : ℝ) : 
  0 < α → 0 < l → 0 < a → 0 < b → 0 < c →
  B = π / 3 →
  (a * c : ℝ) * Real.cos B = 3 / 2 →
  a + c = 4 →
  b ^ 2 = 7 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2525_252526


namespace NUMINAMATH_CALUDE_square_root_statements_l2525_252567

theorem square_root_statements :
  (Real.sqrt 2 * Real.sqrt 5 = Real.sqrt 10) ∧
  (Real.sqrt 2 + Real.sqrt 5 ≠ Real.sqrt 7) ∧
  (Real.sqrt 18 / Real.sqrt 2 = 3) ∧
  (Real.sqrt 12 = 2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_square_root_statements_l2525_252567


namespace NUMINAMATH_CALUDE_expected_value_5X_plus_4_l2525_252568

/-- Distribution of random variable X -/
structure Distribution where
  p0 : ℝ
  p2 : ℝ
  p4 : ℝ
  sum_to_one : p0 + p2 + p4 = 1
  non_negative : p0 ≥ 0 ∧ p2 ≥ 0 ∧ p4 ≥ 0

/-- Expected value of a random variable -/
def expected_value (d : Distribution) : ℝ := 0 * d.p0 + 2 * d.p2 + 4 * d.p4

/-- Theorem: Expected value of 5X+4 equals 16 -/
theorem expected_value_5X_plus_4 (d : Distribution) 
  (h1 : d.p0 = 0.3) 
  (h2 : d.p4 = 0.5) : 
  5 * expected_value d + 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_5X_plus_4_l2525_252568


namespace NUMINAMATH_CALUDE_paths_through_point_c_l2525_252506

/-- The number of paths on a grid from (0,0) to (x,y) moving only right or up -/
def gridPaths (x y : ℕ) : ℕ := Nat.choose (x + y) y

/-- The total number of paths from A(0,0) to B(7,6) passing through C(3,2) on a 7x6 grid -/
def totalPaths : ℕ :=
  gridPaths 3 2 * gridPaths 4 3

theorem paths_through_point_c :
  totalPaths = 200 := by sorry

end NUMINAMATH_CALUDE_paths_through_point_c_l2525_252506


namespace NUMINAMATH_CALUDE_probability_red_second_given_white_first_l2525_252560

-- Define the total number of balls and the number of each color
def total_balls : ℕ := 5
def red_balls : ℕ := 3
def white_balls : ℕ := 2

-- Define the probability of drawing a red ball second, given the first is white
def prob_red_second_given_white_first : ℚ := 3 / 4

-- Theorem statement
theorem probability_red_second_given_white_first :
  (red_balls : ℚ) / (total_balls - 1) = prob_red_second_given_white_first :=
by sorry

end NUMINAMATH_CALUDE_probability_red_second_given_white_first_l2525_252560


namespace NUMINAMATH_CALUDE_set_equals_interval_l2525_252533

-- Define the set S as {x | x > 0 and x ≠ 2}
def S : Set ℝ := {x : ℝ | x > 0 ∧ x ≠ 2}

-- Define the interval representation
def intervalRep : Set ℝ := Set.Ioo 0 2 ∪ Set.Ioi 2

-- Theorem stating the equivalence of the set and the interval representation
theorem set_equals_interval : S = intervalRep := by sorry

end NUMINAMATH_CALUDE_set_equals_interval_l2525_252533


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l2525_252519

theorem diophantine_equation_solutions :
  ∀ x y z : ℕ+,
  z = Nat.gcd x y →
  x + y^2 + z^3 = x * y * z →
  ((x = 4 ∧ y = 2) ∨ (x = 4 ∧ y = 6) ∨ (x = 5 ∧ y = 2) ∨ (x = 5 ∧ y = 3)) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l2525_252519


namespace NUMINAMATH_CALUDE_indeterminate_product_sum_l2525_252503

theorem indeterminate_product_sum (A B : ℝ) 
  (hA : 0 < A ∧ A < 1) (hB : 0 < B ∧ B < 1) : 
  ∃ (x y z : ℝ), x < 1 ∧ y = 1 ∧ z > 1 ∧ 
  (A * B + 0.1 = x ∨ A * B + 0.1 = y ∨ A * B + 0.1 = z) :=
sorry

end NUMINAMATH_CALUDE_indeterminate_product_sum_l2525_252503


namespace NUMINAMATH_CALUDE_min_snack_cost_l2525_252553

/-- Calculates the minimum number of packs/bags needed given the number of items per pack/bag and the total number of items required -/
def min_packs_needed (items_per_pack : ℕ) (total_items_needed : ℕ) : ℕ :=
  (total_items_needed + items_per_pack - 1) / items_per_pack

/-- Represents the problem of buying snacks for soccer players -/
def snack_problem (num_players : ℕ) (juice_per_pack : ℕ) (juice_pack_cost : ℚ) 
                  (apples_per_bag : ℕ) (apple_bag_cost : ℚ) : ℚ :=
  let juice_packs := min_packs_needed juice_per_pack num_players
  let apple_bags := min_packs_needed apples_per_bag num_players
  juice_packs * juice_pack_cost + apple_bags * apple_bag_cost

/-- The theorem stating the minimum amount Danny spends -/
theorem min_snack_cost : 
  snack_problem 17 3 2 5 4 = 28 :=
sorry

end NUMINAMATH_CALUDE_min_snack_cost_l2525_252553


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l2525_252539

def inequality_system (x : ℝ) : Prop :=
  x^2 - 2*x - 3 > 0 ∧ -x^2 - 3*x + 4 ≥ 0

theorem inequality_system_solution_set :
  {x : ℝ | inequality_system x} = {x : ℝ | -4 ≤ x ∧ x < -1} := by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l2525_252539


namespace NUMINAMATH_CALUDE_events_A_B_mutually_exclusive_l2525_252524

/-- Represents the possible outcomes of throwing a fair regular hexahedral die -/
inductive DieOutcome
  | one
  | two
  | three
  | four
  | five
  | six

/-- Defines event A: "the number is odd" -/
def eventA (outcome : DieOutcome) : Prop :=
  outcome = DieOutcome.one ∨ outcome = DieOutcome.three ∨ outcome = DieOutcome.five

/-- Defines event B: "the number is 4" -/
def eventB (outcome : DieOutcome) : Prop :=
  outcome = DieOutcome.four

/-- Theorem stating that events A and B are mutually exclusive -/
theorem events_A_B_mutually_exclusive :
  ∀ (outcome : DieOutcome), ¬(eventA outcome ∧ eventB outcome) :=
by
  sorry


end NUMINAMATH_CALUDE_events_A_B_mutually_exclusive_l2525_252524


namespace NUMINAMATH_CALUDE_fraction_meaningful_l2525_252589

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = 2 / x) ↔ x ≠ 0 := by sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l2525_252589


namespace NUMINAMATH_CALUDE_functional_polynomial_is_constant_l2525_252540

/-- A polynomial satisfying the given functional equation -/
def FunctionalPolynomial (p : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, p x * p y = p x + p y + p (x * y) - 1) ∧ p (-1) = 2

theorem functional_polynomial_is_constant
    (p : ℝ → ℝ) (hp : FunctionalPolynomial p) :
    ∀ x : ℝ, p x = 2 := by
  sorry

end NUMINAMATH_CALUDE_functional_polynomial_is_constant_l2525_252540


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l2525_252534

/-- Given two vectors a and b in a plane with an angle of 30° between them,
    |a| = √3, and |b| = 2, prove that |a + 2b| = √31 -/
theorem vector_sum_magnitude (a b : ℝ × ℝ) :
  let angle := 30 * π / 180
  (norm a = Real.sqrt 3) →
  (norm b = 2) →
  (a.1 * b.1 + a.2 * b.2 = norm a * norm b * Real.cos angle) →
  norm (a + 2 • b) = Real.sqrt 31 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l2525_252534


namespace NUMINAMATH_CALUDE_soccer_league_games_l2525_252564

/-- The number of teams in the soccer league -/
def num_teams : ℕ := 12

/-- The number of times each pair of teams plays against each other -/
def games_per_pair : ℕ := 4

/-- The total number of games played in the season -/
def total_games : ℕ := num_teams * (num_teams - 1) / 2 * games_per_pair

theorem soccer_league_games :
  total_games = 264 :=
sorry

end NUMINAMATH_CALUDE_soccer_league_games_l2525_252564


namespace NUMINAMATH_CALUDE_grass_withering_is_certain_event_l2525_252573

/-- An event that occurs regularly and predictably every year -/
structure AnnualEvent where
  occurs_yearly : Bool
  predictable : Bool

/-- Definition of a certain event in probability theory -/
def CertainEvent (e : AnnualEvent) : Prop :=
  e.occurs_yearly ∧ e.predictable

/-- The withering of grass on a plain as described in the poem -/
def grass_withering : AnnualEvent :=
  { occurs_yearly := true
  , predictable := true }

/-- Theorem stating that the grass withering is a certain event -/
theorem grass_withering_is_certain_event : CertainEvent grass_withering := by
  sorry

end NUMINAMATH_CALUDE_grass_withering_is_certain_event_l2525_252573


namespace NUMINAMATH_CALUDE_permutation_combination_equality_l2525_252527

theorem permutation_combination_equality (n : ℕ) : 
  (n * (n - 1) * (n - 2) = n * (n - 1) * (n - 2) * (n - 3) / 24) → n = 27 := by
  sorry

end NUMINAMATH_CALUDE_permutation_combination_equality_l2525_252527


namespace NUMINAMATH_CALUDE_tree_planting_ratio_l2525_252543

theorem tree_planting_ratio (initial_mahogany : ℕ) (initial_narra : ℕ) 
  (total_fallen : ℕ) (final_trees : ℕ) :
  initial_mahogany = 50 →
  initial_narra = 30 →
  total_fallen = 5 →
  final_trees = 88 →
  ∃ (fallen_narra fallen_mahogany planted : ℕ),
    fallen_narra + fallen_mahogany = total_fallen ∧
    fallen_mahogany = fallen_narra + 1 ∧
    planted = final_trees - (initial_mahogany + initial_narra - total_fallen) ∧
    planted * 2 = fallen_narra * 13 :=
by sorry

end NUMINAMATH_CALUDE_tree_planting_ratio_l2525_252543


namespace NUMINAMATH_CALUDE_probability_of_white_ball_l2525_252536

/-- The probability of drawing a white ball from a bag with red and white balls -/
theorem probability_of_white_ball (red_balls white_balls : ℕ) :
  red_balls = 3 → white_balls = 5 →
  (white_balls : ℚ) / ((red_balls : ℚ) + (white_balls : ℚ)) = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_white_ball_l2525_252536


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_example_l2525_252577

/-- The number of terms in an arithmetic sequence -/
def arithmeticSequenceLength (a₁ aₙ d : ℤ) : ℤ :=
  (aₙ - a₁) / d + 1

/-- Theorem: The arithmetic sequence with first term 2, last term 2017, 
    and common difference 5 has 404 terms -/
theorem arithmetic_sequence_length_example : 
  arithmeticSequenceLength 2 2017 5 = 404 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_example_l2525_252577


namespace NUMINAMATH_CALUDE_straight_part_length_l2525_252545

/-- Represents a river with straight and crooked parts -/
structure River where
  total_length : ℝ
  straight_length : ℝ
  crooked_length : ℝ

/-- The condition that the straight part is three times shorter than the crooked part -/
def straight_three_times_shorter (r : River) : Prop :=
  r.straight_length = r.crooked_length / 3

/-- The theorem stating the length of the straight part given the conditions -/
theorem straight_part_length (r : River) 
  (h1 : r.total_length = 80)
  (h2 : r.total_length = r.straight_length + r.crooked_length)
  (h3 : straight_three_times_shorter r) : 
  r.straight_length = 20 := by
  sorry

end NUMINAMATH_CALUDE_straight_part_length_l2525_252545
