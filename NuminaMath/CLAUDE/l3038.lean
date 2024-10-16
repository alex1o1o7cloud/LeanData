import Mathlib

namespace NUMINAMATH_CALUDE_range_of_a_l3038_303866

/-- The range of a for which ¬p is a necessary but not sufficient condition for ¬q -/
theorem range_of_a (a : ℝ) : 
  (a < 0) →
  (∀ x : ℝ, (x^2 - 4*a*x + 3*a^2 < 0) → 
    ((x^2 - x - 6 ≤ 0) ∨ (x^2 + 2*x - 8 > 0))) →
  (∃ x : ℝ, ((x^2 - x - 6 ≤ 0) ∨ (x^2 + 2*x - 8 > 0)) ∧ 
    (x^2 - 4*a*x + 3*a^2 ≥ 0)) →
  (a ≤ -4 ∨ (-2/3 ≤ a ∧ a < 0)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3038_303866


namespace NUMINAMATH_CALUDE_divisibility_by_five_l3038_303886

theorem divisibility_by_five (k m n : ℕ+) 
  (hk : ¬ 5 ∣ k.val) (hm : ¬ 5 ∣ m.val) (hn : ¬ 5 ∣ n.val) : 
  5 ∣ (k.val^2 - m.val^2) ∨ 5 ∣ (m.val^2 - n.val^2) ∨ 5 ∣ (n.val^2 - k.val^2) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_five_l3038_303886


namespace NUMINAMATH_CALUDE_no_non_zero_integer_solution_l3038_303898

theorem no_non_zero_integer_solution :
  ∀ (a b c n : ℤ), 6 * (6 * a^2 + 3 * b^2 + c^2) = 5 * n^2 → a = 0 ∧ b = 0 ∧ c = 0 ∧ n = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_non_zero_integer_solution_l3038_303898


namespace NUMINAMATH_CALUDE_square_of_1085_l3038_303867

theorem square_of_1085 : (1085 : ℕ)^2 = 1177225 := by
  sorry

end NUMINAMATH_CALUDE_square_of_1085_l3038_303867


namespace NUMINAMATH_CALUDE_expression_evaluation_l3038_303897

theorem expression_evaluation : 
  (((3 : ℚ) + 6 + 9) / ((2 : ℚ) + 5 + 8) - ((2 : ℚ) + 5 + 8) / ((3 : ℚ) + 6 + 9)) = 11 / 30 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3038_303897


namespace NUMINAMATH_CALUDE_hexagons_in_50th_ring_l3038_303839

/-- Represents the number of hexagons in a ring of a hexagonal arrangement -/
def hexagonsInRing (n : ℕ) : ℕ := 6 * n

/-- The hexagonal arrangement has the following properties:
    1. The center is a regular hexagon of unit side length
    2. Surrounded by rings of unit hexagons
    3. The first ring consists of 6 unit hexagons
    4. The second ring contains 12 unit hexagons -/
axiom hexagonal_arrangement_properties : True

theorem hexagons_in_50th_ring : 
  hexagonsInRing 50 = 300 := by sorry

end NUMINAMATH_CALUDE_hexagons_in_50th_ring_l3038_303839


namespace NUMINAMATH_CALUDE_only_negative_number_l3038_303879

theorem only_negative_number (a b c d : ℤ) (h1 : a = 5) (h2 : b = 1) (h3 : c = -2) (h4 : d = 0) :
  (a < 0 ∨ b < 0 ∨ c < 0 ∨ d < 0) ∧ (c < 0) ∧ (a ≥ 0) ∧ (b ≥ 0) ∧ (d ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_only_negative_number_l3038_303879


namespace NUMINAMATH_CALUDE_decagon_angle_property_l3038_303812

theorem decagon_angle_property (n : ℕ) : 
  (n - 2) * 180 = 360 * 4 ↔ n = 10 := by sorry

end NUMINAMATH_CALUDE_decagon_angle_property_l3038_303812


namespace NUMINAMATH_CALUDE_triangle_sine_b_l3038_303831

theorem triangle_sine_b (A B C a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π → -- Triangle angle condition
  a > 0 ∧ b > 0 ∧ c > 0 → -- Positive side lengths
  a = 2 * Real.sin (A/2) * Real.sin (B/2) * Real.sin (C/2) → -- Law of sines
  b = 2 * Real.sin (A/2) * Real.sin (C/2) → -- Law of sines
  c = 2 * Real.sin (A/2) * Real.sin (B/2) → -- Law of sines
  a + c = 2*b → -- Given condition
  A - C = π/3 → -- Given condition
  Real.sin B = Real.sqrt 39 / 8 := by sorry

end NUMINAMATH_CALUDE_triangle_sine_b_l3038_303831


namespace NUMINAMATH_CALUDE_log_product_equals_one_l3038_303809

theorem log_product_equals_one : Real.log 2 / Real.log 5 * (2 * Real.log 5 / (2 * Real.log 2)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_product_equals_one_l3038_303809


namespace NUMINAMATH_CALUDE_expression_mod_18_l3038_303824

theorem expression_mod_18 : (234 * 18 - 23 * 9 + 5) % 18 = 14 := by
  sorry

end NUMINAMATH_CALUDE_expression_mod_18_l3038_303824


namespace NUMINAMATH_CALUDE_inscribed_circle_area_l3038_303851

/-- The area of a circle inscribed in a sector of a circle -/
theorem inscribed_circle_area (R a : ℝ) (h₁ : R > 0) (h₂ : a > 0) :
  let r := R * a / (R + a)
  π * r^2 = π * (R * a / (R + a))^2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_area_l3038_303851


namespace NUMINAMATH_CALUDE_greatest_power_of_three_in_factorial_l3038_303823

theorem greatest_power_of_three_in_factorial :
  (∃ (n : ℕ), n = 9 ∧ 
   ∀ (k : ℕ), 3^k ∣ Nat.factorial 22 → k ≤ n) ∧
   (∀ (m : ℕ), m > 9 → ¬(3^m ∣ Nat.factorial 22)) := by
sorry

end NUMINAMATH_CALUDE_greatest_power_of_three_in_factorial_l3038_303823


namespace NUMINAMATH_CALUDE_quadratic_equation_conversion_l3038_303864

theorem quadratic_equation_conversion (x : ℝ) : 
  (x - 2) * (x + 3) = 1 ↔ x^2 + x - 7 = 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_conversion_l3038_303864


namespace NUMINAMATH_CALUDE_theater_ticket_sales_l3038_303826

theorem theater_ticket_sales (adult_price kid_price profit kid_tickets : ℕ) 
  (h1 : adult_price = 6)
  (h2 : kid_price = 2)
  (h3 : profit = 750)
  (h4 : kid_tickets = 75) :
  ∃ (adult_tickets : ℕ), adult_tickets * adult_price + kid_tickets * kid_price = profit ∧
                          adult_tickets + kid_tickets = 175 :=
by sorry

end NUMINAMATH_CALUDE_theater_ticket_sales_l3038_303826


namespace NUMINAMATH_CALUDE_faye_pencil_count_l3038_303888

/-- Given that Faye arranges her pencils in rows of 5 and can make 7 full rows, 
    prove that she has 35 pencils. -/
theorem faye_pencil_count (pencils_per_row : ℕ) (num_rows : ℕ) 
  (h1 : pencils_per_row = 5)
  (h2 : num_rows = 7) : 
  pencils_per_row * num_rows = 35 := by
  sorry

#check faye_pencil_count

end NUMINAMATH_CALUDE_faye_pencil_count_l3038_303888


namespace NUMINAMATH_CALUDE_perpendicular_lines_parallel_lines_l3038_303870

-- Define the lines l₁ and l₂
def l₁ (m : ℝ) (x y : ℝ) : Prop := (m - 1) * x + 2 * y - m = 0
def l₂ (m : ℝ) (x y : ℝ) : Prop := x + m * y + m - 2 = 0

-- Define perpendicularity condition
def perpendicular (m : ℝ) : Prop := (m - 1) / 2 * (1 / m) = -1

-- Define parallelism condition
def parallel (m : ℝ) : Prop := (m - 1) / 2 = 1 / m

-- Theorem for perpendicular lines
theorem perpendicular_lines (m : ℝ) : perpendicular m → m = 1/3 := by sorry

-- Theorem for parallel lines
theorem parallel_lines (m : ℝ) : parallel m → m = 2 ∨ m = -1 := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_parallel_lines_l3038_303870


namespace NUMINAMATH_CALUDE_shortest_distance_l3038_303880

/-- The shortest distance between two points given their x and y displacements -/
theorem shortest_distance (x_displacement y_displacement : ℝ) :
  x_displacement = 4 →
  y_displacement = 3 →
  Real.sqrt (x_displacement ^ 2 + y_displacement ^ 2) = 5 := by
sorry

end NUMINAMATH_CALUDE_shortest_distance_l3038_303880


namespace NUMINAMATH_CALUDE_store_money_made_l3038_303843

/-- Represents the total money made from pencil sales -/
def total_money_made (eraser_price regular_price short_price : ℚ)
                     (eraser_sold regular_sold short_sold : ℕ) : ℚ :=
  eraser_price * eraser_sold + regular_price * regular_sold + short_price * short_sold

/-- Theorem stating that the store made $194 from the given pencil sales -/
theorem store_money_made :
  total_money_made 0.8 0.5 0.4 200 40 35 = 194 := by sorry

end NUMINAMATH_CALUDE_store_money_made_l3038_303843


namespace NUMINAMATH_CALUDE_min_distance_to_A_l3038_303849

-- Define the space
variable (X : Type) [NormedAddCommGroup X] [InnerProductSpace ℝ X] [CompleteSpace X]

-- Define points A, B, and P
variable (A B P : X)

-- Define the conditions
variable (h1 : ‖A - B‖ = 4)
variable (h2 : ‖P - A‖ - ‖P - B‖ = 3)

-- State the theorem
theorem min_distance_to_A :
  ∃ (min_dist : ℝ), min_dist = 7/2 ∧ ∀ P, ‖P - A‖ - ‖P - B‖ = 3 → ‖P - A‖ ≥ min_dist :=
sorry

end NUMINAMATH_CALUDE_min_distance_to_A_l3038_303849


namespace NUMINAMATH_CALUDE_candy_probability_theorem_l3038_303889

/-- The probability of selecting the same candy type for the first and last candy -/
def same_type_probability (lollipops chocolate jelly : ℕ) : ℚ :=
  let total := lollipops + chocolate + jelly
  let p_lollipop := (lollipops : ℚ) / total * (lollipops - 1) / (total - 1)
  let p_chocolate := (chocolate : ℚ) / total * (chocolate - 1) / (total - 1)
  let p_jelly := (jelly : ℚ) / total * (jelly - 1) / (total - 1)
  p_lollipop + p_chocolate + p_jelly

theorem candy_probability_theorem :
  same_type_probability 2 3 5 = 14 / 45 := by
  sorry

#eval same_type_probability 2 3 5

end NUMINAMATH_CALUDE_candy_probability_theorem_l3038_303889


namespace NUMINAMATH_CALUDE_correct_algebraic_equality_l3038_303844

theorem correct_algebraic_equality (x y : ℝ) : -4 * x * y + 3 * x * y = -x * y := by
  sorry

end NUMINAMATH_CALUDE_correct_algebraic_equality_l3038_303844


namespace NUMINAMATH_CALUDE_only_B_and_C_have_inverses_l3038_303801

-- Define the set of functions
inductive Function : Type
| A | B | C | D | E

-- Define the property of having an inverse
def has_inverse (f : Function) : Prop :=
  match f with
  | Function.A => False
  | Function.B => True
  | Function.C => True
  | Function.D => False
  | Function.E => False

-- Theorem statement
theorem only_B_and_C_have_inverses :
  ∀ f : Function, has_inverse f ↔ (f = Function.B ∨ f = Function.C) :=
by sorry

end NUMINAMATH_CALUDE_only_B_and_C_have_inverses_l3038_303801


namespace NUMINAMATH_CALUDE_jack_christina_lindy_problem_l3038_303873

/-- The problem setup and solution for Jack, Christina, and Lindy's movement --/
theorem jack_christina_lindy_problem (
  initial_distance : ℝ) 
  (jack_speed christina_speed lindy_speed : ℝ) 
  (h1 : initial_distance = 150)
  (h2 : jack_speed = 7)
  (h3 : christina_speed = 8)
  (h4 : lindy_speed = 10) :
  let meeting_time := initial_distance / (jack_speed + christina_speed)
  lindy_speed * meeting_time = 100 := by
  sorry


end NUMINAMATH_CALUDE_jack_christina_lindy_problem_l3038_303873


namespace NUMINAMATH_CALUDE_dot_product_range_l3038_303869

/-- Given points A and B in a 2D Cartesian coordinate system,
    and P on the curve y = √(1-x²), prove that the dot product
    BP · BA is bounded by 0 and 1+√2. -/
theorem dot_product_range (x y : ℝ) :
  let A : ℝ × ℝ := (1, 0)
  let B : ℝ × ℝ := (0, -1)
  let P : ℝ × ℝ := (x, y)
  y = Real.sqrt (1 - x^2) →
  0 ≤ ((P.1 - B.1) * (A.1 - B.1) + (P.2 - B.2) * (A.2 - B.2)) ∧
  ((P.1 - B.1) * (A.1 - B.1) + (P.2 - B.2) * (A.2 - B.2)) ≤ 1 + Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_dot_product_range_l3038_303869


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3038_303845

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 1 + a 2 + a 3 = 4) →
  (a 2 + a 3 + a 4 = -2) →
  (a 3 + a 4 + a 5 + a 6 + a 7 + a 8 = 7/8) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3038_303845


namespace NUMINAMATH_CALUDE_x_gt_y_iff_x_minus_y_plus_sin_gt_zero_l3038_303808

theorem x_gt_y_iff_x_minus_y_plus_sin_gt_zero (x y : ℝ) :
  x > y ↔ x - y + Real.sin (x - y) > 0 := by sorry

end NUMINAMATH_CALUDE_x_gt_y_iff_x_minus_y_plus_sin_gt_zero_l3038_303808


namespace NUMINAMATH_CALUDE_distribution_five_to_three_l3038_303882

/-- The number of ways to distribute n distinct objects into k distinct groups,
    where each group must contain at least min_per_group objects and
    at most max_per_group objects. -/
def distribution_count (n k min_per_group max_per_group : ℕ) : ℕ := sorry

/-- Theorem: There are 30 ways to distribute 5 distinct objects into 3 distinct groups,
    where each group must contain at least 1 object and at most 2 objects. -/
theorem distribution_five_to_three : distribution_count 5 3 1 2 = 30 := by sorry

end NUMINAMATH_CALUDE_distribution_five_to_three_l3038_303882


namespace NUMINAMATH_CALUDE_majorization_iff_transformable_l3038_303863

/-- Represents a triplet of real numbers -/
structure Triplet where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Defines majorization relation between two triplets -/
def majorizes (α β : Triplet) : Prop :=
  α.a ≥ β.a ∧ α.a + α.b ≥ β.a + β.b ∧ α.a + α.b + α.c = β.a + β.b + β.c

/-- Represents the allowed operations on triplets -/
inductive Operation
  | op1 : Operation  -- (k, j, i) ↔ (k-1, j+1, i)
  | op2 : Operation  -- (k, j, i) ↔ (k-1, j, i+1)
  | op3 : Operation  -- (k, j, i) ↔ (k, j-1, i+1)

/-- Applies an operation to a triplet -/
def applyOperation (t : Triplet) (op : Operation) : Triplet :=
  match op with
  | Operation.op1 => ⟨t.a - 1, t.b + 1, t.c⟩
  | Operation.op2 => ⟨t.a - 1, t.b, t.c + 1⟩
  | Operation.op3 => ⟨t.a, t.b - 1, t.c + 1⟩

/-- Checks if one triplet can be obtained from another using allowed operations -/
def canObtain (α β : Triplet) : Prop :=
  ∃ (ops : List Operation), β = ops.foldl applyOperation α

/-- Main theorem: Majorization is equivalent to ability to transform using allowed operations -/
theorem majorization_iff_transformable (α β : Triplet) :
  majorizes α β ↔ canObtain α β := by sorry

end NUMINAMATH_CALUDE_majorization_iff_transformable_l3038_303863


namespace NUMINAMATH_CALUDE_profit_percent_for_2_3_ratio_l3038_303832

/-- Given a cost price to selling price ratio of 2:3, the profit percent is 50%. -/
theorem profit_percent_for_2_3_ratio :
  ∀ (cp sp : ℝ), cp > 0 → sp > 0 →
  cp / sp = 2 / 3 →
  ((sp - cp) / cp) * 100 = 50 := by
sorry

end NUMINAMATH_CALUDE_profit_percent_for_2_3_ratio_l3038_303832


namespace NUMINAMATH_CALUDE_locus_of_Y_l3038_303871

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a line in 2D space -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Represents a trapezoid -/
structure Trapezoid :=
  (A B C D : Point)

/-- Defines a perpendicular line to the bases of a trapezoid -/
def perpendicularLine (t : Trapezoid) : Line := sorry

/-- Defines a point on a given line -/
def pointOnLine (l : Line) : Point := sorry

/-- Constructs perpendiculars from points to lines -/
def perpendicular (p : Point) (l : Line) : Line := sorry

/-- Finds the intersection of two lines -/
def lineIntersection (l1 l2 : Line) : Point := sorry

/-- Checks if a point is on a line -/
def isPointOnLine (p : Point) (l : Line) : Prop := sorry

/-- Checks if a line divides a segment in a given ratio -/
def dividesSameRatio (l1 l2 : Line) (seg1 seg2 : Point × Point) : Prop := sorry

/-- Main theorem: The locus of point Y is a line perpendicular to the bases -/
theorem locus_of_Y (t : Trapezoid) (l : Line) : 
  ∃ l' : Line, 
    (∀ X : Point, isPointOnLine X l → 
      let BX := Line.mk sorry sorry sorry
      let CX := Line.mk sorry sorry sorry
      let perp1 := perpendicular t.A BX
      let perp2 := perpendicular t.D CX
      let Y := lineIntersection perp1 perp2
      isPointOnLine Y l') ∧ 
    dividesSameRatio l' l (t.A, t.D) (t.B, t.C) := by
  sorry

end NUMINAMATH_CALUDE_locus_of_Y_l3038_303871


namespace NUMINAMATH_CALUDE_smaller_number_problem_l3038_303883

theorem smaller_number_problem (x y : ℕ+) 
  (h_product : x * y = 323)
  (h_difference : x - y = 2) :
  y = 17 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l3038_303883


namespace NUMINAMATH_CALUDE_larger_number_of_two_l3038_303810

theorem larger_number_of_two (x y : ℝ) : 
  x - y = 7 → x + y = 41 → max x y = 24 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_of_two_l3038_303810


namespace NUMINAMATH_CALUDE_distance_to_line_implies_ab_bound_l3038_303815

theorem distance_to_line_implies_ab_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let P : ℝ × ℝ := (1, 1)
  let line (x y : ℝ) := (a + 1) * x + (b + 1) * y - 2 = 0
  let distance_to_line := |((a + 1) * P.1 + (b + 1) * P.2 - 2)| / Real.sqrt ((a + 1)^2 + (b + 1)^2)
  distance_to_line = 1 → a * b ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_line_implies_ab_bound_l3038_303815


namespace NUMINAMATH_CALUDE_f_value_for_specific_inputs_l3038_303842

-- Define the function f
def f (m n k p : ℕ) : ℤ := (n^2 - m) * (n^k - m^p)

-- Theorem statement
theorem f_value_for_specific_inputs :
  f 5 3 2 3 = -464 :=
by sorry

end NUMINAMATH_CALUDE_f_value_for_specific_inputs_l3038_303842


namespace NUMINAMATH_CALUDE_shekar_average_marks_l3038_303835

def shekar_scores : List ℕ := [76, 65, 82, 62, 85]

theorem shekar_average_marks :
  (shekar_scores.sum : ℚ) / shekar_scores.length = 74 := by sorry

end NUMINAMATH_CALUDE_shekar_average_marks_l3038_303835


namespace NUMINAMATH_CALUDE_quadratic_function_property_quadratic_function_property_independent_of_b_l3038_303876

theorem quadratic_function_property (c d h b : ℝ) : 
  let f (x : ℝ) := c * x^2
  let x₁ := b - d - h
  let x₂ := b - d
  let x₃ := b + d
  let x₄ := b + d + h
  let y₁ := f x₁
  let y₂ := f x₂
  let y₃ := f x₃
  let y₄ := f x₄
  (y₁ + y₄) - (y₂ + y₃) = 2 * c * h * (2 * d + h) :=
by sorry

theorem quadratic_function_property_independent_of_b (c d h : ℝ) :
  ∀ b₁ b₂ : ℝ, 
  let f (x : ℝ) := c * x^2
  let x₁ (b : ℝ) := b - d - h
  let x₂ (b : ℝ) := b - d
  let x₃ (b : ℝ) := b + d
  let x₄ (b : ℝ) := b + d + h
  let y₁ (b : ℝ) := f (x₁ b)
  let y₂ (b : ℝ) := f (x₂ b)
  let y₃ (b : ℝ) := f (x₃ b)
  let y₄ (b : ℝ) := f (x₄ b)
  (y₁ b₁ + y₄ b₁) - (y₂ b₁ + y₃ b₁) = (y₁ b₂ + y₄ b₂) - (y₂ b₂ + y₃ b₂) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_property_quadratic_function_property_independent_of_b_l3038_303876


namespace NUMINAMATH_CALUDE_bus_delay_l3038_303859

/-- Proves that walking at 4/5 of usual speed results in a 5-minute delay -/
theorem bus_delay (usual_time : ℝ) (h : usual_time = 20) : 
  usual_time * (5/4) - usual_time = 5 := by
  sorry

end NUMINAMATH_CALUDE_bus_delay_l3038_303859


namespace NUMINAMATH_CALUDE_investment_principal_is_200_l3038_303841

/-- Represents the simple interest investment scenario -/
structure SimpleInterestInvestment where
  principal : ℝ
  rate : ℝ
  amount_after_2_years : ℝ
  amount_after_5_years : ℝ

/-- The simple interest investment satisfies the given conditions -/
def satisfies_conditions (investment : SimpleInterestInvestment) : Prop :=
  investment.amount_after_2_years = investment.principal * (1 + 2 * investment.rate) ∧
  investment.amount_after_5_years = investment.principal * (1 + 5 * investment.rate) ∧
  investment.amount_after_2_years = 260 ∧
  investment.amount_after_5_years = 350

/-- Theorem stating that the investment with the given conditions has a principal of $200 -/
theorem investment_principal_is_200 :
  ∃ (investment : SimpleInterestInvestment), 
    satisfies_conditions investment ∧ investment.principal = 200 := by
  sorry

end NUMINAMATH_CALUDE_investment_principal_is_200_l3038_303841


namespace NUMINAMATH_CALUDE_rectangle_cut_theorem_l3038_303807

/-- Represents a figure cut from the rectangle -/
structure Figure where
  area : ℕ
  perimeter : ℕ

/-- The problem statement -/
theorem rectangle_cut_theorem :
  ∃ (figures : List Figure),
    figures.length = 5 ∧
    (figures.map Figure.area).sum = 30 ∧
    (∀ f ∈ figures, f.perimeter = 2 * f.area) ∧
    (∃ x : ℕ, figures.map Figure.area = [x, x+1, x+2, x+3, x+4]) :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_cut_theorem_l3038_303807


namespace NUMINAMATH_CALUDE_sum_bounds_l3038_303804

theorem sum_bounds (r s t u : ℝ) 
  (eq : 5*r + 4*s + 3*t + 6*u = 100)
  (h1 : r ≥ s) (h2 : s ≥ t) (h3 : t ≥ u) (h4 : u ≥ 0) :
  20 ≤ r + s + t + u ∧ r + s + t + u ≤ 25 := by
  sorry

end NUMINAMATH_CALUDE_sum_bounds_l3038_303804


namespace NUMINAMATH_CALUDE_left_handed_women_percentage_l3038_303895

theorem left_handed_women_percentage
  (total : ℕ)
  (right_handed : ℕ)
  (left_handed : ℕ)
  (men : ℕ)
  (women : ℕ)
  (h1 : right_handed = 3 * left_handed)
  (h2 : men = 3 * women / 2)
  (h3 : total = right_handed + left_handed)
  (h4 : total = men + women)
  (h5 : right_handed ≥ men)
  (h6 : right_handed = men) :
  women = left_handed ∧ left_handed * 100 / total = 25 :=
sorry

end NUMINAMATH_CALUDE_left_handed_women_percentage_l3038_303895


namespace NUMINAMATH_CALUDE_probability_at_least_one_green_l3038_303887

theorem probability_at_least_one_green (total : ℕ) (red : ℕ) (green : ℕ) (choose : ℕ) :
  total = red + green →
  total = 10 →
  red = 6 →
  green = 4 →
  choose = 3 →
  (1 : ℚ) - (Nat.choose red choose : ℚ) / (Nat.choose total choose : ℚ) = 5 / 6 :=
by sorry

end NUMINAMATH_CALUDE_probability_at_least_one_green_l3038_303887


namespace NUMINAMATH_CALUDE_min_red_beads_l3038_303840

/-- Represents a necklace with blue and red beads. -/
structure Necklace where
  blue_count : ℕ
  red_count : ℕ
  cyclic : Bool
  segment_condition : Bool

/-- Checks if a necklace satisfies the given conditions. -/
def is_valid_necklace (n : Necklace) : Prop :=
  n.blue_count = 50 ∧
  n.cyclic ∧
  n.segment_condition

/-- Theorem stating the minimum number of red beads required. -/
theorem min_red_beads (n : Necklace) :
  is_valid_necklace n → n.red_count ≥ 29 := by
  sorry

#check min_red_beads

end NUMINAMATH_CALUDE_min_red_beads_l3038_303840


namespace NUMINAMATH_CALUDE_orange_bags_weight_l3038_303853

/-- If 12 bags of oranges weigh 24 pounds, then 8 bags of oranges weigh 16 pounds. -/
theorem orange_bags_weight (total_weight : ℝ) (total_bags : ℕ) (target_bags : ℕ) :
  total_weight = 24 ∧ total_bags = 12 ∧ target_bags = 8 →
  (target_bags : ℝ) * (total_weight / total_bags) = 16 :=
by sorry

end NUMINAMATH_CALUDE_orange_bags_weight_l3038_303853


namespace NUMINAMATH_CALUDE_stream_speed_l3038_303877

theorem stream_speed (boat_speed : ℝ) (downstream_distance : ℝ) (upstream_distance : ℝ) :
  boat_speed = 30 →
  downstream_distance = 80 →
  upstream_distance = 40 →
  (downstream_distance / (boat_speed + x) = upstream_distance / (boat_speed - x)) →
  x = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l3038_303877


namespace NUMINAMATH_CALUDE_tshirt_sale_duration_l3038_303874

/-- Calculates the duration of a t-shirt sale given the number of shirts sold,
    their prices, and the revenue rate per minute. -/
theorem tshirt_sale_duration
  (total_shirts : ℕ)
  (black_shirts : ℕ)
  (white_shirts : ℕ)
  (black_price : ℚ)
  (white_price : ℚ)
  (revenue_rate : ℚ)
  (h1 : total_shirts = 200)
  (h2 : black_shirts = total_shirts / 2)
  (h3 : white_shirts = total_shirts / 2)
  (h4 : black_price = 30)
  (h5 : white_price = 25)
  (h6 : revenue_rate = 220) :
  (black_shirts * black_price + white_shirts * white_price) / revenue_rate = 25 := by
  sorry

end NUMINAMATH_CALUDE_tshirt_sale_duration_l3038_303874


namespace NUMINAMATH_CALUDE_race_winning_post_distance_l3038_303868

theorem race_winning_post_distance
  (speed_ratio : ℝ)
  (head_start : ℝ)
  (h_speed_ratio : speed_ratio = 1.75)
  (h_head_start : head_start = 84)
  : ∃ (distance : ℝ),
    distance = 196 ∧
    distance / speed_ratio = (distance - head_start) / 1 :=
by sorry

end NUMINAMATH_CALUDE_race_winning_post_distance_l3038_303868


namespace NUMINAMATH_CALUDE_new_years_appetizer_l3038_303846

/-- The number of bags of chips Alex bought for his New Year's Eve appetizer -/
def num_bags : ℕ := 3

/-- The cost of each bag of chips in dollars -/
def cost_per_bag : ℚ := 1

/-- The cost of creme fraiche in dollars -/
def cost_creme_fraiche : ℚ := 5

/-- The cost of caviar in dollars -/
def cost_caviar : ℚ := 73

/-- The total cost per person in dollars -/
def cost_per_person : ℚ := 27

theorem new_years_appetizer :
  (cost_per_bag * num_bags + cost_creme_fraiche + cost_caviar) / num_bags = cost_per_person :=
by
  sorry

end NUMINAMATH_CALUDE_new_years_appetizer_l3038_303846


namespace NUMINAMATH_CALUDE_no_integer_solution_l3038_303850

theorem no_integer_solution :
  ∀ (a b : ℤ), 
    0 ≤ a ∧ 
    0 < b ∧ 
    a < 9 ∧ 
    b < 4 →
    ¬(1 < (a : ℝ) + (b : ℝ) * Real.sqrt 5 ∧ (a : ℝ) + (b : ℝ) * Real.sqrt 5 < 9 + 4 * Real.sqrt 5) :=
by
  sorry


end NUMINAMATH_CALUDE_no_integer_solution_l3038_303850


namespace NUMINAMATH_CALUDE_watch_synchronization_l3038_303822

/-- The number of seconds in a full rotation of a standard watch -/
def full_rotation : ℕ := 12 * 60 * 60

/-- The number of seconds Glafira's watch gains per day -/
def glafira_gain : ℕ := 12

/-- The number of seconds Gavrila's watch loses per day -/
def gavrila_loss : ℕ := 18

/-- The combined deviation of both watches per day -/
def combined_deviation : ℕ := glafira_gain + gavrila_loss

theorem watch_synchronization :
  (full_rotation / combined_deviation : ℚ) = 1440 := by sorry

end NUMINAMATH_CALUDE_watch_synchronization_l3038_303822


namespace NUMINAMATH_CALUDE_crackers_per_friend_l3038_303829

/-- Given that Matthew had 23 crackers initially, has 11 crackers left, and gave equal numbers of crackers to 2 friends, prove that each friend ate 6 crackers. -/
theorem crackers_per_friend (initial_crackers : ℕ) (remaining_crackers : ℕ) (num_friends : ℕ) :
  initial_crackers = 23 →
  remaining_crackers = 11 →
  num_friends = 2 →
  (initial_crackers - remaining_crackers) / num_friends = 6 :=
by sorry

end NUMINAMATH_CALUDE_crackers_per_friend_l3038_303829


namespace NUMINAMATH_CALUDE_combined_work_time_l3038_303893

def worker_a_time : ℝ := 10
def worker_b_time : ℝ := 15

theorem combined_work_time : 
  let combined_rate := (1 / worker_a_time) + (1 / worker_b_time)
  1 / combined_rate = 6 := by sorry

end NUMINAMATH_CALUDE_combined_work_time_l3038_303893


namespace NUMINAMATH_CALUDE_fencing_required_l3038_303818

/-- Calculates the required fencing for a rectangular field -/
theorem fencing_required (area : ℝ) (side : ℝ) : 
  area = 810 ∧ side = 30 → 
  ∃ (other_side : ℝ), 
    area = side * other_side ∧ 
    side + other_side + side = 87 := by
  sorry

end NUMINAMATH_CALUDE_fencing_required_l3038_303818


namespace NUMINAMATH_CALUDE_cards_in_play_l3038_303860

/-- The number of cards in a standard deck --/
def standard_deck : ℕ := 52

/-- The number of cards kept away --/
def cards_kept_away : ℕ := 2

/-- Theorem: The number of cards being played with is 50 --/
theorem cards_in_play (deck : ℕ) (kept_away : ℕ) 
  (h1 : deck = standard_deck) (h2 : kept_away = cards_kept_away) : 
  deck - kept_away = 50 := by
  sorry

end NUMINAMATH_CALUDE_cards_in_play_l3038_303860


namespace NUMINAMATH_CALUDE_sin_210_degrees_l3038_303813

theorem sin_210_degrees : Real.sin (210 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_210_degrees_l3038_303813


namespace NUMINAMATH_CALUDE_possible_values_of_a_l3038_303856

-- Define the sets P and S
def P : Set ℝ := {x | x^2 + x - 6 = 0}
def S (a : ℝ) : Set ℝ := {x | a * x + 1 = 0}

-- Define the set of possible values for a
def A : Set ℝ := {0, 1/3, -1/2}

-- Theorem statement
theorem possible_values_of_a :
  ∀ a : ℝ, (S a ⊆ P) ↔ (a ∈ A) := by sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l3038_303856


namespace NUMINAMATH_CALUDE_shaded_triangle_area_sum_l3038_303865

/-- The sum of areas of shaded triangles in a right triangle with legs of length 8,
    when divided by connecting midpoints and shading one of four resulting triangles
    for 100 iterations, is approximately 11 square units. -/
theorem shaded_triangle_area_sum (n : ℕ) : 
  let initial_area : ℝ := 32
  let shaded_ratio : ℝ := 1 / 4
  let series_sum : ℝ := initial_area * (1 - shaded_ratio^n) / (1 - shaded_ratio)
  ∃ (ε : ℝ), ε > 0 ∧ |series_sum - 11| < ε :=
sorry

#check shaded_triangle_area_sum 100

end NUMINAMATH_CALUDE_shaded_triangle_area_sum_l3038_303865


namespace NUMINAMATH_CALUDE_triangle_formation_l3038_303828

/-- Given two sticks of lengths 3 and 5, determine if a third stick of length l can form a triangle with them. -/
def can_form_triangle (l : ℝ) : Prop :=
  l > 0 ∧ l + 3 > 5 ∧ l + 5 > 3 ∧ 3 + 5 > l

theorem triangle_formation :
  can_form_triangle 5 ∧
  ¬can_form_triangle 2 ∧
  ¬can_form_triangle 8 ∧
  ¬can_form_triangle 11 :=
sorry

end NUMINAMATH_CALUDE_triangle_formation_l3038_303828


namespace NUMINAMATH_CALUDE_benjamin_total_steps_l3038_303847

/-- Calculates the total distance traveled in steps given various modes of transportation -/
def total_steps_traveled (steps_per_mile : ℕ) (initial_walk : ℕ) (subway_miles : ℕ) (second_walk : ℕ) (cab_miles : ℕ) : ℕ :=
  initial_walk + (subway_miles * steps_per_mile) + second_walk + (cab_miles * steps_per_mile)

/-- The total steps traveled by Benjamin is 24000 -/
theorem benjamin_total_steps :
  total_steps_traveled 2000 2000 7 3000 3 = 24000 := by
  sorry


end NUMINAMATH_CALUDE_benjamin_total_steps_l3038_303847


namespace NUMINAMATH_CALUDE_minimum_n_for_inequality_l3038_303896

theorem minimum_n_for_inequality :
  (∃ (n : ℕ), ∀ (x y z : ℝ), (x^2 + y^2 + z^2)^2 ≤ n * (x^4 + y^4 + z^4)) ∧
  (∀ (x y z : ℝ), (x^2 + y^2 + z^2)^2 ≤ 3 * (x^4 + y^4 + z^4)) ∧
  (∀ (m : ℕ), m < 3 → ∃ (x y z : ℝ), (x^2 + y^2 + z^2)^2 > m * (x^4 + y^4 + z^4)) :=
by sorry

end NUMINAMATH_CALUDE_minimum_n_for_inequality_l3038_303896


namespace NUMINAMATH_CALUDE_sandys_phone_bill_l3038_303858

theorem sandys_phone_bill (kim_age : ℕ) (sandy_age : ℕ) (sandy_bill : ℕ) :
  kim_age = 10 →
  sandy_age + 2 = 3 * (kim_age + 2) →
  sandy_bill = 10 * sandy_age →
  sandy_bill = 340 := by
  sorry

end NUMINAMATH_CALUDE_sandys_phone_bill_l3038_303858


namespace NUMINAMATH_CALUDE_set_operations_l3038_303855

def A : Set ℕ := {x | x > 0 ∧ x < 9}
def B : Set ℕ := {1, 2, 3}
def C : Set ℕ := {3, 4, 5, 6}

theorem set_operations :
  (A ∩ B = {1, 2, 3}) ∧
  (A ∩ C = {3, 4, 5, 6}) ∧
  (A ∩ (B ∪ C) = {1, 2, 3, 4, 5, 6}) ∧
  (A ∪ (B ∩ C) = {1, 2, 3, 4, 5, 6, 7, 8}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l3038_303855


namespace NUMINAMATH_CALUDE_M_equals_N_l3038_303825

/-- Definition of set M -/
def M : Set ℤ := {u | ∃ m n l : ℤ, u = 12*m + 8*n + 4*l}

/-- Definition of set N -/
def N : Set ℤ := {u | ∃ p q r : ℤ, u = 20*p + 16*q + 12*r}

/-- Theorem stating that M equals N -/
theorem M_equals_N : M = N := by sorry

end NUMINAMATH_CALUDE_M_equals_N_l3038_303825


namespace NUMINAMATH_CALUDE_two_and_one_third_of_eighteen_is_fortytwo_l3038_303806

theorem two_and_one_third_of_eighteen_is_fortytwo : 
  (7 : ℚ) / 3 * 18 = 42 := by sorry

end NUMINAMATH_CALUDE_two_and_one_third_of_eighteen_is_fortytwo_l3038_303806


namespace NUMINAMATH_CALUDE_race_time_difference_l3038_303834

/-- Proves that the difference in time taken by two teams to complete a 300-mile course is 3 hours,
    given that one team's speed is 5 mph greater than the other team's speed of 20 mph. -/
theorem race_time_difference (distance : ℝ) (speed_E : ℝ) (speed_diff : ℝ) : 
  distance = 300 → 
  speed_E = 20 → 
  speed_diff = 5 → 
  distance / speed_E - distance / (speed_E + speed_diff) = 3 := by
sorry

end NUMINAMATH_CALUDE_race_time_difference_l3038_303834


namespace NUMINAMATH_CALUDE_m_minus_n_equals_l3038_303852

def M : Set Nat := {1, 3, 5, 7, 9}
def N : Set Nat := {2, 3, 5}

def setDifference (A B : Set Nat) : Set Nat :=
  {x | x ∈ A ∧ x ∉ B}

theorem m_minus_n_equals : setDifference M N = {1, 7, 9} := by
  sorry

end NUMINAMATH_CALUDE_m_minus_n_equals_l3038_303852


namespace NUMINAMATH_CALUDE_rectangle_area_change_l3038_303816

/-- Given a rectangle with dimensions 4 and 6, if shortening one side by 1
    results in an area of 18, then shortening the other side by 1
    results in an area of 20. -/
theorem rectangle_area_change (l w : ℝ) : 
  l = 4 ∧ w = 6 ∧ 
  ((l - 1) * w = 18 ∨ l * (w - 1) = 18) →
  (l * (w - 1) = 20 ∨ (l - 1) * w = 20) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l3038_303816


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_not_opposite_l3038_303819

structure Bag where
  red : Nat
  white : Nat
  black : Nat

def draw_two_balls (b : Bag) : Nat := b.red + b.white + b.black - 2

def exactly_one_white (b : Bag) : Prop := 
  ∃ (x : Nat), x = 1 ∧ x ≤ b.white ∧ x ≤ draw_two_balls b

def exactly_two_white (b : Bag) : Prop := 
  ∃ (x : Nat), x = 2 ∧ x ≤ b.white ∧ x ≤ draw_two_balls b

def mutually_exclusive (p q : Prop) : Prop :=
  ¬(p ∧ q)

def opposite (p q : Prop) : Prop :=
  (p ↔ ¬q) ∧ (q ↔ ¬p)

theorem events_mutually_exclusive_not_opposite 
  (b : Bag) (h1 : b.red = 3) (h2 : b.white = 2) (h3 : b.black = 1) : 
  mutually_exclusive (exactly_one_white b) (exactly_two_white b) ∧ 
  ¬(opposite (exactly_one_white b) (exactly_two_white b)) := by
  sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_not_opposite_l3038_303819


namespace NUMINAMATH_CALUDE_unique_a_value_l3038_303802

/-- The function f(x) = ax³ - 3x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x + 1

/-- The theorem stating that a = 4 is the unique value satisfying the condition -/
theorem unique_a_value : ∃! a : ℝ, ∀ x : ℝ, x ∈ Set.Icc (-1) 1 → f a x ≥ 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_a_value_l3038_303802


namespace NUMINAMATH_CALUDE_area_inside_Q_outside_P_R_l3038_303892

-- Define the circles
def circle_P : Real := 1
def circle_Q : Real := 2
def circle_R : Real := 1

-- Define the centers of the circles
def center_P : ℝ × ℝ := (0, 0)
def center_R : ℝ × ℝ := (2, 0)
def center_Q : ℝ × ℝ := (0, 0)

-- Define the tangency conditions
def Q_R_tangent : Prop := 
  (center_Q.1 - center_R.1)^2 + (center_Q.2 - center_R.2)^2 = (circle_Q + circle_R)^2

def R_P_tangent : Prop :=
  (center_R.1 - center_P.1)^2 + (center_R.2 - center_P.2)^2 = (circle_R + circle_P)^2

-- Theorem statement
theorem area_inside_Q_outside_P_R : 
  Q_R_tangent → R_P_tangent → 
  (π * circle_Q^2) - (π * circle_P^2) - (π * circle_R^2) = 2 * π := by
  sorry

end NUMINAMATH_CALUDE_area_inside_Q_outside_P_R_l3038_303892


namespace NUMINAMATH_CALUDE_max_value_properties_l3038_303872

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / (1 + x) - Real.log x

theorem max_value_properties (x₀ : ℝ) 
  (h₁ : ∀ x > 0, f x ≤ f x₀) 
  (h₂ : x₀ > 0) :
  f x₀ = x₀ ∧ f x₀ < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_properties_l3038_303872


namespace NUMINAMATH_CALUDE_jacob_shooting_improvement_l3038_303875

/-- Represents the number of shots Jacob made in the fourth game -/
def shots_made_fourth_game : ℕ := 9

/-- Represents Jacob's initial number of shots -/
def initial_shots : ℕ := 45

/-- Represents Jacob's initial number of successful shots -/
def initial_successful_shots : ℕ := 18

/-- Represents the number of shots Jacob attempted in the fourth game -/
def fourth_game_attempts : ℕ := 15

/-- Represents Jacob's initial shooting average as a rational number -/
def initial_average : ℚ := 2/5

/-- Represents Jacob's final shooting average as a rational number -/
def final_average : ℚ := 9/20

theorem jacob_shooting_improvement :
  (initial_successful_shots + shots_made_fourth_game : ℚ) / (initial_shots + fourth_game_attempts) = final_average :=
sorry

end NUMINAMATH_CALUDE_jacob_shooting_improvement_l3038_303875


namespace NUMINAMATH_CALUDE_purchasing_power_increase_l3038_303833

theorem purchasing_power_increase (original_price : ℝ) (money : ℝ) (h : money > 0) :
  let new_price := 0.8 * original_price
  let original_quantity := money / original_price
  let new_quantity := money / new_price
  new_quantity = 1.25 * original_quantity :=
by sorry

end NUMINAMATH_CALUDE_purchasing_power_increase_l3038_303833


namespace NUMINAMATH_CALUDE_marias_painting_earnings_l3038_303814

/-- Calculates Maria's earnings from selling a painting --/
theorem marias_painting_earnings
  (brush_cost : ℕ)
  (canvas_cost_multiplier : ℕ)
  (paint_cost_per_liter : ℕ)
  (paint_liters : ℕ)
  (selling_price : ℕ)
  (h1 : brush_cost = 20)
  (h2 : canvas_cost_multiplier = 3)
  (h3 : paint_cost_per_liter = 8)
  (h4 : paint_liters ≥ 5)
  (h5 : selling_price = 200) :
  selling_price - (brush_cost + canvas_cost_multiplier * brush_cost + paint_cost_per_liter * paint_liters) = 80 :=
by sorry

end NUMINAMATH_CALUDE_marias_painting_earnings_l3038_303814


namespace NUMINAMATH_CALUDE_twelve_integer_chords_l3038_303848

/-- Represents a circle with a point inside it -/
structure CircleWithPoint where
  radius : ℝ
  distanceToCenter : ℝ

/-- Counts the number of integer-length chords through a point in a circle -/
def countIntegerChords (c : CircleWithPoint) : ℕ :=
  sorry

/-- Theorem stating that for a circle with radius 15 and a point 9 units from the center,
    there are exactly 12 integer-length chords through that point -/
theorem twelve_integer_chords :
  let c : CircleWithPoint := { radius := 15, distanceToCenter := 9 }
  countIntegerChords c = 12 := by
  sorry

end NUMINAMATH_CALUDE_twelve_integer_chords_l3038_303848


namespace NUMINAMATH_CALUDE_chord_length_squared_l3038_303881

/-- Two circles with given radii and center distance, intersecting at point P with equal chords QP and PR --/
structure IntersectingCircles where
  r1 : ℝ
  r2 : ℝ
  center_distance : ℝ
  chord_length : ℝ
  h1 : r1 = 10
  h2 : r2 = 7
  h3 : center_distance = 15
  h4 : chord_length > 0

/-- The square of the chord length in the given configuration is 154 --/
theorem chord_length_squared (ic : IntersectingCircles) : ic.chord_length ^ 2 = 154 := by
  sorry

#check chord_length_squared

end NUMINAMATH_CALUDE_chord_length_squared_l3038_303881


namespace NUMINAMATH_CALUDE_divisibility_of_subset_products_l3038_303861

def P (A : Finset Nat) : Nat := A.prod id

theorem divisibility_of_subset_products :
  let S : Finset Nat := Finset.range 2010
  let n : Nat := Nat.choose 2010 99
  let subsets : Finset (Finset Nat) := S.powerset.filter (fun A => A.card = 99)
  2010 ∣ subsets.sum P := by sorry

end NUMINAMATH_CALUDE_divisibility_of_subset_products_l3038_303861


namespace NUMINAMATH_CALUDE_two_distinct_roots_l3038_303830

theorem two_distinct_roots (p : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁ - 3) * (x₁ - 2) - p^2 = 0 ∧ (x₂ - 3) * (x₂ - 2) - p^2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_two_distinct_roots_l3038_303830


namespace NUMINAMATH_CALUDE_derivative_cos_2x_plus_1_l3038_303854

theorem derivative_cos_2x_plus_1 (x : ℝ) :
  deriv (fun x => Real.cos (2 * x + 1)) x = -2 * Real.sin (2 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_derivative_cos_2x_plus_1_l3038_303854


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l3038_303891

theorem fractional_equation_solution (k : ℝ) :
  (∃ x : ℝ, x ≠ 0 ∧ x ≠ 1 ∧ (3 / x + 6 / (x - 1) - (x + k) / (x * (x - 1)) = 0)) ↔ k ≠ -3 ∧ k ≠ 5 :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l3038_303891


namespace NUMINAMATH_CALUDE_sum_1_to_15_mod_11_l3038_303884

theorem sum_1_to_15_mod_11 : (List.range 15).sum % 11 = 10 := by sorry

end NUMINAMATH_CALUDE_sum_1_to_15_mod_11_l3038_303884


namespace NUMINAMATH_CALUDE_houses_with_one_pet_l3038_303803

/-- Represents the number of houses with different pet combinations in a neighborhood --/
structure PetHouses where
  total : ℕ
  dogs : ℕ
  cats : ℕ
  birds : ℕ
  dogsCats : ℕ
  catsBirds : ℕ
  dogsBirds : ℕ

/-- Theorem stating the number of houses with only one type of pet --/
theorem houses_with_one_pet (h : PetHouses) 
  (h_total : h.total = 75)
  (h_dogs : h.dogs = 40)
  (h_cats : h.cats = 30)
  (h_birds : h.birds = 8)
  (h_dogs_cats : h.dogsCats = 10)
  (h_cats_birds : h.catsBirds = 5)
  (h_dogs_birds : h.dogsBirds = 0) :
  h.dogs + h.cats + h.birds - h.dogsCats - h.catsBirds - h.dogsBirds = 48 := by
  sorry


end NUMINAMATH_CALUDE_houses_with_one_pet_l3038_303803


namespace NUMINAMATH_CALUDE_cricket_game_overs_l3038_303805

theorem cricket_game_overs (total_target : ℝ) (initial_rate : ℝ) (remaining_overs : ℝ) (required_rate : ℝ) 
  (h1 : total_target = 282)
  (h2 : initial_rate = 3.6)
  (h3 : remaining_overs = 40)
  (h4 : required_rate = 6.15) :
  ∃ (initial_overs : ℝ), 
    initial_overs * initial_rate + remaining_overs * required_rate = total_target ∧ 
    initial_overs = 10 := by
  sorry

end NUMINAMATH_CALUDE_cricket_game_overs_l3038_303805


namespace NUMINAMATH_CALUDE_dawn_savings_percentage_l3038_303817

theorem dawn_savings_percentage (annual_salary : ℕ) (monthly_savings : ℕ) : annual_salary = 48000 → monthly_savings = 400 → (monthly_savings : ℚ) / ((annual_salary : ℚ) / 12) = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_dawn_savings_percentage_l3038_303817


namespace NUMINAMATH_CALUDE_coprime_pairs_count_l3038_303894

def count_coprime_pairs : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 
    let (a, b) := p
    1 ≤ a ∧ a ≤ b ∧ b ≤ 5 ∧ Nat.gcd a b = 1) 
    (Finset.product (Finset.range 6) (Finset.range 6))).card

theorem coprime_pairs_count : count_coprime_pairs = 10 := by
  sorry

end NUMINAMATH_CALUDE_coprime_pairs_count_l3038_303894


namespace NUMINAMATH_CALUDE_addition_problem_l3038_303878

theorem addition_problem : (-5 : ℤ) + 8 + (-4) = -1 := by
  sorry

end NUMINAMATH_CALUDE_addition_problem_l3038_303878


namespace NUMINAMATH_CALUDE_tax_rate_above_40k_l3038_303836

/-- Proves that the tax rate on income above $40,000 is 20% given the conditions --/
theorem tax_rate_above_40k (total_income : ℝ) (total_tax : ℝ) :
  total_income = 58000 →
  total_tax = 8000 →
  (∃ (rate_above_40k : ℝ),
    total_tax = 0.11 * 40000 + rate_above_40k * (total_income - 40000) ∧
    rate_above_40k = 0.20) :=
by
  sorry

end NUMINAMATH_CALUDE_tax_rate_above_40k_l3038_303836


namespace NUMINAMATH_CALUDE_symmetry_xoz_of_point_l3038_303838

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Performs symmetry about the xOz plane -/
def symmetryXOZ (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := p.z }

theorem symmetry_xoz_of_point :
  let A : Point3D := { x := 9, y := 8, z := 5 }
  symmetryXOZ A = { x := 9, y := -8, z := 5 } := by
  sorry

end NUMINAMATH_CALUDE_symmetry_xoz_of_point_l3038_303838


namespace NUMINAMATH_CALUDE_imaginary_unit_sum_l3038_303885

theorem imaginary_unit_sum (i : ℂ) (hi : i^2 = -1) : i + i^2 + i^3 = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_sum_l3038_303885


namespace NUMINAMATH_CALUDE_count_valid_antibirthdays_l3038_303857

/-- Represents a date in day.month format -/
structure Date :=
  (day : ℕ)
  (month : ℕ)

/-- Checks if a date is valid -/
def is_valid_date (d : Date) : Prop :=
  1 ≤ d.month ∧ d.month ≤ 12 ∧ 1 ≤ d.day ∧ d.day ≤ 31

/-- Swaps the day and month of a date -/
def swap_date (d : Date) : Date :=
  ⟨d.month, d.day⟩

/-- Checks if a date has a valid anti-birthday -/
def has_valid_antibirthday (d : Date) : Prop :=
  is_valid_date d ∧ 
  is_valid_date (swap_date d) ∧ 
  d.day ≠ d.month

/-- The number of days in a year with valid anti-birthdays -/
def days_with_valid_antibirthdays : ℕ := 132

/-- Theorem stating the number of days with valid anti-birthdays -/
theorem count_valid_antibirthdays : 
  (∀ d : Date, has_valid_antibirthday d) → 
  days_with_valid_antibirthdays = 132 := by
  sorry

#check count_valid_antibirthdays

end NUMINAMATH_CALUDE_count_valid_antibirthdays_l3038_303857


namespace NUMINAMATH_CALUDE_negation_of_sum_even_both_even_l3038_303820

theorem negation_of_sum_even_both_even :
  (¬ ∀ (a b : ℤ), Even (a + b) → (Even a ∧ Even b)) ↔
  (∃ (a b : ℤ), Even (a + b) ∧ (¬ Even a ∨ ¬ Even b)) :=
sorry

end NUMINAMATH_CALUDE_negation_of_sum_even_both_even_l3038_303820


namespace NUMINAMATH_CALUDE_team_cost_comparison_l3038_303862

/-- The cost calculation for Team A and Team B based on the number of people and ticket price --/
def cost_comparison (n : ℕ+) (x : ℝ) : Prop :=
  let cost_A := x + (3/4) * x * (n - 1)
  let cost_B := (4/5) * x * n
  (n = 5 → cost_A = cost_B) ∧
  (n > 5 → cost_A < cost_B) ∧
  (n < 5 → cost_A > cost_B)

/-- Theorem stating the cost comparison between Team A and Team B --/
theorem team_cost_comparison (n : ℕ+) (x : ℝ) (hx : x > 0) :
  cost_comparison n x := by
  sorry

end NUMINAMATH_CALUDE_team_cost_comparison_l3038_303862


namespace NUMINAMATH_CALUDE_percent_relation_l3038_303800

theorem percent_relation (x y : ℝ) (h : 0.3 * (x - y) = 0.2 * (x + y)) : y = 0.2 * x := by
  sorry

end NUMINAMATH_CALUDE_percent_relation_l3038_303800


namespace NUMINAMATH_CALUDE_onion_bag_cost_l3038_303890

/-- The cost of one bag of onions -/
def cost_of_one_bag (price_per_onion : ℕ) (total_onions : ℕ) (num_bags : ℕ) : ℕ :=
  (price_per_onion * total_onions) / num_bags

/-- Theorem stating the cost of one bag of onions -/
theorem onion_bag_cost :
  let price_per_onion := 200
  let total_onions := 180
  let num_bags := 6
  cost_of_one_bag price_per_onion total_onions num_bags = 6000 := by
  sorry

end NUMINAMATH_CALUDE_onion_bag_cost_l3038_303890


namespace NUMINAMATH_CALUDE_parabola_conditions_imply_a_range_l3038_303899

theorem parabola_conditions_imply_a_range (a : ℝ) : 
  (a - 1 > 0) →  -- parabola y=(a-1)x^2 opens upwards
  (2*a - 3 < 0) →  -- parabola y=(2a-3)x^2 opens downwards
  (|a - 1| > |2*a - 3|) →  -- parabola y=(a-1)x^2 has a wider opening than y=(2a-3)x^2
  (4/3 < a ∧ a < 3/2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_conditions_imply_a_range_l3038_303899


namespace NUMINAMATH_CALUDE_lunks_for_dozen_apples_l3038_303837

-- Define the exchange rates
def lunks_per_kunk : ℚ := 7 / 4
def apples_per_kunk : ℚ := 5 / 3

-- Define a dozen
def dozen : ℕ := 12

-- Theorem statement
theorem lunks_for_dozen_apples : 
  ∃ (l : ℚ), l = dozen * (lunks_per_kunk / apples_per_kunk) ∧ l = 12.6 := by
sorry

end NUMINAMATH_CALUDE_lunks_for_dozen_apples_l3038_303837


namespace NUMINAMATH_CALUDE_fraction_simplification_l3038_303827

theorem fraction_simplification (y : ℝ) (h : y = 3) : 
  (y^6 + 8*y^3 + 16) / (y^3 + 4) = 31 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3038_303827


namespace NUMINAMATH_CALUDE_vectors_orthogonal_l3038_303821

theorem vectors_orthogonal (x : ℝ) : 
  x = 28/3 → (3 * x + 4 * (-7) = 0) := by sorry

end NUMINAMATH_CALUDE_vectors_orthogonal_l3038_303821


namespace NUMINAMATH_CALUDE_cuboid_volume_l3038_303811

/-- Given a cuboid with perimeters of opposite faces A, B, and C, prove its volume is 240 cubic centimeters -/
theorem cuboid_volume (A B C : ℝ) (hA : A = 20) (hB : B = 32) (hC : C = 28) :
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧
  2 * (x + y) = A ∧
  2 * (y + z) = B ∧
  2 * (x + z) = C ∧
  x * y * z = 240 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_volume_l3038_303811
