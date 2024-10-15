import Mathlib

namespace NUMINAMATH_CALUDE_parallelogram_side_length_l2976_297617

/-- Represents a parallelogram with side lengths -/
structure Parallelogram where
  ab : ℝ
  bc : ℝ
  cd : ℝ
  da : ℝ

/-- The property that opposite sides of a parallelogram are equal -/
def Parallelogram.oppositeSidesEqual (p : Parallelogram) : Prop :=
  p.ab = p.cd ∧ p.bc = p.da

/-- The theorem to be proved -/
theorem parallelogram_side_length 
  (p : Parallelogram) 
  (h1 : p.oppositeSidesEqual) 
  (h2 : p.ab + p.bc + p.cd + p.da = 14) 
  (h3 : p.da = 5) : 
  p.ab = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_side_length_l2976_297617


namespace NUMINAMATH_CALUDE_square_difference_divided_by_nine_l2976_297659

theorem square_difference_divided_by_nine : (109^2 - 100^2) / 9 = 209 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_divided_by_nine_l2976_297659


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2976_297607

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x - 1| ≥ 5} = {x : ℝ | x ≥ 6 ∨ x ≤ -4} := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2976_297607


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2976_297650

-- Problem 1
theorem simplify_expression_1 :
  Real.sqrt 8 + 2 * Real.sqrt 3 - (Real.sqrt 27 - Real.sqrt 2) = 3 * Real.sqrt 2 - Real.sqrt 3 :=
by sorry

-- Problem 2
theorem simplify_expression_2 (a b : ℝ) (ha : a > 0) :
  Real.sqrt (4 * a^2 * b^3) = 2 * a * b * Real.sqrt b :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2976_297650


namespace NUMINAMATH_CALUDE_expand_expression_l2976_297660

theorem expand_expression (a b : ℝ) : 3 * a * (5 * a - 2 * b) = 15 * a^2 - 6 * a * b := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2976_297660


namespace NUMINAMATH_CALUDE_tile_perimeter_change_l2976_297613

/-- Represents a shape made of square tiles -/
structure TileShape where
  tiles : ℕ
  perimeter : ℕ

/-- Adds tiles to a shape and returns the new perimeter -/
def add_tiles (shape : TileShape) (new_tiles : ℕ) : Set ℕ :=
  sorry

theorem tile_perimeter_change (initial_shape : TileShape) :
  initial_shape.tiles = 10 →
  initial_shape.perimeter = 16 →
  ∃ (new_perimeter : Set ℕ),
    new_perimeter = add_tiles initial_shape 2 ∧
    new_perimeter = {23, 25} :=
by sorry

end NUMINAMATH_CALUDE_tile_perimeter_change_l2976_297613


namespace NUMINAMATH_CALUDE_water_remaining_calculation_l2976_297638

/-- Calculates the remaining water in a bucket after some has leaked out. -/
def remaining_water (initial : ℚ) (leaked : ℚ) : ℚ :=
  initial - leaked

/-- Theorem stating that given the initial amount and leaked amount, 
    the remaining water is 0.50 gallon. -/
theorem water_remaining_calculation (initial leaked : ℚ) 
  (h1 : initial = 3/4) 
  (h2 : leaked = 1/4) : 
  remaining_water initial leaked = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_water_remaining_calculation_l2976_297638


namespace NUMINAMATH_CALUDE_baker_revenue_l2976_297637

/-- The intended revenue for a baker selling birthday cakes -/
theorem baker_revenue (n : ℝ) : 
  (∀ (reduced_price : ℝ), reduced_price = 0.8 * n → 10 * reduced_price = 8 * n) →
  8 * n = 8 * n := by sorry

end NUMINAMATH_CALUDE_baker_revenue_l2976_297637


namespace NUMINAMATH_CALUDE_equation_solution_l2976_297600

theorem equation_solution (x : ℝ) (hx : x ≠ 0) :
  (5 * x)^4 = (15 * x)^3 → x = 27 / 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2976_297600


namespace NUMINAMATH_CALUDE_expression_evaluation_l2976_297657

theorem expression_evaluation :
  let x : ℚ := -3
  let numerator := 7 + x * (4 + x) - 4^2
  let denominator := x - 4 + x^2
  numerator / denominator = -6 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2976_297657


namespace NUMINAMATH_CALUDE_cube_difference_equals_product_plus_constant_l2976_297640

theorem cube_difference_equals_product_plus_constant
  (x y : ℤ)
  (h1 : x ≥ 1)
  (h2 : y ≥ 1)
  (h3 : x^3 - y^3 = x*y + 61) :
  x = 6 ∧ y = 5 := by
sorry

end NUMINAMATH_CALUDE_cube_difference_equals_product_plus_constant_l2976_297640


namespace NUMINAMATH_CALUDE_sqrt_equation_l2976_297689

theorem sqrt_equation (n : ℝ) : Real.sqrt (8 + n) = 9 → n = 73 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_l2976_297689


namespace NUMINAMATH_CALUDE_reflect_triangle_xy_l2976_297623

/-- A triangle in a 2D coordinate plane -/
structure Triangle where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ

/-- Reflection of a point over the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- Reflection of a point over the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- Composition of reflections over x-axis and y-axis -/
def reflect_xy (p : ℝ × ℝ) : ℝ × ℝ := reflect_y (reflect_x p)

/-- Theorem: Reflecting a triangle over x-axis then y-axis negates both coordinates -/
theorem reflect_triangle_xy (t : Triangle) :
  let t' := Triangle.mk (reflect_xy t.v1) (reflect_xy t.v2) (reflect_xy t.v3)
  t'.v1 = (-t.v1.1, -t.v1.2) ∧
  t'.v2 = (-t.v2.1, -t.v2.2) ∧
  t'.v3 = (-t.v3.1, -t.v3.2) := by
  sorry

end NUMINAMATH_CALUDE_reflect_triangle_xy_l2976_297623


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l2976_297624

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  r₁ + r₂ = -b / a :=
by sorry

theorem sum_of_roots_specific_equation :
  let r₁ := (-(-10) + Real.sqrt ((-10)^2 - 4*2*3)) / (2*2)
  let r₂ := (-(-10) - Real.sqrt ((-10)^2 - 4*2*3)) / (2*2)
  r₁ + r₂ = 5 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l2976_297624


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2976_297628

theorem inequality_equivalence (x y : ℝ) :
  (2 * y + 3 * x > Real.sqrt (9 * x^2)) ↔
  ((x ≥ 0 ∧ y > 0) ∨ (x < 0 ∧ y > -3 * x)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2976_297628


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2976_297632

theorem imaginary_part_of_complex_fraction : 
  let i : ℂ := Complex.I
  let z : ℂ := (7 + i) / (3 + 4 * i)
  Complex.im z = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2976_297632


namespace NUMINAMATH_CALUDE_number_division_problem_l2976_297663

theorem number_division_problem (n : ℕ) : 
  (n / (555 + 445) = 2 * (555 - 445)) ∧ 
  (n % (555 + 445) = 40) → 
  n = 220040 := by
sorry

end NUMINAMATH_CALUDE_number_division_problem_l2976_297663


namespace NUMINAMATH_CALUDE_complex_multiplication_result_l2976_297622

theorem complex_multiplication_result : ∃ (a b : ℝ), (Complex.I + 1) * (2 - Complex.I) = Complex.mk a b ∧ a = 3 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_result_l2976_297622


namespace NUMINAMATH_CALUDE_inverse_as_polynomial_of_N_l2976_297648

def N : Matrix (Fin 2) (Fin 2) ℚ := !![3, 0; 2, -4]

theorem inverse_as_polynomial_of_N :
  let c : ℚ := 1 / 36
  let d : ℚ := -1 / 12
  N⁻¹ = c • (N ^ 2) + d • (1 : Matrix (Fin 2) (Fin 2) ℚ) := by sorry

end NUMINAMATH_CALUDE_inverse_as_polynomial_of_N_l2976_297648


namespace NUMINAMATH_CALUDE_stick_division_theorem_l2976_297697

/-- Represents a stick with markings -/
structure MarkedStick where
  divisions : List Nat

/-- Calculates the number of pieces a stick is divided into when cut at all markings -/
def numberOfPieces (stick : MarkedStick) : Nat :=
  sorry

/-- The theorem to be proven -/
theorem stick_division_theorem :
  let stick : MarkedStick := { divisions := [10, 12, 15] }
  numberOfPieces stick = 28 := by
  sorry

end NUMINAMATH_CALUDE_stick_division_theorem_l2976_297697


namespace NUMINAMATH_CALUDE_triangle_area_l2976_297662

/-- The area of a triangle with vertices at (-4,3), (0,6), and (2,-2) is 19 square units. -/
theorem triangle_area : 
  let A : ℝ × ℝ := (-4, 3)
  let B : ℝ × ℝ := (0, 6)
  let C : ℝ × ℝ := (2, -2)
  abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2) = 19 := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_l2976_297662


namespace NUMINAMATH_CALUDE_complex_coordinates_l2976_297653

theorem complex_coordinates (z : ℂ) : z = Complex.I * (2 - Complex.I) → (z.re = 1 ∧ z.im = 2) := by
  sorry

end NUMINAMATH_CALUDE_complex_coordinates_l2976_297653


namespace NUMINAMATH_CALUDE_dispatch_plans_count_l2976_297629

-- Define the total number of students
def total_students : ℕ := 6

-- Define the number of students needed for each day
def sunday_students : ℕ := 2
def friday_students : ℕ := 1
def saturday_students : ℕ := 1

-- Define the total number of students needed
def total_needed : ℕ := sunday_students + friday_students + saturday_students

-- Theorem statement
theorem dispatch_plans_count : 
  (Nat.choose total_students sunday_students) * 
  (Nat.choose (total_students - sunday_students) friday_students) * 
  (Nat.choose (total_students - sunday_students - friday_students) saturday_students) = 180 :=
by sorry

end NUMINAMATH_CALUDE_dispatch_plans_count_l2976_297629


namespace NUMINAMATH_CALUDE_dilation_of_negative_i_l2976_297644

def dilation (c k z : ℂ) : ℂ := c + k * (z - c)

theorem dilation_of_negative_i :
  let c : ℂ := 2 - 3*I
  let k : ℝ := 3
  let z : ℂ := -I
  dilation c k z = -4 + 3*I := by
  sorry

end NUMINAMATH_CALUDE_dilation_of_negative_i_l2976_297644


namespace NUMINAMATH_CALUDE_matrix_multiplication_result_l2976_297614

theorem matrix_multiplication_result : 
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![3, -1; 6, -4]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![9, -3; 2, 2]
  A * B = !![25, -11; 46, -26] := by
  sorry

end NUMINAMATH_CALUDE_matrix_multiplication_result_l2976_297614


namespace NUMINAMATH_CALUDE_no_four_digit_sum12_div11and5_l2976_297654

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ :=
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

theorem no_four_digit_sum12_div11and5 :
  ¬ ∃ n : ℕ, is_four_digit n ∧ digit_sum n = 12 ∧ n % 11 = 0 ∧ n % 5 = 0 :=
sorry

end NUMINAMATH_CALUDE_no_four_digit_sum12_div11and5_l2976_297654


namespace NUMINAMATH_CALUDE_bill_with_tip_divisibility_l2976_297633

theorem bill_with_tip_divisibility (x : ℕ) : ∃ k : ℕ, (11 * x) = (10 * k) := by
  sorry

end NUMINAMATH_CALUDE_bill_with_tip_divisibility_l2976_297633


namespace NUMINAMATH_CALUDE_intersection_M_complement_N_l2976_297696

-- Define the universal set U as ℝ
def U := ℝ

-- Define set M
def M : Set ℝ := {-1, 0, 1}

-- Define set N
def N : Set ℝ := {x : ℝ | x^2 + x = 0}

-- Theorem statement
theorem intersection_M_complement_N : M ∩ (Set.univ \ N) = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_complement_N_l2976_297696


namespace NUMINAMATH_CALUDE_coefficient_of_b_squared_l2976_297618

theorem coefficient_of_b_squared (a : ℝ) : 
  (∃ b₁ b₂ : ℝ, b₁ + b₂ = 4.5 ∧ 
    (∀ b : ℝ, 4 * b^4 - a * b^2 + 100 = 0 → b ≤ b₁ ∧ b ≤ b₂) ∧
    (4 * b₁^4 - a * b₁^2 + 100 = 0) ∧ 
    (4 * b₂^4 - a * b₂^2 + 100 = 0)) →
  a = 4.5 := by
sorry

end NUMINAMATH_CALUDE_coefficient_of_b_squared_l2976_297618


namespace NUMINAMATH_CALUDE_area_PQR_approx_5_96_l2976_297691

-- Define the square pyramid
def square_pyramid (side_length : ℝ) (height : ℝ) :=
  {base_side : ℝ // base_side = side_length ∧ height > 0}

-- Define points P, Q, and R
def point_P (pyramid : square_pyramid 4 8) : ℝ × ℝ × ℝ := sorry
def point_Q (pyramid : square_pyramid 4 8) : ℝ × ℝ × ℝ := sorry
def point_R (pyramid : square_pyramid 4 8) : ℝ × ℝ × ℝ := sorry

-- Define the area of triangle PQR
def area_PQR (pyramid : square_pyramid 4 8) : ℝ := sorry

-- Theorem statement
theorem area_PQR_approx_5_96 (pyramid : square_pyramid 4 8) :
  ∃ ε > 0, |area_PQR pyramid - 5.96| < ε :=
sorry

end NUMINAMATH_CALUDE_area_PQR_approx_5_96_l2976_297691


namespace NUMINAMATH_CALUDE_hyperbola_m_value_l2976_297668

-- Define the hyperbola equation
def hyperbola_equation (x y m : ℝ) : Prop :=
  x^2 / m - y^2 / (3 + m) = 1

-- Define the focus point
def focus : ℝ × ℝ := (2, 0)

-- Theorem statement
theorem hyperbola_m_value :
  ∀ m : ℝ, (∀ x y : ℝ, hyperbola_equation x y m) → focus.1 = 2 → focus.2 = 0 → m = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_m_value_l2976_297668


namespace NUMINAMATH_CALUDE_football_players_count_l2976_297612

theorem football_players_count (total_players cricket_players hockey_players softball_players : ℕ) :
  total_players = 55 →
  cricket_players = 15 →
  hockey_players = 12 →
  softball_players = 15 →
  total_players = cricket_players + hockey_players + softball_players + 13 :=
by
  sorry

end NUMINAMATH_CALUDE_football_players_count_l2976_297612


namespace NUMINAMATH_CALUDE_smallest_T_for_162_l2976_297602

/-- Represents the removal process of tokens in a circle -/
def removeTokens (T : ℕ) : ℕ → ℕ
| 0 => T
| n + 1 => removeTokens (T / 2) n

/-- Checks if a given T results in 162 as the last token -/
def lastTokenIs162 (T : ℕ) : Prop :=
  removeTokens T (Nat.log2 T) = 162

/-- Theorem stating that 209 is the smallest T where the last token is 162 -/
theorem smallest_T_for_162 :
  lastTokenIs162 209 ∧ ∀ k < 209, ¬lastTokenIs162 k :=
sorry

end NUMINAMATH_CALUDE_smallest_T_for_162_l2976_297602


namespace NUMINAMATH_CALUDE_square_difference_equals_product_l2976_297686

theorem square_difference_equals_product : (15 + 7)^2 - (7^2 + 15^2) = 210 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equals_product_l2976_297686


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l2976_297636

theorem at_least_one_not_less_than_two (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  max (a + 1/b) (max (b + 1/c) (c + 1/a)) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l2976_297636


namespace NUMINAMATH_CALUDE_maddie_tshirt_cost_l2976_297651

/-- Calculates the total cost of T-shirts bought by Maddie -/
def total_cost (white_packs blue_packs : ℕ) (white_per_pack blue_per_pack : ℕ) (cost_per_shirt : ℕ) : ℕ :=
  let total_shirts := white_packs * white_per_pack + blue_packs * blue_per_pack
  total_shirts * cost_per_shirt

/-- Theorem stating that Maddie spent $66 on T-shirts -/
theorem maddie_tshirt_cost :
  total_cost 2 4 5 3 3 = 66 := by
  sorry

end NUMINAMATH_CALUDE_maddie_tshirt_cost_l2976_297651


namespace NUMINAMATH_CALUDE_ab_greater_ac_l2976_297649

theorem ab_greater_ac (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) : a * b > a * c := by
  sorry

end NUMINAMATH_CALUDE_ab_greater_ac_l2976_297649


namespace NUMINAMATH_CALUDE_board_number_after_60_minutes_l2976_297630

/-- Calculates the product of digits of a natural number -/
def productOfDigits (n : ℕ) : ℕ := sorry

/-- Applies the transformation rule to a number -/
def transform (n : ℕ) : ℕ := productOfDigits n + 12

/-- Applies the transformation n times to the initial number -/
def applyNTimes (initial : ℕ) (n : ℕ) : ℕ := sorry

theorem board_number_after_60_minutes :
  applyNTimes 27 60 = 14 := by sorry

end NUMINAMATH_CALUDE_board_number_after_60_minutes_l2976_297630


namespace NUMINAMATH_CALUDE_z_sixth_power_l2976_297692

theorem z_sixth_power (z : ℂ) : 
  z = (Real.sqrt 3 + Complex.I) / 2 → 
  z^6 = (1 + Real.sqrt 3) / 4 - ((Real.sqrt 3 + 1) / 8) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_z_sixth_power_l2976_297692


namespace NUMINAMATH_CALUDE_product_of_roots_l2976_297647

theorem product_of_roots (a b c : ℂ) : 
  (3 * a^3 - 9 * a^2 + a - 7 = 0) ∧ 
  (3 * b^3 - 9 * b^2 + b - 7 = 0) ∧ 
  (3 * c^3 - 9 * c^2 + c - 7 = 0) →
  a * b * c = 7/3 := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_l2976_297647


namespace NUMINAMATH_CALUDE_right_triangle_from_conditions_l2976_297634

/-- Given a triangle ABC with side lengths a, b, and c satisfying certain conditions,
    prove that it is a right triangle. -/
theorem right_triangle_from_conditions (a b c : ℝ) (h1 : a + c = 2 * b) (h2 : c - a = 1 / 2 * b) :
  c^2 = a^2 + b^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_from_conditions_l2976_297634


namespace NUMINAMATH_CALUDE_train_passing_jogger_train_passes_jogger_in_39_seconds_l2976_297675

/-- The time taken for a train to pass a jogger -/
theorem train_passing_jogger (jogger_speed : ℝ) (train_speed : ℝ) 
  (train_length : ℝ) (initial_distance : ℝ) : ℝ :=
  let jogger_speed_ms := jogger_speed * 1000 / 3600
  let train_speed_ms := train_speed * 1000 / 3600
  let relative_speed := train_speed_ms - jogger_speed_ms
  let total_distance := initial_distance + train_length
  total_distance / relative_speed

/-- Proof that the train passes the jogger in 39 seconds -/
theorem train_passes_jogger_in_39_seconds : 
  train_passing_jogger 9 45 120 270 = 39 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_jogger_train_passes_jogger_in_39_seconds_l2976_297675


namespace NUMINAMATH_CALUDE_marathon_distance_theorem_l2976_297665

/-- Represents the length of a marathon in miles and yards. -/
structure MarathonLength where
  miles : ℕ
  yards : ℕ

/-- Calculates the total distance in miles and yards after running multiple marathons. -/
def totalDistance (marathonLength : MarathonLength) (numMarathons : ℕ) : MarathonLength :=
  let totalMiles := marathonLength.miles * numMarathons
  let totalYards := marathonLength.yards * numMarathons
  let extraMiles := totalYards / 1760
  let remainingYards := totalYards % 1760
  { miles := totalMiles + extraMiles, yards := remainingYards }

theorem marathon_distance_theorem :
  let marathonLength : MarathonLength := { miles := 26, yards := 385 }
  let numMarathons : ℕ := 15
  let result := totalDistance marathonLength numMarathons
  result.miles = 393 ∧ result.yards = 495 := by
  sorry

end NUMINAMATH_CALUDE_marathon_distance_theorem_l2976_297665


namespace NUMINAMATH_CALUDE_x_value_theorem_l2976_297655

theorem x_value_theorem (x y : ℝ) (h : x / (x - 2) = (y^2 + 3*y - 2) / (y^2 + 3*y + 1)) :
  x = 2*y^2 + 6*y + 4 := by
  sorry

end NUMINAMATH_CALUDE_x_value_theorem_l2976_297655


namespace NUMINAMATH_CALUDE_right_triangle_area_l2976_297661

theorem right_triangle_area (a b c : ℕ) : 
  a = 7 →                  -- One leg is 7
  a * a + b * b = c * c →  -- Pythagorean theorem (right triangle)
  a * b = 168 →            -- Area is 84 (2 * 84 = 168)
  (∃ (S : ℕ), S = 84 ∧ S = a * b / 2) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2976_297661


namespace NUMINAMATH_CALUDE_least_number_with_remainder_l2976_297673

theorem least_number_with_remainder (n : ℕ) : n = 125 →
  (∃ k : ℕ, n = 20 * k + 5) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, m = 20 * k + 5)) :=
by sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_l2976_297673


namespace NUMINAMATH_CALUDE_extremum_condition_l2976_297616

open Real

-- Define a differentiable function
variable (f : ℝ → ℝ) (hf : Differentiable ℝ f)

-- Define the concept of an extremum
def HasExtremumAt (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∀ x, f x ≤ f x₀ ∨ ∀ x, f x ≥ f x₀

-- Theorem statement
theorem extremum_condition (x₀ : ℝ) :
  (HasExtremumAt f x₀ → deriv f x₀ = 0) ∧
  ∃ g : ℝ → ℝ, Differentiable ℝ g ∧ deriv g 0 = 0 ∧ ¬HasExtremumAt g 0 := by
  sorry

end NUMINAMATH_CALUDE_extremum_condition_l2976_297616


namespace NUMINAMATH_CALUDE_bananas_undetermined_l2976_297609

/-- Represents Philip's fruit collection -/
structure FruitCollection where
  totalOranges : ℕ
  orangeGroups : ℕ
  orangesPerGroup : ℕ
  bananaGroups : ℕ

/-- Philip's actual fruit collection -/
def philipsCollection : FruitCollection := {
  totalOranges := 384,
  orangeGroups := 16,
  orangesPerGroup := 24,
  bananaGroups := 345
}

/-- Predicate to check if the number of bananas can be determined -/
def canDetermineBananas (c : FruitCollection) : Prop :=
  ∃ (bananasPerGroup : ℕ), True  -- Placeholder, always true

/-- Theorem stating that the number of bananas cannot be determined -/
theorem bananas_undetermined (c : FruitCollection) 
  (h1 : c.totalOranges = c.orangeGroups * c.orangesPerGroup) :
  ¬ canDetermineBananas c := by
  sorry

#check bananas_undetermined philipsCollection

end NUMINAMATH_CALUDE_bananas_undetermined_l2976_297609


namespace NUMINAMATH_CALUDE_amys_birthday_money_l2976_297684

theorem amys_birthday_money (initial : ℕ) (chore_money : ℕ) (final_total : ℕ) : 
  initial = 2 → chore_money = 13 → final_total = 18 → 
  final_total - (initial + chore_money) = 3 := by
  sorry

end NUMINAMATH_CALUDE_amys_birthday_money_l2976_297684


namespace NUMINAMATH_CALUDE_equivalent_operations_l2976_297643

theorem equivalent_operations (x : ℚ) : 
  (x * (4/5)) / (4/7) = x * (7/5) := by
  sorry

end NUMINAMATH_CALUDE_equivalent_operations_l2976_297643


namespace NUMINAMATH_CALUDE_polynomial_equality_l2976_297681

theorem polynomial_equality (a a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2 * x + 1) ^ 4 = a + a₁ * x + a₂ * x ^ 2 + a₃ * x ^ 3 + a₄ * x ^ 4) →
  a - a₁ + a₂ - a₃ + a₄ = 1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_equality_l2976_297681


namespace NUMINAMATH_CALUDE_ken_steak_purchase_l2976_297641

/-- The cost of one pound of steak, given the conditions of Ken's purchase -/
def steak_cost (total_pounds : ℕ) (paid : ℕ) (change : ℕ) : ℕ :=
  (paid - change) / total_pounds

theorem ken_steak_purchase :
  steak_cost 2 20 6 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ken_steak_purchase_l2976_297641


namespace NUMINAMATH_CALUDE_largest_integer_in_interval_l2976_297671

theorem largest_integer_in_interval : 
  ∃ (x : ℤ), (1/4 : ℚ) < (x : ℚ)/6 ∧ (x : ℚ)/6 < 7/9 ∧ 
  ∀ (y : ℤ), ((1/4 : ℚ) < (y : ℚ)/6 ∧ (y : ℚ)/6 < 7/9) → y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_in_interval_l2976_297671


namespace NUMINAMATH_CALUDE_x_not_negative_one_l2976_297667

theorem x_not_negative_one (x : ℝ) (h : (x + 1)^0 = 1) : x ≠ -1 := by
  sorry

end NUMINAMATH_CALUDE_x_not_negative_one_l2976_297667


namespace NUMINAMATH_CALUDE_min_sum_squared_eccentricities_l2976_297619

/-- Given an ellipse and a hyperbola sharing the same foci, with one of their
    intersection points P forming an angle ∠F₁PF₂ = 60°, and their respective
    eccentricities e₁ and e₂, the minimum value of e₁² + e₂² is 1 + √3/2. -/
theorem min_sum_squared_eccentricities (e₁ e₂ : ℝ) 
  (h_ellipse : e₁ ∈ Set.Ioo 0 1)
  (h_hyperbola : e₂ > 1)
  (h_shared_foci : True)  -- Represents the condition that the ellipse and hyperbola share foci
  (h_intersection : True)  -- Represents the condition that P is an intersection point
  (h_angle : True)  -- Represents the condition that ∠F₁PF₂ = 60°
  : (∀ ε₁ ε₂, ε₁ ∈ Set.Ioo 0 1 → ε₂ > 1 → ε₁^2 + ε₂^2 ≥ 1 + Real.sqrt 3 / 2) ∧ 
    (∃ ε₁ ε₂, ε₁ ∈ Set.Ioo 0 1 ∧ ε₂ > 1 ∧ ε₁^2 + ε₂^2 = 1 + Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squared_eccentricities_l2976_297619


namespace NUMINAMATH_CALUDE_pyramid_volume_in_cylinder_l2976_297672

/-- Given a cylinder of volume V and a pyramid inscribed in it such that:
    - The base of the pyramid is an isosceles triangle with angle α between equal sides
    - The pyramid's base is inscribed in the base of the cylinder
    - The pyramid's apex coincides with the midpoint of one of the cylinder's generatrices
    Then the volume of the pyramid is (V / (6π)) * sin(α) * cos²(α/2) -/
theorem pyramid_volume_in_cylinder (V : ℝ) (α : ℝ) : ℝ := by
  sorry

end NUMINAMATH_CALUDE_pyramid_volume_in_cylinder_l2976_297672


namespace NUMINAMATH_CALUDE_no_relationship_between_mites_and_wilt_resistance_l2976_297670

def total_plants : ℕ := 88
def infected_plants : ℕ := 33
def resistant_infected : ℕ := 19
def susceptible_infected : ℕ := 14
def not_infected_plants : ℕ := 55
def resistant_not_infected : ℕ := 28
def susceptible_not_infected : ℕ := 27

def chi_square (n a b c d : ℕ) : ℚ :=
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

def critical_value : ℚ := 3841 / 1000

theorem no_relationship_between_mites_and_wilt_resistance :
  chi_square total_plants resistant_infected resistant_not_infected 
             susceptible_infected susceptible_not_infected < critical_value := by
  sorry

end NUMINAMATH_CALUDE_no_relationship_between_mites_and_wilt_resistance_l2976_297670


namespace NUMINAMATH_CALUDE_larger_number_problem_l2976_297669

theorem larger_number_problem (L S : ℕ) (hL : L > S) : 
  L - S = 1365 → L = 6 * S + 15 → L = 1635 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l2976_297669


namespace NUMINAMATH_CALUDE_set_intersection_example_l2976_297685

theorem set_intersection_example : 
  let A : Set ℕ := {1, 2, 3, 4}
  let B : Set ℕ := {2, 4, 6}
  A ∩ B = {2, 4} := by
sorry

end NUMINAMATH_CALUDE_set_intersection_example_l2976_297685


namespace NUMINAMATH_CALUDE_cost_per_book_l2976_297666

def total_books : ℕ := 14
def total_spent : ℕ := 224

theorem cost_per_book : total_spent / total_books = 16 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_book_l2976_297666


namespace NUMINAMATH_CALUDE_parabola_coefficient_sum_l2976_297674

/-- A parabola with equation x = ay² + by + c, vertex at (3, -6), and passing through (2, -4) -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  vertex_condition : 3 = a * (-6)^2 + b * (-6) + c
  point_condition : 2 = a * (-4)^2 + b * (-4) + c

/-- The sum of coefficients a, b, and c for the given parabola is -25/4 -/
theorem parabola_coefficient_sum (p : Parabola) : p.a + p.b + p.c = -25/4 := by
  sorry

#check parabola_coefficient_sum

end NUMINAMATH_CALUDE_parabola_coefficient_sum_l2976_297674


namespace NUMINAMATH_CALUDE_optimal_pricing_strategy_l2976_297679

/-- Represents the pricing strategy of a merchant -/
structure MerchantPricing where
  list_price : ℝ
  purchase_discount : ℝ
  marked_price : ℝ
  selling_discount : ℝ
  profit_margin : ℝ

/-- Calculates the purchase price based on the list price and purchase discount -/
def purchase_price (m : MerchantPricing) : ℝ :=
  m.list_price * (1 - m.purchase_discount)

/-- Calculates the selling price based on the marked price and selling discount -/
def selling_price (m : MerchantPricing) : ℝ :=
  m.marked_price * (1 - m.selling_discount)

/-- Calculates the profit based on the selling price and purchase price -/
def profit (m : MerchantPricing) : ℝ :=
  selling_price m - purchase_price m

/-- Theorem stating the optimal marked price for the merchant's pricing strategy -/
theorem optimal_pricing_strategy (m : MerchantPricing) 
  (h1 : m.purchase_discount = 0.3)
  (h2 : m.selling_discount = 0.2)
  (h3 : m.profit_margin = 0.3)
  (h4 : profit m = m.profit_margin * selling_price m) :
  m.marked_price = 1.25 * m.list_price := by
  sorry


end NUMINAMATH_CALUDE_optimal_pricing_strategy_l2976_297679


namespace NUMINAMATH_CALUDE_smallest_number_in_specific_set_l2976_297605

theorem smallest_number_in_specific_set (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Three positive integers
  (a + b + c) / 3 = 30 →   -- Arithmetic mean is 30
  (min (max a b) (max b c)) = 31 →  -- Median is 31
  max a (max b c) = 31 + 8 →  -- Largest number is 8 more than median
  min a (min b c) = 20 := by  -- Smallest number is 20
sorry

end NUMINAMATH_CALUDE_smallest_number_in_specific_set_l2976_297605


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_not_necessary_condition_l2976_297688

def is_arithmetic_sequence (a b c d : ℝ) : Prop :=
  b - a = c - b ∧ c - b = d - c

theorem arithmetic_sequence_sum_property :
  ∀ (a b c d : ℝ), is_arithmetic_sequence a b c d → a + d = b + c :=
sorry

theorem not_necessary_condition :
  ∃ (a b c d : ℝ), a + d = b + c ∧ ¬(is_arithmetic_sequence a b c d) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_not_necessary_condition_l2976_297688


namespace NUMINAMATH_CALUDE_second_number_problem_l2976_297606

theorem second_number_problem (x y : ℤ) : 
  y = 2 * x - 3 → 
  x + y = 57 → 
  y = 37 := by
sorry

end NUMINAMATH_CALUDE_second_number_problem_l2976_297606


namespace NUMINAMATH_CALUDE_largest_divisible_by_all_less_than_cube_root_l2976_297664

theorem largest_divisible_by_all_less_than_cube_root : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (k : ℕ), k > 0 ∧ k < n^(1/3) → n % k = 0) ∧
  (∀ (m : ℕ), m > n → ∃ (j : ℕ), j > 0 ∧ j < m^(1/3) ∧ m % j ≠ 0) ∧
  n = 420 := by
sorry

end NUMINAMATH_CALUDE_largest_divisible_by_all_less_than_cube_root_l2976_297664


namespace NUMINAMATH_CALUDE_parallel_line_to_hyperbola_asymptote_l2976_297683

/-- Given a hyperbola x²/16 - y²/9 = 1 and a line y = kx - 1 parallel to one of its asymptotes, 
    prove that k = 3/4 -/
theorem parallel_line_to_hyperbola_asymptote (k : ℝ) 
  (h1 : k > 0)
  (h2 : ∃ (x y : ℝ), y = k * x - 1 ∧ (x^2 / 16 - y^2 / 9 = 1) ∧ 
        (∀ (x' y' : ℝ), x'^2 / 16 - y'^2 / 9 = 1 → 
          (y - y') / (x - x') = k ∨ (y - y') / (x - x') = -k)) : 
  k = 3/4 := by sorry

end NUMINAMATH_CALUDE_parallel_line_to_hyperbola_asymptote_l2976_297683


namespace NUMINAMATH_CALUDE_black_region_area_l2976_297698

theorem black_region_area (large_square_side : ℝ) (small_square_side : ℝ) :
  large_square_side = 10 →
  small_square_side = 4 →
  (large_square_side ^ 2) - 2 * (small_square_side ^ 2) = 68 := by
  sorry

end NUMINAMATH_CALUDE_black_region_area_l2976_297698


namespace NUMINAMATH_CALUDE_circle_area_ratio_l2976_297625

/-- Given two circles X and Y, where an arc of 60° on X has the same length as an arc of 20° on Y,
    the ratio of the area of X to the area of Y is 9. -/
theorem circle_area_ratio (X Y : Real) (hX : X > 0) (hY : Y > 0) 
  (h : X * (60 / 360) = Y * (20 / 360)) : 
  (X^2 * Real.pi) / (Y^2 * Real.pi) = 9 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l2976_297625


namespace NUMINAMATH_CALUDE_integral_of_constant_one_equals_one_l2976_297678

-- Define the constant function f(x) = 1
def f : ℝ → ℝ := λ x => 1

-- State the theorem
theorem integral_of_constant_one_equals_one :
  ∫ x in (0:ℝ)..1, f x = 1 := by sorry

end NUMINAMATH_CALUDE_integral_of_constant_one_equals_one_l2976_297678


namespace NUMINAMATH_CALUDE_quadratic_roots_l2976_297680

theorem quadratic_roots (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : 2 * a^2 + a * a + b = 0 ∧ 2 * b^2 + a * b + b = 0) : 
  a = 1/2 ∧ b = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_l2976_297680


namespace NUMINAMATH_CALUDE_robotics_club_theorem_l2976_297676

theorem robotics_club_theorem (total : ℕ) (cs : ℕ) (eng : ℕ) (both : ℕ) 
  (h1 : total = 120)
  (h2 : cs = 75)
  (h3 : eng = 50)
  (h4 : both = 10) :
  total - (cs + eng - both) = 5 := by
  sorry

end NUMINAMATH_CALUDE_robotics_club_theorem_l2976_297676


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l2976_297690

/-- Given a square with diagonal length 10√2 cm, its area is 100 cm². -/
theorem square_area_from_diagonal (diagonal : ℝ) (area : ℝ) : 
  diagonal = 10 * Real.sqrt 2 → area = diagonal ^ 2 / 2 → area = 100 := by sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l2976_297690


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_no_real_roots_iff_negative_discriminant_l2976_297645

theorem quadratic_no_real_roots :
  ∀ (x : ℝ), x^2 + x + 2 ≠ 0 :=
by
  sorry

-- Auxiliary definitions and theorems
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

theorem no_real_roots_iff_negative_discriminant (a b c : ℝ) (ha : a ≠ 0) :
  (∀ x : ℝ, a*x^2 + b*x + c ≠ 0) ↔ discriminant a b c < 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_no_real_roots_iff_negative_discriminant_l2976_297645


namespace NUMINAMATH_CALUDE_time_to_return_home_l2976_297626

/-- The time it takes Eric to go to the park -/
def time_to_park : ℕ := 20 + 10

/-- The factor by which the return trip is longer than the trip to the park -/
def return_factor : ℕ := 3

/-- Theorem: The time it takes Eric to return home is 90 minutes -/
theorem time_to_return_home : time_to_park * return_factor = 90 := by
  sorry

end NUMINAMATH_CALUDE_time_to_return_home_l2976_297626


namespace NUMINAMATH_CALUDE_emmett_jumping_jacks_l2976_297603

/-- The number of jumping jacks Emmett did -/
def jumping_jacks : ℕ := sorry

/-- The number of pushups Emmett did -/
def pushups : ℕ := 8

/-- The number of situps Emmett did -/
def situps : ℕ := 20

/-- The total number of exercises Emmett did -/
def total_exercises : ℕ := jumping_jacks + pushups + situps

/-- The percentage of exercises that were pushups -/
def pushup_percentage : ℚ := 1/5

theorem emmett_jumping_jacks : 
  jumping_jacks = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_emmett_jumping_jacks_l2976_297603


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l2976_297699

/-- Given a geometric series with first term a and common ratio r -/
def geometric_series (a r : ℝ) : ℕ → ℝ := fun n => a * r^n

/-- Sum of the geometric series up to infinity -/
def series_sum (a r : ℝ) : ℝ := 24

/-- Sum of terms with odd powers of r -/
def odd_powers_sum (a r : ℝ) : ℝ := 9

/-- Theorem: If the sum of a geometric series is 24 and the sum of terms with odd powers of r is 9, then r = 3/5 -/
theorem geometric_series_ratio (a r : ℝ) (h1 : series_sum a r = 24) (h2 : odd_powers_sum a r = 9) :
  r = 3/5 := by sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l2976_297699


namespace NUMINAMATH_CALUDE_newton_method_convergence_l2976_297677

noncomputable def newtonSequence : ℕ → ℝ
  | 0 => 2
  | n + 1 => (newtonSequence n ^ 2 + 2) / (2 * newtonSequence n)

theorem newton_method_convergence :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |newtonSequence n - Real.sqrt 2| < ε :=
by sorry

end NUMINAMATH_CALUDE_newton_method_convergence_l2976_297677


namespace NUMINAMATH_CALUDE_dinner_task_assignments_l2976_297652

theorem dinner_task_assignments (n : ℕ) (h : n = 5) : 
  (n.choose 2) * ((n - 2).choose 1) = 30 := by
  sorry

end NUMINAMATH_CALUDE_dinner_task_assignments_l2976_297652


namespace NUMINAMATH_CALUDE_inequality_proof_l2976_297694

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) :
  2 * x + 1 / (x^2 - 2*x*y + y^2) ≥ 2 * y + 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2976_297694


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2976_297608

theorem simplify_and_evaluate (x y : ℚ) (hx : x = 1/2) (hy : y = -3) :
  (x - 2*y)^2 - (x + y)*(x - y) - 5*y^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2976_297608


namespace NUMINAMATH_CALUDE_at_least_one_red_certain_l2976_297682

/-- Represents the number of red balls in the pocket -/
def num_red_balls : ℕ := 2

/-- Represents the number of white balls in the pocket -/
def num_white_balls : ℕ := 1

/-- Represents the total number of balls in the pocket -/
def total_balls : ℕ := num_red_balls + num_white_balls

/-- Represents the number of balls drawn from the pocket -/
def num_drawn : ℕ := 2

/-- Theorem stating that drawing at least one red ball when drawing 2 balls
    from a pocket containing 2 red balls and 1 white ball is a certain event -/
theorem at_least_one_red_certain :
  (num_red_balls.choose num_drawn + num_red_balls.choose (num_drawn - 1) * num_white_balls.choose 1) / total_balls.choose num_drawn = 1 :=
sorry

end NUMINAMATH_CALUDE_at_least_one_red_certain_l2976_297682


namespace NUMINAMATH_CALUDE_opposite_pairs_l2976_297646

theorem opposite_pairs : 
  ¬((-2 : ℝ) = -(1/2)) ∧ 
  ¬(|(-1)| = -1) ∧ 
  ¬(((-3)^2 : ℝ) = -(3^2)) ∧ 
  (-5 : ℝ) = -(-(-5)) := by sorry

end NUMINAMATH_CALUDE_opposite_pairs_l2976_297646


namespace NUMINAMATH_CALUDE_philatelist_stamps_problem_l2976_297610

theorem philatelist_stamps_problem :
  ∃! x : ℕ, x % 3 = 1 ∧ x % 5 = 3 ∧ x % 7 = 5 ∧ 150 < x ∧ x ≤ 300 ∧ x = 208 := by
  sorry

end NUMINAMATH_CALUDE_philatelist_stamps_problem_l2976_297610


namespace NUMINAMATH_CALUDE_fraction_equality_l2976_297631

theorem fraction_equality (x y b : ℝ) (hb : b ≠ 0) :
  x / b = y / b → x = y := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l2976_297631


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2976_297627

-- Define the repeating decimals
def repeating_six : ℚ := 2/3
def repeating_seven : ℚ := 7/9

-- State the theorem
theorem sum_of_repeating_decimals : 
  repeating_six + repeating_seven = 13/9 := by sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2976_297627


namespace NUMINAMATH_CALUDE_polynomial_expansion_l2976_297639

theorem polynomial_expansion :
  ∀ x : ℝ, (3 * x - 2) * (2 * x^2 + 4 * x - 6) = 6 * x^3 + 8 * x^2 - 26 * x + 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l2976_297639


namespace NUMINAMATH_CALUDE_land_conversion_equation_l2976_297620

/-- Represents the land conversion scenario in a village --/
theorem land_conversion_equation (x : ℝ) : True :=
  let original_forest : ℝ := 108
  let original_arable : ℝ := 54
  let conversion_percentage : ℝ := 0.2
  let new_forest : ℝ := original_forest + x
  let new_arable : ℝ := original_arable - x
  let equation := (new_arable = conversion_percentage * new_forest)
by
  sorry

end NUMINAMATH_CALUDE_land_conversion_equation_l2976_297620


namespace NUMINAMATH_CALUDE_divisibility_by_27_l2976_297601

theorem divisibility_by_27 (t : ℤ) : 
  27 ∣ (7 * (27 * t + 16)^4 + 19 * (27 * t + 16) + 25) := by
sorry

end NUMINAMATH_CALUDE_divisibility_by_27_l2976_297601


namespace NUMINAMATH_CALUDE_correct_mark_is_90_l2976_297604

/-- Proves that the correct mark is 90 given the problem conditions --/
theorem correct_mark_is_90 (n : ℕ) (initial_avg correct_avg wrong_mark : ℚ) :
  n = 10 →
  initial_avg = 100 →
  correct_avg = 96 →
  wrong_mark = 50 →
  ∃ x : ℚ, (n * initial_avg - wrong_mark + x) / n = correct_avg ∧ x = 90 :=
by sorry

end NUMINAMATH_CALUDE_correct_mark_is_90_l2976_297604


namespace NUMINAMATH_CALUDE_ball_bounce_distance_l2976_297695

/-- Calculates the total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (reboundFactor : ℝ) (bounces : ℕ) : ℝ :=
  let downDistances := List.range (bounces + 1) |>.map (fun i => initialHeight * reboundFactor^i)
  let upDistances := List.range bounces |>.map (fun i => initialHeight * reboundFactor^(i+1))
  (downDistances.sum + upDistances.sum)

/-- The total distance traveled by a ball dropped from 150 feet, rebounding 1/3 of its fall distance each time, after 5 bounces is equal to 298.14 feet -/
theorem ball_bounce_distance :
  totalDistance 150 (1/3) 5 = 298.14 := by
  sorry

end NUMINAMATH_CALUDE_ball_bounce_distance_l2976_297695


namespace NUMINAMATH_CALUDE_find_p_l2976_297635

theorem find_p : ∃ (d q : ℝ), ∀ (x : ℝ),
  (4 * x^2 - 2 * x + 5/2) * (d * x^2 + p * x + q) = 12 * x^4 - 7 * x^3 + 12 * x^2 - 15/2 * x + 10/2 →
  p = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_find_p_l2976_297635


namespace NUMINAMATH_CALUDE_sum_of_squares_geq_twice_product_l2976_297693

theorem sum_of_squares_geq_twice_product (a b : ℝ) : a^2 + b^2 ≥ 2*a*b := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_geq_twice_product_l2976_297693


namespace NUMINAMATH_CALUDE_parabola_focus_l2976_297658

/-- The focus of the parabola y^2 = 8x is at the point (2, 0) -/
theorem parabola_focus (x y : ℝ) : 
  (∀ x y, y^2 = 8*x ↔ (x - 2)^2 + y^2 = 4) := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_l2976_297658


namespace NUMINAMATH_CALUDE_smallest_non_odd_units_digit_l2976_297621

def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

def units_digit (n : ℕ) : ℕ := n % 10

def is_digit (d : ℕ) : Prop := d < 10

theorem smallest_non_odd_units_digit :
  ∀ d : ℕ, is_digit d →
    (∀ n : ℕ, is_odd n → units_digit n ≠ d) →
    (∀ e : ℕ, is_digit e → (∀ m : ℕ, is_odd m → units_digit m ≠ e) → d ≤ e) →
    d = 0 :=
sorry

end NUMINAMATH_CALUDE_smallest_non_odd_units_digit_l2976_297621


namespace NUMINAMATH_CALUDE_large_shoes_count_l2976_297615

/-- The number of pairs of large-size shoes initially stocked by the shop -/
def L : ℕ := sorry

/-- The number of pairs of medium-size shoes initially stocked by the shop -/
def medium_shoes : ℕ := 50

/-- The number of pairs of small-size shoes initially stocked by the shop -/
def small_shoes : ℕ := 24

/-- The number of pairs of shoes sold by the shop -/
def sold_shoes : ℕ := 83

/-- The number of pairs of shoes left after selling -/
def left_shoes : ℕ := 13

theorem large_shoes_count : L = 22 := by
  sorry

end NUMINAMATH_CALUDE_large_shoes_count_l2976_297615


namespace NUMINAMATH_CALUDE_woman_birth_year_l2976_297611

/-- A woman born in the latter half of the nineteenth century was y years old in the year y^2. -/
theorem woman_birth_year (y : ℕ) (h1 : 1850 ≤ y^2 - y) (h2 : y^2 - y < 1900) (h3 : y^2 = y + 1892) : 
  y^2 - y = 1892 := by
  sorry

end NUMINAMATH_CALUDE_woman_birth_year_l2976_297611


namespace NUMINAMATH_CALUDE_joan_cake_flour_l2976_297642

theorem joan_cake_flour (recipe_total : ℕ) (already_added : ℕ) (h1 : recipe_total = 7) (h2 : already_added = 3) :
  recipe_total - already_added = 4 := by
  sorry

end NUMINAMATH_CALUDE_joan_cake_flour_l2976_297642


namespace NUMINAMATH_CALUDE_derivative_of_y_l2976_297656

-- Define the function y
def y (x : ℝ) : ℝ := 2 * x^2 - 2 * x + 1

-- State the theorem
theorem derivative_of_y (x : ℝ) : 
  deriv y x = 4 * x - 2 := by sorry

end NUMINAMATH_CALUDE_derivative_of_y_l2976_297656


namespace NUMINAMATH_CALUDE_parabola_equation_l2976_297687

/-- A parabola with vertex at the origin, focus on the y-axis, and a point P(m, 1) on the parabola that is 5 units away from the focus has the standard equation x^2 = 16y. -/
theorem parabola_equation (m : ℝ) : 
  let p : ℝ → ℝ → Prop := λ x y => x^2 = 16*y  -- Standard equation of the parabola
  let focus : ℝ × ℝ := (0, 4)  -- Focus on y-axis, 4 units above origin
  let vertex : ℝ × ℝ := (0, 0)  -- Vertex at origin
  let point_on_parabola : ℝ × ℝ := (m, 1)  -- Given point on parabola
  (vertex = (0, 0)) →  -- Vertex condition
  (focus.1 = 0) →  -- Focus on y-axis condition
  ((point_on_parabola.1 - focus.1)^2 + (point_on_parabola.2 - focus.2)^2 = 5^2) →  -- Distance condition
  p point_on_parabola.1 point_on_parabola.2  -- Conclusion: point satisfies parabola equation
  := by sorry

end NUMINAMATH_CALUDE_parabola_equation_l2976_297687
