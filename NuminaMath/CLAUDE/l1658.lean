import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_three_consecutive_cubes_divisible_by_nine_l1658_165848

theorem sum_of_three_consecutive_cubes_divisible_by_nine (n : ℕ) :
  ∃ k : ℤ, (n^3 + (n+1)^3 + (n+2)^3 : ℤ) = 9 * k := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_consecutive_cubes_divisible_by_nine_l1658_165848


namespace NUMINAMATH_CALUDE_plane_equation_correct_l1658_165849

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space defined by parametric equations -/
structure Line3D where
  x : ℝ → ℝ
  y : ℝ → ℝ
  z : ℝ → ℝ

/-- A plane in 3D space defined by the equation Ax + By + Cz + D = 0 -/
structure Plane where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- Check if a point lies on a plane -/
def pointOnPlane (plane : Plane) (point : Point3D) : Prop :=
  plane.A * point.x + plane.B * point.y + plane.C * point.z + plane.D = 0

/-- Check if a line is contained in a plane -/
def lineInPlane (plane : Plane) (line : Line3D) : Prop :=
  ∀ t, plane.A * line.x t + plane.B * line.y t + plane.C * line.z t + plane.D = 0

/-- The given point (1,2,-3) -/
def givenPoint : Point3D := ⟨1, 2, -3⟩

/-- The given line (x - 2)/4 = (y + 3)/(-6) = (z - 4)/2 -/
def givenLine : Line3D :=
  ⟨λ t => 4*t + 2, λ t => -6*t - 3, λ t => 2*t + 4⟩

/-- The plane we want to prove is correct -/
def resultPlane : Plane := ⟨3, 1, -3, 2⟩

theorem plane_equation_correct :
  pointOnPlane resultPlane givenPoint ∧
  lineInPlane resultPlane givenLine ∧
  resultPlane.A > 0 ∧
  Nat.gcd (Nat.gcd (Int.natAbs resultPlane.A) (Int.natAbs resultPlane.B))
          (Nat.gcd (Int.natAbs resultPlane.C) (Int.natAbs resultPlane.D)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_plane_equation_correct_l1658_165849


namespace NUMINAMATH_CALUDE_dinner_lunch_ratio_l1658_165851

/-- Represents the amount of bread eaten at each meal in grams -/
structure BreadConsumption where
  breakfast : ℕ
  lunch : ℕ
  dinner : ℕ

/-- Proves that given the conditions, the ratio of dinner bread to lunch bread is 8:1 -/
theorem dinner_lunch_ratio (b : BreadConsumption) : 
  b.dinner = 240 ∧ 
  ∃ k : ℕ, b.dinner = k * b.lunch ∧ 
  b.dinner = 6 * b.breakfast ∧ 
  b.breakfast + b.lunch + b.dinner = 310 → 
  b.dinner / b.lunch = 8 := by
  sorry

end NUMINAMATH_CALUDE_dinner_lunch_ratio_l1658_165851


namespace NUMINAMATH_CALUDE_zacks_marbles_l1658_165868

theorem zacks_marbles (M : ℕ) : 
  (∃ k : ℕ, M = 3 * k + 5) → 
  (M - 60 = 5) → 
  M = 65 := by
sorry

end NUMINAMATH_CALUDE_zacks_marbles_l1658_165868


namespace NUMINAMATH_CALUDE_fraction_above_line_l1658_165837

/-- A square in the 2D plane --/
structure Square where
  bottomLeft : ℝ × ℝ
  topRight : ℝ × ℝ

/-- A line in the 2D plane defined by two points --/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Calculate the area of a square --/
def squareArea (s : Square) : ℝ :=
  let (x1, y1) := s.bottomLeft
  let (x2, y2) := s.topRight
  (x2 - x1) * (y2 - y1)

/-- Calculate the area of the part of the square above the line --/
def areaAboveLine (s : Square) (l : Line) : ℝ :=
  sorry  -- The actual calculation would go here

/-- The main theorem --/
theorem fraction_above_line (s : Square) (l : Line) : 
  s.bottomLeft = (4, 0) → 
  s.topRight = (9, 5) → 
  l.point1 = (4, 1) → 
  l.point2 = (9, 5) → 
  areaAboveLine s l / squareArea s = 9 / 10 := by
  sorry


end NUMINAMATH_CALUDE_fraction_above_line_l1658_165837


namespace NUMINAMATH_CALUDE_maci_pen_cost_l1658_165878

/-- The total cost of pens for Maci --/
def total_cost (blue_pens red_pens : ℕ) (blue_cost : ℚ) : ℚ :=
  (blue_pens : ℚ) * blue_cost + (red_pens : ℚ) * (2 * blue_cost)

/-- Theorem stating that Maci's total cost for pens is $4.00 --/
theorem maci_pen_cost :
  total_cost 10 15 (1/10) = 4 := by
  sorry

end NUMINAMATH_CALUDE_maci_pen_cost_l1658_165878


namespace NUMINAMATH_CALUDE_sum_of_xyz_is_718_l1658_165839

noncomputable def a : ℝ := -1 / Real.sqrt 3
noncomputable def b : ℝ := (3 + Real.sqrt 7) / 3

theorem sum_of_xyz_is_718 (ha : a^2 = 9/27) (hb : b^2 = (3 + Real.sqrt 7)^2 / 9)
  (ha_neg : a < 0) (hb_pos : b > 0)
  (h_expr : ∃ (x y z : ℕ+), (a + b)^3 = (x : ℝ) * Real.sqrt y / z) :
  ∃ (x y z : ℕ+), (a + b)^3 = (x : ℝ) * Real.sqrt y / z ∧ x + y + z = 718 :=
sorry

end NUMINAMATH_CALUDE_sum_of_xyz_is_718_l1658_165839


namespace NUMINAMATH_CALUDE_min_fraction_value_l1658_165862

theorem min_fraction_value (x y : ℝ) (hx : -5 ≤ x ∧ x ≤ -3) (hy : -3 ≤ y ∧ y ≤ 1) :
  (∀ x' y', -5 ≤ x' ∧ x' ≤ -3 → -3 ≤ y' ∧ y' ≤ 1 → (x' + y') / x' ≥ (x + y) / x) →
  (x + y) / x = 0.8 := by
sorry

end NUMINAMATH_CALUDE_min_fraction_value_l1658_165862


namespace NUMINAMATH_CALUDE_special_trapezoid_base_difference_l1658_165824

/-- A trapezoid with specific angle and side length properties -/
structure SpecialTrapezoid where
  /-- The measure of one angle at the larger base in degrees -/
  angle1 : ℝ
  /-- The measure of the other angle at the larger base in degrees -/
  angle2 : ℝ
  /-- The length of the shorter leg -/
  shorter_leg : ℝ
  /-- The length of the larger base -/
  larger_base : ℝ
  /-- The length of the shorter base -/
  shorter_base : ℝ
  /-- Condition: One angle at the larger base is 60° -/
  angle1_is_60 : angle1 = 60
  /-- Condition: The other angle at the larger base is 30° -/
  angle2_is_30 : angle2 = 30
  /-- Condition: The shorter leg is 5 units long -/
  shorter_leg_is_5 : shorter_leg = 5

/-- Theorem: The difference between the bases of the special trapezoid is 10 units -/
theorem special_trapezoid_base_difference (t : SpecialTrapezoid) :
  t.larger_base - t.shorter_base = 10 := by
  sorry


end NUMINAMATH_CALUDE_special_trapezoid_base_difference_l1658_165824


namespace NUMINAMATH_CALUDE_cube_sum_2001_l1658_165821

theorem cube_sum_2001 :
  ∀ a b c : ℕ+,
  a^3 + b^3 + c^3 = 2001 ↔ (a = 10 ∧ b = 10 ∧ c = 1) ∨ (a = 10 ∧ b = 1 ∧ c = 10) ∨ (a = 1 ∧ b = 10 ∧ c = 10) :=
by sorry

end NUMINAMATH_CALUDE_cube_sum_2001_l1658_165821


namespace NUMINAMATH_CALUDE_expression_evaluation_l1658_165806

theorem expression_evaluation (x y : ℕ) (hx : x = 3) (hy : y = 4) :
  5 * x^(y + 1) + 6 * y^(x + 1) + 2 * x * y = 2775 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1658_165806


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1658_165825

theorem inequality_system_solution (x : ℝ) :
  (2 * (x - 1) < x + 3) ∧ ((x + 1) / 3 - x < 3) → -4 < x ∧ x < 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1658_165825


namespace NUMINAMATH_CALUDE_freely_falling_body_time_l1658_165811

/-- The acceleration due to gravity in m/s² -/
def g : ℝ := 9.808

/-- The additional distance fallen in meters -/
def additional_distance : ℝ := 49.34

/-- The additional time of fall in seconds -/
def additional_time : ℝ := 1.3

/-- The initial time of fall in seconds -/
def initial_time : ℝ := 7.088

theorem freely_falling_body_time :
  g * (initial_time * additional_time + 0.5 * additional_time^2) = additional_distance := by
  sorry

end NUMINAMATH_CALUDE_freely_falling_body_time_l1658_165811


namespace NUMINAMATH_CALUDE_product_digits_sum_base7_l1658_165895

/-- Converts a base-7 number to base-10 --/
def toBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-7 --/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Computes the sum of digits of a number in base-7 --/
def sumOfDigitsBase7 (n : ℕ) : ℕ := sorry

/-- The main theorem --/
theorem product_digits_sum_base7 :
  let x := 35
  let y := 21
  sumOfDigitsBase7 (toBase7 (toBase10 x * toBase10 y)) = 15 := by sorry

end NUMINAMATH_CALUDE_product_digits_sum_base7_l1658_165895


namespace NUMINAMATH_CALUDE_solomon_collected_66_cans_l1658_165872

/-- The number of cans collected by Solomon, Juwan, and Levi -/
structure CanCollection where
  solomon : ℕ
  juwan : ℕ
  levi : ℕ

/-- The conditions of the can collection problem -/
def validCollection (c : CanCollection) : Prop :=
  c.solomon = 3 * c.juwan ∧
  c.levi = c.juwan / 2 ∧
  c.solomon + c.juwan + c.levi = 99

/-- Theorem stating that Solomon collected 66 cans -/
theorem solomon_collected_66_cans :
  ∃ (c : CanCollection), validCollection c ∧ c.solomon = 66 := by
  sorry

end NUMINAMATH_CALUDE_solomon_collected_66_cans_l1658_165872


namespace NUMINAMATH_CALUDE_nabla_problem_l1658_165822

-- Define the ∇ operation
def nabla (a b : ℕ) : ℕ := 3 + a^b

-- Theorem to prove
theorem nabla_problem : nabla (nabla 2 1) 4 = 628 := by
  sorry

end NUMINAMATH_CALUDE_nabla_problem_l1658_165822


namespace NUMINAMATH_CALUDE_correct_product_l1658_165853

theorem correct_product (a b c : ℚ) : 
  a = 0.125 → b = 3.2 → c = 4.0 → 
  (125 : ℚ) * 320 = 40000 → a * b = c := by
sorry

end NUMINAMATH_CALUDE_correct_product_l1658_165853


namespace NUMINAMATH_CALUDE_complement_A_inter_B_range_of_a_l1658_165896

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 3*x - 10 ≤ 0}
def B : Set ℝ := {x | 4 < x ∧ x < 6}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem for the complement of A ∩ B
theorem complement_A_inter_B :
  ∀ x : ℝ, x ∉ (A ∩ B) ↔ (x ≤ 4 ∨ x > 5) := by sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) :
  (A ∪ B) ⊆ C a → a ≥ 6 := by sorry

end NUMINAMATH_CALUDE_complement_A_inter_B_range_of_a_l1658_165896


namespace NUMINAMATH_CALUDE_largest_n_with_conditions_l1658_165826

theorem largest_n_with_conditions : 
  ∃ (m : ℤ), 139^2 = m^3 - 1 ∧ 
  ∃ (a : ℤ), 2 * 139 + 83 = a^2 ∧
  ∀ (n : ℤ), n > 139 → 
    (∀ (m : ℤ), n^2 ≠ m^3 - 1 ∨ ¬∃ (a : ℤ), 2 * n + 83 = a^2) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_with_conditions_l1658_165826


namespace NUMINAMATH_CALUDE_prob_three_diff_suits_probability_three_different_suits_l1658_165803

/-- A standard deck of cards -/
def StandardDeck : ℕ := 52

/-- The number of suits in a standard deck -/
def NumSuits : ℕ := 4

/-- The number of cards in each suit -/
def CardsPerSuit : ℕ := StandardDeck / NumSuits

/-- The probability of picking three cards of different suits from a standard deck without replacement -/
theorem prob_three_diff_suits : 
  (39 / 51) * (26 / 50) = 169 / 425 := by sorry

/-- The main theorem: probability of picking three cards of different suits -/
theorem probability_three_different_suits :
  let p := (CardsPerSuit * (NumSuits - 1) / (StandardDeck - 1)) * 
           (CardsPerSuit * (NumSuits - 2) / (StandardDeck - 2))
  p = 169 / 425 := by sorry

end NUMINAMATH_CALUDE_prob_three_diff_suits_probability_three_different_suits_l1658_165803


namespace NUMINAMATH_CALUDE_max_value_xy_over_z_l1658_165833

theorem max_value_xy_over_z (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 4 * x^2 - 3 * x * y + y^2 - z = 0) :
  ∃ (M : ℝ), M = 1 ∧ ∀ (w : ℝ), w = x * y / z → w ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_xy_over_z_l1658_165833


namespace NUMINAMATH_CALUDE_area_between_circles_and_x_axis_l1658_165841

/-- The area of the region bound by two circles and the x-axis -/
theorem area_between_circles_and_x_axis 
  (center_C : ℝ × ℝ) 
  (center_D : ℝ × ℝ) 
  (radius : ℝ) : 
  center_C = (3, 5) → 
  center_D = (13, 5) → 
  radius = 5 → 
  ∃ (area : ℝ), area = 50 - 25 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_area_between_circles_and_x_axis_l1658_165841


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1658_165888

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property 
  (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a) 
  (h_prod : a 7 * a 12 = 5) : 
  a 8 * a 9 * a 10 * a 11 = 25 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1658_165888


namespace NUMINAMATH_CALUDE_initial_piggy_bank_amount_l1658_165869

-- Define the variables
def weekly_allowance : ℕ := 10
def weeks : ℕ := 8
def final_amount : ℕ := 83

-- Define the function to calculate the amount added to the piggy bank
def amount_added (w : ℕ) : ℕ := w * (weekly_allowance / 2)

-- Theorem statement
theorem initial_piggy_bank_amount :
  ∃ (initial : ℕ), initial + amount_added weeks = final_amount :=
sorry

end NUMINAMATH_CALUDE_initial_piggy_bank_amount_l1658_165869


namespace NUMINAMATH_CALUDE_inscribed_circle_rectangle_area_l1658_165827

/-- Given a rectangle with an inscribed circle of radius 7 and a length-to-width ratio of 3:1,
    prove that the area of the rectangle is 588. -/
theorem inscribed_circle_rectangle_area :
  ∀ (length width radius : ℝ),
    radius = 7 →
    length = 3 * width →
    width = 2 * radius →
    length * width = 588 :=
by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_rectangle_area_l1658_165827


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1658_165808

/-- Proves that the speed of a boat in still water is 22 km/hr, given the conditions -/
theorem boat_speed_in_still_water :
  let stream_speed : ℝ := 5
  let downstream_distance : ℝ := 108
  let downstream_time : ℝ := 4
  let downstream_speed : ℝ := downstream_distance / downstream_time
  let boat_speed_still : ℝ := downstream_speed - stream_speed
  boat_speed_still = 22 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1658_165808


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l1658_165847

theorem simplify_and_rationalize (x : ℝ) : 
  1 / (2 + 1 / (Real.sqrt 5 + 2)) = Real.sqrt 5 / 5 := by sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l1658_165847


namespace NUMINAMATH_CALUDE_bryans_deposit_l1658_165835

theorem bryans_deposit (mark_deposit : ℕ) (bryan_deposit : ℕ) 
  (h1 : mark_deposit = 88)
  (h2 : bryan_deposit < 5 * mark_deposit)
  (h3 : mark_deposit + bryan_deposit = 400) :
  bryan_deposit = 312 := by
sorry

end NUMINAMATH_CALUDE_bryans_deposit_l1658_165835


namespace NUMINAMATH_CALUDE_student_calculation_error_l1658_165818

theorem student_calculation_error (x : ℝ) : 
  (8/7) * x = (4/5) * x + 15.75 → x = 45.9375 := by
  sorry

end NUMINAMATH_CALUDE_student_calculation_error_l1658_165818


namespace NUMINAMATH_CALUDE_length_of_cd_l1658_165854

/-- Given a line segment CD with points M and N on it, prove that CD has length 57.6 -/
theorem length_of_cd (C D M N : ℝ × ℝ) : 
  (∃ t : ℝ, M = (1 - t) • C + t • D ∧ 0 < t ∧ t < 1/2) →  -- M is on CD and same side of midpoint
  (∃ s : ℝ, N = (1 - s) • C + s • D ∧ 0 < s ∧ s < 1/2) →  -- N is on CD and same side of midpoint
  (dist C M) / (dist M D) = 3/5 →                         -- M divides CD in ratio 3:5
  (dist C N) / (dist N D) = 4/5 →                         -- N divides CD in ratio 4:5
  dist M N = 4 →                                          -- Length of MN is 4
  dist C D = 57.6 :=                                      -- Length of CD is 57.6
by sorry

end NUMINAMATH_CALUDE_length_of_cd_l1658_165854


namespace NUMINAMATH_CALUDE_train_length_l1658_165855

/-- The length of a train given its speed and time to pass a point --/
theorem train_length (speed : ℝ) (time : ℝ) (h1 : speed = 36) (h2 : time = 16) :
  speed * time * (5 / 18) = 160 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l1658_165855


namespace NUMINAMATH_CALUDE_symmetry_of_points_l1658_165867

/-- The line of symmetry --/
def line_of_symmetry (x y : ℝ) : Prop := x - y - 1 = 0

/-- Check if two points are symmetric with respect to a line --/
def is_symmetric (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  let midpoint_x := (x₁ + x₂) / 2
  let midpoint_y := (y₁ + y₂) / 2
  line_of_symmetry midpoint_x midpoint_y ∧
  (y₂ - y₁) / (x₂ - x₁) = -1

theorem symmetry_of_points :
  is_symmetric 2 2 3 1 :=
sorry

end NUMINAMATH_CALUDE_symmetry_of_points_l1658_165867


namespace NUMINAMATH_CALUDE_inequality_theorem_l1658_165831

theorem inequality_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / c) + (c / b) ≥ (4 * a) / (a + b) ∧
  ((a / c) + (c / b) = (4 * a) / (a + b) ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_l1658_165831


namespace NUMINAMATH_CALUDE_junk_mail_distribution_l1658_165897

/-- Given a block with houses and junk mail to distribute, calculate the number of pieces per house -/
def junk_mail_per_house (num_houses : ℕ) (total_junk_mail : ℕ) : ℕ :=
  total_junk_mail / num_houses

/-- Theorem: In a block with 20 houses and 640 pieces of junk mail, each house receives 32 pieces -/
theorem junk_mail_distribution :
  junk_mail_per_house 20 640 = 32 := by
  sorry

end NUMINAMATH_CALUDE_junk_mail_distribution_l1658_165897


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l1658_165804

theorem parallelogram_base_length 
  (area : ℝ) (height : ℝ) (base : ℝ) 
  (h1 : area = 576) 
  (h2 : height = 48) 
  (h3 : area = base * height) : 
  base = 12 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l1658_165804


namespace NUMINAMATH_CALUDE_unknown_number_proof_l1658_165885

theorem unknown_number_proof (x : ℝ) : 
  (10 + 30 + x) / 3 = (20 + 40 + 6) / 3 + 8 ↔ x = 50 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_proof_l1658_165885


namespace NUMINAMATH_CALUDE_case_A_case_B_case_C_case_D_l1658_165889

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the number of solutions for a triangle
inductive TriangleSolutions
  | Unique
  | Two
  | None

-- Function to determine the number of solutions for a triangle
def triangleSolutions (t : Triangle) : TriangleSolutions := sorry

-- Theorem for case A
theorem case_A :
  let t : Triangle := { a := 5, b := 7, c := 8, A := 0, B := 0, C := 0 }
  triangleSolutions t = TriangleSolutions.Unique := by sorry

-- Theorem for case B
theorem case_B :
  let t : Triangle := { a := 0, b := 18, c := 20, A := 0, B := 60 * π / 180, C := 0 }
  triangleSolutions t = TriangleSolutions.None := by sorry

-- Theorem for case C
theorem case_C :
  let t : Triangle := { a := 8, b := 8 * Real.sqrt 2, c := 0, A := 0, B := 45 * π / 180, C := 0 }
  triangleSolutions t = TriangleSolutions.Two := by sorry

-- Theorem for case D
theorem case_D :
  let t : Triangle := { a := 30, b := 25, c := 0, A := 150 * π / 180, B := 0, C := 0 }
  triangleSolutions t = TriangleSolutions.Unique := by sorry

end NUMINAMATH_CALUDE_case_A_case_B_case_C_case_D_l1658_165889


namespace NUMINAMATH_CALUDE_line_through_point_forming_triangle_l1658_165881

theorem line_through_point_forming_triangle : ∃ (a b : ℝ), 
  (∀ x y : ℝ, (x / a + y / b = 1) → ((-2) / a + 2 / b = 1)) ∧ 
  (1/2 * |a * b| = 1) ∧
  ((a = -1 ∧ b = -2) ∨ (a = 2 ∧ b = 1)) := by sorry

end NUMINAMATH_CALUDE_line_through_point_forming_triangle_l1658_165881


namespace NUMINAMATH_CALUDE_rectangle_area_relation_l1658_165879

/-- For a rectangle with area 12 and sides of length x and y, 
    the relationship between y and x is y = 12/x -/
theorem rectangle_area_relation (x y : ℝ) (h : x * y = 12) : y = 12 / x := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_relation_l1658_165879


namespace NUMINAMATH_CALUDE_sequence_ratio_l1658_165816

/-- Given a sequence a with sum S of its first n terms satisfying 3S_n - 6 = 2a_n,
    prove that S_5 / a_5 = 11/16 -/
theorem sequence_ratio (a : ℕ → ℚ) (S : ℕ → ℚ)
  (h : ∀ n, 3 * S n - 6 = 2 * a n) :
  S 5 / a 5 = 11 / 16 := by
  sorry

end NUMINAMATH_CALUDE_sequence_ratio_l1658_165816


namespace NUMINAMATH_CALUDE_Q_proper_subset_of_P_l1658_165814

def P : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def Q : Set ℕ := {2, 3, 5, 6}

theorem Q_proper_subset_of_P : Q ⊂ P := by
  sorry

end NUMINAMATH_CALUDE_Q_proper_subset_of_P_l1658_165814


namespace NUMINAMATH_CALUDE_shooting_probabilities_l1658_165873

-- Define the probabilities for each ring
def P_10 : ℝ := 0.24
def P_9 : ℝ := 0.28
def P_8 : ℝ := 0.19
def P_7 : ℝ := 0.16
def P_below_7 : ℝ := 0.13

-- Theorem for the three probability calculations
theorem shooting_probabilities :
  (P_10 + P_9 = 0.52) ∧
  (P_10 + P_9 + P_8 + P_7 = 0.87) ∧
  (P_7 + P_below_7 = 0.29) := by
  sorry

end NUMINAMATH_CALUDE_shooting_probabilities_l1658_165873


namespace NUMINAMATH_CALUDE_license_plate_count_l1658_165809

/-- The number of consonants available for the first character -/
def num_consonants : ℕ := 20

/-- The number of vowels available for the second and third characters -/
def num_vowels : ℕ := 6

/-- The number of digits and special symbols available for the fourth character -/
def num_digits_and_symbols : ℕ := 12

/-- The total number of possible license plates -/
def total_plates : ℕ := num_consonants * num_vowels * num_vowels * num_digits_and_symbols

theorem license_plate_count : total_plates = 103680 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l1658_165809


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l1658_165830

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) 
  (h2 : a^2 + b^2 + c^2 = 16) 
  (h3 : a*b + b*c + c*a = 9) 
  (h4 : a^2 + b^2 = 10) : 
  a + b + c = Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l1658_165830


namespace NUMINAMATH_CALUDE_ellipse_parameters_sum_l1658_165865

-- Define the foci
def F₁ : ℝ × ℝ := (1, 0)
def F₂ : ℝ × ℝ := (5, 0)

-- Define the constant sum of distances
def distance_sum : ℝ := 10

-- Define the ellipse
def is_on_ellipse (P : ℝ × ℝ) : Prop :=
  Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) +
  Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = distance_sum

-- Define the general form of the ellipse equation
def ellipse_equation (h k a b : ℝ) (P : ℝ × ℝ) : Prop :=
  (P.1 - h)^2 / a^2 + (P.2 - k)^2 / b^2 = 1

-- Theorem statement
theorem ellipse_parameters_sum :
  ∃ (h k a b : ℝ),
    (∀ P, is_on_ellipse P ↔ ellipse_equation h k a b P) ∧
    h + k + a + b = 8 + Real.sqrt 21 :=
sorry

end NUMINAMATH_CALUDE_ellipse_parameters_sum_l1658_165865


namespace NUMINAMATH_CALUDE_monster_family_eyes_total_l1658_165874

/-- The number of eyes in the extended monster family -/
def monster_family_eyes : ℕ :=
  let mom_eyes := 1
  let dad_eyes := 3
  let mom_dad_kids_eyes := 3 * 4
  let mom_previous_child_eyes := 5
  let dad_previous_children_eyes := 6 + 2
  let dad_ex_wife_eyes := 1
  let dad_ex_wife_partner_eyes := 7
  let dad_ex_wife_child_eyes := 8
  mom_eyes + dad_eyes + mom_dad_kids_eyes + mom_previous_child_eyes +
  dad_previous_children_eyes + dad_ex_wife_eyes + dad_ex_wife_partner_eyes +
  dad_ex_wife_child_eyes

/-- The total number of eyes in the extended monster family is 45 -/
theorem monster_family_eyes_total :
  monster_family_eyes = 45 := by sorry

end NUMINAMATH_CALUDE_monster_family_eyes_total_l1658_165874


namespace NUMINAMATH_CALUDE_lucy_age_is_12_l1658_165863

def sisters_ages : List Nat := [2, 4, 6, 10, 12, 14]

def movie_pair (ages : List Nat) : Prop :=
  ∃ a b, a ∈ ages ∧ b ∈ ages ∧ a ≠ b ∧ a + b = 20

def basketball_pair (ages : List Nat) : Prop :=
  ∃ a b, a ∈ ages ∧ b ∈ ages ∧ a ≠ b ∧ a ≤ 10 ∧ b ≤ 10

def staying_home (lucy_age : Nat) (ages : List Nat) : Prop :=
  lucy_age ∈ ages ∧ ∃ a, a ∈ ages ∧ a ≠ lucy_age

theorem lucy_age_is_12 :
  movie_pair sisters_ages →
  basketball_pair sisters_ages →
  ∃ lucy_age, staying_home lucy_age sisters_ages ∧ lucy_age = 12 :=
by sorry

end NUMINAMATH_CALUDE_lucy_age_is_12_l1658_165863


namespace NUMINAMATH_CALUDE_parallelogram_fourth_vertex_l1658_165882

def parallelogram (A B C D : ℂ) : Prop :=
  D - A = C - B

theorem parallelogram_fourth_vertex 
  (A B C D : ℂ) 
  (h1 : A = 1 + 3*I) 
  (h2 : B = -I) 
  (h3 : C = 2 + I) 
  (h4 : parallelogram A B C D) : 
  D = 3 + 5*I :=
sorry

end NUMINAMATH_CALUDE_parallelogram_fourth_vertex_l1658_165882


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1658_165861

-- Define the sets A and B
def A : Set ℝ := {x | 0 < x ∧ x < 2}
def B : Set ℝ := {x | -1 < x ∧ x < 1}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1658_165861


namespace NUMINAMATH_CALUDE_stove_repair_ratio_l1658_165871

theorem stove_repair_ratio :
  let stove_cost : ℚ := 1200
  let total_cost : ℚ := 1400
  let wall_cost : ℚ := total_cost - stove_cost
  (wall_cost / stove_cost) = 1 / 6 := by
sorry

end NUMINAMATH_CALUDE_stove_repair_ratio_l1658_165871


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l1658_165884

/-- For a circle with area 4π, the diameter is 4 -/
theorem circle_diameter_from_area : 
  ∀ (r : ℝ), r > 0 → π * r^2 = 4 * π → 2 * r = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l1658_165884


namespace NUMINAMATH_CALUDE_line_transformation_l1658_165812

/-- Given a line with equation y = -3/4x + 5, prove that a new line M with one-third the slope
and three times the y-intercept has the equation y = -1/4x + 15. -/
theorem line_transformation (x y : ℝ) :
  (y = -3/4 * x + 5) →
  ∃ (M : ℝ → ℝ),
    (∀ x, M x = -1/4 * x + 15) ∧
    (∀ x, M x = 1/3 * (-3/4) * x + 3 * 5) :=
by sorry

end NUMINAMATH_CALUDE_line_transformation_l1658_165812


namespace NUMINAMATH_CALUDE_sequence_property_l1658_165802

def sequence_sum (a : ℕ → ℝ) (n : ℕ) : ℝ := 2 * a n - a 1

theorem sequence_property (a : ℕ → ℝ) :
  (∀ n, sequence_sum a n = 2 * a n - a 1) →
  (2 * (a 2 + 1) = a 3 + a 1) →
  ∀ n, a n = 2^n :=
by sorry

end NUMINAMATH_CALUDE_sequence_property_l1658_165802


namespace NUMINAMATH_CALUDE_fraction_sum_bound_l1658_165843

theorem fraction_sum_bound (a b c : ℕ+) (h : (a : ℚ)⁻¹ + (b : ℚ)⁻¹ + (c : ℚ)⁻¹ < 1) :
  (a : ℚ)⁻¹ + (b : ℚ)⁻¹ + (c : ℚ)⁻¹ ≤ 41 / 42 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_bound_l1658_165843


namespace NUMINAMATH_CALUDE_multiplier_problem_l1658_165832

/-- Given a = 5, b = 30, and 40 ab = 1800, prove that the multiplier m such that m * a = 30 is equal to 6. -/
theorem multiplier_problem (a b : ℝ) (h1 : a = 5) (h2 : b = 30) (h3 : 40 * a * b = 1800) :
  ∃ m : ℝ, m * a = 30 ∧ m = 6 := by
sorry

end NUMINAMATH_CALUDE_multiplier_problem_l1658_165832


namespace NUMINAMATH_CALUDE_derivative_not_critical_point_l1658_165829

-- Define the function g(x) as the derivative of f(x)
def g (a : ℝ) (x : ℝ) : ℝ := (x - 1) * (x^2 - 3*x + a)

-- State the theorem
theorem derivative_not_critical_point (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, (deriv f) x = g a x) →  -- The derivative of f is g
  (deriv f) 1 ≠ 0 →             -- 1 is not a critical point
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_derivative_not_critical_point_l1658_165829


namespace NUMINAMATH_CALUDE_students_taking_both_languages_l1658_165859

theorem students_taking_both_languages (total : ℕ) (french : ℕ) (german : ℕ) (neither : ℕ) :
  total = 79 →
  french = 41 →
  german = 22 →
  neither = 25 →
  french + german - (total - neither) = 9 :=
by sorry

end NUMINAMATH_CALUDE_students_taking_both_languages_l1658_165859


namespace NUMINAMATH_CALUDE_equation_equivalence_l1658_165836

theorem equation_equivalence (p q : ℝ) 
  (hp_nonzero : p ≠ 0) (hp_not_five : p ≠ 5) 
  (hq_nonzero : q ≠ 0) (hq_not_seven : q ≠ 7) :
  (3 / p + 5 / q = 1 / 3) ↔ (p = 9 * q / (q - 15)) :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l1658_165836


namespace NUMINAMATH_CALUDE_correct_calculation_l1658_165890

theorem correct_calculation (a : ℝ) : (2*a)^2 / (4*a) = a := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1658_165890


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_z_purely_imaginary_iff_l1658_165846

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := m * (3 + Complex.I) - (2 + Complex.I)

-- Theorem 1: z is in the fourth quadrant when 2/3 < m < 1
theorem z_in_fourth_quadrant (m : ℝ) (h1 : 2/3 < m) (h2 : m < 1) :
  (z m).re > 0 ∧ (z m).im < 0 :=
sorry

-- Theorem 2: z is purely imaginary iff m = 2/3
theorem z_purely_imaginary_iff (m : ℝ) :
  (z m).re = 0 ∧ (z m).im ≠ 0 ↔ m = 2/3 :=
sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_z_purely_imaginary_iff_l1658_165846


namespace NUMINAMATH_CALUDE_right_triangle_height_l1658_165894

theorem right_triangle_height (a b c h : ℝ) : 
  a = 25 → b = 20 → c^2 = a^2 - b^2 → h * a = 2 * (1/2 * b * c) → h = 12 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_height_l1658_165894


namespace NUMINAMATH_CALUDE_amy_work_hours_school_year_l1658_165819

/-- Calculates the number of hours per week Amy must work during the school year
    to meet her financial goal, given her summer work schedule and earnings,
    and her school year work duration and earnings goal. -/
theorem amy_work_hours_school_year 
  (summer_weeks : ℕ) 
  (summer_hours_per_week : ℕ) 
  (summer_earnings : ℕ) 
  (school_year_weeks : ℕ) 
  (school_year_earnings_goal : ℕ) 
  (h1 : summer_weeks = 8)
  (h2 : summer_hours_per_week = 40)
  (h3 : summer_earnings = 3200)
  (h4 : school_year_weeks = 32)
  (h5 : school_year_earnings_goal = 4800) :
  (school_year_earnings_goal * summer_weeks * summer_hours_per_week) / 
  (summer_earnings * school_year_weeks) = 15 :=
by
  sorry

#check amy_work_hours_school_year

end NUMINAMATH_CALUDE_amy_work_hours_school_year_l1658_165819


namespace NUMINAMATH_CALUDE_train_speed_l1658_165840

/-- Proves that a train with given parameters has a specific speed -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (man_speed : ℝ) : 
  train_length = 300 →
  crossing_time = 9 →
  man_speed = 3 →
  ∃ (train_speed : ℝ), train_speed = 117 ∧ 
    (train_speed * 1000 / 3600 + man_speed * 1000 / 3600) * crossing_time = train_length :=
by sorry


end NUMINAMATH_CALUDE_train_speed_l1658_165840


namespace NUMINAMATH_CALUDE_solution_exists_in_interval_l1658_165893

-- Define the function f(x) = x^3 + x - 3
def f (x : ℝ) : ℝ := x^3 + x - 3

-- State the theorem
theorem solution_exists_in_interval :
  ∃! r : ℝ, r ∈ Set.Icc 1 2 ∧ f r = 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_exists_in_interval_l1658_165893


namespace NUMINAMATH_CALUDE_share_price_increase_l1658_165842

theorem share_price_increase (P : ℝ) (h1 : P > 0) : 
  let Q2 := 1.5 * P
  let Q1 := P * (1 + X / 100)
  X = 20 →
  Q2 = Q1 * 1.25 ∧ Q2 = 1.5 * P :=
by sorry

end NUMINAMATH_CALUDE_share_price_increase_l1658_165842


namespace NUMINAMATH_CALUDE_females_together_count_females_apart_count_l1658_165898

/-- The number of male students -/
def num_male : ℕ := 4

/-- The number of female students -/
def num_female : ℕ := 3

/-- The total number of students -/
def total_students : ℕ := num_male + num_female

/-- Calculates the number of arrangements where female students must stand together -/
def arrangements_females_together : ℕ :=
  (Nat.factorial num_female) * (Nat.factorial (num_male + 1))

/-- Calculates the number of arrangements where no two female students can stand next to each other -/
def arrangements_females_apart : ℕ :=
  (Nat.factorial num_male) * (Nat.choose (num_male + 1) num_female)

/-- Theorem stating the number of arrangements where female students must stand together -/
theorem females_together_count : arrangements_females_together = 720 := by
  sorry

/-- Theorem stating the number of arrangements where no two female students can stand next to each other -/
theorem females_apart_count : arrangements_females_apart = 1440 := by
  sorry

end NUMINAMATH_CALUDE_females_together_count_females_apart_count_l1658_165898


namespace NUMINAMATH_CALUDE_candidate_percentage_l1658_165813

theorem candidate_percentage (passing_mark total_mark : ℕ) 
  (h1 : passing_mark = 160)
  (h2 : total_mark = 300)
  (h3 : (60 : ℕ) * total_mark / 100 = passing_mark + 20)
  (h4 : passing_mark - 40 > 0) : 
  (passing_mark - 40) * 100 / total_mark = 40 := by
  sorry

end NUMINAMATH_CALUDE_candidate_percentage_l1658_165813


namespace NUMINAMATH_CALUDE_set_operations_l1658_165880

-- Define the sets A and B
def A : Set ℝ := {x | -2 < x ∧ x < 3}
def B : Set ℝ := {x | -3 < x ∧ x ≤ 3}

-- State the theorem
theorem set_operations :
  (Aᶜ : Set ℝ) = {x | x ≥ 3 ∨ x ≤ -2} ∧
  (A ∩ B : Set ℝ) = {x | -2 < x ∧ x < 3} ∧
  ((A ∩ B)ᶜ : Set ℝ) = {x | x ≥ 3 ∨ x ≤ -2} ∧
  (Aᶜ ∩ B : Set ℝ) = {x | (-3 < x ∧ x ≤ -2) ∨ x = 3} := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l1658_165880


namespace NUMINAMATH_CALUDE_first_perfect_square_all_remainders_l1658_165845

theorem first_perfect_square_all_remainders : 
  ∀ n : ℕ, n ≤ 20 → 
    (∃ k ≤ n, k^2 % 10 = 0) ∧ 
    (∃ k ≤ n, k^2 % 10 = 1) ∧ 
    (∃ k ≤ n, k^2 % 10 = 2) ∧ 
    (∃ k ≤ n, k^2 % 10 = 3) ∧ 
    (∃ k ≤ n, k^2 % 10 = 4) ∧ 
    (∃ k ≤ n, k^2 % 10 = 5) ∧ 
    (∃ k ≤ n, k^2 % 10 = 6) ∧ 
    (∃ k ≤ n, k^2 % 10 = 7) ∧ 
    (∃ k ≤ n, k^2 % 10 = 8) ∧ 
    (∃ k ≤ n, k^2 % 10 = 9) ↔ 
    n = 20 :=
by sorry

end NUMINAMATH_CALUDE_first_perfect_square_all_remainders_l1658_165845


namespace NUMINAMATH_CALUDE_last_digit_power_of_two_cycle_last_digit_2018th_power_of_two_l1658_165883

def last_digit (n : ℕ) : ℕ := n % 10

def power_of_two (n : ℕ) : ℕ := 2^n

theorem last_digit_power_of_two_cycle (n : ℕ) :
  last_digit (power_of_two n) = last_digit (power_of_two (n % 4)) :=
sorry

theorem last_digit_2018th_power_of_two :
  last_digit (power_of_two 2018) = 4 :=
sorry

end NUMINAMATH_CALUDE_last_digit_power_of_two_cycle_last_digit_2018th_power_of_two_l1658_165883


namespace NUMINAMATH_CALUDE_triangle_shape_l1658_165805

theorem triangle_shape (A B C : Real) (hABC : A + B + C = π) 
  (h : Real.sin A ^ 2 + Real.sin B ^ 2 < Real.sin C ^ 2) : 
  ∃ (a b c : Real), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    a ^ 2 + b ^ 2 - c ^ 2 < 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_shape_l1658_165805


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l1658_165886

theorem complex_number_quadrant (z : ℂ) (h : z / Complex.I = 2 - 3 * Complex.I) : 
  Complex.re z > 0 ∧ Complex.im z > 0 :=
sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l1658_165886


namespace NUMINAMATH_CALUDE_Q_subset_P_l1658_165899

-- Define the sets P and Q
def P : Set ℝ := {x : ℝ | x < 2}
def Q : Set ℝ := {x : ℝ | x^2 < 1}

-- Theorem statement
theorem Q_subset_P : Q ⊆ P := by sorry

end NUMINAMATH_CALUDE_Q_subset_P_l1658_165899


namespace NUMINAMATH_CALUDE_intersection_implies_sin_2α_l1658_165866

noncomputable section

-- Define the line l
def line_l (α : Real) (t : Real) : Real × Real :=
  (-1 + t * Real.cos α, -3 + t * Real.sin α)

-- Define the curve C
def curve_C (θ : Real) : Real × Real :=
  let ρ := 4 * Real.cos θ
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define the distance between two points
def distance (p1 p2 : Real × Real) : Real :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem intersection_implies_sin_2α (α : Real) :
  ∃ (t1 t2 θ1 θ2 : Real),
    let A := line_l α t1
    let B := line_l α t2
    curve_C θ1 = A ∧
    curve_C θ2 = B ∧
    distance A B = 2 →
    Real.sin (2 * α) = 2/3 := by
  sorry

end

end NUMINAMATH_CALUDE_intersection_implies_sin_2α_l1658_165866


namespace NUMINAMATH_CALUDE_inverse_proportion_points_order_l1658_165856

/-- Given points A(x₁, -6), B(x₂, -2), C(x₃, 3) on the graph of y = -12/x,
    prove that x₃ < x₁ < x₂ -/
theorem inverse_proportion_points_order (x₁ x₂ x₃ : ℝ) : 
  (-6 : ℝ) = -12 / x₁ → 
  (-2 : ℝ) = -12 / x₂ → 
  (3 : ℝ) = -12 / x₃ → 
  x₃ < x₁ ∧ x₁ < x₂ := by
  sorry

#check inverse_proportion_points_order

end NUMINAMATH_CALUDE_inverse_proportion_points_order_l1658_165856


namespace NUMINAMATH_CALUDE_min_area_line_correct_l1658_165820

/-- A line passing through a point (2, 1) and intersecting the positive x and y axes -/
structure MinAreaLine where
  /-- The slope of the line -/
  m : ℝ
  /-- The y-intercept of the line -/
  b : ℝ
  /-- The line passes through (2, 1) -/
  passes_through : m * 2 + b = 1
  /-- The line intersects the positive x-axis -/
  x_intercept_positive : -b / m > 0
  /-- The line intersects the positive y-axis -/
  y_intercept_positive : b > 0

/-- The equation of the line that minimizes the area of the triangle formed with the axes -/
def min_area_line_equation (l : MinAreaLine) : Prop :=
  l.m = -1/2 ∧ l.b = 2

theorem min_area_line_correct (l : MinAreaLine) :
  min_area_line_equation l ↔ l.m * 1 + l.b * 2 = 4 :=
sorry

end NUMINAMATH_CALUDE_min_area_line_correct_l1658_165820


namespace NUMINAMATH_CALUDE_juan_friends_seating_l1658_165870

theorem juan_friends_seating (n : ℕ) : n = 5 :=
  by
    -- Define the conditions
    have juan_fixed : True := True.intro
    have jamal_next_to_juan : ℕ := 2
    have total_arrangements : ℕ := 48

    -- State the relationship between n and the conditions
    have seating_equation : jamal_next_to_juan * Nat.factorial (n - 1) = total_arrangements := by sorry

    -- Prove that n = 5 satisfies the equation
    sorry

end NUMINAMATH_CALUDE_juan_friends_seating_l1658_165870


namespace NUMINAMATH_CALUDE_three_X_five_l1658_165834

/-- The operation X defined for real numbers -/
def X (a b : ℝ) : ℝ := b + 15 * a - 2 * a^2

/-- Theorem stating that 3X5 equals 32 -/
theorem three_X_five : X 3 5 = 32 := by
  sorry

end NUMINAMATH_CALUDE_three_X_five_l1658_165834


namespace NUMINAMATH_CALUDE_imaginary_unit_power_l1658_165876

theorem imaginary_unit_power (i : ℂ) : i^2 = -1 → i^607 = -i := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_power_l1658_165876


namespace NUMINAMATH_CALUDE_additive_inverses_and_quadratic_roots_l1658_165810

theorem additive_inverses_and_quadratic_roots :
  (∀ x y : ℝ, (∃ z : ℝ, x + z = 0 ∧ y + z = 0) → x + y = 0) ∧
  (∀ q : ℝ, (∀ x : ℝ, x^2 + x + q ≠ 0) → q > -1) := by
  sorry

end NUMINAMATH_CALUDE_additive_inverses_and_quadratic_roots_l1658_165810


namespace NUMINAMATH_CALUDE_area_of_R2_l1658_165800

/-- Rectangle R1 -/
structure Rectangle1 where
  side : ℝ
  area : ℝ

/-- Rectangle R2 -/
structure Rectangle2 where
  diagonal : ℝ

/-- Given conditions -/
def given_conditions : Prop :=
  ∃ (R1 : Rectangle1) (R2 : Rectangle2),
    R1.side = 4 ∧
    R1.area = 32 ∧
    R2.diagonal = 20 ∧
    -- Similarity condition (ratio of sides is the same)
    ∃ (k : ℝ), k > 0 ∧ R2.diagonal = k * (R1.side * (R1.area / R1.side).sqrt)

/-- Theorem: Area of R2 is 160 square inches -/
theorem area_of_R2 : given_conditions → ∃ (R2 : Rectangle2), R2.diagonal = 20 ∧ R2.diagonal^2 / 2 = 160 :=
sorry

end NUMINAMATH_CALUDE_area_of_R2_l1658_165800


namespace NUMINAMATH_CALUDE_helen_cookies_l1658_165857

/-- The number of chocolate chip cookies Helen baked yesterday -/
def yesterday_chocolate : ℕ := 19

/-- The number of chocolate chip cookies Helen baked this morning -/
def today_chocolate : ℕ := 237

/-- The difference between the number of chocolate chip cookies and raisin cookies Helen baked -/
def chocolate_raisin_diff : ℕ := 25

/-- The number of raisin cookies Helen baked -/
def raisin_cookies : ℕ := 231

theorem helen_cookies :
  raisin_cookies = (yesterday_chocolate + today_chocolate) - chocolate_raisin_diff := by
  sorry

end NUMINAMATH_CALUDE_helen_cookies_l1658_165857


namespace NUMINAMATH_CALUDE_y_satisfies_differential_equation_l1658_165838

noncomputable def y (x : ℝ) : ℝ := x / (x - 1) + x^2

theorem y_satisfies_differential_equation (x : ℝ) :
  x * (x - 1) * (deriv y x) + y x = x^2 * (2 * x - 1) :=
by sorry

end NUMINAMATH_CALUDE_y_satisfies_differential_equation_l1658_165838


namespace NUMINAMATH_CALUDE_system_solution_l1658_165852

theorem system_solution (x y : ℚ) : 
  (3 * x - 2 * y = 8) ∧ (x + 3 * y = 7) → x = 38 / 11 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1658_165852


namespace NUMINAMATH_CALUDE_system_solution_range_l1658_165850

theorem system_solution_range (x y k : ℝ) : 
  (2 * x + y = 2 * k - 1) → 
  (x + 2 * y = -4) → 
  (x + y > 1) → 
  (k > 4) := by
sorry

end NUMINAMATH_CALUDE_system_solution_range_l1658_165850


namespace NUMINAMATH_CALUDE_banana_count_l1658_165858

/-- The number of bananas in the fruit shop. -/
def bananas : ℕ := 30

/-- The number of apples in the fruit shop. -/
def apples : ℕ := 4 * bananas

/-- The number of persimmons in the fruit shop. -/
def persimmons : ℕ := 3 * bananas

/-- Theorem stating that the number of bananas is 30, given the conditions. -/
theorem banana_count : bananas = 30 := by
  have h1 : apples + persimmons = 210 := by sorry
  sorry

end NUMINAMATH_CALUDE_banana_count_l1658_165858


namespace NUMINAMATH_CALUDE_circle_condition_l1658_165828

/-- The equation of a potential circle with parameter m -/
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 + 4*x - 2*y + 5*m = 0

/-- Theorem stating the necessary and sufficient condition for the equation to represent a circle -/
theorem circle_condition (m : ℝ) :
  (∃ (x₀ y₀ r : ℝ), ∀ (x y : ℝ), circle_equation x y m ↔ (x - x₀)^2 + (y - y₀)^2 = r^2) ↔ m < 1 :=
sorry

end NUMINAMATH_CALUDE_circle_condition_l1658_165828


namespace NUMINAMATH_CALUDE_matching_pair_guarantee_l1658_165887

/-- The number of different colors of plates -/
def num_colors : ℕ := 5

/-- The total number of plates to be pulled out -/
def total_plates : ℕ := 6

/-- The minimum number of plates needed to guarantee a matching pair -/
def min_matching_pair : ℕ := total_plates

theorem matching_pair_guarantee :
  min_matching_pair = total_plates :=
sorry

end NUMINAMATH_CALUDE_matching_pair_guarantee_l1658_165887


namespace NUMINAMATH_CALUDE_cos_alpha_value_l1658_165864

theorem cos_alpha_value (α : Real) : 
  (∃ x y : Real, x ≤ 0 ∧ y = -4/3 * x ∧ 
   x = Real.cos α ∧ y = Real.sin α) → 
  Real.cos α = -3/5 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l1658_165864


namespace NUMINAMATH_CALUDE_tennis_ball_ratio_l1658_165875

theorem tennis_ball_ratio (total_ordered : ℕ) (extra_yellow : ℕ) : 
  total_ordered = 288 →
  extra_yellow = 90 →
  let white := total_ordered / 2
  let yellow := total_ordered / 2 + extra_yellow
  (white : ℚ) / yellow = 8 / 13 := by
sorry

end NUMINAMATH_CALUDE_tennis_ball_ratio_l1658_165875


namespace NUMINAMATH_CALUDE_cylinder_cross_section_area_l1658_165891

/-- Represents a cylinder with given dimensions and arc --/
structure Cylinder :=
  (radius : ℝ)
  (height : ℝ)
  (arc_angle : ℝ)

/-- Calculates the area of the cross-section of the cylinder --/
def cross_section_area (c : Cylinder) : ℝ := sorry

/-- Checks if a number is not divisible by the square of any prime --/
def not_divisible_by_square_prime (n : ℕ) : Prop := sorry

/-- Main theorem about the cross-section area of the specific cylinder --/
theorem cylinder_cross_section_area :
  let c := Cylinder.mk 7 10 (150 * π / 180)
  ∃ (d e : ℕ) (f : ℕ),
    cross_section_area c = d * π + e * Real.sqrt f ∧
    not_divisible_by_square_prime f ∧
    d = 60 ∧ e = 70 ∧ f = 3 := by sorry

end NUMINAMATH_CALUDE_cylinder_cross_section_area_l1658_165891


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l1658_165860

/-- A line passing through point (2, 1) with equal intercepts on the coordinate axes -/
structure EqualInterceptLine where
  /-- The slope of the line -/
  m : ℝ
  /-- The y-intercept of the line -/
  b : ℝ
  /-- The line passes through point (2, 1) -/
  point_condition : m * 2 + b = 1
  /-- The line has equal intercepts on the coordinate axes -/
  equal_intercepts : b = 0 ∨ m = -1

/-- The equation of a line with equal intercepts passing through (2, 1) is either 2x - y = 0 or x + y - 3 = 0 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (l.m = 1/2 ∧ l.b = 0) ∨ (l.m = -1 ∧ l.b = 3) :=
sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l1658_165860


namespace NUMINAMATH_CALUDE_union_complement_equals_set_l1658_165817

def U : Set Nat := {1, 2, 3, 4, 5}
def M : Set Nat := {1, 4}
def N : Set Nat := {2, 5}

theorem union_complement_equals_set : N ∪ (U \ M) = {2, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_union_complement_equals_set_l1658_165817


namespace NUMINAMATH_CALUDE_min_value_quadratic_form_l1658_165801

theorem min_value_quadratic_form (x y : ℝ) :
  2 * x^2 + 3 * x * y + 2 * y^2 ≥ 0 ∧
  (2 * x^2 + 3 * x * y + 2 * y^2 = 0 ↔ x = 0 ∧ y = 0) :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_form_l1658_165801


namespace NUMINAMATH_CALUDE_infinite_dividing_planes_l1658_165823

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  -- Add necessary fields here

/-- A plane that intersects a regular tetrahedron -/
structure IntersectingPlane where
  -- Add necessary fields here

/-- Predicate to check if a plane divides a tetrahedron into two equal parts -/
def divides_equally (t : RegularTetrahedron) (p : IntersectingPlane) : Prop :=
  sorry

/-- The set of planes that divide a regular tetrahedron into two equal parts -/
def dividing_planes (t : RegularTetrahedron) : Set IntersectingPlane :=
  {p : IntersectingPlane | divides_equally t p}

/-- Theorem stating that there are infinitely many planes that divide a regular tetrahedron equally -/
theorem infinite_dividing_planes (t : RegularTetrahedron) :
  Set.Infinite (dividing_planes t) :=
sorry

end NUMINAMATH_CALUDE_infinite_dividing_planes_l1658_165823


namespace NUMINAMATH_CALUDE_dragon_jewel_ratio_l1658_165807

theorem dragon_jewel_ratio :
  ∀ (initial_jewels : ℕ),
    initial_jewels - 3 + 6 = 24 →
    (6 : ℚ) / initial_jewels = 2 / 7 :=
by
  sorry

end NUMINAMATH_CALUDE_dragon_jewel_ratio_l1658_165807


namespace NUMINAMATH_CALUDE_even_factors_count_l1658_165892

/-- The number of even natural-number factors of 2^2 * 3^1 * 7^2 -/
def num_even_factors : ℕ := 12

/-- The prime factorization of n -/
def n : ℕ := 2^2 * 3^1 * 7^2

/-- A function that counts the number of even natural-number factors of n -/
def count_even_factors (n : ℕ) : ℕ := sorry

theorem even_factors_count :
  count_even_factors n = num_even_factors := by sorry

end NUMINAMATH_CALUDE_even_factors_count_l1658_165892


namespace NUMINAMATH_CALUDE_area_of_region_l1658_165844

-- Define the region
def region (x y : ℝ) : Prop :=
  |x - 2*y^2| + x + 2*y^2 ≤ 8 - 4*y

-- Define symmetry about Y-axis
def symmetricAboutYAxis (S : Set (ℝ × ℝ)) : Prop :=
  ∀ x y, (x, y) ∈ S ↔ (-x, y) ∈ S

-- Theorem statement
theorem area_of_region :
  ∃ S : Set (ℝ × ℝ),
    (∀ x y, (x, y) ∈ S ↔ region x y) ∧
    symmetricAboutYAxis S ∧
    MeasureTheory.volume S = 30 := by
  sorry

end NUMINAMATH_CALUDE_area_of_region_l1658_165844


namespace NUMINAMATH_CALUDE_triangle_side_lengths_l1658_165877

theorem triangle_side_lengths 
  (a b c : ℚ) 
  (perimeter : a + b + c = 24)
  (relation : a + 2 * b = 2 * c)
  (ratio : a = (1 / 2) * b) :
  a = 16 / 3 ∧ b = 32 / 3 ∧ c = 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_lengths_l1658_165877


namespace NUMINAMATH_CALUDE_last_two_digits_of_product_l1658_165815

theorem last_two_digits_of_product (n : ℕ) : 
  (33 * 92025^1989) % 100 = 25 := by
  sorry

#eval (33 * 92025^1989) % 100

end NUMINAMATH_CALUDE_last_two_digits_of_product_l1658_165815
