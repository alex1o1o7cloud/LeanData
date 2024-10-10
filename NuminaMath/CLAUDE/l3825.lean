import Mathlib

namespace garden_area_increase_l3825_382587

/-- Given a rectangular garden with length 60 feet and width 20 feet,
    prove that changing it to a square garden with the same perimeter
    increases the area by 400 square feet. -/
theorem garden_area_increase :
  let rectangular_length : ℝ := 60
  let rectangular_width : ℝ := 20
  let perimeter : ℝ := 2 * (rectangular_length + rectangular_width)
  let square_side : ℝ := perimeter / 4
  let rectangular_area : ℝ := rectangular_length * rectangular_width
  let square_area : ℝ := square_side * square_side
  square_area - rectangular_area = 400 :=
by sorry


end garden_area_increase_l3825_382587


namespace oldest_daughter_ages_l3825_382575

def is_valid_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 168

def has_ambiguous_sum (a b c : ℕ) : Prop :=
  ∃ (x y z : ℕ), is_valid_triple x y z ∧ 
  x + y + z = a + b + c ∧ (x ≠ a ∨ y ≠ b ∨ z ≠ c)

theorem oldest_daughter_ages :
  ∀ (a b c : ℕ), is_valid_triple a b c → has_ambiguous_sum a b c →
  (max a (max b c) = 12 ∨ max a (max b c) = 14 ∨ max a (max b c) = 21) :=
by sorry

end oldest_daughter_ages_l3825_382575


namespace certain_number_proof_l3825_382592

theorem certain_number_proof : ∃ x : ℝ, (0.60 * x = 0.50 * 600) ∧ (x = 500) := by
  sorry

end certain_number_proof_l3825_382592


namespace perfect_square_fraction_count_l3825_382507

theorem perfect_square_fraction_count : 
  ∃! (S : Finset ℤ), 
    (∀ n ∈ S, n ≠ 20 ∧ ∃ k : ℤ, (n : ℚ) / (20 - n) = k^2) ∧ 
    Finset.card S = 4 :=
by sorry

end perfect_square_fraction_count_l3825_382507


namespace min_value_expression_l3825_382584

theorem min_value_expression (x y z k : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hk : k > 0) :
  (k * 4 * z) / (2 * x + y) + (k * 4 * x) / (y + 2 * z) + (k * y) / (x + z) ≥ 3 * k :=
by sorry

end min_value_expression_l3825_382584


namespace opposite_of_2023_l3825_382591

-- Define the concept of opposite for integers
def opposite (n : ℤ) : ℤ := -n

-- Theorem statement
theorem opposite_of_2023 : opposite 2023 = -2023 := by
  sorry

end opposite_of_2023_l3825_382591


namespace right_triangle_third_side_product_l3825_382564

theorem right_triangle_third_side_product (a b c : ℝ) : 
  (a = 6 ∧ b = 8 ∧ a^2 + b^2 = c^2) ∨ (a = 6 ∧ c = 8 ∧ a^2 + b^2 = c^2) →
  c * b = 20 * Real.sqrt 7 := by
  sorry

end right_triangle_third_side_product_l3825_382564


namespace cube_triangle_areas_sum_l3825_382571

/-- Represents a 3D point in space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a triangle in 3D space -/
structure Triangle3D where
  a : Point3D
  b : Point3D
  c : Point3D

/-- The vertices of a 2x2x2 cube -/
def cubeVertices : List Point3D := [
  ⟨0, 0, 0⟩, ⟨0, 0, 2⟩, ⟨0, 2, 0⟩, ⟨0, 2, 2⟩,
  ⟨2, 0, 0⟩, ⟨2, 0, 2⟩, ⟨2, 2, 0⟩, ⟨2, 2, 2⟩
]

/-- All possible triangles formed by the vertices of the cube -/
def cubeTriangles : List Triangle3D := sorry

/-- Calculates the area of a triangle in 3D space -/
def triangleArea (t : Triangle3D) : ℝ := sorry

/-- The sum of areas of all triangles formed by the cube vertices -/
def totalArea : ℝ := (cubeTriangles.map triangleArea).sum

/-- The theorem to be proved -/
theorem cube_triangle_areas_sum :
  ∃ (m n p : ℕ), totalArea = m + Real.sqrt n + Real.sqrt p ∧ m + n + p = 7728 := by
  sorry

end cube_triangle_areas_sum_l3825_382571


namespace encryption_assignment_exists_l3825_382572

/-- Represents a user on the platform -/
structure User :=
  (id : Nat)

/-- Represents an encryption key -/
structure EncryptionKey :=
  (id : Nat)

/-- Represents a messaging channel between two users -/
structure Channel :=
  (user1 : User)
  (user2 : User)
  (key : EncryptionKey)

/-- The total number of users on the platform -/
def totalUsers : Nat := 105

/-- The total number of available encryption keys -/
def totalKeys : Nat := 100

/-- A function that assigns an encryption key to a channel between two users -/
def assignKey : User → User → EncryptionKey := sorry

/-- Theorem stating that there exists a key assignment satisfying the required property -/
theorem encryption_assignment_exists :
  ∃ (assignKey : User → User → EncryptionKey),
    ∀ (u1 u2 u3 u4 : User),
      u1 ≠ u2 ∧ u1 ≠ u3 ∧ u1 ≠ u4 ∧ u2 ≠ u3 ∧ u2 ≠ u4 ∧ u3 ≠ u4 →
        ¬(assignKey u1 u2 = assignKey u1 u3 ∧
          assignKey u1 u2 = assignKey u1 u4 ∧
          assignKey u1 u2 = assignKey u2 u3 ∧
          assignKey u1 u2 = assignKey u2 u4 ∧
          assignKey u1 u2 = assignKey u3 u4) :=
by sorry

end encryption_assignment_exists_l3825_382572


namespace max_value_of_sum_l3825_382521

theorem max_value_of_sum (a b c d : ℝ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_sum : a / b + b / c + c / d + d / a = 4) (h_prod : a * c = b * d) :
  ∃ (max : ℝ), max = -12 ∧ ∀ (a' b' c' d' : ℝ),
    a' / b' + b' / c' + c' / d' + d' / a' = 4 → a' * c' = b' * d' →
    a' / c' + b' / d' + c' / a' + d' / b' ≤ max :=
by sorry

end max_value_of_sum_l3825_382521


namespace price_reduction_achieves_target_profit_l3825_382548

/-- Represents the supermarket's mineral water sales scenario -/
structure MineralWaterSales where
  costPrice : ℝ
  initialSellingPrice : ℝ
  initialMonthlySales : ℝ
  salesIncrease : ℝ
  targetMonthlyProfit : ℝ

/-- Calculates the monthly profit given a price reduction -/
def monthlyProfit (s : MineralWaterSales) (priceReduction : ℝ) : ℝ :=
  let newPrice := s.initialSellingPrice - priceReduction
  let newSales := s.initialMonthlySales + s.salesIncrease * priceReduction
  (newPrice - s.costPrice) * newSales

/-- Theorem stating that a 7 yuan price reduction achieves the target monthly profit -/
theorem price_reduction_achieves_target_profit (s : MineralWaterSales) 
    (h1 : s.costPrice = 24)
    (h2 : s.initialSellingPrice = 36)
    (h3 : s.initialMonthlySales = 60)
    (h4 : s.salesIncrease = 10)
    (h5 : s.targetMonthlyProfit = 650) :
    monthlyProfit s 7 = s.targetMonthlyProfit := by
  sorry

end price_reduction_achieves_target_profit_l3825_382548


namespace angle_A_measure_l3825_382555

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem angle_A_measure (t : Triangle) : 
  t.a = Real.sqrt 3 → t.b = 1 → t.B = π / 6 → t.A = π / 3 := by
  sorry


end angle_A_measure_l3825_382555


namespace new_to_original_detergent_water_ratio_l3825_382556

/-- Represents the ratio of bleach, detergent, and water in a solution -/
structure SolutionRatio where
  bleach : ℚ
  detergent : ℚ
  water : ℚ

/-- The original solution ratio -/
def originalRatio : SolutionRatio :=
  { bleach := 2, detergent := 40, water := 100 }

/-- The new amount of water in liters -/
def newWaterAmount : ℚ := 300

/-- The new amount of detergent in liters -/
def newDetergentAmount : ℚ := 60

/-- The factor by which the bleach to detergent ratio is increased -/
def bleachDetergentIncreaseFactor : ℚ := 3

theorem new_to_original_detergent_water_ratio :
  (newDetergentAmount / (originalRatio.water * newWaterAmount / originalRatio.water)) = 2 / 5 := by
  sorry

end new_to_original_detergent_water_ratio_l3825_382556


namespace inequality_solution_l3825_382581

theorem inequality_solution (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) :
  (x / (x - 1) + (x + 3) / (2 * x) ≥ 4) ↔ (0 < x ∧ x < 1) := by sorry

end inequality_solution_l3825_382581


namespace find_divisor_l3825_382554

theorem find_divisor (dividend quotient remainder divisor : ℕ) : 
  dividend = 16698 →
  quotient = 89 →
  remainder = 14 →
  divisor * quotient + remainder = dividend →
  divisor = 187 := by
sorry

end find_divisor_l3825_382554


namespace hyperbola_asymptotes_l3825_382567

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0, b > 0, 
    and eccentricity e = √3, prove that its asymptotes are y = ±√2 x -/
theorem hyperbola_asymptotes 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (he : Real.sqrt ((a^2 + b^2) / a^2) = Real.sqrt 3) :
  ∃ (k : ℝ), k = Real.sqrt 2 ∧ 
  (∀ (x y : ℝ), (x^2 / a^2 - y^2 / b^2 = 1) → (y = k * x ∨ y = -k * x)) :=
by sorry

end hyperbola_asymptotes_l3825_382567


namespace hannahs_running_distance_l3825_382586

/-- Hannah's running distances problem -/
theorem hannahs_running_distance :
  -- Define the distances
  let monday_distance : ℕ := 9000
  let friday_distance : ℕ := 2095
  let additional_distance : ℕ := 2089

  -- Define the relation between distances
  ∀ wednesday_distance : ℕ,
    monday_distance = wednesday_distance + friday_distance + additional_distance →
    wednesday_distance = 4816 := by
  sorry

end hannahs_running_distance_l3825_382586


namespace tangent_line_to_circle_l3825_382505

/-- The circle C with center (1, 0) and radius 5 -/
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + p.2^2 = 25}

/-- The point M -/
def M : ℝ × ℝ := (-3, 3)

/-- The proposed tangent line -/
def tangentLine (x y : ℝ) : Prop :=
  4 * x - 3 * y + 21 = 0

/-- Theorem stating that the proposed line is tangent to C at M -/
theorem tangent_line_to_circle :
  (M ∈ C) ∧
  (∃ (p : ℝ × ℝ), p ∈ C ∧ p ≠ M ∧ tangentLine p.1 p.2) ∧
  (∀ (q : ℝ × ℝ), q ∈ C → q ≠ M → ¬tangentLine q.1 q.2) :=
sorry

end tangent_line_to_circle_l3825_382505


namespace unique_integer_divisible_by_16_cube_root_between_9_and_9_1_l3825_382519

theorem unique_integer_divisible_by_16_cube_root_between_9_and_9_1 :
  ∃! n : ℕ+, 
    (∃ k : ℕ, n = 16 * k) ∧ 
    9 < (n : ℝ) ^ (1/3) ∧ 
    (n : ℝ) ^ (1/3) < 9.1 ∧
    n = 736 := by
  sorry

end unique_integer_divisible_by_16_cube_root_between_9_and_9_1_l3825_382519


namespace leading_coefficient_of_p_l3825_382545

/-- The polynomial in question -/
def p (x : ℝ) : ℝ := 5*(x^5 - 2*x^4 + 3*x^2) - 8*(x^5 + x^3 - x) + 6*(3*x^5 - x^4 + 2)

/-- The leading coefficient of a polynomial -/
def leading_coefficient (p : ℝ → ℝ) : ℝ := 
  sorry -- Definition of leading coefficient

theorem leading_coefficient_of_p : leading_coefficient p = 15 := by
  sorry

end leading_coefficient_of_p_l3825_382545


namespace point_b_coordinates_l3825_382509

/-- Given a line segment AB parallel to the x-axis with length 3 and point A at coordinates (-1, 2),
    the coordinates of point B are either (-4, 2) or (2, 2). -/
theorem point_b_coordinates :
  ∀ (A B : ℝ × ℝ),
  A = (-1, 2) →
  norm (B.1 - A.1) = 3 →
  B.2 = A.2 →
  B = (-4, 2) ∨ B = (2, 2) :=
by sorry

end point_b_coordinates_l3825_382509


namespace fourth_power_of_nested_square_root_l3825_382539

theorem fourth_power_of_nested_square_root : (Real.sqrt (2 + Real.sqrt (2 + Real.sqrt 2)))^4 = 16 := by
  sorry

end fourth_power_of_nested_square_root_l3825_382539


namespace mask_count_l3825_382527

theorem mask_count (num_boxes : ℕ) (capacity : ℕ) (lacking : ℕ) (total_masks : ℕ) : 
  num_boxes = 18 → 
  capacity = 15 → 
  lacking = 3 → 
  total_masks = num_boxes * (capacity - lacking) → 
  total_masks = 216 := by
sorry

end mask_count_l3825_382527


namespace expression_simplification_l3825_382568

theorem expression_simplification (y : ℝ) : 3*y + 4*y^2 + 2 - (7 - 3*y - 4*y^2) = 8*y^2 + 6*y - 5 := by
  sorry

end expression_simplification_l3825_382568


namespace number_problem_l3825_382535

theorem number_problem (N p q : ℝ) 
  (h1 : N / p = 4)
  (h2 : N / q = 18)
  (h3 : p - q = 0.5833333333333334) : 
  N = 3 := by
sorry

end number_problem_l3825_382535


namespace linear_equation_solution_l3825_382506

theorem linear_equation_solution (x y : ℝ) : 
  3 * x - y = 5 → y = 3 * x - 5 := by
  sorry

end linear_equation_solution_l3825_382506


namespace geometric_sequence_sum_l3825_382514

theorem geometric_sequence_sum (n : ℕ) :
  let a : ℝ := 1
  let r : ℝ := 1/2
  let sum : ℝ := a * (1 - r^n) / (1 - r)
  sum = 31/16 → n = 5 := by
sorry

end geometric_sequence_sum_l3825_382514


namespace game_points_sum_l3825_382580

def g (n : ℕ) : ℕ :=
  if n % 3 = 0 then 8
  else if n % 2 = 0 then 3
  else 0

def allie_rolls : List ℕ := [6, 2, 5, 3, 4]
def carlos_rolls : List ℕ := [3, 2, 2, 6, 1]

theorem game_points_sum : 
  (List.sum (List.map g allie_rolls)) + (List.sum (List.map g carlos_rolls)) = 44 := by
  sorry

end game_points_sum_l3825_382580


namespace parabola_tangents_and_triangle_l3825_382530

/-- Parabola equation: 8y = x^2 + 16 -/
def parabola (x y : ℝ) : Prop := 8 * y = x^2 + 16

/-- Point M coordinates -/
def M : ℝ × ℝ := (3, 0)

/-- Tangent line equation -/
def tangent_line (m b : ℝ) (x y : ℝ) : Prop := y = m * x + b

/-- Points of tangency A and B -/
def A : ℝ × ℝ := (-2, 2.5)
def B : ℝ × ℝ := (8, 10)

/-- Main theorem -/
theorem parabola_tangents_and_triangle :
  ∃ (m₁ b₁ m₂ b₂ : ℝ),
    /- Tangent equations -/
    (∀ x y, tangent_line m₁ b₁ x y ↔ y = -1/2 * x + 1.5) ∧
    (∀ x y, tangent_line m₂ b₂ x y ↔ y = 2 * x - 6) ∧
    /- Angle between tangents -/
    (Real.arctan ((m₂ - m₁) / (1 + m₁ * m₂)) = Real.pi / 2) ∧
    /- Area of triangle ABM -/
    (1/2 * Real.sqrt ((M.1 - A.1)^2 + (M.2 - A.2)^2) *
     Real.sqrt ((M.1 - B.1)^2 + (M.2 - B.2)^2) = 125/4) := by
  sorry

end parabola_tangents_and_triangle_l3825_382530


namespace student_average_age_l3825_382504

theorem student_average_age 
  (num_students : ℕ) 
  (teacher_age : ℕ) 
  (average_with_teacher : ℚ) :
  num_students = 20 →
  teacher_age = 36 →
  average_with_teacher = 16 →
  (num_students * (average_with_teacher : ℚ) + teacher_age) / (num_students + 1 : ℚ) = average_with_teacher →
  (num_students * (average_with_teacher : ℚ) + teacher_age - teacher_age) / num_students = 15 := by
  sorry

end student_average_age_l3825_382504


namespace gus_buys_two_dozen_l3825_382536

def golf_balls_per_dozen : ℕ := 12

def dans_dozens : ℕ := 5
def chris_golf_balls : ℕ := 48
def total_golf_balls : ℕ := 132

def gus_dozens : ℕ := total_golf_balls / golf_balls_per_dozen - dans_dozens - chris_golf_balls / golf_balls_per_dozen

theorem gus_buys_two_dozen : gus_dozens = 2 := by
  sorry

end gus_buys_two_dozen_l3825_382536


namespace machine_a_production_rate_l3825_382569

/-- The number of sprockets produced by each machine -/
def total_sprockets : ℕ := 880

/-- The additional time taken by Machine P compared to Machine Q -/
def time_difference : ℕ := 10

/-- The production rate of Machine Q relative to Machine A -/
def q_rate_relative_to_a : ℚ := 11/10

/-- The production rate of Machine A in sprockets per hour -/
def machine_a_rate : ℚ := 8

/-- The production rate of Machine Q in sprockets per hour -/
def machine_q_rate : ℚ := q_rate_relative_to_a * machine_a_rate

/-- The time taken by Machine Q to produce the total sprockets -/
def machine_q_time : ℚ := total_sprockets / machine_q_rate

/-- The time taken by Machine P to produce the total sprockets -/
def machine_p_time : ℚ := machine_q_time + time_difference

theorem machine_a_production_rate :
  (total_sprockets : ℚ) = machine_a_rate * machine_p_time ∧
  (total_sprockets : ℚ) = machine_q_rate * machine_q_time ∧
  machine_a_rate = 8 := by
  sorry

end machine_a_production_rate_l3825_382569


namespace max_value_abc_fraction_l3825_382558

theorem max_value_abc_fraction (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a * b * c * (a + b + c)) / ((a + b)^2 * (b + c)^2) ≤ (1 : ℝ) / 4 := by
sorry

end max_value_abc_fraction_l3825_382558


namespace temperature_difference_qianan_l3825_382516

/-- The temperature difference between two times of day -/
def temperature_difference (temp1 : Int) (temp2 : Int) : Int :=
  temp2 - temp1

/-- Proof that the temperature difference between 10 a.m. and midnight is 9°C -/
theorem temperature_difference_qianan : 
  let midnight_temp : Int := -4
  let morning_temp : Int := 5
  temperature_difference midnight_temp morning_temp = 9 := by
sorry

end temperature_difference_qianan_l3825_382516


namespace integral_equals_pi_over_four_plus_e_minus_one_l3825_382500

theorem integral_equals_pi_over_four_plus_e_minus_one : 
  ∫ (x : ℝ) in (0)..(1), (Real.sqrt (1 - x^2) + Real.exp x) = π/4 + Real.exp 1 - 1 := by
  sorry

end integral_equals_pi_over_four_plus_e_minus_one_l3825_382500


namespace equation_describes_cone_l3825_382534

/-- Cylindrical coordinates -/
structure CylindricalCoord where
  r : ℝ
  θ : ℝ
  z : ℝ

/-- Predicate for points satisfying θ = 2z -/
def SatisfiesEquation (p : CylindricalCoord) : Prop :=
  p.θ = 2 * p.z

/-- Predicate for points on a cone -/
def OnCone (p : CylindricalCoord) (α : ℝ) : Prop :=
  p.r = α * p.z

theorem equation_describes_cone :
  ∃ α : ℝ, ∀ p : CylindricalCoord, SatisfiesEquation p → OnCone p α :=
sorry

end equation_describes_cone_l3825_382534


namespace definite_integral_evaluation_l3825_382563

theorem definite_integral_evaluation : ∫ x in (1:ℝ)..2, (3 * x^2 - 1) = 6 := by
  sorry

end definite_integral_evaluation_l3825_382563


namespace ninety_eight_squared_l3825_382559

theorem ninety_eight_squared : 98 * 98 = 9604 := by
  sorry

end ninety_eight_squared_l3825_382559


namespace remainder_theorem_l3825_382551

theorem remainder_theorem : ∃ q : ℕ, 2^160 + 160 = q * (2^80 + 2^40 + 1) + 159 :=
sorry

end remainder_theorem_l3825_382551


namespace chinese_chess_pieces_sum_l3825_382595

theorem chinese_chess_pieces_sum :
  ∀ (Rook Knight Cannon : ℕ),
    Rook / Knight = 2 →
    Cannon / Rook = 4 →
    Cannon - Knight = 56 →
    Rook + Knight + Cannon = 88 :=
by
  sorry

end chinese_chess_pieces_sum_l3825_382595


namespace fraction_equals_zero_l3825_382531

theorem fraction_equals_zero (x y : ℝ) :
  (x - 5) / (5 * x + y) = 0 ∧ y ≠ -5 * x → x = 5 ∧ y ≠ -25 := by
  sorry

end fraction_equals_zero_l3825_382531


namespace money_problem_l3825_382512

theorem money_problem (a b : ℝ) 
  (h1 : 6 * a + b > 78)
  (h2 : 4 * a - b = 42)
  (h3 : a ≥ 0)  -- Assuming money can't be negative
  (h4 : b ≥ 0)  -- Assuming money can't be negative
  : a > 12 ∧ b > 6 :=
by
  sorry

end money_problem_l3825_382512


namespace complex_equation_difference_l3825_382544

theorem complex_equation_difference (a b : ℝ) :
  (a : ℂ) + b * Complex.I = (1 + 2 * Complex.I) * (3 - Complex.I) + (1 + Complex.I) / (1 - Complex.I) →
  a - b = -1 := by
sorry

end complex_equation_difference_l3825_382544


namespace ellipse_and_line_intersection_l3825_382562

-- Define the ellipse C
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the line l
def line (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x + 2

-- Define the theorem
theorem ellipse_and_line_intersection :
  ∀ (a b : ℝ),
  a > b ∧ b > 0 ∧
  2 * b = 2 ∧
  (Real.sqrt 6) / 3 = Real.sqrt (a^2 - b^2) / a →
  (∀ (x y : ℝ), ellipse a b x y ↔ x^2 / 3 + y^2 = 1) ∧
  (∀ (k : ℝ),
    (∃ (x₁ y₁ x₂ y₂ : ℝ),
      ellipse a b x₁ y₁ ∧
      ellipse a b x₂ y₂ ∧
      line k x₁ y₁ ∧
      line k x₂ y₂ ∧
      x₁ ≠ x₂ ∧
      x₁ * x₂ + y₁ * y₂ > 0) ↔
    (k > 1 ∧ k < Real.sqrt 13 / Real.sqrt 3) ∨
    (k < -1 ∧ k > -Real.sqrt 13 / Real.sqrt 3)) :=
by sorry

end ellipse_and_line_intersection_l3825_382562


namespace third_term_value_l3825_382543

/-- An arithmetic sequence with five terms -/
structure ArithmeticSequence :=
  (a : ℝ)  -- First term
  (d : ℝ)  -- Common difference

/-- The third term of an arithmetic sequence -/
def third_term (seq : ArithmeticSequence) : ℝ := seq.a + 2 * seq.d

theorem third_term_value :
  ∀ seq : ArithmeticSequence,
  seq.a = 12 ∧ seq.a + 4 * seq.d = 32 →
  third_term seq = 22 :=
by sorry

end third_term_value_l3825_382543


namespace tomatoes_picked_yesterday_l3825_382557

def initial_tomatoes : ℕ := 160
def tomatoes_left_after_yesterday : ℕ := 104

theorem tomatoes_picked_yesterday :
  initial_tomatoes - tomatoes_left_after_yesterday = 56 := by
  sorry

end tomatoes_picked_yesterday_l3825_382557


namespace daniel_gpa_probability_l3825_382501

structure GradeSystem where
  a_points : ℕ
  b_points : ℕ
  c_points : ℕ
  d_points : ℕ

structure SubjectGrades where
  math : ℕ
  history : ℕ
  english : ℕ
  science : ℕ

def gpa (gs : GradeSystem) (sg : SubjectGrades) : ℚ :=
  (sg.math + sg.history + sg.english + sg.science : ℚ) / 4

def english_prob_a : ℚ := 1/5
def english_prob_b : ℚ := 1/3
def english_prob_c : ℚ := 1 - english_prob_a - english_prob_b

def science_prob_a : ℚ := 1/3
def science_prob_b : ℚ := 1/2
def science_prob_c : ℚ := 1/6

theorem daniel_gpa_probability (gs : GradeSystem) 
  (h1 : gs.a_points = 4 ∧ gs.b_points = 3 ∧ gs.c_points = 2 ∧ gs.d_points = 1) :
  let prob_gpa_gte_3_25 := 
    english_prob_a * science_prob_a +
    english_prob_a * science_prob_b +
    english_prob_b * science_prob_a +
    english_prob_b * science_prob_b
  prob_gpa_gte_3_25 = 4/9 := by
  sorry

end daniel_gpa_probability_l3825_382501


namespace cubic_factorization_l3825_382518

theorem cubic_factorization (y : ℝ) : y^3 - 4*y^2 + 4*y = y*(y-2)^2 := by
  sorry

end cubic_factorization_l3825_382518


namespace perfect_square_polynomial_l3825_382528

theorem perfect_square_polynomial (x : ℤ) : 
  (∃ y : ℤ, x^4 + x^3 + x^2 + x + 1 = y^2) ↔ x = -1 ∨ x = 0 ∨ x = 3 :=
by sorry

end perfect_square_polynomial_l3825_382528


namespace line_equation_and_intersection_l3825_382513

/-- The slope of the first line -/
def m : ℚ := 3 / 4

/-- The y-intercept of the first line -/
def b : ℚ := 3 / 2

/-- The slope of the second line -/
def m' : ℚ := -1

/-- The y-intercept of the second line -/
def b' : ℚ := 7

/-- The x-coordinate of the intersection point -/
def x_intersect : ℚ := 11 / 7

/-- The y-coordinate of the intersection point -/
def y_intersect : ℚ := 25 / 7

theorem line_equation_and_intersection :
  (∀ x y : ℚ, 3 * (x - 2) + (-4) * (y - 3) = 0 ↔ y = m * x + b) ∧
  (m * x_intersect + b = m' * x_intersect + b') ∧
  (y_intersect = m * x_intersect + b) ∧
  (y_intersect = m' * x_intersect + b') := by
  sorry

end line_equation_and_intersection_l3825_382513


namespace triangle_side_calculation_l3825_382540

theorem triangle_side_calculation (a b c : ℝ) (A B C : ℝ) :
  a = 4 →
  A = π / 6 →
  B = π / 3 →
  (a / Real.sin A = b / Real.sin B) →
  b = 4 * Real.sqrt 3 := by
  sorry

end triangle_side_calculation_l3825_382540


namespace two_digit_triple_sum_product_l3825_382561

def digit_sum (n : ℕ) : ℕ := sorry

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def all_digits_different (p q r : ℕ) : Prop :=
  let digits := [p / 10, p % 10, q / 10, q % 10, r / 10, r % 10]
  ∀ i j, i ≠ j → digits.nthLe i sorry ≠ digits.nthLe j sorry

theorem two_digit_triple_sum_product (p q r : ℕ) : 
  is_two_digit p ∧ is_two_digit q ∧ is_two_digit r ∧ 
  all_digits_different p q r ∧
  p * q * digit_sum r = p * digit_sum q * r ∧
  p * digit_sum q * r = digit_sum p * q * r →
  ((p = 12 ∧ q = 36 ∧ r = 48) ∨ (p = 21 ∧ q = 63 ∧ r = 84)) :=
sorry

end two_digit_triple_sum_product_l3825_382561


namespace star_theorems_l3825_382570

variable {S : Type*} [Inhabited S] [Nontrivial S]
variable (star : S → S → S)

axiom star_property : ∀ a b : S, star a (star b a) = b

theorem star_theorems :
  (∀ a b : S, star (star a (star b a)) (star a b) = a) ∧
  (∀ b : S, star b (star b b) = b) ∧
  (∀ a b : S, star (star a b) (star b (star a b)) = b) :=
by sorry

end star_theorems_l3825_382570


namespace pizza_toppings_l3825_382599

theorem pizza_toppings (total_slices : ℕ) (cheese_slices : ℕ) (bacon_slices : ℕ) 
  (h1 : total_slices = 15)
  (h2 : cheese_slices = 8)
  (h3 : bacon_slices = 13)
  (h4 : ∀ slice, slice ∈ Finset.range total_slices → 
    (slice ∈ Finset.range cheese_slices ∨ slice ∈ Finset.range bacon_slices)) :
  ∃ both_toppings : ℕ, both_toppings = 6 ∧ 
    cheese_slices + bacon_slices - both_toppings = total_slices :=
by
  sorry

end pizza_toppings_l3825_382599


namespace quadratic_root_relation_l3825_382565

/-- 
Given a quadratic equation ax^2 + bx + c = 0, 
if the sum of its roots is twice their difference, 
then 3b^2 = 16ac 
-/
theorem quadratic_root_relation (a b c : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : a ≠ 0)
  (h₂ : a * x₁^2 + b * x₁ + c = 0)
  (h₃ : a * x₂^2 + b * x₂ + c = 0)
  (h₄ : x₁ + x₂ = 2 * (x₁ - x₂)) : 
  3 * b^2 = 16 * a * c := by
  sorry

end quadratic_root_relation_l3825_382565


namespace v_1013_equals_5_l3825_382576

def g : ℕ → ℕ
| 1 => 5
| 2 => 3
| 3 => 4
| 4 => 1
| 5 => 2
| _ => 0  -- Default case for completeness

def v : ℕ → ℕ
| 0 => 3
| (n + 1) => g (v n)

theorem v_1013_equals_5 : v 1013 = 5 := by
  sorry

end v_1013_equals_5_l3825_382576


namespace intersection_of_A_and_B_l3825_382589

def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x : ℝ | x < 1}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -1 ≤ x ∧ x < 1} := by sorry

end intersection_of_A_and_B_l3825_382589


namespace jose_profit_share_l3825_382597

/-- Calculates the share of profit for an investor given their investment amount, duration, and the total profit and investment-months. -/
def shareOfProfit (investment : ℕ) (duration : ℕ) (totalProfit : ℕ) (totalInvestmentMonths : ℕ) : ℕ :=
  (investment * duration * totalProfit) / totalInvestmentMonths

theorem jose_profit_share (tomInvestment jose_investment : ℕ) (tomDuration joseDuration : ℕ) (totalProfit : ℕ)
    (h1 : tomInvestment = 30000)
    (h2 : jose_investment = 45000)
    (h3 : tomDuration = 12)
    (h4 : joseDuration = 10)
    (h5 : totalProfit = 72000) :
  shareOfProfit jose_investment joseDuration totalProfit (tomInvestment * tomDuration + jose_investment * joseDuration) = 40000 := by
  sorry

#eval shareOfProfit 45000 10 72000 (30000 * 12 + 45000 * 10)

end jose_profit_share_l3825_382597


namespace geometric_progression_ratio_l3825_382549

theorem geometric_progression_ratio (a b c d : ℝ) : 
  0 < a → a < b → b < c → c < d → d = 2*a →
  (d - a) * (a^2 / (b - a) + b^2 / (c - b) + c^2 / (d - c)) = (a + b + c)^2 →
  b * c * d / a^3 = 4 := by
sorry

end geometric_progression_ratio_l3825_382549


namespace double_divide_four_equals_twelve_l3825_382593

theorem double_divide_four_equals_twelve (x : ℝ) : (2 * x) / 4 = 12 → x = 24 := by
  sorry

end double_divide_four_equals_twelve_l3825_382593


namespace initial_girls_count_l3825_382538

theorem initial_girls_count (total : ℕ) (initial_girls : ℕ) : 
  (initial_girls : ℚ) / total = 35 / 100 →
  ((initial_girls : ℚ) - 3) / (total : ℚ) = 25 / 100 →
  initial_girls = 11 := by
  sorry

end initial_girls_count_l3825_382538


namespace combined_value_of_a_and_b_l3825_382510

-- Define the conversion rate from paise to rupees
def paise_to_rupees (paise : ℚ) : ℚ := paise / 100

-- Define the value of a in rupees
def a : ℚ := (paise_to_rupees 95) / 0.005

-- Define the value of b in rupees
def b : ℚ := 3 * a - 50

-- Theorem statement
theorem combined_value_of_a_and_b : a + b = 710 := by sorry

end combined_value_of_a_and_b_l3825_382510


namespace comic_stacking_arrangements_l3825_382594

def batman_comics : ℕ := 8
def superman_comics : ℕ := 6
def wonder_woman_comics : ℕ := 3

theorem comic_stacking_arrangements :
  (batman_comics.factorial * superman_comics.factorial * wonder_woman_comics.factorial) *
  (batman_comics + superman_comics + wonder_woman_comics).choose 3 = 1040486400 :=
by sorry

end comic_stacking_arrangements_l3825_382594


namespace exponent_division_simplification_l3825_382596

theorem exponent_division_simplification (a b : ℝ) :
  (-a * b)^5 / (-a * b)^3 = a^2 * b^2 := by
  sorry

end exponent_division_simplification_l3825_382596


namespace sum_le_product_plus_two_l3825_382546

theorem sum_le_product_plus_two (x y z : ℝ) (h : x^2 + y^2 + z^2 = 2) :
  x + y + z ≤ x*y*z + 2 := by sorry

end sum_le_product_plus_two_l3825_382546


namespace triangle_side_values_l3825_382550

theorem triangle_side_values (A B C : Real) (a b c : Real) :
  c = Real.sqrt 3 →
  C = π / 3 →
  Real.sin B = 2 * Real.sin A →
  a ^ 2 + b ^ 2 - a * b = 3 →
  (a = 1 ∧ b = 2) := by
  sorry

end triangle_side_values_l3825_382550


namespace value_of_two_minus_c_l3825_382541

theorem value_of_two_minus_c (c d : ℤ) 
  (h1 : 5 + c = 6 - d) 
  (h2 : 3 + d = 8 + c) : 
  2 - c = -1 := by
sorry

end value_of_two_minus_c_l3825_382541


namespace factorization_equality_l3825_382553

theorem factorization_equality (x y : ℝ) : x + x^2 - y - y^2 = (x + y + 1) * (x - y) := by
  sorry

end factorization_equality_l3825_382553


namespace bottles_used_second_game_l3825_382582

theorem bottles_used_second_game :
  let total_bottles : ℕ := 10 * 20
  let bottles_used_first_game : ℕ := 70
  let bottles_left_after_second_game : ℕ := 20
  let bottles_used_second_game : ℕ := total_bottles - bottles_used_first_game - bottles_left_after_second_game
  bottles_used_second_game = 110 := by sorry

end bottles_used_second_game_l3825_382582


namespace dandelion_seed_production_dandelion_seed_production_proof_l3825_382532

/-- Calculates the total number of seeds produced by dandelion plants in three months --/
theorem dandelion_seed_production : ℕ :=
  let initial_seeds := 50
  let germination_rate := 0.60
  let one_month_growth_rate := 0.80
  let two_month_growth_rate := 0.10
  let three_month_growth_rate := 0.10
  let one_month_seed_production := 60
  let two_month_seed_production := 40
  let three_month_seed_production := 20

  let germinated_seeds := (initial_seeds : ℚ) * germination_rate
  let one_month_plants := germinated_seeds * one_month_growth_rate
  let two_month_plants := germinated_seeds * two_month_growth_rate
  let three_month_plants := germinated_seeds * three_month_growth_rate

  let one_month_seeds := (one_month_plants * one_month_seed_production).floor
  let two_month_seeds := (two_month_plants * two_month_seed_production).floor
  let three_month_seeds := (three_month_plants * three_month_seed_production).floor

  let total_seeds := one_month_seeds + two_month_seeds + three_month_seeds

  1620

theorem dandelion_seed_production_proof : dandelion_seed_production = 1620 := by
  sorry

end dandelion_seed_production_dandelion_seed_production_proof_l3825_382532


namespace hexagon_coins_proof_l3825_382537

/-- The number of coins needed to construct a hexagon with side length n -/
def hexagon_coins (n : ℕ) : ℕ := 3 * n * (n - 1) + 1

theorem hexagon_coins_proof :
  (hexagon_coins 2 = 7) ∧
  (hexagon_coins 3 = 19) ∧
  (hexagon_coins 10 = 271) :=
by sorry

end hexagon_coins_proof_l3825_382537


namespace negation_existential_geq_zero_l3825_382502

theorem negation_existential_geq_zero :
  ¬(∃ x : ℝ, x + 1 ≥ 0) ↔ ∀ x : ℝ, x + 1 < 0 :=
by sorry

end negation_existential_geq_zero_l3825_382502


namespace businessmen_beverage_problem_l3825_382515

theorem businessmen_beverage_problem (total : ℕ) (coffee tea soda : ℕ) 
  (coffee_tea coffee_soda tea_soda : ℕ) (all_three : ℕ) 
  (h_total : total = 30)
  (h_coffee : coffee = 15)
  (h_tea : tea = 13)
  (h_soda : soda = 8)
  (h_coffee_tea : coffee_tea = 6)
  (h_coffee_soda : coffee_soda = 2)
  (h_tea_soda : tea_soda = 3)
  (h_all_three : all_three = 1) : 
  total - (coffee + tea + soda - coffee_tea - coffee_soda - tea_soda + all_three) = 4 := by
sorry

end businessmen_beverage_problem_l3825_382515


namespace max_turtles_on_board_l3825_382522

/-- Represents a rectangular board -/
structure Board :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a position on the board -/
structure Position :=
  (row : ℕ)
  (col : ℕ)

/-- Represents a turtle on the board -/
structure Turtle :=
  (position : Position)
  (last_move : Direction)

/-- Possible directions of movement -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Defines a valid move for a turtle -/
def valid_move (b : Board) (t : Turtle) (new_pos : Position) : Prop :=
  (new_pos.row < b.rows) ∧
  (new_pos.col < b.cols) ∧
  ((t.last_move = Direction.Up ∨ t.last_move = Direction.Down) →
    (new_pos.row = t.position.row ∧ (new_pos.col = t.position.col + 1 ∨ new_pos.col = t.position.col - 1))) ∧
  ((t.last_move = Direction.Left ∨ t.last_move = Direction.Right) →
    (new_pos.col = t.position.col ∧ (new_pos.row = t.position.row + 1 ∨ new_pos.row = t.position.row - 1)))

/-- Defines a valid configuration of turtles on the board -/
def valid_configuration (b : Board) (turtles : List Turtle) : Prop :=
  ∀ t1 t2 : Turtle, t1 ∈ turtles → t2 ∈ turtles → t1 ≠ t2 →
    t1.position ≠ t2.position

/-- Theorem: The maximum number of turtles that can move indefinitely on a 101x99 board is 9800 -/
theorem max_turtles_on_board :
  ∀ (turtles : List Turtle),
    valid_configuration (Board.mk 101 99) turtles →
    (∀ (n : ℕ), ∃ (new_turtles : List Turtle),
      valid_configuration (Board.mk 101 99) new_turtles ∧
      turtles.length = new_turtles.length ∧
      ∀ (t : Turtle), t ∈ turtles →
        ∃ (new_t : Turtle), new_t ∈ new_turtles ∧
          valid_move (Board.mk 101 99) t new_t.position) →
    turtles.length ≤ 9800 :=
sorry

end max_turtles_on_board_l3825_382522


namespace inscribed_circle_and_square_l3825_382523

theorem inscribed_circle_and_square (r : ℝ) (s : ℝ) : 
  -- Circle inscribed in a 3-4-5 right triangle
  r = 1 →
  -- Square concentric with circle and inside it
  s * Real.sqrt 2 = 2 →
  -- Side length of square is √2
  s = Real.sqrt 2 ∧
  -- Area between circle and square is π - 2
  π * r^2 - s^2 = π - 2 := by
sorry

end inscribed_circle_and_square_l3825_382523


namespace parabola_tangent_to_line_l3825_382517

/-- A parabola y = ax^2 + bx + 2 is tangent to the line y = 2x + 3 if and only if a = -1 and b = 4 -/
theorem parabola_tangent_to_line (a b : ℝ) : 
  (∃ x : ℝ, ax^2 + bx + 2 = 2*x + 3 ∧ 
   ∀ y : ℝ, y ≠ x → ax^2 + bx + 2 ≠ 2*y + 3) ↔ 
  (a = -1 ∧ b = 4) :=
sorry

end parabola_tangent_to_line_l3825_382517


namespace sin_product_equality_l3825_382566

theorem sin_product_equality : 
  Real.sin (8 * π / 180) * Real.sin (40 * π / 180) * Real.sin (70 * π / 180) * Real.sin (82 * π / 180) = 3 * Real.sqrt 3 / 16 := by
  sorry

end sin_product_equality_l3825_382566


namespace cookie_ratio_l3825_382520

/-- Prove that the ratio of Chris's cookies to Kenny's cookies is 1:2 -/
theorem cookie_ratio (total : ℕ) (glenn : ℕ) (kenny : ℕ) (chris : ℕ)
  (h1 : total = 33)
  (h2 : glenn = 24)
  (h3 : glenn = 4 * kenny)
  (h4 : total = chris + kenny + glenn) :
  chris = kenny / 2 := by
  sorry

end cookie_ratio_l3825_382520


namespace equal_area_rectangles_width_l3825_382508

/-- Given two rectangles of equal area, where one rectangle has dimensions 8 inches by 45 inches,
    and the other has a length of 15 inches, prove that the width of the second rectangle is 24 inches. -/
theorem equal_area_rectangles_width (area jordan_length jordan_width carol_length : ℝ)
    (h1 : area = jordan_length * jordan_width)
    (h2 : area = carol_length * (area / carol_length))
    (h3 : jordan_length = 8)
    (h4 : jordan_width = 45)
    (h5 : carol_length = 15) :
    area / carol_length = 24 := by
  sorry

end equal_area_rectangles_width_l3825_382508


namespace largest_n_binomial_equality_l3825_382598

theorem largest_n_binomial_equality : ∃ (n : ℕ), (Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 n) ∧ 
  (∀ (m : ℕ), Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 m → m ≤ n) :=
by
  -- The proof goes here
  sorry

end largest_n_binomial_equality_l3825_382598


namespace jens_age_difference_l3825_382542

/-- Proves that the difference between 3 times Jen's son's current age and Jen's current age is 7 years -/
theorem jens_age_difference (jen_age_at_birth : ℕ) (son_current_age : ℕ) (jen_current_age : ℕ) : 
  jen_age_at_birth = 25 →
  son_current_age = 16 →
  jen_current_age = 41 →
  3 * son_current_age - jen_current_age = 7 := by
  sorry

end jens_age_difference_l3825_382542


namespace jersey_shoe_cost_ratio_l3825_382529

/-- Given the information about Jeff's purchase of shoes and jerseys,
    prove that the ratio of the cost of one jersey to one pair of shoes is 1:4 -/
theorem jersey_shoe_cost_ratio :
  ∀ (total_cost shoe_cost : ℕ) (shoe_pairs jersey_count : ℕ),
    total_cost = 560 →
    shoe_cost = 480 →
    shoe_pairs = 6 →
    jersey_count = 4 →
    (total_cost - shoe_cost) / jersey_count / (shoe_cost / shoe_pairs) = 1 / 4 := by
  sorry

end jersey_shoe_cost_ratio_l3825_382529


namespace max_increase_two_letters_l3825_382578

/-- Represents the sets of letters for each position in the license plate --/
structure LetterSets :=
  (first : Finset Char)
  (second : Finset Char)
  (third : Finset Char)

/-- Calculates the total number of possible license plates --/
def totalPlates (sets : LetterSets) : ℕ :=
  sets.first.card * sets.second.card * sets.third.card

/-- The initial configuration of letter sets --/
def initialSets : LetterSets :=
  { first := {'C', 'H', 'L', 'P', 'R'},
    second := {'A', 'I', 'O'},
    third := {'D', 'M', 'N', 'T'} }

/-- Theorem stating the maximum increase in license plates after adding two letters --/
theorem max_increase_two_letters :
  ∃ (newSets : LetterSets), 
    (newSets.first.card + newSets.second.card + newSets.third.card = 
     initialSets.first.card + initialSets.second.card + initialSets.third.card + 2) ∧
    (totalPlates newSets - totalPlates initialSets = 40) ∧
    ∀ (otherSets : LetterSets), 
      (otherSets.first.card + otherSets.second.card + otherSets.third.card = 
       initialSets.first.card + initialSets.second.card + initialSets.third.card + 2) →
      (totalPlates otherSets - totalPlates initialSets ≤ 40) :=
by sorry


end max_increase_two_letters_l3825_382578


namespace system_equation_sum_l3825_382526

theorem system_equation_sum (a b c x y z : ℝ) 
  (eq1 : 11 * x + b * y + c * z = 0)
  (eq2 : a * x + 19 * y + c * z = 0)
  (eq3 : a * x + b * y + 37 * z = 0)
  (ha : a ≠ 11)
  (hx : x ≠ 0) :
  a / (a - 11) + b / (b - 19) + c / (c - 37) = 1 := by
sorry

end system_equation_sum_l3825_382526


namespace average_miles_per_year_approx_2000_l3825_382533

/-- Calculates the approximate average miles rowed per year -/
def approximateAverageMilesPerYear (currentAge : ℕ) (ageReceived : ℕ) (totalMiles : ℕ) : ℕ :=
  let yearsRowing := currentAge - ageReceived
  let exactAverage := totalMiles / yearsRowing
  -- Round to the nearest thousand
  (exactAverage + 500) / 1000 * 1000

/-- Theorem stating that the average miles rowed per year is approximately 2000 -/
theorem average_miles_per_year_approx_2000 :
  approximateAverageMilesPerYear 63 50 25048 = 2000 := by
  sorry

#eval approximateAverageMilesPerYear 63 50 25048

end average_miles_per_year_approx_2000_l3825_382533


namespace square_triangle_area_ratio_l3825_382573

/-- Given a square with side length s, where R is the midpoint of one side,
    S is the midpoint of a diagonal, and V is a vertex,
    prove that the area of triangle RSV is √2/16 of the square's area. -/
theorem square_triangle_area_ratio (s : ℝ) (h : s > 0) :
  let square_area := s^2
  let r_to_s := s / 2
  let s_to_v := s * Real.sqrt 2 / 2
  let r_to_v := s
  let triangle_height := s * Real.sqrt 2 / 4
  let triangle_area := 1 / 2 * r_to_s * triangle_height
  triangle_area / square_area = Real.sqrt 2 / 16 := by
sorry

end square_triangle_area_ratio_l3825_382573


namespace first_candidate_percentage_l3825_382547

/-- Given an election between two candidates where the total number of votes is 600
    and the second candidate received 240 votes, prove that the first candidate
    received 60% of the votes. -/
theorem first_candidate_percentage (total_votes : ℕ) (second_candidate_votes : ℕ)
  (h1 : total_votes = 600)
  (h2 : second_candidate_votes = 240) :
  (total_votes - second_candidate_votes : ℚ) / total_votes * 100 = 60 := by
  sorry

end first_candidate_percentage_l3825_382547


namespace sequence_divisibility_l3825_382525

/-- A sequence of 2007 elements, each either 2 or 3 -/
def Sequence := Fin 2007 → Fin 2

/-- The property that all elements of a sequence are divisible by 5 -/
def AllDivisibleBy5 (x : Fin 2007 → ℤ) : Prop :=
  ∀ i, x i % 5 = 0

/-- The main theorem -/
theorem sequence_divisibility (a : Sequence) (x : Fin 2007 → ℤ)
    (h : ∀ i : Fin 2007, (a i.val + 2 : Fin 2) * x i + x ((i + 2) % 2007) ≡ 0 [ZMOD 5]) :
    AllDivisibleBy5 x := by
  sorry

end sequence_divisibility_l3825_382525


namespace movie_choice_l3825_382585

-- Define the set of all movies
def Movies : Set Char := {'A', 'B', 'C', 'D', 'E'}

-- Define the acceptable movies for each person
def Zhao : Set Char := Movies \ {'B'}
def Zhang : Set Char := {'B', 'C', 'D', 'E'}
def Li : Set Char := Movies \ {'C'}
def Liu : Set Char := Movies \ {'E'}

-- Theorem statement
theorem movie_choice : Zhao ∩ Zhang ∩ Li ∩ Liu = {'D'} := by
  sorry

end movie_choice_l3825_382585


namespace factorization_x4_3x2_1_l3825_382583

theorem factorization_x4_3x2_1 (x : ℝ) :
  x^4 - 3*x^2 + 1 = (x^2 + x - 1) * (x^2 - x - 1) := by sorry

end factorization_x4_3x2_1_l3825_382583


namespace value_of_N_l3825_382577

theorem value_of_N : ∃ N : ℕ, (15 * N = 45 * 2003) ∧ (N = 6009) := by
  sorry

end value_of_N_l3825_382577


namespace garden_roller_diameter_l3825_382503

/-- The diameter of a garden roller given its length, area covered, and number of revolutions. -/
theorem garden_roller_diameter
  (length : ℝ)
  (area_covered : ℝ)
  (revolutions : ℕ)
  (h1 : length = 2)
  (h2 : area_covered = 52.8)
  (h3 : revolutions = 6)
  (h4 : Real.pi = 22 / 7) :
  ∃ (diameter : ℝ), diameter = 1.4 ∧ 
    area_covered = revolutions * Real.pi * diameter * length :=
by sorry

end garden_roller_diameter_l3825_382503


namespace unbroken_seashells_l3825_382511

def total_seashells : ℕ := 7
def broken_seashells : ℕ := 4

theorem unbroken_seashells :
  total_seashells - broken_seashells = 3 := by sorry

end unbroken_seashells_l3825_382511


namespace evan_needs_seven_l3825_382552

-- Define the given amounts
def david_found : ℕ := 12
def evan_initial : ℕ := 1
def watch_cost : ℕ := 20

-- Define Evan's total after receiving money from David
def evan_total : ℕ := evan_initial + david_found

-- Theorem to prove
theorem evan_needs_seven : watch_cost - evan_total = 7 := by
  sorry

end evan_needs_seven_l3825_382552


namespace spade_calculation_l3825_382588

-- Define the ⋄ operation
def spade (a b : ℝ) : ℝ := |a - b|

-- Theorem statement
theorem spade_calculation : spade (spade 1 2) (spade 9 (spade 5 4)) = 7 := by
  sorry

end spade_calculation_l3825_382588


namespace cube_root_27_fourth_root_81_sixth_root_64_eq_18_l3825_382579

theorem cube_root_27_fourth_root_81_sixth_root_64_eq_18 :
  (27 : ℝ) ^ (1/3) * (81 : ℝ) ^ (1/4) * (64 : ℝ) ^ (1/6) = 18 := by
  sorry

end cube_root_27_fourth_root_81_sixth_root_64_eq_18_l3825_382579


namespace star_properties_l3825_382524

noncomputable def star (x y : ℝ) : ℝ := Real.log (10^x + 10^y) / Real.log 10

theorem star_properties :
  (∀ a b : ℝ, star a b = star b a) ∧
  (∀ a b c : ℝ, star (star a b) c = star a (star b c)) ∧
  (∀ a b c : ℝ, star a b + c = star (a + c) (b + c)) ∧
  (∃ a b c : ℝ, star a b * c ≠ star (a * c) (b * c)) :=
by sorry

end star_properties_l3825_382524


namespace f_pi_sixth_value_l3825_382574

/-- Given a function f(x) = 2sin(ωx + φ) where for all x, f(π/3 + x) = f(-x),
    prove that f(π/6) is either -2 or 2. -/
theorem f_pi_sixth_value (ω φ : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 2 * Real.sin (ω * x + φ)
  (∀ x, f (π / 3 + x) = f (-x)) →
  f (π / 6) = -2 ∨ f (π / 6) = 2 :=
by sorry

end f_pi_sixth_value_l3825_382574


namespace twins_age_problem_l3825_382590

theorem twins_age_problem (x : ℕ) : 
  (x + 1) * (x + 1) = x * x + 11 → x = 5 := by
sorry

end twins_age_problem_l3825_382590


namespace four_composition_odd_l3825_382560

-- Define a type for real-valued functions
def RealFunction := ℝ → ℝ

-- Define what it means for a function to be odd
def IsOdd (f : RealFunction) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem four_composition_odd (f : RealFunction) (h : IsOdd f) :
  IsOdd (fun x ↦ f (f (f (f x)))) := by
  sorry

end four_composition_odd_l3825_382560
