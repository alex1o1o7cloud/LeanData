import Mathlib

namespace not_all_equilateral_triangles_congruent_l1166_116688

/-- An equilateral triangle is a triangle where all three sides are of equal length. -/
structure EquilateralTriangle where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- Two triangles are congruent if they have the same size and shape. -/
def congruent (t1 t2 : EquilateralTriangle) : Prop :=
  t1.side_length = t2.side_length

theorem not_all_equilateral_triangles_congruent :
  ∃ t1 t2 : EquilateralTriangle, ¬(congruent t1 t2) :=
sorry

end not_all_equilateral_triangles_congruent_l1166_116688


namespace quadratic_inequality_implies_a_bound_l1166_116606

theorem quadratic_inequality_implies_a_bound (a : ℝ) : 
  (∀ x : ℝ, x ≥ 1 → x^2 + a*x + 9 ≥ 0) → a ≥ -6 := by
  sorry

end quadratic_inequality_implies_a_bound_l1166_116606


namespace no_integer_solution_l1166_116630

theorem no_integer_solution : ¬∃ (x y : ℤ), x * y + 4 = 40 ∧ x + y = 14 := by sorry

end no_integer_solution_l1166_116630


namespace compound_interest_doubling_l1166_116675

/-- The annual interest rate as a decimal -/
def r : ℝ := 0.15

/-- The compound interest factor for one year -/
def factor : ℝ := 1 + r

/-- The number of years we're proving about -/
def years : ℕ := 5

theorem compound_interest_doubling :
  (∀ n : ℕ, n < years → factor ^ n ≤ 2) ∧
  factor ^ years > 2 := by
  sorry

end compound_interest_doubling_l1166_116675


namespace larger_number_is_fifty_l1166_116622

theorem larger_number_is_fifty (a b : ℝ) : 
  (4 * b = 5 * a) → (b - a = 10) → b = 50 := by
  sorry

end larger_number_is_fifty_l1166_116622


namespace final_ratio_is_four_to_one_l1166_116654

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Represents the transformations applied to the rectangle -/
def transform (r : Rectangle) : Rectangle :=
  let r1 := Rectangle.mk (2 * r.length) (r.width / 2)
  let r2 := 
    if 2 * r1.length > r1.width
    then Rectangle.mk (r1.length + 1) (r1.width - 4)
    else Rectangle.mk (r1.length - 4) (r1.width + 1)
  let r3 := 
    if r2.length > r2.width
    then Rectangle.mk r2.length (r2.width - 1)
    else Rectangle.mk r2.length (r2.width - 1)
  r3

/-- The theorem stating that after transformations, the ratio of sides is 4:1 -/
theorem final_ratio_is_four_to_one (r : Rectangle) :
  let final := transform r
  (final.length : ℚ) / final.width = 4 :=
sorry

end final_ratio_is_four_to_one_l1166_116654


namespace problem_statement_l1166_116607

theorem problem_statement :
  (∀ k : ℕ, (∀ a b : ℕ+, ab + (a + 1) * (b + 1) ≠ 2^k) → Nat.Prime (k + 1)) ∧
  (∃ k : ℕ, Nat.Prime (k + 1) ∧ ∃ a b : ℕ+, ab + (a + 1) * (b + 1) = 2^k) := by
  sorry

end problem_statement_l1166_116607


namespace valid_numbers_l1166_116623

def isValidNumber (n : ℕ) : Prop :=
  n ≥ 500 ∧ n < 1000 ∧
  (n / 100 % 2 = 1) ∧
  ((n / 10) % 10 % 2 = 0) ∧
  (n % 10 % 2 = 0) ∧
  (n / 100 % 3 = 0) ∧
  ((n / 10) % 10 % 3 = 0) ∧
  (n % 10 % 3 ≠ 0)

theorem valid_numbers :
  {n : ℕ | isValidNumber n} = {902, 904, 908, 962, 964, 968} :=
by sorry

end valid_numbers_l1166_116623


namespace remainder_4063_div_97_l1166_116673

theorem remainder_4063_div_97 : 4063 % 97 = 86 := by
  sorry

end remainder_4063_div_97_l1166_116673


namespace deer_families_stayed_l1166_116632

theorem deer_families_stayed (total : ℕ) (moved_out : ℕ) (h1 : total = 79) (h2 : moved_out = 34) :
  total - moved_out = 45 := by
  sorry

end deer_families_stayed_l1166_116632


namespace exists_valid_cylinder_arrangement_l1166_116621

/-- Represents a straight circular cylinder in 3D space -/
structure Cylinder where
  center : ℝ × ℝ × ℝ
  radius : ℝ
  height : ℝ

/-- Checks if two cylinders have a common boundary point -/
def havesCommonPoint (c1 c2 : Cylinder) : Prop := sorry

/-- Represents an arrangement of six cylinders -/
def CylinderArrangement := Fin 6 → Cylinder

/-- Checks if a given arrangement satisfies the condition that each cylinder
    has a common point with every other cylinder -/
def isValidArrangement (arr : CylinderArrangement) : Prop :=
  ∀ i j, i ≠ j → havesCommonPoint (arr i) (arr j)

/-- The main theorem stating that there exists a valid arrangement of six cylinders -/
theorem exists_valid_cylinder_arrangement :
  ∃ (arr : CylinderArrangement), isValidArrangement arr := by sorry

end exists_valid_cylinder_arrangement_l1166_116621


namespace two_roots_implication_l1166_116642

/-- If a quadratic trinomial ax^2 + bx + c has two roots, 
    then the trinomial 3ax^2 + 2(a + b)x + (b + c) also has two roots. -/
theorem two_roots_implication (a b c : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) →
  (∃ u v : ℝ, u ≠ v ∧ 3 * a * u^2 + 2 * (a + b) * u + (b + c) = 0 ∧ 
                    3 * a * v^2 + 2 * (a + b) * v + (b + c) = 0) :=
by sorry

end two_roots_implication_l1166_116642


namespace quadratic_inequality_solution_set_l1166_116686

theorem quadratic_inequality_solution_set :
  {x : ℝ | (x + 3) * (2 - x) < 0} = {x : ℝ | x < -3 ∨ x > 2} := by sorry

end quadratic_inequality_solution_set_l1166_116686


namespace square_difference_l1166_116672

theorem square_difference (a b : ℝ) (h1 : a + b = 2) (h2 : a - b = 1) : a^2 - b^2 = 2 := by
  sorry

end square_difference_l1166_116672


namespace minervas_stamps_l1166_116645

/-- Given that Lizette has 813 stamps and 125 more stamps than Minerva,
    prove that Minerva has 688 stamps. -/
theorem minervas_stamps :
  let lizette_stamps : ℕ := 813
  let difference : ℕ := 125
  let minerva_stamps : ℕ := lizette_stamps - difference
  minerva_stamps = 688 := by
sorry

end minervas_stamps_l1166_116645


namespace fred_car_wash_earnings_l1166_116659

/-- The amount Fred earned by washing cars -/
def fred_earnings (initial_amount final_amount : ℕ) : ℕ :=
  final_amount - initial_amount

/-- Proof that Fred earned $4 by washing cars -/
theorem fred_car_wash_earnings : 
  fred_earnings 111 115 = 4 := by sorry

end fred_car_wash_earnings_l1166_116659


namespace parallel_segments_k_value_l1166_116667

/-- Given points A, B, X, and Y on a Cartesian plane, prove that if AB is parallel to XY, then k = -8 -/
theorem parallel_segments_k_value (k : ℝ) : 
  let A : ℝ × ℝ := (-6, 2)
  let B : ℝ × ℝ := (2, -6)
  let X : ℝ × ℝ := (0, 10)
  let Y : ℝ × ℝ := (18, k)
  let slope (p q : ℝ × ℝ) := (q.2 - p.2) / (q.1 - p.1)
  slope A B = slope X Y → k = -8 := by
  sorry

end parallel_segments_k_value_l1166_116667


namespace abs_five_necessary_not_sufficient_l1166_116651

theorem abs_five_necessary_not_sufficient :
  (∀ x : ℝ, x = 5 → |x| = 5) ∧
  ¬(∀ x : ℝ, |x| = 5 → x = 5) :=
by sorry

end abs_five_necessary_not_sufficient_l1166_116651


namespace average_speed_calculation_l1166_116615

theorem average_speed_calculation (total_distance : ℝ) (distance1 : ℝ) (speed1 : ℝ) (distance2 : ℝ) (speed2 : ℝ) :
  total_distance = 80 ∧
  distance1 = 30 ∧
  speed1 = 30 ∧
  distance2 = 50 ∧
  speed2 = 50 →
  (total_distance / (distance1 / speed1 + distance2 / speed2)) = 40 := by
  sorry

end average_speed_calculation_l1166_116615


namespace unique_six_digit_number_l1166_116656

theorem unique_six_digit_number : ∃! n : ℕ,
  100000 ≤ n ∧ n < 1000000 ∧
  n / 100000 = 1 ∧
  3 * n = (n % 100000) * 10 + 1 ∧
  n = 142857 := by
  sorry

end unique_six_digit_number_l1166_116656


namespace trailing_zeros_500_50_l1166_116657

theorem trailing_zeros_500_50 : ∃ n : ℕ, 500^50 = n * 10^100 ∧ n % 10 ≠ 0 := by
  sorry

end trailing_zeros_500_50_l1166_116657


namespace prime_iff_totient_and_divisor_sum_condition_l1166_116687

/-- Euler's totient function -/
def φ : ℕ → ℕ := sorry

/-- Divisor sum function -/
def σ : ℕ → ℕ := sorry

/-- An integer n ≥ 2 is prime if and only if φ(n) divides (n - 1) and (n + 1) divides σ(n) -/
theorem prime_iff_totient_and_divisor_sum_condition (n : ℕ) (h : n ≥ 2) :
  Nat.Prime n ↔ (φ n ∣ n - 1) ∧ (n + 1 ∣ σ n) := by sorry

end prime_iff_totient_and_divisor_sum_condition_l1166_116687


namespace arithmetic_sequence_sum_l1166_116698

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of specific terms in the sequence equals 300 -/
def SumCondition (a : ℕ → ℝ) : Prop :=
  a 3 + a 4 + a 5 + a 7 + a 8 + a 9 = 300

theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h1 : ArithmeticSequence a) (h2 : SumCondition a) : 
  a 2 + a 10 = 100 := by
  sorry

end arithmetic_sequence_sum_l1166_116698


namespace bobs_driving_speed_l1166_116629

/-- Bob's driving problem -/
theorem bobs_driving_speed (initial_speed : ℝ) (initial_time : ℝ) (construction_time : ℝ) (total_time : ℝ) (total_distance : ℝ) :
  initial_speed = 60 →
  initial_time = 1.5 →
  construction_time = 2 →
  total_time = 3.5 →
  total_distance = 180 →
  (initial_speed * initial_time + construction_time * ((total_distance - initial_speed * initial_time) / construction_time) = total_distance) →
  (total_distance - initial_speed * initial_time) / construction_time = 45 :=
by sorry

end bobs_driving_speed_l1166_116629


namespace tim_running_hours_l1166_116604

def days_per_week : ℕ := 7

def previous_running_days : ℕ := 3
def added_running_days : ℕ := 2
def hours_per_run : ℕ := 2

def total_running_days : ℕ := previous_running_days + added_running_days
def total_running_hours : ℕ := total_running_days * hours_per_run

theorem tim_running_hours : total_running_hours = 10 := by
  sorry

end tim_running_hours_l1166_116604


namespace roller_coaster_rides_l1166_116605

def initial_tickets : ℕ := 287
def spent_tickets : ℕ := 134
def earned_tickets : ℕ := 32
def cost_per_ride : ℕ := 17

theorem roller_coaster_rides : 
  (initial_tickets - spent_tickets + earned_tickets) / cost_per_ride = 10 := by
  sorry

end roller_coaster_rides_l1166_116605


namespace smallest_x_value_l1166_116695

theorem smallest_x_value (x y : ℝ) : 
  4 ≤ x ∧ x < 6 →
  6 < y ∧ y < 10 →
  (∃ (n : ℤ), n = ⌊y - x⌋ ∧ n ≤ 5 ∧ ∀ (m : ℤ), m = ⌊y - x⌋ → m ≤ n) →
  x ≥ 4 ∧ ∀ (z : ℝ), (4 ≤ z ∧ z < 6 ∧ 
    (∃ (w : ℝ), 6 < w ∧ w < 10 ∧ 
      (∃ (n : ℤ), n = ⌊w - z⌋ ∧ n ≤ 5 ∧ ∀ (m : ℤ), m = ⌊w - z⌋ → m ≤ n))) →
    z ≥ x :=
by sorry

end smallest_x_value_l1166_116695


namespace sphere_volume_ratio_l1166_116684

theorem sphere_volume_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (4 * Real.pi * r₁^2) / (4 * Real.pi * r₂^2) = 1 / 4 →
  ((4 / 3) * Real.pi * r₁^3) / ((4 / 3) * Real.pi * r₂^3) = 1 / 8 := by
  sorry

end sphere_volume_ratio_l1166_116684


namespace line_angle_of_inclination_l1166_116653

/-- The angle of inclination of the line 2x + 2y - 5 = 0 is 135° -/
theorem line_angle_of_inclination :
  let line := {(x, y) : ℝ × ℝ | 2*x + 2*y - 5 = 0}
  ∃ α : Real, α = 135 * (π / 180) ∧ 
    ∀ (x y : ℝ), (x, y) ∈ line → (Real.tan α = -1) :=
by sorry

end line_angle_of_inclination_l1166_116653


namespace sally_lemonade_sales_l1166_116694

/-- Calculates the total number of lemonade cups sold over two weeks -/
def total_lemonade_cups (last_week : ℕ) (percent_increase : ℕ) : ℕ :=
  let this_week := last_week + (last_week * percent_increase) / 100
  last_week + this_week

/-- Proves that given the conditions, Sally sold 46 cups of lemonade in total -/
theorem sally_lemonade_sales : total_lemonade_cups 20 30 = 46 := by
  sorry

end sally_lemonade_sales_l1166_116694


namespace rain_probability_l1166_116685

/-- The probability of rain on three consecutive days --/
theorem rain_probability (p_sat p_sun p_mon_given_sat : ℝ) 
  (h_sat : p_sat = 0.7)
  (h_sun : p_sun = 0.5)
  (h_mon_given_sat : p_mon_given_sat = 0.4) :
  p_sat * p_sun * p_mon_given_sat = 0.14 := by
  sorry

end rain_probability_l1166_116685


namespace custom_op_two_three_custom_op_nested_l1166_116646

-- Define the custom operation
def custom_op (a b : ℝ) : ℝ := a^2 - b + a*b

-- Theorem 1: 2 * 3 = 7
theorem custom_op_two_three : custom_op 2 3 = 7 := by sorry

-- Theorem 2: (-2) * [2 * (-3)] = 1
theorem custom_op_nested : custom_op (-2) (custom_op 2 (-3)) = 1 := by sorry

end custom_op_two_three_custom_op_nested_l1166_116646


namespace blue_preference_percentage_l1166_116608

def total_responses : ℕ := 70 + 80 + 50 + 70 + 30

def blue_responses : ℕ := 80

def percentage_blue : ℚ := blue_responses / total_responses * 100

theorem blue_preference_percentage :
  percentage_blue = 80 / 300 * 100 :=
by sorry

end blue_preference_percentage_l1166_116608


namespace max_tiles_on_floor_l1166_116610

/-- Represents the dimensions of a rectangle -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the maximum number of tiles that can fit on a floor -/
def maxTiles (floor : Dimensions) (tile : Dimensions) : ℕ :=
  let horizontal := (floor.length / tile.length) * (floor.width / tile.width)
  let vertical := (floor.length / tile.width) * (floor.width / tile.length)
  max horizontal vertical

/-- Theorem stating the maximum number of tiles on the given floor -/
theorem max_tiles_on_floor :
  let floor : Dimensions := ⟨100, 150⟩
  let tile : Dimensions := ⟨20, 30⟩
  maxTiles floor tile = 25 := by
  sorry

#eval maxTiles ⟨100, 150⟩ ⟨20, 30⟩

end max_tiles_on_floor_l1166_116610


namespace deposit_equals_3400_l1166_116614

/-- Sheela's monthly income in rupees -/
def monthly_income : ℚ := 22666.67

/-- The percentage of monthly income deposited -/
def deposit_percentage : ℚ := 15

/-- The amount deposited in the bank savings account -/
def deposit_amount : ℚ := (deposit_percentage / 100) * monthly_income

/-- Theorem stating that the deposit amount is equal to 3400 rupees -/
theorem deposit_equals_3400 : deposit_amount = 3400 := by
  sorry

end deposit_equals_3400_l1166_116614


namespace isosceles_triangle_perimeter_l1166_116617

theorem isosceles_triangle_perimeter (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- sides are positive
  (a = 4 ∧ b = 8) ∨ (a = 8 ∧ b = 4) →  -- two sides are 4cm and 8cm
  (a = b ∨ b = c ∨ a = c) →  -- isosceles condition
  a + b + c = 20 :=  -- perimeter is 20cm
by sorry

end isosceles_triangle_perimeter_l1166_116617


namespace trihedral_acute_angles_l1166_116600

/-- A trihedral angle is an angle formed by three planes meeting at a point. -/
structure TrihedralAngle where
  /-- The three plane angles of the trihedral angle -/
  planeAngles : Fin 3 → ℝ
  /-- The three dihedral angles of the trihedral angle -/
  dihedralAngles : Fin 3 → ℝ

/-- A predicate to check if an angle is acute -/
def isAcute (angle : ℝ) : Prop := 0 < angle ∧ angle < Real.pi / 2

/-- The main theorem: if all dihedral angles of a trihedral angle are acute,
    then all its plane angles are also acute -/
theorem trihedral_acute_angles (t : TrihedralAngle) 
  (h : ∀ i : Fin 3, isAcute (t.dihedralAngles i)) :
  ∀ i : Fin 3, isAcute (t.planeAngles i) := by
  sorry

end trihedral_acute_angles_l1166_116600


namespace sum_of_coefficients_is_zero_l1166_116652

-- Define the functions f and g
def f (A B C x : ℝ) : ℝ := A * x + B + C
def g (A B C x : ℝ) : ℝ := B * x + A - C

-- State the theorem
theorem sum_of_coefficients_is_zero 
  (A B C : ℝ) 
  (h1 : A ≠ B) 
  (h2 : C ≠ 0) 
  (h3 : ∀ x, f A B C (g A B C x) - g A B C (f A B C x) = 2 * C) : 
  A + B = 0 := by
sorry

end sum_of_coefficients_is_zero_l1166_116652


namespace mixed_oil_rate_l1166_116696

/-- The rate of mixed oil per litre given specific quantities and prices of three types of oil -/
theorem mixed_oil_rate (quantity1 quantity2 quantity3 : ℚ) (price1 price2 price3 : ℚ) : 
  quantity1 = 12 ∧ quantity2 = 8 ∧ quantity3 = 4 ∧
  price1 = 55 ∧ price2 = 70 ∧ price3 = 82 →
  (quantity1 * price1 + quantity2 * price2 + quantity3 * price3) / (quantity1 + quantity2 + quantity3) = 64.5 := by
  sorry

#check mixed_oil_rate

end mixed_oil_rate_l1166_116696


namespace ellipse_equation_l1166_116691

theorem ellipse_equation (x y : ℝ) : 
  (∃ m n : ℝ, m > 0 ∧ n > 0 ∧ m ≠ n ∧
    m * 2^2 + n * Real.sqrt 2^2 = 1 ∧
    m * (Real.sqrt 2)^2 + n * (Real.sqrt 3)^2 = 1) →
  (x^2 / 8 + y^2 / 4 = 1 ↔ m * x^2 + n * y^2 = 1) :=
by sorry

end ellipse_equation_l1166_116691


namespace M_equals_N_l1166_116640

def M : Set ℝ := {x | ∃ m : ℤ, x = Real.sin ((2 * m - 3) * Real.pi / 6)}

def N : Set ℝ := {y | ∃ n : ℤ, y = Real.cos (n * Real.pi / 3)}

theorem M_equals_N : M = N := by sorry

end M_equals_N_l1166_116640


namespace inequality_solution_implies_m_value_l1166_116690

-- Define the inequality function
def inequality (m : ℝ) (x : ℝ) : Prop :=
  m * (x - 1) > x^2 - x

-- Define the solution set
def solution_set (m : ℝ) : Set ℝ :=
  {x | 1 < x ∧ x < 2}

-- Theorem statement
theorem inequality_solution_implies_m_value :
  ∀ m : ℝ, (∀ x : ℝ, inequality m x ↔ x ∈ solution_set m) → m = 2 :=
by sorry

end inequality_solution_implies_m_value_l1166_116690


namespace tan_alpha_plus_pi_fourth_l1166_116649

theorem tan_alpha_plus_pi_fourth (α β : ℝ) 
  (h1 : Real.tan (α + β) = 2/5)
  (h2 : Real.tan β = 1/3) : 
  Real.tan (α + π/4) = 9/8 := by
  sorry

end tan_alpha_plus_pi_fourth_l1166_116649


namespace exponential_decreasing_base_less_than_one_l1166_116609

theorem exponential_decreasing_base_less_than_one
  (m n : ℝ) (h1 : m > n) (h2 : n > 0) :
  (0.3 : ℝ) ^ m < (0.3 : ℝ) ^ n :=
by sorry

end exponential_decreasing_base_less_than_one_l1166_116609


namespace f_properties_l1166_116676

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)

theorem f_properties :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x : ℝ, deriv f x > 0) ∧
  (∀ k : ℝ, (∀ x : ℝ, x > 0 → f x > k * x) ↔ k ≤ 2) :=
by sorry

end f_properties_l1166_116676


namespace solve_linear_equation_l1166_116625

theorem solve_linear_equation (x : ℝ) : 3*x - 5*x + 7*x = 210 → x = 42 := by
  sorry

end solve_linear_equation_l1166_116625


namespace no_real_solutions_to_equation_l1166_116668

theorem no_real_solutions_to_equation :
  ¬∃ x : ℝ, x ≠ 0 ∧ x ≠ 4 ∧ (3 * x^2 - 15 * x) / (x^2 - 4 * x) = x - 2 := by
  sorry

end no_real_solutions_to_equation_l1166_116668


namespace tangent_sum_inequality_l1166_116699

-- Define an acute triangle
structure AcuteTriangle where
  A : Real
  B : Real
  C : Real
  acute : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π
  
-- Define the perimeter and inradius
def perimeter (t : AcuteTriangle) : Real := sorry
def inradius (t : AcuteTriangle) : Real := sorry

-- State the theorem
theorem tangent_sum_inequality (t : AcuteTriangle) :
  Real.tan t.A + Real.tan t.B + Real.tan t.C ≥ perimeter t / (2 * inradius t) := by
  sorry

end tangent_sum_inequality_l1166_116699


namespace unique_solution_factorial_equation_l1166_116620

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n+1 => (n+1) * factorial n

theorem unique_solution_factorial_equation : 
  ∃! n : ℕ, n * factorial n + 2 * factorial n = 5040 ∧ n = 5 := by
  sorry

end unique_solution_factorial_equation_l1166_116620


namespace complex_fraction_simplification_l1166_116658

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (i^3) / (1 - i) = (1 / 2 : ℂ) - (1 / 2 : ℂ) * i :=
by sorry

end complex_fraction_simplification_l1166_116658


namespace numerator_increase_percentage_l1166_116681

theorem numerator_increase_percentage (original_fraction : ℚ) 
  (denominator_decrease : ℚ) (new_fraction : ℚ) : 
  original_fraction = 3/4 →
  denominator_decrease = 8/100 →
  new_fraction = 15/16 →
  ∃ numerator_increase : ℚ, 
    (original_fraction * (1 + numerator_increase)) / (1 - denominator_decrease) = new_fraction ∧
    numerator_increase = 15/100 := by
  sorry

end numerator_increase_percentage_l1166_116681


namespace arithmetic_sequence_proof_l1166_116627

def arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_proof (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a)
  (h2 : a 5 = 10)
  (h3 : a 1 + a 2 + a 3 = 3) :
  a 1 = -2 ∧ ∃ d : ℝ, d = 3 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
by sorry

end arithmetic_sequence_proof_l1166_116627


namespace tropical_fish_count_l1166_116633

theorem tropical_fish_count (total : ℕ) (koi : ℕ) (h1 : total = 52) (h2 : koi = 37) :
  total - koi = 15 := by
  sorry

end tropical_fish_count_l1166_116633


namespace coin_count_l1166_116670

theorem coin_count (total_sum : ℕ) (coin_type1 coin_type2 : ℕ) (count_type1 : ℕ) :
  total_sum = 7100 →
  coin_type1 = 20 →
  coin_type2 = 25 →
  count_type1 = 290 →
  count_type1 * coin_type1 + (total_sum - count_type1 * coin_type1) / coin_type2 = 342 :=
by sorry

end coin_count_l1166_116670


namespace intersection_of_A_and_B_l1166_116635

def A : Set ℚ := {1, 2, 1/2}

def B : Set ℚ := {y | ∃ x ∈ A, y = x^2}

theorem intersection_of_A_and_B : A ∩ B = {1} := by
  sorry

end intersection_of_A_and_B_l1166_116635


namespace biology_magnet_problem_l1166_116644

def word : Finset Char := {'B', 'I', 'O', 'L', 'O', 'G', 'Y'}
def vowels : Finset Char := {'I', 'O', 'Y'}
def consonants : Finset Char := {'B', 'L', 'G'}

def distinct_collections : ℕ := sorry

theorem biology_magnet_problem :
  (word.card = 7) →
  (vowels ⊆ word) →
  (consonants ⊆ word) →
  (vowels ∩ consonants = ∅) →
  (vowels ∪ consonants = word) →
  (distinct_collections = 12) := by sorry

end biology_magnet_problem_l1166_116644


namespace right_triangle_cube_sides_l1166_116631

theorem right_triangle_cube_sides : ∃ (x : ℝ), 
  let a := x^3
  let b := x^3 - x
  let c := x^3 + x
  a^2 + b^2 = c^2 ∧ a = 8 ∧ b = 6 ∧ c = 10 := by
  sorry

end right_triangle_cube_sides_l1166_116631


namespace factorial_ratio_l1166_116612

theorem factorial_ratio : Nat.factorial 10 / (Nat.factorial 7 * Nat.factorial 2) = 360 := by
  sorry

end factorial_ratio_l1166_116612


namespace range_of_fraction_l1166_116639

theorem range_of_fraction (a b : ℝ) 
  (ha : -6 < a ∧ a < 8) 
  (hb : 2 < b ∧ b < 3) : 
  -3 < a/b ∧ a/b < 4 := by
  sorry

end range_of_fraction_l1166_116639


namespace sin_70_degrees_l1166_116647

theorem sin_70_degrees (a : ℝ) (h : Real.sin (10 * π / 180) = a) : 
  Real.sin (70 * π / 180) = 1 - 2 * a^2 := by
  sorry

end sin_70_degrees_l1166_116647


namespace logarithmic_expression_equality_algebraic_expression_equality_l1166_116660

-- Part 1
theorem logarithmic_expression_equality : 
  2 * Real.log 2 / Real.log 3 - Real.log (32 / 9) / Real.log 3 + Real.log 8 / Real.log 3 - 25 ^ (Real.log 3 / Real.log 5) = -7 := by sorry

-- Part 2
theorem algebraic_expression_equality : 
  (9 / 4) ^ (1 / 2) - (-7.8) ^ 0 - (27 / 8) ^ (2 / 3) + (2 / 3) ^ (-2) = 1 / 2 := by sorry

end logarithmic_expression_equality_algebraic_expression_equality_l1166_116660


namespace number_divisibility_l1166_116661

theorem number_divisibility (n m : ℤ) : 
  n = 859622 ∧ m = 859560 → 
  ∃ k : ℤ, k ≠ 0 ∧ m = n + (-62) ∧ m % k = 0 :=
by sorry

end number_divisibility_l1166_116661


namespace girls_joined_correct_l1166_116693

/-- The number of girls who joined the school -/
def girls_joined : ℕ := 465

/-- The initial number of girls in the school -/
def initial_girls : ℕ := 632

/-- The initial number of boys in the school -/
def initial_boys : ℕ := 410

/-- The difference between the number of girls and boys after some girls joined -/
def girl_boy_difference : ℕ := 687

theorem girls_joined_correct :
  initial_girls + girls_joined = initial_boys + girl_boy_difference :=
by sorry

end girls_joined_correct_l1166_116693


namespace solve_linear_equation_l1166_116663

theorem solve_linear_equation (y : ℚ) (h : -3 * y - 9 = 6 * y + 3) : y = -4/3 := by
  sorry

end solve_linear_equation_l1166_116663


namespace ball_problem_l1166_116665

/-- Represents the contents of a box with balls of two colors -/
structure Box where
  white : ℕ
  red : ℕ

/-- Represents the random variable X (number of red balls drawn from box A) -/
inductive X
  | zero
  | one
  | two

def box_A : Box := { white := 2, red := 2 }
def box_B : Box := { white := 1, red := 3 }

def prob_X (x : X) : ℚ :=
  match x with
  | X.zero => 1/6
  | X.one => 2/3
  | X.two => 1/6

def expected_X : ℚ := 1

def prob_red_from_B : ℚ := 2/3

theorem ball_problem :
  (∀ x : X, prob_X x > 0) ∧ 
  (prob_X X.zero + prob_X X.one + prob_X X.two = 1) ∧
  (0 * prob_X X.zero + 1 * prob_X X.one + 2 * prob_X X.two = expected_X) ∧
  prob_red_from_B = 2/3 := by sorry

end ball_problem_l1166_116665


namespace sarah_connor_wage_ratio_l1166_116697

def connors_hourly_wage : ℝ := 7.20
def sarahs_daily_wage : ℝ := 288
def work_hours : ℕ := 8

theorem sarah_connor_wage_ratio :
  (sarahs_daily_wage / work_hours) / connors_hourly_wage = 5 := by sorry

end sarah_connor_wage_ratio_l1166_116697


namespace initial_investment_interest_rate_l1166_116683

/-- Given an initial investment and an additional investment with their respective interest rates,
    proves that the interest rate of the initial investment is 5% when the total annual income
    equals 6% of the entire investment. -/
theorem initial_investment_interest_rate
  (initial_investment : ℝ)
  (additional_investment : ℝ)
  (additional_rate : ℝ)
  (total_rate : ℝ)
  (h1 : initial_investment = 3000)
  (h2 : additional_investment = 1499.9999999999998)
  (h3 : additional_rate = 0.08)
  (h4 : total_rate = 0.06)
  (h5 : ∃ r : ℝ, initial_investment * r + additional_investment * additional_rate =
                 (initial_investment + additional_investment) * total_rate) :
  ∃ r : ℝ, r = 0.05 ∧
    initial_investment * r + additional_investment * additional_rate =
    (initial_investment + additional_investment) * total_rate :=
sorry

end initial_investment_interest_rate_l1166_116683


namespace symmetric_points_range_l1166_116638

-- Define the functions f and g
def f (a x : ℝ) : ℝ := a - x^2
def g (x : ℝ) : ℝ := x + 1

-- Define the theorem
theorem symmetric_points_range (a : ℝ) :
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ ∃ y : ℝ, f a x = -g y) →
  -1 ≤ a ∧ a ≤ 1 :=
by sorry

end symmetric_points_range_l1166_116638


namespace clean_city_workers_l1166_116650

/-- The number of people in Lizzie's group -/
def lizzies_group : ℕ := 54

/-- The difference in members between Lizzie's group and the other group -/
def difference : ℕ := 17

/-- The total number of people working together to clean the city -/
def total_people : ℕ := lizzies_group + (lizzies_group - difference)

/-- Theorem stating that the total number of people working together is 91 -/
theorem clean_city_workers : total_people = 91 := by sorry

end clean_city_workers_l1166_116650


namespace father_eats_four_papayas_l1166_116655

/-- The number of papayas Jake eats in one week -/
def jake_papayas : ℕ := 3

/-- The number of papayas Jake's brother eats in one week -/
def brother_papayas : ℕ := 5

/-- The number of weeks Jake is planning for -/
def weeks : ℕ := 4

/-- The total number of papayas Jake needs to buy for 4 weeks -/
def total_papayas : ℕ := 48

/-- The number of papayas Jake's father eats in one week -/
def father_papayas : ℕ := (total_papayas - (jake_papayas + brother_papayas) * weeks) / weeks

theorem father_eats_four_papayas : father_papayas = 4 := by
  sorry

end father_eats_four_papayas_l1166_116655


namespace bank_deposit_duration_l1166_116601

theorem bank_deposit_duration (initial_deposit : ℝ) (interest_rate : ℝ) (final_amount : ℝ) :
  initial_deposit = 5600 →
  interest_rate = 0.07 →
  final_amount = 6384 →
  (final_amount - initial_deposit) / (interest_rate * initial_deposit) = 2 := by
  sorry

end bank_deposit_duration_l1166_116601


namespace sixth_term_value_l1166_116624

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- The main theorem -/
theorem sixth_term_value (seq : ArithmeticSequence) 
    (first_term : seq.a 1 = 1)
    (sum_5 : seq.S 5 = 15) :
  seq.a 6 = 6 := by
  sorry


end sixth_term_value_l1166_116624


namespace sqrt_square_eq_x_for_nonneg_l1166_116689

theorem sqrt_square_eq_x_for_nonneg (x : ℝ) (h : x ≥ 0) : (Real.sqrt x)^2 = x := by
  sorry

end sqrt_square_eq_x_for_nonneg_l1166_116689


namespace train_speed_problem_l1166_116674

/-- The speed of the second train given the conditions of the problem -/
theorem train_speed_problem (initial_distance : ℝ) (speed_train1 : ℝ) (distance_before_meet : ℝ) :
  initial_distance = 120 →
  speed_train1 = 30 →
  distance_before_meet = 70 →
  ∃ (speed_train2 : ℝ), 
    speed_train2 = 40 ∧ 
    (speed_train1 + speed_train2) * 1 = distance_before_meet ∧
    initial_distance - distance_before_meet = (speed_train1 + speed_train2) * 1 :=
by
  sorry

#check train_speed_problem

end train_speed_problem_l1166_116674


namespace women_in_room_l1166_116682

theorem women_in_room (initial_men : ℕ) (initial_women : ℕ) : 
  initial_men * 5 = initial_women * 4 →
  (initial_men + 2) = 14 →
  24 = 2 * (initial_women - 3) :=
by
  sorry

end women_in_room_l1166_116682


namespace sqrt_2_irrational_l1166_116671

theorem sqrt_2_irrational : ¬ ∃ (p q : ℤ), q ≠ 0 ∧ Real.sqrt 2 = (p : ℚ) / q := by
  sorry

end sqrt_2_irrational_l1166_116671


namespace students_left_early_l1166_116692

theorem students_left_early (original_groups : ℕ) (students_per_group : ℕ) (remaining_students : ℕ)
  (h1 : original_groups = 3)
  (h2 : students_per_group = 8)
  (h3 : remaining_students = 22) :
  original_groups * students_per_group - remaining_students = 2 := by
  sorry

end students_left_early_l1166_116692


namespace series_sum_equals_one_fourth_l1166_116636

/-- The sum of the infinite series Σ(n=1 to ∞) [3^n / (1 + 3^n + 3^(n+1) + 3^(2n+2))] is equal to 1/4. -/
theorem series_sum_equals_one_fourth :
  ∑' n : ℕ, (3 : ℝ)^n / (1 + 3^n + 3^(n+1) + 3^(2*n+2)) = 1/4 := by
  sorry

end series_sum_equals_one_fourth_l1166_116636


namespace sandy_final_position_l1166_116680

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define Sandy's walk
def sandy_walk (start : Point) : Point :=
  let p1 : Point := ⟨start.x, start.y - 20⟩  -- 20 meters south
  let p2 : Point := ⟨p1.x + 20, p1.y⟩        -- 20 meters east
  let p3 : Point := ⟨p2.x, p2.y + 20⟩        -- 20 meters north
  let p4 : Point := ⟨p3.x + 10, p3.y⟩        -- 10 meters east
  p4

-- Theorem stating that Sandy ends up 10 meters east of her starting point
theorem sandy_final_position (start : Point) : 
  (sandy_walk start).x - start.x = 10 ∧ (sandy_walk start).y = start.y :=
by sorry

end sandy_final_position_l1166_116680


namespace no_such_function_exists_l1166_116626

theorem no_such_function_exists :
  ¬∃ (f : ℤ → ℤ), ∀ (x y : ℤ), f (x + f y) = f x - y := by
  sorry

end no_such_function_exists_l1166_116626


namespace arithmetic_sequence_geometric_mean_l1166_116619

/-- 
Given an arithmetic sequence {a_n} with non-zero common difference d, 
where a_1 = 2d, if a_k is the geometric mean of a_1 and a_{2k+1}, then k = 3.
-/
theorem arithmetic_sequence_geometric_mean (d : ℝ) (k : ℕ) (a : ℕ → ℝ) :
  d ≠ 0 →
  (∀ n, a (n + 1) - a n = d) →
  a 1 = 2 * d →
  a k ^ 2 = a 1 * a (2 * k + 1) →
  k = 3 := by
  sorry

end arithmetic_sequence_geometric_mean_l1166_116619


namespace servant_payment_proof_l1166_116637

/-- Calculates the cash payment for a servant who leaves early -/
def servant_payment (total_salary : ℚ) (turban_value : ℚ) (months_worked : ℚ) : ℚ :=
  (months_worked / 12) * total_salary - turban_value

/-- Proves that a servant working 9 months with given conditions receives Rs. 60 -/
theorem servant_payment_proof :
  let total_salary : ℚ := 120
  let turban_value : ℚ := 30
  let months_worked : ℚ := 9
  servant_payment total_salary turban_value months_worked = 60 := by
sorry

end servant_payment_proof_l1166_116637


namespace certain_number_proof_l1166_116666

theorem certain_number_proof (x : ℝ) : 0.60 * x = (4 / 5) * 25 + 4 → x = 40 := by
  sorry

end certain_number_proof_l1166_116666


namespace area_triangle_ABC_l1166_116679

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the line m
def line_m : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 0}

-- Define the circles A, B, and C
def circle_A : Circle := ⟨(-7, 3), 3⟩
def circle_B : Circle := ⟨(0, -4), 4⟩
def circle_C : Circle := ⟨(9, 5), 5⟩

-- Define the tangent points
def point_A' : ℝ × ℝ := (-7, 0)
def point_B' : ℝ × ℝ := (0, 0)
def point_C' : ℝ × ℝ := (9, 0)

-- Define the properties of the circles and their arrangement
axiom tangent_to_m : 
  circle_A.center.2 = circle_A.radius ∧
  circle_B.center.2 = -circle_B.radius ∧
  circle_C.center.2 = circle_C.radius

axiom external_tangency :
  (circle_A.center.1 - circle_B.center.1)^2 + (circle_A.center.2 - circle_B.center.2)^2 
    = (circle_A.radius + circle_B.radius)^2 ∧
  (circle_C.center.1 - circle_B.center.1)^2 + (circle_C.center.2 - circle_B.center.2)^2 
    = (circle_C.radius + circle_B.radius)^2

axiom B'_between_A'_C' :
  point_A'.1 < point_B'.1 ∧ point_B'.1 < point_C'.1

-- Theorem to prove
theorem area_triangle_ABC : 
  let A := circle_A.center
  let B := circle_B.center
  let C := circle_C.center
  abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2) = 63 := by
  sorry

end area_triangle_ABC_l1166_116679


namespace p_satisfies_conditions_l1166_116669

/-- The polynomial p(x) that satisfies the given conditions -/
def p (x : ℝ) : ℝ := x^2 + 1

/-- Theorem stating that p(x) satisfies the required conditions -/
theorem p_satisfies_conditions :
  (p 3 = 10) ∧
  (∀ x y : ℝ, p x * p y = p x + p y + p (x * y) - 3) := by
  sorry

end p_satisfies_conditions_l1166_116669


namespace article_cost_price_l1166_116613

/-- The cost price of an article given its marked price and profit percentages -/
theorem article_cost_price (marked_price : ℝ) (discount_percent : ℝ) (profit_percent : ℝ) : 
  marked_price = 87.5 → 
  discount_percent = 5 → 
  profit_percent = 25 → 
  (1 - discount_percent / 100) * marked_price = (1 + profit_percent / 100) * (marked_price * (1 - discount_percent / 100) / (1 + profit_percent / 100)) → 
  marked_price * (1 - discount_percent / 100) / (1 + profit_percent / 100) = 66.5 := by
sorry

end article_cost_price_l1166_116613


namespace angle_sum_is_pi_over_four_l1166_116648

theorem angle_sum_is_pi_over_four (α β : Real) : 
  0 < α ∧ α < π/2 →  -- α is acute
  0 < β ∧ β < π/2 →  -- β is acute
  Real.tan α = 1/7 →
  Real.sin β = Real.sqrt 10/10 →
  α + 2*β = π/4 := by
sorry

end angle_sum_is_pi_over_four_l1166_116648


namespace photo_arrangements_eq_288_l1166_116603

/-- The number of ways to arrange teachers and students in a photo. -/
def photoArrangements (numTeachers numMaleStudents numFemaleStudents : ℕ) : ℕ :=
  2 * (numMaleStudents.factorial * numFemaleStudents.factorial * (numFemaleStudents + 1).choose numMaleStudents)

/-- Theorem stating the number of photo arrangements under given conditions. -/
theorem photo_arrangements_eq_288 :
  photoArrangements 2 3 3 = 288 := by
  sorry

end photo_arrangements_eq_288_l1166_116603


namespace arithmetic_sequence_sum_equality_l1166_116662

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Theorem statement
theorem arithmetic_sequence_sum_equality 
  (a : ℕ → ℝ) (h : is_arithmetic_sequence a) : 
  a 1 + a 8 = a 4 + a 5 := by sorry

end arithmetic_sequence_sum_equality_l1166_116662


namespace correct_average_l1166_116616

theorem correct_average (n : ℕ) (initial_avg : ℚ) (incorrect_num correct_num : ℚ) :
  n = 10 ∧ initial_avg = 46 ∧ incorrect_num = 25 ∧ correct_num = 75 →
  (n * initial_avg + (correct_num - incorrect_num)) / n = 51 :=
by sorry

end correct_average_l1166_116616


namespace pi_half_irrational_l1166_116677

theorem pi_half_irrational : Irrational (π / 2) :=
by
  sorry

end pi_half_irrational_l1166_116677


namespace cube_root_problem_l1166_116611

theorem cube_root_problem : (0.07 : ℝ)^3 = 0.000343 := by
  sorry

end cube_root_problem_l1166_116611


namespace jerry_shelf_comparison_l1166_116634

theorem jerry_shelf_comparison : 
  ∀ (initial_action_figures initial_books added_action_figures : ℕ),
    initial_action_figures = 5 →
    initial_books = 9 →
    added_action_figures = 7 →
    (initial_action_figures + added_action_figures) - initial_books = 3 :=
by
  sorry

end jerry_shelf_comparison_l1166_116634


namespace power_function_through_point_l1166_116628

theorem power_function_through_point (a : ℝ) :
  (2 : ℝ) ^ a = (1 / 2 : ℝ) → a = -1 := by
  sorry

end power_function_through_point_l1166_116628


namespace consecutive_integer_product_divisibility_l1166_116602

theorem consecutive_integer_product_divisibility (j : ℤ) : 
  let m := j * (j + 1) * (j + 2) * (j + 3)
  (∃ k : ℤ, m = 11 * k) →
  (∃ k : ℤ, m = 12 * k) ∧
  (∃ k : ℤ, m = 33 * k) ∧
  (∃ k : ℤ, m = 44 * k) ∧
  (∃ k : ℤ, m = 66 * k) ∧
  ¬(∀ j : ℤ, ∃ k : ℤ, m = 24 * k) :=
by sorry

end consecutive_integer_product_divisibility_l1166_116602


namespace probability_of_winning_pair_l1166_116643

def deck_size : ℕ := 10
def red_cards : ℕ := 5
def green_cards : ℕ := 5
def num_letters : ℕ := 5

def winning_pair_count : ℕ := num_letters + 2 * (red_cards.choose 2)

theorem probability_of_winning_pair :
  (winning_pair_count : ℚ) / (deck_size.choose 2) = 5 / 9 := by sorry

end probability_of_winning_pair_l1166_116643


namespace problem_1_problem_2_l1166_116678

-- Problem 1
theorem problem_1 : -1.5 + 1.4 - (-3.6) - 4.3 + (-5.2) = -6 := by
  sorry

-- Problem 2
theorem problem_2 : 17 - 2^3 / (-2) * 3 = 29 := by
  sorry

end problem_1_problem_2_l1166_116678


namespace complex_problem_l1166_116641

theorem complex_problem (a : ℝ) (z₁ : ℂ) (h₁ : a < 0) (h₂ : z₁ = 1 + a * Complex.I) 
  (h₃ : Complex.re (z₁^2) = 0) : 
  a = -1 ∧ Complex.abs ((z₁ / (1 + Complex.I)) + 2) = Real.sqrt 5 := by
  sorry

end complex_problem_l1166_116641


namespace exponential_function_property_l1166_116618

theorem exponential_function_property (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x ∈ Set.Icc 1 2, a^x ≤ a^2) ∧ 
  (∀ x ∈ Set.Icc 1 2, a^x ≥ a^1) ∧
  (a^2 - a^1 = a / 2) →
  a = 1/2 ∨ a = 3/2 :=
by sorry

end exponential_function_property_l1166_116618


namespace fishermen_distribution_l1166_116664

theorem fishermen_distribution (x y z : ℕ) : 
  x + y + z = 16 →
  13 * x + 5 * y + 4 * z = 113 →
  x = 5 ∧ y = 4 ∧ z = 7 := by
sorry

end fishermen_distribution_l1166_116664
