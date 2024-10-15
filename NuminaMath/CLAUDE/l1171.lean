import Mathlib

namespace NUMINAMATH_CALUDE_recommended_sleep_hours_l1171_117157

theorem recommended_sleep_hours (total_sleep : ℝ) (short_sleep : ℝ) (short_days : ℕ) 
  (normal_days : ℕ) (normal_sleep_percentage : ℝ) 
  (h1 : total_sleep = 30)
  (h2 : short_sleep = 3)
  (h3 : short_days = 2)
  (h4 : normal_days = 5)
  (h5 : normal_sleep_percentage = 0.6)
  (h6 : total_sleep = short_sleep * short_days + normal_sleep_percentage * normal_days * recommended_sleep) :
  recommended_sleep = 8 := by
  sorry

end NUMINAMATH_CALUDE_recommended_sleep_hours_l1171_117157


namespace NUMINAMATH_CALUDE_problem_statements_l1171_117121

theorem problem_statements :
  (∀ (p q : Prop), (p ∧ q) → ¬(¬p)) ∧
  (∃ (x : ℝ), x^2 - x - 1 < 0) ↔ ¬(∀ (x : ℝ), x^2 - x - 1 ≥ 0) ∧
  (∃ (a b : ℝ), (a + b > 0) ∧ ¬(a > 5 ∧ b > -5)) ∧
  (∀ (α : ℝ), α < 0 → ∀ (x₁ x₂ : ℝ), x₁ > 0 → x₂ > 0 → x₁ < x₂ → x₁^α > x₂^α) :=
by sorry

end NUMINAMATH_CALUDE_problem_statements_l1171_117121


namespace NUMINAMATH_CALUDE_journey_time_proof_l1171_117183

/-- Proves that the total journey time is 5 hours given the specified conditions -/
theorem journey_time_proof (total_distance : ℝ) (speed1 speed2 : ℝ) (time1 : ℝ) :
  total_distance = 240 ∧ 
  speed1 = 40 ∧ 
  speed2 = 60 ∧ 
  time1 = 3 →
  speed1 * time1 + (total_distance - speed1 * time1) / speed2 + time1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_journey_time_proof_l1171_117183


namespace NUMINAMATH_CALUDE_subtraction_division_fractions_l1171_117186

theorem subtraction_division_fractions : ((3 / 4 : ℚ) - (5 / 8 : ℚ)) / 2 = (1 / 16 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_subtraction_division_fractions_l1171_117186


namespace NUMINAMATH_CALUDE_lawn_mowing_earnings_l1171_117198

theorem lawn_mowing_earnings 
  (lawns_mowed : ℕ) 
  (initial_savings : ℕ) 
  (total_after_mowing : ℕ) 
  (h1 : lawns_mowed = 5)
  (h2 : initial_savings = 7)
  (h3 : total_after_mowing = 47) :
  (total_after_mowing - initial_savings) / lawns_mowed = 8 := by
  sorry

end NUMINAMATH_CALUDE_lawn_mowing_earnings_l1171_117198


namespace NUMINAMATH_CALUDE_warehouse_paint_area_l1171_117173

/-- Calculates the area to be painted in a rectangular warehouse with a door. -/
def areaToBePainted (length width height doorWidth doorHeight : ℝ) : ℝ :=
  2 * (length * height + width * height) - (doorWidth * doorHeight)

/-- Theorem stating the area to be painted for the given warehouse dimensions. -/
theorem warehouse_paint_area :
  areaToBePainted 8 6 3.5 1 2 = 96 := by
  sorry

end NUMINAMATH_CALUDE_warehouse_paint_area_l1171_117173


namespace NUMINAMATH_CALUDE_QY_eq_10_l1171_117104

/-- Circle with center O and radius r -/
structure Circle where
  O : ℝ × ℝ
  r : ℝ

/-- Point outside the circle -/
def Q : ℝ × ℝ := sorry

/-- Circle C -/
def C : Circle := sorry

/-- Points on the circle -/
def X : ℝ × ℝ := sorry
def Y : ℝ × ℝ := sorry
def Z : ℝ × ℝ := sorry

/-- Distances -/
def QX : ℝ := sorry
def QY : ℝ := sorry
def QZ : ℝ := sorry

/-- Q is outside C -/
axiom h_Q_outside : Q ∉ {p | (p.1 - C.O.1)^2 + (p.2 - C.O.2)^2 ≤ C.r^2}

/-- QZ is tangent to C at Z -/
axiom h_QZ_tangent : (Z.1 - C.O.1)^2 + (Z.2 - C.O.2)^2 = C.r^2 ∧
  ((Z.1 - Q.1) * (Z.1 - C.O.1) + (Z.2 - Q.2) * (Z.2 - C.O.2) = 0)

/-- X and Y are on C -/
axiom h_X_on_C : (X.1 - C.O.1)^2 + (X.2 - C.O.2)^2 = C.r^2
axiom h_Y_on_C : (Y.1 - C.O.1)^2 + (Y.2 - C.O.2)^2 = C.r^2

/-- QX < QY -/
axiom h_QX_lt_QY : QX < QY

/-- QX = 5 -/
axiom h_QX_eq_5 : QX = 5

/-- QZ = 2(QY - QX) -/
axiom h_QZ_eq : QZ = 2 * (QY - QX)

/-- Power of a Point theorem -/
axiom power_of_point : QX * QY = QZ^2

theorem QY_eq_10 : QY = 10 := by sorry

end NUMINAMATH_CALUDE_QY_eq_10_l1171_117104


namespace NUMINAMATH_CALUDE_area_between_circles_l1171_117133

/-- The area between two concentric circles -/
theorem area_between_circles (R r : ℝ) (h1 : R = 10) (h2 : r = 4) :
  (π * R^2) - (π * r^2) = 84 * π := by
  sorry

end NUMINAMATH_CALUDE_area_between_circles_l1171_117133


namespace NUMINAMATH_CALUDE_ceiling_sum_of_square_roots_l1171_117120

theorem ceiling_sum_of_square_roots : ⌈Real.sqrt 3⌉ + ⌈Real.sqrt 33⌉ + ⌈Real.sqrt 333⌉ = 27 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sum_of_square_roots_l1171_117120


namespace NUMINAMATH_CALUDE_base10_216_equals_base9_260_l1171_117160

/-- Converts a natural number from base 10 to base 9 --/
def toBase9 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 9 to a natural number in base 10 --/
def fromBase9 (digits : List ℕ) : ℕ :=
  sorry

/-- Checks if all digits in a list are less than 9 --/
def validBase9Digits (digits : List ℕ) : Prop :=
  ∀ d ∈ digits, d < 9

theorem base10_216_equals_base9_260 :
  let base9Digits := [2, 6, 0]
  validBase9Digits base9Digits ∧ fromBase9 base9Digits = 216 :=
by
  sorry

end NUMINAMATH_CALUDE_base10_216_equals_base9_260_l1171_117160


namespace NUMINAMATH_CALUDE_ten_points_chords_l1171_117187

/-- The number of chords connecting n points on a circle -/
def num_chords (n : ℕ) : ℕ :=
  if n ≤ 1 then 0 else (n - 1) * n / 2

/-- The property that the number of chords follows the observed pattern -/
axiom chord_pattern : 
  num_chords 2 = 1 ∧ 
  num_chords 3 = 3 ∧ 
  num_chords 4 = 6 ∧ 
  num_chords 5 = 10 ∧ 
  num_chords 6 = 15

/-- Theorem: The number of chords connecting 10 points on a circle is 45 -/
theorem ten_points_chords : num_chords 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_ten_points_chords_l1171_117187


namespace NUMINAMATH_CALUDE_fraction_of_boys_reading_l1171_117131

theorem fraction_of_boys_reading (total_girls : ℕ) (total_boys : ℕ) 
  (fraction_girls_reading : ℚ) (not_reading : ℕ) :
  total_girls = 12 →
  total_boys = 10 →
  fraction_girls_reading = 5/6 →
  not_reading = 4 →
  (total_boys - (not_reading - (total_girls - (fraction_girls_reading * total_girls).num))) / total_boys = 4/5 := by
  sorry


end NUMINAMATH_CALUDE_fraction_of_boys_reading_l1171_117131


namespace NUMINAMATH_CALUDE_distance_between_points_l1171_117162

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (1, 2)
  let p2 : ℝ × ℝ := (-3, -4)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1171_117162


namespace NUMINAMATH_CALUDE_smallest_multiplier_for_450_cube_l1171_117193

/-- Given a positive integer n, returns true if n is a perfect cube, false otherwise -/
def isPerfectCube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

/-- The smallest positive integer that, when multiplied by 450, results in a perfect cube -/
def smallestMultiplier : ℕ := 60

theorem smallest_multiplier_for_450_cube :
  (isPerfectCube (450 * smallestMultiplier)) ∧
  (∀ n : ℕ, 0 < n → n < smallestMultiplier → ¬(isPerfectCube (450 * n))) :=
sorry

end NUMINAMATH_CALUDE_smallest_multiplier_for_450_cube_l1171_117193


namespace NUMINAMATH_CALUDE_ball_ratio_problem_l1171_117175

theorem ball_ratio_problem (white_balls red_balls : ℕ) : 
  (white_balls : ℚ) / red_balls = 5 / 3 →
  white_balls = 15 →
  red_balls = 9 := by
sorry

end NUMINAMATH_CALUDE_ball_ratio_problem_l1171_117175


namespace NUMINAMATH_CALUDE_husband_age_difference_l1171_117189

-- Define the initial ages
def hannah_initial_age : ℕ := 6
def july_initial_age : ℕ := hannah_initial_age / 2

-- Define the time passed
def years_passed : ℕ := 20

-- Define July's current age
def july_current_age : ℕ := july_initial_age + years_passed

-- Define July's husband's age
def husband_age : ℕ := 25

-- Theorem to prove
theorem husband_age_difference : husband_age - july_current_age = 2 := by
  sorry

end NUMINAMATH_CALUDE_husband_age_difference_l1171_117189


namespace NUMINAMATH_CALUDE_square_of_binomial_theorem_l1171_117110

-- Define the expressions
def expr_A (x y : ℝ) := (x + y) * (x - y)
def expr_B (x y : ℝ) := (-x + y) * (x - y)
def expr_C (x y : ℝ) := (-x + y) * (-x - y)
def expr_D (x y : ℝ) := (-x + y) * (x + y)

-- Define what it means for an expression to be a square of a binomial
def is_square_of_binomial (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (g : ℝ → ℝ → ℝ), ∀ x y, f x y = (g x y)^2

-- State the theorem
theorem square_of_binomial_theorem :
  is_square_of_binomial expr_A ∧
  ¬(is_square_of_binomial expr_B) ∧
  is_square_of_binomial expr_C ∧
  is_square_of_binomial expr_D :=
sorry

end NUMINAMATH_CALUDE_square_of_binomial_theorem_l1171_117110


namespace NUMINAMATH_CALUDE_intersection_implies_sum_l1171_117136

-- Define the functions
def f (x a b : ℝ) : ℝ := -2 * abs (x - a) + b
def g (x c d : ℝ) : ℝ := 2 * abs (x - c) + d

-- State the theorem
theorem intersection_implies_sum (a b c d : ℝ) : 
  (f 1 a b = g 1 c d) ∧ (f 7 a b = g 7 c d) → a + c = 8 := by
  sorry


end NUMINAMATH_CALUDE_intersection_implies_sum_l1171_117136


namespace NUMINAMATH_CALUDE_regression_line_correct_l1171_117130

def points : List (ℝ × ℝ) := [(1, 2), (2, 3), (3, 4), (4, 5)]

def regression_line (points : List (ℝ × ℝ)) : ℝ → ℝ := 
  fun x => x + 1

theorem regression_line_correct : 
  regression_line points = fun x => x + 1 := by sorry

end NUMINAMATH_CALUDE_regression_line_correct_l1171_117130


namespace NUMINAMATH_CALUDE_equal_angle_implies_equal_side_l1171_117158

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Represents the orthocenter of a triangle -/
def orthocenter (t : Triangle) : Point :=
  sorry

/-- Reflects a point with respect to a line segment -/
def reflect (p : Point) (a : Point) (b : Point) : Point :=
  sorry

/-- Checks if two triangles have an equal angle -/
def have_equal_angle (t1 t2 : Triangle) : Prop :=
  sorry

/-- Checks if two triangles have an equal side -/
def have_equal_side (t1 t2 : Triangle) : Prop :=
  sorry

/-- Checks if a triangle is acute -/
def is_acute (t : Triangle) : Prop :=
  sorry

theorem equal_angle_implies_equal_side 
  (ABC : Triangle) 
  (h_acute : is_acute ABC) 
  (H : Point) 
  (h_ortho : H = orthocenter ABC) 
  (A' B' C' : Point) 
  (h_A' : A' = reflect H ABC.B ABC.C) 
  (h_B' : B' = reflect H ABC.C ABC.A) 
  (h_C' : C' = reflect H ABC.A ABC.B) 
  (A'B'C' : Triangle) 
  (h_A'B'C' : A'B'C' = Triangle.mk A' B' C') 
  (h_equal_angle : have_equal_angle ABC A'B'C') :
  have_equal_side ABC A'B'C' :=
sorry

end NUMINAMATH_CALUDE_equal_angle_implies_equal_side_l1171_117158


namespace NUMINAMATH_CALUDE_selection_methods_count_l1171_117184

def total_volunteers : ℕ := 8
def boys : ℕ := 5
def girls : ℕ := 3
def selection_size : ℕ := 3

theorem selection_methods_count : 
  (Nat.choose boys 2 * Nat.choose girls 1) + (Nat.choose boys 1 * Nat.choose girls 2) = 45 := by
  sorry

end NUMINAMATH_CALUDE_selection_methods_count_l1171_117184


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1171_117172

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)
  property1 : a 5 * a 11 = 3
  property2 : a 3 + a 13 = 4

/-- The theorem stating the possible values of a_15 / a_5 -/
theorem geometric_sequence_ratio (seq : GeometricSequence) :
  seq.a 15 / seq.a 5 = 1/3 ∨ seq.a 15 / seq.a 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1171_117172


namespace NUMINAMATH_CALUDE_weight_difference_proof_l1171_117178

/-- Proves that the difference between the average weight of two departing students
    and Joe's weight is -6.5 kg, given the conditions of the original problem. -/
theorem weight_difference_proof (n : ℕ) (x : ℝ) : 
  -- Joe's weight
  let joe_weight : ℝ := 43
  -- Initial average weight
  let initial_avg : ℝ := 30
  -- New average weight after Joe joins
  let new_avg : ℝ := 31
  -- Number of students in original group
  n = (joe_weight - initial_avg) / (new_avg - initial_avg)
  -- Average weight of two departing students
  → x = (new_avg * (n + 1) - initial_avg * (n - 1)) / 2
  -- Difference between average weight of departing students and Joe's weight
  → x - joe_weight = -6.5 := by
  sorry

end NUMINAMATH_CALUDE_weight_difference_proof_l1171_117178


namespace NUMINAMATH_CALUDE_a_share_of_profit_is_correct_l1171_117196

/-- Calculates the share of profit for an investor in a partnership business. -/
def calculate_share_of_profit (investment_a investment_b investment_c total_profit : ℚ) : ℚ :=
  let total_investment := investment_a + investment_b + investment_c
  (investment_a / total_investment) * total_profit

/-- Theorem stating that A's share of the profit is correctly calculated. -/
theorem a_share_of_profit_is_correct (investment_a investment_b investment_c total_profit : ℚ) :
  calculate_share_of_profit investment_a investment_b investment_c total_profit =
  3750 / 1 :=
by sorry

end NUMINAMATH_CALUDE_a_share_of_profit_is_correct_l1171_117196


namespace NUMINAMATH_CALUDE_count_decompositions_l1171_117124

/-- The number of ways to write 4020 in the specified form -/
def M : ℕ := 40000

/-- A function that represents the decomposition of 4020 -/
def decomposition (b₃ b₂ b₁ b₀ : ℕ) : ℕ :=
  b₃ * 1000 + b₂ * 100 + b₁ * 10 + b₀

/-- The theorem stating that M is the correct count -/
theorem count_decompositions :
  M = (Finset.filter (fun (b : ℕ × ℕ × ℕ × ℕ) => 
    let (b₃, b₂, b₁, b₀) := b
    decomposition b₃ b₂ b₁ b₀ = 4020 ∧ 
    b₃ ≤ 99 ∧ b₂ ≤ 99 ∧ b₁ ≤ 99 ∧ b₀ ≤ 99)
    (Finset.product (Finset.range 100) 
      (Finset.product (Finset.range 100) 
        (Finset.product (Finset.range 100) (Finset.range 100))))).card :=
by
  sorry

end NUMINAMATH_CALUDE_count_decompositions_l1171_117124


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonal_l1171_117149

/-- Given a rectangular prism with dimensions a, b, and c, if the total surface area
    is 11 and the sum of all edge lengths is 24, then the length of the body diagonal is 5. -/
theorem rectangular_prism_diagonal (a b c : ℝ) 
  (h1 : 2 * (a * b + b * c + a * c) = 11)  -- total surface area
  (h2 : 4 * (a + b + c) = 24) :            -- sum of all edge lengths
  Real.sqrt (a^2 + b^2 + c^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonal_l1171_117149


namespace NUMINAMATH_CALUDE_square_sum_given_condition_l1171_117156

theorem square_sum_given_condition (x y : ℝ) :
  (2*x + 1)^2 + |y - 1| = 0 → x^2 + y^2 = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_condition_l1171_117156


namespace NUMINAMATH_CALUDE_hotel_flat_fee_calculation_l1171_117114

/-- A hotel charges a flat fee for the first night and a fixed amount for subsequent nights. -/
structure HotelPricing where
  flatFee : ℝ
  subsequentNightFee : ℝ

/-- Calculate the total cost for a given number of nights -/
def totalCost (pricing : HotelPricing) (nights : ℕ) : ℝ :=
  pricing.flatFee + pricing.subsequentNightFee * (nights - 1)

theorem hotel_flat_fee_calculation (pricing : HotelPricing) :
  totalCost pricing 4 = 185 ∧ totalCost pricing 8 = 350 → pricing.flatFee = 61.25 := by
  sorry

end NUMINAMATH_CALUDE_hotel_flat_fee_calculation_l1171_117114


namespace NUMINAMATH_CALUDE_line_satisfies_conditions_l1171_117100

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 + 4*x + 3

-- Define the line
def line (x : ℝ) : ℝ := 6*x - 4

-- Theorem statement
theorem line_satisfies_conditions :
  -- Condition 1: The line passes through (2, 8)
  (line 2 = 8) ∧
  -- Condition 2: There exists a k where x = k intersects both curves 4 units apart
  (∃ k : ℝ, |parabola k - line k| = 4) ∧
  -- Condition 3: The y-intercept is not 0
  (line 0 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_line_satisfies_conditions_l1171_117100


namespace NUMINAMATH_CALUDE_solution_first_equation_solutions_second_equation_l1171_117101

-- First equation
theorem solution_first_equation (x : ℝ) :
  27 * (x + 1)^3 = -64 ↔ x = -7/3 := by sorry

-- Second equation
theorem solutions_second_equation (x : ℝ) :
  (x + 1)^2 = 25 ↔ x = 4 ∨ x = -6 := by sorry

end NUMINAMATH_CALUDE_solution_first_equation_solutions_second_equation_l1171_117101


namespace NUMINAMATH_CALUDE_horner_method_first_step_l1171_117199

def f (x : ℝ) : ℝ := 0.5 * x^5 + 4 * x^4 - 3 * x^2 + x - 1

def horner_first_step (a₅ a₄ : ℝ) (x : ℝ) : ℝ := a₅ * x + a₄

theorem horner_method_first_step :
  horner_first_step 0.5 4 3 = 5.5 :=
sorry

end NUMINAMATH_CALUDE_horner_method_first_step_l1171_117199


namespace NUMINAMATH_CALUDE_sets_intersection_union_l1171_117161

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def B (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b ≤ 0}

-- State the theorem
theorem sets_intersection_union (a b : ℝ) : 
  (A ∪ B a b = Set.univ) ∧ (A ∩ B a b = {x | 3 < x ∧ x ≤ 4}) → a + b = -7 := by
  sorry

end NUMINAMATH_CALUDE_sets_intersection_union_l1171_117161


namespace NUMINAMATH_CALUDE_dog_movement_area_calculation_l1171_117115

/-- Represents the dimensions and constraints of a dog tied to a square doghouse --/
structure DogHouseSetup where
  side_length : ℝ
  tie_point_distance : ℝ
  chain_length : ℝ

/-- Calculates the area in which the dog can move --/
def dog_movement_area (setup : DogHouseSetup) : ℝ :=
  sorry

/-- Theorem stating the area in which the dog can move for the given setup --/
theorem dog_movement_area_calculation (ε : ℝ) (h_ε : ε > 0) :
  ∃ (setup : DogHouseSetup),
    setup.side_length = 1.2 ∧
    setup.tie_point_distance = 0.3 ∧
    setup.chain_length = 3 ∧
    |dog_movement_area setup - 23.693| < ε :=
  sorry

end NUMINAMATH_CALUDE_dog_movement_area_calculation_l1171_117115


namespace NUMINAMATH_CALUDE_remainder_equality_l1171_117148

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem remainder_equality : 
  (sum_factorials 20) % 21 = (sum_factorials 4) % 21 := by
sorry

end NUMINAMATH_CALUDE_remainder_equality_l1171_117148


namespace NUMINAMATH_CALUDE_no_solution_exists_l1171_117155

theorem no_solution_exists (x y : ℕ) (h : x > 1) : (x^7 - 1) / (x - 1) ≠ y^5 + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l1171_117155


namespace NUMINAMATH_CALUDE_a_greater_than_b_l1171_117145

def a : ℕ → ℕ
  | 0 => 0
  | n + 1 => a n ^ 2 + 3

def b : ℕ → ℕ
  | 0 => 0
  | n + 1 => b n ^ 2 + 2 ^ (n + 1)

theorem a_greater_than_b : b 2003 < a 2003 := by
  sorry

end NUMINAMATH_CALUDE_a_greater_than_b_l1171_117145


namespace NUMINAMATH_CALUDE_perfect_square_condition_l1171_117118

theorem perfect_square_condition (x : ℤ) : 
  (∃ y : ℤ, x^2 + 19*x + 95 = y^2) ↔ (x = -14 ∨ x = -5) := by
sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l1171_117118


namespace NUMINAMATH_CALUDE_x_values_theorem_l1171_117116

theorem x_values_theorem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1 / y = 10) (h2 : y + 1 / x = 5 / 12) :
  x = 4 ∨ x = 6 :=
by sorry

end NUMINAMATH_CALUDE_x_values_theorem_l1171_117116


namespace NUMINAMATH_CALUDE_quotient_divisible_by_five_l1171_117195

theorem quotient_divisible_by_five : ∃ k : ℤ, 4^1993 + 6^1993 = 5 * k := by
  sorry

end NUMINAMATH_CALUDE_quotient_divisible_by_five_l1171_117195


namespace NUMINAMATH_CALUDE_triangle_circle_area_l1171_117106

theorem triangle_circle_area (a : ℝ) (h : a > 0) : 
  let angle1 : ℝ := 45 * π / 180
  let angle2 : ℝ := 15 * π / 180
  let angle3 : ℝ := π - angle1 - angle2
  let height : ℝ := a * (Real.sqrt 3 - 1) / (2 * Real.sqrt 3)
  let circle_area : ℝ := π * height^2
  circle_area / 3 = π * a^2 * (2 - Real.sqrt 3) / 18 := by
sorry

end NUMINAMATH_CALUDE_triangle_circle_area_l1171_117106


namespace NUMINAMATH_CALUDE_ab_gt_ac_not_sufficient_nor_necessary_for_b_gt_c_l1171_117154

theorem ab_gt_ac_not_sufficient_nor_necessary_for_b_gt_c :
  ¬(∀ a b c : ℝ, a * b > a * c → b > c) ∧
  ¬(∀ a b c : ℝ, b > c → a * b > a * c) := by
  sorry

end NUMINAMATH_CALUDE_ab_gt_ac_not_sufficient_nor_necessary_for_b_gt_c_l1171_117154


namespace NUMINAMATH_CALUDE_eight_million_factorization_roundness_of_eight_million_l1171_117103

/-- Roundness of a positive integer is the sum of the exponents in its prime factorization -/
def roundness (n : ℕ+) : ℕ := sorry

/-- 8,000,000 can be expressed as 8 × 10^6 -/
theorem eight_million_factorization : (8000000 : ℕ) = 8 * 10^6 := by sorry

/-- The roundness of 8,000,000 is 15 -/
theorem roundness_of_eight_million : roundness 8000000 = 15 := by sorry

end NUMINAMATH_CALUDE_eight_million_factorization_roundness_of_eight_million_l1171_117103


namespace NUMINAMATH_CALUDE_min_triangle_area_l1171_117185

/-- Triangle DEF with vertices D(0,0) and E(24,10), and F having integer coordinates -/
structure Triangle where
  F : ℤ × ℤ

/-- Area of triangle DEF given coordinates of F -/
def triangleArea (t : Triangle) : ℚ :=
  let (x, y) := t.F
  (1 : ℚ) / 2 * |10 * x - 24 * y|

/-- The minimum non-zero area of triangle DEF is 5 -/
theorem min_triangle_area :
  ∃ (t : Triangle), triangleArea t > 0 ∧
  ∀ (t' : Triangle), triangleArea t' > 0 → triangleArea t ≤ triangleArea t' :=
sorry

end NUMINAMATH_CALUDE_min_triangle_area_l1171_117185


namespace NUMINAMATH_CALUDE_fraction_reduction_divisibility_l1171_117190

theorem fraction_reduction_divisibility
  (a b c d n : ℕ)
  (h1 : (a * n + b) % 2017 = 0)
  (h2 : (c * n + d) % 2017 = 0) :
  (a * d - b * c) % 2017 = 0 :=
by sorry

end NUMINAMATH_CALUDE_fraction_reduction_divisibility_l1171_117190


namespace NUMINAMATH_CALUDE_valid_parameterization_l1171_117125

/-- A vector parameterization of a line --/
structure VectorParam where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Checks if a vector parameterization is valid for the line y = 3x + 4 --/
def is_valid_param (p : VectorParam) : Prop :=
  p.b = 3 * p.a + 4 ∧ p.d = 3 * p.c

theorem valid_parameterization (p : VectorParam) :
  is_valid_param p ↔
    (∀ t : ℝ, (p.a + t * p.c, p.b + t * p.d) ∈ {(x, y) : ℝ × ℝ | y = 3 * x + 4}) :=
by sorry

end NUMINAMATH_CALUDE_valid_parameterization_l1171_117125


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1171_117151

-- Define the inequality system
def inequality_system (x : ℝ) : Prop :=
  (2 - x < 0) ∧ (-2 * x < 6)

-- Define the solution set
def solution_set : Set ℝ :=
  {x | x > 2}

-- Theorem statement
theorem inequality_system_solution :
  {x : ℝ | inequality_system x} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1171_117151


namespace NUMINAMATH_CALUDE_sequence_reappearance_l1171_117134

def letter_cycle_length : ℕ := 7
def digit_cycle_length : ℕ := 4

theorem sequence_reappearance :
  Nat.lcm letter_cycle_length digit_cycle_length = 28 := by
  sorry

#check sequence_reappearance

end NUMINAMATH_CALUDE_sequence_reappearance_l1171_117134


namespace NUMINAMATH_CALUDE_sqrt_equality_implies_t_value_l1171_117117

theorem sqrt_equality_implies_t_value :
  ∀ t : ℝ, (Real.sqrt (3 * Real.sqrt (t - 3)) = (10 - t) ^ (1/4)) → t = 37/10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equality_implies_t_value_l1171_117117


namespace NUMINAMATH_CALUDE_bobby_candy_problem_l1171_117108

theorem bobby_candy_problem (C : ℕ) : 
  (C + 36 = 16 + 58) → C = 38 := by
sorry

end NUMINAMATH_CALUDE_bobby_candy_problem_l1171_117108


namespace NUMINAMATH_CALUDE_distance_between_trees_l1171_117180

/-- Given a curved path of length 300 meters with 26 trees planted at equal arc lengths,
    including one at each end, the distance between consecutive trees is 12 meters. -/
theorem distance_between_trees (path_length : ℝ) (num_trees : ℕ) :
  path_length = 300 ∧ num_trees = 26 →
  (path_length / (num_trees - 1 : ℝ)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_trees_l1171_117180


namespace NUMINAMATH_CALUDE_right_triangle_sine_cosine_sum_equality_l1171_117107

theorem right_triangle_sine_cosine_sum_equality (A B C : ℝ) (x y : ℝ) 
  (h1 : A + B + C = π / 2)  -- ∠C is a right angle
  (h2 : 0 ≤ A ∧ A ≤ π / 2)  -- A is an angle in the right triangle
  (h3 : 0 ≤ B ∧ B ≤ π / 2)  -- B is an angle in the right triangle
  (h4 : x = Real.sin A + Real.cos A)  -- Definition of x
  (h5 : y = Real.sin B + Real.cos B)  -- Definition of y
  : x = y := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sine_cosine_sum_equality_l1171_117107


namespace NUMINAMATH_CALUDE_percentage_composition_l1171_117192

theorem percentage_composition (F S T : ℝ) 
  (h1 : F = 0.20 * S) 
  (h2 : S = 0.25 * T) : 
  F = 0.05 * T := by
sorry

end NUMINAMATH_CALUDE_percentage_composition_l1171_117192


namespace NUMINAMATH_CALUDE_x_value_l1171_117177

theorem x_value (x : ℝ) (h1 : x^2 - 2*x = 0) (h2 : x ≠ 0) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l1171_117177


namespace NUMINAMATH_CALUDE_prob_three_faces_is_8_27_l1171_117147

/-- Represents a small cube sawed from a larger painted cube -/
structure SmallCube :=
  (painted_faces : Fin 4)

/-- The set of all small cubes obtained from sawing a painted cube -/
def all_cubes : Finset SmallCube := sorry

/-- The set of small cubes with exactly three painted faces -/
def three_face_cubes : Finset SmallCube := sorry

/-- The probability of selecting a small cube with three painted faces -/
def prob_three_faces : ℚ := (three_face_cubes.card : ℚ) / (all_cubes.card : ℚ)

theorem prob_three_faces_is_8_27 : prob_three_faces = 8 / 27 := by sorry

end NUMINAMATH_CALUDE_prob_three_faces_is_8_27_l1171_117147


namespace NUMINAMATH_CALUDE_milk_pumping_rate_l1171_117152

/-- Calculates the rate of milk pumped into a tanker given initial conditions --/
theorem milk_pumping_rate 
  (initial_milk : ℝ) 
  (pumping_time : ℝ) 
  (add_rate : ℝ) 
  (add_time : ℝ) 
  (milk_left : ℝ) 
  (h1 : initial_milk = 30000)
  (h2 : pumping_time = 4)
  (h3 : add_rate = 1500)
  (h4 : add_time = 7)
  (h5 : milk_left = 28980) :
  (initial_milk + add_rate * add_time - milk_left) / pumping_time = 2880 := by
  sorry

#check milk_pumping_rate

end NUMINAMATH_CALUDE_milk_pumping_rate_l1171_117152


namespace NUMINAMATH_CALUDE_soccer_league_games_l1171_117166

/-- The number of games played in a soccer league with given conditions -/
def total_games (n : ℕ) (promo_per_team : ℕ) : ℕ :=
  (n * (n - 1) + n * promo_per_team) / 2

/-- Theorem: In a soccer league with 15 teams, where each team plays every other team twice 
    and has 2 additional promotional games, the total number of games played is 120 -/
theorem soccer_league_games : total_games 15 2 = 120 := by
  sorry

end NUMINAMATH_CALUDE_soccer_league_games_l1171_117166


namespace NUMINAMATH_CALUDE_sum_of_squares_l1171_117182

theorem sum_of_squares (x y : ℕ+) 
  (h1 : x * y + x + y = 51)
  (h2 : x * x * y + x * y * y = 560) :
  x * x + y * y = 186 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1171_117182


namespace NUMINAMATH_CALUDE_bee_count_l1171_117146

theorem bee_count (flowers : ℕ) (bee_difference : ℕ) : 
  flowers = 5 → bee_difference = 2 → flowers - bee_difference = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_bee_count_l1171_117146


namespace NUMINAMATH_CALUDE_expression_simplification_l1171_117176

theorem expression_simplification (x : ℝ) (h : x ≠ -1) :
  x / (x + 1) - 3 * x / (2 * (x + 1)) - 1 = (-3 * x - 2) / (2 * (x + 1)) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1171_117176


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1171_117126

theorem sum_of_coefficients (b₆ b₅ b₄ b₃ b₂ b₁ b₀ : ℝ) :
  (∀ x : ℝ, (4 * x - 2)^6 = b₆ * x^6 + b₅ * x^5 + b₄ * x^4 + b₃ * x^3 + b₂ * x^2 + b₁ * x + b₀) →
  b₆ + b₅ + b₄ + b₃ + b₂ + b₁ + b₀ = 64 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1171_117126


namespace NUMINAMATH_CALUDE_parabola_coefficient_l1171_117164

/-- Given a parabola y = ax^2 + bx + c with vertex at (p, p) and y-intercept at (0, -3p),
    where p ≠ 0, the coefficient b is equal to 8/p. -/
theorem parabola_coefficient (a b c p : ℝ) : 
  p ≠ 0 → 
  (∀ x, a * x^2 + b * x + c = a * (x - p)^2 + p) → 
  a * 0^2 + b * 0 + c = -3 * p → 
  b = 8 / p := by
  sorry

end NUMINAMATH_CALUDE_parabola_coefficient_l1171_117164


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_ratio_l1171_117174

theorem quadratic_equation_roots_ratio (q : ℝ) : 
  (∃ r₁ r₂ : ℝ, r₁ ≠ 0 ∧ r₂ ≠ 0 ∧ r₁ / r₂ = 3 ∧ 
   r₁^2 + 10*r₁ + q = 0 ∧ r₂^2 + 10*r₂ + q = 0) → 
  q = 18.75 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_ratio_l1171_117174


namespace NUMINAMATH_CALUDE_third_number_proof_l1171_117135

/-- The smallest number greater than 57 that leaves the same remainder as 25 and 57 when divided by 16 -/
def third_number : ℕ := 73

/-- The common divisor -/
def common_divisor : ℕ := 16

theorem third_number_proof :
  (third_number % common_divisor = 25 % common_divisor) ∧
  (third_number % common_divisor = 57 % common_divisor) ∧
  (third_number > 57) ∧
  (∀ n : ℕ, n > 57 ∧ n < third_number →
    (n % common_divisor ≠ 25 % common_divisor ∨
     n % common_divisor ≠ 57 % common_divisor)) :=
by sorry

end NUMINAMATH_CALUDE_third_number_proof_l1171_117135


namespace NUMINAMATH_CALUDE_expand_expression_l1171_117111

theorem expand_expression (x : ℝ) : (x + 4) * (5 * x - 10) = 5 * x^2 + 10 * x - 40 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1171_117111


namespace NUMINAMATH_CALUDE_smallest_prime_with_30_divisors_l1171_117194

/-- A function that counts the number of positive divisors of a natural number -/
def count_divisors (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number is prime -/
def is_prime (n : ℕ) : Prop := sorry

/-- The expression p^3 + 4p^2 + 4p -/
def f (p : ℕ) : ℕ := p^3 + 4*p^2 + 4*p

theorem smallest_prime_with_30_divisors :
  ∀ p : ℕ, is_prime p → (∀ q < p, is_prime q → count_divisors (f q) ≠ 30) →
  count_divisors (f p) = 30 → p = 43 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_with_30_divisors_l1171_117194


namespace NUMINAMATH_CALUDE_greatest_q_minus_r_l1171_117170

theorem greatest_q_minus_r : ∃ (q r : ℕ+), 
  975 = 23 * q + r ∧ 
  ∀ (q' r' : ℕ+), 975 = 23 * q' + r' → (q - r : ℤ) ≥ (q' - r' : ℤ) ∧
  (q - r : ℤ) = 33 :=
sorry

end NUMINAMATH_CALUDE_greatest_q_minus_r_l1171_117170


namespace NUMINAMATH_CALUDE_green_shirt_pairs_l1171_117109

theorem green_shirt_pairs (red_students green_students total_students total_pairs red_red_pairs : ℕ)
  (h1 : red_students = 70)
  (h2 : green_students = 94)
  (h3 : total_students = red_students + green_students)
  (h4 : total_pairs = 82)
  (h5 : red_red_pairs = 28)
  : ∃ green_green_pairs : ℕ, green_green_pairs = 40 ∧
    green_green_pairs = total_pairs - red_red_pairs - (red_students - 2 * red_red_pairs) := by
  sorry

end NUMINAMATH_CALUDE_green_shirt_pairs_l1171_117109


namespace NUMINAMATH_CALUDE_enclosing_triangle_sides_l1171_117169

/-- An isosceles triangle enclosing a circle -/
structure EnclosingTriangle where
  /-- Radius of the enclosed circle -/
  r : ℝ
  /-- Acute angle at the base of the isosceles triangle in radians -/
  θ : ℝ
  /-- Length of the equal sides of the isosceles triangle -/
  a : ℝ
  /-- Length of the base of the isosceles triangle -/
  b : ℝ

/-- The theorem stating the side lengths of the enclosing isosceles triangle -/
theorem enclosing_triangle_sides (t : EnclosingTriangle) 
  (h_r : t.r = 3)
  (h_θ : t.θ = π/6) -- 30° in radians
  : t.a = 4 * Real.sqrt 3 + 6 ∧ t.b = 6 * Real.sqrt 3 + 12 := by
  sorry


end NUMINAMATH_CALUDE_enclosing_triangle_sides_l1171_117169


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1171_117140

theorem geometric_sequence_sum (a q : ℝ) (h1 : a + a * q = 7) (h2 : a * (q^6 - 1) / (q - 1) = 91) :
  a * (1 + q + q^2 + q^3) = 28 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1171_117140


namespace NUMINAMATH_CALUDE_gift_contribution_total_l1171_117167

/-- Proves that the total contribution is $20 given the specified conditions -/
theorem gift_contribution_total (n : ℕ) (min_contribution max_contribution : ℝ) :
  n = 10 →
  min_contribution = 1 →
  max_contribution = 11 →
  (n - 1 : ℝ) * min_contribution + max_contribution = 20 :=
by sorry

end NUMINAMATH_CALUDE_gift_contribution_total_l1171_117167


namespace NUMINAMATH_CALUDE_bikes_total_price_l1171_117112

/-- The total price of Marion's and Stephanie's bikes -/
def total_price (marion_price stephanie_price : ℕ) : ℕ :=
  marion_price + stephanie_price

/-- Theorem stating the total price of Marion's and Stephanie's bikes -/
theorem bikes_total_price :
  ∃ (marion_price stephanie_price : ℕ),
    marion_price = 356 ∧
    stephanie_price = 2 * marion_price ∧
    total_price marion_price stephanie_price = 1068 :=
by
  sorry


end NUMINAMATH_CALUDE_bikes_total_price_l1171_117112


namespace NUMINAMATH_CALUDE_consecutive_circle_selections_l1171_117127

/-- Represents the arrangement of circles in the figure -/
structure CircleArrangement :=
  (total_circles : Nat)
  (long_side_rows : Nat)
  (perpendicular_rows : Nat)

/-- Calculates the number of ways to choose three consecutive circles along the long side -/
def long_side_selections (arr : CircleArrangement) : Nat :=
  (arr.long_side_rows * (arr.long_side_rows + 1)) / 2

/-- Calculates the number of ways to choose three consecutive circles along one perpendicular direction -/
def perpendicular_selections (arr : CircleArrangement) : Nat :=
  (3 * arr.perpendicular_rows + (arr.perpendicular_rows * (arr.perpendicular_rows - 1)) / 2)

/-- The main theorem stating the total number of ways to choose three consecutive circles -/
theorem consecutive_circle_selections (arr : CircleArrangement) 
  (h1 : arr.total_circles = 33)
  (h2 : arr.long_side_rows = 6)
  (h3 : arr.perpendicular_rows = 6) :
  long_side_selections arr + 2 * perpendicular_selections arr = 57 := by
  sorry


end NUMINAMATH_CALUDE_consecutive_circle_selections_l1171_117127


namespace NUMINAMATH_CALUDE_log_cutting_ratio_l1171_117138

/-- Given a log of length 20 feet where each linear foot weighs 150 pounds,
    if the log is cut into two equal pieces each weighing 1500 pounds,
    then the ratio of the length of each cut piece to the length of the original log is 1/2. -/
theorem log_cutting_ratio :
  ∀ (original_length cut_length : ℝ) (weight_per_foot cut_weight : ℝ),
    original_length = 20 →
    weight_per_foot = 150 →
    cut_weight = 1500 →
    cut_length * weight_per_foot = cut_weight →
    cut_length / original_length = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_log_cutting_ratio_l1171_117138


namespace NUMINAMATH_CALUDE_count_sequences_with_at_least_three_heads_l1171_117159

/-- The number of distinct sequences of 10 coin flips containing at least 3 heads -/
def sequences_with_at_least_three_heads : ℕ :=
  2^10 - (Nat.choose 10 0 + Nat.choose 10 1 + Nat.choose 10 2)

/-- Theorem stating that the number of sequences with at least 3 heads is 968 -/
theorem count_sequences_with_at_least_three_heads :
  sequences_with_at_least_three_heads = 968 := by
  sorry

end NUMINAMATH_CALUDE_count_sequences_with_at_least_three_heads_l1171_117159


namespace NUMINAMATH_CALUDE_sqrt_nested_expression_l1171_117144

theorem sqrt_nested_expression : Real.sqrt (25 * Real.sqrt (15 * Real.sqrt 9)) = 5 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_nested_expression_l1171_117144


namespace NUMINAMATH_CALUDE_evaluate_nested_brackets_l1171_117153

-- Define the operation [a,b,c]
def bracket (a b c : ℚ) : ℚ := (a + b) / c

-- State the theorem
theorem evaluate_nested_brackets :
  bracket (bracket 100 50 150) (bracket 4 2 6) (bracket 20 10 30) = 2 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_nested_brackets_l1171_117153


namespace NUMINAMATH_CALUDE_uphill_divisible_by_25_count_l1171_117105

/-- A positive integer is uphill if every digit is strictly greater than the previous digit. -/
def is_uphill (n : ℕ) : Prop :=
  ∀ i j, i < j → (n.digits 10).get! i < (n.digits 10).get! j

/-- A number is divisible by 25 if and only if it ends in 00 or 25. -/
def divisible_by_25 (n : ℕ) : Prop :=
  n % 25 = 0

/-- The count of uphill integers divisible by 25 -/
def count_uphill_divisible_by_25 : ℕ := 3

theorem uphill_divisible_by_25_count :
  (∃ S : Finset ℕ, (∀ n ∈ S, is_uphill n ∧ divisible_by_25 n) ∧
                   (∀ n, is_uphill n → divisible_by_25 n → n ∈ S) ∧
                   S.card = count_uphill_divisible_by_25) :=
sorry

end NUMINAMATH_CALUDE_uphill_divisible_by_25_count_l1171_117105


namespace NUMINAMATH_CALUDE_cookie_box_weight_limit_l1171_117168

/-- The weight limit of a cookie box in pounds, given the weight of each cookie and the number of cookies it can hold. -/
theorem cookie_box_weight_limit (cookie_weight : ℚ) (box_capacity : ℕ) : 
  cookie_weight = 2 → box_capacity = 320 → (cookie_weight * box_capacity) / 16 = 40 := by
  sorry

end NUMINAMATH_CALUDE_cookie_box_weight_limit_l1171_117168


namespace NUMINAMATH_CALUDE_calculation_proof_l1171_117143

theorem calculation_proof :
  (1 : ℝ) = (1/3)^0 ∧
  3 = Real.sqrt 27 ∧
  3 = |-3| ∧
  1 = Real.tan (π/4) →
  (1/3)^0 + Real.sqrt 27 - |-3| + Real.tan (π/4) = 1 + 3 * Real.sqrt 3 - 2 ∧
  ∀ x : ℝ, (x + 2)^2 - 2*(x - 1) = x^2 + 2*x + 6 :=
by sorry

end NUMINAMATH_CALUDE_calculation_proof_l1171_117143


namespace NUMINAMATH_CALUDE_largest_integral_x_l1171_117139

theorem largest_integral_x : ∃ (x : ℤ), 
  (∀ (y : ℤ), (1 : ℚ) / 3 < (y : ℚ) / 5 ∧ (y : ℚ) / 5 < 5 / 8 → y ≤ x) ∧
  (1 : ℚ) / 3 < (x : ℚ) / 5 ∧ (x : ℚ) / 5 < 5 / 8 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integral_x_l1171_117139


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l1171_117113

-- Problem 1
theorem problem_1 : -2.4 + 3.5 - 4.6 + 3.5 = 0 := by sorry

-- Problem 2
theorem problem_2 : (-40) - (-28) - (-19) + (-24) = -17 := by sorry

-- Problem 3
theorem problem_3 : (-3 : ℚ) * (5/6 : ℚ) * (-4/5 : ℚ) * (-1/4 : ℚ) = -1/2 := by sorry

-- Problem 4
theorem problem_4 : (-5/7 : ℚ) * (-4/3 : ℚ) / (-15/7 : ℚ) = -4/9 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l1171_117113


namespace NUMINAMATH_CALUDE_shaded_area_proof_shaded_area_is_sqrt_288_l1171_117150

theorem shaded_area_proof (small_square_area : ℝ) 
  (h1 : small_square_area = 3) 
  (num_small_squares : ℕ) 
  (h2 : num_small_squares = 9) : ℝ :=
  let small_square_side := Real.sqrt small_square_area
  let small_square_diagonal := small_square_side * Real.sqrt 2
  let large_square_side := 2 * small_square_diagonal + small_square_side
  let large_square_area := large_square_side ^ 2
  let total_small_squares_area := num_small_squares * small_square_area
  let shaded_area := large_square_area - total_small_squares_area
  Real.sqrt 288

theorem shaded_area_is_sqrt_288 : shaded_area_proof 3 rfl 9 rfl = Real.sqrt 288 := by sorry

end NUMINAMATH_CALUDE_shaded_area_proof_shaded_area_is_sqrt_288_l1171_117150


namespace NUMINAMATH_CALUDE_race_result_l1171_117142

-- Define the set of runners
inductive Runner : Type
| P : Runner
| Q : Runner
| R : Runner
| S : Runner
| T : Runner

-- Define the relation "beats" between runners
def beats : Runner → Runner → Prop := sorry

-- Define the relation "finishes_before" between runners
def finishes_before : Runner → Runner → Prop := sorry

-- Define what it means for a runner to finish third
def finishes_third : Runner → Prop := sorry

-- State the theorem
theorem race_result : 
  (beats Runner.P Runner.Q) →
  (beats Runner.P Runner.R) →
  (beats Runner.Q Runner.S) →
  (finishes_before Runner.P Runner.T) →
  (finishes_before Runner.T Runner.Q) →
  (¬ finishes_third Runner.P ∧ ¬ finishes_third Runner.S) ∧
  (∃ (x : Runner), x ≠ Runner.P ∧ x ≠ Runner.S ∧ finishes_third x) :=
by sorry

end NUMINAMATH_CALUDE_race_result_l1171_117142


namespace NUMINAMATH_CALUDE_function_passes_through_point_l1171_117132

theorem function_passes_through_point (a : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 3) + 2
  f 3 = 3 := by sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l1171_117132


namespace NUMINAMATH_CALUDE_remainder_problem_l1171_117122

theorem remainder_problem : (56 * 67 * 78) % 15 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1171_117122


namespace NUMINAMATH_CALUDE_students_not_playing_sports_l1171_117181

theorem students_not_playing_sports (total : ℕ) (soccer : ℕ) (volleyball : ℕ) (one_sport : ℕ) : 
  total = 40 → soccer = 20 → volleyball = 19 → one_sport = 15 → 
  ∃ (both : ℕ), 
    both = soccer + volleyball - one_sport ∧
    total - (soccer + volleyball - both) = 13 := by
  sorry

end NUMINAMATH_CALUDE_students_not_playing_sports_l1171_117181


namespace NUMINAMATH_CALUDE_parabola_intersection_comparison_l1171_117137

theorem parabola_intersection_comparison (m n a b : ℝ) : 
  (∀ x, m * x^2 + x ≥ 0 → x ≤ a) →  -- A(a,0) is the rightmost intersection of y = mx^2 + x with x-axis
  (∀ x, n * x^2 + x ≥ 0 → x ≤ b) →  -- B(b,0) is the rightmost intersection of y = nx^2 + x with x-axis
  m * a^2 + a = 0 →                  -- A(a,0) is on the parabola y = mx^2 + x
  n * b^2 + b = 0 →                  -- B(b,0) is on the parabola y = nx^2 + x
  a > b →                            -- A is to the right of B
  a > 0 →                            -- A is in the positive half of x-axis
  b > 0 →                            -- B is in the positive half of x-axis
  m > n :=                           -- Conclusion: m > n
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_comparison_l1171_117137


namespace NUMINAMATH_CALUDE_sheila_hourly_wage_l1171_117179

/-- Sheila's work schedule and earnings --/
structure WorkSchedule where
  hours_mon_wed_fri : ℕ
  hours_tue_thu : ℕ
  weekly_earnings : ℕ

/-- Calculate Sheila's hourly wage --/
def hourly_wage (schedule : WorkSchedule) : ℚ :=
  let total_hours := 3 * schedule.hours_mon_wed_fri + 2 * schedule.hours_tue_thu
  schedule.weekly_earnings / total_hours

/-- Theorem: Sheila's hourly wage is $11 --/
theorem sheila_hourly_wage :
  let sheila_schedule := WorkSchedule.mk 8 6 396
  hourly_wage sheila_schedule = 11 := by sorry

end NUMINAMATH_CALUDE_sheila_hourly_wage_l1171_117179


namespace NUMINAMATH_CALUDE_simplify_expression_l1171_117171

theorem simplify_expression (x : ℝ) : 120 * x - 55 * x = 65 * x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1171_117171


namespace NUMINAMATH_CALUDE_garden_area_calculation_l1171_117129

def garden_length : ℝ := 18
def garden_width : ℝ := 15
def cutout1_side : ℝ := 4
def cutout2_side : ℝ := 2

theorem garden_area_calculation :
  garden_length * garden_width - (cutout1_side * cutout1_side + cutout2_side * cutout2_side) = 250 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_calculation_l1171_117129


namespace NUMINAMATH_CALUDE_sum_equals_seven_eighths_l1171_117191

theorem sum_equals_seven_eighths : 
  let original_sum := 1/2 + 1/4 + 1/8 + 1/16 + 1/32 + 1/64
  let removed_terms := 1/16 + 1/32 + 1/64
  let remaining_terms := original_sum - removed_terms
  remaining_terms = 7/8 := by sorry

end NUMINAMATH_CALUDE_sum_equals_seven_eighths_l1171_117191


namespace NUMINAMATH_CALUDE_min_n_for_equation_property_l1171_117163

theorem min_n_for_equation_property : ∃ (n : ℕ), n = 835 ∧ 
  (∀ (S : Finset ℕ), S.card = n → (∀ x ∈ S, x ≥ 1 ∧ x ≤ 999) →
    ∃ (a b c d : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    a + 2*b + 3*c = d) ∧
  (∀ (m : ℕ), m < n →
    ∃ (T : Finset ℕ), T.card = m ∧ (∀ x ∈ T, x ≥ 1 ∧ x ≤ 999) ∧
    ¬(∃ (a b c d : ℕ), a ∈ T ∧ b ∈ T ∧ c ∈ T ∧ d ∈ T ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    a + 2*b + 3*c = d)) :=
by sorry

end NUMINAMATH_CALUDE_min_n_for_equation_property_l1171_117163


namespace NUMINAMATH_CALUDE_sin_180_degrees_l1171_117128

theorem sin_180_degrees : Real.sin (π) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_180_degrees_l1171_117128


namespace NUMINAMATH_CALUDE_triangle_coverage_convex_polygon_coverage_l1171_117188

-- Define a Circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a Triangle type
structure Triangle where
  a : ℝ × ℝ
  b : ℝ × ℝ
  c : ℝ × ℝ

-- Define a ConvexPolygon type
structure ConvexPolygon where
  vertices : List (ℝ × ℝ)

-- Function to check if a circle covers a point
def covers (c : Circle) (p : ℝ × ℝ) : Prop :=
  (c.center.1 - p.1)^2 + (c.center.2 - p.2)^2 ≤ c.radius^2

-- Function to check if a set of circles covers a triangle
def covers_triangle (circles : List Circle) (t : Triangle) : Prop :=
  ∀ p : ℝ × ℝ, (p = t.a ∨ p = t.b ∨ p = t.c) → ∃ c ∈ circles, covers c p

-- Function to check if a set of circles covers a convex polygon
def covers_polygon (circles : List Circle) (p : ConvexPolygon) : Prop :=
  ∀ v ∈ p.vertices, ∃ c ∈ circles, covers c v

-- Function to calculate the diameter of a convex polygon
def diameter (p : ConvexPolygon) : ℝ :=
  sorry

-- Theorem for triangle coverage
theorem triangle_coverage (t : Triangle) :
  ∃ circles : List Circle, circles.length ≤ 2 ∧ 
  (∀ c ∈ circles, c.radius = 0.5) ∧ 
  covers_triangle circles t :=
sorry

-- Theorem for convex polygon coverage
theorem convex_polygon_coverage (p : ConvexPolygon) :
  diameter p = 1 →
  ∃ circles : List Circle, circles.length ≤ 3 ∧ 
  (∀ c ∈ circles, c.radius = 0.5) ∧ 
  covers_polygon circles p :=
sorry

end NUMINAMATH_CALUDE_triangle_coverage_convex_polygon_coverage_l1171_117188


namespace NUMINAMATH_CALUDE_question_distribution_l1171_117102

-- Define the types for our problem
def TotalQuestions : ℕ := 100
def CorrectAnswersPerStudent : ℕ := 60

-- Define the number of students
def NumStudents : ℕ := 3

-- Define the types of questions
def EasyQuestions (x : ℕ) : Prop := x ≤ TotalQuestions
def MediumQuestions (y : ℕ) : Prop := y ≤ TotalQuestions
def DifficultQuestions (z : ℕ) : Prop := z ≤ TotalQuestions

-- State the theorem
theorem question_distribution 
  (x y z : ℕ) 
  (h1 : EasyQuestions x)
  (h2 : MediumQuestions y)
  (h3 : DifficultQuestions z)
  (h4 : x + y + z = TotalQuestions)
  (h5 : 3 * x + 2 * y + z = NumStudents * CorrectAnswersPerStudent) :
  z - x = 20 :=
sorry

end NUMINAMATH_CALUDE_question_distribution_l1171_117102


namespace NUMINAMATH_CALUDE_triangle_inequality_theorem_l1171_117197

def triangle_inequality (f g : ℝ → ℝ) (A B : ℝ) : Prop :=
  f (Real.cos A) * g (Real.sin B) > f (Real.sin B) * g (Real.cos A)

theorem triangle_inequality_theorem 
  (f g : ℝ → ℝ) 
  (hf : Differentiable ℝ f) 
  (hg : Differentiable ℝ g)
  (hg_pos : ∀ x, g x > 0)
  (h_deriv : ∀ x, (deriv f x) * (g x) - (f x) * (deriv g x) > 0)
  (A B C : ℝ)
  (h_obtuse : C > Real.pi / 2)
  (h_triangle : A + B + C = Real.pi) :
  triangle_inequality f g A B :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_theorem_l1171_117197


namespace NUMINAMATH_CALUDE_cos_2alpha_value_l1171_117141

theorem cos_2alpha_value (α : ℝ) (h : (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 1/2) :
  Real.cos (2 * α) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_value_l1171_117141


namespace NUMINAMATH_CALUDE_triangle_sides_from_divided_areas_l1171_117165

/-- Given a triangle with an inscribed circle, if the segments from the vertices to the center
    of the inscribed circle divide the triangle's area into parts of 28, 60, and 80,
    then the sides of the triangle are 14, 30, and 40. -/
theorem triangle_sides_from_divided_areas (a b c : ℝ) (r : ℝ) :
  (1/2 * a * r = 28) →
  (1/2 * b * r = 60) →
  (1/2 * c * r = 80) →
  (a = 14 ∧ b = 30 ∧ c = 40) :=
by sorry

end NUMINAMATH_CALUDE_triangle_sides_from_divided_areas_l1171_117165


namespace NUMINAMATH_CALUDE_absolute_difference_of_opposite_signs_l1171_117123

theorem absolute_difference_of_opposite_signs (m n : ℤ) : 
  (abs m = 5) → (abs n = 2) → (m * n < 0) → abs (m - n) = 7 := by
  sorry

end NUMINAMATH_CALUDE_absolute_difference_of_opposite_signs_l1171_117123


namespace NUMINAMATH_CALUDE_PL_length_l1171_117119

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a rectangle -/
structure Rectangle :=
  (topLeft : Point) (bottomRight : Point)

/-- The square WXYZ -/
def square : Rectangle :=
  { topLeft := { x := 0, y := 2 },
    bottomRight := { x := 2, y := 0 } }

/-- The length of PL -/
def PL : ℝ := 1

/-- States that two rectangles are congruent -/
def congruentRectangles (r1 r2 : Rectangle) : Prop :=
  (r1.bottomRight.x - r1.topLeft.x) * (r1.topLeft.y - r1.bottomRight.y) =
  (r2.bottomRight.x - r2.topLeft.x) * (r2.topLeft.y - r2.bottomRight.y)

/-- The theorem to be proved -/
theorem PL_length :
  ∀ (LMNO PQRS : Rectangle),
    congruentRectangles LMNO PQRS →
    PL = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_PL_length_l1171_117119
