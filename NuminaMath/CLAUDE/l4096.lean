import Mathlib

namespace NUMINAMATH_CALUDE_balloon_fraction_after_tripling_l4096_409616

theorem balloon_fraction_after_tripling (total : ℝ) (h : total > 0) :
  let yellow_initial := (2/3) * total
  let green_initial := total - yellow_initial
  let green_after := 3 * green_initial
  let total_after := yellow_initial + green_after
  green_after / total_after = 3/5 := by
sorry

end NUMINAMATH_CALUDE_balloon_fraction_after_tripling_l4096_409616


namespace NUMINAMATH_CALUDE_multiply_by_9999_l4096_409646

theorem multiply_by_9999 : ∃ x : ℕ, x * 9999 = 4690640889 ∧ x = 469131 := by sorry

end NUMINAMATH_CALUDE_multiply_by_9999_l4096_409646


namespace NUMINAMATH_CALUDE_only_parallel_corresponding_angles_has_converse_l4096_409680

-- Define the basic geometric concepts
def Line : Type := sorry
def Angle : Type := sorry
def Triangle : Type := sorry

-- Define the geometric relations
def vertical_angles (a b : Angle) : Prop := sorry
def parallel_lines (l1 l2 : Line) : Prop := sorry
def corresponding_angles (a b : Angle) (l1 l2 : Line) : Prop := sorry
def congruent_triangles (t1 t2 : Triangle) : Prop := sorry
def right_angle (a : Angle) : Prop := sorry

-- Define the theorems
def vertical_angles_theorem (a b : Angle) : 
  vertical_angles a b → a = b := sorry

def parallel_corresponding_angles_theorem (l1 l2 : Line) (a b : Angle) :
  parallel_lines l1 l2 → corresponding_angles a b l1 l2 → a = b := sorry

def congruent_triangles_angles_theorem (t1 t2 : Triangle) (a1 a2 : Angle) :
  congruent_triangles t1 t2 → corresponding_angles a1 a2 t1 t2 → a1 = a2 := sorry

def right_angles_equal_theorem (a b : Angle) :
  right_angle a → right_angle b → a = b := sorry

-- The main theorem to prove
theorem only_parallel_corresponding_angles_has_converse :
  ∃ (l1 l2 : Line) (a b : Angle),
    (corresponding_angles a b l1 l2 ∧ a = b → parallel_lines l1 l2) ∧
    (¬∃ (a b : Angle), a = b → vertical_angles a b) ∧
    (¬∃ (t1 t2 : Triangle) (a1 a2 a3 b1 b2 b3 : Angle),
      a1 = b1 ∧ a2 = b2 ∧ a3 = b3 → congruent_triangles t1 t2) ∧
    (¬∃ (a b : Angle), a = b → right_angle a ∧ right_angle b) := by
  sorry

end NUMINAMATH_CALUDE_only_parallel_corresponding_angles_has_converse_l4096_409680


namespace NUMINAMATH_CALUDE_green_balls_count_l4096_409620

theorem green_balls_count (total : ℕ) (white yellow red purple : ℕ) (prob : ℚ) :
  total = 60 ∧
  white = 22 ∧
  yellow = 17 ∧
  red = 3 ∧
  purple = 1 ∧
  prob = 95 / 100 ∧
  (total - red - purple : ℚ) / total = prob →
  total - white - yellow - red - purple = 17 := by
  sorry

end NUMINAMATH_CALUDE_green_balls_count_l4096_409620


namespace NUMINAMATH_CALUDE_cube_root_of_1331_l4096_409644

theorem cube_root_of_1331 (y : ℝ) (h1 : y > 0) (h2 : y^3 = 1331) : y = 11 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_1331_l4096_409644


namespace NUMINAMATH_CALUDE_towels_used_is_285_l4096_409607

/-- Calculates the total number of towels used in a gym over 4 hours -/
def totalTowelsUsed (firstHourGuests : ℕ) : ℕ :=
  let secondHourGuests := firstHourGuests + (firstHourGuests * 20 / 100)
  let thirdHourGuests := secondHourGuests + (secondHourGuests * 25 / 100)
  let fourthHourGuests := thirdHourGuests + (thirdHourGuests / 3)
  firstHourGuests + secondHourGuests + thirdHourGuests + fourthHourGuests

/-- Theorem stating that the total number of towels used is 285 -/
theorem towels_used_is_285 :
  totalTowelsUsed 50 = 285 := by
  sorry

#eval totalTowelsUsed 50

end NUMINAMATH_CALUDE_towels_used_is_285_l4096_409607


namespace NUMINAMATH_CALUDE_max_xy_under_constraint_l4096_409694

theorem max_xy_under_constraint (x y : ℕ+) (h : 27 * x.val + 35 * y.val ≤ 945) :
  x.val * y.val ≤ 234 :=
by sorry

end NUMINAMATH_CALUDE_max_xy_under_constraint_l4096_409694


namespace NUMINAMATH_CALUDE_odd_numbers_equality_l4096_409689

theorem odd_numbers_equality (a b c d k m : ℕ) : 
  Odd a → Odd b → Odd c → Odd d →
  0 < a → a < b → b < c → c < d →
  a * d = b * c →
  a + d = 2^k →
  b + c = 2^m →
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_odd_numbers_equality_l4096_409689


namespace NUMINAMATH_CALUDE_infinite_solutions_imply_d_value_l4096_409621

theorem infinite_solutions_imply_d_value (d : ℚ) :
  (∀ (x : ℚ), 3 * (5 + 2 * d * x) = 15 * x + 15) → d = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_imply_d_value_l4096_409621


namespace NUMINAMATH_CALUDE_unique_quadratic_m_l4096_409612

def is_quadratic_coefficient (m : ℝ) : Prop :=
  |m| = 2 ∧ m - 2 ≠ 0

theorem unique_quadratic_m :
  ∃! m : ℝ, is_quadratic_coefficient m ∧ m = -2 :=
sorry

end NUMINAMATH_CALUDE_unique_quadratic_m_l4096_409612


namespace NUMINAMATH_CALUDE_f_geq_a_iff_a_in_range_l4096_409674

/-- The function f(x) = x^2 - 2ax + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 2

/-- The domain of x -/
def domain : Set ℝ := {x | x ≥ -1}

/-- The theorem stating the condition for a -/
theorem f_geq_a_iff_a_in_range (a : ℝ) : 
  (∀ x ∈ domain, f a x ≥ a) ↔ -3 ≤ a ∧ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_f_geq_a_iff_a_in_range_l4096_409674


namespace NUMINAMATH_CALUDE_flower_bed_area_l4096_409618

theorem flower_bed_area (a b c : ℝ) (ha : a = 6) (hb : b = 8) (hc : c = 10) :
  (1/2) * a * b = 24 := by
  sorry

end NUMINAMATH_CALUDE_flower_bed_area_l4096_409618


namespace NUMINAMATH_CALUDE_power_of_64_l4096_409693

theorem power_of_64 : (64 : ℝ) ^ (5/6) = 32 := by
  have h1 : (64 : ℝ) = 2^6 := by sorry
  sorry

end NUMINAMATH_CALUDE_power_of_64_l4096_409693


namespace NUMINAMATH_CALUDE_tangent_line_equation_l4096_409639

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 2*y + 1 = 0

/-- A point on the line -/
def point : ℝ × ℝ := (-2, 5)

/-- Possible equations of the tangent line -/
def tangent_line_eq1 (x : ℝ) : Prop := x = -2
def tangent_line_eq2 (x y : ℝ) : Prop := 15*x + 8*y - 10 = 0

/-- The main theorem -/
theorem tangent_line_equation :
  ∃ (x y : ℝ), (x = point.1 ∧ y = point.2) ∧
  (∀ (x' y' : ℝ), circle_equation x' y' →
    (tangent_line_eq1 x ∨ tangent_line_eq2 x y) ∧
    (x = x' ∧ y = y' → ¬circle_equation x y)) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l4096_409639


namespace NUMINAMATH_CALUDE_length_width_difference_l4096_409670

/-- The length of the basketball court in meters -/
def court_length : ℝ := 31

/-- The width of the basketball court in meters -/
def court_width : ℝ := 17

/-- The perimeter of the basketball court in meters -/
def court_perimeter : ℝ := 96

theorem length_width_difference : court_length - court_width = 14 := by
  sorry

end NUMINAMATH_CALUDE_length_width_difference_l4096_409670


namespace NUMINAMATH_CALUDE_ruler_measurement_l4096_409617

/-- Represents a ruler with marks at specific positions -/
structure Ruler :=
  (marks : List ℝ)

/-- Checks if a length can be measured using the given ruler -/
def can_measure (r : Ruler) (length : ℝ) : Prop :=
  ∃ (coeffs : List ℤ), length = (List.zip r.marks coeffs).foldl (λ acc (m, c) => acc + m * c) 0

theorem ruler_measurement (r : Ruler) (h : r.marks = [0, 7, 11]) :
  (can_measure r 8) ∧ (can_measure r 5) := by
  sorry

end NUMINAMATH_CALUDE_ruler_measurement_l4096_409617


namespace NUMINAMATH_CALUDE_susan_age_in_five_years_l4096_409677

/-- Represents the ages and time relationships in the problem -/
structure AgeRelationship where
  j : ℕ  -- James' current age
  n : ℕ  -- Janet's current age
  s : ℕ  -- Susan's current age
  x : ℕ  -- Years until James turns 37

/-- The conditions given in the problem -/
def problem_conditions (ar : AgeRelationship) : Prop :=
  (ar.j - 8 = 2 * (ar.n - 8)) ∧  -- 8 years ago, James was twice Janet's age
  (ar.j + ar.x = 37) ∧           -- In x years, James will turn 37
  (ar.s = ar.n - 3)              -- Susan was born when Janet turned 3

/-- The theorem to be proved -/
theorem susan_age_in_five_years (ar : AgeRelationship) 
  (h : problem_conditions ar) : 
  ar.s + 5 = ar.n + 2 := by
  sorry


end NUMINAMATH_CALUDE_susan_age_in_five_years_l4096_409677


namespace NUMINAMATH_CALUDE_special_function_inequality_l4096_409669

/-- A function f: ℝ → ℝ satisfying f(x) + f''(x) > 1 for all x -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x + (deriv^[2] f) x > 1

/-- Theorem stating the relationship between f(2) - 1 and e^(f(3) - 1) -/
theorem special_function_inequality (f : ℝ → ℝ) (hf : SpecialFunction f) :
  f 2 - 1 < Real.exp (f 3 - 1) := by
  sorry

end NUMINAMATH_CALUDE_special_function_inequality_l4096_409669


namespace NUMINAMATH_CALUDE_trapezoid_area_sum_l4096_409675

/-- Represents a trapezoid with side lengths a, b, c, and d. -/
structure Trapezoid where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Calculates the possible areas of a trapezoid. -/
def possibleAreas (t : Trapezoid) : Set ℝ :=
  sorry

/-- Checks if a number is not divisible by the square of any prime. -/
def notDivisibleBySquareOfPrime (n : ℕ) : Prop :=
  sorry

/-- The main theorem about the trapezoid areas. -/
theorem trapezoid_area_sum (t : Trapezoid) 
    (h1 : t.a = 4 ∧ t.b = 6 ∧ t.c = 8 ∧ t.d = 10) :
    ∃ (r₁ r₂ r₃ : ℚ) (n₁ n₂ : ℕ),
      (∀ A ∈ possibleAreas t, ∃ k, A = k * (r₁ * Real.sqrt n₁ + r₂ * Real.sqrt n₂ + r₃)) ∧
      notDivisibleBySquareOfPrime n₁ ∧
      notDivisibleBySquareOfPrime n₂ ∧
      ⌊r₁ + r₂ + r₃ + n₁ + n₂⌋ = 26 :=
by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_sum_l4096_409675


namespace NUMINAMATH_CALUDE_triangle_max_area_l4096_409698

/-- Given a triangle ABC with AB = 10 and BC:AC = 35:36, its maximum area is 1260 -/
theorem triangle_max_area (A B C : ℝ × ℝ) :
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let s := (AB + BC + AC) / 2
  let area := Real.sqrt (s * (s - AB) * (s - BC) * (s - AC))
  AB = 10 ∧ BC / AC = 35 / 36 → area ≤ 1260 := by
  sorry

#check triangle_max_area

end NUMINAMATH_CALUDE_triangle_max_area_l4096_409698


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l4096_409632

/-- Given two real numbers p and q with arithmetic mean 10, and a third real number r
    such that r - p = 30, prove that the arithmetic mean of q and r is 25. -/
theorem arithmetic_mean_problem (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10) 
  (h2 : r - p = 30) : 
  (q + r) / 2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l4096_409632


namespace NUMINAMATH_CALUDE_taco_truck_profit_l4096_409691

/-- Calculate the profit for a taco truck given the total beef, beef per taco, selling price, and cost to make. -/
theorem taco_truck_profit
  (total_beef : ℝ)
  (beef_per_taco : ℝ)
  (selling_price : ℝ)
  (cost_to_make : ℝ)
  (h1 : total_beef = 100)
  (h2 : beef_per_taco = 0.25)
  (h3 : selling_price = 2)
  (h4 : cost_to_make = 1.5) :
  (total_beef / beef_per_taco) * (selling_price - cost_to_make) = 200 :=
by
  sorry

#check taco_truck_profit

end NUMINAMATH_CALUDE_taco_truck_profit_l4096_409691


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l4096_409653

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - x + 2 ≥ 0) ↔ (∃ x : ℝ, x^2 - x + 2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l4096_409653


namespace NUMINAMATH_CALUDE_exponent_multiplication_l4096_409604

theorem exponent_multiplication (a : ℝ) : a * a^2 * a^3 = a^6 := by sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l4096_409604


namespace NUMINAMATH_CALUDE_unique_valid_equation_l4096_409668

theorem unique_valid_equation (a b : ℝ) : 
  (∃ a b : ℝ, a ≠ 0 ∨ b ≠ 0 ∧ Real.sqrt (a^2 + b^2) = 1) ∧ 
  (∀ a b : ℝ, a ≠ 0 ∨ b ≠ 0 → Real.sqrt (a^2 + b^2) ≠ a - b) ∧
  (∀ a b : ℝ, a ≠ 0 ∨ b ≠ 0 → Real.sqrt (a^2 + b^2) ≠ 3*(a + b)) :=
by sorry


end NUMINAMATH_CALUDE_unique_valid_equation_l4096_409668


namespace NUMINAMATH_CALUDE_total_gum_pieces_l4096_409676

theorem total_gum_pieces (packages : ℕ) (pieces_per_package : ℕ) 
  (h1 : packages = 27) (h2 : pieces_per_package = 18) : 
  packages * pieces_per_package = 486 := by
  sorry

end NUMINAMATH_CALUDE_total_gum_pieces_l4096_409676


namespace NUMINAMATH_CALUDE_dining_bill_calculation_l4096_409695

theorem dining_bill_calculation (total_bill : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) 
  (h1 : total_bill = 198)
  (h2 : tax_rate = 0.1)
  (h3 : tip_rate = 0.2) : 
  ∃ (food_price : ℝ), 
    food_price * (1 + tax_rate) * (1 + tip_rate) = total_bill ∧ 
    food_price = 150 := by
  sorry

end NUMINAMATH_CALUDE_dining_bill_calculation_l4096_409695


namespace NUMINAMATH_CALUDE_coin_ratio_is_one_one_one_l4096_409622

/-- Represents the types of coins in the bag -/
inductive CoinType
  | OneRupee
  | FiftyPaise
  | TwentyFivePaise

/-- Represents the value of a coin in rupees -/
def coinValue : CoinType → Rat
  | CoinType.OneRupee => 1
  | CoinType.FiftyPaise => 1/2
  | CoinType.TwentyFivePaise => 1/4

/-- Represents the number of coins of each type -/
def numCoins : CoinType → Nat
  | _ => 40

/-- The total value of all coins in the bag -/
def totalValue : Rat := 70

/-- Theorem stating that the ratio of coin counts is 1:1:1 -/
theorem coin_ratio_is_one_one_one :
  numCoins CoinType.OneRupee = numCoins CoinType.FiftyPaise ∧
  numCoins CoinType.OneRupee = numCoins CoinType.TwentyFivePaise ∧
  (numCoins CoinType.OneRupee : Rat) * coinValue CoinType.OneRupee +
  (numCoins CoinType.FiftyPaise : Rat) * coinValue CoinType.FiftyPaise +
  (numCoins CoinType.TwentyFivePaise : Rat) * coinValue CoinType.TwentyFivePaise = totalValue :=
by sorry


end NUMINAMATH_CALUDE_coin_ratio_is_one_one_one_l4096_409622


namespace NUMINAMATH_CALUDE_x_fourth_minus_reciprocal_fourth_l4096_409651

theorem x_fourth_minus_reciprocal_fourth (x : ℝ) (h : x^2 - Real.sqrt 6 * x + 1 = 0) :
  |x^4 - 1/x^4| = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_minus_reciprocal_fourth_l4096_409651


namespace NUMINAMATH_CALUDE_expression_simplification_l4096_409671

theorem expression_simplification (x y : ℝ) (h : y ≠ 0) :
  ((x^2 + y^2) - (x - y)^2 + 2*y*(x - y)) / (4*y) = x - (1/2)*y :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l4096_409671


namespace NUMINAMATH_CALUDE_worker_completion_times_l4096_409609

def job_completion_time (worker1_time worker2_time : ℝ) : Prop :=
  (1 / worker1_time + 1 / worker2_time = 1 / 8) ∧
  (worker1_time = worker2_time - 12)

theorem worker_completion_times :
  ∃ (worker1_time worker2_time : ℝ),
    job_completion_time worker1_time worker2_time ∧
    worker1_time = 24 ∧
    worker2_time = 12 := by
  sorry

end NUMINAMATH_CALUDE_worker_completion_times_l4096_409609


namespace NUMINAMATH_CALUDE_boys_less_than_four_sevenths_l4096_409685

/-- Represents a class of students with two hiking trips -/
structure HikingClass where
  boys : ℕ
  girls : ℕ
  boys_trip1 : ℕ
  girls_trip1 : ℕ
  boys_trip2 : ℕ
  girls_trip2 : ℕ

/-- The conditions of the hiking trips -/
def validHikingClass (c : HikingClass) : Prop :=
  c.boys_trip1 < (2 * (c.boys_trip1 + c.girls_trip1)) / 5 ∧
  c.boys_trip2 < (2 * (c.boys_trip2 + c.girls_trip2)) / 5 ∧
  c.boys_trip1 + c.boys_trip2 ≥ c.boys ∧
  c.girls_trip1 ≤ c.girls ∧
  c.girls_trip2 ≤ c.girls

/-- The main theorem to prove -/
theorem boys_less_than_four_sevenths (c : HikingClass) 
  (h : validHikingClass c) : 
  c.boys < (4 * (c.boys + c.girls)) / 7 := by
  sorry

end NUMINAMATH_CALUDE_boys_less_than_four_sevenths_l4096_409685


namespace NUMINAMATH_CALUDE_x_fourth_coefficient_equals_a_9_l4096_409696

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The sequence a_n = 2n + 2 -/
def a (n : ℕ) : ℕ := 2 * n + 2

/-- The theorem to prove -/
theorem x_fourth_coefficient_equals_a_9 :
  binomial 5 4 + binomial 6 4 = a 9 := by sorry

end NUMINAMATH_CALUDE_x_fourth_coefficient_equals_a_9_l4096_409696


namespace NUMINAMATH_CALUDE_correct_factorization_l4096_409661

theorem correct_factorization (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_correct_factorization_l4096_409661


namespace NUMINAMATH_CALUDE_binary_arithmetic_equality_l4096_409614

-- Define a function to convert binary to decimal
def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

-- Define the binary numbers
def b1101 : List Bool := [true, false, true, true]
def b1111 : List Bool := [true, true, true, true]
def b1001 : List Bool := [true, false, false, true]
def b10 : List Bool := [false, true]
def b1010 : List Bool := [false, true, false, true]

-- State the theorem
theorem binary_arithmetic_equality :
  (binary_to_decimal b1101 + binary_to_decimal b1111) -
  (binary_to_decimal b1001 * binary_to_decimal b10) =
  binary_to_decimal b1010 := by
  sorry


end NUMINAMATH_CALUDE_binary_arithmetic_equality_l4096_409614


namespace NUMINAMATH_CALUDE_log_equation_solution_l4096_409672

theorem log_equation_solution (x : ℝ) :
  Real.log x / Real.log 8 = 1.75 → x = 32 * Real.sqrt (Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l4096_409672


namespace NUMINAMATH_CALUDE_measure_17kg_cranberries_l4096_409652

/-- Represents a two-pan scale -/
structure TwoPanScale :=
  (leftPan : ℝ)
  (rightPan : ℝ)

/-- Represents the state of the cranberry measurement process -/
structure CranberryMeasurement :=
  (totalAmount : ℝ)
  (weightAmount : ℝ)
  (scale : TwoPanScale)
  (weighingsUsed : ℕ)

/-- Definition of a valid weighing operation -/
def validWeighing (m : CranberryMeasurement) : Prop :=
  m.scale.leftPan = m.scale.rightPan ∧ m.weighingsUsed ≤ 2

/-- The main theorem to prove -/
theorem measure_17kg_cranberries :
  ∃ (m : CranberryMeasurement),
    m.totalAmount = 22 ∧
    m.weightAmount = 2 ∧
    validWeighing m ∧
    ∃ (amount : ℝ), amount = 17 ∧ amount ≤ m.totalAmount :=
sorry

end NUMINAMATH_CALUDE_measure_17kg_cranberries_l4096_409652


namespace NUMINAMATH_CALUDE_speeding_ticket_percentage_l4096_409643

theorem speeding_ticket_percentage
  (exceed_limit_percent : ℝ)
  (no_ticket_percent : ℝ)
  (h1 : exceed_limit_percent = 14.285714285714285)
  (h2 : no_ticket_percent = 30) :
  (1 - no_ticket_percent / 100) * exceed_limit_percent = 10 :=
by sorry

end NUMINAMATH_CALUDE_speeding_ticket_percentage_l4096_409643


namespace NUMINAMATH_CALUDE_solution_value_l4096_409684

theorem solution_value (a b : ℝ) (h : a^2 + b^2 - 4*a - 6*b + 13 = 0) : 
  (a - b)^2023 = -1 := by sorry

end NUMINAMATH_CALUDE_solution_value_l4096_409684


namespace NUMINAMATH_CALUDE_min_abs_phi_l4096_409611

/-- Given a function f(x) = 2sin(ωx + φ) with ω > 0, prove that the minimum value of |φ| is π/2 
    under the following conditions:
    1. Three consecutive intersection points with y = b (0 < b < 2) are at x = π/6, 5π/6, 7π/6
    2. f(x) reaches its minimum value at x = 3π/2 -/
theorem min_abs_phi (ω : ℝ) (φ : ℝ) (b : ℝ) (h_ω : ω > 0) (h_b : 0 < b ∧ b < 2) : 
  (∃ (k : ℤ), φ = 2 * π * k - 3 * π / 2) →
  (∀ (x : ℝ), 2 * Real.sin (ω * x + φ) = b → 
    (x = π / 6 ∨ x = 5 * π / 6 ∨ x = 7 * π / 6)) →
  (∀ (x : ℝ), 2 * Real.sin (ω * 3 * π / 2 + φ) ≤ 2 * Real.sin (ω * x + φ)) →
  ω = 2 ∧ (∀ (ψ : ℝ), |ψ| ≥ π / 2) := by
  sorry

end NUMINAMATH_CALUDE_min_abs_phi_l4096_409611


namespace NUMINAMATH_CALUDE_expression_evaluation_l4096_409628

theorem expression_evaluation : 
  Real.sqrt 5 * 5^(1/2 : ℝ) + 18 / 3 * 2 - 9^(3/2 : ℝ) + 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4096_409628


namespace NUMINAMATH_CALUDE_final_cell_count_l4096_409623

/-- Calculates the number of cells after a given number of days, 
    given an initial population and a tripling period. -/
def cell_population (initial_cells : ℕ) (tripling_period : ℕ) (total_days : ℕ) : ℕ :=
  initial_cells * (3 ^ (total_days / tripling_period))

/-- Theorem stating that given the specific conditions of the problem, 
    the final cell population after 9 days is 45. -/
theorem final_cell_count : cell_population 5 3 9 = 45 := by
  sorry

end NUMINAMATH_CALUDE_final_cell_count_l4096_409623


namespace NUMINAMATH_CALUDE_vanya_number_theorem_l4096_409631

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Define the property that a number satisfies the condition
def satisfiesCondition (n : ℕ) : Prop := n + sumOfDigits n = 2021

-- Theorem statement
theorem vanya_number_theorem : 
  (∀ n : ℕ, satisfiesCondition n ↔ (n = 2014 ∨ n = 1996)) := by sorry

end NUMINAMATH_CALUDE_vanya_number_theorem_l4096_409631


namespace NUMINAMATH_CALUDE_minimum_groups_l4096_409625

/-- A function that determines if a number belongs to the set G_k -/
def in_G_k (n : ℕ) (k : ℕ) : Prop :=
  n % 6 = k ∧ 1 ≤ n ∧ n ≤ 600

/-- A function that checks if two numbers can be in the same group -/
def can_be_in_same_group (a b : ℕ) : Prop :=
  (a + b) % 6 = 0

/-- A valid grouping of numbers -/
def valid_grouping (groups : List (List ℕ)) : Prop :=
  (∀ group ∈ groups, ∀ a ∈ group, ∀ b ∈ group, a ≠ b → can_be_in_same_group a b) ∧
  (∀ n, 1 ≤ n ∧ n ≤ 600 → ∃ group ∈ groups, n ∈ group)

theorem minimum_groups :
  ∃ (groups : List (List ℕ)), valid_grouping groups ∧
    (∀ (other_groups : List (List ℕ)), valid_grouping other_groups →
      groups.length ≤ other_groups.length) ∧
    groups.length = 202 := by
  sorry

end NUMINAMATH_CALUDE_minimum_groups_l4096_409625


namespace NUMINAMATH_CALUDE_subtraction_of_decimals_l4096_409645

/-- Subtraction of two specific decimal numbers -/
theorem subtraction_of_decimals : (678.90 : ℝ) - (123.45 : ℝ) = 555.55 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_decimals_l4096_409645


namespace NUMINAMATH_CALUDE_number_equality_l4096_409627

theorem number_equality (x : ℝ) : (0.4 * x = 0.25 * 80) → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_number_equality_l4096_409627


namespace NUMINAMATH_CALUDE_quadratic_coefficient_sum_l4096_409656

theorem quadratic_coefficient_sum (k : ℤ) : 
  (∃ x y : ℤ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + k*x + 25 = 0 ∧ y^2 + k*y + 25 = 0) → 
  k = 26 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_sum_l4096_409656


namespace NUMINAMATH_CALUDE_total_food_is_point_nine_l4096_409613

/-- The amount of cat food Jake needs to serve each day for one cat -/
def food_for_one_cat : ℝ := 0.5

/-- The extra amount of cat food needed for the second cat -/
def extra_food_for_second_cat : ℝ := 0.4

/-- The total amount of cat food Jake needs to serve each day for two cats -/
def total_food_for_two_cats : ℝ := food_for_one_cat + extra_food_for_second_cat

theorem total_food_is_point_nine :
  total_food_for_two_cats = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_total_food_is_point_nine_l4096_409613


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l4096_409608

theorem cubic_equation_solution (x y z : ℕ) : 
  x^3 + 4*y^3 = 16*z^3 + 4*x*y*z → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l4096_409608


namespace NUMINAMATH_CALUDE_new_city_buildings_count_l4096_409647

/-- Calculates the total number of buildings for the new city project --/
def new_city_buildings (pittsburgh_stores : ℕ) (pittsburgh_hospitals : ℕ) (pittsburgh_schools : ℕ) (pittsburgh_police : ℕ) : ℕ :=
  (pittsburgh_stores / 2) + (pittsburgh_hospitals * 2) + (pittsburgh_schools - 50) + (pittsburgh_police + 5)

/-- Theorem stating that the total number of buildings for the new city is 2175 --/
theorem new_city_buildings_count : 
  new_city_buildings 2000 500 200 20 = 2175 := by
  sorry

end NUMINAMATH_CALUDE_new_city_buildings_count_l4096_409647


namespace NUMINAMATH_CALUDE_inequality_proof_l4096_409648

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 / (a^3 + b^3 + a*b*c) + 1 / (b^3 + c^3 + a*b*c) + 1 / (c^3 + a^3 + a*b*c) ≤ 1 / (a*b*c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4096_409648


namespace NUMINAMATH_CALUDE_equation_solution_l4096_409678

theorem equation_solution :
  ∃ x : ℤ, 45 - (x - (37 - (15 - 18))) = 57 ∧ x = 28 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l4096_409678


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_diagonal_intersection_property_l4096_409658

/-- A point in a 2D plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A circle in a 2D plane -/
structure Circle :=
  (center : Point)
  (radius : ℝ)

/-- A quadrilateral in a 2D plane -/
structure Quadrilateral :=
  (A : Point)
  (B : Point)
  (C : Point)
  (D : Point)

/-- Checks if a quadrilateral is cyclic -/
def is_cyclic (q : Quadrilateral) (c : Circle) : Prop :=
  -- Definition omitted for brevity
  sorry

/-- Calculates the intersection point of two line segments -/
def intersection (p1 p2 p3 p4 : Point) : Point :=
  -- Definition omitted for brevity
  sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ :=
  -- Definition omitted for brevity
  sorry

/-- The main theorem -/
theorem cyclic_quadrilateral_diagonal_intersection_property
  (ABCD : Quadrilateral) (c : Circle) (X : Point) :
  is_cyclic ABCD c →
  X = intersection ABCD.A ABCD.C ABCD.B ABCD.D →
  distance X ABCD.A * distance X ABCD.C = distance X ABCD.B * distance X ABCD.D :=
by
  sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_diagonal_intersection_property_l4096_409658


namespace NUMINAMATH_CALUDE_no_natural_solutions_l4096_409681

theorem no_natural_solutions (k x y z : ℕ) (h : k > 3) :
  x^2 + y^2 + z^2 ≠ k * x * y * z :=
sorry

end NUMINAMATH_CALUDE_no_natural_solutions_l4096_409681


namespace NUMINAMATH_CALUDE_min_value_fraction_equality_condition_l4096_409666

theorem min_value_fraction (x : ℝ) (h : x > 8) : x^2 / (x - 8) ≥ 32 :=
by sorry

theorem equality_condition (x : ℝ) (h : x > 8) : 
  x^2 / (x - 8) = 32 ↔ x = 16 :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_equality_condition_l4096_409666


namespace NUMINAMATH_CALUDE_order_of_abc_l4096_409635

theorem order_of_abc (a b c : ℝ) 
  (h1 : Real.sqrt (1 + 2*a) = Real.exp b)
  (h2 : Real.exp b = 1 / (1 - c))
  (h3 : 1 / (1 - c) = 1.01) : 
  a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_order_of_abc_l4096_409635


namespace NUMINAMATH_CALUDE_quadratic_factor_difference_l4096_409642

/-- Given a quadratic expression that can be factored, prove the difference of its factors' constants -/
theorem quadratic_factor_difference (a b : ℤ) : 
  (∀ y, 2 * y^2 + 5 * y - 12 = (2 * y + a) * (y + b)) → 
  a - b = -7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factor_difference_l4096_409642


namespace NUMINAMATH_CALUDE_list_median_is_106_l4096_409637

def sequence_sum (n : ℕ) : ℕ := n * (n + 1) / 2

def list_length : ℕ := sequence_sum 150

def median_position : ℕ := (list_length + 1) / 2

theorem list_median_is_106 : ∃ (n : ℕ), 
  n = 106 ∧ 
  sequence_sum (n - 1) < median_position ∧ 
  median_position ≤ sequence_sum n :=
sorry

end NUMINAMATH_CALUDE_list_median_is_106_l4096_409637


namespace NUMINAMATH_CALUDE_harry_fish_count_harry_fish_count_proof_l4096_409663

/-- Given three friends with fish, prove Harry has 224 fish -/
theorem harry_fish_count : ℕ → ℕ → ℕ → Prop :=
  fun sam_fish joe_fish harry_fish =>
    sam_fish = 7 ∧
    joe_fish = 8 * sam_fish ∧
    harry_fish = 4 * joe_fish →
    harry_fish = 224

/-- Proof of the theorem -/
theorem harry_fish_count_proof : ∃ (sam_fish joe_fish harry_fish : ℕ),
  harry_fish_count sam_fish joe_fish harry_fish :=
by
  sorry

end NUMINAMATH_CALUDE_harry_fish_count_harry_fish_count_proof_l4096_409663


namespace NUMINAMATH_CALUDE_goose_price_after_increases_l4096_409633

-- Define the initial prices
def initial_goose_price : ℝ := 0.8
def initial_wine_price : ℝ := 0.4

-- Define the price increase factor
def price_increase_factor : ℝ := 1.2

theorem goose_price_after_increases (goose_price : ℝ) (wine_price : ℝ) :
  goose_price = initial_goose_price ∧ 
  wine_price = initial_wine_price ∧
  goose_price + wine_price = 1 ∧
  goose_price + 0.5 * wine_price = 1 →
  goose_price * price_increase_factor * price_increase_factor < 1 := by
  sorry

#check goose_price_after_increases

end NUMINAMATH_CALUDE_goose_price_after_increases_l4096_409633


namespace NUMINAMATH_CALUDE_min_side_triangle_l4096_409688

theorem min_side_triangle (a b c : ℝ) (A B C : ℝ) : 
  a + b = 2 → C = 2 * π / 3 → c ≥ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_side_triangle_l4096_409688


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l4096_409634

-- Define the universal set U
def U : Finset Nat := {0, 1, 2, 3}

-- Define set A
def A : Finset Nat := {1, 3}

-- Theorem statement
theorem complement_of_A_in_U : 
  (U \ A) = {0, 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l4096_409634


namespace NUMINAMATH_CALUDE_smallest_other_integer_l4096_409660

theorem smallest_other_integer (m n x : ℕ+) : 
  (m = 50 ∨ n = 50) →
  Nat.gcd m.val n.val = x.val + 5 →
  Nat.lcm m.val n.val = x.val * (x.val + 5) →
  (m ≠ 50 → m ≥ 10) ∧ (n ≠ 50 → n ≥ 10) :=
by sorry

end NUMINAMATH_CALUDE_smallest_other_integer_l4096_409660


namespace NUMINAMATH_CALUDE_polynomial_identity_l4096_409655

theorem polynomial_identity (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2*x + Real.sqrt 3)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l4096_409655


namespace NUMINAMATH_CALUDE_abs_square_of_complex_l4096_409659

theorem abs_square_of_complex (z : ℂ) : z = 5 + 2*I → Complex.abs (z^2) = 29 := by
  sorry

end NUMINAMATH_CALUDE_abs_square_of_complex_l4096_409659


namespace NUMINAMATH_CALUDE_multiples_of_5_or_7_not_35_l4096_409699

def count_multiples (n : ℕ) (d : ℕ) : ℕ := (n / d : ℕ)

theorem multiples_of_5_or_7_not_35 : 
  (count_multiples 3000 5) + (count_multiples 3000 7) - (count_multiples 3000 35) = 943 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_5_or_7_not_35_l4096_409699


namespace NUMINAMATH_CALUDE_fraction_problem_l4096_409692

theorem fraction_problem (x : ℚ) :
  (x / (4 * x + 5) = 3 / 7) → x = -3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l4096_409692


namespace NUMINAMATH_CALUDE_boat_downstream_distance_l4096_409636

/-- Calculates the distance traveled downstream by a boat given its own speed, speed against current, and time. -/
def distance_downstream (boat_speed : ℝ) (speed_against_current : ℝ) (time : ℝ) : ℝ :=
  let current_speed : ℝ := boat_speed - speed_against_current
  let downstream_speed : ℝ := boat_speed + current_speed
  downstream_speed * time

/-- Proves that a boat with given specifications travels 255 km downstream in 6 hours. -/
theorem boat_downstream_distance :
  distance_downstream 40 37.5 6 = 255 := by
  sorry

end NUMINAMATH_CALUDE_boat_downstream_distance_l4096_409636


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l4096_409605

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 1) ↔ (∃ x : ℝ, x ≤ 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l4096_409605


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l4096_409683

theorem sum_of_coefficients (a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, x * (1 - 2*x)^4 = a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₂ + a₃ + a₄ + a₅ = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l4096_409683


namespace NUMINAMATH_CALUDE_millet_exceeds_60_percent_on_day_4_l4096_409602

/-- Represents the state of the bird feeder on a given day -/
structure FeederState where
  millet : Float
  other_seeds : Float

/-- Calculates the next day's feeder state -/
def next_day_state (state : FeederState) : FeederState :=
  { millet := state.millet * 0.7 + 0.3,
    other_seeds := state.other_seeds * 0.5 + 0.7 }

/-- Calculates the proportion of millet in the feeder -/
def millet_proportion (state : FeederState) : Float :=
  state.millet / (state.millet + state.other_seeds)

/-- Initial state of the feeder -/
def initial_state : FeederState := { millet := 0.3, other_seeds := 0.7 }

theorem millet_exceeds_60_percent_on_day_4 :
  let day1 := initial_state
  let day2 := next_day_state day1
  let day3 := next_day_state day2
  let day4 := next_day_state day3
  (millet_proportion day1 ≤ 0.6) ∧
  (millet_proportion day2 ≤ 0.6) ∧
  (millet_proportion day3 ≤ 0.6) ∧
  (millet_proportion day4 > 0.6) :=
by sorry

end NUMINAMATH_CALUDE_millet_exceeds_60_percent_on_day_4_l4096_409602


namespace NUMINAMATH_CALUDE_isabel_camera_pictures_l4096_409673

/-- Represents the number of pictures in Isabel's photo upload scenario -/
structure IsabelPictures where
  phone : ℕ
  camera : ℕ
  albums : ℕ
  pics_per_album : ℕ

/-- The theorem stating the number of pictures Isabel uploaded from her camera -/
theorem isabel_camera_pictures (p : IsabelPictures) 
  (h1 : p.phone = 2)
  (h2 : p.albums = 3)
  (h3 : p.pics_per_album = 2)
  (h4 : p.albums * p.pics_per_album = p.phone + p.camera) :
  p.camera = 4 := by
  sorry

#check isabel_camera_pictures

end NUMINAMATH_CALUDE_isabel_camera_pictures_l4096_409673


namespace NUMINAMATH_CALUDE_problem_solution_l4096_409667

def M (a : ℝ) : Set ℝ := {x | x * (x - a - 1) < 0}
def N : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

theorem problem_solution :
  (M 1 = {x | 0 < x ∧ x < 2}) ∧
  ({a : ℝ | M a ⊆ N} = Set.Icc (-2) 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4096_409667


namespace NUMINAMATH_CALUDE_tennis_ball_ratio_l4096_409626

/-- The number of tennis balls originally ordered -/
def total_balls : ℕ := 114

/-- The number of extra yellow balls sent by mistake -/
def extra_yellow : ℕ := 50

/-- The number of white balls received -/
def white_balls : ℕ := total_balls / 2

/-- The number of yellow balls received -/
def yellow_balls : ℕ := total_balls / 2 + extra_yellow

/-- The ratio of white balls to yellow balls after the error -/
def ball_ratio : ℚ := white_balls / yellow_balls

theorem tennis_ball_ratio : ball_ratio = 57 / 107 := by sorry

end NUMINAMATH_CALUDE_tennis_ball_ratio_l4096_409626


namespace NUMINAMATH_CALUDE_triangle_longest_side_l4096_409630

/-- Given a triangle with sides of lengths 7, x+4, and 2x+1, and a perimeter of 36,
    prove that the length of the longest side is 17. -/
theorem triangle_longest_side (x : ℝ) : 
  (7 : ℝ) + (x + 4) + (2*x + 1) = 36 → 
  max 7 (max (x + 4) (2*x + 1)) = 17 :=
by sorry

end NUMINAMATH_CALUDE_triangle_longest_side_l4096_409630


namespace NUMINAMATH_CALUDE_total_people_all_tribes_l4096_409682

/-- Represents a tribe with cannoneers, women, and men -/
structure Tribe where
  cannoneers : ℕ
  women : ℕ
  men : ℕ

/-- Calculates the total number of people in a tribe -/
def total_people (t : Tribe) : ℕ := t.cannoneers + t.women + t.men

/-- Represents the conditions for Tribe A -/
def tribe_a : Tribe :=
  { cannoneers := 63,
    women := 2 * 63,
    men := 2 * (2 * 63) }

/-- Represents the conditions for Tribe B -/
def tribe_b : Tribe :=
  { cannoneers := 45,
    women := 45 / 3,
    men := 3 * (45 / 3) }

/-- Represents the conditions for Tribe C -/
def tribe_c : Tribe :=
  { cannoneers := 108,
    women := 108 / 2,
    men := 108 / 2 }

theorem total_people_all_tribes : 
  total_people tribe_a + total_people tribe_b + total_people tribe_c = 834 := by
  sorry

end NUMINAMATH_CALUDE_total_people_all_tribes_l4096_409682


namespace NUMINAMATH_CALUDE_area_between_circle_and_squares_l4096_409657

theorem area_between_circle_and_squares :
  let outer_square_side : ℝ := 2
  let circle_radius : ℝ := 1/2
  let inner_square_side : ℝ := 1.8
  let outer_square_area : ℝ := outer_square_side^2
  let inner_square_area : ℝ := inner_square_side^2
  let circle_area : ℝ := π * circle_radius^2
  let area_between : ℝ := outer_square_area - inner_square_area - (outer_square_area - circle_area)
  area_between = 0.76 := by sorry

end NUMINAMATH_CALUDE_area_between_circle_and_squares_l4096_409657


namespace NUMINAMATH_CALUDE_target_probability_value_l4096_409690

/-- The probability of hitting the target on a single shot -/
def hit_probability : ℝ := 0.85

/-- The probability of missing the target on a single shot -/
def miss_probability : ℝ := 1 - hit_probability

/-- The probability of missing the first two shots and hitting the third shot -/
def target_probability : ℝ := miss_probability * miss_probability * hit_probability

theorem target_probability_value : target_probability = 0.019125 := by
  sorry

end NUMINAMATH_CALUDE_target_probability_value_l4096_409690


namespace NUMINAMATH_CALUDE_only_13_remains_prime_l4096_409649

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def reverse_digits (n : ℕ) : ℕ :=
  let rec aux (n acc : ℕ) : ℕ :=
    if n = 0 then acc
    else aux (n / 10) (acc * 10 + n % 10)
  aux n 0

def remains_prime_when_reversed (n : ℕ) : Prop :=
  is_prime n ∧ is_prime (reverse_digits n)

theorem only_13_remains_prime : 
  (remains_prime_when_reversed 13) ∧ 
  (¬remains_prime_when_reversed 29) ∧ 
  (¬remains_prime_when_reversed 53) ∧ 
  (¬remains_prime_when_reversed 23) ∧ 
  (¬remains_prime_when_reversed 41) :=
sorry

end NUMINAMATH_CALUDE_only_13_remains_prime_l4096_409649


namespace NUMINAMATH_CALUDE_jason_has_36_seashells_l4096_409629

/-- The number of seashells Jason has now, given his initial count and the number he gave away. -/
def jasonsSeashells (initialCount gaveAway : ℕ) : ℕ :=
  initialCount - gaveAway

/-- Theorem stating that Jason has 36 seashells after giving some away. -/
theorem jason_has_36_seashells : jasonsSeashells 49 13 = 36 := by
  sorry

end NUMINAMATH_CALUDE_jason_has_36_seashells_l4096_409629


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l4096_409664

theorem quadratic_roots_relation (d e : ℝ) : 
  (∃ r s : ℝ, 2 * r^2 - 4 * r - 6 = 0 ∧ 2 * s^2 - 4 * s - 6 = 0 ∧
   ∀ x : ℝ, x^2 + d * x + e = 0 ↔ x = r - 3 ∨ x = s - 3) →
  e = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l4096_409664


namespace NUMINAMATH_CALUDE_football_team_selection_l4096_409697

theorem football_team_selection (n : ℕ) (k : ℕ) :
  let total_students : ℕ := 31
  let team_size : ℕ := 11
  let remaining_students : ℕ := total_students - 2
  (Nat.choose total_students team_size) - (Nat.choose remaining_students team_size) =
    2 * (Nat.choose remaining_students (team_size - 1)) + (Nat.choose remaining_students (team_size - 2)) :=
by sorry

end NUMINAMATH_CALUDE_football_team_selection_l4096_409697


namespace NUMINAMATH_CALUDE_pet_store_birds_l4096_409654

/-- The number of birds in a cage -/
def birds_in_cage (parrots parakeets finches cockatiels canaries lovebirds toucans : ℕ) : ℕ :=
  parrots + parakeets + finches + cockatiels + canaries + lovebirds + toucans

/-- The total number of birds in the pet store -/
def total_birds : ℕ :=
  birds_in_cage 6 2 0 0 0 0 0 +  -- Cage 1
  birds_in_cage 4 3 5 0 0 0 0 +  -- Cage 2
  birds_in_cage 2 4 0 1 0 0 0 +  -- Cage 3
  birds_in_cage 3 5 0 0 2 0 0 +  -- Cage 4
  birds_in_cage 7 0 0 0 0 4 0 +  -- Cage 5
  birds_in_cage 4 2 3 0 0 0 1    -- Cage 6

theorem pet_store_birds : total_birds = 58 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_birds_l4096_409654


namespace NUMINAMATH_CALUDE_r_amount_calculation_l4096_409687

def total_amount : ℝ := 5000

theorem r_amount_calculation (p_amount q_amount r_amount : ℝ) 
  (h1 : p_amount + q_amount + r_amount = total_amount)
  (h2 : r_amount = (2/3) * (p_amount + q_amount)) :
  r_amount = 2000 := by
  sorry

end NUMINAMATH_CALUDE_r_amount_calculation_l4096_409687


namespace NUMINAMATH_CALUDE_matching_color_probability_l4096_409665

-- Define the number of jelly beans for each person
def abe_green : ℕ := 1
def abe_red : ℕ := 2
def bob_green : ℕ := 2
def bob_yellow : ℕ := 1
def bob_red : ℕ := 1

-- Define the total number of jelly beans for each person
def abe_total : ℕ := abe_green + abe_red
def bob_total : ℕ := bob_green + bob_yellow + bob_red

-- Define the probability of matching colors
def prob_match : ℚ := (abe_green * bob_green + abe_red * bob_red) / (abe_total * bob_total)

-- Theorem statement
theorem matching_color_probability :
  prob_match = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_matching_color_probability_l4096_409665


namespace NUMINAMATH_CALUDE_rhombus_diagonals_l4096_409650

-- Define the rhombus
structure Rhombus where
  perimeter : ℝ
  diagonal_difference : ℝ
  diagonal1 : ℝ
  diagonal2 : ℝ

-- State the theorem
theorem rhombus_diagonals (r : Rhombus) :
  r.perimeter = 100 ∧ r.diagonal_difference = 34 →
  r.diagonal1 = 14 ∧ r.diagonal2 = 48 := by
  sorry


end NUMINAMATH_CALUDE_rhombus_diagonals_l4096_409650


namespace NUMINAMATH_CALUDE_sandys_change_l4096_409615

/-- Calculates Sandy's change after shopping for toys -/
theorem sandys_change 
  (football_price : ℝ)
  (baseball_price : ℝ)
  (basketball_price : ℝ)
  (football_count : ℕ)
  (baseball_count : ℕ)
  (basketball_count : ℕ)
  (pounds_paid : ℝ)
  (euros_paid : ℝ)
  (h1 : football_price = 9.14)
  (h2 : baseball_price = 6.81)
  (h3 : basketball_price = 7.95)
  (h4 : football_count = 3)
  (h5 : baseball_count = 2)
  (h6 : basketball_count = 4)
  (h7 : pounds_paid = 50)
  (h8 : euros_paid = 20) :
  let pounds_spent := football_price * football_count + baseball_price * baseball_count
  let euros_spent := basketball_price * basketball_count
  let pounds_change := pounds_paid - pounds_spent
  let euros_change := max (euros_paid - euros_spent) 0
  (pounds_change = 8.96 ∧ euros_change = 0) :=
by sorry

end NUMINAMATH_CALUDE_sandys_change_l4096_409615


namespace NUMINAMATH_CALUDE_sues_family_travel_l4096_409606

/-- Given a constant speed and travel time, calculates the distance traveled -/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: Sue's family traveled 300 miles to the campground -/
theorem sues_family_travel : distance_traveled 60 5 = 300 := by
  sorry

end NUMINAMATH_CALUDE_sues_family_travel_l4096_409606


namespace NUMINAMATH_CALUDE_quadrilateral_interior_angles_sum_l4096_409619

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- A quadrilateral is a polygon with 4 sides -/
def is_quadrilateral (n : ℕ) : Prop := n = 4

theorem quadrilateral_interior_angles_sum :
  ∀ n : ℕ, is_quadrilateral n → sum_interior_angles n = 360 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_interior_angles_sum_l4096_409619


namespace NUMINAMATH_CALUDE_logarithm_problem_l4096_409686

theorem logarithm_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hx1 : x ≠ 1) (hy1 : y ≠ 1)
  (h1 : Real.log x / Real.log 3 = Real.log 81 / Real.log y)
  (h2 : x * y^2 = 729) :
  (Real.log (x / y) / Real.log 3)^2 = (206 - 90 * Real.sqrt 5) / 4 := by
sorry

end NUMINAMATH_CALUDE_logarithm_problem_l4096_409686


namespace NUMINAMATH_CALUDE_delores_initial_amount_l4096_409679

/-- The amount of money Delores had initially -/
def initial_amount : ℕ := sorry

/-- The cost of the computer -/
def computer_cost : ℕ := 400

/-- The cost of the printer -/
def printer_cost : ℕ := 40

/-- The amount of money Delores had left after purchases -/
def remaining_amount : ℕ := 10

/-- Theorem stating that Delores' initial amount was $450 -/
theorem delores_initial_amount : 
  initial_amount = computer_cost + printer_cost + remaining_amount := by sorry

end NUMINAMATH_CALUDE_delores_initial_amount_l4096_409679


namespace NUMINAMATH_CALUDE_discount_percentage_proof_l4096_409603

/-- Given a marked price and cost price, where the cost price is 25% of the marked price,
    and a discount percentage such that the selling price after discount is equal to twice
    the cost price, prove that the discount percentage is 50%. -/
theorem discount_percentage_proof (MP CP : ℝ) (D : ℝ) 
    (h1 : CP = 0.25 * MP) 
    (h2 : MP * (1 - D / 100) = 2 * CP) : 
  D = 50 := by
  sorry

#check discount_percentage_proof

end NUMINAMATH_CALUDE_discount_percentage_proof_l4096_409603


namespace NUMINAMATH_CALUDE_braking_distance_properties_l4096_409640

/-- Represents the braking distance in meters -/
def braking_distance (v : ℝ) : ℝ := 0.25 * v

/-- The maximum legal speed on highways in km/h -/
def max_legal_speed : ℝ := 120

theorem braking_distance_properties :
  (braking_distance 60 = 15) ∧
  (braking_distance 128 = 32) ∧
  (128 > max_legal_speed) := by
  sorry

end NUMINAMATH_CALUDE_braking_distance_properties_l4096_409640


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l4096_409641

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if a point (x, y) lies on a line -/
def lies_on (x y : ℝ) (l : Line) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- The problem statement -/
theorem line_through_point_parallel_to_line :
  ∃ (l : Line),
    lies_on (-1) 2 l ∧
    parallel l { a := 2, b := -3, c := 4 } ∧
    l = { a := 2, b := -3, c := 8 } := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l4096_409641


namespace NUMINAMATH_CALUDE_salary_increase_proof_l4096_409638

def original_salary : ℝ := 60
def percentage_increase : ℝ := 13.333333333333334
def new_salary : ℝ := 68

theorem salary_increase_proof :
  original_salary * (1 + percentage_increase / 100) = new_salary := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_proof_l4096_409638


namespace NUMINAMATH_CALUDE_quadratic_factorization_l4096_409624

theorem quadratic_factorization (a : ℝ) : 
  (∃ m n : ℝ, ∀ x y : ℝ, 
    x^2 + 7*x*y + a*y^2 - 5*x - 45*y - 24 = (x - 8 + m*y) * (x + 3 + n*y)) → 
  a = 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l4096_409624


namespace NUMINAMATH_CALUDE_volleyball_club_girls_l4096_409600

theorem volleyball_club_girls (total : ℕ) (present : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 36 →
  present = 24 →
  boys + girls = total →
  boys + (1/3 : ℚ) * girls = present →
  girls = 18 :=
by sorry

end NUMINAMATH_CALUDE_volleyball_club_girls_l4096_409600


namespace NUMINAMATH_CALUDE_marbles_left_l4096_409662

def initial_marbles : ℕ := 350
def marbles_given : ℕ := 175

theorem marbles_left : initial_marbles - marbles_given = 175 := by
  sorry

end NUMINAMATH_CALUDE_marbles_left_l4096_409662


namespace NUMINAMATH_CALUDE_thirtieth_triangular_number_properties_l4096_409610

/-- Calculate the nth triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Calculate the number of dots in the perimeter of the nth triangular figure -/
def perimeter_dots (n : ℕ) : ℕ := n + 2 * (n - 1)

theorem thirtieth_triangular_number_properties :
  (triangular_number 30 = 465) ∧ (perimeter_dots 30 = 88) := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_triangular_number_properties_l4096_409610


namespace NUMINAMATH_CALUDE_smallest_bob_number_l4096_409601

def alice_number : ℕ := 36

def has_all_prime_factors (n m : ℕ) : Prop :=
  ∀ p : ℕ, p.Prime → (p ∣ n → p ∣ m)

theorem smallest_bob_number :
  ∃ (bob_number : ℕ), 
    bob_number > 0 ∧
    has_all_prime_factors alice_number bob_number ∧
    (∀ k : ℕ, k > 0 ∧ has_all_prime_factors alice_number k → bob_number ≤ k) ∧
    bob_number = 6 := by
  sorry

end NUMINAMATH_CALUDE_smallest_bob_number_l4096_409601
