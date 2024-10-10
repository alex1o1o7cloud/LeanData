import Mathlib

namespace inequality_holds_iff_a_in_range_l1188_118833

theorem inequality_holds_iff_a_in_range (a : ℝ) : 
  (∀ (x : ℝ) (θ : ℝ), 0 ≤ θ ∧ θ ≤ π/2 → 
    (x + 3 + 2 * Real.sin θ * Real.cos θ)^2 + (x + a * Real.sin θ + a * Real.cos θ)^2 ≥ 1/8) ↔ 
  (a ≥ 7/2 ∨ a ≤ Real.sqrt 6) :=
by sorry

end inequality_holds_iff_a_in_range_l1188_118833


namespace power_function_value_l1188_118882

-- Define a power function type
def PowerFunction := ℝ → ℝ

-- Define the property of passing through a point for a power function
def PassesThroughPoint (f : PowerFunction) (x y : ℝ) : Prop :=
  f x = y

-- State the theorem
theorem power_function_value (f : PowerFunction) :
  PassesThroughPoint f 9 (1/3) → f 25 = 1/5 := by
  sorry

end power_function_value_l1188_118882


namespace plot_perimeter_is_180_l1188_118874

/-- A rectangular plot with specific dimensions and fencing cost -/
structure RectangularPlot where
  width : ℝ
  length : ℝ
  fencingRate : ℝ
  totalFencingCost : ℝ
  lengthWidthRelation : length = width + 10
  costRelation : totalFencingCost = fencingRate * (2 * (length + width))

/-- The perimeter of a rectangular plot -/
def perimeter (plot : RectangularPlot) : ℝ :=
  2 * (plot.length + plot.width)

/-- Theorem: The perimeter of the specific plot is 180 meters -/
theorem plot_perimeter_is_180 (plot : RectangularPlot)
  (h1 : plot.fencingRate = 6.5)
  (h2 : plot.totalFencingCost = 1170) :
  perimeter plot = 180 := by
  sorry

end plot_perimeter_is_180_l1188_118874


namespace negation_or_implies_both_false_l1188_118899

theorem negation_or_implies_both_false (p q : Prop) :
  ¬(p ∨ q) → (¬p ∧ ¬q) := by
  sorry

end negation_or_implies_both_false_l1188_118899


namespace min_value_cube_root_plus_inverse_square_l1188_118814

theorem min_value_cube_root_plus_inverse_square (x : ℝ) (h : x > 0) :
  3 * x^(1/3) + 1 / x^2 ≥ 4 ∧
  (3 * x^(1/3) + 1 / x^2 = 4 ↔ x = 1) :=
by sorry

end min_value_cube_root_plus_inverse_square_l1188_118814


namespace consecutive_odd_numbers_sum_l1188_118842

theorem consecutive_odd_numbers_sum (n₁ n₂ n₃ : ℕ) : 
  n₁ = 9 →
  n₂ = n₁ + 2 →
  n₃ = n₂ + 2 →
  Odd n₁ →
  Odd n₂ →
  Odd n₃ →
  11 * n₁ - (3 * n₃ + 4 * n₂) = 16 :=
by sorry

end consecutive_odd_numbers_sum_l1188_118842


namespace area_ratio_squares_l1188_118851

/-- Given squares A, B, and C with the following properties:
  - The perimeter of square A is 16 units
  - The perimeter of square B is 32 units
  - The side length of square C is 4 times the side length of square B
  Prove that the ratio of the area of square B to the area of square C is 1/16 -/
theorem area_ratio_squares (a b c : ℝ) 
  (ha : 4 * a = 16) 
  (hb : 4 * b = 32) 
  (hc : c = 4 * b) : 
  (b ^ 2) / (c ^ 2) = 1 / 16 := by
sorry

end area_ratio_squares_l1188_118851


namespace gcd_735_1287_l1188_118868

theorem gcd_735_1287 : Nat.gcd 735 1287 = 3 := by
  sorry

end gcd_735_1287_l1188_118868


namespace a_n_properties_smallest_n_perfect_square_sum_l1188_118855

/-- The largest n-digit number that is neither the sum nor the difference of two perfect squares -/
def a_n (n : ℕ) : ℕ := 10^n - 2

/-- The sum of squares of digits of a number -/
def sum_of_squares_of_digits (m : ℕ) : ℕ := sorry

/-- Theorem stating the properties of a_n -/
theorem a_n_properties :
  ∀ (n : ℕ), n > 2 →
  (∀ (x y : ℕ), a_n n ≠ x^2 + y^2 ∧ a_n n ≠ x^2 - y^2) ∧
  (∀ (m : ℕ), m < n → ∃ (x y : ℕ), 10^m - 2 = x^2 + y^2 ∨ 10^m - 2 = x^2 - y^2) :=
sorry

/-- Theorem stating the smallest n for which the sum of squares of digits of a_n is a perfect square -/
theorem smallest_n_perfect_square_sum :
  ∃ (k : ℕ), sum_of_squares_of_digits (a_n 66) = k^2 ∧
  ∀ (n : ℕ), n < 66 → ¬∃ (k : ℕ), sum_of_squares_of_digits (a_n n) = k^2 :=
sorry

end a_n_properties_smallest_n_perfect_square_sum_l1188_118855


namespace sum_of_repeating_decimals_l1188_118831

def repeating_decimal_12 : ℚ := 4 / 33
def repeating_decimal_03 : ℚ := 1 / 33
def repeating_decimal_006 : ℚ := 2 / 333

theorem sum_of_repeating_decimals :
  repeating_decimal_12 + repeating_decimal_03 + repeating_decimal_006 = 19041 / 120879 :=
by sorry

end sum_of_repeating_decimals_l1188_118831


namespace cone_radius_l1188_118887

/-- Given a cone with slant height 5 cm and lateral surface area 15π cm², 
    prove that the radius of the base is 3 cm. -/
theorem cone_radius (l : ℝ) (A : ℝ) (r : ℝ) : 
  l = 5 → A = 15 * Real.pi → A = Real.pi * r * l → r = 3 := by
  sorry

end cone_radius_l1188_118887


namespace existence_of_m_l1188_118862

def x : ℕ → ℚ
  | 0 => 7
  | n + 1 => (x n ^ 2 + 7 * x n + 8) / (x n + 8)

theorem existence_of_m : ∃ m : ℕ, 
  123 ≤ m ∧ m ≤ 242 ∧ 
  x m ≤ 6 + 1 / (2^18) ∧
  ∀ k : ℕ, 0 < k ∧ k < m → x k > 6 + 1 / (2^18) := by
  sorry

end existence_of_m_l1188_118862


namespace divisors_not_div_by_seven_l1188_118845

def number_to_factorize : ℕ := 420

-- Define a function to count divisors not divisible by 7
def count_divisors_not_div_by_seven (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem divisors_not_div_by_seven :
  count_divisors_not_div_by_seven number_to_factorize = 12 := by sorry

end divisors_not_div_by_seven_l1188_118845


namespace parking_lot_problem_l1188_118888

theorem parking_lot_problem (medium_fee small_fee total_cars total_fee : ℕ)
  (h1 : medium_fee = 15)
  (h2 : small_fee = 8)
  (h3 : total_cars = 30)
  (h4 : total_fee = 324) :
  ∃ (medium_cars small_cars : ℕ),
    medium_cars + small_cars = total_cars ∧
    medium_cars * medium_fee + small_cars * small_fee = total_fee ∧
    medium_cars = 12 ∧
    small_cars = 18 :=
by sorry

end parking_lot_problem_l1188_118888


namespace points_eight_units_from_negative_three_l1188_118881

def distance (x y : ℝ) : ℝ := |x - y|

theorem points_eight_units_from_negative_three :
  ∀ x : ℝ, distance x (-3) = 8 ↔ x = -11 ∨ x = 5 := by
  sorry

end points_eight_units_from_negative_three_l1188_118881


namespace absolute_value_inequality_l1188_118839

theorem absolute_value_inequality (x : ℝ) : 
  abs x + abs (2 * x - 3) ≥ 6 ↔ x ≤ -1 ∨ x ≥ 3 := by
  sorry

end absolute_value_inequality_l1188_118839


namespace bryan_bus_time_l1188_118802

/-- Represents the travel time for Bryan's commute -/
structure CommuteTimes where
  walkToStation : ℕ  -- Time to walk from house to bus station
  walkToWork : ℕ     -- Time to walk from bus station to work
  totalYearlyTime : ℕ -- Total yearly commute time in hours
  daysWorked : ℕ     -- Number of days worked per year

/-- Calculates the one-way bus ride time in minutes -/
def onewayBusTime (c : CommuteTimes) : ℕ :=
  let totalDailyTime := (c.totalYearlyTime * 60) / c.daysWorked
  let totalWalkTime := 2 * (c.walkToStation + c.walkToWork)
  (totalDailyTime - totalWalkTime) / 2

/-- Theorem stating that Bryan's one-way bus ride time is 20 minutes -/
theorem bryan_bus_time :
  let c := CommuteTimes.mk 5 5 365 365
  onewayBusTime c = 20 := by
  sorry

end bryan_bus_time_l1188_118802


namespace boat_length_boat_length_is_three_l1188_118804

/-- The length of a boat given its breadth, sinking depth, and the mass of a man. -/
theorem boat_length (breadth : ℝ) (sinking_depth : ℝ) (man_mass : ℝ) 
  (water_density : ℝ) (gravity : ℝ) : ℝ :=
  let volume := man_mass * gravity / (water_density * gravity)
  volume / (breadth * sinking_depth)

/-- Proof that the length of the boat is 3 meters given specific conditions. -/
theorem boat_length_is_three :
  boat_length 2 0.01 60 1000 9.81 = 3 := by
  sorry

end boat_length_boat_length_is_three_l1188_118804


namespace quadratic_function_domain_range_conditions_l1188_118841

/-- Given a quadratic function f(x) = -1/2 * x^2 + x with domain [m, n] and range [k*m, k*n],
    prove that m = 2(1 - k) and n = 0 must be satisfied. -/
theorem quadratic_function_domain_range_conditions
  (f : ℝ → ℝ)
  (m n k : ℝ)
  (h_f : ∀ x, f x = -1/2 * x^2 + x)
  (h_domain : Set.Icc m n = {x | f x ∈ Set.Icc (k * m) (k * n)})
  (h_m_lt_n : m < n)
  (h_k_gt_1 : k > 1) :
  m = 2 * (1 - k) ∧ n = 0 := by
  sorry

end quadratic_function_domain_range_conditions_l1188_118841


namespace unique_number_with_gcd_l1188_118890

theorem unique_number_with_gcd (n : ℕ) : 
  70 < n ∧ n < 80 ∧ Nat.gcd 15 n = 5 → n = 75 := by
  sorry

end unique_number_with_gcd_l1188_118890


namespace complex_sum_zero_l1188_118801

theorem complex_sum_zero : (1 - Complex.I) ^ 10 + (1 + Complex.I) ^ 10 = 0 := by sorry

end complex_sum_zero_l1188_118801


namespace triangle_sides_simplification_l1188_118892

theorem triangle_sides_simplification (a b c : ℝ) 
  (h1 : a + b > c) 
  (h2 : b + c > a) 
  (h3 : a + c > b) 
  (h4 : a > 0) 
  (h5 : b > 0) 
  (h6 : c > 0) : 
  |c - a - b| + |c + b - a| = 2 * b := by
  sorry

end triangle_sides_simplification_l1188_118892


namespace system_solution_l1188_118807

theorem system_solution (x y : ℝ) : 
  (x^2 + x*y + y^2) / (x^2 - x*y + y^2) = 3 →
  x^3 + y^3 = 2 →
  x = 1 ∧ y = 1 := by
sorry

end system_solution_l1188_118807


namespace total_books_l1188_118810

/-- The number of books on a mystery shelf -/
def mystery_books_per_shelf : ℕ := 7

/-- The number of books on a picture book shelf -/
def picture_books_per_shelf : ℕ := 5

/-- The number of books on a science fiction shelf -/
def scifi_books_per_shelf : ℕ := 8

/-- The number of books on a biography shelf -/
def biography_books_per_shelf : ℕ := 6

/-- The number of mystery shelves -/
def mystery_shelves : ℕ := 8

/-- The number of picture book shelves -/
def picture_shelves : ℕ := 2

/-- The number of science fiction shelves -/
def scifi_shelves : ℕ := 3

/-- The number of biography shelves -/
def biography_shelves : ℕ := 4

/-- The total number of books on Megan's shelves -/
theorem total_books : 
  mystery_books_per_shelf * mystery_shelves + 
  picture_books_per_shelf * picture_shelves + 
  scifi_books_per_shelf * scifi_shelves + 
  biography_books_per_shelf * biography_shelves = 114 := by
  sorry

end total_books_l1188_118810


namespace probability_two_fives_l1188_118866

def num_dice : ℕ := 12
def num_sides : ℕ := 6
def target_value : ℕ := 5
def target_count : ℕ := 2

theorem probability_two_fives (num_dice : ℕ) (num_sides : ℕ) (target_value : ℕ) (target_count : ℕ) :
  num_dice = 12 →
  num_sides = 6 →
  target_value = 5 →
  target_count = 2 →
  (Nat.choose num_dice target_count : ℚ) * (1 / num_sides : ℚ)^target_count * ((num_sides - 1) / num_sides : ℚ)^(num_dice - target_count) =
  (66 * 5^10 : ℚ) / 6^12 :=
by sorry

end probability_two_fives_l1188_118866


namespace ninety_seventh_rising_number_l1188_118844

/-- A rising number is a positive integer where each digit is larger than each of the digits to its left. -/
def IsRisingNumber (n : ℕ) : Prop := sorry

/-- The total count of five-digit rising numbers. -/
def TotalFiveDigitRisingNumbers : ℕ := 126

/-- The nth five-digit rising number when arranged from smallest to largest. -/
def NthFiveDigitRisingNumber (n : ℕ) : ℕ := sorry

theorem ninety_seventh_rising_number :
  NthFiveDigitRisingNumber 97 = 24678 := by sorry

end ninety_seventh_rising_number_l1188_118844


namespace mia_tv_watching_time_l1188_118826

def minutes_in_day : ℕ := 1440

def studying_minutes : ℕ := 288

theorem mia_tv_watching_time :
  ∃ (x : ℚ), 
    x > 0 ∧ 
    x < 1 ∧ 
    (1 / 4 : ℚ) * (1 - x) * minutes_in_day = studying_minutes ∧
    x = 1 / 5 := by
  sorry

end mia_tv_watching_time_l1188_118826


namespace hcl_moles_in_reaction_l1188_118876

-- Define the reaction components
structure ReactionComponent where
  name : String
  moles : ℚ

-- Define the reaction
def reaction (hcl koh kcl h2o : ReactionComponent) : Prop :=
  hcl.name = "HCl" ∧ koh.name = "KOH" ∧ kcl.name = "KCl" ∧ h2o.name = "H2O" ∧
  hcl.moles = koh.moles ∧ hcl.moles = kcl.moles ∧ hcl.moles = h2o.moles

-- Theorem statement
theorem hcl_moles_in_reaction 
  (hcl koh kcl h2o : ReactionComponent)
  (h1 : reaction hcl koh kcl h2o)
  (h2 : koh.moles = 1)
  (h3 : kcl.moles = 1) :
  hcl.moles = 1 := by
  sorry


end hcl_moles_in_reaction_l1188_118876


namespace volume_of_inscribed_sphere_l1188_118873

theorem volume_of_inscribed_sphere (edge_length : ℝ) (h : edge_length = 6) :
  let radius : ℝ := edge_length / 2
  let sphere_volume : ℝ := (4 / 3) * Real.pi * radius ^ 3
  sphere_volume = 36 * Real.pi := by
  sorry

end volume_of_inscribed_sphere_l1188_118873


namespace max_abs_z_value_l1188_118898

theorem max_abs_z_value (a b c z : ℂ) 
  (h1 : Complex.abs a = Complex.abs b)
  (h2 : Complex.abs a = 2 * Complex.abs c)
  (h3 : Complex.abs a > 0)
  (h4 : a * z^2 + b * z + c = 0) :
  Complex.abs z ≤ (1 + Real.sqrt 3) / 2 :=
sorry

end max_abs_z_value_l1188_118898


namespace puppies_given_to_friends_l1188_118854

/-- The number of puppies Alyssa started with -/
def initial_puppies : ℕ := 12

/-- The number of puppies Alyssa has left -/
def remaining_puppies : ℕ := 5

/-- The number of puppies Alyssa gave to her friends -/
def given_puppies : ℕ := initial_puppies - remaining_puppies

theorem puppies_given_to_friends : given_puppies = 7 := by
  sorry

end puppies_given_to_friends_l1188_118854


namespace original_numbers_proof_l1188_118817

theorem original_numbers_proof (x y : ℤ) 
  (sum_condition : x + y = 2022)
  (modified_sum_condition : (x - 5) / 10 + 10 * y + 1 = 2252) :
  x = 1815 ∧ y = 207 := by
  sorry

end original_numbers_proof_l1188_118817


namespace exam_marks_percentage_l1188_118847

theorem exam_marks_percentage (full_marks A_marks B_marks C_marks D_marks : ℝ) : 
  full_marks = 500 →
  A_marks = B_marks * 0.9 →
  B_marks = C_marks * 1.25 →
  C_marks = D_marks * 0.8 →
  A_marks = 360 →
  D_marks / full_marks = 0.8 :=
by sorry

end exam_marks_percentage_l1188_118847


namespace f_monotonicity_and_intersection_l1188_118870

noncomputable def f (x : ℝ) := x^3 - 3*x - 1

theorem f_monotonicity_and_intersection (x : ℝ) :
  (∀ x₁ x₂, x₁ < x₂ ∧ ((x₁ < -1 ∧ x₂ < -1) ∨ (x₁ > 1 ∧ x₂ > 1)) → f x₁ < f x₂) ∧
  (∀ x₁ x₂, -1 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 1 → f x₁ ≥ f x₂) ∧
  (∀ m : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧ f x₁ = m ∧ f x₂ = m ∧ f x₃ = m) ↔ -3 < m ∧ m < 1) :=
by sorry

end f_monotonicity_and_intersection_l1188_118870


namespace cube_equation_solution_l1188_118843

theorem cube_equation_solution :
  ∃! x : ℝ, (12 - x)^3 = x^3 ∧ x = 12 := by sorry

end cube_equation_solution_l1188_118843


namespace cos_135_and_point_on_unit_circle_l1188_118813

theorem cos_135_and_point_on_unit_circle :
  let angle : Real := 135 * π / 180
  let Q : ℝ × ℝ := (Real.cos angle, Real.sin angle)
  (Real.cos angle = -Real.sqrt 2 / 2) ∧
  (Q = (-Real.sqrt 2 / 2, Real.sqrt 2 / 2)) := by
  sorry

end cos_135_and_point_on_unit_circle_l1188_118813


namespace not_geometric_complement_sequence_l1188_118825

/-- Given a geometric sequence a, b, c with common ratio q ≠ 1,
    prove that 1-a, 1-b, 1-c cannot form a geometric sequence. -/
theorem not_geometric_complement_sequence 
  (a b c q : ℝ) 
  (h1 : b = a * q) 
  (h2 : c = b * q) 
  (h3 : q ≠ 1) : 
  ¬ ∃ r : ℝ, (1 - b = (1 - a) * r ∧ 1 - c = (1 - b) * r) :=
sorry

end not_geometric_complement_sequence_l1188_118825


namespace reflection_of_P_across_x_axis_l1188_118859

/-- Represents a point in 2D Cartesian coordinates -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis -/
def reflectAcrossXAxis (p : Point2D) : Point2D :=
  { x := p.x, y := -p.y }

theorem reflection_of_P_across_x_axis :
  let P : Point2D := { x := 2, y := -3 }
  reflectAcrossXAxis P = { x := 2, y := 3 } := by
  sorry

end reflection_of_P_across_x_axis_l1188_118859


namespace linear_equation_integer_solution_l1188_118803

theorem linear_equation_integer_solution : ∃ (x y : ℤ), 2 * x + y - 1 = 0 := by
  sorry

end linear_equation_integer_solution_l1188_118803


namespace consecutive_even_integers_sum_l1188_118836

theorem consecutive_even_integers_sum (x : ℝ) :
  (x - 2) * x * (x + 2) = 48 * ((x - 2) + x + (x + 2)) →
  (x - 2) + x + (x + 2) = 6 * Real.sqrt 37 := by
  sorry

end consecutive_even_integers_sum_l1188_118836


namespace points_same_side_of_line_l1188_118812

theorem points_same_side_of_line (a : ℝ) : 
  (∃ (s : ℝ), s * (3 * 3 - 2 * 1 + a) > 0 ∧ s * (3 * (-4) - 2 * 6 + a) > 0) ↔ 
  (a < -7 ∨ a > 24) :=
sorry

end points_same_side_of_line_l1188_118812


namespace triangle_angle_C_l1188_118896

noncomputable def f (x θ : Real) : Real :=
  2 * Real.sin x * Real.cos (θ / 2) ^ 2 + Real.cos x * Real.sin θ - Real.sin x

theorem triangle_angle_C (θ A B C : Real) (a b c : Real) :
  0 < θ ∧ θ < Real.pi →
  f A θ = Real.sqrt 3 / 2 →
  a = 1 →
  b = Real.sqrt 2 →
  A + B + C = Real.pi →
  Real.sin A / a = Real.sin B / b →
  Real.sin A / a = Real.sin C / c →
  (C = 7 * Real.pi / 12 ∨ C = Real.pi / 12) := by
  sorry

end triangle_angle_C_l1188_118896


namespace quadratic_real_root_and_inequality_l1188_118850

theorem quadratic_real_root_and_inequality (a b c : ℝ) :
  (∃ x : ℝ, (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a) = 0) ∧
  (a + b + c)^2 ≥ 3 * (a * b + b * c + c * a) := by
  sorry

end quadratic_real_root_and_inequality_l1188_118850


namespace orchard_trees_l1188_118805

theorem orchard_trees (total : ℕ) (pure_fuji : ℕ) (cross_pollinated : ℕ) (pure_gala : ℕ) :
  pure_gala = 39 →
  cross_pollinated = (total : ℚ) * (1 / 10) →
  pure_fuji = (total : ℚ) * (3 / 4) →
  pure_fuji + pure_gala + cross_pollinated = total →
  pure_fuji + cross_pollinated = 221 := by
sorry

end orchard_trees_l1188_118805


namespace disjunction_false_implies_both_false_l1188_118834

theorem disjunction_false_implies_both_false (p q : Prop) :
  (¬(p ∨ q)) → (¬p ∧ ¬q) := by
  sorry

end disjunction_false_implies_both_false_l1188_118834


namespace expression_evaluation_l1188_118897

theorem expression_evaluation :
  3 + 3 * Real.sqrt 3 + 1 / (3 + Real.sqrt 3) + 1 / (3 - Real.sqrt 3) = 4 + 3 * Real.sqrt 3 :=
by sorry

end expression_evaluation_l1188_118897


namespace min_operation_result_l1188_118856

def S : Finset Nat := {4, 6, 8, 12, 14, 18}

def operation (a b c : Nat) : Nat :=
  (a + b) * c - min a (min b c)

theorem min_operation_result :
  ∃ (result : Nat), result = 52 ∧
  ∀ (a b c : Nat), a ∈ S → b ∈ S → c ∈ S →
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  operation a b c ≥ result :=
sorry

end min_operation_result_l1188_118856


namespace sector_central_angle_l1188_118893

theorem sector_central_angle (area : ℝ) (perimeter : ℝ) (h1 : area = 5) (h2 : perimeter = 9) :
  ∃ (r : ℝ) (l : ℝ),
    2 * r + l = perimeter ∧
    1/2 * l * r = area ∧
    (l / r = 5/2 ∨ l / r = 8/5) :=
sorry

end sector_central_angle_l1188_118893


namespace isosceles_when_neg_one_is_root_right_triangle_when_equal_roots_equilateral_triangle_roots_l1188_118858

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0

/-- The quadratic equation associated with the triangle -/
def quadratic (t : Triangle) (x : ℝ) : ℝ :=
  (t.a + t.c) * x^2 + 2 * t.b * x + (t.a - t.c)

theorem isosceles_when_neg_one_is_root (t : Triangle) :
  quadratic t (-1) = 0 → t.a = t.b :=
sorry

theorem right_triangle_when_equal_roots (t : Triangle) :
  (2 * t.b)^2 = 4 * (t.a + t.c) * (t.a - t.c) → t.a^2 = t.b^2 + t.c^2 :=
sorry

theorem equilateral_triangle_roots (t : Triangle) (h : t.a = t.b ∧ t.b = t.c) :
  ∃ x y, x = 0 ∧ y = -1 ∧ quadratic t x = 0 ∧ quadratic t y = 0 :=
sorry

end isosceles_when_neg_one_is_root_right_triangle_when_equal_roots_equilateral_triangle_roots_l1188_118858


namespace function_divisibility_property_l1188_118895

def PositiveInt := {n : ℕ // n > 0}

theorem function_divisibility_property 
  (f : PositiveInt → PositiveInt) 
  (h : ∀ (m n : PositiveInt), (m.val^2 + (f n).val) ∣ (m.val * (f m).val + n.val)) :
  ∀ (n : PositiveInt), (f n).val = n.val :=
sorry

end function_divisibility_property_l1188_118895


namespace power_product_equality_l1188_118815

theorem power_product_equality : 3^2 * 5 * 7^2 * 11 = 24255 := by sorry

end power_product_equality_l1188_118815


namespace pentagonal_faces_count_l1188_118838

/-- A convex polyhedron with pentagon and hexagon faces -/
structure ConvexPolyhedron where
  -- Number of pentagonal faces
  n : ℕ
  -- Number of hexagonal faces
  k : ℕ
  -- The polyhedron is convex
  convex : True
  -- Faces are either pentagons or hexagons
  faces_pentagon_or_hexagon : True
  -- Exactly three edges meet at each vertex
  three_edges_per_vertex : True

/-- The number of pentagonal faces in a convex polyhedron with pentagon and hexagon faces -/
theorem pentagonal_faces_count (p : ConvexPolyhedron) : p.n = 12 := by
  sorry

end pentagonal_faces_count_l1188_118838


namespace division_remainder_problem_l1188_118871

theorem division_remainder_problem :
  let dividend : ℕ := 171
  let divisor : ℕ := 21
  let quotient : ℕ := 8
  let remainder : ℕ := dividend - divisor * quotient
  remainder = 3 := by sorry

end division_remainder_problem_l1188_118871


namespace g_neg_three_l1188_118806

def g (x : ℝ) : ℝ := x^2 - x + 2*x^3

theorem g_neg_three : g (-3) = -42 := by sorry

end g_neg_three_l1188_118806


namespace range_a_all_real_range_a_interval_l1188_118811

/-- The function f(x) = x^2 + ax + 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 3

/-- Theorem for the range of 'a' when f(x) ≥ a for all real x -/
theorem range_a_all_real (a : ℝ) :
  (∀ x : ℝ, f a x ≥ a) ↔ a ≤ 3 :=
sorry

/-- Theorem for the range of 'a' when f(x) ≥ a for x in [-2, 2] -/
theorem range_a_interval (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (-2) 2 → f a x ≥ a) ↔ a ∈ Set.Icc (-6) 2 :=
sorry

end range_a_all_real_range_a_interval_l1188_118811


namespace shooting_stars_count_l1188_118821

theorem shooting_stars_count (bridget reginald sam emma max : ℕ) : 
  bridget = 14 →
  reginald = bridget - 2 →
  sam = reginald + 4 →
  emma = sam + 3 →
  max = bridget - 7 →
  sam - ((bridget + reginald + sam + emma + max) / 5 : ℚ) = 2.4 :=
by sorry

end shooting_stars_count_l1188_118821


namespace inequality_solution_range_function_minimum_value_l1188_118830

-- Part 1
theorem inequality_solution_range (a : ℝ) :
  (∀ x : ℝ, |x + 1| + |x - 2| ≥ a) → a ≤ 3 := by sorry

-- Part 2
theorem function_minimum_value (a : ℝ) :
  (∃ m : ℝ, m = 5 ∧ ∀ x : ℝ, |x + 1| + 2 * |x - a| ≥ m) →
  a = 4 ∨ a = -6 := by sorry

end inequality_solution_range_function_minimum_value_l1188_118830


namespace hyperbola_asymptotes_l1188_118869

/-- Given a hyperbola with equation x²/9 - y²/m = 1 and one focus at (-5, 0),
    prove that its asymptotes are y = ±(4/3)x -/
theorem hyperbola_asymptotes (m : ℝ) :
  (∃ (x y : ℝ), x^2/9 - y^2/m = 1) →
  (5^2 = 9 + m) →
  (∀ (x y : ℝ), x^2/9 - y^2/m = 1 → (y = (4/3)*x ∨ y = -(4/3)*x)) :=
by sorry

end hyperbola_asymptotes_l1188_118869


namespace meet_time_theorem_l1188_118829

def round_time_P : ℕ := 252
def round_time_Q : ℕ := 198
def round_time_R : ℕ := 315

theorem meet_time_theorem : 
  Nat.lcm (Nat.lcm round_time_P round_time_Q) round_time_R = 13860 := by
  sorry

end meet_time_theorem_l1188_118829


namespace driveway_wheel_count_inconsistent_l1188_118852

/-- Represents the number of wheels on various vehicles and items in Jordan's driveway --/
structure DrivewayCounts where
  carCount : ℕ
  bikeCount : ℕ
  trashCanCount : ℕ
  tricycleCount : ℕ
  rollerSkatesPairCount : ℕ

/-- Calculates the total number of wheels based on the counts of vehicles and items --/
def totalWheels (counts : DrivewayCounts) : ℕ :=
  4 * counts.carCount +
  2 * counts.bikeCount +
  2 * counts.trashCanCount +
  3 * counts.tricycleCount +
  4 * counts.rollerSkatesPairCount

/-- Theorem stating that given the conditions, it's impossible to have 25 wheels in total --/
theorem driveway_wheel_count_inconsistent :
  ∀ (counts : DrivewayCounts),
    counts.carCount = 2 ∧
    counts.bikeCount = 2 ∧
    counts.trashCanCount = 1 ∧
    counts.tricycleCount = 1 ∧
    counts.rollerSkatesPairCount = 1 →
    totalWheels counts ≠ 25 :=
by
  sorry

end driveway_wheel_count_inconsistent_l1188_118852


namespace sphere_volume_l1188_118883

theorem sphere_volume (surface_area : Real) (volume : Real) : 
  surface_area = 100 * Real.pi → volume = (500 / 3) * Real.pi := by
  sorry

end sphere_volume_l1188_118883


namespace factors_of_28350_l1188_118864

/-- The number of positive factors of 28350 -/
def num_factors_28350 : ℕ := sorry

/-- 28350 is the number we are analyzing -/
def n : ℕ := 28350

theorem factors_of_28350 : num_factors_28350 = 48 := by sorry

end factors_of_28350_l1188_118864


namespace work_distribution_l1188_118880

theorem work_distribution (p : ℕ) (x : ℚ) (h1 : 0 < p) (h2 : 0 ≤ x) (h3 : x < 1) :
  p * 1 = (1 - x) * p * (3/2) → x = 1/3 := by
  sorry

end work_distribution_l1188_118880


namespace shopping_money_calculation_l1188_118861

theorem shopping_money_calculation (remaining_money : ℝ) (spent_percentage : ℝ) 
  (h1 : remaining_money = 224)
  (h2 : spent_percentage = 0.3)
  (h3 : remaining_money = (1 - spent_percentage) * original_amount) :
  original_amount = 320 :=
by
  sorry

end shopping_money_calculation_l1188_118861


namespace sample_size_calculation_l1188_118837

/-- Represents the sample size calculation for three communities --/
theorem sample_size_calculation 
  (pop_A pop_B pop_C : ℕ) 
  (sample_C : ℕ) 
  (h1 : pop_A = 600) 
  (h2 : pop_B = 1200) 
  (h3 : pop_C = 1500) 
  (h4 : sample_C = 15) : 
  ∃ n : ℕ, n * pop_C = sample_C * (pop_A + pop_B + pop_C) ∧ n = 33 :=
sorry

end sample_size_calculation_l1188_118837


namespace power_zero_l1188_118824

theorem power_zero (x : ℝ) (h : x ≠ 0) : x^0 = 1 := by
  sorry

end power_zero_l1188_118824


namespace largest_five_digit_sum_20_l1188_118818

def digits (n : ℕ) : List ℕ :=
  if n < 10 then [n] else (n % 10) :: digits (n / 10)

def digit_sum (n : ℕ) : ℕ :=
  (digits n).sum

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

theorem largest_five_digit_sum_20 :
  ∀ n : ℕ, is_five_digit n → digit_sum n = 20 → n ≤ 99200 :=
sorry

end largest_five_digit_sum_20_l1188_118818


namespace set_operations_l1188_118886

-- Define the sets A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 4 < x ∧ x < 10}

-- Define the intervals for the results
def interval_3_10 : Set ℝ := {x | 3 ≤ x ∧ x < 10}
def open_4_7 : Set ℝ := {x | 4 < x ∧ x < 7}
def union_4_7_7_10 : Set ℝ := {x | (4 < x ∧ x < 7) ∨ (7 ≤ x ∧ x < 10)}

-- State the theorem
theorem set_operations :
  (A ∪ B = interval_3_10) ∧
  (A ∩ B = open_4_7) ∧
  ((Set.univ \ A) ∩ B = union_4_7_7_10) := by sorry

end set_operations_l1188_118886


namespace problem_solution_l1188_118877

theorem problem_solution : ∃ m : ℚ, 15 + m * (25/3) = 6 * (25/3) - 10 ∧ m = 3 := by
  sorry

end problem_solution_l1188_118877


namespace trapezoid_segment_length_l1188_118840

-- Define a trapezoid PQRS
structure Trapezoid :=
  (P Q R S : ℝ × ℝ)

-- Define the length function
def length (A B : ℝ × ℝ) : ℝ := sorry

-- Define the area of a triangle
def area_triangle (A B C : ℝ × ℝ) : ℝ := sorry

theorem trapezoid_segment_length (PQRS : Trapezoid) :
  length PQRS.P PQRS.S + length PQRS.R PQRS.Q = 270 →
  area_triangle PQRS.P PQRS.Q PQRS.R / area_triangle PQRS.P PQRS.S PQRS.R = 5 / 4 →
  length PQRS.P PQRS.S = 150 := by
  sorry

end trapezoid_segment_length_l1188_118840


namespace remaining_area_ratio_l1188_118816

/-- The ratio of remaining areas of two squares after cutting out smaller squares -/
theorem remaining_area_ratio (side_c side_d cut_side : ℕ) 
  (hc : side_c = 48) 
  (hd : side_d = 60) 
  (hcut : cut_side = 12) : 
  (side_c^2 - cut_side^2) / (side_d^2 - cut_side^2) = 5/8 := by
  sorry

end remaining_area_ratio_l1188_118816


namespace x_eq_4_is_linear_l1188_118857

/-- A linear equation with one variable is of the form ax + b = 0, where a ≠ 0 and x is the variable. -/
def is_linear_equation_one_var (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The function f(x) = x - 4 represents the equation x = 4. -/
def f (x : ℝ) : ℝ := x - 4

theorem x_eq_4_is_linear :
  is_linear_equation_one_var f :=
sorry

end x_eq_4_is_linear_l1188_118857


namespace inverse_exponential_function_l1188_118827

noncomputable def f (a : ℝ) : ℝ → ℝ := fun x => Real.log x / Real.log a

theorem inverse_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f := f a
  (∀ x, f (a^x) = x) ∧ (∀ y, a^(f y) = y) ∧ f 2 = 1 → f = fun x => Real.log x / Real.log 2 := by
  sorry

end inverse_exponential_function_l1188_118827


namespace village_population_problem_l1188_118828

theorem village_population_problem (final_population : ℕ) : 
  final_population = 5265 → ∃ original : ℕ, 
    (original : ℚ) * (9/10) * (3/4) = final_population ∧ original = 7800 :=
by
  sorry

end village_population_problem_l1188_118828


namespace f_satisfies_conditions_l1188_118860

/-- A quadratic function with specific properties -/
def f (x : ℝ) : ℝ := -6*x^2 + 36*x - 30

/-- Theorem stating that f satisfies the required conditions -/
theorem f_satisfies_conditions :
  f 1 = 0 ∧ f 5 = 0 ∧ f 3 = 24 := by
  sorry

end f_satisfies_conditions_l1188_118860


namespace min_value_of_x_l1188_118846

theorem min_value_of_x (x : ℝ) (h1 : x > 0) (h2 : Real.log x / Real.log 3 ≥ Real.log 9 / Real.log 3 + (1/3) * (Real.log x / Real.log 3)) :
  x ≥ 27 ∧ ∀ y : ℝ, y > 0 → Real.log y / Real.log 3 ≥ Real.log 9 / Real.log 3 + (1/3) * (Real.log y / Real.log 3) → y ≥ x → y ≥ 27 :=
sorry

end min_value_of_x_l1188_118846


namespace cafe_choices_l1188_118885

/-- The number of ways two people can choose different items from a set of n items -/
def differentChoices (n : ℕ) : ℕ := n * (n - 1)

/-- The number of menu items in the café -/
def menuItems : ℕ := 12

/-- Theorem: The number of ways Alex and Jamie can choose different dishes from a menu of 12 items is 132 -/
theorem cafe_choices : differentChoices menuItems = 132 := by
  sorry

end cafe_choices_l1188_118885


namespace melanies_turnips_l1188_118863

/-- The number of turnips Benny grew -/
def bennys_turnips : ℕ := 113

/-- The total number of turnips grown by Melanie and Benny -/
def total_turnips : ℕ := 252

/-- Melanie's turnips are equal to the total minus Benny's -/
theorem melanies_turnips : ℕ := total_turnips - bennys_turnips

#check melanies_turnips

end melanies_turnips_l1188_118863


namespace solution_set_abs_b_greater_than_two_l1188_118853

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2|

-- Part 1: Solution set of the inequality
theorem solution_set (x : ℝ) : f x + f (x + 1) ≥ 5 ↔ x ≥ 4 ∨ x ≤ -1 := by
  sorry

-- Part 2: Proof that |b| > 2
theorem abs_b_greater_than_two (a b : ℝ) (h1 : |a| > 1) (h2 : f (a * b) > |a| * f (b / a)) : |b| > 2 := by
  sorry

end solution_set_abs_b_greater_than_two_l1188_118853


namespace ratio_a_to_c_l1188_118809

theorem ratio_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 2)
  (hcd : c / d = 4 / 1)
  (hdb : d / b = 1 / 3) :
  a / c = 15 / 8 := by
  sorry

end ratio_a_to_c_l1188_118809


namespace odd_function_interval_l1188_118878

/-- A function f is odd on an interval [a, b] if the interval is symmetric about the origin -/
def is_odd_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  a = -b ∧ ∀ x ∈ Set.Icc a b, f (-x) = -f x

/-- The theorem stating that if f is an odd function on [t, t^2 - 3t - 3], then t = -1 -/
theorem odd_function_interval (f : ℝ → ℝ) (t : ℝ) :
  is_odd_on_interval f t (t^2 - 3*t - 3) → t = -1 := by
  sorry


end odd_function_interval_l1188_118878


namespace soda_cans_calculation_correct_l1188_118800

/-- Given that S cans of soda can be purchased for Q dimes, and 1 dollar is worth 10 dimes,
    this function calculates the number of cans that can be purchased for D dollars. -/
def soda_cans_for_dollars (S Q D : ℚ) : ℚ :=
  10 * D * S / Q

/-- Theorem stating that the number of cans that can be purchased for D dollars
    is correctly calculated by the soda_cans_for_dollars function. -/
theorem soda_cans_calculation_correct (S Q D : ℚ) (hS : S > 0) (hQ : Q > 0) (hD : D ≥ 0) :
  soda_cans_for_dollars S Q D = 10 * D * S / Q :=
by sorry

end soda_cans_calculation_correct_l1188_118800


namespace triangle_side_length_l1188_118835

-- Define the triangle XYZ
structure Triangle where
  X : Real
  Y : Real
  Z : Real
  x : Real
  y : Real
  z : Real

-- State the theorem
theorem triangle_side_length (t : Triangle) 
  (h1 : t.y = 7)
  (h2 : t.z = 6)
  (h3 : Real.cos (t.Y - t.Z) = 17/18) :
  t.x = Real.sqrt 65 := by
sorry

end triangle_side_length_l1188_118835


namespace four_square_rectangle_exists_l1188_118889

/-- Represents a color --/
structure Color : Type

/-- Represents a square on the grid --/
structure Square : Type :=
  (x : ℤ)
  (y : ℤ)
  (color : Color)

/-- Represents an infinite grid of colored squares --/
def InfiniteGrid : Type := ℤ → ℤ → Color

/-- Checks if four squares form a rectangle parallel to grid lines --/
def IsRectangle (s1 s2 s3 s4 : Square) : Prop :=
  (s1.x = s2.x ∧ s3.x = s4.x ∧ s1.y = s3.y ∧ s2.y = s4.y) ∨
  (s1.x = s3.x ∧ s2.x = s4.x ∧ s1.y = s2.y ∧ s3.y = s4.y)

/-- Main theorem: There always exist four squares of the same color forming a rectangle --/
theorem four_square_rectangle_exists (n : ℕ) (h : n ≥ 2) (grid : InfiniteGrid) :
  ∃ (s1 s2 s3 s4 : Square),
    s1.color = s2.color ∧ s2.color = s3.color ∧ s3.color = s4.color ∧
    IsRectangle s1 s2 s3 s4 := by
  sorry

end four_square_rectangle_exists_l1188_118889


namespace opposite_of_2023_l1188_118820

theorem opposite_of_2023 : ∀ x : ℤ, x + 2023 = 0 ↔ x = -2023 := by
  sorry

end opposite_of_2023_l1188_118820


namespace smallest_common_multiple_10_15_gt_100_l1188_118884

theorem smallest_common_multiple_10_15_gt_100 : ∃ (n : ℕ), n > 100 ∧ n.lcm 10 = n ∧ n.lcm 15 = n ∧ ∀ (m : ℕ), m > 100 ∧ m.lcm 10 = m ∧ m.lcm 15 = m → n ≤ m :=
  sorry

end smallest_common_multiple_10_15_gt_100_l1188_118884


namespace cos_30_tan_45_equality_l1188_118819

theorem cos_30_tan_45_equality : 2 * Real.cos (30 * π / 180) - Real.tan (45 * π / 180) = Real.sqrt 3 - 1 := by
  sorry

end cos_30_tan_45_equality_l1188_118819


namespace defective_shipped_percentage_is_correct_l1188_118891

/-- Percentage of units with Type A defects in the first stage -/
def type_a_defect_rate : ℝ := 0.07

/-- Percentage of units with Type B defects in the second stage -/
def type_b_defect_rate : ℝ := 0.08

/-- Percentage of Type A defects that are reworked and repaired -/
def type_a_rework_rate : ℝ := 0.40

/-- Percentage of Type B defects that are reworked and repaired -/
def type_b_rework_rate : ℝ := 0.30

/-- Percentage of remaining Type A defects that are shipped -/
def type_a_ship_rate : ℝ := 0.03

/-- Percentage of remaining Type B defects that are shipped -/
def type_b_ship_rate : ℝ := 0.06

/-- The percentage of defective units (Type A or B) shipped for sale -/
def defective_shipped_percentage : ℝ :=
  type_a_defect_rate * (1 - type_a_rework_rate) * type_a_ship_rate +
  type_b_defect_rate * (1 - type_b_rework_rate) * type_b_ship_rate

theorem defective_shipped_percentage_is_correct :
  defective_shipped_percentage = 0.00462 := by
  sorry

end defective_shipped_percentage_is_correct_l1188_118891


namespace emilys_skirt_cost_l1188_118822

theorem emilys_skirt_cost (art_supplies_cost shoes_original_price total_spent : ℝ)
  (skirt_count : ℕ) (shoe_discount_rate : ℝ) :
  art_supplies_cost = 20 →
  skirt_count = 2 →
  shoes_original_price = 30 →
  shoe_discount_rate = 0.15 →
  total_spent = 50 →
  let shoes_discounted_price := shoes_original_price * (1 - shoe_discount_rate)
  let skirts_total_cost := total_spent - art_supplies_cost - shoes_discounted_price
  let skirt_cost := skirts_total_cost / skirt_count
  skirt_cost = 2.25 := by
  sorry

end emilys_skirt_cost_l1188_118822


namespace subtraction_proof_l1188_118875

theorem subtraction_proof : 25.705 - 3.289 = 22.416 := by
  sorry

end subtraction_proof_l1188_118875


namespace total_points_scored_l1188_118894

/-- Given a player who plays 13 games and scores 7 points in each game,
    the total number of points scored is equal to 91. -/
theorem total_points_scored (games : ℕ) (points_per_game : ℕ) : 
  games = 13 → points_per_game = 7 → games * points_per_game = 91 := by
  sorry

end total_points_scored_l1188_118894


namespace unique_solution_implies_a_greater_than_one_l1188_118849

-- Define the function f(x) = 2ax^2 - x - 1
def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * x^2 - x - 1

-- State the theorem
theorem unique_solution_implies_a_greater_than_one :
  ∀ a : ℝ, (∃! x : ℝ, x ∈ (Set.Ioo 0 1) ∧ f a x = 0) → a > 1 := by
  sorry

end unique_solution_implies_a_greater_than_one_l1188_118849


namespace exponent_division_l1188_118872

theorem exponent_division (a : ℝ) : a^7 / a^4 = a^3 := by
  sorry

end exponent_division_l1188_118872


namespace colonization_theorem_l1188_118848

def blue_planets : ℕ := 7
def orange_planets : ℕ := 8
def blue_cost : ℕ := 3
def orange_cost : ℕ := 2
def total_units : ℕ := 21

def colonization_ways (b o bc oc t : ℕ) : ℕ :=
  (Nat.choose b 7 * Nat.choose o 0) +
  (Nat.choose b 5 * Nat.choose o 3) +
  (Nat.choose b 3 * Nat.choose o 6)

theorem colonization_theorem :
  colonization_ways blue_planets orange_planets blue_cost orange_cost total_units = 2157 := by
  sorry

end colonization_theorem_l1188_118848


namespace intersection_implies_m_range_l1188_118867

/-- The set M defined by the equation 3x^2 + 4y^2 - 6mx + 3m^2 - 12 = 0 -/
def M (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 3 * p.1^2 + 4 * p.2^2 - 6 * m * p.1 + 3 * m^2 - 12 = 0}

/-- The set N defined by the equation 2y^2 - 12x + 9 = 0 -/
def N : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2 * p.2^2 - 12 * p.1 + 9 = 0}

/-- Theorem stating that if M and N have a non-empty intersection,
    then m is in the range [-5/4, 11/4] -/
theorem intersection_implies_m_range :
  ∀ m : ℝ, (M m ∩ N).Nonempty → -5/4 ≤ m ∧ m ≤ 11/4 := by
  sorry

end intersection_implies_m_range_l1188_118867


namespace systematic_sample_fourth_element_l1188_118808

/-- Represents a systematic sample from a population -/
structure SystematicSample where
  population_size : ℕ
  sample_size : ℕ
  sample : Finset ℕ
  h_sample_size : sample.card = sample_size
  h_valid_sample : ∀ n ∈ sample, n ≤ population_size

/-- Checks if a given set of numbers forms an arithmetic sequence -/
def is_arithmetic_sequence (s : Finset ℕ) : Prop :=
  ∃ a d : ℤ, ∀ n ∈ s, ∃ k : ℕ, (n : ℤ) = a + k * d

theorem systematic_sample_fourth_element
  (s : SystematicSample)
  (h_pop : s.population_size = 52)
  (h_sample : s.sample_size = 4)
  (h_elements : {5, 31, 44} ⊆ s.sample)
  (h_arithmetic : is_arithmetic_sequence s.sample) :
  18 ∈ s.sample := by
  sorry


end systematic_sample_fourth_element_l1188_118808


namespace trailing_zeros_of_product_trailing_zeros_of_product_is_90_l1188_118823

/-- The number of trailing zeros in the product of 20^50 and 50^20 -/
theorem trailing_zeros_of_product : ℕ :=
  let a := 20^50
  let b := 50^20
  let product := a * b
  90

/-- Proof that the number of trailing zeros in the product of 20^50 and 50^20 is 90 -/
theorem trailing_zeros_of_product_is_90 :
  trailing_zeros_of_product = 90 := by sorry

end trailing_zeros_of_product_trailing_zeros_of_product_is_90_l1188_118823


namespace tau_phi_equality_characterization_l1188_118865

/-- Number of natural numbers dividing n -/
def tau (n : ℕ) : ℕ := sorry

/-- Number of natural numbers less than n that are relatively prime to n -/
def phi (n : ℕ) : ℕ := sorry

/-- Predicate for n having exactly two different prime divisors -/
def has_two_prime_divisors (n : ℕ) : Prop := sorry

/-- Main theorem -/
theorem tau_phi_equality_characterization (n : ℕ) :
  has_two_prime_divisors n ∧ tau (phi n) = phi (tau n) ↔
  ∃ k : ℕ, n = 3 * 2^(2^k - 1) :=
sorry

end tau_phi_equality_characterization_l1188_118865


namespace complement_of_union_l1188_118879

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {2, 3, 4}
def N : Set Nat := {4, 5}

theorem complement_of_union :
  (U \ (M ∪ N)) = {1, 6} := by sorry

end complement_of_union_l1188_118879


namespace at_least_three_correct_guesses_l1188_118832

-- Define the type for colors
inductive Color
| Red | Orange | Yellow | Green | Blue | Indigo | Violet

-- Define the type for dwarves
structure Dwarf where
  id : Fin 6
  seenHats : Finset Color

-- Define the game setup
structure GameSetup where
  allHats : Finset Color
  hiddenHat : Color
  dwarves : Fin 6 → Dwarf

-- Define the guessing strategy
def guessNearestClockwise (d : Dwarf) (allColors : Finset Color) : Color :=
  sorry

-- Theorem statement
theorem at_least_three_correct_guesses 
  (setup : GameSetup)
  (h1 : setup.allHats.card = 7)
  (h2 : ∀ d : Fin 6, (setup.dwarves d).seenHats.card = 5)
  (h3 : ∀ d : Fin 6, (setup.dwarves d).seenHats ⊆ setup.allHats)
  (h4 : setup.hiddenHat ∈ setup.allHats) :
  ∃ (correctGuesses : Finset (Fin 6)), 
    correctGuesses.card ≥ 3 ∧ 
    ∀ d ∈ correctGuesses, guessNearestClockwise (setup.dwarves d) setup.allHats = setup.hiddenHat :=
sorry

end at_least_three_correct_guesses_l1188_118832
