import Mathlib

namespace NUMINAMATH_CALUDE_unplanted_field_fraction_l3947_394739

theorem unplanted_field_fraction (a b c x : ℝ) : 
  a = 5 → b = 12 → c^2 = a^2 + b^2 → x = 5/3 → 
  x^2 / (a * b / 2) = 5/54 := by sorry

end NUMINAMATH_CALUDE_unplanted_field_fraction_l3947_394739


namespace NUMINAMATH_CALUDE_total_carrots_grown_l3947_394758

theorem total_carrots_grown (sandy_carrots sam_carrots : ℕ) 
  (h1 : sandy_carrots = 6) 
  (h2 : sam_carrots = 3) : 
  sandy_carrots + sam_carrots = 9 := by
  sorry

end NUMINAMATH_CALUDE_total_carrots_grown_l3947_394758


namespace NUMINAMATH_CALUDE_square_division_has_triangle_l3947_394724

/-- A convex polygon within a square --/
structure PolygonInSquare where
  sides : ℕ
  convex : Bool
  inSquare : Bool

/-- Represents a division of a square into polygons --/
def SquareDivision := List PolygonInSquare

/-- Checks if all polygons in the division are convex and within the square --/
def isValidDivision (d : SquareDivision) : Prop :=
  d.all (λ p => p.convex ∧ p.inSquare)

/-- Checks if all polygons have distinct number of sides --/
def hasDistinctSides (d : SquareDivision) : Prop :=
  d.map (λ p => p.sides) |>.Nodup

/-- Checks if there's a triangle in the division --/
def hasTriangle (d : SquareDivision) : Prop :=
  d.any (λ p => p.sides = 3)

theorem square_division_has_triangle (d : SquareDivision) :
  d.length > 1 → isValidDivision d → hasDistinctSides d → hasTriangle d := by
  sorry

end NUMINAMATH_CALUDE_square_division_has_triangle_l3947_394724


namespace NUMINAMATH_CALUDE_max_y_value_l3947_394749

theorem max_y_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^3 + y^3 = (4*x - 5*y)*y) : 
  y ≤ 1/3 :=
sorry

end NUMINAMATH_CALUDE_max_y_value_l3947_394749


namespace NUMINAMATH_CALUDE_roots_not_in_interval_l3947_394712

theorem roots_not_in_interval (a : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  ∀ x : ℝ, (a^x + a^(-x) = 2*a) → x ∉ Set.Icc (-1 : ℝ) 1 := by
  sorry

end NUMINAMATH_CALUDE_roots_not_in_interval_l3947_394712


namespace NUMINAMATH_CALUDE_triangle_n_range_l3947_394756

-- Define a triangle with the given properties
structure Triangle where
  n : ℝ
  angle1 : ℝ := 180 - n
  angle2 : ℝ
  angle3 : ℝ
  sum_of_angles : angle1 + angle2 + angle3 = 180
  angle_difference : max angle1 (max angle2 angle3) - min angle1 (min angle2 angle3) = 24

-- Theorem statement
theorem triangle_n_range (t : Triangle) : 104 ≤ t.n ∧ t.n ≤ 136 := by
  sorry

end NUMINAMATH_CALUDE_triangle_n_range_l3947_394756


namespace NUMINAMATH_CALUDE_fraction_meaningful_l3947_394737

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 1)) ↔ x ≠ 1 := by
sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l3947_394737


namespace NUMINAMATH_CALUDE_solutions_for_20_l3947_394729

/-- The number of distinct integer solutions (x,y) for |x| + |y| = n -/
def num_solutions (n : ℕ) : ℕ := 4 * n

theorem solutions_for_20 : num_solutions 20 = 80 := by
  sorry

end NUMINAMATH_CALUDE_solutions_for_20_l3947_394729


namespace NUMINAMATH_CALUDE_sum_of_all_alternating_sums_l3947_394765

-- Define the set of numbers
def S : Finset ℕ := Finset.range 9

-- Define the alternating sum function
noncomputable def alternatingSum (subset : Finset ℕ) : ℤ :=
  sorry

-- Define the modified alternating sum that adds 9 again if present
noncomputable def modifiedAlternatingSum (subset : Finset ℕ) : ℤ :=
  if 9 ∈ subset then alternatingSum subset + 9 else alternatingSum subset

-- Theorem statement
theorem sum_of_all_alternating_sums : 
  (Finset.powerset S).sum modifiedAlternatingSum = 2304 :=
sorry

end NUMINAMATH_CALUDE_sum_of_all_alternating_sums_l3947_394765


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3947_394731

theorem absolute_value_inequality (x : ℝ) (h : x ≠ 2) :
  |((3 * x - 2) / (x - 2))| > 3 ↔ x ∈ Set.Ioo (4/3) 2 ∪ Set.Ioi 2 :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3947_394731


namespace NUMINAMATH_CALUDE_arithmetic_mean_three_digit_multiples_of_seven_l3947_394754

theorem arithmetic_mean_three_digit_multiples_of_seven :
  let first : ℕ := 105
  let last : ℕ := 994
  let count : ℕ := 128
  let sum : ℕ := count * (first + last) / 2
  (sum : ℚ) / count = 549.5 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_three_digit_multiples_of_seven_l3947_394754


namespace NUMINAMATH_CALUDE_class_size_from_mark_change_l3947_394708

/-- Given a class where one pupil's mark was increased by 20 points, 
    causing the class average to rise by 1/2, prove that there are 40 pupils in the class. -/
theorem class_size_from_mark_change (mark_increase : ℕ) (average_increase : ℚ) : 
  mark_increase = 20 → average_increase = 1/2 → (mark_increase : ℚ) / average_increase = 40 := by
  sorry

end NUMINAMATH_CALUDE_class_size_from_mark_change_l3947_394708


namespace NUMINAMATH_CALUDE_cubic_sum_minus_product_l3947_394705

theorem cubic_sum_minus_product (a b c : ℝ) 
  (sum_eq : a + b + c = 11) 
  (sum_products_eq : a * b + a * c + b * c = 25) : 
  a^3 + b^3 + c^3 - 3*a*b*c = 506 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_minus_product_l3947_394705


namespace NUMINAMATH_CALUDE_stadium_width_l3947_394747

theorem stadium_width (length height diagonal : ℝ) 
  (h_length : length = 24)
  (h_height : height = 16)
  (h_diagonal : diagonal = 34) :
  ∃ width : ℝ, width = 18 ∧ diagonal^2 = length^2 + width^2 + height^2 := by
sorry

end NUMINAMATH_CALUDE_stadium_width_l3947_394747


namespace NUMINAMATH_CALUDE_binomial_and_power_of_two_l3947_394751

theorem binomial_and_power_of_two : Nat.choose 8 3 = 56 ∧ 2^(Nat.choose 8 3) = 2^56 := by
  sorry

end NUMINAMATH_CALUDE_binomial_and_power_of_two_l3947_394751


namespace NUMINAMATH_CALUDE_greatest_integer_quadratic_inequality_l3947_394722

theorem greatest_integer_quadratic_inequality :
  ∃ (n : ℤ), n^2 - 17*n + 72 ≤ 0 ∧ n = 9 ∧ ∀ (m : ℤ), m^2 - 17*m + 72 ≤ 0 → m ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_quadratic_inequality_l3947_394722


namespace NUMINAMATH_CALUDE_power_division_subtraction_addition_l3947_394732

theorem power_division_subtraction_addition : (-6)^4 / 6^2 - 2^5 + 4^2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_power_division_subtraction_addition_l3947_394732


namespace NUMINAMATH_CALUDE_solve_equation_l3947_394795

theorem solve_equation : ∃ x : ℝ, 7 * (x - 1) = 21 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3947_394795


namespace NUMINAMATH_CALUDE_curve_equation_l3947_394772

/-- Given vectors and their relationships, prove the equation of the resulting curve. -/
theorem curve_equation (x y : ℝ) : 
  let m₁ : ℝ × ℝ := (0, x)
  let n₁ : ℝ × ℝ := (1, 1)
  let m₂ : ℝ × ℝ := (x, 0)
  let n₂ : ℝ × ℝ := (y^2, 1)
  let m : ℝ × ℝ := m₁ + Real.sqrt 2 • n₂
  let n : ℝ × ℝ := m₂ - Real.sqrt 2 • n₁
  (m.1 * n.2 = m.2 * n.1) →  -- m is parallel to n
  x^2 / 2 + y^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_curve_equation_l3947_394772


namespace NUMINAMATH_CALUDE_min_seats_occupied_min_occupied_seats_is_fifty_l3947_394789

/-- Represents the number of seats in a row -/
def total_seats : Nat := 200

/-- Represents the size of each group (one person + three empty seats) -/
def group_size : Nat := 4

/-- The minimum number of occupied seats required -/
def min_occupied_seats : Nat := total_seats / group_size

/-- Theorem stating the minimum number of occupied seats -/
theorem min_seats_occupied (n : Nat) : 
  n ≥ min_occupied_seats → 
  ∀ (new_seat : Nat), new_seat > n ∧ new_seat ≤ total_seats → 
  ∃ (occupied_seat : Nat), occupied_seat ≤ n ∧ (new_seat = occupied_seat + 1 ∨ new_seat = occupied_seat - 1) :=
sorry

/-- Theorem proving the minimum number of occupied seats is indeed 50 -/
theorem min_occupied_seats_is_fifty : min_occupied_seats = 50 :=
sorry

end NUMINAMATH_CALUDE_min_seats_occupied_min_occupied_seats_is_fifty_l3947_394789


namespace NUMINAMATH_CALUDE_factor_expression_l3947_394738

theorem factor_expression (x : ℝ) : 16 * x^4 - 4 * x^2 = 4 * x^2 * (2*x + 1) * (2*x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3947_394738


namespace NUMINAMATH_CALUDE_received_a_implies_met_criteria_l3947_394707

/-- Represents the criteria for receiving an A on the exam -/
structure ExamCriteria where
  multiple_choice_correct : ℝ
  extra_credit_completed : Bool

/-- Represents a student's exam performance -/
structure ExamPerformance where
  multiple_choice_correct : ℝ
  extra_credit_completed : Bool
  received_a : Bool

/-- The criteria for receiving an A on the exam -/
def a_criteria : ExamCriteria :=
  { multiple_choice_correct := 90
  , extra_credit_completed := true }

/-- Predicate to check if a student's performance meets the criteria for an A -/
def meets_a_criteria (performance : ExamPerformance) (criteria : ExamCriteria) : Prop :=
  performance.multiple_choice_correct ≥ criteria.multiple_choice_correct ∧
  performance.extra_credit_completed = criteria.extra_credit_completed

/-- Theorem stating that if a student received an A, they must have met the criteria -/
theorem received_a_implies_met_criteria (student : ExamPerformance) :
  student.received_a → meets_a_criteria student a_criteria := by
  sorry

end NUMINAMATH_CALUDE_received_a_implies_met_criteria_l3947_394707


namespace NUMINAMATH_CALUDE_solution_satisfies_equation_l3947_394706

-- Define the logarithm function with base 5
noncomputable def log5 (x : ℝ) : ℝ := Real.log x / Real.log 5

-- Define the equation
def equation (x : ℝ) : Prop :=
  2 * log5 x - 3 * log5 4 = 1

-- Theorem statement
theorem solution_satisfies_equation :
  equation (4 * Real.sqrt 5) ∧ equation (-4 * Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_solution_satisfies_equation_l3947_394706


namespace NUMINAMATH_CALUDE_ship_age_conversion_l3947_394718

/-- Converts an octal number represented as (a, b, c) to its decimal equivalent -/
def octal_to_decimal (a b c : ℕ) : ℕ := c * 8^2 + b * 8^1 + a * 8^0

/-- The age of the sunken pirate ship in octal -/
def ship_age_octal : ℕ × ℕ × ℕ := (7, 4, 2)

theorem ship_age_conversion :
  octal_to_decimal ship_age_octal.1 ship_age_octal.2.1 ship_age_octal.2.2 = 482 := by
  sorry

end NUMINAMATH_CALUDE_ship_age_conversion_l3947_394718


namespace NUMINAMATH_CALUDE_overlap_area_is_zero_l3947_394755

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

/-- Calculate the area of overlap between two triangles -/
def areaOfOverlap (t1 t2 : Triangle) : ℝ :=
  sorry

/-- The main theorem stating that the area of overlap is zero -/
theorem overlap_area_is_zero :
  let t1 := Triangle.mk (Point.mk 0 0) (Point.mk 2 2) (Point.mk 2 0)
  let t2 := Triangle.mk (Point.mk 0 2) (Point.mk 2 2) (Point.mk 0 0)
  areaOfOverlap t1 t2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_overlap_area_is_zero_l3947_394755


namespace NUMINAMATH_CALUDE_greatest_x_value_l3947_394745

theorem greatest_x_value (x : ℝ) : 
  x ≠ 9 → 
  (x^2 - x - 90) / (x - 9) = 2 / (x + 7) → 
  x ≤ -4 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_value_l3947_394745


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l3947_394796

/-- Given a hyperbola with equation x²/16 - y²/25 = 1, 
    its asymptotes have the equation y = ±(5/4)x -/
theorem hyperbola_asymptotes (x y : ℝ) :
  x^2 / 16 - y^2 / 25 = 1 →
  ∃ (k : ℝ), k = 5/4 ∧ (y = k*x ∨ y = -k*x) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l3947_394796


namespace NUMINAMATH_CALUDE_winning_candidate_percentage_l3947_394762

def candidate1_votes : ℕ := 6136
def candidate2_votes : ℕ := 7636
def candidate3_votes : ℕ := 11628

def total_votes : ℕ := candidate1_votes + candidate2_votes + candidate3_votes

def winning_votes : ℕ := max candidate1_votes (max candidate2_votes candidate3_votes)

def winning_percentage : ℚ := (winning_votes : ℚ) / (total_votes : ℚ) * 100

theorem winning_candidate_percentage :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ |winning_percentage - 45.78| < ε :=
sorry

end NUMINAMATH_CALUDE_winning_candidate_percentage_l3947_394762


namespace NUMINAMATH_CALUDE_solution_of_exponential_equation_l3947_394728

theorem solution_of_exponential_equation :
  ∃ x : ℝ, (2 : ℝ) ^ x = 8 ∧ x = 3 := by sorry

end NUMINAMATH_CALUDE_solution_of_exponential_equation_l3947_394728


namespace NUMINAMATH_CALUDE_sets_properties_l3947_394735

open Set

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 3*x - 4 ≥ 0}
def B : Set ℝ := {x | x - 5 < 0}

-- State the theorem
theorem sets_properties :
  (A ∩ B = Icc 4 5) ∧
  (A ∪ B = univ) ∧
  (Aᶜ = Ioo (-1) 4) := by sorry

end NUMINAMATH_CALUDE_sets_properties_l3947_394735


namespace NUMINAMATH_CALUDE_income_2005_between_3600_and_3800_l3947_394723

/-- Represents the income data for farmers in a certain region -/
structure FarmerIncome where
  initialYear : Nat
  initialWageIncome : ℝ
  initialOtherIncome : ℝ
  wageGrowthRate : ℝ
  otherIncomeIncrease : ℝ

/-- Calculates the average income of farmers after a given number of years -/
def averageIncomeAfterYears (data : FarmerIncome) (years : Nat) : ℝ :=
  data.initialWageIncome * (1 + data.wageGrowthRate) ^ years +
  data.initialOtherIncome + data.otherIncomeIncrease * years

/-- Theorem stating that the average income in 2005 will be between 3600 and 3800 yuan -/
theorem income_2005_between_3600_and_3800 (data : FarmerIncome) 
  (h1 : data.initialYear = 2003)
  (h2 : data.initialWageIncome = 1800)
  (h3 : data.initialOtherIncome = 1350)
  (h4 : data.wageGrowthRate = 0.06)
  (h5 : data.otherIncomeIncrease = 160) :
  3600 ≤ averageIncomeAfterYears data 2 ∧ averageIncomeAfterYears data 2 ≤ 3800 := by
  sorry

#eval averageIncomeAfterYears 
  { initialYear := 2003
    initialWageIncome := 1800
    initialOtherIncome := 1350
    wageGrowthRate := 0.06
    otherIncomeIncrease := 160 } 2

end NUMINAMATH_CALUDE_income_2005_between_3600_and_3800_l3947_394723


namespace NUMINAMATH_CALUDE_hidden_number_l3947_394703

theorem hidden_number (x : ℝ) (hidden : ℝ) : 
  x = -1 → (2 + hidden * x) / 3 = -1 → hidden = 5 := by
  sorry

end NUMINAMATH_CALUDE_hidden_number_l3947_394703


namespace NUMINAMATH_CALUDE_quadratic_symmetric_derivative_l3947_394781

-- Define a quadratic function symmetric about x = 1
def f (a b : ℝ) (x : ℝ) : ℝ := a * (x - 1)^2 + b

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 2 * a * (x - 1)

theorem quadratic_symmetric_derivative (a b : ℝ) :
  (f' a 0 = -2) → (f' a 2 = 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_symmetric_derivative_l3947_394781


namespace NUMINAMATH_CALUDE_total_trees_is_fifteen_l3947_394760

/-- The number of apple trees Ava planted -/
def ava_trees : ℕ := 9

/-- The difference between Ava's and Lily's trees -/
def difference : ℕ := 3

/-- The number of apple trees Lily planted -/
def lily_trees : ℕ := ava_trees - difference

/-- The total number of apple trees planted by Ava and Lily -/
def total_trees : ℕ := ava_trees + lily_trees

theorem total_trees_is_fifteen : total_trees = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_trees_is_fifteen_l3947_394760


namespace NUMINAMATH_CALUDE_ellipse_trajectory_and_minimum_l3947_394727

-- Define the ellipse
def ellipse (x y : ℝ) : Prop :=
  x^2 + y^2/4 = 1 ∧ x > 0 ∧ y > 0

-- Define the tangent line
def tangent_line (x₀ y₀ x y : ℝ) : Prop :=
  y = -4*x₀/y₀ * (x - x₀) + y₀

-- Define point M
def point_M (x y : ℝ) : Prop :=
  ∃ x₀ y₀, ellipse x₀ y₀ ∧
  ∃ xA yB, tangent_line x₀ y₀ xA 0 ∧ tangent_line x₀ y₀ 0 yB ∧
  x = xA ∧ y = yB

theorem ellipse_trajectory_and_minimum (x y : ℝ) :
  point_M x y →
  (1/x^2 + 4/y^2 = 1 ∧ x > 1 ∧ y > 2) ∧
  (∀ x' y', point_M x' y' → x'^2 + y'^2 ≥ 9) ∧
  (∃ x₀ y₀, point_M x₀ y₀ ∧ x₀^2 + y₀^2 = 9) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_trajectory_and_minimum_l3947_394727


namespace NUMINAMATH_CALUDE_system_solution_l3947_394717

theorem system_solution (x y k : ℝ) : 
  (2 * x + y = 1) → 
  (x + 2 * y = k - 2) → 
  (x - y = 2) → 
  (k = 1) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3947_394717


namespace NUMINAMATH_CALUDE_sqrt_175_range_l3947_394761

theorem sqrt_175_range : 13 < Real.sqrt 175 ∧ Real.sqrt 175 < 14 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_175_range_l3947_394761


namespace NUMINAMATH_CALUDE_max_teams_tied_for_most_wins_l3947_394798

/-- Represents a round-robin tournament --/
structure Tournament where
  n : ℕ  -- number of teams
  games : Fin n → Fin n → Bool
  -- games i j is true if team i wins against team j
  irreflexive : ∀ i, games i i = false
  asymmetric : ∀ i j, games i j = !games j i

/-- The number of wins for a team in a tournament --/
def wins (t : Tournament) (i : Fin t.n) : ℕ :=
  (Finset.univ.filter (λ j => t.games i j)).card

/-- The maximum number of wins in a tournament --/
def max_wins (t : Tournament) : ℕ :=
  Finset.univ.sup (wins t)

/-- The number of teams tied for the maximum number of wins --/
def num_teams_with_max_wins (t : Tournament) : ℕ :=
  (Finset.univ.filter (λ i => wins t i = max_wins t)).card

theorem max_teams_tied_for_most_wins :
  ∃ t : Tournament, t.n = 8 ∧ num_teams_with_max_wins t = 7 ∧
  ∀ t' : Tournament, t'.n = 8 → num_teams_with_max_wins t' ≤ 7 := by
  sorry

end NUMINAMATH_CALUDE_max_teams_tied_for_most_wins_l3947_394798


namespace NUMINAMATH_CALUDE_committee_formation_count_l3947_394790

theorem committee_formation_count : Nat.choose 15 6 = 5005 := by
  sorry

end NUMINAMATH_CALUDE_committee_formation_count_l3947_394790


namespace NUMINAMATH_CALUDE_multiples_of_five_up_to_hundred_l3947_394744

theorem multiples_of_five_up_to_hundred :
  ∃ n : ℕ, n = 100 ∧ (∃! k : ℕ, k = 20 ∧ (∀ m : ℕ, 1 ≤ m ∧ m ≤ n → (m % 5 = 0 ↔ m ∈ Finset.range k))) :=
by sorry

end NUMINAMATH_CALUDE_multiples_of_five_up_to_hundred_l3947_394744


namespace NUMINAMATH_CALUDE_inscribed_box_radius_l3947_394726

/-- A rectangular box inscribed in a sphere -/
structure InscribedBox where
  r : ℝ  -- radius of the sphere
  x : ℝ  -- width of the box
  y : ℝ  -- length of the box
  z : ℝ  -- height of the box

/-- Properties of the inscribed box -/
def InscribedBoxProperties (box : InscribedBox) : Prop :=
  box.x > 0 ∧ box.y > 0 ∧ box.z > 0 ∧  -- dimensions are positive
  box.z = 3 * box.x ∧  -- ratio between height and width is 1:3
  4 * (box.x + box.y + box.z) = 72 ∧  -- sum of edge lengths
  2 * (box.x * box.y + box.y * box.z + box.x * box.z) = 162 ∧  -- surface area
  4 * box.r^2 = box.x^2 + box.y^2 + box.z^2  -- inscribed in sphere

theorem inscribed_box_radius (box : InscribedBox) 
  (h : InscribedBoxProperties box) : box.r = 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_box_radius_l3947_394726


namespace NUMINAMATH_CALUDE_inequality_selection_l3947_394778

/-- Given positive real numbers a, b, c, and a function f with minimum value 4,
    prove that a + b + c = 4 and find the minimum value of (1/4)a² + (1/9)b² + c² -/
theorem inequality_selection (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : ∀ x, |x + a| + |x - b| + c ≥ 4)
  (h5 : ∃ x, |x + a| + |x - b| + c = 4) :
  (a + b + c = 4) ∧
  (∀ a' b' c', a' > 0 → b' > 0 → c' > 0 → a' + b' + c' = 4 →
    (1/4) * a'^2 + (1/9) * b'^2 + c'^2 ≥ 8/7) ∧
  (∃ a' b' c', a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ a' + b' + c' = 4 ∧
    (1/4) * a'^2 + (1/9) * b'^2 + c'^2 = 8/7) :=
by sorry

end NUMINAMATH_CALUDE_inequality_selection_l3947_394778


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3947_394779

def A : Set (ℝ × ℝ) := {p | 3 * p.1 - p.2 = 7}
def B : Set (ℝ × ℝ) := {p | 2 * p.1 + p.2 = 3}

theorem intersection_of_A_and_B : A ∩ B = {(2, -1)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3947_394779


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3947_394701

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_equation_solution :
  ∃ (z : ℂ), (1 - i * z + 3 * i = -1 + i * z + 3 * i) ∧ (z = -i) :=
by
  sorry


end NUMINAMATH_CALUDE_complex_equation_solution_l3947_394701


namespace NUMINAMATH_CALUDE_parallelogram_most_analogous_to_parallelepiped_l3947_394791

-- Define the characteristics of a parallelepiped
structure Parallelepiped :=
  (has_parallel_faces : Bool)

-- Define planar figures
inductive PlanarFigure
| Triangle
| Trapezoid
| Parallelogram
| Rectangle

-- Define the analogy relation
def is_analogous (p : Parallelepiped) (f : PlanarFigure) : Prop :=
  match f with
  | PlanarFigure.Parallelogram => p.has_parallel_faces
  | _ => False

-- Theorem statement
theorem parallelogram_most_analogous_to_parallelepiped :
  ∀ (p : Parallelepiped) (f : PlanarFigure),
    p.has_parallel_faces →
    is_analogous p f →
    f = PlanarFigure.Parallelogram :=
sorry

end NUMINAMATH_CALUDE_parallelogram_most_analogous_to_parallelepiped_l3947_394791


namespace NUMINAMATH_CALUDE_marie_gift_boxes_l3947_394784

/-- Represents the number of gift boxes Marie used to pack chocolate eggs. -/
def num_gift_boxes (total_eggs : ℕ) (egg_weight : ℕ) (remaining_weight : ℕ) : ℕ :=
  let total_weight := total_eggs * egg_weight
  let melted_weight := total_weight - remaining_weight
  let eggs_per_box := melted_weight / egg_weight
  total_eggs / eggs_per_box

/-- Proves that Marie packed the chocolate eggs in 4 gift boxes. -/
theorem marie_gift_boxes :
  num_gift_boxes 12 10 90 = 4 := by
  sorry

end NUMINAMATH_CALUDE_marie_gift_boxes_l3947_394784


namespace NUMINAMATH_CALUDE_smallest_n_for_3003_combinations_l3947_394782

theorem smallest_n_for_3003_combinations : ∃ (N : ℕ), N > 0 ∧ (
  (∀ k < N, Nat.choose k 5 < 3003) ∧
  Nat.choose N 5 = 3003
) := by sorry

end NUMINAMATH_CALUDE_smallest_n_for_3003_combinations_l3947_394782


namespace NUMINAMATH_CALUDE_conference_exchanges_l3947_394776

/-- The number of business card exchanges in a conference -/
def businessCardExchanges (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a conference of 10 people, where each person exchanges 
    business cards with every other person exactly once, 
    the total number of exchanges is 45 -/
theorem conference_exchanges : businessCardExchanges 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_conference_exchanges_l3947_394776


namespace NUMINAMATH_CALUDE_grouping_ways_correct_l3947_394759

/-- The number of ways to place 4 men and 5 women into three groups. -/
def groupingWays : ℕ := 360

/-- The total number of men. -/
def numMen : ℕ := 4

/-- The total number of women. -/
def numWomen : ℕ := 5

/-- The number of groups. -/
def numGroups : ℕ := 3

/-- The size of each group. -/
def groupSize : ℕ := 3

/-- Predicate to check if a group composition is valid. -/
def validGroup (men women : ℕ) : Prop :=
  men > 0 ∧ women > 0 ∧ men + women = groupSize

/-- The theorem stating the number of ways to group people. -/
theorem grouping_ways_correct :
  ∃ (g1_men g1_women g2_men g2_women g3_men g3_women : ℕ),
    validGroup g1_men g1_women ∧
    validGroup g2_men g2_women ∧
    validGroup g3_men g3_women ∧
    g1_men + g2_men + g3_men = numMen ∧
    g1_women + g2_women + g3_women = numWomen ∧
    groupingWays = 360 :=
  sorry

end NUMINAMATH_CALUDE_grouping_ways_correct_l3947_394759


namespace NUMINAMATH_CALUDE_field_trip_minibusses_l3947_394734

theorem field_trip_minibusses (num_vans : ℕ) (students_per_van : ℕ) (students_per_minibus : ℕ) (total_students : ℕ) : 
  num_vans = 6 →
  students_per_van = 10 →
  students_per_minibus = 24 →
  total_students = 156 →
  (total_students - num_vans * students_per_van) / students_per_minibus = 4 := by
sorry

end NUMINAMATH_CALUDE_field_trip_minibusses_l3947_394734


namespace NUMINAMATH_CALUDE_daps_equivalent_to_dips_l3947_394788

/-- The number of daps equivalent to 1 dop -/
def daps_per_dop : ℚ := 5 / 4

/-- The number of dops equivalent to 1 dip -/
def dops_per_dip : ℚ := 3 / 10

/-- The number of dips we want to convert to daps -/
def target_dips : ℚ := 60

theorem daps_equivalent_to_dips : 
  (daps_per_dop * dops_per_dip * target_dips : ℚ) = 45/2 := by sorry

end NUMINAMATH_CALUDE_daps_equivalent_to_dips_l3947_394788


namespace NUMINAMATH_CALUDE_count_valid_bases_for_216_l3947_394736

theorem count_valid_bases_for_216 :
  ∃! (n : ℕ), n > 0 ∧ (∃ (S : Finset ℕ), 
    (∀ b ∈ S, b > 0 ∧ ∃ k : ℕ, k > 0 ∧ b^k = 216) ∧
    S.card = n ∧
    (∀ b : ℕ, b > 0 → (∃ k : ℕ, k > 0 ∧ b^k = 216) → b ∈ S)) :=
sorry

end NUMINAMATH_CALUDE_count_valid_bases_for_216_l3947_394736


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3947_394743

theorem sum_of_coefficients (a b c : ℕ+) : 
  (∃ (x y : ℝ), x * x = 5 ∧ y * y = 7 ∧
    x + 1/x + y + 1/y = (a * x + b * y) / c ∧
    ∀ (a' b' c' : ℕ+), 
      (∃ (x' y' : ℝ), x' * x' = 5 ∧ y' * y' = 7 ∧
        x' + 1/x' + y' + 1/y' = (a' * x' + b' * y') / c') →
      c ≤ c') →
  a + b + c = 117 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3947_394743


namespace NUMINAMATH_CALUDE_speed_in_still_water_l3947_394742

def upstream_speed : ℝ := 20
def downstream_speed : ℝ := 28

theorem speed_in_still_water : 
  (upstream_speed + downstream_speed) / 2 = 24 := by sorry

end NUMINAMATH_CALUDE_speed_in_still_water_l3947_394742


namespace NUMINAMATH_CALUDE_vector_parallel_proof_l3947_394719

def vector_a (m : ℚ) : Fin 2 → ℚ := ![1, m]
def vector_b : Fin 2 → ℚ := ![3, -2]

def parallel (u v : Fin 2 → ℚ) : Prop :=
  ∃ (k : ℚ), ∀ (i : Fin 2), u i = k * v i

theorem vector_parallel_proof (m : ℚ) :
  parallel (vector_a m + vector_b) vector_b → m = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_proof_l3947_394719


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3947_394770

def M : Set ℝ := {x : ℝ | -5 < x ∧ x < 3}
def N : Set ℝ := {x : ℝ | -2 < x ∧ x < 4}

theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | -2 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3947_394770


namespace NUMINAMATH_CALUDE_first_quartile_of_list_l3947_394748

def list : List ℝ := [42, 24, 30, 28, 26, 19, 33, 35]

def median (l : List ℝ) : ℝ := sorry

def first_quartile (l : List ℝ) : ℝ :=
  let m := median l
  median (l.filter (· < m))

theorem first_quartile_of_list : first_quartile list = 25 := by sorry

end NUMINAMATH_CALUDE_first_quartile_of_list_l3947_394748


namespace NUMINAMATH_CALUDE_integral_of_f_l3947_394702

theorem integral_of_f (f : ℝ → ℝ) (h : ∀ x, f x = x^2 + 2 * ∫ x in (0:ℝ)..1, f x) : 
  ∫ x in (0:ℝ)..1, f x = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_integral_of_f_l3947_394702


namespace NUMINAMATH_CALUDE_tank_fill_time_l3947_394767

/-- The time to fill a tank with a pump and a leak -/
theorem tank_fill_time (pump_rate leak_rate : ℝ) (pump_rate_pos : pump_rate > 0) 
  (leak_rate_pos : leak_rate > 0) (pump_faster : pump_rate > leak_rate) :
  let fill_time := 1 / (pump_rate - leak_rate)
  fill_time = 1 / (1 / 2 - 1 / 26) :=
by
  sorry

#eval 1 / (1 / 2 - 1 / 26)

end NUMINAMATH_CALUDE_tank_fill_time_l3947_394767


namespace NUMINAMATH_CALUDE_fraction_zero_at_five_l3947_394775

theorem fraction_zero_at_five (x : ℝ) : 
  (x - 5) / (6 * x - 12) = 0 ↔ x = 5 ∧ 6 * x - 12 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_fraction_zero_at_five_l3947_394775


namespace NUMINAMATH_CALUDE_new_car_travel_distance_l3947_394764

/-- The distance traveled by the older car in miles -/
def older_car_distance : ℝ := 150

/-- The percentage increase in distance for the new car -/
def percentage_increase : ℝ := 0.30

/-- The distance traveled by the new car in miles -/
def new_car_distance : ℝ := older_car_distance * (1 + percentage_increase)

theorem new_car_travel_distance : new_car_distance = 195 := by
  sorry

end NUMINAMATH_CALUDE_new_car_travel_distance_l3947_394764


namespace NUMINAMATH_CALUDE_total_time_is_twelve_years_l3947_394709

/-- Represents the time taken for each activity in months -/
structure ActivityTime where
  shape : ℕ
  climb_learn : ℕ
  climb_each : ℕ
  dive_learn : ℕ
  dive_caves : ℕ

/-- Calculates the total time taken for all activities -/
def total_time (t : ActivityTime) (num_summits : ℕ) : ℕ :=
  t.shape + t.climb_learn + (num_summits * t.climb_each) + t.dive_learn + t.dive_caves

/-- Theorem stating that the total time to complete all goals is 12 years -/
theorem total_time_is_twelve_years (t : ActivityTime) (num_summits : ℕ) :
  t.shape = 24 ∧ 
  t.climb_learn = 2 * t.shape ∧ 
  num_summits = 7 ∧ 
  t.climb_each = 5 ∧ 
  t.dive_learn = 13 ∧ 
  t.dive_caves = 24 →
  total_time t num_summits = 12 * 12 := by
  sorry

#check total_time_is_twelve_years

end NUMINAMATH_CALUDE_total_time_is_twelve_years_l3947_394709


namespace NUMINAMATH_CALUDE_quadratic_odd_coefficients_irrational_roots_l3947_394773

theorem quadratic_odd_coefficients_irrational_roots (a b c : ℤ) :
  (Odd a ∧ Odd b ∧ Odd c) →
  ∀ x : ℚ, a * x^2 + b * x + c ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_odd_coefficients_irrational_roots_l3947_394773


namespace NUMINAMATH_CALUDE_racket_deal_cost_l3947_394713

/-- Calculates the total cost of two rackets given a store's deal and the full price of each racket. -/
def totalCostTwoRackets (fullPrice : ℕ) : ℕ :=
  fullPrice + (fullPrice - fullPrice / 2)

/-- Theorem stating that the total cost of two rackets is $90 given the specific conditions. -/
theorem racket_deal_cost :
  totalCostTwoRackets 60 = 90 := by
  sorry

#eval totalCostTwoRackets 60

end NUMINAMATH_CALUDE_racket_deal_cost_l3947_394713


namespace NUMINAMATH_CALUDE_larger_interior_angle_measure_l3947_394780

/-- A circular arch bridge constructed with congruent isosceles trapezoids -/
structure CircularArchBridge where
  /-- The number of trapezoids in the bridge construction -/
  num_trapezoids : ℕ
  /-- The measure of the larger interior angle of each trapezoid in degrees -/
  larger_interior_angle : ℝ
  /-- The two end trapezoids rest horizontally on the ground -/
  end_trapezoids_horizontal : Prop

/-- Theorem stating the measure of the larger interior angle in a circular arch bridge with 12 trapezoids -/
theorem larger_interior_angle_measure (bridge : CircularArchBridge) 
  (h1 : bridge.num_trapezoids = 12)
  (h2 : bridge.end_trapezoids_horizontal) :
  bridge.larger_interior_angle = 97.5 := by
  sorry

#check larger_interior_angle_measure

end NUMINAMATH_CALUDE_larger_interior_angle_measure_l3947_394780


namespace NUMINAMATH_CALUDE_valid_numbers_l3947_394700

def is_valid_number (a b : Nat) : Prop :=
  let n := 201800 + 10 * a + b
  n % 5 = 1 ∧ n % 11 = 8

theorem valid_numbers : 
  ∀ a b : Nat, 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 →
  is_valid_number a b ↔ (a = 3 ∧ b = 1) ∨ (a = 8 ∧ b = 6) :=
by sorry

end NUMINAMATH_CALUDE_valid_numbers_l3947_394700


namespace NUMINAMATH_CALUDE_congruence_problem_l3947_394746

theorem congruence_problem (x : ℤ) 
  (h1 : (4 + x) % (2^3) = 3^2 % (2^3))
  (h2 : (6 + x) % (3^3) = 2^3 % (3^3))
  (h3 : (8 + x) % (5^3) = 7^2 % (5^3)) :
  x % 30 = 17 := by
sorry

end NUMINAMATH_CALUDE_congruence_problem_l3947_394746


namespace NUMINAMATH_CALUDE_simplify_fraction_l3947_394769

theorem simplify_fraction (x y : ℝ) (hx : x ≠ 0) : (x * y) / (3 * x) = y / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3947_394769


namespace NUMINAMATH_CALUDE_triangle_altitude_l3947_394763

/-- Given a triangle with area 600 square feet and base length 30 feet,
    prove that its altitude is 40 feet. -/
theorem triangle_altitude (A : ℝ) (b : ℝ) (h : ℝ) 
    (area_eq : A = 600)
    (base_eq : b = 30)
    (area_formula : A = (1/2) * b * h) : h = 40 := by
  sorry

end NUMINAMATH_CALUDE_triangle_altitude_l3947_394763


namespace NUMINAMATH_CALUDE_binomial_coefficient_two_l3947_394752

theorem binomial_coefficient_two (n : ℕ+) : Nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_two_l3947_394752


namespace NUMINAMATH_CALUDE_part_one_part_two_l3947_394793

-- Define the sets A, B, and U
def A : Set ℝ := {x | -2 < x ∧ x < 4}
def B (m : ℝ) : Set ℝ := {x | 2*m - 1 < x ∧ x < m + 3}
def U : Set ℝ := {x | x ≤ 4}

-- Part 1
theorem part_one (m : ℝ) (h : m = -1) :
  (Uᶜ ∩ A)ᶜ ∪ B m = {x | x < 2 ∨ x = 4} ∧
  A ∩ (Uᶜ ∩ B m)ᶜ = {x | 2 ≤ x ∧ x < 4} := by sorry

-- Part 2
theorem part_two :
  {m : ℝ | A ∪ B m = A} = {m | -1/2 ≤ m ∧ m ≤ 1} ∪ {m | 4 ≤ m} := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3947_394793


namespace NUMINAMATH_CALUDE_arithmetic_mean_min_value_l3947_394785

theorem arithmetic_mean_min_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : Real.sqrt (a * b) = 1) :
  (a + b) / 2 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_min_value_l3947_394785


namespace NUMINAMATH_CALUDE_find_n_l3947_394725

theorem find_n (d Q r m n : ℝ) (hr : r > 0) (hm : m < (1 + r)^n) 
  (hQ : Q = d / ((1 + r)^n - m)) :
  n = Real.log (d / Q + m) / Real.log (1 + r) := by
  sorry

end NUMINAMATH_CALUDE_find_n_l3947_394725


namespace NUMINAMATH_CALUDE_license_plate_count_l3947_394710

/-- The number of consonants excluding 'Y' -/
def num_consonants_no_y : ℕ := 19

/-- The number of vowels including 'Y' -/
def num_vowels : ℕ := 6

/-- The number of consonants including 'Y' -/
def num_consonants_with_y : ℕ := 21

/-- The number of even digits -/
def num_even_digits : ℕ := 5

/-- The total number of valid license plates -/
def total_license_plates : ℕ := num_consonants_no_y * num_vowels * num_consonants_with_y * num_even_digits

theorem license_plate_count : total_license_plates = 11970 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l3947_394710


namespace NUMINAMATH_CALUDE_white_tiles_count_l3947_394720

theorem white_tiles_count (total : Nat) (yellow : Nat) (purple : Nat) : 
  total = 20 → yellow = 3 → purple = 6 → 
  ∃ (blue white : Nat), blue = yellow + 1 ∧ white = total - (yellow + blue + purple) ∧ white = 7 := by
  sorry

end NUMINAMATH_CALUDE_white_tiles_count_l3947_394720


namespace NUMINAMATH_CALUDE_square_area_increase_l3947_394799

/-- The increase in area of a square when its side length is increased by 2 -/
theorem square_area_increase (a : ℝ) : 
  (a + 2)^2 - a^2 = 4*a + 4 := by sorry

end NUMINAMATH_CALUDE_square_area_increase_l3947_394799


namespace NUMINAMATH_CALUDE_f_minimum_value_F_monotonicity_l3947_394711

noncomputable section

def f (x : ℝ) : ℝ := x * Real.log x

def F (a : ℝ) (x : ℝ) : ℝ := a * x^2 + Real.log x + 1

theorem f_minimum_value (x : ℝ) (hx : x > 0) :
  ∃ (min : ℝ), min = -1 / Real.exp 1 ∧ f x ≥ min := by sorry

theorem F_monotonicity (a : ℝ) (x : ℝ) (hx : x > 0) :
  (a ≥ 0 → StrictMono (F a)) ∧
  (a < 0 → 
    (∀ y z, 0 < y ∧ y < z ∧ z < Real.sqrt (-1 / (2 * a)) → F a y < F a z) ∧
    (∀ y z, Real.sqrt (-1 / (2 * a)) < y ∧ y < z → F a y > F a z)) := by sorry

end NUMINAMATH_CALUDE_f_minimum_value_F_monotonicity_l3947_394711


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l3947_394787

theorem nested_fraction_evaluation :
  2 + 2 / (2 + 2 / (2 + 3)) = 17 / 6 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l3947_394787


namespace NUMINAMATH_CALUDE_probability_at_least_one_inferior_l3947_394783

def total_pencils : ℕ := 10
def good_pencils : ℕ := 8
def inferior_pencils : ℕ := 2
def drawn_pencils : ℕ := 2

theorem probability_at_least_one_inferior :
  let total_ways := Nat.choose total_pencils drawn_pencils
  let ways_no_inferior := Nat.choose good_pencils drawn_pencils
  (total_ways - ways_no_inferior : ℚ) / total_ways = 17 / 45 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_inferior_l3947_394783


namespace NUMINAMATH_CALUDE_inequality_proof_l3947_394766

theorem inequality_proof (a b c e f : ℝ) 
  (h1 : a > b) (h2 : e > f) (h3 : c > 0) : f - a*c < e - b*c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3947_394766


namespace NUMINAMATH_CALUDE_train_distance_l3947_394771

/-- The distance between two towns given train speeds and meeting time -/
theorem train_distance (faster_speed slower_speed meeting_time : ℝ) : 
  faster_speed = 48 ∧ 
  faster_speed = slower_speed + 6 ∧ 
  meeting_time = 5 →
  (faster_speed + slower_speed) * meeting_time = 450 := by
sorry

end NUMINAMATH_CALUDE_train_distance_l3947_394771


namespace NUMINAMATH_CALUDE_modified_short_bingo_first_column_possibilities_l3947_394757

theorem modified_short_bingo_first_column_possibilities : 
  (Finset.univ.filter (λ x : Finset (Fin 12) => x.card = 5)).card = 95040 := by
  sorry

end NUMINAMATH_CALUDE_modified_short_bingo_first_column_possibilities_l3947_394757


namespace NUMINAMATH_CALUDE_area_of_quadrilateral_DFEJ_l3947_394797

/-- Given a right isosceles triangle ABC with side lengths AB = AC = 10 and BC = 10√2,
    and points D, E, F as midpoints of AB, BC, AC respectively,
    and J as the midpoint of DE,
    the area of quadrilateral DFEJ is 6.25. -/
theorem area_of_quadrilateral_DFEJ (A B C D E F J : ℝ × ℝ) : 
  let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  A = (0, 0) →
  B = (0, 10) →
  C = (10, 0) →
  d A B = 10 →
  d A C = 10 →
  d B C = 10 * Real.sqrt 2 →
  D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  E = ((B.1 + C.1) / 2, (B.2 + C.2) / 2) →
  F = ((A.1 + C.1) / 2, (A.2 + C.2) / 2) →
  J = ((D.1 + E.1) / 2, (D.2 + E.2) / 2) →
  abs ((D.1 * F.2 + F.1 * E.2 + E.1 * J.2 + J.1 * D.2) -
       (D.2 * F.1 + F.2 * E.1 + E.2 * J.1 + J.2 * D.1)) / 2 = 6.25 :=
by sorry

end NUMINAMATH_CALUDE_area_of_quadrilateral_DFEJ_l3947_394797


namespace NUMINAMATH_CALUDE_multiple_of_six_square_greater_144_less_30_l3947_394741

theorem multiple_of_six_square_greater_144_less_30 (x : ℕ) 
  (h1 : ∃ k : ℕ, x = 6 * k)
  (h2 : x^2 > 144)
  (h3 : x < 30) :
  x = 18 ∨ x = 24 := by
sorry

end NUMINAMATH_CALUDE_multiple_of_six_square_greater_144_less_30_l3947_394741


namespace NUMINAMATH_CALUDE_fuel_tank_capacity_l3947_394715

/-- The capacity of a fuel tank in gallons. -/
def tank_capacity : ℝ := 218

/-- The volume of fuel A added to the tank in gallons. -/
def fuel_A_volume : ℝ := 122

/-- The percentage of ethanol in fuel A by volume. -/
def fuel_A_ethanol_percentage : ℝ := 0.12

/-- The percentage of ethanol in fuel B by volume. -/
def fuel_B_ethanol_percentage : ℝ := 0.16

/-- The total volume of ethanol in the full tank in gallons. -/
def total_ethanol_volume : ℝ := 30

theorem fuel_tank_capacity : 
  fuel_A_volume * fuel_A_ethanol_percentage + 
  (tank_capacity - fuel_A_volume) * fuel_B_ethanol_percentage = 
  total_ethanol_volume :=
by sorry

end NUMINAMATH_CALUDE_fuel_tank_capacity_l3947_394715


namespace NUMINAMATH_CALUDE_compression_force_l3947_394721

/-- Compression force calculation for cylindrical pillars -/
theorem compression_force (T H L : ℝ) : 
  T = 3 → H = 9 → L = (30 * T^5) / H^3 → L = 10 := by
  sorry

end NUMINAMATH_CALUDE_compression_force_l3947_394721


namespace NUMINAMATH_CALUDE_sandys_initial_money_l3947_394768

/-- Sandy's shopping problem -/
theorem sandys_initial_money 
  (shirt_cost : ℝ) 
  (jacket_cost : ℝ) 
  (pocket_money : ℝ) 
  (h1 : shirt_cost = 12.14)
  (h2 : jacket_cost = 9.28)
  (h3 : pocket_money = 7.43) :
  shirt_cost + jacket_cost + pocket_money = 28.85 := by
sorry

end NUMINAMATH_CALUDE_sandys_initial_money_l3947_394768


namespace NUMINAMATH_CALUDE_floor_sqrt_20_squared_l3947_394794

theorem floor_sqrt_20_squared : ⌊Real.sqrt 20⌋^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_20_squared_l3947_394794


namespace NUMINAMATH_CALUDE_bicycle_sprocket_rotation_l3947_394714

theorem bicycle_sprocket_rotation (large_teeth small_teeth : ℕ) (large_revolution : ℝ) :
  large_teeth = 48 →
  small_teeth = 20 →
  large_revolution = 1 →
  (large_teeth : ℝ) / small_teeth * (2 * Real.pi * large_revolution) = 4.8 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_bicycle_sprocket_rotation_l3947_394714


namespace NUMINAMATH_CALUDE_sequence_general_term_l3947_394733

/-- Given a sequence {a_n} where S_n is the sum of its first n terms and S_n = 1 - (2/3)a_n,
    prove that the general term a_n is equal to (3/5) * (2/5)^(n-1). -/
theorem sequence_general_term (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : ∀ n, S n = 1 - (2/3) * a n) :
    ∀ n, a n = (3/5) * (2/5)^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_general_term_l3947_394733


namespace NUMINAMATH_CALUDE_s_equals_one_l3947_394753

theorem s_equals_one (k R : ℝ) (h : |k + R| / |R| = 0) : |k + 2*R| / |2*k + R| = 1 := by
  sorry

end NUMINAMATH_CALUDE_s_equals_one_l3947_394753


namespace NUMINAMATH_CALUDE_g_in_M_l3947_394774

-- Define the set M
def M : Set (ℝ → ℝ) :=
  {f | ∀ x₁ x₂ : ℝ, |x₁| ≤ 1 → |x₂| ≤ 1 → |f x₁ - f x₂| ≤ 4 * |x₁ - x₂|}

-- Define the function g
def g (x : ℝ) : ℝ := x^2 + 2*x - 1

-- Theorem statement
theorem g_in_M : g ∈ M := by
  sorry

end NUMINAMATH_CALUDE_g_in_M_l3947_394774


namespace NUMINAMATH_CALUDE_sqrt_product_sqrt_equals_product_sqrt_main_theorem_l3947_394704

theorem sqrt_product_sqrt_equals_product_sqrt (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
  Real.sqrt (a * Real.sqrt b) = Real.sqrt a * Real.sqrt (Real.sqrt b) :=
by sorry

theorem main_theorem : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_product_sqrt_equals_product_sqrt_main_theorem_l3947_394704


namespace NUMINAMATH_CALUDE_coin_problem_l3947_394740

/-- Represents the types of coins --/
inductive CoinType
  | Penny
  | Nickel
  | Dime
  | Quarter
  | HalfDollar

/-- Returns the value of a coin in cents --/
def coinValue (c : CoinType) : ℕ :=
  match c with
  | .Penny => 1
  | .Nickel => 5
  | .Dime => 10
  | .Quarter => 25
  | .HalfDollar => 50

/-- Represents a collection of coins --/
structure CoinCollection where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ
  halfDollars : ℕ

/-- Calculates the total number of coins in a collection --/
def totalCoins (c : CoinCollection) : ℕ :=
  c.pennies + c.nickels + c.dimes + c.quarters + c.halfDollars

/-- Calculates the total value of coins in a collection in cents --/
def totalValue (c : CoinCollection) : ℕ :=
  c.pennies * coinValue CoinType.Penny +
  c.nickels * coinValue CoinType.Nickel +
  c.dimes * coinValue CoinType.Dime +
  c.quarters * coinValue CoinType.Quarter +
  c.halfDollars * coinValue CoinType.HalfDollar

/-- The main theorem --/
theorem coin_problem (c : CoinCollection) 
  (h1 : totalCoins c = 11)
  (h2 : totalValue c = 143)
  (h3 : c.pennies ≥ 1)
  (h4 : c.nickels ≥ 1)
  (h5 : c.dimes ≥ 1)
  (h6 : c.quarters ≥ 1)
  (h7 : c.halfDollars ≥ 1) :
  c.dimes = 4 := by
  sorry

end NUMINAMATH_CALUDE_coin_problem_l3947_394740


namespace NUMINAMATH_CALUDE_quadratic_shift_sum_l3947_394750

/-- Given a quadratic function f(x) = 3x^2 - 2x + 5, when shifted 7 units right
    and 3 units up, the resulting function g(x) = ax^2 + bx + c
    satisfies a + b + c = 128 -/
theorem quadratic_shift_sum (f g : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f x = 3 * x^2 - 2 * x + 5) →
  (∀ x, g x = f (x - 7) + 3) →
  (∀ x, g x = a * x^2 + b * x + c) →
  a + b + c = 128 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_shift_sum_l3947_394750


namespace NUMINAMATH_CALUDE_quadratic_two_roots_l3947_394716

-- Define the quadratic equation
def quadratic (x k : ℝ) : ℝ := x^2 + 2*x + k

-- Define the condition for two distinct real roots
def has_two_distinct_real_roots (k : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ quadratic x k = 0 ∧ quadratic y k = 0

-- State the theorem
theorem quadratic_two_roots :
  has_two_distinct_real_roots 0 ∧ 
  ∀ k : ℝ, has_two_distinct_real_roots k → k = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_roots_l3947_394716


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l3947_394786

theorem repeating_decimal_to_fraction :
  ∃ (n : ℕ) (d : ℕ), d ≠ 0 ∧ (n / d : ℚ) = ∑' k, 6 * (1 / 10 : ℚ)^(k + 1) ∧ n / d = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l3947_394786


namespace NUMINAMATH_CALUDE_negative_two_squared_times_negative_two_squared_l3947_394792

theorem negative_two_squared_times_negative_two_squared : -2^2 * (-2)^2 = -16 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_squared_times_negative_two_squared_l3947_394792


namespace NUMINAMATH_CALUDE_integral_of_f_l3947_394730

-- Define the function f(x) = |x + 2|
def f (x : ℝ) : ℝ := |x + 2|

-- State the theorem
theorem integral_of_f : ∫ x in (-4)..3, f x = 29/2 := by sorry

end NUMINAMATH_CALUDE_integral_of_f_l3947_394730


namespace NUMINAMATH_CALUDE_greg_extra_books_l3947_394777

theorem greg_extra_books (megan_books kelcie_books greg_books : ℕ) : 
  megan_books = 32 →
  kelcie_books = megan_books / 4 →
  greg_books > 2 * kelcie_books →
  megan_books + kelcie_books + greg_books = 65 →
  greg_books - 2 * kelcie_books = 9 := by
sorry

end NUMINAMATH_CALUDE_greg_extra_books_l3947_394777
