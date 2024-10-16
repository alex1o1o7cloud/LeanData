import Mathlib

namespace NUMINAMATH_CALUDE_fraction_problem_l2626_262672

theorem fraction_problem (N : ℝ) (f : ℝ) : 
  N = 90 → 
  3 + (1/2) * (1/3) * f * N = (1/15) * N → 
  f = 1/5 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l2626_262672


namespace NUMINAMATH_CALUDE_largest_difference_l2626_262631

theorem largest_difference (A B C D E F : ℕ) 
  (hA : A = 3 * 1005^1006)
  (hB : B = 1005^1006)
  (hC : C = 1004 * 1005^1005)
  (hD : D = 3 * 1005^1005)
  (hE : E = 1005^1005)
  (hF : F = 1005^1004) :
  (A - B > B - C) ∧ 
  (A - B > C - D) ∧ 
  (A - B > D - E) ∧ 
  (A - B > E - F) :=
by sorry

end NUMINAMATH_CALUDE_largest_difference_l2626_262631


namespace NUMINAMATH_CALUDE_job_completion_time_l2626_262603

/-- The number of days it takes for two workers to complete a job together,
    given their individual work rates. -/
def days_to_complete (rate_a rate_b : ℚ) : ℚ :=
  1 / (rate_a + rate_b)

theorem job_completion_time 
  (rate_a rate_b : ℚ) 
  (h1 : rate_a = rate_b) 
  (h2 : rate_b = 1 / 12) : 
  days_to_complete rate_a rate_b = 6 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l2626_262603


namespace NUMINAMATH_CALUDE_expression_evaluation_l2626_262613

theorem expression_evaluation :
  let x : ℤ := 25
  let y : ℤ := 30
  let z : ℤ := 7
  (x - (y - z)) - ((x - y) - z) = 14 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2626_262613


namespace NUMINAMATH_CALUDE_perpendicular_bisector_of_intersecting_circles_l2626_262611

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 6*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0

-- Define the perpendicular bisector
def perp_bisector (x y : ℝ) : Prop := 3*x - y - 9 = 0

-- Theorem statement
theorem perpendicular_bisector_of_intersecting_circles :
  ∀ (A B : ℝ × ℝ),
  circle1 A.1 A.2 ∧ circle1 B.1 B.2 ∧
  circle2 A.1 A.2 ∧ circle2 B.1 B.2 ∧
  A ≠ B →
  perp_bisector ((A.1 + B.1) / 2) ((A.2 + B.2) / 2) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_of_intersecting_circles_l2626_262611


namespace NUMINAMATH_CALUDE_least_colors_for_hidden_edges_l2626_262642

/-- The size of the grid (both width and height) -/
def gridSize : ℕ := 7

/-- The total number of edges in the grid -/
def totalEdges : ℕ := 2 * gridSize * (gridSize - 1)

/-- The expected number of hidden edges given N colors -/
def expectedHiddenEdges (N : ℕ) : ℚ := totalEdges / N

/-- Theorem stating the least N for which the expected number of hidden edges is less than 3 -/
theorem least_colors_for_hidden_edges :
  ∀ N : ℕ, N ≥ 29 ↔ expectedHiddenEdges N < 3 :=
sorry

end NUMINAMATH_CALUDE_least_colors_for_hidden_edges_l2626_262642


namespace NUMINAMATH_CALUDE_other_man_age_is_36_l2626_262681

/-- The age of the other man in the group problem -/
def other_man_age : ℕ := 36

/-- The number of men in the initial group -/
def num_men : ℕ := 9

/-- The increase in average age when two women replace two men -/
def avg_age_increase : ℕ := 4

/-- The age of one of the men in the group -/
def known_man_age : ℕ := 32

/-- The average age of the two women -/
def women_avg_age : ℕ := 52

/-- The theorem stating that given the conditions, the age of the other man is 36 -/
theorem other_man_age_is_36 :
  (num_men * avg_age_increase = 2 * women_avg_age - (other_man_age + known_man_age)) →
  other_man_age = 36 := by
  sorry

#check other_man_age_is_36

end NUMINAMATH_CALUDE_other_man_age_is_36_l2626_262681


namespace NUMINAMATH_CALUDE_boat_stream_speed_l2626_262657

/-- Proves that the speed of a stream is 5 km/hr given the conditions of the boat problem -/
theorem boat_stream_speed 
  (boat_speed : ℝ) 
  (distance : ℝ) 
  (time : ℝ) 
  (h1 : boat_speed = 22) 
  (h2 : distance = 81) 
  (h3 : time = 3) : 
  ∃ stream_speed : ℝ, 
    stream_speed = 5 ∧ 
    (boat_speed + stream_speed) * time = distance := by
  sorry

end NUMINAMATH_CALUDE_boat_stream_speed_l2626_262657


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l2626_262697

theorem quadratic_root_problem (m : ℝ) : 
  (1 : ℝ) ^ 2 + m * 1 - 4 = 0 → 
  ∃ (x : ℝ), x ≠ 1 ∧ x ^ 2 + m * x - 4 = 0 ∧ x = -4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l2626_262697


namespace NUMINAMATH_CALUDE_milk_dilution_l2626_262690

theorem milk_dilution (whole_milk : ℝ) (added_skimmed_milk : ℝ) 
  (h1 : whole_milk = 1) 
  (h2 : added_skimmed_milk = 1/4) : 
  let initial_cream := 0.05 * whole_milk
  let initial_skimmed := 0.95 * whole_milk
  let total_volume := whole_milk + added_skimmed_milk
  let final_cream_percentage := initial_cream / total_volume
  final_cream_percentage = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_milk_dilution_l2626_262690


namespace NUMINAMATH_CALUDE_divisibility_condition_l2626_262683

-- Define the predicate for divisibility
def divides (m n : ℤ) : Prop := ∃ k : ℤ, n = m * k

theorem divisibility_condition (p a : ℤ) : 
  (p ≥ 2) → 
  (a ≥ 1) → 
  Prime p → 
  p ≠ a → 
  (divides (a + p) (a^2 + p^2) ↔ 
    ((a = p) ∨ (a = p^2 - p) ∨ (a = 2*p^2 - p))) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_condition_l2626_262683


namespace NUMINAMATH_CALUDE_books_sold_l2626_262627

theorem books_sold (initial_books final_books : ℕ) 
  (h1 : initial_books = 255)
  (h2 : final_books = 145) :
  initial_books - final_books = 110 := by
  sorry

end NUMINAMATH_CALUDE_books_sold_l2626_262627


namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l2626_262636

theorem smallest_sum_of_reciprocals (x y : ℕ+) : 
  x ≠ y → (1 : ℚ) / x + (1 : ℚ) / y = 1 / 12 → (x : ℕ) + y ≥ 49 := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l2626_262636


namespace NUMINAMATH_CALUDE_nancy_pears_l2626_262647

theorem nancy_pears (total_pears alyssa_pears : ℕ) 
  (h1 : total_pears = 59)
  (h2 : alyssa_pears = 42) :
  total_pears - alyssa_pears = 17 := by
sorry

end NUMINAMATH_CALUDE_nancy_pears_l2626_262647


namespace NUMINAMATH_CALUDE_apples_in_basket_l2626_262634

/-- Represents the number of oranges in the basket -/
def oranges : ℕ := sorry

/-- Represents the number of apples in the basket -/
def apples : ℕ := 4 * oranges

/-- The total number of fruits consumed if 2/3 of each fruit's quantity is eaten -/
def consumed_fruits : ℕ := 50

theorem apples_in_basket : apples = 60 := by
  sorry

end NUMINAMATH_CALUDE_apples_in_basket_l2626_262634


namespace NUMINAMATH_CALUDE_shifted_line_equation_l2626_262606

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Shifts a line vertically by a given amount -/
def shift_line (l : Line) (shift : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + shift }

theorem shifted_line_equation (x y : ℝ) :
  let original_line := Line.mk (-2) 0
  let shifted_line := shift_line original_line 3
  y = shifted_line.slope * x + shifted_line.intercept ↔ y = -2 * x + 3 := by
  sorry

end NUMINAMATH_CALUDE_shifted_line_equation_l2626_262606


namespace NUMINAMATH_CALUDE_fraction_simplification_l2626_262650

theorem fraction_simplification (b y : ℝ) (h : b^2 ≠ y^2) :
  (Real.sqrt (b^2 + y^2) + (y^2 - b^2) / Real.sqrt (b^2 + y^2)) / (b^2 - y^2) = (b^2 + y^2) / (b^2 - y^2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2626_262650


namespace NUMINAMATH_CALUDE_combined_tax_rate_l2626_262614

theorem combined_tax_rate 
  (john_tax_rate : Real) 
  (ingrid_tax_rate : Real)
  (john_income : Real)
  (ingrid_income : Real)
  (h1 : john_tax_rate = 0.30)
  (h2 : ingrid_tax_rate = 0.40)
  (h3 : john_income = 56000)
  (h4 : ingrid_income = 72000) :
  let total_tax := john_tax_rate * john_income + ingrid_tax_rate * ingrid_income
  let total_income := john_income + ingrid_income
  total_tax / total_income = 0.35625 := by
sorry

end NUMINAMATH_CALUDE_combined_tax_rate_l2626_262614


namespace NUMINAMATH_CALUDE_geometric_sequence_terms_l2626_262671

theorem geometric_sequence_terms (n : ℕ) (a₁ q : ℝ) 
  (h1 : a₁^3 * q^3 = 3)
  (h2 : a₁^3 * q^(3*n - 6) = 9)
  (h3 : a₁^n * q^(n*(n-1)/2) = 729) :
  n = 12 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_terms_l2626_262671


namespace NUMINAMATH_CALUDE_trig_product_equality_l2626_262649

theorem trig_product_equality : 
  Real.sin (4 * Real.pi / 3) * Real.cos (5 * Real.pi / 6) * Real.tan (-4 * Real.pi / 3) = -3 * Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_trig_product_equality_l2626_262649


namespace NUMINAMATH_CALUDE_sum_of_a_and_d_l2626_262610

theorem sum_of_a_and_d (a b c d : ℝ) 
  (h1 : a * b + a * c + b * d + c * d = 42) 
  (h2 : b + c = 6) : 
  a + d = 7 := by
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_d_l2626_262610


namespace NUMINAMATH_CALUDE_square_difference_divided_by_ten_l2626_262609

theorem square_difference_divided_by_ten : (305^2 - 295^2) / 10 = 600 := by sorry

end NUMINAMATH_CALUDE_square_difference_divided_by_ten_l2626_262609


namespace NUMINAMATH_CALUDE_one_positive_integer_solution_l2626_262655

theorem one_positive_integer_solution : 
  ∃! (n : ℕ), n > 0 ∧ (25 : ℝ) - 5 * n > 15 :=
by sorry

end NUMINAMATH_CALUDE_one_positive_integer_solution_l2626_262655


namespace NUMINAMATH_CALUDE_power_sum_to_quadratic_expression_l2626_262645

theorem power_sum_to_quadratic_expression (x : ℝ) :
  5 * (3 : ℝ)^x = 243 →
  (x + 2) * (x - 2) = 21 - 10 * (Real.log 5 / Real.log 3) + (Real.log 5 / Real.log 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_to_quadratic_expression_l2626_262645


namespace NUMINAMATH_CALUDE_plum_cost_l2626_262687

theorem plum_cost (total_fruits : ℕ) (total_cost : ℕ) (peach_cost : ℕ) (plum_count : ℕ) :
  total_fruits = 32 →
  total_cost = 52 →
  peach_cost = 1 →
  plum_count = 20 →
  ∃ (plum_cost : ℕ), plum_cost = 2 ∧ 
    plum_cost * plum_count + peach_cost * (total_fruits - plum_count) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_plum_cost_l2626_262687


namespace NUMINAMATH_CALUDE_inverse_g_inverse_14_l2626_262600

def g (x : ℝ) : ℝ := 3 * x - 4

theorem inverse_g_inverse_14 : 
  (Function.invFun g) ((Function.invFun g) 14) = 10 / 3 := by sorry

end NUMINAMATH_CALUDE_inverse_g_inverse_14_l2626_262600


namespace NUMINAMATH_CALUDE_divisibility_by_360_l2626_262659

theorem divisibility_by_360 (p : ℕ) (h_prime : Nat.Prime p) (h_greater_than_5 : p > 5) :
  360 ∣ (p^4 - 5*p^2 + 4) := by
sorry

end NUMINAMATH_CALUDE_divisibility_by_360_l2626_262659


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2626_262698

theorem complex_equation_solution (i : ℂ) (z : ℂ) (h1 : i * i = -1) (h2 : i * z = 4 + 3 * i) : z = 3 - 4 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2626_262698


namespace NUMINAMATH_CALUDE_polynomial_expansion_l2626_262608

theorem polynomial_expansion (x : ℝ) :
  (3*x^2 + 2*x - 5)*(x - 2) - (x - 2)*(x^2 - 5*x + 28) + (4*x - 7)*(x - 2)*(x + 4) =
  6*x^3 + 4*x^2 - 93*x + 122 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l2626_262608


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2626_262689

-- Define the set of real numbers between -3 and 2
def OpenInterval : Set ℝ := {x | -3 < x ∧ x < 2}

-- Define the quadratic function ax^2 + bx + c
def QuadraticF (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the reversed quadratic function cx^2 + bx + a
def ReversedQuadraticF (a b c : ℝ) (x : ℝ) : ℝ := c * x^2 + b * x + a

-- Define the solution set of the reversed quadratic inequality
def ReversedSolutionSet : Set ℝ := {x | x < -1/3 ∨ x > 1/2}

theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h : ∀ x ∈ OpenInterval, QuadraticF a b c x > 0) :
  ∀ x, ReversedQuadraticF a b c x > 0 ↔ x ∈ ReversedSolutionSet :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2626_262689


namespace NUMINAMATH_CALUDE_nancy_metal_bead_sets_l2626_262620

/-- The number of metal bead sets Nancy bought -/
def metal_bead_sets : ℕ := 2

/-- The cost of one set of crystal beads in dollars -/
def crystal_bead_cost : ℕ := 9

/-- The cost of one set of metal beads in dollars -/
def metal_bead_cost : ℕ := 10

/-- The total amount Nancy spent in dollars -/
def total_spent : ℕ := 29

/-- Proof that Nancy bought 2 sets of metal beads -/
theorem nancy_metal_bead_sets :
  crystal_bead_cost + metal_bead_cost * metal_bead_sets = total_spent :=
by sorry

end NUMINAMATH_CALUDE_nancy_metal_bead_sets_l2626_262620


namespace NUMINAMATH_CALUDE_opposite_plus_two_equals_zero_l2626_262695

theorem opposite_plus_two_equals_zero (a : ℤ) (h : a = -2) : a + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_opposite_plus_two_equals_zero_l2626_262695


namespace NUMINAMATH_CALUDE_intersection_product_l2626_262660

-- Define the sets T and S
def T (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | a * p.1 + p.2 - 3 = 0}
def S (b : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 - p.2 - b = 0}

-- State the theorem
theorem intersection_product (a b : ℝ) : 
  S b ∩ T a = {(2, 1)} → a * b = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_product_l2626_262660


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2626_262638

theorem imaginary_part_of_z (z : ℂ) (h : (3 - 4*I)*z = 5) : z.im = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2626_262638


namespace NUMINAMATH_CALUDE_satisfying_function_form_l2626_262673

/-- A function from integers to integers satisfying the given conditions -/
def SatisfyingFunction (f : ℤ → ℤ) : Prop :=
  (∀ m : ℤ, f (m + 8) ≤ f m + 8) ∧
  (∀ m : ℤ, f (m + 11) ≥ f m + 11)

/-- The main theorem stating that any satisfying function is of the form f(m) = m + a -/
theorem satisfying_function_form (f : ℤ → ℤ) (h : SatisfyingFunction f) :
  ∃ a : ℤ, ∀ m : ℤ, f m = m + a := by
  sorry

end NUMINAMATH_CALUDE_satisfying_function_form_l2626_262673


namespace NUMINAMATH_CALUDE_right_angled_triangle_not_axisymmetric_l2626_262685

-- Define the types of geometric figures
inductive GeometricFigure
  | Angle
  | EquilateralTriangle
  | LineSegment
  | RightAngledTriangle

-- Define the property of being axisymmetric
def isAxisymmetric : GeometricFigure → Prop :=
  fun figure =>
    match figure with
    | GeometricFigure.Angle => true
    | GeometricFigure.EquilateralTriangle => true
    | GeometricFigure.LineSegment => true
    | GeometricFigure.RightAngledTriangle => false

-- Theorem statement
theorem right_angled_triangle_not_axisymmetric :
  ∀ (figure : GeometricFigure),
    ¬(isAxisymmetric figure) ↔ figure = GeometricFigure.RightAngledTriangle :=
by
  sorry

end NUMINAMATH_CALUDE_right_angled_triangle_not_axisymmetric_l2626_262685


namespace NUMINAMATH_CALUDE_total_spears_l2626_262643

/-- The number of spears that can be made from one sapling -/
def spears_per_sapling : ℕ := 3

/-- The number of spears that can be made from one log -/
def spears_per_log : ℕ := 9

/-- The number of saplings available -/
def num_saplings : ℕ := 6

/-- The number of logs available -/
def num_logs : ℕ := 1

/-- Theorem: The total number of spears Marcy can make is 27 -/
theorem total_spears : 
  spears_per_sapling * num_saplings + spears_per_log * num_logs = 27 := by
  sorry


end NUMINAMATH_CALUDE_total_spears_l2626_262643


namespace NUMINAMATH_CALUDE_cheerleader_size6_count_l2626_262653

/-- Represents the number of cheerleaders needing each uniform size -/
structure CheerleaderSizes where
  size2 : ℕ
  size6 : ℕ
  size12 : ℕ

/-- The conditions of the cheerleader uniform problem -/
def cheerleader_uniform_problem (s : CheerleaderSizes) : Prop :=
  s.size2 = 4 ∧
  s.size12 * 2 = s.size6 ∧
  s.size2 + s.size6 + s.size12 = 19

/-- The theorem stating the solution to the cheerleader uniform problem -/
theorem cheerleader_size6_count :
  ∃ s : CheerleaderSizes, cheerleader_uniform_problem s ∧ s.size6 = 10 :=
sorry

end NUMINAMATH_CALUDE_cheerleader_size6_count_l2626_262653


namespace NUMINAMATH_CALUDE_count_valid_numbers_l2626_262691

def is_valid_number (n : ℕ) : Prop :=
  let tens := n / 10
  let units := n % 10
  let a := tens + units
  10 ≤ n ∧ n < 100 ∧
  (3*n % 10 + 5*n % 10 + 7*n % 10 + 9*n % 10 = a)

theorem count_valid_numbers :
  ∃ (S : Finset ℕ), (∀ n ∈ S, is_valid_number n) ∧ S.card = 3 :=
sorry

end NUMINAMATH_CALUDE_count_valid_numbers_l2626_262691


namespace NUMINAMATH_CALUDE_angle_bisector_inequality_l2626_262694

-- Define a triangle with side lengths and angle bisectors
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  t_a : ℝ
  t_b : ℝ
  t_c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b
  bisector_a : t_a = (b * c * (a + c - b).sqrt) / (b + c)
  bisector_b : t_b = (a * c * (b + c - a).sqrt) / (a + c)

-- State the theorem
theorem angle_bisector_inequality (t : Triangle) : (t.t_a + t.t_b) / (t.a + t.b) < 4/3 := by
  sorry

end NUMINAMATH_CALUDE_angle_bisector_inequality_l2626_262694


namespace NUMINAMATH_CALUDE_nested_sqrt_value_l2626_262665

theorem nested_sqrt_value : 
  ∃ x : ℝ, x = Real.sqrt (3 - x) ∧ x = (-1 + Real.sqrt 13) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_sqrt_value_l2626_262665


namespace NUMINAMATH_CALUDE_sum_and_count_result_l2626_262646

def sum_of_integers (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ :=
  (b - a) / 2 + 1

def x : ℕ := sum_of_integers 10 20

def y : ℕ := count_even_integers 10 20

theorem sum_and_count_result : x + y = 171 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_count_result_l2626_262646


namespace NUMINAMATH_CALUDE_cubs_cardinals_home_run_difference_l2626_262661

/-- The number of home runs scored by the Chicago Cubs in the game -/
def cubs_home_runs : ℕ := 2 + 1 + 2

/-- The number of home runs scored by the Cardinals in the game -/
def cardinals_home_runs : ℕ := 1 + 1

/-- The difference in home runs between the Cubs and the Cardinals -/
def home_run_difference : ℕ := cubs_home_runs - cardinals_home_runs

theorem cubs_cardinals_home_run_difference :
  home_run_difference = 3 :=
by sorry

end NUMINAMATH_CALUDE_cubs_cardinals_home_run_difference_l2626_262661


namespace NUMINAMATH_CALUDE_zach_ben_score_difference_l2626_262675

theorem zach_ben_score_difference :
  ∀ (zach_score ben_score : ℕ),
    zach_score = 42 →
    ben_score = 21 →
    zach_score - ben_score = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_zach_ben_score_difference_l2626_262675


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l2626_262693

/-- Represents the atomic weight of an element in atomic mass units (amu) -/
def atomic_weight (element : String) : ℝ :=
  match element with
  | "Al" => 26.98
  | "O"  => 16.00
  | "C"  => 12.01
  | "N"  => 14.01
  | "H"  => 1.008
  | _    => 0  -- Default case, though not used in this problem

/-- Calculates the molecular weight of a compound given its composition -/
def molecular_weight (Al O C N H : ℕ) : ℝ :=
  Al * atomic_weight "Al" +
  O  * atomic_weight "O"  +
  C  * atomic_weight "C"  +
  N  * atomic_weight "N"  +
  H  * atomic_weight "H"

/-- Theorem stating that the molecular weight of the given compound is 146.022 amu -/
theorem compound_molecular_weight :
  molecular_weight 2 3 1 2 4 = 146.022 := by
  sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l2626_262693


namespace NUMINAMATH_CALUDE_eight_star_three_equals_fiftythree_l2626_262623

-- Define the operation ⋆
def star (a b : ℤ) : ℤ := 4*a + 6*b + 3

-- Theorem statement
theorem eight_star_three_equals_fiftythree : star 8 3 = 53 := by sorry

end NUMINAMATH_CALUDE_eight_star_three_equals_fiftythree_l2626_262623


namespace NUMINAMATH_CALUDE_probability_of_seven_in_three_elevenths_l2626_262640

-- Define the fraction
def fraction : ℚ := 3 / 11

-- Define the decimal representation as a sequence of digits
def decimal_representation : ℕ → ℕ
  | 0 => 0  -- The digit before the decimal point
  | (n + 1) => if n % 2 = 0 then 2 else 7  -- The repeating pattern 27

-- Define the probability of selecting a 7
def probability_of_seven : ℚ := 1 / 2

-- Theorem statement
theorem probability_of_seven_in_three_elevenths :
  (∃ (n : ℕ), decimal_representation n = 7) ∧ 
  (∀ (m : ℕ), m ≠ 0 → decimal_representation m = decimal_representation (m + 2)) →
  probability_of_seven = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_probability_of_seven_in_three_elevenths_l2626_262640


namespace NUMINAMATH_CALUDE_john_earnings_l2626_262670

/-- Calculates the money earned by John for repairing cars -/
def money_earned (total_cars : ℕ) (standard_cars : ℕ) (standard_time : ℕ) (hourly_rate : ℕ) : ℕ :=
  let remaining_cars := total_cars - standard_cars
  let standard_total_time := standard_cars * standard_time
  let remaining_time := remaining_cars * (standard_time + standard_time / 2)
  let total_time := standard_total_time + remaining_time
  let total_hours := (total_time + 59) / 60  -- Ceiling division
  hourly_rate * total_hours

/-- Theorem stating that John earns $80 for repairing the cars -/
theorem john_earnings : money_earned 5 3 40 20 = 80 := by
  sorry

end NUMINAMATH_CALUDE_john_earnings_l2626_262670


namespace NUMINAMATH_CALUDE_arithmetic_equation_l2626_262679

theorem arithmetic_equation : 
  (5 / 6 : ℚ) - (-2 : ℚ) + (1 + 1 / 6 : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equation_l2626_262679


namespace NUMINAMATH_CALUDE_zoo_feeding_days_l2626_262680

def num_lions : ℕ := 3
def num_tigers : ℕ := 2
def num_leopards : ℕ := 5
def num_hyenas : ℕ := 4

def lion_consumption : ℕ := 25
def tiger_consumption : ℕ := 20
def leopard_consumption : ℕ := 15
def hyena_consumption : ℕ := 10

def total_meat : ℕ := 1200

def daily_consumption : ℕ :=
  num_lions * lion_consumption +
  num_tigers * tiger_consumption +
  num_leopards * leopard_consumption +
  num_hyenas * hyena_consumption

theorem zoo_feeding_days :
  (total_meat / daily_consumption : ℕ) = 5 := by sorry

end NUMINAMATH_CALUDE_zoo_feeding_days_l2626_262680


namespace NUMINAMATH_CALUDE_prob_different_colors_is_three_fourths_l2626_262678

/-- The number of color options for shorts -/
def shorts_colors : ℕ := 3

/-- The number of color options for jerseys -/
def jersey_colors : ℕ := 4

/-- The total number of possible combinations of shorts and jerseys -/
def total_combinations : ℕ := shorts_colors * jersey_colors

/-- The number of combinations where shorts and jerseys have different colors -/
def different_color_combinations : ℕ := shorts_colors * (jersey_colors - 1)

/-- The probability of choosing different colors for shorts and jersey -/
def prob_different_colors : ℚ := different_color_combinations / total_combinations

/-- Theorem stating that the probability of choosing different colors for shorts and jersey is 3/4 -/
theorem prob_different_colors_is_three_fourths : prob_different_colors = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_prob_different_colors_is_three_fourths_l2626_262678


namespace NUMINAMATH_CALUDE_cubic_root_sum_squared_l2626_262688

theorem cubic_root_sum_squared (p q r t : ℝ) : 
  (p + q + r = 8) →
  (p * q + p * r + q * r = 14) →
  (p * q * r = 2) →
  (t = Real.sqrt p + Real.sqrt q + Real.sqrt r) →
  (t^4 - 16*t^2 - 12*t = -8) := by sorry

end NUMINAMATH_CALUDE_cubic_root_sum_squared_l2626_262688


namespace NUMINAMATH_CALUDE_polynomial_multiplication_l2626_262635

theorem polynomial_multiplication (x z : ℝ) :
  (3 * x^5 - 7 * z^3) * (9 * x^10 + 21 * x^5 * z^3 + 49 * z^6) = 27 * x^15 - 343 * z^9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_l2626_262635


namespace NUMINAMATH_CALUDE_bev_is_third_oldest_l2626_262641

/-- Represents the age of a person -/
structure Age : Type where
  value : ℕ

/-- Represents a person with their name and age -/
structure Person : Type where
  name : String
  age : Age

/-- Defines the "older than" relation between two people -/
def olderThan (p1 p2 : Person) : Prop :=
  p1.age.value > p2.age.value

theorem bev_is_third_oldest 
  (andy bev cao dhruv elcim : Person)
  (h1 : olderThan dhruv bev)
  (h2 : olderThan bev elcim)
  (h3 : olderThan andy elcim)
  (h4 : olderThan bev andy)
  (h5 : olderThan cao bev) :
  ∃ (x y : Person), 
    (olderThan x bev ∧ olderThan y bev) ∧
    (∀ (z : Person), z ≠ x ∧ z ≠ y → olderThan bev z ∨ z = bev) :=
by sorry

end NUMINAMATH_CALUDE_bev_is_third_oldest_l2626_262641


namespace NUMINAMATH_CALUDE_folded_paper_corner_distance_l2626_262648

/-- Represents a square sheet of paper with white front and black back -/
structure Paper where
  side : ℝ
  area : ℝ
  area_eq : area = side * side

/-- Represents the folded state of the paper -/
structure FoldedPaper where
  paper : Paper
  fold_length : ℝ
  black_area : ℝ
  white_area : ℝ
  black_twice_white : black_area = 2 * white_area
  areas_sum : black_area + white_area = paper.area

/-- The theorem to be proved -/
theorem folded_paper_corner_distance 
  (p : Paper) 
  (fp : FoldedPaper) 
  (h_area : p.area = 18) 
  (h_fp_paper : fp.paper = p) :
  Real.sqrt 2 * fp.fold_length = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_folded_paper_corner_distance_l2626_262648


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2626_262699

/-- Given a geometric sequence {a_n} where a₄ = 4, prove that a₂ * a₆ = 16 -/
theorem geometric_sequence_product (a : ℕ → ℝ) : 
  (∀ n m : ℕ, a (n + m) = a n * a m) →  -- geometric sequence property
  a 4 = 4 →                            -- given condition
  a 2 * a 6 = 16 :=                    -- theorem to prove
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l2626_262699


namespace NUMINAMATH_CALUDE_blue_paint_amount_l2626_262639

/-- Represents a paint mixture with blue, red, and white components -/
structure PaintMixture where
  blue : ℝ
  red : ℝ
  white : ℝ

/-- Calculates the total amount of paint in the mixture -/
def PaintMixture.total (m : PaintMixture) : ℝ := m.blue + m.red + m.white

/-- Theorem: Given the conditions, prove that 140 ounces of blue paint are added -/
theorem blue_paint_amount (m : PaintMixture) 
  (h1 : m.blue / m.total = 0.7)  -- Blue paint is 70% of the mixture
  (h2 : m.white = 20)            -- 20 ounces of white paint are added
  (h3 : m.white / m.total = 0.1) -- White paint is 10% of the mixture
  : m.blue = 140 := by
  sorry

#eval 140 -- Expected output

end NUMINAMATH_CALUDE_blue_paint_amount_l2626_262639


namespace NUMINAMATH_CALUDE_side_length_is_seven_l2626_262630

noncomputable def triangle_side_length (a c : ℝ) (B : ℝ) : ℝ :=
  Real.sqrt (a^2 + c^2 - 2*a*c*(Real.cos B))

theorem side_length_is_seven :
  let a : ℝ := 3 * Real.sqrt 3
  let c : ℝ := 2
  let B : ℝ := 150 * π / 180
  triangle_side_length a c B = 7 := by
sorry

end NUMINAMATH_CALUDE_side_length_is_seven_l2626_262630


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_problem_l2626_262633

theorem geometric_arithmetic_sequence_problem (b q a d : ℝ) 
  (h1 : b = a + d)
  (h2 : b * q = a + 3 * d)
  (h3 : b * q^2 = a + 6 * d)
  (h4 : b * (b * q) * (b * q^2) = 64) :
  b = 8 / 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_problem_l2626_262633


namespace NUMINAMATH_CALUDE_not_cube_of_integer_l2626_262663

theorem not_cube_of_integer : ¬ ∃ k : ℤ, (10^150 + 5 * 10^100 + 1 : ℤ) = k^3 := by
  sorry

end NUMINAMATH_CALUDE_not_cube_of_integer_l2626_262663


namespace NUMINAMATH_CALUDE_square_area_ratio_l2626_262619

theorem square_area_ratio (d : ℝ) (h : d > 0) :
  let small_square_side := d / Real.sqrt 2
  let big_square_side := d
  let small_square_area := small_square_side ^ 2
  let big_square_area := big_square_side ^ 2
  big_square_area / small_square_area = 2 := by
sorry

end NUMINAMATH_CALUDE_square_area_ratio_l2626_262619


namespace NUMINAMATH_CALUDE_mystery_number_sum_l2626_262666

theorem mystery_number_sum : Int → Prop :=
  fun result =>
    let mystery_number : Int := 47
    let added_number : Int := 45
    result = mystery_number + added_number

#check mystery_number_sum 92

end NUMINAMATH_CALUDE_mystery_number_sum_l2626_262666


namespace NUMINAMATH_CALUDE_unique_solution_abs_equation_l2626_262686

theorem unique_solution_abs_equation :
  ∃! x : ℝ, |x - 10| + |x - 14| = |3*x - 42| :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_abs_equation_l2626_262686


namespace NUMINAMATH_CALUDE_right_triangle_count_l2626_262616

/-- Represents a right triangle with integer vertices and right angle at the origin -/
structure RightTriangle where
  a : ℤ × ℤ
  b : ℤ × ℤ

/-- Checks if a point is the incenter of a right triangle -/
def is_incenter (t : RightTriangle) (m : ℚ × ℚ) : Prop :=
  sorry

/-- Counts the number of right triangles with given incenter -/
def count_triangles (p : ℕ) : ℕ :=
  sorry

theorem right_triangle_count (p : ℕ) (h : Nat.Prime p) :
  count_triangles p = 108 ∨ count_triangles p = 42 ∨ count_triangles p = 60 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_count_l2626_262616


namespace NUMINAMATH_CALUDE_base_8_units_digit_l2626_262629

theorem base_8_units_digit : ∃ n : ℕ, (356 * 78 + 49) % 8 = 1 ∧ (356 * 78 + 49) = 8 * n + 1 := by
  sorry

end NUMINAMATH_CALUDE_base_8_units_digit_l2626_262629


namespace NUMINAMATH_CALUDE_driver_weekly_distance_l2626_262644

def weekday_distance (speed1 speed2 speed3 time1 time2 time3 : ℕ) : ℕ :=
  speed1 * time1 + speed2 * time2 + speed3 * time3

def sunday_distance (speed time : ℕ) : ℕ :=
  speed * time

def weekly_distance (weekday_dist sunday_dist days_per_week : ℕ) : ℕ :=
  weekday_dist * days_per_week + sunday_dist

theorem driver_weekly_distance :
  let weekday_dist := weekday_distance 30 25 40 3 4 2
  let sunday_dist := sunday_distance 35 5
  weekly_distance weekday_dist sunday_dist 6 = 1795 := by sorry

end NUMINAMATH_CALUDE_driver_weekly_distance_l2626_262644


namespace NUMINAMATH_CALUDE_sum_equals_3004_5_l2626_262692

/-- Define the recursive function for the sum -/
def S (n : ℕ) : ℚ :=
  if n = 0 then 2
  else if n = 1 then 3 + (1/3) * 2
  else (2003 - n + 1 : ℚ) + (1/3) * S (n-1)

/-- The main theorem stating that S(2001) equals 3004.5 -/
theorem sum_equals_3004_5 : S 2001 = 3004.5 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_3004_5_l2626_262692


namespace NUMINAMATH_CALUDE_number_of_factors_46464_l2626_262607

theorem number_of_factors_46464 : Nat.card (Nat.divisors 46464) = 36 := by
  sorry

end NUMINAMATH_CALUDE_number_of_factors_46464_l2626_262607


namespace NUMINAMATH_CALUDE_selection_schemes_six_four_two_l2626_262625

/-- The number of ways to select 4 students from 6 to visit 4 cities,
    where 2 specific students cannot visit one particular city. -/
def selection_schemes (n : ℕ) (k : ℕ) (restricted : ℕ) : ℕ :=
  (n - restricted) * (n - restricted - 1) * (n - 2) * (n - 3)

theorem selection_schemes_six_four_two :
  selection_schemes 6 4 2 = 240 := by
  sorry

end NUMINAMATH_CALUDE_selection_schemes_six_four_two_l2626_262625


namespace NUMINAMATH_CALUDE_card_game_profit_general_card_game_profit_l2626_262682

/-- Expected profit function for the card guessing game -/
def expected_profit (r b g : ℕ) : ℚ :=
  (b - r : ℚ) + (2 * (r - b : ℚ) / (r + b : ℚ)) * g

/-- Theorem stating the expected profit for the specific game instance -/
theorem card_game_profit :
  expected_profit 2011 2012 2011 = 1 / 4023 := by
  sorry

/-- Theorem for the general case of the card guessing game -/
theorem general_card_game_profit (r b g : ℕ) (h : r + b > 0) :
  expected_profit r b g =
    (b - r : ℚ) + (2 * (r - b : ℚ) / (r + b : ℚ)) * g := by
  sorry

end NUMINAMATH_CALUDE_card_game_profit_general_card_game_profit_l2626_262682


namespace NUMINAMATH_CALUDE_equal_hire_probability_l2626_262662

/-- Represents the hiring process for a factory with n job openings and n applicants. -/
structure HiringProcess (n : ℕ) where
  (n_ge_3 : n ≥ 3)
  (job_openings : Fin n)
  (applicants : Fin n)
  (qualified : Fin n → Fin n → Prop)
  (qualified_condition : ∀ i j : Fin n, qualified i j ↔ i.val ≥ j.val)
  (arrival_order : Fin n → Fin n)
  (is_hired : Fin n → Prop)

/-- The probability of an applicant being hired. -/
def hire_probability (hp : HiringProcess n) (applicant : Fin n) : ℝ :=
  sorry

/-- Theorem stating that applicants n and n-1 have the same probability of being hired. -/
theorem equal_hire_probability (hp : HiringProcess n) :
  hire_probability hp ⟨n - 1, sorry⟩ = hire_probability hp ⟨n - 2, sorry⟩ :=
sorry

end NUMINAMATH_CALUDE_equal_hire_probability_l2626_262662


namespace NUMINAMATH_CALUDE_combination_problem_classification_l2626_262624

-- Define a type for the scenarios
inductive Scenario
| sets_two_elements
| round_robin_tournament
| two_digit_number_formation
| two_digit_number_no_repeat

-- Define what it means for a scenario to be a combination problem
def is_combination_problem (s : Scenario) : Prop :=
  match s with
  | Scenario.sets_two_elements => True
  | Scenario.round_robin_tournament => True
  | Scenario.two_digit_number_formation => False
  | Scenario.two_digit_number_no_repeat => False

-- Theorem statement
theorem combination_problem_classification :
  (is_combination_problem Scenario.sets_two_elements) ∧
  (is_combination_problem Scenario.round_robin_tournament) ∧
  (¬ is_combination_problem Scenario.two_digit_number_formation) ∧
  (¬ is_combination_problem Scenario.two_digit_number_no_repeat) := by
  sorry


end NUMINAMATH_CALUDE_combination_problem_classification_l2626_262624


namespace NUMINAMATH_CALUDE_eleven_overtake_points_l2626_262684

/-- Represents a point on a circular track -/
structure TrackPoint where
  position : ℝ
  mk_mod : position ≥ 0 ∧ position < 1

/-- Represents the movement of a person on the track -/
structure Movement where
  speed : ℝ
  startPoint : TrackPoint

/-- Calculates the number of distinct overtake points -/
def countOvertakePoints (pedestrian : Movement) (cyclist : Movement) : ℕ :=
  sorry

/-- Main theorem: There are exactly 11 distinct overtake points -/
theorem eleven_overtake_points :
  ∀ (start : TrackPoint) (pedSpeed : ℝ),
    pedSpeed > 0 →
    let cycSpeed := pedSpeed * 1.55
    let pedestrian := Movement.mk pedSpeed start
    let cyclist := Movement.mk cycSpeed start
    countOvertakePoints pedestrian cyclist = 11 :=
  sorry

end NUMINAMATH_CALUDE_eleven_overtake_points_l2626_262684


namespace NUMINAMATH_CALUDE_hallies_hourly_wage_l2626_262667

/-- Proves that Hallie's hourly wage is $10 given her work schedule and earnings --/
theorem hallies_hourly_wage (hourly_wage : ℚ) : 
  (7 + 5 + 7 : ℚ) * hourly_wage + (18 + 12 + 20 : ℚ) = 240 → hourly_wage = 10 := by
  sorry

#eval (190 : ℚ) / 19  -- This should evaluate to 10

end NUMINAMATH_CALUDE_hallies_hourly_wage_l2626_262667


namespace NUMINAMATH_CALUDE_executive_committee_selection_l2626_262628

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem executive_committee_selection (total_members senior_members committee_size : ℕ) 
  (h1 : total_members = 30)
  (h2 : senior_members = 10)
  (h3 : committee_size = 5) :
  (choose senior_members 2 * choose (total_members - senior_members) 3 +
   choose senior_members 3 * choose (total_members - senior_members) 2 +
   choose senior_members 4 * choose (total_members - senior_members) 1 +
   choose senior_members 5) = 78552 := by
  sorry

end NUMINAMATH_CALUDE_executive_committee_selection_l2626_262628


namespace NUMINAMATH_CALUDE_chairs_for_play_l2626_262674

theorem chairs_for_play (rows : ℕ) (chairs_per_row : ℕ) 
  (h1 : rows = 27) (h2 : chairs_per_row = 16) : 
  rows * chairs_per_row = 432 := by
  sorry

end NUMINAMATH_CALUDE_chairs_for_play_l2626_262674


namespace NUMINAMATH_CALUDE_complement_A_in_U_intersection_A_B_l2626_262605

-- Define the universal set U
def U : Set ℝ := {x | x ≤ 4}

-- Define set A
def A : Set ℝ := {x | -2 < x ∧ x < 3}

-- Define set B
def B : Set ℝ := {x | -3 < x ∧ x ≤ 1}

-- Theorem for the complement of A in U
theorem complement_A_in_U : 
  (U \ A) = {x | x ≤ -2 ∨ (3 ≤ x ∧ x ≤ 4)} := by sorry

-- Theorem for the intersection of A and B
theorem intersection_A_B : 
  (A ∩ B) = {x | -2 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_intersection_A_B_l2626_262605


namespace NUMINAMATH_CALUDE_hexagon_side_count_l2626_262617

/-- A convex hexagon with two distinct side lengths -/
structure ConvexHexagon where
  side1 : ℕ  -- Length of the first type of side
  side2 : ℕ  -- Length of the second type of side
  count1 : ℕ -- Number of sides with length side1
  count2 : ℕ -- Number of sides with length side2
  distinct : side1 ≠ side2
  total_sides : count1 + count2 = 6
  perimeter : side1 * count1 + side2 * count2 = 38

theorem hexagon_side_count (h : ConvexHexagon) (h_side1 : h.side1 = 7) (h_side2 : h.side2 = 4) :
  h.count2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_side_count_l2626_262617


namespace NUMINAMATH_CALUDE_sqrt_eight_minus_sqrt_two_l2626_262658

theorem sqrt_eight_minus_sqrt_two : Real.sqrt 8 - Real.sqrt 2 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_minus_sqrt_two_l2626_262658


namespace NUMINAMATH_CALUDE_arithmetic_sequence_count_100_l2626_262637

/-- The number of ways to select 3 different numbers from 1 to 100 
    that form an arithmetic sequence in their original order -/
def arithmeticSequenceCount : ℕ := 2450

/-- A function that counts the number of arithmetic sequences of length 3
    that can be formed from numbers 1 to n -/
def countArithmeticSequences (n : ℕ) : ℕ :=
  sorry

theorem arithmetic_sequence_count_100 : 
  countArithmeticSequences 100 = arithmeticSequenceCount := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_count_100_l2626_262637


namespace NUMINAMATH_CALUDE_factorization_mn_minus_9m_l2626_262618

theorem factorization_mn_minus_9m (m n : ℝ) : m * n - 9 * m = m * (n - 9) := by
  sorry

end NUMINAMATH_CALUDE_factorization_mn_minus_9m_l2626_262618


namespace NUMINAMATH_CALUDE_floor_expression_equals_twelve_l2626_262601

theorem floor_expression_equals_twelve (n : ℕ) (h : n = 1006) : 
  ⌊((n + 1)^3 / ((n - 1) * n) - (n - 1)^3 / (n * (n + 1)) + 5 : ℝ)⌋ = 12 := by
  sorry

end NUMINAMATH_CALUDE_floor_expression_equals_twelve_l2626_262601


namespace NUMINAMATH_CALUDE_sum_interior_angles_heptagon_l2626_262696

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- A heptagon is a polygon with 7 sides -/
def heptagon_sides : ℕ := 7

/-- Theorem: The sum of the interior angles of a heptagon is 900 degrees -/
theorem sum_interior_angles_heptagon :
  sum_interior_angles heptagon_sides = 900 := by
  sorry

end NUMINAMATH_CALUDE_sum_interior_angles_heptagon_l2626_262696


namespace NUMINAMATH_CALUDE_grass_field_width_l2626_262656

/-- Given a rectangular grass field with length 85 m, surrounded by a 2.5 m wide path 
    with an area of 1450 sq m, the width of the grass field is 200 m. -/
theorem grass_field_width (field_length : ℝ) (path_width : ℝ) (path_area : ℝ) :
  field_length = 85 →
  path_width = 2.5 →
  path_area = 1450 →
  ∃ field_width : ℝ,
    (field_length + 2 * path_width) * (field_width + 2 * path_width) -
    field_length * field_width = path_area ∧
    field_width = 200 :=
by sorry

end NUMINAMATH_CALUDE_grass_field_width_l2626_262656


namespace NUMINAMATH_CALUDE_sufficient_condition_range_l2626_262612

theorem sufficient_condition_range (a : ℝ) : 
  (∀ x : ℝ, x = 1 → x > a) → a < 1 := by sorry

end NUMINAMATH_CALUDE_sufficient_condition_range_l2626_262612


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2626_262651

theorem complex_fraction_simplification :
  let z₁ : ℂ := 4 + 6 * I
  let z₂ : ℂ := 4 - 6 * I
  z₁ / z₂ - z₂ / z₁ = 24 * I / 13 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2626_262651


namespace NUMINAMATH_CALUDE_simplify_expression_l2626_262622

theorem simplify_expression (a : ℝ) (h1 : a ≠ -1) (h2 : a ≠ 2) (h3 : a ≠ -2) :
  (3 / (a + 1) - a + 1) / ((a^2 - 4) / (a^2 + 2*a + 1)) = -a - 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2626_262622


namespace NUMINAMATH_CALUDE_popcorn_soda_cost_l2626_262676

/-- Calculate the total cost of popcorn and soda purchases with discounts and tax --/
theorem popcorn_soda_cost : ∃ (total_cost : ℚ),
  (let popcorn_price : ℚ := 14.7 / 5
   let soda_price : ℚ := 2
   let popcorn_quantity : ℕ := 4
   let soda_quantity : ℕ := 3
   let popcorn_discount : ℚ := 0.1
   let soda_discount : ℚ := 0.05
   let popcorn_tax : ℚ := 0.06
   let soda_tax : ℚ := 0.07

   let popcorn_subtotal : ℚ := popcorn_price * popcorn_quantity
   let soda_subtotal : ℚ := soda_price * soda_quantity

   let popcorn_discounted : ℚ := popcorn_subtotal * (1 - popcorn_discount)
   let soda_discounted : ℚ := soda_subtotal * (1 - soda_discount)

   let popcorn_total : ℚ := popcorn_discounted * (1 + popcorn_tax)
   let soda_total : ℚ := soda_discounted * (1 + soda_tax)

   total_cost = popcorn_total + soda_total) ∧
  (total_cost ≥ 17.31 ∧ total_cost < 17.33) := by
  sorry

#eval (14.7 / 5 * 4 * 0.9 * 1.06 + 2 * 3 * 0.95 * 1.07 : ℚ)

end NUMINAMATH_CALUDE_popcorn_soda_cost_l2626_262676


namespace NUMINAMATH_CALUDE_vacation_activities_l2626_262668

def pleasant_days (n : ℕ) : ℕ := 
  (n + 1) / 2 - ((n + 1) / 6 + (n + 1) / 10 - (n + 1) / 30)

def boring_days (n : ℕ) : ℕ := 
  (n + 1) / 2 - ((n / 3 + 1) / 2 + (n / 5 + 1) / 2 - (n / 15 + 1) / 2)

theorem vacation_activities (n : ℕ) (h : n = 89) : 
  pleasant_days n = 24 ∧ boring_days n = 24 := by
  sorry

end NUMINAMATH_CALUDE_vacation_activities_l2626_262668


namespace NUMINAMATH_CALUDE_box_volume_l2626_262664

/-- Given a box with specified dimensions, prove its volume is 3888 cubic inches. -/
theorem box_volume : 
  ∀ (height length width : ℝ),
    height = 12 →
    length = 3 * height →
    length = 4 * width →
    height * length * width = 3888 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_l2626_262664


namespace NUMINAMATH_CALUDE_ditch_length_greater_than_70_l2626_262654

/-- Represents a square field with irrigation ditches -/
structure IrrigatedField where
  side_length : ℝ
  ditch_length : ℝ
  max_distance_to_ditch : ℝ

/-- Theorem stating that the total length of ditches in the irrigated field is greater than 70 units -/
theorem ditch_length_greater_than_70 (field : IrrigatedField) 
  (h1 : field.side_length = 12)
  (h2 : field.max_distance_to_ditch ≤ 1) :
  field.ditch_length > 70 := by
  sorry

end NUMINAMATH_CALUDE_ditch_length_greater_than_70_l2626_262654


namespace NUMINAMATH_CALUDE_no_solutions_for_diophantine_equation_l2626_262677

theorem no_solutions_for_diophantine_equation :
  ¬∃ (m : ℕ+) (p q : ℕ), 
    Nat.Prime p ∧ Nat.Prime q ∧ 2^(m : ℕ) * p^2 + 1 = q^7 := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_for_diophantine_equation_l2626_262677


namespace NUMINAMATH_CALUDE_concert_hall_audience_l2626_262632

theorem concert_hall_audience (total_seats : ℕ) 
  (h_total : total_seats = 1260)
  (h_glasses : (7 : ℚ) / 18 * total_seats = number_with_glasses)
  (h_male_no_glasses : (6 : ℚ) / 11 * (total_seats - number_with_glasses) = number_male_no_glasses) :
  number_male_no_glasses = 420 := by
  sorry

end NUMINAMATH_CALUDE_concert_hall_audience_l2626_262632


namespace NUMINAMATH_CALUDE_correct_sum_after_change_l2626_262602

def number1 : ℕ := 935641
def number2 : ℕ := 471850
def incorrect_sum : ℕ := 1417491
def digit_to_change : ℕ := 7
def new_digit : ℕ := 8

theorem correct_sum_after_change :
  ∃ (changed_number2 : ℕ),
    (changed_number2 ≠ number2) ∧
    (∃ (pos : ℕ),
      (number2 / 10^pos) % 10 = digit_to_change ∧
      changed_number2 = number2 + (new_digit - digit_to_change) * 10^pos) ∧
    (number1 + changed_number2 = incorrect_sum) :=
  sorry

end NUMINAMATH_CALUDE_correct_sum_after_change_l2626_262602


namespace NUMINAMATH_CALUDE_chess_game_draw_probability_l2626_262626

theorem chess_game_draw_probability (P_A_not_losing P_B_not_losing : ℝ) 
  (h1 : P_A_not_losing = 0.8)
  (h2 : P_B_not_losing = 0.7)
  (h3 : ∀ P_A_win P_B_win P_draw : ℝ, 
    P_A_win + P_draw = P_A_not_losing → 
    P_B_win + P_draw = P_B_not_losing → 
    P_A_win + P_B_win + P_draw = 1 → 
    P_draw = 0.5) :
  ∃ P_draw : ℝ, P_draw = 0.5 := by
sorry

end NUMINAMATH_CALUDE_chess_game_draw_probability_l2626_262626


namespace NUMINAMATH_CALUDE_rotated_angle_measure_l2626_262652

/-- Given an initial angle of 50 degrees that is rotated 540 degrees clockwise,
    the resulting new acute angle is also 50 degrees. -/
theorem rotated_angle_measure (initial_angle rotation : ℝ) (h1 : initial_angle = 50)
    (h2 : rotation = 540) : 
    (initial_angle + rotation) % 360 = 50 ∨ 
    (360 - (initial_angle + rotation) % 360) = 50 := by
  sorry

end NUMINAMATH_CALUDE_rotated_angle_measure_l2626_262652


namespace NUMINAMATH_CALUDE_a_in_closed_unit_interval_l2626_262615

-- Define the set P
def P : Set ℝ := {x | x^2 ≤ 1}

-- Define the set M
def M (a : ℝ) : Set ℝ := {a}

-- Theorem statement
theorem a_in_closed_unit_interval (a : ℝ) (h : P ∪ M a = P) : a ∈ Set.Icc (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_a_in_closed_unit_interval_l2626_262615


namespace NUMINAMATH_CALUDE_product_of_consecutive_integers_l2626_262669

theorem product_of_consecutive_integers (n : ℕ) : 
  n = 5 → (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) = 30240 := by
  sorry

end NUMINAMATH_CALUDE_product_of_consecutive_integers_l2626_262669


namespace NUMINAMATH_CALUDE_remainder_6_pow_23_mod_5_l2626_262621

theorem remainder_6_pow_23_mod_5 : 6^23 % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_6_pow_23_mod_5_l2626_262621


namespace NUMINAMATH_CALUDE_distance_between_points_l2626_262604

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (2, 0)
  let p2 : ℝ × ℝ := (5, 9)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 3 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l2626_262604
