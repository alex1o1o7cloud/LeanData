import Mathlib

namespace NUMINAMATH_CALUDE_x_value_l4174_417481

theorem x_value : ∃ x : ℝ, 0.25 * x = 0.20 * 1000 - 30 ∧ x = 680 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l4174_417481


namespace NUMINAMATH_CALUDE_sandwich_cost_proof_l4174_417401

/-- The cost of a single soda in dollars -/
def soda_cost : ℚ := 87/100

/-- The number of sandwiches purchased -/
def num_sandwiches : ℕ := 2

/-- The number of sodas purchased -/
def num_sodas : ℕ := 4

/-- The total cost of the purchase in dollars -/
def total_cost : ℚ := 646/100

/-- The cost of a single sandwich in dollars -/
def sandwich_cost : ℚ := 149/100

theorem sandwich_cost_proof :
  num_sandwiches * sandwich_cost + num_sodas * soda_cost = total_cost :=
by sorry

end NUMINAMATH_CALUDE_sandwich_cost_proof_l4174_417401


namespace NUMINAMATH_CALUDE_total_fish_caught_l4174_417421

def leo_fish : ℕ := 40
def agrey_fish : ℕ := leo_fish + 20

theorem total_fish_caught : leo_fish + agrey_fish = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_caught_l4174_417421


namespace NUMINAMATH_CALUDE_ellipse_other_intersection_l4174_417464

/-- Define an ellipse with foci at (0,0) and (4,0) that intersects the x-axis at (1,0) -/
def ellipse (x : ℝ) : Prop :=
  (|x| + |x - 4|) = 4

/-- The other point of intersection of the ellipse with the x-axis -/
def other_intersection : ℝ := 4

/-- Theorem stating that the other point of intersection is (4,0) -/
theorem ellipse_other_intersection :
  ellipse other_intersection ∧ other_intersection ≠ 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_other_intersection_l4174_417464


namespace NUMINAMATH_CALUDE_stating_min_positive_temperatures_l4174_417426

/-- Represents the number of participants at the conference -/
def num_participants : ℕ := 9

/-- Represents the total number of positive records -/
def positive_records : ℕ := 36

/-- Represents the total number of negative records -/
def negative_records : ℕ := 36

/-- Represents the minimum number of participants with positive temperatures -/
def min_positive_temps : ℕ := 3

/-- 
Theorem stating that given the conditions of the meteorological conference,
the minimum number of participants with positive temperatures is 3.
-/
theorem min_positive_temperatures : 
  ∀ y : ℕ, 
  y ≤ num_participants →
  y * (y - 1) + (num_participants - y) * (num_participants - 1 - y) = positive_records →
  y ≥ min_positive_temps :=
by sorry

end NUMINAMATH_CALUDE_stating_min_positive_temperatures_l4174_417426


namespace NUMINAMATH_CALUDE_subcommittee_count_l4174_417478

def committee_size : ℕ := 7
def subcommittee_size : ℕ := 3

theorem subcommittee_count : 
  Nat.choose committee_size subcommittee_size = 35 := by
  sorry

end NUMINAMATH_CALUDE_subcommittee_count_l4174_417478


namespace NUMINAMATH_CALUDE_highest_probability_l4174_417455

-- Define the sample space
variable (Ω : Type*)

-- Define the events A, B, and C
variable (A B C : Set Ω)

-- Define a probability measure
variable (P : Set Ω → ℝ)

-- State the theorem
theorem highest_probability 
  (h_subset1 : C ⊆ B) 
  (h_subset2 : B ⊆ A) 
  (h_prob : ∀ X : Set Ω, 0 ≤ P X ∧ P X ≤ 1) 
  (h_monotone : ∀ X Y : Set Ω, X ⊆ Y → P X ≤ P Y) : 
  P A ≥ P B ∧ P A ≥ P C :=
by sorry

end NUMINAMATH_CALUDE_highest_probability_l4174_417455


namespace NUMINAMATH_CALUDE_last_term_is_zero_l4174_417405

def first_term : ℤ := 0
def differences : List ℤ := [2, 4, -1, 0, -5, -3, 3]

theorem last_term_is_zero :
  first_term + differences.sum = 0 := by sorry

end NUMINAMATH_CALUDE_last_term_is_zero_l4174_417405


namespace NUMINAMATH_CALUDE_real_part_of_fraction_l4174_417461

theorem real_part_of_fraction (z : ℂ) (x : ℝ) (h1 : z.im ≠ 0) (h2 : Complex.abs z = 2) (h3 : z.re = x) :
  (1 / (1 - z)).re = (1 - x) / (5 - 2 * x) := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_fraction_l4174_417461


namespace NUMINAMATH_CALUDE_min_k_value_l4174_417453

theorem min_k_value (f : ℝ → ℝ) (k : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f x = k * (x^2 - x + 1) - x^4 * (1 - x)^4) →
  (∀ x ∈ Set.Icc 0 1, f x ≥ 0) →
  k ≥ 1 / 192 :=
by sorry

end NUMINAMATH_CALUDE_min_k_value_l4174_417453


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l4174_417403

theorem y_intercept_of_line (x y : ℝ) : 
  (x + y - 1 = 0) → (0 + y - 1 = 0 → y = 1) := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l4174_417403


namespace NUMINAMATH_CALUDE_smaller_rss_better_fit_l4174_417414

/-- A regression model -/
structure RegressionModel where
  /-- The residual sum of squares of the model -/
  rss : ℝ
  /-- A measure of the model's fit quality -/
  fit_quality : ℝ

/-- The relationship between residual sum of squares and fit quality -/
axiom better_fit (m1 m2 : RegressionModel) :
  m1.rss < m2.rss → m1.fit_quality > m2.fit_quality

/-- Theorem: A smaller residual sum of squares indicates a better fit -/
theorem smaller_rss_better_fit (m1 m2 : RegressionModel) :
  m1.rss < m2.rss → m1.fit_quality > m2.fit_quality := by
  sorry


end NUMINAMATH_CALUDE_smaller_rss_better_fit_l4174_417414


namespace NUMINAMATH_CALUDE_parabola_axis_distance_l4174_417431

/-- Given a parabola x^2 = ay, if the distance from the point (0,1) to its axis of symmetry is 2, then a = -12 or a = 4. -/
theorem parabola_axis_distance (a : ℝ) : 
  (∀ x y : ℝ, x^2 = a*y → 
    (|y - 1 - (-a/4)| = 2 ↔ (a = -12 ∨ a = 4))) :=
by sorry

end NUMINAMATH_CALUDE_parabola_axis_distance_l4174_417431


namespace NUMINAMATH_CALUDE_average_sales_is_84_l4174_417467

/-- Sales data for each month -/
def sales : List Int := [120, 80, -20, 100, 140]

/-- Number of months -/
def num_months : Nat := 5

/-- Theorem: The average sales per month is 84 dollars -/
theorem average_sales_is_84 : (sales.sum / num_months : Int) = 84 := by
  sorry

end NUMINAMATH_CALUDE_average_sales_is_84_l4174_417467


namespace NUMINAMATH_CALUDE_right_triangle_altitude_segment_length_l4174_417412

/-- A right triangle with specific altitude properties -/
structure RightTriangleWithAltitudes where
  -- The lengths of the segments on the hypotenuse
  hypotenuse_segment1 : ℝ
  hypotenuse_segment2 : ℝ
  -- The length of one segment on a leg
  leg_segment : ℝ
  -- Ensure the hypotenuse segments are positive
  hyp_seg1_pos : 0 < hypotenuse_segment1
  hyp_seg2_pos : 0 < hypotenuse_segment2
  -- Ensure the leg segment is positive
  leg_seg_pos : 0 < leg_segment

/-- The theorem stating the length of the unknown segment -/
theorem right_triangle_altitude_segment_length 
  (triangle : RightTriangleWithAltitudes) 
  (h1 : triangle.hypotenuse_segment1 = 4)
  (h2 : triangle.hypotenuse_segment2 = 6)
  (h3 : triangle.leg_segment = 3) :
  ∃ y : ℝ, y = 4.5 ∧ 
    (triangle.leg_segment / triangle.hypotenuse_segment1 = 
     (triangle.leg_segment + y) / (triangle.hypotenuse_segment1 + triangle.hypotenuse_segment2)) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_altitude_segment_length_l4174_417412


namespace NUMINAMATH_CALUDE_other_number_proof_l4174_417475

/-- Given two positive integers with specific HCF and LCM, prove that if one number is 36, the other is 154 -/
theorem other_number_proof (a b : ℕ+) (h1 : Nat.gcd a b = 14) (h2 : Nat.lcm a b = 396) (h3 : a = 36) : b = 154 := by
  sorry

end NUMINAMATH_CALUDE_other_number_proof_l4174_417475


namespace NUMINAMATH_CALUDE_variance_scaled_sample_l4174_417494

-- Define a sample type
def Sample := List ℝ

-- Define the variance of a sample
noncomputable def variance (s : Sample) : ℝ := sorry

-- Define a function to scale a sample by a factor
def scaleSample (c : ℝ) (s : Sample) : Sample := sorry

-- Theorem statement
theorem variance_scaled_sample (s : Sample) (h : variance s = 3) :
  variance (scaleSample 2 s) = 4 * variance s := by sorry

end NUMINAMATH_CALUDE_variance_scaled_sample_l4174_417494


namespace NUMINAMATH_CALUDE_tangent_lines_intersection_l4174_417432

/-- The function f(x) -/
def f (t : ℝ) (x : ℝ) : ℝ := x^3 + (t - 1) * x^2 - 1

/-- The derivative of f(x) -/
def f' (t : ℝ) (x : ℝ) : ℝ := 3 * x^2 + 2 * (t - 1) * x

theorem tangent_lines_intersection (t k : ℝ) (h_k : k ≠ 0) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    f' t x₁ = k ∧ 
    f' t x₂ = k ∧ 
    2 * x₁ - 1 = f t x₁ ∧ 
    2 * x₂ - 1 = f t x₂) →
  t + k = 7 := by
  sorry

end NUMINAMATH_CALUDE_tangent_lines_intersection_l4174_417432


namespace NUMINAMATH_CALUDE_two_true_propositions_l4174_417442

theorem two_true_propositions :
  let original := ∀ x : ℝ, x > 0 → x^2 > 0
  let converse := ∀ x : ℝ, x^2 > 0 → x > 0
  let negation := ∃ x : ℝ, x > 0 ∧ x^2 ≤ 0
  let contrapositive := ∀ x : ℝ, x^2 ≤ 0 → x ≤ 0
  (original ∧ ¬converse ∧ ¬negation ∧ contrapositive) :=
by
  sorry

end NUMINAMATH_CALUDE_two_true_propositions_l4174_417442


namespace NUMINAMATH_CALUDE_janes_calculation_l4174_417425

theorem janes_calculation (x y z : ℝ) 
  (h1 : x - (y - z) = 23) 
  (h2 : x - y - z = 7) : 
  x - y = 15 := by
sorry

end NUMINAMATH_CALUDE_janes_calculation_l4174_417425


namespace NUMINAMATH_CALUDE_schoolchildren_count_l4174_417439

/-- The number of schoolchildren in the group -/
def S : ℕ := 135

/-- The number of buses initially provided -/
def n : ℕ := 6

/-- The number of schoolchildren in each bus after redistribution -/
def m : ℕ := 27

theorem schoolchildren_count :
  -- Initially, 22 people per bus with 3 left over
  S = 22 * n + 3 ∧
  -- After redistribution
  S = (n - 1) * m ∧
  -- No more than 18 buses
  n ≤ 18 ∧
  -- Each bus can hold no more than 36 people
  m ≤ 36 ∧
  -- m is greater than 22 (implied by the redistribution)
  m > 22 :=
by sorry

end NUMINAMATH_CALUDE_schoolchildren_count_l4174_417439


namespace NUMINAMATH_CALUDE_centroid_property_l4174_417428

/-- Given a triangle PQR with vertices P(-2,4), Q(6,3), and R(2,-5),
    prove that if S(x,y) is the centroid of the triangle, then 7x + 3y = 16 -/
theorem centroid_property (P Q R S : ℝ × ℝ) (x y : ℝ) :
  P = (-2, 4) →
  Q = (6, 3) →
  R = (2, -5) →
  S = (x, y) →
  S = ((P.1 + Q.1 + R.1) / 3, (P.2 + Q.2 + R.2) / 3) →
  7 * x + 3 * y = 16 := by
sorry

end NUMINAMATH_CALUDE_centroid_property_l4174_417428


namespace NUMINAMATH_CALUDE_sum_of_decimals_l4174_417407

/-- The sum of 5.47 and 2.359 is equal to 7.829 -/
theorem sum_of_decimals : (5.47 : ℚ) + (2.359 : ℚ) = (7.829 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_decimals_l4174_417407


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l4174_417454

theorem cubic_sum_theorem (a b c : ℝ) 
  (h1 : a + b + c = 2) 
  (h2 : a * b + a * c + b * c = 2) 
  (h3 : a * b * c = 3) : 
  a^3 + b^3 + c^3 = 9 := by sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l4174_417454


namespace NUMINAMATH_CALUDE_sin_square_equation_solution_l4174_417496

theorem sin_square_equation_solution (x : ℝ) :
  (Real.sin (3 * x))^2 + (Real.sin (4 * x))^2 = (Real.sin (5 * x))^2 + (Real.sin (6 * x))^2 →
  (∃ l : ℤ, x = l * π / 2) ∨ (∃ n : ℤ, x = n * π / 9) :=
by sorry

end NUMINAMATH_CALUDE_sin_square_equation_solution_l4174_417496


namespace NUMINAMATH_CALUDE_find_unknown_number_l4174_417429

theorem find_unknown_number (x : ℝ) : 
  (20 + 40 + 60) / 3 = ((60 + 35 + x) / 3) + 5 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_find_unknown_number_l4174_417429


namespace NUMINAMATH_CALUDE_factor_expression_l4174_417476

theorem factor_expression (x : ℝ) : 75 * x^13 + 200 * x^26 = 25 * x^13 * (3 + 8 * x^13) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l4174_417476


namespace NUMINAMATH_CALUDE_distance_origin_to_point_l4174_417457

/-- The distance from the origin (0, 0) to the point (12, -5) in a rectangular coordinate system is 13 units. -/
theorem distance_origin_to_point :
  Real.sqrt (12^2 + (-5)^2) = 13 := by sorry

end NUMINAMATH_CALUDE_distance_origin_to_point_l4174_417457


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l4174_417484

theorem absolute_value_equation_solution :
  ∃! y : ℝ, |y - 4| + 3 * y = 10 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l4174_417484


namespace NUMINAMATH_CALUDE_crayon_ratio_l4174_417427

theorem crayon_ratio (total : ℕ) (broken_percent : ℚ) (slightly_used : ℕ) 
  (h1 : total = 120)
  (h2 : broken_percent = 1/5)
  (h3 : slightly_used = 56) : 
  (total - (broken_percent * total).num - slightly_used) / total = 1/3 := by
sorry

end NUMINAMATH_CALUDE_crayon_ratio_l4174_417427


namespace NUMINAMATH_CALUDE_y_derivative_l4174_417487

noncomputable section

open Real

def y (x : ℝ) : ℝ := (1/6) * log ((1 - sinh (2*x)) / (2 + sinh (2*x)))

theorem y_derivative (x : ℝ) : 
  deriv y x = cosh (2*x) / (sinh (2*x)^2 + sinh (2*x) - 2) :=
by sorry

end NUMINAMATH_CALUDE_y_derivative_l4174_417487


namespace NUMINAMATH_CALUDE_prob_three_is_half_l4174_417440

/-- The decimal representation of 7/11 -/
def decimal_rep : ℚ := 7 / 11

/-- The repeating sequence in the decimal representation -/
def repeating_sequence : List ℕ := [6, 3]

/-- The probability of selecting a specific digit from the repeating sequence -/
def prob_digit (d : ℕ) : ℚ :=
  (repeating_sequence.count d : ℚ) / repeating_sequence.length

theorem prob_three_is_half :
  prob_digit 3 = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_prob_three_is_half_l4174_417440


namespace NUMINAMATH_CALUDE_minute_hand_angle_2h40m_l4174_417458

/-- The angle turned by the minute hand when the hour hand moves for a given time -/
def minute_hand_angle (hours : ℝ) (minutes : ℝ) : ℝ :=
  -(hours * 360 + minutes * 6)

/-- Theorem: When the hour hand moves for 2 hours and 40 minutes, 
    the angle turned by the minute hand is -960° -/
theorem minute_hand_angle_2h40m :
  minute_hand_angle 2 40 = -960 := by sorry

end NUMINAMATH_CALUDE_minute_hand_angle_2h40m_l4174_417458


namespace NUMINAMATH_CALUDE_quadratic_completing_square_l4174_417495

theorem quadratic_completing_square :
  ∀ x : ℝ, x^2 - 6*x + 8 = 0 ↔ (x - 3)^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_completing_square_l4174_417495


namespace NUMINAMATH_CALUDE_probability_two_black_balls_is_one_fifth_l4174_417437

def total_balls : ℕ := 15
def white_balls : ℕ := 8
def black_balls : ℕ := 7

def probability_two_black_balls : ℚ :=
  (black_balls.choose 2 : ℚ) / (total_balls.choose 2)

theorem probability_two_black_balls_is_one_fifth :
  probability_two_black_balls = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_black_balls_is_one_fifth_l4174_417437


namespace NUMINAMATH_CALUDE_regular_polygon_diagonals_l4174_417417

theorem regular_polygon_diagonals (n : ℕ) : n > 2 → (n * (n - 3)) / 2 = 90 → n = 15 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_diagonals_l4174_417417


namespace NUMINAMATH_CALUDE_modulus_of_complex_product_l4174_417488

theorem modulus_of_complex_product : ∃ (z : ℂ), z = (1 + Complex.I) * (3 - 4 * Complex.I) ∧ Complex.abs z = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_product_l4174_417488


namespace NUMINAMATH_CALUDE_polynomial_equality_l4174_417473

theorem polynomial_equality (m n : ℤ) : 
  (∀ x : ℤ, (x - 4) * (x + 8) = x^2 + m*x + n) → 
  (m = 4 ∧ n = -32) := by
sorry

end NUMINAMATH_CALUDE_polynomial_equality_l4174_417473


namespace NUMINAMATH_CALUDE_function_composition_equality_l4174_417416

theorem function_composition_equality (a b c d : ℝ) :
  let f := fun (x : ℝ) => a * x + b
  let g := fun (x : ℝ) => c * x + d
  (∀ x, f (g x) = g (f x)) ↔ (b = d ∨ a = c + 1) :=
sorry

end NUMINAMATH_CALUDE_function_composition_equality_l4174_417416


namespace NUMINAMATH_CALUDE_clothing_company_wage_promise_l4174_417434

/-- Represents the wage calculation and constraints for skilled workers in a clothing company. -/
theorem clothing_company_wage_promise (base_salary : ℝ) (wage_a : ℝ) (wage_b : ℝ) 
  (hours_per_day : ℝ) (days_per_month : ℝ) (time_a : ℝ) (time_b : ℝ) :
  base_salary = 800 →
  wage_a = 16 →
  wage_b = 12 →
  hours_per_day = 8 →
  days_per_month = 25 →
  time_a = 2 →
  time_b = 1 →
  ∀ a : ℝ, 
    a ≥ (hours_per_day * days_per_month - 2 * a) / 2 →
    a ≥ 0 →
    a ≤ hours_per_day * days_per_month / (2 * time_a) →
    base_salary + wage_a * a + wage_b * (hours_per_day * days_per_month / time_b - 2 * a / time_b) < 3000 :=
by sorry

end NUMINAMATH_CALUDE_clothing_company_wage_promise_l4174_417434


namespace NUMINAMATH_CALUDE_max_black_cells_l4174_417479

/-- Represents a board with black and white cells -/
def Board (n : ℕ) := Fin (2*n+1) → Fin (2*n+1) → Bool

/-- Checks if a 2x2 sub-board has at most 2 black cells -/
def ValidSubBoard (b : Board n) (i j : Fin (2*n)) : Prop :=
  (b i j).toNat + (b i (j+1)).toNat + (b (i+1) j).toNat + (b (i+1) (j+1)).toNat ≤ 2

/-- A board is valid if all its 2x2 sub-boards have at most 2 black cells -/
def ValidBoard (b : Board n) : Prop :=
  ∀ i j, ValidSubBoard b i j

/-- Counts the number of black cells in a board -/
def CountBlackCells (b : Board n) : ℕ :=
  (Finset.univ.sum fun i => (Finset.univ.sum fun j => (b i j).toNat))

/-- The maximum number of black cells in a valid (2n+1) × (2n+1) board is (2n+1)(n+1) -/
theorem max_black_cells (n : ℕ) :
  (∃ b : Board n, ValidBoard b ∧ CountBlackCells b = (2*n+1)*(n+1)) ∧
  (∀ b : Board n, ValidBoard b → CountBlackCells b ≤ (2*n+1)*(n+1)) := by
  sorry

end NUMINAMATH_CALUDE_max_black_cells_l4174_417479


namespace NUMINAMATH_CALUDE_jill_study_time_l4174_417459

/-- Calculates the total minutes Jill studies over 3 days given her study pattern -/
def total_study_minutes (day1_hours : ℕ) : ℕ :=
  let day2_hours := 2 * day1_hours
  let day3_hours := day2_hours - 1
  let total_hours := day1_hours + day2_hours + day3_hours
  total_hours * 60

/-- Proves that Jill's study pattern results in 540 minutes of total study time -/
theorem jill_study_time : total_study_minutes 2 = 540 := by
  sorry

end NUMINAMATH_CALUDE_jill_study_time_l4174_417459


namespace NUMINAMATH_CALUDE_dodecahedron_regions_count_l4174_417409

/-- The number of regions formed by the planes of a dodecahedron's faces --/
def num_regions : ℕ := 185

/-- The Euler characteristic for 3D space --/
def euler_characteristic : ℤ := -1

/-- The number of vertices formed by the intersecting planes --/
def num_vertices : ℕ := 52

/-- The number of edges formed by the intersecting planes --/
def num_edges : ℕ := 300

/-- The number of faces formed by the intersecting planes --/
def num_faces : ℕ := 432

/-- Theorem stating that the number of regions is correct given the Euler characteristic and the numbers of vertices, edges, and faces --/
theorem dodecahedron_regions_count :
  (num_vertices : ℤ) - (num_edges : ℤ) + (num_faces : ℤ) - (num_regions : ℤ) = euler_characteristic :=
by sorry

end NUMINAMATH_CALUDE_dodecahedron_regions_count_l4174_417409


namespace NUMINAMATH_CALUDE_omega_range_l4174_417466

open Real

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := cos (ω * x + φ)

theorem omega_range (ω φ α : ℝ) :
  ω > 0 →
  f ω φ α = 0 →
  deriv (f ω φ) α > 0 →
  (∀ x ∈ Set.Icc α (π + α), ¬ IsLocalMin (f ω φ) x) →
  ω ∈ Set.Ioo 1 (3/2) :=
sorry

end NUMINAMATH_CALUDE_omega_range_l4174_417466


namespace NUMINAMATH_CALUDE_nested_sum_equals_2002_l4174_417404

def nested_sum (n : ℕ) : ℚ :=
  if n = 0 then 2
  else n + 1 + (1 / 2) * nested_sum (n - 1)

theorem nested_sum_equals_2002 : nested_sum 1001 = 2002 := by
  sorry

end NUMINAMATH_CALUDE_nested_sum_equals_2002_l4174_417404


namespace NUMINAMATH_CALUDE_certain_number_is_26_l4174_417452

/-- The least positive integer divisible by every integer from 10 to 15 inclusive -/
def j : ℕ := sorry

/-- j is divisible by every integer from 10 to 15 inclusive -/
axiom j_divisible : ∀ k : ℕ, 10 ≤ k → k ≤ 15 → k ∣ j

/-- j is the least such positive integer -/
axiom j_least : ∀ m : ℕ, m > 0 → (∀ k : ℕ, 10 ≤ k → k ≤ 15 → k ∣ m) → j ≤ m

/-- The number that j is divided by to get 2310 -/
def x : ℕ := sorry

/-- j divided by x equals 2310 -/
axiom j_div_x : j / x = 2310

theorem certain_number_is_26 : x = 26 := by sorry

end NUMINAMATH_CALUDE_certain_number_is_26_l4174_417452


namespace NUMINAMATH_CALUDE_tangent_line_to_ln_curve_l4174_417462

theorem tangent_line_to_ln_curve (k : ℝ) :
  (∃ x : ℝ, x > 0 ∧ k * x = Real.log x ∧ k = 1 / x) → k = 1 / Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_ln_curve_l4174_417462


namespace NUMINAMATH_CALUDE_right_triangle_circles_coincide_l4174_417472

-- Define the triangle ABC
structure Triangle (A B C : ℝ × ℝ) : Prop where
  right_angle_at_B : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

-- Define the circles
def circle_BC (B C D : ℝ × ℝ) : Prop :=
  (D.1 - B.1) * (C.1 - D.1) + (D.2 - B.2) * (C.2 - D.2) = 0

def circle_AB (A B E : ℝ × ℝ) : Prop :=
  (E.1 - A.1) * (B.1 - E.1) + (E.2 - A.2) * (B.2 - E.2) = 0

-- Define the theorem
theorem right_triangle_circles_coincide 
  (A B C D E : ℝ × ℝ) 
  (h_triangle : Triangle A B C) 
  (h_circle_BC : circle_BC B C D) 
  (h_circle_AB : circle_AB A B E) 
  (h_area : abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2 = 200)
  (h_AC : ((C.1 - A.1)^2 + (C.2 - A.2)^2).sqrt = 40) :
  D = E := by
  sorry


end NUMINAMATH_CALUDE_right_triangle_circles_coincide_l4174_417472


namespace NUMINAMATH_CALUDE_integer_product_sum_l4174_417430

theorem integer_product_sum (x y : ℤ) : y = x + 2 ∧ x * y = 644 → x + y = 50 := by
  sorry

end NUMINAMATH_CALUDE_integer_product_sum_l4174_417430


namespace NUMINAMATH_CALUDE_equation_solution_l4174_417402

theorem equation_solution : 
  ∃! x : ℚ, (3 * x - 17) / 4 = (x + 12) / 5 ∧ x = 133 / 11 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l4174_417402


namespace NUMINAMATH_CALUDE_parabola_point_and_focus_l4174_417419

theorem parabola_point_and_focus (m : ℝ) (p : ℝ) : 
  p > 0 →
  ((-3)^2 = 2 * p * m) →
  (m + p / 2)^2 + (3 - p / 2)^2 = 5^2 →
  ((m = 1/2 ∧ p = 9) ∨ (m = 9/2 ∧ p = 1)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_point_and_focus_l4174_417419


namespace NUMINAMATH_CALUDE_quadratic_factorization_l4174_417483

theorem quadratic_factorization (a x : ℝ) : a * x^2 - 2 * a * x + a = a * (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l4174_417483


namespace NUMINAMATH_CALUDE_bill_lines_count_l4174_417451

/-- The number of sides in a triangle -/
def triangle_sides : ℕ := 3

/-- The number of sides in a square -/
def square_sides : ℕ := 4

/-- The number of sides in a pentagon -/
def pentagon_sides : ℕ := 5

/-- The number of triangles Bill drew -/
def triangles_drawn : ℕ := 12

/-- The number of squares Bill drew -/
def squares_drawn : ℕ := 8

/-- The number of pentagons Bill drew -/
def pentagons_drawn : ℕ := 4

/-- The total number of lines Bill drew -/
def total_lines : ℕ := 
  triangles_drawn * triangle_sides + 
  squares_drawn * square_sides + 
  pentagons_drawn * pentagon_sides

theorem bill_lines_count : total_lines = 88 := by
  sorry

end NUMINAMATH_CALUDE_bill_lines_count_l4174_417451


namespace NUMINAMATH_CALUDE_min_floor_sum_l4174_417449

theorem min_floor_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ (min : ℕ), min = 3 ∧
  (⌊(a^2 + b^2) / (a + b)⌋ + ⌊(b^2 + c^2) / (b + c)⌋ + ⌊(c^2 + a^2) / (c + a)⌋ ≥ min) ∧
  ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 →
    ⌊(x^2 + y^2) / (x + y)⌋ + ⌊(y^2 + z^2) / (y + z)⌋ + ⌊(z^2 + x^2) / (z + x)⌋ ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_floor_sum_l4174_417449


namespace NUMINAMATH_CALUDE_school_travel_time_l4174_417410

theorem school_travel_time (usual_rate : ℝ) (usual_time : ℝ) : 
  (usual_time > 0) →
  (7/6 * usual_rate * (usual_time - 5) = usual_rate * usual_time) →
  usual_time = 35 := by
sorry

end NUMINAMATH_CALUDE_school_travel_time_l4174_417410


namespace NUMINAMATH_CALUDE_expression_evaluation_l4174_417441

theorem expression_evaluation : (((2200 - 2081)^2 + 100) : ℚ) / 196 = 73 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4174_417441


namespace NUMINAMATH_CALUDE_solution_set_transformation_l4174_417456

theorem solution_set_transformation (a b c : ℝ) :
  (∀ x, ax^2 + b*x + c < 0 ↔ x < -2 ∨ x > 1/3) →
  (∀ x, c*x^2 - b*x + a ≥ 0 ↔ x ≤ -3 ∨ x ≥ 1/2) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_transformation_l4174_417456


namespace NUMINAMATH_CALUDE_rhombus_area_l4174_417468

/-- The area of a rhombus with side length 4 cm and an angle of 30° between adjacent sides is 8√3 cm². -/
theorem rhombus_area (side_length : ℝ) (angle : ℝ) :
  side_length = 4 →
  angle = 30 * π / 180 →
  let area := side_length * side_length * Real.sin angle
  area = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l4174_417468


namespace NUMINAMATH_CALUDE_pizza_eaten_after_six_trips_l4174_417445

def eat_pizza (n : ℕ) : ℚ :=
  1 - (2/3)^n

theorem pizza_eaten_after_six_trips :
  eat_pizza 6 = 665/729 := by sorry

end NUMINAMATH_CALUDE_pizza_eaten_after_six_trips_l4174_417445


namespace NUMINAMATH_CALUDE_population_increase_l4174_417485

theorem population_increase (birth_rate : ℚ) (death_rate : ℚ) (seconds_per_day : ℕ) :
  birth_rate = 7 / 2 →
  death_rate = 3 / 2 →
  seconds_per_day = 24 * 3600 →
  (birth_rate - death_rate) * seconds_per_day = 172800 := by
  sorry

end NUMINAMATH_CALUDE_population_increase_l4174_417485


namespace NUMINAMATH_CALUDE_total_cups_doubled_is_60_l4174_417450

/-- Represents the ratio of ingredients in the recipe -/
structure RecipeRatio where
  butter : ℕ
  flour : ℕ
  sugar : ℕ

/-- Calculates the total cups of ingredients for a doubled recipe -/
def totalCupsDoubled (ratio : RecipeRatio) (flourCups : ℕ) : ℕ :=
  let partSize := flourCups / ratio.flour
  let butterCups := ratio.butter * partSize
  let sugarCups := ratio.sugar * partSize
  2 * (butterCups + flourCups + sugarCups)

/-- Theorem: Given the recipe ratio and flour quantity, the total cups for a doubled recipe is 60 -/
theorem total_cups_doubled_is_60 :
  totalCupsDoubled ⟨2, 5, 3⟩ 15 = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_cups_doubled_is_60_l4174_417450


namespace NUMINAMATH_CALUDE_expression_factorization_l4174_417424

theorem expression_factorization (b : ℝ) :
  (9 * b^3 + 126 * b^2 - 11) - (-8 * b^3 + 2 * b^2 - 11) = b^2 * (17 * b + 124) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l4174_417424


namespace NUMINAMATH_CALUDE_cos_alpha_plus_five_sixths_pi_l4174_417420

theorem cos_alpha_plus_five_sixths_pi (α : Real) 
  (h : Real.sin (α + π / 3) = 1 / 4) : 
  Real.cos (α + 5 * π / 6) = -1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_plus_five_sixths_pi_l4174_417420


namespace NUMINAMATH_CALUDE_sum_and_count_integers_l4174_417469

def sum_integers (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ := (b - a) / 2 + 1

theorem sum_and_count_integers (x y : ℕ) :
  x = sum_integers 40 60 ∧
  y = count_even_integers 40 60 ∧
  x + y = 1061 →
  x = 1050 := by sorry

end NUMINAMATH_CALUDE_sum_and_count_integers_l4174_417469


namespace NUMINAMATH_CALUDE_women_work_hours_l4174_417411

/-- Represents the work rate of men and women -/
structure WorkRate where
  men : ℕ
  men_days : ℕ
  men_hours : ℕ
  women : ℕ
  women_days : ℕ
  women_hours : ℕ
  women_to_men_ratio : ℚ

/-- The given work scenario -/
def work_scenario : WorkRate where
  men := 15
  men_days := 21
  men_hours := 8
  women := 21
  women_days := 36
  women_hours := 0  -- This is what we need to prove
  women_to_men_ratio := 2/3

/-- Theorem stating that the women's work hours per day is 5 -/
theorem women_work_hours (w : WorkRate) (h : w = work_scenario) : w.women_hours = 5 := by
  sorry


end NUMINAMATH_CALUDE_women_work_hours_l4174_417411


namespace NUMINAMATH_CALUDE_ball_probabilities_l4174_417497

/-- Given a bag of balls with the following properties:
  - There are 10 balls in total.
  - The probability of drawing a black ball is 2/5.
  - The probability of drawing at least one white ball when drawing two balls is 19/20.

  This theorem proves:
  1. The probability of drawing two black balls is 6/45.
  2. The number of white balls is 5.
-/
theorem ball_probabilities
  (total_balls : ℕ)
  (prob_black : ℚ)
  (prob_at_least_one_white : ℚ)
  (h_total : total_balls = 10)
  (h_prob_black : prob_black = 2 / 5)
  (h_prob_white : prob_at_least_one_white = 19 / 20) :
  (∃ (black_balls white_balls : ℕ),
    black_balls + white_balls ≤ total_balls ∧
    (black_balls : ℚ) / total_balls = prob_black ∧
    1 - (total_balls - white_balls) * (total_balls - white_balls - 1) / (total_balls * (total_balls - 1)) = prob_at_least_one_white ∧
    black_balls * (black_balls - 1) / (total_balls * (total_balls - 1)) = 6 / 45 ∧
    white_balls = 5) :=
by sorry

end NUMINAMATH_CALUDE_ball_probabilities_l4174_417497


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l4174_417474

def U : Set ℕ := {x | x ≤ 8}
def A : Set ℕ := {1, 3, 7}
def B : Set ℕ := {2, 3, 8}

theorem complement_intersection_theorem :
  (U \ A) ∩ (U \ B) = {0, 4, 5, 6} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l4174_417474


namespace NUMINAMATH_CALUDE_sandwich_non_condiment_percentage_l4174_417408

theorem sandwich_non_condiment_percentage :
  let total_weight : ℝ := 150
  let condiment_weight : ℝ := 45
  let non_condiment_weight : ℝ := total_weight - condiment_weight
  let non_condiment_fraction : ℝ := non_condiment_weight / total_weight
  non_condiment_fraction * 100 = 70 := by
sorry

end NUMINAMATH_CALUDE_sandwich_non_condiment_percentage_l4174_417408


namespace NUMINAMATH_CALUDE_weight_of_b_l4174_417493

theorem weight_of_b (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 42)
  (h2 : (a + b) / 2 = 40)
  (h3 : (b + c) / 2 = 43) : 
  b = 40 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_b_l4174_417493


namespace NUMINAMATH_CALUDE_average_value_of_sequence_l4174_417433

theorem average_value_of_sequence (z : ℝ) : 
  (0 + 3*z + 6*z + 12*z + 24*z) / 5 = 9*z := by sorry

end NUMINAMATH_CALUDE_average_value_of_sequence_l4174_417433


namespace NUMINAMATH_CALUDE_polar_coordinate_equivalence_l4174_417418

/-- 
Given a point in polar coordinates (-5, 5π/7), prove that it is equivalent 
to the point (5, 12π/7) in standard polar coordinate representation, 
where r > 0 and 0 ≤ θ < 2π.
-/
theorem polar_coordinate_equivalence :
  ∀ (r θ : ℝ), 
  r = -5 ∧ θ = (5 * Real.pi) / 7 →
  ∃ (r' θ' : ℝ),
    r' > 0 ∧ 
    0 ≤ θ' ∧ 
    θ' < 2 * Real.pi ∧
    r' = 5 ∧ 
    θ' = (12 * Real.pi) / 7 ∧
    (r * (Real.cos θ), r * (Real.sin θ)) = (r' * (Real.cos θ'), r' * (Real.sin θ')) :=
by sorry

end NUMINAMATH_CALUDE_polar_coordinate_equivalence_l4174_417418


namespace NUMINAMATH_CALUDE_linear_function_value_l4174_417438

/-- A linear function on the Cartesian plane -/
def linear_function (f : ℝ → ℝ) : Prop :=
  ∃ m c : ℝ, ∀ x, f x = m * x + c

theorem linear_function_value
  (f : ℝ → ℝ)
  (h1 : linear_function f)
  (h2 : ∀ x, f (x + 4) - f x = 10)
  (h3 : f 0 = 3) :
  f 20 = 53 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_value_l4174_417438


namespace NUMINAMATH_CALUDE_trajectory_of_center_l4174_417436

-- Define the circles F1 and F2
def circle_F1 (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1
def circle_F2 (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 25

-- Define the property of being externally tangent
def externally_tangent (C_x C_y R : ℝ) : Prop :=
  (C_x + 1)^2 + C_y^2 = (R + 1)^2

-- Define the property of being internally tangent
def internally_tangent (C_x C_y R : ℝ) : Prop :=
  (C_x - 1)^2 + C_y^2 = (5 - R)^2

-- Theorem stating the trajectory of the center C
theorem trajectory_of_center :
  ∀ C_x C_y R : ℝ,
  externally_tangent C_x C_y R →
  internally_tangent C_x C_y R →
  C_x^2 / 9 + C_y^2 / 8 = 1 :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_center_l4174_417436


namespace NUMINAMATH_CALUDE_plane_equation_l4174_417489

/-- The equation of a plane passing through a point and parallel to another plane -/
theorem plane_equation (x y z : ℝ) : ∃ (A B C D : ℤ),
  -- The plane passes through the point (2,3,-1)
  A * 2 + B * 3 + C * (-1) + D = 0 ∧
  -- The plane is parallel to 3x - 4y + 2z = 5
  ∃ (k : ℝ), k ≠ 0 ∧ A = k * 3 ∧ B = k * (-4) ∧ C = k * 2 ∧
  -- The equation is in the form Ax + By + Cz + D = 0
  A * x + B * y + C * z + D = 0 ∧
  -- A is positive
  A > 0 ∧
  -- The greatest common divisor of |A|, |B|, |C|, and |D| is 1
  Nat.gcd (Nat.gcd (Nat.gcd (Int.natAbs A) (Int.natAbs B)) (Int.natAbs C)) (Int.natAbs D) = 1 ∧
  -- The specific solution
  A = 3 ∧ B = -4 ∧ C = 2 ∧ D = 8 := by
sorry

end NUMINAMATH_CALUDE_plane_equation_l4174_417489


namespace NUMINAMATH_CALUDE_apples_left_is_340_l4174_417448

/-- The number of baskets --/
def num_baskets : ℕ := 11

/-- The number of children --/
def num_children : ℕ := 10

/-- The total number of apples initially --/
def total_apples : ℕ := 1000

/-- The sum of the first n natural numbers --/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of apples picked by all children --/
def apples_picked : ℕ := num_children * sum_first_n num_baskets

/-- The number of apples left after picking --/
def apples_left : ℕ := total_apples - apples_picked

theorem apples_left_is_340 : apples_left = 340 := by
  sorry

end NUMINAMATH_CALUDE_apples_left_is_340_l4174_417448


namespace NUMINAMATH_CALUDE_max_value_constraint_l4174_417492

theorem max_value_constraint (x y : ℝ) (h : x^2 + y^2 + x*y = 1) :
  ∃ (M : ℝ), M = 5 ∧ ∀ (a b : ℝ), a^2 + b^2 + a*b = 1 → 3*a - 2*b ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_constraint_l4174_417492


namespace NUMINAMATH_CALUDE_mutually_exclusive_complementary_l4174_417460

-- Define the sample space
def Ω : Type := Unit

-- Define the events
def A : Set Ω := sorry
def B : Set Ω := sorry
def C : Set Ω := sorry
def D : Set Ω := sorry

-- Theorem for mutually exclusive events
theorem mutually_exclusive :
  (A ∩ B = ∅) ∧ (A ∩ C = ∅) ∧ (B ∩ D = ∅) :=
sorry

-- Theorem for complementary events
theorem complementary :
  B ∪ D = Set.univ ∧ B ∩ D = ∅ :=
sorry

end NUMINAMATH_CALUDE_mutually_exclusive_complementary_l4174_417460


namespace NUMINAMATH_CALUDE_special_polygon_area_l4174_417499

/-- A polygon with 32 congruent sides, where each side is perpendicular to its adjacent sides -/
structure SpecialPolygon where
  sides : ℕ
  side_length : ℝ
  perimeter : ℝ
  sides_eq : sides = 32
  perimeter_eq : perimeter = 64
  perimeter_calc : perimeter = sides * side_length

/-- The area of the special polygon -/
def polygon_area (p : SpecialPolygon) : ℝ :=
  36 * p.side_length ^ 2

theorem special_polygon_area (p : SpecialPolygon) : polygon_area p = 144 := by
  sorry

end NUMINAMATH_CALUDE_special_polygon_area_l4174_417499


namespace NUMINAMATH_CALUDE_domino_placement_theorem_l4174_417444

/-- Represents a chessboard with dimensions n x n -/
structure Chessboard (n : ℕ) where
  size : n > 0

/-- Represents a domino with dimensions 1 x 2 -/
structure Domino where

/-- Represents a position on the chessboard -/
structure Position where
  x : ℝ
  y : ℝ

/-- Checks if a position is strictly within the chessboard boundaries -/
def Position.isWithinBoard (p : Position) (b : Chessboard n) : Prop :=
  0 < p.x ∧ p.x < n ∧ 0 < p.y ∧ p.y < n

/-- Represents a domino placement on the chessboard -/
structure DominoPlacement (b : Chessboard n) where
  center : Position
  isValid : center.isWithinBoard b

/-- Represents a configuration of domino placements on the chessboard -/
def Configuration (b : Chessboard n) := List (DominoPlacement b)

/-- Counts the number of dominoes in a configuration -/
def countDominoes (config : Configuration b) : ℕ := config.length

theorem domino_placement_theorem (b : Chessboard 8) :
  (∃ config : Configuration b, countDominoes config ≥ 40) ∧
  (∃ config : Configuration b, countDominoes config ≥ 41) ∧
  (∃ config : Configuration b, countDominoes config > 41) := by
  sorry

end NUMINAMATH_CALUDE_domino_placement_theorem_l4174_417444


namespace NUMINAMATH_CALUDE_characterization_of_good_numbers_l4174_417480

def is_good (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∣ n → (d + 1) ∣ (n + 1)

theorem characterization_of_good_numbers (n : ℕ) :
  is_good n ↔ n = 1 ∨ (Nat.Prime n ∧ n % 2 = 1) :=
sorry

end NUMINAMATH_CALUDE_characterization_of_good_numbers_l4174_417480


namespace NUMINAMATH_CALUDE_star_equality_l4174_417463

/-- Binary operation ★ on ordered pairs of integers -/
def star (a b c d : ℤ) : ℤ × ℤ := (a - c, b + d)

/-- Theorem stating that if (5,4) ★ (1,1) = (x,y) ★ (4,3), then x = 8 -/
theorem star_equality (x y : ℤ) :
  star 5 4 1 1 = star x y 4 3 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_star_equality_l4174_417463


namespace NUMINAMATH_CALUDE_line_intersects_ellipse_l4174_417498

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 5 + y^2 / 4 = 1

-- Define the line
def line (m x y : ℝ) : Prop := m * x + y + m - 1 = 0

-- Theorem statement
theorem line_intersects_ellipse (m : ℝ) : 
  ∃ (x y : ℝ), ellipse x y ∧ line m x y :=
sorry

end NUMINAMATH_CALUDE_line_intersects_ellipse_l4174_417498


namespace NUMINAMATH_CALUDE_ellipse_circle_intersection_l4174_417491

/-- Given an ellipse and a circle with specific properties, prove that a line passing through the origin and intersecting the circle at two points satisfying a dot product condition has specific equations. -/
theorem ellipse_circle_intersection (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) : 
  let e := Real.sqrt 3 / 2
  let t_area := Real.sqrt 3
  let c := Real.sqrt (a^2 - b^2)
  let ellipse := fun (x y : ℝ) ↦ x^2 / a^2 + y^2 / b^2 = 1
  let circle := fun (x y : ℝ) ↦ (x - a)^2 + (y - b)^2 = (a / b)^2
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    e = c / a →
    t_area = c * b →
    (∃ k : ℝ, (y₁ = k * x₁ ∧ y₂ = k * x₂) ∨ (x₁ = 0 ∧ x₂ = 0)) →
    circle x₁ y₁ →
    circle x₂ y₂ →
    (x₁ - a) * (x₂ - a) + (y₁ - b) * (y₂ - b) = -2 →
    (y₁ = 0 ∧ y₂ = 0) ∨ (∃ k : ℝ, k = 4/3 ∧ y₁ = k * x₁ ∧ y₂ = k * x₂) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_circle_intersection_l4174_417491


namespace NUMINAMATH_CALUDE_triangle_side_length_l4174_417465

theorem triangle_side_length (a b c : ℝ) (C : ℝ) : 
  a = 2 → b = 3 → C = Real.pi / 3 → c = Real.sqrt 7 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l4174_417465


namespace NUMINAMATH_CALUDE_lindas_hourly_rate_l4174_417435

/-- Proves that Linda's hourly rate for babysitting is $10.00 -/
theorem lindas_hourly_rate (application_fee : ℝ) (num_colleges : ℕ) (hours_worked : ℝ) :
  application_fee = 25 →
  num_colleges = 6 →
  hours_worked = 15 →
  (application_fee * num_colleges) / hours_worked = 10 := by
  sorry

end NUMINAMATH_CALUDE_lindas_hourly_rate_l4174_417435


namespace NUMINAMATH_CALUDE_pen_collection_l4174_417443

theorem pen_collection (initial_pens : ℕ) (mike_pens : ℕ) (sharon_pens : ℕ) : 
  initial_pens = 25 →
  mike_pens = 22 →
  sharon_pens = 19 →
  2 * (initial_pens + mike_pens) - sharon_pens = 75 := by
  sorry

end NUMINAMATH_CALUDE_pen_collection_l4174_417443


namespace NUMINAMATH_CALUDE_unique_element_implies_a_value_l4174_417422

def A (a : ℝ) : Set ℝ := {x | a * x^2 + 2 * x + 1 = 0}

theorem unique_element_implies_a_value (a : ℝ) : (∃! x, x ∈ A a) → a = 0 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_element_implies_a_value_l4174_417422


namespace NUMINAMATH_CALUDE_snooker_tournament_revenue_l4174_417486

theorem snooker_tournament_revenue
  (vip_price : ℚ)
  (general_price : ℚ)
  (total_tickets : ℕ)
  (ticket_difference : ℕ)
  (h1 : vip_price = 45)
  (h2 : general_price = 20)
  (h3 : total_tickets = 320)
  (h4 : ticket_difference = 276)
  : ∃ (vip_tickets general_tickets : ℕ),
    vip_tickets + general_tickets = total_tickets ∧
    vip_tickets = general_tickets - ticket_difference ∧
    vip_price * vip_tickets + general_price * general_tickets = 6950 := by
  sorry

#check snooker_tournament_revenue

end NUMINAMATH_CALUDE_snooker_tournament_revenue_l4174_417486


namespace NUMINAMATH_CALUDE_volume_of_specific_polyhedron_l4174_417423

/-- Represents a cube in 3D space -/
structure Cube where
  edge_length : ℝ

/-- Represents a convex polyhedron formed by planes passing through midpoints of cube edges -/
structure ConvexPolyhedron where
  cube : Cube

/-- Calculate the volume of the convex polyhedron -/
def volume (p : ConvexPolyhedron) : ℝ :=
  sorry

/-- Theorem stating the volume of the specific convex polyhedron -/
theorem volume_of_specific_polyhedron :
  ∀ (c : Cube) (p : ConvexPolyhedron),
    c.edge_length = 2 →
    p.cube = c →
    volume p = 32 / 3 :=
  sorry

end NUMINAMATH_CALUDE_volume_of_specific_polyhedron_l4174_417423


namespace NUMINAMATH_CALUDE_solution_set_theorem_l4174_417482

-- Define the solution sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | x^2 + x - 6 < 0}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Theorem statement
theorem solution_set_theorem (a b : ℝ) : 
  ({x : ℝ | x^2 + a*x + b < 0} = A_intersect_B) → a + b = -3 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_theorem_l4174_417482


namespace NUMINAMATH_CALUDE_coins_sold_proof_l4174_417400

def beth_initial_coins : ℕ := 250
def carl_gift_coins : ℕ := 75
def sell_percentage : ℚ := 60 / 100

theorem coins_sold_proof :
  let total_coins := beth_initial_coins + carl_gift_coins
  ⌊(sell_percentage * total_coins : ℚ)⌋ = 195 := by sorry

end NUMINAMATH_CALUDE_coins_sold_proof_l4174_417400


namespace NUMINAMATH_CALUDE_rectangle_diagonal_squares_l4174_417471

/-- The number of unit squares that the diagonals of a rectangle pass through -/
def diagonalSquares (width : ℕ) (height : ℕ) : ℕ :=
  2 * (width - 1 + height - 1 + 1) - 2

/-- Theorem: For a 20 × 19 rectangle with one corner at the origin and sides parallel to the coordinate axes,
    the number of unit squares that the two diagonals pass through is 74. -/
theorem rectangle_diagonal_squares :
  diagonalSquares 20 19 = 74 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_squares_l4174_417471


namespace NUMINAMATH_CALUDE_inverse_proportion_point_ordering_l4174_417490

/-- Given an inverse proportion function and three points on its graph, 
    prove the ordering of their x-coordinates. -/
theorem inverse_proportion_point_ordering (k : ℝ) (a b c : ℝ) : 
  (∃ (k : ℝ), -3 = -((k^2 + 1) / a) ∧ 
               -2 = -((k^2 + 1) / b) ∧ 
                1 = -((k^2 + 1) / c)) →
  c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_point_ordering_l4174_417490


namespace NUMINAMATH_CALUDE_final_sum_theorem_l4174_417413

theorem final_sum_theorem (T x y : ℝ) (h : x + y = T) :
  (2 * (2 * x + 4) + 2 * (3 * y + 4)) = 6 * T + 16 := by
  sorry

end NUMINAMATH_CALUDE_final_sum_theorem_l4174_417413


namespace NUMINAMATH_CALUDE_elevator_problem_solution_l4174_417447

/-- Represents the elevator problem with given conditions -/
structure ElevatorProblem where
  total_floors : ℕ
  first_half_time : ℕ
  mid_section_rate : ℕ
  final_section_rate : ℕ

/-- Calculates the total time in hours for the elevator to reach the bottom -/
def total_time (p : ElevatorProblem) : ℚ :=
  let first_half := p.first_half_time
  let mid_section := (p.total_floors / 4) * p.mid_section_rate
  let final_section := (p.total_floors / 4) * p.final_section_rate
  (first_half + mid_section + final_section) / 60

/-- Theorem stating that for the given problem, the total time is 2 hours -/
theorem elevator_problem_solution :
  let problem := ElevatorProblem.mk 20 15 5 16
  total_time problem = 2 := by sorry

end NUMINAMATH_CALUDE_elevator_problem_solution_l4174_417447


namespace NUMINAMATH_CALUDE_sport_water_amount_l4174_417477

/-- Represents the ratio of flavoring, corn syrup, and water in a drink formulation -/
structure DrinkRatio :=
  (flavoring : ℚ)
  (corn_syrup : ℚ)
  (water : ℚ)

/-- The standard formulation of the drink -/
def standard_ratio : DrinkRatio :=
  { flavoring := 1, corn_syrup := 12, water := 30 }

/-- The sport formulation of the drink -/
def sport_ratio (std : DrinkRatio) : DrinkRatio :=
  { flavoring := std.flavoring,
    corn_syrup := std.corn_syrup / 3,
    water := std.water * 2 }

/-- Calculates the amount of water given the amount of corn syrup and the drink ratio -/
def water_amount (corn_syrup_amount : ℚ) (ratio : DrinkRatio) : ℚ :=
  (corn_syrup_amount / ratio.corn_syrup) * ratio.water

theorem sport_water_amount :
  water_amount 4 (sport_ratio standard_ratio) = 60 := by
  sorry

end NUMINAMATH_CALUDE_sport_water_amount_l4174_417477


namespace NUMINAMATH_CALUDE_tan_2016_in_terms_of_sin_36_l4174_417470

theorem tan_2016_in_terms_of_sin_36 (a : ℝ) (h : Real.sin (36 * π / 180) = a) :
  Real.tan (2016 * π / 180) = a / Real.sqrt (1 - a^2) := by
  sorry

end NUMINAMATH_CALUDE_tan_2016_in_terms_of_sin_36_l4174_417470


namespace NUMINAMATH_CALUDE_mrs_taylor_purchase_cost_l4174_417406

/-- Calculates the total cost of smart televisions and soundbars with discounts -/
def total_cost (tv_count : ℕ) (tv_price : ℚ) (tv_discount : ℚ)
                (soundbar_count : ℕ) (soundbar_price : ℚ) (soundbar_discount : ℚ) : ℚ :=
  let tv_total := tv_count * tv_price * (1 - tv_discount)
  let soundbar_total := soundbar_count * soundbar_price * (1 - soundbar_discount)
  tv_total + soundbar_total

/-- Theorem stating that Mrs. Taylor's purchase totals $2085 -/
theorem mrs_taylor_purchase_cost :
  total_cost 2 750 0.15 3 300 0.10 = 2085 := by
  sorry

end NUMINAMATH_CALUDE_mrs_taylor_purchase_cost_l4174_417406


namespace NUMINAMATH_CALUDE_f_neg_two_l4174_417446

def f (x : ℝ) : ℝ := x^2 + 2*x - 1

theorem f_neg_two : f (-2) = -1 := by sorry

end NUMINAMATH_CALUDE_f_neg_two_l4174_417446


namespace NUMINAMATH_CALUDE_pasture_rent_is_175_l4174_417415

/-- Represents the rent share of a person -/
structure RentShare where
  oxen : ℕ
  months : ℕ
  share : ℚ

/-- Calculates the total rent of a pasture given the rent shares of three people -/
def totalRent (a b c : RentShare) : ℚ :=
  let totalOxenMonths := a.oxen * a.months + b.oxen * b.months + c.oxen * c.months
  (c.share * totalOxenMonths) / (c.oxen * c.months)

/-- Theorem stating that the total rent is 175 given the problem conditions -/
theorem pasture_rent_is_175 (a b c : RentShare)
  (ha : a.oxen = 10 ∧ a.months = 7)
  (hb : b.oxen = 12 ∧ b.months = 5)
  (hc : c.oxen = 15 ∧ c.months = 3 ∧ c.share = 45) :
  totalRent a b c = 175 := by
  sorry

#eval totalRent
  { oxen := 10, months := 7, share := 0 }
  { oxen := 12, months := 5, share := 0 }
  { oxen := 15, months := 3, share := 45 }

end NUMINAMATH_CALUDE_pasture_rent_is_175_l4174_417415
