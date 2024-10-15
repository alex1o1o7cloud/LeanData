import Mathlib

namespace NUMINAMATH_CALUDE_sam_investment_result_l14_1499

-- Define the compound interest function
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

-- Define the problem parameters
def initial_investment : ℝ := 10000
def first_rate : ℝ := 0.20
def first_time : ℕ := 3
def multiplier : ℝ := 3
def second_rate : ℝ := 0.15
def second_time : ℕ := 1

-- Theorem statement
theorem sam_investment_result :
  let first_phase := compound_interest initial_investment first_rate first_time
  let second_phase := compound_interest (first_phase * multiplier) second_rate second_time
  second_phase = 59616 := by sorry

end NUMINAMATH_CALUDE_sam_investment_result_l14_1499


namespace NUMINAMATH_CALUDE_vanya_number_theorem_l14_1473

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Define the property that a number satisfies the condition
def satisfiesCondition (n : ℕ) : Prop := n + sumOfDigits n = 2021

-- Theorem statement
theorem vanya_number_theorem : 
  (∀ n : ℕ, satisfiesCondition n ↔ (n = 2014 ∨ n = 1996)) := by sorry

end NUMINAMATH_CALUDE_vanya_number_theorem_l14_1473


namespace NUMINAMATH_CALUDE_factorize_3x_minus_12x_squared_factorize_negative_x_squared_plus_6xy_minus_9y_squared_factorize_n_squared_m_minus_2_plus_2_minus_m_factorize_a_squared_plus_4b_squared_squared_minus_16a_squared_b_squared_l14_1453

-- Problem 1
theorem factorize_3x_minus_12x_squared (x : ℝ) :
  3*x - 12*x^2 = 3*x*(1-4*x) := by sorry

-- Problem 2
theorem factorize_negative_x_squared_plus_6xy_minus_9y_squared (x y : ℝ) :
  -x^2 + 6*x*y - 9*y^2 = -(x-3*y)^2 := by sorry

-- Problem 3
theorem factorize_n_squared_m_minus_2_plus_2_minus_m (m n : ℝ) :
  n^2*(m-2) + (2-m) = (m-2)*(n+1)*(n-1) := by sorry

-- Problem 4
theorem factorize_a_squared_plus_4b_squared_squared_minus_16a_squared_b_squared (a b : ℝ) :
  (a^2 + 4*b^2)^2 - 16*a^2*b^2 = (a+2*b)^2 * (a-2*b)^2 := by sorry

end NUMINAMATH_CALUDE_factorize_3x_minus_12x_squared_factorize_negative_x_squared_plus_6xy_minus_9y_squared_factorize_n_squared_m_minus_2_plus_2_minus_m_factorize_a_squared_plus_4b_squared_squared_minus_16a_squared_b_squared_l14_1453


namespace NUMINAMATH_CALUDE_x_intercept_of_line_l14_1496

/-- The x-intercept of the line 4x + 6y = 24 is (6, 0) -/
theorem x_intercept_of_line (x y : ℝ) : 
  4 * x + 6 * y = 24 → y = 0 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_of_line_l14_1496


namespace NUMINAMATH_CALUDE_expression_evaluation_l14_1479

theorem expression_evaluation : 
  Real.sqrt 5 * 5^(1/2 : ℝ) + 18 / 3 * 2 - 9^(3/2 : ℝ) + 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l14_1479


namespace NUMINAMATH_CALUDE_sector_area_l14_1439

/-- The area of a sector with radius 2 and perimeter equal to the circumference of its circle is 4π - 2 -/
theorem sector_area (r : ℝ) (θ : ℝ) : 
  r = 2 → 
  2 * r + r * θ = 2 * π * r → 
  (1/2) * r^2 * θ = 4 * π - 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l14_1439


namespace NUMINAMATH_CALUDE_triangle_ABC_coordinates_l14_1481

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a triangle given three points -/
def triangleArea (a b c : Point) : ℝ := sorry

/-- Checks if a point is on a coordinate axis -/
def isOnAxis (p : Point) : Prop :=
  p.x = 0 ∨ p.y = 0

theorem triangle_ABC_coordinates :
  let a : Point := ⟨2, 0⟩
  let b : Point := ⟨0, 3⟩
  ∀ c : Point,
    triangleArea a b c = 6 ∧ isOnAxis c →
    c = ⟨0, 9⟩ ∨ c = ⟨0, -3⟩ ∨ c = ⟨-2, 0⟩ ∨ c = ⟨6, 0⟩ :=
by sorry

end NUMINAMATH_CALUDE_triangle_ABC_coordinates_l14_1481


namespace NUMINAMATH_CALUDE_isabellas_hair_length_l14_1445

/-- Isabella's hair length problem -/
theorem isabellas_hair_length :
  ∀ (current_length future_length growth : ℕ),
  future_length = 22 →
  growth = 4 →
  future_length = current_length + growth →
  current_length = 18 := by
  sorry

end NUMINAMATH_CALUDE_isabellas_hair_length_l14_1445


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l14_1437

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 3*x - 28 ≤ 0}
def N : Set ℝ := {x | x^2 - x - 2 > 0}

-- State the theorem
theorem intersection_of_M_and_N : 
  M ∩ N = {x : ℝ | (-4 ≤ x ∧ x < -1) ∨ (2 < x ∧ x ≤ 7)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l14_1437


namespace NUMINAMATH_CALUDE_coplanar_condition_l14_1477

open Real

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define the origin and points
variable (O A B C D : V)

-- Define the condition for coplanarity
def are_coplanar (A B C D : V) : Prop :=
  ∃ (a b c d : ℝ), a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∨ d ≠ 0 ∧ 
  a • (A - O) + b • (B - O) + c • (C - O) + d • (D - O) = 0

-- State the theorem
theorem coplanar_condition (k' : ℝ) :
  (4 • (A - O) - 3 • (B - O) + 6 • (C - O) + k' • (D - O) = 0) →
  (are_coplanar O A B C D ↔ k' = -7) := by
  sorry

end NUMINAMATH_CALUDE_coplanar_condition_l14_1477


namespace NUMINAMATH_CALUDE_no_solution_fractional_equation_l14_1444

theorem no_solution_fractional_equation :
  ∀ x : ℝ, x ≠ 2 → ¬ (1 / (x - 2) = (1 - x) / (2 - x) - 3) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_fractional_equation_l14_1444


namespace NUMINAMATH_CALUDE_square_of_99_l14_1404

theorem square_of_99 : 99 * 99 = 9801 := by
  sorry

end NUMINAMATH_CALUDE_square_of_99_l14_1404


namespace NUMINAMATH_CALUDE_cat_ratio_l14_1495

theorem cat_ratio (melanie_cats jacob_cats : ℕ) 
  (melanie_twice_annie : melanie_cats = 2 * (melanie_cats / 2))
  (jacob_has_90 : jacob_cats = 90)
  (melanie_has_60 : melanie_cats = 60) :
  (melanie_cats / 2) / jacob_cats = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cat_ratio_l14_1495


namespace NUMINAMATH_CALUDE_new_students_average_age_l14_1421

/-- Proves that the average age of new students is 32 years given the conditions of the problem -/
theorem new_students_average_age
  (original_average : ℝ)
  (original_strength : ℕ)
  (new_students : ℕ)
  (average_decrease : ℝ)
  (h1 : original_average = 40)
  (h2 : original_strength = 15)
  (h3 : new_students = 15)
  (h4 : average_decrease = 4) :
  let new_average := original_average - average_decrease
  let total_original := original_strength * original_average
  let total_new := (original_strength + new_students) * new_average - total_original
  total_new / new_students = 32 := by
  sorry

#check new_students_average_age

end NUMINAMATH_CALUDE_new_students_average_age_l14_1421


namespace NUMINAMATH_CALUDE_top_field_is_nine_l14_1467

/-- Represents a labelling of the figure -/
def Labelling := Fin 9 → Fin 9

/-- Check if a labelling is valid -/
def is_valid (l : Labelling) : Prop :=
  let s := l 0 + l 1 + l 2 -- sum of top row
  (l 0 + l 1 + l 2 = s) ∧
  (l 3 + l 4 + l 5 = s) ∧
  (l 6 + l 7 + l 8 = s) ∧
  (l 0 + l 3 + l 6 = s) ∧
  (l 1 + l 4 + l 7 = s) ∧
  (l 2 + l 5 + l 8 = s) ∧
  (l 0 + l 4 + l 8 = s) ∧
  (l 2 + l 4 + l 6 = s) ∧
  Function.Injective l

theorem top_field_is_nine (l : Labelling) (h : is_valid l) : l 0 = 9 := by
  sorry

#check top_field_is_nine

end NUMINAMATH_CALUDE_top_field_is_nine_l14_1467


namespace NUMINAMATH_CALUDE_min_sum_with_gcd_and_divisibility_l14_1438

theorem min_sum_with_gcd_and_divisibility (a b : ℕ+) :
  (Nat.gcd a b = 2015) →
  ((a + b) ∣ ((a - b)^2016 + b^2016)) →
  (∀ c d : ℕ+, (Nat.gcd c d = 2015) → ((c + d) ∣ ((c - d)^2016 + d^2016)) → (a + b ≤ c + d)) →
  a + b = 10075 := by
sorry

end NUMINAMATH_CALUDE_min_sum_with_gcd_and_divisibility_l14_1438


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l14_1482

def M : Set ℤ := {0}
def N : Set ℤ := {x | -1 < x ∧ x < 1}

theorem intersection_of_M_and_N : M ∩ N = {0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l14_1482


namespace NUMINAMATH_CALUDE_power_inequality_l14_1466

theorem power_inequality (a b : ℕ) (ha : a ≥ 2) (hb : b ≥ 3) :
  a^b + 1 ≥ b * (a + 1) := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l14_1466


namespace NUMINAMATH_CALUDE_sin_alpha_value_l14_1455

theorem sin_alpha_value (α : Real) (h1 : π/2 < α ∧ α < π) (h2 : 3 * Real.sin (2 * α) = Real.cos α) : 
  Real.sin α = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l14_1455


namespace NUMINAMATH_CALUDE_angle_bisection_limit_l14_1452

/-- The limit of an alternating series of angle bisections in a 60° angle -/
theorem angle_bisection_limit (θ : Real) (h : θ = 60) : 
  (∑' n, (-1)^n * (1/2)^(n+1)) * θ = 20 := by
  sorry

end NUMINAMATH_CALUDE_angle_bisection_limit_l14_1452


namespace NUMINAMATH_CALUDE_students_above_120_l14_1461

/-- Represents the probability density function of a normal distribution -/
noncomputable def normal_pdf (μ σ : ℝ) (x : ℝ) : ℝ := sorry

/-- Represents the cumulative distribution function of a normal distribution -/
noncomputable def normal_cdf (μ σ : ℝ) (x : ℝ) : ℝ := sorry

/-- The math scores follow a normal distribution with mean 110 and some standard deviation σ -/
axiom score_distribution (σ : ℝ) (x : ℝ) : 
  normal_pdf 110 σ x = normal_pdf 110 σ x

/-- The probability of scoring between 100 and 110 is 0.2 -/
axiom prob_100_to_110 (σ : ℝ) : 
  normal_cdf 110 σ 110 - normal_cdf 110 σ 100 = 0.2

/-- The total number of students is 800 -/
def total_students : ℕ := 800

/-- Theorem: Given the conditions, 240 students will score above 120 -/
theorem students_above_120 (σ : ℝ) : 
  (1 - normal_cdf 110 σ 120) * total_students = 240 := by sorry

end NUMINAMATH_CALUDE_students_above_120_l14_1461


namespace NUMINAMATH_CALUDE_largest_x_satisfying_equation_l14_1409

theorem largest_x_satisfying_equation : 
  ∃ (x : ℚ), x = 3/25 ∧ 
  (∀ y : ℚ, y ≥ 0 → Real.sqrt (3 * y) = 5 * y → y ≤ x) ∧
  Real.sqrt (3 * x) = 5 * x := by
  sorry

end NUMINAMATH_CALUDE_largest_x_satisfying_equation_l14_1409


namespace NUMINAMATH_CALUDE_project_scores_analysis_l14_1428

def scores : List ℝ := [8, 10, 9, 7, 7, 9, 8, 9]

def mode (l : List ℝ) : ℝ := sorry

def median (l : List ℝ) : ℝ := sorry

def range (l : List ℝ) : ℝ := sorry

def mean (l : List ℝ) : ℝ := sorry

theorem project_scores_analysis :
  mode scores = 9 ∧
  median scores = 8.5 ∧
  range scores = 3 ∧
  mean scores ≠ 8.4 := by sorry

end NUMINAMATH_CALUDE_project_scores_analysis_l14_1428


namespace NUMINAMATH_CALUDE_diagonal_length_in_special_quadrilateral_l14_1492

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a quadrilateral ABCD with diagonals intersecting at E -/
structure Quadrilateral :=
  (A B C D E : Point)

/-- The length of a line segment between two points -/
def distance (p q : Point) : ℝ := sorry

/-- The area of a triangle given three points -/
def triangleArea (p q r : Point) : ℝ := sorry

/-- Main theorem -/
theorem diagonal_length_in_special_quadrilateral 
  (ABCD : Quadrilateral) 
  (h1 : distance ABCD.A ABCD.B = 10)
  (h2 : distance ABCD.C ABCD.D = 15)
  (h3 : distance ABCD.A ABCD.C = 18)
  (h4 : triangleArea ABCD.A ABCD.E ABCD.D = triangleArea ABCD.B ABCD.E ABCD.C) :
  distance ABCD.A ABCD.E = 7.2 := by sorry

end NUMINAMATH_CALUDE_diagonal_length_in_special_quadrilateral_l14_1492


namespace NUMINAMATH_CALUDE_oliver_vowel_learning_time_l14_1497

/-- The number of days Oliver takes to learn one alphabet -/
def days_per_alphabet : ℕ := 5

/-- The number of vowels in the English alphabet -/
def number_of_vowels : ℕ := 5

/-- The total number of days Oliver needs to finish learning all vowels -/
def total_days : ℕ := days_per_alphabet * number_of_vowels

theorem oliver_vowel_learning_time : total_days = 25 := by
  sorry

end NUMINAMATH_CALUDE_oliver_vowel_learning_time_l14_1497


namespace NUMINAMATH_CALUDE_right_triangle_parity_l14_1432

theorem right_triangle_parity (a b c : ℕ) (h_right : a^2 + b^2 = c^2) :
  (Even a ∧ Even b ∧ Even c) ∨
  (Even a ∧ Odd b ∧ Odd c) ∨
  (Odd a ∧ Even b ∧ Odd c) ∨
  (Odd a ∧ Odd b ∧ Even c) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_parity_l14_1432


namespace NUMINAMATH_CALUDE_chess_game_probability_l14_1408

theorem chess_game_probability (prob_A_win prob_A_not_lose : ℝ) 
  (h1 : prob_A_win = 0.4)
  (h2 : prob_A_not_lose = 0.8) : 
  1 - prob_A_not_lose = 0.6 :=
by
  sorry


end NUMINAMATH_CALUDE_chess_game_probability_l14_1408


namespace NUMINAMATH_CALUDE_exponent_multiplication_l14_1415

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l14_1415


namespace NUMINAMATH_CALUDE_binomial_20_choose_6_l14_1450

theorem binomial_20_choose_6 : Nat.choose 20 6 = 19380 := by sorry

end NUMINAMATH_CALUDE_binomial_20_choose_6_l14_1450


namespace NUMINAMATH_CALUDE_quadratic_single_solution_l14_1485

theorem quadratic_single_solution (b : ℝ) (hb : b ≠ 0) :
  (∃! x, 3 * x^2 + b * x + 10 = 0) →
  (∃ x, 3 * x^2 + b * x + 10 = 0 ∧ x = -Real.sqrt 30 / 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_single_solution_l14_1485


namespace NUMINAMATH_CALUDE_ellipse_chord_slope_ellipse_chord_slope_at_4_2_l14_1471

/-- The slope of a chord in an ellipse given its midpoint -/
theorem ellipse_chord_slope (x₁ y₁ x₂ y₂ : ℝ) : 
  (x₁^2 / 36 + y₁^2 / 9 = 1) →  -- Point (x₁, y₁) is on the ellipse
  (x₂^2 / 36 + y₂^2 / 9 = 1) →  -- Point (x₂, y₂) is on the ellipse
  ((x₁ + x₂) / 2 = 4) →         -- Midpoint x-coordinate is 4
  ((y₁ + y₂) / 2 = 2) →         -- Midpoint y-coordinate is 2
  (y₂ - y₁) / (x₂ - x₁) = -1/2  -- Slope of the chord
:= by sorry

/-- The main theorem stating the slope of the chord with midpoint (4, 2) -/
theorem ellipse_chord_slope_at_4_2 : 
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁^2 / 36 + y₁^2 / 9 = 1) ∧ 
    (x₂^2 / 36 + y₂^2 / 9 = 1) ∧ 
    ((x₁ + x₂) / 2 = 4) ∧ 
    ((y₁ + y₂) / 2 = 2) ∧ 
    (y₂ - y₁) / (x₂ - x₁) = -1/2
:= by sorry

end NUMINAMATH_CALUDE_ellipse_chord_slope_ellipse_chord_slope_at_4_2_l14_1471


namespace NUMINAMATH_CALUDE_lawsuit_probability_comparison_l14_1474

def probability_lawsuit1_win : ℝ := 0.3
def probability_lawsuit2_win : ℝ := 0.5
def probability_lawsuit3_win : ℝ := 0.4

def probability_lawsuit1_lose : ℝ := 1 - probability_lawsuit1_win
def probability_lawsuit2_lose : ℝ := 1 - probability_lawsuit2_win
def probability_lawsuit3_lose : ℝ := 1 - probability_lawsuit3_win

def probability_win_all : ℝ := probability_lawsuit1_win * probability_lawsuit2_win * probability_lawsuit3_win
def probability_lose_all : ℝ := probability_lawsuit1_lose * probability_lawsuit2_lose * probability_lawsuit3_lose

theorem lawsuit_probability_comparison :
  (probability_lose_all - probability_win_all) / probability_win_all * 100 = 250 := by
sorry

end NUMINAMATH_CALUDE_lawsuit_probability_comparison_l14_1474


namespace NUMINAMATH_CALUDE_orchids_in_vase_l14_1442

/-- Represents the number of roses initially in the vase -/
def initial_roses : ℕ := 9

/-- Represents the number of orchids initially in the vase -/
def initial_orchids : ℕ := 6

/-- Represents the number of roses in the vase now -/
def current_roses : ℕ := 3

/-- Represents the difference between the number of orchids and roses in the vase now -/
def orchid_rose_difference : ℕ := 10

/-- Represents the number of orchids in the vase now -/
def current_orchids : ℕ := current_roses + orchid_rose_difference

theorem orchids_in_vase : current_orchids = 13 := by sorry

end NUMINAMATH_CALUDE_orchids_in_vase_l14_1442


namespace NUMINAMATH_CALUDE_weight_difference_l14_1406

/-- Given Mildred weighs 59 pounds and Carol weighs 9 pounds, 
    prove that Mildred is 50 pounds heavier than Carol. -/
theorem weight_difference (mildred_weight carol_weight : ℕ) 
  (h1 : mildred_weight = 59) 
  (h2 : carol_weight = 9) : 
  mildred_weight - carol_weight = 50 := by
  sorry

end NUMINAMATH_CALUDE_weight_difference_l14_1406


namespace NUMINAMATH_CALUDE_race_probability_l14_1469

theorem race_probability (total_cars : ℕ) (prob_Y prob_Z prob_XYZ : ℝ) : 
  total_cars = 18 → 
  prob_Y = 1/12 → 
  prob_Z = 1/6 → 
  prob_XYZ = 0.375 → 
  ∃ prob_X : ℝ, 
    prob_X + prob_Y + prob_Z = prob_XYZ ∧ 
    prob_X = 0.125 := by
  sorry

end NUMINAMATH_CALUDE_race_probability_l14_1469


namespace NUMINAMATH_CALUDE_man_pants_count_l14_1422

theorem man_pants_count (t_shirts : ℕ) (total_ways : ℕ) (pants : ℕ) : 
  t_shirts = 8 → 
  total_ways = 72 → 
  total_ways = t_shirts * pants → 
  pants = 9 := by sorry

end NUMINAMATH_CALUDE_man_pants_count_l14_1422


namespace NUMINAMATH_CALUDE_homothetic_image_containment_l14_1468

-- Define a convex polygon
def ConvexPolygon (P : Set (Point)) : Prop := sorry

-- Define a homothetic transformation
def HomotheticTransformation (center : Point) (k : ℝ) (P : Set Point) : Set Point := sorry

-- Define that a set is contained within another set
def IsContainedIn (A B : Set Point) : Prop := sorry

-- The theorem statement
theorem homothetic_image_containment 
  (P : Set Point) (h : ConvexPolygon P) :
  ∃ (center : Point), 
    IsContainedIn (HomotheticTransformation center (1/2) P) P := by
  sorry

end NUMINAMATH_CALUDE_homothetic_image_containment_l14_1468


namespace NUMINAMATH_CALUDE_mechanic_rate_is_75_l14_1436

/-- Calculates the mechanic's hourly rate given the total work time, part cost, and total amount paid -/
def mechanicHourlyRate (workTime : ℕ) (partCost : ℕ) (totalPaid : ℕ) : ℕ :=
  (totalPaid - partCost) / workTime

/-- Proves that the mechanic's hourly rate is $75 given the problem conditions -/
theorem mechanic_rate_is_75 :
  mechanicHourlyRate 2 150 300 = 75 := by
  sorry

end NUMINAMATH_CALUDE_mechanic_rate_is_75_l14_1436


namespace NUMINAMATH_CALUDE_mutually_exclusive_not_contradictory_l14_1412

/-- Represents the total number of products -/
def total_products : ℕ := 5

/-- Represents the number of genuine products -/
def genuine_products : ℕ := 3

/-- Represents the number of defective products -/
def defective_products : ℕ := 2

/-- Represents the number of products selected -/
def selected_products : ℕ := 2

/-- Represents the event of selecting exactly one defective product -/
def event_one_defective (selected : ℕ) : Prop :=
  selected = 1

/-- Represents the event of selecting exactly two genuine products -/
def event_two_genuine (selected : ℕ) : Prop :=
  selected = 2

/-- Theorem stating that the events are mutually exclusive and not contradictory -/
theorem mutually_exclusive_not_contradictory :
  (¬ (event_one_defective selected_products ∧ event_two_genuine selected_products)) ∧
  (∃ (x : ℕ), ¬ (event_one_defective x ∨ event_two_genuine x)) :=
sorry

end NUMINAMATH_CALUDE_mutually_exclusive_not_contradictory_l14_1412


namespace NUMINAMATH_CALUDE_mens_tshirt_interval_l14_1478

/-- Represents the shop selling T-shirts -/
structure TShirtShop where
  womens_interval : ℕ  -- Minutes between women's T-shirt sales
  womens_price : ℕ     -- Price of women's T-shirts in dollars
  mens_price : ℕ        -- Price of men's T-shirts in dollars
  daily_hours : ℕ      -- Hours open per day
  weekly_days : ℕ      -- Days open per week
  weekly_revenue : ℕ   -- Total weekly revenue in dollars

/-- Calculates the interval between men's T-shirt sales -/
def mens_interval (shop : TShirtShop) : ℕ :=
  sorry

/-- Theorem stating that the men's T-shirt sale interval is 40 minutes -/
theorem mens_tshirt_interval (shop : TShirtShop) 
  (h1 : shop.womens_interval = 30)
  (h2 : shop.womens_price = 18)
  (h3 : shop.mens_price = 15)
  (h4 : shop.daily_hours = 12)
  (h5 : shop.weekly_days = 7)
  (h6 : shop.weekly_revenue = 4914) :
  mens_interval shop = 40 := by
    sorry

end NUMINAMATH_CALUDE_mens_tshirt_interval_l14_1478


namespace NUMINAMATH_CALUDE_min_cost_container_l14_1401

/-- Represents the dimensions and costs of a rectangular container. -/
structure Container where
  volume : ℝ
  height : ℝ
  baseCost : ℝ
  sideCost : ℝ

/-- Calculates the total cost of constructing the container. -/
def totalCost (c : Container) (length width : ℝ) : ℝ :=
  c.baseCost * length * width + c.sideCost * 2 * (length + width) * c.height

/-- Theorem stating that the minimum cost to construct the given container is 1600 yuan. -/
theorem min_cost_container (c : Container) 
  (h_volume : c.volume = 4)
  (h_height : c.height = 1)
  (h_baseCost : c.baseCost = 200)
  (h_sideCost : c.sideCost = 100) :
  ∃ (cost : ℝ), cost = 1600 ∧ ∀ (length width : ℝ), length * width * c.height = c.volume → 
    totalCost c length width ≥ cost := by
  sorry

#check min_cost_container

end NUMINAMATH_CALUDE_min_cost_container_l14_1401


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l14_1431

theorem decimal_to_fraction (x : ℚ) : x = 0.38 → x = 19/50 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l14_1431


namespace NUMINAMATH_CALUDE_triangle_longest_side_l14_1472

/-- Given a triangle with sides of lengths 7, x+4, and 2x+1, and a perimeter of 36,
    prove that the length of the longest side is 17. -/
theorem triangle_longest_side (x : ℝ) : 
  (7 : ℝ) + (x + 4) + (2*x + 1) = 36 → 
  max 7 (max (x + 4) (2*x + 1)) = 17 :=
by sorry

end NUMINAMATH_CALUDE_triangle_longest_side_l14_1472


namespace NUMINAMATH_CALUDE_water_cube_product_l14_1465

/-- Definition of a water cube number -/
def V (a b c : ℝ) : ℝ := a^3 + b^3 + c^3 - 3*a*b*c

/-- Theorem: The product of two water cube numbers is a water cube number -/
theorem water_cube_product (a b c x y z : ℝ) :
  V a b c * V x y z = V (a*x + b*y + c*z) (b*x + c*y + a*z) (c*x + a*y + b*z) := by
  sorry

end NUMINAMATH_CALUDE_water_cube_product_l14_1465


namespace NUMINAMATH_CALUDE_perpendicular_lines_parallel_l14_1464

-- Define the types for lines and planes in 3D space
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_parallel (m n : Line) (α : Plane) :
  perp m α → perp n α → para m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_parallel_l14_1464


namespace NUMINAMATH_CALUDE_divisor_problem_l14_1420

theorem divisor_problem (n : ℕ) (h : n = 13294) : 
  ∃ (d : ℕ), d > 1 ∧ (n - 5) % d = 0 ∧ d = 13289 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l14_1420


namespace NUMINAMATH_CALUDE_polynomial_simplification_l14_1424

theorem polynomial_simplification (x : ℝ) : 
  (3 * x^2 + 4 * x + 6) * (x - 2) - (x - 2) * (2 * x^2 + 5 * x - 72) + (2 * x - 15) * (x - 2) * (x + 4) = 
  3 * x^3 - 14 * x^2 + 34 * x - 36 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l14_1424


namespace NUMINAMATH_CALUDE_range_of_f_l14_1440

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 + 4 * x - 5

-- Define the domain of x
def domain : Set ℝ := { x | -3 ≤ x ∧ x < 2 }

-- State the theorem
theorem range_of_f :
  ∃ (y_min y_max : ℝ), y_min = -7 ∧ y_max = 11 ∧
  ∀ y, (∃ x ∈ domain, f x = y) ↔ y_min ≤ y ∧ y < y_max :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l14_1440


namespace NUMINAMATH_CALUDE_tooth_permutations_l14_1443

def word_length : ℕ := 5
def t_occurrences : ℕ := 3
def o_occurrences : ℕ := 2

theorem tooth_permutations : 
  (word_length.factorial) / (t_occurrences.factorial * o_occurrences.factorial) = 10 := by
  sorry

end NUMINAMATH_CALUDE_tooth_permutations_l14_1443


namespace NUMINAMATH_CALUDE_oliver_card_arrangement_l14_1411

/-- Calculates the minimum number of pages required to arrange Oliver's baseball cards --/
def min_pages_for_cards : ℕ :=
  let cards_per_page : ℕ := 3
  let new_cards : ℕ := 2
  let old_cards : ℕ := 10
  let rare_cards : ℕ := 3
  let pages_for_new_cards : ℕ := 1
  let pages_for_rare_cards : ℕ := 1
  let remaining_old_cards : ℕ := old_cards - rare_cards
  let pages_for_remaining_old_cards : ℕ := (remaining_old_cards + cards_per_page - 1) / cards_per_page

  pages_for_new_cards + pages_for_rare_cards + pages_for_remaining_old_cards

theorem oliver_card_arrangement :
  min_pages_for_cards = 5 := by
  sorry

end NUMINAMATH_CALUDE_oliver_card_arrangement_l14_1411


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l14_1407

/-- Two real numbers vary inversely if their product is constant. -/
def VaryInversely (p q : ℝ → ℝ) :=
  ∃ k : ℝ, ∀ x : ℝ, p x * q x = k

theorem inverse_variation_problem (p q : ℝ → ℝ) 
    (h_inverse : VaryInversely p q)
    (h_initial : p 1 = 800 ∧ q 1 = 0.5) :
    p 2 = 1600 → q 2 = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l14_1407


namespace NUMINAMATH_CALUDE_smallest_positive_integer_3003m_55555n_l14_1488

theorem smallest_positive_integer_3003m_55555n :
  ∃ (k : ℕ), k > 0 ∧ (∃ (m n : ℤ), k = 3003 * m + 55555 * n) ∧
  ∀ (j : ℕ), j > 0 → (∃ (x y : ℤ), j = 3003 * x + 55555 * y) → k ≤ j :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_3003m_55555n_l14_1488


namespace NUMINAMATH_CALUDE_local_value_in_product_l14_1460

/-- The face value of a digit is the digit itself -/
def faceValue (d : ℕ) : ℕ := d

/-- The local value of a digit in a number is the digit multiplied by its place value -/
def localValue (d : ℕ) (placeValue : ℕ) : ℕ := d * placeValue

/-- The product of two numbers -/
def product (a b : ℕ) : ℕ := a * b

/-- Theorem: In the product of the face value of 7 and the local value of 6 in 7098060,
    the local value of 6 is 6000 -/
theorem local_value_in_product :
  let number : ℕ := 7098060
  let fv7 : ℕ := faceValue 7
  let lv6 : ℕ := localValue 6 1000
  let prod : ℕ := product fv7 lv6
  localValue 6 1000 = 6000 := by sorry

end NUMINAMATH_CALUDE_local_value_in_product_l14_1460


namespace NUMINAMATH_CALUDE_division_equals_500_l14_1462

theorem division_equals_500 : (35 : ℝ) / 0.07 = 500 := by
  sorry

end NUMINAMATH_CALUDE_division_equals_500_l14_1462


namespace NUMINAMATH_CALUDE_team_a_prefers_best_of_five_l14_1425

/-- Represents the probability of Team A winning a non-deciding game -/
def team_a_win_prob : ℝ := 0.6

/-- Represents the probability of Team B winning a non-deciding game -/
def team_b_win_prob : ℝ := 0.4

/-- Represents the probability of either team winning a deciding game -/
def deciding_game_win_prob : ℝ := 0.5

/-- Calculates the probability of Team A winning a best-of-three series -/
def best_of_three_win_prob : ℝ := 
  team_a_win_prob^2 + 2 * team_a_win_prob * team_b_win_prob * deciding_game_win_prob

/-- Calculates the probability of Team A winning a best-of-five series -/
def best_of_five_win_prob : ℝ := 
  team_a_win_prob^3 + 
  3 * team_a_win_prob^2 * team_b_win_prob + 
  6 * team_a_win_prob^2 * team_b_win_prob^2 * deciding_game_win_prob

/-- Theorem stating that Team A has a higher probability of winning in a best-of-five series -/
theorem team_a_prefers_best_of_five : best_of_five_win_prob > best_of_three_win_prob := by
  sorry

end NUMINAMATH_CALUDE_team_a_prefers_best_of_five_l14_1425


namespace NUMINAMATH_CALUDE_dartboard_central_angle_l14_1402

theorem dartboard_central_angle (probability : ℝ) (central_angle : ℝ) : 
  probability = 1 / 8 → central_angle = 45 := by
  sorry

end NUMINAMATH_CALUDE_dartboard_central_angle_l14_1402


namespace NUMINAMATH_CALUDE_range_of_a_l14_1449

theorem range_of_a (x : ℝ) (a : ℝ) : 
  x ∈ Set.Ioo 0 π → 
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 2 * Real.sin (x₁ + π/3) = a ∧ 2 * Real.sin (x₂ + π/3) = a) → 
  a ∈ Set.Ioo (Real.sqrt 3) 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l14_1449


namespace NUMINAMATH_CALUDE_coin_ratio_is_one_one_one_l14_1470

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


end NUMINAMATH_CALUDE_coin_ratio_is_one_one_one_l14_1470


namespace NUMINAMATH_CALUDE_abs_two_minus_sqrt_five_l14_1419

theorem abs_two_minus_sqrt_five : 
  |2 - Real.sqrt 5| = Real.sqrt 5 - 2 := by sorry

end NUMINAMATH_CALUDE_abs_two_minus_sqrt_five_l14_1419


namespace NUMINAMATH_CALUDE_power_function_through_point_l14_1417

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ a

-- Theorem statement
theorem power_function_through_point (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : f 2 = Real.sqrt 2) : 
  f 27 = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_point_l14_1417


namespace NUMINAMATH_CALUDE_smallest_n_for_factorization_l14_1489

theorem smallest_n_for_factorization : 
  ∀ n : ℤ, 
  (∃ A B : ℤ, ∀ x : ℝ, 3 * x^2 + n * x + 72 = (3 * x + A) * (x + B)) → 
  n ≥ 35 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_factorization_l14_1489


namespace NUMINAMATH_CALUDE_new_person_age_l14_1491

/-- Given a group of 10 persons where replacing a 45-year-old person with a new person
    decreases the average age by 3 years, the age of the new person is 15 years. -/
theorem new_person_age (initial_avg : ℝ) : 
  (10 * initial_avg - 45 + 15) / 10 = initial_avg - 3 := by
  sorry

#check new_person_age

end NUMINAMATH_CALUDE_new_person_age_l14_1491


namespace NUMINAMATH_CALUDE_shells_added_calculation_l14_1416

/-- Calculates the amount of shells added given initial weight, percentage increase, and final weight -/
def shells_added (initial_weight : ℝ) (percent_increase : ℝ) (final_weight : ℝ) : ℝ :=
  final_weight - initial_weight

/-- Theorem stating that given the problem conditions, the amount of shells added is 23 pounds -/
theorem shells_added_calculation (initial_weight : ℝ) (percent_increase : ℝ) (final_weight : ℝ)
  (h1 : initial_weight = 5)
  (h2 : percent_increase = 150)
  (h3 : final_weight = 28) :
  shells_added initial_weight percent_increase final_weight = 23 := by
  sorry

#eval shells_added 5 150 28

end NUMINAMATH_CALUDE_shells_added_calculation_l14_1416


namespace NUMINAMATH_CALUDE_students_play_both_football_and_cricket_l14_1493

/-- Represents the number of students who play both football and cricket -/
def students_play_both (total students_football students_cricket students_neither : ℕ) : ℕ :=
  students_football + students_cricket - (total - students_neither)

/-- Theorem stating that given the conditions, 140 students play both football and cricket -/
theorem students_play_both_football_and_cricket :
  students_play_both 410 325 175 50 = 140 := by
  sorry

end NUMINAMATH_CALUDE_students_play_both_football_and_cricket_l14_1493


namespace NUMINAMATH_CALUDE_sequence_sum_l14_1458

/-- Given a sequence {a_n} where a_1 = 1 and S_n = n^2 * a_n for all positive integers n,
    prove that the sum of the first n terms (S_n) is equal to 2n / (n+1). -/
theorem sequence_sum (a : ℕ+ → ℝ) (S : ℕ+ → ℝ) :
  a 1 = 1 →
  (∀ n : ℕ+, S n = n^2 * a n) →
  ∀ n : ℕ+, S n = (2 * n : ℝ) / (n + 1) :=
sorry

end NUMINAMATH_CALUDE_sequence_sum_l14_1458


namespace NUMINAMATH_CALUDE_abs_inequality_solution_l14_1486

-- Define the solution set for |x+1| < 5
def solution_set : Set ℝ := {x : ℝ | |x + 1| < 5}

-- Define the open interval (-6, 4)
def open_interval : Set ℝ := Set.Ioo (-6) 4

-- Theorem stating that the solution set is equal to the open interval
theorem abs_inequality_solution : solution_set = open_interval := by sorry

end NUMINAMATH_CALUDE_abs_inequality_solution_l14_1486


namespace NUMINAMATH_CALUDE_negative_m_exponent_division_l14_1429

theorem negative_m_exponent_division (m : ℝ) : (-m)^6 / (-m)^3 = -m^3 := by sorry

end NUMINAMATH_CALUDE_negative_m_exponent_division_l14_1429


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l14_1403

theorem binomial_coefficient_equality (m : ℕ) : 
  (Nat.choose 17 (3*m - 1) = Nat.choose 17 (2*m + 3)) → (m = 3 ∨ m = 4) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l14_1403


namespace NUMINAMATH_CALUDE_average_price_of_cow_l14_1430

/-- Given the total price for 2 cows and 8 goats, and the average price of a goat,
    prove that the average price of a cow is 460 rupees. -/
theorem average_price_of_cow (total_price : ℕ) (goat_price : ℕ) (cow_count : ℕ) (goat_count : ℕ) :
  total_price = 1400 →
  goat_price = 60 →
  cow_count = 2 →
  goat_count = 8 →
  (total_price - goat_count * goat_price) / cow_count = 460 := by
  sorry

end NUMINAMATH_CALUDE_average_price_of_cow_l14_1430


namespace NUMINAMATH_CALUDE_n_range_theorem_l14_1434

theorem n_range_theorem (x y m n : ℝ) :
  n ≤ x ∧ x < y ∧ y ≤ n + 1 ∧
  m ∈ Set.Ioo x y ∧
  |y| = |m| + |x| →
  -1 < n ∧ n < 1 :=
by sorry

end NUMINAMATH_CALUDE_n_range_theorem_l14_1434


namespace NUMINAMATH_CALUDE_fuel_tank_capacity_l14_1400

/-- Proves that the capacity of a fuel tank is 212 gallons given specific conditions about fuel composition and volume. -/
theorem fuel_tank_capacity : ∃ (C : ℝ), 
  (0.12 * 98 + 0.16 * (C - 98) = 30) ∧ 
  C = 212 := by
  sorry

end NUMINAMATH_CALUDE_fuel_tank_capacity_l14_1400


namespace NUMINAMATH_CALUDE_inscribed_square_area_l14_1454

/-- The area of a square inscribed in a circular segment with an arc of 60° and radius 2√3 + √17 is 1. -/
theorem inscribed_square_area (R : ℝ) (h : R = 2 * Real.sqrt 3 + Real.sqrt 17) : 
  let segment_arc : ℝ := 60 * π / 180
  let square_side : ℝ := (R * (Real.sqrt 17 - 2 * Real.sqrt 3)) / 5
  square_side ^ 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l14_1454


namespace NUMINAMATH_CALUDE_F_range_l14_1457

noncomputable def F (x : ℝ) : ℝ := |2 * x + 4| - |x - 2|

theorem F_range :
  Set.range F = Set.Ici (-4) :=
sorry

end NUMINAMATH_CALUDE_F_range_l14_1457


namespace NUMINAMATH_CALUDE_cylinder_max_volume_ratio_l14_1487

/-- The ratio of height to base radius of a cylinder with surface area 6π when its volume is maximized -/
theorem cylinder_max_volume_ratio : 
  ∃ (h r : ℝ), 
    h > 0 ∧ r > 0 ∧  -- Ensure positive height and radius
    2 * π * r^2 + 2 * π * r * h = 6 * π ∧  -- Surface area condition
    (∀ (h' r' : ℝ), 
      h' > 0 ∧ r' > 0 ∧ 
      2 * π * r'^2 + 2 * π * r' * h' = 6 * π → 
      π * r^2 * h ≥ π * r'^2 * h') →  -- Volume maximization condition
    h / r = 2 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_max_volume_ratio_l14_1487


namespace NUMINAMATH_CALUDE_cycle_price_proof_l14_1447

theorem cycle_price_proof (selling_price : ℝ) (gain_percent : ℝ) (original_price : ℝ) : 
  selling_price = 1620 → 
  gain_percent = 8 → 
  selling_price = original_price * (1 + gain_percent / 100) → 
  original_price = 1500 := by
  sorry

#check cycle_price_proof

end NUMINAMATH_CALUDE_cycle_price_proof_l14_1447


namespace NUMINAMATH_CALUDE_identify_genuine_coins_l14_1441

/-- Represents the result of a weighing -/
inductive WeighResult
  | Equal : WeighResult
  | Unequal : WeighResult

/-- Represents a set of coins -/
structure CoinSet where
  total : Nat
  fake : Nat
  h_fake_count : fake ≤ 1
  h_total : total = 101

/-- Represents a weighing action -/
def weighing (left right : Nat) : WeighResult :=
  sorry

/-- The main theorem to prove -/
theorem identify_genuine_coins (coins : CoinSet) :
  ∃ (genuine : Nat), genuine ≥ 50 ∧
    ∀ (left right : Nat),
      left + right ≤ coins.total →
      (weighing left right = WeighResult.Equal →
        genuine = left + right) ∧
      (weighing left right = WeighResult.Unequal →
        genuine = coins.total - (left + right)) :=
  sorry

end NUMINAMATH_CALUDE_identify_genuine_coins_l14_1441


namespace NUMINAMATH_CALUDE_P_proper_subset_Q_l14_1448

def P : Set ℕ := {1, 2, 4}
def Q : Set ℕ := {1, 2, 4, 8}

theorem P_proper_subset_Q : P ⊂ Q := by sorry

end NUMINAMATH_CALUDE_P_proper_subset_Q_l14_1448


namespace NUMINAMATH_CALUDE_circle_tangent_to_x_axis_at_origin_l14_1427

theorem circle_tangent_to_x_axis_at_origin 
  (D E F : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + D*x + E*y + F = 0 → 
    (∃ r : ℝ, r > 0 ∧ 
      ∀ x y : ℝ, (x^2 + y^2 = r^2) ↔ (x^2 + y^2 + D*x + E*y + F = 0)) ∧
    (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x| < δ → 
      ∃ y : ℝ, |y| < ε ∧ x^2 + y^2 + D*x + E*y + F = 0) ∧
    (0^2 + 0^2 + D*0 + E*0 + F = 0)) →
  D = 0 ∧ F = 0 ∧ E ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_to_x_axis_at_origin_l14_1427


namespace NUMINAMATH_CALUDE_discriminant_of_specific_quadratic_l14_1414

/-- The discriminant of a quadratic equation ax^2 + bx + c -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- Theorem: The discriminant of 5x^2 - 11x + 4 is 41 -/
theorem discriminant_of_specific_quadratic : discriminant 5 (-11) 4 = 41 := by
  sorry

end NUMINAMATH_CALUDE_discriminant_of_specific_quadratic_l14_1414


namespace NUMINAMATH_CALUDE_harrison_extra_pages_l14_1498

def minimum_pages : ℕ := 25
def sam_pages : ℕ := 100

def pam_pages (sam : ℕ) : ℕ := sam / 2

def harrison_pages (pam : ℕ) : ℕ := pam - 15

theorem harrison_extra_pages :
  harrison_pages (pam_pages sam_pages) - minimum_pages = 10 :=
by sorry

end NUMINAMATH_CALUDE_harrison_extra_pages_l14_1498


namespace NUMINAMATH_CALUDE_rectangle_breadth_l14_1480

theorem rectangle_breadth (area : ℝ) (length_ratio : ℝ) (breadth : ℝ) : 
  area = 460 →
  length_ratio = 1.15 →
  area = (length_ratio * breadth) * breadth →
  breadth = 20 := by
sorry

end NUMINAMATH_CALUDE_rectangle_breadth_l14_1480


namespace NUMINAMATH_CALUDE_mistaken_divisor_problem_l14_1446

theorem mistaken_divisor_problem (dividend : ℕ) (correct_divisor : ℕ) (correct_quotient : ℕ) (mistaken_quotient : ℕ) 
  (h1 : dividend = correct_divisor * correct_quotient)
  (h2 : correct_divisor = 21)
  (h3 : correct_quotient = 40)
  (h4 : mistaken_quotient = 70)
  (h5 : ∃ (mistaken_divisor : ℕ), dividend = mistaken_divisor * mistaken_quotient) :
  ∃ (mistaken_divisor : ℕ), mistaken_divisor = 12 ∧ dividend = mistaken_divisor * mistaken_quotient := by
  sorry

end NUMINAMATH_CALUDE_mistaken_divisor_problem_l14_1446


namespace NUMINAMATH_CALUDE_friday_temperature_l14_1413

/-- Given the average temperatures for two sets of four days and the temperature on Monday,
    prove that the temperature on Friday is 36 degrees. -/
theorem friday_temperature
  (avg_mon_to_thu : (mon + tue + wed + thu) / 4 = 48)
  (avg_tue_to_fri : (tue + wed + thu + fri) / 4 = 46)
  (monday_temp : mon = 44)
  : fri = 36 := by
  sorry

end NUMINAMATH_CALUDE_friday_temperature_l14_1413


namespace NUMINAMATH_CALUDE_circle_tangent_to_x_axis_l14_1410

-- Define the center of the circle
def center : ℝ × ℝ := (-3, 4)

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop :=
  (x + 3)^2 + (y - 4)^2 = 16

-- Theorem statement
theorem circle_tangent_to_x_axis :
  -- The circle has center at (-3, 4)
  ∃ (x y : ℝ), circle_equation x y ∧ (x, y) = center ∧
  -- The circle is tangent to the x-axis
  ∃ (x : ℝ), circle_equation x 0 ∧
  -- The equation represents a circle
  ∀ (p : ℝ × ℝ), p ∈ {p | circle_equation p.1 p.2} ↔ 
    (p.1 - center.1)^2 + (p.2 - center.2)^2 = 4^2 :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_x_axis_l14_1410


namespace NUMINAMATH_CALUDE_partnership_investment_ratio_l14_1405

/-- Given a partnership business where:
  * A's investment is k times B's investment
  * A's investment period is twice B's investment period
  * B's profit is 7000
  * Total profit is 49000
  Prove that the ratio of A's investment to B's investment is 3:1 -/
theorem partnership_investment_ratio 
  (k : ℚ) 
  (b_profit : ℚ) 
  (total_profit : ℚ) 
  (h1 : b_profit = 7000)
  (h2 : total_profit = 49000)
  (h3 : k * b_profit * 2 + b_profit = total_profit) : 
  k = 3 := by
sorry

end NUMINAMATH_CALUDE_partnership_investment_ratio_l14_1405


namespace NUMINAMATH_CALUDE_correct_prices_l14_1475

/-- Prices of items in a shopping scenario -/
def shopping_prices (total belt pants shirt shoes : ℝ) : Prop :=
  -- Total cost condition
  total = belt + pants + shirt + shoes ∧
  -- Pants price condition
  pants = belt - 2.93 ∧
  -- Shirt price condition
  shirt = 1.5 * pants ∧
  -- Shoes price condition
  shoes = 3 * shirt

/-- Theorem stating the correct prices for the shopping scenario -/
theorem correct_prices : 
  ∃ (belt pants shirt shoes : ℝ),
    shopping_prices 205.93 belt pants shirt shoes ∧ 
    belt = 28.305 ∧ 
    pants = 25.375 ∧ 
    shirt = 38.0625 ∧ 
    shoes = 114.1875 :=
by
  sorry

end NUMINAMATH_CALUDE_correct_prices_l14_1475


namespace NUMINAMATH_CALUDE_sum_of_roots_l14_1451

theorem sum_of_roots (p q r s : ℝ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (∀ x, x^2 - 12*p*x - 13*q = 0 ↔ x = r ∨ x = s) →
  (∀ x, x^2 - 12*r*x - 13*s = 0 ↔ x = p ∨ x = q) →
  p + q + r + s = 2028 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_l14_1451


namespace NUMINAMATH_CALUDE_conference_hall_tables_l14_1463

/-- Represents the setup of a conference hall --/
structure ConferenceHall where
  tables : ℕ
  chairs_per_table : ℕ
  chair_legs : ℕ
  table_legs : ℕ
  sofa_legs : ℕ
  total_legs : ℕ

/-- The conference hall setup satisfies the given conditions --/
def valid_setup (hall : ConferenceHall) : Prop :=
  hall.chairs_per_table = 8 ∧
  hall.chair_legs = 4 ∧
  hall.table_legs = 5 ∧
  hall.sofa_legs = 6 ∧
  hall.total_legs = 760

/-- The number of sofas is half the number of tables --/
def sofa_table_relation (hall : ConferenceHall) : Prop :=
  2 * (hall.tables / 2) = hall.tables

/-- The total number of legs is correctly calculated --/
def correct_leg_count (hall : ConferenceHall) : Prop :=
  hall.total_legs = 
    hall.chair_legs * (hall.chairs_per_table * hall.tables) +
    hall.table_legs * hall.tables +
    hall.sofa_legs * (hall.tables / 2)

/-- Theorem stating that given the conditions, there are 19 tables in the hall --/
theorem conference_hall_tables (hall : ConferenceHall) :
  valid_setup hall → sofa_table_relation hall → correct_leg_count hall → hall.tables = 19 := by
  sorry


end NUMINAMATH_CALUDE_conference_hall_tables_l14_1463


namespace NUMINAMATH_CALUDE_total_amount_distributed_l14_1484

/-- The total amount distributed when an amount of money is equally divided among a group of people. -/
def total_amount (num_people : ℕ) (amount_per_person : ℕ) : ℕ :=
  num_people * amount_per_person

/-- Theorem stating that the total amount distributed is 42900 when 22 people each receive 1950. -/
theorem total_amount_distributed : total_amount 22 1950 = 42900 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_distributed_l14_1484


namespace NUMINAMATH_CALUDE_union_when_k_neg_one_intersection_condition_l14_1456

-- Define sets A and B
def A : Set ℝ := {x | (x - 2) / (x + 1) < 0}
def B (k : ℝ) : Set ℝ := {x | k < x ∧ x < 2 - k}

-- Theorem 1: When k = -1, A ∪ B = (-1, 3)
theorem union_when_k_neg_one :
  A ∪ B (-1) = Set.Ioo (-1) 3 := by sorry

-- Theorem 2: A ∩ B = B if and only if k ∈ [0, +∞)
theorem intersection_condition (k : ℝ) :
  A ∩ B k = B k ↔ k ∈ Set.Ici 0 := by sorry

end NUMINAMATH_CALUDE_union_when_k_neg_one_intersection_condition_l14_1456


namespace NUMINAMATH_CALUDE_right_triangle_from_sine_condition_l14_1490

theorem right_triangle_from_sine_condition (A B C : Real) (h1 : 0 < A) (h2 : A < π/2) 
  (h3 : 0 < B) (h4 : B < π/2) (h5 : A + B + C = π) 
  (h6 : Real.sin A ^ 2 + Real.sin B ^ 2 = Real.sin (A + B)) : 
  C = π/2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_from_sine_condition_l14_1490


namespace NUMINAMATH_CALUDE_integer_count_in_sequence_l14_1494

def arithmeticSequence (n : ℕ) : ℚ :=
  8505 / (5 ^ n)

def isInteger (q : ℚ) : Prop :=
  ∃ (z : ℤ), q = z

theorem integer_count_in_sequence :
  (∃ (k : ℕ), k > 0 ∧
    (∀ (n : ℕ), n < k → isInteger (arithmeticSequence n)) ∧
    ¬isInteger (arithmeticSequence k)) →
  (∃! (k : ℕ), k = 3 ∧
    (∀ (n : ℕ), n < k → isInteger (arithmeticSequence n)) ∧
    ¬isInteger (arithmeticSequence k)) :=
by sorry

end NUMINAMATH_CALUDE_integer_count_in_sequence_l14_1494


namespace NUMINAMATH_CALUDE_rectangle_measurement_error_l14_1459

theorem rectangle_measurement_error (L W : ℝ) (x : ℝ) 
  (h1 : L > 0) (h2 : W > 0) (h3 : x > 0) :
  (1.12 * L) * ((1 - 0.01 * x) * W) = 1.064 * (L * W) → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_measurement_error_l14_1459


namespace NUMINAMATH_CALUDE_art_earnings_l14_1423

/-- The total money earned from an art contest prize and selling paintings -/
def total_money_earned (prize : ℕ) (num_paintings : ℕ) (price_per_painting : ℕ) : ℕ :=
  prize + num_paintings * price_per_painting

/-- Theorem: Given a prize of $150 and selling 3 paintings for $50 each, the total money earned is $300 -/
theorem art_earnings : total_money_earned 150 3 50 = 300 := by
  sorry

end NUMINAMATH_CALUDE_art_earnings_l14_1423


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l14_1426

theorem complex_fraction_equality (a b : ℝ) (i : ℂ) (h1 : i * i = -1) (h2 : a + b * i = i * (1 - i)) :
  (a + b * i) / (a - b * i) = i := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l14_1426


namespace NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l14_1433

theorem lcm_of_ratio_and_hcf (a b : ℕ+) : 
  (a : ℚ) / b = 5 / 13 → 
  Nat.gcd a b = 19 → 
  Nat.lcm a b = 1235 := by
sorry

end NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l14_1433


namespace NUMINAMATH_CALUDE_fraction_five_thirteenths_digit_sum_l14_1476

theorem fraction_five_thirteenths_digit_sum : 
  ∃ (a b c d : ℕ), 
    (a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10) ∧ 
    (5 : ℚ) / 13 = (a * 1000 + b * 100 + c * 10 + d) / 9999 ∧ 
    a + b + c + d = 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_five_thirteenths_digit_sum_l14_1476


namespace NUMINAMATH_CALUDE_average_after_17th_inning_l14_1435

/-- Represents a batsman's statistics -/
structure BatsmanStats where
  innings : Nat
  totalRuns : Nat
  average : Rat

/-- Calculates the new average after an inning -/
def newAverage (stats : BatsmanStats) (runsScored : Nat) : Rat :=
  (stats.totalRuns + runsScored) / (stats.innings + 1)

/-- Theorem: If a batsman's average increases by 3 after scoring 66 runs in the 17th inning, 
    then his average after the 17th inning is 18 -/
theorem average_after_17th_inning 
  (stats : BatsmanStats) 
  (h1 : stats.innings = 16)
  (h2 : newAverage stats 66 = stats.average + 3) :
  newAverage stats 66 = 18 := by
  sorry

end NUMINAMATH_CALUDE_average_after_17th_inning_l14_1435


namespace NUMINAMATH_CALUDE_units_digit_of_n_squared_plus_two_to_n_l14_1483

theorem units_digit_of_n_squared_plus_two_to_n (n : ℕ) : 
  n = 2018^2 + 2^2018 → (n^2 + 2^n) % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_n_squared_plus_two_to_n_l14_1483


namespace NUMINAMATH_CALUDE_F_minimum_value_G_two_zeros_range_inequality_for_positive_x_l14_1418

noncomputable section

-- Define the functions
def f (x : ℝ) : ℝ := (1/2) * x^2
def g (a : ℝ) (x : ℝ) : ℝ := a * Real.log x
def F (a : ℝ) (x : ℝ) : ℝ := f x * g a x
def G (a : ℝ) (x : ℝ) : ℝ := f x - g a x + (a - 1) * x

-- Theorem 1: Minimum value of F(x)
theorem F_minimum_value (a : ℝ) (h : a > 0) :
  ∃ x₀ : ℝ, x₀ > 0 ∧ F a x₀ = -a / (4 * Real.exp 1) ∧ ∀ x > 0, F a x ≥ -a / (4 * Real.exp 1) :=
sorry

-- Theorem 2: Range of a for G(x) to have two zeros
theorem G_two_zeros_range :
  ∃ a₁ a₂ : ℝ, a₁ = (2 * Real.exp 1 - 1) / (2 * Real.exp 1^2 + 2 * Real.exp 1) ∧
               a₂ = 1/2 ∧
               ∀ a : ℝ, (∃ x₁ x₂ : ℝ, 1/Real.exp 1 < x₁ ∧ x₁ < x₂ ∧ x₂ < Real.exp 1 ∧
                                      G a x₁ = 0 ∧ G a x₂ = 0) ↔
                        (a₁ < a ∧ a < a₂) :=
sorry

-- Theorem 3: Inequality for x > 0
theorem inequality_for_positive_x (x : ℝ) (h : x > 0) :
  Real.log x + 3 / (4 * x^2) - 1 / Real.exp x > 0 :=
sorry

end NUMINAMATH_CALUDE_F_minimum_value_G_two_zeros_range_inequality_for_positive_x_l14_1418
