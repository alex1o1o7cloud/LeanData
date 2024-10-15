import Mathlib

namespace NUMINAMATH_CALUDE_survey_problem_l395_39523

theorem survey_problem (A B C : ℝ) 
  (h_A : A = 50)
  (h_B : B = 30)
  (h_C : C = 20)
  (h_union : A + B + C - 17 = 78) 
  (h_multiple : 17 ≤ A + B + C - 78) :
  A + B + C - 78 = 5 := by
sorry

end NUMINAMATH_CALUDE_survey_problem_l395_39523


namespace NUMINAMATH_CALUDE_exponent_multiplication_specific_exponent_multiplication_l395_39584

theorem exponent_multiplication (a b c : ℕ) : (10 : ℝ) ^ a * (10 : ℝ) ^ b = (10 : ℝ) ^ (a + b) :=
by sorry

theorem specific_exponent_multiplication : (10 : ℝ) ^ 10000 * (10 : ℝ) ^ 8000 = (10 : ℝ) ^ 18000 :=
by sorry

end NUMINAMATH_CALUDE_exponent_multiplication_specific_exponent_multiplication_l395_39584


namespace NUMINAMATH_CALUDE_area_of_specific_l_shaped_figure_l395_39598

/-- Represents an L-shaped figure with given dimensions -/
structure LShapedFigure where
  bottom_length : ℕ
  bottom_width : ℕ
  central_length : ℕ
  central_width : ℕ
  top_length : ℕ
  top_width : ℕ

/-- Calculates the area of an L-shaped figure -/
def area_of_l_shaped_figure (f : LShapedFigure) : ℕ :=
  f.bottom_length * f.bottom_width +
  f.central_length * f.central_width +
  f.top_length * f.top_width

/-- Theorem stating that the area of the given L-shaped figure is 81 square units -/
theorem area_of_specific_l_shaped_figure :
  let f : LShapedFigure := {
    bottom_length := 10,
    bottom_width := 6,
    central_length := 4,
    central_width := 4,
    top_length := 5,
    top_width := 1
  }
  area_of_l_shaped_figure f = 81 := by
  sorry

end NUMINAMATH_CALUDE_area_of_specific_l_shaped_figure_l395_39598


namespace NUMINAMATH_CALUDE_cost_of_candies_l395_39577

def candies_per_box : ℕ := 30
def cost_per_box : ℕ := 8
def total_candies : ℕ := 450

theorem cost_of_candies :
  (total_candies / candies_per_box) * cost_per_box = 120 := by
sorry

end NUMINAMATH_CALUDE_cost_of_candies_l395_39577


namespace NUMINAMATH_CALUDE_fencing_rate_proof_l395_39554

/-- Given a rectangular plot with the following properties:
    - The length is 10 meters more than the width
    - The perimeter is 340 meters
    - The total cost of fencing is 2210 Rs
    Prove that the rate per meter for fencing is 6.5 Rs -/
theorem fencing_rate_proof (width : ℝ) (length : ℝ) (perimeter : ℝ) (total_cost : ℝ) :
  length = width + 10 →
  perimeter = 340 →
  perimeter = 2 * (length + width) →
  total_cost = 2210 →
  total_cost / perimeter = 6.5 := by
  sorry

end NUMINAMATH_CALUDE_fencing_rate_proof_l395_39554


namespace NUMINAMATH_CALUDE_equation_equivalence_l395_39505

theorem equation_equivalence :
  ∀ x : ℝ, x^2 - 2*x - 2 = 0 ↔ (x - 1)^2 = 3 :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l395_39505


namespace NUMINAMATH_CALUDE_equation_solution_l395_39517

theorem equation_solution : ∃! x : ℝ, 2 * x + 1 = x - 1 ∧ x = -2 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l395_39517


namespace NUMINAMATH_CALUDE_expression_evaluation_l395_39572

theorem expression_evaluation : 
  (Real.sqrt 3 - 4 * Real.sin (20 * π / 180) + 8 * (Real.sin (20 * π / 180))^3) / 
  (2 * Real.sin (20 * π / 180) * Real.sin (480 * π / 180)) = 
  2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l395_39572


namespace NUMINAMATH_CALUDE_vincent_animal_books_l395_39587

/-- The number of books about animals Vincent bought -/
def num_animal_books : ℕ := sorry

/-- The cost of each book -/
def book_cost : ℕ := 16

/-- The total number of books about outer space and trains -/
def num_other_books : ℕ := 1 + 3

/-- The total amount Vincent spent on books -/
def total_spent : ℕ := 224

theorem vincent_animal_books : 
  num_animal_books = 10 := by sorry

end NUMINAMATH_CALUDE_vincent_animal_books_l395_39587


namespace NUMINAMATH_CALUDE_race_results_l395_39504

/-- Represents a runner in the race -/
structure Runner where
  pace : ℕ  -- pace in minutes per mile
  breakTime : ℕ  -- break time in minutes
  breakStart : ℕ  -- time at which the break starts in minutes

/-- Calculates the total time taken by a runner to complete the race -/
def totalTime (r : Runner) (raceDistance : ℕ) : ℕ :=
  let distanceBeforeBreak := r.breakStart / r.pace
  let distanceAfterBreak := raceDistance - distanceBeforeBreak
  r.breakStart + r.breakTime + distanceAfterBreak * r.pace

/-- The main theorem stating the total time for each runner -/
theorem race_results (raceDistance : ℕ) (runner1 runner2 runner3 : Runner) : 
  raceDistance = 15 ∧ 
  runner1.pace = 6 ∧ runner1.breakTime = 3 ∧ runner1.breakStart = 42 ∧
  runner2.pace = 7 ∧ runner2.breakTime = 5 ∧ runner2.breakStart = 49 ∧
  runner3.pace = 8 ∧ runner3.breakTime = 7 ∧ runner3.breakStart = 56 →
  totalTime runner1 raceDistance = 93 ∧
  totalTime runner2 raceDistance = 110 ∧
  totalTime runner3 raceDistance = 127 := by
  sorry

end NUMINAMATH_CALUDE_race_results_l395_39504


namespace NUMINAMATH_CALUDE_book_arrangement_theorem_l395_39537

/-- The number of ways to arrange books of different languages on a shelf. -/
def arrange_books (arabic : ℕ) (german : ℕ) (spanish : ℕ) : ℕ :=
  (Nat.factorial 3) * (Nat.factorial arabic) * (Nat.factorial german) * (Nat.factorial spanish)

/-- Theorem: The number of ways to arrange 10 books (2 Arabic, 4 German, 4 Spanish) on a shelf,
    keeping books of the same language together, is equal to 6912. -/
theorem book_arrangement_theorem :
  arrange_books 2 4 4 = 6912 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_theorem_l395_39537


namespace NUMINAMATH_CALUDE_integer_solution_proof_l395_39535

theorem integer_solution_proof (a b c : ℤ) :
  a + b + c = 24 →
  a^2 + b^2 + c^2 = 210 →
  a * b * c = 440 →
  ({a, b, c} : Set ℤ) = {5, 8, 11} := by
  sorry

end NUMINAMATH_CALUDE_integer_solution_proof_l395_39535


namespace NUMINAMATH_CALUDE_art_piece_original_price_l395_39546

/-- Proves that the original purchase price of an art piece is $4000 -/
theorem art_piece_original_price :
  ∀ (original_price future_price : ℝ),
  future_price = 3 * original_price →
  future_price - original_price = 8000 →
  original_price = 4000 := by
sorry

end NUMINAMATH_CALUDE_art_piece_original_price_l395_39546


namespace NUMINAMATH_CALUDE_division_of_fractions_l395_39568

theorem division_of_fractions : (3 : ℚ) / (6 / 11) = 11 / 2 := by
  sorry

end NUMINAMATH_CALUDE_division_of_fractions_l395_39568


namespace NUMINAMATH_CALUDE_max_value_sum_of_fractions_l395_39531

theorem max_value_sum_of_fractions (a b c : ℝ) 
  (nonneg_a : 0 ≤ a) (nonneg_b : 0 ≤ b) (nonneg_c : 0 ≤ c)
  (sum_eq_three : a + b + c = 3) :
  (a * b) / (a + b) + (a * c) / (a + c) + (b * c) / (b + c) ≤ 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_max_value_sum_of_fractions_l395_39531


namespace NUMINAMATH_CALUDE_lost_pages_problem_l395_39520

/-- Calculates the number of lost pages of stickers -/
def lost_pages (stickers_per_page : ℕ) (initial_pages : ℕ) (remaining_stickers : ℕ) : ℕ :=
  (stickers_per_page * initial_pages - remaining_stickers) / stickers_per_page

theorem lost_pages_problem :
  let stickers_per_page : ℕ := 20
  let initial_pages : ℕ := 12
  let remaining_stickers : ℕ := 220
  lost_pages stickers_per_page initial_pages remaining_stickers = 1 := by
  sorry

end NUMINAMATH_CALUDE_lost_pages_problem_l395_39520


namespace NUMINAMATH_CALUDE_income_increase_theorem_l395_39538

/-- Represents the student's income sources and total income -/
structure StudentIncome where
  scholarship : ℝ
  partTimeJob : ℝ
  parentalSupport : ℝ
  totalIncome : ℝ

/-- Theorem stating the relationship between income sources and total income increase -/
theorem income_increase_theorem (income : StudentIncome) 
  (h1 : income.scholarship + income.partTimeJob + income.parentalSupport = income.totalIncome)
  (h2 : 2 * income.scholarship + income.partTimeJob + income.parentalSupport = 1.05 * income.totalIncome)
  (h3 : income.scholarship + 2 * income.partTimeJob + income.parentalSupport = 1.15 * income.totalIncome) :
  income.scholarship + income.partTimeJob + 2 * income.parentalSupport = 1.8 * income.totalIncome := by
  sorry


end NUMINAMATH_CALUDE_income_increase_theorem_l395_39538


namespace NUMINAMATH_CALUDE_fraction_simplification_l395_39544

theorem fraction_simplification (x : ℝ) (h : x ≠ -3) : 
  (x^2 - 9) / (x^2 + 6*x + 9) - (2*x + 1) / (2*x + 6) = -7 / (2*x + 6) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l395_39544


namespace NUMINAMATH_CALUDE_point_on_330_degree_angle_l395_39583

/-- For any point P (x, y) ≠ (0, 0) on the terminal side of a 330° angle, y/x = -√3/3 -/
theorem point_on_330_degree_angle (x y : ℝ) : 
  (x ≠ 0 ∨ y ≠ 0) →  -- Point is not the origin
  (x, y) ∈ {p : ℝ × ℝ | ∃ (r : ℝ), r > 0 ∧ p.1 = r * Real.cos (330 * π / 180) ∧ p.2 = r * Real.sin (330 * π / 180)} →  -- Point is on the terminal side of 330° angle
  y / x = -Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_point_on_330_degree_angle_l395_39583


namespace NUMINAMATH_CALUDE_spherical_coordinates_of_point_A_l395_39536

theorem spherical_coordinates_of_point_A :
  let x : ℝ := (3 * Real.sqrt 3) / 2
  let y : ℝ := 9 / 2
  let z : ℝ := 3
  let r : ℝ := Real.sqrt (x^2 + y^2 + z^2)
  let θ : ℝ := Real.arctan (y / x)
  let φ : ℝ := Real.arccos (z / r)
  (r = 6) ∧ (θ = π / 3) ∧ (φ = π / 3) := by sorry

end NUMINAMATH_CALUDE_spherical_coordinates_of_point_A_l395_39536


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l395_39533

open Set

/-- The universal set U -/
def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}

/-- Set A -/
def A : Set Nat := {3, 4, 5}

/-- Set B -/
def B : Set Nat := {1, 3, 6}

/-- Theorem stating that the intersection of A and the complement of B in U equals {4, 5} -/
theorem intersection_A_complement_B : A ∩ (U \ B) = {4, 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l395_39533


namespace NUMINAMATH_CALUDE_doughnut_sharing_l395_39561

/-- The number of doughnuts in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens Samuel bought -/
def samuel_dozens : ℕ := 2

/-- The number of dozens Cathy bought -/
def cathy_dozens : ℕ := 3

/-- The number of doughnuts each person received -/
def doughnuts_per_person : ℕ := 6

/-- The number of people who bought the doughnuts (Samuel and Cathy) -/
def buyers : ℕ := 2

theorem doughnut_sharing :
  let total_doughnuts := samuel_dozens * dozen + cathy_dozens * dozen
  let total_people := total_doughnuts / doughnuts_per_person
  total_people - buyers = 8 := by
  sorry

end NUMINAMATH_CALUDE_doughnut_sharing_l395_39561


namespace NUMINAMATH_CALUDE_problem_proof_l395_39525

theorem problem_proof : (5 * 12) / (180 / 3) + 61 = 62 := by
  sorry

end NUMINAMATH_CALUDE_problem_proof_l395_39525


namespace NUMINAMATH_CALUDE_max_value_constrained_product_l395_39516

theorem max_value_constrained_product (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 5*x + 3*y < 75) :
  x * y * (75 - 5*x - 3*y) ≤ 3125/3 :=
sorry

end NUMINAMATH_CALUDE_max_value_constrained_product_l395_39516


namespace NUMINAMATH_CALUDE_oil_quantity_function_correct_l395_39578

/-- Represents the remaining oil quantity in a tank as a function of time -/
def Q (t : ℝ) : ℝ := 20 - 0.2 * t

/-- The initial oil quantity in the tank -/
def initial_quantity : ℝ := 20

/-- The rate at which oil flows out of the tank (in liters per minute) -/
def flow_rate : ℝ := 0.2

theorem oil_quantity_function_correct :
  ∀ t : ℝ, t ≥ 0 →
  Q t = initial_quantity - flow_rate * t ∧
  Q t ≥ 0 ∧
  (Q t = 0 → t = initial_quantity / flow_rate) :=
sorry

end NUMINAMATH_CALUDE_oil_quantity_function_correct_l395_39578


namespace NUMINAMATH_CALUDE_arithmetic_sequence_count_l395_39509

theorem arithmetic_sequence_count (a₁ aₙ d : ℤ) (n : ℕ) :
  a₁ = 165 ∧ aₙ = 35 ∧ d = -5 →
  aₙ = a₁ + (n - 1) * d →
  n = 27 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_count_l395_39509


namespace NUMINAMATH_CALUDE_bug_returns_probability_l395_39557

def bug_probability (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | k + 1 => 1/3 * (1 - bug_probability k)

theorem bug_returns_probability :
  bug_probability 12 = 44287 / 177147 :=
by sorry

end NUMINAMATH_CALUDE_bug_returns_probability_l395_39557


namespace NUMINAMATH_CALUDE_bird_families_to_africa_l395_39527

theorem bird_families_to_africa (total : ℕ) (to_asia : ℕ) (remaining : ℕ) : 
  total = 85 → to_asia = 37 → remaining = 25 → total - to_asia - remaining = 23 :=
by sorry

end NUMINAMATH_CALUDE_bird_families_to_africa_l395_39527


namespace NUMINAMATH_CALUDE_max_polygon_area_l395_39506

/-- A point with integer coordinates satisfying the given conditions -/
structure ValidPoint where
  x : ℕ+
  y : ℕ+
  cond1 : x ∣ (2 * y + 1)
  cond2 : y ∣ (2 * x + 1)

/-- The set of all valid points -/
def ValidPoints : Set ValidPoint := {p : ValidPoint | True}

/-- The area of a polygon formed by a set of points -/
noncomputable def polygonArea (points : Set ValidPoint) : ℝ := sorry

/-- The maximum area of a polygon formed by valid points -/
theorem max_polygon_area :
  ∃ (points : Set ValidPoint), points ⊆ ValidPoints ∧ polygonArea points = 20 ∧
    ∀ (otherPoints : Set ValidPoint), otherPoints ⊆ ValidPoints →
      polygonArea otherPoints ≤ 20 := by sorry

end NUMINAMATH_CALUDE_max_polygon_area_l395_39506


namespace NUMINAMATH_CALUDE_small_circle_radius_l395_39510

/-- Given a large circle with radius 10 meters containing seven smaller congruent circles
    that fit exactly along its diameter, prove that the radius of each smaller circle is 10/7 meters. -/
theorem small_circle_radius (R : ℝ) (n : ℕ) (r : ℝ) : 
  R = 10 → n = 7 → 2 * R = n * (2 * r) → r = 10 / 7 := by sorry

end NUMINAMATH_CALUDE_small_circle_radius_l395_39510


namespace NUMINAMATH_CALUDE_average_speed_round_trip_l395_39559

/-- Given uphill speed V₁ and downhill speed V₂, 
    the average speed for a round trip is (2 * V₁ * V₂) / (V₁ + V₂) -/
theorem average_speed_round_trip (V₁ V₂ : ℝ) (h₁ : V₁ > 0) (h₂ : V₂ > 0) :
  let s : ℝ := 1  -- Assume unit distance for simplicity
  let t_up : ℝ := s / V₁
  let t_down : ℝ := s / V₂
  let total_distance : ℝ := 2 * s
  let total_time : ℝ := t_up + t_down
  total_distance / total_time = (2 * V₁ * V₂) / (V₁ + V₂) :=
by sorry

end NUMINAMATH_CALUDE_average_speed_round_trip_l395_39559


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_one_l395_39532

theorem fraction_zero_implies_x_one :
  ∀ x : ℝ, (x - 1) / (2 * x - 4) = 0 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_one_l395_39532


namespace NUMINAMATH_CALUDE_hyperbola_equation_l395_39590

/-- Given a hyperbola with the specified properties, prove its equation -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := 5 / 4
  let c := 5
  (c / a = e) →
  (c^2 = a^2 + b^2) →
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 / 16 - y^2 / 9 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l395_39590


namespace NUMINAMATH_CALUDE_eggs_per_basket_l395_39528

theorem eggs_per_basket (total_eggs : ℕ) (num_baskets : ℕ) 
  (h1 : total_eggs = 8484) (h2 : num_baskets = 303) :
  total_eggs / num_baskets = 28 := by
  sorry

end NUMINAMATH_CALUDE_eggs_per_basket_l395_39528


namespace NUMINAMATH_CALUDE_profit_calculation_l395_39579

/-- Profit calculation for a company --/
theorem profit_calculation (total_profit second_half_profit first_half_profit : ℚ) :
  total_profit = 3635000 →
  first_half_profit = second_half_profit + 2750000 →
  total_profit = first_half_profit + second_half_profit →
  second_half_profit = 442500 := by
sorry

end NUMINAMATH_CALUDE_profit_calculation_l395_39579


namespace NUMINAMATH_CALUDE_complex_magnitude_l395_39556

theorem complex_magnitude (z : ℂ) (h : z = Complex.mk 2 (-1)) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l395_39556


namespace NUMINAMATH_CALUDE_rectangle_circle_tangent_l395_39573

/-- Given a circle with radius 6 cm tangent to two shorter sides and one longer side of a rectangle,
    and the area of the rectangle being three times the area of the circle,
    prove that the length of the shorter side of the rectangle is 12 cm. -/
theorem rectangle_circle_tangent (circle_radius : ℝ) (rectangle_area : ℝ) (circle_area : ℝ) :
  circle_radius = 6 →
  rectangle_area = 3 * circle_area →
  circle_area = Real.pi * circle_radius^2 →
  (12 : ℝ) = 2 * circle_radius :=
by sorry

end NUMINAMATH_CALUDE_rectangle_circle_tangent_l395_39573


namespace NUMINAMATH_CALUDE_prob_different_colors_for_given_box_l395_39519

/-- A box containing balls of two colors -/
structure Box where
  small_balls : ℕ
  black_balls : ℕ

/-- The probability of drawing two balls of different colors -/
def prob_different_colors (b : Box) : ℚ :=
  let total_balls := b.small_balls + b.black_balls
  let different_color_combinations := b.small_balls * b.black_balls
  let total_combinations := (total_balls * (total_balls - 1)) / 2
  different_color_combinations / total_combinations

/-- The theorem stating the probability of drawing two balls of different colors -/
theorem prob_different_colors_for_given_box :
  prob_different_colors { small_balls := 3, black_balls := 1 } = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_prob_different_colors_for_given_box_l395_39519


namespace NUMINAMATH_CALUDE_exists_special_number_l395_39521

/-- A function that returns the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A function that returns the number of digits in a natural number -/
def num_digits (n : ℕ) : ℕ := sorry

/-- A function that checks if all digits of a natural number are non-zero -/
def all_digits_nonzero (n : ℕ) : Prop := sorry

/-- Theorem: There exists a 1000-digit natural number with all non-zero digits that is divisible by the sum of its digits -/
theorem exists_special_number : 
  ∃ n : ℕ, 
    num_digits n = 1000 ∧ 
    all_digits_nonzero n ∧ 
    n % sum_of_digits n = 0 :=
sorry

end NUMINAMATH_CALUDE_exists_special_number_l395_39521


namespace NUMINAMATH_CALUDE_sequence_property_l395_39549

def a (n : ℕ) (x : ℝ) : ℝ := 1 + x^(n+1) + x^(n+2)

theorem sequence_property (x : ℝ) :
  (a 2 x)^2 = (a 1 x) * (a 3 x) →
  ∀ n ≥ 3, (a n x)^2 = a n x :=
by sorry

end NUMINAMATH_CALUDE_sequence_property_l395_39549


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l395_39526

theorem imaginary_part_of_z (z : ℂ) (h : z + z * Complex.I = 2) : 
  z.im = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l395_39526


namespace NUMINAMATH_CALUDE_sum_squares_possible_values_l395_39543

/-- Given a positive real number A, prove that for any y in the open interval (0, A^2),
    there exists a sequence of positive real numbers {x_j} such that the sum of x_j equals A
    and the sum of x_j^2 equals y. -/
theorem sum_squares_possible_values (A : ℝ) (hA : A > 0) (y : ℝ) (hy1 : y > 0) (hy2 : y < A^2) :
  ∃ (x : ℕ → ℝ), (∀ j, x j > 0) ∧
    (∑' j, x j) = A ∧
    (∑' j, (x j)^2) = y :=
sorry

end NUMINAMATH_CALUDE_sum_squares_possible_values_l395_39543


namespace NUMINAMATH_CALUDE_min_avg_of_two_l395_39580

theorem min_avg_of_two (a b c d : ℕ+) : 
  (a + b + c + d : ℝ) / 4 = 50 → 
  max c d ≤ 130 →
  (a + b : ℝ) / 2 ≥ 35 :=
by sorry

end NUMINAMATH_CALUDE_min_avg_of_two_l395_39580


namespace NUMINAMATH_CALUDE_arrangement_count_l395_39541

-- Define the number of red and blue balls
def red_balls : ℕ := 8
def blue_balls : ℕ := 9
def total_balls : ℕ := red_balls + blue_balls

-- Define the number of jars
def num_jars : ℕ := 2

-- Define a function to calculate the number of distinguishable arrangements
def count_arrangements (red : ℕ) (blue : ℕ) (jars : ℕ) : ℕ :=
  sorry -- The actual implementation would go here

-- State the theorem
theorem arrangement_count :
  count_arrangements red_balls blue_balls num_jars = 7 :=
by sorry

end NUMINAMATH_CALUDE_arrangement_count_l395_39541


namespace NUMINAMATH_CALUDE_line_relationships_exhaustive_l395_39596

-- Define the possible spatial relationships between lines
inductive LineRelationship
  | Parallel
  | Intersecting
  | Skew

-- Define a line in 3D space
structure Line3D where
  -- We don't need to specify the exact representation of a line here
  -- as it's not relevant for the statement of the theorem

-- Define the relationship between two lines
def relationshipBetweenLines (l1 l2 : Line3D) : LineRelationship :=
  sorry -- The actual implementation is not needed for the statement

-- Theorem statement
theorem line_relationships_exhaustive (l1 l2 : Line3D) :
  ∃ (r : LineRelationship), relationshipBetweenLines l1 l2 = r :=
sorry

end NUMINAMATH_CALUDE_line_relationships_exhaustive_l395_39596


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_l395_39511

-- Define the propositions p and q
def p (x : ℝ) : Prop := x = 1
def q (x : ℝ) : Prop := x - 1 = Real.sqrt (x - 1)

-- Theorem stating that p is a sufficient but not necessary condition for q
theorem p_sufficient_not_necessary :
  (∀ x : ℝ, p x → q x) ∧ (∃ x : ℝ, q x ∧ ¬p x) := by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_l395_39511


namespace NUMINAMATH_CALUDE_jimin_calculation_l395_39539

theorem jimin_calculation (x : ℤ) (h : 20 - x = 60) : 34 * x = -1360 := by
  sorry

end NUMINAMATH_CALUDE_jimin_calculation_l395_39539


namespace NUMINAMATH_CALUDE_constant_term_is_135_l395_39575

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the constant term in the expansion
def constant_term (x : ℝ) : ℝ :=
  binomial 6 2 * 3^2

-- Theorem statement
theorem constant_term_is_135 :
  constant_term = 135 := by sorry

end NUMINAMATH_CALUDE_constant_term_is_135_l395_39575


namespace NUMINAMATH_CALUDE_min_max_abs_quadratic_minus_linear_l395_39562

theorem min_max_abs_quadratic_minus_linear (y : ℝ) :
  ∃ (y : ℝ), ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 → |x^2 - x*y| ≤ 0 ∧
  (∀ (y : ℝ), ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 2 ∧ |x^2 - x*y| ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_min_max_abs_quadratic_minus_linear_l395_39562


namespace NUMINAMATH_CALUDE_ken_kept_twenty_pencils_l395_39551

/-- The number of pencils Ken initially had -/
def initial_pencils : ℕ := 50

/-- The number of pencils Ken gave to Manny -/
def pencils_to_manny : ℕ := 10

/-- The number of additional pencils Ken gave to Nilo compared to Manny -/
def additional_pencils_to_nilo : ℕ := 10

/-- The number of pencils Ken kept -/
def pencils_kept : ℕ := initial_pencils - (pencils_to_manny + (pencils_to_manny + additional_pencils_to_nilo))

theorem ken_kept_twenty_pencils : pencils_kept = 20 := by sorry

end NUMINAMATH_CALUDE_ken_kept_twenty_pencils_l395_39551


namespace NUMINAMATH_CALUDE_rectangle_area_l395_39550

/-- Given a wire of length 32 cm bent into a rectangle with a length-to-width ratio of 5:3,
    the area of the resulting rectangle is 60 cm². -/
theorem rectangle_area (wire_length : ℝ) (length : ℝ) (width : ℝ) : 
  wire_length = 32 →
  length / width = 5 / 3 →
  2 * (length + width) = wire_length →
  length * width = 60 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l395_39550


namespace NUMINAMATH_CALUDE_computer_pricing_l395_39513

/-- 
Given a computer's selling price and profit percentage, 
calculate the new selling price for a different profit percentage.
-/
theorem computer_pricing (initial_price : ℝ) (initial_profit_percent : ℝ) 
  (new_profit_percent : ℝ) (new_price : ℝ) :
  initial_price = (1 + initial_profit_percent / 100) * (initial_price / (1 + initial_profit_percent / 100)) →
  new_price = (1 + new_profit_percent / 100) * (initial_price / (1 + initial_profit_percent / 100)) →
  initial_price = 2240 →
  initial_profit_percent = 40 →
  new_profit_percent = 50 →
  new_price = 2400 :=
by sorry

end NUMINAMATH_CALUDE_computer_pricing_l395_39513


namespace NUMINAMATH_CALUDE_emilys_weight_l395_39565

/-- Given Heather's weight and the difference between Heather and Emily's weights,
    prove that Emily's weight is 9 pounds. -/
theorem emilys_weight (heathers_weight : ℕ) (weight_difference : ℕ)
  (hw : heathers_weight = 87)
  (diff : weight_difference = 78)
  : heathers_weight - weight_difference = 9 := by
  sorry

#check emilys_weight

end NUMINAMATH_CALUDE_emilys_weight_l395_39565


namespace NUMINAMATH_CALUDE_negation_of_exists_square_greater_than_power_of_two_l395_39569

theorem negation_of_exists_square_greater_than_power_of_two :
  (¬ ∃ (n : ℕ+), n.val ^ 2 > 2 ^ n.val) ↔ ∀ (n : ℕ+), n.val ^ 2 ≤ 2 ^ n.val :=
by sorry

end NUMINAMATH_CALUDE_negation_of_exists_square_greater_than_power_of_two_l395_39569


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l395_39567

def A : Set (ℝ × ℝ) := {p | p.2 = 2 * p.1 + 5}
def B : Set (ℝ × ℝ) := {p | p.2 = 1 - 2 * p.1}

theorem intersection_of_A_and_B : A ∩ B = {(-1, 3)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l395_39567


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_values_l395_39574

theorem fraction_zero_implies_x_values (x : ℝ) : 
  (x ^ 2 - 4) / x = 0 → x = 2 ∨ x = -2 :=
by sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_values_l395_39574


namespace NUMINAMATH_CALUDE_samson_sandwich_count_l395_39597

/-- The number of sandwiches Samson ate at lunch on Monday -/
def lunch_sandwiches : ℕ := sorry

/-- The number of sandwiches Samson ate at dinner on Monday -/
def dinner_sandwiches : ℕ := 2 * lunch_sandwiches

/-- The number of sandwiches Samson ate for breakfast on Tuesday -/
def tuesday_breakfast : ℕ := 1

/-- The total number of sandwiches Samson ate on Monday -/
def monday_total : ℕ := lunch_sandwiches + dinner_sandwiches

/-- The total number of sandwiches Samson ate on Tuesday -/
def tuesday_total : ℕ := tuesday_breakfast

theorem samson_sandwich_count : lunch_sandwiches = 3 := by
  have h1 : monday_total = tuesday_total + 8 := by sorry
  sorry

end NUMINAMATH_CALUDE_samson_sandwich_count_l395_39597


namespace NUMINAMATH_CALUDE_ice_cream_distribution_l395_39548

theorem ice_cream_distribution (total_sandwiches : ℕ) (num_nieces : ℕ) 
  (h1 : total_sandwiches = 1857)
  (h2 : num_nieces = 37) :
  (total_sandwiches / num_nieces : ℕ) = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_ice_cream_distribution_l395_39548


namespace NUMINAMATH_CALUDE_fifth_rollercoaster_speed_l395_39524

/-- Theorem: Given 5 rollercoasters with specific speeds and average, prove the speed of the fifth rollercoaster -/
theorem fifth_rollercoaster_speed 
  (v₁ v₂ v₃ v₄ v₅ : ℝ) 
  (h1 : v₁ = 50)
  (h2 : v₂ = 62)
  (h3 : v₃ = 73)
  (h4 : v₄ = 70)
  (h_avg : (v₁ + v₂ + v₃ + v₄ + v₅) / 5 = 59) :
  v₅ = 40 := by
  sorry

end NUMINAMATH_CALUDE_fifth_rollercoaster_speed_l395_39524


namespace NUMINAMATH_CALUDE_x_current_age_l395_39503

/-- Proves that X's current age is 45 years given the provided conditions -/
theorem x_current_age (x y : ℕ) : 
  (x - 3 = 2 * (y - 3)) →  -- X's age was double Y's age three years ago
  (x + y + 14 = 83) →      -- Seven years from now, the sum of their ages will be 83
  x = 45 :=                -- X's current age is 45
by
  sorry

#check x_current_age

end NUMINAMATH_CALUDE_x_current_age_l395_39503


namespace NUMINAMATH_CALUDE_park_diameter_l395_39529

/-- Given a circular park with a central pond, vegetable garden, and jogging path,
    this theorem proves that the diameter of the outer boundary is 64 feet. -/
theorem park_diameter (pond_diameter vegetable_width jogging_width : ℝ) 
  (h1 : pond_diameter = 20)
  (h2 : vegetable_width = 12)
  (h3 : jogging_width = 10) :
  2 * (pond_diameter / 2 + vegetable_width + jogging_width) = 64 := by
  sorry

end NUMINAMATH_CALUDE_park_diameter_l395_39529


namespace NUMINAMATH_CALUDE_sum_of_last_two_digits_11_pow_2002_l395_39564

theorem sum_of_last_two_digits_11_pow_2002 : ∃ n : ℕ, 
  11^2002 = n * 100 + 21 ∧ n ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_sum_of_last_two_digits_11_pow_2002_l395_39564


namespace NUMINAMATH_CALUDE_train_speed_calculation_l395_39581

-- Define the length of the train in meters
def train_length : ℝ := 83.33333333333334

-- Define the time taken to cross the pole in seconds
def crossing_time : ℝ := 5

-- Define the speed of the train in km/hr
def train_speed : ℝ := 60

-- Theorem to prove
theorem train_speed_calculation :
  train_speed = (train_length / 1000) / (crossing_time / 3600) :=
by sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l395_39581


namespace NUMINAMATH_CALUDE_units_digit_of_7_power_2006_l395_39594

theorem units_digit_of_7_power_2006 : ∃ n : ℕ, 7^2006 ≡ 9 [ZMOD 10] := by sorry

end NUMINAMATH_CALUDE_units_digit_of_7_power_2006_l395_39594


namespace NUMINAMATH_CALUDE_probability_both_selected_l395_39518

theorem probability_both_selected (prob_ram : ℚ) (prob_ravi : ℚ) 
  (h1 : prob_ram = 4/7) (h2 : prob_ravi = 1/5) : 
  prob_ram * prob_ravi = 4/35 := by
  sorry

end NUMINAMATH_CALUDE_probability_both_selected_l395_39518


namespace NUMINAMATH_CALUDE_girl_transfer_problem_l395_39553

/-- Represents the number of girls in each group before and after transfers -/
structure GirlCounts where
  initial_B : ℕ
  initial_A : ℕ
  initial_C : ℕ
  final : ℕ

/-- Represents the number of girls transferred between groups -/
structure GirlTransfers where
  from_A : ℕ
  from_B : ℕ
  from_C : ℕ

/-- The theorem statement for the girl transfer problem -/
theorem girl_transfer_problem (g : GirlCounts) (t : GirlTransfers) : 
  g.initial_A = g.initial_B + 4 →
  g.initial_B = g.initial_C + 1 →
  t.from_C = 2 →
  g.final = g.initial_A - t.from_A + t.from_C →
  g.final = g.initial_B - t.from_B + t.from_A →
  g.final = g.initial_C - t.from_C + t.from_B →
  t.from_A = 5 ∧ t.from_B = 4 := by
  sorry


end NUMINAMATH_CALUDE_girl_transfer_problem_l395_39553


namespace NUMINAMATH_CALUDE_room_assignment_count_l395_39566

/-- The number of rooms in the lodge -/
def num_rooms : ℕ := 6

/-- The number of friends checking in -/
def num_friends : ℕ := 7

/-- The maximum number of friends allowed per room -/
def max_per_room : ℕ := 2

/-- The minimum number of unoccupied rooms -/
def min_unoccupied : ℕ := 1

/-- A function that calculates the number of ways to assign friends to rooms -/
def assign_rooms : ℕ := sorry

/-- Theorem stating that the number of ways to assign rooms is 128520 -/
theorem room_assignment_count : assign_rooms = 128520 := by sorry

end NUMINAMATH_CALUDE_room_assignment_count_l395_39566


namespace NUMINAMATH_CALUDE_inequality_implications_l395_39500

theorem inequality_implications (a b : ℝ) (h : a > b) :
  (a + 2 > b + 2) ∧
  (-a < -b) ∧
  (2 * a > 2 * b) ∧
  ∃ c : ℝ, ¬(a * c^2 > b * c^2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_implications_l395_39500


namespace NUMINAMATH_CALUDE_average_waiting_time_for_first_bite_l395_39534

/-- The average waiting time for the first bite in a fishing scenario --/
theorem average_waiting_time_for_first_bite 
  (time_interval : ℝ) 
  (first_rod_bites : ℝ) 
  (second_rod_bites : ℝ) 
  (total_bites : ℝ) 
  (h1 : time_interval = 6)
  (h2 : first_rod_bites = 3)
  (h3 : second_rod_bites = 2)
  (h4 : total_bites = first_rod_bites + second_rod_bites) :
  (time_interval / total_bites) = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_average_waiting_time_for_first_bite_l395_39534


namespace NUMINAMATH_CALUDE_hotel_guests_count_l395_39530

/-- The number of guests attending the Oates reunion -/
def oates_attendees : ℕ := 50

/-- The number of guests attending the Hall reunion -/
def hall_attendees : ℕ := 62

/-- The number of guests attending both reunions -/
def both_attendees : ℕ := 12

/-- The total number of guests at the hotel -/
def total_guests : ℕ := (oates_attendees - both_attendees) + (hall_attendees - both_attendees) + both_attendees

theorem hotel_guests_count :
  total_guests = 100 := by sorry

end NUMINAMATH_CALUDE_hotel_guests_count_l395_39530


namespace NUMINAMATH_CALUDE_proportion_solution_l395_39558

theorem proportion_solution (x : ℝ) : (0.75 / x = 10 / 8) → x = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l395_39558


namespace NUMINAMATH_CALUDE_strawberry_picking_l395_39515

/-- The number of baskets Lilibeth fills -/
def baskets : ℕ := 6

/-- The number of strawberries each basket holds -/
def strawberries_per_basket : ℕ := 50

/-- The number of Lilibeth's friends who pick the same amount as her -/
def friends : ℕ := 3

/-- The total number of strawberries picked by Lilibeth and her friends -/
def total_strawberries : ℕ := (friends + 1) * (baskets * strawberries_per_basket)

theorem strawberry_picking :
  total_strawberries = 1200 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_picking_l395_39515


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l395_39502

/-- Given a geometric sequence {aₙ}, prove that if a₅ - a₁ = 15 and a₄ - a₂ = 6,
    then (a₃ = 4 and q = 2) or (a₃ = -4 and q = 1/2) -/
theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- Definition of geometric sequence
  a 5 - a 1 = 15 →              -- First given condition
  a 4 - a 2 = 6 →               -- Second given condition
  ((a 3 = 4 ∧ q = 2) ∨ (a 3 = -4 ∧ q = 1/2)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l395_39502


namespace NUMINAMATH_CALUDE_freshman_class_size_l395_39547

theorem freshman_class_size :
  ∃ n : ℕ, n < 500 ∧ n % 25 = 24 ∧ n % 19 = 11 ∧
  ∀ m : ℕ, m < n → (m % 25 ≠ 24 ∨ m % 19 ≠ 11) :=
by sorry

end NUMINAMATH_CALUDE_freshman_class_size_l395_39547


namespace NUMINAMATH_CALUDE_card_flip_game_l395_39545

theorem card_flip_game (n k : ℕ) (hn : Odd n) (hk : Even k) (hkn : k < n) :
  ∀ (t : ℕ), ∃ (i : ℕ), i < n ∧ Even (t * k / n + (if i < t * k % n then 1 else 0)) := by
  sorry

end NUMINAMATH_CALUDE_card_flip_game_l395_39545


namespace NUMINAMATH_CALUDE_hidden_lattice_points_l395_39571

theorem hidden_lattice_points (n : ℕ+) : 
  ∃ a b : ℤ, ∀ i j : ℕ, i < n ∧ j < n → Nat.gcd (Int.toNat (a + i)) (Int.toNat (b + j)) > 1 := by
  sorry

end NUMINAMATH_CALUDE_hidden_lattice_points_l395_39571


namespace NUMINAMATH_CALUDE_expand_and_simplify_l395_39586

theorem expand_and_simplify (x : ℝ) (h : x ≠ 0) :
  (3 / 4) * (8 / x - 15 * x^3 + 6 * x) = 6 / x - 45 / 4 * x^3 + 9 / 2 * x := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l395_39586


namespace NUMINAMATH_CALUDE_cookies_theorem_l395_39599

def cookies_problem (total_cookies : ℕ) : Prop :=
  let father_cookies := (total_cookies : ℚ) * (1 / 10)
  let mother_cookies := father_cookies / 2
  let brother_cookies := mother_cookies + 2
  let sister_cookies := brother_cookies * (3 / 2)
  let aunt_cookies := father_cookies * 2
  let cousin_cookies := aunt_cookies * (4 / 5)
  let grandmother_cookies := cousin_cookies / 3
  let eaten_cookies := father_cookies + mother_cookies + brother_cookies + 
                       sister_cookies + aunt_cookies + cousin_cookies + 
                       grandmother_cookies
  let monica_cookies := total_cookies - eaten_cookies.floor
  monica_cookies = 120

theorem cookies_theorem : cookies_problem 400 := by
  sorry

end NUMINAMATH_CALUDE_cookies_theorem_l395_39599


namespace NUMINAMATH_CALUDE_sabrina_cookies_l395_39555

/-- The number of cookies Sabrina had at the start -/
def initial_cookies : ℕ := 20

/-- The number of cookies Sabrina gave to her brother -/
def cookies_to_brother : ℕ := 10

/-- The number of cookies Sabrina's mother gave her -/
def cookies_from_mother : ℕ := cookies_to_brother / 2

/-- The fraction of cookies Sabrina gave to her sister -/
def fraction_to_sister : ℚ := 2 / 3

/-- The number of cookies Sabrina has left -/
def remaining_cookies : ℕ := 5

theorem sabrina_cookies :
  initial_cookies = cookies_to_brother + 
    (initial_cookies - cookies_to_brother + cookies_from_mother) * (1 - fraction_to_sister) :=
by sorry

end NUMINAMATH_CALUDE_sabrina_cookies_l395_39555


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l395_39522

/-- A line with equal x and y intercepts passing through a given point. -/
structure EqualInterceptLine where
  -- The x-coordinate of the point the line passes through
  x : ℝ
  -- The y-coordinate of the point the line passes through
  y : ℝ
  -- The common intercept value
  a : ℝ
  -- The line passes through the point (x, y)
  point_on_line : x / a + y / a = 1

/-- The equation of a line with equal x and y intercepts passing through (2,1) is x + y - 3 = 0 -/
theorem equal_intercept_line_equation :
  ∀ (l : EqualInterceptLine), l.x = 2 ∧ l.y = 1 → (λ x y => x + y - 3 = 0) = (λ x y => x / l.a + y / l.a = 1) :=
by sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l395_39522


namespace NUMINAMATH_CALUDE_fraction_problem_l395_39588

theorem fraction_problem (N : ℝ) (f : ℝ) : 
  (0.4 * N = 180) → 
  (f * (1/3) * (2/5) * N = 15) → 
  f = 1/4 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l395_39588


namespace NUMINAMATH_CALUDE_democrat_count_l395_39589

theorem democrat_count (total : ℕ) (female : ℕ) (male : ℕ) : 
  total = 870 →
  female + male = total →
  (female / 2 : ℚ) + (male / 4 : ℚ) = total / 3 →
  female / 2 = 145 := by
  sorry

end NUMINAMATH_CALUDE_democrat_count_l395_39589


namespace NUMINAMATH_CALUDE_combinatorial_number_identity_l395_39514

theorem combinatorial_number_identity (n r : ℕ) (h1 : n > r) (h2 : r ≥ 1) :
  Nat.choose n r = (n / r) * Nat.choose (n - 1) (r - 1) := by
  sorry

end NUMINAMATH_CALUDE_combinatorial_number_identity_l395_39514


namespace NUMINAMATH_CALUDE_andrews_to_jeffreys_steps_ratio_l395_39542

theorem andrews_to_jeffreys_steps_ratio : 
  ∃ (a b : ℕ), a ≠ 0 ∧ b ≠ 0 ∧ 150 * b = 200 * a ∧ a = 3 ∧ b = 4 := by
  sorry

end NUMINAMATH_CALUDE_andrews_to_jeffreys_steps_ratio_l395_39542


namespace NUMINAMATH_CALUDE_consecutive_integers_product_812_sum_57_l395_39570

theorem consecutive_integers_product_812_sum_57 :
  ∀ x : ℕ, x > 0 ∧ x * (x + 1) = 812 → x + (x + 1) = 57 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_812_sum_57_l395_39570


namespace NUMINAMATH_CALUDE_fraction_product_cube_specific_fraction_product_l395_39512

theorem fraction_product_cube (a b c d : ℚ) :
  (a / b)^3 * (c / d)^3 = ((a * c) / (b * d))^3 :=
by sorry

theorem specific_fraction_product :
  (8 / 9 : ℚ)^3 * (3 / 5 : ℚ)^3 = 512 / 3375 :=
by sorry

end NUMINAMATH_CALUDE_fraction_product_cube_specific_fraction_product_l395_39512


namespace NUMINAMATH_CALUDE_total_doors_is_3600_l395_39501

/-- Calculates the number of doors needed for a building with uniform floor plans -/
def doorsForUniformBuilding (floors : ℕ) (apartmentsPerFloor : ℕ) (doorsPerApartment : ℕ) : ℕ :=
  floors * apartmentsPerFloor * doorsPerApartment

/-- Calculates the number of doors needed for a building with alternating floor plans -/
def doorsForAlternatingBuilding (floors : ℕ) (oddApartments : ℕ) (evenApartments : ℕ) (doorsPerApartment : ℕ) : ℕ :=
  ((floors + 1) / 2 * oddApartments + (floors / 2) * evenApartments) * doorsPerApartment

/-- The total number of doors needed for all four buildings -/
def totalDoors : ℕ :=
  doorsForUniformBuilding 15 5 8 +
  doorsForUniformBuilding 25 6 10 +
  doorsForAlternatingBuilding 20 7 5 9 +
  doorsForAlternatingBuilding 10 8 4 7

theorem total_doors_is_3600 : totalDoors = 3600 := by
  sorry

end NUMINAMATH_CALUDE_total_doors_is_3600_l395_39501


namespace NUMINAMATH_CALUDE_theater_ticket_contradiction_l395_39552

theorem theater_ticket_contradiction :
  ∀ (adult_price child_price : ℚ) 
    (total_tickets adult_tickets : ℕ) 
    (total_receipts : ℚ),
  adult_price = 12 →
  total_tickets = 130 →
  adult_tickets = 90 →
  total_receipts = 840 →
  ¬(adult_price * adult_tickets + 
    child_price * (total_tickets - adult_tickets) = 
    total_receipts) :=
by
  sorry

#check theater_ticket_contradiction

end NUMINAMATH_CALUDE_theater_ticket_contradiction_l395_39552


namespace NUMINAMATH_CALUDE_hat_problem_inconsistent_l395_39585

/-- Represents the number of hats of each color --/
structure HatCounts where
  blue : ℕ
  green : ℕ
  red : ℕ
  yellow : ℕ

/-- Checks if the given hat counts satisfy the problem conditions --/
def satisfies_conditions (counts : HatCounts) : Prop :=
  counts.blue + counts.green + counts.red + counts.yellow = 150 ∧
  counts.blue = 2 * counts.green ∧
  8 * counts.blue + 10 * counts.green + 12 * counts.red + 15 * counts.yellow = 1280

/-- Theorem stating the inconsistency in the problem --/
theorem hat_problem_inconsistent : 
  ∀ (counts : HatCounts), satisfies_conditions counts → counts.red = 0 ∧ counts.yellow = 0 := by
  sorry

#check hat_problem_inconsistent

end NUMINAMATH_CALUDE_hat_problem_inconsistent_l395_39585


namespace NUMINAMATH_CALUDE_rocky_miles_total_l395_39563

/-- Calculates the total miles run by Rocky in the first three days of training -/
def rocky_miles : ℕ :=
  let day1 : ℕ := 4
  let day2 : ℕ := 2 * day1
  let day3 : ℕ := 3 * day2
  day1 + day2 + day3

theorem rocky_miles_total : rocky_miles = 36 := by
  sorry

end NUMINAMATH_CALUDE_rocky_miles_total_l395_39563


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l395_39595

theorem min_value_trig_expression (x : ℝ) : 
  (Real.sin x)^4 + (Real.cos x)^4 + 2 ≥ (2/3) * ((Real.sin x)^2 + (Real.cos x)^2 + 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l395_39595


namespace NUMINAMATH_CALUDE_sum_c_plus_d_l395_39560

theorem sum_c_plus_d (a b c d : ℝ) 
  (h1 : a + b = 5)
  (h2 : b + c = 6)
  (h3 : a + d = 2) :
  c + d = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_c_plus_d_l395_39560


namespace NUMINAMATH_CALUDE_parabola_c_value_l395_39582

/-- A parabola with equation y = 2x^2 + bx + c passes through the points (1, 4) and (5, 4). -/
theorem parabola_c_value (b c : ℝ) : 
  (4 = 2 * 1^2 + b * 1 + c) → 
  (4 = 2 * 5^2 + b * 5 + c) → 
  c = 14 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l395_39582


namespace NUMINAMATH_CALUDE_line_through_quadrants_l395_39540

/-- A line y = kx + b passes through the second, third, and fourth quadrants if and only if
    k and b satisfy the conditions: k + b = -5 and kb = 6 -/
theorem line_through_quadrants (k b : ℝ) : 
  (∀ x y : ℝ, y = k * x + b → 
    ((x < 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0) ∨ (x > 0 ∧ y < 0))) ↔ 
  (k + b = -5 ∧ k * b = 6) := by
  sorry


end NUMINAMATH_CALUDE_line_through_quadrants_l395_39540


namespace NUMINAMATH_CALUDE_children_on_bus_after_stop_l395_39507

/-- The number of children on a bus after a stop, given the initial number,
    the number who got on, and the relationship between those who got on and off. -/
theorem children_on_bus_after_stop
  (initial : ℕ)
  (got_on : ℕ)
  (h1 : initial = 28)
  (h2 : got_on = 82)
  (h3 : ∃ (got_off : ℕ), got_on = got_off + 2) :
  initial + got_on - (got_on - 2) = 28 := by
  sorry


end NUMINAMATH_CALUDE_children_on_bus_after_stop_l395_39507


namespace NUMINAMATH_CALUDE_chocolate_cost_720_l395_39593

/-- Calculates the cost of buying a certain number of chocolate candies given the following conditions:
  - A box contains 30 chocolate candies
  - A box costs $10
  - If a customer buys more than 20 boxes, they get a 10% discount
-/
def chocolateCost (numCandies : ℕ) : ℚ :=
  let boxSize := 30
  let boxPrice := 10
  let discountThreshold := 20
  let discountRate := 0.1
  let numBoxes := (numCandies + boxSize - 1) / boxSize  -- Ceiling division
  let totalCost := numBoxes * boxPrice
  if numBoxes > discountThreshold then
    totalCost * (1 - discountRate)
  else
    totalCost

theorem chocolate_cost_720 : chocolateCost 720 = 216 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_cost_720_l395_39593


namespace NUMINAMATH_CALUDE_money_left_l395_39508

def initial_amount : ℚ := 200.50
def sweets_cost : ℚ := 35.25
def stickers_cost : ℚ := 10.75
def friend_gift : ℚ := 25.20
def num_friends : ℕ := 4
def charity_donation : ℚ := 15.30

theorem money_left : 
  initial_amount - (sweets_cost + stickers_cost + friend_gift * num_friends + charity_donation) = 38.40 := by
  sorry

end NUMINAMATH_CALUDE_money_left_l395_39508


namespace NUMINAMATH_CALUDE_parabola_max_value_l395_39576

/-- A parabola that opens downward and has its vertex at (2, -3) has a maximum value of -3 -/
theorem parabola_max_value (a b c : ℝ) (h_downward : a < 0) 
  (h_vertex : ∀ x, a * x^2 + b * x + c ≤ a * 2^2 + b * 2 + c) 
  (h_vertex_y : a * 2^2 + b * 2 + c = -3) : 
  ∀ x, a * x^2 + b * x + c ≤ -3 :=
sorry

end NUMINAMATH_CALUDE_parabola_max_value_l395_39576


namespace NUMINAMATH_CALUDE_train_passing_jogger_time_l395_39591

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger_time
  (jogger_speed : ℝ)
  (train_speed : ℝ)
  (train_length : ℝ)
  (initial_distance : ℝ)
  (h1 : jogger_speed = 9 / 3.6)  -- Convert 9 km/hr to m/s
  (h2 : train_speed = 45 / 3.6)  -- Convert 45 km/hr to m/s
  (h3 : train_length = 120)
  (h4 : initial_distance = 180) :
  (initial_distance + train_length) / (train_speed - jogger_speed) = 30 := by
sorry

end NUMINAMATH_CALUDE_train_passing_jogger_time_l395_39591


namespace NUMINAMATH_CALUDE_calculate_second_discount_l395_39592

/-- Given an article with a list price and two successive discounts, 
    calculate the second discount percentage. -/
theorem calculate_second_discount 
  (list_price : ℝ) 
  (first_discount : ℝ) 
  (final_price : ℝ) 
  (h1 : list_price = 70) 
  (h2 : first_discount = 10) 
  (h3 : final_price = 61.74) : 
  ∃ (second_discount : ℝ), 
    final_price = list_price * (1 - first_discount / 100) * (1 - second_discount / 100) ∧ 
    second_discount = 2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_second_discount_l395_39592
