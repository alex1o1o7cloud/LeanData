import Mathlib

namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2567_256707

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 1) + 1
  f 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2567_256707


namespace NUMINAMATH_CALUDE_area_of_R_l2567_256730

-- Define the points and line segments
def A : ℝ × ℝ := (-36, 0)
def B : ℝ × ℝ := (36, 0)
def C : ℝ × ℝ := (0, 0)
def D : ℝ × ℝ := (0, 30)

def AB : Set (ℝ × ℝ) := {p | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p = ((1 - t) • A.1 + t • B.1, 0)}
def CD : Set (ℝ × ℝ) := {p | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p = (0, (1 - t) • C.2 + t • D.2)}

-- Define the region R
def R : Set (ℝ × ℝ) := {p | ∃ x y : ℝ, -36 ≤ x ∧ x ≤ 36 ∧ -30 ≤ y ∧ y ≤ 30 ∧ p = (x/2, y/2)}

-- State the theorem
theorem area_of_R : MeasureTheory.volume R = 1080 := by sorry

end NUMINAMATH_CALUDE_area_of_R_l2567_256730


namespace NUMINAMATH_CALUDE_exists_k_for_all_n_l2567_256792

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Statement of the problem -/
theorem exists_k_for_all_n (n : ℕ) (hn : n > 0) : 
  ∃ k : ℕ, k > 0 ∧ sumOfDigits k = n ∧ sumOfDigits (k^2) = n^2 := by sorry

end NUMINAMATH_CALUDE_exists_k_for_all_n_l2567_256792


namespace NUMINAMATH_CALUDE_cosine_sine_sum_equality_l2567_256753

theorem cosine_sine_sum_equality : 
  Real.cos (43 * π / 180) * Real.cos (13 * π / 180) + 
  Real.sin (43 * π / 180) * Real.sin (13 * π / 180) = 
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_sum_equality_l2567_256753


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l2567_256748

theorem cube_root_equation_solution :
  ∃ x : ℝ, x = 1/11 ∧ (5 + 2/x)^(1/3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l2567_256748


namespace NUMINAMATH_CALUDE_min_value_expression_l2567_256789

theorem min_value_expression (a b c : ℝ) 
  (ha : 0 ≤ a ∧ a < 1) (hb : 0 ≤ b ∧ b < 1) (hc : 0 ≤ c ∧ c < 1) : 
  (1 / ((2 - a) * (2 - b) * (2 - c))) + (1 / ((2 + a) * (2 + b) * (2 + c))) ≥ 1/8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2567_256789


namespace NUMINAMATH_CALUDE_min_value_of_x_plus_four_over_x_squared_min_value_achieved_l2567_256796

theorem min_value_of_x_plus_four_over_x_squared (x : ℝ) (h : x > 0) :
  x + 4 / x^2 ≥ 3 :=
sorry

theorem min_value_achieved (x : ℝ) (h : x > 0) :
  x + 4 / x^2 = 3 ↔ x = 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_x_plus_four_over_x_squared_min_value_achieved_l2567_256796


namespace NUMINAMATH_CALUDE_no_common_terms_l2567_256794

-- Define the sequences a_n and b_n
def a_n (a : ℝ) (n : ℕ) : ℝ := a * n + 2
def b_n (b : ℝ) (n : ℕ) : ℝ := b * n + 1

-- Theorem statement
theorem no_common_terms (a b : ℝ) (h : a > b) :
  ∀ n : ℕ, a_n a n ≠ b_n b n := by
  sorry

end NUMINAMATH_CALUDE_no_common_terms_l2567_256794


namespace NUMINAMATH_CALUDE_parabola_equation_correct_l2567_256770

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the equation of a line in the form ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents the equation of a parabola in the form ax^2 + bxy + cy^2 + dx + ey + f = 0 -/
structure ParabolaEq where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ

/-- The focus of the parabola -/
def focus : Point := ⟨5, 2⟩

/-- The directrix of the parabola -/
def directrix : Line := ⟨5, 2, 25⟩

/-- The equation of the parabola -/
def parabolaEq : ParabolaEq := ⟨4, -20, 25, -40, -16, -509⟩

/-- Checks if the given parabola equation satisfies the conditions -/
def isValidParabolaEq (eq : ParabolaEq) : Prop :=
  eq.a > 0 ∧ Int.gcd eq.a.natAbs (Int.gcd eq.b.natAbs (Int.gcd eq.c.natAbs (Int.gcd eq.d.natAbs (Int.gcd eq.e.natAbs eq.f.natAbs)))) = 1

/-- Theorem stating that the given parabola equation is correct and satisfies the conditions -/
theorem parabola_equation_correct :
  isValidParabolaEq parabolaEq ∧
  ∀ (p : Point),
    (p.x - focus.x)^2 + (p.y - focus.y)^2 = 
    ((directrix.a * p.x + directrix.b * p.y - directrix.c)^2) / (directrix.a^2 + directrix.b^2) ↔
    parabolaEq.a * p.x^2 + parabolaEq.b * p.x * p.y + parabolaEq.c * p.y^2 + 
    parabolaEq.d * p.x + parabolaEq.e * p.y + parabolaEq.f = 0 := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_correct_l2567_256770


namespace NUMINAMATH_CALUDE_article_cost_l2567_256752

theorem article_cost (C : ℝ) (S : ℝ) : 
  S = 1.25 * C →                            -- 25% profit
  (0.8 * C + 0.3 * (0.8 * C) = S - 10.50) → -- 30% profit on reduced cost and price
  C = 50 := by
sorry

end NUMINAMATH_CALUDE_article_cost_l2567_256752


namespace NUMINAMATH_CALUDE_smallest_k_inequality_l2567_256761

theorem smallest_k_inequality (a b c : ℕ+) : 
  ∃ (k : ℕ+), k = 1297 ∧ 
  (16 * a^2 + 36 * b^2 + 81 * c^2) * (81 * a^2 + 36 * b^2 + 16 * c^2) < k * (a^2 + b^2 + c^2)^2 ∧
  ∀ (m : ℕ+), m < 1297 → 
  (16 * a^2 + 36 * b^2 + 81 * c^2) * (81 * a^2 + 36 * b^2 + 16 * c^2) ≥ m * (a^2 + b^2 + c^2)^2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_inequality_l2567_256761


namespace NUMINAMATH_CALUDE_cube_root_equation_sum_l2567_256774

theorem cube_root_equation_sum (x y z : ℕ+) :
  (4 : ℝ) * ((7 : ℝ)^(1/3) - (6 : ℝ)^(1/3))^(1/2) = (x : ℝ)^(1/3) + (y : ℝ)^(1/3) - (z : ℝ)^(1/3) →
  x + y + z = 75 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_sum_l2567_256774


namespace NUMINAMATH_CALUDE_sin_double_angle_special_case_l2567_256787

theorem sin_double_angle_special_case (φ : ℝ) :
  (7 : ℝ) / 13 + Real.sin φ = Real.cos φ →
  Real.sin (2 * φ) = 120 / 169 := by
sorry

end NUMINAMATH_CALUDE_sin_double_angle_special_case_l2567_256787


namespace NUMINAMATH_CALUDE_double_area_rectangle_l2567_256768

/-- The area of a rectangle with dimensions 50 cm × 160 cm is exactly double
    the area of a rectangle with dimensions 50 cm × 80 cm. -/
theorem double_area_rectangle : 
  ∀ (width height new_height : ℕ), 
    width = 50 → height = 80 → new_height = 160 →
    2 * (width * height) = width * new_height := by
  sorry

end NUMINAMATH_CALUDE_double_area_rectangle_l2567_256768


namespace NUMINAMATH_CALUDE_expression_value_l2567_256750

theorem expression_value (a b c d x : ℝ) : 
  (c / 3 = -(-2 * d)) →
  (2 * a = 1 / (-b)) →
  (|x| = 9) →
  (2 * a * b - 6 * d + c - x / 3 = -4 ∨ 2 * a * b - 6 * d + c - x / 3 = 2) :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l2567_256750


namespace NUMINAMATH_CALUDE_product_and_sum_of_three_two_digit_integers_l2567_256713

theorem product_and_sum_of_three_two_digit_integers : ∃ (a b c : ℕ), 
  10 ≤ a ∧ a < 100 ∧
  10 ≤ b ∧ b < 100 ∧
  10 ≤ c ∧ c < 100 ∧
  a * b * c = 636405 ∧
  a + b + c = 259 := by
sorry

end NUMINAMATH_CALUDE_product_and_sum_of_three_two_digit_integers_l2567_256713


namespace NUMINAMATH_CALUDE_smallest_winning_number_l2567_256759

theorem smallest_winning_number : ∃ (N : ℕ), 
  (0 ≤ N ∧ N ≤ 1999) ∧ 
  (∃ (k : ℕ), 1900 ≤ 2 * N + 100 * k ∧ 2 * N + 100 * k ≤ 1999) ∧
  (∀ (M : ℕ), M < N → ¬∃ (j : ℕ), 1900 ≤ 2 * M + 100 * j ∧ 2 * M + 100 * j ≤ 1999) ∧
  N = 800 := by
  sorry

end NUMINAMATH_CALUDE_smallest_winning_number_l2567_256759


namespace NUMINAMATH_CALUDE_sector_area_one_radian_unit_radius_l2567_256727

/-- The area of a circular sector with central angle 1 radian and radius 1 unit is 1/2 square units. -/
theorem sector_area_one_radian_unit_radius : 
  let θ : Real := 1  -- Central angle in radians
  let r : Real := 1  -- Radius
  let sector_area := (1/2) * r * r * θ
  sector_area = 1/2 := by sorry

end NUMINAMATH_CALUDE_sector_area_one_radian_unit_radius_l2567_256727


namespace NUMINAMATH_CALUDE_angle_terminal_side_l2567_256733

theorem angle_terminal_side (α : Real) (x : Real) :
  (∃ P : Real × Real, P = (x, 4) ∧ P.1 = x * Real.cos α ∧ P.2 = 4 * Real.sin α) →
  Real.sin α = 4/5 →
  x = 3 ∨ x = -3 := by
sorry

end NUMINAMATH_CALUDE_angle_terminal_side_l2567_256733


namespace NUMINAMATH_CALUDE_probability_of_common_books_l2567_256765

def total_books : ℕ := 12
def books_to_choose : ℕ := 6
def common_books : ℕ := 3

theorem probability_of_common_books :
  (Nat.choose total_books common_books * 
   Nat.choose (total_books - common_books) (books_to_choose - common_books) * 
   Nat.choose (total_books - common_books) (books_to_choose - common_books)) / 
  (Nat.choose total_books books_to_choose * Nat.choose total_books books_to_choose) = 220 / 1215 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_common_books_l2567_256765


namespace NUMINAMATH_CALUDE_retail_price_calculation_l2567_256763

/-- The retail price of a machine, given wholesale price, discount, and profit margin -/
theorem retail_price_calculation (wholesale_price discount profit_margin : ℚ) 
  (h1 : wholesale_price = 126)
  (h2 : discount = 10/100)
  (h3 : profit_margin = 20/100)
  (h4 : profit_margin * wholesale_price + wholesale_price = (1 - discount) * retail_price) :
  retail_price = 168 := by
  sorry

#check retail_price_calculation

end NUMINAMATH_CALUDE_retail_price_calculation_l2567_256763


namespace NUMINAMATH_CALUDE_ball_probabilities_l2567_256739

/-- Represents the total number of balls in the bag -/
def total_balls : ℕ := 6

/-- Represents the number of white balls in the bag -/
def white_balls : ℕ := 4

/-- Represents the number of red balls in the bag -/
def red_balls : ℕ := 2

/-- Represents the number of balls drawn -/
def drawn_balls : ℕ := 2

/-- Calculates the probability of drawing two red balls -/
def prob_two_red : ℚ := (red_balls * (red_balls - 1)) / (total_balls * (total_balls - 1))

/-- Calculates the probability of drawing at least one red ball -/
def prob_at_least_one_red : ℚ := 1 - (white_balls * (white_balls - 1)) / (total_balls * (total_balls - 1))

theorem ball_probabilities :
  prob_two_red = 1/15 ∧ prob_at_least_one_red = 3/5 := by sorry

end NUMINAMATH_CALUDE_ball_probabilities_l2567_256739


namespace NUMINAMATH_CALUDE_gcd_problem_l2567_256762

theorem gcd_problem (h : Prime 103) : 
  Nat.gcd (103^7 + 1) (103^7 + 103^5 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l2567_256762


namespace NUMINAMATH_CALUDE_larger_variance_greater_fluctuation_l2567_256729

-- Define a type for our data set
def DataSet := List ℝ

-- Define variance for a data set
def variance (data : DataSet) : ℝ := sorry

-- Define a measure of fluctuation for a data set
def fluctuation (data : DataSet) : ℝ := sorry

-- Theorem stating that larger variance implies greater fluctuation
theorem larger_variance_greater_fluctuation 
  (data1 data2 : DataSet) :
  variance data1 > variance data2 → fluctuation data1 > fluctuation data2 := by
  sorry

end NUMINAMATH_CALUDE_larger_variance_greater_fluctuation_l2567_256729


namespace NUMINAMATH_CALUDE_coefficient_expansion_l2567_256716

theorem coefficient_expansion (m : ℝ) : 
  (∃ c : ℝ, c = -160 ∧ c = 20 * m^3) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_expansion_l2567_256716


namespace NUMINAMATH_CALUDE_five_by_five_grid_properties_l2567_256781

/-- Represents a square grid --/
structure Grid :=
  (size : ℕ)

/-- Counts the number of squares in a grid --/
def count_squares (g : Grid) : ℕ :=
  sorry

/-- Counts the number of pairs of parallel lines in a grid --/
def count_parallel_pairs (g : Grid) : ℕ :=
  sorry

/-- Counts the number of rectangles in a grid --/
def count_rectangles (g : Grid) : ℕ :=
  sorry

/-- Theorem stating the properties of a 5x5 grid --/
theorem five_by_five_grid_properties :
  let g : Grid := ⟨5⟩
  count_squares g = 55 ∧
  count_parallel_pairs g = 30 ∧
  count_rectangles g = 225 :=
by sorry

end NUMINAMATH_CALUDE_five_by_five_grid_properties_l2567_256781


namespace NUMINAMATH_CALUDE_curve_not_parabola_l2567_256701

/-- The equation of the curve -/
def curve_equation (m : ℝ) (x y : ℝ) : Prop :=
  m * x^2 + (m + 1) * y^2 = m * (m + 1)

/-- Definition of a parabola in general form -/
def is_parabola (f : ℝ → ℝ → Prop) : Prop :=
  ∃ a b c d e : ℝ, a ≠ 0 ∧
    ∀ x y, f x y ↔ a * x^2 + b * x * y + c * y^2 + d * x + e * y = 0

/-- Theorem: The curve cannot be a parabola -/
theorem curve_not_parabola :
  ∀ m : ℝ, ¬(is_parabola (curve_equation m)) :=
sorry

end NUMINAMATH_CALUDE_curve_not_parabola_l2567_256701


namespace NUMINAMATH_CALUDE_pascal_triangle_probability_l2567_256700

/-- Represents Pascal's Triangle up to a given number of rows -/
def PascalTriangle (n : ℕ) : List (List ℕ) :=
  sorry

/-- Counts the number of elements equal to a given value in Pascal's Triangle -/
def countElementsEqual (triangle : List (List ℕ)) (value : ℕ) : ℕ :=
  sorry

/-- Calculates the total number of elements in Pascal's Triangle -/
def totalElements (triangle : List (List ℕ)) : ℕ :=
  sorry

theorem pascal_triangle_probability (n : ℕ) :
  n = 20 →
  let triangle := PascalTriangle n
  let ones := countElementsEqual triangle 1
  let twos := countElementsEqual triangle 2
  let total := totalElements triangle
  (ones + twos : ℚ) / total = 57 / 210 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_probability_l2567_256700


namespace NUMINAMATH_CALUDE_intersection_y_intercept_sum_l2567_256731

/-- Given two lines that intersect at (3,6), prove that the sum of their y-intercepts is 6 -/
theorem intersection_y_intercept_sum (c d : ℝ) : 
  (3 = (1/3) * 6 + c) →   -- First line passes through (3,6)
  (6 = (1/3) * 3 + d) →   -- Second line passes through (3,6)
  c + d = 6 := by
sorry

end NUMINAMATH_CALUDE_intersection_y_intercept_sum_l2567_256731


namespace NUMINAMATH_CALUDE_expected_jumps_is_eight_l2567_256724

/-- Represents the behavior of a trainer --/
structure Trainer where
  jumps : ℕ
  gives_treat : Bool

/-- The expected number of jumps before getting a treat --/
def expected_jumps (trainers : List Trainer) : ℝ :=
  sorry

/-- The list of trainers with their behaviors --/
def dog_trainers : List Trainer :=
  [{ jumps := 0, gives_treat := true },
   { jumps := 5, gives_treat := true },
   { jumps := 3, gives_treat := false }]

/-- The main theorem stating the expected number of jumps --/
theorem expected_jumps_is_eight :
  expected_jumps dog_trainers = 8 := by
  sorry

end NUMINAMATH_CALUDE_expected_jumps_is_eight_l2567_256724


namespace NUMINAMATH_CALUDE_points_per_enemy_is_10_l2567_256783

/-- The number of points for killing one enemy in Tom's game -/
def points_per_enemy : ℕ := sorry

/-- The number of enemies Tom killed -/
def enemies_killed : ℕ := 150

/-- Tom's total score -/
def total_score : ℕ := 2250

/-- The bonus multiplier for killing at least 100 enemies -/
def bonus_multiplier : ℚ := 1.5

theorem points_per_enemy_is_10 :
  points_per_enemy = 10 ∧
  enemies_killed ≥ 100 ∧
  (points_per_enemy * enemies_killed : ℚ) * bonus_multiplier = total_score := by
  sorry

end NUMINAMATH_CALUDE_points_per_enemy_is_10_l2567_256783


namespace NUMINAMATH_CALUDE_function_equality_l2567_256703

theorem function_equality (f g : ℕ+ → ℕ+) 
  (h1 : ∀ n : ℕ+, f (g n) = f n + 1) 
  (h2 : ∀ n : ℕ+, g (f n) = g n + 1) : 
  ∀ n : ℕ+, f n = g n := by
sorry

end NUMINAMATH_CALUDE_function_equality_l2567_256703


namespace NUMINAMATH_CALUDE_inequality_proof_l2567_256769

theorem inequality_proof (a b : ℝ) (n : ℤ) (ha : a > 0) (hb : b > 0) :
  (1 + a / b) ^ n + (1 + b / a) ^ n ≥ 2^(n + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2567_256769


namespace NUMINAMATH_CALUDE_bill_work_hours_l2567_256744

/-- Calculates the total pay for a given number of hours worked, 
    with a base rate for the first 40 hours and a double rate thereafter. -/
def calculatePay (baseRate : ℕ) (hours : ℕ) : ℕ :=
  if hours ≤ 40 then
    baseRate * hours
  else
    baseRate * 40 + baseRate * 2 * (hours - 40)

/-- Proves that working 50 hours results in a total pay of $1200, 
    given the specified pay rates. -/
theorem bill_work_hours (baseRate : ℕ) (totalPay : ℕ) :
  baseRate = 20 → totalPay = 1200 → ∃ hours, calculatePay baseRate hours = totalPay ∧ hours = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_bill_work_hours_l2567_256744


namespace NUMINAMATH_CALUDE_dana_total_earnings_l2567_256754

/-- Dana's hourly wage in dollars -/
def hourly_wage : ℕ := 13

/-- Hours worked on Friday -/
def friday_hours : ℕ := 9

/-- Hours worked on Saturday -/
def saturday_hours : ℕ := 10

/-- Hours worked on Sunday -/
def sunday_hours : ℕ := 3

/-- Calculate total earnings given hourly wage and hours worked -/
def total_earnings (wage : ℕ) (hours_fri hours_sat hours_sun : ℕ) : ℕ :=
  wage * (hours_fri + hours_sat + hours_sun)

/-- Theorem stating Dana's total earnings -/
theorem dana_total_earnings :
  total_earnings hourly_wage friday_hours saturday_hours sunday_hours = 286 := by
  sorry

end NUMINAMATH_CALUDE_dana_total_earnings_l2567_256754


namespace NUMINAMATH_CALUDE_goldfish_preference_l2567_256715

theorem goldfish_preference (total_students : ℕ) (johnson_fraction : ℚ) (henderson_fraction : ℚ) (total_preference : ℕ) :
  total_students = 30 →
  johnson_fraction = 1/6 →
  henderson_fraction = 1/5 →
  total_preference = 31 →
  ∃ feldstein_fraction : ℚ,
    feldstein_fraction = 2/3 ∧
    total_preference = johnson_fraction * total_students + henderson_fraction * total_students + feldstein_fraction * total_students :=
by
  sorry

end NUMINAMATH_CALUDE_goldfish_preference_l2567_256715


namespace NUMINAMATH_CALUDE_three_digit_divisible_by_17_l2567_256712

theorem three_digit_divisible_by_17 : 
  (Finset.filter (fun k : ℕ => 100 ≤ 17 * k ∧ 17 * k ≤ 999) (Finset.range 1000)).card = 53 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_divisible_by_17_l2567_256712


namespace NUMINAMATH_CALUDE_sum_m_n_equals_five_l2567_256717

theorem sum_m_n_equals_five (m n : ℕ) (a b : ℝ) 
  (h1 : a > 0) (h2 : b > 0) 
  (h3 : a = n * b) (h4 : a + b = m * (a - b)) : 
  m + n = 5 :=
sorry

end NUMINAMATH_CALUDE_sum_m_n_equals_five_l2567_256717


namespace NUMINAMATH_CALUDE_triangle_area_l2567_256772

theorem triangle_area (a b : ℝ) (C : ℝ) (h1 : a = 7) (h2 : b = 11) (h3 : C = 60 * π / 180) :
  (1 / 2) * a * b * Real.sin C = (77 * Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2567_256772


namespace NUMINAMATH_CALUDE_coat_price_l2567_256719

/-- The final price of a coat after discounts and tax -/
def finalPrice (originalPrice discountOne discountTwo coupon salesTax : ℚ) : ℚ :=
  ((originalPrice * (1 - discountOne) * (1 - discountTwo) - coupon) * (1 + salesTax))

/-- Theorem stating the final price of the coat -/
theorem coat_price : 
  finalPrice 150 0.3 0.1 10 0.05 = 88.725 := by sorry

end NUMINAMATH_CALUDE_coat_price_l2567_256719


namespace NUMINAMATH_CALUDE_sharp_composition_l2567_256743

def sharp (N : ℝ) : ℝ := 0.4 * N * 1.5

theorem sharp_composition : sharp (sharp (sharp 80)) = 17.28 := by
  sorry

end NUMINAMATH_CALUDE_sharp_composition_l2567_256743


namespace NUMINAMATH_CALUDE_crabapple_theorem_l2567_256725

/-- The number of possible sequences of crabapple recipients in a week -/
def crabapple_sequences (num_students : ℕ) (classes_per_week : ℕ) : ℕ :=
  num_students ^ classes_per_week

/-- Theorem stating the number of possible sequences for Mrs. Crabapple's class -/
theorem crabapple_theorem :
  crabapple_sequences 15 5 = 759375 := by
  sorry

end NUMINAMATH_CALUDE_crabapple_theorem_l2567_256725


namespace NUMINAMATH_CALUDE_special_function_inequality_l2567_256728

/-- A function satisfying the given differential inequality -/
structure SpecialFunction where
  f : ℝ → ℝ
  diff : Differentiable ℝ f
  domain : ∀ x, x < 0 → f x ≠ 0
  ineq : ∀ x, x < 0 → 2 * f x + x * deriv f x > x^2

/-- The main theorem -/
theorem special_function_inequality (φ : SpecialFunction) :
  ∀ x, (x + 2016)^2 * φ.f (x + 2016) - 4 * φ.f (-2) > 0 ↔ x < -2018 :=
sorry

end NUMINAMATH_CALUDE_special_function_inequality_l2567_256728


namespace NUMINAMATH_CALUDE_geometric_sequence_increasing_iff_135_l2567_256737

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- The sequence is positive -/
def positive_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n > 0

/-- The sequence is increasing -/
def increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) > a n

/-- The condition a_1 < a_3 < a_5 -/
def condition_135 (a : ℕ → ℝ) : Prop :=
  a 1 < a 3 ∧ a 3 < a 5

theorem geometric_sequence_increasing_iff_135 
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_pos : positive_sequence a) :
  increasing_sequence a ↔ condition_135 a :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_increasing_iff_135_l2567_256737


namespace NUMINAMATH_CALUDE_sequence_property_l2567_256741

theorem sequence_property (a : ℕ → ℤ) 
  (h : ∀ n : ℕ, a 1 + a (2 * n - 1) = 4 * n - 6) : 
  a 7 = 11 := by
sorry

end NUMINAMATH_CALUDE_sequence_property_l2567_256741


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2567_256711

theorem inequality_solution_set : 
  ∀ x : ℝ, -x^2 + 3*x + 4 > 0 ↔ -1 < x ∧ x < 4 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2567_256711


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2567_256740

theorem triangle_perimeter (a b c : ℝ) : 
  a = 2 → (b - 2)^2 + |c - 3| = 0 → a + b + c = 7 := by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l2567_256740


namespace NUMINAMATH_CALUDE_pencil_length_l2567_256760

theorem pencil_length : ∀ (L : ℝ), 
  (L > 0) →                          -- Ensure positive length
  ((1/8) * L + (1/2) * (7/8) * L + 7/2 = L) →  -- Parts sum to total
  (L = 8) :=                         -- Total length is 8
by
  sorry

end NUMINAMATH_CALUDE_pencil_length_l2567_256760


namespace NUMINAMATH_CALUDE_lcm_factor_proof_l2567_256757

theorem lcm_factor_proof (A B : ℕ+) (X : ℕ) : 
  Nat.gcd A B = 23 →
  Nat.lcm A B = 23 * 15 * X →
  A = 368 →
  X = 16 := by
sorry

end NUMINAMATH_CALUDE_lcm_factor_proof_l2567_256757


namespace NUMINAMATH_CALUDE_simplified_fourth_root_l2567_256779

theorem simplified_fourth_root (a b : ℕ+) :
  (2^9 * 3^5 : ℝ)^(1/4) = (a : ℝ) * ((b : ℝ)^(1/4)) → a + b = 18 := by
  sorry

end NUMINAMATH_CALUDE_simplified_fourth_root_l2567_256779


namespace NUMINAMATH_CALUDE_cube_edge_length_from_paint_cost_l2567_256751

/-- Proves that a cube with a specific edge length costs $1.60 to paint given certain paint properties -/
theorem cube_edge_length_from_paint_cost 
  (paint_cost_per_quart : ℝ) 
  (paint_coverage_per_quart : ℝ) 
  (total_paint_cost : ℝ) : 
  paint_cost_per_quart = 3.20 →
  paint_coverage_per_quart = 1200 →
  total_paint_cost = 1.60 →
  ∃ (edge_length : ℝ), 
    edge_length = 10 ∧ 
    total_paint_cost = (6 * edge_length^2) / paint_coverage_per_quart * paint_cost_per_quart :=
by
  sorry


end NUMINAMATH_CALUDE_cube_edge_length_from_paint_cost_l2567_256751


namespace NUMINAMATH_CALUDE_geometry_theorem_l2567_256790

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicularLP : Line → Plane → Prop)
variable (perpendicularPP : Plane → Plane → Prop)
variable (intersects : Line → Plane → Prop)
variable (distinct : Line → Line → Prop)
variable (distinctP : Plane → Plane → Prop)

-- Define the lines and planes
variable (m n : Line)
variable (α β : Plane)

-- State the theorem
theorem geometry_theorem 
  (h_distinct_lines : distinct m n)
  (h_distinct_planes : distinctP α β) :
  (perpendicularPP α β ∧ perpendicularLP m α → ¬(intersects m β)) ∧
  (perpendicular m n ∧ perpendicularLP m α → ¬(intersects n α)) :=
sorry

end NUMINAMATH_CALUDE_geometry_theorem_l2567_256790


namespace NUMINAMATH_CALUDE_parking_theorem_l2567_256799

/-- Represents the number of empty parking spaces -/
def total_spaces : ℕ := 10

/-- Represents the number of cars to be parked -/
def num_cars : ℕ := 3

/-- Represents the number of empty spaces required between cars -/
def spaces_between : ℕ := 1

/-- Calculates the number of parking arrangements given the constraints -/
def parking_arrangements (total : ℕ) (cars : ℕ) (spaces : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number of parking arrangements is 40 -/
theorem parking_theorem : 
  parking_arrangements total_spaces num_cars spaces_between = 40 :=
sorry

end NUMINAMATH_CALUDE_parking_theorem_l2567_256799


namespace NUMINAMATH_CALUDE_circle_properties_l2567_256710

theorem circle_properties (k : ℚ) : 
  let circle_eq (x y : ℚ) := x^2 + 2*x + y^2 = 1992
  ∃ (x y : ℚ), 
    circle_eq 42 12 ∧ 
    circle_eq x y ∧ 
    y - 12 = k * (x - 42) := by
  sorry

end NUMINAMATH_CALUDE_circle_properties_l2567_256710


namespace NUMINAMATH_CALUDE_percent_difference_z_x_of_w_l2567_256738

theorem percent_difference_z_x_of_w (y q w x z : ℝ) 
  (hw : w = 0.60 * q)
  (hq : q = 0.60 * y)
  (hz : z = 0.54 * y)
  (hx : x = 1.30 * w) :
  (z - x) / w = 0.20 := by
sorry

end NUMINAMATH_CALUDE_percent_difference_z_x_of_w_l2567_256738


namespace NUMINAMATH_CALUDE_max_sqrt_sum_l2567_256714

theorem max_sqrt_sum (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 20) :
  Real.sqrt (x + 36) + Real.sqrt (20 - x) + 2 * Real.sqrt x ≤ Real.sqrt 261 := by
  sorry

end NUMINAMATH_CALUDE_max_sqrt_sum_l2567_256714


namespace NUMINAMATH_CALUDE_cinema_uses_systematic_sampling_l2567_256764

/-- Represents a sampling method --/
inductive SamplingMethod
| Lottery
| Stratified
| RandomNumberTable
| Systematic

/-- Represents a cinema with rows and seats per row --/
structure Cinema where
  rows : Nat
  seatsPerRow : Nat

/-- Represents a selection rule for seats --/
structure SelectionRule where
  endDigit : Nat

/-- Determines if a sampling method is systematic based on cinema layout and selection rule --/
def isSystematicSampling (c : Cinema) (r : SelectionRule) : Prop :=
  r.endDigit = c.seatsPerRow % 10 ∧ c.seatsPerRow % 10 ≠ 0

/-- Theorem stating that the given cinema scenario uses systematic sampling --/
theorem cinema_uses_systematic_sampling (c : Cinema) (r : SelectionRule) :
  c.rows = 50 → c.seatsPerRow = 30 → r.endDigit = 8 →
  isSystematicSampling c r ∧ SamplingMethod.Systematic = SamplingMethod.Systematic := by
  sorry

end NUMINAMATH_CALUDE_cinema_uses_systematic_sampling_l2567_256764


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2567_256791

theorem imaginary_part_of_complex_fraction (i : ℂ) (h : i * i = -1) :
  let z : ℂ := (2 + i) / i
  Complex.im z = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2567_256791


namespace NUMINAMATH_CALUDE_xiaoyings_journey_equations_correct_l2567_256771

/-- Represents a journey with uphill and downhill sections -/
structure Journey where
  total_distance : ℝ
  total_time : ℝ
  uphill_speed : ℝ
  downhill_speed : ℝ

/-- Xiaoying's journey to school -/
def xiaoyings_journey : Journey where
  total_distance := 1.2  -- 1200 meters converted to kilometers
  total_time := 16
  uphill_speed := 3
  downhill_speed := 5

/-- The system of equations representing Xiaoying's journey -/
def journey_equations (j : Journey) (x y : ℝ) : Prop :=
  (j.uphill_speed / 60 * x + j.downhill_speed / 60 * y = j.total_distance) ∧
  (x + y = j.total_time)

theorem xiaoyings_journey_equations_correct :
  journey_equations xiaoyings_journey = λ x y ↦ 
    (3 / 60 * x + 5 / 60 * y = 1.2) ∧ (x + y = 16) := by sorry

end NUMINAMATH_CALUDE_xiaoyings_journey_equations_correct_l2567_256771


namespace NUMINAMATH_CALUDE_jose_profit_share_l2567_256742

/-- Calculates the share of profit for an investor given their investment amount, 
    investment duration, total investment-months, and total profit. -/
def shareOfProfit (investment : ℕ) (duration : ℕ) (totalInvestmentMonths : ℕ) (totalProfit : ℕ) : ℚ :=
  (investment * duration : ℚ) / totalInvestmentMonths * totalProfit

theorem jose_profit_share 
  (tom_investment : ℕ) (tom_duration : ℕ) 
  (jose_investment : ℕ) (jose_duration : ℕ) 
  (total_profit : ℕ) : 
  tom_investment = 30000 → 
  tom_duration = 12 → 
  jose_investment = 45000 → 
  jose_duration = 10 → 
  total_profit = 45000 → 
  shareOfProfit jose_investment jose_duration 
    (tom_investment * tom_duration + jose_investment * jose_duration) total_profit = 25000 := by
  sorry

#check jose_profit_share

end NUMINAMATH_CALUDE_jose_profit_share_l2567_256742


namespace NUMINAMATH_CALUDE_jose_wandering_time_l2567_256749

/-- Given a distance of 4 kilometers and a speed of 2 kilometers per hour,
    the time taken is 2 hours. -/
theorem jose_wandering_time :
  let distance : ℝ := 4  -- Distance in kilometers
  let speed : ℝ := 2     -- Speed in kilometers per hour
  let time := distance / speed
  time = 2 := by sorry

end NUMINAMATH_CALUDE_jose_wandering_time_l2567_256749


namespace NUMINAMATH_CALUDE_trig_identity_l2567_256766

theorem trig_identity : (2 * Real.cos (10 * π / 180) - Real.sin (20 * π / 180)) / Real.sin (70 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2567_256766


namespace NUMINAMATH_CALUDE_candy_bar_profit_l2567_256758

/-- Represents the candy bar sale problem -/
structure CandyBarSale where
  total_bars : ℕ
  bulk_price : ℚ
  bulk_quantity : ℕ
  regular_price : ℚ
  regular_quantity : ℕ
  selling_price : ℚ
  selling_quantity : ℕ

/-- Calculates the profit for the candy bar sale -/
def calculate_profit (sale : CandyBarSale) : ℚ :=
  let cost_per_bar := sale.bulk_price / sale.bulk_quantity
  let total_cost := cost_per_bar * sale.total_bars
  let revenue_per_bar := sale.selling_price / sale.selling_quantity
  let total_revenue := revenue_per_bar * sale.total_bars
  total_revenue - total_cost

/-- The main theorem stating that the profit is $350 -/
theorem candy_bar_profit :
  let sale : CandyBarSale := {
    total_bars := 1200,
    bulk_price := 3,
    bulk_quantity := 8,
    regular_price := 2,
    regular_quantity := 5,
    selling_price := 2,
    selling_quantity := 3
  }
  calculate_profit sale = 350 := by
  sorry


end NUMINAMATH_CALUDE_candy_bar_profit_l2567_256758


namespace NUMINAMATH_CALUDE_smallest_d_for_inverse_l2567_256726

/-- The function g(x) = (x - 3)^2 - 7 -/
def g (x : ℝ) : ℝ := (x - 3)^2 - 7

/-- The property of being strictly increasing on an interval -/
def StrictlyIncreasing (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y → f x < f y

/-- The smallest value of d for which g has an inverse function on [d, ∞) -/
theorem smallest_d_for_inverse : 
  (∃ d : ℝ, StrictlyIncreasing g d ∧ 
    (∀ c : ℝ, c < d → ¬StrictlyIncreasing g c)) ∧ 
  (∀ d : ℝ, StrictlyIncreasing g d → d ≥ 3) ∧
  StrictlyIncreasing g 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_d_for_inverse_l2567_256726


namespace NUMINAMATH_CALUDE_mens_wages_proof_l2567_256793

-- Define the number of men, women, and boys
def num_men : ℕ := 5
def num_boys : ℕ := 8

-- Define the total earnings
def total_earnings : ℚ := 90

-- Define the relationship between men, women, and boys
axiom men_women_equality : ∃ w : ℕ, num_men = w
axiom women_boys_equality : ∃ w : ℕ, w = num_boys

-- Define the theorem
theorem mens_wages_proof :
  ∃ (wage_man wage_woman wage_boy : ℚ),
    wage_man > 0 ∧ wage_woman > 0 ∧ wage_boy > 0 ∧
    num_men * wage_man + num_boys * wage_boy + num_boys * wage_woman = total_earnings ∧
    num_men * wage_man = 30 :=
sorry

end NUMINAMATH_CALUDE_mens_wages_proof_l2567_256793


namespace NUMINAMATH_CALUDE_minimum_value_of_sum_of_reciprocals_l2567_256709

theorem minimum_value_of_sum_of_reciprocals (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : Real.sqrt 3 = Real.sqrt (3^a * 3^b)) : 
  (∀ x y, x > 0 → y > 0 → 1/x + 1/y ≥ 1/a + 1/b) → 1/a + 1/b = 4 :=
sorry

end NUMINAMATH_CALUDE_minimum_value_of_sum_of_reciprocals_l2567_256709


namespace NUMINAMATH_CALUDE_ellipse_chord_fixed_point_l2567_256788

/-- The fixed point theorem for ellipse chords -/
theorem ellipse_chord_fixed_point 
  (a b A B : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hab : a > b) 
  (hAB : A ≠ 0 ∧ B ≠ 0) :
  ∃ M : ℝ × ℝ, ∀ P : ℝ × ℝ,
    (A * P.1 + B * P.2 = 1) →  -- P is on line l
    ∃ Q R : ℝ × ℝ,
      (R.1^2 / a^2 + R.2^2 / b^2 = 1) ∧  -- R is on ellipse Γ
      (∃ t : ℝ, Q = ⟨t * P.1, t * P.2⟩) ∧  -- Q is on ray OP
      (Q.1^2 + Q.2^2) * (P.1^2 + P.2^2) = (R.1^2 + R.2^2)^2 →  -- |OQ| * |OP| = |OR|^2
      ∃ l_P : Set (ℝ × ℝ),
        (∀ X ∈ l_P, ∃ s : ℝ, X = ⟨s * Q.1, s * Q.2⟩) ∧  -- l_P is a line through Q
        M ∈ l_P ∧  -- M is on l_P
        M = (A * a^2, B * b^2) :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_chord_fixed_point_l2567_256788


namespace NUMINAMATH_CALUDE_seashell_fraction_proof_l2567_256704

def dozen : ℕ := 12

theorem seashell_fraction_proof 
  (mimi_shells : ℕ) 
  (kyle_shells : ℕ) 
  (leigh_shells : ℕ) :
  mimi_shells = 2 * dozen →
  kyle_shells = 2 * mimi_shells →
  leigh_shells = 16 →
  (leigh_shells : ℚ) / (kyle_shells : ℚ) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_seashell_fraction_proof_l2567_256704


namespace NUMINAMATH_CALUDE_part_one_part_two_l2567_256745

-- Define the arithmetic sequence
def a (n : ℕ) : ℝ := 1 + 2 * (n - 1)

-- Part I
theorem part_one : 
  ∃ (m : ℕ), m > 0 ∧ 
  (1 / (a 1)^2) * (1 / (a m)^2) = (1 / (a 4)^2)^2 → 
  m = 25 :=
sorry

-- Part II
theorem part_two (a₁ d : ℝ) (h₁ : a₁ > 0) (h₂ : d > 0) :
  ∀ (n : ℕ), n > 0 →
  ¬ (∃ (k : ℝ), 
    1 / (a₁ + (n - 1) * d) - 1 / (a₁ + n * d) = 
    1 / (a₁ + n * d) - 1 / (a₁ + (n + 1) * d)) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2567_256745


namespace NUMINAMATH_CALUDE_eighteen_power_mnp_l2567_256735

theorem eighteen_power_mnp (m n p : ℕ) (P Q R : ℕ) 
  (hP : P = 2^m) (hQ : Q = 3^n) (hR : R = 5^p) :
  18^(m*n*p) = P^(n*p) * Q^(2*m*p) := by
  sorry

end NUMINAMATH_CALUDE_eighteen_power_mnp_l2567_256735


namespace NUMINAMATH_CALUDE_mary_potatoes_l2567_256798

/-- The number of potatoes Mary has now, given the initial number and the number of new potatoes left -/
def total_potatoes (initial : ℕ) (new_left : ℕ) : ℕ :=
  initial + new_left

/-- Theorem stating that Mary has 11 potatoes given the initial conditions -/
theorem mary_potatoes : total_potatoes 8 3 = 11 := by
  sorry

end NUMINAMATH_CALUDE_mary_potatoes_l2567_256798


namespace NUMINAMATH_CALUDE_solution_set_when_a_eq_3_range_of_a_l2567_256736

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| - |x + 1|

-- Part 1
theorem solution_set_when_a_eq_3 :
  {x : ℝ | f 3 x ≥ 2*x + 3} = {x : ℝ | x ≤ -1/4} := by sorry

-- Part 2
theorem range_of_a :
  ∀ a : ℝ, (∃ x ∈ Set.Icc 1 2, f a x ≤ |x - 5|) → a ∈ Set.Icc (-4) 7 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_eq_3_range_of_a_l2567_256736


namespace NUMINAMATH_CALUDE_pet_store_profit_l2567_256732

/-- The profit calculation for a pet store reselling geckos --/
theorem pet_store_profit (brandon_price : ℕ) (pet_store_markup : ℕ → ℕ) : 
  brandon_price = 100 → 
  (∀ x, pet_store_markup x = 3 * x + 5) →
  pet_store_markup brandon_price - brandon_price = 205 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_profit_l2567_256732


namespace NUMINAMATH_CALUDE_more_birds_than_nests_l2567_256795

theorem more_birds_than_nests :
  let birds : ℕ := 6
  let nests : ℕ := 3
  birds - nests = 3 :=
by sorry

end NUMINAMATH_CALUDE_more_birds_than_nests_l2567_256795


namespace NUMINAMATH_CALUDE_bowTie_equation_solution_l2567_256721

/-- The infinite nested radical operation -/
noncomputable def bowTie (a b : ℝ) : ℝ := a + Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + Real.sqrt b)))

/-- Theorem stating that if 5 ⋈ z = 12, then z = 42 -/
theorem bowTie_equation_solution :
  ∃ z : ℝ, bowTie 5 z = 12 → z = 42 := by
  sorry

end NUMINAMATH_CALUDE_bowTie_equation_solution_l2567_256721


namespace NUMINAMATH_CALUDE_sin_40_tan_10_minus_sqrt_3_l2567_256784

theorem sin_40_tan_10_minus_sqrt_3 : 
  Real.sin (40 * π / 180) * (Real.tan (10 * π / 180) - Real.sqrt 3) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sin_40_tan_10_minus_sqrt_3_l2567_256784


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2567_256773

/-- Given a geometric sequence {a_n} where the sum of the first n terms is S_n = 2^n + r,
    prove that r = -1 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (r : ℝ) :
  (∀ n : ℕ, S n = 2^n + r) →
  (∀ n : ℕ, n ≥ 1 → a n = S n - S (n-1)) →
  (∀ n : ℕ, n ≥ 2 → a n = 2 * a (n-1)) →
  r = -1 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_sum_l2567_256773


namespace NUMINAMATH_CALUDE_work_completion_time_l2567_256708

-- Define the work rate of A
def work_rate_A : ℚ := 1 / 60

-- Define the work done by A in 15 days
def work_done_A : ℚ := 15 * work_rate_A

-- Define the remaining work after A's 15 days
def remaining_work : ℚ := 1 - work_done_A

-- Define B's work rate based on completing the remaining work in 30 days
def work_rate_B : ℚ := remaining_work / 30

-- Define the combined work rate of A and B
def combined_work_rate : ℚ := work_rate_A + work_rate_B

-- Theorem to prove
theorem work_completion_time : (1 : ℚ) / combined_work_rate = 24 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2567_256708


namespace NUMINAMATH_CALUDE_count_fractions_l2567_256746

theorem count_fractions : ∃ (S : Finset Nat), 
  (∀ n ∈ S, 15 < n ∧ n < 240 ∧ ¬(n % 3 = 0 ∨ n % 5 = 0) ∧ (15 : ℚ) / 16 < (n : ℚ) / 15 ∧ (n : ℚ) / 15 < 1 / 15) ∧
  S.card = 8 ∧
  (∀ n : Nat, 15 < n ∧ n < 240 ∧ ¬(n % 3 = 0 ∨ n % 5 = 0) ∧ (15 : ℚ) / 16 < (n : ℚ) / 15 ∧ (n : ℚ) / 15 < 1 / 15 → n ∈ S) :=
by sorry

end NUMINAMATH_CALUDE_count_fractions_l2567_256746


namespace NUMINAMATH_CALUDE_parabola_focus_parameter_l2567_256778

/-- Given a parabola with equation x^2 = 2py and focus at (0, 2), prove that p = 4 -/
theorem parabola_focus_parameter : ∀ p : ℝ, 
  (∀ x y : ℝ, x^2 = 2*p*y) →  -- parabola equation
  (0, 2) = (0, p/2) →        -- focus coordinates
  p = 4 := by
sorry

end NUMINAMATH_CALUDE_parabola_focus_parameter_l2567_256778


namespace NUMINAMATH_CALUDE_exponent_rule_l2567_256767

theorem exponent_rule (x : ℝ) : 2 * x^2 * (3 * x^2) = 6 * x^4 := by
  sorry

end NUMINAMATH_CALUDE_exponent_rule_l2567_256767


namespace NUMINAMATH_CALUDE_painting_fraction_l2567_256706

def total_students : ℕ := 50
def field_fraction : ℚ := 1 / 5
def classroom_left : ℕ := 10

theorem painting_fraction :
  (total_students - (field_fraction * total_students).num - classroom_left) / total_students = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_painting_fraction_l2567_256706


namespace NUMINAMATH_CALUDE_star_value_l2567_256782

/-- Operation star defined as a * b = 3a - b^3 -/
def star (a b : ℝ) : ℝ := 3 * a - b^3

/-- Theorem: If a * 3 = 63, then a = 30 -/
theorem star_value (a : ℝ) (h : star a 3 = 63) : a = 30 := by
  sorry

end NUMINAMATH_CALUDE_star_value_l2567_256782


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l2567_256775

/-- Given an ellipse with semi-major axis a and semi-minor axis b,
    and a line y = kx intersecting the ellipse at points B and C,
    if the product of the slopes of AB and AC is -3/4,
    then the eccentricity of the ellipse is 1/2. -/
theorem ellipse_eccentricity (a b : ℝ) (k : ℝ) :
  a > b ∧ b > 0 →
  ∃ (B C : ℝ × ℝ),
    (B.1^2 / a^2 + B.2^2 / b^2 = 1) ∧
    (C.1^2 / a^2 + C.2^2 / b^2 = 1) ∧
    (B.2 = k * B.1) ∧
    (C.2 = k * C.1) ∧
    ((B.2 - b) / B.1 * (C.2 + b) / C.1 = -3/4) →
  Real.sqrt (1 - b^2 / a^2) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l2567_256775


namespace NUMINAMATH_CALUDE_david_pushups_l2567_256797

theorem david_pushups (class_average : ℕ) (david_pushups : ℕ) 
  (h1 : class_average = 30)
  (h2 : david_pushups = 44)
  (h3 : ∃ (zachary_pushups : ℕ), david_pushups = zachary_pushups + 9)
  (h4 : ∃ (hailey_pushups : ℕ), 
    (∃ (zachary_pushups : ℕ), zachary_pushups = 2 * hailey_pushups) ∧ 
    hailey_pushups = class_average - (class_average / 10)) :
  david_pushups = 63 := by
sorry

end NUMINAMATH_CALUDE_david_pushups_l2567_256797


namespace NUMINAMATH_CALUDE_probability_of_white_ball_l2567_256776

/-- The probability of drawing a white ball from a bag of red and white balls -/
theorem probability_of_white_ball (total : ℕ) (red : ℕ) (white : ℕ) :
  total = red + white →
  white > 0 →
  total > 0 →
  (white : ℚ) / (total : ℚ) = 4 / 9 :=
by
  sorry

end NUMINAMATH_CALUDE_probability_of_white_ball_l2567_256776


namespace NUMINAMATH_CALUDE_units_digit_of_7_power_2006_l2567_256786

theorem units_digit_of_7_power_2006 : ∃ n : ℕ, 7^2006 ≡ 9 [ZMOD 10] := by sorry

end NUMINAMATH_CALUDE_units_digit_of_7_power_2006_l2567_256786


namespace NUMINAMATH_CALUDE_students_wanting_fruit_l2567_256755

theorem students_wanting_fruit (red_apples green_apples extra_apples : ℕ) :
  let total_apples := red_apples + green_apples
  let students := total_apples - extra_apples
  students = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_students_wanting_fruit_l2567_256755


namespace NUMINAMATH_CALUDE_carlys_dogs_l2567_256702

theorem carlys_dogs (total_nails : ℕ) (three_legged_dogs : ℕ) (nails_per_paw : ℕ) :
  total_nails = 164 →
  three_legged_dogs = 3 →
  nails_per_paw = 4 →
  ∃ (four_legged_dogs : ℕ),
    four_legged_dogs * 4 * nails_per_paw + three_legged_dogs * 3 * nails_per_paw = total_nails ∧
    four_legged_dogs + three_legged_dogs = 11 :=
by sorry

end NUMINAMATH_CALUDE_carlys_dogs_l2567_256702


namespace NUMINAMATH_CALUDE_shooting_scenarios_correct_l2567_256718

/-- Represents a shooting scenario with a total number of shots and hits -/
structure ShootingScenario where
  totalShots : Nat
  totalHits : Nat

/-- Calculates the number of possible situations for Scenario 1 -/
def scenario1Situations (s : ShootingScenario) : Nat :=
  if s.totalShots = 10 ∧ s.totalHits = 7 then
    12
  else
    0

/-- Calculates the number of possible situations for Scenario 2 -/
def scenario2Situations (s : ShootingScenario) : Nat :=
  if s.totalShots = 10 then
    144
  else
    0

/-- Calculates the number of possible situations for Scenario 3 -/
def scenario3Situations (s : ShootingScenario) : Nat :=
  if s.totalShots = 10 ∧ s.totalHits = 6 then
    50
  else
    0

theorem shooting_scenarios_correct :
  ∀ s : ShootingScenario,
    (scenario1Situations s = 12 ∨ scenario1Situations s = 0) ∧
    (scenario2Situations s = 144 ∨ scenario2Situations s = 0) ∧
    (scenario3Situations s = 50 ∨ scenario3Situations s = 0) :=
by sorry

end NUMINAMATH_CALUDE_shooting_scenarios_correct_l2567_256718


namespace NUMINAMATH_CALUDE_only_one_true_statement_l2567_256723

/-- Two lines are non-coincident -/
def NonCoincidentLines (m n : Line) : Prop :=
  m ≠ n

/-- Two planes are non-coincident -/
def NonCoincidentPlanes (α β : Plane) : Prop :=
  α ≠ β

/-- A line is parallel to a plane -/
def LineParallelToPlane (l : Line) (p : Plane) : Prop :=
  sorry

/-- A line is perpendicular to a plane -/
def LinePerpendicularToPlane (l : Line) (p : Plane) : Prop :=
  sorry

/-- Two lines are parallel -/
def ParallelLines (l1 l2 : Line) : Prop :=
  sorry

/-- Two lines intersect -/
def LinesIntersect (l1 l2 : Line) : Prop :=
  sorry

/-- Two planes are perpendicular -/
def PerpendicularPlanes (p1 p2 : Plane) : Prop :=
  sorry

/-- Two lines are perpendicular -/
def PerpendicularLines (l1 l2 : Line) : Prop :=
  sorry

/-- Projection of a line onto a plane -/
def ProjectionOntoPlane (l : Line) (p : Plane) : Line :=
  sorry

theorem only_one_true_statement
  (m n : Line) (α β : Plane)
  (h_lines : NonCoincidentLines m n)
  (h_planes : NonCoincidentPlanes α β) :
  (LineParallelToPlane m α ∧ LineParallelToPlane n α → ¬LinesIntersect m n) ∨
  (LinePerpendicularToPlane m α ∧ LinePerpendicularToPlane n α → ParallelLines m n) ∨
  (PerpendicularPlanes α β ∧ PerpendicularLines m n ∧ LinePerpendicularToPlane m α → LinePerpendicularToPlane n β) ∨
  (PerpendicularLines (ProjectionOntoPlane m α) (ProjectionOntoPlane n α) → PerpendicularLines m n) :=
by sorry

end NUMINAMATH_CALUDE_only_one_true_statement_l2567_256723


namespace NUMINAMATH_CALUDE_geometric_mean_of_4_and_16_l2567_256722

theorem geometric_mean_of_4_and_16 : 
  ∃ x : ℝ, x > 0 ∧ x^2 = 4 * 16 ∧ (x = 8 ∨ x = -8) :=
by sorry

end NUMINAMATH_CALUDE_geometric_mean_of_4_and_16_l2567_256722


namespace NUMINAMATH_CALUDE_mother_daughter_age_relation_l2567_256785

theorem mother_daughter_age_relation : 
  ∀ (mother_current_age daughter_future_age : ℕ),
    mother_current_age = 41 →
    daughter_future_age = 26 →
    ∃ (years_ago : ℕ),
      years_ago = 5 ∧
      mother_current_age - years_ago = 2 * (daughter_future_age - 3 - years_ago) :=
by
  sorry

end NUMINAMATH_CALUDE_mother_daughter_age_relation_l2567_256785


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_three_l2567_256720

theorem fraction_zero_implies_x_equals_three (x : ℝ) :
  (x^2 - 9) / (x + 3) = 0 ∧ x + 3 ≠ 0 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_three_l2567_256720


namespace NUMINAMATH_CALUDE_quadratic_function_max_m_l2567_256747

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_function_max_m (a b c : ℝ) (ha : a ≠ 0) :
  (∀ x, f a b c (x - 4) = f a b c (2 - x)) →
  (∀ x ∈ Set.Ioo 0 2, f a b c x ≤ ((x + 1) / 2)^2) →
  (∃ x, ∀ y, f a b c x ≤ f a b c y) →
  (∃ x, f a b c x = 0) →
  (∃ m > 1, ∀ m' > m, ¬∃ t, ∀ x ∈ Set.Icc 1 m', f a b c (x + t) ≤ x) ∧
  (∃ t, ∀ x ∈ Set.Icc 1 9, f a b c (x + t) ≤ x) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_max_m_l2567_256747


namespace NUMINAMATH_CALUDE_ln_inequality_range_l2567_256705

theorem ln_inequality_range (x : ℝ) (m : ℝ) (h : x > 0) :
  (∀ x > 0, Real.log x ≤ x * Real.exp (m^2 - m - 1)) ↔ (m ≤ 0 ∨ m ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_ln_inequality_range_l2567_256705


namespace NUMINAMATH_CALUDE_a_range_l2567_256756

-- Define the ⊗ operation
def otimes (x y : ℝ) : ℝ := x * (1 - y)

-- State the theorem
theorem a_range (a : ℝ) : 
  (∀ x : ℝ, otimes (x - a) (x + 1) < 1) → -2 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_a_range_l2567_256756


namespace NUMINAMATH_CALUDE_smallestPalindromeNumber_satisfies_conditions_l2567_256734

/-- A function to check if a number is a palindrome in a given base -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop :=
  (n.digits base).reverse = n.digits base

/-- The smallest positive integer greater than 10 that is a palindrome in base 2 and 4, and is odd -/
def smallestPalindromeNumber : ℕ := 17

/-- Theorem stating that smallestPalindromeNumber satisfies all conditions -/
theorem smallestPalindromeNumber_satisfies_conditions :
  smallestPalindromeNumber > 10 ∧
  isPalindrome smallestPalindromeNumber 2 ∧
  isPalindrome smallestPalindromeNumber 4 ∧
  Odd smallestPalindromeNumber ∧
  ∀ n : ℕ, n > 10 → isPalindrome n 2 → isPalindrome n 4 → Odd n →
    n ≥ smallestPalindromeNumber :=
by sorry

#eval smallestPalindromeNumber

end NUMINAMATH_CALUDE_smallestPalindromeNumber_satisfies_conditions_l2567_256734


namespace NUMINAMATH_CALUDE_routes_in_3x3_grid_l2567_256777

/-- The number of routes from top-left to bottom-right in a 3x3 grid -/
def number_of_routes : ℕ := 20

/-- The size of the grid -/
def grid_size : ℕ := 3

/-- The total number of moves required to reach the bottom-right corner -/
def total_moves : ℕ := 2 * grid_size

/-- The number of right moves (or down moves) required -/
def moves_in_one_direction : ℕ := grid_size

theorem routes_in_3x3_grid : 
  number_of_routes = Nat.choose total_moves moves_in_one_direction := by
  sorry

end NUMINAMATH_CALUDE_routes_in_3x3_grid_l2567_256777


namespace NUMINAMATH_CALUDE_cubic_function_zeros_l2567_256780

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x^2 + 1

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 - 6 * x

theorem cubic_function_zeros (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) →
  0 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_zeros_l2567_256780
