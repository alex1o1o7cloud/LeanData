import Mathlib

namespace NUMINAMATH_CALUDE_bisecting_line_sum_l2207_220712

/-- Triangle PQR with vertices P(0, 10), Q(3, 0), and R(10, 0) -/
structure Triangle :=
  (P : ℝ × ℝ)
  (Q : ℝ × ℝ)
  (R : ℝ × ℝ)

/-- A line represented by its slope and y-intercept -/
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

/-- The line that bisects the area of the triangle -/
def bisecting_line (t : Triangle) : Line :=
  sorry

/-- The theorem to be proved -/
theorem bisecting_line_sum (t : Triangle) :
  let pqr := Triangle.mk (0, 10) (3, 0) (10, 0)
  let l := bisecting_line pqr
  l.slope + l.intercept = -5 :=
sorry

end NUMINAMATH_CALUDE_bisecting_line_sum_l2207_220712


namespace NUMINAMATH_CALUDE_min_value_expression_l2207_220792

theorem min_value_expression (n : ℕ+) : 
  (n : ℝ) / 2 + 24 / (n : ℝ) ≥ 7 ∧ ∃ m : ℕ+, (m : ℝ) / 2 + 24 / (m : ℝ) = 7 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2207_220792


namespace NUMINAMATH_CALUDE_log_ratio_independence_l2207_220710

theorem log_ratio_independence (P K a b : ℝ) 
  (hP : P > 0) (hK : K > 0) (ha : a > 0 ∧ a ≠ 1) (hb : b > 0 ∧ b ≠ 1) : 
  (Real.log P / Real.log a) / (Real.log K / Real.log a) = 
  (Real.log P / Real.log b) / (Real.log K / Real.log b) := by
  sorry

end NUMINAMATH_CALUDE_log_ratio_independence_l2207_220710


namespace NUMINAMATH_CALUDE_cos_225_degrees_l2207_220701

theorem cos_225_degrees : Real.cos (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_225_degrees_l2207_220701


namespace NUMINAMATH_CALUDE_cookies_left_for_sonny_l2207_220788

theorem cookies_left_for_sonny (total : ℕ) (brother sister cousin : ℕ) 
  (h1 : total = 45)
  (h2 : brother = 12)
  (h3 : sister = 9)
  (h4 : cousin = 7) :
  total - (brother + sister + cousin) = 17 := by
  sorry

end NUMINAMATH_CALUDE_cookies_left_for_sonny_l2207_220788


namespace NUMINAMATH_CALUDE_product_over_sum_equals_6608_l2207_220707

theorem product_over_sum_equals_6608 : 
  (1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9) / (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10) = 6608 := by
  sorry

end NUMINAMATH_CALUDE_product_over_sum_equals_6608_l2207_220707


namespace NUMINAMATH_CALUDE_circle_and_line_properties_l2207_220734

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 4

-- Define the line l₁ passing through A(1,0)
def line_l1 (x y : ℝ) : Prop := ∃ k : ℝ, y = k * (x - 1)

-- Define tangent line condition
def is_tangent (line : (ℝ → ℝ → Prop)) (circle : (ℝ → ℝ → Prop)) : Prop :=
  ∃ x y : ℝ, line x y ∧ circle x y ∧
  ∀ x' y' : ℝ, line x' y' → circle x' y' → (x', y') = (x, y)

-- Define the slope angle of π/4
def slope_angle_pi_4 (line : (ℝ → ℝ → Prop)) : Prop :=
  ∃ k : ℝ, (∀ x y : ℝ, line x y ↔ y = k * (x - 1)) ∧ k = 1

-- Main theorem
theorem circle_and_line_properties :
  (is_tangent line_l1 circle_C →
    (∀ x y : ℝ, line_l1 x y ↔ (x = 1 ∨ 3*x - 4*y - 3 = 0))) ∧
  (slope_angle_pi_4 line_l1 →
    ∃ P Q : ℝ × ℝ,
      circle_C P.1 P.2 ∧ circle_C Q.1 Q.2 ∧
      line_l1 P.1 P.2 ∧ line_l1 Q.1 Q.2 ∧
      ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2) = (4, 3)) :=
sorry

end NUMINAMATH_CALUDE_circle_and_line_properties_l2207_220734


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2207_220784

theorem inequality_system_solution (x : ℝ) :
  (x - 1 < 2*x + 1) ∧ ((2*x - 5) / 3 ≤ 1) → -2 < x ∧ x ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2207_220784


namespace NUMINAMATH_CALUDE_fraction_equality_l2207_220721

theorem fraction_equality : 
  (3 / 7 + 5 / 8) / (5 / 12 + 2 / 15) = 295 / 154 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l2207_220721


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_12_l2207_220757

theorem largest_four_digit_divisible_by_12 : ∃ n : ℕ, n = 9996 ∧ 
  n % 12 = 0 ∧ 
  n ≤ 9999 ∧ 
  n ≥ 1000 ∧
  ∀ m : ℕ, m % 12 = 0 ∧ m ≤ 9999 ∧ m ≥ 1000 → m ≤ n := by
  sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_12_l2207_220757


namespace NUMINAMATH_CALUDE_quadratic_max_min_difference_l2207_220754

def f (x : ℝ) := x^2 - 4*x - 6

theorem quadratic_max_min_difference :
  let x_min := -3
  let x_max := 4
  ∃ (x_min_value x_max_value : ℝ),
    (∀ x, x_min ≤ x ∧ x ≤ x_max → f x ≥ x_min_value) ∧
    (∃ x, x_min ≤ x ∧ x ≤ x_max ∧ f x = x_min_value) ∧
    (∀ x, x_min ≤ x ∧ x ≤ x_max → f x ≤ x_max_value) ∧
    (∃ x, x_min ≤ x ∧ x ≤ x_max ∧ f x = x_max_value) ∧
    x_max_value - x_min_value = 25 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_max_min_difference_l2207_220754


namespace NUMINAMATH_CALUDE_robot_center_movement_l2207_220728

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circular robot -/
structure CircularRobot where
  center : Point
  radius : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to check if a point remains on a line -/
def remainsOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Predicate to check if a point is on the boundary of a circular robot -/
def isOnBoundary (p : Point) (r : CircularRobot) : Prop :=
  (p.x - r.center.x)^2 + (p.y - r.center.y)^2 = r.radius^2

/-- The main theorem -/
theorem robot_center_movement
  (r : CircularRobot)
  (h : ∀ (p : Point), isOnBoundary p r → ∃ (l : Line), ∀ (t : ℝ), remainsOnLine p l) :
  ¬ (∀ (t : ℝ), ∃ (l : Line), remainsOnLine r.center l) :=
sorry

end NUMINAMATH_CALUDE_robot_center_movement_l2207_220728


namespace NUMINAMATH_CALUDE_fernanda_savings_after_payments_l2207_220732

/-- Calculates the total amount in Fernanda's savings account after receiving payments from debtors -/
theorem fernanda_savings_after_payments (aryan_debt kyro_debt : ℚ) 
  (h1 : aryan_debt = 1200)
  (h2 : aryan_debt = 2 * kyro_debt)
  (h3 : aryan_payment = 0.6 * aryan_debt)
  (h4 : kyro_payment = 0.8 * kyro_debt)
  (h5 : initial_savings = 300) :
  initial_savings + aryan_payment + kyro_payment = 1500 := by
  sorry

end NUMINAMATH_CALUDE_fernanda_savings_after_payments_l2207_220732


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l2207_220747

/-- For a parabola with equation y^2 = ax, if the distance from its focus to its directrix is 2, then a = 4. -/
theorem parabola_focus_directrix_distance (a : ℝ) : 
  (∃ y : ℝ → ℝ, ∀ x, (y x)^2 = a * x) →  -- Parabola equation
  (∃ f d : ℝ, abs (f - d) = 2) →        -- Distance between focus and directrix
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l2207_220747


namespace NUMINAMATH_CALUDE_trigonometric_expression_evaluation_l2207_220704

theorem trigonometric_expression_evaluation : 
  (2 * Real.sin (100 * π / 180) - Real.cos (70 * π / 180)) / Real.cos (20 * π / 180) = 2 * Real.sqrt 3 - 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_evaluation_l2207_220704


namespace NUMINAMATH_CALUDE_random_points_probability_l2207_220759

/-- The probability that a randomly chosen point y is greater than another randomly chosen point x
    but less than three times x, where both x and y are chosen uniformly from the interval [0, 1] -/
theorem random_points_probability : Real := by
  sorry

end NUMINAMATH_CALUDE_random_points_probability_l2207_220759


namespace NUMINAMATH_CALUDE_tennis_game_wins_l2207_220785

theorem tennis_game_wins (total_games : ℕ) (player_a_wins player_b_wins player_c_wins : ℕ) :
  total_games = 6 →
  player_a_wins = 5 →
  player_b_wins = 2 →
  player_c_wins = 1 →
  ∃ player_d_wins : ℕ, player_d_wins = 4 ∧ player_a_wins + player_b_wins + player_c_wins + player_d_wins = 2 * total_games :=
by sorry

end NUMINAMATH_CALUDE_tennis_game_wins_l2207_220785


namespace NUMINAMATH_CALUDE_total_boys_in_three_sections_l2207_220744

theorem total_boys_in_three_sections (section1_total : ℕ) (section2_total : ℕ) (section3_total : ℕ)
  (section1_girls_ratio : ℚ) (section2_boys_ratio : ℚ) (section3_boys_ratio : ℚ) :
  section1_total = 160 →
  section2_total = 200 →
  section3_total = 240 →
  section1_girls_ratio = 1/4 →
  section2_boys_ratio = 3/5 →
  section3_boys_ratio = 7/12 →
  (section1_total - section1_total * section1_girls_ratio) +
  (section2_total * section2_boys_ratio) +
  (section3_total * section3_boys_ratio) = 380 := by
sorry

end NUMINAMATH_CALUDE_total_boys_in_three_sections_l2207_220744


namespace NUMINAMATH_CALUDE_deck_cost_per_square_foot_l2207_220751

/-- Proves the cost per square foot for deck construction given the dimensions, sealant cost, and total cost paid. -/
theorem deck_cost_per_square_foot 
  (length : ℝ) 
  (width : ℝ) 
  (sealant_cost_per_sq_ft : ℝ) 
  (total_cost : ℝ) 
  (h1 : length = 30) 
  (h2 : width = 40) 
  (h3 : sealant_cost_per_sq_ft = 1) 
  (h4 : total_cost = 4800) : 
  ∃ (cost_per_sq_ft : ℝ), 
    cost_per_sq_ft = 3 ∧ 
    total_cost = length * width * (cost_per_sq_ft + sealant_cost_per_sq_ft) :=
by sorry


end NUMINAMATH_CALUDE_deck_cost_per_square_foot_l2207_220751


namespace NUMINAMATH_CALUDE_representation_of_real_number_l2207_220750

theorem representation_of_real_number (x : ℝ) (hx : 0 < x ∧ x ≤ 1) :
  ∃ (n : ℕ → ℕ), 
    (∀ k, n (k + 1) / n k ∈ ({2, 3, 4} : Set ℕ)) ∧ 
    (∑' k, (1 : ℝ) / n k) = x :=
sorry

end NUMINAMATH_CALUDE_representation_of_real_number_l2207_220750


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l2207_220724

theorem trigonometric_inequality (x : ℝ) (h : x ∈ Set.Ioo 0 (3 * π / 8)) :
  (1 / Real.sin (x / 3)) + (1 / Real.sin (8 * x / 3)) > 
  Real.sin (3 * x / 2) / (Real.sin (x / 2) * Real.sin (2 * x)) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l2207_220724


namespace NUMINAMATH_CALUDE_problem_statement_l2207_220731

theorem problem_statement (x y : ℕ) (hx : x = 4) (hy : y = 3) : 5 * x + 2 * y * 3 = 38 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2207_220731


namespace NUMINAMATH_CALUDE_computer_table_cost_price_l2207_220762

/-- Proves that the cost price of a computer table is 2500 when the selling price is 3000 
    and the markup is 20% -/
theorem computer_table_cost_price 
  (markup_percentage : ℝ) 
  (selling_price : ℝ) : 
  markup_percentage = 20 →
  selling_price = 3000 →
  (100 + markup_percentage) / 100 * (selling_price / (1 + markup_percentage / 100)) = 2500 := by
  sorry

end NUMINAMATH_CALUDE_computer_table_cost_price_l2207_220762


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2207_220740

-- Define the property that a function f must satisfy
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x * f y + f (x + y) = x * y

-- State the theorem
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, SatisfiesEquation f ↔ (∀ x, f x = x - 1) ∨ (∀ x, f x = -x - 1) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2207_220740


namespace NUMINAMATH_CALUDE_sqrt_equation_implies_sum_and_reciprocal_l2207_220752

theorem sqrt_equation_implies_sum_and_reciprocal (x : ℝ) (h : x > 0) :
  Real.sqrt x - 1 / Real.sqrt x = 2 * Real.sqrt 3 → x + 1 / x = 14 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_implies_sum_and_reciprocal_l2207_220752


namespace NUMINAMATH_CALUDE_parking_garage_spaces_l2207_220775

theorem parking_garage_spaces (level1 level2 level3 level4 : ℕ) : 
  level1 = 90 →
  level3 = level2 + 12 →
  level4 = level3 - 9 →
  level1 + level2 + level3 + level4 = 399 →
  level2 = level1 + 8 :=
by
  sorry

end NUMINAMATH_CALUDE_parking_garage_spaces_l2207_220775


namespace NUMINAMATH_CALUDE_integer_fraction_equality_l2207_220799

theorem integer_fraction_equality (d : ℤ) : ∃ m n : ℤ, d * (m^2 - n) = n - 2*m + 1 := by
  sorry

end NUMINAMATH_CALUDE_integer_fraction_equality_l2207_220799


namespace NUMINAMATH_CALUDE_complex_modulus_one_l2207_220764

theorem complex_modulus_one (z : ℂ) (h : z * (1 + Complex.I) = (1 - Complex.I)) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_one_l2207_220764


namespace NUMINAMATH_CALUDE_ellipse_vector_dot_product_range_l2207_220702

/-- The ellipse equation -/
def on_ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- Point M -/
def M : ℝ × ℝ := (1, 0)

/-- Vector dot product -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- Vector from M to a point -/
def vector_MA (A : ℝ × ℝ) : ℝ × ℝ :=
  (A.1 - M.1, A.2 - M.2)

theorem ellipse_vector_dot_product_range :
  ∀ A B : ℝ × ℝ,
  on_ellipse A.1 A.2 →
  on_ellipse B.1 B.2 →
  dot_product (vector_MA A) (vector_MA B) = 0 →
  ∃ x : ℝ, x = dot_product (vector_MA A) (A.1 - B.1, A.2 - B.2) ∧
           2/3 ≤ x ∧ x ≤ 9 :=
sorry

end NUMINAMATH_CALUDE_ellipse_vector_dot_product_range_l2207_220702


namespace NUMINAMATH_CALUDE_closest_to_fraction_l2207_220793

def options : List ℝ := [50, 500, 1500, 1600, 2000]

theorem closest_to_fraction (options : List ℝ) :
  let fraction : ℝ := 351 / 0.22
  let differences := options.map (λ x => |x - fraction|)
  let min_diff := differences.minimum?
  let closest := options.find? (λ x => |x - fraction| = min_diff.get!)
  closest = some 1600 := by
  sorry

end NUMINAMATH_CALUDE_closest_to_fraction_l2207_220793


namespace NUMINAMATH_CALUDE_z_modulus_l2207_220795

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation for z
def z_equation (z : ℂ) : Prop := z + 2 * i = (3 - i^3) / (1 + i)

-- Theorem statement
theorem z_modulus (z : ℂ) (h : z_equation z) : Complex.abs z = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_z_modulus_l2207_220795


namespace NUMINAMATH_CALUDE_bisected_tangents_iff_parabola_l2207_220738

/-- A curve in the xy-plane -/
structure Curve where
  -- The equation of the curve
  equation : ℝ → ℝ → Prop

/-- Property that any tangent line segment between the point of tangency and the x-axis is bisected by the y-axis -/
def has_bisected_tangents (c : Curve) : Prop :=
  ∀ (x y : ℝ), c.equation x y →
    ∃ (slope : ℝ), 
      -- The tangent line at (x, y) intersects the x-axis at (-x, 0)
      y = slope * (x - (-x))

/-- A parabola of the form y^2 = Cx -/
def is_parabola (c : Curve) : Prop :=
  ∃ (C : ℝ), ∀ (x y : ℝ), c.equation x y ↔ y^2 = C * x

/-- Theorem stating the equivalence between the bisected tangents property and being a parabola -/
theorem bisected_tangents_iff_parabola (c : Curve) :
  has_bisected_tangents c ↔ is_parabola c :=
sorry

end NUMINAMATH_CALUDE_bisected_tangents_iff_parabola_l2207_220738


namespace NUMINAMATH_CALUDE_cubic_root_sum_l2207_220776

theorem cubic_root_sum (u v w : ℝ) : 
  (u^3 - 6*u^2 + 11*u - 6 = 0) →
  (v^3 - 6*v^2 + 11*v - 6 = 0) →
  (w^3 - 6*w^2 + 11*w - 6 = 0) →
  u*v/w + v*w/u + w*u/v = 49/6 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l2207_220776


namespace NUMINAMATH_CALUDE_roll_12_with_8_dice_l2207_220700

/-- The number of ways to roll a sum of 12 with 8 fair 6-sided dice -/
def waysToRoll12With8Dice : ℕ := sorry

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The number of dice -/
def numDice : ℕ := 8

/-- The target sum -/
def targetSum : ℕ := 12

theorem roll_12_with_8_dice :
  waysToRoll12With8Dice = 330 := by sorry

end NUMINAMATH_CALUDE_roll_12_with_8_dice_l2207_220700


namespace NUMINAMATH_CALUDE_janice_earnings_this_week_l2207_220772

/-- Calculates Janice's earnings after deductions for a week -/
def janice_earnings (regular_daily_rate : ℚ) (days_worked : ℕ) 
  (weekday_overtime_rate : ℚ) (weekend_overtime_rate : ℚ)
  (weekday_overtime_shifts : ℕ) (weekend_overtime_shifts : ℕ)
  (tips : ℚ) (tax_rate : ℚ) : ℚ :=
  let regular_earnings := regular_daily_rate * days_worked
  let weekday_overtime := weekday_overtime_rate * weekday_overtime_shifts
  let weekend_overtime := weekend_overtime_rate * weekend_overtime_shifts
  let total_before_tax := regular_earnings + weekday_overtime + weekend_overtime + tips
  let tax := tax_rate * total_before_tax
  total_before_tax - tax

/-- Theorem stating Janice's earnings after deductions -/
theorem janice_earnings_this_week : 
  janice_earnings 30 6 15 20 2 1 10 (1/10) = 216 := by
  sorry


end NUMINAMATH_CALUDE_janice_earnings_this_week_l2207_220772


namespace NUMINAMATH_CALUDE_factorization_x8_minus_81_l2207_220790

theorem factorization_x8_minus_81 (x : ℝ) : 
  x^8 - 81 = (x^2 - 3) * (x^2 + 3) * (x^4 + 9) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x8_minus_81_l2207_220790


namespace NUMINAMATH_CALUDE_limit_equivalence_l2207_220723

def has_limit (u : ℕ → ℝ) (L : ℝ) : Prop :=
  ∀ ε : ℝ, ∃ N : ℕ, ∀ n : ℕ, (ε > 0 ∧ n ≥ N) → |L - u n| ≤ ε

def alt_def1 (u : ℕ → ℝ) (L : ℝ) : Prop :=
  ∀ ε : ℝ, ε ≤ 0 ∨ (∃ N : ℕ, ∀ n : ℕ, |L - u n| ≤ ε ∨ n < N)

def alt_def2 (u : ℕ → ℝ) (L : ℝ) : Prop :=
  ∀ ε : ℝ, ∀ n : ℕ, ∃ N : ℕ, (ε > 0 ∧ n ≥ N) → |L - u n| ≤ ε

def alt_def3 (u : ℕ → ℝ) (L : ℝ) : Prop :=
  ∀ ε : ℝ, ∃ N : ℕ, ∀ n : ℕ, (ε > 0 ∧ n > N) → |L - u n| < ε

def alt_def4 (u : ℕ → ℝ) (L : ℝ) : Prop :=
  ∃ N : ℕ, ∀ ε : ℝ, ∀ n : ℕ, (ε > 0 ∧ n ≥ N) → |L - u n| ≤ ε

theorem limit_equivalence (u : ℕ → ℝ) (L : ℝ) :
  (has_limit u L ↔ alt_def1 u L) ∧
  (has_limit u L ↔ alt_def3 u L) ∧
  ¬(has_limit u L ↔ alt_def2 u L) ∧
  ¬(has_limit u L ↔ alt_def4 u L) := by
  sorry

end NUMINAMATH_CALUDE_limit_equivalence_l2207_220723


namespace NUMINAMATH_CALUDE_strawberry_picker_l2207_220711

/-- Given three people picking strawberries, proves that one person picked 200 strawberries -/
theorem strawberry_picker (total jonathan_matthew matthew_zac : ℕ) 
  (h_total : total = 550)
  (h_jonathan_matthew : jonathan_matthew = 350)
  (h_matthew_zac : matthew_zac = 250) :
  ∃ (jonathan matthew zac : ℕ),
    jonathan + matthew + zac = total ∧
    jonathan + matthew = jonathan_matthew ∧
    matthew + zac = matthew_zac ∧
    zac = 200 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_picker_l2207_220711


namespace NUMINAMATH_CALUDE_business_hours_per_week_l2207_220708

-- Define the operating hours for weekdays and weekends
def weekdayHours : ℕ := 6
def weekendHours : ℕ := 4

-- Define the number of weekdays and weekend days in a week
def weekdays : ℕ := 5
def weekendDays : ℕ := 2

-- Define the total hours open in a week
def totalHoursOpen : ℕ := weekdayHours * weekdays + weekendHours * weekendDays

-- Theorem statement
theorem business_hours_per_week :
  totalHoursOpen = 38 := by
sorry

end NUMINAMATH_CALUDE_business_hours_per_week_l2207_220708


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2207_220768

theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x, a * x^2 - 3*x + b > 4 ↔ x < 1 ∨ x > 2) →
  (a = 1 ∧ b = 6) ∧
  (∀ c : ℝ,
    (c > 2 → ∀ x, x^2 - (c+2)*x + 2*c < 0 ↔ 2 < x ∧ x < c) ∧
    (c = 2 → ∀ x, ¬(x^2 - (c+2)*x + 2*c < 0)) ∧
    (c < 2 → ∀ x, x^2 - (c+2)*x + 2*c < 0 ↔ c < x ∧ x < 2)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2207_220768


namespace NUMINAMATH_CALUDE_roundness_of_250000_l2207_220720

/-- The roundness of a positive integer is the sum of the exponents in its prime factorization. -/
def roundness (n : ℕ+) : ℕ := sorry

/-- The roundness of 250,000 is 10. -/
theorem roundness_of_250000 : roundness 250000 = 10 := by sorry

end NUMINAMATH_CALUDE_roundness_of_250000_l2207_220720


namespace NUMINAMATH_CALUDE_max_sum_of_factors_of_24_l2207_220787

theorem max_sum_of_factors_of_24 : 
  ∀ (a b : ℕ), a * b = 24 → a + b ≤ 25 :=
by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_of_24_l2207_220787


namespace NUMINAMATH_CALUDE_square_root_of_8_factorial_over_70_l2207_220777

theorem square_root_of_8_factorial_over_70 : 
  let factorial_8 := 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1
  Real.sqrt (factorial_8 / 70) = 24 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_8_factorial_over_70_l2207_220777


namespace NUMINAMATH_CALUDE_binomial_60_3_l2207_220779

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by sorry

end NUMINAMATH_CALUDE_binomial_60_3_l2207_220779


namespace NUMINAMATH_CALUDE_solve_run_problem_l2207_220719

def run_problem (speed2 : ℝ) : Prop :=
  let time1 : ℝ := 0.5
  let speed1 : ℝ := 10
  let time2 : ℝ := 0.5
  let time3 : ℝ := 0.25
  let speed3 : ℝ := 8
  let total_distance : ℝ := 17
  (speed1 * time1) + (speed2 * time2) + (speed3 * time3) = total_distance

theorem solve_run_problem : 
  ∃ (speed2 : ℝ), run_problem speed2 ∧ speed2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_solve_run_problem_l2207_220719


namespace NUMINAMATH_CALUDE_hyperbola_m_value_l2207_220745

def hyperbola_equation (m : ℝ) (x y : ℝ) : Prop :=
  m * x^2 + y^2 = 1

def imaginary_axis_twice_real_axis (m : ℝ) : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 2*a = b ∧
  ∀ x y : ℝ, hyperbola_equation m x y ↔ (x/a)^2 - (y/b)^2 = 1

theorem hyperbola_m_value :
  ∀ m : ℝ, imaginary_axis_twice_real_axis m → m = -1/4 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_m_value_l2207_220745


namespace NUMINAMATH_CALUDE_marks_difference_l2207_220758

/-- Given that the average mark in chemistry and mathematics is 55,
    prove that the difference between the total marks in all three subjects
    and the marks in physics is 110. -/
theorem marks_difference (P C M : ℝ) 
    (h1 : (C + M) / 2 = 55) : 
    (P + C + M) - P = 110 := by
  sorry

end NUMINAMATH_CALUDE_marks_difference_l2207_220758


namespace NUMINAMATH_CALUDE_power_multiplication_l2207_220709

theorem power_multiplication (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l2207_220709


namespace NUMINAMATH_CALUDE_purple_flowers_killed_is_40_l2207_220748

/-- Represents the florist's bouquet problem -/
structure BouquetProblem where
  flowers_per_bouquet : ℕ
  initial_seeds_per_color : ℕ
  num_colors : ℕ
  red_killed : ℕ
  yellow_killed : ℕ
  orange_killed : ℕ
  bouquets_made : ℕ

/-- Calculates the number of purple flowers killed by the fungus -/
def purple_flowers_killed (problem : BouquetProblem) : ℕ :=
  let total_initial := problem.initial_seeds_per_color * problem.num_colors
  let red_left := problem.initial_seeds_per_color - problem.red_killed
  let yellow_left := problem.initial_seeds_per_color - problem.yellow_killed
  let orange_left := problem.initial_seeds_per_color - problem.orange_killed
  let total_needed := problem.flowers_per_bouquet * problem.bouquets_made
  let non_purple_left := red_left + yellow_left + orange_left
  problem.initial_seeds_per_color - (total_needed - non_purple_left)

/-- Theorem stating that the number of purple flowers killed is 40 -/
theorem purple_flowers_killed_is_40 (problem : BouquetProblem) 
    (h1 : problem.flowers_per_bouquet = 9)
    (h2 : problem.initial_seeds_per_color = 125)
    (h3 : problem.num_colors = 4)
    (h4 : problem.red_killed = 45)
    (h5 : problem.yellow_killed = 61)
    (h6 : problem.orange_killed = 30)
    (h7 : problem.bouquets_made = 36) :
    purple_flowers_killed problem = 40 := by
  sorry

end NUMINAMATH_CALUDE_purple_flowers_killed_is_40_l2207_220748


namespace NUMINAMATH_CALUDE_prob_three_spades_two_hearts_correct_l2207_220756

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Fin 52)

/-- Represents the suit of a card -/
inductive Suit
| spades | hearts | diamonds | clubs

/-- Represents the rank of a card -/
def Rank := Fin 13

/-- The probability of drawing three spades followed by two hearts from a standard deck -/
def prob_three_spades_two_hearts : ℚ :=
  432 / 6497400

theorem prob_three_spades_two_hearts_correct (d : Deck) :
  prob_three_spades_two_hearts = 
    (13 * 12 * 11 * 13 * 12) / (52 * 51 * 50 * 49 * 48) :=
by sorry

end NUMINAMATH_CALUDE_prob_three_spades_two_hearts_correct_l2207_220756


namespace NUMINAMATH_CALUDE_sum_is_composite_l2207_220717

theorem sum_is_composite (a b : ℕ) (h : 34 * a = 43 * b) : 
  ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ a + b = x * y :=
by sorry

end NUMINAMATH_CALUDE_sum_is_composite_l2207_220717


namespace NUMINAMATH_CALUDE_sum_square_of_sum_and_diff_l2207_220755

theorem sum_square_of_sum_and_diff (x y : ℝ) 
  (sum_eq : x + y = 60) 
  (diff_eq : x - y = 10) : 
  (x + y)^2 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_sum_square_of_sum_and_diff_l2207_220755


namespace NUMINAMATH_CALUDE_bries_slacks_count_l2207_220783

/-- Proves that Brie has 8 slacks given the conditions of the problem -/
theorem bries_slacks_count :
  ∀ (total_blouses total_skirts total_slacks : ℕ)
    (blouses_in_hamper skirts_in_hamper slacks_in_hamper : ℕ)
    (clothes_to_wash : ℕ),
  total_blouses = 12 →
  total_skirts = 6 →
  blouses_in_hamper = (75 * total_blouses) / 100 →
  skirts_in_hamper = (50 * total_skirts) / 100 →
  slacks_in_hamper = (25 * total_slacks) / 100 →
  clothes_to_wash = 14 →
  clothes_to_wash = blouses_in_hamper + skirts_in_hamper + slacks_in_hamper →
  total_slacks = 8 := by
sorry

end NUMINAMATH_CALUDE_bries_slacks_count_l2207_220783


namespace NUMINAMATH_CALUDE_problem_solution_l2207_220781

theorem problem_solution (a b : ℝ) 
  (h : ∀ x : ℝ, x ∈ Set.Icc 0 1 → |a * x + b - Real.sqrt (1 - x^2)| ≤ (Real.sqrt 2 - 1) / 2) : 
  a = -1 ∧ b = (Real.sqrt 2 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2207_220781


namespace NUMINAMATH_CALUDE_f_g_5_l2207_220780

def g (x : ℝ) : ℝ := 4 * x - 5

def f (x : ℝ) : ℝ := 6 * x + 11

theorem f_g_5 : f (g 5) = 101 := by
  sorry

end NUMINAMATH_CALUDE_f_g_5_l2207_220780


namespace NUMINAMATH_CALUDE_short_hair_dog_count_is_six_l2207_220791

/-- Represents the dog grooming scenario -/
structure DogGrooming where
  shortHairDryTime : ℕ
  fullHairDryTime : ℕ
  fullHairDogCount : ℕ
  totalDryTime : ℕ

/-- The number of short-haired dogs in the grooming scenario -/
def shortHairDogCount (dg : DogGrooming) : ℕ :=
  (dg.totalDryTime - dg.fullHairDogCount * dg.fullHairDryTime) / dg.shortHairDryTime

/-- Theorem stating the number of short-haired dogs in the given scenario -/
theorem short_hair_dog_count_is_six :
  let dg : DogGrooming := {
    shortHairDryTime := 10,
    fullHairDryTime := 20,
    fullHairDogCount := 9,
    totalDryTime := 240
  }
  shortHairDogCount dg = 6 := by sorry

end NUMINAMATH_CALUDE_short_hair_dog_count_is_six_l2207_220791


namespace NUMINAMATH_CALUDE_can_cut_one_more_square_l2207_220737

/-- Represents a grid of squares -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a square that can be cut from the grid -/
structure Square :=
  (size : ℕ)

/-- Function to calculate the number of cells in a grid -/
def grid_cells (g : Grid) : ℕ := g.rows * g.cols

/-- Function to calculate the number of cells in a square -/
def square_cells (s : Square) : ℕ := s.size * s.size

/-- Function to calculate the number of cells remaining after cutting squares -/
def remaining_cells (g : Grid) (s : Square) (n : ℕ) : ℕ :=
  grid_cells g - n * square_cells s

/-- Theorem stating that after cutting 99 2x2 squares from a 29x29 grid, 
    at least one more 2x2 square can be cut -/
theorem can_cut_one_more_square (g : Grid) (s : Square) :
  g.rows = 29 → g.cols = 29 → s.size = 2 →
  ∃ (m : ℕ), m > 99 ∧ remaining_cells g s m ≥ square_cells s :=
by sorry

end NUMINAMATH_CALUDE_can_cut_one_more_square_l2207_220737


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2207_220753

theorem trigonometric_identity : 
  Real.sin (347 * π / 180) * Real.cos (148 * π / 180) + 
  Real.sin (77 * π / 180) * Real.cos (58 * π / 180) = 
  Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2207_220753


namespace NUMINAMATH_CALUDE_optimal_triangle_sides_l2207_220765

noncomputable def minTriangleSides (S : ℝ) (x : ℝ) : ℝ × ℝ × ℝ :=
  let BC := 2 * Real.sqrt (S * Real.tan (x / 2))
  let AB := Real.sqrt (S / (Real.sin (x / 2) * Real.cos (x / 2)))
  (BC, AB, AB)

theorem optimal_triangle_sides (S : ℝ) (x : ℝ) (h1 : 0 < S) (h2 : 0 < x) (h3 : x < π) :
  let (BC, AB, AC) := minTriangleSides S x
  BC = 2 * Real.sqrt (S * Real.tan (x / 2)) ∧
  AB = AC ∧
  AB = Real.sqrt (S / (Real.sin (x / 2) * Real.cos (x / 2))) ∧
  ∀ (BC' AB' AC' : ℝ), 
    (BC' * AB' * Real.sin x) / 2 = S → 
    BC' ≥ BC :=
by sorry

end NUMINAMATH_CALUDE_optimal_triangle_sides_l2207_220765


namespace NUMINAMATH_CALUDE_triangle_inequality_l2207_220774

/-- The length of the shortest altitude of a triangle, or 0 if the points are collinear -/
noncomputable def m (P Q R : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

/-- For any four points on a plane, the inequality m(ABC) ≤ m(ABX) + m(AXC) + m(XBC) holds -/
theorem triangle_inequality (A B C X : EuclideanSpace ℝ (Fin 2)) :
  m A B C ≤ m A B X + m A X C + m X B C := by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2207_220774


namespace NUMINAMATH_CALUDE_derivative_x_squared_sin_x_l2207_220794

theorem derivative_x_squared_sin_x :
  ∀ x : ℝ, deriv (λ x => x^2 * Real.sin x) x = 2 * x * Real.sin x + x^2 * Real.cos x :=
by sorry

end NUMINAMATH_CALUDE_derivative_x_squared_sin_x_l2207_220794


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2207_220718

def A : Set ℝ := {x | x < 2}
def B : Set ℝ := {x | 1 < x ∧ x < 7}

theorem union_of_A_and_B : A ∪ B = {x | x < 7} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2207_220718


namespace NUMINAMATH_CALUDE_prime_value_problem_l2207_220746

theorem prime_value_problem : ∃ p : ℕ, 
  Prime p ∧ 
  (5 * p) % 4 = 3 ∧ 
  Prime (13 * p + 2) ∧ 
  13 * p + 2 = 41 := by
sorry

end NUMINAMATH_CALUDE_prime_value_problem_l2207_220746


namespace NUMINAMATH_CALUDE_joan_sofa_cost_l2207_220771

theorem joan_sofa_cost (joan karl : ℕ) 
  (h1 : joan + karl = 600)
  (h2 : 2 * joan = karl + 90) : 
  joan = 230 := by
sorry

end NUMINAMATH_CALUDE_joan_sofa_cost_l2207_220771


namespace NUMINAMATH_CALUDE_sara_savings_l2207_220769

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The number of quarters Sara has -/
def sara_quarters : ℕ := 11

/-- Theorem: Sara's total savings in cents -/
theorem sara_savings : quarter_value * sara_quarters = 275 := by
  sorry

end NUMINAMATH_CALUDE_sara_savings_l2207_220769


namespace NUMINAMATH_CALUDE_odd_function_domain_sum_l2207_220797

def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_domain_sum (f : ℝ → ℝ) (a b : ℝ) 
  (h1 : OddFunction f) 
  (h2 : Set.range f = {-1, 2, a, b}) : 
  a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_domain_sum_l2207_220797


namespace NUMINAMATH_CALUDE_toy_cost_l2207_220705

theorem toy_cost (saved : ℕ) (allowance : ℕ) (num_toys : ℕ) :
  saved = 21 →
  allowance = 15 →
  num_toys = 6 →
  (saved + allowance) / num_toys = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_toy_cost_l2207_220705


namespace NUMINAMATH_CALUDE_ellipse_properties_l2207_220742

/-- An ellipse E with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0

/-- Points on the coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The ellipse problem setup -/
structure EllipseProblem where
  E : Ellipse
  O : Point
  A : Point
  B : Point
  C : Point
  M : Point
  N : Point
  h_O : O.x = 0 ∧ O.y = 0
  h_A : A.x = E.a ∧ A.y = 0
  h_B : B.x = 0 ∧ B.y = E.b
  h_C : C.x = -E.a ∧ C.y = 0
  h_M_on_AB : ∃ t : ℝ, 0 < t ∧ t < 1 ∧ M.x = (1 - t) * A.x + t * B.x ∧ M.y = (1 - t) * A.y + t * B.y
  h_BM_AM : (B.x - M.x)^2 + (B.y - M.y)^2 = 4 * ((M.x - A.x)^2 + (M.y - A.y)^2)
  h_OM_slope : M.y / M.x = Real.sqrt 5 / 10
  h_N_midpoint : N.x = (B.x + C.x) / 2 ∧ N.y = (B.y + C.y) / 2
  h_N_reflection : ∃ S : Point, (S.x - N.x) * E.b = (S.y - N.y) * E.a ∧ S.y = 13/2

/-- The main theorem stating the properties of the ellipse -/
theorem ellipse_properties (prob : EllipseProblem) :
  (prob.E.a^2 - prob.E.b^2) / prob.E.a^2 = 4/5 ∧
  prob.E.a = 3 * Real.sqrt 5 ∧ prob.E.b = 3 :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2207_220742


namespace NUMINAMATH_CALUDE_square_equals_self_only_zero_and_one_l2207_220786

theorem square_equals_self_only_zero_and_one :
  ∀ x : ℝ, x^2 = x ↔ x = 0 ∨ x = 1 := by sorry

end NUMINAMATH_CALUDE_square_equals_self_only_zero_and_one_l2207_220786


namespace NUMINAMATH_CALUDE_badminton_equipment_purchase_l2207_220743

/-- Represents the cost of purchasing badminton equipment from Store A -/
def cost_store_a (x : ℝ) : ℝ := 1760 + 40 * x

/-- Represents the cost of purchasing badminton equipment from Store B -/
def cost_store_b (x : ℝ) : ℝ := 1920 + 32 * x

theorem badminton_equipment_purchase (x : ℝ) (h : x > 16) :
  (x > 20 → cost_store_b x < cost_store_a x) ∧
  (x < 20 → cost_store_a x < cost_store_b x) := by
  sorry

#check badminton_equipment_purchase

end NUMINAMATH_CALUDE_badminton_equipment_purchase_l2207_220743


namespace NUMINAMATH_CALUDE_tangent_length_circle_l2207_220761

/-- The length of the tangent line from a point on a circle to the circle itself -/
theorem tangent_length_circle (x y : ℝ) : 
  x^2 + (y - 2)^2 = 4 → 
  x = 2 → 
  y = 2 → 
  Real.sqrt ((x - 0)^2 + (y - 2)^2 - 4) = 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_length_circle_l2207_220761


namespace NUMINAMATH_CALUDE_smallest_square_ending_2016_l2207_220778

theorem smallest_square_ending_2016 : ∃ (n : ℕ), n = 996 ∧ 
  (∀ (m : ℕ), m < n → m^2 % 10000 ≠ 2016) ∧ n^2 % 10000 = 2016 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_ending_2016_l2207_220778


namespace NUMINAMATH_CALUDE_y₁_gt_y₂_l2207_220763

/-- A linear function f(x) = -2x + 1 -/
def f (x : ℝ) : ℝ := -2 * x + 1

/-- Point A on the graph of f -/
def A : ℝ × ℝ := (1, f 1)

/-- Point B on the graph of f -/
def B : ℝ × ℝ := (3, f 3)

/-- y₁ coordinate of point A -/
def y₁ : ℝ := (A.2)

/-- y₂ coordinate of point B -/
def y₂ : ℝ := (B.2)

theorem y₁_gt_y₂ : y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_y₁_gt_y₂_l2207_220763


namespace NUMINAMATH_CALUDE_bernardo_wins_l2207_220760

def game_winner (N : ℕ) : Prop :=
  N ≤ 999 ∧
  2 * N < 1000 ∧
  2 * N + 75 < 1000 ∧
  4 * N + 150 < 1000 ∧
  4 * N + 225 < 1000 ∧
  8 * N + 450 < 1000 ∧
  8 * N + 525 < 1000 ∧
  16 * N + 1050 ≥ 1000

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem bernardo_wins :
  (∃ N : ℕ, game_winner N ∧ 
    (∀ M : ℕ, M < N → ¬game_winner M) ∧
    N = 56 ∧
    sum_of_digits N = 11) :=
  sorry

end NUMINAMATH_CALUDE_bernardo_wins_l2207_220760


namespace NUMINAMATH_CALUDE_abs_value_difference_l2207_220716

theorem abs_value_difference (a b : ℝ) (ha : |a| = 3) (hb : |b| = 5) (hab : a > b) :
  a - b = 8 := by
  sorry

end NUMINAMATH_CALUDE_abs_value_difference_l2207_220716


namespace NUMINAMATH_CALUDE_sqrt_x_plus_4_meaningful_l2207_220715

theorem sqrt_x_plus_4_meaningful (x : ℝ) : 
  (∃ y : ℝ, y^2 = x + 4) ↔ x ≥ -4 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_4_meaningful_l2207_220715


namespace NUMINAMATH_CALUDE_pascals_triangle_row20_l2207_220733

def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem pascals_triangle_row20 : 
  (binomial 20 6 = 38760) ∧ 
  (binomial 20 6 / binomial 20 2 = 204) := by
sorry

end NUMINAMATH_CALUDE_pascals_triangle_row20_l2207_220733


namespace NUMINAMATH_CALUDE_nancy_jade_amount_l2207_220713

/-- The amount of jade (in grams) needed for a giraffe statue -/
def giraffe_jade : ℝ := 120

/-- The price (in dollars) of a giraffe statue -/
def giraffe_price : ℝ := 150

/-- The amount of jade (in grams) needed for an elephant statue -/
def elephant_jade : ℝ := 2 * giraffe_jade

/-- The price (in dollars) of an elephant statue -/
def elephant_price : ℝ := 350

/-- The additional revenue (in dollars) from making elephant statues instead of giraffe statues -/
def additional_revenue : ℝ := 400

/-- The theorem stating the amount of jade Nancy has -/
theorem nancy_jade_amount :
  ∃ (J : ℝ), J > 0 ∧
    (J / elephant_jade) * elephant_price - (J / giraffe_jade) * giraffe_price = additional_revenue ∧
    J = 1920 := by
  sorry

end NUMINAMATH_CALUDE_nancy_jade_amount_l2207_220713


namespace NUMINAMATH_CALUDE_three_percent_difference_l2207_220722

theorem three_percent_difference (x y : ℝ) 
  (hx : 3 = 0.15 * x) 
  (hy : 3 = 0.10 * y) : 
  x - y = -10 := by
sorry

end NUMINAMATH_CALUDE_three_percent_difference_l2207_220722


namespace NUMINAMATH_CALUDE_specific_hexagon_area_l2207_220703

/-- Hexagon formed by two overlapping equilateral triangles -/
structure Hexagon where
  /-- Side length of the equilateral triangles -/
  side_length : ℝ
  /-- Rotation angle in radians -/
  rotation_angle : ℝ
  /-- The hexagon is symmetric about a central point -/
  symmetric : Bool
  /-- Points A and A' coincide -/
  coincident_points : Bool

/-- Calculate the area of the hexagon -/
def hexagon_area (h : Hexagon) : ℝ :=
  sorry

/-- Theorem stating the area of the specific hexagon -/
theorem specific_hexagon_area :
  let h : Hexagon := {
    side_length := 2,
    rotation_angle := Real.pi / 6,  -- 30 degrees in radians
    symmetric := true,
    coincident_points := true
  }
  hexagon_area h = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_specific_hexagon_area_l2207_220703


namespace NUMINAMATH_CALUDE_madam_arrangements_count_l2207_220726

/-- The number of unique arrangements of the letters in the word MADAM -/
def madam_arrangements : ℕ := 30

/-- The total number of letters in the word MADAM -/
def total_letters : ℕ := 5

/-- The number of times the letter M appears in MADAM -/
def m_count : ℕ := 2

/-- The number of times the letter A appears in MADAM -/
def a_count : ℕ := 2

/-- Theorem stating that the number of unique arrangements of the letters in MADAM is 30 -/
theorem madam_arrangements_count :
  madam_arrangements = Nat.factorial total_letters / (Nat.factorial m_count * Nat.factorial a_count) :=
by sorry

end NUMINAMATH_CALUDE_madam_arrangements_count_l2207_220726


namespace NUMINAMATH_CALUDE_students_not_enrolled_l2207_220727

theorem students_not_enrolled (total : ℕ) (french : ℕ) (german : ℕ) (both : ℕ)
  (h_total : total = 78)
  (h_french : french = 41)
  (h_german : german = 22)
  (h_both : both = 9) :
  total - (french + german - both) = 24 := by
  sorry

end NUMINAMATH_CALUDE_students_not_enrolled_l2207_220727


namespace NUMINAMATH_CALUDE_x_range_l2207_220782

theorem x_range (x : ℝ) (h : |2*x + 1| + |2*x - 5| = 6) : 
  x ∈ Set.Icc (-1/2 : ℝ) (5/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_x_range_l2207_220782


namespace NUMINAMATH_CALUDE_tangent_circles_F_value_l2207_220739

/-- Circle C₁ with equation x² + y² = 1 -/
def C₁ : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}

/-- Circle C₂ with equation x² + y² - 6x - 8y + F = 0 -/
def C₂ (F : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 - 6*p.1 - 8*p.2 + F = 0}

/-- Two circles are internally tangent if the distance between their centers
    is equal to the absolute difference of their radii -/
def internally_tangent (S T : Set (ℝ × ℝ)) : Prop :=
  ∃ (c₁ c₂ : ℝ × ℝ) (r₁ r₂ : ℝ),
    (∀ p, p ∈ S ↔ (p.1 - c₁.1)^2 + (p.2 - c₁.2)^2 = r₁^2) ∧
    (∀ p, p ∈ T ↔ (p.1 - c₂.1)^2 + (p.2 - c₂.2)^2 = r₂^2) ∧
    (c₂.1 - c₁.1)^2 + (c₂.2 - c₁.2)^2 = (r₂ - r₁)^2

/-- If C₁ and C₂ are internally tangent, then F = -11 -/
theorem tangent_circles_F_value :
  internally_tangent C₁ (C₂ F) → F = -11 := by sorry

end NUMINAMATH_CALUDE_tangent_circles_F_value_l2207_220739


namespace NUMINAMATH_CALUDE_difference_of_place_values_l2207_220735

def place_value (digit : ℕ) (place : ℕ) : ℕ := digit * (10 ^ place)

def sum_place_values_27242 : ℕ := place_value 2 0 + place_value 2 2

def sum_place_values_7232062 : ℕ := place_value 2 1 + place_value 2 6

theorem difference_of_place_values : 
  sum_place_values_7232062 - sum_place_values_27242 = 1999818 := by sorry

end NUMINAMATH_CALUDE_difference_of_place_values_l2207_220735


namespace NUMINAMATH_CALUDE_shape_area_l2207_220730

-- Define the shape
structure Shape where
  sides_equal : Bool
  right_angles : Bool
  num_squares : Nat
  small_square_side : Real

-- Define the theorem
theorem shape_area (s : Shape) 
  (h1 : s.sides_equal = true) 
  (h2 : s.right_angles = true) 
  (h3 : s.num_squares = 8) 
  (h4 : s.small_square_side = 2) : 
  s.num_squares * (s.small_square_side * s.small_square_side) = 32 := by
  sorry

end NUMINAMATH_CALUDE_shape_area_l2207_220730


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l2207_220773

/-- Two points are symmetric about a line if their midpoint lies on that line -/
def symmetric_points (x₁ y₁ x₂ y₂ k b : ℝ) : Prop :=
  let mx := (x₁ + x₂) / 2
  let my := (y₁ + y₂) / 2
  k * mx - my + b = 0

/-- The theorem statement -/
theorem symmetric_points_sum (m k : ℝ) :
  symmetric_points 1 2 (-1) m k 3 → m + k = 5 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l2207_220773


namespace NUMINAMATH_CALUDE_cos_negative_23pi_over_4_l2207_220767

theorem cos_negative_23pi_over_4 : Real.cos (-23 * Real.pi / 4) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_23pi_over_4_l2207_220767


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2207_220736

theorem quadratic_inequality (a b c : ℝ) (ha : a ≠ 0)
  (hroot : ∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ 2 * a * x^2 + b * x + c = 0) :
  min c (a + c + 1) ≤ max (|b - a + 1|) (|b + a - 1|) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2207_220736


namespace NUMINAMATH_CALUDE_max_consecutive_indivisible_l2207_220766

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

def is_indivisible (n : ℕ) : Prop :=
  ∀ a b : ℕ, (100 ≤ a ∧ a ≤ 999) → (100 ≤ b ∧ b ≤ 999) → n ≠ a * b

theorem max_consecutive_indivisible :
  ∀ start : ℕ, is_five_digit start →
    ∃ k : ℕ, k ≤ 99 ∧ ¬(is_indivisible (start + k + 1)) :=
by sorry

end NUMINAMATH_CALUDE_max_consecutive_indivisible_l2207_220766


namespace NUMINAMATH_CALUDE_line_equivalence_l2207_220741

/-- Given a line in vector form, prove it's equivalent to a specific slope-intercept form --/
theorem line_equivalence (x y : ℝ) : 
  4 * (x + 2) - 3 * (y - 8) = 0 ↔ y = (4/3) * x + 32/3 := by
  sorry

end NUMINAMATH_CALUDE_line_equivalence_l2207_220741


namespace NUMINAMATH_CALUDE_max_three_roots_l2207_220706

-- Define the function f(x)
def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- State the theorem
theorem max_three_roots 
  (a b c : ℝ) 
  (x₁ x₂ : ℝ) 
  (h1 : ∃ x₁' x₂', x₁' ≠ x₂' ∧ (∀ x, x ≠ x₁' ∧ x ≠ x₂' → (deriv (f a b c)) x ≠ 0)) 
  (h2 : f a b c x₁ = x₁) :
  ∃ S : Finset ℝ, (∀ x, 3*(f a b c x)^2 + 2*a*(f a b c x) + b = 0 ↔ x ∈ S) ∧ Finset.card S ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_max_three_roots_l2207_220706


namespace NUMINAMATH_CALUDE_probability_of_sum_5_is_one_thirty_sixth_l2207_220789

/-- A fair 6-sided die with distinct numbers 1 through 6 -/
def FairDie : Type := Fin 6

/-- The number of dice rolled -/
def numDice : ℕ := 3

/-- The target sum we're aiming for -/
def targetSum : ℕ := 5

/-- The set of all possible outcomes when rolling three fair 6-sided dice -/
def allOutcomes : Finset (FairDie × FairDie × FairDie) := sorry

/-- The set of favorable outcomes (those that sum to targetSum) -/
def favorableOutcomes : Finset (FairDie × FairDie × FairDie) := sorry

/-- The probability of rolling a total of 5 with three fair 6-sided dice -/
def probabilityOfSum5 : ℚ :=
  (Finset.card favorableOutcomes : ℚ) / (Finset.card allOutcomes : ℚ)

/-- Theorem stating that the probability of rolling a sum of 5 with three fair 6-sided dice is 1/36 -/
theorem probability_of_sum_5_is_one_thirty_sixth :
  probabilityOfSum5 = 1 / 36 := by sorry

end NUMINAMATH_CALUDE_probability_of_sum_5_is_one_thirty_sixth_l2207_220789


namespace NUMINAMATH_CALUDE_point_movement_theorem_l2207_220729

/-- Represents the final position of a point on a number line after a series of movements -/
def final_position (initial : Int) (right_move : Int) (left_move : Int) : Int :=
  initial + right_move - left_move

/-- Theorem stating that given the specific movements in the problem, 
    the final position is -5 -/
theorem point_movement_theorem :
  final_position (-3) 5 7 = -5 := by
  sorry

end NUMINAMATH_CALUDE_point_movement_theorem_l2207_220729


namespace NUMINAMATH_CALUDE_negation_false_l2207_220714

theorem negation_false : ¬∃ a : ℝ, a ≤ 2 ∧ a^2 ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_negation_false_l2207_220714


namespace NUMINAMATH_CALUDE_operations_to_equality_l2207_220749

theorem operations_to_equality (a b : ℕ) (h : a = 515 ∧ b = 53) : 
  ∃ n : ℕ, n = 21 ∧ a - 11 * n = b + 11 * n :=
by sorry

end NUMINAMATH_CALUDE_operations_to_equality_l2207_220749


namespace NUMINAMATH_CALUDE_book_reading_time_l2207_220796

/-- Calculates the number of weeks needed to read a book given the total pages and pages read per week. -/
def weeks_to_read (total_pages : ℕ) (pages_per_day : ℕ) (reading_days_per_week : ℕ) : ℕ :=
  total_pages / (pages_per_day * reading_days_per_week)

/-- Proves that it takes 7 weeks to read a 2100-page book when reading 100 pages on 3 days per week. -/
theorem book_reading_time : weeks_to_read 2100 100 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_book_reading_time_l2207_220796


namespace NUMINAMATH_CALUDE_equation_implies_a_equals_four_l2207_220725

theorem equation_implies_a_equals_four (a : ℝ) :
  (∀ x : ℝ, x^2 + a*x + 4 = (x + 2)^2) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_implies_a_equals_four_l2207_220725


namespace NUMINAMATH_CALUDE_box_volume_l2207_220798

/-- The volume of a rectangular box with given dimensions -/
theorem box_volume (height length width : ℝ) 
  (h_height : height = 12)
  (h_length : length = 3 * height)
  (h_width : width = length / 4) :
  height * length * width = 3888 :=
by sorry

end NUMINAMATH_CALUDE_box_volume_l2207_220798


namespace NUMINAMATH_CALUDE_scientific_notation_239000000_l2207_220770

theorem scientific_notation_239000000 :
  239000000 = 2.39 * (10 ^ 8) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_239000000_l2207_220770
