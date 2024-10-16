import Mathlib

namespace NUMINAMATH_CALUDE_curve_properties_l208_20843

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a parametric curve in 2D space -/
structure ParametricCurve where
  x : ℝ → ℝ
  y : ℝ → ℝ

def curveC (a : ℝ) : ParametricCurve :=
  { x := λ t => 1 + 2*t,
    y := λ t => a*t^2 }

def pointOnCurve (p : Point2D) (c : ParametricCurve) : Prop :=
  ∃ t : ℝ, c.x t = p.x ∧ c.y t = p.y

theorem curve_properties (a : ℝ) :
  pointOnCurve ⟨3, 1⟩ (curveC a) →
  (a = 1) ∧
  (∀ x y : ℝ, (x - 1)^2 = 4*y ↔ pointOnCurve ⟨x, y⟩ (curveC a)) :=
by sorry

end NUMINAMATH_CALUDE_curve_properties_l208_20843


namespace NUMINAMATH_CALUDE_min_value_theorem_l208_20810

theorem min_value_theorem (x y : ℝ) :
  3 * |x - y| + |2 * x - 5| = x + 1 →
  2 * x + y ≥ 4 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l208_20810


namespace NUMINAMATH_CALUDE_sin_double_alpha_l208_20868

theorem sin_double_alpha (α : Real) 
  (h : Real.cos (α - Real.pi/4) = Real.sqrt 2 / 4) : 
  Real.sin (2 * α) = -3/4 := by
sorry

end NUMINAMATH_CALUDE_sin_double_alpha_l208_20868


namespace NUMINAMATH_CALUDE_euler_family_mean_age_l208_20869

def euler_family_ages : List ℕ := [6, 6, 6, 6, 8, 8, 16]

theorem euler_family_mean_age :
  (euler_family_ages.sum / euler_family_ages.length : ℚ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_euler_family_mean_age_l208_20869


namespace NUMINAMATH_CALUDE_initial_wage_illiterate_employees_l208_20878

/-- Calculates the initial daily average wage of illiterate employees in an NGO. -/
theorem initial_wage_illiterate_employees
  (illiterate_count : ℕ)
  (literate_count : ℕ)
  (new_illiterate_wage : ℚ)
  (average_decrease : ℚ)
  (h1 : illiterate_count = 20)
  (h2 : literate_count = 10)
  (h3 : new_illiterate_wage = 10)
  (h4 : average_decrease = 10) :
  ∃ (initial_wage : ℚ),
    initial_wage = 25 ∧
    (illiterate_count : ℚ) * (initial_wage - new_illiterate_wage) =
      ((illiterate_count : ℚ) + (literate_count : ℚ)) * average_decrease :=
sorry

end NUMINAMATH_CALUDE_initial_wage_illiterate_employees_l208_20878


namespace NUMINAMATH_CALUDE_chess_tournament_win_loss_difference_l208_20872

theorem chess_tournament_win_loss_difference 
  (total_games : ℕ) 
  (total_score : ℚ) 
  (wins : ℕ) 
  (losses : ℕ) 
  (draws : ℕ) :
  total_games = 42 →
  total_score = 30 →
  wins + losses + draws = total_games →
  (wins : ℚ) + (1/2 : ℚ) * draws = total_score →
  wins - losses = 18 :=
by sorry

end NUMINAMATH_CALUDE_chess_tournament_win_loss_difference_l208_20872


namespace NUMINAMATH_CALUDE_unique_prime_in_sequence_l208_20884

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def seven_digit_number (B : ℕ) : ℕ := 9031511 * 10 + B

theorem unique_prime_in_sequence :
  ∃! B : ℕ, B < 10 ∧ is_prime (seven_digit_number B) ∧ seven_digit_number B = 9031517 :=
sorry

end NUMINAMATH_CALUDE_unique_prime_in_sequence_l208_20884


namespace NUMINAMATH_CALUDE_factor_expression_l208_20829

theorem factor_expression (x : ℝ) : 72 * x^11 + 162 * x^22 = 18 * x^11 * (4 + 9 * x^11) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l208_20829


namespace NUMINAMATH_CALUDE_sin_shift_l208_20821

theorem sin_shift (x : ℝ) :
  let f (x : ℝ) := Real.sin (4 * x)
  let g (x : ℝ) := f (x + π / 12)
  let h (x : ℝ) := Real.sin (4 * x + π / 3)
  g = h :=
by sorry

end NUMINAMATH_CALUDE_sin_shift_l208_20821


namespace NUMINAMATH_CALUDE_card_rotation_result_l208_20842

-- Define the positions on the card
inductive Position
  | TopLeft
  | TopRight
  | BottomLeft
  | BottomRight

-- Define the colors of the triangles
inductive Color
  | LightGrey
  | DarkGrey

-- Define the card as a function mapping colors to positions
def Card := Color → Position

-- Define the initial card configuration
def initialCard : Card :=
  fun c => match c with
    | Color.LightGrey => Position.BottomRight
    | Color.DarkGrey => Position.BottomLeft

-- Define the rotation about the lower edge
def rotateLowerEdge (card : Card) : Card :=
  fun c => match card c with
    | Position.BottomLeft => Position.TopLeft
    | Position.BottomRight => Position.TopRight
    | p => p

-- Define the rotation about the right-hand edge
def rotateRightEdge (card : Card) : Card :=
  fun c => match card c with
    | Position.TopRight => Position.TopLeft
    | Position.BottomRight => Position.BottomLeft
    | p => p

-- Theorem statement
theorem card_rotation_result :
  let finalCard := rotateRightEdge (rotateLowerEdge initialCard)
  finalCard Color.LightGrey = Position.TopLeft ∧
  finalCard Color.DarkGrey = Position.TopRight := by
  sorry

end NUMINAMATH_CALUDE_card_rotation_result_l208_20842


namespace NUMINAMATH_CALUDE_diagonals_100_sided_polygon_l208_20883

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: The number of diagonals in a polygon with 100 sides is 4850 -/
theorem diagonals_100_sided_polygon : num_diagonals 100 = 4850 := by
  sorry

end NUMINAMATH_CALUDE_diagonals_100_sided_polygon_l208_20883


namespace NUMINAMATH_CALUDE_number_of_children_selected_l208_20881

def total_boys : ℕ := 5
def total_girls : ℕ := 5
def prob_three_boys_three_girls : ℚ := 100 / 210

theorem number_of_children_selected (n : ℕ) : 
  (total_boys = 5 ∧ total_girls = 5 ∧ 
   prob_three_boys_three_girls = 100 / (Nat.choose (total_boys + total_girls) n)) → 
  n = 6 := by
  sorry

end NUMINAMATH_CALUDE_number_of_children_selected_l208_20881


namespace NUMINAMATH_CALUDE_angle_between_skew_lines_range_l208_20819

-- Define skew lines
structure SkewLine where
  -- We don't need to define the internal structure of a skew line for this problem

-- Define the angle between two skew lines
def angle_between_skew_lines (a b : SkewLine) : ℝ :=
  sorry -- The actual implementation is not needed for the statement

-- Theorem statement
theorem angle_between_skew_lines_range (a b : SkewLine) :
  let θ := angle_between_skew_lines a b
  0 < θ ∧ θ ≤ π/2 :=
sorry

end NUMINAMATH_CALUDE_angle_between_skew_lines_range_l208_20819


namespace NUMINAMATH_CALUDE_triangle_tangent_product_range_l208_20855

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    satisfying a^2 + b^2 + √2ab = c^2, prove that 0 < tan A * tan (2*B) < 1/2 -/
theorem triangle_tangent_product_range (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a^2 + b^2 + Real.sqrt 2 * a * b = c^2 →
  0 < Real.tan A * Real.tan (2 * B) ∧ Real.tan A * Real.tan (2 * B) < 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_tangent_product_range_l208_20855


namespace NUMINAMATH_CALUDE_fraction_evaluation_l208_20822

theorem fraction_evaluation : (1/4 - 1/6) / (1/3 - 1/4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l208_20822


namespace NUMINAMATH_CALUDE_bill_amount_is_1550_l208_20867

/-- Calculates the amount of a bill given its true discount, due date, and interest rate. -/
def bill_amount (true_discount : ℚ) (months : ℚ) (annual_rate : ℚ) : ℚ :=
  let present_value := true_discount / (annual_rate * (months / 12) / (1 + annual_rate * (months / 12)))
  present_value + true_discount

/-- Theorem stating that the bill amount is 1550 given the specified conditions. -/
theorem bill_amount_is_1550 :
  bill_amount 150 9 (16 / 100) = 1550 := by
  sorry

end NUMINAMATH_CALUDE_bill_amount_is_1550_l208_20867


namespace NUMINAMATH_CALUDE_point_in_first_quadrant_l208_20862

/-- Given a complex number Z = 1 + i, prove that the point corresponding to 1/Z + Z 
    lies in the first quadrant. -/
theorem point_in_first_quadrant (Z : ℂ) (h : Z = 1 + Complex.I) : 
  let W := Z⁻¹ + Z
  0 < W.re ∧ 0 < W.im := by
  sorry

end NUMINAMATH_CALUDE_point_in_first_quadrant_l208_20862


namespace NUMINAMATH_CALUDE_ed_hotel_stay_l208_20876

/-- The number of hours Ed stayed in the hotel last night -/
def hours_stayed : ℕ := 6

/-- The cost per hour for staying at night -/
def night_cost_per_hour : ℚ := 3/2

/-- The cost per hour for staying in the morning -/
def morning_cost_per_hour : ℚ := 2

/-- Ed's initial money -/
def initial_money : ℕ := 80

/-- The number of hours Ed stayed in the morning -/
def morning_hours : ℕ := 4

/-- The amount of money Ed had left after paying for his stay -/
def money_left : ℕ := 63

theorem ed_hotel_stay :
  hours_stayed * night_cost_per_hour + 
  morning_hours * morning_cost_per_hour = 
  initial_money - money_left :=
by sorry

end NUMINAMATH_CALUDE_ed_hotel_stay_l208_20876


namespace NUMINAMATH_CALUDE_sum_first_eight_super_nice_l208_20835

def is_prime (n : ℕ) : Prop := sorry

def is_super_nice (n : ℕ) : Prop :=
  (∃ p q r : ℕ, is_prime p ∧ is_prime q ∧ is_prime r ∧ p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ n = p * q * r) ∨
  (∃ p : ℕ, is_prime p ∧ n = p^4)

def first_eight_super_nice : List ℕ :=
  [16, 30, 42, 66, 70, 81, 105, 110]

theorem sum_first_eight_super_nice :
  (∀ n ∈ first_eight_super_nice, is_super_nice n) ∧
  (∀ m : ℕ, m < 16 → ¬is_super_nice m) ∧
  (∀ m : ℕ, m > 110 ∧ is_super_nice m → ∃ n ∈ first_eight_super_nice, m > n) ∧
  (List.sum first_eight_super_nice = 520) :=
by sorry

end NUMINAMATH_CALUDE_sum_first_eight_super_nice_l208_20835


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_negative_l208_20858

theorem sum_of_reciprocals_negative (a b c : ℝ) 
  (sum_zero : a + b + c = 0) 
  (product_pos : a * b * c > 0) : 
  1 / a + 1 / b + 1 / c < 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_negative_l208_20858


namespace NUMINAMATH_CALUDE_sin_cos_identity_l208_20893

theorem sin_cos_identity : 
  Real.sin (20 * π / 180) * Real.cos (10 * π / 180) - 
  Real.cos (160 * π / 180) * Real.cos (80 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l208_20893


namespace NUMINAMATH_CALUDE_two_numbers_sum_and_ratio_l208_20841

theorem two_numbers_sum_and_ratio (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x + y = 900) (h4 : y = 19 * x) : x = 45 ∧ y = 855 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_sum_and_ratio_l208_20841


namespace NUMINAMATH_CALUDE_eggs_remaining_l208_20852

theorem eggs_remaining (initial_eggs : ℕ) (eggs_removed : ℕ) (eggs_left : ℕ) : 
  initial_eggs = 47 → eggs_removed = 5 → eggs_left = initial_eggs - eggs_removed → eggs_left = 42 := by
  sorry

end NUMINAMATH_CALUDE_eggs_remaining_l208_20852


namespace NUMINAMATH_CALUDE_nine_expressions_cover_1_to_13_l208_20879

def nine_expressions : List (ℕ → Prop) :=
  [ (λ n => n = ((9 / 9) ^ (9 - 9))),
    (λ n => n = ((9 / 9) + (9 / 9))),
    (λ n => n = ((9 / 9) + (9 / 9) + (9 / 9))),
    (λ n => n = ((9 / 9) + (9 / 9)) ^ 2),
    (λ n => n = ((9 * 9 + 9) / 9 - 9) + 9),
    (λ n => n = ((9 / 9) + (9 / 9) + (9 / 9)) ^ 2 - 3),
    (λ n => n = ((9 / 9) + (9 / 9)) ^ 2 - (9 / 9)),
    (λ n => n = ((9 / 9) + (9 / 9)) ^ 3 - (9 / 9)),
    (λ n => n = 9),
    (λ n => n = (99 - 9) / 9),
    (λ n => n = 9 + (9 / 9) + (9 / 9)),
    (λ n => n = ((9 / 9) + (9 / 9)) ^ 3 - 4),
    (λ n => n = ((9 / 9) + (9 / 9)) ^ 2 + 9) ]

theorem nine_expressions_cover_1_to_13 :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 13 → ∃ expr ∈ nine_expressions, expr n :=
sorry

end NUMINAMATH_CALUDE_nine_expressions_cover_1_to_13_l208_20879


namespace NUMINAMATH_CALUDE_f_difference_960_480_l208_20807

def sum_of_divisors (n : ℕ) : ℕ := sorry

def f (n : ℕ) : ℚ := (sum_of_divisors n : ℚ) / n

theorem f_difference_960_480 : f 960 - f 480 = 1 / 40 := by sorry

end NUMINAMATH_CALUDE_f_difference_960_480_l208_20807


namespace NUMINAMATH_CALUDE_bobs_muffin_cost_l208_20875

/-- The cost of a single muffin for Bob -/
def muffin_cost (muffins_per_day : ℕ) (days_per_week : ℕ) (selling_price : ℚ) (weekly_profit : ℚ) : ℚ :=
  let total_muffins : ℕ := muffins_per_day * days_per_week
  let total_revenue : ℚ := (total_muffins : ℚ) * selling_price
  let total_cost : ℚ := total_revenue - weekly_profit
  total_cost / (total_muffins : ℚ)

/-- Theorem stating that Bob's muffin cost is $0.75 -/
theorem bobs_muffin_cost :
  muffin_cost 12 7 (3/2) 63 = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_bobs_muffin_cost_l208_20875


namespace NUMINAMATH_CALUDE_miles_difference_l208_20864

/-- Calculates the difference in miles driven between two drivers -/
theorem miles_difference (gervais_avg_daily : ℕ) (gervais_days : ℕ) (henri_total : ℕ) : 
  henri_total - (gervais_avg_daily * gervais_days) = 305 :=
by
  sorry

#check miles_difference 315 3 1250

end NUMINAMATH_CALUDE_miles_difference_l208_20864


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l208_20890

/-- Represents a plane in 3D space -/
structure Plane where
  -- We don't need to define the internal structure of a plane for this problem

/-- Perpendicularity relation between planes -/
def perpendicular (p q : Plane) : Prop :=
  sorry

/-- Parallelism relation between planes -/
def parallel (p q : Plane) : Prop :=
  sorry

theorem sufficient_not_necessary_condition 
  (α β γ : Plane) 
  (h_different : α ≠ β ∧ β ≠ γ ∧ α ≠ γ) 
  (h_perp : perpendicular α γ) :
  (∀ (α β γ : Plane), parallel α β → perpendicular β γ) ∧
  (∃ (α β γ : Plane), perpendicular β γ ∧ ¬parallel α β) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l208_20890


namespace NUMINAMATH_CALUDE_inverse_variation_example_l208_20839

/-- Given two quantities that vary inversely, this function represents their relationship -/
def inverse_variation (k : ℝ) (a b : ℝ) : Prop := a * b = k

/-- Theorem: For inverse variation, if b = 0.5 when a = 800, then b = 0.125 when a = 3200 -/
theorem inverse_variation_example (k : ℝ) :
  inverse_variation k 800 0.5 → inverse_variation k 3200 0.125 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_example_l208_20839


namespace NUMINAMATH_CALUDE_vertical_line_condition_l208_20849

/-- Given two points A and B, if the line AB has an angle of inclination of 90°, then a = 0 -/
theorem vertical_line_condition (a : ℝ) : 
  let A : ℝ × ℝ := (1 + a, 2 * a)
  let B : ℝ × ℝ := (1 - a, 3)
  (A.1 = B.1) →  -- This condition represents a vertical line (90° inclination)
  a = 0 := by
  sorry

end NUMINAMATH_CALUDE_vertical_line_condition_l208_20849


namespace NUMINAMATH_CALUDE_root_quadratic_equation_l208_20834

theorem root_quadratic_equation (m : ℝ) : 
  m^2 - m - 110 = 0 → (m - 1)^2 + m = 111 := by
  sorry

end NUMINAMATH_CALUDE_root_quadratic_equation_l208_20834


namespace NUMINAMATH_CALUDE_fourth_month_sales_l208_20811

def sales_month1 : ℕ := 3435
def sales_month2 : ℕ := 3920
def sales_month3 : ℕ := 3855
def sales_month5 : ℕ := 3560
def sales_month6 : ℕ := 2000
def average_sale : ℕ := 3500
def num_months : ℕ := 6

theorem fourth_month_sales :
  ∃ (sales_month4 : ℕ),
    sales_month4 = 4230 ∧
    (sales_month1 + sales_month2 + sales_month3 + sales_month4 + sales_month5 + sales_month6) / num_months = average_sale :=
by sorry

end NUMINAMATH_CALUDE_fourth_month_sales_l208_20811


namespace NUMINAMATH_CALUDE_simplified_fraction_sum_l208_20886

theorem simplified_fraction_sum (a b : ℕ) (h : a = 49 ∧ b = 84) :
  let (n, d) := (a / gcd a b, b / gcd a b)
  n + d = 19 := by
  sorry

end NUMINAMATH_CALUDE_simplified_fraction_sum_l208_20886


namespace NUMINAMATH_CALUDE_cells_with_three_neighbors_count_l208_20896

/-- Represents a rectangular grid --/
structure RectangularGrid where
  a : ℕ
  b : ℕ
  h_a : a ≥ 3
  h_b : b ≥ 3

/-- Two cells are neighboring if they share a common side --/
def neighboring (grid : RectangularGrid) : Prop := sorry

/-- The number of cells with exactly four neighboring cells --/
def cells_with_four_neighbors (grid : RectangularGrid) : ℕ :=
  (grid.a - 2) * (grid.b - 2)

/-- The number of cells with exactly three neighboring cells --/
def cells_with_three_neighbors (grid : RectangularGrid) : ℕ :=
  2 * (grid.a - 2) + 2 * (grid.b - 2)

/-- Main theorem: In a rectangular grid where 23 cells have exactly four neighboring cells,
    the number of cells with exactly three neighboring cells is 48 --/
theorem cells_with_three_neighbors_count
  (grid : RectangularGrid)
  (h : cells_with_four_neighbors grid = 23) :
  cells_with_three_neighbors grid = 48 := by
  sorry

end NUMINAMATH_CALUDE_cells_with_three_neighbors_count_l208_20896


namespace NUMINAMATH_CALUDE_sum_reciprocal_squares_cubic_l208_20871

theorem sum_reciprocal_squares_cubic (a b c : ℝ) : 
  a^3 - 12*a^2 + 20*a - 3 = 0 →
  b^3 - 12*b^2 + 20*b - 3 = 0 →
  c^3 - 12*c^2 + 20*c - 3 = 0 →
  a ≠ b → b ≠ c → a ≠ c →
  1/a^2 + 1/b^2 + 1/c^2 = 328/9 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocal_squares_cubic_l208_20871


namespace NUMINAMATH_CALUDE_original_price_after_changes_l208_20814

/-- Given an item with original price x, increased by q% and then reduced by r%,
    resulting in a final price of 2 dollars, prove that the original price x
    is equal to 20000 / (10000 + 100 * (q - r) - q * r) -/
theorem original_price_after_changes (q r : ℝ) (x : ℝ) 
    (h1 : x * (1 + q / 100) * (1 - r / 100) = 2) :
  x = 20000 / (10000 + 100 * (q - r) - q * r) := by
  sorry


end NUMINAMATH_CALUDE_original_price_after_changes_l208_20814


namespace NUMINAMATH_CALUDE_tan_sum_product_equals_sqrt_three_l208_20812

theorem tan_sum_product_equals_sqrt_three : 
  Real.tan (17 * π / 180) + Real.tan (43 * π / 180) + 
  Real.sqrt 3 * Real.tan (17 * π / 180) * Real.tan (43 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_product_equals_sqrt_three_l208_20812


namespace NUMINAMATH_CALUDE_rational_solutions_are_integer_l208_20899

theorem rational_solutions_are_integer (a b : ℤ) :
  ∃ (x y : ℚ), y - 2*x = a ∧ y^2 - x*y + x^2 = b →
  ∃ (x' y' : ℤ), (x' : ℚ) = x ∧ (y' : ℚ) = y := by
sorry

end NUMINAMATH_CALUDE_rational_solutions_are_integer_l208_20899


namespace NUMINAMATH_CALUDE_divisible_by_six_l208_20891

theorem divisible_by_six (m : ℕ) : ∃ k : ℤ, (m : ℤ)^3 + 11 * m = 6 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_six_l208_20891


namespace NUMINAMATH_CALUDE_two_day_count_l208_20831

/-- The number of people counted on the second day -/
def second_day_count : ℕ := 500

/-- The number of people counted on the first day -/
def first_day_count : ℕ := 2 * second_day_count

/-- The total number of people counted on both days -/
def total_count : ℕ := first_day_count + second_day_count

theorem two_day_count : total_count = 1500 := by sorry

end NUMINAMATH_CALUDE_two_day_count_l208_20831


namespace NUMINAMATH_CALUDE_wrapping_paper_fraction_l208_20803

theorem wrapping_paper_fraction (total_fraction : ℚ) (num_presents : ℕ) :
  total_fraction = 5/12 ∧ num_presents = 5 →
  total_fraction / num_presents = 1/12 := by
sorry

end NUMINAMATH_CALUDE_wrapping_paper_fraction_l208_20803


namespace NUMINAMATH_CALUDE_driving_equation_correct_l208_20856

/-- Represents a driving trip with a stop -/
structure DrivingTrip where
  speed_before_stop : ℝ
  speed_after_stop : ℝ
  stop_duration : ℝ
  total_distance : ℝ
  total_time : ℝ

/-- The equation for calculating the total distance is correct -/
theorem driving_equation_correct (trip : DrivingTrip) 
  (h1 : trip.speed_before_stop = 60)
  (h2 : trip.speed_after_stop = 90)
  (h3 : trip.stop_duration = 1/2)
  (h4 : trip.total_distance = 270)
  (h5 : trip.total_time = 4) :
  ∃ t : ℝ, 60 * t + 90 * (7/2 - t) = 270 ∧ 
           0 ≤ t ∧ t ≤ trip.total_time - trip.stop_duration :=
by sorry

end NUMINAMATH_CALUDE_driving_equation_correct_l208_20856


namespace NUMINAMATH_CALUDE_book_sharing_probability_l208_20826

/-- The number of students sharing books -/
def num_students : ℕ := 2

/-- The number of books being shared -/
def num_books : ℕ := 3

/-- The total number of possible book distribution scenarios -/
def total_scenarios : ℕ := 8

/-- The number of scenarios where one student gets all books and the other gets none -/
def favorable_scenarios : ℕ := 2

/-- The probability of one student getting all books and the other getting none -/
def probability : ℚ := favorable_scenarios / total_scenarios

theorem book_sharing_probability :
  probability = 1/4 := by sorry

end NUMINAMATH_CALUDE_book_sharing_probability_l208_20826


namespace NUMINAMATH_CALUDE_imaginary_town_population_l208_20860

theorem imaginary_town_population (n m p : ℕ) 
  (h1 : n^2 + 150 = m^2 + 1) 
  (h2 : n^2 + 300 = p^2) : 
  4 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_imaginary_town_population_l208_20860


namespace NUMINAMATH_CALUDE_probability_three_primes_and_at_least_one_eight_l208_20824

/-- A function that returns true if a number is prime, false otherwise -/
def isPrime (n : ℕ) : Bool := sorry

/-- The set of prime numbers on an 8-sided die -/
def primesOnDie : Finset ℕ := {2, 3, 5, 7}

/-- The probability of rolling a prime number on a single 8-sided die -/
def probPrime : ℚ := (Finset.card primesOnDie : ℚ) / 8

/-- The probability of rolling an 8 on a single 8-sided die -/
def probEight : ℚ := 1 / 8

/-- The number of ways to choose 3 dice out of 6 -/
def chooseThreeOutOfSix : ℕ := Nat.choose 6 3

theorem probability_three_primes_and_at_least_one_eight :
  let probExactlyThreePrimes := chooseThreeOutOfSix * probPrime^3 * (1 - probPrime)^3
  let probAtLeastOneEight := 1 - (1 - probEight)^6
  probExactlyThreePrimes * probAtLeastOneEight = 2899900 / 16777216 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_primes_and_at_least_one_eight_l208_20824


namespace NUMINAMATH_CALUDE_race_theorem_l208_20833

/-- Represents a runner in the race -/
structure Runner :=
  (speed : ℝ)

/-- The length of the race in meters -/
def race_length : ℝ := 1000

/-- The distance runner A finishes ahead of runner C -/
def a_ahead_of_c : ℝ := 200

/-- The distance runner B finishes ahead of runner C -/
def b_ahead_of_c : ℝ := 157.89473684210532

theorem race_theorem (A B C : Runner) :
  A.speed > B.speed ∧ B.speed > C.speed →
  a_ahead_of_c = A.speed * race_length / C.speed - race_length →
  b_ahead_of_c = B.speed * race_length / C.speed - race_length →
  A.speed * race_length / B.speed - race_length = a_ahead_of_c - b_ahead_of_c :=
by sorry

end NUMINAMATH_CALUDE_race_theorem_l208_20833


namespace NUMINAMATH_CALUDE_triangle_area_l208_20870

/-- The area of a triangle with vertices at (-2,3), (7,-3), and (4,6) is 31.5 square units. -/
theorem triangle_area : 
  let A : ℝ × ℝ := (-2, 3)
  let B : ℝ × ℝ := (7, -3)
  let C : ℝ × ℝ := (4, 6)
  let area := (1/2) * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)|
  area = 31.5 := by
sorry


end NUMINAMATH_CALUDE_triangle_area_l208_20870


namespace NUMINAMATH_CALUDE_sqrt_calculations_l208_20801

theorem sqrt_calculations : 
  (2 * Real.sqrt 3 - Real.sqrt 12 + Real.sqrt 27 = 3 * Real.sqrt 3) ∧
  ((Real.sqrt 18 - Real.sqrt 3) * Real.sqrt 12 = 6 * Real.sqrt 6 - 6) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_calculations_l208_20801


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l208_20865

/-- Surface area of a rectangular solid -/
def surface_area (length width depth : ℝ) : ℝ :=
  2 * (length * width + length * depth + width * depth)

/-- Theorem: Total surface area of a rectangular solid with given dimensions -/
theorem rectangular_solid_surface_area (a : ℝ) :
  surface_area a (a + 2) (a - 1) = 6 * a^2 + 4 * a - 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l208_20865


namespace NUMINAMATH_CALUDE_tea_shop_problem_l208_20813

/-- Tea shop problem -/
theorem tea_shop_problem 
  (cost_A : ℝ) 
  (cost_B : ℝ) 
  (num_B_more : ℕ) 
  (cost_ratio : ℝ) 
  (total_boxes : ℕ) 
  (sell_price_A : ℝ) 
  (sell_price_B : ℝ) 
  (discount : ℝ) 
  (profit : ℝ)
  (h1 : cost_A = 4000)
  (h2 : cost_B = 8400)
  (h3 : num_B_more = 10)
  (h4 : cost_ratio = 1.4)
  (h5 : total_boxes = 100)
  (h6 : sell_price_A = 300)
  (h7 : sell_price_B = 400)
  (h8 : discount = 0.3)
  (h9 : profit = 5800) :
  ∃ (cost_per_A cost_per_B : ℝ) (num_A num_B : ℕ),
    cost_per_A = 200 ∧ 
    cost_per_B = 280 ∧ 
    num_A = 40 ∧ 
    num_B = 60 ∧
    cost_B / cost_per_B - cost_A / cost_per_A = num_B_more ∧
    cost_per_B = cost_ratio * cost_per_A ∧
    num_A + num_B = total_boxes ∧
    (sell_price_A - cost_per_A) * (num_A / 2) + 
    (sell_price_A * (1 - discount) - cost_per_A) * (num_A / 2) +
    (sell_price_B - cost_per_B) * (num_B / 2) + 
    (sell_price_B * (1 - discount) - cost_per_B) * (num_B / 2) = profit :=
by
  sorry

end NUMINAMATH_CALUDE_tea_shop_problem_l208_20813


namespace NUMINAMATH_CALUDE_book_pricing_and_plans_l208_20800

/-- Represents the cost and quantity of book sets -/
structure BookOrder where
  laoShe : ℕ
  classical : ℕ
  totalCost : ℕ

/-- Represents a purchasing plan -/
structure PurchasePlan where
  laoShe : ℕ
  classical : ℕ

def validPurchasePlan (plan : PurchasePlan) (laoSheCost classical_cost : ℕ) : Prop :=
  plan.laoShe + plan.classical = 20 ∧
  plan.laoShe ≤ 2 * plan.classical ∧
  plan.laoShe * laoSheCost + plan.classical * classical_cost ≤ 1720

theorem book_pricing_and_plans :
  ∃! (laoSheCost classical_cost : ℕ),
    (∀ (order : BookOrder),
      (order.laoShe = 4 ∧ order.classical = 2 ∧ order.totalCost = 480) ∨
      (order.laoShe = 2 ∧ order.classical = 3 ∧ order.totalCost = 520) →
      order.laoShe * laoSheCost + order.classical * classical_cost = order.totalCost) ∧
    (∃! (plans : List PurchasePlan),
      plans.length = 2 ∧
      ∀ plan ∈ plans, validPurchasePlan plan laoSheCost classical_cost) :=
by sorry

end NUMINAMATH_CALUDE_book_pricing_and_plans_l208_20800


namespace NUMINAMATH_CALUDE_f_24_18_mod_89_l208_20895

/-- The function f(x) = x^2 - 2 -/
def f (x : ℤ) : ℤ := x^2 - 2

/-- f^n denotes f applied n times -/
def f_iter (n : ℕ) : ℤ → ℤ :=
  match n with
  | 0 => id
  | n + 1 => f ∘ f_iter n

/-- The main theorem stating that f^24(18) ≡ 47 (mod 89) -/
theorem f_24_18_mod_89 : f_iter 24 18 ≡ 47 [ZMOD 89] := by
  sorry


end NUMINAMATH_CALUDE_f_24_18_mod_89_l208_20895


namespace NUMINAMATH_CALUDE_optimal_layoffs_maximizes_benefit_l208_20838

/-- Represents the number of employees to lay off for maximum economic benefit -/
def optimal_layoffs (a : ℕ) : ℚ :=
  if 70 < a ∧ a ≤ 140 then a - 70
  else if 140 < a ∧ a < 210 then a / 2
  else 0

theorem optimal_layoffs_maximizes_benefit (a b : ℕ) :
  140 < 2 * a ∧ 2 * a < 420 ∧ 
  ∃ k, a = 2 * k ∧
  (∀ x : ℚ, 0 < x ∧ x ≤ a / 2 →
    ((2 * a - x) * (b + 0.01 * b * x) - 0.4 * b * x) ≤
    ((2 * a - optimal_layoffs a) * (b + 0.01 * b * optimal_layoffs a) - 0.4 * b * optimal_layoffs a)) :=
by sorry

end NUMINAMATH_CALUDE_optimal_layoffs_maximizes_benefit_l208_20838


namespace NUMINAMATH_CALUDE_parabola_through_fixed_point_l208_20816

-- Define the line equation
def line_equation (a x y : ℝ) : Prop := (a - 1) * x - y + 2 * a + 1 = 0

-- Define the fixed point P
def fixed_point : ℝ × ℝ := (-2, 3)

-- Theorem statement
theorem parabola_through_fixed_point :
  (∀ a : ℝ, line_equation a (fixed_point.1) (fixed_point.2)) ∧
  (∃ p : ℝ, p > 0 ∧ 
    ((∀ x y : ℝ, (x, y) = fixed_point → y^2 = -2*p*x) ∨
     (∀ x y : ℝ, (x, y) = fixed_point → x^2 = 2*p*y))) :=
sorry

end NUMINAMATH_CALUDE_parabola_through_fixed_point_l208_20816


namespace NUMINAMATH_CALUDE_infinite_geometric_series_sum_l208_20854

/-- The sum of an infinite geometric series with first term 5/3 and common ratio 1/3 is 5/2. -/
theorem infinite_geometric_series_sum :
  let a : ℚ := 5/3  -- First term
  let r : ℚ := 1/3  -- Common ratio
  let S : ℚ := a / (1 - r)  -- Sum formula for infinite geometric series
  S = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_sum_l208_20854


namespace NUMINAMATH_CALUDE_table_tennis_expected_scores_l208_20888

/-- Win probability for a match-up -/
structure MatchProbability where
  team_a_win : ℚ
  team_b_win : ℚ
  sum_to_one : team_a_win + team_b_win = 1

/-- Team scores -/
structure TeamScores where
  team_a : ℕ
  team_b : ℕ
  sum_to_three : team_a + team_b = 3

/-- Expected value of a discrete random variable -/
def expectedValue (probs : List ℚ) (values : List ℚ) : ℚ :=
  (probs.zip values).map (fun (p, v) => p * v) |>.sum

/-- Main theorem -/
theorem table_tennis_expected_scores 
  (match1 : MatchProbability) 
  (match2 : MatchProbability) 
  (match3 : MatchProbability) 
  (h1 : match1.team_a_win = 2/3)
  (h2 : match2.team_a_win = 2/5)
  (h3 : match3.team_a_win = 2/5) :
  let scores := TeamScores
  let ξ_probs := [8/75, 28/75, 2/5, 3/25]
  let ξ_values := [3, 2, 1, 0]
  let η_probs := [3/25, 2/5, 28/75, 8/75]
  let η_values := [3, 2, 1, 0]
  expectedValue ξ_probs ξ_values = 22/15 ∧ 
  expectedValue η_probs η_values = 23/15 := by
sorry


end NUMINAMATH_CALUDE_table_tennis_expected_scores_l208_20888


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_line_line_equation_proof_l208_20861

/-- A line in the 2D plane represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def point_on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def parallel_lines (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem line_through_point_parallel_to_line 
  (given_line : Line) 
  (given_point : Point) 
  (result_line : Line) : Prop :=
  given_line.a = 1 ∧ 
  given_line.b = -2 ∧ 
  given_line.c = 3 ∧
  given_point.x = -1 ∧ 
  given_point.y = 3 ∧
  result_line.a = 1 ∧ 
  result_line.b = -2 ∧ 
  result_line.c = 7 →
  point_on_line given_point result_line ∧ 
  parallel_lines given_line result_line

-- The proof goes here
theorem line_equation_proof : line_through_point_parallel_to_line 
  (Line.mk 1 (-2) 3) 
  (Point.mk (-1) 3) 
  (Line.mk 1 (-2) 7) := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_parallel_to_line_line_equation_proof_l208_20861


namespace NUMINAMATH_CALUDE_smallest_solution_and_ratio_l208_20892

-- Define the equation
def equation (x : ℝ) : Prop := (7 * x / 5) - 2 = 4 / x

-- Define the form of x
def x_form (a b c d : ℤ) (x : ℝ) : Prop :=
  x = (a + b * Real.sqrt c) / d

theorem smallest_solution_and_ratio :
  ∃ (a b c d : ℤ) (x : ℝ),
    equation x ∧
    x_form a b c d x ∧
    (∀ y, equation y → x ≤ y) ∧
    a * c * d / b = -5775 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_and_ratio_l208_20892


namespace NUMINAMATH_CALUDE_fifth_largest_divisor_of_n_l208_20857

def n : ℕ := 1209600000

/-- The fifth-largest divisor of n -/
def fifth_largest_divisor : ℕ := 75600000

/-- A function that returns the kth largest divisor of a number -/
def kth_largest_divisor (m k : ℕ) : ℕ := sorry

theorem fifth_largest_divisor_of_n :
  kth_largest_divisor n 5 = fifth_largest_divisor := by sorry

end NUMINAMATH_CALUDE_fifth_largest_divisor_of_n_l208_20857


namespace NUMINAMATH_CALUDE_coefficient_of_b_l208_20830

theorem coefficient_of_b (a b : ℝ) (h1 : 7 * a = b) (h2 : b = 15) (h3 : 42 * a * b = 675) :
  42 * a = 45 := by
sorry

end NUMINAMATH_CALUDE_coefficient_of_b_l208_20830


namespace NUMINAMATH_CALUDE_ages_sum_l208_20828

/-- Represents the ages of Samantha, Ravi, and Kim -/
structure Ages where
  samantha : ℝ
  ravi : ℝ
  kim : ℝ

/-- The conditions given in the problem -/
def satisfiesConditions (ages : Ages) : Prop :=
  ages.samantha = ages.ravi + 10 ∧
  ages.samantha + 12 = 3 * (ages.ravi - 5) ∧
  ages.kim = ages.ravi / 2

/-- The theorem to be proved -/
theorem ages_sum (ages : Ages) : 
  satisfiesConditions ages → ages.samantha + ages.ravi + ages.kim = 56.25 := by
  sorry


end NUMINAMATH_CALUDE_ages_sum_l208_20828


namespace NUMINAMATH_CALUDE_unique_equidistant_point_l208_20825

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in a 2D plane using the general form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
def Point := ℝ × ℝ

/-- The distance between a point and a line -/
def distancePointToLine (p : Point) (l : Line) : ℝ :=
  sorry

/-- The distance between a point and a circle -/
def distancePointToCircle (p : Point) (c : Circle) : ℝ :=
  sorry

/-- Checks if two lines are parallel -/
def areParallel (l1 l2 : Line) : Prop :=
  sorry

/-- Theorem stating that there is exactly one point equidistant from a circle
    and two parallel tangents under specific conditions -/
theorem unique_equidistant_point
  (c : Circle)
  (l1 l2 : Line)
  (h1 : c.radius = 4)
  (h2 : areParallel l1 l2)
  (h3 : distancePointToLine c.center l1 = 6)
  (h4 : distancePointToLine c.center l2 = 6) :
  ∃! p : Point,
    distancePointToCircle p c = distancePointToLine p l1 ∧
    distancePointToCircle p c = distancePointToLine p l2 :=
  sorry

end NUMINAMATH_CALUDE_unique_equidistant_point_l208_20825


namespace NUMINAMATH_CALUDE_facebook_group_removal_l208_20897

/-- Proves the number of removed members from a Facebook group --/
theorem facebook_group_removal (initial_members : ℕ) (messages_per_day : ℕ) (total_messages_week : ℕ) : 
  initial_members = 150 →
  messages_per_day = 50 →
  total_messages_week = 45500 →
  (initial_members - (initial_members - 20)) * messages_per_day * 7 = total_messages_week :=
by
  sorry

end NUMINAMATH_CALUDE_facebook_group_removal_l208_20897


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l208_20882

theorem quadratic_equation_solution : ∃ x : ℝ, 3 * x^2 - 6 * x + 3 = 0 ∧ x = 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l208_20882


namespace NUMINAMATH_CALUDE_transformed_curve_equation_l208_20837

/-- Given a curve and a scaling transformation, prove the equation of the transformed curve -/
theorem transformed_curve_equation (x y x' y' : ℝ) :
  y = (1/3) * Real.sin (2 * x) →  -- Original curve equation
  x' = 2 * x →                    -- x-scaling
  y' = 3 * y →                    -- y-scaling
  y' = Real.sin x' :=             -- Transformed curve equation
by sorry

end NUMINAMATH_CALUDE_transformed_curve_equation_l208_20837


namespace NUMINAMATH_CALUDE_trig_identity_l208_20840

theorem trig_identity (α β : ℝ) : 
  (Real.cos α - Real.cos β)^2 - (Real.sin α - Real.sin β)^2 = 
  -4 * (Real.sin ((α - β)/2))^2 * Real.cos (α + β) := by sorry

end NUMINAMATH_CALUDE_trig_identity_l208_20840


namespace NUMINAMATH_CALUDE_calculation_proofs_l208_20877

theorem calculation_proofs :
  (1.4 + (-0.2) + 0.6 + (-1.8) = 0) ∧
  ((-1/6 + 3/2 - 5/12) * (-48) = -44) ∧
  ((-1/3)^3 * (-3)^2 * (-1)^2011 = 1/3) ∧
  (-1^3 * (-5) / ((-3)^2 + 2 * (-5)) = -5) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proofs_l208_20877


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l208_20847

def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {x | x^2 - 2*x - 3 < 0}

theorem intersection_of_A_and_B : A ∩ B = Set.Ioo 1 3 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l208_20847


namespace NUMINAMATH_CALUDE_brooke_jumping_jacks_l208_20832

def sidney_monday : ℕ := 20
def sidney_tuesday : ℕ := 36
def sidney_wednesday : ℕ := 40
def sidney_thursday : ℕ := 50

def sidney_total : ℕ := sidney_monday + sidney_tuesday + sidney_wednesday + sidney_thursday

def brooke_multiplier : ℕ := 3

theorem brooke_jumping_jacks : sidney_total * brooke_multiplier = 438 := by
  sorry

end NUMINAMATH_CALUDE_brooke_jumping_jacks_l208_20832


namespace NUMINAMATH_CALUDE_average_weight_of_children_l208_20880

/-- The average weight of all children given the weights of boys, girls, and toddlers -/
theorem average_weight_of_children 
  (num_boys : ℕ) (num_girls : ℕ) (num_toddlers : ℕ)
  (avg_weight_boys : ℝ) (avg_weight_girls : ℝ) (avg_weight_toddlers : ℝ)
  (h_num_boys : num_boys = 8)
  (h_num_girls : num_girls = 5)
  (h_num_toddlers : num_toddlers = 3)
  (h_avg_weight_boys : avg_weight_boys = 160)
  (h_avg_weight_girls : avg_weight_girls = 130)
  (h_avg_weight_toddlers : avg_weight_toddlers = 40)
  (h_total_children : num_boys + num_girls + num_toddlers = 16) :
  let total_weight := num_boys * avg_weight_boys + num_girls * avg_weight_girls + num_toddlers * avg_weight_toddlers
  total_weight / (num_boys + num_girls + num_toddlers) = 128.125 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_of_children_l208_20880


namespace NUMINAMATH_CALUDE_symmetry_axis_of_quadratic_l208_20804

/-- A quadratic function of the form y = (x + h)^2 has a symmetry axis of x = -h -/
theorem symmetry_axis_of_quadratic (h : ℝ) : 
  let f : ℝ → ℝ := λ x => (x + h)^2
  ∀ x : ℝ, f ((-h) - (x - (-h))) = f x := by
  sorry

end NUMINAMATH_CALUDE_symmetry_axis_of_quadratic_l208_20804


namespace NUMINAMATH_CALUDE_man_mass_on_boat_l208_20836

/-- The mass of a man who causes a boat to sink by a certain amount. -/
def mass_of_man (boat_length boat_breadth boat_sink_height : ℝ) : ℝ :=
  boat_length * boat_breadth * boat_sink_height * 1000

/-- Theorem stating that a man who causes a 7m x 3m boat to sink by 1cm has a mass of 210 kg. -/
theorem man_mass_on_boat : 
  mass_of_man 7 3 0.01 = 210 := by sorry

end NUMINAMATH_CALUDE_man_mass_on_boat_l208_20836


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_inequality_l208_20809

/-- A cyclic quadrilateral is a quadrilateral whose vertices all lie on a single circle. -/
structure CyclicQuadrilateral (P : Type*) [MetricSpace P] :=
  (A B C D : P)
  (cyclic : ∃ (center : P) (radius : ℝ), dist center A = radius ∧ dist center B = radius ∧ dist center C = radius ∧ dist center D = radius)
  (distinct : A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A)

/-- The theorem states that in a cyclic quadrilateral ABCD where AB is the longest side,
    the sum of AB and BD is greater than the sum of AC and CD. -/
theorem cyclic_quadrilateral_inequality {P : Type*} [MetricSpace P] (Q : CyclicQuadrilateral P) :
  (∀ X Y : P, dist Q.A Q.B ≥ dist X Y) →
  dist Q.A Q.B + dist Q.B Q.D > dist Q.A Q.C + dist Q.C Q.D :=
sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_inequality_l208_20809


namespace NUMINAMATH_CALUDE_markup_is_ten_l208_20853

/-- Calculates the markup given shop price, tax rate, and profit -/
def calculate_markup (shop_price : ℝ) (tax_rate : ℝ) (profit : ℝ) : ℝ :=
  shop_price - (shop_price * (1 - tax_rate) - profit)

theorem markup_is_ten :
  let shop_price : ℝ := 90
  let tax_rate : ℝ := 0.1
  let profit : ℝ := 1
  calculate_markup shop_price tax_rate profit = 10 := by
sorry

end NUMINAMATH_CALUDE_markup_is_ten_l208_20853


namespace NUMINAMATH_CALUDE_pascal_triangle_formula_l208_20889

-- Define factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Define binomial coefficient
def binomial_coeff (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

-- Theorem statement
theorem pascal_triangle_formula (n k : ℕ) (h : k ≤ n) :
  binomial_coeff n k = factorial n / (factorial k * factorial (n - k)) :=
by sorry

end NUMINAMATH_CALUDE_pascal_triangle_formula_l208_20889


namespace NUMINAMATH_CALUDE_subset_size_bound_l208_20894

/-- Given a natural number n ≥ 2, we define a set A and a family of subsets S with certain properties. -/
theorem subset_size_bound (n : ℕ) (h_n : n ≥ 2) :
  ∃ (A : Finset ℕ) (S : Finset (Finset ℕ)),
    (A = Finset.range (2^(n+1) + 1)) ∧
    (S.card = 2^n) ∧
    (∀ s ∈ S, s ⊆ A) ∧
    (∀ (a b : Finset ℕ) (x y z : ℕ),
      a ∈ S → b ∈ S → x ∈ A → y ∈ A → z ∈ A →
      x < y → y < z → y ∈ a → z ∈ a → x ∈ b → z ∈ b →
      a.card < b.card) →
    ∃ s ∈ S, s.card ≤ 4 * n :=
by
  sorry


end NUMINAMATH_CALUDE_subset_size_bound_l208_20894


namespace NUMINAMATH_CALUDE_specific_ellipse_semi_minor_axis_l208_20827

/-- Represents an ellipse with given properties -/
structure Ellipse where
  center : ℝ × ℝ
  focus : ℝ × ℝ
  semi_major_endpoint : ℝ × ℝ

/-- Calculates the semi-minor axis of an ellipse -/
def semi_minor_axis (e : Ellipse) : ℝ :=
  sorry

/-- Theorem stating the semi-minor axis of the specific ellipse is √21 -/
theorem specific_ellipse_semi_minor_axis :
  let e : Ellipse := {
    center := (0, 0),
    focus := (0, -2),
    semi_major_endpoint := (0, 5)
  }
  semi_minor_axis e = Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_specific_ellipse_semi_minor_axis_l208_20827


namespace NUMINAMATH_CALUDE_system_solution_ratio_l208_20823

theorem system_solution_ratio (k x y z : ℝ) : 
  x ≠ 0 → y ≠ 0 → z ≠ 0 →
  x + k*y + 5*z = 0 →
  4*x + k*y - 3*z = 0 →
  3*x + 5*y - 4*z = 0 →
  x*z / (y^2) = 2/15 := by
sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l208_20823


namespace NUMINAMATH_CALUDE_hexagon_ratio_theorem_l208_20815

/-- Represents a hexagon with specific properties -/
structure Hexagon where
  /-- Total area of the hexagon in square units -/
  total_area : ℝ
  /-- Width of the hexagon -/
  width : ℝ
  /-- Height of the rectangle below PQ -/
  rect_height : ℝ
  /-- Area below PQ -/
  area_below_pq : ℝ
  /-- Ensures the hexagon consists of 7 unit squares -/
  area_constraint : total_area = 7
  /-- Ensures PQ bisects the hexagon area -/
  bisect_constraint : area_below_pq = total_area / 2
  /-- Ensures the triangle base is half the hexagon width -/
  triangle_base_constraint : width / 2 = width - (width / 2)

/-- The main theorem to prove -/
theorem hexagon_ratio_theorem (h : Hexagon) : 
  let xq := (h.area_below_pq - h.width * h.rect_height) / (h.width / 4)
  let qy := h.width - xq
  xq / qy = 3 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_ratio_theorem_l208_20815


namespace NUMINAMATH_CALUDE_expand_polynomial_l208_20850

theorem expand_polynomial (x y : ℝ) : 
  (1 + x^2 + y^3) * (1 - x^3 - y^3) = 1 + x^2 - x^3 - y^3 - x^5 - x^2 * y^3 - x^3 * y^3 - y^6 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l208_20850


namespace NUMINAMATH_CALUDE_least_divisible_by_7_11_13_l208_20863

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem least_divisible_by_7_11_13 : 
  (∀ n : ℕ, n > 0 ∧ is_divisible_by n 7 ∧ is_divisible_by n 11 ∧ is_divisible_by n 13 → n ≥ 1001) ∧ 
  (is_divisible_by 1001 7 ∧ is_divisible_by 1001 11 ∧ is_divisible_by 1001 13) :=
by sorry

end NUMINAMATH_CALUDE_least_divisible_by_7_11_13_l208_20863


namespace NUMINAMATH_CALUDE_marbles_distribution_l208_20845

theorem marbles_distribution (total_marbles : ℕ) (num_boys : ℕ) (marbles_per_boy : ℕ) : 
  total_marbles = 35 → num_boys = 5 → marbles_per_boy = total_marbles / num_boys → marbles_per_boy = 7 := by
  sorry

end NUMINAMATH_CALUDE_marbles_distribution_l208_20845


namespace NUMINAMATH_CALUDE_integer_solutions_of_equation_l208_20851

theorem integer_solutions_of_equation :
  ∀ n : ℤ, (1/3 : ℚ) * n^4 - (1/21 : ℚ) * n^3 - n^2 - (11/21 : ℚ) * n + (4/42 : ℚ) = 0 ↔ n = -1 ∨ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_of_equation_l208_20851


namespace NUMINAMATH_CALUDE_laptop_discount_l208_20898

theorem laptop_discount (initial_discount additional_discount : ℝ) 
  (h1 : initial_discount = 0.3)
  (h2 : additional_discount = 0.5) : 
  1 - (1 - initial_discount) * (1 - additional_discount) = 0.65 := by
  sorry

end NUMINAMATH_CALUDE_laptop_discount_l208_20898


namespace NUMINAMATH_CALUDE_temperature_difference_l208_20806

theorem temperature_difference (M L : ℝ) (N : ℝ) : 
  (M = L + N) →
  (abs ((M - 7) - (L + 5)) = 4) →
  (∃ N₁ N₂ : ℝ, (N = N₁ ∨ N = N₂) ∧ N₁ * N₂ = 128) :=
by sorry

end NUMINAMATH_CALUDE_temperature_difference_l208_20806


namespace NUMINAMATH_CALUDE_max_students_equal_distribution_l208_20818

theorem max_students_equal_distribution (pens toys : ℕ) (h_pens : pens = 451) (h_toys : toys = 410) :
  (∃ (students : ℕ), students > 0 ∧ pens % students = 0 ∧ toys % students = 0 ∧
    ∀ (n : ℕ), n > students → (pens % n ≠ 0 ∨ toys % n ≠ 0)) ↔
  (Nat.gcd pens toys = 41) :=
sorry

end NUMINAMATH_CALUDE_max_students_equal_distribution_l208_20818


namespace NUMINAMATH_CALUDE_area_of_rectangle_l208_20885

-- Define the points
def P : ℝ × ℝ := (0, 4)
def Q : ℝ × ℝ := (3, 4)
def R : ℝ × ℝ := (3, 0)

-- Define the length of PR
def PR_length : ℝ := 5

-- Define the property that PQR is a right triangle
def is_right_triangle (P Q R : ℝ × ℝ) (PR_length : ℝ) : Prop :=
  (Q.1 - P.1)^2 + (R.2 - Q.2)^2 = PR_length^2

-- Define the area of the rectangle
def rectangle_area (P Q R : ℝ × ℝ) : ℝ :=
  (Q.1 - P.1) * (Q.2 - R.2)

-- The theorem to be proved
theorem area_of_rectangle : 
  is_right_triangle P Q R PR_length → rectangle_area P Q R = 12 :=
by sorry

end NUMINAMATH_CALUDE_area_of_rectangle_l208_20885


namespace NUMINAMATH_CALUDE_point_transformation_l208_20805

/-- Rotate a point (x, y) counterclockwise by 90° around (h, k) -/
def rotate90 (x y h k : ℝ) : ℝ × ℝ :=
  (h - (y - k), k + (x - h))

/-- Reflect a point (x, y) about the line y = -x -/
def reflectAboutNegX (x y : ℝ) : ℝ × ℝ :=
  (-y, -x)

theorem point_transformation (c d : ℝ) : 
  let (x₁, y₁) := rotate90 c d 2 3
  let (x₂, y₂) := reflectAboutNegX x₁ y₁
  (x₂ = 7 ∧ y₂ = -10) → d - c = -7 := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_l208_20805


namespace NUMINAMATH_CALUDE_inequality_proof_l208_20820

theorem inequality_proof (a : ℝ) : (a^2 + a + 2) / Real.sqrt (a^2 + a + 1) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l208_20820


namespace NUMINAMATH_CALUDE_prime_factors_sum_l208_20802

theorem prime_factors_sum (w x y z t : ℕ) : 
  2^w * 3^x * 5^y * 7^z * 11^t = 2310 → 2*w + 3*x + 5*y + 7*z + 11*t = 28 := by
sorry

end NUMINAMATH_CALUDE_prime_factors_sum_l208_20802


namespace NUMINAMATH_CALUDE_periodic_function_l208_20859

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def satisfies_condition (f : ℝ → ℝ) : Prop := ∀ x, f x + f (10 - x) = 4

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem periodic_function (f : ℝ → ℝ) 
  (h1 : is_even f) 
  (h2 : satisfies_condition f) : 
  is_periodic f 20 := by
  sorry

end NUMINAMATH_CALUDE_periodic_function_l208_20859


namespace NUMINAMATH_CALUDE_length_to_height_ratio_l208_20873

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  height : ℝ
  width : ℝ
  length : ℝ

/-- Calculates the volume of a box given its dimensions -/
def volume (b : BoxDimensions) : ℝ :=
  b.height * b.width * b.length

/-- Theorem stating the ratio of length to height for a specific box -/
theorem length_to_height_ratio (b : BoxDimensions) :
  b.height = 12 →
  b.length = 4 * b.width →
  volume b = 3888 →
  b.length / b.height = 3 := by
  sorry

end NUMINAMATH_CALUDE_length_to_height_ratio_l208_20873


namespace NUMINAMATH_CALUDE_divisibility_condition_l208_20808

theorem divisibility_condition (a n : ℕ+) : 
  n ∣ ((a + 1)^n.val - a^n.val) ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l208_20808


namespace NUMINAMATH_CALUDE_intersection_implies_sum_l208_20846

def f (x a b : ℝ) : ℝ := -|x - a| + b
def g (x c d : ℝ) : ℝ := |x - c| - d

theorem intersection_implies_sum (a b c d : ℝ) :
  f 1 a b = 4 ∧ g 1 c d = 4 ∧ f 7 a b = 2 ∧ g 7 c d = 2 → a + c = 8 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_sum_l208_20846


namespace NUMINAMATH_CALUDE_pen_purchase_problem_l208_20887

theorem pen_purchase_problem :
  ∀ (x y : ℕ),
    1.7 * (x : ℝ) + 1.2 * (y : ℝ) = 15 →
    x = 6 ∧ y = 4 :=
by sorry

end NUMINAMATH_CALUDE_pen_purchase_problem_l208_20887


namespace NUMINAMATH_CALUDE_probability_one_black_ball_l208_20866

def total_balls : ℕ := 4
def black_balls : ℕ := 2
def white_balls : ℕ := 2
def drawn_balls : ℕ := 2

theorem probability_one_black_ball :
  (Nat.choose black_balls 1 * Nat.choose white_balls 1) / Nat.choose total_balls drawn_balls = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_probability_one_black_ball_l208_20866


namespace NUMINAMATH_CALUDE_magician_min_earnings_l208_20848

/-- Represents the earnings of a magician selling card decks --/
def magician_earnings (initial_decks : ℕ) (remaining_decks : ℕ) (full_price : ℕ) (discounted_price : ℕ) : ℕ :=
  (initial_decks - remaining_decks) * discounted_price

/-- Theorem stating the minimum earnings of the magician --/
theorem magician_min_earnings :
  let initial_decks : ℕ := 15
  let remaining_decks : ℕ := 3
  let full_price : ℕ := 3
  let discounted_price : ℕ := 2
  magician_earnings initial_decks remaining_decks full_price discounted_price ≥ 24 := by
  sorry

#check magician_min_earnings

end NUMINAMATH_CALUDE_magician_min_earnings_l208_20848


namespace NUMINAMATH_CALUDE_point_position_on_line_l208_20844

/-- Given points on a line, prove the position of a point P satisfying a ratio condition -/
theorem point_position_on_line (a b c d e : ℝ) :
  ∀ (O A B C D E P : ℝ),
    O < A ∧ A < B ∧ B < C ∧ C < D ∧  -- Points are ordered on the line
    A - O = 2 * a ∧                  -- OA = 2a
    B - O = b ∧                      -- OB = b
    C - O = 3 * c ∧                  -- OC = 3c
    D - O = d ∧                      -- OD = d
    E - O = e ∧                      -- OE = e
    B ≤ P ∧ P ≤ C ∧                  -- P is between B and C
    (A - P) * (P - E) = (B - P) * (P - C) →  -- AP:PE = BP:PC
  P - O = (b * e - 6 * a * c) / (2 * a + 3 * c - b - e) :=
by sorry

end NUMINAMATH_CALUDE_point_position_on_line_l208_20844


namespace NUMINAMATH_CALUDE_gcd_six_digit_repeated_is_1001_l208_20874

/-- A function that generates a six-digit number by repeating a three-digit number -/
def repeat_three_digit (n : ℕ) : ℕ :=
  1001 * n

/-- The set of all six-digit numbers formed by repeating a three-digit number -/
def six_digit_repeated_set : Set ℕ :=
  {m | ∃ n, 100 ≤ n ∧ n < 1000 ∧ m = repeat_three_digit n}

/-- Theorem stating that the greatest common divisor of all numbers in the set is 1001 -/
theorem gcd_six_digit_repeated_is_1001 :
  ∃ d, d > 0 ∧ (∀ m ∈ six_digit_repeated_set, d ∣ m) ∧
  (∀ k, k > 0 → (∀ m ∈ six_digit_repeated_set, k ∣ m) → k ≤ d) ∧ d = 1001 := by
  sorry

end NUMINAMATH_CALUDE_gcd_six_digit_repeated_is_1001_l208_20874


namespace NUMINAMATH_CALUDE_larger_cuboid_width_l208_20817

/-- Represents the dimensions of a cuboid -/
structure Cuboid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a cuboid -/
def volume (c : Cuboid) : ℝ := c.length * c.width * c.height

/-- The smaller cuboid -/
def small_cuboid : Cuboid := { length := 5, width := 6, height := 3 }

/-- The larger cuboid -/
def large_cuboid (w : ℝ) : Cuboid := { length := 18, width := w, height := 2 }

/-- The number of smaller cuboids that can be formed from the larger cuboid -/
def num_small_cuboids : ℕ := 6

theorem larger_cuboid_width :
  ∃ w : ℝ, volume (large_cuboid w) = num_small_cuboids * volume small_cuboid ∧ w = 15 := by
  sorry

end NUMINAMATH_CALUDE_larger_cuboid_width_l208_20817
