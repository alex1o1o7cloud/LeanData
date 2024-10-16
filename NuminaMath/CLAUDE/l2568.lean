import Mathlib

namespace NUMINAMATH_CALUDE_max_cubes_in_box_l2568_256833

/-- The maximum number of 27 cubic centimetre cubes that can fit in a rectangular box -/
def max_cubes (l w h : ℕ) (cube_volume : ℕ) : ℕ :=
  (l * w * h) / cube_volume

/-- Theorem: The maximum number of 27 cubic centimetre cubes that can fit in a 
    rectangular box measuring 8 cm x 9 cm x 12 cm is 32 -/
theorem max_cubes_in_box : max_cubes 8 9 12 27 = 32 := by
  sorry

end NUMINAMATH_CALUDE_max_cubes_in_box_l2568_256833


namespace NUMINAMATH_CALUDE_fraction_simplification_l2568_256848

theorem fraction_simplification (a b : ℝ) (h1 : a ≠ b) (h2 : b ≠ 0) :
  let x := a^2 / b^2
  (a^2 + b^2) / (a^2 - b^2) = (x + 1) / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2568_256848


namespace NUMINAMATH_CALUDE_even_decreasing_properties_l2568_256837

/-- A function that is even and monotonically decreasing on (0, +∞) -/
def EvenDecreasingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∧ 
  (∀ x y, 0 < x ∧ x < y → f y < f x)

theorem even_decreasing_properties (f : ℝ → ℝ) (hf : EvenDecreasingFunction f) :
  (∃ a : ℝ, f (2 * a) ≥ f (-a)) ∧ 
  (f π ≤ f (-3)) ∧
  (f (-Real.sqrt 3 / 2) < f (4 / 5)) ∧
  (∃ a : ℝ, f (a^2 + 1) ≥ f 1) := by
  sorry

end NUMINAMATH_CALUDE_even_decreasing_properties_l2568_256837


namespace NUMINAMATH_CALUDE_function_rate_comparison_l2568_256890

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - x
def g (x : ℝ) : ℝ := x^2 + x

-- Define the derivatives of f and g
def f' (x : ℝ) : ℝ := 2*x - 1
def g' (x : ℝ) : ℝ := 2*x + 1

theorem function_rate_comparison :
  (∃ x : ℝ, f' x = 2 * g' x ∧ x = -3/2) ∧
  (¬ ∃ x : ℝ, f' x = g' x) := by
  sorry


end NUMINAMATH_CALUDE_function_rate_comparison_l2568_256890


namespace NUMINAMATH_CALUDE_percentage_of_amount_l2568_256800

theorem percentage_of_amount (amount : ℝ) :
  (25 : ℝ) / 100 * amount = 150 → amount = 600 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_amount_l2568_256800


namespace NUMINAMATH_CALUDE_gcd_of_78_and_182_l2568_256858

theorem gcd_of_78_and_182 : Nat.gcd 78 182 = 26 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_78_and_182_l2568_256858


namespace NUMINAMATH_CALUDE_video_game_points_calculation_l2568_256830

/-- Calculate points earned in a video game level --/
theorem video_game_points_calculation
  (points_per_enemy : ℕ)
  (bonus_points : ℕ)
  (total_enemies : ℕ)
  (defeated_enemies : ℕ)
  (bonuses_earned : ℕ)
  (h1 : points_per_enemy = 15)
  (h2 : bonus_points = 50)
  (h3 : total_enemies = 25)
  (h4 : defeated_enemies = total_enemies - 5)
  (h5 : bonuses_earned = 2)
  : defeated_enemies * points_per_enemy + bonuses_earned * bonus_points = 400 := by
  sorry

#check video_game_points_calculation

end NUMINAMATH_CALUDE_video_game_points_calculation_l2568_256830


namespace NUMINAMATH_CALUDE_paper_thickness_l2568_256831

/-- Given that 400 sheets of paper are 4 cm thick, prove that 600 sheets of the same paper would be 6 cm thick. -/
theorem paper_thickness (sheets : ℕ) (thickness : ℝ) 
  (h1 : 400 * (thickness / 400) = 4) -- 400 sheets are 4 cm thick
  (h2 : sheets = 600) -- We want to prove for 600 sheets
  : sheets * (thickness / 400) = 6 := by
  sorry

end NUMINAMATH_CALUDE_paper_thickness_l2568_256831


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2568_256875

theorem solution_set_inequality (x : ℝ) :
  (x + 1) * (x - 2) ≤ 0 ↔ -1 ≤ x ∧ x ≤ 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2568_256875


namespace NUMINAMATH_CALUDE_bart_earnings_l2568_256815

/-- The amount of money Bart receives for each question he answers in a survey. -/
def amount_per_question : ℚ := 1/5

/-- The number of questions in each survey. -/
def questions_per_survey : ℕ := 10

/-- The number of surveys Bart completed on Monday. -/
def monday_surveys : ℕ := 3

/-- The number of surveys Bart completed on Tuesday. -/
def tuesday_surveys : ℕ := 4

/-- The total amount of money Bart earned for the surveys completed on Monday and Tuesday. -/
def total_earnings : ℚ := 14

theorem bart_earnings : 
  amount_per_question * (questions_per_survey * (monday_surveys + tuesday_surveys)) = total_earnings := by
  sorry

end NUMINAMATH_CALUDE_bart_earnings_l2568_256815


namespace NUMINAMATH_CALUDE_find_y_value_l2568_256851

theorem find_y_value (x y : ℝ) (h1 : 1.5 * x = 0.75 * y) (h2 : x = 24) : y = 48 := by
  sorry

end NUMINAMATH_CALUDE_find_y_value_l2568_256851


namespace NUMINAMATH_CALUDE_arctan_sum_l2568_256838

theorem arctan_sum : Real.arctan (3/4) + Real.arctan (4/3) = π / 2 := by sorry

end NUMINAMATH_CALUDE_arctan_sum_l2568_256838


namespace NUMINAMATH_CALUDE_unique_solution_l2568_256842

-- Define the colors
inductive Color
| Red
| Blue

-- Define a structure for clothing
structure Clothing :=
  (tshirt : Color)
  (shorts : Color)

-- Define the children
structure Children :=
  (alyna : Clothing)
  (bohdan : Clothing)
  (vika : Clothing)
  (grysha : Clothing)

-- Define the conditions
def satisfiesConditions (c : Children) : Prop :=
  (c.alyna.tshirt = Color.Red) ∧
  (c.bohdan.tshirt = Color.Red) ∧
  (c.alyna.shorts ≠ c.bohdan.shorts) ∧
  (c.vika.tshirt ≠ c.grysha.tshirt) ∧
  (c.vika.shorts = Color.Blue) ∧
  (c.grysha.shorts = Color.Blue) ∧
  (c.alyna.tshirt ≠ c.vika.tshirt) ∧
  (c.alyna.shorts ≠ c.vika.shorts)

-- Define the correct answer
def correctAnswer : Children :=
  { alyna := { tshirt := Color.Red, shorts := Color.Red },
    bohdan := { tshirt := Color.Red, shorts := Color.Blue },
    vika := { tshirt := Color.Blue, shorts := Color.Blue },
    grysha := { tshirt := Color.Red, shorts := Color.Blue } }

-- Theorem statement
theorem unique_solution :
  ∀ c : Children, satisfiesConditions c → c = correctAnswer :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l2568_256842


namespace NUMINAMATH_CALUDE_petrol_price_increase_l2568_256806

theorem petrol_price_increase (original_price original_consumption : ℝ) 
  (h_positive_price : original_price > 0) 
  (h_positive_consumption : original_consumption > 0) : 
  let consumption_reduction := 23.076923076923073 / 100
  let new_consumption := original_consumption * (1 - consumption_reduction)
  let new_price := original_price * original_consumption / new_consumption
  (new_price - original_price) / original_price = 0.3 := by
sorry

end NUMINAMATH_CALUDE_petrol_price_increase_l2568_256806


namespace NUMINAMATH_CALUDE_H_range_l2568_256891

def H (x : ℝ) : ℝ := |x + 2| - |x - 4| + 3

theorem H_range : 
  (∀ x, 5 ≤ H x ∧ H x ≤ 9) ∧ 
  (∃ x, H x = 9) ∧
  (∀ ε > 0, ∃ x, H x < 5 + ε) :=
sorry

end NUMINAMATH_CALUDE_H_range_l2568_256891


namespace NUMINAMATH_CALUDE_systematic_sampling_l2568_256822

/-- Systematic sampling problem -/
theorem systematic_sampling 
  (total_items : ℕ) 
  (selected_items : ℕ) 
  (first_selected : ℕ) 
  (group_number : ℕ) :
  total_items = 3000 →
  selected_items = 150 →
  first_selected = 11 →
  group_number = 61 →
  (group_number - 1) * (total_items / selected_items) + first_selected = 1211 :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_l2568_256822


namespace NUMINAMATH_CALUDE_amoeba_count_after_ten_days_l2568_256853

/-- The number of amoebas in the puddle after n days -/
def amoeba_count (n : ℕ) : ℕ :=
  3^n

/-- The number of days the amoeba growth process continues -/
def days : ℕ := 10

/-- Theorem: The number of amoebas after 10 days is 59049 -/
theorem amoeba_count_after_ten_days :
  amoeba_count days = 59049 := by
  sorry

end NUMINAMATH_CALUDE_amoeba_count_after_ten_days_l2568_256853


namespace NUMINAMATH_CALUDE_symmetric_line_equation_l2568_256850

-- Define the original line
def original_line (x y : ℝ) : Prop := 4 * x - 3 * y + 5 = 0

-- Define a point on the x-axis
def x_axis_point (x : ℝ) : ℝ × ℝ := (x, 0)

-- Define symmetry with respect to x-axis
def symmetric_wrt_x_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = q.1 ∧ p.2 = -q.2

-- State the theorem
theorem symmetric_line_equation :
  ∀ (x y : ℝ),
  (∃ (x₀ : ℝ), original_line x₀ 0) →
  (∀ (p q : ℝ × ℝ),
    original_line p.1 p.2 →
    symmetric_wrt_x_axis p q →
    4 * q.1 + 3 * q.2 + 5 = 0) :=
sorry

end NUMINAMATH_CALUDE_symmetric_line_equation_l2568_256850


namespace NUMINAMATH_CALUDE_det_A_l2568_256832

def A : Matrix (Fin 4) (Fin 4) ℤ :=
  ![![1, -3, 3, 2],
    ![0, 5, -1, 0],
    ![4, -2, 1, 0],
    ![0, 0, 0, 6]]

theorem det_A : Matrix.det A = -270 := by
  sorry

end NUMINAMATH_CALUDE_det_A_l2568_256832


namespace NUMINAMATH_CALUDE_games_per_month_is_seven_l2568_256872

/-- Represents the number of baseball games in a season. -/
def games_per_season : ℕ := 14

/-- Represents the number of months in a season. -/
def months_per_season : ℕ := 2

/-- Calculates the number of baseball games played in a month. -/
def games_per_month : ℕ := games_per_season / months_per_season

/-- Theorem stating that the number of baseball games played in a month is 7. -/
theorem games_per_month_is_seven : games_per_month = 7 := by sorry

end NUMINAMATH_CALUDE_games_per_month_is_seven_l2568_256872


namespace NUMINAMATH_CALUDE_victor_percentage_marks_l2568_256877

def marks_obtained : ℝ := 240
def maximum_marks : ℝ := 300

theorem victor_percentage_marks : 
  (marks_obtained / maximum_marks) * 100 = 80 := by sorry

end NUMINAMATH_CALUDE_victor_percentage_marks_l2568_256877


namespace NUMINAMATH_CALUDE_abc_divides_sum_power_seven_l2568_256820

theorem abc_divides_sum_power_seven (a b c : ℕ+) 
  (hab : a ∣ b^2) (hbc : b ∣ c^2) (hca : c ∣ a^2) : 
  (a * b * c) ∣ (a + b + c)^7 := by
  sorry

end NUMINAMATH_CALUDE_abc_divides_sum_power_seven_l2568_256820


namespace NUMINAMATH_CALUDE_blue_pens_to_pencils_ratio_l2568_256847

theorem blue_pens_to_pencils_ratio 
  (blue_pens black_pens red_pens pencils : ℕ) : 
  black_pens = blue_pens + 10 →
  pencils = 8 →
  red_pens = pencils - 2 →
  blue_pens + black_pens + red_pens = 48 →
  blue_pens = 2 * pencils :=
by sorry

end NUMINAMATH_CALUDE_blue_pens_to_pencils_ratio_l2568_256847


namespace NUMINAMATH_CALUDE_smallest_sum_reciprocals_l2568_256813

theorem smallest_sum_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 24) :
  ∃ (a b : ℕ+), a ≠ b ∧ (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 24 ∧ (a : ℕ) + b = 98 ∧
  ∀ (c d : ℕ+), c ≠ d → (1 : ℚ) / c + (1 : ℚ) / d = (1 : ℚ) / 24 → (c : ℕ) + d ≥ 98 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_reciprocals_l2568_256813


namespace NUMINAMATH_CALUDE_prob_HTTH_is_one_sixteenth_l2568_256846

/-- The probability of obtaining the sequence HTTH in four consecutive fair coin tosses -/
def prob_HTTH : ℚ := 1 / 16

/-- A fair coin toss is modeled as a probability space with two outcomes -/
structure FairCoin where
  sample_space : Type
  prob : sample_space → ℚ
  head : sample_space
  tail : sample_space
  fair_head : prob head = 1 / 2
  fair_tail : prob tail = 1 / 2
  total_prob : prob head + prob tail = 1

/-- Four consecutive fair coin tosses -/
def four_tosses (c : FairCoin) : Type := c.sample_space × c.sample_space × c.sample_space × c.sample_space

/-- The probability of a specific sequence of four tosses -/
def sequence_prob (c : FairCoin) (s : four_tosses c) : ℚ :=
  c.prob s.1 * c.prob s.2.1 * c.prob s.2.2.1 * c.prob s.2.2.2

/-- Theorem: The probability of obtaining HTTH in four consecutive fair coin tosses is 1/16 -/
theorem prob_HTTH_is_one_sixteenth (c : FairCoin) :
  sequence_prob c (c.head, c.tail, c.tail, c.head) = prob_HTTH := by
  sorry

end NUMINAMATH_CALUDE_prob_HTTH_is_one_sixteenth_l2568_256846


namespace NUMINAMATH_CALUDE_cube_sum_of_roots_l2568_256836

theorem cube_sum_of_roots (p q r : ℝ) : 
  (p^3 - 2*p^2 + p - 3 = 0) → 
  (q^3 - 2*q^2 + q - 3 = 0) → 
  (r^3 - 2*r^2 + r - 3 = 0) → 
  p^3 + q^3 + r^3 = 11 := by sorry

end NUMINAMATH_CALUDE_cube_sum_of_roots_l2568_256836


namespace NUMINAMATH_CALUDE_dave_tickets_proof_l2568_256807

/-- The number of tickets Dave won initially -/
def initial_tickets : ℕ := 11

/-- The number of tickets Dave spent on a beanie -/
def spent_tickets : ℕ := 5

/-- The number of additional tickets Dave won later -/
def additional_tickets : ℕ := 10

/-- The number of tickets Dave has now -/
def current_tickets : ℕ := 16

/-- Theorem stating that the initial number of tickets is correct given the conditions -/
theorem dave_tickets_proof :
  initial_tickets - spent_tickets + additional_tickets = current_tickets :=
by sorry

end NUMINAMATH_CALUDE_dave_tickets_proof_l2568_256807


namespace NUMINAMATH_CALUDE_digit_sum_subtraction_l2568_256811

theorem digit_sum_subtraction (M N P Q : ℕ) : 
  (M ≤ 9 ∧ N ≤ 9 ∧ P ≤ 9 ∧ Q ≤ 9) →
  (10 * M + N) + (10 * P + M) = 10 * Q + N →
  (10 * M + N) - (10 * P + M) = N →
  Q = 0 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_subtraction_l2568_256811


namespace NUMINAMATH_CALUDE_x_squared_minus_2x_minus_3_is_quadratic_l2568_256840

/-- Definition of a quadratic equation -/
def is_quadratic_equation (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ ∀ x, a * x^2 + b * x + c = 0 → true

/-- The equation x² - 2x - 3 = 0 is a quadratic equation -/
theorem x_squared_minus_2x_minus_3_is_quadratic :
  is_quadratic_equation 1 (-2) (-3) := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_2x_minus_3_is_quadratic_l2568_256840


namespace NUMINAMATH_CALUDE_repeating_decimal_ratio_l2568_256892

/-- Represents a repeating decimal with a two-digit repeating part -/
def RepeatingDecimal (a b : ℕ) : ℚ := (10 * a + b : ℚ) / 99

theorem repeating_decimal_ratio :
  (RepeatingDecimal 5 4) / (RepeatingDecimal 1 8) = 3 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_ratio_l2568_256892


namespace NUMINAMATH_CALUDE_train_length_calculation_l2568_256881

/-- Calculates the length of a train given its speed and time to cross a point. -/
def trainLength (speed : Real) (time : Real) : Real :=
  speed * time

theorem train_length_calculation (speed : Real) (time : Real) 
  (h1 : speed = 18 * (1000 / 3600)) -- 18 km/h converted to m/s
  (h2 : time = 200) : 
  trainLength speed time = 1000 := by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_train_length_calculation_l2568_256881


namespace NUMINAMATH_CALUDE_rotate_point_D_l2568_256835

def rotate90Clockwise (x y : ℝ) : ℝ × ℝ := (y, -x)

theorem rotate_point_D : 
  let D : ℝ × ℝ := (-3, 2)
  rotate90Clockwise D.1 D.2 = (2, -3) := by sorry

end NUMINAMATH_CALUDE_rotate_point_D_l2568_256835


namespace NUMINAMATH_CALUDE_smallest_square_tiling_l2568_256896

/-- The smallest square perfectly tiled by 3x4 rectangles -/
def smallest_tiled_square : ℕ := 12

/-- The number of 3x4 rectangles needed to tile the smallest square -/
def num_rectangles : ℕ := 9

/-- The area of a 3x4 rectangle -/
def rectangle_area : ℕ := 3 * 4

theorem smallest_square_tiling :
  (smallest_tiled_square * smallest_tiled_square) % rectangle_area = 0 ∧
  num_rectangles * rectangle_area = smallest_tiled_square * smallest_tiled_square ∧
  ∀ n : ℕ, n < smallest_tiled_square → (n * n) % rectangle_area ≠ 0 := by
  sorry

#check smallest_square_tiling

end NUMINAMATH_CALUDE_smallest_square_tiling_l2568_256896


namespace NUMINAMATH_CALUDE_lisa_pencils_count_l2568_256856

/-- The number of pencils Gloria has initially -/
def gloria_initial : ℕ := 2

/-- The total number of pencils after Lisa gives hers to Gloria -/
def total_pencils : ℕ := 101

/-- The number of pencils Lisa has initially -/
def lisa_initial : ℕ := total_pencils - gloria_initial

theorem lisa_pencils_count : lisa_initial = 99 := by
  sorry

end NUMINAMATH_CALUDE_lisa_pencils_count_l2568_256856


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2568_256859

theorem quadratic_equation_roots (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 + (2*k + 3)*x₁ + k^2 = 0 ∧ 
    x₂^2 + (2*k + 3)*x₂ + k^2 = 0 ∧
    1/x₁ + 1/x₂ = -1) → 
  k = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2568_256859


namespace NUMINAMATH_CALUDE_square_rectangle_area_difference_l2568_256843

theorem square_rectangle_area_difference : 
  let square_side : ℝ := 8
  let rect_length : ℝ := 10
  let rect_width : ℝ := 5
  let square_area := square_side ^ 2
  let rect_area := rect_length * rect_width
  square_area - rect_area = 14 := by
  sorry

end NUMINAMATH_CALUDE_square_rectangle_area_difference_l2568_256843


namespace NUMINAMATH_CALUDE_expansion_equals_cube_l2568_256808

theorem expansion_equals_cube : 16^3 + 3*(16^2)*2 + 3*16*(2^2) + 2^3 = 5832 := by
  sorry

end NUMINAMATH_CALUDE_expansion_equals_cube_l2568_256808


namespace NUMINAMATH_CALUDE_stock_loss_percentage_l2568_256873

theorem stock_loss_percentage (total_stock : ℝ) (profit_percentage : ℝ) (profit_portion : ℝ) (overall_loss : ℝ) :
  total_stock = 12500 →
  profit_percentage = 10 →
  profit_portion = 20 →
  overall_loss = 250 →
  ∃ (L : ℝ),
    overall_loss = (1 - profit_portion / 100) * total_stock * (L / 100) - (profit_portion / 100) * total_stock * (profit_percentage / 100) ∧
    L = 5 := by
  sorry

end NUMINAMATH_CALUDE_stock_loss_percentage_l2568_256873


namespace NUMINAMATH_CALUDE_vector_expression_not_equal_AD_l2568_256895

/-- Given vectors in a plane or space, prove that the expression
    (MB + AD) - BM is not equal to AD. -/
theorem vector_expression_not_equal_AD
  (A B C D M O : EuclideanSpace ℝ (Fin n)) :
  (M - B + (A - D)) - (B - M) ≠ A - D := by sorry

end NUMINAMATH_CALUDE_vector_expression_not_equal_AD_l2568_256895


namespace NUMINAMATH_CALUDE_sum_of_integers_l2568_256810

theorem sum_of_integers (x y : ℕ+) 
  (h1 : x^2 + y^2 = 250)
  (h2 : x * y = 120)
  (h3 : x^2 - y^2 = 130) :
  x + y = 10 * Real.sqrt 4.9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l2568_256810


namespace NUMINAMATH_CALUDE_first_eligible_retirement_year_l2568_256817

/-- Rule of 70 retirement eligibility -/
def rule_of_70 (age : ℕ) (years_employed : ℕ) : Prop :=
  age + years_employed ≥ 70

/-- Year of hire -/
def hire_year : ℕ := 1990

/-- Age at hire -/
def age_at_hire : ℕ := 32

/-- First year of retirement eligibility -/
def retirement_year : ℕ := 2009

/-- Theorem: The employee is first eligible to retire in 2009 -/
theorem first_eligible_retirement_year :
  rule_of_70 (age_at_hire + (retirement_year - hire_year)) (retirement_year - hire_year) ∧
  ∀ (year : ℕ), year < retirement_year → 
    ¬rule_of_70 (age_at_hire + (year - hire_year)) (year - hire_year) :=
by sorry

end NUMINAMATH_CALUDE_first_eligible_retirement_year_l2568_256817


namespace NUMINAMATH_CALUDE_care_package_weight_l2568_256827

theorem care_package_weight (initial_weight : ℝ) (brownies_factor : ℝ) (additional_jelly_beans : ℝ) (gummy_worms_factor : ℝ) :
  initial_weight = 2 ∧
  brownies_factor = 3 ∧
  additional_jelly_beans = 2 ∧
  gummy_worms_factor = 2 →
  (((initial_weight * brownies_factor + additional_jelly_beans) * gummy_worms_factor) : ℝ) = 16 := by
  sorry

end NUMINAMATH_CALUDE_care_package_weight_l2568_256827


namespace NUMINAMATH_CALUDE_zero_in_interval_l2568_256829

def f (x : ℝ) := 2*x + 3*x

theorem zero_in_interval : ∃ x ∈ Set.Ioo (-1 : ℝ) 0, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_in_interval_l2568_256829


namespace NUMINAMATH_CALUDE_range_of_a_l2568_256878

/-- The function f(x) = |x^3 + ax + b| -/
def f (a b x : ℝ) : ℝ := |x^3 + a*x + b|

/-- Theorem stating the range of 'a' given the conditions -/
theorem range_of_a (a b : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 1 → x₂ ∈ Set.Icc 0 1 →
    f a b x₁ - f a b x₂ ≤ 2 * |x₁ - x₂|) →
  a ∈ Set.Icc (-2) (-1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2568_256878


namespace NUMINAMATH_CALUDE_extreme_value_implies_a_l2568_256865

def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 3*x - 9

theorem extreme_value_implies_a (a : ℝ) :
  (∃ (ε : ℝ), ∀ (x : ℝ), |x - (-3)| < ε → f a x ≤ f a (-3) ∨ f a x ≥ f a (-3)) →
  a = 5 :=
sorry

end NUMINAMATH_CALUDE_extreme_value_implies_a_l2568_256865


namespace NUMINAMATH_CALUDE_height_of_taller_tree_l2568_256879

/-- Given two trees with different base elevations, prove that the height of the taller tree is 60 feet. -/
theorem height_of_taller_tree : 
  ∀ (h1 h2 : ℝ), -- heights of the two trees
  h1 > h2 → -- first tree is taller
  h1 - h2 = 20 → -- first tree's top is 20 feet above the second tree's top
  ∃ (b1 b2 : ℝ), -- base elevations of the two trees
    b1 - b2 = 8 ∧ -- base of the first tree is 8 feet higher
    (h1 / (h2 + (b1 - b2))) = 5/4 → -- ratio of heights from respective bases is 4:5
  h1 = 60 := by
sorry

end NUMINAMATH_CALUDE_height_of_taller_tree_l2568_256879


namespace NUMINAMATH_CALUDE_twenty_one_billion_scientific_notation_l2568_256889

/-- The scientific notation representation of 21 billion -/
def twenty_one_billion_scientific : ℝ := 2.1 * (10 ^ 9)

/-- The value of 21 billion -/
def twenty_one_billion : ℝ := 21 * (10 ^ 9)

theorem twenty_one_billion_scientific_notation :
  twenty_one_billion = twenty_one_billion_scientific :=
by sorry

end NUMINAMATH_CALUDE_twenty_one_billion_scientific_notation_l2568_256889


namespace NUMINAMATH_CALUDE_no_repeating_stock_price_l2568_256883

theorem no_repeating_stock_price (n : ℕ) : ¬ ∃ (k l : ℕ), k + l > 0 ∧ k + l ≤ 365 ∧ (1 + n / 100 : ℚ)^k * (1 - n / 100 : ℚ)^l = 1 := by
  sorry

end NUMINAMATH_CALUDE_no_repeating_stock_price_l2568_256883


namespace NUMINAMATH_CALUDE_correct_operation_is_multiplication_by_two_l2568_256825

theorem correct_operation_is_multiplication_by_two (N : ℝ) (x : ℝ) :
  (N / 10 = (5 / 100) * (N * x)) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_is_multiplication_by_two_l2568_256825


namespace NUMINAMATH_CALUDE_pet_shop_stock_worth_l2568_256885

/-- The total worth of the stock in a pet shop -/
def stock_worth (num_puppies num_kittens puppy_price kitten_price : ℕ) : ℕ :=
  num_puppies * puppy_price + num_kittens * kitten_price

/-- Theorem stating that the stock worth is 100 given the specific conditions -/
theorem pet_shop_stock_worth :
  stock_worth 2 4 20 15 = 100 := by
  sorry

end NUMINAMATH_CALUDE_pet_shop_stock_worth_l2568_256885


namespace NUMINAMATH_CALUDE_solution_mixture_problem_l2568_256884

/-- Solution X and Y mixture problem -/
theorem solution_mixture_problem 
  (total : ℝ) (total_pos : 0 < total)
  (x : ℝ) (x_nonneg : 0 ≤ x) (x_le_total : x ≤ total)
  (ha : x * 0.2 + (total - x) * 0.3 = total * 0.22) :
  x / total = 0.8 := by
sorry

end NUMINAMATH_CALUDE_solution_mixture_problem_l2568_256884


namespace NUMINAMATH_CALUDE_promotion_savings_l2568_256860

/-- The price of a single pair of shoes -/
def shoe_price : ℕ := 40

/-- The discount amount for Promotion B -/
def promotion_b_discount : ℕ := 15

/-- Calculate the total cost using Promotion A -/
def cost_promotion_a (price : ℕ) : ℕ :=
  price + price / 2

/-- Calculate the total cost using Promotion B -/
def cost_promotion_b (price : ℕ) (discount : ℕ) : ℕ :=
  price + (price - discount)

/-- Theorem: The difference in cost between Promotion B and Promotion A is $5 -/
theorem promotion_savings : 
  cost_promotion_b shoe_price promotion_b_discount - cost_promotion_a shoe_price = 5 := by
  sorry

end NUMINAMATH_CALUDE_promotion_savings_l2568_256860


namespace NUMINAMATH_CALUDE_petes_nickels_spent_l2568_256874

theorem petes_nickels_spent (total_received : ℕ) (total_spent : ℕ) (raymonds_dimes_left : ℕ) 
  (h1 : total_received = 500)
  (h2 : total_spent = 200)
  (h3 : raymonds_dimes_left = 7) :
  (total_spent - (raymonds_dimes_left * 10)) / 5 = 14 := by
  sorry

end NUMINAMATH_CALUDE_petes_nickels_spent_l2568_256874


namespace NUMINAMATH_CALUDE_simplify_2A_minus_3B_value_2A_minus_3B_l2568_256828

-- Define A and B as functions of x and y
def A (x y : ℝ) : ℝ := 3 * x^2 - x + 2 * y - 4 * x * y
def B (x y : ℝ) : ℝ := 2 * x^2 - 3 * x - y + x * y

-- Theorem for the simplified form of 2A - 3B
theorem simplify_2A_minus_3B (x y : ℝ) :
  2 * A x y - 3 * B x y = 7 * x + 7 * y - 11 * x * y :=
sorry

-- Theorem for the value of 2A - 3B under given conditions
theorem value_2A_minus_3B :
  ∃ (x y : ℝ), x + y = -6/7 ∧ x * y = 1 ∧ 2 * A x y - 3 * B x y = -17 :=
sorry

end NUMINAMATH_CALUDE_simplify_2A_minus_3B_value_2A_minus_3B_l2568_256828


namespace NUMINAMATH_CALUDE_sum_of_four_rationals_l2568_256812

theorem sum_of_four_rationals (a₁ a₂ a₃ a₄ : ℚ) : 
  ({a₁ * a₂, a₁ * a₃, a₁ * a₄, a₂ * a₃, a₂ * a₄, a₃ * a₄} : Set ℚ) = 
    {-24, -2, -3/2, -1/8, 1, 3} → 
  a₁ + a₂ + a₃ + a₄ = 9/4 ∨ a₁ + a₂ + a₃ + a₄ = -9/4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_rationals_l2568_256812


namespace NUMINAMATH_CALUDE_blue_markers_count_l2568_256826

def total_markers : ℕ := 3343
def red_markers : ℕ := 2315

theorem blue_markers_count : total_markers - red_markers = 1028 := by
  sorry

end NUMINAMATH_CALUDE_blue_markers_count_l2568_256826


namespace NUMINAMATH_CALUDE_no_six_odd_reciprocals_sum_to_one_l2568_256834

theorem no_six_odd_reciprocals_sum_to_one :
  ¬ ∃ (a b c d e f : ℕ),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
    d ≠ e ∧ d ≠ f ∧
    e ≠ f ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 ∧
    Odd a ∧ Odd b ∧ Odd c ∧ Odd d ∧ Odd e ∧ Odd f ∧
    1 / a + 1 / b + 1 / c + 1 / d + 1 / e + 1 / f = 1 := by
  sorry


end NUMINAMATH_CALUDE_no_six_odd_reciprocals_sum_to_one_l2568_256834


namespace NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l2568_256863

theorem lcm_of_ratio_and_hcf (a b : ℕ+) : 
  (a : ℚ) / b = 3 / 4 → Nat.gcd a b = 4 → Nat.lcm a b = 48 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l2568_256863


namespace NUMINAMATH_CALUDE_squared_difference_of_quadratic_roots_l2568_256887

theorem squared_difference_of_quadratic_roots :
  ∀ d e : ℝ, (3 * d^2 + 10 * d - 25 = 0) → (3 * e^2 + 10 * e - 25 = 0) →
  (d - e)^2 = 400 / 9 := by
  sorry

end NUMINAMATH_CALUDE_squared_difference_of_quadratic_roots_l2568_256887


namespace NUMINAMATH_CALUDE_oil_leak_calculation_l2568_256819

theorem oil_leak_calculation (total_leak : ℕ) (initial_leak : ℕ) (h1 : total_leak = 11687) (h2 : initial_leak = 6522) :
  total_leak - initial_leak = 5165 := by
  sorry

end NUMINAMATH_CALUDE_oil_leak_calculation_l2568_256819


namespace NUMINAMATH_CALUDE_third_square_is_G_l2568_256809

-- Define the set of squares
inductive Square : Type
| A | B | C | D | E | F | G | H

-- Define the placement order
def PlacementOrder : List Square := [Square.F, Square.H, Square.G, Square.D, Square.A, Square.B, Square.C, Square.E]

-- Define the size of each small square
def SmallSquareSize : Nat := 2

-- Define the size of the large square
def LargeSquareSize : Nat := 4

-- Define the total number of squares
def TotalSquares : Nat := 8

-- Define the visibility property
def IsFullyVisible (s : Square) : Prop := s = Square.E

-- Define the third placed square
def ThirdPlacedSquare : Square := PlacementOrder[2]

-- Theorem statement
theorem third_square_is_G :
  (∀ s : Square, s ≠ Square.E → ¬IsFullyVisible s) →
  IsFullyVisible Square.E →
  TotalSquares = 8 →
  SmallSquareSize = 2 →
  LargeSquareSize = 4 →
  ThirdPlacedSquare = Square.G :=
by sorry

end NUMINAMATH_CALUDE_third_square_is_G_l2568_256809


namespace NUMINAMATH_CALUDE_phoenix_flight_l2568_256868

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r^(n - 1)

theorem phoenix_flight :
  let a₁ := 3
  let r := 3
  ∀ n : ℕ, n < 8 → geometric_sequence a₁ r n ≤ 6560 ∧
  geometric_sequence a₁ r 8 > 6560 :=
by sorry

end NUMINAMATH_CALUDE_phoenix_flight_l2568_256868


namespace NUMINAMATH_CALUDE_square_of_trinomial_l2568_256805

theorem square_of_trinomial (a b c : ℝ) : 
  (a - 2*b - 3*c)^2 = a^2 - 4*a*b + 4*b^2 - 6*a*c + 12*b*c + 9*c^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_trinomial_l2568_256805


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l2568_256866

/-- The speed of a boat in still water, given the rate of current and distance travelled downstream. -/
theorem boat_speed_in_still_water (current_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  current_speed = 4 →
  downstream_distance = 10.4 →
  downstream_time = 24 / 60 →
  ∃ (boat_speed : ℝ), boat_speed = 22 ∧ downstream_distance = (boat_speed + current_speed) * downstream_time :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l2568_256866


namespace NUMINAMATH_CALUDE_speed_of_current_l2568_256871

/-- Calculates the speed of the current given the rowing speed in still water,
    distance covered downstream, and time taken. -/
theorem speed_of_current
  (rowing_speed : ℝ)
  (distance : ℝ)
  (time : ℝ)
  (h1 : rowing_speed = 120)
  (h2 : distance = 0.5)
  (h3 : time = 9.99920006399488 / 3600) :
  rowing_speed + (distance / time - rowing_speed) = 180 :=
by sorry

#check speed_of_current

end NUMINAMATH_CALUDE_speed_of_current_l2568_256871


namespace NUMINAMATH_CALUDE_quadratic_vertex_property_l2568_256857

/-- Given a quadratic function y = x^2 - 2x + n with vertex (m, 1), prove that m - n = -1 -/
theorem quadratic_vertex_property (n m : ℝ) : 
  (∀ x, x^2 - 2*x + n = (x - m)^2 + 1) → m - n = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_vertex_property_l2568_256857


namespace NUMINAMATH_CALUDE_volume_is_12pi_l2568_256823

/-- Represents a solid object with three views and dimensions -/
structure Solid where
  frontView : Real × Real
  sideView : Real × Real
  topView : Real × Real

/-- Calculates the volume of a solid based on its views and dimensions -/
def volumeOfSolid (s : Solid) : Real := sorry

/-- Theorem stating that the volume of the given solid is 12π cm³ -/
theorem volume_is_12pi (s : Solid) : volumeOfSolid s = 12 * Real.pi := by sorry

end NUMINAMATH_CALUDE_volume_is_12pi_l2568_256823


namespace NUMINAMATH_CALUDE_inscribed_circle_rectangle_area_l2568_256880

/-- Proves that the area of a rectangle with an inscribed circle of radius 7 and a length-to-width ratio of 3:1 is equal to 588 -/
theorem inscribed_circle_rectangle_area (r : ℝ) (ratio : ℝ) : 
  r = 7 → ratio = 3 → 2 * r * ratio * 2 * r = 588 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_rectangle_area_l2568_256880


namespace NUMINAMATH_CALUDE_prime_sequence_equality_l2568_256864

theorem prime_sequence_equality (p : ℕ) 
  (h1 : Nat.Prime p) 
  (h2 : Nat.Prime (p + 10)) 
  (h3 : Nat.Prime (p + 14)) 
  (h4 : Nat.Prime (2 * p + 1)) 
  (h5 : Nat.Prime (4 * p + 1)) : p = 3 := by
  sorry

end NUMINAMATH_CALUDE_prime_sequence_equality_l2568_256864


namespace NUMINAMATH_CALUDE_sock_selection_theorem_l2568_256855

theorem sock_selection_theorem : 
  (Finset.univ.filter (fun x : Finset (Fin 8) => x.card = 4)).card = 70 := by
  sorry

end NUMINAMATH_CALUDE_sock_selection_theorem_l2568_256855


namespace NUMINAMATH_CALUDE_fraction_ordering_l2568_256801

theorem fraction_ordering : (4 : ℚ) / 17 < 6 / 25 ∧ 6 / 25 < 8 / 31 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ordering_l2568_256801


namespace NUMINAMATH_CALUDE_rhombus_side_length_l2568_256882

-- Define the rhombus ABCD
structure Rhombus :=
  (A B C D : ℝ × ℝ)

-- Define the conditions
def on_parabola (p : ℝ × ℝ) : Prop :=
  p.2 = p.1^2

def parallel_to_x_axis (p q : ℝ × ℝ) : Prop :=
  p.2 = q.2

def area (r : Rhombus) : ℝ := 128

-- Define the theorem
theorem rhombus_side_length (ABCD : Rhombus) :
  (on_parabola ABCD.A) →
  (on_parabola ABCD.B) →
  (on_parabola ABCD.C) →
  (on_parabola ABCD.D) →
  (ABCD.A = (0, 0)) →
  (parallel_to_x_axis ABCD.B ABCD.C) →
  (area ABCD = 128) →
  Real.sqrt ((ABCD.B.1 - ABCD.C.1)^2 + (ABCD.B.2 - ABCD.C.2)^2) = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_rhombus_side_length_l2568_256882


namespace NUMINAMATH_CALUDE_transform_quadratic_l2568_256898

/-- The original quadratic function -/
def g (x : ℝ) : ℝ := 2 * x^2 + 2

/-- The transformed function -/
def f (x : ℝ) : ℝ := 2 * (x + 3)^2 + 1

/-- Theorem stating that f is the result of transforming g -/
theorem transform_quadratic : 
  ∀ x : ℝ, f x = g (x + 3) - 1 := by sorry

end NUMINAMATH_CALUDE_transform_quadratic_l2568_256898


namespace NUMINAMATH_CALUDE_inverse_proportion_decrease_l2568_256802

theorem inverse_proportion_decrease (x y : ℝ) (k : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = k) :
  let x_new := 1.1 * x
  let y_new := k / x_new
  (y - y_new) / y = 1 / 11 := by sorry

end NUMINAMATH_CALUDE_inverse_proportion_decrease_l2568_256802


namespace NUMINAMATH_CALUDE_fraction_numerator_greater_than_denominator_l2568_256893

theorem fraction_numerator_greater_than_denominator
  (x : ℝ)
  (h1 : -1 ≤ x)
  (h2 : x ≤ 3)
  : 4 * x + 2 > 8 - 3 * x ↔ 6 / 7 < x ∧ x ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_fraction_numerator_greater_than_denominator_l2568_256893


namespace NUMINAMATH_CALUDE_two_digit_divisible_by_3_and_4_with_tens_greater_than_ones_l2568_256867

/-- A function that returns true if a natural number is a two-digit number -/
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- A function that returns the tens digit of a natural number -/
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

/-- A function that returns the ones digit of a natural number -/
def ones_digit (n : ℕ) : ℕ := n % 10

/-- The main theorem to be proved -/
theorem two_digit_divisible_by_3_and_4_with_tens_greater_than_ones :
  ∃! (s : Finset ℕ), 
    s.card = 4 ∧ 
    (∀ n ∈ s, 
      n > 0 ∧ 
      is_two_digit n ∧ 
      n % 3 = 0 ∧ 
      n % 4 = 0 ∧ 
      tens_digit n > ones_digit n) := by
  sorry

end NUMINAMATH_CALUDE_two_digit_divisible_by_3_and_4_with_tens_greater_than_ones_l2568_256867


namespace NUMINAMATH_CALUDE_apple_count_l2568_256876

theorem apple_count (X : ℕ) : ∃ Y : ℕ, Y = 5 * X + 5 :=
  by
  -- Let Sarah_apples be the number of apples Sarah has
  let Sarah_apples := X
  -- Let Jackie_apples be the number of apples Jackie has
  let Jackie_apples := 2 * Sarah_apples
  -- Let Adam_apples be the number of apples Adam has
  let Adam_apples := Jackie_apples + 5
  -- Let Y be the total number of apples
  let Y := Sarah_apples + Jackie_apples + Adam_apples
  -- Prove that Y = 5 * X + 5
  sorry

end NUMINAMATH_CALUDE_apple_count_l2568_256876


namespace NUMINAMATH_CALUDE_total_oreos_count_l2568_256818

/-- The number of Oreos Jordan has -/
def jordan_oreos : ℕ := 11

/-- The number of Oreos James has -/
def james_oreos : ℕ := 2 * jordan_oreos + 3

/-- The total number of Oreos -/
def total_oreos : ℕ := jordan_oreos + james_oreos

theorem total_oreos_count : total_oreos = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_oreos_count_l2568_256818


namespace NUMINAMATH_CALUDE_convex_polyhedron_three_equal_edges_l2568_256899

/-- Represents an edge of a polyhedron --/
structure Edge :=
  (length : ℝ)

/-- Represents a vertex of a polyhedron --/
structure Vertex :=
  (edges : Fin 3 → Edge)

/-- Represents a convex polyhedron --/
structure ConvexPolyhedron :=
  (vertices : Set Vertex)
  (convex : Bool)
  (edge_equality : ∀ v : Vertex, v ∈ vertices → ∃ (i j : Fin 3), i ≠ j ∧ (v.edges i).length = (v.edges j).length)

/-- The main theorem: if a convex polyhedron satisfies the given conditions, it has at least three equal edges --/
theorem convex_polyhedron_three_equal_edges (P : ConvexPolyhedron) : 
  ∃ (e₁ e₂ e₃ : Edge), e₁ ≠ e₂ ∧ e₂ ≠ e₃ ∧ e₁ ≠ e₃ ∧ e₁.length = e₂.length ∧ e₂.length = e₃.length :=
sorry

end NUMINAMATH_CALUDE_convex_polyhedron_three_equal_edges_l2568_256899


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l2568_256894

theorem chess_tournament_participants (n : ℕ) : 
  (n * (n - 1)) / 2 = 171 → n = 19 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l2568_256894


namespace NUMINAMATH_CALUDE_initial_money_l2568_256816

theorem initial_money (x : ℝ) : x + 13 + 3 = 18 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_money_l2568_256816


namespace NUMINAMATH_CALUDE_existence_of_divisible_power_sum_l2568_256821

theorem existence_of_divisible_power_sum (a b : ℕ) (h : b > 1) :
  ∃ n : ℕ, n < b^2 ∧ b ∣ (a^n + n) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_divisible_power_sum_l2568_256821


namespace NUMINAMATH_CALUDE_equation_represents_ellipse_and_hyperbola_l2568_256844

-- Define the equation
def equation (x y : ℝ) : Prop := y^4 - 6*x^4 = 3*y^2 - 2

-- Define what constitutes an ellipse in this context
def is_ellipse (f : ℝ → ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ (∀ x y : ℝ, f x y ↔ y^2 = a*x^2 + b)

-- Define what constitutes a hyperbola in this context
def is_hyperbola (f : ℝ → ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ (∀ x y : ℝ, f x y ↔ y^2 = a*x^2 + b)

-- Theorem statement
theorem equation_represents_ellipse_and_hyperbola :
  (is_ellipse equation) ∧ (is_hyperbola equation) :=
sorry

end NUMINAMATH_CALUDE_equation_represents_ellipse_and_hyperbola_l2568_256844


namespace NUMINAMATH_CALUDE_smallest_a_value_l2568_256870

theorem smallest_a_value (a : ℝ) (h_a_pos : a > 0) : 
  (∀ x > 0, x + a / x ≥ 4) ↔ a ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_a_value_l2568_256870


namespace NUMINAMATH_CALUDE_fraction_equality_l2568_256814

theorem fraction_equality (a b : ℝ) (h : a ≠ -b) : (-a + b) / (-a - b) = (a - b) / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2568_256814


namespace NUMINAMATH_CALUDE_megan_earnings_l2568_256804

/-- The amount of money Megan earned from selling necklaces -/
def money_earned (bead_necklaces gem_necklaces cost_per_necklace : ℕ) : ℕ :=
  (bead_necklaces + gem_necklaces) * cost_per_necklace

/-- Theorem stating that Megan earned 90 dollars from selling necklaces -/
theorem megan_earnings : money_earned 7 3 9 = 90 := by
  sorry

end NUMINAMATH_CALUDE_megan_earnings_l2568_256804


namespace NUMINAMATH_CALUDE_product_in_unit_interval_sufficient_not_necessary_l2568_256852

theorem product_in_unit_interval_sufficient_not_necessary (a b : ℝ) :
  (((0 ≤ a ∧ a ≤ 1) ∧ (0 ≤ b ∧ b ≤ 1)) → (0 ≤ a * b ∧ a * b ≤ 1)) ∧
  ¬(((0 ≤ a * b ∧ a * b ≤ 1) → ((0 ≤ a ∧ a ≤ 1) ∧ (0 ≤ b ∧ b ≤ 1)))) :=
by sorry

end NUMINAMATH_CALUDE_product_in_unit_interval_sufficient_not_necessary_l2568_256852


namespace NUMINAMATH_CALUDE_min_sum_abc_l2568_256803

/-- The number of divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- Given positive integers A, B, C satisfying the conditions, 
    the minimum value of A + B + C is 91 -/
theorem min_sum_abc (A B C : ℕ+) 
  (hA : num_divisors A = 7)
  (hB : num_divisors B = 6)
  (hC : num_divisors C = 3)
  (hAB : num_divisors (A * B) = 24)
  (hBC : num_divisors (B * C) = 10) :
  (∀ A' B' C' : ℕ+, 
    num_divisors A' = 7 → 
    num_divisors B' = 6 → 
    num_divisors C' = 3 → 
    num_divisors (A' * B') = 24 → 
    num_divisors (B' * C') = 10 → 
    A + B + C ≤ A' + B' + C') ∧ 
  A + B + C = 91 := by
sorry

end NUMINAMATH_CALUDE_min_sum_abc_l2568_256803


namespace NUMINAMATH_CALUDE_least_common_multiple_15_36_l2568_256869

theorem least_common_multiple_15_36 : Nat.lcm 15 36 = 180 := by
  sorry

end NUMINAMATH_CALUDE_least_common_multiple_15_36_l2568_256869


namespace NUMINAMATH_CALUDE_smallest_aab_value_exists_valid_digit_pair_l2568_256839

/-- Represents a pair of distinct digits from 1 to 9 -/
structure DigitPair where
  a : Nat
  b : Nat
  a_in_range : a ≥ 1 ∧ a ≤ 9
  b_in_range : b ≥ 1 ∧ b ≤ 9
  distinct : a ≠ b

/-- Converts a DigitPair to a two-digit number -/
def to_two_digit (p : DigitPair) : Nat :=
  10 * p.a + p.b

/-- Converts a DigitPair to a three-digit number AAB -/
def to_three_digit (p : DigitPair) : Nat :=
  100 * p.a + 10 * p.a + p.b

/-- The main theorem stating the smallest possible value of AAB -/
theorem smallest_aab_value (p : DigitPair) 
  (h : to_two_digit p = (to_three_digit p) / 8) : 
  to_three_digit p ≥ 773 := by
  sorry

/-- The existence of a DigitPair satisfying the conditions -/
theorem exists_valid_digit_pair : 
  ∃ p : DigitPair, to_two_digit p = (to_three_digit p) / 8 ∧ to_three_digit p = 773 := by
  sorry

end NUMINAMATH_CALUDE_smallest_aab_value_exists_valid_digit_pair_l2568_256839


namespace NUMINAMATH_CALUDE_collinear_vectors_l2568_256897

/-- Given vectors a and b, if 2a - b is collinear with b, then n = 9 -/
theorem collinear_vectors (a b : ℝ × ℝ) (n : ℝ) 
  (ha : a = (1, 3))
  (hb : b = (3, n))
  (hcol : ∃ (k : ℝ), 2 • a - b = k • b) :
  n = 9 := by
sorry

end NUMINAMATH_CALUDE_collinear_vectors_l2568_256897


namespace NUMINAMATH_CALUDE_average_age_of_three_l2568_256854

/-- The average age of three people given the average age of two of them and the age of the third -/
theorem average_age_of_three (age_a : ℝ) (age_b : ℝ) (age_c : ℝ) 
  (h1 : (age_a + age_c) / 2 = 29) 
  (h2 : age_b = 23) : 
  (age_a + age_b + age_c) / 3 = 27 := by
  sorry


end NUMINAMATH_CALUDE_average_age_of_three_l2568_256854


namespace NUMINAMATH_CALUDE_bouquet_calculation_l2568_256824

def max_bouquets (total_flowers : ℕ) (flowers_per_bouquet : ℕ) (wilted_flowers : ℕ) : ℕ :=
  (total_flowers - wilted_flowers) / flowers_per_bouquet

theorem bouquet_calculation (total_flowers : ℕ) (flowers_per_bouquet : ℕ) (wilted_flowers : ℕ) 
  (h1 : total_flowers = 53)
  (h2 : flowers_per_bouquet = 7)
  (h3 : wilted_flowers = 18) :
  max_bouquets total_flowers flowers_per_bouquet wilted_flowers = 5 := by
  sorry

end NUMINAMATH_CALUDE_bouquet_calculation_l2568_256824


namespace NUMINAMATH_CALUDE_president_and_committee_selection_l2568_256886

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem president_and_committee_selection :
  let total_people : ℕ := 10
  let committee_size : ℕ := 3
  let president_choices : ℕ := total_people
  let committee_choices : ℕ := choose (total_people - 1) committee_size
  president_choices * committee_choices = 840 :=
by sorry

end NUMINAMATH_CALUDE_president_and_committee_selection_l2568_256886


namespace NUMINAMATH_CALUDE_problem_1_l2568_256845

theorem problem_1 : 
  4 + 1/4 - 19/5 + 4/5 + 11/4 = 4 := by sorry

end NUMINAMATH_CALUDE_problem_1_l2568_256845


namespace NUMINAMATH_CALUDE_apple_loss_fraction_l2568_256841

/-- Calculates the fraction of loss given the cost price and selling price -/
def fractionOfLoss (costPrice sellingPrice : ℚ) : ℚ :=
  (costPrice - sellingPrice) / costPrice

theorem apple_loss_fraction :
  fractionOfLoss 19 18 = 1 / 19 := by
  sorry

end NUMINAMATH_CALUDE_apple_loss_fraction_l2568_256841


namespace NUMINAMATH_CALUDE_football_team_size_l2568_256849

/-- Represents the composition of a football team -/
structure FootballTeam where
  total : ℕ
  throwers : ℕ
  rightHanded : ℕ
  leftHanded : ℕ

/-- The properties of our specific football team -/
def ourTeam : FootballTeam where
  total := 70
  throwers := 40
  rightHanded := 60
  leftHanded := 70 - 40 - (60 - 40)

theorem football_team_size : 
  ∀ (team : FootballTeam), 
  team.throwers = 40 ∧ 
  team.rightHanded = 60 ∧ 
  team.leftHanded = (team.total - team.throwers) / 3 ∧
  team.rightHanded = team.throwers + 2 * (team.total - team.throwers) / 3 →
  team.total = 70 := by
  sorry

#check football_team_size

end NUMINAMATH_CALUDE_football_team_size_l2568_256849


namespace NUMINAMATH_CALUDE_guitar_price_proof_l2568_256862

/-- The price Gerald paid for the guitar -/
def gerald_price : ℝ := 250

/-- The price Hendricks paid for the guitar -/
def hendricks_price : ℝ := 200

/-- The percentage discount Hendricks got compared to Gerald's price -/
def discount_percentage : ℝ := 20

theorem guitar_price_proof :
  hendricks_price = gerald_price * (1 - discount_percentage / 100) →
  gerald_price = 250 := by
  sorry

end NUMINAMATH_CALUDE_guitar_price_proof_l2568_256862


namespace NUMINAMATH_CALUDE_unique_number_between_2_and_5_l2568_256861

theorem unique_number_between_2_and_5 (n : ℕ) : 
  2 < n ∧ n < 5 ∧ n < 10 ∧ n < 4 → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_between_2_and_5_l2568_256861


namespace NUMINAMATH_CALUDE_card_probability_l2568_256888

def cards : Finset ℕ := Finset.range 11

def group_A : Finset ℕ := cards.filter (λ x => x % 2 = 1)
def group_B : Finset ℕ := cards.filter (λ x => x % 2 = 0)

def average (a b c : ℕ) : ℚ := (a + b + c : ℚ) / 3

def favorable_outcomes : Finset (ℕ × ℕ) :=
  (group_A.product group_B).filter (λ (a, b) => a + b < 6)

theorem card_probability :
  (favorable_outcomes.card : ℚ) / (group_A.card * group_B.card) = 1 / 10 :=
by sorry

end NUMINAMATH_CALUDE_card_probability_l2568_256888
