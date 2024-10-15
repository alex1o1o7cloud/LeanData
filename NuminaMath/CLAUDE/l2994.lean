import Mathlib

namespace NUMINAMATH_CALUDE_rabbit_clearing_theorem_l2994_299457

/-- Represents the area one rabbit can clear in a day given the land dimensions, number of rabbits, and days to clear -/
def rabbit_clearing_rate (length width : ℕ) (num_rabbits days_to_clear : ℕ) : ℚ :=
  (length * width : ℚ) / 9 / (num_rabbits * days_to_clear)

/-- Theorem stating that given the specific conditions, one rabbit clears 10 square yards per day -/
theorem rabbit_clearing_theorem :
  rabbit_clearing_rate 200 900 100 20 = 10 := by
  sorry

#eval rabbit_clearing_rate 200 900 100 20

end NUMINAMATH_CALUDE_rabbit_clearing_theorem_l2994_299457


namespace NUMINAMATH_CALUDE_smallest_valid_seating_l2994_299403

/-- Represents a circular table with chairs and people seated. -/
structure CircularTable where
  totalChairs : ℕ
  seatedPeople : ℕ

/-- Checks if the seating arrangement is valid. -/
def isValidSeating (table : CircularTable) : Prop :=
  table.seatedPeople > 0 ∧ 
  table.seatedPeople ≤ table.totalChairs ∧
  ∀ n : ℕ, n ≤ table.seatedPeople → ∃ m : ℕ, m < n ∧ (n - m = 1 ∨ m - n = 1 ∨ n = 1)

/-- The theorem to be proved. -/
theorem smallest_valid_seating (table : CircularTable) :
  table.totalChairs = 75 →
  (isValidSeating table ∧ ∀ t : CircularTable, t.totalChairs = 75 → isValidSeating t → t.seatedPeople ≥ table.seatedPeople) →
  table.seatedPeople = 25 := by
  sorry

end NUMINAMATH_CALUDE_smallest_valid_seating_l2994_299403


namespace NUMINAMATH_CALUDE_first_fund_profit_percentage_l2994_299402

/-- Proves that the profit percentage of the first mutual fund is approximately 2.82% given the specified conditions --/
theorem first_fund_profit_percentage 
  (total_investment : ℝ) 
  (investment_higher_profit : ℝ) 
  (second_fund_profit : ℝ) 
  (total_profit : ℝ) 
  (h1 : total_investment = 1900)
  (h2 : investment_higher_profit = 1700)
  (h3 : second_fund_profit = 0.02)
  (h4 : total_profit = 52)
  : ∃ (first_fund_profit : ℝ), 
    (first_fund_profit * investment_higher_profit + 
     second_fund_profit * (total_investment - investment_higher_profit) = total_profit) ∧
    (abs (first_fund_profit - 0.0282) < 0.0001) :=
by sorry

end NUMINAMATH_CALUDE_first_fund_profit_percentage_l2994_299402


namespace NUMINAMATH_CALUDE_initial_mean_equals_correct_mean_l2994_299426

/-- Proves that the initial mean is equal to the correct mean when one value is incorrectly copied --/
theorem initial_mean_equals_correct_mean (n : ℕ) (correct_value incorrect_value : ℝ) (correct_mean : ℝ) :
  n = 25 →
  correct_value = 165 →
  incorrect_value = 130 →
  correct_mean = 191.4 →
  (n * correct_mean - correct_value + incorrect_value) / n = correct_mean := by
  sorry

#check initial_mean_equals_correct_mean

end NUMINAMATH_CALUDE_initial_mean_equals_correct_mean_l2994_299426


namespace NUMINAMATH_CALUDE_max_b_for_zero_in_range_l2994_299449

/-- The maximum value of b such that 0 is in the range of the quadratic function g(x) = x^2 - 7x + b is 49/4. -/
theorem max_b_for_zero_in_range : 
  ∀ b : ℝ, (∃ x : ℝ, x^2 - 7*x + b = 0) ↔ b ≤ 49/4 :=
sorry

end NUMINAMATH_CALUDE_max_b_for_zero_in_range_l2994_299449


namespace NUMINAMATH_CALUDE_revenue_calculation_l2994_299482

/-- Calculates the total revenue given the salary expense and ratio of salary to stock purchase --/
def total_revenue (salary_expense : ℚ) (salary_ratio : ℚ) (stock_ratio : ℚ) : ℚ :=
  salary_expense * (salary_ratio + stock_ratio) / salary_ratio

/-- Proves that the total revenue is 3000 given the conditions --/
theorem revenue_calculation :
  let salary_expense : ℚ := 800
  let salary_ratio : ℚ := 4
  let stock_ratio : ℚ := 11
  total_revenue salary_expense salary_ratio stock_ratio = 3000 := by
sorry

#eval total_revenue 800 4 11

end NUMINAMATH_CALUDE_revenue_calculation_l2994_299482


namespace NUMINAMATH_CALUDE_shaded_fraction_is_one_fourth_l2994_299443

/-- Represents a square board with shaded regions -/
structure Board :=
  (size : ℕ)
  (shaded_area : ℚ)

/-- Calculates the fraction of shaded area on the board -/
def shaded_fraction (b : Board) : ℚ :=
  b.shaded_area / (b.size * b.size : ℚ)

/-- Represents the specific board configuration described in the problem -/
def problem_board : Board :=
  { size := 4,
    shaded_area := 4 }

/-- Theorem stating that the shaded fraction of the problem board is 1/4 -/
theorem shaded_fraction_is_one_fourth :
  shaded_fraction problem_board = 1/4 := by
  sorry

#check shaded_fraction_is_one_fourth

end NUMINAMATH_CALUDE_shaded_fraction_is_one_fourth_l2994_299443


namespace NUMINAMATH_CALUDE_paint_mixture_fraction_l2994_299466

theorem paint_mixture_fraction (original_intensity replacement_intensity new_intensity : ℝ) 
  (h1 : original_intensity = 0.5)
  (h2 : replacement_intensity = 0.25)
  (h3 : new_intensity = 0.45) :
  ∃ (x : ℝ), 
    x ≥ 0 ∧ x ≤ 1 ∧
    original_intensity * (1 - x) + replacement_intensity * x = new_intensity ∧
    x = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_paint_mixture_fraction_l2994_299466


namespace NUMINAMATH_CALUDE_population_problem_l2994_299408

theorem population_problem : ∃ (n : ℕ), 
  (∃ (m k : ℕ), 
    (n^2 + 200 = m^2 + 1) ∧ 
    (n^2 + 500 = k^2) ∧ 
    (21 ∣ n^2) ∧ 
    (n^2 = 9801)) := by
  sorry

end NUMINAMATH_CALUDE_population_problem_l2994_299408


namespace NUMINAMATH_CALUDE_translation_theorem_l2994_299452

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Translation of a point in 2D space -/
def translate (p : Point2D) (dx dy : ℝ) : Point2D :=
  ⟨p.x + dx, p.y + dy⟩

/-- Theorem: Given a translation of line segment AB to A'B', 
    if A(-2,3) corresponds to A'(3,2) and B corresponds to B'(4,0), 
    then the coordinates of B are (-1,1) -/
theorem translation_theorem 
  (A : Point2D) (A' : Point2D) (B' : Point2D)
  (h1 : A = ⟨-2, 3⟩)
  (h2 : A' = ⟨3, 2⟩)
  (h3 : B' = ⟨4, 0⟩)
  (h4 : ∃ (dx dy : ℝ), A' = translate A dx dy ∧ B' = translate ⟨-1, 1⟩ dx dy) :
  ∃ (B : Point2D), B = ⟨-1, 1⟩ ∧ B' = translate B (A'.x - A.x) (A'.y - A.y) :=
sorry

end NUMINAMATH_CALUDE_translation_theorem_l2994_299452


namespace NUMINAMATH_CALUDE_fifteenth_shape_black_tiles_l2994_299438

/-- The dimension of the nth shape in the sequence -/
def shape_dimension (n : ℕ) : ℕ := 2 * n - 1

/-- The total number of tiles in the nth shape -/
def total_tiles (n : ℕ) : ℕ := (shape_dimension n) ^ 2

/-- The number of black tiles in the nth shape -/
def black_tiles (n : ℕ) : ℕ := (total_tiles n + 1) / 2

theorem fifteenth_shape_black_tiles :
  black_tiles 15 = 421 := by sorry

end NUMINAMATH_CALUDE_fifteenth_shape_black_tiles_l2994_299438


namespace NUMINAMATH_CALUDE_grass_eating_problem_l2994_299436

/-- Amount of grass on one hectare initially -/
def initial_grass : ℝ := sorry

/-- Amount of grass that regrows on one hectare in one week -/
def grass_regrowth : ℝ := sorry

/-- Amount of grass one cow eats in one week -/
def cow_consumption : ℝ := sorry

/-- Number of cows that eat all grass on given hectares in given weeks -/
def cows_needed (hectares weeks : ℕ) : ℕ := sorry

theorem grass_eating_problem :
  (3 : ℕ) * cow_consumption * 2 = 2 * initial_grass + 4 * grass_regrowth ∧
  (2 : ℕ) * cow_consumption * 4 = 2 * initial_grass + 8 * grass_regrowth →
  cows_needed 6 6 = 5 := by sorry

end NUMINAMATH_CALUDE_grass_eating_problem_l2994_299436


namespace NUMINAMATH_CALUDE_inequality_for_negative_numbers_l2994_299454

theorem inequality_for_negative_numbers (a b : ℝ) (h : a < b ∧ b < 0) :
  a^2 > a*b ∧ a*b > b^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_for_negative_numbers_l2994_299454


namespace NUMINAMATH_CALUDE_expression_factorization_l2994_299400

theorem expression_factorization (a : ℝ) : 
  (10 * a^4 - 160 * a^3 - 32) - (-2 * a^4 - 16 * a^3 + 32) = 4 * (3 * a^3 * (a - 12) - 16) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l2994_299400


namespace NUMINAMATH_CALUDE_tea_store_theorem_l2994_299416

/-- Represents the number of ways to buy items from a tea store. -/
def teaStoreCombinations (cups saucers spoons : ℕ) : ℕ :=
  let cupSaucer := cups * saucers
  let cupSpoon := cups * spoons
  let saucerSpoon := saucers * spoons
  let all := cups * saucers * spoons
  cups + saucers + spoons + cupSaucer + cupSpoon + saucerSpoon + all

/-- Theorem stating the total number of combinations for a specific tea store inventory. -/
theorem tea_store_theorem :
  teaStoreCombinations 5 3 4 = 119 := by
  sorry

end NUMINAMATH_CALUDE_tea_store_theorem_l2994_299416


namespace NUMINAMATH_CALUDE_sum_remainder_l2994_299458

theorem sum_remainder (a b c : ℕ) : 
  a < 7 → b < 7 → c < 7 → a > 0 → b > 0 → c > 0 →
  (a * b * c) % 7 = 2 →
  (3 * c) % 7 = 1 →
  (4 * b) % 7 = (2 + b) % 7 →
  (a + b + c) % 7 = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_remainder_l2994_299458


namespace NUMINAMATH_CALUDE_sandy_first_shop_books_l2994_299451

/-- Represents the problem of Sandy's book purchases -/
def SandyBookProblem (first_shop_books : ℕ) : Prop :=
  let total_spent : ℕ := 2160
  let second_shop_books : ℕ := 55
  let average_price : ℕ := 18
  (total_spent : ℚ) / (first_shop_books + second_shop_books : ℚ) = average_price

/-- Proves that Sandy bought 65 books from the first shop -/
theorem sandy_first_shop_books :
  SandyBookProblem 65 := by sorry

end NUMINAMATH_CALUDE_sandy_first_shop_books_l2994_299451


namespace NUMINAMATH_CALUDE_wire_cut_ratio_l2994_299411

theorem wire_cut_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) : 
  (4 * (a / 4) = 6 * (b / 6)) → a / b = 1 := by
sorry

end NUMINAMATH_CALUDE_wire_cut_ratio_l2994_299411


namespace NUMINAMATH_CALUDE_smallest_winning_number_l2994_299445

theorem smallest_winning_number : ∃ N : ℕ, 
  (N = 28) ∧ 
  (0 ≤ N) ∧ (N ≤ 999) ∧
  (36 * N < 2000) ∧
  (72 * N ≥ 2000) ∧
  (∀ M : ℕ, M < N → 
    (M = 0) ∨ (M > 999) ∨ 
    (36 * M ≥ 2000) ∨ 
    (72 * M < 2000)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_winning_number_l2994_299445


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2994_299420

/-- A hyperbola with the given properties has eccentricity √3 -/
theorem hyperbola_eccentricity (a b c : ℝ) (P : ℝ × ℝ) :
  let F₁ : ℝ × ℝ := (-c, 0)
  let F₂ : ℝ × ℝ := (c, 0)
  let e := c / a
  -- Hyperbola equation
  (P.1 / a) ^ 2 - (P.2 / b) ^ 2 = 1 ∧
  -- Line through F₁ at 30° inclination
  (P.2 + c * Real.tan (30 * π / 180)) / (P.1 + c) = Real.tan (30 * π / 180) ∧
  -- Circle with diameter PF₁ passes through F₂
  (P.1 - (-c)) ^ 2 + P.2 ^ 2 = (2 * c) ^ 2 ∧
  -- Standard hyperbola relations
  c ^ 2 = a ^ 2 + b ^ 2 ∧
  P.1 > 0 -- P is on the right branch
  →
  e = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2994_299420


namespace NUMINAMATH_CALUDE_notebook_cost_l2994_299412

/-- The cost of a notebook and a pen given two equations -/
theorem notebook_cost (n p : ℚ) 
  (eq1 : 3 * n + 4 * p = 3.75)
  (eq2 : 5 * n + 2 * p = 3.05) :
  n = 0.3357 := by
  sorry

end NUMINAMATH_CALUDE_notebook_cost_l2994_299412


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2994_299405

theorem complex_equation_solution (z a b : ℂ) : 
  z = (1 - Complex.I)^2 + 1 + 3 * Complex.I →
  z^2 + a * z + b = 1 - Complex.I →
  a.im = 0 →
  b.im = 0 →
  a = -2 ∧ b = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2994_299405


namespace NUMINAMATH_CALUDE_function_range_l2994_299430

theorem function_range (f : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f x = a + b * Real.cos x + c * Real.sin x) →
  f 0 = 1 →
  f (-π/4) = a →
  (∀ x ∈ Set.Icc 0 (π/2), |f x| ≤ Real.sqrt 2) →
  a ∈ Set.Icc 0 (4 + 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_function_range_l2994_299430


namespace NUMINAMATH_CALUDE_five_T_three_l2994_299494

-- Define the operation T
def T (a b : ℤ) : ℤ := 4*a + 7*b + 2*a*b

-- Theorem statement
theorem five_T_three : T 5 3 = 71 := by
  sorry

end NUMINAMATH_CALUDE_five_T_three_l2994_299494


namespace NUMINAMATH_CALUDE_two_tangent_circles_l2994_299413

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are tangent to each other -/
def are_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

/-- Counts the number of circles with radius 4 that are tangent to both given circles -/
def count_tangent_circles (c1 c2 : Circle) : ℕ :=
  sorry

theorem two_tangent_circles 
  (c1 c2 : Circle) 
  (h1 : c1.radius = 2) 
  (h2 : c2.radius = 2) 
  (h3 : are_tangent c1 c2) :
  count_tangent_circles c1 c2 = 2 :=
sorry

end NUMINAMATH_CALUDE_two_tangent_circles_l2994_299413


namespace NUMINAMATH_CALUDE_three_valid_configurations_l2994_299485

/-- Represents a square in the figure -/
structure Square :=
  (id : Nat)

/-- Represents the cross-shaped figure -/
def CrossFigure := List Square

/-- Represents the additional squares -/
def AdditionalSquares := List Square

/-- Represents a configuration after adding a square to the cross figure -/
def Configuration := CrossFigure × Square

/-- Checks if a configuration can be folded into a topless cubical box -/
def canFoldIntoCube (config : Configuration) : Bool :=
  sorry

/-- The main theorem stating that exactly three configurations can be folded into a topless cubical box -/
theorem three_valid_configurations 
  (cross : CrossFigure) 
  (additional : AdditionalSquares) : 
  (cross.length = 5) → 
  (additional.length = 8) → 
  (∃! (n : Nat), n = (List.filter canFoldIntoCube (List.map (λ s => (cross, s)) additional)).length ∧ n = 3) :=
sorry

end NUMINAMATH_CALUDE_three_valid_configurations_l2994_299485


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l2994_299474

/-- Two circles are externally tangent if the distance between their centers
    equals the sum of their radii -/
def externally_tangent (r1 r2 d : ℝ) : Prop := d = r1 + r2

/-- Given two circles with radii 2 and 3, whose centers are 5 units apart,
    prove that they are externally tangent -/
theorem circles_externally_tangent :
  let r1 : ℝ := 2
  let r2 : ℝ := 3
  let d : ℝ := 5
  externally_tangent r1 r2 d := by
sorry

end NUMINAMATH_CALUDE_circles_externally_tangent_l2994_299474


namespace NUMINAMATH_CALUDE_stating_count_valid_starters_l2994_299473

/-- 
Represents the number of boys who can start the game to ensure it goes for at least a full turn 
in a circular arrangement of m boys and n girls.
-/
def valid_starters (m n : ℕ) : ℕ :=
  m - n

/-- 
Theorem stating that the number of valid starters is m - n, 
given that there are more boys than girls.
-/
theorem count_valid_starters (m n : ℕ) (h : m > n) : 
  valid_starters m n = m - n := by
  sorry

end NUMINAMATH_CALUDE_stating_count_valid_starters_l2994_299473


namespace NUMINAMATH_CALUDE_smallest_top_cube_sum_divisible_by_four_l2994_299486

/-- Represents the configuration of the bottom layer of the pyramid -/
structure BottomLayer :=
  (a b c d e f g h i : ℕ)
  (all_different : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧
                   b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧
                   c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧
                   d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧
                   e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧
                   f ≠ g ∧ f ≠ h ∧ f ≠ i ∧
                   g ≠ h ∧ g ≠ i ∧
                   h ≠ i)

/-- Calculates the sum of the top cube given the bottom layer configuration -/
def topCubeSum (bl : BottomLayer) : ℕ :=
  bl.a + bl.c + bl.g + bl.i + 2 * (bl.b + bl.d + bl.f + bl.h) + 4 * bl.e

/-- Theorem stating that the smallest possible sum for the top cube divisible by 4 is 64 -/
theorem smallest_top_cube_sum_divisible_by_four :
  ∀ bl : BottomLayer, ∃ n : ℕ, n ≥ topCubeSum bl ∧ n % 4 = 0 ∧ n ≥ 64 :=
sorry

end NUMINAMATH_CALUDE_smallest_top_cube_sum_divisible_by_four_l2994_299486


namespace NUMINAMATH_CALUDE_rect_to_polar_conversion_l2994_299461

/-- Conversion from rectangular to polar coordinates -/
theorem rect_to_polar_conversion (x y : ℝ) (h : (x, y) = (8, 2 * Real.sqrt 3)) :
  ∃ (r θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧
  r = 2 * Real.sqrt 19 ∧ θ = Real.pi / 6 ∧
  x = r * Real.cos θ ∧ y = r * Real.sin θ :=
by sorry

end NUMINAMATH_CALUDE_rect_to_polar_conversion_l2994_299461


namespace NUMINAMATH_CALUDE_matchstick_rearrangement_l2994_299464

theorem matchstick_rearrangement : |(22 : ℝ) / 7 - Real.pi| < (1 : ℝ) / 10 := by
  sorry

end NUMINAMATH_CALUDE_matchstick_rearrangement_l2994_299464


namespace NUMINAMATH_CALUDE_rectangle_lengths_l2994_299490

/-- Given a square and two rectangles with specific properties, prove their lengths -/
theorem rectangle_lengths (square_side : ℝ) (rect1_width rect2_width : ℝ) 
  (h1 : square_side = 6)
  (h2 : rect1_width = 4)
  (h3 : rect2_width = 3)
  (h4 : square_side * square_side = rect1_width * (square_side * square_side / rect1_width))
  (h5 : rect2_width * (square_side * square_side / (2 * rect2_width)) = square_side * square_side / 2) :
  (square_side * square_side / rect1_width, square_side * square_side / (2 * rect2_width)) = (9, 6) := by
  sorry

#check rectangle_lengths

end NUMINAMATH_CALUDE_rectangle_lengths_l2994_299490


namespace NUMINAMATH_CALUDE_sum_of_fractions_l2994_299450

theorem sum_of_fractions : (1 : ℚ) / 3 + (1 : ℚ) / 4 = 7 / 12 := by sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l2994_299450


namespace NUMINAMATH_CALUDE_number_to_add_for_divisibility_l2994_299447

theorem number_to_add_for_divisibility (n m k : ℕ) (h1 : n = 956734) (h2 : m = 412) (h3 : k = 390) :
  (n + k) % m = 0 := by
  sorry

end NUMINAMATH_CALUDE_number_to_add_for_divisibility_l2994_299447


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l2994_299421

/-- Represents a repeating decimal in the form 0.nnn... where n is a single digit -/
def RepeatingDecimal (n : ℕ) : ℚ := n / 9

theorem repeating_decimal_sum :
  RepeatingDecimal 6 - RepeatingDecimal 4 + RepeatingDecimal 8 = 10 / 9 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l2994_299421


namespace NUMINAMATH_CALUDE_max_radius_circle_through_points_l2994_299460

/-- A circle in a rectangular coordinate system -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if a point lies on a circle -/
def lieOnCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- The maximum possible radius of a circle passing through (16, 0) and (-16, 0) is 16 -/
theorem max_radius_circle_through_points :
  ∃ (c : Circle), lieOnCircle c (16, 0) ∧ lieOnCircle c (-16, 0) →
  ∀ (c' : Circle), lieOnCircle c' (16, 0) ∧ lieOnCircle c' (-16, 0) →
  c'.radius ≤ 16 :=
sorry

end NUMINAMATH_CALUDE_max_radius_circle_through_points_l2994_299460


namespace NUMINAMATH_CALUDE_equation_roots_l2994_299453

theorem equation_roots (k : ℝ) : 
  (∃ x y : ℂ, x ≠ y ∧ 
    (x / (x + 1) + 2 * x / (x + 3) = k * x) ∧ 
    (y / (y + 1) + 2 * y / (y + 3) = k * y) ∧
    (∀ z : ℂ, z / (z + 1) + 2 * z / (z + 3) = k * z → z = x ∨ z = y)) ↔ 
  k = 5/3 :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_l2994_299453


namespace NUMINAMATH_CALUDE_range_of_z_l2994_299414

theorem range_of_z (α β z : ℝ) 
  (h1 : -2 < α ∧ α ≤ 3) 
  (h2 : 2 < β ∧ β ≤ 4) 
  (h3 : z = 2*α - (1/2)*β) : 
  -6 < z ∧ z < 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_z_l2994_299414


namespace NUMINAMATH_CALUDE_parabola_vertex_l2994_299495

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := x^2 - 6*x + 5

/-- The vertex coordinates of the parabola -/
def vertex : ℝ × ℝ := (3, -4)

/-- Theorem: The vertex coordinates of the parabola y = x^2 - 6x + 5 are (3, -4) -/
theorem parabola_vertex : 
  ∀ x : ℝ, parabola x ≥ parabola (vertex.1) ∧ parabola (vertex.1) = vertex.2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2994_299495


namespace NUMINAMATH_CALUDE_cubic_integer_root_l2994_299435

theorem cubic_integer_root (p q : ℤ) : 
  (∃ (x : ℝ), x^3 - p*x - q = 0 ∧ x = 4 - Real.sqrt 10) →
  ((-8 : ℝ)^3 - p*(-8) - q = 0) :=
sorry

end NUMINAMATH_CALUDE_cubic_integer_root_l2994_299435


namespace NUMINAMATH_CALUDE_car_price_calculation_l2994_299456

theorem car_price_calculation (reduced_price : ℝ) (discount_percentage : ℝ) (original_price : ℝ) : 
  reduced_price = 7500 ∧ 
  discount_percentage = 25 ∧ 
  reduced_price = original_price * (1 - discount_percentage / 100) → 
  original_price = 10000 := by
sorry

end NUMINAMATH_CALUDE_car_price_calculation_l2994_299456


namespace NUMINAMATH_CALUDE_sin_cos_identity_l2994_299401

theorem sin_cos_identity :
  Real.sin (68 * π / 180) * Real.sin (67 * π / 180) - 
  Real.sin (23 * π / 180) * Real.cos (68 * π / 180) = 
  Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l2994_299401


namespace NUMINAMATH_CALUDE_billboard_fully_lit_probability_l2994_299489

/-- The number of words in the billboard text -/
def num_words : ℕ := 5

/-- The probability of seeing the billboard fully lit -/
def fully_lit_probability : ℚ := 1 / num_words

/-- Theorem stating that the probability of seeing the billboard fully lit is 1/5 -/
theorem billboard_fully_lit_probability :
  fully_lit_probability = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_billboard_fully_lit_probability_l2994_299489


namespace NUMINAMATH_CALUDE_binomial_square_constant_l2994_299455

theorem binomial_square_constant (c : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, 9 * x^2 - 24 * x + c = (a * x + b)^2) → c = 16 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_constant_l2994_299455


namespace NUMINAMATH_CALUDE_sandy_has_24_red_balloons_l2994_299429

/-- The number of red balloons Sandy has -/
def sandys_red_balloons (saras_red_balloons total_red_balloons : ℕ) : ℕ :=
  total_red_balloons - saras_red_balloons

/-- Theorem stating that Sandy has 24 red balloons -/
theorem sandy_has_24_red_balloons :
  sandys_red_balloons 31 55 = 24 := by
  sorry

end NUMINAMATH_CALUDE_sandy_has_24_red_balloons_l2994_299429


namespace NUMINAMATH_CALUDE_marys_oranges_l2994_299441

theorem marys_oranges :
  ∀ (oranges : ℕ),
    (14 + oranges + 6 - 3 = 26) →
    oranges = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_marys_oranges_l2994_299441


namespace NUMINAMATH_CALUDE_train_speed_l2994_299437

/-- The speed of a train given its length, time to pass a person, and the person's speed -/
theorem train_speed (train_length : ℝ) (passing_time : ℝ) (person_speed : ℝ) :
  train_length = 125 →
  passing_time = 6 →
  person_speed = 5 →
  ∃ (train_speed : ℝ), (abs (train_speed - 70) < 0.5 ∧
    train_speed * 1000 / 3600 + person_speed * 1000 / 3600 = train_length / passing_time) :=
by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2994_299437


namespace NUMINAMATH_CALUDE_triangle_side_difference_l2994_299481

-- Define the triangle sides
def side1 : ℝ := 7
def side2 : ℝ := 10

-- Define the valid range for x
def valid_x (x : ℤ) : Prop :=
  x > 0 ∧ x + side1 > side2 ∧ x + side2 > side1 ∧ side1 + side2 > x

-- Theorem statement
theorem triangle_side_difference :
  (∃ (max min : ℤ), 
    (∀ x : ℤ, valid_x x → x ≤ max) ∧
    (∀ x : ℤ, valid_x x → x ≥ min) ∧
    (∀ x : ℤ, valid_x x → min ≤ x ∧ x ≤ max) ∧
    (max - min = 12)) :=
sorry

end NUMINAMATH_CALUDE_triangle_side_difference_l2994_299481


namespace NUMINAMATH_CALUDE_beau_age_proof_l2994_299480

theorem beau_age_proof (sons_age_today : ℕ) (sons_are_triplets : Bool) : 
  sons_age_today = 16 ∧ sons_are_triplets = true → 42 = (sons_age_today - 3) * 3 + 3 :=
by
  sorry

end NUMINAMATH_CALUDE_beau_age_proof_l2994_299480


namespace NUMINAMATH_CALUDE_bucket_weight_l2994_299423

theorem bucket_weight (p q : ℝ) : ℝ :=
  let one_quarter_full := p
  let three_quarters_full := q
  let full_weight := -1/2 * p + 3/2 * q
  full_weight

#check bucket_weight

end NUMINAMATH_CALUDE_bucket_weight_l2994_299423


namespace NUMINAMATH_CALUDE_problem_solution_l2994_299404

theorem problem_solution : (2023^2 - 2023 - 4^2) / 2023 = 2022 - 16/2023 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2994_299404


namespace NUMINAMATH_CALUDE_sues_trail_mix_composition_sues_dried_fruit_percentage_proof_l2994_299428

/-- The percentage of dried fruit in Sue's trail mix -/
def sues_dried_fruit_percentage : ℝ := 70

theorem sues_trail_mix_composition :
  sues_dried_fruit_percentage = 70 :=
by
  -- Proof goes here
  sorry

/-- Sue's trail mix nuts percentage -/
def sues_nuts_percentage : ℝ := 30

/-- Jane's trail mix nuts percentage -/
def janes_nuts_percentage : ℝ := 60

/-- Jane's trail mix chocolate chips percentage -/
def janes_chocolate_percentage : ℝ := 40

/-- Combined mixture nuts percentage -/
def combined_nuts_percentage : ℝ := 45

/-- Combined mixture dried fruit percentage -/
def combined_dried_fruit_percentage : ℝ := 35

/-- Sue's trail mix consists of only nuts and dried fruit -/
axiom sues_mix_composition :
  sues_nuts_percentage + sues_dried_fruit_percentage = 100

/-- The combined mixture percentages are consistent with individual mixes -/
axiom combined_mix_consistency (s j : ℝ) :
  s > 0 ∧ j > 0 →
  sues_nuts_percentage * s + janes_nuts_percentage * j = combined_nuts_percentage * (s + j) ∧
  sues_dried_fruit_percentage * s = combined_dried_fruit_percentage * (s + j)

theorem sues_dried_fruit_percentage_proof :
  sues_dried_fruit_percentage = 70 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_sues_trail_mix_composition_sues_dried_fruit_percentage_proof_l2994_299428


namespace NUMINAMATH_CALUDE_expression_value_at_negative_two_l2994_299478

theorem expression_value_at_negative_two :
  let x : ℤ := -2
  (3 * x + 4)^2 - 2 * x = 8 := by sorry

end NUMINAMATH_CALUDE_expression_value_at_negative_two_l2994_299478


namespace NUMINAMATH_CALUDE_ordering_abc_l2994_299483

theorem ordering_abc :
  let a : ℝ := Real.log 2
  let b : ℝ := 2023 / 2022
  let c : ℝ := Real.log 2023 / Real.log 2022
  a < c ∧ c < b := by sorry

end NUMINAMATH_CALUDE_ordering_abc_l2994_299483


namespace NUMINAMATH_CALUDE_tree_age_at_height_l2994_299439

/-- Represents the growth of a tree over time. -/
def tree_growth (initial_height : ℝ) (growth_rate : ℝ) (initial_age : ℝ) (years : ℝ) : ℝ :=
  initial_height + growth_rate * years

/-- Theorem stating the age of the tree when it reaches a specific height. -/
theorem tree_age_at_height (initial_height : ℝ) (growth_rate : ℝ) (initial_age : ℝ) (final_height : ℝ) :
  initial_height = 5 →
  growth_rate = 3 →
  initial_age = 1 →
  final_height = 23 →
  ∃ (years : ℝ), tree_growth initial_height growth_rate initial_age years = final_height ∧ initial_age + years = 7 :=
by
  sorry


end NUMINAMATH_CALUDE_tree_age_at_height_l2994_299439


namespace NUMINAMATH_CALUDE_parallelogram_vertex_sum_l2994_299459

/-- A parallelogram with vertices A, B, C, and D in 2D space -/
structure Parallelogram :=
  (A B C D : ℝ × ℝ)

/-- The property that the diagonals of a parallelogram bisect each other -/
def diagonals_bisect (p : Parallelogram) : Prop :=
  let midpoint_AD := ((p.A.1 + p.D.1) / 2, (p.A.2 + p.D.2) / 2)
  let midpoint_BC := ((p.B.1 + p.C.1) / 2, (p.B.2 + p.C.2) / 2)
  midpoint_AD = midpoint_BC

/-- The sum of coordinates of a point -/
def sum_coordinates (p : ℝ × ℝ) : ℝ :=
  p.1 + p.2

/-- The main theorem -/
theorem parallelogram_vertex_sum :
  ∀ (p : Parallelogram),
    p.A = (2, 3) →
    p.B = (5, 7) →
    p.D = (11, -1) →
    diagonals_bisect p →
    sum_coordinates p.C = 3 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_vertex_sum_l2994_299459


namespace NUMINAMATH_CALUDE_restaurant_location_l2994_299479

theorem restaurant_location (A B C : ℝ × ℝ) : 
  let road_y : ℝ := 0
  let A_x : ℝ := 0
  let A_y : ℝ := 300
  let B_y : ℝ := road_y
  let dist_AB : ℝ := 500
  A = (A_x, A_y) →
  B.2 = road_y →
  Real.sqrt ((B.1 - A_x)^2 + (B.2 - A_y)^2) = dist_AB →
  C.2 = road_y →
  Real.sqrt ((C.1 - A_x)^2 + (C.2 - A_y)^2) = Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) →
  C.1 = 200 := by
sorry

end NUMINAMATH_CALUDE_restaurant_location_l2994_299479


namespace NUMINAMATH_CALUDE_origin_outside_circle_l2994_299491

theorem origin_outside_circle (a : ℝ) (h : 0 < a ∧ a < 1) : 
  let circle_equation (x y : ℝ) := x^2 + y^2 + 2*a*x + 2*y + (a - 1)^2
  circle_equation 0 0 > 0 := by
  sorry

end NUMINAMATH_CALUDE_origin_outside_circle_l2994_299491


namespace NUMINAMATH_CALUDE_yearly_increase_fraction_l2994_299415

/-- 
Given an initial amount that increases by a fraction each year, 
this theorem proves that the fraction is 0.125 when the initial amount 
is 3200 and becomes 4050 after two years.
-/
theorem yearly_increase_fraction 
  (initial_amount : ℝ) 
  (final_amount : ℝ) 
  (f : ℝ) 
  (h1 : initial_amount = 3200) 
  (h2 : final_amount = 4050) 
  (h3 : final_amount = initial_amount * (1 + f)^2) : 
  f = 0.125 := by
sorry

end NUMINAMATH_CALUDE_yearly_increase_fraction_l2994_299415


namespace NUMINAMATH_CALUDE_pair_farm_animals_l2994_299440

/-- Represents the number of ways to pair animals of different species -/
def pairAnimals (cows pigs horses : ℕ) : ℕ :=
  let cowPigPairs := cows * pigs
  let remainingPairs := Nat.factorial horses
  cowPigPairs * remainingPairs

/-- Theorem stating the number of ways to pair 5 cows, 4 pigs, and 7 horses -/
theorem pair_farm_animals :
  pairAnimals 5 4 7 = 100800 := by
  sorry

#eval pairAnimals 5 4 7

end NUMINAMATH_CALUDE_pair_farm_animals_l2994_299440


namespace NUMINAMATH_CALUDE_sum_of_squares_l2994_299425

theorem sum_of_squares (a b c : ℝ) : 
  a * b + b * c + a * c = 131 →
  a + b + c = 22 →
  a^2 + b^2 + c^2 = 222 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l2994_299425


namespace NUMINAMATH_CALUDE_fraction_equality_l2994_299432

theorem fraction_equality (a b c d e : ℚ) 
  (h1 : a / b = 1 / 2)
  (h2 : c / d = 1 / 2)
  (h3 : e / 5 = 1 / 2)
  (h4 : b + d + 5 ≠ 0) :
  (a + c + e) / (b + d + 5) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l2994_299432


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2994_299419

theorem imaginary_part_of_z : Complex.im ((-3 + Complex.I) / Complex.I^3) = -3 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2994_299419


namespace NUMINAMATH_CALUDE_davids_daughter_age_l2994_299434

/-- David's current age -/
def david_age : ℕ := 40

/-- Number of years in the future when David's age will be twice his daughter's -/
def years_until_double : ℕ := 16

/-- David's daughter's current age -/
def daughter_age : ℕ := 12

/-- Theorem stating that David's daughter is 12 years old today -/
theorem davids_daughter_age :
  daughter_age = 12 ∧
  david_age + years_until_double = 2 * (daughter_age + years_until_double) :=
by sorry

end NUMINAMATH_CALUDE_davids_daughter_age_l2994_299434


namespace NUMINAMATH_CALUDE_lines_dont_intersect_if_points_not_coplanar_l2994_299465

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space defined by two points -/
structure Line3D where
  p1 : Point3D
  p2 : Point3D

/-- Check if four points are coplanar -/
def are_coplanar (a b c d : Point3D) : Prop := sorry

/-- Check if two lines intersect -/
def lines_intersect (l1 l2 : Line3D) : Prop := sorry

theorem lines_dont_intersect_if_points_not_coplanar 
  (a b c d : Point3D) 
  (h : ¬ are_coplanar a b c d) : 
  ¬ lines_intersect (Line3D.mk a b) (Line3D.mk c d) := by
  sorry

end NUMINAMATH_CALUDE_lines_dont_intersect_if_points_not_coplanar_l2994_299465


namespace NUMINAMATH_CALUDE_longest_segment_in_cylinder_l2994_299469

-- Define the cylinder
def cylinder_radius : ℝ := 5
def cylinder_height : ℝ := 10

-- Theorem statement
theorem longest_segment_in_cylinder :
  let diagonal := Real.sqrt (cylinder_height ^ 2 + (2 * cylinder_radius) ^ 2)
  diagonal = 10 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_longest_segment_in_cylinder_l2994_299469


namespace NUMINAMATH_CALUDE_no_triple_squares_l2994_299462

theorem no_triple_squares : ¬∃ (m n k : ℕ), 
  (∃ a : ℕ, m^2 + n + k = a^2) ∧ 
  (∃ b : ℕ, n^2 + k + m = b^2) ∧ 
  (∃ c : ℕ, k^2 + m + n = c^2) := by
sorry

end NUMINAMATH_CALUDE_no_triple_squares_l2994_299462


namespace NUMINAMATH_CALUDE_x_squared_minus_nine_y_squared_l2994_299448

theorem x_squared_minus_nine_y_squared (x y : ℝ) 
  (eq1 : x + 3*y = -1) 
  (eq2 : x - 3*y = 5) : 
  x^2 - 9*y^2 = -5 := by
sorry

end NUMINAMATH_CALUDE_x_squared_minus_nine_y_squared_l2994_299448


namespace NUMINAMATH_CALUDE_x1_value_l2994_299463

theorem x1_value (x1 x2 x3 x4 : ℝ) 
  (h_order : 0 ≤ x4 ∧ x4 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1) 
  (h_sum : (1-x1)^2 + (x1-x2)^2 + (x2-x3)^2 + (x3-x4)^2 + x4^2 = 9/16) : 
  x1 = 1 - 15 / Real.sqrt 80 := by
  sorry

end NUMINAMATH_CALUDE_x1_value_l2994_299463


namespace NUMINAMATH_CALUDE_range_of_m_symmetrical_circle_equation_existence_of_m_for_circle_through_origin_l2994_299433

-- Define the circle C and line l
def circle_C (x y m : ℝ) : Prop := x^2 + y^2 + x - 6*y + m = 0
def line_l (x y : ℝ) : Prop := x + y - 3 = 0

-- Theorem 1: Range of m
theorem range_of_m :
  ∀ m : ℝ, (∃ x y : ℝ, circle_C x y m) → m < 37/4 :=
sorry

-- Theorem 2: Equation of symmetrical circle
theorem symmetrical_circle_equation :
  ∀ m : ℝ, (∃ x y : ℝ, circle_C x y m ∧ line_l x y) →
  (∀ x y : ℝ, x^2 + (y - 7/2)^2 = 1/8) :=
sorry

-- Theorem 3: Existence of m for circle through origin
theorem existence_of_m_for_circle_through_origin :
  ∃ m : ℝ, m = -3/2 ∧
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    circle_C x₁ y₁ m ∧ circle_C x₂ y₂ m ∧
    line_l x₁ y₁ ∧ line_l x₂ y₂ ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧
    (∃ a b r : ℝ, (x₁ - a)^2 + (y₁ - b)^2 = r^2 ∧
                  (x₂ - a)^2 + (y₂ - b)^2 = r^2 ∧
                  a^2 + b^2 = r^2)) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_symmetrical_circle_equation_existence_of_m_for_circle_through_origin_l2994_299433


namespace NUMINAMATH_CALUDE_equilateral_hyperbola_equation_l2994_299484

-- Define the hyperbola
def Hyperbola (x y : ℝ) := x^2 - y^2 = 8

-- Define the line containing one focus
def FocusLine (x y : ℝ) := 3*x - 4*y + 12 = 0

-- Theorem statement
theorem equilateral_hyperbola_equation :
  ∃ (a b : ℝ), 
    -- One focus is on the line
    FocusLine a b ∧
    -- The focus is on the x-axis (real axis)
    b = 0 ∧
    -- The hyperbola passes through this focus
    Hyperbola a b ∧
    -- The hyperbola is equilateral (a² = b²)
    a^2 = (8:ℝ) := by
  sorry

end NUMINAMATH_CALUDE_equilateral_hyperbola_equation_l2994_299484


namespace NUMINAMATH_CALUDE_vector_equation_solution_l2994_299477

def vector_sum (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 + b.1, a.2 + b.2)

def scalar_mult (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)

theorem vector_equation_solution (x y : ℝ) 
  (h : vector_sum (x, 1) (scalar_mult 2 (2, y)) = (5, -3)) : 
  x + y = -1 := by
  sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l2994_299477


namespace NUMINAMATH_CALUDE_probability_theorem_l2994_299470

structure ProfessionalGroup where
  women_percentage : ℝ
  men_percentage : ℝ
  nonbinary_percentage : ℝ
  women_engineer_percentage : ℝ
  women_doctor_percentage : ℝ
  men_engineer_percentage : ℝ
  men_doctor_percentage : ℝ
  nonbinary_engineer_percentage : ℝ
  nonbinary_translator_percentage : ℝ

def probability_selection (group : ProfessionalGroup) : ℝ :=
  group.women_percentage * group.women_engineer_percentage +
  group.men_percentage * group.men_doctor_percentage +
  group.nonbinary_percentage * group.nonbinary_translator_percentage

theorem probability_theorem (group : ProfessionalGroup) 
  (h1 : group.women_percentage = 0.70)
  (h2 : group.men_percentage = 0.20)
  (h3 : group.nonbinary_percentage = 0.10)
  (h4 : group.women_engineer_percentage = 0.20)
  (h5 : group.men_doctor_percentage = 0.20)
  (h6 : group.nonbinary_translator_percentage = 0.20) :
  probability_selection group = 0.20 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l2994_299470


namespace NUMINAMATH_CALUDE_polygon_sides_l2994_299410

theorem polygon_sides (n : ℕ) : 
  (n ≥ 3) →
  ((n - 2) * 180 + 360 = 1260) →
  n = 7 :=
by sorry

end NUMINAMATH_CALUDE_polygon_sides_l2994_299410


namespace NUMINAMATH_CALUDE_exists_multiple_indecomposable_factorizations_l2994_299431

/-- The set V_n for a given positive integer n -/
def V_n (n : ℕ) : Set ℕ := {m : ℕ | ∃ k : ℕ+, m = 1 + k * n}

/-- A number is indecomposable in V_n if it cannot be expressed as a product of two numbers in V_n -/
def Indecomposable (n : ℕ) (m : ℕ) : Prop :=
  m ∈ V_n n ∧ ∀ p q : ℕ, p ∈ V_n n → q ∈ V_n n → p * q ≠ m

/-- Main theorem: There exists a number in V_n that can be expressed as a product of
    indecomposable numbers in V_n in more than one way -/
theorem exists_multiple_indecomposable_factorizations (n : ℕ) (h : n > 2) :
  ∃ r : ℕ, r ∈ V_n n ∧
    ∃ (a b c d : ℕ) (ha : Indecomposable n a) (hb : Indecomposable n b)
      (hc : Indecomposable n c) (hd : Indecomposable n d),
    r = a * b ∧ r = c * d ∧ (a ≠ c ∨ b ≠ d) :=
  sorry

end NUMINAMATH_CALUDE_exists_multiple_indecomposable_factorizations_l2994_299431


namespace NUMINAMATH_CALUDE_cubic_linear_inequality_l2994_299417

theorem cubic_linear_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  a^3 + b^3 + a + b ≥ 4 * a * b := by
  sorry

end NUMINAMATH_CALUDE_cubic_linear_inequality_l2994_299417


namespace NUMINAMATH_CALUDE_data_center_connections_l2994_299446

theorem data_center_connections (n : ℕ) (k : ℕ) (h1 : n = 30) (h2 : k = 4) :
  (n * k) / 2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_data_center_connections_l2994_299446


namespace NUMINAMATH_CALUDE_triangle_area_from_altitudes_l2994_299499

/-- Given a triangle ABC with altitudes h_a, h_b, and h_c, 
    its area S is equal to 
    1 / sqrt((1/h_a + 1/h_b + 1/h_c) * (1/h_a + 1/h_b - 1/h_c) * 
             (1/h_a + 1/h_c - 1/h_b) * (1/h_b + 1/h_c - 1/h_a)) -/
theorem triangle_area_from_altitudes (h_a h_b h_c : ℝ) (h_pos_a : h_a > 0) (h_pos_b : h_b > 0) (h_pos_c : h_c > 0) :
  let S := 1 / Real.sqrt ((1/h_a + 1/h_b + 1/h_c) * (1/h_a + 1/h_b - 1/h_c) * 
                          (1/h_a + 1/h_c - 1/h_b) * (1/h_b + 1/h_c - 1/h_a))
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    S = (a * h_a) / 2 ∧ S = (b * h_b) / 2 ∧ S = (c * h_c) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_from_altitudes_l2994_299499


namespace NUMINAMATH_CALUDE_prob_red_pen_is_two_fifths_l2994_299476

/-- The number of colored pens -/
def total_pens : ℕ := 5

/-- The number of pens to be selected -/
def selected_pens : ℕ := 2

/-- The number of ways to select 2 pens out of 5 -/
def total_selections : ℕ := Nat.choose total_pens selected_pens

/-- The number of ways to select a red pen and another different color -/
def red_selections : ℕ := total_pens - 1

/-- The probability of selecting a red pen when choosing 2 different colored pens out of 5 -/
def prob_red_pen : ℚ := red_selections / total_selections

theorem prob_red_pen_is_two_fifths : prob_red_pen = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_prob_red_pen_is_two_fifths_l2994_299476


namespace NUMINAMATH_CALUDE_two_numbers_problem_l2994_299497

theorem two_numbers_problem : ∃ (x y : ℕ), 
  (x + y = 1244) ∧ 
  (10 * x + 3 = (y - 2) / 10) ∧
  (x = 12) ∧ 
  (y = 1232) := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_problem_l2994_299497


namespace NUMINAMATH_CALUDE_travel_ways_theorem_l2994_299487

/-- The number of different ways to travel between two places given the number of bus, train, and ferry routes -/
def total_travel_ways (buses trains ferries : ℕ) : ℕ :=
  buses + trains + ferries

/-- Theorem stating that with 5 buses, 6 trains, and 2 ferries, there are 13 ways to travel -/
theorem travel_ways_theorem :
  total_travel_ways 5 6 2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_travel_ways_theorem_l2994_299487


namespace NUMINAMATH_CALUDE_square_area_equal_perimeter_l2994_299498

/-- The area of a square with perimeter equal to a triangle with sides 6.1, 8.2, and 9.7 -/
theorem square_area_equal_perimeter (s : Real) (h1 : s > 0) 
  (h2 : 4 * s = 6.1 + 8.2 + 9.7) : s^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_area_equal_perimeter_l2994_299498


namespace NUMINAMATH_CALUDE_hula_hoop_radius_l2994_299493

theorem hula_hoop_radius (diameter : ℝ) (h : diameter = 14) : diameter / 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_hula_hoop_radius_l2994_299493


namespace NUMINAMATH_CALUDE_angle_C_measure_l2994_299418

-- Define the triangle and its angles
structure Triangle :=
  (A B C : ℝ)

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  t.B = t.A + 20 ∧ t.C = t.A + 40 ∧ t.A + t.B + t.C = 180

-- Theorem statement
theorem angle_C_measure (t : Triangle) :
  satisfies_conditions t → t.C = 80 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_measure_l2994_299418


namespace NUMINAMATH_CALUDE_completing_square_quadratic_l2994_299488

theorem completing_square_quadratic (x : ℝ) :
  (x^2 - 4*x - 2 = 0) ↔ ((x - 2)^2 = 6) :=
sorry

end NUMINAMATH_CALUDE_completing_square_quadratic_l2994_299488


namespace NUMINAMATH_CALUDE_circle_center_is_3_0_l2994_299406

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of a circle in standard form -/
def circle_equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

/-- The given circle equation -/
def given_circle_equation (x y : ℝ) : Prop :=
  (x - 3)^2 + y^2 = 9

theorem circle_center_is_3_0 :
  ∃ (c : Circle), (∀ x y : ℝ, circle_equation c x y ↔ given_circle_equation x y) ∧ c.center = (3, 0) := by
  sorry

end NUMINAMATH_CALUDE_circle_center_is_3_0_l2994_299406


namespace NUMINAMATH_CALUDE_apple_juice_production_l2994_299467

/-- Calculates the amount of apples used for apple juice production in million tons -/
def applesForJuice (totalApples : ℝ) (ciderPercent : ℝ) (freshPercent : ℝ) (juicePercent : ℝ) : ℝ :=
  let ciderApples := ciderPercent * totalApples
  let remainingApples := totalApples - ciderApples
  let freshApples := freshPercent * remainingApples
  let exportedApples := remainingApples - freshApples
  juicePercent * exportedApples

theorem apple_juice_production :
  applesForJuice 6 0.3 0.4 0.6 = 1.512 := by
  sorry

end NUMINAMATH_CALUDE_apple_juice_production_l2994_299467


namespace NUMINAMATH_CALUDE_heat_bulls_difference_l2994_299442

/-- The number of games won by the Chicago Bulls -/
def bulls_games : ℕ := 70

/-- The total number of games won by both the Chicago Bulls and the Miami Heat -/
def total_games : ℕ := 145

/-- The number of games won by the Miami Heat -/
def heat_games : ℕ := total_games - bulls_games

/-- Theorem stating the difference in games won between the Miami Heat and the Chicago Bulls -/
theorem heat_bulls_difference : heat_games - bulls_games = 5 := by
  sorry

end NUMINAMATH_CALUDE_heat_bulls_difference_l2994_299442


namespace NUMINAMATH_CALUDE_vertex_of_quadratic_l2994_299422

-- Define the quadratic function
def f (x : ℝ) : ℝ := -2 * (x - 1)^2 - 3

-- Theorem stating that the vertex of the quadratic function is at (1, -3)
theorem vertex_of_quadratic :
  ∃ (x y : ℝ), x = 1 ∧ y = -3 ∧ ∀ (t : ℝ), f t ≤ f x :=
sorry

end NUMINAMATH_CALUDE_vertex_of_quadratic_l2994_299422


namespace NUMINAMATH_CALUDE_three_lines_intersection_l2994_299492

/-- Three lines intersect at a single point if and only if k = -2/7 --/
theorem three_lines_intersection (x y k : ℚ) : 
  (y = 3*x + 2 ∧ y = -4*x - 14 ∧ y = 2*x + k) ↔ k = -2/7 := by
  sorry

end NUMINAMATH_CALUDE_three_lines_intersection_l2994_299492


namespace NUMINAMATH_CALUDE_delta_u_zero_l2994_299424

def u (n : ℕ) : ℤ := n^3 - n

def delta (k : ℕ) : (ℕ → ℤ) → (ℕ → ℤ) :=
  match k with
  | 0 => id
  | k+1 => fun f n => f (n+1) - f n

theorem delta_u_zero (k : ℕ) :
  (∀ n, delta k u n = 0) ↔ k ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_delta_u_zero_l2994_299424


namespace NUMINAMATH_CALUDE_ellipse_angle_bisector_l2994_299444

/-- Definition of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 8 + y^2 / 4 = 1

/-- Definition of a point being on a chord through F -/
def is_on_chord_through_F (x y : ℝ) : Prop := 
  ∃ (m : ℝ), y = m * (x - 2)

/-- Definition of the angle equality condition -/
def angle_equality (p : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (y₁ / (x₁ - p)) = -(y₂ / (x₂ - p))

/-- The main theorem -/
theorem ellipse_angle_bisector :
  ∃! (p : ℝ), p > 0 ∧ 
  (∀ x₁ y₁ x₂ y₂ : ℝ, 
    is_on_ellipse x₁ y₁ ∧ is_on_ellipse x₂ y₂ ∧
    is_on_chord_through_F x₁ y₁ ∧ is_on_chord_through_F x₂ y₂ ∧
    x₁ ≠ x₂ →
    angle_equality p x₁ y₁ x₂ y₂) ∧
  p = 2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_angle_bisector_l2994_299444


namespace NUMINAMATH_CALUDE_reflection_theorem_l2994_299475

/-- A reflection in 2D space -/
structure Reflection2D where
  /-- The function that performs the reflection -/
  apply : ℝ × ℝ → ℝ × ℝ

/-- Theorem: Given a reflection that maps (3, -2) to (7, 6), it will map (0, 4) to (80/29, -84/29) -/
theorem reflection_theorem (r : Reflection2D) 
  (h1 : r.apply (3, -2) = (7, 6)) :
  r.apply (0, 4) = (80/29, -84/29) := by
sorry


end NUMINAMATH_CALUDE_reflection_theorem_l2994_299475


namespace NUMINAMATH_CALUDE_at_least_one_red_not_basic_event_l2994_299427

structure Ball := (color : String)

def bag : Multiset Ball := 
  2 • {Ball.mk "red"} + 2 • {Ball.mk "white"} + 2 • {Ball.mk "black"}

def is_basic_event (event : Set (Ball × Ball)) : Prop :=
  ∃ (b1 b2 : Ball), event = {(b1, b2)}

def at_least_one_red (pair : Ball × Ball) : Prop :=
  (pair.1.color = "red") ∨ (pair.2.color = "red")

theorem at_least_one_red_not_basic_event :
  ¬ (is_basic_event {pair | at_least_one_red pair}) :=
sorry

end NUMINAMATH_CALUDE_at_least_one_red_not_basic_event_l2994_299427


namespace NUMINAMATH_CALUDE_haleigh_cats_count_l2994_299471

/-- The number of cats Haleigh has -/
def num_cats : ℕ := 10

/-- The number of dogs Haleigh has -/
def num_dogs : ℕ := 4

/-- The total number of leggings needed -/
def total_leggings : ℕ := 14

/-- Each animal needs one pair of leggings -/
def leggings_per_animal : ℕ := 1

theorem haleigh_cats_count :
  num_cats = total_leggings - num_dogs * leggings_per_animal :=
by sorry

end NUMINAMATH_CALUDE_haleigh_cats_count_l2994_299471


namespace NUMINAMATH_CALUDE_greatest_integer_b_for_all_real_domain_l2994_299472

theorem greatest_integer_b_for_all_real_domain : ∃ (b : ℤ), 
  (∀ (x : ℝ), x^2 + (b : ℝ) * x + 7 ≠ 0) ∧ 
  (∀ (b' : ℤ), (∀ (x : ℝ), x^2 + (b' : ℝ) * x + 7 ≠ 0) → b' ≤ b) ∧
  b = 5 := by sorry

end NUMINAMATH_CALUDE_greatest_integer_b_for_all_real_domain_l2994_299472


namespace NUMINAMATH_CALUDE_square_tiles_count_l2994_299468

/-- Represents a box of pentagonal and square tiles -/
structure TileBox where
  pentagonal : ℕ
  square : ℕ

/-- The total number of tiles in the box -/
def TileBox.total (box : TileBox) : ℕ := box.pentagonal + box.square

/-- The total number of edges in the box -/
def TileBox.edges (box : TileBox) : ℕ := 5 * box.pentagonal + 4 * box.square

theorem square_tiles_count (box : TileBox) : 
  box.total = 30 ∧ box.edges = 122 → box.square = 28 := by
  sorry

end NUMINAMATH_CALUDE_square_tiles_count_l2994_299468


namespace NUMINAMATH_CALUDE_range_of_m_l2994_299407

noncomputable def f (x : ℝ) : ℝ := if x ≥ 0 then x^2 else -x^2

theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Iic 1, f (x + m) ≤ -f x) → m ∈ Set.Ici (-2) := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l2994_299407


namespace NUMINAMATH_CALUDE_kat_strength_training_frequency_l2994_299409

/-- Kat's weekly training schedule -/
structure TrainingSchedule where
  strength_duration : ℝ  -- Duration of each strength training session in hours
  strength_frequency : ℝ  -- Number of strength training sessions per week
  boxing_duration : ℝ     -- Duration of each boxing session in hours
  boxing_frequency : ℝ    -- Number of boxing sessions per week
  total_hours : ℝ         -- Total training hours per week

/-- Theorem stating that Kat does strength training 3 times a week -/
theorem kat_strength_training_frequency 
  (schedule : TrainingSchedule) 
  (h1 : schedule.strength_duration = 1)
  (h2 : schedule.boxing_duration = 1.5)
  (h3 : schedule.boxing_frequency = 4)
  (h4 : schedule.total_hours = 9)
  (h5 : schedule.total_hours = schedule.strength_duration * schedule.strength_frequency + 
                               schedule.boxing_duration * schedule.boxing_frequency) :
  schedule.strength_frequency = 3 := by
  sorry

#check kat_strength_training_frequency

end NUMINAMATH_CALUDE_kat_strength_training_frequency_l2994_299409


namespace NUMINAMATH_CALUDE_kids_from_unnamed_school_l2994_299496

theorem kids_from_unnamed_school (riverside_total : ℕ) (mountaintop_total : ℕ) (total_admitted : ℕ)
  (riverside_denied_percent : ℚ) (mountaintop_denied_percent : ℚ) (unnamed_denied_percent : ℚ)
  (h1 : riverside_total = 120)
  (h2 : mountaintop_total = 50)
  (h3 : total_admitted = 148)
  (h4 : riverside_denied_percent = 1/5)
  (h5 : mountaintop_denied_percent = 1/2)
  (h6 : unnamed_denied_percent = 7/10) :
  ∃ (unnamed_total : ℕ),
    unnamed_total = 90 ∧
    total_admitted = 
      (riverside_total - riverside_total * riverside_denied_percent) +
      (mountaintop_total - mountaintop_total * mountaintop_denied_percent) +
      (unnamed_total - unnamed_total * unnamed_denied_percent) :=
by sorry

end NUMINAMATH_CALUDE_kids_from_unnamed_school_l2994_299496
