import Mathlib

namespace NUMINAMATH_CALUDE_not_all_sqrt5_periodic_all_sqrt3_periodic_l2825_282512

-- Define the function types
def RealFunction := ℝ → ℝ

-- Define the functional equations
def SatisfiesSqrt5Equation (f : RealFunction) : Prop :=
  ∀ x : ℝ, f (x - 1) + f (x + 1) = Real.sqrt 5 * f x

def SatisfiesSqrt3Equation (g : RealFunction) : Prop :=
  ∀ x : ℝ, g (x - 1) + g (x + 1) = Real.sqrt 3 * g x

-- Define periodicity
def IsPeriodic (f : RealFunction) : Prop :=
  ∃ T : ℝ, T ≠ 0 ∧ ∀ x : ℝ, f (x + T) = f x

-- Theorem statements
theorem not_all_sqrt5_periodic :
  ∃ f : RealFunction, SatisfiesSqrt5Equation f ∧ ¬IsPeriodic f :=
sorry

theorem all_sqrt3_periodic :
  ∀ g : RealFunction, SatisfiesSqrt3Equation g → IsPeriodic g :=
sorry

end NUMINAMATH_CALUDE_not_all_sqrt5_periodic_all_sqrt3_periodic_l2825_282512


namespace NUMINAMATH_CALUDE_starting_number_is_271_l2825_282575

/-- A function that checks if a natural number contains the digit 1 -/
def contains_one (n : ℕ) : Bool := sorry

/-- The count of numbers from 1 to 1000 (exclusive) that do not contain the digit 1 -/
def count_no_one_to_1000 : ℕ := sorry

/-- The theorem to prove -/
theorem starting_number_is_271 (count_between : ℕ) 
  (h1 : count_between = 728) 
  (h2 : ∀ n ∈ Finset.range (1000 - 271), 
    ¬contains_one (n + 271) ↔ n < count_between) : 
  271 = 1000 - count_between - 1 :=
sorry

end NUMINAMATH_CALUDE_starting_number_is_271_l2825_282575


namespace NUMINAMATH_CALUDE_power_sum_equality_l2825_282543

theorem power_sum_equality : (3^2)^3 + (2^3)^2 = 793 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l2825_282543


namespace NUMINAMATH_CALUDE_line_circle_intersection_l2825_282525

/-- Given a point (a,b) outside the circle x^2 + y^2 = r^2, 
    the line ax + by = r^2 intersects the circle and does not pass through the center. -/
theorem line_circle_intersection (a b r : ℝ) (hr : r > 0) (h_outside : a^2 + b^2 > r^2) :
  ∃ (x y : ℝ), x^2 + y^2 = r^2 ∧ a*x + b*y = r^2 ∧ (x ≠ 0 ∨ y ≠ 0) := by
  sorry


end NUMINAMATH_CALUDE_line_circle_intersection_l2825_282525


namespace NUMINAMATH_CALUDE_min_sum_of_quadratic_roots_l2825_282588

theorem min_sum_of_quadratic_roots (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : ∃ x : ℝ, x^2 + a*x + 2*b = 0)
  (h2 : ∃ x : ℝ, x^2 + 2*b*x + a = 0) :
  a + b ≥ 6 := by
sorry

end NUMINAMATH_CALUDE_min_sum_of_quadratic_roots_l2825_282588


namespace NUMINAMATH_CALUDE_sin_cos_square_identity_l2825_282569

theorem sin_cos_square_identity (α : ℝ) : (Real.sin α + Real.cos α)^2 = 1 + Real.sin (2 * α) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_square_identity_l2825_282569


namespace NUMINAMATH_CALUDE_quadratic_minimum_value_l2825_282523

theorem quadratic_minimum_value :
  let f : ℝ → ℝ := λ x ↦ x^2 - 4*x + 5
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ f x_min = 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_value_l2825_282523


namespace NUMINAMATH_CALUDE_five_fridays_in_october_implies_five_mondays_in_november_l2825_282561

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a specific date in a month -/
structure Date :=
  (day : Nat)
  (dayOfWeek : DayOfWeek)

def october_has_five_fridays (year : Nat) : Prop :=
  ∃ dates : List Date,
    dates.length = 5 ∧
    ∀ d ∈ dates, d.dayOfWeek = DayOfWeek.Friday ∧ d.day ≤ 31

def november_has_five_mondays (year : Nat) : Prop :=
  ∃ dates : List Date,
    dates.length = 5 ∧
    ∀ d ∈ dates, d.dayOfWeek = DayOfWeek.Monday ∧ d.day ≤ 30

theorem five_fridays_in_october_implies_five_mondays_in_november (year : Nat) :
  october_has_five_fridays year → november_has_five_mondays year :=
by
  sorry


end NUMINAMATH_CALUDE_five_fridays_in_october_implies_five_mondays_in_november_l2825_282561


namespace NUMINAMATH_CALUDE_board_numbers_problem_l2825_282528

theorem board_numbers_problem (a b c : ℕ) :
  70 ≤ a ∧ a < 80 ∧
  60 ≤ b ∧ b < 70 ∧
  50 ≤ c ∧ c < 60 ∧
  a + b = 147 ∧
  120 ≤ a + c ∧ a + c < 130 ∧
  120 ≤ b + c ∧ b + c < 130 ∧
  a + c ≠ b + c →
  a = 78 :=
by sorry

end NUMINAMATH_CALUDE_board_numbers_problem_l2825_282528


namespace NUMINAMATH_CALUDE_tile_area_calculation_l2825_282535

/-- Given a rectangular room and tiles covering a fraction of it, calculate the area of each tile. -/
theorem tile_area_calculation (room_length room_width : ℝ) (num_tiles : ℕ) (fraction_covered : ℚ) :
  room_length = 12 →
  room_width = 20 →
  num_tiles = 40 →
  fraction_covered = 1/6 →
  (room_length * room_width * fraction_covered) / num_tiles = 1 := by
  sorry

end NUMINAMATH_CALUDE_tile_area_calculation_l2825_282535


namespace NUMINAMATH_CALUDE_complex_equation_sum_l2825_282533

theorem complex_equation_sum (a b : ℝ) (i : ℂ) : 
  i * i = -1 →
  (a + 2 * i) / i = b - i →
  a + b = 3 := by sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l2825_282533


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l2825_282542

/-- The volume of a rectangular box given the areas of its faces -/
theorem rectangular_box_volume (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (area1 : a * b = 36) (area2 : b * c = 12) (area3 : a * c = 9) : 
  a * b * c = 144 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l2825_282542


namespace NUMINAMATH_CALUDE_ladybug_dots_count_l2825_282516

/-- The number of ladybugs Andre caught on Monday -/
def monday_ladybugs : ℕ := 8

/-- The number of ladybugs Andre caught on Tuesday -/
def tuesday_ladybugs : ℕ := 5

/-- The number of dots each ladybug has -/
def dots_per_ladybug : ℕ := 6

/-- The total number of dots on all ladybugs caught by Andre -/
def total_dots : ℕ := (monday_ladybugs + tuesday_ladybugs) * dots_per_ladybug

theorem ladybug_dots_count : total_dots = 78 := by
  sorry

end NUMINAMATH_CALUDE_ladybug_dots_count_l2825_282516


namespace NUMINAMATH_CALUDE_complex_modulus_l2825_282563

theorem complex_modulus (z : ℂ) : (2 - I) * z = 3 + I → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l2825_282563


namespace NUMINAMATH_CALUDE_min_rubles_to_win_l2825_282585

/-- Represents the state of the game machine --/
structure GameState :=
  (points : ℕ)
  (rubles : ℕ)

/-- Defines the possible moves in the game --/
inductive Move
  | AddOne
  | Double

/-- Applies a move to the current game state --/
def applyMove (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.AddOne => { points := state.points + 1, rubles := state.rubles + 1 }
  | Move.Double => { points := state.points * 2, rubles := state.rubles + 2 }

/-- Checks if the given state is valid (not exceeding 50 points) --/
def isValidState (state : GameState) : Prop :=
  state.points ≤ 50

/-- Checks if the given state is a winning state (exactly 50 points) --/
def isWinningState (state : GameState) : Prop :=
  state.points = 50

/-- Theorem stating that 11 rubles is the minimum amount needed to win --/
theorem min_rubles_to_win :
  ∃ (moves : List Move),
    let finalState := moves.foldl applyMove { points := 0, rubles := 0 }
    isValidState finalState ∧
    isWinningState finalState ∧
    finalState.rubles = 11 ∧
    (∀ (otherMoves : List Move),
      let otherFinalState := otherMoves.foldl applyMove { points := 0, rubles := 0 }
      isValidState otherFinalState → isWinningState otherFinalState →
      otherFinalState.rubles ≥ 11) :=
by sorry


end NUMINAMATH_CALUDE_min_rubles_to_win_l2825_282585


namespace NUMINAMATH_CALUDE_square_sum_theorem_l2825_282548

theorem square_sum_theorem (x y : ℝ) 
  (h1 : (x + y)^4 + (x - y)^4 = 4112)
  (h2 : x^2 - y^2 = 16) : 
  x^2 + y^2 = 34 := by
sorry

end NUMINAMATH_CALUDE_square_sum_theorem_l2825_282548


namespace NUMINAMATH_CALUDE_upward_shift_quadratic_l2825_282559

/-- The original quadratic function -/
def f (x : ℝ) : ℝ := -x^2

/-- The amount of upward shift -/
def shift : ℝ := 2

/-- The shifted function -/
def g (x : ℝ) : ℝ := f x + shift

theorem upward_shift_quadratic :
  ∀ x : ℝ, g x = -(x^2) + 2 := by
  sorry

end NUMINAMATH_CALUDE_upward_shift_quadratic_l2825_282559


namespace NUMINAMATH_CALUDE_cube_inequality_l2825_282532

theorem cube_inequality (a b : ℝ) (h : a > b) : a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_inequality_l2825_282532


namespace NUMINAMATH_CALUDE_data_analytics_course_hours_l2825_282526

/-- Calculates the total hours spent on a course given the course duration and weekly time commitments. -/
def total_course_hours (weeks : ℕ) (class_hours_1 : ℕ) (class_hours_2 : ℕ) (class_hours_3 : ℕ) (homework_hours : ℕ) : ℕ :=
  weeks * (class_hours_1 + class_hours_2 + class_hours_3 + homework_hours)

/-- Theorem stating that the total hours spent on the described course is 336. -/
theorem data_analytics_course_hours : 
  total_course_hours 24 3 3 4 4 = 336 := by
  sorry

end NUMINAMATH_CALUDE_data_analytics_course_hours_l2825_282526


namespace NUMINAMATH_CALUDE_product_divisible_by_five_l2825_282508

theorem product_divisible_by_five (a b : ℕ+) :
  (5 ∣ (a * b)) → ¬(¬(5 ∣ a) ∧ ¬(5 ∣ b)) := by
  sorry

end NUMINAMATH_CALUDE_product_divisible_by_five_l2825_282508


namespace NUMINAMATH_CALUDE_circle_points_speeds_l2825_282537

/-- Two points moving along a unit circle -/
structure CirclePoints where
  v₁ : ℝ  -- Speed of the first point
  v₂ : ℝ  -- Speed of the second point

/-- Conditions for the circle points -/
def satisfies_conditions (cp : CirclePoints) : Prop :=
  cp.v₁ > 0 ∧ cp.v₂ > 0 ∧  -- Positive speeds
  cp.v₁ - cp.v₂ = 1 / 720 ∧  -- Meet every 12 minutes (720 seconds)
  1 / cp.v₂ - 1 / cp.v₁ = 10  -- First point is 10 seconds faster

/-- The theorem to be proved -/
theorem circle_points_speeds (cp : CirclePoints) 
  (h : satisfies_conditions cp) : cp.v₁ = 1/80 ∧ cp.v₂ = 1/90 := by
  sorry

end NUMINAMATH_CALUDE_circle_points_speeds_l2825_282537


namespace NUMINAMATH_CALUDE_cubic_fraction_factorization_l2825_282504

theorem cubic_fraction_factorization (x y z : ℝ) :
  ((x^2 - y^2)^3 + (y^2 - z^2)^3 + (z^2 - x^2)^3) / ((x - y)^3 + (y - z)^3 + (z - x)^3) = (x + y) * (y + z) * (z + x) :=
by sorry

end NUMINAMATH_CALUDE_cubic_fraction_factorization_l2825_282504


namespace NUMINAMATH_CALUDE_blue_gumdrops_after_replacement_l2825_282594

theorem blue_gumdrops_after_replacement (total : ℕ) (blue_percent : ℚ) (brown_percent : ℚ) 
  (red_percent : ℚ) (yellow_percent : ℚ) (h_total : total = 150)
  (h_blue : blue_percent = 1/4) (h_brown : brown_percent = 1/4)
  (h_red : red_percent = 1/5) (h_yellow : yellow_percent = 1/10)
  (h_sum : blue_percent + brown_percent + red_percent + yellow_percent < 1) :
  let initial_blue := ⌈total * blue_percent⌉
  let initial_red := ⌊total * red_percent⌋
  let replaced_red := ⌊initial_red * (3/4)⌋
  initial_blue + replaced_red = 60 := by
  sorry

end NUMINAMATH_CALUDE_blue_gumdrops_after_replacement_l2825_282594


namespace NUMINAMATH_CALUDE_max_value_expression_l2825_282558

theorem max_value_expression (a b c d : ℕ) : 
  a ∈ ({1, 2, 3, 4} : Set ℕ) →
  b ∈ ({1, 2, 3, 4} : Set ℕ) →
  c ∈ ({1, 2, 3, 4} : Set ℕ) →
  d ∈ ({1, 2, 3, 4} : Set ℕ) →
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
  c * a^b - d ≤ 127 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l2825_282558


namespace NUMINAMATH_CALUDE_shaded_fraction_of_square_l2825_282589

/-- Given a square with side length x, where P is at a corner and Q is at the midpoint of an adjacent side,
    the fraction of the square's interior that is shaded is 3/4. -/
theorem shaded_fraction_of_square (x : ℝ) (h : x > 0) : 
  let square_area := x^2
  let triangle_area := (1/2) * x * (x/2)
  let shaded_area := square_area - triangle_area
  shaded_area / square_area = 3/4 := by
sorry

end NUMINAMATH_CALUDE_shaded_fraction_of_square_l2825_282589


namespace NUMINAMATH_CALUDE_sqrt_product_plus_one_l2825_282550

theorem sqrt_product_plus_one : 
  Real.sqrt ((35 : ℝ) * 34 * 33 * 32 + 1) = 1121 := by sorry

end NUMINAMATH_CALUDE_sqrt_product_plus_one_l2825_282550


namespace NUMINAMATH_CALUDE_existence_of_mn_l2825_282596

theorem existence_of_mn (p : ℕ) (hp : p.Prime) (hp_gt_5 : p > 5) :
  ∃ m n : ℕ, m + n < p ∧ p ∣ (2^m * 3^n - 1) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_mn_l2825_282596


namespace NUMINAMATH_CALUDE_intersection_when_m_neg_three_subset_condition_l2825_282557

-- Define sets A and B
def A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 4}
def B (m : ℝ) : Set ℝ := {x | 2*m - 1 ≤ x ∧ x ≤ m + 1}

-- Theorem 1: A ∩ B when m = -3
theorem intersection_when_m_neg_three :
  A ∩ B (-3) = {x | -3 ≤ x ∧ x ≤ -2} := by sorry

-- Theorem 2: B ⊆ A iff m ≥ -1
theorem subset_condition :
  ∀ m : ℝ, B m ⊆ A ↔ m ≥ -1 := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_neg_three_subset_condition_l2825_282557


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2825_282595

/-- Given an arithmetic sequence {a_n} with common ratio q ≠ 1,
    if a_1 * a_2 * a_3 = -1/8 and (a_2, a_4, a_3) forms an arithmetic sequence,
    then the sum of the first 4 terms of {a_n} is equal to 5/8. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  q ≠ 1 →
  (∀ n, a (n + 1) = a n * q) →
  a 1 * a 2 * a 3 = -1/8 →
  2 * a 4 = a 2 + a 3 →
  (a 1 + a 2 + a 3 + a 4 = 5/8) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2825_282595


namespace NUMINAMATH_CALUDE_polygon_exterior_angles_l2825_282555

theorem polygon_exterior_angles (n : ℕ) (exterior_angle : ℝ) : 
  (n > 2) → (exterior_angle = 45) → (n * exterior_angle = 360) → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_polygon_exterior_angles_l2825_282555


namespace NUMINAMATH_CALUDE_special_numbers_theorem_l2825_282502

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

def satisfies_condition (n : ℕ) : Prop :=
  90 ≤ n ∧ n ≤ 150 ∧ digit_sum (digit_sum n) = 1

theorem special_numbers_theorem : 
  {n : ℕ | satisfies_condition n} = {91, 100, 109, 118, 127, 136, 145} := by
  sorry

end NUMINAMATH_CALUDE_special_numbers_theorem_l2825_282502


namespace NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l2825_282503

theorem quadratic_inequality_equivalence :
  ∀ x : ℝ, x * (2 * x + 3) < 9 ↔ x ∈ Set.Ioo (-5/2) 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l2825_282503


namespace NUMINAMATH_CALUDE_seth_purchase_difference_l2825_282578

/-- Calculates the difference in cost between discounted ice cream and yogurt purchases. -/
def ice_cream_yogurt_cost_difference (
  ice_cream_cartons : ℕ)
  (yogurt_cartons : ℕ)
  (ice_cream_price : ℚ)
  (yogurt_price : ℚ)
  (ice_cream_discount : ℚ)
  (yogurt_discount : ℚ) : ℚ :=
  let ice_cream_cost := ice_cream_cartons * ice_cream_price * (1 - ice_cream_discount)
  let yogurt_cost := yogurt_cartons * yogurt_price * (1 - yogurt_discount)
  ice_cream_cost - yogurt_cost

theorem seth_purchase_difference :
  ice_cream_yogurt_cost_difference 20 2 6 1 (1/10) (1/5) = 1064/10 := by
  sorry

end NUMINAMATH_CALUDE_seth_purchase_difference_l2825_282578


namespace NUMINAMATH_CALUDE_arithmetic_sequence_range_l2825_282592

/-- 
For an arithmetic sequence with first term a₁ = -10 and common difference d,
if the 10th term and all subsequent terms are positive,
then 10/9 < d ≤ 5/4.
-/
theorem arithmetic_sequence_range (d : ℝ) : 
  (∀ n : ℕ, n ≥ 10 → -10 + (n - 1) * d > 0) → 
  10/9 < d ∧ d ≤ 5/4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_range_l2825_282592


namespace NUMINAMATH_CALUDE_total_carriages_l2825_282584

theorem total_carriages (euston norfolk norwich flying_scotsman : ℕ) : 
  euston = norfolk + 20 →
  norwich = 100 →
  flying_scotsman = norwich + 20 →
  euston = 130 →
  euston + norfolk + norwich + flying_scotsman = 460 := by
  sorry

end NUMINAMATH_CALUDE_total_carriages_l2825_282584


namespace NUMINAMATH_CALUDE_quadratic_minimum_value_l2825_282590

def f (x : ℝ) : ℝ := 2 * x^2 + 6 * x + 5

theorem quadratic_minimum_value :
  (f 1 = 13) →
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ f x_min = 0.5 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_value_l2825_282590


namespace NUMINAMATH_CALUDE_parabola_equilateral_distance_l2825_282527

/-- Parabola structure -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ
  directrix : ℝ → ℝ → Prop

/-- Point on a parabola -/
def PointOnParabola (p : Parabola) (point : ℝ × ℝ) : Prop :=
  p.equation point.1 point.2

/-- Perpendicular to directrix -/
def PerpendicularToDirectrix (p : Parabola) (point : ℝ × ℝ) (foot : ℝ × ℝ) : Prop :=
  p.directrix foot.1 foot.2 ∧ 
  (point.1 - foot.1) * (p.focus.1 - foot.1) + (point.2 - foot.2) * (p.focus.2 - foot.2) = 0

/-- Equilateral triangle -/
def IsEquilateralTriangle (a b c : ℝ × ℝ) : Prop :=
  let dist := fun (p q : ℝ × ℝ) => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  dist a b = dist b c ∧ dist b c = dist c a

/-- Main theorem -/
theorem parabola_equilateral_distance (p : Parabola) (point : ℝ × ℝ) (foot : ℝ × ℝ) :
  p.equation = fun x y => y^2 = 6*x →
  PointOnParabola p point →
  PerpendicularToDirectrix p point foot →
  IsEquilateralTriangle point foot p.focus →
  Real.sqrt ((point.1 - p.focus.1)^2 + (point.2 - p.focus.2)^2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_parabola_equilateral_distance_l2825_282527


namespace NUMINAMATH_CALUDE_triangle_area_l2825_282593

theorem triangle_area (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) : 
  (1/2) * a * b = 24 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2825_282593


namespace NUMINAMATH_CALUDE_no_double_composition_square_minus_two_l2825_282529

theorem no_double_composition_square_minus_two :
  ¬ ∃ f : ℝ → ℝ, ∀ x : ℝ, f (f x) = x^2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_no_double_composition_square_minus_two_l2825_282529


namespace NUMINAMATH_CALUDE_max_profit_multimedia_devices_l2825_282521

/-- Represents the profit function for multimedia devices -/
def profit_function (x : ℝ) : ℝ := -0.1 * x + 20

/-- Represents the constraint on the quantity of devices -/
def quantity_constraint (x : ℝ) : Prop := 4 * x ≥ 50 - x

/-- Theorem stating the maximum profit and optimal quantity of type A devices -/
theorem max_profit_multimedia_devices :
  ∃ (x : ℝ), 
    quantity_constraint x ∧ 
    profit_function x = 19 ∧ 
    x = 10 ∧
    ∀ (y : ℝ), quantity_constraint y → profit_function y ≤ profit_function x :=
by
  sorry


end NUMINAMATH_CALUDE_max_profit_multimedia_devices_l2825_282521


namespace NUMINAMATH_CALUDE_desktop_computers_sold_l2825_282541

theorem desktop_computers_sold (total : ℕ) (laptops : ℕ) (netbooks : ℕ) (desktops : ℕ)
  (h1 : total = 72)
  (h2 : laptops = total / 2)
  (h3 : netbooks = total / 3)
  (h4 : desktops = total - laptops - netbooks) :
  desktops = 12 := by
  sorry

end NUMINAMATH_CALUDE_desktop_computers_sold_l2825_282541


namespace NUMINAMATH_CALUDE_genevieve_error_count_l2825_282518

/-- The number of lines of code Genevieve has written -/
def total_lines : ℕ := 4300

/-- The number of lines per debug block -/
def lines_per_block : ℕ := 100

/-- The number of errors found in the first block -/
def initial_errors : ℕ := 3

/-- The increase in errors found per block -/
def error_increase : ℕ := 1

/-- The number of completed debug blocks -/
def num_blocks : ℕ := total_lines / lines_per_block

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (n : ℕ) (a₁ : ℕ) (d : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- The total number of errors fixed -/
def total_errors : ℕ := arithmetic_sum num_blocks initial_errors error_increase

theorem genevieve_error_count :
  total_errors = 1032 := by sorry

end NUMINAMATH_CALUDE_genevieve_error_count_l2825_282518


namespace NUMINAMATH_CALUDE_min_value_x_plus_four_over_x_l2825_282580

theorem min_value_x_plus_four_over_x (x : ℝ) (h : x > 0) :
  x + 4 / x ≥ 4 ∧ (x + 4 / x = 4 ↔ x = 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_four_over_x_l2825_282580


namespace NUMINAMATH_CALUDE_smallest_number_with_remainders_l2825_282549

theorem smallest_number_with_remainders : ∃ (b : ℕ), b = 87 ∧
  b % 5 = 2 ∧ b % 4 = 3 ∧ b % 7 = 1 ∧
  ∀ (n : ℕ), n % 5 = 2 ∧ n % 4 = 3 ∧ n % 7 = 1 → b ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainders_l2825_282549


namespace NUMINAMATH_CALUDE_complex_imaginary_operation_l2825_282598

theorem complex_imaginary_operation : Complex.I - (1 / Complex.I) = 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_imaginary_operation_l2825_282598


namespace NUMINAMATH_CALUDE_bricks_for_wall_l2825_282556

/-- Calculates the number of bricks needed to build a wall -/
def bricks_needed (wall_length wall_height wall_thickness brick_length brick_width brick_height : ℕ) : ℕ :=
  let wall_volume := wall_length * wall_height * wall_thickness
  let brick_volume := brick_length * brick_width * brick_height
  (wall_volume + brick_volume - 1) / brick_volume

/-- Theorem stating the number of bricks needed for the given wall and brick dimensions -/
theorem bricks_for_wall : bricks_needed 800 600 2 5 11 6 = 2910 := by
  sorry

end NUMINAMATH_CALUDE_bricks_for_wall_l2825_282556


namespace NUMINAMATH_CALUDE_inscribed_squares_problem_l2825_282517

theorem inscribed_squares_problem (a b : ℝ) : 
  let small_area : ℝ := 16
  let large_area : ℝ := 18
  let rotation_angle : ℝ := 30 * π / 180
  let small_side : ℝ := Real.sqrt small_area
  let large_side : ℝ := Real.sqrt large_area
  a + b = large_side ∧ 
  Real.sqrt (a^2 + b^2) = 2 * small_side * Real.cos rotation_angle →
  a * b = -15 := by
sorry

end NUMINAMATH_CALUDE_inscribed_squares_problem_l2825_282517


namespace NUMINAMATH_CALUDE_existence_condition_l2825_282582

theorem existence_condition (a : ℝ) : 
  (∃ x y : ℝ, Real.sqrt (2 * x * y + a) = x + y + 17) ↔ a ≥ -289/2 := by
sorry

end NUMINAMATH_CALUDE_existence_condition_l2825_282582


namespace NUMINAMATH_CALUDE_triangle_side_difference_l2825_282546

theorem triangle_side_difference (x : ℤ) : 
  (x > 5 ∧ x < 11) → (11 - 6 = 4) := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_difference_l2825_282546


namespace NUMINAMATH_CALUDE_not_all_new_releases_implies_exists_not_new_and_not_all_new_l2825_282576

-- Define the universe of books in the library
variable (Book : Type)

-- Define the property of being a new release
variable (is_new_release : Book → Prop)

-- Define the library as a set of books
variable (library : Set Book)

-- Theorem stating that if not all books are new releases, 
-- then there exists a book that is not a new release and not all books are new releases
theorem not_all_new_releases_implies_exists_not_new_and_not_all_new
  (h : ¬(∀ b ∈ library, is_new_release b)) :
  (∃ b ∈ library, ¬(is_new_release b)) ∧ ¬(∀ b ∈ library, is_new_release b) := by
sorry

end NUMINAMATH_CALUDE_not_all_new_releases_implies_exists_not_new_and_not_all_new_l2825_282576


namespace NUMINAMATH_CALUDE_prob_A_miss_at_least_once_prob_A_hit_twice_B_hit_once_l2825_282539

-- Define the probabilities
def prob_A_hit : ℚ := 2/3
def prob_B_hit : ℚ := 3/4

-- Define the number of shots
def num_shots : ℕ := 3

-- Theorem for part (I)
theorem prob_A_miss_at_least_once :
  1 - prob_A_hit ^ num_shots = 19/27 := by sorry

-- Theorem for part (II)
theorem prob_A_hit_twice_B_hit_once :
  (Nat.choose num_shots 2 : ℚ) * prob_A_hit^2 * (1 - prob_A_hit) *
  (Nat.choose num_shots 1 : ℚ) * prob_B_hit * (1 - prob_B_hit)^2 = 1/16 := by sorry

end NUMINAMATH_CALUDE_prob_A_miss_at_least_once_prob_A_hit_twice_B_hit_once_l2825_282539


namespace NUMINAMATH_CALUDE_m_salary_percentage_l2825_282553

/-- The percentage of m's salary compared to n's salary -/
def salary_percentage (total_salary n_salary : ℚ) : ℚ :=
  (total_salary - n_salary) / n_salary * 100

/-- Proof that m's salary is 120% of n's salary -/
theorem m_salary_percentage :
  let total_salary : ℚ := 572
  let n_salary : ℚ := 260
  salary_percentage total_salary n_salary = 120 := by
  sorry

end NUMINAMATH_CALUDE_m_salary_percentage_l2825_282553


namespace NUMINAMATH_CALUDE_sum_greater_than_2e_squared_l2825_282564

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x

theorem sum_greater_than_2e_squared (a : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) 
  (hf₁ : f a x₁ = 1) (hf₂ : f a x₂ = 1) : 
  x₁ + x₂ > 2 * (Real.exp 1) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_greater_than_2e_squared_l2825_282564


namespace NUMINAMATH_CALUDE_sum_odd_9_to_39_l2825_282566

/-- Sum of first n consecutive odd integers -/
def sum_first_n_odd (n : ℕ) : ℕ := n ^ 2

/-- The nth odd integer -/
def nth_odd (n : ℕ) : ℕ := 2 * n - 1

/-- Sum of odd integers from a to b inclusive -/
def sum_odd_range (a b : ℕ) : ℕ :=
  sum_first_n_odd ((b - 1) / 2 + 1) - sum_first_n_odd ((a - 1) / 2)

theorem sum_odd_9_to_39 :
  sum_odd_range 9 39 = 384 :=
sorry

end NUMINAMATH_CALUDE_sum_odd_9_to_39_l2825_282566


namespace NUMINAMATH_CALUDE_additional_miles_is_33_l2825_282544

/-- Represents the distances between locations in Kona's trip -/
structure TripDistances where
  apartment_to_bakery : ℕ
  bakery_to_grandma : ℕ
  grandma_to_apartment : ℕ

/-- Calculates the additional miles driven with bakery stop compared to without -/
def additional_miles (d : TripDistances) : ℕ :=
  d.apartment_to_bakery + d.bakery_to_grandma + d.grandma_to_apartment - 2 * d.grandma_to_apartment

/-- Theorem stating that the additional miles driven with bakery stop is 33 -/
theorem additional_miles_is_33 (d : TripDistances) 
    (h1 : d.apartment_to_bakery = 9)
    (h2 : d.bakery_to_grandma = 24)
    (h3 : d.grandma_to_apartment = 27) : 
  additional_miles d = 33 := by
  sorry

end NUMINAMATH_CALUDE_additional_miles_is_33_l2825_282544


namespace NUMINAMATH_CALUDE_flour_added_indeterminate_l2825_282571

/-- Represents the ingredients in cups -/
structure Ingredients where
  sugar : ℕ
  flour : ℕ
  salt : ℕ

/-- Represents the current state of Mary's baking process -/
structure BakingState where
  recipe : Ingredients
  flour_added : ℕ
  sugar_to_add : ℕ
  salt_to_add : ℕ

/-- The recipe requirements -/
def recipe : Ingredients :=
  { sugar := 11, flour := 6, salt := 9 }

/-- Theorem stating that the amount of flour already added cannot be uniquely determined -/
theorem flour_added_indeterminate (state : BakingState) : 
  state.recipe = recipe → 
  state.sugar_to_add = state.salt_to_add + 2 → 
  ∃ (x y : ℕ), x ≠ y ∧ 
    (∃ (state1 state2 : BakingState), 
      state1.flour_added = x ∧ 
      state2.flour_added = y ∧ 
      state1.recipe = state.recipe ∧ 
      state2.recipe = state.recipe ∧ 
      state1.sugar_to_add = state.sugar_to_add ∧ 
      state2.sugar_to_add = state.sugar_to_add ∧ 
      state1.salt_to_add = state.salt_to_add ∧ 
      state2.salt_to_add = state.salt_to_add) :=
by
  sorry

end NUMINAMATH_CALUDE_flour_added_indeterminate_l2825_282571


namespace NUMINAMATH_CALUDE_buffy_stolen_apples_l2825_282505

theorem buffy_stolen_apples (initial_apples : ℕ) (fallen_apples : ℕ) (remaining_apples : ℕ) 
  (h1 : initial_apples = 79)
  (h2 : fallen_apples = 26)
  (h3 : remaining_apples = 8) :
  initial_apples - fallen_apples - remaining_apples = 45 :=
by sorry

end NUMINAMATH_CALUDE_buffy_stolen_apples_l2825_282505


namespace NUMINAMATH_CALUDE_simplify_expression_l2825_282531

theorem simplify_expression : (9 * 10^8) / (3 * 10^3) = 300000 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2825_282531


namespace NUMINAMATH_CALUDE_complex_number_equality_l2825_282540

theorem complex_number_equality : ∀ (i : ℂ), i * i = -1 →
  (2 * i) / (1 + i) = 1 + i := by sorry

end NUMINAMATH_CALUDE_complex_number_equality_l2825_282540


namespace NUMINAMATH_CALUDE_solution_set_f_range_of_m_l2825_282581

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 1|
def g (x m : ℝ) : ℝ := -|x + 3| + m

-- Statement 1: Solution set of f(x) + x^2 - 1 > 0
theorem solution_set_f (x : ℝ) : f x + x^2 - 1 > 0 ↔ x > 1 ∨ x < 0 := by sorry

-- Statement 2: Range of m when solution set of f(x) < g(x) is non-empty
theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, f x < g x m) → m > 4 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_range_of_m_l2825_282581


namespace NUMINAMATH_CALUDE_vector_subtraction_scalar_multiplication_l2825_282514

def vector_operation (v1 v2 : ℝ × ℝ) (s : ℝ) : ℝ × ℝ :=
  (v1.1 - s * v2.1, v1.2 - s * v2.2)

theorem vector_subtraction_scalar_multiplication :
  vector_operation (3, -8) (2, -6) 5 = (-7, 22) := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_scalar_multiplication_l2825_282514


namespace NUMINAMATH_CALUDE_binomial_coefficient_x_plus_two_to_seven_coefficient_of_x_fifth_power_l2825_282572

theorem binomial_coefficient_x_plus_two_to_seven (x : ℝ) : 
  (Finset.range 8).sum (λ k => (Nat.choose 7 k : ℝ) * x^k * 2^(7-k)) = 
    x^7 + 14*x^6 + 84*x^5 + 280*x^4 + 560*x^3 + 672*x^2 + 448*x + 128 :=
by sorry

theorem coefficient_of_x_fifth_power : 
  (Finset.range 8).sum (λ k => (Nat.choose 7 k : ℝ) * 1^k * 2^(7-k) * 
    (if k = 5 then 1 else 0)) = 84 :=
by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_x_plus_two_to_seven_coefficient_of_x_fifth_power_l2825_282572


namespace NUMINAMATH_CALUDE_simon_age_is_10_l2825_282513

def alvin_age : ℕ := 30

def simon_age : ℕ := alvin_age / 2 - 5

theorem simon_age_is_10 : simon_age = 10 := by
  sorry

end NUMINAMATH_CALUDE_simon_age_is_10_l2825_282513


namespace NUMINAMATH_CALUDE_six_pointed_star_perimeter_l2825_282570

/-- A regular hexagon with perimeter 3 meters -/
structure RegularHexagon :=
  (perimeter : ℝ)
  (is_regular : perimeter = 3)

/-- A six-pointed star formed by extending the sides of a regular hexagon -/
structure SixPointedStar (h : RegularHexagon) :=
  (perimeter : ℝ)

/-- The perimeter of the six-pointed star is 4√3 meters -/
theorem six_pointed_star_perimeter (h : RegularHexagon) (s : SixPointedStar h) :
  s.perimeter = 4 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_six_pointed_star_perimeter_l2825_282570


namespace NUMINAMATH_CALUDE_students_in_both_clubs_l2825_282599

theorem students_in_both_clubs
  (total_students : ℕ)
  (drama_club : ℕ)
  (science_club : ℕ)
  (drama_or_science : ℕ)
  (h1 : total_students = 300)
  (h2 : drama_club = 100)
  (h3 : science_club = 140)
  (h4 : drama_or_science = 210) :
  drama_club + science_club - drama_or_science = 30 := by
  sorry

end NUMINAMATH_CALUDE_students_in_both_clubs_l2825_282599


namespace NUMINAMATH_CALUDE_children_retaking_test_l2825_282545

theorem children_retaking_test (total : ℝ) (passed : ℝ) (retaking : ℝ) : 
  total = 698.0 → passed = 105.0 → retaking = total - passed → retaking = 593.0 := by
sorry

end NUMINAMATH_CALUDE_children_retaking_test_l2825_282545


namespace NUMINAMATH_CALUDE_circle_lattice_point_uniqueness_l2825_282552

theorem circle_lattice_point_uniqueness (r : ℝ) (hr : r > 0) :
  ∃! (x y : ℤ), (↑x - Real.sqrt 2)^2 + (↑y - 1/3)^2 = r^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_lattice_point_uniqueness_l2825_282552


namespace NUMINAMATH_CALUDE_bee_count_l2825_282551

theorem bee_count (initial_bees : ℕ) (incoming_bees : ℕ) : 
  initial_bees = 16 → incoming_bees = 8 → initial_bees + incoming_bees = 24 := by
  sorry

end NUMINAMATH_CALUDE_bee_count_l2825_282551


namespace NUMINAMATH_CALUDE_marathon_total_distance_l2825_282547

/-- Represents the length of a marathon in miles and yards -/
structure Marathon where
  miles : ℕ
  yards : ℕ

/-- Represents a total distance in miles and yards -/
structure TotalDistance where
  miles : ℕ
  yards : ℕ
  h : yards < 1760

def marathon_length : Marathon := { miles := 26, yards := 385 }

def yards_per_mile : ℕ := 1760

def num_marathons : ℕ := 10

theorem marathon_total_distance :
  ∃ (m : ℕ) (y : ℕ) (h : y < 1760),
    (m * yards_per_mile + y) = 
      (num_marathons * marathon_length.miles * yards_per_mile + 
       num_marathons * marathon_length.yards) ∧
    y = 330 := by sorry

end NUMINAMATH_CALUDE_marathon_total_distance_l2825_282547


namespace NUMINAMATH_CALUDE_fraction_simplification_l2825_282579

theorem fraction_simplification (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (x^2 + x*y) / (x*y) * y^2 / (x + y) = y :=
sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2825_282579


namespace NUMINAMATH_CALUDE_sum_of_sequence_equals_11920_l2825_282524

def integerSequence : List Nat := List.range 40 |>.map (fun i => 103 + 10 * i)

theorem sum_of_sequence_equals_11920 : (integerSequence.sum = 11920) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_sequence_equals_11920_l2825_282524


namespace NUMINAMATH_CALUDE_jeremy_purchase_l2825_282511

theorem jeremy_purchase (computer_price : ℝ) (accessory_percentage : ℝ) (initial_money_factor : ℝ) : 
  computer_price = 3000 →
  accessory_percentage = 0.1 →
  initial_money_factor = 2 →
  let accessory_price := computer_price * accessory_percentage
  let initial_money := computer_price * initial_money_factor
  let total_spent := computer_price + accessory_price
  initial_money - total_spent = 2700 := by
sorry

end NUMINAMATH_CALUDE_jeremy_purchase_l2825_282511


namespace NUMINAMATH_CALUDE_slope_parallel_sufficient_not_necessary_l2825_282519

-- Define a structure for a line with a slope
structure Line where
  slope : ℝ

-- Define what it means for two lines to be parallel
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

theorem slope_parallel_sufficient_not_necessary :
  ∃ (l1 l2 : Line),
    (parallel l1 l2 → l1.slope = l2.slope) ∧
    ∃ (l3 l4 : Line), l3.slope = l4.slope ∧ ¬ parallel l3 l4 := by
  sorry

end NUMINAMATH_CALUDE_slope_parallel_sufficient_not_necessary_l2825_282519


namespace NUMINAMATH_CALUDE_house_of_cards_impossible_l2825_282568

theorem house_of_cards_impossible (decks : ℕ) (cards_per_deck : ℕ) (layers : ℕ) : 
  decks = 36 → cards_per_deck = 104 → layers = 64 → 
  ¬ ∃ (cards_per_layer : ℕ), (decks * cards_per_deck) = (layers * cards_per_layer) :=
by
  sorry

end NUMINAMATH_CALUDE_house_of_cards_impossible_l2825_282568


namespace NUMINAMATH_CALUDE_car_average_speed_l2825_282522

theorem car_average_speed 
  (total_time : ℝ) 
  (initial_time : ℝ) 
  (initial_speed : ℝ) 
  (remaining_speed : ℝ) 
  (h1 : total_time = 24) 
  (h2 : initial_time = 4) 
  (h3 : initial_speed = 35) 
  (h4 : remaining_speed = 53) : 
  (initial_speed * initial_time + remaining_speed * (total_time - initial_time)) / total_time = 50 := by
  sorry

end NUMINAMATH_CALUDE_car_average_speed_l2825_282522


namespace NUMINAMATH_CALUDE_g_behavior_l2825_282554

def g (x : ℝ) : ℝ := 3 * x^3 - 5 * x^2 + 1

theorem g_behavior :
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x > N → g x > M) ∧
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x < N → g x < M) := by
sorry

end NUMINAMATH_CALUDE_g_behavior_l2825_282554


namespace NUMINAMATH_CALUDE_friends_weekly_biking_distance_l2825_282538

/-- The total distance two friends bike in a week -/
def total_distance_biked (onur_daily_distance : ℕ) (hanil_extra_distance : ℕ) (days_per_week : ℕ) : ℕ :=
  (onur_daily_distance * days_per_week) + ((onur_daily_distance + hanil_extra_distance) * days_per_week)

/-- Theorem stating the total distance biked by Onur and Hanil in a week -/
theorem friends_weekly_biking_distance :
  total_distance_biked 250 40 5 = 2700 := by
  sorry

#eval total_distance_biked 250 40 5

end NUMINAMATH_CALUDE_friends_weekly_biking_distance_l2825_282538


namespace NUMINAMATH_CALUDE_star_equation_solution_l2825_282577

-- Define the star operation
def star (a b : ℚ) : ℚ := a * b + 3 * b - a

-- State the theorem
theorem star_equation_solution :
  ∀ y : ℚ, star 4 y = 40 → y = 44 / 7 := by
  sorry

end NUMINAMATH_CALUDE_star_equation_solution_l2825_282577


namespace NUMINAMATH_CALUDE_dans_balloons_l2825_282509

theorem dans_balloons (dans_balloons : ℕ) (tims_balloons : ℕ) : 
  tims_balloons = 203 → 
  tims_balloons = 7 * dans_balloons → 
  dans_balloons = 29 := by
sorry

end NUMINAMATH_CALUDE_dans_balloons_l2825_282509


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2825_282520

theorem arithmetic_geometric_sequence (d : ℝ) (a : ℕ → ℝ) :
  d ≠ 0 ∧
  (∀ n, a (n + 1) = a n + d) ∧
  a 1 = 1 ∧
  (a 3) ^ 2 = a 1 * a 13 →
  d = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2825_282520


namespace NUMINAMATH_CALUDE_no_m_exists_for_all_x_inequality_l2825_282573

theorem no_m_exists_for_all_x_inequality :
  ¬ ∃ m : ℝ, ∀ x : ℝ, m * x^2 - 2*x - m + 1 < 0 := by
  sorry

end NUMINAMATH_CALUDE_no_m_exists_for_all_x_inequality_l2825_282573


namespace NUMINAMATH_CALUDE_initial_students_count_l2825_282560

theorem initial_students_count (initial_avg : ℝ) (new_student_weight : ℝ) (new_avg : ℝ) :
  initial_avg = 28 →
  new_student_weight = 4 →
  new_avg = 27.2 →
  ∃ n : ℕ, n * initial_avg + new_student_weight = (n + 1) * new_avg ∧ n = 29 :=
by sorry

end NUMINAMATH_CALUDE_initial_students_count_l2825_282560


namespace NUMINAMATH_CALUDE_truck_filling_time_l2825_282506

/-- Calculates the total time to fill a truck with stone blocks -/
theorem truck_filling_time 
  (truck_capacity : ℕ)
  (rate_per_person : ℕ)
  (initial_workers : ℕ)
  (initial_duration : ℕ)
  (additional_workers : ℕ)
  (h1 : truck_capacity = 6000)
  (h2 : rate_per_person = 250)
  (h3 : initial_workers = 2)
  (h4 : initial_duration = 4)
  (h5 : additional_workers = 6) :
  ∃ (total_time : ℕ), total_time = 6 ∧ 
  (initial_workers * rate_per_person * initial_duration + 
   (initial_workers + additional_workers) * rate_per_person * (total_time - initial_duration) = truck_capacity) :=
by
  sorry


end NUMINAMATH_CALUDE_truck_filling_time_l2825_282506


namespace NUMINAMATH_CALUDE_greatest_common_divisor_of_sum_l2825_282501

/-- An arithmetic sequence with positive integer terms and perfect square common difference -/
structure ArithmeticSequence where
  first_term : ℕ+
  common_difference : ℕ
  is_perfect_square : ∃ (n : ℕ), n^2 = common_difference

/-- The sum of the first 15 terms of an arithmetic sequence -/
def sum_first_15_terms (seq : ArithmeticSequence) : ℕ :=
  15 * seq.first_term + 105 * seq.common_difference

/-- 15 is the greatest positive integer that always divides the sum of the first 15 terms -/
theorem greatest_common_divisor_of_sum (seq : ArithmeticSequence) :
  (∃ (m : ℕ+), m > 15 ∧ (m : ℕ) ∣ sum_first_15_terms seq) → False := by sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_of_sum_l2825_282501


namespace NUMINAMATH_CALUDE_system_solution_implies_m_value_l2825_282587

theorem system_solution_implies_m_value (x y m : ℝ) : 
  (2 * x + y = 6 * m) →
  (3 * x - 2 * y = 2 * m) →
  (x / 3 - y / 5 = 4) →
  m = 15 := by
sorry

end NUMINAMATH_CALUDE_system_solution_implies_m_value_l2825_282587


namespace NUMINAMATH_CALUDE_hexagon_side_length_l2825_282574

/-- A regular hexagon with perimeter 42 cm has sides of length 7 cm each. -/
theorem hexagon_side_length (perimeter : ℝ) (num_sides : ℕ) (h1 : perimeter = 42) (h2 : num_sides = 6) :
  perimeter / num_sides = 7 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_side_length_l2825_282574


namespace NUMINAMATH_CALUDE_total_length_climbed_result_l2825_282583

/-- The total length of ladders climbed by two workers in inches -/
def total_length_climbed (keaton_ladder_height : ℕ) (keaton_climbs : ℕ) 
  (reece_ladder_diff : ℕ) (reece_climbs : ℕ) : ℕ :=
  let reece_ladder_height := keaton_ladder_height - reece_ladder_diff
  let keaton_total := keaton_ladder_height * keaton_climbs
  let reece_total := reece_ladder_height * reece_climbs
  (keaton_total + reece_total) * 12

/-- Theorem stating the total length climbed by both workers -/
theorem total_length_climbed_result : 
  total_length_climbed 30 20 4 15 = 11880 := by
  sorry

end NUMINAMATH_CALUDE_total_length_climbed_result_l2825_282583


namespace NUMINAMATH_CALUDE_incircle_identity_l2825_282507

-- Define a triangle with an incircle
structure TriangleWithIncircle where
  -- The sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- The semi-perimeter
  p : ℝ
  -- The inradius
  r : ℝ
  -- The angle APB
  α : ℝ
  -- Conditions
  side_positive : 0 < a ∧ 0 < b ∧ 0 < c
  semi_perimeter : p = (a + b + c) / 2
  inradius_positive : 0 < r
  angle_positive : 0 < α ∧ α < π / 2

-- The theorem to prove
theorem incircle_identity (t : TriangleWithIncircle) :
  1 / (t.p - t.b) + 1 / (t.p - t.c) = 2 / (t.r * Real.tan t.α) := by
  sorry

end NUMINAMATH_CALUDE_incircle_identity_l2825_282507


namespace NUMINAMATH_CALUDE_min_coach_handshakes_l2825_282534

/-- Represents the number of handshakes involving coaches -/
def coach_handshakes (nA nB : ℕ) : ℕ := 
  620 - (nA.choose 2 + nB.choose 2 + nA * nB)

/-- The main theorem to prove -/
theorem min_coach_handshakes : 
  ∃ (nA nB : ℕ), 
    nA = nB + 2 ∧ 
    nA > 0 ∧ 
    nB > 0 ∧
    ∀ (mA mB : ℕ), 
      mA = mB + 2 → 
      mA > 0 → 
      mB > 0 → 
      coach_handshakes nA nB ≤ coach_handshakes mA mB ∧
      coach_handshakes nA nB = 189 :=
sorry

end NUMINAMATH_CALUDE_min_coach_handshakes_l2825_282534


namespace NUMINAMATH_CALUDE_value_std_dev_below_mean_l2825_282597

def mean : ℝ := 16.2
def std_dev : ℝ := 2.3
def value : ℝ := 11.6

theorem value_std_dev_below_mean : 
  (mean - value) / std_dev = 2 := by sorry

end NUMINAMATH_CALUDE_value_std_dev_below_mean_l2825_282597


namespace NUMINAMATH_CALUDE_projectile_max_height_l2825_282500

/-- The height function of the projectile -/
def h (t : ℝ) : ℝ := -20 * t^2 + 50 * t + 10

/-- The maximum height reached by the projectile -/
def max_height : ℝ := 41.25

/-- Theorem stating that the maximum value of h(t) is equal to max_height -/
theorem projectile_max_height : 
  ∀ t : ℝ, h t ≤ max_height ∧ ∃ t₀ : ℝ, h t₀ = max_height :=
sorry

end NUMINAMATH_CALUDE_projectile_max_height_l2825_282500


namespace NUMINAMATH_CALUDE_third_number_is_two_l2825_282562

def is_valid_sequence (seq : List Nat) : Prop :=
  seq.length = 37 ∧
  seq.toFinset = Finset.range 37 ∧
  ∀ i j, i < j → j < seq.length → (seq.take j).sum % seq[j]! = 0

theorem third_number_is_two (seq : List Nat) :
  is_valid_sequence seq →
  seq[0]! = 37 →
  seq[1]! = 1 →
  seq[2]! = 2 :=
by sorry

end NUMINAMATH_CALUDE_third_number_is_two_l2825_282562


namespace NUMINAMATH_CALUDE_rotation_and_scaling_l2825_282586

def rotate90Clockwise (z : ℂ) : ℂ := -z.im + z.re * Complex.I

theorem rotation_and_scaling :
  let z : ℂ := 3 + 4 * Complex.I
  let rotated := rotate90Clockwise z
  let scaled := 2 * rotated
  scaled = -8 - 6 * Complex.I := by sorry

end NUMINAMATH_CALUDE_rotation_and_scaling_l2825_282586


namespace NUMINAMATH_CALUDE_triangle_problem_l2825_282515

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  (0 < a) ∧ (0 < b) ∧ (0 < c) →
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) →
  A + B + C = π →
  2 * Real.cos C * (a * Real.cos B + b * Real.cos A) = c →
  c = Real.sqrt 7 →
  1/2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 2 →
  C = π/3 ∧ a + b = 5 := by
sorry


end NUMINAMATH_CALUDE_triangle_problem_l2825_282515


namespace NUMINAMATH_CALUDE_curve_equation_and_no_fixed_point_l2825_282567

-- Define the circle C2
def C2 (x y : ℝ) : Prop := x^2 + (y-2)^2 = 1

-- Define the curve C1
def C1 (x y : ℝ) : Prop :=
  ∀ (x' y' : ℝ), C2 x' y' → (x - x')^2 + (y - y')^2 > 0 ∧
  (y + 1 = Real.sqrt ((x - x')^2 + (y - y')^2) - 1)

-- Define the point N
def N (b : ℝ) : ℝ × ℝ := (0, b)

-- Define the angle equality condition
def angle_equality (P Q : ℝ × ℝ) (N : ℝ × ℝ) : Prop :=
  (P.1 - N.1)^2 + (P.2 - N.2)^2 = (Q.1 - N.1)^2 + (Q.2 - N.2)^2

theorem curve_equation_and_no_fixed_point :
  (∀ x y : ℝ, C1 x y ↔ x^2 = 8*y) ∧
  (∀ b : ℝ, b < 0 →
    ∀ P Q : ℝ × ℝ, C1 P.1 P.2 → C1 Q.1 Q.2 → P ≠ Q →
    angle_equality P Q (N b) →
    ¬∃ F : ℝ × ℝ, ∀ P Q : ℝ × ℝ, C1 P.1 P.2 → C1 Q.1 Q.2 → P ≠ Q →
      angle_equality P Q (N b) → (Q.2 - P.2) * F.1 = (Q.1 - P.1) * F.2 + (P.1 * Q.2 - Q.1 * P.2)) :=
sorry

end NUMINAMATH_CALUDE_curve_equation_and_no_fixed_point_l2825_282567


namespace NUMINAMATH_CALUDE_not_solution_one_l2825_282510

theorem not_solution_one (x : ℂ) (h1 : x^2 + x + 1 = 0) (h2 : x ≠ 0) : x ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_not_solution_one_l2825_282510


namespace NUMINAMATH_CALUDE_todd_ate_eight_cupcakes_l2825_282530

/-- The number of cupcakes Todd ate -/
def cupcakes_eaten (initial : ℕ) (packages : ℕ) (per_package : ℕ) : ℕ :=
  initial - (packages * per_package)

/-- Proof that Todd ate 8 cupcakes -/
theorem todd_ate_eight_cupcakes :
  cupcakes_eaten 18 5 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_todd_ate_eight_cupcakes_l2825_282530


namespace NUMINAMATH_CALUDE_upstream_speed_is_26_l2825_282591

/-- The speed of a man rowing in a stream -/
structure RowingSpeed :=
  (stillWater : ℝ)
  (downstream : ℝ)

/-- Calculate the speed of the man rowing upstream -/
def upstreamSpeed (s : RowingSpeed) : ℝ :=
  2 * s.stillWater - s.downstream

/-- Theorem: The speed of the man rowing upstream is 26 kmph -/
theorem upstream_speed_is_26 (s : RowingSpeed)
  (h1 : s.stillWater = 28)
  (h2 : s.downstream = 30) :
  upstreamSpeed s = 26 := by
  sorry

end NUMINAMATH_CALUDE_upstream_speed_is_26_l2825_282591


namespace NUMINAMATH_CALUDE_largest_c_for_f_range_containing_2_l2825_282565

-- Define the function f
def f (c : ℝ) (x : ℝ) : ℝ := x^2 - 6*x + c

-- State the theorem
theorem largest_c_for_f_range_containing_2 :
  (∃ (c : ℝ), ∀ (d : ℝ), 
    (∃ (x : ℝ), f d x = 2) → d ≤ c) ∧
  (∃ (x : ℝ), f 11 x = 2) :=
sorry

end NUMINAMATH_CALUDE_largest_c_for_f_range_containing_2_l2825_282565


namespace NUMINAMATH_CALUDE_one_divides_six_digit_number_l2825_282536

/-- Represents a 6-digit number of the form abacab -/
def SixDigitNumber (a b c : ℕ) : ℕ :=
  100000 * a + 10000 * b + 1000 * a + 100 * c + 10 * a + b

/-- Theorem stating that 1 is a factor of any SixDigitNumber -/
theorem one_divides_six_digit_number (a b c : ℕ) (h1 : a ≠ 0) (h2 : a < 10) (h3 : b < 10) (h4 : c < 10) :
  1 ∣ SixDigitNumber a b c := by
  sorry


end NUMINAMATH_CALUDE_one_divides_six_digit_number_l2825_282536
