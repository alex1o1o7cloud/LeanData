import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_roots_range_l2987_298777

/-- A quadratic equation of the form kx^2 - 2x - 1 = 0 has two distinct real roots -/
def has_two_distinct_real_roots (k : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ k * x^2 - 2*x - 1 = 0 ∧ k * y^2 - 2*y - 1 = 0

/-- The range of k for which the quadratic equation has two distinct real roots -/
theorem quadratic_roots_range :
  ∀ k : ℝ, has_two_distinct_real_roots k ↔ k > -1 ∧ k ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l2987_298777


namespace NUMINAMATH_CALUDE_solve_for_y_l2987_298776

theorem solve_for_y (x y : ℝ) (h1 : x^2 + 2*x = y - 4) (h2 : x = -6) : y = 28 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l2987_298776


namespace NUMINAMATH_CALUDE_candle_weight_theorem_l2987_298786

/-- The weight of beeswax used in each candle, in ounces. -/
def beeswax_weight : ℕ := 8

/-- The weight of coconut oil used in each candle, in ounces. -/
def coconut_oil_weight : ℕ := 1

/-- The number of candles Ethan makes. -/
def num_candles : ℕ := 10 - 3

/-- The total weight of one candle, in ounces. -/
def candle_weight : ℕ := beeswax_weight + coconut_oil_weight

/-- The combined weight of all candles, in ounces. -/
def total_weight : ℕ := num_candles * candle_weight

theorem candle_weight_theorem : total_weight = 63 := by
  sorry

end NUMINAMATH_CALUDE_candle_weight_theorem_l2987_298786


namespace NUMINAMATH_CALUDE_raisin_count_proof_l2987_298729

/-- Given 5 boxes of raisins with a total of 437 raisins, where one box has 72 raisins,
    another has 74 raisins, and the remaining three boxes have an equal number of raisins,
    prove that each of these three boxes contains 97 raisins. -/
theorem raisin_count_proof (total_raisins : ℕ) (total_boxes : ℕ) 
  (box1_raisins : ℕ) (box2_raisins : ℕ) (other_boxes_raisins : ℕ) :
  total_raisins = 437 →
  total_boxes = 5 →
  box1_raisins = 72 →
  box2_raisins = 74 →
  total_raisins = box1_raisins + box2_raisins + 3 * other_boxes_raisins →
  other_boxes_raisins = 97 := by
  sorry

end NUMINAMATH_CALUDE_raisin_count_proof_l2987_298729


namespace NUMINAMATH_CALUDE_perpendicular_lines_parallel_l2987_298750

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the parallel relation between two lines
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_parallel (a b : Line) (α : Plane) :
  perpendicular a α → perpendicular b α → parallel a b :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_parallel_l2987_298750


namespace NUMINAMATH_CALUDE_items_per_crate_l2987_298774

theorem items_per_crate (novels : ℕ) (comics : ℕ) (documentaries : ℕ) (albums : ℕ) (crates : ℕ) :
  novels = 145 →
  comics = 271 →
  documentaries = 419 →
  albums = 209 →
  crates = 116 →
  (novels + comics + documentaries + albums) / crates = 9 := by
sorry

end NUMINAMATH_CALUDE_items_per_crate_l2987_298774


namespace NUMINAMATH_CALUDE_sqrt_three_irrational_neg_one_rational_two_rational_three_rational_l2987_298759

theorem sqrt_three_irrational :
  ∀ (x : ℝ), x ^ 2 = 3 → ¬ (∃ (a b : ℤ), b ≠ 0 ∧ x = a / b) :=
by sorry

theorem neg_one_rational : ∃ (a b : ℤ), b ≠ 0 ∧ -1 = a / b :=
by sorry

theorem two_rational : ∃ (a b : ℤ), b ≠ 0 ∧ 2 = a / b :=
by sorry

theorem three_rational : ∃ (a b : ℤ), b ≠ 0 ∧ 3 = a / b :=
by sorry

end NUMINAMATH_CALUDE_sqrt_three_irrational_neg_one_rational_two_rational_three_rational_l2987_298759


namespace NUMINAMATH_CALUDE_vertex_when_m_3_n_values_max_3_m_range_two_points_l2987_298731

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := -x^2 + 4*x + m - 1

-- Theorem 1: Vertex when m = 3
theorem vertex_when_m_3 :
  let m := 3
  ∃ (x y : ℝ), x = 2 ∧ y = 6 ∧ 
    ∀ (t : ℝ), f m t ≤ f m x :=
sorry

-- Theorem 2: Values of n when maximum is 3
theorem n_values_max_3 :
  let m := 3
  ∀ (n : ℝ), (∀ (x : ℝ), n ≤ x ∧ x ≤ n + 2 → f m x ≤ 3) ∧
             (∃ (x : ℝ), n ≤ x ∧ x ≤ n + 2 ∧ f m x = 3) →
    n = 2 + Real.sqrt 3 ∨ n = -Real.sqrt 3 :=
sorry

-- Theorem 3: Range of m for exactly two points 3 units from x-axis
theorem m_range_two_points :
  ∀ (m : ℝ), (∃! (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ (f m x₁ = 3 ∨ f m x₁ = -3) ∧ (f m x₂ = 3 ∨ f m x₂ = -3)) ↔
    -6 < m ∧ m < 0 :=
sorry

end NUMINAMATH_CALUDE_vertex_when_m_3_n_values_max_3_m_range_two_points_l2987_298731


namespace NUMINAMATH_CALUDE_vector_perpendicular_l2987_298791

/-- Given vectors a and b, prove that a - b is perpendicular to b -/
theorem vector_perpendicular (a b : ℝ × ℝ) (h1 : a = (1, 0)) (h2 : b = (1/2, 1/2)) :
  (a - b) • b = 0 := by
  sorry

end NUMINAMATH_CALUDE_vector_perpendicular_l2987_298791


namespace NUMINAMATH_CALUDE_total_sum_calculation_l2987_298772

theorem total_sum_calculation (maggie_share : ℚ) (total_sum : ℚ) : 
  maggie_share = 7500 → 
  maggie_share = (1/8 : ℚ) * total_sum → 
  total_sum = 60000 := by sorry

end NUMINAMATH_CALUDE_total_sum_calculation_l2987_298772


namespace NUMINAMATH_CALUDE_erased_grid_squares_l2987_298703

/-- Represents a square grid with erased line segments -/
structure ErasedSquareGrid :=
  (size : Nat)
  (erasedLines : Nat)

/-- Counts the number of squares of a given size in the grid -/
def countSquares (grid : ErasedSquareGrid) (squareSize : Nat) : Nat :=
  sorry

/-- Calculates the total number of squares of all sizes in the grid -/
def totalSquares (grid : ErasedSquareGrid) : Nat :=
  sorry

/-- The main theorem stating that a 4x4 grid with 2 erased lines has 22 squares -/
theorem erased_grid_squares :
  let grid : ErasedSquareGrid := ⟨4, 2⟩
  totalSquares grid = 22 :=
by sorry

end NUMINAMATH_CALUDE_erased_grid_squares_l2987_298703


namespace NUMINAMATH_CALUDE_consecutive_odd_divisibility_l2987_298773

theorem consecutive_odd_divisibility (m n : ℤ) : 
  (∃ k : ℤ, m = 2*k + 1 ∧ n = 2*k + 3) → 
  (∃ l : ℤ, 7*m^2 - 5*n^2 - 2 = 8*l) :=
sorry

end NUMINAMATH_CALUDE_consecutive_odd_divisibility_l2987_298773


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2987_298712

/-- An arithmetic sequence with given terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  a2_eq_9 : a 2 = 9
  a5_eq_33 : a 5 = 33

/-- The common difference of an arithmetic sequence is 8 given the conditions -/
theorem arithmetic_sequence_common_difference
  (seq : ArithmeticSequence) :
  ∃ d : ℝ, (∀ n, seq.a (n + 1) - seq.a n = d) ∧ d = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2987_298712


namespace NUMINAMATH_CALUDE_min_value_of_f_l2987_298700

/-- The function to be minimized -/
def f (x y : ℝ) : ℝ := 3 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 8 * y + 10

/-- The theorem stating the minimum value of the function -/
theorem min_value_of_f :
  ∃ (min : ℝ), min = 13/5 ∧ ∀ (x y : ℝ), f x y ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2987_298700


namespace NUMINAMATH_CALUDE_paint_calculation_l2987_298747

theorem paint_calculation (P : ℚ) : 
  (1/6 : ℚ) * P + (1/5 : ℚ) * (P - (1/6 : ℚ) * P) = 120 → P = 360 := by
sorry

end NUMINAMATH_CALUDE_paint_calculation_l2987_298747


namespace NUMINAMATH_CALUDE_complex_equation_unit_modulus_l2987_298752

theorem complex_equation_unit_modulus (z : ℂ) (h : 11 * z^10 + 10 * Complex.I * z^9 + 10 * Complex.I * z - 11 = 0) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_unit_modulus_l2987_298752


namespace NUMINAMATH_CALUDE_base_conversion_theorem_l2987_298716

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (a b c : ℕ) : ℕ := a * 7^2 + b * 7 + c

/-- Theorem: If 451 in base 7 equals xy in base 10 (where x and y are single digits),
    then (x * y) / 10 = 0.6 -/
theorem base_conversion_theorem (x y : ℕ) (h1 : x < 10) (h2 : y < 10) 
    (h3 : base7ToBase10 4 5 1 = 10 * x + y) : 
    (x * y : ℚ) / 10 = 6 / 10 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_theorem_l2987_298716


namespace NUMINAMATH_CALUDE_fraction_identity_l2987_298767

theorem fraction_identity (M N a b x : ℝ) (h1 : x ≠ a) (h2 : x ≠ b) (h3 : a ≠ b) :
  (M * x + N) / ((x - a) * (x - b)) = 
  (M * a + N) / (a - b) * (1 / (x - a)) - (M * b + N) / (a - b) * (1 / (x - b)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_identity_l2987_298767


namespace NUMINAMATH_CALUDE_quadratic_integer_solution_l2987_298711

theorem quadratic_integer_solution (a : ℤ) : 
  a < 0 → 
  (∃ x : ℤ, a * x^2 - 2*(a-3)*x + (a-2) = 0) ↔ 
  (a = -10 ∨ a = -4) :=
sorry

end NUMINAMATH_CALUDE_quadratic_integer_solution_l2987_298711


namespace NUMINAMATH_CALUDE_solve_equation_l2987_298764

theorem solve_equation (y : ℝ) : 
  Real.sqrt (2 + Real.sqrt (3 * y - 4)) = Real.sqrt 7 → y = 29 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2987_298764


namespace NUMINAMATH_CALUDE_probability_of_one_out_of_four_l2987_298748

theorem probability_of_one_out_of_four (S : Finset α) (h : S.card = 4) :
  ∀ a ∈ S, (1 : ℝ) / S.card = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_one_out_of_four_l2987_298748


namespace NUMINAMATH_CALUDE_salary_restoration_l2987_298720

theorem salary_restoration (original_salary : ℝ) (original_salary_positive : original_salary > 0) : 
  let reduced_salary := original_salary * (1 - 0.2)
  reduced_salary * (1 + 0.25) = original_salary := by
sorry

end NUMINAMATH_CALUDE_salary_restoration_l2987_298720


namespace NUMINAMATH_CALUDE_magpie_call_not_correlation_l2987_298794

-- Define a type for statements
inductive Statement
| HeavySnow : Statement
| GreatTeachers : Statement
| Smoking : Statement
| MagpieCall : Statement

-- Define a predicate for correlation
def IsCorrelation (s : Statement) : Prop :=
  match s with
  | Statement.HeavySnow => True
  | Statement.GreatTeachers => True
  | Statement.Smoking => True
  | Statement.MagpieCall => False

-- Theorem statement
theorem magpie_call_not_correlation :
  ∀ s : Statement, 
    (s = Statement.HeavySnow ∨ s = Statement.GreatTeachers ∨ s = Statement.Smoking → IsCorrelation s) ∧
    (s = Statement.MagpieCall → ¬IsCorrelation s) :=
by sorry

end NUMINAMATH_CALUDE_magpie_call_not_correlation_l2987_298794


namespace NUMINAMATH_CALUDE_middle_number_proof_l2987_298710

theorem middle_number_proof (x y z : ℕ) 
  (h1 : x < y) (h2 : y < z)
  (h3 : x + y = 14) (h4 : x + z = 20) (h5 : y + z = 22)
  (h6 : x + y + z = 27) : y = 7 := by
  sorry

end NUMINAMATH_CALUDE_middle_number_proof_l2987_298710


namespace NUMINAMATH_CALUDE_jackie_free_time_l2987_298760

/-- Calculates the free time given the time spent on various activities and the total time available. -/
def free_time (work_hours exercise_hours sleep_hours total_hours : ℕ) : ℕ :=
  total_hours - (work_hours + exercise_hours + sleep_hours)

/-- Proves that Jackie has 5 hours of free time given her daily schedule. -/
theorem jackie_free_time :
  let work_hours : ℕ := 8
  let exercise_hours : ℕ := 3
  let sleep_hours : ℕ := 8
  let total_hours : ℕ := 24
  free_time work_hours exercise_hours sleep_hours total_hours = 5 := by
  sorry

end NUMINAMATH_CALUDE_jackie_free_time_l2987_298760


namespace NUMINAMATH_CALUDE_average_of_first_12_even_numbers_l2987_298769

def first_12_even_numbers : List ℤ :=
  [-12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]

theorem average_of_first_12_even_numbers :
  (List.sum first_12_even_numbers) / (List.length first_12_even_numbers) = -1 := by
sorry

end NUMINAMATH_CALUDE_average_of_first_12_even_numbers_l2987_298769


namespace NUMINAMATH_CALUDE_car_dealership_problem_l2987_298755

theorem car_dealership_problem (initial_cars : ℕ) (initial_silver_percent : ℚ)
  (new_cars : ℕ) (total_silver_percent : ℚ)
  (h1 : initial_cars = 40)
  (h2 : initial_silver_percent = 15 / 100)
  (h3 : new_cars = 80)
  (h4 : total_silver_percent = 25 / 100) :
  (new_cars - (total_silver_percent * (initial_cars + new_cars) - initial_silver_percent * initial_cars)) / new_cars = 70 / 100 := by
  sorry

end NUMINAMATH_CALUDE_car_dealership_problem_l2987_298755


namespace NUMINAMATH_CALUDE_friendly_point_properties_l2987_298744

def is_friendly_point (x y : ℝ) : Prop :=
  ∃ m n : ℝ, m - n = 6 ∧ m - 1 = x ∧ 3*n + 1 = y

theorem friendly_point_properties :
  (¬ is_friendly_point 7 1) ∧ 
  (is_friendly_point 6 4) ∧
  (∀ x y t : ℝ, x + y = 2 → 2*x - y = t → is_friendly_point x y → t = 10) := by
  sorry

end NUMINAMATH_CALUDE_friendly_point_properties_l2987_298744


namespace NUMINAMATH_CALUDE_soft_drink_bottles_l2987_298799

theorem soft_drink_bottles (small_bottles : ℕ) : 
  (10000 : ℕ) * (85 : ℕ) / 100 + small_bottles * (88 : ℕ) / 100 = (13780 : ℕ) →
  small_bottles = (6000 : ℕ) :=
by sorry

end NUMINAMATH_CALUDE_soft_drink_bottles_l2987_298799


namespace NUMINAMATH_CALUDE_complement_of_A_l2987_298746

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x - 2 > 0}

theorem complement_of_A : Set.compl A = {x : ℝ | x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l2987_298746


namespace NUMINAMATH_CALUDE_equation_satisfied_at_nine_l2987_298756

/-- The sum of an infinite geometric series with first term a and common ratio r. -/
noncomputable def geometricSum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

/-- Left-hand side of the equation -/
noncomputable def leftHandSide : ℝ := 
  (geometricSum 1 (1/3)) * (geometricSum 1 (-1/3))

/-- Right-hand side of the equation -/
noncomputable def rightHandSide (y : ℝ) : ℝ := 
  geometricSum 1 (1/y)

/-- The theorem stating that the equation is satisfied when y = 9 -/
theorem equation_satisfied_at_nine : 
  leftHandSide = rightHandSide 9 := by sorry

end NUMINAMATH_CALUDE_equation_satisfied_at_nine_l2987_298756


namespace NUMINAMATH_CALUDE_four_digit_number_proof_l2987_298722

def is_valid_number (n : ℕ) : Prop :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  (n = 10 * 23) ∧
  (a + b + c + d = 26) ∧
  ((b * d) / 10 % 10 = a + c) ∧
  (∃ m : ℕ, b * d - c^2 = 2^m)

theorem four_digit_number_proof :
  is_valid_number 1979 :=
by sorry

end NUMINAMATH_CALUDE_four_digit_number_proof_l2987_298722


namespace NUMINAMATH_CALUDE_correct_calculation_l2987_298745

theorem correct_calculation (x : ℤ) (h : x + 44 - 39 = 63) : (x + 39) - 44 = 53 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2987_298745


namespace NUMINAMATH_CALUDE_min_a_for_inequality_l2987_298702

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (x^3 - 3*x + 3) - a * Real.exp x - x

theorem min_a_for_inequality :
  ∃ (a : ℝ), a = 1 - 1 / Real.exp 1 ∧
  (∀ (x : ℝ), x ≥ -2 → f a x ≤ 0) ∧
  (∀ (b : ℝ), (∀ (x : ℝ), x ≥ -2 → f b x ≤ 0) → b ≥ a) :=
sorry

end NUMINAMATH_CALUDE_min_a_for_inequality_l2987_298702


namespace NUMINAMATH_CALUDE_candy_ratio_l2987_298708

theorem candy_ratio (cherry : ℕ) (grape : ℕ) (apple : ℕ) (total_cost : ℚ) :
  grape = 3 * cherry →
  apple = 2 * grape →
  total_cost = 200 →
  (cherry + grape + apple) * (5/2) = total_cost →
  grape / cherry = 3 := by
sorry

end NUMINAMATH_CALUDE_candy_ratio_l2987_298708


namespace NUMINAMATH_CALUDE_hilary_pakora_orders_l2987_298796

/-- Represents the cost of a meal at Delicious Delhi restaurant -/
structure MealCost where
  samosas : ℕ
  pakoras : ℕ
  lassi : ℕ
  tip_percent : ℚ
  total_with_tax : ℚ

/-- Calculates the number of pakora orders given the meal cost details -/
def calculate_pakora_orders (meal : MealCost) : ℚ :=
  let samosa_cost := 2 * meal.samosas
  let lassi_cost := 2 * meal.lassi
  let pakora_cost := 3 * meal.pakoras
  let subtotal := samosa_cost + lassi_cost + pakora_cost
  let total_with_tip := subtotal * (1 + meal.tip_percent)
  (meal.total_with_tax - total_with_tip) / 3

/-- Theorem stating that Hilary bought 4 orders of pakoras -/
theorem hilary_pakora_orders :
  let meal := MealCost.mk 3 4 1 (1/4) 25
  calculate_pakora_orders meal = 4 := by
  sorry

end NUMINAMATH_CALUDE_hilary_pakora_orders_l2987_298796


namespace NUMINAMATH_CALUDE_integer_points_count_l2987_298749

/-- Represents a line segment on a number line -/
structure LineSegment where
  start : ℝ
  length : ℝ

/-- Counts the number of integer points covered by a line segment -/
def count_integer_points (segment : LineSegment) : ℕ :=
  sorry

/-- Theorem stating that a line segment of length 2020 covers either 2020 or 2021 integer points -/
theorem integer_points_count (segment : LineSegment) :
  segment.length = 2020 → count_integer_points segment = 2020 ∨ count_integer_points segment = 2021 :=
sorry

end NUMINAMATH_CALUDE_integer_points_count_l2987_298749


namespace NUMINAMATH_CALUDE_equation_proof_l2987_298740

theorem equation_proof : 529 + 2 * 23 * 3 + 9 = 676 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l2987_298740


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l2987_298709

/-- Given a right triangle ABC with angle C = 90°, BC = 6, and tan B = 0.75, prove that AC = 4.5 -/
theorem right_triangle_side_length (A B C : ℝ × ℝ) : 
  let triangle := (A, B, C)
  (∃ (AC BC : ℝ), 
    -- ABC is a right triangle with angle C = 90°
    (C.2 - A.2) * (B.1 - A.1) = (C.1 - A.1) * (B.2 - A.2) ∧
    -- BC = 6
    Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 6 ∧
    -- tan B = 0.75
    (C.2 - B.2) / (C.1 - B.1) = 0.75 ∧
    -- AC is the length we're solving for
    AC = Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)) →
  AC = 4.5 := by
sorry


end NUMINAMATH_CALUDE_right_triangle_side_length_l2987_298709


namespace NUMINAMATH_CALUDE_chocolate_gain_percent_l2987_298782

/-- Given that the cost price of 121 chocolates equals the selling price of 77 chocolates,
    the gain percent is (4400 / 77)%. -/
theorem chocolate_gain_percent :
  ∀ (cost_price selling_price : ℝ),
  cost_price > 0 →
  selling_price > 0 →
  121 * cost_price = 77 * selling_price →
  (selling_price - cost_price) / cost_price * 100 = 4400 / 77 := by
sorry

end NUMINAMATH_CALUDE_chocolate_gain_percent_l2987_298782


namespace NUMINAMATH_CALUDE_two_week_riding_hours_l2987_298738

/-- Represents the number of hours Bethany rides on a given day -/
def daily_riding_hours (day : Nat) : Real :=
  match day % 7 with
  | 1 | 3 | 5 => 1    -- Monday, Wednesday, Friday
  | 2 | 4 => 0.5      -- Tuesday, Thursday
  | 6 => 2            -- Saturday
  | _ => 0            -- Sunday

/-- Calculates the total riding hours over a given number of days -/
def total_riding_hours (days : Nat) : Real :=
  (List.range days).map daily_riding_hours |>.sum

/-- Proves that Bethany rides for 12 hours over a 2-week period -/
theorem two_week_riding_hours :
  total_riding_hours 14 = 12 := by
  sorry

end NUMINAMATH_CALUDE_two_week_riding_hours_l2987_298738


namespace NUMINAMATH_CALUDE_inequality_problem_l2987_298706

theorem inequality_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 4/b = 1) :
  (ab ≥ 16) ∧ 
  (2*a + b ≥ 6 + 4*Real.sqrt 2) ∧ 
  (1/a^2 + 16/b^2 ≥ 1/2) ∧
  ¬(∀ (a b : ℝ), a > 0 → b > 0 → 1/a + 4/b = 1 → a - b < 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_problem_l2987_298706


namespace NUMINAMATH_CALUDE_yellow_jelly_bean_probability_l2987_298704

theorem yellow_jelly_bean_probability :
  let red : ℕ := 4
  let green : ℕ := 8
  let yellow : ℕ := 9
  let blue : ℕ := 5
  let total : ℕ := red + green + yellow + blue
  (yellow : ℚ) / total = 9 / 26 := by sorry

end NUMINAMATH_CALUDE_yellow_jelly_bean_probability_l2987_298704


namespace NUMINAMATH_CALUDE_concert_attendance_l2987_298743

/-- The number of students from School A who went to the concert -/
def school_a_students : ℕ := 15 * 30

/-- The number of students from School B who went to the concert -/
def school_b_students : ℕ := 18 * 7 + 5 * 6

/-- The number of students from School C who went to the concert -/
def school_c_students : ℕ := 13 * 33 + 10 * 4

/-- The total number of students who went to the concert -/
def total_students : ℕ := school_a_students + school_b_students + school_c_students

theorem concert_attendance : total_students = 1075 := by
  sorry

end NUMINAMATH_CALUDE_concert_attendance_l2987_298743


namespace NUMINAMATH_CALUDE_triangle_properties_l2987_298792

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem -/
theorem triangle_properties (t : Triangle) : 
  (t.a * Real.cos t.C + (t.c - 3 * t.b) * Real.cos t.A = 0) → 
  (Real.cos t.A = 1 / 3) ∧
  (Real.sqrt 2 = 1 / 2 * t.b * t.c * Real.sin t.A) →
  (t.b - t.c = 2) →
  (t.a = 2 * Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l2987_298792


namespace NUMINAMATH_CALUDE_cos_nineteen_pi_fourths_l2987_298779

theorem cos_nineteen_pi_fourths : Real.cos (19 * π / 4) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_nineteen_pi_fourths_l2987_298779


namespace NUMINAMATH_CALUDE_unique_positive_integer_solution_l2987_298733

theorem unique_positive_integer_solution : 
  ∃! (x : ℕ), x > 0 ∧ 12 * x = x^2 + 36 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_positive_integer_solution_l2987_298733


namespace NUMINAMATH_CALUDE_book_cost_theorem_l2987_298705

/-- Calculates the cost of a single book given the total budget, remaining money, number of series bought, books per series, and tax rate. -/
def calculate_book_cost (total_budget : ℚ) (remaining_money : ℚ) (series_bought : ℕ) (books_per_series : ℕ) (tax_rate : ℚ) : ℚ :=
  let total_spent := total_budget - remaining_money
  let books_bought := series_bought * books_per_series
  let pre_tax_total := total_spent / (1 + tax_rate)
  let pre_tax_per_book := pre_tax_total / books_bought
  pre_tax_per_book * (1 + tax_rate)

/-- The cost of each book is approximately $5.96 given the problem conditions. -/
theorem book_cost_theorem :
  let total_budget : ℚ := 200
  let remaining_money : ℚ := 56
  let series_bought : ℕ := 3
  let books_per_series : ℕ := 8
  let tax_rate : ℚ := 1/10
  abs (calculate_book_cost total_budget remaining_money series_bought books_per_series tax_rate - 596/100) < 1/100 := by
  sorry


end NUMINAMATH_CALUDE_book_cost_theorem_l2987_298705


namespace NUMINAMATH_CALUDE_root_in_interval_l2987_298793

def f (x : ℝ) := x^3 + x - 8

theorem root_in_interval :
  f 1 < 0 →
  f 1.5 < 0 →
  f 1.75 < 0 →
  f 2 > 0 →
  ∃ x, x ∈ Set.Ioo 1.75 2 ∧ f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_root_in_interval_l2987_298793


namespace NUMINAMATH_CALUDE_composite_has_at_least_three_divisors_l2987_298788

-- Define what it means for a number to be composite
def IsComposite (n : ℕ) : Prop :=
  ∃ k : ℕ, 1 < k ∧ k < n ∧ k ∣ n

-- Define the number of divisors function
def NumDivisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

-- Theorem statement
theorem composite_has_at_least_three_divisors (n : ℕ) (h : IsComposite n) :
  NumDivisors n ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_composite_has_at_least_three_divisors_l2987_298788


namespace NUMINAMATH_CALUDE_divisibility_by_24_l2987_298724

theorem divisibility_by_24 (n : ℕ) (h_odd : Odd n) (h_not_div_3 : ¬3 ∣ n) :
  24 ∣ (n^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_24_l2987_298724


namespace NUMINAMATH_CALUDE_equality_condition_l2987_298785

theorem equality_condition (a b c d : ℝ) :
  a + b * c * d = (a + b) * (a + c) * (a + d) ↔ a^2 + a * (b + c + d) + b * c + b * d + c * d = 1 :=
sorry

end NUMINAMATH_CALUDE_equality_condition_l2987_298785


namespace NUMINAMATH_CALUDE_angle_BAD_measure_l2987_298707

-- Define the triangle ABC
structure Triangle (A B C : ℝ × ℝ) : Prop where
  -- No specific conditions needed for a general triangle

-- Define an isosceles triangle
def IsIsosceles (t : Triangle A B C) : Prop :=
  dist A B = dist A C

-- Define the angle BAD
def AngleBAD (A B D : ℝ × ℝ) : ℝ := sorry

-- Define the angle DAC
def AngleDAC (A C D : ℝ × ℝ) : ℝ := sorry

-- Define the distance between two points
def dist (p q : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem angle_BAD_measure 
  (A B C D : ℝ × ℝ) 
  (t1 : Triangle A B C) 
  (t2 : Triangle A B D) :
  IsIsosceles t1 →
  IsIsosceles t2 →
  AngleDAC A C D = 39 →
  AngleBAD A B D = 70.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_BAD_measure_l2987_298707


namespace NUMINAMATH_CALUDE_min_value_problem_l2987_298763

theorem min_value_problem (m n : ℝ) (hm : m > 0) (hn : n > 0) (heq : 2*m + n = 1) :
  (1/m) + (2/n) ≥ 8 ∧ ((1/m) + (2/n) = 8 ↔ n = 2*m ∧ n = 1/2) :=
sorry

end NUMINAMATH_CALUDE_min_value_problem_l2987_298763


namespace NUMINAMATH_CALUDE_polynomial_roots_unit_circle_l2987_298798

theorem polynomial_roots_unit_circle (a b c : ℂ) :
  (∀ w : ℂ, w^3 + Complex.abs a * w^2 + Complex.abs b * w + Complex.abs c = 0 → Complex.abs w = 1) →
  (Complex.abs c = 1 ∧ 
   ∀ x : ℂ, x^3 + Complex.abs a * x^2 + Complex.abs b * x + Complex.abs c = 0 ↔ 
            x^3 + Complex.abs a * x^2 + Complex.abs a * x + 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_unit_circle_l2987_298798


namespace NUMINAMATH_CALUDE_dispersion_measures_l2987_298732

-- Define a sample as a list of real numbers
def Sample := List Real

-- Define the concept of a statistic as a function from a sample to a real number
def Statistic := Sample → Real

-- Define the concept of measuring dispersion
def MeasuresDispersion (s : Statistic) : Prop := sorry

-- Define standard deviation
def StandardDeviation : Statistic := sorry

-- Define median
def Median : Statistic := sorry

-- Define range
def Range : Statistic := sorry

-- Define mean
def Mean : Statistic := sorry

-- Theorem stating that only standard deviation and range measure dispersion
theorem dispersion_measures (sample : Sample) :
  MeasuresDispersion StandardDeviation ∧
  MeasuresDispersion Range ∧
  ¬MeasuresDispersion Median ∧
  ¬MeasuresDispersion Mean :=
sorry

end NUMINAMATH_CALUDE_dispersion_measures_l2987_298732


namespace NUMINAMATH_CALUDE_problem_solution_l2987_298780

theorem problem_solution : ∃ y : ℕ, (8000 * 6000 : ℕ) = 480 * (10 ^ y) ∧ y = 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2987_298780


namespace NUMINAMATH_CALUDE_children_doing_both_A_and_B_l2987_298751

/-- The number of children who can do both A and B -/
def X : ℕ := 19

/-- The total number of children -/
def total : ℕ := 48

/-- The number of children who can do A -/
def A : ℕ := 38

/-- The number of children who can do B -/
def B : ℕ := 29

theorem children_doing_both_A_and_B :
  X = A + B - total :=
by sorry

end NUMINAMATH_CALUDE_children_doing_both_A_and_B_l2987_298751


namespace NUMINAMATH_CALUDE_circle_and_line_theorem_l2987_298717

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line y = x
def line_y_eq_x (x y : ℝ) : Prop := y = x

-- Define the line y = -x + 2
def line_intercept (x y : ℝ) : Prop := y = -x + 2

-- Define the line l
def line_l (m : ℝ) (x y : ℝ) : Prop := x = m * y + 3/2

-- Define the dot product of two vectors
def dot_product (x1 y1 x2 y2 : ℝ) : ℝ := x1 * x2 + y1 * y2

theorem circle_and_line_theorem :
  -- Circle C passes through (1, √3)
  circle_C 1 (Real.sqrt 3) →
  -- The center of C is on the line y = x
  ∃ a : ℝ, line_y_eq_x a a ∧ ∀ x y : ℝ, circle_C x y ↔ (x - a)^2 + (y - a)^2 = 4 →
  -- The chord intercepted by y = -x + 2 has length 2√2
  ∃ x1 y1 x2 y2 : ℝ, 
    line_intercept x1 y1 ∧ line_intercept x2 y2 ∧ 
    circle_C x1 y1 ∧ circle_C x2 y2 ∧
    (x2 - x1)^2 + (y2 - y1)^2 = 8 →
  -- Line l passes through (3/2, 0) and intersects C at P and Q
  ∃ m xP yP xQ yQ : ℝ,
    line_l m (3/2) 0 ∧ 
    circle_C xP yP ∧ circle_C xQ yQ ∧
    line_l m xP yP ∧ line_l m xQ yQ ∧
    -- OP · OQ = -2
    dot_product xP yP xQ yQ = -2 →
  -- Conclusion 1: Equation of circle C
  (∀ x y : ℝ, circle_C x y ↔ x^2 + y^2 = 4) ∧
  -- Conclusion 2: Equation of line l
  (m = Real.sqrt 5 / 2 ∨ m = -Real.sqrt 5 / 2) ∧
  (∀ x y : ℝ, line_l m x y ↔ 2*x + m*y - 3 = 0) := by
sorry

end NUMINAMATH_CALUDE_circle_and_line_theorem_l2987_298717


namespace NUMINAMATH_CALUDE_triangle_theorem_l2987_298728

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.c * Real.sin (t.A - t.B) = t.b * Real.sin (t.C - t.A)) :
  (t.a ^ 2 = t.b * t.c → t.A = π / 3) ∧
  (t.a = 2 ∧ Real.cos t.A = 4 / 5 → t.a + t.b + t.c = 2 + Real.sqrt 13) :=
by sorry

end NUMINAMATH_CALUDE_triangle_theorem_l2987_298728


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2987_298727

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if the angle between its two asymptotes is 60°, then its eccentricity e
    is either 2 or 2√3/3 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let asymptote_angle := Real.pi / 3
  let eccentricity := Real.sqrt (1 + b^2 / a^2)
  asymptote_angle = Real.arctan (b / a) * 2 →
  eccentricity = 2 ∨ eccentricity = 2 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2987_298727


namespace NUMINAMATH_CALUDE_variables_positively_correlated_l2987_298784

/-- Represents a simple linear regression model -/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ

/-- Defines positive correlation between variables in a linear regression model -/
def positively_correlated (model : LinearRegression) : Prop :=
  model.slope > 0

/-- The specific linear regression model given in the problem -/
def given_model : LinearRegression :=
  { slope := 0.5, intercept := 2 }

/-- Theorem stating that the variables in the given model are positively correlated -/
theorem variables_positively_correlated : 
  positively_correlated given_model := by sorry

end NUMINAMATH_CALUDE_variables_positively_correlated_l2987_298784


namespace NUMINAMATH_CALUDE_negation_of_conditional_l2987_298757

theorem negation_of_conditional (x y : ℝ) :
  ¬(((x - 1) * (y + 2) = 0) → (x = 1 ∨ y = -2)) ↔
  (((x - 1) * (y + 2) ≠ 0) → (x ≠ 1 ∧ y ≠ -2)) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_conditional_l2987_298757


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l2987_298775

/-- A geometric sequence is a sequence where the ratio between any two consecutive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The theorem states that in a geometric sequence where the product of the first and fifth terms is 16,
    the third term is either 4 or -4. -/
theorem geometric_sequence_third_term
  (a : ℕ → ℝ)
  (h_geo : IsGeometricSequence a)
  (h_prod : a 1 * a 5 = 16) :
  a 3 = 4 ∨ a 3 = -4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l2987_298775


namespace NUMINAMATH_CALUDE_expression_simplification_l2987_298754

theorem expression_simplification (x : ℝ) (h : x^2 + 2*x - 6 = 0) :
  ((x - 1) / (x - 3) - (x + 1) / x) / ((x^2 + 3*x) / (x^2 - 6*x + 9)) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2987_298754


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l2987_298701

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  (∃ s : ℝ, s = x + y ∧ f x = 0 ∧ f y = 0 → s = -b / a) :=
sorry

theorem sum_of_roots_specific_quadratic :
  let f : ℝ → ℝ := λ x => x^2 - 7*x + 2 - 16
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  (∃ s : ℝ, s = x + y ∧ f x = 0 ∧ f y = 0 → s = 7) :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l2987_298701


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2987_298735

theorem imaginary_part_of_z (z : ℂ) (h : 1 + 2*I = I * z) : z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2987_298735


namespace NUMINAMATH_CALUDE_sqrt_inequality_l2987_298741

theorem sqrt_inequality : Real.sqrt 7 - 1 > Real.sqrt 11 - Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l2987_298741


namespace NUMINAMATH_CALUDE_women_in_room_l2987_298715

theorem women_in_room (initial_men : ℕ) (initial_women : ℕ) : 
  (initial_men : ℚ) / initial_women = 7 / 9 →
  initial_men + 8 = 28 →
  2 * (initial_women - 2) = 48 :=
by sorry

end NUMINAMATH_CALUDE_women_in_room_l2987_298715


namespace NUMINAMATH_CALUDE_tourist_money_theorem_l2987_298758

/-- Represents the amount of money a tourist has at the end of each day -/
def money_after_day (initial_money : ℚ) (day : ℕ) : ℚ :=
  match day with
  | 0 => initial_money
  | n + 1 => (money_after_day initial_money n) / 2 - 100

/-- Theorem stating that if a tourist spends half their money plus 100 Ft each day for 5 days
    and ends up with no money, they must have started with 6200 Ft -/
theorem tourist_money_theorem :
  ∃ (initial_money : ℚ), 
    (money_after_day initial_money 5 = 0) ∧ 
    (initial_money = 6200) :=
by sorry

end NUMINAMATH_CALUDE_tourist_money_theorem_l2987_298758


namespace NUMINAMATH_CALUDE_parallelogram_bisector_intersection_inside_l2987_298719

/-- A parallelogram is a quadrilateral with opposite sides parallel. -/
structure Parallelogram where
  vertices : Fin 4 → ℝ × ℝ
  is_parallelogram : sorry

/-- An angle bisector of a parallelogram is a line that bisects one of its angles. -/
def angle_bisector (p : Parallelogram) (i : Fin 4) : Set (ℝ × ℝ) := sorry

/-- The pairwise intersections of angle bisectors of a parallelogram. -/
def bisector_intersections (p : Parallelogram) : Set (ℝ × ℝ) := sorry

/-- A point is inside a parallelogram if it's in the interior of the parallelogram. -/
def inside_parallelogram (p : Parallelogram) (point : ℝ × ℝ) : Prop := sorry

theorem parallelogram_bisector_intersection_inside 
  (p : Parallelogram) : 
  ∃ (point : ℝ × ℝ), point ∈ bisector_intersections p ∧ inside_parallelogram p point :=
sorry

end NUMINAMATH_CALUDE_parallelogram_bisector_intersection_inside_l2987_298719


namespace NUMINAMATH_CALUDE_abc_sum_l2987_298713

theorem abc_sum (A B C : ℕ+) (h1 : Nat.gcd A.val (Nat.gcd B.val C.val) = 1)
  (h2 : (A : ℝ) * Real.log 5 / Real.log 500 + (B : ℝ) * Real.log 2 / Real.log 500 = C) :
  A + B + C = 6 := by
sorry

end NUMINAMATH_CALUDE_abc_sum_l2987_298713


namespace NUMINAMATH_CALUDE_water_added_to_container_l2987_298730

theorem water_added_to_container (capacity : ℝ) (initial_percentage : ℝ) (final_fraction : ℝ) :
  capacity = 120 →
  initial_percentage = 0.35 →
  final_fraction = 3/4 →
  (final_fraction * capacity) - (initial_percentage * capacity) = 48 :=
by sorry

end NUMINAMATH_CALUDE_water_added_to_container_l2987_298730


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2987_298714

theorem right_triangle_hypotenuse (a b c : ℝ) :
  a = 12 ∧ b = 16 ∧ c^2 = a^2 + b^2 → c = 20 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2987_298714


namespace NUMINAMATH_CALUDE_students_in_both_band_and_chorus_l2987_298781

/-- Calculates the number of students in both band and chorus -/
def students_in_both (total : ℕ) (band : ℕ) (chorus : ℕ) (band_or_chorus : ℕ) : ℕ :=
  band + chorus - band_or_chorus

/-- Proves that the number of students in both band and chorus is 50 -/
theorem students_in_both_band_and_chorus :
  students_in_both 300 120 180 250 = 50 := by
  sorry

end NUMINAMATH_CALUDE_students_in_both_band_and_chorus_l2987_298781


namespace NUMINAMATH_CALUDE_trig_problem_l2987_298768

theorem trig_problem (α : Real) 
  (h1 : 0 < α) (h2 : α < Real.pi / 2) (h3 : Real.sin α = 4 / 5) : 
  (Real.sin α ^ 2 + Real.sin (2 * α)) / (Real.cos α ^ 2 + Real.cos (2 * α)) = 20 ∧ 
  Real.tan (α - 5 * Real.pi / 4) = 1 / 7 := by sorry

end NUMINAMATH_CALUDE_trig_problem_l2987_298768


namespace NUMINAMATH_CALUDE_tangent_line_minimum_value_l2987_298797

/-- The curve function -/
def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 2

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 3*x^2 - 4*x

/-- Point A -/
def A : ℝ × ℝ := (2, 2)

/-- The line on which point A lies -/
def line (m n l : ℝ) (x y : ℝ) : Prop := m*x + n*y = l

theorem tangent_line_minimum_value (m n l : ℝ) (hm : m > 0) (hn : n > 0) :
  line m n l A.1 A.2 →
  f' A.1 = 4 →
  ∀ k₁ k₂ : ℝ, k₁ > 0 → k₂ > 0 → line k₁ k₂ l A.1 A.2 → 
  1/k₁ + 2/k₂ ≥ 6 + 4*Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_minimum_value_l2987_298797


namespace NUMINAMATH_CALUDE_stock_worth_equation_l2987_298753

/-- Proves that the total worth of stock satisfies the given equation based on the problem conditions --/
theorem stock_worth_equation (W : ℝ) 
  (h1 : 0.25 * W * 0.15 - 0.40 * W * 0.05 + 0.35 * W * 0.10 = 750) : 
  0.0525 * W = 750 := by
  sorry

end NUMINAMATH_CALUDE_stock_worth_equation_l2987_298753


namespace NUMINAMATH_CALUDE_point_outside_region_implies_a_range_l2987_298765

theorem point_outside_region_implies_a_range (a : ℝ) : 
  (2 - (4 * a^2 + 3 * a - 2) * 2 - 4 ≥ 0) → 
  (a ∈ Set.Icc (-1 : ℝ) (1/4 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_point_outside_region_implies_a_range_l2987_298765


namespace NUMINAMATH_CALUDE_picture_difference_is_eight_l2987_298789

/-- The number of pictures Ralph has -/
def ralph_pictures : ℕ := 26

/-- The number of pictures Derrick has -/
def derrick_pictures : ℕ := 34

/-- The difference in the number of pictures between Derrick and Ralph -/
def picture_difference : ℕ := derrick_pictures - ralph_pictures

theorem picture_difference_is_eight : picture_difference = 8 := by
  sorry

end NUMINAMATH_CALUDE_picture_difference_is_eight_l2987_298789


namespace NUMINAMATH_CALUDE_bella_apple_consumption_l2987_298790

/-- The fraction of apples Bella consumes from what Grace picks -/
def bella_fraction : ℚ := 1 / 18

/-- The number of apples Bella eats per day -/
def bella_daily_apples : ℕ := 6

/-- The number of apples Grace has left after 6 weeks -/
def grace_remaining_apples : ℕ := 504

/-- The number of weeks in the problem -/
def weeks : ℕ := 6

/-- The number of days in a week -/
def days_per_week : ℕ := 7

theorem bella_apple_consumption :
  bella_fraction = 1 / 18 :=
sorry

end NUMINAMATH_CALUDE_bella_apple_consumption_l2987_298790


namespace NUMINAMATH_CALUDE_percentage_six_years_or_more_l2987_298721

def employee_distribution (x : ℕ) : List ℕ :=
  [4*x, 7*x, 5*x, 4*x, 3*x, 3*x, 2*x, 2*x, 2*x, 2*x]

def total_employees (x : ℕ) : ℕ :=
  List.sum (employee_distribution x)

def employees_six_years_or_more (x : ℕ) : ℕ :=
  List.sum (List.drop 6 (employee_distribution x))

theorem percentage_six_years_or_more (x : ℕ) :
  (employees_six_years_or_more x : ℚ) / (total_employees x : ℚ) * 100 = 2222 / 100 :=
sorry

end NUMINAMATH_CALUDE_percentage_six_years_or_more_l2987_298721


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2987_298736

theorem arithmetic_calculation : 
  let a := 65 * ((13/3 + 7/2) / (11/5 - 5/3))
  ∃ (n : ℕ) (m : ℚ), 0 ≤ m ∧ m < 1 ∧ a = n + m ∧ n = 954 ∧ m = 33/48 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2987_298736


namespace NUMINAMATH_CALUDE_trig_simplification_l2987_298771

theorem trig_simplification (α : ℝ) :
  Real.cos (π / 3 + α) + Real.sin (π / 6 + α) = Real.cos α := by sorry

end NUMINAMATH_CALUDE_trig_simplification_l2987_298771


namespace NUMINAMATH_CALUDE_mixture_weight_theorem_l2987_298718

/-- Atomic weight of Aluminum in g/mol -/
def Al_weight : ℝ := 26.98

/-- Atomic weight of Phosphorus in g/mol -/
def P_weight : ℝ := 30.97

/-- Atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 16.00

/-- Atomic weight of Sodium in g/mol -/
def Na_weight : ℝ := 22.99

/-- Atomic weight of Sulfur in g/mol -/
def S_weight : ℝ := 32.07

/-- Molecular weight of Aluminum phosphate (AlPO4) in g/mol -/
def AlPO4_weight : ℝ := Al_weight + P_weight + 4 * O_weight

/-- Molecular weight of Sodium sulfate (Na2SO4) in g/mol -/
def Na2SO4_weight : ℝ := 2 * Na_weight + S_weight + 4 * O_weight

/-- Total weight of the mixture in grams -/
def total_mixture_weight : ℝ := 5 * AlPO4_weight + 3 * Na2SO4_weight

theorem mixture_weight_theorem :
  total_mixture_weight = 1035.90 := by sorry

end NUMINAMATH_CALUDE_mixture_weight_theorem_l2987_298718


namespace NUMINAMATH_CALUDE_simplify_fraction_l2987_298734

theorem simplify_fraction (a b : ℝ) (h1 : a ≠ -b) (h2 : a ≠ 2*b) (h3 : a ≠ b) :
  (a + 2*b) / (a + b) - (a - b) / (a - 2*b) / ((a^2 - b^2) / (a^2 - 4*a*b + 4*b^2)) = 4*b / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2987_298734


namespace NUMINAMATH_CALUDE_calculation_result_l2987_298737

theorem calculation_result : 90 + 5 * 12 / (180 / 3) = 91 := by
  sorry

end NUMINAMATH_CALUDE_calculation_result_l2987_298737


namespace NUMINAMATH_CALUDE_subset_intersection_iff_range_l2987_298723

def A (a : ℝ) : Set ℝ := {x | 2*a + 1 ≤ x ∧ x ≤ 3*a - 5}
def B : Set ℝ := {x | 3 ≤ x ∧ x ≤ 22}

theorem subset_intersection_iff_range (a : ℝ) :
  A a ⊆ (A a ∩ B) ↔ 1 ≤ a ∧ a ≤ 9 := by sorry

end NUMINAMATH_CALUDE_subset_intersection_iff_range_l2987_298723


namespace NUMINAMATH_CALUDE_disjunction_truth_l2987_298770

theorem disjunction_truth (p q : Prop) : (p ∨ q) → (p ∨ q) :=
  sorry

end NUMINAMATH_CALUDE_disjunction_truth_l2987_298770


namespace NUMINAMATH_CALUDE_union_equal_M_l2987_298739

def M : Set Char := {'a', 'b', 'c', 'd', 'e'}
def N : Set Char := {'b', 'd', 'e'}

theorem union_equal_M : M ∪ N = M := by sorry

end NUMINAMATH_CALUDE_union_equal_M_l2987_298739


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l2987_298726

/-- Calculates the molecular weight of a compound given the atomic weights and number of atoms of each element. -/
def molecular_weight (al_weight o_weight h_weight : ℝ) (al_count o_count h_count : ℕ) : ℝ :=
  al_weight * al_count + o_weight * o_count + h_weight * h_count

/-- Theorem stating that the molecular weight of a compound with 1 Aluminium, 3 Oxygen, and 3 Hydrogen atoms is 78.001 g/mol. -/
theorem compound_molecular_weight :
  let al_weight : ℝ := 26.98
  let o_weight : ℝ := 15.999
  let h_weight : ℝ := 1.008
  let al_count : ℕ := 1
  let o_count : ℕ := 3
  let h_count : ℕ := 3
  molecular_weight al_weight o_weight h_weight al_count o_count h_count = 78.001 := by
  sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l2987_298726


namespace NUMINAMATH_CALUDE_chinese_gcd_168_93_l2987_298778

def chinese_gcd (a b : ℕ) : ℕ := sorry

def chinese_gcd_sequence (a b : ℕ) : List (ℕ × ℕ) := sorry

theorem chinese_gcd_168_93 :
  let seq := chinese_gcd_sequence 168 93
  (57, 18) ∈ seq ∧
  (3, 18) ∈ seq ∧
  (3, 3) ∈ seq ∧
  (6, 9) ∉ seq ∧
  chinese_gcd 168 93 = 3 := by sorry

end NUMINAMATH_CALUDE_chinese_gcd_168_93_l2987_298778


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2987_298761

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, (a * x^2 + 3 * x + 2 > 0) ↔ (b < x ∧ x < 1)) →
  (a = -5 ∧ b = -2/5) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2987_298761


namespace NUMINAMATH_CALUDE_regular_polygon_diagonals_l2987_298795

theorem regular_polygon_diagonals (n : ℕ) (h1 : n > 2) (h2 : (n - 2) * 180 / n = 120) :
  n - 3 = 3 := by sorry

end NUMINAMATH_CALUDE_regular_polygon_diagonals_l2987_298795


namespace NUMINAMATH_CALUDE_average_marks_proof_l2987_298787

def scores : List ℕ := [76, 65, 82, 67, 55, 89, 74, 63, 78, 71]

theorem average_marks_proof :
  (scores.sum / scores.length : ℚ) = 72 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_proof_l2987_298787


namespace NUMINAMATH_CALUDE_sim_tetrahedron_volume_l2987_298742

/-- A tetrahedron with similar but not all equal triangular faces -/
structure SimTetrahedron where
  /-- The faces are similar triangles -/
  similar_faces : Bool
  /-- Not all faces are equal -/
  not_all_equal : Bool
  /-- Any two faces share at least one pair of equal edges, not counting the common edge -/
  shared_equal_edges : Bool
  /-- Two edges in one face have lengths 3 and 5 -/
  edge_lengths : (ℝ × ℝ)

/-- The volume of a SimTetrahedron is either (55 * √6) / 18 or (11 * √10) / 10 -/
theorem sim_tetrahedron_volume (t : SimTetrahedron) : 
  t.similar_faces ∧ t.not_all_equal ∧ t.shared_equal_edges ∧ t.edge_lengths = (3, 5) →
  (∃ v : ℝ, v = (55 * Real.sqrt 6) / 18 ∨ v = (11 * Real.sqrt 10) / 10) :=
by sorry

end NUMINAMATH_CALUDE_sim_tetrahedron_volume_l2987_298742


namespace NUMINAMATH_CALUDE_equiangular_equilateral_parallelogram_is_square_l2987_298725

-- Define a parallelogram
class Parallelogram (P : Type) where
  -- Add any necessary properties of a parallelogram

-- Define the property of being equiangular
class Equiangular (P : Type) where
  -- All angles are equal

-- Define the property of being equilateral
class Equilateral (P : Type) where
  -- All sides have equal length

-- Define a square
class Square (P : Type) extends Parallelogram P where
  -- A square is a parallelogram with additional properties

-- Theorem statement
theorem equiangular_equilateral_parallelogram_is_square 
  (P : Type) [Parallelogram P] [Equiangular P] [Equilateral P] : Square P :=
by sorry

end NUMINAMATH_CALUDE_equiangular_equilateral_parallelogram_is_square_l2987_298725


namespace NUMINAMATH_CALUDE_chord_midpoint_trajectory_midpoint_PQ_trajectory_l2987_298766

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 12*y + 24 = 0

-- Define point P
def point_P : ℝ × ℝ := (0, 5)

-- Theorem for the trajectory of chord midpoints
theorem chord_midpoint_trajectory (x y : ℝ) : 
  (∃ (a b : ℝ), circle_C a b ∧ circle_C (2*x - a) (2*y - b) ∧ (x + a = 0 ∨ y + b = 5)) →
  x^2 + y^2 + 2*x - 11*y + 30 = 0 :=
sorry

-- Theorem for the trajectory of midpoint M of PQ
theorem midpoint_PQ_trajectory (x y : ℝ) :
  (∃ (q_x q_y : ℝ), circle_C q_x q_y ∧ x = (q_x + point_P.1) / 2 ∧ y = (q_y + point_P.2) / 2) →
  x^2 + y^2 + 2*x - 11*y - 11/4 = 0 :=
sorry

end NUMINAMATH_CALUDE_chord_midpoint_trajectory_midpoint_PQ_trajectory_l2987_298766


namespace NUMINAMATH_CALUDE_complex_function_inequality_l2987_298783

/-- Given a ∈ (0,1) and f(z) = z^2 - z + a for z ∈ ℂ,
    for any z ∈ ℂ with |z| ≥ 1, there exists z₀ ∈ ℂ with |z₀| = 1
    such that |f(z₀)| ≤ |f(z)| -/
theorem complex_function_inequality (a : ℝ) (ha : 0 < a ∧ a < 1) :
  let f : ℂ → ℂ := fun z ↦ z^2 - z + a
  ∀ z : ℂ, Complex.abs z ≥ 1 →
    ∃ z₀ : ℂ, Complex.abs z₀ = 1 ∧ Complex.abs (f z₀) ≤ Complex.abs (f z) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_function_inequality_l2987_298783


namespace NUMINAMATH_CALUDE_semicircle_radius_theorem_l2987_298762

/-- Theorem: Given a rectangle with length 48 cm and width 24 cm, and a semicircle
    attached to one side of the rectangle (with the diameter equal to the length
    of the rectangle), if the perimeter of the combined shape is 144 cm, then the
    radius of the semicircle is 48 / (π + 2) cm. -/
theorem semicircle_radius_theorem (rectangle_length : ℝ) (rectangle_width : ℝ) 
    (combined_perimeter : ℝ) (semicircle_radius : ℝ) :
  rectangle_length = 48 →
  rectangle_width = 24 →
  combined_perimeter = 144 →
  combined_perimeter = 2 * rectangle_width + rectangle_length + π * semicircle_radius →
  semicircle_radius = 48 / (π + 2) :=
by sorry

end NUMINAMATH_CALUDE_semicircle_radius_theorem_l2987_298762
