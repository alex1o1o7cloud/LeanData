import Mathlib

namespace NUMINAMATH_CALUDE_land_reaping_l2869_286928

/-- Given that 4 men can reap 40 acres in 15 days, prove that 16 men can reap 320 acres in 30 days. -/
theorem land_reaping (men_initial : ℕ) (acres_initial : ℕ) (days_initial : ℕ)
                     (men_final : ℕ) (days_final : ℕ) :
  men_initial = 4 →
  acres_initial = 40 →
  days_initial = 15 →
  men_final = 16 →
  days_final = 30 →
  (men_final * days_final * acres_initial) / (men_initial * days_initial) = 320 := by
  sorry

#check land_reaping

end NUMINAMATH_CALUDE_land_reaping_l2869_286928


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l2869_286931

/-- Represents a pyramid with an equilateral triangular base and isosceles lateral faces -/
structure Pyramid where
  base_side_length : ℝ
  lateral_side_length : ℝ
  (base_is_equilateral : base_side_length = 2)
  (lateral_is_isosceles : lateral_side_length = 3)

/-- Represents a cube inscribed in the pyramid -/
structure InscribedCube (p : Pyramid) where
  side_length : ℝ
  (base_on_pyramid_base : True)
  (top_vertices_touch_midpoints : True)

/-- The volume of the inscribed cube -/
def cube_volume (p : Pyramid) (c : InscribedCube p) : ℝ :=
  c.side_length ^ 3

theorem inscribed_cube_volume (p : Pyramid) (c : InscribedCube p) :
  cube_volume p c = (4 * Real.sqrt 2 - 3) ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_l2869_286931


namespace NUMINAMATH_CALUDE_sum_of_naturals_equals_406_l2869_286911

theorem sum_of_naturals_equals_406 (n : ℕ) : (n * (n + 1)) / 2 = 406 → n = 28 := by sorry

end NUMINAMATH_CALUDE_sum_of_naturals_equals_406_l2869_286911


namespace NUMINAMATH_CALUDE_system_solution_l2869_286910

theorem system_solution (x y b : ℝ) : 
  (4 * x + 2 * y = b) → 
  (3 * x + 4 * y = 3 * b) → 
  (x = 3) → 
  (b = -1) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2869_286910


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l2869_286991

theorem triangle_angle_measure (a b : ℝ) (A B : Real) :
  0 < a ∧ 0 < b ∧ 0 < A ∧ A < π ∧ 0 < B ∧ B < π →
  b = 2 * a * Real.sin B →
  Real.sin A = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l2869_286991


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l2869_286970

/-- The complex number z defined as (i+2)/i is located in the fourth quadrant of the complex plane. -/
theorem complex_number_in_fourth_quadrant :
  let z : ℂ := (Complex.I + 2) / Complex.I
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l2869_286970


namespace NUMINAMATH_CALUDE_only_two_random_events_l2869_286921

-- Define the universe of events
inductive Event : Type
| real_number_multiplication : Event
| draw_odd_numbered_ball : Event
| win_lottery : Event
| number_inequality : Event

-- Define a predicate for random events
def is_random_event : Event → Prop
| Event.real_number_multiplication => False
| Event.draw_odd_numbered_ball => True
| Event.win_lottery => True
| Event.number_inequality => False

-- Theorem statement
theorem only_two_random_events :
  (∀ e : Event, is_random_event e ↔ (e = Event.draw_odd_numbered_ball ∨ e = Event.win_lottery)) :=
by sorry

end NUMINAMATH_CALUDE_only_two_random_events_l2869_286921


namespace NUMINAMATH_CALUDE_defective_product_selection_l2869_286987

theorem defective_product_selection (n m k : ℕ) (hn : n = 100) (hm : m = 98) (hk : k = 3) :
  let total := n
  let qualified := m
  let defective := n - m
  let select := k
  Nat.choose n k - Nat.choose m k = 
    Nat.choose defective 1 * Nat.choose qualified 2 + 
    Nat.choose defective 2 * Nat.choose qualified 1 :=
by sorry

end NUMINAMATH_CALUDE_defective_product_selection_l2869_286987


namespace NUMINAMATH_CALUDE_max_notebooks_is_11_l2869_286907

/-- Represents the number of notebooks in a pack -/
inductive NotebookPack
  | Single
  | Pack4
  | Pack7

/-- The cost of a notebook pack in dollars -/
def cost (pack : NotebookPack) : ℕ :=
  match pack with
  | .Single => 2
  | .Pack4 => 6
  | .Pack7 => 9

/-- The number of notebooks in a pack -/
def notebooks (pack : NotebookPack) : ℕ :=
  match pack with
  | .Single => 1
  | .Pack4 => 4
  | .Pack7 => 7

/-- Maria's budget in dollars -/
def budget : ℕ := 15

/-- A purchase combination is valid if it doesn't exceed the budget -/
def isValidPurchase (singles pack4s pack7s : ℕ) : Prop :=
  singles * cost .Single + pack4s * cost .Pack4 + pack7s * cost .Pack7 ≤ budget

/-- The total number of notebooks for a given purchase combination -/
def totalNotebooks (singles pack4s pack7s : ℕ) : ℕ :=
  singles * notebooks .Single + pack4s * notebooks .Pack4 + pack7s * notebooks .Pack7

/-- Theorem: The maximum number of notebooks that can be purchased with the given budget is 11 -/
theorem max_notebooks_is_11 :
    ∀ singles pack4s pack7s : ℕ,
      isValidPurchase singles pack4s pack7s →
      totalNotebooks singles pack4s pack7s ≤ 11 ∧
      ∃ s p4 p7 : ℕ, isValidPurchase s p4 p7 ∧ totalNotebooks s p4 p7 = 11 :=
  sorry


end NUMINAMATH_CALUDE_max_notebooks_is_11_l2869_286907


namespace NUMINAMATH_CALUDE_two_volunteers_same_project_l2869_286927

/-- The number of volunteers -/
def num_volunteers : ℕ := 3

/-- The number of projects -/
def num_projects : ℕ := 7

/-- The probability that exactly two volunteers are assigned to the same project -/
def probability_two_same_project : ℚ := 18/49

theorem two_volunteers_same_project :
  (num_volunteers = 3) →
  (num_projects = 7) →
  (∀ volunteer, volunteer ≤ num_volunteers → ∃! project, project ≤ num_projects) →
  probability_two_same_project = 18/49 := by
  sorry

end NUMINAMATH_CALUDE_two_volunteers_same_project_l2869_286927


namespace NUMINAMATH_CALUDE_stratified_sample_size_l2869_286995

/-- Represents a stratified sampling scenario by gender -/
structure StratifiedSample where
  total_population : ℕ
  male_population : ℕ
  male_sample : ℕ
  total_sample : ℕ

/-- Theorem stating that given the conditions, the total sample size is 36 -/
theorem stratified_sample_size
  (s : StratifiedSample)
  (h1 : s.total_population = 120)
  (h2 : s.male_population = 80)
  (h3 : s.male_sample = 24)
  (h4 : s.male_sample / s.total_sample = s.male_population / s.total_population) :
  s.total_sample = 36 := by
  sorry

#check stratified_sample_size

end NUMINAMATH_CALUDE_stratified_sample_size_l2869_286995


namespace NUMINAMATH_CALUDE_tammy_second_day_speed_l2869_286926

-- Define the total climbing time
def total_time : ℝ := 14

-- Define the relationship between first and second day's climbing time
def time_relationship (h₁ h₂ : ℝ) : Prop := h₂ = h₁ - 2

-- Define the relationship between first and second day's speed
def speed_relationship (s₁ s₂ : ℝ) : Prop := s₂ = s₁ + 0.5

-- Define the total distance climbed
def total_distance : ℝ := 52

-- Define the first day's distance as 60% of total distance
def first_day_distance : ℝ := 0.6 * total_distance

-- Define the second day's distance as 40% of total distance
def second_day_distance : ℝ := 0.4 * total_distance

-- Theorem statement
theorem tammy_second_day_speed 
  (h₁ h₂ s₁ s₂ : ℝ) 
  (h_total : h₁ + h₂ = total_time)
  (h_time : time_relationship h₁ h₂)
  (h_speed : speed_relationship s₁ s₂)
  (h_distance₁ : s₁ * h₁ = first_day_distance)
  (h_distance₂ : s₂ * h₂ = second_day_distance) :
  s₂ = 4.4 := by
  sorry

end NUMINAMATH_CALUDE_tammy_second_day_speed_l2869_286926


namespace NUMINAMATH_CALUDE_birds_in_tree_l2869_286973

theorem birds_in_tree (initial_birds final_birds : ℕ) 
  (h1 : initial_birds = 29) 
  (h2 : final_birds = 42) : 
  final_birds - initial_birds = 13 := by
  sorry

end NUMINAMATH_CALUDE_birds_in_tree_l2869_286973


namespace NUMINAMATH_CALUDE_min_value_of_x_l2869_286992

theorem min_value_of_x (x : ℝ) (h1 : x > 0) 
  (h2 : Real.log x / Real.log 3 ≥ Real.log 9 / Real.log 3 - (1/3) * (Real.log x / Real.log 3)) :
  x ≥ Real.sqrt 27 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_x_l2869_286992


namespace NUMINAMATH_CALUDE_reciprocal_gp_sum_l2869_286934

/-- Given a geometric progression with n terms, first term 1, common ratio r^2 (r ≠ 0),
    and sum s^3, the sum of the geometric progression formed by the reciprocals of each term
    is s^3 / r^2 -/
theorem reciprocal_gp_sum (n : ℕ) (r s : ℝ) (hr : r ≠ 0) :
  let original_sum := (1 - r^(2*n)) / (1 - r^2)
  let reciprocal_sum := (1 - (1/r^2)^n) / (1 - 1/r^2)
  original_sum = s^3 → reciprocal_sum = s^3 / r^2 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_gp_sum_l2869_286934


namespace NUMINAMATH_CALUDE_taehyung_calculation_l2869_286943

theorem taehyung_calculation (x : ℝ) (h : 5 * x = 30) : x / 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_taehyung_calculation_l2869_286943


namespace NUMINAMATH_CALUDE_chord_line_equation_l2869_286940

/-- The equation of a line passing through a chord of an ellipse, given the chord's midpoint -/
theorem chord_line_equation (A B : ℝ × ℝ) (M : ℝ × ℝ) :
  M = (4, 2) →
  M.1 = (A.1 + B.1) / 2 →
  M.2 = (A.2 + B.2) / 2 →
  A.1^2 + 4 * A.2^2 = 36 →
  B.1^2 + 4 * B.2^2 = 36 →
  ∃ (k : ℝ), ∀ (x y : ℝ), (x = A.1 ∧ y = A.2) ∨ (x = B.1 ∧ y = B.2) → x + 2*y - 8 = 0 :=
by sorry

end NUMINAMATH_CALUDE_chord_line_equation_l2869_286940


namespace NUMINAMATH_CALUDE_triangle_special_progression_l2869_286961

theorem triangle_special_progression (A B C : Real) (a b c : Real) :
  -- Triangle conditions
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) ∧ (A + B + C = π) →
  (a > 0) ∧ (b > 0) ∧ (c > 0) →
  -- Law of sines
  (a / Real.sin A = b / Real.sin B) ∧ (b / Real.sin B = c / Real.sin C) →
  -- Arithmetic progression of sides
  (2 * b = a + c) →
  -- Geometric progression of sines
  (Real.sin B)^2 = Real.sin A * Real.sin C →
  -- Conclusion
  B = π/3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_special_progression_l2869_286961


namespace NUMINAMATH_CALUDE_inequality_problem_l2869_286930

theorem inequality_problem (x y a b : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0)
  (h1 : x^2 < a^2) (h2 : y^3 < b^3) :
  ∃ (s : Finset (Fin 4)),
    s.card = 3 ∧
    (∀ i ∈ s, match i with
      | 0 => x^2 + y^2 < a^2 + b^2
      | 1 => x^2 - y^2 < a^2 - b^2
      | 2 => x^2 * y^3 < a^2 * b^3
      | 3 => x^2 / y^3 < a^2 / b^3
    ) ∧
    (∀ i ∉ s, ¬(match i with
      | 0 => x^2 + y^2 < a^2 + b^2
      | 1 => x^2 - y^2 < a^2 - b^2
      | 2 => x^2 * y^3 < a^2 * b^3
      | 3 => x^2 / y^3 < a^2 / b^3
    )) := by sorry

end NUMINAMATH_CALUDE_inequality_problem_l2869_286930


namespace NUMINAMATH_CALUDE_sine_problem_l2869_286920

theorem sine_problem (α : ℝ) (h : Real.sin (2 * Real.pi / 3 - α) + Real.sin α = 4 * Real.sqrt 3 / 5) :
  Real.sin (α + 7 * Real.pi / 6) = -4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sine_problem_l2869_286920


namespace NUMINAMATH_CALUDE_worker_b_completion_time_worker_b_time_is_9_l2869_286952

/-- Given workers a, b, and c who can complete a task together or individually,
    this theorem proves the time taken by worker b to complete the task alone. -/
theorem worker_b_completion_time
  (total_rate : ℝ)
  (rate_a : ℝ)
  (rate_b : ℝ)
  (rate_c : ℝ)
  (h1 : total_rate = rate_a + rate_b + rate_c)
  (h2 : total_rate = 1 / 4)
  (h3 : rate_a = 1 / 12)
  (h4 : rate_c = 1 / 18) :
  rate_b = 1 / 9 := by
  sorry

/-- The time taken by worker b to complete the task alone -/
def time_b : ℝ := 9

/-- Proves that the time taken by worker b is indeed 9 days -/
theorem worker_b_time_is_9
  (total_rate : ℝ)
  (rate_a : ℝ)
  (rate_b : ℝ)
  (rate_c : ℝ)
  (h1 : total_rate = rate_a + rate_b + rate_c)
  (h2 : total_rate = 1 / 4)
  (h3 : rate_a = 1 / 12)
  (h4 : rate_c = 1 / 18) :
  time_b = 1 / rate_b := by
  sorry

end NUMINAMATH_CALUDE_worker_b_completion_time_worker_b_time_is_9_l2869_286952


namespace NUMINAMATH_CALUDE_contractor_payment_l2869_286906

/-- A contractor's payment problem -/
theorem contractor_payment
  (total_days : ℕ)
  (payment_per_day : ℚ)
  (fine_per_day : ℚ)
  (absent_days : ℕ)
  (h1 : total_days = 30)
  (h2 : payment_per_day = 25)
  (h3 : fine_per_day = 7.5)
  (h4 : absent_days = 10)
  (h5 : absent_days ≤ total_days) :
  (total_days - absent_days) * payment_per_day - absent_days * fine_per_day = 425 :=
by sorry

end NUMINAMATH_CALUDE_contractor_payment_l2869_286906


namespace NUMINAMATH_CALUDE_reciprocal_inequality_l2869_286908

theorem reciprocal_inequality (a b : ℝ) (h1 : 0 < a) (h2 : a < b) : 1 / a < 1 / b := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_inequality_l2869_286908


namespace NUMINAMATH_CALUDE_target_parabola_satisfies_conditions_l2869_286955

/-- Represents a parabola with specific properties -/
structure Parabola where
  -- Equation coefficients
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ
  -- Conditions
  passes_through : a * 2^2 + b * 2 * 8 + c * 8^2 + d * 2 + e * 8 + f = 0
  focus_y : ℤ := 4
  vertex_on_y_axis : a * 0^2 + b * 0 * 4 + c * 4^2 + d * 0 + e * 4 + f = 0
  c_positive : c > 0
  gcd_one : Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (Int.natAbs a) (Int.natAbs b)) (Int.natAbs c)) (Int.natAbs d)) (Int.natAbs e)) (Int.natAbs f) = 1

/-- The specific parabola we want to prove -/
def target_parabola : Parabola :=
  { a := 0,
    b := 0,
    c := 1,
    d := -8,
    e := -8,
    f := 16,
    passes_through := sorry,
    focus_y := 4,
    vertex_on_y_axis := sorry,
    c_positive := sorry,
    gcd_one := sorry }

/-- Theorem stating that the target parabola satisfies all conditions -/
theorem target_parabola_satisfies_conditions : 
  ∃ (p : Parabola), p = target_parabola := by sorry

end NUMINAMATH_CALUDE_target_parabola_satisfies_conditions_l2869_286955


namespace NUMINAMATH_CALUDE_long_jump_competition_l2869_286957

/-- The long jump competition problem -/
theorem long_jump_competition (first second third fourth : ℝ) : 
  first = 22 →
  second = first + 1 →
  third < second →
  fourth = third + 3 →
  fourth = 24 →
  second - third = 2 :=
by sorry

end NUMINAMATH_CALUDE_long_jump_competition_l2869_286957


namespace NUMINAMATH_CALUDE_x_plus_2y_equals_8_l2869_286963

theorem x_plus_2y_equals_8 (x y : ℝ) 
  (h1 : (x + y) / 3 = 1.6666666666666667)
  (h2 : 2 * x + y = 7) : 
  x + 2 * y = 8 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_2y_equals_8_l2869_286963


namespace NUMINAMATH_CALUDE_password_probability_l2869_286914

/-- Represents the composition of a password --/
structure Password :=
  (first_letter : Char)
  (middle_digit : Nat)
  (last_letter : Char)

/-- Defines the set of vowels --/
def vowels : Set Char := {'A', 'E', 'I', 'O', 'U'}

/-- Defines the set of even single-digit numbers --/
def even_single_digits : Set Nat := {0, 2, 4, 6, 8}

/-- The total number of letters in the alphabet --/
def alphabet_size : Nat := 26

/-- The number of vowels --/
def vowel_count : Nat := 5

/-- The number of single-digit numbers --/
def single_digit_count : Nat := 10

/-- The number of even single-digit numbers --/
def even_single_digit_count : Nat := 5

/-- Theorem stating the probability of a specific password pattern --/
theorem password_probability :
  (((vowel_count : ℚ) / alphabet_size) *
   ((even_single_digit_count : ℚ) / single_digit_count) *
   ((alphabet_size - vowel_count : ℚ) / alphabet_size)) =
  (105 : ℚ) / 1352 := by
  sorry

end NUMINAMATH_CALUDE_password_probability_l2869_286914


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l2869_286917

/-- The perimeter of a rhombus with diagonals measuring 24 feet and 16 feet is 16√13 feet. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 16) :
  let side := Real.sqrt ((d1/2)^2 + (d2/2)^2)
  4 * side = 16 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l2869_286917


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_r_value_l2869_286999

-- Define the sum of the first n terms of the geometric sequence
def S (n : ℕ) (r : ℚ) : ℚ := 3^(n-1) - r

-- Define the geometric sequence
def a (n : ℕ) (r : ℚ) : ℚ := S n r - S (n-1) r

-- Theorem statement
theorem geometric_sequence_sum_r_value :
  ∃ (r : ℚ), ∀ (n : ℕ), n ≥ 2 → a n r = 2 * 3^(n-2) ∧ a 1 r = 1 - r → r = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_r_value_l2869_286999


namespace NUMINAMATH_CALUDE_four_digit_divisible_by_50_l2869_286965

theorem four_digit_divisible_by_50 : 
  (Finset.filter 
    (fun n : ℕ => n ≥ 1000 ∧ n < 10000 ∧ n % 100 = 50 ∧ n % 50 = 0) 
    (Finset.range 10000)).card = 90 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_divisible_by_50_l2869_286965


namespace NUMINAMATH_CALUDE_total_distance_covered_l2869_286939

-- Define the given conditions
def cycling_time : ℚ := 30 / 60  -- 30 minutes in hours
def cycling_rate : ℚ := 12       -- 12 mph
def skating_time : ℚ := 45 / 60  -- 45 minutes in hours
def skating_rate : ℚ := 8        -- 8 mph
def total_time : ℚ := 75 / 60    -- 1 hour and 15 minutes in hours

-- State the theorem
theorem total_distance_covered : 
  cycling_time * cycling_rate + skating_time * skating_rate = 12 := by
  sorry -- The proof is omitted as per instructions

end NUMINAMATH_CALUDE_total_distance_covered_l2869_286939


namespace NUMINAMATH_CALUDE_power_relation_l2869_286985

theorem power_relation (x a : ℝ) (h : x^(-a) = 3) : x^(2*a) = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_power_relation_l2869_286985


namespace NUMINAMATH_CALUDE_price_increase_achieves_target_profit_l2869_286936

/-- Represents the supermarket's pomelo sales scenario -/
structure PomeloSales where
  initial_profit_per_kg : ℝ
  initial_daily_sales : ℝ
  price_increase : ℝ
  sales_decrease_per_yuan : ℝ
  target_daily_profit : ℝ

/-- Calculates the daily profit based on the price increase -/
def daily_profit (s : PomeloSales) : ℝ :=
  (s.initial_profit_per_kg + s.price_increase) *
  (s.initial_daily_sales - s.sales_decrease_per_yuan * s.price_increase)

/-- Theorem stating that a 5 yuan price increase achieves the target profit -/
theorem price_increase_achieves_target_profit (s : PomeloSales)
  (h1 : s.initial_profit_per_kg = 10)
  (h2 : s.initial_daily_sales = 500)
  (h3 : s.sales_decrease_per_yuan = 20)
  (h4 : s.target_daily_profit = 6000)
  (h5 : s.price_increase = 5) :
  daily_profit s = s.target_daily_profit :=
by sorry


end NUMINAMATH_CALUDE_price_increase_achieves_target_profit_l2869_286936


namespace NUMINAMATH_CALUDE_furniture_store_problem_l2869_286937

/-- Furniture store problem -/
theorem furniture_store_problem 
  (a : ℝ) 
  (table_price : ℝ → ℝ) 
  (chair_price : ℝ → ℝ) 
  (table_retail : ℝ) 
  (chair_retail : ℝ) 
  (set_price : ℝ) 
  (h1 : table_price a = a) 
  (h2 : chair_price a = a - 140) 
  (h3 : table_retail = 380) 
  (h4 : chair_retail = 160) 
  (h5 : set_price = 940) 
  (h6 : 600 / (a - 140) = 1300 / a) 
  (x : ℝ) 
  (h7 : x + 5 * x + 20 ≤ 200) 
  (profit : ℝ → ℝ) 
  (h8 : profit x = (set_price - table_price a - 4 * chair_price a) * (1/2 * x) + 
                   (table_retail - table_price a) * (1/2 * x) + 
                   (chair_retail - chair_price a) * (5 * x + 20 - 4 * (1/2 * x))) :
  a = 260 ∧ 
  (∃ (max_x : ℝ), max_x = 30 ∧ 
    (∀ y, y + 5 * y + 20 ≤ 200 → profit y ≤ profit max_x) ∧ 
    profit max_x = 9200) := by
  sorry

end NUMINAMATH_CALUDE_furniture_store_problem_l2869_286937


namespace NUMINAMATH_CALUDE_selling_price_is_180_l2869_286923

/-- Calculates the selling price per machine to break even -/
def selling_price_per_machine (cost_parts : ℕ) (cost_patent : ℕ) (num_machines : ℕ) : ℕ :=
  (cost_parts + cost_patent) / num_machines

/-- Theorem: The selling price per machine is $180 -/
theorem selling_price_is_180 :
  selling_price_per_machine 3600 4500 45 = 180 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_is_180_l2869_286923


namespace NUMINAMATH_CALUDE_chord_length_difference_l2869_286945

theorem chord_length_difference (r₁ r₂ : ℝ) (hr₁ : r₁ = 26) (hr₂ : r₂ = 5) :
  let longest_chord := 2 * r₁
  let shortest_chord := 2 * Real.sqrt (r₁^2 - (r₁ - r₂)^2)
  longest_chord - shortest_chord = 52 - 2 * Real.sqrt 235 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_difference_l2869_286945


namespace NUMINAMATH_CALUDE_folded_paper_area_ratio_l2869_286996

/-- Represents a square piece of paper -/
structure Paper where
  side : ℝ
  area : ℝ
  area_eq : area = side ^ 2

/-- Represents the folded paper -/
structure FoldedPaper where
  original : Paper
  new_area : ℝ

/-- Theorem stating the ratio of areas after folding -/
theorem folded_paper_area_ratio (p : Paper) (fp : FoldedPaper) 
  (h_fp : fp.original = p) 
  (h_fold : fp.new_area = (7 / 8) * p.area) : 
  fp.new_area / p.area = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_folded_paper_area_ratio_l2869_286996


namespace NUMINAMATH_CALUDE_inequality_proof_l2869_286909

theorem inequality_proof (a b : ℝ) (n : ℕ) (ha : a > 0) (hb : b > 0) :
  (1 + a / b) ^ n + (1 + b / a) ^ n ≥ 2^(n + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2869_286909


namespace NUMINAMATH_CALUDE_canoe_kayak_difference_l2869_286929

/-- Represents the rental information for canoes and kayaks --/
structure RentalInfo where
  canoe_cost : ℕ  -- Cost of renting a canoe per day
  kayak_cost : ℕ  -- Cost of renting a kayak per day
  canoe_kayak_ratio : Rat  -- Ratio of canoes to kayaks rented
  total_revenue : ℕ  -- Total revenue from rentals

/-- 
Given rental information, proves that the difference between 
the number of canoes and kayaks rented is 4
--/
theorem canoe_kayak_difference (info : RentalInfo) 
  (h1 : info.canoe_cost = 14)
  (h2 : info.kayak_cost = 15)
  (h3 : info.canoe_kayak_ratio = 3/2)
  (h4 : info.total_revenue = 288) :
  ∃ (c k : ℕ), c = k + 4 ∧ 
    c * info.canoe_cost + k * info.kayak_cost = info.total_revenue ∧
    (c : Rat) / k = info.canoe_kayak_ratio := by
  sorry

end NUMINAMATH_CALUDE_canoe_kayak_difference_l2869_286929


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2869_286977

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) :
  GeometricSequence a → a 5 * a 14 = 5 → a 8 * a 9 * a 10 * a 11 = 25 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l2869_286977


namespace NUMINAMATH_CALUDE_manager_average_salary_l2869_286962

/-- Proves that the average salary of managers is $90,000 given the company's employee structure and salary information --/
theorem manager_average_salary 
  (num_managers : ℕ) 
  (num_associates : ℕ) 
  (associate_avg_salary : ℝ) 
  (company_avg_salary : ℝ) : 
  num_managers = 15 → 
  num_associates = 75 → 
  associate_avg_salary = 30000 → 
  company_avg_salary = 40000 → 
  (num_managers * (num_managers * company_avg_salary - num_associates * associate_avg_salary) / 
   (num_managers * (num_managers + num_associates))) = 90000 := by
  sorry

end NUMINAMATH_CALUDE_manager_average_salary_l2869_286962


namespace NUMINAMATH_CALUDE_total_amount_correct_l2869_286902

/-- Represents the total amount lent out in rupees -/
def total_amount : ℝ := 11501.6

/-- Represents the amount lent at 8% p.a. in rupees -/
def amount_at_8_percent : ℝ := 15008

/-- Represents the total interest received after one year in rupees -/
def total_interest : ℝ := 850

/-- Theorem stating that the total amount lent out is correct given the conditions -/
theorem total_amount_correct :
  ∃ (amount_at_10_percent : ℝ),
    amount_at_8_percent + amount_at_10_percent = total_amount ∧
    0.08 * amount_at_8_percent + 0.1 * amount_at_10_percent = total_interest :=
by sorry

end NUMINAMATH_CALUDE_total_amount_correct_l2869_286902


namespace NUMINAMATH_CALUDE_system_solutions_l2869_286950

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := x^4 + (7/2)*x^2*y + 2*y^3 = 0
def equation2 (x y : ℝ) : Prop := 4*x^2 + 7*x*y + 2*y^3 = 0

-- Define the solution set
def solution_set : Set (ℝ × ℝ) :=
  {(0, 0), (2, -1), (-11/2, -11/2)}

-- Theorem stating that the solution set contains all and only solutions to the system
theorem system_solutions :
  ∀ (x y : ℝ), (equation1 x y ∧ equation2 x y) ↔ (x, y) ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_system_solutions_l2869_286950


namespace NUMINAMATH_CALUDE_probability_three_out_of_ten_l2869_286946

/-- The probability of selecting at least one defective item from a set of products -/
def probability_at_least_one_defective (total : ℕ) (defective : ℕ) (selected : ℕ) : ℚ :=
  1 - (Nat.choose (total - defective) selected : ℚ) / (Nat.choose total selected : ℚ)

/-- Theorem stating the probability of selecting at least one defective item
    when 3 out of 10 items are defective and 3 items are randomly selected -/
theorem probability_three_out_of_ten :
  probability_at_least_one_defective 10 3 3 = 17 / 24 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_out_of_ten_l2869_286946


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l2869_286958

theorem inscribed_circle_radius (s : ℝ) (r : ℝ) (h : s > 0) :
  3 * s = π * r^2 ∧ r = (s * Real.sqrt 3) / 6 →
  r = 6 * Real.sqrt 3 / π :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l2869_286958


namespace NUMINAMATH_CALUDE_line_tangent_to_ellipse_l2869_286989

theorem line_tangent_to_ellipse (m : ℝ) :
  (∀ x y : ℝ, y = m * x + 1 ∧ x^2 + 4 * y^2 = 1 → 
    ∃! p : ℝ × ℝ, p.1^2 + 4 * p.2^2 = 1 ∧ p.2 = m * p.1 + 1) →
  m^2 = 3/4 := by
sorry

end NUMINAMATH_CALUDE_line_tangent_to_ellipse_l2869_286989


namespace NUMINAMATH_CALUDE_parabola_point_coordinates_l2869_286905

/-- A point on a parabola with a specific distance to its focus -/
def PointOnParabola (x y : ℝ) : Prop :=
  x^2 = 4*y ∧ (x - 0)^2 + (y - 1/4)^2 = 10^2

/-- The coordinates of the point satisfy the given conditions -/
theorem parabola_point_coordinates :
  ∀ x y : ℝ, PointOnParabola x y → (x = 6 ∨ x = -6) ∧ y = 9 :=
by sorry

end NUMINAMATH_CALUDE_parabola_point_coordinates_l2869_286905


namespace NUMINAMATH_CALUDE_vector_b_coordinates_l2869_286933

def vector_a : ℝ × ℝ := (3, -4)

def opposite_direction (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k < 0 ∧ w = (k * v.1, k * v.2)

theorem vector_b_coordinates :
  ∀ (b : ℝ × ℝ),
    opposite_direction vector_a b →
    Real.sqrt (b.1^2 + b.2^2) = 10 →
    b = (-6, 8) := by sorry

end NUMINAMATH_CALUDE_vector_b_coordinates_l2869_286933


namespace NUMINAMATH_CALUDE_cab_driver_average_income_l2869_286959

theorem cab_driver_average_income (incomes : List ℝ) 
  (h1 : incomes = [300, 150, 750, 200, 600]) : 
  (incomes.sum / incomes.length) = 400 := by
  sorry

end NUMINAMATH_CALUDE_cab_driver_average_income_l2869_286959


namespace NUMINAMATH_CALUDE_asterisk_replacement_l2869_286981

theorem asterisk_replacement : ∃ x : ℝ, (x / 21) * (x / 84) = 1 ∧ x = 42 := by sorry

end NUMINAMATH_CALUDE_asterisk_replacement_l2869_286981


namespace NUMINAMATH_CALUDE_simplify_fraction_l2869_286913

theorem simplify_fraction : (180 : ℚ) / 270 = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2869_286913


namespace NUMINAMATH_CALUDE_square_of_97_l2869_286953

theorem square_of_97 : 97 * 97 = 9409 := by
  sorry

end NUMINAMATH_CALUDE_square_of_97_l2869_286953


namespace NUMINAMATH_CALUDE_cell_division_3_hours_l2869_286901

/-- The number of cells after a given number of 30-minute intervals -/
def num_cells (n : ℕ) : ℕ := 2^n

/-- The number of 30-minute intervals in 3 hours -/
def intervals_in_3_hours : ℕ := 6

theorem cell_division_3_hours : 
  num_cells intervals_in_3_hours = 128 := by
  sorry

end NUMINAMATH_CALUDE_cell_division_3_hours_l2869_286901


namespace NUMINAMATH_CALUDE_trader_pens_sold_l2869_286912

/-- Calculates the number of pens sold given the gain and gain percentage -/
def pens_sold (gain_in_pens : ℕ) (gain_percentage : ℕ) : ℕ :=
  (gain_in_pens * 100) / gain_percentage

theorem trader_pens_sold : pens_sold 40 40 = 100 := by
  sorry

end NUMINAMATH_CALUDE_trader_pens_sold_l2869_286912


namespace NUMINAMATH_CALUDE_new_average_age_with_teacher_l2869_286935

theorem new_average_age_with_teacher 
  (num_students : ℕ) 
  (student_avg_age : ℝ) 
  (teacher_age : ℕ) 
  (h1 : num_students = 20) 
  (h2 : student_avg_age = 15) 
  (h3 : teacher_age = 36) : 
  (num_students * student_avg_age + teacher_age) / (num_students + 1) = 16 := by
  sorry

end NUMINAMATH_CALUDE_new_average_age_with_teacher_l2869_286935


namespace NUMINAMATH_CALUDE_games_required_equals_participants_minus_one_l2869_286964

/-- Represents a single-elimination tournament -/
structure SingleEliminationTournament where
  participants : ℕ
  games_required : ℕ

/-- The number of games required in a single-elimination tournament is one less than the number of participants -/
theorem games_required_equals_participants_minus_one 
  (tournament : SingleEliminationTournament) 
  (h : tournament.participants = 512) : 
  tournament.games_required = 511 := by
  sorry

end NUMINAMATH_CALUDE_games_required_equals_participants_minus_one_l2869_286964


namespace NUMINAMATH_CALUDE_smallest_integer_gcd_with_18_l2869_286993

theorem smallest_integer_gcd_with_18 : ∃ n : ℕ, n > 100 ∧ n.gcd 18 = 6 ∧ ∀ m : ℕ, m > 100 ∧ m.gcd 18 = 6 → n ≤ m := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_gcd_with_18_l2869_286993


namespace NUMINAMATH_CALUDE_base_10_to_base_2_l2869_286922

theorem base_10_to_base_2 (n : Nat) (h : n = 123) :
  ∃ (a b c d e f g : Nat),
    a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 0 ∧ f = 1 ∧ g = 1 ∧
    n = a * 2^6 + b * 2^5 + c * 2^4 + d * 2^3 + e * 2^2 + f * 2^1 + g * 2^0 :=
by sorry

end NUMINAMATH_CALUDE_base_10_to_base_2_l2869_286922


namespace NUMINAMATH_CALUDE_prob_at_least_one_karnataka_l2869_286954

/-- The probability of selecting at least one student from Karnataka -/
theorem prob_at_least_one_karnataka (total : ℕ) (karnataka : ℕ) (selected : ℕ)
  (h1 : total = 10)
  (h2 : karnataka = 3)
  (h3 : selected = 4) :
  (1 : ℚ) - (Nat.choose (total - karnataka) selected : ℚ) / (Nat.choose total selected : ℚ) = 5/6 :=
sorry

end NUMINAMATH_CALUDE_prob_at_least_one_karnataka_l2869_286954


namespace NUMINAMATH_CALUDE_vertical_shift_proof_l2869_286942

/-- Represents a line in slope-intercept form -/
structure Line where
  slope : ℚ
  intercept : ℚ

/-- Shifts a line vertically by a given amount -/
def vertical_shift (l : Line) (shift : ℚ) : Line :=
  { slope := l.slope, intercept := l.intercept + shift }

theorem vertical_shift_proof (x : ℚ) :
  let l1 : Line := { slope := -3/4, intercept := 0 }
  let l2 : Line := { slope := -3/4, intercept := -4 }
  vertical_shift l1 (-4) = l2 := by
  sorry

end NUMINAMATH_CALUDE_vertical_shift_proof_l2869_286942


namespace NUMINAMATH_CALUDE_parallel_iff_a_eq_two_l2869_286919

/-- Two lines in the plane -/
structure TwoLines where
  line1 : ℝ → ℝ → ℝ  -- represents ax + 2y = 0
  line2 : ℝ → ℝ → ℝ  -- represents x + y = 1

/-- The condition for parallelism -/
def parallel (l : TwoLines) (a : ℝ) : Prop :=
  ∀ x y, l.line1 x y = 0 ∧ l.line2 x y = 1 → 
    ∃ k : ℝ, k ≠ 0 ∧ (a = k ∧ 2 = k)

/-- The theorem stating that a=2 is necessary and sufficient for parallelism -/
theorem parallel_iff_a_eq_two (l : TwoLines) : 
  (∀ a, parallel l a ↔ a = 2) := by sorry

end NUMINAMATH_CALUDE_parallel_iff_a_eq_two_l2869_286919


namespace NUMINAMATH_CALUDE_probability_theorem_l2869_286924

/-- Probability of reaching point (0, n) for a particle with given movement rules -/
def probability_reach_n (n : ℕ) : ℚ :=
  2/3 + 1/12 * (1 - (-1/3)^(n-1))

/-- Movement rules for the particle -/
structure MovementRules where
  prob_move_1 : ℚ := 2/3
  prob_move_2 : ℚ := 1/3
  vector_1 : Fin 2 → ℤ := ![0, 1]
  vector_2 : Fin 2 → ℤ := ![0, 2]

theorem probability_theorem (n : ℕ) (rules : MovementRules) :
  probability_reach_n n =
  2/3 + 1/12 * (1 - (-1/3)^(n-1)) :=
by sorry

end NUMINAMATH_CALUDE_probability_theorem_l2869_286924


namespace NUMINAMATH_CALUDE_line_separate_from_circle_l2869_286994

/-- Given a point (a, b) within the unit circle, prove that the line ax + by = 1 is separate from the circle -/
theorem line_separate_from_circle (a b : ℝ) (h : a^2 + b^2 < 1) :
  ∀ x y : ℝ, x^2 + y^2 = 1 → a*x + b*y ≠ 1 := by sorry

end NUMINAMATH_CALUDE_line_separate_from_circle_l2869_286994


namespace NUMINAMATH_CALUDE_question_always_truthful_l2869_286932

-- Define the types of residents
inductive ResidentType
| Knight
| Liar

-- Define the possible answers
inductive Answer
| Yes
| No

-- Define a function to represent the truth about having a crocodile
def hasCrocodile : ResidentType → Bool → Answer
| ResidentType.Knight, true => Answer.Yes
| ResidentType.Knight, false => Answer.No
| ResidentType.Liar, true => Answer.No
| ResidentType.Liar, false => Answer.Yes

-- Define the function that represents the response to the question
def responseToQuestion (resident : ResidentType) (hasCroc : Bool) : Answer :=
  hasCrocodile resident hasCroc

-- Theorem: The response to the question always gives the truthful answer
theorem question_always_truthful (resident : ResidentType) (hasCroc : Bool) :
  responseToQuestion resident hasCroc = hasCrocodile ResidentType.Knight hasCroc :=
by sorry

end NUMINAMATH_CALUDE_question_always_truthful_l2869_286932


namespace NUMINAMATH_CALUDE_button_sequence_l2869_286948

theorem button_sequence (a : ℕ → ℕ) : 
  a 1 = 1 →                 -- First term is 1
  (∀ n : ℕ, a (n + 1) = 3 * a n) →  -- Common ratio is 3
  a 6 = 243 →               -- Sixth term is 243
  a 5 = 81 :=               -- Prove fifth term is 81
by sorry

end NUMINAMATH_CALUDE_button_sequence_l2869_286948


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l2869_286986

/-- An arithmetic sequence starting with 1, having a common difference of -2, and ending with -89, has 46 terms. -/
theorem arithmetic_sequence_length :
  ∀ (a : ℕ → ℤ), 
    (a 0 = 1) →  -- First term is 1
    (∀ n, a (n + 1) - a n = -2) →  -- Common difference is -2
    (∃ N, a N = -89 ∧ ∀ k, k > N → a k < -89) →  -- Sequence ends at -89
    (∃ N, N = 46 ∧ a (N - 1) = -89) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l2869_286986


namespace NUMINAMATH_CALUDE_expected_correct_answers_l2869_286998

theorem expected_correct_answers 
  (total_problems : ℕ) 
  (katya_probability : ℚ) 
  (pen_probability : ℚ) 
  (katya_problems : ℕ) :
  total_problems = 20 →
  katya_probability = 4/5 →
  pen_probability = 1/2 →
  katya_problems ≥ 10 →
  katya_problems ≤ total_problems →
  (katya_problems : ℚ) * katya_probability + 
  (total_problems - katya_problems : ℚ) * pen_probability ≥ 13 := by
sorry

end NUMINAMATH_CALUDE_expected_correct_answers_l2869_286998


namespace NUMINAMATH_CALUDE_product_sum_of_three_numbers_l2869_286997

theorem product_sum_of_three_numbers 
  (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 149) 
  (h2 : a + b + c = 17) : 
  a * b + b * c + c * a = 70 := by
sorry

end NUMINAMATH_CALUDE_product_sum_of_three_numbers_l2869_286997


namespace NUMINAMATH_CALUDE_triangle_properties_l2869_286971

theorem triangle_properties (a b c A B C : ℝ) (r : ℝ) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  -- Condition: √3b = a(√3cosC - sinC)
  (Real.sqrt 3 * b = a * (Real.sqrt 3 * Real.cos C - Real.sin C)) →
  -- Condition: a = 8
  (a = 8) →
  -- Condition: Radius of incircle = √3
  (r = Real.sqrt 3) →
  -- Proof that angle A = 2π/3
  (A = 2 * Real.pi / 3) ∧
  -- Proof that perimeter = 18
  (a + b + c = 18) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l2869_286971


namespace NUMINAMATH_CALUDE_absolute_value_nonnegative_and_two_plus_two_not_zero_l2869_286966

theorem absolute_value_nonnegative_and_two_plus_two_not_zero :
  (∀ x : ℝ, |x| ≥ 0) ∧ ¬(2 + 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_nonnegative_and_two_plus_two_not_zero_l2869_286966


namespace NUMINAMATH_CALUDE_special_rectangle_perimeter_l2869_286918

/-- A rectangle with integer dimensions where the area equals the perimeter minus 4 -/
structure SpecialRectangle where
  length : ℕ
  width : ℕ
  not_square : length ≠ width
  area_perimeter_relation : length * width = 2 * (length + width) - 4

/-- The perimeter of a SpecialRectangle is 26 -/
theorem special_rectangle_perimeter (r : SpecialRectangle) : 2 * (r.length + r.width) = 26 := by
  sorry

end NUMINAMATH_CALUDE_special_rectangle_perimeter_l2869_286918


namespace NUMINAMATH_CALUDE_person_B_age_l2869_286990

theorem person_B_age (a b c d e f g : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  c = d / 2 →
  d = e - 3 →
  f = a * d →
  g = b + e →
  a + b + c + d + e + f + g = 292 →
  b = 14 := by
sorry

end NUMINAMATH_CALUDE_person_B_age_l2869_286990


namespace NUMINAMATH_CALUDE_expression_value_l2869_286976

theorem expression_value : ∀ a b : ℝ, 
  (a - 2)^2 + |b + 3| = 0 → 
  3*a^2*b - (2*a*b^2 - 2*(a*b - 3/2*a^2*b) + a*b) + 3*a*b^2 = 12 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l2869_286976


namespace NUMINAMATH_CALUDE_inequality_proof_l2869_286969

def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

theorem inequality_proof (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (∀ x : ℝ, f 1 x ≤ 1 ↔ 0 ≤ x ∧ x ≤ 2) →
  1/m + 1/(2*n) = 1 →
  m + 2*n ≥ 4 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2869_286969


namespace NUMINAMATH_CALUDE_expo_volunteer_selection_l2869_286938

/-- The number of volunteers --/
def total_volunteers : ℕ := 5

/-- The number of volunteers to be selected --/
def selected_volunteers : ℕ := 4

/-- The number of tasks --/
def total_tasks : ℕ := 4

/-- The number of restricted tasks --/
def restricted_tasks : ℕ := 2

/-- The number of volunteers restricted to certain tasks --/
def restricted_volunteers : ℕ := 2

/-- The number of unrestricted volunteers --/
def unrestricted_volunteers : ℕ := total_volunteers - restricted_volunteers

theorem expo_volunteer_selection :
  (Nat.choose restricted_volunteers 1 * Nat.choose restricted_tasks 1 * (unrestricted_volunteers).factorial) +
  (Nat.choose restricted_volunteers 2 * (unrestricted_volunteers).factorial) = 36 := by
  sorry

end NUMINAMATH_CALUDE_expo_volunteer_selection_l2869_286938


namespace NUMINAMATH_CALUDE_prime_power_divides_l2869_286975

theorem prime_power_divides (p a n : ℕ) : 
  Prime p → a > 0 → n > 0 → p ∣ a^n → p^n ∣ a^n := by sorry

end NUMINAMATH_CALUDE_prime_power_divides_l2869_286975


namespace NUMINAMATH_CALUDE_largest_number_problem_l2869_286972

theorem largest_number_problem (A B C : ℝ) 
  (sum_eq : A + B + C = 50)
  (first_eq : A = 2 * B - 43)
  (third_eq : C = (1/2) * A + 5) :
  max A (max B C) = B ∧ B = 27.375 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_problem_l2869_286972


namespace NUMINAMATH_CALUDE_log_equation_solution_l2869_286982

theorem log_equation_solution (x : ℝ) :
  x > 0 → ((Real.log x / Real.log 2) * (Real.log x / Real.log 5) = Real.log 10 / Real.log 2) ↔
  x = 2 ^ Real.sqrt (6 * Real.log 2) :=
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2869_286982


namespace NUMINAMATH_CALUDE_greatest_integer_satisfying_conditions_l2869_286916

theorem greatest_integer_satisfying_conditions : ∃ n : ℕ, 
  n < 200 ∧ 
  ∃ k : ℕ, n + 2 = 9 * k ∧
  ∃ l : ℕ, n + 4 = 10 * l ∧
  ∀ m : ℕ, m < 200 → 
    (∃ p : ℕ, m + 2 = 9 * p) → 
    (∃ q : ℕ, m + 4 = 10 * q) → 
    m ≤ n ∧
  n = 166 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_satisfying_conditions_l2869_286916


namespace NUMINAMATH_CALUDE_number_line_relations_l2869_286978

/-- Definition of "A is k related to B" --/
def is_k_related (A B C : ℝ) (k : ℝ) : Prop :=
  |C - A| = k * |C - B| ∧ k > 1

/-- Problem statement --/
theorem number_line_relations (x t k : ℝ) : 
  let A := -3
  let B := 6
  let P := x
  let Q := 6 - 2*t
  (
    /- Part 1 -/
    (is_k_related A B P 2 → x = 3) ∧ 
    
    /- Part 2 -/
    (|x + 2| + |x - 1| = 3 ∧ is_k_related A B P k → 1/8 ≤ k ∧ k ≤ 4/5) ∧
    
    /- Part 3 -/
    (is_k_related (-3 + t) A Q 3 → t = 3/2)
  ) := by sorry

end NUMINAMATH_CALUDE_number_line_relations_l2869_286978


namespace NUMINAMATH_CALUDE_sarah_copies_360_pages_l2869_286951

/-- The number of pages Sarah will copy for a meeting -/
def total_pages (num_people : ℕ) (copies_per_person : ℕ) (pages_per_contract : ℕ) : ℕ :=
  num_people * copies_per_person * pages_per_contract

/-- Proof that Sarah will copy 360 pages for the meeting -/
theorem sarah_copies_360_pages : 
  total_pages 9 2 20 = 360 := by
  sorry

end NUMINAMATH_CALUDE_sarah_copies_360_pages_l2869_286951


namespace NUMINAMATH_CALUDE_average_of_remaining_numbers_l2869_286944

theorem average_of_remaining_numbers 
  (total_average : ℝ) 
  (avg_group1 : ℝ) 
  (avg_group2 : ℝ) 
  (h1 : total_average = 3.9) 
  (h2 : avg_group1 = 3.4) 
  (h3 : avg_group2 = 3.85) : 
  (6 * total_average - 2 * avg_group1 - 2 * avg_group2) / 2 = 4.45 := by
sorry

#eval (6 * 3.9 - 2 * 3.4 - 2 * 3.85) / 2

end NUMINAMATH_CALUDE_average_of_remaining_numbers_l2869_286944


namespace NUMINAMATH_CALUDE_complex_magnitude_l2869_286900

theorem complex_magnitude (z : ℂ) (h : Complex.I * z = 3 + Complex.I) : 
  Complex.abs z = Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2869_286900


namespace NUMINAMATH_CALUDE_expected_carrot_yield_l2869_286941

def garden_length_steps : ℕ := 25
def garden_width_steps : ℕ := 35
def step_length_feet : ℕ := 3
def yield_per_sqft : ℚ := 3/4

theorem expected_carrot_yield :
  let garden_length_feet : ℕ := garden_length_steps * step_length_feet
  let garden_width_feet : ℕ := garden_width_steps * step_length_feet
  let garden_area_sqft : ℕ := garden_length_feet * garden_width_feet
  garden_area_sqft * yield_per_sqft = 5906.25 := by
  sorry

end NUMINAMATH_CALUDE_expected_carrot_yield_l2869_286941


namespace NUMINAMATH_CALUDE_largest_n_with_odd_residues_l2869_286983

theorem largest_n_with_odd_residues : ∃ (n : ℕ), n = 505 ∧ n > 10 ∧
  (∀ (k : ℕ), 2 ≤ k^2 ∧ k^2 ≤ n / 2 → n % (k^2) % 2 = 1) ∧
  (∀ (m : ℕ), m > n → ∃ (j : ℕ), 2 ≤ j^2 ∧ j^2 ≤ m / 2 ∧ m % (j^2) % 2 = 0) :=
sorry

end NUMINAMATH_CALUDE_largest_n_with_odd_residues_l2869_286983


namespace NUMINAMATH_CALUDE_equation_transformation_l2869_286979

theorem equation_transformation (x y : ℝ) (h : y = x - 1/x) :
  x^6 + x^5 - 5*x^4 + 2*x^3 - 5*x^2 + x + 1 = 0 ↔ x^2 * (y^2 + y - 3) = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_transformation_l2869_286979


namespace NUMINAMATH_CALUDE_max_additional_pens_is_four_l2869_286956

def initial_amount : ℕ := 100
def remaining_amount : ℕ := 61
def pens_bought : ℕ := 3

def cost_per_pen : ℕ := (initial_amount - remaining_amount) / pens_bought

def max_additional_pens : ℕ := remaining_amount / cost_per_pen

theorem max_additional_pens_is_four :
  max_additional_pens = 4 := by sorry

end NUMINAMATH_CALUDE_max_additional_pens_is_four_l2869_286956


namespace NUMINAMATH_CALUDE_total_angle_extrema_l2869_286904

/-- A sequence of k positive real numbers -/
def PositiveSequence (k : ℕ) := { seq : Fin k → ℝ // ∀ i, seq i > 0 }

/-- The total angle of rotation for a given sequence of segment lengths -/
noncomputable def TotalAngle (k : ℕ) (seq : Fin k → ℝ) : ℝ := sorry

/-- A permutation of indices -/
def Permutation (k : ℕ) := { perm : Fin k → Fin k // Function.Bijective perm }

theorem total_angle_extrema (k : ℕ) (a : PositiveSequence k) :
  ∃ (max_perm min_perm : Permutation k),
    (∀ i j : Fin k, i ≤ j → (max_perm.val i).val ≤ (max_perm.val j).val) ∧
    (∀ i j : Fin k, i ≤ j → (min_perm.val i).val ≥ (min_perm.val j).val) ∧
    (∀ p : Permutation k,
      TotalAngle k (a.val ∘ p.val) ≤ TotalAngle k (a.val ∘ max_perm.val) ∧
      TotalAngle k (a.val ∘ p.val) ≥ TotalAngle k (a.val ∘ min_perm.val)) :=
sorry

end NUMINAMATH_CALUDE_total_angle_extrema_l2869_286904


namespace NUMINAMATH_CALUDE_negation_of_no_vegetarian_students_eat_at_cafeteria_l2869_286968

-- Define the universe of discourse
variable (Student : Type)

-- Define predicates
variable (isVegetarian : Student → Prop)
variable (eatsAtCafeteria : Student → Prop)

-- State the theorem
theorem negation_of_no_vegetarian_students_eat_at_cafeteria :
  ¬(∀ s : Student, isVegetarian s → ¬(eatsAtCafeteria s)) ↔
  ∃ s : Student, isVegetarian s ∧ eatsAtCafeteria s :=
by sorry


end NUMINAMATH_CALUDE_negation_of_no_vegetarian_students_eat_at_cafeteria_l2869_286968


namespace NUMINAMATH_CALUDE_middle_school_eight_total_games_l2869_286949

/-- Represents a basketball conference -/
structure BasketballConference where
  numTeams : ℕ
  intraConferenceGamesPerPair : ℕ
  nonConferenceGamesPerTeam : ℕ

/-- Calculate the total number of games in a season for a given basketball conference -/
def totalGamesInSeason (conf : BasketballConference) : ℕ :=
  let intraConferenceGames := conf.numTeams.choose 2 * conf.intraConferenceGamesPerPair
  let nonConferenceGames := conf.numTeams * conf.nonConferenceGamesPerTeam
  intraConferenceGames + nonConferenceGames

/-- The "Middle School Eight" basketball conference -/
def middleSchoolEight : BasketballConference :=
  { numTeams := 8
  , intraConferenceGamesPerPair := 2
  , nonConferenceGamesPerTeam := 4 }

theorem middle_school_eight_total_games :
  totalGamesInSeason middleSchoolEight = 88 := by
  sorry


end NUMINAMATH_CALUDE_middle_school_eight_total_games_l2869_286949


namespace NUMINAMATH_CALUDE_point_ordering_on_reciprocal_function_l2869_286903

/-- Given points on the graph of y = k/x where k > 0, prove a < c < b -/
theorem point_ordering_on_reciprocal_function (k a b c : ℝ) : 
  k > 0 → 
  a * (-2) = k → 
  b * 2 = k → 
  c * 3 = k → 
  a < c ∧ c < b := by
  sorry

end NUMINAMATH_CALUDE_point_ordering_on_reciprocal_function_l2869_286903


namespace NUMINAMATH_CALUDE_absolute_value_of_h_l2869_286915

theorem absolute_value_of_h (h : ℝ) : 
  (∃ x y : ℝ, x^2 - 4*h*x = 8 ∧ y^2 - 4*h*y = 8 ∧ x^2 + y^2 = 80) → 
  |h| = 2 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_of_h_l2869_286915


namespace NUMINAMATH_CALUDE_exp_gt_one_plus_x_l2869_286925

theorem exp_gt_one_plus_x (x : ℝ) (h : x ≠ 0) : Real.exp x > 1 + x := by
  sorry

end NUMINAMATH_CALUDE_exp_gt_one_plus_x_l2869_286925


namespace NUMINAMATH_CALUDE_smallest_three_digit_sum_10_l2869_286960

/-- Given a three-digit number with distinct digits x, y, and z, where x + y + z = 10,
    and x < y < z, the smallest such number is 127. -/
theorem smallest_three_digit_sum_10 (x y z : ℕ) 
  (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (h_sum : x + y + z = 10)
  (h_order : x < y ∧ y < z)
  (h_three_digit : 100 ≤ 100*x + 10*y + z ∧ 100*x + 10*y + z < 1000) :
  100*x + 10*y + z = 127 := by
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_sum_10_l2869_286960


namespace NUMINAMATH_CALUDE_point_location_l2869_286967

theorem point_location (m : ℝ) :
  (m < 0 ∧ 1 > 0) →  -- P (m, 1) is in the second quadrant
  (-m > 0 ∧ 0 = 0)   -- Q (-m, 0) is on the positive half of the x-axis
  := by sorry

end NUMINAMATH_CALUDE_point_location_l2869_286967


namespace NUMINAMATH_CALUDE_swimming_speed_problem_l2869_286947

/-- Given a person who swims for 2 hours at speed S and runs for 1 hour at speed 4S, 
    if the total distance covered is 12 miles, then S must equal 2 miles per hour. -/
theorem swimming_speed_problem (S : ℝ) : 
  (2 * S + 1 * (4 * S) = 12) → S = 2 := by
  sorry

end NUMINAMATH_CALUDE_swimming_speed_problem_l2869_286947


namespace NUMINAMATH_CALUDE_largest_inscribed_circle_l2869_286980

/-- The plane region defined by the given inequalities -/
def PlaneRegion : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ 4 * p.1 + 3 * p.2 - 12 ≤ 0}

/-- The circle with center (1,1) and radius 1 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 1)^2 = 1}

/-- The circle is inscribed in the plane region -/
def IsInscribed (c : Set (ℝ × ℝ)) (r : Set (ℝ × ℝ)) : Prop :=
  c ⊆ r ∧ ∃ p q s : ℝ × ℝ, p ∈ c ∧ p ∈ r ∧ q ∈ c ∧ q ∈ r ∧ s ∈ c ∧ s ∈ r ∧
    p.1 = 0 ∧ q.2 = 0 ∧ 4 * s.1 + 3 * s.2 = 12

/-- The circle is the largest inscribed circle -/
theorem largest_inscribed_circle :
  IsInscribed Circle PlaneRegion ∧
  ∀ c : Set (ℝ × ℝ), IsInscribed c PlaneRegion → MeasureTheory.volume c ≤ MeasureTheory.volume Circle :=
by sorry

end NUMINAMATH_CALUDE_largest_inscribed_circle_l2869_286980


namespace NUMINAMATH_CALUDE_base6_subtraction_l2869_286974

-- Define a function to convert a list of digits in base 6 to a natural number
def base6ToNat (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 6 * acc + d) 0

-- Define a function to convert a natural number to a list of digits in base 6
def natToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
    aux n []

-- State the theorem
theorem base6_subtraction :
  let a := base6ToNat [5, 5, 5]
  let b := base6ToNat [5, 5]
  let c := base6ToNat [2, 0, 2]
  let result := base6ToNat [6, 1, 4]
  (a + b) - c = result := by sorry

end NUMINAMATH_CALUDE_base6_subtraction_l2869_286974


namespace NUMINAMATH_CALUDE_triangle_shape_l2869_286984

theorem triangle_shape (a b : ℝ) (A B : ℝ) (hA : 0 < A ∧ A < π) (hB : 0 < B ∧ B < π) 
  (h : a * Real.cos A = b * Real.cos B) :
  A = B ∨ A + B = π / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_shape_l2869_286984


namespace NUMINAMATH_CALUDE_triangle_area_l2869_286988

/-- The area of the triangle bounded by the x-axis and two lines -/
theorem triangle_area (line1 line2 : ℝ × ℝ → ℝ) : 
  (line1 = fun (x, y) ↦ x - 2*y - 4) →
  (line2 = fun (x, y) ↦ 2*x + y - 5) →
  (∃ x₁ x₂ y : ℝ, 
    line1 (x₁, 0) = 0 ∧ 
    line2 (x₂, 0) = 0 ∧ 
    line1 (x₁, y) = 0 ∧ 
    line2 (x₁, y) = 0 ∧ 
    y > 0) →
  (1/2 * (x₁ - x₂) * y = 9/20) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l2869_286988
