import Mathlib

namespace NUMINAMATH_CALUDE_train_average_speed_l1839_183901

theorem train_average_speed (distance : ℝ) (time : ℝ) (speed : ℝ) :
  distance = 80 →
  time = 8 →
  speed = distance / time →
  speed = 10 :=
by sorry

end NUMINAMATH_CALUDE_train_average_speed_l1839_183901


namespace NUMINAMATH_CALUDE_probability_reach_target_l1839_183932

-- Define the step type
inductive Step
| Left
| Right
| Up
| Down

-- Define the position type
structure Position :=
  (x : Int) (y : Int)

-- Define the function to take a step
def takeStep (pos : Position) (step : Step) : Position :=
  match step with
  | Step.Left  => ⟨pos.x - 1, pos.y⟩
  | Step.Right => ⟨pos.x + 1, pos.y⟩
  | Step.Up    => ⟨pos.x, pos.y + 1⟩
  | Step.Down  => ⟨pos.x, pos.y - 1⟩

-- Define the probability of a single step
def stepProbability : ℚ := 1/4

-- Define the function to check if a position is (3,1)
def isTarget (pos : Position) : Bool :=
  pos.x = 3 ∧ pos.y = 1

-- Define the theorem
theorem probability_reach_target :
  ∃ (paths : Finset (List Step)),
    (∀ path ∈ paths, path.length ≤ 8) ∧
    (∀ path ∈ paths, isTarget (path.foldl takeStep ⟨0, 0⟩)) ∧
    (paths.card : ℚ) * stepProbability ^ 8 = 7/128 :=
sorry

end NUMINAMATH_CALUDE_probability_reach_target_l1839_183932


namespace NUMINAMATH_CALUDE_assignment_methods_count_l1839_183959

def number_of_teachers : ℕ := 5
def number_of_question_types : ℕ := 3

/- Define a function that calculates the number of ways to assign teachers to question types -/
def assignment_methods : ℕ := sorry

/- Theorem stating that the number of assignment methods is 150 -/
theorem assignment_methods_count : assignment_methods = 150 := by sorry

end NUMINAMATH_CALUDE_assignment_methods_count_l1839_183959


namespace NUMINAMATH_CALUDE_bernie_postcard_transaction_l1839_183927

theorem bernie_postcard_transaction (initial_postcards : ℕ) 
  (sell_price : ℕ) (buy_price : ℕ) : 
  initial_postcards = 18 → 
  sell_price = 15 → 
  buy_price = 5 → 
  (initial_postcards / 2 * sell_price) / buy_price = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_bernie_postcard_transaction_l1839_183927


namespace NUMINAMATH_CALUDE_game_draw_probability_l1839_183983

theorem game_draw_probability (p_win p_not_lose : ℝ) 
  (h_win : p_win = 0.3) 
  (h_not_lose : p_not_lose = 0.8) : 
  p_not_lose - p_win = 0.5 := by
sorry

end NUMINAMATH_CALUDE_game_draw_probability_l1839_183983


namespace NUMINAMATH_CALUDE_geometry_statements_l1839_183949

-- Define the types for lines and planes
def Line : Type := sorry
def Plane : Type := sorry

-- Define the relationships between lines and planes
def parallel (l1 l2 : Line) : Prop := sorry
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def contained_in (l : Line) (p : Plane) : Prop := sorry
def parallel_plane (p1 p2 : Plane) : Prop := sorry
def perpendicular_plane (p1 p2 : Plane) : Prop := sorry
def skew (l1 l2 : Line) : Prop := sorry

theorem geometry_statements 
  (a b : Line) (α β : Plane) : 
  -- Statement 2
  (perpendicular a b ∧ perpendicular a α ∧ ¬contained_in b α → parallel b α) ∧
  -- Statement 3
  (perpendicular_plane α β ∧ perpendicular a α ∧ perpendicular b β → perpendicular a b) ∧
  -- Statement 1 (not necessarily true)
  ¬(parallel a b ∧ contained_in b α → parallel a α ∨ contained_in a α) ∧
  -- Statement 4 (not necessarily true)
  ¬(skew a b ∧ contained_in a α ∧ contained_in b β → parallel_plane α β) :=
sorry

end NUMINAMATH_CALUDE_geometry_statements_l1839_183949


namespace NUMINAMATH_CALUDE_floor_sqrt_150_l1839_183980

theorem floor_sqrt_150 : ⌊Real.sqrt 150⌋ = 12 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_150_l1839_183980


namespace NUMINAMATH_CALUDE_max_difference_when_a_is_one_sum_geq_three_iff_abs_a_geq_one_l1839_183977

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |x - 2*a|
def g (a x : ℝ) : ℝ := |x + a|

-- Part 1
theorem max_difference_when_a_is_one :
  ∃ (m : ℝ), ∀ (x : ℝ), f 1 x - g 1 x ≤ m ∧ ∃ (y : ℝ), f 1 y - g 1 y = m ∧ m = 3 :=
sorry

-- Part 2
theorem sum_geq_three_iff_abs_a_geq_one (a : ℝ) :
  (∀ x : ℝ, f a x + g a x ≥ 3) ↔ |a| ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_max_difference_when_a_is_one_sum_geq_three_iff_abs_a_geq_one_l1839_183977


namespace NUMINAMATH_CALUDE_special_function_is_negation_l1839_183909

/-- A function satisfying the given functional equation -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x - y) = f x + f (f y - f (-x)) + x

/-- The main theorem: if f satisfies the functional equation, then f(x) = -x for all x -/
theorem special_function_is_negation (f : ℝ → ℝ) (h : special_function f) :
  ∀ x : ℝ, f x = -x :=
sorry

end NUMINAMATH_CALUDE_special_function_is_negation_l1839_183909


namespace NUMINAMATH_CALUDE_cycling_trip_tailwind_time_l1839_183997

theorem cycling_trip_tailwind_time 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (speed_with_tailwind : ℝ) 
  (speed_against_wind : ℝ) 
  (h1 : total_distance = 150) 
  (h2 : total_time = 12) 
  (h3 : speed_with_tailwind = 15) 
  (h4 : speed_against_wind = 10) : 
  ∃ (time_with_tailwind : ℝ), 
    time_with_tailwind = 6 ∧ 
    speed_with_tailwind * time_with_tailwind + 
    speed_against_wind * (total_time - time_with_tailwind) = total_distance := by
  sorry

end NUMINAMATH_CALUDE_cycling_trip_tailwind_time_l1839_183997


namespace NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l1839_183950

/-- The decimal representation of 0.7888... -/
def repeating_decimal : ℚ := 0.7 + (8 / 9) / 10

theorem repeating_decimal_as_fraction :
  repeating_decimal = 71 / 90 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l1839_183950


namespace NUMINAMATH_CALUDE_leo_marbles_l1839_183917

theorem leo_marbles (total_marbles : ℕ) (marbles_per_pack : ℕ) 
  (manny_fraction : ℚ) (neil_fraction : ℚ) :
  total_marbles = 400 →
  marbles_per_pack = 10 →
  manny_fraction = 1/4 →
  neil_fraction = 1/8 →
  (total_marbles / marbles_per_pack : ℚ) * (1 - manny_fraction - neil_fraction) = 25 := by
  sorry

end NUMINAMATH_CALUDE_leo_marbles_l1839_183917


namespace NUMINAMATH_CALUDE_integer_divisibility_problem_l1839_183989

theorem integer_divisibility_problem (n : ℤ) :
  (5 ∣ (3 * n - 2)) ∧ (7 ∣ (2 * n + 1)) ↔ ∃ m : ℤ, n = 35 * m + 24 := by
  sorry

end NUMINAMATH_CALUDE_integer_divisibility_problem_l1839_183989


namespace NUMINAMATH_CALUDE_express_delivery_growth_rate_l1839_183947

theorem express_delivery_growth_rate 
  (initial_revenue : ℝ)
  (final_revenue : ℝ)
  (years : ℕ)
  (h1 : initial_revenue = 400)
  (h2 : final_revenue = 576)
  (h3 : years = 2) :
  ∃ (growth_rate : ℝ), 
    growth_rate = 0.2 ∧ 
    initial_revenue * (1 + growth_rate) ^ years = final_revenue :=
sorry

end NUMINAMATH_CALUDE_express_delivery_growth_rate_l1839_183947


namespace NUMINAMATH_CALUDE_michelle_crayons_l1839_183900

/-- The number of crayons Michelle has -/
def total_crayons (num_boxes : ℕ) (crayons_per_box : ℕ) : ℕ :=
  num_boxes * crayons_per_box

/-- Proof that Michelle has 35 crayons -/
theorem michelle_crayons : total_crayons 7 5 = 35 := by
  sorry

end NUMINAMATH_CALUDE_michelle_crayons_l1839_183900


namespace NUMINAMATH_CALUDE_product_of_square_roots_l1839_183981

theorem product_of_square_roots (x y z : ℝ) (hx : x = 75) (hy : y = 48) (hz : z = 3) :
  Real.sqrt x * Real.sqrt y * Real.sqrt z = 60 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_product_of_square_roots_l1839_183981


namespace NUMINAMATH_CALUDE_expression_value_l1839_183938

/-- Given that px³ + qx + 3 = 2005 when x = 3, prove that px³ + qx + 3 = -1999 when x = -3 -/
theorem expression_value (p q : ℝ) : 
  (27 * p + 3 * q + 3 = 2005) → (-27 * p - 3 * q + 3 = -1999) := by
sorry

end NUMINAMATH_CALUDE_expression_value_l1839_183938


namespace NUMINAMATH_CALUDE_lindsey_october_savings_l1839_183902

/-- Represents the amount of money Lindsey saved in October -/
def october_savings : ℕ := 37

/-- Represents Lindsey's savings in September -/
def september_savings : ℕ := 50

/-- Represents Lindsey's savings in November -/
def november_savings : ℕ := 11

/-- Represents the amount Lindsey's mom gave her -/
def mom_gift : ℕ := 25

/-- Represents the cost of the video game -/
def video_game_cost : ℕ := 87

/-- Represents the amount Lindsey had left after buying the video game -/
def remaining_money : ℕ := 36

/-- Represents the condition that Lindsey saved more than $75 -/
def saved_more_than_75 : Prop :=
  september_savings + october_savings + november_savings > 75

theorem lindsey_october_savings : 
  september_savings + october_savings + november_savings + mom_gift - video_game_cost = remaining_money ∧
  saved_more_than_75 :=
sorry

end NUMINAMATH_CALUDE_lindsey_october_savings_l1839_183902


namespace NUMINAMATH_CALUDE_max_min_x_plus_reciprocal_l1839_183971

theorem max_min_x_plus_reciprocal (x : ℝ) (h : 12 = x^2 + 1/x^2) :
  (∀ y : ℝ, y ≠ 0 → 12 = y^2 + 1/y^2 → x + 1/x ≤ Real.sqrt 14) ∧
  (∀ y : ℝ, y ≠ 0 → 12 = y^2 + 1/y^2 → -Real.sqrt 14 ≤ x + 1/x) :=
by sorry

end NUMINAMATH_CALUDE_max_min_x_plus_reciprocal_l1839_183971


namespace NUMINAMATH_CALUDE_find_h_l1839_183924

-- Define the two quadratic functions
def f (h j x : ℝ) : ℝ := 4 * (x - h)^2 + j
def g (h k x : ℝ) : ℝ := 3 * (x - h)^2 + k

-- State the theorem
theorem find_h : 
  ∃ (h j k : ℝ),
    (f h j 0 = 2024) ∧ 
    (g h k 0 = 2025) ∧
    (∃ (x₁ x₂ y₁ y₂ : ℤ), x₁ > 0 ∧ x₂ > 0 ∧ y₁ > 0 ∧ y₂ > 0 ∧ 
      f h j (x₁ : ℝ) = 0 ∧ f h j (x₂ : ℝ) = 0 ∧
      g h k (y₁ : ℝ) = 0 ∧ g h k (y₂ : ℝ) = 0) →
    h = 22.5 := by
  sorry


end NUMINAMATH_CALUDE_find_h_l1839_183924


namespace NUMINAMATH_CALUDE_square_of_105_l1839_183903

theorem square_of_105 : (105 : ℕ)^2 = 11025 := by
  sorry

end NUMINAMATH_CALUDE_square_of_105_l1839_183903


namespace NUMINAMATH_CALUDE_equation_two_roots_l1839_183967

-- Define the equation
def equation (x k : ℂ) : Prop :=
  x / (x + 3) + x / (x + 4) = k * x

-- Define the set of valid k values
def valid_k_values : Set ℂ :=
  {0, 7/12, Complex.I, -Complex.I}

-- Theorem statement
theorem equation_two_roots (k : ℂ) :
  (∃! (r₁ r₂ : ℂ), r₁ ≠ r₂ ∧ equation r₁ k ∧ equation r₂ k) ↔ k ∈ valid_k_values :=
sorry

end NUMINAMATH_CALUDE_equation_two_roots_l1839_183967


namespace NUMINAMATH_CALUDE_curve_equation_l1839_183988

/-- Given a curve of the form ax^2 + by^2 = 2 passing through the points (0, 5/3) and (1, 1),
    prove that its equation is 16/25 * x^2 + 9/25 * y^2 = 1. -/
theorem curve_equation (a b : ℝ) (h1 : a * 0^2 + b * (5/3)^2 = 2) (h2 : a * 1^2 + b * 1^2 = 2) :
  ∃ (x y : ℝ), 16/25 * x^2 + 9/25 * y^2 = 1 ↔ a * x^2 + b * y^2 = 2 := by
sorry

end NUMINAMATH_CALUDE_curve_equation_l1839_183988


namespace NUMINAMATH_CALUDE_square_diff_product_plus_square_l1839_183923

theorem square_diff_product_plus_square (a b : ℝ) 
  (h1 : a - b = 5) 
  (h2 : a * b = 2) : 
  a^2 - a*b + b^2 = 27 := by
  sorry

end NUMINAMATH_CALUDE_square_diff_product_plus_square_l1839_183923


namespace NUMINAMATH_CALUDE_painted_cube_probability_l1839_183942

/-- Represents a 5x5x5 cube with three adjacent faces painted -/
structure PaintedCube :=
  (size : ℕ)
  (total_cubes : ℕ)
  (painted_faces : ℕ)

/-- Calculates the number of unit cubes with exactly three painted faces -/
def three_painted_faces (cube : PaintedCube) : ℕ := sorry

/-- Calculates the number of unit cubes with exactly one painted face -/
def one_painted_face (cube : PaintedCube) : ℕ := sorry

/-- Calculates the total number of ways to choose two unit cubes -/
def total_choices (cube : PaintedCube) : ℕ := sorry

/-- Calculates the number of ways to choose one cube with three painted faces and one with one painted face -/
def favorable_choices (cube : PaintedCube) : ℕ := sorry

/-- The main theorem stating the probability -/
theorem painted_cube_probability (cube : PaintedCube) 
  (h1 : cube.size = 5)
  (h2 : cube.total_cubes = 125)
  (h3 : cube.painted_faces = 3) :
  (favorable_choices cube : ℚ) / (total_choices cube : ℚ) = 8 / 235 := by sorry

end NUMINAMATH_CALUDE_painted_cube_probability_l1839_183942


namespace NUMINAMATH_CALUDE_red_balloons_total_l1839_183960

/-- The number of red balloons Sara has -/
def sara_red : ℕ := 31

/-- The number of red balloons Sandy has -/
def sandy_red : ℕ := 24

/-- The total number of red balloons Sara and Sandy have -/
def total_red : ℕ := sara_red + sandy_red

theorem red_balloons_total : total_red = 55 := by
  sorry

end NUMINAMATH_CALUDE_red_balloons_total_l1839_183960


namespace NUMINAMATH_CALUDE_smallest_common_multiple_of_5_and_8_l1839_183962

theorem smallest_common_multiple_of_5_and_8 : 
  ∃ (n : ℕ), n > 0 ∧ Even n ∧ 5 ∣ n ∧ 8 ∣ n ∧ ∀ (m : ℕ), m > 0 → Even m → 5 ∣ m → 8 ∣ m → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_of_5_and_8_l1839_183962


namespace NUMINAMATH_CALUDE_point_outside_circle_l1839_183952

def imaginary_unit : ℂ := Complex.I

theorem point_outside_circle (a b : ℝ) (h : a + b * imaginary_unit = (2 + imaginary_unit) / (1 - imaginary_unit)) :
  a^2 + b^2 > 2 := by sorry

end NUMINAMATH_CALUDE_point_outside_circle_l1839_183952


namespace NUMINAMATH_CALUDE_expression_simplification_l1839_183982

theorem expression_simplification (x : ℝ) (h : x = 5) :
  (2 / (x^2 - 2*x) - (x - 6) / (x^2 - 4*x + 4) / ((x - 6) / (x - 2))) = -1/5 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l1839_183982


namespace NUMINAMATH_CALUDE_sum_mod_nine_l1839_183972

theorem sum_mod_nine : (9023 + 9024 + 9025 + 9026 + 9027) % 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_nine_l1839_183972


namespace NUMINAMATH_CALUDE_eugenes_living_room_length_l1839_183965

/-- Represents the properties of a rectangular room --/
structure RectangularRoom where
  width : ℝ
  area : ℝ
  length : ℝ

/-- Theorem stating the length of Eugene's living room --/
theorem eugenes_living_room_length (room : RectangularRoom)
  (h1 : room.width = 14)
  (h2 : room.area = 215.6)
  (h3 : room.area = room.length * room.width) :
  room.length = 15.4 := by
  sorry

end NUMINAMATH_CALUDE_eugenes_living_room_length_l1839_183965


namespace NUMINAMATH_CALUDE_find_divisor_l1839_183946

theorem find_divisor (dividend quotient remainder divisor : ℕ) : 
  dividend = 15968 →
  quotient = 89 →
  remainder = 37 →
  dividend = divisor * quotient + remainder →
  divisor = 179 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l1839_183946


namespace NUMINAMATH_CALUDE_square_perimeter_ratio_l1839_183974

theorem square_perimeter_ratio (a b : ℝ) (h : a > 0) (k : b > 0) (area_ratio : a^2 / b^2 = 49 / 64) :
  a / b = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_ratio_l1839_183974


namespace NUMINAMATH_CALUDE_expression_simplification_l1839_183964

theorem expression_simplification :
  3 + Real.sqrt 3 + (1 / (3 + Real.sqrt 3)) + (1 / (Real.sqrt 3 - 3)) = 3 + (2 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1839_183964


namespace NUMINAMATH_CALUDE_line_equation_l1839_183919

/-- A line passing through point A(1,4) with the sum of its intercepts on the two axes equal to zero -/
structure LineWithZeroSumIntercepts where
  /-- The slope of the line -/
  slope : ℝ
  /-- The y-intercept of the line -/
  y_intercept : ℝ
  /-- The line passes through point A(1,4) -/
  passes_through_A : 4 = slope * 1 + y_intercept
  /-- The sum of intercepts on the two axes is zero -/
  zero_sum_intercepts : (-y_intercept / slope) + y_intercept = 0

/-- The equation of the line is either 4x - y = 0 or x - y + 3 = 0 -/
theorem line_equation (l : LineWithZeroSumIntercepts) :
  (l.slope = 4 ∧ l.y_intercept = 0) ∨ (l.slope = 1 ∧ l.y_intercept = 3) :=
sorry

end NUMINAMATH_CALUDE_line_equation_l1839_183919


namespace NUMINAMATH_CALUDE_trapezoid_sides_l1839_183921

/-- Proves that a trapezoid with given area, height, and difference between parallel sides has specific lengths for its parallel sides -/
theorem trapezoid_sides (area : ℝ) (height : ℝ) (side_diff : ℝ) 
  (h_area : area = 594) 
  (h_height : height = 22) 
  (h_side_diff : side_diff = 6) :
  ∃ (a b : ℝ), 
    (a + b) * height / 2 = area ∧ 
    a - b = side_diff ∧ 
    a = 30 ∧ 
    b = 24 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_sides_l1839_183921


namespace NUMINAMATH_CALUDE_tree_distance_l1839_183905

theorem tree_distance (n : ℕ) (d : ℝ) (h1 : n = 8) (h2 : d = 80) :
  let distance_between (i j : ℕ) := d * (j - i) / 4
  distance_between 1 n = 140 := by
  sorry

end NUMINAMATH_CALUDE_tree_distance_l1839_183905


namespace NUMINAMATH_CALUDE_equation_solution_l1839_183912

theorem equation_solution (a b c : ℝ) (h : 1 / a - 1 / b = 2 / c) : c = a * b * (b - a) / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1839_183912


namespace NUMINAMATH_CALUDE_curve_symmetry_condition_l1839_183937

/-- Given a curve y = (mx + n) / (tx + u) symmetric about y = x, prove m - u = 0 -/
theorem curve_symmetry_condition 
  (m n t u : ℝ) 
  (hm : m ≠ 0) (hn : n ≠ 0) (ht : t ≠ 0) (hu : u ≠ 0)
  (h_symmetry : ∀ x y : ℝ, y = (m * x + n) / (t * x + u) ↔ x = (m * y + n) / (t * y + u)) :
  m - u = 0 := by
  sorry

end NUMINAMATH_CALUDE_curve_symmetry_condition_l1839_183937


namespace NUMINAMATH_CALUDE_min_value_theorem_l1839_183928

theorem min_value_theorem (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  (1 / (a + 1) + 4 / (b + 1)) ≥ 9/4 ∧
  ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ a₀ + b₀ = 2 ∧ 1 / (a₀ + 1) + 4 / (b₀ + 1) = 9/4 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1839_183928


namespace NUMINAMATH_CALUDE_tom_waits_six_months_l1839_183910

/-- Represents Tom's medication and doctor visit costs --/
structure MedicationCosts where
  pills_per_day : ℕ
  doctor_visit_cost : ℕ
  pill_cost : ℕ
  insurance_coverage : ℚ
  total_annual_cost : ℕ

/-- Calculates the number of months between doctor visits --/
def months_between_visits (costs : MedicationCosts) : ℚ :=
  let annual_medication_cost := costs.pills_per_day * 365 * costs.pill_cost * (1 - costs.insurance_coverage)
  let annual_doctor_cost := costs.total_annual_cost - annual_medication_cost
  let visits_per_year := annual_doctor_cost / costs.doctor_visit_cost
  12 / visits_per_year

/-- Theorem stating that Tom waits 6 months between doctor visits --/
theorem tom_waits_six_months (costs : MedicationCosts) 
  (h1 : costs.pills_per_day = 2)
  (h2 : costs.doctor_visit_cost = 400)
  (h3 : costs.pill_cost = 5)
  (h4 : costs.insurance_coverage = 4/5)
  (h5 : costs.total_annual_cost = 1530) :
  months_between_visits costs = 6 := by
  sorry


end NUMINAMATH_CALUDE_tom_waits_six_months_l1839_183910


namespace NUMINAMATH_CALUDE_glasses_cost_glasses_cost_proof_l1839_183993

/-- Calculate the total cost of glasses after discounts -/
theorem glasses_cost (frame_cost lens_cost : ℝ) 
  (insurance_coverage : ℝ) (frame_coupon : ℝ) : ℝ :=
  let discounted_lens_cost := lens_cost * (1 - insurance_coverage)
  let discounted_frame_cost := frame_cost - frame_coupon
  discounted_lens_cost + discounted_frame_cost

/-- Prove that the total cost of glasses after discounts is $250 -/
theorem glasses_cost_proof :
  glasses_cost 200 500 0.8 50 = 250 := by
  sorry

end NUMINAMATH_CALUDE_glasses_cost_glasses_cost_proof_l1839_183993


namespace NUMINAMATH_CALUDE_price_is_400_l1839_183978

/-- The price per phone sold by Aliyah and Vivienne -/
def price_per_phone (vivienne_phones : ℕ) (aliyah_extra_phones : ℕ) (total_revenue : ℕ) : ℚ :=
  total_revenue / (vivienne_phones + (vivienne_phones + aliyah_extra_phones))

/-- Theorem stating that the price per phone is $400 -/
theorem price_is_400 :
  price_per_phone 40 10 36000 = 400 := by
  sorry

end NUMINAMATH_CALUDE_price_is_400_l1839_183978


namespace NUMINAMATH_CALUDE_book_pages_l1839_183914

/-- The number of pages Frank reads per day -/
def pages_per_day : ℕ := 22

/-- The number of days it took Frank to finish the book -/
def days_to_finish : ℕ := 569

/-- The total number of pages in the book -/
def total_pages : ℕ := pages_per_day * days_to_finish

theorem book_pages : total_pages = 12518 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_l1839_183914


namespace NUMINAMATH_CALUDE_work_rate_proof_l1839_183948

/-- Given work rates for individuals and pairs, prove the work rate for a specific pair -/
theorem work_rate_proof 
  (c_rate : ℚ)
  (bc_rate : ℚ)
  (ca_rate : ℚ)
  (h1 : c_rate = 1 / 24)
  (h2 : bc_rate = 1 / 3)
  (h3 : ca_rate = 1 / 4) :
  ∃ (ab_rate : ℚ), ab_rate = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_work_rate_proof_l1839_183948


namespace NUMINAMATH_CALUDE_newspaper_conference_overlap_l1839_183929

theorem newspaper_conference_overlap (total : ℕ) (writers : ℕ) (editors : ℕ) 
  (h_total : total = 110)
  (h_writers : writers = 45)
  (h_editors : editors ≥ 39)
  (h_max_overlap : ∀ overlap : ℕ, overlap ≤ 26)
  (h_neither : ∀ overlap : ℕ, 2 * overlap = total - writers - editors + overlap) :
  ∃ overlap : ℕ, overlap = 26 ∧ 
    writers + editors - overlap + 2 * overlap = total ∧
    overlap = total - writers - editors + overlap :=
by sorry

end NUMINAMATH_CALUDE_newspaper_conference_overlap_l1839_183929


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_length_l1839_183992

/-- An isosceles triangle with base 8 and side difference 2 has sides of length 10 or 6 -/
theorem isosceles_triangle_side_length (AC BC : ℝ) : 
  BC = 8 → 
  |AC - BC| = 2 → 
  (AC = 10 ∨ AC = 6) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_side_length_l1839_183992


namespace NUMINAMATH_CALUDE_subtraction_base_8_to_10_l1839_183944

def base_8_to_10 (n : ℕ) : ℕ :=
  let digits := n.digits 8
  (List.range digits.length).foldl (λ acc i => acc + digits[i]! * (8 ^ i)) 0

theorem subtraction_base_8_to_10 :
  base_8_to_10 (4725 - 2367) = 1246 :=
sorry

end NUMINAMATH_CALUDE_subtraction_base_8_to_10_l1839_183944


namespace NUMINAMATH_CALUDE_only_football_fans_l1839_183958

/-- Represents the number of people in different categories in a class --/
structure ClassPreferences where
  total : Nat
  baseballAndFootball : Nat
  onlyBaseball : Nat
  neitherSport : Nat
  onlyFootball : Nat

/-- The theorem stating the number of people who only like football --/
theorem only_football_fans (c : ClassPreferences) : c.onlyFootball = 3 :=
  by
  have h1 : c.total = 16 := by sorry
  have h2 : c.baseballAndFootball = 5 := by sorry
  have h3 : c.onlyBaseball = 2 := by sorry
  have h4 : c.neitherSport = 6 := by sorry
  have h5 : c.total = c.baseballAndFootball + c.onlyBaseball + c.onlyFootball + c.neitherSport := by sorry
  sorry

#check only_football_fans

end NUMINAMATH_CALUDE_only_football_fans_l1839_183958


namespace NUMINAMATH_CALUDE_functional_equation_unique_solution_l1839_183953

open Set

theorem functional_equation_unique_solution
  (f : ℝ → ℝ) (a b : ℝ) :
  (0 < a) →
  (0 < b) →
  (∀ x, 0 ≤ x → 0 ≤ f x) →
  (∀ x, 0 ≤ x → f (f x) + a * f x = b * (a + b) * x) →
  (∀ x, 0 ≤ x → f x = b * x) :=
sorry

end NUMINAMATH_CALUDE_functional_equation_unique_solution_l1839_183953


namespace NUMINAMATH_CALUDE_system_of_equations_l1839_183973

theorem system_of_equations (x y c d : ℝ) 
  (eq1 : 8 * x - 5 * y = c)
  (eq2 : 12 * y - 18 * x = d)
  (x_nonzero : x ≠ 0)
  (y_nonzero : y ≠ 0)
  (d_nonzero : d ≠ 0) :
  c / d = -16 / 27 := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_l1839_183973


namespace NUMINAMATH_CALUDE_correct_calculation_l1839_183955

theorem correct_calculation : (-4) * (-3) * (-5) = -60 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1839_183955


namespace NUMINAMATH_CALUDE_inequality_solution_l1839_183908

theorem inequality_solution (x : ℝ) : (2 - x < 1) ↔ (x > 1) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1839_183908


namespace NUMINAMATH_CALUDE_percentage_difference_l1839_183911

theorem percentage_difference (x y : ℝ) (h : x = 8 * y) :
  (x - y) / x * 100 = 87.5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l1839_183911


namespace NUMINAMATH_CALUDE_divisibility_property_l1839_183975

theorem divisibility_property (m n : ℕ) (h : 24 ∣ (m * n + 1)) : 24 ∣ (m + n) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l1839_183975


namespace NUMINAMATH_CALUDE_expression_evaluation_l1839_183943

theorem expression_evaluation (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  (x^(2*y) * y^(2*x)) / (y^(2*y) * x^(2*x)) = (x/y)^(2*(y-x)) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1839_183943


namespace NUMINAMATH_CALUDE_teachers_count_l1839_183994

/-- Represents the total number of faculty and students in the school -/
def total_population : ℕ := 2400

/-- Represents the total number of individuals in the sample -/
def sample_size : ℕ := 160

/-- Represents the number of students in the sample -/
def students_in_sample : ℕ := 150

/-- Calculates the number of teachers in the school -/
def number_of_teachers : ℕ :=
  total_population - (total_population * students_in_sample) / sample_size

theorem teachers_count : number_of_teachers = 150 := by
  sorry

end NUMINAMATH_CALUDE_teachers_count_l1839_183994


namespace NUMINAMATH_CALUDE_equal_area_division_l1839_183930

/-- Represents a configuration of five unit squares in the coordinate plane -/
structure SquareConfiguration where
  /-- The x-coordinate of the point where the dividing line intersects the x-axis -/
  c : ℝ

/-- The total area of the square configuration -/
def totalArea : ℝ := 5

/-- The area of the triangle formed by the dividing line -/
def triangleArea (config : SquareConfiguration) : ℝ :=
  2 * (3 - config.c)

/-- The condition for equal division of area -/
def equalAreaCondition (config : SquareConfiguration) : Prop :=
  triangleArea config = totalArea / 2

/-- Theorem stating that the equal area condition is satisfied when c = 1.75 -/
theorem equal_area_division (config : SquareConfiguration) :
  equalAreaCondition config ↔ config.c = 1.75 := by sorry

end NUMINAMATH_CALUDE_equal_area_division_l1839_183930


namespace NUMINAMATH_CALUDE_two_red_one_spade_probability_l1839_183985

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (suits : Nat)
  (cards_per_suit : Nat)
  (red_suits : Nat)
  (black_suits : Nat)

/-- Calculates the probability of drawing two red cards followed by a spade -/
def probability_two_red_one_spade (d : Deck) : Rat :=
  if d.total_cards = 52 ∧ d.suits = 4 ∧ d.cards_per_suit = 13 ∧ d.red_suits = 2 ∧ d.black_suits = 2
  then 13 / 204
  else 0

/-- Theorem stating the probability of drawing two red cards followed by a spade from a standard deck -/
theorem two_red_one_spade_probability (d : Deck) :
  d.total_cards = 52 ∧ d.suits = 4 ∧ d.cards_per_suit = 13 ∧ d.red_suits = 2 ∧ d.black_suits = 2 →
  probability_two_red_one_spade d = 13 / 204 := by
  sorry

end NUMINAMATH_CALUDE_two_red_one_spade_probability_l1839_183985


namespace NUMINAMATH_CALUDE_fixed_point_theorem_tangent_dot_product_range_l1839_183907

-- Define the curves C and M
def C (x y : ℝ) : Prop := y^2 = 4*x
def M (x y : ℝ) : Prop := (x-1)^2 + y^2 = 4 ∧ x ≥ 1

-- Define the line l
def L (m n : ℝ) (x y : ℝ) : Prop := x = m*y + n

-- Define points A and B on curve C and line l
def A_B_on_C_and_L (m n x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  C x₁ y₁ ∧ C x₂ y₂ ∧ L m n x₁ y₁ ∧ L m n x₂ y₂

-- Theorem 1
theorem fixed_point_theorem (m n x₁ y₁ x₂ y₂ : ℝ) :
  A_B_on_C_and_L m n x₁ y₁ x₂ y₂ →
  x₁*x₂ + y₁*y₂ = -4 →
  ∃ (m : ℝ), L m 2 2 0 :=
sorry

-- Theorem 2
theorem tangent_dot_product_range (m n x₁ y₁ x₂ y₂ : ℝ) :
  A_B_on_C_and_L m n x₁ y₁ x₂ y₂ →
  (∃ (x y : ℝ), M x y ∧ L m n x y) →
  (x₁-1)*(x₂-1) + y₁*y₂ ≤ -8 :=
sorry

end NUMINAMATH_CALUDE_fixed_point_theorem_tangent_dot_product_range_l1839_183907


namespace NUMINAMATH_CALUDE_hexagon_around_convex_curve_l1839_183904

/-- A convex curve in a 2D plane -/
structure ConvexCurve where
  -- Add necessary fields/axioms for a convex curve

/-- A hexagon in a 2D plane -/
structure Hexagon where
  -- Add necessary fields for a hexagon (e.g., vertices, sides)

/-- Predicate to check if a hexagon is circumscribed around a convex curve -/
def is_circumscribed (h : Hexagon) (c : ConvexCurve) : Prop :=
  sorry

/-- Predicate to check if all internal angles of a hexagon are equal -/
def has_equal_angles (h : Hexagon) : Prop :=
  sorry

/-- Predicate to check if opposite sides of a hexagon are equal -/
def has_equal_opposite_sides (h : Hexagon) : Prop :=
  sorry

/-- Predicate to check if a hexagon has an axis of symmetry -/
def has_symmetry_axis (h : Hexagon) : Prop :=
  sorry

/-- Theorem: For any convex curve, there exists a circumscribed hexagon with equal angles, 
    equal opposite sides, and an axis of symmetry -/
theorem hexagon_around_convex_curve (c : ConvexCurve) : 
  ∃ h : Hexagon, 
    is_circumscribed h c ∧ 
    has_equal_angles h ∧ 
    has_equal_opposite_sides h ∧ 
    has_symmetry_axis h :=
by
  sorry

end NUMINAMATH_CALUDE_hexagon_around_convex_curve_l1839_183904


namespace NUMINAMATH_CALUDE_income_calculation_l1839_183935

/-- Represents a person's financial situation -/
structure FinancialSituation where
  income : ℕ
  expenditure : ℕ
  savings : ℕ

/-- Proves that given the conditions, the person's income is 10000 -/
theorem income_calculation (f : FinancialSituation) 
  (h1 : f.income * 8 = f.expenditure * 10)  -- income : expenditure = 10 : 8
  (h2 : f.savings = 2000)                   -- savings are 2000
  (h3 : f.income = f.expenditure + f.savings) -- income = expenditure + savings
  : f.income = 10000 := by
  sorry


end NUMINAMATH_CALUDE_income_calculation_l1839_183935


namespace NUMINAMATH_CALUDE_symmetric_points_sum_power_l1839_183934

theorem symmetric_points_sum_power (a b : ℝ) : 
  (∃ (P1 P2 : ℝ × ℝ), 
    P1 = (a - 1, 5) ∧ 
    P2 = (2, b - 1) ∧ 
    P1.1 = P2.1 ∧ 
    P1.2 = -P2.2) →
  (a + b)^2016 = 1 := by
sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_power_l1839_183934


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l1839_183926

/-- Given a triangle ABC where BC = a and AC = b, and a and b are roots of x^2 - 2√3x + 2 = 0,
    prove that the measure of angle C is 2π/3 and the length of AB is √10 -/
theorem triangle_abc_properties (a b : ℝ) (A B C : ℝ) :
  a^2 - 2 * Real.sqrt 3 * a + 2 = 0 →
  b^2 - 2 * Real.sqrt 3 * b + 2 = 0 →
  2 * Real.cos (A + B) = 1 →
  C = 2 * π / 3 ∧ (a^2 + b^2 - 2 * a * b * Real.cos C) = 10 := by
  sorry


end NUMINAMATH_CALUDE_triangle_abc_properties_l1839_183926


namespace NUMINAMATH_CALUDE_width_of_identical_rectangles_l1839_183984

/-- Given six identical rectangles forming a larger rectangle PQRS, prove that the width of each identical rectangle is 30 -/
theorem width_of_identical_rectangles (w : ℝ) : 
  (6 : ℝ) * w^2 = 5400 ∧ 3 * w = 2 * (2 * w) → w = 30 := by
  sorry

end NUMINAMATH_CALUDE_width_of_identical_rectangles_l1839_183984


namespace NUMINAMATH_CALUDE_highest_power_of_three_in_M_l1839_183945

def M : ℕ := sorry

theorem highest_power_of_three_in_M : 
  (∃ k : ℕ, M = 3 * k) ∧ ¬(∃ k : ℕ, M = 9 * k) := by sorry

end NUMINAMATH_CALUDE_highest_power_of_three_in_M_l1839_183945


namespace NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l1839_183940

theorem cubic_roots_sum_cubes (p q r : ℝ) : 
  (3 * p^3 + 4 * p^2 - 200 * p + 5 = 0) →
  (3 * q^3 + 4 * q^2 - 200 * q + 5 = 0) →
  (3 * r^3 + 4 * r^2 - 200 * r + 5 = 0) →
  (p + q + 1)^3 + (q + r + 1)^3 + (r + p + 1)^3 = 24 :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l1839_183940


namespace NUMINAMATH_CALUDE_sarah_bottle_caps_l1839_183939

/-- Given that Sarah initially had 26 bottle caps and now has 29 in total,
    prove that she bought 3 bottle caps. -/
theorem sarah_bottle_caps (initial : ℕ) (total : ℕ) (bought : ℕ) 
    (h1 : initial = 26) 
    (h2 : total = 29) 
    (h3 : total = initial + bought) : bought = 3 := by
  sorry

end NUMINAMATH_CALUDE_sarah_bottle_caps_l1839_183939


namespace NUMINAMATH_CALUDE_combined_girls_avg_is_87_l1839_183925

/-- Represents a high school with average scores -/
structure School where
  boys_avg : ℝ
  girls_avg : ℝ
  combined_avg : ℝ

/-- Represents the combined average score for boys at both schools -/
def combined_boys_avg : ℝ := 73

/-- Theorem: The combined average score for girls at both schools is 87 -/
theorem combined_girls_avg_is_87 
  (lincoln : School)
  (grant : School)
  (h1 : lincoln.boys_avg = 68)
  (h2 : lincoln.girls_avg = 80)
  (h3 : lincoln.combined_avg = 72)
  (h4 : grant.boys_avg = 75)
  (h5 : grant.girls_avg = 88)
  (h6 : grant.combined_avg = 82)
  (h7 : combined_boys_avg = 73) :
  ∃ (combined_girls_avg : ℝ), combined_girls_avg = 87 := by
  sorry


end NUMINAMATH_CALUDE_combined_girls_avg_is_87_l1839_183925


namespace NUMINAMATH_CALUDE_petyas_fruits_l1839_183995

theorem petyas_fruits (total : ℕ) (apples tangerines oranges : ℕ) : 
  total = 20 →
  apples + tangerines + oranges = total →
  tangerines * 6 = apples →
  apples > oranges →
  oranges = 6 :=
by sorry

end NUMINAMATH_CALUDE_petyas_fruits_l1839_183995


namespace NUMINAMATH_CALUDE_workers_wage_increase_l1839_183986

theorem workers_wage_increase (original_wage : ℝ) (increase_percentage : ℝ) (new_wage : ℝ) : 
  increase_percentage = 40 →
  new_wage = 35 →
  new_wage = original_wage * (1 + increase_percentage / 100) →
  original_wage = 25 := by
sorry

end NUMINAMATH_CALUDE_workers_wage_increase_l1839_183986


namespace NUMINAMATH_CALUDE_junior_score_l1839_183951

theorem junior_score (n : ℝ) (junior_score : ℝ) : 
  n > 0 →
  (0.3 * n * junior_score + 0.7 * n * 75) / n = 78 →
  junior_score = 85 := by
sorry

end NUMINAMATH_CALUDE_junior_score_l1839_183951


namespace NUMINAMATH_CALUDE_optimal_profit_is_1368_l1839_183979

/-- Represents the types of apples -/
inductive AppleType
| A
| B
| C

/-- Represents the configuration of cars for each apple type -/
structure CarConfiguration where
  typeA : ℕ
  typeB : ℕ
  typeC : ℕ

/-- The total number of cars -/
def totalCars : ℕ := 40

/-- The total tons of apples -/
def totalTons : ℕ := 200

/-- Returns the tons per car for a given apple type -/
def tonsPerCar (t : AppleType) : ℕ :=
  match t with
  | AppleType.A => 6
  | AppleType.B => 5
  | AppleType.C => 4

/-- Returns the profit per ton for a given apple type -/
def profitPerTon (t : AppleType) : ℕ :=
  match t with
  | AppleType.A => 5
  | AppleType.B => 7
  | AppleType.C => 8

/-- Checks if a car configuration is valid -/
def isValidConfiguration (config : CarConfiguration) : Prop :=
  config.typeA + config.typeB + config.typeC = totalCars ∧
  config.typeA * tonsPerCar AppleType.A + 
  config.typeB * tonsPerCar AppleType.B + 
  config.typeC * tonsPerCar AppleType.C = totalTons ∧
  config.typeA ≥ 4 ∧ config.typeB ≥ 4 ∧ config.typeC ≥ 4

/-- Calculates the profit for a given car configuration -/
def calculateProfit (config : CarConfiguration) : ℕ :=
  config.typeA * tonsPerCar AppleType.A * profitPerTon AppleType.A +
  config.typeB * tonsPerCar AppleType.B * profitPerTon AppleType.B +
  config.typeC * tonsPerCar AppleType.C * profitPerTon AppleType.C

/-- The optimal car configuration -/
def optimalConfig : CarConfiguration :=
  { typeA := 4, typeB := 32, typeC := 4 }

theorem optimal_profit_is_1368 :
  isValidConfiguration optimalConfig ∧
  calculateProfit optimalConfig = 1368 ∧
  ∀ (config : CarConfiguration), 
    isValidConfiguration config → 
    calculateProfit config ≤ calculateProfit optimalConfig :=
by sorry

end NUMINAMATH_CALUDE_optimal_profit_is_1368_l1839_183979


namespace NUMINAMATH_CALUDE_willie_stickers_l1839_183956

theorem willie_stickers (initial_stickers given_away : ℕ) 
  (h1 : initial_stickers = 36)
  (h2 : given_away = 7) : 
  initial_stickers - given_away = 29 := by
  sorry

end NUMINAMATH_CALUDE_willie_stickers_l1839_183956


namespace NUMINAMATH_CALUDE_intersection_M_N_l1839_183998

-- Define set M
def M : Set ℝ := {x | ∃ y, y = Real.sqrt (x - x^2)}

-- Define set N
def N : Set ℝ := {y | ∃ x, y = Real.sin x}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Icc 0 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1839_183998


namespace NUMINAMATH_CALUDE_smallest_number_with_remainders_l1839_183968

theorem smallest_number_with_remainders : ∃! n : ℕ,
  n > 0 ∧
  n % 7 = 2 ∧
  n % 11 = 2 ∧
  n % 13 = 2 ∧
  n % 17 = 3 ∧
  n % 23 = 0 ∧
  n % 5 = 0 ∧
  (∀ m : ℕ, m > 0 ∧
    m % 7 = 2 ∧
    m % 11 = 2 ∧
    m % 13 = 2 ∧
    m % 17 = 3 ∧
    m % 23 = 0 ∧
    m % 5 = 0 → m ≥ n) ∧
  n = 391410 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainders_l1839_183968


namespace NUMINAMATH_CALUDE_parabola_hyperbola_foci_coincide_l1839_183913

/-- The value of n for which the focus of the parabola y^2 = 8x coincides with 
    one of the foci of the hyperbola x^2/3 - y^2/n = 1 -/
theorem parabola_hyperbola_foci_coincide : ∃ n : ℝ,
  (∀ x y : ℝ, y^2 = 8*x → x^2/3 - y^2/n = 1) ∧ 
  (∃ x y : ℝ, y^2 = 8*x ∧ x^2/3 - y^2/n = 1 ∧ x = 2 ∧ y = 0) →
  n = 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_foci_coincide_l1839_183913


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1839_183991

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x > 0}
def B : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 1}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1839_183991


namespace NUMINAMATH_CALUDE_valid_arrangement_has_four_rows_of_seven_l1839_183936

/-- Represents a seating arrangement -/
structure SeatingArrangement where
  rows_of_seven : ℕ
  rows_of_six : ℕ

/-- Checks if a seating arrangement is valid -/
def is_valid_arrangement (s : SeatingArrangement) : Prop :=
  s.rows_of_seven * 7 + s.rows_of_six * 6 = 52

/-- Theorem stating that the valid arrangement has 4 rows of 7 people -/
theorem valid_arrangement_has_four_rows_of_seven :
  ∃ (s : SeatingArrangement), is_valid_arrangement s ∧ s.rows_of_seven = 4 := by
  sorry

end NUMINAMATH_CALUDE_valid_arrangement_has_four_rows_of_seven_l1839_183936


namespace NUMINAMATH_CALUDE_equation_is_linear_l1839_183915

/-- Definition of a linear equation with two variables -/
def is_linear_equation_two_vars (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ), ∀ x y, f x y ↔ a * x + b * y = c

/-- The equation 3x = 2y -/
def equation (x y : ℝ) : Prop := 3 * x = 2 * y

/-- Theorem: The equation 3x = 2y is a linear equation with two variables -/
theorem equation_is_linear : is_linear_equation_two_vars equation := by
  sorry

end NUMINAMATH_CALUDE_equation_is_linear_l1839_183915


namespace NUMINAMATH_CALUDE_james_steak_purchase_l1839_183963

/-- Represents the buy one get one free deal -/
def buyOneGetOneFree (x : ℝ) : ℝ := 2 * x

/-- Represents the price per pound in dollars -/
def pricePerPound : ℝ := 15

/-- Represents the total amount James paid in dollars -/
def totalPaid : ℝ := 150

/-- Theorem stating that James bought 20 pounds of steaks -/
theorem james_steak_purchase :
  ∃ (x : ℝ), x > 0 ∧ x * pricePerPound = totalPaid ∧ buyOneGetOneFree x = 20 :=
by
  sorry


end NUMINAMATH_CALUDE_james_steak_purchase_l1839_183963


namespace NUMINAMATH_CALUDE_student_subtraction_problem_l1839_183969

theorem student_subtraction_problem (x : ℝ) (h : x = 155) :
  ∃! y : ℝ, 2 * x - y = 110 ∧ y = 200 := by
sorry

end NUMINAMATH_CALUDE_student_subtraction_problem_l1839_183969


namespace NUMINAMATH_CALUDE_square_area_ratio_l1839_183931

theorem square_area_ratio (r : ℝ) (h : r > 0) : 
  (4 * r^2) / (2 * r^2) = 2 := by sorry

end NUMINAMATH_CALUDE_square_area_ratio_l1839_183931


namespace NUMINAMATH_CALUDE_carson_counted_six_clouds_l1839_183918

/-- The number of clouds Carson counted that look like funny animals -/
def carson_clouds : ℕ := sorry

/-- The number of clouds Carson's little brother counted that look like dragons -/
def brother_clouds : ℕ := sorry

/-- The total number of clouds counted -/
def total_clouds : ℕ := 24

theorem carson_counted_six_clouds :
  carson_clouds = 6 ∧
  brother_clouds = 3 * carson_clouds ∧
  carson_clouds + brother_clouds = total_clouds :=
sorry

end NUMINAMATH_CALUDE_carson_counted_six_clouds_l1839_183918


namespace NUMINAMATH_CALUDE_difference_of_sum_and_difference_of_squares_l1839_183961

theorem difference_of_sum_and_difference_of_squares (x y : ℝ) 
  (h1 : x + y = 8) (h2 : x^2 - y^2 = 16) : x - y = 2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_sum_and_difference_of_squares_l1839_183961


namespace NUMINAMATH_CALUDE_game_result_l1839_183970

def g (n : ℕ) : ℕ :=
  if n % 3 = 0 then 8
  else if n % 2 = 0 then 3
  else 0

def allie_rolls : List ℕ := [2, 6, 3, 2, 5]
def carl_rolls : List ℕ := [1, 4, 3, 6, 6]

def total_points (rolls : List ℕ) : ℕ :=
  rolls.map g |>.sum

theorem game_result : total_points allie_rolls * total_points carl_rolls = 594 := by
  sorry

end NUMINAMATH_CALUDE_game_result_l1839_183970


namespace NUMINAMATH_CALUDE_geometric_series_relation_l1839_183999

/-- Given real numbers c and d satisfying an infinite geometric series equation,
    prove that another related infinite geometric series equals 3/4. -/
theorem geometric_series_relation (c d : ℝ) 
    (h : (c / d) / (1 - 1 / d) = 6) :
    (c / (c + 2 * d)) / (1 - 1 / (c + 2 * d)) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_relation_l1839_183999


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l1839_183941

theorem equilateral_triangle_perimeter (side_length : ℝ) (h : side_length = 7) :
  3 * side_length = 21 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l1839_183941


namespace NUMINAMATH_CALUDE_potato_shipment_l1839_183906

/-- The initial amount of potatoes shipped in kg -/
def initial_potatoes : ℕ := 6500

/-- The amount of damaged potatoes in kg -/
def damaged_potatoes : ℕ := 150

/-- The weight of each bag of potatoes in kg -/
def bag_weight : ℕ := 50

/-- The price of each bag of potatoes in dollars -/
def bag_price : ℕ := 72

/-- The total revenue from selling the potatoes in dollars -/
def total_revenue : ℕ := 9144

theorem potato_shipment :
  initial_potatoes = 
    (total_revenue / bag_price) * bag_weight + damaged_potatoes :=
by sorry

end NUMINAMATH_CALUDE_potato_shipment_l1839_183906


namespace NUMINAMATH_CALUDE_minimum_value_theorem_l1839_183916

theorem minimum_value_theorem (x y m : ℝ) :
  x > 0 →
  y > 0 →
  (4 / x + 9 / y = m) →
  (∀ a b : ℝ, a > 0 → b > 0 → 4 / a + 9 / b = m → x + y ≤ a + b) →
  x + y = 5 / 6 →
  m = 30 := by
sorry

end NUMINAMATH_CALUDE_minimum_value_theorem_l1839_183916


namespace NUMINAMATH_CALUDE_right_triangle_solution_l1839_183933

theorem right_triangle_solution (A B C : Real) (a b c : ℝ) :
  -- Given conditions
  (A + B + C = π) →  -- Sum of angles in a triangle
  (C = π / 2) →      -- Right angle at C
  (a = Real.sqrt 5) →
  (b = Real.sqrt 15) →
  (a^2 + b^2 = c^2) →  -- Pythagorean theorem
  (Real.tan A = a / b) →  -- Definition of tangent
  -- Conclusions
  (c = 2 * Real.sqrt 5) ∧
  (A = π / 6) ∧  -- 30 degrees in radians
  (B = π / 3) :=  -- 60 degrees in radians
by sorry

end NUMINAMATH_CALUDE_right_triangle_solution_l1839_183933


namespace NUMINAMATH_CALUDE_bus_rows_theorem_l1839_183954

/-- Represents the state of passengers on a bus -/
structure BusState where
  initial : Nat
  first_stop_board : Nat
  first_stop_leave : Nat
  second_stop_board : Nat
  second_stop_leave : Nat
  empty_seats : Nat
  seats_per_row : Nat

/-- Calculates the number of rows on the bus given its state -/
def calculate_rows (state : BusState) : Nat :=
  let total_passengers := state.initial + 
    (state.first_stop_board - state.first_stop_leave) + 
    (state.second_stop_board - state.second_stop_leave)
  let total_seats := total_passengers + state.empty_seats
  total_seats / state.seats_per_row

/-- Theorem stating that given the problem conditions, the bus has 23 rows -/
theorem bus_rows_theorem (state : BusState) 
  (h1 : state.initial = 16)
  (h2 : state.first_stop_board = 15)
  (h3 : state.first_stop_leave = 3)
  (h4 : state.second_stop_board = 17)
  (h5 : state.second_stop_leave = 10)
  (h6 : state.empty_seats = 57)
  (h7 : state.seats_per_row = 4) :
  calculate_rows state = 23 := by
  sorry

#eval calculate_rows {
  initial := 16,
  first_stop_board := 15,
  first_stop_leave := 3,
  second_stop_board := 17,
  second_stop_leave := 10,
  empty_seats := 57,
  seats_per_row := 4
}

end NUMINAMATH_CALUDE_bus_rows_theorem_l1839_183954


namespace NUMINAMATH_CALUDE_common_projection_l1839_183920

def v1 : ℝ × ℝ := (3, 2)
def v2 : ℝ × ℝ := (1, 5)

def is_projection (p : ℝ × ℝ) (v : ℝ × ℝ) (u : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), p = (t * u.1, t * u.2) ∧ (p.1 - v.1) * u.1 + (p.2 - v.2) * u.2 = 0

theorem common_projection :
  ∃ (u : ℝ × ℝ), is_projection v1 v1 u ∧ is_projection v1 v2 u :=
sorry

end NUMINAMATH_CALUDE_common_projection_l1839_183920


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_l1839_183987

/-- The area of an equilateral triangle with altitude 2√3 is 4√3 -/
theorem equilateral_triangle_area (h : ℝ) (altitude_eq : h = 2 * Real.sqrt 3) :
  let side : ℝ := 2 * h / Real.sqrt 3
  let area : ℝ := 1/2 * side * h
  area = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_l1839_183987


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1839_183957

theorem quadratic_inequality_solution : 
  {x : ℝ | x^2 - 5*x + 6 < 0} = Set.Ioo 2 3 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1839_183957


namespace NUMINAMATH_CALUDE_average_of_five_numbers_l1839_183996

theorem average_of_five_numbers (x : ℝ) :
  let numbers := [-4*x, 0, 4*x, 12*x, 20*x]
  (numbers.sum / numbers.length : ℝ) = 6.4 * x := by
sorry

end NUMINAMATH_CALUDE_average_of_five_numbers_l1839_183996


namespace NUMINAMATH_CALUDE_factor_expression_l1839_183976

theorem factor_expression (a : ℝ) : 58 * a^2 + 174 * a = 58 * a * (a + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1839_183976


namespace NUMINAMATH_CALUDE_min_value_theorem_l1839_183990

theorem min_value_theorem (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h_eq : (2 / x) + (3 / y) + (5 / z) = 10) :
  ∀ a b c : ℝ, 0 < a ∧ 0 < b ∧ 0 < c ∧ (2 / a) + (3 / b) + (5 / c) = 10 →
  x^4 * y^3 * z^2 ≤ a^4 * b^3 * c^2 ∧
  x^4 * y^3 * z^2 = 390625 / 1296 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1839_183990


namespace NUMINAMATH_CALUDE_distinct_colorings_l1839_183966

-- Define the symmetry group of the circle
inductive CircleSymmetry
| id : CircleSymmetry
| rot120 : CircleSymmetry
| rot240 : CircleSymmetry
| refl1 : CircleSymmetry
| refl2 : CircleSymmetry
| refl3 : CircleSymmetry

-- Define the coloring function
def Coloring := Fin 3 → Fin 3

-- Define the action of symmetries on colorings
def act (g : CircleSymmetry) (c : Coloring) : Coloring :=
  sorry

-- Define the fixed points under a symmetry
def fixedPoints (g : CircleSymmetry) : Nat :=
  sorry

-- The main theorem
theorem distinct_colorings : 
  (List.sum (List.map fixedPoints [CircleSymmetry.id, CircleSymmetry.rot120, 
    CircleSymmetry.rot240, CircleSymmetry.refl1, CircleSymmetry.refl2, 
    CircleSymmetry.refl3])) / 6 = 10 := by
  sorry

end NUMINAMATH_CALUDE_distinct_colorings_l1839_183966


namespace NUMINAMATH_CALUDE_imaginary_part_of_3_minus_4i_l1839_183922

theorem imaginary_part_of_3_minus_4i :
  Complex.im (3 - 4 * Complex.I) = -4 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_3_minus_4i_l1839_183922
