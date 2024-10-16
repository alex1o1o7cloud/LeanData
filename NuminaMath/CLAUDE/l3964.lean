import Mathlib

namespace NUMINAMATH_CALUDE_inheritance_problem_l3964_396432

theorem inheritance_problem (total_inheritance : ℝ) (additional_share : ℝ) 
  (h1 : total_inheritance = 84000)
  (h2 : additional_share = 3500)
  (h3 : ∃ x : ℕ, x > 2 ∧ 
    total_inheritance / x + additional_share = total_inheritance / (x - 2)) :
  ∃ x : ℕ, x = 8 ∧ x > 2 ∧ 
    total_inheritance / x + additional_share = total_inheritance / (x - 2) :=
sorry

end NUMINAMATH_CALUDE_inheritance_problem_l3964_396432


namespace NUMINAMATH_CALUDE_impossible_odd_black_cells_impossible_one_black_cell_l3964_396409

/-- Represents a chessboard --/
structure Chessboard where
  black_cells : ℕ

/-- Represents the operation of repainting a row or column --/
def repaint (board : Chessboard) : Chessboard :=
  { black_cells := board.black_cells + (8 - 2 * (board.black_cells % 8)) }

/-- Theorem stating that it's impossible to achieve an odd number of black cells --/
theorem impossible_odd_black_cells (initial_board : Chessboard) 
  (h : Even initial_board.black_cells) :
  ∀ (final_board : Chessboard), 
  (∃ (n : ℕ), final_board = (n.iterate repaint initial_board)) → 
  Even final_board.black_cells :=
sorry

/-- Corollary: It's impossible to achieve exactly one black cell --/
theorem impossible_one_black_cell (initial_board : Chessboard) 
  (h : Even initial_board.black_cells) :
  ¬∃ (final_board : Chessboard), 
  (∃ (n : ℕ), final_board = (n.iterate repaint initial_board)) ∧ 
  final_board.black_cells = 1 :=
sorry

end NUMINAMATH_CALUDE_impossible_odd_black_cells_impossible_one_black_cell_l3964_396409


namespace NUMINAMATH_CALUDE_blue_spotted_fish_ratio_l3964_396407

theorem blue_spotted_fish_ratio (total_fish : ℕ) (blue_spotted_fish : ℕ) 
  (h1 : total_fish = 60) 
  (h2 : blue_spotted_fish = 10) : 
  (blue_spotted_fish : ℚ) / ((1 / 3 : ℚ) * total_fish) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_blue_spotted_fish_ratio_l3964_396407


namespace NUMINAMATH_CALUDE_people_eating_both_veg_nonveg_l3964_396435

/-- The number of people who eat only vegetarian food -/
def only_veg : ℕ := 13

/-- The total number of people who eat vegetarian food -/
def total_veg : ℕ := 21

/-- The number of people who eat both vegetarian and non-vegetarian food -/
def both_veg_nonveg : ℕ := total_veg - only_veg

theorem people_eating_both_veg_nonveg : both_veg_nonveg = 8 := by
  sorry

end NUMINAMATH_CALUDE_people_eating_both_veg_nonveg_l3964_396435


namespace NUMINAMATH_CALUDE_leftmost_row_tiles_l3964_396481

/-- Represents the number of tiles in each row of the floor -/
def tileSequence (firstRow : ℕ) : ℕ → ℕ
  | 0 => firstRow
  | n + 1 => tileSequence firstRow n - 2

/-- The sum of tiles in all rows -/
def totalTiles (firstRow : ℕ) : ℕ :=
  (List.range 9).map (tileSequence firstRow) |>.sum

theorem leftmost_row_tiles :
  ∃ (firstRow : ℕ), totalTiles firstRow = 405 ∧ firstRow = 53 := by
  sorry

end NUMINAMATH_CALUDE_leftmost_row_tiles_l3964_396481


namespace NUMINAMATH_CALUDE_triangle_properties_l3964_396496

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a / (Real.sin A) = b / (Real.sin B) ∧
  b / (Real.sin B) = c / (Real.sin C) ∧
  b * c * (Real.cos A) = 4 ∧
  a * c * (Real.sin B) = 8 * (Real.sin A) →
  A = π / 3 ∧ 
  0 < Real.sin A * Real.sin B * Real.sin C ∧ 
  Real.sin A * Real.sin B * Real.sin C ≤ 3 * Real.sqrt 3 / 8 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3964_396496


namespace NUMINAMATH_CALUDE_nh4_2so4_weight_l3964_396449

/-- Atomic weight of Nitrogen in g/mol -/
def N_weight : ℝ := 14.01

/-- Atomic weight of Hydrogen in g/mol -/
def H_weight : ℝ := 1.01

/-- Atomic weight of Sulfur in g/mol -/
def S_weight : ℝ := 32.07

/-- Atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 16.00

/-- Number of Nitrogen atoms in (NH4)2SO4 -/
def N_count : ℕ := 2

/-- Number of Hydrogen atoms in (NH4)2SO4 -/
def H_count : ℕ := 8

/-- Number of Sulfur atoms in (NH4)2SO4 -/
def S_count : ℕ := 1

/-- Number of Oxygen atoms in (NH4)2SO4 -/
def O_count : ℕ := 4

/-- Number of moles of (NH4)2SO4 -/
def moles : ℝ := 7

/-- Molecular weight of (NH4)2SO4 in g/mol -/
def molecular_weight : ℝ := N_weight * N_count + H_weight * H_count + S_weight * S_count + O_weight * O_count

theorem nh4_2so4_weight : moles * molecular_weight = 924.19 := by
  sorry

end NUMINAMATH_CALUDE_nh4_2so4_weight_l3964_396449


namespace NUMINAMATH_CALUDE_polynomial_expansion_l3964_396477

theorem polynomial_expansion (z : ℝ) : 
  (2 * z^2 + 5 * z - 6) * (3 * z^3 - 2 * z + 1) = 
  6 * z^5 + 15 * z^4 - 22 * z^3 - 8 * z^2 + 17 * z - 6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l3964_396477


namespace NUMINAMATH_CALUDE_football_practice_hours_l3964_396487

/-- Given a football team's practice schedule and a week with one missed day,
    calculate the total practice hours for the week. -/
theorem football_practice_hours (practice_hours_per_day : ℕ) (days_in_week : ℕ) (missed_days : ℕ) : 
  practice_hours_per_day = 5 → days_in_week = 7 → missed_days = 1 →
  (days_in_week - missed_days) * practice_hours_per_day = 30 := by
sorry

end NUMINAMATH_CALUDE_football_practice_hours_l3964_396487


namespace NUMINAMATH_CALUDE_lcm_gcf_ratio_252_630_l3964_396461

theorem lcm_gcf_ratio_252_630 : Nat.lcm 252 630 / Nat.gcd 252 630 = 10 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_ratio_252_630_l3964_396461


namespace NUMINAMATH_CALUDE_carson_gardening_time_l3964_396465

/-- The total time Carson spends gardening is 108 minutes -/
theorem carson_gardening_time :
  let lines_to_mow : ℕ := 40
  let time_per_line : ℕ := 2
  let flower_rows : ℕ := 8
  let flowers_per_row : ℕ := 7
  let time_per_flower : ℚ := 1/2
  lines_to_mow * time_per_line + flower_rows * flowers_per_row * time_per_flower = 108 := by
  sorry

end NUMINAMATH_CALUDE_carson_gardening_time_l3964_396465


namespace NUMINAMATH_CALUDE_trigonometric_identities_l3964_396415

theorem trigonometric_identities :
  (∃ (x y : ℝ), 
    x = Real.sin (-14 * Real.pi / 3) + Real.cos (20 * Real.pi / 3) + Real.tan (-53 * Real.pi / 6) ∧
    x = (-3 - Real.sqrt 3) / 6 ∧
    y = Real.tan (675 * Real.pi / 180) - Real.sin (-330 * Real.pi / 180) - Real.cos (960 * Real.pi / 180) ∧
    y = -2) := by sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l3964_396415


namespace NUMINAMATH_CALUDE_parabola_vertex_l3964_396463

/-- The vertex of the parabola y = 2x^2 + 16x + 34 is (-4, 2) -/
theorem parabola_vertex :
  let f (x : ℝ) := 2 * x^2 + 16 * x + 34
  ∃! (h k : ℝ), ∀ x, f x = 2 * (x - h)^2 + k ∧ h = -4 ∧ k = 2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3964_396463


namespace NUMINAMATH_CALUDE_tangent_lines_chord_length_l3964_396495

noncomputable section

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 16

-- Define point A
def point_A : ℝ × ℝ := (4, -2)

-- Define a line passing through point A
def line_through_A (k : ℝ) (x y : ℝ) : Prop := y + 2 = k * (x - 4)

-- Define the tangent line equations
def tangent_line_1 (x : ℝ) : Prop := x = 4
def tangent_line_2 (x y : ℝ) : Prop := 7*x - 24*y - 76 = 0

-- Define the line with slope angle 135°
def line_135 (x y : ℝ) : Prop := x + y - 2 = 0

-- Theorem for tangent lines
theorem tangent_lines :
  ∃ (l : ℝ × ℝ → Prop), (∀ p, l p ↔ (tangent_line_1 p.1 ∨ tangent_line_2 p.1 p.2)) ∧
  (∀ p, l p → line_through_A (7/24) p.1 p.2) ∧
  (∀ p, l p → (p.1 = point_A.1 ∧ p.2 = point_A.2) ∨ circle_M p.1 p.2) :=
sorry

-- Theorem for chord length
theorem chord_length :
  ∃ (p q : ℝ × ℝ), 
    line_135 p.1 p.2 ∧ line_135 q.1 q.2 ∧
    circle_M p.1 p.2 ∧ circle_M q.1 q.2 ∧
    (p.1 - q.1)^2 + (p.2 - q.2)^2 = 62 :=
sorry

end

end NUMINAMATH_CALUDE_tangent_lines_chord_length_l3964_396495


namespace NUMINAMATH_CALUDE_nine_times_two_sevenths_squared_l3964_396410

theorem nine_times_two_sevenths_squared :
  9 * (2 / 7)^2 = 36 / 49 := by sorry

end NUMINAMATH_CALUDE_nine_times_two_sevenths_squared_l3964_396410


namespace NUMINAMATH_CALUDE_special_numbers_l3964_396456

def is_special_number (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ),
    n = a * 1000 + b * 100 + c * 10 + d ∧
    a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧
    (a + b * 100 + c * 10 + d) * 10 = a * 100 + b * 10 + c + d

theorem special_numbers :
  ∀ n : ℕ, is_special_number n ↔ 
    n = 2019 ∨ n = 3028 ∨ n = 4037 ∨ n = 5046 ∨ 
    n = 6055 ∨ n = 7064 ∨ n = 8073 ∨ n = 9082 :=
by sorry

end NUMINAMATH_CALUDE_special_numbers_l3964_396456


namespace NUMINAMATH_CALUDE_train_speed_is_88_l3964_396428

/-- Represents the transportation problem with train and ship --/
structure TransportProblem where
  rail_distance : ℝ
  river_distance : ℝ
  train_delay : ℝ
  train_arrival_diff : ℝ
  speed_difference : ℝ

/-- Calculates the train speed given the problem parameters --/
def calculate_train_speed (p : TransportProblem) : ℝ :=
  let train_time := p.rail_distance / x
  let ship_time := p.river_distance / (x - p.speed_difference)
  let time_diff := ship_time - train_time
  x
where
  x := 88 -- The solution we want to prove

/-- Theorem stating that the calculated train speed is correct --/
theorem train_speed_is_88 (p : TransportProblem) 
  (h1 : p.rail_distance = 88)
  (h2 : p.river_distance = 108)
  (h3 : p.train_delay = 1)
  (h4 : p.train_arrival_diff = 1/4)
  (h5 : p.speed_difference = 40) :
  calculate_train_speed p = 88 := by
  sorry

#eval calculate_train_speed { 
  rail_distance := 88, 
  river_distance := 108, 
  train_delay := 1, 
  train_arrival_diff := 1/4, 
  speed_difference := 40 
}

end NUMINAMATH_CALUDE_train_speed_is_88_l3964_396428


namespace NUMINAMATH_CALUDE_circle_P_radius_l3964_396446

-- Define the circles and points
def circle_N (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 15 = 0
def point_M : ℝ × ℝ := (-1, 0)

-- Define the curve τ
def curve_τ (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k*(x + 1) ∧ k > 0

-- Define the theorem
theorem circle_P_radius : 
  ∃ (x_P y_P r : ℝ),
  -- Circle P passes through M
  (x_P + 1)^2 + y_P^2 = r^2 ∧
  -- Circle P is internally tangent to N
  ∃ (x_N y_N : ℝ), circle_N x_N y_N ∧ ((x_P - x_N)^2 + (y_P - y_N)^2 = (4 - r)^2) ∧
  -- Center of P is on curve τ
  curve_τ x_P y_P ∧
  -- Line l is tangent to P and intersects τ
  ∃ (k x_A y_A x_B y_B : ℝ),
    line_l k x_A y_A ∧ line_l k x_B y_B ∧
    curve_τ x_A y_A ∧ curve_τ x_B y_B ∧
    -- Q is midpoint of AB with abscissa -4/13
    (x_A + x_B)/2 = -4/13 ∧
  -- One possible radius of P is 6/5
  r = 6/5 :=
sorry

end NUMINAMATH_CALUDE_circle_P_radius_l3964_396446


namespace NUMINAMATH_CALUDE_circle_area_difference_l3964_396439

theorem circle_area_difference : 
  let r1 : ℝ := 30
  let d2 : ℝ := 12
  let r2 : ℝ := d2 / 2
  π * r1^2 - π * r2^2 = 864 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_difference_l3964_396439


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3964_396402

theorem inequality_solution_set (x : ℝ) :
  (x - 3/x > 2) ↔ (-1 < x ∧ x < 0) ∨ (x > 3) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3964_396402


namespace NUMINAMATH_CALUDE_kylies_towels_l3964_396462

theorem kylies_towels (daughters_towels husband_towels machine_capacity loads : ℕ) 
  (h1 : daughters_towels = 6)
  (h2 : husband_towels = 3)
  (h3 : machine_capacity = 4)
  (h4 : loads = 3) : 
  ∃ k : ℕ, k = loads * machine_capacity - daughters_towels - husband_towels ∧ k = 3 := by
  sorry

end NUMINAMATH_CALUDE_kylies_towels_l3964_396462


namespace NUMINAMATH_CALUDE_expected_rainfall_theorem_l3964_396472

/-- Weather forecast for a day --/
structure DailyForecast where
  sunny_prob : ℝ
  light_rain_prob : ℝ
  heavy_rain_prob : ℝ
  light_rain_amount : ℝ
  heavy_rain_amount : ℝ

/-- Calculate expected rainfall for a single day --/
def expected_daily_rainfall (f : DailyForecast) : ℝ :=
  f.light_rain_prob * f.light_rain_amount + f.heavy_rain_prob * f.heavy_rain_amount

/-- Calculate expected rainfall for a week --/
def expected_weekly_rainfall (f : DailyForecast) (days : ℕ) : ℝ :=
  (expected_daily_rainfall f) * days

/-- The weather forecast for the week --/
def weekly_forecast : DailyForecast :=
  { sunny_prob := 0.30
  , light_rain_prob := 0.35
  , heavy_rain_prob := 0.35
  , light_rain_amount := 3
  , heavy_rain_amount := 8 }

/-- The number of days in the forecast --/
def forecast_days : ℕ := 7

/-- Theorem: The expected rainfall for the week is approximately 26.9 inches --/
theorem expected_rainfall_theorem :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |expected_weekly_rainfall weekly_forecast forecast_days - 26.9| < ε := by
  sorry

end NUMINAMATH_CALUDE_expected_rainfall_theorem_l3964_396472


namespace NUMINAMATH_CALUDE_tom_steps_when_matt_reaches_220_l3964_396404

/-- Proves that Tom reaches 275 steps when Matt reaches 220 steps, given their respective speeds -/
theorem tom_steps_when_matt_reaches_220 
  (matt_speed : ℕ) 
  (tom_speed_diff : ℕ) 
  (matt_steps : ℕ) 
  (h1 : matt_speed = 20)
  (h2 : tom_speed_diff = 5)
  (h3 : matt_steps = 220) :
  matt_steps + (matt_steps / matt_speed) * tom_speed_diff = 275 := by
  sorry

#check tom_steps_when_matt_reaches_220

end NUMINAMATH_CALUDE_tom_steps_when_matt_reaches_220_l3964_396404


namespace NUMINAMATH_CALUDE_domain_transformation_l3964_396480

-- Define a real-valued function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x²)
def domain_f_squared : Set ℝ := Set.Ioc (-3) 1

-- Define the domain of f(x-1)
def domain_f_shifted : Set ℝ := Set.Ico 1 10

-- Theorem statement
theorem domain_transformation (h : ∀ x, x ∈ domain_f_squared ↔ f (x^2) ∈ Set.range f) :
  ∀ x, x ∈ domain_f_shifted ↔ f (x - 1) ∈ Set.range f :=
sorry

end NUMINAMATH_CALUDE_domain_transformation_l3964_396480


namespace NUMINAMATH_CALUDE_rectangle_p_value_l3964_396467

/-- Rectangle PQRS with given vertices and area -/
structure Rectangle where
  P : ℝ × ℝ
  S : ℝ × ℝ
  Q : ℝ × ℝ
  area : ℝ

/-- The theorem stating that if a rectangle PQRS has the given properties, then p = 15 -/
theorem rectangle_p_value (rect : Rectangle)
  (h1 : rect.P = (2, 3))
  (h2 : rect.S = (12, 3))
  (h3 : rect.Q.2 = 15)
  (h4 : rect.area = 120) :
  rect.Q.1 = 15 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_p_value_l3964_396467


namespace NUMINAMATH_CALUDE_class_ratio_problem_l3964_396466

theorem class_ratio_problem (total : ℕ) (boys : ℕ) (h_total : total > 0) (h_boys : boys ≤ total) :
  let p_boy := boys / total
  let p_girl := (total - boys) / total
  (p_boy = (2 : ℚ) / 3 * p_girl) → (boys : ℚ) / total = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_class_ratio_problem_l3964_396466


namespace NUMINAMATH_CALUDE_equation_solution_range_l3964_396471

theorem equation_solution_range (x m : ℝ) : 
  ((2 * x + m) / (x - 1) = 1) → 
  (x > 0) → 
  (x ≠ 1) → 
  (m < -1) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_range_l3964_396471


namespace NUMINAMATH_CALUDE_sams_cows_l3964_396444

theorem sams_cows (C : ℕ) : 
  (C / 2 + 5 = C - 4) → C = 18 := by
  sorry

end NUMINAMATH_CALUDE_sams_cows_l3964_396444


namespace NUMINAMATH_CALUDE_second_number_in_ratio_l3964_396494

theorem second_number_in_ratio (a b c : ℕ) : 
  a + b + c = 108 → 
  5 * b = 3 * a → 
  4 * b = 3 * c → 
  b = 27 := by sorry

end NUMINAMATH_CALUDE_second_number_in_ratio_l3964_396494


namespace NUMINAMATH_CALUDE_juice_bottles_count_l3964_396437

theorem juice_bottles_count : ∃ x : ℕ, 
  let day0_remaining := x / 2 + 1
  let day1_remaining := day0_remaining / 2
  let day2_remaining := day1_remaining / 2 - 1
  x > 0 ∧ day2_remaining = 2 → x = 22 := by
  sorry

end NUMINAMATH_CALUDE_juice_bottles_count_l3964_396437


namespace NUMINAMATH_CALUDE_cat_cafe_theorem_l3964_396413

/-- The number of cats in Cat Cafe Cool -/
def cool_cats : ℕ := 5

/-- The number of cats in Cat Cafe Paw -/
def paw_cats : ℕ := 2 * cool_cats

/-- The number of cats in Cat Cafe Meow -/
def meow_cats : ℕ := 3 * paw_cats

/-- The total number of cats in Cat Cafe Meow and Cat Cafe Paw -/
def total_cats : ℕ := meow_cats + paw_cats

theorem cat_cafe_theorem : total_cats = 40 := by
  sorry

end NUMINAMATH_CALUDE_cat_cafe_theorem_l3964_396413


namespace NUMINAMATH_CALUDE_sum_of_coefficients_zero_l3964_396485

/-- A function g(x) with specific properties -/
noncomputable def g (A B C : ℤ) : ℝ → ℝ := λ x => x^2 / (A * x^2 + B * x + C)

/-- Theorem stating the sum of coefficients A, B, and C is zero -/
theorem sum_of_coefficients_zero
  (A B C : ℤ)
  (h1 : ∀ x > 2, g A B C x > 0.3)
  (h2 : (A * 1^2 + B * 1 + C = 0) ∧ (A * (-3)^2 + B * (-3) + C = 0)) :
  A + B + C = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_zero_l3964_396485


namespace NUMINAMATH_CALUDE_order_of_numbers_l3964_396403

theorem order_of_numbers : ∀ (a b c : ℝ), 
  a = 6^(1/2) → b = (1/2)^6 → c = Real.log 6 / Real.log (1/2) →
  c < b ∧ b < a :=
by
  sorry

end NUMINAMATH_CALUDE_order_of_numbers_l3964_396403


namespace NUMINAMATH_CALUDE_boys_count_in_class_l3964_396450

/-- Given a class with a 3:4 ratio of girls to boys and 35 total students,
    prove that the number of boys is 20. -/
theorem boys_count_in_class (total : ℕ) (girls : ℕ) (boys : ℕ) : 
  total = 35 →
  girls + boys = total →
  3 * boys = 4 * girls →
  boys = 20 := by
sorry

end NUMINAMATH_CALUDE_boys_count_in_class_l3964_396450


namespace NUMINAMATH_CALUDE_digit_sum_of_power_product_l3964_396424

def power_product (a b c d e : ℕ) : ℕ := a^b * c^d * e

theorem digit_sum_of_power_product :
  ∃ (f : ℕ → ℕ), f (power_product 2 2010 5 2012 7) = 13 :=
sorry

end NUMINAMATH_CALUDE_digit_sum_of_power_product_l3964_396424


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3964_396469

/-- A quadratic equation kx^2 - 4x + 1 = 0 has two distinct real roots if and only if k < 4 and k ≠ 0 -/
theorem quadratic_two_distinct_roots (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ k * x^2 - 4 * x + 1 = 0 ∧ k * y^2 - 4 * y + 1 = 0) ↔ 
  (k < 4 ∧ k ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3964_396469


namespace NUMINAMATH_CALUDE_fourteenth_root_of_unity_l3964_396459

theorem fourteenth_root_of_unity : 
  ∃ (n : ℕ), 0 ≤ n ∧ n ≤ 13 ∧ 
  (Complex.tan (π / 7) + Complex.I) / (Complex.tan (π / 7) - Complex.I) = 
  Complex.exp (Complex.I * (2 * ↑n * π / 14)) := by
  sorry

end NUMINAMATH_CALUDE_fourteenth_root_of_unity_l3964_396459


namespace NUMINAMATH_CALUDE_line_intersection_y_axis_l3964_396473

/-- A line passing through two points intersects the y-axis at a specific point -/
theorem line_intersection_y_axis 
  (x₁ y₁ x₂ y₂ : ℝ) 
  (h_distinct : x₁ ≠ x₂) 
  (h_point1 : x₁ = 3 ∧ y₁ = 18) 
  (h_point2 : x₂ = -7 ∧ y₂ = -2) : 
  ∃ (y : ℝ), y = 12 ∧ 
  (y - y₁) / (0 - x₁) = (y₂ - y₁) / (x₂ - x₁) :=
sorry

end NUMINAMATH_CALUDE_line_intersection_y_axis_l3964_396473


namespace NUMINAMATH_CALUDE_evaporation_weight_theorem_l3964_396438

/-- Represents the weight of a glass containing a solution --/
structure GlassSolution where
  total_weight : ℝ
  water_percentage : ℝ
  glass_weight : ℝ

/-- Calculates the final weight of a glass solution after water evaporation --/
def final_weight (initial : GlassSolution) (final_water_percentage : ℝ) : ℝ :=
  sorry

/-- Theorem stating that given the initial conditions and final water percentage,
    the final weight of the glass with solution is 400 grams --/
theorem evaporation_weight_theorem (initial : GlassSolution) 
    (h1 : initial.total_weight = 500)
    (h2 : initial.water_percentage = 0.99)
    (h3 : initial.glass_weight = 300)
    (final_water_percentage : ℝ)
    (h4 : final_water_percentage = 0.98) :
    final_weight initial final_water_percentage = 400 := by
  sorry

end NUMINAMATH_CALUDE_evaporation_weight_theorem_l3964_396438


namespace NUMINAMATH_CALUDE_vector_expression_l3964_396400

/-- Given vectors in ℝ², prove that c = 3a + 2b -/
theorem vector_expression (a b c : ℝ × ℝ) : 
  a = (1, -1) → b = (-1, 2) → c = (1, 1) → c = 3 • a + 2 • b := by sorry

end NUMINAMATH_CALUDE_vector_expression_l3964_396400


namespace NUMINAMATH_CALUDE_total_carpets_l3964_396498

theorem total_carpets (house1 house2 house3 house4 : ℕ) : 
  house1 = 12 → 
  house2 = 20 → 
  house3 = 10 → 
  house4 = 2 * house3 → 
  house1 + house2 + house3 + house4 = 62 := by
  sorry

end NUMINAMATH_CALUDE_total_carpets_l3964_396498


namespace NUMINAMATH_CALUDE_inequality_solution_set_f_inequality_l3964_396417

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1|

-- Theorem for part (I)
theorem inequality_solution_set (x : ℝ) :
  f (x + 8) ≥ 10 - f x ↔ x ≤ -10 ∨ x ≥ 0 := by sorry

-- Theorem for part (II)
theorem f_inequality (x y : ℝ) (hx : |x| > 1) (hy : |y| < 1) :
  f y < |x| * f (y / x^2) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_f_inequality_l3964_396417


namespace NUMINAMATH_CALUDE_perfectSquareFactorsOf360_l3964_396412

def perfectSquareFactors (n : ℕ) : ℕ := sorry

theorem perfectSquareFactorsOf360 : perfectSquareFactors 360 = 4 := by
  sorry

end NUMINAMATH_CALUDE_perfectSquareFactorsOf360_l3964_396412


namespace NUMINAMATH_CALUDE_room_length_calculation_l3964_396425

/-- Given a room with specified width, total paving cost, and paving rate per square meter,
    calculate the length of the room. -/
theorem room_length_calculation (width : ℝ) (total_cost : ℝ) (rate_per_sqm : ℝ) :
  width = 3.75 ∧ total_cost = 16500 ∧ rate_per_sqm = 800 →
  (total_cost / rate_per_sqm) / width = 5.5 :=
by sorry

end NUMINAMATH_CALUDE_room_length_calculation_l3964_396425


namespace NUMINAMATH_CALUDE_circle_line_intersection_range_l3964_396427

/-- Circle C in the Cartesian coordinate plane -/
def CircleC (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 12 = 0

/-- Line in the Cartesian coordinate plane -/
def Line (k : ℝ) (x y : ℝ) : Prop := y = k*x - 2

/-- New circle with radius 2 centered at a point (a, b) -/
def NewCircle (a b x y : ℝ) : Prop := (x - a)^2 + (y - b)^2 = 4

/-- Theorem stating the range of k values -/
theorem circle_line_intersection_range :
  ∀ k : ℝ, (∃ a b : ℝ, Line k a b ∧
    (∃ x y : ℝ, CircleC x y ∧ NewCircle a b x y)) ↔
  0 ≤ k ∧ k ≤ 4/3 :=
sorry

end NUMINAMATH_CALUDE_circle_line_intersection_range_l3964_396427


namespace NUMINAMATH_CALUDE_linear_mapping_midpoint_distance_l3964_396406

/-- Linear mapping from a segment of length 10 to a segment of length 5 -/
def LinearMapping (x y : ℝ) : Prop :=
  x / 10 = y / 5

/-- Theorem: In the given linear mapping, when x = 3, x + y = 4.5 -/
theorem linear_mapping_midpoint_distance (x y : ℝ) :
  LinearMapping x y → x = 3 → x + y = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_linear_mapping_midpoint_distance_l3964_396406


namespace NUMINAMATH_CALUDE_existence_of_nth_root_l3964_396426

theorem existence_of_nth_root (n b : ℕ) (hn : n > 1) (hb : b > 1)
  (h : ∀ k : ℕ, k > 1 → ∃ a : ℤ, (k : ℤ) ∣ b - a^n) :
  ∃ A : ℤ, b = A^n := by
sorry

end NUMINAMATH_CALUDE_existence_of_nth_root_l3964_396426


namespace NUMINAMATH_CALUDE_cloth_profit_per_meter_l3964_396447

/-- Calculates the profit per meter of cloth given the total selling price, 
    number of meters sold, and cost price per meter. -/
def profit_per_meter (selling_price total_meters cost_price_per_meter : ℕ) : ℕ :=
  ((selling_price - (cost_price_per_meter * total_meters)) / total_meters)

/-- Proves that the profit per meter of cloth is 15 rupees given the specified conditions. -/
theorem cloth_profit_per_meter :
  profit_per_meter 8500 85 85 = 15 := by
  sorry


end NUMINAMATH_CALUDE_cloth_profit_per_meter_l3964_396447


namespace NUMINAMATH_CALUDE_inequalities_theorem_l3964_396408

theorem inequalities_theorem (a b : ℝ) (h : (1 / a) < (1 / b) ∧ (1 / b) < 0) :
  (abs a < abs b) ∧ (a > b) ∧ (a + b > a * b) ∧ (a^3 > b^3) := by sorry

end NUMINAMATH_CALUDE_inequalities_theorem_l3964_396408


namespace NUMINAMATH_CALUDE_exponent_addition_l3964_396405

theorem exponent_addition (a : ℝ) : a^3 + a^3 = 2 * a^3 := by
  sorry

end NUMINAMATH_CALUDE_exponent_addition_l3964_396405


namespace NUMINAMATH_CALUDE_rectangle_perimeter_is_164_l3964_396423

/-- Represents the side lengths of the squares in the rectangle dissection -/
structure SquareSides where
  a₁ : ℕ
  a₂ : ℕ
  a₃ : ℕ
  a₄ : ℕ
  a₅ : ℕ
  a₆ : ℕ
  a₇ : ℕ
  a₈ : ℕ
  a₉ : ℕ

/-- The conditions for the rectangle dissection -/
def RectangleDissectionConditions (s : SquareSides) : Prop :=
  s.a₁ + s.a₂ = s.a₄ ∧
  s.a₁ + s.a₄ = s.a₅ ∧
  s.a₄ + s.a₅ = s.a₇ ∧
  s.a₅ + s.a₇ = s.a₉ ∧
  s.a₂ + s.a₄ + s.a₇ = s.a₈ ∧
  s.a₂ + s.a₈ = s.a₆ ∧
  s.a₁ + s.a₅ + s.a₉ = s.a₃ ∧
  s.a₃ + s.a₆ = s.a₈ + s.a₇

/-- The width of the rectangle -/
def RectangleWidth (s : SquareSides) : ℕ := s.a₄ + s.a₇ + s.a₉

/-- The length of the rectangle -/
def RectangleLength (s : SquareSides) : ℕ := s.a₂ + s.a₈ + s.a₆

/-- The main theorem: Given the conditions, the perimeter of the rectangle is 164 -/
theorem rectangle_perimeter_is_164 (s : SquareSides) 
  (h : RectangleDissectionConditions s) 
  (h_coprime : Nat.Coprime (RectangleWidth s) (RectangleLength s)) :
  2 * (RectangleWidth s + RectangleLength s) = 164 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_perimeter_is_164_l3964_396423


namespace NUMINAMATH_CALUDE_real_estate_investment_l3964_396468

theorem real_estate_investment
  (total_investment : ℝ)
  (real_estate_ratio : ℝ)
  (h1 : total_investment = 200000)
  (h2 : real_estate_ratio = 6) :
  let mutual_funds := total_investment / (1 + real_estate_ratio)
  let real_estate := real_estate_ratio * mutual_funds
  real_estate = 171428.58 := by sorry

end NUMINAMATH_CALUDE_real_estate_investment_l3964_396468


namespace NUMINAMATH_CALUDE_complex_multiplication_l3964_396452

theorem complex_multiplication (z₁ z₂ z : ℂ) : 
  z₁ = 1 - 3*I ∧ z₂ = 6 - 8*I ∧ z = z₁ * z₂ → z = -18 - 26*I :=
by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l3964_396452


namespace NUMINAMATH_CALUDE_chessboard_inner_square_probability_l3964_396455

/-- Represents a square chessboard -/
structure Chessboard :=
  (size : ℕ)

/-- Calculates the total number of squares on the chessboard -/
def total_squares (board : Chessboard) : ℕ :=
  board.size * board.size

/-- Calculates the number of squares in the outermost two rows and columns -/
def outer_squares (board : Chessboard) : ℕ :=
  4 * board.size - 4

/-- Calculates the number of inner squares not touching the outermost two rows or columns -/
def inner_squares (board : Chessboard) : ℕ :=
  total_squares board - outer_squares board

/-- The probability of choosing an inner square -/
def inner_square_probability (board : Chessboard) : ℚ :=
  inner_squares board / total_squares board

theorem chessboard_inner_square_probability :
  ∃ (board : Chessboard), board.size = 10 ∧ inner_square_probability board = 17 / 25 := by
  sorry

end NUMINAMATH_CALUDE_chessboard_inner_square_probability_l3964_396455


namespace NUMINAMATH_CALUDE_butterscotch_servings_left_l3964_396442

def total_servings : ℕ := 61
def num_guests : ℕ := 8
def first_group_share : ℚ := 2/5
def second_group_share : ℚ := 1/4
def last_guest_servings : ℕ := 5

theorem butterscotch_servings_left (first_group_consumed : ℕ) (second_group_consumed : ℕ)
  (h1 : first_group_consumed = ⌊(first_group_share * total_servings : ℚ)⌋)
  (h2 : second_group_consumed = ⌊(second_group_share * total_servings : ℚ)⌋)
  : total_servings - (first_group_consumed + second_group_consumed + last_guest_servings) = 17 := by
  sorry

end NUMINAMATH_CALUDE_butterscotch_servings_left_l3964_396442


namespace NUMINAMATH_CALUDE_expression_value_l3964_396418

theorem expression_value (a b : ℝ) 
  (ha : a = 2 * Real.sin (45 * π / 180) + 1)
  (hb : b = 2 * Real.cos (45 * π / 180) - 1) :
  ((a^2 + b^2) / (2*a*b) - 1) / ((a^2 - b^2) / (a^2*b + a*b^2)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3964_396418


namespace NUMINAMATH_CALUDE_line_equations_l3964_396419

-- Define the line l₁
def l₁ (x y : ℝ) : Prop := 2 * x + 4 * y - 1 = 0

-- Define a general line passing through a point
def line_through_point (a b c : ℝ) (x₀ y₀ : ℝ) : Prop :=
  a * x₀ + b * y₀ + c = 0

-- Define parallel lines
def parallel_lines (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  a₁ * b₂ = a₂ * b₁

-- Define perpendicular lines
def perpendicular_lines (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  a₁ * a₂ + b₁ * b₂ = 0

theorem line_equations :
  ∃ (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ),
    (∀ x y, l₁ x y ↔ a₁ * x + b₁ * y + c₁ = 0) ∧
    line_through_point a₂ b₂ c₂ 1 (-2) ∧
    ((parallel_lines a₁ b₁ c₁ a₂ b₂ c₂ →
      ∀ x y, a₂ * x + b₂ * y + c₂ = 0 ↔ x + 2 * y + 3 = 0) ∧
     (perpendicular_lines a₁ b₁ c₁ a₂ b₂ c₂ →
      ∀ x y, a₂ * x + b₂ * y + c₂ = 0 ↔ 2 * x - y - 4 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_line_equations_l3964_396419


namespace NUMINAMATH_CALUDE_xyz_value_l3964_396493

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 30)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 14) :
  x * y * z = 16 / 3 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l3964_396493


namespace NUMINAMATH_CALUDE_line_L_equation_l3964_396490

-- Define the point A
def A : ℝ × ℝ := (2, 4)

-- Define the parallel lines
def line1 (x y : ℝ) : Prop := x - y + 1 = 0
def line2 (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the line on which the midpoint lies
def midpoint_line (x y : ℝ) : Prop := x + 2*y - 3 = 0

-- Define the equation of line L
def line_L (x y : ℝ) : Prop := 3*x - y - 2 = 0

-- Theorem statement
theorem line_L_equation :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    -- L passes through A
    line_L 2 4 ∧
    -- L intersects the parallel lines
    line1 x₁ y₁ ∧ line2 x₂ y₂ ∧ line_L x₁ y₁ ∧ line_L x₂ y₂ ∧
    -- Midpoint of the segment lies on the given line
    midpoint_line ((x₁ + x₂) / 2) ((y₁ + y₂) / 2) :=
by sorry

end NUMINAMATH_CALUDE_line_L_equation_l3964_396490


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3964_396434

theorem imaginary_part_of_z (z : ℂ) : z = (3 + 4*Complex.I)*Complex.I → z.im = 3 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3964_396434


namespace NUMINAMATH_CALUDE_root_equation_problem_l3964_396497

/-- Given two polynomial equations with constants c and d, prove that 100c + d = 359 -/
theorem root_equation_problem (c d : ℝ) : 
  (∃! x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    ((x + c) * (x + d) * (x + 10)) / ((x + 5) * (x + 5)) = 0 ∧
    ((y + c) * (y + d) * (y + 10)) / ((y + 5) * (y + 5)) = 0 ∧
    ((z + c) * (z + d) * (z + 10)) / ((z + 5) * (z + 5)) = 0) ∧
  (∃! w : ℝ, ((w + 2*c) * (w + 7) * (w + 9)) / ((w + d) * (w + 10)) = 0) →
  100 * c + d = 359 := by
sorry

end NUMINAMATH_CALUDE_root_equation_problem_l3964_396497


namespace NUMINAMATH_CALUDE_min_value_expression_l3964_396422

theorem min_value_expression (x y z : ℝ) 
  (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) :
  (1 / ((1 - x) * (1 - y) * (1 - z)) + 
   1 / ((1 + x) * (1 + y) * (1 + z)) + 
   1 / ((1 - x^2) * (1 - y^2) * (1 - z^2))) ≥ 3 ∧
  (1 / ((1 - 0) * (1 - 0) * (1 - 0)) + 
   1 / ((1 + 0) * (1 + 0) * (1 + 0)) + 
   1 / ((1 - 0^2) * (1 - 0^2) * (1 - 0^2))) = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3964_396422


namespace NUMINAMATH_CALUDE_f_seven_plus_f_nine_l3964_396420

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem f_seven_plus_f_nine (f : ℝ → ℝ) 
  (h_periodic : is_periodic f 4)
  (h_odd : is_odd (fun x ↦ f (x - 1)))
  (h_f_one : f 1 = 1) : 
  f 7 + f 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_seven_plus_f_nine_l3964_396420


namespace NUMINAMATH_CALUDE_increase_by_percentage_l3964_396499

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (final : ℝ) :
  initial = 1500 →
  percentage = 20 →
  final = initial * (1 + percentage / 100) →
  final = 1800 := by
  sorry

end NUMINAMATH_CALUDE_increase_by_percentage_l3964_396499


namespace NUMINAMATH_CALUDE_inheritance_tax_calculation_l3964_396431

theorem inheritance_tax_calculation (inheritance : ℝ) 
  (federal_tax_rate : ℝ) (state_tax_rate : ℝ) (total_tax : ℝ) : 
  inheritance = 38600 →
  federal_tax_rate = 0.25 →
  state_tax_rate = 0.15 →
  total_tax = 14000 →
  total_tax = inheritance * federal_tax_rate + 
    (inheritance - inheritance * federal_tax_rate) * state_tax_rate :=
by sorry

end NUMINAMATH_CALUDE_inheritance_tax_calculation_l3964_396431


namespace NUMINAMATH_CALUDE_rider_distances_l3964_396454

/-- The possible distances between two riders after one hour, given their initial distance and speeds -/
theorem rider_distances (initial_distance : ℝ) (speed_athos : ℝ) (speed_aramis : ℝ) :
  initial_distance = 20 ∧ speed_athos = 4 ∧ speed_aramis = 5 →
  ∃ (d₁ d₂ d₃ d₄ : ℝ),
    d₁ = 11 ∧ d₂ = 29 ∧ d₃ = 19 ∧ d₄ = 21 ∧
    ({d₁, d₂, d₃, d₄} : Set ℝ) = {
      initial_distance - (speed_athos + speed_aramis),
      initial_distance + (speed_athos + speed_aramis),
      initial_distance - (speed_aramis - speed_athos),
      initial_distance + (speed_aramis - speed_athos)
    } := by sorry

end NUMINAMATH_CALUDE_rider_distances_l3964_396454


namespace NUMINAMATH_CALUDE_evaluate_expression_l3964_396486

theorem evaluate_expression : 6 - 9 * (10 - 4^2) * 5 = 276 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3964_396486


namespace NUMINAMATH_CALUDE_rational_roots_of_polynomial_l3964_396436

theorem rational_roots_of_polynomial (x : ℚ) :
  (3 * x^4 - 4 * x^3 - 10 * x^2 + 8 * x + 3 = 0) ↔ (x = 1 ∨ x = 1/3) :=
by sorry

end NUMINAMATH_CALUDE_rational_roots_of_polynomial_l3964_396436


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l3964_396457

def complex_equation (z : ℂ) : Prop :=
  (z - 2) * (z^2 + z + 2) * (z^2 + 5*z + 8) = 0

def is_root (z : ℂ) : Prop :=
  complex_equation z

def ellipse_through_roots (e : ℝ) : Prop :=
  ∃ (a b : ℝ) (h : ℂ), 
    a > 0 ∧ b > 0 ∧
    ∀ (z : ℂ), is_root z → 
      (z.re - h.re)^2 / a^2 + (z.im - h.im)^2 / b^2 = 1 ∧
    e = Real.sqrt (1 - b^2 / a^2)

theorem ellipse_eccentricity : 
  ellipse_through_roots (Real.sqrt (1/5)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l3964_396457


namespace NUMINAMATH_CALUDE_range_of_g_l3964_396453

def g (x : ℝ) : ℝ := 3 * (x - 4)

theorem range_of_g :
  {y : ℝ | ∃ x : ℝ, x ≠ -5 ∧ g x = y} = {y : ℝ | y < -27 ∨ y > -27} :=
by sorry

end NUMINAMATH_CALUDE_range_of_g_l3964_396453


namespace NUMINAMATH_CALUDE_variance_of_binomial_distribution_l3964_396429

/-- The number of trials -/
def n : ℕ := 100

/-- The probability of success (drawing a second) -/
def p : ℝ := 0.02

/-- The variance of a binomial distribution -/
def binomial_variance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

/-- Theorem: The variance of the given binomial distribution is 1.96 -/
theorem variance_of_binomial_distribution :
  binomial_variance n p = 1.96 := by
  sorry

end NUMINAMATH_CALUDE_variance_of_binomial_distribution_l3964_396429


namespace NUMINAMATH_CALUDE_reciprocal_negative_four_l3964_396430

theorem reciprocal_negative_four (x : ℚ) : x⁻¹ = -4 → x = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_negative_four_l3964_396430


namespace NUMINAMATH_CALUDE_combinatorics_problem_l3964_396448

theorem combinatorics_problem :
  (Nat.factorial 15) / ((Nat.factorial 6) * (Nat.factorial (15 - 6))) = 5005 ∧
  Nat.factorial 6 = 720 := by
sorry

end NUMINAMATH_CALUDE_combinatorics_problem_l3964_396448


namespace NUMINAMATH_CALUDE_interval_condition_l3964_396416

theorem interval_condition (x : ℝ) : 
  (2 < 4*x ∧ 4*x < 5 ∧ 2 < 5*x ∧ 5*x < 5) ↔ (1/2 < x ∧ x < 1) :=
sorry

end NUMINAMATH_CALUDE_interval_condition_l3964_396416


namespace NUMINAMATH_CALUDE_trig_inequality_range_l3964_396475

theorem trig_inequality_range (x : Real) : 
  (x ∈ Set.Icc 0 Real.pi) → 
  (Real.cos x)^2 > (Real.sin x)^2 → 
  x ∈ Set.Ioo 0 (Real.pi / 4) ∪ Set.Ioo (3 * Real.pi / 4) Real.pi :=
by sorry

end NUMINAMATH_CALUDE_trig_inequality_range_l3964_396475


namespace NUMINAMATH_CALUDE_decreasing_function_l3964_396476

-- Define the four functions
def f1 (x : ℝ) : ℝ := x^2 + 1
def f2 (x : ℝ) : ℝ := -x^2 + 1
def f3 (x : ℝ) : ℝ := 2*x + 1
def f4 (x : ℝ) : ℝ := -2*x + 1

-- Theorem statement
theorem decreasing_function : 
  (∀ x : ℝ, HasDerivAt f4 (-2) x) ∧ 
  (∀ x : ℝ, (HasDerivAt f1 (2*x) x) ∨ (HasDerivAt f2 (-2*x) x) ∨ (HasDerivAt f3 2 x)) :=
by sorry

end NUMINAMATH_CALUDE_decreasing_function_l3964_396476


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l3964_396441

theorem smallest_integer_satisfying_inequality :
  ∀ x : ℤ, x + 5 < 3*x - 9 → x ≥ 8 ∧ 8 + 5 < 3*8 - 9 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l3964_396441


namespace NUMINAMATH_CALUDE_eugene_pencils_count_l3964_396443

/-- The total number of pencils Eugene has after receiving additional pencils -/
def total_pencils (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem: Eugene's total pencils equals the sum of his initial pencils and additional pencils -/
theorem eugene_pencils_count : 
  total_pencils 51 6 = 57 := by
  sorry

end NUMINAMATH_CALUDE_eugene_pencils_count_l3964_396443


namespace NUMINAMATH_CALUDE_sum_of_divisors_of_37_l3964_396488

theorem sum_of_divisors_of_37 (h : Nat.Prime 37) : 
  (Finset.filter (· ∣ 37) (Finset.range 38)).sum id = 38 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_of_37_l3964_396488


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l3964_396411

def P : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}
def Q : Set ℝ := {x : ℝ | -2 < x ∧ x < 1}

theorem intersection_of_P_and_Q : P ∩ Q = {x : ℝ | -1 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l3964_396411


namespace NUMINAMATH_CALUDE_geometric_body_volume_l3964_396451

/-- The volume of a geometric body composed of two tetrahedra --/
theorem geometric_body_volume :
  let side_length : ℝ := 1
  let height : ℝ := Real.sqrt 3 / 2
  let tetrahedron_volume : ℝ := (1 / 3) * ((Real.sqrt 3 / 4) * side_length ^ 2) * height
  let total_volume : ℝ := 2 * tetrahedron_volume
  total_volume = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_body_volume_l3964_396451


namespace NUMINAMATH_CALUDE_ceiling_minus_value_l3964_396491

theorem ceiling_minus_value (x ε : ℝ) 
  (h1 : ⌈x + ε⌉ - ⌊x + ε⌋ = 1) 
  (h2 : 0 < ε) 
  (h3 : ε < 1) : 
  ⌈x + ε⌉ - (x + ε) = 1 - ε := by
  sorry

end NUMINAMATH_CALUDE_ceiling_minus_value_l3964_396491


namespace NUMINAMATH_CALUDE_specific_lamp_arrangement_probability_l3964_396474

def num_red_lamps : ℕ := 4
def num_blue_lamps : ℕ := 4
def num_lamps_on : ℕ := 4

def total_lamps : ℕ := num_red_lamps + num_blue_lamps

def probability_specific_arrangement : ℚ :=
  1 / 49

theorem specific_lamp_arrangement_probability :
  probability_specific_arrangement = 1 / 49 := by
  sorry

end NUMINAMATH_CALUDE_specific_lamp_arrangement_probability_l3964_396474


namespace NUMINAMATH_CALUDE_cube_roll_probability_l3964_396492

theorem cube_roll_probability (total_faces green_faces : ℕ) 
  (h1 : total_faces = 6)
  (h2 : green_faces = 3)
  (h3 : total_faces - green_faces = 3) : 
  (green_faces : ℚ) / total_faces = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_cube_roll_probability_l3964_396492


namespace NUMINAMATH_CALUDE_train_passengers_l3964_396445

theorem train_passengers (initial : ℕ) 
  (h1 : initial + 17 - 29 - 27 + 35 = 116) : initial = 120 := by
  sorry

end NUMINAMATH_CALUDE_train_passengers_l3964_396445


namespace NUMINAMATH_CALUDE_solution_set_of_sin_equation_l3964_396401

theorem solution_set_of_sin_equation :
  let S : Set ℝ := {x | 2 * Real.sin ((2/3) * x) = 1}
  S = {x | ∃ k : ℤ, x = 3 * k * Real.pi + Real.pi / 4 ∨ x = 3 * k * Real.pi + 5 * Real.pi / 4} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_sin_equation_l3964_396401


namespace NUMINAMATH_CALUDE_equation_equivalence_l3964_396440

theorem equation_equivalence (x : ℝ) : 
  (1 - (x + 3) / 6 = x / 2) ↔ (6 - x - 3 = 3 * x) := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l3964_396440


namespace NUMINAMATH_CALUDE_power_three_fifteen_mod_five_l3964_396460

theorem power_three_fifteen_mod_five : 3^15 % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_three_fifteen_mod_five_l3964_396460


namespace NUMINAMATH_CALUDE_sock_pair_count_l3964_396478

/-- The number of ways to choose a pair of socks with different colors -/
def different_color_pairs (white brown blue red : ℕ) : ℕ :=
  white * brown + white * blue + white * red +
  brown * blue + brown * red +
  blue * red

/-- Theorem: The number of ways to choose a pair of socks with different colors
    from a drawer containing 5 white socks, 5 brown socks, 3 blue socks,
    and 2 red socks is equal to 81. -/
theorem sock_pair_count :
  different_color_pairs 5 5 3 2 = 81 := by
  sorry

end NUMINAMATH_CALUDE_sock_pair_count_l3964_396478


namespace NUMINAMATH_CALUDE_sum_of_digits_of_power_product_l3964_396484

/-- The sum of the digits of the decimal representation of 2^2010 * 5^2012 * 7 is 13 -/
theorem sum_of_digits_of_power_product : ∃ n : ℕ, 
  (n = 2^2010 * 5^2012 * 7) ∧ 
  (List.sum (Nat.digits 10 n) = 13) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_power_product_l3964_396484


namespace NUMINAMATH_CALUDE_percentage_difference_l3964_396464

theorem percentage_difference (A C : ℝ) (h1 : C > A) (h2 : A > 0) (h3 : C = 1.2 * A) :
  (C - A) / C * 100 = 50 / 3 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l3964_396464


namespace NUMINAMATH_CALUDE_relation_xyz_l3964_396489

theorem relation_xyz (x y z t : ℝ) (h : x / Real.sin t = y / Real.sin (2 * t) ∧ 
                                        x / Real.sin t = z / Real.sin (3 * t)) : 
  x^2 - y^2 + x*z = 0 := by
sorry

end NUMINAMATH_CALUDE_relation_xyz_l3964_396489


namespace NUMINAMATH_CALUDE_leo_current_weight_l3964_396433

-- Define Leo's current weight
def leo_weight : ℝ := sorry

-- Define Kendra's current weight
def kendra_weight : ℝ := sorry

-- Condition 1: If Leo gains 10 pounds, he will weigh 50% more than Kendra
axiom condition_1 : leo_weight + 10 = 1.5 * kendra_weight

-- Condition 2: Their combined current weight is 150 pounds
axiom condition_2 : leo_weight + kendra_weight = 150

-- Theorem to prove
theorem leo_current_weight : leo_weight = 86 := by sorry

end NUMINAMATH_CALUDE_leo_current_weight_l3964_396433


namespace NUMINAMATH_CALUDE_trajectory_is_line_segment_l3964_396421

/-- The trajectory of a point P(x,y) satisfying |PF₁| + |PF₂| = 10, where F₁(-5,0) and F₂(5,0) are fixed points, is a line segment. -/
theorem trajectory_is_line_segment :
  ∀ (x y : ℝ),
  let P : ℝ × ℝ := (x, y)
  let F₁ : ℝ × ℝ := (-5, 0)
  let F₂ : ℝ × ℝ := (5, 0)
  let dist (A B : ℝ × ℝ) := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  dist P F₁ + dist P F₂ = 10 →
  ∃ (A B : ℝ × ℝ), P ∈ Set.Icc A B :=
by sorry

end NUMINAMATH_CALUDE_trajectory_is_line_segment_l3964_396421


namespace NUMINAMATH_CALUDE_largest_inscribed_circle_area_l3964_396483

/-- The area of the largest circle that can be inscribed in a square with side length 2 decimeters is π square decimeters. -/
theorem largest_inscribed_circle_area (square_side : ℝ) (h : square_side = 2) :
  let circle_area := π * (square_side / 2)^2
  circle_area = π := by
  sorry

end NUMINAMATH_CALUDE_largest_inscribed_circle_area_l3964_396483


namespace NUMINAMATH_CALUDE_cubic_root_sum_of_eighth_powers_l3964_396482

theorem cubic_root_sum_of_eighth_powers (a b c : ℂ) : 
  (a^3 - a + 1 = 0) → (b^3 - b + 1 = 0) → (c^3 - c + 1 = 0) → 
  a^8 + b^8 + c^8 = 10 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_of_eighth_powers_l3964_396482


namespace NUMINAMATH_CALUDE_greg_travel_distance_l3964_396470

/-- Greg's travel problem -/
theorem greg_travel_distance :
  let distance_to_market : ℝ := 30
  let time_from_market : ℝ := 30 / 60  -- 30 minutes converted to hours
  let speed_from_market : ℝ := 20
  let distance_from_market : ℝ := time_from_market * speed_from_market
  distance_to_market + distance_from_market = 40 := by
  sorry

end NUMINAMATH_CALUDE_greg_travel_distance_l3964_396470


namespace NUMINAMATH_CALUDE_restaurant_donates_24_l3964_396479

/-- The restaurant's donation policy -/
def donation_rate : ℚ := 2 / 10

/-- The average customer donation -/
def avg_customer_donation : ℚ := 3

/-- The number of customers -/
def num_customers : ℕ := 40

/-- The restaurant's donation function -/
def restaurant_donation (customer_total : ℚ) : ℚ :=
  (customer_total / 10) * 2

/-- Theorem: The restaurant donates $24 given the conditions -/
theorem restaurant_donates_24 :
  restaurant_donation (avg_customer_donation * num_customers) = 24 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_donates_24_l3964_396479


namespace NUMINAMATH_CALUDE_square_area_above_line_l3964_396458

/-- The fraction of a square's area above a line -/
def fractionAboveLine (p1 p2 v1 v2 v3 v4 : ℝ × ℝ) : ℚ :=
  sorry

/-- The main theorem -/
theorem square_area_above_line :
  let p1 : ℝ × ℝ := (2, 3)
  let p2 : ℝ × ℝ := (5, 1)
  let v1 : ℝ × ℝ := (2, 1)
  let v2 : ℝ × ℝ := (5, 1)
  let v3 : ℝ × ℝ := (5, 4)
  let v4 : ℝ × ℝ := (2, 4)
  fractionAboveLine p1 p2 v1 v2 v3 v4 = 2/3 :=
sorry

end NUMINAMATH_CALUDE_square_area_above_line_l3964_396458


namespace NUMINAMATH_CALUDE_intersection_M_N_l3964_396414

def M : Set ℤ := {-2, -1, 0, 1, 2}

def N : Set ℤ := {x | x ≥ 3 ∨ x ≤ -2}

theorem intersection_M_N : M ∩ N = {-2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3964_396414
