import Mathlib

namespace NUMINAMATH_CALUDE_additional_pots_in_warm_hour_l763_76374

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- The time in minutes to produce a pot when the machine is cold -/
def cold_production_time : ℕ := 6

/-- The time in minutes to produce a pot when the machine is warm -/
def warm_production_time : ℕ := 5

/-- Theorem stating the difference in pot production between warm and cold hours -/
theorem additional_pots_in_warm_hour :
  (minutes_per_hour / warm_production_time) - (minutes_per_hour / cold_production_time) = 2 :=
by sorry

end NUMINAMATH_CALUDE_additional_pots_in_warm_hour_l763_76374


namespace NUMINAMATH_CALUDE_correct_average_calculation_l763_76369

def total_numbers : ℕ := 20
def initial_average : ℚ := 35
def incorrect_numbers : List (ℚ × ℚ) := [(90, 45), (73, 36), (85, 42), (-45, -27), (64, 35)]

theorem correct_average_calculation :
  let incorrect_sum := initial_average * total_numbers
  let adjustment := (incorrect_numbers.map (λ (x : ℚ × ℚ) => x.1 - x.2)).sum
  let correct_sum := incorrect_sum + adjustment
  correct_sum / total_numbers = 41.8 := by sorry

end NUMINAMATH_CALUDE_correct_average_calculation_l763_76369


namespace NUMINAMATH_CALUDE_optimal_boat_combinations_l763_76368

/-- Represents a combination of large and small boats -/
structure BoatCombination where
  large_boats : Nat
  small_boats : Nat

/-- Checks if a boat combination is valid for the given number of people -/
def is_valid_combination (total_people : Nat) (large_capacity : Nat) (small_capacity : Nat) (combo : BoatCombination) : Prop :=
  combo.large_boats * large_capacity + combo.small_boats * small_capacity = total_people

theorem optimal_boat_combinations : 
  ∃ (combo1 combo2 : BoatCombination),
    combo1 ≠ combo2 ∧
    is_valid_combination 43 7 4 combo1 ∧
    is_valid_combination 43 7 4 combo2 :=
by sorry

end NUMINAMATH_CALUDE_optimal_boat_combinations_l763_76368


namespace NUMINAMATH_CALUDE_tax_assignment_correct_l763_76385

/-- Represents the different types of budget levels in the Russian tax system -/
inductive BudgetLevel
  | Federal
  | Regional

/-- Represents the different types of taxes in the Russian tax system -/
inductive TaxType
  | PropertyTax
  | FederalTax
  | ProfitTax
  | RegionalTax
  | TransportFee

/-- Assigns a tax to a budget level -/
def assignTax (tax : TaxType) : BudgetLevel :=
  match tax with
  | TaxType.PropertyTax => BudgetLevel.Regional
  | TaxType.FederalTax => BudgetLevel.Federal
  | TaxType.ProfitTax => BudgetLevel.Regional
  | TaxType.RegionalTax => BudgetLevel.Regional
  | TaxType.TransportFee => BudgetLevel.Regional

/-- Theorem stating the correct assignment of taxes to budget levels -/
theorem tax_assignment_correct :
  (assignTax TaxType.PropertyTax = BudgetLevel.Regional) ∧
  (assignTax TaxType.FederalTax = BudgetLevel.Federal) ∧
  (assignTax TaxType.ProfitTax = BudgetLevel.Regional) ∧
  (assignTax TaxType.RegionalTax = BudgetLevel.Regional) ∧
  (assignTax TaxType.TransportFee = BudgetLevel.Regional) :=
by sorry

end NUMINAMATH_CALUDE_tax_assignment_correct_l763_76385


namespace NUMINAMATH_CALUDE_last_two_pieces_l763_76358

def pieces : Finset Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def is_odd (n : Nat) : Bool := n % 2 = 1

def product_is_24 (s : Finset Nat) : Bool :=
  s.prod id = 24

def removal_process (s : Finset Nat) : Finset Nat :=
  let after_odd_removal := s.filter (fun n => ¬is_odd n)
  let after_product_removal := after_odd_removal.filter (fun n => ¬product_is_24 {n})
  after_product_removal

theorem last_two_pieces (s : Finset Nat) :
  s = pieces →
  (removal_process s = {2, 8} ∨ removal_process s = {6, 8}) :=
sorry

end NUMINAMATH_CALUDE_last_two_pieces_l763_76358


namespace NUMINAMATH_CALUDE_parabola_vertex_l763_76354

/-- Given a quadratic function f(x) = -x^2 + px + q where f(x) ≤ 0 has roots at x = -2 and x = 8,
    the vertex of the parabola defined by f(x) is at (3, -7). -/
theorem parabola_vertex (p q : ℝ) (f : ℝ → ℝ) 
    (h_f : ∀ x, f x = -x^2 + p*x + q)
    (h_roots : f (-2) = 0 ∧ f 8 = 0)
    (h_solution : ∀ x, x ∈ Set.Icc (-2) 8 ↔ f x ≤ 0) :
    ∃ (vertex : ℝ × ℝ), vertex = (3, -7) ∧ 
    ∀ x, f x ≤ f (vertex.1) ∧ 
    (x ≠ vertex.1 → f x < f (vertex.1)) := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l763_76354


namespace NUMINAMATH_CALUDE_square_difference_l763_76357

theorem square_difference (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l763_76357


namespace NUMINAMATH_CALUDE_quadratic_sum_l763_76301

/-- Given a quadratic function g(x) = dx^2 + ex + f, 
    if g(0) = 8 and g(1) = 5, then d + e + 2f = 13 -/
theorem quadratic_sum (d e f : ℝ) : 
  let g : ℝ → ℝ := λ x ↦ d * x^2 + e * x + f
  (g 0 = 8) → (g 1 = 5) → d + e + 2 * f = 13 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l763_76301


namespace NUMINAMATH_CALUDE_opposite_of_negative_fraction_l763_76319

theorem opposite_of_negative_fraction :
  -(-(1 / 2023)) = 1 / 2023 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_fraction_l763_76319


namespace NUMINAMATH_CALUDE_eccentricity_range_l763_76355

/-- An ellipse with foci F₁ and F₂ -/
structure Ellipse where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- A point on the ellipse -/
structure PointOnEllipse (C : Ellipse) where
  P : ℝ × ℝ
  on_ellipse : sorry -- Condition for P being on the ellipse C

/-- The distance between two points in ℝ² -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- The eccentricity of an ellipse -/
def eccentricity (C : Ellipse) : ℝ := sorry

theorem eccentricity_range (C : Ellipse) (P : PointOnEllipse C) 
  (h : distance P.P C.F₁ = 3/2 * distance C.F₁ C.F₂) : 
  1/4 ≤ eccentricity C ∧ eccentricity C ≤ 1/2 := by sorry

end NUMINAMATH_CALUDE_eccentricity_range_l763_76355


namespace NUMINAMATH_CALUDE_ratio_sum_problem_l763_76365

theorem ratio_sum_problem (a b : ℝ) : 
  a / b = 3 / 8 → b - a = 20 → a + b = 44 := by sorry

end NUMINAMATH_CALUDE_ratio_sum_problem_l763_76365


namespace NUMINAMATH_CALUDE_tom_stamp_collection_tom_final_collection_l763_76322

theorem tom_stamp_collection (tom_initial : ℕ) (mike_gift : ℕ) : ℕ :=
  let harry_gift := 2 * mike_gift + 10
  let sarah_gift := 3 * mike_gift - 5
  let total_gifts := mike_gift + harry_gift + sarah_gift
  tom_initial + total_gifts

theorem tom_final_collection :
  tom_stamp_collection 3000 17 = 3107 := by
  sorry

end NUMINAMATH_CALUDE_tom_stamp_collection_tom_final_collection_l763_76322


namespace NUMINAMATH_CALUDE_common_terms_theorem_l763_76378

def a (n : ℕ) : ℤ := 3 * n - 19
def b (n : ℕ) : ℕ := 2^n

-- c_n is the nth common term of sequences a and b in ascending order
def c (n : ℕ) : ℕ := 2^(2*n - 1)

theorem common_terms_theorem (n : ℕ) :
  ∃ (m k : ℕ), a m = b k ∧ c n = b k ∧ 
  (∀ (i j : ℕ), i < m ∧ j < k → a i ≠ b j) ∧
  (∀ (i j : ℕ), a i = b j → i ≥ m ∨ j ≥ k) :=
sorry

end NUMINAMATH_CALUDE_common_terms_theorem_l763_76378


namespace NUMINAMATH_CALUDE_right_triangle_identification_l763_76371

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem right_triangle_identification :
  ¬ is_right_triangle 2 3 4 ∧
  ¬ is_right_triangle (Real.sqrt 3) (Real.sqrt 4) (Real.sqrt 5) ∧
  is_right_triangle 3 4 5 ∧
  ¬ is_right_triangle 7 8 9 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_identification_l763_76371


namespace NUMINAMATH_CALUDE_rectangle_13_squares_ratio_l763_76326

/-- A rectangle that can be divided into 13 equal squares -/
structure Rectangle13Squares where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b
  h_divisible : ∃ s : ℝ, 0 < s ∧ (a = 13 * s ∧ b = s) ∨ (a = s ∧ b = 13 * s)

/-- The ratio of the longer side to the shorter side is 13:1 -/
theorem rectangle_13_squares_ratio (rect : Rectangle13Squares) :
  (max rect.a rect.b) / (min rect.a rect.b) = 13 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_13_squares_ratio_l763_76326


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l763_76356

theorem polynomial_divisibility (p q : ℚ) : 
  (∀ x : ℚ, (x + 1) * (x - 2) ∣ (x^5 - x^4 + x^3 - p*x^2 + q*x - 6)) ↔ 
  (p = 0 ∧ q = -9) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l763_76356


namespace NUMINAMATH_CALUDE_february_first_is_monday_l763_76384

/-- Represents the days of the week -/
inductive Weekday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a day in February -/
structure FebruaryDay where
  day : Nat
  weekday : Weekday

/-- Defines the properties of February in the given year -/
structure FebruaryProperties where
  days : List FebruaryDay
  monday_count : Nat
  thursday_count : Nat
  first_day : Weekday

/-- Theorem stating that if February has exactly four Mondays and four Thursdays,
    then February 1 must be a Monday -/
theorem february_first_is_monday
  (feb : FebruaryProperties)
  (h1 : feb.monday_count = 4)
  (h2 : feb.thursday_count = 4)
  : feb.first_day = Weekday.Monday := by
  sorry

end NUMINAMATH_CALUDE_february_first_is_monday_l763_76384


namespace NUMINAMATH_CALUDE_initial_mushroom_amount_l763_76387

-- Define the initial amount of mushrooms
def initial_amount : ℕ := sorry

-- Define the amount of mushrooms eaten
def eaten_amount : ℕ := 8

-- Define the amount of mushrooms left
def left_amount : ℕ := 7

-- Theorem stating that the initial amount is 15 pounds
theorem initial_mushroom_amount :
  initial_amount = eaten_amount + left_amount ∧ initial_amount = 15 := by
  sorry

end NUMINAMATH_CALUDE_initial_mushroom_amount_l763_76387


namespace NUMINAMATH_CALUDE_equation_one_solution_system_of_equations_solution_l763_76324

-- Equation (1)
theorem equation_one_solution (x : ℝ) : 
  2 * (x - 2) - 3 * (4 * x - 1) = 9 * (1 - x) ↔ x = -10 := by sorry

-- System of Equations (2)
theorem system_of_equations_solution (x y : ℝ) :
  (4 * (x - y - 1) = 3 * (1 - y) - 2 ∧ x / 2 + y / 3 = 2) ↔ (x = 2 ∧ y = 3) := by sorry

end NUMINAMATH_CALUDE_equation_one_solution_system_of_equations_solution_l763_76324


namespace NUMINAMATH_CALUDE_flea_difference_l763_76343

def flea_treatment (initial_fleas : ℕ) (treatments : ℕ) : ℕ :=
  initial_fleas / (2^treatments)

theorem flea_difference (initial_fleas : ℕ) :
  flea_treatment initial_fleas 4 = 14 →
  initial_fleas - flea_treatment initial_fleas 4 = 210 := by
sorry

end NUMINAMATH_CALUDE_flea_difference_l763_76343


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_l763_76377

theorem complex_magnitude_equation (x : ℝ) (h1 : x > 0) :
  Complex.abs (3 + 4 * x * Complex.I) = 5 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_l763_76377


namespace NUMINAMATH_CALUDE_first_number_value_l763_76329

theorem first_number_value (y x : ℚ) : 
  (y + 76 + x) / 3 = 5 → x = -63 → y = 2 := by
  sorry

end NUMINAMATH_CALUDE_first_number_value_l763_76329


namespace NUMINAMATH_CALUDE_recycling_team_points_l763_76353

/-- Represents the recycling data for a team member -/
structure RecyclingData where
  paper : Nat
  plastic : Nat
  aluminum : Nat

/-- Calculates the points earned for a given recycling data -/
def calculate_points (data : RecyclingData) : Nat :=
  (data.paper / 12) + (data.plastic / 6) + (data.aluminum / 4)

/-- The recycling data for each team member -/
def team_data : List RecyclingData := [
  { paper := 35, plastic := 15, aluminum := 5 },   -- Zoe
  { paper := 28, plastic := 18, aluminum := 8 },   -- Friend 1
  { paper := 22, plastic := 10, aluminum := 6 },   -- Friend 2
  { paper := 40, plastic := 20, aluminum := 10 },  -- Friend 3
  { paper := 18, plastic := 12, aluminum := 8 }    -- Friend 4
]

/-- Theorem: The recycling team earned 28 points -/
theorem recycling_team_points : 
  (team_data.map calculate_points).sum = 28 := by
  sorry

end NUMINAMATH_CALUDE_recycling_team_points_l763_76353


namespace NUMINAMATH_CALUDE_longest_wait_time_l763_76363

def initial_wait : ℕ := 20

def license_renewal_wait (t : ℕ) : ℕ := 2 * t + 8

def registration_update_wait (t : ℕ) : ℕ := 4 * t + 14

def driving_record_wait (t : ℕ) : ℕ := 3 * t - 16

theorem longest_wait_time :
  let tasks := [initial_wait,
                license_renewal_wait initial_wait,
                registration_update_wait initial_wait,
                driving_record_wait initial_wait]
  registration_update_wait initial_wait = 94 ∧
  ∀ t ∈ tasks, t ≤ registration_update_wait initial_wait :=
by sorry

end NUMINAMATH_CALUDE_longest_wait_time_l763_76363


namespace NUMINAMATH_CALUDE_sleep_hours_calculation_l763_76316

def hours_in_day : ℕ := 24
def work_hours : ℕ := 6
def chore_hours : ℕ := 5

theorem sleep_hours_calculation :
  hours_in_day - (work_hours + chore_hours) = 13 := by
  sorry

end NUMINAMATH_CALUDE_sleep_hours_calculation_l763_76316


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l763_76398

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x + 1| > 2
def q (x : ℝ) : Prop := x > 2

-- State the theorem
theorem not_p_sufficient_not_necessary_for_not_q :
  (∀ x, ¬(p x) → ¬(q x)) ∧ 
  ¬(∀ x, ¬(q x) → ¬(p x)) := by
  sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l763_76398


namespace NUMINAMATH_CALUDE_linear_function_point_comparison_l763_76393

/-- 
Given a linear function y = kx + 3 with negative slope k,
prove that for points A(-3, y₁) and B(1, y₂) on this function,
y₁ is greater than y₂.
-/
theorem linear_function_point_comparison 
  (k : ℝ) (y₁ y₂ : ℝ) 
  (h_k_neg : k < 0)
  (h_y₁ : y₁ = k * (-3) + 3)
  (h_y₂ : y₂ = k * 1 + 3) :
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_linear_function_point_comparison_l763_76393


namespace NUMINAMATH_CALUDE_parabola_focus_focus_of_specific_parabola_l763_76311

/-- The focus of a parabola y = ax^2 + c is at (0, 1/(4a) + c) -/
theorem parabola_focus (a c : ℝ) (h : a ≠ 0) :
  let f : ℝ × ℝ := (0, 1/(4*a) + c)
  ∀ x y : ℝ, y = a * x^2 + c → (x - f.1)^2 + (y - f.2)^2 = (y - c + 1/(4*a))^2 :=
by sorry

/-- The focus of the parabola y = 9x^2 - 5 is at (0, -179/36) -/
theorem focus_of_specific_parabola :
  let f : ℝ × ℝ := (0, -179/36)
  ∀ x y : ℝ, y = 9 * x^2 - 5 → (x - f.1)^2 + (y - f.2)^2 = (y + 5 + 1/36)^2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_focus_of_specific_parabola_l763_76311


namespace NUMINAMATH_CALUDE_count_numbers_theorem_l763_76392

/-- A function that checks if a natural number contains the digit 4 in its decimal representation -/
def contains_four (n : ℕ) : Prop := sorry

/-- The count of numbers from 1 to 1000 that are divisible by 4 and do not contain the digit 4 -/
def count_numbers : ℕ := sorry

/-- Theorem stating that the count of numbers from 1 to 1000 that are divisible by 4 
    and do not contain the digit 4 is equal to 162 -/
theorem count_numbers_theorem : count_numbers = 162 := by sorry

end NUMINAMATH_CALUDE_count_numbers_theorem_l763_76392


namespace NUMINAMATH_CALUDE_color_film_fraction_l763_76334

theorem color_film_fraction (x y : ℝ) (x_pos : x > 0) (y_pos : y > 0) : 
  let total_bw := 20 * x
  let total_color := 8 * y
  let selected_bw := y / (5 * x) * total_bw
  let selected_color := total_color
  let total_selected := selected_bw + selected_color
  selected_color / total_selected = 40 / 41 := by
sorry

end NUMINAMATH_CALUDE_color_film_fraction_l763_76334


namespace NUMINAMATH_CALUDE_five_by_seven_not_tileable_l763_76340

/-- Represents a rectangular board -/
structure Board :=
  (length : ℕ)
  (width : ℕ)

/-- Represents a domino -/
structure Domino :=
  (length : ℕ)
  (width : ℕ)

/-- Checks if a board can be tiled with dominos -/
def can_be_tiled (b : Board) (d : Domino) : Prop :=
  (b.length * b.width) % (d.length * d.width) = 0

/-- The theorem stating that a 5×7 board cannot be tiled with 2×1 dominos -/
theorem five_by_seven_not_tileable :
  ¬(can_be_tiled (Board.mk 5 7) (Domino.mk 2 1)) :=
sorry

end NUMINAMATH_CALUDE_five_by_seven_not_tileable_l763_76340


namespace NUMINAMATH_CALUDE_cost_price_calculation_l763_76338

/-- Proves that the cost price of an article is 480, given the selling price and profit percentage -/
theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) : 
  selling_price = 595.2 → 
  profit_percentage = 24 → 
  ∃ (cost_price : ℝ), 
    cost_price = 480 ∧ 
    selling_price = cost_price * (1 + profit_percentage / 100) := by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l763_76338


namespace NUMINAMATH_CALUDE_august_math_problems_l763_76379

theorem august_math_problems (first_answer second_answer third_answer : ℕ) : 
  first_answer = 600 →
  second_answer = 2 * first_answer →
  third_answer = first_answer + second_answer - 400 →
  first_answer + second_answer + third_answer = 3200 := by
sorry

end NUMINAMATH_CALUDE_august_math_problems_l763_76379


namespace NUMINAMATH_CALUDE_min_third_side_length_l763_76332

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c

/-- Checks if the given sides satisfy the triangle inequality -/
def satisfies_triangle_inequality (a b c : ℕ+) : Prop :=
  a + b > c ∧ b + c > a ∧ a + c > b

/-- Theorem: The minimum length of the third side in a triangle with two sides
    being multiples of 42 and 72 respectively is 7 -/
theorem min_third_side_length (t : Triangle) 
    (h1 : ∃ (k : ℕ+), t.a = 42 * k ∨ t.b = 42 * k ∨ t.c = 42 * k)
    (h2 : ∃ (m : ℕ+), t.a = 72 * m ∨ t.b = 72 * m ∨ t.c = 72 * m)
    (h3 : satisfies_triangle_inequality t.a t.b t.c) :
    min t.a (min t.b t.c) ≥ 7 := by
  sorry

end NUMINAMATH_CALUDE_min_third_side_length_l763_76332


namespace NUMINAMATH_CALUDE_race_result_l763_76348

/-- Represents a runner in a race -/
structure Runner where
  speed : ℝ
  time : ℝ

/-- Calculates the distance traveled by a runner given time -/
def distanceTraveled (runner : Runner) (t : ℝ) : ℝ :=
  runner.speed * t

/-- The race problem setup -/
def raceProblem : Prop :=
  ∃ (A B : Runner),
    -- The race is 1000 meters long
    distanceTraveled A A.time = 1000 ∧
    -- A finishes the race in 115 seconds
    A.time = 115 ∧
    -- B finishes 10 seconds after A
    B.time = A.time + 10 ∧
    -- The distance by which A beats B is 80 meters
    1000 - distanceTraveled B A.time = 80

theorem race_result : raceProblem := by
  sorry

#check race_result

end NUMINAMATH_CALUDE_race_result_l763_76348


namespace NUMINAMATH_CALUDE_distance_to_cemetery_l763_76345

/-- The distance from the school to the Martyrs' Cemetery in kilometers. -/
def distance : ℝ := 216

/-- The original scheduled time for the journey in minutes. -/
def scheduled_time : ℝ := 180

/-- The time saved in minutes when the bus increases speed by 1/5 after 1 hour. -/
def time_saved_1 : ℝ := 20

/-- The time saved in minutes when the bus increases speed by 1/3 after 72 km. -/
def time_saved_2 : ℝ := 30

/-- The distance traveled at original speed before increasing speed by 1/3. -/
def initial_distance : ℝ := 72

theorem distance_to_cemetery :
  (1 + 1/5) * (scheduled_time - 60 - time_saved_1) = scheduled_time - 60 ∧
  (1 + 1/3) * (scheduled_time - time_saved_2) = scheduled_time ∧
  distance = initial_distance / (1 - 2/3) := by
  sorry

end NUMINAMATH_CALUDE_distance_to_cemetery_l763_76345


namespace NUMINAMATH_CALUDE_trig_identity_l763_76360

theorem trig_identity (x : ℝ) (h : Real.sin (x + π/6) = 1/4) :
  Real.sin (5*π/6 - x) + (Real.cos (π/3 - x))^2 = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l763_76360


namespace NUMINAMATH_CALUDE_safe_cracking_l763_76342

def Password := Fin 10 → Fin 10

def isValidPassword (p : Password) : Prop :=
  (∀ i j : Fin 7, i ≠ j → p i ≠ p j) ∧ (∀ i : Fin 7, p i < 10)

def Attempt := Fin 7 → Fin 10

def isSuccessfulAttempt (p : Password) (a : Attempt) : Prop :=
  ∃ i : Fin 7, p i = a i

theorem safe_cracking (p : Password) (h : isValidPassword p) :
  ∃ attempts : Fin 6 → Attempt,
    ∃ i : Fin 6, isSuccessfulAttempt p (attempts i) :=
sorry

end NUMINAMATH_CALUDE_safe_cracking_l763_76342


namespace NUMINAMATH_CALUDE_perpendicular_and_parallel_relationships_l763_76323

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- State the theorem
theorem perpendicular_and_parallel_relationships 
  (l m : Line) (α β : Plane)
  (h1 : perpendicular_line_plane l α)
  (h2 : contained_in m β) :
  (parallel_planes α β → perpendicular_lines l m) ∧
  (parallel_lines l m → perpendicular_planes α β) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_and_parallel_relationships_l763_76323


namespace NUMINAMATH_CALUDE_scientific_notation_218_million_l763_76305

theorem scientific_notation_218_million :
  ∃ (a : ℝ) (n : ℤ), 
    1 ≤ a ∧ a < 10 ∧ 
    218000000 = a * (10 : ℝ) ^ n ∧
    a = 2.18 ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_218_million_l763_76305


namespace NUMINAMATH_CALUDE_task_completion_theorem_l763_76347

/-- Represents the number of workers and days to complete a task. -/
structure WorkerDays where
  workers : ℕ
  days : ℕ

/-- Represents the conditions of the problem. -/
structure TaskConditions where
  original : WorkerDays
  reduced : WorkerDays
  increased : WorkerDays

/-- The theorem to prove based on the given conditions. -/
theorem task_completion_theorem (conditions : TaskConditions) : 
  conditions.original.workers = 60 ∧ conditions.original.days = 10 :=
by
  have h1 : conditions.reduced.workers = conditions.original.workers - 20 := by sorry
  have h2 : conditions.reduced.days = conditions.original.days + 5 := by sorry
  have h3 : conditions.increased.workers = conditions.original.workers + 15 := by sorry
  have h4 : conditions.increased.days = conditions.original.days - 2 := by sorry
  have h5 : conditions.original.workers * conditions.original.days = 
            conditions.reduced.workers * conditions.reduced.days := by sorry
  have h6 : conditions.original.workers * conditions.original.days = 
            conditions.increased.workers * conditions.increased.days := by sorry
  sorry

end NUMINAMATH_CALUDE_task_completion_theorem_l763_76347


namespace NUMINAMATH_CALUDE_prob_reach_vertical_from_1_2_l763_76333

/-- Represents a point on a 2D grid -/
structure Point where
  x : Int
  y : Int

/-- Represents the square boundary -/
def square_boundary : Set Point :=
  {p | p.x = 0 ∨ p.x = 4 ∨ p.y = 0 ∨ p.y = 4}

/-- Represents the vertical sides of the square -/
def vertical_sides : Set Point :=
  {p | (p.x = 0 ∨ p.x = 4) ∧ 0 ≤ p.y ∧ p.y ≤ 4}

/-- Represents a single random jump -/
def random_jump (p : Point) : Set Point :=
  {q | (|q.x - p.x| + |q.y - p.y| = 1) ∧ q ∈ square_boundary}

/-- The probability of reaching a vertical side from a given starting point -/
noncomputable def prob_reach_vertical (start : Point) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem prob_reach_vertical_from_1_2 :
  prob_reach_vertical ⟨1, 2⟩ = 5/8 := by sorry

end NUMINAMATH_CALUDE_prob_reach_vertical_from_1_2_l763_76333


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l763_76309

theorem consecutive_integers_sum (n : ℕ) (h1 : n > 0) 
  (h2 : n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) = 2070) :
  n + 5 = 347 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l763_76309


namespace NUMINAMATH_CALUDE_building_height_calculation_l763_76310

/-- Given a flagstaff and a building casting shadows under the same sun angle, 
    calculate the height of the building. -/
theorem building_height_calculation 
  (flagstaff_height : ℝ) 
  (flagstaff_shadow : ℝ) 
  (building_shadow : ℝ) 
  (h_flagstaff : flagstaff_height = 17.5) 
  (h_flagstaff_shadow : flagstaff_shadow = 40.25)
  (h_building_shadow : building_shadow = 28.75) :
  (flagstaff_height * building_shadow) / flagstaff_shadow = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_building_height_calculation_l763_76310


namespace NUMINAMATH_CALUDE_tracy_balloons_l763_76325

theorem tracy_balloons (brooke_initial : ℕ) (brooke_added : ℕ) (tracy_initial : ℕ) (total_after : ℕ) :
  brooke_initial = 12 →
  brooke_added = 8 →
  tracy_initial = 6 →
  total_after = 35 →
  ∃ (tracy_added : ℕ),
    brooke_initial + brooke_added + (tracy_initial + tracy_added) / 2 = total_after ∧
    tracy_added = 24 :=
by sorry

end NUMINAMATH_CALUDE_tracy_balloons_l763_76325


namespace NUMINAMATH_CALUDE_sum_and_fraction_difference_l763_76388

theorem sum_and_fraction_difference (x y : ℝ) 
  (h1 : x + y = 480) 
  (h2 : x / y = 0.8) : 
  y - x = 53.34 := by sorry

end NUMINAMATH_CALUDE_sum_and_fraction_difference_l763_76388


namespace NUMINAMATH_CALUDE_colored_paper_problem_l763_76396

/-- The initial number of colored paper pieces Yuna had -/
def initial_yuna : ℕ := 100

/-- The initial number of colored paper pieces Yoojung had -/
def initial_yoojung : ℕ := 210

/-- The number of pieces Yoojung gave to Yuna -/
def transferred : ℕ := 30

/-- The difference in pieces after the transfer -/
def difference : ℕ := 50

theorem colored_paper_problem :
  initial_yuna = 100 ∧
  initial_yoojung = 210 ∧
  transferred = 30 ∧
  difference = 50 ∧
  initial_yoojung - transferred = initial_yuna + transferred + difference :=
by sorry

end NUMINAMATH_CALUDE_colored_paper_problem_l763_76396


namespace NUMINAMATH_CALUDE_find_multiple_l763_76312

theorem find_multiple (n m : ℝ) (h1 : n + n + m * n + 4 * n = 104) (h2 : n = 13) : m = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_multiple_l763_76312


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l763_76308

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

/-- The theorem states that for a geometric sequence where a_1 * a_5 = a_3, the value of a_3 is 1. -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geom : IsGeometricSequence a) 
  (h_prop : a 1 * a 5 = a 3) : 
  a 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l763_76308


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l763_76302

theorem sum_of_reciprocals_of_roots (x₁ x₂ : ℝ) : 
  x₁^2 - 14*x₁ + 8 = 0 → 
  x₂^2 - 14*x₂ + 8 = 0 → 
  x₁ ≠ x₂ →
  1/x₁ + 1/x₂ = 7/4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l763_76302


namespace NUMINAMATH_CALUDE_f_properties_l763_76327

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + a * Real.exp x

theorem f_properties :
  ∀ a : ℝ,
  (a = -1 →
    (∀ x y : ℝ, x < y → x < 0 → y < 0 → f a x < f a y) ∧
    (∀ x y : ℝ, x < y → x > 0 → y > 0 → f a x > f a y)) ∧
  (a ≥ 0 →
    ∀ x : ℝ, ¬∃ y : ℝ, (∀ z : ℝ, f a z ≤ f a y) ∨ (∀ z : ℝ, f a z ≥ f a y)) ∧
  (a < 0 →
    (∃ x : ℝ, ∀ y : ℝ, f a y ≤ f a x) ∧
    (¬∃ x : ℝ, ∀ y : ℝ, f a y ≥ f a x) ∧
    (∃ x : ℝ, x = Real.log (-1/a) ∧ ∀ y : ℝ, f a y ≤ f a x) ∧
    (∃ x : ℝ, x = Real.log (-1/a) ∧ f a x = Real.log (-1/a) - 1)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l763_76327


namespace NUMINAMATH_CALUDE_unique_solution_square_equation_l763_76361

theorem unique_solution_square_equation :
  ∀ x y : ℕ+, x^2 = y^2 + 7*y + 6 ↔ x = 6 ∧ y = 3 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_square_equation_l763_76361


namespace NUMINAMATH_CALUDE_square_side_length_l763_76375

theorem square_side_length (k : ℝ) (s d : ℝ) (h1 : s > 0) (h2 : d > 0) (h3 : s + d = k) (h4 : d = s * Real.sqrt 2) :
  s = k / (1 + Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_square_side_length_l763_76375


namespace NUMINAMATH_CALUDE_f_max_value_l763_76306

/-- The quadratic function f(x) = -x^2 + 2x + 4 -/
def f (x : ℝ) : ℝ := -x^2 + 2*x + 4

/-- The maximum value of f(x) is 5 -/
theorem f_max_value : ∃ (M : ℝ), M = 5 ∧ ∀ (x : ℝ), f x ≤ M :=
  sorry

end NUMINAMATH_CALUDE_f_max_value_l763_76306


namespace NUMINAMATH_CALUDE_samuel_fraction_l763_76391

theorem samuel_fraction (total : ℝ) (spent : ℝ) (left : ℝ) :
  total = 240 →
  spent = (1 / 5) * total →
  left = 132 →
  (left + spent) / total = 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_samuel_fraction_l763_76391


namespace NUMINAMATH_CALUDE_even_function_theorem_l763_76339

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The domain of a function -/
def Domain (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x, x ∈ s ↔ ∃ y, f x = y

theorem even_function_theorem (a b : ℝ) :
  let f := fun (x : ℝ) ↦ a * x^2 + b * x + 3 * a + b
  IsEven f ∧ Domain f (Set.Icc (a - 1) (2 * a)) →
  a + b = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_even_function_theorem_l763_76339


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l763_76394

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| - 2 * |x + a|

-- Part 1: Solution set for f(x) > 1 when a = 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x > 1} = {x : ℝ | -2 < x ∧ x < -2/3} :=
sorry

-- Part 2: Range of a for f(x) > 0 when x ∈ [2, 3]
theorem range_of_a_part2 :
  {a : ℝ | ∀ x ∈ Set.Icc 2 3, f a x > 0} = {a : ℝ | -5/2 < a ∧ a < -2} :=
sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l763_76394


namespace NUMINAMATH_CALUDE_leadership_selection_l763_76386

/-- The number of ways to choose a president, vice-president, and committee from a group --/
def choose_leadership (total : ℕ) (committee_size : ℕ) : ℕ :=
  total * (total - 1) * (Nat.choose (total - 2) committee_size)

/-- The problem statement --/
theorem leadership_selection :
  choose_leadership 10 3 = 5040 := by
  sorry

end NUMINAMATH_CALUDE_leadership_selection_l763_76386


namespace NUMINAMATH_CALUDE_larry_channels_l763_76337

/-- Calculates the final number of channels for Larry given the initial count and subsequent changes. -/
def final_channels (initial : ℕ) (removed : ℕ) (replaced : ℕ) (reduced : ℕ) (sports : ℕ) (supreme : ℕ) : ℕ :=
  initial - removed + replaced - reduced + sports + supreme

/-- Theorem stating that Larry's final channel count is 147 given the specific changes. -/
theorem larry_channels : 
  final_channels 150 20 12 10 8 7 = 147 := by
  sorry

end NUMINAMATH_CALUDE_larry_channels_l763_76337


namespace NUMINAMATH_CALUDE_johnny_work_hours_l763_76382

/-- Given Johnny's hourly wage and total earnings, prove the number of hours he worked -/
theorem johnny_work_hours (hourly_wage : ℝ) (total_earnings : ℝ) (h1 : hourly_wage = 6.75) (h2 : total_earnings = 67.5) :
  total_earnings / hourly_wage = 10 := by
  sorry

end NUMINAMATH_CALUDE_johnny_work_hours_l763_76382


namespace NUMINAMATH_CALUDE_molly_total_distance_l763_76381

/-- The total distance Molly swam over two days -/
def total_distance (saturday_distance sunday_distance : ℕ) : ℕ :=
  saturday_distance + sunday_distance

/-- Theorem stating that Molly's total swimming distance is 430 meters -/
theorem molly_total_distance : total_distance 250 180 = 430 := by
  sorry

end NUMINAMATH_CALUDE_molly_total_distance_l763_76381


namespace NUMINAMATH_CALUDE_right_triangle_existence_l763_76351

theorem right_triangle_existence (a q : ℝ) (ha : a > 0) (hq : q > 0) :
  ∃ (b c : ℝ), 
    b > 0 ∧ c > 0 ∧
    a^2 + b^2 = c^2 ∧
    (b^2 / c) = q :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_existence_l763_76351


namespace NUMINAMATH_CALUDE_polynomial_expansion_l763_76328

theorem polynomial_expansion (x : ℝ) : 
  (x^3 - 3*x^2 + 3*x - 1) * (x^2 + 3*x + 3) = x^5 - 3*x^3 - x^2 + 3*x := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l763_76328


namespace NUMINAMATH_CALUDE_go_games_theorem_l763_76359

/-- The number of complete Go games that can be played simultaneously -/
def maxSimultaneousGames (totalBalls : ℕ) (ballsPerGame : ℕ) : ℕ :=
  totalBalls / ballsPerGame

theorem go_games_theorem :
  maxSimultaneousGames 901 53 = 17 := by
  sorry

end NUMINAMATH_CALUDE_go_games_theorem_l763_76359


namespace NUMINAMATH_CALUDE_min_value_of_f_l763_76304

noncomputable def f (x : ℝ) := 12 * x - x^3

theorem min_value_of_f :
  ∃ (m : ℝ), m = -16 ∧ ∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 → f x ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l763_76304


namespace NUMINAMATH_CALUDE_four_three_eight_nine_has_two_prime_products_l763_76395

-- Define set C
def C : Set ℕ := {n : ℕ | ∃ k : ℕ, n = 4 * k + 1}

-- Define prime with respect to C
def isPrimeWrtC (k : ℕ) : Prop :=
  k ∈ C ∧ ∀ a b : ℕ, a ∈ C → b ∈ C → k ≠ a * b

-- Define the property of being expressible as product of two primes wrt C in two ways
def hasTwoPrimeProductsInC (n : ℕ) : Prop :=
  ∃ a b c d : ℕ,
    a ≠ c ∧ b ≠ d ∧
    n = a * b ∧ n = c * d ∧
    isPrimeWrtC a ∧ isPrimeWrtC b ∧ isPrimeWrtC c ∧ isPrimeWrtC d

-- Theorem statement
theorem four_three_eight_nine_has_two_prime_products :
  4389 ∈ C ∧ hasTwoPrimeProductsInC 4389 :=
sorry

end NUMINAMATH_CALUDE_four_three_eight_nine_has_two_prime_products_l763_76395


namespace NUMINAMATH_CALUDE_unique_prime_fraction_l763_76349

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem unique_prime_fraction :
  ∀ a b : ℕ,
    a > 0 →
    b > 0 →
    a ≠ b →
    is_prime (a * b^2 / (a + b)) →
    a = 6 ∧ b = 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_fraction_l763_76349


namespace NUMINAMATH_CALUDE_smallest_integer_l763_76366

theorem smallest_integer (a b : ℕ) (ha : a = 60) (h_lcm_gcd : Nat.lcm a b / Nat.gcd a b = 44) :
  b ≥ 165 ∧ ∃ (b' : ℕ), b' = 165 ∧ Nat.lcm a b' / Nat.gcd a b' = 44 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_l763_76366


namespace NUMINAMATH_CALUDE_not_divisible_by_four_l763_76389

def a : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => a n * a (n + 1) + 1

theorem not_divisible_by_four : ¬ (4 ∣ a 2008) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_four_l763_76389


namespace NUMINAMATH_CALUDE_solve_system_l763_76372

theorem solve_system (x y : ℝ) 
  (eq1 : 2 * x - y = 5) 
  (eq2 : x + 2 * y = 5) : 
  x = 3 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l763_76372


namespace NUMINAMATH_CALUDE_quadratic_completing_square_l763_76317

theorem quadratic_completing_square (x : ℝ) :
  x^2 + 8*x + 7 = 0 ↔ (x + 4)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_completing_square_l763_76317


namespace NUMINAMATH_CALUDE_chalkboard_width_l763_76303

theorem chalkboard_width (width : ℝ) (length : ℝ) (area : ℝ) : 
  length = 2 * width →
  area = width * length →
  area = 18 →
  width = 3 :=
by sorry

end NUMINAMATH_CALUDE_chalkboard_width_l763_76303


namespace NUMINAMATH_CALUDE_stationery_store_profit_l763_76318

/-- Profit data for a week at a stationery store -/
structure WeekProfit :=
  (mon tue wed thu fri sat sun : ℝ)
  (total : ℝ)
  (sum_condition : mon + tue + wed + thu + fri + sat + sun = total)

/-- Theorem stating the properties of the profit data -/
theorem stationery_store_profit 
  (w : WeekProfit)
  (h1 : w.mon = -27.8)
  (h2 : w.tue = -70.3)
  (h3 : w.wed = 200)
  (h4 : w.thu = 138.1)
  (h5 : w.sun = 188)
  (h6 : w.total = 458) :
  (w.fri = -8 → w.sat = 38) ∧
  (w.sat = w.fri + 10 → w.sat = 20) ∧
  (w.fri < 0 → w.sat > 0 → w.sat > 30) :=
by sorry

end NUMINAMATH_CALUDE_stationery_store_profit_l763_76318


namespace NUMINAMATH_CALUDE_two_polygons_edges_l763_76383

theorem two_polygons_edges (a b : ℕ) : 
  a + b = 2014 →
  a * (a - 3) / 2 + b * (b - 3) / 2 = 1014053 →
  a ≤ b →
  a = 952 := by sorry

end NUMINAMATH_CALUDE_two_polygons_edges_l763_76383


namespace NUMINAMATH_CALUDE_two_points_determine_line_l763_76364

-- Define a Point type
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define a Line type
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

-- Two points determine a unique line
theorem two_points_determine_line (P Q : Point) (h : P ≠ Q) :
  ∃! L : Line, (L.a * P.x + L.b * P.y + L.c = 0) ∧ (L.a * Q.x + L.b * Q.y + L.c = 0) :=
sorry

end NUMINAMATH_CALUDE_two_points_determine_line_l763_76364


namespace NUMINAMATH_CALUDE_ice_cube_volume_l763_76390

theorem ice_cube_volume (original_volume : ℝ) : 
  (original_volume > 0) →
  (original_volume * (1/4) * (1/4) = 0.4) →
  original_volume = 6.4 := by
sorry

end NUMINAMATH_CALUDE_ice_cube_volume_l763_76390


namespace NUMINAMATH_CALUDE_distribution_scheme_count_l763_76346

/-- The number of ways to choose 2 items from 4 items -/
def choose_4_2 : ℕ := 6

/-- The number of ways to arrange 3 items in 3 positions -/
def arrange_3_3 : ℕ := 6

/-- The number of ways to distribute 4 students into 3 laboratories -/
def distribute_students : ℕ := choose_4_2 * arrange_3_3

theorem distribution_scheme_count :
  distribute_students = 36 :=
by sorry

end NUMINAMATH_CALUDE_distribution_scheme_count_l763_76346


namespace NUMINAMATH_CALUDE_letter_lock_max_letters_l763_76331

theorem letter_lock_max_letters (n : ℕ) : 
  (n ^ 3 - 1 ≤ 215) ∧ (∀ m : ℕ, m > n → m ^ 3 - 1 > 215) → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_letter_lock_max_letters_l763_76331


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l763_76313

theorem rectangle_perimeter (square_perimeter : ℝ) (h : square_perimeter = 100) :
  let square_side := square_perimeter / 4
  let rectangle_length := square_side
  let rectangle_width := square_side / 2
  2 * (rectangle_length + rectangle_width) = 75 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l763_76313


namespace NUMINAMATH_CALUDE_xiao_ming_envelopes_l763_76336

def red_envelopes : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

def total_sum : ℕ := red_envelopes.sum

def each_person_sum : ℕ := total_sum / 3

def father_envelopes : List ℕ := [1, 3]
def mother_envelopes : List ℕ := [8, 9]

theorem xiao_ming_envelopes :
  ∀ (xm : List ℕ),
    xm.length = 4 →
    father_envelopes.length = 4 →
    mother_envelopes.length = 4 →
    xm.sum = each_person_sum →
    father_envelopes.sum = each_person_sum →
    mother_envelopes.sum = each_person_sum →
    (∀ x ∈ xm, x ∈ red_envelopes) →
    (∀ x ∈ father_envelopes, x ∈ red_envelopes) →
    (∀ x ∈ mother_envelopes, x ∈ red_envelopes) →
    (∀ x ∈ red_envelopes, x ∈ xm ∨ x ∈ father_envelopes ∨ x ∈ mother_envelopes) →
    6 ∈ xm ∧ 11 ∈ xm :=
by
  sorry

#check xiao_ming_envelopes

end NUMINAMATH_CALUDE_xiao_ming_envelopes_l763_76336


namespace NUMINAMATH_CALUDE_perfect_score_correct_l763_76362

/-- The perfect score for a single game, given that 3 perfect games result in 63 points. -/
def perfect_score : ℕ := 21

/-- The total score for three perfect games. -/
def three_game_score : ℕ := 63

/-- Theorem stating that the perfect score for a single game is correct. -/
theorem perfect_score_correct : perfect_score * 3 = three_game_score := by
  sorry

end NUMINAMATH_CALUDE_perfect_score_correct_l763_76362


namespace NUMINAMATH_CALUDE_a_fourth_plus_b_fourth_l763_76397

theorem a_fourth_plus_b_fourth (a b : ℝ) 
  (h1 : a^2 - b^2 = 8) 
  (h2 : a * b = 2) : 
  a^4 + b^4 = 56 := by
sorry

end NUMINAMATH_CALUDE_a_fourth_plus_b_fourth_l763_76397


namespace NUMINAMATH_CALUDE_product_of_distinct_solutions_l763_76315

theorem product_of_distinct_solutions (x y : ℝ) : 
  x ≠ 0 → y ≠ 0 → x ≠ y → (x + 3 / x = y + 3 / y) → x * y = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_of_distinct_solutions_l763_76315


namespace NUMINAMATH_CALUDE_irregular_quadrilateral_tiles_plane_l763_76344

-- Define an irregular quadrilateral
structure IrregularQuadrilateral where
  vertices : Fin 4 → ℝ × ℝ

-- Define a tiling of the plane
def PlaneTiling (Q : Type) := ℝ × ℝ → Q

-- Define the property of being a valid tiling (no gaps or overlaps)
def IsValidTiling (Q : Type) (tiling : PlaneTiling Q) : Prop := sorry

-- Theorem statement
theorem irregular_quadrilateral_tiles_plane (q : IrregularQuadrilateral) :
  ∃ (tiling : PlaneTiling IrregularQuadrilateral), IsValidTiling IrregularQuadrilateral tiling :=
sorry

end NUMINAMATH_CALUDE_irregular_quadrilateral_tiles_plane_l763_76344


namespace NUMINAMATH_CALUDE_largest_angle_cosine_l763_76320

theorem largest_angle_cosine (a b c : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : c = 4) :
  let cos_largest := min (min ((a^2 + b^2 - c^2) / (2*a*b)) ((b^2 + c^2 - a^2) / (2*b*c))) ((c^2 + a^2 - b^2) / (2*c*a))
  cos_largest = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_cosine_l763_76320


namespace NUMINAMATH_CALUDE_p_or_q_is_true_l763_76307

open Real

-- Define the statements p and q
def p : Prop := ∀ x, (deriv (λ x => 3 * x^2 + Real.log 3)) x = 6 * x + 3

def q : Prop := ∀ x, x ∈ Set.Ioo (-3 : ℝ) 1 ↔ 
  (deriv (λ x => (3 - x^2) * Real.exp x)) x > 0

-- Theorem statement
theorem p_or_q_is_true : p ∨ q := by sorry

end NUMINAMATH_CALUDE_p_or_q_is_true_l763_76307


namespace NUMINAMATH_CALUDE_product_sum_zero_l763_76367

theorem product_sum_zero (a b c d : ℚ) : 
  (∀ x, (2*x^2 - 3*x + 5)*(9 - 3*x) = a*x^3 + b*x^2 + c*x + d) → 
  27*a + 9*b + 3*c + d = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_zero_l763_76367


namespace NUMINAMATH_CALUDE_simple_interest_problem_l763_76321

/-- Proves that for a principal of 800 at simple interest, if increasing the
    interest rate by 5% results in 400 more interest, then the time period is 10 years. -/
theorem simple_interest_problem (r : ℝ) (t : ℝ) :
  (800 * r * t / 100) + 400 = 800 * (r + 5) * t / 100 →
  t = 10 :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l763_76321


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l763_76330

theorem algebraic_expression_value (x y : ℝ) :
  x^4 + 6*x^2*y + 9*y^2 + 2*x^2 + 6*y + 4 = 7 →
  (x^4 + 6*x^2*y + 9*y^2 - 2*x^2 - 6*y - 1 = -2) ∨
  (x^4 + 6*x^2*y + 9*y^2 - 2*x^2 - 6*y - 1 = 14) := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l763_76330


namespace NUMINAMATH_CALUDE_road_repaving_l763_76335

/-- Proves that the number of inches repaved before today is 4133,
    given the total repaved and the amount repaved today. -/
theorem road_repaving (total_repaved : ℕ) (repaved_today : ℕ)
    (h1 : total_repaved = 4938)
    (h2 : repaved_today = 805) :
    total_repaved - repaved_today = 4133 := by
  sorry

end NUMINAMATH_CALUDE_road_repaving_l763_76335


namespace NUMINAMATH_CALUDE_parabola_hyperbola_tangency_l763_76399

theorem parabola_hyperbola_tangency (n : ℝ) : 
  (∃ x y : ℝ, y = x^2 + 6 ∧ y^2 - n*x^2 = 4 ∧ 
    ∀ x' y' : ℝ, y' = x'^2 + 6 → y'^2 - n*x'^2 = 4 → (x', y') = (x, y)) →
  (n = 12 + 4*Real.sqrt 7 ∨ n = 12 - 4*Real.sqrt 7) :=
sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_tangency_l763_76399


namespace NUMINAMATH_CALUDE_farm_animal_count_l763_76370

/-- Represents the distribution of animals on a farm --/
structure FarmDistribution where
  chicken_coops : List Nat
  duck_coops : List Nat
  geese_coop : Nat
  quail_coop : Nat
  turkey_coops : List Nat
  cow_sheds : List Nat
  pig_sections : List Nat

/-- Calculates the total number of animals on the farm --/
def total_animals (farm : FarmDistribution) : Nat :=
  (farm.chicken_coops.sum + farm.duck_coops.sum + farm.geese_coop + 
   farm.quail_coop + farm.turkey_coops.sum + farm.cow_sheds.sum + 
   farm.pig_sections.sum)

/-- Theorem stating that the total number of animals on the farm is 431 --/
theorem farm_animal_count (farm : FarmDistribution) 
  (h1 : farm.chicken_coops = [60, 45, 55])
  (h2 : farm.duck_coops = [40, 35])
  (h3 : farm.geese_coop = 20)
  (h4 : farm.quail_coop = 50)
  (h5 : farm.turkey_coops = [10, 10])
  (h6 : farm.cow_sheds = [20, 10, 6])
  (h7 : farm.pig_sections = [15, 25, 30, 0]) :
  total_animals farm = 431 := by
  sorry

#eval total_animals {
  chicken_coops := [60, 45, 55],
  duck_coops := [40, 35],
  geese_coop := 20,
  quail_coop := 50,
  turkey_coops := [10, 10],
  cow_sheds := [20, 10, 6],
  pig_sections := [15, 25, 30, 0]
}

end NUMINAMATH_CALUDE_farm_animal_count_l763_76370


namespace NUMINAMATH_CALUDE_ellipse_equation_l763_76376

/-- The standard equation of an ellipse with given foci and major axis length -/
theorem ellipse_equation (a b c : ℝ) (h1 : c^2 = 5) (h2 : a = 5) (h3 : b^2 = a^2 - c^2) :
  ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1 ↔ (x^2 / 25) + (y^2 / 20) = 1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l763_76376


namespace NUMINAMATH_CALUDE_simplify_expression_l763_76350

theorem simplify_expression (a b c : ℝ) (h : (c - a) / (c - b) = 1) :
  (5 * b - 2 * a) / (c - a) = 3 * a / (c - a) :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l763_76350


namespace NUMINAMATH_CALUDE_parallelogram_network_l763_76314

theorem parallelogram_network (first_set : ℕ) (total_parallelograms : ℕ) 
  (h1 : first_set = 8) 
  (h2 : total_parallelograms = 784) : 
  ∃ (second_set : ℕ), 
    second_set > 0 ∧ 
    (first_set - 1) * (second_set - 1) = total_parallelograms := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_network_l763_76314


namespace NUMINAMATH_CALUDE_special_numbers_l763_76380

def last_digit (n : ℕ) : ℕ := n % 10

theorem special_numbers : 
  {n : ℕ | (last_digit n) * 2016 = n} = {4032, 8064, 12096, 16128} :=
by sorry

end NUMINAMATH_CALUDE_special_numbers_l763_76380


namespace NUMINAMATH_CALUDE_platform_length_calculation_l763_76300

/-- Calculates the length of a platform given train parameters -/
theorem platform_length_calculation (train_length : ℝ) (time_platform : ℝ) (time_pole : ℝ) :
  train_length = 300 →
  time_platform = 39 →
  time_pole = 18 →
  ∃ platform_length : ℝ,
    (platform_length > 348) ∧ (platform_length < 349) ∧
    (train_length + platform_length) / time_platform = train_length / time_pole :=
by
  sorry

#check platform_length_calculation

end NUMINAMATH_CALUDE_platform_length_calculation_l763_76300


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_rectangle_l763_76341

/-- Eccentricity of an ellipse with foci at opposite corners of a 4x3 rectangle 
    and passing through the other two corners -/
theorem ellipse_eccentricity_rectangle (a b c : ℝ) : 
  a = 4 →
  b = 3 →
  c = 2 →
  b^2 = 3*a →
  a^2 - c^2 = b^2 →
  c/a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_rectangle_l763_76341


namespace NUMINAMATH_CALUDE_zeros_of_f_l763_76352

def f (x : ℝ) : ℝ := x^2 - x - 2

theorem zeros_of_f :
  {x : ℝ | f x = 0} = {-1, 2} := by sorry

end NUMINAMATH_CALUDE_zeros_of_f_l763_76352


namespace NUMINAMATH_CALUDE_rectangle_diagonals_not_always_perpendicular_and_equal_l763_76373

/-- A rectangle is a quadrilateral with four right angles -/
structure Rectangle where
  sides : Fin 4 → ℝ
  angles : Fin 4 → ℝ
  is_right_angle : ∀ i, angles i = π / 2

/-- Diagonals of a shape -/
def diagonals (r : Rectangle) : Fin 2 → ℝ := sorry

/-- Two real numbers are equal -/
def are_equal (a b : ℝ) : Prop := a = b

/-- Two lines are perpendicular if they form a right angle -/
def are_perpendicular (a b : ℝ) : Prop := sorry

theorem rectangle_diagonals_not_always_perpendicular_and_equal (r : Rectangle) : 
  ¬(are_equal (diagonals r 0) (diagonals r 1) ∧ are_perpendicular (diagonals r 0) (diagonals r 1)) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonals_not_always_perpendicular_and_equal_l763_76373
