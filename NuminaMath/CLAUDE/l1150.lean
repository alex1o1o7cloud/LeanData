import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_sequence_2017th_term_l1150_115022

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem arithmetic_sequence_2017th_term 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_a1 : a 1 = 2)
  (h_geom : geometric_sequence (λ n => match n with
    | 1 => a 1 - 1
    | 2 => a 3
    | 3 => a 5 + 5
    | _ => 0
  )) :
  a 2017 = 1010 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2017th_term_l1150_115022


namespace NUMINAMATH_CALUDE_ricks_road_trip_l1150_115087

/-- Rick's road trip problem -/
theorem ricks_road_trip (D : ℝ) : 
  D > 0 ∧ 
  40 = D / 2 → 
  D + 2 * D + 40 + 2 * (D + 2 * D + 40) = 840 := by
  sorry

end NUMINAMATH_CALUDE_ricks_road_trip_l1150_115087


namespace NUMINAMATH_CALUDE_tangent_line_curve_l1150_115051

-- Define the line equation
def line (x y : ℝ) : Prop := x - y + 2 = 0

-- Define the curve equation
def curve (x y a : ℝ) : Prop := y = Real.log x + a

-- Define the tangency condition
def is_tangent (a : ℝ) : Prop :=
  ∃ x y : ℝ, line x y ∧ curve x y a ∧
    (∀ x' y' : ℝ, x' ≠ x → line x' y' → curve x' y' a → (y' - y) / (x' - x) ≠ 1 / x)

-- Theorem statement
theorem tangent_line_curve (a : ℝ) : is_tangent a → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_curve_l1150_115051


namespace NUMINAMATH_CALUDE_solution_set_y_geq_4_min_value_reciprocal_sum_l1150_115043

-- Define the quadratic function
def y (a b x : ℝ) : ℝ := a * x^2 + (b - 2) * x + 3

-- Part 1
theorem solution_set_y_geq_4 (a b : ℝ) :
  (∀ x : ℝ, y a b x > 0 ↔ -1 < x ∧ x < 3) →
  (∀ x : ℝ, y a b x ≥ 4 ↔ x = 1) :=
sorry

-- Part 2
theorem min_value_reciprocal_sum (a b : ℝ) :
  a > 0 →
  b > 0 →
  y a b 1 = 2 →
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → y a' b' 1 = 2 → 1/a' + 4/b' ≥ 1/a + 4/b) →
  1/a + 4/b = 9 :=
sorry

end NUMINAMATH_CALUDE_solution_set_y_geq_4_min_value_reciprocal_sum_l1150_115043


namespace NUMINAMATH_CALUDE_maria_towels_l1150_115065

theorem maria_towels (green_towels white_towels given_to_mother : ℕ) 
  (h1 : green_towels = 58)
  (h2 : white_towels = 43)
  (h3 : given_to_mother = 87) :
  green_towels + white_towels - given_to_mother = 14 :=
by sorry

end NUMINAMATH_CALUDE_maria_towels_l1150_115065


namespace NUMINAMATH_CALUDE_randy_store_trips_l1150_115073

/-- The number of trips Randy makes to the store each month -/
def trips_per_month (initial_amount : ℕ) (amount_per_trip : ℕ) (remaining_amount : ℕ) (months_per_year : ℕ) : ℕ :=
  ((initial_amount - remaining_amount) / amount_per_trip) / months_per_year

/-- Proof that Randy makes 4 trips to the store each month -/
theorem randy_store_trips :
  trips_per_month 200 2 104 12 = 4 := by
  sorry

end NUMINAMATH_CALUDE_randy_store_trips_l1150_115073


namespace NUMINAMATH_CALUDE_gabby_fruit_ratio_l1150_115075

/-- Represents the number of fruits Gabby harvested -/
structure FruitHarvest where
  watermelons : ℕ
  peaches : ℕ
  plums : ℕ

/-- Conditions of Gabby's fruit harvest -/
def gabbyHarvest : FruitHarvest where
  watermelons := 1
  peaches := 13
  plums := 39

theorem gabby_fruit_ratio :
  let h := gabbyHarvest
  h.watermelons = 1 ∧
  h.peaches = h.watermelons + 12 ∧
  h.watermelons + h.peaches + h.plums = 53 →
  h.plums / h.peaches = 3 := by
  sorry

end NUMINAMATH_CALUDE_gabby_fruit_ratio_l1150_115075


namespace NUMINAMATH_CALUDE_max_distance_PQ_l1150_115062

noncomputable section

-- Define the real parameters m and n
variables (m n : ℝ)

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := m * x - n * y - 5 * m + n = 0
def l₂ (x y : ℝ) : Prop := n * x + m * y - 5 * m - n = 0

-- Define the circle C
def C (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1

-- Define the intersection point P
def P (x y : ℝ) : Prop := l₁ m n x y ∧ l₂ m n x y

-- Define point Q on circle C
def Q (x y : ℝ) : Prop := C x y

-- State the theorem
theorem max_distance_PQ (hm : m^2 + n^2 ≠ 0) :
  ∃ (px py qx qy : ℝ), P m n px py ∧ Q qx qy ∧
  ∀ (px' py' qx' qy' : ℝ), P m n px' py' → Q qx' qy' →
  (px - qx)^2 + (py - qy)^2 ≤ (6 + 2 * Real.sqrt 2)^2 :=
sorry

end NUMINAMATH_CALUDE_max_distance_PQ_l1150_115062


namespace NUMINAMATH_CALUDE_plot_length_is_58_l1150_115079

/-- Represents a rectangular plot with given properties -/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  fencing_cost_per_meter : ℝ
  total_fencing_cost : ℝ
  length_breadth_difference : ℝ

/-- Calculates the length of the plot given the conditions -/
def calculate_plot_length (plot : RectangularPlot) : ℝ :=
  plot.breadth + plot.length_breadth_difference

/-- Theorem stating that the length of the plot is 58 meters under given conditions -/
theorem plot_length_is_58 (plot : RectangularPlot) 
  (h1 : plot.length = plot.breadth + 16)
  (h2 : plot.fencing_cost_per_meter = 26.5)
  (h3 : plot.total_fencing_cost = 5300)
  (h4 : plot.length_breadth_difference = 16) : 
  calculate_plot_length plot = 58 := by
  sorry

#eval calculate_plot_length { breadth := 42, length := 58, fencing_cost_per_meter := 26.5, total_fencing_cost := 5300, length_breadth_difference := 16 }

end NUMINAMATH_CALUDE_plot_length_is_58_l1150_115079


namespace NUMINAMATH_CALUDE_soccer_match_players_l1150_115015

theorem soccer_match_players (total_socks : ℕ) (socks_per_player : ℕ) : 
  total_socks = 16 → socks_per_player = 2 → total_socks / socks_per_player = 8 := by
  sorry

end NUMINAMATH_CALUDE_soccer_match_players_l1150_115015


namespace NUMINAMATH_CALUDE_picnic_age_problem_l1150_115074

theorem picnic_age_problem (initial_count : ℕ) (new_count : ℕ) (new_avg_age : ℝ) (final_avg_age : ℝ) :
  initial_count = 15 →
  new_count = 15 →
  new_avg_age = 15 →
  final_avg_age = 15.5 →
  ∃ (initial_avg_age : ℝ),
    initial_avg_age * initial_count + new_avg_age * new_count = 
    final_avg_age * (initial_count + new_count) ∧
    initial_avg_age = 16 := by
  sorry

end NUMINAMATH_CALUDE_picnic_age_problem_l1150_115074


namespace NUMINAMATH_CALUDE_fraction_equality_l1150_115038

theorem fraction_equality : ∀ x : ℝ, x ≠ 0 ∧ x^2 + 1 ≠ 0 →
  (x^2 + 5*x - 6) / (x^4 + x^2) = (-6 : ℝ) / x^2 + (0*x + 7) / (x^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1150_115038


namespace NUMINAMATH_CALUDE_solution_is_two_lines_l1150_115052

-- Define the equation
def equation (x y : ℝ) : Prop := (x + y)^2 = x^2 + y^2 + 4*x

-- Define the solution set
def solution_set : Set (ℝ × ℝ) :=
  {p | equation p.1 p.2}

-- Define the two lines
def y_axis : Set (ℝ × ℝ) := {p | p.1 = 0}
def horizontal_line : Set (ℝ × ℝ) := {p | p.2 = 2}

-- Theorem statement
theorem solution_is_two_lines :
  solution_set = y_axis ∪ horizontal_line :=
sorry

end NUMINAMATH_CALUDE_solution_is_two_lines_l1150_115052


namespace NUMINAMATH_CALUDE_vector_collinearity_l1150_115050

/-- Given vectors m, n, and k in ℝ², prove that if m - 2n is collinear with k, then t = 1 -/
theorem vector_collinearity (m n k : ℝ × ℝ) (t : ℝ) 
  (hm : m = (Real.sqrt 3, 1)) 
  (hn : n = (0, -1)) 
  (hk : k = (t, Real.sqrt 3)) 
  (hcol : ∃ (c : ℝ), c • (m - 2 • n) = k) : 
  t = 1 := by
  sorry

end NUMINAMATH_CALUDE_vector_collinearity_l1150_115050


namespace NUMINAMATH_CALUDE_emily_numbers_l1150_115034

theorem emily_numbers (n : ℕ) : 
  (n % 5 = 0 ∧ n % 10 = 0) → 
  (∃ d : ℕ, d < 10 ∧ d ≠ 0 ∧ n / 10 % 10 = d) →
  (∃ count : ℕ, count = 9 ∧ 
    ∀ d : ℕ, d < 10 ∧ d ≠ 0 → 
    ∃ m : ℕ, m % 5 = 0 ∧ m % 10 = 0 ∧ m / 10 % 10 = d) :=
by
  sorry

#check emily_numbers

end NUMINAMATH_CALUDE_emily_numbers_l1150_115034


namespace NUMINAMATH_CALUDE_chocolate_candy_price_difference_l1150_115096

/-- Proves the difference in cost between a discounted chocolate and a taxed candy bar --/
theorem chocolate_candy_price_difference 
  (initial_money : ℝ)
  (chocolate_price gum_price candy_price soda_price : ℝ)
  (chocolate_discount gum_candy_tax : ℝ) :
  initial_money = 20 →
  chocolate_price = 7 →
  gum_price = 3 →
  candy_price = 2 →
  soda_price = 1.5 →
  chocolate_discount = 0.15 →
  gum_candy_tax = 0.08 →
  chocolate_price * (1 - chocolate_discount) - (candy_price * (1 + gum_candy_tax)) = 3.95 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_candy_price_difference_l1150_115096


namespace NUMINAMATH_CALUDE_distance_along_stream_is_16_l1150_115061

-- Define the boat's speed in still water
def boat_speed : ℝ := 11

-- Define the distance traveled against the stream in one hour
def distance_against_stream : ℝ := 6

-- Define the stream speed
def stream_speed : ℝ := boat_speed - distance_against_stream

-- Define the boat's speed along the stream
def speed_along_stream : ℝ := boat_speed + stream_speed

-- Theorem to prove
theorem distance_along_stream_is_16 : speed_along_stream = 16 := by
  sorry

end NUMINAMATH_CALUDE_distance_along_stream_is_16_l1150_115061


namespace NUMINAMATH_CALUDE_linda_coin_fraction_l1150_115027

/-- The fraction of Linda's coins representing states that joined the union during 1790-1799 -/
def fraction_of_coins (total_coins : ℕ) (states_joined : ℕ) : ℚ :=
  states_joined / total_coins

/-- Proof that the fraction of Linda's coins representing states from 1790-1799 is 4/15 -/
theorem linda_coin_fraction :
  fraction_of_coins 30 8 = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_linda_coin_fraction_l1150_115027


namespace NUMINAMATH_CALUDE_bracket_six_times_bracket_three_l1150_115092

-- Define a function for the square bracket operation
def bracket (x : ℕ) : ℕ :=
  if x % 2 = 0 then
    x / 2 + 1
  else
    2 * x + 1

-- Theorem statement
theorem bracket_six_times_bracket_three : bracket 6 * bracket 3 = 28 := by
  sorry

end NUMINAMATH_CALUDE_bracket_six_times_bracket_three_l1150_115092


namespace NUMINAMATH_CALUDE_coffee_table_books_l1150_115016

/-- Represents the number of books Henry has in different locations and actions he takes. -/
structure HenryBooks where
  total : ℕ
  boxed : ℕ
  boxCount : ℕ
  roomDonate : ℕ
  kitchenDonate : ℕ
  newPickup : ℕ
  finalCount : ℕ

/-- Calculates the number of books on Henry's coffee table. -/
def booksOnCoffeeTable (h : HenryBooks) : ℕ :=
  h.total - (h.boxed * h.boxCount + h.roomDonate + h.kitchenDonate) - (h.finalCount - h.newPickup)

/-- Theorem stating that the number of books on Henry's coffee table is 4. -/
theorem coffee_table_books :
  let h : HenryBooks := {
    total := 99,
    boxed := 15,
    boxCount := 3,
    roomDonate := 21,
    kitchenDonate := 18,
    newPickup := 12,
    finalCount := 23
  }
  booksOnCoffeeTable h = 4 := by
  sorry

end NUMINAMATH_CALUDE_coffee_table_books_l1150_115016


namespace NUMINAMATH_CALUDE_division_preserves_inequality_l1150_115026

theorem division_preserves_inequality (a b : ℝ) (h : a > b) : a / 3 > b / 3 := by
  sorry

end NUMINAMATH_CALUDE_division_preserves_inequality_l1150_115026


namespace NUMINAMATH_CALUDE_store_prices_l1150_115080

def price_X : ℝ := 80 * (1 + 0.12)
def price_Y : ℝ := price_X * (1 - 0.15)
def price_Z : ℝ := price_Y * (1 + 0.25)

theorem store_prices :
  price_X = 89.6 ∧ price_Y = 76.16 ∧ price_Z = 95.20 := by
  sorry

end NUMINAMATH_CALUDE_store_prices_l1150_115080


namespace NUMINAMATH_CALUDE_father_son_age_sum_l1150_115078

/-- Given the ages of a father and son 25 years ago and their current age ratio,
    prove that the sum of their present ages is 300 years. -/
theorem father_son_age_sum : ℕ → ℕ → Prop :=
  fun (s f : ℕ) =>
    (f = 4 * s) →                  -- 25 years ago, father was 4 times as old as son
    (f + 25 = 3 * (s + 25)) →      -- Now, father is 3 times as old as son
    ((s + 25) + (f + 25) = 300)    -- Sum of their present ages is 300

/-- Proof of the theorem -/
lemma prove_father_son_age_sum : ∃ (s f : ℕ), father_son_age_sum s f := by
  sorry

#check prove_father_son_age_sum

end NUMINAMATH_CALUDE_father_son_age_sum_l1150_115078


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l1150_115090

/-- A quadratic equation x^2 + 5x + k = 0 has distinct real roots if and only if k < 25/4 -/
theorem quadratic_distinct_roots (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 5*x + k = 0 ∧ y^2 + 5*y + k = 0) ↔ k < 25/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l1150_115090


namespace NUMINAMATH_CALUDE_account_balance_first_year_l1150_115072

/-- Proves that given an initial deposit and interest accrued, the account balance
    at the end of the first year is the sum of the initial deposit and interest accrued. -/
theorem account_balance_first_year
  (initial_deposit : ℝ)
  (interest_accrued : ℝ)
  (h1 : initial_deposit = 1000)
  (h2 : interest_accrued = 100) :
  initial_deposit + interest_accrued = 1100 := by
  sorry

end NUMINAMATH_CALUDE_account_balance_first_year_l1150_115072


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1150_115035

theorem contrapositive_equivalence (a b : ℝ) :
  (¬(a = 0 ∧ b = 0) → a^2 + b^2 ≠ 0) ↔ (a^2 + b^2 = 0 → a = 0 ∧ b = 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l1150_115035


namespace NUMINAMATH_CALUDE_square_side_lengths_l1150_115012

theorem square_side_lengths (a b : ℝ) (h1 : a > b) (h2 : a - b = 2) (h3 : a^2 - b^2 = 40) : 
  a = 11 ∧ b = 9 := by
sorry

end NUMINAMATH_CALUDE_square_side_lengths_l1150_115012


namespace NUMINAMATH_CALUDE_cosine_function_properties_l1150_115082

theorem cosine_function_properties (a b c d : ℝ) (ha : a > 0) :
  (∀ x, ∃ y, y = a * Real.cos (b * x + c) + d) →
  (a = 4) →
  (2 * Real.pi / b = Real.pi / 2) →
  (b = 4 ∧ ∀ c₁ c₂, ∃ b', 
    (∀ x, ∃ y, y = a * Real.cos (b' * x + c₁) + d) ∧
    (∀ x, ∃ y, y = a * Real.cos (b' * x + c₂) + d) ∧
    (2 * Real.pi / b' = Real.pi / 2)) :=
by sorry

end NUMINAMATH_CALUDE_cosine_function_properties_l1150_115082


namespace NUMINAMATH_CALUDE_basketball_game_score_theorem_l1150_115008

/-- Represents a team's scores for each quarter -/
structure TeamScores :=
  (q1 : ℕ) (q2 : ℕ) (q3 : ℕ) (q4 : ℕ)

/-- Checks if a sequence of four numbers is an arithmetic sequence -/
def isArithmeticSequence (s : TeamScores) : Prop :=
  s.q2 - s.q1 = s.q3 - s.q2 ∧ s.q3 - s.q2 = s.q4 - s.q3 ∧ s.q2 > s.q1

/-- Checks if a sequence of four numbers is a geometric sequence -/
def isGeometricSequence (s : TeamScores) : Prop :=
  s.q2 / s.q1 = s.q3 / s.q2 ∧ s.q3 / s.q2 = s.q4 / s.q3 ∧ s.q2 > s.q1

/-- Calculates the total score for a team -/
def totalScore (s : TeamScores) : ℕ := s.q1 + s.q2 + s.q3 + s.q4

theorem basketball_game_score_theorem 
  (eagles lions : TeamScores) : 
  isArithmeticSequence eagles →
  isGeometricSequence lions →
  eagles.q1 = lions.q1 + 2 →
  eagles.q1 + eagles.q2 + eagles.q3 = lions.q1 + lions.q2 + lions.q3 →
  totalScore eagles ≤ 100 →
  totalScore lions ≤ 100 →
  totalScore eagles + totalScore lions = 144 := by
  sorry

#check basketball_game_score_theorem

end NUMINAMATH_CALUDE_basketball_game_score_theorem_l1150_115008


namespace NUMINAMATH_CALUDE_max_area_rectangle_l1150_115046

/-- The maximum area of a rectangle with perimeter P is P²/16 -/
theorem max_area_rectangle (P : ℝ) (h : P > 0) : 
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 2*x + 2*y = P ∧ 
  x*y = P^2/16 ∧ 
  ∀ (a b : ℝ), a > 0 → b > 0 → 2*a + 2*b = P → a*b ≤ P^2/16 := by
sorry

end NUMINAMATH_CALUDE_max_area_rectangle_l1150_115046


namespace NUMINAMATH_CALUDE_ice_cream_stacking_permutations_l1150_115039

theorem ice_cream_stacking_permutations : Nat.factorial 5 = 120 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_stacking_permutations_l1150_115039


namespace NUMINAMATH_CALUDE_marys_hourly_rate_l1150_115044

/-- Represents Mary's work schedule and pay structure --/
structure WorkSchedule where
  maxHours : ℕ
  regularHours : ℕ
  overtimeRate : ℚ
  maxEarnings : ℚ

/-- Calculates the regular hourly rate given a work schedule --/
def regularHourlyRate (schedule : WorkSchedule) : ℚ :=
  let overtimeHours := schedule.maxHours - schedule.regularHours
  let x := schedule.maxEarnings / (schedule.regularHours + overtimeHours * schedule.overtimeRate)
  x

/-- Theorem stating that Mary's regular hourly rate is $8 --/
theorem marys_hourly_rate :
  let schedule := WorkSchedule.mk 80 20 1.25 760
  regularHourlyRate schedule = 8 := by
  sorry

end NUMINAMATH_CALUDE_marys_hourly_rate_l1150_115044


namespace NUMINAMATH_CALUDE_salary_increase_percentage_l1150_115056

/-- Proves that given an initial monthly salary of $6000 and total earnings of $259200 after 3 years,
    with a salary increase occurring after 1 year, the percentage increase in salary is 30%. -/
theorem salary_increase_percentage 
  (initial_salary : ℝ) 
  (total_earnings : ℝ) 
  (increase_percentage : ℝ) :
  initial_salary = 6000 →
  total_earnings = 259200 →
  total_earnings = 12 * initial_salary + 24 * (initial_salary + initial_salary * increase_percentage / 100) →
  increase_percentage = 30 := by
  sorry

#check salary_increase_percentage

end NUMINAMATH_CALUDE_salary_increase_percentage_l1150_115056


namespace NUMINAMATH_CALUDE_triangle_angle_sum_rounded_l1150_115058

-- Define a structure for a triangle with actual and rounded angles
structure Triangle where
  P' : ℝ
  Q' : ℝ
  R' : ℝ
  P : ℤ
  Q : ℤ
  R : ℤ

-- Define the properties of a valid triangle
def is_valid_triangle (t : Triangle) : Prop :=
  t.P' + t.Q' + t.R' = 180 ∧
  t.P' > 0 ∧ t.Q' > 0 ∧ t.R' > 0 ∧
  (t.P' - 0.5 ≤ t.P ∧ t.P ≤ t.P' + 0.5) ∧
  (t.Q' - 0.5 ≤ t.Q ∧ t.Q ≤ t.Q' + 0.5) ∧
  (t.R' - 0.5 ≤ t.R ∧ t.R ≤ t.R' + 0.5)

-- Theorem statement
theorem triangle_angle_sum_rounded (t : Triangle) :
  is_valid_triangle t → (t.P + t.Q + t.R = 179 ∨ t.P + t.Q + t.R = 180 ∨ t.P + t.Q + t.R = 181) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_rounded_l1150_115058


namespace NUMINAMATH_CALUDE_technician_average_salary_l1150_115010

/-- Calculates the average salary of technicians in a workshop --/
theorem technician_average_salary
  (total_workers : ℕ)
  (total_average : ℝ)
  (non_tech_average : ℝ)
  (num_technicians : ℕ)
  (h1 : total_workers = 14)
  (h2 : total_average = 10000)
  (h3 : non_tech_average = 8000)
  (h4 : num_technicians = 7) :
  (total_workers * total_average - (total_workers - num_technicians) * non_tech_average) / num_technicians = 12000 := by
sorry

end NUMINAMATH_CALUDE_technician_average_salary_l1150_115010


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1150_115047

theorem quadratic_equation_solution (b : ℝ) : 
  (2 * (-5)^2 + b * (-5) - 20 = 0) → b = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1150_115047


namespace NUMINAMATH_CALUDE_equal_coin_count_l1150_115068

/-- Represents the value of each coin type in cents -/
def coin_value : Fin 5 → ℕ
  | 0 => 1    -- penny
  | 1 => 5    -- nickel
  | 2 => 10   -- dime
  | 3 => 25   -- quarter
  | 4 => 50   -- half dollar

/-- The theorem statement -/
theorem equal_coin_count (x : ℕ) (h : x * (coin_value 0 + coin_value 1 + coin_value 2 + coin_value 3 + coin_value 4) = 273) :
  5 * x = 15 := by
  sorry

#check equal_coin_count

end NUMINAMATH_CALUDE_equal_coin_count_l1150_115068


namespace NUMINAMATH_CALUDE_derivative_at_pi_third_l1150_115028

theorem derivative_at_pi_third (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, f x = x^2 * (deriv f (π/3)) + Real.sin x) : 
  deriv f (π/3) = 3 / (6 - 4*π) := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_pi_third_l1150_115028


namespace NUMINAMATH_CALUDE_product_equals_zero_l1150_115024

theorem product_equals_zero (n : ℤ) (h : n = 1) : (n - 2) * (n - 1) * n * (n + 1) * (n + 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_zero_l1150_115024


namespace NUMINAMATH_CALUDE_officers_count_l1150_115005

/-- The number of ways to choose 4 distinct officers from a group of 15 people -/
def choose_officers (n : ℕ) : ℕ :=
  n * (n - 1) * (n - 2) * (n - 3)

/-- Theorem: There are 32,760 ways to choose 4 distinct officers from a group of 15 people -/
theorem officers_count : choose_officers 15 = 32760 := by
  sorry

end NUMINAMATH_CALUDE_officers_count_l1150_115005


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l1150_115042

/-- Atomic weight of Calcium -/
def Ca_weight : ℝ := 40.08

/-- Atomic weight of Oxygen -/
def O_weight : ℝ := 16.00

/-- Atomic weight of Hydrogen -/
def H_weight : ℝ := 1.008

/-- Atomic weight of Nitrogen -/
def N_weight : ℝ := 14.01

/-- Atomic weight of Carbon-12 -/
def C12_weight : ℝ := 12.00

/-- Atomic weight of Carbon-13 -/
def C13_weight : ℝ := 13.003

/-- Percentage of Carbon-12 in the compound -/
def C12_percentage : ℝ := 0.95

/-- Percentage of Carbon-13 in the compound -/
def C13_percentage : ℝ := 0.05

/-- Average atomic weight of Carbon in the compound -/
def C_avg_weight : ℝ := C12_percentage * C12_weight + C13_percentage * C13_weight

/-- Number of Calcium atoms in the compound -/
def Ca_count : ℕ := 2

/-- Number of Oxygen atoms in the compound -/
def O_count : ℕ := 3

/-- Number of Hydrogen atoms in the compound -/
def H_count : ℕ := 2

/-- Number of Nitrogen atoms in the compound -/
def N_count : ℕ := 1

/-- Number of Carbon atoms in the compound -/
def C_count : ℕ := 1

/-- Molecular weight of the compound -/
def molecular_weight : ℝ :=
  Ca_count * Ca_weight + O_count * O_weight + H_count * H_weight +
  N_count * N_weight + C_count * C_avg_weight

theorem compound_molecular_weight :
  molecular_weight = 156.22615 := by sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l1150_115042


namespace NUMINAMATH_CALUDE_all_statements_imply_target_l1150_115097

theorem all_statements_imply_target (p q r : Prop) :
  ((¬p ∧ ¬r ∧ q) → ((p ∧ q) → ¬r)) ∧
  ((p ∧ ¬r ∧ ¬q) → ((p ∧ q) → ¬r)) ∧
  ((¬p ∧ r ∧ q) → ((p ∧ q) → ¬r)) ∧
  ((p ∧ r ∧ ¬q) → ((p ∧ q) → ¬r)) :=
by sorry

end NUMINAMATH_CALUDE_all_statements_imply_target_l1150_115097


namespace NUMINAMATH_CALUDE_chord_length_is_sqrt_6_l1150_115036

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x + y^2 + 4*x - 4*y + 6 = 0

-- Define the line l
def line_l (k x y : ℝ) : Prop := k*x + y + 4 = 0

-- Define the line m
def line_m (k x y : ℝ) : Prop := y = x + k

-- Theorem statement
theorem chord_length_is_sqrt_6 (k : ℝ) :
  (∃ x y : ℝ, line_l k x y ∧ circle_C x y) →  -- l is a symmetric axis of C
  (∃ x y : ℝ, line_m k x y ∧ circle_C x y) →  -- m intersects C
  (∃ x1 y1 x2 y2 : ℝ, 
    line_m k x1 y1 ∧ circle_C x1 y1 ∧
    line_m k x2 y2 ∧ circle_C x2 y2 ∧
    (x1 - x2)^2 + (y1 - y2)^2 = 6) :=
by sorry

end NUMINAMATH_CALUDE_chord_length_is_sqrt_6_l1150_115036


namespace NUMINAMATH_CALUDE_ceiling_painting_fraction_l1150_115095

def total_ceilings : ℕ := 28
def first_week_ceilings : ℕ := 12
def remaining_ceilings : ℕ := 13

theorem ceiling_painting_fraction :
  (total_ceilings - first_week_ceilings - remaining_ceilings) / first_week_ceilings = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_painting_fraction_l1150_115095


namespace NUMINAMATH_CALUDE_negative_marks_for_wrong_answer_l1150_115041

def total_questions : ℕ := 150
def correct_answers : ℕ := 120
def total_score : ℕ := 420
def correct_score : ℕ := 4

theorem negative_marks_for_wrong_answer :
  ∃ (x : ℚ), 
    (correct_score * correct_answers : ℚ) - 
    (x * (total_questions - correct_answers)) = total_score ∧
    x = 2 := by sorry

end NUMINAMATH_CALUDE_negative_marks_for_wrong_answer_l1150_115041


namespace NUMINAMATH_CALUDE_g_satisfies_equation_l1150_115021

-- Define the polynomial g(x)
def g (x : ℝ) : ℝ := -4 * x^5 + 7 * x^3 - 5 * x^2 - x + 6

-- State the theorem
theorem g_satisfies_equation : ∀ x : ℝ, 4 * x^5 - 3 * x^3 + x + g x = 7 * x^3 - 5 * x^2 + 6 := by
  sorry

end NUMINAMATH_CALUDE_g_satisfies_equation_l1150_115021


namespace NUMINAMATH_CALUDE_expression_evaluation_1_expression_evaluation_2_l1150_115037

theorem expression_evaluation_1 : (1 * (-4.5) - (-5 - (2/3)) - 2.5 - (7 + (2/3))) = -9 := by sorry

theorem expression_evaluation_2 : (-4^2 / (-2)^3 - (4/9) * (-3/2)^2) = 1 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_1_expression_evaluation_2_l1150_115037


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1150_115059

theorem trigonometric_identity (α : Real) 
  (h1 : Real.tan (α + π/4) = 1/2) 
  (h2 : -π/2 < α) 
  (h3 : α < 0) : 
  Real.sin (2*α) + 2 * (Real.sin α)^2 = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1150_115059


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1150_115000

theorem max_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a * b * c + a + c = b) :
  ∃ (p : ℝ), p = 2 / (a^2 + 1) - 2 / (b^2 + 1) + 3 / (c^2 + 1) ∧
  p ≤ 10/3 ∧ 
  ∃ (a' b' c' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ 
    a' * b' * c' + a' + c' = b' ∧
    2 / (a'^2 + 1) - 2 / (b'^2 + 1) + 3 / (c'^2 + 1) = 10/3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1150_115000


namespace NUMINAMATH_CALUDE_yellow_ball_count_l1150_115099

theorem yellow_ball_count (total : ℕ) (red : ℕ) (yellow : ℕ) (prob_red : ℚ) : 
  red = 9 →
  yellow + red = total →
  prob_red = 1/3 →
  prob_red = red / total →
  yellow = 18 :=
sorry

end NUMINAMATH_CALUDE_yellow_ball_count_l1150_115099


namespace NUMINAMATH_CALUDE_sin_shift_equivalence_l1150_115076

open Real

theorem sin_shift_equivalence (x : ℝ) : 
  sin (2 * x + π / 6) = sin (2 * (x + π / 4) - π / 3) := by sorry

end NUMINAMATH_CALUDE_sin_shift_equivalence_l1150_115076


namespace NUMINAMATH_CALUDE_inequality_range_l1150_115057

theorem inequality_range (b : ℝ) : 
  (b > 0 ∧ ∃ y : ℝ, |y - 5| + 2 * |y - 2| > b) → 0 < b ∧ b < 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l1150_115057


namespace NUMINAMATH_CALUDE_f_2023_equals_107_l1150_115031

-- Define the property of the function f
def has_property (f : ℕ → ℝ) : Prop :=
  ∀ (a b n : ℕ), a > 0 → b > 0 → n > 0 → a + b = 2^n → f a + f b = (n^2 + 1 : ℝ)

-- Theorem statement
theorem f_2023_equals_107 (f : ℕ → ℝ) (h : has_property f) : f 2023 = 107 := by
  sorry

end NUMINAMATH_CALUDE_f_2023_equals_107_l1150_115031


namespace NUMINAMATH_CALUDE_quadratic_zero_discriminant_geometric_progression_l1150_115025

/-- 
Given a quadratic equation ax^2 + 6bx + 9c = 0 with zero discriminant,
prove that a, b, and c form a geometric progression.
-/
theorem quadratic_zero_discriminant_geometric_progression 
  (a b c : ℝ) 
  (h_quad : ∀ x, a * x^2 + 6 * b * x + 9 * c = 0)
  (h_discr : (6 * b)^2 - 4 * a * (9 * c) = 0) :
  ∃ r : ℝ, b = a * r ∧ c = b * r :=
sorry

end NUMINAMATH_CALUDE_quadratic_zero_discriminant_geometric_progression_l1150_115025


namespace NUMINAMATH_CALUDE_relationship_abc_l1150_115002

theorem relationship_abc : 
  let a : ℝ := 2^(1/2)
  let b : ℝ := 3^(1/3)
  let c : ℝ := Real.log 2
  b > a ∧ a > c := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l1150_115002


namespace NUMINAMATH_CALUDE_descending_order_proof_l1150_115048

def original_numbers : List ℝ := [1.64, 2.1, 0.09, 1.2]
def sorted_numbers : List ℝ := [2.1, 1.64, 1.2, 0.09]

theorem descending_order_proof :
  (sorted_numbers.zip (sorted_numbers.tail!)).all (fun (a, b) => a ≥ b) ∧
  sorted_numbers.toFinset = original_numbers.toFinset :=
by sorry

end NUMINAMATH_CALUDE_descending_order_proof_l1150_115048


namespace NUMINAMATH_CALUDE_factorial_inequality_l1150_115014

/-- A function satisfying the given property -/
def special_function (f : ℕ → ℕ) : Prop :=
  ∀ w x y z : ℕ, f (f (f z)) * f (w * x * f (y * f z)) = z^2 * f (x * f y) * f w

/-- The main theorem -/
theorem factorial_inequality (f : ℕ → ℕ) (h : special_function f) : 
  ∀ n : ℕ, f (n.factorial) ≥ n.factorial :=
sorry

end NUMINAMATH_CALUDE_factorial_inequality_l1150_115014


namespace NUMINAMATH_CALUDE_roger_shelves_theorem_l1150_115033

def shelves_needed (total_books : ℕ) (books_taken : ℕ) (books_per_shelf : ℕ) : ℕ :=
  let remaining_books := total_books - books_taken
  (remaining_books + books_per_shelf - 1) / books_per_shelf

theorem roger_shelves_theorem :
  shelves_needed 24 3 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_roger_shelves_theorem_l1150_115033


namespace NUMINAMATH_CALUDE_jigi_score_l1150_115054

theorem jigi_score (max_score : ℕ) (gibi_percent mike_percent lizzy_percent : ℚ) 
  (average_mark : ℕ) (h1 : max_score = 700) (h2 : gibi_percent = 59/100) 
  (h3 : mike_percent = 99/100) (h4 : lizzy_percent = 67/100) (h5 : average_mark = 490) : 
  (4 * average_mark - (gibi_percent + mike_percent + lizzy_percent) * max_score) / max_score = 55/100 :=
sorry

end NUMINAMATH_CALUDE_jigi_score_l1150_115054


namespace NUMINAMATH_CALUDE_intersection_is_empty_l1150_115084

-- Define the sets A and B
def A : Set String := {s | s = "line"}
def B : Set String := {s | s = "ellipse"}

-- Theorem statement
theorem intersection_is_empty : A ∩ B = ∅ := by
  sorry

end NUMINAMATH_CALUDE_intersection_is_empty_l1150_115084


namespace NUMINAMATH_CALUDE_grocery_store_inventory_l1150_115070

theorem grocery_store_inventory (regular_soda diet_soda apples : ℕ) 
  (h1 : regular_soda = 72)
  (h2 : diet_soda = 32)
  (h3 : apples = 78) :
  regular_soda + diet_soda - apples = 26 := by
  sorry

end NUMINAMATH_CALUDE_grocery_store_inventory_l1150_115070


namespace NUMINAMATH_CALUDE_managers_salary_l1150_115004

theorem managers_salary (num_employees : ℕ) (avg_salary : ℚ) (salary_increase : ℚ) : 
  num_employees = 18 →
  avg_salary = 2000 →
  salary_increase = 200 →
  (num_employees * avg_salary + (avg_salary + salary_increase) * (num_employees + 1) - num_employees * avg_salary) = 5800 := by
sorry

end NUMINAMATH_CALUDE_managers_salary_l1150_115004


namespace NUMINAMATH_CALUDE_point_on_line_l1150_115023

/-- Given three points M, N, and P in the 2D plane, where P lies on the line passing through M and N,
    prove that the y-coordinate of P is 2. -/
theorem point_on_line (M N P : ℝ × ℝ) : 
  M = (2, -1) → N = (4, 5) → P.1 = 3 → 
  (P.2 - M.2) / (P.1 - M.1) = (N.2 - M.2) / (N.1 - M.1) → 
  P.2 = 2 := by
sorry

end NUMINAMATH_CALUDE_point_on_line_l1150_115023


namespace NUMINAMATH_CALUDE_seven_rows_of_ten_for_79_people_l1150_115020

/-- Represents a seating arrangement with rows of either 9 or 10 people -/
structure SeatingArrangement where
  rows_of_9 : ℕ
  rows_of_10 : ℕ

/-- The total number of people in a seating arrangement -/
def total_people (s : SeatingArrangement) : ℕ :=
  9 * s.rows_of_9 + 10 * s.rows_of_10

/-- Theorem stating that for 79 people, there are 7 rows of 10 people -/
theorem seven_rows_of_ten_for_79_people :
  ∃ (s : SeatingArrangement), total_people s = 79 ∧ s.rows_of_10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_seven_rows_of_ten_for_79_people_l1150_115020


namespace NUMINAMATH_CALUDE_not_all_points_follow_linear_relation_l1150_115063

-- Define the type for our data points
structure DataPoint where
  n : Nat
  w : Nat

-- Define our dataset
def dataset : List DataPoint := [
  { n := 1, w := 55 },
  { n := 2, w := 110 },
  { n := 3, w := 160 },
  { n := 4, w := 200 },
  { n := 5, w := 254 },
  { n := 6, w := 300 },
  { n := 7, w := 350 }
]

-- Theorem statement
theorem not_all_points_follow_linear_relation :
  ∃ point : DataPoint, point ∈ dataset ∧ point.w ≠ 55 * point.n := by
  sorry


end NUMINAMATH_CALUDE_not_all_points_follow_linear_relation_l1150_115063


namespace NUMINAMATH_CALUDE_power_sum_equality_l1150_115083

theorem power_sum_equality : (-1 : ℤ) ^ 53 + 3 ^ (2^3 + 5^2 - 7^2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l1150_115083


namespace NUMINAMATH_CALUDE_saree_discount_problem_l1150_115029

/-- Calculates the second discount percentage given the original price, first discount percentage, and final sale price. -/
def second_discount_percentage (original_price first_discount_percent final_price : ℚ) : ℚ :=
  let price_after_first_discount := original_price * (1 - first_discount_percent / 100)
  let second_discount_amount := price_after_first_discount - final_price
  (second_discount_amount / price_after_first_discount) * 100

/-- Theorem stating that for the given conditions, the second discount percentage is 15%. -/
theorem saree_discount_problem :
  second_discount_percentage 450 20 306 = 15 := by sorry

end NUMINAMATH_CALUDE_saree_discount_problem_l1150_115029


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1150_115003

def A : Set ℤ := {-1, 0, 3}
def B : Set ℤ := {-1, 1, 2, 3}

theorem union_of_A_and_B : A ∪ B = {-1, 0, 1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1150_115003


namespace NUMINAMATH_CALUDE_distance_home_to_school_l1150_115007

/-- The distance between home and school given the travel conditions --/
theorem distance_home_to_school :
  ∀ (D T : ℝ),
  (3 * (T + 7/60) = D) →
  (6 * (T - 8/60) = D) →
  D = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_distance_home_to_school_l1150_115007


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1150_115049

theorem partial_fraction_decomposition :
  ∃! (P Q R : ℚ),
    ∀ (x : ℚ), x ≠ 1 ∧ x ≠ 4 ∧ x ≠ 6 →
      (x^2 - 9) / ((x - 1) * (x - 4) * (x - 6)) =
      P / (x - 1) + Q / (x - 4) + R / (x - 6) ∧
      P = -8/15 ∧ Q = -7/6 ∧ R = 27/10 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1150_115049


namespace NUMINAMATH_CALUDE_rectangular_field_width_l1150_115053

theorem rectangular_field_width (width length : ℝ) (perimeter : ℝ) : 
  length = (7 / 5) * width →
  perimeter = 2 * length + 2 * width →
  perimeter = 240 →
  width = 50 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_width_l1150_115053


namespace NUMINAMATH_CALUDE_probability_two_shirts_one_shorts_one_socks_l1150_115081

def num_shirts : ℕ := 3
def num_shorts : ℕ := 7
def num_socks : ℕ := 4
def num_selected : ℕ := 4

def total_articles : ℕ := num_shirts + num_shorts + num_socks

def favorable_outcomes : ℕ := (num_shirts.choose 2) * (num_shorts.choose 1) * (num_socks.choose 1)
def total_outcomes : ℕ := total_articles.choose num_selected

theorem probability_two_shirts_one_shorts_one_socks :
  (favorable_outcomes : ℚ) / total_outcomes = 84 / 1001 :=
sorry

end NUMINAMATH_CALUDE_probability_two_shirts_one_shorts_one_socks_l1150_115081


namespace NUMINAMATH_CALUDE_pizza_piece_volume_l1150_115089

/-- The volume of a piece of pizza -/
theorem pizza_piece_volume (thickness : ℝ) (diameter : ℝ) (num_pieces : ℕ) :
  thickness = 1/3 →
  diameter = 18 →
  num_pieces = 18 →
  (π * (diameter/2)^2 * thickness) / num_pieces = 3*π/2 := by
  sorry

end NUMINAMATH_CALUDE_pizza_piece_volume_l1150_115089


namespace NUMINAMATH_CALUDE_x_minus_y_equals_one_l1150_115018

theorem x_minus_y_equals_one (x y : ℝ) 
  (h1 : x^2 + y^2 = 25) 
  (h2 : x + y = 7) 
  (h3 : x > y) : 
  x - y = 1 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_one_l1150_115018


namespace NUMINAMATH_CALUDE_f_properties_l1150_115006

/-- Definition of an odd function -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- Definition of the function f -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then a^x - 1 else 1 - a^(-x)

theorem f_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  OddFunction (f a) ∧ 
  (f a 2 + f a (-2) = 0) ∧
  (∀ x, f a x = if x ≥ 0 then a^x - 1 else 1 - a^(-x)) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l1150_115006


namespace NUMINAMATH_CALUDE_horner_method_v2_l1150_115067

def f (x : ℝ) : ℝ := 2*x^5 - x^4 + 2*x^2 + 5*x + 3

def horner_v2 (x v0 v1 : ℝ) : ℝ := v1 * x

theorem horner_method_v2 (x v0 v1 : ℝ) (hx : x = 3) (hv0 : v0 = 2) (hv1 : v1 = 5) :
  horner_v2 x v0 v1 = 15 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_v2_l1150_115067


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l1150_115013

/-- Given an arithmetic sequence with the first four terms as specified,
    prove that the fifth term is 123/40 -/
theorem arithmetic_sequence_fifth_term
  (x y : ℚ)
  (h1 : (x + y) - (x - y) = (x - y) - (x * y))
  (h2 : (x - y) - (x * y) = (x * y) - (x / y))
  (h3 : y ≠ 0)
  : (x / y) + ((x / y) - (x * y)) = 123/40 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l1150_115013


namespace NUMINAMATH_CALUDE_larger_segment_is_70_l1150_115064

/-- A triangle with sides 40, 50, and 90 units -/
structure Triangle where
  side_a : ℝ
  side_b : ℝ
  side_c : ℝ
  ha : side_a = 40
  hb : side_b = 50
  hc : side_c = 90

/-- The altitude dropped on the side of length 90 -/
def altitude (t : Triangle) : ℝ := sorry

/-- The larger segment cut off on the side of length 90 -/
def larger_segment (t : Triangle) : ℝ := sorry

/-- Theorem stating that the larger segment is 70 units -/
theorem larger_segment_is_70 (t : Triangle) : larger_segment t = 70 := by sorry

end NUMINAMATH_CALUDE_larger_segment_is_70_l1150_115064


namespace NUMINAMATH_CALUDE_chloe_min_score_l1150_115088

/-- The minimum score needed on the fifth test to achieve a given average -/
def min_score_for_average (test1 test2 test3 test4 : ℚ) (required_avg : ℚ) : ℚ :=
  5 * required_avg - (test1 + test2 + test3 + test4)

/-- Proof that Chloe needs at least 86% on her fifth test -/
theorem chloe_min_score :
  let test1 : ℚ := 84
  let test2 : ℚ := 87
  let test3 : ℚ := 78
  let test4 : ℚ := 90
  let required_avg : ℚ := 85
  min_score_for_average test1 test2 test3 test4 required_avg = 86 := by
  sorry

#eval min_score_for_average 84 87 78 90 85

end NUMINAMATH_CALUDE_chloe_min_score_l1150_115088


namespace NUMINAMATH_CALUDE_simplify_fraction_l1150_115077

theorem simplify_fraction : (180 / 16) * (5 / 120) * (8 / 3) = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1150_115077


namespace NUMINAMATH_CALUDE_angle_of_inclination_special_line_l1150_115030

/-- The angle of inclination of a line passing through points (1,0) and (0,-1) is π/4 -/
theorem angle_of_inclination_special_line : 
  let A : ℝ × ℝ := (1, 0)
  let B : ℝ × ℝ := (0, -1)
  let m : ℝ := (B.2 - A.2) / (B.1 - A.1)
  Real.arctan m = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_of_inclination_special_line_l1150_115030


namespace NUMINAMATH_CALUDE_equation_solutions_l1150_115071

theorem equation_solutions :
  (∀ x : ℝ, (x + 4) * (x - 2) = 3 * (x - 2) ↔ x = -1 ∨ x = 2) ∧
  (∀ x : ℝ, x^2 - x - 3 = 0 ↔ x = (1 + Real.sqrt 13) / 2 ∨ x = (1 - Real.sqrt 13) / 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1150_115071


namespace NUMINAMATH_CALUDE_line_passes_through_parabola_vertex_l1150_115019

/-- The number of values of a for which the line y = ax + a passes through the vertex of the parabola y = x^2 + ax -/
theorem line_passes_through_parabola_vertex : 
  ∃! (s : Finset ℝ), (∀ a ∈ s, ∃ x y : ℝ, 
    (y = a*x + a) ∧ 
    (y = x^2 + a*x) ∧ 
    (∀ x' y' : ℝ, y' = x'^2 + a*x' → y' ≥ y)) ∧ 
  Finset.card s = 2 := by
sorry

end NUMINAMATH_CALUDE_line_passes_through_parabola_vertex_l1150_115019


namespace NUMINAMATH_CALUDE_circle_radius_circumference_relation_l1150_115009

theorem circle_radius_circumference_relation (r : ℝ) (h : r > 0) :
  let c₁ := 2 * Real.pi * r
  let c₂ := 2 * Real.pi * (2 * r)
  c₂ = 2 * c₁ :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_circumference_relation_l1150_115009


namespace NUMINAMATH_CALUDE_rectangle_width_l1150_115060

theorem rectangle_width (area : ℝ) (length width : ℝ) : 
  area = 63 →
  width = length - 2 →
  area = length * width →
  width = 7 := by
sorry

end NUMINAMATH_CALUDE_rectangle_width_l1150_115060


namespace NUMINAMATH_CALUDE_blank_value_l1150_115011

theorem blank_value : (6 : ℝ) / Real.sqrt 18 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_blank_value_l1150_115011


namespace NUMINAMATH_CALUDE_expression_evaluation_l1150_115069

theorem expression_evaluation :
  let x : ℝ := 2
  let y : ℝ := -1
  3 * (2 * x^2 * y - x * y^2) - (4 * x^2 * y + x * y^2) = -16 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1150_115069


namespace NUMINAMATH_CALUDE_equation_solution_l1150_115040

theorem equation_solution :
  ∃! x : ℚ, x ≠ -3 ∧ (x^2 + 3*x + 4) / (x + 3) = x + 6 :=
by
  use -7/3
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1150_115040


namespace NUMINAMATH_CALUDE_g_difference_l1150_115085

noncomputable def g (n : ℤ) : ℝ :=
  (2 + Real.sqrt 2) / 4 * ((1 + Real.sqrt 2) / 2) ^ n + 
  (2 - Real.sqrt 2) / 4 * ((1 - Real.sqrt 2) / 2) ^ n

theorem g_difference (n : ℤ) : g (n + 1) - g (n - 1) = (Real.sqrt 2 / 2) * g n := by
  sorry

end NUMINAMATH_CALUDE_g_difference_l1150_115085


namespace NUMINAMATH_CALUDE_cube_opposite_face_l1150_115001

structure Cube where
  faces : Finset Char
  adjacent : Char → Char → Prop

def opposite (c : Cube) (x y : Char) : Prop :=
  x ∈ c.faces ∧ y ∈ c.faces ∧ x ≠ y ∧ ¬c.adjacent x y

theorem cube_opposite_face (c : Cube) :
  c.faces = {'А', 'Б', 'В', 'Г', 'Д', 'Е'} →
  c.adjacent 'В' 'А' →
  c.adjacent 'В' 'Д' →
  c.adjacent 'В' 'Е' →
  opposite c 'В' 'Г' := by
  sorry

end NUMINAMATH_CALUDE_cube_opposite_face_l1150_115001


namespace NUMINAMATH_CALUDE_composite_function_solution_l1150_115066

theorem composite_function_solution (f g : ℝ → ℝ) (a : ℝ) 
  (hf : ∀ x, f x = x / 3 + 2)
  (hg : ∀ x, g x = 5 - 2 * x)
  (h : f (g a) = 4) :
  a = -1/2 := by
sorry

end NUMINAMATH_CALUDE_composite_function_solution_l1150_115066


namespace NUMINAMATH_CALUDE_problem_solution_l1150_115032

noncomputable def AC (x : ℝ) : ℝ × ℝ := (Real.cos (x/2) + Real.sin (x/2), Real.sin (x/2))

noncomputable def BC (x : ℝ) : ℝ × ℝ := (Real.sin (x/2) - Real.cos (x/2), 2 * Real.cos (x/2))

noncomputable def f (x : ℝ) : ℝ := (AC x).1 * (BC x).1 + (AC x).2 * (BC x).2

theorem problem_solution :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ ∀ (S : ℝ), S > 0 ∧ (∀ (x : ℝ), f (x + S) = f x) → T ≤ S) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) → f x ≤ 1) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) → f x ≥ -1) ∧
  (∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x = 1) ∧
  (∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x = -1) ∧
  (∀ (x₁ x₂ : ℝ), x₁ ∈ Set.Ioo Real.pi (3 * Real.pi) ∧ x₂ ∈ Set.Ioo Real.pi (3 * Real.pi) ∧
    f x₁ = Real.sqrt 6 / 2 ∧ f x₂ = Real.sqrt 6 / 2 ∧ x₁ ≠ x₂ →
    x₁ + x₂ = 11 * Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1150_115032


namespace NUMINAMATH_CALUDE_baseball_distribution_l1150_115098

theorem baseball_distribution (total : ℕ) (classes : ℕ) (h1 : total = 43) (h2 : classes = 6) :
  total % classes = 1 := by
  sorry

end NUMINAMATH_CALUDE_baseball_distribution_l1150_115098


namespace NUMINAMATH_CALUDE_ivan_apple_purchase_l1150_115091

theorem ivan_apple_purchase (mini_pies : ℕ) (apples_per_mini_pie : ℚ) (leftover_apples : ℕ) 
  (h1 : mini_pies = 24)
  (h2 : apples_per_mini_pie = 1/2)
  (h3 : leftover_apples = 36) :
  (mini_pies : ℚ) * apples_per_mini_pie + leftover_apples = 48 := by
  sorry

end NUMINAMATH_CALUDE_ivan_apple_purchase_l1150_115091


namespace NUMINAMATH_CALUDE_smallest_multiple_of_7_greater_than_neg_50_l1150_115093

theorem smallest_multiple_of_7_greater_than_neg_50 :
  ∀ n : ℤ, n > -50 ∧ n % 7 = 0 → n ≥ -49 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_7_greater_than_neg_50_l1150_115093


namespace NUMINAMATH_CALUDE_square_area_from_vertices_l1150_115017

/-- The area of a square with adjacent vertices at (1,5) and (4,-2) is 58 -/
theorem square_area_from_vertices : 
  let x1 : ℝ := 1
  let y1 : ℝ := 5
  let x2 : ℝ := 4
  let y2 : ℝ := -2
  let side_length : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
  let area : ℝ := side_length^2
  area = 58 := by sorry

end NUMINAMATH_CALUDE_square_area_from_vertices_l1150_115017


namespace NUMINAMATH_CALUDE_initial_water_percentage_l1150_115094

theorem initial_water_percentage (
  initial_volume : ℝ) 
  (kola_percentage : ℝ)
  (added_sugar : ℝ) 
  (added_water : ℝ) 
  (added_kola : ℝ)
  (final_sugar_percentage : ℝ) :
  initial_volume = 340 →
  kola_percentage = 6 →
  added_sugar = 3.2 →
  added_water = 10 →
  added_kola = 6.8 →
  final_sugar_percentage = 14.111111111111112 →
  ∃ initial_water_percentage : ℝ,
    initial_water_percentage = 80 ∧
    initial_water_percentage + kola_percentage + (100 - initial_water_percentage - kola_percentage) = 100 ∧
    (((100 - initial_water_percentage - kola_percentage) / 100 * initial_volume + added_sugar) / 
      (initial_volume + added_sugar + added_water + added_kola)) * 100 = final_sugar_percentage :=
by sorry

end NUMINAMATH_CALUDE_initial_water_percentage_l1150_115094


namespace NUMINAMATH_CALUDE_system_solution_l1150_115055

theorem system_solution (x y k : ℝ) : 
  x + 2*y = k + 1 →
  2*x + y = 1 →
  x + y = 3 →
  k = 7 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l1150_115055


namespace NUMINAMATH_CALUDE_defect_probability_l1150_115045

/-- The probability of a randomly chosen unit being defective from two machines -/
theorem defect_probability
  (machine_a_ratio : ℝ)
  (machine_b_ratio : ℝ)
  (machine_a_defect_rate : ℝ)
  (machine_b_defect_rate : ℝ)
  (h1 : machine_a_ratio = 0.4)
  (h2 : machine_b_ratio = 0.6)
  (h3 : machine_a_ratio + machine_b_ratio = 1)
  (h4 : machine_a_defect_rate = 9 / 1000)
  (h5 : machine_b_defect_rate = 1 / 50) :
  machine_a_ratio * machine_a_defect_rate + machine_b_ratio * machine_b_defect_rate = 0.0156 :=
by sorry


end NUMINAMATH_CALUDE_defect_probability_l1150_115045


namespace NUMINAMATH_CALUDE_bus_seating_capacity_l1150_115086

/-- Calculates the total number of people who can sit in a bus with the given seating arrangement. -/
theorem bus_seating_capacity
  (left_seats : ℕ)
  (right_seats_difference : ℕ)
  (people_per_seat : ℕ)
  (back_seat_capacity : ℕ)
  (h1 : left_seats = 15)
  (h2 : right_seats_difference = 3)
  (h3 : people_per_seat = 3)
  (h4 : back_seat_capacity = 10) :
  left_seats * people_per_seat +
  (left_seats - right_seats_difference) * people_per_seat +
  back_seat_capacity = 91 := by
  sorry

end NUMINAMATH_CALUDE_bus_seating_capacity_l1150_115086
