import Mathlib

namespace cat_weight_ratio_l2702_270215

theorem cat_weight_ratio (female_weight male_weight : ℝ) : 
  female_weight = 2 →
  male_weight > female_weight →
  female_weight + male_weight = 6 →
  male_weight / female_weight = 2 := by
sorry

end cat_weight_ratio_l2702_270215


namespace range_of_f_l2702_270270

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x + 5

-- Define the domain
def domain : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 5}

-- State the theorem
theorem range_of_f : 
  {y : ℝ | ∃ x ∈ domain, f x = y} = {y : ℝ | 1 ≤ y ∧ y ≤ 10} := by sorry

end range_of_f_l2702_270270


namespace max_sum_of_squares_l2702_270280

theorem max_sum_of_squares (a b c d : ℝ) : 
  a + b = 15 →
  a * b + c + d = 78 →
  a * d + b * c = 160 →
  c * d = 96 →
  ∃ (max : ℝ), (∀ (a' b' c' d' : ℝ), 
    a' + b' = 15 →
    a' * b' + c' + d' = 78 →
    a' * d' + b' * c' = 160 →
    c' * d' = 96 →
    a'^2 + b'^2 + c'^2 + d'^2 ≤ max) ∧
  max = 717 :=
sorry

end max_sum_of_squares_l2702_270280


namespace equation_solution_l2702_270207

theorem equation_solution (x y : ℝ) : 9 * x^2 - 25 * y^2 = 0 ↔ x = (5/3) * y ∨ x = -(5/3) * y := by
  sorry

end equation_solution_l2702_270207


namespace problem_statement_l2702_270219

-- Define the base 10 logarithm
noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the problem statement
theorem problem_statement (a b : ℝ) :
  a > 0 ∧ b > 0 ∧
  (∃ (m n : ℕ), 
    Real.sqrt (log a) = m ∧
    Real.sqrt (log b) = n ∧
    m + 2*n + m^2 + (n^2/2) = 150) ∧
  a^2 * b = 10^81 →
  a * b = 10^85 := by
  sorry

end problem_statement_l2702_270219


namespace hyperbola_focal_distance_l2702_270294

/-- Given a hyperbola with equation x²/64 - y²/36 = 1 and foci F₁ and F₂,
    if P is a point on the hyperbola and |PF₁| = 17, then |PF₂| = 33 -/
theorem hyperbola_focal_distance (P F₁ F₂ : ℝ × ℝ) :
  (∃ x y : ℝ, P = (x, y) ∧ x^2/64 - y^2/36 = 1) →  -- P is on the hyperbola
  (∃ c : ℝ, c > 0 ∧ F₁ = (c, 0) ∧ F₂ = (-c, 0)) →  -- F₁ and F₂ are foci
  abs (P.1 - F₁.1) + abs (P.1 - F₁.2) = 17 →       -- |PF₁| = 17
  abs (P.1 - F₂.1) + abs (P.1 - F₂.2) = 33 :=      -- |PF₂| = 33
by sorry

end hyperbola_focal_distance_l2702_270294


namespace geometric_arithmetic_sequence_ratio_l2702_270254

/-- Given a geometric sequence {a_n} with common ratio q, prove that if 16a_1, 4a_2, and a_3 form an arithmetic sequence, then q = 4 -/
theorem geometric_arithmetic_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- {a_n} is a geometric sequence with common ratio q
  (16 * a 1 + a 3 = 2 * (4 * a 2)) →  -- 16a_1, 4a_2, a_3 form an arithmetic sequence
  q = 4 := by
sorry

end geometric_arithmetic_sequence_ratio_l2702_270254


namespace opposite_reciprocal_abs_one_result_l2702_270295

theorem opposite_reciprocal_abs_one_result (a b c d m : ℝ) : 
  (a = -b) → 
  (c * d = 1) → 
  (|m| = 1) → 
  ((a + b) * c * d - 2009 * m = -2009 ∨ (a + b) * c * d - 2009 * m = 2009) :=
by sorry

end opposite_reciprocal_abs_one_result_l2702_270295


namespace specific_trapezoid_area_l2702_270253

/-- Represents a circumscribed isosceles trapezoid -/
structure CircumscribedIsoscelesTrapezoid where
  long_base : ℝ
  base_angle : ℝ

/-- Calculates the area of a circumscribed isosceles trapezoid -/
def area (t : CircumscribedIsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem stating that the area of the specific trapezoid is 84 -/
theorem specific_trapezoid_area :
  let t : CircumscribedIsoscelesTrapezoid := {
    long_base := 24,
    base_angle := Real.arcsin 0.6
  }
  area t = 84 := by sorry

end specific_trapezoid_area_l2702_270253


namespace weight_of_replaced_person_l2702_270289

/-- Given a group of 6 persons, if replacing one person with a new person weighing 74 kg
    increases the average weight by 1.5 kg, then the weight of the person being replaced is 65 kg. -/
theorem weight_of_replaced_person (group_size : ℕ) (new_person_weight : ℝ) (average_increase : ℝ) :
  group_size = 6 →
  new_person_weight = 74 →
  average_increase = 1.5 →
  ∃ (original_average : ℝ) (replaced_person_weight : ℝ),
    group_size * (original_average + average_increase) =
    group_size * original_average - replaced_person_weight + new_person_weight ∧
    replaced_person_weight = 65 :=
by sorry

end weight_of_replaced_person_l2702_270289


namespace correct_head_start_for_dead_heat_l2702_270261

/-- The fraction of the race length that runner a should give as a head start to runner b -/
def head_start_fraction (speed_ratio : ℚ) : ℚ :=
  1 - (1 / speed_ratio)

/-- Theorem stating the correct head start fraction for the given speed ratio -/
theorem correct_head_start_for_dead_heat (race_length : ℚ) (speed_a speed_b : ℚ) 
  (h_speed : speed_a = 16/15 * speed_b) (h_positive : speed_b > 0) :
  head_start_fraction (speed_a / speed_b) * race_length = 1/16 * race_length :=
by sorry

end correct_head_start_for_dead_heat_l2702_270261


namespace initial_workers_count_l2702_270284

/-- The time it takes one person to complete the task -/
def total_time : ℕ := 40

/-- The time the initial group works -/
def initial_work_time : ℕ := 4

/-- The number of additional people joining -/
def additional_workers : ℕ := 2

/-- The time the expanded group works -/
def expanded_work_time : ℕ := 8

/-- Proves that the initial number of workers is 2 -/
theorem initial_workers_count : 
  ∃ (x : ℕ), 
    (initial_work_time * x + expanded_work_time * (x + additional_workers)) / total_time = 1 ∧ 
    x = 2 := by
  sorry

end initial_workers_count_l2702_270284


namespace exist_permutation_sum_all_nines_l2702_270258

/-- A function that checks if two natural numbers have the same digits (permutation) -/
def is_permutation (m n : ℕ) : Prop := sorry

/-- A function that checks if a natural number consists of all 9s -/
def all_nines (n : ℕ) : Prop := sorry

/-- Theorem stating the existence of two natural numbers satisfying the given conditions -/
theorem exist_permutation_sum_all_nines : 
  ∃ (m n : ℕ), is_permutation m n ∧ all_nines (m + n) := by sorry

end exist_permutation_sum_all_nines_l2702_270258


namespace area_between_sine_and_constant_line_l2702_270268

theorem area_between_sine_and_constant_line : 
  let f : ℝ → ℝ := λ x => Real.sin x
  let g : ℝ → ℝ := λ _ => (1/2 : ℝ)
  let lower_bound : ℝ := 0
  let upper_bound : ℝ := Real.pi
  ∃ (area : ℝ), area = ∫ x in lower_bound..upper_bound, |f x - g x| ∧ area = Real.sqrt 3 - Real.pi / 3 :=
by
  sorry

end area_between_sine_and_constant_line_l2702_270268


namespace courtney_marble_weight_l2702_270271

/-- The weight of Courtney's marble collection --/
def marbleCollectionWeight (firstJarCount : ℕ) (firstJarWeight : ℚ) 
  (secondJarWeight : ℚ) (thirdJarWeight : ℚ) : ℚ :=
  firstJarCount * firstJarWeight + 
  (2 * firstJarCount) * secondJarWeight + 
  (firstJarCount / 4) * thirdJarWeight

/-- Theorem stating the total weight of Courtney's marble collection --/
theorem courtney_marble_weight : 
  marbleCollectionWeight 80 (35/100) (45/100) (25/100) = 105 := by
  sorry

end courtney_marble_weight_l2702_270271


namespace sandwich_count_l2702_270232

/-- Represents the number of days in the workweek -/
def workweek_days : ℕ := 6

/-- Represents the cost of a donut in cents -/
def donut_cost : ℕ := 80

/-- Represents the cost of a sandwich in cents -/
def sandwich_cost : ℕ := 120

/-- Represents the condition that the total expenditure is an exact number of dollars -/
def is_exact_dollar_amount (sandwiches : ℕ) : Prop :=
  ∃ (dollars : ℕ), sandwich_cost * sandwiches + donut_cost * (workweek_days - sandwiches) = 100 * dollars

theorem sandwich_count : 
  ∃! (sandwiches : ℕ), sandwiches ≤ workweek_days ∧ is_exact_dollar_amount sandwiches ∧ sandwiches = 3 :=
sorry

end sandwich_count_l2702_270232


namespace counterfeit_coin_location_l2702_270200

/-- Represents a coin that can be either genuine or counterfeit -/
inductive Coin
  | genuine
  | counterfeit

/-- Represents the result of a weighing operation -/
inductive WeighingResult
  | equal
  | notEqual

/-- Function to perform a weighing operation on two pairs of coins -/
def weighPairs (c1 c2 c3 c4 : Coin) : WeighingResult :=
  sorry

/-- Theorem stating that we can narrow down the location of the counterfeit coin -/
theorem counterfeit_coin_location
  (coins : Fin 6 → Coin)
  (h_one_counterfeit : ∃! i, coins i = Coin.counterfeit) :
  (weighPairs (coins 0) (coins 1) (coins 2) (coins 3) = WeighingResult.equal
    → (coins 4 = Coin.counterfeit ∨ coins 5 = Coin.counterfeit))
  ∧
  (weighPairs (coins 0) (coins 1) (coins 2) (coins 3) = WeighingResult.notEqual
    → (coins 0 = Coin.counterfeit ∨ coins 1 = Coin.counterfeit ∨
       coins 2 = Coin.counterfeit ∨ coins 3 = Coin.counterfeit)) :=
  sorry

end counterfeit_coin_location_l2702_270200


namespace nested_sqrt_simplification_l2702_270286

theorem nested_sqrt_simplification (y : ℝ) (h : y ≥ 0) :
  Real.sqrt (y * Real.sqrt (y^3 * Real.sqrt y)) = y^(9/4) := by sorry

end nested_sqrt_simplification_l2702_270286


namespace population_below_five_percent_in_five_years_population_below_five_percent_in_2011_l2702_270220

/-- Represents the yearly decrease rate of the sparrow population -/
def yearly_decrease_rate : ℝ := 0.5

/-- Represents the target percentage of the original population -/
def target_percentage : ℝ := 0.05

/-- Calculates the population after a given number of years -/
def population_after_years (initial_population : ℝ) (years : ℕ) : ℝ :=
  initial_population * (yearly_decrease_rate ^ years)

/-- Theorem: It takes 5 years for the population to become less than 5% of the original -/
theorem population_below_five_percent_in_five_years (initial_population : ℝ) 
  (h : initial_population > 0) : 
  population_after_years initial_population 5 < target_percentage * initial_population ∧
  ∀ n : ℕ, n < 5 → population_after_years initial_population n ≥ target_percentage * initial_population :=
by sorry

/-- The year when the population becomes less than 5% of the original -/
def year_below_five_percent : ℕ := 2011

/-- Theorem: The population becomes less than 5% of the original in 2011 -/
theorem population_below_five_percent_in_2011 (initial_year : ℕ) (h : initial_year = 2006) :
  year_below_five_percent - initial_year = 5 :=
by sorry

end population_below_five_percent_in_five_years_population_below_five_percent_in_2011_l2702_270220


namespace solution_volume_l2702_270245

/-- Given a solution with 1.5 liters of pure acid and a concentration of 30%,
    prove that the total volume of the solution is 5 liters. -/
theorem solution_volume (volume_acid : ℝ) (concentration : ℝ) :
  volume_acid = 1.5 →
  concentration = 0.30 →
  (volume_acid / concentration) = 5 := by
  sorry

end solution_volume_l2702_270245


namespace second_quadrant_transformation_l2702_270256

/-- A point in the 2D plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is in the second quadrant. -/
def isInSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Theorem: If P(a,b) is in the second quadrant, then Q(-b,1-a) is also in the second quadrant. -/
theorem second_quadrant_transformation (a b : ℝ) :
  isInSecondQuadrant ⟨a, b⟩ → isInSecondQuadrant ⟨-b, 1-a⟩ := by
  sorry


end second_quadrant_transformation_l2702_270256


namespace aiden_sleep_fraction_l2702_270291

/-- Proves that 15 minutes is equal to 1/4 of an hour, given that an hour has 60 minutes. -/
theorem aiden_sleep_fraction (minutes_in_hour : ℕ) (aiden_sleep_minutes : ℕ) : 
  minutes_in_hour = 60 → aiden_sleep_minutes = 15 → 
  (aiden_sleep_minutes : ℚ) / minutes_in_hour = 1 / 4 := by
  sorry

end aiden_sleep_fraction_l2702_270291


namespace simplify_radical_expression_l2702_270229

theorem simplify_radical_expression : 
  Real.sqrt 7 - Real.sqrt 28 + Real.sqrt 63 = 2 * Real.sqrt 7 := by
  sorry

end simplify_radical_expression_l2702_270229


namespace base_conversion_problem_l2702_270246

-- Define a function to convert a number from base n to decimal
def to_decimal (digits : List Nat) (n : Nat) : Nat :=
  digits.enum.foldr (fun (i, digit) acc => acc + digit * n ^ i) 0

-- Define the problem statement
theorem base_conversion_problem (n : Nat) (d : Nat) :
  n > 0 →  -- n is a positive integer
  d < 10 →  -- d is a digit
  to_decimal [4, 5, d] n = 392 →  -- 45d in base n equals 392
  to_decimal [4, 5, 7] n = to_decimal [2, 1, d, 5] 7 →  -- 457 in base n equals 21d5 in base 7
  n + d = 12 := by
  sorry

end base_conversion_problem_l2702_270246


namespace not_p_or_q_is_false_l2702_270204

-- Define proposition p
def p : Prop := ∀ x : ℝ, (λ x : ℝ => x^3) (-x) = -((λ x : ℝ => x^3) x)

-- Define proposition q
def q : Prop := ∀ a b c : ℝ, b^2 = a*c → ∃ r : ℝ, (a = b/r ∧ b = c*r) ∨ (a = b*r ∧ b = c/r)

-- Theorem to prove
theorem not_p_or_q_is_false : ¬(¬p ∨ q) := by sorry

end not_p_or_q_is_false_l2702_270204


namespace abc_product_equals_k_absolute_value_l2702_270202

theorem abc_product_equals_k_absolute_value 
  (a b c k : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ k ≠ 0) 
  (h_eq : a + k / b = b + k / c ∧ b + k / c = c + k / a) : 
  |a * b * c| = |k| := by
  sorry

end abc_product_equals_k_absolute_value_l2702_270202


namespace painted_portion_is_five_eighths_additional_painting_needed_l2702_270296

-- Define the bridge as having a total length of 1
def bridge_length : ℝ := 1

-- Define the painted portion of the bridge
def painted_portion : ℝ → Prop := λ x => 
  -- The painted and unpainted portions sum to the total length
  x + (bridge_length - x) = bridge_length ∧
  -- If the painted portion increases by 30%, the unpainted portion decreases by 50%
  1.3 * x + 0.5 * (bridge_length - x) = bridge_length

-- Theorem: The painted portion is 5/8 of the bridge length
theorem painted_portion_is_five_eighths : 
  ∃ x : ℝ, painted_portion x ∧ x = 5/8 * bridge_length :=
sorry

-- Theorem: An additional 1/8 of the bridge length needs to be painted to have half the bridge painted
theorem additional_painting_needed : 
  ∃ x : ℝ, painted_portion x ∧ x + 1/8 * bridge_length = 1/2 * bridge_length :=
sorry

end painted_portion_is_five_eighths_additional_painting_needed_l2702_270296


namespace arithmetic_sequence_terms_l2702_270248

theorem arithmetic_sequence_terms (a₁ : ℝ) (aₙ : ℝ) (d : ℝ) (n : ℕ) :
  a₁ = 10 → aₙ = 150 → d = 5 → aₙ = a₁ + (n - 1) * d → n = 29 := by
  sorry

end arithmetic_sequence_terms_l2702_270248


namespace new_person_weight_l2702_270298

theorem new_person_weight (initial_count : ℕ) (replaced_weight : ℝ) (avg_increase : ℝ) :
  initial_count = 4 →
  replaced_weight = 70 →
  avg_increase = 3 →
  (initial_count : ℝ) * avg_increase + replaced_weight = 82 := by
  sorry

#check new_person_weight

end new_person_weight_l2702_270298


namespace parallel_lines_l2702_270225

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel (m1 m2 : ℝ) : Prop := m1 = m2

/-- Two lines are coincident if and only if they have the same slope and y-intercept -/
def coincident (m1 b1 m2 b2 : ℝ) : Prop := m1 = m2 ∧ b1 = b2

theorem parallel_lines (a : ℝ) : 
  (parallel (-a/2) (-3/(a-1)) ∧ ¬coincident (-a/2) (-1/2) (-3/(a-1)) (-1/(a-1))) → 
  a = -2 :=
sorry

end parallel_lines_l2702_270225


namespace cube_split_with_2023_l2702_270244

theorem cube_split_with_2023 (m : ℕ) (h1 : m > 1) : 
  (∃ (k : ℕ), 2 * k + 1 = 2023 ∧ 
   k ≥ (m + 2) * (m - 1) / 2 - m + 1 ∧ 
   k < (m + 2) * (m - 1) / 2 + 1) → 
  m = 45 := by
sorry

end cube_split_with_2023_l2702_270244


namespace custom_op_eight_twelve_l2702_270292

/-- The custom operation @ for positive integers -/
def custom_op (a b : ℕ+) : ℚ :=
  (a * b : ℚ) / (a + b + 1 : ℚ)

/-- Theorem stating that 8 @ 12 = 96/21 -/
theorem custom_op_eight_twelve : custom_op 8 12 = 96 / 21 := by
  sorry

end custom_op_eight_twelve_l2702_270292


namespace smallest_number_l2702_270264

theorem smallest_number (a b c d : ℝ) : 
  a = -2024 → b = -2022 → c = -2022.5 → d = 0 →
  (a < -2023 ∧ b > -2023 ∧ c > -2023 ∧ d > -2023) := by
  sorry

end smallest_number_l2702_270264


namespace boxes_with_neither_l2702_270274

theorem boxes_with_neither (total : ℕ) (pencils : ℕ) (pens : ℕ) (both : ℕ) 
  (h1 : total = 15)
  (h2 : pencils = 8)
  (h3 : pens = 5)
  (h4 : both = 4) :
  total - (pencils + pens - both) = 6 := by
sorry

end boxes_with_neither_l2702_270274


namespace alissa_presents_l2702_270290

theorem alissa_presents (ethan_presents : ℕ) (alissa_additional : ℕ) :
  ethan_presents = 31 →
  alissa_additional = 22 →
  ethan_presents + alissa_additional = 53 :=
by sorry

end alissa_presents_l2702_270290


namespace only_cylinder_has_rectangular_front_view_l2702_270259

-- Define the solid figures
inductive SolidFigure
  | Cylinder
  | TriangularPyramid
  | Sphere
  | Cone

-- Define the front view shapes
inductive FrontViewShape
  | Rectangle
  | Triangle
  | Circle

-- Function to determine the front view shape of a solid figure
def frontViewShape (figure : SolidFigure) : FrontViewShape :=
  match figure with
  | SolidFigure.Cylinder => FrontViewShape.Rectangle
  | SolidFigure.TriangularPyramid => FrontViewShape.Triangle
  | SolidFigure.Sphere => FrontViewShape.Circle
  | SolidFigure.Cone => FrontViewShape.Triangle

-- Theorem stating that only the cylinder has a rectangular front view
theorem only_cylinder_has_rectangular_front_view :
  ∀ (figure : SolidFigure),
    frontViewShape figure = FrontViewShape.Rectangle ↔ figure = SolidFigure.Cylinder :=
by sorry

end only_cylinder_has_rectangular_front_view_l2702_270259


namespace square_plot_area_l2702_270297

/-- Given a square plot with fencing cost of 58 Rs per foot and total fencing cost of 1160 Rs,
    the area of the plot is 25 square feet. -/
theorem square_plot_area (side_length : ℝ) (perimeter : ℝ) (area : ℝ) : 
  perimeter = 4 * side_length →
  58 * perimeter = 1160 →
  area = side_length ^ 2 →
  area = 25 := by
sorry

end square_plot_area_l2702_270297


namespace wx_plus_yz_equals_99_l2702_270236

theorem wx_plus_yz_equals_99 
  (w x y z : ℝ) 
  (h1 : w + x + y = -2)
  (h2 : w + x + z = 4)
  (h3 : w + y + z = 19)
  (h4 : x + y + z = 12) :
  w * x + y * z = 99 := by
sorry

end wx_plus_yz_equals_99_l2702_270236


namespace unique_solution_system_l2702_270243

/-- The system of equations has a unique solution (67/9, 1254/171) -/
theorem unique_solution_system :
  ∃! (x y : ℚ), (3 * x - 4 * y = -7) ∧ (6 * x - 5 * y = 8) :=
by
  sorry

end unique_solution_system_l2702_270243


namespace max_value_when_a_zero_one_zero_iff_a_positive_l2702_270287

noncomputable section

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := a * x - 1 / x - (a + 1) * Real.log x

-- Theorem for part 1
theorem max_value_when_a_zero :
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f 0 y ≤ f 0 x ∧ f 0 x = -1 :=
sorry

-- Theorem for part 2
theorem one_zero_iff_a_positive :
  ∀ (a : ℝ), (∃! (x : ℝ), x > 0 ∧ f a x = 0) ↔ a > 0 :=
sorry

end

end max_value_when_a_zero_one_zero_iff_a_positive_l2702_270287


namespace smallest_root_of_quadratic_l2702_270247

theorem smallest_root_of_quadratic (x : ℝ) :
  9 * x^2 - 45 * x + 50 = 0 → x ≥ 5/3 :=
by
  sorry

end smallest_root_of_quadratic_l2702_270247


namespace sqrt_equation_solutions_l2702_270250

theorem sqrt_equation_solutions :
  ∀ x : ℝ, (Real.sqrt (9 * x - 4) + 15 / Real.sqrt (9 * x - 4) = 8) ↔ (x = 29 / 9 ∨ x = 13 / 9) :=
by sorry

end sqrt_equation_solutions_l2702_270250


namespace sports_club_overlap_l2702_270272

theorem sports_club_overlap (total : ℕ) (badminton : ℕ) (tennis : ℕ) (neither : ℕ) 
  (h1 : total = 27)
  (h2 : badminton = 17)
  (h3 : tennis = 19)
  (h4 : neither = 2) :
  badminton + tennis - total + neither = 11 := by
  sorry

end sports_club_overlap_l2702_270272


namespace line_circle_intersection_range_l2702_270238

/-- Given a line intersecting a circle, prove the range of the parameter a -/
theorem line_circle_intersection_range (a : ℝ) : 
  (∃ A B : ℝ × ℝ, A ≠ B ∧ 
   (A.1 + A.2 + a = 0) ∧ (B.1 + B.2 + a = 0) ∧
   (A.1^2 + A.2^2 = 1) ∧ (B.1^2 + B.2^2 = 1) ∧
   (‖(A.1, A.2)‖ + ‖(B.1, B.2)‖)^2 ≥ ((A.1 - B.1)^2 + (A.2 - B.2)^2)) →
  a ∈ Set.Icc (-Real.sqrt 2) (-1) ∪ Set.Icc 1 (Real.sqrt 2) :=
by sorry

end line_circle_intersection_range_l2702_270238


namespace unique_solution_for_quadratic_equation_l2702_270249

/-- Given an equation (x+m)^2 - (x^2+n^2) = (m-n)^2 where m and n are unequal non-zero constants,
    prove that the unique solution for x in the form x = am + bn has a = 0 and b = -m + n. -/
theorem unique_solution_for_quadratic_equation 
  (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hmn : m ≠ n) :
  ∃! (a b : ℝ), ∀ (x : ℝ), 
    (x + m)^2 - (x^2 + n^2) = (m - n)^2 ↔ x = a*m + b*n ∧ a = 0 ∧ b = -m + n := by
  sorry

end unique_solution_for_quadratic_equation_l2702_270249


namespace alice_investment_ratio_l2702_270223

theorem alice_investment_ratio (initial_investment : ℝ) 
  (alice_final : ℝ) (bob_final : ℝ) :
  initial_investment = 2000 →
  bob_final = 6 * initial_investment →
  bob_final = alice_final + 8000 →
  alice_final / initial_investment = 2 :=
by sorry

end alice_investment_ratio_l2702_270223


namespace library_card_lineup_l2702_270234

theorem library_card_lineup : Nat.factorial 8 = 40320 := by
  sorry

end library_card_lineup_l2702_270234


namespace molly_current_age_l2702_270239

/-- Represents Molly's age and candle information --/
structure MollysBirthday where
  last_year_candles : ℕ
  additional_candles : ℕ
  friend_gift_candles : ℕ

/-- Calculates Molly's current age based on her birthday information --/
def current_age (mb : MollysBirthday) : ℕ :=
  mb.last_year_candles + 1

/-- Theorem stating Molly's current age --/
theorem molly_current_age (mb : MollysBirthday)
  (h1 : mb.last_year_candles = 14)
  (h2 : mb.additional_candles = 6)
  (h3 : mb.friend_gift_candles = 3) :
  current_age mb = 15 := by
  sorry

end molly_current_age_l2702_270239


namespace geometric_progression_sum_equality_l2702_270273

/-- Proves the equality for sums of geometric progression terms -/
theorem geometric_progression_sum_equality 
  (a q : ℝ) (n : ℕ) (h : q ≠ 1) :
  let S : ℕ → ℝ := λ k => a * (q^k - 1) / (q - 1)
  S n * (S (3*n) - S (2*n)) = (S (2*n) - S n)^2 := by
sorry

end geometric_progression_sum_equality_l2702_270273


namespace complex_number_in_fourth_quadrant_l2702_270201

/-- A complex number z satisfying z ⋅ i = 3 + 4i corresponds to a point in the fourth quadrant of the complex plane. -/
theorem complex_number_in_fourth_quadrant (z : ℂ) : z * Complex.I = 3 + 4 * Complex.I → z.re > 0 ∧ z.im < 0 := by
  sorry

end complex_number_in_fourth_quadrant_l2702_270201


namespace least_three_digit_product_12_l2702_270276

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_product (n : ℕ) : ℕ :=
  (n / 100) * ((n / 10) % 10) * (n % 10)

theorem least_three_digit_product_12 :
  ∀ n : ℕ, is_three_digit n → digit_product n = 12 → 126 ≤ n :=
sorry

end least_three_digit_product_12_l2702_270276


namespace patricia_lemon_heads_l2702_270217

/-- The number of Lemon Heads Patricia ate -/
def eaten : ℕ := 15

/-- The number of Lemon Heads Patricia gave to her friend -/
def given : ℕ := 5

/-- The number of Lemon Heads in each package -/
def per_package : ℕ := 3

/-- The function to calculate the number of packages -/
def calculate_packages (total : ℕ) : ℕ :=
  (total + per_package - 1) / per_package

/-- Theorem stating that Patricia originally had 7 packages of Lemon Heads -/
theorem patricia_lemon_heads : calculate_packages (eaten + given) = 7 := by
  sorry

end patricia_lemon_heads_l2702_270217


namespace sum_of_reciprocals_of_roots_l2702_270205

theorem sum_of_reciprocals_of_roots (x : ℝ) : 
  x^2 - 15*x + 36 = 0 → 
  ∃ s₁ s₂ : ℝ, s₁ ≠ s₂ ∧ x^2 - 15*x + 36 = (x - s₁) * (x - s₂) ∧ 
  (1 / s₁ + 1 / s₂ = 5 / 12) :=
by sorry

end sum_of_reciprocals_of_roots_l2702_270205


namespace hyperbola_asymptotes_l2702_270212

/-- The equation of a hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 25 - y^2 / 4 = 1

/-- The equation of the asymptotes -/
def asymptote_equation (x y : ℝ) : Prop :=
  y = 2/5 * x ∨ y = -2/5 * x

/-- Theorem: The asymptotes of the given hyperbola are y = ±(2/5)x -/
theorem hyperbola_asymptotes :
  ∀ x y : ℝ, hyperbola_equation x y → asymptote_equation x y :=
sorry

end hyperbola_asymptotes_l2702_270212


namespace cube_greater_than_l2702_270206

theorem cube_greater_than (a b : ℝ) : a > b → ¬(a^3 ≤ b^3) := by
  sorry

end cube_greater_than_l2702_270206


namespace train_length_l2702_270267

/-- The length of a train given its speed, time to cross a platform, and the platform's length -/
theorem train_length (speed : ℝ) (time : ℝ) (platform_length : ℝ) : 
  speed = 90 * (1000 / 3600) → 
  time = 25 → 
  platform_length = 400.05 → 
  speed * time - platform_length = 224.95 := by sorry

end train_length_l2702_270267


namespace quadratic_circle_properties_l2702_270230

/-- A quadratic function that intersects both coordinate axes at three points -/
structure QuadraticFunction where
  b : ℝ
  intersects_axes : ∃ (x₁ x₂ y : ℝ), x₁ ≠ x₂ ∧ 
    x₁^2 + 2*x₁ + b = 0 ∧ 
    x₂^2 + 2*x₂ + b = 0 ∧ 
    b = y

/-- The circle passing through the three intersection points -/
def circle_equation (f : QuadraticFunction) (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - (f.b + 1)*y + f.b = 0

theorem quadratic_circle_properties (f : QuadraticFunction) :
  (f.b < 1 ∧ f.b ≠ 0) ∧
  (∀ x y, circle_equation f x y ↔ x^2 + y^2 + 2*x - (f.b + 1)*y + f.b = 0) ∧
  circle_equation f (-2) 1 ∧
  circle_equation f 0 1 := by
  sorry

end quadratic_circle_properties_l2702_270230


namespace quadratic_inequality_solution_l2702_270252

theorem quadratic_inequality_solution (x : ℝ) :
  (x^2 + 3*x < 10) ↔ (-5 < x ∧ x < 2) :=
by sorry

end quadratic_inequality_solution_l2702_270252


namespace point_motion_l2702_270216

/-- Given two points A and B on a number line, prove properties about their motion and positions. -/
theorem point_motion (a b : ℝ) (h : |a + 20| + |b - 12| = 0) :
  -- 1. Initial positions
  (a = -20 ∧ b = 12) ∧ 
  -- 2. Time when A and B are equidistant from origin
  (∃ t : ℝ, t = 2 ∧ |a - 6*t| = |b - 2*t|) ∧
  -- 3. Times when A and B are 8 units apart
  (∃ t : ℝ, (t = 3 ∨ t = 5 ∨ t = 10) ∧ 
    |a - 6*t - (b - 2*t)| = 8) := by
  sorry

end point_motion_l2702_270216


namespace coprime_in_ten_consecutive_integers_l2702_270240

theorem coprime_in_ten_consecutive_integers (k : ℤ) :
  ∃ n ∈ Finset.range 10, ∀ m ∈ Finset.range 10, m ≠ n → Int.gcd (k + n) (k + m) = 1 := by
  sorry

end coprime_in_ten_consecutive_integers_l2702_270240


namespace quadratic_transformation_l2702_270299

theorem quadratic_transformation (x : ℝ) :
  (x^2 - 4*x + 3 = 0) → (∃ h k : ℝ, (x + h)^2 = k ∧ k = 1) :=
by
  sorry

end quadratic_transformation_l2702_270299


namespace cost_of_gums_in_dollars_l2702_270281

-- Define the cost of one piece of gum in cents
def cost_per_gum : ℕ := 5

-- Define the number of pieces of gum
def num_gums : ℕ := 2000

-- Define the number of cents in a dollar
def cents_per_dollar : ℕ := 100

-- Theorem to prove
theorem cost_of_gums_in_dollars : 
  (cost_per_gum * num_gums) / cents_per_dollar = 100 := by
  sorry

end cost_of_gums_in_dollars_l2702_270281


namespace symmetric_point_coordinates_l2702_270283

/-- Given a point M with polar coordinates (6, 11π/6), 
    the Cartesian coordinates of the point symmetric to M 
    with respect to the y-axis are (-3√3, -3) -/
theorem symmetric_point_coordinates : 
  let r : ℝ := 6
  let θ : ℝ := 11 * π / 6
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (- x, y) = (-3 * Real.sqrt 3, -3) := by sorry

end symmetric_point_coordinates_l2702_270283


namespace ellipse_and_tangents_l2702_270255

/-- An ellipse with given properties -/
structure Ellipse :=
  (a : ℝ)
  (b : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : a^2 * (9/4) + b^2 = a^2 * b^2)  -- Passes through (1, 3/2)
  (h4 : a^2 - b^2 = (1/4) * a^2)  -- Eccentricity = 1/2

/-- The main theorem about the ellipse and tangent lines -/
theorem ellipse_and_tangents (C : Ellipse) :
  C.a = 2 ∧ C.b = Real.sqrt 3 ∧
  ∀ (r : ℝ) (hr : 0 < r ∧ r < 3/2),
    ∃ (k : ℝ), ∀ (M N : ℝ × ℝ),
      (M.1^2 / 4 + M.2^2 / 3 = 1) →
      (N.1^2 / 4 + N.2^2 / 3 = 1) →
      (∃ (k1 k2 : ℝ),
        (M.2 - 3/2 = k1 * (M.1 - 1)) ∧
        (N.2 - 3/2 = k2 * (N.1 - 1)) ∧
        ((k1 * (1 - M.1))^2 + (3/2 - M.2)^2 = r^2) ∧
        ((k2 * (1 - N.1))^2 + (3/2 - N.2)^2 = r^2) ∧
        k1 = -k2) →
      (N.2 - M.2) / (N.1 - M.1) = k ∧ k = 1/2 :=
by sorry

end ellipse_and_tangents_l2702_270255


namespace exponent_multiplication_l2702_270235

theorem exponent_multiplication (x : ℝ) : x^3 * x^3 = x^6 := by
  sorry

end exponent_multiplication_l2702_270235


namespace f_is_even_and_increasing_l2702_270263

-- Define the function f(x) = 2x^2 + 4
def f (x : ℝ) : ℝ := 2 * x^2 + 4

-- State the theorem
theorem f_is_even_and_increasing :
  (∀ x, f x = f (-x)) ∧ 
  (∀ x y, 0 < x ∧ x < y ∧ y < 1 → f x < f y) :=
sorry

end f_is_even_and_increasing_l2702_270263


namespace parallel_and_perpendicular_properties_l2702_270279

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines
variable (parallel : Line → Line → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem parallel_and_perpendicular_properties 
  (a b c : Line) (y : Plane) :
  (∀ a b c, parallel a b → parallel b c → parallel a c) ∧
  (∀ a b, perpendicular a y → perpendicular b y → parallel a b) :=
sorry

end parallel_and_perpendicular_properties_l2702_270279


namespace jeopardy_episode_length_l2702_270277

/-- The length of one episode of Jeopardy in minutes -/
def jeopardy_length : ℝ := sorry

/-- The length of one episode of Wheel of Fortune in minutes -/
def wheel_of_fortune_length : ℝ := sorry

/-- The total number of episodes James watched -/
def total_episodes : ℕ := sorry

/-- The total time James spent watching TV in minutes -/
def total_watch_time : ℝ := sorry

theorem jeopardy_episode_length :
  jeopardy_length = 20 ∧
  wheel_of_fortune_length = 2 * jeopardy_length ∧
  total_episodes = 4 ∧
  total_watch_time = 120 ∧
  total_watch_time = 2 * jeopardy_length + 2 * wheel_of_fortune_length :=
by sorry

end jeopardy_episode_length_l2702_270277


namespace second_player_wins_l2702_270278

/-- Represents the state of the game board as a list of integers -/
def GameBoard := List Nat

/-- The initial game board with 2022 ones -/
def initialBoard : GameBoard := List.replicate 2022 1

/-- A player in the game -/
inductive Player
| First
| Second

/-- The result of a game -/
inductive GameResult
| FirstWin
| SecondWin
| Draw

/-- A move in the game, represented by the index of the first number to be replaced -/
def Move := Nat

/-- Apply a move to the game board -/
def applyMove (board : GameBoard) (move : Move) : GameBoard :=
  sorry

/-- Check if a player has won -/
def hasWon (board : GameBoard) : Bool :=
  sorry

/-- Check if the game is a draw -/
def isDraw (board : GameBoard) : Bool :=
  sorry

/-- A strategy for a player -/
def Strategy := GameBoard → Move

/-- The second player's strategy -/
def secondPlayerStrategy : Strategy :=
  sorry

/-- The game result when both players play optimally -/
def gameResult (firstStrategy secondStrategy : Strategy) : GameResult :=
  sorry

/-- Theorem stating that the second player has a winning strategy -/
theorem second_player_wins :
  ∃ (secondStrategy : Strategy),
    ∀ (firstStrategy : Strategy),
      gameResult firstStrategy secondStrategy = GameResult.SecondWin :=
sorry

end second_player_wins_l2702_270278


namespace painted_subcubes_count_l2702_270208

/-- Represents a cube with painted faces -/
structure PaintedCube :=
  (size : ℕ)
  (isPainted : ℕ → ℕ → ℕ → Bool)

/-- Counts subcubes with at least two painted faces -/
def countSubcubesWithTwoPaintedFaces (cube : PaintedCube) : ℕ :=
  sorry

/-- The main theorem -/
theorem painted_subcubes_count (cube : PaintedCube) 
  (h1 : cube.size = 4)
  (h2 : ∀ x y z, (x = 0 ∨ x = cube.size - 1 ∨ 
                  y = 0 ∨ y = cube.size - 1 ∨ 
                  z = 0 ∨ z = cube.size - 1) → 
                 cube.isPainted x y z = true) :
  countSubcubesWithTwoPaintedFaces cube = 32 :=
sorry

end painted_subcubes_count_l2702_270208


namespace brothers_identity_l2702_270218

-- Define the two brothers
inductive Brother
| Tweedledum
| Tweedledee

-- Define a function to represent a brother's statement
def statement (b : Brother) : Brother :=
  match b with
  | Brother.Tweedledum => Brother.Tweedledum
  | Brother.Tweedledee => Brother.Tweedledee

-- Define the consistency of statements
def consistent (first second : Brother) : Prop :=
  (statement first = Brother.Tweedledum) ∧ (statement second = Brother.Tweedledee)

-- Theorem: The only consistent scenario is when both brothers tell the truth
theorem brothers_identity :
  ∀ (first second : Brother),
    consistent first second →
    (first = Brother.Tweedledum ∧ second = Brother.Tweedledee) :=
by sorry


end brothers_identity_l2702_270218


namespace isosceles_when_neg_one_root_right_triangle_when_equal_roots_equilateral_roots_l2702_270231

/-- Triangle represented by side lengths -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c

/-- Quadratic equation associated with a triangle -/
def triangleQuadratic (t : Triangle) (x : ℝ) : ℝ :=
  (t.a + t.b) * x^2 + 2 * t.c * x + (t.b - t.a)

theorem isosceles_when_neg_one_root (t : Triangle) :
  triangleQuadratic t (-1) = 0 → t.b = t.c := by sorry

theorem right_triangle_when_equal_roots (t : Triangle) :
  (2 * t.c)^2 = 4 * (t.a + t.b) * (t.b - t.a) → t.a^2 + t.c^2 = t.b^2 := by sorry

theorem equilateral_roots (t : Triangle) :
  t.a = t.b ∧ t.b = t.c →
  (∃ x : ℝ, triangleQuadratic t x = 0) →
  (triangleQuadratic t 0 = 0 ∧ triangleQuadratic t (-1) = 0) := by sorry

end isosceles_when_neg_one_root_right_triangle_when_equal_roots_equilateral_roots_l2702_270231


namespace intersection_theorem_l2702_270275

/-- The line y = x + m intersects the circle x^2 + y^2 - 2x + 4y - 4 = 0 at two distinct points A and B. -/
def intersects_at_two_points (m : ℝ) : Prop :=
  ∃ A B : ℝ × ℝ, A ≠ B ∧
    (A.1^2 + A.2^2 - 2*A.1 + 4*A.2 - 4 = 0) ∧
    (B.1^2 + B.2^2 - 2*B.1 + 4*B.2 - 4 = 0) ∧
    (A.2 = A.1 + m) ∧ (B.2 = B.1 + m)

/-- The circle with diameter AB passes through the origin. -/
def circle_passes_origin (A B : ℝ × ℝ) : Prop :=
  (A.1 * B.1 + A.2 * B.2 = 0)

/-- Main theorem about the intersection of the line and circle. -/
theorem intersection_theorem :
  (∀ m : ℝ, intersects_at_two_points m ↔ -3-3*Real.sqrt 2 < m ∧ m < -3+3*Real.sqrt 2) ∧
  (∀ m : ℝ, intersects_at_two_points m →
    (∃ A B : ℝ × ℝ, A ≠ B ∧ circle_passes_origin A B) →
    (m = -4 ∨ m = 1)) :=
sorry

end intersection_theorem_l2702_270275


namespace cara_seating_arrangement_l2702_270233

theorem cara_seating_arrangement (n : ℕ) (k : ℕ) : n = 7 ∧ k = 2 → Nat.choose n k = 21 := by
  sorry

end cara_seating_arrangement_l2702_270233


namespace intersection_of_A_and_B_l2702_270203

def A : Set ℝ := {x | x^2 - 2*x = 0}
def B : Set ℝ := {0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 2} := by
  sorry

end intersection_of_A_and_B_l2702_270203


namespace fudge_difference_l2702_270210

/-- Converts pounds to ounces -/
def pounds_to_ounces (pounds : ℚ) : ℚ := 16 * pounds

theorem fudge_difference (marina_fudge : ℚ) (lazlo_fudge : ℚ) : 
  marina_fudge = 4.5 →
  lazlo_fudge = pounds_to_ounces 4 - 6 →
  pounds_to_ounces marina_fudge - lazlo_fudge = 14 := by
  sorry

end fudge_difference_l2702_270210


namespace existence_of_irrational_term_l2702_270269

theorem existence_of_irrational_term (a : ℕ → ℝ) 
  (h_pos : ∀ n, a n > 0)
  (h_rec : ∀ n, (a (n + 1))^2 = a n + 1) :
  ∃ n, Irrational (a n) :=
sorry

end existence_of_irrational_term_l2702_270269


namespace find_number_l2702_270282

theorem find_number : ∃ x : ℝ, 3 * (2 * x + 9) = 51 :=
by
  sorry

end find_number_l2702_270282


namespace fair_coin_head_is_random_event_l2702_270214

/-- Represents the possible outcomes of a coin toss -/
inductive CoinOutcome
  | Head
  | Tail

/-- Represents different types of events -/
inductive EventType
  | Impossible
  | Certain
  | Random
  | Definite

/-- A fair coin toss -/
structure FairCoinToss where
  /-- The coin has two possible outcomes -/
  outcome : CoinOutcome
  /-- The probability of getting heads is 0.5 -/
  prob_head : ℝ := 0.5
  /-- The probability of getting tails is 0.5 -/
  prob_tail : ℝ := 0.5
  /-- The probabilities sum to 1 -/
  prob_sum : prob_head + prob_tail = 1

/-- The theorem stating that tossing a fair coin with the head facing up is a random event -/
theorem fair_coin_head_is_random_event (toss : FairCoinToss) : 
  EventType.Random = 
    match toss.outcome with
    | CoinOutcome.Head => EventType.Random
    | CoinOutcome.Tail => EventType.Random :=
by
  sorry


end fair_coin_head_is_random_event_l2702_270214


namespace divisibility_by_eleven_l2702_270288

theorem divisibility_by_eleven (n : ℤ) : 
  11 ∣ (n^2001 - n^4) ↔ n % 11 = 0 ∨ n % 11 = 1 := by
  sorry

end divisibility_by_eleven_l2702_270288


namespace score_79_implies_93_correct_l2702_270241

/-- Represents the grading system for a test -/
structure TestGrade where
  total_questions : ℕ
  correct_answers : ℕ
  score : ℤ

/-- Theorem stating that for a 100-question test with the given grading system,
    a score of 79 implies 93 correct answers -/
theorem score_79_implies_93_correct
  (test : TestGrade)
  (h1 : test.total_questions = 100)
  (h2 : test.score = test.correct_answers - 2 * (test.total_questions - test.correct_answers))
  (h3 : test.score = 79) :
  test.correct_answers = 93 := by
  sorry

end score_79_implies_93_correct_l2702_270241


namespace cube_vertex_shapes_l2702_270285

-- Define a cube
structure Cube where
  vertices : Finset (Fin 8)
  edges : Finset (Fin 8 × Fin 8)
  faces : Finset (Finset (Fin 8))

-- Define the types of shapes we're interested in
inductive ShapeType
  | Rectangle
  | NonRectangleParallelogram
  | IsoscelesRightTetrahedron
  | RegularTetrahedron
  | RightTetrahedron

-- Function to check if 4 vertices form a specific shape
def formsShape (c : Cube) (v : Finset (Fin 8)) (s : ShapeType) : Prop :=
  v.card = 4 ∧ v ⊆ c.vertices ∧ match s with
    | ShapeType.Rectangle => sorry
    | ShapeType.NonRectangleParallelogram => sorry
    | ShapeType.IsoscelesRightTetrahedron => sorry
    | ShapeType.RegularTetrahedron => sorry
    | ShapeType.RightTetrahedron => sorry

-- Theorem statement
theorem cube_vertex_shapes (c : Cube) :
  (∃ v, formsShape c v ShapeType.Rectangle) ∧
  (∃ v, formsShape c v ShapeType.IsoscelesRightTetrahedron) ∧
  (∃ v, formsShape c v ShapeType.RegularTetrahedron) ∧
  (∃ v, formsShape c v ShapeType.RightTetrahedron) ∧
  (¬ ∃ v, formsShape c v ShapeType.NonRectangleParallelogram) :=
sorry

end cube_vertex_shapes_l2702_270285


namespace probability_red_before_green_l2702_270211

def red_chips : ℕ := 4
def green_chips : ℕ := 3
def total_chips : ℕ := red_chips + green_chips

def favorable_arrangements : ℕ := Nat.choose (total_chips - 1) green_chips
def total_arrangements : ℕ := Nat.choose total_chips green_chips

theorem probability_red_before_green :
  (favorable_arrangements : ℚ) / total_arrangements = 4 / 7 := by
  sorry

end probability_red_before_green_l2702_270211


namespace debby_soda_bottles_l2702_270221

/-- The number of soda bottles Debby drinks per day -/
def soda_per_day : ℕ := 9

/-- The number of days the soda bottles lasted -/
def days_lasted : ℕ := 40

/-- The total number of soda bottles Debby bought -/
def total_soda_bottles : ℕ := soda_per_day * days_lasted

/-- Theorem stating that the total number of soda bottles Debby bought is 360 -/
theorem debby_soda_bottles : total_soda_bottles = 360 := by
  sorry

end debby_soda_bottles_l2702_270221


namespace book_magazine_cost_l2702_270228

theorem book_magazine_cost (x y : ℝ) 
  (h1 : 3 * x + 2 * y = 18.40)
  (h2 : 2 * x + 3 * y = 17.60) :
  2 * x + y = 11.20 := by
  sorry

end book_magazine_cost_l2702_270228


namespace craig_walking_distance_l2702_270265

/-- The distance Craig rode on the bus in miles -/
def bus_distance : ℝ := 3.83

/-- The difference between the bus distance and walking distance in miles -/
def distance_difference : ℝ := 3.67

/-- The distance Craig walked in miles -/
def walking_distance : ℝ := bus_distance - distance_difference

theorem craig_walking_distance : walking_distance = 0.16 := by
  sorry

end craig_walking_distance_l2702_270265


namespace cos_max_value_l2702_270237

theorem cos_max_value (a b : ℝ) (h : Real.cos (a + b) = Real.cos a + Real.cos b) :
  ∃ (max : ℝ), max = Real.sqrt 3 - 1 ∧ Real.cos a ≤ max ∧ 
  ∃ (a₀ b₀ : ℝ), Real.cos (a₀ + b₀) = Real.cos a₀ + Real.cos b₀ ∧ Real.cos a₀ = max :=
by sorry

end cos_max_value_l2702_270237


namespace circle_graph_proportion_l2702_270251

theorem circle_graph_proportion (total_degrees : ℝ) (sector_degrees : ℝ) 
  (h1 : total_degrees = 360) 
  (h2 : sector_degrees = 180) : 
  sector_degrees / total_degrees = 1/2 := by
  sorry

end circle_graph_proportion_l2702_270251


namespace sqrt_20_plus_sqrt_5_over_sqrt_5_minus_2_l2702_270293

theorem sqrt_20_plus_sqrt_5_over_sqrt_5_minus_2 :
  (Real.sqrt 20 + Real.sqrt 5) / Real.sqrt 5 - 2 = 1 := by sorry

end sqrt_20_plus_sqrt_5_over_sqrt_5_minus_2_l2702_270293


namespace elective_schemes_count_l2702_270262

/-- The number of courses offered -/
def total_courses : ℕ := 9

/-- The number of mutually exclusive courses -/
def exclusive_courses : ℕ := 3

/-- The number of courses each student must choose -/
def courses_to_choose : ℕ := 4

/-- The number of different elective schemes -/
def elective_schemes : ℕ := 75

theorem elective_schemes_count :
  (Nat.choose exclusive_courses 1 * Nat.choose (total_courses - exclusive_courses) (courses_to_choose - 1)) +
  (Nat.choose (total_courses - exclusive_courses) courses_to_choose) = elective_schemes :=
by sorry

end elective_schemes_count_l2702_270262


namespace cobbler_charge_percentage_l2702_270266

theorem cobbler_charge_percentage (mold_cost : ℝ) (hourly_rate : ℝ) (hours_worked : ℝ) (total_paid : ℝ)
  (h1 : mold_cost = 250)
  (h2 : hourly_rate = 75)
  (h3 : hours_worked = 8)
  (h4 : total_paid = 730) :
  (1 - total_paid / (mold_cost + hourly_rate * hours_worked)) * 100 = 20 := by
  sorry

end cobbler_charge_percentage_l2702_270266


namespace polynomial_remainder_l2702_270213

theorem polynomial_remainder (p : ℝ → ℝ) (h1 : p 2 = 3) (h2 : p 3 = 9) :
  ∃ q : ℝ → ℝ, ∀ x, p x = (x - 2) * (x - 3) * q x + (6 * x - 9) := by
  sorry

end polynomial_remainder_l2702_270213


namespace stella_annual_income_l2702_270224

/-- Calculates the annual income given monthly income and months of unpaid leave -/
def annual_income (monthly_income : ℕ) (unpaid_leave_months : ℕ) : ℕ :=
  monthly_income * (12 - unpaid_leave_months)

/-- Theorem: Given Stella's monthly income and unpaid leave, her annual income is 49190 dollars -/
theorem stella_annual_income :
  annual_income 4919 2 = 49190 := by
  sorry

end stella_annual_income_l2702_270224


namespace least_number_with_remainder_least_number_is_174_main_result_l2702_270242

theorem least_number_with_remainder (n : ℕ) : 
  (n % 34 = 4 ∧ n % 5 = 4) → n ≥ 174 := by
  sorry

theorem least_number_is_174 : 
  174 % 34 = 4 ∧ 174 % 5 = 4 := by
  sorry

theorem main_result : 
  ∀ n : ℕ, (n % 34 = 4 ∧ n % 5 = 4) → n ≥ 174 ∧ (174 % 34 = 4 ∧ 174 % 5 = 4) := by
  sorry

end least_number_with_remainder_least_number_is_174_main_result_l2702_270242


namespace smallest_cube_root_with_small_fraction_l2702_270257

theorem smallest_cube_root_with_small_fraction : 
  ∃ (m : ℕ) (r : ℝ), 
    0 < r ∧ r < 1/2000 ∧ 
    (∃ (n : ℕ), n > 0 ∧ m = (n + r)^3) ∧
    (∀ (k : ℕ) (s : ℝ), 0 < k ∧ k < 26 → 0 < s ∧ s < 1/2000 → ¬(∃ (l : ℕ), l = (k + s)^3)) ∧
    (∃ (n : ℕ) (r : ℝ), n = 26 ∧ 0 < r ∧ r < 1/2000 ∧ (∃ (m : ℕ), m = (n + r)^3)) :=
by sorry

end smallest_cube_root_with_small_fraction_l2702_270257


namespace kelly_initial_games_l2702_270226

/-- The number of games Kelly needs to give away -/
def games_to_give : ℕ := 15

/-- The number of games Kelly will have left after giving away some games -/
def games_left : ℕ := 35

/-- Kelly's initial number of games -/
def initial_games : ℕ := games_left + games_to_give

theorem kelly_initial_games : initial_games = 50 := by sorry

end kelly_initial_games_l2702_270226


namespace integral_rational_function_l2702_270260

open Real

theorem integral_rational_function (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 3) (h3 : x ≠ -1) :
  (deriv fun x => 2*x + 4*log (abs x) + log (abs (x - 3)) - 2*log (abs (x + 1))) x =
  (2*x^3 - x^2 - 7*x - 12) / (x*(x-3)*(x+1)) :=
by sorry

end integral_rational_function_l2702_270260


namespace beach_towel_loads_l2702_270227

/-- The number of laundry loads required for beach towels during a vacation. -/
def laundry_loads (num_families : ℕ) (people_per_family : ℕ) (days : ℕ) 
                  (towels_per_person_per_day : ℕ) (towels_per_load : ℕ) : ℕ :=
  (num_families * people_per_family * days * towels_per_person_per_day) / towels_per_load

theorem beach_towel_loads : 
  laundry_loads 7 6 10 2 10 = 84 := by sorry

end beach_towel_loads_l2702_270227


namespace bob_walking_distance_l2702_270222

/-- Proves that Bob walked 16 miles when he met Yolanda given the problem conditions --/
theorem bob_walking_distance (total_distance : ℝ) (yolanda_speed : ℝ) (bob_speed : ℝ) 
  (head_start : ℝ) : 
  total_distance = 31 ∧ 
  yolanda_speed = 3 ∧ 
  bob_speed = 4 ∧ 
  head_start = 1 →
  ∃ t : ℝ, t > 0 ∧ yolanda_speed * (t + head_start) + bob_speed * t = total_distance ∧ 
  bob_speed * t = 16 :=
by sorry

end bob_walking_distance_l2702_270222


namespace smallest_palindrome_base2_base4_l2702_270209

/-- Convert a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec toBinaryAux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else toBinaryAux (m / 2) ((m % 2) :: acc)
  toBinaryAux n []

/-- Convert a natural number to its base 4 representation as a list of digits -/
def toBase4 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec toBase4Aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else toBase4Aux (m / 4) ((m % 4) :: acc)
  toBase4Aux n []

/-- Check if a list is a palindrome -/
def isPalindrome (l : List ℕ) : Prop :=
  l = l.reverse

/-- The main theorem statement -/
theorem smallest_palindrome_base2_base4 :
  ∀ n : ℕ, n > 10 →
  (isPalindrome (toBinary n) ∧ isPalindrome (toBase4 n)) →
  n ≥ 17 :=
sorry

end smallest_palindrome_base2_base4_l2702_270209
