import Mathlib

namespace NUMINAMATH_CALUDE_medal_award_scenario_l3674_367437

/-- The number of ways to award medals in a specific race scenario -/
def medal_award_ways (total_sprinters : ℕ) (italian_sprinters : ℕ) : ℕ :=
  let non_italian_sprinters := total_sprinters - italian_sprinters
  italian_sprinters * non_italian_sprinters * (non_italian_sprinters - 1)

/-- Theorem stating the number of ways to award medals in the given scenario -/
theorem medal_award_scenario : medal_award_ways 10 4 = 120 := by
  sorry

end NUMINAMATH_CALUDE_medal_award_scenario_l3674_367437


namespace NUMINAMATH_CALUDE_complex_sum_zero_l3674_367494

noncomputable def w : ℂ := Complex.exp (Complex.I * (3 * Real.pi / 8))

theorem complex_sum_zero :
  w / (1 + w^3) + w^2 / (1 + w^5) + w^3 / (1 + w^7) = 0 := by sorry

end NUMINAMATH_CALUDE_complex_sum_zero_l3674_367494


namespace NUMINAMATH_CALUDE_similar_triangle_longest_side_l3674_367480

/-- Given a triangle with sides 8, 10, and 12, and a similar triangle with perimeter 150,
    prove that the longest side of the similar triangle is 60. -/
theorem similar_triangle_longest_side
  (a b c : ℝ)
  (h_original : a = 8 ∧ b = 10 ∧ c = 12)
  (h_similar_perimeter : ∃ k : ℝ, k * (a + b + c) = 150)
  : ∃ x y z : ℝ, x = k * a ∧ y = k * b ∧ z = k * c ∧ max x (max y z) = 60 :=
sorry

end NUMINAMATH_CALUDE_similar_triangle_longest_side_l3674_367480


namespace NUMINAMATH_CALUDE_millicent_book_fraction_l3674_367425

/-- Given that:
    - Harold has 1/2 as many books as Millicent
    - Harold brings 1/3 of his books to the new home
    - The new home's library capacity is 5/6 of Millicent's old library capacity
    Prove that Millicent brings 2/3 of her books to the new home -/
theorem millicent_book_fraction (M : ℝ) (F : ℝ) (H : ℝ) :
  H = (1 / 2 : ℝ) * M →
  (1 / 3 : ℝ) * H + F * M = (5 / 6 : ℝ) * M →
  F = (2 / 3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_millicent_book_fraction_l3674_367425


namespace NUMINAMATH_CALUDE_fractional_linear_conjugacy_l3674_367410

/-- Given a fractional linear function f(x) = (ax + b) / (cx + d) where c ≠ 0 and ad ≠ bc,
    there exist functions φ and g such that f(x) = φ⁻¹(g(φ(x))). -/
theorem fractional_linear_conjugacy 
  {a b c d : ℝ} (hc : c ≠ 0) (had : a * d ≠ b * c) :
  ∃ (φ : ℝ → ℝ) (g : ℝ → ℝ),
    Function.Bijective φ ∧
    (∀ x, (a * x + b) / (c * x + d) = φ⁻¹ (g (φ x))) :=
by sorry

end NUMINAMATH_CALUDE_fractional_linear_conjugacy_l3674_367410


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l3674_367427

/-- Given a triangle ABC where C = π/3, b = √2, and c = √3, prove that angle A = 5π/12 -/
theorem triangle_angle_measure (A B C : ℝ) (a b c : ℝ) : 
  C = π/3 → b = Real.sqrt 2 → c = Real.sqrt 3 → 
  0 < a ∧ 0 < b ∧ 0 < c →
  a + b > c ∧ b + c > a ∧ c + a > b →
  A + B + C = π →
  A = 5*π/12 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l3674_367427


namespace NUMINAMATH_CALUDE_sally_picked_42_peaches_l3674_367413

/-- The number of peaches Sally picked -/
def peaches_picked (initial peaches_now : ℕ) : ℕ := peaches_now - initial

/-- Theorem stating that Sally picked 42 peaches -/
theorem sally_picked_42_peaches : peaches_picked 13 55 = 42 := by
  sorry

end NUMINAMATH_CALUDE_sally_picked_42_peaches_l3674_367413


namespace NUMINAMATH_CALUDE_triangle_problem_l3674_367445

/-- Given a triangle ABC with internal angles A, B, C and opposite sides a, b, c,
    vectors m and n, and the condition that b + c = a, prove that A = π/4 and sin(B + C) = √2/2 -/
theorem triangle_problem (A B C a b c : ℝ) (m n : ℝ × ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  m = (Real.sin A, 1) →
  n = (1, -Real.cos A) →
  m.1 * n.1 + m.2 * n.2 = 0 →
  b + c = a →
  (A = π/4 ∧ Real.sin (B + C) = Real.sqrt 2 / 2) := by sorry

end NUMINAMATH_CALUDE_triangle_problem_l3674_367445


namespace NUMINAMATH_CALUDE_oldest_sister_clothing_amount_l3674_367463

/-- Proves that the oldest sister's clothing amount is the difference between Nicole's final amount and the sum of the younger sisters' amounts. -/
theorem oldest_sister_clothing_amount 
  (nicole_initial : ℕ) 
  (nicole_final : ℕ) 
  (first_older_sister : ℕ) 
  (next_oldest_sister : ℕ) 
  (h1 : nicole_initial = 10)
  (h2 : first_older_sister = nicole_initial / 2)
  (h3 : next_oldest_sister = nicole_initial + 2)
  (h4 : nicole_final = 36) :
  nicole_final - (nicole_initial + first_older_sister + next_oldest_sister) = 9 := by
sorry

end NUMINAMATH_CALUDE_oldest_sister_clothing_amount_l3674_367463


namespace NUMINAMATH_CALUDE_exists_column_with_n_colors_l3674_367489

/-- Represents a color in the grid -/
structure Color where
  id : Nat

/-- Represents a cell in the grid -/
structure Cell where
  row : Nat
  col : Nat
  color : Color

/-- Represents the grid -/
structure Grid where
  size : Nat
  cells : List Cell

/-- Checks if a subgrid contains all n^2 colors -/
def containsAllColors (g : Grid) (n : Nat) (startRow : Nat) (startCol : Nat) : Prop :=
  ∀ c : Color, ∃ cell ∈ g.cells, 
    cell.row ≥ startRow ∧ cell.row < startRow + n ∧
    cell.col ≥ startCol ∧ cell.col < startCol + n ∧
    cell.color = c

/-- Checks if a row contains n distinct colors -/
def rowHasNColors (g : Grid) (n : Nat) (row : Nat) : Prop :=
  ∃ colors : List Color, colors.length = n ∧
    (∀ c ∈ colors, ∃ cell ∈ g.cells, cell.row = row ∧ cell.color = c) ∧
    (∀ cell ∈ g.cells, cell.row = row → cell.color ∈ colors)

/-- Checks if a column contains exactly n distinct colors -/
def columnHasExactlyNColors (g : Grid) (n : Nat) (col : Nat) : Prop :=
  ∃ colors : List Color, colors.length = n ∧
    (∀ c ∈ colors, ∃ cell ∈ g.cells, cell.col = col ∧ cell.color = c) ∧
    (∀ cell ∈ g.cells, cell.col = col → cell.color ∈ colors)

/-- The main theorem -/
theorem exists_column_with_n_colors (g : Grid) (n : Nat) :
  (∃ m : Nat, g.size = m * n) →
  (∀ i j : Nat, i < g.size - n + 1 → j < g.size - n + 1 → containsAllColors g n i j) →
  (∃ row : Nat, row < g.size ∧ rowHasNColors g n row) →
  (∃ col : Nat, col < g.size ∧ columnHasExactlyNColors g n col) :=
by sorry

end NUMINAMATH_CALUDE_exists_column_with_n_colors_l3674_367489


namespace NUMINAMATH_CALUDE_x_percent_of_z_l3674_367426

theorem x_percent_of_z (x y z : ℝ) (h1 : x = 1.30 * y) (h2 : y = 0.60 * z) : 
  x = 0.78 * z := by sorry

end NUMINAMATH_CALUDE_x_percent_of_z_l3674_367426


namespace NUMINAMATH_CALUDE_corporation_employees_l3674_367460

theorem corporation_employees (total : ℕ) (part_time : ℕ) (full_time : ℕ) :
  total = 65134 →
  part_time = 2041 →
  full_time = total - part_time →
  full_time = 63093 := by
sorry

end NUMINAMATH_CALUDE_corporation_employees_l3674_367460


namespace NUMINAMATH_CALUDE_sin_75_times_sin_15_l3674_367477

theorem sin_75_times_sin_15 :
  Real.sin (75 * π / 180) * Real.sin (15 * π / 180) = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_sin_75_times_sin_15_l3674_367477


namespace NUMINAMATH_CALUDE_refrigerator_savings_l3674_367496

/-- Calculates the savings when paying cash for a refrigerator instead of installments --/
theorem refrigerator_savings (cash_price deposit installment_amount : ℕ) (num_installments : ℕ) :
  cash_price = 8000 →
  deposit = 3000 →
  installment_amount = 300 →
  num_installments = 30 →
  deposit + num_installments * installment_amount - cash_price = 4000 := by
  sorry

end NUMINAMATH_CALUDE_refrigerator_savings_l3674_367496


namespace NUMINAMATH_CALUDE_cos_120_degrees_l3674_367414

theorem cos_120_degrees : Real.cos (2 * π / 3) = -(1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_cos_120_degrees_l3674_367414


namespace NUMINAMATH_CALUDE_jack_walking_time_l3674_367432

/-- Represents the walking parameters and time for a person -/
structure WalkingData where
  steps_per_minute : ℕ
  step_length : ℕ
  time_to_school : ℚ

/-- Calculates the distance walked based on walking data -/
def distance_walked (data : WalkingData) : ℚ :=
  (data.steps_per_minute : ℚ) * (data.step_length : ℚ) * data.time_to_school / 100

theorem jack_walking_time 
  (dave : WalkingData)
  (jack : WalkingData)
  (h1 : dave.steps_per_minute = 80)
  (h2 : dave.step_length = 80)
  (h3 : dave.time_to_school = 20)
  (h4 : jack.steps_per_minute = 120)
  (h5 : jack.step_length = 50)
  (h6 : distance_walked dave = distance_walked jack) :
  jack.time_to_school = 64/3 := by
  sorry

end NUMINAMATH_CALUDE_jack_walking_time_l3674_367432


namespace NUMINAMATH_CALUDE_triangle_cosine_problem_l3674_367419

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if b / cos B = c / cos C and cos A = 2/3, then cos B = √6 / 6 -/
theorem triangle_cosine_problem (a b c : ℝ) (A B C : ℝ) :
  b / Real.cos B = c / Real.cos C →
  Real.cos A = 2/3 →
  Real.cos B = Real.sqrt 6 / 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_problem_l3674_367419


namespace NUMINAMATH_CALUDE_gp_common_ratio_l3674_367458

/-- Given a geometric progression where the ratio of the sum of the first 6 terms
    to the sum of the first 3 terms is 217, prove that the common ratio is 6. -/
theorem gp_common_ratio (a r : ℝ) (h : r ≠ 1) :
  (a * (1 - r^6) / (1 - r)) / (a * (1 - r^3) / (1 - r)) = 217 →
  r = 6 := by
sorry

end NUMINAMATH_CALUDE_gp_common_ratio_l3674_367458


namespace NUMINAMATH_CALUDE_employee_payment_l3674_367471

theorem employee_payment (x y : ℝ) : 
  x + y = 770 →
  x = 1.2 * y →
  y = 350 := by
sorry

end NUMINAMATH_CALUDE_employee_payment_l3674_367471


namespace NUMINAMATH_CALUDE_gumball_probability_l3674_367476

/-- Given a jar with pink and blue gumballs, where the probability of drawing two blue
    gumballs in a row with replacement is 36/49, the probability of drawing a pink gumball
    is 1/7. -/
theorem gumball_probability (blue pink : ℝ) : 
  blue + pink = 1 →
  blue * blue = 36 / 49 →
  pink = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_gumball_probability_l3674_367476


namespace NUMINAMATH_CALUDE_calculation_proof_l3674_367429

theorem calculation_proof : (3127 - 2972)^3 / 343 = 125 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3674_367429


namespace NUMINAMATH_CALUDE_max_successful_teams_l3674_367420

/-- Represents a football tournament --/
structure Tournament :=
  (num_teams : ℕ)
  (points_for_win : ℕ)
  (points_for_draw : ℕ)
  (points_for_loss : ℕ)

/-- Calculate the maximum possible points for a single team in the tournament --/
def max_points (t : Tournament) : ℕ :=
  (t.num_teams - 1) * t.points_for_win

/-- Calculate the minimum points required for a team to be considered successful --/
def min_success_points (t : Tournament) : ℕ :=
  (max_points t + 1) / 2

/-- The main theorem stating the maximum number of successful teams --/
theorem max_successful_teams (t : Tournament) 
  (h1 : t.num_teams = 16)
  (h2 : t.points_for_win = 3)
  (h3 : t.points_for_draw = 1)
  (h4 : t.points_for_loss = 0) :
  ∃ (n : ℕ), n = 15 ∧ 
  (∀ (m : ℕ), m > n → 
    ¬ ∃ (points : List ℕ), 
      points.length = t.num_teams ∧
      points.sum = (t.num_teams * (t.num_teams - 1) / 2) * t.points_for_win ∧
      (points.filter (λ x => x ≥ min_success_points t)).length = m) :=
sorry

end NUMINAMATH_CALUDE_max_successful_teams_l3674_367420


namespace NUMINAMATH_CALUDE_detergent_in_altered_solution_l3674_367428

/-- Represents the ratio of bleach, detergent, and water in a solution -/
structure SolutionRatio where
  bleach : ℕ
  detergent : ℕ
  water : ℕ

/-- Calculates the new ratio after altering the solution -/
def alter_ratio (r : SolutionRatio) : SolutionRatio :=
  { bleach := 3 * r.bleach,
    detergent := r.detergent,
    water := 2 * r.water }

/-- Theorem: Given the conditions, the altered solution contains 60 liters of detergent -/
theorem detergent_in_altered_solution 
  (original_ratio : SolutionRatio)
  (h_original : original_ratio = ⟨2, 40, 100⟩)
  (h_water : (alter_ratio original_ratio).water = 300) :
  (alter_ratio original_ratio).detergent = 60 :=
sorry

end NUMINAMATH_CALUDE_detergent_in_altered_solution_l3674_367428


namespace NUMINAMATH_CALUDE_yellow_heavier_than_green_l3674_367467

/-- The weight difference between two blocks -/
def weight_difference (yellow_weight green_weight : Real) : Real :=
  yellow_weight - green_weight

/-- Theorem: The yellow block weighs 0.2 pounds more than the green block -/
theorem yellow_heavier_than_green :
  let yellow_weight : Real := 0.6
  let green_weight : Real := 0.4
  weight_difference yellow_weight green_weight = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_yellow_heavier_than_green_l3674_367467


namespace NUMINAMATH_CALUDE_specialPrimes_eq_l3674_367499

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

/-- A function that checks if a number has all digits 0 to b-1 exactly once in base b -/
def hasAllDigitsOnce (n : ℕ) (b : ℕ) : Prop :=
  ∃ (digits : List ℕ), 
    (digits.length = b) ∧
    (∀ d, d ∈ digits → d < b) ∧
    (digits.toFinset = Finset.range b) ∧
    (n = digits.foldr (λ d acc => acc * b + d) 0)

/-- The set of prime numbers with the special digit property -/
def specialPrimes : Set ℕ :=
  {p | ∃ b : ℕ, isPrime p ∧ b > 1 ∧ hasAllDigitsOnce p b}

/-- The theorem stating that the set of special primes is equal to {2, 5, 7, 11, 19} -/
theorem specialPrimes_eq : specialPrimes = {2, 5, 7, 11, 19} := by sorry

end NUMINAMATH_CALUDE_specialPrimes_eq_l3674_367499


namespace NUMINAMATH_CALUDE_clock_hand_overlaps_in_day_l3674_367423

/-- Represents the number of overlaps between clock hands in a given time period -/
def clockHandOverlaps (hourRotations minuteRotations : ℕ) : ℕ :=
  minuteRotations - hourRotations

theorem clock_hand_overlaps_in_day :
  clockHandOverlaps 2 24 = 22 := by
  sorry

#eval clockHandOverlaps 2 24

end NUMINAMATH_CALUDE_clock_hand_overlaps_in_day_l3674_367423


namespace NUMINAMATH_CALUDE_problem_2013_l3674_367459

theorem problem_2013 : 
  (2013^3 - 2 * 2013^2 * 2014 + 3 * 2013 * 2014^2 - 2014^3 + 1) / (2013 * 2014) = 2013 := by
  sorry

end NUMINAMATH_CALUDE_problem_2013_l3674_367459


namespace NUMINAMATH_CALUDE_league_members_count_l3674_367448

/-- Represents the cost of items and total expenditure in the Rockham Soccer League --/
structure LeagueCosts where
  sock_cost : ℕ
  tshirt_cost_difference : ℕ
  set_discount : ℕ
  total_expenditure : ℕ

/-- Calculates the number of members in the Rockham Soccer League --/
def calculate_members (costs : LeagueCosts) : ℕ :=
  sorry

/-- Theorem stating that the number of members in the league is 150 --/
theorem league_members_count (costs : LeagueCosts)
  (h1 : costs.sock_cost = 5)
  (h2 : costs.tshirt_cost_difference = 6)
  (h3 : costs.set_discount = 3)
  (h4 : costs.total_expenditure = 3100) :
  calculate_members costs = 150 :=
sorry

end NUMINAMATH_CALUDE_league_members_count_l3674_367448


namespace NUMINAMATH_CALUDE_function_extrema_l3674_367403

/-- The function f(x) = 1 + 3x - x³ has a minimum value of -1 and a maximum value of 3. -/
theorem function_extrema :
  ∃ (a b : ℝ), (∀ x : ℝ, 1 + 3 * x - x^3 ≥ a) ∧
                (∃ x : ℝ, 1 + 3 * x - x^3 = a) ∧
                (∀ x : ℝ, 1 + 3 * x - x^3 ≤ b) ∧
                (∃ x : ℝ, 1 + 3 * x - x^3 = b) ∧
                a = -1 ∧ b = 3 := by
  sorry

end NUMINAMATH_CALUDE_function_extrema_l3674_367403


namespace NUMINAMATH_CALUDE_weekly_wage_problem_l3674_367408

/-- The weekly wage problem -/
theorem weekly_wage_problem (Rm Hm Rn Hn : ℝ) 
  (h1 : Rm * Hm + Rn * Hn = 770)
  (h2 : Rm * Hm = 1.3 * (Rn * Hn)) :
  Rn * Hn = 335 := by
  sorry

end NUMINAMATH_CALUDE_weekly_wage_problem_l3674_367408


namespace NUMINAMATH_CALUDE_train_length_l3674_367473

/-- The length of a train given its speed, the speed of a person running in the opposite direction, and the time it takes for the train to pass the person. -/
theorem train_length (train_speed : ℝ) (person_speed : ℝ) (passing_time : ℝ) :
  train_speed = 50 →
  person_speed = 5 →
  passing_time = 7.2 →
  ∃ (train_length : ℝ), 
    (train_length ≥ 109.9 ∧ train_length ≤ 110.1) ∧
    train_length = (train_speed + person_speed) * passing_time * (1000 / 3600) :=
by sorry


end NUMINAMATH_CALUDE_train_length_l3674_367473


namespace NUMINAMATH_CALUDE_investment_percentage_l3674_367454

/-- Given two investors with equal initial investments, where one investor's value quadruples and ends up with $1900 more than the other, prove that the other investor's final value is 20% of their initial investment. -/
theorem investment_percentage (initial_investment : ℝ) (jackson_final : ℝ) (brandon_final : ℝ) :
  initial_investment > 0 →
  jackson_final = 4 * initial_investment →
  jackson_final - brandon_final = 1900 →
  brandon_final / initial_investment = 0.2 := by
sorry

end NUMINAMATH_CALUDE_investment_percentage_l3674_367454


namespace NUMINAMATH_CALUDE_max_of_min_values_l3674_367491

/-- The function f(x) for a given m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*m*x + 8*m + 4

/-- The minimum value of f(x) for a given m -/
def min_value (m : ℝ) : ℝ := f m m

/-- The function representing all minimum values of f(x) for different m -/
def g (m : ℝ) : ℝ := -m^2 + 8*m + 4

/-- The maximum of all minimum values of f(x) -/
theorem max_of_min_values :
  (⨆ (m : ℝ), min_value m) = 20 :=
sorry

end NUMINAMATH_CALUDE_max_of_min_values_l3674_367491


namespace NUMINAMATH_CALUDE_cost_per_dozen_is_240_l3674_367497

/-- Calculates the cost per dozen donuts given the total number of donuts,
    selling price per donut, desired profit, and total number of dozens. -/
def cost_per_dozen (total_donuts : ℕ) (price_per_donut : ℚ) (desired_profit : ℚ) (total_dozens : ℕ) : ℚ :=
  let total_sales := total_donuts * price_per_donut
  let total_cost := total_sales - desired_profit
  total_cost / total_dozens

/-- Proves that the cost per dozen donuts is $2.40 given the specified conditions. -/
theorem cost_per_dozen_is_240 :
  cost_per_dozen 120 1 96 10 = 240 / 100 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_dozen_is_240_l3674_367497


namespace NUMINAMATH_CALUDE_textbook_weight_difference_l3674_367483

/-- The weight difference between Kelly's chemistry and geometry textbooks -/
theorem textbook_weight_difference :
  let chemistry_weight : ℚ := 712 / 100
  let geometry_weight : ℚ := 62 / 100
  chemistry_weight - geometry_weight = 650 / 100 :=
by sorry

end NUMINAMATH_CALUDE_textbook_weight_difference_l3674_367483


namespace NUMINAMATH_CALUDE_histogram_frequency_l3674_367415

theorem histogram_frequency (m : ℕ) (S1 S2 S3 : ℚ) :
  m ≥ 3 →
  S1 + S2 + S3 = (1 : ℚ) / 4 * (1 - (S1 + S2 + S3)) →
  S2 - S1 = S3 - S2 →
  S1 = (1 : ℚ) / 20 →
  (120 : ℚ) * S3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_histogram_frequency_l3674_367415


namespace NUMINAMATH_CALUDE_sqrt_of_square_positive_l3674_367479

theorem sqrt_of_square_positive (a : ℝ) (h : a > 0) : Real.sqrt (a^2) = a := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_square_positive_l3674_367479


namespace NUMINAMATH_CALUDE_g_properties_l3674_367482

/-- Given a function f(x) = a - b cos(x) with maximum value 5/2 and minimum value -1/2,
    we define g(x) = -4a sin(bx) and prove its properties. -/
theorem g_properties (a b : ℝ) (f : ℝ → ℝ) (g : ℝ → ℝ)
  (hf : f = fun x ↦ a - b * Real.cos x)
  (hmax : ∀ x, f x ≤ 5/2)
  (hmin : ∀ x, -1/2 ≤ f x)
  (hg : g = fun x ↦ -4 * a * Real.sin (b * x)) :
  (∃ x, g x = 4) ∧
  (∃ x, g x = -4) ∧
  (∃ T > 0, ∀ x, g (x + T) = g x ∧ ∀ S, 0 < S → S < T → ∃ y, g (y + S) ≠ g y) ∧
  (∀ x, -4 ≤ g x ∧ g x ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_g_properties_l3674_367482


namespace NUMINAMATH_CALUDE_eugene_pencils_eugene_final_pencils_l3674_367464

theorem eugene_pencils (initial_pencils : ℕ) (received_pencils : ℕ) 
  (pack_size : ℕ) (num_friends : ℕ) (given_away : ℕ) : ℕ :=
  let total_after_receiving := initial_pencils + received_pencils
  let total_in_packs := pack_size * (num_friends + 1)
  let total_before_giving := total_after_receiving + total_in_packs
  total_before_giving - given_away

theorem eugene_final_pencils :
  eugene_pencils 51 6 12 3 8 = 97 := by
  sorry

end NUMINAMATH_CALUDE_eugene_pencils_eugene_final_pencils_l3674_367464


namespace NUMINAMATH_CALUDE_unique_room_setup_l3674_367436

/-- Represents the number of people, stools, and chairs in a room -/
structure RoomSetup where
  people : ℕ
  stools : ℕ
  chairs : ℕ

/-- Checks if a given room setup satisfies all conditions -/
def isValidSetup (setup : RoomSetup) : Prop :=
  2 * setup.people + 3 * setup.stools + 4 * setup.chairs = 32 ∧
  setup.people > setup.stools ∧
  setup.people > setup.chairs ∧
  setup.people < setup.stools + setup.chairs

/-- The theorem stating that there is only one valid room setup -/
theorem unique_room_setup :
  ∃! setup : RoomSetup, isValidSetup setup ∧ 
    setup.people = 5 ∧ setup.stools = 2 ∧ setup.chairs = 4 := by
  sorry


end NUMINAMATH_CALUDE_unique_room_setup_l3674_367436


namespace NUMINAMATH_CALUDE_vector_magnitude_l3674_367452

theorem vector_magnitude (a b : ℝ × ℝ) :
  a = (2, 1) →
  a.1 * b.1 + a.2 * b.2 = 10 →
  (a.1 + b.1)^2 + (a.2 + b.2)^2 = 50 →
  b.1^2 + b.2^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l3674_367452


namespace NUMINAMATH_CALUDE_wizard_elixir_combinations_l3674_367418

/-- Represents the number of magical herbs available. -/
def num_herbs : ℕ := 4

/-- Represents the number of enchanted stones available. -/
def num_stones : ℕ := 6

/-- Represents the number of herbs that are incompatible with one specific stone. -/
def incompatible_herbs : ℕ := 3

/-- Represents the number of valid combinations for the wizard's elixir. -/
def valid_combinations : ℕ := num_herbs * num_stones - incompatible_herbs

theorem wizard_elixir_combinations :
  valid_combinations = 21 :=
by sorry

end NUMINAMATH_CALUDE_wizard_elixir_combinations_l3674_367418


namespace NUMINAMATH_CALUDE_work_completion_time_l3674_367412

/-- Represents the time it takes for worker B to complete the work alone -/
def time_B_alone : ℝ := 10

/-- Represents the time it takes for worker A to complete the work alone -/
def time_A_alone : ℝ := 4

/-- Represents the time A and B work together -/
def time_together : ℝ := 2

/-- Represents the time B works alone after A leaves -/
def time_B_after_A : ℝ := 3.0000000000000004

/-- Theorem stating that given the conditions, B can finish the work alone in 10 days -/
theorem work_completion_time :
  time_B_alone = 10 :=
sorry

end NUMINAMATH_CALUDE_work_completion_time_l3674_367412


namespace NUMINAMATH_CALUDE_golf_strokes_over_par_l3674_367468

/-- Calculates the number of strokes over par in a golf game. -/
def strokes_over_par (rounds : ℕ) (avg_strokes_per_hole : ℕ) (par_per_hole : ℕ) : ℕ :=
  let total_holes := rounds * 18
  let total_strokes := avg_strokes_per_hole * total_holes
  let total_par := par_per_hole * total_holes
  total_strokes - total_par

/-- Proves that given 9 rounds of golf, an average of 4 strokes per hole, 
    and a par value of 3 per hole, the number of strokes over par is 162. -/
theorem golf_strokes_over_par :
  strokes_over_par 9 4 3 = 162 := by
  sorry

end NUMINAMATH_CALUDE_golf_strokes_over_par_l3674_367468


namespace NUMINAMATH_CALUDE_sean_needs_six_packs_l3674_367435

def bedroom_bulbs : ℕ := 2
def bathroom_bulbs : ℕ := 1
def kitchen_bulbs : ℕ := 1
def basement_bulbs : ℕ := 4
def bulbs_per_pack : ℕ := 2

def total_non_garage_bulbs : ℕ := bedroom_bulbs + bathroom_bulbs + kitchen_bulbs + basement_bulbs

def garage_bulbs : ℕ := total_non_garage_bulbs / 2

def total_bulbs : ℕ := total_non_garage_bulbs + garage_bulbs

theorem sean_needs_six_packs : (total_bulbs + bulbs_per_pack - 1) / bulbs_per_pack = 6 := by
  sorry

end NUMINAMATH_CALUDE_sean_needs_six_packs_l3674_367435


namespace NUMINAMATH_CALUDE_unique_solution_logarithmic_equation_l3674_367421

theorem unique_solution_logarithmic_equation :
  ∃! (x : ℝ), x > 0 ∧ x^(Real.log x / Real.log 10) = x^4 / 10000 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_logarithmic_equation_l3674_367421


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l3674_367444

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 6) :
  (1 / x + 1 / y) ≥ 2 / 3 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 6 ∧ 1 / x + 1 / y = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l3674_367444


namespace NUMINAMATH_CALUDE_parabola_line_intersection_triangle_area_l3674_367486

/-- Given a parabola y = x^2 + 2 and a line y = r, if the triangle formed by the vertex of the parabola
    and the two intersections of the line and parabola has an area A such that 10 ≤ A ≤ 50,
    then 10^(2/3) + 2 ≤ r ≤ 50^(2/3) + 2. -/
theorem parabola_line_intersection_triangle_area (r : ℝ) : 
  let parabola := fun x : ℝ => x^2 + 2
  let line := fun _ : ℝ => r
  let vertex := (0, parabola 0)
  let intersections := {x : ℝ | parabola x = line x}
  let triangle_area := (r - 2)^(3/2) / 2
  10 ≤ triangle_area ∧ triangle_area ≤ 50 → 10^(2/3) + 2 ≤ r ∧ r ≤ 50^(2/3) + 2 := by
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_triangle_area_l3674_367486


namespace NUMINAMATH_CALUDE_fraction_multiplication_l3674_367434

theorem fraction_multiplication (a b x : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hx : x ≠ 0) :
  (3 * a * b) / x * (2 * x^2) / (9 * a * b^2) = (2 * x) / (3 * b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l3674_367434


namespace NUMINAMATH_CALUDE_polynomial_equality_l3674_367461

theorem polynomial_equality (a b : ℝ) : 
  (∀ x : ℝ, (x + a) * (x - 3) = x^2 + 2*x - b) → a - b = -10 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l3674_367461


namespace NUMINAMATH_CALUDE_parabola_vertex_l3674_367407

/-- The vertex of the parabola y = (x+2)^2 - 1 is at the point (-2, -1) -/
theorem parabola_vertex (x y : ℝ) : 
  y = (x + 2)^2 - 1 → (∀ x' y', y' = (x' + 2)^2 - 1 → y ≤ y') → x = -2 ∧ y = -1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3674_367407


namespace NUMINAMATH_CALUDE_factorization_theorem_1_factorization_theorem_2_l3674_367462

-- Theorem 1
theorem factorization_theorem_1 (m n : ℝ) : m^3*n - 9*m*n = m*n*(m+3)*(m-3) := by
  sorry

-- Theorem 2
theorem factorization_theorem_2 (a : ℝ) : a^3 + a - 2*a^2 = a*(a-1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_theorem_1_factorization_theorem_2_l3674_367462


namespace NUMINAMATH_CALUDE_intersection_line_passes_through_circles_l3674_367484

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4*x
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 4*y = 0

-- Define the line
def line (x y : ℝ) : Prop := y = -x

-- Theorem statement
theorem intersection_line_passes_through_circles :
  ∀ x y : ℝ, circle1 x y ∧ circle2 x y → line x y :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_passes_through_circles_l3674_367484


namespace NUMINAMATH_CALUDE_road_trip_mileage_l3674_367443

/-- Calculates the final mileage of a car after a road trip -/
def final_mileage (initial_mileage : ℕ) (efficiency : ℕ) (tank_capacity : ℕ) (refills : ℕ) : ℕ :=
  initial_mileage + efficiency * tank_capacity * refills

theorem road_trip_mileage :
  final_mileage 1728 30 20 2 = 2928 := by
  sorry

end NUMINAMATH_CALUDE_road_trip_mileage_l3674_367443


namespace NUMINAMATH_CALUDE_don_buys_from_shop_B_l3674_367470

/-- The number of bottles Don buys from Shop A -/
def bottlesFromA : ℕ := 150

/-- The number of bottles Don buys from Shop C -/
def bottlesFromC : ℕ := 220

/-- The total number of bottles Don buys -/
def totalBottles : ℕ := 550

/-- The number of bottles Don buys from Shop B -/
def bottlesFromB : ℕ := totalBottles - (bottlesFromA + bottlesFromC)

theorem don_buys_from_shop_B : bottlesFromB = 180 := by
  sorry

end NUMINAMATH_CALUDE_don_buys_from_shop_B_l3674_367470


namespace NUMINAMATH_CALUDE_inequality_implies_linear_form_l3674_367481

/-- A function satisfying the given inequality for all real x and y -/
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) + y ≤ f (f (f x))

/-- The theorem stating that any function satisfying the inequality must be of the form f(x) = C - x -/
theorem inequality_implies_linear_form {f : ℝ → ℝ} (h : SatisfiesInequality f) :
  ∃ C : ℝ, ∀ x : ℝ, f x = C - x :=
sorry

end NUMINAMATH_CALUDE_inequality_implies_linear_form_l3674_367481


namespace NUMINAMATH_CALUDE_vasya_fool_count_l3674_367485

theorem vasya_fool_count (misha petya kolya vasya : ℕ) : 
  misha + petya + kolya + vasya = 16 →
  misha ≥ 1 → petya ≥ 1 → kolya ≥ 1 → vasya ≥ 1 →
  petya + kolya = 9 →
  misha > petya → misha > kolya → misha > vasya →
  vasya = 1 := by
sorry

end NUMINAMATH_CALUDE_vasya_fool_count_l3674_367485


namespace NUMINAMATH_CALUDE_vector_subtraction_l3674_367442

/-- Given two vectors a and b in ℝ², prove that their difference is equal to a specific vector. -/
theorem vector_subtraction (a b : ℝ × ℝ) (h1 : a = (1, 2)) (h2 : b = (3, 1)) :
  b - a = (2, -1) := by sorry

end NUMINAMATH_CALUDE_vector_subtraction_l3674_367442


namespace NUMINAMATH_CALUDE_complement_of_union_l3674_367457

-- Define the universe set U
def U : Finset Nat := {0, 1, 2, 3, 4}

-- Define set M
def M : Finset Nat := {0, 4}

-- Define set N
def N : Finset Nat := {2, 4}

-- Theorem statement
theorem complement_of_union :
  (U \ (M ∪ N)) = {1, 3} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l3674_367457


namespace NUMINAMATH_CALUDE_problem_solution_l3674_367416

theorem problem_solution : ∃ x : ℚ, (x + x / 4 = 80 - 80 / 4) ∧ x = 48 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3674_367416


namespace NUMINAMATH_CALUDE_difference_of_squares_l3674_367466

theorem difference_of_squares (m : ℝ) : m^2 - 16 = (m + 4) * (m - 4) := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3674_367466


namespace NUMINAMATH_CALUDE_angle_R_measure_l3674_367400

/-- A hexagon with specific angle properties -/
structure Hexagon where
  F : ℝ  -- Angle F in degrees
  I : ℝ  -- Angle I in degrees
  G : ℝ  -- Angle G in degrees
  U : ℝ  -- Angle U in degrees
  R : ℝ  -- Angle R in degrees
  E : ℝ  -- Angle E in degrees
  sum_angles : F + I + G + U + R + E = 720  -- Sum of angles in a hexagon
  supplementary : G + U = 180  -- G and U are supplementary
  congruent : F = I ∧ I = R ∧ R = E  -- F, I, R, and E are congruent

/-- The measure of angle R in the specific hexagon is 135 degrees -/
theorem angle_R_measure (h : Hexagon) : h.R = 135 := by
  sorry

end NUMINAMATH_CALUDE_angle_R_measure_l3674_367400


namespace NUMINAMATH_CALUDE_total_toy_cost_l3674_367453

def football_cost : ℝ := 5.71
def marbles_cost : ℝ := 6.59

theorem total_toy_cost : football_cost + marbles_cost = 12.30 := by
  sorry

end NUMINAMATH_CALUDE_total_toy_cost_l3674_367453


namespace NUMINAMATH_CALUDE_expression_equality_l3674_367433

theorem expression_equality : (π - 1) ^ 0 + 4 * Real.sin (π / 4) - Real.sqrt 8 + abs (-3) = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3674_367433


namespace NUMINAMATH_CALUDE_purse_value_is_107_percent_of_dollar_l3674_367417

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "penny" => 1
  | "nickel" => 5
  | "dime" => 10
  | "quarter" => 25
  | _ => 0

/-- Calculates the total value of coins in cents -/
def total_value (pennies nickels dimes quarters : ℕ) : ℕ :=
  pennies * coin_value "penny" +
  nickels * coin_value "nickel" +
  dimes * coin_value "dime" +
  quarters * coin_value "quarter"

theorem purse_value_is_107_percent_of_dollar (pennies nickels dimes quarters : ℕ) 
  (h_pennies : pennies = 2)
  (h_nickels : nickels = 3)
  (h_dimes : dimes = 4)
  (h_quarters : quarters = 2) :
  (total_value pennies nickels dimes quarters : ℚ) / 100 = 107 / 100 := by
  sorry

end NUMINAMATH_CALUDE_purse_value_is_107_percent_of_dollar_l3674_367417


namespace NUMINAMATH_CALUDE_cube_volume_ratio_l3674_367492

theorem cube_volume_ratio (e : ℝ) (h : e > 0) :
  let small_cube_volume := e^3
  let large_cube_volume := (4*e)^3
  large_cube_volume / small_cube_volume = 64 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_ratio_l3674_367492


namespace NUMINAMATH_CALUDE_prob_selected_twice_correct_most_likely_selected_correct_l3674_367438

/-- Represents the total number of students --/
def total_students : ℕ := 60

/-- Represents the number of students selected in each round --/
def selected_per_round : ℕ := 45

/-- Probability of a student being selected in both rounds --/
def prob_selected_twice : ℚ := 9 / 16

/-- The most likely number of students selected in both rounds --/
def most_likely_selected : ℕ := 34

/-- Function to calculate the probability of being selected in both rounds --/
def calculate_prob_selected_twice : ℚ :=
  (Nat.choose (total_students - 1) (selected_per_round - 1) / Nat.choose total_students selected_per_round) ^ 2

/-- Function to calculate the probability of exactly n students being selected in both rounds --/
def prob_n_selected (n : ℕ) : ℚ :=
  (Nat.choose total_students n * Nat.choose (total_students - n) (selected_per_round - n) * Nat.choose (selected_per_round - n) (selected_per_round - n)) /
  (Nat.choose total_students selected_per_round * Nat.choose total_students selected_per_round)

theorem prob_selected_twice_correct :
  calculate_prob_selected_twice = prob_selected_twice :=
sorry

theorem most_likely_selected_correct :
  ∀ n, 30 ≤ n ∧ n ≤ 45 → prob_n_selected most_likely_selected ≥ prob_n_selected n :=
sorry

end NUMINAMATH_CALUDE_prob_selected_twice_correct_most_likely_selected_correct_l3674_367438


namespace NUMINAMATH_CALUDE_oplus_problem_l3674_367405

-- Define the operation ⊕
def oplus (a b : ℝ) (x y : ℝ) : ℝ := a * x + b * y - a * b

-- State the theorem
theorem oplus_problem (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h1 : oplus a b 1 2 = 3) (h2 : oplus a b 2 3 = 6) :
  oplus a b 3 4 = 9 := by sorry

end NUMINAMATH_CALUDE_oplus_problem_l3674_367405


namespace NUMINAMATH_CALUDE_probability_cpp_l3674_367495

/-- The probability of an employee knowing C++ in a software company -/
theorem probability_cpp (total_employees : ℕ) (cpp_fraction : ℚ) : 
  total_employees = 600 →
  cpp_fraction = 3 / 10 →
  (cpp_fraction * total_employees) / total_employees = 3 / 10 :=
by sorry

end NUMINAMATH_CALUDE_probability_cpp_l3674_367495


namespace NUMINAMATH_CALUDE_inequality_proof_l3674_367490

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  1/a + 1/b + 4/c + 16/d ≥ 64/(a + b + c + d) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3674_367490


namespace NUMINAMATH_CALUDE_tiles_remaining_l3674_367488

theorem tiles_remaining (initial_tiles : ℕ) : 
  initial_tiles = 2022 → 
  (initial_tiles - initial_tiles / 6 - (initial_tiles - initial_tiles / 6) / 5 - 
   (initial_tiles - initial_tiles / 6 - (initial_tiles - initial_tiles / 6) / 5) / 4) = 1011 := by
  sorry

end NUMINAMATH_CALUDE_tiles_remaining_l3674_367488


namespace NUMINAMATH_CALUDE_root_implies_m_value_l3674_367446

theorem root_implies_m_value (m : ℝ) : 
  (∃ x : ℝ, x^2 - m*x - 3 = 0 ∧ x = 3) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_m_value_l3674_367446


namespace NUMINAMATH_CALUDE_vector_problem_l3674_367441

/-- Two vectors are non-collinear if they are not scalar multiples of each other -/
def NonCollinear (a b : ℝ × ℝ) : Prop :=
  ¬∃ (k : ℝ), b.1 = k * a.1 ∧ b.2 = k * a.2

theorem vector_problem (a b c : ℝ × ℝ) (x : ℝ) :
  NonCollinear a b →
  a = (1, 2) →
  b = (x, 6) →
  ‖a - b‖ = 2 * Real.sqrt 5 →
  c = (2 • a) + b →
  c = (1, 10) := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l3674_367441


namespace NUMINAMATH_CALUDE_max_value_of_a_minus_b_squared_l3674_367409

theorem max_value_of_a_minus_b_squared (a b : ℝ) (h : a^2 + b^2 = 4) :
  (∀ x y : ℝ, x^2 + y^2 = 4 → (x - y)^2 ≤ 8) ∧ 
  (∃ x y : ℝ, x^2 + y^2 = 4 ∧ (x - y)^2 = 8) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_minus_b_squared_l3674_367409


namespace NUMINAMATH_CALUDE_smallest_v_in_consecutive_cubes_sum_l3674_367430

theorem smallest_v_in_consecutive_cubes_sum (w x y u v : ℕ) :
  w < x ∧ x < y ∧ y < u ∧ u < v →
  (∃ a, w = a^3) ∧ (∃ b, x = b^3) ∧ (∃ c, y = c^3) ∧ (∃ d, u = d^3) ∧ (∃ e, v = e^3) →
  w^3 + x^3 + y^3 + u^3 = v^3 →
  v ≥ 6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_v_in_consecutive_cubes_sum_l3674_367430


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_not_necessary_condition_l3674_367401

def is_arithmetic_sequence (a b c d : ℝ) : Prop :=
  b - a = c - b ∧ c - b = d - c

theorem arithmetic_sequence_sum_property :
  ∀ (a b c d : ℝ), is_arithmetic_sequence a b c d → a + d = b + c :=
sorry

theorem not_necessary_condition :
  ∃ (a b c d : ℝ), a + d = b + c ∧ ¬(is_arithmetic_sequence a b c d) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_not_necessary_condition_l3674_367401


namespace NUMINAMATH_CALUDE_product_difference_sum_l3674_367478

theorem product_difference_sum (P Q R S : ℕ+) : 
  P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ Q ≠ R ∧ Q ≠ S ∧ R ≠ S →
  P * Q = 120 →
  R * S = 120 →
  P - Q = R + S →
  P = 30 := by
sorry

end NUMINAMATH_CALUDE_product_difference_sum_l3674_367478


namespace NUMINAMATH_CALUDE_toms_average_score_l3674_367404

theorem toms_average_score (subjects_sem1 subjects_sem2 : ℕ)
  (avg_score_sem1 avg_score_5_sem2 avg_score_all : ℚ) :
  subjects_sem1 = 3 →
  subjects_sem2 = 7 →
  avg_score_sem1 = 85 →
  avg_score_5_sem2 = 78 →
  avg_score_all = 80 →
  (subjects_sem1 * avg_score_sem1 + 5 * avg_score_5_sem2 + 2 * ((subjects_sem1 + subjects_sem2) * avg_score_all - subjects_sem1 * avg_score_sem1 - 5 * avg_score_5_sem2) / 2) / (subjects_sem1 + subjects_sem2) = avg_score_all :=
by sorry

end NUMINAMATH_CALUDE_toms_average_score_l3674_367404


namespace NUMINAMATH_CALUDE_laptop_price_l3674_367449

theorem laptop_price (upfront_percentage : ℚ) (upfront_payment : ℚ) 
  (h1 : upfront_percentage = 20 / 100)
  (h2 : upfront_payment = 240) : 
  upfront_payment / upfront_percentage = 1200 := by
sorry

end NUMINAMATH_CALUDE_laptop_price_l3674_367449


namespace NUMINAMATH_CALUDE_intersection_A_B_l3674_367424

def A : Set ℝ := {x | -1 < x ∧ x ≤ 1}
def B : Set ℝ := {-1, 0, 1}

theorem intersection_A_B : A ∩ B = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3674_367424


namespace NUMINAMATH_CALUDE_complex_simplification_and_multiplication_l3674_367469

theorem complex_simplification_and_multiplication :
  ((-5 + 3 * Complex.I) - (2 - 7 * Complex.I)) * (2 * Complex.I) = -20 - 14 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_simplification_and_multiplication_l3674_367469


namespace NUMINAMATH_CALUDE_max_value_g_geq_seven_l3674_367455

theorem max_value_g_geq_seven (a b : ℝ) (h_a : a ≤ -1) : 
  let f := fun x : ℝ => Real.exp x * (x^2 + a*x + 1)
  let g := fun x : ℝ => 2*x^3 + 3*(b+1)*x^2 + 6*b*x + 6
  let x_min_f := -(a + 1)
  (∀ x : ℝ, g x ≥ g x_min_f) → 
  (∀ x : ℝ, f x ≥ f x_min_f) → 
  ∃ x : ℝ, g x ≥ 7 :=
by sorry

end NUMINAMATH_CALUDE_max_value_g_geq_seven_l3674_367455


namespace NUMINAMATH_CALUDE_max_value_of_f_l3674_367411

/-- The domain of the function f -/
def Domain (c d e : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 > 0 ∧ p.2 > 0 ∧ c - d * p.1 - e * p.2 > 0}

/-- The function f -/
def f (a b c d e : ℝ) (p : ℝ × ℝ) : ℝ :=
  a * p.1 * b * p.2 * (c - d * p.1 - e * p.2)

/-- Theorem stating the maximum value of f -/
theorem max_value_of_f (a b c d e : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) :
  ∃ M : ℝ, M = (a / d) * (b / e) * (c / 3)^3 ∧
  ∀ p ∈ Domain c d e, f a b c d e p ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3674_367411


namespace NUMINAMATH_CALUDE_punger_baseball_cards_l3674_367493

/-- Given the number of packs, cards per pack, and cards per page, 
    calculate the number of pages needed to store all cards. -/
def pages_needed (packs : ℕ) (cards_per_pack : ℕ) (cards_per_page : ℕ) : ℕ :=
  (packs * cards_per_pack + cards_per_page - 1) / cards_per_page

theorem punger_baseball_cards : 
  pages_needed 60 7 10 = 42 := by
  sorry

end NUMINAMATH_CALUDE_punger_baseball_cards_l3674_367493


namespace NUMINAMATH_CALUDE_stock_market_value_l3674_367451

/-- Prove that for a stock with an 8% dividend rate and a 20% yield, the market value is 40% of the face value. -/
theorem stock_market_value (face_value : ℝ) (dividend_rate : ℝ) (yield : ℝ) :
  dividend_rate = 0.08 →
  yield = 0.20 →
  (dividend_rate * face_value) / yield = 0.40 * face_value :=
by sorry

end NUMINAMATH_CALUDE_stock_market_value_l3674_367451


namespace NUMINAMATH_CALUDE_range_of_m_value_of_m_l3674_367456

-- Define the quadratic equation
def quadratic_eq (m x : ℝ) : Prop :=
  x^2 - 2*(1-m)*x + m^2 = 0

-- Define the roots of the equation
def roots (m x₁ x₂ : ℝ) : Prop :=
  quadratic_eq m x₁ ∧ quadratic_eq m x₂ ∧ x₁ ≠ x₂

-- Define the additional condition
def additional_condition (m x₁ x₂ : ℝ) : Prop :=
  x₁^2 + 12*m + x₂^2 = 10

-- Theorem for the range of m
theorem range_of_m (m : ℝ) :
  (∃ x₁ x₂, roots m x₁ x₂) → m ≤ 1/2 :=
sorry

-- Theorem for the value of m given the additional condition
theorem value_of_m (m x₁ x₂ : ℝ) :
  roots m x₁ x₂ → additional_condition m x₁ x₂ → m = -3 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_value_of_m_l3674_367456


namespace NUMINAMATH_CALUDE_trinomial_cube_l3674_367440

theorem trinomial_cube (x : ℝ) : 
  (x^2 - 2*x + 1)^3 = x^6 - 6*x^5 + 15*x^4 - 20*x^3 + 15*x^2 - 6*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_trinomial_cube_l3674_367440


namespace NUMINAMATH_CALUDE_discount_percentage_calculation_l3674_367465

theorem discount_percentage_calculation (washing_machine_cost dryer_cost total_paid : ℚ) : 
  washing_machine_cost = 100 →
  dryer_cost = washing_machine_cost - 30 →
  total_paid = 153 →
  (washing_machine_cost + dryer_cost - total_paid) / (washing_machine_cost + dryer_cost) * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_discount_percentage_calculation_l3674_367465


namespace NUMINAMATH_CALUDE_square_area_from_adjacent_vertices_l3674_367450

/-- The area of a square with adjacent vertices at (1, -2) and (-3, 5) is 65 -/
theorem square_area_from_adjacent_vertices : 
  let p1 : ℝ × ℝ := (1, -2)
  let p2 : ℝ × ℝ := (-3, 5)
  let side_length := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  side_length^2 = 65 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_adjacent_vertices_l3674_367450


namespace NUMINAMATH_CALUDE_problem_statements_l3674_367422

theorem problem_statements :
  (∀ x : ℝ, (x^2 - 4*x + 3 = 0 → x = 3) ↔ (x ≠ 3 → x^2 - 4*x + 3 ≠ 0)) ∧
  (¬(∀ x : ℝ, x^2 - x + 2 > 0) ↔ (∃ x : ℝ, x^2 - x + 2 ≤ 0)) ∧
  (∀ p q : Prop, (p ∧ q) → (p ∧ q)) ∧
  (∀ x : ℝ, x > -1 → x^2 + 4*x + 3 > 0) ∧
  (∃ x : ℝ, x^2 + 4*x + 3 > 0 ∧ x ≤ -1) :=
by sorry

end NUMINAMATH_CALUDE_problem_statements_l3674_367422


namespace NUMINAMATH_CALUDE_sum_of_factors_l3674_367474

theorem sum_of_factors (y : ℝ) (p q r s t : ℝ) : 
  512 * y^3 + 27 = (p * y + q) * (r * y^2 + s * y + t) → 
  p + q + r + s + t = 60 := by
sorry

end NUMINAMATH_CALUDE_sum_of_factors_l3674_367474


namespace NUMINAMATH_CALUDE_angle_in_second_quadrant_implies_complement_in_first_quadrant_l3674_367475

/-- If the terminal side of angle α is in the second quadrant, then π - α is in the first quadrant -/
theorem angle_in_second_quadrant_implies_complement_in_first_quadrant (α : Real) : 
  (∃ k : ℤ, 2 * k * π + π / 2 < α ∧ α < 2 * k * π + π) → 
  (∃ m : ℤ, 2 * m * π < π - α ∧ π - α < 2 * m * π + π / 2) :=
by sorry

end NUMINAMATH_CALUDE_angle_in_second_quadrant_implies_complement_in_first_quadrant_l3674_367475


namespace NUMINAMATH_CALUDE_gcf_of_450_and_144_l3674_367498

theorem gcf_of_450_and_144 : Nat.gcd 450 144 = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_450_and_144_l3674_367498


namespace NUMINAMATH_CALUDE_grain_mixture_pricing_l3674_367402

/-- Calculates the selling price of a grain given its cost price and profit percentage -/
def sellingPrice (costPrice : ℚ) (profitPercentage : ℚ) : ℚ :=
  costPrice * (1 + profitPercentage / 100)

/-- Represents the grain mixture problem -/
theorem grain_mixture_pricing
  (wheat_weight : ℚ) (wheat_price : ℚ) (wheat_profit : ℚ)
  (rice_weight : ℚ) (rice_price : ℚ) (rice_profit : ℚ)
  (barley_weight : ℚ) (barley_price : ℚ) (barley_profit : ℚ)
  (h_wheat_weight : wheat_weight = 30)
  (h_wheat_price : wheat_price = 11.5)
  (h_wheat_profit : wheat_profit = 30)
  (h_rice_weight : rice_weight = 20)
  (h_rice_price : rice_price = 14.25)
  (h_rice_profit : rice_profit = 25)
  (h_barley_weight : barley_weight = 15)
  (h_barley_price : barley_price = 10)
  (h_barley_profit : barley_profit = 35) :
  let total_weight := wheat_weight + rice_weight + barley_weight
  let total_selling_price := sellingPrice (wheat_weight * wheat_price) wheat_profit +
                             sellingPrice (rice_weight * rice_price) rice_profit +
                             sellingPrice (barley_weight * barley_price) barley_profit
  total_selling_price / total_weight = 15.5 := by
  sorry

end NUMINAMATH_CALUDE_grain_mixture_pricing_l3674_367402


namespace NUMINAMATH_CALUDE_recipe_soap_amount_l3674_367472

/-- Given a container capacity, ounces per cup, and total soap amount, 
    calculate the amount of soap per cup of water. -/
def soapPerCup (containerCapacity : ℚ) (ouncesPerCup : ℚ) (totalSoap : ℚ) : ℚ :=
  totalSoap / (containerCapacity / ouncesPerCup)

/-- Prove that the recipe calls for 3 tablespoons of soap per cup of water. -/
theorem recipe_soap_amount :
  soapPerCup 40 8 15 = 3 := by
  sorry

end NUMINAMATH_CALUDE_recipe_soap_amount_l3674_367472


namespace NUMINAMATH_CALUDE_jennifer_cards_left_l3674_367406

/-- Given that Jennifer has 72 cards initially and 61 cards are eaten,
    prove that she will have 11 cards left. -/
theorem jennifer_cards_left (initial_cards : ℕ) (eaten_cards : ℕ) 
  (h1 : initial_cards = 72) 
  (h2 : eaten_cards = 61) : 
  initial_cards - eaten_cards = 11 := by
  sorry

end NUMINAMATH_CALUDE_jennifer_cards_left_l3674_367406


namespace NUMINAMATH_CALUDE_age_of_17th_student_l3674_367447

theorem age_of_17th_student
  (total_students : Nat)
  (average_age_all : ℝ)
  (num_students_group1 : Nat)
  (average_age_group1 : ℝ)
  (num_students_group2 : Nat)
  (average_age_group2 : ℝ)
  (h1 : total_students = 17)
  (h2 : average_age_all = 17)
  (h3 : num_students_group1 = 5)
  (h4 : average_age_group1 = 14)
  (h5 : num_students_group2 = 9)
  (h6 : average_age_group2 = 16) :
  ℝ := by
  sorry

#check age_of_17th_student

end NUMINAMATH_CALUDE_age_of_17th_student_l3674_367447


namespace NUMINAMATH_CALUDE_no_snuggly_numbers_l3674_367487

/-- A two-digit positive integer is 'snuggly' if it equals the sum of its nonzero tens digit, 
    the cube of its units digit, and 5. -/
def is_snuggly (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ ∃ a b : ℕ, 
    n = 10 * a + b ∧ 
    1 ≤ a ∧ a ≤ 9 ∧ 
    0 ≤ b ∧ b ≤ 9 ∧
    n = a + b^3 + 5

theorem no_snuggly_numbers : ¬∃ n : ℕ, is_snuggly n :=
sorry

end NUMINAMATH_CALUDE_no_snuggly_numbers_l3674_367487


namespace NUMINAMATH_CALUDE_men_to_women_ratio_l3674_367431

/-- Represents a co-ed softball team -/
structure SoftballTeam where
  men : ℕ
  women : ℕ

/-- Properties of the softball team -/
def validTeam (team : SoftballTeam) : Prop :=
  team.women = team.men + 6 ∧ team.men + team.women = 16

theorem men_to_women_ratio (team : SoftballTeam) (h : validTeam team) :
  (team.men : ℚ) / team.women = 5 / 11 := by
  sorry

end NUMINAMATH_CALUDE_men_to_women_ratio_l3674_367431


namespace NUMINAMATH_CALUDE_sum_inequality_l3674_367439

theorem sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (a + c))) + (1 / (c^3 * (a + b))) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l3674_367439
