import Mathlib

namespace NUMINAMATH_CALUDE_income_data_mean_difference_l236_23662

/-- Represents the income data for a group of families -/
structure IncomeData where
  num_families : ℕ
  min_income : ℕ
  max_income : ℕ
  incorrect_max_income : ℕ

/-- Calculates the difference between the mean of incorrect data and actual data -/
def mean_difference (data : IncomeData) : ℚ :=
  (data.incorrect_max_income - data.max_income) / data.num_families

/-- Theorem stating the difference in means for the given scenario -/
theorem income_data_mean_difference :
  ∀ (data : IncomeData),
  data.num_families = 500 →
  data.min_income = 12000 →
  data.max_income = 150000 →
  data.incorrect_max_income = 1500000 →
  mean_difference data = 2700 := by
  sorry

end NUMINAMATH_CALUDE_income_data_mean_difference_l236_23662


namespace NUMINAMATH_CALUDE_exists_always_white_cell_l236_23689

-- Define the grid plane
def GridPlane := ℤ × ℤ

-- Define the state of a cell (Black or White)
inductive CellState
| Black
| White

-- Define the initial state of the grid
def initial_grid : GridPlane → CellState :=
  sorry

-- Define the polygon M
def M : Set GridPlane :=
  sorry

-- Axiom: M covers more than one cell
axiom M_size : ∃ (c1 c2 : GridPlane), c1 ≠ c2 ∧ c1 ∈ M ∧ c2 ∈ M

-- Define a valid shift of M
def valid_shift (s : GridPlane) : Prop :=
  sorry

-- Define the state of the grid after a shift
def shift_grid (g : GridPlane → CellState) (s : GridPlane) : GridPlane → CellState :=
  sorry

-- Define the state of the grid after any number of shifts
def final_grid : GridPlane → CellState :=
  sorry

-- The theorem to prove
theorem exists_always_white_cell :
  ∃ (c : GridPlane), final_grid c = CellState.White :=
sorry

end NUMINAMATH_CALUDE_exists_always_white_cell_l236_23689


namespace NUMINAMATH_CALUDE_fixed_point_exponential_function_l236_23645

theorem fixed_point_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 1) + 1
  f 1 = 2 := by sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_function_l236_23645


namespace NUMINAMATH_CALUDE_equal_probability_events_l236_23696

/-- Given a jar with 'a' white balls and 'b' black balls, where a ≠ b, this theorem proves that
    the probability of Event A (at some point, the number of drawn white balls equals the number
    of drawn black balls) is equal to the probability of Event B (at some point, the number of
    white balls remaining in the jar equals the number of black balls remaining in the jar),
    and that this probability is (2 * min(a, b)) / (a + b). -/
theorem equal_probability_events (a b : ℕ) (h : a ≠ b) :
  let total := a + b
  let prob_A := (2 * min a b) / total
  let prob_B := (2 * min a b) / total
  prob_A = prob_B ∧ prob_A = (2 * min a b) / total := by
  sorry

#check equal_probability_events

end NUMINAMATH_CALUDE_equal_probability_events_l236_23696


namespace NUMINAMATH_CALUDE_sum_of_solutions_l236_23676

theorem sum_of_solutions (x : ℝ) : (x + 16 / x = 12) → (∃ y : ℝ, y + 16 / y = 12 ∧ y ≠ x) → x + y = 12 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l236_23676


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l236_23671

/-- The line equation passes through a fixed point for all real m -/
theorem line_passes_through_fixed_point :
  ∀ (m : ℝ), (2 * m + 1) * 3 + (m + 1) * 1 - 7 * m - 4 = 0 := by
  sorry

#check line_passes_through_fixed_point

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l236_23671


namespace NUMINAMATH_CALUDE_soccer_game_analysis_l236_23638

-- Define the players
inductive Player : Type
| Amandine : Player
| Bobby : Player
| Charles : Player

-- Define the game structure
structure Game where
  total_phases : ℕ
  amandine_field : ℕ
  bobby_field : ℕ
  charles_goalkeeper : ℕ

-- Define the theorem
theorem soccer_game_analysis (g : Game) 
  (h1 : g.amandine_field = 12)
  (h2 : g.bobby_field = 21)
  (h3 : g.charles_goalkeeper = 8)
  (h4 : g.total_phases = g.amandine_field + (g.total_phases - g.amandine_field))
  (h5 : g.total_phases = g.bobby_field + (g.total_phases - g.bobby_field))
  (h6 : g.total_phases = (g.total_phases - g.charles_goalkeeper) + g.charles_goalkeeper) :
  g.total_phases = 25 ∧ (∃ n : ℕ, n = 6 ∧ n % 2 = 0 ∧ (n + 1) ≤ g.total_phases - g.amandine_field) := by
  sorry


end NUMINAMATH_CALUDE_soccer_game_analysis_l236_23638


namespace NUMINAMATH_CALUDE_point_A_in_second_quadrant_l236_23697

/-- A point is in the second quadrant if its x-coordinate is negative and its y-coordinate is positive -/
def is_in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The coordinates of point A -/
def point_A : ℝ × ℝ := (-3, 4)

/-- Theorem: Point A is located in the second quadrant -/
theorem point_A_in_second_quadrant :
  is_in_second_quadrant point_A.1 point_A.2 := by
  sorry

end NUMINAMATH_CALUDE_point_A_in_second_quadrant_l236_23697


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l236_23634

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = x*y) :
  x + 2*y ≥ 8 ∧ (x + 2*y = 8 ↔ x = 2*y) :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l236_23634


namespace NUMINAMATH_CALUDE_q_value_proof_l236_23621

theorem q_value_proof (p q : ℝ) 
  (h1 : 1 < p) (h2 : p < q) 
  (h3 : 1/p + 1/q = 3/2) 
  (h4 : p * q = 12) : 
  q = 9 + 3 * Real.sqrt 23 := by
sorry

end NUMINAMATH_CALUDE_q_value_proof_l236_23621


namespace NUMINAMATH_CALUDE_cube_surface_area_from_prism_l236_23683

/-- The surface area of a cube with the same volume as a rectangular prism -/
theorem cube_surface_area_from_prism (a b c : ℝ) (h1 : a = 5) (h2 : b = 4) (h3 : c = 40) :
  6 * (((a * b * c) ^ (1/3 : ℝ)) ^ 2) = 600 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_from_prism_l236_23683


namespace NUMINAMATH_CALUDE_quadratic_sine_interpolation_l236_23685

theorem quadratic_sine_interpolation (f : ℝ → ℝ) (h : f = λ x => -4 / Real.pi ^ 2 * x ^ 2 + 4 / Real.pi * x) :
  f 0 = 0 ∧ f (Real.pi / 2) = 1 ∧ f Real.pi = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sine_interpolation_l236_23685


namespace NUMINAMATH_CALUDE_leading_coefficient_of_g_l236_23604

/-- A polynomial g satisfying g(x + 1) - g(x) = 6x + 6 for all x has leading coefficient 3 -/
theorem leading_coefficient_of_g (g : ℝ → ℝ) 
  (h : ∀ x, g (x + 1) - g x = 6 * x + 6) :
  ∃ a b c : ℝ, (∀ x, g x = 3 * x^2 + a * x + b) ∧ c = 3 ∧ c ≠ 0 ∧ 
  (∀ d, (∀ x, g x = d * x^2 + a * x + b) → d ≤ c) := by
  sorry

end NUMINAMATH_CALUDE_leading_coefficient_of_g_l236_23604


namespace NUMINAMATH_CALUDE_piglet_straws_l236_23627

theorem piglet_straws (total_straws : ℕ) (adult_pig_fraction : ℚ) (num_piglets : ℕ) :
  total_straws = 300 →
  adult_pig_fraction = 3/5 →
  num_piglets = 20 →
  (adult_pig_fraction * total_straws : ℚ) = (total_straws - adult_pig_fraction * total_straws : ℚ) →
  (total_straws - adult_pig_fraction * total_straws) / num_piglets = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_piglet_straws_l236_23627


namespace NUMINAMATH_CALUDE_percentage_problem_l236_23679

theorem percentage_problem (x : ℝ) (h : (30/100) * (15/100) * x = 18) :
  (15/100) * (30/100) * x = 18 := by sorry

end NUMINAMATH_CALUDE_percentage_problem_l236_23679


namespace NUMINAMATH_CALUDE_jake_has_fewer_balloons_l236_23619

/-- The number of balloons each person has in the park scenario -/
structure BalloonCounts where
  allan : ℕ
  jake_initial : ℕ
  jake_bought : ℕ
  emily : ℕ

/-- The difference in balloon count between Jake and the combined total of Allan and Emily -/
def balloon_difference (counts : BalloonCounts) : ℤ :=
  (counts.jake_initial + counts.jake_bought : ℤ) - (counts.allan + counts.emily)

/-- Theorem stating that Jake has 4 fewer balloons than Allan and Emily combined -/
theorem jake_has_fewer_balloons (counts : BalloonCounts)
  (h1 : counts.allan = 6)
  (h2 : counts.jake_initial = 3)
  (h3 : counts.jake_bought = 4)
  (h4 : counts.emily = 5) :
  balloon_difference counts = -4 := by
  sorry

end NUMINAMATH_CALUDE_jake_has_fewer_balloons_l236_23619


namespace NUMINAMATH_CALUDE_floor_sqrt_48_squared_l236_23678

theorem floor_sqrt_48_squared : ⌊Real.sqrt 48⌋^2 = 36 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_48_squared_l236_23678


namespace NUMINAMATH_CALUDE_product_xyz_is_one_l236_23643

theorem product_xyz_is_one 
  (x y z : ℝ) 
  (h1 : x + 1/y = 2) 
  (h2 : y + 1/z = 2) 
  (hy_nonzero : y ≠ 0) 
  (hz_nonzero : z ≠ 0) : 
  x * y * z = 1 := by
sorry

end NUMINAMATH_CALUDE_product_xyz_is_one_l236_23643


namespace NUMINAMATH_CALUDE_representatives_count_l236_23682

/-- The number of ways to select 3 representatives from 4 boys and 4 girls,
    with at least two girls among them. -/
def select_representatives : ℕ :=
  Nat.choose 4 3 + Nat.choose 4 2 * Nat.choose 4 1

/-- Theorem stating that the number of ways to select the representatives is 28. -/
theorem representatives_count : select_representatives = 28 := by
  sorry

end NUMINAMATH_CALUDE_representatives_count_l236_23682


namespace NUMINAMATH_CALUDE_point_M_satisfies_conditions_l236_23663

-- Define the function f(x) = 2x^2 + 1
def f (x : ℝ) : ℝ := 2 * x^2 + 1

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 4 * x

theorem point_M_satisfies_conditions :
  let x₀ : ℝ := -2
  let y₀ : ℝ := 9
  f x₀ = y₀ ∧ f' x₀ = -8 := by
  sorry

end NUMINAMATH_CALUDE_point_M_satisfies_conditions_l236_23663


namespace NUMINAMATH_CALUDE_movie_ticket_change_l236_23681

/-- Calculates the change received by two sisters buying movie tickets -/
theorem movie_ticket_change (full_price : ℚ) (discount_percent : ℚ) (brought_money : ℚ) : 
  full_price = 8 →
  discount_percent = 25 / 100 →
  brought_money = 25 →
  let discounted_price := full_price * (1 - discount_percent)
  let total_cost := full_price + discounted_price
  brought_money - total_cost = 11 := by
sorry

end NUMINAMATH_CALUDE_movie_ticket_change_l236_23681


namespace NUMINAMATH_CALUDE_laptop_tote_weight_difference_l236_23673

/-- Represents the weights of various items in pounds -/
structure Weights where
  karens_tote : ℝ
  kevins_empty_briefcase : ℝ
  kevins_full_briefcase : ℝ
  kevins_work_papers : ℝ
  kevins_laptop : ℝ

/-- Conditions of the problem -/
def problem_conditions (w : Weights) : Prop :=
  w.karens_tote = 8 ∧
  w.karens_tote = 2 * w.kevins_empty_briefcase ∧
  w.kevins_full_briefcase = 2 * w.karens_tote ∧
  w.kevins_work_papers = (w.kevins_full_briefcase - w.kevins_empty_briefcase) / 6 ∧
  w.kevins_laptop = w.kevins_full_briefcase - w.kevins_empty_briefcase - w.kevins_work_papers

/-- The theorem to be proved -/
theorem laptop_tote_weight_difference (w : Weights) 
  (h : problem_conditions w) : w.kevins_laptop - w.karens_tote = 2 := by
  sorry

end NUMINAMATH_CALUDE_laptop_tote_weight_difference_l236_23673


namespace NUMINAMATH_CALUDE_product_of_sum_and_cube_sum_l236_23624

theorem product_of_sum_and_cube_sum (x y : ℝ) 
  (h1 : x + y = 9) 
  (h2 : x^3 + y^3 = 351) : 
  x * y = 14 := by
sorry

end NUMINAMATH_CALUDE_product_of_sum_and_cube_sum_l236_23624


namespace NUMINAMATH_CALUDE_constant_term_is_60_l236_23648

/-- The constant term in the binomial expansion of (2x^2 - 1/x)^6 -/
def constant_term : ℤ :=
  (Finset.range 7).sum (fun r => 
    (-1)^r * (Nat.choose 6 r) * 2^(6-r) * 
    if 12 - 3*r = 0 then 1 else 0)

/-- The constant term in the binomial expansion of (2x^2 - 1/x)^6 is 60 -/
theorem constant_term_is_60 : constant_term = 60 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_is_60_l236_23648


namespace NUMINAMATH_CALUDE_factorial_ratio_l236_23658

def factorial : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem factorial_ratio (n : ℕ) (h : n > 0) : 
  factorial n / factorial (n - 1) = n := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l236_23658


namespace NUMINAMATH_CALUDE_favorite_numbers_sum_l236_23632

/-- Given the favorite numbers of Misty, Glory, and Dawn, prove their sum is 1500 -/
theorem favorite_numbers_sum (glory_fav : ℕ) (misty_fav : ℕ) (dawn_fav : ℕ) 
  (h1 : glory_fav = 450)
  (h2 : misty_fav * 3 = glory_fav)
  (h3 : dawn_fav = glory_fav * 2) :
  misty_fav + glory_fav + dawn_fav = 1500 := by
  sorry

end NUMINAMATH_CALUDE_favorite_numbers_sum_l236_23632


namespace NUMINAMATH_CALUDE_bus_encounters_l236_23684

-- Define the schedule and travel time
def austin_departure_interval : ℕ := 2
def sanantonio_departure_interval : ℕ := 2
def sanantonio_departure_offset : ℕ := 1
def travel_time : ℕ := 7

-- Define the number of encounters
def encounters : ℕ := 4

-- Theorem statement
theorem bus_encounters :
  (austin_departure_interval = 2) →
  (sanantonio_departure_interval = 2) →
  (sanantonio_departure_offset = 1) →
  (travel_time = 7) →
  (encounters = 4) := by
  sorry

end NUMINAMATH_CALUDE_bus_encounters_l236_23684


namespace NUMINAMATH_CALUDE_solution_existence_l236_23669

theorem solution_existence (x y : ℝ) : 
  |x + 1| + (y - 8)^2 = 0 → x = -1 ∧ y = 8 := by
  sorry

end NUMINAMATH_CALUDE_solution_existence_l236_23669


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l236_23670

theorem sufficient_not_necessary_condition :
  (∀ x : ℝ, x > 2 → x - 1 > 0) ∧
  ¬(∀ x : ℝ, x - 1 > 0 → x > 2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l236_23670


namespace NUMINAMATH_CALUDE_pyarelal_loss_is_900_l236_23629

/-- Calculates Pyarelal's share of the loss given the investment ratio and total loss -/
def pyarelal_loss (pyarelal_capital : ℚ) (total_loss : ℚ) : ℚ :=
  let ashok_capital := (1 : ℚ) / 9 * pyarelal_capital
  let total_capital := ashok_capital + pyarelal_capital
  let pyarelal_ratio := pyarelal_capital / total_capital
  pyarelal_ratio * total_loss

/-- Theorem stating that Pyarelal's loss is 900 given the conditions of the problem -/
theorem pyarelal_loss_is_900 (pyarelal_capital : ℚ) (h : pyarelal_capital > 0) :
  pyarelal_loss pyarelal_capital 1000 = 900 := by
  sorry

end NUMINAMATH_CALUDE_pyarelal_loss_is_900_l236_23629


namespace NUMINAMATH_CALUDE_b_55_divisible_by_55_l236_23647

/-- Function that generates b_n as described in the problem -/
def b (n : ℕ) : ℕ := sorry

/-- Theorem stating that b(55) is divisible by 55 -/
theorem b_55_divisible_by_55 : 55 ∣ b 55 := by sorry

end NUMINAMATH_CALUDE_b_55_divisible_by_55_l236_23647


namespace NUMINAMATH_CALUDE_inequality_solution_l236_23644

theorem inequality_solution (a x : ℝ) :
  (a < 0 ∨ a > 1 → (((x - a) / (x - a^2) < 0) ↔ (a < x ∧ x < a^2))) ∧
  (0 < a ∧ a < 1 → (((x - a) / (x - a^2) < 0) ↔ (a^2 < x ∧ x < a))) ∧
  (a = 0 ∨ a = 1 → ¬∃x, (x - a) / (x - a^2) < 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l236_23644


namespace NUMINAMATH_CALUDE_largest_four_digit_product_of_primes_l236_23674

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that checks if a number is a four-digit positive integer -/
def isFourDigit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem largest_four_digit_product_of_primes :
  ∃ (n x y : ℕ),
    isFourDigit n ∧
    isPrime x ∧
    isPrime y ∧
    x < 10 ∧
    y < 10 ∧
    isPrime (10 * y + x) ∧
    n = x * y * (10 * y + x) ∧
    (∀ (m a b : ℕ),
      isFourDigit m →
      isPrime a →
      isPrime b →
      a < 10 →
      b < 10 →
      isPrime (10 * b + a) →
      m = a * b * (10 * b + a) →
      m ≤ n) ∧
    n = 1533 :=
  sorry

end NUMINAMATH_CALUDE_largest_four_digit_product_of_primes_l236_23674


namespace NUMINAMATH_CALUDE_prob_at_least_one_8_l236_23605

/-- The number of sides on each die -/
def numSides : ℕ := 8

/-- The probability of getting at least one 8 when rolling two fair 8-sided dice -/
def probAtLeastOne8 : ℚ := 15 / 64

/-- Theorem: The probability of getting at least one 8 when rolling two fair 8-sided dice is 15/64 -/
theorem prob_at_least_one_8 : 
  probAtLeastOne8 = (numSides^2 - (numSides - 1)^2) / numSides^2 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_8_l236_23605


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_l236_23631

theorem complex_magnitude_equation (n : ℝ) :
  n > 0 → (Complex.abs (5 + n * Complex.I) = 5 * Real.sqrt 6 ↔ n = 5 * Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_l236_23631


namespace NUMINAMATH_CALUDE_degree_of_P_l236_23623

/-- The degree of a monomial in two variables --/
def monomialDegree (a b : ℕ) : ℕ := a + b

/-- The degree of a polynomial is the maximum degree of its monomials --/
def polynomialDegree (degrees : List ℕ) : ℕ := List.foldl max 0 degrees

/-- The polynomial -3a²b + 7a²b² - 2ab --/
def P (a b : ℝ) : ℝ := -3 * a^2 * b + 7 * a^2 * b^2 - 2 * a * b

theorem degree_of_P : 
  polynomialDegree [monomialDegree 2 1, monomialDegree 2 2, monomialDegree 1 1] = 4 := by
  sorry

end NUMINAMATH_CALUDE_degree_of_P_l236_23623


namespace NUMINAMATH_CALUDE_smallest_number_l236_23664

theorem smallest_number (π : ℝ) (h : π > 0) : min (-π) (min (-2) (min 0 (Real.sqrt 3))) = -π := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l236_23664


namespace NUMINAMATH_CALUDE_milk_water_ratio_l236_23661

theorem milk_water_ratio (initial_volume : ℚ) (initial_milk_ratio : ℚ) (initial_water_ratio : ℚ) (added_water : ℚ) : 
  initial_volume = 45 ∧ 
  initial_milk_ratio = 4 ∧ 
  initial_water_ratio = 1 ∧ 
  added_water = 18 → 
  let initial_milk := (initial_milk_ratio / (initial_milk_ratio + initial_water_ratio)) * initial_volume
  let initial_water := (initial_water_ratio / (initial_milk_ratio + initial_water_ratio)) * initial_volume
  let final_water := initial_water + added_water
  let final_milk_ratio := initial_milk / final_water
  let final_water_ratio := final_water / final_water
  final_milk_ratio / final_water_ratio = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_milk_water_ratio_l236_23661


namespace NUMINAMATH_CALUDE_complex_number_properties_l236_23637

theorem complex_number_properties (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  ((a + b)^2 = a^2 + 2*a*b + b^2) ∧ 
  (a^2 = a*b → a = b) ∧ 
  (∃ c : ℂ, c ≠ 0 ∧ c + 1/c = 0) ∧
  (∃ x y : ℂ, Complex.abs x = Complex.abs y ∧ x ≠ y ∧ x ≠ -y) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_properties_l236_23637


namespace NUMINAMATH_CALUDE_circular_arrangement_multiple_of_four_l236_23690

/-- Represents a child in the circular arrangement -/
inductive Child
| Boy
| Girl

/-- Represents the circular arrangement of children -/
def CircularArrangement := List Child

/-- Counts the number of children whose right-hand neighbor is of the same gender -/
def countSameGenderNeighbors (arrangement : CircularArrangement) : Nat :=
  sorry

/-- Counts the number of children whose right-hand neighbor is of a different gender -/
def countDifferentGenderNeighbors (arrangement : CircularArrangement) : Nat :=
  sorry

/-- Checks if the arrangement satisfies the equal neighbor condition -/
def hasEqualNeighbors (arrangement : CircularArrangement) : Prop :=
  countSameGenderNeighbors arrangement = countDifferentGenderNeighbors arrangement

theorem circular_arrangement_multiple_of_four 
  (arrangement : CircularArrangement) 
  (h : hasEqualNeighbors arrangement) :
  ∃ k : Nat, arrangement.length = 4 * k :=
sorry

end NUMINAMATH_CALUDE_circular_arrangement_multiple_of_four_l236_23690


namespace NUMINAMATH_CALUDE_expression_equals_588_times_10_to_1007_l236_23686

theorem expression_equals_588_times_10_to_1007 :
  (3^1006 + 7^1007)^2 - (3^1006 - 7^1007)^2 = 588 * 10^1007 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_588_times_10_to_1007_l236_23686


namespace NUMINAMATH_CALUDE_trapezoid_angle_bisector_inscribed_circle_l236_23641

noncomputable section

/-- Represents a point in 2D space -/
structure Point := (x : ℝ) (y : ℝ)

/-- Represents a trapezoid ABCD -/
structure Trapezoid :=
  (A B C D : Point)

/-- Represents a triangle -/
structure Triangle :=
  (A B C : Point)

/-- Represents a circle -/
structure Circle :=
  (center : Point) (radius : ℝ)

/-- The length of a line segment between two points -/
def distance (p q : Point) : ℝ := sorry

/-- The angle between three points -/
def angle (p q r : Point) : ℝ := sorry

/-- The angle bisector of an angle -/
def angleBisector (p q r : Point) : Point := sorry

/-- Check if a point lies on a line segment -/
def onSegment (p q r : Point) : Prop := sorry

/-- Check if a circle is inscribed in a triangle -/
def isInscribed (c : Circle) (t : Triangle) : Prop := sorry

/-- Check if a point is a tangent point of a circle on a line segment -/
def isTangentPoint (p : Point) (c : Circle) (q r : Point) : Prop := sorry

theorem trapezoid_angle_bisector_inscribed_circle 
  (ABCD : Trapezoid) (E M H : Point) (c : Circle) :
  onSegment E ABCD.B ABCD.C →
  angleBisector ABCD.B ABCD.A ABCD.D = E →
  isInscribed c (Triangle.mk ABCD.A ABCD.B E) →
  isTangentPoint M c ABCD.A ABCD.B →
  isTangentPoint H c ABCD.B E →
  distance ABCD.A ABCD.B = 2 →
  distance M H = 1 →
  angle ABCD.B ABCD.A ABCD.D = 120 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_angle_bisector_inscribed_circle_l236_23641


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonal_l236_23650

theorem rectangular_prism_diagonal (l w h : ℝ) (hl : l = 6) (hw : w = 8) (hh : h = 15) :
  Real.sqrt (l^2 + w^2 + h^2) = Real.sqrt 325 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonal_l236_23650


namespace NUMINAMATH_CALUDE_amp_calculation_l236_23608

-- Define the & operation
def amp (a b : ℤ) : ℤ := (a + b) * (a - b)

-- State the theorem
theorem amp_calculation : amp (amp 10 4) 2 = 7052 := by
  sorry

end NUMINAMATH_CALUDE_amp_calculation_l236_23608


namespace NUMINAMATH_CALUDE_system_equation_ratio_l236_23642

theorem system_equation_ratio (x y c d : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0)
  (eq1 : 8 * x - 6 * y = c) (eq2 : 12 * y - 18 * x = d) :
  c / d = -4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_system_equation_ratio_l236_23642


namespace NUMINAMATH_CALUDE_nancy_savings_l236_23614

-- Define the number of quarters in a dozen
def dozen_quarters : ℕ := 12

-- Define the value of a quarter in cents
def quarter_value : ℕ := 25

-- Define the number of cents in a dollar
def cents_per_dollar : ℕ := 100

-- Theorem to prove
theorem nancy_savings : (dozen_quarters * quarter_value) / cents_per_dollar = 3 := by
  sorry

end NUMINAMATH_CALUDE_nancy_savings_l236_23614


namespace NUMINAMATH_CALUDE_math_team_combinations_l236_23636

/-- The number of girls in the math club -/
def num_girls : ℕ := 4

/-- The number of boys in the math club -/
def num_boys : ℕ := 6

/-- The number of girls to be selected for the team -/
def girls_in_team : ℕ := 3

/-- The number of boys to be selected for the team -/
def boys_in_team : ℕ := 2

/-- The total number of possible team combinations -/
def total_combinations : ℕ := 60

theorem math_team_combinations :
  (Nat.choose num_girls girls_in_team) * (Nat.choose num_boys boys_in_team) = total_combinations :=
sorry

end NUMINAMATH_CALUDE_math_team_combinations_l236_23636


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l236_23653

theorem right_triangle_hypotenuse : ∀ (a b c : ℝ),
  -- Right-angled triangle condition (Pythagorean theorem)
  c^2 = a^2 + b^2 →
  -- Sum of squares of all sides is 2500
  a^2 + b^2 + c^2 = 2500 →
  -- Difference between hypotenuse and one side is 10
  c - a = 10 →
  -- Prove that the hypotenuse length is 25√2
  c = 25 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l236_23653


namespace NUMINAMATH_CALUDE_vincent_stickers_l236_23656

/-- The number of packs Vincent bought yesterday -/
def yesterday_packs : ℕ := 15

/-- The total number of packs Vincent has -/
def total_packs : ℕ := 40

/-- The number of additional packs Vincent bought today -/
def additional_packs : ℕ := total_packs - yesterday_packs

theorem vincent_stickers :
  additional_packs = 10 ∧ additional_packs > 0 := by
  sorry

end NUMINAMATH_CALUDE_vincent_stickers_l236_23656


namespace NUMINAMATH_CALUDE_parallel_lines_l236_23672

/-- Two lines in the form ax + by + c = 0 are parallel if and only if they have the same a and b coefficients. -/
def are_parallel (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  a1 = a2 ∧ b1 = b2 ∧ c1 ≠ c2

/-- The line x + 2y + 2 = 0 is parallel to the line x + 2y + 1 = 0. -/
theorem parallel_lines : are_parallel 1 2 2 1 2 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_l236_23672


namespace NUMINAMATH_CALUDE_three_random_events_l236_23601

/-- Represents an event that can occur in a probability space. -/
structure Event where
  description : String
  is_random : Bool

/-- The set of events we're considering. -/
def events : List Event := [
  ⟨"Selecting 3 out of 10 glass cups (8 good quality, 2 defective), all 3 selected are good quality", true⟩,
  ⟨"Randomly pressing a digit and it happens to be correct when forgetting the last digit of a phone number", true⟩,
  ⟨"Opposite electric charges attract each other", false⟩,
  ⟨"A person wins the first prize in a sports lottery", true⟩
]

/-- Counts the number of random events in a list of events. -/
def countRandomEvents (events : List Event) : Nat :=
  events.filter (·.is_random) |>.length

/-- The main theorem stating that exactly three of the given events are random. -/
theorem three_random_events : countRandomEvents events = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_random_events_l236_23601


namespace NUMINAMATH_CALUDE_positive_solution_x_l236_23694

theorem positive_solution_x (x y z : ℝ) 
  (eq1 : x * y = 10 - 3 * x - 2 * y)
  (eq2 : y * z = 8 - 3 * y - 2 * z)
  (eq3 : x * z = 40 - 5 * x - 3 * z)
  (x_pos : x > 0) :
  x = 3 := by
sorry

end NUMINAMATH_CALUDE_positive_solution_x_l236_23694


namespace NUMINAMATH_CALUDE_range_of_a_part1_range_of_a_part2_l236_23617

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 3

-- Theorem for the first part of the problem
theorem range_of_a_part1 (a : ℝ) :
  (∃ x ∈ Set.Icc 1 3, f a x > 0) → a < 4 :=
sorry

-- Theorem for the second part of the problem
theorem range_of_a_part2 (a : ℝ) :
  (∀ x ∈ Set.Icc 1 4, f a x ≥ -a) → a ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_part1_range_of_a_part2_l236_23617


namespace NUMINAMATH_CALUDE_stock_trade_profit_l236_23688

/-- Represents the stock trading scenario --/
structure StockTrade where
  initial_price : ℝ
  price_changes : List ℝ
  num_shares : ℕ
  buying_fee : ℝ
  selling_fee : ℝ
  transaction_tax : ℝ

/-- Calculates the final price of the stock --/
def final_price (trade : StockTrade) : ℝ :=
  trade.initial_price + trade.price_changes.sum

/-- Calculates the profit from the stock trade --/
def calculate_profit (trade : StockTrade) : ℝ :=
  let cost := trade.initial_price * trade.num_shares * (1 + trade.buying_fee)
  let revenue := (final_price trade) * trade.num_shares * (1 - trade.selling_fee - trade.transaction_tax)
  revenue - cost

/-- Theorem stating that the profit from the given stock trade is 889.5 yuan --/
theorem stock_trade_profit (trade : StockTrade) 
  (h1 : trade.initial_price = 27)
  (h2 : trade.price_changes = [4, 4.5, -1, -2.5, -6, 2])
  (h3 : trade.num_shares = 1000)
  (h4 : trade.buying_fee = 0.0015)
  (h5 : trade.selling_fee = 0.0015)
  (h6 : trade.transaction_tax = 0.001) :
  calculate_profit trade = 889.5 := by
  sorry

end NUMINAMATH_CALUDE_stock_trade_profit_l236_23688


namespace NUMINAMATH_CALUDE_tv_screen_diagonal_l236_23665

theorem tv_screen_diagonal (d : ℝ) : d > 0 → d^2 = 17^2 + 76 → d = Real.sqrt 365 := by
  sorry

end NUMINAMATH_CALUDE_tv_screen_diagonal_l236_23665


namespace NUMINAMATH_CALUDE_acquainted_pairs_bound_l236_23657

/-- Represents a company with n persons, where each person has no more than d acquaintances,
    and there exists a group of k persons (k ≥ d) who are not acquainted with each other. -/
structure Company where
  n : ℕ  -- Total number of persons
  d : ℕ  -- Maximum number of acquaintances per person
  k : ℕ  -- Size of the group of unacquainted persons
  h1 : k ≥ d  -- Condition that k is not less than d

/-- The number of acquainted pairs in the company -/
def acquaintedPairs (c : Company) : ℕ := sorry

/-- Theorem stating that the number of acquainted pairs is not greater than ⌊n²/4⌋ -/
theorem acquainted_pairs_bound (c : Company) : 
  acquaintedPairs c ≤ (c.n^2) / 4 := by sorry

end NUMINAMATH_CALUDE_acquainted_pairs_bound_l236_23657


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l236_23625

theorem arithmetic_mean_problem (y : ℚ) : 
  ((y + 10) + 20 + (3 * y) + 18 + (3 * y + 6) + 12) / 6 = 30 → y = 114 / 7 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l236_23625


namespace NUMINAMATH_CALUDE_binary_11011011_to_base4_l236_23649

def binary_to_decimal (b : List Bool) : ℕ :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

def decimal_to_base4 (n : ℕ) : List (Fin 4) :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) : List (Fin 4) :=
      if m = 0 then [] else (m % 4) :: aux (m / 4)
    aux n |>.reverse

theorem binary_11011011_to_base4 :
  decimal_to_base4 (binary_to_decimal [true, true, false, true, true, false, true, true]) =
  [3, 1, 2, 3] :=
by sorry

end NUMINAMATH_CALUDE_binary_11011011_to_base4_l236_23649


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l236_23659

/-- An arithmetic sequence. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  a 2 + a 4 + a 7 + a 11 = 44 →
  a 3 + a 5 + a 10 = 33 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l236_23659


namespace NUMINAMATH_CALUDE_base7_146_equals_83_l236_23620

/-- Converts a base 7 number to base 10 -/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

theorem base7_146_equals_83 :
  base7ToBase10 [6, 4, 1] = 83 := by sorry

end NUMINAMATH_CALUDE_base7_146_equals_83_l236_23620


namespace NUMINAMATH_CALUDE_pottery_rim_diameter_l236_23600

theorem pottery_rim_diameter 
  (chord_length : ℝ) 
  (segment_height : ℝ) 
  (h1 : chord_length = 16) 
  (h2 : segment_height = 2) : 
  ∃ (diameter : ℝ), diameter = 34 ∧ 
  (∃ (radius : ℝ), 
    radius * 2 = diameter ∧
    radius^2 = (radius - segment_height)^2 + (chord_length / 2)^2) :=
by sorry

end NUMINAMATH_CALUDE_pottery_rim_diameter_l236_23600


namespace NUMINAMATH_CALUDE_prob_six_given_hugo_wins_l236_23626

/-- The number of players in the game -/
def num_players : ℕ := 5

/-- The number of sides on each die -/
def die_sides : ℕ := 6

/-- The probability of rolling a 6 on a single die -/
def prob_roll_six : ℚ := 1 / die_sides

/-- The probability of Hugo winning the game -/
def prob_hugo_wins : ℚ := 1 / num_players

/-- The probability that Hugo wins given his first roll was a 6 -/
noncomputable def prob_hugo_wins_given_six : ℚ := 875 / 1296

/-- Theorem: The probability that Hugo's first roll was 6, given that he won the game -/
theorem prob_six_given_hugo_wins :
  (prob_roll_six * prob_hugo_wins_given_six) / prob_hugo_wins = 4375 / 7776 := by sorry

end NUMINAMATH_CALUDE_prob_six_given_hugo_wins_l236_23626


namespace NUMINAMATH_CALUDE_sqrt_inequality_l236_23651

theorem sqrt_inequality (p q x₁ x₂ : ℝ) (hp : p > 0) (hq : q > 0) (hpq : p + q = 1) :
  p * Real.sqrt x₁ + q * Real.sqrt x₂ ≤ Real.sqrt (p * x₁ + q * x₂) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l236_23651


namespace NUMINAMATH_CALUDE_remainder_of_large_number_l236_23610

theorem remainder_of_large_number (n : Nat) (d : Nat) (h : d = 180) :
  n = 1234567890123 → n % d = 123 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_large_number_l236_23610


namespace NUMINAMATH_CALUDE_parallel_tangents_imply_a_range_l236_23695

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then -x^2 + (2*a - 2)*x
  else x^3 - (3*a + 3)*x^2 + a*x

-- Define the derivative of f(x)
noncomputable def f_prime (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then -2*x + 2*a - 2
  else 3*x^2 - 6*(a + 1)*x + a

-- Theorem statement
theorem parallel_tangents_imply_a_range (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    f_prime a x₁ = f_prime a x₂ ∧ f_prime a x₂ = f_prime a x₃) →
  -1 < a ∧ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_tangents_imply_a_range_l236_23695


namespace NUMINAMATH_CALUDE_walnut_trees_before_planting_l236_23607

theorem walnut_trees_before_planting (trees_to_plant : ℕ) (final_trees : ℕ) 
  (h1 : trees_to_plant = 104)
  (h2 : final_trees = 211) :
  final_trees - trees_to_plant = 107 := by
  sorry

end NUMINAMATH_CALUDE_walnut_trees_before_planting_l236_23607


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l236_23622

def A : Set ℝ := {x | -1 < x ∧ x < 3}
def B : Set ℝ := {-1, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l236_23622


namespace NUMINAMATH_CALUDE_eva_second_semester_maths_score_l236_23611

/-- Represents Eva's scores in a semester -/
structure SemesterScores where
  maths : ℕ
  arts : ℕ
  science : ℕ

/-- Calculates the total score for a semester -/
def totalScore (scores : SemesterScores) : ℕ :=
  scores.maths + scores.arts + scores.science

theorem eva_second_semester_maths_score :
  ∀ (first second : SemesterScores),
    first.maths = second.maths + 10 →
    first.arts + 15 = second.arts →
    first.science + (first.science / 3) = second.science →
    second.arts = 90 →
    second.science = 90 →
    totalScore first + totalScore second = 485 →
    second.maths = 80 := by
  sorry

end NUMINAMATH_CALUDE_eva_second_semester_maths_score_l236_23611


namespace NUMINAMATH_CALUDE_library_visitors_library_visitors_proof_l236_23698

/-- Calculates the average number of visitors on non-Sunday days in a library -/
theorem library_visitors (sunday_avg : ℕ) (month_avg : ℕ) : ℕ :=
  let total_days : ℕ := 30
  let sunday_count : ℕ := 5
  let other_days : ℕ := total_days - sunday_count
  let total_visitors : ℕ := month_avg * total_days
  let sunday_visitors : ℕ := sunday_avg * sunday_count
  (total_visitors - sunday_visitors) / other_days

/-- Proves that the average number of visitors on non-Sunday days is 240 -/
theorem library_visitors_proof :
  library_visitors 660 310 = 240 := by
sorry

end NUMINAMATH_CALUDE_library_visitors_library_visitors_proof_l236_23698


namespace NUMINAMATH_CALUDE_triangle_properties_l236_23691

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  Real.sqrt 3 * Real.tan t.A * Real.tan t.B - Real.tan t.A - Real.tan t.B = Real.sqrt 3 ∧
  t.c = 2 ∧
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = Real.pi

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) :
  t.C = Real.pi / 3 ∧ 20 / 3 < t.a^2 + t.b^2 ∧ t.a^2 + t.b^2 ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l236_23691


namespace NUMINAMATH_CALUDE_solution_x_l236_23675

-- Define m and n as distinct non-zero real constants
variable (m n : ℝ) (h : m ≠ n) (hm : m ≠ 0) (hn : n ≠ 0)

-- Define the equation
def equation (x : ℝ) : Prop :=
  (x + m)^2 - 3*(x + n)^2 = m^2 - 3*n^2

-- Theorem statement
theorem solution_x (x : ℝ) : 
  equation m n x → (x = 0 ∨ x = m - 3*n) :=
by sorry

end NUMINAMATH_CALUDE_solution_x_l236_23675


namespace NUMINAMATH_CALUDE_equal_volume_implies_equal_breadth_l236_23635

/-- Represents the volume of earth dug in a project -/
structure EarthVolume where
  depth : ℝ
  length : ℝ
  breadth : ℝ

/-- Calculates the volume of earth dug -/
def calculateVolume (v : EarthVolume) : ℝ :=
  v.depth * v.length * v.breadth

theorem equal_volume_implies_equal_breadth 
  (project1 : EarthVolume)
  (project2 : EarthVolume)
  (h1 : project1.depth = 100)
  (h2 : project1.length = 25)
  (h3 : project1.breadth = 30)
  (h4 : project2.depth = 75)
  (h5 : project2.length = 20)
  (h6 : calculateVolume project1 = calculateVolume project2) :
  project2.breadth = 50 := by
sorry

end NUMINAMATH_CALUDE_equal_volume_implies_equal_breadth_l236_23635


namespace NUMINAMATH_CALUDE_fraction_product_equals_fifteen_thirty_seconds_l236_23602

theorem fraction_product_equals_fifteen_thirty_seconds :
  (3 + 5 + 7) / (2 + 4 + 6) * (1 + 3 + 5) / (6 + 8 + 10) = 15 / 32 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_equals_fifteen_thirty_seconds_l236_23602


namespace NUMINAMATH_CALUDE_square_difference_times_three_l236_23680

theorem square_difference_times_three : (538^2 - 462^2) * 3 = 228000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_times_three_l236_23680


namespace NUMINAMATH_CALUDE_students_not_playing_sports_l236_23616

theorem students_not_playing_sports (total : ℕ) (football : ℕ) (tennis : ℕ) (basketball : ℕ)
  (football_tennis : ℕ) (football_basketball : ℕ) (tennis_basketball : ℕ) (all_three : ℕ)
  (h_total : total = 50)
  (h_football : football = 26)
  (h_tennis : tennis = 20)
  (h_basketball : basketball = 15)
  (h_football_tennis : football_tennis = 9)
  (h_football_basketball : football_basketball = 7)
  (h_tennis_basketball : tennis_basketball = 6)
  (h_all_three : all_three = 4) :
  total - (football + tennis + basketball - football_tennis - football_basketball - tennis_basketball + all_three) = 7 := by
sorry

end NUMINAMATH_CALUDE_students_not_playing_sports_l236_23616


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l236_23666

theorem quadratic_root_relation (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, 
    (2 * x₁^2 - (2*m + 1)*x₁ + m^2 - 9*m + 39 = 0) ∧ 
    (2 * x₂^2 - (2*m + 1)*x₂ + m^2 - 9*m + 39 = 0) ∧ 
    (x₂ = 2 * x₁)) → 
  (m = 10 ∨ m = 7) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l236_23666


namespace NUMINAMATH_CALUDE_common_chord_and_length_l236_23633

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 8*y - 8 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y - 2 = 0

-- Define the equation of the common chord
def common_chord (x y : ℝ) : Prop := x + 2*y - 1 = 0

-- State the theorem
theorem common_chord_and_length :
  -- The equation of the common chord
  (∀ x y : ℝ, circle1 x y ∧ circle2 x y → common_chord x y) ∧
  -- The length of the common chord
  (∃ a b c d : ℝ,
    circle1 a b ∧ circle2 a b ∧ circle1 c d ∧ circle2 c d ∧
    common_chord a b ∧ common_chord c d ∧
    ((a - c)^2 + (b - d)^2) = 20) :=
by sorry

end NUMINAMATH_CALUDE_common_chord_and_length_l236_23633


namespace NUMINAMATH_CALUDE_base5ToBinary_110_equals_11110_l236_23668

-- Define a function to convert a number from base 5 to decimal
def base5ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

-- Define a function to convert a decimal number to binary
def decimalToBinary (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec go (m : Nat) (acc : List Nat) :=
    if m = 0 then acc
    else go (m / 2) ((m % 2) :: acc)
  go n []

-- Theorem statement
theorem base5ToBinary_110_equals_11110 :
  decimalToBinary (base5ToDecimal [0, 1, 1]) = [1, 1, 1, 1, 0] := by
  sorry

end NUMINAMATH_CALUDE_base5ToBinary_110_equals_11110_l236_23668


namespace NUMINAMATH_CALUDE_line_passes_through_circle_center_l236_23640

/-- The line 2x - y = 0 passes through the center of the circle (x-a)² + (y-2a)² = 1 for all real a -/
theorem line_passes_through_circle_center (a : ℝ) : 2 * a - 2 * a = 0 := by sorry

end NUMINAMATH_CALUDE_line_passes_through_circle_center_l236_23640


namespace NUMINAMATH_CALUDE_line_always_intersects_ellipse_l236_23609

/-- A line with equation y = kx + 2, where k is a real number. -/
structure Line where
  k : ℝ

/-- An ellipse with equation x² + y²/m = 1, where m is a positive real number. -/
structure Ellipse where
  m : ℝ
  h_pos : 0 < m

/-- 
Given a line y = kx + 2 and an ellipse x² + y²/m = 1,
if the line always intersects the ellipse for all real k,
then m is greater than or equal to 4.
-/
theorem line_always_intersects_ellipse (e : Ellipse) :
  (∀ k : ℝ, ∃ x y : ℝ, y = k * x + 2 ∧ x^2 + y^2 / e.m = 1) →
  4 ≤ e.m :=
sorry

end NUMINAMATH_CALUDE_line_always_intersects_ellipse_l236_23609


namespace NUMINAMATH_CALUDE_g_zero_at_three_l236_23687

/-- The polynomial function g(x) -/
def g (x s : ℝ) : ℝ := 3*x^5 - 2*x^4 + x^3 - 4*x^2 + 5*x + s

/-- Theorem stating that g(3) = 0 when s = -573 -/
theorem g_zero_at_three : g 3 (-573) = 0 := by sorry

end NUMINAMATH_CALUDE_g_zero_at_three_l236_23687


namespace NUMINAMATH_CALUDE_max_min_x_plus_y_l236_23699

theorem max_min_x_plus_y (x y : ℝ) :
  (|x + 2| + |1 - x| = 9 - |y - 5| - |1 + y|) →
  (∃ (a b : ℝ), (∀ z w : ℝ, |z + 2| + |1 - z| = 9 - |w - 5| - |1 + w| → x + y ≤ a ∧ b ≤ z + w) ∧
                 a = 6 ∧ b = -3) :=
by sorry

end NUMINAMATH_CALUDE_max_min_x_plus_y_l236_23699


namespace NUMINAMATH_CALUDE_trig_identity_proof_l236_23667

theorem trig_identity_proof : 
  Real.sin (17 * π / 180) * Real.cos (43 * π / 180) + 
  Real.sin (73 * π / 180) * Real.sin (43 * π / 180) = 
  Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l236_23667


namespace NUMINAMATH_CALUDE_rahims_book_purchase_l236_23628

/-- Given Rahim's book purchases, prove the amount paid for books from the first shop -/
theorem rahims_book_purchase (first_shop_books : ℕ) (second_shop_books : ℕ) 
  (second_shop_cost : ℕ) (average_price : ℕ) (h1 : first_shop_books = 42) 
  (h2 : second_shop_books = 22) (h3 : second_shop_cost = 248) (h4 : average_price = 12) :
  (first_shop_books + second_shop_books) * average_price - second_shop_cost = 520 := by
  sorry

#check rahims_book_purchase

end NUMINAMATH_CALUDE_rahims_book_purchase_l236_23628


namespace NUMINAMATH_CALUDE_grid_cutting_ways_l236_23655

-- Define the shape of the grid
def GridShape : Type := Unit  -- Placeholder for the specific grid shape

-- Define the property of being cuttable into 1×2 rectangles
def IsCuttableInto1x2Rectangles (g : GridShape) : Prop := sorry

-- Define the function that counts the number of ways to cut the grid
def NumberOfWaysToCut (g : GridShape) : ℕ := sorry

-- The main theorem
theorem grid_cutting_ways (g : GridShape) : 
  IsCuttableInto1x2Rectangles g → NumberOfWaysToCut g = 27 := by sorry

end NUMINAMATH_CALUDE_grid_cutting_ways_l236_23655


namespace NUMINAMATH_CALUDE_fraction_simplification_l236_23618

theorem fraction_simplification : (20 - 20) / (20 + 20) = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l236_23618


namespace NUMINAMATH_CALUDE_system_solution_proof_l236_23652

theorem system_solution_proof :
  ∃ (x y z : ℝ),
    x = 48 ∧ y = 16 ∧ z = 12 ∧
    (x * y) / (5 * x + 4 * y) = 6 ∧
    (x * z) / (3 * x + 2 * z) = 8 ∧
    (y * z) / (3 * y + 5 * z) = 6 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_proof_l236_23652


namespace NUMINAMATH_CALUDE_hex_tile_difference_l236_23606

/-- Represents the number of tiles in a hexagonal arrangement --/
structure HexTileArrangement where
  blue : ℕ
  green : ℕ

/-- Calculates the number of tiles needed for a border around a hexagonal arrangement --/
def border_tiles (side_length : ℕ) : ℕ := 6 * side_length

/-- Adds a border of green tiles to an existing arrangement --/
def add_border (arrangement : HexTileArrangement) (border_size : ℕ) : HexTileArrangement :=
  { blue := arrangement.blue,
    green := arrangement.green + border_tiles border_size }

/-- The main theorem to prove --/
theorem hex_tile_difference :
  let initial := HexTileArrangement.mk 12 8
  let first_border := add_border initial 3
  let second_border := add_border first_border 4
  second_border.green - second_border.blue = 38 := by sorry

end NUMINAMATH_CALUDE_hex_tile_difference_l236_23606


namespace NUMINAMATH_CALUDE_cube_root_fourth_power_equals_81_l236_23615

theorem cube_root_fourth_power_equals_81 (y : ℝ) : (y^(1/3))^4 = 81 → y = 27 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_fourth_power_equals_81_l236_23615


namespace NUMINAMATH_CALUDE_constant_expression_inequality_solution_l236_23639

-- Part 1: Prove that the expression simplifies to -9 for all real x
theorem constant_expression (x : ℝ) : x * (x - 6) - (3 - x)^2 = -9 := by
  sorry

-- Part 2: Prove that the solution to the inequality is x < 5
theorem inequality_solution : 
  {x : ℝ | x - 2*(x - 3) > 1} = {x : ℝ | x < 5} := by
  sorry

end NUMINAMATH_CALUDE_constant_expression_inequality_solution_l236_23639


namespace NUMINAMATH_CALUDE_sum_of_cyclic_equations_l236_23612

theorem sum_of_cyclic_equations (p q r : ℝ) 
  (distinct : p ≠ q ∧ q ≠ r ∧ r ≠ p)
  (eq1 : q = p * (4 - p))
  (eq2 : r = q * (4 - q))
  (eq3 : p = r * (4 - r)) :
  p + q + r = 6 ∨ p + q + r = 7 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cyclic_equations_l236_23612


namespace NUMINAMATH_CALUDE_maria_car_trip_l236_23613

/-- Proves that the fraction of remaining distance traveled between first and second stops is 1/4 -/
theorem maria_car_trip (total_distance : ℝ) (remaining_after_second : ℝ) 
  (h1 : total_distance = 280)
  (h2 : remaining_after_second = 105) :
  let first_stop := total_distance / 2
  let remaining_after_first := total_distance - first_stop
  let distance_between_stops := remaining_after_first - remaining_after_second
  distance_between_stops / remaining_after_first = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_maria_car_trip_l236_23613


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l236_23660

theorem quadratic_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0) ↔ (m < -2 ∨ m > 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l236_23660


namespace NUMINAMATH_CALUDE_min_groups_for_students_l236_23603

theorem min_groups_for_students (total_students : ℕ) (max_group_size : ℕ) (h1 : total_students = 30) (h2 : max_group_size = 12) :
  ∃ (num_groups : ℕ), 
    num_groups * (total_students / num_groups) = total_students ∧
    (total_students / num_groups) ≤ max_group_size ∧
    ∀ (k : ℕ), k * (total_students / k) = total_students ∧ (total_students / k) ≤ max_group_size → k ≥ num_groups :=
by
  sorry

end NUMINAMATH_CALUDE_min_groups_for_students_l236_23603


namespace NUMINAMATH_CALUDE_square_root_of_difference_l236_23677

theorem square_root_of_difference (n : ℕ+) :
  Real.sqrt ((10^(2*n.val) - 1)/9 - 2*(10^n.val - 1)/9) = (10^n.val - 1)/3 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_difference_l236_23677


namespace NUMINAMATH_CALUDE_fortieth_number_is_twelve_l236_23630

/-- Represents the value in a specific position of the arrangement --/
def arrangementValue (position : ℕ) : ℕ :=
  let rowNum : ℕ := (position - 1).sqrt + 1
  2 * rowNum

/-- The theorem stating that the 40th number in the arrangement is 12 --/
theorem fortieth_number_is_twelve : arrangementValue 40 = 12 := by
  sorry

end NUMINAMATH_CALUDE_fortieth_number_is_twelve_l236_23630


namespace NUMINAMATH_CALUDE_all_functions_have_clever_value_point_l236_23646

-- Define the concept of a "clever value point"
def has_clever_value_point (f : ℝ → ℝ) : Prop :=
  ∃ x₀ : ℝ, f x₀ = deriv f x₀

-- State the theorem
theorem all_functions_have_clever_value_point :
  (has_clever_value_point (λ x : ℝ => x^2)) ∧
  (has_clever_value_point (λ x : ℝ => Real.exp (-x))) ∧
  (has_clever_value_point (λ x : ℝ => Real.log x)) ∧
  (has_clever_value_point (λ x : ℝ => Real.tan x)) :=
sorry

end NUMINAMATH_CALUDE_all_functions_have_clever_value_point_l236_23646


namespace NUMINAMATH_CALUDE_not_square_difference_l236_23693

/-- The square difference formula -/
def square_difference (p q : ℝ) : ℝ := p^2 - q^2

/-- Expression that cannot be directly represented by the square difference formula -/
def problematic_expression (a : ℝ) : ℝ := (a - 1) * (-a + 1)

/-- Theorem stating that the problematic expression cannot be directly represented
    by the square difference formula for any real values of p and q -/
theorem not_square_difference :
  ∀ (a p q : ℝ), problematic_expression a ≠ square_difference p q :=
by sorry

end NUMINAMATH_CALUDE_not_square_difference_l236_23693


namespace NUMINAMATH_CALUDE_garage_cars_count_l236_23654

/-- The number of cars in Connor's garage -/
def num_cars : ℕ := 10

/-- The number of bicycles in the garage -/
def num_bicycles : ℕ := 20

/-- The number of motorcycles in the garage -/
def num_motorcycles : ℕ := 5

/-- The total number of wheels in the garage -/
def total_wheels : ℕ := 90

/-- The number of wheels on a bicycle -/
def wheels_per_bicycle : ℕ := 2

/-- The number of wheels on a car -/
def wheels_per_car : ℕ := 4

/-- The number of wheels on a motorcycle -/
def wheels_per_motorcycle : ℕ := 2

theorem garage_cars_count :
  num_bicycles * wheels_per_bicycle +
  num_cars * wheels_per_car +
  num_motorcycles * wheels_per_motorcycle = total_wheels :=
by sorry

end NUMINAMATH_CALUDE_garage_cars_count_l236_23654


namespace NUMINAMATH_CALUDE_abs_opposite_equal_l236_23692

theorem abs_opposite_equal (x : ℝ) : |x| = |-x| := by sorry

end NUMINAMATH_CALUDE_abs_opposite_equal_l236_23692
