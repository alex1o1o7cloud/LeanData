import Mathlib

namespace NUMINAMATH_CALUDE_competition_result_count_l3209_320993

/-- Represents a team's score composition -/
structure TeamScore where
  threes : ℕ  -- number of 3-point problems solved
  fives  : ℕ  -- number of 5-point problems solved

/-- Calculates the total score for a team -/
def totalScore (t : TeamScore) : ℕ := 3 * t.threes + 5 * t.fives

/-- Represents the scores of all three teams -/
structure CompetitionResult where
  team1 : TeamScore
  team2 : TeamScore
  team3 : TeamScore

/-- Checks if a competition result is valid -/
def isValidResult (r : CompetitionResult) : Prop :=
  totalScore r.team1 + totalScore r.team2 + totalScore r.team3 = 32

/-- Counts the number of valid competition results -/
def countValidResults : ℕ := sorry

theorem competition_result_count :
  countValidResults = 255 := by sorry

end NUMINAMATH_CALUDE_competition_result_count_l3209_320993


namespace NUMINAMATH_CALUDE_library_books_count_l3209_320939

/-- The number of books in a library after two years of purchases -/
def library_books (initial_books : ℕ) (books_last_year : ℕ) (multiplier : ℕ) : ℕ :=
  initial_books + books_last_year + multiplier * books_last_year

/-- Theorem stating that the library now has 300 books -/
theorem library_books_count : library_books 100 50 3 = 300 := by
  sorry

end NUMINAMATH_CALUDE_library_books_count_l3209_320939


namespace NUMINAMATH_CALUDE_randy_walks_dog_twice_daily_l3209_320972

/-- The number of times Randy walks his dog per day -/
def walks_per_day (wipes_per_pack : ℕ) (packs_for_360_days : ℕ) : ℕ :=
  (wipes_per_pack * packs_for_360_days) / 360

theorem randy_walks_dog_twice_daily :
  walks_per_day 120 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_randy_walks_dog_twice_daily_l3209_320972


namespace NUMINAMATH_CALUDE_sum_of_3rd_4th_5th_terms_l3209_320928

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ n, a (n + 1) = r * a n

theorem sum_of_3rd_4th_5th_terms
  (a : ℕ → ℝ)
  (h_positive : ∀ n, a n > 0)
  (h_geometric : geometric_sequence a)
  (h_ratio : ∃ (r : ℝ), ∀ n, a (n + 1) = 2 * a n)
  (h_sum_first_three : a 1 + a 2 + a 3 = 21) :
  a 3 + a 4 + a 5 = 84 :=
sorry

end NUMINAMATH_CALUDE_sum_of_3rd_4th_5th_terms_l3209_320928


namespace NUMINAMATH_CALUDE_root_minus_one_quadratic_equation_l3209_320904

theorem root_minus_one_quadratic_equation (p : ℚ) :
  (∀ x, (2*p - 1) * x^2 + 2*(1 - p) * x + 3*p = 0 ↔ x = -1) ↔ p = 3/7 := by
  sorry

end NUMINAMATH_CALUDE_root_minus_one_quadratic_equation_l3209_320904


namespace NUMINAMATH_CALUDE_range_of_expression_l3209_320990

theorem range_of_expression (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 ≤ β ∧ β ≤ π/2) :
  -π/6 < 2*α - β/3 ∧ 2*α - β/3 < π := by
  sorry

end NUMINAMATH_CALUDE_range_of_expression_l3209_320990


namespace NUMINAMATH_CALUDE_random_selection_probability_l3209_320936

theorem random_selection_probability (m : ℝ) : 
  m > -1 →
  (1 - (-1)) / (m - (-1)) = 2/5 →
  m = 4 := by
sorry

end NUMINAMATH_CALUDE_random_selection_probability_l3209_320936


namespace NUMINAMATH_CALUDE_m_equals_one_sufficient_not_necessary_for_abs_m_equals_one_l3209_320938

theorem m_equals_one_sufficient_not_necessary_for_abs_m_equals_one :
  (∀ m : ℝ, m = 1 → |m| = 1) ∧
  (∃ m : ℝ, |m| = 1 ∧ m ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_m_equals_one_sufficient_not_necessary_for_abs_m_equals_one_l3209_320938


namespace NUMINAMATH_CALUDE_shopping_trip_cost_theorem_l3209_320922

/-- Calculates the total cost of James' shopping trip -/
def shopping_trip_cost : ℝ :=
  let milk_price : ℝ := 4.50
  let milk_tax_rate : ℝ := 0.20
  let bananas_price : ℝ := 3.00
  let bananas_tax_rate : ℝ := 0.15
  let baguette_price : ℝ := 2.50
  let cereal_price : ℝ := 6.00
  let cereal_discount : ℝ := 0.20
  let cereal_tax_rate : ℝ := 0.12
  let eggs_price : ℝ := 3.50
  let eggs_coupon : ℝ := 1.00
  let eggs_tax_rate : ℝ := 0.18

  let milk_total := milk_price * (1 + milk_tax_rate)
  let bananas_total := bananas_price * (1 + bananas_tax_rate)
  let baguette_total := baguette_price
  let cereal_discounted := cereal_price * (1 - cereal_discount)
  let cereal_total := cereal_discounted * (1 + cereal_tax_rate)
  let eggs_discounted := eggs_price - eggs_coupon
  let eggs_total := eggs_discounted * (1 + eggs_tax_rate)

  milk_total + bananas_total + baguette_total + cereal_total + eggs_total

theorem shopping_trip_cost_theorem : shopping_trip_cost = 19.68 := by
  sorry

end NUMINAMATH_CALUDE_shopping_trip_cost_theorem_l3209_320922


namespace NUMINAMATH_CALUDE_marble_probability_l3209_320935

def total_marbles : ℕ := 15
def green_marbles : ℕ := 8
def purple_marbles : ℕ := 7
def trials : ℕ := 6
def green_choices : ℕ := 3

def prob_green : ℚ := green_marbles / total_marbles
def prob_purple : ℚ := purple_marbles / total_marbles

theorem marble_probability : 
  (Nat.choose trials green_choices : ℚ) * 
  (prob_green ^ green_choices) * 
  (prob_purple ^ (trials - green_choices)) * 
  prob_purple = 4913248/34171875 := by sorry

end NUMINAMATH_CALUDE_marble_probability_l3209_320935


namespace NUMINAMATH_CALUDE_y_equal_y_greater_l3209_320966

-- Define the functions y₁ and y₂
def y₁ (x : ℝ) : ℝ := -x + 3
def y₂ (x : ℝ) : ℝ := 2 + x

-- Theorem 1: y₁ = y₂ when x = 1/2
theorem y_equal (x : ℝ) : y₁ x = y₂ x ↔ x = 1/2 := by sorry

-- Theorem 2: y₁ = 2y₂ + 5 when x = -2
theorem y_greater (x : ℝ) : y₁ x = 2 * y₂ x + 5 ↔ x = -2 := by sorry

end NUMINAMATH_CALUDE_y_equal_y_greater_l3209_320966


namespace NUMINAMATH_CALUDE_factorization_proof_l3209_320918

theorem factorization_proof (m : ℝ) : 4 - m^2 = (2 + m) * (2 - m) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l3209_320918


namespace NUMINAMATH_CALUDE_monotonic_increase_intervals_l3209_320919

/-- The function f(x) = x³ - 3x -/
def f (x : ℝ) : ℝ := x^3 - 3*x

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 - 3

theorem monotonic_increase_intervals (x : ℝ) :
  StrictMonoOn f (Set.Iio (-1)) ∧ StrictMonoOn f (Set.Ioi 1) :=
sorry

end NUMINAMATH_CALUDE_monotonic_increase_intervals_l3209_320919


namespace NUMINAMATH_CALUDE_specific_square_figure_perimeter_l3209_320976

/-- A figure composed of squares arranged in a specific pattern -/
structure SquareFigure where
  squareSideLength : ℝ
  horizontalSegments : ℕ
  verticalSegments : ℕ

/-- The perimeter of a SquareFigure -/
def perimeter (f : SquareFigure) : ℝ :=
  (f.horizontalSegments + f.verticalSegments) * f.squareSideLength * 2

/-- Theorem stating that the perimeter of the specific square figure is 52 -/
theorem specific_square_figure_perimeter :
  ∃ (f : SquareFigure),
    f.squareSideLength = 2 ∧
    f.horizontalSegments = 16 ∧
    f.verticalSegments = 10 ∧
    perimeter f = 52 := by
  sorry

end NUMINAMATH_CALUDE_specific_square_figure_perimeter_l3209_320976


namespace NUMINAMATH_CALUDE_expression_simplification_l3209_320971

theorem expression_simplification (x y : ℚ) 
  (hx : x = 1/2) (hy : y = 8) : 
  y * (5 * x - 4 * y) + (x - 2 * y)^2 = 17/4 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3209_320971


namespace NUMINAMATH_CALUDE_rick_ironed_45_pieces_l3209_320989

/-- Represents Rick's ironing rates and time spent ironing --/
structure IroningData where
  weekday_shirt_rate : ℕ
  weekday_pants_rate : ℕ
  weekday_jacket_rate : ℕ
  weekend_shirt_rate : ℕ
  weekend_pants_rate : ℕ
  weekend_jacket_rate : ℕ
  weekday_shirt_time : ℕ
  weekday_pants_time : ℕ
  weekday_jacket_time : ℕ
  weekend_shirt_time : ℕ
  weekend_pants_time : ℕ
  weekend_jacket_time : ℕ

/-- Calculates the total number of pieces of clothing ironed --/
def total_ironed (data : IroningData) : ℕ :=
  (data.weekday_shirt_rate * data.weekday_shirt_time + data.weekend_shirt_rate * data.weekend_shirt_time) +
  (data.weekday_pants_rate * data.weekday_pants_time + data.weekend_pants_rate * data.weekend_pants_time) +
  (data.weekday_jacket_rate * data.weekday_jacket_time + data.weekend_jacket_rate * data.weekend_jacket_time)

/-- Theorem stating that Rick irons 45 pieces of clothing given the specified rates and times --/
theorem rick_ironed_45_pieces : 
  ∀ (data : IroningData), 
    data.weekday_shirt_rate = 4 ∧ 
    data.weekday_pants_rate = 3 ∧ 
    data.weekday_jacket_rate = 2 ∧
    data.weekend_shirt_rate = 5 ∧ 
    data.weekend_pants_rate = 4 ∧ 
    data.weekend_jacket_rate = 3 ∧
    data.weekday_shirt_time = 2 ∧ 
    data.weekday_pants_time = 3 ∧ 
    data.weekday_jacket_time = 1 ∧
    data.weekend_shirt_time = 3 ∧ 
    data.weekend_pants_time = 2 ∧ 
    data.weekend_jacket_time = 1 
    → total_ironed data = 45 := by
  sorry

end NUMINAMATH_CALUDE_rick_ironed_45_pieces_l3209_320989


namespace NUMINAMATH_CALUDE_symmetry_of_f_l3209_320916

-- Define a function f from real numbers to real numbers
variable (f : ℝ → ℝ)

-- Define the condition given in the problem
axiom functional_equation : ∀ x : ℝ, f (x + 5) = f (9 - x)

-- State the theorem to be proved
theorem symmetry_of_f : 
  (∀ x : ℝ, f (7 + x) = f (7 - x)) := by sorry

end NUMINAMATH_CALUDE_symmetry_of_f_l3209_320916


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l3209_320923

theorem simplify_and_rationalize : 
  (Real.sqrt 3 / Real.sqrt 4) * (Real.sqrt 5 / Real.sqrt 6) * (Real.sqrt 8 / Real.sqrt 9) = Real.sqrt 15 / 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l3209_320923


namespace NUMINAMATH_CALUDE_A_characterization_and_inequality_l3209_320933

def f (x : ℝ) : ℝ := |2*x + 1| + |x - 2|

def A : Set ℝ := {x | f x < 3}

theorem A_characterization_and_inequality :
  (A = {x : ℝ | -2/3 < x ∧ x < 0}) ∧
  (∀ s t : ℝ, s ∈ A → t ∈ A → |1 - t/s| < |t - 1/s|) := by sorry

end NUMINAMATH_CALUDE_A_characterization_and_inequality_l3209_320933


namespace NUMINAMATH_CALUDE_george_has_twelve_blocks_l3209_320929

/-- The number of blocks George has -/
def georgesBlocks (numBoxes : ℕ) (blocksPerBox : ℕ) : ℕ :=
  numBoxes * blocksPerBox

/-- Theorem: George has 12 blocks given 2 boxes with 6 blocks each -/
theorem george_has_twelve_blocks :
  georgesBlocks 2 6 = 12 := by
  sorry

end NUMINAMATH_CALUDE_george_has_twelve_blocks_l3209_320929


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3209_320945

theorem polynomial_factorization :
  ∀ x : ℝ, (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 7*x + 1) * (x^2 + 4*x + 7) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3209_320945


namespace NUMINAMATH_CALUDE_notebook_distribution_l3209_320920

/-- Proves that the ratio of notebooks per child to the number of children is 1:8 
    given the conditions in the problem. -/
theorem notebook_distribution (C : ℕ) (N : ℚ) : 
  (∃ (k : ℕ), N = k / C) →  -- Number of notebooks each child got is a fraction of number of children
  (16 = 2 * k / C) →        -- If number of children halved, each would get 16 notebooks
  (C * N = 512) →           -- Total notebooks distributed is 512
  N / C = 1 / 8 :=          -- Ratio of notebooks per child to number of children is 1:8
by sorry

end NUMINAMATH_CALUDE_notebook_distribution_l3209_320920


namespace NUMINAMATH_CALUDE_binary_10011_equals_19_l3209_320955

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldr (fun (i, bit) acc => acc + if bit then 2^i else 0) 0

theorem binary_10011_equals_19 : 
  binary_to_decimal [true, false, false, true, true] = 19 := by
  sorry

end NUMINAMATH_CALUDE_binary_10011_equals_19_l3209_320955


namespace NUMINAMATH_CALUDE_new_shoes_cost_l3209_320948

theorem new_shoes_cost (repair_cost : ℝ) (repair_duration : ℝ) (new_duration : ℝ) (percentage_increase : ℝ) :
  repair_cost = 14.50 →
  repair_duration = 1 →
  new_duration = 2 →
  percentage_increase = 0.10344827586206897 →
  ∃ (new_cost : ℝ), new_cost / new_duration = repair_cost / repair_duration * (1 + percentage_increase) ∧ new_cost = 32 :=
by sorry

end NUMINAMATH_CALUDE_new_shoes_cost_l3209_320948


namespace NUMINAMATH_CALUDE_hotel_room_charges_l3209_320991

theorem hotel_room_charges (P R G : ℝ) 
  (h1 : P = R - 0.5 * R) 
  (h2 : P = G - 0.1 * G) : 
  R = 1.8 * G := by
sorry

end NUMINAMATH_CALUDE_hotel_room_charges_l3209_320991


namespace NUMINAMATH_CALUDE_al_karhi_square_root_approximation_l3209_320937

theorem al_karhi_square_root_approximation 
  (N a r : ℝ) 
  (h1 : N > 0) 
  (h2 : a > 0) 
  (h3 : a^2 ≤ N) 
  (h4 : (a+1)^2 > N) 
  (h5 : r = N - a^2) 
  (h6 : r < 2*a + 1) : 
  ∃ (ε : ℝ), ε > 0 ∧ |Real.sqrt N - (a + r / (2*a + 1))| < ε :=
sorry

end NUMINAMATH_CALUDE_al_karhi_square_root_approximation_l3209_320937


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l3209_320925

theorem sum_of_solutions_quadratic (x : ℝ) : 
  let equation := -16 * x^2 + 72 * x - 108
  let sum_of_roots := -72 / (-16)
  equation = 0 → sum_of_roots = 9/2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l3209_320925


namespace NUMINAMATH_CALUDE_new_person_weight_l3209_320914

theorem new_person_weight (original_count : ℕ) (original_average : ℝ) (leaving_weight : ℝ) (average_increase : ℝ) :
  original_count = 20 →
  leaving_weight = 92 →
  average_increase = 4.5 →
  (original_count * (original_average + average_increase) - (original_count - 1) * original_average) = 182 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l3209_320914


namespace NUMINAMATH_CALUDE_f_72_value_l3209_320924

/-- A function satisfying f(ab) = f(a) + f(b) for all a and b -/
def MultiplicativeToAdditive (f : ℕ → ℝ) : Prop :=
  ∀ a b : ℕ, f (a * b) = f a + f b

/-- The main theorem -/
theorem f_72_value (f : ℕ → ℝ) (p q : ℝ) 
    (h1 : MultiplicativeToAdditive f) 
    (h2 : f 2 = p) 
    (h3 : f 3 = q) : 
  f 72 = 3 * p + 2 * q := by
  sorry

end NUMINAMATH_CALUDE_f_72_value_l3209_320924


namespace NUMINAMATH_CALUDE_smallest_k_for_congruence_l3209_320978

theorem smallest_k_for_congruence : 
  (∃ k : ℕ, k > 0 ∧ (201 + k) % (24 + k) = (9 + k) % (24 + k) ∧
    ∀ m : ℕ, m > 0 ∧ m < k → (201 + m) % (24 + m) ≠ (9 + m) % (24 + m)) ∧
  201 % 24 = 9 % 24 →
  (∃ k : ℕ, k = 8 ∧ k > 0 ∧ (201 + k) % (24 + k) = (9 + k) % (24 + k) ∧
    ∀ m : ℕ, m > 0 ∧ m < k → (201 + m) % (24 + m) ≠ (9 + m) % (24 + m)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_congruence_l3209_320978


namespace NUMINAMATH_CALUDE_division_problem_l3209_320950

theorem division_problem :
  ∃ (quotient : ℕ), 136 = 15 * quotient + 1 ∧ quotient = 9 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3209_320950


namespace NUMINAMATH_CALUDE_jill_earnings_l3209_320909

/-- Calculates the total earnings of a waitress given her work conditions --/
def waitress_earnings (hourly_wage : ℝ) (tip_rate : ℝ) (shifts : ℕ) (hours_per_shift : ℕ) (average_orders_per_hour : ℝ) : ℝ :=
  let total_hours : ℝ := shifts * hours_per_shift
  let wage_earnings : ℝ := total_hours * hourly_wage
  let total_orders : ℝ := total_hours * average_orders_per_hour
  let tip_earnings : ℝ := tip_rate * total_orders
  wage_earnings + tip_earnings

/-- Theorem stating that Jill's earnings for the week are $240.00 --/
theorem jill_earnings :
  waitress_earnings 4 0.15 3 8 40 = 240 := by
  sorry

end NUMINAMATH_CALUDE_jill_earnings_l3209_320909


namespace NUMINAMATH_CALUDE_john_smith_payment_l3209_320961

def number_of_cakes : ℕ := 3
def cost_per_cake : ℕ := 12
def number_of_people_splitting_cost : ℕ := 2

theorem john_smith_payment (total_cost : ℕ) (johns_share : ℕ) : 
  total_cost = number_of_cakes * cost_per_cake →
  johns_share = total_cost / number_of_people_splitting_cost →
  johns_share = 18 := by
sorry

end NUMINAMATH_CALUDE_john_smith_payment_l3209_320961


namespace NUMINAMATH_CALUDE_distance_XY_is_80_l3209_320973

/-- The distance from X to Y in miles. -/
def distance_XY : ℝ := 80

/-- Yolanda's walking rate in miles per hour. -/
def yolanda_rate : ℝ := 8

/-- Bob's walking rate in miles per hour. -/
def bob_rate : ℝ := 9

/-- The distance Bob walked when they met, in miles. -/
def bob_distance : ℝ := 38.11764705882353

/-- The time difference between Yolanda and Bob's start times, in hours. -/
def time_difference : ℝ := 1

theorem distance_XY_is_80 :
  distance_XY = yolanda_rate * (time_difference + bob_distance / bob_rate) + bob_distance :=
sorry

end NUMINAMATH_CALUDE_distance_XY_is_80_l3209_320973


namespace NUMINAMATH_CALUDE_smallest_number_divisibility_l3209_320931

theorem smallest_number_divisibility (n : ℕ) : 
  (∀ m : ℕ, m < 551245 → ¬(∃ k₁ k₂ k₃ k₄ k₅ : ℕ, 
    m + 5 = 9 * k₁ ∧ 
    m + 5 = 70 * k₂ ∧ 
    m + 5 = 25 * k₃ ∧ 
    m + 5 = 21 * k₄ ∧ 
    m + 5 = 49 * k₅)) ∧ 
  (∃ k₁ k₂ k₃ k₄ k₅ : ℕ, 
    551245 + 5 = 9 * k₁ ∧ 
    551245 + 5 = 70 * k₂ ∧ 
    551245 + 5 = 25 * k₃ ∧ 
    551245 + 5 = 21 * k₄ ∧ 
    551245 + 5 = 49 * k₅) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisibility_l3209_320931


namespace NUMINAMATH_CALUDE_f_g_deriv_signs_l3209_320996

-- Define f and g as real-valued functions
variable (f g : ℝ → ℝ)

-- Define the conditions
axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom g_even : ∀ x : ℝ, g (-x) = g x
axiom f_deriv_pos : ∀ x : ℝ, x > 0 → deriv f x > 0
axiom g_deriv_pos : ∀ x : ℝ, x > 0 → deriv g x > 0

-- State the theorem
theorem f_g_deriv_signs :
  ∀ x : ℝ, x < 0 → deriv f x > 0 ∧ deriv g x < 0 :=
sorry

end NUMINAMATH_CALUDE_f_g_deriv_signs_l3209_320996


namespace NUMINAMATH_CALUDE_strictly_increasing_function_inequality_l3209_320926

theorem strictly_increasing_function_inequality (k : ℕ) (f : ℕ → ℕ)
  (h_increasing : ∀ m n, m < n → f m < f n)
  (h_composite : ∀ n, f (f n) = k * n) :
  ∀ n : ℕ, n ≠ 0 → (2 * k * n) / (k + 1) ≤ f n ∧ f n ≤ (k + 1) * n / 2 :=
by sorry

end NUMINAMATH_CALUDE_strictly_increasing_function_inequality_l3209_320926


namespace NUMINAMATH_CALUDE_julie_count_correct_l3209_320908

/-- Represents the number of people with a given name in the crowd -/
structure NameCount where
  barry : ℕ
  kevin : ℕ
  julie : ℕ
  joe : ℕ

/-- Represents the proportion of nice people for each name -/
structure NiceProportion where
  barry : ℚ
  kevin : ℚ
  julie : ℚ
  joe : ℚ

/-- The total number of nice people in the crowd -/
def totalNicePeople : ℕ := 99

/-- The actual count of people with each name -/
def actualCount : NameCount where
  barry := 24
  kevin := 20
  julie := 80  -- This is what we want to prove
  joe := 50

/-- The proportion of nice people for each name -/
def niceProportion : NiceProportion where
  barry := 1
  kevin := 1/2
  julie := 3/4
  joe := 1/10

/-- Calculates the number of nice people for a given name -/
def niceCount (count : ℕ) (proportion : ℚ) : ℚ :=
  (count : ℚ) * proportion

/-- Theorem stating that the number of people named Julie is correct -/
theorem julie_count_correct :
  actualCount.julie = 80 ∧
  (niceCount actualCount.barry niceProportion.barry +
   niceCount actualCount.kevin niceProportion.kevin +
   niceCount actualCount.julie niceProportion.julie +
   niceCount actualCount.joe niceProportion.joe : ℚ) = totalNicePeople :=
by sorry

end NUMINAMATH_CALUDE_julie_count_correct_l3209_320908


namespace NUMINAMATH_CALUDE_triangle_with_consecutive_sides_and_obtuse_angle_l3209_320917

-- Define a triangle with sides of consecutive natural numbers
def ConsecutiveSidedTriangle (a b c : ℕ) : Prop :=
  (b = a + 1) ∧ (c = b + 1) ∧ (a > 0)

-- Define the condition for the largest angle to be obtuse
def HasObtuseAngle (a b c : ℕ) : Prop :=
  let cosLargestAngle := (a^2 + b^2 - c^2) / (2 * a * b)
  cosLargestAngle < 0

-- Theorem statement
theorem triangle_with_consecutive_sides_and_obtuse_angle
  (a b c : ℕ) (h1 : ConsecutiveSidedTriangle a b c) (h2 : HasObtuseAngle a b c) :
  (a = 2 ∧ b = 3 ∧ c = 4) :=
sorry

end NUMINAMATH_CALUDE_triangle_with_consecutive_sides_and_obtuse_angle_l3209_320917


namespace NUMINAMATH_CALUDE_incorrect_height_calculation_l3209_320963

theorem incorrect_height_calculation (n : ℕ) (initial_avg real_avg actual_height : ℝ) 
  (h1 : n = 20)
  (h2 : initial_avg = 175)
  (h3 : real_avg = 174.25)
  (h4 : actual_height = 136) :
  ∃ (incorrect_height : ℝ),
    incorrect_height = n * initial_avg - (n * real_avg - actual_height) ∧
    incorrect_height = 151 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_height_calculation_l3209_320963


namespace NUMINAMATH_CALUDE_stream_speed_l3209_320902

/-- Proves that the speed of a stream is 8 kmph given the conditions of the boat's travel -/
theorem stream_speed (boat_speed : ℝ) (downstream_distance : ℝ) (upstream_distance : ℝ) 
  (h1 : boat_speed = 24)
  (h2 : downstream_distance = 64)
  (h3 : upstream_distance = 32)
  (h4 : downstream_distance / (boat_speed + x) = upstream_distance / (boat_speed - x)) :
  x = 8 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l3209_320902


namespace NUMINAMATH_CALUDE_tank_capacity_l3209_320932

theorem tank_capacity (num_trucks : ℕ) (tanks_per_truck : ℕ) (total_capacity : ℕ) : 
  num_trucks = 3 → 
  tanks_per_truck = 3 → 
  total_capacity = 1350 → 
  (total_capacity / (num_trucks * tanks_per_truck) : ℚ) = 150 := by
sorry

end NUMINAMATH_CALUDE_tank_capacity_l3209_320932


namespace NUMINAMATH_CALUDE_symmetry_of_sine_function_l3209_320927

/-- Given a function f(x) = sin(wx + π/4) where w > 0 and 
    the minimum positive period of f(x) is π, 
    prove that the graph of f(x) is symmetrical about the line x = π/8 -/
theorem symmetry_of_sine_function (w : ℝ) (h1 : w > 0) :
  let f : ℝ → ℝ := λ x ↦ Real.sin (w * x + π / 4)
  (∀ x : ℝ, f (x + π) = f x) →  -- minimum positive period is π
  ∀ x : ℝ, f (π / 4 - x) = f (π / 4 + x) := by
sorry

end NUMINAMATH_CALUDE_symmetry_of_sine_function_l3209_320927


namespace NUMINAMATH_CALUDE_abc_value_l3209_320997

theorem abc_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (eq1 : a * (b + c) = 156)
  (eq2 : b * (c + a) = 168)
  (eq3 : c * (a + b) = 180) :
  a * b * c = 288 * Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_abc_value_l3209_320997


namespace NUMINAMATH_CALUDE_walking_rate_ratio_l3209_320940

theorem walking_rate_ratio (usual_time new_time distance : ℝ) 
  (h1 : usual_time = 36)
  (h2 : new_time = usual_time - 4)
  (h3 : distance > 0)
  (h4 : usual_time > 0)
  (h5 : new_time > 0) :
  (distance / new_time) / (distance / usual_time) = 9 / 8 := by
sorry

end NUMINAMATH_CALUDE_walking_rate_ratio_l3209_320940


namespace NUMINAMATH_CALUDE_thirteen_sided_polygon_property_n_sided_polygon_property_l3209_320905

-- Define a polygon type
structure Polygon :=
  (sides : ℕ)
  (vertices : Fin sides → ℝ × ℝ)

-- Define a line type
structure Line :=
  (a b c : ℝ)

-- Function to check if a line contains a side of a polygon
def line_contains_side (l : Line) (p : Polygon) (i : Fin p.sides) : Prop :=
  -- Implementation details omitted
  sorry

-- Function to count how many sides of a polygon a line contains
def count_sides_on_line (l : Line) (p : Polygon) : ℕ :=
  -- Implementation details omitted
  sorry

-- Theorem for 13-sided polygons
theorem thirteen_sided_polygon_property :
  ∀ (p : Polygon), p.sides = 13 →
  ∃ (l : Line), ∃ (i : Fin p.sides),
    line_contains_side l p i ∧
    count_sides_on_line l p = 1 :=
sorry

-- Theorem for n-sided polygons where n > 13
theorem n_sided_polygon_property :
  ∀ (n : ℕ), n > 13 →
  ∃ (p : Polygon), p.sides = n ∧
  ∀ (l : Line), ∀ (i : Fin p.sides),
    line_contains_side l p i →
    count_sides_on_line l p ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_thirteen_sided_polygon_property_n_sided_polygon_property_l3209_320905


namespace NUMINAMATH_CALUDE_towel_shrinkage_l3209_320951

theorem towel_shrinkage (L B : ℝ) (h_positive : L > 0 ∧ B > 0) : 
  let new_length := 0.8 * L
  let new_area := 0.72 * (L * B)
  ∃ (new_breadth : ℝ), new_area = new_length * new_breadth ∧ new_breadth = 0.9 * B :=
sorry

end NUMINAMATH_CALUDE_towel_shrinkage_l3209_320951


namespace NUMINAMATH_CALUDE_root_value_theorem_l3209_320964

theorem root_value_theorem (a : ℝ) (h : a^2 + 2*a - 1 = 0) : -a^2 - 2*a + 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_root_value_theorem_l3209_320964


namespace NUMINAMATH_CALUDE_arithmetic_progression_product_not_square_l3209_320958

/-- Four distinct positive integers in arithmetic progression -/
structure ArithmeticProgression :=
  (a : ℕ+) -- First term
  (r : ℕ+) -- Common difference
  (distinct : a < a + r ∧ a + r < a + 2*r ∧ a + 2*r < a + 3*r)

/-- The product of four terms in arithmetic progression is not a perfect square -/
theorem arithmetic_progression_product_not_square (ap : ArithmeticProgression) :
  ¬ ∃ (m : ℕ), (ap.a * (ap.a + ap.r) * (ap.a + 2*ap.r) * (ap.a + 3*ap.r) : ℕ) = m^2 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_progression_product_not_square_l3209_320958


namespace NUMINAMATH_CALUDE_distance_to_centroid_l3209_320995

-- Define a triangle by its side lengths
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

-- Define a point inside the triangle by its distances from the vertices
structure InnerPoint (t : Triangle) where
  p : ℝ
  q : ℝ
  r : ℝ
  pos_p : 0 < p
  pos_q : 0 < q
  pos_r : 0 < r

-- Theorem statement
theorem distance_to_centroid (t : Triangle) (d : InnerPoint t) :
  ∃ (ds : ℝ), ds^2 = (3 * (d.p^2 + d.q^2 + d.r^2) - (t.a^2 + t.b^2 + t.c^2)) / 9 :=
sorry

end NUMINAMATH_CALUDE_distance_to_centroid_l3209_320995


namespace NUMINAMATH_CALUDE_oranges_picked_total_l3209_320934

/-- The number of oranges Joan picked -/
def joan_oranges : ℕ := 37

/-- The number of oranges Sara picked -/
def sara_oranges : ℕ := 10

/-- The total number of oranges picked -/
def total_oranges : ℕ := joan_oranges + sara_oranges

theorem oranges_picked_total :
  total_oranges = 47 := by sorry

end NUMINAMATH_CALUDE_oranges_picked_total_l3209_320934


namespace NUMINAMATH_CALUDE_sum_a_c_l3209_320915

theorem sum_a_c (a b c d : ℝ) 
  (h1 : a * b + b * c + c * d + d * a = 42)
  (h2 : b + d = 6) : 
  a + c = 7 := by sorry

end NUMINAMATH_CALUDE_sum_a_c_l3209_320915


namespace NUMINAMATH_CALUDE_p_satisfies_conditions_l3209_320900

/-- A cubic polynomial p(x) satisfying specific conditions -/
def p (x : ℝ) : ℝ := -x^3 + 4*x^2 - 7*x - 3

/-- Theorem stating that p(x) satisfies the given conditions -/
theorem p_satisfies_conditions :
  p 1 = -7 ∧ p 2 = -9 ∧ p 3 = -15 ∧ p 4 = -31 := by
  sorry

#eval p 1
#eval p 2
#eval p 3
#eval p 4

end NUMINAMATH_CALUDE_p_satisfies_conditions_l3209_320900


namespace NUMINAMATH_CALUDE_maximum_marks_proof_l3209_320967

/-- Given that a student needs 33% of total marks to pass, got 59 marks, and failed by 40 marks,
    prove that the maximum marks are 300. -/
theorem maximum_marks_proof (pass_percentage : Real) (obtained_marks : ℕ) (failing_margin : ℕ) :
  pass_percentage = 0.33 →
  obtained_marks = 59 →
  failing_margin = 40 →
  ∃ (max_marks : ℕ), max_marks = 300 ∧ pass_percentage * max_marks = obtained_marks + failing_margin :=
by sorry

end NUMINAMATH_CALUDE_maximum_marks_proof_l3209_320967


namespace NUMINAMATH_CALUDE_percentage_passed_all_subjects_l3209_320974

theorem percentage_passed_all_subjects 
  (fail_hindi : Real) 
  (fail_english : Real) 
  (fail_both : Real) 
  (fail_math : Real) 
  (h1 : fail_hindi = 0.2) 
  (h2 : fail_english = 0.7) 
  (h3 : fail_both = 0.1) 
  (h4 : fail_math = 0.5) : 
  (1 - (fail_hindi + fail_english - fail_both)) * (1 - fail_math) = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_percentage_passed_all_subjects_l3209_320974


namespace NUMINAMATH_CALUDE_carpet_shaded_area_l3209_320970

theorem carpet_shaded_area (S T : ℝ) : 
  12 / S = 4 →
  S / T = 2 →
  S > 0 →
  T > 0 →
  S^2 + 8 * T^2 = 27 := by
sorry

end NUMINAMATH_CALUDE_carpet_shaded_area_l3209_320970


namespace NUMINAMATH_CALUDE_club_president_vicepresident_selection_l3209_320965

theorem club_president_vicepresident_selection (total_members : ℕ) (boys : ℕ) (girls : ℕ) 
  (h1 : total_members = 30)
  (h2 : boys = 18)
  (h3 : girls = 12)
  (h4 : total_members = boys + girls) :
  (boys * (total_members - 1)) = 522 := by
  sorry

end NUMINAMATH_CALUDE_club_president_vicepresident_selection_l3209_320965


namespace NUMINAMATH_CALUDE_inner_triangle_perimeter_is_270_l3209_320957

/-- Triangle ABC with given side lengths and parallel lines -/
structure TriangleWithParallels where
  -- Side lengths of triangle ABC
  AB : ℝ
  BC : ℝ
  AC : ℝ
  -- Lengths of segments formed by parallel lines
  ℓA_segment : ℝ
  ℓB_segment : ℝ
  ℓC_segment : ℝ
  -- Conditions
  AB_positive : AB > 0
  BC_positive : BC > 0
  AC_positive : AC > 0
  ℓA_segment_positive : ℓA_segment > 0
  ℓB_segment_positive : ℓB_segment > 0
  ℓC_segment_positive : ℓC_segment > 0
  AB_eq : AB = 150
  BC_eq : BC = 270
  AC_eq : AC = 210
  ℓA_segment_eq : ℓA_segment = 60
  ℓB_segment_eq : ℓB_segment = 50
  ℓC_segment_eq : ℓC_segment = 20

/-- The perimeter of the inner triangle formed by parallel lines -/
def innerTrianglePerimeter (t : TriangleWithParallels) : ℝ := sorry

/-- Theorem: The perimeter of the inner triangle is 270 -/
theorem inner_triangle_perimeter_is_270 (t : TriangleWithParallels) :
  innerTrianglePerimeter t = 270 := by
  sorry

end NUMINAMATH_CALUDE_inner_triangle_perimeter_is_270_l3209_320957


namespace NUMINAMATH_CALUDE_females_with_advanced_degrees_l3209_320988

theorem females_with_advanced_degrees 
  (total_employees : ℕ) 
  (total_females : ℕ) 
  (employees_with_advanced_degrees : ℕ) 
  (males_with_college_degree_only : ℕ) 
  (h1 : total_employees = 180) 
  (h2 : total_females = 110) 
  (h3 : employees_with_advanced_degrees = 90) 
  (h4 : males_with_college_degree_only = 35) :
  total_females - (total_employees - employees_with_advanced_degrees) + 
  (employees_with_advanced_degrees - (total_employees - total_females - males_with_college_degree_only)) = 55 :=
by sorry

end NUMINAMATH_CALUDE_females_with_advanced_degrees_l3209_320988


namespace NUMINAMATH_CALUDE_square_of_negative_square_l3209_320954

theorem square_of_negative_square (m : ℝ) : (-m^2)^2 = m^4 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_square_l3209_320954


namespace NUMINAMATH_CALUDE_water_added_proof_l3209_320981

def container_problem (capacity : ℝ) (initial_fill : ℝ) (final_fill : ℝ) : Prop :=
  let initial_volume := capacity * initial_fill
  let final_volume := capacity * final_fill
  final_volume - initial_volume = 20

theorem water_added_proof :
  container_problem 80 0.5 0.75 :=
sorry

end NUMINAMATH_CALUDE_water_added_proof_l3209_320981


namespace NUMINAMATH_CALUDE_gcd_of_B_is_two_l3209_320942

def B : Set ℕ := {n | ∃ x : ℕ, n = (x - 1) + x + (x + 1) + (x + 2) ∧ x > 0}

theorem gcd_of_B_is_two :
  ∃ d : ℕ, d > 0 ∧ (∀ n ∈ B, d ∣ n) ∧ (∀ m : ℕ, (∀ n ∈ B, m ∣ n) → m ≤ d) ∧ d = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_B_is_two_l3209_320942


namespace NUMINAMATH_CALUDE_chord_length_l3209_320913

-- Define the circle C
def circle_C (x y a : ℝ) : Prop :=
  x^2 + y^2 - 2*a*x + 4*a*y + 5*a^2 - 25 = 0

-- Define line l₁
def line_l1 (x y : ℝ) : Prop :=
  x + y + 2 = 0

-- Define line l₂
def line_l2 (x y : ℝ) : Prop :=
  3*x + 4*y - 5 = 0

-- Define the center of the circle
def center (a : ℝ) : ℝ × ℝ :=
  (a, -2*a)

-- State that the center of circle C lies on line l₁
axiom center_on_l1 (a : ℝ) :
  line_l1 (center a).1 (center a).2

-- Theorem: The length of the chord formed by intersecting circle C with line l₂ is 8
theorem chord_length : ℝ := by
  sorry

end NUMINAMATH_CALUDE_chord_length_l3209_320913


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l3209_320921

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 241) 
  (h2 : a*b + b*c + a*c = 100) : 
  a + b + c = 21 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l3209_320921


namespace NUMINAMATH_CALUDE_square_perimeter_l3209_320952

/-- The sum of the lengths of all sides of a square with side length 5 cm is 20 cm. -/
theorem square_perimeter (side_length : ℝ) (h : side_length = 5) : 
  4 * side_length = 20 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l3209_320952


namespace NUMINAMATH_CALUDE_valid_subset_of_A_l3209_320956

def A : Set ℝ := {x | x ≥ 0}

theorem valid_subset_of_A : 
  ({1, 2} : Set ℝ) ⊆ A ∧ 
  ¬({x : ℝ | x ≤ 1} ⊆ A) ∧ 
  ¬({-1, 0, 1} ⊆ A) ∧ 
  ¬(Set.univ ⊆ A) :=
sorry

end NUMINAMATH_CALUDE_valid_subset_of_A_l3209_320956


namespace NUMINAMATH_CALUDE_rope_cutting_probability_rope_cutting_probability_is_one_third_l3209_320901

/-- The probability of cutting a rope of length 3 into two segments,
    each at least 1 unit long, when cut at a random position. -/
theorem rope_cutting_probability : ℝ :=
  let rope_length : ℝ := 3
  let min_segment_length : ℝ := 1
  let favorable_cut_length : ℝ := rope_length - 2 * min_segment_length
  favorable_cut_length / rope_length

/-- The probability of cutting a rope of length 3 into two segments,
    each at least 1 unit long, when cut at a random position, is 1/3. -/
theorem rope_cutting_probability_is_one_third :
  rope_cutting_probability = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_rope_cutting_probability_rope_cutting_probability_is_one_third_l3209_320901


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_fraction_l3209_320959

theorem reciprocal_of_negative_fraction :
  ((-5 : ℚ) / 3)⁻¹ = -3 / 5 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_fraction_l3209_320959


namespace NUMINAMATH_CALUDE_train_speed_conversion_l3209_320986

/-- Conversion factor from meters per second to kilometers per hour -/
def mps_to_kmph : ℝ := 3.6

/-- Train's speed in meters per second -/
def train_speed_mps : ℝ := 45.0036

/-- Train's speed in kilometers per hour -/
def train_speed_kmph : ℝ := train_speed_mps * mps_to_kmph

theorem train_speed_conversion :
  train_speed_kmph = 162.013 := by sorry

end NUMINAMATH_CALUDE_train_speed_conversion_l3209_320986


namespace NUMINAMATH_CALUDE_contact_list_count_is_38_l3209_320910

/-- The number of people on Jerome's contact list at the end of the month -/
def contact_list_count : ℕ :=
  let classmates : ℕ := 20
  let out_of_school_friends : ℕ := classmates / 2
  let immediate_family : ℕ := 3
  let added_contacts : ℕ := 5 + 7
  let removed_contacts : ℕ := 3 + 4
  classmates + out_of_school_friends + immediate_family + added_contacts - removed_contacts

/-- Theorem stating that the number of people on Jerome's contact list at the end of the month is 38 -/
theorem contact_list_count_is_38 : contact_list_count = 38 := by
  sorry

end NUMINAMATH_CALUDE_contact_list_count_is_38_l3209_320910


namespace NUMINAMATH_CALUDE_original_strip_length_is_57_l3209_320911

/-- Represents the folded strip configuration -/
structure FoldedStrip where
  width : ℝ
  folded_length : ℝ
  trapezium_count : ℕ

/-- Calculates the length of the original strip before folding -/
def original_strip_length (fs : FoldedStrip) : ℝ :=
  sorry

/-- Theorem stating the length of the original strip -/
theorem original_strip_length_is_57 (fs : FoldedStrip) 
  (h_width : fs.width = 3)
  (h_folded_length : fs.folded_length = 27)
  (h_trapezium_count : fs.trapezium_count = 4) :
  original_strip_length fs = 57 :=
sorry

end NUMINAMATH_CALUDE_original_strip_length_is_57_l3209_320911


namespace NUMINAMATH_CALUDE_min_lines_is_seven_l3209_320960

/-- A line in a Cartesian coordinate system --/
structure Line where
  k : ℝ
  b : ℝ
  k_nonzero : k ≠ 0

/-- The quadrants a line passes through --/
def quadrants (l : Line) : Set (Fin 4) :=
  sorry

/-- The minimum number of lines needed to ensure two lines pass through the same quadrants --/
def min_lines_same_quadrants : ℕ :=
  sorry

/-- Theorem stating that the minimum number of lines is 7 --/
theorem min_lines_is_seven : min_lines_same_quadrants = 7 := by
  sorry

end NUMINAMATH_CALUDE_min_lines_is_seven_l3209_320960


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_range_l3209_320975

/-- A point in the fourth quadrant has positive x-coordinate and negative y-coordinate -/
def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

/-- The theorem states that if a point P(a, a-2) is in the fourth quadrant, then 0 < a < 2 -/
theorem point_in_fourth_quadrant_range (a : ℝ) :
  fourth_quadrant a (a - 2) → 0 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_range_l3209_320975


namespace NUMINAMATH_CALUDE_parallel_resistance_calculation_l3209_320977

/-- 
Represents the combined resistance of two resistors connected in parallel.
x: resistance of the first resistor in ohms
y: resistance of the second resistor in ohms
r: combined resistance in ohms
-/
def parallel_resistance (x y : ℝ) (r : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ r > 0 ∧ (1 / r = 1 / x + 1 / y)

theorem parallel_resistance_calculation :
  ∃ (r : ℝ), parallel_resistance 4 6 r ∧ r = 2.4 := by sorry

end NUMINAMATH_CALUDE_parallel_resistance_calculation_l3209_320977


namespace NUMINAMATH_CALUDE_i_power_difference_zero_l3209_320982

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem i_power_difference_zero : i^45 - i^305 = 0 := by
  sorry

end NUMINAMATH_CALUDE_i_power_difference_zero_l3209_320982


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_intersection_A_B_nonempty_iff_l3209_320984

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B (k : ℝ) : Set ℝ := {x | x ≤ k}

-- Part 1
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B 1) = {x | 1 < x ∧ x < 3} := by sorry

-- Part 2
theorem intersection_A_B_nonempty_iff (k : ℝ) :
  (A ∩ B k).Nonempty ↔ k ≥ -1 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_intersection_A_B_nonempty_iff_l3209_320984


namespace NUMINAMATH_CALUDE_sin_translation_l3209_320983

/-- Given a function f(x) = sin(2x), when translated π/3 units to the right,
    the resulting function g(x) is equal to sin(2x - 2π/3). -/
theorem sin_translation (x : ℝ) :
  let f : ℝ → ℝ := fun x => Real.sin (2 * x)
  let g : ℝ → ℝ := fun x => f (x - π / 3)
  g x = Real.sin (2 * x - 2 * π / 3) :=
by sorry

end NUMINAMATH_CALUDE_sin_translation_l3209_320983


namespace NUMINAMATH_CALUDE_length_AD_is_zero_l3209_320944

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  let ab := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let bc := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let ca := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  ab = 9 ∧ bc = 40 ∧ ca = 41

-- Define right angle at C
def RightAngleC (A B C : ℝ × ℝ) : Prop :=
  (B.1 - C.1) * (A.1 - C.1) + (B.2 - C.2) * (A.2 - C.2) = 0

-- Define the circumscribed circle ω
def CircumscribedCircle (ω : Set (ℝ × ℝ)) (A B C : ℝ × ℝ) : Prop :=
  ∀ P : ℝ × ℝ, P ∈ ω ↔ (P.1 - A.1)^2 + (P.2 - A.2)^2 = 
                      (P.1 - B.1)^2 + (P.2 - B.2)^2 ∧
                      (P.1 - B.1)^2 + (P.2 - B.2)^2 = 
                      (P.1 - C.1)^2 + (P.2 - C.2)^2

-- Define point D
def PointD (D : ℝ × ℝ) (ω : Set (ℝ × ℝ)) (A C : ℝ × ℝ) : Prop :=
  D ∈ ω ∧ 
  (D.1 - (A.1 + C.1)/2) * (C.2 - A.2) = (D.2 - (A.2 + C.2)/2) * (C.1 - A.1) ∧
  (D.1 - A.1) * (C.1 - A.1) + (D.2 - A.2) * (C.2 - A.2) < 0

theorem length_AD_is_zero 
  (A B C D : ℝ × ℝ) (ω : Set (ℝ × ℝ)) : 
  Triangle A B C → 
  RightAngleC A B C → 
  CircumscribedCircle ω A B C → 
  PointD D ω A C → 
  Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2) = 0 := by
    sorry

end NUMINAMATH_CALUDE_length_AD_is_zero_l3209_320944


namespace NUMINAMATH_CALUDE_marbles_given_correct_l3209_320968

/-- The number of marbles Jack gave to Josh -/
def marbles_given (initial final : ℕ) : ℕ := final - initial

/-- Theorem stating that the number of marbles given is the difference between final and initial counts -/
theorem marbles_given_correct (initial final : ℕ) (h : final ≥ initial) :
  marbles_given initial final = final - initial :=
by sorry

end NUMINAMATH_CALUDE_marbles_given_correct_l3209_320968


namespace NUMINAMATH_CALUDE_rectangle_area_with_inscribed_circle_l3209_320992

theorem rectangle_area_with_inscribed_circle (r : ℝ) (ratio : ℝ) : 
  r = 8 → ratio = 3 → 
  let width := 2 * r
  let length := ratio * width
  width * length = 768 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_with_inscribed_circle_l3209_320992


namespace NUMINAMATH_CALUDE_line_circle_separation_l3209_320946

theorem line_circle_separation (a b : ℝ) (h_inside : a^2 + b^2 < 1) (h_not_origin : (a, b) ≠ (0, 0)) :
  ∀ x y : ℝ, (x^2 + y^2 = 1) → (a*x + b*y ≠ 1) := by
  sorry

end NUMINAMATH_CALUDE_line_circle_separation_l3209_320946


namespace NUMINAMATH_CALUDE_actual_speed_proof_l3209_320943

theorem actual_speed_proof (time_reduction : Real) (speed_increase : Real) 
  (h1 : time_reduction = Real.pi / 4)
  (h2 : speed_increase = Real.sqrt 15) : 
  ∃ (actual_speed : Real), actual_speed = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_actual_speed_proof_l3209_320943


namespace NUMINAMATH_CALUDE_initial_observations_count_l3209_320912

theorem initial_observations_count 
  (initial_avg : ℝ) 
  (new_obs : ℝ) 
  (avg_decrease : ℝ) 
  (h1 : initial_avg = 12)
  (h2 : new_obs = 5)
  (h3 : avg_decrease = 1) :
  ∃ n : ℕ, 
    (n : ℝ) * initial_avg = ((n : ℝ) + 1) * (initial_avg - avg_decrease) - new_obs ∧ 
    n = 6 :=
by sorry

end NUMINAMATH_CALUDE_initial_observations_count_l3209_320912


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l3209_320994

/-- Represents a parabola in the form y = a(x - h)^2 + k -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift (p : Parabola) (right : ℝ) (down : ℝ) : Parabola :=
  { a := p.a
    h := p.h + right
    k := p.k - down }

theorem parabola_shift_theorem (p : Parabola) :
  p.a = 3 ∧ p.h = 4 ∧ p.k = 2 →
  (shift p 1 3).a = 3 ∧ (shift p 1 3).h = 5 ∧ (shift p 1 3).k = -1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_theorem_l3209_320994


namespace NUMINAMATH_CALUDE_apples_theorem_l3209_320907

def apples_problem (initial_apples : ℕ) (ricki_removes : ℕ) (days : ℕ) : Prop :=
  let samson_removes := 2 * ricki_removes
  let bindi_removes := 3 * samson_removes
  let total_daily_removal := ricki_removes + samson_removes + bindi_removes
  let total_weekly_removal := total_daily_removal * days
  total_weekly_removal = initial_apples + 2150

theorem apples_theorem : apples_problem 1000 50 7 := by
  sorry

end NUMINAMATH_CALUDE_apples_theorem_l3209_320907


namespace NUMINAMATH_CALUDE_product_expansion_sum_l3209_320998

theorem product_expansion_sum (a b c d : ℝ) : 
  (∀ x, (2 * x^2 - 3 * x + 5) * (5 - x) = a * x^3 + b * x^2 + c * x + d) →
  27 * a + 9 * b + 3 * c + d = 28 := by
sorry

end NUMINAMATH_CALUDE_product_expansion_sum_l3209_320998


namespace NUMINAMATH_CALUDE_mean_home_runs_l3209_320987

def number_of_players : ℕ := 9

def home_run_distribution : List (ℕ × ℕ) :=
  [(5, 2), (6, 3), (8, 2), (10, 1), (12, 1)]

def total_home_runs : ℕ :=
  (home_run_distribution.map (λ (hr, count) => hr * count)).sum

theorem mean_home_runs :
  (total_home_runs : ℚ) / number_of_players = 66 / 9 := by sorry

end NUMINAMATH_CALUDE_mean_home_runs_l3209_320987


namespace NUMINAMATH_CALUDE_valid_parts_characterization_valid_parts_complete_l3209_320903

/-- A type representing the possible numbers of equal parts. -/
inductive ValidParts : Nat → Prop where
  | two : ValidParts 2
  | three : ValidParts 3
  | four : ValidParts 4
  | six : ValidParts 6
  | eight : ValidParts 8
  | twelve : ValidParts 12
  | twentyfour : ValidParts 24

/-- The total number of cells in the figure. -/
def totalCells : Nat := 24

/-- A function that checks if a number divides the total number of cells evenly. -/
def isDivisor (n : Nat) : Prop := totalCells % n = 0

/-- The main theorem stating that the valid numbers of parts are exactly those that divide the total number of cells evenly. -/
theorem valid_parts_characterization (n : Nat) : 
  ValidParts n ↔ (isDivisor n ∧ n > 1) :=
sorry

/-- The theorem stating that the list of valid parts is complete. -/
theorem valid_parts_complete : 
  ∀ n, isDivisor n ∧ n > 1 → ValidParts n :=
sorry

end NUMINAMATH_CALUDE_valid_parts_characterization_valid_parts_complete_l3209_320903


namespace NUMINAMATH_CALUDE_at_least_one_zero_l3209_320906

theorem at_least_one_zero (a b : ℝ) : (a ≠ 0 ∧ b ≠ 0) → False := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_zero_l3209_320906


namespace NUMINAMATH_CALUDE_tan_alpha_and_expression_l3209_320953

theorem tan_alpha_and_expression (α : Real) (h : Real.tan (α / 2) = 2) :
  Real.tan α = -4/3 ∧ (6 * Real.sin α + Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 7/6 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_and_expression_l3209_320953


namespace NUMINAMATH_CALUDE_absolute_value_properties_problem_solutions_l3209_320947

theorem absolute_value_properties :
  (∀ x y : ℝ, |x - y| = |y - x|) ∧
  (∀ x : ℝ, |x| ≥ 0) ∧
  (∀ x : ℝ, |x| = 0 ↔ x = 0) ∧
  (∀ x y : ℝ, |x + y| ≤ |x| + |y|) :=
sorry

theorem problem_solutions :
  (|3 - (-2)| = 5) ∧
  (∀ x : ℝ, |x + 2| = 3 → (x = 1 ∨ x = -5)) ∧
  (∃ m : ℝ, (∀ x : ℝ, |x - 1| + |x + 3| ≥ m) ∧ (∃ x : ℝ, |x - 1| + |x + 3| = m) ∧ m = 4) ∧
  (∃ m : ℝ, (∀ x : ℝ, |x + 1| + |x - 2| + |x - 4| ≥ m) ∧ (|2 + 1| + |2 - 2| + |2 - 4| = m) ∧ m = 5) ∧
  (∀ x y z : ℝ, (|x + 1| + |x - 2|) * (|y - 2| + |y + 1|) * (|z - 3| + |z + 1|) = 36 →
    (-3 ≤ x + y + z ∧ x + y + z ≤ 7)) :=
sorry

end NUMINAMATH_CALUDE_absolute_value_properties_problem_solutions_l3209_320947


namespace NUMINAMATH_CALUDE_vip_ticket_price_l3209_320962

/-- Represents the price of concert tickets and savings --/
structure ConcertTickets where
  savings : ℕ
  vipTickets : ℕ
  regularTickets : ℕ
  regularPrice : ℕ
  remainingMoney : ℕ

/-- Theorem: The price of each VIP ticket is $100 --/
theorem vip_ticket_price (ct : ConcertTickets)
  (h1 : ct.savings = 500)
  (h2 : ct.vipTickets = 2)
  (h3 : ct.regularTickets = 3)
  (h4 : ct.regularPrice = 50)
  (h5 : ct.remainingMoney = 150) :
  (ct.savings - ct.remainingMoney - ct.regularTickets * ct.regularPrice) / ct.vipTickets = 100 := by
  sorry


end NUMINAMATH_CALUDE_vip_ticket_price_l3209_320962


namespace NUMINAMATH_CALUDE_addition_puzzle_l3209_320930

theorem addition_puzzle (A B C D : Nat) : 
  A < 10 → B < 10 → C < 10 → D < 10 →
  A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
  700 + 10 * A + 5 + 100 * B + 70 + C = 900 + 30 + 8 →
  D = 9 := by
sorry

end NUMINAMATH_CALUDE_addition_puzzle_l3209_320930


namespace NUMINAMATH_CALUDE_f_of_4_eq_17_g_of_2_eq_29_l3209_320980

-- Define the function f
def f (x : ℝ) : ℝ := 5 * x - 3

-- Define the function g
def g (t : ℝ) : ℝ := 4 * t^3 + 2 * t - 7

-- Theorem for f(4) = 17
theorem f_of_4_eq_17 : f 4 = 17 := by sorry

-- Theorem for g(2) = 29
theorem g_of_2_eq_29 : g 2 = 29 := by sorry

end NUMINAMATH_CALUDE_f_of_4_eq_17_g_of_2_eq_29_l3209_320980


namespace NUMINAMATH_CALUDE_length_PQ_value_l3209_320985

/-- Triangle ABC with given side lengths and angle bisectors --/
structure TriangleABC where
  -- Side lengths
  AB : ℝ
  BC : ℝ
  CA : ℝ
  -- AH is altitude
  AH : ℝ
  -- Q and P are intersection points of angle bisectors with altitude
  AQ : ℝ
  AP : ℝ
  -- Conditions
  side_lengths : AB = 6 ∧ BC = 10 ∧ CA = 8
  altitude : AH = 4.8
  angle_bisector_intersections : AQ = 20/3 ∧ AP = 3

/-- The length of PQ in the given triangle configuration --/
def length_PQ (t : TriangleABC) : ℝ := t.AQ - t.AP

/-- Theorem stating that the length of PQ is 3.67 --/
theorem length_PQ_value (t : TriangleABC) : length_PQ t = 3.67 := by
  sorry

end NUMINAMATH_CALUDE_length_PQ_value_l3209_320985


namespace NUMINAMATH_CALUDE_function_with_period_3_is_periodic_l3209_320979

-- Define a function f from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the periodicity condition
def is_periodic_with_period (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

-- State the theorem
theorem function_with_period_3_is_periodic :
  (∀ x, f (x + 3) = f x) → ∃ p > 0, is_periodic_with_period f p :=
sorry

end NUMINAMATH_CALUDE_function_with_period_3_is_periodic_l3209_320979


namespace NUMINAMATH_CALUDE_complex_fourth_power_l3209_320999

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_fourth_power : (1 - i) ^ 4 = -4 := by
  sorry

end NUMINAMATH_CALUDE_complex_fourth_power_l3209_320999


namespace NUMINAMATH_CALUDE_barbara_savings_l3209_320949

/-- The number of weeks needed to save for a wristwatch -/
def weeks_to_save (watch_cost : ℕ) (weekly_allowance : ℕ) (current_savings : ℕ) : ℕ :=
  ((watch_cost - current_savings) + weekly_allowance - 1) / weekly_allowance

/-- Theorem: Given the conditions, Barbara needs 16 more weeks to save for the wristwatch -/
theorem barbara_savings : weeks_to_save 100 5 20 = 16 := by
  sorry

end NUMINAMATH_CALUDE_barbara_savings_l3209_320949


namespace NUMINAMATH_CALUDE_inequality_solution_l3209_320941

theorem inequality_solution (x : ℝ) : 
  1 / (x + 2) + 7 / (x + 6) ≥ 1 ↔ 
  x ≤ -6 ∨ (-2 < x ∧ x ≤ -Real.sqrt 15) ∨ x ≥ Real.sqrt 15 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3209_320941


namespace NUMINAMATH_CALUDE_johns_water_usage_l3209_320969

/-- Calculates the total water usage for John's showers over 4 weeks -/
def total_water_usage (weeks : ℕ) (shower_frequency : ℕ) (shower_duration : ℕ) (water_per_minute : ℕ) : ℕ :=
  let days := weeks * 7
  let num_showers := days / shower_frequency
  let water_per_shower := shower_duration * water_per_minute
  num_showers * water_per_shower

/-- Proves that John's total water usage over 4 weeks is 280 gallons -/
theorem johns_water_usage : total_water_usage 4 2 10 2 = 280 := by
  sorry

end NUMINAMATH_CALUDE_johns_water_usage_l3209_320969
