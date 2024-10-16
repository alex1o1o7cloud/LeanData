import Mathlib

namespace NUMINAMATH_CALUDE_dessert_eating_contest_l1230_123003

theorem dessert_eating_contest (student1_pie : ℚ) (student2_pie : ℚ) (student3_cake : ℚ) :
  student1_pie = 5/6 ∧ student2_pie = 7/8 ∧ student3_cake = 1/2 →
  max student1_pie student2_pie - student3_cake = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_dessert_eating_contest_l1230_123003


namespace NUMINAMATH_CALUDE_train_speed_problem_l1230_123044

-- Define the speeds and times
def speed_A : ℝ := 90
def time_A : ℝ := 9
def time_B : ℝ := 4

-- Theorem statement
theorem train_speed_problem :
  ∃ (speed_B : ℝ),
    speed_B > 0 ∧
    speed_A * time_A = speed_B * time_B ∧
    speed_B = 202.5 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_problem_l1230_123044


namespace NUMINAMATH_CALUDE_tissue_usage_l1230_123005

theorem tissue_usage (initial_tissues : ℕ) (remaining_tissues : ℕ) 
  (alice_usage : ℕ) (bob_multiplier : ℕ) (eve_reduction : ℕ) : 
  initial_tissues = 97 →
  remaining_tissues = 47 →
  alice_usage = 12 →
  bob_multiplier = 2 →
  eve_reduction = 3 →
  initial_tissues - remaining_tissues + 
  alice_usage + bob_multiplier * alice_usage + (alice_usage - eve_reduction) = 95 :=
by sorry

end NUMINAMATH_CALUDE_tissue_usage_l1230_123005


namespace NUMINAMATH_CALUDE_max_element_of_S_l1230_123040

def S : Set ℚ := {x | ∃ (p q : ℕ), x = p / q ∧ q ≤ 2009 ∧ x < 1257 / 2009}

theorem max_element_of_S :
  ∃ (p₀ q₀ : ℕ), 
    (p₀ : ℚ) / q₀ ∈ S ∧ 
    (∀ (x : ℚ), x ∈ S → x ≤ (p₀ : ℚ) / q₀) ∧
    (Nat.gcd p₀ q₀ = 1) ∧
    p₀ = 229 ∧ 
    q₀ = 366 ∧ 
    p₀ + q₀ = 595 := by
  sorry

end NUMINAMATH_CALUDE_max_element_of_S_l1230_123040


namespace NUMINAMATH_CALUDE_cricket_bat_profit_l1230_123052

/-- Calculates the profit amount for a cricket bat sale -/
theorem cricket_bat_profit (selling_price : ℝ) (profit_percentage : ℝ) : 
  selling_price = 850 ∧ profit_percentage = 36 →
  (selling_price - selling_price / (1 + profit_percentage / 100)) = 225 := by
sorry

end NUMINAMATH_CALUDE_cricket_bat_profit_l1230_123052


namespace NUMINAMATH_CALUDE_smallest_coin_arrangement_l1230_123087

/-- The number of divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- The number of proper divisors of a positive integer greater than 2 -/
def num_proper_divisors_gt_2 (n : ℕ+) : ℕ := sorry

/-- Checks if all divisors d of n where 2 < d < n, n/d is an integer -/
def all_divisors_divide (n : ℕ+) : Prop := sorry

theorem smallest_coin_arrangement :
  ∃ (n : ℕ+), num_divisors n = 19 ∧ 
              num_proper_divisors_gt_2 n = 17 ∧ 
              all_divisors_divide n ∧
              (∀ m : ℕ+, m < n → 
                (num_divisors m ≠ 19 ∨ 
                 num_proper_divisors_gt_2 m ≠ 17 ∨ 
                 ¬all_divisors_divide m)) ∧
              n = 2700 := by sorry

end NUMINAMATH_CALUDE_smallest_coin_arrangement_l1230_123087


namespace NUMINAMATH_CALUDE_distinct_z_values_l1230_123039

def is_valid_number (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def swap_digits (x : ℕ) : ℕ :=
  let a := x / 100
  let b := (x / 10) % 10
  let c := x % 10
  100 * b + 10 * a + c

def z (x : ℕ) : ℕ := Int.natAbs (x - swap_digits x)

theorem distinct_z_values (x : ℕ) (hx : is_valid_number x) : 
  ∃ (S : Finset ℕ), (∀ n, n ∈ S ↔ ∃ y, is_valid_number y ∧ z y = n) ∧ Finset.card S = 9 :=
sorry

end NUMINAMATH_CALUDE_distinct_z_values_l1230_123039


namespace NUMINAMATH_CALUDE_problem_solution_l1230_123055

theorem problem_solution :
  45 / (8 - 3/7) = 315/53 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1230_123055


namespace NUMINAMATH_CALUDE_linear_independence_of_polynomial_basis_l1230_123026

theorem linear_independence_of_polynomial_basis :
  ∀ (α₁ α₂ α₃ α₄ : ℝ),
  (∀ x : ℝ, α₁ + α₂ * x + α₃ * x^2 + α₄ * x^3 = 0) →
  (α₁ = 0 ∧ α₂ = 0 ∧ α₃ = 0 ∧ α₄ = 0) :=
by sorry

end NUMINAMATH_CALUDE_linear_independence_of_polynomial_basis_l1230_123026


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1230_123078

theorem simplify_and_evaluate :
  let x : ℚ := -1
  let y : ℚ := 1
  let expr := (x + y) * (x - y) - (4 * x^3 * y - 8 * x * y^3) / (2 * x * y)
  expr = -x^2 + 3*y^2 ∧ expr = 2 := by
sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1230_123078


namespace NUMINAMATH_CALUDE_range_of_a_l1230_123007

/-- The function f(x) = -x^2 + 4x + a has a zero in the interval [-3, 3] -/
def has_zero_in_interval (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, x ∈ Set.Icc (-3) 3 ∧ f x = 0

/-- The quadratic function f(x) = -x^2 + 4x + a -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 4*x + a

theorem range_of_a :
  ∀ a : ℝ, has_zero_in_interval (f a) ↔ a ∈ Set.Icc (-3) 21 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1230_123007


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l1230_123025

-- Problem 1
theorem problem_1 : 0.25 * 1.25 * 32 = 10 := by sorry

-- Problem 2
theorem problem_2 : 4/5 * 5/11 + 5/11 / 5 = 5/11 := by sorry

-- Problem 3
theorem problem_3 : 7 - 4/9 - 5/9 = 6 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l1230_123025


namespace NUMINAMATH_CALUDE_minimum_games_for_95_percent_win_rate_l1230_123028

theorem minimum_games_for_95_percent_win_rate : 
  ∃ N : ℕ, (N = 37 ∧ (1 + N : ℚ) / (3 + N) ≥ 95 / 100) ∧
  ∀ M : ℕ, M < N → (1 + M : ℚ) / (3 + M) < 95 / 100 := by
  sorry

end NUMINAMATH_CALUDE_minimum_games_for_95_percent_win_rate_l1230_123028


namespace NUMINAMATH_CALUDE_employee_age_when_hired_l1230_123077

theorem employee_age_when_hired (age_when_hired : ℕ) (years_worked : ℕ) : 
  age_when_hired + years_worked = 70 →
  years_worked = 19 →
  age_when_hired = 51 := by
  sorry

end NUMINAMATH_CALUDE_employee_age_when_hired_l1230_123077


namespace NUMINAMATH_CALUDE_cubic_factorization_l1230_123060

theorem cubic_factorization (x : ℝ) : x^3 - 16*x = x*(x+4)*(x-4) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l1230_123060


namespace NUMINAMATH_CALUDE_range_of_sum_l1230_123082

theorem range_of_sum (a b : ℝ) (h : a^2 - a*b + b^2 = a + b) :
  ∃ t : ℝ, t = a + b ∧ 0 ≤ t ∧ t ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_sum_l1230_123082


namespace NUMINAMATH_CALUDE_max_basketballs_l1230_123092

/-- Represents the prices and quantities of soccer balls and basketballs --/
structure BallPurchase where
  soccer_price : ℝ
  basketball_price : ℝ
  soccer_quantity : ℕ
  basketball_quantity : ℕ

/-- The conditions of the ball purchase problem --/
def valid_purchase (p : BallPurchase) : Prop :=
  p.basketball_price = 2 * p.soccer_price - 30 ∧
  3 * p.soccer_price * p.soccer_quantity = 2 * p.basketball_price * p.basketball_quantity ∧
  p.soccer_quantity + p.basketball_quantity = 200 ∧
  p.soccer_price * p.soccer_quantity + p.basketball_price * p.basketball_quantity ≤ 15500

/-- The theorem stating the maximum number of basketballs that can be purchased --/
theorem max_basketballs (p : BallPurchase) :
  valid_purchase p → p.basketball_quantity ≤ 116 := by
  sorry

end NUMINAMATH_CALUDE_max_basketballs_l1230_123092


namespace NUMINAMATH_CALUDE_optimal_investment_l1230_123091

/-- Represents an investment project with maximum profit and loss rates. -/
structure Project where
  max_profit_rate : ℝ
  max_loss_rate : ℝ

/-- Represents the investment scenario with two projects and constraints. -/
structure InvestmentScenario where
  project_a : Project
  project_b : Project
  total_investment : ℝ
  max_potential_loss : ℝ

/-- Calculates the potential loss for a given investment allocation. -/
def potential_loss (scenario : InvestmentScenario) (invest_a : ℝ) (invest_b : ℝ) : ℝ :=
  invest_a * scenario.project_a.max_loss_rate + invest_b * scenario.project_b.max_loss_rate

/-- Calculates the potential profit for a given investment allocation. -/
def potential_profit (scenario : InvestmentScenario) (invest_a : ℝ) (invest_b : ℝ) : ℝ :=
  invest_a * scenario.project_a.max_profit_rate + invest_b * scenario.project_b.max_profit_rate

/-- Theorem stating that the given investment allocation maximizes potential profits
    while satisfying all constraints. -/
theorem optimal_investment (scenario : InvestmentScenario)
    (h_project_a : scenario.project_a = { max_profit_rate := 1, max_loss_rate := 0.3 })
    (h_project_b : scenario.project_b = { max_profit_rate := 0.5, max_loss_rate := 0.1 })
    (h_total_investment : scenario.total_investment = 100000)
    (h_max_potential_loss : scenario.max_potential_loss = 18000) :
    ∀ (x y : ℝ),
      x + y ≤ scenario.total_investment →
      potential_loss scenario x y ≤ scenario.max_potential_loss →
      potential_profit scenario x y ≤ potential_profit scenario 40000 60000 :=
  sorry

end NUMINAMATH_CALUDE_optimal_investment_l1230_123091


namespace NUMINAMATH_CALUDE_skew_edges_count_l1230_123067

/-- Represents a cube in 3D space -/
structure Cube where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Represents a line in 3D space -/
structure Line where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Checks if a line lies on a face of the cube -/
def lineOnFace (c : Cube) (l : Line) : Prop :=
  sorry

/-- Counts the number of edges not in the same plane as the given line -/
def countSkewEdges (c : Cube) (l : Line) : ℕ :=
  sorry

/-- Main theorem: The number of skew edges is either 4, 6, 7, or 8 -/
theorem skew_edges_count (c : Cube) (l : Line) 
  (h : lineOnFace c l) : 
  (countSkewEdges c l = 4) ∨ 
  (countSkewEdges c l = 6) ∨ 
  (countSkewEdges c l = 7) ∨ 
  (countSkewEdges c l = 8) :=
sorry

end NUMINAMATH_CALUDE_skew_edges_count_l1230_123067


namespace NUMINAMATH_CALUDE_product_sum_of_digits_77_sevens_77_threes_l1230_123042

/-- Represents a string of digits repeated n times -/
def RepeatedDigitString (digit : Nat) (n : Nat) : Nat :=
  -- Definition omitted for brevity
  sorry

/-- Calculates the sum of digits in a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  -- Definition omitted for brevity
  sorry

/-- The main theorem to prove -/
theorem product_sum_of_digits_77_sevens_77_threes :
  sumOfDigits (RepeatedDigitString 7 77 * RepeatedDigitString 3 77) = 231 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_of_digits_77_sevens_77_threes_l1230_123042


namespace NUMINAMATH_CALUDE_complex_division_theorem_l1230_123085

theorem complex_division_theorem : 
  let z₁ : ℂ := Complex.mk 1 (-1)
  let z₂ : ℂ := Complex.mk 3 1
  z₂ / z₁ = Complex.mk 1 2 := by
sorry

end NUMINAMATH_CALUDE_complex_division_theorem_l1230_123085


namespace NUMINAMATH_CALUDE_perfect_square_last_two_digits_product_even_l1230_123024

theorem perfect_square_last_two_digits_product_even (n : ℤ) : 
  ∃ (a b : ℤ), 0 ≤ a ∧ a < 10 ∧ 0 ≤ b ∧ b < 10 ∧ 
  (n^2 % 100 = 10 * a + b) ∧ 
  Even (a * b) := by
sorry

end NUMINAMATH_CALUDE_perfect_square_last_two_digits_product_even_l1230_123024


namespace NUMINAMATH_CALUDE_abcd_multiplication_l1230_123032

theorem abcd_multiplication (A B C D : ℕ) : 
  A < 10 → B < 10 → C < 10 → D < 10 →
  A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
  (1000 * A + 100 * B + 10 * C + D) * 2 = 10000 * A + 1000 * B + 100 * C + 10 * D →
  A + B = 1 := by
sorry

end NUMINAMATH_CALUDE_abcd_multiplication_l1230_123032


namespace NUMINAMATH_CALUDE_sequence_with_constant_triple_sum_l1230_123095

theorem sequence_with_constant_triple_sum :
  ∃! (a : Fin 8 → ℝ), 
    a 0 = 5 ∧ 
    a 7 = 8 ∧ 
    (∀ i : Fin 6, a i + a (i + 1) + a (i + 2) = 20) := by
  sorry

end NUMINAMATH_CALUDE_sequence_with_constant_triple_sum_l1230_123095


namespace NUMINAMATH_CALUDE_xy_necessary_not_sufficient_l1230_123064

theorem xy_necessary_not_sufficient (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (∀ x y, x > 0 → y > 0 → x + y < 4 → x * y < 4) ∧
  (∃ x y, x > 0 ∧ y > 0 ∧ x * y < 4 ∧ x + y ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_xy_necessary_not_sufficient_l1230_123064


namespace NUMINAMATH_CALUDE_greaterElementSumOfS_l1230_123033

def S : Finset ℕ := {8, 5, 1, 13, 34, 3, 21, 2}

def greaterElementSum (s : Finset ℕ) : ℕ :=
  s.sum (λ x => (s.filter (λ y => y < x)).card * x)

theorem greaterElementSumOfS : greaterElementSum S = 484 := by
  sorry

end NUMINAMATH_CALUDE_greaterElementSumOfS_l1230_123033


namespace NUMINAMATH_CALUDE_intersection_point_is_unique_l1230_123011

/-- The first line: x - 2y - 4 = 0 -/
def line1 (x y : ℝ) : Prop := x - 2*y - 4 = 0

/-- The second line: x + 3y + 6 = 0 -/
def line2 (x y : ℝ) : Prop := x + 3*y + 6 = 0

/-- The intersection point (0, -2) -/
def intersection_point : ℝ × ℝ := (0, -2)

/-- Theorem stating that (0, -2) is the unique intersection point of the two lines -/
theorem intersection_point_is_unique :
  (∃! p : ℝ × ℝ, line1 p.1 p.2 ∧ line2 p.1 p.2) ∧
  (line1 intersection_point.1 intersection_point.2) ∧
  (line2 intersection_point.1 intersection_point.2) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_is_unique_l1230_123011


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_five_l1230_123098

theorem fraction_zero_implies_x_negative_five (x : ℝ) :
  (x + 5) / (x - 2) = 0 → x = -5 := by
sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_five_l1230_123098


namespace NUMINAMATH_CALUDE_complex_on_ellipse_real_fraction_l1230_123004

theorem complex_on_ellipse_real_fraction (z : ℂ) :
  let x : ℝ := z.re
  let y : ℝ := z.im
  (x^2 / 9 + y^2 / 16 = 1) →
  ((z - (1 + I)) / (z - I)).im = 0 →
  z = Complex.mk ((3 * Real.sqrt 15) / 4) 1 ∨
  z = Complex.mk (-(3 * Real.sqrt 15) / 4) 1 :=
by sorry

end NUMINAMATH_CALUDE_complex_on_ellipse_real_fraction_l1230_123004


namespace NUMINAMATH_CALUDE_total_books_l1230_123041

theorem total_books (x : ℚ) : ℚ := by
  -- Betty's books
  let betty_books := x

  -- Sister's books: x + (1/4)x
  let sister_books := x + (1/4) * x

  -- Cousin's books: 2 * (x + (1/4)x)
  let cousin_books := 2 * (x + (1/4) * x)

  -- Total books
  let total := betty_books + sister_books + cousin_books

  -- Prove that total = (19/4)x
  sorry

end NUMINAMATH_CALUDE_total_books_l1230_123041


namespace NUMINAMATH_CALUDE_max_gcd_13n_plus_4_8n_plus_3_l1230_123059

theorem max_gcd_13n_plus_4_8n_plus_3 (n : ℕ+) : 
  (Nat.gcd (13 * n + 4) (8 * n + 3) ≤ 7) ∧ 
  (∃ m : ℕ+, Nat.gcd (13 * m + 4) (8 * m + 3) = 7) := by
sorry

end NUMINAMATH_CALUDE_max_gcd_13n_plus_4_8n_plus_3_l1230_123059


namespace NUMINAMATH_CALUDE_cost_of_oil_l1230_123017

/-- The cost of oil given the total cost of groceries and the costs of beef and chicken -/
theorem cost_of_oil (total_cost beef_cost chicken_cost : ℝ) : 
  total_cost = 16 → beef_cost = 12 → chicken_cost = 3 → 
  total_cost - (beef_cost + chicken_cost) = 1 := by
sorry

end NUMINAMATH_CALUDE_cost_of_oil_l1230_123017


namespace NUMINAMATH_CALUDE_fraction_subtraction_simplification_l1230_123038

theorem fraction_subtraction_simplification :
  ∃ (a b : ℚ), a = 9/19 ∧ b = 5/57 ∧ a - b = 22/57 ∧ (∀ (c d : ℤ), c ≠ 0 → 22/57 = c/d → (c = 22 ∧ d = 57 ∨ c = -22 ∧ d = -57)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_subtraction_simplification_l1230_123038


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l1230_123084

theorem complex_number_in_fourth_quadrant :
  let z : ℂ := (2 - I) / (1 + I)
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l1230_123084


namespace NUMINAMATH_CALUDE_school_travel_speed_l1230_123079

/-- Proves that the speed on the second day is 10 km/hr given the conditions of the problem -/
theorem school_travel_speed 
  (distance : ℝ) 
  (speed_day1 : ℝ) 
  (late_time : ℝ) 
  (early_time : ℝ) 
  (h1 : distance = 2.5) 
  (h2 : speed_day1 = 5) 
  (h3 : late_time = 7 / 60) 
  (h4 : early_time = 8 / 60) : 
  let correct_time := distance / speed_day1
  let actual_time_day2 := correct_time - late_time - early_time
  distance / actual_time_day2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_school_travel_speed_l1230_123079


namespace NUMINAMATH_CALUDE_paulines_convertibles_l1230_123090

/-- Calculates the number of convertibles in Pauline's car collection --/
theorem paulines_convertibles (total : ℕ) (regular_percent trucks_percent sedans_percent sports_percent suvs_percent : ℚ) :
  total = 125 →
  regular_percent = 38/100 →
  trucks_percent = 12/100 →
  sedans_percent = 17/100 →
  sports_percent = 22/100 →
  suvs_percent = 6/100 →
  ∃ (regular trucks sedans sports suvs convertibles : ℕ),
    regular = ⌊(regular_percent * total : ℚ)⌋ ∧
    trucks = ⌊(trucks_percent * total : ℚ)⌋ ∧
    sedans = ⌊(sedans_percent * total : ℚ)⌋ ∧
    sports = ⌊(sports_percent * total : ℚ)⌋ ∧
    suvs = ⌊(suvs_percent * total : ℚ)⌋ ∧
    convertibles = total - (regular + trucks + sedans + sports + suvs) ∧
    convertibles = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_paulines_convertibles_l1230_123090


namespace NUMINAMATH_CALUDE_final_book_count_l1230_123074

/-- Represents the number of books in each genre -/
structure BookCollection :=
  (novels : ℕ)
  (science : ℕ)
  (cookbooks : ℕ)
  (philosophy : ℕ)
  (history : ℕ)
  (selfHelp : ℕ)

/-- Represents the donation percentages for each genre -/
structure DonationPercentages :=
  (novels : ℚ)
  (science : ℚ)
  (cookbooks : ℚ)
  (philosophy : ℚ)
  (history : ℚ)
  (selfHelp : ℚ)

def initialCollection : BookCollection :=
  { novels := 75
  , science := 55
  , cookbooks := 40
  , philosophy := 35
  , history := 25
  , selfHelp := 20 }

def donationPercentages : DonationPercentages :=
  { novels := 3/5
  , science := 3/4
  , cookbooks := 1/2
  , philosophy := 3/10
  , history := 1/4
  , selfHelp := 1 }

def recyclePercentage : ℚ := 1/20

def newBooksAcquired : ℕ := 24

theorem final_book_count
  (total : ℕ)
  (h1 : total = initialCollection.novels + initialCollection.science +
                initialCollection.cookbooks + initialCollection.philosophy +
                initialCollection.history + initialCollection.selfHelp)
  (h2 : total = 250) :
  ∃ (donatedBooks recycledBooks remainingBooks : ℕ),
    donatedBooks = ⌊initialCollection.novels * donationPercentages.novels⌋ +
                   ⌊initialCollection.science * donationPercentages.science⌋ +
                   ⌊initialCollection.cookbooks * donationPercentages.cookbooks⌋ +
                   ⌊initialCollection.philosophy * donationPercentages.philosophy⌋ +
                   ⌊initialCollection.history * donationPercentages.history⌋ +
                   ⌊initialCollection.selfHelp * donationPercentages.selfHelp⌋ ∧
    recycledBooks = ⌊(donatedBooks : ℚ) * recyclePercentage⌋ ∧
    remainingBooks = total - donatedBooks + recycledBooks ∧
    remainingBooks + newBooksAcquired = 139 :=
by sorry

end NUMINAMATH_CALUDE_final_book_count_l1230_123074


namespace NUMINAMATH_CALUDE_juan_distance_l1230_123015

/-- Given a time in hours and a speed in miles per hour, calculates the distance traveled in miles. -/
def distance (time : Real) (speed : Real) : Real :=
  time * speed

/-- Theorem: Juan's distance traveled given his time and speed -/
theorem juan_distance : distance 80.0 10.0 = 800.0 := by
  sorry

end NUMINAMATH_CALUDE_juan_distance_l1230_123015


namespace NUMINAMATH_CALUDE_determinant_transformation_l1230_123020

theorem determinant_transformation (x y z w : ℝ) :
  Matrix.det ![![x, y], ![z, w]] = 6 →
  Matrix.det ![![x, 5*x + 4*y], ![z, 5*z + 4*w]] = 24 := by
  sorry

end NUMINAMATH_CALUDE_determinant_transformation_l1230_123020


namespace NUMINAMATH_CALUDE_photo_arrangements_l1230_123071

/-- The number of male students -/
def num_male_students : ℕ := 4

/-- The number of female students -/
def num_female_students : ℕ := 3

/-- The total number of students -/
def total_students : ℕ := num_male_students + num_female_students

theorem photo_arrangements :
  /- (1) Arrangements with male student A at one of the ends -/
  (∃ (n : ℕ), n = 1440 ∧ 
    n = 2 * (Nat.factorial (total_students - 1))) ∧
  /- (2) Arrangements where female students B and C are not next to each other -/
  (∃ (m : ℕ), m = 3600 ∧ 
    m = (Nat.factorial (total_students - 2)) * (total_students * (total_students - 1) / 2)) ∧
  /- (3) Arrangements where female student B is not at the ends and C is not in the middle -/
  (∃ (k : ℕ), k = 3120 ∧ 
    k = (Nat.factorial (total_students - 2)) * 4 + (Nat.factorial (total_students - 2)) * 4 * 5) :=
by sorry

end NUMINAMATH_CALUDE_photo_arrangements_l1230_123071


namespace NUMINAMATH_CALUDE_opposite_of_negative_fraction_l1230_123097

theorem opposite_of_negative_fraction :
  let x : ℚ := -4/5
  let opposite (y : ℚ) : ℚ := -y
  opposite x = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_fraction_l1230_123097


namespace NUMINAMATH_CALUDE_badminton_match_31_probability_l1230_123073

def badminton_match_probability (p : ℝ) : ℝ :=
  4 * p^3 * (1 - p)

theorem badminton_match_31_probability :
  badminton_match_probability (2/3) = 8/27 := by
  sorry

end NUMINAMATH_CALUDE_badminton_match_31_probability_l1230_123073


namespace NUMINAMATH_CALUDE_arithmetic_geometric_seq_l1230_123035

/-- An arithmetic sequence with common difference 3 -/
def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + 3

/-- a_1, a_3, and a_4 form a geometric sequence -/
def geometric_subseq (a : ℕ → ℤ) : Prop :=
  (a 3) ^ 2 = (a 1) * (a 4)

theorem arithmetic_geometric_seq (a : ℕ → ℤ) 
  (h1 : arithmetic_seq a) 
  (h2 : geometric_subseq a) : 
  a 2 = -9 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_seq_l1230_123035


namespace NUMINAMATH_CALUDE_angle_sum_around_point_l1230_123053

theorem angle_sum_around_point (y : ℝ) : 
  (6*y + 3*y + 4*y + 2*y = 360) → y = 24 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_around_point_l1230_123053


namespace NUMINAMATH_CALUDE_optimal_garden_dimensions_l1230_123083

/-- Represents a rectangular garden with one side along a house wall. -/
structure Garden where
  width : ℝ  -- Width of the garden (perpendicular to the house)
  length : ℝ  -- Length of the garden (parallel to the house)

/-- Calculates the area of a rectangular garden. -/
def Garden.area (g : Garden) : ℝ := g.width * g.length

/-- Calculates the cost of fencing for three sides of the garden. -/
def Garden.fenceCost (g : Garden) : ℝ := 10 * (g.length + 2 * g.width)

/-- Theorem stating the optimal dimensions of the garden. -/
theorem optimal_garden_dimensions (houseLength : ℝ) (totalFenceCost : ℝ) :
  houseLength = 300 → totalFenceCost = 2000 →
  ∃ (g : Garden),
    g.fenceCost = totalFenceCost ∧
    g.length = 100 ∧
    ∀ (g' : Garden), g'.fenceCost = totalFenceCost → g.area ≥ g'.area :=
sorry

end NUMINAMATH_CALUDE_optimal_garden_dimensions_l1230_123083


namespace NUMINAMATH_CALUDE_sequence_sum_proof_l1230_123010

-- Define the sequence a_n and its sum S_n
def S (n : ℕ) : ℕ := 2^(n+1) - 2

-- Define the sequence b_n
def b (n : ℕ) : ℝ := 2^n * n

-- Theorem statement
theorem sequence_sum_proof (n : ℕ) :
  (∀ k, S k = 2^(k+1) - 2) →
  (∀ k, b k = 2^k * k) →
  (∃ T : ℕ → ℝ, T n = (n + 1) * 2^(n + 1) - 2) :=
by sorry

end NUMINAMATH_CALUDE_sequence_sum_proof_l1230_123010


namespace NUMINAMATH_CALUDE_square_division_l1230_123086

theorem square_division (s : ℝ) (x : ℝ) : 
  s = 2 →  -- side length of the square
  (4 * (1/2 * s * x) + (s^2 - 4 * (1/2 * s * x))) = (s^2 / 5) →  -- equal areas condition
  x = 4/5 :=
by sorry

end NUMINAMATH_CALUDE_square_division_l1230_123086


namespace NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l1230_123008

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 28 ways to distribute 6 indistinguishable balls into 3 distinguishable boxes -/
theorem distribute_six_balls_three_boxes :
  distribute_balls 6 3 = 28 := by
  sorry

end NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l1230_123008


namespace NUMINAMATH_CALUDE_cosine_value_for_given_point_l1230_123050

theorem cosine_value_for_given_point :
  ∀ α : Real,
  let P : Real × Real := (2 * Real.cos (120 * π / 180), Real.sqrt 2 * Real.sin (225 * π / 180))
  (Real.cos α = P.1 / Real.sqrt (P.1^2 + P.2^2) ∧
   Real.sin α = P.2 / Real.sqrt (P.1^2 + P.2^2)) →
  Real.cos α = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_value_for_given_point_l1230_123050


namespace NUMINAMATH_CALUDE_scientific_notation_of_1650000000_l1230_123072

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coefficient : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation with a specified number of significant figures -/
def toScientificNotation (x : ℝ) (sigFigs : ℕ) : ScientificNotation :=
  sorry

/-- The original number to be expressed in scientific notation -/
def originalNumber : ℝ := 1650000000

/-- The number of significant figures to keep -/
def sigFigs : ℕ := 3

theorem scientific_notation_of_1650000000 :
  toScientificNotation originalNumber sigFigs =
    ScientificNotation.mk 1.65 9 (by norm_num) :=
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_1650000000_l1230_123072


namespace NUMINAMATH_CALUDE_max_salary_specific_team_l1230_123054

/-- Represents a basketball team -/
structure BasketballTeam where
  players : Nat
  minSalary : Nat
  salaryCap : Nat

/-- Calculates the maximum possible salary for a single player in a basketball team -/
def maxPlayerSalary (team : BasketballTeam) : Nat :=
  team.salaryCap - (team.players - 1) * team.minSalary

/-- Theorem stating the maximum possible salary for a single player
    in a specific basketball team configuration -/
theorem max_salary_specific_team :
  let team : BasketballTeam := {
    players := 25,
    minSalary := 18000,
    salaryCap := 900000
  }
  maxPlayerSalary team = 468000 := by
  sorry

#eval maxPlayerSalary {players := 25, minSalary := 18000, salaryCap := 900000}

end NUMINAMATH_CALUDE_max_salary_specific_team_l1230_123054


namespace NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l1230_123012

/-- Represents a geometric sequence with a given first term and common ratio -/
def GeometricSequence (a : ℝ) (r : ℝ) : ℕ → ℝ := fun n => a * r ^ (n - 1)

/-- The fourth term of a geometric sequence given its first and sixth terms -/
theorem fourth_term_of_geometric_sequence (a₁ : ℝ) (a₆ : ℝ) :
  a₁ > 0 → a₆ > 0 →
  ∃ (r : ℝ), r > 0 ∧ GeometricSequence a₁ r 6 = a₆ →
  GeometricSequence a₁ r 4 = 1536 :=
by
  sorry

#check fourth_term_of_geometric_sequence 512 125

end NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l1230_123012


namespace NUMINAMATH_CALUDE_total_spent_on_car_parts_l1230_123043

def speakers : ℚ := 235.87
def newTires : ℚ := 281.45
def steeringWheelCover : ℚ := 179.99
def seatCovers : ℚ := 122.31
def headlights : ℚ := 98.63

theorem total_spent_on_car_parts : 
  speakers + newTires + steeringWheelCover + seatCovers + headlights = 918.25 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_on_car_parts_l1230_123043


namespace NUMINAMATH_CALUDE_range_of_abc_l1230_123065

theorem range_of_abc (a b c : ℝ) 
  (ha : -1 < a) (hab : a < b) (hb : b < 1) 
  (hc : 2 < c) (hc' : c < 3) :
  ∀ y, (∃ x, x = (a - b) * c ∧ -6 < x ∧ x < 0) ↔ -6 < y ∧ y < 0 :=
by sorry

end NUMINAMATH_CALUDE_range_of_abc_l1230_123065


namespace NUMINAMATH_CALUDE_impossibility_of_forming_parallelepiped_l1230_123070

/-- Represents the dimensions of a rectangular parallelepiped -/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Checks if a parallelepiped can be formed from smaller parallelepipeds -/
def can_form_parallelepiped (large : Dimensions) (small : Dimensions) : Prop :=
  ∃ (n : ℕ), 
    n * (small.length * small.width * small.height) = large.length * large.width * large.height ∧
    ∀ (face : ℕ), face ∈ 
      [large.length * large.width, large.width * large.height, large.length * large.height] →
      ∃ (a b : ℕ), a * small.length * small.width + b * small.length * small.height + 
                   (n - a - b) * small.width * small.height = face

theorem impossibility_of_forming_parallelepiped : 
  ¬ can_form_parallelepiped 
    (Dimensions.mk 3 4 5) 
    (Dimensions.mk 2 2 1) := by
  sorry

end NUMINAMATH_CALUDE_impossibility_of_forming_parallelepiped_l1230_123070


namespace NUMINAMATH_CALUDE_finance_class_competition_l1230_123062

theorem finance_class_competition (total : ℕ) (abacus : ℕ) (cash_counting : ℕ) (neither : ℕ) :
  total = 48 →
  abacus = 28 →
  cash_counting = 23 →
  neither = 5 →
  ∃ n : ℕ, n = 8 ∧ 
    total = abacus + cash_counting - n + neither :=
by sorry

end NUMINAMATH_CALUDE_finance_class_competition_l1230_123062


namespace NUMINAMATH_CALUDE_weekly_payment_problem_l1230_123075

/-- The weekly payment problem -/
theorem weekly_payment_problem (payment_B : ℝ) (payment_ratio : ℝ) : 
  payment_B = 180 →
  payment_ratio = 1.5 →
  payment_B + payment_ratio * payment_B = 450 := by
  sorry

end NUMINAMATH_CALUDE_weekly_payment_problem_l1230_123075


namespace NUMINAMATH_CALUDE_degree_three_polynomial_l1230_123061

/-- The polynomial f(x) -/
def f (x : ℝ) : ℝ := 2 - 6*x + 4*x^2 - 5*x^3 + 7*x^4

/-- The polynomial g(x) -/
def g (x : ℝ) : ℝ := 4 - 3*x - 7*x^3 + 11*x^4

/-- The combined polynomial h(x) = f(x) + c*g(x) -/
def h (c : ℝ) (x : ℝ) : ℝ := f x + c * g x

/-- The coefficient of x^4 in h(x) -/
def coeff_x4 (c : ℝ) : ℝ := 7 + 11*c

/-- The coefficient of x^3 in h(x) -/
def coeff_x3 (c : ℝ) : ℝ := -5 - 7*c

theorem degree_three_polynomial :
  ∃ c : ℝ, coeff_x4 c = 0 ∧ coeff_x3 c ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_degree_three_polynomial_l1230_123061


namespace NUMINAMATH_CALUDE_ginas_college_expenses_l1230_123057

/-- Calculates the total college expenses for Gina -/
def total_college_expenses (credits : ℕ) (cost_per_credit : ℕ) (num_textbooks : ℕ) (cost_per_textbook : ℕ) (facilities_fee : ℕ) : ℕ :=
  credits * cost_per_credit + num_textbooks * cost_per_textbook + facilities_fee

/-- Proves that Gina's total college expenses are $7100 -/
theorem ginas_college_expenses :
  total_college_expenses 14 450 5 120 200 = 7100 := by
  sorry

end NUMINAMATH_CALUDE_ginas_college_expenses_l1230_123057


namespace NUMINAMATH_CALUDE_negative_difference_l1230_123027

theorem negative_difference (a b : ℝ) : -(a - b) = -a + b := by
  sorry

end NUMINAMATH_CALUDE_negative_difference_l1230_123027


namespace NUMINAMATH_CALUDE_fib_fraction_numerator_l1230_123063

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Theorem: The simplified fraction of (F₂₀₀₃/F₂₀₀₂ - F₂₀₀₄/F₂₀₀₃) has numerator 1 -/
theorem fib_fraction_numerator :
  (fib 2003 : ℚ) / fib 2002 - (fib 2004 : ℚ) / fib 2003 = 1 / (fib 2002 * fib 2003) :=
by sorry

end NUMINAMATH_CALUDE_fib_fraction_numerator_l1230_123063


namespace NUMINAMATH_CALUDE_gummy_bear_manufacturing_time_l1230_123016

/-- The time needed to manufacture gummy bears for a given number of packets -/
def manufacturingTime (bearsPerMinute : ℕ) (bearsPerPacket : ℕ) (numPackets : ℕ) : ℕ :=
  (numPackets * bearsPerPacket) / bearsPerMinute

theorem gummy_bear_manufacturing_time :
  manufacturingTime 300 50 240 = 40 := by
  sorry

end NUMINAMATH_CALUDE_gummy_bear_manufacturing_time_l1230_123016


namespace NUMINAMATH_CALUDE_distance_to_town_l1230_123069

theorem distance_to_town (d : ℝ) : 
  (¬ (d ≥ 8)) → (¬ (d ≤ 7)) → (¬ (d ≤ 6)) → (7 < d ∧ d < 8) :=
by sorry

end NUMINAMATH_CALUDE_distance_to_town_l1230_123069


namespace NUMINAMATH_CALUDE_product_expansion_l1230_123045

theorem product_expansion (x : ℝ) : (x + 3) * (x + 7) * (x - 2) = x^3 + 8*x^2 + x - 42 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l1230_123045


namespace NUMINAMATH_CALUDE_line_parameterization_l1230_123023

/-- Given a line y = 2x - 40 parameterized by (x,y) = (g(t), 20t - 14),
    prove that g(t) = 10t + 13 -/
theorem line_parameterization (g : ℝ → ℝ) :
  (∀ x y, y = 2*x - 40 ↔ ∃ t, x = g t ∧ y = 20*t - 14) →
  ∀ t, g t = 10*t + 13 := by
sorry

end NUMINAMATH_CALUDE_line_parameterization_l1230_123023


namespace NUMINAMATH_CALUDE_arithmetic_progression_rth_term_l1230_123036

/-- Definition of the sum function for the arithmetic progression -/
def S (n : ℕ) : ℝ := 4 * n + 5 * n^2

/-- The rth term of the arithmetic progression -/
def a (r : ℕ) : ℝ := 10 * r - 1

/-- Theorem stating that a(r) is the rth term of the arithmetic progression -/
theorem arithmetic_progression_rth_term (r : ℕ) :
  a r = S r - S (r - 1) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_rth_term_l1230_123036


namespace NUMINAMATH_CALUDE_square_equals_product_solution_l1230_123030

theorem square_equals_product_solution :
  ∀ a b : ℕ, a^2 = b * (b + 7) → (a = 0 ∧ b = 0) ∨ (a = 12 ∧ b = 9) := by
  sorry

end NUMINAMATH_CALUDE_square_equals_product_solution_l1230_123030


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1230_123046

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 - a*x + 2*a > 0) ↔ (0 < a ∧ a < 8) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1230_123046


namespace NUMINAMATH_CALUDE_smallest_n_for_factors_l1230_123081

theorem smallest_n_for_factors (k : ℕ) : 
  (∀ m : ℕ, m > 0 → (5^2 ∣ m * 2^k * 6^2 * 7^3) → (3^3 ∣ m * 2^k * 6^2 * 7^3) → m ≥ 75) ∧
  (5^2 ∣ 75 * 2^k * 6^2 * 7^3) ∧
  (3^3 ∣ 75 * 2^k * 6^2 * 7^3) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_factors_l1230_123081


namespace NUMINAMATH_CALUDE_range_of_h_l1230_123009

def f (x : ℝ) : ℝ := 4 * x - 3

def h (x : ℝ) : ℝ := f (f (f x))

theorem range_of_h :
  let S : Set ℝ := {y | ∃ x ∈ Set.Icc (-1 : ℝ) 3, h x = y}
  S = Set.Icc (-127 : ℝ) 129 := by
  sorry

end NUMINAMATH_CALUDE_range_of_h_l1230_123009


namespace NUMINAMATH_CALUDE_sum_of_squares_l1230_123006

theorem sum_of_squares (a b c : ℝ) 
  (eq1 : a^2 + 2*b = 7)
  (eq2 : b^2 + 4*c = -7)
  (eq3 : c^2 + 6*a = -14) :
  a^2 + b^2 + c^2 = 14 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1230_123006


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_implies_a_value_l1230_123094

-- Define the hyperbola
def hyperbola (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 = 1

-- Define the asymptote
def asymptote (x y : ℝ) : Prop :=
  Real.sqrt 3 * x + y = 0

-- Theorem statement
theorem hyperbola_asymptote_implies_a_value :
  ∀ a : ℝ, (∃ x y : ℝ, hyperbola a x y ∧ asymptote x y) →
  a = Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_implies_a_value_l1230_123094


namespace NUMINAMATH_CALUDE_function_relationship_l1230_123037

def main (f : ℝ → ℝ) : Prop :=
  (∀ x, f (2 - x) = f x) ∧
  (∀ x, f (x + 2) = f (x - 2)) ∧
  (∀ x₁ x₂, x₁ ∈ Set.Icc 1 3 → x₂ ∈ Set.Icc 1 3 → x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) < 0) →
  f 2016 = f 2014 ∧ f 2014 > f 2015

theorem function_relationship : main f :=
sorry

end NUMINAMATH_CALUDE_function_relationship_l1230_123037


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_one_l1230_123096

-- Define the displacement function
def displacement (t : ℝ) : ℝ := 2 * t^3

-- Define the velocity function as the derivative of displacement
def velocity (t : ℝ) : ℝ := 6 * t^2

-- Theorem statement
theorem instantaneous_velocity_at_one :
  velocity 1 = 6 := by sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_one_l1230_123096


namespace NUMINAMATH_CALUDE_jerry_cut_eight_pine_trees_l1230_123093

/-- The number of logs produced by one pine tree -/
def logs_per_pine : ℕ := 80

/-- The number of logs produced by one maple tree -/
def logs_per_maple : ℕ := 60

/-- The number of logs produced by one walnut tree -/
def logs_per_walnut : ℕ := 100

/-- The number of maple trees Jerry cut -/
def maple_trees : ℕ := 3

/-- The number of walnut trees Jerry cut -/
def walnut_trees : ℕ := 4

/-- The total number of logs Jerry got -/
def total_logs : ℕ := 1220

/-- Theorem stating that Jerry cut 8 pine trees -/
theorem jerry_cut_eight_pine_trees :
  ∃ (pine_trees : ℕ), pine_trees * logs_per_pine + 
                      maple_trees * logs_per_maple + 
                      walnut_trees * logs_per_walnut = total_logs ∧ 
                      pine_trees = 8 := by
  sorry

end NUMINAMATH_CALUDE_jerry_cut_eight_pine_trees_l1230_123093


namespace NUMINAMATH_CALUDE_unit_circle_complex_bound_l1230_123080

theorem unit_circle_complex_bound (z : ℂ) (h : Complex.abs z = 1) :
  ∃ (zmin zmax : ℂ),
    Complex.abs (z^3 - 3*z - 2) ≤ Real.sqrt 27 ∧
    Complex.abs (zmax^3 - 3*zmax - 2) = Real.sqrt 27 ∧
    Complex.abs (zmin^3 - 3*zmin - 2) = 0 ∧
    Complex.abs zmax = 1 ∧
    Complex.abs zmin = 1 :=
by sorry

end NUMINAMATH_CALUDE_unit_circle_complex_bound_l1230_123080


namespace NUMINAMATH_CALUDE_major_axis_length_major_axis_length_is_eight_l1230_123019

/-- An ellipse with foci at (3, -4 + 2√3) and (3, -4 - 2√3), tangent to both x and y axes -/
structure TangentEllipse where
  /-- The ellipse is tangent to the x-axis -/
  tangent_x : Bool
  /-- The ellipse is tangent to the y-axis -/
  tangent_y : Bool
  /-- The first focus of the ellipse -/
  focus1 : ℝ × ℝ
  /-- The second focus of the ellipse -/
  focus2 : ℝ × ℝ
  /-- Condition that the first focus is at (3, -4 + 2√3) -/
  h1 : focus1 = (3, -4 + 2 * Real.sqrt 3)
  /-- Condition that the second focus is at (3, -4 - 2√3) -/
  h2 : focus2 = (3, -4 - 2 * Real.sqrt 3)
  /-- Condition that the ellipse is tangent to the x-axis -/
  h3 : tangent_x = true
  /-- Condition that the ellipse is tangent to the y-axis -/
  h4 : tangent_y = true

/-- The length of the major axis of the ellipse is 8 -/
theorem major_axis_length (e : TangentEllipse) : ℝ :=
  8

/-- The theorem stating that the major axis length of the given ellipse is 8 -/
theorem major_axis_length_is_eight (e : TangentEllipse) : 
  major_axis_length e = 8 := by sorry

end NUMINAMATH_CALUDE_major_axis_length_major_axis_length_is_eight_l1230_123019


namespace NUMINAMATH_CALUDE_fathers_age_l1230_123000

/-- The father's age given the son's age ratio conditions -/
theorem fathers_age (man_age : ℝ) (father_age : ℝ) : 
  man_age = (2 / 5) * father_age ∧ 
  man_age + 14 = (1 / 2) * (father_age + 14) → 
  father_age = 70 := by
sorry

end NUMINAMATH_CALUDE_fathers_age_l1230_123000


namespace NUMINAMATH_CALUDE_greatest_x_value_l1230_123051

theorem greatest_x_value (x : ℤ) (h : (2.134 : ℝ) * (10 : ℝ) ^ (x : ℝ) < 21000) :
  x ≤ 3 ∧ ∃ y : ℤ, y > 3 → (2.134 : ℝ) * (10 : ℝ) ^ (y : ℝ) ≥ 21000 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_value_l1230_123051


namespace NUMINAMATH_CALUDE_inequality_proof_l1230_123049

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a * b * c ≤ 1 / 9 ∧
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1230_123049


namespace NUMINAMATH_CALUDE_green_brunette_percentage_is_54_l1230_123088

/-- Represents the hair and eye color distribution of an island's population -/
structure IslandPopulation where
  blueBrunettes : ℕ
  blueBlondes : ℕ
  greenBlondes : ℕ
  greenBrunettes : ℕ

/-- The proportion of brunettes among blue-eyed inhabitants is 65% -/
def blueBrunettesProportion (pop : IslandPopulation) : Prop :=
  (pop.blueBrunettes : ℚ) / (pop.blueBrunettes + pop.blueBlondes) = 13 / 20

/-- The proportion of blue-eyed among blondes is 70% -/
def blueBlondeProportion (pop : IslandPopulation) : Prop :=
  (pop.blueBlondes : ℚ) / (pop.blueBlondes + pop.greenBlondes) = 7 / 10

/-- The proportion of blondes among green-eyed inhabitants is 10% -/
def greenBlondeProportion (pop : IslandPopulation) : Prop :=
  (pop.greenBlondes : ℚ) / (pop.greenBlondes + pop.greenBrunettes) = 1 / 10

/-- The percentage of green-eyed brunettes in the total population -/
def greenBrunettePercentage (pop : IslandPopulation) : ℚ :=
  (pop.greenBrunettes : ℚ) / (pop.blueBrunettes + pop.blueBlondes + pop.greenBlondes + pop.greenBrunettes) * 100

/-- Theorem stating that the percentage of green-eyed brunettes is 54% -/
theorem green_brunette_percentage_is_54 (pop : IslandPopulation) :
  blueBrunettesProportion pop → blueBlondeProportion pop → greenBlondeProportion pop →
  greenBrunettePercentage pop = 54 := by
  sorry

end NUMINAMATH_CALUDE_green_brunette_percentage_is_54_l1230_123088


namespace NUMINAMATH_CALUDE_baseball_division_games_l1230_123034

theorem baseball_division_games 
  (N M : ℕ) 
  (h1 : N > 2 * M) 
  (h2 : M > 4) 
  (h3 : 2 * N + 5 * M = 82) : 
  2 * N = 52 := by
  sorry

end NUMINAMATH_CALUDE_baseball_division_games_l1230_123034


namespace NUMINAMATH_CALUDE_linda_classmates_l1230_123029

/-- The number of cookies each student receives -/
def cookies_per_student : ℕ := 10

/-- The number of cookies in one dozen -/
def cookies_per_dozen : ℕ := 12

/-- The number of dozens of cookies in each batch -/
def dozens_per_batch : ℕ := 4

/-- The number of batches of chocolate chip cookies Linda made -/
def chocolate_chip_batches : ℕ := 2

/-- The number of batches of oatmeal raisin cookies Linda made -/
def oatmeal_raisin_batches : ℕ := 1

/-- The number of additional batches Linda needs to bake -/
def additional_batches : ℕ := 2

/-- The total number of cookies Linda will have after baking all batches -/
def total_cookies : ℕ := 
  (chocolate_chip_batches + oatmeal_raisin_batches + additional_batches) * 
  dozens_per_batch * cookies_per_dozen

/-- The number of Linda's classmates -/
def number_of_classmates : ℕ := total_cookies / cookies_per_student

theorem linda_classmates : number_of_classmates = 24 := by
  sorry

end NUMINAMATH_CALUDE_linda_classmates_l1230_123029


namespace NUMINAMATH_CALUDE_max_t_is_e_l1230_123048

theorem max_t_is_e (t : ℝ) : 
  (∀ a b : ℝ, 0 < a → a < b → b < t → b * Real.log a < a * Real.log b) →
  t ≤ Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_max_t_is_e_l1230_123048


namespace NUMINAMATH_CALUDE_angle_between_c_and_a_plus_b_is_zero_l1230_123031

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem angle_between_c_and_a_plus_b_is_zero
  (a b c : V)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (hab : ‖a‖ = ‖b‖)
  (habgt : ‖a‖ > ‖a + b‖)
  (hc_eq : ‖c‖ = ‖a + b‖) :
  Real.arccos (inner c (a + b) / (‖c‖ * ‖a + b‖)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_c_and_a_plus_b_is_zero_l1230_123031


namespace NUMINAMATH_CALUDE_handshake_problem_l1230_123058

theorem handshake_problem (a b : ℕ) : 
  a + b = 20 →
  (a * (a - 1)) / 2 + (b * (b - 1)) / 2 = 106 →
  a * b = 84 := by
sorry

end NUMINAMATH_CALUDE_handshake_problem_l1230_123058


namespace NUMINAMATH_CALUDE_cos_pi_fourth_plus_alpha_l1230_123066

theorem cos_pi_fourth_plus_alpha (α : ℝ) 
  (h : Real.sin (π / 4 - α) = 1 / 2) : 
  Real.cos (π / 4 + α) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_fourth_plus_alpha_l1230_123066


namespace NUMINAMATH_CALUDE_smallest_non_factor_product_l1230_123099

def is_factor (n m : ℕ) : Prop := m % n = 0

theorem smallest_non_factor_product (a b : ℕ) : 
  a ≠ b → 
  a > 0 → 
  b > 0 → 
  is_factor a 48 → 
  is_factor b 48 → 
  ¬(is_factor (a * b) 48) → 
  (∀ (x y : ℕ), x ≠ y → x > 0 → y > 0 → is_factor x 48 → is_factor y 48 → 
    ¬(is_factor (x * y) 48) → a * b ≤ x * y) → 
  a * b = 18 :=
sorry

end NUMINAMATH_CALUDE_smallest_non_factor_product_l1230_123099


namespace NUMINAMATH_CALUDE_f_increasing_iff_a_ge_four_l1230_123056

/-- The function f(x) = ax - x^3 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x - x^3

/-- The theorem statement -/
theorem f_increasing_iff_a_ge_four (a : ℝ) :
  (∀ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 ∧ x2 < 1 → f a x2 - f a x1 > x2 - x1) ↔ a ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_iff_a_ge_four_l1230_123056


namespace NUMINAMATH_CALUDE_value_added_to_numbers_l1230_123013

theorem value_added_to_numbers (n : ℕ) (original_avg new_avg x : ℝ) 
  (h1 : n = 15)
  (h2 : original_avg = 40)
  (h3 : new_avg = 54)
  (h4 : n * new_avg = n * original_avg + n * x) :
  x = 14 := by
  sorry

end NUMINAMATH_CALUDE_value_added_to_numbers_l1230_123013


namespace NUMINAMATH_CALUDE_paths_AC_count_l1230_123021

/-- The number of paths from A to B -/
def paths_AB : Nat := 2

/-- The number of paths from B to C -/
def paths_BC : Nat := 2

/-- The number of direct paths from A to C -/
def direct_paths_AC : Nat := 1

/-- The total number of paths from A to C -/
def total_paths_AC : Nat := paths_AB * paths_BC + direct_paths_AC

theorem paths_AC_count : total_paths_AC = 5 := by
  sorry

end NUMINAMATH_CALUDE_paths_AC_count_l1230_123021


namespace NUMINAMATH_CALUDE_average_weight_problem_l1230_123018

/-- The average weight problem -/
theorem average_weight_problem 
  (weight_A weight_B weight_C weight_D : ℝ)
  (h1 : (weight_A + weight_B + weight_C) / 3 = 60)
  (h2 : weight_A = 87)
  (h3 : (weight_B + weight_C + weight_D + (weight_D + 3)) / 4 = 64) :
  (weight_A + weight_B + weight_C + weight_D) / 4 = 65 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_problem_l1230_123018


namespace NUMINAMATH_CALUDE_non_student_ticket_price_l1230_123022

/-- Proves that the price of a non-student ticket was $8 -/
theorem non_student_ticket_price :
  let total_tickets : ℕ := 150
  let student_ticket_price : ℕ := 5
  let total_revenue : ℕ := 930
  let student_tickets_sold : ℕ := 90
  let non_student_tickets_sold : ℕ := 60
  let non_student_ticket_price : ℕ := (total_revenue - student_ticket_price * student_tickets_sold) / non_student_tickets_sold
  non_student_ticket_price = 8 := by
  sorry

end NUMINAMATH_CALUDE_non_student_ticket_price_l1230_123022


namespace NUMINAMATH_CALUDE_sequence_150th_term_l1230_123047

def sequence_term (n : ℕ) : ℕ := sorry

theorem sequence_150th_term : sequence_term 150 = 2280 := by sorry

end NUMINAMATH_CALUDE_sequence_150th_term_l1230_123047


namespace NUMINAMATH_CALUDE_sum_divisible_by_143_l1230_123002

theorem sum_divisible_by_143 : ∃ k : ℕ, (1000 * 1001) / 2 = 143 * k := by
  sorry

end NUMINAMATH_CALUDE_sum_divisible_by_143_l1230_123002


namespace NUMINAMATH_CALUDE_faculty_reduction_l1230_123089

theorem faculty_reduction (original : ℝ) (reduction_percentage : ℝ) : 
  original = 253.25 → 
  reduction_percentage = 0.23 →
  ⌊original - (original * reduction_percentage)⌋ = 195 := by
sorry

end NUMINAMATH_CALUDE_faculty_reduction_l1230_123089


namespace NUMINAMATH_CALUDE_infinitely_many_friendly_squares_l1230_123076

/-- A number is friendly if the set {1, 2, ..., N} can be partitioned into disjoint pairs 
    where the sum of each pair is a perfect square -/
def is_friendly (N : ℕ) : Prop :=
  ∃ (partition : List (ℕ × ℕ)), 
    (∀ (pair : ℕ × ℕ), pair ∈ partition → pair.1 ∈ Finset.range N ∧ pair.2 ∈ Finset.range N) ∧
    (∀ n ∈ Finset.range N, ∃ (pair : ℕ × ℕ), pair ∈ partition ∧ (n = pair.1 ∨ n = pair.2)) ∧
    (∀ (pair : ℕ × ℕ), pair ∈ partition → ∃ k : ℕ, pair.1 + pair.2 = k^2)

/-- There are infinitely many friendly perfect squares -/
theorem infinitely_many_friendly_squares :
  ∀ (p : ℕ), p ≥ 2 → ∃ (N : ℕ), N = 2^(2*p - 3) ∧ is_friendly N ∧ ∃ (k : ℕ), N = k^2 :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_friendly_squares_l1230_123076


namespace NUMINAMATH_CALUDE_solve_salary_problem_l1230_123068

def salary_problem (salary : ℝ) : Prop :=
  let food_expense := (1 / 5 : ℝ) * salary
  let rent_expense := (1 / 10 : ℝ) * salary
  let clothes_expense := (3 / 5 : ℝ) * salary
  let remaining := salary - (food_expense + rent_expense + clothes_expense)
  remaining = 17000

theorem solve_salary_problem : 
  ∃ (salary : ℝ), salary_problem salary ∧ salary = 170000 := by
  sorry

end NUMINAMATH_CALUDE_solve_salary_problem_l1230_123068


namespace NUMINAMATH_CALUDE_two_disjoint_triangles_l1230_123014

/-- A graph with 7 vertices where each vertex has degree 4 -/
structure SouslikGraph where
  vertices : Finset (Fin 7)
  edges : Finset (Fin 7 × Fin 7)
  vertex_count : vertices.card = 7
  edge_symmetric : ∀ (u v : Fin 7), (u, v) ∈ edges ↔ (v, u) ∈ edges
  degree_four : ∀ (v : Fin 7), (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card = 4

/-- A triangle in the graph -/
def Triangle (G : SouslikGraph) (t : Finset (Fin 7)) : Prop :=
  t.card = 3 ∧ ∀ (u v : Fin 7), u ∈ t → v ∈ t → u ≠ v → (u, v) ∈ G.edges

/-- The main theorem: there always exist two disjoint triangles -/
theorem two_disjoint_triangles (G : SouslikGraph) :
  ∃ (t1 t2 : Finset (Fin 7)), Triangle G t1 ∧ Triangle G t2 ∧ t1 ∩ t2 = ∅ := by
  sorry

end NUMINAMATH_CALUDE_two_disjoint_triangles_l1230_123014


namespace NUMINAMATH_CALUDE_exists_problem_solved_by_all_l1230_123001

/-- Represents a problem on the exam -/
def Problem : Type := ℕ

/-- Represents a student in the class -/
def Student : Type := ℕ

/-- Given n students and 2^(n-1) problems, if for each pair of distinct problems
    there is at least one student who has solved both and at least one student
    who has solved one but not the other, then there exists a problem solved by
    all n students. -/
theorem exists_problem_solved_by_all
  (n : ℕ)
  (problems : Finset Problem)
  (students : Finset Student)
  (solved : Problem → Student → Prop)
  (h_num_students : students.card = n)
  (h_num_problems : problems.card = 2^(n-1))
  (h_solved_both : ∀ p q : Problem, p ≠ q →
    ∃ s : Student, solved p s ∧ solved q s)
  (h_solved_one_not_other : ∀ p q : Problem, p ≠ q →
    ∃ s : Student, (solved p s ∧ ¬solved q s) ∨ (solved q s ∧ ¬solved p s)) :
  ∃ p : Problem, ∀ s : Student, solved p s :=
sorry

end NUMINAMATH_CALUDE_exists_problem_solved_by_all_l1230_123001
