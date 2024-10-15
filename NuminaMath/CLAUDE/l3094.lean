import Mathlib

namespace NUMINAMATH_CALUDE_merchandise_profit_analysis_l3094_309472

/-- Represents a store's merchandise sales model -/
structure MerchandiseModel where
  original_cost : ℝ
  original_price : ℝ
  original_sales : ℝ
  price_decrease_step : ℝ
  sales_increase_step : ℝ

/-- Calculate the profit given a price decrease -/
def profit (model : MerchandiseModel) (price_decrease : ℝ) : ℝ :=
  (model.original_sales + model.sales_increase_step * price_decrease) *
  (model.original_price - price_decrease - model.original_cost)

theorem merchandise_profit_analysis (model : MerchandiseModel)
  (h1 : model.original_cost = 80)
  (h2 : model.original_price = 100)
  (h3 : model.original_sales = 100)
  (h4 : model.price_decrease_step = 1)
  (h5 : model.sales_increase_step = 10) :
  profit model 0 = 2000 ∧
  (∀ x, profit model x = -10 * x^2 + 100 * x + 2000) ∧
  (∃ x, profit model x = 2250 ∧ ∀ y, profit model y ≤ profit model x) ∧
  (∀ p, 92 ≤ p ∧ p ≤ 98 ↔ profit model (100 - p) ≥ 2160) :=
by sorry

end NUMINAMATH_CALUDE_merchandise_profit_analysis_l3094_309472


namespace NUMINAMATH_CALUDE_harper_gift_cost_l3094_309440

/-- The total amount spent on teacher appreciation gifts --/
def total_gift_cost (son_teachers daughter_teachers gift_cost : ℕ) : ℕ :=
  (son_teachers + daughter_teachers) * gift_cost

/-- Theorem: Harper's total gift cost is $70 --/
theorem harper_gift_cost :
  total_gift_cost 3 4 10 = 70 := by
  sorry

end NUMINAMATH_CALUDE_harper_gift_cost_l3094_309440


namespace NUMINAMATH_CALUDE_football_team_ratio_l3094_309426

/-- Given a football team with the following properties:
  * There are 70 players in total
  * 52 players are throwers
  * All throwers are right-handed
  * There are 64 right-handed players in total
  Prove that the ratio of left-handed players to non-throwers is 1:3 -/
theorem football_team_ratio (total_players : ℕ) (throwers : ℕ) (right_handed : ℕ)
  (h1 : total_players = 70)
  (h2 : throwers = 52)
  (h3 : right_handed = 64) :
  (total_players - right_handed) * 3 = total_players - throwers :=
by sorry

end NUMINAMATH_CALUDE_football_team_ratio_l3094_309426


namespace NUMINAMATH_CALUDE_delivery_speed_l3094_309477

/-- Given the conditions of the delivery problem, prove that the required average speed is 30 km/h -/
theorem delivery_speed (d : ℝ) (t : ℝ) (v : ℝ) : 
  (d / 60 = t - 1/4) →  -- Condition for moderate traffic
  (d / 20 = t + 1/4) →  -- Condition for traffic jams
  (d / v = 1/2) →       -- Condition for arriving exactly at 18:00
  v = 30 := by
  sorry

end NUMINAMATH_CALUDE_delivery_speed_l3094_309477


namespace NUMINAMATH_CALUDE_remaining_dogs_l3094_309452

theorem remaining_dogs (total_pets : ℕ) (dogs_given : ℕ) : 
  total_pets = 189 → dogs_given = 10 → 
  (10 : ℚ) / 27 * total_pets - dogs_given = 60 := by
sorry

end NUMINAMATH_CALUDE_remaining_dogs_l3094_309452


namespace NUMINAMATH_CALUDE_sin_90_degrees_l3094_309483

theorem sin_90_degrees : Real.sin (π / 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_90_degrees_l3094_309483


namespace NUMINAMATH_CALUDE_preimage_of_5_1_l3094_309465

/-- The mapping f that transforms a point (x, y) to (x+y, 2x-y) -/
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, 2 * p.1 - p.2)

/-- Theorem stating that the pre-image of (5, 1) under f is (2, 3) -/
theorem preimage_of_5_1 : f (2, 3) = (5, 1) := by sorry

end NUMINAMATH_CALUDE_preimage_of_5_1_l3094_309465


namespace NUMINAMATH_CALUDE_quadratic_sets_problem_l3094_309442

theorem quadratic_sets_problem (p q : ℝ) :
  let A := {x : ℝ | x^2 + p*x + 15 = 0}
  let B := {x : ℝ | x^2 - 5*x + q = 0}
  (A ∩ B = {3}) →
  (p = -8 ∧ q = 6 ∧ A ∪ B = {2, 3, 5}) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_sets_problem_l3094_309442


namespace NUMINAMATH_CALUDE_point_transformation_l3094_309446

def rotate_270_clockwise (x y h k : ℝ) : ℝ × ℝ :=
  (h - (y - k), k + (x - h))

def reflect_about_y_eq_x (x y : ℝ) : ℝ × ℝ :=
  (y, x)

theorem point_transformation (a b : ℝ) :
  let (x₁, y₁) := rotate_270_clockwise a b 2 3
  let (x₂, y₂) := reflect_about_y_eq_x x₁ y₁
  (x₂ = 4 ∧ y₂ = -7) → b - a = -7 := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_l3094_309446


namespace NUMINAMATH_CALUDE_red_balls_count_l3094_309431

theorem red_balls_count (total_balls : ℕ) (red_probability : ℚ) (h1 : total_balls = 20) (h2 : red_probability = 1/4) :
  (red_probability * total_balls : ℚ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_red_balls_count_l3094_309431


namespace NUMINAMATH_CALUDE_borrowing_interest_rate_l3094_309475

/-- Proves that the interest rate at which a person borrowed money is 4% per annum
    given the specified conditions. -/
theorem borrowing_interest_rate : 
  ∀ (principal : ℝ) (borrowing_time : ℝ) (lending_rate : ℝ) (lending_time : ℝ) (yearly_gain : ℝ),
  principal = 5000 →
  borrowing_time = 2 →
  lending_rate = 0.06 →
  lending_time = 2 →
  yearly_gain = 100 →
  ∃ (borrowing_rate : ℝ),
    borrowing_rate = 0.04 ∧
    principal * lending_rate * lending_time - 
    principal * borrowing_rate * borrowing_time = 
    yearly_gain * borrowing_time :=
by sorry

end NUMINAMATH_CALUDE_borrowing_interest_rate_l3094_309475


namespace NUMINAMATH_CALUDE_odds_against_C_winning_l3094_309435

-- Define the type for horses
inductive Horse : Type
| A
| B
| C

-- Define the function for odds against winning
def oddsAgainst (h : Horse) : ℚ :=
  match h with
  | Horse.A => 4 / 1
  | Horse.B => 3 / 4
  | Horse.C => 27 / 8

-- State the theorem
theorem odds_against_C_winning :
  (∀ h : Horse, oddsAgainst h > 0) →  -- Ensure all odds are positive
  (∀ h1 h2 : Horse, h1 ≠ h2 → oddsAgainst h1 ≠ oddsAgainst h2) →  -- No ties
  oddsAgainst Horse.A = 4 / 1 →
  oddsAgainst Horse.B = 3 / 4 →
  oddsAgainst Horse.C = 27 / 8 := by
  sorry


end NUMINAMATH_CALUDE_odds_against_C_winning_l3094_309435


namespace NUMINAMATH_CALUDE_intersection_unique_l3094_309478

/-- The system of linear equations representing two lines -/
def line_system (x y : ℚ) : Prop :=
  8 * x - 5 * y = 40 ∧ 6 * x + 2 * y = 14

/-- The intersection point of the two lines -/
def intersection_point : ℚ × ℚ := (75/23, -64/23)

/-- Theorem stating that the intersection point is the unique solution to the system of equations -/
theorem intersection_unique :
  line_system intersection_point.1 intersection_point.2 ∧
  ∀ x y, line_system x y → (x, y) = intersection_point :=
by sorry

end NUMINAMATH_CALUDE_intersection_unique_l3094_309478


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l3094_309421

theorem simplify_complex_fraction (y : ℝ) 
  (h1 : y ≠ 4) (h2 : y ≠ 2) (h3 : y ≠ 5) (h4 : y ≠ 7) (h5 : y ≠ 1) :
  (y^2 - 4*y + 3) / (y^2 - 6*y + 8) / ((y^2 - 9*y + 20) / (y^2 - 9*y + 14)) = 
  ((y - 3) * (y - 7)) / ((y - 1) * (y - 5)) :=
by sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l3094_309421


namespace NUMINAMATH_CALUDE_claire_balloons_l3094_309476

/-- The number of balloons Claire has at the end of the fair --/
def final_balloon_count (initial : ℕ) (given_to_girl : ℕ) (floated_away : ℕ) (given_away : ℕ) (taken_from_coworker : ℕ) : ℕ :=
  initial - given_to_girl - floated_away - given_away + taken_from_coworker

/-- Theorem stating that Claire ends up with 39 balloons --/
theorem claire_balloons : 
  final_balloon_count 50 1 12 9 11 = 39 := by
  sorry

end NUMINAMATH_CALUDE_claire_balloons_l3094_309476


namespace NUMINAMATH_CALUDE_no_valid_n_l3094_309410

theorem no_valid_n : ¬∃ (n : ℕ), n > 0 ∧ 
  (100 ≤ n / 4 ∧ n / 4 ≤ 999) ∧ 
  (100 ≤ 4 * n ∧ 4 * n ≤ 999) := by
  sorry

end NUMINAMATH_CALUDE_no_valid_n_l3094_309410


namespace NUMINAMATH_CALUDE_bill_difference_l3094_309413

theorem bill_difference (john_tip peter_tip : ℝ) (john_percent peter_percent : ℝ) :
  john_tip = 4 →
  peter_tip = 3 →
  john_percent = 20 / 100 →
  peter_percent = 15 / 100 →
  john_tip = john_percent * (john_tip / john_percent) →
  peter_tip = peter_percent * (peter_tip / peter_percent) →
  (john_tip / john_percent) - (peter_tip / peter_percent) = 0 :=
by
  sorry

#check bill_difference

end NUMINAMATH_CALUDE_bill_difference_l3094_309413


namespace NUMINAMATH_CALUDE_a_less_than_two_necessary_and_sufficient_l3094_309492

theorem a_less_than_two_necessary_and_sufficient (a : ℝ) :
  (∀ x : ℝ, |x - 2| + |x| > a) ↔ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_a_less_than_two_necessary_and_sufficient_l3094_309492


namespace NUMINAMATH_CALUDE_symmetric_implies_abs_even_abs_even_not_sufficient_for_symmetric_l3094_309495

/-- A function f: ℝ → ℝ is symmetric about the origin if f(-x) = -f(x) for all x ∈ ℝ -/
def SymmetricAboutOrigin (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem symmetric_implies_abs_even (f : ℝ → ℝ) :
  SymmetricAboutOrigin f → EvenFunction (fun x ↦ |f x|) :=
by sorry

theorem abs_even_not_sufficient_for_symmetric :
  ∃ f : ℝ → ℝ, EvenFunction (fun x ↦ |f x|) ∧ ¬SymmetricAboutOrigin f :=
by sorry

end NUMINAMATH_CALUDE_symmetric_implies_abs_even_abs_even_not_sufficient_for_symmetric_l3094_309495


namespace NUMINAMATH_CALUDE_intersection_M_N_l3094_309414

def M : Set ℝ := {x | |x| ≤ 2}
def N : Set ℝ := {x | x^2 + 2*x - 3 ≤ 0}

theorem intersection_M_N : M ∩ N = {x : ℝ | -2 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3094_309414


namespace NUMINAMATH_CALUDE_coin_problem_l3094_309433

def is_valid_amount (n : ℕ) : Prop :=
  ∃ (x : ℕ), 
    n = 5 * x ∧ 
    n ≤ 100000 ∧ 
    x % 12 = 3 ∧ 
    x % 18 = 3 ∧ 
    x % 45 = 3 ∧ 
    x % 11 = 0

def valid_amounts : Set ℕ :=
  {1815, 11715, 21615, 31515, 41415, 51315, 61215, 71115, 81015, 90915}

theorem coin_problem : 
  ∀ n : ℕ, is_valid_amount n ↔ n ∈ valid_amounts :=
sorry

end NUMINAMATH_CALUDE_coin_problem_l3094_309433


namespace NUMINAMATH_CALUDE_difference_of_squares_l3094_309469

theorem difference_of_squares (m : ℝ) : m^2 - 4 = (m + 2) * (m - 2) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3094_309469


namespace NUMINAMATH_CALUDE_percentage_4_plus_years_l3094_309471

/-- Represents the number of employees in each year group -/
structure EmployeeDistribution :=
  (less_than_1 : ℕ)
  (one_to_2 : ℕ)
  (two_to_3 : ℕ)
  (three_to_4 : ℕ)
  (four_to_5 : ℕ)
  (five_to_6 : ℕ)
  (six_to_7 : ℕ)
  (seven_to_8 : ℕ)
  (eight_to_9 : ℕ)
  (nine_to_10 : ℕ)
  (ten_plus : ℕ)

/-- Calculates the total number of employees -/
def total_employees (d : EmployeeDistribution) : ℕ :=
  d.less_than_1 + d.one_to_2 + d.two_to_3 + d.three_to_4 + d.four_to_5 + 
  d.five_to_6 + d.six_to_7 + d.seven_to_8 + d.eight_to_9 + d.nine_to_10 + d.ten_plus

/-- Calculates the number of employees who have worked for 4 years or more -/
def employees_4_plus_years (d : EmployeeDistribution) : ℕ :=
  d.four_to_5 + d.five_to_6 + d.six_to_7 + d.seven_to_8 + d.eight_to_9 + d.nine_to_10 + d.ten_plus

/-- Theorem: The percentage of employees who have worked for 4 years or more is 37.5% -/
theorem percentage_4_plus_years (d : EmployeeDistribution) : 
  (employees_4_plus_years d : ℚ) / (total_employees d : ℚ) = 375 / 1000 :=
sorry


end NUMINAMATH_CALUDE_percentage_4_plus_years_l3094_309471


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l3094_309422

theorem quadratic_equal_roots (c : ℝ) : 
  (∃ x : ℝ, x^2 - 4*x + c = 0 ∧ (∀ y : ℝ, y^2 - 4*y + c = 0 → y = x)) → c = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l3094_309422


namespace NUMINAMATH_CALUDE_direct_variation_problem_l3094_309402

-- Define the direct variation relationship
def direct_variation (y x : ℝ) := ∃ k : ℝ, y = k * x

-- State the theorem
theorem direct_variation_problem :
  ∀ y : ℝ → ℝ,
  (∀ x : ℝ, direct_variation (y x) x) →
  y 4 = 8 →
  y (-8) = -16 :=
by
  sorry

end NUMINAMATH_CALUDE_direct_variation_problem_l3094_309402


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3094_309420

theorem quadratic_factorization (p q : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ = -2 ∧ x₂ = 3/2 ∧ 
   ∀ x : ℝ, 2*x^2 + p*x + q = 0 ↔ x = x₁ ∨ x = x₂) →
  ∀ x : ℝ, 2*x^2 + p*x + q = 0 ↔ (x + 2)*(2*x - 3) = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3094_309420


namespace NUMINAMATH_CALUDE_sum_prime_factors_2310_l3094_309415

def prime_factors (n : ℕ) : List ℕ := sorry

theorem sum_prime_factors_2310 : (prime_factors 2310).sum = 28 := by sorry

end NUMINAMATH_CALUDE_sum_prime_factors_2310_l3094_309415


namespace NUMINAMATH_CALUDE_cube_sum_theorem_l3094_309487

theorem cube_sum_theorem (a b c : ℝ) 
  (sum_eq : a + b + c = 4)
  (sum_prod_eq : a * b + a * c + b * c = 6)
  (prod_eq : a * b * c = -8) :
  a^3 + b^3 + c^3 = 8 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_theorem_l3094_309487


namespace NUMINAMATH_CALUDE_smallest_distance_complex_circles_l3094_309432

/-- The smallest possible distance between two complex numbers on given circles -/
theorem smallest_distance_complex_circles :
  ∀ (z w : ℂ),
  Complex.abs (z - (2 + 4*Complex.I)) = 2 →
  Complex.abs (w - (8 + 6*Complex.I)) = 4 →
  ∀ (d : ℝ),
  d = Complex.abs (z - w) →
  d ≥ Real.sqrt 10 - 6 ∧
  ∃ (z₀ w₀ : ℂ),
    Complex.abs (z₀ - (2 + 4*Complex.I)) = 2 ∧
    Complex.abs (w₀ - (8 + 6*Complex.I)) = 4 ∧
    Complex.abs (z₀ - w₀) = Real.sqrt 10 - 6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_distance_complex_circles_l3094_309432


namespace NUMINAMATH_CALUDE_fred_has_61_cards_l3094_309447

/-- The number of baseball cards Fred has after all transactions -/
def fred_final_cards (initial : ℕ) (given_to_mary : ℕ) (found_in_box : ℕ) (given_to_john : ℕ) (purchased : ℕ) : ℕ :=
  initial - given_to_mary + found_in_box - given_to_john + purchased

/-- Theorem stating that Fred ends up with 61 cards -/
theorem fred_has_61_cards :
  fred_final_cards 26 18 40 12 25 = 61 := by
  sorry

end NUMINAMATH_CALUDE_fred_has_61_cards_l3094_309447


namespace NUMINAMATH_CALUDE_arith_seq_mono_increasing_iff_a2_gt_a1_l3094_309470

/-- An arithmetic sequence -/
def ArithmeticSeq (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A monotonically increasing sequence -/
def MonoIncreasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) > a n

/-- Theorem: For an arithmetic sequence, a_2 > a_1 is equivalent to the sequence being monotonically increasing -/
theorem arith_seq_mono_increasing_iff_a2_gt_a1 (a : ℕ → ℝ) (h : ArithmeticSeq a) :
  a 2 > a 1 ↔ MonoIncreasing a := by sorry

end NUMINAMATH_CALUDE_arith_seq_mono_increasing_iff_a2_gt_a1_l3094_309470


namespace NUMINAMATH_CALUDE_flight_duration_theorem_l3094_309416

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  hk : minutes < 60

/-- Calculates the difference between two times in minutes -/
def timeDifferenceInMinutes (t1 t2 : Time) : ℕ :=
  (t2.hours - t1.hours) * 60 + (t2.minutes - t1.minutes)

/-- Converts minutes to hours and minutes -/
def minutesToTime (totalMinutes : ℕ) : Time :=
  { hours := totalMinutes / 60,
    minutes := totalMinutes % 60,
    hk := by sorry }

theorem flight_duration_theorem (departureTime arrivalTime : Time) 
    (hDepart : departureTime.hours = 9 ∧ departureTime.minutes = 20)
    (hArrive : arrivalTime.hours = 13 ∧ arrivalTime.minutes = 45)
    (delay : ℕ)
    (hDelay : delay = 25) :
  let actualDuration := minutesToTime (timeDifferenceInMinutes departureTime arrivalTime + delay)
  actualDuration.hours + actualDuration.minutes = 29 := by
  sorry

end NUMINAMATH_CALUDE_flight_duration_theorem_l3094_309416


namespace NUMINAMATH_CALUDE_f_increasing_iff_a_in_range_l3094_309486

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 4 then a * x - 8 else x^2 - 2 * a * x

-- Define what it means for a function to be increasing
def IsIncreasing (g : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → g x < g y

-- State the theorem
theorem f_increasing_iff_a_in_range (a : ℝ) :
  IsIncreasing (f a) ↔ (0 < a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_f_increasing_iff_a_in_range_l3094_309486


namespace NUMINAMATH_CALUDE_x_over_y_value_l3094_309479

theorem x_over_y_value (x y : ℝ) (h1 : x * y = 1) (h2 : x > 0) (h3 : y > 0) (h4 : y = 0.16666666666666666) :
  x / y = 36 := by
  sorry

end NUMINAMATH_CALUDE_x_over_y_value_l3094_309479


namespace NUMINAMATH_CALUDE_second_term_is_twelve_l3094_309403

/-- A geometric sequence with a sum formula -/
structure GeometricSequence where
  a : ℝ  -- The common ratio multiplier
  sequence : ℕ → ℝ
  sum : ℕ → ℝ
  sum_formula : ∀ n : ℕ, sum n = a * 3^n - 2
  is_geometric : ∀ n : ℕ, sequence (n + 2) * sequence n = (sequence (n + 1))^2

/-- The second term of the geometric sequence is 12 -/
theorem second_term_is_twelve (seq : GeometricSequence) : seq.sequence 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_second_term_is_twelve_l3094_309403


namespace NUMINAMATH_CALUDE_parallel_lines_symmetry_intersecting_lines_symmetry_l3094_309460

-- Define a type for lines in a plane
structure Line2D where
  slope : ℝ
  intercept : ℝ

-- Define a type for points in a plane
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a function to check if two lines are parallel
def are_parallel (l1 l2 : Line2D) : Prop :=
  l1.slope = l2.slope

-- Define a function to check if two lines intersect
def intersect (l1 l2 : Line2D) : Prop :=
  ¬(are_parallel l1 l2)

-- Define a type for axis of symmetry
structure AxisOfSymmetry where
  line : Line2D

-- Define a type for center of symmetry
structure CenterOfSymmetry where
  point : Point2D

-- Theorem for parallel lines
theorem parallel_lines_symmetry (l1 l2 : Line2D) (h : are_parallel l1 l2) :
  ∃ (axis : AxisOfSymmetry), (∀ (center : CenterOfSymmetry), True) :=
sorry

-- Theorem for intersecting lines
theorem intersecting_lines_symmetry (l1 l2 : Line2D) (h : intersect l1 l2) :
  ∃ (axis1 axis2 : AxisOfSymmetry) (center : CenterOfSymmetry),
    axis1.line.slope * axis2.line.slope = -1 :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_symmetry_intersecting_lines_symmetry_l3094_309460


namespace NUMINAMATH_CALUDE_no_integer_cube_equals_3n2_plus_3n_plus_7_l3094_309428

theorem no_integer_cube_equals_3n2_plus_3n_plus_7 :
  ¬ ∃ (m n : ℤ), m^3 = 3*n^2 + 3*n + 7 := by
sorry

end NUMINAMATH_CALUDE_no_integer_cube_equals_3n2_plus_3n_plus_7_l3094_309428


namespace NUMINAMATH_CALUDE_new_average_production_l3094_309419

/-- Given a company's production data, prove that the new average daily production is 45 units. -/
theorem new_average_production (n : ℕ) (past_average : ℝ) (today_production : ℝ) :
  n = 9 →
  past_average = 40 →
  today_production = 90 →
  (n * past_average + today_production) / (n + 1) = 45 := by
  sorry

end NUMINAMATH_CALUDE_new_average_production_l3094_309419


namespace NUMINAMATH_CALUDE_p_shape_points_for_10cm_square_l3094_309434

/-- Calculates the number of unique points on a "P" shape formed from a square --/
def count_points_on_p_shape (side_length : ℕ) : ℕ :=
  let points_per_side := side_length + 1
  let total_points := points_per_side * 3
  let corner_points := 2
  total_points - corner_points

/-- Theorem stating that a "P" shape formed from a 10 cm square has 31 unique points --/
theorem p_shape_points_for_10cm_square :
  count_points_on_p_shape 10 = 31 := by
  sorry

#eval count_points_on_p_shape 10

end NUMINAMATH_CALUDE_p_shape_points_for_10cm_square_l3094_309434


namespace NUMINAMATH_CALUDE_min_people_to_remove_l3094_309474

def total_people : ℕ := 73

def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def people_to_remove (n : ℕ) : ℕ := total_people - n

theorem min_people_to_remove :
  ∃ n : ℕ, is_square n ∧
    (∀ m : ℕ, is_square m → people_to_remove m ≥ people_to_remove n) ∧
    people_to_remove n = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_people_to_remove_l3094_309474


namespace NUMINAMATH_CALUDE_fgh_supermarkets_l3094_309438

theorem fgh_supermarkets (total : ℕ) (difference : ℕ) (us_count : ℕ) : 
  total = 84 → difference = 10 → us_count = total / 2 + difference / 2 → us_count = 47 := by
  sorry

end NUMINAMATH_CALUDE_fgh_supermarkets_l3094_309438


namespace NUMINAMATH_CALUDE_locus_of_P_l3094_309401

/-- The locus of point P given the conditions in the problem -/
theorem locus_of_P (F Q T P : ℝ × ℝ) (l : Set (ℝ × ℝ)) : 
  F = (2, 0) →
  (∀ (x y : ℝ), (x, y) ∈ l ↔ ∃ (k : ℝ), y = k * (x - 2)) →
  Q.1 = 0 →
  Q ∈ l →
  (T.2 = 0 ∧ (Q.1 - T.1) * (F.1 - Q.1) = (F.2 - Q.2) * (Q.2 - T.2)) →
  (T.1 - Q.1)^2 + (T.2 - Q.2)^2 = (P.1 - Q.1)^2 + (P.2 - Q.2)^2 →
  P.2^2 = 8 * P.1 :=
by sorry

end NUMINAMATH_CALUDE_locus_of_P_l3094_309401


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3094_309436

theorem sufficient_but_not_necessary (p q : Prop) :
  (p ∧ q → p ∨ q) ∧ ¬(p ∨ q → p ∧ q) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3094_309436


namespace NUMINAMATH_CALUDE_cubic_equation_root_l3094_309459

theorem cubic_equation_root (c d : ℚ) : 
  (3 + Real.sqrt 5)^3 + c * (3 + Real.sqrt 5)^2 + d * (3 + Real.sqrt 5) + 15 = 0 → 
  d = -37/2 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l3094_309459


namespace NUMINAMATH_CALUDE_count_special_numbers_l3094_309427

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def satisfies_condition (n : ℕ) : Prop :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  n = a + b^2 + c^3

theorem count_special_numbers :
  ∃ (S : Finset ℕ), (∀ n ∈ S, is_three_digit n ∧ satisfies_condition n) ∧
                    (∀ n, is_three_digit n → satisfies_condition n → n ∈ S) ∧
                    Finset.card S = 4 :=
sorry

end NUMINAMATH_CALUDE_count_special_numbers_l3094_309427


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3094_309445

/-- 
Given an arithmetic sequence {aₙ} with common difference d,
prove that d = 1 when S₈ = 8a₅ - 4, where Sₙ is the sum of the first n terms.
-/
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℚ)  -- The arithmetic sequence
  (d : ℚ)      -- The common difference
  (S : ℕ → ℚ)  -- The sum function
  (h1 : ∀ n, a (n + 1) = a n + d)  -- Definition of arithmetic sequence
  (h2 : ∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2)  -- Sum formula for arithmetic sequence
  (h3 : S 8 = 8 * a 5 - 4)  -- Given condition
  : d = 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3094_309445


namespace NUMINAMATH_CALUDE_quadratic_root_one_iff_sum_zero_l3094_309493

theorem quadratic_root_one_iff_sum_zero (a b c : ℝ) :
  (∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ x = 1) ↔ a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_one_iff_sum_zero_l3094_309493


namespace NUMINAMATH_CALUDE_intersection_equals_open_unit_interval_l3094_309488

-- Define the sets M and N
def M : Set ℝ := {x | Real.log (1 - x) < 0}
def N : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

-- Define the open interval (0, 1)
def open_unit_interval : Set ℝ := {x | 0 < x ∧ x < 1}

-- State the theorem
theorem intersection_equals_open_unit_interval : M ∩ N = open_unit_interval := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_open_unit_interval_l3094_309488


namespace NUMINAMATH_CALUDE_quadratic_monotone_increasing_l3094_309497

/-- A quadratic function f(x) = x^2 + 2ax - 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x - 1

/-- The derivative of f(x) with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 2*x + 2*a

theorem quadratic_monotone_increasing (a : ℝ) (h : a > 1) :
  ∀ x > 1, Monotone (fun x => f a x) := by sorry

end NUMINAMATH_CALUDE_quadratic_monotone_increasing_l3094_309497


namespace NUMINAMATH_CALUDE_percentage_puppies_greater_profit_l3094_309463

/-- Calculates the percentage of puppies that can be sold for a greater profit -/
theorem percentage_puppies_greater_profit (total_puppies : ℕ) (puppies_more_than_4_spots : ℕ) 
  (h1 : total_puppies = 10)
  (h2 : puppies_more_than_4_spots = 6) :
  (puppies_more_than_4_spots : ℚ) / total_puppies * 100 = 60 := by
  sorry


end NUMINAMATH_CALUDE_percentage_puppies_greater_profit_l3094_309463


namespace NUMINAMATH_CALUDE_equation_solution_l3094_309444

theorem equation_solution :
  ∀ x : ℚ, x ≠ 2 →
  (7 * x / (x - 2) - 5 / (x - 2) = 3 / (x - 2)) ↔ x = 8 / 7 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3094_309444


namespace NUMINAMATH_CALUDE_inequality_proof_l3094_309467

theorem inequality_proof (a b : ℝ) (n : ℕ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : 1/a + 1/b = 1) : 
  (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n+1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3094_309467


namespace NUMINAMATH_CALUDE_f_minimum_g_solution_set_l3094_309411

-- Define the function f
def f (x : ℝ) : ℝ := |x - 5| - |x - 2|

-- Theorem for the minimum value of f
theorem f_minimum : ∀ x : ℝ, f x ≥ -3 :=
sorry

-- Define the inequality function g
def g (x : ℝ) : ℝ := x^2 - 8*x + 15 + f x

-- Theorem for the solution set of g(x) ≤ 0
theorem g_solution_set : 
  ∀ x : ℝ, g x ≤ 0 ↔ 5 - Real.sqrt 3 ≤ x ∧ x ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_f_minimum_g_solution_set_l3094_309411


namespace NUMINAMATH_CALUDE_unique_real_sqrt_negative_square_l3094_309468

theorem unique_real_sqrt_negative_square :
  ∃! x : ℝ, ∃ y : ℝ, y ^ 2 = -(2 * x - 3) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_real_sqrt_negative_square_l3094_309468


namespace NUMINAMATH_CALUDE_cherries_eaten_l3094_309404

theorem cherries_eaten (initial : ℝ) (remaining : ℝ) (eaten : ℝ)
  (h1 : initial = 67.5)
  (h2 : remaining = 42.25)
  (h3 : eaten = initial - remaining) :
  eaten = 25.25 := by sorry

end NUMINAMATH_CALUDE_cherries_eaten_l3094_309404


namespace NUMINAMATH_CALUDE_rectangleABCD_area_is_196_l3094_309473

/-- Represents a rectangle with width and length -/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- The area of rectangle ABCD formed by three identical rectangles -/
def rectangleABCD_area (small_rect : Rectangle) : ℝ :=
  (2 * small_rect.width) * small_rect.length

theorem rectangleABCD_area_is_196 (small_rect : Rectangle) 
  (h1 : small_rect.width = 7)
  (h2 : small_rect.length = 2 * small_rect.width) :
  rectangleABCD_area small_rect = 196 := by
  sorry

#eval rectangleABCD_area { width := 7, length := 14 }

end NUMINAMATH_CALUDE_rectangleABCD_area_is_196_l3094_309473


namespace NUMINAMATH_CALUDE_function_minimum_condition_l3094_309499

/-- A function f(x) = x^2 - 2ax + a has a minimum value in the interval (-∞, 1) if and only if a < 1 -/
theorem function_minimum_condition (a : ℝ) : 
  (∃ (x₀ : ℝ), x₀ < 1 ∧ ∀ (x : ℝ), x < 1 → (x^2 - 2*a*x + a) ≥ (x₀^2 - 2*a*x₀ + a)) ↔ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_function_minimum_condition_l3094_309499


namespace NUMINAMATH_CALUDE_number_of_teams_l3094_309466

/-- The number of teams in the league -/
def n : ℕ := sorry

/-- The total number of games played in the season -/
def total_games : ℕ := 4900

/-- Each team faces every other team this many times -/
def games_per_pair : ℕ := 4

theorem number_of_teams : 
  (n * games_per_pair * (n - 1)) / 2 = total_games ∧ n = 50 := by sorry

end NUMINAMATH_CALUDE_number_of_teams_l3094_309466


namespace NUMINAMATH_CALUDE_tangent_line_ratio_l3094_309429

theorem tangent_line_ratio (x₁ x₂ : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0) : 
  (∃ (m b : ℝ), 
    (∀ x, m * x + b = x^2 ↔ x = x₁) ∧ 
    (∀ x, m * x + b = x^3 ↔ x = x₂)) → 
  x₁ / x₂ = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_ratio_l3094_309429


namespace NUMINAMATH_CALUDE_ship_speed_in_still_water_l3094_309457

/-- Given a ship with downstream speed of 32 km/h and upstream speed of 28 km/h,
    prove that its speed in still water is 30 km/h. -/
theorem ship_speed_in_still_water 
  (downstream_speed : ℝ) 
  (upstream_speed : ℝ) 
  (h1 : downstream_speed = 32)
  (h2 : upstream_speed = 28)
  (h3 : ∃ (ship_speed stream_speed : ℝ), 
    ship_speed > stream_speed ∧
    ship_speed + stream_speed = downstream_speed ∧
    ship_speed - stream_speed = upstream_speed) :
  ∃ (ship_speed : ℝ), ship_speed = 30 := by
sorry

end NUMINAMATH_CALUDE_ship_speed_in_still_water_l3094_309457


namespace NUMINAMATH_CALUDE_lemonade_stand_total_profit_l3094_309481

/-- Calculates the profit for a single day of lemonade stand operation -/
def daily_profit (lemon_cost sugar_cost cup_cost extra_cost price_per_cup cups_sold : ℕ) : ℕ :=
  price_per_cup * cups_sold - (lemon_cost + sugar_cost + cup_cost + extra_cost)

/-- Represents the lemonade stand operation for three days -/
def lemonade_stand_profit : Prop :=
  let day1_profit := daily_profit 10 5 3 0 4 21
  let day2_profit := daily_profit 12 6 4 0 5 18
  let day3_profit := daily_profit 8 4 3 2 4 25
  day1_profit + day2_profit + day3_profit = 217

theorem lemonade_stand_total_profit : lemonade_stand_profit := by
  sorry

end NUMINAMATH_CALUDE_lemonade_stand_total_profit_l3094_309481


namespace NUMINAMATH_CALUDE_billion_scientific_notation_l3094_309482

theorem billion_scientific_notation : 
  (1.1 * 10^9 : ℝ) = 1100000000 := by sorry

end NUMINAMATH_CALUDE_billion_scientific_notation_l3094_309482


namespace NUMINAMATH_CALUDE_expansion_properties_l3094_309455

def n : ℕ := 5

def general_term (r : ℕ) : ℚ × ℤ → ℚ := λ (c, p) ↦ c * (2^(10 - r) * (-1)^r)

theorem expansion_properties :
  let tenth_term := general_term 9 (1, -8)
  let constant_term := general_term 5 (1, 0)
  let max_coeff_term := general_term 3 (1, 4)
  (tenth_term = -20) ∧
  (constant_term = -8064) ∧
  (max_coeff_term = -15360) ∧
  (∀ r : ℕ, r ≤ 10 → |general_term r (1, 10 - 2*r)| ≤ |max_coeff_term|) :=
by sorry

end NUMINAMATH_CALUDE_expansion_properties_l3094_309455


namespace NUMINAMATH_CALUDE_quadrilateral_area_is_3_root_6_over_2_l3094_309417

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a quadrilateral in 3D space -/
structure Quadrilateral where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Given a cube of side length 2, calculates the area of quadrilateral ABCD where
    A and C are diagonally opposite vertices, and B and D are quarter points on
    opposite edges not containing A or C -/
def quadrilateralArea (cube : Point3D → Bool) (A B C D : Point3D) : ℝ :=
  sorry

/-- Theorem stating that the area of the quadrilateral ABCD in the given conditions is 3√6/2 -/
theorem quadrilateral_area_is_3_root_6_over_2 
  (cube : Point3D → Bool) 
  (A B C D : Point3D) 
  (h_cube_side : ∀ (p q : Point3D), cube p ∧ cube q → 
    (p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2 ≤ 4)
  (h_A_C_diagonal : cube A ∧ cube C ∧ 
    (A.x - C.x)^2 + (A.y - C.y)^2 + (A.z - C.z)^2 = 12)
  (h_B_quarter : ∃ (p q : Point3D), cube p ∧ cube q ∧
    (B.x - p.x)^2 + (B.y - p.y)^2 + (B.z - p.z)^2 = 1/4 ∧
    (B.x - q.x)^2 + (B.y - q.y)^2 + (B.z - q.z)^2 = 9/4)
  (h_D_quarter : ∃ (r s : Point3D), cube r ∧ cube s ∧
    (D.x - r.x)^2 + (D.y - r.y)^2 + (D.z - r.z)^2 = 1/4 ∧
    (D.x - s.x)^2 + (D.y - s.y)^2 + (D.z - s.z)^2 = 9/4)
  : quadrilateralArea cube A B C D = 3 * Real.sqrt 6 / 2 :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_area_is_3_root_6_over_2_l3094_309417


namespace NUMINAMATH_CALUDE_det_A_nonzero_l3094_309489

def matrix_A (n : ℕ) (a : ℤ) : Matrix (Fin n) (Fin n) ℤ :=
  λ i j => a^(i.val * j.val + 1)

theorem det_A_nonzero {n : ℕ} {a : ℤ} (h : a > 1) :
  Matrix.det (matrix_A n a) ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_det_A_nonzero_l3094_309489


namespace NUMINAMATH_CALUDE_odd_cube_difference_divisible_by_24_l3094_309461

theorem odd_cube_difference_divisible_by_24 (n : ℤ) : 
  24 ∣ ((2 * n + 1)^3 - (2 * n + 1)) := by sorry

end NUMINAMATH_CALUDE_odd_cube_difference_divisible_by_24_l3094_309461


namespace NUMINAMATH_CALUDE_johns_break_time_l3094_309458

/-- Given the dancing times of John and James, prove that John's break was 1 hour long. -/
theorem johns_break_time (john_first_dance : ℝ) (john_second_dance : ℝ) 
  (james_dance_multiplier : ℝ) (total_dance_time : ℝ) :
  john_first_dance = 3 →
  john_second_dance = 5 →
  james_dance_multiplier = 1/3 →
  total_dance_time = 20 →
  ∃ (break_time : ℝ),
    total_dance_time = john_first_dance + john_second_dance + 
      ((john_first_dance + john_second_dance + break_time) + 
       james_dance_multiplier * (john_first_dance + john_second_dance + break_time)) ∧
    break_time = 1 := by
  sorry


end NUMINAMATH_CALUDE_johns_break_time_l3094_309458


namespace NUMINAMATH_CALUDE_triangle_properties_l3094_309430

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The cosine law for triangle ABC -/
def cosineLaw (t : Triangle) : Prop :=
  t.a^2 = t.b^2 + t.c^2 - 2 * t.b * t.c * Real.cos t.A

/-- The given condition for the triangle -/
def givenCondition (t : Triangle) : Prop :=
  3 * t.a * Real.cos t.A = t.b * Real.cos t.C + t.c * Real.cos t.B

theorem triangle_properties (t : Triangle) 
  (h1 : givenCondition t) 
  (h2 : 0 < t.A ∧ t.A < π) : 
  Real.cos t.A = 1/3 ∧ 
  (t.a = 3 → 
    ∃ (S : ℝ), S = (9 * Real.sqrt 2) / 4 ∧ 
    ∀ (S' : ℝ), S' = 1/2 * t.b * t.c * Real.sin t.A → S' ≤ S) := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l3094_309430


namespace NUMINAMATH_CALUDE_pie_eating_contest_l3094_309407

/-- Pie-eating contest theorem -/
theorem pie_eating_contest 
  (adam bill sierra taylor : ℕ) -- Number of pies eaten by each participant
  (h1 : adam = bill + 3) -- Adam eats three more pies than Bill
  (h2 : sierra = 2 * bill) -- Sierra eats twice as many pies as Bill
  (h3 : taylor = (adam + bill + sierra) / 3) -- Taylor eats the average of Adam, Bill, and Sierra
  (h4 : sierra = 12) -- Sierra ate 12 pies
  : adam + bill + sierra + taylor = 36 ∧ adam + bill + sierra + taylor ≤ 50 := by
  sorry

end NUMINAMATH_CALUDE_pie_eating_contest_l3094_309407


namespace NUMINAMATH_CALUDE_first_day_revenue_l3094_309437

/-- Calculates the revenue from ticket sales given the number of senior and student tickets sold and their prices -/
def revenue (senior_count : ℕ) (student_count : ℕ) (senior_price : ℚ) (student_price : ℚ) : ℚ :=
  senior_count * senior_price + student_count * student_price

/-- Represents the ticket sales scenario -/
structure TicketSales where
  day1_senior : ℕ
  day1_student : ℕ
  day2_senior : ℕ
  day2_student : ℕ
  day2_revenue : ℚ
  student_price : ℚ

/-- Theorem stating that the first day's revenue is $79 -/
theorem first_day_revenue (ts : TicketSales) 
  (h1 : ts.day1_senior = 4)
  (h2 : ts.day1_student = 3)
  (h3 : ts.day2_senior = 12)
  (h4 : ts.day2_student = 10)
  (h5 : ts.day2_revenue = 246)
  (h6 : ts.student_price = 9)
  : ∃ (senior_price : ℚ), revenue ts.day1_senior ts.day1_student senior_price ts.student_price = 79 := by
  sorry

end NUMINAMATH_CALUDE_first_day_revenue_l3094_309437


namespace NUMINAMATH_CALUDE_intersection_in_fourth_quadrant_l3094_309454

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line by two points
def Line (p1 p2 : Point2D) :=
  {p : Point2D | (p.y - p1.y) * (p2.x - p1.x) = (p.x - p1.x) * (p2.y - p1.y)}

-- Define a vertical line by a point
def VerticalLine (p : Point2D) :=
  {q : Point2D | q.x = p.x}

-- Define the intersection of two lines
def Intersection (l1 l2 : Set Point2D) :=
  {p : Point2D | p ∈ l1 ∧ p ∈ l2}

theorem intersection_in_fourth_quadrant :
  let l := Line ⟨-3, 0⟩ ⟨0, -5⟩
  let l' := VerticalLine ⟨2, 4⟩
  let i := Intersection l l'
  ∀ p ∈ i, p.x > 0 ∧ p.y < 0 := by sorry

end NUMINAMATH_CALUDE_intersection_in_fourth_quadrant_l3094_309454


namespace NUMINAMATH_CALUDE_smallest_group_size_exists_group_size_l3094_309485

theorem smallest_group_size (n : ℕ) : 
  (n % 6 = 1) ∧ (n % 8 = 3) ∧ (n % 9 = 5) → n ≥ 187 :=
by sorry

theorem exists_group_size : 
  ∃ n : ℕ, (n % 6 = 1) ∧ (n % 8 = 3) ∧ (n % 9 = 5) ∧ n = 187 :=
by sorry

end NUMINAMATH_CALUDE_smallest_group_size_exists_group_size_l3094_309485


namespace NUMINAMATH_CALUDE_orthic_triangle_similarity_l3094_309409

/-- A triangle with angles A, B, and C in degrees -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_180 : A + B + C = 180
  positive : 0 < A ∧ 0 < B ∧ 0 < C

/-- The orthic triangle of a given triangle -/
def orthicTriangle (t : Triangle) : Triangle where
  A := 180 - 2 * t.A
  B := 180 - 2 * t.B
  C := 180 - 2 * t.C
  sum_180 := sorry
  positive := sorry

/-- Two triangles are similar if their corresponding angles are equal -/
def similar (t1 t2 : Triangle) : Prop :=
  t1.A = t2.A ∧ t1.B = t2.B ∧ t1.C = t2.C

theorem orthic_triangle_similarity (t : Triangle) 
  (h_not_right : t.A ≠ 90 ∧ t.B ≠ 90 ∧ t.C ≠ 90) :
  similar t (orthicTriangle t) ↔ t.A = 60 ∧ t.B = 60 ∧ t.C = 60 := by
  sorry

end NUMINAMATH_CALUDE_orthic_triangle_similarity_l3094_309409


namespace NUMINAMATH_CALUDE_factorization_of_x_squared_minus_one_l3094_309490

theorem factorization_of_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_x_squared_minus_one_l3094_309490


namespace NUMINAMATH_CALUDE_m_value_min_value_l3094_309484

-- Define the solution set A
def A (m : ℤ) : Set ℝ := {x : ℝ | |x + 1| + |x - m| < 5}

-- Theorem 1
theorem m_value (m : ℤ) (h : 3 ∈ A m) : m = 3 := by sorry

-- Theorem 2
theorem min_value (a b c : ℝ) (h : a + 2*b + 2*c = 3) : 
  ∃ (min : ℝ), min = 1 ∧ a^2 + b^2 + c^2 ≥ min := by sorry

end NUMINAMATH_CALUDE_m_value_min_value_l3094_309484


namespace NUMINAMATH_CALUDE_twelve_foldable_configurations_l3094_309406

/-- Represents a position on the periphery of the cross-shaped arrangement -/
inductive PeripheryPosition
| Top
| Right
| Bottom
| Left

/-- Represents the cross-shaped arrangement of 5 squares with an additional square -/
structure CrossArrangement :=
  (additional_square_position : PeripheryPosition)
  (additional_square_offset : Fin 3)

/-- Predicate to determine if a given arrangement can be folded into a cube with one face open -/
def can_fold_to_cube (arrangement : CrossArrangement) : Prop :=
  sorry

/-- The main theorem stating that exactly 12 configurations can be folded into a cube -/
theorem twelve_foldable_configurations :
  (∃ (configurations : Finset CrossArrangement),
    configurations.card = 12 ∧
    (∀ c ∈ configurations, can_fold_to_cube c) ∧
    (∀ c : CrossArrangement, can_fold_to_cube c → c ∈ configurations)) :=
sorry

end NUMINAMATH_CALUDE_twelve_foldable_configurations_l3094_309406


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l3094_309456

/-- A sequence of real numbers. -/
def Sequence := ℕ → ℝ

/-- Predicate for a geometric sequence. -/
def IsGeometric (a : Sequence) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The condition given in the problem. -/
def Condition (a : Sequence) : Prop :=
  ∀ n : ℕ, n ≥ 2 → a (n + 1) * a (n - 1) = a n ^ 2

theorem condition_necessary_not_sufficient :
  (∀ a : Sequence, IsGeometric a → Condition a) ∧
  (∃ a : Sequence, Condition a ∧ ¬IsGeometric a) :=
sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l3094_309456


namespace NUMINAMATH_CALUDE_least_positive_integer_divisible_by_four_primes_l3094_309491

theorem least_positive_integer_divisible_by_four_primes : 
  ∃ (n : ℕ), (n > 0) ∧ 
  (∃ (p₁ p₂ p₃ p₄ : ℕ), Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
    p₁ * p₂ * p₃ * p₄ = n) ∧
  (∀ m : ℕ, m < n → 
    ¬(∃ (q₁ q₂ q₃ q₄ : ℕ), Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄ ∧ 
      q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₃ ≠ q₄ ∧
      q₁ * q₂ * q₃ * q₄ = m)) ∧
  n = 210 := by
sorry

end NUMINAMATH_CALUDE_least_positive_integer_divisible_by_four_primes_l3094_309491


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3094_309448

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * x + 1

-- Define the theorem
theorem quadratic_function_properties (a : ℝ) (h1 : 1/3 ≤ a) (h2 : a ≤ 1) :
  -- 1. Minimum value of f(x) on [1, 3]
  (∃ (x : ℝ), x ∈ Set.Icc 1 3 ∧ ∀ (y : ℝ), y ∈ Set.Icc 1 3 → f a x ≤ f a y) ∧
  (∃ (x : ℝ), x ∈ Set.Icc 1 3 ∧ f a x = 1 - 1/a) ∧
  -- 2. Minimum value of M(a) - N(a)
  (∃ (M N : ℝ → ℝ),
    (∀ (x : ℝ), x ∈ Set.Icc 1 3 → f a x ≤ M a) ∧
    (∃ (y : ℝ), y ∈ Set.Icc 1 3 ∧ f a y = M a) ∧
    (∀ (x : ℝ), x ∈ Set.Icc 1 3 → N a ≤ f a x) ∧
    (∃ (z : ℝ), z ∈ Set.Icc 1 3 ∧ f a z = N a) ∧
    (∀ (b : ℝ), 1/3 ≤ b ∧ b ≤ 1 → 1/2 ≤ M b - N b) ∧
    (∃ (c : ℝ), 1/3 ≤ c ∧ c ≤ 1 ∧ M c - N c = 1/2)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3094_309448


namespace NUMINAMATH_CALUDE_parallel_vectors_l3094_309494

def a : ℝ × ℝ := (1, -1)
def b : ℝ → ℝ × ℝ := λ x => (x, 1)

theorem parallel_vectors (x : ℝ) : 
  (∃ k : ℝ, b x = k • a) → x = -1 := by sorry

end NUMINAMATH_CALUDE_parallel_vectors_l3094_309494


namespace NUMINAMATH_CALUDE_crepe_myrtle_count_l3094_309450

theorem crepe_myrtle_count (total : ℕ) (pink : ℕ) (red : ℕ) (white : ℕ) : 
  total = 42 →
  pink = total / 3 →
  red = 2 →
  white = total - (pink + red) →
  white = 26 := by
sorry

end NUMINAMATH_CALUDE_crepe_myrtle_count_l3094_309450


namespace NUMINAMATH_CALUDE_smallest_group_size_l3094_309423

theorem smallest_group_size (n : ℕ) : n = 154 ↔ 
  n > 0 ∧ 
  n % 6 = 1 ∧ 
  n % 8 = 2 ∧ 
  n % 9 = 4 ∧ 
  ∀ m : ℕ, m > 0 → m % 6 = 1 → m % 8 = 2 → m % 9 = 4 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_group_size_l3094_309423


namespace NUMINAMATH_CALUDE_amy_biking_distance_l3094_309462

theorem amy_biking_distance (x : ℝ) : 
  x + (2 * x - 3) = 33 → x = 12 := by sorry

end NUMINAMATH_CALUDE_amy_biking_distance_l3094_309462


namespace NUMINAMATH_CALUDE_inequality_proof_l3094_309496

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  (x^3 / ((1 + y) * (1 + z))) + (y^3 / ((1 + z) * (1 + x))) + (z^3 / ((1 + x) * (1 + y))) ≥ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3094_309496


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l3094_309424

theorem absolute_value_equation_solution :
  ∃! x : ℚ, |x - 5| = 3 * x + 6 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l3094_309424


namespace NUMINAMATH_CALUDE_train_passing_tree_l3094_309400

/-- Proves that a train of given length and speed takes a specific time to pass a tree -/
theorem train_passing_tree (train_length : ℝ) (train_speed_kmh : ℝ) (time : ℝ) :
  train_length = 280 →
  train_speed_kmh = 72 →
  time = train_length / (train_speed_kmh * (5/18)) →
  time = 14 := by
  sorry

#check train_passing_tree

end NUMINAMATH_CALUDE_train_passing_tree_l3094_309400


namespace NUMINAMATH_CALUDE_simplify_expression_l3094_309480

theorem simplify_expression : (6^7 + 4^6) * (1^5 - (-1)^5)^10 = 290938368 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3094_309480


namespace NUMINAMATH_CALUDE_system_solution_l3094_309498

theorem system_solution (x y : ℝ) : 
  (x + 2*y = 2 ∧ x - 2*y = 6) ↔ (x = 4 ∧ y = -1) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3094_309498


namespace NUMINAMATH_CALUDE_tank_capacity_is_72_liters_l3094_309405

/-- The total capacity of a water tank in liters. -/
def tank_capacity : ℝ := 72

/-- The amount of water in the tank when it's 40% full, in liters. -/
def water_at_40_percent : ℝ := 0.4 * tank_capacity

/-- The amount of water in the tank when it's 10% empty, in liters. -/
def water_at_10_percent_empty : ℝ := 0.9 * tank_capacity

/-- Theorem stating that the tank capacity is 72 liters, given the condition. -/
theorem tank_capacity_is_72_liters :
  water_at_10_percent_empty - water_at_40_percent = 36 →
  tank_capacity = 72 :=
by sorry

end NUMINAMATH_CALUDE_tank_capacity_is_72_liters_l3094_309405


namespace NUMINAMATH_CALUDE_daily_production_l3094_309441

/-- The number of bottles per case -/
def bottles_per_case : ℕ := 5

/-- The number of cases required for daily production -/
def cases_per_day : ℕ := 12000

/-- The total number of bottles produced per day -/
def total_bottles : ℕ := bottles_per_case * cases_per_day

theorem daily_production :
  total_bottles = 60000 :=
by sorry

end NUMINAMATH_CALUDE_daily_production_l3094_309441


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l3094_309425

theorem quadratic_roots_property (a b : ℝ) : 
  (3 * a^2 + 9 * a - 21 = 0) →
  (3 * b^2 + 9 * b - 21 = 0) →
  (3 * a - 4) * (6 * b - 8) = 14 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l3094_309425


namespace NUMINAMATH_CALUDE_unreachable_corner_l3094_309451

/-- A point in 3D space with integer coordinates -/
structure Point3D where
  x : Int
  y : Int
  z : Int

/-- The set of 7 vertices of a cube, excluding (1,1,1) -/
def cube_vertices : Set Point3D :=
  { ⟨0,0,0⟩, ⟨0,0,1⟩, ⟨0,1,0⟩, ⟨1,0,0⟩, ⟨0,1,1⟩, ⟨1,0,1⟩, ⟨1,1,0⟩ }

/-- Symmetry transformation with respect to another point -/
def symmetry_transform (p : Point3D) (center : Point3D) : Point3D :=
  ⟨2 * center.x - p.x, 2 * center.y - p.y, 2 * center.z - p.z⟩

/-- The set of points reachable through symmetry transformations -/
def reachable_points : Set Point3D :=
  sorry -- Definition of reachable points through symmetry transformations

theorem unreachable_corner : ⟨1,1,1⟩ ∉ reachable_points := by
  sorry

#check unreachable_corner

end NUMINAMATH_CALUDE_unreachable_corner_l3094_309451


namespace NUMINAMATH_CALUDE_binomial_expansion_properties_l3094_309443

theorem binomial_expansion_properties :
  let n : ℕ := 5
  let a : ℝ := 1
  let b : ℝ := 2
  -- The coefficient of the third term
  (Finset.sum (Finset.range 1) (fun k => (n.choose k) * a^(n-k) * b^k)) = 40 ∧
  -- The sum of all binomial coefficients
  (Finset.sum (Finset.range (n+1)) (fun k => n.choose k)) = 32 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_properties_l3094_309443


namespace NUMINAMATH_CALUDE_inequality_not_holding_l3094_309408

theorem inequality_not_holding (x y : ℝ) (h : x > y) : ¬(-2*x > -2*y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_not_holding_l3094_309408


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l3094_309453

theorem quadratic_no_real_roots :
  ∀ (x : ℝ), x^2 - 3*x + 3 ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l3094_309453


namespace NUMINAMATH_CALUDE_solution_set_eq_singleton_l3094_309439

/-- The set of solutions to the system of equations x + y = 2 and x - y = 0 -/
def solution_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 = 2 ∧ p.1 - p.2 = 0}

/-- Theorem stating that the solution set is equal to {(1, 1)} -/
theorem solution_set_eq_singleton : solution_set = {(1, 1)} := by
  sorry

#check solution_set_eq_singleton

end NUMINAMATH_CALUDE_solution_set_eq_singleton_l3094_309439


namespace NUMINAMATH_CALUDE_exists_fixed_point_l3094_309412

def S : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 999}

def is_fixed_point (f : S → S) (a : S) : Prop := f a = a

def satisfies_condition (f : S → S) : Prop :=
  ∀ n : S, (f^[n + f n + 1] n = n) ∧ (f^[n * f n] n = n)

theorem exists_fixed_point (f : S → S) (h : satisfies_condition f) :
  ∃ a : S, is_fixed_point f a := by
  sorry

end NUMINAMATH_CALUDE_exists_fixed_point_l3094_309412


namespace NUMINAMATH_CALUDE_number_sequence_properties_l3094_309418

/-- Represents the sequence formed by concatenating numbers from 1 to 999 -/
def NumberSequence : Type := List Nat

/-- Constructs the NumberSequence -/
def createSequence : NumberSequence := sorry

/-- Counts the total number of digits in the sequence -/
def countDigits (seq : NumberSequence) : Nat := sorry

/-- Counts the occurrences of a specific digit in the sequence -/
def countDigitOccurrences (seq : NumberSequence) (digit : Nat) : Nat := sorry

/-- Finds the digit at a specific position in the sequence -/
def digitAtPosition (seq : NumberSequence) (position : Nat) : Nat := sorry

theorem number_sequence_properties (seq : NumberSequence) :
  seq = createSequence →
  (countDigits seq = 2889) ∧
  (countDigitOccurrences seq 1 = 300) ∧
  (digitAtPosition seq 2016 = 8) := by
  sorry

end NUMINAMATH_CALUDE_number_sequence_properties_l3094_309418


namespace NUMINAMATH_CALUDE_gcd_45_105_l3094_309449

theorem gcd_45_105 : Nat.gcd 45 105 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_45_105_l3094_309449


namespace NUMINAMATH_CALUDE_quadratic_integer_solution_count_l3094_309464

theorem quadratic_integer_solution_count : ∃ (S : Finset ℚ),
  (∀ k ∈ S, |k| < 100 ∧ ∃ x : ℤ, 3 * x^2 + k * x + 8 = 0) ∧
  (∀ k : ℚ, |k| < 100 → (∃ x : ℤ, 3 * x^2 + k * x + 8 = 0) → k ∈ S) ∧
  Finset.card S = 40 :=
sorry

end NUMINAMATH_CALUDE_quadratic_integer_solution_count_l3094_309464
