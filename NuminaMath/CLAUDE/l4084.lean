import Mathlib

namespace line_slope_through_point_l4084_408485

theorem line_slope_through_point (x y k : ℝ) : 
  x = 2 → y = Real.sqrt 3 → y = k * x → k = Real.sqrt 3 / 2 := by
  sorry

end line_slope_through_point_l4084_408485


namespace minimize_sum_of_squares_l4084_408477

/-- The quadratic equation in x has only integer roots -/
def has_integer_roots (k : ℚ) : Prop :=
  ∃ x₁ x₂ : ℤ, k * x₁^2 + (3 - 3*k) * x₁ + (2*k - 6) = 0 ∧
              k * x₂^2 + (3 - 3*k) * x₂ + (2*k - 6) = 0

/-- The quadratic equation in y has two positive integer roots -/
def has_positive_integer_roots (k t : ℚ) : Prop :=
  ∃ y₁ y₂ : ℕ, (k + 3) * y₁^2 - 15 * y₁ + t = 0 ∧
              (k + 3) * y₂^2 - 15 * y₂ + t = 0 ∧
              y₁ ≠ y₂

theorem minimize_sum_of_squares (k t : ℚ) :
  has_integer_roots k →
  has_positive_integer_roots k t →
  (k = 3/4 ∧ t = 15) →
  ∃ y₁ y₂ : ℕ, (k + 3) * y₁^2 - 15 * y₁ + t = 0 ∧
              (k + 3) * y₂^2 - 15 * y₂ + t = 0 ∧
              y₁^2 + y₂^2 = 8 ∧
              ∀ y₁' y₂' : ℕ, (k + 3) * y₁'^2 - 15 * y₁' + t = 0 →
                             (k + 3) * y₂'^2 - 15 * y₂' + t = 0 →
                             y₁'^2 + y₂'^2 ≥ 8 :=
by sorry

end minimize_sum_of_squares_l4084_408477


namespace grandfather_cake_blue_candles_l4084_408496

/-- The number of blue candles on Caleb's grandfather's birthday cake -/
def blue_candles (total_candles yellow_candles red_candles : ℕ) : ℕ :=
  total_candles - (yellow_candles + red_candles)

/-- Theorem stating the number of blue candles on the cake -/
theorem grandfather_cake_blue_candles :
  blue_candles 79 27 14 = 38 := by
  sorry

end grandfather_cake_blue_candles_l4084_408496


namespace f_decreasing_on_interval_l4084_408468

def f (a b : ℝ) (x : ℝ) := a * x^2 + b * x - 2

theorem f_decreasing_on_interval (a b : ℝ) :
  (∀ x ∈ Set.Icc (1 + a) 2, f a b x = f a b (-x)) →
  (∀ x ∈ Set.Icc 1 2, ∀ y ∈ Set.Icc 1 2, x < y → f a b x > f a b y) :=
sorry

end f_decreasing_on_interval_l4084_408468


namespace sequence_inequality_l4084_408435

theorem sequence_inequality (a : ℕ → ℕ) 
  (h0 : ∀ n, a n > 0)
  (h1 : a 1 > a 0)
  (h2 : ∀ n ∈ Finset.range 99, a (n + 2) = 3 * a (n + 1) - 2 * a n) :
  a 100 > 2^99 := by
  sorry

end sequence_inequality_l4084_408435


namespace parabola_equation_l4084_408408

/-- Represents a parabola with equation y^2 = 2px -/
structure Parabola where
  p : ℝ
  h_pos : p > 0

/-- A point on a parabola -/
structure ParabolaPoint (para : Parabola) where
  x : ℝ
  y : ℝ
  h_on_parabola : y^2 = 2 * para.p * x

theorem parabola_equation (para : Parabola) 
  (point : ParabolaPoint para)
  (h_ordinate : point.y = -4 * Real.sqrt 2)
  (h_distance : point.x + para.p / 2 = 6) :
  para.p = 4 ∨ para.p = 8 :=
sorry

end parabola_equation_l4084_408408


namespace average_first_16_even_numbers_l4084_408437

theorem average_first_16_even_numbers :
  let first_even : ℕ := 2
  let last_even : ℕ := 32
  let count : ℕ := 16
  (first_even + last_even) / 2 = 17 :=
by sorry

end average_first_16_even_numbers_l4084_408437


namespace gcf_420_144_l4084_408409

theorem gcf_420_144 : Nat.gcd 420 144 = 12 := by
  sorry

end gcf_420_144_l4084_408409


namespace susan_bob_cat_difference_l4084_408429

/-- Proves that Susan has 8 more cats than Bob after all exchanges -/
theorem susan_bob_cat_difference :
  let susan_initial : ℕ := 21
  let bob_initial : ℕ := 3
  let susan_received : ℕ := 5
  let bob_received : ℕ := 7
  let susan_gave : ℕ := 4
  let susan_final := susan_initial + susan_received - susan_gave
  let bob_final := bob_initial + bob_received + susan_gave
  susan_final - bob_final = 8 := by
  sorry

end susan_bob_cat_difference_l4084_408429


namespace inverse_proportion_k_value_l4084_408444

theorem inverse_proportion_k_value (y : ℝ → ℝ) (k : ℝ) :
  (∀ x, x ≠ 0 → y x = k / x) →  -- Inverse proportion function
  y 3 = 2 →                     -- Passes through (3, 2)
  k = 6 :=                      -- Prove k = 6
by sorry

end inverse_proportion_k_value_l4084_408444


namespace quadratic_function_range_quadratic_function_range_restricted_l4084_408449

theorem quadratic_function_range (a : ℝ) :
  (∀ x : ℝ, x^2 - 2*a*x + 2 ≥ a) ↔ -2 ≤ a ∧ a ≤ 1 :=
sorry

theorem quadratic_function_range_restricted (a : ℝ) :
  (∀ x : ℝ, x ≥ -1 → x^2 - 2*a*x + 2 ≥ a) ↔ -3 ≤ a ∧ a ≤ 1 :=
sorry

end quadratic_function_range_quadratic_function_range_restricted_l4084_408449


namespace mans_upstream_speed_l4084_408473

/-- Given a man's downstream speed and the stream speed, calculate the man's upstream speed. -/
theorem mans_upstream_speed
  (downstream_speed : ℝ)
  (stream_speed : ℝ)
  (h1 : downstream_speed = 11)
  (h2 : stream_speed = 1.5) :
  downstream_speed - 2 * stream_speed = 8 := by
  sorry

end mans_upstream_speed_l4084_408473


namespace probability_x_plus_y_leq_6_l4084_408465

/-- The probability that a randomly selected point (x, y) in the rectangle
    [0, 4] × [0, 8] satisfies x + y ≤ 6 is 3/8. -/
theorem probability_x_plus_y_leq_6 :
  let total_area : ℝ := 4 * 8
  let valid_area : ℝ := (1 / 2) * 4 * 6
  valid_area / total_area = 3 / 8 :=
by sorry

end probability_x_plus_y_leq_6_l4084_408465


namespace prep_school_cost_per_semester_l4084_408402

/-- The cost per semester for John's son's prep school -/
def cost_per_semester (total_cost : ℕ) (years : ℕ) (semesters_per_year : ℕ) : ℕ :=
  total_cost / (years * semesters_per_year)

/-- Proof that the cost per semester is $20,000 -/
theorem prep_school_cost_per_semester :
  cost_per_semester 520000 13 2 = 20000 := by
  sorry

end prep_school_cost_per_semester_l4084_408402


namespace fifteen_factorial_digit_sum_l4084_408479

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem fifteen_factorial_digit_sum :
  ∃ (H M T : ℕ),
    H < 10 ∧ M < 10 ∧ T < 10 ∧
    factorial 15 = 1307674 * 10^6 + H * 10^5 + M * 10^3 + 776 * 10^2 + T * 10 + 80 ∧
    H + M + T = 17 := by
  sorry

end fifteen_factorial_digit_sum_l4084_408479


namespace single_digit_between_zero_and_two_l4084_408405

theorem single_digit_between_zero_and_two : 
  ∃! n : ℕ, n < 10 ∧ 0 < n ∧ n < 2 :=
by sorry

end single_digit_between_zero_and_two_l4084_408405


namespace female_democrats_count_l4084_408499

theorem female_democrats_count (total : ℕ) (female : ℕ) (male : ℕ) : 
  total = 750 →
  female + male = total →
  female / 2 + male / 4 = total / 3 →
  female / 2 = 125 :=
by sorry

end female_democrats_count_l4084_408499


namespace quadratic_real_roots_l4084_408423

/-- The quadratic equation (m-1)x^2 - 2x + 1 = 0 has real roots if and only if m ≤ 2 and m ≠ 1 -/
theorem quadratic_real_roots (m : ℝ) : 
  (∃ x : ℝ, (m - 1) * x^2 - 2 * x + 1 = 0) ↔ (m ≤ 2 ∧ m ≠ 1) :=
sorry

end quadratic_real_roots_l4084_408423


namespace greatest_divisor_l4084_408455

def problem (n : ℕ) : Prop :=
  n > 0 ∧
  ∃ q1 q2 : ℕ, 1255 = n * q1 + 8 ∧ 1490 = n * q2 + 11 ∧
  ∀ m : ℕ, m > 0 → (∃ r1 r2 : ℕ, 1255 = m * r1 + 8 ∧ 1490 = m * r2 + 11) → m ≤ n

theorem greatest_divisor : problem 29 := by
  sorry

end greatest_divisor_l4084_408455


namespace purchase_price_l4084_408431

/-- The total price of a purchase of shirts and a tie -/
def total_price (shirt_price : ℝ) (tie_price : ℝ) (discount : ℝ) : ℝ :=
  2 * shirt_price + tie_price + shirt_price * (1 - discount)

/-- The proposition that the total price is 3500 rubles -/
theorem purchase_price :
  ∃ (shirt_price tie_price : ℝ),
    2 * shirt_price + tie_price = 2600 ∧
    total_price shirt_price tie_price 0.25 = 3500 ∧
    shirt_price = 1200 := by
  sorry

end purchase_price_l4084_408431


namespace solution_range_l4084_408436

theorem solution_range (x : ℝ) : 
  x > 9 → 
  Real.sqrt (x - 5 * Real.sqrt (x - 9)) + 3 = Real.sqrt (x + 5 * Real.sqrt (x - 9)) - 3 → 
  x ≥ 20.80 := by
  sorry

end solution_range_l4084_408436


namespace modified_triangle_sum_l4084_408488

/-- Represents the sum of numbers in the nth row of the modified triangular array -/
def f : ℕ → ℕ
  | 0 => 0  -- We define f(0) as 0 to make the function total
  | 1 => 0  -- First row starts with 0
  | (n + 2) => 2 * f (n + 1) + (n + 2) * (n + 2)

theorem modified_triangle_sum : f 100 = 2^100 - 10000 := by
  sorry

end modified_triangle_sum_l4084_408488


namespace alpha_value_l4084_408403

theorem alpha_value (f : ℝ → ℝ) (α : ℝ) 
  (h1 : ∀ x, f x = 4 / (1 - x)) 
  (h2 : f α = 2) : 
  α = -1 := by
sorry

end alpha_value_l4084_408403


namespace basil_leaves_count_l4084_408417

theorem basil_leaves_count (basil_pots rosemary_pots thyme_pots : ℕ)
  (rosemary_leaves_per_pot thyme_leaves_per_pot : ℕ)
  (total_leaves : ℕ) :
  basil_pots = 3 →
  rosemary_pots = 9 →
  thyme_pots = 6 →
  rosemary_leaves_per_pot = 18 →
  thyme_leaves_per_pot = 30 →
  total_leaves = 354 →
  ∃ (basil_leaves_per_pot : ℕ),
    basil_leaves_per_pot * basil_pots +
    rosemary_leaves_per_pot * rosemary_pots +
    thyme_leaves_per_pot * thyme_pots = total_leaves ∧
    basil_leaves_per_pot = 4 :=
by sorry

end basil_leaves_count_l4084_408417


namespace winter_olympics_souvenir_sales_l4084_408466

/-- Daily sales volume as a function of selling price -/
def daily_sales (x : ℝ) : ℝ := -10 * x + 740

/-- Daily profit as a function of selling price -/
def daily_profit (x : ℝ) : ℝ := daily_sales x * (x - 40)

/-- The selling price is between 44 and 52 yuan -/
def valid_price (x : ℝ) : Prop := 44 ≤ x ∧ x ≤ 52

theorem winter_olympics_souvenir_sales :
  ∃ (x : ℝ), valid_price x ∧
  (daily_profit x = 2400 → x = 50) ∧
  (∀ y, valid_price y → daily_profit y ≤ daily_profit 52) ∧
  daily_profit 52 = 2640 := by
  sorry


end winter_olympics_souvenir_sales_l4084_408466


namespace test_question_count_l4084_408426

/-- Given a test with two-point and four-point questions, prove the total number of questions. -/
theorem test_question_count (two_point_count four_point_count : ℕ) 
  (h1 : two_point_count = 30)
  (h2 : four_point_count = 10) :
  two_point_count + four_point_count = 40 := by
  sorry

#check test_question_count

end test_question_count_l4084_408426


namespace stock_worth_calculation_l4084_408418

/-- Proves that the total worth of the stock is 20000 given the specified conditions --/
theorem stock_worth_calculation (stock_worth : ℝ) : 
  (0.2 * stock_worth * 1.1 + 0.8 * stock_worth * 0.95 = stock_worth - 400) → 
  stock_worth = 20000 := by
  sorry

end stock_worth_calculation_l4084_408418


namespace grade_students_count_l4084_408419

theorem grade_students_count : ∃ n : ℕ, 
  400 < n ∧ n < 500 ∧
  n % 3 = 2 ∧
  n % 5 = 3 ∧
  n % 7 = 2 ∧
  n = 443 :=
by sorry

end grade_students_count_l4084_408419


namespace gcd_21n_plus_4_14n_plus_3_gcd_factorial_plus_one_gcd_F_m_F_n_l4084_408469

-- Problem 1
theorem gcd_21n_plus_4_14n_plus_3 (n : ℕ+) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := by sorry

-- Problem 2
theorem gcd_factorial_plus_one (n : ℕ) : Nat.gcd (Nat.factorial n + 1) (Nat.factorial (n + 1) + 1) = 1 := by sorry

-- Problem 3
def F (k : ℕ) : ℕ := 2^(2^k) + 1

theorem gcd_F_m_F_n (m n : ℕ) (h : m ≠ n) : Nat.gcd (F m) (F n) = 1 := by sorry

end gcd_21n_plus_4_14n_plus_3_gcd_factorial_plus_one_gcd_F_m_F_n_l4084_408469


namespace max_a_part_1_range_a_part_2_l4084_408400

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |2*x - a| + a
def g (x : ℝ) : ℝ := |2*x - 1|

-- Part I
theorem max_a_part_1 : 
  (∃ a_max : ℝ, ∀ a : ℝ, (∀ x : ℝ, g x ≤ 5 → f a x ≤ 6) → a ≤ a_max) ∧
  (∀ x : ℝ, g x ≤ 5 → f 1 x ≤ 6) :=
sorry

-- Part II
theorem range_a_part_2 : 
  {a : ℝ | ∀ x : ℝ, f a x + g x ≥ 3} = {a : ℝ | a ≥ 2} :=
sorry

end max_a_part_1_range_a_part_2_l4084_408400


namespace star_arrangement_count_l4084_408446

/-- The number of distinct arrangements of 12 objects on a regular six-pointed star -/
def star_arrangements : ℕ := 479001600

/-- The number of rotational and reflectional symmetries of a regular six-pointed star -/
def star_symmetries : ℕ := 12

/-- The total number of ways to arrange 12 objects in 12 positions -/
def total_arrangements : ℕ := Nat.factorial 12

theorem star_arrangement_count : 
  star_arrangements = total_arrangements / star_symmetries := by
  sorry

end star_arrangement_count_l4084_408446


namespace martha_total_time_l4084_408421

def router_time : ℕ := 10

def on_hold_time : ℕ := 6 * router_time

def yelling_time : ℕ := on_hold_time / 2

def total_time : ℕ := router_time + on_hold_time + yelling_time

theorem martha_total_time : total_time = 100 := by
  sorry

end martha_total_time_l4084_408421


namespace min_value_m_plus_n_l4084_408481

theorem min_value_m_plus_n (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : Real.sqrt (a * b) = 2) 
  (m n : ℝ) (h4 : m = b + 1/a) (h5 : n = a + 1/b) : 
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ Real.sqrt (x * y) = 2 → m + n ≤ x + y + 1/x + 1/y ∧ m + n ≥ 5 := by
  sorry

end min_value_m_plus_n_l4084_408481


namespace complex_number_quadrant_l4084_408493

theorem complex_number_quadrant (z : ℂ) (h : z * (1 - Complex.I) = 4 * Complex.I) :
  z.re < 0 ∧ z.im > 0 := by
  sorry

end complex_number_quadrant_l4084_408493


namespace negation_equivalence_l4084_408424

theorem negation_equivalence (a b c : ℝ) :
  ¬(a + b + c = 3 → a^2 + b^2 + c^2 ≥ 3) ↔ (a + b + c = 3 ∧ a^2 + b^2 + c^2 < 3) :=
by sorry

end negation_equivalence_l4084_408424


namespace election_votes_l4084_408461

theorem election_votes (total_votes : ℕ) (invalid_percent : ℚ) 
  (a_excess_percent : ℚ) (c_percent : ℚ) 
  (h_total : total_votes = 6800)
  (h_invalid : invalid_percent = 30 / 100)
  (h_a_excess : a_excess_percent = 18 / 100)
  (h_c : c_percent = 12 / 100) : 
  ∃ (b_votes c_votes : ℕ), 
    b_votes + c_votes = 2176 ∧ 
    b_votes + c_votes + (b_votes + (a_excess_percent * total_votes).floor) = 
      (total_votes * (1 - invalid_percent)).floor ∧
    c_votes = (c_percent * total_votes).floor := by
  sorry

end election_votes_l4084_408461


namespace earth_central_angle_special_case_l4084_408498

/-- Represents a point on Earth's surface -/
structure EarthPoint where
  latitude : Real
  longitude : Real

/-- The Earth, assumed to be a perfect sphere -/
structure Earth where
  center : Point
  radius : Real

/-- Calculates the central angle between two points on Earth -/
def centralAngle (earth : Earth) (p1 p2 : EarthPoint) : Real :=
  sorry

theorem earth_central_angle_special_case (earth : Earth) :
  let a : EarthPoint := { latitude := 0, longitude := 100 }
  let b : EarthPoint := { latitude := 30, longitude := -90 }
  centralAngle earth a b = 180 := by sorry

end earth_central_angle_special_case_l4084_408498


namespace salary_calculation_l4084_408484

/-- Represents the number of turbans given as part of the salary -/
def turbans : ℕ := sorry

/-- The annual base salary in rupees -/
def base_salary : ℕ := 90

/-- The price of each turban in rupees -/
def turban_price : ℕ := 70

/-- The number of months the servant worked -/
def months_worked : ℕ := 9

/-- The amount in rupees the servant received when leaving -/
def amount_received : ℕ := 50

/-- The total annual salary in rupees -/
def total_annual_salary : ℕ := base_salary + turbans * turban_price

/-- The fraction of the year the servant worked -/
def fraction_worked : ℚ := 3 / 4

theorem salary_calculation :
  (fraction_worked * total_annual_salary : ℚ) = (amount_received + turban_price : ℕ) → turbans = 1 :=
by sorry

end salary_calculation_l4084_408484


namespace money_division_l4084_408474

theorem money_division (a b c : ℚ) : 
  (4 * a = 5 * b) → 
  (5 * b = 10 * c) → 
  (c = 160) → 
  (a + b + c = 880) := by
sorry

end money_division_l4084_408474


namespace sand_delivery_theorem_l4084_408472

/-- The amount of sand remaining after a truck's journey -/
def sand_remaining (initial : Real) (loss : Real) : Real :=
  initial - loss

/-- The total amount of sand from all trucks -/
def total_sand (truck1 : Real) (truck2 : Real) (truck3 : Real) : Real :=
  truck1 + truck2 + truck3

theorem sand_delivery_theorem :
  let truck1_initial : Real := 4.1
  let truck1_loss : Real := 2.4
  let truck2_initial : Real := 5.7
  let truck2_loss : Real := 3.6
  let truck3_initial : Real := 8.2
  let truck3_loss : Real := 1.9
  total_sand
    (sand_remaining truck1_initial truck1_loss)
    (sand_remaining truck2_initial truck2_loss)
    (sand_remaining truck3_initial truck3_loss) = 10.1 := by
  sorry

end sand_delivery_theorem_l4084_408472


namespace prime_iff_divides_factorial_plus_one_l4084_408451

theorem prime_iff_divides_factorial_plus_one (n : ℕ) (h : n ≥ 2) :
  Nat.Prime n ↔ n ∣ (Nat.factorial (n - 1) + 1) := by
  sorry

end prime_iff_divides_factorial_plus_one_l4084_408451


namespace angle4_is_35_degrees_l4084_408495

-- Define angles as real numbers (in degrees)
variable (angle1 angle2 angle3 angle4 : ℝ)

-- State the theorem
theorem angle4_is_35_degrees
  (h1 : angle1 + angle2 = 180)
  (h2 : angle3 = angle4) :
  angle4 = 35 := by
  sorry

end angle4_is_35_degrees_l4084_408495


namespace total_pages_called_l4084_408471

def pages_last_week : ℝ := 10.2
def pages_this_week : ℝ := 8.6

theorem total_pages_called :
  pages_last_week + pages_this_week = 18.8 := by
  sorry

end total_pages_called_l4084_408471


namespace exam_score_problem_l4084_408439

/-- Given an exam with 150 questions, where correct answers score 5 marks,
    wrong answers lose 2 marks, and the total score is 370,
    prove that the number of correctly answered questions is 95. -/
theorem exam_score_problem (total_questions : ℕ) (correct_score : ℤ) (wrong_score : ℤ) (total_score : ℤ)
  (h_total : total_questions = 150)
  (h_correct : correct_score = 5)
  (h_wrong : wrong_score = -2)
  (h_score : total_score = 370) :
  ∃ (correct_answers : ℕ),
    correct_answers = 95 ∧
    correct_answers ≤ total_questions ∧
    (correct_score * correct_answers + wrong_score * (total_questions - correct_answers) = total_score) :=
by sorry


end exam_score_problem_l4084_408439


namespace imaginary_part_of_complex_fraction_l4084_408412

theorem imaginary_part_of_complex_fraction (i : ℂ) (h : i^2 = -1) :
  let z := (10 * i) / (3 + i)
  Complex.im z = 3 := by sorry

end imaginary_part_of_complex_fraction_l4084_408412


namespace factorization_coefficient_sum_l4084_408491

theorem factorization_coefficient_sum : 
  ∃ (a b c d e f g h i j k l m n o p : ℤ),
  (∀ x y : ℝ, 
    81 * x^8 - 256 * y^8 = 
    (a*x + b*y) * 
    (c*x^2 + d*x*y + e*y^2) * 
    (f*x^3 + g*x*y^2 + h*y^3) * 
    (i*x + j*y) * 
    (k*x^2 + l*x*y + m*y^2) * 
    (n*x^3 + o*x*y^2 + p*y^3)) →
  a + b + c + d + e + f + g + h + i + j + k + l + m + n + o + p = 40 :=
sorry

end factorization_coefficient_sum_l4084_408491


namespace unique_monic_polynomial_l4084_408407

/-- A monic polynomial of degree 3 satisfying f(0) = 3 and f(2) = 19 -/
def f : ℝ → ℝ :=
  fun x ↦ x^3 + x^2 + 2*x + 3

/-- Theorem stating that f is the unique monic polynomial of degree 3 satisfying the given conditions -/
theorem unique_monic_polynomial :
  (∀ x, f x = x^3 + x^2 + 2*x + 3) ∧
  (∀ p : ℝ → ℝ, (∃ a b c : ℝ, ∀ x, p x = x^3 + a*x^2 + b*x + c) →
    p 0 = 3 → p 2 = 19 → p = f) := by
  sorry

end unique_monic_polynomial_l4084_408407


namespace recurrence_sequence_property_l4084_408425

/-- A sequence of integers satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℤ) : Prop :=
  (∀ n, a n ≠ -1) ∧
  (∀ n, a (n + 2) = (a n + 2006) / (a (n + 1) + 1))

/-- The theorem stating the properties of the recurrence sequence -/
theorem recurrence_sequence_property (a : ℕ → ℤ) (h : RecurrenceSequence a) :
  ∃ x y : ℤ, x * y = 2006 ∧ (∀ n, a n = x ∨ a n = y) ∧ (∀ n, a n = a (n + 2)) := by
  sorry

end recurrence_sequence_property_l4084_408425


namespace roots_product_equality_l4084_408430

theorem roots_product_equality (p q : ℝ) (α β γ δ : ℝ) 
  (h1 : α^2 + p*α - 2 = 0)
  (h2 : β^2 + p*β - 2 = 0)
  (h3 : γ^2 + q*γ - 2 = 0)
  (h4 : δ^2 + q*δ - 2 = 0) :
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = -(p^2 - q^2) := by
  sorry

end roots_product_equality_l4084_408430


namespace james_lifting_time_l4084_408464

/-- Calculates the number of days until James can lift heavy again after an injury -/
def daysUntilHeavyLifting (painSubsideDays : ℕ) (healingMultiplier : ℕ) (waitAfterHealingDays : ℕ) (waitBeforeHeavyWeeks : ℕ) : ℕ :=
  let fullHealingDays := painSubsideDays * healingMultiplier
  let totalBeforeExercise := fullHealingDays + waitAfterHealingDays
  let waitBeforeHeavyDays := waitBeforeHeavyWeeks * 7
  totalBeforeExercise + waitBeforeHeavyDays

/-- Theorem stating that given the specific conditions, James can lift heavy after 39 days -/
theorem james_lifting_time :
  daysUntilHeavyLifting 3 5 3 3 = 39 := by
  sorry

end james_lifting_time_l4084_408464


namespace monic_quartic_value_at_zero_l4084_408453

/-- A monic quartic polynomial is a polynomial of degree 4 with leading coefficient 1 -/
def MonicQuarticPolynomial (h : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, h x = x^4 + a*x^3 + b*x^2 + c*x + d

/-- The main theorem -/
theorem monic_quartic_value_at_zero 
  (h : ℝ → ℝ) 
  (monic_quartic : MonicQuarticPolynomial h)
  (h_neg_two : h (-2) = -4)
  (h_one : h 1 = -1)
  (h_three : h 3 = -9)
  (h_five : h 5 = -25) : 
  h 0 = -30 := by sorry

end monic_quartic_value_at_zero_l4084_408453


namespace total_distance_driven_l4084_408411

theorem total_distance_driven (initial_speed initial_time : ℝ) : 
  initial_speed = 30 ∧ initial_time = 0.5 →
  (initial_speed * initial_time) + (2 * initial_speed * (2 * initial_time)) = 75 := by
  sorry

end total_distance_driven_l4084_408411


namespace angle_in_third_quadrant_l4084_408452

/-- Given an angle θ in the second quadrant satisfying the equation
    cos(θ/2) - sin(θ/2) = √(1 - sin(θ)), prove that θ/2 is in the third quadrant. -/
theorem angle_in_third_quadrant (θ : Real) 
  (h1 : π < θ ∧ θ < 3*π/2) -- θ is in the second quadrant
  (h2 : Real.cos (θ/2) - Real.sin (θ/2) = Real.sqrt (1 - Real.sin θ)) :
  π < θ/2 ∧ θ/2 < 3*π/2 := by
  sorry


end angle_in_third_quadrant_l4084_408452


namespace isosceles_triangle_perimeter_l4084_408420

-- Define an isosceles triangle with side lengths 3 and 7
def IsoscelesTriangle (a b c : ℝ) : Prop :=
  (a = 3 ∧ b = 7 ∧ c = 7) ∨ (a = 7 ∧ b = 3 ∧ c = 7) ∨ (a = 7 ∧ b = 7 ∧ c = 3)

-- Define the perimeter of a triangle
def Perimeter (a b c : ℝ) : ℝ := a + b + c

-- Theorem statement
theorem isosceles_triangle_perimeter :
  ∀ a b c : ℝ, IsoscelesTriangle a b c → Perimeter a b c = 17 :=
by
  sorry


end isosceles_triangle_perimeter_l4084_408420


namespace cos_555_degrees_l4084_408457

theorem cos_555_degrees : Real.cos (555 * π / 180) = -(Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end cos_555_degrees_l4084_408457


namespace polynomial_solutions_l4084_408440

def f (a b c d : ℝ) (x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

theorem polynomial_solutions (a b c d : ℝ) :
  (f a b c d 4 = 102) →
  (f a b c d 3 = 102) →
  (f a b c d (-3) = 102) →
  (f a b c d (-4) = 102) →
  ({x : ℝ | f a b c d x = 246} = {0, 5, -5}) :=
by sorry

end polynomial_solutions_l4084_408440


namespace divisibility_problem_l4084_408456

theorem divisibility_problem (m p a : ℕ) (hp : Prime p) (hm : m > 0) 
  (h1 : p ∣ (m^2 - 2)) (h2 : ∃ a : ℕ, a > 0 ∧ p ∣ (a^2 + m - 2)) :
  ∃ b : ℕ, b > 0 ∧ p ∣ (b^2 - m - 2) := by
  sorry

end divisibility_problem_l4084_408456


namespace right_triangle_area_l4084_408404

theorem right_triangle_area (base height hypotenuse : ℝ) :
  base = 12 →
  hypotenuse = 13 →
  base^2 + height^2 = hypotenuse^2 →
  (1/2) * base * height = 30 := by
sorry

end right_triangle_area_l4084_408404


namespace parallelepiped_sphere_properties_l4084_408415

/-- Represents a parallelepiped ABCDA₁B₁C₁D₁ with a sphere Ω touching its edges -/
structure Parallelepiped where
  -- Edge length of A₁A
  edge_length : ℝ
  -- Volume of the parallelepiped
  volume : ℝ
  -- Radius of the sphere Ω
  sphere_radius : ℝ
  -- A₁A is perpendicular to ABCD
  edge_perpendicular : edge_length > 0
  -- Sphere Ω touches BB₁, B₁C₁, C₁C, CB, C₁D₁, and AD
  sphere_touches_edges : True
  -- Ω touches C₁D₁ at K where C₁K = 9 and KD₁ = 4
  sphere_touch_point : edge_length > 13

/-- The theorem stating the properties of the parallelepiped and sphere -/
theorem parallelepiped_sphere_properties : 
  ∃ (p : Parallelepiped), 
    p.edge_length = 18 ∧ 
    p.volume = 3888 ∧ 
    p.sphere_radius = 3 * Real.sqrt 13 := by
  sorry

end parallelepiped_sphere_properties_l4084_408415


namespace zoo_layout_l4084_408483

theorem zoo_layout (tiger_enclosures : ℕ) (zebra_enclosures_per_tiger : ℕ) (giraffe_enclosure_ratio : ℕ)
  (tigers_per_enclosure : ℕ) (zebras_per_enclosure : ℕ) (total_animals : ℕ)
  (h1 : tiger_enclosures = 4)
  (h2 : zebra_enclosures_per_tiger = 2)
  (h3 : giraffe_enclosure_ratio = 3)
  (h4 : tigers_per_enclosure = 4)
  (h5 : zebras_per_enclosure = 10)
  (h6 : total_animals = 144) :
  (total_animals - (tiger_enclosures * tigers_per_enclosure + tiger_enclosures * zebra_enclosures_per_tiger * zebras_per_enclosure)) / 
  (giraffe_enclosure_ratio * tiger_enclosures * zebra_enclosures_per_tiger) = 2 :=
by sorry

end zoo_layout_l4084_408483


namespace four_heads_in_five_tosses_l4084_408438

/-- The probability of getting exactly k successes in n trials with probability p of success in each trial. -/
def binomialProbability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- Theorem: The probability of getting exactly 4 heads in 5 tosses of a fair coin is 0.15625. -/
theorem four_heads_in_five_tosses :
  binomialProbability 5 4 (1/2) = 0.15625 := by
sorry

end four_heads_in_five_tosses_l4084_408438


namespace chord_length_is_7_exists_unique_P_l4084_408447

-- Define the circle C₁
def C₁ (x y : ℝ) : Prop := (x + 4)^2 + y^2 = 16

-- Define point F
def F : ℝ × ℝ := (-2, 0)

-- Define the line x = -4
def line_x_eq_neg_4 (x y : ℝ) : Prop := x = -4

-- Define the property that G is the midpoint of GT
def is_midpoint_GT (G T : ℝ × ℝ) : Prop :=
  G.1 = (G.1 + T.1) / 2 ∧ G.2 = (G.2 + T.2) / 2

-- Theorem 1: The length of the chord cut by FG on C₁ is 7
theorem chord_length_is_7 (G : ℝ × ℝ) (T : ℝ × ℝ) :
  C₁ G.1 G.2 →
  line_x_eq_neg_4 T.1 T.2 →
  is_midpoint_GT G T →
  ∃ (A B : ℝ × ℝ), C₁ A.1 A.2 ∧ C₁ B.1 B.2 ∧ 
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 49 :=
sorry

-- Theorem 2: There exists a unique point P(4, 0) such that |GP| = 2|GF| for all G on C₁
theorem exists_unique_P (P : ℝ × ℝ) :
  P = (4, 0) ↔
  ∀ (G : ℝ × ℝ), C₁ G.1 G.2 →
    (G.1 - P.1)^2 + (G.2 - P.2)^2 = 4 * ((G.1 - F.1)^2 + (G.2 - F.2)^2) :=
sorry

end chord_length_is_7_exists_unique_P_l4084_408447


namespace root_product_theorem_l4084_408489

theorem root_product_theorem (n r : ℝ) (a b : ℝ) : 
  (a^2 - n*a + 6 = 0) → 
  (b^2 - n*b + 6 = 0) → 
  ((a + 2/b)^2 - r*(a + 2/b) + s = 0) → 
  ((b + 2/a)^2 - r*(b + 2/a) + s = 0) → 
  s = 32/3 := by sorry

end root_product_theorem_l4084_408489


namespace exists_valid_chain_l4084_408410

/-- A chain between two integers is a finite sequence of positive integers
    where the product of any two consecutive elements is divisible by their sum. -/
def IsValidChain (chain : List Nat) : Prop :=
  chain.length ≥ 2 ∧
  ∀ i, i + 1 < chain.length →
    (chain[i]! * chain[i+1]!) % (chain[i]! + chain[i+1]!) = 0

/-- For any two integers greater than 2, there exists a valid chain between them. -/
theorem exists_valid_chain (m n : Nat) (hm : m > 2) (hn : n > 2) :
  ∃ (chain : List Nat), chain.head! = m ∧ chain.getLast! = n ∧ IsValidChain chain :=
sorry

end exists_valid_chain_l4084_408410


namespace sqrt_3_binary_representation_l4084_408414

open Real

theorem sqrt_3_binary_representation (n : ℕ+) :
  ¬ (2^(n.val + 1) ∣ ⌊2^(2 * n.val) * Real.sqrt 3⌋) := by
  sorry

end sqrt_3_binary_representation_l4084_408414


namespace wilsborough_savings_l4084_408459

/-- Mrs. Wilsborough's concert ticket purchase problem -/
theorem wilsborough_savings : 
  let initial_savings : ℕ := 500
  let vip_ticket_price : ℕ := 100
  let regular_ticket_price : ℕ := 50
  let vip_tickets_bought : ℕ := 2
  let regular_tickets_bought : ℕ := 3
  let total_spent : ℕ := vip_ticket_price * vip_tickets_bought + regular_ticket_price * regular_tickets_bought
  let remaining_savings : ℕ := initial_savings - total_spent
  remaining_savings = 150 := by sorry

end wilsborough_savings_l4084_408459


namespace alcohol_mixture_percentage_l4084_408486

/-- Given an initial solution of 5 liters containing 40% alcohol,
    after adding 2 liters of water and 1 liter of pure alcohol,
    the resulting mixture contains 37.5% alcohol. -/
theorem alcohol_mixture_percentage :
  let initial_volume : ℝ := 5
  let initial_alcohol_percentage : ℝ := 40 / 100
  let water_added : ℝ := 2
  let pure_alcohol_added : ℝ := 1
  let final_volume : ℝ := initial_volume + water_added + pure_alcohol_added
  let final_alcohol_volume : ℝ := initial_volume * initial_alcohol_percentage + pure_alcohol_added
  final_alcohol_volume / final_volume = 3 / 8 := by
sorry

end alcohol_mixture_percentage_l4084_408486


namespace ticket_problem_l4084_408401

theorem ticket_problem (T : ℚ) : 
  (1/2 : ℚ) * T + (1/4 : ℚ) * ((1/2 : ℚ) * T) = 3600 → T = 5760 := by
  sorry

#check ticket_problem

end ticket_problem_l4084_408401


namespace unique_solution_system_l4084_408422

/-- The system of equations:
    3^y * 81 = 9^(x^2)
    lg y = lg x - lg 0.5
    has only one positive real solution (x, y) = (2, 4) -/
theorem unique_solution_system (x y : ℝ) 
  (h1 : (3 : ℝ)^y * 81 = 9^(x^2))
  (h2 : Real.log y / Real.log 10 = Real.log x / Real.log 10 - Real.log 0.5 / Real.log 10)
  (h3 : x > 0)
  (h4 : y > 0) : 
  x = 2 ∧ y = 4 :=
sorry

end unique_solution_system_l4084_408422


namespace smallest_period_is_40_l4084_408427

/-- A function satisfying the given condition -/
def SatisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 5) + f (x - 5) = f x

/-- The period of a function -/
def IsPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

/-- The smallest positive period of functions satisfying the condition -/
theorem smallest_period_is_40 :
  ∀ f : ℝ → ℝ, SatisfiesCondition f →
    (∃ p : ℝ, p > 0 ∧ IsPeriod f p ∧
      ∀ q : ℝ, q > 0 → IsPeriod f q → p ≤ q) →
    (∃ p : ℝ, p > 0 ∧ IsPeriod f p ∧
      ∀ q : ℝ, q > 0 → IsPeriod f q → p ≤ q) ∧ p = 40 :=
by sorry

end smallest_period_is_40_l4084_408427


namespace investment_problem_l4084_408467

/-- Proves that Raghu's investment is 2656.25 given the conditions of the problem -/
theorem investment_problem (raghu : ℝ) : 
  let trishul := 0.9 * raghu
  let vishal := 1.1 * trishul
  let chandni := 1.15 * vishal
  (raghu + trishul + vishal + chandni = 10700) →
  raghu = 2656.25 := by
sorry

end investment_problem_l4084_408467


namespace total_blocks_l4084_408476

def num_boxes : ℕ := 2
def blocks_per_box : ℕ := 6

theorem total_blocks : num_boxes * blocks_per_box = 12 := by
  sorry

end total_blocks_l4084_408476


namespace game_result_l4084_408463

/-- A game between two players where the winner gains 2 points and the loser loses 1 point. -/
structure Game where
  total_games : ℕ
  games_won_by_player1 : ℕ
  final_score_player2 : ℤ

/-- Theorem stating that if player1 wins exactly 3 games and player2 has a final score of 5,
    then the total number of games played is 7. -/
theorem game_result (g : Game) 
  (h1 : g.games_won_by_player1 = 3)
  (h2 : g.final_score_player2 = 5) :
  g.total_games = 7 := by
  sorry

end game_result_l4084_408463


namespace smallest_multiple_of_one_to_five_l4084_408475

theorem smallest_multiple_of_one_to_five : ∃ n : ℕ+, 
  (∀ m : ℕ, 1 ≤ m ∧ m ≤ 5 → m ∣ n) ∧
  (∀ k : ℕ+, (∀ m : ℕ, 1 ≤ m ∧ m ≤ 5 → m ∣ k) → n ≤ k) ∧
  n = 60 :=
by sorry

end smallest_multiple_of_one_to_five_l4084_408475


namespace at_least_four_same_probability_l4084_408454

/-- The number of sides on each die -/
def numSides : ℕ := 6

/-- The number of dice rolled -/
def numDice : ℕ := 5

/-- The probability of rolling a specific value on a single die -/
def singleDieProbability : ℚ := 1 / numSides

/-- The probability that all five dice show the same number -/
def allSameProbability : ℚ := singleDieProbability ^ (numDice - 1)

/-- The probability that exactly four dice show the same number and one die shows a different number -/
def fourSameProbability : ℚ :=
  (numDice : ℚ) * (singleDieProbability ^ (numDice - 2)) * (1 - singleDieProbability)

/-- The theorem stating the probability of at least four out of five fair six-sided dice showing the same value -/
theorem at_least_four_same_probability :
  allSameProbability + fourSameProbability = 13 / 648 := by
  sorry

end at_least_four_same_probability_l4084_408454


namespace integer_solutions_l4084_408497

/-- The equation whose solutions we're interested in -/
def equation (k x : ℝ) : Prop :=
  (k^2 - 2*k)*x^2 - (6*k - 4)*x + 8 = 0

/-- Predicate to check if a number is an integer -/
def isInteger (x : ℝ) : Prop := ∃ n : ℤ, x = n

/-- The main theorem stating the conditions for integer solutions -/
theorem integer_solutions (k : ℝ) :
  (∀ x : ℝ, equation k x → isInteger x) ↔ (k = 1 ∨ k = -2 ∨ k = 2/3) :=
sorry

end integer_solutions_l4084_408497


namespace top_of_second_column_is_20_l4084_408478

/-- Represents a 7x6 grid of numbers -/
def Grid := Fin 7 → Fin 6 → ℤ

/-- The given grid satisfies the problem conditions -/
def satisfies_conditions (g : Grid) : Prop :=
  -- Row is an arithmetic sequence with first element 15 and common difference 0
  (∀ i : Fin 7, g i 0 = 15) ∧
  -- Third column is an arithmetic sequence containing 10 and 5
  (g 2 1 = 10 ∧ g 2 2 = 5) ∧
  -- Second column's bottom element is -10
  (g 1 5 = -10) ∧
  -- Each column is an arithmetic sequence
  (∀ j : Fin 6, ∃ d : ℤ, ∀ i : Fin 5, g 1 (i + 1) = g 1 i + d) ∧
  (∀ j : Fin 6, ∃ d : ℤ, ∀ i : Fin 5, g 2 (i + 1) = g 2 i + d)

/-- The theorem to be proved -/
theorem top_of_second_column_is_20 (g : Grid) (h : satisfies_conditions g) : g 1 0 = 20 := by
  sorry

end top_of_second_column_is_20_l4084_408478


namespace journey_distance_l4084_408428

theorem journey_distance (total_distance : ℝ) (bike_speed walking_speed : ℝ) 
  (h1 : bike_speed = 12)
  (h2 : walking_speed = 4)
  (h3 : (3/4 * total_distance) / bike_speed + (1/4 * total_distance) / walking_speed = 1) :
  1/4 * total_distance = 2 := by
  sorry

end journey_distance_l4084_408428


namespace sin_cube_identity_l4084_408487

theorem sin_cube_identity (θ : ℝ) : 
  Real.sin θ ^ 3 = (-1/4) * Real.sin (3*θ) + (3/4) * Real.sin θ := by
  sorry

end sin_cube_identity_l4084_408487


namespace local_min_iff_a_lt_one_l4084_408492

/-- The function f(x) defined as (x-1)^2 * (x-a) -/
def f (a : ℝ) (x : ℝ) : ℝ := (x - 1)^2 * (x - a)

/-- x = 1 is a local minimum point of f(x) if and only if a < 1 -/
theorem local_min_iff_a_lt_one (a : ℝ) :
  (∃ δ > 0, ∀ x, |x - 1| < δ → f a x ≥ f a 1) ↔ a < 1 := by
  sorry

end local_min_iff_a_lt_one_l4084_408492


namespace rals_age_l4084_408432

theorem rals_age (suri_age suri_age_in_3_years ral_age : ℕ) :
  suri_age_in_3_years = suri_age + 3 →
  suri_age_in_3_years = 16 →
  ral_age = 2 * suri_age →
  ral_age = 26 := by
  sorry

end rals_age_l4084_408432


namespace find_divisor_l4084_408480

theorem find_divisor (dividend quotient remainder : ℕ) 
  (h1 : dividend = 15698)
  (h2 : quotient = 89)
  (h3 : remainder = 14)
  (h4 : dividend = quotient * 176 + remainder) :
  176 = dividend / quotient :=
by sorry

end find_divisor_l4084_408480


namespace lifeguard_swim_speed_l4084_408462

theorem lifeguard_swim_speed 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (front_crawl_time : ℝ) 
  (breaststroke_speed : ℝ) 
  (h1 : total_distance = 500)
  (h2 : total_time = 12)
  (h3 : front_crawl_time = 8)
  (h4 : breaststroke_speed = 35)
  : ∃ front_crawl_speed : ℝ, 
    front_crawl_speed * front_crawl_time + 
    breaststroke_speed * (total_time - front_crawl_time) = total_distance ∧ 
    front_crawl_speed = 45 := by
  sorry

end lifeguard_swim_speed_l4084_408462


namespace log_sum_simplification_l4084_408470

theorem log_sum_simplification :
  1 / (Real.log 3 / Real.log 12 + 2) +
  1 / (Real.log 2 / Real.log 8 + 2) +
  1 / (Real.log 3 / Real.log 9 + 2) = 2 := by
  sorry

end log_sum_simplification_l4084_408470


namespace polynomial_factorization_l4084_408434

theorem polynomial_factorization (x : ℝ) : x^4 - 64 = (x^2 - 8) * (x^2 + 8) := by
  sorry

end polynomial_factorization_l4084_408434


namespace solving_linear_equations_count_l4084_408450

/-- Given a total number of math homework problems, calculate the number of
    solving linear equations problems, knowing that 40% are Algebra problems
    and half of the Algebra problems are solving linear equations. -/
def solvingLinearEquationsProblems (total : ℕ) : ℕ :=
  (total * 40 / 100) / 2

/-- Proof that for 140 total math homework problems, the number of
    solving linear equations problems is 28. -/
theorem solving_linear_equations_count :
  solvingLinearEquationsProblems 140 = 28 := by
  sorry

#eval solvingLinearEquationsProblems 140

end solving_linear_equations_count_l4084_408450


namespace units_digit_of_n_l4084_408443

theorem units_digit_of_n (m n : ℕ) : 
  m * n = 31^6 ∧ m % 10 = 3 → n % 10 = 7 := by sorry

end units_digit_of_n_l4084_408443


namespace beam_width_calculation_beam_width_250_pounds_l4084_408441

/-- The maximum load a beam can support is directly proportional to its width -/
def load_proportional_to_width (load width : ℝ) : Prop :=
  ∃ k : ℝ, load = k * width

/-- Theorem: Given the proportionality between load and width, and a reference beam,
    calculate the width of a beam supporting a specific load -/
theorem beam_width_calculation
  (reference_width reference_load target_load : ℝ)
  (h_positive : reference_width > 0 ∧ reference_load > 0 ∧ target_load > 0)
  (h_prop : load_proportional_to_width reference_load reference_width)
  (h_prop_target : load_proportional_to_width target_load (target_load * reference_width / reference_load)) :
  load_proportional_to_width target_load ((target_load * reference_width) / reference_load) :=
by sorry

/-- The width of a beam supporting 250 pounds, given a reference beam of 3.5 inches
    supporting 583.3333 pounds, is 1.5 inches -/
theorem beam_width_250_pounds :
  (250 : ℝ) * (3.5 : ℝ) / (583.3333 : ℝ) = (1.5 : ℝ) :=
by sorry

end beam_width_calculation_beam_width_250_pounds_l4084_408441


namespace evaluate_expression_l4084_408494

theorem evaluate_expression : 3000 * (3000 ^ 3001) = 3000 ^ 3002 := by
  sorry

end evaluate_expression_l4084_408494


namespace smallest_number_l4084_408445

theorem smallest_number (S : Set ℤ) (h : S = {-4, -2, 0, 1}) : 
  ∃ m ∈ S, ∀ n ∈ S, m ≤ n ∧ m = -4 := by
  sorry

end smallest_number_l4084_408445


namespace function_inequality_implies_upper_bound_l4084_408448

theorem function_inequality_implies_upper_bound (a : ℝ) :
  (∀ x1 ∈ Set.Icc (1/2 : ℝ) 3, ∃ x2 ∈ Set.Icc 2 3, x1 + 4/x1 ≥ 2^x2 + a) →
  a ≤ 0 := by
  sorry

end function_inequality_implies_upper_bound_l4084_408448


namespace function_range_l4084_408442

-- Define the function
def f (x : ℝ) := x^2 - 2*x

-- Define the domain
def domain : Set ℝ := { x | -1 ≤ x ∧ x ≤ 3 }

-- State the theorem
theorem function_range : 
  { y | ∃ x ∈ domain, f x = y } = { y | -1 ≤ y ∧ y ≤ 3 } := by sorry

end function_range_l4084_408442


namespace inequality_proof_l4084_408406

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a * b * c ≤ 1 / 9 ∧ 
  (a / (b + c) + b / (a + c) + c / (a + b)) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
  sorry

end inequality_proof_l4084_408406


namespace seq_is_bounded_l4084_408490

-- Define P(n) as the product of all digits of n
def P (n : ℕ) : ℕ := sorry

-- Define the sequence (n_k)
def seq (k : ℕ) : ℕ → ℕ
  | n₁ => match k with
    | 0 => n₁
    | k + 1 => seq k n₁ + P (seq k n₁)

-- Theorem statement
theorem seq_is_bounded (n₁ : ℕ) : ∃ M : ℕ, ∀ k : ℕ, seq k n₁ ≤ M := by sorry

end seq_is_bounded_l4084_408490


namespace monday_production_tuesday_production_wednesday_production_thursday_pricing_l4084_408433

/-- Represents the recipe and conditions for making Zippies -/
structure ZippieRecipe where
  gluze_per_batch : ℚ
  blurpos_per_batch : ℚ
  zippies_per_batch : ℕ
  gluze_price : ℚ
  zippie_price : ℚ
  zippie_profit : ℚ

/-- The standard Zippie recipe -/
def standard_recipe : ZippieRecipe :=
  { gluze_per_batch := 4
  , blurpos_per_batch := 3
  , zippies_per_batch := 60
  , gluze_price := 1.8
  , zippie_price := 0.5
  , zippie_profit := 0.3 }

/-- Theorem for Monday's production -/
theorem monday_production (recipe : ZippieRecipe) (gluze_used : ℚ) :
  recipe = standard_recipe →
  gluze_used = 28 →
  gluze_used / recipe.gluze_per_batch * recipe.blurpos_per_batch = 21 :=
sorry

/-- Theorem for Tuesday's production -/
theorem tuesday_production (recipe : ZippieRecipe) (ingredient_used : ℚ) :
  recipe = standard_recipe →
  ingredient_used = 48 →
  (ingredient_used / recipe.gluze_per_batch * recipe.blurpos_per_batch = 36 ∨
   ingredient_used / recipe.blurpos_per_batch * recipe.gluze_per_batch = 64) :=
sorry

/-- Theorem for Wednesday's production -/
theorem wednesday_production (recipe : ZippieRecipe) (gluze_available blurpos_available : ℚ) :
  recipe = standard_recipe →
  gluze_available = 64 →
  blurpos_available = 42 →
  min (gluze_available / recipe.gluze_per_batch) (blurpos_available / recipe.blurpos_per_batch) * recipe.zippies_per_batch = 840 :=
sorry

/-- Theorem for Thursday's pricing -/
theorem thursday_pricing (recipe : ZippieRecipe) :
  recipe = standard_recipe →
  (recipe.zippie_price - recipe.zippie_profit) * recipe.zippies_per_batch - recipe.gluze_price * recipe.gluze_per_batch = 1.6 * recipe.blurpos_per_batch :=
sorry

end monday_production_tuesday_production_wednesday_production_thursday_pricing_l4084_408433


namespace prob_neither_chooses_D_l4084_408482

/-- Represents the four projects --/
inductive Project : Type
  | A
  | B
  | C
  | D

/-- Represents the outcome of both students' choices --/
structure Outcome :=
  (fanfan : Project)
  (lelle : Project)

/-- The set of all possible outcomes --/
def all_outcomes : Finset Outcome :=
  sorry

/-- The set of outcomes where neither student chooses project D --/
def outcomes_without_D : Finset Outcome :=
  sorry

/-- The probability of an event is the number of favorable outcomes
    divided by the total number of outcomes --/
def probability (event : Finset Outcome) : Rat :=
  (event.card : Rat) / (all_outcomes.card : Rat)

/-- The main theorem: probability of neither student choosing D is 1/2 --/
theorem prob_neither_chooses_D :
  probability outcomes_without_D = 1 / 2 :=
sorry

end prob_neither_chooses_D_l4084_408482


namespace tan_difference_special_angle_l4084_408460

theorem tan_difference_special_angle (α : Real) :
  2 * Real.tan α = 3 * Real.tan (π / 8) →
  Real.tan (α - π / 8) = (5 * Real.sqrt 2 + 1) / 49 := by
  sorry

end tan_difference_special_angle_l4084_408460


namespace original_price_calculation_l4084_408416

/-- Given a 6% rebate followed by a 10% sales tax, if the final price is Rs. 6876.1,
    then the original price was Rs. 6650. -/
theorem original_price_calculation (original_price : ℝ) : 
  (original_price * (1 - 0.06) * (1 + 0.10) = 6876.1) → 
  (original_price = 6650) := by sorry

end original_price_calculation_l4084_408416


namespace xy_upper_bound_and_min_value_l4084_408458

theorem xy_upper_bound_and_min_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 4) :
  x * y ≤ 4 ∧ ∃ (min : ℝ), min = 9/5 ∧ ∀ (a b : ℝ), a > 0 → b > 0 → a + b = 4 → 1/(a+1) + 4/b ≥ min :=
by sorry

end xy_upper_bound_and_min_value_l4084_408458


namespace cloth_cost_price_l4084_408413

theorem cloth_cost_price
  (total_length : ℕ)
  (selling_price : ℕ)
  (profit_per_meter : ℕ)
  (h1 : total_length = 60)
  (h2 : selling_price = 8400)
  (h3 : profit_per_meter = 12) :
  (selling_price - total_length * profit_per_meter) / total_length = 128 :=
by sorry

end cloth_cost_price_l4084_408413
