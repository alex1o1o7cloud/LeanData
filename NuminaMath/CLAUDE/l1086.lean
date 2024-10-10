import Mathlib

namespace inequality_proof_l1086_108687

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / (a + b) + 1 / (b + c) + 1 / (c + a) ≥ 
  (a + b + c)^2 / (a * b * (a + b) + b * c * (b + c) + c * a * (c + a)) := by
  sorry

end inequality_proof_l1086_108687


namespace smallest_distance_complex_circles_l1086_108609

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

end smallest_distance_complex_circles_l1086_108609


namespace profit_percentage_previous_year_l1086_108673

theorem profit_percentage_previous_year
  (revenue_prev : ℝ)
  (revenue_2009 : ℝ)
  (profit_prev : ℝ)
  (profit_2009 : ℝ)
  (h1 : revenue_2009 = 0.8 * revenue_prev)
  (h2 : profit_2009 = 0.11 * revenue_2009)
  (h3 : profit_2009 = 0.8800000000000001 * profit_prev) :
  profit_prev = 0.1 * revenue_prev :=
sorry

end profit_percentage_previous_year_l1086_108673


namespace max_x_minus_y_l1086_108692

theorem max_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  ∃ (z : ℝ), z = x - y ∧ z ≤ 1 + 3 * Real.sqrt 2 ∧
  ∀ (w : ℝ), w = x - y → w ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end max_x_minus_y_l1086_108692


namespace coin_flip_probability_l1086_108647

def num_flips : ℕ := 12

def favorable_outcomes : ℕ := (
  Nat.choose num_flips 7 + 
  Nat.choose num_flips 8 + 
  Nat.choose num_flips 9 + 
  Nat.choose num_flips 10 + 
  Nat.choose num_flips 11 + 
  Nat.choose num_flips 12
)

def total_outcomes : ℕ := 2^num_flips

theorem coin_flip_probability : 
  (favorable_outcomes : ℚ) / total_outcomes = 793 / 2048 :=
by sorry

end coin_flip_probability_l1086_108647


namespace exists_21_win_stretch_l1086_108690

/-- Represents the cumulative wins of a chess player over 77 days -/
def CumulativeWins := Fin 78 → ℕ

/-- The conditions for the chess player's winning record -/
def ValidWinningRecord (x : CumulativeWins) : Prop :=
  (∀ i : Fin 77, x (i + 1) > x i) ∧ 
  (∀ i : Fin 71, x (i + 7) - x i ≤ 12) ∧
  x 0 = 0 ∧ x 77 ≤ 132

/-- The theorem stating that there exists a stretch of consecutive days with exactly 21 wins -/
theorem exists_21_win_stretch (x : CumulativeWins) (h : ValidWinningRecord x) : 
  ∃ i j : Fin 78, i < j ∧ x j - x i = 21 := by
  sorry


end exists_21_win_stretch_l1086_108690


namespace inscribed_square_area_equals_rectangle_area_l1086_108686

/-- Given a right triangle with legs a and b, and a square with side length x
    inscribed such that one angle coincides with the right angle of the triangle
    and one vertex lies on the hypotenuse, the area of the square is equal to
    the area of the rectangle formed by the remaining segments of the legs. -/
theorem inscribed_square_area_equals_rectangle_area 
  (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : 0 < x ∧ x < min a b) : 
  x^2 = (a - x) * (b - x) := by
  sorry

end inscribed_square_area_equals_rectangle_area_l1086_108686


namespace max_value_expression_l1086_108643

theorem max_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x^2 + y - Real.sqrt (x^4 + y^2)) / x ≤ (1 : ℝ) / 2 ∧
  ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ (x₀^2 + y₀ - Real.sqrt (x₀^4 + y₀^2)) / x₀ = (1 : ℝ) / 2 :=
sorry

end max_value_expression_l1086_108643


namespace smallest_group_size_l1086_108651

theorem smallest_group_size (n : ℕ) : n = 154 ↔ 
  n > 0 ∧ 
  n % 6 = 1 ∧ 
  n % 8 = 2 ∧ 
  n % 9 = 4 ∧ 
  ∀ m : ℕ, m > 0 → m % 6 = 1 → m % 8 = 2 → m % 9 = 4 → n ≤ m :=
by sorry

end smallest_group_size_l1086_108651


namespace f_properties_l1086_108695

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (a^x - a^(-x)) / (a - 1)

-- Theorem statement
theorem f_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x : ℝ, f a (-x) = -f a x) ∧
  StrictMono (f a) :=
by sorry

end

end f_properties_l1086_108695


namespace fifth_term_is_81_l1086_108667

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 2) / a (n + 1) = a (n + 1) / a n
  sum_1_3 : a 1 + a 3 = 10
  sum_2_4 : a 2 + a 4 = -30

/-- The fifth term of the geometric sequence is 81 -/
theorem fifth_term_is_81 (seq : GeometricSequence) : seq.a 5 = 81 := by
  sorry

end fifth_term_is_81_l1086_108667


namespace negation_distribution_l1086_108638

theorem negation_distribution (m : ℝ) : -(m - 2) = -m + 2 := by
  sorry

end negation_distribution_l1086_108638


namespace find_m_l1086_108625

theorem find_m : ∃ m : ℚ, 
  (∀ x : ℚ, 4 * x + 2 * m = 5 * x + 1 ↔ 3 * x = 6 * x - 1) → 
  m = 2 / 3 := by
  sorry

end find_m_l1086_108625


namespace simplify_complex_fraction_l1086_108649

theorem simplify_complex_fraction (y : ℝ) 
  (h1 : y ≠ 4) (h2 : y ≠ 2) (h3 : y ≠ 5) (h4 : y ≠ 7) (h5 : y ≠ 1) :
  (y^2 - 4*y + 3) / (y^2 - 6*y + 8) / ((y^2 - 9*y + 20) / (y^2 - 9*y + 14)) = 
  ((y - 3) * (y - 7)) / ((y - 1) * (y - 5)) :=
by sorry

end simplify_complex_fraction_l1086_108649


namespace no_savings_l1086_108678

-- Define the prices and fees
def in_store_price : ℚ := 129.99
def online_payment : ℚ := 29.99
def shipping_fee : ℚ := 11.99

-- Define the number of online payments
def num_payments : ℕ := 4

-- Define the function to calculate savings in cents
def savings_in_cents : ℚ :=
  (in_store_price - (num_payments * online_payment + shipping_fee)) * 100

-- Theorem statement
theorem no_savings : savings_in_cents = 0 := by
  sorry

end no_savings_l1086_108678


namespace p_shape_points_for_10cm_square_l1086_108656

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

end p_shape_points_for_10cm_square_l1086_108656


namespace sum_of_f_symmetric_points_sum_of_roots_l1086_108622

-- Define the cubic function f
def f (x : ℝ) : ℝ := x^3 + 2*x - 1

-- Theorem 1
theorem sum_of_f_symmetric_points (x₁ x₂ : ℝ) (h : x₁ + x₂ = 0) : 
  f x₁ + f x₂ = -2 := by sorry

-- Theorem 2
theorem sum_of_roots (m n : ℝ) 
  (hm : m^3 - 3*m^2 + 5*m - 4 = 0) 
  (hn : n^3 - 3*n^2 + 5*n - 2 = 0) : 
  m + n = 2 := by sorry

end sum_of_f_symmetric_points_sum_of_roots_l1086_108622


namespace tangent_line_and_extrema_l1086_108665

def f (x : ℝ) := x^3 + 3*x^2 - 9*x + 1

theorem tangent_line_and_extrema :
  ∃ (y : ℝ → ℝ),
    (∀ x, y x = -9*x + 1) ∧
    (∀ x, x ∈ [-1, 2] → f x ≤ 12) ∧
    (∀ x, x ∈ [-1, 2] → f x ≥ -4) ∧
    (∃ x₁ ∈ [-1, 2], f x₁ = 12) ∧
    (∃ x₂ ∈ [-1, 2], f x₂ = -4) := by
  sorry

end tangent_line_and_extrema_l1086_108665


namespace consecutive_negative_integers_sum_l1086_108661

theorem consecutive_negative_integers_sum (n : ℤ) : 
  n < 0 ∧ n * (n + 1) = 2142 → n + (n + 1) = -93 := by
  sorry

end consecutive_negative_integers_sum_l1086_108661


namespace max_value_expression_max_value_achievable_l1086_108685

theorem max_value_expression (x y : ℝ) :
  (x + 2*y + 3) / Real.sqrt (x^2 + y^2 + 1) ≤ Real.sqrt 14 :=
sorry

theorem max_value_achievable :
  ∃ x y : ℝ, (x + 2*y + 3) / Real.sqrt (x^2 + y^2 + 1) = Real.sqrt 14 :=
sorry

end max_value_expression_max_value_achievable_l1086_108685


namespace distance_school_to_david_value_total_distance_sum_l1086_108698

/-- The distance Craig walked from school to David's house -/
def distance_school_to_david : ℝ := sorry

/-- The distance Craig walked from David's house to his own house -/
def distance_david_to_craig : ℝ := 0.7

/-- The total distance Craig walked -/
def total_distance : ℝ := 0.9

/-- Theorem stating that the distance from school to David's house is 0.2 miles -/
theorem distance_school_to_david_value :
  distance_school_to_david = 0.2 :=
by
  sorry

/-- Theorem stating that the total distance is the sum of the two parts -/
theorem total_distance_sum :
  total_distance = distance_school_to_david + distance_david_to_craig :=
by
  sorry

end distance_school_to_david_value_total_distance_sum_l1086_108698


namespace sum_greater_than_double_smaller_l1086_108626

theorem sum_greater_than_double_smaller (a b c : ℝ) 
  (h1 : a > c) (h2 : b > c) : a + b > 2 * c := by
  sorry

end sum_greater_than_double_smaller_l1086_108626


namespace never_sunday_date_l1086_108604

/-- Represents a day of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Represents a month of the year -/
inductive Month
| January
| February
| March
| April
| May
| June
| July
| August
| September
| October
| November
| December

/-- Function to determine the day of the week for a given date in a month -/
def dayOfWeek (date : Nat) (month : Month) (isLeapYear : Bool) : DayOfWeek :=
  sorry

/-- Theorem stating that 31 is the only date that can never be a Sunday in any month of a year -/
theorem never_sunday_date :
  ∀ (date : Nat),
    (∀ (month : Month) (isLeapYear : Bool),
      dayOfWeek date month isLeapYear ≠ DayOfWeek.Sunday) ↔ date = 31 :=
by sorry

end never_sunday_date_l1086_108604


namespace value_of_expression_l1086_108606

theorem value_of_expression (a b : ℝ) (h : 2 * a + 4 * b = 3) :
  4 * a + 8 * b - 2 = 4 := by
  sorry

end value_of_expression_l1086_108606


namespace sum_prime_factors_2310_l1086_108659

def prime_factors (n : ℕ) : List ℕ := sorry

theorem sum_prime_factors_2310 : (prime_factors 2310).sum = 28 := by sorry

end sum_prime_factors_2310_l1086_108659


namespace odds_against_C_winning_l1086_108628

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


end odds_against_C_winning_l1086_108628


namespace franks_books_l1086_108641

theorem franks_books (a b c : ℤ) (n : ℕ) (p d t : ℕ) :
  p = 2 * a →
  d = 3 * b →
  t = 2 * c * (3 * b) →
  n * p = t →
  n * d = t →
  ∃ (k : ℤ), n = 2 * k ∧ k = c := by
  sorry

end franks_books_l1086_108641


namespace range_of_a_l1086_108605

-- Define proposition p
def p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 > -2 * a * x - 8

-- Define proposition q
def q (a : ℝ) : Prop := ∃ (h k r : ℝ), ∀ (x y : ℝ), 
  x^2 + y^2 - 4*x + a = 0 ↔ (x - h)^2 + (y - k)^2 = r^2

-- Main theorem
theorem range_of_a : 
  (∃ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a)) → 
  {a : ℝ | a < 0 ∨ (a ≥ 4 ∧ a < 8)} = {a : ℝ | p a ∨ q a} :=
sorry

end range_of_a_l1086_108605


namespace quadrilateral_area_is_3_root_6_over_2_l1086_108635

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

end quadrilateral_area_is_3_root_6_over_2_l1086_108635


namespace flight_duration_theorem_l1086_108634

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

end flight_duration_theorem_l1086_108634


namespace machine_working_time_l1086_108615

/-- The number of shirts made by the machine -/
def total_shirts : ℕ := 196

/-- The number of shirts the machine can make per minute -/
def shirts_per_minute : ℕ := 7

/-- The time worked by the machine in minutes -/
def time_worked : ℕ := total_shirts / shirts_per_minute

theorem machine_working_time : time_worked = 28 := by
  sorry

end machine_working_time_l1086_108615


namespace first_day_revenue_l1086_108603

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

end first_day_revenue_l1086_108603


namespace quadratic_roots_property_l1086_108653

theorem quadratic_roots_property (a b : ℝ) : 
  (3 * a^2 + 9 * a - 21 = 0) →
  (3 * b^2 + 9 * b - 21 = 0) →
  (3 * a - 4) * (6 * b - 8) = 14 := by
sorry

end quadratic_roots_property_l1086_108653


namespace transaction_result_l1086_108666

theorem transaction_result (car_sale_price motorcycle_sale_price : ℝ)
  (car_loss_percent motorcycle_gain_percent : ℝ)
  (h1 : car_sale_price = 18000)
  (h2 : motorcycle_sale_price = 10000)
  (h3 : car_loss_percent = 10)
  (h4 : motorcycle_gain_percent = 25) :
  car_sale_price + motorcycle_sale_price =
  (car_sale_price / (100 - car_loss_percent) * 100) +
  (motorcycle_sale_price / (100 + motorcycle_gain_percent) * 100) :=
by sorry

end transaction_result_l1086_108666


namespace geometric_series_ratio_l1086_108636

theorem geometric_series_ratio (a r : ℝ) (h1 : r ≠ 1) : 
  (∃ (S : ℝ), S = a / (1 - r) ∧ S = 18) →
  (∃ (S_odd : ℝ), S_odd = a * r / (1 - r^2) ∧ S_odd = 6) →
  r = 1/2 := by
sorry

end geometric_series_ratio_l1086_108636


namespace journey_time_proof_l1086_108663

/-- Proves that a journey of 336 km, with the first half traveled at 21 km/hr
    and the second half at 24 km/hr, takes 15 hours to complete. -/
theorem journey_time_proof (total_distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) :
  total_distance = 336 ∧ speed1 = 21 ∧ speed2 = 24 →
  (total_distance / 2 / speed1) + (total_distance / 2 / speed2) = 15 := by
  sorry

end journey_time_proof_l1086_108663


namespace triangle_properties_l1086_108611

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


end triangle_properties_l1086_108611


namespace coin_problem_l1086_108655

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

end coin_problem_l1086_108655


namespace brown_mm_averages_l1086_108642

def brown_smiley_counts : List Nat := [9, 12, 8, 8, 3]
def brown_star_counts : List Nat := [7, 14, 11, 6, 10]

theorem brown_mm_averages :
  let smiley_avg := (brown_smiley_counts.sum : ℚ) / brown_smiley_counts.length
  let star_avg := (brown_star_counts.sum : ℚ) / brown_star_counts.length
  smiley_avg = 8 ∧ star_avg = 9.6 := by
  sorry

end brown_mm_averages_l1086_108642


namespace suit_price_increase_l1086_108668

/-- Proves that the percentage increase in the price of a suit is 30% -/
theorem suit_price_increase (original_price : ℝ) (final_price : ℝ) : 
  original_price = 200 →
  final_price = 182 →
  ∃ (increase_percentage : ℝ),
    increase_percentage = 30 ∧
    final_price = original_price * (1 + increase_percentage / 100) * 0.7 :=
by sorry

end suit_price_increase_l1086_108668


namespace special_triangle_angles_l1086_108646

-- Define a triangle with sides a, b, c and angles A, B, C
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the properties of the specific triangle
def SpecialTriangle (t : Triangle) : Prop :=
  t.C = 2 * t.A ∧ t.b = 2 * t.a

-- Theorem statement
theorem special_triangle_angles (t : Triangle) 
  (h : SpecialTriangle t) : 
  t.A = 30 ∧ t.B = 90 ∧ t.C = 60 := by
  sorry

end special_triangle_angles_l1086_108646


namespace zero_point_in_interval_l1086_108614

noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 / x

theorem zero_point_in_interval :
  ∃ (c : ℝ), 2 < c ∧ c < 3 ∧ f c = 0 :=
by
  sorry

end zero_point_in_interval_l1086_108614


namespace special_inequality_l1086_108699

/-- The equation x^2 - 4x + |a-3| = 0 has real roots with respect to x -/
def has_real_roots (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - 4*x + |a-3| = 0

/-- The inequality t^2 - 2at + 12 < 0 holds for all a in [-1, 7] -/
def inequality_holds (t : ℝ) : Prop :=
  ∀ a : ℝ, -1 ≤ a ∧ a ≤ 7 → t^2 - 2*a*t + 12 < 0

theorem special_inequality (t : ℝ) :
  (∃ a : ℝ, has_real_roots a) →
  inequality_holds t →
  3 < t ∧ t < 4 := by
  sorry

end special_inequality_l1086_108699


namespace second_term_is_twelve_l1086_108616

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

end second_term_is_twelve_l1086_108616


namespace angle_covered_in_three_layers_l1086_108630

/-- Given a 90-degree angle covered by some angles with the same vertex in two or three layers,
    if the sum of the angles is 290 degrees, then the measure of the angle covered in three layers is 20 degrees. -/
theorem angle_covered_in_three_layers 
  (total_angle : ℝ) 
  (sum_of_angles : ℝ) 
  (angle_covered_three_layers : ℝ) 
  (angle_covered_two_layers : ℝ) 
  (h1 : total_angle = 90)
  (h2 : sum_of_angles = 290)
  (h3 : angle_covered_three_layers + angle_covered_two_layers = total_angle)
  (h4 : 3 * angle_covered_three_layers + 2 * angle_covered_two_layers = sum_of_angles) :
  angle_covered_three_layers = 20 := by
  sorry

end angle_covered_in_three_layers_l1086_108630


namespace absent_workers_fraction_l1086_108680

theorem absent_workers_fraction (p : ℕ) (W : ℝ) (h : p > 0) :
  let work_per_person := W / p
  let absent_fraction : ℝ → ℝ := λ x => x
  let present_workers : ℝ → ℝ := λ x => p * (1 - x)
  let increased_work_per_person := work_per_person * 1.2
  increased_work_per_person = W / (present_workers (absent_fraction (1/6))) :=
by sorry

end absent_workers_fraction_l1086_108680


namespace quadratic_equal_roots_l1086_108650

theorem quadratic_equal_roots (c : ℝ) : 
  (∃ x : ℝ, x^2 - 4*x + c = 0 ∧ (∀ y : ℝ, y^2 - 4*y + c = 0 → y = x)) → c = 4 := by
  sorry

end quadratic_equal_roots_l1086_108650


namespace equation_solution_l1086_108613

theorem equation_solution : ∀ (x : ℝ) (number : ℝ),
  x = 4 →
  7 * (x - 1) = number →
  number = 21 := by
  sorry

end equation_solution_l1086_108613


namespace at_least_one_survives_one_of_each_type_survives_l1086_108633

-- Define the survival probabilities
def survival_rate_A : ℚ := 5/6
def survival_rate_B : ℚ := 4/5

-- Define the number of trees of each type
def num_trees_A : ℕ := 2
def num_trees_B : ℕ := 2

-- Define the total number of trees
def total_trees : ℕ := num_trees_A + num_trees_B

-- Theorem for the probability that at least one tree survives
theorem at_least_one_survives :
  1 - (1 - survival_rate_A)^num_trees_A * (1 - survival_rate_B)^num_trees_B = 899/900 := by
  sorry

-- Theorem for the probability that one tree of each type survives
theorem one_of_each_type_survives :
  num_trees_A * survival_rate_A * (1 - survival_rate_A) *
  num_trees_B * survival_rate_B * (1 - survival_rate_B) = 4/45 := by
  sorry

end at_least_one_survives_one_of_each_type_survives_l1086_108633


namespace absolute_value_equation_solution_l1086_108652

theorem absolute_value_equation_solution :
  ∃! x : ℚ, |x - 5| = 3 * x + 6 :=
by
  -- The proof would go here
  sorry

end absolute_value_equation_solution_l1086_108652


namespace complex_absolute_value_l1086_108682

theorem complex_absolute_value : 
  Complex.abs (7/4 - 3*Complex.I + Real.sqrt 3) = 
  (Real.sqrt (241 + 56*Real.sqrt 3))/4 := by
sorry

end complex_absolute_value_l1086_108682


namespace polynomial_division_remainder_l1086_108621

theorem polynomial_division_remainder : ∃ q : Polynomial ℤ, 
  2 * X^4 + 10 * X^3 - 45 * X^2 - 55 * X + 52 = 
  (X^2 + 8 * X - 6) * q + (-211 * X + 142) := by
  sorry

end polynomial_division_remainder_l1086_108621


namespace division_threefold_change_l1086_108600

theorem division_threefold_change (a b c d : ℤ) (h : a = b * c + d) :
  ∃ (d' : ℤ), (3 * a) = (3 * b) * c + d' ∧ d' = 3 * d :=
sorry

end division_threefold_change_l1086_108600


namespace amanda_loan_l1086_108683

/-- Calculates the earnings for a given number of hours based on a cyclic payment structure -/
def calculateEarnings (hours : ℕ) : ℕ :=
  let fullCycles := hours / 4
  let remainingHours := hours % 4
  let earningsPerCycle := 10
  let earningsFromFullCycles := fullCycles * earningsPerCycle
  let earningsFromRemainingHours := 
    if remainingHours = 1 then 2
    else if remainingHours = 2 then 5
    else if remainingHours = 3 then 7
    else 0
  earningsFromFullCycles + earningsFromRemainingHours

theorem amanda_loan (x : ℕ) : 
  (x = calculateEarnings 50) → x = 125 := by
  sorry

end amanda_loan_l1086_108683


namespace square_with_tens_digit_7_l1086_108675

/-- A square number with tens digit 7 has units digit 6 -/
theorem square_with_tens_digit_7 (n : ℕ) :
  (n^2 / 10) % 10 = 7 → n^2 % 10 = 6 := by
  sorry

end square_with_tens_digit_7_l1086_108675


namespace tangent_circle_radius_l1086_108676

/-- A circle tangent to coordinate axes and hypotenuse of a 45-45-90 triangle --/
structure TangentCircle where
  O : ℝ × ℝ  -- Center of the circle
  r : ℝ      -- Radius of the circle
  h : ℝ      -- hypotenuse length of the 45-45-90 triangle

/-- The circle is tangent to both axes and the hypotenuse --/
def is_tangent (c : TangentCircle) : Prop :=
  c.O.1 = c.r ∧ c.O.2 = c.r ∧ c.O.1 + c.O.2 + c.r = c.h

theorem tangent_circle_radius (c : TangentCircle) 
  (h_hypotenuse : c.h = 2 * Real.sqrt 2)
  (h_tangent : is_tangent c) : 
  c.r = Real.sqrt 2 :=
sorry

end tangent_circle_radius_l1086_108676


namespace root_product_of_equation_l1086_108632

theorem root_product_of_equation : ∃ (x y : ℝ), 
  (Real.sqrt (2 * x^2 + 8 * x + 1) - x = 3) ∧
  (Real.sqrt (2 * y^2 + 8 * y + 1) - y = 3) ∧
  (x ≠ y) ∧ (x * y = -8) := by
  sorry

end root_product_of_equation_l1086_108632


namespace contractor_payment_proof_l1086_108689

/-- Calculates the total amount received by a contractor given the contract details and attendance. -/
def contractorPayment (totalDays duration : ℕ) (dailyWage dailyFine : ℚ) (absentDays : ℕ) : ℚ :=
  let workingDays := duration - absentDays
  let earnings := workingDays * dailyWage
  let fines := absentDays * dailyFine
  earnings - fines

/-- Proves that the contractor receives Rs. 490 under the given conditions. -/
theorem contractor_payment_proof :
  contractorPayment 30 30 25 (7.5 : ℚ) 8 = 490 := by
  sorry

#eval contractorPayment 30 30 25 (7.5 : ℚ) 8

end contractor_payment_proof_l1086_108689


namespace tens_digit_of_nine_power_2023_l1086_108664

theorem tens_digit_of_nine_power_2023 : 9^2023 % 100 = 29 := by
  sorry

end tens_digit_of_nine_power_2023_l1086_108664


namespace fermat_1000_units_digit_l1086_108660

/-- Fermat number F_n is defined as 2^(2^n) + 1 -/
def fermat_number (n : ℕ) : ℕ := 2^(2^n) + 1

/-- The units digit of a natural number -/
def units_digit (n : ℕ) : ℕ := n % 10

theorem fermat_1000_units_digit :
  units_digit (fermat_number 1000) = 7 := by sorry

end fermat_1000_units_digit_l1086_108660


namespace geometric_sequence_sum_l1086_108670

theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) :
  (∀ n, a (n + 1) = a n * r) →  -- geometric sequence condition
  a 4 + a 6 = 3 →               -- given condition
  a 4^2 + 2*a 4*a 6 + a 5*a 7 = 9 := by
sorry

end geometric_sequence_sum_l1086_108670


namespace complex_product_real_l1086_108601

theorem complex_product_real (m : ℝ) : 
  (Complex.I : ℂ) * (1 - m * Complex.I) + (m^2 : ℂ) * (1 - m * Complex.I) ∈ Set.range Complex.ofReal → 
  m = 1 :=
by sorry

end complex_product_real_l1086_108601


namespace price_theorem_min_bottles_theorem_l1086_108662

-- Define variables
def peanut_oil_price : ℝ := sorry
def corn_oil_price : ℝ := sorry
def peanut_oil_sell_price : ℝ := 60
def peanut_oil_purchased : ℕ := 50

-- Define conditions
axiom condition1 : 20 * peanut_oil_price + 30 * corn_oil_price = 2200
axiom condition2 : 30 * peanut_oil_price + 10 * corn_oil_price = 1900

-- Define theorems to prove
theorem price_theorem :
  peanut_oil_price = 50 ∧ corn_oil_price = 40 :=
sorry

theorem min_bottles_theorem :
  ∀ n : ℕ, n * peanut_oil_sell_price > peanut_oil_purchased * peanut_oil_price →
  n ≥ 42 :=
sorry

end price_theorem_min_bottles_theorem_l1086_108662


namespace triangle_perimeter_l1086_108619

theorem triangle_perimeter (a b c : ℝ) : 
  a = 2 ∧ b = 5 ∧ 
  c^2 - 8*c + 12 = 0 ∧
  a + b > c ∧ a + c > b ∧ b + c > a →
  a + b + c = 13 := by
sorry

end triangle_perimeter_l1086_108619


namespace black_area_after_four_changes_l1086_108631

/-- Represents the fraction of black area remaining after a certain number of changes --/
def blackAreaFraction (changes : ℕ) : ℚ :=
  (3/4) ^ changes

/-- The number of changes applied to the triangle --/
def totalChanges : ℕ := 4

/-- Theorem stating that after four changes, the fraction of the original area that remains black is 81/256 --/
theorem black_area_after_four_changes :
  blackAreaFraction totalChanges = 81/256 := by
  sorry

#eval blackAreaFraction totalChanges

end black_area_after_four_changes_l1086_108631


namespace xiaoming_pe_grade_l1086_108627

/-- Calculates the semester physical education grade based on given scores and weights -/
def calculate_semester_grade (extracurricular_score midterm_score final_score : ℚ) 
  (extracurricular_weight midterm_weight final_weight : ℕ) : ℚ :=
  (extracurricular_score * extracurricular_weight + 
   midterm_score * midterm_weight + 
   final_score * final_weight) / 
  (extracurricular_weight + midterm_weight + final_weight)

/-- Xiaoming's physical education grade theorem -/
theorem xiaoming_pe_grade :
  let max_score : ℚ := 100
  let extracurricular_score : ℚ := 95
  let midterm_score : ℚ := 90
  let final_score : ℚ := 85
  let extracurricular_weight : ℕ := 2
  let midterm_weight : ℕ := 4
  let final_weight : ℕ := 4
  calculate_semester_grade extracurricular_score midterm_score final_score
    extracurricular_weight midterm_weight final_weight = 89 := by
  sorry


end xiaoming_pe_grade_l1086_108627


namespace twelve_foldable_configurations_l1086_108618

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

end twelve_foldable_configurations_l1086_108618


namespace basketball_probabilities_l1086_108644

/-- A series of 6 independent Bernoulli trials with probability of success 1/3 -/
def bernoulli_trials (n : ℕ) (p : ℝ) := n = 6 ∧ p = 1/3

/-- Probability of two failures before the first success -/
def prob_two_failures_before_success (p : ℝ) : ℝ := (1 - p)^2 * p

/-- Probability of exactly k successes in n trials -/
def prob_exactly_k_successes (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p^k * (1 - p)^(n - k)

/-- Expected number of successes -/
def expected_successes (n : ℕ) (p : ℝ) : ℝ := n * p

/-- Variance of the number of successes -/
def variance_successes (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

theorem basketball_probabilities (n : ℕ) (p : ℝ) 
  (h : bernoulli_trials n p) : 
  prob_two_failures_before_success p = 4/27 ∧
  prob_exactly_k_successes n 3 p = 160/729 ∧
  expected_successes n p = 2 ∧
  variance_successes n p = 4/3 := by
  sorry

end basketball_probabilities_l1086_108644


namespace log_ratio_squared_l1086_108637

theorem log_ratio_squared (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ x y : ℝ, x = Real.log a ∧ y = Real.log b ∧ 2 * x^2 - 4 * x + 1 = 0 ∧ 2 * y^2 - 4 * y + 1 = 0) →
  (Real.log (a / b))^2 = 2 := by
sorry

end log_ratio_squared_l1086_108637


namespace pure_imaginary_iff_x_eq_one_l1086_108639

theorem pure_imaginary_iff_x_eq_one (x : ℝ) :
  x = 1 ↔ (Complex.mk (x^2 - 1) (x + 1)).im ≠ 0 ∧ (Complex.mk (x^2 - 1) (x + 1)).re = 0 :=
sorry

end pure_imaginary_iff_x_eq_one_l1086_108639


namespace cherries_eaten_l1086_108617

theorem cherries_eaten (initial : ℝ) (remaining : ℝ) (eaten : ℝ)
  (h1 : initial = 67.5)
  (h2 : remaining = 42.25)
  (h3 : eaten = initial - remaining) :
  eaten = 25.25 := by sorry

end cherries_eaten_l1086_108617


namespace factorization_of_2x_squared_minus_2_l1086_108612

theorem factorization_of_2x_squared_minus_2 (x : ℝ) : 2 * x^2 - 2 = 2 * (x - 1) * (x + 1) := by
  sorry

end factorization_of_2x_squared_minus_2_l1086_108612


namespace greatest_integer_for_all_real_domain_l1086_108672

theorem greatest_integer_for_all_real_domain : 
  ∃ (b : ℤ), (∀ (x : ℝ), (x^2 + b*x + 10 ≠ 0)) ∧ 
  (∀ (c : ℤ), c > b → ∃ (x : ℝ), x^2 + c*x + 10 = 0) ∧ 
  b = 6 := by
  sorry

end greatest_integer_for_all_real_domain_l1086_108672


namespace tan_and_trig_identity_l1086_108693

open Real

theorem tan_and_trig_identity (α : ℝ) (h : tan (α + π/4) = 1/3) : 
  tan α = -1/2 ∧ 
  2 * sin α ^ 2 - sin (π - α) * sin (π/2 - α) + sin (3*π/2 + α) ^ 2 = 8/5 := by
  sorry

end tan_and_trig_identity_l1086_108693


namespace eighth_pitch_frequency_l1086_108620

/-- Twelve-tone Equal Temperament system -/
structure TwelveToneEqualTemperament where
  /-- The frequency ratio between consecutive pitches -/
  ratio : ℝ
  /-- The ratio is the twelfth root of 2 -/
  ratio_def : ratio = Real.rpow 2 (1/12)

/-- The frequency of a pitch in the Twelve-tone Equal Temperament system -/
def frequency (system : TwelveToneEqualTemperament) (first_pitch : ℝ) (n : ℕ) : ℝ :=
  first_pitch * (system.ratio ^ (n - 1))

/-- Theorem: The frequency of the eighth pitch is the seventh root of 2 times the first pitch -/
theorem eighth_pitch_frequency (system : TwelveToneEqualTemperament) (f : ℝ) :
  frequency system f 8 = f * Real.rpow 2 (7/12) := by
  sorry

end eighth_pitch_frequency_l1086_108620


namespace sequence_sum_l1086_108691

theorem sequence_sum (n : ℕ) (s : ℕ → ℕ) : n = 2010 →
  (∀ i, i ∈ Finset.range (n - 1) → s (i + 1) = s i + 1) →
  (Finset.sum (Finset.range n) s = 5307) →
  (Finset.sum (Finset.range 1005) (fun i => s (2 * i))) = 2151 := by
  sorry

end sequence_sum_l1086_108691


namespace tan_alpha_value_l1086_108696

theorem tan_alpha_value (α : ℝ) (h : Real.tan (α - π/4) = 1/5) : 
  Real.tan α = 3/2 := by
sorry

end tan_alpha_value_l1086_108696


namespace combination_equality_l1086_108640

theorem combination_equality (x : ℕ) : 
  (Nat.choose 7 x = Nat.choose 6 5 + Nat.choose 6 4) → (x = 2 ∨ x = 5) := by
  sorry

end combination_equality_l1086_108640


namespace total_blue_balloons_l1086_108648

/-- The number of blue balloons Joan and Melanie have in total -/
def total_balloons (joan_balloons melanie_balloons : ℕ) : ℕ :=
  joan_balloons + melanie_balloons

/-- Theorem stating that Joan and Melanie have 81 blue balloons in total -/
theorem total_blue_balloons :
  total_balloons 40 41 = 81 := by
  sorry

end total_blue_balloons_l1086_108648


namespace tangent_line_ratio_l1086_108610

theorem tangent_line_ratio (x₁ x₂ : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0) : 
  (∃ (m b : ℝ), 
    (∀ x, m * x + b = x^2 ↔ x = x₁) ∧ 
    (∀ x, m * x + b = x^3 ↔ x = x₂)) → 
  x₁ / x₂ = 4 / 3 := by
sorry

end tangent_line_ratio_l1086_108610


namespace smallest_solution_abs_equation_l1086_108645

theorem smallest_solution_abs_equation :
  ∃ (x : ℝ), x * |x| = 3 * x + 2 ∧
  (∀ (y : ℝ), y * |y| = 3 * y + 2 → x ≤ y) ∧
  x = -2 := by
  sorry

end smallest_solution_abs_equation_l1086_108645


namespace football_team_ratio_l1086_108654

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

end football_team_ratio_l1086_108654


namespace most_likely_outcome_l1086_108684

/-- The probability of a child being a girl -/
def p_girl : ℚ := 3/5

/-- The probability of a child being a boy -/
def p_boy : ℚ := 2/5

/-- The number of children born -/
def n : ℕ := 3

/-- The probability of having 2 girls and 1 boy out of 3 children -/
def p_two_girls_one_boy : ℚ := 54/125

theorem most_likely_outcome :
  p_two_girls_one_boy = Nat.choose n 2 * p_girl^2 * p_boy ∧
  p_two_girls_one_boy > p_boy^n ∧
  p_two_girls_one_boy > p_girl^n ∧
  p_two_girls_one_boy > Nat.choose n 1 * p_girl * p_boy^2 :=
by sorry

end most_likely_outcome_l1086_108684


namespace intersection_M_N_l1086_108658

def M : Set ℝ := {x | |x| ≤ 2}
def N : Set ℝ := {x | x^2 + 2*x - 3 ≤ 0}

theorem intersection_M_N : M ∩ N = {x : ℝ | -2 ≤ x ∧ x ≤ 1} := by sorry

end intersection_M_N_l1086_108658


namespace noah_total_capacity_l1086_108679

-- Define Ali's closet capacity
def ali_closet_capacity : ℕ := 200

-- Define the ratio of Noah's closet capacity to Ali's
def noah_closet_ratio : ℚ := 1 / 4

-- Define the number of Noah's closets
def noah_closet_count : ℕ := 2

-- Theorem statement
theorem noah_total_capacity :
  noah_closet_count * (noah_closet_ratio * ali_closet_capacity) = 100 := by
  sorry


end noah_total_capacity_l1086_108679


namespace segment_length_l1086_108674

theorem segment_length : Real.sqrt 157 = Real.sqrt ((8 - 2)^2 + (18 - 7)^2) := by sorry

end segment_length_l1086_108674


namespace incenter_distance_l1086_108669

-- Define the triangle PQR
def Triangle (P Q R : ℝ × ℝ) : Prop :=
  let pq := Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)
  let qr := Real.sqrt ((R.1 - Q.1)^2 + (R.2 - Q.2)^2)
  let rp := Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2)
  pq = 17 ∧ qr = 15 ∧ rp = 8

-- Define the incenter
def Incenter (P Q R J : ℝ × ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧
  (J.1 - P.1)^2 + (J.2 - P.2)^2 = r^2 ∧
  (J.1 - Q.1)^2 + (J.2 - Q.2)^2 = r^2 ∧
  (J.1 - R.1)^2 + (J.2 - R.2)^2 = r^2

-- Theorem statement
theorem incenter_distance (P Q R J : ℝ × ℝ) :
  Triangle P Q R → Incenter P Q R J →
  (J.1 - P.1)^2 + (J.2 - P.2)^2 = 34 := by
  sorry

end incenter_distance_l1086_108669


namespace function_identity_l1086_108697

-- Define the property that the function f must satisfy
def SatisfiesProperty (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 + f y) = y + (f x)^2

-- State the theorem
theorem function_identity {f : ℝ → ℝ} (h : SatisfiesProperty f) : 
  ∀ x : ℝ, f x = x := by sorry

end function_identity_l1086_108697


namespace farm_animals_l1086_108607

theorem farm_animals (horses cows : ℕ) : 
  horses = 6 * cows →  -- Initial ratio of horses to cows is 6:1
  (horses - 15) = 3 * (cows + 15) →  -- New ratio after transaction is 3:1
  (horses - 15) - (cows + 15) = 70 := by  -- Difference after transaction is 70
sorry

end farm_animals_l1086_108607


namespace complex_sum_theorem_l1086_108688

theorem complex_sum_theorem : 
  let Z₁ : ℂ := (1 - Complex.I) / (1 + Complex.I)
  let Z₂ : ℂ := (3 - Complex.I) * Complex.I
  Z₁ + Z₂ = 1 + 2 * Complex.I :=
by sorry

end complex_sum_theorem_l1086_108688


namespace bill_difference_l1086_108657

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

end bill_difference_l1086_108657


namespace halfway_fraction_l1086_108677

theorem halfway_fraction : (3 / 4 + 5 / 6) / 2 = 19 / 24 := by
  sorry

end halfway_fraction_l1086_108677


namespace sufficient_but_not_necessary_l1086_108602

theorem sufficient_but_not_necessary (p q : Prop) :
  (p ∧ q → p ∨ q) ∧ ¬(p ∨ q → p ∧ q) := by
  sorry

end sufficient_but_not_necessary_l1086_108602


namespace game_installation_time_ratio_l1086_108681

theorem game_installation_time_ratio :
  ∀ (install_time : ℝ),
    install_time > 0 →
    10 + install_time + 3 * (10 + install_time) = 60 →
    install_time / 10 = 1 / 2 := by
  sorry

end game_installation_time_ratio_l1086_108681


namespace power_function_property_l1086_108624

/-- A power function is a function of the form f(x) = x^n for some real number n. -/
def PowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ n : ℝ, ∀ x : ℝ, f x = x ^ n

/-- A function lies in the first and third quadrants if it's positive for positive x
    and negative for negative x. -/
def LiesInFirstAndThirdQuadrants (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, (x > 0 → f x > 0) ∧ (x < 0 → f x < 0)

theorem power_function_property
  (f : ℝ → ℝ)
  (h_power : PowerFunction f)
  (h_quadrants : LiesInFirstAndThirdQuadrants f)
  (h_inequality : f 3 < f 2) :
  f (-3) > f (-2) := by
  sorry

end power_function_property_l1086_108624


namespace line_slope_problem_l1086_108608

theorem line_slope_problem (n : ℝ) : 
  n > 0 → 
  (n - 5) / (2 - n) = 2 * n → 
  n = 2.5 := by
sorry

end line_slope_problem_l1086_108608


namespace cricket_average_increase_l1086_108623

/-- Proves that the increase in average runs per innings is 5 -/
theorem cricket_average_increase
  (initial_average : ℝ)
  (initial_innings : ℕ)
  (next_innings_runs : ℝ)
  (h1 : initial_average = 32)
  (h2 : initial_innings = 20)
  (h3 : next_innings_runs = 137) :
  let total_runs := initial_average * initial_innings
  let new_innings := initial_innings + 1
  let new_total_runs := total_runs + next_innings_runs
  let new_average := new_total_runs / new_innings
  new_average - initial_average = 5 := by
sorry

end cricket_average_increase_l1086_108623


namespace difference_of_max_min_F_l1086_108629

-- Define the function F
def F (x y : ℝ) : ℝ := 4 * x + y

-- State the theorem
theorem difference_of_max_min_F :
  ∀ x y : ℝ, x > 0 → y > 0 → 4 * x + 1 / x + y + 9 / y = 26 →
  (∃ (max min : ℝ), (∀ x' y' : ℝ, x' > 0 → y' > 0 → 4 * x' + 1 / x' + y' + 9 / y' = 26 → F x' y' ≤ max) ∧
                    (∀ x' y' : ℝ, x' > 0 → y' > 0 → 4 * x' + 1 / x' + y' + 9 / y' = 26 → F x' y' ≥ min) ∧
                    (max - min = 24)) :=
by sorry

end difference_of_max_min_F_l1086_108629


namespace arithmetic_sequence_formula_l1086_108671

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_formula 
  (a : ℕ → ℤ) 
  (h_arith : arithmetic_sequence a) 
  (h_a1 : a 1 = 39) 
  (h_sum : a 1 + a 3 = 74) : 
  ∀ n : ℕ, a n = -2 * n + 41 := by
sorry

end arithmetic_sequence_formula_l1086_108671


namespace greatest_integer_fraction_l1086_108694

theorem greatest_integer_fraction (x : ℤ) : 
  (∃ k : ℤ, (x^2 + 4*x + 9) / (x - 4) = k) → x ≤ 5 :=
by sorry

end greatest_integer_fraction_l1086_108694
