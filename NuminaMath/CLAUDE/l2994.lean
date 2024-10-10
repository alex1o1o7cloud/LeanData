import Mathlib

namespace vector_problem_l2994_299431

/-- Given two collinear vectors a and b in ℝ², with b = (1, -2) and a ⋅ b = -10,
    prove that a = (-2, 4) and |a + c| = 5 where c = (6, -7) -/
theorem vector_problem (a b c : ℝ × ℝ) : 
  (∃ (k : ℝ), a = k • b) →  -- a and b are collinear
  b = (1, -2) → 
  a.1 * b.1 + a.2 * b.2 = -10 →  -- dot product
  c = (6, -7) → 
  a = (-2, 4) ∧ 
  Real.sqrt ((a.1 + c.1)^2 + (a.2 + c.2)^2) = 5 := by
sorry

end vector_problem_l2994_299431


namespace monotonic_decreasing_range_l2994_299488

def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*m*x + 1

theorem monotonic_decreasing_range (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ < 1 → f m x₁ > f m x₂) →
  m ≥ 1 :=
by
  sorry

end monotonic_decreasing_range_l2994_299488


namespace arithmetic_sequence_cosine_l2994_299487

/-- Given an arithmetic sequence {a_n} where a₁ + a₅ + a₉ = 8π, 
    prove that cos(a₃ + a₇) = -1/2 -/
theorem arithmetic_sequence_cosine (a : ℕ → ℝ) 
    (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
    (h_sum : a 1 + a 5 + a 9 = 8 * Real.pi) :
    Real.cos (a 3 + a 7) = -1/2 := by
  sorry

end arithmetic_sequence_cosine_l2994_299487


namespace expression_equals_49_l2994_299404

theorem expression_equals_49 (x : ℝ) : (x + 2)^2 + 2*(x + 2)*(5 - x) + (5 - x)^2 = 49 := by
  sorry

end expression_equals_49_l2994_299404


namespace reciprocal_inequality_l2994_299413

theorem reciprocal_inequality (a b : ℝ) :
  (∀ a b, b < a ∧ a < 0 → 1/b > 1/a) ∧
  (∃ a b, 1/b > 1/a ∧ ¬(b < a ∧ a < 0)) :=
sorry

end reciprocal_inequality_l2994_299413


namespace largest_valid_number_l2994_299408

def is_valid_number (a b c d e : Nat) : Prop :=
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧
  d > e ∧
  c > d + e ∧
  b > c + d + e ∧
  a > b + c + d + e

def number_value (a b c d e : Nat) : Nat :=
  a * 10000 + b * 1000 + c * 100 + d * 10 + e

theorem largest_valid_number :
  ∀ a b c d e : Nat,
    is_valid_number a b c d e →
    number_value a b c d e ≤ 95210 :=
by sorry

end largest_valid_number_l2994_299408


namespace intersection_point_expression_value_l2994_299433

/-- Given a point P(a,b) at the intersection of y=x-2 and y=1/x,
    prove that (a-a²/(a+b)) ÷ (a²b²/(a²-b²)) equals 2 -/
theorem intersection_point_expression_value (a b : ℝ) 
  (h1 : b = a - 2)
  (h2 : b = 1 / a)
  (h3 : a ≠ 0)
  (h4 : b ≠ 0)
  (h5 : a ≠ b)
  (h6 : a + b ≠ 0) :
  (a - a^2 / (a + b)) / (a^2 * b^2 / (a^2 - b^2)) = 2 :=
by sorry

end intersection_point_expression_value_l2994_299433


namespace wedge_volume_l2994_299412

/-- The volume of a wedge cut from a cylindrical log --/
theorem wedge_volume (d : ℝ) (angle : ℝ) : 
  d = 20 →
  angle = 60 →
  (1 / 6) * d^3 * π = 667 * π := by
  sorry

end wedge_volume_l2994_299412


namespace camp_distribution_correct_l2994_299442

/-- Represents a summer camp with three sub-camps -/
structure SummerCamp where
  totalStudents : Nat
  sampleSize : Nat
  firstDrawn : Nat
  campIEnd : Nat
  campIIEnd : Nat

/-- Calculates the number of students drawn from each camp -/
def campDistribution (camp : SummerCamp) : (Nat × Nat × Nat) :=
  sorry

/-- Theorem stating the correct distribution of sampled students across camps -/
theorem camp_distribution_correct (camp : SummerCamp) 
  (h1 : camp.totalStudents = 720)
  (h2 : camp.sampleSize = 60)
  (h3 : camp.firstDrawn = 4)
  (h4 : camp.campIEnd = 360)
  (h5 : camp.campIIEnd = 640) :
  campDistribution camp = (30, 24, 6) := by
  sorry

end camp_distribution_correct_l2994_299442


namespace division_of_decimals_l2994_299405

theorem division_of_decimals : (0.045 : ℚ) / (0.009 : ℚ) = 5 := by sorry

end division_of_decimals_l2994_299405


namespace prob_two_dice_show_two_is_15_64_l2994_299419

/-- The number of sides on each die -/
def num_sides : ℕ := 8

/-- The probability of at least one of two fair n-sided dice showing a specific number -/
def prob_at_least_one (n : ℕ) : ℚ :=
  1 - (n - 1)^2 / n^2

/-- The probability of at least one of two fair 8-sided dice showing a 2 -/
def prob_two_dice_show_two : ℚ := prob_at_least_one num_sides

theorem prob_two_dice_show_two_is_15_64 : 
  prob_two_dice_show_two = 15 / 64 := by
  sorry

end prob_two_dice_show_two_is_15_64_l2994_299419


namespace hyperbola_foci_distance_l2994_299498

/-- The distance between the foci of a hyperbola given by the equation 3x^2 - 18x - 2y^2 - 4y = 48 -/
theorem hyperbola_foci_distance : 
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, 3 * x^2 - 18 * x - 2 * y^2 - 4 * y = 48) →
    (a^2 = 53 / 3) →
    (b^2 = 53 / 6) →
    (c^2 = a^2 + b^2) →
    (2 * c = 2 * Real.sqrt (53 / 2)) :=
by sorry

end hyperbola_foci_distance_l2994_299498


namespace no_base_square_l2994_299470

theorem no_base_square (b : ℕ) : b > 1 → ¬∃ (n : ℕ), 2 * b^2 + 3 * b + 2 = n^2 := by
  sorry

end no_base_square_l2994_299470


namespace exam_average_score_l2994_299499

/-- Given an exam with a maximum score and the percentages scored by three students,
    calculate the average mark scored by all three students. -/
theorem exam_average_score (max_score : ℕ) (amar_percent bhavan_percent chetan_percent : ℕ) :
  max_score = 900 ∧ amar_percent = 64 ∧ bhavan_percent = 36 ∧ chetan_percent = 44 →
  (amar_percent * max_score / 100 + bhavan_percent * max_score / 100 + chetan_percent * max_score / 100) / 3 = 432 :=
by sorry

end exam_average_score_l2994_299499


namespace student_weight_l2994_299427

theorem student_weight (student_weight sister_weight : ℝ) 
  (h1 : student_weight - 5 = 2 * sister_weight)
  (h2 : student_weight + sister_weight = 110) :
  student_weight = 75 := by
sorry

end student_weight_l2994_299427


namespace probability_three_heads_seven_tosses_prove_probability_three_heads_seven_tosses_l2994_299403

/-- The probability of getting exactly 3 heads in 7 fair coin tosses -/
theorem probability_three_heads_seven_tosses : ℚ :=
  35 / 128

/-- Prove that the probability of getting exactly 3 heads in 7 fair coin tosses is 35/128 -/
theorem prove_probability_three_heads_seven_tosses :
  probability_three_heads_seven_tosses = 35 / 128 := by
  sorry

end probability_three_heads_seven_tosses_prove_probability_three_heads_seven_tosses_l2994_299403


namespace apple_distribution_l2994_299466

theorem apple_distribution (total_apples : ℕ) (red_percentage : ℚ) (classmates : ℕ) (extra_red : ℕ) : 
  total_apples = 80 →
  red_percentage = 3/5 →
  classmates = 6 →
  extra_red = 3 →
  (↑(total_apples) * red_percentage - extra_red) / classmates = 7.5 →
  ∃ (apples_per_classmate : ℕ), apples_per_classmate = 7 ∧ 
    apples_per_classmate * classmates ≤ ↑(total_apples) * red_percentage - extra_red ∧
    (apples_per_classmate + 1) * classmates > ↑(total_apples) * red_percentage - extra_red :=
by sorry

end apple_distribution_l2994_299466


namespace pet_store_earnings_l2994_299467

theorem pet_store_earnings :
  let num_kittens : ℕ := 2
  let num_puppies : ℕ := 1
  let kitten_price : ℕ := 6
  let puppy_price : ℕ := 5
  (num_kittens * kitten_price + num_puppies * puppy_price : ℕ) = 17 := by
  sorry

end pet_store_earnings_l2994_299467


namespace least_possible_third_side_l2994_299478

theorem least_possible_third_side (a b : ℝ) (ha : a = 7) (hb : b = 24) :
  let c := Real.sqrt (b^2 - a^2)
  c = Real.sqrt 527 ∧ c ≤ a ∧ c ≤ b := by sorry

end least_possible_third_side_l2994_299478


namespace gym_distance_proof_l2994_299453

/-- The distance between Wang Lei's home and the gym --/
def gym_distance : ℕ := 1500

/-- Wang Lei's walking speed in meters per minute --/
def wang_lei_speed : ℕ := 40

/-- The older sister's walking speed in meters per minute --/
def older_sister_speed : ℕ := wang_lei_speed + 20

/-- Time taken by the older sister to reach the gym in minutes --/
def time_to_gym : ℕ := 25

/-- Distance from the meeting point to the gym in meters --/
def meeting_point_distance : ℕ := 300

theorem gym_distance_proof :
  gym_distance = older_sister_speed * time_to_gym ∧
  gym_distance = wang_lei_speed * (time_to_gym + meeting_point_distance / wang_lei_speed) :=
by sorry

end gym_distance_proof_l2994_299453


namespace root_sum_reciprocal_product_l2994_299429

theorem root_sum_reciprocal_product (p q r s : ℂ) : 
  (p^4 + 6*p^3 + 13*p^2 + 7*p + 3 = 0) →
  (q^4 + 6*q^3 + 13*q^2 + 7*q + 3 = 0) →
  (r^4 + 6*r^3 + 13*r^2 + 7*r + 3 = 0) →
  (s^4 + 6*s^3 + 13*s^2 + 7*s + 3 = 0) →
  (1 / (p*q*r) + 1 / (p*q*s) + 1 / (p*r*s) + 1 / (q*r*s) = -2) :=
by sorry

end root_sum_reciprocal_product_l2994_299429


namespace christinas_total_distance_l2994_299430

/-- The total distance Christina walks in a week given her routine -/
def christinas_weekly_distance (school_distance : ℕ) (friend_distance : ℕ) : ℕ :=
  (school_distance * 2 * 4) + (school_distance * 2 + friend_distance * 2)

/-- Theorem stating that Christina's total distance for the week is 74km -/
theorem christinas_total_distance :
  christinas_weekly_distance 7 2 = 74 := by
  sorry

end christinas_total_distance_l2994_299430


namespace min_value_expression_l2994_299441

theorem min_value_expression (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
  (5 * r) / (3 * p + 2 * q) + (5 * p) / (2 * q + 3 * r) + (2 * q) / (p + r) ≥ 4 ∧
  ((5 * r) / (3 * p + 2 * q) + (5 * p) / (2 * q + 3 * r) + (2 * q) / (p + r) = 4 ↔ 3 * p = 2 * q ∧ 2 * q = 3 * r) :=
by sorry

end min_value_expression_l2994_299441


namespace checkerboard_squares_l2994_299415

/-- Represents a square on the checkerboard -/
structure Square where
  size : Nat
  row : Nat
  col : Nat

/-- The size of the checkerboard -/
def boardSize : Nat := 10

/-- Checks if a square is valid on the board -/
def isValidSquare (s : Square) : Bool :=
  s.size > 0 && s.size <= boardSize && s.row + s.size <= boardSize && s.col + s.size <= boardSize

/-- Counts the number of black squares in a given square -/
def countBlackSquares (s : Square) : Nat :=
  sorry

/-- Counts the number of valid squares with at least 6 black squares -/
def countValidSquares : Nat :=
  sorry

theorem checkerboard_squares : countValidSquares = 155 := by
  sorry

end checkerboard_squares_l2994_299415


namespace decagon_diagonal_intersections_l2994_299424

/-- The number of vertices in a regular decagon -/
def n : ℕ := 10

/-- The number of distinct intersection points of diagonals in the interior of a regular decagon -/
def intersection_points (n : ℕ) : ℕ := Nat.choose n 4

theorem decagon_diagonal_intersections :
  intersection_points n = 210 :=
sorry

end decagon_diagonal_intersections_l2994_299424


namespace root_equation_problem_l2994_299420

theorem root_equation_problem (p q : ℝ) 
  (h1 : ∃! (r1 r2 r3 : ℝ), r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧
    (∀ x : ℝ, (x + p) * (x + q) * (x + 15) = 0 ↔ x = r1 ∨ x = r2 ∨ x = r3) ∧
    (∀ x : ℝ, x ≠ -4))
  (h2 : ∃! (s1 s2 : ℝ), s1 ≠ s2 ∧
    (∀ x : ℝ, (x + 2*p) * (x + 4) * (x + 9) = 0 ↔ x = s1 ∨ x = s2) ∧
    (∀ x : ℝ, x ≠ -q ∧ x ≠ -15)) :
  100 * p + q = -191 := by
sorry

end root_equation_problem_l2994_299420


namespace parallelogram_diagonal_intersection_l2994_299472

/-- Given a parallelogram with opposite vertices at (2, -4) and (14, 10),
    the coordinates of the point where the diagonals intersect are (8, 3). -/
theorem parallelogram_diagonal_intersection :
  let v1 : ℝ × ℝ := (2, -4)
  let v2 : ℝ × ℝ := (14, 10)
  let midpoint : ℝ × ℝ := ((v1.1 + v2.1) / 2, (v1.2 + v2.2) / 2)
  midpoint = (8, 3) := by sorry

end parallelogram_diagonal_intersection_l2994_299472


namespace hyperbola_eccentricity_range_l2994_299402

/-- The eccentricity of a hyperbola with the given conditions is between 1 and 2 -/
theorem hyperbola_eccentricity_range (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  ∃ (x y e : ℝ),
    x^2 / a^2 - y^2 / b^2 = 1 ∧
    x ≥ a ∧
    ∃ (f1 f2 : ℝ × ℝ),
      (∀ (p : ℝ × ℝ), p.1^2 / a^2 - p.2^2 / b^2 = 1 →
        |p.1 - f1.1| - |p.1 - f2.1| = 2 * a * e) ∧
      |x - f1.1| = 3 * |x - f2.1| →
    1 < e ∧ e ≤ 2 :=
sorry

end hyperbola_eccentricity_range_l2994_299402


namespace bell_pepper_slices_l2994_299461

theorem bell_pepper_slices (num_peppers : ℕ) (slices_per_pepper : ℕ) (smaller_pieces : ℕ) : 
  num_peppers = 5 →
  slices_per_pepper = 20 →
  smaller_pieces = 3 →
  let total_slices := num_peppers * slices_per_pepper
  let large_slices := total_slices / 2
  let small_pieces := large_slices * smaller_pieces
  total_slices - large_slices + small_pieces = 200 := by
  sorry

end bell_pepper_slices_l2994_299461


namespace singer_hourly_rate_l2994_299425

/-- Given a singer hired for 3 hours, with a 20% tip, and a total payment of $54, 
    the hourly rate for the singer is $15. -/
theorem singer_hourly_rate (hours : ℕ) (tip_percentage : ℚ) (total_payment : ℚ) :
  hours = 3 →
  tip_percentage = 1/5 →
  total_payment = 54 →
  ∃ (hourly_rate : ℚ), 
    hourly_rate * hours * (1 + tip_percentage) = total_payment ∧
    hourly_rate = 15 :=
by sorry

end singer_hourly_rate_l2994_299425


namespace certain_number_value_l2994_299484

theorem certain_number_value : ∀ (t k certain_number : ℝ),
  t = 5 / 9 * (k - certain_number) →
  t = 75 →
  k = 167 →
  certain_number = 32 := by
sorry

end certain_number_value_l2994_299484


namespace binomial_expansion_fourth_fifth_terms_sum_zero_l2994_299449

/-- Given a binomial expansion (a-b)^n where n ≥ 2, ab ≠ 0, and a = mb with m = k + 2 and k a positive integer,
    prove that n = 2m + 3 makes the sum of the fourth and fifth terms zero. -/
theorem binomial_expansion_fourth_fifth_terms_sum_zero 
  (n : ℕ) (a b : ℝ) (m k : ℕ) :
  n ≥ 2 →
  a ≠ 0 →
  b ≠ 0 →
  k > 0 →
  m = k + 2 →
  a = m * b →
  (n = 2 * m + 3 ↔ 
    (Nat.choose n 3) * (a - b)^(n - 3) * b^3 + 
    (Nat.choose n 4) * (a - b)^(n - 4) * b^4 = 0) :=
by sorry

end binomial_expansion_fourth_fifth_terms_sum_zero_l2994_299449


namespace circle_roll_path_length_l2994_299459

/-- The total path length of a point on a circle rolling without slipping -/
theorem circle_roll_path_length
  (r : ℝ) -- radius of the circle
  (θ_flat : ℝ) -- angle rolled on flat surface in radians
  (θ_slope : ℝ) -- angle rolled on slope in radians
  (h_radius : r = 4 / Real.pi)
  (h_flat : θ_flat = 3 * Real.pi / 2)
  (h_slope : θ_slope = Real.pi / 2)
  (h_total : θ_flat + θ_slope = 2 * Real.pi) :
  2 * Real.pi * r = 8 :=
sorry

end circle_roll_path_length_l2994_299459


namespace smallest_non_prime_non_square_no_small_factors_l2994_299450

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

def has_no_prime_factor_less_than (n k : ℕ) : Prop :=
  ∀ m : ℕ, m < k → is_prime m → ¬(n % m = 0)

theorem smallest_non_prime_non_square_no_small_factors :
  ∀ n : ℕ, n > 0 →
    (¬is_prime n ∧ ¬is_perfect_square n ∧ has_no_prime_factor_less_than n 70) →
    n ≥ 5183 :=
sorry

end smallest_non_prime_non_square_no_small_factors_l2994_299450


namespace median_salary_is_40000_l2994_299417

/-- Represents a position in the company with its title, number of employees, and salary. -/
structure Position where
  title : String
  count : Nat
  salary : Nat

/-- The list of positions in the company. -/
def positions : List Position := [
  ⟨"President", 1, 160000⟩,
  ⟨"Vice-President", 4, 105000⟩,
  ⟨"Director", 15, 80000⟩,
  ⟨"Associate Director", 10, 55000⟩,
  ⟨"Senior Manager", 20, 40000⟩,
  ⟨"Administrative Specialist", 50, 28000⟩
]

/-- The total number of employees in the company. -/
def totalEmployees : Nat := 100

/-- Calculates the median salary of the employees. -/
def medianSalary (pos : List Position) (total : Nat) : Nat :=
  sorry

/-- Theorem stating that the median salary is $40,000. -/
theorem median_salary_is_40000 :
  medianSalary positions totalEmployees = 40000 := by
  sorry

end median_salary_is_40000_l2994_299417


namespace range_of_m_l2994_299482

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ < 0 ∧ x₂ < 0 ∧ x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0

def q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

-- Define the theorem
theorem range_of_m :
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m) → m ∈ Set.Ioo 1 2 ∪ Set.Ici 3 :=
sorry

end range_of_m_l2994_299482


namespace percentage_not_sold_l2994_299418

def initial_stock : ℕ := 620
def monday_sales : ℕ := 50
def tuesday_sales : ℕ := 82
def wednesday_sales : ℕ := 60
def thursday_sales : ℕ := 48
def friday_sales : ℕ := 40

theorem percentage_not_sold (initial_stock monday_sales tuesday_sales wednesday_sales thursday_sales friday_sales : ℕ) :
  (initial_stock - (monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales)) / initial_stock * 100 =
  (620 - (50 + 82 + 60 + 48 + 40)) / 620 * 100 :=
by sorry

end percentage_not_sold_l2994_299418


namespace commercial_time_calculation_l2994_299448

theorem commercial_time_calculation (num_programs : ℕ) (program_duration : ℕ) (commercial_fraction : ℚ) : 
  num_programs = 6 → 
  program_duration = 30 → 
  commercial_fraction = 1/4 → 
  (↑num_programs * ↑program_duration : ℚ) * commercial_fraction = 45 := by
  sorry

end commercial_time_calculation_l2994_299448


namespace halfway_fraction_l2994_299444

theorem halfway_fraction : ∃ (n d : ℕ), d ≠ 0 ∧ (n : ℚ) / d = (1 : ℚ) / 3 / 2 + (3 : ℚ) / 4 / 2 ∧ n = 13 ∧ d = 24 := by
  sorry

end halfway_fraction_l2994_299444


namespace customer_difference_l2994_299421

theorem customer_difference (initial : ℕ) (remaining : ℕ) 
  (h1 : initial = 19) (h2 : remaining = 4) : 
  initial - remaining = 15 := by
  sorry

end customer_difference_l2994_299421


namespace line_points_k_value_l2994_299494

/-- Given a line containing the points (-1, 6), (6, k), and (20, 3), prove that k = 5 -/
theorem line_points_k_value :
  ∀ k : ℝ,
  (∃ m b : ℝ,
    (m * (-1) + b = 6) ∧
    (m * 6 + b = k) ∧
    (m * 20 + b = 3)) →
  k = 5 :=
by sorry

end line_points_k_value_l2994_299494


namespace min_value_theorem_l2994_299437

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a + b = 1 →
  (∀ x y : ℝ, 0 < x → x < 2 → y = 1 + Real.sin (π * x) → a * x + b * y = 1) →
  (1 / a + 2 / b) ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end min_value_theorem_l2994_299437


namespace jessie_current_weight_l2994_299471

def initial_weight : ℝ := 69
def weight_lost : ℝ := 35

theorem jessie_current_weight : 
  initial_weight - weight_lost = 34 := by sorry

end jessie_current_weight_l2994_299471


namespace probability_same_length_segments_l2994_299435

def regular_pentagon_segments : Finset ℕ := sorry

theorem probability_same_length_segments :
  let S := regular_pentagon_segments
  let total_segments := S.card
  let same_type_segments := (total_segments / 2) - 1
  (same_type_segments : ℚ) / ((total_segments - 1) : ℚ) = 4 / 9 := by sorry

end probability_same_length_segments_l2994_299435


namespace cone_slant_height_l2994_299445

/-- Given a cone with base radius 2 cm and an unfolded side forming a sector
    with central angle 120°, prove that its slant height is 6 cm. -/
theorem cone_slant_height (r : ℝ) (θ : ℝ) (x : ℝ) 
    (h_r : r = 2)
    (h_θ : θ = 120)
    (h_arc_length : θ / 360 * (2 * Real.pi * x) = 2 * Real.pi * r) :
  x = 6 := by
  sorry

end cone_slant_height_l2994_299445


namespace pages_read_per_year_l2994_299483

/-- The number of pages read in a year given monthly reading habits and book lengths -/
theorem pages_read_per_year
  (novels_per_month : ℕ)
  (pages_per_novel : ℕ)
  (months_per_year : ℕ)
  (h1 : novels_per_month = 4)
  (h2 : pages_per_novel = 200)
  (h3 : months_per_year = 12) :
  novels_per_month * pages_per_novel * months_per_year = 9600 :=
by sorry

end pages_read_per_year_l2994_299483


namespace inequality_proof_l2994_299452

theorem inequality_proof (x y : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : x^2 + y^2 ≤ 1) :
  |x^2 + 2*x*y - y^2| ≤ Real.sqrt 2 := by
sorry

end inequality_proof_l2994_299452


namespace journey_speed_calculation_l2994_299434

/-- Prove that given a journey of 225 km completed in 10 hours, 
    where the first half is traveled at 21 km/hr, 
    the speed for the second half of the journey is approximately 24.23 km/hr. -/
theorem journey_speed_calculation 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (first_half_speed : ℝ) 
  (h1 : total_distance = 225) 
  (h2 : total_time = 10) 
  (h3 : first_half_speed = 21) : 
  ∃ (second_half_speed : ℝ), 
    (abs (second_half_speed - 24.23) < 0.01) ∧ 
    (total_distance / 2 / first_half_speed + total_distance / 2 / second_half_speed = total_time) :=
by sorry

end journey_speed_calculation_l2994_299434


namespace sine_function_problem_l2994_299490

theorem sine_function_problem (a b c : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * Real.sin x + b * x + c
  (f 0 = -2) → (f (Real.pi / 2) = 1) → (f (-Real.pi / 2) = -5) := by
  sorry

end sine_function_problem_l2994_299490


namespace percentage_problem_l2994_299473

theorem percentage_problem (P : ℝ) : 
  (P * 100 + 60 = 100) → P = 0.4 := by
  sorry

end percentage_problem_l2994_299473


namespace quadratic_inequality_solution_l2994_299401

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, -1/2 < x ∧ x < 1/3 ↔ a * x^2 + b * x + 20 > 0) →
  a = -12 :=
by sorry

end quadratic_inequality_solution_l2994_299401


namespace closest_integer_to_cube_root_l2994_299491

theorem closest_integer_to_cube_root : ∃ (n : ℤ), 
  n = 8 ∧ ∀ (m : ℤ), |m - (5^3 + 7^3)^(1/3)| ≥ |n - (5^3 + 7^3)^(1/3)| := by
  sorry

end closest_integer_to_cube_root_l2994_299491


namespace line_points_determine_m_l2994_299423

-- Define the points on the line
def point1 : ℝ × ℝ := (7, 10)
def point2 : ℝ → ℝ × ℝ := λ m ↦ (-3, m)
def point3 : ℝ × ℝ := (-11, 5)

-- Define the condition that the points are collinear
def collinear (p q r : ℝ × ℝ) : Prop :=
  (q.2 - p.2) * (r.1 - q.1) = (r.2 - q.2) * (q.1 - p.1)

-- Theorem statement
theorem line_points_determine_m :
  collinear point1 (point2 m) point3 → m = 65 / 9 := by
  sorry

end line_points_determine_m_l2994_299423


namespace impossible_to_break_record_duke_game_impossible_l2994_299432

/-- Represents the constraints and conditions for Duke's basketball game --/
structure GameConstraints where
  old_record : ℕ
  points_to_tie : ℕ
  points_to_break : ℕ
  free_throws : ℕ
  regular_baskets : ℕ
  normal_three_pointers : ℕ
  max_attempts : ℕ

/-- Calculates the total points scored based on the number of each type of shot --/
def total_points (ft reg tp : ℕ) : ℕ :=
  ft + 2 * reg + 3 * tp

/-- Theorem stating that it's impossible to break the record under the given constraints --/
theorem impossible_to_break_record (gc : GameConstraints) : 
  ¬∃ (tp : ℕ), 
    total_points gc.free_throws gc.regular_baskets tp = gc.old_record + gc.points_to_tie + gc.points_to_break ∧
    gc.free_throws + gc.regular_baskets + tp ≤ gc.max_attempts :=
by
  sorry

/-- The specific game constraints for Duke's final game --/
def duke_game : GameConstraints :=
  { old_record := 257
  , points_to_tie := 17
  , points_to_break := 5
  , free_throws := 5
  , regular_baskets := 4
  , normal_three_pointers := 2
  , max_attempts := 10
  }

/-- Theorem applying the impossibility proof to Duke's specific game --/
theorem duke_game_impossible : 
  ¬∃ (tp : ℕ), 
    total_points duke_game.free_throws duke_game.regular_baskets tp = 
      duke_game.old_record + duke_game.points_to_tie + duke_game.points_to_break ∧
    duke_game.free_throws + duke_game.regular_baskets + tp ≤ duke_game.max_attempts :=
by
  apply impossible_to_break_record duke_game

end impossible_to_break_record_duke_game_impossible_l2994_299432


namespace man_money_problem_l2994_299481

theorem man_money_problem (x : ℝ) : 
  (((2 * (2 * (2 * (2 * x - 50) - 60) - 70) - 80) = 0) ↔ (x = 53.75)) := by
  sorry

end man_money_problem_l2994_299481


namespace workday_end_time_l2994_299469

-- Define the start time of the workday
def start_time : Nat := 8

-- Define the lunch break start time
def lunch_start : Nat := 13

-- Define the duration of the workday in hours (excluding lunch)
def workday_duration : Nat := 8

-- Define the duration of the lunch break in hours
def lunch_duration : Nat := 1

-- Theorem to prove the end time of the workday
theorem workday_end_time :
  start_time + workday_duration + lunch_duration = 17 := by
  sorry

#check workday_end_time

end workday_end_time_l2994_299469


namespace problem_solution_l2994_299457

theorem problem_solution (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_prod : x * y * z = 1)
  (h_eq1 : x + 1 / z = 6)
  (h_eq2 : y + 1 / x = 30) :
  z + 1 / y = 38 / 179 := by
sorry

end problem_solution_l2994_299457


namespace polynomial_remainder_remainder_theorem_cube_area_is_six_probability_half_tan_135_is_negative_one_l2994_299460

-- Problem 1
theorem polynomial_remainder : Int → Int := 
  fun x ↦ 2 * x^3 - 3 * x^2 + x - 1

theorem remainder_theorem (p : Int → Int) (a : Int) :
  p (-1) = -7 → ∃ q : Int → Int, ∀ x, p x = (x + 1) * q x + -7 := by sorry

-- Problem 2
def cube_surface_area (side_length : ℝ) : ℝ := 6 * side_length^2

theorem cube_area_is_six : cube_surface_area 1 = 6 := by sorry

-- Problem 3
def probability_white (red white : ℕ) : ℚ := white / (red + white)

theorem probability_half : probability_white 10 10 = 1/2 := by sorry

-- Problem 4
theorem tan_135_is_negative_one : Real.tan (135 * π / 180) = -1 := by sorry

end polynomial_remainder_remainder_theorem_cube_area_is_six_probability_half_tan_135_is_negative_one_l2994_299460


namespace total_red_pencils_l2994_299486

/-- The number of pencil packs bought -/
def total_packs : ℕ := 15

/-- The number of red pencils in a normal pack -/
def red_per_normal_pack : ℕ := 1

/-- The number of packs with extra red pencils -/
def packs_with_extra : ℕ := 3

/-- The number of extra red pencils in special packs -/
def extra_red_per_special_pack : ℕ := 2

/-- Theorem stating the total number of red pencils bought -/
theorem total_red_pencils : 
  total_packs * red_per_normal_pack + packs_with_extra * extra_red_per_special_pack = 21 :=
by
  sorry


end total_red_pencils_l2994_299486


namespace circle_center_radius_sum_l2994_299485

/-- Given a circle D with equation x^2 + 14x + y^2 - 8y = -64,
    prove that the sum of its center coordinates and radius is -2 -/
theorem circle_center_radius_sum :
  ∀ (c d s : ℝ),
  (∀ (x y : ℝ), x^2 + 14*x + y^2 - 8*y = -64 ↔ (x - c)^2 + (y - d)^2 = s^2) →
  c + d + s = -2 :=
by sorry

end circle_center_radius_sum_l2994_299485


namespace condition_relationship_l2994_299446

theorem condition_relationship (x : ℝ) :
  (∀ x, x > 2 → x^2 > 4) ∧ 
  (∃ x, x^2 > 4 ∧ ¬(x > 2)) :=
by sorry

end condition_relationship_l2994_299446


namespace solve_equation_l2994_299477

theorem solve_equation (x : ℚ) : 5 * (x - 10) = 3 * (3 - 3 * x) + 9 → x = 34 / 7 := by
  sorry

end solve_equation_l2994_299477


namespace angle_system_solution_l2994_299440

theorem angle_system_solution (k : ℤ) :
  let x : ℝ := π/3 + k*π
  let y : ℝ := k*π
  (x - y = π/3) ∧ (Real.tan x - Real.tan y = Real.sqrt 3) := by
  sorry

end angle_system_solution_l2994_299440


namespace distribute_four_balls_l2994_299411

/-- The number of ways to distribute n distinguishable balls into 2 indistinguishable boxes -/
def distribute_balls (n : ℕ) : ℕ :=
  (n + 1)

/-- The number of ways to distribute 4 distinguishable balls into 2 indistinguishable boxes is 8 -/
theorem distribute_four_balls : distribute_balls 4 = 8 := by
  sorry

end distribute_four_balls_l2994_299411


namespace complex_number_location_l2994_299439

theorem complex_number_location (Z : ℂ) : Z = Complex.I :=
  by
  -- Define Z
  have h1 : Z = (Real.sqrt 2 - Complex.I ^ 3) / (1 - Real.sqrt 2 * Complex.I) := by sorry
  
  -- Define properties of complex numbers
  have h2 : Complex.I ^ 2 = -1 := by sorry
  have h3 : Complex.I ^ 3 = -Complex.I := by sorry
  
  -- Proof steps would go here
  sorry

end complex_number_location_l2994_299439


namespace expression_value_l2994_299428

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (2 * x - 3 * y) - f (x + y) = -2 * x + 8 * y

/-- The main theorem stating that the given expression is always equal to 4 -/
theorem expression_value (f : ℝ → ℝ) (h : FunctionalEquation f) :
  ∀ t, f 4*t ≠ f 3*t → (f 5*t - f t) / (f 4*t - f 3*t) = 4 := by
  sorry

end expression_value_l2994_299428


namespace original_selling_price_with_loss_l2994_299493

-- Define the selling price with 10% gain
def selling_price_with_gain : ℝ := 660

-- Define the gain percentage
def gain_percentage : ℝ := 0.1

-- Define the loss percentage
def loss_percentage : ℝ := 0.1

-- Theorem to prove
theorem original_selling_price_with_loss :
  let cost_price := selling_price_with_gain / (1 + gain_percentage)
  let selling_price_with_loss := cost_price * (1 - loss_percentage)
  selling_price_with_loss = 540 := by sorry

end original_selling_price_with_loss_l2994_299493


namespace m_range_l2994_299410

-- Define the sets A and B
def A : Set ℝ := {x | x^2 < 16}
def B (m : ℝ) : Set ℝ := {x | x < m}

-- State the theorem
theorem m_range (m : ℝ) : A ∩ B m = A → m ≥ 4 := by
  sorry


end m_range_l2994_299410


namespace quadratic_equations_solutions_quadratic_equations_all_solutions_l2994_299479

theorem quadratic_equations_solutions :
  (∃ x : ℝ, x^2 - 2*x - 1 = 0) ∧
  (∃ x : ℝ, (x - 2)^2 = 2*x - 4) :=
by
  constructor
  · use 1 + Real.sqrt 2
    sorry
  · use 2
    sorry

theorem quadratic_equations_all_solutions :
  (∀ x : ℝ, x^2 - 2*x - 1 = 0 ↔ (x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2)) ∧
  (∀ x : ℝ, (x - 2)^2 = 2*x - 4 ↔ (x = 2 ∨ x = 4)) :=
by
  constructor
  · intro x
    sorry
  · intro x
    sorry

end quadratic_equations_solutions_quadratic_equations_all_solutions_l2994_299479


namespace infinitely_many_palindromes_l2994_299443

/-- Arithmetic progression term -/
def a (n : ℕ+) : ℕ := 18 + 19 * (n - 1)

/-- Repunit -/
def R (k : ℕ) : ℕ := (10^k - 1) / 9

/-- k values -/
def k (t : ℕ) : ℕ := 18 * t + 6

theorem infinitely_many_palindromes :
  ∀ m : ℕ, ∃ t : ℕ, t > m ∧ ∃ n : ℕ+, R (k t) = a n :=
sorry

end infinitely_many_palindromes_l2994_299443


namespace non_shaded_perimeter_l2994_299456

/-- Given a rectangle with dimensions and a shaded area, calculate the perimeter of the non-shaded region --/
theorem non_shaded_perimeter (large_width large_height ext_width ext_height shaded_area : ℝ) :
  large_width = 12 →
  large_height = 8 →
  ext_width = 5 →
  ext_height = 2 →
  shaded_area = 104 →
  let total_area := large_width * large_height + ext_width * ext_height
  let non_shaded_area := total_area - shaded_area
  let non_shaded_width := ext_height
  let non_shaded_height := non_shaded_area / non_shaded_width
  2 * (non_shaded_width + non_shaded_height) = 6 :=
by sorry

end non_shaded_perimeter_l2994_299456


namespace absolute_value_inequality_solution_l2994_299422

theorem absolute_value_inequality_solution :
  {x : ℝ | |2 - x| ≤ 1} = Set.Icc 1 3 := by sorry

end absolute_value_inequality_solution_l2994_299422


namespace recurring_decimal_to_fraction_l2994_299468

/-- Given that 0.overline{02} = 2/99, prove that 2.overline{06} = 68/33 -/
theorem recurring_decimal_to_fraction :
  (∃ (x : ℚ), x = 2 / 99 ∧ (∀ n : ℕ, (x * 10^(3*n) - ⌊x * 10^(3*n)⌋ = 0.02))) →
  (∃ (y : ℚ), y = 68 / 33 ∧ (∀ n : ℕ, (y - 2 - ⌊y - 2⌋ = 0.06))) :=
by sorry

end recurring_decimal_to_fraction_l2994_299468


namespace five_two_difference_in_book_pages_l2994_299414

/-- Count occurrences of a digit in a range of numbers -/
def countDigit (d : Nat) (start finish : Nat) : Nat :=
  sorry

/-- The difference between occurrences of 5 and 2 in page numbers -/
def diffFiveTwo (totalPages : Nat) : Int :=
  (countDigit 5 1 totalPages : Int) - (countDigit 2 1 totalPages : Int)

/-- Theorem stating the difference between 5's and 2's in a 625-page book -/
theorem five_two_difference_in_book_pages : diffFiveTwo 625 = 20 := by
  sorry

end five_two_difference_in_book_pages_l2994_299414


namespace perfect_square_trinomial_l2994_299476

theorem perfect_square_trinomial (m : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + m*x + 9 = (x + a)^2) → (m = 6 ∨ m = -6) := by
  sorry

end perfect_square_trinomial_l2994_299476


namespace lg_100_is_proposition_l2994_299416

/-- A proposition is a declarative sentence that can be judged to be true or false. -/
def IsProposition (s : String) : Prop := 
  ∃ (truthValue : Bool), (∀ (evaluation : String → Bool), evaluation s = truthValue)

/-- The statement "lg 100 = 2" -/
def statement : String := "lg 100 = 2"

/-- Theorem: The statement "lg 100 = 2" is a proposition -/
theorem lg_100_is_proposition : IsProposition statement := by
  sorry

end lg_100_is_proposition_l2994_299416


namespace least_with_twelve_factors_l2994_299400

/-- The number of positive factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- A function that returns true if n is the least positive integer with exactly k factors -/
def is_least_with_factors (n k : ℕ+) : Prop :=
  (num_factors n = k) ∧ ∀ m : ℕ+, m < n → num_factors m ≠ k

theorem least_with_twelve_factors :
  is_least_with_factors 72 12 := by sorry

end least_with_twelve_factors_l2994_299400


namespace total_frog_eyes_l2994_299489

/-- The number of frogs in the pond -/
def num_frogs : ℕ := 6

/-- The number of eyes each frog has -/
def eyes_per_frog : ℕ := 2

/-- Theorem: The total number of frog eyes in the pond is equal to the product of the number of frogs and the number of eyes per frog -/
theorem total_frog_eyes : num_frogs * eyes_per_frog = 12 := by
  sorry

end total_frog_eyes_l2994_299489


namespace f_decreasing_on_interval_l2994_299462

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x

-- Theorem statement
theorem f_decreasing_on_interval :
  ∀ x ∈ Set.Ioo 0 2, f' x < 0 :=
sorry

end f_decreasing_on_interval_l2994_299462


namespace committee_selection_with_fixed_member_l2994_299464

/-- The number of ways to select a committee with a fixed member -/
def select_committee (total : ℕ) (committee_size : ℕ) (fixed_members : ℕ) : ℕ :=
  Nat.choose (total - fixed_members) (committee_size - fixed_members)

/-- Theorem: Selecting a 4-person committee from 12 people with one fixed member -/
theorem committee_selection_with_fixed_member :
  select_committee 12 4 1 = 165 := by
  sorry

end committee_selection_with_fixed_member_l2994_299464


namespace min_value_trig_expression_l2994_299436

open Real

theorem min_value_trig_expression (θ : Real) (h : 0 < θ ∧ θ < π/2) :
  3 * cos θ + 1 / sin θ + 2 * tan θ ≥ 3 * Real.rpow 6 (1/3) := by
  sorry

end min_value_trig_expression_l2994_299436


namespace journey_distance_l2994_299438

theorem journey_distance (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) 
  (h1 : total_time = 20)
  (h2 : speed1 = 10)
  (h3 : speed2 = 15) : 
  ∃ (distance : ℝ), 
    distance / (2 * speed1) + distance / (2 * speed2) = total_time ∧ 
    distance = 240 := by
  sorry

end journey_distance_l2994_299438


namespace multiplier_value_l2994_299495

def f (x : ℝ) : ℝ := 3 * x - 5

theorem multiplier_value (x : ℝ) (h : x = 3) :
  ∃ m : ℝ, m * f x - 10 = f (x - 2) ∧ m = 2 := by
  sorry

end multiplier_value_l2994_299495


namespace hyperbola_eccentricity_l2994_299474

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 (a > 0, b > 0) and right focus F at (c, 0),
    if a line perpendicular to y = -bx/a passes through F and intersects the left branch of the hyperbola
    at point B such that vector FB = 2 * vector FA (where A is the foot of the perpendicular),
    then the eccentricity of the hyperbola is √5. -/
theorem hyperbola_eccentricity (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let F : ℝ × ℝ := (c, 0)
  let perpendicular_line := {(x, y) : ℝ × ℝ | y = a / b * (x - c)}
  let hyperbola := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1}
  let A := (a^2 / c, a * b / c)
  ∃ B : ℝ × ℝ, B.1 < 0 ∧ B ∈ hyperbola ∧ B ∈ perpendicular_line ∧
    (B.1 - F.1, B.2 - F.2) = (2 * (A.1 - F.1), 2 * (A.2 - F.2)) →
  c^2 / a^2 = 5 :=
sorry

end hyperbola_eccentricity_l2994_299474


namespace unique_function_satisfying_conditions_l2994_299406

-- Define the function f
def f (x : ℝ) : ℝ := x + 1

-- State the theorem
theorem unique_function_satisfying_conditions :
  (∀ (x : ℝ), x ≠ 0 → f x = x * f (1 / x)) ∧
  (∀ (x y : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ x ≠ -y → f x + f y = 1 + f (x + y)) ∧
  (∀ (g : ℝ → ℝ), 
    ((∀ (x : ℝ), x ≠ 0 → g x = x * g (1 / x)) ∧
     (∀ (x y : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ x ≠ -y → g x + g y = 1 + g (x + y)))
    → (∀ (x : ℝ), x ≠ 0 → g x = f x)) :=
by sorry

end unique_function_satisfying_conditions_l2994_299406


namespace mean_of_four_numbers_with_given_variance_l2994_299447

/-- Given a set of four positive real numbers with a specific variance, prove that their mean is 2. -/
theorem mean_of_four_numbers_with_given_variance 
  (x₁ x₂ x₃ x₄ : ℝ) 
  (pos₁ : 0 < x₁) (pos₂ : 0 < x₂) (pos₃ : 0 < x₃) (pos₄ : 0 < x₄)
  (variance_eq : (1/4) * (x₁^2 + x₂^2 + x₃^2 + x₄^2 - 16) = 
                 (1/4) * ((x₁ - (x₁ + x₂ + x₃ + x₄)/4)^2 + 
                          (x₂ - (x₁ + x₂ + x₃ + x₄)/4)^2 + 
                          (x₃ - (x₁ + x₂ + x₃ + x₄)/4)^2 + 
                          (x₄ - (x₁ + x₂ + x₃ + x₄)/4)^2)) :
  (x₁ + x₂ + x₃ + x₄) / 4 = 2 := by
  sorry

end mean_of_four_numbers_with_given_variance_l2994_299447


namespace baby_sea_turtles_on_sand_l2994_299426

theorem baby_sea_turtles_on_sand (total : ℕ) (swept_fraction : ℚ) (remaining : ℕ) : 
  total = 42 → 
  swept_fraction = 1/3 → 
  remaining = total - (total * swept_fraction).floor → 
  remaining = 28 := by
sorry

end baby_sea_turtles_on_sand_l2994_299426


namespace intersection_when_a_is_3_range_of_a_when_intersection_empty_l2994_299458

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 2 - a ≤ x ∧ x ≤ 2 + a}
def B : Set ℝ := {x | x ≤ 1 ∨ x ≥ 4}

-- Theorem for part 1
theorem intersection_when_a_is_3 :
  A 3 ∩ B = {x | -1 ≤ x ∧ x ≤ 1} ∪ {x | 4 ≤ x ∧ x ≤ 5} :=
by sorry

-- Theorem for part 2
theorem range_of_a_when_intersection_empty :
  ∀ a : ℝ, a > 0 → (A a ∩ B = ∅) → (0 < a ∧ a < 1) :=
by sorry

end intersection_when_a_is_3_range_of_a_when_intersection_empty_l2994_299458


namespace sixteen_million_scientific_notation_l2994_299465

/-- Given a number n, returns true if it's in scientific notation -/
def is_scientific_notation (n : ℝ) : Prop :=
  ∃ (a : ℝ) (b : ℤ), 1 ≤ a ∧ a < 10 ∧ n = a * (10 : ℝ) ^ b

theorem sixteen_million_scientific_notation :
  is_scientific_notation 16000000 ∧
  16000000 = 1.6 * (10 : ℝ) ^ 7 :=
sorry

end sixteen_million_scientific_notation_l2994_299465


namespace teacher_periods_per_day_l2994_299480

/-- Represents the number of periods a teacher teaches per day -/
def periods_per_day : ℕ := 5

/-- Represents the number of working days per month -/
def days_per_month : ℕ := 24

/-- Represents the payment per period in dollars -/
def payment_per_period : ℕ := 5

/-- Represents the number of months worked -/
def months_worked : ℕ := 6

/-- Represents the total earnings in dollars -/
def total_earnings : ℕ := 3600

/-- Theorem stating that given the conditions, the teacher teaches 5 periods per day -/
theorem teacher_periods_per_day :
  periods_per_day * days_per_month * months_worked * payment_per_period = total_earnings :=
sorry

end teacher_periods_per_day_l2994_299480


namespace faulty_meter_theorem_l2994_299454

/-- A shopkeeper sells goods using a faulty meter -/
structure Shopkeeper where
  profit_percent : ℝ
  supposed_weight : ℝ
  actual_weight : ℝ

/-- Calculate the weight difference of the faulty meter -/
def faulty_meter_weight (s : Shopkeeper) : ℝ :=
  s.supposed_weight - s.actual_weight

/-- Theorem stating the weight of the faulty meter -/
theorem faulty_meter_theorem (s : Shopkeeper) 
  (h1 : s.profit_percent = 11.11111111111111 / 100)
  (h2 : s.supposed_weight = 1000)
  (h3 : s.actual_weight = (1 - s.profit_percent) * s.supposed_weight) :
  faulty_meter_weight s = 100 := by
  sorry

end faulty_meter_theorem_l2994_299454


namespace system_solution_l2994_299451

theorem system_solution (a b c x y z : ℝ) 
  (h1 : x + y + z = 0)
  (h2 : c * x + a * y + b * z = 0)
  (h3 : (x + b)^2 + (y + c)^2 + (z + a)^2 = a^2 + b^2 + c^2)
  (h4 : a ≠ b)
  (h5 : b ≠ c)
  (h6 : a ≠ c) :
  ((x = 0 ∧ y = 0 ∧ z = 0) ∨ 
   (x = a - b ∧ y = b - c ∧ z = c - a)) :=
by sorry

end system_solution_l2994_299451


namespace smallest_rectangle_containing_circle_l2994_299492

theorem smallest_rectangle_containing_circle (r : ℝ) (h : r = 6) :
  (2 * r) * (2 * r) = 144 := by sorry

end smallest_rectangle_containing_circle_l2994_299492


namespace circle_area_difference_l2994_299475

theorem circle_area_difference (π : ℝ) : 
  let r1 : ℝ := 30
  let d2 : ℝ := 30
  let r2 : ℝ := d2 / 2
  let area1 : ℝ := π * r1^2
  let area2 : ℝ := π * r2^2
  area1 - area2 = 675 * π := by sorry

end circle_area_difference_l2994_299475


namespace division_chain_l2994_299407

theorem division_chain : (180 / 6) / 3 = 10 := by sorry

end division_chain_l2994_299407


namespace two_heads_five_coins_l2994_299463

/-- The probability of getting exactly k heads when tossing n fair coins -/
def coinTossProbability (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (1 / 2) ^ n

/-- Theorem: The probability of getting exactly two heads when tossing five fair coins is 5/16 -/
theorem two_heads_five_coins : coinTossProbability 5 2 = 5 / 16 := by
  sorry

end two_heads_five_coins_l2994_299463


namespace not_all_regular_pentagons_congruent_l2994_299409

-- Define a regular pentagon
structure RegularPentagon where
  sideLength : ℝ
  sideLength_pos : sideLength > 0

-- Define congruence for regular pentagons
def congruent (p1 p2 : RegularPentagon) : Prop :=
  p1.sideLength = p2.sideLength

-- Theorem statement
theorem not_all_regular_pentagons_congruent :
  ∃ (p1 p2 : RegularPentagon), ¬(congruent p1 p2) := by
  sorry

end not_all_regular_pentagons_congruent_l2994_299409


namespace sum_of_squares_nonzero_iff_one_nonzero_l2994_299455

theorem sum_of_squares_nonzero_iff_one_nonzero (a b : ℝ) :
  a^2 + b^2 ≠ 0 ↔ a ≠ 0 ∨ b ≠ 0 := by sorry

end sum_of_squares_nonzero_iff_one_nonzero_l2994_299455


namespace hidden_sea_portion_l2994_299496

/-- Represents the composition of the landscape visible from an airplane window -/
structure Landscape where
  cloud : ℚ  -- Fraction of landscape covered by cloud
  island : ℚ  -- Fraction of landscape occupied by island
  sea : ℚ    -- Fraction of landscape occupied by sea

/-- The conditions of the landscape as described in the problem -/
def airplane_view : Landscape where
  cloud := 1/2
  island := 1/3
  sea := 2/3

theorem hidden_sea_portion (L : Landscape) 
  (h1 : L.cloud = 1/2)
  (h2 : L.island = 1/3)
  (h3 : L.cloud + L.island + L.sea = 1) :
  L.cloud * L.sea = 5/12 := by
  sorry

#check hidden_sea_portion

end hidden_sea_portion_l2994_299496


namespace largest_non_representable_l2994_299497

def is_composite (n : ℕ) : Prop := ∃ m k, 1 < m ∧ 1 < k ∧ n = m * k

def is_representable (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 0 < a ∧ is_composite b ∧ n = 36 * a + b

theorem largest_non_representable : 
  (∀ n > 187, is_representable n) ∧ ¬is_representable 187 := by sorry

end largest_non_representable_l2994_299497
