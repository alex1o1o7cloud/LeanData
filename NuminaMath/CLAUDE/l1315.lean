import Mathlib

namespace NUMINAMATH_CALUDE_circle_symmetry_line_l1315_131503

theorem circle_symmetry_line (a b : ℝ) :
  let circle := fun (x y : ℝ) => x^2 + y^2 - 4*x + 2*y + 1 = 0
  let line := fun (x y : ℝ) => a*x - 2*b*y - 1 = 0
  let symmetric := ∀ (x y : ℝ), circle x y → (∃ (x' y' : ℝ), circle x' y' ∧ line ((x + x')/2) ((y + y')/2))
  symmetric → a*b ≤ 1/16 :=
by sorry

end NUMINAMATH_CALUDE_circle_symmetry_line_l1315_131503


namespace NUMINAMATH_CALUDE_cucumber_weight_problem_l1315_131506

/-- Proves that the initial weight of cucumbers is 100 pounds given the conditions -/
theorem cucumber_weight_problem (initial_water_percent : Real) 
                                 (final_water_percent : Real)
                                 (final_weight : Real) :
  initial_water_percent = 0.99 →
  final_water_percent = 0.98 →
  final_weight = 50 →
  ∃ (initial_weight : Real),
    initial_weight = 100 ∧
    (1 - initial_water_percent) * initial_weight = (1 - final_water_percent) * final_weight :=
by
  sorry

#check cucumber_weight_problem

end NUMINAMATH_CALUDE_cucumber_weight_problem_l1315_131506


namespace NUMINAMATH_CALUDE_part_one_part_two_l1315_131563

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x - 1|

-- Part I
theorem part_one (m : ℝ) (h_m : m > 0) :
  (∀ x, x ∈ Set.Icc (-2 : ℝ) 2 ↔ f (x + 1/2) ≤ 2*m + 1) → m = 3/2 := by
  sorry

-- Part II
theorem part_two :
  (∃ a : ℝ, ∀ x y : ℝ, f x ≤ 2^y + a/(2^y) + |2*x + 3|) ∧
  (∀ a : ℝ, (∀ x y : ℝ, f x ≤ 2^y + a/(2^y) + |2*x + 3|) → a ≥ 4) := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1315_131563


namespace NUMINAMATH_CALUDE_right_triangle_sets_l1315_131521

def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem right_triangle_sets :
  ¬(is_right_triangle 3 4 6) ∧
  (is_right_triangle 7 24 25) ∧
  (is_right_triangle 6 8 10) ∧
  (is_right_triangle 9 12 15) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l1315_131521


namespace NUMINAMATH_CALUDE_train_crossing_time_l1315_131528

/-- Proves that a train of given length and speed takes the calculated time to cross an electric pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 600 → 
  train_speed_kmh = 144 → 
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) → 
  crossing_time = 15 := by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1315_131528


namespace NUMINAMATH_CALUDE_vector_magnitude_l1315_131515

/-- Given two vectors a and b in R², if (a - 2b) is perpendicular to a, then the magnitude of b is √5. -/
theorem vector_magnitude (a b : ℝ × ℝ) (h : a = (-1, 3) ∧ b.1 = 1) :
  (a - 2 • b) • a = 0 → ‖b‖ = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l1315_131515


namespace NUMINAMATH_CALUDE_min_purses_needed_l1315_131530

/-- Represents a distribution of coins into purses -/
def CoinDistribution := List Nat

/-- Checks if a distribution is valid for a given number of sailors -/
def isValidDistribution (d : CoinDistribution) (n : Nat) : Prop :=
  (d.sum = 60) ∧ (∃ (x : Nat), d.sum = n * x)

/-- Checks if a distribution is valid for all required sailor counts -/
def isValidForAllSailors (d : CoinDistribution) : Prop :=
  isValidDistribution d 2 ∧
  isValidDistribution d 3 ∧
  isValidDistribution d 4 ∧
  isValidDistribution d 5

/-- The main theorem stating the minimum number of purses needed -/
theorem min_purses_needed :
  ∃ (d : CoinDistribution),
    d.length = 9 ∧
    isValidForAllSailors d ∧
    ∀ (d' : CoinDistribution),
      isValidForAllSailors d' →
      d'.length ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_purses_needed_l1315_131530


namespace NUMINAMATH_CALUDE_beyonce_song_count_l1315_131565

/-- The total number of songs released by Beyonce -/
def total_songs (singles : ℕ) (albums_15 : ℕ) (songs_per_album_15 : ℕ) (albums_20 : ℕ) (songs_per_album_20 : ℕ) : ℕ :=
  singles + albums_15 * songs_per_album_15 + albums_20 * songs_per_album_20

/-- Theorem stating that Beyonce has released 55 songs in total -/
theorem beyonce_song_count :
  total_songs 5 2 15 1 20 = 55 := by
  sorry


end NUMINAMATH_CALUDE_beyonce_song_count_l1315_131565


namespace NUMINAMATH_CALUDE_square_side_length_l1315_131519

theorem square_side_length (area : ℝ) (side : ℝ) :
  area = 169 →
  side * side = area →
  side = 13 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l1315_131519


namespace NUMINAMATH_CALUDE_stamp_collection_problem_l1315_131510

/-- The number of red stamps Simon has -/
def simon_red_stamps : ℕ := 30

/-- The price of a red stamp in cents -/
def red_stamp_price : ℕ := 50

/-- The price of a white stamp in cents -/
def white_stamp_price : ℕ := 20

/-- The difference in earnings between Simon and Peter in dollars -/
def earnings_difference : ℚ := 1

/-- The number of white stamps Peter has -/
def peter_white_stamps : ℕ := 70

theorem stamp_collection_problem :
  (simon_red_stamps * red_stamp_price : ℚ) / 100 - 
  (peter_white_stamps * white_stamp_price : ℚ) / 100 = earnings_difference :=
sorry

end NUMINAMATH_CALUDE_stamp_collection_problem_l1315_131510


namespace NUMINAMATH_CALUDE_peter_glasses_purchase_l1315_131529

/-- Represents the purchase of glasses by Peter --/
def glassesPurchase (smallCost largeCost initialMoney smallCount change : ℕ) : Prop :=
  ∃ (largeCount : ℕ),
    smallCost * smallCount + largeCost * largeCount = initialMoney - change

theorem peter_glasses_purchase :
  glassesPurchase 3 5 50 8 1 →
  ∃ (largeCount : ℕ), largeCount = 5 ∧ glassesPurchase 3 5 50 8 1 := by
  sorry

end NUMINAMATH_CALUDE_peter_glasses_purchase_l1315_131529


namespace NUMINAMATH_CALUDE_prob_sum_seven_l1315_131513

/-- The number of sides on each die -/
def num_sides : ℕ := 6

/-- The set of all possible outcomes when throwing two dice -/
def all_outcomes : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range num_sides) (Finset.range num_sides)

/-- The set of outcomes where the sum is 7 -/
def sum_seven : Finset (ℕ × ℕ) :=
  all_outcomes.filter (λ p => p.1 + p.2 + 2 = 7)

/-- The probability of getting a sum of 7 when throwing two fair dice -/
theorem prob_sum_seven :
  (sum_seven.card : ℚ) / all_outcomes.card = 1 / 6 := by
  sorry


end NUMINAMATH_CALUDE_prob_sum_seven_l1315_131513


namespace NUMINAMATH_CALUDE_min_value_expression_l1315_131516

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 8) :
  (a + 3 * b) * (b + 3 * c) * (a * c + 2) ≥ 288 ∧
  ∃ (a' b' c' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ a' * b' * c' = 8 ∧
    (a' + 3 * b') * (b' + 3 * c') * (a' * c' + 2) = 288 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1315_131516


namespace NUMINAMATH_CALUDE_parallel_lines_parameter_l1315_131566

/-- Given two lines in the plane, prove that the parameter 'a' must equal 4 for the lines to be parallel -/
theorem parallel_lines_parameter (a : ℝ) : 
  (∀ x y : ℝ, 3 * x + (1 - a) * y + 1 = 0 ↔ x - y + 2 = 0) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_parameter_l1315_131566


namespace NUMINAMATH_CALUDE_no_primes_in_range_l1315_131504

theorem no_primes_in_range (n : ℕ) (h : n > 2) :
  ∀ p, Prime p → ¬(n.factorial + 2 < p ∧ p < n.factorial + n + 1) := by
  sorry

end NUMINAMATH_CALUDE_no_primes_in_range_l1315_131504


namespace NUMINAMATH_CALUDE_inscribed_square_side_length_l1315_131577

/-- A right triangle with sides 5, 12, and 13 -/
structure RightTriangle where
  pq : ℝ
  qr : ℝ
  pr : ℝ
  is_right : pq^2 + qr^2 = pr^2
  pq_eq : pq = 5
  qr_eq : qr = 12
  pr_eq : pr = 13

/-- A square inscribed in the right triangle -/
structure InscribedSquare (t : RightTriangle) where
  side_length : ℝ
  on_hypotenuse : side_length ≤ t.pr
  on_legs : side_length ≤ t.pq ∧ side_length ≤ t.qr

/-- The side length of the inscribed square is 156/25 -/
theorem inscribed_square_side_length (t : RightTriangle) (s : InscribedSquare t) :
  s.side_length = 156 / 25 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_side_length_l1315_131577


namespace NUMINAMATH_CALUDE_problem_solution_l1315_131569

theorem problem_solution (a n : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * n * 45 * 49) : n = 945 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1315_131569


namespace NUMINAMATH_CALUDE_quadratic_completion_square_l1315_131592

theorem quadratic_completion_square (a : ℝ) : 
  (a > 0) → 
  (∃ n : ℝ, ∀ x : ℝ, x^2 + a*x + 27 = (x + n)^2 + 3) → 
  a = 4 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_completion_square_l1315_131592


namespace NUMINAMATH_CALUDE_eunice_pots_l1315_131590

/-- Given a total number of seeds and a number of seeds per pot (except for the last pot),
    calculate the number of pots needed. -/
def calculate_pots (total_seeds : ℕ) (seeds_per_pot : ℕ) : ℕ :=
  (total_seeds - 1) / seeds_per_pot + 1

/-- Theorem stating that with 10 seeds and 3 seeds per pot (except for the last pot),
    the number of pots needed is 4. -/
theorem eunice_pots : calculate_pots 10 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_eunice_pots_l1315_131590


namespace NUMINAMATH_CALUDE_geometric_progression_common_ratio_l1315_131533

theorem geometric_progression_common_ratio :
  ∀ (a : ℝ) (r : ℝ),
    a > 0 →  -- First term is positive
    r > 0 →  -- Common ratio is positive (to ensure all terms are positive)
    a = a * r + a * r^2 + a * r^3 →  -- First term equals sum of next three terms
    r = (Real.sqrt 5 - 1) / 3 :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_common_ratio_l1315_131533


namespace NUMINAMATH_CALUDE_binomial_square_coeff_l1315_131567

/-- If ax^2 + 8x + 16 is the square of a binomial, then a = 1 -/
theorem binomial_square_coeff (a : ℝ) : 
  (∃ r s : ℝ, ∀ x : ℝ, a * x^2 + 8 * x + 16 = (r * x + s)^2) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_coeff_l1315_131567


namespace NUMINAMATH_CALUDE_gp_ratio_proof_l1315_131554

/-- Given a geometric progression where the ratio of the sum of the first 6 terms
    to the sum of the first 3 terms is 28, prove that the common ratio is 3. -/
theorem gp_ratio_proof (a r : ℝ) (hr : r ≠ 1) :
  (a * (1 - r^6) / (1 - r)) / (a * (1 - r^3) / (1 - r)) = 28 →
  r = 3 :=
by sorry

end NUMINAMATH_CALUDE_gp_ratio_proof_l1315_131554


namespace NUMINAMATH_CALUDE_a_greater_than_b_greater_than_one_l1315_131562

theorem a_greater_than_b_greater_than_one
  (n : ℕ) (a b : ℝ) 
  (h_n : n ≥ 2)
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_a_eq : a^n = a + 1)
  (h_b_eq : b^(2*n) = b + 3*a) :
  a > b ∧ b > 1 := by
sorry

end NUMINAMATH_CALUDE_a_greater_than_b_greater_than_one_l1315_131562


namespace NUMINAMATH_CALUDE_fan_airflow_in_week_l1315_131531

/-- Calculates the total airflow created by a fan in one week -/
theorem fan_airflow_in_week 
  (airflow_rate : ℝ) 
  (daily_operation_time : ℝ) 
  (days_in_week : ℕ) 
  (seconds_in_minute : ℕ) : 
  airflow_rate * daily_operation_time * (days_in_week : ℝ) * (seconds_in_minute : ℝ) = 42000 :=
by
  -- Assuming airflow_rate = 10, daily_operation_time = 10, days_in_week = 7, seconds_in_minute = 60
  sorry

#check fan_airflow_in_week

end NUMINAMATH_CALUDE_fan_airflow_in_week_l1315_131531


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1315_131581

theorem inequality_equivalence (x : ℝ) : (1/2 * x - 1 > 0) ↔ (x > 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1315_131581


namespace NUMINAMATH_CALUDE_sum_coordinates_reflection_l1315_131578

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define reflection over x-axis
def reflect_x (p : Point) : Point :=
  (p.1, -p.2)

theorem sum_coordinates_reflection (y : ℝ) :
  let C : Point := (3, y)
  let D : Point := reflect_x C
  C.1 + C.2 + D.1 + D.2 = 6 := by
sorry

end NUMINAMATH_CALUDE_sum_coordinates_reflection_l1315_131578


namespace NUMINAMATH_CALUDE_digit_pair_sum_l1315_131574

/-- Two different digits that form two-digit numbers whose sum is 202 -/
structure DigitPair where
  a : ℕ
  b : ℕ
  a_is_digit : a ≥ 1 ∧ a ≤ 9
  b_is_digit : b ≥ 0 ∧ b ≤ 9
  a_ne_b : a ≠ b
  sum_eq_202 : 10 * a + b + 10 * b + a = 202

/-- The sum of the digits in a DigitPair is 12 -/
theorem digit_pair_sum (p : DigitPair) : p.a + p.b = 12 := by
  sorry

end NUMINAMATH_CALUDE_digit_pair_sum_l1315_131574


namespace NUMINAMATH_CALUDE_mod_eq_two_l1315_131525

theorem mod_eq_two (n : ℤ) : 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -5678 [ZMOD 10] → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_mod_eq_two_l1315_131525


namespace NUMINAMATH_CALUDE_flashlight_distance_difference_l1315_131579

/-- The visibility distance of Veronica's flashlight in feet -/
def veronica_distance : ℕ := 1000

/-- The visibility distance of Freddie's flashlight in feet -/
def freddie_distance : ℕ := 3 * veronica_distance

/-- The visibility distance of Velma's flashlight in feet -/
def velma_distance : ℕ := 5 * freddie_distance - 2000

/-- The difference in visibility distance between Velma's and Veronica's flashlights -/
theorem flashlight_distance_difference : velma_distance - veronica_distance = 12000 := by
  sorry

end NUMINAMATH_CALUDE_flashlight_distance_difference_l1315_131579


namespace NUMINAMATH_CALUDE_xy_product_l1315_131588

theorem xy_product (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 162) : x * y = 21 := by
  sorry

end NUMINAMATH_CALUDE_xy_product_l1315_131588


namespace NUMINAMATH_CALUDE_gcd_7_factorial_8_factorial_l1315_131586

theorem gcd_7_factorial_8_factorial : Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = Nat.factorial 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_7_factorial_8_factorial_l1315_131586


namespace NUMINAMATH_CALUDE_vector_equation_solution_l1315_131587

def a : ℝ × ℝ × ℝ := (1, 3, -2)
def b : ℝ × ℝ × ℝ := (2, 1, 0)

def cross_product (v w : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v.2.1 * w.2.2 - v.2.2 * w.2.1,
   v.2.2 * w.1 - v.1 * w.2.2,
   v.1 * w.2.1 - v.2.1 * w.1)

def dot_product (v w : ℝ × ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2.1 * w.2.1 + v.2.2 * w.2.2

theorem vector_equation_solution :
  ∃ (p q r : ℝ),
    (5, 2, -3) = (p * a.1 + q * b.1 + r * (cross_product a b).1,
                  p * a.2.1 + q * b.2.1 + r * (cross_product a b).2.1,
                  p * a.2.2 + q * b.2.2 + r * (cross_product a b).2.2) →
    r = 17 / 45 := by
  sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l1315_131587


namespace NUMINAMATH_CALUDE_smallest_divisible_by_79_and_83_l1315_131560

theorem smallest_divisible_by_79_and_83 :
  ∃ (m : ℕ), 
    m > 0 ∧
    79 ∣ (m^3 - 3*m^2 + 2*m) ∧
    83 ∣ (m^3 - 3*m^2 + 2*m) ∧
    (∀ (k : ℕ), k > 0 ∧ k < m → ¬(79 ∣ (k^3 - 3*k^2 + 2*k) ∧ 83 ∣ (k^3 - 3*k^2 + 2*k))) ∧
    m = 3715 :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_79_and_83_l1315_131560


namespace NUMINAMATH_CALUDE_division_problem_l1315_131599

theorem division_problem : (120 : ℚ) / ((6 : ℚ) / 2 * 3) = 120 / 9 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1315_131599


namespace NUMINAMATH_CALUDE_travel_time_at_different_speeds_l1315_131517

/-- The travel time between two points given different speeds -/
theorem travel_time_at_different_speeds 
  (initial_speed : ℝ) 
  (initial_time : ℝ) 
  (new_speed : ℝ) 
  (h1 : initial_speed = 80) 
  (h2 : initial_time = 16/3) 
  (h3 : new_speed = 50) : 
  ∃ (new_time : ℝ), 
    (initial_speed * initial_time = new_speed * new_time) ∧ 
    (abs (new_time - 8.53) < 0.01) :=
by sorry

end NUMINAMATH_CALUDE_travel_time_at_different_speeds_l1315_131517


namespace NUMINAMATH_CALUDE_perpendicular_line_value_l1315_131568

theorem perpendicular_line_value (θ : Real) (h : Real.tan θ = -3) :
  2 / (3 * Real.sin θ ^ 2 - Real.cos θ ^ 2) = 10/13 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_value_l1315_131568


namespace NUMINAMATH_CALUDE_equation_solutions_l1315_131527

theorem equation_solutions :
  (∃ x : ℝ, 2 * (2 - x) - 5 * (2 - x) = 9 ∧ x = 5) ∧
  (∃ x : ℝ, x / 3 - (3 * x - 1) / 6 = 1 ∧ x = -5) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1315_131527


namespace NUMINAMATH_CALUDE_x_range_l1315_131598

theorem x_range (x y : ℝ) (h1 : 4 * x + y = 3) (h2 : -2 < y) (h3 : y ≤ 7) :
  1 ≤ x ∧ x < 5/4 := by
  sorry

end NUMINAMATH_CALUDE_x_range_l1315_131598


namespace NUMINAMATH_CALUDE_exists_x_where_inequality_fails_l1315_131509

theorem exists_x_where_inequality_fails : ∃ x : ℝ, x > 0 ∧ 2^x - x^2 < 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_x_where_inequality_fails_l1315_131509


namespace NUMINAMATH_CALUDE_days_not_played_in_june_l1315_131555

/-- The number of days in June. -/
def june_days : ℕ := 30

/-- The number of songs Vivian plays per day. -/
def vivian_songs : ℕ := 10

/-- The number of songs Clara plays per day. -/
def clara_songs : ℕ := vivian_songs - 2

/-- The total number of songs both Vivian and Clara listened to in June. -/
def total_songs : ℕ := 396

/-- The number of days they played songs in June. -/
def days_played : ℕ := total_songs / (vivian_songs + clara_songs)

theorem days_not_played_in_june : june_days - days_played = 8 := by
  sorry

end NUMINAMATH_CALUDE_days_not_played_in_june_l1315_131555


namespace NUMINAMATH_CALUDE_minimum_employees_needed_l1315_131571

theorem minimum_employees_needed (forest_employees : ℕ) (marine_employees : ℕ) (both_employees : ℕ)
  (h1 : forest_employees = 95)
  (h2 : marine_employees = 80)
  (h3 : both_employees = 35)
  (h4 : both_employees ≤ forest_employees ∧ both_employees ≤ marine_employees) :
  forest_employees + marine_employees - both_employees = 140 :=
by sorry

end NUMINAMATH_CALUDE_minimum_employees_needed_l1315_131571


namespace NUMINAMATH_CALUDE_product_plus_one_is_square_l1315_131500

theorem product_plus_one_is_square (x y : ℕ) (h : x * y = (x + 2) * (y - 2)) :
  x * y + 1 = (x + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_product_plus_one_is_square_l1315_131500


namespace NUMINAMATH_CALUDE_paul_work_time_l1315_131542

-- Define the work rates and time
def george_work_rate : ℚ := 3 / (5 * 9)
def total_work : ℚ := 1
def george_paul_time : ℚ := 4
def george_initial_work : ℚ := 3 / 5

-- Theorem statement
theorem paul_work_time (paul_work_rate : ℚ) : 
  george_work_rate + paul_work_rate = (total_work - george_initial_work) / george_paul_time →
  total_work / paul_work_rate = 90 / 13 := by
  sorry

end NUMINAMATH_CALUDE_paul_work_time_l1315_131542


namespace NUMINAMATH_CALUDE_geometric_progression_unique_p_l1315_131556

theorem geometric_progression_unique_p : 
  ∃! (p : ℝ), p > 0 ∧ (2 * Real.sqrt p) ^ 2 = (p - 2) * (-3 - p) := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_unique_p_l1315_131556


namespace NUMINAMATH_CALUDE_remainder_nine_eight_mod_five_l1315_131540

theorem remainder_nine_eight_mod_five : 9^8 % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_nine_eight_mod_five_l1315_131540


namespace NUMINAMATH_CALUDE_closure_M_intersect_N_l1315_131534

-- Define the set M
def M : Set ℝ := {x | 2/x < 1}

-- Define the set N
def N : Set ℝ := {y | ∃ x, y = Real.sqrt (x - 1)}

-- State the theorem
theorem closure_M_intersect_N :
  (closure M) ∩ N = Set.Icc 0 2 := by sorry

end NUMINAMATH_CALUDE_closure_M_intersect_N_l1315_131534


namespace NUMINAMATH_CALUDE_total_wicks_count_l1315_131535

/-- The length of the spool in feet -/
def spool_length : ℕ := 15

/-- The length of short wicks in inches -/
def short_wick : ℕ := 6

/-- The length of long wicks in inches -/
def long_wick : ℕ := 12

/-- Conversion factor from feet to inches -/
def feet_to_inches : ℕ := 12

theorem total_wicks_count : ∃ (n : ℕ), 
  n * short_wick + n * long_wick = spool_length * feet_to_inches ∧
  n + n = 20 := by sorry

end NUMINAMATH_CALUDE_total_wicks_count_l1315_131535


namespace NUMINAMATH_CALUDE_alcohol_solution_percentage_l1315_131558

theorem alcohol_solution_percentage (initial_volume : ℝ) (initial_percentage : ℝ) 
  (added_alcohol : ℝ) (added_water : ℝ) : 
  initial_volume = 40 →
  initial_percentage = 5 →
  added_alcohol = 4.5 →
  added_water = 5.5 →
  let initial_alcohol := initial_volume * (initial_percentage / 100)
  let final_alcohol := initial_alcohol + added_alcohol
  let final_volume := initial_volume + added_alcohol + added_water
  let final_percentage := (final_alcohol / final_volume) * 100
  final_percentage = 13 := by
sorry

end NUMINAMATH_CALUDE_alcohol_solution_percentage_l1315_131558


namespace NUMINAMATH_CALUDE_inequality_range_l1315_131538

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, (3 : ℝ)^(x^2 - 2*a*x) > (1/3 : ℝ)^(x + 1)) → 
  -1/2 < a ∧ a < 3/2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l1315_131538


namespace NUMINAMATH_CALUDE_exists_nonconvergent_sequence_l1315_131539

/-- A sequence of natural numbers -/
def Sequence := ℕ → ℕ

/-- Property: The sequence is increasing -/
def IsIncreasing (a : Sequence) : Prop :=
  ∀ n, a n < a (n + 1)

/-- Property: Each term is either the arithmetic mean or the geometric mean of its neighbors -/
def IsMeanOfNeighbors (a : Sequence) : Prop :=
  ∀ n, (2 * a (n + 1) = a n + a (n + 2)) ∨ (a (n + 1) * a (n + 1) = a n * a (n + 2))

/-- Property: The sequence is an arithmetic progression from a certain point -/
def EventuallyArithmetic (a : Sequence) : Prop :=
  ∃ N, ∀ n ≥ N, a (n + 2) - a (n + 1) = a (n + 1) - a n

/-- Property: The sequence is a geometric progression from a certain point -/
def EventuallyGeometric (a : Sequence) : Prop :=
  ∃ N, ∀ n ≥ N, a (n + 2) * a n = a (n + 1) * a (n + 1)

/-- The main theorem -/
theorem exists_nonconvergent_sequence :
  ∃ (a : Sequence), IsIncreasing a ∧ IsMeanOfNeighbors a ∧
    ¬(EventuallyArithmetic a ∨ EventuallyGeometric a) :=
sorry

end NUMINAMATH_CALUDE_exists_nonconvergent_sequence_l1315_131539


namespace NUMINAMATH_CALUDE_valid_pairs_l1315_131596

theorem valid_pairs : 
  ∀ m n : ℕ, 
    (∃ k : ℕ, m + 1 = n * k) ∧ 
    (∃ l : ℕ, n^2 - n + 1 = m * l) → 
    ((m = 1 ∧ n = 1) ∨ (m = 1 ∧ n = 2) ∨ (m = 3 ∧ n = 2)) := by
  sorry

end NUMINAMATH_CALUDE_valid_pairs_l1315_131596


namespace NUMINAMATH_CALUDE_train_speed_l1315_131593

/-- 
Given a train with length 150 meters that crosses an electric pole in 3 seconds,
prove that its speed is 50 meters per second.
-/
theorem train_speed (length : ℝ) (time : ℝ) (speed : ℝ) : 
  length = 150 ∧ 
  time = 3 ∧ 
  speed = length / time → 
  speed = 50 := by sorry

end NUMINAMATH_CALUDE_train_speed_l1315_131593


namespace NUMINAMATH_CALUDE_binomial_10_1_l1315_131589

theorem binomial_10_1 : Nat.choose 10 1 = 10 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_1_l1315_131589


namespace NUMINAMATH_CALUDE_discriminant_nonnegativity_l1315_131548

theorem discriminant_nonnegativity (x : ℤ) :
  x^2 * (49 - 40 * x^2) ≥ 0 ↔ x = 0 ∨ x = 1 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_discriminant_nonnegativity_l1315_131548


namespace NUMINAMATH_CALUDE_yellow_papers_in_ten_by_ten_square_l1315_131532

/-- Represents a square arrangement of colored papers -/
structure ColoredSquare where
  size : Nat
  redPeriphery : Bool

/-- Calculates the number of yellow papers in a ColoredSquare -/
def yellowPapers (square : ColoredSquare) : Nat :=
  if square.redPeriphery then
    square.size * square.size - (4 * square.size - 4)
  else
    square.size * square.size

/-- Theorem stating that a 10x10 ColoredSquare with red periphery has 64 yellow papers -/
theorem yellow_papers_in_ten_by_ten_square :
  yellowPapers { size := 10, redPeriphery := true } = 64 := by
  sorry

#eval yellowPapers { size := 10, redPeriphery := true }

end NUMINAMATH_CALUDE_yellow_papers_in_ten_by_ten_square_l1315_131532


namespace NUMINAMATH_CALUDE_solve_for_m_l1315_131502

theorem solve_for_m : ∃ m : ℝ, 
  (∀ x y : ℝ, x = 1 ∧ y = -1 → 2*x + m + y = 0) → m = -1 :=
by sorry

end NUMINAMATH_CALUDE_solve_for_m_l1315_131502


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_23_l1315_131523

theorem greatest_three_digit_multiple_of_23 : ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 23 = 0 → n ≤ 991 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_23_l1315_131523


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l1315_131550

theorem negation_of_universal_statement :
  (¬∀ x : ℝ, x > 2 → x^2 - 2*x > 0) ↔ (∃ x : ℝ, x > 2 ∧ x^2 - 2*x ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l1315_131550


namespace NUMINAMATH_CALUDE_final_sum_theorem_l1315_131557

theorem final_sum_theorem (a b S : ℝ) (h : a + b = S) :
  2 * (a + 5) + 2 * (b - 5) = 2 * S := by
  sorry

end NUMINAMATH_CALUDE_final_sum_theorem_l1315_131557


namespace NUMINAMATH_CALUDE_marks_jump_height_l1315_131549

theorem marks_jump_height :
  ∀ (mark_height lisa_height jacob_height james_height : ℝ),
    lisa_height = 2 * mark_height →
    jacob_height = 2 * lisa_height →
    james_height = 16 →
    james_height = 2/3 * jacob_height →
    mark_height = 6 := by
  sorry

end NUMINAMATH_CALUDE_marks_jump_height_l1315_131549


namespace NUMINAMATH_CALUDE_parallel_linear_functions_min_value_l1315_131551

/-- Two linear functions with parallel graphs not parallel to coordinate axes -/
structure ParallelLinearFunctions where
  f : ℝ → ℝ
  g : ℝ → ℝ
  parallel : ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, f x = a * x + b) ∧ (∀ x, g x = a * x + c)

/-- The minimum value of a quadratic function -/
def quadratic_min (h : ℝ → ℝ) : ℝ := sorry

theorem parallel_linear_functions_min_value 
  (funcs : ParallelLinearFunctions)
  (h_min : quadratic_min (λ x => (funcs.f x)^2 + 5 * funcs.g x) = -17) :
  quadratic_min (λ x => (funcs.g x)^2 + 5 * funcs.f x) = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_linear_functions_min_value_l1315_131551


namespace NUMINAMATH_CALUDE_expand_expression_l1315_131575

theorem expand_expression (x : ℝ) : -2 * (5 * x^3 - 7 * x^2 + x - 4) = -10 * x^3 + 14 * x^2 - 2 * x + 8 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1315_131575


namespace NUMINAMATH_CALUDE_revenue_calculation_l1315_131585

-- Define the initial and final counts of phones
def samsung_start : ℕ := 14
def samsung_end : ℕ := 10
def iphone_start : ℕ := 8
def iphone_end : ℕ := 5

-- Define the number of damaged phones
def samsung_damaged : ℕ := 2
def iphone_damaged : ℕ := 1

-- Define the retail prices
def samsung_price : ℚ := 800
def iphone_price : ℚ := 1000

-- Define the discount and tax rates
def samsung_discount : ℚ := 0.10
def samsung_tax : ℚ := 0.12
def iphone_discount : ℚ := 0.15
def iphone_tax : ℚ := 0.10

-- Calculate the number of phones sold
def samsung_sold : ℕ := samsung_start - samsung_end - samsung_damaged
def iphone_sold : ℕ := iphone_start - iphone_end - iphone_damaged

-- Calculate the final price for each phone type after discount and tax
def samsung_final_price : ℚ := samsung_price * (1 - samsung_discount) * (1 + samsung_tax)
def iphone_final_price : ℚ := iphone_price * (1 - iphone_discount) * (1 + iphone_tax)

-- Calculate the total revenue
def total_revenue : ℚ := samsung_final_price * samsung_sold + iphone_final_price * iphone_sold

-- Theorem to prove
theorem revenue_calculation : total_revenue = 3482.80 := by
  sorry

end NUMINAMATH_CALUDE_revenue_calculation_l1315_131585


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l1315_131541

theorem inequality_and_equality_condition (n : ℕ+) :
  (1/3 : ℝ) * n.val^2 + (1/2 : ℝ) * n.val + (1/6 : ℝ) ≥ (n.val.factorial : ℝ)^((2 : ℝ) / n.val) ∧
  ((1/3 : ℝ) * n.val^2 + (1/2 : ℝ) * n.val + (1/6 : ℝ) = (n.val.factorial : ℝ)^((2 : ℝ) / n.val) ↔ n = 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l1315_131541


namespace NUMINAMATH_CALUDE_lineup_count_l1315_131508

def team_size : ℕ := 18

def lineup_positions : List String := ["goalkeeper", "center-back", "center-back", "left-back", "right-back", "midfielder", "midfielder", "midfielder"]

def number_of_lineups : ℕ :=
  team_size *
  (team_size - 1) * (team_size - 2) *
  (team_size - 3) *
  (team_size - 4) *
  (team_size - 5) * (team_size - 6) * (team_size - 7)

theorem lineup_count :
  number_of_lineups = 95414400 :=
by sorry

end NUMINAMATH_CALUDE_lineup_count_l1315_131508


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1315_131507

/-- Arithmetic sequence sum -/
def arithmetic_sum (a1 : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * a1 + (n * (n - 1) : ℚ) / 2 * d

/-- Arithmetic sequence term -/
def arithmetic_term (a1 : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  a1 + (n - 1 : ℚ) * d

theorem arithmetic_sequence_common_difference :
  ∃ (a1 : ℚ), 
    arithmetic_sum a1 (1/5) 5 = 6 ∧ 
    arithmetic_term a1 (1/5) 2 = 1 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1315_131507


namespace NUMINAMATH_CALUDE_line_through_center_45deg_l1315_131537

/-- The circle C in the 2D plane -/
def circle_C (x y : ℝ) : Prop :=
  (x + 1)^2 + (y - 2)^2 = 4

/-- The center of circle C -/
def center_C : ℝ × ℝ := (-1, 2)

/-- A line with slope 1 (45° angle) passing through a point -/
def line_45deg (x y a b : ℝ) : Prop :=
  y - b = x - a

/-- The equation of the line we're interested in -/
def target_line (x y : ℝ) : Prop :=
  x - y + 3 = 0

/-- Theorem stating that the target line passes through the center of circle C
    and has a slope angle of 45° -/
theorem line_through_center_45deg :
  (∀ x y, target_line x y ↔ line_45deg x y (center_C.1) (center_C.2)) ∧
  target_line (center_C.1) (center_C.2) :=
sorry

end NUMINAMATH_CALUDE_line_through_center_45deg_l1315_131537


namespace NUMINAMATH_CALUDE_geometric_sequence_special_ratio_l1315_131572

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Define the problem statement
theorem geometric_sequence_special_ratio
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_geom : geometric_sequence a q)
  (h_arith : a 2 - (1/2 * a 3) = (1/2 * a 3) - a 1) :
  q = (Real.sqrt 5 + 1) / 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_special_ratio_l1315_131572


namespace NUMINAMATH_CALUDE_cistern_filling_time_l1315_131594

theorem cistern_filling_time (partial_fill_time : ℝ) (partial_fill_fraction : ℝ) 
  (h1 : partial_fill_time = 5)
  (h2 : partial_fill_fraction = 1 / 11) :
  partial_fill_time / partial_fill_fraction = 55 := by
  sorry

end NUMINAMATH_CALUDE_cistern_filling_time_l1315_131594


namespace NUMINAMATH_CALUDE_tape_length_l1315_131511

/-- Given 15 pieces of tape, each 20 cm long, overlapping by 5 cm, 
    the total length is 230 cm -/
theorem tape_length (n : ℕ) (piece_length overlap : ℝ) 
  (h1 : n = 15)
  (h2 : piece_length = 20)
  (h3 : overlap = 5) :
  piece_length + (n - 1) * (piece_length - overlap) = 230 :=
by
  sorry

end NUMINAMATH_CALUDE_tape_length_l1315_131511


namespace NUMINAMATH_CALUDE_correct_reassembly_probability_l1315_131524

/-- Represents the number of subcubes in each dimension of the larger cube -/
def cubeDimension : ℕ := 3

/-- Represents the total number of subcubes in the larger cube -/
def totalSubcubes : ℕ := cubeDimension ^ 3

/-- Represents the number of corner subcubes -/
def cornerCubes : ℕ := 8

/-- Represents the number of edge subcubes -/
def edgeCubes : ℕ := 12

/-- Represents the number of face subcubes -/
def faceCubes : ℕ := 6

/-- Represents the number of center subcubes -/
def centerCubes : ℕ := 1

/-- Represents the number of possible orientations for a corner subcube -/
def cornerOrientations : ℕ := 3

/-- Represents the number of possible orientations for an edge subcube -/
def edgeOrientations : ℕ := 2

/-- Represents the number of possible orientations for a face subcube -/
def faceOrientations : ℕ := 4

/-- Represents the total number of possible orientations for any subcube -/
def totalOrientations : ℕ := 24

/-- Calculates the number of correct reassemblings -/
def correctReassemblings : ℕ :=
  (cornerOrientations ^ cornerCubes) * (cornerCubes.factorial) *
  (edgeOrientations ^ edgeCubes) * (edgeCubes.factorial) *
  (faceOrientations ^ faceCubes) * (faceCubes.factorial) *
  (centerCubes.factorial)

/-- Calculates the total number of possible reassemblings -/
def totalReassemblings : ℕ :=
  (totalOrientations ^ totalSubcubes) * (totalSubcubes.factorial)

/-- Theorem: The probability of correctly reassembling the cube is equal to
    the ratio of correct reassemblings to total possible reassemblings -/
theorem correct_reassembly_probability :
  (correctReassemblings : ℚ) / totalReassemblings =
  (correctReassemblings : ℚ) / totalReassemblings :=
by
  sorry

end NUMINAMATH_CALUDE_correct_reassembly_probability_l1315_131524


namespace NUMINAMATH_CALUDE_maintenance_check_interval_l1315_131573

theorem maintenance_check_interval (original_interval : ℝ) : 
  (original_interval * 1.2 = 60) → original_interval = 50 := by
  sorry

end NUMINAMATH_CALUDE_maintenance_check_interval_l1315_131573


namespace NUMINAMATH_CALUDE_min_sum_squares_l1315_131576

theorem min_sum_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 8) :
  ∃ (m : ℝ), (∀ a b c : ℝ, a^3 + b^3 + c^3 - 3*a*b*c = 8 → a^2 + b^2 + c^2 ≥ m) ∧
             (x^2 + y^2 + z^2 = m) ∧
             (m = 40/7) :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1315_131576


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_attained_l1315_131543

theorem min_value_expression (x : ℝ) :
  (15 - x) * (8 - x) * (15 + x) * (8 + x) ≥ -6480.25 :=
by sorry

theorem min_value_attained (ε : ℝ) (hε : ε > 0) :
  ∃ x : ℝ, (15 - x) * (8 - x) * (15 + x) * (8 + x) < -6480.25 + ε :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_attained_l1315_131543


namespace NUMINAMATH_CALUDE_distance_to_directrix_l1315_131561

/-- A parabola C with equation y² = 2px and a point A(1, √5) lying on it -/
structure Parabola where
  p : ℝ
  A : ℝ × ℝ
  h1 : A.1 = 1
  h2 : A.2 = Real.sqrt 5
  h3 : A.2^2 = 2 * p * A.1

/-- The distance from point A to the directrix of parabola C is 9/4 -/
theorem distance_to_directrix (C : Parabola) : 
  C.A.1 + C.p / 2 = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_directrix_l1315_131561


namespace NUMINAMATH_CALUDE_prime_condition_characterization_l1315_131595

def is_prime_for_all (a : ℕ+) : Prop :=
  ∀ n : ℕ, n < a → Nat.Prime (4 * n^2 + a)

theorem prime_condition_characterization :
  ∀ a : ℕ+, is_prime_for_all a ↔ (a = 3 ∨ a = 7) :=
sorry

end NUMINAMATH_CALUDE_prime_condition_characterization_l1315_131595


namespace NUMINAMATH_CALUDE_blue_paint_cans_l1315_131546

def paint_mixture (total_cans : ℕ) (blue_ratio green_ratio : ℕ) : ℕ := 
  (blue_ratio * total_cans) / (blue_ratio + green_ratio)

theorem blue_paint_cans : paint_mixture 45 4 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_blue_paint_cans_l1315_131546


namespace NUMINAMATH_CALUDE_division_remainder_and_divisibility_l1315_131526

theorem division_remainder_and_divisibility : 
  let dividend : ℕ := 1234567
  let divisor : ℕ := 256
  let remainder : ℕ := dividend % divisor
  remainder = 933 ∧ ¬(∃ k : ℕ, remainder = 7 * k) := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_and_divisibility_l1315_131526


namespace NUMINAMATH_CALUDE_factoring_expression_l1315_131591

theorem factoring_expression (a b : ℝ) : 6 * a^2 * b + 2 * a = 2 * a * (3 * a * b + 1) := by
  sorry

end NUMINAMATH_CALUDE_factoring_expression_l1315_131591


namespace NUMINAMATH_CALUDE_count_figures_l1315_131501

/-- The number of large triangles in Figure 1 -/
def large_triangles : ℕ := 8

/-- The number of medium triangles in Figure 1 -/
def medium_triangles : ℕ := 4

/-- The number of small triangles in Figure 1 -/
def small_triangles : ℕ := 4

/-- The number of small squares (1x1) in Figure 2 -/
def small_squares : ℕ := 20

/-- The number of medium squares (2x2) in Figure 2 -/
def medium_squares : ℕ := 10

/-- The number of large squares (3x3) in Figure 2 -/
def large_squares : ℕ := 4

/-- The number of largest squares (4x4) in Figure 2 -/
def largest_square : ℕ := 1

/-- Theorem stating the total number of triangles in Figure 1 and squares in Figure 2 -/
theorem count_figures :
  (large_triangles + medium_triangles + small_triangles = 16) ∧
  (small_squares + medium_squares + large_squares + largest_square = 35) := by
  sorry

end NUMINAMATH_CALUDE_count_figures_l1315_131501


namespace NUMINAMATH_CALUDE_becky_anna_size_ratio_l1315_131564

/-- Theorem: Given the sizes of Anna, Becky, and Ginger, prove the ratio of Becky's to Anna's size --/
theorem becky_anna_size_ratio :
  ∀ (anna_size becky_size ginger_size : ℕ),
  anna_size = 2 →
  ∃ k : ℕ, becky_size = k * anna_size →
  ginger_size = 2 * becky_size - 4 →
  ginger_size = 8 →
  becky_size / anna_size = 3 := by
sorry

end NUMINAMATH_CALUDE_becky_anna_size_ratio_l1315_131564


namespace NUMINAMATH_CALUDE_distribute_seven_to_twelve_l1315_131505

/-- The number of ways to distribute distinct items to recipients -/
def distribute_items (num_items : ℕ) (num_recipients : ℕ) : ℕ :=
  num_recipients ^ num_items

/-- Theorem: Distributing 7 distinct items to 12 recipients results in 35,831,808 ways -/
theorem distribute_seven_to_twelve :
  distribute_items 7 12 = 35831808 := by
  sorry

end NUMINAMATH_CALUDE_distribute_seven_to_twelve_l1315_131505


namespace NUMINAMATH_CALUDE_z_coordinate_for_x_7_l1315_131520

/-- A line passing through two points in 3D space -/
structure Line3D where
  point1 : ℝ × ℝ × ℝ
  point2 : ℝ × ℝ × ℝ

/-- Given a line and an x-coordinate, find the corresponding z-coordinate -/
def find_z_coordinate (line : Line3D) (x : ℝ) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem z_coordinate_for_x_7 :
  let line := Line3D.mk (1, 3, 2) (4, 4, -1)
  find_z_coordinate line 7 = -4 := by sorry

end NUMINAMATH_CALUDE_z_coordinate_for_x_7_l1315_131520


namespace NUMINAMATH_CALUDE_function_equality_proof_l1315_131570

theorem function_equality_proof (f : ℝ → ℝ) 
  (h₁ : ∀ x, x > 0 → f x > 0)
  (h₂ : ∀ x₁ x₂, x₁ > 0 → x₂ > 0 → 
    |x₁ * f x₂ - x₂ * f x₁| = (f x₁ + f x₂) * (x₂ - x₁)) :
  ∃ c : ℝ, c > 0 ∧ ∀ x, x > 0 → f x = c / x :=
sorry

end NUMINAMATH_CALUDE_function_equality_proof_l1315_131570


namespace NUMINAMATH_CALUDE_inverse_proportion_exists_l1315_131582

theorem inverse_proportion_exists (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : x₁ < 0) (h2 : 0 < x₂) (h3 : y₁ > y₂) : 
  ∃ k : ℝ, k < 0 ∧ y₁ = k / x₁ ∧ y₂ = k / x₂ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_exists_l1315_131582


namespace NUMINAMATH_CALUDE_smallest_book_count_l1315_131545

theorem smallest_book_count (b : ℕ) : 
  (b % 6 = 5) ∧ (b % 8 = 7) ∧ (b % 9 = 2) → 
  (∀ n : ℕ, n < b → ¬((n % 6 = 5) ∧ (n % 8 = 7) ∧ (n % 9 = 2))) → 
  b = 119 := by
sorry

end NUMINAMATH_CALUDE_smallest_book_count_l1315_131545


namespace NUMINAMATH_CALUDE_balance_balls_l1315_131553

-- Define the weights of balls relative to blue balls
def red_weight : ℚ := 2
def orange_weight : ℚ := 7/3
def silver_weight : ℚ := 5/3

-- Theorem statement
theorem balance_balls :
  5 * red_weight + 3 * orange_weight + 4 * silver_weight = 71/3 := by
  sorry

end NUMINAMATH_CALUDE_balance_balls_l1315_131553


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_of_roots_l1315_131514

theorem sum_of_fourth_powers_of_roots (p q r s : ℂ) : 
  (p^4 - p^3 + p^2 - 3*p + 3 = 0) →
  (q^4 - q^3 + q^2 - 3*q + 3 = 0) →
  (r^4 - r^3 + r^2 - 3*r + 3 = 0) →
  (s^4 - s^3 + s^2 - 3*s + 3 = 0) →
  p^4 + q^4 + r^4 + s^4 = 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_of_roots_l1315_131514


namespace NUMINAMATH_CALUDE_baseball_bat_price_baseball_bat_price_is_10_l1315_131536

/-- Calculates the selling price of a baseball bat given the total revenue and prices of other items -/
theorem baseball_bat_price (total_revenue : ℝ) (cards_price : ℝ) (glove_original_price : ℝ) (glove_discount : ℝ) (cleats_price : ℝ) (cleats_quantity : ℕ) : ℝ :=
  let glove_price := glove_original_price * (1 - glove_discount)
  let known_revenue := cards_price + glove_price + (cleats_price * cleats_quantity)
  total_revenue - known_revenue

/-- Proves that the baseball bat price is $10 given the specific conditions -/
theorem baseball_bat_price_is_10 :
  baseball_bat_price 79 25 30 0.2 10 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_baseball_bat_price_baseball_bat_price_is_10_l1315_131536


namespace NUMINAMATH_CALUDE_task_completion_probability_l1315_131583

theorem task_completion_probability (p1 p2 : ℚ) (h1 : p1 = 3/8) (h2 : p2 = 3/5) :
  p1 * (1 - p2) = 3/20 := by
  sorry

end NUMINAMATH_CALUDE_task_completion_probability_l1315_131583


namespace NUMINAMATH_CALUDE_impossible_equal_sum_configuration_l1315_131544

/-- Represents a configuration of digits placed in circles -/
structure DigitConfiguration where
  placement : Fin 10 → Fin 10
  bijective : Function.Bijective placement

/-- Represents a segment in the configuration -/
inductive Segment
| S1 | S2 | S3 | S4 | S5 | S6

/-- Returns the three positions (as Fin 10) for a given segment -/
def segmentPositions (s : Segment) : Fin 3 → Fin 10 :=
  sorry -- Implementation details omitted for simplicity

/-- Calculates the sum of digits on a segment for a given configuration -/
def segmentSum (config : DigitConfiguration) (s : Segment) : Nat :=
  (segmentPositions s 0).val + (segmentPositions s 1).val + (segmentPositions s 2).val

/-- Theorem stating the impossibility of the required configuration -/
theorem impossible_equal_sum_configuration :
  ¬ ∃ (config : DigitConfiguration),
    ∀ (s1 s2 : Segment), segmentSum config s1 = segmentSum config s2 :=
  sorry

end NUMINAMATH_CALUDE_impossible_equal_sum_configuration_l1315_131544


namespace NUMINAMATH_CALUDE_solution_in_interval_one_two_l1315_131580

theorem solution_in_interval_one_two :
  ∃ x : ℝ, 2^x + x = 4 ∧ x ∈ Set.Icc 1 2 := by sorry

end NUMINAMATH_CALUDE_solution_in_interval_one_two_l1315_131580


namespace NUMINAMATH_CALUDE_symmetry_about_origin_l1315_131518

-- Define a function f and its inverse
variable (f : ℝ → ℝ)
variable (f_inv : ℝ → ℝ)

-- Define the property of f_inv being the inverse of f
axiom inverse_relation : ∀ x, f_inv (f x) = x ∧ f (f_inv x) = x

-- Define the third function g
def g (x : ℝ) : ℝ := -f_inv (-x)

-- Theorem stating that g is symmetric to f_inv about the origin
theorem symmetry_about_origin :
  ∀ x, g (-x) = -g x :=
sorry

end NUMINAMATH_CALUDE_symmetry_about_origin_l1315_131518


namespace NUMINAMATH_CALUDE_triangle_cosine_inequality_l1315_131547

theorem triangle_cosine_inequality (A B C : ℝ) (h_non_obtuse : 0 ≤ A ∧ 0 ≤ B ∧ 0 ≤ C ∧ A + B + C = π) :
  (1 - Real.cos (2 * A)) * (1 - Real.cos (2 * B)) / (1 - Real.cos (2 * C)) +
  (1 - Real.cos (2 * C)) * (1 - Real.cos (2 * A)) / (1 - Real.cos (2 * B)) +
  (1 - Real.cos (2 * B)) * (1 - Real.cos (2 * C)) / (1 - Real.cos (2 * A)) ≥ 9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_inequality_l1315_131547


namespace NUMINAMATH_CALUDE_no_base6_digit_divisible_by_7_l1315_131552

theorem no_base6_digit_divisible_by_7 : 
  ¬ ∃ (d : ℕ), d < 6 ∧ (869 + 42 * d) % 7 = 0 := by
  sorry

#check no_base6_digit_divisible_by_7

end NUMINAMATH_CALUDE_no_base6_digit_divisible_by_7_l1315_131552


namespace NUMINAMATH_CALUDE_remainder_problem_l1315_131584

theorem remainder_problem (d r : ℤ) : 
  d > 1 → 
  2024 % d = r → 
  3250 % d = r → 
  4330 % d = r → 
  d - r = 2 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l1315_131584


namespace NUMINAMATH_CALUDE_p_and_q_sufficient_not_necessary_for_not_p_false_l1315_131559

theorem p_and_q_sufficient_not_necessary_for_not_p_false (p q : Prop) :
  (∃ (p q : Prop), (p ∧ q → ¬¬p) ∧ ¬(¬¬p → p ∧ q)) :=
sorry

end NUMINAMATH_CALUDE_p_and_q_sufficient_not_necessary_for_not_p_false_l1315_131559


namespace NUMINAMATH_CALUDE_regular_27gon_trapezoid_l1315_131597

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- A trapezoid is a quadrilateral with at least one pair of parallel sides -/
def IsTrapezoid (a b c d : ℝ × ℝ) : Prop := sorry

/-- Main theorem: Among any 7 vertices of a regular 27-gon, 4 can be selected that form a trapezoid -/
theorem regular_27gon_trapezoid (P : RegularPolygon 27) 
  (S : Finset (Fin 27)) (hS : S.card = 7) : 
  ∃ (a b c d : Fin 27), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ 
    IsTrapezoid (P.vertices a) (P.vertices b) (P.vertices c) (P.vertices d) :=
sorry

end NUMINAMATH_CALUDE_regular_27gon_trapezoid_l1315_131597


namespace NUMINAMATH_CALUDE_problem_statement_l1315_131522

theorem problem_statement (a b : ℝ) :
  (4 / (Real.sqrt 6 + Real.sqrt 2) - 1 / (Real.sqrt 3 + Real.sqrt 2) = Real.sqrt a - Real.sqrt b) →
  a - b = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1315_131522


namespace NUMINAMATH_CALUDE_parallel_perpendicular_lines_l1315_131512

/-- Given a point A and a line l, find the equations of lines passing through A
    that are parallel and perpendicular to l. -/
theorem parallel_perpendicular_lines
  (A : ℝ × ℝ)
  (l : ℝ → ℝ → Prop)
  (h_A : A = (2, 2))
  (h_l : l = fun x y ↦ 3 * x + 4 * y - 20 = 0) :
  ∃ (l_parallel l_perpendicular : ℝ → ℝ → Prop),
    (∀ x y, l_parallel x y ↔ 3 * x + 4 * y - 14 = 0) ∧
    (∀ x y, l_perpendicular x y ↔ 4 * x - 3 * y - 2 = 0) ∧
    (∀ x y, l_parallel x y → l_parallel A.1 A.2) ∧
    (∀ x y, l_perpendicular x y → l_perpendicular A.1 A.2) ∧
    (∀ x₁ y₁ x₂ y₂, l x₁ y₁ → l x₂ y₂ → (y₂ - y₁) * 4 = (x₂ - x₁) * 3) ∧
    (∀ x₁ y₁ x₂ y₂, l_parallel x₁ y₁ → l_parallel x₂ y₂ → (y₂ - y₁) * 4 = (x₂ - x₁) * 3) ∧
    (∀ x₁ y₁ x₂ y₂, l x₁ y₁ → l_perpendicular x₂ y₂ → (y₂ - y₁) * 3 = -(x₂ - x₁) * 4) :=
by sorry

end NUMINAMATH_CALUDE_parallel_perpendicular_lines_l1315_131512
