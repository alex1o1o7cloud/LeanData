import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1854_185488

theorem sum_of_squares_of_roots (p q r : ℝ) : 
  (3 * p^3 - 2 * p^2 + 6 * p + 15 = 0) →
  (3 * q^3 - 2 * q^2 + 6 * q + 15 = 0) →
  (3 * r^3 - 2 * r^2 + 6 * r + 15 = 0) →
  p^2 + q^2 + r^2 = -32/9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1854_185488


namespace NUMINAMATH_CALUDE_seven_classes_tournament_l1854_185422

/-- Calculate the number of matches in a round-robin tournament -/
def numberOfMatches (n : ℕ) : ℕ :=
  n * (n - 1) / 2

/-- Theorem: For 7 classes in a round-robin tournament, the total number of matches is 21 -/
theorem seven_classes_tournament : numberOfMatches 7 = 21 := by
  sorry

end NUMINAMATH_CALUDE_seven_classes_tournament_l1854_185422


namespace NUMINAMATH_CALUDE_percent_y_of_x_l1854_185459

theorem percent_y_of_x (x y : ℝ) (h : (1/2) * (x - y) = (1/5) * (x + y)) :
  y = (3/7) * x := by
  sorry

end NUMINAMATH_CALUDE_percent_y_of_x_l1854_185459


namespace NUMINAMATH_CALUDE_largest_valid_number_l1854_185485

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧  -- three-digit integer
  (n / 100 = 8) ∧  -- starts with 8
  (∀ d, d ≠ 0 ∧ d ∈ [n / 100, (n / 10) % 10, n % 10] → n % d = 0)  -- divisible by each distinct, non-zero digit

theorem largest_valid_number :
  is_valid_number 864 ∧ ∀ n, is_valid_number n → n ≤ 864 :=
sorry

end NUMINAMATH_CALUDE_largest_valid_number_l1854_185485


namespace NUMINAMATH_CALUDE_prime_squares_end_in_nine_l1854_185452

theorem prime_squares_end_in_nine :
  ∀ p q : ℕ,
  Prime p → Prime q →
  (p * p + q * q) % 10 = 9 →
  ((p = 2 ∧ q = 5) ∨ (p = 5 ∧ q = 2)) := by
sorry

end NUMINAMATH_CALUDE_prime_squares_end_in_nine_l1854_185452


namespace NUMINAMATH_CALUDE_gmat_test_problem_l1854_185406

theorem gmat_test_problem (second_correct : ℝ) (neither_correct : ℝ) (both_correct : ℝ)
  (h1 : second_correct = 65)
  (h2 : neither_correct = 5)
  (h3 : both_correct = 55) :
  100 - neither_correct - (second_correct - both_correct) = 85 :=
by
  sorry

end NUMINAMATH_CALUDE_gmat_test_problem_l1854_185406


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l1854_185431

/-- Given a hyperbola C with equation x²/a² - y²/b² = 1 where a, b > 0,
    focal length 10, and point P(3, 4) on one of its asymptotes,
    prove that the standard equation of C is x²/9 - y²/16 = 1 -/
theorem hyperbola_standard_equation 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hf : (2 : ℝ) * Real.sqrt (a^2 + b^2) = 10) 
  (hp : (3 : ℝ)^2 / a^2 - (4 : ℝ)^2 / b^2 = 0) :
  ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 / 9 - y^2 / 16 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l1854_185431


namespace NUMINAMATH_CALUDE_parabola_intersection_l1854_185404

-- Define the two parabolas
def f (x : ℝ) : ℝ := 3*x^2 + 4*x - 5
def g (x : ℝ) : ℝ := x^2 + 11

-- Theorem stating the intersection points
theorem parabola_intersection :
  (∃ (x y : ℝ), f x = g x ∧ y = f x) ↔
  (∃ (x y : ℝ), (x = -4 ∧ y = 27) ∨ (x = 2 ∧ y = 15)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_l1854_185404


namespace NUMINAMATH_CALUDE_sum_of_polynomials_l1854_185481

-- Define the polynomials
def f (x : ℝ) : ℝ := -6 * x^3 - 4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -7 * x^2 + 6 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 7 * x + 3

-- State the theorem
theorem sum_of_polynomials (x : ℝ) : 
  f x + g x + h x = -6 * x^3 - 5 * x^2 + 15 * x - 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_polynomials_l1854_185481


namespace NUMINAMATH_CALUDE_payment_difference_l1854_185412

def original_price : Float := 42.00000000000004
def discount_rate : Float := 0.10
def tip_rate : Float := 0.15

def discounted_price : Float := original_price * (1 - discount_rate)

def john_payment : Float := original_price + (original_price * tip_rate)
def jane_payment : Float := discounted_price + (discounted_price * tip_rate)

theorem payment_difference : john_payment - jane_payment = 4.830000000000005 := by
  sorry

end NUMINAMATH_CALUDE_payment_difference_l1854_185412


namespace NUMINAMATH_CALUDE_concert_revenue_l1854_185461

theorem concert_revenue (total_attendees : ℕ) (reserved_price unreserved_price : ℚ)
  (reserved_sold unreserved_sold : ℕ) :
  total_attendees = reserved_sold + unreserved_sold →
  reserved_price = 25 →
  unreserved_price = 20 →
  reserved_sold = 246 →
  unreserved_sold = 246 →
  (reserved_sold : ℚ) * reserved_price + (unreserved_sold : ℚ) * unreserved_price = 11070 :=
by sorry

end NUMINAMATH_CALUDE_concert_revenue_l1854_185461


namespace NUMINAMATH_CALUDE_floor_abs_sum_l1854_185438

theorem floor_abs_sum : ⌊|(-7.9 : ℝ)|⌋ + |⌊(-7.9 : ℝ)⌋| = 15 := by sorry

end NUMINAMATH_CALUDE_floor_abs_sum_l1854_185438


namespace NUMINAMATH_CALUDE_box_tie_length_l1854_185468

/-- Calculates the length of string used to tie a box given the initial length,
    the amount given away, and the fraction of the remainder used. -/
def string_used_for_box (initial_length : ℝ) (given_away : ℝ) (fraction_used : ℚ) : ℝ :=
  (initial_length - given_away) * (fraction_used : ℝ)

/-- Proves that given a string of 90 cm, after removing 30 cm, and using 8/15 of the remainder,
    the length used to tie the box is 32 cm. -/
theorem box_tie_length : 
  string_used_for_box 90 30 (8/15) = 32 := by sorry

end NUMINAMATH_CALUDE_box_tie_length_l1854_185468


namespace NUMINAMATH_CALUDE_ghost_entry_exit_ways_l1854_185437

def num_windows : ℕ := 8

theorem ghost_entry_exit_ways :
  (num_windows : ℕ) * (num_windows - 1) = 56 := by
  sorry

end NUMINAMATH_CALUDE_ghost_entry_exit_ways_l1854_185437


namespace NUMINAMATH_CALUDE_roger_trips_l1854_185469

def trays_per_trip : ℕ := 4
def total_trays : ℕ := 12

theorem roger_trips : (total_trays + trays_per_trip - 1) / trays_per_trip = 3 := by
  sorry

end NUMINAMATH_CALUDE_roger_trips_l1854_185469


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1854_185476

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, ∃ r : ℝ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_property (a : ℕ → ℝ) :
  geometric_sequence a →
  a 5 + a 7 = 2 * Real.pi →
  a 6 * (a 4 + 2 * a 6 + a 8) = 4 * Real.pi^2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1854_185476


namespace NUMINAMATH_CALUDE_jelly_beans_in_jar_X_l1854_185491

/-- The number of jelly beans in jar X -/
def jarX (total : ℕ) (y : ℕ) : ℕ := 3 * y - 400

/-- The total number of jelly beans in both jars -/
def totalBeans (x y : ℕ) : ℕ := x + y

theorem jelly_beans_in_jar_X :
  ∃ (y : ℕ), totalBeans (jarX 1200 y) y = 1200 ∧ jarX 1200 y = 800 := by
  sorry

end NUMINAMATH_CALUDE_jelly_beans_in_jar_X_l1854_185491


namespace NUMINAMATH_CALUDE_absolute_value_not_always_zero_l1854_185467

theorem absolute_value_not_always_zero : ¬ (∀ x : ℝ, |x| = 0) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_not_always_zero_l1854_185467


namespace NUMINAMATH_CALUDE_factor_calculation_l1854_185492

theorem factor_calculation (n f : ℝ) : n = 121 ∧ n * f - 140 = 102 → f = 2 := by
  sorry

end NUMINAMATH_CALUDE_factor_calculation_l1854_185492


namespace NUMINAMATH_CALUDE_equation_solution_l1854_185458

theorem equation_solution : 
  ∃ x : ℝ, (x + 6) / (x - 3) = 4 ∧ x = 6 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1854_185458


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_three_halves_l1854_185450

theorem sqrt_expression_equals_three_halves :
  (Real.sqrt 48 + (1/4) * Real.sqrt 12) / Real.sqrt 27 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_three_halves_l1854_185450


namespace NUMINAMATH_CALUDE_two_points_at_distance_from_line_l1854_185405

/-- A line in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Distance between a point and a line in 3D space -/
def distance_point_to_line (p : ℝ × ℝ × ℝ) (l : Line3D) : ℝ :=
  sorry

/-- Check if a line segment is perpendicular to a line in 3D space -/
def is_perpendicular (p1 p2 : ℝ × ℝ × ℝ) (l : Line3D) : Prop :=
  sorry

theorem two_points_at_distance_from_line 
  (L : Line3D) (d : ℝ) (P : ℝ × ℝ × ℝ) :
  ∃ (Q1 Q2 : ℝ × ℝ × ℝ),
    distance_point_to_line Q1 L = d ∧
    distance_point_to_line Q2 L = d ∧
    is_perpendicular P Q1 L ∧
    is_perpendicular P Q2 L :=
  sorry

end NUMINAMATH_CALUDE_two_points_at_distance_from_line_l1854_185405


namespace NUMINAMATH_CALUDE_book_price_percentage_l1854_185443

theorem book_price_percentage (suggested_retail_price : ℝ) : 
  suggested_retail_price > 0 →
  let marked_price := 0.6 * suggested_retail_price
  let alice_paid := 0.6 * marked_price
  alice_paid / suggested_retail_price = 0.36 := by
  sorry

end NUMINAMATH_CALUDE_book_price_percentage_l1854_185443


namespace NUMINAMATH_CALUDE_palmer_photos_l1854_185415

theorem palmer_photos (initial_photos : ℕ) (final_photos : ℕ) (third_fourth_week_photos : ℕ) :
  initial_photos = 100 →
  final_photos = 380 →
  third_fourth_week_photos = 80 →
  ∃ (first_week_photos : ℕ),
    first_week_photos = 67 ∧
    final_photos = initial_photos + first_week_photos + 2 * first_week_photos + third_fourth_week_photos :=
by sorry

end NUMINAMATH_CALUDE_palmer_photos_l1854_185415


namespace NUMINAMATH_CALUDE_certain_number_value_l1854_185418

def is_smallest_multiplier (n : ℕ) (x : ℝ) : Prop :=
  n > 0 ∧ ∃ (y : ℕ), n * x = y^2 ∧
  ∀ (m : ℕ), m > 0 → m < n → ¬∃ (z : ℕ), m * x = z^2

theorem certain_number_value (n : ℕ) (x : ℝ) :
  is_smallest_multiplier n x → n = 3 → x = 1 :=
by sorry

end NUMINAMATH_CALUDE_certain_number_value_l1854_185418


namespace NUMINAMATH_CALUDE_total_time_is_34_hours_l1854_185494

/-- Calculates the total time spent on drawing and coloring pictures. -/
def total_time (num_pictures : ℕ) (draw_time : ℝ) (color_time_reduction : ℝ) : ℝ :=
  let color_time := draw_time * (1 - color_time_reduction)
  num_pictures * (draw_time + color_time)

/-- Proves that the total time spent on all pictures is 34 hours. -/
theorem total_time_is_34_hours :
  total_time 10 2 0.3 = 34 := by
  sorry

end NUMINAMATH_CALUDE_total_time_is_34_hours_l1854_185494


namespace NUMINAMATH_CALUDE_root_implies_k_value_l1854_185435

theorem root_implies_k_value (k : ℝ) : 
  (3 : ℝ)^4 + k * (3 : ℝ)^2 + 27 = 0 → k = -12 := by
sorry

end NUMINAMATH_CALUDE_root_implies_k_value_l1854_185435


namespace NUMINAMATH_CALUDE_complex_powers_sum_l1854_185498

theorem complex_powers_sum : 
  (((Complex.I * Real.sqrt 3 - 1) / 2) ^ 6 + ((Complex.I * Real.sqrt 3 + 1) / (-2)) ^ 6 = 2) ∧
  (∀ n : ℕ, Odd n → ((Complex.I + 1) / Real.sqrt 2) ^ (4 * n) + ((1 - Complex.I) / Real.sqrt 2) ^ (4 * n) = -2) :=
by sorry

end NUMINAMATH_CALUDE_complex_powers_sum_l1854_185498


namespace NUMINAMATH_CALUDE_cube_diff_divisibility_l1854_185402

theorem cube_diff_divisibility (a b : ℤ) (n : ℕ) 
  (ha : Odd a) (hb : Odd b) : 
  (2^n : ℤ) ∣ (a^3 - b^3) ↔ (2^n : ℤ) ∣ (a - b) := by
  sorry

end NUMINAMATH_CALUDE_cube_diff_divisibility_l1854_185402


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l1854_185463

theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) (h1 : x * y = k) 
  (h2 : ∃ x0 y0, x0 + y0 = 36 ∧ x0 = 3 * y0 ∧ x0 * y0 = k) :
  x = -6 → y = -40.5 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l1854_185463


namespace NUMINAMATH_CALUDE_hash_property_l1854_185479

/-- Operation # for non-negative integers -/
def hash (a b : ℕ) : ℕ := 4 * a^2 + 4 * b^2 + 8 * a * b

/-- Theorem: If a # b = 100, then (a + b) + 5 = 10 for non-negative integers a and b -/
theorem hash_property (a b : ℕ) (h : hash a b = 100) : (a + b) + 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_hash_property_l1854_185479


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_l1854_185451

theorem complex_magnitude_equation (a : ℝ) : 
  Complex.abs ((1 + Complex.I) / (a * Complex.I)) = Real.sqrt 2 → a = 1 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_l1854_185451


namespace NUMINAMATH_CALUDE_max_value_sum_of_reciprocals_l1854_185489

theorem max_value_sum_of_reciprocals (a b : ℝ) (h : a + b = 2) :
  (∃ (x : ℝ), ∀ (y : ℝ), (1 / (a^2 + 1) + 1 / (b^2 + 1)) ≤ y) ∧
  (∀ (ε : ℝ), ε > 0 → ∃ (a' b' : ℝ), a' + b' = 2 ∧ 
    (1 / (a'^2 + 1) + 1 / (b'^2 + 1)) > (Real.sqrt 2 + 1) / 2 - ε) :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_of_reciprocals_l1854_185489


namespace NUMINAMATH_CALUDE_power_windows_count_l1854_185417

theorem power_windows_count (total : ℕ) (power_steering : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 65)
  (h2 : power_steering = 45)
  (h3 : both = 17)
  (h4 : neither = 12) :
  total - neither - (power_steering - both) = 25 := by
  sorry

end NUMINAMATH_CALUDE_power_windows_count_l1854_185417


namespace NUMINAMATH_CALUDE_range_of_a_for_false_proposition_l1854_185432

theorem range_of_a_for_false_proposition :
  (∀ x ∈ Set.Icc 0 1, 2 * x + a ≥ 0) ↔ a > 0 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_for_false_proposition_l1854_185432


namespace NUMINAMATH_CALUDE_most_frequent_is_mode_l1854_185475

/-- The mode of a dataset is the value that appears most frequently. -/
def mode (dataset : Multiset α) [DecidableEq α] : Set α :=
  {x | ∀ y, dataset.count x ≥ dataset.count y}

/-- The most frequent data in a dataset is the mode. -/
theorem most_frequent_is_mode (dataset : Multiset α) [DecidableEq α] :
  ∀ x ∈ mode dataset, ∀ y, dataset.count x ≥ dataset.count y :=
sorry

end NUMINAMATH_CALUDE_most_frequent_is_mode_l1854_185475


namespace NUMINAMATH_CALUDE_exists_term_divisible_by_2006_l1854_185474

theorem exists_term_divisible_by_2006 : ∃ n : ℤ, (2006 : ℤ) ∣ (n^3 - (2*n + 1)^2) := by
  sorry

end NUMINAMATH_CALUDE_exists_term_divisible_by_2006_l1854_185474


namespace NUMINAMATH_CALUDE_pi_power_zero_plus_two_power_neg_two_l1854_185421

theorem pi_power_zero_plus_two_power_neg_two :
  (-Real.pi)^(0 : ℤ) + 2^(-2 : ℤ) = 5/4 := by sorry

end NUMINAMATH_CALUDE_pi_power_zero_plus_two_power_neg_two_l1854_185421


namespace NUMINAMATH_CALUDE_tangent_sum_product_l1854_185441

theorem tangent_sum_product (α β : ℝ) : 
  let γ := Real.arctan (-Real.tan (α + β))
  Real.tan α + Real.tan β + Real.tan γ = Real.tan α * Real.tan β * Real.tan γ := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_product_l1854_185441


namespace NUMINAMATH_CALUDE_sarah_brother_apple_ratio_l1854_185428

def sarah_apples : ℕ := 45
def brother_apples : ℕ := 9

theorem sarah_brother_apple_ratio :
  sarah_apples / brother_apples = 5 :=
sorry

end NUMINAMATH_CALUDE_sarah_brother_apple_ratio_l1854_185428


namespace NUMINAMATH_CALUDE_grid_removal_l1854_185447

theorem grid_removal (n : ℕ) (h : n ≥ 10) :
  ∀ (grid : Fin n → Fin n → Bool),
  (∃ (rows : Finset (Fin n)),
    rows.card = n - 10 ∧
    ∀ (j : Fin n), ∃ (i : Fin n), i ∉ rows ∧ grid i j = true) ∨
  (∃ (cols : Finset (Fin n)),
    cols.card = n - 10 ∧
    ∀ (i : Fin n), ∃ (j : Fin n), j ∉ cols ∧ grid i j = false) :=
sorry

end NUMINAMATH_CALUDE_grid_removal_l1854_185447


namespace NUMINAMATH_CALUDE_perfect_square_sum_l1854_185436

theorem perfect_square_sum (x y : ℕ) : 
  (∃ z : ℕ, 3^x + 7^y = z^2) → 
  Even y → 
  x = 1 ∧ y = 0 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_sum_l1854_185436


namespace NUMINAMATH_CALUDE_not_prime_2011_2111_plus_2500_l1854_185454

theorem not_prime_2011_2111_plus_2500 : ¬ Nat.Prime (2011 * 2111 + 2500) := by
  sorry

end NUMINAMATH_CALUDE_not_prime_2011_2111_plus_2500_l1854_185454


namespace NUMINAMATH_CALUDE_complex_absolute_value_l1854_185426

theorem complex_absolute_value (a : ℝ) : 
  let z₁ : ℂ := a + 2*I
  let z₂ : ℂ := 2 - I
  (∃ (b : ℝ), z₁ / z₂ = b*I) → Complex.abs z₁ = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l1854_185426


namespace NUMINAMATH_CALUDE_pizza_problem_l1854_185430

theorem pizza_problem (total_money : ℕ) (pizza_cost : ℕ) (bill_initial : ℕ) (bill_final : ℕ) :
  total_money = 42 →
  pizza_cost = 11 →
  bill_initial = 30 →
  bill_final = 39 →
  (total_money - (bill_final - bill_initial)) / pizza_cost = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_pizza_problem_l1854_185430


namespace NUMINAMATH_CALUDE_discount_is_ten_percent_l1854_185429

/-- Calculates the discount percentage on a retail price given wholesale price, retail price, and profit percentage. -/
def discount_percentage (wholesale_price retail_price profit_percentage : ℚ) : ℚ :=
  let profit := wholesale_price * profit_percentage
  let selling_price := wholesale_price + profit
  let discount_amount := retail_price - selling_price
  (discount_amount / retail_price) * 100

/-- Theorem stating that the discount percentage is 10% given the problem conditions. -/
theorem discount_is_ten_percent :
  discount_percentage 108 144 0.2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_discount_is_ten_percent_l1854_185429


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l1854_185414

/-- For the quadratic equation x^2 + 4x√2 + k = 0, prove that k = 8 makes the discriminant zero and the roots real and equal. -/
theorem quadratic_equal_roots (k : ℝ) : 
  (∀ x, x^2 + 4*x*Real.sqrt 2 + k = 0) →
  (k = 8 ↔ (∃! r, r^2 + 4*r*Real.sqrt 2 + k = 0)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l1854_185414


namespace NUMINAMATH_CALUDE_max_value_fraction_l1854_185471

theorem max_value_fraction (x : ℝ) : 
  (3 * x^2 + 9 * x + 20) / (3 * x^2 + 9 * x + 7) ≤ 53 ∧ 
  ∀ ε > 0, ∃ y : ℝ, (3 * y^2 + 9 * y + 20) / (3 * y^2 + 9 * y + 7) > 53 - ε :=
by sorry

end NUMINAMATH_CALUDE_max_value_fraction_l1854_185471


namespace NUMINAMATH_CALUDE_equation_solution_l1854_185462

theorem equation_solution : 
  ∃! x : ℝ, (2 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 4) ∧ x = -15 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1854_185462


namespace NUMINAMATH_CALUDE_only_zero_has_linear_factors_l1854_185413

/-- A polynomial in x and y with a parameter k -/
def poly (k : ℤ) (x y : ℤ) : ℤ := x^2 + 4*x*y + 2*x + k*y - k

/-- Predicate for linear factors with integer coefficients -/
def has_linear_integer_factors (k : ℤ) : Prop :=
  ∃ (a b c d e f : ℤ), ∀ (x y : ℤ),
    poly k x y = (a*x + b*y + c) * (d*x + e*y + f)

theorem only_zero_has_linear_factors :
  ∀ k : ℤ, has_linear_integer_factors k ↔ k = 0 :=
by sorry

end NUMINAMATH_CALUDE_only_zero_has_linear_factors_l1854_185413


namespace NUMINAMATH_CALUDE_zephyria_license_plates_l1854_185499

/-- The number of letters in the English alphabet -/
def num_letters : ℕ := 26

/-- The number of possible digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of letters in a Zephyrian license plate -/
def num_plate_letters : ℕ := 3

/-- The number of digits in a Zephyrian license plate -/
def num_plate_digits : ℕ := 4

/-- The total number of possible valid license plates in Zephyria -/
def total_license_plates : ℕ := num_letters ^ num_plate_letters * num_digits ^ num_plate_digits

theorem zephyria_license_plates :
  total_license_plates = 175760000 :=
by sorry

end NUMINAMATH_CALUDE_zephyria_license_plates_l1854_185499


namespace NUMINAMATH_CALUDE_digit_puzzle_solutions_l1854_185434

def is_valid_solution (a b : ℕ) : Prop :=
  a ≠ b ∧
  a < 10 ∧ b < 10 ∧
  10 ≤ 10 * b + a ∧ 10 * b + a < 100 ∧
  10 * b + a ≠ a * b ∧
  a ^ b = 10 * b + a

theorem digit_puzzle_solutions :
  {(a, b) : ℕ × ℕ | is_valid_solution a b} =
  {(2, 5), (6, 2), (4, 3)} := by sorry

end NUMINAMATH_CALUDE_digit_puzzle_solutions_l1854_185434


namespace NUMINAMATH_CALUDE_ascending_order_of_powers_of_two_l1854_185487

theorem ascending_order_of_powers_of_two :
  let a := (2 : ℝ) ^ (1/3 : ℝ)
  let b := (2 : ℝ) ^ (3/8 : ℝ)
  let c := (2 : ℝ) ^ (2/5 : ℝ)
  let d := (2 : ℝ) ^ (4/9 : ℝ)
  let e := (2 : ℝ) ^ (1/2 : ℝ)
  a < b ∧ b < c ∧ c < d ∧ d < e := by sorry

end NUMINAMATH_CALUDE_ascending_order_of_powers_of_two_l1854_185487


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1854_185472

theorem partial_fraction_decomposition (x : ℝ) (h1 : x ≠ 7) (h2 : x ≠ -2) :
  (3 * x + 12) / (x^2 - 5*x - 14) = (11/3) / (x - 7) + (-2/3) / (x + 2) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1854_185472


namespace NUMINAMATH_CALUDE_shopkeeper_payment_l1854_185420

/-- The total cost of a purchase given the quantity and price per unit -/
def totalCost (quantity : ℕ) (pricePerUnit : ℕ) : ℕ :=
  quantity * pricePerUnit

/-- The problem statement -/
theorem shopkeeper_payment : 
  let grapeQuantity : ℕ := 8
  let grapePrice : ℕ := 70
  let mangoQuantity : ℕ := 9
  let mangoPrice : ℕ := 50
  let grapeCost := totalCost grapeQuantity grapePrice
  let mangoCost := totalCost mangoQuantity mangoPrice
  grapeCost + mangoCost = 1010 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_payment_l1854_185420


namespace NUMINAMATH_CALUDE_least_common_period_l1854_185470

-- Define the property of the function
def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 5) + f (x - 5) = f x

-- Define what it means for a function to be periodic with period p
def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

-- The main theorem
theorem least_common_period :
  ∃ p : ℝ, p > 0 ∧
  (∀ f : ℝ → ℝ, satisfies_condition f → is_periodic f p) ∧
  (∀ q : ℝ, 0 < q ∧ q < p →
    ∃ g : ℝ → ℝ, satisfies_condition g ∧ ¬is_periodic g q) ∧
  p = 30 :=
sorry

end NUMINAMATH_CALUDE_least_common_period_l1854_185470


namespace NUMINAMATH_CALUDE_divisor_for_5_pow_100_mod_13_l1854_185403

theorem divisor_for_5_pow_100_mod_13 (D : ℕ+) :
  (5^100 : ℕ) % D = 13 → D = 5^100 - 13 + 1 := by
sorry

end NUMINAMATH_CALUDE_divisor_for_5_pow_100_mod_13_l1854_185403


namespace NUMINAMATH_CALUDE_pushup_difference_l1854_185427

theorem pushup_difference (david_pushups : ℕ) (total_pushups : ℕ) (zachary_pushups : ℕ) :
  david_pushups = 51 →
  total_pushups = 53 →
  david_pushups > zachary_pushups →
  total_pushups = david_pushups + zachary_pushups →
  david_pushups - zachary_pushups = 49 := by
  sorry

end NUMINAMATH_CALUDE_pushup_difference_l1854_185427


namespace NUMINAMATH_CALUDE_number_difference_l1854_185409

theorem number_difference (x y : ℝ) (h1 : x + y = 25) (h2 : x * y = 144) : |x - y| = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l1854_185409


namespace NUMINAMATH_CALUDE_sum_of_factors_180_l1854_185419

/-- The sum of positive factors of a natural number n -/
def sum_of_factors (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of positive factors of 180 is 546 -/
theorem sum_of_factors_180 : sum_of_factors 180 = 546 := by sorry

end NUMINAMATH_CALUDE_sum_of_factors_180_l1854_185419


namespace NUMINAMATH_CALUDE_rain_on_tuesdays_l1854_185445

theorem rain_on_tuesdays 
  (monday_count : ℕ) 
  (tuesday_count : ℕ) 
  (rain_per_monday : ℝ) 
  (total_rain_difference : ℝ) 
  (h1 : monday_count = 7)
  (h2 : tuesday_count = 9)
  (h3 : rain_per_monday = 1.5)
  (h4 : total_rain_difference = 12) :
  (monday_count * rain_per_monday + total_rain_difference) / tuesday_count = 2.5 := by
sorry

end NUMINAMATH_CALUDE_rain_on_tuesdays_l1854_185445


namespace NUMINAMATH_CALUDE_comparison_of_powers_l1854_185493

theorem comparison_of_powers (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hab : a ≠ b) : 
  (a^a * b^b > a^b * b^a) ∧ 
  (a^a * b^b * c^c > (a*b*c)^((a+b+c)/3)) := by
  sorry

end NUMINAMATH_CALUDE_comparison_of_powers_l1854_185493


namespace NUMINAMATH_CALUDE_vector_equation_solution_parallel_vectors_solution_l1854_185444

/-- Given vectors in R^2 -/
def a : Fin 2 → ℚ := ![3, 2]
def b : Fin 2 → ℚ := ![-1, 2]
def c : Fin 2 → ℚ := ![4, 1]

/-- Part 1: Prove that m = 5/9 and n = 8/9 satisfy a = m*b + n*c -/
theorem vector_equation_solution :
  ∃ (m n : ℚ), (m = 5/9 ∧ n = 8/9) ∧ (∀ i : Fin 2, a i = m * b i + n * c i) :=
sorry

/-- Part 2: Prove that k = -16/13 makes (a + k*c) parallel to (2*b - a) -/
theorem parallel_vectors_solution :
  ∃ (k : ℚ), k = -16/13 ∧
  ∃ (t : ℚ), ∀ i : Fin 2, (a i + k * c i) = t * (2 * b i - a i) :=
sorry

end NUMINAMATH_CALUDE_vector_equation_solution_parallel_vectors_solution_l1854_185444


namespace NUMINAMATH_CALUDE_inequality_proof_l1854_185453

theorem inequality_proof (x y z : ℝ) 
  (h1 : y ≥ 2 * z) 
  (h2 : 2 * z ≥ 4 * x)
  (h3 : 2 * (x^3 + y^3 + z^3) + 15 * (x * y^2 + y * z^2 + z * x^2) ≥ 
        16 * (x^2 * y + y^2 * z + z^2 * x) + 2 * x * y * z) :
  4 * x + y ≥ 4 * z := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1854_185453


namespace NUMINAMATH_CALUDE_median_in_60_64_interval_l1854_185416

/-- Represents the score intervals --/
inductive ScoreInterval
| interval_45_49
| interval_50_54
| interval_55_59
| interval_60_64
| interval_65_69

/-- Represents the frequency of each score interval --/
def frequency : ScoreInterval → Nat
| ScoreInterval.interval_45_49 => 10
| ScoreInterval.interval_50_54 => 15
| ScoreInterval.interval_55_59 => 20
| ScoreInterval.interval_60_64 => 25
| ScoreInterval.interval_65_69 => 30

/-- The total number of students --/
def totalStudents : Nat := 100

/-- Function to calculate the cumulative frequency up to a given interval --/
def cumulativeFrequency (interval : ScoreInterval) : Nat :=
  match interval with
  | ScoreInterval.interval_45_49 => frequency ScoreInterval.interval_45_49
  | ScoreInterval.interval_50_54 => frequency ScoreInterval.interval_45_49 + frequency ScoreInterval.interval_50_54
  | ScoreInterval.interval_55_59 => frequency ScoreInterval.interval_45_49 + frequency ScoreInterval.interval_50_54 + frequency ScoreInterval.interval_55_59
  | ScoreInterval.interval_60_64 => frequency ScoreInterval.interval_45_49 + frequency ScoreInterval.interval_50_54 + frequency ScoreInterval.interval_55_59 + frequency ScoreInterval.interval_60_64
  | ScoreInterval.interval_65_69 => totalStudents

/-- The median position --/
def medianPosition : Nat := (totalStudents + 1) / 2

/-- Theorem stating that the median score falls in the interval 60-64 --/
theorem median_in_60_64_interval :
  cumulativeFrequency ScoreInterval.interval_55_59 < medianPosition ∧
  medianPosition ≤ cumulativeFrequency ScoreInterval.interval_60_64 :=
sorry

end NUMINAMATH_CALUDE_median_in_60_64_interval_l1854_185416


namespace NUMINAMATH_CALUDE_honda_production_l1854_185478

theorem honda_production (day_shift : ℕ) (second_shift : ℕ) : 
  day_shift = 4 * second_shift → 
  day_shift + second_shift = 5500 → 
  day_shift = 4400 := by
sorry

end NUMINAMATH_CALUDE_honda_production_l1854_185478


namespace NUMINAMATH_CALUDE_total_chips_eaten_l1854_185460

/-- 
Given that John eats x bags of chips for dinner and 2x bags after dinner,
prove that the total number of bags eaten is 3x.
-/
theorem total_chips_eaten (x : ℕ) : x + 2*x = 3*x := by
  sorry

end NUMINAMATH_CALUDE_total_chips_eaten_l1854_185460


namespace NUMINAMATH_CALUDE_average_problem_l1854_185464

theorem average_problem (x : ℝ) : 
  let numbers := [54, 55, 57, 58, 59, 62, 62, 63, x]
  (numbers.sum / numbers.length : ℝ) = 60 → x = 70 := by
sorry

end NUMINAMATH_CALUDE_average_problem_l1854_185464


namespace NUMINAMATH_CALUDE_club_membership_l1854_185486

theorem club_membership (total_members event_participants : ℕ) 
  (h_total : total_members = 30)
  (h_event : event_participants = 18)
  (h_participation : ∃ (men women : ℕ), 
    men + women = total_members ∧ 
    men + (women / 3) = event_participants) : 
  ∃ (men : ℕ), men = 12 ∧ 
    ∃ (women : ℕ), men + women = total_members ∧ 
      men + (women / 3) = event_participants :=
sorry

end NUMINAMATH_CALUDE_club_membership_l1854_185486


namespace NUMINAMATH_CALUDE_min_value_expression_l1854_185407

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ) (hx : x > 0) (hy : y > 0), 
    (|6*x - 4*y| + |3*(x + y*Real.sqrt 3) + 2*(x*Real.sqrt 3 - y)|) / Real.sqrt (x^2 + y^2) 
    ≤ (|6*a - 4*b| + |3*(a + b*Real.sqrt 3) + 2*(a*Real.sqrt 3 - b)|) / Real.sqrt (a^2 + b^2))
  ∧ 
  (∀ (x y : ℝ) (hx : x > 0) (hy : y > 0), 
    (|6*x - 4*y| + |3*(x + y*Real.sqrt 3) + 2*(x*Real.sqrt 3 - y)|) / Real.sqrt (x^2 + y^2) 
    ≥ Real.sqrt 39) := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1854_185407


namespace NUMINAMATH_CALUDE_combined_experience_is_68_l1854_185423

/-- Calculates the combined experience of James, John, and Mike -/
def combinedExperience (james_current : ℕ) (years_ago : ℕ) (john_multiplier : ℕ) (john_when_mike_started : ℕ) : ℕ :=
  let james_past := james_current - years_ago
  let john_past := john_multiplier * james_past
  let john_current := john_past + years_ago
  let mike_experience := john_current - john_when_mike_started
  james_current + john_current + mike_experience

/-- The combined experience of James, John, and Mike is 68 years -/
theorem combined_experience_is_68 :
  combinedExperience 20 8 2 16 = 68 := by
  sorry

end NUMINAMATH_CALUDE_combined_experience_is_68_l1854_185423


namespace NUMINAMATH_CALUDE_inverse_as_linear_combination_l1854_185495

theorem inverse_as_linear_combination (N : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : N = ![![3, 0], ![2, -4]]) : 
  ∃ (c d : ℝ), N⁻¹ = c • N + d • (1 : Matrix (Fin 2) (Fin 2) ℝ) ∧ 
  c = (1 : ℝ) / 12 ∧ d = (1 : ℝ) / 12 := by
sorry

end NUMINAMATH_CALUDE_inverse_as_linear_combination_l1854_185495


namespace NUMINAMATH_CALUDE_square_area_with_two_edge_representations_l1854_185490

theorem square_area_with_two_edge_representations (x : ℝ) :
  (3 * x - 12 = 18 - 2 * x) →
  (3 * x - 12)^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_area_with_two_edge_representations_l1854_185490


namespace NUMINAMATH_CALUDE_laptop_arrangement_impossible_l1854_185401

/-- Represents the number of laptops of each type in a row -/
structure LaptopRow :=
  (typeA : ℕ)
  (typeB : ℕ)
  (typeC : ℕ)

/-- The total number of laptops -/
def totalLaptops : ℕ := 44

/-- The number of rows -/
def numRows : ℕ := 5

/-- Checks if a LaptopRow satisfies the ratio condition -/
def satisfiesRatio (row : LaptopRow) : Prop :=
  3 * row.typeA = 2 * row.typeB ∧ 2 * row.typeC = 3 * row.typeB

/-- Checks if a LaptopRow has at least one of each type -/
def hasAllTypes (row : LaptopRow) : Prop :=
  row.typeA > 0 ∧ row.typeB > 0 ∧ row.typeC > 0

/-- Theorem stating the impossibility of the laptop arrangement -/
theorem laptop_arrangement_impossible : 
  ¬ ∃ (row : LaptopRow), 
    (row.typeA + row.typeB + row.typeC) * numRows = totalLaptops ∧
    satisfiesRatio row ∧
    hasAllTypes row :=
by sorry

end NUMINAMATH_CALUDE_laptop_arrangement_impossible_l1854_185401


namespace NUMINAMATH_CALUDE_min_zeros_odd_period_two_l1854_185439

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function f: ℝ → ℝ has period 2 if f(x) = f(x+2) for all x ∈ ℝ -/
def HasPeriodTwo (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (x + 2)

/-- The number of zeros of a function f: ℝ → ℝ in an interval [a, b] -/
def NumberOfZeros (f : ℝ → ℝ) (a b : ℝ) : ℕ :=
  sorry  -- Definition omitted for brevity

/-- The theorem stating the minimum number of zeros for an odd function with period 2 -/
theorem min_zeros_odd_period_two (f : ℝ → ℝ) (h_odd : IsOdd f) (h_period : HasPeriodTwo f) :
  NumberOfZeros f 0 2009 ≥ 2010 :=
sorry

end NUMINAMATH_CALUDE_min_zeros_odd_period_two_l1854_185439


namespace NUMINAMATH_CALUDE_sum_of_divisors_420_l1854_185424

def sum_of_divisors (n : ℕ) : ℕ := (Finset.filter (·∣n) (Finset.range (n + 1))).sum id

theorem sum_of_divisors_420 : sum_of_divisors 420 = 1344 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_420_l1854_185424


namespace NUMINAMATH_CALUDE_probability_point_between_B_and_E_l1854_185496

/-- Given a line segment AB with points A, B, C, D, and E such that AB = 4AD = 8BE = 2BC,
    the probability that a randomly chosen point on AB lies between B and E is 1/8. -/
theorem probability_point_between_B_and_E (A B C D E : ℝ) : 
  A < D ∧ D < E ∧ E < B ∧ B < C →  -- Points are ordered on the line
  (B - A) = 4 * (D - A) →          -- AB = 4AD
  (B - A) = 8 * (B - E) →          -- AB = 8BE
  (B - A) = 2 * (C - B) →          -- AB = 2BC
  (B - E) / (B - A) = 1 / 8 :=     -- Probability is 1/8
by sorry

end NUMINAMATH_CALUDE_probability_point_between_B_and_E_l1854_185496


namespace NUMINAMATH_CALUDE_opposite_face_is_E_l1854_185473

/-- Represents the faces of a cube --/
inductive Face
  | A | B | C | D | E | F

/-- Represents a cube --/
structure Cube where
  faces : List Face
  top : Face
  adjacent : Face → List Face
  opposite : Face → Face

/-- The cube configuration described in the problem --/
def problem_cube : Cube :=
  { faces := [Face.A, Face.B, Face.C, Face.D, Face.E, Face.F]
  , top := Face.F
  , adjacent := fun f => match f with
    | Face.D => [Face.A, Face.B, Face.C]
    | _ => sorry  -- We don't have information about other adjacencies
  , opposite := sorry  -- To be proven
  }

theorem opposite_face_is_E :
  problem_cube.opposite Face.A = Face.E :=
by sorry

end NUMINAMATH_CALUDE_opposite_face_is_E_l1854_185473


namespace NUMINAMATH_CALUDE_ab_value_l1854_185466

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 11) : a * b = 1 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l1854_185466


namespace NUMINAMATH_CALUDE_m_range_l1854_185425

-- Define the condition function
def condition (x m : ℝ) : Prop := x^2 + m*x - 2*m^2 < 0

-- Define the sufficient condition
def sufficient_condition (x : ℝ) : Prop := -2 < x ∧ x < 3

-- State the theorem
theorem m_range (m : ℝ) :
  (m > 0) →
  (∀ x, sufficient_condition x → condition x m) →
  (∃ x, condition x m ∧ ¬sufficient_condition x) →
  m ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_m_range_l1854_185425


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l1854_185448

/-- An arithmetic sequence with common difference d -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

/-- Three terms of a sequence form a geometric sequence -/
def geometric_sequence_terms (a b c : ℝ) : Prop :=
  b^2 = a * c

theorem arithmetic_geometric_ratio 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (h1 : d ≠ 0) 
  (h2 : arithmetic_sequence a d) 
  (h3 : geometric_sequence_terms (a 1) (a 3) (a 9)) : 
  3 * (a 3) / (a 16) = 9 / 16 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l1854_185448


namespace NUMINAMATH_CALUDE_percentage_problem_l1854_185483

theorem percentage_problem (N : ℝ) (h : 0.2 * N = 1000) : 1.2 * N = 6000 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1854_185483


namespace NUMINAMATH_CALUDE_impossibleConfig6_impossibleConfig4_impossibleConfig3_l1854_185484

/-- Represents the sign at a vertex -/
inductive Sign
| Positive
| Negative

/-- Represents a dodecagon configuration -/
def DodecagonConfig := Fin 12 → Sign

/-- Initial configuration with A₁ negative and others positive -/
def initialConfig : DodecagonConfig :=
  fun i => if i = 0 then Sign.Negative else Sign.Positive

/-- Applies the sign-flipping operation to n consecutive vertices starting at index i -/
def flipSigns (config : DodecagonConfig) (i : Fin 12) (n : Nat) : DodecagonConfig :=
  fun j => if (j - i) % 12 < n then
    match config j with
    | Sign.Positive => Sign.Negative
    | Sign.Negative => Sign.Positive
    else config j

/-- Checks if only A₂ is negative in the configuration -/
def onlyA₂Negative (config : DodecagonConfig) : Prop :=
  config 1 = Sign.Negative ∧ ∀ i, i ≠ 1 → config i = Sign.Positive

/-- Main theorem for 6 consecutive vertices -/
theorem impossibleConfig6 :
  ∀ (ops : List (Fin 12)), 
  let finalConfig := (ops.foldl (fun cfg i => flipSigns cfg i 6) initialConfig)
  ¬(onlyA₂Negative finalConfig) := by sorry

/-- Main theorem for 4 consecutive vertices -/
theorem impossibleConfig4 :
  ∀ (ops : List (Fin 12)), 
  let finalConfig := (ops.foldl (fun cfg i => flipSigns cfg i 4) initialConfig)
  ¬(onlyA₂Negative finalConfig) := by sorry

/-- Main theorem for 3 consecutive vertices -/
theorem impossibleConfig3 :
  ∀ (ops : List (Fin 12)), 
  let finalConfig := (ops.foldl (fun cfg i => flipSigns cfg i 3) initialConfig)
  ¬(onlyA₂Negative finalConfig) := by sorry

end NUMINAMATH_CALUDE_impossibleConfig6_impossibleConfig4_impossibleConfig3_l1854_185484


namespace NUMINAMATH_CALUDE_total_cost_is_1400_l1854_185449

def cost_of_suits (off_the_rack_cost : ℕ) (tailoring_cost : ℕ) : ℕ :=
  off_the_rack_cost + (3 * off_the_rack_cost + tailoring_cost)

theorem total_cost_is_1400 :
  cost_of_suits 300 200 = 1400 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_1400_l1854_185449


namespace NUMINAMATH_CALUDE_fifty_factorial_trailing_zeros_l1854_185456

/-- The number of trailing zeros in n! -/
def trailing_zeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

/-- 50! has exactly 12 trailing zeros -/
theorem fifty_factorial_trailing_zeros : trailing_zeros 50 = 12 := by
  sorry

end NUMINAMATH_CALUDE_fifty_factorial_trailing_zeros_l1854_185456


namespace NUMINAMATH_CALUDE_largest_value_l1854_185400

theorem largest_value (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : a + b = 1) :
  a^2 + b^2 = max (max (max (a^2 + b^2) (2*a*b)) a) (1/2) := by
  sorry

end NUMINAMATH_CALUDE_largest_value_l1854_185400


namespace NUMINAMATH_CALUDE_cubic_root_sum_l1854_185446

theorem cubic_root_sum (p q r : ℝ) : 
  p^3 - 2*p - 2 = 0 → 
  q^3 - 2*q - 2 = 0 → 
  r^3 - 2*r - 2 = 0 → 
  p*(q - r)^2 + q*(r - p)^2 + r*(p - q)^2 = -18 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l1854_185446


namespace NUMINAMATH_CALUDE_one_is_monomial_l1854_185411

/-- Definition of a monomial as an algebraic expression with one term -/
def isMonomial (expr : ℕ → ℚ) : Prop :=
  ∃ (c : ℚ) (n : ℕ), ∀ (k : ℕ), expr k = if k = n then c else 0

/-- Theorem: 1 is a monomial -/
theorem one_is_monomial : isMonomial (fun _ ↦ 1) := by
  sorry

end NUMINAMATH_CALUDE_one_is_monomial_l1854_185411


namespace NUMINAMATH_CALUDE_simplify_expression_l1854_185410

theorem simplify_expression (x : ℝ) :
  4 * x - 8 * x^2 + 10 - (5 - 4 * x + 8 * x^2) = -16 * x^2 + 8 * x + 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1854_185410


namespace NUMINAMATH_CALUDE_apron_sewing_ratio_l1854_185440

/-- Prove that the ratio of aprons sewn today to aprons sewn before today is 3:1 -/
theorem apron_sewing_ratio :
  let total_aprons : ℕ := 150
  let aprons_before : ℕ := 13
  let aprons_tomorrow : ℕ := 49
  let aprons_remaining : ℕ := 2 * aprons_tomorrow
  let aprons_sewn_before_tomorrow : ℕ := total_aprons - aprons_remaining
  let aprons_today : ℕ := aprons_sewn_before_tomorrow - aprons_before
  ∃ (n : ℕ), n > 0 ∧ aprons_today = 3 * n ∧ aprons_before = n :=
by
  sorry

end NUMINAMATH_CALUDE_apron_sewing_ratio_l1854_185440


namespace NUMINAMATH_CALUDE_geometric_sequence_second_term_l1854_185465

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_second_term
  (a : ℕ → ℚ)
  (h_geometric : GeometricSequence a)
  (h_fifth : a 5 = 48)
  (h_sixth : a 6 = 72) :
  a 2 = 128 / 9 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_second_term_l1854_185465


namespace NUMINAMATH_CALUDE_max_power_of_15_l1854_185457

-- Define pow function
def pow (n : ℕ) : ℕ := sorry

-- Define the product of pow(n) from 2 to 2200
def product_pow : ℕ := sorry

-- Theorem statement
theorem max_power_of_15 :
  (∀ m : ℕ, m > 10 → ¬(pow (15^m) ∣ product_pow)) ∧
  (pow (15^10) ∣ product_pow) :=
sorry

end NUMINAMATH_CALUDE_max_power_of_15_l1854_185457


namespace NUMINAMATH_CALUDE_peach_boxes_count_l1854_185442

def peaches_per_basket : ℕ := 23
def num_baskets : ℕ := 7
def peaches_eaten : ℕ := 7
def peaches_per_box : ℕ := 13

theorem peach_boxes_count :
  let total_peaches := peaches_per_basket * num_baskets
  let remaining_peaches := total_peaches - peaches_eaten
  (remaining_peaches / peaches_per_box : ℕ) = 11 := by
  sorry

end NUMINAMATH_CALUDE_peach_boxes_count_l1854_185442


namespace NUMINAMATH_CALUDE_monotonic_function_a_range_l1854_185433

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - x - 2

-- State the theorem
theorem monotonic_function_a_range :
  (∀ a : ℝ, Monotone (f a) → a ∈ Set.Icc (- Real.sqrt 3) (Real.sqrt 3)) ∧
  (∀ a : ℝ, a ∈ Set.Icc (- Real.sqrt 3) (Real.sqrt 3) → Monotone (f a)) :=
sorry

end NUMINAMATH_CALUDE_monotonic_function_a_range_l1854_185433


namespace NUMINAMATH_CALUDE_units_digit_sum_of_powers_l1854_185482

theorem units_digit_sum_of_powers : (24^4 + 42^4) % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_of_powers_l1854_185482


namespace NUMINAMATH_CALUDE_cos_315_degrees_l1854_185455

theorem cos_315_degrees : Real.cos (315 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_315_degrees_l1854_185455


namespace NUMINAMATH_CALUDE_paco_sweet_cookies_left_l1854_185477

/-- The number of sweet cookies Paco has left -/
def sweet_cookies_left (initial_sweet : ℕ) (eaten_sweet : ℕ) : ℕ :=
  initial_sweet - eaten_sweet

/-- Theorem: Paco has 19 sweet cookies left -/
theorem paco_sweet_cookies_left : 
  sweet_cookies_left 34 15 = 19 := by
  sorry

end NUMINAMATH_CALUDE_paco_sweet_cookies_left_l1854_185477


namespace NUMINAMATH_CALUDE_smallest_shift_for_even_function_l1854_185480

theorem smallest_shift_for_even_function (f g : ℝ → ℝ) (σ : ℝ) : 
  (∀ x, f x = Real.sin (2 * x + π / 3)) →
  (∀ x, g x = f (x + σ)) →
  (∀ x, g (-x) = g x) →
  σ > 0 →
  (∀ σ' > 0, (∀ x, f (x + σ') = f (-x + σ')) → σ' ≥ σ) →
  σ = π / 12 := by sorry

end NUMINAMATH_CALUDE_smallest_shift_for_even_function_l1854_185480


namespace NUMINAMATH_CALUDE_factorization_problems_l1854_185497

theorem factorization_problems (m a x : ℝ) : 
  (9 * m^2 - 4 = (3 * m + 2) * (3 * m - 2)) ∧ 
  (2 * a * x^2 + 12 * a * x + 18 * a = 2 * a * (x + 3)^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problems_l1854_185497


namespace NUMINAMATH_CALUDE_reflection_coordinates_l1854_185408

/-- Given points P and M in a 2D plane, this function returns the coordinates of Q, 
    which is the reflection of P about M. -/
def reflection_point (P M : ℝ × ℝ) : ℝ × ℝ :=
  (2 * M.1 - P.1, 2 * M.2 - P.2)

theorem reflection_coordinates :
  let P : ℝ × ℝ := (1, -2)
  let M : ℝ × ℝ := (3, 0)
  reflection_point P M = (5, 2) := by sorry

end NUMINAMATH_CALUDE_reflection_coordinates_l1854_185408
