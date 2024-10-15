import Mathlib

namespace NUMINAMATH_CALUDE_bus_and_walking_problem_l3993_399302

/-- Proof of the bus and walking problem -/
theorem bus_and_walking_problem
  (total_distance : ℝ)
  (walking_speed : ℝ)
  (bus_speed : ℝ)
  (rest_time : ℝ)
  (h1 : total_distance = 21)
  (h2 : walking_speed = 4)
  (h3 : bus_speed = 60)
  (h4 : rest_time = 1/6) -- 10 minutes in hours
  : ∃ (x y : ℝ),
    x + y = total_distance ∧
    x / bus_speed + total_distance / bus_speed = rest_time + y / walking_speed ∧
    x = 19 ∧
    y = 2 := by
  sorry


end NUMINAMATH_CALUDE_bus_and_walking_problem_l3993_399302


namespace NUMINAMATH_CALUDE_draw_balls_count_l3993_399308

/-- The number of ways to draw 3 balls in order from a bin of 12 balls, 
    where each ball remains outside the bin after it is drawn. -/
def draw_balls : ℕ :=
  12 * 11 * 10

/-- Theorem stating that the number of ways to draw 3 balls in order 
    from a bin of 12 balls, where each ball remains outside the bin 
    after it is drawn, is equal to 1320. -/
theorem draw_balls_count : draw_balls = 1320 := by
  sorry

end NUMINAMATH_CALUDE_draw_balls_count_l3993_399308


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l3993_399312

theorem geometric_sequence_third_term :
  ∀ (a : ℕ → ℕ),
    (∀ n, a n > 0) →  -- Sequence of positive integers
    (∃ r : ℕ, ∀ n, a (n + 1) = a n * r) →  -- Geometric sequence
    a 1 = 5 →  -- First term is 5
    a 5 = 405 →  -- Fifth term is 405
    a 3 = 45 :=  -- Third term is 45
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l3993_399312


namespace NUMINAMATH_CALUDE_line_passes_through_I_III_IV_l3993_399368

-- Define the line
def line (x : ℝ) : ℝ := 5 * x - 2

-- Define the quadrants
def in_quadrant_I (x y : ℝ) : Prop := x > 0 ∧ y > 0
def in_quadrant_II (x y : ℝ) : Prop := x < 0 ∧ y > 0
def in_quadrant_III (x y : ℝ) : Prop := x < 0 ∧ y < 0
def in_quadrant_IV (x y : ℝ) : Prop := x > 0 ∧ y < 0

-- Theorem statement
theorem line_passes_through_I_III_IV :
  (∃ x y : ℝ, y = line x ∧ in_quadrant_I x y) ∧
  (∃ x y : ℝ, y = line x ∧ in_quadrant_III x y) ∧
  (∃ x y : ℝ, y = line x ∧ in_quadrant_IV x y) ∧
  ¬(∃ x y : ℝ, y = line x ∧ in_quadrant_II x y) :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_I_III_IV_l3993_399368


namespace NUMINAMATH_CALUDE_equation_solution_l3993_399316

theorem equation_solution (x : ℝ) :
  x ≠ -2 ∧ x ≠ -3 ∧ x ≠ 1 →
  (3 * x + 2) / (x^2 + 5 * x + 6) = 3 * x / (x - 1) →
  3 * x^3 + 12 * x^2 + 19 * x + 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3993_399316


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l3993_399366

theorem root_sum_reciprocal (p q r A B C : ℝ) : 
  p ≠ q ∧ q ≠ r ∧ p ≠ r →
  (∀ x : ℝ, x^3 - 30*x^2 + 105*x - 114 = 0 ↔ x = p ∨ x = q ∨ x = r) →
  (∀ s : ℝ, s ≠ p ∧ s ≠ q ∧ s ≠ r → 
    1 / (s^3 - 30*s^2 + 105*s - 114) = A / (s - p) + B / (s - q) + C / (s - r)) →
  1/A + 1/B + 1/C = 300 := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l3993_399366


namespace NUMINAMATH_CALUDE_find_S_l3993_399339

theorem find_S (R S : ℕ) (h1 : 111111111111 - 222222 = (R + S)^2) (h2 : S > 0) : S = 333332 := by
  sorry

end NUMINAMATH_CALUDE_find_S_l3993_399339


namespace NUMINAMATH_CALUDE_problem_solution_l3993_399331

theorem problem_solution (x y : ℝ) (h1 : x + 3*y = 6) (h2 : x*y = -12) : 
  x^2 + 9*y^2 = 108 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3993_399331


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l3993_399399

theorem triangle_abc_properties (A B C : Real) (a b c : Real) :
  B = 150 * π / 180 →
  a = Real.sqrt 3 * c →
  b = 2 * Real.sqrt 7 →
  Real.sin A + Real.sqrt 3 * Real.sin C = Real.sqrt 2 / 2 →
  (∃ (S : Real), S = a * b * Real.sin C / 2 ∧ S = Real.sqrt 3) ∧
  C = 15 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l3993_399399


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l3993_399365

theorem smallest_n_satisfying_conditions : ∃ (n : ℕ), 
  (n = 626) ∧ 
  (∀ m : ℕ, m > 0 → m < n → 
    (Real.sqrt (m : ℝ) - Real.sqrt ((m - 1) : ℝ) ≥ 0.02 ∨ 
     Real.sin (Real.pi / Real.sqrt (m : ℝ)) ≤ 0.5)) ∧
  (Real.sqrt (n : ℝ) - Real.sqrt ((n - 1) : ℝ) < 0.02) ∧
  (Real.sin (Real.pi / Real.sqrt (n : ℝ)) > 0.5) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l3993_399365


namespace NUMINAMATH_CALUDE_merry_go_round_revolutions_merry_go_round_specific_case_l3993_399378

/-- Given two circular paths with different radii, prove that the number of revolutions
    needed to cover the same distance is inversely proportional to their radii. -/
theorem merry_go_round_revolutions (r1 r2 n1 : ℝ) (hr1 : r1 > 0) (hr2 : r2 > 0) (hn1 : n1 > 0) :
  let n2 := (r1 * n1) / r2
  2 * Real.pi * r1 * n1 = 2 * Real.pi * r2 * n2 := by sorry

/-- Prove that for the specific case of r1 = 30, r2 = 10, and n1 = 36, 
    the number of revolutions n2 for the second path is 108. -/
theorem merry_go_round_specific_case :
  let r1 : ℝ := 30
  let r2 : ℝ := 10
  let n1 : ℝ := 36
  let n2 := (r1 * n1) / r2
  n2 = 108 := by sorry

end NUMINAMATH_CALUDE_merry_go_round_revolutions_merry_go_round_specific_case_l3993_399378


namespace NUMINAMATH_CALUDE_power_of_i_product_l3993_399377

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem power_of_i_product : i^45 * i^105 = -1 := by sorry

end NUMINAMATH_CALUDE_power_of_i_product_l3993_399377


namespace NUMINAMATH_CALUDE_customers_in_other_countries_l3993_399330

theorem customers_in_other_countries 
  (total_customers : ℕ) 
  (us_customers : ℕ) 
  (h1 : total_customers = 7422) 
  (h2 : us_customers = 723) : 
  total_customers - us_customers = 6699 := by
  sorry

end NUMINAMATH_CALUDE_customers_in_other_countries_l3993_399330


namespace NUMINAMATH_CALUDE_triangle_side_length_l3993_399304

/-- Given a triangle ABC with the following properties:
  * A = 60°
  * a = 6√3
  * b = 12
  * S_ABC = 18√3
  Prove that c = 6 -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) (S : ℝ) :
  A = π / 3 →  -- 60° in radians
  a = 6 * Real.sqrt 3 →
  b = 12 →
  S = 18 * Real.sqrt 3 →
  c = 6 := by
  sorry


end NUMINAMATH_CALUDE_triangle_side_length_l3993_399304


namespace NUMINAMATH_CALUDE_max_value_of_f_l3993_399371

-- Define the function f(x) = x(4 - x)
def f (x : ℝ) := x * (4 - x)

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), M = 4 ∧ ∀ x, 0 < x ∧ x < 4 → f x ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3993_399371


namespace NUMINAMATH_CALUDE_line_slope_problem_l3993_399343

/-- Given m > 0 and points (m,1) and (2,√m) on a line with slope 2m, prove m = 4 -/
theorem line_slope_problem (m : ℝ) (h1 : m > 0) : 
  (2 * m = (Real.sqrt m - 1) / (2 - m)) → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_problem_l3993_399343


namespace NUMINAMATH_CALUDE_factor_4t_squared_minus_64_l3993_399384

theorem factor_4t_squared_minus_64 (t : ℝ) : 4 * t^2 - 64 = 4 * (t - 4) * (t + 4) := by
  sorry

end NUMINAMATH_CALUDE_factor_4t_squared_minus_64_l3993_399384


namespace NUMINAMATH_CALUDE_share_ratio_l3993_399301

def total_amount : ℕ := 544

def shares (A B C : ℕ) : Prop :=
  A + B + C = total_amount ∧ 4 * B = C

theorem share_ratio (A B C : ℕ) (h : shares A B C) (hA : A = 64) (hB : B = 96) (hC : C = 384) :
  A * 3 = B * 2 := by sorry

end NUMINAMATH_CALUDE_share_ratio_l3993_399301


namespace NUMINAMATH_CALUDE_base_eight_digits_of_1728_l3993_399367

theorem base_eight_digits_of_1728 : ∃ n : ℕ, n > 0 ∧ 8^(n-1) ≤ 1728 ∧ 1728 < 8^n ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_eight_digits_of_1728_l3993_399367


namespace NUMINAMATH_CALUDE_number_of_petri_dishes_l3993_399379

/-- The number of petri dishes in a lab, given the total number of germs and germs per dish -/
theorem number_of_petri_dishes 
  (total_germs : ℝ) 
  (germs_per_dish : ℝ) 
  (h1 : total_germs = 0.036 * 10^5)
  (h2 : germs_per_dish = 47.99999999999999)
  : ℤ :=
75

#check number_of_petri_dishes

end NUMINAMATH_CALUDE_number_of_petri_dishes_l3993_399379


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l3993_399355

-- Define the vectors a and b
def a (m : ℝ) : Fin 2 → ℝ := λ i => match i with
  | 0 => 2
  | 1 => m

def b (m : ℝ) : Fin 2 → ℝ := λ i => match i with
  | 0 => m
  | 1 => 2

-- Define the parallelism condition
def parallel (v w : Fin 2 → ℝ) : Prop :=
  v 0 * w 1 = v 1 * w 0

-- State the theorem
theorem parallel_vectors_m_value :
  ∀ m : ℝ, parallel (a m) (b m) → m = 2 ∨ m = -2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l3993_399355


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_relation_l3993_399338

theorem binomial_expansion_coefficient_relation (n : ℕ) : 
  (2 * n * (n - 1) = 7 * (2 * n)) → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_relation_l3993_399338


namespace NUMINAMATH_CALUDE_two_numbers_sum_and_lcm_l3993_399398

theorem two_numbers_sum_and_lcm : ∃ (x y : ℕ), 
  x + y = 316 ∧ 
  Nat.lcm x y = 4560 ∧ 
  x = 199 ∧ 
  y = 117 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_sum_and_lcm_l3993_399398


namespace NUMINAMATH_CALUDE_min_employees_for_pollution_monitoring_l3993_399337

/-- Calculates the minimum number of employees needed given the number of employees
    who can monitor different types of pollution. -/
def minimum_employees (water : ℕ) (air : ℕ) (soil : ℕ) 
                      (water_air : ℕ) (air_soil : ℕ) (water_soil : ℕ) 
                      (all_three : ℕ) : ℕ :=
  water + air + soil - water_air - air_soil - water_soil + all_three

/-- Theorem stating that given the specific numbers from the problem,
    the minimum number of employees needed is 165. -/
theorem min_employees_for_pollution_monitoring : 
  minimum_employees 95 80 45 30 20 15 10 = 165 := by
  sorry

#eval minimum_employees 95 80 45 30 20 15 10

end NUMINAMATH_CALUDE_min_employees_for_pollution_monitoring_l3993_399337


namespace NUMINAMATH_CALUDE_inequality_proof_l3993_399344

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 1 / b - c) * (b + 1 / c - a) + 
  (b + 1 / c - a) * (c + 1 / a - b) + 
  (c + 1 / a - b) * (a + 1 / b - c) ≥ 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3993_399344


namespace NUMINAMATH_CALUDE_fraction_addition_l3993_399314

theorem fraction_addition : (1 : ℚ) / 4 + (3 : ℚ) / 8 = (5 : ℚ) / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l3993_399314


namespace NUMINAMATH_CALUDE_cookies_remaining_l3993_399392

theorem cookies_remaining (initial : ℕ) (given : ℕ) (eaten : ℕ) : 
  initial = 36 → given = 14 → eaten = 10 → initial - (given + eaten) = 12 := by
  sorry

end NUMINAMATH_CALUDE_cookies_remaining_l3993_399392


namespace NUMINAMATH_CALUDE_common_difference_is_half_l3993_399326

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_4_6 : a 4 + a 6 = 6
  sum_5 : (a 1 + a 2 + a 3 + a 4 + a 5 : ℚ) = 10

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℚ :=
  seq.a 2 - seq.a 1

/-- Theorem stating that the common difference is 1/2 -/
theorem common_difference_is_half (seq : ArithmeticSequence) :
  common_difference seq = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_common_difference_is_half_l3993_399326


namespace NUMINAMATH_CALUDE_stream_speed_l3993_399362

theorem stream_speed (boat_speed : ℝ) (stream_speed : ℝ) : 
  boat_speed = 39 →
  (1 / (boat_speed - stream_speed)) = (2 * (1 / (boat_speed + stream_speed))) →
  stream_speed = 13 := by
sorry

end NUMINAMATH_CALUDE_stream_speed_l3993_399362


namespace NUMINAMATH_CALUDE_minimize_sum_of_squares_l3993_399364

theorem minimize_sum_of_squares (s : ℝ) (hs : s > 0) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = s ∧
  ∀ (a b : ℝ), a > 0 → b > 0 → a + b = s → x^2 + y^2 ≤ a^2 + b^2 ∧
  x^2 + y^2 = s^2 / 2 ∧ x = s / 2 ∧ y = s / 2 := by
  sorry

end NUMINAMATH_CALUDE_minimize_sum_of_squares_l3993_399364


namespace NUMINAMATH_CALUDE_square_sum_from_product_and_sum_l3993_399341

theorem square_sum_from_product_and_sum (p q : ℝ) 
  (h1 : p * q = 9) 
  (h2 : p + q = 6) : 
  p^2 + q^2 = 18 := by
sorry

end NUMINAMATH_CALUDE_square_sum_from_product_and_sum_l3993_399341


namespace NUMINAMATH_CALUDE_x_plus_y_equals_plus_minus_three_l3993_399327

theorem x_plus_y_equals_plus_minus_three (x y : ℝ) 
  (h1 : |x| = 1) 
  (h2 : |y| = 2) 
  (h3 : x * y > 0) : 
  x + y = 3 ∨ x + y = -3 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_plus_minus_three_l3993_399327


namespace NUMINAMATH_CALUDE_backpacking_cooks_l3993_399322

/-- The number of ways to choose k items from n items --/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of people in the group --/
def total_people : ℕ := 10

/-- The number of people willing to cook --/
def eligible_people : ℕ := total_people - 1

/-- The number of cooks needed --/
def cooks_needed : ℕ := 2

theorem backpacking_cooks : choose eligible_people cooks_needed = 36 := by
  sorry

end NUMINAMATH_CALUDE_backpacking_cooks_l3993_399322


namespace NUMINAMATH_CALUDE_tornado_distance_l3993_399313

theorem tornado_distance (car_distance lawn_chair_distance birdhouse_distance : ℝ)
  (h1 : lawn_chair_distance = 2 * car_distance)
  (h2 : birdhouse_distance = 3 * lawn_chair_distance)
  (h3 : birdhouse_distance = 1200) :
  car_distance = 200 := by
sorry

end NUMINAMATH_CALUDE_tornado_distance_l3993_399313


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_product_l3993_399360

/-- A geometric sequence with common ratio r -/
def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum_product (a : ℕ → ℝ) (r : ℝ) :
  geometric_sequence a r →
  a 4 + a 8 = -2 →
  a 6 * (a 2 + 2 * a 6 + a 10) = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_product_l3993_399360


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3993_399393

-- Define the sets A and B
def A : Set ℝ := {x | 3 - x > 0 ∧ x + 2 > 0}
def B : Set ℝ := {m | 3 > 2 * m - 1}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x < 3} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3993_399393


namespace NUMINAMATH_CALUDE_matching_instrument_probability_l3993_399346

/-- The probability of selecting a matching cello-viola pair -/
theorem matching_instrument_probability
  (total_cellos : ℕ)
  (total_violas : ℕ)
  (matching_pairs : ℕ)
  (h1 : total_cellos = 800)
  (h2 : total_violas = 600)
  (h3 : matching_pairs = 100) :
  (matching_pairs : ℚ) / (total_cellos * total_violas) = 1 / 4800 :=
by sorry

end NUMINAMATH_CALUDE_matching_instrument_probability_l3993_399346


namespace NUMINAMATH_CALUDE_card_count_l3993_399387

theorem card_count (black red spades diamonds hearts clubs : ℕ) : 
  black = 7 →
  red = 6 →
  diamonds = 2 * spades →
  hearts = 2 * diamonds →
  clubs = 6 →
  black = spades + clubs →
  red = diamonds + hearts →
  spades + diamonds + hearts + clubs = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_card_count_l3993_399387


namespace NUMINAMATH_CALUDE_remainder_4039_div_31_l3993_399397

theorem remainder_4039_div_31 : 4039 % 31 = 9 := by
  sorry

end NUMINAMATH_CALUDE_remainder_4039_div_31_l3993_399397


namespace NUMINAMATH_CALUDE_original_triangle_area_l3993_399349

/-- Given a triangle whose dimensions are quadrupled to form a new triangle with an area of 256 square feet,
    prove that the area of the original triangle is 16 square feet. -/
theorem original_triangle_area (original : ℝ) (new : ℝ) : 
  new = 256 → -- area of the new triangle
  new = original * 16 → -- relationship between new and original areas
  original = 16 := by
sorry

end NUMINAMATH_CALUDE_original_triangle_area_l3993_399349


namespace NUMINAMATH_CALUDE_distance_between_squares_l3993_399323

/-- Given a configuration of two squares where:
    - The smaller square has a perimeter of 8 cm
    - The larger square has an area of 49 cm²
    This theorem states that the distance between point A (top-right corner of the larger square)
    and point B (top-left corner of the smaller square) is approximately 10.3 cm. -/
theorem distance_between_squares (small_square_perimeter : ℝ) (large_square_area : ℝ)
    (h1 : small_square_perimeter = 8)
    (h2 : large_square_area = 49) :
    ∃ (distance : ℝ), abs (distance - Real.sqrt 106) < 0.1 ∧
    distance = Real.sqrt ((large_square_area.sqrt + small_square_perimeter / 4) ^ 2 +
    (large_square_area.sqrt - small_square_perimeter / 4) ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_distance_between_squares_l3993_399323


namespace NUMINAMATH_CALUDE_range_of_fraction_l3993_399376

theorem range_of_fraction (x y : ℝ) (hx : 1 < x ∧ x < 6) (hy : 2 < y ∧ y < 8) :
  1/8 < x/y ∧ x/y < 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_fraction_l3993_399376


namespace NUMINAMATH_CALUDE_problem_solution_l3993_399336

theorem problem_solution (a : ℝ) (h : a/3 - 3/a = 4) :
  (a^8 - 6561) / (81 * a^4) * (3 * a) / (a^2 + 9) = 72 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3993_399336


namespace NUMINAMATH_CALUDE_masha_creates_more_words_l3993_399388

/-- Represents a word as a list of characters -/
def Word := List Char

/-- Counts the number of distinct words formed by removing exactly two letters from a given word -/
def countDistinctWordsRemovingTwo (w : Word) : Nat :=
  sorry

/-- The word "ИНТЕГРИРОВАНИЕ" -/
def integrirovanie : Word :=
  ['И', 'Н', 'Т', 'Е', 'Г', 'Р', 'И', 'Р', 'О', 'В', 'А', 'Н', 'И', 'Е']

/-- The word "СУПЕРКОМПЬЮТЕР" -/
def superkomputer : Word :=
  ['С', 'У', 'П', 'Е', 'Р', 'К', 'О', 'М', 'П', 'Ь', 'Ю', 'Т', 'Е', 'Р']

theorem masha_creates_more_words :
  countDistinctWordsRemovingTwo superkomputer > countDistinctWordsRemovingTwo integrirovanie :=
sorry

end NUMINAMATH_CALUDE_masha_creates_more_words_l3993_399388


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l3993_399390

theorem arithmetic_mean_problem (a : ℝ) : 
  ((2 * a + 16) + (3 * a - 8)) / 2 = 74 → a = 28 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l3993_399390


namespace NUMINAMATH_CALUDE_solve_for_t_l3993_399363

theorem solve_for_t (s t : ℝ) (eq1 : 7 * s + 3 * t = 84) (eq2 : s = t - 3) : t = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_t_l3993_399363


namespace NUMINAMATH_CALUDE_class_age_problem_l3993_399350

/-- Proves that if the average age of 6 people remains 19 years after adding a 1-year-old person,
    then the original average was calculated 1 year ago. -/
theorem class_age_problem (initial_total_age : ℕ) (years_passed : ℕ) : 
  initial_total_age / 6 = 19 →
  (initial_total_age + 6 * years_passed + 1) / 7 = 19 →
  years_passed = 1 := by
sorry

end NUMINAMATH_CALUDE_class_age_problem_l3993_399350


namespace NUMINAMATH_CALUDE_max_class_size_is_17_l3993_399318

/-- Represents a school with students and buses -/
structure School where
  total_students : ℕ
  num_buses : ℕ
  seats_per_bus : ℕ

/-- Checks if it's possible to seat all students with the given max class size -/
def can_seat_all (s : School) (max_class_size : ℕ) : Prop :=
  ∀ (class_sizes : List ℕ),
    (class_sizes.sum = s.total_students) →
    (∀ size ∈ class_sizes, size ≤ max_class_size) →
    ∃ (allocation : List (List ℕ)),
      (allocation.length ≤ s.num_buses) ∧
      (∀ bus ∈ allocation, bus.sum ≤ s.seats_per_bus) ∧
      (allocation.join.sum = s.total_students)

/-- The theorem to be proved -/
theorem max_class_size_is_17 (s : School) 
    (h1 : s.total_students = 920)
    (h2 : s.num_buses = 16)
    (h3 : s.seats_per_bus = 71) :
  (can_seat_all s 17 ∧ ¬can_seat_all s 18) := by
  sorry

end NUMINAMATH_CALUDE_max_class_size_is_17_l3993_399318


namespace NUMINAMATH_CALUDE_ios_department_larger_l3993_399386

theorem ios_department_larger (n m : ℕ) : 
  (7 * n + 15 * m = 15 * n + 9 * m) → m > n := by
  sorry

end NUMINAMATH_CALUDE_ios_department_larger_l3993_399386


namespace NUMINAMATH_CALUDE_square_sum_difference_l3993_399394

theorem square_sum_difference : 3^2 + 7^2 - 5^2 = 33 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_difference_l3993_399394


namespace NUMINAMATH_CALUDE_pet_store_dogs_l3993_399324

/-- Given a ratio of cats to dogs and the number of cats, calculate the number of dogs -/
def calculate_dogs (cat_ratio : ℕ) (dog_ratio : ℕ) (num_cats : ℕ) : ℕ :=
  (num_cats / cat_ratio) * dog_ratio

/-- Theorem: Given the ratio of cats to dogs is 2:3 and there are 14 cats, there are 21 dogs -/
theorem pet_store_dogs : calculate_dogs 2 3 14 = 21 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_dogs_l3993_399324


namespace NUMINAMATH_CALUDE_mame_on_top_probability_l3993_399358

/-- Represents a piece of paper with 8 quadrants -/
structure Paper :=
  (quadrants : Fin 8)

/-- The probability of a specific quadrant being on top -/
def probability_on_top (p : Paper) : ℚ :=
  1 / 8

/-- The quadrant where "MAME" is written -/
def mame_quadrant : Fin 8 := 0

/-- Theorem: The probability of "MAME" being on top is 1/8 -/
theorem mame_on_top_probability :
  probability_on_top {quadrants := mame_quadrant} = 1 / 8 := by
  sorry


end NUMINAMATH_CALUDE_mame_on_top_probability_l3993_399358


namespace NUMINAMATH_CALUDE_find_set_B_l3993_399333

universe u

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}

theorem find_set_B (A B : Set Nat) 
  (h1 : (U \ (A ∪ B)) = {1, 3})
  (h2 : (A ∩ (U \ B)) = {2, 5}) :
  B = {4, 6, 7} := by
  sorry

end NUMINAMATH_CALUDE_find_set_B_l3993_399333


namespace NUMINAMATH_CALUDE_min_boxes_for_cube_l3993_399342

/-- The width of the box in centimeters -/
def box_width : ℕ := 8

/-- The length of the box in centimeters -/
def box_length : ℕ := 12

/-- The height of the box in centimeters -/
def box_height : ℕ := 30

/-- The volume of a single box in cubic centimeters -/
def box_volume : ℕ := box_width * box_length * box_height

/-- The side length of the smallest cube that can be formed -/
def cube_side : ℕ := Nat.lcm (Nat.lcm box_width box_length) box_height

/-- The volume of the smallest cube that can be formed -/
def cube_volume : ℕ := cube_side ^ 3

/-- The theorem stating the minimum number of boxes needed to form a cube -/
theorem min_boxes_for_cube : cube_volume / box_volume = 600 := by
  sorry

end NUMINAMATH_CALUDE_min_boxes_for_cube_l3993_399342


namespace NUMINAMATH_CALUDE_rectangle_area_l3993_399306

/-- Given a rectangle with perimeter 28 cm and width 6 cm, prove its area is 48 square cm. -/
theorem rectangle_area (perimeter width : ℝ) (h1 : perimeter = 28) (h2 : width = 6) :
  let length := (perimeter - 2 * width) / 2
  width * length = 48 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l3993_399306


namespace NUMINAMATH_CALUDE_strictly_increasing_inverse_sum_identity_l3993_399359

theorem strictly_increasing_inverse_sum_identity 
  (f : ℝ → ℝ) 
  (h_incr : ∀ x y, x < y → f x < f y) 
  (h_inv : Function.Bijective f) 
  (h_sum : ∀ x, f x + (Function.invFun f) x = 2 * x) : 
  ∃ b : ℝ, ∀ x, f x = x + b :=
sorry

end NUMINAMATH_CALUDE_strictly_increasing_inverse_sum_identity_l3993_399359


namespace NUMINAMATH_CALUDE_equation_solution_l3993_399361

theorem equation_solution : 
  ∃! x : ℝ, Real.sqrt x + 3 * Real.sqrt (x^2 + 8*x) + Real.sqrt (x + 8) = 40 - 3*x := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3993_399361


namespace NUMINAMATH_CALUDE_rock_sale_price_per_pound_l3993_399334

theorem rock_sale_price_per_pound 
  (average_weight : ℝ) 
  (num_rocks : ℕ) 
  (total_sale : ℝ) 
  (h1 : average_weight = 1.5)
  (h2 : num_rocks = 10)
  (h3 : total_sale = 60) :
  total_sale / (average_weight * num_rocks) = 4 := by
sorry

end NUMINAMATH_CALUDE_rock_sale_price_per_pound_l3993_399334


namespace NUMINAMATH_CALUDE_factory_material_usage_extension_l3993_399321

/-- Given a factory with m tons of raw materials and an original plan to use a tons per day (a > 1),
    prove that if the factory reduces daily usage by 1 ton, it can use the materials for m / (a(a-1))
    additional days compared to the original plan. -/
theorem factory_material_usage_extension (m a : ℝ) (ha : a > 1) :
  let original_days := m / a
  let new_days := m / (a - 1)
  new_days - original_days = m / (a * (a - 1)) := by sorry

end NUMINAMATH_CALUDE_factory_material_usage_extension_l3993_399321


namespace NUMINAMATH_CALUDE_square_perimeter_l3993_399370

theorem square_perimeter (area : ℝ) (perimeter : ℝ) : 
  area = 588 → perimeter = 56 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l3993_399370


namespace NUMINAMATH_CALUDE_unique_data_set_l3993_399375

def mean (xs : Fin 4 → ℕ+) : ℚ :=
  (xs 0 + xs 1 + xs 2 + xs 3 : ℚ) / 4

def median (xs : Fin 4 → ℕ+) : ℚ :=
  (xs 1 + xs 2 : ℚ) / 2

def variance (xs : Fin 4 → ℕ+) (μ : ℚ) : ℚ :=
  ((xs 0 - μ)^2 + (xs 1 - μ)^2 + (xs 2 - μ)^2 + (xs 3 - μ)^2) / 4

def stdDev (xs : Fin 4 → ℕ+) (μ : ℚ) : ℚ :=
  (variance xs μ).sqrt

theorem unique_data_set (xs : Fin 4 → ℕ+) 
    (h_ordered : ∀ i j : Fin 4, i ≤ j → xs i ≤ xs j)
    (h_mean : mean xs = 2)
    (h_median : median xs = 2)
    (h_stddev : stdDev xs 2 = 1) :
    xs 0 = 1 ∧ xs 1 = 1 ∧ xs 2 = 3 ∧ xs 3 = 3 := by
  sorry

#check unique_data_set

end NUMINAMATH_CALUDE_unique_data_set_l3993_399375


namespace NUMINAMATH_CALUDE_linear_function_value_l3993_399307

/-- Given a linear function f(x) = ax + b, if f(3) = 7 and f(5) = -1, then f(0) = 19 -/
theorem linear_function_value (a b : ℝ) (f : ℝ → ℝ) 
    (h_def : ∀ x, f x = a * x + b)
    (h_3 : f 3 = 7)
    (h_5 : f 5 = -1) : 
  f 0 = 19 := by
sorry

end NUMINAMATH_CALUDE_linear_function_value_l3993_399307


namespace NUMINAMATH_CALUDE_circle_area_and_circumference_l3993_399329

theorem circle_area_and_circumference (r : ℝ) (h : r > 0) :
  ∃ (A C : ℝ),
    A = π * r^2 ∧
    C = 2 * π * r :=
by sorry

end NUMINAMATH_CALUDE_circle_area_and_circumference_l3993_399329


namespace NUMINAMATH_CALUDE_c_investment_is_10500_l3993_399317

/-- Represents the investment and profit distribution in a partnership business -/
structure Partnership where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  total_profit : ℕ
  a_profit_share : ℕ

/-- Calculates C's investment given the partnership details -/
def calculate_c_investment (p : Partnership) : ℕ :=
  p.total_profit * p.a_investment / p.a_profit_share - p.a_investment - p.b_investment

/-- Theorem stating that C's investment is 10500 given the problem conditions -/
theorem c_investment_is_10500 (p : Partnership) 
  (h1 : p.a_investment = 6300)
  (h2 : p.b_investment = 4200)
  (h3 : p.total_profit = 12500)
  (h4 : p.a_profit_share = 3750) :
  calculate_c_investment p = 10500 := by
  sorry

#eval calculate_c_investment {
  a_investment := 6300, 
  b_investment := 4200, 
  c_investment := 0,  -- This value doesn't affect the calculation
  total_profit := 12500, 
  a_profit_share := 3750
}

end NUMINAMATH_CALUDE_c_investment_is_10500_l3993_399317


namespace NUMINAMATH_CALUDE_expression_evaluation_l3993_399385

theorem expression_evaluation : 
  (1/8)^(1/3) - Real.log 2 / Real.log 3 * Real.log 27 / Real.log 4 + 
  (Real.log (Real.sqrt 2) / Real.log 10 + Real.log (Real.sqrt 5) / Real.log 10) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3993_399385


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_l3993_399347

theorem isosceles_triangle_area : 
  ∀ a b : ℕ,
  a > 0 ∧ b > 0 →
  2 * a + b = 12 →
  (a + a > b ∧ a + b > a) →
  (∃ (s : ℝ), s * s = (a * a : ℝ) - (b * b / 4 : ℝ)) →
  (a * s / 2 : ℝ) = 4 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_area_l3993_399347


namespace NUMINAMATH_CALUDE_polygon_diagonals_l3993_399348

theorem polygon_diagonals (n : ℕ) (h : (n - 2) * 180 + 360 = 1800) : n - 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_polygon_diagonals_l3993_399348


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3993_399325

/-- The sum of the first n terms of a geometric sequence -/
def geometric_sum (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₁ * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_sequence_sum (n : ℕ) :
  geometric_sum 1 (1/3) n = 121/81 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3993_399325


namespace NUMINAMATH_CALUDE_parabola_and_line_properties_l3993_399309

-- Define the parabola C: y^2 = 2px
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

-- Define point A
def point_A : ℝ × ℝ := (2, -4)

-- Define point B
def point_B : ℝ × ℝ := (0, 2)

-- Theorem statement
theorem parabola_and_line_properties
  (p : ℝ)
  (h_p_pos : p > 0)
  (h_A_on_C : parabola p point_A.1 point_A.2) :
  -- Part 1: Equation of parabola and its directrix
  (∃ (x y : ℝ), parabola 4 x y ∧ y^2 = 8*x) ∧
  (∃ (x : ℝ), x = -2) ∧
  -- Part 2: Equations of line l
  (∃ (x y : ℝ),
    (x = 0 ∨ y = 2 ∨ x - y + 2 = 0) ∧
    (x = point_B.1 ∧ y = point_B.2) ∧
    (∃! (z : ℝ), parabola 4 x z ∧ z = y)) :=
sorry

end NUMINAMATH_CALUDE_parabola_and_line_properties_l3993_399309


namespace NUMINAMATH_CALUDE_age_double_time_l3993_399315

/-- Proves that the number of years until a man's age is twice his son's age is 2,
    given that the man is currently 22 years older than his son and the son is currently 20 years old. -/
theorem age_double_time : ∃ (x : ℕ), 
  (20 + x) * 2 = (20 + 22 + x) ∧ x = 2 := by sorry

end NUMINAMATH_CALUDE_age_double_time_l3993_399315


namespace NUMINAMATH_CALUDE_periodicity_2pi_l3993_399369

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * Real.cos y

/-- The periodicity theorem -/
theorem periodicity_2pi (f : ℝ → ℝ) (h : FunctionalEquation f) :
  ∀ x : ℝ, f (x + 2 * Real.pi) = f x :=
by sorry

end NUMINAMATH_CALUDE_periodicity_2pi_l3993_399369


namespace NUMINAMATH_CALUDE_thirty_five_power_pq_l3993_399382

theorem thirty_five_power_pq (p q : ℤ) (A B : ℝ) (hA : A = 5^p) (hB : B = 7^q) :
  A^q * B^p = 35^(p*q) := by
  sorry

end NUMINAMATH_CALUDE_thirty_five_power_pq_l3993_399382


namespace NUMINAMATH_CALUDE_product_simplification_l3993_399380

theorem product_simplification (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 := by
  sorry

end NUMINAMATH_CALUDE_product_simplification_l3993_399380


namespace NUMINAMATH_CALUDE_square_plus_one_geq_two_abs_l3993_399372

theorem square_plus_one_geq_two_abs (x : ℝ) : x^2 + 1 ≥ 2 * |x| := by
  sorry

end NUMINAMATH_CALUDE_square_plus_one_geq_two_abs_l3993_399372


namespace NUMINAMATH_CALUDE_exist_decreasing_lcm_sequence_l3993_399305

theorem exist_decreasing_lcm_sequence :
  ∃ (a : Fin 100 → ℕ),
    (∀ i j : Fin 100, i < j → a i < a j) ∧
    (∀ i : Fin 99, Nat.lcm (a i) (a (i + 1)) > Nat.lcm (a (i + 1)) (a (i + 2))) :=
by sorry

end NUMINAMATH_CALUDE_exist_decreasing_lcm_sequence_l3993_399305


namespace NUMINAMATH_CALUDE_square_plus_product_equals_square_l3993_399320

theorem square_plus_product_equals_square (x y : ℤ) :
  x^2 + x*y = y^2 ↔ x = 0 ∧ y = 0 := by sorry

end NUMINAMATH_CALUDE_square_plus_product_equals_square_l3993_399320


namespace NUMINAMATH_CALUDE_football_game_score_l3993_399391

theorem football_game_score (total_points winning_margin : ℕ) 
  (h1 : total_points = 34) 
  (h2 : winning_margin = 14) : 
  ∃ (panthers_score cougars_score : ℕ), 
    panthers_score + cougars_score = total_points ∧ 
    cougars_score = panthers_score + winning_margin ∧ 
    panthers_score = 10 := by
  sorry

end NUMINAMATH_CALUDE_football_game_score_l3993_399391


namespace NUMINAMATH_CALUDE_man_upstream_speed_l3993_399395

/-- Calculates the upstream speed of a man given his downstream speed and the stream speed. -/
def upstream_speed (downstream_speed stream_speed : ℝ) : ℝ :=
  downstream_speed - 2 * stream_speed

/-- Theorem stating that given a downstream speed of 14 km/h and a stream speed of 3 km/h, 
    the upstream speed is 8 km/h. -/
theorem man_upstream_speed : 
  upstream_speed 14 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_man_upstream_speed_l3993_399395


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_l3993_399381

theorem rectangle_area_diagonal (l w d : ℝ) (h1 : l / w = 5 / 2) (h2 : l^2 + w^2 = d^2) :
  l * w = (10 / 29) * d^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_l3993_399381


namespace NUMINAMATH_CALUDE_max_product_constraint_l3993_399310

theorem max_product_constraint (a b : ℝ) (h : a + b = 5) : 
  a * b ≤ 25 / 4 ∧ (a * b = 25 / 4 ↔ a = 5 / 2 ∧ b = 5 / 2) := by
  sorry

end NUMINAMATH_CALUDE_max_product_constraint_l3993_399310


namespace NUMINAMATH_CALUDE_expand_polynomial_l3993_399303

theorem expand_polynomial (x : ℝ) : (2 + x^2) * (1 - x^4) = -x^6 + x^2 - 2*x^4 + 2 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l3993_399303


namespace NUMINAMATH_CALUDE_incorrect_vs_correct_operations_l3993_399340

theorem incorrect_vs_correct_operations (x : ℝ) :
  (x / 8 - 12 = 18) → (x * 8 * 12 = 23040) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_vs_correct_operations_l3993_399340


namespace NUMINAMATH_CALUDE_kangaroo_distance_after_four_hops_l3993_399383

/-- The distance traveled by a kangaroo after a certain number of hops,
    where each hop is 1/4 of the remaining distance to the target. -/
def kangaroo_distance (target : ℚ) (hops : ℕ) : ℚ :=
  target * (1 - (3/4)^hops)

/-- Theorem: A kangaroo starting at 0 aiming for 2, hopping 1/4 of the remaining
    distance each time, will travel 175/128 units after 4 hops. -/
theorem kangaroo_distance_after_four_hops :
  kangaroo_distance 2 4 = 175 / 128 := by
  sorry

#eval kangaroo_distance 2 4

end NUMINAMATH_CALUDE_kangaroo_distance_after_four_hops_l3993_399383


namespace NUMINAMATH_CALUDE_greatest_common_remainder_l3993_399345

theorem greatest_common_remainder (a b c : ℕ) (h : a = 25 ∧ b = 57 ∧ c = 105) :
  ∃ (k : ℕ), k > 0 ∧ 
    (∃ (r : ℕ), a % k = r ∧ b % k = r ∧ c % k = r) ∧
    (∀ (m : ℕ), m > k → ¬(∃ (s : ℕ), a % m = s ∧ b % m = s ∧ c % m = s)) ∧
  k = 16 := by
sorry

end NUMINAMATH_CALUDE_greatest_common_remainder_l3993_399345


namespace NUMINAMATH_CALUDE_modified_cube_vertices_l3993_399332

/-- Calculates the number of vertices in a modified cube -/
def modifiedCubeVertices (initialSideLength : ℕ) (removedSideLength : ℕ) : ℕ :=
  8 * (3 * 4 - 3)

/-- Theorem stating that a cube of side length 5 with smaller cubes of side length 2 
    removed from each corner has 64 vertices -/
theorem modified_cube_vertices :
  modifiedCubeVertices 5 2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_modified_cube_vertices_l3993_399332


namespace NUMINAMATH_CALUDE_count_numbers_with_remainder_l3993_399335

theorem count_numbers_with_remainder (n : ℕ) : 
  (Finset.filter (fun N => N > 17 ∧ 2017 % N = 17) (Finset.range (2017 + 1))).card = 13 := by
  sorry

end NUMINAMATH_CALUDE_count_numbers_with_remainder_l3993_399335


namespace NUMINAMATH_CALUDE_ellipse1_passes_through_points_ellipse2_passes_through_point_ellipse3_passes_through_point_ellipse2_axis_ratio_ellipse3_axis_ratio_l3993_399356

-- Define the ellipse equations
def ellipse1 (x y : ℝ) : Prop := x^2 / 9 + y^2 / 3 = 1
def ellipse2 (x y : ℝ) : Prop := x^2 / 9 + y^2 = 1
def ellipse3 (x y : ℝ) : Prop := y^2 / 81 + x^2 / 9 = 1

-- Theorem for the first ellipse
theorem ellipse1_passes_through_points :
  ellipse1 (Real.sqrt 6) 1 ∧ ellipse1 (-Real.sqrt 3) (-Real.sqrt 2) := by sorry

-- Theorems for the second and third ellipses
theorem ellipse2_passes_through_point : ellipse2 3 0 := by sorry
theorem ellipse3_passes_through_point : ellipse3 3 0 := by sorry

theorem ellipse2_axis_ratio :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a = 3 * b ∧
  ∀ (x y : ℝ), ellipse2 x y ↔ x^2 / a^2 + y^2 / b^2 = 1 := by sorry

theorem ellipse3_axis_ratio :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a = 3 * b ∧
  ∀ (x y : ℝ), ellipse3 x y ↔ y^2 / a^2 + x^2 / b^2 = 1 := by sorry

end NUMINAMATH_CALUDE_ellipse1_passes_through_points_ellipse2_passes_through_point_ellipse3_passes_through_point_ellipse2_axis_ratio_ellipse3_axis_ratio_l3993_399356


namespace NUMINAMATH_CALUDE_jenny_donut_order_l3993_399354

def donut_combinations (total_donuts : ℕ) (kinds : ℕ) (min_per_kind : ℕ) : ℕ :=
  let two_kinds := Nat.choose kinds 2 * Nat.choose (total_donuts - 2 * min_per_kind + 2 - 1) (2 - 1)
  let three_kinds := Nat.choose kinds 3 * Nat.choose (total_donuts - 3 * min_per_kind + 3 - 1) (3 - 1)
  two_kinds + three_kinds

theorem jenny_donut_order : donut_combinations 8 5 2 = 110 := by
  sorry

end NUMINAMATH_CALUDE_jenny_donut_order_l3993_399354


namespace NUMINAMATH_CALUDE_macaron_fraction_l3993_399352

theorem macaron_fraction (mitch joshua miles renz : ℕ) (total_kids : ℕ) :
  mitch = 20 →
  joshua = mitch + 6 →
  2 * joshua = miles →
  total_kids = 68 →
  2 * total_kids = mitch + joshua + miles + renz →
  renz + 1 = miles * 19 / 26 :=
by sorry

end NUMINAMATH_CALUDE_macaron_fraction_l3993_399352


namespace NUMINAMATH_CALUDE_least_four_digit_solution_l3993_399373

theorem least_four_digit_solution (x : ℕ) : 
  (x ≥ 1000 ∧ x < 10000) →
  (5 * x ≡ 15 [ZMOD 20]) →
  (3 * x + 7 ≡ 19 [ZMOD 8]) →
  (-3 * x + 2 ≡ x [ZMOD 14]) →
  (∀ y : ℕ, y ≥ 1000 ∧ y < x →
    ¬(5 * y ≡ 15 [ZMOD 20] ∧
      3 * y + 7 ≡ 19 [ZMOD 8] ∧
      -3 * y + 2 ≡ y [ZMOD 14])) →
  x = 1032 :=
by sorry

end NUMINAMATH_CALUDE_least_four_digit_solution_l3993_399373


namespace NUMINAMATH_CALUDE_transistor_count_2002_l3993_399357

def moores_law (initial_year final_year : ℕ) (initial_transistors : ℕ) : ℕ :=
  initial_transistors * 2^((final_year - initial_year) / 2)

theorem transistor_count_2002 :
  moores_law 1988 2002 500000 = 64000000 := by
  sorry

end NUMINAMATH_CALUDE_transistor_count_2002_l3993_399357


namespace NUMINAMATH_CALUDE_rectangle_discrepancy_exists_l3993_399374

/-- Represents a point in a 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle with sides parallel to the axes -/
structure Rectangle where
  x1 : ℝ
  y1 : ℝ
  x2 : ℝ
  y2 : ℝ

/-- The side length of the square -/
def squareSideLength : ℝ := 10^2019

/-- The total number of marked points -/
def totalPoints : ℕ := 10^4038

/-- A set of points marked in the square -/
def markedPoints : Set Point := sorry

/-- Calculates the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℝ :=
  (r.x2 - r.x1) * (r.y2 - r.y1)

/-- Counts the number of points inside a rectangle -/
def pointsInRectangle (r : Rectangle) (points : Set Point) : ℕ := sorry

/-- The main theorem to be proved -/
theorem rectangle_discrepancy_exists :
  ∃ (r : Rectangle),
    r.x1 ≥ 0 ∧ r.y1 ≥ 0 ∧ r.x2 ≤ squareSideLength ∧ r.y2 ≤ squareSideLength ∧
    |rectangleArea r - (pointsInRectangle r markedPoints : ℝ)| ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_discrepancy_exists_l3993_399374


namespace NUMINAMATH_CALUDE_tensor_result_l3993_399353

-- Define the sets P and Q
def P : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}
def Q : Set ℝ := {x | x > 1}

-- Define the ⊗ operation
def tensorOp (P Q : Set ℝ) : Set ℝ := {x | x ∈ P ∪ Q ∧ x ∉ P ∩ Q}

-- Theorem statement
theorem tensor_result : tensorOp P Q = {x | (0 ≤ x ∧ x ≤ 1) ∨ (x > 2)} := by
  sorry

end NUMINAMATH_CALUDE_tensor_result_l3993_399353


namespace NUMINAMATH_CALUDE_problem_solution_l3993_399328

theorem problem_solution (x : ℚ) : (5 * x - 8 = 15 * x + 4) → (3 * (x + 9) = 129 / 5) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3993_399328


namespace NUMINAMATH_CALUDE_compass_leg_swap_impossible_l3993_399389

/-- Represents a point on the grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- Represents the state of the compass -/
structure CompassState where
  leg1 : GridPoint
  leg2 : GridPoint

/-- The squared distance between two grid points -/
def squaredDistance (p1 p2 : GridPoint) : ℤ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- A valid move of the compass -/
def isValidMove (start finish : CompassState) : Prop :=
  (start.leg1 = finish.leg1 ∧ squaredDistance start.leg1 start.leg2 = squaredDistance start.leg1 finish.leg2) ∨
  (start.leg2 = finish.leg2 ∧ squaredDistance start.leg1 start.leg2 = squaredDistance finish.leg1 start.leg2)

/-- A sequence of valid moves -/
def isValidMoveSequence : List CompassState → Prop
  | [] => True
  | [_] => True
  | s1 :: s2 :: rest => isValidMove s1 s2 ∧ isValidMoveSequence (s2 :: rest)

/-- The main theorem stating it's impossible to swap compass legs -/
theorem compass_leg_swap_impossible (start finish : CompassState) (moves : List CompassState) :
  isValidMoveSequence (start :: moves ++ [finish]) →
  squaredDistance start.leg1 start.leg2 = squaredDistance finish.leg1 finish.leg2 →
  ¬(start.leg1 = finish.leg2 ∧ start.leg2 = finish.leg1) :=
sorry

end NUMINAMATH_CALUDE_compass_leg_swap_impossible_l3993_399389


namespace NUMINAMATH_CALUDE_hyperbola_circle_tangent_radius_l3993_399351

/-- The radius of a circle that is tangent to the asymptotes of a specific hyperbola -/
theorem hyperbola_circle_tangent_radius : ∀ (r : ℝ), r > 0 →
  (∀ (x y : ℝ), x^2 / 9 - y^2 / 4 = 1 →
    (∃ (t : ℝ), (x - 3)^2 + y^2 = r^2 ∧
      (y = (2/3) * x ∨ y = -(2/3) * x) ∧
      (∀ (x' y' : ℝ), (x' - 3)^2 + y'^2 < r^2 →
        y' ≠ (2/3) * x' ∧ y' ≠ -(2/3) * x'))) →
  r = 6 * Real.sqrt 13 / 13 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_circle_tangent_radius_l3993_399351


namespace NUMINAMATH_CALUDE_non_collinear_implies_nonzero_l3993_399396

-- Define the vector type
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define the collinearity relation
def collinear (a b : V) : Prop := ∃ (k : ℝ), a = k • b

-- State the theorem
theorem non_collinear_implies_nonzero (a b : V) : 
  ¬(collinear a b) → a ≠ 0 ∧ b ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_non_collinear_implies_nonzero_l3993_399396


namespace NUMINAMATH_CALUDE_C_work_duration_l3993_399319

-- Define the work rates and durations
def work_rate_A : ℚ := 1 / 30
def work_rate_B : ℚ := 1 / 30
def days_A_worked : ℕ := 10
def days_B_worked : ℕ := 10
def days_C_worked : ℕ := 10

-- Define the total work as 1 (representing 100%)
def total_work : ℚ := 1

-- Theorem to prove
theorem C_work_duration :
  let work_done_A : ℚ := work_rate_A * days_A_worked
  let work_done_B : ℚ := work_rate_B * days_B_worked
  let work_done_C : ℚ := total_work - (work_done_A + work_done_B)
  let work_rate_C : ℚ := work_done_C / days_C_worked
  (total_work / work_rate_C : ℚ) = 30 := by
  sorry

end NUMINAMATH_CALUDE_C_work_duration_l3993_399319


namespace NUMINAMATH_CALUDE_billy_feeds_twice_daily_l3993_399300

/-- The number of times Billy feeds his horses per day -/
def feedings_per_day (num_horses : ℕ) (oats_per_meal : ℕ) (total_oats : ℕ) (days : ℕ) : ℕ :=
  (total_oats / days) / (num_horses * oats_per_meal)

/-- Theorem: Billy feeds his horses twice a day -/
theorem billy_feeds_twice_daily :
  feedings_per_day 4 4 96 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_billy_feeds_twice_daily_l3993_399300


namespace NUMINAMATH_CALUDE_remainder_1234567_div_12_l3993_399311

theorem remainder_1234567_div_12 : 1234567 % 12 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_1234567_div_12_l3993_399311
