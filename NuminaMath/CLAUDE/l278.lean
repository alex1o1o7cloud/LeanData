import Mathlib

namespace NUMINAMATH_CALUDE_sequence_b_decreasing_l278_27838

/-- Given a sequence {a_n} that satisfies the following conditions:
    1) a_1 = 2
    2) 2 * a_n * a_{n+1} = a_n^2 + 1
    Define b_n = (a_n - 1) / (a_n + 1)
    Then the sequence {b_n} is decreasing. -/
theorem sequence_b_decreasing (a : ℕ → ℝ) (b : ℕ → ℝ) :
  a 1 = 2 ∧
  (∀ n : ℕ, 2 * a n * a (n + 1) = a n ^ 2 + 1) ∧
  (∀ n : ℕ, b n = (a n - 1) / (a n + 1)) →
  ∀ n : ℕ, b (n + 1) < b n :=
by sorry

end NUMINAMATH_CALUDE_sequence_b_decreasing_l278_27838


namespace NUMINAMATH_CALUDE_unique_prime_divisibility_l278_27861

theorem unique_prime_divisibility : 
  ∀ p : ℕ, Prime p → 
  (p = 3 ↔ 
    ∃! a : ℕ, a ∈ Finset.range p ∧ 
    p ∣ (a^3 - 3*a + 1)) := by sorry

end NUMINAMATH_CALUDE_unique_prime_divisibility_l278_27861


namespace NUMINAMATH_CALUDE_geometric_mean_problem_l278_27857

theorem geometric_mean_problem (a b c : ℝ) 
  (h1 : b^2 = a*c)  -- b is the geometric mean of a and c
  (h2 : a*b*c = 27) : b = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_problem_l278_27857


namespace NUMINAMATH_CALUDE_kevin_cards_l278_27827

theorem kevin_cards (initial_cards lost_cards : ℝ) 
  (h1 : initial_cards = 47.0)
  (h2 : lost_cards = 7.0) : 
  initial_cards - lost_cards = 40.0 := by
  sorry

end NUMINAMATH_CALUDE_kevin_cards_l278_27827


namespace NUMINAMATH_CALUDE_sales_tax_percentage_l278_27866

theorem sales_tax_percentage (total_before_tax : ℝ) (total_with_tax : ℝ) : 
  total_before_tax = 150 → 
  total_with_tax = 162 → 
  (total_with_tax - total_before_tax) / total_before_tax * 100 = 8 := by
  sorry

end NUMINAMATH_CALUDE_sales_tax_percentage_l278_27866


namespace NUMINAMATH_CALUDE_p_amount_l278_27812

theorem p_amount : ∃ (p : ℚ), p = 49 ∧ p = (2 * (1/7) * p + 35) := by
  sorry

end NUMINAMATH_CALUDE_p_amount_l278_27812


namespace NUMINAMATH_CALUDE_average_of_25_results_l278_27828

theorem average_of_25_results (first_12_avg : ℝ) (last_12_avg : ℝ) (result_13 : ℝ) 
  (h1 : first_12_avg = 14)
  (h2 : last_12_avg = 17)
  (h3 : result_13 = 228) :
  (12 * first_12_avg + result_13 + 12 * last_12_avg) / 25 = 24 := by
  sorry

end NUMINAMATH_CALUDE_average_of_25_results_l278_27828


namespace NUMINAMATH_CALUDE_balls_sold_l278_27834

theorem balls_sold (selling_price : ℕ) (cost_price : ℕ) (loss : ℕ) : 
  selling_price = 720 →
  loss = 5 * cost_price →
  cost_price = 120 →
  selling_price + loss = 11 * cost_price :=
by
  sorry

end NUMINAMATH_CALUDE_balls_sold_l278_27834


namespace NUMINAMATH_CALUDE_inequality_proof_l278_27833

theorem inequality_proof (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hne : ¬(a = b ∧ b = c)) : 
  2 * (a^3 + b^3 + c^3) > a^2 * (b + c) + b^2 * (a + c) + c^2 * (a + b) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l278_27833


namespace NUMINAMATH_CALUDE_function_domain_range_sum_l278_27867

-- Define the function f
def f (x : ℝ) : ℝ := -x^2 + 2*x

-- Define the domain and range
def is_valid_domain_and_range (m n : ℝ) : Prop :=
  (∀ x, m ≤ x ∧ x ≤ n → 3*m ≤ f x ∧ f x ≤ 3*n) ∧
  (∃ x, m ≤ x ∧ x ≤ n ∧ f x = 3*m) ∧
  (∃ x, m ≤ x ∧ x ≤ n ∧ f x = 3*n)

-- State the theorem
theorem function_domain_range_sum :
  ∃ m n : ℝ, is_valid_domain_and_range m n ∧ m = -1 ∧ n = 0 ∧ m + n = -1 :=
sorry

end NUMINAMATH_CALUDE_function_domain_range_sum_l278_27867


namespace NUMINAMATH_CALUDE_triple_equality_l278_27858

theorem triple_equality (a b c : ℝ) : 
  a * (b^2 + c) = c * (c + a * b) →
  b * (c^2 + a) = a * (a + b * c) →
  c * (a^2 + b) = b * (b + c * a) →
  a = b ∧ b = c :=
by sorry

end NUMINAMATH_CALUDE_triple_equality_l278_27858


namespace NUMINAMATH_CALUDE_plain_calculations_l278_27819

/-- Given information about two plains A and B, prove various calculations about their areas, populations, and elevation difference. -/
theorem plain_calculations (area_B area_A : ℝ) (pop_density_A pop_density_B : ℝ) 
  (distance_AB : ℝ) (elevation_gradient : ℝ) :
  area_B = 200 →
  area_A = area_B - 50 →
  pop_density_A = 50 →
  pop_density_B = 75 →
  distance_AB = 25 →
  elevation_gradient = 500 / 10 →
  (area_A = 150 ∧ 
   area_A * pop_density_A + area_B * pop_density_B = 22500 ∧
   elevation_gradient * distance_AB = 125) :=
by sorry

end NUMINAMATH_CALUDE_plain_calculations_l278_27819


namespace NUMINAMATH_CALUDE_chongqing_population_scientific_notation_l278_27887

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Conversion from a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

/-- The population of Chongqing at the end of 2022 -/
def chongqingPopulation : ℕ := 32000000

theorem chongqing_population_scientific_notation :
  toScientificNotation (chongqingPopulation : ℝ) =
    ScientificNotation.mk 3.2 7 (by norm_num) :=
  sorry

end NUMINAMATH_CALUDE_chongqing_population_scientific_notation_l278_27887


namespace NUMINAMATH_CALUDE_imaginary_unit_equation_l278_27842

theorem imaginary_unit_equation (a : ℝ) (h1 : a > 0) :
  Complex.abs ((a + Complex.I) / Complex.I) = 2 → a = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_equation_l278_27842


namespace NUMINAMATH_CALUDE_markers_given_l278_27839

theorem markers_given (initial : ℕ) (total : ℕ) (given : ℕ) : 
  initial = 217 → total = 326 → given = total - initial → given = 109 := by
sorry

end NUMINAMATH_CALUDE_markers_given_l278_27839


namespace NUMINAMATH_CALUDE_square_difference_l278_27831

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 12) : (x - y)^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l278_27831


namespace NUMINAMATH_CALUDE_divisor_of_smallest_six_digit_multiple_l278_27859

def smallest_six_digit_number : Nat := 100000
def given_number : Nat := 100011
def divisor : Nat := 33337

theorem divisor_of_smallest_six_digit_multiple :
  (given_number = smallest_six_digit_number + 11) →
  (∀ n : Nat, n < given_number → n < smallest_six_digit_number ∨ given_number % n ≠ 0) →
  (given_number % divisor = 0) →
  (given_number / divisor = 3) :=
by sorry

end NUMINAMATH_CALUDE_divisor_of_smallest_six_digit_multiple_l278_27859


namespace NUMINAMATH_CALUDE_positive_numbers_not_all_equal_l278_27835

/-- Given positive numbers a, b, and c that are not all equal -/
theorem positive_numbers_not_all_equal 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_not_all_equal : ¬(a = b ∧ b = c)) : 
  /- 1. (a-b)² + (b-c)² + (c-a)² ≠ 0 -/
  ((a - b)^2 + (b - c)^2 + (c - a)^2 ≠ 0) ∧ 
  /- 2. At least one of a > b, a < b, or a = b is true -/
  (a > b ∨ a < b ∨ a = b) ∧ 
  /- 3. It is possible for a ≠ c, b ≠ c, and a ≠ b to all be true simultaneously -/
  ∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z := by
  sorry

end NUMINAMATH_CALUDE_positive_numbers_not_all_equal_l278_27835


namespace NUMINAMATH_CALUDE_hypotenuse_length_is_5_sqrt_211_l278_27855

/-- Right triangle ABC with specific properties -/
structure RightTriangle where
  -- Points
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  X : ℝ × ℝ
  Y : ℝ × ℝ
  -- AB and AC are legs of the right triangle
  ab_leg : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0
  -- X is on AB
  x_on_ab : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ X = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2))
  -- Y is on AC
  y_on_ac : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ Y = (A.1 + s * (C.1 - A.1), A.2 + s * (C.2 - A.2))
  -- AX:XB = 2:3
  ax_xb_ratio : dist A X / dist X B = 2 / 3
  -- AY:YC = 2:3
  ay_yc_ratio : dist A Y / dist Y C = 2 / 3
  -- BY = 18 units
  by_length : dist B Y = 18
  -- CX = 15 units
  cx_length : dist C X = 15

/-- The length of hypotenuse BC in the right triangle -/
def hypotenuseLength (t : RightTriangle) : ℝ :=
  dist t.B t.C

/-- Theorem: The length of hypotenuse BC is 5√211 units -/
theorem hypotenuse_length_is_5_sqrt_211 (t : RightTriangle) :
  hypotenuseLength t = 5 * Real.sqrt 211 := by
  sorry


end NUMINAMATH_CALUDE_hypotenuse_length_is_5_sqrt_211_l278_27855


namespace NUMINAMATH_CALUDE_factorization_equality_l278_27808

theorem factorization_equality (a : ℝ) : 2 * a^2 - 8 = 2 * (a + 2) * (a - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l278_27808


namespace NUMINAMATH_CALUDE_larger_screen_diagonal_l278_27878

theorem larger_screen_diagonal (d : ℝ) : 
  d ^ 2 = 17 ^ 2 + 36 → d = 5 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_larger_screen_diagonal_l278_27878


namespace NUMINAMATH_CALUDE_product_polynomial_sum_l278_27875

theorem product_polynomial_sum (g h : ℚ) : 
  (∀ d : ℚ, (7 * d^2 - 3 * d + g) * (3 * d^2 + h * d - 5) = 
   21 * d^4 - 44 * d^3 - 35 * d^2 + 14 * d + 15) →
  g + h = -28/9 := by
sorry

end NUMINAMATH_CALUDE_product_polynomial_sum_l278_27875


namespace NUMINAMATH_CALUDE_equation_solutions_l278_27864

theorem equation_solutions :
  (∀ x : ℝ, x^2 - 2*x = 0 ↔ x = 0 ∨ x = 2) ∧
  (∀ x : ℝ, x^2 - 4*x + 1 = 0 ↔ x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l278_27864


namespace NUMINAMATH_CALUDE_concyclic_intersection_points_l278_27809

structure Circle where
  center : Point
  radius : ℝ

structure Chord (c : Circle) where
  endpoint1 : Point
  endpoint2 : Point

def midpoint_of_arc (c : Circle) (ch : Chord c) : Point := sorry

def intersect_chords (c : Circle) (ch1 ch2 : Chord c) : Point := sorry

def concyclic (p1 p2 p3 p4 : Point) : Prop := sorry

theorem concyclic_intersection_points 
  (c : Circle) 
  (bc : Chord c) 
  (a : Point) 
  (ad ae : Chord c) 
  (f g : Point) :
  a = midpoint_of_arc c bc →
  f = intersect_chords c bc ad →
  g = intersect_chords c bc ae →
  concyclic (ad.endpoint2) (ae.endpoint2) f g :=
sorry

end NUMINAMATH_CALUDE_concyclic_intersection_points_l278_27809


namespace NUMINAMATH_CALUDE_parabola_symmetry_l278_27863

/-- Given two parabolas, prove that they are symmetrical about the x-axis -/
theorem parabola_symmetry (x : ℝ) : 
  let f (x : ℝ) := (x - 1)^2 + 3
  let g (x : ℝ) := -(x - 1)^2 - 3
  ∀ x, f x = -g x := by
  sorry

end NUMINAMATH_CALUDE_parabola_symmetry_l278_27863


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l278_27814

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_pos : ∀ n, a n > 0)
  (h_a2 : a 2 = 2)
  (h_sum : 2 * a 3 + a 4 = 16) :
  a 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l278_27814


namespace NUMINAMATH_CALUDE_right_triangle_area_in_square_yards_l278_27837

/-- The area of a right triangle with legs of 60 feet and 80 feet in square yards -/
theorem right_triangle_area_in_square_yards : 
  let leg1 : ℝ := 60
  let leg2 : ℝ := 80
  let triangle_area_sqft : ℝ := (1/2) * leg1 * leg2
  let sqft_per_sqyd : ℝ := 9
  triangle_area_sqft / sqft_per_sqyd = 800/3 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_in_square_yards_l278_27837


namespace NUMINAMATH_CALUDE_harry_friday_speed_l278_27860

-- Define Harry's running speeds
def monday_speed : ℝ := 10
def tuesday_to_thursday_increase : ℝ := 0.5
def friday_increase : ℝ := 0.6

-- Define the function to calculate speed increase
def speed_increase (base_speed : ℝ) (increase_percentage : ℝ) : ℝ :=
  base_speed * (1 + increase_percentage)

-- Theorem statement
theorem harry_friday_speed :
  let tuesday_to_thursday_speed := speed_increase monday_speed tuesday_to_thursday_increase
  let friday_speed := speed_increase tuesday_to_thursday_speed friday_increase
  friday_speed = 24 := by sorry

end NUMINAMATH_CALUDE_harry_friday_speed_l278_27860


namespace NUMINAMATH_CALUDE_situps_total_l278_27885

/-- The number of sit-ups Barney can do in one minute -/
def barney_situps : ℕ := 45

/-- The number of sit-ups Carrie can do in one minute -/
def carrie_situps : ℕ := 2 * barney_situps

/-- The number of sit-ups Jerrie can do in one minute -/
def jerrie_situps : ℕ := carrie_situps + 5

/-- The number of minutes Barney performs sit-ups -/
def barney_minutes : ℕ := 1

/-- The number of minutes Carrie performs sit-ups -/
def carrie_minutes : ℕ := 2

/-- The number of minutes Jerrie performs sit-ups -/
def jerrie_minutes : ℕ := 3

/-- The total number of sit-ups performed by all three people -/
def total_situps : ℕ := barney_situps * barney_minutes + 
                        carrie_situps * carrie_minutes + 
                        jerrie_situps * jerrie_minutes

theorem situps_total : total_situps = 510 := by
  sorry

end NUMINAMATH_CALUDE_situps_total_l278_27885


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l278_27892

theorem quadratic_root_problem (a : ℝ) : 
  (3 : ℝ)^2 - (a + 2) * 3 + 2 * a = 0 → 
  ∃ x : ℝ, x^2 - (a + 2) * x + 2 * a = 0 ∧ x ≠ 3 ∧ x = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l278_27892


namespace NUMINAMATH_CALUDE_one_thirds_in_nine_fifths_l278_27800

theorem one_thirds_in_nine_fifths : (9 : ℚ) / 5 / (1 : ℚ) / 3 = 27 / 5 := by sorry

end NUMINAMATH_CALUDE_one_thirds_in_nine_fifths_l278_27800


namespace NUMINAMATH_CALUDE_power_of_64_five_sixths_l278_27836

theorem power_of_64_five_sixths : (64 : ℝ) ^ (5/6) = 32 := by sorry

end NUMINAMATH_CALUDE_power_of_64_five_sixths_l278_27836


namespace NUMINAMATH_CALUDE_steves_nickels_l278_27823

theorem steves_nickels (nickels dimes : ℕ) : 
  dimes = nickels + 4 →
  5 * nickels + 10 * dimes = 70 →
  nickels = 2 := by
sorry

end NUMINAMATH_CALUDE_steves_nickels_l278_27823


namespace NUMINAMATH_CALUDE_ellipse_C_properties_l278_27816

/-- Given an ellipse C with equation x²/a² + y²/b² = 1 (a > b > 0), 
    eccentricity √3/3, and major axis length 2√3 --/
def ellipse_C (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ 
  (a^2 - b^2) / a^2 = 3 / 9 ∧
  2 * a = 2 * Real.sqrt 3

/-- The equation of ellipse C --/
def ellipse_C_equation (x y : ℝ) : Prop :=
  x^2 / 3 + y^2 / 2 = 1

/-- Circle O with major axis of ellipse C as its diameter --/
def circle_O (x y : ℝ) : Prop :=
  x^2 + y^2 = 3

/-- Point on circle O --/
def point_on_circle_O (M : ℝ × ℝ) : Prop :=
  circle_O M.1 M.2

/-- Line perpendicular to OM passing through M --/
def perpendicular_line (M : ℝ × ℝ) (x y : ℝ) : Prop :=
  M.1 * (x - M.1) + M.2 * (y - M.2) = 0

theorem ellipse_C_properties (a b : ℝ) (h : ellipse_C a b) :
  (∀ x y, ellipse_C_equation x y ↔ x^2 / a^2 + y^2 / b^2 = 1) ∧
  (∀ M : ℝ × ℝ, point_on_circle_O M →
    ∃ x y, perpendicular_line M x y ∧ x = 1 ∧ y = 0) :=
sorry

end NUMINAMATH_CALUDE_ellipse_C_properties_l278_27816


namespace NUMINAMATH_CALUDE_equation_solutions_l278_27879

theorem equation_solutions (x m : ℝ) : 
  ((3 * x - m) / 2 - (x + m) / 3 = 5 / 6) →
  (m = -1 → x = 0) ∧
  (x = 5 → (1 / 2) * m^2 + 2 * m = 30) := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l278_27879


namespace NUMINAMATH_CALUDE_cat_litter_cost_l278_27884

/-- Proves the cost of a cat litter container given specific conditions --/
theorem cat_litter_cost 
  (container_size : ℕ) 
  (litter_box_capacity : ℕ) 
  (change_frequency : ℕ) 
  (total_cost : ℕ) 
  (total_days : ℕ) 
  (h1 : container_size = 45)
  (h2 : litter_box_capacity = 15)
  (h3 : change_frequency = 7)
  (h4 : total_cost = 210)
  (h5 : total_days = 210) :
  total_cost / (total_days / change_frequency * litter_box_capacity / container_size) = 21 := by
  sorry


end NUMINAMATH_CALUDE_cat_litter_cost_l278_27884


namespace NUMINAMATH_CALUDE_expected_points_is_16_l278_27897

/-- The probability of a successful free throw -/
def free_throw_probability : ℝ := 0.8

/-- The number of free throw opportunities in a game -/
def opportunities : ℕ := 10

/-- The number of attempts per free throw opportunity -/
def attempts_per_opportunity : ℕ := 2

/-- The number of points awarded for each successful hit -/
def points_per_hit : ℕ := 1

/-- The expected number of points scored in a game -/
def expected_points : ℝ :=
  (opportunities : ℝ) * (attempts_per_opportunity : ℝ) * free_throw_probability * (points_per_hit : ℝ)

theorem expected_points_is_16 : expected_points = 16 := by sorry

end NUMINAMATH_CALUDE_expected_points_is_16_l278_27897


namespace NUMINAMATH_CALUDE_investment_ratio_l278_27881

/-- Given three investors A, B, and C with the following conditions:
  1. A invests the same amount as B
  2. A invests 2/3 of what C invests
  3. Total profit is 11000
  4. C's share of the profit is 3000
Prove that the ratio of A's investment to B's investment is 1:1 -/
theorem investment_ratio (a b c : ℝ) (h1 : a = b) (h2 : a = (2/3) * c)
  (total_profit : ℝ) (h3 : total_profit = 11000)
  (c_share : ℝ) (h4 : c_share = 3000) :
  a / b = 1 := by
  sorry

end NUMINAMATH_CALUDE_investment_ratio_l278_27881


namespace NUMINAMATH_CALUDE_cyclic_inequality_l278_27817

theorem cyclic_inequality (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (y + z) / (2 * x) + (z + x) / (2 * y) + (x + y) / (2 * z) ≥ 
  2 * x / (y + z) + 2 * y / (z + x) + 2 * z / (x + y) := by
  sorry

end NUMINAMATH_CALUDE_cyclic_inequality_l278_27817


namespace NUMINAMATH_CALUDE_a_in_closed_unit_interval_l278_27851

def P : Set ℝ := {x | x^2 ≤ 1}
def M (a : ℝ) : Set ℝ := {a}

theorem a_in_closed_unit_interval (a : ℝ) :
  P ∪ M a = P → a ∈ Set.Icc (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_a_in_closed_unit_interval_l278_27851


namespace NUMINAMATH_CALUDE_students_not_in_chorus_or_band_l278_27877

theorem students_not_in_chorus_or_band 
  (total : ℕ) (chorus : ℕ) (band : ℕ) (both : ℕ) 
  (h1 : total = 50) 
  (h2 : chorus = 18) 
  (h3 : band = 26) 
  (h4 : both = 2) : 
  total - (chorus + band - both) = 8 := by
  sorry

end NUMINAMATH_CALUDE_students_not_in_chorus_or_band_l278_27877


namespace NUMINAMATH_CALUDE_count_integers_with_8_and_9_between_700_and_1000_l278_27818

def count_integers_with_8_and_9 (lower_bound upper_bound : ℕ) : ℕ :=
  (upper_bound - lower_bound + 1) / 100 * 2

theorem count_integers_with_8_and_9_between_700_and_1000 :
  count_integers_with_8_and_9 700 1000 = 6 := by
  sorry

end NUMINAMATH_CALUDE_count_integers_with_8_and_9_between_700_and_1000_l278_27818


namespace NUMINAMATH_CALUDE_same_solution_implies_a_equals_four_l278_27894

theorem same_solution_implies_a_equals_four (a : ℝ) : 
  (∃ x : ℝ, 2 * x + 1 = 3) →
  (∃ x : ℝ, 2 - (a - x) / 3 = 1) →
  (∀ x : ℝ, 2 * x + 1 = 3 ↔ 2 - (a - x) / 3 = 1) →
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_same_solution_implies_a_equals_four_l278_27894


namespace NUMINAMATH_CALUDE_sandy_kim_age_multiple_l278_27848

/-- Proves that Sandy will be 3 times as old as Kim in two years -/
theorem sandy_kim_age_multiple :
  ∀ (sandy_age kim_age : ℕ) (sandy_bill : ℕ),
    sandy_bill = 10 * sandy_age →
    sandy_bill = 340 →
    kim_age = 10 →
    (sandy_age + 2) / (kim_age + 2) = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_sandy_kim_age_multiple_l278_27848


namespace NUMINAMATH_CALUDE_water_flow_speed_equation_l278_27890

/-- The speed of water flow in a river where two boats meet under specific conditions -/
def water_flow_speed : ℝ → Prop := λ V =>
  -- Speed of boat A in still water
  let speed_A : ℝ := 44
  -- Speed of boat B in still water
  let speed_B : ℝ := V^2
  -- Normal meeting time
  let normal_time : ℝ := 11
  -- Delayed meeting time
  let delayed_time : ℝ := 11.25
  -- Delay of boat B
  let delay : ℝ := 2/3
  -- Equation representing the scenario
  5 * V^2 - 8 * V - 132 = 0

theorem water_flow_speed_equation : ∃ V : ℝ, water_flow_speed V := by
  sorry

end NUMINAMATH_CALUDE_water_flow_speed_equation_l278_27890


namespace NUMINAMATH_CALUDE_lateral_surface_area_of_specific_pyramid_l278_27806

/-- Regular hexagonal pyramid -/
structure RegularHexagonalPyramid where
  baseEdgeLength : ℝ
  volume : ℝ

/-- Lateral surface area of a regular hexagonal pyramid -/
def lateralSurfaceArea (pyramid : RegularHexagonalPyramid) : ℝ :=
  sorry

theorem lateral_surface_area_of_specific_pyramid :
  let pyramid : RegularHexagonalPyramid :=
    { baseEdgeLength := 2
    , volume := 2 * Real.sqrt 3 }
  lateralSurfaceArea pyramid = 12 := by
  sorry

end NUMINAMATH_CALUDE_lateral_surface_area_of_specific_pyramid_l278_27806


namespace NUMINAMATH_CALUDE_routes_in_3x3_grid_l278_27825

/-- The number of routes from top-left to bottom-right in a 3x3 grid -/
def number_of_routes : ℕ := 20

/-- The size of the grid -/
def grid_size : ℕ := 3

/-- The total number of moves required to reach the bottom-right corner -/
def total_moves : ℕ := 2 * grid_size

/-- The number of right moves (or down moves) required -/
def moves_in_one_direction : ℕ := grid_size

theorem routes_in_3x3_grid : 
  number_of_routes = Nat.choose total_moves moves_in_one_direction := by
  sorry

end NUMINAMATH_CALUDE_routes_in_3x3_grid_l278_27825


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l278_27826

/-- Given an ellipse with semi-major axis a and semi-minor axis b,
    and a line y = kx intersecting the ellipse at points B and C,
    if the product of the slopes of AB and AC is -3/4,
    then the eccentricity of the ellipse is 1/2. -/
theorem ellipse_eccentricity (a b : ℝ) (k : ℝ) :
  a > b ∧ b > 0 →
  ∃ (B C : ℝ × ℝ),
    (B.1^2 / a^2 + B.2^2 / b^2 = 1) ∧
    (C.1^2 / a^2 + C.2^2 / b^2 = 1) ∧
    (B.2 = k * B.1) ∧
    (C.2 = k * C.1) ∧
    ((B.2 - b) / B.1 * (C.2 + b) / C.1 = -3/4) →
  Real.sqrt (1 - b^2 / a^2) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l278_27826


namespace NUMINAMATH_CALUDE_bank_savings_exceed_target_l278_27891

def geometric_sum (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

def initial_deposit := 5
def daily_ratio := 2
def target_amount := 5000  -- 50 dollars in cents

theorem bank_savings_exceed_target :
  ∃ n : ℕ, 
    n = 10 ∧ 
    geometric_sum initial_deposit daily_ratio n ≥ target_amount ∧
    ∀ m : ℕ, m < n → geometric_sum initial_deposit daily_ratio m < target_amount :=
by sorry

end NUMINAMATH_CALUDE_bank_savings_exceed_target_l278_27891


namespace NUMINAMATH_CALUDE_absolute_value_equals_negative_l278_27898

theorem absolute_value_equals_negative (a : ℝ) : |a| = -a → a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equals_negative_l278_27898


namespace NUMINAMATH_CALUDE_parallel_vectors_l278_27868

-- Define the vectors a and b
def a : ℝ × ℝ := (-1, 1)
def b (t : ℝ) : ℝ × ℝ := (3, t)

-- Define the parallel condition
def is_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

-- Theorem statement
theorem parallel_vectors (t : ℝ) :
  is_parallel (b t) (a.1 + (b t).1, a.2 + (b t).2) → t = -3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_l278_27868


namespace NUMINAMATH_CALUDE_pole_length_after_cut_l278_27862

theorem pole_length_after_cut (original_length : ℝ) (cut_percentage : ℝ) (new_length : ℝ) : 
  original_length = 20 →
  cut_percentage = 30 →
  new_length = original_length * (1 - cut_percentage / 100) →
  new_length = 14 :=
by sorry

end NUMINAMATH_CALUDE_pole_length_after_cut_l278_27862


namespace NUMINAMATH_CALUDE_susies_house_rooms_l278_27843

/-- The number of rooms in Susie's house -/
def number_of_rooms : ℕ := 6

/-- The time it takes Susie to vacuum the whole house, in hours -/
def total_vacuum_time : ℝ := 2

/-- The time it takes Susie to vacuum one room, in minutes -/
def time_per_room : ℝ := 20

/-- Theorem stating that the number of rooms in Susie's house is 6 -/
theorem susies_house_rooms :
  number_of_rooms = (total_vacuum_time * 60) / time_per_room :=
by sorry

end NUMINAMATH_CALUDE_susies_house_rooms_l278_27843


namespace NUMINAMATH_CALUDE_gas_cost_theorem_l278_27802

/-- Calculates the total cost of gas for Ryosuke's travels in a day -/
def calculate_gas_cost (trip1_start trip1_end trip2_start trip2_end : ℕ) 
                       (fuel_efficiency : ℚ) (gas_price : ℚ) : ℚ :=
  let total_distance : ℕ := (trip1_end - trip1_start) + (trip2_end - trip2_start)
  let gallons_used : ℚ := total_distance / fuel_efficiency
  let total_cost : ℚ := gallons_used * gas_price
  total_cost

/-- The total cost of gas for Ryosuke's travels is approximately $10.11 -/
theorem gas_cost_theorem : 
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/100 ∧ 
  |calculate_gas_cost 63102 63135 63135 63166 25 (395/100) - (1011/100)| < ε :=
sorry

end NUMINAMATH_CALUDE_gas_cost_theorem_l278_27802


namespace NUMINAMATH_CALUDE_quadratic_function_range_l278_27830

/-- A quadratic function with a positive leading coefficient -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a > 0 ∧ ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_function_range
  (f : ℝ → ℝ)
  (h1 : QuadraticFunction f)
  (h2 : ∀ x : ℝ, f x = f (4 - x))
  (h3 : ∀ x : ℝ, f (1 - 2*x^2) < f (1 + 2*x - x^2)) :
  ∀ x : ℝ, f (1 - 2*x^2) < f (1 + 2*x - x^2) → -2 < x ∧ x < 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l278_27830


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l278_27844

theorem necessary_but_not_sufficient :
  ∀ x : ℝ,
  (x + 2 = 0 → x^2 - 4 = 0) ∧
  ¬(x^2 - 4 = 0 → x + 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l278_27844


namespace NUMINAMATH_CALUDE_tangent_problem_l278_27896

theorem tangent_problem (α β : Real) 
  (h1 : Real.tan (α + β) = 1/2) 
  (h2 : Real.tan β = 1/3) : 
  Real.tan (α - π/4) = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_problem_l278_27896


namespace NUMINAMATH_CALUDE_factorial_prime_factorization_l278_27810

theorem factorial_prime_factorization (x i a m p : ℕ) : 
  x = (List.range 8).foldl (· * ·) 1 →
  x = 2^i * 3^a * 5^m * 7^p →
  i + a + m + p = 11 →
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_factorial_prime_factorization_l278_27810


namespace NUMINAMATH_CALUDE_distance_after_two_hours_l278_27846

/-- The distance between two people walking in opposite directions for 2 hours -/
theorem distance_after_two_hours 
  (jay_speed : ℝ) 
  (paul_speed : ℝ) 
  (h1 : jay_speed = 0.8 / 15) 
  (h2 : paul_speed = 3 / 30) 
  : jay_speed * 120 + paul_speed * 120 = 18.4 := by
  sorry

end NUMINAMATH_CALUDE_distance_after_two_hours_l278_27846


namespace NUMINAMATH_CALUDE_tournament_sequences_l278_27822

/-- Represents a team in the tournament -/
structure Team :=
  (players : Finset ℕ)
  (size : players.card = 7)

/-- Represents a tournament between two teams -/
structure Tournament :=
  (teamA : Team)
  (teamB : Team)

/-- Represents a sequence of matches in the tournament -/
def MatchSequence (t : Tournament) := Finset ℕ

/-- The number of possible match sequences in a tournament -/
def numSequences (t : Tournament) : ℕ := Nat.choose 14 7

/-- Theorem: The number of possible match sequences in a tournament
    between two teams of 7 players each is equal to C(14,7) -/
theorem tournament_sequences (t : Tournament) :
  numSequences t = 3432 :=
by sorry

end NUMINAMATH_CALUDE_tournament_sequences_l278_27822


namespace NUMINAMATH_CALUDE_min_value_of_x_plus_four_over_x_squared_min_value_achieved_l278_27850

theorem min_value_of_x_plus_four_over_x_squared (x : ℝ) (h : x > 0) :
  x + 4 / x^2 ≥ 3 :=
sorry

theorem min_value_achieved (x : ℝ) (h : x > 0) :
  x + 4 / x^2 = 3 ↔ x = 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_x_plus_four_over_x_squared_min_value_achieved_l278_27850


namespace NUMINAMATH_CALUDE_parallel_equal_segment_construction_l278_27882

/-- Represents a point on a 2D grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- Represents a vector on a 2D grid -/
structure GridVector where
  dx : ℤ
  dy : ℤ

/-- Calculates the vector between two grid points -/
def vectorBetween (a b : GridPoint) : GridVector :=
  { dx := b.x - a.x, dy := b.y - a.y }

/-- Translates a point by a vector -/
def translatePoint (p : GridPoint) (v : GridVector) : GridPoint :=
  { x := p.x + v.dx, y := p.y + v.dy }

/-- Calculates the squared length of a vector -/
def vectorLengthSquared (v : GridVector) : ℤ :=
  v.dx * v.dx + v.dy * v.dy

theorem parallel_equal_segment_construction 
  (a b c : GridPoint) : 
  let v := vectorBetween a b
  let d := translatePoint c v
  (vectorBetween c d = v) ∧ 
  (vectorLengthSquared (vectorBetween a b) = vectorLengthSquared (vectorBetween c d)) :=
by sorry


end NUMINAMATH_CALUDE_parallel_equal_segment_construction_l278_27882


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l278_27886

def A : Set ℕ := {x | 0 ≤ x ∧ x ≤ 2}
def B : Set ℕ := {1, 2, 3}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l278_27886


namespace NUMINAMATH_CALUDE_system_solution_l278_27874

theorem system_solution (a : ℝ) : 
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    (x₁^2 + y₁^2 = 26 * (y₁ * Real.sin a - x₁ * Real.cos a) ∧
     x₁^2 + y₁^2 = 26 * (y₁ * Real.cos (2*a) - x₁ * Real.sin (2*a))) ∧
    (x₂^2 + y₂^2 = 26 * (y₂ * Real.sin a - x₂ * Real.cos a) ∧
     x₂^2 + y₂^2 = 26 * (y₂ * Real.cos (2*a) - x₂ * Real.sin (2*a))) ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 24^2) →
  (∃ n : ℤ, a = π/6 + (2/3) * Real.arctan (5/12) + (2*π*n)/3 ∨
            a = π/6 - (2/3) * Real.arctan (5/12) + (2*π*n)/3 ∨
            a = π/6 + (2*π*n)/3) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l278_27874


namespace NUMINAMATH_CALUDE_inequality_proof_l278_27805

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a * b + b * c + c * a = 1) :
  Real.sqrt (a + 1 / a) + Real.sqrt (b + 1 / b) + Real.sqrt (c + 1 / c) ≥ 
    2 * (Real.sqrt a + Real.sqrt b + Real.sqrt c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l278_27805


namespace NUMINAMATH_CALUDE_number_approximation_l278_27869

-- Define the function f
def f (x : ℝ) : ℝ := x

-- Define the approximation relation
def approx (x y : ℝ) : Prop := abs (x - y) < 0.000000000000001

-- State the theorem
theorem number_approximation (x : ℝ) :
  approx (f (69.28 * 0.004) / x) 9.237333333333334 →
  approx x 0.03 :=
by
  sorry

end NUMINAMATH_CALUDE_number_approximation_l278_27869


namespace NUMINAMATH_CALUDE_bahs_to_yahs_conversion_l278_27870

/-- The number of bahs in one rah -/
def bahs_per_rah : ℚ := 18 / 30

/-- The number of yahs in one rah -/
def yahs_per_rah : ℚ := 10 / 6

/-- Proves that 432 bahs are equal to 1200 yahs -/
theorem bahs_to_yahs_conversion : 
  432 * bahs_per_rah = 1200 / yahs_per_rah := by sorry

end NUMINAMATH_CALUDE_bahs_to_yahs_conversion_l278_27870


namespace NUMINAMATH_CALUDE_tommy_balloons_l278_27888

/-- Given that Tommy had 26 balloons initially and received 34 more from his mom,
    prove that he ended up with 60 balloons in total. -/
theorem tommy_balloons (initial_balloons : ℕ) (mom_gift : ℕ) : 
  initial_balloons = 26 → mom_gift = 34 → initial_balloons + mom_gift = 60 := by
sorry

end NUMINAMATH_CALUDE_tommy_balloons_l278_27888


namespace NUMINAMATH_CALUDE_prime_factorization_l278_27871

theorem prime_factorization (n : ℕ) (h : n ≥ 2) :
  ∃ (primes : List ℕ), (∀ p ∈ primes, Nat.Prime p) ∧ (n = primes.prod) := by
  sorry

end NUMINAMATH_CALUDE_prime_factorization_l278_27871


namespace NUMINAMATH_CALUDE_max_abs_quadratic_function_bound_l278_27824

theorem max_abs_quadratic_function_bound (a b : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 + a*x + b
  let M := ⨆ (x : ℝ) (hx : x ∈ Set.Icc (-1) 1), |f x|
  M ≥ (1/2 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_max_abs_quadratic_function_bound_l278_27824


namespace NUMINAMATH_CALUDE_more_birds_than_nests_l278_27849

theorem more_birds_than_nests :
  let birds : ℕ := 6
  let nests : ℕ := 3
  birds - nests = 3 :=
by sorry

end NUMINAMATH_CALUDE_more_birds_than_nests_l278_27849


namespace NUMINAMATH_CALUDE_parabola_point_to_directrix_distance_l278_27801

/-- Represents a parabola opening to the right with equation y² = 2px -/
structure Parabola where
  p : ℝ
  opens_right : p > 0

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

theorem parabola_point_to_directrix_distance
  (C : Parabola) (A : Point)
  (h1 : A.x = 1)
  (h2 : A.y = Real.sqrt 5)
  (h3 : A.y ^ 2 = 2 * C.p * A.x) :
  A.x + C.p / 2 = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_to_directrix_distance_l278_27801


namespace NUMINAMATH_CALUDE_negation_of_proposition_l278_27865

theorem negation_of_proposition :
  (¬ (∀ n : ℕ, n^2 < 3*n + 4)) ↔ (∃ n : ℕ, n^2 ≥ 3*n + 4) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l278_27865


namespace NUMINAMATH_CALUDE_cistern_wet_surface_area_l278_27841

/-- Calculates the total wet surface area of a rectangular cistern -/
def total_wet_surface_area (length width depth : ℝ) : ℝ :=
  length * width +  -- bottom area
  2 * length * depth +  -- long sides area
  2 * width * depth  -- short sides area

/-- Theorem: The total wet surface area of a cistern with given dimensions is 62 square meters -/
theorem cistern_wet_surface_area :
  total_wet_surface_area 8 4 1.25 = 62 := by
  sorry


end NUMINAMATH_CALUDE_cistern_wet_surface_area_l278_27841


namespace NUMINAMATH_CALUDE_second_reduction_percentage_store_price_reduction_l278_27856

/-- Given two successive price reductions, calculates the second reduction percentage. -/
theorem second_reduction_percentage 
  (first_reduction : ℝ) 
  (final_price_percentage : ℝ) : ℝ :=
let remaining_after_first := 1 - first_reduction
let second_reduction := 1 - (final_price_percentage / remaining_after_first)
second_reduction

/-- Proves that for the given conditions, the second reduction percentage is 23.5%. -/
theorem store_price_reduction : 
  second_reduction_percentage 0.15 0.765 = 0.235 := by
sorry

end NUMINAMATH_CALUDE_second_reduction_percentage_store_price_reduction_l278_27856


namespace NUMINAMATH_CALUDE_triangle_reconstruction_l278_27853

-- Define the centers of the squares
structure SquareCenters where
  O₁ : ℝ × ℝ
  O₂ : ℝ × ℝ
  O₃ : ℝ × ℝ

-- Define a 90-degree rotation around a point
def rotate90 (center : ℝ × ℝ) (point : ℝ × ℝ) : ℝ × ℝ :=
  sorry

-- Define the composition of rotations
def compositeRotation (centers : SquareCenters) (point : ℝ × ℝ) : ℝ × ℝ :=
  rotate90 centers.O₃ (rotate90 centers.O₂ (rotate90 centers.O₁ point))

-- Theorem stating the existence of an invariant point
theorem triangle_reconstruction (centers : SquareCenters) :
  ∃ (B : ℝ × ℝ), compositeRotation centers B = B :=
sorry

end NUMINAMATH_CALUDE_triangle_reconstruction_l278_27853


namespace NUMINAMATH_CALUDE_fraction_equality_l278_27852

theorem fraction_equality (x y : ℚ) (h : x / y = 2 / 7) : (x + y) / y = 9 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l278_27852


namespace NUMINAMATH_CALUDE_expansion_properties_l278_27815

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the expansion term
def expansionTerm (n r : ℕ) (x : ℝ) : ℝ := 
  (binomial n r : ℝ) * (2^(n-r)) * (3^r) * (x^(n - (4/3)*r))

theorem expansion_properties :
  ∃ (n : ℕ) (x : ℝ),
  -- Condition: ratio of binomial coefficients
  (binomial n 2 : ℝ) / (binomial n 1 : ℝ) = 5/2 →
  -- 1. n = 6
  n = 6 ∧
  -- 2. Coefficient of x^2 term
  (∃ (r : ℕ), n - (4/3)*r = 2 ∧ 
    expansionTerm n r 1 = 4320) ∧
  -- 3. Term with maximum coefficient
  (∃ (r : ℕ), ∀ (k : ℕ), 
    expansionTerm n r x ≥ expansionTerm n k x ∧
    expansionTerm n r 1 = 4860 ∧
    n - (4/3)*r = 2/3) :=
sorry

end NUMINAMATH_CALUDE_expansion_properties_l278_27815


namespace NUMINAMATH_CALUDE_container_production_l278_27880

/-- Represents the production rate of containers per worker per hour -/
def container_rate : ℝ := by sorry

/-- Represents the production rate of covers per worker per hour -/
def cover_rate : ℝ := by sorry

/-- The number of containers produced by 80 workers in 2 hours -/
def containers_80_2 : ℝ := 320

/-- The number of covers produced by 80 workers in 2 hours -/
def covers_80_2 : ℝ := 160

/-- The number of containers produced by 100 workers in 3 hours -/
def containers_100_3 : ℝ := 450

/-- The number of covers produced by 100 workers in 3 hours -/
def covers_100_3 : ℝ := 300

/-- The number of covers produced by 40 workers in 4 hours -/
def covers_40_4 : ℝ := 160

theorem container_production :
  80 * 2 * container_rate = containers_80_2 ∧
  80 * 2 * cover_rate = covers_80_2 ∧
  100 * 3 * container_rate = containers_100_3 ∧
  100 * 3 * cover_rate = covers_100_3 ∧
  40 * 4 * cover_rate = covers_40_4 →
  40 * 4 * container_rate = 160 := by sorry

end NUMINAMATH_CALUDE_container_production_l278_27880


namespace NUMINAMATH_CALUDE_ellipse_equation_l278_27876

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- A parabola with equation y² = 4px where p is the focal distance -/
structure Parabola where
  p : ℝ
  h_pos : 0 < p

/-- A circle with center (h, k) and radius r -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ
  h_pos : 0 < r

/-- The theorem stating the conditions and the result to be proved -/
theorem ellipse_equation (e : Ellipse) (p : Parabola) (c : Circle) 
  (h_focus : e.a^2 - e.b^2 = p.p^2) 
  (h_major_axis : 2 * e.a = c.r) 
  (h_parabola : p.p^2 = 3) :
  e.a^2 = 4 ∧ e.b^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_l278_27876


namespace NUMINAMATH_CALUDE_range_when_a_b_one_values_of_a_b_for_range_zero_one_max_min_a_squared_plus_b_squared_l278_27883

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

-- Part I(i)
theorem range_when_a_b_one (x : ℝ) (h : x ∈ Set.Icc 0 1) :
  f 1 1 x ∈ Set.Icc 1 3 :=
sorry

-- Part I(ii)
theorem values_of_a_b_for_range_zero_one :
  (∀ x ∈ Set.Icc 0 1, f a b x ∈ Set.Icc 0 1) →
  ((a = 0 ∧ b = 0) ∨ (a = -2 ∧ b = 1)) :=
sorry

-- Part II
theorem max_min_a_squared_plus_b_squared 
  (h1 : ∀ x : ℝ, |x| ≥ 2 → f a b x ≥ 0)
  (h2 : ∃ x ∈ Set.Ioo 2 3, ∀ y ∈ Set.Ioo 2 3, f a b x ≥ f a b y)
  (h3 : ∃ x ∈ Set.Ioo 2 3, f a b x = 1) :
  (a^2 + b^2 ≥ 32 ∧ a^2 + b^2 ≤ 74) :=
sorry

end NUMINAMATH_CALUDE_range_when_a_b_one_values_of_a_b_for_range_zero_one_max_min_a_squared_plus_b_squared_l278_27883


namespace NUMINAMATH_CALUDE_flour_cups_needed_l278_27813

-- Define the total number of 1/4 cup scoops
def total_scoops : ℚ := 15

-- Define the amounts of other ingredients in cups
def white_sugar : ℚ := 1
def brown_sugar : ℚ := 1/4
def oil : ℚ := 1/2

-- Define the conversion factor from scoops to cups
def scoops_to_cups : ℚ := 1/4

-- Theorem to prove
theorem flour_cups_needed :
  let other_ingredients_scoops := white_sugar / scoops_to_cups + brown_sugar / scoops_to_cups + oil / scoops_to_cups
  let flour_scoops := total_scoops - other_ingredients_scoops
  let flour_cups := flour_scoops * scoops_to_cups
  flour_cups = 2 := by sorry

end NUMINAMATH_CALUDE_flour_cups_needed_l278_27813


namespace NUMINAMATH_CALUDE_ammonia_formed_l278_27889

/-- Represents a chemical compound in the reaction -/
inductive Compound
| NH4NO3
| NaOH
| NH3
| H2O
| NaNO3

/-- Represents the stoichiometric coefficients in the balanced equation -/
def reaction_coefficients : Compound → ℕ
| Compound.NH4NO3 => 1
| Compound.NaOH => 1
| Compound.NH3 => 1
| Compound.H2O => 1
| Compound.NaNO3 => 1

/-- The number of moles of each reactant available -/
def available_moles : Compound → ℕ
| Compound.NH4NO3 => 2
| Compound.NaOH => 2
| _ => 0

/-- Theorem stating that 2 moles of NH3 are formed in the reaction -/
theorem ammonia_formed :
  let limiting_reactant := min (available_moles Compound.NH4NO3) (available_moles Compound.NaOH)
  limiting_reactant * (reaction_coefficients Compound.NH3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ammonia_formed_l278_27889


namespace NUMINAMATH_CALUDE_base9_726_to_base3_l278_27829

/-- Converts a digit from base 9 to two digits in base 3 -/
def base9ToBase3Digit (d : Nat) : Nat × Nat :=
  sorry

/-- Converts a number from base 9 to base 3 -/
def base9ToBase3 (n : Nat) : Nat :=
  sorry

theorem base9_726_to_base3 :
  base9ToBase3 726 = 210220 :=
sorry

end NUMINAMATH_CALUDE_base9_726_to_base3_l278_27829


namespace NUMINAMATH_CALUDE_linear_function_decreasing_l278_27804

/-- A linear function y = mx + b where m is the slope and b is the y-intercept -/
structure LinearFunction where
  slope : ℝ
  intercept : ℝ

/-- The linear function y = (k-3)x + 2 -/
def f (k : ℝ) : LinearFunction :=
  { slope := k - 3, intercept := 2 }

/-- A function is decreasing if for any x1 < x2, f(x1) > f(x2) -/
def isDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2

/-- The main theorem: The linear function y = (k-3)x + 2 is decreasing iff k < 3 -/
theorem linear_function_decreasing (k : ℝ) :
  isDecreasing (fun x ↦ (f k).slope * x + (f k).intercept) ↔ k < 3 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_decreasing_l278_27804


namespace NUMINAMATH_CALUDE_number_of_lineups_l278_27821

def team_size : ℕ := 15
def lineup_size : ℕ := 5
def cant_play_together : ℕ := 3
def injured : ℕ := 1

theorem number_of_lineups :
  (Nat.choose (team_size - cant_play_together - injured) lineup_size) +
  (cant_play_together * Nat.choose (team_size - cant_play_together - injured) (lineup_size - 1)) = 1452 :=
by sorry

end NUMINAMATH_CALUDE_number_of_lineups_l278_27821


namespace NUMINAMATH_CALUDE_intersection_point_is_solution_l278_27899

/-- The intersection point of two lines -/
def intersection_point : ℝ × ℝ := (3.5, -1.25)

/-- The first line equation -/
def line1 (x y : ℝ) : Prop := 5 * x - 2 * y = 20

/-- The second line equation -/
def line2 (x y : ℝ) : Prop := 3 * x + 2 * y = 8

theorem intersection_point_is_solution :
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y ∧
  ∀ x' y', line1 x' y' ∧ line2 x' y' → x' = x ∧ y' = y :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_is_solution_l278_27899


namespace NUMINAMATH_CALUDE_intersection_distance_l278_27895

/-- The distance between the intersection points of a parabola and a circle -/
theorem intersection_distance (x1 y1 x2 y2 : ℝ) : 
  (y1^2 = 12*x1) →
  (x1^2 + y1^2 - 4*x1 - 6*y1 = 0) →
  (y2^2 = 12*x2) →
  (x2^2 + y2^2 - 4*x2 - 6*y2 = 0) →
  x1 ≠ x2 ∨ y1 ≠ y2 →
  ((x2 - x1)^2 + (y2 - y1)^2)^(1/2) = 3 * 13^(1/2) :=
by sorry

end NUMINAMATH_CALUDE_intersection_distance_l278_27895


namespace NUMINAMATH_CALUDE_largest_ball_radius_is_four_l278_27872

/-- Represents a torus in 3D space -/
structure Torus where
  inner_radius : ℝ
  outer_radius : ℝ
  circle_center : ℝ × ℝ × ℝ
  circle_radius : ℝ

/-- Represents a spherical ball in 3D space -/
structure SphericalBall where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- The largest spherical ball that can be placed on top of a torus -/
def largest_ball_on_torus (t : Torus) : SphericalBall :=
  { center := (0, 0, 4),
    radius := 4 }

/-- Theorem stating that the largest ball on the torus has radius 4 -/
theorem largest_ball_radius_is_four (t : Torus) 
  (h1 : t.inner_radius = 3)
  (h2 : t.outer_radius = 5)
  (h3 : t.circle_center = (4, 0, 1))
  (h4 : t.circle_radius = 1) :
  (largest_ball_on_torus t).radius = 4 := by
  sorry

#check largest_ball_radius_is_four

end NUMINAMATH_CALUDE_largest_ball_radius_is_four_l278_27872


namespace NUMINAMATH_CALUDE_tennis_racket_weight_tennis_racket_weight_proof_l278_27845

theorem tennis_racket_weight : ℝ → ℝ → Prop :=
  fun (racket_weight bicycle_weight : ℝ) =>
    (10 * racket_weight = 8 * bicycle_weight) →
    (4 * bicycle_weight = 120) →
    racket_weight = 24

-- Proof
theorem tennis_racket_weight_proof :
  ∃ (racket_weight bicycle_weight : ℝ),
    tennis_racket_weight racket_weight bicycle_weight :=
by
  sorry

end NUMINAMATH_CALUDE_tennis_racket_weight_tennis_racket_weight_proof_l278_27845


namespace NUMINAMATH_CALUDE_circle_C_equation_l278_27840

-- Define the circles and line
def circle_M (r : ℝ) (x y : ℝ) : Prop := (x + 2)^2 + (y + 2)^2 = r^2
def line_symmetry (x y : ℝ) : Prop := x + y + 2 = 0

-- Define the symmetry condition
def symmetric_circles (C_center : ℝ × ℝ) (r : ℝ) : Prop :=
  let (a, b) := C_center
  (a - (-2)) / 2 + (b - (-2)) / 2 + 2 = 0 ∧ (b + 2) / (a + 2) = 1

-- Theorem statement
theorem circle_C_equation :
  ∀ (r : ℝ), r > 0 →
  ∃ (C_center : ℝ × ℝ),
    (symmetric_circles C_center r) ∧
    ((1 : ℝ) - C_center.1)^2 + ((1 : ℝ) - C_center.2)^2 = 
    C_center.1^2 + C_center.2^2 →
    ∀ (x y : ℝ), x^2 + y^2 = 2 ↔ 
      ((x - C_center.1)^2 + (y - C_center.2)^2 = C_center.1^2 + C_center.2^2) :=
by
  sorry


end NUMINAMATH_CALUDE_circle_C_equation_l278_27840


namespace NUMINAMATH_CALUDE_shortest_side_length_l278_27873

/-- A triangle with an inscribed circle -/
structure InscribedCircleTriangle where
  /-- The length of the first segment of the partitioned side -/
  segment1 : ℝ
  /-- The length of the second segment of the partitioned side -/
  segment2 : ℝ
  /-- The radius of the inscribed circle -/
  radius : ℝ
  /-- The angle opposite the partitioned side in radians -/
  opposite_angle : ℝ

/-- The theorem stating the length of the shortest side of the triangle -/
theorem shortest_side_length (t : InscribedCircleTriangle)
  (h1 : t.segment1 = 7)
  (h2 : t.segment2 = 9)
  (h3 : t.radius = 5)
  (h4 : t.opposite_angle = π / 3) :
  ∃ (shortest_side : ℝ), shortest_side = 20 * (2 + Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_shortest_side_length_l278_27873


namespace NUMINAMATH_CALUDE_exponential_inequality_range_l278_27832

theorem exponential_inequality_range (a : ℝ) : 
  (∀ x : ℝ, (2 : ℝ)^(x^2 - 4*x) > (2 : ℝ)^(2*a*x + a)) ↔ -4 < a ∧ a < -1 := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_range_l278_27832


namespace NUMINAMATH_CALUDE_triangle_side_length_l278_27811

/-- Given a triangle ABC with sides a, b, c and angle C, 
    prove that c = √19 when a + b = 5, ab = 2, and C = 60° --/
theorem triangle_side_length (a b c : ℝ) (C : ℝ) : 
  a + b = 5 → ab = 2 → C = π / 3 → c = Real.sqrt 19 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l278_27811


namespace NUMINAMATH_CALUDE_kangaroo_jump_distance_l278_27803

/-- Proves that a kangaroo jumping up and down a mountain with specific jump patterns covers a total distance of 3036 meters. -/
theorem kangaroo_jump_distance (total_jumps : ℕ) (uphill_distance downhill_distance : ℝ) 
  (h1 : total_jumps = 2024)
  (h2 : uphill_distance = 1)
  (h3 : downhill_distance = 3)
  (h4 : ∃ (uphill_jumps downhill_jumps : ℕ), 
    uphill_jumps + downhill_jumps = total_jumps ∧ 
    uphill_jumps = 3 * downhill_jumps) :
  ∃ (total_distance : ℝ), total_distance = 3036 := by
  sorry

end NUMINAMATH_CALUDE_kangaroo_jump_distance_l278_27803


namespace NUMINAMATH_CALUDE_min_value_expression_l278_27893

theorem min_value_expression (k : ℕ) (hk : k > 0) : 
  (10 : ℝ) / 3 + 32 / 10 ≤ (k : ℝ) / 3 + 32 / k :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l278_27893


namespace NUMINAMATH_CALUDE_box_weight_is_42_l278_27820

/-- The weight of a box of books -/
def box_weight (book_weight : ℕ) (num_books : ℕ) : ℕ :=
  book_weight * num_books

/-- Theorem: The weight of a box of books is 42 pounds -/
theorem box_weight_is_42 :
  box_weight 3 14 = 42 := by
  sorry

end NUMINAMATH_CALUDE_box_weight_is_42_l278_27820


namespace NUMINAMATH_CALUDE_dice_sum_probability_l278_27807

-- Define a die as having 6 sides
def die_sides : ℕ := 6

-- Define the number of dice rolled
def num_dice : ℕ := 4

-- Define the total number of possible outcomes
def total_outcomes : ℕ := die_sides ^ num_dice

-- Define a function to calculate favorable outcomes
noncomputable def favorable_outcomes : ℕ := 
  -- This function would calculate the number of favorable outcomes
  -- Based on the problem, this should be 480, but we don't assume this knowledge
  sorry

-- Theorem statement
theorem dice_sum_probability : 
  (favorable_outcomes : ℚ) / total_outcomes = 10 / 27 := by
  sorry

end NUMINAMATH_CALUDE_dice_sum_probability_l278_27807


namespace NUMINAMATH_CALUDE_factor_x6_minus_64_l278_27847

theorem factor_x6_minus_64 (x : ℝ) : 
  x^6 - 64 = (x - 2) * (x + 2) * (x^2 + 2*x + 4) * (x^2 - 2*x + 4) := by
  sorry

end NUMINAMATH_CALUDE_factor_x6_minus_64_l278_27847


namespace NUMINAMATH_CALUDE_hourly_charge_is_correct_l278_27854

/-- The hourly charge for renting a bike -/
def hourly_charge : ℝ := 7

/-- The fixed fee for renting a bike -/
def fixed_fee : ℝ := 17

/-- The number of hours Tom rented the bike -/
def rental_hours : ℝ := 9

/-- The total cost Tom paid for renting the bike -/
def total_cost : ℝ := 80

/-- Theorem stating that the hourly charge is correct given the conditions -/
theorem hourly_charge_is_correct : 
  fixed_fee + rental_hours * hourly_charge = total_cost := by sorry

end NUMINAMATH_CALUDE_hourly_charge_is_correct_l278_27854
