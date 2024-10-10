import Mathlib

namespace rectangle_ratio_l1806_180630

theorem rectangle_ratio (w l : ℝ) (h1 : w = 5) (h2 : l * w = 75) : l / w = 3 := by
  sorry

end rectangle_ratio_l1806_180630


namespace unique_whole_number_between_l1806_180677

theorem unique_whole_number_between (N : ℤ) : 
  (5.5 < (N : ℚ) / 4 ∧ (N : ℚ) / 4 < 6) ↔ N = 23 := by
sorry

end unique_whole_number_between_l1806_180677


namespace park_visitors_l1806_180640

theorem park_visitors (total : ℕ) (men_fraction : ℚ) (women_student_fraction : ℚ) 
  (h1 : total = 1260)
  (h2 : men_fraction = 7 / 18)
  (h3 : women_student_fraction = 6 / 11) :
  (total : ℚ) * (1 - men_fraction) * women_student_fraction = 420 := by
  sorry

end park_visitors_l1806_180640


namespace min_k_existence_l1806_180614

open Real

theorem min_k_existence (k : ℕ) : (∃ x₀ : ℝ, x₀ > 2 ∧ k * (x₀ - 2) > x₀ * (log x₀ + 1)) ↔ k ≥ 5 :=
sorry

end min_k_existence_l1806_180614


namespace equation_solutions_l1806_180634

theorem equation_solutions (n : ℕ+) : 
  (∃! (s : Finset (ℕ+ × ℕ+ × ℕ+)), 
    s.card = 10 ∧ 
    ∀ (x y z : ℕ+), (x, y, z) ∈ s ↔ 3*x + 3*y + 2*z = n) ↔ 
  n = 17 := by sorry

end equation_solutions_l1806_180634


namespace tangent_parallel_to_chord_l1806_180652

/-- The curve function -/
def f (x : ℝ) : ℝ := 4*x - x^2

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 4 - 2*x

theorem tangent_parallel_to_chord :
  let A : ℝ × ℝ := (4, 0)
  let B : ℝ × ℝ := (2, 4)
  let P : ℝ × ℝ := (3, 3)
  let chord_slope : ℝ := (B.2 - A.2) / (B.1 - A.1)
  P.2 = f P.1 ∧ f' P.1 = chord_slope := by sorry

end tangent_parallel_to_chord_l1806_180652


namespace parabola_directrix_tangent_circle_l1806_180633

/-- Given a parabola y^2 = -2px where p > 0, if its directrix is tangent to the circle (x-5)^2 + y^2 = 25, then p = 20 -/
theorem parabola_directrix_tangent_circle (p : ℝ) : 
  p > 0 →
  (∃ x y : ℝ, y^2 = -2*p*x) →
  (∃ x : ℝ, x = p/2 ∧ (x-5)^2 = 25) →
  p = 20 := by
  sorry

end parabola_directrix_tangent_circle_l1806_180633


namespace degree_of_g_l1806_180685

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := -9*x^5 + 5*x^4 + 2*x^2 - x + 6

-- Define the theorem
theorem degree_of_g (g : ℝ → ℝ) :
  (∃ c : ℝ, ∀ x : ℝ, f x + g x = c) →  -- degree of f(x) + g(x) is 0
  (∃ a₅ a₄ a₃ a₂ a₁ a₀ : ℝ, a₅ ≠ 0 ∧ 
    ∀ x : ℝ, g x = a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →  -- g(x) is a polynomial of degree 5
  true :=
by sorry

end degree_of_g_l1806_180685


namespace negation_of_universal_proposition_l1806_180603

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) := by sorry

end negation_of_universal_proposition_l1806_180603


namespace garden_length_is_40_l1806_180654

/-- Represents a rectangular garden with given properties -/
structure Garden where
  total_distance : ℝ
  length_walks : ℕ
  perimeter_walks : ℕ
  width_ratio : ℝ
  length : ℝ

/-- Theorem stating that the garden's length is 40 meters given the conditions -/
theorem garden_length_is_40 (g : Garden)
  (h1 : g.total_distance = 960)
  (h2 : g.length_walks = 24)
  (h3 : g.perimeter_walks = 8)
  (h4 : g.width_ratio = 1/2)
  (h5 : g.length * g.length_walks = g.total_distance)
  (h6 : (2 * g.length + 2 * (g.width_ratio * g.length)) * g.perimeter_walks = g.total_distance) :
  g.length = 40 := by
  sorry

end garden_length_is_40_l1806_180654


namespace meter_to_skips_conversion_l1806_180616

/-- Proves that 1 meter is equivalent to (g*b*f*d)/(a*e*h*c) skips given the measurement relationships -/
theorem meter_to_skips_conversion
  (a b c d e f g h : ℝ)
  (hops_to_skips : a * 1 = b)
  (jumps_to_hops : c * 1 = d)
  (leaps_to_jumps : e * 1 = f)
  (leaps_to_meters : g * 1 = h)
  (a_pos : 0 < a)
  (c_pos : 0 < c)
  (e_pos : 0 < e)
  (h_pos : 0 < h) :
  1 = (g * b * f * d) / (a * e * h * c) :=
sorry

end meter_to_skips_conversion_l1806_180616


namespace four_valid_numbers_l1806_180694

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 1000 ∧ n ≤ 9999) ∧
  (∃ (a b c d : ℕ), n = 1000 * a + 100 * b + 10 * c + d ∧
    ((a = 1 ∧ b = 9 ∧ c = 8 ∧ d = 8) ∨
     (a = 1 ∧ b = 8 ∧ c = 9 ∧ d = 8) ∨
     (a = 1 ∧ b = 8 ∧ c = 8 ∧ d = 9) ∨
     (a = 9 ∧ b = 1 ∧ c = 8 ∧ d = 8) ∨
     (a = 9 ∧ b = 8 ∧ c = 1 ∧ d = 8) ∨
     (a = 9 ∧ b = 8 ∧ c = 8 ∧ d = 1) ∨
     (a = 8 ∧ b = 1 ∧ c = 9 ∧ d = 8) ∨
     (a = 8 ∧ b = 1 ∧ c = 8 ∧ d = 9) ∨
     (a = 8 ∧ b = 9 ∧ c = 1 ∧ d = 8) ∨
     (a = 8 ∧ b = 9 ∧ c = 8 ∧ d = 1) ∨
     (a = 8 ∧ b = 8 ∧ c = 1 ∧ d = 9) ∨
     (a = 8 ∧ b = 8 ∧ c = 9 ∧ d = 1))) ∧
  n % 11 = 8

theorem four_valid_numbers :
  ∃! (s : Finset ℕ), (∀ n ∈ s, is_valid_number n) ∧ s.card = 4 :=
sorry

end four_valid_numbers_l1806_180694


namespace geometric_sum_specific_l1806_180666

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_specific :
  geometric_sum (3/4) (3/4) 12 = 48758625/16777216 := by
  sorry

end geometric_sum_specific_l1806_180666


namespace point_movement_l1806_180649

def number_line_move (start : ℤ) (move : ℤ) : ℤ :=
  start + move

theorem point_movement :
  let point_A : ℤ := -3
  let movement : ℤ := 6
  let point_B : ℤ := number_line_move point_A movement
  point_B = 3 := by sorry

end point_movement_l1806_180649


namespace factorization_bound_l1806_180636

/-- The number of ways to factorize k into a product of integers greater than 1 -/
def f (k : ℕ) : ℕ :=
  sorry

/-- Theorem: For any integer n > 1 and any prime factor p of n,
    the number of ways to factorize n is less than or equal to n/p -/
theorem factorization_bound (n : ℕ) (p : ℕ) (h1 : n > 1) (h2 : Nat.Prime p) (h3 : p ∣ n) :
  f n ≤ n / p :=
sorry

end factorization_bound_l1806_180636


namespace book_sale_revenue_l1806_180690

theorem book_sale_revenue (total_books : ℕ) (sold_books : ℕ) (price_per_book : ℕ) 
  (h1 : sold_books = (2 * total_books) / 3)
  (h2 : total_books - sold_books = 36)
  (h3 : price_per_book = 4) :
  sold_books * price_per_book = 288 := by
sorry

end book_sale_revenue_l1806_180690


namespace twelve_bushes_for_sixty_zucchinis_l1806_180668

/-- The number of blueberry bushes needed to obtain a given number of zucchinis -/
def bushes_needed (zucchinis : ℕ) (containers_per_bush : ℕ) (containers_per_trade : ℕ) (zucchinis_per_trade : ℕ) : ℕ :=
  (zucchinis * containers_per_trade) / (zucchinis_per_trade * containers_per_bush)

/-- Theorem: 12 bushes are needed to obtain 60 zucchinis -/
theorem twelve_bushes_for_sixty_zucchinis :
  bushes_needed 60 10 6 3 = 12 := by
  sorry

end twelve_bushes_for_sixty_zucchinis_l1806_180668


namespace quadratic_inequality_empty_iff_a_eq_one_l1806_180624

/-- The quadratic function f(x) = ax^2 - (a+1)x + 1 --/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (a + 1) * x + 1

/-- The solution set of f(x) < 0 is empty --/
def has_empty_solution_set (a : ℝ) : Prop :=
  ∀ x : ℝ, f a x ≥ 0

theorem quadratic_inequality_empty_iff_a_eq_one :
  ∀ a : ℝ, has_empty_solution_set a ↔ a = 1 := by sorry

end quadratic_inequality_empty_iff_a_eq_one_l1806_180624


namespace sin_2012_degrees_l1806_180681

theorem sin_2012_degrees : Real.sin (2012 * π / 180) = -Real.sin (32 * π / 180) := by
  sorry

end sin_2012_degrees_l1806_180681


namespace not_prime_n_l1806_180602

theorem not_prime_n (p a b c n : ℕ) : 
  Prime p → 
  0 < a → 0 < b → 0 < c → 0 < n →
  a < p → b < p → c < p →
  p^2 ∣ a + (n-1) * b →
  p^2 ∣ b + (n-1) * c →
  p^2 ∣ c + (n-1) * a →
  ¬ Prime n :=
by sorry


end not_prime_n_l1806_180602


namespace pencils_taken_l1806_180699

theorem pencils_taken (initial_pencils : ℕ) (pencils_left : ℕ) (h1 : initial_pencils = 34) (h2 : pencils_left = 12) :
  initial_pencils - pencils_left = 22 := by
sorry

end pencils_taken_l1806_180699


namespace bridge_length_l1806_180628

/-- The length of a bridge given specific train and crossing conditions -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 135 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  ∃ (bridge_length : ℝ),
    bridge_length = (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length ∧
    bridge_length = 240 :=
by sorry

end bridge_length_l1806_180628


namespace initial_candies_l1806_180686

theorem initial_candies : ∃ x : ℕ, (x - 29) / 13 = 15 ∧ x = 224 := by
  sorry

end initial_candies_l1806_180686


namespace kim_shirts_l1806_180695

theorem kim_shirts (D : ℕ) : 
  (2 / 3 : ℚ) * (12 * D) = 32 → D = 4 := by
  sorry

end kim_shirts_l1806_180695


namespace apple_distribution_l1806_180670

theorem apple_distribution (total_apples : ℕ) (new_people : ℕ) (apple_decrease : ℕ) :
  total_apples = 1430 →
  new_people = 45 →
  apple_decrease = 9 →
  ∃ (original_people : ℕ),
    original_people > 0 ∧
    (total_apples / original_people : ℚ) - (total_apples / (original_people + new_people) : ℚ) = apple_decrease ∧
    total_apples / original_people = 22 :=
by sorry

end apple_distribution_l1806_180670


namespace hexagon_interior_angles_sum_l1806_180658

/-- The sum of interior angles of a polygon with n sides is (n - 2) * 180 degrees -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- A hexagon has 6 sides -/
def hexagon_sides : ℕ := 6

/-- The sum of the measures of the six interior angles of a hexagon is 720 degrees -/
theorem hexagon_interior_angles_sum :
  sum_interior_angles hexagon_sides = 720 := by
  sorry

end hexagon_interior_angles_sum_l1806_180658


namespace coffee_package_size_l1806_180639

theorem coffee_package_size (total_coffee : ℕ) (large_package_size : ℕ) (large_package_count : ℕ) (small_package_count_diff : ℕ) :
  total_coffee = 115 →
  large_package_size = 10 →
  large_package_count = 7 →
  small_package_count_diff = 2 →
  ∃ (small_package_size : ℕ),
    small_package_size = 5 ∧
    total_coffee = (large_package_size * large_package_count) + (small_package_size * (large_package_count + small_package_count_diff)) :=
by sorry

end coffee_package_size_l1806_180639


namespace jaden_final_car_count_l1806_180641

/-- The number of toy cars Jaden has after all transactions -/
def final_car_count (initial : ℕ) (bought : ℕ) (birthday : ℕ) (to_sister : ℕ) (to_friend : ℕ) : ℕ :=
  initial + bought + birthday - to_sister - to_friend

/-- Theorem stating that Jaden's final car count is 43 -/
theorem jaden_final_car_count :
  final_car_count 14 28 12 8 3 = 43 := by
  sorry

end jaden_final_car_count_l1806_180641


namespace quadratic_inequality_has_solution_l1806_180638

theorem quadratic_inequality_has_solution : ∃ x : ℝ, x^2 + 2*x - 3 < 0 := by
  sorry

end quadratic_inequality_has_solution_l1806_180638


namespace september_birth_percentage_l1806_180679

theorem september_birth_percentage (total_authors : ℕ) (september_authors : ℕ) :
  total_authors = 120 →
  september_authors = 15 →
  (september_authors : ℚ) / (total_authors : ℚ) * 100 = 12.5 := by
  sorry

end september_birth_percentage_l1806_180679


namespace complex_equation_solution_l1806_180692

theorem complex_equation_solution (a b c : ℤ) : 
  (a * (3 - Complex.I)^4 + b * (3 - Complex.I)^3 + c * (3 - Complex.I)^2 + b * (3 - Complex.I) + a = 0) →
  (Int.gcd a (Int.gcd b c) = 1) →
  (abs c = 109) := by
  sorry

end complex_equation_solution_l1806_180692


namespace smallest_1755_more_than_sum_of_digits_l1806_180655

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- Property that a number is 1755 more than the sum of its digits -/
def is1755MoreThanSumOfDigits (n : ℕ) : Prop :=
  n = sumOfDigits n + 1755

/-- Theorem stating that 1770 is the smallest natural number that is 1755 more than the sum of its digits -/
theorem smallest_1755_more_than_sum_of_digits :
  (1770 = sumOfDigits 1770 + 1755) ∧
  ∀ m : ℕ, m < 1770 → m ≠ sumOfDigits m + 1755 :=
by sorry

end smallest_1755_more_than_sum_of_digits_l1806_180655


namespace max_total_length_tetrahedron_l1806_180647

/-- Represents a tetrahedron with edge lengths a, b, c, d, e, f --/
structure Tetrahedron where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ
  positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f

/-- The condition that at most one edge is longer than 1 --/
def atMostOneLongerThanOne (t : Tetrahedron) : Prop :=
  (t.a ≤ 1 ∨ (t.b ≤ 1 ∧ t.c ≤ 1 ∧ t.d ≤ 1 ∧ t.e ≤ 1 ∧ t.f ≤ 1)) ∧
  (t.b ≤ 1 ∨ (t.a ≤ 1 ∧ t.c ≤ 1 ∧ t.d ≤ 1 ∧ t.e ≤ 1 ∧ t.f ≤ 1)) ∧
  (t.c ≤ 1 ∨ (t.a ≤ 1 ∧ t.b ≤ 1 ∧ t.d ≤ 1 ∧ t.e ≤ 1 ∧ t.f ≤ 1)) ∧
  (t.d ≤ 1 ∨ (t.a ≤ 1 ∧ t.b ≤ 1 ∧ t.c ≤ 1 ∧ t.e ≤ 1 ∧ t.f ≤ 1)) ∧
  (t.e ≤ 1 ∨ (t.a ≤ 1 ∧ t.b ≤ 1 ∧ t.c ≤ 1 ∧ t.d ≤ 1 ∧ t.f ≤ 1)) ∧
  (t.f ≤ 1 ∨ (t.a ≤ 1 ∧ t.b ≤ 1 ∧ t.c ≤ 1 ∧ t.d ≤ 1 ∧ t.e ≤ 1))

/-- The total length of all edges in a tetrahedron --/
def totalLength (t : Tetrahedron) : ℝ := t.a + t.b + t.c + t.d + t.e + t.f

/-- The theorem stating the maximum total length of edges in a tetrahedron --/
theorem max_total_length_tetrahedron :
  ∀ t : Tetrahedron, atMostOneLongerThanOne t → totalLength t ≤ 5 + Real.sqrt 3 :=
sorry

end max_total_length_tetrahedron_l1806_180647


namespace quadratic_real_roots_quadratic_integer_roots_l1806_180632

/-- The quadratic equation kx^2 + (k+1)x + (k-1) = 0 -/
def quadratic_equation (k x : ℝ) : Prop :=
  k * x^2 + (k + 1) * x + (k - 1) = 0

/-- The set of k values for which the equation has real roots -/
def real_roots_set : Set ℝ :=
  {k | (3 - 2 * Real.sqrt 3) / 3 ≤ k ∧ k ≤ (3 + 2 * Real.sqrt 3) / 3}

/-- The set of k values for which the equation has integer roots -/
def integer_roots_set : Set ℝ :=
  {0, 1, -1/7}

theorem quadratic_real_roots :
  ∀ k : ℝ, (∃ x : ℝ, quadratic_equation k x) ↔ k ∈ real_roots_set :=
sorry

theorem quadratic_integer_roots :
  ∀ k : ℝ, (∃ x : ℤ, quadratic_equation k (x : ℝ)) ↔ k ∈ integer_roots_set :=
sorry

end quadratic_real_roots_quadratic_integer_roots_l1806_180632


namespace solve_jerichos_money_problem_l1806_180622

def jerichos_money_problem (initial_amount debt_to_annika : ℕ) : Prop :=
  let debt_to_manny := 2 * debt_to_annika
  let total_debt := debt_to_annika + debt_to_manny
  let remaining_amount := initial_amount - total_debt
  (initial_amount = 3 * 90) ∧ 
  (debt_to_annika = 20) ∧
  (remaining_amount = 210)

theorem solve_jerichos_money_problem :
  jerichos_money_problem 270 20 := by sorry

end solve_jerichos_money_problem_l1806_180622


namespace inequality_proof_l1806_180605

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3/2 := by
  sorry

end inequality_proof_l1806_180605


namespace sin_15_cos_15_value_l1806_180661

theorem sin_15_cos_15_value : (1/4) * Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1/16 := by
  sorry

end sin_15_cos_15_value_l1806_180661


namespace sixth_point_equals_initial_l1806_180697

/-- Triangle in a plane --/
structure Triangle where
  A₀ : ℝ × ℝ
  A₁ : ℝ × ℝ
  A₂ : ℝ × ℝ

/-- Symmetric point with respect to a given point --/
def symmetric_point (P : ℝ × ℝ) (A : ℝ × ℝ) : ℝ × ℝ :=
  (2 * A.1 - P.1, 2 * A.2 - P.2)

/-- Generate the next point in the sequence --/
def next_point (P : ℝ × ℝ) (i : ℕ) (T : Triangle) : ℝ × ℝ :=
  match i % 3 with
  | 0 => symmetric_point P T.A₀
  | 1 => symmetric_point P T.A₁
  | _ => symmetric_point P T.A₂

/-- Generate the i-th point in the sequence --/
def P (i : ℕ) (P₀ : ℝ × ℝ) (T : Triangle) : ℝ × ℝ :=
  match i with
  | 0 => P₀
  | n + 1 => next_point (P n P₀ T) (n + 1) T

theorem sixth_point_equals_initial (P₀ : ℝ × ℝ) (T : Triangle) :
  P 6 P₀ T = P₀ := by
  sorry

end sixth_point_equals_initial_l1806_180697


namespace dvaneft_percentage_range_l1806_180680

/-- Represents the share packages in the auction --/
structure SharePackages where
  razneft : ℕ
  dvaneft : ℕ
  trineft : ℕ

/-- Represents the prices of individual shares --/
structure SharePrices where
  razneft : ℝ
  dvaneft : ℝ
  trineft : ℝ

/-- Main theorem about the percentage range of Dvaneft shares --/
theorem dvaneft_percentage_range 
  (packages : SharePackages) 
  (prices : SharePrices) : 
  /- Total shares of Razneft and Dvaneft equals shares of Trineft -/
  (packages.razneft + packages.dvaneft = packages.trineft) → 
  /- Dvaneft package is 3 times cheaper than Razneft package -/
  (3 * prices.dvaneft * packages.dvaneft = prices.razneft * packages.razneft) → 
  /- Total cost of Razneft and Dvaneft equals cost of Trineft -/
  (prices.razneft * packages.razneft + prices.dvaneft * packages.dvaneft = 
   prices.trineft * packages.trineft) → 
  /- Price difference between Razneft and Dvaneft share is between 10,000 and 18,000 -/
  (10000 ≤ prices.razneft - prices.dvaneft ∧ prices.razneft - prices.dvaneft ≤ 18000) → 
  /- Price of Trineft share is between 18,000 and 42,000 -/
  (18000 ≤ prices.trineft ∧ prices.trineft ≤ 42000) → 
  /- The percentage of Dvaneft shares is between 15% and 25% -/
  (15 ≤ 100 * packages.dvaneft / (packages.razneft + packages.dvaneft + packages.trineft) ∧
   100 * packages.dvaneft / (packages.razneft + packages.dvaneft + packages.trineft) ≤ 25) :=
by sorry

end dvaneft_percentage_range_l1806_180680


namespace find_divisor_l1806_180604

theorem find_divisor (dividend quotient remainder : ℕ) (h1 : dividend = 217) (h2 : quotient = 54) (h3 : remainder = 1) :
  ∃ (divisor : ℕ), dividend = divisor * quotient + remainder ∧ divisor = 4 := by
  sorry

end find_divisor_l1806_180604


namespace f_odd_g_even_l1806_180626

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the main property
axiom main_property : ∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * g y

-- Define f(0) = 0
axiom f_zero : f 0 = 0

-- Define f is not identically zero
axiom f_not_zero : ∃ x : ℝ, f x ≠ 0

-- Theorem to prove
theorem f_odd_g_even :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ y : ℝ, g (-y) = g y) :=
sorry

end f_odd_g_even_l1806_180626


namespace special_function_at_2021_l1806_180673

/-- A function satisfying the given conditions -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x > 0, f x > 0) ∧ 
  (∀ x y, x > y ∧ y > 0 → f (x - y) = Real.sqrt (f (x * y) + 2))

/-- The main theorem stating that any function satisfying SpecialFunction has f(2021) = 2 -/
theorem special_function_at_2021 (f : ℝ → ℝ) (h : SpecialFunction f) : f 2021 = 2 := by
  sorry

end special_function_at_2021_l1806_180673


namespace no_first_quadrant_intersection_l1806_180662

/-- A linear function y = -3x + m -/
def linear_function (x : ℝ) (m : ℝ) : ℝ := -3 * x + m

/-- The first quadrant of the coordinate plane -/
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

theorem no_first_quadrant_intersection :
  ∀ x y : ℝ, first_quadrant x y → linear_function x (-1) ≠ y := by
  sorry

end no_first_quadrant_intersection_l1806_180662


namespace correct_admin_in_sample_l1806_180683

/-- Represents the composition of staff in a school -/
structure StaffComposition where
  total : ℕ
  administrative : ℕ
  teaching : ℕ
  teaching_support : ℕ

/-- Represents a stratified sample from the staff -/
structure StratifiedSample where
  size : ℕ
  administrative : ℕ

/-- Calculates the correct number of administrative personnel in a stratified sample -/
def calculate_admin_in_sample (staff : StaffComposition) (sample : StratifiedSample) : ℕ :=
  (staff.administrative * sample.size) / staff.total

/-- The theorem to be proved -/
theorem correct_admin_in_sample (staff : StaffComposition) (sample : StratifiedSample) : 
  staff.total = 200 →
  staff.administrative = 24 →
  staff.teaching = 10 * staff.teaching_support →
  sample.size = 50 →
  calculate_admin_in_sample staff sample = 6 := by
  sorry

end correct_admin_in_sample_l1806_180683


namespace tank_filling_time_l1806_180645

theorem tank_filling_time (fill_rate_1 fill_rate_2 remaining_time : ℝ) 
  (h1 : fill_rate_1 = 1 / 20)
  (h2 : fill_rate_2 = 1 / 60)
  (h3 : remaining_time = 20.000000000000004)
  : ∃ t : ℝ, t * (fill_rate_1 + fill_rate_2) + remaining_time * fill_rate_2 = 1 ∧ t = 10 := by
  sorry

end tank_filling_time_l1806_180645


namespace inequality_coverage_l1806_180669

theorem inequality_coverage (a : ℝ) : 
  (∀ x : ℝ, (2 * a - x > 1 ∧ 2 * x + 5 > 3 * a) → (1 ≤ x ∧ x ≤ 6)) →
  (7/3 ≤ a ∧ a ≤ 7/2) :=
by sorry

end inequality_coverage_l1806_180669


namespace data_transmission_time_l1806_180642

theorem data_transmission_time : 
  let num_blocks : ℕ := 60
  let chunks_per_block : ℕ := 512
  let transmission_rate : ℕ := 120  -- chunks per second
  let total_chunks : ℕ := num_blocks * chunks_per_block
  let transmission_time_seconds : ℕ := total_chunks / transmission_rate
  let transmission_time_minutes : ℕ := transmission_time_seconds / 60
  transmission_time_minutes = 4 := by
  sorry

end data_transmission_time_l1806_180642


namespace least_absolute_prime_l1806_180644

theorem least_absolute_prime (n : ℤ) : 
  Nat.Prime n.natAbs → 101 * n^2 ≤ 3600 → (∀ m : ℤ, Nat.Prime m.natAbs → 101 * m^2 ≤ 3600 → n.natAbs ≤ m.natAbs) → n.natAbs = 2 :=
by sorry

end least_absolute_prime_l1806_180644


namespace cube_sum_of_roots_l1806_180623

theorem cube_sum_of_roots (p q r : ℝ) : 
  (p^3 - p^2 + p - 2 = 0) → 
  (q^3 - q^2 + q - 2 = 0) → 
  (r^3 - r^2 + r - 2 = 0) → 
  p^3 + q^3 + r^3 = 4 := by
sorry

end cube_sum_of_roots_l1806_180623


namespace odd_function_representation_l1806_180608

def f (x : ℝ) : ℝ := x * (abs x - 2)

theorem odd_function_representation (f : ℝ → ℝ) :
  (∀ x, f (-x) = -f x) →  -- f is an odd function
  (∀ x ≥ 0, f x = x^2 - 2*x) →  -- definition for x ≥ 0
  (∀ x, f x = x * (abs x - 2)) :=  -- claim to prove
by
  sorry

end odd_function_representation_l1806_180608


namespace negation_of_existence_cubic_inequality_negation_l1806_180691

theorem negation_of_existence (f : ℝ → Prop) :
  (¬ ∃ x : ℝ, x > 0 ∧ f x) ↔ (∀ x : ℝ, x > 0 → ¬ f x) :=
by sorry

theorem cubic_inequality_negation :
  (¬ ∃ x : ℝ, x > 0 ∧ x^3 - x + 1 > 0) ↔ (∀ x : ℝ, x > 0 → x^3 - x + 1 ≤ 0) :=
by sorry

end negation_of_existence_cubic_inequality_negation_l1806_180691


namespace intersection_symmetry_l1806_180660

/-- Prove that if a line y = kx intersects a circle (x-1)^2 + y^2 = 1 at two points 
    symmetric with respect to the line x - y + b = 0, then k = -1 and b = -1. -/
theorem intersection_symmetry (k b : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    -- Line equation
    y₁ = k * x₁ ∧ y₂ = k * x₂ ∧
    -- Circle equation
    (x₁ - 1)^2 + y₁^2 = 1 ∧ (x₂ - 1)^2 + y₂^2 = 1 ∧
    -- Symmetry condition
    (x₁ + x₂) / 2 - (y₁ + y₂) / 2 + b = 0) →
  k = -1 ∧ b = -1 := by
sorry

end intersection_symmetry_l1806_180660


namespace units_digit_of_power_plus_six_l1806_180631

theorem units_digit_of_power_plus_six (y : ℕ+) :
  (7^y.val + 6) % 10 = 9 ↔ y.val % 4 = 3 := by sorry

end units_digit_of_power_plus_six_l1806_180631


namespace square_sum_given_conditions_l1806_180646

theorem square_sum_given_conditions (x y : ℝ) 
  (h1 : (x + y)^2 = 16) 
  (h2 : x * y = 6) : 
  x^2 + y^2 = 4 := by
sorry

end square_sum_given_conditions_l1806_180646


namespace factor_expression_l1806_180674

theorem factor_expression (x : ℝ) : 63 * x^19 + 147 * x^38 = 21 * x^19 * (3 + 7 * x^19) := by
  sorry

end factor_expression_l1806_180674


namespace bus_fare_cost_l1806_180651

/-- Represents the cost of a bus fare for one person one way -/
def bus_fare : ℝ := 1.5

/-- Represents the cost of zoo entry for one person -/
def zoo_entry : ℝ := 5

/-- Represents the total money brought -/
def total_money : ℝ := 40

/-- Represents the money left after zoo entry and bus fare -/
def money_left : ℝ := 24

/-- Proves that the bus fare cost per person one way is $1.50 -/
theorem bus_fare_cost : 
  2 * zoo_entry + 4 * bus_fare = total_money - money_left :=
by sorry

end bus_fare_cost_l1806_180651


namespace constant_m_value_l1806_180688

theorem constant_m_value (x y z m : ℝ) :
  (5^2 / (x + y) = m / (x + 2*z)) ∧ (m / (x + 2*z) = 7^2 / (y - 2*z)) →
  m = 74 := by
  sorry

end constant_m_value_l1806_180688


namespace max_m_value_l1806_180687

theorem max_m_value (m : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → x^2 - x + 1/2 ≥ m) → 
  m ≤ 1/4 := by
sorry

end max_m_value_l1806_180687


namespace correct_subtraction_l1806_180610

theorem correct_subtraction (x : ℤ) : x - 64 = 122 → x - 46 = 140 := by
  sorry

end correct_subtraction_l1806_180610


namespace total_fruits_is_54_l1806_180663

/-- The total number of fruits picked by all people -/
def total_fruits (melanie_plums melanie_apples dan_plums dan_oranges sally_plums sally_cherries thomas_plums thomas_peaches : ℕ) : ℕ :=
  melanie_plums + melanie_apples + dan_plums + dan_oranges + sally_plums + sally_cherries + thomas_plums + thomas_peaches

/-- Theorem stating that the total number of fruits picked is 54 -/
theorem total_fruits_is_54 :
  total_fruits 4 6 9 2 3 10 15 5 = 54 := by
  sorry

#eval total_fruits 4 6 9 2 3 10 15 5

end total_fruits_is_54_l1806_180663


namespace expression_value_l1806_180648

theorem expression_value : 
  let x : ℝ := 2
  let y : ℝ := -3
  let z : ℝ := 1
  2 * x^2 + 3 * y^2 - z^2 + 4 * x * y - 6 * x * z = -2 := by
  sorry

end expression_value_l1806_180648


namespace range_of_m_l1806_180619

theorem range_of_m (x m : ℝ) : 
  (∀ x, (|x - 3| ≤ 2 → (x - m + 1) * (x - m - 1) ≤ 0) ∧
        ((x < 1 ∨ x > 5) → (x < m - 1 ∨ x > m + 1)) ∧
        (∃ x, (x < m - 1 ∨ x > m + 1) ∧ ¬(x < 1 ∨ x > 5)))
  → 2 ≤ m ∧ m ≤ 4 :=
by sorry

end range_of_m_l1806_180619


namespace x_less_than_2_necessary_not_sufficient_l1806_180671

theorem x_less_than_2_necessary_not_sufficient :
  (∃ x : ℝ, x < 2 ∧ x^2 - 2*x ≥ 0) ∧
  (∀ x : ℝ, x^2 - 2*x < 0 → x < 2) :=
by sorry

end x_less_than_2_necessary_not_sufficient_l1806_180671


namespace sum_of_base6_series_l1806_180656

/-- Represents a number in base 6 -/
def Base6 := Nat

/-- Converts a base 6 number to decimal -/
def to_decimal (n : Base6) : Nat :=
  sorry

/-- Converts a decimal number to base 6 -/
def to_base6 (n : Nat) : Base6 :=
  sorry

/-- The sum of an arithmetic series in base 6 -/
def arithmetic_sum_base6 (first last : Base6) (common_diff : Base6) : Base6 :=
  sorry

/-- Theorem: The sum of the series 2₆ + 4₆ + 6₆ + ⋯ + 100₆ in base 6 is 1330₆ -/
theorem sum_of_base6_series : 
  arithmetic_sum_base6 (to_base6 2) (to_base6 36) (to_base6 2) = to_base6 342 :=
by sorry

end sum_of_base6_series_l1806_180656


namespace simultaneous_ring_theorem_l1806_180684

def bell_ring_time (start_hour start_minute : ℕ) (interval_minutes : ℕ) : ℕ × ℕ := sorry

def next_simultaneous_ring 
  (start_hour start_minute : ℕ) 
  (interval1 interval2 interval3 : ℕ) : ℕ × ℕ := sorry

theorem simultaneous_ring_theorem 
  (h1 : interval1 = 18)
  (h2 : interval2 = 24)
  (h3 : interval3 = 30)
  (h4 : start_hour = 10)
  (h5 : start_minute = 0) :
  next_simultaneous_ring start_hour start_minute interval1 interval2 interval3 = (16, 0) := by
  sorry

end simultaneous_ring_theorem_l1806_180684


namespace sin_2alpha_l1806_180678

theorem sin_2alpha (α : ℝ) (h : Real.sin (α + π/4) = 1/3) : Real.sin (2*α) = -7/9 := by
  sorry

end sin_2alpha_l1806_180678


namespace range_of_a_l1806_180676

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, ∃ y₁ y₂ : ℝ, y₁ ≠ y₂ ∧ 
    Real.exp x * (y₁ - x) - a * Real.exp (2 * y₁ - x) = 0 ∧
    Real.exp x * (y₂ - x) - a * Real.exp (2 * y₂ - x) = 0) →
  0 < a ∧ a < 1 / (2 * Real.exp 1) :=
by sorry

end range_of_a_l1806_180676


namespace donna_marcia_pencils_l1806_180643

/-- The number of pencils Cindi bought -/
def cindi_pencils : ℕ := 60

/-- The number of pencils Marcia bought -/
def marcia_pencils : ℕ := 2 * cindi_pencils

/-- The number of pencils Donna bought -/
def donna_pencils : ℕ := 3 * marcia_pencils

/-- The total number of pencils bought by Donna and Marcia -/
def total_pencils : ℕ := donna_pencils + marcia_pencils

theorem donna_marcia_pencils :
  total_pencils = 480 :=
sorry

end donna_marcia_pencils_l1806_180643


namespace smaller_square_area_percentage_l1806_180693

/-- A circle with an inscribed square and a smaller square -/
structure CircleWithSquares where
  /-- Radius of the circle -/
  r : ℝ
  /-- Side length of the larger square (inscribed in the circle) -/
  large_side : ℝ
  /-- Side length of the smaller square -/
  small_side : ℝ
  /-- The larger square is inscribed in the circle -/
  large_inscribed : large_side = 2 * r
  /-- The smaller square shares one side with the larger square -/
  shared_side : small_side ≤ large_side
  /-- Two vertices of the smaller square are on the circle -/
  vertices_on_circle : small_side^2 + (large_side/2 + small_side/2)^2 = r^2

/-- The area of the smaller square is 0% of the area of the larger square -/
theorem smaller_square_area_percentage (c : CircleWithSquares) :
  (c.small_side^2) / (c.large_side^2) = 0 := by
  sorry

end smaller_square_area_percentage_l1806_180693


namespace empty_set_cardinality_zero_l1806_180657

theorem empty_set_cardinality_zero : Finset.card (∅ : Finset α) = 0 := by sorry

end empty_set_cardinality_zero_l1806_180657


namespace smallest_sum_of_four_primes_l1806_180606

def is_prime (n : ℕ) : Prop := sorry

def digits (n : ℕ) : List ℕ := sorry

theorem smallest_sum_of_four_primes : 
  ∃ (a b c d : ℕ),
    is_prime a ∧ is_prime b ∧ is_prime c ∧ is_prime d ∧
    (10 < d) ∧ (d < 100) ∧
    (a < 10) ∧ (b < 10) ∧ (c < 10) ∧
    (digits a ++ digits b ++ digits c ++ digits d).Nodup ∧
    (digits a ++ digits b ++ digits c ++ digits d).length = 9 ∧
    (∀ i, i ∈ digits a ++ digits b ++ digits c ++ digits d → 1 ≤ i ∧ i ≤ 9) ∧
    a + b + c + d = 53 ∧
    (∀ w x y z : ℕ, 
      is_prime w ∧ is_prime x ∧ is_prime y ∧ is_prime z ∧
      (10 < z) ∧ (z < 100) ∧
      (w < 10) ∧ (x < 10) ∧ (y < 10) ∧
      (digits w ++ digits x ++ digits y ++ digits z).Nodup ∧
      (digits w ++ digits x ++ digits y ++ digits z).length = 9 ∧
      (∀ i, i ∈ digits w ++ digits x ++ digits y ++ digits z → 1 ≤ i ∧ i ≤ 9) →
      w + x + y + z ≥ 53) :=
by sorry

end smallest_sum_of_four_primes_l1806_180606


namespace hyperbola_eccentricity_l1806_180620

-- Define the hyperbola
structure Hyperbola where
  asymptote_slope : ℝ

-- Define eccentricity
def eccentricity (h : Hyperbola) : Set ℝ :=
  {e : ℝ | e = 2 ∨ e = (2 * Real.sqrt 3) / 3}

-- Theorem statement
theorem hyperbola_eccentricity (h : Hyperbola) 
  (h_asymptote : h.asymptote_slope = Real.sqrt 3) : 
  ∃ e : ℝ, e ∈ eccentricity h := by
  sorry

end hyperbola_eccentricity_l1806_180620


namespace min_balls_for_twenty_of_one_color_l1806_180627

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  black : Nat

/-- The minimum number of balls needed to guarantee at least n balls of a single color -/
def minBallsForColor (counts : BallCounts) (n : Nat) : Nat :=
  sorry

/-- The specific ball counts in the problem -/
def problemCounts : BallCounts :=
  { red := 30, green := 25, yellow := 25, blue := 18, white := 15, black := 12 }

/-- The theorem stating the minimum number of balls to draw -/
theorem min_balls_for_twenty_of_one_color :
    minBallsForColor problemCounts 20 = 103 := by
  sorry

end min_balls_for_twenty_of_one_color_l1806_180627


namespace ducks_in_lake_l1806_180625

/-- The number of ducks swimming in a lake after multiple groups join -/
def total_ducks (initial : ℕ) (first_group : ℕ) (additional : ℕ) : ℕ :=
  initial + first_group + additional

/-- Theorem stating the total number of ducks in the lake -/
theorem ducks_in_lake : 
  ∀ x : ℕ, total_ducks 13 20 x = 33 + x :=
by
  sorry

end ducks_in_lake_l1806_180625


namespace square_25_solutions_l1806_180689

theorem square_25_solutions (x : ℝ) (h : x^2 = 25) :
  (∃ y : ℝ, y^2 = 25 ∧ y ≠ x) ∧
  (∀ z : ℝ, z^2 = 25 → z = x ∨ z = -x) ∧
  x + (-x) = 0 ∧
  x * (-x) = -25 := by
sorry

end square_25_solutions_l1806_180689


namespace negation_of_universal_proposition_l1806_180618

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 + 1 ≥ 1)) ↔ (∃ x₀ : ℝ, x₀^2 + 1 < 1) :=
by sorry

end negation_of_universal_proposition_l1806_180618


namespace recurrence_solution_l1806_180629

def recurrence_relation (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → a n - 3 * a (n - 1) - 10 * a (n - 2) = 28 * (5 ^ n)

def initial_conditions (a : ℕ → ℝ) : Prop :=
  a 0 = 25 ∧ a 1 = 120

def general_term (n : ℕ) : ℝ :=
  (20 * n + 10) * (5 ^ n) + 15 * ((-2) ^ n)

theorem recurrence_solution (a : ℕ → ℝ) :
  recurrence_relation a ∧ initial_conditions a →
  ∀ n : ℕ, a n = general_term n := by
  sorry

end recurrence_solution_l1806_180629


namespace complex_exponential_identity_l1806_180637

theorem complex_exponential_identity :
  (Complex.exp (Complex.I * Real.pi * (125 / 180)))^40 = Complex.exp (Complex.I * Real.pi * (40 / 180) * (-1)) :=
by sorry

end complex_exponential_identity_l1806_180637


namespace contest_age_fraction_l1806_180612

theorem contest_age_fraction (total_participants : ℕ) (F : ℚ) : 
  total_participants = 500 →
  (F + F / 8 : ℚ) = 0.5625 →
  F = 0.5 :=
by sorry

end contest_age_fraction_l1806_180612


namespace calculation_proof_l1806_180600

theorem calculation_proof : (Real.sqrt 2 - Real.sqrt 3) * (Real.sqrt 2 + Real.sqrt 3) + (2 * Real.sqrt 2 - 1)^2 = 8 - 4 * Real.sqrt 2 := by
  sorry

end calculation_proof_l1806_180600


namespace common_tangents_count_l1806_180615

-- Define the circles
def circle_C1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x + 4*y + 4 = 0
def circle_C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y - 4 = 0

-- Define the function to count common tangents
def count_common_tangents (C1 C2 : (ℝ → ℝ → Prop)) : ℕ := sorry

-- Theorem statement
theorem common_tangents_count :
  count_common_tangents circle_C1 circle_C2 = 3 := by sorry

end common_tangents_count_l1806_180615


namespace lemonade_water_calculation_l1806_180672

/-- Represents the ratio of ingredients in the lemonade recipe -/
structure LemonadeRatio where
  water : ℚ
  lemon_juice : ℚ
  sugar : ℚ

/-- Calculates the amount of water needed for a given lemonade recipe and total volume -/
def water_needed (ratio : LemonadeRatio) (total_volume : ℚ) (quarts_per_gallon : ℚ) : ℚ :=
  let total_parts := ratio.water + ratio.lemon_juice + ratio.sugar
  let water_fraction := ratio.water / total_parts
  water_fraction * total_volume * quarts_per_gallon

/-- Proves that the amount of water needed for the given recipe and volume is 4 quarts -/
theorem lemonade_water_calculation (ratio : LemonadeRatio) 
  (h1 : ratio.water = 6)
  (h2 : ratio.lemon_juice = 2)
  (h3 : ratio.sugar = 1)
  (h4 : quarts_per_gallon = 4) :
  water_needed ratio (3/2) quarts_per_gallon = 4 := by
  sorry

#eval water_needed ⟨6, 2, 1⟩ (3/2) 4

end lemonade_water_calculation_l1806_180672


namespace purely_imaginary_complex_number_l1806_180698

/-- Given a real number m and a complex number z defined as z = (2m² + m - 1) + (-m² - 3m - 2)i,
    if z is purely imaginary, then m = 1/2. -/
theorem purely_imaginary_complex_number (m : ℝ) :
  let z : ℂ := Complex.mk (2 * m^2 + m - 1) (-m^2 - 3*m - 2)
  (z.re = 0 ∧ z.im ≠ 0) → m = 1/2 := by
  sorry

end purely_imaginary_complex_number_l1806_180698


namespace gcd_lcm_sum_6_18_24_l1806_180607

def gcd3 (a b c : ℕ) : ℕ := Nat.gcd a (Nat.gcd b c)

def lcm3 (a b c : ℕ) : ℕ := Nat.lcm a (Nat.lcm b c)

theorem gcd_lcm_sum_6_18_24 : 
  gcd3 6 18 24 + lcm3 6 18 24 = 78 := by
  sorry

end gcd_lcm_sum_6_18_24_l1806_180607


namespace adam_tattoo_count_l1806_180621

/-- The number of tattoos on each of Jason's arms -/
def jason_arm_tattoos : ℕ := 2

/-- The number of tattoos on each of Jason's legs -/
def jason_leg_tattoos : ℕ := 3

/-- The number of arms Jason has -/
def jason_arms : ℕ := 2

/-- The number of legs Jason has -/
def jason_legs : ℕ := 2

/-- The total number of tattoos Jason has -/
def jason_total_tattoos : ℕ := jason_arm_tattoos * jason_arms + jason_leg_tattoos * jason_legs

/-- Adam has three more than twice as many tattoos as Jason -/
def adam_tattoos : ℕ := 2 * jason_total_tattoos + 3

theorem adam_tattoo_count : adam_tattoos = 23 := by sorry

end adam_tattoo_count_l1806_180621


namespace smallest_number_problem_l1806_180611

theorem smallest_number_problem (a b c : ℝ) 
  (h1 : a < b) (h2 : b < c) 
  (h3 : a + b + c = 100)
  (h4 : c = 2 * a)
  (h5 : c - b = 10) : 
  a = 22 := by
sorry

end smallest_number_problem_l1806_180611


namespace sufficient_not_necessary_condition_l1806_180659

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in a 2D plane defined by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to check if a point is on a line -/
def isOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The specific line l: x + y - 1 = 0 -/
def lineL : Line := { a := 1, b := 1, c := -1 }

/-- The specific point condition x = 2 and y = -1 -/
def specificPoint : Point := { x := 2, y := -1 }

theorem sufficient_not_necessary_condition :
  (∀ p : Point, p = specificPoint → isOnLine p lineL) ∧
  ¬(∀ p : Point, isOnLine p lineL → p = specificPoint) := by
  sorry

end sufficient_not_necessary_condition_l1806_180659


namespace largest_four_digit_congruent_to_12_mod_19_l1806_180650

theorem largest_four_digit_congruent_to_12_mod_19 : ∀ n : ℕ,
  n < 10000 → n ≡ 12 [ZMOD 19] → n ≤ 9987 :=
by
  sorry

end largest_four_digit_congruent_to_12_mod_19_l1806_180650


namespace event3_mutually_exclusive_l1806_180609

-- Define the set of numbers
def NumberSet : Set Nat := {n : Nat | 1 ≤ n ∧ n ≤ 9}

-- Define the property of being even
def IsEven (n : Nat) : Prop := ∃ k : Nat, n = 2 * k

-- Define the property of being odd
def IsOdd (n : Nat) : Prop := ∃ k : Nat, n = 2 * k + 1

-- Define the events
def Event1 (a b : Nat) : Prop :=
  a ∈ NumberSet ∧ b ∈ NumberSet ∧ ((IsEven a ∧ IsOdd b) ∨ (IsOdd a ∧ IsEven b))

def Event2 (a b : Nat) : Prop :=
  a ∈ NumberSet ∧ b ∈ NumberSet ∧ (IsOdd a ∨ IsOdd b) ∧ (IsOdd a ∧ IsOdd b)

def Event3 (a b : Nat) : Prop :=
  a ∈ NumberSet ∧ b ∈ NumberSet ∧ (IsOdd a ∨ IsOdd b) ∧ (IsEven a ∧ IsEven b)

def Event4 (a b : Nat) : Prop :=
  a ∈ NumberSet ∧ b ∈ NumberSet ∧ (IsOdd a ∨ IsOdd b) ∧ (IsEven a ∨ IsEven b)

-- Theorem statement
theorem event3_mutually_exclusive :
  ∀ a b : Nat,
    (Event3 a b → ¬Event1 a b) ∧
    (Event3 a b → ¬Event2 a b) ∧
    (Event3 a b → ¬Event4 a b) :=
sorry

end event3_mutually_exclusive_l1806_180609


namespace jacket_cost_calculation_l1806_180653

/-- The amount Joan spent on shorts -/
def shorts_cost : ℚ := 15

/-- The amount Joan spent on a shirt -/
def shirt_cost : ℚ := 12.51

/-- The total amount Joan spent on clothing -/
def total_cost : ℚ := 42.33

/-- The amount Joan spent on the jacket -/
def jacket_cost : ℚ := total_cost - shorts_cost - shirt_cost

theorem jacket_cost_calculation : jacket_cost = 14.82 := by
  sorry

end jacket_cost_calculation_l1806_180653


namespace sufficient_conditions_for_quadratic_inequality_l1806_180682

theorem sufficient_conditions_for_quadratic_inequality :
  (∀ x, x < 4 → x^2 - 2*x - 8 < 0) ∨
  (∀ x, 0 < x ∧ x < 4 → x^2 - 2*x - 8 < 0) ∨
  (∀ x, -2 < x ∧ x < 4 → x^2 - 2*x - 8 < 0) ∨
  (∀ x, -2 < x ∧ x < 3 → x^2 - 2*x - 8 < 0) :=
by sorry

end sufficient_conditions_for_quadratic_inequality_l1806_180682


namespace min_value_problem_l1806_180601

theorem min_value_problem (x : ℝ) (h : x ≥ 3/2) :
  (∀ y, y ≥ 3/2 → (2*x^2 - 2*x + 1)/(x - 1) ≤ (2*y^2 - 2*y + 1)/(y - 1)) →
  (2*x^2 - 2*x + 1)/(x - 1) = 2*Real.sqrt 2 + 2 :=
sorry

end min_value_problem_l1806_180601


namespace class_size_l1806_180675

theorem class_size :
  let both := 5  -- number of people who like both baseball and football
  let baseball_only := 2  -- number of people who only like baseball
  let football_only := 3  -- number of people who only like football
  let neither := 6  -- number of people who like neither baseball nor football
  both + baseball_only + football_only + neither = 16 := by
  sorry

end class_size_l1806_180675


namespace percentage_calculation_l1806_180664

theorem percentage_calculation (x : ℝ) (h : 0.3 * 0.4 * x = 36) : 
  0.5 * (0.4 * 0.3 * x) = 18 := by
  sorry

end percentage_calculation_l1806_180664


namespace math_club_team_selection_l1806_180667

theorem math_club_team_selection (boys girls team_size : ℕ) 
  (h_boys : boys = 10) 
  (h_girls : girls = 12) 
  (h_team_size : team_size = 8) : 
  (Nat.choose (boys + girls) team_size) - 
  (Nat.choose girls team_size) - 
  (Nat.choose boys team_size) = 319230 := by
  sorry

end math_club_team_selection_l1806_180667


namespace smallest_multiple_of_45_and_75_not_20_l1806_180613

theorem smallest_multiple_of_45_and_75_not_20 : 
  ∃ n : ℕ, n > 0 ∧ 45 ∣ n ∧ 75 ∣ n ∧ ¬(20 ∣ n) ∧ 
  ∀ m : ℕ, m > 0 → 45 ∣ m → 75 ∣ m → ¬(20 ∣ m) → n ≤ m :=
by
  use 225
  sorry

end smallest_multiple_of_45_and_75_not_20_l1806_180613


namespace point_on_number_line_l1806_180617

theorem point_on_number_line (A : ℝ) : (|A| = 3) ↔ (A = 3 ∨ A = -3) := by
  sorry

end point_on_number_line_l1806_180617


namespace four_spheres_existence_l1806_180665

-- Define a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a sphere in 3D space
structure Sphere where
  center : Point3D
  radius : ℝ

-- Define a ray starting from a point
structure Ray where
  origin : Point3D
  direction : Point3D

-- Function to check if a point is inside a sphere
def isInside (p : Point3D) (s : Sphere) : Prop := sorry

-- Function to check if two spheres intersect
def intersect (s1 s2 : Sphere) : Prop := sorry

-- Function to check if a ray intersects a sphere
def rayIntersectsSphere (r : Ray) (s : Sphere) : Prop := sorry

-- Theorem statement
theorem four_spheres_existence (A : Point3D) : 
  ∃ (s1 s2 s3 s4 : Sphere),
    (¬ isInside A s1) ∧ (¬ isInside A s2) ∧ (¬ isInside A s3) ∧ (¬ isInside A s4) ∧
    (¬ intersect s1 s2) ∧ (¬ intersect s1 s3) ∧ (¬ intersect s1 s4) ∧
    (¬ intersect s2 s3) ∧ (¬ intersect s2 s4) ∧ (¬ intersect s3 s4) ∧
    (∀ (r : Ray), r.origin = A → 
      rayIntersectsSphere r s1 ∨ rayIntersectsSphere r s2 ∨ 
      rayIntersectsSphere r s3 ∨ rayIntersectsSphere r s4) :=
by
  sorry

end four_spheres_existence_l1806_180665


namespace max_travel_distance_proof_l1806_180635

/-- The distance (in km) a tire can travel on the front wheel before wearing out -/
def front_tire_lifespan : ℝ := 25000

/-- The distance (in km) a tire can travel on the rear wheel before wearing out -/
def rear_tire_lifespan : ℝ := 15000

/-- The maximum distance (in km) a motorcycle can travel before its tires are completely worn out,
    given that the tires are exchanged between front and rear wheels at the optimal time -/
def max_travel_distance : ℝ := 18750

/-- Theorem stating that the calculated maximum travel distance is correct -/
theorem max_travel_distance_proof :
  max_travel_distance = (front_tire_lifespan * rear_tire_lifespan) / (front_tire_lifespan / 2 + rear_tire_lifespan / 2) :=
by sorry

end max_travel_distance_proof_l1806_180635


namespace average_annual_growth_rate_l1806_180696

/-- Proves that the average annual growth rate is 20% given the initial and final revenues --/
theorem average_annual_growth_rate 
  (initial_revenue : ℝ) 
  (final_revenue : ℝ) 
  (years : ℕ) 
  (h1 : initial_revenue = 280)
  (h2 : final_revenue = 403.2)
  (h3 : years = 2) :
  ∃ (growth_rate : ℝ), 
    growth_rate = 0.2 ∧ 
    final_revenue = initial_revenue * (1 + growth_rate) ^ years :=
by sorry


end average_annual_growth_rate_l1806_180696
