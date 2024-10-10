import Mathlib

namespace geometric_sequence_sum_l3073_307382

/-- The sum of a geometric sequence with 6 terms, initial term 10, and common ratio 3 is 3640 -/
theorem geometric_sequence_sum : 
  let a : ℕ := 10  -- initial term
  let r : ℕ := 3   -- common ratio
  let n : ℕ := 6   -- number of terms
  a * (r^n - 1) / (r - 1) = 3640 := by
sorry

end geometric_sequence_sum_l3073_307382


namespace degree_of_minus_5xy_squared_l3073_307318

/-- The type of monomials with integer coefficients in two variables -/
structure Monomial :=
  (coeff : ℤ)
  (x_exp : ℕ)
  (y_exp : ℕ)

/-- The degree of a monomial is the sum of its exponents -/
def degree (m : Monomial) : ℕ := m.x_exp + m.y_exp

/-- The monomial -5xy^2 -/
def m : Monomial := ⟨-5, 1, 2⟩

theorem degree_of_minus_5xy_squared :
  degree m = 3 := by sorry

end degree_of_minus_5xy_squared_l3073_307318


namespace highest_score_calculation_l3073_307345

theorem highest_score_calculation (scores : Finset ℕ) (lowest highest : ℕ) :
  Finset.card scores = 15 →
  (Finset.sum scores id) / 15 = 90 →
  ((Finset.sum scores id) - lowest - highest) / 13 = 92 →
  lowest = 65 →
  highest = 89 := by
  sorry

end highest_score_calculation_l3073_307345


namespace position_function_correct_l3073_307357

/-- The velocity function --/
def v (t : ℝ) : ℝ := 3 * t^2 - 1

/-- The position function --/
def s (t : ℝ) : ℝ := t^3 - t + 0.05

/-- Theorem stating that s is the correct position function --/
theorem position_function_correct :
  (∀ t, (deriv s) t = v t) ∧ s 0 = 0.05 := by
  sorry

end position_function_correct_l3073_307357


namespace kindergarten_boys_count_l3073_307323

/-- Given a kindergarten with a 2:3 ratio of boys to girls and 18 girls, prove there are 12 boys -/
theorem kindergarten_boys_count (total_girls : ℕ) (boys_to_girls_ratio : ℚ) : 
  total_girls = 18 → boys_to_girls_ratio = 2/3 → 
  (total_girls : ℚ) * boys_to_girls_ratio = 12 := by
sorry

end kindergarten_boys_count_l3073_307323


namespace complex_quadratic_roots_l3073_307390

theorem complex_quadratic_roots (z : ℂ) : 
  z^2 = -63 + 16*I ∧ (7 + 4*I)^2 = -63 + 16*I → 
  z = 7 + 4*I ∨ z = -7 - 4*I :=
by sorry

end complex_quadratic_roots_l3073_307390


namespace chocolate_bar_distribution_l3073_307325

theorem chocolate_bar_distribution (total_bars : ℕ) (total_boxes : ℕ) (bars_per_box : ℕ) :
  total_bars = 640 →
  total_boxes = 20 →
  total_bars = total_boxes * bars_per_box →
  bars_per_box = 32 := by
  sorry

end chocolate_bar_distribution_l3073_307325


namespace divisors_of_600_l3073_307324

theorem divisors_of_600 : Nat.card {d : ℕ | d ∣ 600} = 24 := by
  sorry

end divisors_of_600_l3073_307324


namespace tank_capacity_l3073_307375

theorem tank_capacity : 
  ∀ (capacity : ℝ),
  (capacity / 4 + 150 = 2 * capacity / 3) →
  capacity = 360 := by
sorry

end tank_capacity_l3073_307375


namespace juniors_in_sports_l3073_307380

def total_students : ℕ := 500
def junior_percentage : ℚ := 40 / 100
def sports_percentage : ℚ := 70 / 100

theorem juniors_in_sports :
  (total_students : ℚ) * junior_percentage * sports_percentage = 140 := by
  sorry

end juniors_in_sports_l3073_307380


namespace max_min_product_l3073_307384

theorem max_min_product (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c)
  (h4 : a + b + c = 8) (h5 : a * b + b * c + c * a = 16) :
  ∃ m : ℝ, m = min (a * b) (min (b * c) (c * a)) ∧ m ≤ 16 / 9 ∧
  ∃ a' b' c' : ℝ, 0 ≤ a' ∧ 0 ≤ b' ∧ 0 ≤ c' ∧
  a' + b' + c' = 8 ∧ a' * b' + b' * c' + c' * a' = 16 ∧
  min (a' * b') (min (b' * c') (c' * a')) = 16 / 9 := by
  sorry

end max_min_product_l3073_307384


namespace last_stage_less_than_2014_l3073_307368

theorem last_stage_less_than_2014 :
  ∀ k : ℕ, k > 0 → (2 * k^2 - 2 * k + 1 < 2014) ↔ k ≤ 32 :=
by sorry

end last_stage_less_than_2014_l3073_307368


namespace intersection_of_A_and_B_l3073_307355

-- Define the sets A and B
def A : Set ℝ := {x | x - 2 < 3}
def B : Set ℝ := {x | 2*x - 3 < 3*x - 2}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -1 < x ∧ x < 5} := by
  sorry

end intersection_of_A_and_B_l3073_307355


namespace bryan_collected_from_four_continents_l3073_307343

/-- The number of books Bryan collected per continent -/
def books_per_continent : ℕ := 122

/-- The total number of books Bryan collected -/
def total_books : ℕ := 488

/-- The number of continents Bryan collected books from -/
def num_continents : ℕ := total_books / books_per_continent

theorem bryan_collected_from_four_continents :
  num_continents = 4 := by sorry

end bryan_collected_from_four_continents_l3073_307343


namespace hyperbola_solutions_l3073_307361

-- Define the hyperbola equation
def hyperbola (x y : ℤ) : Prop := x^2 - y^2 = 2500^2

-- Define a function to count the number of integer solutions
def count_solutions : ℕ := sorry

-- Theorem stating that the number of solutions is 70
theorem hyperbola_solutions : count_solutions = 70 := by sorry

end hyperbola_solutions_l3073_307361


namespace square_sum_fifteen_l3073_307387

theorem square_sum_fifteen (x y : ℝ) 
  (h1 : y + 4 = (x - 2)^2) 
  (h2 : x + 4 = (y - 2)^2) 
  (h3 : x ≠ y) : 
  x^2 + y^2 = 15 := by
sorry

end square_sum_fifteen_l3073_307387


namespace bread_pieces_in_pond_l3073_307311

theorem bread_pieces_in_pond :
  ∀ (total : ℕ),
    (∃ (duck1 duck2 duck3 : ℕ),
      duck1 = total / 2 ∧
      duck2 = 13 ∧
      duck3 = 7 ∧
      duck1 + duck2 + duck3 + 30 = total) →
    total = 100 := by
sorry

end bread_pieces_in_pond_l3073_307311


namespace salt_solution_mixture_l3073_307314

theorem salt_solution_mixture (x : ℝ) : 
  (0.6 * x = 0.1 * (x + 1)) → x = 0.2 := by
  sorry

end salt_solution_mixture_l3073_307314


namespace last_digit_of_expression_l3073_307394

-- Define a function to get the last digit of a natural number
def lastDigit (n : ℕ) : ℕ := n % 10

-- Define the main theorem
theorem last_digit_of_expression : lastDigit (33 * 3 - (1984^1984 - 1)) = 5 := by
  sorry

end last_digit_of_expression_l3073_307394


namespace sqrt_x_minus_2_meaningful_l3073_307319

theorem sqrt_x_minus_2_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 2) ↔ x ≥ 2 := by
  sorry

end sqrt_x_minus_2_meaningful_l3073_307319


namespace softball_players_count_l3073_307359

theorem softball_players_count (cricket hockey football total : ℕ) 
  (h1 : cricket = 16)
  (h2 : hockey = 12)
  (h3 : football = 18)
  (h4 : total = 59) :
  total - (cricket + hockey + football) = 13 := by
  sorry

end softball_players_count_l3073_307359


namespace danny_collection_difference_l3073_307348

/-- The number of bottle caps Danny found at the park -/
def bottle_caps_found : ℕ := 11

/-- The number of wrappers Danny found at the park -/
def wrappers_found : ℕ := 28

/-- The difference between wrappers and bottle caps found at the park -/
def difference : ℕ := wrappers_found - bottle_caps_found

theorem danny_collection_difference : difference = 17 := by
  sorry

end danny_collection_difference_l3073_307348


namespace cubic_equation_solutions_l3073_307313

theorem cubic_equation_solutions (x y z n : ℕ+) :
  x^3 + y^3 + z^3 = n * x^2 * y^2 * z^2 ↔ n = 1 ∨ n = 3 := by
  sorry

end cubic_equation_solutions_l3073_307313


namespace smallest_four_digit_divisible_proof_l3073_307365

/-- The smallest four-digit number divisible by 2, 3, 8, and 9 -/
def smallest_four_digit_divisible : ℕ := 1008

/-- Predicate to check if a number is four digits -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem smallest_four_digit_divisible_proof :
  is_four_digit smallest_four_digit_divisible ∧
  smallest_four_digit_divisible % 2 = 0 ∧
  smallest_four_digit_divisible % 3 = 0 ∧
  smallest_four_digit_divisible % 8 = 0 ∧
  smallest_four_digit_divisible % 9 = 0 ∧
  ∀ n : ℕ, is_four_digit n →
    n % 2 = 0 → n % 3 = 0 → n % 8 = 0 → n % 9 = 0 →
    n ≥ smallest_four_digit_divisible :=
by sorry

#eval smallest_four_digit_divisible

end smallest_four_digit_divisible_proof_l3073_307365


namespace exponential_function_sum_of_extrema_l3073_307367

theorem exponential_function_sum_of_extrema (a : ℝ) : 
  a > 0 → 
  a ≠ 1 → 
  (max (a^1) (a^2) + min (a^1) (a^2) = 6) → 
  a = 2 := by sorry

end exponential_function_sum_of_extrema_l3073_307367


namespace arithmetic_sequence_property_l3073_307333

/-- An arithmetic sequence {a_n} -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) (h : arithmetic_sequence a) :
  a 6 + a 8 = 16 → a 4 = 1 → a 10 = 15 := by sorry

end arithmetic_sequence_property_l3073_307333


namespace fraction_equality_l3073_307377

theorem fraction_equality (p q s u : ℚ) 
  (h1 : p / q = 3 / 5) 
  (h2 : s / u = 8 / 11) : 
  (4 * p * s - 3 * q * u) / (5 * q * u - 8 * p * s) = -69 / 83 := by
  sorry

end fraction_equality_l3073_307377


namespace carrie_first_day_miles_l3073_307389

/-- Represents the four-day trip driven by Carrie -/
structure CarrieTrip where
  day1 : ℕ
  day2 : ℕ
  day3 : ℕ
  day4 : ℕ
  chargeDistance : ℕ
  chargeCount : ℕ

/-- The conditions of Carrie's trip -/
def tripConditions (trip : CarrieTrip) : Prop :=
  trip.day2 = trip.day1 + 124 ∧
  trip.day3 = 159 ∧
  trip.day4 = 189 ∧
  trip.chargeDistance = 106 ∧
  trip.chargeCount = 7 ∧
  trip.day1 + trip.day2 + trip.day3 + trip.day4 = trip.chargeDistance * trip.chargeCount

/-- Theorem stating that Carrie drove 135 miles on the first day -/
theorem carrie_first_day_miles :
  ∀ (trip : CarrieTrip), tripConditions trip → trip.day1 = 135 :=
by sorry

end carrie_first_day_miles_l3073_307389


namespace arrangement_count_is_factorial_squared_l3073_307306

/-- The number of ways to arrange 5 different objects in a 5x5 grid,
    such that each row and each column contains exactly one object. -/
def arrangement_count : ℕ := (5 : ℕ).factorial ^ 2

/-- Theorem stating that the number of arrangements is equal to (5!)^2 -/
theorem arrangement_count_is_factorial_squared :
  arrangement_count = 14400 := by sorry

end arrangement_count_is_factorial_squared_l3073_307306


namespace tank_fill_problem_l3073_307310

theorem tank_fill_problem (tank_capacity : ℚ) (added_amount : ℚ) (final_fraction : ℚ) :
  tank_capacity = 48 →
  added_amount = 8 →
  final_fraction = 9/10 →
  (tank_capacity * final_fraction - added_amount) / tank_capacity = 4/10 :=
by sorry

end tank_fill_problem_l3073_307310


namespace arithmetic_sequence_solution_l3073_307302

theorem arithmetic_sequence_solution (x : ℝ) (h1 : x ≠ 0) (h2 : x < 4) : 
  (⌊x⌋ + 1 - x + ⌊x⌋ = x + 1 - (⌊x⌋ + 1)) → 
  (x = 0.5 ∨ x = 1.5 ∨ x = 2.5 ∨ x = 3.5) := by
sorry

end arithmetic_sequence_solution_l3073_307302


namespace inscribed_cube_volume_l3073_307392

theorem inscribed_cube_volume (large_cube_edge : ℝ) (small_cube_edge : ℝ) 
  (h1 : large_cube_edge = 12)
  (h2 : small_cube_edge * Real.sqrt 3 = large_cube_edge) : 
  small_cube_edge ^ 3 = 192 * Real.sqrt 3 := by
  sorry

#check inscribed_cube_volume

end inscribed_cube_volume_l3073_307392


namespace marble_collection_weight_l3073_307328

/-- The weight of Courtney's marble collection -/
def total_weight (jar1_count : ℕ) (jar1_weight : ℚ) (jar2_weight : ℚ) (jar3_weight : ℚ) (jar4_weight : ℚ) : ℚ :=
  let jar2_count := 2 * jar1_count
  let jar3_count := (1 : ℚ) / 4 * jar1_count
  let jar4_count := (3 : ℚ) / 5 * jar2_count
  jar1_count * jar1_weight + jar2_count * jar2_weight + jar3_count * jar3_weight + jar4_count * jar4_weight

/-- Theorem stating the total weight of Courtney's marble collection -/
theorem marble_collection_weight :
  total_weight 80 (35 / 100) (45 / 100) (25 / 100) (55 / 100) = 1578 / 10 := by
  sorry

end marble_collection_weight_l3073_307328


namespace inscribable_iff_equal_sums_l3073_307350

/-- A convex polyhedral angle -/
structure ConvexPolyhedralAngle where
  -- Add necessary fields

/-- The property of being inscribable in a cone -/
def isInscribableInCone (angle : ConvexPolyhedralAngle) : Prop :=
  sorry

/-- The property of having equal sums of opposite dihedral angles -/
def hasEqualSumsOfOppositeDihedralAngles (angle : ConvexPolyhedralAngle) : Prop :=
  sorry

/-- Theorem: A convex polyhedral angle can be inscribed in a cone if and only if 
    the sums of its opposite dihedral angles are equal -/
theorem inscribable_iff_equal_sums 
  (angle : ConvexPolyhedralAngle) : 
  isInscribableInCone angle ↔ hasEqualSumsOfOppositeDihedralAngles angle :=
sorry

end inscribable_iff_equal_sums_l3073_307350


namespace consecutive_integers_product_l3073_307329

theorem consecutive_integers_product (a b c d e : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
  b = a + 1 ∧ c = a + 2 ∧ d = a + 3 ∧ e = a + 4 ∧
  a * b * c * d * e = 15120 →
  e = 9 := by sorry

end consecutive_integers_product_l3073_307329


namespace f_leq_g_l3073_307396

/-- Given functions f and g, prove that f(x) ≤ g(x) for all x > 0 when a ≥ 1 -/
theorem f_leq_g (x a : ℝ) (hx : x > 0) (ha : a ≥ 1) :
  Real.log x + 2 * x ≤ a * (x^2 + x) := by
  sorry

end f_leq_g_l3073_307396


namespace card_numbers_proof_l3073_307376

def is_valid_sequence (seq : List ℕ) : Prop :=
  seq.length = 9 ∧
  (∀ n, n ∈ seq → 1 ≤ n ∧ n ≤ 9) ∧
  (∀ i, i < seq.length - 2 → 
    ¬(seq[i]! < seq[i+1]! ∧ seq[i+1]! < seq[i+2]!) ∧
    ¬(seq[i]! > seq[i+1]! ∧ seq[i+1]! > seq[i+2]!))

def visible_sequence : List ℕ := [1, 3, 4, 6, 7, 8]

theorem card_numbers_proof :
  ∀ (seq : List ℕ),
  is_valid_sequence seq →
  seq.take 1 = [1] →
  seq.drop 1 = 3 :: visible_sequence.drop 2 →
  seq[1]! = 5 ∧ seq[4]! = 2 ∧ seq[5]! = 9 :=
sorry

end card_numbers_proof_l3073_307376


namespace product_of_four_primes_l3073_307373

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- Theorem stating the properties of the product of four specific primes -/
theorem product_of_four_primes (A B : ℕ) 
  (hA : isPrime A) 
  (hB : isPrime B) 
  (hAminusB : isPrime (A - B)) 
  (hAplusB : isPrime (A + B)) : 
  ∃ (p : ℕ), p = A * B * (A - B) * (A + B) ∧ Even p ∧ p % 3 = 0 := by
  sorry


end product_of_four_primes_l3073_307373


namespace least_integer_sum_l3073_307362

theorem least_integer_sum (x y z : ℕ+) (h : (2 : ℕ) * x.val = (5 : ℕ) * y.val ∧ (5 : ℕ) * y.val = (6 : ℕ) * z.val) :
  (∃ (n : ℤ), x.val + n + z.val = 26 ∧ ∀ (m : ℤ), x.val + m + z.val = 26 → n ≤ m) →
  (∃ (n : ℤ), x.val + n + z.val = 26 ∧ n = 6) :=
by sorry

end least_integer_sum_l3073_307362


namespace xy_zero_necessary_not_sufficient_l3073_307339

theorem xy_zero_necessary_not_sufficient (x y : ℝ) :
  (∀ x, x = 0 → x * y = 0) ∧ 
  ¬(∀ x y, x * y = 0 → x = 0) :=
sorry

end xy_zero_necessary_not_sufficient_l3073_307339


namespace tammy_orange_trees_l3073_307301

/-- The number of oranges Tammy can pick from each tree per day -/
def oranges_per_tree_per_day : ℕ := 12

/-- The price of a 6-pack of oranges in dollars -/
def price_per_6pack : ℕ := 2

/-- The total earnings in dollars after 3 weeks -/
def total_earnings : ℕ := 840

/-- The number of days in 3 weeks -/
def days_in_3_weeks : ℕ := 21

/-- The number of orange trees Tammy has -/
def number_of_trees : ℕ := 10

theorem tammy_orange_trees :
  number_of_trees * oranges_per_tree_per_day * days_in_3_weeks =
  (total_earnings / price_per_6pack) * 6 :=
sorry

end tammy_orange_trees_l3073_307301


namespace arithmetic_sequence_properties_l3073_307395

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  s : ℕ → ℤ  -- The sum of the first n terms
  first_term : a 1 = 31
  sum_equality : s 10 = s 22

/-- The sum formula for the arithmetic sequence -/
def sum_formula (n : ℕ) : ℤ := 32 * n - n^2

/-- Theorem stating the properties of the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n, seq.s n = sum_formula n) ∧
  (∃ n, ∀ m, seq.s m ≤ seq.s n) ∧
  (seq.s 16 = 256) := by
  sorry


end arithmetic_sequence_properties_l3073_307395


namespace least_crayons_l3073_307331

def is_divisible_by (n m : ℕ) : Prop := m ∣ n

theorem least_crayons (n : ℕ) : 
  (is_divisible_by n 3 ∧ 
   is_divisible_by n 4 ∧ 
   is_divisible_by n 5 ∧ 
   is_divisible_by n 7 ∧ 
   is_divisible_by n 8) →
  (∀ m : ℕ, m < n → 
    ¬(is_divisible_by m 3 ∧ 
      is_divisible_by m 4 ∧ 
      is_divisible_by m 5 ∧ 
      is_divisible_by m 7 ∧ 
      is_divisible_by m 8)) →
  n = 840 := by
sorry

end least_crayons_l3073_307331


namespace rectangular_solid_diagonal_l3073_307344

theorem rectangular_solid_diagonal (a b c : ℝ) 
  (h1 : 2 * (a * b + b * c + a * c) = 30)
  (h2 : 4 * (a + b + c) = 28) :
  (a^2 + b^2 + c^2).sqrt = (19 : ℝ).sqrt := by
  sorry

end rectangular_solid_diagonal_l3073_307344


namespace problem_solution_l3073_307334

def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

theorem problem_solution :
  (∀ x : ℝ, f 2 x ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) ∧
  (∀ x : ℝ, f 2 x + f 2 (x + 5) ≥ 5) :=
sorry

end problem_solution_l3073_307334


namespace parametric_equations_represent_line_l3073_307303

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := 3 * x + 4 * y + 1 = 0

/-- The parametric equations -/
def parametric_x (t : ℝ) : ℝ := 1 - 4 * t
def parametric_y (t : ℝ) : ℝ := -1 + 3 * t

/-- Theorem stating that the parametric equations represent the line -/
theorem parametric_equations_represent_line :
  ∀ t : ℝ, line_equation (parametric_x t) (parametric_y t) :=
by
  sorry

end parametric_equations_represent_line_l3073_307303


namespace least_number_of_cookies_cookies_solution_mohan_cookies_l3073_307308

theorem least_number_of_cookies (x : ℕ) : 
  (x % 6 = 5) ∧ (x % 9 = 3) ∧ (x % 11 = 7) → x ≥ 83 :=
by sorry

theorem cookies_solution : 
  (83 % 6 = 5) ∧ (83 % 9 = 3) ∧ (83 % 11 = 7) :=
by sorry

theorem mohan_cookies : 
  ∃ (x : ℕ), (x % 6 = 5) ∧ (x % 9 = 3) ∧ (x % 11 = 7) ∧ 
  (∀ (y : ℕ), (y % 6 = 5) ∧ (y % 9 = 3) ∧ (y % 11 = 7) → x ≤ y) ∧
  x = 83 :=
by sorry

end least_number_of_cookies_cookies_solution_mohan_cookies_l3073_307308


namespace exists_large_number_with_exchangeable_digits_l3073_307398

/-- A function that checks if two natural numbers have the same set of prime divisors -/
def samePrimeDivisors (a b : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p ∣ a ↔ p ∣ b)

/-- A function that checks if a number can have two distinct non-zero digits exchanged -/
def canExchangeDigits (n : ℕ) : Prop :=
  ∃ (d₁ d₂ : ℕ) (k m : ℕ),
    d₁ ≠ d₂ ∧ d₁ ≠ 0 ∧ d₂ ≠ 0 ∧
    (∃ n₁ n₂ : ℕ,
      n₁ = n + (d₁ - d₂) * 10^k ∧
      n₂ = n + (d₂ - d₁) * 10^m ∧
      samePrimeDivisors n₁ n₂)

/-- The main theorem -/
theorem exists_large_number_with_exchangeable_digits :
  ∃ n : ℕ, n > 10^1000 ∧ ¬(10 ∣ n) ∧ canExchangeDigits n :=
sorry

end exists_large_number_with_exchangeable_digits_l3073_307398


namespace chocolate_box_problem_l3073_307372

/-- Calculates the number of additional boxes needed to store chocolates --/
def additional_boxes_needed (total_chocolates : ℕ) (chocolates_not_in_box : ℕ) (existing_boxes : ℕ) (friend_chocolates : ℕ) : ℕ :=
  let chocolates_in_boxes := total_chocolates - chocolates_not_in_box
  let total_chocolates_to_box := chocolates_in_boxes + friend_chocolates
  let chocolates_per_box := chocolates_in_boxes / existing_boxes
  let total_boxes_needed := (total_chocolates_to_box + chocolates_per_box - 1) / chocolates_per_box
  total_boxes_needed - existing_boxes

theorem chocolate_box_problem :
  additional_boxes_needed 50 5 3 25 = 2 := by
  sorry

end chocolate_box_problem_l3073_307372


namespace problem_solution_l3073_307300

theorem problem_solution (m n c d a : ℝ) 
  (h1 : m + n = 0)  -- m and n are opposite numbers
  (h2 : c * d = 1)  -- c and d are reciprocals
  (h3 : a = ⌊Real.sqrt 5⌋) -- a is the integer part of √5
  : Real.sqrt (c * d) + 2 * (m + n) - a = -1 := by
  sorry

end problem_solution_l3073_307300


namespace certain_number_proof_l3073_307388

theorem certain_number_proof : ∃ (x : ℚ), (2994 / x = 179) → x = 167 / 10 := by
  sorry

end certain_number_proof_l3073_307388


namespace expression_evaluation_l3073_307385

theorem expression_evaluation (x y z : ℝ) 
  (hz : z = y - 11)
  (hy : y = x + 3)
  (hx : x = 5) :
  ((x + 3) / (x + 2)) * ((y - 2) / (y - 3)) * ((z + 9) / (z + 7)) = 72 / 35 ∧
  x + 2 ≠ 0 ∧ y - 3 ≠ 0 ∧ z + 7 ≠ 0 := by
  sorry

end expression_evaluation_l3073_307385


namespace exists_non_increasing_log_l3073_307312

-- Define the logarithmic function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- State the theorem
theorem exists_non_increasing_log :
  ∃ (a : ℝ), a > 0 ∧ a ≠ 1 ∧ ¬(∀ (x y : ℝ), x > y → log a x > log a y) :=
by sorry

end exists_non_increasing_log_l3073_307312


namespace special_triangle_base_l3073_307326

/-- A triangle with specific side length properties -/
structure SpecialTriangle where
  left : ℝ
  right : ℝ
  base : ℝ
  sum_of_sides : left + right + base = 50
  right_longer : right = left + 2
  left_value : left = 12

/-- The base of a SpecialTriangle is 24 -/
theorem special_triangle_base (t : SpecialTriangle) : t.base = 24 := by
  sorry

end special_triangle_base_l3073_307326


namespace distance_between_points_l3073_307346

/-- The distance between equidistant points A, B, and C, given specific travel conditions. -/
theorem distance_between_points (v_car v_train t : ℝ) (h1 : v_car = 80) (h2 : v_train = 50) (h3 : t = 7) :
  let S := v_car * t * (25800 / 210)
  S = 861 := by sorry

end distance_between_points_l3073_307346


namespace work_completion_time_l3073_307304

theorem work_completion_time (D_A : ℝ) 
  (h1 : D_A > 0)
  (h2 : 1 / D_A + 2 / D_A = 1 / 4) : 
  D_A = 12 := by
sorry

end work_completion_time_l3073_307304


namespace simplify_expression_l3073_307342

theorem simplify_expression (a b : ℝ) : (1 : ℝ) * (2 * a) * (3 * a^2 * b) * (4 * a^3 * b^2) * (5 * a^4 * b^3) = 120 * a^10 * b^6 := by
  sorry

end simplify_expression_l3073_307342


namespace labourer_absence_solution_l3073_307336

/-- Represents the problem of calculating a labourer's absence days --/
def LabourerAbsence (total_days work_pay absence_fine total_received : ℚ) : Prop :=
  ∃ (days_worked days_absent : ℚ),
    days_worked + days_absent = total_days ∧
    work_pay * days_worked - absence_fine * days_absent = total_received ∧
    days_absent = 5

/-- Theorem stating the solution to the labourer absence problem --/
theorem labourer_absence_solution :
  LabourerAbsence 25 2 (1/2) (75/2) :=
sorry

end labourer_absence_solution_l3073_307336


namespace ellipse_focus_d_value_l3073_307378

/-- Definition of an ellipse with given properties -/
structure Ellipse where
  /-- The ellipse is in the first quadrant -/
  first_quadrant : Bool
  /-- The ellipse is tangent to both x-axis and y-axis -/
  tangent_to_axes : Bool
  /-- One focus of the ellipse -/
  focus1 : ℝ × ℝ
  /-- The other focus of the ellipse -/
  focus2 : ℝ × ℝ

/-- Theorem stating the value of d for the given ellipse -/
theorem ellipse_focus_d_value (e : Ellipse) : 
  e.first_quadrant = true ∧ 
  e.tangent_to_axes = true ∧ 
  e.focus1 = (4, 10) ∧ 
  e.focus2.1 = e.focus2.2 ∧ 
  e.focus2.2 = 10 → 
  e.focus2.1 = 25 := by
sorry

end ellipse_focus_d_value_l3073_307378


namespace max_value_theorem_l3073_307341

theorem max_value_theorem (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : a^2 + b^2/2 = 1) :
  ∃ (M : ℝ), M = (3 * Real.sqrt 2) / 4 ∧ a * Real.sqrt (1 + b^2) ≤ M ∧
  ∃ (a₀ b₀ : ℝ), a₀ * Real.sqrt (1 + b₀^2) = M :=
sorry

end max_value_theorem_l3073_307341


namespace polynomial_degree_three_l3073_307305

def f (x : ℝ) : ℝ := 2 - 15*x + 4*x^2 - 5*x^3 + 7*x^4
def g (x : ℝ) : ℝ := 4 - 3*x - 8*x^3 + 12*x^4

def c : ℚ := -7/12

theorem polynomial_degree_three :
  ∃ (a b d : ℝ), ∀ (x : ℝ),
    f x + c * g x = a*x^3 + b*x^2 + d*x + (2 + 4*c) ∧ a ≠ 0 := by
  sorry

end polynomial_degree_three_l3073_307305


namespace visiting_students_theorem_l3073_307371

/-- Represents a set of students visiting each other's homes -/
structure VisitingStudents where
  n : ℕ  -- number of students
  d : ℕ  -- number of days
  assignment : Fin n → Finset (Fin d)

/-- A valid assignment means no subset is contained within another subset -/
def ValidAssignment (vs : VisitingStudents) : Prop :=
  ∀ i j : Fin vs.n, i ≠ j → ¬(vs.assignment i ⊆ vs.assignment j)

theorem visiting_students_theorem :
  (¬∃ vs : VisitingStudents, vs.n = 30 ∧ vs.d = 4 ∧ ValidAssignment vs) ∧
  (¬∃ vs : VisitingStudents, vs.n = 30 ∧ vs.d = 5 ∧ ValidAssignment vs) ∧
  (∃ vs : VisitingStudents, vs.n = 30 ∧ vs.d = 7 ∧ ValidAssignment vs) ∧
  (∃ vs : VisitingStudents, vs.n = 30 ∧ vs.d = 10 ∧ ValidAssignment vs) :=
by sorry

end visiting_students_theorem_l3073_307371


namespace smallest_n_congruence_eight_satisfies_congruence_eight_is_smallest_smallest_positive_integer_congruence_l3073_307353

theorem smallest_n_congruence (n : ℕ) : 
  (n > 0 ∧ 19 * n ≡ 5678 [ZMOD 11]) → n ≥ 8 :=
by sorry

theorem eight_satisfies_congruence : 19 * 8 ≡ 5678 [ZMOD 11] :=
by sorry

theorem eight_is_smallest : 
  ∀ m : ℕ, m > 0 ∧ m < 8 → ¬(19 * m ≡ 5678 [ZMOD 11]) :=
by sorry

theorem smallest_positive_integer_congruence : 
  ∃! n : ℕ, n > 0 ∧ 19 * n ≡ 5678 [ZMOD 11] ∧ 
  ∀ m : ℕ, m > 0 ∧ 19 * m ≡ 5678 [ZMOD 11] → n ≤ m :=
by sorry

end smallest_n_congruence_eight_satisfies_congruence_eight_is_smallest_smallest_positive_integer_congruence_l3073_307353


namespace part_one_part_two_l3073_307360

/-- The set A -/
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}

/-- The set B -/
def B (m : ℝ) : Set ℝ := {x | 2 - m ≤ x ∧ x ≤ 2*m - 3}

/-- Part 1: p is sufficient but not necessary for q -/
theorem part_one (m : ℝ) : (∀ x, x ∈ A → x ∈ B m) ∧ (∃ x, x ∈ B m ∧ x ∉ A) → m ≥ 4 := by
  sorry

/-- Part 2: A ∪ B = A -/
theorem part_two (m : ℝ) : A ∪ B m = A → m ≤ 3 := by
  sorry

end part_one_part_two_l3073_307360


namespace brendas_age_l3073_307366

/-- Given the ages of Addison, Brenda, and Janet, prove that Brenda's age is 8/3 years. -/
theorem brendas_age (A B J : ℚ) 
  (h1 : A = 4 * B)  -- Addison's age is four times Brenda's age
  (h2 : J = B + 8)  -- Janet is eight years older than Brenda
  (h3 : A = J)      -- Addison and Janet are twins (same age)
  : B = 8 / 3 := by
  sorry

end brendas_age_l3073_307366


namespace union_cardinality_l3073_307349

def A : Finset ℕ := {1, 3, 5}
def B : Finset ℕ := {2, 3}

theorem union_cardinality : Finset.card (A ∪ B) = 4 := by
  sorry

end union_cardinality_l3073_307349


namespace downstream_speed_l3073_307338

/-- Represents the speed of a man rowing in different conditions -/
structure RowingSpeed where
  upstream : ℝ
  stillWater : ℝ
  downstream : ℝ

/-- Theorem stating the relationship between upstream, stillwater, and downstream speeds -/
theorem downstream_speed (s : RowingSpeed) 
  (h1 : s.upstream = 15) 
  (h2 : s.stillWater = 25) : 
  s.downstream = 35 := by
  sorry

end downstream_speed_l3073_307338


namespace sum_of_solutions_quadratic_l3073_307337

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (∃ r s : ℝ, (36 - 18 * x - x^2 = 0 ↔ x = r ∨ x = s) ∧ r + s = 18) := by
  sorry

end sum_of_solutions_quadratic_l3073_307337


namespace product_of_ab_values_l3073_307320

theorem product_of_ab_values (a b : ℝ) (h1 : a + 1/b = 4) (h2 : 1/a + b = 16/15) : 
  (5/3 * 3/5 : ℝ) = 1 := by sorry

end product_of_ab_values_l3073_307320


namespace larger_share_theorem_l3073_307393

/-- Given two investments and a total profit, calculates the share of profit for the larger investment -/
def calculate_larger_share (investment1 : ℕ) (investment2 : ℕ) (total_profit : ℕ) : ℕ :=
  let larger_investment := max investment1 investment2
  let total_investment := investment1 + investment2
  (larger_investment * total_profit) / total_investment

theorem larger_share_theorem (investment1 investment2 total_profit : ℕ) 
  (h1 : investment1 = 22500) 
  (h2 : investment2 = 35000) 
  (h3 : total_profit = 13800) :
  calculate_larger_share investment1 investment2 total_profit = 8400 := by
  sorry

#eval calculate_larger_share 22500 35000 13800

end larger_share_theorem_l3073_307393


namespace quadrilateral_area_is_correct_l3073_307330

/-- The area of a quadrilateral with vertices at (2, 2), (2, -1), (3, -1), and (2007, 2008) -/
def quadrilateralArea : ℝ := 2008006.5

/-- The vertices of the quadrilateral -/
def vertices : List (ℝ × ℝ) := [(2, 2), (2, -1), (3, -1), (2007, 2008)]

/-- Theorem stating that the area of the quadrilateral with the given vertices is 2008006.5 -/
theorem quadrilateral_area_is_correct : 
  let area := quadrilateralArea
  ∃ (f : List (ℝ × ℝ) → ℝ), f vertices = area :=
by
  sorry


end quadrilateral_area_is_correct_l3073_307330


namespace repeating_decimal_equals_fraction_l3073_307364

-- Define the repeating decimal 6.8181...
def repeating_decimal : ℚ := 6 + 81 / 99

-- Theorem stating that the repeating decimal equals 75/11
theorem repeating_decimal_equals_fraction : repeating_decimal = 75 / 11 := by
  sorry

end repeating_decimal_equals_fraction_l3073_307364


namespace x_percent_of_x_equals_nine_l3073_307340

theorem x_percent_of_x_equals_nine (x : ℝ) : 
  x > 0 → (x / 100) * x = 9 → x = 30 := by
  sorry

end x_percent_of_x_equals_nine_l3073_307340


namespace periodic_function_l3073_307358

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.sin (Real.pi * x + φ)

theorem periodic_function (φ : ℝ) :
  ∃ t : ℝ, t ≠ 0 ∧ ∀ x : ℝ, f (x + t) φ = f x φ :=
by
  use 2
  sorry

end periodic_function_l3073_307358


namespace constant_term_expansion_l3073_307399

theorem constant_term_expansion (k : ℕ+) : k = 1 ↔ k^4 * (Nat.choose 6 4) < 120 := by sorry

end constant_term_expansion_l3073_307399


namespace first_discount_percentage_l3073_307321

def original_price : ℝ := 345
def final_price : ℝ := 227.70
def second_discount : ℝ := 0.25

theorem first_discount_percentage :
  ∃ (d : ℝ), d ≥ 0 ∧ d ≤ 1 ∧
  original_price * (1 - d) * (1 - second_discount) = final_price ∧
  d = 0.12 :=
sorry

end first_discount_percentage_l3073_307321


namespace vanessa_deleted_files_l3073_307386

/-- Calculates the number of deleted files given the initial number of music files,
    initial number of video files, and the number of remaining files. -/
def deleted_files (initial_music : ℕ) (initial_video : ℕ) (remaining : ℕ) : ℕ :=
  initial_music + initial_video - remaining

/-- Theorem stating that Vanessa deleted 30 files from her flash drive. -/
theorem vanessa_deleted_files :
  deleted_files 16 48 34 = 30 := by
  sorry

end vanessa_deleted_files_l3073_307386


namespace square_root_problem_l3073_307352

theorem square_root_problem (m a b c n : ℝ) (hm : m > 0) :
  (Real.sqrt m = 2*n + 1 ∧ Real.sqrt m = 4 - 3*n) →
  (|a - 1| + Real.sqrt b + (c - n)^2 = 0) →
  (m = 121 ∨ m = 121/25) ∧ Real.sqrt (a + b + c) = Real.sqrt 6 ∨ Real.sqrt (a + b + c) = -Real.sqrt 6 := by
  sorry

end square_root_problem_l3073_307352


namespace sin_cos_difference_l3073_307379

theorem sin_cos_difference (θ₁ θ₂ θ₃ θ₄ : Real) 
  (h₁ : θ₁ = 17 * π / 180)
  (h₂ : θ₂ = 47 * π / 180)
  (h₃ : θ₃ = 73 * π / 180)
  (h₄ : θ₄ = 43 * π / 180) : 
  Real.sin θ₁ * Real.cos θ₂ - Real.sin θ₃ * Real.cos θ₄ = -1/2 := by
  sorry

end sin_cos_difference_l3073_307379


namespace inequality_proof_l3073_307354

theorem inequality_proof (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) : a < a * b^2 ∧ a * b^2 < a * b := by
  sorry

end inequality_proof_l3073_307354


namespace elliptical_track_distance_l3073_307383

/-- Represents the properties of an elliptical track with two objects moving on it. -/
structure EllipticalTrack where
  /-- Half the circumference of the track in yards -/
  half_circumference : ℝ
  /-- Distance traveled by object B at first meeting in yards -/
  first_meeting_distance : ℝ
  /-- Distance object A is short of completing a lap at second meeting in yards -/
  second_meeting_shortfall : ℝ

/-- Theorem stating the total distance around the track given specific conditions -/
theorem elliptical_track_distance 
  (track : EllipticalTrack)
  (h1 : track.first_meeting_distance = 150)
  (h2 : track.second_meeting_shortfall = 90)
  (h3 : (track.first_meeting_distance) / (track.half_circumference - track.first_meeting_distance) = 
        (track.half_circumference + track.second_meeting_shortfall) / 
        (2 * track.half_circumference - track.second_meeting_shortfall)) :
  2 * track.half_circumference = 720 := by
  sorry


end elliptical_track_distance_l3073_307383


namespace arithmetic_equation_l3073_307316

theorem arithmetic_equation : 3 * 13 + 3 * 14 + 3 * 17 + 11 = 143 := by
  sorry

end arithmetic_equation_l3073_307316


namespace seven_twelfths_decimal_l3073_307322

theorem seven_twelfths_decimal : 
  (7 : ℚ) / 12 = 0.5833333333333333 := by sorry

end seven_twelfths_decimal_l3073_307322


namespace solve_square_root_equation_l3073_307356

theorem solve_square_root_equation (x : ℝ) : 
  Real.sqrt (5 * x + 9) = 12 → x = 27 := by sorry

end solve_square_root_equation_l3073_307356


namespace derivative_ln_at_e_l3073_307397

open Real

theorem derivative_ln_at_e (f : ℝ → ℝ) (h : ∀ x, f x = log x) : 
  deriv f e = 1 / e := by
  sorry

end derivative_ln_at_e_l3073_307397


namespace unique_solution_condition_inequality_condition_l3073_307391

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - 1
def g (a x : ℝ) : ℝ := a * |x - 1|

-- Theorem for part I
theorem unique_solution_condition (a : ℝ) :
  (∃! x, |f x| = g a x) ↔ a < 0 := by sorry

-- Theorem for part II
theorem inequality_condition (a : ℝ) :
  (∀ x, f x ≥ g a x) ↔ a ≤ -2 := by sorry

end unique_solution_condition_inequality_condition_l3073_307391


namespace largest_decimal_l3073_307315

theorem largest_decimal : ∀ (a b c d e : ℝ), 
  a = 0.989 → b = 0.998 → c = 0.981 → d = 0.899 → e = 0.9801 →
  (b ≥ a ∧ b ≥ c ∧ b ≥ d ∧ b ≥ e) := by
  sorry

end largest_decimal_l3073_307315


namespace janes_current_age_l3073_307381

theorem janes_current_age :
  let min_age : ℕ := 25
  let years_until_dara_eligible : ℕ := 14
  let years_until_half_age : ℕ := 6
  let dara_current_age : ℕ := min_age - years_until_dara_eligible
  ∀ jane_age : ℕ,
    (dara_current_age + years_until_half_age = (jane_age + years_until_half_age) / 2) →
    jane_age = 28 :=
by
  sorry

end janes_current_age_l3073_307381


namespace arithmetic_geometric_sequence_l3073_307317

/-- 
Given an arithmetic sequence {a_n} with common difference -2,
if a_1, a_4, and a_5 form a geometric sequence, then a_3 = 5.
-/
theorem arithmetic_geometric_sequence (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) = a n - 2) →  -- arithmetic sequence with common difference -2
  (a 4)^2 = a 1 * a 5 →         -- a_1, a_4, a_5 form a geometric sequence
  a 3 = 5 := by
sorry

end arithmetic_geometric_sequence_l3073_307317


namespace reciprocal_of_negative_two_l3073_307370

theorem reciprocal_of_negative_two :
  ((-2 : ℝ)⁻¹ = -1/2) := by sorry

end reciprocal_of_negative_two_l3073_307370


namespace prime_sum_of_squares_l3073_307351

theorem prime_sum_of_squares (p : ℕ) (hp : Nat.Prime p) :
  (∃ k : ℕ, p = 4 * k + 1) → (∃ a b : ℤ, p = a^2 + b^2) ∧
  (∃ k : ℕ, p = 8 * k + 3) → (∃ a b c : ℤ, p = a^2 + b^2 + c^2) :=
by sorry

end prime_sum_of_squares_l3073_307351


namespace extreme_point_inequality_l3073_307347

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log (x + 1) + (1/2) * x^2 - x

theorem extreme_point_inequality (a : ℝ) (x₁ x₂ : ℝ) :
  0 < a → a < 1 →
  x₁ < x₂ →
  x₁ = -Real.sqrt (1 - a) →
  x₂ = Real.sqrt (1 - a) →
  f a x₁ < f a x₂ →
  f a x₂ > f a x₁ →
  f a x₂ / x₁ < 1/2 := by
sorry

end extreme_point_inequality_l3073_307347


namespace dans_work_time_l3073_307374

theorem dans_work_time (D : ℝ) : 
  (∀ (annie_rate : ℝ), annie_rate = 1 / 10 →
   ∀ (dan_rate : ℝ), dan_rate = 1 / D →
   6 * dan_rate + 6 * annie_rate = 1) →
  D = 15 := by
sorry

end dans_work_time_l3073_307374


namespace sequence_problem_l3073_307363

theorem sequence_problem : ∃ (x y : ℝ), 
  (x^2 = 2*y) ∧ (2*y = x + 20) ∧ ((x + y = 4) ∨ (x + y = 35/2)) := by
  sorry

end sequence_problem_l3073_307363


namespace timothy_total_cost_l3073_307369

/-- The total cost of Timothy's purchases -/
def total_cost (land_acres : ℕ) (land_price_per_acre : ℕ) 
               (house_price : ℕ) 
               (cow_count : ℕ) (cow_price : ℕ) 
               (chicken_count : ℕ) (chicken_price : ℕ) 
               (solar_install_hours : ℕ) (solar_install_price_per_hour : ℕ) 
               (solar_equipment_price : ℕ) : ℕ :=
  land_acres * land_price_per_acre + 
  house_price + 
  cow_count * cow_price + 
  chicken_count * chicken_price + 
  solar_install_hours * solar_install_price_per_hour + 
  solar_equipment_price

/-- Theorem stating that Timothy's total cost is $147,700 -/
theorem timothy_total_cost : 
  total_cost 30 20 120000 20 1000 100 5 6 100 6000 = 147700 := by
  sorry

end timothy_total_cost_l3073_307369


namespace exists_pair_satisfying_condition_l3073_307307

theorem exists_pair_satisfying_condition (r : Fin 5 → ℝ) : 
  ∃ (i j : Fin 5), i ≠ j ∧ 0 ≤ (r i - r j) / (1 + r i * r j) ∧ (r i - r j) / (1 + r i * r j) ≤ 1 := by
  sorry

end exists_pair_satisfying_condition_l3073_307307


namespace total_crayons_l3073_307332

def new_crayons : ℕ := 2
def used_crayons : ℕ := 4
def broken_crayons : ℕ := 8

theorem total_crayons : new_crayons + used_crayons + broken_crayons = 14 := by
  sorry

end total_crayons_l3073_307332


namespace min_sum_squares_l3073_307335

theorem min_sum_squares (a b c t : ℝ) (h : a + b + c = t) :
  a^2 + b^2 + c^2 ≥ t^2 / 3 := by
  sorry

end min_sum_squares_l3073_307335


namespace reciprocal_problem_l3073_307309

theorem reciprocal_problem (x : ℝ) (h : 8 * x = 3) : 200 / x = 1600 / 3 := by
  sorry

end reciprocal_problem_l3073_307309


namespace car_travel_distance_l3073_307327

-- Define the initial conditions
def initial_distance : ℝ := 180
def initial_time : ℝ := 4
def next_time : ℝ := 3

-- Define the theorem
theorem car_travel_distance :
  let speed := initial_distance / initial_time
  speed * next_time = 135 := by
  sorry

end car_travel_distance_l3073_307327
