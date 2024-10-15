import Mathlib

namespace NUMINAMATH_CALUDE_marbles_left_l1720_172091

theorem marbles_left (initial_marbles given_marbles : ℝ) :
  initial_marbles = 9.0 →
  given_marbles = 3.0 →
  initial_marbles - given_marbles = 6.0 := by
sorry

end NUMINAMATH_CALUDE_marbles_left_l1720_172091


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l1720_172079

theorem smallest_n_congruence : ∃! n : ℕ, (∀ a ∈ Finset.range 9, n % (a + 2) = (a + 1)) ∧ 
  (∀ m : ℕ, m < n → ∃ a ∈ Finset.range 9, m % (a + 2) ≠ (a + 1)) ∧ n = 2519 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l1720_172079


namespace NUMINAMATH_CALUDE_line_points_relation_l1720_172009

/-- Given a line in the xy-coordinate system with equation x = 2y + 5,
    if (m, n) and (m + 1, n + k) are two points on this line,
    then k = 1/2 -/
theorem line_points_relation (m n k : ℝ) : 
  (m = 2 * n + 5) →  -- (m, n) is on the line
  (m + 1 = 2 * (n + k) + 5) →  -- (m + 1, n + k) is on the line
  k = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_line_points_relation_l1720_172009


namespace NUMINAMATH_CALUDE_rancher_loss_rancher_specific_loss_l1720_172006

/-- Calculates the total monetary loss for a rancher given specific conditions --/
theorem rancher_loss (initial_cattle : ℕ) (initial_rate : ℕ) (dead_cattle : ℕ) 
  (sick_cost : ℕ) (reduced_price : ℕ) : ℕ :=
  let expected_revenue := initial_cattle * initial_rate
  let remaining_cattle := initial_cattle - dead_cattle
  let revenue_remaining := remaining_cattle * reduced_price
  let additional_cost := dead_cattle * sick_cost
  let total_loss := (expected_revenue - revenue_remaining) + additional_cost
  total_loss

/-- Proves that the rancher's total monetary loss is $310,500 given the specific conditions --/
theorem rancher_specific_loss : 
  rancher_loss 500 700 350 80 450 = 310500 := by
  sorry

end NUMINAMATH_CALUDE_rancher_loss_rancher_specific_loss_l1720_172006


namespace NUMINAMATH_CALUDE_function_range_l1720_172033

def f (x : ℝ) : ℝ := x^2 - 2*x - 3

theorem function_range :
  {y | ∃ x ∈ Set.Ioo (-1) 2, f x = y} = Set.Icc (-4) 0 := by sorry

end NUMINAMATH_CALUDE_function_range_l1720_172033


namespace NUMINAMATH_CALUDE_total_savings_theorem_l1720_172083

/-- Represents the savings of a child in various currencies and denominations -/
structure Savings where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ
  one_dollar_bills : ℕ
  five_dollar_bills : ℕ
  two_dollar_canadian_coins : ℕ
  one_dollar_canadian_coins : ℕ
  five_dollar_canadian_bills : ℕ
  one_pound_uk_coins : ℕ

/-- Conversion rates for different currencies -/
structure ConversionRates where
  british_pound_to_usd : ℚ
  canadian_dollar_to_usd : ℚ

/-- Calculates the total savings in US dollars -/
def calculate_total_savings (teagan_savings : Savings) (rex_savings : Savings) (toni_savings : Savings) (rates : ConversionRates) : ℚ :=
  sorry

/-- Theorem stating the total savings of the three kids -/
theorem total_savings_theorem (teagan_savings rex_savings toni_savings : Savings) (rates : ConversionRates) :
  teagan_savings.pennies = 200 ∧
  teagan_savings.one_dollar_bills = 15 ∧
  teagan_savings.two_dollar_canadian_coins = 13 ∧
  rex_savings.nickels = 100 ∧
  rex_savings.quarters = 45 ∧
  rex_savings.one_pound_uk_coins = 8 ∧
  rex_savings.one_dollar_canadian_coins = 20 ∧
  toni_savings.dimes = 330 ∧
  toni_savings.five_dollar_bills = 12 ∧
  toni_savings.five_dollar_canadian_bills = 7 ∧
  rates.british_pound_to_usd = 138/100 ∧
  rates.canadian_dollar_to_usd = 76/100 →
  calculate_total_savings teagan_savings rex_savings toni_savings rates = 19885/100 := by
  sorry

end NUMINAMATH_CALUDE_total_savings_theorem_l1720_172083


namespace NUMINAMATH_CALUDE_largest_x_value_l1720_172075

theorem largest_x_value (x : ℝ) : 
  (((17 * x^2 - 40 * x + 15) / (4 * x - 3) + 7 * x = 9 * x - 3) ∧ 
   (∀ y : ℝ, ((17 * y^2 - 40 * y + 15) / (4 * y - 3) + 7 * y = 9 * y - 3) → y ≤ x)) 
  → x = 2/3 := by sorry

end NUMINAMATH_CALUDE_largest_x_value_l1720_172075


namespace NUMINAMATH_CALUDE_absolute_value_equation_l1720_172095

theorem absolute_value_equation (x y : ℝ) :
  |2*x - Real.sqrt y| = 2*x + Real.sqrt y → y = 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_l1720_172095


namespace NUMINAMATH_CALUDE_women_handshakes_fifteen_couples_l1720_172043

/-- The number of handshakes among women in a group of married couples -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a group of 15 married couples, if only women shake hands with other women
    (excluding their spouses), the total number of handshakes is 105. -/
theorem women_handshakes_fifteen_couples :
  handshakes 15 = 105 := by
  sorry

end NUMINAMATH_CALUDE_women_handshakes_fifteen_couples_l1720_172043


namespace NUMINAMATH_CALUDE_frequency_histogram_interval_length_l1720_172067

/-- Given a frequency histogram interval [a,b), prove that its length |a-b| equals m/h,
    where m is the frequency and h is the histogram height for this interval. -/
theorem frequency_histogram_interval_length
  (a b m h : ℝ)
  (h_interval : a < b)
  (h_frequency : m > 0)
  (h_height : h > 0)
  (h_histogram : h = m / (b - a)) :
  b - a = m / h :=
sorry

end NUMINAMATH_CALUDE_frequency_histogram_interval_length_l1720_172067


namespace NUMINAMATH_CALUDE_candy_given_to_janet_and_emily_l1720_172022

-- Define the initial amount of candy
def initial_candy : ℝ := 78.5

-- Define the amount left after giving to Janet
def left_after_janet : ℝ := 68.75

-- Define the amount given to Emily
def given_to_emily : ℝ := 2.25

-- Theorem to prove
theorem candy_given_to_janet_and_emily :
  initial_candy - left_after_janet + given_to_emily = 12 := by
  sorry

end NUMINAMATH_CALUDE_candy_given_to_janet_and_emily_l1720_172022


namespace NUMINAMATH_CALUDE_complex_magnitude_power_l1720_172029

theorem complex_magnitude_power : Complex.abs ((3 + 2*Complex.I)^6) = 2197 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_power_l1720_172029


namespace NUMINAMATH_CALUDE_library_shelves_l1720_172061

/-- The number of type C shelves in a library with given conditions -/
theorem library_shelves (total_books : ℕ) (books_per_a : ℕ) (books_per_b : ℕ) (books_per_c : ℕ)
  (percent_a : ℚ) (percent_b : ℚ) (percent_c : ℚ) :
  total_books = 200000 →
  books_per_a = 12 →
  books_per_b = 15 →
  books_per_c = 20 →
  percent_a = 2/5 →
  percent_b = 7/20 →
  percent_c = 1/4 →
  percent_a + percent_b + percent_c = 1 →
  ∃ (shelves_a shelves_b : ℕ),
    ↑shelves_a * books_per_a ≥ ↑total_books * percent_a ∧
    ↑shelves_b * books_per_b ≥ ↑total_books * percent_b ∧
    2500 * books_per_c = ↑total_books * percent_c :=
by sorry


end NUMINAMATH_CALUDE_library_shelves_l1720_172061


namespace NUMINAMATH_CALUDE_smallest_divisible_by_6_and_35_after_2015_l1720_172012

theorem smallest_divisible_by_6_and_35_after_2015 :
  ∀ n : ℕ, n > 2015 ∧ 6 ∣ n ∧ 35 ∣ n → n ≥ 2100 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_6_and_35_after_2015_l1720_172012


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l1720_172005

theorem complex_fraction_evaluation :
  2 + (3 / (2 + (1 / (2 + (1/2))))) = 13/4 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l1720_172005


namespace NUMINAMATH_CALUDE_emiliano_consumption_theorem_l1720_172052

/-- Represents the number of fruits in a basket -/
structure FruitBasket where
  apples : ℕ
  oranges : ℕ
  bananas : ℕ

/-- Calculates the number of fruits Emiliano consumes -/
def emilianoConsumption (basket : FruitBasket) : ℕ :=
  (3 * basket.apples / 5) + (2 * basket.oranges / 3) + (4 * basket.bananas / 7)

/-- Theorem: Given the conditions, Emiliano consumes 16 fruits -/
theorem emiliano_consumption_theorem (basket : FruitBasket) 
  (h1 : basket.apples = 15)
  (h2 : basket.apples = 4 * basket.oranges)
  (h3 : basket.bananas = 3 * basket.oranges) :
  emilianoConsumption basket = 16 := by
  sorry


end NUMINAMATH_CALUDE_emiliano_consumption_theorem_l1720_172052


namespace NUMINAMATH_CALUDE_suzy_age_l1720_172011

theorem suzy_age (mary_age : ℕ) (suzy_age : ℕ) : 
  mary_age = 8 → 
  suzy_age + 4 = 2 * (mary_age + 4) → 
  suzy_age = 20 := by
sorry

end NUMINAMATH_CALUDE_suzy_age_l1720_172011


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1720_172010

/-- A monotonic continuous function on the real numbers satisfying f(x)·f(y) = f(x+y) -/
def FunctionalEquationSolution (f : ℝ → ℝ) : Prop :=
  Monotone f ∧ Continuous f ∧ ∀ x y : ℝ, f x * f y = f (x + y)

/-- The solution to the functional equation is of the form f(x) = a^x for some a > 0 and a ≠ 1 -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : FunctionalEquationSolution f) :
  ∃ a : ℝ, a > 0 ∧ a ≠ 1 ∧ ∀ x : ℝ, f x = a^x :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1720_172010


namespace NUMINAMATH_CALUDE_running_speed_calculation_l1720_172037

/-- Proves that given the specified conditions, the running speed must be 6 mph -/
theorem running_speed_calculation (total_distance : ℝ) (running_time : ℝ) (walking_speed : ℝ) (walking_time : ℝ)
  (h1 : total_distance = 3)
  (h2 : running_time = 20 / 60)
  (h3 : walking_speed = 2)
  (h4 : walking_time = 30 / 60) :
  ∃ (running_speed : ℝ), running_speed * running_time + walking_speed * walking_time = total_distance ∧ running_speed = 6 := by
  sorry

end NUMINAMATH_CALUDE_running_speed_calculation_l1720_172037


namespace NUMINAMATH_CALUDE_M_in_fourth_quadrant_l1720_172088

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def is_in_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The given point M -/
def M : Point :=
  { x := 2, y := -5 }

/-- Theorem stating that M is in the fourth quadrant -/
theorem M_in_fourth_quadrant : is_in_fourth_quadrant M := by
  sorry


end NUMINAMATH_CALUDE_M_in_fourth_quadrant_l1720_172088


namespace NUMINAMATH_CALUDE_sally_total_spent_l1720_172064

-- Define the amounts spent on peaches and cherries
def peaches_cost : ℚ := 12.32
def cherries_cost : ℚ := 11.54

-- Define the total cost
def total_cost : ℚ := peaches_cost + cherries_cost

-- Theorem statement
theorem sally_total_spent : total_cost = 23.86 := by
  sorry

end NUMINAMATH_CALUDE_sally_total_spent_l1720_172064


namespace NUMINAMATH_CALUDE_triangle_ratio_theorem_l1720_172046

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  AB = 6 ∧ BC = 8 ∧ AC = 10

-- Define point D on AC
def PointOnLine (D A C : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (A.1 + t * (C.1 - A.1), A.2 + t * (C.2 - A.2))

-- Define the distance BD
def DistanceBD (B D : ℝ × ℝ) : Prop :=
  Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) = 6

-- Define the ratio AD:DC
def RatioADDC (A D C : ℝ × ℝ) : Prop :=
  let AD := Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2)
  let DC := Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2)
  AD / DC = 18 / 7

-- Theorem statement
theorem triangle_ratio_theorem (A B C D : ℝ × ℝ) :
  Triangle A B C → PointOnLine D A C → DistanceBD B D → RatioADDC A D C :=
by sorry

end NUMINAMATH_CALUDE_triangle_ratio_theorem_l1720_172046


namespace NUMINAMATH_CALUDE_expression_equals_100_l1720_172090

theorem expression_equals_100 : (50 - (2050 - 250)) + (2050 - (250 - 50)) = 100 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_100_l1720_172090


namespace NUMINAMATH_CALUDE_x_neq_one_necessary_not_sufficient_l1720_172063

theorem x_neq_one_necessary_not_sufficient :
  (∃ x : ℝ, x ≠ 1 ∧ x^2 - 3*x + 2 = 0) ∧
  (∀ x : ℝ, x^2 - 3*x + 2 ≠ 0 → x ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_x_neq_one_necessary_not_sufficient_l1720_172063


namespace NUMINAMATH_CALUDE_irreducible_fractions_exist_l1720_172069

theorem irreducible_fractions_exist : ∃ (a b : ℕ), 
  Nat.gcd a b = 1 ∧ Nat.gcd (a + 1) b = 1 ∧ Nat.gcd (a + 1) (b + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_irreducible_fractions_exist_l1720_172069


namespace NUMINAMATH_CALUDE_floor_sqrt_20_squared_l1720_172027

theorem floor_sqrt_20_squared : ⌊Real.sqrt 20⌋^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_20_squared_l1720_172027


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l1720_172060

theorem sum_of_x_and_y (x y : ℝ) (h : x^2 + y^2 = 10*x - 6*y - 34) : x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l1720_172060


namespace NUMINAMATH_CALUDE_smallest_sum_of_sequence_l1720_172023

theorem smallest_sum_of_sequence (E F G H : ℤ) : 
  E > 0 ∧ F > 0 ∧ G > 0 →  -- E, F, G are positive integers
  ∃ d : ℤ, G - F = F - E ∧ F - E = d →  -- E, F, G form an arithmetic sequence
  ∃ r : ℚ, G = F * r ∧ H = G * r →  -- F, G, H form a geometric sequence
  G / F = 7 / 4 →  -- Given ratio
  ∀ E' F' G' H' : ℤ,
    (E' > 0 ∧ F' > 0 ∧ G' > 0 ∧
     ∃ d' : ℤ, G' - F' = F' - E' ∧ F' - E' = d' ∧
     ∃ r' : ℚ, G' = F' * r' ∧ H' = G' * r' ∧
     G' / F' = 7 / 4) →
    E + F + G + H ≤ E' + F' + G' + H' →
  E + F + G + H = 97 := by
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_sequence_l1720_172023


namespace NUMINAMATH_CALUDE_john_movie_count_l1720_172098

/-- The number of movies John has -/
def num_movies : ℕ := 100

/-- The trade-in value of each VHS in dollars -/
def vhs_value : ℕ := 2

/-- The cost of each DVD in dollars -/
def dvd_cost : ℕ := 10

/-- The total cost to replace all movies in dollars -/
def total_cost : ℕ := 800

theorem john_movie_count :
  (dvd_cost * num_movies) - (vhs_value * num_movies) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_john_movie_count_l1720_172098


namespace NUMINAMATH_CALUDE_max_area_rectangle_in_ellipse_l1720_172056

/-- Given an ellipse b² x² + a² y² = a² b², prove that the rectangle with the largest possible area
    inscribed in the ellipse has vertices at (±(a/2)√2, ±(b/2)√2) -/
theorem max_area_rectangle_in_ellipse (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let ellipse := {p : ℝ × ℝ | b^2 * p.1^2 + a^2 * p.2^2 = a^2 * b^2}
  let inscribed_rectangle (p : ℝ × ℝ) := 
    {q : ℝ × ℝ | q ∈ ellipse ∧ |q.1| ≤ |p.1| ∧ |q.2| ≤ |p.2|}
  let area (p : ℝ × ℝ) := 4 * |p.1 * p.2|
  ∃ (p : ℝ × ℝ), p ∈ ellipse ∧
    (∀ q : ℝ × ℝ, q ∈ ellipse → area q ≤ area p) ∧
    p = (a/2 * Real.sqrt 2, b/2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_max_area_rectangle_in_ellipse_l1720_172056


namespace NUMINAMATH_CALUDE_equation_solution_l1720_172042

theorem equation_solution :
  ∃ x : ℚ, (2 * x + 5 * x = 500 - (4 * x + 6 * x + 10)) ∧ x = 490 / 17 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1720_172042


namespace NUMINAMATH_CALUDE_visits_neither_country_l1720_172073

/-- Given a group of people and information about their visits to Iceland and Norway,
    calculate the number of people who have visited neither country. -/
theorem visits_neither_country
  (total : ℕ)
  (visited_iceland : ℕ)
  (visited_norway : ℕ)
  (visited_both : ℕ)
  (h_total : total = 90)
  (h_iceland : visited_iceland = 55)
  (h_norway : visited_norway = 33)
  (h_both : visited_both = 51) :
  total - (visited_iceland + visited_norway - visited_both) = 53 := by
  sorry

#check visits_neither_country

end NUMINAMATH_CALUDE_visits_neither_country_l1720_172073


namespace NUMINAMATH_CALUDE_jenna_smoothies_l1720_172031

/-- Given that Jenna can make 15 smoothies from 3 strawberries, 
    prove that she can make 90 smoothies from 18 strawberries. -/
theorem jenna_smoothies (smoothies_per_three : ℕ) (strawberries : ℕ) 
  (h1 : smoothies_per_three = 15) 
  (h2 : strawberries = 18) : 
  (smoothies_per_three * strawberries) / 3 = 90 := by
  sorry

end NUMINAMATH_CALUDE_jenna_smoothies_l1720_172031


namespace NUMINAMATH_CALUDE_solution_set_eq_singleton_l1720_172066

/-- The solution set of the system of equations x + y = 1 and x^2 - y^2 = 9 -/
def solution_set : Set (ℝ × ℝ) :=
  {p | p.1 + p.2 = 1 ∧ p.1^2 - p.2^2 = 9}

/-- Theorem stating that the solution set contains only the point (5, -4) -/
theorem solution_set_eq_singleton :
  solution_set = {(5, -4)} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_eq_singleton_l1720_172066


namespace NUMINAMATH_CALUDE_min_value_of_expression_equality_condition_l1720_172007

theorem min_value_of_expression (a : ℝ) (h : a > 1) : a + 1 / (a - 1) ≥ 3 :=
sorry

theorem equality_condition (a : ℝ) (h : a > 1) : 
  a + 1 / (a - 1) = 3 ↔ a = 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_equality_condition_l1720_172007


namespace NUMINAMATH_CALUDE_binomial_coefficient_two_l1720_172016

theorem binomial_coefficient_two (n : ℕ) (h : n ≥ 2) : Nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_two_l1720_172016


namespace NUMINAMATH_CALUDE_reciprocal_sum_pairs_l1720_172015

theorem reciprocal_sum_pairs : 
  ∃! k : ℕ, k > 0 ∧ 
  (∃ S : Finset (ℕ × ℕ), 
    (∀ (m n : ℕ), (m, n) ∈ S ↔ m > 0 ∧ n > 0 ∧ 1 / m + 1 / n = 1 / 5) ∧
    Finset.card S = k) :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_sum_pairs_l1720_172015


namespace NUMINAMATH_CALUDE_remainder_problem_l1720_172086

theorem remainder_problem : 123456789012 % 252 = 144 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1720_172086


namespace NUMINAMATH_CALUDE_inequality_not_always_preserved_l1720_172025

theorem inequality_not_always_preserved (a b : ℝ) (h : a < b) :
  ∃ m : ℝ, ¬(m * a > m * b) :=
sorry

end NUMINAMATH_CALUDE_inequality_not_always_preserved_l1720_172025


namespace NUMINAMATH_CALUDE_value_of_x_l1720_172074

theorem value_of_x (w y z x : ℝ) 
  (hw : w = 90)
  (hz : z = 2/3 * w)
  (hy : y = 1/4 * z)
  (hx : x = 1/2 * y) : 
  x = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l1720_172074


namespace NUMINAMATH_CALUDE_cube_sum_eq_triple_product_l1720_172053

theorem cube_sum_eq_triple_product (a b c : ℝ) (h : a + b + c = 0) :
  a^3 + b^3 + c^3 = 3*a*b*c := by sorry

end NUMINAMATH_CALUDE_cube_sum_eq_triple_product_l1720_172053


namespace NUMINAMATH_CALUDE_tile_coverage_l1720_172021

/-- Represents the dimensions of a rectangle in inches -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangle given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Converts feet to inches -/
def feetToInches (feet : ℕ) : ℕ := feet * 12

theorem tile_coverage (tile : Dimensions) (region : Dimensions) : 
  tile.length = 2 ∧ tile.width = 6 ∧ 
  region.length = feetToInches 3 ∧ region.width = feetToInches 4 → 
  (area region / area tile : ℕ) = 144 := by
  sorry

#check tile_coverage

end NUMINAMATH_CALUDE_tile_coverage_l1720_172021


namespace NUMINAMATH_CALUDE_min_value_theorem_equality_condition_l1720_172077

theorem min_value_theorem (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  a^2 + b^2 + 1/a^2 + 1/b^2 + b/a ≥ 2 * Real.sqrt 5 := by
  sorry

theorem equality_condition (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ (a₀ b₀ : ℝ), a₀ ≠ 0 ∧ b₀ ≠ 0 ∧ 
    a₀^2 + b₀^2 + 1/a₀^2 + 1/b₀^2 + b₀/a₀ = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_equality_condition_l1720_172077


namespace NUMINAMATH_CALUDE_evaluate_expression_l1720_172096

theorem evaluate_expression : 3 * 307 + 4 * 307 + 2 * 307 + 307^2 = 97012 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1720_172096


namespace NUMINAMATH_CALUDE_performance_arrangements_l1720_172081

/-- The number of performances of each type -/
def num_singing : ℕ := 2
def num_dance : ℕ := 3
def num_variety : ℕ := 3

/-- The total number of performances -/
def total_performances : ℕ := num_singing + num_dance + num_variety

/-- Number of ways to arrange performances with singing at beginning and end -/
def arrangement_singing_ends : ℕ := 1440

/-- Number of ways to arrange performances with non-adjacent singing -/
def arrangement_non_adjacent_singing : ℕ := 30240

/-- Number of ways to arrange performances with adjacent singing and non-adjacent dance -/
def arrangement_adjacent_singing_non_adjacent_dance : ℕ := 2880

theorem performance_arrangements :
  (total_performances = 8) →
  (arrangement_singing_ends = 1440) ∧
  (arrangement_non_adjacent_singing = 30240) ∧
  (arrangement_adjacent_singing_non_adjacent_dance = 2880) :=
by sorry

end NUMINAMATH_CALUDE_performance_arrangements_l1720_172081


namespace NUMINAMATH_CALUDE_repeating_decimal_denominator_l1720_172051

theorem repeating_decimal_denominator : ∃ (n d : ℕ), d > 0 ∧ (n / d : ℚ) = 2 / 3 ∧ 
  (∀ (n' d' : ℕ), d' > 0 → (n' / d' : ℚ) = 2 / 3 → d ≤ d') := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_denominator_l1720_172051


namespace NUMINAMATH_CALUDE_unique_solution_l1720_172003

theorem unique_solution : ∃! (x y z : ℤ),
  (y^4 + 2*z^2) % 3 = 2 ∧
  (3*x^4 + z^2) % 5 = 1 ∧
  y^4 + 2*z^2 = 3*x^4 + z^2 - 6 ∧
  x = 5 ∧ y = 3 ∧ z = 19 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_l1720_172003


namespace NUMINAMATH_CALUDE_average_age_problem_l1720_172093

theorem average_age_problem (a c : ℝ) : 
  (a + c) / 2 = 32 →
  ((a + c) + 23) / 3 = 29 :=
by
  sorry

end NUMINAMATH_CALUDE_average_age_problem_l1720_172093


namespace NUMINAMATH_CALUDE_not_always_both_false_l1720_172084

theorem not_always_both_false (p q : Prop) : 
  ¬(p ∧ q) → (¬p ∧ ¬q) → False :=
sorry

end NUMINAMATH_CALUDE_not_always_both_false_l1720_172084


namespace NUMINAMATH_CALUDE_real_part_of_z_l1720_172008

theorem real_part_of_z (z : ℂ) (h1 : Complex.abs (z - 1) = 2) (h2 : Complex.abs (z^2 - 1) = 6) :
  z.re = 5/4 := by sorry

end NUMINAMATH_CALUDE_real_part_of_z_l1720_172008


namespace NUMINAMATH_CALUDE_tank_filling_time_l1720_172080

/-- Proves that the first pipe takes 5 hours to fill the tank alone given the conditions of the problem -/
theorem tank_filling_time (T : ℝ) 
  (h1 : T > 0)  -- Ensuring T is positive
  (h2 : 1/T + 1/4 - 1/20 = 1/2.5) : T = 5 := by
  sorry


end NUMINAMATH_CALUDE_tank_filling_time_l1720_172080


namespace NUMINAMATH_CALUDE_parallel_line_necessary_not_sufficient_l1720_172026

-- Define the type for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes and lines
variable (parallelPlanes : Plane → Plane → Prop)
variable (parallelLineToPlane : Line → Plane → Prop)

-- Define the subset relation for a line being in a plane
variable (lineInPlane : Line → Plane → Prop)

-- State the theorem
theorem parallel_line_necessary_not_sufficient
  (α β : Plane) (m : Line)
  (distinct : α ≠ β)
  (m_in_α : lineInPlane m α) :
  (parallelPlanes α β → parallelLineToPlane m β) ∧
  ¬(parallelLineToPlane m β → parallelPlanes α β) :=
sorry

end NUMINAMATH_CALUDE_parallel_line_necessary_not_sufficient_l1720_172026


namespace NUMINAMATH_CALUDE_pascal_triangle_symmetry_and_sum_l1720_172082

def pascal_triangle (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

theorem pascal_triangle_symmetry_and_sum (n : ℕ) :
  pascal_triangle 48 46 = pascal_triangle 48 2 ∧
  pascal_triangle 48 46 + pascal_triangle 48 2 = 2256 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_symmetry_and_sum_l1720_172082


namespace NUMINAMATH_CALUDE_school_supplies_cost_l1720_172054

/-- Calculates the total cost of school supplies for a class after applying a discount -/
def total_cost_after_discount (num_students : ℕ) 
                               (num_pens num_notebooks num_binders num_highlighters : ℕ)
                               (cost_pen cost_notebook cost_binder cost_highlighter : ℚ)
                               (discount : ℚ) : ℚ :=
  let cost_per_student := num_pens * cost_pen + 
                          num_notebooks * cost_notebook + 
                          num_binders * cost_binder + 
                          num_highlighters * cost_highlighter
  let total_cost := num_students * cost_per_student
  total_cost - discount

/-- Theorem stating the total cost of school supplies after discount -/
theorem school_supplies_cost :
  total_cost_after_discount 30 5 3 1 2 0.5 1.25 4.25 0.75 100 = 260 :=
by sorry

end NUMINAMATH_CALUDE_school_supplies_cost_l1720_172054


namespace NUMINAMATH_CALUDE_pencil_distribution_problem_l1720_172039

theorem pencil_distribution_problem :
  ∃! n : ℕ, 10 ≤ n ∧ n < 100 ∧ 6 ∣ n ∧ 9 ∣ n ∧ n % 7 = 1 ∧ n = 36 :=
by sorry

end NUMINAMATH_CALUDE_pencil_distribution_problem_l1720_172039


namespace NUMINAMATH_CALUDE_max_ab_perpendicular_lines_l1720_172045

theorem max_ab_perpendicular_lines (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∀ x y : ℝ, 2 * x + (2 * a - 4) * y + 1 = 0 ↔ 2 * b * x + y - 2 = 0) →
  (∀ x₁ y₁ x₂ y₂ : ℝ, 
    (2 * x₁ + (2 * a - 4) * y₁ + 1 = 0 ∧ 2 * x₂ + (2 * a - 4) * y₂ + 1 = 0 ∧ x₁ ≠ x₂) →
    (2 * b * x₁ + y₁ - 2 = 0 ∧ 2 * b * x₂ + y₂ - 2 = 0 ∧ x₁ ≠ x₂) →
    ((y₂ - y₁) / (x₂ - x₁)) * ((y₂ - y₁) / (x₂ - x₁)) = -1) →
  ∀ c : ℝ, a * b ≤ c → c = 1/2 := by
sorry

end NUMINAMATH_CALUDE_max_ab_perpendicular_lines_l1720_172045


namespace NUMINAMATH_CALUDE_car_distance_in_30_minutes_l1720_172071

theorem car_distance_in_30_minutes 
  (train_speed : ℝ) 
  (car_speed_ratio : ℝ) 
  (time : ℝ) 
  (h1 : train_speed = 90) 
  (h2 : car_speed_ratio = 2/3) 
  (h3 : time = 1/2) : 
  car_speed_ratio * train_speed * time = 30 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_in_30_minutes_l1720_172071


namespace NUMINAMATH_CALUDE_pure_imaginary_iff_m_eq_3_second_quadrant_iff_m_between_1_and_3_l1720_172076

-- Define the complex number z as a function of real m
def z (m : ℝ) : ℂ := (m^2 - 2*m - 3 : ℝ) + (m^2 - 1 : ℝ) * Complex.I

-- Part 1: z is a pure imaginary number iff m = 3
theorem pure_imaginary_iff_m_eq_3 :
  ∀ m : ℝ, (z m).re = 0 ↔ m = 3 :=
sorry

-- Part 2: z is in the second quadrant iff 1 < m < 3
theorem second_quadrant_iff_m_between_1_and_3 :
  ∀ m : ℝ, ((z m).re < 0 ∧ (z m).im > 0) ↔ (1 < m ∧ m < 3) :=
sorry

end NUMINAMATH_CALUDE_pure_imaginary_iff_m_eq_3_second_quadrant_iff_m_between_1_and_3_l1720_172076


namespace NUMINAMATH_CALUDE_square_difference_division_problem_solution_l1720_172089

theorem square_difference_division (a b : ℕ) (h : a > b) :
  (a^2 - b^2) / (a - b) = a + b :=
by sorry

theorem problem_solution :
  (275^2 - 245^2) / 30 = 520 :=
by sorry

end NUMINAMATH_CALUDE_square_difference_division_problem_solution_l1720_172089


namespace NUMINAMATH_CALUDE_quadrilateral_area_l1720_172028

/-- Represents a triangle with its area -/
structure Triangle where
  area : ℝ

/-- Represents the diagram with triangles PQR and XYZ -/
structure Diagram where
  pqr : Triangle
  xyz : Triangle
  smallestTriangleArea : ℝ
  smallestTrianglesCount : ℕ

theorem quadrilateral_area (d : Diagram) 
  (h1 : d.pqr.area = 50)
  (h2 : d.xyz.area = 200)
  (h3 : d.smallestTriangleArea = 1)
  (h4 : d.smallestTrianglesCount = 10)
  : d.xyz.area - d.pqr.area = 150 := by
  sorry

#check quadrilateral_area

end NUMINAMATH_CALUDE_quadrilateral_area_l1720_172028


namespace NUMINAMATH_CALUDE_hydrangea_year_calculation_l1720_172017

/-- The year Lily started buying hydrangeas -/
def start_year : ℕ := 1989

/-- The cost of each hydrangea plant in dollars -/
def plant_cost : ℕ := 20

/-- The total amount Lily has spent on hydrangeas in dollars -/
def total_spent : ℕ := 640

/-- The year up to which Lily has spent the total amount on hydrangeas -/
def end_year : ℕ := 2021

/-- Theorem stating that the calculated end year is correct -/
theorem hydrangea_year_calculation :
  end_year = start_year + (total_spent / plant_cost) :=
by sorry

end NUMINAMATH_CALUDE_hydrangea_year_calculation_l1720_172017


namespace NUMINAMATH_CALUDE_unique_two_digit_number_l1720_172034

/-- Represents a two-digit number -/
def TwoDigitNumber := { n : ℕ // n ≥ 10 ∧ n < 100 }

/-- Returns the tens digit of a two-digit number -/
def tens_digit (n : TwoDigitNumber) : ℕ := n.val / 10

/-- Returns the units digit of a two-digit number -/
def units_digit (n : TwoDigitNumber) : ℕ := n.val % 10

/-- The sum of digits of a two-digit number -/
def sum_of_digits (n : TwoDigitNumber) : ℕ := tens_digit n + units_digit n

/-- The product of digits of a two-digit number -/
def product_of_digits (n : TwoDigitNumber) : ℕ := tens_digit n * units_digit n

theorem unique_two_digit_number : 
  ∃! (n : TwoDigitNumber), 
    n.val = 4 * sum_of_digits n ∧ 
    n.val = 3 * product_of_digits n ∧
    n.val = 24 := by sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_l1720_172034


namespace NUMINAMATH_CALUDE_complex_modulus_range_l1720_172001

theorem complex_modulus_range (a : ℝ) : 
  (∀ θ : ℝ, Complex.abs ((a + Real.cos θ) + (2 * a - Real.sin θ) * Complex.I) ≤ 2) ↔ 
  a ∈ Set.Icc (-(Real.sqrt 5 / 5)) (Real.sqrt 5 / 5) := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_range_l1720_172001


namespace NUMINAMATH_CALUDE_minimum_students_l1720_172044

theorem minimum_students (b g : ℕ) : 
  (2 * (b / 2) = 2 * (g * 2 / 3) + 5) →  -- Half of boys equals 2/3 of girls plus 5
  (b ≥ g) →                             -- There are at least as many boys as girls
  (b + g ≥ 17) ∧                        -- The total number of students is at least 17
  (∀ b' g' : ℕ, (2 * (b' / 2) = 2 * (g' * 2 / 3) + 5) → (b' + g' < 17) → (b' < g')) :=
by
  sorry

#check minimum_students

end NUMINAMATH_CALUDE_minimum_students_l1720_172044


namespace NUMINAMATH_CALUDE_smallest_multiple_45_60_not_25_l1720_172070

theorem smallest_multiple_45_60_not_25 : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (45 ∣ n) ∧ 
  (60 ∣ n) ∧ 
  ¬(25 ∣ n) ∧
  (∀ m : ℕ, m > 0 ∧ (45 ∣ m) ∧ (60 ∣ m) ∧ ¬(25 ∣ m) → n ≤ m) ∧
  n = 180 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_45_60_not_25_l1720_172070


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1720_172068

/-- The eccentricity of the hyperbola x^2 - y^2 = 1 is √2 -/
theorem hyperbola_eccentricity :
  let h : ℝ → ℝ → Prop := λ x y ↦ x^2 - y^2 = 1
  ∃ e : ℝ, e = Real.sqrt 2 ∧ ∀ x y : ℝ, h x y → e = (Real.sqrt (x^2 + y^2)) / (Real.sqrt (x^2 - 1)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1720_172068


namespace NUMINAMATH_CALUDE_remainder_theorem_l1720_172024

theorem remainder_theorem (x : ℕ+) (h : (7 * x.val) % 29 = 1) :
  (13 + x.val) % 29 = 9 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1720_172024


namespace NUMINAMATH_CALUDE_last_item_to_second_recipient_l1720_172002

/-- Represents the cyclic distribution of items among recipients. -/
def cyclicDistribution (items : ℕ) (recipients : ℕ) : ℕ :=
  (items - 1) % recipients + 1

/-- Theorem stating that in a cyclic distribution of 278 items among 6 recipients,
    the 2nd recipient in the initial order receives the last item. -/
theorem last_item_to_second_recipient :
  cyclicDistribution 278 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_last_item_to_second_recipient_l1720_172002


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l1720_172020

theorem trigonometric_simplification (x y : ℝ) :
  Real.sin x ^ 2 + Real.sin (x + y) ^ 2 - 2 * Real.sin x * Real.cos y * Real.sin (x + y) = Real.cos y ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l1720_172020


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_three_numbers_l1720_172018

theorem arithmetic_mean_of_three_numbers (a b c : ℕ) (h : a = 18 ∧ b = 27 ∧ c = 45) : 
  (a + b + c) / 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_three_numbers_l1720_172018


namespace NUMINAMATH_CALUDE_square_plus_inverse_square_equals_six_l1720_172047

theorem square_plus_inverse_square_equals_six (m : ℝ) (h : m^2 - 2*m - 1 = 0) : 
  m^2 + 1/m^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_inverse_square_equals_six_l1720_172047


namespace NUMINAMATH_CALUDE_sin_90_degrees_l1720_172032

theorem sin_90_degrees : Real.sin (π / 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_90_degrees_l1720_172032


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l1720_172058

theorem quadratic_equal_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 + 4*x + m = 0 ∧ 
   ∀ y : ℝ, y^2 + 4*y + m = 0 → y = x) → 
  m = 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l1720_172058


namespace NUMINAMATH_CALUDE_special_operation_example_l1720_172050

def special_operation (a b : ℝ) : ℝ := 2 * a + 5 * b

theorem special_operation_example : special_operation 4 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_special_operation_example_l1720_172050


namespace NUMINAMATH_CALUDE_balls_to_one_pile_l1720_172004

/-- Represents a configuration of piles of balls -/
structure BallConfiguration (n : ℕ) where
  piles : List ℕ
  sum_balls : List.sum piles = 2^n

/-- Represents a move between two piles -/
inductive Move (n : ℕ)
| move : (a b : ℕ) → a ≥ b → a + b ≤ 2^n → Move n

/-- Represents a sequence of moves -/
def MoveSequence (n : ℕ) := List (Move n)

/-- Applies a move to a configuration -/
def applyMove (config : BallConfiguration n) (m : Move n) : BallConfiguration n :=
  sorry

/-- Applies a sequence of moves to a configuration -/
def applyMoveSequence (config : BallConfiguration n) (seq : MoveSequence n) : BallConfiguration n :=
  sorry

/-- Checks if all balls are in one pile -/
def isOnePile (config : BallConfiguration n) : Prop :=
  ∃ p, config.piles = [p]

/-- The main theorem to prove -/
theorem balls_to_one_pile (n : ℕ) (initial : BallConfiguration n) :
  ∃ (seq : MoveSequence n), isOnePile (applyMoveSequence initial seq) :=
sorry

end NUMINAMATH_CALUDE_balls_to_one_pile_l1720_172004


namespace NUMINAMATH_CALUDE_ping_pong_dominating_subset_l1720_172097

/-- Represents a ping-pong match result between two players -/
inductive MatchResult
  | Win
  | Loss

/-- Represents a team of ping-pong players -/
def Team := Fin 1000

/-- Represents the result of all matches between two teams -/
def MatchResults := Team → Team → MatchResult

theorem ping_pong_dominating_subset (results : MatchResults) :
  ∃ (dominating_team : Bool) (subset : Finset Team),
    subset.card ≤ 10 ∧
    ∀ (opponent : Team),
      ∃ (player : Team),
        player ∈ subset ∧
        ((dominating_team = true  ∧ results player opponent = MatchResult.Win) ∨
         (dominating_team = false ∧ results opponent player = MatchResult.Loss)) :=
sorry

end NUMINAMATH_CALUDE_ping_pong_dominating_subset_l1720_172097


namespace NUMINAMATH_CALUDE_east_bus_speed_l1720_172049

/-- The speed of a bus traveling east, given that it and another bus traveling
    west at 60 mph end up 460 miles apart after 4 hours. -/
theorem east_bus_speed : ℝ := by
  -- Define the speed of the west-traveling bus
  let west_speed : ℝ := 60
  -- Define the time of travel
  let time : ℝ := 4
  -- Define the total distance between buses after travel
  let total_distance : ℝ := 460
  -- Define the speed of the east-traveling bus
  let east_speed : ℝ := (total_distance / time) - west_speed
  -- Assert that the east_speed is equal to 55
  have h : east_speed = 55 := by sorry
  -- Return the speed of the east-traveling bus
  exact east_speed

end NUMINAMATH_CALUDE_east_bus_speed_l1720_172049


namespace NUMINAMATH_CALUDE_aquaflow_pump_solution_l1720_172092

/-- Represents the Aquaflow system pumping problem -/
def AquaflowPump (initial_rate : ℝ) (increased_rate : ℝ) (target_volume : ℝ) : Prop :=
  let initial_time := 30 -- minutes
  let initial_volume := initial_rate * (initial_time / 60)
  let remaining_volume := target_volume - initial_volume
  let increased_time := (remaining_volume / increased_rate) * 60
  initial_time + increased_time = 75

/-- Theorem stating the solution to the Aquaflow pumping problem -/
theorem aquaflow_pump_solution :
  AquaflowPump 360 480 540 := by
  sorry

end NUMINAMATH_CALUDE_aquaflow_pump_solution_l1720_172092


namespace NUMINAMATH_CALUDE_moles_of_MgO_formed_l1720_172041

-- Define the chemical elements and compounds
inductive Chemical
| Mg
| CO2
| MgO
| C

-- Define a structure to represent a chemical equation
structure ChemicalEquation :=
  (reactants : List (Chemical × ℕ))
  (products : List (Chemical × ℕ))

-- Define the balanced chemical equation
def balancedEquation : ChemicalEquation :=
  { reactants := [(Chemical.Mg, 2), (Chemical.CO2, 1)]
  , products := [(Chemical.MgO, 2), (Chemical.C, 1)] }

-- Define the available moles of reactants
def availableMg : ℕ := 2
def availableCO2 : ℕ := 1

-- Theorem to prove
theorem moles_of_MgO_formed :
  availableMg = 2 →
  availableCO2 = 1 →
  (balancedEquation.reactants.map (λ (c, n) => (c, n)) = [(Chemical.Mg, 2), (Chemical.CO2, 1)]) →
  (balancedEquation.products.map (λ (c, n) => (c, n)) = [(Chemical.MgO, 2), (Chemical.C, 1)]) →
  ∃ (molesOfMgO : ℕ), molesOfMgO = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_moles_of_MgO_formed_l1720_172041


namespace NUMINAMATH_CALUDE_min_value_a_l1720_172035

theorem min_value_a (a : ℝ) : 
  (∀ x : ℝ, x > 0 → a * x * Real.exp x - x - Real.log x ≥ 0) → 
  a ≥ 1 / Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_min_value_a_l1720_172035


namespace NUMINAMATH_CALUDE_greatest_five_digit_with_product_120_sum_18_l1720_172013

/-- Represents a five-digit number -/
def FiveDigitNumber := Fin 100000

/-- Returns true if the number is five digits -/
def isFiveDigit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

/-- Returns the product of digits of a natural number -/
def digitProduct (n : ℕ) : ℕ := sorry

/-- Returns the sum of digits of a natural number -/
def digitSum (n : ℕ) : ℕ := sorry

/-- The greatest five-digit number whose digits have a product of 120 -/
def N : FiveDigitNumber := sorry

theorem greatest_five_digit_with_product_120_sum_18 :
  isFiveDigit N.val ∧ 
  digitProduct N.val = 120 ∧ 
  (∀ m : FiveDigitNumber, digitProduct m.val = 120 → m.val ≤ N.val) →
  digitSum N.val = 18 := by sorry

end NUMINAMATH_CALUDE_greatest_five_digit_with_product_120_sum_18_l1720_172013


namespace NUMINAMATH_CALUDE_andy_candy_problem_l1720_172099

/-- The number of teachers who gave Andy candy canes -/
def num_teachers : ℕ := sorry

/-- The number of candy canes Andy gets from his parents -/
def candy_from_parents : ℕ := 2

/-- The number of candy canes Andy gets from each teacher -/
def candy_per_teacher : ℕ := 3

/-- The fraction of candy canes Andy buys compared to what he was given -/
def buy_fraction : ℚ := 1 / 7

/-- The number of candy canes that cause one cavity -/
def candy_per_cavity : ℕ := 4

/-- The total number of cavities Andy gets -/
def total_cavities : ℕ := 16

theorem andy_candy_problem :
  let total_candy := candy_from_parents + num_teachers * candy_per_teacher
  let bought_candy := (total_candy : ℚ) * buy_fraction
  (↑total_candy + bought_candy) / candy_per_cavity = total_cavities ↔ num_teachers = 18 := by
  sorry

end NUMINAMATH_CALUDE_andy_candy_problem_l1720_172099


namespace NUMINAMATH_CALUDE_janets_class_size_l1720_172000

/-- The number of children in Janet's class -/
def num_children : ℕ := 35

/-- The number of chaperones -/
def num_chaperones : ℕ := 5

/-- The number of additional lunches -/
def additional_lunches : ℕ := 3

/-- The cost of each lunch in dollars -/
def lunch_cost : ℕ := 7

/-- The total cost of all lunches in dollars -/
def total_cost : ℕ := 308

theorem janets_class_size :
  num_children + num_chaperones + 1 + additional_lunches = total_cost / lunch_cost :=
sorry

end NUMINAMATH_CALUDE_janets_class_size_l1720_172000


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1720_172014

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, x^2 + a*x + b < 0 ↔ 2 < x ∧ x < 4) → b - a = 14 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1720_172014


namespace NUMINAMATH_CALUDE_wheel_radii_theorem_l1720_172059

/-- The ratio of revolutions per minute of wheel A to wheel B -/
def revolution_ratio : ℚ := 1200 / 1500

/-- The total length from the outer radius of wheel A to the outer radius of wheel B in cm -/
def total_length : ℝ := 9

/-- The radius of wheel A in cm -/
def radius_A : ℝ := 2.5

/-- The radius of wheel B in cm -/
def radius_B : ℝ := 2

theorem wheel_radii_theorem :
  revolution_ratio = 4 / 5 ∧
  2 * (radius_A + radius_B) = total_length ∧
  radius_A * 4 = radius_B * 5 := by
  sorry

end NUMINAMATH_CALUDE_wheel_radii_theorem_l1720_172059


namespace NUMINAMATH_CALUDE_contrapositive_inequality_l1720_172055

theorem contrapositive_inequality (a b c : ℝ) :
  (¬(a + c < b + c) → ¬(a < b)) ↔ (a < b → a + c < b + c) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_inequality_l1720_172055


namespace NUMINAMATH_CALUDE_isosceles_triangle_third_vertex_y_coordinate_l1720_172094

/-- 
Given an isosceles triangle with:
- Base vertices at (3, 5) and (13, 5)
- Two equal sides of length 10 units
- Third vertex in the first quadrant

Prove that the y-coordinate of the third vertex is 5 + 5√3
-/
theorem isosceles_triangle_third_vertex_y_coordinate :
  ∀ (x y : ℝ),
  x > 0 →  -- First quadrant condition for x
  y > 5 →  -- First quadrant condition for y
  (x - 3)^2 + (y - 5)^2 = 100 →  -- Distance from (3, 5) is 10
  (x - 13)^2 + (y - 5)^2 = 100 →  -- Distance from (13, 5) is 10
  y = 5 + 5 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_third_vertex_y_coordinate_l1720_172094


namespace NUMINAMATH_CALUDE_unique_solution_l1720_172038

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  value : Nat
  h1 : value ≥ 100
  h2 : value ≤ 999

/-- Check if a number has distinct digits in ascending order -/
def hasDistinctAscendingDigits (n : ThreeDigitNumber) : Prop :=
  let d1 := n.value / 100
  let d2 := (n.value / 10) % 10
  let d3 := n.value % 10
  d1 < d2 ∧ d2 < d3

/-- Check if all words in the name of a number start with the same letter -/
def allWordsSameInitial (n : ThreeDigitNumber) : Prop :=
  sorry

/-- Check if a number has identical digits -/
def hasIdenticalDigits (n : ThreeDigitNumber) : Prop :=
  let d1 := n.value / 100
  let d2 := (n.value / 10) % 10
  let d3 := n.value % 10
  d1 = d2 ∧ d2 = d3

/-- Check if all words in the name of a number start with different letters -/
def allWordsDifferentInitials (n : ThreeDigitNumber) : Prop :=
  sorry

/-- The main theorem stating the unique solution to the problem -/
theorem unique_solution :
  ∃! (n1 n2 : ThreeDigitNumber),
    (hasDistinctAscendingDigits n1 ∧ allWordsSameInitial n1) ∧
    (hasIdenticalDigits n2 ∧ allWordsDifferentInitials n2) ∧
    n1.value = 147 ∧ n2.value = 111 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1720_172038


namespace NUMINAMATH_CALUDE_green_ball_probability_l1720_172040

-- Define the total number of balls
def total_balls : ℕ := 20

-- Define the number of red balls
def red_balls : ℕ := 5

-- Define the number of yellow balls
def yellow_balls : ℕ := 5

-- Define the number of green balls
def green_balls : ℕ := 10

-- Define the probability of drawing a green ball given it's not red
def prob_green_given_not_red : ℚ := green_balls / (total_balls - red_balls)

-- Theorem statement
theorem green_ball_probability :
  prob_green_given_not_red = 2/3 :=
sorry

end NUMINAMATH_CALUDE_green_ball_probability_l1720_172040


namespace NUMINAMATH_CALUDE_smallest_b_value_l1720_172062

theorem smallest_b_value (a b c : ℤ) 
  (h1 : a < b) (h2 : b < c)
  (h3 : b * b = a * c)  -- Geometric progression condition
  (h4 : a + b = 2 * c)  -- Arithmetic progression condition
  : b ≥ 2 ∧ ∃ (a' b' c' : ℤ), a' < b' ∧ b' < c' ∧ b' * b' = a' * c' ∧ a' + b' = 2 * c' ∧ b' = 2 :=
by sorry

#check smallest_b_value

end NUMINAMATH_CALUDE_smallest_b_value_l1720_172062


namespace NUMINAMATH_CALUDE_complex_simplification_l1720_172019

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- The given complex number -/
noncomputable def z : ℂ := (9 + 2 * i) / (2 + i)

/-- The theorem stating that the given complex number equals 4 - i -/
theorem complex_simplification : z = 4 - i := by sorry

end NUMINAMATH_CALUDE_complex_simplification_l1720_172019


namespace NUMINAMATH_CALUDE_f_min_f_min_range_g_max_min_l1720_172036

-- Define the function f(x) = |x-2| + |x-3|
def f (x : ℝ) : ℝ := |x - 2| + |x - 3|

-- Define the function g(x) = |x-2| + |x-3| - |x-1|
def g (x : ℝ) : ℝ := |x - 2| + |x - 3| - |x - 1|

-- Theorem stating the minimum value of f(x)
theorem f_min : ∃ (x : ℝ), ∀ (y : ℝ), f x ≤ f y ∧ f x = 1 :=
sorry

-- Theorem stating the range where f(x) is minimized
theorem f_min_range : ∀ (x : ℝ), f x = 1 → 2 ≤ x ∧ x < 3 :=
sorry

-- Main theorem
theorem g_max_min :
  (∃ (x : ℝ), ∀ (y : ℝ), f x ≤ f y) →
  (∃ (a b : ℝ), (∀ (x : ℝ), f x = 1 → g x ≤ a ∧ b ≤ g x) ∧ a = 0 ∧ b = -1) :=
sorry

end NUMINAMATH_CALUDE_f_min_f_min_range_g_max_min_l1720_172036


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l1720_172087

/-- An isosceles right triangle with legs of length 8 units -/
structure IsoscelesRightTriangle where
  /-- The length of each leg -/
  leg_length : ℝ
  /-- The leg length is 8 units -/
  leg_is_eight : leg_length = 8

/-- The inscribed circle of the isosceles right triangle -/
def inscribed_circle (t : IsoscelesRightTriangle) : ℝ := sorry

/-- Theorem: The radius of the inscribed circle in an isosceles right triangle
    with legs of length 8 units is 8 - 4√2 -/
theorem inscribed_circle_radius (t : IsoscelesRightTriangle) :
  inscribed_circle t = 8 - 4 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l1720_172087


namespace NUMINAMATH_CALUDE_zoes_purchase_cost_l1720_172065

/-- The total cost of Zoe's purchase for herself and her family -/
def total_cost (num_people : ℕ) (soda_cost pizza_cost icecream_cost topping_cost : ℚ) 
  (num_toppings icecream_per_person : ℕ) : ℚ :=
  let soda_total := num_people * soda_cost
  let pizza_total := num_people * (pizza_cost + num_toppings * topping_cost)
  let icecream_total := num_people * icecream_per_person * icecream_cost
  soda_total + pizza_total + icecream_total

/-- Theorem stating that Zoe's total purchase cost is $54.00 -/
theorem zoes_purchase_cost :
  total_cost 6 0.5 1 3 0.75 2 2 = 54 := by
  sorry

end NUMINAMATH_CALUDE_zoes_purchase_cost_l1720_172065


namespace NUMINAMATH_CALUDE_not_right_triangle_l1720_172048

theorem not_right_triangle (a b c : ℚ) (ha : a = 2/3) (hb : b = 2) (hc : c = 5/4) :
  ¬(a^2 + b^2 = c^2) := by sorry

end NUMINAMATH_CALUDE_not_right_triangle_l1720_172048


namespace NUMINAMATH_CALUDE_mario_salary_increase_l1720_172078

/-- Proves that Mario's salary increase is 0% given the conditions of the problem -/
theorem mario_salary_increase (mario_salary_this_year : ℝ) 
  (bob_salary_last_year : ℝ) (bob_salary_increase : ℝ) :
  mario_salary_this_year = 4000 →
  bob_salary_last_year = 3 * mario_salary_this_year →
  bob_salary_increase = 0.2 →
  (mario_salary_this_year / bob_salary_last_year * 3 - 1) * 100 = 0 := by
  sorry

end NUMINAMATH_CALUDE_mario_salary_increase_l1720_172078


namespace NUMINAMATH_CALUDE_simplify_fraction_l1720_172057

theorem simplify_fraction : 8 * (15 / 9) * (-45 / 40) = -1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1720_172057


namespace NUMINAMATH_CALUDE_sequence_is_arithmetic_l1720_172072

/-- Given a sequence a_n with sum of first n terms S_n = n^2 + 1, prove it's arithmetic -/
theorem sequence_is_arithmetic (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (sum_def : ∀ n : ℕ, S n = n^2 + 1) 
  (sum_relation : ∀ n : ℕ, S n = (n : ℝ) * (a 1 + a n) / 2) :
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d :=
by sorry

end NUMINAMATH_CALUDE_sequence_is_arithmetic_l1720_172072


namespace NUMINAMATH_CALUDE_parabola_points_theorem_l1720_172030

/-- Parabola passing through given points -/
def parabola (a c : ℝ) (x : ℝ) : ℝ := a * x^2 + x + c

theorem parabola_points_theorem :
  ∃ (a c m n : ℝ),
    (parabola a c 0 = -2) ∧
    (parabola a c 1 = 1) ∧
    (parabola a c 2 = m) ∧
    (parabola a c n = -2) ∧
    (a = 2) ∧
    (c = -2) ∧
    (m = 8) ∧
    (n = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_parabola_points_theorem_l1720_172030


namespace NUMINAMATH_CALUDE_least_possible_area_of_square_l1720_172085

/-- Represents the measurement of a square's side length to the nearest centimeter. -/
def MeasuredSideLength : ℝ := 4

/-- The minimum possible actual side length given the measured side length. -/
def MinActualSideLength : ℝ := MeasuredSideLength - 0.5

/-- Calculates the area of a square given its side length. -/
def SquareArea (sideLength : ℝ) : ℝ := sideLength * sideLength

/-- The least possible actual area of the square. -/
def LeastPossibleArea : ℝ := SquareArea MinActualSideLength

theorem least_possible_area_of_square :
  LeastPossibleArea = 12.25 := by
  sorry

end NUMINAMATH_CALUDE_least_possible_area_of_square_l1720_172085
