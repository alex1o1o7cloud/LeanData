import Mathlib

namespace NUMINAMATH_CALUDE_decagon_diagonals_l3777_377792

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A decagon has 10 sides -/
def decagon_sides : ℕ := 10

theorem decagon_diagonals : num_diagonals decagon_sides = 35 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonals_l3777_377792


namespace NUMINAMATH_CALUDE_kitten_food_consumption_l3777_377721

/-- Proves that given the conditions, each kitten eats 0.75 cans of food per day -/
theorem kitten_food_consumption
  (num_kittens : ℕ)
  (num_adult_cats : ℕ)
  (initial_food : ℕ)
  (additional_food : ℕ)
  (days : ℕ)
  (adult_cat_consumption : ℚ)
  (h1 : num_kittens = 4)
  (h2 : num_adult_cats = 3)
  (h3 : initial_food = 7)
  (h4 : additional_food = 35)
  (h5 : days = 7)
  (h6 : adult_cat_consumption = 1)
  : (initial_food + additional_food - num_adult_cats * adult_cat_consumption * days) / (num_kittens * days) = 0.75 := by
  sorry


end NUMINAMATH_CALUDE_kitten_food_consumption_l3777_377721


namespace NUMINAMATH_CALUDE_complex_equation_proof_l3777_377734

theorem complex_equation_proof (a b : ℝ) : (-2 * I + 1 : ℂ) = a + b * I → a - b = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_proof_l3777_377734


namespace NUMINAMATH_CALUDE_gcd_18_30_l3777_377798

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_18_30_l3777_377798


namespace NUMINAMATH_CALUDE_smallest_n_l3777_377797

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem smallest_n : ∃ (n : ℕ), n > 0 ∧ 
  is_factor 25 (n * 2^5 * 6^2 * 7^3) ∧ 
  is_factor 27 (n * 2^5 * 6^2 * 7^3) ∧
  (∀ (m : ℕ), m > 0 → 
    is_factor 25 (m * 2^5 * 6^2 * 7^3) → 
    is_factor 27 (m * 2^5 * 6^2 * 7^3) → 
    m ≥ n) ∧
  n = 75 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_l3777_377797


namespace NUMINAMATH_CALUDE_greatest_integer_for_domain_all_reals_l3777_377748

theorem greatest_integer_for_domain_all_reals : 
  ∃ (b : ℤ), b = 11 ∧ 
  (∀ (c : ℤ), c > b → 
    ∃ (x : ℝ), 2 * x^2 + c * x + 18 = 0) ∧
  (∀ (x : ℝ), 2 * x^2 + b * x + 18 ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_for_domain_all_reals_l3777_377748


namespace NUMINAMATH_CALUDE_units_digit_of_k_squared_plus_two_to_k_l3777_377742

def k : ℕ := 2010^2 + 2^2010

theorem units_digit_of_k_squared_plus_two_to_k (k : ℕ := 2010^2 + 2^2010) : 
  (k^2 + 2^k) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_k_squared_plus_two_to_k_l3777_377742


namespace NUMINAMATH_CALUDE_area_triangle_pqr_l3777_377733

/-- Given two lines intersecting at point P(2,8), with slopes 1 and 3 respectively,
    and Q and R being the intersections of these lines with the x-axis,
    the area of triangle PQR is 64/3. -/
theorem area_triangle_pqr :
  let P : ℝ × ℝ := (2, 8)
  let slope1 : ℝ := 1
  let slope2 : ℝ := 3
  let Q : ℝ × ℝ := (P.1 - P.2 / slope1, 0)
  let R : ℝ × ℝ := (P.1 - P.2 / slope2, 0)
  let area : ℝ := (1 / 2) * |Q.1 - R.1| * P.2
  area = 64 / 3 := by
sorry

end NUMINAMATH_CALUDE_area_triangle_pqr_l3777_377733


namespace NUMINAMATH_CALUDE_lcm_20_45_75_l3777_377707

theorem lcm_20_45_75 : Nat.lcm 20 (Nat.lcm 45 75) = 900 := by
  sorry

end NUMINAMATH_CALUDE_lcm_20_45_75_l3777_377707


namespace NUMINAMATH_CALUDE_two_tangents_iff_m_gt_two_l3777_377787

/-- The circle equation: x^2 + y^2 + mx + 1 = 0 -/
def circle_equation (x y m : ℝ) : Prop := x^2 + y^2 + m*x + 1 = 0

/-- The point A -/
def point_A : ℝ × ℝ := (1, 0)

/-- Condition for two tangents to be drawn from a point to a circle -/
def two_tangents_condition (m : ℝ) : Prop :=
  let center := (-m/2, 0)
  let radius_squared := m^2/4 - 1
  let distance_squared := (point_A.1 - center.1)^2 + (point_A.2 - center.2)^2
  distance_squared > radius_squared ∧ radius_squared > 0

theorem two_tangents_iff_m_gt_two :
  ∀ m : ℝ, two_tangents_condition m ↔ m > 2 :=
sorry

end NUMINAMATH_CALUDE_two_tangents_iff_m_gt_two_l3777_377787


namespace NUMINAMATH_CALUDE_all_three_live_to_75_exactly_one_lives_to_75_at_least_one_lives_to_75_l3777_377723

def probability_live_to_75 : ℝ := 0.60

-- Probability that all three policyholders live to 75
theorem all_three_live_to_75 : 
  probability_live_to_75 ^ 3 = 0.216 := by sorry

-- Probability that exactly one out of three policyholders lives to 75
theorem exactly_one_lives_to_75 : 
  3 * probability_live_to_75 * (1 - probability_live_to_75) ^ 2 = 0.288 := by sorry

-- Probability that at least one out of three policyholders lives to 75
theorem at_least_one_lives_to_75 : 
  1 - (1 - probability_live_to_75) ^ 3 = 0.936 := by sorry

end NUMINAMATH_CALUDE_all_three_live_to_75_exactly_one_lives_to_75_at_least_one_lives_to_75_l3777_377723


namespace NUMINAMATH_CALUDE_melanie_trout_l3777_377713

def melanie_catch : ℕ → ℕ → Prop
| m, t => t = 2 * m

theorem melanie_trout (tom_catch : ℕ) (h : melanie_catch 8 tom_catch) (h2 : tom_catch = 16) : 
  8 = 8 := by sorry

end NUMINAMATH_CALUDE_melanie_trout_l3777_377713


namespace NUMINAMATH_CALUDE_comic_book_collection_comparison_l3777_377718

/-- Represents the number of comic books in a collection after a given number of months -/
def comic_books (initial : ℕ) (monthly_addition : ℕ) (months : ℕ) : ℕ :=
  initial + monthly_addition * months

/-- The month when LaShawn's collection becomes at least three times Kymbrea's -/
def target_month : ℕ := 33

theorem comic_book_collection_comparison :
  ∀ m : ℕ, m < target_month →
    3 * comic_books 50 3 m > comic_books 20 5 m ∧
    3 * comic_books 50 3 target_month ≤ comic_books 20 5 target_month :=
by sorry

end NUMINAMATH_CALUDE_comic_book_collection_comparison_l3777_377718


namespace NUMINAMATH_CALUDE_min_real_roots_l3777_377732

/-- A polynomial of degree 12 with real coefficients -/
def RealPolynomial12 : Type := { p : Polynomial ℝ // p.degree = 12 }

/-- The roots of a polynomial -/
def roots (p : RealPolynomial12) : Finset ℂ := sorry

/-- The number of distinct absolute values among the roots -/
def distinctAbsValues (p : RealPolynomial12) : ℕ := sorry

/-- The number of real roots of a polynomial -/
def realRootCount (p : RealPolynomial12) : ℕ := sorry

/-- The theorem stating the minimum number of real roots -/
theorem min_real_roots (p : RealPolynomial12) 
  (h : distinctAbsValues p = 6) : 
  ∃ q : RealPolynomial12, realRootCount q = 1 ∧ 
    ∀ r : RealPolynomial12, realRootCount r ≥ 1 := by sorry

end NUMINAMATH_CALUDE_min_real_roots_l3777_377732


namespace NUMINAMATH_CALUDE_rad_polynomial_characterization_l3777_377746

/-- rad(n) is the product of all distinct prime factors of n -/
def rad (n : ℕ+) : ℕ+ := sorry

/-- A number is square-free if it's not divisible by any perfect square other than 1 -/
def IsSquareFree (n : ℕ+) : Prop := sorry

/-- Polynomial with rational coefficients -/
def RationalPolynomial := Polynomial ℚ

theorem rad_polynomial_characterization (P : RationalPolynomial) :
  (∃ (s : Set ℕ+), Set.Infinite s ∧ ∀ n ∈ s, (P.eval n : ℚ) = (rad n : ℚ)) ↔
  (∃ b : ℕ+, P = Polynomial.monomial 1 (1 / (b : ℚ))) ∨
  (∃ k : ℕ+, IsSquareFree k ∧ P = Polynomial.C (k : ℚ)) := by sorry

end NUMINAMATH_CALUDE_rad_polynomial_characterization_l3777_377746


namespace NUMINAMATH_CALUDE_r_value_when_n_is_3_l3777_377770

theorem r_value_when_n_is_3 : 
  ∀ (n s r : ℕ), 
    s = 2^n - 1 → 
    r = 3^s - s → 
    n = 3 → 
    r = 2180 := by
  sorry

end NUMINAMATH_CALUDE_r_value_when_n_is_3_l3777_377770


namespace NUMINAMATH_CALUDE_coupon_probability_l3777_377783

theorem coupon_probability (n m k : ℕ) (hn : n = 17) (hm : m = 9) (hk : k = 6) :
  (Nat.choose k k * Nat.choose (n - k) (m - k)) / Nat.choose n m = 3 / 442 := by
  sorry

end NUMINAMATH_CALUDE_coupon_probability_l3777_377783


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l3777_377762

theorem smallest_prime_divisor_of_sum (n : ℕ) : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ (3^19 + 11^13) ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ (3^19 + 11^13) → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l3777_377762


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3777_377731

/-- Given that the solution set of ax^2 + bx + 1 > 0 is (-1/2, 1/3), prove that a - b = -5 -/
theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, ax^2 + b*x + 1 > 0 ↔ -1/2 < x ∧ x < 1/3) → a - b = -5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3777_377731


namespace NUMINAMATH_CALUDE_tangent_line_at_one_l3777_377760

open Real

noncomputable def f (x : ℝ) := log x - 3 * x

theorem tangent_line_at_one :
  ∃ (m b : ℝ), (∀ x y, y = m * x + b ↔ 2 * x + y + 1 = 0) ∧
               m = deriv f 1 ∧
               f 1 = m * 1 + b := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_l3777_377760


namespace NUMINAMATH_CALUDE_desired_line_equation_l3777_377755

-- Define the lines
def line1 (x y : ℝ) : Prop := 2 * x - y + 4 = 0
def line2 (x y : ℝ) : Prop := x - y + 5 = 0
def line3 (x y : ℝ) : Prop := x - 2 * y = 0

-- Define the intersection point
def intersection (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Define perpendicularity
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- The theorem to prove
theorem desired_line_equation : 
  ∃ (x y : ℝ), intersection x y ∧ 
  (∃ (m : ℝ), perpendicular m (1/2) ∧ 
  (∀ (x' y' : ℝ), 2 * x' + y' - 8 = 0 ↔ y' - y = m * (x' - x))) :=
sorry

end NUMINAMATH_CALUDE_desired_line_equation_l3777_377755


namespace NUMINAMATH_CALUDE_ten_teams_in_tournament_l3777_377739

/-- The number of games played in a round-robin tournament with n teams -/
def games_played (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a round-robin tournament with 45 games, there were 10 teams -/
theorem ten_teams_in_tournament (h : games_played 10 = 45) : 
  ∃ (n : ℕ), n = 10 ∧ games_played n = 45 :=
by sorry

end NUMINAMATH_CALUDE_ten_teams_in_tournament_l3777_377739


namespace NUMINAMATH_CALUDE_money_spent_on_baseball_gear_l3777_377796

def initial_amount : ℕ := 67
def amount_left : ℕ := 34

theorem money_spent_on_baseball_gear :
  initial_amount - amount_left = 33 := by sorry

end NUMINAMATH_CALUDE_money_spent_on_baseball_gear_l3777_377796


namespace NUMINAMATH_CALUDE_sector_area_l3777_377740

theorem sector_area (angle : Real) (radius : Real) : 
  angle = 150 * π / 180 → 
  radius = 2 → 
  (angle * radius^2) / 2 = (5/3) * π := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l3777_377740


namespace NUMINAMATH_CALUDE_minimum_days_to_plant_trees_l3777_377767

def tree_sequence (n : ℕ) : ℕ := 2^(n + 1) - 2

theorem minimum_days_to_plant_trees :
  ∃ (n : ℕ), n > 0 ∧ tree_sequence n ≥ 100 ∧ ∀ m : ℕ, m > 0 → m < n → tree_sequence m < 100 :=
by sorry

end NUMINAMATH_CALUDE_minimum_days_to_plant_trees_l3777_377767


namespace NUMINAMATH_CALUDE_fraction_irreducible_fraction_simplification_l3777_377710

-- Part (a)
theorem fraction_irreducible (a : ℤ) : 
  Int.gcd (a^3 + 2*a) (a^4 + 3*a^2 + 1) = 1 := by sorry

-- Part (b)
theorem fraction_simplification (n : ℤ) : 
  Int.gcd (5*n + 6) (8*n + 7) = 1 ∨ Int.gcd (5*n + 6) (8*n + 7) = 13 := by sorry

end NUMINAMATH_CALUDE_fraction_irreducible_fraction_simplification_l3777_377710


namespace NUMINAMATH_CALUDE_cases_in_1995_l3777_377795

/-- Calculates the number of cases in a given year assuming a linear decrease --/
def casesInYear (initialYear initialCases finalYear finalCases targetYear : ℕ) : ℕ :=
  let totalYears := finalYear - initialYear
  let totalDecrease := initialCases - finalCases
  let targetYearsSinceStart := targetYear - initialYear
  let decrease := (targetYearsSinceStart * totalDecrease) / totalYears
  initialCases - decrease

/-- Theorem stating that the number of cases in 1995 is 263,125 --/
theorem cases_in_1995 : 
  casesInYear 1970 700000 2010 1000 1995 = 263125 := by
  sorry

#eval casesInYear 1970 700000 2010 1000 1995

end NUMINAMATH_CALUDE_cases_in_1995_l3777_377795


namespace NUMINAMATH_CALUDE_expression_evaluation_l3777_377722

theorem expression_evaluation (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h : x = z / y) :
  (x - z / x) * (y + 1 / (z * y)) = (x^4 - z^3 + x^2 * (z^2 - z)) / (z * x^2) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3777_377722


namespace NUMINAMATH_CALUDE_garage_wheel_count_l3777_377776

/-- Calculates the total number of wheels in a garage given the quantities of various vehicles --/
def total_wheels (bicycles cars tricycles single_axle_trailers double_axle_trailers eighteen_wheelers : ℕ) : ℕ :=
  bicycles * 2 + cars * 4 + tricycles * 3 + single_axle_trailers * 2 + double_axle_trailers * 4 + eighteen_wheelers * 18

/-- Proves that the total number of wheels in the garage is 97 --/
theorem garage_wheel_count :
  total_wheels 5 12 3 2 2 1 = 97 := by
  sorry

end NUMINAMATH_CALUDE_garage_wheel_count_l3777_377776


namespace NUMINAMATH_CALUDE_correct_num_cars_l3777_377711

/-- Represents the number of cars taken on the hike -/
def num_cars : ℕ := 3

/-- Represents the number of taxis taken on the hike -/
def num_taxis : ℕ := 6

/-- Represents the number of vans taken on the hike -/
def num_vans : ℕ := 2

/-- Represents the number of people in each car -/
def people_per_car : ℕ := 4

/-- Represents the number of people in each taxi -/
def people_per_taxi : ℕ := 6

/-- Represents the number of people in each van -/
def people_per_van : ℕ := 5

/-- Represents the total number of people on the hike -/
def total_people : ℕ := 58

/-- Theorem stating that the number of cars is correct given the conditions -/
theorem correct_num_cars :
  num_cars * people_per_car +
  num_taxis * people_per_taxi +
  num_vans * people_per_van = total_people :=
by sorry

end NUMINAMATH_CALUDE_correct_num_cars_l3777_377711


namespace NUMINAMATH_CALUDE_primitive_triples_theorem_l3777_377709

/-- A triple of positive integers (a, b, c) is primitive if they have no common prime factors -/
def isPrimitive (a b c : ℕ+) : Prop :=
  ∀ p : ℕ, Nat.Prime p → ¬(p ∣ a.val ∧ p ∣ b.val ∧ p ∣ c.val)

/-- Each number in the triple divides the sum of the other two -/
def eachDividesSumOfOthers (a b c : ℕ+) : Prop :=
  a ∣ (b + c) ∧ b ∣ (a + c) ∧ c ∣ (a + b)

/-- The main theorem -/
theorem primitive_triples_theorem :
  ∀ a b c : ℕ+, a ≤ b → b ≤ c →
  isPrimitive a b c → eachDividesSumOfOthers a b c →
  (a, b, c) = (1, 1, 1) ∨ (a, b, c) = (1, 1, 2) ∨ (a, b, c) = (1, 2, 3) :=
sorry

end NUMINAMATH_CALUDE_primitive_triples_theorem_l3777_377709


namespace NUMINAMATH_CALUDE_male_average_score_l3777_377757

theorem male_average_score (total_students : ℕ) (male_students : ℕ) (female_students : ℕ)
  (overall_average : ℚ) (female_average : ℚ) :
  total_students = male_students + female_students →
  total_students = 28 →
  male_students = 8 →
  female_students = 20 →
  overall_average = 90 →
  female_average = 92 →
  (total_students : ℚ) * overall_average = 
    (male_students : ℚ) * ((total_students : ℚ) * overall_average - (female_students : ℚ) * female_average) / (male_students : ℚ) + 
    (female_students : ℚ) * female_average →
  ((total_students : ℚ) * overall_average - (female_students : ℚ) * female_average) / (male_students : ℚ) = 85 :=
by sorry

end NUMINAMATH_CALUDE_male_average_score_l3777_377757


namespace NUMINAMATH_CALUDE_bakery_doughnuts_l3777_377730

/-- The number of doughnuts in each box -/
def doughnuts_per_box : ℕ := 10

/-- The number of boxes sold -/
def boxes_sold : ℕ := 27

/-- The number of doughnuts given away -/
def doughnuts_given_away : ℕ := 30

/-- The total number of doughnuts made by the bakery -/
def total_doughnuts : ℕ := doughnuts_per_box * boxes_sold + doughnuts_given_away

theorem bakery_doughnuts : total_doughnuts = 300 := by
  sorry

end NUMINAMATH_CALUDE_bakery_doughnuts_l3777_377730


namespace NUMINAMATH_CALUDE_cupcake_distribution_l3777_377769

theorem cupcake_distribution (total_cupcakes : ℕ) (num_children : ℕ) 
  (h1 : total_cupcakes = 96) 
  (h2 : num_children = 8) 
  (h3 : total_cupcakes % num_children = 0) : 
  total_cupcakes / num_children = 12 := by
  sorry

#check cupcake_distribution

end NUMINAMATH_CALUDE_cupcake_distribution_l3777_377769


namespace NUMINAMATH_CALUDE_combined_age_when_mike_is_24_l3777_377750

/-- Calculates the combined age of Mike, Barbara, Tom, and Peter when Mike is 24 years old -/
def combinedAgeWhenMikeIs24 (mikesInitialAge barbarasInitialAge tomsInitialAge petersInitialAge : ℕ) : ℕ :=
  let ageIncrease := 24 - mikesInitialAge
  24 + (barbarasInitialAge + ageIncrease) + (tomsInitialAge + ageIncrease) + (petersInitialAge + ageIncrease)

/-- Theorem stating the combined age of Mike, Barbara, Tom, and Peter when Mike is 24 years old -/
theorem combined_age_when_mike_is_24 :
  ∀ (mikesInitialAge : ℕ),
    mikesInitialAge = 16 →
    ∀ (barbarasInitialAge : ℕ),
      barbarasInitialAge = mikesInitialAge / 2 →
      ∀ (tomsInitialAge : ℕ),
        tomsInitialAge = barbarasInitialAge + 4 →
        ∀ (petersInitialAge : ℕ),
          petersInitialAge = 2 * tomsInitialAge →
          combinedAgeWhenMikeIs24 mikesInitialAge barbarasInitialAge tomsInitialAge petersInitialAge = 92 :=
by
  sorry


end NUMINAMATH_CALUDE_combined_age_when_mike_is_24_l3777_377750


namespace NUMINAMATH_CALUDE_sum_series_eq_factorial_minus_one_l3777_377702

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def sum_series (n : ℕ) : ℕ := 
  Finset.sum (Finset.range (n + 1)) (λ k => k * factorial k)

theorem sum_series_eq_factorial_minus_one (n : ℕ) : 
  sum_series n = factorial (n + 1) - 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_series_eq_factorial_minus_one_l3777_377702


namespace NUMINAMATH_CALUDE_tens_digit_of_13_pow_2017_l3777_377754

theorem tens_digit_of_13_pow_2017 : ∃ n : ℕ, 13^2017 ≡ 30 + n [ZMOD 100] :=
by sorry

end NUMINAMATH_CALUDE_tens_digit_of_13_pow_2017_l3777_377754


namespace NUMINAMATH_CALUDE_identify_coefficients_l3777_377727

-- Define the coefficients of a quadratic equation ax^2 + bx + c = 0
structure QuadraticCoefficients where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define our specific quadratic equation 2x^2 - x - 5 = 0
def our_quadratic : QuadraticCoefficients := ⟨2, -1, -5⟩

-- Theorem to prove
theorem identify_coefficients :
  our_quadratic.a = 2 ∧ our_quadratic.b = -1 := by
  sorry

end NUMINAMATH_CALUDE_identify_coefficients_l3777_377727


namespace NUMINAMATH_CALUDE_second_point_x_coordinate_l3777_377766

/-- Given two points on a line, prove that the x-coordinate of the second point is m + 5 -/
theorem second_point_x_coordinate 
  (m n : ℝ) 
  (h1 : m = 2 * n + 5) 
  (h2 : m + 5 = 2 * (n + 2.5) + 5) : 
  m + 5 = m + 5 := by
  sorry

end NUMINAMATH_CALUDE_second_point_x_coordinate_l3777_377766


namespace NUMINAMATH_CALUDE_gcd_612_468_is_36_l3777_377763

theorem gcd_612_468_is_36 : Nat.gcd 612 468 = 36 := by
  sorry

end NUMINAMATH_CALUDE_gcd_612_468_is_36_l3777_377763


namespace NUMINAMATH_CALUDE_square_area_relationship_l3777_377771

/-- Given a square with side length a+b, prove that the relationship between 
    the areas of three squares formed within it can be expressed as a^2 + b^2 = c^2. -/
theorem square_area_relationship (a b c : ℝ) : 
  (∃ (total_area : ℝ), total_area = (a + b)^2) → 
  a^2 + b^2 = c^2 := by
  sorry

end NUMINAMATH_CALUDE_square_area_relationship_l3777_377771


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l3777_377774

-- First expression
theorem simplify_expression_1 (x : ℝ) : 2 * x - 3 * (x - 1) = 3 - x := by sorry

-- Second expression
theorem simplify_expression_2 (a b : ℝ) : 
  6 * (a * b^2 - a^2 * b) - 2 * (3 * a^2 * b + a * b^2) = 4 * a * b^2 - 12 * a^2 * b := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l3777_377774


namespace NUMINAMATH_CALUDE_donna_truck_weight_l3777_377738

/-- The weight of Donna's fully loaded truck -/
def truck_weight : ℕ :=
  let empty_truck_weight : ℕ := 12000
  let soda_crate_weight : ℕ := 50
  let soda_crate_count : ℕ := 20
  let dryer_weight : ℕ := 3000
  let dryer_count : ℕ := 3
  let soda_weight : ℕ := soda_crate_weight * soda_crate_count
  let produce_weight : ℕ := 2 * soda_weight
  let dryers_weight : ℕ := dryer_weight * dryer_count
  empty_truck_weight + soda_weight + produce_weight + dryers_weight

/-- Theorem stating that Donna's fully loaded truck weighs 24,000 pounds -/
theorem donna_truck_weight : truck_weight = 24000 := by
  sorry

end NUMINAMATH_CALUDE_donna_truck_weight_l3777_377738


namespace NUMINAMATH_CALUDE_max_value_theorem_max_value_achievable_l3777_377761

theorem max_value_theorem (x y : ℝ) :
  (3 * x + 4 * y + 5) / Real.sqrt (3 * x^2 + 4 * y^2 + 6) ≤ Real.sqrt 50 :=
by sorry

theorem max_value_achievable :
  ∃ x y : ℝ, (3 * x + 4 * y + 5) / Real.sqrt (3 * x^2 + 4 * y^2 + 6) = Real.sqrt 50 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_max_value_achievable_l3777_377761


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_l3777_377704

/-- An ellipse with equation x²/4 + y²/2 = 1 -/
def Ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 2 = 1

/-- A line with equation y = k(x-1) -/
def Line (k x y : ℝ) : Prop :=
  y = k * (x - 1)

/-- The area of a triangle given three points -/
noncomputable def TriangleArea (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

/-- The main theorem -/
theorem ellipse_line_intersection (k : ℝ) :
  (∃ x1 y1 x2 y2,
    Ellipse x1 y1 ∧ Ellipse x2 y2 ∧
    Line k x1 y1 ∧ Line k x2 y2 ∧
    (x1 ≠ x2 ∨ y1 ≠ y2) ∧
    TriangleArea 2 0 x1 y1 x2 y2 = Real.sqrt 10 / 3) ↔
  k = 1 ∨ k = -1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_l3777_377704


namespace NUMINAMATH_CALUDE_adams_final_balance_l3777_377703

/-- Calculates the final balance after a series of transactions --/
def final_balance (initial : ℚ) (spent : List ℚ) (received : List ℚ) : ℚ :=
  initial - spent.sum + received.sum

/-- Theorem: Adam's final balance is $10.75 --/
theorem adams_final_balance :
  let initial : ℚ := 5
  let spent : List ℚ := [2, 1.5, 0.75]
  let received : List ℚ := [3, 2, 5]
  final_balance initial spent received = 10.75 := by
  sorry

end NUMINAMATH_CALUDE_adams_final_balance_l3777_377703


namespace NUMINAMATH_CALUDE_identity_function_only_solution_l3777_377768

theorem identity_function_only_solution :
  ∀ f : ℝ → ℝ, (∀ x y : ℝ, f (x + y) = x + f (f y)) → (∀ x : ℝ, f x = x) :=
by sorry

end NUMINAMATH_CALUDE_identity_function_only_solution_l3777_377768


namespace NUMINAMATH_CALUDE_tan_seven_pi_fourth_l3777_377765

theorem tan_seven_pi_fourth : Real.tan (7 * π / 4) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_seven_pi_fourth_l3777_377765


namespace NUMINAMATH_CALUDE_cross_country_winning_scores_l3777_377749

/-- The number of teams in the cross-country meet -/
def num_teams : ℕ := 2

/-- The number of runners per team -/
def runners_per_team : ℕ := 6

/-- The total number of runners -/
def total_runners : ℕ := num_teams * runners_per_team

/-- The sum of positions from 1 to n -/
def sum_positions (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The total sum of all positions -/
def total_sum : ℕ := sum_positions total_runners

/-- A winning score is less than half of the total sum -/
def is_winning_score (score : ℕ) : Prop := score < total_sum / 2

/-- The minimum possible score for a team -/
def min_score : ℕ := sum_positions runners_per_team

/-- The maximum possible winning score -/
def max_winning_score : ℕ := total_sum / 2 - 1

/-- The number of different possible winning scores -/
def num_winning_scores : ℕ := max_winning_score - min_score + 1

theorem cross_country_winning_scores :
  num_winning_scores = 18 := by sorry

end NUMINAMATH_CALUDE_cross_country_winning_scores_l3777_377749


namespace NUMINAMATH_CALUDE_unique_solution_trigonometric_equation_l3777_377791

theorem unique_solution_trigonometric_equation :
  ∃! (n : ℕ), n > 0 ∧ Real.sin (π / (3 * n)) + Real.cos (π / (3 * n)) = Real.sqrt (2 * n) / 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_trigonometric_equation_l3777_377791


namespace NUMINAMATH_CALUDE_system_solution_is_correct_l3777_377745

/-- The solution set of the system of inequalities {2x + 3 ≤ x + 2, (x + 1) / 3 > x - 1} -/
def solution_set : Set ℝ := {x : ℝ | x ≤ -1}

/-- The first inequality of the system -/
def inequality1 (x : ℝ) : Prop := 2 * x + 3 ≤ x + 2

/-- The second inequality of the system -/
def inequality2 (x : ℝ) : Prop := (x + 1) / 3 > x - 1

theorem system_solution_is_correct :
  ∀ x : ℝ, x ∈ solution_set ↔ (inequality1 x ∧ inequality2 x) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_is_correct_l3777_377745


namespace NUMINAMATH_CALUDE_solution_range_l3777_377753

theorem solution_range (x y z : ℝ) 
  (sum_eq : x + y + z = 5) 
  (prod_eq : x*y + y*z + z*x = 3) : 
  -1 ≤ x ∧ x ≤ 13/3 ∧ 
  -1 ≤ y ∧ y ≤ 13/3 ∧ 
  -1 ≤ z ∧ z ≤ 13/3 := by
sorry

end NUMINAMATH_CALUDE_solution_range_l3777_377753


namespace NUMINAMATH_CALUDE_willson_work_hours_l3777_377775

theorem willson_work_hours : 
  let monday : ℚ := 3/4
  let tuesday : ℚ := 1/2
  let wednesday : ℚ := 2/3
  let thursday : ℚ := 5/6
  let friday : ℚ := 75/60
  monday + tuesday + wednesday + thursday + friday = 4 := by
sorry

end NUMINAMATH_CALUDE_willson_work_hours_l3777_377775


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l3777_377744

theorem completing_square_equivalence :
  ∀ x : ℝ, 3 * x^2 + 4 * x + 1 = 0 ↔ (x + 2/3)^2 = 1/9 :=
by sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l3777_377744


namespace NUMINAMATH_CALUDE_solve_equation_l3777_377715

theorem solve_equation (m : ℝ) (h : m + (m + 2) + (m + 4) = 24) : m = 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3777_377715


namespace NUMINAMATH_CALUDE_min_area_intersecting_hyperbolas_l3777_377784

/-- A set in ℝ² is convex if for any two points in the set, 
    the line segment connecting them is also in the set -/
def is_convex (S : Set (ℝ × ℝ)) : Prop :=
  ∀ (p q : ℝ × ℝ), p ∈ S → q ∈ S → 
    ∀ (t : ℝ), 0 ≤ t ∧ t ≤ 1 → (1 - t) • p + t • q ∈ S

/-- A set intersects a hyperbola if there exists a point in the set 
    that satisfies the hyperbola equation -/
def intersects_hyperbola (S : Set (ℝ × ℝ)) (k : ℝ) : Prop :=
  ∃ (x y : ℝ), (x, y) ∈ S ∧ x * y = k

/-- The area of a set in ℝ² -/
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

/-- The main theorem: The minimum area of a convex set intersecting 
    both branches of xy = 1 and xy = -1 is 4 -/
theorem min_area_intersecting_hyperbolas :
  (∃ (S : Set (ℝ × ℝ)), 
    is_convex S ∧ 
    intersects_hyperbola S 1 ∧ 
    intersects_hyperbola S (-1)) →
  (∀ (S : Set (ℝ × ℝ)), 
    is_convex S → 
    intersects_hyperbola S 1 → 
    intersects_hyperbola S (-1) → 
    area S ≥ 4) ∧
  (∃ (S : Set (ℝ × ℝ)), 
    is_convex S ∧ 
    intersects_hyperbola S 1 ∧ 
    intersects_hyperbola S (-1) ∧ 
    area S = 4) :=
by sorry

end NUMINAMATH_CALUDE_min_area_intersecting_hyperbolas_l3777_377784


namespace NUMINAMATH_CALUDE_dot_product_sum_l3777_377777

/-- Given vectors in ℝ², prove that the dot product of (a + b) and c equals 6 -/
theorem dot_product_sum (a b c : ℝ × ℝ) (ha : a = (1, -2)) (hb : b = (3, 4)) (hc : c = (2, -1)) :
  ((a.1 + b.1, a.2 + b.2) • c) = 6 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_sum_l3777_377777


namespace NUMINAMATH_CALUDE_tank_capacity_l3777_377700

/-- Represents a tank with a leak and an inlet pipe. -/
structure Tank where
  capacity : ℝ
  leak_empty_time : ℝ
  inlet_rate : ℝ
  combined_empty_time : ℝ

/-- Theorem stating the relationship between tank properties and its capacity. -/
theorem tank_capacity (t : Tank) 
  (h1 : t.leak_empty_time = 6)
  (h2 : t.inlet_rate = 3 * 60)  -- 3 liters per minute converted to per hour
  (h3 : t.combined_empty_time = 8) :
  t.capacity = 4320 / 7 := by
  sorry

#check tank_capacity

end NUMINAMATH_CALUDE_tank_capacity_l3777_377700


namespace NUMINAMATH_CALUDE_hyperbola_focus_asymptote_distance_l3777_377741

/-- A hyperbola with equation x^2 - y^2/b^2 = 1 where b > 0 -/
structure Hyperbola where
  b : ℝ
  h_pos : b > 0

/-- The asymptote of a hyperbola -/
structure Asymptote where
  slope : ℝ
  y_intercept : ℝ

/-- The focus of a hyperbola -/
structure Focus where
  x : ℝ
  y : ℝ

/-- Distance between a point and a line -/
def distance_point_to_line (p : ℝ × ℝ) (l : Asymptote) : ℝ :=
  sorry

/-- Theorem: If one asymptote of the hyperbola x^2 - y^2/b^2 = 1 (b > 0) is y = 2x, 
    then the distance from the focus to this asymptote is 2 -/
theorem hyperbola_focus_asymptote_distance 
  (h : Hyperbola) 
  (a : Asymptote) 
  (f : Focus) 
  (h_asymptote : a.slope = 2 ∧ a.y_intercept = 0) : 
  distance_point_to_line (f.x, f.y) a = 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_focus_asymptote_distance_l3777_377741


namespace NUMINAMATH_CALUDE_midpoint_quadrilateral_perpendicular_diagonals_l3777_377752

/-- Represents a point on a circle --/
structure CirclePoint where
  angle : Real

/-- Represents a quadrilateral formed by midpoints of arcs on a circle --/
structure MidpointQuadrilateral where
  p1 : CirclePoint
  p2 : CirclePoint
  p3 : CirclePoint
  p4 : CirclePoint

/-- Calculates the angle between two diagonals of a quadrilateral --/
def diagonalAngle (q : MidpointQuadrilateral) : Real :=
  -- Implementation details omitted
  sorry

/-- States that the diagonals of a quadrilateral formed by midpoints of four arcs on a circle are perpendicular --/
theorem midpoint_quadrilateral_perpendicular_diagonals 
  (c : CirclePoint → CirclePoint → CirclePoint → CirclePoint → MidpointQuadrilateral) :
  ∀ (p1 p2 p3 p4 : CirclePoint), 
    diagonalAngle (c p1 p2 p3 p4) = Real.pi / 2 := by
  sorry

#check midpoint_quadrilateral_perpendicular_diagonals

end NUMINAMATH_CALUDE_midpoint_quadrilateral_perpendicular_diagonals_l3777_377752


namespace NUMINAMATH_CALUDE_min_value_expression_l3777_377728

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  ∃ m : ℝ, m = -2031948.5 ∧ 
  ∀ x y : ℝ, x > 0 → y > 0 → 
    (x + 1/y) * (x + 1/y - 2023) + (y + 1/x) * (y + 1/x - 2023) ≥ m ∧
    (a + 1/b) * (a + 1/b - 2023) + (b + 1/a) * (b + 1/a - 2023) = m :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3777_377728


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l3777_377758

/-- The area of a square with diagonal length 12√2 cm is 144 cm² -/
theorem square_area_from_diagonal : ∀ s : ℝ,
  s > 0 →
  s * s * 2 = (12 * Real.sqrt 2) ^ 2 →
  s * s = 144 :=
by sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l3777_377758


namespace NUMINAMATH_CALUDE_rice_yields_variance_l3777_377729

def rice_yields : List ℝ := [9.8, 9.9, 10.1, 10, 10.2]

theorem rice_yields_variance : 
  let n : ℕ := rice_yields.length
  let mean : ℝ := rice_yields.sum / n
  let variance : ℝ := (rice_yields.map (fun x => (x - mean)^2)).sum / n
  variance = 0.02 := by sorry

end NUMINAMATH_CALUDE_rice_yields_variance_l3777_377729


namespace NUMINAMATH_CALUDE_disease_test_probability_l3777_377714

theorem disease_test_probability (p_disease : ℝ) (p_positive_given_disease : ℝ) (p_positive_given_no_disease : ℝ) :
  p_disease = 1 / 300 →
  p_positive_given_disease = 1 →
  p_positive_given_no_disease = 0.03 →
  (p_disease * p_positive_given_disease) / 
  (p_disease * p_positive_given_disease + (1 - p_disease) * p_positive_given_no_disease) = 100 / 997 := by
  sorry

end NUMINAMATH_CALUDE_disease_test_probability_l3777_377714


namespace NUMINAMATH_CALUDE_inequality_preservation_l3777_377764

theorem inequality_preservation (a b c : ℝ) (h : a > b) : 
  a / (c^2 + 1) > b / (c^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l3777_377764


namespace NUMINAMATH_CALUDE_popsicle_problem_l3777_377780

/-- Popsicle Making Problem -/
theorem popsicle_problem (total_money : ℕ) (mold_cost : ℕ) (stick_pack_cost : ℕ) 
  (juice_cost : ℕ) (total_sticks : ℕ) (remaining_sticks : ℕ) :
  total_money = 10 →
  mold_cost = 3 →
  stick_pack_cost = 1 →
  juice_cost = 2 →
  total_sticks = 100 →
  remaining_sticks = 40 →
  (total_money - mold_cost - stick_pack_cost) / juice_cost * 
    ((total_sticks - remaining_sticks) / ((total_money - mold_cost - stick_pack_cost) / juice_cost)) = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_popsicle_problem_l3777_377780


namespace NUMINAMATH_CALUDE_hundredth_odd_and_following_even_l3777_377779

theorem hundredth_odd_and_following_even :
  (∃ n : ℕ, n = 100 ∧ 2 * n - 1 = 199) ∧
  (∃ m : ℕ, m = 200 ∧ m = 199 + 1 ∧ Even m) := by
  sorry

end NUMINAMATH_CALUDE_hundredth_odd_and_following_even_l3777_377779


namespace NUMINAMATH_CALUDE_supremum_of_expression_is_zero_l3777_377782

open Real

theorem supremum_of_expression_is_zero :
  ∀ ε > 0, ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧
  x * y * z * (x + y + z) / (x + y + z)^3 < ε :=
by sorry

end NUMINAMATH_CALUDE_supremum_of_expression_is_zero_l3777_377782


namespace NUMINAMATH_CALUDE_path_length_is_894_l3777_377735

/-- The length of the path with fencing and a bridge. -/
def path_length (pole_spacing : ℕ) (bridge_length : ℕ) (total_poles : ℕ) : ℕ :=
  let poles_one_side := total_poles / 2
  let intervals := poles_one_side - 1
  intervals * pole_spacing + bridge_length

/-- Theorem stating the length of the path given the conditions. -/
theorem path_length_is_894 :
  path_length 6 42 286 = 894 := by
  sorry

end NUMINAMATH_CALUDE_path_length_is_894_l3777_377735


namespace NUMINAMATH_CALUDE_mark_amy_age_difference_mark_amy_age_difference_proof_l3777_377778

theorem mark_amy_age_difference : ℕ → Prop :=
  fun age_difference =>
    ∃ (mark_current_age amy_current_age : ℕ),
      amy_current_age = 15 ∧
      mark_current_age + 5 = 27 ∧
      mark_current_age - amy_current_age = age_difference ∧
      age_difference = 7

-- The proof is omitted
theorem mark_amy_age_difference_proof : mark_amy_age_difference 7 := by
  sorry

end NUMINAMATH_CALUDE_mark_amy_age_difference_mark_amy_age_difference_proof_l3777_377778


namespace NUMINAMATH_CALUDE_sin_square_sum_range_l3777_377736

theorem sin_square_sum_range (α β : ℝ) (h : 3 * (Real.sin α)^2 - 2 * Real.sin α + 2 * (Real.sin β)^2 = 0) :
  ∃ (x : ℝ), x = (Real.sin α)^2 + (Real.sin β)^2 ∧ 0 ≤ x ∧ x ≤ 4/9 :=
sorry

end NUMINAMATH_CALUDE_sin_square_sum_range_l3777_377736


namespace NUMINAMATH_CALUDE_eighth_group_sample_digit_l3777_377756

/-- Represents the systematic sampling method described in the problem -/
def systematicSample (t : ℕ) (k : ℕ) : ℕ :=
  (t + k) % 10

/-- The theorem to prove -/
theorem eighth_group_sample_digit (t : ℕ) (h : t = 7) : systematicSample t 8 = 5 := by
  sorry

end NUMINAMATH_CALUDE_eighth_group_sample_digit_l3777_377756


namespace NUMINAMATH_CALUDE_range_of_a_l3777_377720

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (x^2 + (a+2)*x + 1) * ((3-2*a)*x^2 + 5*x + (3-2*a)) ≥ 0) →
  a ∈ Set.Icc (-4 : ℝ) 0 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l3777_377720


namespace NUMINAMATH_CALUDE_x_gt_neg_two_necessary_not_sufficient_for_x_sq_lt_four_l3777_377708

theorem x_gt_neg_two_necessary_not_sufficient_for_x_sq_lt_four :
  (∀ x : ℝ, x^2 < 4 → x > -2) ∧
  (∃ x : ℝ, x > -2 ∧ x^2 ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_x_gt_neg_two_necessary_not_sufficient_for_x_sq_lt_four_l3777_377708


namespace NUMINAMATH_CALUDE_nina_shells_to_liam_l3777_377790

theorem nina_shells_to_liam (oliver liam nina : ℕ) 
  (h1 : liam = 3 * oliver) 
  (h2 : nina = 4 * liam) 
  (h3 : oliver > 0) : 
  (nina - (oliver + liam + nina) / 3) / nina = 7 / 36 := by
  sorry

end NUMINAMATH_CALUDE_nina_shells_to_liam_l3777_377790


namespace NUMINAMATH_CALUDE_plastic_rings_total_weight_l3777_377747

theorem plastic_rings_total_weight 
  (orange : ℝ) (purple : ℝ) (white : ℝ) (blue : ℝ) (red : ℝ) (green : ℝ)
  (h_orange : orange = 0.08)
  (h_purple : purple = 0.33)
  (h_white : white = 0.42)
  (h_blue : blue = 0.59)
  (h_red : red = 0.24)
  (h_green : green = 0.16) :
  orange + purple + white + blue + red + green = 1.82 := by
sorry

end NUMINAMATH_CALUDE_plastic_rings_total_weight_l3777_377747


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l3777_377737

theorem cubic_equation_solution :
  ∀ x : ℝ, x^3 - 4*x = 0 ↔ x = 0 ∨ x = -2 ∨ x = 2 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l3777_377737


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3777_377706

/-- The polynomial P(x) = x + x^3 + x^9 + x^27 + x^81 + x^243 -/
def P (x : ℝ) : ℝ := x + x^3 + x^9 + x^27 + x^81 + x^243

theorem polynomial_division_remainder :
  (∃ Q₁ : ℝ → ℝ, P = fun x ↦ (x - 1) * Q₁ x + 6) ∧
  (∃ Q₂ : ℝ → ℝ, P = fun x ↦ (x^2 - 1) * Q₂ x + 6*x) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3777_377706


namespace NUMINAMATH_CALUDE_min_value_problem_l3777_377794

theorem min_value_problem (a b m n : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_m : 0 < m) (h_pos_n : 0 < n)
  (h_sum : a + b = 1) (h_prod : m * n = 2) :
  (a * m + b * n) * (b * m + a * n) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l3777_377794


namespace NUMINAMATH_CALUDE_age_sum_problem_l3777_377785

theorem age_sum_problem (a b c : ℕ+) : 
  a = b ∧ a > c ∧ a * b * c = 162 → a + b + c = 20 := by
  sorry

end NUMINAMATH_CALUDE_age_sum_problem_l3777_377785


namespace NUMINAMATH_CALUDE_train_length_l3777_377717

/-- Given a train that crosses a tunnel and a platform, calculate its length -/
theorem train_length (tunnel_length platform_length : ℝ) 
                     (tunnel_time platform_time : ℝ) 
                     (h1 : tunnel_length = 1200)
                     (h2 : platform_length = 180)
                     (h3 : tunnel_time = 45)
                     (h4 : platform_time = 15) : 
  ∃ (train_length : ℝ), 
    (train_length + tunnel_length) / tunnel_time = 
    (train_length + platform_length) / platform_time ∧ 
    train_length = 330 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3777_377717


namespace NUMINAMATH_CALUDE_max_p_value_l3777_377788

/-- The maximum value of p for two rectangular boxes with given conditions -/
theorem max_p_value (m n p : ℕ+) (h1 : m ≤ n) (h2 : n ≤ p)
  (h3 : 2 * (m * n * p) = (m + 2) * (n + 2) * (p + 2)) : 
  p ≤ 130 := by
sorry

end NUMINAMATH_CALUDE_max_p_value_l3777_377788


namespace NUMINAMATH_CALUDE_beach_count_theorem_l3777_377705

/-- The total count of oysters and crabs over two days -/
def total_count (initial_oysters initial_crabs : ℕ) : ℕ :=
  initial_oysters + (initial_oysters / 2) +
  initial_crabs + (initial_crabs * 2 / 3)

/-- Theorem stating the total count for the given initial numbers -/
theorem beach_count_theorem :
  total_count 50 72 = 195 := by
  sorry

end NUMINAMATH_CALUDE_beach_count_theorem_l3777_377705


namespace NUMINAMATH_CALUDE_equation_system_solutions_equation_system_unique_solutions_l3777_377789

/-- The system of equations has four solutions: (1, 1, 1) and three cyclic permutations of another triple -/
theorem equation_system_solutions :
  ∃ (a b c : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧
  (∀ (x y z : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧
    x^2 - y = (z - 1)^2 ∧
    y^2 - z = (x - 1)^2 ∧
    z^2 - x = (y - 1)^2 →
    ((x = 1 ∧ y = 1 ∧ z = 1) ∨
     (x = a ∧ y = b ∧ z = c) ∨
     (x = b ∧ y = c ∧ z = a) ∨
     (x = c ∧ y = a ∧ z = b))) :=
by sorry

/-- The system of equations has exactly four solutions -/
theorem equation_system_unique_solutions :
  ∃! (s : Finset (ℝ × ℝ × ℝ)), s.card = 4 ∧
  (∀ (x y z : ℝ), (x, y, z) ∈ s ↔
    x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧
    x^2 - y = (z - 1)^2 ∧
    y^2 - z = (x - 1)^2 ∧
    z^2 - x = (y - 1)^2) :=
by sorry

end NUMINAMATH_CALUDE_equation_system_solutions_equation_system_unique_solutions_l3777_377789


namespace NUMINAMATH_CALUDE_sum_smallest_largest_angle_l3777_377793

/-- A hexagon with angles in arithmetic progression -/
structure ArithmeticHexagon where
  /-- The smallest angle of the hexagon -/
  a : ℝ
  /-- The common difference between consecutive angles -/
  n : ℝ
  /-- The angles are non-negative -/
  a_nonneg : 0 ≤ a
  n_nonneg : 0 ≤ n
  /-- The sum of all angles in a hexagon is 720° -/
  sum_angles : a + (a + n) + (a + 2*n) + (a + 3*n) + (a + 4*n) + (a + 5*n) = 720

/-- The sum of the smallest and largest angles in an arithmetic hexagon is 240° -/
theorem sum_smallest_largest_angle (h : ArithmeticHexagon) : h.a + (h.a + 5*h.n) = 240 := by
  sorry


end NUMINAMATH_CALUDE_sum_smallest_largest_angle_l3777_377793


namespace NUMINAMATH_CALUDE_triangle_inequality_l3777_377719

theorem triangle_inequality (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  a + b > c ∧ b + c > a ∧ a + c > b →
  ¬(a = 3 ∧ b = 4 ∧ c = 7) := by
  sorry

#check triangle_inequality

end NUMINAMATH_CALUDE_triangle_inequality_l3777_377719


namespace NUMINAMATH_CALUDE_real_part_of_one_plus_i_squared_is_zero_l3777_377716

theorem real_part_of_one_plus_i_squared_is_zero :
  Complex.re ((1 : ℂ) + Complex.I) ^ 2 = 0 := by sorry

end NUMINAMATH_CALUDE_real_part_of_one_plus_i_squared_is_zero_l3777_377716


namespace NUMINAMATH_CALUDE_smallest_solution_abs_equation_l3777_377773

theorem smallest_solution_abs_equation :
  let f := fun x : ℝ => x * |x| - (3 * x - 2)
  ∃ x₀ : ℝ, f x₀ = 0 ∧ ∀ x : ℝ, f x = 0 → x₀ ≤ x ∧ x₀ = (-3 - Real.sqrt 17) / 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_abs_equation_l3777_377773


namespace NUMINAMATH_CALUDE_solve_equations_l3777_377726

theorem solve_equations :
  (∃ x₁ x₂ : ℝ, (x₁ - 3)^2 + 2*x₁*(x₁ - 3) = 0 ∧ (x₂ - 3)^2 + 2*x₂*(x₂ - 3) = 0 ∧ x₁ = 3 ∧ x₂ = 1) ∧
  (∃ y₁ y₂ : ℝ, y₁^2 - 4*y₁ + 1 = 0 ∧ y₂^2 - 4*y₂ + 1 = 0 ∧ y₁ = 2 + Real.sqrt 3 ∧ y₂ = 2 - Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_solve_equations_l3777_377726


namespace NUMINAMATH_CALUDE_cube_sum_minus_triple_product_l3777_377712

theorem cube_sum_minus_triple_product (x y z : ℝ) 
  (h1 : x + y + z = 8) 
  (h2 : x*y + y*z + z*x = 20) : 
  x^3 + y^3 + z^3 - 3*x*y*z = 32 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_minus_triple_product_l3777_377712


namespace NUMINAMATH_CALUDE_average_difference_l3777_377759

theorem average_difference (x : ℝ) : 
  (20 + 40 + 60) / 3 = ((x + 80 + 15) / 3) + 5 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_average_difference_l3777_377759


namespace NUMINAMATH_CALUDE_fraction_enlargement_l3777_377799

theorem fraction_enlargement (x y : ℝ) (h : 3 * x - y ≠ 0) :
  (2 * (3 * x) * (3 * y)) / (3 * (3 * x) - 3 * y) = 3 * ((2 * x * y) / (3 * x - y)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_enlargement_l3777_377799


namespace NUMINAMATH_CALUDE_optimal_rental_plan_l3777_377772

/-- Represents the capacity and cost of different car types -/
structure CarType where
  capacity : ℕ
  cost : ℕ

/-- Represents a rental plan -/
structure RentalPlan where
  type_a : ℕ
  type_b : ℕ

/-- Calculates the total capacity of a rental plan -/
def total_capacity (plan : RentalPlan) (a : CarType) (b : CarType) : ℕ :=
  plan.type_a * a.capacity + plan.type_b * b.capacity

/-- Calculates the total cost of a rental plan -/
def total_cost (plan : RentalPlan) (a : CarType) (b : CarType) : ℕ :=
  plan.type_a * a.cost + plan.type_b * b.cost

/-- Checks if a rental plan is valid for the given total goods -/
def is_valid_plan (plan : RentalPlan) (a : CarType) (b : CarType) (total_goods : ℕ) : Prop :=
  total_capacity plan a b = total_goods

/-- Theorem: The optimal rental plan for transporting 27 tons of goods is 1 type A car and 6 type B cars, with a total cost of 820 yuan -/
theorem optimal_rental_plan :
  ∃ (a b : CarType) (optimal_plan : RentalPlan),
    -- Given conditions
    (2 * a.capacity + 3 * b.capacity = 18) ∧
    (a.capacity + 2 * b.capacity = 11) ∧
    (a.cost = 100) ∧
    (b.cost = 120) ∧
    -- Optimal plan
    (optimal_plan.type_a = 1) ∧
    (optimal_plan.type_b = 6) ∧
    -- Plan is valid
    (is_valid_plan optimal_plan a b 27) ∧
    -- Plan is optimal (minimum cost)
    (∀ (plan : RentalPlan),
      is_valid_plan plan a b 27 →
      total_cost optimal_plan a b ≤ total_cost plan a b) ∧
    -- Total cost is 820 yuan
    (total_cost optimal_plan a b = 820) :=
  sorry

end NUMINAMATH_CALUDE_optimal_rental_plan_l3777_377772


namespace NUMINAMATH_CALUDE_triangle_side_length_l3777_377701

theorem triangle_side_length (X Y Z : ℝ) : 
  -- Triangle XYZ with right angle at X
  X^2 + Y^2 = Z^2 →
  -- YZ = 20
  Z = 20 →
  -- tan Z = 3 cos Y
  (Real.tan Z) = 3 * (Real.cos Y) →
  -- XY = (40√2)/3
  Y = (40 * Real.sqrt 2) / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3777_377701


namespace NUMINAMATH_CALUDE_triangle_inequality_l3777_377781

theorem triangle_inequality (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) 
  (hSum : A + B + C = π) : 
  Real.tan (A/2)^2 + Real.tan (B/2)^2 + Real.tan (C/2)^2 + 
  8 * Real.sin (A/2) * Real.sin (B/2) * Real.sin (C/2) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3777_377781


namespace NUMINAMATH_CALUDE_line_equation_l3777_377786

/-- A line in the 2D plane represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point (x, y) lies on a given line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Check if a line has equal intercepts on x and y axes -/
def Line.has_equal_intercepts (l : Line) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ (-l.c / l.a = -l.c / l.b)

theorem line_equation (l : Line) :
  l.has_equal_intercepts ∧ l.contains 1 2 →
  (l.a = 2 ∧ l.b = -1 ∧ l.c = 0) ∨ (l.a = 1 ∧ l.b = 1 ∧ l.c = -3) :=
sorry

end NUMINAMATH_CALUDE_line_equation_l3777_377786


namespace NUMINAMATH_CALUDE_rectangle_side_sum_l3777_377724

theorem rectangle_side_sum (x y : ℝ) : 
  (2 * x + 4 = 10) → (8 * y - 2 = 10) → x + y = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_sum_l3777_377724


namespace NUMINAMATH_CALUDE_tangent_line_x_intercept_l3777_377725

-- Define the function f(x) = x³ - 2x² + 3x + 1
def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 3*x + 1

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 4*x + 3

theorem tangent_line_x_intercept :
  let slope : ℝ := f' 1
  let y_intercept : ℝ := f 1 - slope * 1
  let x_intercept : ℝ := -y_intercept / slope
  x_intercept = -1/2 := by sorry

end NUMINAMATH_CALUDE_tangent_line_x_intercept_l3777_377725


namespace NUMINAMATH_CALUDE_james_travel_distance_l3777_377743

/-- Calculates the total distance traveled during a road trip with multiple legs -/
def total_distance (speeds : List ℝ) (durations : List ℝ) : ℝ :=
  (List.zip speeds durations).map (fun (s, t) => s * t) |>.sum

/-- Theorem: James' total travel distance is 995.0 miles -/
theorem james_travel_distance : 
  let speeds : List ℝ := [80.0, 65.0, 75.0, 70.0]
  let durations : List ℝ := [2.0, 4.0, 3.0, 5.0]
  total_distance speeds durations = 995.0 := by
  sorry


end NUMINAMATH_CALUDE_james_travel_distance_l3777_377743


namespace NUMINAMATH_CALUDE_baseball_league_games_l3777_377751

/-- The number of games played in a baseball league -/
def total_games (n : ℕ) (games_per_matchup : ℕ) : ℕ :=
  n * (n - 1) * games_per_matchup / 2

/-- Theorem: In a 12-team league where each team plays 4 games with every other team, 
    the total number of games played is 264 -/
theorem baseball_league_games : 
  total_games 12 4 = 264 := by
  sorry

end NUMINAMATH_CALUDE_baseball_league_games_l3777_377751
