import Mathlib

namespace NUMINAMATH_CALUDE_mark_born_in_1978_l728_72894

/-- The year of the first AMC 8 -/
def first_amc8_year : ℕ := 1985

/-- The year Mark took the ninth AMC 8 -/
def ninth_amc8_year : ℕ := first_amc8_year + 8

/-- Mark's age when he took the ninth AMC 8 -/
def marks_age : ℕ := 15

/-- Mark's birth year -/
def marks_birth_year : ℕ := ninth_amc8_year - marks_age

theorem mark_born_in_1978 : marks_birth_year = 1978 := by sorry

end NUMINAMATH_CALUDE_mark_born_in_1978_l728_72894


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l728_72860

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 > 0) ↔ (∃ x : ℝ, x^2 ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l728_72860


namespace NUMINAMATH_CALUDE_g_eval_l728_72806

def g (x : ℝ) : ℝ := 2 * x^2 - 2 * x + 11

theorem g_eval : 3 * g 2 + 2 * g (-4) = 147 := by sorry

end NUMINAMATH_CALUDE_g_eval_l728_72806


namespace NUMINAMATH_CALUDE_bound_on_expression_l728_72844

theorem bound_on_expression (x y z : ℝ) (h : x^2 + y^2 + z^2 = 25) :
  -15 ≤ x - 2*y + 2*z ∧ x - 2*y + 2*z ≤ 15 := by
  sorry

end NUMINAMATH_CALUDE_bound_on_expression_l728_72844


namespace NUMINAMATH_CALUDE_complex_sum_problem_l728_72800

theorem complex_sum_problem (p q r s t u : ℝ) : 
  s = 5 →
  t = -p - r →
  (p + q * Complex.I) + (r + s * Complex.I) + (t + u * Complex.I) = -6 * Complex.I →
  u + q = -11 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_problem_l728_72800


namespace NUMINAMATH_CALUDE_rectangle_area_change_l728_72810

theorem rectangle_area_change (L W : ℝ) (h1 : L * W = 600) : 
  (0.8 * L) * (1.3 * W) = 624 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l728_72810


namespace NUMINAMATH_CALUDE_cardinality_of_S_l728_72811

/-- The number of elements in a set -/
def C (A : Set ℝ) : ℕ := sorry

/-- The operation * defined on sets -/
def star (A B : Set ℝ) : ℕ :=
  if C A ≥ C B then C A - C B else C B - C A

/-- The set B parameterized by a -/
def B (a : ℝ) : Set ℝ :=
  {x : ℝ | (x + a) * (x^3 + a*x^2 + 2*x) = 0}

/-- The set A -/
def A : Set ℝ := {1, 2}

/-- The set S of all possible values of a -/
def S : Set ℝ :=
  {a : ℝ | star A (B a) = 1 ∧ C A = 2}

theorem cardinality_of_S : C S = 3 := by sorry

end NUMINAMATH_CALUDE_cardinality_of_S_l728_72811


namespace NUMINAMATH_CALUDE_paco_ate_five_sweet_cookies_l728_72842

/-- Represents the number of cookies Paco had and ate -/
structure CookieCount where
  initial_sweet : Nat
  initial_salty : Nat
  eaten_salty : Nat
  sweet_salty_difference : Nat

/-- Calculates the number of sweet cookies Paco ate -/
def sweet_cookies_eaten (c : CookieCount) : Nat :=
  c.eaten_salty + c.sweet_salty_difference

/-- Theorem: Paco ate 5 sweet cookies -/
theorem paco_ate_five_sweet_cookies (c : CookieCount)
  (h1 : c.initial_sweet = 37)
  (h2 : c.initial_salty = 11)
  (h3 : c.eaten_salty = 2)
  (h4 : c.sweet_salty_difference = 3) :
  sweet_cookies_eaten c = 5 := by
  sorry

end NUMINAMATH_CALUDE_paco_ate_five_sweet_cookies_l728_72842


namespace NUMINAMATH_CALUDE_max_dot_product_l728_72841

/-- An ellipse with focus on the x-axis -/
structure Ellipse where
  /-- The b parameter in the ellipse equation x^2/4 + y^2/b^2 = 1 -/
  b : ℝ
  /-- The eccentricity of the ellipse -/
  e : ℝ
  /-- Condition that the eccentricity is 1/2 -/
  h_e : e = 1/2

/-- A point on the ellipse -/
structure PointOnEllipse (ε : Ellipse) where
  x : ℝ
  y : ℝ
  /-- The point satisfies the ellipse equation -/
  h_on_ellipse : x^2/4 + y^2/ε.b^2 = 1

/-- The left focus of the ellipse -/
def leftFocus (ε : Ellipse) : ℝ × ℝ := sorry

/-- The right vertex of the ellipse -/
def rightVertex (ε : Ellipse) : ℝ × ℝ := sorry

/-- The dot product of vectors PF and PA -/
def dotProduct (ε : Ellipse) (p : PointOnEllipse ε) : ℝ := sorry

/-- Theorem: The maximum value of the dot product of PF and PA is 4 -/
theorem max_dot_product (ε : Ellipse) :
  ∃ (max : ℝ), max = 4 ∧ ∀ (p : PointOnEllipse ε), dotProduct ε p ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_dot_product_l728_72841


namespace NUMINAMATH_CALUDE_total_vehicles_proof_l728_72874

/-- The number of vehicles involved in accidents last year -/
def accidents : ℕ := 2000

/-- The number of vehicles per 100 million that are involved in accidents -/
def accident_rate : ℕ := 100

/-- The total number of vehicles that traveled on the highway last year -/
def total_vehicles : ℕ := 2000000000

/-- Theorem stating that the total number of vehicles is correct given the accident rate and number of accidents -/
theorem total_vehicles_proof :
  accidents * (100000000 / accident_rate) = total_vehicles :=
sorry

end NUMINAMATH_CALUDE_total_vehicles_proof_l728_72874


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l728_72880

theorem regular_polygon_sides (n : ℕ) (interior_angle : ℝ) : 
  n ≥ 3 → interior_angle = 144 → (n - 2) * 180 = n * interior_angle → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l728_72880


namespace NUMINAMATH_CALUDE_unique_solution_for_n_l728_72869

theorem unique_solution_for_n : ∃! n : ℚ, (1 / (n + 2)) + (2 / (n + 2)) + ((n + 1) / (n + 2)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_n_l728_72869


namespace NUMINAMATH_CALUDE_sara_letters_total_l728_72808

/-- The number of letters Sara sent in January -/
def january_letters : ℕ := 6

/-- The number of letters Sara sent in February -/
def february_letters : ℕ := 9

/-- The number of letters Sara sent in March -/
def march_letters : ℕ := 3 * january_letters

/-- The total number of letters Sara sent -/
def total_letters : ℕ := january_letters + february_letters + march_letters

theorem sara_letters_total : total_letters = 33 := by
  sorry

end NUMINAMATH_CALUDE_sara_letters_total_l728_72808


namespace NUMINAMATH_CALUDE_min_value_and_inequality_l728_72829

-- Define the function f
def f (a b x : ℝ) : ℝ := |x + a| + |2*x - b|

-- State the theorem
theorem min_value_and_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (hmin : ∀ x, f a b x ≥ 1) 
  (hmin_exists : ∃ x, f a b x = 1) : 
  (2*a + b = 2) ∧ 
  (∀ t : ℝ, (∀ a b : ℝ, a > 0 → b > 0 → a + 2*b ≥ t*a*b) → t ≤ 9/2) :=
sorry

end NUMINAMATH_CALUDE_min_value_and_inequality_l728_72829


namespace NUMINAMATH_CALUDE_trig_simplification_l728_72845

theorem trig_simplification :
  (Real.sin (30 * π / 180) + Real.sin (60 * π / 180)) /
  (Real.cos (30 * π / 180) + Real.cos (60 * π / 180)) =
  Real.tan (45 * π / 180) := by sorry

end NUMINAMATH_CALUDE_trig_simplification_l728_72845


namespace NUMINAMATH_CALUDE_students_playing_both_sports_l728_72879

theorem students_playing_both_sports (total : ℕ) (hockey : ℕ) (basketball : ℕ) (neither : ℕ) :
  total = 50 →
  hockey = 30 →
  basketball = 35 →
  neither = 10 →
  hockey + basketball - (total - neither) = 25 :=
by sorry

end NUMINAMATH_CALUDE_students_playing_both_sports_l728_72879


namespace NUMINAMATH_CALUDE_food_additives_budget_percentage_l728_72863

theorem food_additives_budget_percentage :
  ∀ (total_degrees : ℝ) 
    (microphotonics_percent : ℝ) 
    (home_electronics_percent : ℝ) 
    (genetically_modified_microorganisms_percent : ℝ) 
    (industrial_lubricants_percent : ℝ) 
    (basic_astrophysics_degrees : ℝ),
  total_degrees = 360 →
  microphotonics_percent = 14 →
  home_electronics_percent = 19 →
  genetically_modified_microorganisms_percent = 24 →
  industrial_lubricants_percent = 8 →
  basic_astrophysics_degrees = 90 →
  ∃ (food_additives_percent : ℝ),
    food_additives_percent = 10 ∧
    microphotonics_percent + home_electronics_percent + 
    genetically_modified_microorganisms_percent + industrial_lubricants_percent +
    (basic_astrophysics_degrees / total_degrees * 100) + food_additives_percent = 100 :=
by sorry

end NUMINAMATH_CALUDE_food_additives_budget_percentage_l728_72863


namespace NUMINAMATH_CALUDE_square_sum_equality_l728_72821

theorem square_sum_equality (x : ℝ) : (x + 2)^2 + 2*(x + 2)*(5 - x) + (5 - x)^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equality_l728_72821


namespace NUMINAMATH_CALUDE_min_value_of_expression_l728_72859

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_geometric_mean : Real.sqrt 2 = Real.sqrt (4^a * 2^b)) :
  (∀ x y : ℝ, x > 0 → y > 0 → 
    Real.sqrt 2 = Real.sqrt (4^x * 2^y) → 1/x + 2/y ≥ 1/a + 2/b) →
  1/a + 2/b = 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l728_72859


namespace NUMINAMATH_CALUDE_min_value_implies_a_geq_two_l728_72892

/-- The function f(x) defined as x^2 - 2ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 1

/-- Theorem: If the minimum value of f(x) in the interval [-1, 2] is f(2), then a ≥ 2 -/
theorem min_value_implies_a_geq_two (a : ℝ) :
  (∀ x ∈ Set.Icc (-1) 2, f a x ≥ f a 2) → a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_implies_a_geq_two_l728_72892


namespace NUMINAMATH_CALUDE_correct_rounded_result_l728_72850

def round_to_nearest_hundred (n : ℤ) : ℤ :=
  (n + 50) / 100 * 100

theorem correct_rounded_result : round_to_nearest_hundred (68 + 57) = 100 := by
  sorry

end NUMINAMATH_CALUDE_correct_rounded_result_l728_72850


namespace NUMINAMATH_CALUDE_police_coverage_l728_72814

-- Define the set of intersections
inductive Intersection : Type
  | A | B | C | D | E | F | G | H | I | J | K

-- Define the streets as sets of intersections
def horizontal_streets : List (List Intersection) :=
  [[Intersection.A, Intersection.B, Intersection.C, Intersection.D],
   [Intersection.E, Intersection.F, Intersection.G],
   [Intersection.H, Intersection.I, Intersection.J, Intersection.K]]

def vertical_streets : List (List Intersection) :=
  [[Intersection.A, Intersection.E, Intersection.H],
   [Intersection.B, Intersection.F, Intersection.I],
   [Intersection.D, Intersection.G, Intersection.J]]

def diagonal_streets : List (List Intersection) :=
  [[Intersection.H, Intersection.F, Intersection.C],
   [Intersection.C, Intersection.G, Intersection.K]]

def all_streets : List (List Intersection) :=
  horizontal_streets ++ vertical_streets ++ diagonal_streets

-- Define the function to check if a street is covered by the given intersections
def street_covered (street : List Intersection) (officers : List Intersection) : Prop :=
  ∃ i ∈ street, i ∈ officers

-- Theorem statement
theorem police_coverage :
  ∀ (street : List Intersection),
    street ∈ all_streets →
    street_covered street [Intersection.B, Intersection.G, Intersection.H] :=
by
  sorry


end NUMINAMATH_CALUDE_police_coverage_l728_72814


namespace NUMINAMATH_CALUDE_slope_is_negative_one_l728_72886

/-- The slope of a line through two points is -1 -/
theorem slope_is_negative_one (P Q : ℝ × ℝ) : 
  P = (-3, 8) → 
  Q.1 = 5 → 
  Q.2 = 0 → 
  (Q.2 - P.2) / (Q.1 - P.1) = -1 := by
sorry

end NUMINAMATH_CALUDE_slope_is_negative_one_l728_72886


namespace NUMINAMATH_CALUDE_fourth_root_equation_solution_l728_72868

theorem fourth_root_equation_solution :
  ∃! x : ℝ, (2 - x / 2) ^ (1/4 : ℝ) = 2 ∧ x = -28 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equation_solution_l728_72868


namespace NUMINAMATH_CALUDE_polyhedron_volume_l728_72878

/-- The volume of a polyhedron composed of a regular quadrilateral prism and two regular quadrilateral pyramids -/
theorem polyhedron_volume (prism_volume pyramid_volume : ℝ) 
  (h_prism : prism_volume = Real.sqrt 2 - 1)
  (h_pyramid : pyramid_volume = 1 / 6) :
  prism_volume + 2 * pyramid_volume = Real.sqrt 2 - 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_volume_l728_72878


namespace NUMINAMATH_CALUDE_simplify_expression1_simplify_expression2_l728_72826

-- Problem 1
theorem simplify_expression1 (m n : ℝ) :
  2 * m^2 * n - 3 * m * n + 8 - 3 * m^2 * n + 5 * m * n - 3 = -m^2 * n + 2 * m * n + 5 :=
by sorry

-- Problem 2
theorem simplify_expression2 (a b : ℝ) :
  2 * (2 * a - 3 * b) - 3 * (2 * b - 3 * a) = 13 * a - 12 * b :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression1_simplify_expression2_l728_72826


namespace NUMINAMATH_CALUDE_probability_of_queen_in_standard_deck_l728_72890

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (queens : ℕ)

/-- Calculates the probability of drawing a Queen from a given deck -/
def probability_of_queen (d : Deck) : ℚ :=
  d.queens / d.total_cards

/-- Theorem stating the probability of drawing a Queen from a standard deck -/
theorem probability_of_queen_in_standard_deck :
  ∃ (d : Deck), d.total_cards = 52 ∧ d.queens = 4 ∧ probability_of_queen d = 1 / 13 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_queen_in_standard_deck_l728_72890


namespace NUMINAMATH_CALUDE_inequality_proof_l728_72881

theorem inequality_proof (a b c d : ℝ) (h : a * d - b * c = 1) :
  a^2 + b^2 + c^2 + d^2 + a * b + c * d ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l728_72881


namespace NUMINAMATH_CALUDE_same_prime_factors_power_of_two_l728_72884

theorem same_prime_factors_power_of_two (b m n : ℕ) 
  (hb : b ≠ 1) (hmn : m ≠ n) 
  (h_same_factors : ∀ p : ℕ, Nat.Prime p → (p ∣ (b^m - 1) ↔ p ∣ (b^n - 1))) :
  ∃ k : ℕ, b + 1 = 2^k := by sorry

end NUMINAMATH_CALUDE_same_prime_factors_power_of_two_l728_72884


namespace NUMINAMATH_CALUDE_order_of_numbers_l728_72803

theorem order_of_numbers : Real.log 0.45 < 0.45 ∧ 0.45 < 50.4 := by
  sorry

end NUMINAMATH_CALUDE_order_of_numbers_l728_72803


namespace NUMINAMATH_CALUDE_first_day_income_l728_72830

/-- Given a sequence where each term is double the previous term,
    and the 10th term is 18, prove that the first term is 0.03515625 -/
theorem first_day_income (a : ℕ → ℝ) (h1 : ∀ n, a (n + 1) = 2 * a n) (h2 : a 10 = 18) :
  a 1 = 0.03515625 := by
  sorry

end NUMINAMATH_CALUDE_first_day_income_l728_72830


namespace NUMINAMATH_CALUDE_lcm_gcf_problem_l728_72836

theorem lcm_gcf_problem (n : ℕ) : 
  Nat.lcm n 16 = 48 → Nat.gcd n 16 = 18 → n = 54 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_problem_l728_72836


namespace NUMINAMATH_CALUDE_max_leftover_fruits_l728_72867

theorem max_leftover_fruits (A G : ℕ) : 
  (A % 7 ≤ 6) ∧ (G % 7 ≤ 6) ∧ 
  ∃ (A₀ G₀ : ℕ), A₀ % 7 = 6 ∧ G₀ % 7 = 6 :=
by sorry

end NUMINAMATH_CALUDE_max_leftover_fruits_l728_72867


namespace NUMINAMATH_CALUDE_lockridge_marching_band_max_size_l728_72897

theorem lockridge_marching_band_max_size :
  ∀ n : ℕ,
  (22 * n ≡ 2 [ZMOD 24]) →
  (22 * n < 1000) →
  (∀ m : ℕ, (22 * m ≡ 2 [ZMOD 24]) → (22 * m < 1000) → (22 * m ≤ 22 * n)) →
  22 * n = 770 :=
by sorry

end NUMINAMATH_CALUDE_lockridge_marching_band_max_size_l728_72897


namespace NUMINAMATH_CALUDE_price_of_first_oil_l728_72862

/-- Given two oils mixed together, prove the price of the first oil. -/
theorem price_of_first_oil :
  -- Define the volumes of oils
  let volume_first : ℝ := 10
  let volume_second : ℝ := 5
  -- Define the price of the second oil
  let price_second : ℝ := 68
  -- Define the price of the mixture
  let price_mixture : ℝ := 56
  -- Define the total volume
  let volume_total : ℝ := volume_first + volume_second
  -- The equation that represents the mixing of oils
  ∀ price_first : ℝ,
    volume_first * price_first + volume_second * price_second =
    volume_total * price_mixture →
    -- Prove that the price of the first oil is 50
    price_first = 50 := by
  sorry

end NUMINAMATH_CALUDE_price_of_first_oil_l728_72862


namespace NUMINAMATH_CALUDE_iwatch_price_l728_72835

theorem iwatch_price (iphone_price : ℝ) (iphone_discount : ℝ) (iwatch_discount : ℝ) 
  (cashback : ℝ) (total_cost : ℝ) :
  iphone_price = 800 ∧
  iphone_discount = 0.15 ∧
  iwatch_discount = 0.10 ∧
  cashback = 0.02 ∧
  total_cost = 931 →
  ∃ (iwatch_price : ℝ),
    iwatch_price = 300 ∧
    (1 - cashback) * ((1 - iphone_discount) * iphone_price + 
    (1 - iwatch_discount) * iwatch_price) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_iwatch_price_l728_72835


namespace NUMINAMATH_CALUDE_largest_n_with_2020_sets_l728_72882

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Define a_n as the number of sets S of positive integers
-- such that the sum of F_k for k in S equals n
def a (n : ℕ) : ℕ := sorry

-- State the theorem
theorem largest_n_with_2020_sets :
  ∃ n : ℕ, a n = 2020 ∧ ∀ m : ℕ, m > n → a m ≠ 2020 ∧ n = fib 2022 - 1 := by
  sorry

end NUMINAMATH_CALUDE_largest_n_with_2020_sets_l728_72882


namespace NUMINAMATH_CALUDE_percentage_of_singles_l728_72812

def total_hits : ℕ := 45
def home_runs : ℕ := 2
def triples : ℕ := 2
def doubles : ℕ := 8

def non_singles : ℕ := home_runs + triples + doubles
def singles : ℕ := total_hits - non_singles

theorem percentage_of_singles :
  (singles : ℚ) / total_hits * 100 = 73 := by sorry

end NUMINAMATH_CALUDE_percentage_of_singles_l728_72812


namespace NUMINAMATH_CALUDE_direction_vector_form_l728_72898

/-- Given a line passing through two points, prove that its direction vector
    has a specific form. -/
theorem direction_vector_form (p₁ p₂ : ℝ × ℝ) (b : ℝ) : 
  p₁ = (-3, 4) →
  p₂ = (2, -1) →
  (p₂.1 - p₁.1, p₂.2 - p₁.2) = (b * (p₂.2 - p₁.2), p₂.2 - p₁.2) →
  b = 1 := by
  sorry

#check direction_vector_form

end NUMINAMATH_CALUDE_direction_vector_form_l728_72898


namespace NUMINAMATH_CALUDE_equation_is_parabola_l728_72833

-- Define the equation
def equation (x y : ℝ) : Prop :=
  |y - 3| = Real.sqrt ((x + 4)^2 + (y - 1)^2)

-- Theorem statement
theorem equation_is_parabola :
  ∃ (a b c : ℝ), a ≠ 0 ∧
  (∀ x y : ℝ, equation x y ↔ y = a * x^2 + b * x + c) :=
sorry

end NUMINAMATH_CALUDE_equation_is_parabola_l728_72833


namespace NUMINAMATH_CALUDE_arithmetic_computation_l728_72820

theorem arithmetic_computation : -12 * 3 - (-4 * -5) + (-8 * -6) + 2 = -6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l728_72820


namespace NUMINAMATH_CALUDE_opposite_number_on_line_l728_72885

theorem opposite_number_on_line (a : ℝ) : (a + (a - 6) = 0) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_number_on_line_l728_72885


namespace NUMINAMATH_CALUDE_condition_p_sufficient_not_necessary_l728_72864

theorem condition_p_sufficient_not_necessary :
  (∀ x y : ℝ, x > 1 ∧ y > 1 → x + y > 2 ∧ x * y > 1) ∧
  (∃ x y : ℝ, x + y > 2 ∧ x * y > 1 ∧ ¬(x > 1 ∧ y > 1)) := by
  sorry

end NUMINAMATH_CALUDE_condition_p_sufficient_not_necessary_l728_72864


namespace NUMINAMATH_CALUDE_equation_roots_and_sum_l728_72896

theorem equation_roots_and_sum : ∃ (c d : ℝ),
  (∃! (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ 
    (∀ x : ℝ, (x + 3) * (x + c) * (x - 9) = 0 ↔ (x = r₁ ∨ x = r₂))) ∧
  (∃! (s₁ s₂ s₃ : ℝ), s₁ ≠ s₂ ∧ s₁ ≠ s₃ ∧ s₂ ≠ s₃ ∧ 
    (∀ x : ℝ, (x - c) * (x - 7) * (x + 5) = 0 ↔ (x = s₁ ∨ x = s₂ ∨ x = s₃))) ∧
  80 * c + 10 * d = 650 :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_and_sum_l728_72896


namespace NUMINAMATH_CALUDE_pirate_coin_distribution_l728_72831

/-- The number of rounds in the coin distribution process -/
def y : ℕ := sorry

/-- The total number of coins Pete has after distribution -/
def peteCoins : ℕ := y * (y + 1) / 2

/-- The total number of coins Paul has after distribution -/
def paulCoins : ℕ := y

/-- The ratio of Pete's coins to Paul's coins -/
def coinRatio : ℕ := 5

theorem pirate_coin_distribution :
  peteCoins = coinRatio * paulCoins ∧ peteCoins + paulCoins = 54 := by
  sorry

end NUMINAMATH_CALUDE_pirate_coin_distribution_l728_72831


namespace NUMINAMATH_CALUDE_decoration_count_l728_72876

theorem decoration_count (total : ℕ) (nails : ℕ) (h1 : nails = 50) 
  (h2 : nails = (2 : ℚ) / 3 * total) : 
  total - nails - (2 : ℚ) / 5 * (total - nails) = 15 := by
  sorry

end NUMINAMATH_CALUDE_decoration_count_l728_72876


namespace NUMINAMATH_CALUDE_minimum_fare_increase_l728_72865

/-- Represents the fare structure for a taxi service -/
structure FareStructure where
  n : ℝ  -- Total number of passengers
  t : ℝ  -- Base fare
  X : ℝ  -- Fare increase for businessmen

/-- Calculates the total revenue under the given fare structure -/
def totalRevenue (f : FareStructure) : ℝ :=
  0.75 * f.n * f.t + 0.2 * f.n * (f.t + f.X)

/-- Theorem stating the minimum fare increase that doesn't decrease total revenue -/
theorem minimum_fare_increase (f : FareStructure) :
  (∀ X : ℝ, totalRevenue { n := f.n, t := f.t, X := X } ≥ f.n * f.t → X ≥ f.t / 4) ∧
  totalRevenue { n := f.n, t := f.t, X := f.t / 4 } ≥ f.n * f.t :=
by sorry

end NUMINAMATH_CALUDE_minimum_fare_increase_l728_72865


namespace NUMINAMATH_CALUDE_rectangle_breadth_l728_72852

theorem rectangle_breadth (L B : ℝ) (h1 : L / B = 25 / 16) (h2 : L * B = 200 * 200) : B = 160 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_breadth_l728_72852


namespace NUMINAMATH_CALUDE_inequality_equivalence_l728_72857

theorem inequality_equivalence (x : ℝ) : 
  (-1/3 : ℝ) ≤ (5-x)/2 ∧ (5-x)/2 < (1/3 : ℝ) ↔ (13/3 : ℝ) < x ∧ x ≤ (17/3 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l728_72857


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l728_72802

open Set

-- Define the sets A and B
def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {x | -1 < x ∧ x < 2}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x | x > -1} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l728_72802


namespace NUMINAMATH_CALUDE_cubic_sum_of_roots_l728_72856

theorem cubic_sum_of_roots (m n r s : ℝ) : 
  (r^2 - m*r - n = 0) → (s^2 - m*s - n = 0) → r^3 + s^3 = m^3 + 3*n*m :=
by sorry

end NUMINAMATH_CALUDE_cubic_sum_of_roots_l728_72856


namespace NUMINAMATH_CALUDE_quadratic_roots_theorem_l728_72828

theorem quadratic_roots_theorem (k : ℝ) (a b : ℝ) : 
  (∀ x, k * (x^2 - x) + x + 2 = 0 ↔ x = a ∨ x = b) →
  (a / b + b / a = 3 / 7) →
  (∃ k₁ k₂ : ℝ, 
    k₁ = (20 + Real.sqrt 988) / 14 ∧
    k₂ = (20 - Real.sqrt 988) / 14 ∧
    (k = k₁ ∨ k = k₂) ∧
    k₁ / k₂ + k₂ / k₁ = -104 / 21) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_theorem_l728_72828


namespace NUMINAMATH_CALUDE_solution_range_l728_72877

theorem solution_range (x a : ℝ) : 
  3 * x + 2 * (3 * a + 1) = 6 * x + a → x ≥ 0 → a ≥ -2/5 := by
  sorry

end NUMINAMATH_CALUDE_solution_range_l728_72877


namespace NUMINAMATH_CALUDE_problem_solution_l728_72825

theorem problem_solution (a : ℚ) : a + a / 4 = 6 / 2 → a = 12 / 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l728_72825


namespace NUMINAMATH_CALUDE_cost_doubling_l728_72891

theorem cost_doubling (t b : ℝ) (t_pos : t > 0) (b_pos : b > 0) :
  let original_cost := t * b^4
  let new_cost := t * (2*b)^4
  (new_cost / original_cost) * 100 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_cost_doubling_l728_72891


namespace NUMINAMATH_CALUDE_equation_solution_verify_solution_l728_72816

/-- The solution to the equation √((3x-1)/(x+4)) + 3 - 4√((x+4)/(3x-1)) = 0 -/
theorem equation_solution :
  ∃ x : ℝ, (Real.sqrt ((3 * x - 1) / (x + 4)) + 3 - 4 * Real.sqrt ((x + 4) / (3 * x - 1)) = 0) ∧
           x = 5 / 2 := by
  sorry

/-- Verification that 5/2 is indeed the solution -/
theorem verify_solution :
  let x : ℝ := 5 / 2
  Real.sqrt ((3 * x - 1) / (x + 4)) + 3 - 4 * Real.sqrt ((x + 4) / (3 * x - 1)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_verify_solution_l728_72816


namespace NUMINAMATH_CALUDE_stock_bond_relationship_l728_72872

/-- Represents the investment portfolio of Matthew -/
structure Portfolio where
  expensive_stock_value : ℝ  -- Value of the more expensive stock per share
  expensive_stock_shares : ℕ  -- Number of shares of the more expensive stock
  cheap_stock_shares : ℕ     -- Number of shares of the cheaper stock
  bond_face_value : ℝ        -- Face value of the bond
  bond_coupon_rate : ℝ       -- Annual coupon rate of the bond
  bond_discount : ℝ          -- Discount rate at which the bond was purchased
  total_assets : ℝ           -- Total value of assets in stocks and bond
  bond_market_value : ℝ      -- Current market value of the bond

/-- Theorem stating the relationship between the more expensive stock value and the bond market value -/
theorem stock_bond_relationship (p : Portfolio) 
  (h1 : p.expensive_stock_shares = 14)
  (h2 : p.cheap_stock_shares = 26)
  (h3 : p.bond_face_value = 1000)
  (h4 : p.bond_coupon_rate = 0.06)
  (h5 : p.bond_discount = 0.03)
  (h6 : p.total_assets = 2106)
  (h7 : p.expensive_stock_value * p.expensive_stock_shares + 
        (p.expensive_stock_value / 2) * p.cheap_stock_shares + 
        p.bond_market_value = p.total_assets) :
  p.bond_market_value = 2106 - 27 * p.expensive_stock_value := by
  sorry

end NUMINAMATH_CALUDE_stock_bond_relationship_l728_72872


namespace NUMINAMATH_CALUDE_cubic_root_sum_l728_72805

theorem cubic_root_sum (a b c : ℝ) : 
  (a^3 - 15*a^2 + 22*a - 8 = 0) →
  (b^3 - 15*b^2 + 22*b - 8 = 0) →
  (c^3 - 15*c^2 + 22*c - 8 = 0) →
  (a / ((1/a) + b*c) + b / ((1/b) + c*a) + c / ((1/c) + a*b) = 20 + 1/9) := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l728_72805


namespace NUMINAMATH_CALUDE_correct_percentage_l728_72809

theorem correct_percentage (y : ℕ) (y_pos : y > 0) : 
  let total := 7 * y
  let incorrect := 2 * y
  let correct := total - incorrect
  (correct : ℚ) / total * 100 = 500 / 7 := by
  sorry

end NUMINAMATH_CALUDE_correct_percentage_l728_72809


namespace NUMINAMATH_CALUDE_pizza_cost_three_pizzas_cost_l728_72899

/-- The cost of all pizzas given the number of pizzas, slices per pizza, and the cost of a subset of slices. -/
theorem pizza_cost (num_pizzas : ℕ) (slices_per_pizza : ℕ) (subset_slices : ℕ) (subset_cost : ℚ) : ℚ :=
  let total_slices := num_pizzas * slices_per_pizza
  let cost_per_slice := subset_cost / subset_slices
  total_slices * cost_per_slice

/-- Proof that 3 pizzas with 12 slices each cost $72, given that 5 slices cost $10. -/
theorem three_pizzas_cost : pizza_cost 3 12 5 10 = 72 := by
  sorry

end NUMINAMATH_CALUDE_pizza_cost_three_pizzas_cost_l728_72899


namespace NUMINAMATH_CALUDE_estimate_battery_usage_l728_72858

def sample_size : ℕ := 6
def class_size : ℕ := 45
def battery_usage : List ℕ := [33, 25, 28, 26, 25, 31]

theorem estimate_battery_usage :
  (class_size : ℝ) * ((battery_usage.sum : ℝ) / sample_size) = 1260 := by
  sorry

end NUMINAMATH_CALUDE_estimate_battery_usage_l728_72858


namespace NUMINAMATH_CALUDE_vector_at_zero_l728_72889

/-- A line parameterized by t in 3D space -/
structure ParametricLine where
  point : ℝ → ℝ × ℝ × ℝ

/-- The given line satisfies the conditions -/
def given_line : ParametricLine where
  point := λ t => sorry

theorem vector_at_zero (l : ParametricLine)
  (h1 : l.point (-2) = (2, 6, 16))
  (h2 : l.point 1 = (-1, -4, -10)) :
  l.point 0 = (0, 2/3, 16/3) := by
  sorry

end NUMINAMATH_CALUDE_vector_at_zero_l728_72889


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_3_seconds_l728_72848

-- Define the displacement function
def displacement (t : ℝ) : ℝ := t^2 - t

-- Define the velocity function as the derivative of displacement
def velocity (t : ℝ) : ℝ := 2 * t - 1

-- Theorem statement
theorem instantaneous_velocity_at_3_seconds : 
  velocity 3 = 5 := by sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_3_seconds_l728_72848


namespace NUMINAMATH_CALUDE_yolanda_total_points_l728_72839

/-- Calculate the total points scored by a basketball player over a season. -/
def total_points_scored (games : ℕ) (free_throws two_pointers three_pointers : ℕ) : ℕ :=
  games * (free_throws * 1 + two_pointers * 2 + three_pointers * 3)

/-- Theorem: Yolanda's total points scored over the entire season is 345. -/
theorem yolanda_total_points : 
  total_points_scored 15 4 5 3 = 345 := by
  sorry

end NUMINAMATH_CALUDE_yolanda_total_points_l728_72839


namespace NUMINAMATH_CALUDE_rectangle_square_problem_l728_72815

/-- Given a rectangle with length-to-width ratio 2:1 and area 50 cm², 
    and a square with the same area as the rectangle, prove:
    1. The rectangle's length is 10 cm and width is 5 cm
    2. The difference between the square's side length and the rectangle's width is 5(√2 - 1) cm -/
theorem rectangle_square_problem (length width : ℝ) (square_side : ℝ) : 
  length = 2 * width → 
  length * width = 50 → 
  square_side^2 = 50 → 
  (length = 10 ∧ width = 5) ∧ 
  square_side - width = 5 * (Real.sqrt 2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_square_problem_l728_72815


namespace NUMINAMATH_CALUDE_max_type_a_stationery_l728_72887

/-- Represents the number of items for each stationery type -/
structure Stationery where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Calculates the total cost of stationery -/
def totalCost (s : Stationery) : ℕ :=
  3 * s.a + 2 * s.b + s.c

/-- Checks if the stationery purchase satisfies all conditions -/
def isValidPurchase (s : Stationery) : Prop :=
  s.b = s.a - 2 ∧
  3 * s.a ≤ 33 ∧
  totalCost s = 66

/-- Theorem: The maximum number of Type A stationery that can be purchased is 11 -/
theorem max_type_a_stationery :
  ∃ (s : Stationery), isValidPurchase s ∧
  (∀ (t : Stationery), isValidPurchase t → t.a ≤ s.a) ∧
  s.a = 11 := by
  sorry


end NUMINAMATH_CALUDE_max_type_a_stationery_l728_72887


namespace NUMINAMATH_CALUDE_smallest_b_value_l728_72875

theorem smallest_b_value (a b : ℕ+) (h1 : a.val - b.val = 6) 
  (h2 : Nat.gcd ((a.val^3 + b.val^3) / (a.val + b.val)) (a.val * b.val) = 9) :
  b.val ≥ 3 ∧ ∃ (a' b' : ℕ+), b'.val = 3 ∧ a'.val - b'.val = 6 ∧ 
    Nat.gcd ((a'.val^3 + b'.val^3) / (a'.val + b'.val)) (a'.val * b'.val) = 9 :=
by sorry


end NUMINAMATH_CALUDE_smallest_b_value_l728_72875


namespace NUMINAMATH_CALUDE_solution_difference_l728_72843

-- Define the equation
def equation (x : ℝ) : Prop :=
  (6 * x - 18) / (x^2 + x - 20) = x + 3

-- Define the theorem
theorem solution_difference :
  ∃ (p q : ℝ), 
    p ≠ q ∧
    equation p ∧
    equation q ∧
    p > q ∧
    p - q = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_solution_difference_l728_72843


namespace NUMINAMATH_CALUDE_x_convergence_bound_l728_72873

def x : ℕ → ℚ
  | 0 => 5
  | n + 1 => (x n ^ 2 + 5 * x n + 4) / (x n + 6)

theorem x_convergence_bound :
  ∃ m : ℕ, (∀ k < m, x k > 4 + 1 / 2^20) ∧
           (x m ≤ 4 + 1 / 2^20) ∧
           (81 ≤ m) ∧ (m ≤ 242) :=
by sorry

end NUMINAMATH_CALUDE_x_convergence_bound_l728_72873


namespace NUMINAMATH_CALUDE_f_is_quadratic_l728_72832

/-- Definition of a quadratic equation in one variable -/
def is_quadratic_one_var (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x^2 - 1 -/
def f (x : ℝ) : ℝ := x^2 - 1

/-- Theorem stating that f is a quadratic equation in one variable -/
theorem f_is_quadratic : is_quadratic_one_var f := by sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l728_72832


namespace NUMINAMATH_CALUDE_equation_solution_l728_72895

theorem equation_solution : 
  ∃! x : ℝ, x ≠ -4 ∧ (7 * x / (x + 4) - 5 / (x + 4) = 2 / (x + 4)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l728_72895


namespace NUMINAMATH_CALUDE_one_beaver_still_working_l728_72851

/-- The number of beavers still working on their home -/
def beavers_still_working (initial : ℕ) (left : ℕ) : ℕ :=
  initial - left

/-- Proof that 1 beaver is still working on their home -/
theorem one_beaver_still_working :
  beavers_still_working 2 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_beaver_still_working_l728_72851


namespace NUMINAMATH_CALUDE_count_common_divisors_9240_8820_l728_72824

/-- The number of positive divisors that 9240 and 8820 have in common -/
def common_divisors_count : ℕ := 24

/-- Theorem stating that the number of positive divisors that 9240 and 8820 have in common is 24 -/
theorem count_common_divisors_9240_8820 : 
  (Nat.divisors 9240 ∩ Nat.divisors 8820).card = common_divisors_count := by
  sorry

end NUMINAMATH_CALUDE_count_common_divisors_9240_8820_l728_72824


namespace NUMINAMATH_CALUDE_sequence_properties_l728_72854

def S (n : ℕ) : ℤ := 2 * n^2 - 10 * n

def a (n : ℕ) : ℤ := 4 * n - 5

theorem sequence_properties :
  (∀ n : ℕ, S (n + 1) - S n = a (n + 1)) ∧
  (∃ n : ℕ, ∀ m : ℕ, S m ≥ S n) ∧
  (∃ n : ℕ, S n = -12 ∧ ∀ m : ℕ, S m ≥ S n) :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_l728_72854


namespace NUMINAMATH_CALUDE_at_least_two_same_number_l728_72849

/-- The number of dice being rolled -/
def num_dice : ℕ := 8

/-- The number of sides on each die -/
def num_sides : ℕ := 6

/-- The probability of at least two dice showing the same number -/
def prob_same_number : ℝ := 1

theorem at_least_two_same_number :
  num_dice > num_sides → prob_same_number = 1 := by sorry

end NUMINAMATH_CALUDE_at_least_two_same_number_l728_72849


namespace NUMINAMATH_CALUDE_maria_candy_eaten_l728_72822

/-- Given that Maria initially had 67 pieces of candy and now has 3 pieces,
    prove that she ate 64 pieces of candy. -/
theorem maria_candy_eaten (initial : ℕ) (remaining : ℕ) (eaten : ℕ) : 
  initial = 67 → remaining = 3 → eaten = initial - remaining → eaten = 64 := by
  sorry

end NUMINAMATH_CALUDE_maria_candy_eaten_l728_72822


namespace NUMINAMATH_CALUDE_ben_win_probability_l728_72855

theorem ben_win_probability (lose_prob : ℚ) (h1 : lose_prob = 3 / 7) (h2 : ¬ ∃ tie_prob : ℚ, tie_prob ≠ 0) : 
  1 - lose_prob = 4 / 7 := by
sorry

end NUMINAMATH_CALUDE_ben_win_probability_l728_72855


namespace NUMINAMATH_CALUDE_books_loaned_out_l728_72818

/-- Proves that the number of books loaned out is 30 given the initial count, return rate, and final count -/
theorem books_loaned_out 
  (initial_count : ℕ) 
  (return_rate : ℚ) 
  (final_count : ℕ) 
  (h1 : initial_count = 75)
  (h2 : return_rate = 4/5)
  (h3 : final_count = 69) :
  (initial_count - final_count : ℚ) / (1 - return_rate) = 30 := by
sorry

end NUMINAMATH_CALUDE_books_loaned_out_l728_72818


namespace NUMINAMATH_CALUDE_worker_productivity_increase_l728_72893

theorem worker_productivity_increase 
  (initial_value : ℝ) 
  (final_value : ℝ) 
  (increase_percentage : ℝ) 
  (h1 : initial_value = 2500)
  (h2 : final_value = 2809)
  (h3 : final_value = initial_value * (1 + increase_percentage / 100)^2) :
  increase_percentage = 6 := by
sorry

end NUMINAMATH_CALUDE_worker_productivity_increase_l728_72893


namespace NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l728_72823

theorem square_sum_zero_implies_both_zero (a b : ℝ) : a^2 + b^2 = 0 → a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l728_72823


namespace NUMINAMATH_CALUDE_cindy_marbles_l728_72866

theorem cindy_marbles (initial_marbles : ℕ) (friends : ℕ) (marbles_per_friend : ℕ) :
  initial_marbles = 500 →
  friends = 4 →
  marbles_per_friend = 80 →
  4 * (initial_marbles - friends * marbles_per_friend) = 720 :=
by sorry

end NUMINAMATH_CALUDE_cindy_marbles_l728_72866


namespace NUMINAMATH_CALUDE_inverse_mod_53_l728_72801

theorem inverse_mod_53 (h : (17⁻¹ : ZMod 53) = 13) : (36⁻¹ : ZMod 53) = 40 := by
  sorry

end NUMINAMATH_CALUDE_inverse_mod_53_l728_72801


namespace NUMINAMATH_CALUDE_min_a_value_l728_72807

noncomputable section

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * Real.log x - 3 * x^2 - 11 * x

-- Define the inequality condition
def inequality_condition (a : ℝ) : Prop :=
  ∀ x > 0, f x ≤ (a - 3) * x^2 + (2 * a - 13) * x + 1

-- State the theorem
theorem min_a_value :
  ∃ a : ℤ, (inequality_condition a ∧ ∀ b : ℤ, b < a → ¬inequality_condition b) :=
sorry

end

end NUMINAMATH_CALUDE_min_a_value_l728_72807


namespace NUMINAMATH_CALUDE_fraction_of_roots_equals_exponents_l728_72813

theorem fraction_of_roots_equals_exponents (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a^3 * b)^(1/2) / (a * b)^(1/3) = a^(7/6) * b^(1/6) := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_roots_equals_exponents_l728_72813


namespace NUMINAMATH_CALUDE_same_sign_product_and_quotient_abs_l728_72840

theorem same_sign_product_and_quotient_abs (a b : ℚ) (hb : b ≠ 0) :
  (a * b > 0 ↔ |a| / |b| > 0) ∧ (a * b < 0 ↔ |a| / |b| < 0) ∧ (a * b = 0 ↔ |a| / |b| = 0) :=
sorry

end NUMINAMATH_CALUDE_same_sign_product_and_quotient_abs_l728_72840


namespace NUMINAMATH_CALUDE_naples_pizza_weight_l728_72834

/-- The total weight of pizza eaten by Rachel and Bella in Naples -/
def total_pizza_weight (rachel_pizza : ℕ) (rachel_mushrooms : ℕ) (rachel_olives : ℕ)
                       (bella_pizza : ℕ) (bella_cheese : ℕ) (bella_onions : ℕ) : ℕ :=
  (rachel_pizza + rachel_mushrooms + rachel_olives) + (bella_pizza + bella_cheese + bella_onions)

/-- Theorem stating the total weight of pizza eaten by Rachel and Bella in Naples -/
theorem naples_pizza_weight :
  total_pizza_weight 598 100 50 354 75 55 = 1232 := by
  sorry

end NUMINAMATH_CALUDE_naples_pizza_weight_l728_72834


namespace NUMINAMATH_CALUDE_janna_weekly_sleep_l728_72861

/-- The number of hours Janna sleeps in a week -/
def total_sleep_hours (weekday_sleep : ℕ) (weekend_sleep : ℕ) (weekdays : ℕ) (weekend_days : ℕ) : ℕ :=
  weekday_sleep * weekdays + weekend_sleep * weekend_days

/-- Theorem stating that Janna sleeps 51 hours in a week -/
theorem janna_weekly_sleep :
  total_sleep_hours 7 8 5 2 = 51 := by
  sorry

end NUMINAMATH_CALUDE_janna_weekly_sleep_l728_72861


namespace NUMINAMATH_CALUDE_middle_number_proof_l728_72817

theorem middle_number_proof (x y z : ℕ) : 
  x < y ∧ y < z →
  x + y = 18 →
  x + z = 23 →
  y + z = 27 →
  y = 11 := by
sorry

end NUMINAMATH_CALUDE_middle_number_proof_l728_72817


namespace NUMINAMATH_CALUDE_route_down_length_l728_72838

/-- Proves that the length of the route down the mountain is 15 miles -/
theorem route_down_length (time_up time_down : ℝ) (rate_up rate_down : ℝ) 
  (h1 : time_up = time_down)
  (h2 : rate_down = 1.5 * rate_up)
  (h3 : rate_up = 5)
  (h4 : time_up = 2) : 
  rate_down * time_down = 15 := by
  sorry

end NUMINAMATH_CALUDE_route_down_length_l728_72838


namespace NUMINAMATH_CALUDE_reflected_light_ray_equation_l728_72870

/-- Given a point P and its reflection P' across the x-axis, and another point Q,
    this function returns true if the given equation represents the line through P' and Q -/
def is_reflected_line_equation (P Q : ℝ × ℝ) (equation : ℝ → ℝ → ℝ) : Prop :=
  let P' := (P.1, -P.2)  -- Reflection of P across x-axis
  (equation P'.1 P'.2 = 0) ∧ (equation Q.1 Q.2 = 0)

/-- The main theorem stating that 4x + y - 5 = 0 is the equation of the 
    reflected light ray for the given points -/
theorem reflected_light_ray_equation :
  is_reflected_line_equation (2, 3) (1, 1) (fun x y => 4*x + y - 5) := by
  sorry

#check reflected_light_ray_equation

end NUMINAMATH_CALUDE_reflected_light_ray_equation_l728_72870


namespace NUMINAMATH_CALUDE_correct_hardbacks_verify_selections_l728_72853

def total_books : ℕ := 8
def paperbacks : ℕ := 2
def selections_with_paperback : ℕ := 36

def hardbacks : ℕ := total_books - paperbacks

theorem correct_hardbacks : hardbacks = 6 := by sorry

theorem verify_selections :
  Nat.choose total_books 3 - Nat.choose hardbacks 3 = selections_with_paperback := by sorry

end NUMINAMATH_CALUDE_correct_hardbacks_verify_selections_l728_72853


namespace NUMINAMATH_CALUDE_zuminglish_seven_letter_words_l728_72871

/-- Represents the ending of a word -/
inductive WordEnding
| CC  -- Two consonants
| CV  -- Consonant followed by vowel
| VC  -- Vowel followed by consonant

/-- Represents the rules of Zuminglish -/
structure Zuminglish where
  -- The number of n-letter words ending in each type
  count : ℕ → WordEnding → ℕ
  -- Initial conditions for 2-letter words
  init_CC : count 2 WordEnding.CC = 4
  init_CV : count 2 WordEnding.CV = 2
  init_VC : count 2 WordEnding.VC = 2
  -- Recursive relations
  rec_CC : ∀ n, count (n+1) WordEnding.CC = 2 * (count n WordEnding.CC + count n WordEnding.VC)
  rec_CV : ∀ n, count (n+1) WordEnding.CV = count n WordEnding.CC
  rec_VC : ∀ n, count (n+1) WordEnding.VC = 2 * count n WordEnding.CV

/-- The main theorem stating the number of valid 7-letter words in Zuminglish -/
theorem zuminglish_seven_letter_words (z : Zuminglish) :
  z.count 7 WordEnding.CC + z.count 7 WordEnding.CV + z.count 7 WordEnding.VC = 912 := by
  sorry


end NUMINAMATH_CALUDE_zuminglish_seven_letter_words_l728_72871


namespace NUMINAMATH_CALUDE_inequality_proof_l728_72837

theorem inequality_proof (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (a - b)^2 / (8*a) < (a + b) / 2 - Real.sqrt (a*b) ∧
  (a + b) / 2 - Real.sqrt (a*b) < (a - b)^2 / (8*b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l728_72837


namespace NUMINAMATH_CALUDE_mike_score_is_99_percent_l728_72883

/-- Represents the exam scores of four students -/
structure ExamScores where
  gibi : ℝ
  jigi : ℝ
  mike : ℝ
  lizzy : ℝ

/-- Theorem stating that Mike's score is 99% given the conditions -/
theorem mike_score_is_99_percent 
  (scores : ExamScores)
  (h_gibi : scores.gibi = 59)
  (h_jigi : scores.jigi = 55)
  (h_lizzy : scores.lizzy = 67)
  (h_max_score : ℝ := 700)
  (h_average : (scores.gibi + scores.jigi + scores.mike + scores.lizzy) / 4 * h_max_score / 100 = 490) :
  scores.mike = 99 := by
sorry

end NUMINAMATH_CALUDE_mike_score_is_99_percent_l728_72883


namespace NUMINAMATH_CALUDE_expand_and_simplify_l728_72888

theorem expand_and_simplify (x : ℝ) : (x + 5) * (4 * x - 9 - 3) = 4 * x^2 + 8 * x - 60 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l728_72888


namespace NUMINAMATH_CALUDE_sin_graph_transform_l728_72819

/-- Given a function f : ℝ → ℝ where f x = sin x for all x ∈ ℝ,
    prove that shifting its graph left by π/3 and then halving the x-coordinates
    results in the function g where g x = sin(2x + π/3) for all x ∈ ℝ. -/
theorem sin_graph_transform (f g : ℝ → ℝ) (h₁ : ∀ x, f x = Real.sin x) 
  (h₂ : ∀ x, g x = f (2*x + π/3)) : ∀ x, g x = Real.sin (2*x + π/3) := by
  sorry

end NUMINAMATH_CALUDE_sin_graph_transform_l728_72819


namespace NUMINAMATH_CALUDE_fred_basketball_games_l728_72847

def games_this_year : ℕ := 36
def total_games : ℕ := 47

def games_last_year : ℕ := total_games - games_this_year

theorem fred_basketball_games : games_last_year = 11 := by
  sorry

end NUMINAMATH_CALUDE_fred_basketball_games_l728_72847


namespace NUMINAMATH_CALUDE_problem_solution_l728_72846

def A (a b : ℝ) : Set ℝ := {x | a * x^2 + b * x + 1 = 0}
def B : Set ℝ := {-1, 1}

theorem problem_solution (a b : ℝ) :
  (B ⊆ A a b → a = -1) ∧
  (A a b ∩ B ≠ ∅ → a^2 - b^2 + 2*a = -1) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l728_72846


namespace NUMINAMATH_CALUDE_positive_quadratic_form_condition_l728_72804

theorem positive_quadratic_form_condition (a b c : ℝ) : 
  (∀ x y : ℝ, x^2 + x*y + y^2 + a*x + b*y + c > 0) → 
  a^2 - a*b + b^2 < 3*c :=
by sorry

end NUMINAMATH_CALUDE_positive_quadratic_form_condition_l728_72804


namespace NUMINAMATH_CALUDE_complex_expression_evaluation_l728_72827

theorem complex_expression_evaluation :
  (7 - 3 * Complex.I) - 3 * (2 + 4 * Complex.I) = 1 - 15 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_evaluation_l728_72827
