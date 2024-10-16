import Mathlib

namespace NUMINAMATH_CALUDE_abc_product_l3118_311884

theorem abc_product (a b c : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h_sum : a + b + c = 30)
  (h_eq : (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c + 300 / (a * b * c) = 1) :
  a * b * c = 768 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_l3118_311884


namespace NUMINAMATH_CALUDE_custom_op_theorem_l3118_311850

/-- Custom operation ⊗ defined as x ⊗ y = x^3 + y^3 -/
def custom_op (x y : ℝ) : ℝ := x^3 + y^3

/-- Theorem stating that h ⊗ (h ⊗ h) = h^3 + 8h^9 -/
theorem custom_op_theorem (h : ℝ) : custom_op h (custom_op h h) = h^3 + 8*h^9 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_theorem_l3118_311850


namespace NUMINAMATH_CALUDE_chess_tournament_games_l3118_311876

/-- The number of games played in a chess tournament -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a chess tournament with 20 participants, where each participant
    plays exactly one game with each of the remaining participants,
    the total number of games played is 190. -/
theorem chess_tournament_games :
  num_games 20 = 190 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l3118_311876


namespace NUMINAMATH_CALUDE_cookies_left_l3118_311834

def dozen : ℕ := 12

def cookies_baked : ℕ := 5 * dozen
def cookies_sold_to_stone : ℕ := 2 * dozen
def cookies_sold_to_brock : ℕ := 7
def cookies_sold_to_katy : ℕ := 2 * cookies_sold_to_brock

theorem cookies_left :
  cookies_baked - (cookies_sold_to_stone + cookies_sold_to_brock + cookies_sold_to_katy) = 15 := by
  sorry

end NUMINAMATH_CALUDE_cookies_left_l3118_311834


namespace NUMINAMATH_CALUDE_min_dials_for_lighting_l3118_311842

/-- Represents a stack of 12-sided dials -/
def DialStack := ℕ → Fin 12 → Fin 12

/-- The sum of numbers in a column of the dial stack -/
def columnSum (stack : DialStack) (column : Fin 12) : ℕ :=
  sorry

/-- Predicate that checks if all column sums have the same remainder mod 12 -/
def allColumnSumsEqualMod12 (stack : DialStack) : Prop :=
  ∀ i j : Fin 12, columnSum stack i % 12 = columnSum stack j % 12

/-- The minimum number of dials required for the Christmas tree to light up -/
theorem min_dials_for_lighting : 
  ∃ (n : ℕ), n = 12 ∧ 
  (∃ (stack : DialStack), (∀ i : ℕ, i < n → ∃ (dial : Fin 12 → Fin 12), stack i = dial) ∧ 
   allColumnSumsEqualMod12 stack) ∧
  (∀ (m : ℕ), m < n → 
   ∀ (stack : DialStack), (∀ i : ℕ, i < m → ∃ (dial : Fin 12 → Fin 12), stack i = dial) → 
   ¬allColumnSumsEqualMod12 stack) :=
sorry

end NUMINAMATH_CALUDE_min_dials_for_lighting_l3118_311842


namespace NUMINAMATH_CALUDE_lions_and_majestic_l3118_311883

-- Define the universe
variable (U : Type)

-- Define the predicates
variable (Lion : U → Prop)
variable (Majestic : U → Prop)
variable (Bird : U → Prop)

-- State the given conditions
variable (h1 : ∀ x, Lion x → Majestic x)
variable (h2 : ∀ x, Bird x → ¬Lion x)

-- Theorem to prove
theorem lions_and_majestic :
  (∀ x, Lion x → ¬Bird x) ∧ (∃ x, Majestic x ∧ ¬Bird x) :=
sorry

end NUMINAMATH_CALUDE_lions_and_majestic_l3118_311883


namespace NUMINAMATH_CALUDE_functional_polynomial_characterization_l3118_311825

/-- A polynomial that satisfies the given functional equation -/
def FunctionalPolynomial (p : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, 1 + p x = (p (x - 1) + p (x + 1)) / 2

theorem functional_polynomial_characterization :
  ∀ p : ℝ → ℝ, FunctionalPolynomial p →
  ∃ b c : ℝ, ∀ x : ℝ, p x = x^2 + b*x + c :=
sorry

end NUMINAMATH_CALUDE_functional_polynomial_characterization_l3118_311825


namespace NUMINAMATH_CALUDE_sam_initial_puppies_l3118_311800

/-- The number of puppies Sam gave away -/
def puppies_given : ℝ := 2.0

/-- The number of puppies Sam has now -/
def puppies_remaining : ℕ := 4

/-- The initial number of puppies Sam had -/
def initial_puppies : ℝ := puppies_given + puppies_remaining

theorem sam_initial_puppies : initial_puppies = 6.0 := by
  sorry

end NUMINAMATH_CALUDE_sam_initial_puppies_l3118_311800


namespace NUMINAMATH_CALUDE_wheel_rotation_l3118_311837

/-- Theorem: Rotation of a wheel in radians
  Given a wheel with radius 20 cm rotating counterclockwise,
  if a point on its circumference moves through an arc length of 40 cm,
  then the wheel has rotated 2 radians.
-/
theorem wheel_rotation (radius : ℝ) (arc_length : ℝ) (angle : ℝ) :
  radius = 20 →
  arc_length = 40 →
  angle = arc_length / radius →
  angle = 2 := by
sorry

end NUMINAMATH_CALUDE_wheel_rotation_l3118_311837


namespace NUMINAMATH_CALUDE_two_numbers_sum_and_digit_removal_l3118_311809

theorem two_numbers_sum_and_digit_removal (x y : ℕ) : 
  x + y = 2014 ∧ 
  3 * (x / 100) = y + 6 ∧ 
  x > y → 
  (x = 1963 ∧ y = 51) ∨ (x = 51 ∧ y = 1963) :=
by sorry

end NUMINAMATH_CALUDE_two_numbers_sum_and_digit_removal_l3118_311809


namespace NUMINAMATH_CALUDE_vegetable_load_weight_l3118_311801

/-- Calculates the total weight of a load of vegetables given the weight of a crate, 
    the weight of a carton, and the number of crates and cartons. -/
def totalWeight (crateWeight cartonWeight : ℕ) (numCrates numCartons : ℕ) : ℕ :=
  crateWeight * numCrates + cartonWeight * numCartons

/-- Proves that the total weight of a specific load of vegetables is 96 kilograms. -/
theorem vegetable_load_weight :
  totalWeight 4 3 12 16 = 96 := by
  sorry

end NUMINAMATH_CALUDE_vegetable_load_weight_l3118_311801


namespace NUMINAMATH_CALUDE_no_solution_implies_m_equals_two_l3118_311872

theorem no_solution_implies_m_equals_two (m : ℝ) : 
  (∀ x : ℝ, x ≠ 1 → (2 * x) / (x - 1) - 1 ≠ m / (x - 1)) → m = 2 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_implies_m_equals_two_l3118_311872


namespace NUMINAMATH_CALUDE_initial_gasoline_percentage_l3118_311893

/-- Proves that the initial gasoline percentage is 95% given the problem conditions --/
theorem initial_gasoline_percentage
  (initial_volume : ℝ)
  (initial_ethanol_percentage : ℝ)
  (desired_ethanol_percentage : ℝ)
  (added_ethanol : ℝ)
  (h1 : initial_volume = 54)
  (h2 : initial_ethanol_percentage = 0.05)
  (h3 : desired_ethanol_percentage = 0.10)
  (h4 : added_ethanol = 3)
  (h5 : initial_volume * initial_ethanol_percentage + added_ethanol = 
        (initial_volume + added_ethanol) * desired_ethanol_percentage) :
  1 - initial_ethanol_percentage = 0.95 := by
  sorry

end NUMINAMATH_CALUDE_initial_gasoline_percentage_l3118_311893


namespace NUMINAMATH_CALUDE_twentieth_bend_is_71_l3118_311836

/-- The function that returns the nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- The spiral arrangement of natural numbers where bends occur at prime numbers -/
def spiralBend (n : ℕ) : ℕ := nthPrime n

theorem twentieth_bend_is_71 : spiralBend 20 = 71 := by sorry

end NUMINAMATH_CALUDE_twentieth_bend_is_71_l3118_311836


namespace NUMINAMATH_CALUDE_new_ratio_after_adding_water_l3118_311897

/-- Given a mixture of alcohol and water with an initial ratio and known quantities,
    this theorem proves the new ratio after adding water. -/
theorem new_ratio_after_adding_water
  (initial_alcohol : ℝ)
  (initial_water : ℝ)
  (added_water : ℝ)
  (h1 : initial_alcohol / initial_water = 4 / 3)
  (h2 : initial_alcohol = 20)
  (h3 : added_water = 4) :
  initial_alcohol / (initial_water + added_water) = 20 / 19 := by
  sorry

end NUMINAMATH_CALUDE_new_ratio_after_adding_water_l3118_311897


namespace NUMINAMATH_CALUDE_product_of_separated_evens_l3118_311854

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

def circular_arrangement (n : ℕ) : Prop := n ≥ 1 ∧ n ≤ 16

theorem product_of_separated_evens (a d : ℕ) : 
  circular_arrangement a → 
  circular_arrangement d → 
  is_even a → 
  is_even d → 
  (∃ b c, circular_arrangement b ∧ circular_arrangement c ∧ 
    ((a < b ∧ b < c ∧ c < d) ∨ (d < a ∧ a < b ∧ b < c) ∨ 
     (c < d ∧ d < a ∧ a < b) ∨ (b < c ∧ c < d ∧ d < a))) →
  a * d = 120 :=
sorry

end NUMINAMATH_CALUDE_product_of_separated_evens_l3118_311854


namespace NUMINAMATH_CALUDE_square_perimeter_l3118_311855

theorem square_perimeter : ∀ (x₁ x₂ : ℝ),
  x₁^2 + 4*x₁ + 3 = 7 →
  x₂^2 + 4*x₂ + 3 = 7 →
  x₁ ≠ x₂ →
  4 * |x₂ - x₁| = 16 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_square_perimeter_l3118_311855


namespace NUMINAMATH_CALUDE_factorization_equality_l3118_311805

theorem factorization_equality (x y : ℝ) :
  (1 - x^2) * (1 - y^2) - 4*x*y = (x*y - 1 + x + y) * (x*y - 1 - x - y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3118_311805


namespace NUMINAMATH_CALUDE_intersection_sum_problem_l3118_311802

theorem intersection_sum_problem (digits : Finset ℕ) 
  (h_digits : digits.card = 6 ∧ digits ⊆ Finset.range 10 ∧ 1 ∈ digits)
  (vertical : Finset ℕ) (horizontal : Finset ℕ)
  (h_vert : vertical.card = 3 ∧ vertical ⊆ digits)
  (h_horiz : horizontal.card = 4 ∧ horizontal ⊆ digits)
  (h_intersect : (vertical ∩ horizontal).card = 1)
  (h_vert_sum : vertical.sum id = 25)
  (h_horiz_sum : horizontal.sum id = 14) :
  digits.sum id = 31 := by
sorry

end NUMINAMATH_CALUDE_intersection_sum_problem_l3118_311802


namespace NUMINAMATH_CALUDE_infinite_series_sum_l3118_311863

/-- The sum of the infinite series ∑(n=1 to ∞) (3n - 2) / (n(n + 1)(n + 3)) is equal to 55/12 -/
theorem infinite_series_sum : 
  (∑' n : ℕ, (3 * n - 2) / (n * (n + 1) * (n + 3))) = 55 / 12 :=
by sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l3118_311863


namespace NUMINAMATH_CALUDE_cyclic_fraction_product_l3118_311882

theorem cyclic_fraction_product (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h : (x + y) / z = (y + z) / x ∧ (y + z) / x = (z + x) / y) :
  ((x + y) * (y + z) * (z + x)) / (x * y * z) = -1 ∨
  ((x + y) * (y + z) * (z + x)) / (x * y * z) = 8 :=
by sorry

end NUMINAMATH_CALUDE_cyclic_fraction_product_l3118_311882


namespace NUMINAMATH_CALUDE_equation_transformation_l3118_311840

theorem equation_transformation (x y : ℝ) : x - 2 = y - 2 → x = y := by
  sorry

end NUMINAMATH_CALUDE_equation_transformation_l3118_311840


namespace NUMINAMATH_CALUDE_expression_equality_l3118_311889

theorem expression_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x^2 = 1 / y^2) :
  (x^2 - 1/x^2) * (y^2 + 1/y^2) = x^4 - y^4 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3118_311889


namespace NUMINAMATH_CALUDE_committees_with_president_count_l3118_311847

/-- The number of different five-student committees with an elected president
    that can be chosen from a group of ten students -/
def committees_with_president : ℕ :=
  (Nat.choose 10 5) * 5

/-- Theorem stating that the number of committees with a president is 1260 -/
theorem committees_with_president_count :
  committees_with_president = 1260 := by
  sorry

end NUMINAMATH_CALUDE_committees_with_president_count_l3118_311847


namespace NUMINAMATH_CALUDE_initial_stock_proof_l3118_311822

/-- The initial number of books in John's bookshop -/
def initial_books : ℕ := 1400

/-- The number of books sold over 5 days -/
def books_sold : ℕ := 402

/-- The percentage of books sold, expressed as a real number between 0 and 1 -/
def percentage_sold : ℝ := 0.2871428571428571

theorem initial_stock_proof :
  (books_sold : ℝ) / initial_books = percentage_sold :=
by sorry

end NUMINAMATH_CALUDE_initial_stock_proof_l3118_311822


namespace NUMINAMATH_CALUDE_variance_of_sick_cows_l3118_311830

/-- The variance of a binomial distribution with n trials and probability p --/
def binomial_variance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

/-- The number of cows in the pasture --/
def num_cows : ℕ := 10

/-- The incidence rate of the disease --/
def incidence_rate : ℝ := 0.02

/-- Theorem stating that the variance of the number of sick cows is 0.196 --/
theorem variance_of_sick_cows :
  binomial_variance num_cows incidence_rate = 0.196 := by
  sorry

end NUMINAMATH_CALUDE_variance_of_sick_cows_l3118_311830


namespace NUMINAMATH_CALUDE_f_properties_l3118_311817

def f (x : ℝ) : ℝ := x^3 - x

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧
  (∀ x, x < -Real.sqrt 3 / 3 → (deriv f) x > 0) ∧
  (∀ x, x > Real.sqrt 3 / 3 → (deriv f) x > 0) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3118_311817


namespace NUMINAMATH_CALUDE_infinite_points_in_S_l3118_311852

-- Define the set of points satisfying the conditions
def S : Set (ℚ × ℚ) :=
  {p | p.1 > 0 ∧ p.2 > 0 ∧ p.1 * p.2 ≤ 5}

-- Theorem statement
theorem infinite_points_in_S : Set.Infinite S := by
  sorry

end NUMINAMATH_CALUDE_infinite_points_in_S_l3118_311852


namespace NUMINAMATH_CALUDE_second_number_value_l3118_311862

theorem second_number_value (A B : ℝ) (h1 : A = 200) (h2 : 0.3 * A = 0.6 * B + 30) : B = 50 := by
  sorry

end NUMINAMATH_CALUDE_second_number_value_l3118_311862


namespace NUMINAMATH_CALUDE_blueberry_picking_relationships_l3118_311895

/-- Represents the relationship between y₁ and x for blueberry picking -/
def y₁ (x : ℝ) : ℝ := 60 + 18 * x

/-- Represents the relationship between y₂ and x for blueberry picking -/
def y₂ (x : ℝ) : ℝ := 150 + 15 * x

/-- Theorem stating the relationships between y₁, y₂, and x when blueberry picking amount exceeds 10 kg -/
theorem blueberry_picking_relationships (x : ℝ) (h : x > 10) :
  y₁ x = 60 + 18 * x ∧ y₂ x = 150 + 15 * x := by
  sorry

end NUMINAMATH_CALUDE_blueberry_picking_relationships_l3118_311895


namespace NUMINAMATH_CALUDE_cos_180_degrees_l3118_311829

theorem cos_180_degrees : Real.cos (π) = -1 := by
  sorry

end NUMINAMATH_CALUDE_cos_180_degrees_l3118_311829


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3118_311864

-- Define an isosceles triangle with sides a, b, and c
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isIsosceles : (a = b ∧ c ≠ a) ∨ (b = c ∧ a ≠ b) ∨ (a = c ∧ b ≠ a)
  validTriangle : a + b > c ∧ b + c > a ∧ a + c > b

-- Define the perimeter of a triangle
def perimeter (t : IsoscelesTriangle) : ℝ := t.a + t.b + t.c

-- Theorem statement
theorem isosceles_triangle_perimeter :
  ∀ t : IsoscelesTriangle,
    (t.a = 4 ∧ t.b = 7) ∨ (t.a = 7 ∧ t.b = 4) →
    perimeter t = 15 ∨ perimeter t = 18 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3118_311864


namespace NUMINAMATH_CALUDE_scientific_notation_570_million_l3118_311810

theorem scientific_notation_570_million :
  (570000000 : ℝ) = 5.7 * (10 : ℝ) ^ 8 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_570_million_l3118_311810


namespace NUMINAMATH_CALUDE_sqrt_27_div_3_eq_sqrt_3_l3118_311898

theorem sqrt_27_div_3_eq_sqrt_3 : Real.sqrt 27 / 3 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_27_div_3_eq_sqrt_3_l3118_311898


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3118_311851

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℚ) :
  arithmetic_sequence a →
  a 3 + a 11 = 3 →
  a 5 + a 6 + a 10 = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3118_311851


namespace NUMINAMATH_CALUDE_y_increase_for_x_increase_l3118_311812

/-- Given a line with the following properties:
    1. When x increases by 2 units, y increases by 5 units.
    2. The line passes through the point (1, 1).
    3. We consider an x-value increase of 8 units.
    
    This theorem proves that the y-value will increase by 20 units. -/
theorem y_increase_for_x_increase (slope : ℚ) (x_increase y_increase : ℚ) :
  slope = 5 / 2 →
  x_increase = 8 →
  y_increase = slope * x_increase →
  y_increase = 20 := by
  sorry

end NUMINAMATH_CALUDE_y_increase_for_x_increase_l3118_311812


namespace NUMINAMATH_CALUDE_sequence_sum_l3118_311831

theorem sequence_sum (P Q R S T U V : ℝ) : 
  S = 7 ∧ 
  P + Q + R = 27 ∧ 
  Q + R + S = 27 ∧ 
  R + S + T = 27 ∧ 
  S + T + U = 27 ∧ 
  T + U + V = 27 → 
  P + V = 0 := by sorry

end NUMINAMATH_CALUDE_sequence_sum_l3118_311831


namespace NUMINAMATH_CALUDE_min_value_expression_l3118_311841

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x / (x + 2 * y)) + (y / x) ≥ Real.sqrt 2 - 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3118_311841


namespace NUMINAMATH_CALUDE_difference_of_cubes_divisible_by_eight_l3118_311808

theorem difference_of_cubes_divisible_by_eight (a b : ℤ) : 
  ∃ k : ℤ, (2*a + 1)^3 - (2*b + 1)^3 = 8*k := by
  sorry

end NUMINAMATH_CALUDE_difference_of_cubes_divisible_by_eight_l3118_311808


namespace NUMINAMATH_CALUDE_weight_of_b_l3118_311880

/-- Given three weights a, b, and c, prove that b = 33 under the given conditions. -/
theorem weight_of_b (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →
  (a + b) / 2 = 41 →
  (b + c) / 2 = 43 →
  b = 33 := by
sorry

end NUMINAMATH_CALUDE_weight_of_b_l3118_311880


namespace NUMINAMATH_CALUDE_function_difference_l3118_311804

theorem function_difference (f : ℕ+ → ℕ+) 
  (h_mono : ∀ m n : ℕ+, m < n → f m < f n)
  (h_comp : ∀ n : ℕ+, f (f n) = 3 * n) :
  f 2202 - f 2022 = 510 := by
sorry

end NUMINAMATH_CALUDE_function_difference_l3118_311804


namespace NUMINAMATH_CALUDE_sqrt_six_minus_one_over_two_lt_one_l3118_311877

theorem sqrt_six_minus_one_over_two_lt_one : (Real.sqrt 6 - 1) / 2 < 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_six_minus_one_over_two_lt_one_l3118_311877


namespace NUMINAMATH_CALUDE_marble_sculpture_second_week_cut_l3118_311875

/-- Proves that the percentage of marble cut away in the second week is 20% --/
theorem marble_sculpture_second_week_cut (
  original_weight : ℝ)
  (first_week_cut_percent : ℝ)
  (third_week_cut_percent : ℝ)
  (final_weight : ℝ)
  (h1 : original_weight = 250)
  (h2 : first_week_cut_percent = 30)
  (h3 : third_week_cut_percent = 25)
  (h4 : final_weight = 105)
  : ∃ (second_week_cut_percent : ℝ),
    second_week_cut_percent = 20 ∧
    final_weight = original_weight *
      (1 - first_week_cut_percent / 100) *
      (1 - second_week_cut_percent / 100) *
      (1 - third_week_cut_percent / 100) :=
by sorry

end NUMINAMATH_CALUDE_marble_sculpture_second_week_cut_l3118_311875


namespace NUMINAMATH_CALUDE_pluto_orbit_scientific_notation_l3118_311806

/-- The radius of Pluto's orbit in kilometers -/
def pluto_orbit_radius : ℝ := 5900000000

/-- The scientific notation representation of Pluto's orbit radius -/
def pluto_orbit_scientific : ℝ := 5.9 * (10 ^ 9)

/-- Theorem stating that the radius of Pluto's orbit is equal to its scientific notation representation -/
theorem pluto_orbit_scientific_notation : pluto_orbit_radius = pluto_orbit_scientific := by
  sorry

end NUMINAMATH_CALUDE_pluto_orbit_scientific_notation_l3118_311806


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l3118_311853

theorem quadratic_roots_relation (p q n : ℝ) (r₁ r₂ : ℝ) : 
  (∀ x, x^2 + q*x + p = 0 ↔ x = r₁ ∨ x = r₂) →
  (∀ x, x^2 + p*x + n = 0 ↔ x = 3*r₁ ∨ x = 3*r₂) →
  p ≠ 0 → q ≠ 0 → n ≠ 0 →
  n / q = -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l3118_311853


namespace NUMINAMATH_CALUDE_road_repair_theorem_l3118_311846

/-- Represents the road repair scenario -/
structure RoadRepair where
  initial_workers : ℕ
  initial_days : ℕ
  worked_days : ℕ
  additional_workers : ℕ

/-- Calculates the number of additional days needed to complete the work -/
def additional_days_needed (repair : RoadRepair) : ℚ :=
  let total_work := repair.initial_workers * repair.initial_days
  let work_done := repair.initial_workers * repair.worked_days
  let remaining_work := total_work - work_done
  let new_workforce := repair.initial_workers + repair.additional_workers
  remaining_work / new_workforce

/-- The theorem stating that 6 additional days are needed to complete the work -/
theorem road_repair_theorem (repair : RoadRepair)
  (h1 : repair.initial_workers = 24)
  (h2 : repair.initial_days = 12)
  (h3 : repair.worked_days = 4)
  (h4 : repair.additional_workers = 8) :
  additional_days_needed repair = 6 := by
  sorry

#eval additional_days_needed ⟨24, 12, 4, 8⟩

end NUMINAMATH_CALUDE_road_repair_theorem_l3118_311846


namespace NUMINAMATH_CALUDE_max_gcd_thirteen_numbers_sum_1988_l3118_311867

theorem max_gcd_thirteen_numbers_sum_1988 (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ a₁₂ a₁₃ : ℕ) 
  (h_sum : a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ + a₁₂ + a₁₃ = 1988) :
  Nat.gcd a₁ (Nat.gcd a₂ (Nat.gcd a₃ (Nat.gcd a₄ (Nat.gcd a₅ (Nat.gcd a₆ (Nat.gcd a₇ (Nat.gcd a₈ (Nat.gcd a₉ (Nat.gcd a₁₀ (Nat.gcd a₁₁ (Nat.gcd a₁₂ a₁₃))))))))))) ≤ 142 :=
by
  sorry

end NUMINAMATH_CALUDE_max_gcd_thirteen_numbers_sum_1988_l3118_311867


namespace NUMINAMATH_CALUDE_smaller_square_area_l3118_311818

/-- The area of the smaller square formed by inscribing two right triangles in a larger square --/
theorem smaller_square_area (s : ℝ) (h : s = 4) : 
  let diagonal_smaller := s
  let side_smaller := diagonal_smaller / Real.sqrt 2
  side_smaller ^ 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_smaller_square_area_l3118_311818


namespace NUMINAMATH_CALUDE_solve_probability_problem_l3118_311890

/-- Given three independent events A, B, and C with their respective probabilities -/
def probability_problem (P_A P_B P_C : ℝ) : Prop :=
  0 ≤ P_A ∧ P_A ≤ 1 ∧
  0 ≤ P_B ∧ P_B ≤ 1 ∧
  0 ≤ P_C ∧ P_C ≤ 1 →
  -- All three events occur simultaneously
  P_A * P_B * P_C = 0.612 ∧
  -- At least two events do not occur
  (1 - P_A) * (1 - P_B) * P_C +
  (1 - P_A) * P_B * (1 - P_C) +
  P_A * (1 - P_B) * (1 - P_C) +
  (1 - P_A) * (1 - P_B) * (1 - P_C) = 0.059

/-- The theorem stating the solution to the probability problem -/
theorem solve_probability_problem :
  probability_problem 0.9 0.8 0.85 := by
  sorry


end NUMINAMATH_CALUDE_solve_probability_problem_l3118_311890


namespace NUMINAMATH_CALUDE_power_two_mod_seven_l3118_311894

theorem power_two_mod_seven : 2^2010 ≡ 1 [MOD 7] := by sorry

end NUMINAMATH_CALUDE_power_two_mod_seven_l3118_311894


namespace NUMINAMATH_CALUDE_degree_of_derivative_P_l3118_311823

/-- The polynomial we are working with -/
def P (x : ℝ) : ℝ := (x^2 + 1)^5 * (x^4 + 1)^2

/-- The degree of a polynomial -/
noncomputable def degree (p : ℝ → ℝ) : ℕ := sorry

/-- The derivative of a polynomial -/
noncomputable def derivative (p : ℝ → ℝ) : ℝ → ℝ := sorry

theorem degree_of_derivative_P :
  degree (derivative P) = 17 := by sorry

end NUMINAMATH_CALUDE_degree_of_derivative_P_l3118_311823


namespace NUMINAMATH_CALUDE_total_earnings_proof_l3118_311824

structure LaundryShop where
  regular_rate : ℝ
  delicate_rate : ℝ
  bulky_rate : ℝ

def day_earnings (shop : LaundryShop) (regular_kilos delicate_kilos : ℝ) (bulky_items : ℕ) (delicate_discount : ℝ := 0) : ℝ :=
  shop.regular_rate * regular_kilos +
  shop.delicate_rate * delicate_kilos * (1 - delicate_discount) +
  shop.bulky_rate * (bulky_items : ℝ)

theorem total_earnings_proof (shop : LaundryShop)
  (h1 : shop.regular_rate = 3)
  (h2 : shop.delicate_rate = 4)
  (h3 : shop.bulky_rate = 5)
  (h4 : day_earnings shop 7 4 2 = 47)
  (h5 : day_earnings shop 10 6 3 = 69)
  (h6 : day_earnings shop 20 4 0 0.2 = 72.8) :
  day_earnings shop 7 4 2 + day_earnings shop 10 6 3 + day_earnings shop 20 4 0 0.2 = 188.8 := by
  sorry

#eval day_earnings ⟨3, 4, 5⟩ 7 4 2 + day_earnings ⟨3, 4, 5⟩ 10 6 3 + day_earnings ⟨3, 4, 5⟩ 20 4 0 0.2

end NUMINAMATH_CALUDE_total_earnings_proof_l3118_311824


namespace NUMINAMATH_CALUDE_solution_mixture_problem_l3118_311859

theorem solution_mixture_problem (x : ℝ) :
  -- Solution 1 composition
  x + 80 = 100 →
  -- Solution 2 composition
  45 + 55 = 100 →
  -- Mixture composition (50% each solution)
  (x + 45) / 2 + (80 + 55) / 2 = 100 →
  -- Mixture contains 67.5% carbonated water
  (80 + 55) / 2 = 67.5 →
  -- Conclusion: Solution 1 is 20% lemonade
  x = 20 := by
sorry

end NUMINAMATH_CALUDE_solution_mixture_problem_l3118_311859


namespace NUMINAMATH_CALUDE_apple_count_theorem_l3118_311839

/-- The number of apples originally on the tree -/
def original_apples : ℕ := 9

/-- The number of apples picked from the tree -/
def picked_apples : ℕ := 2

/-- The number of apples remaining on the tree -/
def remaining_apples : ℕ := 7

/-- Theorem stating that the original number of apples is equal to
    the sum of remaining and picked apples -/
theorem apple_count_theorem :
  original_apples = remaining_apples + picked_apples :=
by sorry

end NUMINAMATH_CALUDE_apple_count_theorem_l3118_311839


namespace NUMINAMATH_CALUDE_polynomial_equality_implies_sum_of_squares_l3118_311857

theorem polynomial_equality_implies_sum_of_squares (a b c d e f : ℤ) :
  (∀ x : ℝ, 8 * x^3 + 125 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) →
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 767 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_implies_sum_of_squares_l3118_311857


namespace NUMINAMATH_CALUDE_f_max_at_seven_l3118_311821

/-- The quadratic function we're analyzing -/
def f (y : ℝ) : ℝ := y^2 - 14*y + 24

/-- The theorem stating that f achieves its maximum at y = 7 -/
theorem f_max_at_seven :
  ∀ y : ℝ, f y ≤ f 7 := by
  sorry

end NUMINAMATH_CALUDE_f_max_at_seven_l3118_311821


namespace NUMINAMATH_CALUDE_travel_days_is_22_l3118_311887

/-- Represents the travel data for a person over a period of days -/
structure TravelData where
  /-- Total number of days -/
  total_days : ℕ
  /-- Days traveled by metro in the morning and taxi in the evening -/
  metro_morning_taxi_evening : ℕ
  /-- Days traveled by taxi in the morning and metro in the evening -/
  taxi_morning_metro_evening : ℕ
  /-- Days traveled by metro in both morning and evening -/
  metro_both : ℕ
  /-- The sum of metro trips in the morning equals 11 -/
  morning_metro_sum : metro_morning_taxi_evening + metro_both = 11
  /-- The sum of metro trips in the evening equals 21 -/
  evening_metro_sum : taxi_morning_metro_evening + metro_both = 21
  /-- The sum of taxi trips (either morning or evening) equals 12 -/
  taxi_sum : metro_morning_taxi_evening + taxi_morning_metro_evening = 12
  /-- The total days is the sum of all trip types -/
  total_days_sum : total_days = metro_morning_taxi_evening + taxi_morning_metro_evening + metro_both

/-- Theorem stating that given the travel conditions, the total number of days is 22 -/
theorem travel_days_is_22 (data : TravelData) : data.total_days = 22 := by
  sorry

end NUMINAMATH_CALUDE_travel_days_is_22_l3118_311887


namespace NUMINAMATH_CALUDE_tom_helicopter_rental_cost_l3118_311865

/-- The total cost for renting a helicopter -/
def helicopter_rental_cost (hours_per_day : ℕ) (days : ℕ) (hourly_rate : ℕ) : ℕ :=
  hours_per_day * days * hourly_rate

/-- Theorem stating the total cost for Tom's helicopter rental -/
theorem tom_helicopter_rental_cost :
  helicopter_rental_cost 2 3 75 = 450 := by
  sorry

end NUMINAMATH_CALUDE_tom_helicopter_rental_cost_l3118_311865


namespace NUMINAMATH_CALUDE_trig_expression_equality_l3118_311816

theorem trig_expression_equality : 
  (Real.sin (30 * π / 180) * Real.cos (24 * π / 180) + 
   Real.cos (150 * π / 180) * Real.cos (84 * π / 180)) / 
  (Real.sin (34 * π / 180) * Real.cos (16 * π / 180) + 
   Real.cos (146 * π / 180) * Real.cos (76 * π / 180)) = 
  Real.sin (51 * π / 180) / Real.sin (55 * π / 180) := by
sorry

end NUMINAMATH_CALUDE_trig_expression_equality_l3118_311816


namespace NUMINAMATH_CALUDE_max_value_of_expression_max_value_achievable_l3118_311845

theorem max_value_of_expression (x : ℝ) :
  x^6 / (x^12 + 4*x^9 - 6*x^6 + 16*x^3 + 64) ≤ 1/26 :=
by sorry

theorem max_value_achievable :
  ∃ x : ℝ, x^6 / (x^12 + 4*x^9 - 6*x^6 + 16*x^3 + 64) = 1/26 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_max_value_achievable_l3118_311845


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l3118_311899

-- Define the variables as positive real numbers
variable (x y z : ℝ) 

-- Define the hypothesis that x, y, and z are positive
variable (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)

-- Define the main theorem
theorem cyclic_sum_inequality :
  let f (a b c : ℝ) := Real.sqrt (a / (b + c)) * Real.sqrt ((a * b + a * c + b^2 + c^2) / (b^2 + c^2))
  f x y z + f y z x + f z x y ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l3118_311899


namespace NUMINAMATH_CALUDE_backpacks_at_swap_meet_l3118_311814

/-- Proves that the number of backpacks sold at the swap meet is 17 -/
theorem backpacks_at_swap_meet : 
  ∀ (x : ℕ), 
    (48 : ℕ) = x + 10 + (48 - x - 10) → -- Total backpacks
    576 + 442 = 18 * x + 25 * 10 + 22 * (48 - x - 10) → -- Revenue equation
    x = 17 := by
  sorry

end NUMINAMATH_CALUDE_backpacks_at_swap_meet_l3118_311814


namespace NUMINAMATH_CALUDE_sum_prod_nonzero_digits_equals_46_pow_2009_l3118_311861

/-- The number of digits in the problem -/
def n : ℕ := 2009

/-- Calculate the product of non-zero digits for a given natural number -/
def prod_nonzero_digits (k : ℕ) : ℕ := sorry

/-- Sum of products of non-zero digits for integers from 1 to 10^n -/
def sum_prod_nonzero_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating the sum of products of non-zero digits for integers from 1 to 10^2009 -/
theorem sum_prod_nonzero_digits_equals_46_pow_2009 :
  sum_prod_nonzero_digits n = 46^n := by sorry

end NUMINAMATH_CALUDE_sum_prod_nonzero_digits_equals_46_pow_2009_l3118_311861


namespace NUMINAMATH_CALUDE_binomial_expansion_102_l3118_311871

theorem binomial_expansion_102 : 
  102^4 - 4 * 102^3 + 6 * 102^2 - 4 * 102 + 1 = 104040401 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_102_l3118_311871


namespace NUMINAMATH_CALUDE_valid_triplets_l3118_311881

def is_valid_triplet (m n p : ℕ) : Prop :=
  m > 0 ∧ n > 0 ∧ Nat.Prime p ∧ (Nat.choose m 3 - 4 = p^n)

theorem valid_triplets :
  ∀ m n p : ℕ, is_valid_triplet m n p ↔ (m = 7 ∧ n = 1 ∧ p = 31) ∨ (m = 6 ∧ n = 4 ∧ p = 2) :=
sorry

end NUMINAMATH_CALUDE_valid_triplets_l3118_311881


namespace NUMINAMATH_CALUDE_abs_x_minus_one_lt_two_necessary_not_sufficient_for_x_times_x_minus_three_lt_zero_l3118_311858

theorem abs_x_minus_one_lt_two_necessary_not_sufficient_for_x_times_x_minus_three_lt_zero :
  (∃ x : ℝ, |x - 1| < 2 ∧ ¬(x * (x - 3) < 0)) ∧
  (∀ x : ℝ, x * (x - 3) < 0 → |x - 1| < 2) :=
by sorry

end NUMINAMATH_CALUDE_abs_x_minus_one_lt_two_necessary_not_sufficient_for_x_times_x_minus_three_lt_zero_l3118_311858


namespace NUMINAMATH_CALUDE_factorial_300_trailing_zeros_l3118_311848

def trailing_zeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

theorem factorial_300_trailing_zeros :
  trailing_zeros 300 = 74 := by sorry

end NUMINAMATH_CALUDE_factorial_300_trailing_zeros_l3118_311848


namespace NUMINAMATH_CALUDE_A_infinite_l3118_311826

/-- A function that represents z = n^4 + a -/
def z (n a : ℕ) : ℕ := n^4 + a

/-- The set of natural numbers a such that z(n, a) is composite for all n -/
def A : Set ℕ := {a : ℕ | ∀ n : ℕ, ¬ Nat.Prime (z n a)}

/-- Theorem stating that A is infinite -/
theorem A_infinite : Set.Infinite A := by sorry

end NUMINAMATH_CALUDE_A_infinite_l3118_311826


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l3118_311838

/-- Two-dimensional vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Check if two vectors are parallel -/
def areParallel (v w : Vector2D) : Prop :=
  v.x * w.y = v.y * w.x

theorem parallel_vectors_m_value (m : ℝ) :
  let a : Vector2D := ⟨2, m⟩
  let b : Vector2D := ⟨m, 2⟩
  areParallel a b → m = 2 ∨ m = -2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l3118_311838


namespace NUMINAMATH_CALUDE_factor_sum_l3118_311828

theorem factor_sum (P Q : ℝ) : 
  (∃ b c : ℝ, (X^2 - 4*X + 8) * (X^2 + b*X + c) = X^4 + P*X^2 + Q) → 
  P + Q = 64 :=
by sorry

end NUMINAMATH_CALUDE_factor_sum_l3118_311828


namespace NUMINAMATH_CALUDE_go_complexity_vs_universe_atoms_l3118_311870

/-- The upper limit of the state space complexity of Go -/
def M : ℝ := 3^361

/-- The total number of atoms of ordinary matter in the observable universe -/
def N : ℝ := 10^80

/-- The logarithm base 10 of 3 -/
def lg3 : ℝ := 0.48

/-- Theorem stating that M/N is approximately equal to 10^93 -/
theorem go_complexity_vs_universe_atoms : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |M / N - 10^93| < ε := by sorry

end NUMINAMATH_CALUDE_go_complexity_vs_universe_atoms_l3118_311870


namespace NUMINAMATH_CALUDE_train_speed_l3118_311807

/-- Proves that the speed of a train is 45 km/hr given specific conditions --/
theorem train_speed (train_length bridge_length crossing_time : ℝ) 
  (h1 : train_length = 150)
  (h2 : bridge_length = 225)
  (h3 : crossing_time = 30)
  (h4 : (1 : ℝ) / 3.6 = 1 / 3.6) : -- Conversion factor
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3118_311807


namespace NUMINAMATH_CALUDE_circle_to_hyperbola_l3118_311843

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 5*y + 4 = 0

-- Define the intersection points of circle C with coordinate axes
def intersection_points (C : (ℝ → ℝ → Prop)) : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), (x = 0 ∨ y = 0) ∧ C x y ∧ p = (x, y)}

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop := (y - 1)^2 / 1 - x^2 / 15 = 1

-- Theorem statement
theorem circle_to_hyperbola :
  ∀ (focus vertex : ℝ × ℝ),
    focus ∈ intersection_points circle_C →
    vertex ∈ intersection_points circle_C →
    focus ≠ vertex →
    (∀ x y : ℝ, hyperbola_equation x y ↔
      ∃ (a b : ℝ),
        a > 0 ∧ b > 0 ∧
        (y - vertex.2)^2 / a^2 - (x - vertex.1)^2 / b^2 = 1 ∧
        (focus.1 - vertex.1)^2 + (focus.2 - vertex.2)^2 = a^2 + b^2) :=
sorry

end NUMINAMATH_CALUDE_circle_to_hyperbola_l3118_311843


namespace NUMINAMATH_CALUDE_quadratic_equation_a_value_l3118_311820

/-- Given that (a + 1)x^2 + (a^2 + 1) + 8x = 9 is a quadratic equation in terms of x, prove that a = 2√2 -/
theorem quadratic_equation_a_value (a : ℝ) :
  (∀ x : ℝ, ∃ b c : ℝ, (a + 1) * x^2 + (a^2 + 1) + 8 * x = 9 ∧ 
   (a + 1) * x^2 + 8 * x + (a^2 - 8) = 0 ∧
   (a + 1) ≠ 0) →
  a = 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_a_value_l3118_311820


namespace NUMINAMATH_CALUDE_smallest_divisible_by_15_16_18_l3118_311811

theorem smallest_divisible_by_15_16_18 : 
  ∃ n : ℕ+, (∀ m : ℕ+, (15 ∣ m) ∧ (16 ∣ m) ∧ (18 ∣ m) → n ≤ m) ∧ 
             (15 ∣ n) ∧ (16 ∣ n) ∧ (18 ∣ n) :=
by
  use 720
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_15_16_18_l3118_311811


namespace NUMINAMATH_CALUDE_sequence_on_line_geometric_l3118_311878

/-- Given a sequence {a_n} where (n, a_n) is on the line 2x - y + 1 = 0,
    if a_1, a_4, and a_m form a geometric sequence, then m = 13. -/
theorem sequence_on_line_geometric (a : ℕ → ℝ) :
  (∀ n, 2 * n - a n + 1 = 0) →
  (∃ r, a 4 = a 1 * r ∧ a m = a 4 * r) →
  m = 13 :=
by sorry

end NUMINAMATH_CALUDE_sequence_on_line_geometric_l3118_311878


namespace NUMINAMATH_CALUDE_odd_even_function_inequalities_l3118_311892

-- Define the properties of functions f and g
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x
def is_increasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x < f y
def coincide_nonneg (f g : ℝ → ℝ) : Prop := ∀ x, x ≥ 0 → f x = g x

-- State the theorem
theorem odd_even_function_inequalities
  (f g : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_even : is_even g)
  (h_incr : is_increasing f)
  (h_coinc : coincide_nonneg f g)
  {a b : ℝ}
  (h_ab : a > b)
  (h_b_pos : b > 0) :
  (f b - f (-a) > g a - g (-b)) ∧
  (f a - f (-b) > g b - g (-a)) :=
by sorry

end NUMINAMATH_CALUDE_odd_even_function_inequalities_l3118_311892


namespace NUMINAMATH_CALUDE_problem_statement_l3118_311896

theorem problem_statement (a b c d e : ℕ+) : 
  a * b * c * d * e = 362880 →
  a * b + a + b = 728 →
  b * c + b + c = 342 →
  c * d + c + d = 464 →
  d * e + d + e = 780 →
  (a : ℤ) - (e : ℤ) = 172 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3118_311896


namespace NUMINAMATH_CALUDE_sqrt_factorial_five_squared_l3118_311891

theorem sqrt_factorial_five_squared (n : ℕ) : n = 5 → Real.sqrt ((n.factorial : ℝ) * n.factorial) = 120 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_factorial_five_squared_l3118_311891


namespace NUMINAMATH_CALUDE_toy_poodle_height_l3118_311886

/-- Heights of different poodle types in inches -/
structure PoodleHeights where
  standard : ℝ
  miniature : ℝ
  toy : ℝ
  moyen : ℝ
  klein : ℝ

/-- Conversion factor from inches to centimeters -/
def inch_to_cm : ℝ := 2.54

/-- Given conditions for poodle heights -/
def poodle_conditions (h : PoodleHeights) : Prop :=
  h.standard = h.miniature + 8.5 ∧
  h.miniature = h.toy + 6.25 ∧
  h.moyen = h.standard - 3.75 ∧
  h.moyen = h.toy + 4.75 ∧
  h.klein = (h.toy + h.moyen) / 2 ∧
  h.standard = 28

theorem toy_poodle_height (h : PoodleHeights) 
  (hcond : poodle_conditions h) : 
  h.toy * inch_to_cm = 33.655 := by
  sorry

end NUMINAMATH_CALUDE_toy_poodle_height_l3118_311886


namespace NUMINAMATH_CALUDE_fish_count_l3118_311879

theorem fish_count (num_bowls : ℕ) (fish_per_bowl : ℕ) (h1 : num_bowls = 261) (h2 : fish_per_bowl = 23) :
  num_bowls * fish_per_bowl = 6003 := by
  sorry

end NUMINAMATH_CALUDE_fish_count_l3118_311879


namespace NUMINAMATH_CALUDE_coords_of_A_wrt_origin_l3118_311866

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The origin of the Cartesian coordinate system -/
def origin : Point := ⟨0, 0⟩

/-- The coordinates of a point with respect to the origin -/
def coordsWrtOrigin (p : Point) : ℝ × ℝ := (p.x, p.y)

/-- Theorem: The coordinates of point A(-1,3) with respect to the origin are (-1,3) -/
theorem coords_of_A_wrt_origin :
  let A : Point := ⟨-1, 3⟩
  coordsWrtOrigin A = (-1, 3) := by sorry

end NUMINAMATH_CALUDE_coords_of_A_wrt_origin_l3118_311866


namespace NUMINAMATH_CALUDE_two_std_dev_below_mean_l3118_311827

/-- A normal distribution with given mean and standard deviation -/
structure NormalDistribution where
  mean : ℝ
  std_dev : ℝ
  std_dev_pos : 0 < std_dev

/-- The value that is exactly n standard deviations less than the mean -/
def value_n_std_dev_below (d : NormalDistribution) (n : ℝ) : ℝ :=
  d.mean - n * d.std_dev

/-- Theorem: For a normal distribution with mean 14.0 and standard deviation 1.5,
    the value that is exactly 2 standard deviations less than the mean is 11.0 -/
theorem two_std_dev_below_mean :
  ∃ (d : NormalDistribution),
    d.mean = 14.0 ∧
    d.std_dev = 1.5 ∧
    value_n_std_dev_below d 2 = 11.0 := by
  sorry

end NUMINAMATH_CALUDE_two_std_dev_below_mean_l3118_311827


namespace NUMINAMATH_CALUDE_quadratic_prime_values_l3118_311833

theorem quadratic_prime_values (p : ℕ) (hp : p > 1) :
  ∀ x : ℕ, 0 ≤ x ∧ x < p →
    (Nat.Prime (x^2 - x + p) ↔ (x = 0 ∨ x = 1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_prime_values_l3118_311833


namespace NUMINAMATH_CALUDE_net_amount_calculation_l3118_311856

/-- Calculate the net amount received from selling puppies --/
def net_amount_from_puppies (luna_puppies stella_puppies : ℕ)
                            (luna_sold stella_sold : ℕ)
                            (luna_price stella_price : ℕ)
                            (luna_cost stella_cost : ℕ) : ℕ :=
  let luna_revenue := luna_sold * luna_price
  let stella_revenue := stella_sold * stella_price
  let luna_expenses := luna_puppies * luna_cost
  let stella_expenses := stella_puppies * stella_cost
  (luna_revenue + stella_revenue) - (luna_expenses + stella_expenses)

theorem net_amount_calculation :
  net_amount_from_puppies 10 14 8 10 200 250 80 90 = 2040 := by
  sorry

end NUMINAMATH_CALUDE_net_amount_calculation_l3118_311856


namespace NUMINAMATH_CALUDE_f_properties_l3118_311849

def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x

theorem f_properties :
  (∃ (x : ℝ), -2 < x ∧ x < 2 ∧ f x = 5 ∧ ∀ (y : ℝ), -2 < y ∧ y < 2 → f y ≤ 5) ∧
  (∀ (m : ℝ), ∃ (x : ℝ), -2 < x ∧ x < 2 ∧ f x < m) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3118_311849


namespace NUMINAMATH_CALUDE_cab_driver_income_l3118_311819

def cab_driver_problem (day1 day2 day4 day5 : ℕ) (average : ℕ) : Prop :=
  let total := 5 * average
  let known_sum := day1 + day2 + day4 + day5
  let day3 := total - known_sum
  (day1 = 200) ∧ (day2 = 150) ∧ (day4 = 400) ∧ (day5 = 500) ∧ (average = 400) → day3 = 750

theorem cab_driver_income :
  cab_driver_problem 200 150 400 500 400 := by
  sorry

end NUMINAMATH_CALUDE_cab_driver_income_l3118_311819


namespace NUMINAMATH_CALUDE_unique_solution_system_l3118_311873

theorem unique_solution_system (x y z : ℝ) : 
  x^3 = 3*x - 12*y + 50 ∧ 
  y^3 = 12*y + 3*z - 2 ∧ 
  z^3 = 27*z + 27*x → 
  x = 2 ∧ y = 4 ∧ z = 6 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l3118_311873


namespace NUMINAMATH_CALUDE_max_square_triangle_area_ratio_l3118_311803

/-- A triangle with vertices A, B, and C. -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A square with vertices X, Y, Z, and V. -/
structure Square where
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  V : ℝ × ℝ

/-- The area of a triangle. -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- The area of a square. -/
def squareArea (s : Square) : ℝ := sorry

/-- Predicate to check if a point is on a line segment. -/
def isOnSegment (p : ℝ × ℝ) (a : ℝ × ℝ) (b : ℝ × ℝ) : Prop := sorry

/-- Predicate to check if two line segments are parallel. -/
def areParallel (a1 : ℝ × ℝ) (b1 : ℝ × ℝ) (a2 : ℝ × ℝ) (b2 : ℝ × ℝ) : Prop := sorry

/-- The main theorem stating the maximum area ratio. -/
theorem max_square_triangle_area_ratio 
  (t : Triangle) 
  (s : Square) 
  (h1 : isOnSegment s.X t.A t.B)
  (h2 : isOnSegment s.Y t.B t.C)
  (h3 : isOnSegment s.Z t.C t.A)
  (h4 : isOnSegment s.V t.A t.C)
  (h5 : areParallel s.V s.Z t.A t.B) :
  squareArea s / triangleArea t ≤ 1/2 := by sorry

end NUMINAMATH_CALUDE_max_square_triangle_area_ratio_l3118_311803


namespace NUMINAMATH_CALUDE_john_pushups_l3118_311874

def zachary_pushups : ℕ := 51
def david_pushups_difference : ℕ := 22
def john_pushups_difference : ℕ := 4

theorem john_pushups : 
  zachary_pushups + david_pushups_difference - john_pushups_difference = 69 :=
by
  sorry

end NUMINAMATH_CALUDE_john_pushups_l3118_311874


namespace NUMINAMATH_CALUDE_min_value_of_product_l3118_311860

-- Define the quadratic function f
noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the theorem
theorem min_value_of_product (a b c : ℝ) (x₁ x₂ x₃ : ℝ) :
  a ≠ 0 →
  f a b c (-1) = 0 →
  (∀ x : ℝ, f a b c x ≥ x) →
  (∀ x : ℝ, 0 < x → x < 2 → f a b c x ≤ (x + 1)^2 / 4) →
  0 < x₁ → x₁ < 2 →
  0 < x₂ → x₂ < 2 →
  0 < x₃ → x₃ < 2 →
  1 / x₁ + 1 / x₂ + 1 / x₃ = 3 →
  ∃ (m : ℝ), m = 1 ∧ ∀ y₁ y₂ y₃ : ℝ,
    0 < y₁ → y₁ < 2 →
    0 < y₂ → y₂ < 2 →
    0 < y₃ → y₃ < 2 →
    1 / y₁ + 1 / y₂ + 1 / y₃ = 3 →
    m ≤ f a b c y₁ * f a b c y₂ * f a b c y₃ :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_of_product_l3118_311860


namespace NUMINAMATH_CALUDE_not_general_term_formula_l3118_311815

def alternating_sequence (n : ℕ) : ℤ := (-1)^(n + 1)

theorem not_general_term_formula :
  ∃ n : ℕ, ((-1 : ℤ)^n ≠ alternating_sequence n) ∧
  ((-1 : ℤ)^(n + 1) = alternating_sequence n) ∧
  ((-1 : ℤ)^(n - 1) = alternating_sequence n) ∧
  (if n % 2 = 0 then -1 else 1 : ℤ) = alternating_sequence n :=
sorry

end NUMINAMATH_CALUDE_not_general_term_formula_l3118_311815


namespace NUMINAMATH_CALUDE_r_squared_perfect_fit_l3118_311844

/-- Linear regression model with zero error -/
structure LinearRegressionModel where
  n : ℕ
  x : Fin n → ℝ
  y : Fin n → ℝ
  a : ℝ
  b : ℝ
  h : ∀ i, y i = b * x i + a

/-- Coefficient of determination (R-squared) -/
def r_squared (model : LinearRegressionModel) : ℝ :=
  sorry

/-- Theorem: R-squared equals 1 for a perfect fit linear regression model -/
theorem r_squared_perfect_fit (model : LinearRegressionModel) :
  r_squared model = 1 :=
sorry

end NUMINAMATH_CALUDE_r_squared_perfect_fit_l3118_311844


namespace NUMINAMATH_CALUDE_computer_store_optimal_solution_l3118_311869

/-- Represents the profit optimization problem for a computer store. -/
def ComputerStoreProblem (total_computers : ℕ) (profit_A profit_B : ℕ) : Prop :=
  ∃ (x : ℕ) (y : ℤ),
    -- Total number of computers is fixed
    x + (total_computers - x) = total_computers ∧
    -- Profit calculation
    y = -100 * x + 50000 ∧
    -- Constraint on type B computers
    (total_computers - x) ≤ 3 * x ∧
    -- x is the optimal number of type A computers
    ∀ (x' : ℕ), x' ≠ x →
      (-100 * x' + 50000 : ℤ) ≤ (-100 * x + 50000 : ℤ) ∧
    -- Maximum profit is achieved
    y = 47500

/-- Theorem stating the existence of an optimal solution for the computer store problem. -/
theorem computer_store_optimal_solution :
  ComputerStoreProblem 100 400 500 :=
sorry

end NUMINAMATH_CALUDE_computer_store_optimal_solution_l3118_311869


namespace NUMINAMATH_CALUDE_library_book_selection_l3118_311868

theorem library_book_selection (math_books : Nat) (literature_books : Nat) (english_books : Nat) :
  math_books = 3 →
  literature_books = 5 →
  english_books = 8 →
  math_books + literature_books + english_books = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_library_book_selection_l3118_311868


namespace NUMINAMATH_CALUDE_mr_c_net_loss_l3118_311813

/-- Represents the value of a house and its transactions -/
structure HouseTransaction where
  initial_value : ℝ
  first_sale_loss_percent : ℝ
  second_sale_gain_percent : ℝ
  additional_tax : ℝ

/-- Calculates the net loss for Mr. C after two transactions -/
def net_loss (t : HouseTransaction) : ℝ :=
  let first_sale_price := t.initial_value * (1 - t.first_sale_loss_percent)
  let second_sale_price := first_sale_price * (1 + t.second_sale_gain_percent) + t.additional_tax
  second_sale_price - t.initial_value

/-- Theorem stating that Mr. C's net loss is $1560 -/
theorem mr_c_net_loss :
  let t : HouseTransaction := {
    initial_value := 8000,
    first_sale_loss_percent := 0.15,
    second_sale_gain_percent := 0.2,
    additional_tax := 200
  }
  net_loss t = 1560 := by
  sorry

end NUMINAMATH_CALUDE_mr_c_net_loss_l3118_311813


namespace NUMINAMATH_CALUDE_odd_red_faces_count_6_6_2_l3118_311888

/-- Represents a rectangular block with given dimensions -/
structure Block where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents a cube with a certain number of red faces -/
structure Cube where
  redFaces : ℕ

/-- Function to calculate the number of cubes with odd red faces -/
def oddRedFacesCount (b : Block) : ℕ :=
  -- Implementation details omitted
  sorry

/-- Theorem stating the number of cubes with odd red faces in the given block -/
theorem odd_red_faces_count_6_6_2 :
  let block := Block.mk 6 6 2
  oddRedFacesCount block = 40 := by
  sorry

end NUMINAMATH_CALUDE_odd_red_faces_count_6_6_2_l3118_311888


namespace NUMINAMATH_CALUDE_fish_pond_population_l3118_311832

/-- The number of fish initially tagged and returned to the pond -/
def initial_tagged : ℕ := 80

/-- The number of fish caught in the second catch -/
def second_catch : ℕ := 80

/-- The number of tagged fish found in the second catch -/
def tagged_in_second : ℕ := 2

/-- The approximate total number of fish in the pond -/
def total_fish : ℕ := 3200

/-- Theorem stating that the given conditions lead to the approximate number of fish in the pond -/
theorem fish_pond_population :
  (initial_tagged : ℚ) / total_fish = (tagged_in_second : ℚ) / second_catch :=
by sorry

end NUMINAMATH_CALUDE_fish_pond_population_l3118_311832


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l3118_311835

theorem fraction_product_simplification :
  (3 : ℚ) / 4 * (4 : ℚ) / 5 * (5 : ℚ) / 6 * (6 : ℚ) / 7 * (7 : ℚ) / 8 = (3 : ℚ) / 8 :=
by sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l3118_311835


namespace NUMINAMATH_CALUDE_sin_80_minus_sin_20_over_cos_20_l3118_311885

theorem sin_80_minus_sin_20_over_cos_20 :
  (2 * Real.sin (80 * π / 180) - Real.sin (20 * π / 180)) / Real.cos (20 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_80_minus_sin_20_over_cos_20_l3118_311885
