import Mathlib

namespace NUMINAMATH_CALUDE_child_money_distribution_l326_32667

/-- Prove that for three children with shares in the ratio 2:3:4, 
    where the second child's share is $300, the total amount is $900. -/
theorem child_money_distribution (a b c : ℕ) : 
  a + b + c = 9 ∧ 
  2 * b = 3 * a ∧ 
  4 * b = 3 * c ∧ 
  b = 300 → 
  a + b + c = 900 := by
sorry

end NUMINAMATH_CALUDE_child_money_distribution_l326_32667


namespace NUMINAMATH_CALUDE_linear_function_eight_value_l326_32632

/-- A function satisfying f(x + y) = f(x) + f(y) for all real x and y -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y

theorem linear_function_eight_value
    (f : ℝ → ℝ)
    (h1 : LinearFunction f)
    (h2 : f 7 = 8) :
    f 8 = 64 / 7 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_eight_value_l326_32632


namespace NUMINAMATH_CALUDE_diagonal_FH_range_l326_32643

/-- Represents a quadrilateral with integer side lengths and diagonals -/
structure Quadrilateral where
  EF : ℕ
  FG : ℕ
  GH : ℕ
  HE : ℕ
  EH : ℕ
  FH : ℕ

/-- The specific quadrilateral from the problem -/
def specificQuad : Quadrilateral where
  EF := 7
  FG := 13
  GH := 7
  HE := 20
  EH := 0  -- We don't know the exact value, but it's an integer
  FH := 0  -- This is what we're trying to prove

theorem diagonal_FH_range (q : Quadrilateral) (h : q = specificQuad) : 
  14 ≤ q.FH ∧ q.FH ≤ 19 := by
  sorry

#check diagonal_FH_range

end NUMINAMATH_CALUDE_diagonal_FH_range_l326_32643


namespace NUMINAMATH_CALUDE_pet_shop_ducks_l326_32624

theorem pet_shop_ducks (total : ℕ) (cats : ℕ) (ducks : ℕ) (parrots : ℕ) : 
  cats = 56 →
  ducks = total / 12 →
  ducks = (ducks + parrots) / 4 →
  total = cats + ducks + parrots →
  ducks = 7 := by
sorry

end NUMINAMATH_CALUDE_pet_shop_ducks_l326_32624


namespace NUMINAMATH_CALUDE_decimal_6_to_binary_l326_32660

def binary_representation (n : ℕ) : List Bool :=
  sorry

theorem decimal_6_to_binary :
  binary_representation 6 = [true, true, false] :=
sorry

end NUMINAMATH_CALUDE_decimal_6_to_binary_l326_32660


namespace NUMINAMATH_CALUDE_money_distribution_l326_32664

theorem money_distribution (w x y z : ℝ) (h1 : w = 375) 
  (h2 : x = 6 * w) (h3 : y = 2 * w) (h4 : z = 4 * w) : 
  x - y = 1500 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l326_32664


namespace NUMINAMATH_CALUDE_sqrt_3600_equals_60_l326_32652

theorem sqrt_3600_equals_60 : Real.sqrt 3600 = 60 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3600_equals_60_l326_32652


namespace NUMINAMATH_CALUDE_parallelepiped_volume_l326_32683

def vector1 : ℝ × ℝ × ℝ := (3, 4, 5)
def vector2 (m : ℝ) : ℝ × ℝ × ℝ := (2, m, 3)
def vector3 (m : ℝ) : ℝ × ℝ × ℝ := (2, 3, m)

def volume (a b c : ℝ × ℝ × ℝ) : ℝ :=
  let (a1, a2, a3) := a
  let (b1, b2, b3) := b
  let (c1, c2, c3) := c
  abs (a1 * (b2 * c3 - b3 * c2) - a2 * (b1 * c3 - b3 * c1) + a3 * (b1 * c2 - b2 * c1))

theorem parallelepiped_volume (m : ℝ) :
  m > 0 →
  volume vector1 (vector2 m) (vector3 m) = 20 →
  m = (9 + Real.sqrt 249) / 6 :=
by sorry

end NUMINAMATH_CALUDE_parallelepiped_volume_l326_32683


namespace NUMINAMATH_CALUDE_star_polygon_n_value_l326_32681

/-- A regular star polygon with n points, where each point has two angles. -/
structure StarPolygon where
  n : ℕ
  angle_C : ℝ
  angle_D : ℝ

/-- Properties of the star polygon -/
def is_valid_star_polygon (s : StarPolygon) : Prop :=
  s.n > 0 ∧
  s.angle_C > 0 ∧
  s.angle_D > 0 ∧
  s.angle_C = s.angle_D - 15 ∧
  s.n * 15 = 360

theorem star_polygon_n_value (s : StarPolygon) (h : is_valid_star_polygon s) : s.n = 24 := by
  sorry

#check star_polygon_n_value

end NUMINAMATH_CALUDE_star_polygon_n_value_l326_32681


namespace NUMINAMATH_CALUDE_triple_consecutive_primes_l326_32674

theorem triple_consecutive_primes (p : ℤ) : 
  (Nat.Prime p.natAbs ∧ Nat.Prime (p + 2).natAbs ∧ Nat.Prime (p + 4).natAbs) ↔ p = 3 :=
sorry

end NUMINAMATH_CALUDE_triple_consecutive_primes_l326_32674


namespace NUMINAMATH_CALUDE_ryan_load_is_correct_l326_32689

/-- The number of packages Sarah's trucks can carry in one load -/
def sarah_load : ℕ := 18

/-- The total number of packages shipped by both services -/
def total_packages : ℕ := 198

/-- Predicate to check if a number is a valid load size for Ryan's trucks -/
def is_valid_ryan_load (n : ℕ) : Prop :=
  n > sarah_load ∧ total_packages % n = 0

/-- The number of packages Ryan's trucks can carry in one load -/
def ryan_load : ℕ := 22

theorem ryan_load_is_correct : 
  is_valid_ryan_load ryan_load ∧ 
  ∀ (n : ℕ), is_valid_ryan_load n → n ≥ ryan_load :=
sorry

end NUMINAMATH_CALUDE_ryan_load_is_correct_l326_32689


namespace NUMINAMATH_CALUDE_intersection_points_are_correct_l326_32676

/-- The set of intersection points of the given lines -/
def intersection_points : Set (ℝ × ℝ) :=
  {p | (∃ (x y : ℝ), p = (x, y) ∧
    (3 * x - 2 * y = 12 ∨
     2 * x + 4 * y = 8 ∨
     -5 * x + 15 * y = 30 ∨
     x = -3) ∧
    (3 * x - 2 * y = 12 ∨
     2 * x + 4 * y = 8 ∨
     -5 * x + 15 * y = 30 ∨
     x = -3) ∧
    (3 * x - 2 * y = 12 ∨
     2 * x + 4 * y = 8 ∨
     -5 * x + 15 * y = 30 ∨
     x = -3))}

/-- The theorem stating that the intersection points are (4, 0) and (-3, -10.5) -/
theorem intersection_points_are_correct :
  intersection_points = {(4, 0), (-3, -10.5)} :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_are_correct_l326_32676


namespace NUMINAMATH_CALUDE_f_max_min_on_interval_l326_32666

-- Define the function f(x)
def f (x : ℝ) := x^3 - 12*x

-- Define the interval
def interval : Set ℝ := { x | -3 ≤ x ∧ x ≤ 3 }

-- Theorem statement
theorem f_max_min_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ interval, f x ≤ max) ∧
    (∃ x ∈ interval, f x = max) ∧
    (∀ x ∈ interval, min ≤ f x) ∧
    (∃ x ∈ interval, f x = min) ∧
    max = 16 ∧ min = -16 := by
  sorry

end NUMINAMATH_CALUDE_f_max_min_on_interval_l326_32666


namespace NUMINAMATH_CALUDE_annabelle_allowance_l326_32618

/-- Proves that Annabelle's weekly allowance is $30 given the problem conditions -/
theorem annabelle_allowance :
  ∀ A : ℚ, (1/3 : ℚ) * A + 8 + 12 = A → A = 30 := by
  sorry

end NUMINAMATH_CALUDE_annabelle_allowance_l326_32618


namespace NUMINAMATH_CALUDE_school_function_participants_l326_32685

theorem school_function_participants (boys girls : ℕ) 
  (h1 : 2 * (boys - girls) = 3 * 400)
  (h2 : 3 * girls = 4 * 150)
  (h3 : 2 * boys + 3 * girls = 3 * 550) :
  boys + girls = 800 := by
  sorry

end NUMINAMATH_CALUDE_school_function_participants_l326_32685


namespace NUMINAMATH_CALUDE_age_ratio_problem_l326_32616

/-- Given two people A and B, where:
    1. The ratio of their present ages is 6:3
    2. The ratio between A's age at a certain point in the past and B's age at a certain point in the future is the same as their present ratio
    3. The ratio between A's age 4 years hence and B's age 4 years ago is 5
    Prove that the ratio between A's age 4 years ago and B's age 4 years hence is 1:1 -/
theorem age_ratio_problem (a b : ℕ) (h1 : a = 2 * b) 
  (h2 : ∀ (x y : ℤ), a + x = 2 * (b + y))
  (h3 : (a + 4) / (b - 4 : ℚ) = 5) :
  (a - 4 : ℚ) / (b + 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l326_32616


namespace NUMINAMATH_CALUDE_fraction_meaningful_l326_32699

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = (1 - x) / (1 + x)) ↔ x ≠ -1 := by sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l326_32699


namespace NUMINAMATH_CALUDE_quadratic_properties_l326_32633

def f (x : ℝ) := x^2 - 2*x - 1

theorem quadratic_properties :
  (∃ (x y : ℝ), (x, y) = (1, -2) ∧ ∀ t, f t ≥ f x) ∧
  (∃ (x₁ x₂ : ℝ), x₁ = 1 + Real.sqrt 2 ∧ 
                  x₂ = 1 - Real.sqrt 2 ∧ 
                  f x₁ = 0 ∧ 
                  f x₂ = 0 ∧
                  ∀ x, f x = 0 → x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_properties_l326_32633


namespace NUMINAMATH_CALUDE_expression_value_l326_32657

theorem expression_value (a b c d x : ℝ) 
  (h1 : a * b = 1)
  (h2 : c + d = 0)
  (h3 : |x| = 3) :
  2 * x^2 - (a * b - c - d) + |a * b + 3| = 21 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l326_32657


namespace NUMINAMATH_CALUDE_modular_inverse_of_5_mod_26_l326_32620

theorem modular_inverse_of_5_mod_26 : ∃ x : ℕ, x ≤ 25 ∧ (5 * x) % 26 = 1 :=
by
  use 21
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_5_mod_26_l326_32620


namespace NUMINAMATH_CALUDE_total_books_is_182_l326_32630

/-- The number of books each person has -/
def joan_books : ℕ := 10
def tom_books : ℕ := 38
def lisa_books : ℕ := 27
def steve_books : ℕ := 45
def kim_books : ℕ := 14
def alex_books : ℕ := 48

/-- The total number of books -/
def total_books : ℕ := joan_books + tom_books + lisa_books + steve_books + kim_books + alex_books

/-- Theorem stating that the total number of books is 182 -/
theorem total_books_is_182 : total_books = 182 := by
  sorry

end NUMINAMATH_CALUDE_total_books_is_182_l326_32630


namespace NUMINAMATH_CALUDE_pushups_sum_is_350_l326_32672

/-- The number of push-ups done by Zachary, David, and Emily -/
def total_pushups (zachary_pushups : ℕ) (david_extra : ℕ) : ℕ :=
  let david_pushups := zachary_pushups + david_extra
  let emily_pushups := 2 * david_pushups
  zachary_pushups + david_pushups + emily_pushups

/-- Theorem stating that the total number of push-ups is 350 -/
theorem pushups_sum_is_350 : total_pushups 44 58 = 350 := by
  sorry

end NUMINAMATH_CALUDE_pushups_sum_is_350_l326_32672


namespace NUMINAMATH_CALUDE_rectangle_area_l326_32675

/-- A rectangle with perimeter 36 and length three times its width has area 60.75 -/
theorem rectangle_area (width : ℝ) (length : ℝ) : 
  (width + length) * 2 = 36 →
  length = 3 * width →
  width * length = 60.75 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l326_32675


namespace NUMINAMATH_CALUDE_zoo_open_hours_proof_l326_32610

/-- The number of hours the zoo is open in one day -/
def zoo_open_hours : ℕ := 8

/-- The number of new visitors entering the zoo every hour -/
def visitors_per_hour : ℕ := 50

/-- The percentage of total visitors who go to the gorilla exhibit -/
def gorilla_exhibit_percentage : ℚ := 80 / 100

/-- The number of visitors who go to the gorilla exhibit in one day -/
def gorilla_exhibit_visitors : ℕ := 320

/-- Theorem stating that the zoo is open for 8 hours given the conditions -/
theorem zoo_open_hours_proof :
  zoo_open_hours * visitors_per_hour * gorilla_exhibit_percentage = gorilla_exhibit_visitors :=
by sorry

end NUMINAMATH_CALUDE_zoo_open_hours_proof_l326_32610


namespace NUMINAMATH_CALUDE_least_positive_integer_divisible_by_four_primes_l326_32682

theorem least_positive_integer_divisible_by_four_primes : 
  ∃ (n : ℕ), (n > 0) ∧ 
  (∃ (p₁ p₂ p₃ p₄ : ℕ), Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
    n % p₁ = 0 ∧ n % p₂ = 0 ∧ n % p₃ = 0 ∧ n % p₄ = 0) ∧
  (∀ (m : ℕ), m > 0 → 
    (∃ (q₁ q₂ q₃ q₄ : ℕ), Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄ ∧ 
      q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₃ ≠ q₄ ∧
      m % q₁ = 0 ∧ m % q₂ = 0 ∧ m % q₃ = 0 ∧ m % q₄ = 0) → 
    m ≥ n) ∧
  n = 210 := by
sorry

end NUMINAMATH_CALUDE_least_positive_integer_divisible_by_four_primes_l326_32682


namespace NUMINAMATH_CALUDE_square_of_negative_product_l326_32617

theorem square_of_negative_product (a b : ℝ) : (-a * b)^2 = a^2 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_product_l326_32617


namespace NUMINAMATH_CALUDE_monster_feeding_interval_l326_32621

/-- Represents the monster's feeding pattern over 300 years -/
structure MonsterFeedingPattern where
  interval : ℕ  -- The interval at which the monster rises
  total_consumed : ℕ  -- Total number of people consumed over 300 years
  first_ship : ℕ  -- Number of people on the first ship

/-- Theorem stating the conditions and the conclusion about the monster's feeding interval -/
theorem monster_feeding_interval (m : MonsterFeedingPattern) : 
  m.total_consumed = 847 ∧ 
  m.first_ship = 121 ∧ 
  m.total_consumed = m.first_ship + 2 * m.first_ship + 4 * m.first_ship → 
  m.interval = 100 := by
  sorry

end NUMINAMATH_CALUDE_monster_feeding_interval_l326_32621


namespace NUMINAMATH_CALUDE_unique_solution_exists_l326_32627

theorem unique_solution_exists : ∃! x : ℝ, 0.6667 * x - 10 = 0.25 * x := by sorry

end NUMINAMATH_CALUDE_unique_solution_exists_l326_32627


namespace NUMINAMATH_CALUDE_sqrt_two_plus_one_times_sqrt_two_minus_one_equals_one_l326_32629

theorem sqrt_two_plus_one_times_sqrt_two_minus_one_equals_one :
  (Real.sqrt 2 + 1) * (Real.sqrt 2 - 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_plus_one_times_sqrt_two_minus_one_equals_one_l326_32629


namespace NUMINAMATH_CALUDE_binomial_equation_unique_solution_l326_32673

theorem binomial_equation_unique_solution :
  ∃! n : ℕ, (Nat.choose 25 n + Nat.choose 25 12 = Nat.choose 26 13) ∧ n = 13 := by
  sorry

end NUMINAMATH_CALUDE_binomial_equation_unique_solution_l326_32673


namespace NUMINAMATH_CALUDE_constant_term_expansion_l326_32678

/-- The constant term in the expansion of (2x + 1/x - 1)^5 is -161 -/
theorem constant_term_expansion : 
  let f : ℝ → ℝ := λ x => (2*x + 1/x - 1)^5
  ∃ g : ℝ → ℝ, ∀ x ≠ 0, f x = g x + (-161) := by
  sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l326_32678


namespace NUMINAMATH_CALUDE_double_base_exponent_l326_32623

theorem double_base_exponent (a b y : ℝ) (ha : a > 0) (hb : b > 0) (hy : y > 0) :
  (2 * a)^(2 * b) = a^(2 * b) * y^(2 * b) → y = 2 := by
  sorry

end NUMINAMATH_CALUDE_double_base_exponent_l326_32623


namespace NUMINAMATH_CALUDE_range_of_a_l326_32609

-- Define the quadratic function
def f (x : ℝ) : ℝ := -x^2 + 2*x + 2

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc a 3, f x ∈ Set.Icc (-1) 3) ∧
  (∀ y ∈ Set.Icc (-1) 3, ∃ x ∈ Set.Icc a 3, f x = y) →
  a ∈ Set.Icc (-1) 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l326_32609


namespace NUMINAMATH_CALUDE_expression_value_l326_32644

theorem expression_value : (2^2 * 5) / (8 * 10) * (3 * 4 * 8) / (2 * 5 * 3) = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l326_32644


namespace NUMINAMATH_CALUDE_max_stickers_purchasable_l326_32679

theorem max_stickers_purchasable (budget : ℚ) (unit_cost : ℚ) : 
  budget = 10 → unit_cost = 3/4 → 
  (∃ (n : ℕ), n * unit_cost ≤ budget ∧ 
    ∀ (m : ℕ), m * unit_cost ≤ budget → m ≤ n) → 
  (∃ (max_stickers : ℕ), max_stickers = 13) :=
by sorry

end NUMINAMATH_CALUDE_max_stickers_purchasable_l326_32679


namespace NUMINAMATH_CALUDE_sixth_term_of_arithmetic_sequence_l326_32670

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sixth_term_of_arithmetic_sequence (a : ℕ → ℚ) 
  (h_arith : arithmetic_sequence a) 
  (h_3 : a 3 = 4) 
  (h_7 : a 7 = 10) : 
  a 6 = 17/2 := by
sorry

end NUMINAMATH_CALUDE_sixth_term_of_arithmetic_sequence_l326_32670


namespace NUMINAMATH_CALUDE_chef_cherry_pies_l326_32604

/-- Given a chef with an initial number of cherries, some used cherries, and a fixed number of cherries required per pie, 
    this function calculates the maximum number of additional pies that can be made with the remaining cherries. -/
def max_additional_pies (initial_cherries used_cherries cherries_per_pie : ℕ) : ℕ :=
  (initial_cherries - used_cherries) / cherries_per_pie

/-- Theorem stating that for the given values, the maximum number of additional pies is 4. -/
theorem chef_cherry_pies : max_additional_pies 500 350 35 = 4 := by
  sorry

end NUMINAMATH_CALUDE_chef_cherry_pies_l326_32604


namespace NUMINAMATH_CALUDE_cosine_sum_problem_l326_32615

theorem cosine_sum_problem (α : Real) 
  (h : Real.sin (π / 2 + α) = 1 / 3) : 
  Real.cos (2 * α) + Real.cos α = -4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_problem_l326_32615


namespace NUMINAMATH_CALUDE_journey_distance_l326_32693

theorem journey_distance (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) :
  total_time = 10 ∧ speed1 = 21 ∧ speed2 = 24 →
  ∃ (distance : ℝ), distance = 224 ∧
    total_time = (distance / 2) / speed1 + (distance / 2) / speed2 := by
  sorry

end NUMINAMATH_CALUDE_journey_distance_l326_32693


namespace NUMINAMATH_CALUDE_sum_of_squared_pairs_of_roots_l326_32612

theorem sum_of_squared_pairs_of_roots (p q r : ℝ) : 
  (p + q + r = 15) → (p * q + q * r + r * p = 25) → 
  (p + q)^2 + (q + r)^2 + (r + p)^2 = 400 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squared_pairs_of_roots_l326_32612


namespace NUMINAMATH_CALUDE_cone_properties_l326_32606

/-- Properties of a cone with specific dimensions -/
theorem cone_properties (r h l : ℝ) : 
  r = 2 → -- base radius is 2
  π * l = 2 * π * r → -- lateral surface unfolds to a semicircle
  l^2 = r^2 + h^2 → -- Pythagorean theorem
  (l = 4 ∧ (1/3) * π * r^2 * h = (8 * Real.sqrt 3 / 3) * π) := by sorry

end NUMINAMATH_CALUDE_cone_properties_l326_32606


namespace NUMINAMATH_CALUDE_max_value_of_a_l326_32655

theorem max_value_of_a (a b c d : ℝ) 
  (h1 : b + c + d = 3 - a) 
  (h2 : 2 * b^2 + 3 * c^2 + 6 * d^2 = 5 - a^2) : 
  ∃ (max_a : ℝ), max_a = 2 ∧ ∀ a', (∃ b' c' d', b' + c' + d' = 3 - a' ∧ 
    2 * b'^2 + 3 * c'^2 + 6 * d'^2 = 5 - a'^2) → a' ≤ max_a :=
sorry

end NUMINAMATH_CALUDE_max_value_of_a_l326_32655


namespace NUMINAMATH_CALUDE_hyperbola_center_l326_32637

/-- The center of a hyperbola is the midpoint of its foci -/
theorem hyperbola_center (f1 f2 : ℝ × ℝ) (c : ℝ × ℝ) : 
  f1 = (6, -2) → f2 = (10, 6) → c = ((f1.1 + f2.1) / 2, (f1.2 + f2.2) / 2) → c = (8, 2) := by
  sorry

#check hyperbola_center

end NUMINAMATH_CALUDE_hyperbola_center_l326_32637


namespace NUMINAMATH_CALUDE_line_properties_l326_32669

def line_vector (t : ℝ) : ℝ × ℝ × ℝ := sorry

theorem line_properties :
  (∃ line_vector : ℝ → ℝ × ℝ × ℝ,
    (line_vector (-2) = (2, 4, 7)) ∧
    (line_vector 1 = (-1, 0, -3))) →
  (∃ line_vector : ℝ → ℝ × ℝ × ℝ,
    (line_vector (-2) = (2, 4, 7)) ∧
    (line_vector 1 = (-1, 0, -3)) ∧
    (line_vector (-1) = (1, 8, 5)) ∧
    (¬ ∃ t : ℝ, line_vector t = (3, 604, -6))) :=
by sorry

end NUMINAMATH_CALUDE_line_properties_l326_32669


namespace NUMINAMATH_CALUDE_coin_toss_probabilities_l326_32671

/-- The probability of getting two heads when tossing two coins -/
def P₁ : ℝ := sorry

/-- The probability of getting two tails when tossing two coins -/
def P₂ : ℝ := sorry

/-- The probability of getting one head and one tail when tossing two coins -/
def P₃ : ℝ := sorry

/-- All probabilities are non-negative -/
axiom prob_nonneg : 0 ≤ P₁ ∧ 0 ≤ P₂ ∧ 0 ≤ P₃

/-- The sum of all probabilities is 1 -/
axiom prob_sum_one : P₁ + P₂ + P₃ = 1

/-- Theorem stating the relationships between P₁, P₂, and P₃ -/
theorem coin_toss_probabilities :
  (P₁ + P₂ = P₃) ∧
  (P₁ + P₂ + P₃ = 1) ∧
  (P₃ = 2*P₁) ∧
  (P₃ = 2*P₂) := by
  sorry

end NUMINAMATH_CALUDE_coin_toss_probabilities_l326_32671


namespace NUMINAMATH_CALUDE_salary_increase_l326_32686

theorem salary_increase (num_employees : ℕ) (avg_salary : ℚ) (manager_salary : ℚ) :
  num_employees = 20 ∧ avg_salary = 1500 ∧ manager_salary = 14100 →
  (((num_employees : ℚ) * avg_salary + manager_salary) / (num_employees + 1 : ℚ)) - avg_salary = 600 := by
sorry

end NUMINAMATH_CALUDE_salary_increase_l326_32686


namespace NUMINAMATH_CALUDE_triangle_cosine_theorem_l326_32663

theorem triangle_cosine_theorem (A B C : ℝ) (h_triangle : A + B + C = π) 
  (h_cos_A : Real.cos A = 4/5) (h_cos_B : Real.cos B = 7/25) : 
  Real.cos C = 44/125 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_theorem_l326_32663


namespace NUMINAMATH_CALUDE_rectangular_prism_surface_area_l326_32642

/-- The surface area of a rectangular prism -/
def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Theorem: The surface area of a rectangular prism with dimensions 10, 6, and 5 is 280 -/
theorem rectangular_prism_surface_area :
  surface_area 10 6 5 = 280 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_surface_area_l326_32642


namespace NUMINAMATH_CALUDE_pascal_triangle_first_25_rows_sum_l326_32688

/-- The number of elements in the nth row of Pascal's Triangle -/
def pascal_row_elements (n : ℕ) : ℕ := n + 1

/-- The sum of elements in the first n rows of Pascal's Triangle -/
def pascal_triangle_sum (n : ℕ) : ℕ := (n + 1) * (n + 2) / 2

theorem pascal_triangle_first_25_rows_sum :
  pascal_triangle_sum 24 = 325 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_first_25_rows_sum_l326_32688


namespace NUMINAMATH_CALUDE_cyclist_distance_l326_32649

theorem cyclist_distance (travel_time : ℝ) (car_distance : ℝ) (speed_difference : ℝ) :
  travel_time = 8 →
  car_distance = 48 →
  speed_difference = 5 →
  let car_speed := car_distance / travel_time
  let cyclist_speed := car_speed - speed_difference
  cyclist_speed * travel_time = 8 := by sorry

end NUMINAMATH_CALUDE_cyclist_distance_l326_32649


namespace NUMINAMATH_CALUDE_files_deleted_l326_32668

/-- Given the initial number of files and the number of files left after deletion,
    prove that the number of files deleted is 14. -/
theorem files_deleted (initial_files : ℕ) (files_left : ℕ) 
  (h1 : initial_files = 21) 
  (h2 : files_left = 7) : 
  initial_files - files_left = 14 := by
  sorry


end NUMINAMATH_CALUDE_files_deleted_l326_32668


namespace NUMINAMATH_CALUDE_remaining_cooking_times_l326_32653

/-- Calculates the remaining cooking time in seconds for a food item -/
def remainingCookingTime (recommendedTime actualTime : ℕ) : ℕ :=
  (recommendedTime - actualTime) * 60

/-- Represents the cooking times for different food items -/
structure CookingTimes where
  frenchFries : ℕ
  chickenNuggets : ℕ
  mozzarellaSticks : ℕ

/-- Theorem stating the remaining cooking times for each food item -/
theorem remaining_cooking_times 
  (recommended : CookingTimes) 
  (actual : CookingTimes) : 
  remainingCookingTime recommended.frenchFries actual.frenchFries = 600 ∧
  remainingCookingTime recommended.chickenNuggets actual.chickenNuggets = 780 ∧
  remainingCookingTime recommended.mozzarellaSticks actual.mozzarellaSticks = 300 :=
by
  sorry

#check remaining_cooking_times (CookingTimes.mk 12 18 8) (CookingTimes.mk 2 5 3)

end NUMINAMATH_CALUDE_remaining_cooking_times_l326_32653


namespace NUMINAMATH_CALUDE_sum_geq_sqrt_products_l326_32602

theorem sum_geq_sqrt_products (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a + b + c ≥ Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a) := by
  sorry

end NUMINAMATH_CALUDE_sum_geq_sqrt_products_l326_32602


namespace NUMINAMATH_CALUDE_cricket_players_l326_32656

theorem cricket_players (total : ℕ) (basketball : ℕ) (both : ℕ) 
  (h1 : total = 880) 
  (h2 : basketball = 600) 
  (h3 : both = 220) : 
  total = (total - basketball + both) + basketball - both :=
by sorry

end NUMINAMATH_CALUDE_cricket_players_l326_32656


namespace NUMINAMATH_CALUDE_yellow_ball_probability_l326_32640

-- Define the number of black and yellow balls
def num_black_balls : ℕ := 4
def num_yellow_balls : ℕ := 6

-- Define the total number of balls
def total_balls : ℕ := num_black_balls + num_yellow_balls

-- Define the probability of drawing a yellow ball
def prob_yellow : ℚ := num_yellow_balls / total_balls

-- Theorem statement
theorem yellow_ball_probability : prob_yellow = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_yellow_ball_probability_l326_32640


namespace NUMINAMATH_CALUDE_carson_age_carson_age_real_l326_32646

/-- Given the ages of Aunt Anna, Maria, and Carson, prove Carson's age -/
theorem carson_age (anna_age : ℕ) (maria_age : ℕ) (carson_age : ℕ) : 
  anna_age = 60 →
  maria_age = 2 * anna_age / 3 →
  carson_age = maria_age - 7 →
  carson_age = 33 := by
sorry

/-- Alternative formulation using real numbers for more precise calculations -/
theorem carson_age_real (anna_age : ℝ) (maria_age : ℝ) (carson_age : ℝ) : 
  anna_age = 60 →
  maria_age = 2 / 3 * anna_age →
  carson_age = maria_age - 7 →
  carson_age = 33 := by
sorry

end NUMINAMATH_CALUDE_carson_age_carson_age_real_l326_32646


namespace NUMINAMATH_CALUDE_perpendicular_diameter_bisects_chord_equal_central_angles_equal_arcs_equal_chords_equal_arcs_equal_arcs_equal_central_angles_l326_32638

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a chord
structure Chord (c : Circle) where
  endpoint1 : ℝ × ℝ
  endpoint2 : ℝ × ℝ

-- Define an arc
structure Arc (c : Circle) where
  startPoint : ℝ × ℝ
  endPoint : ℝ × ℝ

-- Define a central angle
def CentralAngle (c : Circle) (a : Arc c) : ℝ := sorry

-- Define the length of a chord
def chordLength (c : Circle) (ch : Chord c) : ℝ := sorry

-- Define the length of an arc
def arcLength (c : Circle) (a : Arc c) : ℝ := sorry

-- Define a diameter
def Diameter (c : Circle) := Chord c

-- Define perpendicularity between a diameter and a chord
def isPerpendicular (d : Diameter c) (ch : Chord c) : Prop := sorry

-- Define bisection of a chord
def bisectsChord (d : Diameter c) (ch : Chord c) : Prop := sorry

-- Theorem 1: A diameter perpendicular to a chord bisects the chord
theorem perpendicular_diameter_bisects_chord (c : Circle) (d : Diameter c) (ch : Chord c) :
  isPerpendicular d ch → bisectsChord d ch := sorry

-- Theorem 2: Equal central angles correspond to equal arcs
theorem equal_central_angles_equal_arcs (c : Circle) (a1 a2 : Arc c) :
  CentralAngle c a1 = CentralAngle c a2 → arcLength c a1 = arcLength c a2 := sorry

-- Theorem 3: Equal chords correspond to equal arcs
theorem equal_chords_equal_arcs (c : Circle) (ch1 ch2 : Chord c) (a1 a2 : Arc c) :
  chordLength c ch1 = chordLength c ch2 → arcLength c a1 = arcLength c a2 := sorry

-- Theorem 4: Equal arcs correspond to equal central angles
theorem equal_arcs_equal_central_angles (c : Circle) (a1 a2 : Arc c) :
  arcLength c a1 = arcLength c a2 → CentralAngle c a1 = CentralAngle c a2 := sorry

end NUMINAMATH_CALUDE_perpendicular_diameter_bisects_chord_equal_central_angles_equal_arcs_equal_chords_equal_arcs_equal_arcs_equal_central_angles_l326_32638


namespace NUMINAMATH_CALUDE_quadratic_roots_theorem_l326_32641

theorem quadratic_roots_theorem (a b c : ℝ) :
  (∃ x y : ℝ, x^2 - (a+b)*x + (a*b-c^2) = 0 ∧ y^2 - (a+b)*y + (a*b-c^2) = 0) ∧
  (∃! x : ℝ, x^2 - (a+b)*x + (a*b-c^2) = 0 ↔ a = b ∧ c = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_theorem_l326_32641


namespace NUMINAMATH_CALUDE_root_product_negative_l326_32625

-- Define a monotonic function
def Monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

-- State the theorem
theorem root_product_negative
  (f : ℝ → ℝ) (a x₁ x₂ : ℝ)
  (h_monotonic : Monotonic f)
  (h_root : f a = 0)
  (h_order : x₁ < a ∧ a < x₂) :
  f x₁ * f x₂ < 0 :=
sorry

end NUMINAMATH_CALUDE_root_product_negative_l326_32625


namespace NUMINAMATH_CALUDE_right_triangles_bc_length_l326_32608

/-- Given two right triangles ABC and ABD where B is vertically above A,
    and C and D lie on the horizontal axis, prove that if AC = 20, AD = 45,
    and BD = 13, then BC = 47. -/
theorem right_triangles_bc_length (A B C D : ℝ × ℝ) : 
  (∃ k : ℝ, B = (A.1, A.2 + k)) →  -- B is vertically above A
  (C.2 = A.2 ∧ D.2 = A.2) →        -- C and D lie on the horizontal axis
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2 →  -- ABC is a right triangle
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = (D.1 - A.1)^2 + (D.2 - A.2)^2 →  -- ABD is a right triangle
  Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 20 →  -- AC = 20
  Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2) = 45 →  -- AD = 45
  Real.sqrt ((D.1 - B.1)^2 + (D.2 - B.2)^2) = 13 →  -- BD = 13
  Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 47    -- BC = 47
  := by sorry


end NUMINAMATH_CALUDE_right_triangles_bc_length_l326_32608


namespace NUMINAMATH_CALUDE_fraction_transformation_l326_32634

theorem fraction_transformation (x : ℚ) : (3 - 2*x) / (5 + x) = 1/2 → x = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_transformation_l326_32634


namespace NUMINAMATH_CALUDE_boxes_to_fill_l326_32600

theorem boxes_to_fill (total_boxes filled_boxes : ℝ) 
  (h1 : total_boxes = 25.75) 
  (h2 : filled_boxes = 17.5) : 
  total_boxes - filled_boxes = 8.25 := by
  sorry

end NUMINAMATH_CALUDE_boxes_to_fill_l326_32600


namespace NUMINAMATH_CALUDE_kristy_ate_two_cookies_l326_32611

def cookies_problem (total_baked : ℕ) (brother_took : ℕ) (friend1_took : ℕ) (friend2_took : ℕ) (friend3_took : ℕ) (cookies_left : ℕ) : Prop :=
  total_baked = 22 ∧
  brother_took = 1 ∧
  friend1_took = 3 ∧
  friend2_took = 5 ∧
  friend3_took = 5 ∧
  cookies_left = 6 ∧
  total_baked - (brother_took + friend1_took + friend2_took + friend3_took + cookies_left) = 2

theorem kristy_ate_two_cookies :
  ∀ (total_baked brother_took friend1_took friend2_took friend3_took cookies_left : ℕ),
  cookies_problem total_baked brother_took friend1_took friend2_took friend3_took cookies_left →
  total_baked - (brother_took + friend1_took + friend2_took + friend3_took + cookies_left) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_kristy_ate_two_cookies_l326_32611


namespace NUMINAMATH_CALUDE_not_in_range_iff_c_in_interval_l326_32691

/-- The function g(x) defined in terms of c -/
def g (c : ℝ) (x : ℝ) : ℝ := x^2 + c*x + 3

/-- Theorem stating that -3 is not in the range of g(x) iff c ∈ (-2√6, 2√6) -/
theorem not_in_range_iff_c_in_interval (c : ℝ) : 
  (∀ x : ℝ, g c x ≠ -3) ↔ c ∈ Set.Ioo (-2 * Real.sqrt 6) (2 * Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_not_in_range_iff_c_in_interval_l326_32691


namespace NUMINAMATH_CALUDE_solve_equation_l326_32650

theorem solve_equation (x : ℝ) : (x^3 * 6^2) / 432 = 144 → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l326_32650


namespace NUMINAMATH_CALUDE_island_distance_l326_32665

theorem island_distance (n : ℝ) : 
  let a := 8*n
  let b := 5*n
  let c := 7*n
  let α := 60 * π / 180
  a^2 + b^2 - 2*a*b*Real.cos α = c^2 := by sorry

end NUMINAMATH_CALUDE_island_distance_l326_32665


namespace NUMINAMATH_CALUDE_circle_triangle_area_ratio_l326_32613

/-- For a right triangle circumscribed about a circle -/
theorem circle_triangle_area_ratio
  (h a b R : ℝ)
  (h_positive : h > 0)
  (R_positive : R > 0)
  (right_triangle : a^2 + b^2 = h^2)
  (circumradius : R = h / 2) :
  π * R^2 / (a * b / 2) = π * h / (4 * R) :=
by sorry

end NUMINAMATH_CALUDE_circle_triangle_area_ratio_l326_32613


namespace NUMINAMATH_CALUDE_hidden_dots_count_l326_32628

/-- Represents a standard six-sided die -/
def StandardDie : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- The sum of all numbers on a standard die -/
def SumOfDie : ℕ := Finset.sum StandardDie id

/-- The number of dice in the stack -/
def NumberOfDice : ℕ := 4

/-- The visible numbers on the dice -/
def VisibleNumbers : Finset ℕ := {1, 2, 2, 3, 4, 4, 5, 6, 6}

/-- The theorem stating the number of hidden dots -/
theorem hidden_dots_count : 
  NumberOfDice * SumOfDie - Finset.sum VisibleNumbers id = 51 := by sorry

end NUMINAMATH_CALUDE_hidden_dots_count_l326_32628


namespace NUMINAMATH_CALUDE_negation_constant_geometric_sequence_l326_32677

theorem negation_constant_geometric_sequence :
  ¬(∀ (a : ℕ → ℝ), (∀ n : ℕ, a n = a 0) → (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n)) ↔
  (∃ (a : ℕ → ℝ), (∀ n : ℕ, a n = a 0) ∧ ¬(∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n)) :=
by sorry

end NUMINAMATH_CALUDE_negation_constant_geometric_sequence_l326_32677


namespace NUMINAMATH_CALUDE_timothy_cows_l326_32694

def total_cost : ℕ := 147700
def land_cost : ℕ := 30 * 20
def house_cost : ℕ := 120000
def cow_cost : ℕ := 1000
def chicken_cost : ℕ := 100 * 5
def solar_installation_cost : ℕ := 6 * 100
def solar_equipment_cost : ℕ := 6000

def other_costs : ℕ := land_cost + house_cost + chicken_cost + solar_installation_cost + solar_equipment_cost

theorem timothy_cows :
  (total_cost - other_costs) / cow_cost = 20 := by sorry

end NUMINAMATH_CALUDE_timothy_cows_l326_32694


namespace NUMINAMATH_CALUDE_gondor_thursday_laptops_l326_32605

/-- Represents the earnings and repair counts for Gondor --/
structure GondorEarnings where
  phone_repair_price : ℕ
  laptop_repair_price : ℕ
  monday_phones : ℕ
  tuesday_phones : ℕ
  wednesday_laptops : ℕ
  total_earnings : ℕ

/-- Calculates the number of laptops repaired on Thursday --/
def thursday_laptops (g : GondorEarnings) : ℕ :=
  let mon_tue_wed_earnings := g.phone_repair_price * (g.monday_phones + g.tuesday_phones) + 
                              g.laptop_repair_price * g.wednesday_laptops
  let thursday_earnings := g.total_earnings - mon_tue_wed_earnings
  thursday_earnings / g.laptop_repair_price

/-- Theorem stating that Gondor repaired 4 laptops on Thursday --/
theorem gondor_thursday_laptops :
  let g : GondorEarnings := {
    phone_repair_price := 10,
    laptop_repair_price := 20,
    monday_phones := 3,
    tuesday_phones := 5,
    wednesday_laptops := 2,
    total_earnings := 200
  }
  thursday_laptops g = 4 := by sorry

end NUMINAMATH_CALUDE_gondor_thursday_laptops_l326_32605


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l326_32603

theorem nested_fraction_evaluation :
  2 + (3 / (4 + (5 / (6 + (7/8))))) = 137/52 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l326_32603


namespace NUMINAMATH_CALUDE_length_of_AC_l326_32601

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D O : ℝ × ℝ)  -- Points in 2D plane

-- Define the conditions
def satisfies_conditions (q : Quadrilateral) : Prop :=
  let d := (λ p1 p2 : ℝ × ℝ => ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2).sqrt)
  d q.O q.A = 5 ∧
  d q.O q.C = 12 ∧
  d q.O q.B = 6 ∧
  d q.O q.D = 5 ∧
  d q.B q.D = 11

-- State the theorem
theorem length_of_AC (q : Quadrilateral) :
  satisfies_conditions q →
  let d := (λ p1 p2 : ℝ × ℝ => ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2).sqrt)
  d q.A q.C = 3 * (71 : ℝ).sqrt :=
by sorry

end NUMINAMATH_CALUDE_length_of_AC_l326_32601


namespace NUMINAMATH_CALUDE_handshake_count_l326_32636

theorem handshake_count (n : ℕ) (h : n = 8) : 
  (2 * n) * (2 * n - 2) / 2 = 112 := by sorry

end NUMINAMATH_CALUDE_handshake_count_l326_32636


namespace NUMINAMATH_CALUDE_log_equation_solution_l326_32614

theorem log_equation_solution (x : ℝ) (h : x > 0) (eq : Real.log (729 : ℝ) / Real.log (3 * x) = x) :
  x = 3 ∧ ¬ ∃ n : ℕ, x = n^2 ∧ ¬ ∃ m : ℕ, x = m^3 ∧ ∃ k : ℕ, x = k := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l326_32614


namespace NUMINAMATH_CALUDE_share_distribution_l326_32692

theorem share_distribution (total : ℝ) (share_a : ℝ) (share_b : ℝ) (share_c : ℝ) :
  total = 246 →
  share_b = 0.65 →
  share_c = 48 →
  share_a + share_b + share_c = 1 →
  share_c / total = 48 / 246 :=
by sorry

end NUMINAMATH_CALUDE_share_distribution_l326_32692


namespace NUMINAMATH_CALUDE_quadratic_downward_solution_nonempty_l326_32626

/-- A quadratic function f(x) = ax² + bx + c opens downwards if a < 0 -/
def opens_downwards (a b c : ℝ) : Prop := a < 0

/-- The solution set of ax² + bx + c < 0 is not empty -/
def solution_set_nonempty (a b c : ℝ) : Prop :=
  ∃ x : ℝ, a * x^2 + b * x + c < 0

/-- If a quadratic function opens downwards, its solution set for f(x) < 0 is not empty -/
theorem quadratic_downward_solution_nonempty (a b c : ℝ) :
  opens_downwards a b c → solution_set_nonempty a b c := by
  sorry

end NUMINAMATH_CALUDE_quadratic_downward_solution_nonempty_l326_32626


namespace NUMINAMATH_CALUDE_half_angle_quadrant_l326_32651

theorem half_angle_quadrant (α : Real) (h : ∃ k : ℤ, 2 * k * π + π / 2 < α ∧ α < 2 * k * π + π) :
  ∃ m : ℤ, (m * π < α / 2 ∧ α / 2 < m * π + π / 2) ∨ 
           (m * π + π < α / 2 ∧ α / 2 < m * π + 3 * π / 2) :=
by sorry

end NUMINAMATH_CALUDE_half_angle_quadrant_l326_32651


namespace NUMINAMATH_CALUDE_function_range_l326_32635

theorem function_range (f : ℝ → ℝ) (a : ℝ) :
  (∀ x : ℝ, f x = |x - 3| + |x - a|) →
  (∀ x : ℝ, f x ≥ 4) →
  (a ≤ -1 ∨ a ≥ 7) :=
by sorry

end NUMINAMATH_CALUDE_function_range_l326_32635


namespace NUMINAMATH_CALUDE_sqrt_four_squared_l326_32654

theorem sqrt_four_squared : (Real.sqrt 4)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_four_squared_l326_32654


namespace NUMINAMATH_CALUDE_gas_cost_calculation_l326_32687

theorem gas_cost_calculation (total_cost : ℚ) : 
  (total_cost / 5 - 15 = total_cost / 7) → 
  total_cost = 262.5 := by
sorry

end NUMINAMATH_CALUDE_gas_cost_calculation_l326_32687


namespace NUMINAMATH_CALUDE_orange_flower_count_l326_32661

/-- Represents the number of flowers of each color in a garden -/
structure FlowerGarden where
  orange : ℕ
  red : ℕ
  yellow : ℕ
  pink : ℕ
  purple : ℕ

/-- Theorem stating the conditions and the result to be proved -/
theorem orange_flower_count (g : FlowerGarden) : 
  g.orange + g.red + g.yellow + g.pink + g.purple = 105 →
  g.red = 2 * g.orange →
  g.yellow = g.red - 5 →
  g.pink = g.purple →
  g.pink + g.purple = 30 →
  g.orange = 16 := by
  sorry


end NUMINAMATH_CALUDE_orange_flower_count_l326_32661


namespace NUMINAMATH_CALUDE_collinear_vectors_x_equals_three_l326_32622

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

/-- Given vectors a and b, prove that if they are collinear, then x = 3 -/
theorem collinear_vectors_x_equals_three (x : ℝ) :
  let a : ℝ × ℝ := (2, 4)
  let b : ℝ × ℝ := (x, 6)
  collinear a b → x = 3 := by
  sorry


end NUMINAMATH_CALUDE_collinear_vectors_x_equals_three_l326_32622


namespace NUMINAMATH_CALUDE_shortest_altitude_of_triangle_l326_32631

/-- Given a triangle with sides 18, 24, and 30, its shortest altitude has length 18 -/
theorem shortest_altitude_of_triangle (a b c h1 h2 h3 : ℝ) : 
  a = 18 ∧ b = 24 ∧ c = 30 →
  a^2 + b^2 = c^2 →
  h1 = a ∧ h2 = b ∧ h3 = (2 * (1/2 * a * b)) / c →
  min h1 (min h2 h3) = 18 := by
sorry

end NUMINAMATH_CALUDE_shortest_altitude_of_triangle_l326_32631


namespace NUMINAMATH_CALUDE_car_distance_18_hours_l326_32696

/-- Calculates the total distance traveled by a car with increasing speed -/
def totalDistance (initialSpeed : ℕ) (speedIncrease : ℕ) (hours : ℕ) : ℕ :=
  let finalSpeed := initialSpeed + speedIncrease * (hours - 1)
  hours * (initialSpeed + finalSpeed) / 2

/-- Theorem stating the total distance traveled by the car in 18 hours -/
theorem car_distance_18_hours :
  totalDistance 30 5 18 = 1305 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_18_hours_l326_32696


namespace NUMINAMATH_CALUDE_range_equal_shifted_l326_32690

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Define the range of a function
def range (g : ℝ → ℝ) := {y : ℝ | ∃ x, g x = y}

-- Theorem statement
theorem range_equal_shifted : range f = range (fun x ↦ f (x + 1)) := by sorry

end NUMINAMATH_CALUDE_range_equal_shifted_l326_32690


namespace NUMINAMATH_CALUDE_prob_same_roll_6_7_l326_32680

/-- The probability of rolling a specific number on a fair die with n sides -/
def prob_roll (n : ℕ) : ℚ := 1 / n

/-- The probability of rolling the same number on two dice with sides n and m -/
def prob_same_roll (n m : ℕ) : ℚ := (prob_roll n) * (prob_roll m)

theorem prob_same_roll_6_7 :
  prob_same_roll 6 7 = 1 / 42 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_roll_6_7_l326_32680


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l326_32607

theorem quadratic_roots_condition (c : ℝ) : 
  (∀ x : ℝ, x^2 - 3*x + c = 0 ↔ x = (3 + Real.sqrt c) / 2 ∨ x = (3 - Real.sqrt c) / 2) → 
  c = 9/5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l326_32607


namespace NUMINAMATH_CALUDE_gcd_of_polynomial_and_linear_l326_32662

theorem gcd_of_polynomial_and_linear (a : ℤ) (h : ∃ k : ℤ, a = 360 * k) :
  Int.gcd (a^2 + 6*a + 8) (a + 4) = 4 := by sorry

end NUMINAMATH_CALUDE_gcd_of_polynomial_and_linear_l326_32662


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l326_32659

theorem sqrt_product_simplification (y : ℝ) (h : y ≥ 0) :
  Real.sqrt (48 * y) * Real.sqrt (18 * y) * Real.sqrt (50 * y) = 120 * y * Real.sqrt (3 * y) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l326_32659


namespace NUMINAMATH_CALUDE_circle_division_sum_integer_l326_32639

theorem circle_division_sum_integer :
  ∃ (a b c d e : ℕ),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
    c ≠ d ∧ c ≠ e ∧
    d ≠ e ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
    ∃ (n : ℤ), (a / b : ℚ) + (b / c : ℚ) + (c / d : ℚ) + (d / e : ℚ) + (e / a : ℚ) = n :=
sorry

end NUMINAMATH_CALUDE_circle_division_sum_integer_l326_32639


namespace NUMINAMATH_CALUDE_john_annual_oil_change_cost_l326_32658

/-- Calculates the annual cost of oil changes for a driver named John. -/
theorem john_annual_oil_change_cost :
  ∀ (miles_per_month : ℕ) 
    (miles_per_oil_change : ℕ) 
    (free_oil_changes_per_year : ℕ) 
    (cost_per_oil_change : ℕ),
  miles_per_month = 1000 →
  miles_per_oil_change = 3000 →
  free_oil_changes_per_year = 1 →
  cost_per_oil_change = 50 →
  (12 * miles_per_month / miles_per_oil_change - free_oil_changes_per_year) * cost_per_oil_change = 150 :=
by
  sorry

#check john_annual_oil_change_cost

end NUMINAMATH_CALUDE_john_annual_oil_change_cost_l326_32658


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a5_zero_l326_32647

/-- An arithmetic sequence with common difference d -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_a5_zero
  (a : ℕ → ℝ) (d : ℝ)
  (h_arith : arithmetic_sequence a d)
  (h_d_nonzero : d ≠ 0)
  (h_condition : a 3 + a 9 = a 10 - a 8) :
  a 5 = 0 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a5_zero_l326_32647


namespace NUMINAMATH_CALUDE_double_base_exponent_equality_l326_32698

theorem double_base_exponent_equality (a b x : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) :
  (2 * a) ^ (2 * b) = a ^ b * x ^ 3 → x = (4 ^ b * a ^ b) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_double_base_exponent_equality_l326_32698


namespace NUMINAMATH_CALUDE_unique_five_digit_number_l326_32695

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def remove_digit (n : ℕ) (pos : Fin 5) : ℕ :=
  let digits := [n / 10000, (n / 1000) % 10, (n / 100) % 10, (n / 10) % 10, n % 10]
  let removed := digits.removeNth pos
  removed.foldl (fun acc d => acc * 10 + d) 0

theorem unique_five_digit_number : ∃! n : ℕ, 
  is_five_digit n ∧ 
  (∃ pos : Fin 5, n + remove_digit n pos = 54321) :=
by
  use 49383
  sorry

end NUMINAMATH_CALUDE_unique_five_digit_number_l326_32695


namespace NUMINAMATH_CALUDE_arm_wrestling_streaks_l326_32697

/-- Represents the outcome of a single round of arm wrestling -/
inductive Winner : Type
| Richard : Winner
| Shreyas : Winner

/-- Counts the number of streaks in a list of outcomes -/
def count_streaks (outcomes : List Winner) : Nat :=
  sorry

/-- Generates all possible outcomes for n rounds of arm wrestling -/
def generate_outcomes (n : Nat) : List (List Winner) :=
  sorry

/-- Counts the number of outcomes with more than k streaks in n rounds -/
def count_outcomes_with_more_than_k_streaks (n k : Nat) : Nat :=
  sorry

theorem arm_wrestling_streaks :
  count_outcomes_with_more_than_k_streaks 10 3 = 932 :=
sorry

end NUMINAMATH_CALUDE_arm_wrestling_streaks_l326_32697


namespace NUMINAMATH_CALUDE_tan_value_problem_l326_32648

theorem tan_value_problem (θ : Real) 
  (h1 : 0 < θ) (h2 : θ < π) 
  (h3 : Real.sin θ + Real.cos θ = 1/5) : 
  Real.tan θ = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_problem_l326_32648


namespace NUMINAMATH_CALUDE_xy_greater_than_xz_l326_32619

theorem xy_greater_than_xz (x y z : ℝ) (h1 : x > y) (h2 : y > z) (h3 : x + y + z = 0) : x * y > x * z := by
  sorry

end NUMINAMATH_CALUDE_xy_greater_than_xz_l326_32619


namespace NUMINAMATH_CALUDE_circle_radius_from_chord_and_central_angle_l326_32684

theorem circle_radius_from_chord_and_central_angle (α : ℝ) (h : α > 0 ∧ α < 360) :
  let chord_length : ℝ := 10
  let radius : ℝ := 5 / Real.sin (α * π / 360)
  2 * radius * Real.sin (α * π / 360) = chord_length := by sorry

end NUMINAMATH_CALUDE_circle_radius_from_chord_and_central_angle_l326_32684


namespace NUMINAMATH_CALUDE_series_sum_equals_negative_four_l326_32645

/-- The sum of the infinite series $\sum_{n=1}^\infty \frac{2n^2 - 3n + 2}{n(n+1)(n+2)}$ equals -4. -/
theorem series_sum_equals_negative_four :
  ∑' n : ℕ+, (2 * n^2 - 3 * n + 2 : ℝ) / (n * (n + 1) * (n + 2)) = -4 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_negative_four_l326_32645
