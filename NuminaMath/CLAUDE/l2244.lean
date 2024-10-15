import Mathlib

namespace NUMINAMATH_CALUDE_x_range_l2244_224415

theorem x_range (x : ℝ) : 
  (|x - 1| + |x - 5| = 4) ↔ (1 ≤ x ∧ x ≤ 5) := by
sorry

end NUMINAMATH_CALUDE_x_range_l2244_224415


namespace NUMINAMATH_CALUDE_symmetry_of_lines_l2244_224431

/-- The line of symmetry -/
def line_of_symmetry (x y : ℝ) : Prop := y - x = 0

/-- The original line -/
def original_line (x y : ℝ) : Prop := x - 2*y - 1 = 0

/-- The symmetric line -/
def symmetric_line (x y : ℝ) : Prop := 2*x - y + 1 = 0

/-- Theorem stating that the symmetric_line is indeed symmetric to the original_line
    with respect to the line_of_symmetry -/
theorem symmetry_of_lines :
  ∀ (x y x' y' : ℝ),
    original_line x y →
    line_of_symmetry ((x + x') / 2) ((y + y') / 2) →
    symmetric_line x' y' :=
by sorry

end NUMINAMATH_CALUDE_symmetry_of_lines_l2244_224431


namespace NUMINAMATH_CALUDE_jane_lost_twenty_points_l2244_224497

/-- Represents the card game scenario --/
structure CardGame where
  pointsPerWin : ℕ
  totalRounds : ℕ
  finalPoints : ℕ

/-- Calculates the points lost in the card game --/
def pointsLost (game : CardGame) : ℕ :=
  game.pointsPerWin * game.totalRounds - game.finalPoints

/-- Theorem stating that Jane lost 20 points --/
theorem jane_lost_twenty_points :
  let game : CardGame := {
    pointsPerWin := 10,
    totalRounds := 8,
    finalPoints := 60
  }
  pointsLost game = 20 := by
  sorry


end NUMINAMATH_CALUDE_jane_lost_twenty_points_l2244_224497


namespace NUMINAMATH_CALUDE_cash_preference_factors_l2244_224465

/-- Represents an economic factor influencing payment preference --/
structure EconomicFactor where
  description : String
  favors_cash : Bool

/-- Represents a large retail chain --/
structure RetailChain where
  name : String
  payment_preference : String

/-- Theorem: There exist at least three distinct economic factors that could lead large retail chains to prefer cash payments --/
theorem cash_preference_factors :
  ∃ (f1 f2 f3 : EconomicFactor),
    f1.favors_cash ∧ f2.favors_cash ∧ f3.favors_cash ∧
    f1 ≠ f2 ∧ f1 ≠ f3 ∧ f2 ≠ f3 ∧
    (∃ (rc : RetailChain), rc.payment_preference = "cash") :=
by sorry

/-- Definition: Efficiency of operations as an economic factor --/
def efficiency_factor : EconomicFactor :=
  { description := "Efficiency of Operations", favors_cash := true }

/-- Definition: Cost of handling transactions as an economic factor --/
def cost_factor : EconomicFactor :=
  { description := "Cost of Handling Transactions", favors_cash := true }

/-- Definition: Risk of fraud as an economic factor --/
def risk_factor : EconomicFactor :=
  { description := "Risk of Fraud", favors_cash := true }

end NUMINAMATH_CALUDE_cash_preference_factors_l2244_224465


namespace NUMINAMATH_CALUDE_complex_number_simplification_l2244_224447

theorem complex_number_simplification :
  let i : ℂ := Complex.I
  (i^3) / (1 - i) = (1 / 2 : ℂ) - (1 / 2 : ℂ) * i := by sorry

end NUMINAMATH_CALUDE_complex_number_simplification_l2244_224447


namespace NUMINAMATH_CALUDE_youngest_child_age_l2244_224417

def total_bill : ℚ := 12.25
def mother_meal : ℚ := 3.75
def cost_per_year : ℚ := 0.5

structure Family :=
  (triplet_age : ℕ)
  (youngest_age : ℕ)

def valid_family (f : Family) : Prop :=
  f.youngest_age < f.triplet_age ∧
  mother_meal + cost_per_year * (3 * f.triplet_age + f.youngest_age) = total_bill

theorem youngest_child_age :
  ∃ (f₁ f₂ : Family), valid_family f₁ ∧ valid_family f₂ ∧
    f₁.youngest_age = 2 ∧ f₂.youngest_age = 5 ∧
    ∀ (f : Family), valid_family f → f.youngest_age = 2 ∨ f.youngest_age = 5 :=
sorry

end NUMINAMATH_CALUDE_youngest_child_age_l2244_224417


namespace NUMINAMATH_CALUDE_min_ab_value_l2244_224408

theorem min_ab_value (a b : ℕ+) 
  (h1 : ¬ (7 ∣ (a * b * (a + b))))
  (h2 : (7 ∣ ((a + b)^7 - a^7 - b^7))) :
  ∀ x y : ℕ+, 
    (¬ (7 ∣ (x * y * (x + y)))) → 
    ((7 ∣ ((x + y)^7 - x^7 - y^7))) → 
    a * b ≤ x * y :=
by sorry

end NUMINAMATH_CALUDE_min_ab_value_l2244_224408


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2244_224427

/-- Given a geometric sequence {aₙ}, if a₁a₂a₃ = -8, then a₂ = -2 -/
theorem geometric_sequence_property (a : ℕ → ℝ) :
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence property
  a 1 * a 2 * a 3 = -8 →                -- given condition
  a 2 = -2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2244_224427


namespace NUMINAMATH_CALUDE_perpendicular_lines_coefficient_l2244_224424

/-- Two lines in the form y = mx + b are perpendicular if and only if the product of their slopes is -1 -/
axiom perpendicular_lines_slope_product (m₁ m₂ : ℝ) : 
  m₁ * m₂ = -1 ↔ (∃ (b₁ b₂ : ℝ), ∀ (x y : ℝ), y = m₁ * x + b₁ ↔ y = -1/m₂ * x + b₂)

/-- Given two lines ax + y + 2 = 0 and 3x - y - 2 = 0 that are perpendicular, prove that a = 2/3 -/
theorem perpendicular_lines_coefficient (a : ℝ) :
  (∀ (x y : ℝ), y = -a * x - 2 ↔ y = 3 * x - 2) →
  a = 2/3 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_coefficient_l2244_224424


namespace NUMINAMATH_CALUDE_undetermined_zeros_l2244_224460

theorem undetermined_zeros (f : ℝ → ℝ) (a b : ℝ) (h1 : a < b) (h2 : f a * f b < 0) :
  ∃ (n : ℕ), n ≥ 0 ∧ (∃ (x : ℝ), x ∈ Set.Ioo a b ∧ f x = 0) ∧
  ¬ (∀ (m : ℕ), m ≠ n → ¬ (∃ (x : ℝ), x ∈ Set.Ioo a b ∧ f x = 0 ∧
    (∃ (y : ℝ), y ≠ x ∧ y ∈ Set.Ioo a b ∧ f y = 0))) :=
sorry

end NUMINAMATH_CALUDE_undetermined_zeros_l2244_224460


namespace NUMINAMATH_CALUDE_cube_inequality_negation_l2244_224487

theorem cube_inequality_negation (x y : ℝ) (h : x > y) : 
  ¬(x^3 > y^3) ↔ x^3 ≤ y^3 := by
sorry

end NUMINAMATH_CALUDE_cube_inequality_negation_l2244_224487


namespace NUMINAMATH_CALUDE_lines_coincide_l2244_224423

/-- If three lines y = kx + m, y = mx + n, and y = nx + k have a common point, then k = m = n -/
theorem lines_coincide (k m n : ℝ) (x y : ℝ) 
  (h1 : y = k * x + m)
  (h2 : y = m * x + n)
  (h3 : y = n * x + k) :
  k = m ∧ m = n := by
  sorry

end NUMINAMATH_CALUDE_lines_coincide_l2244_224423


namespace NUMINAMATH_CALUDE_integer_quotient_characterization_l2244_224461

def solution_set : Set (ℤ × ℤ) :=
  {(1, 2), (1, 3), (2, 1), (3, 1), (2, 5), (3, 5), (5, 2), (5, 3), (2, 2)}

theorem integer_quotient_characterization (m n : ℤ) :
  (∃ k : ℤ, (n^3 + 1) = k * (m * n - 1)) ↔ (m, n) ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_integer_quotient_characterization_l2244_224461


namespace NUMINAMATH_CALUDE_problem_statement_l2244_224459

theorem problem_statement (x y : ℝ) (hx : x = 7) (hy : y = -2) : 
  (x - 2*y)^y = 1/121 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l2244_224459


namespace NUMINAMATH_CALUDE_triple_equation_solution_l2244_224498

theorem triple_equation_solution (a b c : ℝ) :
  (a * (b^2 + c) = c * (c + a * b)) ∧
  (b * (c^2 + a) = a * (a + b * c)) ∧
  (c * (a^2 + b) = b * (b + c * a)) →
  (∃ x : ℝ, a = x ∧ b = x ∧ c = x) ∨
  (b = 0 ∧ c = 0) :=
sorry

end NUMINAMATH_CALUDE_triple_equation_solution_l2244_224498


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l2244_224478

open Real

theorem trigonometric_equation_solution (x : ℝ) :
  (∀ K : ℤ, x ≠ π * K / 3) →
  (cos x)^2 = (sin (2 * x))^2 + cos (3 * x) / sin (3 * x) →
  (∃ n : ℤ, x = π / 2 + π * n) ∨ (∃ k : ℤ, x = π / 6 + π * k) ∨ (∃ k : ℤ, x = -π / 6 + π * k) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l2244_224478


namespace NUMINAMATH_CALUDE_negation_of_exists_square_nonpositive_l2244_224441

theorem negation_of_exists_square_nonpositive :
  (¬ ∃ a : ℝ, a^2 ≤ 0) ↔ (∀ a : ℝ, a^2 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_exists_square_nonpositive_l2244_224441


namespace NUMINAMATH_CALUDE_dany_sheep_count_l2244_224434

/-- Represents the number of bushels eaten by sheep and chickens on Dany's farm -/
def farm_bushels (num_sheep : ℕ) : ℕ :=
  2 * num_sheep + 3

/-- Theorem stating that Dany has 16 sheep on his farm -/
theorem dany_sheep_count : ∃ (num_sheep : ℕ), farm_bushels num_sheep = 35 ∧ num_sheep = 16 := by
  sorry

end NUMINAMATH_CALUDE_dany_sheep_count_l2244_224434


namespace NUMINAMATH_CALUDE_successive_discounts_equivalence_l2244_224476

theorem successive_discounts_equivalence :
  let discount1 : ℝ := 0.15
  let discount2 : ℝ := 0.10
  let discount3 : ℝ := 0.05
  let equivalent_single_discount : ℝ := 1 - (1 - discount1) * (1 - discount2) * (1 - discount3)
  equivalent_single_discount = 0.27325 :=
by sorry

end NUMINAMATH_CALUDE_successive_discounts_equivalence_l2244_224476


namespace NUMINAMATH_CALUDE_smallest_with_9_odd_18_even_factors_l2244_224401

/-- The number of odd factors of an integer -/
def num_odd_factors (n : ℕ) : ℕ := sorry

/-- The number of even factors of an integer -/
def num_even_factors (n : ℕ) : ℕ := sorry

/-- Theorem: The smallest integer with exactly 9 odd factors and 18 even factors is 900 -/
theorem smallest_with_9_odd_18_even_factors :
  ∀ n : ℕ, num_odd_factors n = 9 ∧ num_even_factors n = 18 → n ≥ 900 ∧
  ∃ m : ℕ, m = 900 ∧ num_odd_factors m = 9 ∧ num_even_factors m = 18 :=
sorry

end NUMINAMATH_CALUDE_smallest_with_9_odd_18_even_factors_l2244_224401


namespace NUMINAMATH_CALUDE_min_value_theorem_l2244_224471

theorem min_value_theorem (a b c d : ℝ) 
  (hb : b ≠ 0) 
  (hd : d ≠ -1) 
  (h1 : (a^2 - Real.log a) / b = (c - 1) / (d + 1))
  (h2 : (a^2 - Real.log a) / b = 1) :
  ∃ (min : ℝ), min = 2 ∧ ∀ (x y : ℝ), (x^2 - Real.log x) / y = (c - 1) / (d + 1) → 
    (x - c)^2 + (y - d)^2 ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2244_224471


namespace NUMINAMATH_CALUDE_cousins_distribution_l2244_224407

/-- The number of ways to distribute n indistinguishable objects into k distinct containers -/
def distribute (n k : ℕ) : ℕ := sorry

/-- There are 4 rooms -/
def num_rooms : ℕ := 4

/-- There are 5 cousins -/
def num_cousins : ℕ := 5

/-- The number of ways to distribute the cousins into the rooms -/
def num_distributions : ℕ := distribute num_cousins num_rooms

theorem cousins_distribution : num_distributions = 51 := by sorry

end NUMINAMATH_CALUDE_cousins_distribution_l2244_224407


namespace NUMINAMATH_CALUDE_radical_equation_solution_l2244_224449

theorem radical_equation_solution (a b c : ℕ) (h1 : a > 1) (h2 : b > 1) (h3 : c > 1) :
  (∀ N : ℝ, N ≠ 1 → N^((1 : ℝ)/a + 1/(a*b) + 2/(a*b*c)) = N^(17/24)) →
  b = 4 := by
sorry

end NUMINAMATH_CALUDE_radical_equation_solution_l2244_224449


namespace NUMINAMATH_CALUDE_unique_solution_l2244_224477

def is_valid_number (a b : ℕ) : Prop :=
  a < 10 ∧ b < 10 ∧
  (18600 + 10 * a + b) % 3 = 2 ∧
  (18600 + 10 * a + b) % 5 = 3 ∧
  (18600 + 10 * a + b) % 11 = 0

theorem unique_solution :
  ∃! (a b : ℕ), is_valid_number a b ∧ a = 2 ∧ b = 3 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l2244_224477


namespace NUMINAMATH_CALUDE_game_points_difference_l2244_224480

theorem game_points_difference (layla_points nahima_points total_points : ℕ) : 
  layla_points = 70 → total_points = 112 → layla_points + nahima_points = total_points →
  layla_points - nahima_points = 28 := by
sorry

end NUMINAMATH_CALUDE_game_points_difference_l2244_224480


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2244_224406

theorem quadratic_inequality_solution (x : ℝ) :
  (x^2 - 50*x + 575 ≤ 25) ↔ (25 - 5*Real.sqrt 3 ≤ x ∧ x ≤ 25 + 5*Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2244_224406


namespace NUMINAMATH_CALUDE_hamburger_count_l2244_224412

theorem hamburger_count (served left_over : ℕ) (h1 : served = 3) (h2 : left_over = 6) :
  served + left_over = 9 := by
  sorry

end NUMINAMATH_CALUDE_hamburger_count_l2244_224412


namespace NUMINAMATH_CALUDE_initial_price_equation_l2244_224410

/-- The initial price of speakers before discount -/
def initial_price : ℝ := 475

/-- The final price paid after discount -/
def final_price : ℝ := 199

/-- The discount amount saved -/
def discount : ℝ := 276

/-- Theorem stating that the initial price is equal to the sum of the final price and the discount -/
theorem initial_price_equation : initial_price = final_price + discount := by
  sorry

end NUMINAMATH_CALUDE_initial_price_equation_l2244_224410


namespace NUMINAMATH_CALUDE_set_operations_l2244_224414

-- Define the sets A and B
def A : Set ℝ := {x | -2 < x ∧ x < 5}
def B : Set ℝ := {x | -1 ≤ x - 1 ∧ x - 1 ≤ 2}

-- Define the universal set U
def U : Set ℝ := Set.univ

theorem set_operations :
  (A ∪ B = {x | -2 < x ∧ x < 5}) ∧
  (A ∩ B = {x | 0 ≤ x ∧ x ≤ 3}) ∧
  (A ∪ Bᶜ = U) ∧
  (A ∩ Bᶜ = {x | -2 < x ∧ x < 0} ∪ {x | 3 < x ∧ x < 5}) :=
by sorry

end NUMINAMATH_CALUDE_set_operations_l2244_224414


namespace NUMINAMATH_CALUDE_intersection_and_solution_set_l2244_224457

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | x^2 - 5*x + 6 < 0}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := {x | -1 < x ∧ x < 2}

-- Define the solution set of x^2 + ax - b < 0
def solution_set (a b : ℝ) : Set ℝ := {x | x < -1 ∨ x > 2}

theorem intersection_and_solution_set :
  (A ∩ B = A_intersect_B) ∧
  (∀ a b : ℝ, ({x : ℝ | x^2 + a*x + b < 0} = A_intersect_B) →
              ({x : ℝ | x^2 + a*x - b < 0} = solution_set a b)) :=
by sorry

end NUMINAMATH_CALUDE_intersection_and_solution_set_l2244_224457


namespace NUMINAMATH_CALUDE_max_value_abcd_l2244_224413

theorem max_value_abcd (a b c d : ℝ) 
  (nonneg_a : 0 ≤ a) (nonneg_b : 0 ≤ b) (nonneg_c : 0 ≤ c) (nonneg_d : 0 ≤ d)
  (sum_squares : a^2 + b^2 + c^2 + d^2 = 1) :
  2 * a * b * Real.sqrt 2 + 2 * b * c + 2 * c * d ≤ 1 ∧ 
  ∃ a' b' c' d', 0 ≤ a' ∧ 0 ≤ b' ∧ 0 ≤ c' ∧ 0 ≤ d' ∧ 
    a'^2 + b'^2 + c'^2 + d'^2 = 1 ∧
    2 * a' * b' * Real.sqrt 2 + 2 * b' * c' + 2 * c' * d' = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_abcd_l2244_224413


namespace NUMINAMATH_CALUDE_min_value_x_plus_four_over_x_l2244_224486

theorem min_value_x_plus_four_over_x (x : ℝ) (h : x > 0) :
  x + 4 / x ≥ 4 ∧ (x + 4 / x = 4 ↔ x = 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_four_over_x_l2244_224486


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2244_224445

/-- A hyperbola with given asymptotes -/
structure Hyperbola where
  /-- The asymptotes of the hyperbola are x ± 2y = 0 -/
  asymptotes : ∀ (x y : ℝ), x = 2*y ∨ x = -2*y

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := 
  sorry

/-- Theorem stating that the eccentricity of the hyperbola is either √5 or √5/2 -/
theorem hyperbola_eccentricity (h : Hyperbola) : 
  eccentricity h = Real.sqrt 5 ∨ eccentricity h = (Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2244_224445


namespace NUMINAMATH_CALUDE_max_intersections_theorem_l2244_224418

/-- A convex polygon -/
structure ConvexPolygon where
  sides : ℕ

/-- The configuration of two convex polygons where one is contained within the other -/
structure PolygonConfiguration where
  outer : ConvexPolygon
  inner : ConvexPolygon
  inner_contained : inner.sides ≤ outer.sides
  no_coincident_sides : Bool

/-- The maximum number of intersection points between the sides of two polygons in the given configuration -/
def max_intersections (config : PolygonConfiguration) : ℕ :=
  config.inner.sides * config.outer.sides

/-- Theorem stating that the maximum number of intersections is the product of the number of sides -/
theorem max_intersections_theorem (config : PolygonConfiguration) :
  max_intersections config = config.inner.sides * config.outer.sides :=
sorry

end NUMINAMATH_CALUDE_max_intersections_theorem_l2244_224418


namespace NUMINAMATH_CALUDE_correct_bottles_per_pack_l2244_224402

/-- The number of bottles in each pack of soda -/
def bottles_per_pack : ℕ := 6

/-- The number of packs Rebecca bought -/
def packs_bought : ℕ := 3

/-- The number of bottles Rebecca drinks per day -/
def bottles_per_day : ℚ := 1/2

/-- The number of days in the given period -/
def days : ℕ := 28

/-- The number of bottles remaining after the given period -/
def bottles_remaining : ℕ := 4

/-- Theorem stating that the number of bottles in each pack is correct -/
theorem correct_bottles_per_pack :
  bottles_per_pack * packs_bought - (bottles_per_day * days).floor = bottles_remaining :=
sorry

end NUMINAMATH_CALUDE_correct_bottles_per_pack_l2244_224402


namespace NUMINAMATH_CALUDE_quadratic_solution_l2244_224438

theorem quadratic_solution (b : ℝ) : (5^2 + b*5 - 35 = 0) → b = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l2244_224438


namespace NUMINAMATH_CALUDE_flower_pots_distance_l2244_224429

/-- Given 8 equally spaced points on a line, if the distance between
    the first and fifth points is 100, then the distance between
    the first and eighth points is 175. -/
theorem flower_pots_distance (points : Fin 8 → ℝ) 
    (equally_spaced : ∀ i j k : Fin 8, i.val < j.val → j.val < k.val → 
      points k - points j = points j - points i)
    (dist_1_5 : points 4 - points 0 = 100) :
    points 7 - points 0 = 175 := by
  sorry


end NUMINAMATH_CALUDE_flower_pots_distance_l2244_224429


namespace NUMINAMATH_CALUDE_at_op_difference_l2244_224416

-- Define the @ operation
def at_op (x y : ℤ) : ℤ := x * y - 3 * x + y

-- State the theorem
theorem at_op_difference : at_op 5 9 - at_op 9 5 = 16 := by sorry

end NUMINAMATH_CALUDE_at_op_difference_l2244_224416


namespace NUMINAMATH_CALUDE_cat_food_insufficient_l2244_224479

theorem cat_food_insufficient (B S : ℝ) 
  (h1 : B > S) 
  (h2 : B < 2 * S) : 
  4 * B + 4 * S < 3 * (B + 2 * S) := by
sorry

end NUMINAMATH_CALUDE_cat_food_insufficient_l2244_224479


namespace NUMINAMATH_CALUDE_intersection_equivalence_l2244_224409

-- Define the types for our objects
variable (Point Line Plane : Type)

-- Define the relations we need
variable (intersect : Line → Line → Prop)
variable (in_plane : Line → Plane → Prop)
variable (line_intersects_plane : Line → Plane → Prop)
variable (plane_intersects_plane : Plane → Plane → Prop)

-- Define our specific objects
variable (l m : Line) (α β : Plane)

-- State the theorem
theorem intersection_equivalence 
  (h1 : intersect l m)
  (h2 : in_plane l α)
  (h3 : in_plane m α)
  (h4 : ¬ in_plane l β)
  (h5 : ¬ in_plane m β)
  : (line_intersects_plane l β ∨ line_intersects_plane m β) ↔ plane_intersects_plane α β :=
sorry

end NUMINAMATH_CALUDE_intersection_equivalence_l2244_224409


namespace NUMINAMATH_CALUDE_smallest_tangent_circle_l2244_224468

-- Define the given line
def given_line (x y : ℝ) : Prop := x - y - 4 = 0

-- Define the given circle
def given_circle (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 2

-- Define the solution circle
def solution_circle (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 2

-- Define a general circle with center (a, b) and radius r
def general_circle (x y a b r : ℝ) : Prop := (x - a)^2 + (y - b)^2 = r^2

-- Theorem statement
theorem smallest_tangent_circle :
  ∃! (a b r : ℝ),
    (∀ x y, general_circle x y a b r → 
      (∃ x₀ y₀, given_line x₀ y₀ ∧ general_circle x₀ y₀ a b r) ∧
      (∃ x₁ y₁, given_circle x₁ y₁ ∧ general_circle x₁ y₁ a b r)) ∧
    (∀ a' b' r', 
      (∀ x y, general_circle x y a' b' r' → 
        (∃ x₀ y₀, given_line x₀ y₀ ∧ general_circle x₀ y₀ a' b' r') ∧
        (∃ x₁ y₁, given_circle x₁ y₁ ∧ general_circle x₁ y₁ a' b' r')) →
      r ≤ r') ∧
    (∀ x y, general_circle x y a b r ↔ solution_circle x y) :=
sorry

end NUMINAMATH_CALUDE_smallest_tangent_circle_l2244_224468


namespace NUMINAMATH_CALUDE_sphere_in_cylinder_ratio_l2244_224493

theorem sphere_in_cylinder_ratio (r h : ℝ) (hr : r > 0) (hh : h > 0) : 
  (π * r^2 * h = 3 * (4/3 * π * r^3)) → (h / (2 * r) = 2) := by
  sorry

end NUMINAMATH_CALUDE_sphere_in_cylinder_ratio_l2244_224493


namespace NUMINAMATH_CALUDE_club_advantage_l2244_224404

/-- Represents a fitness club with a monthly subscription cost -/
structure FitnessClub where
  name : String
  monthlyCost : ℕ

/-- Represents an attendance pattern -/
inductive AttendancePattern
  | Regular
  | MoodBased

/-- Calculates the yearly cost for a given club and attendance pattern -/
def yearlyCost (club : FitnessClub) (pattern : AttendancePattern) : ℕ :=
  match pattern with
  | AttendancePattern.Regular => club.monthlyCost * 12
  | AttendancePattern.MoodBased => 
      if club.name = "Beta" then club.monthlyCost * 8 else club.monthlyCost * 12

/-- Calculates the number of visits per year for a given attendance pattern -/
def yearlyVisits (pattern : AttendancePattern) : ℕ :=
  match pattern with
  | AttendancePattern.Regular => 96
  | AttendancePattern.MoodBased => 56

/-- Calculates the cost per visit for a given club and attendance pattern -/
def costPerVisit (club : FitnessClub) (pattern : AttendancePattern) : ℚ :=
  (yearlyCost club pattern : ℚ) / (yearlyVisits pattern : ℚ)

theorem club_advantage :
  let alpha : FitnessClub := { name := "Alpha", monthlyCost := 999 }
  let beta : FitnessClub := { name := "Beta", monthlyCost := 1299 }
  (costPerVisit alpha AttendancePattern.Regular < costPerVisit beta AttendancePattern.Regular) ∧
  (costPerVisit beta AttendancePattern.MoodBased < costPerVisit alpha AttendancePattern.MoodBased) := by
  sorry

end NUMINAMATH_CALUDE_club_advantage_l2244_224404


namespace NUMINAMATH_CALUDE_distinct_arrangements_l2244_224499

def word_length : ℕ := 6
def freq_letter1 : ℕ := 1
def freq_letter2 : ℕ := 2
def freq_letter3 : ℕ := 3

theorem distinct_arrangements :
  (word_length.factorial) / (freq_letter1.factorial * freq_letter2.factorial * freq_letter3.factorial) = 60 := by
  sorry

end NUMINAMATH_CALUDE_distinct_arrangements_l2244_224499


namespace NUMINAMATH_CALUDE_f_satisfies_points_l2244_224439

/-- The relation between x and y --/
def f (x : ℝ) : ℝ := 200 - 15 * x - 15 * x^2

/-- The set of points that the function should satisfy --/
def points : List (ℝ × ℝ) := [(0, 200), (1, 170), (2, 120), (3, 50), (4, 0)]

/-- Theorem stating that the function satisfies all given points --/
theorem f_satisfies_points : ∀ (p : ℝ × ℝ), p ∈ points → f p.1 = p.2 := by
  sorry

#check f_satisfies_points

end NUMINAMATH_CALUDE_f_satisfies_points_l2244_224439


namespace NUMINAMATH_CALUDE_y_value_at_x_4_l2244_224463

/-- Given a function y = k * x^(1/4) where y = 3√2 when x = 81, 
    prove that y = 2 when x = 4 -/
theorem y_value_at_x_4 (k : ℝ) :
  (∀ x : ℝ, x > 0 → k * x^(1/4) = 3 * Real.sqrt 2 ↔ x = 81) →
  k * 4^(1/4) = 2 :=
by sorry

end NUMINAMATH_CALUDE_y_value_at_x_4_l2244_224463


namespace NUMINAMATH_CALUDE_initial_truck_distance_l2244_224473

/-- 
Given two trucks on opposite sides of a highway, where:
- Driver A starts driving at 90 km/h
- Driver B starts 1 hour later at 80 km/h
- When they meet, Driver A has driven 140 km farther than Driver B

This theorem proves that the initial distance between the trucks is 940 km.
-/
theorem initial_truck_distance :
  ∀ (t : ℝ) (d_a d_b : ℝ),
  d_a = 90 * (t + 1) →
  d_b = 80 * t →
  d_a = d_b + 140 →
  d_a + d_b = 940 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_truck_distance_l2244_224473


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a6_l2244_224464

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_a6 (a : ℕ → ℝ) (h : is_arithmetic_sequence a) 
  (h2 : a 2 = 4) (h4 : a 4 = 2) : a 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a6_l2244_224464


namespace NUMINAMATH_CALUDE_min_area_rectangle_l2244_224451

theorem min_area_rectangle (l w : ℕ) : 
  l > 0 ∧ w > 0 ∧ 2 * (l + w) = 120 → l * w ≥ 59 := by
  sorry

end NUMINAMATH_CALUDE_min_area_rectangle_l2244_224451


namespace NUMINAMATH_CALUDE_lily_family_vacation_suitcases_l2244_224472

/-- The number of suitcases Lily's family brings on vacation -/
def family_suitcases (num_siblings : ℕ) (suitcases_per_sibling : ℕ) (parent_suitcases : ℕ) : ℕ :=
  num_siblings * suitcases_per_sibling + parent_suitcases

/-- Theorem stating the total number of suitcases Lily's family brings on vacation -/
theorem lily_family_vacation_suitcases :
  family_suitcases 4 2 6 = 14 := by
  sorry

#eval family_suitcases 4 2 6

end NUMINAMATH_CALUDE_lily_family_vacation_suitcases_l2244_224472


namespace NUMINAMATH_CALUDE_sum_real_imag_parts_of_z_l2244_224450

theorem sum_real_imag_parts_of_z : ∃ z : ℂ, z * (2 + Complex.I) = 2 * Complex.I - 1 ∧ 
  z.re + z.im = 1 := by sorry

end NUMINAMATH_CALUDE_sum_real_imag_parts_of_z_l2244_224450


namespace NUMINAMATH_CALUDE_random_subset_is_sample_l2244_224403

/-- Represents a population of elements -/
structure Population (α : Type) where
  elements : Finset α
  size : ℕ
  size_eq : elements.card = size

/-- Represents a sample taken from a population -/
structure Sample (α : Type) where
  elements : Finset α
  size : ℕ
  size_eq : elements.card = size

/-- Defines what it means for a sample to be from a population -/
def is_sample_of {α : Type} (s : Sample α) (p : Population α) : Prop :=
  s.elements ⊆ p.elements ∧ s.size < p.size

/-- The theorem statement -/
theorem random_subset_is_sample 
  {α : Type} (p : Population α) (s : Sample α) 
  (h_p_size : p.size = 50000) 
  (h_s_size : s.size = 2000) 
  (h_subset : s.elements ⊆ p.elements) : 
  is_sample_of s p := by
  sorry


end NUMINAMATH_CALUDE_random_subset_is_sample_l2244_224403


namespace NUMINAMATH_CALUDE_constant_term_expansion_l2244_224454

theorem constant_term_expansion (x : ℝ) : 
  (fun r : ℕ => (-1)^r * (Nat.choose 6 r) * x^(6 - 2*r)) 3 = -20 := by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l2244_224454


namespace NUMINAMATH_CALUDE_honey_savings_l2244_224484

/-- Calculates the savings given daily earnings, number of days worked, and total spent -/
def calculate_savings (daily_earnings : ℕ) (days_worked : ℕ) (total_spent : ℕ) : ℕ :=
  daily_earnings * days_worked - total_spent

/-- Proves that given the problem conditions, Honey's savings are $240 -/
theorem honey_savings :
  let daily_earnings : ℕ := 80
  let days_worked : ℕ := 20
  let total_spent : ℕ := 1360
  calculate_savings daily_earnings days_worked total_spent = 240 := by
sorry

#eval calculate_savings 80 20 1360

end NUMINAMATH_CALUDE_honey_savings_l2244_224484


namespace NUMINAMATH_CALUDE_unripe_oranges_calculation_l2244_224458

/-- The number of sacks of ripe oranges harvested per day -/
def ripe_oranges_per_day : ℕ := 28

/-- The number of days of harvest -/
def harvest_days : ℕ := 26

/-- The total number of sacks of oranges after the harvest period -/
def total_oranges : ℕ := 2080

/-- The number of sacks of unripe oranges harvested per day -/
def unripe_oranges_per_day : ℕ := (total_oranges - ripe_oranges_per_day * harvest_days) / harvest_days

theorem unripe_oranges_calculation :
  unripe_oranges_per_day = 52 :=
by sorry

end NUMINAMATH_CALUDE_unripe_oranges_calculation_l2244_224458


namespace NUMINAMATH_CALUDE_science_club_enrollment_l2244_224483

theorem science_club_enrollment (total : ℕ) (math : ℕ) (physics : ℕ) (both : ℕ)
  (h1 : total = 120)
  (h2 : math = 80)
  (h3 : physics = 50)
  (h4 : both = 15) :
  total - (math + physics - both) = 5 := by
  sorry

end NUMINAMATH_CALUDE_science_club_enrollment_l2244_224483


namespace NUMINAMATH_CALUDE_function_range_lower_bound_l2244_224492

theorem function_range_lower_bound (n : ℕ) (f : ℤ → Fin n) 
  (h : ∀ (x y : ℤ), |x - y| ∈ ({2, 3, 5} : Set ℤ) → f x ≠ f y) : 
  n ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_function_range_lower_bound_l2244_224492


namespace NUMINAMATH_CALUDE_cake_area_increase_percentage_cake_area_increase_percentage_approx_l2244_224485

/-- The percent increase in area of a circular cake when its diameter increases from 8 inches to 10 inches -/
theorem cake_area_increase_percentage : ℝ := by
  -- Define the initial and final diameters
  let initial_diameter : ℝ := 8
  let final_diameter : ℝ := 10
  
  -- Define the function to calculate the area of a circular cake given its diameter
  let cake_area (d : ℝ) : ℝ := Real.pi * (d / 2) ^ 2
  
  -- Calculate the initial and final areas
  let initial_area := cake_area initial_diameter
  let final_area := cake_area final_diameter
  
  -- Calculate the percent increase
  let percent_increase := (final_area - initial_area) / initial_area * 100
  
  -- Prove that the percent increase is 56.25%
  sorry

/-- The result of cake_area_increase_percentage is approximately 56.25 -/
theorem cake_area_increase_percentage_approx :
  |cake_area_increase_percentage - 56.25| < 0.01 := by sorry

end NUMINAMATH_CALUDE_cake_area_increase_percentage_cake_area_increase_percentage_approx_l2244_224485


namespace NUMINAMATH_CALUDE_zain_coin_count_l2244_224433

/-- Represents the number of coins Emerie has -/
structure EmerieCoinCount where
  quarters : Nat
  dimes : Nat
  nickels : Nat

/-- Calculates the total number of coins Zain has -/
def zainTotalCoins (emerie : EmerieCoinCount) : Nat :=
  (emerie.quarters + 10) + (emerie.dimes + 10) + (emerie.nickels + 10)

/-- Theorem: Given Emerie's coin counts, prove that Zain has 48 coins -/
theorem zain_coin_count (emerie : EmerieCoinCount)
  (hq : emerie.quarters = 6)
  (hd : emerie.dimes = 7)
  (hn : emerie.nickels = 5) :
  zainTotalCoins emerie = 48 := by
  sorry


end NUMINAMATH_CALUDE_zain_coin_count_l2244_224433


namespace NUMINAMATH_CALUDE_tan_30_15_product_simplification_l2244_224421

theorem tan_30_15_product_simplification :
  (1 + Real.tan (30 * π / 180)) * (1 + Real.tan (15 * π / 180)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_30_15_product_simplification_l2244_224421


namespace NUMINAMATH_CALUDE_point_in_third_quadrant_l2244_224474

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the first quadrant -/
def isInFirstQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- Definition of the third quadrant -/
def isInThirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- Theorem: If A(m,n) is in the first quadrant, then B(-m,-n) is in the third quadrant -/
theorem point_in_third_quadrant
  (A : Point)
  (hA : isInFirstQuadrant A) :
  isInThirdQuadrant (Point.mk (-A.x) (-A.y)) :=
by sorry

end NUMINAMATH_CALUDE_point_in_third_quadrant_l2244_224474


namespace NUMINAMATH_CALUDE_chimney_bricks_count_l2244_224426

/-- The number of bricks in the chimney. -/
def chimney_bricks : ℕ := 288

/-- The time it takes Brenda to build the chimney alone (in hours). -/
def brenda_time : ℕ := 8

/-- The time it takes Brandon to build the chimney alone (in hours). -/
def brandon_time : ℕ := 12

/-- The reduction in combined output when working together (in bricks per hour). -/
def output_reduction : ℕ := 12

/-- The time it takes Brenda and Brandon to build the chimney together (in hours). -/
def combined_time : ℕ := 6

/-- Theorem stating that the number of bricks in the chimney is 288. -/
theorem chimney_bricks_count : 
  chimney_bricks = 288 ∧
  brenda_time = 8 ∧
  brandon_time = 12 ∧
  output_reduction = 12 ∧
  combined_time = 6 ∧
  (combined_time * ((chimney_bricks / brenda_time + chimney_bricks / brandon_time) - output_reduction) = chimney_bricks) :=
by sorry

end NUMINAMATH_CALUDE_chimney_bricks_count_l2244_224426


namespace NUMINAMATH_CALUDE_percentage_relation_l2244_224419

theorem percentage_relation (X A B : ℝ) (hA : A = 0.05 * X) (hB : B = 0.25 * X) :
  A = 0.2 * B := by sorry

end NUMINAMATH_CALUDE_percentage_relation_l2244_224419


namespace NUMINAMATH_CALUDE_arrangements_three_male_two_female_l2244_224448

/-- The number of ways to arrange 3 male and 2 female students in a row,
    such that the female students do not stand at either end -/
def arrangements (n_male : ℕ) (n_female : ℕ) : ℕ :=
  if n_male = 3 ∧ n_female = 2 then
    (n_male + n_female - 2).choose n_female * n_male.factorial
  else
    0

theorem arrangements_three_male_two_female :
  arrangements 3 2 = 36 :=
sorry

end NUMINAMATH_CALUDE_arrangements_three_male_two_female_l2244_224448


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l2244_224432

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the circles
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define the theorem
theorem triangle_area_theorem (ABC : Triangle) (circle1 circle2 : Circle) 
  (L K M N : ℝ × ℝ) :
  -- Given conditions
  circle1.radius = 1/18 →
  circle2.radius = 2/9 →
  (ABC.A.1 - L.1)^2 + (ABC.A.2 - L.2)^2 = (1/9)^2 →
  (ABC.C.1 - M.1)^2 + (ABC.C.2 - M.2)^2 = (1/6)^2 →
  -- Circle1 touches AB at L and AC at K
  ((ABC.A.1 - L.1)^2 + (ABC.A.2 - L.2)^2 = circle1.radius^2 ∧
   (ABC.B.1 - L.1)^2 + (ABC.B.2 - L.2)^2 = circle1.radius^2) →
  ((ABC.A.1 - K.1)^2 + (ABC.A.2 - K.2)^2 = circle1.radius^2 ∧
   (ABC.C.1 - K.1)^2 + (ABC.C.2 - K.2)^2 = circle1.radius^2) →
  -- Circle2 touches AC at N and BC at M
  ((ABC.A.1 - N.1)^2 + (ABC.A.2 - N.2)^2 = circle2.radius^2 ∧
   (ABC.C.1 - N.1)^2 + (ABC.C.2 - N.2)^2 = circle2.radius^2) →
  ((ABC.B.1 - M.1)^2 + (ABC.B.2 - M.2)^2 = circle2.radius^2 ∧
   (ABC.C.1 - M.1)^2 + (ABC.C.2 - M.2)^2 = circle2.radius^2) →
  -- Circles touch each other
  (circle1.center.1 - circle2.center.1)^2 + (circle1.center.2 - circle2.center.2)^2 = (circle1.radius + circle2.radius)^2 →
  -- Conclusion: Area of triangle ABC is 15/11
  abs ((ABC.B.1 - ABC.A.1) * (ABC.C.2 - ABC.A.2) - (ABC.C.1 - ABC.A.1) * (ABC.B.2 - ABC.A.2)) / 2 = 15/11 :=
by sorry


end NUMINAMATH_CALUDE_triangle_area_theorem_l2244_224432


namespace NUMINAMATH_CALUDE_parallelogram_area_l2244_224488

/-- The area of a parallelogram with base 12 and height 6 is 72 -/
theorem parallelogram_area : 
  ∀ (base height area : ℝ), 
  base = 12 → 
  height = 6 → 
  area = base * height → 
  area = 72 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l2244_224488


namespace NUMINAMATH_CALUDE_cost_price_calculation_l2244_224475

/-- 
Given a product with:
- Marked price of 1100 yuan
- Sold at 80% of the marked price
- Makes a 10% profit

Prove that the cost price is 800 yuan
-/
theorem cost_price_calculation (marked_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) 
  (h1 : marked_price = 1100)
  (h2 : discount_rate = 0.8)
  (h3 : profit_rate = 0.1) :
  marked_price * discount_rate = (1 + profit_rate) * 800 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l2244_224475


namespace NUMINAMATH_CALUDE_wipes_per_pack_l2244_224455

theorem wipes_per_pack (wipes_per_day : ℕ) (days : ℕ) (num_packs : ℕ) : 
  wipes_per_day = 2 → days = 360 → num_packs = 6 → 
  (wipes_per_day * days) / num_packs = 120 := by
  sorry

end NUMINAMATH_CALUDE_wipes_per_pack_l2244_224455


namespace NUMINAMATH_CALUDE_inequality_proof_l2244_224496

theorem inequality_proof (b c : ℝ) (hb : b > 0) (hc : c > 0) :
  (b - c)^2011 * (b + c)^2011 * (c - b)^2011 ≥ (b^2011 - c^2011) * (b^2011 + c^2011) * (c^2011 - b^2011) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2244_224496


namespace NUMINAMATH_CALUDE_nickel_probability_is_one_fourth_l2244_224428

/-- Represents the types of coins in the jar -/
inductive Coin
  | Dime
  | Nickel
  | Penny

/-- The value of each coin type in cents -/
def coinValue : Coin → ℚ
  | Coin.Dime => 10
  | Coin.Nickel => 5
  | Coin.Penny => 1

/-- The total value of each coin type in the jar in cents -/
def totalValue : Coin → ℚ
  | Coin.Dime => 500
  | Coin.Nickel => 250
  | Coin.Penny => 100

/-- The number of coins of each type in the jar -/
def coinCount (c : Coin) : ℚ :=
  totalValue c / coinValue c

/-- The total number of coins in the jar -/
def totalCoins : ℚ :=
  coinCount Coin.Dime + coinCount Coin.Nickel + coinCount Coin.Penny

/-- The probability of choosing a nickel from the jar -/
def nickelProbability : ℚ :=
  coinCount Coin.Nickel / totalCoins

theorem nickel_probability_is_one_fourth :
  nickelProbability = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_nickel_probability_is_one_fourth_l2244_224428


namespace NUMINAMATH_CALUDE_linda_age_l2244_224453

/-- Given that Linda's age is 3 more than 2 times Jane's age, and in 5 years
    the sum of their ages will be 28, prove that Linda's current age is 13. -/
theorem linda_age (j : ℕ) : 
  (j + 5) + ((2 * j + 3) + 5) = 28 → 2 * j + 3 = 13 := by
  sorry

end NUMINAMATH_CALUDE_linda_age_l2244_224453


namespace NUMINAMATH_CALUDE_second_number_in_set_l2244_224466

theorem second_number_in_set (x : ℝ) : 
  (20 + 40 + 60) / 3 = (10 + x + 13) / 3 + 9 → x = 70 := by
  sorry

end NUMINAMATH_CALUDE_second_number_in_set_l2244_224466


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2244_224400

theorem complex_equation_solution (m : ℝ) : 
  let z₁ : ℂ := m^2 - 3*m + m^2*Complex.I
  let z₂ : ℂ := 4 + (5*m + 6)*Complex.I
  z₁ - z₂ = 0 → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2244_224400


namespace NUMINAMATH_CALUDE_complex_expressions_calculation_l2244_224440

-- Define complex number i
noncomputable def i : ℂ := Complex.I

-- Define square root of 3
noncomputable def sqrt3 : ℝ := Real.sqrt 3

theorem complex_expressions_calculation :
  -- Expression 1
  ((1 + 2*i)^2 + 3*(1 - i)) / (2 + i) = 1/5 + 2/5*i ∧
  -- Expression 2
  (1 - i) / (1 + i)^2 + (1 + i) / (1 - i)^2 = -1 ∧
  -- Expression 3
  (1 - sqrt3*i) / (sqrt3 + i)^2 = -1/4 - (sqrt3/4)*i :=
by sorry

end NUMINAMATH_CALUDE_complex_expressions_calculation_l2244_224440


namespace NUMINAMATH_CALUDE_complement_intersection_AB_l2244_224446

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

theorem complement_intersection_AB : (U \ (A ∩ B)) = {1, 4, 5} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_AB_l2244_224446


namespace NUMINAMATH_CALUDE_green_area_percentage_l2244_224435

/-- Represents a square flag with a symmetric pattern -/
structure SymmetricFlag where
  side_length : ℝ
  cross_area_percentage : ℝ
  green_area_percentage : ℝ

/-- The flag satisfies the problem conditions -/
def valid_flag (flag : SymmetricFlag) : Prop :=
  flag.cross_area_percentage = 25 ∧
  flag.green_area_percentage > 0 ∧
  flag.green_area_percentage < flag.cross_area_percentage

/-- The theorem to be proved -/
theorem green_area_percentage (flag : SymmetricFlag) :
  valid_flag flag → flag.green_area_percentage = 4 := by
  sorry

end NUMINAMATH_CALUDE_green_area_percentage_l2244_224435


namespace NUMINAMATH_CALUDE_alcohol_concentration_problem_l2244_224405

/-- Proves that the initial concentration of alcohol in the second vessel is 55% --/
theorem alcohol_concentration_problem (vessel1_capacity : ℝ) (vessel1_concentration : ℝ)
  (vessel2_capacity : ℝ) (total_liquid : ℝ) (final_vessel_capacity : ℝ)
  (final_concentration : ℝ) :
  vessel1_capacity = 2 →
  vessel1_concentration = 20 →
  vessel2_capacity = 6 →
  total_liquid = 8 →
  final_vessel_capacity = 10 →
  final_concentration = 37 →
  ∃ vessel2_concentration : ℝ,
    vessel2_concentration = 55 ∧
    vessel1_capacity * (vessel1_concentration / 100) +
    vessel2_capacity * (vessel2_concentration / 100) =
    final_vessel_capacity * (final_concentration / 100) :=
by sorry


end NUMINAMATH_CALUDE_alcohol_concentration_problem_l2244_224405


namespace NUMINAMATH_CALUDE_sector_central_angle_l2244_224411

/-- Proves that a circular sector with radius 4 cm and area 4 cm² has a central angle of 1/4 radians -/
theorem sector_central_angle (r : ℝ) (area : ℝ) (θ : ℝ) : 
  r = 4 → area = 4 → area = 1/2 * r^2 * θ → θ = 1/4 := by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l2244_224411


namespace NUMINAMATH_CALUDE_root_equation_sum_l2244_224469

theorem root_equation_sum (a b c : ℝ) : 
  (∀ x : ℝ, x^2 + 3*x - 1 = 0 → x^4 + a*x^2 + b*x + c = 0) →
  a + b + 4*c + 100 = 93 := by
sorry

end NUMINAMATH_CALUDE_root_equation_sum_l2244_224469


namespace NUMINAMATH_CALUDE_monica_students_count_l2244_224436

/-- The number of students Monica sees each day -/
def monica_total_students : ℕ :=
  let first_class : ℕ := 20
  let second_third_classes : ℕ := 25 + 25
  let fourth_class : ℕ := first_class / 2
  let fifth_sixth_classes : ℕ := 28 + 28
  first_class + second_third_classes + fourth_class + fifth_sixth_classes

/-- Theorem stating the total number of students Monica sees each day -/
theorem monica_students_count : monica_total_students = 136 := by
  sorry

end NUMINAMATH_CALUDE_monica_students_count_l2244_224436


namespace NUMINAMATH_CALUDE_cube_volume_equals_surface_area_l2244_224462

theorem cube_volume_equals_surface_area (s : ℝ) (h : s > 0) :
  s^3 = 6 * s^2 → s = 6 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_equals_surface_area_l2244_224462


namespace NUMINAMATH_CALUDE_line_not_in_second_quadrant_iff_a_ge_two_l2244_224489

/-- A line that does not pass through the second quadrant -/
structure LineNotInSecondQuadrant where
  a : ℝ
  not_in_second_quadrant : ∀ (x y : ℝ), (a - 2) * y = (3 * a - 1) * x - 4 → ¬(x < 0 ∧ y > 0)

/-- The range of values for a when the line does not pass through the second quadrant -/
theorem line_not_in_second_quadrant_iff_a_ge_two (l : LineNotInSecondQuadrant) :
  l.a ∈ Set.Ici 2 ↔ ∀ (x y : ℝ), (l.a - 2) * y = (3 * l.a - 1) * x - 4 → ¬(x < 0 ∧ y > 0) := by
  sorry

end NUMINAMATH_CALUDE_line_not_in_second_quadrant_iff_a_ge_two_l2244_224489


namespace NUMINAMATH_CALUDE_integer_product_100000_l2244_224422

theorem integer_product_100000 : ∃ (a b : ℤ), 
  a * b = 100000 ∧ 
  a % 10 ≠ 0 ∧ 
  b % 10 ≠ 0 ∧ 
  (a = 32 ∨ b = 32) :=
by sorry

end NUMINAMATH_CALUDE_integer_product_100000_l2244_224422


namespace NUMINAMATH_CALUDE_total_pay_theorem_l2244_224467

/-- Calculates the total pay for a worker given regular and overtime hours --/
def totalPay (regularRate : ℕ) (regularHours : ℕ) (overtimeHours : ℕ) : ℕ :=
  let regularPay := regularRate * regularHours
  let overtimeRate := 2 * regularRate
  let overtimePay := overtimeRate * overtimeHours
  regularPay + overtimePay

/-- Theorem stating that the total pay for the given conditions is $186 --/
theorem total_pay_theorem :
  totalPay 3 40 11 = 186 := by
  sorry

end NUMINAMATH_CALUDE_total_pay_theorem_l2244_224467


namespace NUMINAMATH_CALUDE_rectangle_max_area_l2244_224420

/-- A rectangle with integer sides and perimeter 80 has a maximum area of 400. -/
theorem rectangle_max_area :
  ∀ l w : ℕ,
  l > 0 → w > 0 →
  2 * (l + w) = 80 →
  ∀ l' w' : ℕ,
  l' > 0 → w' > 0 →
  2 * (l' + w') = 80 →
  l * w ≤ 400 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l2244_224420


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l2244_224490

def U : Set Int := {-2, -1, 0, 1, 2}

def A : Set Int := {x | ∃ n : Int, x = 2 / (n - 1) ∧ x ∈ U}

theorem complement_of_A_in_U :
  (U \ A) = {0} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l2244_224490


namespace NUMINAMATH_CALUDE_simplify_fraction_l2244_224494

theorem simplify_fraction : (48 : ℚ) / 72 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2244_224494


namespace NUMINAMATH_CALUDE_mat_weavers_problem_l2244_224452

/-- Given that 4 mat-weavers can weave 4 mats in 4 days, prove that 12 mat-weavers
    are needed to weave 36 mats in 12 days. -/
theorem mat_weavers_problem (weavers_group1 mats_group1 days_group1 : ℕ)
                             (mats_group2 days_group2 : ℕ) :
  weavers_group1 = 4 →
  mats_group1 = 4 →
  days_group1 = 4 →
  mats_group2 = 36 →
  days_group2 = 12 →
  (weavers_group1 * mats_group2 * days_group1 = mats_group1 * days_group2 * 12) :=
by sorry

end NUMINAMATH_CALUDE_mat_weavers_problem_l2244_224452


namespace NUMINAMATH_CALUDE_intercepts_satisfy_equation_intercepts_unique_l2244_224481

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := 5 * x - 2 * y - 10 = 0

/-- The x-intercept of the line -/
def x_intercept : ℝ := 2

/-- The y-intercept of the line -/
def y_intercept : ℝ := -5

/-- Theorem stating that the x-intercept and y-intercept satisfy the line equation -/
theorem intercepts_satisfy_equation : 
  line_equation x_intercept 0 ∧ line_equation 0 y_intercept := by
  sorry

/-- Theorem stating that the x-intercept and y-intercept are unique -/
theorem intercepts_unique :
  ∀ x y : ℝ, line_equation x 0 → x = x_intercept ∧
  ∀ x y : ℝ, line_equation 0 y → y = y_intercept := by
  sorry

end NUMINAMATH_CALUDE_intercepts_satisfy_equation_intercepts_unique_l2244_224481


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2244_224482

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a where a₂ + a₇ = 6,
    prove that 3a₄ + a₆ = 12 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ)
    (h_arithmetic : is_arithmetic_sequence a)
    (h_sum : a 2 + a 7 = 6) :
  3 * a 4 + a 6 = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2244_224482


namespace NUMINAMATH_CALUDE_angle_range_for_point_in_first_quadrant_l2244_224443

def is_in_first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

theorem angle_range_for_point_in_first_quadrant (α : ℝ) :
  0 ≤ α ∧ α ≤ 2 * Real.pi →
  is_in_first_quadrant (Real.tan α) (Real.sin α - Real.cos α) →
  (α ∈ Set.Ioo (Real.pi / 4) (Real.pi / 2)) ∨ (α ∈ Set.Ioo Real.pi (5 * Real.pi / 4)) :=
by sorry

end NUMINAMATH_CALUDE_angle_range_for_point_in_first_quadrant_l2244_224443


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l2244_224470

theorem min_value_expression (x : ℝ) : 
  Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2) ≥ 2 * Real.sqrt 5 := by
  sorry

theorem min_value_achievable : 
  ∃ x : ℝ, Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2) = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l2244_224470


namespace NUMINAMATH_CALUDE_jasons_library_visits_l2244_224425

/-- Jason's library visits in 4 weeks -/
def jasons_visits (williams_weekly_visits : ℕ) (jasons_multiplier : ℕ) (weeks : ℕ) : ℕ :=
  williams_weekly_visits * jasons_multiplier * weeks

/-- Theorem: Jason's library visits in 4 weeks -/
theorem jasons_library_visits :
  jasons_visits 2 4 4 = 32 := by
  sorry

end NUMINAMATH_CALUDE_jasons_library_visits_l2244_224425


namespace NUMINAMATH_CALUDE_coefficient_of_x_l2244_224437

theorem coefficient_of_x (x : ℝ) : 
  let expr := 4*(x - 5) + 3*(2 - 3*x^2 + 6*x) - 10*(3*x - 2)
  ∃ (a b c : ℝ), expr = a*x^2 + (-8)*x + c :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_l2244_224437


namespace NUMINAMATH_CALUDE_sqrt_seven_to_sixth_l2244_224491

theorem sqrt_seven_to_sixth : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_seven_to_sixth_l2244_224491


namespace NUMINAMATH_CALUDE_triangle_inequality_l2244_224456

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a < b + c) (hbc : b < a + c) (hca : c < a + b) :
  a / (b + c - a) + b / (c + a - b) + c / (a + b - c) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2244_224456


namespace NUMINAMATH_CALUDE_equation_equality_l2244_224442

theorem equation_equality (a : ℝ) (h : a ≠ 0) :
  ((1 / a) / ((1 / a) * (1 / a)) - 1 / a) / (1 / a) = (a + 1) * (a - 1) := by
  sorry

end NUMINAMATH_CALUDE_equation_equality_l2244_224442


namespace NUMINAMATH_CALUDE_oscar_height_l2244_224430

/-- Represents the heights of four brothers -/
structure BrothersHeights where
  tobias : ℝ
  victor : ℝ
  peter : ℝ
  oscar : ℝ

/-- The conditions of the problem -/
def heightConditions (h : BrothersHeights) : Prop :=
  h.tobias = 184 ∧
  h.victor - h.tobias = h.tobias - h.peter ∧
  h.peter - h.oscar = h.victor - h.tobias ∧
  (h.tobias + h.victor + h.peter + h.oscar) / 4 = 178

/-- The theorem to prove -/
theorem oscar_height (h : BrothersHeights) :
  heightConditions h → h.oscar = 160 := by
  sorry

end NUMINAMATH_CALUDE_oscar_height_l2244_224430


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_l2244_224495

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- Vector addition -/
def vec_add (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 + b.1, a.2 + b.2)

theorem parallel_vectors_sum (x : ℝ) :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (x, 2)
  are_parallel a b → vec_add a b = (6, 3) := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_l2244_224495


namespace NUMINAMATH_CALUDE_ternary_221_greater_than_binary_10111_l2244_224444

/-- Converts a ternary number (represented as a list of digits) to decimal --/
def ternary_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3^i)) 0

/-- Converts a binary number (represented as a list of digits) to decimal --/
def binary_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (2^i)) 0

/-- The ternary number 221 --/
def a : List Nat := [1, 2, 2]

/-- The binary number 10111 --/
def b : List Nat := [1, 1, 1, 0, 1]

theorem ternary_221_greater_than_binary_10111 :
  ternary_to_decimal a > binary_to_decimal b := by
  sorry

end NUMINAMATH_CALUDE_ternary_221_greater_than_binary_10111_l2244_224444
