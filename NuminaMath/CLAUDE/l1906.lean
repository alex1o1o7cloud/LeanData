import Mathlib

namespace NUMINAMATH_CALUDE_not_or_implies_both_false_l1906_190695

theorem not_or_implies_both_false (p q : Prop) :
  ¬(p ∨ q) → ¬p ∧ ¬q := by
  sorry

end NUMINAMATH_CALUDE_not_or_implies_both_false_l1906_190695


namespace NUMINAMATH_CALUDE_triangle_segment_length_l1906_190654

/-- Triangle ABC with points D and E on BC -/
structure TriangleABC where
  /-- Length of side AB -/
  AB : ℝ
  /-- Length of side BC -/
  BC : ℝ
  /-- Length of side CA -/
  CA : ℝ
  /-- Length of CD -/
  CD : ℝ
  /-- Ratio of BE to EC -/
  BE_EC_ratio : ℝ
  /-- Equality of angles BAE and CAD -/
  angle_equality : Bool

/-- The main theorem -/
theorem triangle_segment_length 
  (t : TriangleABC) 
  (h1 : t.AB = 12) 
  (h2 : t.BC = 16) 
  (h3 : t.CA = 15) 
  (h4 : t.CD = 5) 
  (h5 : t.BE_EC_ratio = 3) 
  (h6 : t.angle_equality = true) : 
  ∃ (BE : ℝ), BE = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_segment_length_l1906_190654


namespace NUMINAMATH_CALUDE_carnation_percentage_l1906_190686

/-- Represents a bouquet of flowers -/
structure Bouquet where
  total : ℝ
  pink : ℝ
  red : ℝ
  pink_roses : ℝ
  pink_carnations : ℝ
  red_roses : ℝ
  red_carnations : ℝ

/-- The theorem stating the percentage of carnations in the bouquet -/
theorem carnation_percentage (b : Bouquet) : 
  b.pink + b.red = b.total →
  b.pink_roses + b.pink_carnations = b.pink →
  b.red_roses + b.red_carnations = b.red →
  b.pink_roses = b.pink / 2 →
  b.red_carnations = b.red * 2 / 3 →
  b.pink = b.total * 7 / 10 →
  (b.pink_carnations + b.red_carnations) / b.total = 11 / 20 := by
sorry

end NUMINAMATH_CALUDE_carnation_percentage_l1906_190686


namespace NUMINAMATH_CALUDE_max_additional_plates_l1906_190678

/-- Represents the sets of letters for each position in the license plate --/
structure LicensePlateSets :=
  (first : Finset Char)
  (second : Finset Char)
  (third : Finset Char)
  (fourth : Finset Char)

/-- The initial license plate sets --/
def initialSets : LicensePlateSets :=
  { first := {'A', 'E', 'I', 'O', 'U'},
    second := {'B', 'C', 'D'},
    third := {'L', 'M', 'N', 'P'},
    fourth := {'S', 'T'} }

/-- The number of new letters that can be added --/
def newLettersCount : Nat := 3

/-- The maximum number of letters that can be added to a single set --/
def maxAddToSet : Nat := 2

/-- Calculates the number of possible license plates --/
def calculatePlates (sets : LicensePlateSets) : Nat :=
  sets.first.card * sets.second.card * sets.third.card * sets.fourth.card

/-- Theorem: The maximum number of additional license plates is 180 --/
theorem max_additional_plates :
  ∃ (newSets : LicensePlateSets),
    (calculatePlates newSets - calculatePlates initialSets = 180) ∧
    (∀ (otherSets : LicensePlateSets),
      (calculatePlates otherSets - calculatePlates initialSets) ≤ 180) :=
sorry


end NUMINAMATH_CALUDE_max_additional_plates_l1906_190678


namespace NUMINAMATH_CALUDE_tangent_point_and_perpendicular_line_l1906_190662

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Define the point P₀
structure Point where
  x : ℝ
  y : ℝ

def P₀ : Point := ⟨-1, -4⟩

-- Define the third quadrant
def in_third_quadrant (p : Point) : Prop := p.x < 0 ∧ p.y < 0

-- Define the tangent line slope
def tangent_slope : ℝ := 4

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := x + 4 * y + 17 = 0

theorem tangent_point_and_perpendicular_line :
  f P₀.x = P₀.y ∧
  f' P₀.x = tangent_slope ∧
  in_third_quadrant P₀ →
  perpendicular_line P₀.x P₀.y :=
by sorry

end NUMINAMATH_CALUDE_tangent_point_and_perpendicular_line_l1906_190662


namespace NUMINAMATH_CALUDE_min_value_expression_l1906_190641

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (6 * c) / (3 * a + b) + (6 * a) / (b + 3 * c) + (2 * b) / (a + c) ≥ 12 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1906_190641


namespace NUMINAMATH_CALUDE_smallest_special_number_after_3429_l1906_190649

/-- A function that returns true if a natural number uses exactly four different digits -/
def uses_four_different_digits (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  n = a * 1000 + b * 100 + c * 10 + d ∧
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10

theorem smallest_special_number_after_3429 :
  ∀ k : ℕ, k > 3429 ∧ k < 3450 → ¬(uses_four_different_digits k) ∧
  uses_four_different_digits 3450 :=
by sorry

end NUMINAMATH_CALUDE_smallest_special_number_after_3429_l1906_190649


namespace NUMINAMATH_CALUDE_mutuallyExclusiveNotContradictoryPairs_l1906_190694

-- Define the events
inductive Event : Type
| Miss : Event
| Hit : Event
| MoreThan4 : Event
| AtLeast5 : Event

-- Define mutual exclusivity
def mutuallyExclusive (e1 e2 : Event) : Prop := sorry

-- Define contradictory events
def contradictory (e1 e2 : Event) : Prop := sorry

-- Define a function to count pairs of events that are mutually exclusive but not contradictory
def countMutuallyExclusiveNotContradictory (events : List Event) : Nat := sorry

-- Theorem to prove
theorem mutuallyExclusiveNotContradictoryPairs :
  let events := [Event.Miss, Event.Hit, Event.MoreThan4, Event.AtLeast5]
  countMutuallyExclusiveNotContradictory events = 2 := by sorry

end NUMINAMATH_CALUDE_mutuallyExclusiveNotContradictoryPairs_l1906_190694


namespace NUMINAMATH_CALUDE_quarters_to_nickels_difference_l1906_190612

/-- The difference in money (in nickels) between two people given their quarter amounts -/
def money_difference_in_nickels (charles_quarters richard_quarters : ℕ) : ℤ :=
  5 * (charles_quarters - richard_quarters)

theorem quarters_to_nickels_difference (q : ℕ) :
  money_difference_in_nickels (5 * q + 3) (q + 7) = 20 * (q - 1) := by
  sorry

end NUMINAMATH_CALUDE_quarters_to_nickels_difference_l1906_190612


namespace NUMINAMATH_CALUDE_train_length_calculation_l1906_190674

/-- The length of each train in kilometers -/
def train_length : ℝ := 0.06

/-- The speed of the faster train in km/hr -/
def fast_train_speed : ℝ := 48

/-- The speed of the slower train in km/hr -/
def slow_train_speed : ℝ := 36

/-- The time taken for the faster train to pass the slower train in seconds -/
def passing_time : ℝ := 36

theorem train_length_calculation :
  let relative_speed := fast_train_speed - slow_train_speed
  let relative_speed_km_per_sec := relative_speed / 3600
  2 * train_length = relative_speed_km_per_sec * passing_time :=
by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l1906_190674


namespace NUMINAMATH_CALUDE_triangle_max_area_l1906_190696

theorem triangle_max_area (A B C : ℝ) (h1 : 0 < A) (h2 : A < π) (h3 : 0 < B) (h4 : B < π) 
  (h5 : 0 < C) (h6 : C < π) (h7 : A + B + C = π) 
  (h8 : Real.tan A * Real.tan B = 1) (h9 : Real.sqrt 3 = 2 * Real.sin (A / 2) * Real.sin (B / 2) * Real.sin (C / 2)) :
  (∃ (S : ℝ → ℝ), (∀ x, S x ≤ S (π/4)) ∧ S A = (3/4) * Real.sin (2*A)) :=
sorry

end NUMINAMATH_CALUDE_triangle_max_area_l1906_190696


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1906_190665

theorem geometric_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (t : ℝ) :
  (∀ n, S n = 2017 * 2016^n - 2018 * t) →
  (∀ n, S (n + 1) - S n = a (n + 1)) →
  (∀ n, a (n + 1) = a n * (a 2 / a 1)) →
  t = 2017 / 2018 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1906_190665


namespace NUMINAMATH_CALUDE_unique_prime_square_sum_l1906_190638

theorem unique_prime_square_sum (p q : ℕ) : 
  Prime p → Prime q → ∃ (n : ℕ), p^(q+1) + q^(p+1) = n^2 → p = 2 ∧ q = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_square_sum_l1906_190638


namespace NUMINAMATH_CALUDE_bungee_cord_extension_l1906_190635

/-- The maximum extension of a bungee cord in a bungee jumping scenario -/
theorem bungee_cord_extension
  (m : ℝ) -- mass of the person
  (H : ℝ) -- maximum fall distance
  (k : ℝ) -- spring constant of the bungee cord
  (L₀ : ℝ) -- original length of the bungee cord
  (h : ℝ) -- extension of the bungee cord
  (g : ℝ) -- gravitational acceleration
  (hpos : h > 0)
  (mpos : m > 0)
  (kpos : k > 0)
  (Hpos : H > 0)
  (L₀pos : L₀ > 0)
  (gpos : g > 0)
  (hooke : k * h = 4 * m * g) -- Hooke's law and maximum tension condition
  (energy : m * g * H = (1/2) * k * h^2) -- Conservation of energy
  : h = H / 2 := by
  sorry

end NUMINAMATH_CALUDE_bungee_cord_extension_l1906_190635


namespace NUMINAMATH_CALUDE_triangle_properties_l1906_190608

noncomputable section

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given conditions for the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  t.b * Real.sin t.A = Real.sqrt 3 * t.a * Real.cos t.B ∧
  t.b = 3 ∧
  Real.sin t.C = 2 * Real.sin t.A

theorem triangle_properties (t : Triangle) (h : TriangleConditions t) :
  t.B = π / 3 ∧ (1 / 2) * t.a * t.c * Real.sin t.B = (3 * Real.sqrt 3) / 2 :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l1906_190608


namespace NUMINAMATH_CALUDE_prime_numbers_existence_l1906_190647

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem prime_numbers_existence : 
  ∃ (a : ℕ), 
    a < 10 ∧ 
    is_prime (11*a - 1) ∧ 
    is_prime (10*a + 1) ∧ 
    is_prime (10*a + 7) ∧ 
    a = 4 :=
sorry

end NUMINAMATH_CALUDE_prime_numbers_existence_l1906_190647


namespace NUMINAMATH_CALUDE_total_cost_of_suits_l1906_190658

def cost_of_first_suit : ℕ := 300

def cost_of_second_suit (first_suit_cost : ℕ) : ℕ :=
  3 * first_suit_cost + 200

def total_cost (first_suit_cost : ℕ) : ℕ :=
  first_suit_cost + cost_of_second_suit first_suit_cost

theorem total_cost_of_suits :
  total_cost cost_of_first_suit = 1400 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_of_suits_l1906_190658


namespace NUMINAMATH_CALUDE_tea_hot_chocolate_difference_mo_drink_difference_l1906_190632

/-- Represents the drinking habits and week data for Mo --/
structure MoDrinkingHabits where
  n : ℕ  -- Number of hot chocolate cups on rainy days
  total_cups : ℕ  -- Total cups drunk in a week
  rainy_days : ℕ  -- Number of rainy days in a week

/-- Theorem stating the difference between tea and hot chocolate cups --/
theorem tea_hot_chocolate_difference (mo : MoDrinkingHabits) 
  (h1 : mo.total_cups = 26)
  (h2 : mo.rainy_days = 1) :
  3 * (7 - mo.rainy_days) - mo.n * mo.rainy_days = 10 := by
  sorry

/-- Main theorem proving the difference is 10 --/
theorem mo_drink_difference : ∃ mo : MoDrinkingHabits, 
  mo.total_cups = 26 ∧ 
  mo.rainy_days = 1 ∧ 
  3 * (7 - mo.rainy_days) - mo.n * mo.rainy_days = 10 := by
  sorry

end NUMINAMATH_CALUDE_tea_hot_chocolate_difference_mo_drink_difference_l1906_190632


namespace NUMINAMATH_CALUDE_demand_exceeds_15000_only_in_7_and_8_l1906_190682

def S (n : ℕ) : ℚ := (n : ℚ) / 90 * (21 * n - n^2 - 5)

def a (n : ℕ) : ℚ := S n - S (n-1)

theorem demand_exceeds_15000_only_in_7_and_8 :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 12 →
    (a n > (3/2) ↔ n = 7 ∨ n = 8) :=
sorry

end NUMINAMATH_CALUDE_demand_exceeds_15000_only_in_7_and_8_l1906_190682


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l1906_190687

theorem coefficient_x_squared_in_expansion (x : ℝ) : 
  (Finset.range 6).sum (fun k => Nat.choose 5 k * (2 * x) ^ k * 1 ^ (5 - k)) = 
  1 + 10 * x + 40 * x^2 + 80 * x^3 + 80 * x^4 + 32 * x^5 :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l1906_190687


namespace NUMINAMATH_CALUDE_total_wheels_is_119_l1906_190645

/-- The total number of wheels in Liam's three garages --/
def total_wheels : ℕ :=
  let first_garage := 
    (3 * 2) + 2 + (6 * 3) + (9 * 1) + (3 * 4)
  let second_garage := 
    (2 * 2) + (1 * 3) + (3 * 1) + (4 * 4) + (1 * 5) + 2
  let third_garage := 
    (3 * 2) + (4 * 3) + 1 + 1 + (2 * 4) + (1 * 5) + 7 - 1
  first_garage + second_garage + third_garage

/-- Theorem stating that the total number of wheels in Liam's three garages is 119 --/
theorem total_wheels_is_119 : total_wheels = 119 := by
  sorry

end NUMINAMATH_CALUDE_total_wheels_is_119_l1906_190645


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l1906_190604

theorem infinite_geometric_series_first_term 
  (r : ℚ) 
  (S : ℚ) 
  (h1 : r = 1 / 4) 
  (h2 : S = 40) 
  (h3 : S = a / (1 - r)) : 
  a = 30 := by
  sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l1906_190604


namespace NUMINAMATH_CALUDE_abs_x_squared_minus_x_lt_two_l1906_190643

theorem abs_x_squared_minus_x_lt_two (x : ℝ) :
  |x^2 - x| < 2 ↔ -1 < x ∧ x < 2 := by sorry

end NUMINAMATH_CALUDE_abs_x_squared_minus_x_lt_two_l1906_190643


namespace NUMINAMATH_CALUDE_train_platform_problem_l1906_190609

/-- The length of a train in meters. -/
def train_length : ℝ := 110

/-- The time taken to cross the first platform in seconds. -/
def time_first : ℝ := 15

/-- The time taken to cross the second platform in seconds. -/
def time_second : ℝ := 20

/-- The length of the second platform in meters. -/
def second_platform_length : ℝ := 250

/-- The length of the first platform in meters. -/
def first_platform_length : ℝ := 160

theorem train_platform_problem :
  (train_length + first_platform_length) / time_first =
  (train_length + second_platform_length) / time_second :=
sorry

end NUMINAMATH_CALUDE_train_platform_problem_l1906_190609


namespace NUMINAMATH_CALUDE_constant_term_expansion_l1906_190627

/-- The constant term in the expansion of (3+x)(x+1/x)^6 -/
def constant_term : ℕ := 60

/-- Theorem: The constant term in the expansion of (3+x)(x+1/x)^6 is 60 -/
theorem constant_term_expansion :
  constant_term = 60 := by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l1906_190627


namespace NUMINAMATH_CALUDE_bread_slices_problem_l1906_190660

theorem bread_slices_problem (initial_slices : ℕ) : 
  (initial_slices : ℚ) * (2/3) - 2 = 6 → initial_slices = 12 := by
  sorry

end NUMINAMATH_CALUDE_bread_slices_problem_l1906_190660


namespace NUMINAMATH_CALUDE_infinite_series_sum_l1906_190610

/-- The sum of the infinite series ∑(n=1 to ∞) (5n-2)/(3^n) is equal to 11/4 -/
theorem infinite_series_sum : 
  (∑' n : ℕ, (5 * n - 2 : ℝ) / (3 ^ n)) = 11 / 4 := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l1906_190610


namespace NUMINAMATH_CALUDE_equation_represents_intersecting_lines_l1906_190684

theorem equation_represents_intersecting_lines (x y : ℝ) :
  (x + y)^2 = x^2 + y^2 + 3*x*y ↔ x*y = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_represents_intersecting_lines_l1906_190684


namespace NUMINAMATH_CALUDE_circle_area_l1906_190690

theorem circle_area (x y : ℝ) : 
  (2 * x^2 + 2 * y^2 + 8 * x - 4 * y - 16 = 0) → 
  (∃ (center_x center_y radius : ℝ), 
    (x - center_x)^2 + (y - center_y)^2 = radius^2 ∧ 
    π * radius^2 = 13 * π) :=
by sorry

end NUMINAMATH_CALUDE_circle_area_l1906_190690


namespace NUMINAMATH_CALUDE_tangent_fraction_equality_l1906_190677

theorem tangent_fraction_equality (α β : Real) 
  (h1 : Real.tan (α - β) = 2) 
  (h2 : Real.tan β = 4) : 
  (7 * Real.sin α - Real.cos α) / (7 * Real.sin α + Real.cos α) = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_tangent_fraction_equality_l1906_190677


namespace NUMINAMATH_CALUDE_committee_arrangement_count_l1906_190601

/-- The number of ways to arrange n indistinguishable objects of type A
    and m indistinguishable objects of type B in a row of (n+m) positions -/
def arrangement_count (n m : ℕ) : ℕ :=
  Nat.choose (n + m) m

/-- Theorem stating that there are 120 ways to arrange 7 indistinguishable objects
    and 3 indistinguishable objects in a row of 10 positions -/
theorem committee_arrangement_count :
  arrangement_count 7 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_committee_arrangement_count_l1906_190601


namespace NUMINAMATH_CALUDE_cube_surface_area_l1906_190631

theorem cube_surface_area (volume : ℝ) (h : volume = 64) : 
  (6 : ℝ) * (volume ^ (1/3 : ℝ))^2 = 96 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l1906_190631


namespace NUMINAMATH_CALUDE_unit_conversions_l1906_190603

-- Define unit conversion factors
def cm_to_dm : ℚ := 1 / 10
def hectare_to_km2 : ℚ := 1 / 100
def yuan_to_jiao : ℚ := 10
def yuan_to_fen : ℚ := 100
def hectare_to_m2 : ℚ := 10000
def dm_to_m : ℚ := 1 / 10
def m_to_cm : ℚ := 100

-- Theorem statement
theorem unit_conversions :
  (70000 * cm_to_dm^2 = 700) ∧
  (800 * hectare_to_km2 = 8) ∧
  (1.65 * yuan_to_jiao = 16.5) ∧
  (400 * hectare_to_m2 = 4000000) ∧
  (0.57 * yuan_to_fen = 57) ∧
  (5000 * dm_to_m^2 = 50) ∧
  (60000 / hectare_to_m2 = 6) ∧
  (9 * m_to_cm = 900) :=
by sorry

end NUMINAMATH_CALUDE_unit_conversions_l1906_190603


namespace NUMINAMATH_CALUDE_no_solutions_diophantine_equation_l1906_190602

theorem no_solutions_diophantine_equation :
  ¬∃ (n x y k : ℕ), n ≥ 1 ∧ x > 0 ∧ y > 0 ∧ k > 1 ∧ 
  Nat.gcd x y = 1 ∧ 3^n = x^k + y^k :=
sorry

end NUMINAMATH_CALUDE_no_solutions_diophantine_equation_l1906_190602


namespace NUMINAMATH_CALUDE_cubic_function_properties_l1906_190628

/-- A cubic function f(x) = ax³ - bx² + c with a > 0 -/
def f (a b c x : ℝ) : ℝ := a * x^3 - b * x^2 + c

/-- The derivative of f with respect to x -/
def f_deriv (a b x : ℝ) : ℝ := 3 * a * x^2 - 2 * b * x

theorem cubic_function_properties
  (a b c : ℝ)
  (ha : a > 0) :
  -- 1. Extreme points when b = 3a
  (b = 3 * a → (∀ x : ℝ, f_deriv a b x = 0 ↔ x = 0 ∨ x = 2)) ∧
  -- 2. Range of b when a = 1 and x²ln(x) ≥ f(x) - 2x - c for x ∈ [3,4]
  (a = 1 → (∀ x : ℝ, x ∈ Set.Icc 3 4 → x^2 * Real.log x ≥ f 1 b c x - 2*x - c) →
    b ≥ 7/2 - Real.log 4) ∧
  -- 3. Existence of three tangent lines
  (b = 3 * a → 5 * a < c → c < 6 * a →
    ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
      f a b c x₁ + f_deriv a b x₁ * (2 - x₁) = a ∧
      f a b c x₂ + f_deriv a b x₂ * (2 - x₂) = a ∧
      f a b c x₃ + f_deriv a b x₃ * (2 - x₃) = a) :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l1906_190628


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1906_190615

theorem contrapositive_equivalence (x : ℝ) :
  (¬(x^2 < 1) → ¬(-1 < x ∧ x < 1)) ↔ ((x ≤ -1 ∨ x ≥ 1) → x^2 ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l1906_190615


namespace NUMINAMATH_CALUDE_polynomial_coefficient_theorem_l1906_190692

theorem polynomial_coefficient_theorem (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, x^4 = a₀ + a₁*(x+2) + a₂*(x+2)^2 + a₃*(x+2)^3 + a₄*(x+2)^4) →
  a₃ = -8 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_theorem_l1906_190692


namespace NUMINAMATH_CALUDE_incorrect_rectangle_l1906_190637

/-- Represents a 3x3 grid of rectangle perimeters --/
structure PerimeterGrid :=
  (top_row : Fin 3 → ℕ)
  (middle_row : Fin 3 → ℕ)
  (bottom_row : Fin 3 → ℕ)

/-- The given grid of perimeters --/
def given_grid : PerimeterGrid :=
  { top_row := ![14, 16, 12],
    middle_row := ![18, 18, 2],
    bottom_row := ![16, 18, 14] }

/-- Predicate to check if a perimeter grid is valid --/
def is_valid_grid (grid : PerimeterGrid) : Prop :=
  ∀ i j, i < 3 → j < 3 → 
    (grid.top_row i > 0) ∧ 
    (grid.middle_row i > 0) ∧ 
    (grid.bottom_row i > 0)

/-- Theorem stating that the rectangle with perimeter 2 is incorrect --/
theorem incorrect_rectangle (grid : PerimeterGrid) 
  (h : is_valid_grid grid) : 
  ∃ i j, grid.middle_row j = 2 ∧ 
    (i = 1 ∨ j = 2) ∧ 
    ¬(∀ k l, k ≠ i ∨ l ≠ j → grid.middle_row l > 2) :=
sorry


end NUMINAMATH_CALUDE_incorrect_rectangle_l1906_190637


namespace NUMINAMATH_CALUDE_triangle_must_be_obtuse_l1906_190640

-- Define a triangle with sides a, b, c and angles A, B, C
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the properties of the triangle
def TriangleProperties (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = Real.pi ∧
  (t.a = 2 * t.b ∨ t.b = 2 * t.c ∨ t.c = 2 * t.a) ∧
  (t.A = Real.pi / 6 ∨ t.B = Real.pi / 6 ∨ t.C = Real.pi / 6)

-- Define an obtuse triangle
def IsObtuseTriangle (t : Triangle) : Prop :=
  t.A > Real.pi / 2 ∨ t.B > Real.pi / 2 ∨ t.C > Real.pi / 2

-- Theorem statement
theorem triangle_must_be_obtuse (t : Triangle) (h : TriangleProperties t) : IsObtuseTriangle t :=
sorry

end NUMINAMATH_CALUDE_triangle_must_be_obtuse_l1906_190640


namespace NUMINAMATH_CALUDE_conic_section_is_hyperbola_l1906_190624

/-- The conic section represented by the equation (2x-7)^2 - 4(y+3)^2 = 169 is a hyperbola. -/
theorem conic_section_is_hyperbola :
  ∃ (a b c d e f : ℝ), 
    (∀ x y : ℝ, (2*x - 7)^2 - 4*(y + 3)^2 = 169 ↔ a*x^2 + b*y^2 + c*x + d*y + e*x*y + f = 0) ∧
    (a > 0 ∧ b < 0) := by
  sorry

end NUMINAMATH_CALUDE_conic_section_is_hyperbola_l1906_190624


namespace NUMINAMATH_CALUDE_annas_money_l1906_190689

theorem annas_money (original spent remaining : ℚ) : 
  spent = (1 : ℚ) / 4 * original →
  remaining = (3 : ℚ) / 4 * original →
  remaining = 24 →
  original = 32 := by
sorry

end NUMINAMATH_CALUDE_annas_money_l1906_190689


namespace NUMINAMATH_CALUDE_negation_of_monotonicity_like_property_l1906_190667

/-- The negation of a monotonicity-like property for a real-valued function -/
theorem negation_of_monotonicity_like_property (f : ℝ → ℝ) :
  (¬ ∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0) ↔
  (∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_monotonicity_like_property_l1906_190667


namespace NUMINAMATH_CALUDE_prob_heads_tails_heads_l1906_190636

/-- A fair coin has equal probability of landing heads or tails -/
def fair_coin (p : ℝ) : Prop := p = 1 / 2

/-- The probability of a specific sequence of n independent events -/
def prob_sequence (p : ℝ) (n : ℕ) : ℝ := p ^ n

/-- The probability of getting heads, then tails, then heads when flipping a fair coin three times -/
theorem prob_heads_tails_heads (p : ℝ) (h_fair : fair_coin p) : 
  prob_sequence p 3 = 1 / 8 := by sorry

end NUMINAMATH_CALUDE_prob_heads_tails_heads_l1906_190636


namespace NUMINAMATH_CALUDE_set_A_proof_l1906_190679

def U : Set Nat := {0, 1, 2, 3, 4, 5}

theorem set_A_proof (A B : Set Nat) 
  (h1 : A ⊆ U)
  (h2 : B ⊆ U)
  (h3 : (U \ A) ∩ B = {0, 4})
  (h4 : (U \ A) ∩ (U \ B) = {3, 5}) :
  A = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_set_A_proof_l1906_190679


namespace NUMINAMATH_CALUDE_unique_k_value_l1906_190676

/-- The equation has infinitely many solutions when the coefficients of x are equal on both sides -/
def has_infinitely_many_solutions (k : ℝ) : Prop :=
  3 * k = 15

/-- The value of k for which the equation has infinitely many solutions -/
def k_value : ℝ := 5

/-- Theorem stating that k_value is the unique solution -/
theorem unique_k_value :
  has_infinitely_many_solutions k_value ∧
  ∀ k : ℝ, has_infinitely_many_solutions k → k = k_value :=
by sorry

end NUMINAMATH_CALUDE_unique_k_value_l1906_190676


namespace NUMINAMATH_CALUDE_factorization_equality_l1906_190630

theorem factorization_equality (a b : ℝ) : 2*a*b - a^2 - b^2 + 4 = (2 + a - b)*(2 - a + b) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1906_190630


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l1906_190646

open Set

def A : Set ℝ := {x : ℝ | -3 < x ∧ x < 6}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 7}

theorem intersection_A_complement_B : A ∩ (𝒰 \ B) = Ioo (-3) 2 ∪ {2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l1906_190646


namespace NUMINAMATH_CALUDE_min_value_fraction_sum_l1906_190629

theorem min_value_fraction_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (1 / a + 9 / b ≥ 8) ∧
  (1 / a + 9 / b = 8 ↔ a = 1/2 ∧ b = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_sum_l1906_190629


namespace NUMINAMATH_CALUDE_example_rearrangements_l1906_190620

def word : String := "EXAMPLE"

def vowels : List Char := ['E', 'E', 'A']
def consonants : List Char := ['X', 'M', 'P', 'L']

def vowel_arrangements : ℕ := 3
def consonant_arrangements : ℕ := 24

theorem example_rearrangements :
  (vowel_arrangements * consonant_arrangements) = 72 :=
by sorry

end NUMINAMATH_CALUDE_example_rearrangements_l1906_190620


namespace NUMINAMATH_CALUDE_tony_age_l1906_190619

/-- Given that Tony and Belinda have a combined age of 56, and Belinda is 40 years old,
    prove that Tony is 16 years old. -/
theorem tony_age (total_age : ℕ) (belinda_age : ℕ) (h1 : total_age = 56) (h2 : belinda_age = 40) :
  total_age - belinda_age = 16 := by
  sorry

end NUMINAMATH_CALUDE_tony_age_l1906_190619


namespace NUMINAMATH_CALUDE_supermarket_prices_l1906_190650

/-- The price of sugar per kilogram -/
def sugar_price : ℝ := sorry

/-- The price of salt per kilogram -/
def salt_price : ℝ := sorry

/-- The price of rice per kilogram -/
def rice_price : ℝ := sorry

/-- The total price of given quantities of sugar, salt, and rice -/
def total_price (sugar_kg salt_kg rice_kg : ℝ) : ℝ :=
  sugar_kg * sugar_price + salt_kg * salt_price + rice_kg * rice_price

theorem supermarket_prices :
  (total_price 5 3 2 = 28) ∧
  (total_price 4 2 1 = 22) ∧
  (sugar_price = 2 * salt_price) ∧
  (rice_price = 3 * salt_price) →
  total_price 6 4 3 = 36.75 := by
sorry

end NUMINAMATH_CALUDE_supermarket_prices_l1906_190650


namespace NUMINAMATH_CALUDE_set_A_equals_zero_one_l1906_190691

def A : Set ℤ := {x | (2 * x - 3) / (x + 1) ≤ 0}

theorem set_A_equals_zero_one : A = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_set_A_equals_zero_one_l1906_190691


namespace NUMINAMATH_CALUDE_tv_production_last_period_avg_l1906_190673

/-- Represents the production of TVs in a factory over a month -/
structure TVProduction where
  totalDays : Nat
  firstPeriodDays : Nat
  firstPeriodAvg : Nat
  monthlyAvg : Nat

/-- Calculates the average production for the last period of the month -/
def lastPeriodAvg (p : TVProduction) : Rat :=
  let lastPeriodDays := p.totalDays - p.firstPeriodDays
  let totalProduction := p.totalDays * p.monthlyAvg
  let firstPeriodProduction := p.firstPeriodDays * p.firstPeriodAvg
  (totalProduction - firstPeriodProduction) / lastPeriodDays

/-- Theorem stating that given the conditions, the average production for the last 5 days is 20 TVs per day -/
theorem tv_production_last_period_avg 
  (p : TVProduction) 
  (h1 : p.totalDays = 30) 
  (h2 : p.firstPeriodDays = 25) 
  (h3 : p.firstPeriodAvg = 50) 
  (h4 : p.monthlyAvg = 45) : 
  lastPeriodAvg p = 20 := by
  sorry

end NUMINAMATH_CALUDE_tv_production_last_period_avg_l1906_190673


namespace NUMINAMATH_CALUDE_b_cubed_is_zero_l1906_190623

theorem b_cubed_is_zero (B : Matrix (Fin 3) (Fin 3) ℝ) (h : B ^ 4 = 0) : B ^ 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_b_cubed_is_zero_l1906_190623


namespace NUMINAMATH_CALUDE_converse_not_always_true_l1906_190648

theorem converse_not_always_true : 
  ¬ (∀ (a b m : ℝ), a < b → a * m^2 < b * m^2) :=
by sorry

end NUMINAMATH_CALUDE_converse_not_always_true_l1906_190648


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l1906_190600

theorem arithmetic_mean_problem (m n : ℝ) 
  (h1 : (m + 2*n) / 2 = 4) 
  (h2 : (2*m + n) / 2 = 5) : 
  (m + n) / 2 = 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l1906_190600


namespace NUMINAMATH_CALUDE_problem_solution_l1906_190681

theorem problem_solution (x y : ℤ) (h1 : x + y = 250) (h2 : x - y = 200) : y = 25 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1906_190681


namespace NUMINAMATH_CALUDE_quotient_change_l1906_190663

theorem quotient_change (a b : ℝ) (h : b ≠ 0) :
  ((100 * a) / (b / 10)) = 1000 * (a / b) := by
sorry

end NUMINAMATH_CALUDE_quotient_change_l1906_190663


namespace NUMINAMATH_CALUDE_number_of_divisors_of_m_l1906_190614

def m : ℕ := 2^5 * 3^6 * 5^7 * 7^8

theorem number_of_divisors_of_m : 
  (Finset.filter (· ∣ m) (Finset.range (m + 1))).card = 3024 :=
sorry

end NUMINAMATH_CALUDE_number_of_divisors_of_m_l1906_190614


namespace NUMINAMATH_CALUDE_average_age_decrease_l1906_190693

theorem average_age_decrease (initial_average : ℝ) : 
  let initial_total_age := 10 * initial_average
  let new_total_age := initial_total_age - 48 + 18
  let new_average := new_total_age / 10
  initial_average - new_average = 3 := by
sorry

end NUMINAMATH_CALUDE_average_age_decrease_l1906_190693


namespace NUMINAMATH_CALUDE_division_not_imply_multiple_and_factor_l1906_190626

theorem division_not_imply_multiple_and_factor :
  ¬ (∀ a b : ℝ, a / b = 5 → (∃ k : ℤ, a = b * k) ∧ (∃ k : ℤ, b * k = a)) := by
  sorry

end NUMINAMATH_CALUDE_division_not_imply_multiple_and_factor_l1906_190626


namespace NUMINAMATH_CALUDE_set_union_problem_l1906_190664

theorem set_union_problem (a b : ℝ) : 
  let M : Set ℝ := {3, 2*a}
  let N : Set ℝ := {a, b}
  M ∩ N = {2} → M ∪ N = {1, 2, 3} := by
sorry

end NUMINAMATH_CALUDE_set_union_problem_l1906_190664


namespace NUMINAMATH_CALUDE_episodes_per_season_l1906_190685

theorem episodes_per_season
  (series1_seasons series2_seasons : ℕ)
  (episodes_lost_per_season : ℕ)
  (remaining_episodes : ℕ)
  (h1 : series1_seasons = 12)
  (h2 : series2_seasons = 14)
  (h3 : episodes_lost_per_season = 2)
  (h4 : remaining_episodes = 364) :
  (remaining_episodes + episodes_lost_per_season * (series1_seasons + series2_seasons)) / (series1_seasons + series2_seasons) = 16 := by
sorry

end NUMINAMATH_CALUDE_episodes_per_season_l1906_190685


namespace NUMINAMATH_CALUDE_total_pencils_after_adding_l1906_190698

def initial_pencils : ℕ := 115
def added_pencils : ℕ := 100

theorem total_pencils_after_adding :
  initial_pencils + added_pencils = 215 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_after_adding_l1906_190698


namespace NUMINAMATH_CALUDE_carnival_activity_order_l1906_190644

/- Define the activities -/
inductive Activity
| Dodgeball
| MagicShow
| SingingContest

/- Define the popularity of each activity -/
def popularity : Activity → Rat
| Activity.Dodgeball => 9/24
| Activity.MagicShow => 4/12
| Activity.SingingContest => 1/3

/- Define the ordering of activities based on popularity -/
def more_popular (a b : Activity) : Prop :=
  popularity a > popularity b

/- Theorem statement -/
theorem carnival_activity_order :
  (more_popular Activity.Dodgeball Activity.MagicShow) ∧
  (more_popular Activity.MagicShow Activity.SingingContest) ∨
  (popularity Activity.MagicShow = popularity Activity.SingingContest) :=
sorry

end NUMINAMATH_CALUDE_carnival_activity_order_l1906_190644


namespace NUMINAMATH_CALUDE_min_turtle_distance_l1906_190655

/-- Represents an observer watching the turtle --/
structure Observer where
  startTime : ℕ
  endTime : ℕ
  distanceObserved : ℕ

/-- Represents the turtle's movement --/
def TurtleMovement (observers : List Observer) : Prop :=
  -- The observation lasts for 6 minutes
  (∀ o ∈ observers, o.startTime ≥ 0 ∧ o.endTime ≤ 6 * 60) ∧
  -- Each observer watches for 1 minute continuously
  (∀ o ∈ observers, o.endTime - o.startTime = 60) ∧
  -- Each observer notes 1 meter of movement
  (∀ o ∈ observers, o.distanceObserved = 1) ∧
  -- The turtle is always being observed
  (∀ t : ℕ, t ≥ 0 ∧ t ≤ 6 * 60 → ∃ o ∈ observers, o.startTime ≤ t ∧ t < o.endTime)

/-- The theorem stating the minimum distance the turtle could have traveled --/
theorem min_turtle_distance (observers : List Observer) 
  (h : TurtleMovement observers) : 
  ∃ d : ℕ, d = 4 ∧ (∀ d' : ℕ, (∃ obs : List Observer, TurtleMovement obs ∧ d' = obs.length) → d ≤ d') :=
sorry

end NUMINAMATH_CALUDE_min_turtle_distance_l1906_190655


namespace NUMINAMATH_CALUDE_root_comparison_l1906_190652

noncomputable def f (x : ℝ) : ℝ := Real.log x - 6 + 2 * x

theorem root_comparison (x₀ : ℝ) (hx₀ : f x₀ = 0) :
  Real.log (Real.log x₀) < Real.log (Real.sqrt x₀) ∧
  Real.log (Real.sqrt x₀) < Real.log x₀ ∧
  Real.log x₀ < (Real.log x₀)^2 := by
  sorry

end NUMINAMATH_CALUDE_root_comparison_l1906_190652


namespace NUMINAMATH_CALUDE_absolute_difference_26th_terms_l1906_190607

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a₁ + d * (n - 1)

theorem absolute_difference_26th_terms : 
  let C := arithmetic_sequence 50 15
  let D := arithmetic_sequence 85 (-20)
  |C 26 - D 26| = 840 := by
sorry

end NUMINAMATH_CALUDE_absolute_difference_26th_terms_l1906_190607


namespace NUMINAMATH_CALUDE_three_lines_intersection_l1906_190616

/-- Three lines intersect at the same point if and only if m = -9 -/
theorem three_lines_intersection (m : ℝ) : 
  (∃ (x y : ℝ), y = 2*x ∧ x + y = 3 ∧ m*x + 2*y + 5 = 0) ↔ m = -9 := by
  sorry

end NUMINAMATH_CALUDE_three_lines_intersection_l1906_190616


namespace NUMINAMATH_CALUDE_max_distance_complex_l1906_190633

theorem max_distance_complex (z : ℂ) (h : Complex.abs z = 3) :
  (⨆ (z : ℂ), Complex.abs ((1 + 2*Complex.I)*z^4 - z^6)) = 81 * (9 + Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_max_distance_complex_l1906_190633


namespace NUMINAMATH_CALUDE_trajectory_of_P_l1906_190668

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 = 8

-- Define the relationship between N, M, and P
def RelationNMP (xn yn xm ym xp yp : ℝ) : Prop :=
  C xn yn ∧ xm = 0 ∧ ym = yn ∧ xp = xn / 2 ∧ yp = yn

-- Theorem statement
theorem trajectory_of_P : 
  ∀ (x y : ℝ), (∃ (xn yn xm ym : ℝ), RelationNMP xn yn xm ym x y) → 
  x^2 / 2 + y^2 / 8 = 1 := by
sorry

end NUMINAMATH_CALUDE_trajectory_of_P_l1906_190668


namespace NUMINAMATH_CALUDE_prob_all_same_color_is_34_455_l1906_190683

def red_marbles : ℕ := 4
def white_marbles : ℕ := 5
def blue_marbles : ℕ := 6
def total_marbles : ℕ := red_marbles + white_marbles + blue_marbles

def prob_all_same_color : ℚ :=
  (red_marbles * (red_marbles - 1) * (red_marbles - 2) +
   white_marbles * (white_marbles - 1) * (white_marbles - 2) +
   blue_marbles * (blue_marbles - 1) * (blue_marbles - 2)) /
  (total_marbles * (total_marbles - 1) * (total_marbles - 2))

theorem prob_all_same_color_is_34_455 : prob_all_same_color = 34 / 455 := by
  sorry

end NUMINAMATH_CALUDE_prob_all_same_color_is_34_455_l1906_190683


namespace NUMINAMATH_CALUDE_determinant_equals_r_plus_s_minus_t_l1906_190611

def quartic_polynomial (r s t : ℝ) (x : ℝ) : ℝ := x^4 + r*x^2 + s*x + t

def det_matrix (a b c d : ℝ) : Matrix (Fin 4) (Fin 4) ℝ :=
  ![![1+a, 1, 1, 1],
    ![1, 1+b, 1, 1],
    ![1, 1, 1+c, 1],
    ![1, 1, 1, 1+d]]

theorem determinant_equals_r_plus_s_minus_t (r s t : ℝ) (a b c d : ℝ) :
  quartic_polynomial r s t a = 0 →
  quartic_polynomial r s t b = 0 →
  quartic_polynomial r s t c = 0 →
  quartic_polynomial r s t d = 0 →
  Matrix.det (det_matrix a b c d) = r + s - t :=
by sorry

end NUMINAMATH_CALUDE_determinant_equals_r_plus_s_minus_t_l1906_190611


namespace NUMINAMATH_CALUDE_math_city_intersections_l1906_190688

/-- Represents a city with a given number of streets -/
structure City where
  num_streets : ℕ
  no_parallel : Bool
  no_triple_intersections : Bool

/-- Calculates the number of intersections in a city -/
def num_intersections (c : City) : ℕ :=
  if c.num_streets ≤ 1 then 0
  else (c.num_streets - 1) * (c.num_streets - 2) / 2

/-- Theorem stating that a city with 12 streets, no parallel streets, 
    and no triple intersections has 66 intersections -/
theorem math_city_intersections :
  ∀ (c : City), c.num_streets = 12 → c.no_parallel = true → 
  c.no_triple_intersections = true → num_intersections c = 66 :=
by
  sorry


end NUMINAMATH_CALUDE_math_city_intersections_l1906_190688


namespace NUMINAMATH_CALUDE_remainder_sum_l1906_190605

theorem remainder_sum (n : ℤ) (h : n % 18 = 11) :
  (n % 2) + (n % 3) + (n % 6) + (n % 9) = 10 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l1906_190605


namespace NUMINAMATH_CALUDE_cube_root_of_product_l1906_190642

theorem cube_root_of_product (x y z : ℕ) : 
  (5^9 * 7^6 * 13^3 : ℝ)^(1/3) = 79625 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_product_l1906_190642


namespace NUMINAMATH_CALUDE_teacher_student_probability_teacher_student_probability_correct_l1906_190618

/-- The probability that neither teacher stands at either end when 2 teachers and 2 students
    stand in a row for a group photo. -/
theorem teacher_student_probability : ℚ :=
  let num_teachers : ℕ := 2
  let num_students : ℕ := 2
  let total_arrangements : ℕ := Nat.factorial 4
  let favorable_arrangements : ℕ := Nat.factorial 2 * Nat.factorial 2
  1 / 6

/-- Proof that the probability is correct. -/
theorem teacher_student_probability_correct : teacher_student_probability = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_teacher_student_probability_teacher_student_probability_correct_l1906_190618


namespace NUMINAMATH_CALUDE_milton_books_total_l1906_190697

/-- The number of zoology books Milton has -/
def zoology_books : ℕ := 16

/-- The number of botany books Milton has -/
def botany_books : ℕ := 4 * zoology_books

/-- The total number of books Milton has -/
def total_books : ℕ := zoology_books + botany_books

theorem milton_books_total : total_books = 80 := by
  sorry

end NUMINAMATH_CALUDE_milton_books_total_l1906_190697


namespace NUMINAMATH_CALUDE_digit_D_is_nine_l1906_190621

/-- Represents a two-digit number --/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  tens_is_digit : tens < 10
  ones_is_digit : ones < 10

def value (n : TwoDigitNumber) : Nat :=
  10 * n.tens + n.ones

theorem digit_D_is_nine
  (A B C D : Nat)
  (A_is_digit : A < 10)
  (B_is_digit : B < 10)
  (C_is_digit : C < 10)
  (D_is_digit : D < 10)
  (addition : value ⟨A, B, A_is_digit, B_is_digit⟩ + value ⟨C, B, C_is_digit, B_is_digit⟩ = value ⟨D, A, D_is_digit, A_is_digit⟩)
  (subtraction : value ⟨A, B, A_is_digit, B_is_digit⟩ - value ⟨C, B, C_is_digit, B_is_digit⟩ = A) :
  D = 9 := by
  sorry

end NUMINAMATH_CALUDE_digit_D_is_nine_l1906_190621


namespace NUMINAMATH_CALUDE_prism_volume_l1906_190634

/-- A right rectangular prism with given face areas has a volume of 30 cubic inches -/
theorem prism_volume (l w h : ℝ) 
  (face1 : l * w = 10)
  (face2 : w * h = 15)
  (face3 : l * h = 6) :
  l * w * h = 30 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l1906_190634


namespace NUMINAMATH_CALUDE_local_max_implies_a_less_than_neg_one_l1906_190622

/-- Given a real number a and a function y = e^x + ax with a local maximum point greater than zero, prove that a < -1 -/
theorem local_max_implies_a_less_than_neg_one (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ IsLocalMax (fun x => Real.exp x + a * x) x) → a < -1 :=
by sorry

end NUMINAMATH_CALUDE_local_max_implies_a_less_than_neg_one_l1906_190622


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_12321_l1906_190669

theorem largest_prime_factor_of_12321 : 
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ 12321 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 12321 → q ≤ p :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_12321_l1906_190669


namespace NUMINAMATH_CALUDE_one_isosceles_triangle_l1906_190606

-- Define a point in 2D space
structure Point :=
  (x : ℤ)
  (y : ℤ)

-- Define a triangle by its three vertices
structure Triangle :=
  (v1 : Point)
  (v2 : Point)
  (v3 : Point)

-- Function to calculate the squared distance between two points
def squaredDistance (p1 p2 : Point) : ℤ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

-- Function to check if a triangle is isosceles
def isIsosceles (t : Triangle) : Prop :=
  let d1 := squaredDistance t.v1 t.v2
  let d2 := squaredDistance t.v2 t.v3
  let d3 := squaredDistance t.v3 t.v1
  d1 = d2 ∨ d2 = d3 ∨ d3 = d1

-- Define the four triangles
def triangle1 : Triangle := ⟨⟨0, 7⟩, ⟨3, 7⟩, ⟨1, 5⟩⟩
def triangle2 : Triangle := ⟨⟨4, 5⟩, ⟨4, 7⟩, ⟨6, 5⟩⟩
def triangle3 : Triangle := ⟨⟨0, 2⟩, ⟨3, 3⟩, ⟨7, 2⟩⟩
def triangle4 : Triangle := ⟨⟨11, 5⟩, ⟨10, 7⟩, ⟨12, 5⟩⟩

-- Theorem: Exactly one of the four triangles is isosceles
theorem one_isosceles_triangle :
  (isIsosceles triangle1 ∨ isIsosceles triangle2 ∨ isIsosceles triangle3 ∨ isIsosceles triangle4) ∧
  ¬(isIsosceles triangle1 ∧ isIsosceles triangle2) ∧
  ¬(isIsosceles triangle1 ∧ isIsosceles triangle3) ∧
  ¬(isIsosceles triangle1 ∧ isIsosceles triangle4) ∧
  ¬(isIsosceles triangle2 ∧ isIsosceles triangle3) ∧
  ¬(isIsosceles triangle2 ∧ isIsosceles triangle4) ∧
  ¬(isIsosceles triangle3 ∧ isIsosceles triangle4) :=
sorry

end NUMINAMATH_CALUDE_one_isosceles_triangle_l1906_190606


namespace NUMINAMATH_CALUDE_problem1_solution_problem2_solution_l1906_190639

-- Problem 1
def problem1 (a b : ℕ) : Prop :=
  a ≠ b ∧
  ∃ p k : ℕ, Prime p ∧ b^2 + a = p^k ∧
  (b^2 + a) ∣ (a^2 + b)

theorem problem1_solution :
  ∀ a b : ℕ, problem1 a b ↔ (a = 5 ∧ b = 2) :=
sorry

-- Problem 2
def problem2 (a b : ℕ) : Prop :=
  a > 1 ∧ b > 1 ∧ a ≠ b ∧
  (b^2 + a - 1) ∣ (a^2 + b - 1)

theorem problem2_solution :
  ∀ a b : ℕ, problem2 a b →
  ∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ p ∣ (b^2 + a - 1) ∧ q ∣ (b^2 + a - 1) :=
sorry

end NUMINAMATH_CALUDE_problem1_solution_problem2_solution_l1906_190639


namespace NUMINAMATH_CALUDE_andrews_age_l1906_190671

theorem andrews_age :
  ∀ (a g : ℕ), 
    g = 15 * a →  -- Grandfather's age is fifteen times Andrew's age
    g - a = 70 →  -- Grandfather was 70 years old when Andrew was born
    a = 5         -- Andrew's age is 5
  := by sorry

end NUMINAMATH_CALUDE_andrews_age_l1906_190671


namespace NUMINAMATH_CALUDE_unique_solution_iff_m_eq_49_div_12_l1906_190625

/-- For a quadratic equation ax^2 + bx + c = 0 to have exactly one solution,
    its discriminant (b^2 - 4ac) must be zero. -/
def has_one_solution (a b c : ℝ) : Prop :=
  b^2 - 4*a*c = 0

/-- The quadratic equation 3x^2 - 7x + m = 0 -/
def quadratic_equation (x m : ℝ) : Prop :=
  3*x^2 - 7*x + m = 0

/-- Theorem: The quadratic equation 3x^2 - 7x + m = 0 has exactly one solution
    if and only if m = 49/12 -/
theorem unique_solution_iff_m_eq_49_div_12 :
  (∃! x, quadratic_equation x m) ↔ m = 49/12 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_iff_m_eq_49_div_12_l1906_190625


namespace NUMINAMATH_CALUDE_negation_equivalence_l1906_190613

theorem negation_equivalence :
  (¬ ∀ x : ℝ, |x| + x^2 ≥ 0) ↔ (∃ x₀ : ℝ, |x₀| + x₀^2 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1906_190613


namespace NUMINAMATH_CALUDE_tee_price_calculation_l1906_190666

/-- The price of a single tee shirt in Linda's store -/
def tee_price : ℝ := 8

/-- The price of a single pair of jeans in Linda's store -/
def jeans_price : ℝ := 11

/-- The number of tee shirts sold in a day -/
def tees_sold : ℕ := 7

/-- The number of jeans sold in a day -/
def jeans_sold : ℕ := 4

/-- The total revenue for the day -/
def total_revenue : ℝ := 100

theorem tee_price_calculation :
  tee_price * tees_sold + jeans_price * jeans_sold = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_tee_price_calculation_l1906_190666


namespace NUMINAMATH_CALUDE_dog_adult_weights_l1906_190675

-- Define the dog breeds
inductive DogBreed
| GoldenRetriever
| Labrador
| Poodle

-- Define the weight progression function
def weightProgression (breed : DogBreed) : ℕ → ℕ
| 0 => match breed with
  | DogBreed.GoldenRetriever => 6
  | DogBreed.Labrador => 8
  | DogBreed.Poodle => 4
| 1 => match breed with
  | DogBreed.GoldenRetriever => 12
  | DogBreed.Labrador => 24
  | DogBreed.Poodle => 16
| 2 => match breed with
  | DogBreed.GoldenRetriever => 24
  | DogBreed.Labrador => 36
  | DogBreed.Poodle => 32
| 3 => match breed with
  | DogBreed.GoldenRetriever => 48
  | DogBreed.Labrador => 72
  | DogBreed.Poodle => 32
| _ => 0

-- Define the final weight increase function
def finalWeightIncrease (breed : DogBreed) : ℕ :=
  match breed with
  | DogBreed.GoldenRetriever => 30
  | DogBreed.Labrador => 30
  | DogBreed.Poodle => 20

-- Define the adult weight function
def adultWeight (breed : DogBreed) : ℕ :=
  weightProgression breed 3 + finalWeightIncrease breed

-- Theorem statement
theorem dog_adult_weights :
  (adultWeight DogBreed.GoldenRetriever = 78) ∧
  (adultWeight DogBreed.Labrador = 102) ∧
  (adultWeight DogBreed.Poodle = 52) := by
  sorry

end NUMINAMATH_CALUDE_dog_adult_weights_l1906_190675


namespace NUMINAMATH_CALUDE_inverse_of_two_mod_185_l1906_190653

theorem inverse_of_two_mod_185 : Int.ModEq 1 185 (2 * 93) := by sorry

end NUMINAMATH_CALUDE_inverse_of_two_mod_185_l1906_190653


namespace NUMINAMATH_CALUDE_counting_sequence_53rd_term_l1906_190670

theorem counting_sequence_53rd_term : 
  let seq : ℕ → ℕ := λ n => n
  seq 53 = 10 := by
  sorry

end NUMINAMATH_CALUDE_counting_sequence_53rd_term_l1906_190670


namespace NUMINAMATH_CALUDE_apple_cost_per_kg_l1906_190661

theorem apple_cost_per_kg (p q : ℚ) : 
  (30 * p + 3 * q = 168) →
  (30 * p + 6 * q = 186) →
  (20 * p = 100) →
  p = 5 := by sorry

end NUMINAMATH_CALUDE_apple_cost_per_kg_l1906_190661


namespace NUMINAMATH_CALUDE_expression_evaluation_l1906_190659

theorem expression_evaluation (a b : ℚ) (h1 : a = -1) (h2 : b = 1/2) :
  5*a*b - 2*(3*a*b - (4*a*b^2 + 1/2*a*b)) - 5*a*b^2 = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1906_190659


namespace NUMINAMATH_CALUDE_factorial_quotient_equals_56_l1906_190699

theorem factorial_quotient_equals_56 :
  ∃! (n : ℕ), n > 0 ∧ n.factorial / (n - 2).factorial = 56 := by
  sorry

end NUMINAMATH_CALUDE_factorial_quotient_equals_56_l1906_190699


namespace NUMINAMATH_CALUDE_pencils_multiple_of_fifty_l1906_190617

/-- Given a number of students, pens, and pencils, we define a valid distribution --/
def ValidDistribution (S P : ℕ) : Prop :=
  S > 0 ∧ S ≤ 50 ∧ 100 % S = 0 ∧ P % S = 0

/-- Theorem stating that the number of pencils must be a multiple of 50 --/
theorem pencils_multiple_of_fifty (P : ℕ) :
  (∃ S : ℕ, ValidDistribution S P) → P % 50 = 0 := by
  sorry

end NUMINAMATH_CALUDE_pencils_multiple_of_fifty_l1906_190617


namespace NUMINAMATH_CALUDE_problem_statement_l1906_190680

-- Define propositions p and q as functions of m
def p (m : ℝ) : Prop := m > 2
def q (m : ℝ) : Prop := m > 1

-- State the theorem
theorem problem_statement (m : ℝ) :
  ((p m ∨ q m) ∧ ¬(p m ∧ q m)) →
  (p m ∧ ¬(q m)) ∧ (1 < m ∧ m ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l1906_190680


namespace NUMINAMATH_CALUDE_largest_w_value_l1906_190657

theorem largest_w_value (w x y z : ℝ) 
  (sum_eq : w + x + y + z = 25)
  (prod_sum_eq : w*x + w*y + w*z + x*y + x*z + y*z = 2*y + 2*z + 193) :
  w ≤ 25/2 ∧ ∃ (w' : ℝ), w' = 25/2 ∧ 
    w' + x' + y' + z' = 25 ∧ 
    w'*x' + w'*y' + w'*z' + x'*y' + x'*z' + y'*z' = 2*y' + 2*z' + 193 :=
sorry

end NUMINAMATH_CALUDE_largest_w_value_l1906_190657


namespace NUMINAMATH_CALUDE_binomial_coefficient_20_10_l1906_190651

theorem binomial_coefficient_20_10 
  (h1 : Nat.choose 18 8 = 43758)
  (h2 : Nat.choose 18 9 = 48620)
  (h3 : Nat.choose 18 10 = 43758) :
  Nat.choose 20 10 = 184756 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_20_10_l1906_190651


namespace NUMINAMATH_CALUDE_x_negative_necessary_not_sufficient_for_quadratic_inequality_l1906_190672

theorem x_negative_necessary_not_sufficient_for_quadratic_inequality :
  (∀ x : ℝ, x^2 + x < 0 → x < 0) ∧
  (∃ x : ℝ, x < 0 ∧ x^2 + x ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_x_negative_necessary_not_sufficient_for_quadratic_inequality_l1906_190672


namespace NUMINAMATH_CALUDE_cost_of_type_B_books_l1906_190656

/-- The cost of purchasing type B books given the total number of books and the number of type A books purchased -/
theorem cost_of_type_B_books (total_books : ℕ) (x : ℕ) (price_B : ℕ) 
  (h_total : total_books = 100)
  (h_price : price_B = 6)
  (h_x_le_total : x ≤ total_books) :
  price_B * (total_books - x) = 6 * (100 - x) :=
by sorry

end NUMINAMATH_CALUDE_cost_of_type_B_books_l1906_190656
