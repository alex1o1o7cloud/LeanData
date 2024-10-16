import Mathlib

namespace NUMINAMATH_CALUDE_power_inequality_l1222_122264

theorem power_inequality (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  (0.2 : ℝ) ^ x < (1/2 : ℝ) ^ x ∧ (1/2 : ℝ) ^ x < 2 ^ x := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l1222_122264


namespace NUMINAMATH_CALUDE_two_visits_count_l1222_122220

/-- Represents the visiting schedule of friends -/
structure VisitSchedule where
  alice : Nat
  beatrix : Nat
  claire : Nat

/-- Calculates the number of days when exactly two friends visit -/
def exactlyTwoVisits (schedule : VisitSchedule) (totalDays : Nat) : Nat :=
  sorry

/-- The main theorem to prove -/
theorem two_visits_count (schedule : VisitSchedule) (totalDays : Nat) :
  schedule.alice = 5 →
  schedule.beatrix = 6 →
  schedule.claire = 8 →
  totalDays = 400 →
  exactlyTwoVisits schedule totalDays = 39 :=
sorry

end NUMINAMATH_CALUDE_two_visits_count_l1222_122220


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1222_122227

theorem arithmetic_calculation : 5 * 7 + 9 * 4 - 30 / 2 + 3^2 = 65 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1222_122227


namespace NUMINAMATH_CALUDE_betty_age_l1222_122272

/-- Given the relationships between Albert, Mary, and Betty's ages, prove Betty's age -/
theorem betty_age (albert mary betty : ℕ) 
  (h1 : albert = 2 * mary)
  (h2 : albert = 4 * betty)
  (h3 : mary = albert - 14) :
  betty = 7 := by
  sorry

end NUMINAMATH_CALUDE_betty_age_l1222_122272


namespace NUMINAMATH_CALUDE_binary_110011_is_51_l1222_122295

def binary_to_decimal (b : List Bool) : ℕ :=
  (List.enumFrom 0 b).foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_110011_is_51 :
  binary_to_decimal [true, true, false, false, true, true] = 51 := by
  sorry

end NUMINAMATH_CALUDE_binary_110011_is_51_l1222_122295


namespace NUMINAMATH_CALUDE_dessert_probability_l1222_122241

theorem dessert_probability (p_dessert : ℝ) (p_dessert_no_coffee : ℝ) :
  p_dessert = 0.6 →
  p_dessert_no_coffee = 0.2 * p_dessert →
  1 - p_dessert = 0.4 := by
sorry

end NUMINAMATH_CALUDE_dessert_probability_l1222_122241


namespace NUMINAMATH_CALUDE_initial_red_marbles_l1222_122244

theorem initial_red_marbles (r g : ℕ) : 
  r * 3 = g * 5 → 
  (r - 18) * 4 = g + 27 → 
  r = 29 := by
sorry

end NUMINAMATH_CALUDE_initial_red_marbles_l1222_122244


namespace NUMINAMATH_CALUDE_line_plane_relationship_l1222_122212

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (intersect : Line → Line → Prop)
variable (parallel : Line → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (line_parallel_or_intersect_plane : Line → Plane → Prop)

-- State the theorem
theorem line_plane_relationship 
  (a b : Line) (β : Plane) 
  (h1 : intersect a b) 
  (h2 : parallel a β) : 
  line_parallel_or_intersect_plane b β :=
sorry

end NUMINAMATH_CALUDE_line_plane_relationship_l1222_122212


namespace NUMINAMATH_CALUDE_abs_five_minus_e_l1222_122286

theorem abs_five_minus_e (e : ℝ) (h : e < 5) : |5 - e| = 5 - e := by sorry

end NUMINAMATH_CALUDE_abs_five_minus_e_l1222_122286


namespace NUMINAMATH_CALUDE_product_equals_zero_l1222_122223

theorem product_equals_zero (a : ℤ) (h : a = 11) :
  (a - 12) * (a - 11) * (a - 10) * (a - 9) * (a - 8) * (a - 7) * (a - 6) * 
  (a - 5) * (a - 4) * (a - 3) * (a - 2) * (a - 1) * a * (a + 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_zero_l1222_122223


namespace NUMINAMATH_CALUDE_ratio_of_numbers_l1222_122276

theorem ratio_of_numbers (A B C : ℝ) (k : ℝ) 
  (h1 : A = k * B)
  (h2 : A = 3 * C)
  (h3 : (A + B + C) / 3 = 88)
  (h4 : A - C = 96) :
  A / B = 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_numbers_l1222_122276


namespace NUMINAMATH_CALUDE_wedding_guests_count_l1222_122231

/-- The number of guests attending the wedding -/
def total_guests : ℕ := 240

/-- The proportion of female guests -/
def female_proportion : ℚ := 3/5

/-- The proportion of female guests from Jay's family -/
def jay_family_proportion : ℚ := 1/2

/-- The number of female guests from Jay's family -/
def jay_family_females : ℕ := 72

theorem wedding_guests_count :
  (jay_family_females : ℚ) = (total_guests : ℚ) * female_proportion * jay_family_proportion :=
by sorry

end NUMINAMATH_CALUDE_wedding_guests_count_l1222_122231


namespace NUMINAMATH_CALUDE_correct_ranking_l1222_122219

-- Define the colleagues
inductive Colleague
| David
| Emily
| Frank

-- Define the years of service comparison
def has_more_years (a b : Colleague) : Prop := sorry

-- Define the statements
def statement_I : Prop := has_more_years Colleague.Emily Colleague.David ∧ has_more_years Colleague.Emily Colleague.Frank
def statement_II : Prop := ¬(has_more_years Colleague.David Colleague.Emily) ∨ ¬(has_more_years Colleague.David Colleague.Frank)
def statement_III : Prop := has_more_years Colleague.Frank Colleague.David ∨ has_more_years Colleague.Frank Colleague.Emily

-- Theorem to prove
theorem correct_ranking :
  (statement_I ∨ statement_II ∨ statement_III) ∧
  ¬(statement_I ∧ statement_II) ∧
  ¬(statement_I ∧ statement_III) ∧
  ¬(statement_II ∧ statement_III) →
  has_more_years Colleague.David Colleague.Frank ∧
  has_more_years Colleague.Frank Colleague.Emily :=
by sorry

end NUMINAMATH_CALUDE_correct_ranking_l1222_122219


namespace NUMINAMATH_CALUDE_opposite_and_abs_of_sqrt3_minus2_l1222_122257

theorem opposite_and_abs_of_sqrt3_minus2 :
  (-(Real.sqrt 3 - 2) = 2 - Real.sqrt 3) ∧
  (|Real.sqrt 3 - 2| = 2 - Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_opposite_and_abs_of_sqrt3_minus2_l1222_122257


namespace NUMINAMATH_CALUDE_incircle_identity_l1222_122240

-- Define a triangle with an incircle
structure TriangleWithIncircle where
  -- The sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- The semi-perimeter
  p : ℝ
  -- The inradius
  r : ℝ
  -- The angle APB
  α : ℝ
  -- Conditions
  side_positive : 0 < a ∧ 0 < b ∧ 0 < c
  semi_perimeter : p = (a + b + c) / 2
  inradius_positive : 0 < r
  angle_positive : 0 < α ∧ α < π / 2

-- The theorem to prove
theorem incircle_identity (t : TriangleWithIncircle) :
  1 / (t.p - t.b) + 1 / (t.p - t.c) = 2 / (t.r * Real.tan t.α) := by
  sorry

end NUMINAMATH_CALUDE_incircle_identity_l1222_122240


namespace NUMINAMATH_CALUDE_largest_sulfuric_acid_percentage_l1222_122237

/-- Represents the largest integer percentage of sulfuric acid solution that can be achieved in the first vessel after transfer -/
def largest_integer_percentage : ℕ := 76

/-- Represents the initial volume of solution in the first vessel -/
def initial_volume_1 : ℚ := 4

/-- Represents the initial volume of solution in the second vessel -/
def initial_volume_2 : ℚ := 3

/-- Represents the initial concentration of sulfuric acid in the first vessel -/
def initial_concentration_1 : ℚ := 70 / 100

/-- Represents the initial concentration of sulfuric acid in the second vessel -/
def initial_concentration_2 : ℚ := 90 / 100

/-- Represents the capacity of each vessel -/
def vessel_capacity : ℚ := 6

theorem largest_sulfuric_acid_percentage :
  ∀ x : ℚ,
  0 ≤ x ∧ x ≤ initial_volume_2 →
  (initial_volume_1 * initial_concentration_1 + x * initial_concentration_2) / (initial_volume_1 + x) ≤ largest_integer_percentage / 100 ∧
  ∃ y : ℚ, 0 < y ∧ y ≤ initial_volume_2 ∧
  (initial_volume_1 * initial_concentration_1 + y * initial_concentration_2) / (initial_volume_1 + y) > (largest_integer_percentage - 1) / 100 ∧
  initial_volume_1 + y ≤ vessel_capacity :=
by sorry

#check largest_sulfuric_acid_percentage

end NUMINAMATH_CALUDE_largest_sulfuric_acid_percentage_l1222_122237


namespace NUMINAMATH_CALUDE_friends_weekly_biking_distance_l1222_122215

/-- The total distance two friends bike in a week -/
def total_distance_biked (onur_daily_distance : ℕ) (hanil_extra_distance : ℕ) (days_per_week : ℕ) : ℕ :=
  (onur_daily_distance * days_per_week) + ((onur_daily_distance + hanil_extra_distance) * days_per_week)

/-- Theorem stating the total distance biked by Onur and Hanil in a week -/
theorem friends_weekly_biking_distance :
  total_distance_biked 250 40 5 = 2700 := by
  sorry

#eval total_distance_biked 250 40 5

end NUMINAMATH_CALUDE_friends_weekly_biking_distance_l1222_122215


namespace NUMINAMATH_CALUDE_min_expression_le_one_l1222_122292

theorem min_expression_le_one (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_eq_three : x + y + z = 3) :
  min (x * (x + y - z)) (min (y * (y + z - x)) (z * (z + x - y))) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_expression_le_one_l1222_122292


namespace NUMINAMATH_CALUDE_additional_miles_is_33_l1222_122259

/-- Represents the distances between locations in Kona's trip -/
structure TripDistances where
  apartment_to_bakery : ℕ
  bakery_to_grandma : ℕ
  grandma_to_apartment : ℕ

/-- Calculates the additional miles driven with bakery stop compared to without -/
def additional_miles (d : TripDistances) : ℕ :=
  d.apartment_to_bakery + d.bakery_to_grandma + d.grandma_to_apartment - 2 * d.grandma_to_apartment

/-- Theorem stating that the additional miles driven with bakery stop is 33 -/
theorem additional_miles_is_33 (d : TripDistances) 
    (h1 : d.apartment_to_bakery = 9)
    (h2 : d.bakery_to_grandma = 24)
    (h3 : d.grandma_to_apartment = 27) : 
  additional_miles d = 33 := by
  sorry

end NUMINAMATH_CALUDE_additional_miles_is_33_l1222_122259


namespace NUMINAMATH_CALUDE_cupcake_packages_l1222_122247

theorem cupcake_packages (x y z : ℕ) (hx : x = 50) (hy : y = 5) (hz : z = 5) :
  (x - y) / z = 9 := by
  sorry

end NUMINAMATH_CALUDE_cupcake_packages_l1222_122247


namespace NUMINAMATH_CALUDE_exists_integer_solution_l1222_122233

theorem exists_integer_solution : ∃ x : ℤ, 2 * x^2 - 3 * x + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_integer_solution_l1222_122233


namespace NUMINAMATH_CALUDE_total_books_on_shelves_l1222_122230

theorem total_books_on_shelves (num_shelves : ℕ) (books_per_shelf : ℕ) :
  num_shelves = 625 → books_per_shelf = 28 → num_shelves * books_per_shelf = 22500 := by
  sorry

end NUMINAMATH_CALUDE_total_books_on_shelves_l1222_122230


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_intersection_l1222_122218

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 144 - y^2 / 81 = 1

-- Define the line
def line (x y : ℝ) : Prop := y = 2*x + 3

-- Define the asymptotes
def asymptote1 (x y : ℝ) : Prop := y = (3/4) * x
def asymptote2 (x y : ℝ) : Prop := y = -(3/4) * x

-- Theorem statement
theorem hyperbola_asymptote_intersection :
  ∃ (x1 y1 x2 y2 : ℝ),
    asymptote1 x1 y1 ∧ line x1 y1 ∧ 
    asymptote2 x2 y2 ∧ line x2 y2 ∧
    x1 = -12/5 ∧ y1 = -9/5 ∧
    x2 = -12/11 ∧ y2 = 9/11 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_intersection_l1222_122218


namespace NUMINAMATH_CALUDE_tiling_condition_l1222_122288

/-- A tile is represented by its dimensions -/
structure Tile :=
  (length : ℕ)
  (width : ℕ)

/-- A grid is represented by its side length -/
structure Grid :=
  (side : ℕ)

/-- Predicate to check if a grid can be tiled by a given tile -/
def can_be_tiled (g : Grid) (t : Tile) : Prop :=
  ∃ (k : ℕ), g.side = k * t.length ∧ g.side * g.side = k * k * (t.length * t.width)

/-- The main theorem stating the condition for tiling an n×n grid with 4×1 tiles -/
theorem tiling_condition (n : ℕ) :
  (∃ (g : Grid) (t : Tile), g.side = n ∧ t.length = 4 ∧ t.width = 1 ∧ can_be_tiled g t) ↔ 
  (∃ (k : ℕ), n = 4 * k) :=
sorry

end NUMINAMATH_CALUDE_tiling_condition_l1222_122288


namespace NUMINAMATH_CALUDE_two_digit_minus_reverse_63_l1222_122210

/-- Reverses a two-digit number -/
def reverse_two_digit (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

/-- Checks if a number is two-digit -/
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

theorem two_digit_minus_reverse_63 (n : ℕ) :
  is_two_digit n ∧ n - reverse_two_digit n = 63 → n = 81 ∨ n = 92 :=
by sorry

end NUMINAMATH_CALUDE_two_digit_minus_reverse_63_l1222_122210


namespace NUMINAMATH_CALUDE_max_groups_is_100_l1222_122243

/-- Represents the number of cards for each value -/
def CardCount : ℕ := 200

/-- Represents the target sum for each group -/
def TargetSum : ℕ := 9

/-- Represents the maximum number of groups that can be formed -/
def MaxGroups : ℕ := 100

/-- Proves that the maximum number of groups that can be formed is 100 -/
theorem max_groups_is_100 :
  ∀ (groups : ℕ) (cards_5 cards_2 cards_1 : ℕ),
    cards_5 = CardCount →
    cards_2 = CardCount →
    cards_1 = CardCount →
    (∀ g : ℕ, g ≤ groups → ∃ (a b c : ℕ),
      a + b + c = TargetSum ∧
      a * 5 + b * 2 + c * 1 = TargetSum ∧
      a ≤ cards_5 ∧ b ≤ cards_2 ∧ c ≤ cards_1) →
    groups ≤ MaxGroups :=
  sorry

end NUMINAMATH_CALUDE_max_groups_is_100_l1222_122243


namespace NUMINAMATH_CALUDE_stratified_sampling_example_l1222_122269

/-- The number of ways to select students using stratified sampling -/
def stratified_sampling_selection (female_count male_count total_selected : ℕ) : ℕ :=
  let female_selected := (female_count * total_selected) / (female_count + male_count)
  let male_selected := total_selected - female_selected
  (Nat.choose female_count female_selected) * (Nat.choose male_count male_selected)

/-- Theorem: The number of ways to select 3 students from 8 female and 4 male students
    using stratified sampling by gender ratio is 112 -/
theorem stratified_sampling_example : stratified_sampling_selection 8 4 3 = 112 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_example_l1222_122269


namespace NUMINAMATH_CALUDE_lcm_hcf_problem_l1222_122278

theorem lcm_hcf_problem (a b : ℕ+) :
  Nat.lcm a b = 2310 →
  Nat.gcd a b = 47 →
  a = 210 →
  b = 517 := by
sorry

end NUMINAMATH_CALUDE_lcm_hcf_problem_l1222_122278


namespace NUMINAMATH_CALUDE_egg_problem_l1222_122200

theorem egg_problem (x y z : ℕ) : 
  x > 0 → y > 0 → z > 0 →
  x + y + z = 100 →
  5 * x + 6 * y + 9 * z = 600 →
  (x = y ∨ y = z ∨ x = z) →
  x = 60 ∧ y = 20 := by
  sorry

end NUMINAMATH_CALUDE_egg_problem_l1222_122200


namespace NUMINAMATH_CALUDE_sum_of_numbers_l1222_122254

theorem sum_of_numbers : 3 + 33 + 333 + 3.33 = 372.33 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l1222_122254


namespace NUMINAMATH_CALUDE_ellipse_min_area_l1222_122207

/-- An ellipse containing two specific circles has a minimum area -/
theorem ellipse_min_area (a b : ℝ) (h_ellipse : ∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 → 
  ((x - 2)^2 + y^2 ≥ 4 ∧ (x + 2)^2 + y^2 ≥ 4)) :
  π * a * b ≥ (3 * Real.sqrt 3 / 2) * π := by
  sorry

#check ellipse_min_area

end NUMINAMATH_CALUDE_ellipse_min_area_l1222_122207


namespace NUMINAMATH_CALUDE_pyramid_intersection_theorem_l1222_122290

structure Pyramid where
  base : Rectangle
  side_edge : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Given a pyramid with a rectangular base and equal side edges, when a plane intersects
    the side edges cutting off segments a, b, c, and d, the equation 1/a + 1/c = 1/b + 1/d holds. -/
theorem pyramid_intersection_theorem (p : Pyramid) (ha : p.a > 0) (hb : p.b > 0) (hc : p.c > 0) (hd : p.d > 0) :
  1 / p.a + 1 / p.c = 1 / p.b + 1 / p.d := by
  sorry

end NUMINAMATH_CALUDE_pyramid_intersection_theorem_l1222_122290


namespace NUMINAMATH_CALUDE_cubic_fraction_factorization_l1222_122268

theorem cubic_fraction_factorization (x y z : ℝ) :
  ((x^2 - y^2)^3 + (y^2 - z^2)^3 + (z^2 - x^2)^3) / ((x - y)^3 + (y - z)^3 + (z - x)^3) = (x + y) * (y + z) * (z + x) :=
by sorry

end NUMINAMATH_CALUDE_cubic_fraction_factorization_l1222_122268


namespace NUMINAMATH_CALUDE_r₂_bound_l1222_122282

/-- The function f(x) = x² - r₂x + r₃ -/
def f (r₂ r₃ : ℝ) (x : ℝ) : ℝ := x^2 - r₂ * x + r₃

/-- The sequence g_n defined recursively -/
def g (r₂ r₃ : ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => f r₂ r₃ (g r₂ r₃ n)

/-- The property that g₂ᵢ < g₂ᵢ₊₁ and g₂ᵢ₊₁ > g₂ᵢ₊₂ for 0 ≤ i ≤ 2011 -/
def alternating_property (r₂ r₃ : ℝ) : Prop :=
  ∀ i, 0 ≤ i ∧ i ≤ 2011 → g r₂ r₃ (2*i) < g r₂ r₃ (2*i + 1) ∧ g r₂ r₃ (2*i + 1) > g r₂ r₃ (2*i + 2)

/-- The property that there exists j such that gᵢ₊₁ > gᵢ for all i > j -/
def eventually_increasing (r₂ r₃ : ℝ) : Prop :=
  ∃ j : ℕ, ∀ i, i > j → g r₂ r₃ (i + 1) > g r₂ r₃ i

/-- The property that the sequence is unbounded -/
def unbounded (r₂ r₃ : ℝ) : Prop :=
  ∀ M : ℝ, ∃ N : ℕ, g r₂ r₃ N > M

theorem r₂_bound (r₂ r₃ : ℝ) 
  (h₁ : alternating_property r₂ r₃)
  (h₂ : eventually_increasing r₂ r₃)
  (h₃ : unbounded r₂ r₃) :
  |r₂| > 2 ∧ ∀ ε > 0, ∃ r₂' r₃', 
    alternating_property r₂' r₃' ∧ 
    eventually_increasing r₂' r₃' ∧ 
    unbounded r₂' r₃' ∧ 
    |r₂'| < 2 + ε :=
sorry

end NUMINAMATH_CALUDE_r₂_bound_l1222_122282


namespace NUMINAMATH_CALUDE_clock_angle_at_eight_thirty_l1222_122270

/-- The angle between clock hands at 8:30 -/
theorem clock_angle_at_eight_thirty :
  let hour_angle : ℝ := (8 * 30 + 30 / 2)
  let minute_angle : ℝ := 180
  let angle_diff : ℝ := |hour_angle - minute_angle|
  angle_diff = 75 := by
  sorry

end NUMINAMATH_CALUDE_clock_angle_at_eight_thirty_l1222_122270


namespace NUMINAMATH_CALUDE_horner_rule_v₄_l1222_122206

-- Define the polynomial coefficients
def a₀ : ℤ := 12
def a₁ : ℤ := 35
def a₂ : ℤ := -8
def a₃ : ℤ := 6
def a₄ : ℤ := 5
def a₅ : ℤ := 3

-- Define x
def x : ℤ := -2

-- Define Horner's Rule steps
def v₀ : ℤ := a₅
def v₁ : ℤ := v₀ * x + a₄
def v₂ : ℤ := v₁ * x + a₃
def v₃ : ℤ := v₂ * x + a₂
def v₄ : ℤ := v₃ * x + a₁

-- Theorem statement
theorem horner_rule_v₄ : v₄ = 83 := by
  sorry

end NUMINAMATH_CALUDE_horner_rule_v₄_l1222_122206


namespace NUMINAMATH_CALUDE_salary_calculation_l1222_122256

theorem salary_calculation (food_fraction : Rat) (rent_fraction : Rat) (clothes_fraction : Rat) 
  (savings_fraction : Rat) (tax_fraction : Rat) (remaining_amount : ℝ) :
  food_fraction = 1/5 →
  rent_fraction = 1/10 →
  clothes_fraction = 3/5 →
  savings_fraction = 1/20 →
  tax_fraction = 1/8 →
  remaining_amount = 18000 →
  ∃ S : ℝ, (7/160 : ℝ) * S = remaining_amount :=
by
  sorry

end NUMINAMATH_CALUDE_salary_calculation_l1222_122256


namespace NUMINAMATH_CALUDE_projectile_max_height_l1222_122224

/-- The height function of the projectile -/
def h (t : ℝ) : ℝ := -20 * t^2 + 50 * t + 10

/-- The maximum height reached by the projectile -/
def max_height : ℝ := 41.25

/-- Theorem stating that the maximum value of h(t) is equal to max_height -/
theorem projectile_max_height : 
  ∀ t : ℝ, h t ≤ max_height ∧ ∃ t₀ : ℝ, h t₀ = max_height :=
sorry

end NUMINAMATH_CALUDE_projectile_max_height_l1222_122224


namespace NUMINAMATH_CALUDE_curve_equation_and_no_fixed_point_l1222_122252

-- Define the circle C2
def C2 (x y : ℝ) : Prop := x^2 + (y-2)^2 = 1

-- Define the curve C1
def C1 (x y : ℝ) : Prop :=
  ∀ (x' y' : ℝ), C2 x' y' → (x - x')^2 + (y - y')^2 > 0 ∧
  (y + 1 = Real.sqrt ((x - x')^2 + (y - y')^2) - 1)

-- Define the point N
def N (b : ℝ) : ℝ × ℝ := (0, b)

-- Define the angle equality condition
def angle_equality (P Q : ℝ × ℝ) (N : ℝ × ℝ) : Prop :=
  (P.1 - N.1)^2 + (P.2 - N.2)^2 = (Q.1 - N.1)^2 + (Q.2 - N.2)^2

theorem curve_equation_and_no_fixed_point :
  (∀ x y : ℝ, C1 x y ↔ x^2 = 8*y) ∧
  (∀ b : ℝ, b < 0 →
    ∀ P Q : ℝ × ℝ, C1 P.1 P.2 → C1 Q.1 Q.2 → P ≠ Q →
    angle_equality P Q (N b) →
    ¬∃ F : ℝ × ℝ, ∀ P Q : ℝ × ℝ, C1 P.1 P.2 → C1 Q.1 Q.2 → P ≠ Q →
      angle_equality P Q (N b) → (Q.2 - P.2) * F.1 = (Q.1 - P.1) * F.2 + (P.1 * Q.2 - Q.1 * P.2)) :=
sorry

end NUMINAMATH_CALUDE_curve_equation_and_no_fixed_point_l1222_122252


namespace NUMINAMATH_CALUDE_p_distance_is_300_l1222_122299

/-- A race between two runners p and q -/
structure Race where
  /-- The speed of runner q in meters per second -/
  q_speed : ℝ
  /-- The length of the race course in meters -/
  race_length : ℝ

/-- The result of the race -/
def race_result (r : Race) : ℝ := 
  let p_speed := 1.2 * r.q_speed
  let p_distance := r.race_length + 50
  p_distance

/-- Theorem: Under the given conditions, p runs 300 meters -/
theorem p_distance_is_300 (r : Race) : 
  r.race_length > 0 ∧ 
  r.q_speed > 0 ∧ 
  r.race_length / r.q_speed = (r.race_length + 50) / (1.2 * r.q_speed) → 
  race_result r = 300 := by
  sorry

end NUMINAMATH_CALUDE_p_distance_is_300_l1222_122299


namespace NUMINAMATH_CALUDE_cone_base_circumference_l1222_122279

/-- For a right circular cone with volume 16π cubic centimeters and height 6 cm,
    the circumference of its base is 4√2π cm. -/
theorem cone_base_circumference :
  ∀ (r : ℝ), 
    (1 / 3 * π * r^2 * 6 = 16 * π) →
    (2 * π * r = 4 * Real.sqrt 2 * π) :=
by sorry

end NUMINAMATH_CALUDE_cone_base_circumference_l1222_122279


namespace NUMINAMATH_CALUDE_f_ordering_l1222_122214

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x

theorem f_ordering : f (-π/3) > f (-1) ∧ f (-1) > f (π/11) := by
  sorry

end NUMINAMATH_CALUDE_f_ordering_l1222_122214


namespace NUMINAMATH_CALUDE_triangle_problem_l1222_122226

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  (0 < a) ∧ (0 < b) ∧ (0 < c) →
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) →
  A + B + C = π →
  2 * Real.cos C * (a * Real.cos B + b * Real.cos A) = c →
  c = Real.sqrt 7 →
  1/2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 2 →
  C = π/3 ∧ a + b = 5 := by
sorry


end NUMINAMATH_CALUDE_triangle_problem_l1222_122226


namespace NUMINAMATH_CALUDE_investment_rate_calculation_l1222_122277

/-- Given an investment of 3000 units yielding an income of 210 units,
    prove that the investment rate is 7%. -/
theorem investment_rate_calculation (investment : ℝ) (income : ℝ) (rate : ℝ) :
  investment = 3000 →
  income = 210 →
  rate = income / investment * 100 →
  rate = 7 := by
  sorry

end NUMINAMATH_CALUDE_investment_rate_calculation_l1222_122277


namespace NUMINAMATH_CALUDE_lines_intersect_at_point_l1222_122291

/-- Represents a 2D point -/
structure Point where
  x : ℚ
  y : ℚ

/-- Represents a line in parametric form -/
structure ParametricLine where
  p : Point  -- Starting point
  v : Point  -- Direction vector

/-- The first line -/
def line1 : ParametricLine :=
  { p := { x := 1, y := 4 },
    v := { x := -2, y := 3 } }

/-- The second line -/
def line2 : ParametricLine :=
  { p := { x := 5, y := 2 },
    v := { x := 1, y := 6 } }

/-- A point on a parametric line -/
def pointOnLine (l : ParametricLine) (t : ℚ) : Point :=
  { x := l.p.x + t * l.v.x,
    y := l.p.y + t * l.v.y }

/-- The proposed intersection point -/
def intersectionPoint : Point :=
  { x := 21 / 5,
    y := -4 / 5 }

theorem lines_intersect_at_point :
  ∃ (t u : ℚ), pointOnLine line1 t = intersectionPoint ∧ pointOnLine line2 u = intersectionPoint :=
sorry

end NUMINAMATH_CALUDE_lines_intersect_at_point_l1222_122291


namespace NUMINAMATH_CALUDE_yuan_equality_l1222_122285

theorem yuan_equality : (3.00 : ℝ) = (3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_yuan_equality_l1222_122285


namespace NUMINAMATH_CALUDE_call_center_ratio_l1222_122289

theorem call_center_ratio (a b : ℕ) (h1 : a > 0) (h2 : b > 0) : 
  (6 / 5 : ℚ) * a * b = (3 / 4 : ℚ) * b * b → a / b = (5 / 8 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_call_center_ratio_l1222_122289


namespace NUMINAMATH_CALUDE_total_votes_cast_l1222_122284

theorem total_votes_cast (total_votes : ℕ) (votes_for : ℕ) (votes_against : ℕ) : 
  votes_for = votes_against + 70 →
  votes_against = (40 : ℕ) * total_votes / 100 →
  total_votes = votes_for + votes_against →
  total_votes = 350 := by
sorry

end NUMINAMATH_CALUDE_total_votes_cast_l1222_122284


namespace NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l1222_122267

theorem quadratic_inequality_equivalence :
  ∀ x : ℝ, x * (2 * x + 3) < 9 ↔ x ∈ Set.Ioo (-5/2) 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l1222_122267


namespace NUMINAMATH_CALUDE_negation_of_implication_l1222_122253

/-- Two lines in a 3D space -/
structure Line3D where
  -- Define necessary properties for a 3D line
  -- This is a simplified representation
  dummy : Unit

/-- Predicate to check if two lines have a common point -/
def have_common_point (l1 l2 : Line3D) : Prop :=
  sorry -- Definition omitted for simplicity

/-- Predicate to check if two lines are skew -/
def are_skew (l1 l2 : Line3D) : Prop :=
  sorry -- Definition omitted for simplicity

theorem negation_of_implication (l1 l2 : Line3D) :
  (¬(¬(have_common_point l1 l2) → are_skew l1 l2)) ↔
  (have_common_point l1 l2 → ¬(are_skew l1 l2)) :=
by
  sorry

#check negation_of_implication

end NUMINAMATH_CALUDE_negation_of_implication_l1222_122253


namespace NUMINAMATH_CALUDE_jellybean_problem_l1222_122229

theorem jellybean_problem (initial_bags : ℕ) (initial_average : ℕ) (average_increase : ℕ) :
  initial_bags = 34 →
  initial_average = 117 →
  average_increase = 7 →
  let total_initial := initial_bags * initial_average
  let new_average := initial_average + average_increase
  let total_new := (initial_bags + 1) * new_average
  total_new - total_initial = 362 :=
by sorry

end NUMINAMATH_CALUDE_jellybean_problem_l1222_122229


namespace NUMINAMATH_CALUDE_sum_odd_9_to_39_l1222_122251

/-- Sum of first n consecutive odd integers -/
def sum_first_n_odd (n : ℕ) : ℕ := n ^ 2

/-- The nth odd integer -/
def nth_odd (n : ℕ) : ℕ := 2 * n - 1

/-- Sum of odd integers from a to b inclusive -/
def sum_odd_range (a b : ℕ) : ℕ :=
  sum_first_n_odd ((b - 1) / 2 + 1) - sum_first_n_odd ((a - 1) / 2)

theorem sum_odd_9_to_39 :
  sum_odd_range 9 39 = 384 :=
sorry

end NUMINAMATH_CALUDE_sum_odd_9_to_39_l1222_122251


namespace NUMINAMATH_CALUDE_smallest_digit_not_in_odd_units_l1222_122287

def odd_units_digits : Set Nat := {1, 3, 5, 7, 9}

def is_digit (n : Nat) : Prop := n < 10

theorem smallest_digit_not_in_odd_units : 
  (∀ d, is_digit d → d ∉ odd_units_digits → d ≥ 0) ∧ 
  (0 ∉ odd_units_digits) ∧ 
  is_digit 0 :=
sorry

end NUMINAMATH_CALUDE_smallest_digit_not_in_odd_units_l1222_122287


namespace NUMINAMATH_CALUDE_prob_A_miss_at_least_once_prob_A_hit_twice_B_hit_once_l1222_122216

-- Define the probabilities
def prob_A_hit : ℚ := 2/3
def prob_B_hit : ℚ := 3/4

-- Define the number of shots
def num_shots : ℕ := 3

-- Theorem for part (I)
theorem prob_A_miss_at_least_once :
  1 - prob_A_hit ^ num_shots = 19/27 := by sorry

-- Theorem for part (II)
theorem prob_A_hit_twice_B_hit_once :
  (Nat.choose num_shots 2 : ℚ) * prob_A_hit^2 * (1 - prob_A_hit) *
  (Nat.choose num_shots 1 : ℚ) * prob_B_hit * (1 - prob_B_hit)^2 = 1/16 := by sorry

end NUMINAMATH_CALUDE_prob_A_miss_at_least_once_prob_A_hit_twice_B_hit_once_l1222_122216


namespace NUMINAMATH_CALUDE_children_retaking_test_l1222_122260

theorem children_retaking_test (total : ℝ) (passed : ℝ) (retaking : ℝ) : 
  total = 698.0 → passed = 105.0 → retaking = total - passed → retaking = 593.0 := by
sorry

end NUMINAMATH_CALUDE_children_retaking_test_l1222_122260


namespace NUMINAMATH_CALUDE_total_annual_interest_l1222_122296

theorem total_annual_interest (total_amount first_part : ℕ) : 
  total_amount = 4000 →
  first_part = 2800 →
  (first_part * 3 + (total_amount - first_part) * 5) / 100 = 144 := by
sorry

end NUMINAMATH_CALUDE_total_annual_interest_l1222_122296


namespace NUMINAMATH_CALUDE_square_side_length_of_unit_area_l1222_122222

/-- The side length of a square with area 1 is 1. -/
theorem square_side_length_of_unit_area : 
  ∀ s : ℝ, s > 0 → s * s = 1 → s = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_of_unit_area_l1222_122222


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l1222_122250

theorem quadratic_one_solution (p : ℝ) : 
  (∃! x : ℝ, 3 * x^2 - 6 * x + p = 0) ↔ p = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l1222_122250


namespace NUMINAMATH_CALUDE_m_salary_percentage_l1222_122261

/-- The percentage of m's salary compared to n's salary -/
def salary_percentage (total_salary n_salary : ℚ) : ℚ :=
  (total_salary - n_salary) / n_salary * 100

/-- Proof that m's salary is 120% of n's salary -/
theorem m_salary_percentage :
  let total_salary : ℚ := 572
  let n_salary : ℚ := 260
  salary_percentage total_salary n_salary = 120 := by
  sorry

end NUMINAMATH_CALUDE_m_salary_percentage_l1222_122261


namespace NUMINAMATH_CALUDE_starting_number_is_271_l1222_122208

/-- A function that checks if a natural number contains the digit 1 -/
def contains_one (n : ℕ) : Bool := sorry

/-- The count of numbers from 1 to 1000 (exclusive) that do not contain the digit 1 -/
def count_no_one_to_1000 : ℕ := sorry

/-- The theorem to prove -/
theorem starting_number_is_271 (count_between : ℕ) 
  (h1 : count_between = 728) 
  (h2 : ∀ n ∈ Finset.range (1000 - 271), 
    ¬contains_one (n + 271) ↔ n < count_between) : 
  271 = 1000 - count_between - 1 :=
sorry

end NUMINAMATH_CALUDE_starting_number_is_271_l1222_122208


namespace NUMINAMATH_CALUDE_lowest_sale_price_percentage_l1222_122238

theorem lowest_sale_price_percentage (list_price : ℝ) (regular_discount_max : ℝ) (additional_discount : ℝ) : 
  list_price = 80 →
  regular_discount_max = 0.5 →
  additional_discount = 0.2 →
  (list_price * (1 - regular_discount_max) - list_price * additional_discount) / list_price = 0.3 := by
sorry

end NUMINAMATH_CALUDE_lowest_sale_price_percentage_l1222_122238


namespace NUMINAMATH_CALUDE_g_behavior_l1222_122262

def g (x : ℝ) : ℝ := 3 * x^3 - 5 * x^2 + 1

theorem g_behavior :
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x > N → g x > M) ∧
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x < N → g x < M) := by
sorry

end NUMINAMATH_CALUDE_g_behavior_l1222_122262


namespace NUMINAMATH_CALUDE_triangle_area_l1222_122258

theorem triangle_area (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) : 
  (1/2) * a * b = 24 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1222_122258


namespace NUMINAMATH_CALUDE_zeros_sum_greater_than_twice_intersection_l1222_122201

noncomputable section

variables (a : ℝ) (x₁ x₂ x₀ : ℝ)

def f (x : ℝ) : ℝ := a * Real.log x - (1/2) * x^2

theorem zeros_sum_greater_than_twice_intersection
  (h₁ : a > Real.exp 1)
  (h₂ : f a x₁ = 0)
  (h₃ : f a x₂ = 0)
  (h₄ : x₁ ≠ x₂)
  (h₅ : x₀ = (x₁ + x₂) / ((a / (x₁ * x₂)) + 1)) :
  x₁ + x₂ > 2 * x₀ := by
  sorry

end NUMINAMATH_CALUDE_zeros_sum_greater_than_twice_intersection_l1222_122201


namespace NUMINAMATH_CALUDE_worker_efficiency_l1222_122283

/-- 
Proves that if worker A is thrice as efficient as worker B, 
and A takes 10 days less than B to complete a job, 
then B alone takes 15 days to complete the job.
-/
theorem worker_efficiency (days_b : ℕ) : 
  (days_b / 3 = days_b - 10) → days_b = 15 := by
  sorry

end NUMINAMATH_CALUDE_worker_efficiency_l1222_122283


namespace NUMINAMATH_CALUDE_block_final_height_l1222_122297

/-- Given a block sliding down one ramp and up another, both at angle θ,
    with initial height h₁, mass m, and coefficient of kinetic friction μₖ,
    the final height h₂ is given by h₂ = h₁ / (1 + μₖ * √3) -/
theorem block_final_height
  (m : ℝ) (h₁ : ℝ) (μₖ : ℝ) (θ : ℝ) 
  (h₁_pos : h₁ > 0)
  (m_pos : m > 0)
  (μₖ_pos : μₖ > 0)
  (θ_val : θ = π/6) :
  let h₂ := h₁ / (1 + μₖ * Real.sqrt 3)
  ∀ ε > 0, abs (h₂ - h₁ / (1 + μₖ * Real.sqrt 3)) < ε :=
by
  sorry

#check block_final_height

end NUMINAMATH_CALUDE_block_final_height_l1222_122297


namespace NUMINAMATH_CALUDE_thor_jump_count_l1222_122246

def jump_distance (n : ℕ) : ℝ := 3 * (3 ^ (n - 1))

theorem thor_jump_count :
  (∀ k < 10, jump_distance k ≤ 29000) ∧
  jump_distance 10 > 29000 :=
by sorry

end NUMINAMATH_CALUDE_thor_jump_count_l1222_122246


namespace NUMINAMATH_CALUDE_sunday_bicycles_bought_l1222_122242

/-- Represents the number of bicycles in Hank's store. -/
def BicycleCount := ℤ

/-- Represents the change in bicycle count for a day. -/
structure DailyChange where
  sold : ℕ
  bought : ℕ

/-- Calculates the net change in bicycle count for a day. -/
def netChange (dc : DailyChange) : ℤ :=
  dc.bought - dc.sold

/-- Represents the changes in bicycle count over three days. -/
structure ThreeDayChanges where
  friday : DailyChange
  saturday : DailyChange
  sunday_sold : ℕ

theorem sunday_bicycles_bought 
  (changes : ThreeDayChanges)
  (h_friday : changes.friday = ⟨10, 15⟩)
  (h_saturday : changes.saturday = ⟨12, 8⟩)
  (h_sunday_sold : changes.sunday_sold = 9)
  (h_net_increase : netChange changes.friday + netChange changes.saturday + 
    (sunday_bought - changes.sunday_sold) = 3)
  : ∃ (sunday_bought : ℕ), sunday_bought = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_sunday_bicycles_bought_l1222_122242


namespace NUMINAMATH_CALUDE_inverse_proportion_comparison_l1222_122248

/-- An inverse proportion function passing through (-2, 4) with points (1, y₁) and (3, y₂) on its graph -/
def InverseProportion (k : ℝ) (y₁ y₂ : ℝ) : Prop :=
  k ≠ 0 ∧ 
  4 = k / (-2) ∧ 
  y₁ = k / 1 ∧ 
  y₂ = k / 3

theorem inverse_proportion_comparison (k : ℝ) (y₁ y₂ : ℝ) 
  (h : InverseProportion k y₁ y₂) : y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_comparison_l1222_122248


namespace NUMINAMATH_CALUDE_sqrt_equality_l1222_122275

theorem sqrt_equality (x : ℝ) (h : x < -1) :
  Real.sqrt ((x + 2) / (1 - (x - 2) / (x + 1))) = Real.sqrt ((x^2 + 3*x + 2) / 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equality_l1222_122275


namespace NUMINAMATH_CALUDE_exists_set_product_eq_sum_squares_l1222_122271

/-- For any finite set of positive integers, there exists a larger finite set
    where the product of its elements equals the sum of their squares. -/
theorem exists_set_product_eq_sum_squares (A : Finset ℕ) : ∃ B : Finset ℕ, 
  (∀ a ∈ A, a ∈ B) ∧ 
  (∀ b ∈ B, b > 0) ∧
  (B.prod id = B.sum (λ x => x^2)) := by
  sorry

end NUMINAMATH_CALUDE_exists_set_product_eq_sum_squares_l1222_122271


namespace NUMINAMATH_CALUDE_golden_ratio_trigonometry_l1222_122209

theorem golden_ratio_trigonometry (m n : ℝ) : 
  m = 2 * Real.sin (18 * π / 180) →
  m^2 + n = 4 →
  (m * Real.sqrt n) / (2 * (Real.cos (27 * π / 180))^2 - 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_golden_ratio_trigonometry_l1222_122209


namespace NUMINAMATH_CALUDE_complex_number_equality_l1222_122217

theorem complex_number_equality : ∀ (i : ℂ), i * i = -1 →
  (2 * i) / (1 + i) = 1 + i := by sorry

end NUMINAMATH_CALUDE_complex_number_equality_l1222_122217


namespace NUMINAMATH_CALUDE_frigate_catches_smuggler_l1222_122213

/-- Represents the chase scenario between a frigate and a smuggler's ship -/
structure ChaseScenario where
  initial_distance : ℝ
  frigate_speed : ℝ
  smuggler_speed : ℝ
  chase_duration : ℝ

/-- Calculates the distance traveled by a ship given its speed and time -/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem stating that the frigate catches up to the smuggler's ship after 3 hours -/
theorem frigate_catches_smuggler (scenario : ChaseScenario) 
    (h1 : scenario.initial_distance = 12)
    (h2 : scenario.frigate_speed = 14)
    (h3 : scenario.smuggler_speed = 10)
    (h4 : scenario.chase_duration = 3) :
    distance_traveled scenario.frigate_speed scenario.chase_duration = 
    scenario.initial_distance + distance_traveled scenario.smuggler_speed scenario.chase_duration :=
  sorry

#check frigate_catches_smuggler

end NUMINAMATH_CALUDE_frigate_catches_smuggler_l1222_122213


namespace NUMINAMATH_CALUDE_initial_ratio_of_men_to_women_l1222_122234

theorem initial_ratio_of_men_to_women 
  (initial_men : ℕ) 
  (initial_women : ℕ) 
  (final_men : ℕ) 
  (final_women : ℕ) 
  (h1 : final_men = initial_men + 2)
  (h2 : final_women = 2 * (initial_women - 3))
  (h3 : final_men = 14)
  (h4 : final_women = 24) :
  initial_men / initial_women = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_initial_ratio_of_men_to_women_l1222_122234


namespace NUMINAMATH_CALUDE_min_real_part_x_l1222_122255

theorem min_real_part_x (x y : ℂ) 
  (eq1 : x + 2 * y^2 = x^4)
  (eq2 : y + 2 * x^2 = y^4) :
  Real.sqrt (Real.sqrt ((1 - Real.sqrt 33) / 2)) ≤ x.re :=
sorry

end NUMINAMATH_CALUDE_min_real_part_x_l1222_122255


namespace NUMINAMATH_CALUDE_even_product_sufficiency_not_necessity_l1222_122205

/-- A function f is even if f(-x) = f(x) for all x --/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem even_product_sufficiency_not_necessity :
  (∀ f g : ℝ → ℝ, IsEven f ∧ IsEven g → IsEven (fun x ↦ f x * g x)) ∧
  (∃ f g : ℝ → ℝ, IsEven (fun x ↦ f x * g x) ∧ (¬IsEven f ∨ ¬IsEven g)) := by
  sorry

end NUMINAMATH_CALUDE_even_product_sufficiency_not_necessity_l1222_122205


namespace NUMINAMATH_CALUDE_paul_picked_72_cans_l1222_122281

/-- The total number of cans Paul picked up -/
def total_cans (saturday_bags : ℕ) (sunday_bags : ℕ) (cans_per_bag : ℕ) : ℕ :=
  (saturday_bags + sunday_bags) * cans_per_bag

/-- Theorem stating that Paul picked up 72 cans in total -/
theorem paul_picked_72_cans :
  total_cans 6 3 8 = 72 := by
  sorry

end NUMINAMATH_CALUDE_paul_picked_72_cans_l1222_122281


namespace NUMINAMATH_CALUDE_agri_product_sales_model_l1222_122293

/-- Agricultural product sales model -/
structure AgriProduct where
  cost_price : ℝ
  sales_quantity : ℝ → ℝ
  max_price : ℝ

/-- Daily sales profit function -/
def daily_profit (p : AgriProduct) (x : ℝ) : ℝ :=
  x * (p.sales_quantity x) - p.cost_price * (p.sales_quantity x)

/-- Theorem stating the properties of the agricultural product sales model -/
theorem agri_product_sales_model (p : AgriProduct) 
  (h_cost : p.cost_price = 20)
  (h_quantity : ∀ x, p.sales_quantity x = -2 * x + 80)
  (h_max_price : p.max_price = 30) :
  (∀ x, daily_profit p x = -2 * x^2 + 120 * x - 1600) ∧
  (∃ x, x ≤ p.max_price ∧ daily_profit p x = 150 ∧ x = 25) :=
sorry

end NUMINAMATH_CALUDE_agri_product_sales_model_l1222_122293


namespace NUMINAMATH_CALUDE_part_one_part_two_l1222_122280

/-- Definition of arithmetic sequence sum -/
def arithmetic_sum (a₁ d : ℚ) (n : ℕ) : ℚ :=
  n * a₁ + (n * (n - 1) / 2) * d

/-- Theorem for part (I) -/
theorem part_one :
  ∃! k : ℕ+, arithmetic_sum (3/2) 1 (k^2) = (arithmetic_sum (3/2) 1 k)^2 :=
sorry

/-- Definition of arithmetic sequence -/
def arithmetic_seq (a₁ d : ℚ) (n : ℕ) : ℚ :=
  a₁ + (n - 1) * d

/-- Theorem for part (II) -/
theorem part_two :
  ∀ a₁ d : ℚ, (∀ k : ℕ+, arithmetic_sum a₁ d (k^2) = (arithmetic_sum a₁ d k)^2) ↔
    ((∀ n, arithmetic_seq a₁ d n = 0) ∨
     (∀ n, arithmetic_seq a₁ d n = 1) ∨
     (∀ n, arithmetic_seq a₁ d n = 2 * n - 1)) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1222_122280


namespace NUMINAMATH_CALUDE_parabola_directrix_l1222_122294

/-- A parabola is defined by its equation in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The directrix of a parabola is a line parallel to the x-axis -/
structure Directrix where
  y : ℝ

/-- Given a parabola y = (x^2 - 4x + 4) / 8, its directrix is y = -2 -/
theorem parabola_directrix (p : Parabola) (d : Directrix) : 
  p.a = 1/8 ∧ p.b = -1/2 ∧ p.c = 1/2 → d.y = -2 := by
  sorry


end NUMINAMATH_CALUDE_parabola_directrix_l1222_122294


namespace NUMINAMATH_CALUDE_system_solution_unique_l1222_122204

theorem system_solution_unique : 
  ∃! (x y : ℝ), (2 * x - 3 * y = 5) ∧ (4 * x - y = 5) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l1222_122204


namespace NUMINAMATH_CALUDE_mangoes_rate_per_kg_l1222_122245

/-- Given the conditions of Harkamal's purchase, prove that the rate per kg of mangoes is 55. -/
theorem mangoes_rate_per_kg (grapes_quantity : ℕ) (grapes_rate : ℕ) (mangoes_quantity : ℕ) (total_paid : ℕ) :
  grapes_quantity = 8 →
  grapes_rate = 80 →
  mangoes_quantity = 9 →
  total_paid = 1135 →
  (total_paid - grapes_quantity * grapes_rate) / mangoes_quantity = 55 := by
  sorry

#eval (1135 - 8 * 80) / 9  -- This should evaluate to 55

end NUMINAMATH_CALUDE_mangoes_rate_per_kg_l1222_122245


namespace NUMINAMATH_CALUDE_vector_equation_solution_l1222_122232

theorem vector_equation_solution :
  let a₁ : ℚ := 181 / 136
  let a₂ : ℚ := 25 / 68
  let v₁ : Fin 2 → ℚ := ![4, -1]
  let v₂ : Fin 2 → ℚ := ![5, 3]
  let result : Fin 2 → ℚ := ![9, 4]
  (a₁ • v₁ + a₂ • v₂) = result := by sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l1222_122232


namespace NUMINAMATH_CALUDE_parallelogram_height_l1222_122273

theorem parallelogram_height (area : ℝ) (base : ℝ) (height : ℝ) :
  area = 360 ∧ base = 30 ∧ area = base * height → height = 12 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_height_l1222_122273


namespace NUMINAMATH_CALUDE_sum_of_sequence_equals_11920_l1222_122211

def integerSequence : List Nat := List.range 40 |>.map (fun i => 103 + 10 * i)

theorem sum_of_sequence_equals_11920 : (integerSequence.sum = 11920) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_sequence_equals_11920_l1222_122211


namespace NUMINAMATH_CALUDE_stratified_sampling_medium_supermarkets_l1222_122236

theorem stratified_sampling_medium_supermarkets 
  (total_sample : ℕ) 
  (large_supermarkets : ℕ) 
  (medium_supermarkets : ℕ) 
  (small_supermarkets : ℕ) 
  (h_total_sample : total_sample = 100)
  (h_large : large_supermarkets = 200)
  (h_medium : medium_supermarkets = 400)
  (h_small : small_supermarkets = 1400) : 
  (total_sample * medium_supermarkets) / (large_supermarkets + medium_supermarkets + small_supermarkets) = 20 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_medium_supermarkets_l1222_122236


namespace NUMINAMATH_CALUDE_distinct_z_values_l1222_122263

def is_two_digit (n : ℤ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℤ) : ℤ :=
  10 * (n % 10) + (n / 10)

def z (x : ℤ) : ℤ := |x - reverse_digits x|

theorem distinct_z_values (x : ℤ) (hx : is_two_digit x) :
  ∃ (S : Finset ℤ), (∀ y, is_two_digit y → z y ∈ S) ∧ Finset.card S = 8 := by
  sorry

#check distinct_z_values

end NUMINAMATH_CALUDE_distinct_z_values_l1222_122263


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1222_122249

theorem inequality_solution_set (x : ℝ) : (3 - 2*x) * (x + 1) ≤ 0 ↔ x < -1 ∨ x ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1222_122249


namespace NUMINAMATH_CALUDE_total_amount_l1222_122203

/-- Represents the distribution of money among three parties -/
structure MoneyDistribution where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The problem setup -/
def problem_setup (d : MoneyDistribution) : Prop :=
  d.y = 0.45 * d.x ∧ 
  d.z = 0.50 * d.x ∧
  d.y = 36

/-- The theorem to prove -/
theorem total_amount (d : MoneyDistribution) :
  problem_setup d → d.x + d.y + d.z = 156 := by
  sorry


end NUMINAMATH_CALUDE_total_amount_l1222_122203


namespace NUMINAMATH_CALUDE_computer_price_increase_l1222_122298

theorem computer_price_increase (d : ℝ) (h1 : 2 * d = 520) (h2 : d > 0) : 
  (338 - d) / d * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_increase_l1222_122298


namespace NUMINAMATH_CALUDE_binomial_identity_sum_identity_l1222_122235

def binomial (n p : ℕ) : ℕ := if p ≤ n then n.factorial / (p.factorial * (n - p).factorial) else 0

theorem binomial_identity (n p : ℕ) (h : n ≥ p ∧ p ≥ 1) :
  binomial n p = (Finset.range (n - p + 1)).sum (fun i => binomial (n - 1 - i) (p - 1)) :=
sorry

theorem sum_identity :
  (Finset.range 97).sum (fun k => (k + 1) * (k + 2) * (k + 3)) = 23527350 :=
sorry

end NUMINAMATH_CALUDE_binomial_identity_sum_identity_l1222_122235


namespace NUMINAMATH_CALUDE_sum_of_basic_terms_divisible_by_four_l1222_122266

/-- A type representing a grid cell that can be either +1 or -1 -/
inductive GridCell
  | pos : GridCell
  | neg : GridCell

/-- A type representing an n × n grid filled with +1 or -1 -/
def Grid (n : ℕ) := Fin n → Fin n → GridCell

/-- A basic term is a product of n cells, no two of which share the same row or column -/
def BasicTerm (n : ℕ) (grid : Grid n) (perm : Equiv.Perm (Fin n)) : ℤ :=
  (Finset.univ.prod fun i => match grid i (perm i) with
    | GridCell.pos => 1
    | GridCell.neg => -1)

/-- The sum of all basic terms for a given grid -/
def SumOfBasicTerms (n : ℕ) (grid : Grid n) : ℤ :=
  (Finset.univ : Finset (Equiv.Perm (Fin n))).sum fun perm => BasicTerm n grid perm

/-- The main theorem: for any n × n grid (n ≥ 4), the sum of all basic terms is divisible by 4 -/
theorem sum_of_basic_terms_divisible_by_four {n : ℕ} (h : n ≥ 4) (grid : Grid n) :
  4 ∣ SumOfBasicTerms n grid := by
  sorry

end NUMINAMATH_CALUDE_sum_of_basic_terms_divisible_by_four_l1222_122266


namespace NUMINAMATH_CALUDE_smallest_next_divisor_after_523_l1222_122221

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

theorem smallest_next_divisor_after_523 (m : ℕ) 
  (h1 : is_five_digit m) 
  (h2 : Even m) 
  (h3 : m % 523 = 0) :
  ∃ (d : ℕ), d > 523 ∧ m % d = 0 ∧ (∀ (k : ℕ), 523 < k ∧ k < d → m % k ≠ 0) → d = 524 :=
sorry

end NUMINAMATH_CALUDE_smallest_next_divisor_after_523_l1222_122221


namespace NUMINAMATH_CALUDE_vector_subtraction_scalar_multiplication_l1222_122225

def vector_operation (v1 v2 : ℝ × ℝ) (s : ℝ) : ℝ × ℝ :=
  (v1.1 - s * v2.1, v1.2 - s * v2.2)

theorem vector_subtraction_scalar_multiplication :
  vector_operation (3, -8) (2, -6) 5 = (-7, 22) := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_scalar_multiplication_l1222_122225


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1222_122265

/-- A hyperbola with foci F₁ and F₂ -/
structure Hyperbola where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- A line passing through a point -/
structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Intersection points of a line with a hyperbola -/
def intersection (h : Hyperbola) (l : Line) : Set (ℝ × ℝ) :=
  sorry

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ :=
  sorry

/-- Check if a triangle is equilateral -/
def is_equilateral (p q r : ℝ × ℝ) : Prop :=
  sorry

theorem hyperbola_eccentricity (h : Hyperbola) (l : Line) :
  l.point = h.F₂ →
  ∃ (A B : ℝ × ℝ), A ∈ intersection h l ∧ B ∈ intersection h l ∧
  is_equilateral h.F₁ A B →
  eccentricity h = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1222_122265


namespace NUMINAMATH_CALUDE_truck_filling_time_l1222_122239

/-- Calculates the total time to fill a truck with stone blocks -/
theorem truck_filling_time 
  (truck_capacity : ℕ)
  (rate_per_person : ℕ)
  (initial_workers : ℕ)
  (initial_duration : ℕ)
  (additional_workers : ℕ)
  (h1 : truck_capacity = 6000)
  (h2 : rate_per_person = 250)
  (h3 : initial_workers = 2)
  (h4 : initial_duration = 4)
  (h5 : additional_workers = 6) :
  ∃ (total_time : ℕ), total_time = 6 ∧ 
  (initial_workers * rate_per_person * initial_duration + 
   (initial_workers + additional_workers) * rate_per_person * (total_time - initial_duration) = truck_capacity) :=
by
  sorry


end NUMINAMATH_CALUDE_truck_filling_time_l1222_122239


namespace NUMINAMATH_CALUDE_opposite_pairs_l1222_122274

theorem opposite_pairs : 
  (3^2 = -(-3^2)) ∧ 
  (3^2 ≠ -2^3) ∧ 
  (3^2 ≠ -(-3)^2) ∧ 
  (-3^2 ≠ -(-3)^2) := by
  sorry

end NUMINAMATH_CALUDE_opposite_pairs_l1222_122274


namespace NUMINAMATH_CALUDE_day_of_week_theorem_l1222_122228

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Given a year and a day number, returns the day of the week -/
def dayOfWeek (year : ℤ) (dayNumber : ℕ) : DayOfWeek := sorry

theorem day_of_week_theorem (M : ℤ) :
  dayOfWeek M 200 = DayOfWeek.Monday →
  dayOfWeek (M + 2) 300 = DayOfWeek.Monday →
  dayOfWeek (M - 1) 100 = DayOfWeek.Tuesday :=
by sorry

end NUMINAMATH_CALUDE_day_of_week_theorem_l1222_122228


namespace NUMINAMATH_CALUDE_deandre_jordan_free_throws_l1222_122202

/-- The probability of scoring at least one point in two free throw attempts -/
def prob_at_least_one_point (success_rate : ℝ) : ℝ :=
  1 - (1 - success_rate) ^ 2

theorem deandre_jordan_free_throws :
  let success_rate : ℝ := 0.4
  prob_at_least_one_point success_rate = 0.64 := by
  sorry

end NUMINAMATH_CALUDE_deandre_jordan_free_throws_l1222_122202
