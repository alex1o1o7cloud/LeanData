import Mathlib

namespace NUMINAMATH_CALUDE_fruit_bowl_oranges_l80_8041

theorem fruit_bowl_oranges :
  let bananas : ℕ := 7
  let apples : ℕ := 2 * bananas
  let pears : ℕ := 4
  let grapes : ℕ := apples / 2
  let total_fruits : ℕ := 40
  let oranges : ℕ := total_fruits - (bananas + apples + pears + grapes)
  oranges = 8 := by sorry

end NUMINAMATH_CALUDE_fruit_bowl_oranges_l80_8041


namespace NUMINAMATH_CALUDE_sharp_composition_50_l80_8042

-- Define the # operation
def sharp (N : ℝ) : ℝ := 0.5 * N + 1

-- Theorem statement
theorem sharp_composition_50 : sharp (sharp (sharp 50)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_sharp_composition_50_l80_8042


namespace NUMINAMATH_CALUDE_modulus_of_complex_fourth_power_l80_8017

theorem modulus_of_complex_fourth_power : 
  Complex.abs ((2 : ℂ) + (3 * Real.sqrt 2) * Complex.I) ^ 4 = 484 := by sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fourth_power_l80_8017


namespace NUMINAMATH_CALUDE_muffin_price_theorem_l80_8069

/-- The price per muffin to raise the required amount -/
def price_per_muffin (total_amount : ℚ) (num_cases : ℕ) (packs_per_case : ℕ) (muffins_per_pack : ℕ) : ℚ :=
  total_amount / (num_cases * packs_per_case * muffins_per_pack)

/-- Theorem: The price per muffin to raise $120 by selling 5 cases of muffins, 
    where each case contains 3 packs and each pack contains 4 muffins, is $2 -/
theorem muffin_price_theorem :
  price_per_muffin 120 5 3 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_muffin_price_theorem_l80_8069


namespace NUMINAMATH_CALUDE_jeff_took_six_cans_l80_8084

/-- Represents the number of soda cans in various stages --/
structure SodaCans where
  initial : ℕ
  taken : ℕ
  final : ℕ

/-- Calculates the number of cans Jeff took from Tim --/
def cans_taken (s : SodaCans) : Prop :=
  s.initial - s.taken + (s.initial - s.taken) / 2 = s.final

/-- The main theorem to prove --/
theorem jeff_took_six_cans : ∃ (s : SodaCans), s.initial = 22 ∧ s.final = 24 ∧ s.taken = 6 ∧ cans_taken s := by
  sorry


end NUMINAMATH_CALUDE_jeff_took_six_cans_l80_8084


namespace NUMINAMATH_CALUDE_hyperbola_symmetry_l80_8087

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola x² - y² = a² -/
def Hyperbola (a : ℝ) : Set Point :=
  {p : Point | p.x^2 - p.y^2 = a^2}

/-- Represents the line y = x - 2 -/
def SymmetryLine : Set Point :=
  {p : Point | p.y = p.x - 2}

/-- Represents the line 2x + 3y = 6 -/
def TangentLine : Set Point :=
  {p : Point | 2 * p.x + 3 * p.y = 6}

/-- Defines symmetry about a line -/
def SymmetricPoint (p : Point) : Point :=
  ⟨p.y + 2, p.x - 2⟩

/-- Defines the curve C₂ symmetric to C₁ about the symmetry line -/
def C₂ (a : ℝ) : Set Point :=
  {p : Point | SymmetricPoint p ∈ Hyperbola a}

/-- States that the tangent line is tangent to C₂ -/
def IsTangent (a : ℝ) : Prop :=
  ∃ p : Point, p ∈ C₂ a ∧ p ∈ TangentLine

theorem hyperbola_symmetry (a : ℝ) (h : a > 0) (h_tangent : IsTangent a) : 
  a = 8 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_symmetry_l80_8087


namespace NUMINAMATH_CALUDE_rectangle_ratio_in_square_config_l80_8016

-- Define the structure of our square-rectangle configuration
structure SquareRectConfig where
  inner_side : ℝ
  rect_short : ℝ
  rect_long : ℝ

-- State the theorem
theorem rectangle_ratio_in_square_config (config : SquareRectConfig) :
  -- The outer square's side is composed of the inner square's side and two short sides of rectangles
  config.inner_side + 2 * 2 * config.rect_short = 3 * config.inner_side →
  -- Two long sides and one short side of rectangles make up the outer square's side
  2 * config.rect_long + config.rect_short = 3 * config.inner_side →
  -- The ratio of long to short sides of the rectangle is 2.5
  config.rect_long / config.rect_short = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_in_square_config_l80_8016


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_plus_8y_equals_16_l80_8005

theorem x_squared_minus_y_squared_plus_8y_equals_16 
  (x y : ℝ) (h : x + y = 4) : x^2 - y^2 + 8*y = 16 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_plus_8y_equals_16_l80_8005


namespace NUMINAMATH_CALUDE_triangle_base_value_l80_8033

theorem triangle_base_value (square_perimeter : ℝ) (triangle_height : ℝ) (x : ℝ) :
  square_perimeter = 64 →
  triangle_height = 32 →
  (square_perimeter / 4) ^ 2 = (1 / 2) * x * triangle_height →
  x = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_base_value_l80_8033


namespace NUMINAMATH_CALUDE_perpendicular_line_correct_l80_8083

/-- A line in polar coordinates passing through (2, 0) and perpendicular to the polar axis --/
def perpendicular_line (ρ θ : ℝ) : Prop :=
  ρ * Real.cos θ = 2

theorem perpendicular_line_correct :
  ∀ ρ θ : ℝ, perpendicular_line ρ θ ↔ 
    (ρ * Real.cos θ = 2 ∧ ρ * Real.sin θ = 0) ∨
    (ρ = 2 ∧ θ = 0) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_correct_l80_8083


namespace NUMINAMATH_CALUDE_allocation_methods_count_l80_8054

def number_of_doctors : ℕ := 3
def number_of_nurses : ℕ := 6
def number_of_schools : ℕ := 3
def doctors_per_school : ℕ := 1
def nurses_per_school : ℕ := 2

theorem allocation_methods_count :
  (Nat.choose number_of_doctors doctors_per_school) *
  (Nat.choose number_of_nurses nurses_per_school) *
  (Nat.choose (number_of_doctors - doctors_per_school) doctors_per_school) *
  (Nat.choose (number_of_nurses - nurses_per_school) nurses_per_school) = 540 := by
  sorry

end NUMINAMATH_CALUDE_allocation_methods_count_l80_8054


namespace NUMINAMATH_CALUDE_office_age_problem_l80_8098

theorem office_age_problem (total_persons : Nat) (avg_age_all : Nat) (group1_size : Nat) 
  (group1_avg_age : Nat) (group2_size : Nat) (group2_avg_age : Nat) :
  total_persons = 19 →
  avg_age_all = 15 →
  group1_size = 5 →
  group1_avg_age = 14 →
  group2_size = 9 →
  group2_avg_age = 16 →
  (total_persons * avg_age_all) - (group1_size * group1_avg_age + group2_size * group2_avg_age) = 71 := by
  sorry

#check office_age_problem

end NUMINAMATH_CALUDE_office_age_problem_l80_8098


namespace NUMINAMATH_CALUDE_system_solution_l80_8076

theorem system_solution : 
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁^2 + x₁*y₁ + y₁ = 1 ∧ y₁^2 + x₁*y₁ + x₁ = 5) ∧
    (x₂^2 + x₂*y₂ + y₂ = 1 ∧ y₂^2 + x₂*y₂ + x₂ = 5) ∧
    x₁ = -1 ∧ y₁ = 3 ∧ x₂ = -1 ∧ y₂ = -2 ∧
    ∀ (x y : ℝ), (x^2 + x*y + y = 1 ∧ y^2 + x*y + x = 5) → 
      ((x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l80_8076


namespace NUMINAMATH_CALUDE_favorite_number_is_27_l80_8002

theorem favorite_number_is_27 : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ 
  n^2 = (n / 10 + n % 10)^3 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_favorite_number_is_27_l80_8002


namespace NUMINAMATH_CALUDE_cages_needed_proof_l80_8089

def initial_gerbils : ℕ := 150
def sold_gerbils : ℕ := 98

theorem cages_needed_proof :
  initial_gerbils - sold_gerbils = 52 :=
by sorry

end NUMINAMATH_CALUDE_cages_needed_proof_l80_8089


namespace NUMINAMATH_CALUDE_complex_equation_real_l80_8094

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_equation_real (a : ℝ) : 
  (((2 * a : ℂ) / (1 + i) + 1 + i).im = 0) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_real_l80_8094


namespace NUMINAMATH_CALUDE_equation_solutions_l80_8040

-- Define the equations as functions
def eqnA (x : ℝ) := (3*x + 1)^2 = 0
def eqnB (x : ℝ) := |2*x + 1| - 6 = 0
def eqnC (x : ℝ) := Real.sqrt (5 - x) + 3 = 0
def eqnD (x : ℝ) := Real.sqrt (4*x + 9) - 7 = 0
def eqnE (x : ℝ) := |5*x - 3| + 2 = -1

-- Define the existence of solutions
def has_solution (f : ℝ → Prop) := ∃ x, f x

-- Theorem statement
theorem equation_solutions :
  (has_solution eqnA) ∧
  (has_solution eqnB) ∧
  (¬ has_solution eqnC) ∧
  (has_solution eqnD) ∧
  (¬ has_solution eqnE) :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l80_8040


namespace NUMINAMATH_CALUDE_shoes_selection_ways_l80_8032

/-- The number of pairs of distinct shoes in the bag -/
def total_pairs : ℕ := 10

/-- The number of shoes taken out -/
def shoes_taken : ℕ := 4

/-- The number of ways to select 4 shoes from 10 pairs such that
    exactly two form a pair and the other two don't form a pair -/
def ways_to_select : ℕ := 1440

/-- Theorem stating the number of ways to select 4 shoes from 10 pairs
    such that exactly two form a pair and the other two don't form a pair -/
theorem shoes_selection_ways (n : ℕ) (h : n = total_pairs) :
  ways_to_select = Nat.choose n 1 * Nat.choose (n - 1) 2 * 2^2 :=
sorry

end NUMINAMATH_CALUDE_shoes_selection_ways_l80_8032


namespace NUMINAMATH_CALUDE_convex_polygon_sides_l80_8079

theorem convex_polygon_sides (n : ℕ) (a₁ : ℝ) (d : ℝ) : 
  n > 2 →
  a₁ = 120 →
  d = 5 →
  (n - 2) * 180 = (2 * a₁ + (n - 1) * d) * n / 2 →
  (∀ k : ℕ, k ≤ n → a₁ + (k - 1) * d < 180) →
  n = 9 := by
sorry

end NUMINAMATH_CALUDE_convex_polygon_sides_l80_8079


namespace NUMINAMATH_CALUDE_max_side_length_of_special_triangle_l80_8011

theorem max_side_length_of_special_triangle (a b c : ℕ) : 
  a < b → b < c →                 -- Three different side lengths
  a + b + c = 24 →                -- Perimeter is 24
  a + b > c → b + c > a → c + a > b →  -- Triangle inequality
  c ≤ 11 := by
sorry

end NUMINAMATH_CALUDE_max_side_length_of_special_triangle_l80_8011


namespace NUMINAMATH_CALUDE_greatest_ba_value_l80_8066

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_divisible_by (a b : ℕ) : Prop := a % b = 0

theorem greatest_ba_value (a b : ℕ) :
  is_prime a →
  is_prime b →
  a < 10 →
  b < 10 →
  is_divisible_by (110 * 10 + a * 10 + b) 55 →
  (∀ a' b' : ℕ, 
    is_prime a' → 
    is_prime b' → 
    a' < 10 → 
    b' < 10 → 
    is_divisible_by (110 * 10 + a' * 10 + b') 55 → 
    b * a ≥ b' * a') →
  b * a = 15 := by
sorry

end NUMINAMATH_CALUDE_greatest_ba_value_l80_8066


namespace NUMINAMATH_CALUDE_fish_added_calculation_james_added_eight_fish_l80_8019

theorem fish_added_calculation (initial_fish : ℕ) (fish_eaten_per_day : ℕ) 
  (days_before_adding : ℕ) (days_after_adding : ℕ) (final_fish : ℕ) : ℕ :=
  let total_days := days_before_adding + days_after_adding
  let total_fish_eaten := total_days * fish_eaten_per_day
  let expected_remaining := initial_fish - total_fish_eaten
  final_fish - expected_remaining
  
-- The main theorem
theorem james_added_eight_fish : 
  fish_added_calculation 60 2 14 7 26 = 8 := by
sorry

end NUMINAMATH_CALUDE_fish_added_calculation_james_added_eight_fish_l80_8019


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l80_8057

theorem roots_of_polynomial (x : ℝ) : 
  x^3 - 3*x^2 - x + 3 = 0 ↔ x = 1 ∨ x = -1 ∨ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l80_8057


namespace NUMINAMATH_CALUDE_H_range_l80_8065

def H (x : ℝ) : ℝ := |x + 2| - |x - 3|

theorem H_range : ∀ y : ℝ, (∃ x : ℝ, H x = y) ↔ -5 ≤ y ∧ y ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_H_range_l80_8065


namespace NUMINAMATH_CALUDE_total_profit_is_100_l80_8020

/-- Calculates the total profit given investments and A's profit share -/
def calculate_total_profit (a_investment : ℕ) (a_months : ℕ) (b_investment : ℕ) (b_months : ℕ) (a_profit_share : ℕ) : ℕ :=
  let a_investment_share := a_investment * a_months
  let b_investment_share := b_investment * b_months
  let total_investment_share := a_investment_share + b_investment_share
  let total_profit := a_profit_share * total_investment_share / a_investment_share
  total_profit

/-- Theorem stating that given the specified investments and A's profit share, the total profit is 100 -/
theorem total_profit_is_100 :
  calculate_total_profit 300 12 200 6 75 = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_profit_is_100_l80_8020


namespace NUMINAMATH_CALUDE_smaller_equals_larger_l80_8030

/-- A circle with an inscribed rectangle and a smaller rectangle -/
structure InscribedRectangles where
  /-- The radius of the circle -/
  r : ℝ
  /-- Half-width of the larger rectangle -/
  a : ℝ
  /-- Half-height of the larger rectangle -/
  b : ℝ
  /-- Proportion of the smaller rectangle's side to the larger rectangle's side -/
  x : ℝ
  /-- The larger rectangle is inscribed in the circle -/
  inscribed : r^2 = a^2 + b^2
  /-- The smaller rectangle has two vertices on the circle -/
  smaller_on_circle : r^2 = (a*x)^2 + (b*x)^2
  /-- The smaller rectangle's side coincides with the larger rectangle's side -/
  coincide : 0 < x ∧ x ≤ 1

/-- The area of the smaller rectangle is equal to the area of the larger rectangle -/
theorem smaller_equals_larger (ir : InscribedRectangles) : 
  (ir.a * ir.x) * (ir.b * ir.x) = ir.a * ir.b := by
  sorry

end NUMINAMATH_CALUDE_smaller_equals_larger_l80_8030


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l80_8038

theorem solve_exponential_equation :
  ∃ t : ℝ, 4 * (4^t) + Real.sqrt (16 * (16^t)) = 32 ∧ t = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l80_8038


namespace NUMINAMATH_CALUDE_carol_picked_29_carrots_l80_8031

/-- The number of carrots Carol picked -/
def carols_carrots (total_carrots good_carrots bad_carrots moms_carrots : ℕ) : ℕ :=
  total_carrots - moms_carrots

/-- Theorem stating that Carol picked 29 carrots -/
theorem carol_picked_29_carrots 
  (total_carrots : ℕ) 
  (good_carrots : ℕ) 
  (bad_carrots : ℕ) 
  (moms_carrots : ℕ) 
  (h1 : total_carrots = good_carrots + bad_carrots)
  (h2 : good_carrots = 38)
  (h3 : bad_carrots = 7)
  (h4 : moms_carrots = 16) :
  carols_carrots total_carrots good_carrots bad_carrots moms_carrots = 29 := by
  sorry

end NUMINAMATH_CALUDE_carol_picked_29_carrots_l80_8031


namespace NUMINAMATH_CALUDE_expression_minimum_l80_8018

theorem expression_minimum (x : ℝ) (h : 1 < x ∧ x < 5) : 
  ∃ (y : ℝ), y = (x^2 - 4*x + 5) / (2*x - 6) ∧ 
  (∀ (z : ℝ), 1 < z ∧ z < 5 → (z^2 - 4*z + 5) / (2*z - 6) ≥ y) ∧
  y = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_minimum_l80_8018


namespace NUMINAMATH_CALUDE_rectangle_length_equal_square_side_l80_8056

/-- The length of a rectangle with width 3 cm and area equal to a 3 cm square -/
theorem rectangle_length_equal_square_side : 
  ∀ (length : ℝ), 
  (3 : ℝ) * length = (3 : ℝ) * (3 : ℝ) → 
  length = (3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_equal_square_side_l80_8056


namespace NUMINAMATH_CALUDE_large_number_arithmetic_l80_8021

/-- The result of a series of arithmetic operations on large numbers. -/
theorem large_number_arithmetic :
  let start : ℕ := 1500000000000
  let subtract : ℕ := 877888888888
  let add : ℕ := 123456789012
  (start - subtract + add : ℕ) = 745567900124 := by
  sorry

end NUMINAMATH_CALUDE_large_number_arithmetic_l80_8021


namespace NUMINAMATH_CALUDE_consecutive_integers_divisibility_l80_8086

theorem consecutive_integers_divisibility : ∃ (a b c : ℕ), 
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧  -- positive integers
  (b = a + 1) ∧ (c = b + 1) ∧    -- consecutive
  (a % 1 = 0) ∧                  -- a divisible by (b - a)^2
  (a % 4 = 0) ∧                  -- a divisible by (c - a)^2
  (b % 1 = 0) :=                 -- b divisible by (c - b)^2
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_divisibility_l80_8086


namespace NUMINAMATH_CALUDE_luncheon_invitees_l80_8045

theorem luncheon_invitees (no_shows : ℕ) (table_capacity : ℕ) (tables_needed : ℕ) :
  no_shows = 10 →
  table_capacity = 7 →
  tables_needed = 2 →
  no_shows + (tables_needed * table_capacity) = 24 := by
sorry

end NUMINAMATH_CALUDE_luncheon_invitees_l80_8045


namespace NUMINAMATH_CALUDE_euler_line_l80_8058

/-- The centroid of a triangle -/
def centroid (A B C : ℝ × ℝ) : ℝ × ℝ := sorry

/-- The orthocenter of a triangle -/
def orthocenter (A B C : ℝ × ℝ) : ℝ × ℝ := sorry

/-- The circumcenter of a triangle -/
def circumcenter (A B C : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Three points are collinear -/
def collinear (P Q R : ℝ × ℝ) : Prop := sorry

theorem euler_line (A B C : ℝ × ℝ) : 
  collinear (centroid A B C) (orthocenter A B C) (circumcenter A B C) := by
  sorry

end NUMINAMATH_CALUDE_euler_line_l80_8058


namespace NUMINAMATH_CALUDE_solution_exists_unique_solution_l80_8070

theorem solution_exists : ∃ x : ℚ, 60 + x * 12 / (180 / 3) = 61 :=
by
  use 5
  sorry

theorem unique_solution (x : ℚ) : 60 + x * 12 / (180 / 3) = 61 ↔ x = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_solution_exists_unique_solution_l80_8070


namespace NUMINAMATH_CALUDE_continuous_stripe_probability_l80_8010

/-- Represents a cube with stripes on its faces -/
structure StripedCube where
  faces : Fin 6 → Fin 3

/-- The probability of a continuous stripe encircling the cube -/
def probability_continuous_stripe : ℚ := 2 / 81

/-- The total number of possible stripe configurations -/
def total_configurations : ℕ := 3^6

/-- The number of configurations that result in a continuous stripe -/
def favorable_configurations : ℕ := 18

theorem continuous_stripe_probability :
  probability_continuous_stripe = favorable_configurations / total_configurations :=
sorry

end NUMINAMATH_CALUDE_continuous_stripe_probability_l80_8010


namespace NUMINAMATH_CALUDE_some_number_value_l80_8091

theorem some_number_value (x : ℝ) : 40 + x * 12 / (180 / 3) = 41 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l80_8091


namespace NUMINAMATH_CALUDE_square_sum_theorem_l80_8044

theorem square_sum_theorem (x y : ℝ) (h1 : x + y = -10) (h2 : x = 25 / y) : x^2 + y^2 = 50 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_theorem_l80_8044


namespace NUMINAMATH_CALUDE_standard_form_is_quadratic_expanded_form_is_quadratic_l80_8008

/-- Definition of a quadratic equation -/
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation ax^2 + bx + c = 0 (where a ≠ 0) is quadratic -/
theorem standard_form_is_quadratic (a b c : ℝ) (ha : a ≠ 0) :
  is_quadratic (λ x => a * x^2 + b * x + c) :=
sorry

/-- The equation (x-2)^2 - 4 = 0 is quadratic -/
theorem expanded_form_is_quadratic :
  is_quadratic (λ x => (x - 2)^2 - 4) :=
sorry

end NUMINAMATH_CALUDE_standard_form_is_quadratic_expanded_form_is_quadratic_l80_8008


namespace NUMINAMATH_CALUDE_range_of_x_l80_8099

theorem range_of_x (a b x : ℝ) (h_a : a ≠ 0) :
  (∀ a b, |a + b| + |a - b| ≥ |a| * (|x - 1| + |x - 2|)) →
  x ∈ Set.Icc (1/2 : ℝ) (5/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l80_8099


namespace NUMINAMATH_CALUDE_total_distance_biking_and_jogging_l80_8049

theorem total_distance_biking_and_jogging 
  (total_time : ℝ) 
  (biking_time : ℝ) 
  (biking_rate : ℝ) 
  (jogging_time : ℝ) 
  (jogging_rate : ℝ) 
  (h1 : total_time = 1.75) -- 1 hour and 45 minutes
  (h2 : biking_time = 1) -- 60 minutes in hours
  (h3 : biking_rate = 12)
  (h4 : jogging_time = 0.75) -- 45 minutes in hours
  (h5 : jogging_rate = 6) : 
  biking_rate * biking_time + jogging_rate * jogging_time = 16.5 := by
  sorry

#check total_distance_biking_and_jogging

end NUMINAMATH_CALUDE_total_distance_biking_and_jogging_l80_8049


namespace NUMINAMATH_CALUDE_line_circle_separate_l80_8003

theorem line_circle_separate (x₀ y₀ a : ℝ) (h1 : x₀^2 + y₀^2 < a^2) (h2 : a > 0) (h3 : (x₀, y₀) ≠ (0, 0)) :
  ∀ x y, x₀*x + y₀*y = a^2 → x^2 + y^2 ≠ a^2 :=
sorry

end NUMINAMATH_CALUDE_line_circle_separate_l80_8003


namespace NUMINAMATH_CALUDE_circle_chords_l80_8061

theorem circle_chords (n : ℕ) (h : n = 10) : 
  (n.choose 2 : ℕ) = 45 := by
  sorry

end NUMINAMATH_CALUDE_circle_chords_l80_8061


namespace NUMINAMATH_CALUDE_tan_beta_value_l80_8063

theorem tan_beta_value (α β : Real) 
  (h1 : Real.tan α = 1/3) 
  (h2 : Real.tan (α + β) = 1/2) : 
  Real.tan β = 1/7 := by
sorry

end NUMINAMATH_CALUDE_tan_beta_value_l80_8063


namespace NUMINAMATH_CALUDE_table_tennis_tournament_l80_8085

theorem table_tennis_tournament (n : ℕ) : 
  (∃ r : ℕ, r ≤ 3 ∧ (n^2 - 7*n - 76 + 2*r = 0) ∧ 
   (n - 3).choose 2 + 6 + r = 50) → 
  (∃! r : ℕ, r = 1 ∧ r ≤ 3 ∧ (n^2 - 7*n - 76 + 2*r = 0) ∧ 
   (n - 3).choose 2 + 6 + r = 50) :=
by sorry

end NUMINAMATH_CALUDE_table_tennis_tournament_l80_8085


namespace NUMINAMATH_CALUDE_four_to_fourth_sum_l80_8028

theorem four_to_fourth_sum : (4 : ℕ) ^ 4 + (4 : ℕ) ^ 4 + (4 : ℕ) ^ 4 + (4 : ℕ) ^ 4 = (4 : ℕ) ^ 5 := by
  sorry

end NUMINAMATH_CALUDE_four_to_fourth_sum_l80_8028


namespace NUMINAMATH_CALUDE_tim_sleep_hours_l80_8071

/-- The number of hours Tim sleeps each day for the first two days -/
def initial_sleep_hours : ℕ := 6

/-- The number of days Tim sleeps for the initial period -/
def initial_days : ℕ := 2

/-- The total number of days Tim sleeps -/
def total_days : ℕ := 4

/-- The total number of hours Tim sleeps over all days -/
def total_sleep_hours : ℕ := 32

/-- Theorem stating that Tim slept 10 hours each for the next 2 days -/
theorem tim_sleep_hours :
  (total_sleep_hours - initial_sleep_hours * initial_days) / (total_days - initial_days) = 10 :=
sorry

end NUMINAMATH_CALUDE_tim_sleep_hours_l80_8071


namespace NUMINAMATH_CALUDE_road_with_ten_trees_length_l80_8046

/-- The length of a road with trees planted at equal intervals -/
def road_length (num_trees : ℕ) (interval : ℝ) : ℝ :=
  (num_trees - 1 : ℝ) * interval

/-- Theorem: The length of a road with 10 trees planted at 10-meter intervals is 90 meters -/
theorem road_with_ten_trees_length :
  road_length 10 10 = 90 := by
  sorry

#eval road_length 10 10

end NUMINAMATH_CALUDE_road_with_ten_trees_length_l80_8046


namespace NUMINAMATH_CALUDE_mixed_number_sum_l80_8022

theorem mixed_number_sum : 
  (2 + 1/10) + (3 + 11/100) + (4 + 111/1000) = 9321/1000 := by sorry

end NUMINAMATH_CALUDE_mixed_number_sum_l80_8022


namespace NUMINAMATH_CALUDE_max_gcd_13n_plus_4_7n_plus_3_l80_8096

theorem max_gcd_13n_plus_4_7n_plus_3 :
  ∃ (k : ℕ+), ∀ (n : ℕ+), Nat.gcd (13 * n + 4) (7 * n + 3) ≤ k ∧
  ∃ (m : ℕ+), Nat.gcd (13 * m + 4) (7 * m + 3) = k ∧
  k = 11 :=
sorry

end NUMINAMATH_CALUDE_max_gcd_13n_plus_4_7n_plus_3_l80_8096


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l80_8097

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 1) :
  (1 / a + 1 / b) ≥ 4 + 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l80_8097


namespace NUMINAMATH_CALUDE_rectangular_plot_length_l80_8025

/-- Given a rectangular plot with the following properties:
  - The length is 20 meters more than the breadth
  - The cost of fencing is 26.50 per meter
  - The total cost of fencing is 5300
  Then the length of the plot is 60 meters. -/
theorem rectangular_plot_length (breadth : ℝ) (length : ℝ) : 
  length = breadth + 20 →
  2 * (length + breadth) * 26.5 = 5300 →
  length = 60 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_length_l80_8025


namespace NUMINAMATH_CALUDE_decoration_time_is_320_l80_8004

/-- Represents the time in minutes for a single step in nail decoration -/
def step_time : ℕ := 20

/-- Represents the time in minutes for pattern creation -/
def pattern_time : ℕ := 40

/-- Represents the number of coating steps (base, paint, glitter) -/
def num_coats : ℕ := 3

/-- Represents the number of people getting their nails decorated -/
def num_people : ℕ := 2

/-- Calculates the total time for nail decoration -/
def total_decoration_time : ℕ :=
  num_people * (2 * num_coats * step_time + pattern_time)

/-- Theorem stating that the total decoration time is 320 minutes -/
theorem decoration_time_is_320 :
  total_decoration_time = 320 :=
sorry

end NUMINAMATH_CALUDE_decoration_time_is_320_l80_8004


namespace NUMINAMATH_CALUDE_trombone_players_count_l80_8026

/-- Represents the Oprah Winfrey High School marching band -/
structure MarchingBand where
  trumpet_weight : ℕ := 5
  clarinet_weight : ℕ := 5
  trombone_weight : ℕ := 10
  tuba_weight : ℕ := 20
  drum_weight : ℕ := 15
  trumpet_count : ℕ := 6
  clarinet_count : ℕ := 9
  tuba_count : ℕ := 3
  drum_count : ℕ := 2
  total_weight : ℕ := 245

/-- Calculates the number of trombone players in the marching band -/
def trombone_players (band : MarchingBand) : ℕ :=
  let other_weight := band.trumpet_weight * band.trumpet_count +
                      band.clarinet_weight * band.clarinet_count +
                      band.tuba_weight * band.tuba_count +
                      band.drum_weight * band.drum_count
  let trombone_total_weight := band.total_weight - other_weight
  trombone_total_weight / band.trombone_weight

/-- Theorem stating that the number of trombone players is 8 -/
theorem trombone_players_count (band : MarchingBand) : trombone_players band = 8 := by
  sorry

end NUMINAMATH_CALUDE_trombone_players_count_l80_8026


namespace NUMINAMATH_CALUDE_negation_of_existential_proposition_l80_8000

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existential_proposition_l80_8000


namespace NUMINAMATH_CALUDE_initial_kittens_count_l80_8051

/-- The number of kittens Tim initially had -/
def initial_kittens : ℕ := 18

/-- The number of kittens Tim gave to Jessica -/
def kittens_to_jessica : ℕ := 3

/-- The number of kittens Tim gave to Sara -/
def kittens_to_sara : ℕ := 6

/-- The number of kittens Tim has left -/
def kittens_left : ℕ := 9

/-- Theorem stating that the initial number of kittens is equal to
    the sum of kittens given away and kittens left -/
theorem initial_kittens_count :
  initial_kittens = kittens_to_jessica + kittens_to_sara + kittens_left :=
by sorry

end NUMINAMATH_CALUDE_initial_kittens_count_l80_8051


namespace NUMINAMATH_CALUDE_theater_ticket_sales_l80_8024

theorem theater_ticket_sales (adult_price child_price : ℕ) 
  (total_tickets adult_tickets child_tickets : ℕ) : 
  adult_price = 12 → 
  child_price = 4 → 
  total_tickets = 130 → 
  adult_tickets = 90 → 
  child_tickets = 40 → 
  adult_price * adult_tickets + child_price * child_tickets = 1240 := by
  sorry

end NUMINAMATH_CALUDE_theater_ticket_sales_l80_8024


namespace NUMINAMATH_CALUDE_smallest_y_for_perfect_square_l80_8088

def x : ℕ := 11 * 36 * 54

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem smallest_y_for_perfect_square : 
  (∃ y : ℕ, y > 0 ∧ is_perfect_square (x * y)) ∧ 
  (∀ z : ℕ, z > 0 ∧ z < 66 → ¬is_perfect_square (x * z)) ∧
  is_perfect_square (x * 66) :=
sorry

end NUMINAMATH_CALUDE_smallest_y_for_perfect_square_l80_8088


namespace NUMINAMATH_CALUDE_even_number_property_l80_8059

/-- Sum of digits function -/
def sum_of_digits : ℕ → ℕ := sorry

/-- Theorem: If the sum of digits of N is 100 and the sum of digits of 5N is 50, then N is even -/
theorem even_number_property (N : ℕ) 
  (h1 : sum_of_digits N = 100) 
  (h2 : sum_of_digits (5 * N) = 50) : 
  Even N := by sorry

end NUMINAMATH_CALUDE_even_number_property_l80_8059


namespace NUMINAMATH_CALUDE_salary_calculation_l80_8001

/-- The monthly salary of a man who saves 20% of his salary and can save Rs. 230 when expenses increase by 20% -/
def monthlySalary : ℝ := 1437.5

theorem salary_calculation (savings_rate : ℝ) (expense_increase : ℝ) (reduced_savings : ℝ)
    (h1 : savings_rate = 0.20)
    (h2 : expense_increase = 0.20)
    (h3 : reduced_savings = 230)
    (h4 : savings_rate * monthlySalary - expense_increase * (savings_rate * monthlySalary) = reduced_savings) :
  monthlySalary = 1437.5 := by
  sorry

end NUMINAMATH_CALUDE_salary_calculation_l80_8001


namespace NUMINAMATH_CALUDE_inequality_proof_l80_8092

theorem inequality_proof (a b c d e f : ℕ) 
  (h1 : (a : ℚ) / b > (c : ℚ) / d)
  (h2 : (c : ℚ) / d > (e : ℚ) / f)
  (h3 : a * f - b * e = 1) :
  d ≥ b + f := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l80_8092


namespace NUMINAMATH_CALUDE_inequality_solution_range_l80_8067

theorem inequality_solution_range (a b x : ℝ) : 
  (a > 0 ∧ b > 0) → 
  (∀ a b, a > 0 → b > 0 → x^2 + 2*x < a/b + 16*b/a) ↔ 
  (-4 < x ∧ x < 2) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l80_8067


namespace NUMINAMATH_CALUDE_smallest_consecutive_even_sum_162_l80_8037

theorem smallest_consecutive_even_sum_162 (n : ℤ) : 
  (∃ (a b c : ℤ), a = n ∧ b = n + 2 ∧ c = n + 4 ∧ a + b + c = 162) → n = 52 := by
sorry

end NUMINAMATH_CALUDE_smallest_consecutive_even_sum_162_l80_8037


namespace NUMINAMATH_CALUDE_flour_calculation_l80_8053

/-- The amount of flour originally called for in the recipe -/
def original_flour : ℝ := 7

/-- The extra amount of flour Mary added -/
def extra_flour : ℝ := 2

/-- The total amount of flour Mary used -/
def total_flour : ℝ := 9

/-- Theorem stating that the original amount of flour plus the extra amount equals the total amount -/
theorem flour_calculation : original_flour + extra_flour = total_flour := by
  sorry

end NUMINAMATH_CALUDE_flour_calculation_l80_8053


namespace NUMINAMATH_CALUDE_half_of_1_01_l80_8055

theorem half_of_1_01 : (1.01 : ℝ) / 2 = 0.505 := by
  sorry

end NUMINAMATH_CALUDE_half_of_1_01_l80_8055


namespace NUMINAMATH_CALUDE_inequality_two_integer_solutions_l80_8036

def has_exactly_two_integer_solutions (a : ℝ) : Prop :=
  ∃ x y : ℤ, x ≠ y ∧
    (x : ℝ)^2 - (a + 1) * (x : ℝ) + a < 0 ∧
    (y : ℝ)^2 - (a + 1) * (y : ℝ) + a < 0 ∧
    ∀ z : ℤ, z ≠ x → z ≠ y → (z : ℝ)^2 - (a + 1) * (z : ℝ) + a ≥ 0

theorem inequality_two_integer_solutions :
  {a : ℝ | has_exactly_two_integer_solutions a} = {a : ℝ | (3 < a ∧ a ≤ 4) ∨ (-2 ≤ a ∧ a < -1)} :=
by sorry

end NUMINAMATH_CALUDE_inequality_two_integer_solutions_l80_8036


namespace NUMINAMATH_CALUDE_spinner_probability_l80_8093

def spinner_numbers : List ℕ := [4, 6, 7, 11, 12, 13, 17, 18]

def total_sections : ℕ := 8

def favorable_outcomes : ℕ := (spinner_numbers.filter (λ x => x > 10)).length

theorem spinner_probability : 
  (favorable_outcomes : ℚ) / total_sections = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_l80_8093


namespace NUMINAMATH_CALUDE_snow_probability_l80_8080

theorem snow_probability (p : ℝ) (n : ℕ) (h1 : p = 3/4) (h2 : n = 4) :
  1 - (1 - p)^n = 255/256 := by
  sorry

end NUMINAMATH_CALUDE_snow_probability_l80_8080


namespace NUMINAMATH_CALUDE_similar_triangle_perimeter_similar_triangle_perimeter_proof_l80_8006

/-- Given an isosceles triangle with two sides of 15 inches and one side of 8 inches,
    a similar triangle with the longest side of 45 inches has a perimeter of 114 inches. -/
theorem similar_triangle_perimeter : ℝ → ℝ → ℝ → ℝ → Prop :=
  fun original_long original_short similar_long perimeter =>
    original_long = 15 →
    original_short = 8 →
    similar_long = 45 →
    perimeter = similar_long + similar_long + (similar_long / original_long * original_short) →
    perimeter = 114

/-- Proof of the theorem -/
theorem similar_triangle_perimeter_proof :
  similar_triangle_perimeter 15 8 45 114 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangle_perimeter_similar_triangle_perimeter_proof_l80_8006


namespace NUMINAMATH_CALUDE_calculate_expression_l80_8072

theorem calculate_expression : -Real.sqrt 4 + |Real.sqrt 2 - 2| - (2023 : ℝ)^0 = -2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l80_8072


namespace NUMINAMATH_CALUDE_red_balls_count_l80_8009

theorem red_balls_count (total : ℕ) (white : ℕ) (green : ℕ) (yellow : ℕ) (purple : ℕ) (p : ℚ) :
  total = 100 →
  white = 20 →
  green = 30 →
  yellow = 10 →
  purple = 3 →
  p = 0.6 →
  p = (white + green + yellow : ℚ) / total →
  ∃ red : ℕ, red = 3 ∧ total = white + green + yellow + red + purple :=
by sorry

end NUMINAMATH_CALUDE_red_balls_count_l80_8009


namespace NUMINAMATH_CALUDE_sharks_winning_percentage_l80_8048

theorem sharks_winning_percentage (N : ℕ) : 
  (∀ k : ℕ, k < N → (1 + k : ℚ) / (4 + k) < 9 / 10) ∧
  (1 + N : ℚ) / (4 + N) ≥ 9 / 10 →
  N = 26 :=
sorry

end NUMINAMATH_CALUDE_sharks_winning_percentage_l80_8048


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l80_8029

theorem cone_lateral_surface_area 
  (r : ℝ) (l : ℝ) (h1 : r = 3) (h2 : l = 10) : 
  π * r * l = 30 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l80_8029


namespace NUMINAMATH_CALUDE_min_operations_to_identify_controllers_l80_8007

/-- The number of light bulbs and buttons -/
def n : ℕ := 64

/-- An operation consists of pressing a set of buttons and recording the on/off state of each light bulb -/
def Operation := Fin n → Bool

/-- A sequence of operations -/
def OperationSequence := List Operation

/-- The result of applying a sequence of operations to all light bulbs -/
def ApplyOperations (ops : OperationSequence) : Fin n → List Bool :=
  fun i => ops.map (fun op => op i)

/-- A mapping from light bulbs to their controlling buttons -/
def ControlMapping := Fin n → Fin n

theorem min_operations_to_identify_controllers :
  ∃ (k : ℕ), 
    (∃ (ops : OperationSequence), ops.length = k ∧
      (∀ (m : ControlMapping), Function.Injective m →
        Function.Injective (ApplyOperations ops ∘ m))) ∧
    (∀ (j : ℕ), j < k →
      ¬∃ (ops : OperationSequence), ops.length = j ∧
        (∀ (m : ControlMapping), Function.Injective m →
          Function.Injective (ApplyOperations ops ∘ m))) ∧
    k = 6 :=
  sorry

end NUMINAMATH_CALUDE_min_operations_to_identify_controllers_l80_8007


namespace NUMINAMATH_CALUDE_expression_evaluation_l80_8074

theorem expression_evaluation : 
  (Real.sqrt 3) / (Real.cos (10 * π / 180)) - 1 / (Real.sin (170 * π / 180)) = -4 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l80_8074


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l80_8034

/-- For an ellipse where the length of the major axis is twice its focal length, the eccentricity is 1/2. -/
theorem ellipse_eccentricity (a c : ℝ) (h : a = 2 * c) : c / a = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l80_8034


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l80_8075

def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}
def B : Set ℝ := {x : ℝ | -3 < x ∧ x ≤ 1}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -1 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l80_8075


namespace NUMINAMATH_CALUDE_lunch_slices_count_l80_8095

/-- The number of slices of pie served during lunch today -/
def lunch_slices : ℕ := sorry

/-- The total number of slices of pie served today -/
def total_slices : ℕ := 12

/-- The number of slices of pie served during dinner today -/
def dinner_slices : ℕ := 5

/-- Theorem stating that the number of slices served during lunch today is 7 -/
theorem lunch_slices_count : lunch_slices = total_slices - dinner_slices := by sorry

end NUMINAMATH_CALUDE_lunch_slices_count_l80_8095


namespace NUMINAMATH_CALUDE_invalid_votes_percentage_l80_8027

theorem invalid_votes_percentage
  (total_votes : ℕ)
  (candidate_a_percentage : ℚ)
  (candidate_a_votes : ℕ)
  (h1 : total_votes = 560000)
  (h2 : candidate_a_percentage = 80 / 100)
  (h3 : candidate_a_votes = 380800) :
  (total_votes - (candidate_a_votes / candidate_a_percentage)) / total_votes = 15 / 100 := by
sorry

end NUMINAMATH_CALUDE_invalid_votes_percentage_l80_8027


namespace NUMINAMATH_CALUDE_inequality_range_of_m_l80_8012

theorem inequality_range_of_m (m : ℝ) : 
  (∀ x : ℝ, (m^2 + 4*m - 5)*x^2 - 4*(m - 1)*x + 3 > 0) ↔ 
  (1 ≤ m ∧ m < 19) :=
sorry

end NUMINAMATH_CALUDE_inequality_range_of_m_l80_8012


namespace NUMINAMATH_CALUDE_range_of_3a_minus_b_l80_8014

theorem range_of_3a_minus_b (a b : ℝ) 
  (h1 : 2 ≤ a + b ∧ a + b ≤ 5) 
  (h2 : -2 ≤ a - b ∧ a - b ≤ 1) : 
  (∀ x, 3*a - b ≤ x → x ≤ 7) ∧ 
  (∀ y, -2 ≤ y → y ≤ 3*a - b) :=
by sorry

end NUMINAMATH_CALUDE_range_of_3a_minus_b_l80_8014


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l80_8015

theorem polynomial_division_remainder (k : ℚ) : 
  (∃! k, ∀ x, (3 * x^3 + k * x^2 + 5 * x - 8) % (3 * x + 4) = 10) ↔ k = 31/4 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l80_8015


namespace NUMINAMATH_CALUDE_unwashed_shirts_l80_8039

theorem unwashed_shirts 
  (short_sleeve : ℕ) 
  (long_sleeve : ℕ) 
  (washed : ℕ) 
  (h1 : short_sleeve = 9)
  (h2 : long_sleeve = 27)
  (h3 : washed = 20) : 
  short_sleeve + long_sleeve - washed = 16 := by
  sorry

end NUMINAMATH_CALUDE_unwashed_shirts_l80_8039


namespace NUMINAMATH_CALUDE_min_value_expression_l80_8052

theorem min_value_expression (x y : ℝ) : (x*y)^2 + (x + 7)^2 + (2*y + 7)^2 ≥ 45 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l80_8052


namespace NUMINAMATH_CALUDE_tangent_line_and_a_range_l80_8064

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + x + 1

-- Define the derivative of f(x)
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + 1

theorem tangent_line_and_a_range (a : ℝ) :
  -- Condition: Tangent line at (1, f(1)) is parallel to 2x - y + 1 = 0
  (f_derivative a 1 = 2) →
  -- Condition: f(x) is decreasing on the interval [-2/3, -1/3]
  (∀ x ∈ Set.Icc (-2/3) (-1/3), f_derivative a x ≤ 0) →
  -- Conclusion 1: Equations of tangent lines passing through (0, 1)
  ((∃ x₀ y₀ : ℝ, f a x₀ = y₀ ∧ f_derivative a x₀ = (y₀ - 1) / x₀ ∧
    ((y₀ = 1 ∧ x₀ = 0) ∨ (y₀ = 11/8 ∧ x₀ = 1/2))) ∧
   (∀ x y : ℝ, (y = x + 1) ∨ (y = 3/4 * x + 1))) ∧
  -- Conclusion 2: Range of a
  (a ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_and_a_range_l80_8064


namespace NUMINAMATH_CALUDE_count_non_negative_numbers_l80_8062

theorem count_non_negative_numbers : 
  let numbers : List ℝ := [-8, 2.1, 1/9, 3, 0, -2.5, 10, -1]
  (numbers.filter (λ x => x ≥ 0)).length = 5 := by
sorry

end NUMINAMATH_CALUDE_count_non_negative_numbers_l80_8062


namespace NUMINAMATH_CALUDE_rectangle_sides_l80_8081

theorem rectangle_sides (x y : ℚ) (h1 : 4 * x = 3 * y) (h2 : x * y = 2 * (x + y)) :
  x = 7 / 2 ∧ y = 14 / 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_sides_l80_8081


namespace NUMINAMATH_CALUDE_quotient_zeros_l80_8073

theorem quotient_zeros (x : ℚ) (h : x = 4227/1000) : 
  ¬ (∃ a b c : ℕ, x / 3 = (a : ℚ) + 1/10 + c/1000) :=
sorry

end NUMINAMATH_CALUDE_quotient_zeros_l80_8073


namespace NUMINAMATH_CALUDE_conditional_probability_haze_wind_l80_8043

theorem conditional_probability_haze_wind (P_haze P_wind P_both : ℝ) 
  (h1 : P_haze = 0.25)
  (h2 : P_wind = 0.4)
  (h3 : P_both = 0.02) :
  P_both / P_haze = 0.08 :=
by sorry

end NUMINAMATH_CALUDE_conditional_probability_haze_wind_l80_8043


namespace NUMINAMATH_CALUDE_bracket_removal_equality_l80_8047

theorem bracket_removal_equality (a b c : ℝ) : a - 2*(b - c) = a - 2*b + 2*c := by
  sorry

end NUMINAMATH_CALUDE_bracket_removal_equality_l80_8047


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l80_8082

theorem trigonometric_equation_solution :
  ∀ t : ℝ, 
    (2 * (Real.cos (2 * t))^6 - (Real.cos (2 * t))^4 + 1.5 * (Real.sin (4 * t))^2 - 3 * (Real.sin (2 * t))^2 = 0) ↔ 
    (∃ k : ℤ, t = (Real.pi / 8) * (2 * ↑k + 1)) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l80_8082


namespace NUMINAMATH_CALUDE_andrena_christel_doll_difference_l80_8035

/-- Proves that Andrena has 2 more dolls than Christel after gift exchanges -/
theorem andrena_christel_doll_difference :
  -- Initial conditions
  ∀ (debelyn_initial christel_initial andrena_initial : ℕ),
  debelyn_initial = 20 →
  christel_initial = 24 →
  -- Gift exchanges
  ∀ (debelyn_to_andrena christel_to_andrena : ℕ),
  debelyn_to_andrena = 2 →
  christel_to_andrena = 5 →
  -- Final condition
  andrena_initial + debelyn_to_andrena + christel_to_andrena =
    debelyn_initial - debelyn_to_andrena + 3 →
  -- Conclusion
  (andrena_initial + debelyn_to_andrena + christel_to_andrena) -
    (christel_initial - christel_to_andrena) = 2 :=
by sorry

end NUMINAMATH_CALUDE_andrena_christel_doll_difference_l80_8035


namespace NUMINAMATH_CALUDE_trig_simplification_l80_8013

theorem trig_simplification :
  (Real.sin (40 * π / 180) + Real.sin (80 * π / 180)) /
  (Real.cos (40 * π / 180) + Real.cos (80 * π / 180)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l80_8013


namespace NUMINAMATH_CALUDE_multiple_with_all_digits_l80_8050

/-- For any integer n, there exists a multiple m of n whose decimal representation
    contains each digit from 0 to 9 at least once. -/
theorem multiple_with_all_digits (n : ℤ) : ∃ m : ℤ,
  (n ∣ m) ∧ (∀ d : ℕ, d < 10 → ∃ k : ℕ, (m.natAbs / 10^k) % 10 = d) := by
  sorry

end NUMINAMATH_CALUDE_multiple_with_all_digits_l80_8050


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l80_8060

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_intersect : ∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ∧ y = 2*x) :
  let e := Real.sqrt (1 + (b/a)^2)
  e > Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l80_8060


namespace NUMINAMATH_CALUDE_same_school_probability_same_school_probability_proof_l80_8078

/-- The probability of selecting two teachers from the same school when randomly choosing
    two teachers out of three from School A and three from School B. -/
theorem same_school_probability : ℚ :=
  let total_teachers : ℕ := 6
  let teachers_per_school : ℕ := 3
  let selected_teachers : ℕ := 2

  2 / 5

/-- Proof that the probability of selecting two teachers from the same school is 2/5. -/
theorem same_school_probability_proof :
  same_school_probability = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_same_school_probability_same_school_probability_proof_l80_8078


namespace NUMINAMATH_CALUDE_only_isosceles_trapezoid_axially_not_centrally_symmetric_l80_8023

-- Define the set of geometric figures
inductive GeometricFigure
  | LineSegment
  | Square
  | Circle
  | IsoscelesTrapezoid
  | Parallelogram

-- Define axial symmetry
def is_axially_symmetric (figure : GeometricFigure) : Prop :=
  match figure with
  | GeometricFigure.LineSegment => true
  | GeometricFigure.Square => true
  | GeometricFigure.Circle => true
  | GeometricFigure.IsoscelesTrapezoid => true
  | GeometricFigure.Parallelogram => false

-- Define central symmetry
def is_centrally_symmetric (figure : GeometricFigure) : Prop :=
  match figure with
  | GeometricFigure.LineSegment => true
  | GeometricFigure.Square => true
  | GeometricFigure.Circle => true
  | GeometricFigure.IsoscelesTrapezoid => false
  | GeometricFigure.Parallelogram => true

-- Theorem stating that only the isosceles trapezoid satisfies the condition
theorem only_isosceles_trapezoid_axially_not_centrally_symmetric :
  ∀ (figure : GeometricFigure),
    (is_axially_symmetric figure ∧ ¬is_centrally_symmetric figure) ↔
    (figure = GeometricFigure.IsoscelesTrapezoid) :=
by
  sorry

end NUMINAMATH_CALUDE_only_isosceles_trapezoid_axially_not_centrally_symmetric_l80_8023


namespace NUMINAMATH_CALUDE_fishermans_red_snappers_l80_8077

/-- The number of Red snappers caught daily -/
def red_snappers : ℕ := sorry

/-- The number of Tunas caught daily -/
def tunas : ℕ := 14

/-- The price of a Red snapper in dollars -/
def red_snapper_price : ℕ := 3

/-- The price of a Tuna in dollars -/
def tuna_price : ℕ := 2

/-- The total daily earnings in dollars -/
def total_earnings : ℕ := 52

theorem fishermans_red_snappers :
  red_snappers * red_snapper_price + tunas * tuna_price = total_earnings ∧
  red_snappers = 8 := by sorry

end NUMINAMATH_CALUDE_fishermans_red_snappers_l80_8077


namespace NUMINAMATH_CALUDE_sugar_flour_ratio_l80_8090

theorem sugar_flour_ratio (flour baking_soda sugar : ℕ) : 
  (flour = 10 * baking_soda) →
  (flour = 8 * (baking_soda + 60)) →
  (sugar = 2000) →
  (sugar * 6 = flour * 5) :=
by sorry

end NUMINAMATH_CALUDE_sugar_flour_ratio_l80_8090


namespace NUMINAMATH_CALUDE_sawz_logging_total_cost_l80_8068

/-- The total cost of trees for Sawz Logging Co. -/
theorem sawz_logging_total_cost :
  let total_trees : ℕ := 850
  let douglas_fir_trees : ℕ := 350
  let ponderosa_pine_trees : ℕ := total_trees - douglas_fir_trees
  let douglas_fir_cost : ℕ := 300
  let ponderosa_pine_cost : ℕ := 225
  let total_cost : ℕ := douglas_fir_trees * douglas_fir_cost + ponderosa_pine_trees * ponderosa_pine_cost
  total_cost = 217500 := by
  sorry

#check sawz_logging_total_cost

end NUMINAMATH_CALUDE_sawz_logging_total_cost_l80_8068
