import Mathlib

namespace NUMINAMATH_CALUDE_temperature_function_and_max_l3125_312531

-- Define the temperature function
def T (a b c d : ℝ) (t : ℝ) : ℝ := a * t^3 + b * t^2 + c * t + d

-- Define the derivative of the temperature function
def T_prime (a b c : ℝ) (t : ℝ) : ℝ := 3 * a * t^2 + 2 * b * t + c

-- State the theorem
theorem temperature_function_and_max (a b c d : ℝ) 
  (ha : a ≠ 0)
  (h1 : T a b c d (-4) = 8)
  (h2 : T a b c d 0 = 60)
  (h3 : T a b c d 1 = 58)
  (h4 : T_prime a b c (-4) = T_prime a b c 4) :
  (∃ (t : ℝ), t ≥ -2 ∧ t ≤ 2 ∧ 
    (∀ (s : ℝ), s ≥ -2 ∧ s ≤ 2 → T 1 0 (-3) 60 t ≥ T 1 0 (-3) 60 s) ∧
    T 1 0 (-3) 60 t = 62) ∧
  (∀ (t : ℝ), T 1 0 (-3) 60 t = t^3 - 3*t + 60) := by
  sorry


end NUMINAMATH_CALUDE_temperature_function_and_max_l3125_312531


namespace NUMINAMATH_CALUDE_product_mod_seven_l3125_312581

theorem product_mod_seven : (2023 * 2024 * 2025 * 2026) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seven_l3125_312581


namespace NUMINAMATH_CALUDE_extreme_value_at_negative_three_l3125_312580

def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 3*x - 9

theorem extreme_value_at_negative_three (a : ℝ) : 
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x ≠ -3 ∧ |x + 3| < ε → f a x ≤ f a (-3)) →
  a = 5 := by sorry

end NUMINAMATH_CALUDE_extreme_value_at_negative_three_l3125_312580


namespace NUMINAMATH_CALUDE_factor_proof_l3125_312536

theorem factor_proof :
  (∃ n : ℤ, 28 = 4 * n) ∧ (∃ m : ℤ, 162 = 9 * m) := by sorry

end NUMINAMATH_CALUDE_factor_proof_l3125_312536


namespace NUMINAMATH_CALUDE_negative_of_negative_six_equals_six_l3125_312516

theorem negative_of_negative_six_equals_six : -(-6) = 6 := by
  sorry

end NUMINAMATH_CALUDE_negative_of_negative_six_equals_six_l3125_312516


namespace NUMINAMATH_CALUDE_hari_contribution_is_8280_l3125_312502

/-- Represents the business partnership between Praveen and Hari -/
structure Partnership where
  praveen_initial : ℕ
  praveen_months : ℕ
  hari_months : ℕ
  profit_ratio_praveen : ℕ
  profit_ratio_hari : ℕ

/-- Calculates Hari's contribution to the partnership -/
def hari_contribution (p : Partnership) : ℕ :=
  (p.praveen_initial * p.praveen_months * p.profit_ratio_hari) / (p.hari_months * p.profit_ratio_praveen)

/-- Theorem stating Hari's contribution in the given scenario -/
theorem hari_contribution_is_8280 :
  let p : Partnership := {
    praveen_initial := 3220,
    praveen_months := 12,
    hari_months := 7,
    profit_ratio_praveen := 2,
    profit_ratio_hari := 3
  }
  hari_contribution p = 8280 := by
  sorry

end NUMINAMATH_CALUDE_hari_contribution_is_8280_l3125_312502


namespace NUMINAMATH_CALUDE_museum_visitors_l3125_312568

theorem museum_visitors (V : ℕ) : 
  (130 : ℕ) + (3 * V / 4 : ℕ) = V → V = 520 :=
by sorry

end NUMINAMATH_CALUDE_museum_visitors_l3125_312568


namespace NUMINAMATH_CALUDE_orange_pyramid_sum_l3125_312550

/-- Calculates the number of oranges in a single layer of the pyramid -/
def oranges_in_layer (n : ℕ) : ℕ := n * n / 2

/-- Calculates the total number of oranges in the pyramid stack -/
def total_oranges (base_size : ℕ) : ℕ :=
  (List.range base_size).map oranges_in_layer |>.sum

/-- The theorem stating that a pyramid with base size 6 contains 44 oranges -/
theorem orange_pyramid_sum : total_oranges 6 = 44 := by
  sorry

#eval total_oranges 6

end NUMINAMATH_CALUDE_orange_pyramid_sum_l3125_312550


namespace NUMINAMATH_CALUDE_expression_value_l3125_312514

theorem expression_value (a b : ℤ) (h : a = b + 1) : 3 + 2*a - 2*b = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3125_312514


namespace NUMINAMATH_CALUDE_tangent_equality_implies_angle_l3125_312594

theorem tangent_equality_implies_angle (x : Real) : 
  0 < x → x < 180 →
  Real.tan ((150 - x) * π / 180) = 
    (Real.sin (150 * π / 180) - Real.sin (x * π / 180)) / 
    (Real.cos (150 * π / 180) - Real.cos (x * π / 180)) →
  x = 110 := by
sorry

end NUMINAMATH_CALUDE_tangent_equality_implies_angle_l3125_312594


namespace NUMINAMATH_CALUDE_factorization_identities_l3125_312592

theorem factorization_identities :
  (∀ x y : ℝ, x^4 - 16*y^4 = (x^2 + 4*y^2)*(x + 2*y)*(x - 2*y)) ∧
  (∀ a : ℝ, -2*a^3 + 12*a^2 - 16*a = -2*a*(a - 2)*(a - 4)) := by
sorry

end NUMINAMATH_CALUDE_factorization_identities_l3125_312592


namespace NUMINAMATH_CALUDE_charlie_score_l3125_312599

theorem charlie_score (team_total : ℕ) (num_players : ℕ) (others_average : ℕ) (h1 : team_total = 60) (h2 : num_players = 8) (h3 : others_average = 5) :
  team_total - (num_players - 1) * others_average = 25 := by
  sorry

end NUMINAMATH_CALUDE_charlie_score_l3125_312599


namespace NUMINAMATH_CALUDE_quadratic_roots_equal_integral_l3125_312505

/-- The roots of the quadratic equation 3x^2 - 6x + c = 0 are equal and integral when the discriminant is zero -/
theorem quadratic_roots_equal_integral (c : ℝ) :
  (∀ x : ℝ, 3 * x^2 - 6 * x + c = 0 ↔ x = 1) ∧ ((-6)^2 - 4 * 3 * c = 0) := by
  sorry


end NUMINAMATH_CALUDE_quadratic_roots_equal_integral_l3125_312505


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l3125_312542

theorem regular_polygon_sides (n : ℕ) : 
  n > 3 → 
  (n : ℚ) / (n * (n - 3) / 2 : ℚ) = 1/4 → 
  n = 11 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l3125_312542


namespace NUMINAMATH_CALUDE_waiter_customers_l3125_312572

theorem waiter_customers (non_tipping : ℕ) (tip_amount : ℕ) (total_tips : ℕ) : 
  non_tipping = 5 → tip_amount = 3 → total_tips = 15 → 
  non_tipping + (total_tips / tip_amount) = 10 :=
by sorry

end NUMINAMATH_CALUDE_waiter_customers_l3125_312572


namespace NUMINAMATH_CALUDE_probability_theorem_l3125_312526

/-- A regular dodecahedron -/
structure RegularDodecahedron where
  vertices : Finset (Fin 20)
  faces : Finset (Finset (Fin 5))
  num_faces : faces.card = 12

/-- Three distinct vertices of a regular dodecahedron -/
def ThreeVertices (d : RegularDodecahedron) := Finset (Fin 3)

/-- The probability that a plane determined by three randomly chosen distinct
    vertices of a regular dodecahedron contains points inside the dodecahedron -/
def probability_plane_intersects_interior (d : RegularDodecahedron) : ℚ :=
  1 - 1 / 9.5

/-- Theorem stating the probability of a plane determined by three randomly chosen
    distinct vertices of a regular dodecahedron containing points inside the dodecahedron -/
theorem probability_theorem (d : RegularDodecahedron) :
  probability_plane_intersects_interior d = 1 - 1 / 9.5 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l3125_312526


namespace NUMINAMATH_CALUDE_unique_intersection_characterization_l3125_312521

/-- A line that has only one common point (-1, -1) with the parabola y = 8x^2 + 10x + 1 -/
def uniqueIntersectionLine (f : ℝ → ℝ) : Prop :=
  (∃! p : ℝ × ℝ, p.1 = -1 ∧ p.2 = -1 ∧ p.2 = f p.1 ∧ p.2 = 8 * p.1^2 + 10 * p.1 + 1) ∧
  (∀ x : ℝ, f x = -6 * x - 7 ∨ (∀ y : ℝ, f y = -1))

/-- The theorem stating that a line has a unique intersection with the parabola
    if and only if it's either y = -6x - 7 or x = -1 -/
theorem unique_intersection_characterization :
  ∀ f : ℝ → ℝ, uniqueIntersectionLine f ↔ 
    (∀ x : ℝ, f x = -6 * x - 7) ∨ (∀ x : ℝ, f x = -1) :=
sorry

end NUMINAMATH_CALUDE_unique_intersection_characterization_l3125_312521


namespace NUMINAMATH_CALUDE_last_two_digits_of_product_l3125_312552

theorem last_two_digits_of_product (k : ℕ) (h : k ≥ 5) :
  ∃ m : ℕ, (k + 1) * (k + 2) * (k + 3) * (k + 4) ≡ 24 [ZMOD 100] :=
sorry

end NUMINAMATH_CALUDE_last_two_digits_of_product_l3125_312552


namespace NUMINAMATH_CALUDE_remainder_theorem_l3125_312510

theorem remainder_theorem (n : ℤ) (h : n % 9 = 4) : (5 * n - 12) % 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3125_312510


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3125_312561

def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_inequality_solution (a b c : ℝ) :
  (∀ x, a * x^2 + b * x + c > 0 ↔ x < -2 ∨ x > 4) →
  f a b c 2 < f a b c (-1) ∧ f a b c (-1) < f a b c 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3125_312561


namespace NUMINAMATH_CALUDE_inheritance_calculation_l3125_312500

-- Define the original inheritance amount
def original_inheritance : ℝ := 45500

-- Define the federal tax rate
def federal_tax_rate : ℝ := 0.25

-- Define the state tax rate
def state_tax_rate : ℝ := 0.15

-- Define the total tax paid
def total_tax_paid : ℝ := 16500

-- Theorem statement
theorem inheritance_calculation :
  let remaining_after_federal := original_inheritance * (1 - federal_tax_rate)
  let state_tax := remaining_after_federal * state_tax_rate
  let total_tax := original_inheritance * federal_tax_rate + state_tax
  total_tax = total_tax_paid :=
by sorry

end NUMINAMATH_CALUDE_inheritance_calculation_l3125_312500


namespace NUMINAMATH_CALUDE_james_fleet_capacity_l3125_312593

/-- Represents the fleet of gas transportation vans --/
structure Fleet :=
  (total_vans : ℕ)
  (large_vans : ℕ)
  (medium_vans : ℕ)
  (small_van : ℕ)
  (medium_capacity : ℕ)
  (small_capacity : ℕ)
  (large_capacity : ℕ)

/-- Calculates the total capacity of the fleet --/
def total_capacity (f : Fleet) : ℕ :=
  f.large_vans * f.medium_capacity +
  f.medium_vans * f.medium_capacity +
  f.small_van * f.small_capacity +
  (f.total_vans - f.large_vans - f.medium_vans - f.small_van) * f.large_capacity

/-- Theorem stating the total capacity of James' fleet --/
theorem james_fleet_capacity :
  ∃ (f : Fleet),
    f.total_vans = 6 ∧
    f.medium_vans = 2 ∧
    f.small_van = 1 ∧
    f.medium_capacity = 8000 ∧
    f.small_capacity = (7 * f.medium_capacity) / 10 ∧
    f.large_capacity = (3 * f.medium_capacity) / 2 ∧
    total_capacity f = 57600 :=
  sorry

end NUMINAMATH_CALUDE_james_fleet_capacity_l3125_312593


namespace NUMINAMATH_CALUDE_exist_abcd_equation_l3125_312596

theorem exist_abcd_equation : ∃ (a b c d : ℕ), 
  (a > 1 ∧ b > 1 ∧ c > 1 ∧ d > 1) ∧
  (1 / a + 1 / (a * b) + 1 / (a * b * c) + 1 / (a * b * c * d) : ℚ) = 37 / 48 ∧
  b = 4 := by sorry

end NUMINAMATH_CALUDE_exist_abcd_equation_l3125_312596


namespace NUMINAMATH_CALUDE_exists_right_triangles_form_consecutive_l3125_312527

/-- A right-angled triangle with integer side lengths -/
structure RightTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  right_angle : a^2 + b^2 = c^2

/-- A triangle with consecutive natural number side lengths -/
structure ConsecutiveTriangle where
  n : ℕ
  sides : Fin 3 → ℕ
  consecutive : sides = fun i => 2*n + i.val - 1

theorem exists_right_triangles_form_consecutive (A : ℕ) :
  ∃ (t : ConsecutiveTriangle) (rt1 rt2 : RightTriangle),
    t.sides 0 = rt1.a + rt2.a ∧
    t.sides 1 = rt1.b ∧
    t.sides 1 = rt2.b ∧
    t.sides 2 = rt1.c + rt2.c ∧
    A = (t.sides 0 * t.sides 1) / 2 :=
sorry

end NUMINAMATH_CALUDE_exists_right_triangles_form_consecutive_l3125_312527


namespace NUMINAMATH_CALUDE_largest_multiple_of_seven_as_sum_of_three_squares_l3125_312509

theorem largest_multiple_of_seven_as_sum_of_three_squares :
  ∃ n : ℕ, 
    (∃ a : ℕ, n = a^2 + (a+1)^2 + (a+2)^2) ∧ 
    7 ∣ n ∧
    n < 10000 ∧
    (∀ m : ℕ, (∃ b : ℕ, m = b^2 + (b+1)^2 + (b+2)^2) → 7 ∣ m → m < 10000 → m ≤ n) ∧
    n = 8750 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_seven_as_sum_of_three_squares_l3125_312509


namespace NUMINAMATH_CALUDE_socks_combination_l3125_312598

/-- The number of ways to choose k items from a set of n items, where order doesn't matter -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- There are 6 socks in the drawer -/
def total_socks : ℕ := 6

/-- We need to choose 4 socks -/
def socks_to_choose : ℕ := 4

/-- The number of ways to choose 4 socks from 6 socks is 15 -/
theorem socks_combination : choose total_socks socks_to_choose = 15 := by
  sorry

end NUMINAMATH_CALUDE_socks_combination_l3125_312598


namespace NUMINAMATH_CALUDE_fish_bucket_problem_l3125_312533

/-- Proves that the number of buckets is 9 given the conditions of the fish problem -/
theorem fish_bucket_problem (total_fish_per_bucket : ℕ) (mackerels_per_bucket : ℕ) (total_mackerels : ℕ)
  (h1 : total_fish_per_bucket = 9)
  (h2 : mackerels_per_bucket = 3)
  (h3 : total_mackerels = 27) :
  total_mackerels / mackerels_per_bucket = 9 := by
  sorry

end NUMINAMATH_CALUDE_fish_bucket_problem_l3125_312533


namespace NUMINAMATH_CALUDE_smallest_odd_n_l3125_312523

def is_smallest_odd (n : ℕ) : Prop :=
  Odd n ∧ 
  (3 : ℝ) ^ ((n + 1)^2 / 5) > 500 ∧ 
  ∀ m : ℕ, Odd m ∧ m < n → (3 : ℝ) ^ ((m + 1)^2 / 5) ≤ 500

theorem smallest_odd_n : is_smallest_odd 6 := by sorry

end NUMINAMATH_CALUDE_smallest_odd_n_l3125_312523


namespace NUMINAMATH_CALUDE_avocado_cost_l3125_312544

theorem avocado_cost (initial_amount : ℕ) (num_avocados : ℕ) (change : ℕ) : 
  initial_amount = 20 → num_avocados = 3 → change = 14 → 
  (initial_amount - change) / num_avocados = 2 := by
  sorry

end NUMINAMATH_CALUDE_avocado_cost_l3125_312544


namespace NUMINAMATH_CALUDE_sad_probability_value_l3125_312525

/-- Represents a person in the company -/
inductive Person : Type
| boy : Fin 3 → Person
| girl : Fin 3 → Person

/-- Represents the love relation between people -/
def loves : Person → Person → Prop := sorry

/-- The sad circumstance where no one is loved by the one they love -/
def sad_circumstance (loves : Person → Person → Prop) : Prop :=
  ∀ p : Person, ∃ q : Person, loves p q ∧ ¬loves q p

/-- The total number of possible love arrangements -/
def total_arrangements : ℕ := 729

/-- The number of sad arrangements -/
def sad_arrangements : ℕ := 156

/-- The probability of the sad circumstance -/
def sad_probability : ℚ := sad_arrangements / total_arrangements

theorem sad_probability_value : sad_probability = 156 / 729 :=
sorry

end NUMINAMATH_CALUDE_sad_probability_value_l3125_312525


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_equality_l3125_312573

/-- Arithmetic sequence {a_n} -/
def a (n : ℕ) : ℝ := 2*n + 2

/-- Geometric sequence {b_n} -/
def b (n : ℕ) : ℝ := 8 * 2^(n-2)

theorem arithmetic_geometric_sequence_equality :
  (a 1 + a 2 = 10) →
  (a 4 - a 3 = 2) →
  (b 2 = a 3) →
  (b 3 = a 7) →
  (a 15 = b 4) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_equality_l3125_312573


namespace NUMINAMATH_CALUDE_octal_subtraction_l3125_312535

def octal_to_decimal (n : ℕ) : ℕ := sorry

def decimal_to_octal (n : ℕ) : ℕ := sorry

theorem octal_subtraction :
  decimal_to_octal (octal_to_decimal 5374 - octal_to_decimal 2645) = 1527 := by sorry

end NUMINAMATH_CALUDE_octal_subtraction_l3125_312535


namespace NUMINAMATH_CALUDE_perfect_cube_property_l3125_312501

theorem perfect_cube_property (x y : ℕ+) (h : ∃ k : ℕ+, x * y^2 = k^3) :
  ∃ m : ℕ+, x^2 * y = m^3 := by
  sorry

end NUMINAMATH_CALUDE_perfect_cube_property_l3125_312501


namespace NUMINAMATH_CALUDE_diameter_endpoints_form_trapezoid_l3125_312522

-- Define the circles and their properties
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem setup
def problem_setup (c₁ c₂ : Circle) (d₁ d₂ : Set (ℝ × ℝ)) : Prop :=
  -- Circles are external to each other
  let (x₁, y₁) := c₁.center
  let (x₂, y₂) := c₂.center
  (x₁ - x₂)^2 + (y₁ - y₂)^2 > (c₁.radius + c₂.radius)^2 ∧
  -- d₁ and d₂ are diameters of c₁ and c₂ respectively
  (∀ p ∈ d₁, dist p c₁.center ≤ c₁.radius) ∧
  (∀ p ∈ d₂, dist p c₂.center ≤ c₂.radius) ∧
  -- The line through one diameter is tangent to the other circle
  (∃ p ∈ d₁, dist p c₂.center = c₂.radius) ∧
  (∃ p ∈ d₂, dist p c₁.center = c₁.radius)

-- Define a trapezoid
def is_trapezoid (quadrilateral : Set (ℝ × ℝ)) : Prop :=
  ∃ (a b c d : ℝ × ℝ), quadrilateral = {a, b, c, d} ∧
  (∃ (m : ℝ), (c.1 - d.1 = m * (a.1 - b.1) ∧ c.2 - d.2 = m * (a.2 - b.2)) ∨
              (b.1 - c.1 = m * (a.1 - d.1) ∧ b.2 - c.2 = m * (a.2 - d.2)))

-- Theorem statement
theorem diameter_endpoints_form_trapezoid (c₁ c₂ : Circle) (d₁ d₂ : Set (ℝ × ℝ)) :
  problem_setup c₁ c₂ d₁ d₂ →
  is_trapezoid (d₁ ∪ d₂) :=
sorry

end NUMINAMATH_CALUDE_diameter_endpoints_form_trapezoid_l3125_312522


namespace NUMINAMATH_CALUDE_cost_of_thousand_gum_l3125_312559

/-- The cost of a single piece of gum in cents -/
def cost_of_one_gum : ℕ := 1

/-- The number of pieces of gum -/
def num_gum : ℕ := 1000

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- The cost of multiple pieces of gum in dollars -/
def cost_in_dollars (n : ℕ) : ℚ :=
  (n * cost_of_one_gum : ℚ) / cents_per_dollar

theorem cost_of_thousand_gum :
  cost_in_dollars num_gum = 10 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_thousand_gum_l3125_312559


namespace NUMINAMATH_CALUDE_molecular_weight_proof_l3125_312589

/-- Given a compound where 7 moles have a total molecular weight of 854 grams,
    prove that the molecular weight of 1 mole is 122 grams/mole. -/
theorem molecular_weight_proof (total_weight : ℝ) (num_moles : ℝ) 
  (h1 : total_weight = 854)
  (h2 : num_moles = 7) :
  total_weight / num_moles = 122 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_proof_l3125_312589


namespace NUMINAMATH_CALUDE_three_possible_values_l3125_312588

def is_single_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def is_five_digit (n : ℕ) : Prop := n ≥ 10000 ∧ n ≤ 99999

def construct_number (a b : ℕ) : ℕ := a * 10000 + 3750 + b

theorem three_possible_values :
  ∃ (s : Finset ℕ), s.card = 3 ∧
    (∀ a ∈ s, is_single_digit a ∧
      (∃ b, is_single_digit b ∧
        is_five_digit (construct_number a b) ∧
        (construct_number a b) % 24 = 0)) ∧
    (∀ a, is_single_digit a →
      (∃ b, is_single_digit b ∧
        is_five_digit (construct_number a b) ∧
        (construct_number a b) % 24 = 0) →
      a ∈ s) :=
sorry

end NUMINAMATH_CALUDE_three_possible_values_l3125_312588


namespace NUMINAMATH_CALUDE_number_sum_15_equals_96_l3125_312583

theorem number_sum_15_equals_96 : ∃ x : ℝ, x + 15 = 96 ∧ x = 81 := by
  sorry

end NUMINAMATH_CALUDE_number_sum_15_equals_96_l3125_312583


namespace NUMINAMATH_CALUDE_peter_speed_proof_l3125_312534

/-- Peter's speed in miles per hour -/
def peter_speed : ℝ := 5

/-- Juan's speed in miles per hour -/
def juan_speed : ℝ := peter_speed + 3

/-- Time traveled in hours -/
def time : ℝ := 1.5

/-- Total distance between Juan and Peter after traveling -/
def total_distance : ℝ := 19.5

theorem peter_speed_proof :
  peter_speed * time + juan_speed * time = total_distance :=
by sorry

end NUMINAMATH_CALUDE_peter_speed_proof_l3125_312534


namespace NUMINAMATH_CALUDE_ab_greater_than_a_plus_b_l3125_312560

theorem ab_greater_than_a_plus_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a - b = a / b) : a * b > a + b := by
  sorry

end NUMINAMATH_CALUDE_ab_greater_than_a_plus_b_l3125_312560


namespace NUMINAMATH_CALUDE_evaluate_fraction_l3125_312556

theorem evaluate_fraction : (18 : ℝ) / (14 * 5.3) = 1.8 / 7.42 := by sorry

end NUMINAMATH_CALUDE_evaluate_fraction_l3125_312556


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3125_312547

/-- Arithmetic sequence with common difference d -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ) (d : ℝ) (m : ℕ)
  (h_arith : arithmetic_sequence a d)
  (h_d_neq_0 : d ≠ 0)
  (h_sum : a 3 + a 6 + a 10 + a 13 = 32)
  (h_am : a m = 8) :
  m = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3125_312547


namespace NUMINAMATH_CALUDE_painter_inventory_theorem_l3125_312537

/-- Represents the painter's paint inventory and room painting capacity -/
structure PainterInventory where
  initialRooms : ℕ
  remainingRooms : ℕ
  lostCans : ℕ

/-- Calculates the number of cans needed to paint a given number of rooms -/
def cansNeeded (inventory : PainterInventory) (rooms : ℕ) : ℕ :=
  (rooms * inventory.lostCans) / (inventory.initialRooms - inventory.remainingRooms)

/-- Theorem stating that under the given conditions, 15 cans are needed to paint 25 rooms -/
theorem painter_inventory_theorem (inventory : PainterInventory) 
  (h1 : inventory.initialRooms = 30)
  (h2 : inventory.remainingRooms = 25)
  (h3 : inventory.lostCans = 3) :
  cansNeeded inventory 25 = 15 := by
  sorry

end NUMINAMATH_CALUDE_painter_inventory_theorem_l3125_312537


namespace NUMINAMATH_CALUDE_product_of_powers_equals_product_of_consecutive_integers_l3125_312577

theorem product_of_powers_equals_product_of_consecutive_integers (k : ℕ) :
  (∃ (a b : ℕ), 2^a * 3^b = k * (k + 1)) ↔ k = 1 ∨ k = 2 ∨ k = 3 ∨ k = 8 := by
  sorry

end NUMINAMATH_CALUDE_product_of_powers_equals_product_of_consecutive_integers_l3125_312577


namespace NUMINAMATH_CALUDE_grain_warehouse_analysis_l3125_312590

def grain_records : List Int := [26, -32, -25, 34, -38, 10]
def current_stock : Int := 480

theorem grain_warehouse_analysis :
  (List.sum grain_records < 0) ∧
  (current_stock - List.sum grain_records = 505) := by
  sorry

end NUMINAMATH_CALUDE_grain_warehouse_analysis_l3125_312590


namespace NUMINAMATH_CALUDE_monotone_increasing_f_implies_a_range_l3125_312546

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 + 2*x - 2 * Real.log x

theorem monotone_increasing_f_implies_a_range (h : Monotone f) :
  (∀ x > 0, 2 * a ≤ x^2 + 2*x) → a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_monotone_increasing_f_implies_a_range_l3125_312546


namespace NUMINAMATH_CALUDE_sin_2x_value_l3125_312567

theorem sin_2x_value (x : ℝ) (h : Real.cos (π / 4 + x) = 3 / 5) : 
  Real.sin (2 * x) = 7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sin_2x_value_l3125_312567


namespace NUMINAMATH_CALUDE_volume_of_rotated_composite_shape_l3125_312566

/-- The volume of a solid formed by rotating a composite shape about the x-axis -/
theorem volume_of_rotated_composite_shape (π : ℝ) :
  let rectangle1_height : ℝ := 6
  let rectangle1_width : ℝ := 1
  let rectangle2_height : ℝ := 3
  let rectangle2_width : ℝ := 3
  let volume1 : ℝ := π * rectangle1_height^2 * rectangle1_width
  let volume2 : ℝ := π * rectangle2_height^2 * rectangle2_width
  let total_volume : ℝ := volume1 + volume2
  total_volume = 63 * π :=
by sorry

end NUMINAMATH_CALUDE_volume_of_rotated_composite_shape_l3125_312566


namespace NUMINAMATH_CALUDE_logical_equivalence_l3125_312507

theorem logical_equivalence (P Q R : Prop) :
  ((P ∧ R) → ¬Q) ↔ (Q → (¬P ∨ ¬R)) := by sorry

end NUMINAMATH_CALUDE_logical_equivalence_l3125_312507


namespace NUMINAMATH_CALUDE_point_on_transformed_plane_l3125_312506

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Applies a similarity transformation to a plane -/
def applySimilarity (p : Plane) (k : ℝ) : Plane :=
  { a := p.a, b := p.b, c := p.c, d := k * p.d }

/-- Checks if a point lies on a plane -/
def pointOnPlane (point : Point3D) (plane : Plane) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

theorem point_on_transformed_plane :
  let A : Point3D := ⟨1, 2, 2⟩
  let a : Plane := ⟨3, 0, -1, 5⟩
  let k : ℝ := -1/5
  let a' : Plane := applySimilarity a k
  pointOnPlane A a' := by sorry

end NUMINAMATH_CALUDE_point_on_transformed_plane_l3125_312506


namespace NUMINAMATH_CALUDE_total_unread_books_is_17_l3125_312532

/-- Represents a book series with total books and read books -/
structure BookSeries where
  total : ℕ
  read : ℕ

/-- Calculates the number of unread books in a series -/
def unread_books (series : BookSeries) : ℕ :=
  series.total - series.read

/-- The three book series -/
def series1 : BookSeries := ⟨14, 8⟩
def series2 : BookSeries := ⟨10, 5⟩
def series3 : BookSeries := ⟨18, 12⟩

/-- Theorem stating that the total number of unread books is 17 -/
theorem total_unread_books_is_17 :
  unread_books series1 + unread_books series2 + unread_books series3 = 17 := by
  sorry

end NUMINAMATH_CALUDE_total_unread_books_is_17_l3125_312532


namespace NUMINAMATH_CALUDE_smallest_number_greater_than_digit_sum_by_1755_l3125_312518

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem smallest_number_greater_than_digit_sum_by_1755 :
  (∀ m : ℕ, m < 1770 → m ≠ sum_of_digits m + 1755) ∧
  1770 = sum_of_digits 1770 + 1755 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_greater_than_digit_sum_by_1755_l3125_312518


namespace NUMINAMATH_CALUDE_nonAthleticParentsCount_l3125_312545

/-- Represents the number of students with various athletic parent combinations -/
structure AthleticParents where
  total : Nat
  athleticDad : Nat
  athleticMom : Nat
  bothAthletic : Nat

/-- Calculates the number of students with both non-athletic parents -/
def nonAthleticParents (ap : AthleticParents) : Nat :=
  ap.total - (ap.athleticDad + ap.athleticMom - ap.bothAthletic)

/-- Theorem stating that given the specific numbers in the problem, 
    the number of students with both non-athletic parents is 19 -/
theorem nonAthleticParentsCount : 
  let ap : AthleticParents := {
    total := 45,
    athleticDad := 17,
    athleticMom := 20,
    bothAthletic := 11
  }
  nonAthleticParents ap = 19 := by
  sorry

end NUMINAMATH_CALUDE_nonAthleticParentsCount_l3125_312545


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l3125_312553

theorem cubic_equation_solution (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hmn : m ≠ n) :
  ∃! (a b c : ℝ), ∀ x : ℝ, 
    (x + m)^3 - (x + n)^3 = (m + n)^3 ∧ x = a * m + b * n + c :=
by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l3125_312553


namespace NUMINAMATH_CALUDE_marcella_shoes_theorem_l3125_312557

/-- Given an initial number of shoe pairs and a number of individual shoes lost,
    calculate the maximum number of complete pairs remaining. -/
def max_remaining_pairs (initial_pairs : ℕ) (shoes_lost : ℕ) : ℕ :=
  initial_pairs - min initial_pairs shoes_lost

/-- Theorem stating that with 27 initial pairs and 9 individual shoes lost,
    the maximum number of complete pairs remaining is 18. -/
theorem marcella_shoes_theorem :
  max_remaining_pairs 27 9 = 18 := by
  sorry

end NUMINAMATH_CALUDE_marcella_shoes_theorem_l3125_312557


namespace NUMINAMATH_CALUDE_weight_difference_theorem_l3125_312584

-- Define the given conditions
def joe_weight : ℝ := 44
def initial_average : ℝ := 30
def new_average : ℝ := 31
def final_average : ℝ := 30

-- Define the number of students in the initial group
def initial_students : ℕ := 13

-- Define the theorem
theorem weight_difference_theorem :
  let total_weight_with_joe := initial_average * initial_students + joe_weight
  let remaining_students := initial_students + 1 - 2
  let final_total_weight := final_average * remaining_students
  let leaving_students_total_weight := total_weight_with_joe - final_total_weight
  let leaving_students_average_weight := leaving_students_total_weight / 2
  leaving_students_average_weight - joe_weight = -7 := by sorry

end NUMINAMATH_CALUDE_weight_difference_theorem_l3125_312584


namespace NUMINAMATH_CALUDE_max_sum_of_cubes_l3125_312515

/-- Given a system of equations, find the maximum value of x³ + y³ + z³ -/
theorem max_sum_of_cubes (x y z : ℝ) : 
  x^3 - x*y*z = 2 → 
  y^3 - x*y*z = 6 → 
  z^3 - x*y*z = 20 → 
  x^3 + y^3 + z^3 ≤ 151/7 := by
sorry

end NUMINAMATH_CALUDE_max_sum_of_cubes_l3125_312515


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3125_312570

def M : Set ℤ := {0, 1, 2, 3, 4}
def N : Set ℤ := {-2, 0, 2}

theorem intersection_of_M_and_N : M ∩ N = {0, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3125_312570


namespace NUMINAMATH_CALUDE_first_group_count_l3125_312519

theorem first_group_count (total_count : Nat) (total_avg : ℝ) (first_group_avg : ℝ) 
  (last_group_count : Nat) (last_group_avg : ℝ) (sixth_number : ℝ) : 
  total_count = 11 →
  total_avg = 10.7 →
  first_group_avg = 10.5 →
  last_group_count = 6 →
  last_group_avg = 11.4 →
  sixth_number = 13.700000000000017 →
  (total_count - last_group_count : ℝ) = 4 := by
sorry

end NUMINAMATH_CALUDE_first_group_count_l3125_312519


namespace NUMINAMATH_CALUDE_min_segments_on_cube_edges_l3125_312520

/-- A cube representation -/
structure Cube where
  vertices : Finset (Fin 8)
  edges : Finset (Fin 8 × Fin 8)

/-- A broken line on the surface of a cube -/
structure BrokenLine where
  segments : Finset (Fin 8 × Fin 8)
  num_segments : Nat
  is_closed : Bool
  vertices_on_cube : Bool

/-- Theorem statement -/
theorem min_segments_on_cube_edges (c : Cube) (bl : BrokenLine) :
  bl.num_segments = 8 ∧ bl.is_closed ∧ bl.vertices_on_cube →
  ∃ (coinciding_segments : Finset (Fin 8 × Fin 8)),
    coinciding_segments ⊆ c.edges ∧
    coinciding_segments ⊆ bl.segments ∧
    coinciding_segments.card = 2 ∧
    ∀ (cs : Finset (Fin 8 × Fin 8)),
      cs ⊆ c.edges ∧ cs ⊆ bl.segments →
      cs.card ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_segments_on_cube_edges_l3125_312520


namespace NUMINAMATH_CALUDE_sector_to_cone_sector_forms_cone_l3125_312563

theorem sector_to_cone (sector_angle : Real) (sector_radius : Real) 
  (base_radius : Real) (slant_height : Real) : Prop :=
  sector_angle = 240 ∧ 
  sector_radius = 12 ∧
  base_radius = 8 ∧
  slant_height = 12 ∧
  sector_angle / 360 * (2 * Real.pi * sector_radius) = 2 * Real.pi * base_radius ∧
  slant_height = sector_radius

theorem sector_forms_cone : 
  ∃ (sector_angle : Real) (sector_radius : Real) 
     (base_radius : Real) (slant_height : Real),
  sector_to_cone sector_angle sector_radius base_radius slant_height := by
  sorry

end NUMINAMATH_CALUDE_sector_to_cone_sector_forms_cone_l3125_312563


namespace NUMINAMATH_CALUDE_mouse_meiosis_observation_l3125_312585

/-- Available materials for mouse cell meiosis observation --/
inductive Material
  | MouseKidney
  | MouseTestis
  | MouseLiver
  | SudanIIIStain
  | GentianVioletSolution
  | JanusGreenBStain
  | DissociationFixativeSolution

/-- Types of cells produced during meiosis --/
inductive DaughterCell
  | Spermatogonial
  | SecondarySpermatocyte
  | Spermatid

/-- Theorem for correct mouse cell meiosis observation procedure --/
theorem mouse_meiosis_observation 
  (available_materials : List Material)
  (meiosis_occurs_in_gonads : Bool)
  (spermatogonial_cells_undergo_mitosis_and_meiosis : Bool) :
  (MouseTestis ∈ available_materials) →
  (DissociationFixativeSolution ∈ available_materials) →
  (GentianVioletSolution ∈ available_materials) →
  meiosis_occurs_in_gonads →
  spermatogonial_cells_undergo_mitosis_and_meiosis →
  (correct_tissue = MouseTestis) ∧
  (post_hypotonic_solution = DissociationFixativeSolution) ∧
  (staining_solution = GentianVioletSolution) ∧
  (daughter_cells = [DaughterCell.Spermatogonial, DaughterCell.SecondarySpermatocyte, DaughterCell.Spermatid]) := by
  sorry

end NUMINAMATH_CALUDE_mouse_meiosis_observation_l3125_312585


namespace NUMINAMATH_CALUDE_min_value_implies_a_l3125_312574

def f (x a : ℝ) : ℝ := |x + 1| + 2 * |x - a|

theorem min_value_implies_a (a : ℝ) : 
  (∀ x, f x a ≥ 5) ∧ (∃ x, f x a = 5) → a = -6 ∨ a = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_implies_a_l3125_312574


namespace NUMINAMATH_CALUDE_complex_modulus_l3125_312587

theorem complex_modulus (z : ℂ) (h : (z - 2) * (1 - Complex.I) = 1 + Complex.I) : 
  Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l3125_312587


namespace NUMINAMATH_CALUDE_positive_sum_of_squares_implies_inequality_l3125_312539

theorem positive_sum_of_squares_implies_inequality 
  (x y z : ℝ) 
  (hpos : x > 0 ∧ y > 0 ∧ z > 0) 
  (hsum : x^2 + y^2 + z^2 = 8) : 
  x^8 + y^8 + z^8 > 16 * Real.sqrt (2/3) := by
  sorry

end NUMINAMATH_CALUDE_positive_sum_of_squares_implies_inequality_l3125_312539


namespace NUMINAMATH_CALUDE_distance_is_13_l3125_312540

/-- The distance between two villages Yolkino and Palkino. -/
def distance_between_villages : ℝ := 13

/-- A point on the highway between Yolkino and Palkino. -/
structure HighwayPoint where
  distance_to_yolkino : ℝ
  distance_to_palkino : ℝ
  sum_is_13 : distance_to_yolkino + distance_to_palkino = 13

/-- The theorem stating that the distance between Yolkino and Palkino is 13 km. -/
theorem distance_is_13 : 
  ∀ (p : HighwayPoint), distance_between_villages = p.distance_to_yolkino + p.distance_to_palkino :=
by
  sorry

end NUMINAMATH_CALUDE_distance_is_13_l3125_312540


namespace NUMINAMATH_CALUDE_squirrel_nut_division_l3125_312562

theorem squirrel_nut_division (n : ℕ) : ¬(5 ∣ (2022 + n * (n + 1))) := by
  sorry

end NUMINAMATH_CALUDE_squirrel_nut_division_l3125_312562


namespace NUMINAMATH_CALUDE_tic_tac_toe_rounds_difference_l3125_312503

theorem tic_tac_toe_rounds_difference 
  (total_rounds : ℕ) 
  (william_wins : ℕ) 
  (h1 : total_rounds = 15) 
  (h2 : william_wins = 10) 
  (h3 : william_wins > total_rounds - william_wins) : 
  william_wins - (total_rounds - william_wins) = 5 := by
sorry

end NUMINAMATH_CALUDE_tic_tac_toe_rounds_difference_l3125_312503


namespace NUMINAMATH_CALUDE_probability_of_one_each_l3125_312530

def drawer_contents : ℕ := 7

def total_items : ℕ := 4 * drawer_contents

def ways_to_select_one_of_each : ℕ := drawer_contents^4

def total_selections : ℕ := (total_items.choose 4)

theorem probability_of_one_each : 
  (ways_to_select_one_of_each : ℚ) / total_selections = 2401 / 20475 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_one_each_l3125_312530


namespace NUMINAMATH_CALUDE_bike_ride_time_l3125_312548

theorem bike_ride_time (distance1 distance2 time1 : ℝ) (distance1_pos : 0 < distance1) (time1_pos : 0 < time1) :
  distance1 = 2 ∧ time1 = 6 ∧ distance2 = 5 →
  distance2 / (distance1 / time1) = 15 := by sorry

end NUMINAMATH_CALUDE_bike_ride_time_l3125_312548


namespace NUMINAMATH_CALUDE_davids_remaining_money_l3125_312564

theorem davids_remaining_money (initial_amount spent_amount remaining_amount : ℕ) :
  initial_amount = 1800 →
  remaining_amount = spent_amount - 800 →
  initial_amount - spent_amount = remaining_amount →
  remaining_amount = 500 := by
sorry

end NUMINAMATH_CALUDE_davids_remaining_money_l3125_312564


namespace NUMINAMATH_CALUDE_current_waiting_room_count_l3125_312565

/-- The number of people in the interview room -/
def interview_room_count : ℕ := 5

/-- The number of people currently in the waiting room -/
def waiting_room_count : ℕ := 22

/-- The condition that if three more people arrive in the waiting room,
    the number becomes five times the number of people in the interview room -/
axiom waiting_room_condition :
  waiting_room_count + 3 = 5 * interview_room_count

theorem current_waiting_room_count :
  waiting_room_count = 22 :=
sorry

end NUMINAMATH_CALUDE_current_waiting_room_count_l3125_312565


namespace NUMINAMATH_CALUDE_sock_pair_selection_l3125_312528

def total_socks : ℕ := 20
def white_socks : ℕ := 6
def brown_socks : ℕ := 7
def blue_socks : ℕ := 3
def red_socks : ℕ := 4

theorem sock_pair_selection :
  (Nat.choose white_socks 2) +
  (Nat.choose brown_socks 2) +
  (Nat.choose blue_socks 2) +
  (red_socks * white_socks) +
  (red_socks * brown_socks) +
  (red_socks * blue_socks) +
  (Nat.choose red_socks 2) = 109 := by
  sorry

end NUMINAMATH_CALUDE_sock_pair_selection_l3125_312528


namespace NUMINAMATH_CALUDE_max_rooks_on_chessboard_sixteen_rooks_achievable_l3125_312576

/-- Represents a chessboard with rooks --/
structure Chessboard :=
  (size : ℕ)
  (white_rooks : ℕ)
  (black_rooks : ℕ)

/-- Predicate to check if the rook placement is valid --/
def valid_placement (board : Chessboard) : Prop :=
  board.white_rooks ≤ board.size * 2 ∧ 
  board.black_rooks ≤ board.size * 2 ∧
  board.white_rooks = board.black_rooks

/-- Theorem stating the maximum number of rooks of each color --/
theorem max_rooks_on_chessboard :
  ∀ (board : Chessboard),
    board.size = 8 →
    valid_placement board →
    board.white_rooks ≤ 16 ∧
    board.black_rooks ≤ 16 :=
by sorry

/-- Theorem stating that 16 rooks of each color is achievable --/
theorem sixteen_rooks_achievable :
  ∃ (board : Chessboard),
    board.size = 8 ∧
    valid_placement board ∧
    board.white_rooks = 16 ∧
    board.black_rooks = 16 :=
by sorry

end NUMINAMATH_CALUDE_max_rooks_on_chessboard_sixteen_rooks_achievable_l3125_312576


namespace NUMINAMATH_CALUDE_internet_service_upgrade_l3125_312595

/-- Represents the internet service with speed and price -/
structure InternetService where
  speed : ℕ  -- Speed in Mbps
  price : ℕ  -- Price in dollars
  deriving Repr

/-- Calculates the yearly price difference between two services -/
def yearlyPriceDifference (s1 s2 : InternetService) : ℕ :=
  (s2.price - s1.price) * 12

/-- The problem statement -/
theorem internet_service_upgrade (current : InternetService)
    (upgrade20 upgrade30 : InternetService)
    (h1 : current.speed = 10 ∧ current.price = 20)
    (h2 : upgrade20.speed = 20 ∧ upgrade20.price = current.price + 10)
    (h3 : upgrade30.speed = 30)
    (h4 : yearlyPriceDifference upgrade20 upgrade30 = 120) :
    upgrade30.price / current.price = 2 := by
  sorry

end NUMINAMATH_CALUDE_internet_service_upgrade_l3125_312595


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_4519_l3125_312558

theorem largest_prime_factor_of_4519 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 4519 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 4519 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_4519_l3125_312558


namespace NUMINAMATH_CALUDE_translation_of_line_segment_l3125_312529

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a translation in 2D space -/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- Applies a translation to a point -/
def applyTranslation (p : Point) (t : Translation) : Point :=
  { x := p.x + t.dx, y := p.y + t.dy }

/-- Theorem: Translation of line segment AB to A'B' -/
theorem translation_of_line_segment (A B A' : Point) (t : Translation) :
  A.x = -2 ∧ A.y = 0 ∧
  B.x = 0 ∧ B.y = 3 ∧
  A'.x = 2 ∧ A'.y = 1 ∧
  A' = applyTranslation A t →
  applyTranslation B t = { x := 4, y := 4 } :=
by sorry

end NUMINAMATH_CALUDE_translation_of_line_segment_l3125_312529


namespace NUMINAMATH_CALUDE_ribbon_per_box_l3125_312578

theorem ribbon_per_box
  (total_ribbon : ℝ)
  (num_boxes : ℕ)
  (remaining_ribbon : ℝ)
  (h1 : total_ribbon = 4.5)
  (h2 : num_boxes = 5)
  (h3 : remaining_ribbon = 1)
  : (total_ribbon - remaining_ribbon) / num_boxes = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_ribbon_per_box_l3125_312578


namespace NUMINAMATH_CALUDE_max_tickets_after_scarf_l3125_312597

def ticket_cost : ℕ := 15
def initial_money : ℕ := 160
def scarf_cost : ℕ := 25

theorem max_tickets_after_scarf : 
  ∀ n : ℕ, n ≤ (initial_money - scarf_cost) / ticket_cost → n ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_max_tickets_after_scarf_l3125_312597


namespace NUMINAMATH_CALUDE_store_a_cheapest_l3125_312554

/-- Represents the cost calculation for purchasing soccer balls from different stores -/
def soccer_ball_cost (num_balls : ℕ) : ℕ → ℕ
| 0 => num_balls * 25 - (num_balls / 10) * 3 * 25  -- Store A
| 1 => num_balls * (25 - 5)                        -- Store B
| 2 => num_balls * 25 - ((num_balls * 25) / 200) * 40  -- Store C
| _ => 0  -- Invalid store

theorem store_a_cheapest :
  let num_balls : ℕ := 58
  soccer_ball_cost num_balls 0 < soccer_ball_cost num_balls 1 ∧
  soccer_ball_cost num_balls 0 < soccer_ball_cost num_balls 2 :=
by sorry

end NUMINAMATH_CALUDE_store_a_cheapest_l3125_312554


namespace NUMINAMATH_CALUDE_existence_of_divisible_n_l3125_312579

theorem existence_of_divisible_n : ∃ (n : ℕ), n > 0 ∧ (2009 * 2010 * 2011) ∣ ((n^2 - 5) * (n^2 + 6) * (n^2 + 30)) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_divisible_n_l3125_312579


namespace NUMINAMATH_CALUDE_room_length_calculation_l3125_312538

/-- Given a rectangular room with known width, paving cost per square meter, and total paving cost,
    calculate the length of the room. -/
theorem room_length_calculation (width : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ)
    (h1 : width = 4)
    (h2 : cost_per_sqm = 800)
    (h3 : total_cost = 17600) :
    total_cost / (width * cost_per_sqm) = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_room_length_calculation_l3125_312538


namespace NUMINAMATH_CALUDE_chair_count_sequence_l3125_312517

theorem chair_count_sequence (a : ℕ → ℕ) 
  (h1 : a 1 = 14)
  (h2 : a 2 = 23)
  (h3 : a 3 = 32)
  (h5 : a 5 = 50)
  (h6 : a 6 = 59)
  (h_arithmetic : ∀ n : ℕ, n ≥ 1 → a (n + 1) - a n = a 2 - a 1) :
  a 4 = 41 := by
  sorry

end NUMINAMATH_CALUDE_chair_count_sequence_l3125_312517


namespace NUMINAMATH_CALUDE_parabola_vertex_l3125_312541

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := 2 * (x + 9)^2 - 3

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (-9, -3)

/-- Theorem: The vertex of the parabola y = 2(x+9)^2 - 3 is at the point (-9, -3) -/
theorem parabola_vertex : 
  (∀ x : ℝ, parabola x ≥ parabola (vertex.1)) ∧ 
  parabola (vertex.1) = vertex.2 := by
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3125_312541


namespace NUMINAMATH_CALUDE_rotation_symmetry_l3125_312508

-- Define the directions
inductive Direction
  | Up
  | Down
  | Left
  | Right

-- Define a square configuration
def SquareConfig := List Direction

-- Define a rotation function
def rotate90Clockwise (config : SquareConfig) : SquareConfig :=
  match config with
  | [a, b, c, d] => [d, a, b, c]
  | _ => []  -- Return empty list for invalid configurations

-- Theorem statement
theorem rotation_symmetry (original : SquareConfig) :
  original = [Direction.Up, Direction.Right, Direction.Down, Direction.Left] →
  rotate90Clockwise original = [Direction.Right, Direction.Down, Direction.Left, Direction.Up] :=
by
  sorry


end NUMINAMATH_CALUDE_rotation_symmetry_l3125_312508


namespace NUMINAMATH_CALUDE_coefficient_equals_49_l3125_312586

/-- The coefficient of x^3y^5 in the expansion of (x+2y)(x-y)^7 -/
def coefficient : ℤ :=
  2 * (Nat.choose 7 4) - (Nat.choose 7 5)

/-- Theorem stating that the coefficient of x^3y^5 in the expansion of (x+2y)(x-y)^7 is 49 -/
theorem coefficient_equals_49 : coefficient = 49 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_equals_49_l3125_312586


namespace NUMINAMATH_CALUDE_bill_donut_order_combinations_l3125_312582

/-- The number of ways to distribute remaining donuts after ensuring at least one of each kind -/
def donut_combinations (total_donuts : ℕ) (donut_kinds : ℕ) (remaining_donuts : ℕ) : ℕ :=
  Nat.choose (remaining_donuts + donut_kinds - 1) (donut_kinds - 1)

/-- Theorem stating the number of combinations for Bill's donut order -/
theorem bill_donut_order_combinations :
  donut_combinations 8 5 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_bill_donut_order_combinations_l3125_312582


namespace NUMINAMATH_CALUDE_percentage_of_students_with_glasses_l3125_312569

def total_students : ℕ := 325
def students_without_glasses : ℕ := 195

theorem percentage_of_students_with_glasses :
  (((total_students - students_without_glasses) : ℚ) / total_students) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_students_with_glasses_l3125_312569


namespace NUMINAMATH_CALUDE_distinct_solutions_count_l3125_312543

-- Define the function g
def g (x : ℝ) : ℝ := x^2 - 6*x + 5

-- State the theorem
theorem distinct_solutions_count :
  ∃! (s : Finset ℝ), (∀ d ∈ s, g (g (g (g d))) = 5) ∧ s.card = 6 := by
  sorry

end NUMINAMATH_CALUDE_distinct_solutions_count_l3125_312543


namespace NUMINAMATH_CALUDE_expression_value_l3125_312551

theorem expression_value :
  ∀ (a b c d : ℤ),
    (∀ n : ℤ, n < 0 → a ≥ n) →  -- a is the largest negative integer
    (a < 0) →                   -- ensure a is negative
    (b = -c) →                  -- b and c are opposite numbers
    (d < 0) →                   -- d is negative
    (abs d = 2) →               -- absolute value of d is 2
    4*a + (b + c) - abs (3*d) = -10 :=
by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3125_312551


namespace NUMINAMATH_CALUDE_new_circle_equation_l3125_312575

/-- Given a circle with equation x^2 + 2x + y^2 = 0, prove that a new circle 
    with the same center and radius 2 has the equation (x+1)^2 + y^2 = 4 -/
theorem new_circle_equation (x y : ℝ) : 
  (∀ x y, x^2 + 2*x + y^2 = 0 → ∃ h k, (x - h)^2 + (y - k)^2 = 1) →
  (∀ x y, (x + 1)^2 + y^2 = 4 ↔ (x - (-1))^2 + (y - 0)^2 = 2^2) :=
by sorry

end NUMINAMATH_CALUDE_new_circle_equation_l3125_312575


namespace NUMINAMATH_CALUDE_monotonic_function_a_range_l3125_312571

/-- A function f is monotonic on an interval [a,b] if it is either
    non-decreasing or non-increasing on that interval. -/
def IsMonotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y) ∨
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f y ≤ f x)

/-- The main theorem stating the range of 'a' for which the given function
    is monotonic on the interval [1,3]. -/
theorem monotonic_function_a_range :
  ∀ a : ℝ,
  (IsMonotonic (fun x => (1/3) * x^3 + a * x^2 + 5 * x + 6) 1 3) →
  (a ≤ -3 ∨ a ≥ -Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_monotonic_function_a_range_l3125_312571


namespace NUMINAMATH_CALUDE_triangle_heights_l3125_312512

theorem triangle_heights (ha hb : ℝ) (d : ℕ) :
  ha = 3 →
  hb = 7 →
  (∃ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a * ha = b * hb ∧
    b * hb = c * d ∧
    a * ha = c * d ∧
    a + b > c ∧ a + c > b ∧ b + c > a) →
  d = 3 ∨ d = 4 ∨ d = 5 :=
by sorry

end NUMINAMATH_CALUDE_triangle_heights_l3125_312512


namespace NUMINAMATH_CALUDE_max_value_of_fraction_l3125_312504

theorem max_value_of_fraction (x y z u v : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hu : u > 0) (hv : v > 0) :
  (x*y + y*z + z*u + u*v) / (2*x^2 + y^2 + 2*z^2 + u^2 + 2*v^2) ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_fraction_l3125_312504


namespace NUMINAMATH_CALUDE_minimize_sum_of_distances_l3125_312549

/-- The point that minimizes the sum of distances to two fixed points lies on the line connecting the first point and the reflection of the second point across the y-axis. -/
theorem minimize_sum_of_distances (A B C : ℝ × ℝ) (h1 : A = (3, 3)) (h2 : B = (-1, -1)) (h3 : C.1 = -3) :
  (∃ k : ℝ, C = (-3, k) ∧ 
    (∀ k' : ℝ, dist A C + dist B C ≤ dist A (C.1, k') + dist B (C.1, k'))) →
  C.2 = -9 :=
by sorry


end NUMINAMATH_CALUDE_minimize_sum_of_distances_l3125_312549


namespace NUMINAMATH_CALUDE_coefficient_of_x_cubed_is_94_l3125_312511

-- Define the polynomials
def p (x : ℝ) : ℝ := 3 * x^3 + 4 * x^2 + 5 * x + 6
def q (x : ℝ) : ℝ := 7 * x^2 + 8 * x + 9

-- Theorem statement
theorem coefficient_of_x_cubed_is_94 :
  ∃ a b c d e : ℝ, p * q = (λ x => a * x^5 + b * x^4 + 94 * x^3 + c * x^2 + d * x + e) :=
sorry

end NUMINAMATH_CALUDE_coefficient_of_x_cubed_is_94_l3125_312511


namespace NUMINAMATH_CALUDE_b_over_a_is_sqrt_2_angle_B_is_45_degrees_l3125_312524

noncomputable section

variable (A B C : ℝ)
variable (a b c : ℝ)

-- Define the triangle ABC
axiom triangle_abc : a > 0 ∧ b > 0 ∧ c > 0

-- Define the relationship between sides and angles
axiom sine_law : a / Real.sin A = b / Real.sin B

-- Given conditions
axiom condition1 : a * Real.sin A * Real.sin B + b * Real.cos A ^ 2 = Real.sqrt 2 * a
axiom condition2 : c ^ 2 = b ^ 2 + Real.sqrt 3 * a ^ 2

-- Theorems to prove
theorem b_over_a_is_sqrt_2 : b / a = Real.sqrt 2 := by sorry

theorem angle_B_is_45_degrees : B = Real.pi / 4 := by sorry

end NUMINAMATH_CALUDE_b_over_a_is_sqrt_2_angle_B_is_45_degrees_l3125_312524


namespace NUMINAMATH_CALUDE_square_binomial_constant_l3125_312591

theorem square_binomial_constant (c : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 100*x + c = (x + a)^2) → c = 2500 :=
by sorry

end NUMINAMATH_CALUDE_square_binomial_constant_l3125_312591


namespace NUMINAMATH_CALUDE_regression_and_range_correct_l3125_312555

/-- Represents a data point with protein content and production cost -/
structure DataPoint where
  x : Float  -- protein content
  y : Float  -- production cost

/-- The set of given data points -/
def dataPoints : List DataPoint := [
  ⟨0, 19⟩, ⟨0.69, 32⟩, ⟨1.39, 40⟩, ⟨1.79, 44⟩, ⟨2.40, 52⟩, ⟨2.56, 53⟩, ⟨2.94, 54⟩
]

/-- The mean of x values -/
def xMean : Float := 1.68

/-- The mean of y values -/
def yMean : Float := 42

/-- The sum of squared differences between x values and their mean -/
def sumSquaredDiffX : Float := 6.79

/-- The sum of the product of differences between x values and their mean,
    and y values and their mean -/
def sumProductDiff : Float := 81.41

/-- Calculates the slope of the regression line -/
def calculateSlope (sumProductDiff sumSquaredDiffX : Float) : Float :=
  sumProductDiff / sumSquaredDiffX

/-- Calculates the y-intercept of the regression line -/
def calculateIntercept (slope xMean yMean : Float) : Float :=
  yMean - slope * xMean

/-- The regression equation -/
def regressionEquation (x : Float) (slope intercept : Float) : Float :=
  slope * x + intercept

/-- Theorem stating the correctness of the regression equation and protein content range -/
theorem regression_and_range_correct :
  let slope := calculateSlope sumProductDiff sumSquaredDiffX
  let intercept := calculateIntercept slope xMean yMean
  (∀ x, regressionEquation x slope intercept = 11.99 * x + 21.86) ∧
  (∀ y, 60 ≤ y ∧ y ≤ 70 → 
    3.18 ≤ (y - intercept) / slope ∧ (y - intercept) / slope ≤ 4.02) := by
  sorry

end NUMINAMATH_CALUDE_regression_and_range_correct_l3125_312555


namespace NUMINAMATH_CALUDE_pool_filling_cost_l3125_312513

/-- Proves that the cost to fill a pool is $5 given the specified conditions -/
theorem pool_filling_cost (fill_time : ℕ) (flow_rate : ℕ) (water_cost : ℚ) : 
  fill_time = 50 → 
  flow_rate = 100 → 
  water_cost = 1 / 1000 → 
  (fill_time * flow_rate : ℚ) * water_cost = 5 := by
  sorry

#check pool_filling_cost

end NUMINAMATH_CALUDE_pool_filling_cost_l3125_312513
