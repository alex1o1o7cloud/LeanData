import Mathlib

namespace NUMINAMATH_CALUDE_least_common_multiple_2_3_4_5_6_sixty_divisible_by_2_3_4_5_6_least_number_of_marbles_l2951_295108

theorem least_common_multiple_2_3_4_5_6 : ∀ n : ℕ, n > 0 → (2 ∣ n) ∧ (3 ∣ n) ∧ (4 ∣ n) ∧ (5 ∣ n) ∧ (6 ∣ n) → n ≥ 60 := by
  sorry

theorem sixty_divisible_by_2_3_4_5_6 : (2 ∣ 60) ∧ (3 ∣ 60) ∧ (4 ∣ 60) ∧ (5 ∣ 60) ∧ (6 ∣ 60) := by
  sorry

theorem least_number_of_marbles : ∃! n : ℕ, n > 0 ∧ (2 ∣ n) ∧ (3 ∣ n) ∧ (4 ∣ n) ∧ (5 ∣ n) ∧ (6 ∣ n) ∧ ∀ m : ℕ, m > 0 → (2 ∣ m) ∧ (3 ∣ m) ∧ (4 ∣ m) ∧ (5 ∣ m) ∧ (6 ∣ m) → n ≤ m := by
  sorry

end NUMINAMATH_CALUDE_least_common_multiple_2_3_4_5_6_sixty_divisible_by_2_3_4_5_6_least_number_of_marbles_l2951_295108


namespace NUMINAMATH_CALUDE_michael_purchase_l2951_295171

/-- The amount Michael paid for his purchases after a discount -/
def amountPaid (suitCost shoesCost discount : ℕ) : ℕ :=
  suitCost + shoesCost - discount

/-- Theorem stating the correct amount Michael paid -/
theorem michael_purchase : amountPaid 430 190 100 = 520 := by
  sorry

end NUMINAMATH_CALUDE_michael_purchase_l2951_295171


namespace NUMINAMATH_CALUDE_last_k_digits_theorem_l2951_295174

theorem last_k_digits_theorem (k : ℕ) (h : k ≥ 2) :
  (∃ n : ℕ+, (10^(10^n.val) : ℤ) ≡ 9^(9^n.val) [ZMOD 10^k]) ↔ k ∈ ({2, 3, 4} : Finset ℕ) :=
sorry

end NUMINAMATH_CALUDE_last_k_digits_theorem_l2951_295174


namespace NUMINAMATH_CALUDE_value_of_a_l2951_295137

theorem value_of_a (a : ℝ) : 
  let A : Set ℝ := {a + 2, (a + 1)^2, a^2 + 3*a + 3}
  1 ∈ A → a = -1 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_l2951_295137


namespace NUMINAMATH_CALUDE_tangent_line_parallel_l2951_295156

/-- Given a function f(x) = ln x - ax, if its derivative at x = 1 is -2, then a = 3 -/
theorem tangent_line_parallel (a : ℝ) : 
  let f : ℝ → ℝ := λ x => Real.log x - a * x
  (deriv f 1 = -2) → a = 3 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_parallel_l2951_295156


namespace NUMINAMATH_CALUDE_count_valid_triples_l2951_295134

def is_valid_triple (a b c : ℕ) : Prop :=
  2 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ a * b * c = 2 * (a * b + b * c + c * a)

theorem count_valid_triples :
  ∃! n : ℕ, ∃ S : Finset (ℕ × ℕ × ℕ),
    S.card = n ∧
    (∀ t ∈ S, is_valid_triple t.1 t.2.1 t.2.2) ∧
    (∀ a b c : ℕ, is_valid_triple a b c → (a, b, c) ∈ S) ∧
    n = 5 :=
sorry

end NUMINAMATH_CALUDE_count_valid_triples_l2951_295134


namespace NUMINAMATH_CALUDE_item_sale_ratio_l2951_295193

theorem item_sale_ratio (c x y : ℝ) (hx : x = 0.85 * c) (hy : y = 1.15 * c) :
  y / x = 23 / 17 := by
  sorry

end NUMINAMATH_CALUDE_item_sale_ratio_l2951_295193


namespace NUMINAMATH_CALUDE_mini_croissant_cost_gala_luncheon_cost_l2951_295158

/-- Calculates the cost of mini croissants for a committee luncheon --/
theorem mini_croissant_cost (people : ℕ) (sandwiches_per_person : ℕ) 
  (croissants_per_pack : ℕ) (pack_price : ℚ) : ℕ → ℚ :=
  λ total_croissants =>
    let packs_needed := (total_croissants + croissants_per_pack - 1) / croissants_per_pack
    packs_needed * pack_price

/-- Proves that the cost of mini croissants for the committee luncheon is $32.00 --/
theorem gala_luncheon_cost : 
  mini_croissant_cost 24 2 12 8 (24 * 2) = 32 := by
  sorry

end NUMINAMATH_CALUDE_mini_croissant_cost_gala_luncheon_cost_l2951_295158


namespace NUMINAMATH_CALUDE_parallelogram_base_l2951_295159

theorem parallelogram_base (area height : ℝ) (h1 : area = 704) (h2 : height = 22) :
  area / height = 32 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_base_l2951_295159


namespace NUMINAMATH_CALUDE_mountaineer_arrangements_l2951_295183

theorem mountaineer_arrangements (total : ℕ) (familiar : ℕ) (groups : ℕ) (familiar_per_group : ℕ) :
  total = 10 →
  familiar = 4 →
  groups = 2 →
  familiar_per_group = 2 →
  (familiar.choose familiar_per_group) * ((total - familiar).choose familiar_per_group) * groups = 120 :=
by sorry

end NUMINAMATH_CALUDE_mountaineer_arrangements_l2951_295183


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2951_295115

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + (a-1)*x + 1 < 0) → a ∈ Set.Ici 3 ∪ Set.Iic (-1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2951_295115


namespace NUMINAMATH_CALUDE_jersey_tshirt_cost_difference_l2951_295178

/-- The cost of a jersey in dollars -/
def jersey_cost : ℕ := 115

/-- The cost of a t-shirt in dollars -/
def tshirt_cost : ℕ := 25

/-- The number of t-shirts sold during the game -/
def tshirts_sold : ℕ := 113

/-- The number of jerseys sold during the game -/
def jerseys_sold : ℕ := 78

theorem jersey_tshirt_cost_difference : jersey_cost - tshirt_cost = 90 := by
  sorry

end NUMINAMATH_CALUDE_jersey_tshirt_cost_difference_l2951_295178


namespace NUMINAMATH_CALUDE_regular_star_points_l2951_295150

/-- An n-pointed regular star -/
structure RegularStar (n : ℕ) :=
  (edge_length : ℝ)
  (angle_A : ℝ)
  (angle_B : ℝ)
  (edge_congruent : edge_length > 0)
  (angle_A_congruent : angle_A > 0)
  (angle_B_congruent : angle_B > 0)
  (angle_difference : angle_B = angle_A + 10)
  (exterior_angle_sum : n * (angle_A + angle_B) = 360)

/-- The number of points in a regular star satisfying the given conditions is 36 -/
theorem regular_star_points : ∃ (n : ℕ), n > 0 ∧ ∃ (star : RegularStar n), n = 36 :=
sorry

end NUMINAMATH_CALUDE_regular_star_points_l2951_295150


namespace NUMINAMATH_CALUDE_painting_survey_l2951_295147

theorem painting_survey (total : ℕ) (not_enjoy_not_understand : ℕ) (enjoy : ℕ) (understand : ℕ) :
  total = 440 →
  not_enjoy_not_understand = 110 →
  enjoy = understand →
  (enjoy : ℚ) / total = 3 / 8 :=
by
  sorry

end NUMINAMATH_CALUDE_painting_survey_l2951_295147


namespace NUMINAMATH_CALUDE_sally_buttons_count_l2951_295176

/-- The number of buttons Sally needs for all shirts -/
def total_buttons (monday_shirts tuesday_shirts wednesday_shirts buttons_per_shirt : ℕ) : ℕ :=
  (monday_shirts + tuesday_shirts + wednesday_shirts) * buttons_per_shirt

/-- Theorem stating that Sally needs 45 buttons for all shirts -/
theorem sally_buttons_count : total_buttons 4 3 2 5 = 45 := by
  sorry

end NUMINAMATH_CALUDE_sally_buttons_count_l2951_295176


namespace NUMINAMATH_CALUDE_irrational_sum_rational_irrational_l2951_295175

theorem irrational_sum_rational_irrational (π : ℝ) (h : Irrational π) : Irrational (5 + π) := by
  sorry

end NUMINAMATH_CALUDE_irrational_sum_rational_irrational_l2951_295175


namespace NUMINAMATH_CALUDE_expand_expression_l2951_295143

theorem expand_expression (x : ℝ) : (x + 3) * (4 * x - 8) = 4 * x^2 + 4 * x - 24 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2951_295143


namespace NUMINAMATH_CALUDE_x_range_theorem_l2951_295103

theorem x_range_theorem (x : ℝ) : 
  (∀ a ∈ Set.Ioo 0 1, (a - 3) * x^2 < (4 * a - 2) * x) ↔ 
  (x ≤ -1 ∨ x ≥ 2/3) := by
sorry

end NUMINAMATH_CALUDE_x_range_theorem_l2951_295103


namespace NUMINAMATH_CALUDE_blocks_remaining_problem_l2951_295110

/-- Given a person with an initial number of blocks and a number of blocks used,
    calculate the remaining number of blocks. -/
def remaining_blocks (initial : ℕ) (used : ℕ) : ℕ :=
  initial - used

/-- Theorem stating that given 78 initial blocks and 19 used blocks,
    the remaining number of blocks is 59. -/
theorem blocks_remaining_problem :
  remaining_blocks 78 19 = 59 := by
  sorry

end NUMINAMATH_CALUDE_blocks_remaining_problem_l2951_295110


namespace NUMINAMATH_CALUDE_ellipsoid_to_hemisphere_radius_l2951_295138

/-- Given an ellipsoid that collapses into a hemisphere while conserving volume,
    this theorem proves that the major radius of the original ellipsoid is 8∛3 cm
    when the minor radius is 4∛3 cm and the major-to-minor radius ratio is 2. -/
theorem ellipsoid_to_hemisphere_radius 
  (b : ℝ) -- minor radius of the ellipsoid
  (a : ℝ) -- major radius of the ellipsoid
  (r : ℝ) -- radius of the hemisphere
  (h1 : b = 4 * Real.rpow 3 (1/3)) -- minor radius is 4∛3 cm
  (h2 : a = 2 * b) -- major radius is twice the minor radius
  (h3 : (4/3) * Real.pi * a * b^2 = (2/3) * Real.pi * r^3) -- volume conservation
  : a = 8 * Real.rpow 3 (1/3) := by
  sorry


end NUMINAMATH_CALUDE_ellipsoid_to_hemisphere_radius_l2951_295138


namespace NUMINAMATH_CALUDE_two_digit_special_number_l2951_295142

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def digit_sum (n : ℕ) : ℕ := 
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

def digit_product (n : ℕ) : ℕ := 
  if n < 10 then n else (n % 10) * digit_product (n / 10)

theorem two_digit_special_number (a b : ℕ) : 
  1 ≤ a ∧ a ≤ 9 ∧ b = 9 →
  10 * a + b = digit_product (10 * a + b) + digit_sum (10 * a + b) →
  is_prime (digit_sum (10 * a + b)) →
  a = 2 ∨ a = 4 ∨ a = 8 := by sorry

end NUMINAMATH_CALUDE_two_digit_special_number_l2951_295142


namespace NUMINAMATH_CALUDE_lucy_shells_found_l2951_295127

theorem lucy_shells_found (initial_shells final_shells : ℝ) 
  (h1 : initial_shells = 68.3)
  (h2 : final_shells = 89.5) :
  final_shells - initial_shells = 21.2 := by
  sorry

end NUMINAMATH_CALUDE_lucy_shells_found_l2951_295127


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l2951_295114

theorem quadratic_root_problem (k : ℤ) (b c : ℤ) (h1 : k > 9) 
  (h2 : k^2 - b*k + c = 0) (h3 : b = 2*k + 1) 
  (h4 : (k-7)^2 - b*(k-7) + c = 0) : c = 3*k := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l2951_295114


namespace NUMINAMATH_CALUDE_cloth_sale_calculation_l2951_295105

/-- Represents the number of meters of cloth sold -/
def meters_sold : ℕ := 30

/-- The total selling price in Rupees -/
def total_selling_price : ℕ := 4500

/-- The profit per meter in Rupees -/
def profit_per_meter : ℕ := 10

/-- The cost price per meter in Rupees -/
def cost_price_per_meter : ℕ := 140

/-- Theorem stating that the number of meters sold is correct given the conditions -/
theorem cloth_sale_calculation :
  meters_sold * (cost_price_per_meter + profit_per_meter) = total_selling_price :=
by sorry

end NUMINAMATH_CALUDE_cloth_sale_calculation_l2951_295105


namespace NUMINAMATH_CALUDE_monotonic_at_most_one_zero_l2951_295102

/-- A function f: ℝ → ℝ is monotonic if for all x₁ < x₂, either f(x₁) ≤ f(x₂) or f(x₁) ≥ f(x₂) -/
def Monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → (f x₁ ≤ f x₂ ∨ f x₁ ≥ f x₂)

/-- A real number x is a zero of f if f(x) = 0 -/
def IsZero (f : ℝ → ℝ) (x : ℝ) : Prop :=
  f x = 0

/-- The number of zeros of f is at most one -/
def AtMostOneZero (f : ℝ → ℝ) : Prop :=
  ∀ x y, IsZero f x → IsZero f y → x = y

theorem monotonic_at_most_one_zero (f : ℝ → ℝ) (h : Monotonic f) : AtMostOneZero f := by
  sorry

end NUMINAMATH_CALUDE_monotonic_at_most_one_zero_l2951_295102


namespace NUMINAMATH_CALUDE_election_winner_votes_l2951_295188

theorem election_winner_votes (total_votes : ℕ) (winner_percentage : ℚ) (vote_difference : ℕ) : 
  winner_percentage = 54/100 →
  vote_difference = 288 →
  ↑total_votes * winner_percentage - ↑total_votes * (1 - winner_percentage) = vote_difference →
  ↑total_votes * winner_percentage = 1944 :=
by sorry

end NUMINAMATH_CALUDE_election_winner_votes_l2951_295188


namespace NUMINAMATH_CALUDE_inequality_proof_l2951_295168

theorem inequality_proof (a : ℝ) (h : a > 3) : 4 / (a - 3) + a ≥ 7 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2951_295168


namespace NUMINAMATH_CALUDE_ratio_xy_system_l2951_295196

theorem ratio_xy_system (x y t : ℝ) 
  (eq1 : 2 * x + 5 * y = 6 * t) 
  (eq2 : 3 * x - y = t) : 
  x / y = 11 / 16 := by
sorry

end NUMINAMATH_CALUDE_ratio_xy_system_l2951_295196


namespace NUMINAMATH_CALUDE_compare_function_values_l2951_295157

/-- Given a quadratic function f(x) = x^2 - bx + c with specific properties,
    prove that f(b^x) ≤ f(c^x) for all real x. -/
theorem compare_function_values (b c : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = x^2 - b*x + c) 
    (h2 : ∀ x, f (1 - x) = f (1 + x)) (h3 : f 0 = 3) : 
    ∀ x, f (b^x) ≤ f (c^x) := by
  sorry

end NUMINAMATH_CALUDE_compare_function_values_l2951_295157


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2951_295152

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x + 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = 3^5 - 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2951_295152


namespace NUMINAMATH_CALUDE_cube_sum_simplification_l2951_295177

theorem cube_sum_simplification (a b c : ℕ) (ha : a = 43) (hb : b = 26) (hc : c = 17) :
  (a^3 + c^3) / (a^3 + b^3) = (a + c) / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_simplification_l2951_295177


namespace NUMINAMATH_CALUDE_equation_solutions_l2951_295160

def is_solution (X Y Z : ℕ) : Prop :=
  X^Y + Y^Z = X * Y * Z

theorem equation_solutions :
  ∀ X Y Z : ℕ,
    is_solution X Y Z ↔
      (X = 1 ∧ Y = 1 ∧ Z = 2) ∨
      (X = 2 ∧ Y = 2 ∧ Z = 2) ∨
      (X = 2 ∧ Y = 2 ∧ Z = 3) ∨
      (X = 4 ∧ Y = 2 ∧ Z = 3) ∨
      (X = 4 ∧ Y = 2 ∧ Z = 4) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2951_295160


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2951_295186

theorem simplify_and_evaluate (y : ℝ) :
  let x : ℝ := -4
  ((x + y)^2 - y * (2 * x + y) - 8 * x) / (2 * x) = -6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2951_295186


namespace NUMINAMATH_CALUDE_unique_intersection_point_l2951_295164

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 9*x^2 + 27*x - 14

-- State the theorem
theorem unique_intersection_point :
  ∃! p : ℝ × ℝ, 
    (∀ x, f x = (p.2) ↔ x = p.1) ∧ 
    p.1 = p.2 ∧ 
    p = (2, 2) := by
  sorry

end NUMINAMATH_CALUDE_unique_intersection_point_l2951_295164


namespace NUMINAMATH_CALUDE_ascending_order_abab_l2951_295198

theorem ascending_order_abab (a b : ℝ) (ha : a > 0) (hb : b < 0) (hab : a + b > 0) :
  -a < b ∧ b < -b ∧ -b < a := by sorry

end NUMINAMATH_CALUDE_ascending_order_abab_l2951_295198


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l2951_295144

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (hn : n = 5264) (hd : d = 17) :
  ∃ (k : ℕ), k ≤ d - 1 ∧ (n - k) % d = 0 ∧ ∀ (m : ℕ), m < k → (n - m) % d ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l2951_295144


namespace NUMINAMATH_CALUDE_amelia_win_probability_l2951_295180

/-- The probability of Amelia's coin landing heads -/
def p_amelia : ℚ := 3/7

/-- The probability of Blaine's coin landing heads -/
def p_blaine : ℚ := 1/4

/-- The probability of Amelia winning the game -/
def p_amelia_wins : ℚ := 9/14

/-- The game described in the problem -/
def coin_game (p_a p_b : ℚ) : ℚ :=
  let p_amelia_first := p_a * (1 - p_b)
  let p_blaine_first := (1 - p_a) * p_b
  let p_both_tails := (1 - p_a) * (1 - p_b)
  let p_amelia_alternate := p_both_tails * (p_a / (1 - (1 - p_a) * (1 - p_b)))
  p_amelia_first + p_amelia_alternate

theorem amelia_win_probability :
  coin_game p_amelia p_blaine = p_amelia_wins :=
sorry

end NUMINAMATH_CALUDE_amelia_win_probability_l2951_295180


namespace NUMINAMATH_CALUDE_square_sum_of_specific_conditions_l2951_295141

theorem square_sum_of_specific_conditions (x y : ℕ+) 
  (h1 : x * y + x + y = 110)
  (h2 : x^2 * y + x * y^2 = 1540) : 
  x^2 + y^2 = 620 := by
sorry

end NUMINAMATH_CALUDE_square_sum_of_specific_conditions_l2951_295141


namespace NUMINAMATH_CALUDE_problem_statement_l2951_295184

theorem problem_statement (a b c d : ℕ+) 
  (h1 : a^3 = b^2) 
  (h2 : c^4 = d^3) 
  (h3 : c - a = 31) : 
  d - b = 229 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2951_295184


namespace NUMINAMATH_CALUDE_consecutive_primes_expression_l2951_295146

theorem consecutive_primes_expression (p q : ℕ) : 
  Prime p → Prime q → p < q → p.succ = q → (p : ℚ) / q = 4 / 5 → 
  25 / 7 + ((2 * q - p) : ℚ) / (2 * q + p) = 4 := by
sorry

end NUMINAMATH_CALUDE_consecutive_primes_expression_l2951_295146


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l2951_295172

theorem quadratic_root_relation (p r : ℝ) (hr : r > 0) :
  (∃ x y : ℝ, x^2 + p*x + r = 0 ∧ y^2 + p*y + r = 0 ∧ y = 2*x) →
  p = Real.sqrt (9*r/2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l2951_295172


namespace NUMINAMATH_CALUDE_cylinder_height_equals_sphere_surface_area_l2951_295154

/-- Given a sphere of radius 6 cm and a right circular cylinder with equal height and diameter,
    if their surface areas are equal, then the height of the cylinder is 6√2 cm. -/
theorem cylinder_height_equals_sphere_surface_area (r h : ℝ) : 
  r = 6 →  -- radius of sphere is 6 cm
  h = 2 * r →  -- height of cylinder equals its diameter
  4 * Real.pi * r^2 = 2 * Real.pi * r * h →  -- surface areas are equal
  h = 6 * Real.sqrt 2 := by
  sorry

#check cylinder_height_equals_sphere_surface_area

end NUMINAMATH_CALUDE_cylinder_height_equals_sphere_surface_area_l2951_295154


namespace NUMINAMATH_CALUDE_election_winner_percentage_l2951_295121

theorem election_winner_percentage (total_votes : ℕ) (majority : ℕ) : 
  total_votes = 480 → majority = 192 → 
  ∃ (winner_percentage : ℚ), 
    winner_percentage = 70 / 100 ∧ 
    (winner_percentage * total_votes : ℚ) - ((1 - winner_percentage) * total_votes : ℚ) = majority :=
by sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l2951_295121


namespace NUMINAMATH_CALUDE_nina_total_problems_l2951_295116

/-- Given the homework assignments for Ruby and the relative amounts for Nina,
    calculate the total number of problems Nina has to complete. -/
theorem nina_total_problems (ruby_math ruby_reading ruby_science : ℕ)
  (nina_math_factor nina_reading_factor nina_science_factor : ℕ) :
  ruby_math = 12 →
  ruby_reading = 4 →
  ruby_science = 5 →
  nina_math_factor = 5 →
  nina_reading_factor = 9 →
  nina_science_factor = 3 →
  ruby_math * nina_math_factor +
  ruby_reading * nina_reading_factor +
  ruby_science * nina_science_factor = 111 := by
sorry

end NUMINAMATH_CALUDE_nina_total_problems_l2951_295116


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2951_295191

theorem quadratic_inequality (a b c : ℝ) (ha : a ≠ 0) :
  (∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ 2 * a * x^2 + b * x + c = 0) →
  min c (a + c + 1) ≤ max (abs (b - a + 1)) (abs (b + a - 1)) ∧
  (min c (a + c + 1) = max (abs (b - a + 1)) (abs (b + a - 1)) ↔
    ((a = 1 ∧ b = 0 ∧ c = 0) ∨ (a ≤ -1 ∧ 2 * a - abs b + c = 0))) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2951_295191


namespace NUMINAMATH_CALUDE_student_percentage_theorem_l2951_295112

theorem student_percentage_theorem (total : ℝ) (third_year_percent : ℝ) (second_year_fraction : ℝ)
  (h1 : third_year_percent = 50)
  (h2 : second_year_fraction = 2/3)
  (h3 : total > 0) :
  let non_third_year := total - (third_year_percent / 100) * total
  let second_year := second_year_fraction * non_third_year
  (total - second_year) / total * 100 = 66.66666666666667 :=
sorry

end NUMINAMATH_CALUDE_student_percentage_theorem_l2951_295112


namespace NUMINAMATH_CALUDE_obtuse_triangle_consecutive_sides_l2951_295163

/-- An obtuse triangle with consecutive natural number side lengths has sides 2, 3, and 4 -/
theorem obtuse_triangle_consecutive_sides : 
  ∀ (a b c : ℕ), 
  (a < b) ∧ (b < c) ∧  -- consecutive
  (c = a + 2) ∧        -- consecutive
  (c^2 > a^2 + b^2) →  -- obtuse (by law of cosines)
  a = 2 ∧ b = 3 ∧ c = 4 := by
sorry

end NUMINAMATH_CALUDE_obtuse_triangle_consecutive_sides_l2951_295163


namespace NUMINAMATH_CALUDE_solve_potato_problem_l2951_295119

def potato_problem (total_potatoes : ℕ) (time_per_potato : ℕ) (remaining_time : ℕ) : Prop :=
  let remaining_potatoes := remaining_time / time_per_potato
  let cooked_potatoes := total_potatoes - remaining_potatoes
  cooked_potatoes = total_potatoes - (remaining_time / time_per_potato)

theorem solve_potato_problem :
  potato_problem 12 6 36 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_potato_problem_l2951_295119


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l2951_295107

theorem sum_of_x_and_y (x y m : ℝ) 
  (eq1 : x + m = 4) 
  (eq2 : y - 3 = m) : 
  x + y = 7 := by
sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l2951_295107


namespace NUMINAMATH_CALUDE_inscribed_triangle_sum_l2951_295139

/-- An equilateral triangle inscribed in an ellipse -/
structure InscribedTriangle where
  /-- The x-coordinate of a vertex of the triangle -/
  x : ℝ
  /-- The y-coordinate of a vertex of the triangle -/
  y : ℝ
  /-- The condition that the vertex lies on the ellipse -/
  on_ellipse : x^2 + 9*y^2 = 9
  /-- The condition that one vertex is at (0, 1) -/
  vertex_at_origin : x = 0 ∧ y = 1
  /-- The condition that one altitude is aligned with the y-axis -/
  altitude_aligned : True  -- This condition is implicitly satisfied by the symmetry of the problem

/-- The theorem stating the result about the inscribed equilateral triangle -/
theorem inscribed_triangle_sum (t : InscribedTriangle) 
  (p q : ℕ) (h_coprime : Nat.Coprime p q) 
  (h_side_length : (12 * Real.sqrt 3 / 13)^2 = p / q) : 
  p + q = 601 := by sorry

end NUMINAMATH_CALUDE_inscribed_triangle_sum_l2951_295139


namespace NUMINAMATH_CALUDE_smallest_k_for_15_digit_period_l2951_295122

/-- Represents a positive rational number with a decimal representation having a period of 30 digits -/
def RationalWith30DigitPeriod : Type := { q : ℚ // q > 0 ∧ ∃ m : ℕ, q = m / (10^30 - 1) }

/-- The theorem statement -/
theorem smallest_k_for_15_digit_period 
  (a b : RationalWith30DigitPeriod)
  (h_diff : ∃ p : ℤ, (a.val - b.val : ℚ) = p / (10^15 - 1)) :
  (∃ k : ℕ, k > 0 ∧ ∃ q : ℤ, (a.val + k * b.val : ℚ) = q / (10^15 - 1)) ∧
  (∀ k : ℕ, k > 0 → k < 6 → ¬∃ q : ℤ, (a.val + k * b.val : ℚ) = q / (10^15 - 1)) :=
sorry

end NUMINAMATH_CALUDE_smallest_k_for_15_digit_period_l2951_295122


namespace NUMINAMATH_CALUDE_childrens_book_balances_weights_l2951_295199

/-- Represents a two-arm scale with items on both sides -/
structure TwoArmScale where
  left_side : ℝ
  right_side : ℝ

/-- Checks if the scale is balanced -/
def is_balanced (scale : TwoArmScale) : Prop :=
  scale.left_side = scale.right_side

/-- The weight of the children's book -/
def childrens_book_weight : ℝ := 1.1

/-- The combined weight of the weights on the right side of the scale -/
def right_side_weight : ℝ := 0.5 + 0.3 + 0.3

/-- Theorem stating that the children's book weight balances the given weights -/
theorem childrens_book_balances_weights :
  is_balanced { left_side := childrens_book_weight, right_side := right_side_weight } :=
by sorry

end NUMINAMATH_CALUDE_childrens_book_balances_weights_l2951_295199


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l2951_295135

theorem fractional_equation_solution :
  ∃ (x : ℚ), (x ≠ 0 ∧ x ≠ 3) → (3 / (x^2 - 3*x) + (x - 1) / (x - 3) = 1) ∧ x = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l2951_295135


namespace NUMINAMATH_CALUDE_power_division_equality_l2951_295129

theorem power_division_equality : (3 : ℕ)^12 / (9 : ℕ)^2 = 6561 := by sorry

end NUMINAMATH_CALUDE_power_division_equality_l2951_295129


namespace NUMINAMATH_CALUDE_rational_number_ordering_l2951_295170

theorem rational_number_ordering : -3^2 < -(1/3) ∧ -(1/3) < (-3)^2 ∧ (-3)^2 = |-3^2| := by
  sorry

end NUMINAMATH_CALUDE_rational_number_ordering_l2951_295170


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l2951_295189

/-- A hyperbola with equation x^2/4 - y^2/16 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2/4 - y^2/16 = 1

/-- Asymptotic lines with equations y = ±2x -/
def asymptotic_lines (x y : ℝ) : Prop := y = 2*x ∨ y = -2*x

/-- Theorem stating that the given hyperbola equation implies the asymptotic lines,
    but the asymptotic lines do not necessarily imply the specific hyperbola equation -/
theorem hyperbola_asymptotes :
  (∀ x y : ℝ, hyperbola x y → asymptotic_lines x y) ∧
  ¬(∀ x y : ℝ, asymptotic_lines x y → hyperbola x y) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l2951_295189


namespace NUMINAMATH_CALUDE_probability_system_failure_correct_l2951_295195

/-- The probability of at least one component failing in a system of m identical components -/
def probability_system_failure (m : ℕ) (P : ℝ) : ℝ :=
  1 - (1 - P)^m

/-- Theorem: The probability of at least one component failing in a system of m identical components
    with individual failure probability P is 1-(1-P)^m -/
theorem probability_system_failure_correct (m : ℕ) (P : ℝ) 
    (h1 : 0 ≤ P) (h2 : P ≤ 1) :
  probability_system_failure m P = 1 - (1 - P)^m :=
by
  sorry

end NUMINAMATH_CALUDE_probability_system_failure_correct_l2951_295195


namespace NUMINAMATH_CALUDE_power_zero_eq_one_neg_half_power_zero_l2951_295151

theorem power_zero_eq_one (x : ℚ) (h : x ≠ 0) : x^0 = 1 := by sorry

theorem neg_half_power_zero : (-1/2 : ℚ)^0 = 1 := by sorry

end NUMINAMATH_CALUDE_power_zero_eq_one_neg_half_power_zero_l2951_295151


namespace NUMINAMATH_CALUDE_project_completion_time_l2951_295125

/-- The number of days B takes to complete the project -/
def B_days : ℕ := 30

/-- The total number of days when A and B work together with A quitting 5 days before completion -/
def total_days : ℕ := 15

/-- The number of days before completion that A quits -/
def A_quit_days : ℕ := 5

/-- The number of days A can complete the project alone -/
def A_days : ℕ := 20

theorem project_completion_time :
  (total_days - A_quit_days) * (1 / A_days + 1 / B_days) + A_quit_days * (1 / B_days) = 1 :=
by sorry

#check project_completion_time

end NUMINAMATH_CALUDE_project_completion_time_l2951_295125


namespace NUMINAMATH_CALUDE_school_pet_ownership_stats_l2951_295185

/-- Represents the school statistics -/
structure SchoolStats where
  total_students : ℕ
  cat_owners : ℕ
  dog_owners : ℕ

/-- Calculates the percentage of students owning a specific pet -/
def pet_ownership_percentage (stats : SchoolStats) (pet_owners : ℕ) : ℚ :=
  (pet_owners : ℚ) / (stats.total_students : ℚ) * 100

/-- Calculates the percent difference between two percentages -/
def percent_difference (p1 p2 : ℚ) : ℚ :=
  abs (p1 - p2)

/-- Theorem stating the correctness of the calculated percentages -/
theorem school_pet_ownership_stats (stats : SchoolStats) 
  (h1 : stats.total_students = 500)
  (h2 : stats.cat_owners = 80)
  (h3 : stats.dog_owners = 100) :
  pet_ownership_percentage stats stats.cat_owners = 16 ∧
  percent_difference (pet_ownership_percentage stats stats.dog_owners) (pet_ownership_percentage stats stats.cat_owners) = 4 := by
  sorry

#eval pet_ownership_percentage ⟨500, 80, 100⟩ 80
#eval percent_difference (pet_ownership_percentage ⟨500, 80, 100⟩ 100) (pet_ownership_percentage ⟨500, 80, 100⟩ 80)

end NUMINAMATH_CALUDE_school_pet_ownership_stats_l2951_295185


namespace NUMINAMATH_CALUDE_angle_D_measure_l2951_295109

theorem angle_D_measure (A B C D : ℝ) : 
  -- ABCD is a convex quadrilateral
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  0 < D ∧ D < π ∧
  A + B + C + D = 2 * π ∧
  -- ∠C = 57°
  C = 57 * π / 180 ∧
  -- sin ∠A + sin ∠B = √2
  Real.sin A + Real.sin B = Real.sqrt 2 ∧
  -- cos ∠A + cos ∠B = 2 - √2
  Real.cos A + Real.cos B = 2 - Real.sqrt 2
  -- Then ∠D = 168°
  → D = 168 * π / 180 := by
sorry

end NUMINAMATH_CALUDE_angle_D_measure_l2951_295109


namespace NUMINAMATH_CALUDE_constant_term_expansion_l2951_295153

theorem constant_term_expansion (a : ℝ) : 
  (∃ k : ℝ, k = -40 ∧ k = (a * 40 + 1 * 1)) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l2951_295153


namespace NUMINAMATH_CALUDE_cost_price_per_metre_l2951_295166

/-- Given a trader who sells cloth, this theorem proves the cost price per metre. -/
theorem cost_price_per_metre
  (total_metres : ℕ)
  (selling_price : ℕ)
  (profit_per_metre : ℕ)
  (h1 : total_metres = 85)
  (h2 : selling_price = 8925)
  (h3 : profit_per_metre = 10) :
  (selling_price - total_metres * profit_per_metre) / total_metres = 95 := by
sorry

end NUMINAMATH_CALUDE_cost_price_per_metre_l2951_295166


namespace NUMINAMATH_CALUDE_students_in_neither_subject_l2951_295132

/- Given: -/
def total_students : ℕ := 120
def chemistry_students : ℕ := 75
def biology_students : ℕ := 50
def both_subjects : ℕ := 15

/- Theorem to prove -/
theorem students_in_neither_subject : 
  total_students - (chemistry_students + biology_students - both_subjects) = 10 := by
  sorry

end NUMINAMATH_CALUDE_students_in_neither_subject_l2951_295132


namespace NUMINAMATH_CALUDE_fair_coin_999th_flip_l2951_295155

/-- A fair coin is a coin that has equal probability of landing heads or tails. -/
def FairCoin : Type := Unit

/-- A sequence of coin flips. -/
def CoinFlips (n : ℕ) := Fin n → Bool

/-- The probability of an event occurring in a fair coin flip. -/
def prob (event : Bool → Prop) : ℚ := sorry

theorem fair_coin_999th_flip (c : FairCoin) (flips : CoinFlips 1000) :
  prob (λ result => result = true) = 1/2 := by sorry

end NUMINAMATH_CALUDE_fair_coin_999th_flip_l2951_295155


namespace NUMINAMATH_CALUDE_perpendicular_planes_theorem_l2951_295179

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between lines and planes
variable (perp : Line → Plane → Prop)

-- Define the parallel relation between planes
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_planes_theorem 
  (α β γ : Plane) (m n : Line)
  (h_diff_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ)
  (h_diff_lines : m ≠ n)
  (h_n_perp_α : perp n α)
  (h_n_perp_β : perp n β)
  (h_m_perp_α : perp m α) :
  perp m β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_theorem_l2951_295179


namespace NUMINAMATH_CALUDE_seating_arrangements_count_l2951_295133

/-- The number of ways to arrange n distinct objects. -/
def arrangements (n : ℕ) : ℕ := n.factorial

/-- The number of ways to arrange 3 adults and 3 children in 6 seats,
    such that no two people of the same type sit together. -/
def seating_arrangements : ℕ :=
  2 * arrangements 3 * arrangements 3

/-- Theorem stating that the number of seating arrangements is 72. -/
theorem seating_arrangements_count :
  seating_arrangements = 72 := by sorry

end NUMINAMATH_CALUDE_seating_arrangements_count_l2951_295133


namespace NUMINAMATH_CALUDE_correct_propositions_l2951_295117

-- Define the proposition from statement ②
def proposition_2 : Prop := 
  (∃ x : ℝ, x^2 + 1 > 3*x) ↔ ¬(∀ x : ℝ, x^2 + 1 ≤ 3*x)

-- Define the proposition from statement ③
def proposition_3 : Prop :=
  (∃ x : ℝ, x^2 - 3*x - 4 = 0 ∧ x ≠ 4) ∧ 
  (∀ x : ℝ, x = 4 → x^2 - 3*x - 4 = 0)

theorem correct_propositions : proposition_2 ∧ proposition_3 := by
  sorry

end NUMINAMATH_CALUDE_correct_propositions_l2951_295117


namespace NUMINAMATH_CALUDE_stamp_book_gcd_l2951_295165

theorem stamp_book_gcd : Nat.gcd (Nat.gcd 1260 1470) 1890 = 210 := by
  sorry

end NUMINAMATH_CALUDE_stamp_book_gcd_l2951_295165


namespace NUMINAMATH_CALUDE_evaluate_sqrt_fraction_l2951_295192

theorem evaluate_sqrt_fraction (y : ℝ) (h : y < 0) :
  Real.sqrt (y / (1 - (y - 2) / y)) = -y / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_sqrt_fraction_l2951_295192


namespace NUMINAMATH_CALUDE_complementary_event_of_A_l2951_295126

-- Define the sample space for a fair cubic die
def DieOutcome := Fin 6

-- Define event A
def EventA (outcome : DieOutcome) : Prop := outcome.val % 2 = 1

-- Define the complementary event of A
def ComplementA (outcome : DieOutcome) : Prop := outcome.val % 2 = 0

-- Theorem statement
theorem complementary_event_of_A :
  ∀ (outcome : DieOutcome), ¬(EventA outcome) ↔ ComplementA outcome :=
by sorry

end NUMINAMATH_CALUDE_complementary_event_of_A_l2951_295126


namespace NUMINAMATH_CALUDE_fermat_number_divisibility_l2951_295130

theorem fermat_number_divisibility (m n : ℕ) (h : m > n) :
  ∃ k : ℕ, 2^(2^m) - 1 = (2^(2^n) + 1) * k := by
  sorry

end NUMINAMATH_CALUDE_fermat_number_divisibility_l2951_295130


namespace NUMINAMATH_CALUDE_hyperbola_line_intersection_l2951_295162

/-- Hyperbola equation: x^2 - y^2 = 4 -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 4

/-- Line equation: y = k(x - 1) -/
def line (k x y : ℝ) : Prop := y = k * (x - 1)

/-- The line intersects the hyperbola at two points -/
def intersects_at_two_points (k : ℝ) : Prop :=
  k ∈ Set.Ioo (-(2 * Real.sqrt 3 / 3)) (-1) ∪ 
      Set.Ioo (-1) 1 ∪ 
      Set.Ioo 1 (2 * Real.sqrt 3 / 3)

/-- The line intersects the hyperbola at exactly one point -/
def intersects_at_one_point (k : ℝ) : Prop :=
  k = 1 ∨ k = -1 ∨ k = 2 * Real.sqrt 3 / 3 ∨ k = -(2 * Real.sqrt 3 / 3)

theorem hyperbola_line_intersection :
  (∀ k : ℝ, intersects_at_two_points k ↔ 
    ∃ x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ ∧ 
      hyperbola x₁ y₁ ∧ hyperbola x₂ y₂ ∧ 
      line k x₁ y₁ ∧ line k x₂ y₂) ∧
  (∀ k : ℝ, intersects_at_one_point k ↔ 
    (∃ x y : ℝ, hyperbola x y ∧ line k x y) ∧
    ∀ x₁ y₁ x₂ y₂ : ℝ, hyperbola x₁ y₁ ∧ hyperbola x₂ y₂ ∧ 
      line k x₁ y₁ ∧ line k x₂ y₂ → x₁ = x₂ ∧ y₁ = y₂) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_line_intersection_l2951_295162


namespace NUMINAMATH_CALUDE_ratio_b_to_sum_ac_l2951_295173

theorem ratio_b_to_sum_ac (a b c : ℤ) 
  (sum_eq : a + b + c = 60)
  (a_eq : a = (b + c) / 3)
  (c_eq : c = 35) : 
  b * 5 = a + c := by sorry

end NUMINAMATH_CALUDE_ratio_b_to_sum_ac_l2951_295173


namespace NUMINAMATH_CALUDE_decimal_expansion_contains_all_digits_l2951_295101

theorem decimal_expansion_contains_all_digits (p : ℕ) (hp : p.Prime) (hp_large : p > 10^9) 
  (hq : (4*p + 1).Prime) : 
  ∀ d : Fin 10, ∃ n : ℕ, (10^n - 1) % (4*p + 1) = d.val * ((10^n - 1) / (4*p + 1)) :=
sorry

end NUMINAMATH_CALUDE_decimal_expansion_contains_all_digits_l2951_295101


namespace NUMINAMATH_CALUDE_coord_relationship_l2951_295169

/-- The relationship between x and y coordinates on lines y = x or y = -x --/
theorem coord_relationship (x y : ℝ) : (y = x ∨ y = -x) → |x| - |y| = 0 := by
  sorry

end NUMINAMATH_CALUDE_coord_relationship_l2951_295169


namespace NUMINAMATH_CALUDE_binary_sum_theorem_l2951_295106

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Represents a binary number as a list of bits -/
def binary_number (bits : List Bool) : Nat := binary_to_decimal bits

theorem binary_sum_theorem :
  let a := binary_number [true, false, true, false, true]
  let b := binary_number [true, true, true]
  let c := binary_number [true, false, true, true, true, false]
  let d := binary_number [true, false, true, false, true, true]
  let sum := binary_number [true, true, true, true, false, false, true]
  a + b + c + d = sum := by sorry

end NUMINAMATH_CALUDE_binary_sum_theorem_l2951_295106


namespace NUMINAMATH_CALUDE_power_of_four_l2951_295128

theorem power_of_four (x : ℕ) : 5^29 * 4^x = 2 * 10^29 → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_power_of_four_l2951_295128


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2951_295145

theorem imaginary_part_of_z (z : ℂ) (h : z * (3 - 4*I) = 5) : z.im = 5/7 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2951_295145


namespace NUMINAMATH_CALUDE_spinner_ice_cream_prices_l2951_295113

-- Define the price of a spinner and an ice cream
variable (s m : ℝ)

-- Define Petya's and Vasya's claims
def petya_claim := 2 * s > 5 * m
def vasya_claim := 3 * s > 8 * m

-- Theorem statement
theorem spinner_ice_cream_prices 
  (h1 : (petya_claim s m ∧ ¬vasya_claim s m) ∨ (¬petya_claim s m ∧ vasya_claim s m))
  (h2 : vasya_claim s m) :
  7 * s ≤ 19 * m := by
  sorry

end NUMINAMATH_CALUDE_spinner_ice_cream_prices_l2951_295113


namespace NUMINAMATH_CALUDE_smallest_five_digit_square_cube_l2951_295181

theorem smallest_five_digit_square_cube : ∃ n : ℕ,
  (10000 ≤ n ∧ n < 100000) ∧  -- five-digit number
  (∃ a : ℕ, n = a^2) ∧        -- perfect square
  (∃ b : ℕ, n = b^3) ∧        -- perfect cube
  (∀ m : ℕ, (10000 ≤ m ∧ m < 100000) ∧ 
            (∃ x : ℕ, m = x^2) ∧ 
            (∃ y : ℕ, m = y^3) → 
    n ≤ m) ∧                  -- smallest such number
  n = 15625                   -- the answer
  := by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_square_cube_l2951_295181


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l2951_295111

/-- Given a polynomial p(x) satisfying specific conditions, 
    prove that its remainder when divided by (x-1)(x+1)(x-3) is -x^2 + 4x + 2 -/
theorem polynomial_remainder_theorem (p : ℝ → ℝ) 
  (h1 : p 1 = 5) (h2 : p 3 = 7) (h3 : p (-1) = 9) :
  ∃ q : ℝ → ℝ, ∀ x, p x = q x * (x - 1) * (x + 1) * (x - 3) + (-x^2 + 4*x + 2) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l2951_295111


namespace NUMINAMATH_CALUDE_range_of_m_l2951_295149

def p (m : ℝ) : Prop := 0 ≤ m ∧ m ≤ 3

def q (m : ℝ) : Prop := (m - 2) * (m - 4) ≤ 0

theorem range_of_m :
  ∀ m : ℝ, (¬(p m ∧ q m) ∧ (p m ∨ q m)) → (m ∈ Set.Icc 0 2 ∪ Set.Ioc 3 4) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2951_295149


namespace NUMINAMATH_CALUDE_twenty_students_no_math_l2951_295118

/-- Represents a class of students with information about their subject choices. -/
structure ClassInfo where
  total : ℕ
  no_science : ℕ
  no_either : ℕ
  both : ℕ

/-- Calculates the number of students who didn't opt for math. -/
def students_no_math (info : ClassInfo) : ℕ :=
  info.total - info.both - (info.no_science - info.no_either)

/-- Theorem stating that in a specific class, 20 students didn't opt for math. -/
theorem twenty_students_no_math :
  let info : ClassInfo := {
    total := 40,
    no_science := 15,
    no_either := 2,
    both := 7
  }
  students_no_math info = 20 := by
  sorry


end NUMINAMATH_CALUDE_twenty_students_no_math_l2951_295118


namespace NUMINAMATH_CALUDE_solve_for_a_l2951_295194

theorem solve_for_a (a b : ℝ) (h1 : b/a = 4) (h2 : b = 24 - 4*a) : a = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l2951_295194


namespace NUMINAMATH_CALUDE_find_a_l2951_295161

def round_to_two_decimal_places (x : ℚ) : ℚ :=
  (⌊x * 100 + 0.5⌋ : ℚ) / 100

theorem find_a : ∃ (a : ℕ), round_to_two_decimal_places (1.322 - (a : ℚ) / 99) = 1.10 ∧ a = 22 := by
  sorry

end NUMINAMATH_CALUDE_find_a_l2951_295161


namespace NUMINAMATH_CALUDE_cereal_cost_l2951_295197

/-- Represents the cost of cereal boxes for a year -/
def cereal_problem (boxes_per_week : ℕ) (weeks_per_year : ℕ) (total_cost : ℕ) : Prop :=
  let total_boxes := boxes_per_week * weeks_per_year
  total_cost / total_boxes = 3

/-- Proves that each box of cereal costs $3 given the problem conditions -/
theorem cereal_cost : cereal_problem 2 52 312 := by
  sorry

end NUMINAMATH_CALUDE_cereal_cost_l2951_295197


namespace NUMINAMATH_CALUDE_modular_congruence_13_pow_6_mod_11_l2951_295100

theorem modular_congruence_13_pow_6_mod_11 : 
  ∃ m : ℕ, 13^6 ≡ m [ZMOD 11] ∧ m < 11 → m = 9 := by
  sorry

end NUMINAMATH_CALUDE_modular_congruence_13_pow_6_mod_11_l2951_295100


namespace NUMINAMATH_CALUDE_jonathan_book_purchase_l2951_295136

theorem jonathan_book_purchase (dictionary_cost dinosaur_book_cost cookbook_cost savings : ℕ) 
  (h1 : dictionary_cost = 11)
  (h2 : dinosaur_book_cost = 19)
  (h3 : cookbook_cost = 7)
  (h4 : savings = 8) :
  dictionary_cost + dinosaur_book_cost + cookbook_cost - savings = 29 := by
  sorry

end NUMINAMATH_CALUDE_jonathan_book_purchase_l2951_295136


namespace NUMINAMATH_CALUDE_radical_conjugate_sum_product_l2951_295190

theorem radical_conjugate_sum_product (x y : ℝ) : 
  (x + Real.sqrt y) + (x - Real.sqrt y) = 8 ∧ 
  (x + Real.sqrt y) * (x - Real.sqrt y) = 15 →
  x + y = 5 := by sorry

end NUMINAMATH_CALUDE_radical_conjugate_sum_product_l2951_295190


namespace NUMINAMATH_CALUDE_catch_fraction_l2951_295187

theorem catch_fraction (joe_catches derek_catches tammy_catches : ℕ) : 
  joe_catches = 23 →
  derek_catches = 2 * joe_catches - 4 →
  tammy_catches = 30 →
  (tammy_catches : ℚ) / derek_catches = 5 / 7 := by
sorry

end NUMINAMATH_CALUDE_catch_fraction_l2951_295187


namespace NUMINAMATH_CALUDE_halloween_costume_payment_l2951_295140

theorem halloween_costume_payment (last_year_cost deposit_percentage price_increase : ℝ) 
  (h1 : last_year_cost = 250)
  (h2 : deposit_percentage = 0.1)
  (h3 : price_increase = 0.4) : 
  (1 + price_increase) * last_year_cost * (1 - deposit_percentage) = 315 :=
by sorry

end NUMINAMATH_CALUDE_halloween_costume_payment_l2951_295140


namespace NUMINAMATH_CALUDE_power_division_rule_l2951_295120

theorem power_division_rule (a : ℝ) : a^8 / a^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_division_rule_l2951_295120


namespace NUMINAMATH_CALUDE_fraction_problem_l2951_295148

theorem fraction_problem :
  let x : ℚ := 4
  let y : ℚ := 15
  (y = x^2 - 1) ∧
  ((x + 2) / (y + 2) > 1/4) ∧
  ((x - 3) / (y - 3) = 1/12) := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l2951_295148


namespace NUMINAMATH_CALUDE_product_remainder_one_l2951_295182

theorem product_remainder_one (a b : ℕ) : 
  a % 3 = 1 → b % 3 = 1 → (a * b) % 3 = 1 := by
sorry

end NUMINAMATH_CALUDE_product_remainder_one_l2951_295182


namespace NUMINAMATH_CALUDE_line_through_point_equal_intercepts_l2951_295167

-- Define a line passing through (1, 3) with equal intercepts
def line_equal_intercepts (a b c : ℝ) : Prop :=
  a * 1 + b * 3 + c = 0 ∧  -- Line passes through (1, 3)
  ∃ k : ℝ, k ≠ 0 ∧ a * k + c = 0 ∧ b * k + c = 0  -- Equal intercepts

-- Theorem statement
theorem line_through_point_equal_intercepts :
  ∀ a b c : ℝ, line_equal_intercepts a b c →
  (a = -3 ∧ b = 1 ∧ c = 0) ∨ (a = 1 ∧ b = 1 ∧ c = -4) :=
by sorry

end NUMINAMATH_CALUDE_line_through_point_equal_intercepts_l2951_295167


namespace NUMINAMATH_CALUDE_base_conversion_theorem_l2951_295124

/-- Conversion from base 7 to base 10 -/
def base7ToBase10 (n : Nat) : Nat :=
  (n / 100) * 7^2 + ((n / 10) % 10) * 7 + (n % 10)

/-- Given 563 in base 7 equals xy in base 10, prove that (x+y)/9 = 11/9 -/
theorem base_conversion_theorem :
  let n := 563
  let xy := base7ToBase10 n
  let x := xy / 10
  let y := xy % 10
  (x + y) / 9 = 11 / 9 := by sorry

end NUMINAMATH_CALUDE_base_conversion_theorem_l2951_295124


namespace NUMINAMATH_CALUDE_odd_function_theorem_l2951_295131

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def satisfies_equation (f : ℝ → ℝ) : Prop := ∀ x, f x + f (2 - x) = 4

-- State the theorem
theorem odd_function_theorem (h1 : is_odd f) (h2 : satisfies_equation f) : f 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_theorem_l2951_295131


namespace NUMINAMATH_CALUDE_difference_in_rubber_bands_l2951_295104

-- Define the number of rubber bands Harper has
def harper_bands : ℕ := 15

-- Define the total number of rubber bands they have together
def total_bands : ℕ := 24

-- Define the number of rubber bands Harper's brother has
def brother_bands : ℕ := total_bands - harper_bands

-- Theorem to prove
theorem difference_in_rubber_bands :
  harper_bands - brother_bands = 6 ∧ brother_bands < harper_bands :=
by sorry

end NUMINAMATH_CALUDE_difference_in_rubber_bands_l2951_295104


namespace NUMINAMATH_CALUDE_equation_solution_l2951_295123

theorem equation_solution : ∃ x : ℝ, (5*x + 9*x = 350 - 10*(x - 4)) ∧ x = 16.25 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2951_295123
