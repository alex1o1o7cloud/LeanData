import Mathlib

namespace NUMINAMATH_CALUDE_geometry_theorem_l3722_372249

-- Define the types for lines and planes
variable {Point : Type*}
variable {Line : Type*}
variable {Plane : Type*}

-- Define the necessary relations
variable (subset : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- Define the theorem
theorem geometry_theorem 
  (m n l : Line) (α β : Plane) 
  (h_different_lines : m ≠ n ∧ m ≠ l ∧ n ≠ l) 
  (h_different_planes : α ≠ β) :
  (¬(subset m α) ∧ (subset n α) ∧ (parallel_lines m n) → parallel_line_plane m α) ∧
  ((subset m α) ∧ (perpendicular_line_plane m β) → perpendicular_planes α β) :=
sorry

end NUMINAMATH_CALUDE_geometry_theorem_l3722_372249


namespace NUMINAMATH_CALUDE_dealer_net_profit_dealer_net_profit_is_97_20_l3722_372212

/-- Calculates the dealer's net profit from selling a desk --/
theorem dealer_net_profit (purchase_price : ℝ) (markup_rate : ℝ) (discount_rate : ℝ) 
  (sales_tax_rate : ℝ) (commission_rate : ℝ) : ℝ :=
  let selling_price := purchase_price / (1 - markup_rate)
  let discounted_price := selling_price * (1 - discount_rate)
  let total_payment := discounted_price * (1 + sales_tax_rate)
  let commission := discounted_price * commission_rate
  total_payment - purchase_price - commission

/-- Proves that the dealer's net profit is $97.20 under the given conditions --/
theorem dealer_net_profit_is_97_20 :
  dealer_net_profit 150 0.5 0.2 0.05 0.02 = 97.20 := by
  sorry

end NUMINAMATH_CALUDE_dealer_net_profit_dealer_net_profit_is_97_20_l3722_372212


namespace NUMINAMATH_CALUDE_special_triangle_properties_l3722_372228

/-- Triangle ABC with specific properties -/
structure SpecialTriangle where
  -- Sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- Angles of the triangle
  A : ℝ
  B : ℝ
  C : ℝ
  -- Conditions
  angle_sum : A + B + C = π
  side_angle_relation : (Real.cos B) / (Real.cos C) = b / (2 * a - c)
  b_value : b = Real.sqrt 7
  area : (1/2) * a * c * Real.sin B = (3 * Real.sqrt 3) / 2

/-- Main theorem about the special triangle -/
theorem special_triangle_properties (t : SpecialTriangle) : t.B = π/3 ∧ t.a + t.c = 5 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_properties_l3722_372228


namespace NUMINAMATH_CALUDE_ratio_of_two_numbers_l3722_372230

theorem ratio_of_two_numbers (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : a + b = 44) (h4 : a - b = 20) : a / b = 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_two_numbers_l3722_372230


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l3722_372225

def diamond (a b : ℤ) : ℤ := 2 * a + b

theorem diamond_equation_solution :
  ∃ y : ℤ, diamond 4 (diamond 3 y) = 17 ∧ y = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l3722_372225


namespace NUMINAMATH_CALUDE_fifth_derivative_y_l3722_372237

noncomputable def y (x : ℝ) : ℝ := (4 * x + 3) * (2 : ℝ)^(-x)

theorem fifth_derivative_y (x : ℝ) :
  (deriv^[5] y) x = (-Real.log 2^5 * (4 * x + 3) + 20 * Real.log 2^4) * (2 : ℝ)^(-x) :=
by sorry

end NUMINAMATH_CALUDE_fifth_derivative_y_l3722_372237


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l3722_372256

theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 4/7
  let a₂ : ℚ := 18/49
  let a₃ : ℚ := 162/343
  let r : ℚ := a₂ / a₁
  r = 63/98 := by sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l3722_372256


namespace NUMINAMATH_CALUDE_machine_production_time_l3722_372280

/-- The number of items the machine can produce in one hour -/
def items_per_hour : ℕ := 90

/-- The number of minutes in one hour -/
def minutes_per_hour : ℕ := 60

/-- The time it takes to produce one item in minutes -/
def time_per_item : ℚ := minutes_per_hour / items_per_hour

theorem machine_production_time : time_per_item = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_machine_production_time_l3722_372280


namespace NUMINAMATH_CALUDE_dice_arithmetic_progression_probability_l3722_372202

def num_dice : ℕ := 4
def faces_per_die : ℕ := 6

def is_arithmetic_progression (nums : Finset ℕ) : Prop :=
  ∃ (a d : ℕ), ∀ i ∈ nums, ∃ k : ℕ, i = a + k * d

def favorable_outcomes : Finset (Finset ℕ) :=
  {{1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}}

theorem dice_arithmetic_progression_probability :
  (Finset.card favorable_outcomes) / (faces_per_die ^ num_dice : ℚ) = 1 / 432 := by
  sorry

end NUMINAMATH_CALUDE_dice_arithmetic_progression_probability_l3722_372202


namespace NUMINAMATH_CALUDE_original_denominator_proof_l3722_372229

theorem original_denominator_proof (d : ℚ) : 
  (5 / d : ℚ) ≠ 0 → -- Ensure the original fraction is well-defined
  ((5 - 3) / (d + 4) : ℚ) = 1 / 3 →
  d = 2 := by
sorry

end NUMINAMATH_CALUDE_original_denominator_proof_l3722_372229


namespace NUMINAMATH_CALUDE_smaller_root_of_equation_l3722_372276

theorem smaller_root_of_equation (x : ℚ) :
  (x - 4/5)^2 + (x - 4/5) * (x - 2/5) + (x - 1/2)^2 = 0 →
  x = 14/15 ∨ x = 4/5 ∧ 14/15 < 4/5 := by
sorry

end NUMINAMATH_CALUDE_smaller_root_of_equation_l3722_372276


namespace NUMINAMATH_CALUDE_range_of_a_l3722_372248

-- Define the set M
def M (a : ℝ) : Set ℝ := {x : ℝ | x^2 - 4*x + 4*a < 0}

-- State the theorem
theorem range_of_a (a : ℝ) : (2 ∉ M a) ↔ a ∈ Set.Ici 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3722_372248


namespace NUMINAMATH_CALUDE_basketball_score_proof_l3722_372222

/-- 
Given a basketball team's scoring pattern:
- 4 games with 10t points each
- g games with 20 points each
- Average score of 28 points per game
Prove that g = 16
-/
theorem basketball_score_proof (t : ℕ) (g : ℕ) : 
  (40 * t + 20 * g) / (4 + g) = 28 → g = 16 := by
  sorry

end NUMINAMATH_CALUDE_basketball_score_proof_l3722_372222


namespace NUMINAMATH_CALUDE_product_of_five_consecutive_integers_divisible_by_10_l3722_372232

theorem product_of_five_consecutive_integers_divisible_by_10 (n : ℕ) :
  ∃ k : ℕ, (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) = 10 * k := by
  sorry

end NUMINAMATH_CALUDE_product_of_five_consecutive_integers_divisible_by_10_l3722_372232


namespace NUMINAMATH_CALUDE_regular_tile_area_l3722_372243

/-- Represents the properties of tiles used to cover a wall -/
structure TileInfo where
  regularLength : ℝ
  regularWidth : ℝ
  jumboLength : ℝ
  jumboWidth : ℝ
  totalTiles : ℝ
  regularTiles : ℝ
  jumboTiles : ℝ

/-- Theorem stating the area covered by regular tiles on a wall -/
theorem regular_tile_area (t : TileInfo) (h1 : t.jumboLength = 3 * t.regularLength)
    (h2 : t.jumboWidth = t.regularWidth)
    (h3 : t.jumboTiles = (1/3) * t.totalTiles)
    (h4 : t.regularTiles = (2/3) * t.totalTiles)
    (h5 : t.regularLength * t.regularWidth * t.regularTiles +
          t.jumboLength * t.jumboWidth * t.jumboTiles = 385) :
    t.regularLength * t.regularWidth * t.regularTiles = 154 := by
  sorry

end NUMINAMATH_CALUDE_regular_tile_area_l3722_372243


namespace NUMINAMATH_CALUDE_soccer_lineup_combinations_l3722_372211

def total_players : ℕ := 16
def rookie_players : ℕ := 4
def goalkeeper_count : ℕ := 1
def defender_count : ℕ := 4
def midfielder_count : ℕ := 4
def forward_count : ℕ := 3

def lineup_combinations : ℕ := 
  total_players * 
  (Nat.choose (total_players - 1) defender_count) * 
  (Nat.choose (total_players - 1 - defender_count) midfielder_count) * 
  (Nat.choose rookie_players 2 * Nat.choose (total_players - rookie_players - goalkeeper_count - defender_count - midfielder_count) 1)

theorem soccer_lineup_combinations : 
  lineup_combinations = 21508800 := by sorry

end NUMINAMATH_CALUDE_soccer_lineup_combinations_l3722_372211


namespace NUMINAMATH_CALUDE_solve_for_b_l3722_372261

theorem solve_for_b (a b : ℝ) (h1 : 3 * a + 2 = 5) (h2 : b - 4 * a = 2) : b = 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_b_l3722_372261


namespace NUMINAMATH_CALUDE_parabola_translation_l3722_372295

/-- A parabola defined by a quadratic function -/
def Parabola (a b c : ℝ) := fun (x : ℝ) => a * x^2 + b * x + c

/-- Translation of a function -/
def translate (f : ℝ → ℝ) (dx dy : ℝ) := fun (x : ℝ) => f (x - dx) + dy

theorem parabola_translation (x : ℝ) :
  translate (Parabola 2 4 1) 1 3 x = Parabola 2 0 0 x := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_l3722_372295


namespace NUMINAMATH_CALUDE_inscribed_circle_area_l3722_372240

/-- A circle inscribed in a right triangle with specific properties -/
structure InscribedCircle (A B C X Y : ℝ × ℝ) :=
  (right_angle : (A.1 - B.1) * (A.1 - C.1) + (A.2 - B.2) * (A.2 - C.2) = 0)
  (ab_length : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 6)
  (tangent_x : (X.1 - A.1) * (B.1 - A.1) + (X.2 - A.2) * (B.2 - A.2) = 0)
  (tangent_y : (Y.1 - A.1) * (C.1 - A.1) + (Y.2 - A.2) * (C.2 - A.2) = 0)
  (opposite_on_bc : ∃ (X' Y' : ℝ × ℝ), 
    (X'.1 - B.1) * (C.1 - B.1) + (X'.2 - B.2) * (C.2 - B.2) = 0 ∧
    (Y'.1 - B.1) * (C.1 - B.1) + (Y'.2 - B.2) * (C.2 - B.2) = 0 ∧
    (X'.1 - X.1)^2 + (X'.2 - X.2)^2 = (Y'.1 - Y.1)^2 + (Y'.2 - Y.2)^2)

/-- The area of the portion of the circle outside the triangle is π - 2 -/
theorem inscribed_circle_area (A B C X Y : ℝ × ℝ) 
  (h : InscribedCircle A B C X Y) : 
  ∃ (r : ℝ), r > 0 ∧ π * r^2 / 4 - r^2 / 2 = π - 2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_area_l3722_372240


namespace NUMINAMATH_CALUDE_unit_digit_of_15_power_l3722_372219

theorem unit_digit_of_15_power (X : ℕ+) : ∃ n : ℕ, 15^(X : ℕ) ≡ 5 [MOD 10] :=
sorry

end NUMINAMATH_CALUDE_unit_digit_of_15_power_l3722_372219


namespace NUMINAMATH_CALUDE_sum_of_selected_numbers_l3722_372221

def numbers : List ℚ := [14/10, 9/10, 12/10, 5/10, 13/10]
def threshold : ℚ := 11/10

theorem sum_of_selected_numbers :
  (numbers.filter (λ x => x ≥ threshold)).sum = 39/10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_selected_numbers_l3722_372221


namespace NUMINAMATH_CALUDE_necklace_cost_calculation_l3722_372206

/-- The cost of a single necklace -/
def necklace_cost : ℕ := sorry

/-- The number of necklaces sold -/
def necklaces_sold : ℕ := 4

/-- The number of rings sold -/
def rings_sold : ℕ := 8

/-- The cost of a single ring -/
def ring_cost : ℕ := 4

/-- The total sales amount -/
def total_sales : ℕ := 80

theorem necklace_cost_calculation :
  necklace_cost = 12 :=
by
  sorry

#check necklace_cost_calculation

end NUMINAMATH_CALUDE_necklace_cost_calculation_l3722_372206


namespace NUMINAMATH_CALUDE_two_digit_product_555_sum_l3722_372284

theorem two_digit_product_555_sum (x y : ℕ) : 
  10 ≤ x ∧ x < 100 ∧ 10 ≤ y ∧ y < 100 ∧ x * y = 555 → x + y = 52 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_product_555_sum_l3722_372284


namespace NUMINAMATH_CALUDE_swim_trunks_price_l3722_372208

def flat_rate_shipping : ℝ := 5.00
def shipping_threshold : ℝ := 50.00
def shipping_rate : ℝ := 0.20
def shirt_price : ℝ := 12.00
def shirt_quantity : ℕ := 3
def socks_price : ℝ := 5.00
def shorts_price : ℝ := 15.00
def shorts_quantity : ℕ := 2
def total_bill : ℝ := 102.00

def known_items_cost : ℝ := shirt_price * shirt_quantity + socks_price + shorts_price * shorts_quantity

theorem swim_trunks_price (x : ℝ) : 
  (known_items_cost + x + shipping_rate * (known_items_cost + x) = total_bill) → 
  x = 14.00 := by
  sorry

end NUMINAMATH_CALUDE_swim_trunks_price_l3722_372208


namespace NUMINAMATH_CALUDE_tan_two_implies_fraction_five_l3722_372271

theorem tan_two_implies_fraction_five (α : Real) (h : Real.tan α = 2) :
  (2 * Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 5 := by
  sorry

end NUMINAMATH_CALUDE_tan_two_implies_fraction_five_l3722_372271


namespace NUMINAMATH_CALUDE_age_ratio_in_two_years_l3722_372217

def brother_age : ℕ := 10
def man_age : ℕ := brother_age + 12

theorem age_ratio_in_two_years :
  (man_age + 2) / (brother_age + 2) = 2 := by sorry

end NUMINAMATH_CALUDE_age_ratio_in_two_years_l3722_372217


namespace NUMINAMATH_CALUDE_extremum_values_l3722_372247

/-- The function f(x) = x³ - ax² - bx + a² has an extremum of 10 at x = 1 -/
def has_extremum (a b : ℝ) : Prop :=
  let f := fun x : ℝ => x^3 - a*x^2 - b*x + a^2
  (∃ ε > 0, ∀ x, 0 < |x - 1| ∧ |x - 1| < ε → f x ≤ f 1) ∧
  (∃ ε > 0, ∀ x, 0 < |x - 1| ∧ |x - 1| < ε → f x ≥ f 1) ∧
  f 1 = 10

theorem extremum_values (a b : ℝ) :
  has_extremum a b → a = -4 ∧ b = 11 := by
  sorry

end NUMINAMATH_CALUDE_extremum_values_l3722_372247


namespace NUMINAMATH_CALUDE_min_sum_m_n_l3722_372242

theorem min_sum_m_n (m n : ℕ+) (h1 : 45 * m = n ^ 3) (h2 : ∃ k : ℕ+, n = 5 * k) :
  (∀ m' n' : ℕ+, 45 * m' = n' ^ 3 → (∃ k' : ℕ+, n' = 5 * k') → m + n ≤ m' + n') →
  m + n = 90 := by
sorry

end NUMINAMATH_CALUDE_min_sum_m_n_l3722_372242


namespace NUMINAMATH_CALUDE_apple_distribution_l3722_372223

theorem apple_distribution (total_apples : ℕ) (total_bags : ℕ) (x : ℕ) :
  total_apples = 109 →
  total_bags = 20 →
  (∃ k : ℕ, k * x + (total_bags - k) * 3 = total_apples ∧ 0 < k ∧ k ≤ total_bags) →
  (x = 10 ∨ x = 52) :=
by sorry

end NUMINAMATH_CALUDE_apple_distribution_l3722_372223


namespace NUMINAMATH_CALUDE_library_book_distribution_l3722_372251

/-- The number of ways to distribute books between the library and checked-out status -/
def distribution_count (total : ℕ) (min_in_library : ℕ) (min_checked_out : ℕ) : ℕ :=
  (total - min_in_library - min_checked_out + 1)

/-- Theorem: For 8 identical books with at least 2 in the library and 2 checked out,
    there are 5 different ways to distribute the books -/
theorem library_book_distribution :
  distribution_count 8 2 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_library_book_distribution_l3722_372251


namespace NUMINAMATH_CALUDE_whale_length_in_crossing_scenario_l3722_372205

/-- The length of a whale in a crossing scenario --/
theorem whale_length_in_crossing_scenario
  (v_fast : ℝ)  -- Initial speed of the faster whale
  (v_slow : ℝ)  -- Initial speed of the slower whale
  (a_fast : ℝ)  -- Acceleration of the faster whale
  (a_slow : ℝ)  -- Acceleration of the slower whale
  (t : ℝ)       -- Time taken for the faster whale to cross the slower whale
  (h_v_fast : v_fast = 18)
  (h_v_slow : v_slow = 15)
  (h_a_fast : a_fast = 1)
  (h_a_slow : a_slow = 0.5)
  (h_t : t = 15) :
  let d_fast := v_fast * t + (1/2) * a_fast * t^2
  let d_slow := v_slow * t + (1/2) * a_slow * t^2
  d_fast - d_slow = 101.25 := by
sorry


end NUMINAMATH_CALUDE_whale_length_in_crossing_scenario_l3722_372205


namespace NUMINAMATH_CALUDE_quadratic_equation_condition_l3722_372299

/-- For the equation (a-2)x^2 + (a+2)x + 3 = 0 to be a quadratic equation in one variable, a ≠ 2 -/
theorem quadratic_equation_condition (a : ℝ) : 
  (∀ x, ∃ y, y = (a - 2) * x^2 + (a + 2) * x + 3) → a ≠ 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_condition_l3722_372299


namespace NUMINAMATH_CALUDE_max_grid_mean_l3722_372231

def Grid := Fin 3 → Fin 3 → ℕ

def valid_grid (g : Grid) : Prop :=
  (∀ i j, g i j ∈ Finset.range 9) ∧
  (∀ n, n ∈ Finset.range 9 → ∃ i j, g i j = n)

def circle_mean (g : Grid) (i j : Fin 2) : ℚ :=
  (g i j + g i (j+1) + g (i+1) j + g (i+1) (j+1)) / 4

def grid_mean (g : Grid) : ℚ :=
  (circle_mean g 0 0 + circle_mean g 0 1 + circle_mean g 1 0 + circle_mean g 1 1) / 4

theorem max_grid_mean :
  ∀ g : Grid, valid_grid g → grid_mean g ≤ 5.8125 :=
sorry

end NUMINAMATH_CALUDE_max_grid_mean_l3722_372231


namespace NUMINAMATH_CALUDE_least_product_consecutive_primes_above_50_l3722_372259

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def consecutive_primes (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ ∀ r : ℕ, is_prime r → (p < r → r < q) → r = p ∨ r = q

theorem least_product_consecutive_primes_above_50 :
  ∃ p q : ℕ, consecutive_primes p q ∧ p > 50 ∧ q > 50 ∧
  p * q = 3127 ∧
  ∀ a b : ℕ, consecutive_primes a b → a > 50 → b > 50 → a * b ≥ 3127 :=
sorry

end NUMINAMATH_CALUDE_least_product_consecutive_primes_above_50_l3722_372259


namespace NUMINAMATH_CALUDE_line_intersects_y_axis_l3722_372255

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line passing through two points
def Line (p1 p2 : Point2D) :=
  {p : Point2D | ∃ t : ℝ, p.x = p1.x + t * (p2.x - p1.x) ∧ p.y = p1.y + t * (p2.y - p1.y)}

-- The given points
def p1 : Point2D := ⟨2, 9⟩
def p2 : Point2D := ⟨4, 15⟩

-- The y-axis
def yAxis : Set Point2D := {p : Point2D | p.x = 0}

-- The intersection point
def intersectionPoint : Point2D := ⟨0, 3⟩

-- The theorem to prove
theorem line_intersects_y_axis :
  intersectionPoint ∈ Line p1 p2 ∩ yAxis := by sorry

end NUMINAMATH_CALUDE_line_intersects_y_axis_l3722_372255


namespace NUMINAMATH_CALUDE_derivative_x_ln_x_l3722_372274

open Real

theorem derivative_x_ln_x (x : ℝ) (h : x > 0) :
  deriv (fun x => x * log x) x = log x + 1 := by
  sorry

end NUMINAMATH_CALUDE_derivative_x_ln_x_l3722_372274


namespace NUMINAMATH_CALUDE_three_W_seven_l3722_372227

-- Define the W operation
def W (a b : ℝ) : ℝ := b + 5 * a - 3 * a^2

-- Theorem to prove
theorem three_W_seven : W 3 7 = -5 := by
  sorry

end NUMINAMATH_CALUDE_three_W_seven_l3722_372227


namespace NUMINAMATH_CALUDE_not_all_isosceles_congruent_l3722_372214

/-- An isosceles triangle -/
structure IsoscelesTriangle where
  side1 : ℝ
  side2 : ℝ
  base : ℝ
  is_isosceles : side1 = side2

/-- Congruence of triangles -/
def are_congruent (t1 t2 : IsoscelesTriangle) : Prop :=
  t1.side1 = t2.side1 ∧ t1.side2 = t2.side2 ∧ t1.base = t2.base

/-- Theorem: Not all isosceles triangles are congruent -/
theorem not_all_isosceles_congruent : 
  ∃ t1 t2 : IsoscelesTriangle, ¬(are_congruent t1 t2) :=
sorry

end NUMINAMATH_CALUDE_not_all_isosceles_congruent_l3722_372214


namespace NUMINAMATH_CALUDE_points_collinearity_l3722_372292

/-- Checks if three points are collinear -/
def are_collinear (x1 y1 x2 y2 x3 y3 : ℝ) : Prop :=
  (y2 - y1) * (x3 - x2) = (y3 - y2) * (x2 - x1)

theorem points_collinearity :
  (are_collinear 1 2 2 4 3 6) ∧
  ¬(are_collinear 2 3 (-2) 1 3 4) := by
  sorry

end NUMINAMATH_CALUDE_points_collinearity_l3722_372292


namespace NUMINAMATH_CALUDE_perimeter_specific_midpoint_triangle_l3722_372285

/-- A solid right prism with regular hexagonal bases -/
structure RightPrism where
  height : ℝ
  base_side_length : ℝ

/-- Midpoint of an edge -/
structure Midpoint where
  edge : String

/-- Triangle formed by three midpoints -/
structure MidpointTriangle where
  point1 : Midpoint
  point2 : Midpoint
  point3 : Midpoint

/-- Calculate the perimeter of the midpoint triangle -/
def perimeter_midpoint_triangle (prism : RightPrism) (triangle : MidpointTriangle) : ℝ :=
  sorry

/-- Theorem stating the perimeter of the specific midpoint triangle -/
theorem perimeter_specific_midpoint_triangle :
  ∀ (prism : RightPrism) (triangle : MidpointTriangle),
  prism.height = 20 ∧ 
  prism.base_side_length = 10 ∧
  triangle.point1 = Midpoint.mk "AB" ∧
  triangle.point2 = Midpoint.mk "BC" ∧
  triangle.point3 = Midpoint.mk "EF" →
  perimeter_midpoint_triangle prism triangle = 45 :=
sorry

end NUMINAMATH_CALUDE_perimeter_specific_midpoint_triangle_l3722_372285


namespace NUMINAMATH_CALUDE_trader_weighted_avg_gain_percentage_l3722_372244

/-- Calculates the weighted average gain percentage for a trader selling three types of pens -/
theorem trader_weighted_avg_gain_percentage
  (quantity_A quantity_B quantity_C : ℕ)
  (cost_A cost_B cost_C : ℚ)
  (gain_quantity_A gain_quantity_B gain_quantity_C : ℕ)
  (h_quantity_A : quantity_A = 60)
  (h_quantity_B : quantity_B = 40)
  (h_quantity_C : quantity_C = 50)
  (h_cost_A : cost_A = 2)
  (h_cost_B : cost_B = 3)
  (h_cost_C : cost_C = 4)
  (h_gain_quantity_A : gain_quantity_A = 20)
  (h_gain_quantity_B : gain_quantity_B = 15)
  (h_gain_quantity_C : gain_quantity_C = 10) :
  let total_cost := quantity_A * cost_A + quantity_B * cost_B + quantity_C * cost_C
  let total_gain := gain_quantity_A * cost_A + gain_quantity_B * cost_B + gain_quantity_C * cost_C
  let weighted_avg_gain_percentage := (total_gain / total_cost) * 100
  weighted_avg_gain_percentage = 28.41 := by
  sorry

end NUMINAMATH_CALUDE_trader_weighted_avg_gain_percentage_l3722_372244


namespace NUMINAMATH_CALUDE_point_translation_proof_l3722_372233

def translate_point (p : ℝ × ℝ) (dx dy : ℝ) : ℝ × ℝ :=
  (p.1 + dx, p.2 + dy)

theorem point_translation_proof :
  let P : ℝ × ℝ := (-3, 4)
  let Q : ℝ × ℝ := translate_point (translate_point P 0 (-3)) 2 0
  Q = (-1, 1) := by sorry

end NUMINAMATH_CALUDE_point_translation_proof_l3722_372233


namespace NUMINAMATH_CALUDE_average_height_problem_l3722_372204

theorem average_height_problem (parker daisy reese : ℕ) : 
  parker + 4 = daisy →
  daisy = reese + 8 →
  reese = 60 →
  (parker + daisy + reese) / 3 = 64 := by
sorry

end NUMINAMATH_CALUDE_average_height_problem_l3722_372204


namespace NUMINAMATH_CALUDE_division_and_addition_l3722_372290

theorem division_and_addition : (10 / (1/5)) + 6 = 56 := by
  sorry

end NUMINAMATH_CALUDE_division_and_addition_l3722_372290


namespace NUMINAMATH_CALUDE_fair_prize_division_l3722_372254

/-- Represents the state of the game --/
structure GameState where
  player1_wins : ℕ
  player2_wins : ℕ

/-- Calculates the probability of a player winning the game from a given state --/
def win_probability (state : GameState) : ℚ :=
  1 - (1/2) ^ (6 - state.player1_wins)

/-- Theorem stating the fair division of the prize --/
theorem fair_prize_division (state : GameState) 
  (h1 : state.player1_wins = 5)
  (h2 : state.player2_wins = 3) :
  let p1_prob := win_probability state
  let p2_prob := 1 - p1_prob
  (p1_prob : ℚ) / p2_prob = 7 / 1 := by sorry

end NUMINAMATH_CALUDE_fair_prize_division_l3722_372254


namespace NUMINAMATH_CALUDE_cuts_through_examples_l3722_372266

-- Define what it means for a line to cut through a curve at a point
def cuts_through (l : ℝ → ℝ) (c : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  -- The line is tangent to the curve at the point
  (∀ x, l x = c x + (l p.1 - c p.1)) ∧
  -- The curve is on both sides of the line near the point
  ∃ δ > 0, ∀ x ∈ Set.Ioo (p.1 - δ) (p.1 + δ), 
    (x < p.1 → c x < l x) ∧ (x > p.1 → c x > l x)

-- Theorem statement
theorem cuts_through_examples :
  cuts_through (λ _ => 0) (λ x => x^3) (0, 0) ∧
  cuts_through (λ x => x) Real.sin (0, 0) ∧
  cuts_through (λ x => x) Real.tan (0, 0) := by
  sorry

end NUMINAMATH_CALUDE_cuts_through_examples_l3722_372266


namespace NUMINAMATH_CALUDE_jacks_tire_slashing_l3722_372253

theorem jacks_tire_slashing (tire_cost window_cost total_cost : ℕ) 
  (h1 : tire_cost = 250)
  (h2 : window_cost = 700)
  (h3 : total_cost = 1450) :
  ∃ (num_tires : ℕ), num_tires * tire_cost + window_cost = total_cost ∧ num_tires = 3 := by
  sorry

end NUMINAMATH_CALUDE_jacks_tire_slashing_l3722_372253


namespace NUMINAMATH_CALUDE_expression_simplification_l3722_372239

theorem expression_simplification (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 2) :
  let f := (((x^2 - x) / (x^2 - 2*x + 1) + 2 / (x - 1)) / ((x^2 - 4) / (x^2 - 1)))
  x = 3 → f = 4 := by sorry

end NUMINAMATH_CALUDE_expression_simplification_l3722_372239


namespace NUMINAMATH_CALUDE_system_of_inequalities_solution_l3722_372263

theorem system_of_inequalities_solution (x : ℝ) : 
  ((x - 1) / 2 < 2 * x + 1 ∧ -3 * (1 - x) ≥ -4) ↔ x ≥ -1/3 := by
  sorry

end NUMINAMATH_CALUDE_system_of_inequalities_solution_l3722_372263


namespace NUMINAMATH_CALUDE_white_balls_count_l3722_372235

theorem white_balls_count (total : ℕ) (p_red p_black : ℚ) (h_total : total = 50)
  (h_red : p_red = 15/100) (h_black : p_black = 45/100) :
  (total : ℚ) * (1 - p_red - p_black) = 20 := by
  sorry

end NUMINAMATH_CALUDE_white_balls_count_l3722_372235


namespace NUMINAMATH_CALUDE_fish_fraction_removed_on_day_five_l3722_372287

/-- Represents the number of fish in Jason's aquarium on a given day -/
def fish (day : ℕ) : ℚ :=
  match day with
  | 0 => 6
  | 1 => 12
  | 2 => 16
  | 3 => 32
  | 4 => 64
  | 5 => 128
  | 6 => 256
  | _ => 0

/-- The fraction of fish removed on day 5 -/
def f : ℚ := 1/4

theorem fish_fraction_removed_on_day_five :
  fish 6 - 4 * f * fish 4 + 15 = 207 :=
sorry

end NUMINAMATH_CALUDE_fish_fraction_removed_on_day_five_l3722_372287


namespace NUMINAMATH_CALUDE_system_solution_l3722_372220

theorem system_solution : 
  ∃! (x y : ℝ), x = 4 * y ∧ x + 2 * y = -12 ∧ x = -8 ∧ y = -2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3722_372220


namespace NUMINAMATH_CALUDE_meaningful_fraction_range_l3722_372245

theorem meaningful_fraction_range (x : ℝ) :
  (|x| - 6 ≠ 0) ↔ (x ≠ 6 ∧ x ≠ -6) := by
  sorry

end NUMINAMATH_CALUDE_meaningful_fraction_range_l3722_372245


namespace NUMINAMATH_CALUDE_production_days_calculation_l3722_372201

theorem production_days_calculation (n : ℕ) : 
  (70 * n + 90 = 75 * (n + 1)) → n = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_production_days_calculation_l3722_372201


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l3722_372207

theorem quadratic_roots_relation (a b n r s : ℝ) : 
  (a^2 - n*a + 3 = 0) →
  (b^2 - n*b + 3 = 0) →
  ((a + 1/b)^2 - r*(a + 1/b) + s = 0) →
  ((b + 1/a)^2 - r*(b + 1/a) + s = 0) →
  s = 16/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l3722_372207


namespace NUMINAMATH_CALUDE_solution_set_f_less_than_2_range_of_a_for_nonempty_solution_l3722_372264

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

-- Theorem for the solution set of f(x) < 2
theorem solution_set_f_less_than_2 :
  {x : ℝ | f x < 2} = {x : ℝ | 1/2 < x ∧ x < 5/2} :=
sorry

-- Theorem for the range of a
theorem range_of_a_for_nonempty_solution (a : ℝ) :
  (∃ x : ℝ, f x < a) ↔ a > 1 :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_less_than_2_range_of_a_for_nonempty_solution_l3722_372264


namespace NUMINAMATH_CALUDE_parallel_iff_a_eq_neg_one_l3722_372215

/-- Two lines in the plane -/
structure TwoLines where
  a : ℝ
  line1 : ℝ × ℝ → Prop := fun (x, y) ↦ a * x + 2 * y + 2 = 0
  line2 : ℝ × ℝ → Prop := fun (x, y) ↦ x + (a - 1) * y + 1 = 0

/-- The lines are parallel -/
def areParallel (lines : TwoLines) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ 
    (∀ (x y : ℝ), lines.line1 (x, y) ↔ lines.line2 (k * x + lines.a, k * y + 2))

/-- The main theorem -/
theorem parallel_iff_a_eq_neg_one (lines : TwoLines) :
  areParallel lines ↔ lines.a = -1 :=
sorry

end NUMINAMATH_CALUDE_parallel_iff_a_eq_neg_one_l3722_372215


namespace NUMINAMATH_CALUDE_hyperbola_foci_l3722_372218

def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 12 = 1

def is_focus (x y : ℝ) : Prop :=
  (x = 4 ∧ y = 0) ∨ (x = -4 ∧ y = 0)

theorem hyperbola_foci :
  ∀ x y : ℝ, hyperbola_equation x y → is_focus x y :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_l3722_372218


namespace NUMINAMATH_CALUDE_man_swimming_speed_l3722_372281

/-- The speed of a man in still water given his downstream and upstream swimming times and distances -/
theorem man_swimming_speed
  (downstream_distance : ℝ)
  (downstream_time : ℝ)
  (upstream_distance : ℝ)
  (upstream_time : ℝ)
  (h1 : downstream_distance = 50)
  (h2 : downstream_time = 4)
  (h3 : upstream_distance = 30)
  (h4 : upstream_time = 6)
  : ∃ (v_m : ℝ), v_m = 8.75 ∧ 
    downstream_distance / downstream_time = v_m + (downstream_distance / downstream_time - v_m) ∧
    upstream_distance / upstream_time = v_m - (downstream_distance / downstream_time - v_m) :=
by
  sorry

#check man_swimming_speed

end NUMINAMATH_CALUDE_man_swimming_speed_l3722_372281


namespace NUMINAMATH_CALUDE_max_value_equality_l3722_372283

theorem max_value_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b)^2 / (a^2 + 2*a*b + b^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_equality_l3722_372283


namespace NUMINAMATH_CALUDE_park_playgroups_l3722_372298

theorem park_playgroups (girls boys parents playgroups : ℕ) 
  (h1 : girls = 14)
  (h2 : boys = 11)
  (h3 : parents = 50)
  (h4 : playgroups = 3)
  (h5 : (girls + boys + parents) % playgroups = 0) :
  (girls + boys + parents) / playgroups = 25 := by
  sorry

end NUMINAMATH_CALUDE_park_playgroups_l3722_372298


namespace NUMINAMATH_CALUDE_number_of_combinations_prob_one_black_prob_at_least_one_blue_l3722_372250

/-- Represents the total number of pens -/
def total_pens : ℕ := 6

/-- Represents the number of black pens -/
def black_pens : ℕ := 3

/-- Represents the number of blue pens -/
def blue_pens : ℕ := 2

/-- Represents the number of red pens -/
def red_pens : ℕ := 1

/-- Represents the number of pens to be selected -/
def selected_pens : ℕ := 3

/-- Theorem stating the number of possible combinations when selecting 3 pens out of 6 -/
theorem number_of_combinations : Nat.choose total_pens selected_pens = 20 := by sorry

/-- Theorem stating the probability of selecting exactly one black pen -/
theorem prob_one_black : (Nat.choose black_pens 1 * Nat.choose (blue_pens + red_pens) 2) / Nat.choose total_pens selected_pens = 9 / 20 := by sorry

/-- Theorem stating the probability of selecting at least one blue pen -/
theorem prob_at_least_one_blue : 1 - (Nat.choose (black_pens + red_pens) selected_pens) / Nat.choose total_pens selected_pens = 4 / 5 := by sorry

end NUMINAMATH_CALUDE_number_of_combinations_prob_one_black_prob_at_least_one_blue_l3722_372250


namespace NUMINAMATH_CALUDE_pencils_given_eq_difference_l3722_372216

/-- The number of pencils Jesse gave to Joshua -/
def pencils_given : ℕ := sorry

/-- The initial number of pencils Jesse had -/
def initial_pencils : ℕ := 78

/-- The remaining number of pencils Jesse has -/
def remaining_pencils : ℕ := 34

/-- Theorem stating that the number of pencils given is equal to the difference between initial and remaining pencils -/
theorem pencils_given_eq_difference : 
  pencils_given = initial_pencils - remaining_pencils := by sorry

end NUMINAMATH_CALUDE_pencils_given_eq_difference_l3722_372216


namespace NUMINAMATH_CALUDE_triangular_number_difference_l3722_372200

/-- The nth triangular number -/
def triangularNumber (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The difference between the 2010th and 2009th triangular numbers is 2010 -/
theorem triangular_number_difference : triangularNumber 2010 - triangularNumber 2009 = 2010 := by
  sorry

end NUMINAMATH_CALUDE_triangular_number_difference_l3722_372200


namespace NUMINAMATH_CALUDE_circle_radius_calculation_l3722_372272

theorem circle_radius_calculation (d PQ QR : ℝ) (h1 : d = 15) (h2 : PQ = 10) (h3 : QR = 8) :
  ∃ r : ℝ, r = 3 * Real.sqrt 5 ∧ PQ * (PQ + QR) = (d - r) * (d + r) := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_calculation_l3722_372272


namespace NUMINAMATH_CALUDE_next_simultaneous_ring_l3722_372279

def library_period : ℕ := 18
def fire_station_period : ℕ := 24
def hospital_period : ℕ := 30

def minutes_in_hour : ℕ := 60

theorem next_simultaneous_ring (start_time : ℕ) :
  ∃ (t : ℕ), t > 0 ∧ 
    t % library_period = 0 ∧ 
    t % fire_station_period = 0 ∧ 
    t % hospital_period = 0 ∧
    t / minutes_in_hour = 6 := by
  sorry

end NUMINAMATH_CALUDE_next_simultaneous_ring_l3722_372279


namespace NUMINAMATH_CALUDE_largest_number_with_property_l3722_372273

/-- Checks if a four-digit number satisfies the property that each of the last two digits
    is equal to the sum of the two preceding digits. -/
def satisfiesProperty (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  (n % 100 = (n / 100 % 10 + n / 10 % 10) % 10) ∧
  (n / 10 % 10 = (n / 1000 + n / 100 % 10) % 10)

/-- Theorem stating that 9099 is the largest four-digit number satisfying the property. -/
theorem largest_number_with_property :
  satisfiesProperty 9099 ∧ ∀ m : ℕ, satisfiesProperty m → m ≤ 9099 :=
sorry

end NUMINAMATH_CALUDE_largest_number_with_property_l3722_372273


namespace NUMINAMATH_CALUDE_correct_mark_calculation_l3722_372252

/-- Proves that if a mark of 83 in a class of 26 pupils increases the class average by 0.5,
    then the correct mark should have been 70. -/
theorem correct_mark_calculation (total_marks : ℝ) (wrong_mark correct_mark : ℝ) : 
  (wrong_mark = 83) →
  (((total_marks + wrong_mark) / 26) = ((total_marks + correct_mark) / 26 + 0.5)) →
  (correct_mark = 70) := by
sorry

end NUMINAMATH_CALUDE_correct_mark_calculation_l3722_372252


namespace NUMINAMATH_CALUDE_train_length_l3722_372267

/-- The length of a train given its speed and time to cross a post -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 40 → time_s = 19.8 → 
  (speed_kmh * 1000 / 3600) * time_s = 220 := by sorry

end NUMINAMATH_CALUDE_train_length_l3722_372267


namespace NUMINAMATH_CALUDE_intersection_condition_l3722_372275

/-- Curve C in the xy-plane -/
def C (x y : ℝ) : Prop := y^2 = 6*x - 2 ∧ y ≥ 0

/-- Line l in the xy-plane -/
def L (x y m : ℝ) : Prop := y = Real.sqrt 3 * x + 2*m

/-- Intersection points of C and L -/
def Intersection (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | C p.1 p.2 ∧ L p.1 p.2 m}

/-- Two distinct intersection points exist -/
def HasTwoDistinctIntersections (m : ℝ) : Prop :=
  ∃ p q : ℝ × ℝ, p ∈ Intersection m ∧ q ∈ Intersection m ∧ p ≠ q

theorem intersection_condition (m : ℝ) :
  HasTwoDistinctIntersections m ↔ -Real.sqrt 3 / 6 ≤ m ∧ m < Real.sqrt 3 / 12 := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_l3722_372275


namespace NUMINAMATH_CALUDE_positive_number_square_sum_l3722_372296

theorem positive_number_square_sum (n : ℝ) : n > 0 ∧ n^2 + n = 210 → n = 14 := by
  sorry

end NUMINAMATH_CALUDE_positive_number_square_sum_l3722_372296


namespace NUMINAMATH_CALUDE_chessboard_polygon_theorem_l3722_372297

/-- A polygon cut out from an infinite chessboard -/
structure ChessboardPolygon where
  black_cells : ℕ              -- number of black cells
  white_cells : ℕ              -- number of white cells
  black_perimeter : ℕ          -- number of black perimeter segments
  white_perimeter : ℕ          -- number of white perimeter segments

/-- Theorem stating the relationship between perimeter segments and cells -/
theorem chessboard_polygon_theorem (p : ChessboardPolygon) :
  p.black_perimeter - p.white_perimeter = 4 * (p.black_cells - p.white_cells) := by
  sorry

end NUMINAMATH_CALUDE_chessboard_polygon_theorem_l3722_372297


namespace NUMINAMATH_CALUDE_edward_book_purchase_l3722_372241

/-- Given that Edward spent $6 on books and each book cost $3, prove that he bought 2 books. -/
theorem edward_book_purchase (total_spent : ℕ) (cost_per_book : ℕ) (h1 : total_spent = 6) (h2 : cost_per_book = 3) :
  total_spent / cost_per_book = 2 := by
  sorry

end NUMINAMATH_CALUDE_edward_book_purchase_l3722_372241


namespace NUMINAMATH_CALUDE_division_problem_l3722_372236

theorem division_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 176 →
  quotient = 12 →
  remainder = 8 →
  dividend = divisor * quotient + remainder →
  divisor = 14 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3722_372236


namespace NUMINAMATH_CALUDE_intersection_is_empty_l3722_372226

def A : Set ℝ := {x | x^2 - 2*x > 0}
def B : Set ℝ := {x | |x + 1| < 0}

theorem intersection_is_empty : A ∩ B = ∅ := by
  sorry

end NUMINAMATH_CALUDE_intersection_is_empty_l3722_372226


namespace NUMINAMATH_CALUDE_spinner_final_direction_l3722_372257

-- Define the direction type
inductive Direction
  | North
  | East
  | South
  | West

-- Define the rotation type
inductive Rotation
  | Clockwise
  | Counterclockwise

-- Define a function to calculate the final direction after a rotation
def rotateSpinner (initialDir : Direction) (rotation : Rotation) (revolutions : ℚ) : Direction :=
  sorry

-- Theorem statement
theorem spinner_final_direction :
  let initialDir := Direction.North
  let clockwiseRot := 7/2
  let counterclockwiseRot := 21/4
  let finalDir := rotateSpinner (rotateSpinner initialDir Rotation.Clockwise clockwiseRot) Rotation.Counterclockwise counterclockwiseRot
  finalDir = Direction.East := by sorry

end NUMINAMATH_CALUDE_spinner_final_direction_l3722_372257


namespace NUMINAMATH_CALUDE_polynomial_not_factorizable_l3722_372268

theorem polynomial_not_factorizable : 
  ¬ ∃ (a b c d : ℤ), ∀ (x : ℝ), 
    x^4 + 3*x^3 + 6*x^2 + 9*x + 12 = (x^2 + a*x + b) * (x^2 + c*x + d) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_not_factorizable_l3722_372268


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l3722_372293

/-- Represents a circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Determines if two circles are externally tangent --/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2) ^ 2 + (y1 - y2) ^ 2 = (c1.radius + c2.radius) ^ 2

theorem circles_externally_tangent : 
  let c1 : Circle := { center := (4, 2), radius := 3 }
  let c2 : Circle := { center := (0, -1), radius := 2 }
  are_externally_tangent c1 c2 := by sorry

end NUMINAMATH_CALUDE_circles_externally_tangent_l3722_372293


namespace NUMINAMATH_CALUDE_hyperbola_to_ellipse_l3722_372210

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 12 = -1

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 16 = 1

-- Theorem statement
theorem hyperbola_to_ellipse :
  ∀ (x y : ℝ),
  hyperbola_equation x y →
  (∃ (a b : ℝ), ellipse_equation a b) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_to_ellipse_l3722_372210


namespace NUMINAMATH_CALUDE_zola_paityn_blue_hat_ratio_l3722_372269

/-- Proves the ratio of Zola's blue hats to Paityn's blue hats -/
theorem zola_paityn_blue_hat_ratio :
  let paityn_red : ℕ := 20
  let paityn_blue : ℕ := 24
  let zola_red : ℕ := (4 * paityn_red) / 5
  let total_hats : ℕ := 54 * 2
  let zola_blue : ℕ := total_hats - paityn_red - paityn_blue - zola_red
  (zola_blue : ℚ) / paityn_blue = 2 := by
  sorry

end NUMINAMATH_CALUDE_zola_paityn_blue_hat_ratio_l3722_372269


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l3722_372270

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

theorem intersection_complement_equality : A ∩ (U \ B) = {1, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l3722_372270


namespace NUMINAMATH_CALUDE_manuscript_cost_theorem_l3722_372282

/-- Represents the cost of typing and revising a manuscript. -/
def manuscript_cost (
  total_pages : ℕ
  ) (
  first_type_cost : ℕ
  ) (
  first_revision_cost : ℕ
  ) (
  second_revision_cost : ℕ
  ) (
  third_plus_revision_cost : ℕ
  ) (
  pages_revised_once : ℕ
  ) (
  pages_revised_twice : ℕ
  ) (
  pages_revised_thrice : ℕ
  ) (
  pages_revised_four_times : ℕ
  ) : ℕ :=
  total_pages * first_type_cost +
  pages_revised_once * first_revision_cost +
  pages_revised_twice * (first_revision_cost + second_revision_cost) +
  pages_revised_thrice * (first_revision_cost + second_revision_cost + third_plus_revision_cost) +
  pages_revised_four_times * (first_revision_cost + second_revision_cost + 2 * third_plus_revision_cost)

/-- Theorem: The total cost of typing and revising the manuscript is $2240. -/
theorem manuscript_cost_theorem :
  manuscript_cost 270 5 3 2 1 90 60 30 20 = 2240 := by
  sorry


end NUMINAMATH_CALUDE_manuscript_cost_theorem_l3722_372282


namespace NUMINAMATH_CALUDE_total_arrangements_l3722_372294

/-- The number of ways to arrange 3 events in 4 venues with at most 2 events per venue -/
def arrangeEvents : ℕ := sorry

/-- The total number of arrangements is 60 -/
theorem total_arrangements : arrangeEvents = 60 := by sorry

end NUMINAMATH_CALUDE_total_arrangements_l3722_372294


namespace NUMINAMATH_CALUDE_quadratic_root_range_l3722_372262

theorem quadratic_root_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x + 3) ↔ x ≥ -3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l3722_372262


namespace NUMINAMATH_CALUDE_tricycle_count_l3722_372286

/-- Represents the number of wheels on a scooter -/
def scooter_wheels : ℕ := 2

/-- Represents the number of wheels on a tricycle -/
def tricycle_wheels : ℕ := 3

/-- Represents the total number of vehicles -/
def total_vehicles : ℕ := 10

/-- Represents the total number of wheels -/
def total_wheels : ℕ := 26

/-- Theorem stating that the number of tricycles must be 6 given the conditions -/
theorem tricycle_count :
  ∃ (scooters tricycles : ℕ),
    scooters + tricycles = total_vehicles ∧
    scooters * scooter_wheels + tricycles * tricycle_wheels = total_wheels ∧
    tricycles = 6 :=
by sorry

end NUMINAMATH_CALUDE_tricycle_count_l3722_372286


namespace NUMINAMATH_CALUDE_valleyball_hockey_league_players_l3722_372289

/-- The cost of a pair of gloves in dollars -/
def glove_cost : ℕ := 7

/-- The additional cost of a helmet compared to gloves in dollars -/
def helmet_additional_cost : ℕ := 8

/-- The total cost to equip all players in the league in dollars -/
def total_league_cost : ℕ := 3570

/-- The number of sets of equipment each player needs -/
def sets_per_player : ℕ := 2

/-- The number of players in the league -/
def num_players : ℕ := 81

theorem valleyball_hockey_league_players :
  num_players * sets_per_player * (glove_cost + (glove_cost + helmet_additional_cost)) = total_league_cost :=
sorry

end NUMINAMATH_CALUDE_valleyball_hockey_league_players_l3722_372289


namespace NUMINAMATH_CALUDE_remaining_money_is_48_6_l3722_372291

/-- Calculates the remaining money in Country B's currency after shopping in Country A -/
def remaining_money_country_b (initial_amount : ℝ) (grocery_ratio : ℝ) (household_ratio : ℝ) 
  (personal_ratio : ℝ) (household_tax : ℝ) (personal_discount : ℝ) (exchange_rate : ℝ) : ℝ :=
  let groceries := initial_amount * grocery_ratio
  let household := initial_amount * household_ratio * (1 + household_tax)
  let personal := initial_amount * personal_ratio * (1 - personal_discount)
  let total_spent := groceries + household + personal
  let remaining_a := initial_amount - total_spent
  remaining_a * exchange_rate

/-- Theorem stating that the remaining money in Country B's currency is 48.6 units -/
theorem remaining_money_is_48_6 : 
  remaining_money_country_b 450 (3/5) (1/6) (1/10) 0.05 0.1 0.8 = 48.6 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_is_48_6_l3722_372291


namespace NUMINAMATH_CALUDE_tetris_single_ratio_is_eight_to_one_l3722_372224

/-- The ratio of points for a tetris to points for a single line -/
def tetris_to_single_ratio (single_points tetris_points : ℕ) : ℚ :=
  tetris_points / single_points

/-- The total score given the number of singles, number of tetrises, points for a single, and points for a tetris -/
def total_score (num_singles num_tetrises single_points tetris_points : ℕ) : ℕ :=
  num_singles * single_points + num_tetrises * tetris_points

theorem tetris_single_ratio_is_eight_to_one :
  ∃ (tetris_points : ℕ),
    single_points = 1000 ∧
    num_singles = 6 ∧
    num_tetrises = 4 ∧
    total_score num_singles num_tetrises single_points tetris_points = 38000 ∧
    tetris_to_single_ratio single_points tetris_points = 8 := by
  sorry

end NUMINAMATH_CALUDE_tetris_single_ratio_is_eight_to_one_l3722_372224


namespace NUMINAMATH_CALUDE_vector_magnitude_l3722_372265

theorem vector_magnitude (a b : ℝ × ℝ) : 
  (3 • a - 2 • b) • (5 • a + b) = 0 → 
  a • b = 1/7 → 
  ‖a‖ = 1 → 
  ‖b‖ = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l3722_372265


namespace NUMINAMATH_CALUDE_consecutive_integers_sqrt_three_sum_l3722_372238

theorem consecutive_integers_sqrt_three_sum (a b : ℤ) : 
  (b = a + 1) → (a < Real.sqrt 3) → (Real.sqrt 3 < b) → (a + b = 3) := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_sqrt_three_sum_l3722_372238


namespace NUMINAMATH_CALUDE_abs_eq_self_iff_nonneg_l3722_372234

theorem abs_eq_self_iff_nonneg (x : ℝ) : |x| = x ↔ x ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_abs_eq_self_iff_nonneg_l3722_372234


namespace NUMINAMATH_CALUDE_removing_2013th_digit_increases_one_seventh_l3722_372213

-- Define the decimal representation of 1/7
def one_seventh_decimal : ℚ := 1 / 7

-- Define the period of the repeating decimal
def period : ℕ := 6

-- Define the position of the digit to be removed
def removed_digit_position : ℕ := 2013

-- Define the function that removes the nth digit after the decimal point
def remove_nth_digit (q : ℚ) (n : ℕ) : ℚ := sorry

-- Theorem statement
theorem removing_2013th_digit_increases_one_seventh :
  remove_nth_digit one_seventh_decimal removed_digit_position > one_seventh_decimal := by
  sorry

end NUMINAMATH_CALUDE_removing_2013th_digit_increases_one_seventh_l3722_372213


namespace NUMINAMATH_CALUDE_inequality_solution_l3722_372288

def solution_set : Set ℝ := Set.union (Set.Icc 2 3) (Set.Ioc 3 48)

theorem inequality_solution (x : ℝ) : 
  x ∈ solution_set ↔ (x ≠ 3 ∧ (x * (x + 2)) / ((x - 3)^2) ≥ 8) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3722_372288


namespace NUMINAMATH_CALUDE_optimal_solution_l3722_372203

/-- Represents a container with its size and count -/
structure Container where
  size : Nat
  count : Nat

/-- Calculates the total volume of water from a list of containers -/
def totalVolume (containers : List Container) : Nat :=
  containers.foldl (fun acc c => acc + c.size * c.count) 0

/-- Calculates the total number of trips for a list of containers -/
def totalTrips (containers : List Container) : Nat :=
  containers.foldl (fun acc c => acc + c.count) 0

/-- Theorem stating that the given solution is optimal -/
theorem optimal_solution (initialVolume timeLimit : Nat) : 
  let targetVolume : Nat := 823
  let containers : List Container := [
    { size := 8, count := 18 },
    { size := 2, count := 1 },
    { size := 5, count := 1 }
  ]
  (initialVolume = 676) →
  (timeLimit = 45) →
  (totalVolume containers + initialVolume ≥ targetVolume) ∧
  (totalTrips containers ≤ timeLimit) ∧
  (∀ (otherContainers : List Container),
    (totalVolume otherContainers + initialVolume ≥ targetVolume) →
    (totalTrips otherContainers ≤ timeLimit) →
    (totalTrips containers ≤ totalTrips otherContainers)) :=
by sorry


end NUMINAMATH_CALUDE_optimal_solution_l3722_372203


namespace NUMINAMATH_CALUDE_circle_equation_l3722_372246

/-- Given a circle passing through points A(0,-6) and B(1,-5), with its center lying on the line x-y+1=0,
    prove that the standard equation of the circle is (x+3)^2 + (y+2)^2 = 25. -/
theorem circle_equation (C : ℝ × ℝ) : 
  (C.1 - C.2 + 1 = 0) →  -- Center lies on the line x-y+1=0
  ((0 : ℝ) - C.1)^2 + ((-6 : ℝ) - C.2)^2 = ((1 : ℝ) - C.1)^2 + ((-5 : ℝ) - C.2)^2 →  -- Circle passes through A and B
  ∀ (x y : ℝ), (x + 3)^2 + (y + 2)^2 = 25 ↔ (x - C.1)^2 + (y - C.2)^2 = ((0 : ℝ) - C.1)^2 + ((-6 : ℝ) - C.2)^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l3722_372246


namespace NUMINAMATH_CALUDE_no_coin_solution_l3722_372277

theorem no_coin_solution : ¬∃ (x y z : ℕ), 
  x + y + z = 50 ∧ 10 * x + 34 * y + 62 * z = 910 := by
  sorry

end NUMINAMATH_CALUDE_no_coin_solution_l3722_372277


namespace NUMINAMATH_CALUDE_odd_squares_difference_is_perfect_square_l3722_372209

theorem odd_squares_difference_is_perfect_square (m n : ℤ) 
  (h_m_odd : Odd m) (h_n_odd : Odd n) 
  (h_divisible : ∃ k : ℤ, n^2 - 1 = k * (m^2 + 1 - n^2)) :
  ∃ k : ℤ, |m^2 + 1 - n^2| = k^2 := by
  sorry

end NUMINAMATH_CALUDE_odd_squares_difference_is_perfect_square_l3722_372209


namespace NUMINAMATH_CALUDE_bathtub_volume_l3722_372278

/-- Represents the problem of calculating the volume of a bathtub filled with jello --/
def BathtubProblem (jello_per_pound : ℚ) (gallons_per_cubic_foot : ℚ) (pounds_per_gallon : ℚ) 
                   (cost_per_tablespoon : ℚ) (total_spent : ℚ) : Prop :=
  let tablespoons := total_spent / cost_per_tablespoon
  let pounds_of_water := tablespoons / jello_per_pound
  let gallons_of_water := pounds_of_water / pounds_per_gallon
  let cubic_feet := gallons_of_water / gallons_per_cubic_foot
  cubic_feet = 6

/-- The main theorem stating that given the problem conditions, the bathtub holds 6 cubic feet of water --/
theorem bathtub_volume : 
  BathtubProblem (3/2) (15/2) 8 (1/2) 270 := by
  sorry

#check bathtub_volume

end NUMINAMATH_CALUDE_bathtub_volume_l3722_372278


namespace NUMINAMATH_CALUDE_class_artworks_l3722_372260

theorem class_artworks (total_students : ℕ) (total_kits : ℕ) 
  (students_one_kit : ℕ) (students_two_kits : ℕ)
  (students_five_works : ℕ) (students_six_works : ℕ) (students_seven_works : ℕ) :
  total_students = 24 →
  total_kits = 36 →
  students_one_kit = 12 →
  students_two_kits = 12 →
  students_five_works = 8 →
  students_six_works = 10 →
  students_seven_works = 6 →
  students_one_kit + students_two_kits = total_students →
  students_five_works + students_six_works + students_seven_works = total_students →
  students_five_works * 5 + students_six_works * 6 + students_seven_works * 7 = 142 :=
by sorry

end NUMINAMATH_CALUDE_class_artworks_l3722_372260


namespace NUMINAMATH_CALUDE_proposition_relations_l3722_372258

-- Define the original proposition
def p (a : ℝ) : Prop := a > 0 → a^2 ≠ 0

-- Define the converse
def converse (a : ℝ) : Prop := a^2 ≠ 0 → a > 0

-- Define the inverse
def inverse (a : ℝ) : Prop := ¬(a > 0) → a^2 = 0

-- Define the contrapositive
def contrapositive (a : ℝ) : Prop := a^2 = 0 → ¬(a > 0)

-- Define the negation
def negation : Prop := ∃ a : ℝ, a > 0 ∧ a^2 = 0

-- Theorem stating the truth values of each related proposition
theorem proposition_relations :
  (∃ a : ℝ, ¬(converse a)) ∧
  (∃ a : ℝ, ¬(inverse a)) ∧
  (∀ a : ℝ, contrapositive a) ∧
  ¬negation :=
sorry

end NUMINAMATH_CALUDE_proposition_relations_l3722_372258
