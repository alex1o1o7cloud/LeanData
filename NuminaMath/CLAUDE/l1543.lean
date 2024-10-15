import Mathlib

namespace NUMINAMATH_CALUDE_sale_price_equals_original_l1543_154361

theorem sale_price_equals_original (x : ℝ) : x > 0 → 0.8 * (1.25 * x) = x := by
  sorry

end NUMINAMATH_CALUDE_sale_price_equals_original_l1543_154361


namespace NUMINAMATH_CALUDE_least_years_to_double_l1543_154336

def interest_rate : ℝ := 0.13

def more_than_doubled (years : ℕ) : Prop :=
  (1 + interest_rate) ^ years > 2

theorem least_years_to_double :
  (∀ y : ℕ, y < 6 → ¬(more_than_doubled y)) ∧ 
  more_than_doubled 6 :=
sorry

end NUMINAMATH_CALUDE_least_years_to_double_l1543_154336


namespace NUMINAMATH_CALUDE_line_slopes_product_l1543_154300

theorem line_slopes_product (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) :
  (∃ θ : ℝ, m = Real.tan (3 * θ) ∧ n = Real.tan θ) →
  m = 9 * n →
  m * n = 81 / 13 := by
  sorry

end NUMINAMATH_CALUDE_line_slopes_product_l1543_154300


namespace NUMINAMATH_CALUDE_range_of_m_l1543_154309

-- Define the propositions p and q
def p (x : ℝ) : Prop := -2 ≤ (4 - x) / 3 ∧ (4 - x) / 3 ≤ 2

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Define the theorem
theorem range_of_m (m : ℝ) :
  (∀ x, ¬(p x) → ¬(q x m)) ∧  -- ¬p is necessary for ¬q
  (∃ x, ¬(p x) ∧ q x m) ∧     -- ¬p is not sufficient for ¬q
  (m > 0) →                   -- m is positive
  m ≥ 9 :=                    -- The range of m
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1543_154309


namespace NUMINAMATH_CALUDE_jogger_count_difference_l1543_154319

/-- Proves the difference in jogger counts between Christopher and Alexander --/
theorem jogger_count_difference :
  ∀ (christopher_count tyson_count alexander_count : ℕ),
  christopher_count = 80 →
  christopher_count = 20 * tyson_count →
  alexander_count = tyson_count + 22 →
  christopher_count - alexander_count = 54 := by
sorry

end NUMINAMATH_CALUDE_jogger_count_difference_l1543_154319


namespace NUMINAMATH_CALUDE_pencil_pen_cost_l1543_154335

/-- The cost of pencils and pens -/
theorem pencil_pen_cost (x y : ℚ) 
  (h1 : 8 * x + 3 * y = 5.1)
  (h2 : 3 * x + 5 * y = 4.95) :
  4 * x + 4 * y = 4.488 := by
  sorry

end NUMINAMATH_CALUDE_pencil_pen_cost_l1543_154335


namespace NUMINAMATH_CALUDE_right_triangle_shorter_leg_l1543_154379

theorem right_triangle_shorter_leg (a b c : ℕ) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = 65 →           -- Hypotenuse length
  a ≤ b →            -- a is the shorter leg
  a = 16 :=          -- The shorter leg is 16 units long
by sorry

end NUMINAMATH_CALUDE_right_triangle_shorter_leg_l1543_154379


namespace NUMINAMATH_CALUDE_find_A_l1543_154363

theorem find_A : ∃ A : ℝ, (∃ B : ℝ, 211.5 = B - A ∧ B = 10 * A) → A = 23.5 := by
  sorry

end NUMINAMATH_CALUDE_find_A_l1543_154363


namespace NUMINAMATH_CALUDE_fractal_sequence_2000_and_sum_l1543_154303

/-- The fractal sequence a_n -/
def fractal_sequence : ℕ → ℕ
  | 0 => 1
  | n + 1 => 
    let k := (n + 1).log2
    if n + 1 = 2^k - 1 then k
    else fractal_sequence (n - 2^(k-1) + 1)

/-- Sum of the first n terms of the fractal sequence -/
def fractal_sum (n : ℕ) : ℕ :=
  (List.range n).map fractal_sequence |>.sum

theorem fractal_sequence_2000_and_sum :
  fractal_sequence 1999 = 2 ∧ fractal_sum 2000 = 4004 := by
  sorry

#eval fractal_sequence 1999
#eval fractal_sum 2000

end NUMINAMATH_CALUDE_fractal_sequence_2000_and_sum_l1543_154303


namespace NUMINAMATH_CALUDE_total_chairs_l1543_154316

/-- Calculates the total number of chairs at a wedding. -/
theorem total_chairs (initial_rows : ℕ) (chairs_per_row : ℕ) (extra_chairs : ℕ) : 
  initial_rows = 7 → chairs_per_row = 12 → extra_chairs = 11 →
  initial_rows * chairs_per_row + extra_chairs = 95 := by
  sorry

end NUMINAMATH_CALUDE_total_chairs_l1543_154316


namespace NUMINAMATH_CALUDE_green_blue_difference_after_double_border_l1543_154364

/-- Represents a hexagonal figure with blue and green tiles -/
structure HexagonalFigure where
  blue_tiles : ℕ
  green_tiles : ℕ

/-- Calculates the number of tiles in a single border layer of a hexagon -/
def border_layer_tiles (layer : ℕ) : ℕ :=
  6 * (2 * layer + 1)

/-- Adds a double border of green tiles to a hexagonal figure -/
def add_double_border (figure : HexagonalFigure) : HexagonalFigure :=
  { blue_tiles := figure.blue_tiles,
    green_tiles := figure.green_tiles + border_layer_tiles 1 + border_layer_tiles 2 }

/-- Theorem: The difference between green and blue tiles after adding a double border is 50 -/
theorem green_blue_difference_after_double_border (initial_figure : HexagonalFigure)
    (h_blue : initial_figure.blue_tiles = 20)
    (h_green : initial_figure.green_tiles = 10) :
    let final_figure := add_double_border initial_figure
    final_figure.green_tiles - final_figure.blue_tiles = 50 := by
  sorry


end NUMINAMATH_CALUDE_green_blue_difference_after_double_border_l1543_154364


namespace NUMINAMATH_CALUDE_farm_chickens_l1543_154312

theorem farm_chickens (total : ℕ) 
  (h1 : (total : ℚ) * (1/5) = (total : ℚ) * (20/100))  -- 20% of chickens are BCM
  (h2 : ((total : ℚ) * (1/5)) * (4/5) = ((total : ℚ) * (20/100)) * (80/100))  -- 80% of BCM are hens
  (h3 : ((total : ℚ) * (1/5)) * (4/5) = 16)  -- There are 16 BCM hens
  : total = 100 := by
sorry

end NUMINAMATH_CALUDE_farm_chickens_l1543_154312


namespace NUMINAMATH_CALUDE_marco_has_largest_number_l1543_154387

def ellen_final (start : ℕ) : ℕ :=
  ((start - 2) * 3) + 4

def marco_final (start : ℕ) : ℕ :=
  ((start * 3) - 3) + 5

def lucia_final (start : ℕ) : ℕ :=
  ((start - 3) + 5) * 3

theorem marco_has_largest_number :
  let ellen_start := 12
  let marco_start := 15
  let lucia_start := 13
  marco_final marco_start > ellen_final ellen_start ∧
  marco_final marco_start > lucia_final lucia_start :=
by sorry

end NUMINAMATH_CALUDE_marco_has_largest_number_l1543_154387


namespace NUMINAMATH_CALUDE_power_of_fraction_l1543_154349

theorem power_of_fraction : (5 / 6 : ℚ) ^ 4 = 625 / 1296 := by sorry

end NUMINAMATH_CALUDE_power_of_fraction_l1543_154349


namespace NUMINAMATH_CALUDE_prob_ace_king_is_4_663_l1543_154315

/-- Represents a standard deck of cards. -/
structure Deck :=
  (total_cards : ℕ := 52)
  (num_aces : ℕ := 4)
  (num_kings : ℕ := 4)

/-- Calculates the probability of drawing an Ace first and a King second from a standard deck. -/
def prob_ace_then_king (d : Deck) : ℚ :=
  (d.num_aces : ℚ) / d.total_cards * d.num_kings / (d.total_cards - 1)

/-- Theorem stating the probability of drawing an Ace first and a King second from a standard deck. -/
theorem prob_ace_king_is_4_663 (d : Deck) : prob_ace_then_king d = 4 / 663 := by
  sorry

end NUMINAMATH_CALUDE_prob_ace_king_is_4_663_l1543_154315


namespace NUMINAMATH_CALUDE_min_n_constant_term_is_correct_l1543_154306

/-- The minimum positive integer n such that the expansion of (x^2 + 1/(2x^3))^n contains a constant term, where x is a positive integer. -/
def min_n_constant_term : ℕ := 5

/-- Predicate to check if the expansion of (x^2 + 1/(2x^3))^n contains a constant term -/
def has_constant_term (n : ℕ) : Prop :=
  ∃ (k : ℕ), 2 * n = 5 * k

theorem min_n_constant_term_is_correct :
  (∀ m : ℕ, m < min_n_constant_term → ¬has_constant_term m) ∧
  has_constant_term min_n_constant_term :=
sorry

end NUMINAMATH_CALUDE_min_n_constant_term_is_correct_l1543_154306


namespace NUMINAMATH_CALUDE_polygon_sides_count_l1543_154323

theorem polygon_sides_count (n : ℕ) : 
  (n - 2) * 180 = 2 * 360 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_count_l1543_154323


namespace NUMINAMATH_CALUDE_cos_alpha_for_point_on_terminal_side_l1543_154305

/-- Given a point P(-3, -4) on the terminal side of angle α, prove that cos α = -3/5 -/
theorem cos_alpha_for_point_on_terminal_side (α : Real) :
  let P : Prod Real Real := (-3, -4)
  ∃ (r : Real), r > 0 ∧ P = (r * Real.cos α, r * Real.sin α) →
  Real.cos α = -3/5 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_for_point_on_terminal_side_l1543_154305


namespace NUMINAMATH_CALUDE_max_stamps_purchasable_l1543_154341

theorem max_stamps_purchasable (stamp_price : ℕ) (budget : ℕ) : 
  stamp_price = 25 → budget = 5000 → 
  ∃ n : ℕ, n * stamp_price ≤ budget ∧ 
  ∀ m : ℕ, m * stamp_price ≤ budget → m ≤ n ∧ 
  n = 200 :=
by sorry

end NUMINAMATH_CALUDE_max_stamps_purchasable_l1543_154341


namespace NUMINAMATH_CALUDE_f_2017_neg_two_eq_three_fifths_l1543_154385

def f (x : ℚ) : ℚ := (1 + x) / (1 - 3*x)

def f_n : ℕ → (ℚ → ℚ)
  | 0 => id
  | n + 1 => f ∘ f_n n

theorem f_2017_neg_two_eq_three_fifths :
  f_n 2017 (-2) = 3/5 := by sorry

end NUMINAMATH_CALUDE_f_2017_neg_two_eq_three_fifths_l1543_154385


namespace NUMINAMATH_CALUDE_school_books_count_l1543_154399

def total_books : ℕ := 58
def sports_books : ℕ := 39

theorem school_books_count : total_books - sports_books = 19 := by
  sorry

end NUMINAMATH_CALUDE_school_books_count_l1543_154399


namespace NUMINAMATH_CALUDE_root_sum_sixth_power_l1543_154328

theorem root_sum_sixth_power (r s : ℝ) 
  (h1 : r + s = Real.sqrt 7)
  (h2 : r * s = 1) : 
  r^6 + s^6 = 527 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_sixth_power_l1543_154328


namespace NUMINAMATH_CALUDE_town_population_problem_l1543_154374

theorem town_population_problem (p : ℕ) : 
  (p + 1500 : ℝ) * 0.8 = p + 1500 + 50 → p = 1750 := by
  sorry

end NUMINAMATH_CALUDE_town_population_problem_l1543_154374


namespace NUMINAMATH_CALUDE_sqrt_sum_theorem_l1543_154311

theorem sqrt_sum_theorem (a b : ℝ) : 
  Real.sqrt ((a - b)^2) + (a - b)^(1/5) = 
    if a ≥ b then 2*(a - b) else 0 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_theorem_l1543_154311


namespace NUMINAMATH_CALUDE_min_k_for_reciprocal_like_l1543_154383

/-- A directed graph representing people liking each other in a group -/
structure LikeGraph where
  n : ℕ  -- number of people
  k : ℕ  -- number of people each person likes
  edges : Fin n → Finset (Fin n)
  outDegree : ∀ v, (edges v).card = k

/-- There exists a pair of people who like each other reciprocally -/
def hasReciprocalLike (g : LikeGraph) : Prop :=
  ∃ i j : Fin g.n, i ≠ j ∧ i ∈ g.edges j ∧ j ∈ g.edges i

/-- The minimum k that guarantees a reciprocal like in a group of 30 people -/
theorem min_k_for_reciprocal_like :
  ∀ k : ℕ, (∀ g : LikeGraph, g.n = 30 ∧ g.k = k → hasReciprocalLike g) ↔ k ≥ 15 :=
sorry

end NUMINAMATH_CALUDE_min_k_for_reciprocal_like_l1543_154383


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l1543_154362

theorem arithmetic_calculations : 
  ((82 - 15) * (32 + 18) = 3350) ∧ ((25 + 4) * 75 = 2175) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l1543_154362


namespace NUMINAMATH_CALUDE_inequality_proof_l1543_154347

theorem inequality_proof (a b c : ℝ) :
  a * b + b * c + c * a + max (|a - b|) (max (|b - c|) (|c - a|)) ≤ 1 + (1/3) * (a + b + c)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1543_154347


namespace NUMINAMATH_CALUDE_function_value_at_symmetry_point_l1543_154396

theorem function_value_at_symmetry_point (ω φ : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 3 * Real.cos (ω * x + φ)
  (∀ x, f (π / 3 + x) = f (π / 3 - x)) →
  f (π / 3) = 3 ∨ f (π / 3) = -3 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_symmetry_point_l1543_154396


namespace NUMINAMATH_CALUDE_ellipse_y_axis_intersection_l1543_154373

/-- Definition of the ellipse with given foci and one intersection point -/
def ellipse (P : ℝ × ℝ) : Prop :=
  let F₁ : ℝ × ℝ := (-1, 3)
  let F₂ : ℝ × ℝ := (4, 1)
  let P₁ : ℝ × ℝ := (0, 1)
  Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) + 
  Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 
  Real.sqrt ((P₁.1 - F₁.1)^2 + (P₁.2 - F₁.2)^2) + 
  Real.sqrt ((P₁.1 - F₂.1)^2 + (P₁.2 - F₂.2)^2)

/-- The theorem stating that (0, -2) is the other intersection point -/
theorem ellipse_y_axis_intersection :
  ∃ (y : ℝ), y ≠ 1 ∧ ellipse (0, y) ∧ y = -2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_y_axis_intersection_l1543_154373


namespace NUMINAMATH_CALUDE_gcd_condition_iff_special_form_l1543_154313

theorem gcd_condition_iff_special_form (n m : ℕ) (hn : n > 0) (hm : m > 0) :
  Nat.gcd ((n + 1)^m - n) ((n + 1)^(m+3) - n) > 1 ↔
  ∃ (k l : ℕ), k > 0 ∧ l > 0 ∧ n = 7*k - 6 ∧ m = 3*l :=
sorry

end NUMINAMATH_CALUDE_gcd_condition_iff_special_form_l1543_154313


namespace NUMINAMATH_CALUDE_rationalize_sqrt_five_l1543_154358

theorem rationalize_sqrt_five : 
  (2 + Real.sqrt 5) / (3 - Real.sqrt 5) = 11/4 + (5/4) * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_sqrt_five_l1543_154358


namespace NUMINAMATH_CALUDE_ny_mets_fans_count_l1543_154322

/-- Represents the number of fans for each team -/
structure FanCounts where
  yankees : ℕ
  mets : ℕ
  red_sox : ℕ

/-- The total number of baseball fans in the town -/
def total_fans : ℕ := 390

/-- Checks if the given fan counts satisfy the ratio conditions -/
def satisfies_ratios (fans : FanCounts) : Prop :=
  3 * fans.mets = 2 * fans.yankees ∧
  4 * fans.red_sox = 5 * fans.mets

/-- Checks if the given fan counts sum up to the total number of fans -/
def satisfies_total (fans : FanCounts) : Prop :=
  fans.yankees + fans.mets + fans.red_sox = total_fans

/-- The main theorem stating that there are 104 NY Mets fans -/
theorem ny_mets_fans_count :
  ∃ (fans : FanCounts),
    satisfies_ratios fans ∧
    satisfies_total fans ∧
    fans.mets = 104 :=
  sorry

end NUMINAMATH_CALUDE_ny_mets_fans_count_l1543_154322


namespace NUMINAMATH_CALUDE_insufficient_pharmacies_l1543_154317

/-- Represents a grid of streets -/
structure StreetGrid where
  north_south : Nat
  west_east : Nat

/-- Represents a pharmacy's coverage area -/
structure PharmacyCoverage where
  width : Nat
  height : Nat

/-- Calculates the number of street segments in a grid -/
def streetSegments (grid : StreetGrid) : Nat :=
  2 * (grid.north_south - 1) * grid.west_east

/-- Calculates the number of intersections covered by a single pharmacy -/
def intersectionsCovered (coverage : PharmacyCoverage) : Nat :=
  (coverage.width - 1) * (coverage.height - 1)

/-- Theorem stating that 12 pharmacies are not enough to cover all street segments -/
theorem insufficient_pharmacies
  (grid : StreetGrid)
  (coverage : PharmacyCoverage)
  (h_grid : grid = { north_south := 10, west_east := 10 })
  (h_coverage : coverage = { width := 7, height := 7 })
  (h_pharmacies : Nat := 12) :
  h_pharmacies * intersectionsCovered coverage < streetSegments grid := by
  sorry

end NUMINAMATH_CALUDE_insufficient_pharmacies_l1543_154317


namespace NUMINAMATH_CALUDE_second_number_value_l1543_154354

theorem second_number_value (A B : ℝ) : 
  A = 15 → 
  0.4 * A = 0.8 * B + 2 → 
  B = 5 := by
sorry

end NUMINAMATH_CALUDE_second_number_value_l1543_154354


namespace NUMINAMATH_CALUDE_unique_congruence_in_range_l1543_154350

theorem unique_congruence_in_range : ∃! n : ℤ, 10 ≤ n ∧ n ≤ 15 ∧ n ≡ 12345 [ZMOD 7] := by
  sorry

end NUMINAMATH_CALUDE_unique_congruence_in_range_l1543_154350


namespace NUMINAMATH_CALUDE_cyclists_problem_l1543_154377

/-- Two cyclists problem -/
theorem cyclists_problem (v₁ v₂ t : ℝ) :
  v₁ > 0 ∧ v₂ > 0 ∧ t > 0 ∧
  v₁ * t = v₂ * (1.5 : ℝ) ∧
  v₂ * t = v₁ * (2/3 : ℝ) →
  t = 1 ∧ v₁ / v₂ = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_cyclists_problem_l1543_154377


namespace NUMINAMATH_CALUDE_solution_set_of_composite_function_l1543_154382

/-- Given a function f(x) = 2x - 1, the solution set of f[f(x)] ≥ 1 is {x | x ≥ 1} -/
theorem solution_set_of_composite_function (f : ℝ → ℝ) (h : ∀ x, f x = 2 * x - 1) :
  {x : ℝ | f (f x) ≥ 1} = {x : ℝ | x ≥ 1} := by sorry

end NUMINAMATH_CALUDE_solution_set_of_composite_function_l1543_154382


namespace NUMINAMATH_CALUDE_existence_of_special_sequence_l1543_154345

theorem existence_of_special_sequence :
  ∃ (a : ℕ → ℝ) (x y : ℝ),
    (∀ n, a n ≠ 0) ∧
    (∀ n, a (n + 2) = x * a (n + 1) + y * a n) ∧
    (∀ r > 0, ∃ i j : ℕ, |a i| < r ∧ r < |a j|) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_sequence_l1543_154345


namespace NUMINAMATH_CALUDE_magazine_purchasing_methods_l1543_154365

/-- Represents the number of magazine types priced at 2 yuan -/
def magazines_2yuan : ℕ := 8

/-- Represents the number of magazine types priced at 1 yuan -/
def magazines_1yuan : ℕ := 3

/-- Represents the total amount spent -/
def total_spent : ℕ := 10

/-- Calculates the number of ways to buy magazines -/
def number_of_ways : ℕ := 
  Nat.choose magazines_2yuan 5 + 
  Nat.choose magazines_2yuan 4 * Nat.choose magazines_1yuan 2

theorem magazine_purchasing_methods :
  number_of_ways = 266 := by sorry

end NUMINAMATH_CALUDE_magazine_purchasing_methods_l1543_154365


namespace NUMINAMATH_CALUDE_perpendicular_parallel_implies_parallel_l1543_154384

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)

-- State the theorem
theorem perpendicular_parallel_implies_parallel
  (α β : Plane) (m n : Line)
  (h_distinct_planes : α ≠ β)
  (h_distinct_lines : m ≠ n)
  (h_m_perp_α : perpendicular m α)
  (h_n_perp_β : perpendicular n β)
  (h_α_parallel_β : parallel α β) :
  parallel_lines m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_implies_parallel_l1543_154384


namespace NUMINAMATH_CALUDE_man_half_father_age_l1543_154342

/-- Prove that the number of years it takes for a man to become half his father's age is 5 -/
theorem man_half_father_age (father_age : ℕ) (man_age : ℕ) (years : ℕ) : 
  father_age = 25 →
  man_age = (2 * father_age) / 5 →
  man_age + years = (father_age + years) / 2 →
  years = 5 := by sorry

end NUMINAMATH_CALUDE_man_half_father_age_l1543_154342


namespace NUMINAMATH_CALUDE_monotonic_decreasing_quadratic_l1543_154366

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x - 1

theorem monotonic_decreasing_quadratic (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, ∀ y ∈ Set.Icc 1 2, x < y → f a x > f a y) →
  a ∈ Set.Ici 2 :=
by sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_quadratic_l1543_154366


namespace NUMINAMATH_CALUDE_angle_with_special_supplement_complement_l1543_154327

theorem angle_with_special_supplement_complement : ∃ (x : ℝ), 
  0 < x ∧ x < 180 ∧ (180 - x) = 5 * (90 - x) ∧ x = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_with_special_supplement_complement_l1543_154327


namespace NUMINAMATH_CALUDE_unique_six_digit_number_with_permutation_multiples_l1543_154369

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

def has_distinct_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 6 ∧ digits.toFinset.card = 6

def is_permutation (a b : ℕ) : Prop :=
  (a.digits 10).toFinset = (b.digits 10).toFinset

theorem unique_six_digit_number_with_permutation_multiples :
  ∃! n : ℕ, is_six_digit n ∧ has_distinct_digits n ∧
    (∀ k : Fin 5, is_permutation n ((k + 2) * n)) ∧
    n = 142857 := by sorry

end NUMINAMATH_CALUDE_unique_six_digit_number_with_permutation_multiples_l1543_154369


namespace NUMINAMATH_CALUDE_smallest_k_for_no_real_roots_l1543_154398

theorem smallest_k_for_no_real_roots :
  ∃ (k : ℤ), k = 3 ∧
  (∀ (x : ℝ), 3 * x * (k * x - 5) - 2 * x^2 + 8 ≠ 0) ∧
  (∀ (k' : ℤ), k' < k →
    ∃ (x : ℝ), 3 * x * (k' * x - 5) - 2 * x^2 + 8 = 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_no_real_roots_l1543_154398


namespace NUMINAMATH_CALUDE_translation_of_segment_l1543_154380

/-- Translation of a point in 2D space -/
def translate (p q : ℝ × ℝ) : ℝ × ℝ := (p.1 + q.1, p.2 + q.2)

theorem translation_of_segment (A B C : ℝ × ℝ) :
  A = (-2, 5) →
  B = (-3, 0) →
  C = (3, 7) →
  translate A (5, 2) = C →
  translate B (5, 2) = (2, 2) := by
  sorry

end NUMINAMATH_CALUDE_translation_of_segment_l1543_154380


namespace NUMINAMATH_CALUDE_factorial_ratio_l1543_154388

theorem factorial_ratio : Nat.factorial 12 / Nat.factorial 11 = 12 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l1543_154388


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1543_154390

-- Part 1
theorem simplify_expression_1 (m n : ℝ) :
  (m + n) * (2 * m + n) + n * (m - n) = 2 * m^2 + 4 * m * n := by
  sorry

-- Part 2
theorem simplify_expression_2 (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 3) :
  ((x + 3) / x - 2) / ((x^2 - 9) / (4 * x)) = -4 / (x + 3) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1543_154390


namespace NUMINAMATH_CALUDE_smallest_block_size_l1543_154334

/-- Given a rectangular block with dimensions l, m, and n,
    where (l-1)(m-1)(n-1) = 210, the smallest possible value of l*m*n is 336. -/
theorem smallest_block_size (l m n : ℕ) (h : (l-1)*(m-1)*(n-1) = 210) :
  l*m*n ≥ 336 ∧ ∃ (l' m' n' : ℕ), (l'-1)*(m'-1)*(n'-1) = 210 ∧ l'*m'*n' = 336 := by
  sorry

end NUMINAMATH_CALUDE_smallest_block_size_l1543_154334


namespace NUMINAMATH_CALUDE_polynomial_properties_l1543_154356

/-- Definition of the polynomial -/
def p (x y : ℝ) : ℝ := x * y^3 - x^2 + 7

/-- The degree of the polynomial -/
def degree_p : ℕ := 4

/-- The number of terms in the polynomial -/
def num_terms_p : ℕ := 3

theorem polynomial_properties :
  (degree_p = 4) ∧ (num_terms_p = 3) := by sorry

end NUMINAMATH_CALUDE_polynomial_properties_l1543_154356


namespace NUMINAMATH_CALUDE_least_number_to_add_l1543_154392

def problem (x : ℕ) : Prop :=
  let lcm := 7 * 11 * 13 * 17 * 19
  (∃ k : ℕ, (625573 + x) = k * lcm) ∧
  (∀ y : ℕ, y < x → ¬∃ k : ℕ, (625573 + y) = k * lcm)

theorem least_number_to_add : problem 21073 := by
  sorry

end NUMINAMATH_CALUDE_least_number_to_add_l1543_154392


namespace NUMINAMATH_CALUDE_angle_measure_proof_l1543_154359

theorem angle_measure_proof (x : ℝ) : 
  (180 - x = 4 * x + 7) → x = 173 / 5 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l1543_154359


namespace NUMINAMATH_CALUDE_homework_problem_count_l1543_154337

theorem homework_problem_count (math_pages : ℕ) (reading_pages : ℕ) (problems_per_page : ℕ) : 
  math_pages = 2 → reading_pages = 4 → problems_per_page = 5 →
  (math_pages + reading_pages) * problems_per_page = 30 := by
  sorry

end NUMINAMATH_CALUDE_homework_problem_count_l1543_154337


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l1543_154338

theorem arithmetic_mean_problem (x : ℚ) : 
  ((x + 10) + 20 + 3*x + 17 + (2*x + 6) + (x + 24)) / 6 = 26 → x = 79/7 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l1543_154338


namespace NUMINAMATH_CALUDE_average_marks_math_chem_l1543_154307

/-- Given the total marks in mathematics and physics is 70, and chemistry score is 20 marks more than physics score, prove that the average marks in mathematics and chemistry is 45. -/
theorem average_marks_math_chem (math physics chem : ℕ) : 
  math + physics = 70 → 
  chem = physics + 20 → 
  (math + chem) / 2 = 45 := by
sorry

end NUMINAMATH_CALUDE_average_marks_math_chem_l1543_154307


namespace NUMINAMATH_CALUDE_skew_lines_distance_in_isosceles_triangle_sphere_setup_l1543_154346

/-- Given an isosceles triangle ABC on plane P with two skew lines passing through A and C,
    tangent to a sphere touching P at B, prove the distance between the lines. -/
theorem skew_lines_distance_in_isosceles_triangle_sphere_setup
  (l a r α : ℝ)
  (hl : l > 0)
  (ha : a > 0)
  (hr : r > 0)
  (hα : 0 < α ∧ α < π / 2)
  (h_isosceles : 2 * a ≤ l) :
  ∃ x : ℝ, x = (2 * a * Real.tan α * Real.sqrt (2 * r * l * Real.sin α - (l^2 + r^2) * Real.sin α^2)) /
              Real.sqrt (l^2 - a^2 * Real.cos α^2) :=
by sorry

end NUMINAMATH_CALUDE_skew_lines_distance_in_isosceles_triangle_sphere_setup_l1543_154346


namespace NUMINAMATH_CALUDE_vector_BC_l1543_154318

/-- Given vectors BA and CA in 2D space, prove that vector BC is their difference. -/
theorem vector_BC (BA CA : Fin 2 → ℝ) (h1 : BA = ![1, 2]) (h2 : CA = ![4, 5]) :
  BA - CA = ![-3, -3] := by
  sorry

end NUMINAMATH_CALUDE_vector_BC_l1543_154318


namespace NUMINAMATH_CALUDE_lindas_additional_dimes_l1543_154386

/-- The number of additional dimes Linda's mother gives her -/
def additional_dimes : ℕ := 2

/-- The initial number of dimes Linda has -/
def initial_dimes : ℕ := 2

/-- The initial number of quarters Linda has -/
def initial_quarters : ℕ := 6

/-- The initial number of nickels Linda has -/
def initial_nickels : ℕ := 5

/-- The number of additional quarters Linda's mother gives her -/
def additional_quarters : ℕ := 10

/-- The total number of coins Linda has after receiving additional coins -/
def total_coins : ℕ := 35

theorem lindas_additional_dimes :
  initial_dimes + initial_quarters + initial_nickels +
  additional_dimes + additional_quarters + 2 * initial_nickels = total_coins :=
by sorry

end NUMINAMATH_CALUDE_lindas_additional_dimes_l1543_154386


namespace NUMINAMATH_CALUDE_fair_distributions_is_square_l1543_154389

/-- The number of permutations of 2n elements with all cycles of even length -/
def fair_distributions (n : ℕ) : ℕ := sorry

/-- The double factorial function -/
def double_factorial (n : ℕ) : ℕ := sorry

theorem fair_distributions_is_square (n : ℕ) : 
  fair_distributions n = (double_factorial (2 * n - 1))^2 := by sorry

end NUMINAMATH_CALUDE_fair_distributions_is_square_l1543_154389


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1543_154339

/-- The speed of a boat in still water, given downstream travel information -/
theorem boat_speed_in_still_water :
  ∀ (boat_speed : ℝ) (current_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ),
  current_speed = 6 →
  downstream_distance = 5.2 →
  downstream_time = 1/5 →
  (boat_speed + current_speed) * downstream_time = downstream_distance →
  boat_speed = 20 := by
sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1543_154339


namespace NUMINAMATH_CALUDE_special_function_properties_l1543_154370

/-- A function satisfying f(ab) = af(b) + bf(a) for all a, b ∈ ℝ -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, f (a * b) = a * f b + b * f a

theorem special_function_properties (f : ℝ → ℝ) 
  (h : special_function f) (h_not_zero : ∃ x, f x ≠ 0) :
  (f 0 = 0 ∧ f 1 = 0) ∧ (∀ x : ℝ, f (-x) = -f x) := by
  sorry

end NUMINAMATH_CALUDE_special_function_properties_l1543_154370


namespace NUMINAMATH_CALUDE_homothety_containment_l1543_154304

/-- A convex polygon in R^2 -/
structure ConvexPolygon where
  vertices : Set (ℝ × ℝ)
  convex : Convex ℝ (convexHull ℝ vertices)

/-- Homothety transformation in R^2 -/
def homothety (center : ℝ × ℝ) (ratio : ℝ) (point : ℝ × ℝ) : ℝ × ℝ :=
  (center.1 + ratio * (point.1 - center.1), center.2 + ratio * (point.2 - center.2))

/-- The image of a set under homothety -/
def homotheticImage (center : ℝ × ℝ) (ratio : ℝ) (s : Set (ℝ × ℝ)) : Set (ℝ × ℝ) :=
  {p | ∃ q ∈ s, p = homothety center ratio q}

theorem homothety_containment (P : ConvexPolygon) :
  ∃ O : ℝ × ℝ, homotheticImage O (-1/2) (convexHull ℝ P.vertices) ⊆ convexHull ℝ P.vertices :=
sorry

end NUMINAMATH_CALUDE_homothety_containment_l1543_154304


namespace NUMINAMATH_CALUDE_grid_last_row_digits_l1543_154340

/-- Represents a 3x4 grid of integers -/
def Grid := Matrix (Fin 3) (Fin 4) ℕ

/-- Check if a grid satisfies the given conditions -/
def is_valid_grid (g : Grid) : Prop :=
  (∀ i j, g i j ∈ Finset.range 7 \ {0}) ∧
  (∀ i j₁ j₂, j₁ ≠ j₂ → g i j₁ ≠ g i j₂) ∧
  (∀ i₁ i₂ j, i₁ ≠ i₂ → g i₁ j ≠ g i₂ j) ∧
  g 1 1 = 5 ∧
  g 2 3 = 6

theorem grid_last_row_digits (g : Grid) (h : is_valid_grid g) :
  g 2 0 * 10000 + g 2 1 * 1000 + g 2 2 * 100 + g 2 3 * 10 + g 1 3 = 46123 :=
by sorry

end NUMINAMATH_CALUDE_grid_last_row_digits_l1543_154340


namespace NUMINAMATH_CALUDE_strawberry_pancakes_l1543_154301

theorem strawberry_pancakes (total : ℕ) (blueberry : ℕ) (banana : ℕ) (chocolate : ℕ) 
  (h1 : total = 150)
  (h2 : blueberry = 45)
  (h3 : banana = 60)
  (h4 : chocolate = 25) :
  total - (blueberry + banana + chocolate) = 20 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_pancakes_l1543_154301


namespace NUMINAMATH_CALUDE_class_gpa_calculation_l1543_154397

/-- Calculates the overall GPA of a class given the number of students and their GPAs in three groups -/
def overall_gpa (total_students : ℕ) (group1_students : ℕ) (group1_gpa : ℚ) 
                (group2_students : ℕ) (group2_gpa : ℚ)
                (group3_students : ℕ) (group3_gpa : ℚ) : ℚ :=
  (group1_students * group1_gpa + group2_students * group2_gpa + group3_students * group3_gpa) / total_students

/-- Theorem stating that the overall GPA of the class is 1030/60 -/
theorem class_gpa_calculation :
  overall_gpa 60 20 15 15 17 25 19 = 1030 / 60 := by
  sorry

#eval overall_gpa 60 20 15 15 17 25 19

end NUMINAMATH_CALUDE_class_gpa_calculation_l1543_154397


namespace NUMINAMATH_CALUDE_binomial_coefficient_times_two_l1543_154352

theorem binomial_coefficient_times_two : 2 * (Nat.choose 30 3) = 8120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_times_two_l1543_154352


namespace NUMINAMATH_CALUDE_limit_of_exponential_l1543_154331

theorem limit_of_exponential (a : ℝ) :
  (a > 1 → ∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x > N → a^x > M) ∧
  (0 < a ∧ a < 1 → ∀ ε : ℝ, ε > 0 → ∃ N : ℝ, ∀ x : ℝ, x > N → a^x < ε) :=
by sorry

end NUMINAMATH_CALUDE_limit_of_exponential_l1543_154331


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1543_154310

/-- The function f(x) = ax + 3 --/
def f (a : ℝ) (x : ℝ) : ℝ := a * x + 3

/-- The zero point of f(x) is in the interval (-1, 2) --/
def has_zero_in_interval (a : ℝ) : Prop :=
  ∃ x : ℝ, -1 < x ∧ x < 2 ∧ f a x = 0

/-- The statement is a necessary but not sufficient condition --/
theorem necessary_but_not_sufficient :
  (∀ a : ℝ, 3 < a ∧ a < 4 → has_zero_in_interval a) ∧
  (∃ a : ℝ, has_zero_in_interval a ∧ (a ≤ 3 ∨ 4 ≤ a)) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1543_154310


namespace NUMINAMATH_CALUDE_toy_shop_problem_l1543_154355

/-- Toy shop problem -/
theorem toy_shop_problem 
  (total_A : ℝ) (total_B : ℝ) (diff : ℕ) (ratio : ℝ) 
  (sell_A : ℝ) (sell_B : ℝ) (total_toys : ℕ) (min_profit : ℝ) :
  total_A = 1200 →
  total_B = 1500 →
  diff = 20 →
  ratio = 1.5 →
  sell_A = 12 →
  sell_B = 20 →
  total_toys = 75 →
  min_profit = 300 →
  ∃ (cost_A cost_B : ℝ) (max_A : ℕ),
    -- Part 1: Cost of toys
    cost_A = 10 ∧ 
    cost_B = 15 ∧
    total_A / cost_A - total_B / cost_B = diff ∧
    cost_B = ratio * cost_A ∧
    -- Part 2: Maximum number of type A toys
    max_A = 25 ∧
    ∀ m : ℕ, 
      m ≤ total_toys →
      (sell_A - cost_A) * m + (sell_B - cost_B) * (total_toys - m) ≥ min_profit →
      m ≤ max_A := by
  sorry

end NUMINAMATH_CALUDE_toy_shop_problem_l1543_154355


namespace NUMINAMATH_CALUDE_floor_equation_difference_l1543_154378

theorem floor_equation_difference : ∃ (x y : ℤ), 
  (∀ z : ℤ, ⌊(z : ℚ) / 3⌋ = 102 → z ≤ x) ∧ 
  (⌊(x : ℚ) / 3⌋ = 102) ∧
  (∀ z : ℤ, ⌊(z : ℚ) / 3⌋ = -102 → y ≤ z) ∧ 
  (⌊(y : ℚ) / 3⌋ = -102) ∧
  (x - y = 614) := by
sorry

end NUMINAMATH_CALUDE_floor_equation_difference_l1543_154378


namespace NUMINAMATH_CALUDE_parabola_equation_l1543_154320

/-- A parabola with vertex at the origin and axis along the y-axis passing through (30, -40) with focus at (0, -45/4) has the equation x^2 = -45/2 * y -/
theorem parabola_equation (p : ℝ × ℝ) (f : ℝ × ℝ) :
  p.1 = 30 ∧ p.2 = -40 ∧ f.1 = 0 ∧ f.2 = -45/4 →
  ∀ x y : ℝ, (x^2 = -45/2 * y ↔ (x - f.1)^2 + (y - f.2)^2 = (y - p.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l1543_154320


namespace NUMINAMATH_CALUDE_square_points_sum_l1543_154343

/-- A square with side length 900 and two points on one of its sides. -/
structure SquareWithPoints where
  /-- The side length of the square -/
  side_length : ℝ
  /-- The angle EOF in degrees -/
  angle_EOF : ℝ
  /-- The length of EF -/
  EF_length : ℝ
  /-- The distance BF expressed as p + q√r -/
  BF_distance : ℝ → ℝ → ℝ → ℝ
  /-- Condition that side_length is 900 -/
  h_side_length : side_length = 900
  /-- Condition that angle EOF is 45° -/
  h_angle_EOF : angle_EOF = 45
  /-- Condition that EF length is 400 -/
  h_EF_length : EF_length = 400
  /-- Condition that BF = p + q√r -/
  h_BF_distance : ∀ p q r, BF_distance p q r = p + q * Real.sqrt r

/-- The theorem stating that p + q + r = 307 for the given conditions -/
theorem square_points_sum (s : SquareWithPoints) (p q r : ℕ) 
  (h_positive : p > 0 ∧ q > 0 ∧ r > 0)
  (h_prime : ∀ (k : ℕ), k > 1 → k ^ 2 ∣ r → k.Prime → False) :
  p + q + r = 307 := by
  sorry

end NUMINAMATH_CALUDE_square_points_sum_l1543_154343


namespace NUMINAMATH_CALUDE_trapezoid_circumradii_relation_l1543_154333

-- Define a trapezoid
structure Trapezoid :=
  (A₁ A₂ A₃ A₄ : ℝ × ℝ)

-- Define the diagonal lengths
def diagonal₁ (t : Trapezoid) : ℝ := sorry
def diagonal₂ (t : Trapezoid) : ℝ := sorry

-- Define the circumradius of a triangle formed by three points of the trapezoid
def circumradius (p₁ p₂ p₃ : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem trapezoid_circumradii_relation (t : Trapezoid) :
  let e := diagonal₁ t
  let f := diagonal₂ t
  let r₁ := circumradius t.A₂ t.A₃ t.A₄
  let r₂ := circumradius t.A₁ t.A₃ t.A₄
  let r₃ := circumradius t.A₁ t.A₂ t.A₄
  let r₄ := circumradius t.A₁ t.A₂ t.A₃
  (r₂ + r₄) / e = (r₁ + r₃) / f := by sorry

end NUMINAMATH_CALUDE_trapezoid_circumradii_relation_l1543_154333


namespace NUMINAMATH_CALUDE_average_speed_round_trip_l1543_154344

/-- Given a round trip with outbound speed of 96 mph and return speed of 88 mph,
    prove that the average speed for the entire trip is (2 * 96 * 88) / (96 + 88) mph. -/
theorem average_speed_round_trip (outbound_speed return_speed : ℝ) 
  (h1 : outbound_speed = 96) 
  (h2 : return_speed = 88) : 
  (2 * outbound_speed * return_speed) / (outbound_speed + return_speed) = 
  (2 * 96 * 88) / (96 + 88) :=
by sorry

end NUMINAMATH_CALUDE_average_speed_round_trip_l1543_154344


namespace NUMINAMATH_CALUDE_min_max_sum_l1543_154360

theorem min_max_sum (a b c d e f : ℕ+) (h : a + b + c + d + e + f = 1800) :
  361 ≤ max (a + b) (max (b + c) (max (c + d) (max (d + e) (e + f)))) := by
  sorry

end NUMINAMATH_CALUDE_min_max_sum_l1543_154360


namespace NUMINAMATH_CALUDE_five_digit_square_theorem_l1543_154395

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def remove_first_digit (n : ℕ) : ℕ := n % 10000
def remove_first_two_digits (n : ℕ) : ℕ := n % 1000
def remove_first_three_digits (n : ℕ) : ℕ := n % 100

def is_valid_number (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧
  is_perfect_square n ∧
  is_perfect_square (remove_first_digit n) ∧
  is_perfect_square (remove_first_two_digits n) ∧
  is_perfect_square (remove_first_three_digits n)

theorem five_digit_square_theorem :
  {n : ℕ | is_valid_number n} = {81225, 34225, 27225, 15625, 75625} :=
by sorry

end NUMINAMATH_CALUDE_five_digit_square_theorem_l1543_154395


namespace NUMINAMATH_CALUDE_infinitely_many_primes_of_the_year_l1543_154332

/-- A prime p is a Prime of the Year if there exists a positive integer n such that n^2 + 1 ≡ 0 (mod p^2007) -/
def PrimeOfTheYear (p : ℕ) : Prop :=
  Nat.Prime p ∧ ∃ n : ℕ, n > 0 ∧ (n^2 + 1) % p^2007 = 0

/-- There are infinitely many Primes of the Year -/
theorem infinitely_many_primes_of_the_year :
  ∀ N : ℕ, ∃ p : ℕ, p > N ∧ PrimeOfTheYear p :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_of_the_year_l1543_154332


namespace NUMINAMATH_CALUDE_lcm_gcd_product_24_36_l1543_154325

theorem lcm_gcd_product_24_36 : Nat.lcm 24 36 * Nat.gcd 24 36 = 864 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_24_36_l1543_154325


namespace NUMINAMATH_CALUDE_dans_age_l1543_154324

theorem dans_age (x : ℕ) : (x + 16 = 4 * (x - 8)) → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_dans_age_l1543_154324


namespace NUMINAMATH_CALUDE_company_merger_managers_percentage_l1543_154376

/-- Represents the percentage of managers in Company 2 -/
def m : ℝ := sorry

/-- The total number of employees in Company 1 -/
def F : ℝ := sorry

/-- The total number of employees in Company 2 -/
def S : ℝ := sorry

theorem company_merger_managers_percentage :
  (0.1 * F + m * S = 0.25 * (F + S)) ∧
  (F = 0.25 * (F + S)) ∧
  (0 < F) ∧ (0 < S) ∧
  (0 ≤ m) ∧ (m ≤ 1) ∧
  (m + 0.1 + 0.6 ≤ 1) →
  m = 0.225 := by sorry

end NUMINAMATH_CALUDE_company_merger_managers_percentage_l1543_154376


namespace NUMINAMATH_CALUDE_projection_vector_l1543_154308

def a : Fin 3 → ℝ := ![0, 1, 1]
def b : Fin 3 → ℝ := ![1, 1, 0]

theorem projection_vector :
  let proj_a_b := (a • b) / (a • a) • a
  proj_a_b = ![0, 1/2, 1/2] := by
sorry

end NUMINAMATH_CALUDE_projection_vector_l1543_154308


namespace NUMINAMATH_CALUDE_least_marbles_theorem_l1543_154321

/-- The least number of marbles that can be divided equally among 4, 5, 7, and 8 children
    and is a perfect square. -/
def least_marbles : ℕ := 19600

/-- Predicate to check if a number is divisible by 4, 5, 7, and 8. -/
def divisible_by_4_5_7_8 (n : ℕ) : Prop :=
  n % 4 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0 ∧ n % 8 = 0

/-- Predicate to check if a number is a perfect square. -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem least_marbles_theorem :
  divisible_by_4_5_7_8 least_marbles ∧
  is_perfect_square least_marbles ∧
  ∀ n : ℕ, n < least_marbles →
    ¬(divisible_by_4_5_7_8 n ∧ is_perfect_square n) :=
by sorry

end NUMINAMATH_CALUDE_least_marbles_theorem_l1543_154321


namespace NUMINAMATH_CALUDE_quadratic_point_on_graph_l1543_154381

/-- Given a quadratic function y = -ax² + 2ax + 3 where a > 0,
    if the point (m, 3) lies on the graph and m ≠ 0, then m = 2 -/
theorem quadratic_point_on_graph (a m : ℝ) (ha : a > 0) (hm : m ≠ 0) :
  (3 = -a * m^2 + 2 * a * m + 3) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_point_on_graph_l1543_154381


namespace NUMINAMATH_CALUDE_majorB_higher_admission_rate_male_higher_admission_rate_l1543_154393

/-- Represents the gender of applicants -/
inductive Gender
| Male
| Female

/-- Represents the major of applicants -/
inductive Major
| A
| B

/-- Data structure for application and admission information -/
structure MajorData where
  applicants : Gender → ℕ
  admissionRate : Gender → ℚ

/-- Calculate the weighted average admission rate for a major -/
def weightedAverageAdmissionRate (data : MajorData) : ℚ :=
  let totalApplicants := data.applicants Gender.Male + data.applicants Gender.Female
  let weightedSum := (data.applicants Gender.Male * data.admissionRate Gender.Male) +
                     (data.applicants Gender.Female * data.admissionRate Gender.Female)
  weightedSum / totalApplicants

/-- Calculate the overall admission rate for a gender across both majors -/
def overallAdmissionRate (majorA : MajorData) (majorB : MajorData) (gender : Gender) : ℚ :=
  let totalApplicants := majorA.applicants gender + majorB.applicants gender
  let admittedA := majorA.applicants gender * majorA.admissionRate gender
  let admittedB := majorB.applicants gender * majorB.admissionRate gender
  (admittedA + admittedB) / totalApplicants

/-- Theorem: The weighted average admission rate of Major B is higher than that of Major A -/
theorem majorB_higher_admission_rate (majorA : MajorData) (majorB : MajorData) :
  weightedAverageAdmissionRate majorB > weightedAverageAdmissionRate majorA := by
  sorry

/-- Theorem: The overall admission rate of males is higher than that of females -/
theorem male_higher_admission_rate (majorA : MajorData) (majorB : MajorData) :
  overallAdmissionRate majorA majorB Gender.Male > overallAdmissionRate majorA majorB Gender.Female := by
  sorry

end NUMINAMATH_CALUDE_majorB_higher_admission_rate_male_higher_admission_rate_l1543_154393


namespace NUMINAMATH_CALUDE_art_gallery_pieces_l1543_154330

theorem art_gallery_pieces (total : ℕ) 
  (h1 : total / 3 = total / 3)  -- 1/3 of pieces are on display
  (h2 : (total / 3) / 6 = (total / 3) / 6)  -- 1/6 of displayed pieces are sculptures
  (h3 : (total * 2 / 3) / 3 = (total * 2 / 3) / 3)  -- 1/3 of non-displayed pieces are paintings
  (h4 : total * 2 / 3 * 2 / 3 = 800)  -- 800 sculptures are not on display
  : total = 1800 := by
  sorry

end NUMINAMATH_CALUDE_art_gallery_pieces_l1543_154330


namespace NUMINAMATH_CALUDE_production_days_l1543_154353

theorem production_days (n : ℕ) 
  (h1 : (n * 50 + 115) / (n + 1) = 55) : n = 12 := by
  sorry

end NUMINAMATH_CALUDE_production_days_l1543_154353


namespace NUMINAMATH_CALUDE_class_average_theorem_l1543_154368

/-- Given a class with three groups of students, where:
    1. 25% of the class averages 80% on a test
    2. 50% of the class averages 65% on the test
    3. The remainder of the class averages 90% on the test
    Prove that the overall class average is 75% -/
theorem class_average_theorem (group1_proportion : Real) (group1_average : Real)
                              (group2_proportion : Real) (group2_average : Real)
                              (group3_proportion : Real) (group3_average : Real) :
  group1_proportion = 0.25 →
  group1_average = 0.80 →
  group2_proportion = 0.50 →
  group2_average = 0.65 →
  group3_proportion = 0.25 →
  group3_average = 0.90 →
  group1_proportion + group2_proportion + group3_proportion = 1 →
  group1_proportion * group1_average +
  group2_proportion * group2_average +
  group3_proportion * group3_average = 0.75 := by
  sorry


end NUMINAMATH_CALUDE_class_average_theorem_l1543_154368


namespace NUMINAMATH_CALUDE_remainder_theorem_l1543_154302

theorem remainder_theorem (n : ℤ) (h : n % 7 = 2) : (3 * n - 7) % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1543_154302


namespace NUMINAMATH_CALUDE_steven_amanda_hike_difference_l1543_154329

/-- The number of hikes Camila has gone on -/
def camila_hikes : ℕ := 7

/-- The number of times Amanda has gone hiking compared to Camila -/
def amanda_multiplier : ℕ := 8

/-- The number of hikes Amanda has gone on -/
def amanda_hikes : ℕ := camila_hikes * amanda_multiplier

/-- The number of hikes Camila plans to go on per week -/
def camila_weekly_plan : ℕ := 4

/-- The number of weeks Camila plans to hike to match Steven -/
def camila_weeks_plan : ℕ := 16

/-- The total number of hikes Camila aims for to match Steven -/
def steven_hikes : ℕ := camila_hikes + camila_weekly_plan * camila_weeks_plan

theorem steven_amanda_hike_difference :
  steven_hikes - amanda_hikes = 15 := by
  sorry

end NUMINAMATH_CALUDE_steven_amanda_hike_difference_l1543_154329


namespace NUMINAMATH_CALUDE_movie_theater_seats_l1543_154394

theorem movie_theater_seats (sections : ℕ) (seats_per_section : ℕ) 
  (h1 : sections = 9)
  (h2 : seats_per_section = 30) :
  sections * seats_per_section = 270 := by
sorry

end NUMINAMATH_CALUDE_movie_theater_seats_l1543_154394


namespace NUMINAMATH_CALUDE_sandy_loses_two_marks_l1543_154391

/-- Represents Sandy's math test results -/
structure SandyTest where
  correct_mark : ℕ  -- marks for each correct sum
  total_sums : ℕ    -- total number of sums attempted
  total_marks : ℕ   -- total marks obtained
  correct_sums : ℕ  -- number of correct sums

/-- Calculates the marks lost for each incorrect sum -/
def marks_lost_per_incorrect (test : SandyTest) : ℚ :=
  let correct_marks := test.correct_mark * test.correct_sums
  let incorrect_sums := test.total_sums - test.correct_sums
  let total_marks_lost := correct_marks - test.total_marks
  (total_marks_lost : ℚ) / incorrect_sums

/-- Theorem stating that Sandy loses 2 marks for each incorrect sum -/
theorem sandy_loses_two_marks (test : SandyTest) 
  (h1 : test.correct_mark = 3)
  (h2 : test.total_sums = 30)
  (h3 : test.total_marks = 50)
  (h4 : test.correct_sums = 22) :
  marks_lost_per_incorrect test = 2 := by
  sorry

#eval marks_lost_per_incorrect { correct_mark := 3, total_sums := 30, total_marks := 50, correct_sums := 22 }

end NUMINAMATH_CALUDE_sandy_loses_two_marks_l1543_154391


namespace NUMINAMATH_CALUDE_quadratic_function_range_l1543_154371

/-- A quadratic function with specific properties -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (x + 2) = f (-x + 2)) ∧ 
  f 0 = 3 ∧ 
  f 2 = 1

/-- The range of m for which the function has max 3 and min 1 on [0,m] -/
def ValidRange (f : ℝ → ℝ) (m : ℝ) : Prop :=
  (∀ x ∈ Set.Icc 0 m, f x ≤ 3) ∧
  (∀ x ∈ Set.Icc 0 m, f x ≥ 1) ∧
  (∃ x ∈ Set.Icc 0 m, f x = 3) ∧
  (∃ x ∈ Set.Icc 0 m, f x = 1)

/-- The main theorem -/
theorem quadratic_function_range (f : ℝ → ℝ) (h : QuadraticFunction f) :
  {m | ValidRange f m} = Set.Icc 2 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l1543_154371


namespace NUMINAMATH_CALUDE_gerald_price_is_264_60_verify_hendricks_price_l1543_154348

-- Define the original price of the guitar
def original_price : ℝ := 280

-- Define Hendricks' discount rate
def hendricks_discount_rate : ℝ := 0.15

-- Define Gerald's discount rate
def gerald_discount_rate : ℝ := 0.10

-- Define the sales tax rate
def sales_tax_rate : ℝ := 0.05

-- Define Hendricks' final price
def hendricks_price : ℝ := 250

-- Function to calculate the final price after discount and tax
def calculate_final_price (price : ℝ) (discount_rate : ℝ) (tax_rate : ℝ) : ℝ :=
  price * (1 - discount_rate) * (1 + tax_rate)

-- Theorem stating that Gerald's price is $264.60
theorem gerald_price_is_264_60 :
  calculate_final_price original_price gerald_discount_rate sales_tax_rate = 264.60 := by
  sorry

-- Theorem verifying Hendricks' price
theorem verify_hendricks_price :
  calculate_final_price original_price hendricks_discount_rate sales_tax_rate = hendricks_price := by
  sorry

end NUMINAMATH_CALUDE_gerald_price_is_264_60_verify_hendricks_price_l1543_154348


namespace NUMINAMATH_CALUDE_august_tips_multiple_l1543_154372

/-- 
Proves that if a worker's tips for one month (August) are 0.625 of their total tips for 7 months, 
and the August tips are some multiple of the average tips for the other 6 months, then this multiple is 10.
-/
theorem august_tips_multiple (total_months : ℕ) (other_months : ℕ) (august_ratio : ℝ) (M : ℝ) : 
  total_months = 7 → 
  other_months = 6 → 
  august_ratio = 0.625 →
  M * (1 / other_months : ℝ) * (1 - august_ratio) * total_months = august_ratio →
  M = 10 := by
  sorry

end NUMINAMATH_CALUDE_august_tips_multiple_l1543_154372


namespace NUMINAMATH_CALUDE_donut_theorem_l1543_154326

def donut_problem (initial : ℕ) (eaten : ℕ) (taken : ℕ) : ℕ :=
  let remaining_after_eaten := initial - eaten
  let remaining_after_taken := remaining_after_eaten - taken
  remaining_after_taken - remaining_after_taken / 2

theorem donut_theorem : donut_problem 50 2 4 = 22 := by
  sorry

end NUMINAMATH_CALUDE_donut_theorem_l1543_154326


namespace NUMINAMATH_CALUDE_heracles_age_l1543_154314

/-- Proves that Heracles' age is 10 years old given the conditions of the problem -/
theorem heracles_age : 
  ∀ (heracles_age : ℕ) (audrey_age : ℕ),
  audrey_age = heracles_age + 7 →
  audrey_age + 3 = 2 * heracles_age →
  heracles_age = 10 := by
sorry

end NUMINAMATH_CALUDE_heracles_age_l1543_154314


namespace NUMINAMATH_CALUDE_exists_max_k_l1543_154367

theorem exists_max_k (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0)
  (h : 5 = k^3 * (x^2/y^2 + y^2/x^2) + k^2 * (x/y + y/x)) :
  ∃ k_max : ℝ, k ≤ k_max ∧
    ∀ k' : ℝ, k' > 0 → 
      (∃ x' y' : ℝ, x' > 0 ∧ y' > 0 ∧ 
        5 = k'^3 * (x'^2/y'^2 + y'^2/x'^2) + k'^2 * (x'/y' + y'/x')) →
      k' ≤ k_max :=
sorry

end NUMINAMATH_CALUDE_exists_max_k_l1543_154367


namespace NUMINAMATH_CALUDE_selling_price_for_target_profit_l1543_154375

-- Define the cost price
def cost_price : ℝ := 40

-- Define the function for monthly sales volume based on selling price
def sales_volume (x : ℝ) : ℝ := 1000 - 10 * x

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - cost_price) * sales_volume x

-- Theorem stating the selling prices that result in 8000 yuan profit
theorem selling_price_for_target_profit : 
  ∃ (x : ℝ), (x = 60 ∨ x = 80) ∧ profit x = 8000 := by
  sorry


end NUMINAMATH_CALUDE_selling_price_for_target_profit_l1543_154375


namespace NUMINAMATH_CALUDE_square_difference_l1543_154357

theorem square_difference (n m : ℕ+) (h : n * (4 * n + 1) = m * (5 * m + 1)) :
  ∃ k : ℕ+, n - m = k^2 := by sorry

end NUMINAMATH_CALUDE_square_difference_l1543_154357


namespace NUMINAMATH_CALUDE_symmetric_function_value_l1543_154351

/-- A function with a graph symmetric about the origin -/
def SymmetricAboutOrigin (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The main theorem -/
theorem symmetric_function_value (f : ℝ → ℝ) 
  (h_sym : SymmetricAboutOrigin f)
  (h_pos : ∀ x > 0, f x = 2^x - 3) : 
  f (-2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_function_value_l1543_154351
