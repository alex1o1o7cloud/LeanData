import Mathlib

namespace NUMINAMATH_CALUDE_walkway_area_is_416_l2488_248809

/-- Represents the dimensions of a flower bed -/
structure FlowerBed where
  length : ℝ
  width : ℝ

/-- Represents the configuration of a garden -/
structure Garden where
  rows : ℕ
  columns : ℕ
  bed : FlowerBed
  walkway_width : ℝ

/-- Calculates the total area of walkways in the garden -/
def walkway_area (g : Garden) : ℝ :=
  let total_width := g.columns * g.bed.length + (g.columns + 1) * g.walkway_width
  let total_height := g.rows * g.bed.width + (g.rows + 1) * g.walkway_width
  let total_area := total_width * total_height
  let beds_area := g.rows * g.columns * g.bed.length * g.bed.width
  total_area - beds_area

/-- Theorem stating that the walkway area for the given garden configuration is 416 square feet -/
theorem walkway_area_is_416 (g : Garden) 
  (h_rows : g.rows = 4)
  (h_columns : g.columns = 3)
  (h_bed_length : g.bed.length = 8)
  (h_bed_width : g.bed.width = 3)
  (h_walkway_width : g.walkway_width = 2) :
  walkway_area g = 416 := by
  sorry

end NUMINAMATH_CALUDE_walkway_area_is_416_l2488_248809


namespace NUMINAMATH_CALUDE_equation_solutions_l2488_248864

open Complex

-- Define the set of solutions
def solutions : Set ℂ :=
  {2, -2, 1 + Complex.I * Real.sqrt 3, 1 - Complex.I * Real.sqrt 3,
   -1 + Complex.I * Real.sqrt 3, -1 - Complex.I * Real.sqrt 3}

-- State the theorem
theorem equation_solutions :
  {x : ℂ | x^6 - 64 = 0} = solutions :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2488_248864


namespace NUMINAMATH_CALUDE_position_relationships_l2488_248824

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (parallel_plane : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)

-- Define the lines and planes
variable (a b : Line)
variable (α β γ : Plane)

-- State the theorem
theorem position_relationships :
  (∀ (a b : Line) (α : Plane), 
    parallel a b → subset b α → parallel_plane a α) = False ∧ 
  (∀ (a b : Line) (α : Plane), 
    parallel a b → parallel_plane a α → parallel_plane b α) = False ∧
  (∀ (a b : Line) (α β γ : Plane),
    intersect α β a → subset b γ → parallel_plane b β → subset a γ → parallel a b) = True :=
sorry

end NUMINAMATH_CALUDE_position_relationships_l2488_248824


namespace NUMINAMATH_CALUDE_cube_volume_surface_area_l2488_248819

theorem cube_volume_surface_area (x : ℝ) : 
  (∃ (s : ℝ), s > 0 ∧ s^3 = 8*x ∧ 6*s^2 = 2*x) → x = 1728 :=
by sorry

end NUMINAMATH_CALUDE_cube_volume_surface_area_l2488_248819


namespace NUMINAMATH_CALUDE_dvd_rental_cost_l2488_248842

theorem dvd_rental_cost (num_dvds : ℕ) (cost_per_dvd : ℚ) : 
  num_dvds = 4 → cost_per_dvd = 6/5 → num_dvds * cost_per_dvd = 24/5 := by
  sorry

end NUMINAMATH_CALUDE_dvd_rental_cost_l2488_248842


namespace NUMINAMATH_CALUDE_resulting_polynomial_degree_is_eight_l2488_248860

/-- The degree of the polynomial resulting from the given operations -/
def resultingPolynomialDegree : ℕ :=
  let expr1 := fun x : ℝ => x^4
  let expr2 := fun x : ℝ => x^2 - 1/x^2
  let expr3 := fun x : ℝ => 1 - 3/x + 3/x^2
  let squaredExpr2 := fun x : ℝ => (expr2 x)^2
  let result := fun x : ℝ => (expr1 x) * (squaredExpr2 x) * (expr3 x)
  8

/-- Theorem stating that the degree of the resulting polynomial is 8 -/
theorem resulting_polynomial_degree_is_eight :
  resultingPolynomialDegree = 8 := by sorry

end NUMINAMATH_CALUDE_resulting_polynomial_degree_is_eight_l2488_248860


namespace NUMINAMATH_CALUDE_parabola_abc_value_l2488_248830

-- Define the parabola
def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem parabola_abc_value (a b c : ℝ) :
  -- Vertex condition
  (∀ x, parabola a b c x = a * (x - 4)^2 + 2) →
  -- Point (2, 0) lies on the parabola
  parabola a b c 2 = 0 →
  -- Conclusion: abc = 12
  a * b * c = 12 := by
  sorry

end NUMINAMATH_CALUDE_parabola_abc_value_l2488_248830


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2488_248841

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 2 * a 4 = 16 →
  (a 2 * a 3 * a 4 = 64 ∨ a 2 * a 3 * a 4 = -64) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2488_248841


namespace NUMINAMATH_CALUDE_minimum_gift_cost_l2488_248893

structure Store :=
  (name : String)
  (mom_gift : Nat)
  (dad_gift : Nat)
  (brother_gift : Nat)
  (sister_gift : Nat)
  (shopping_time : Nat)

def stores : List Store := [
  ⟨"Romashka", 1000, 750, 930, 850, 35⟩,
  ⟨"Oduvanchik", 1050, 790, 910, 800, 30⟩,
  ⟨"Nezabudka", 980, 810, 925, 815, 40⟩,
  ⟨"Landysh", 1100, 755, 900, 820, 25⟩
]

def travel_time : Nat := 30
def start_time : Nat := 16 * 60 + 35
def close_time : Nat := 20 * 60

def is_valid_shopping_plan (plan : List Store) : Bool :=
  let total_time := plan.foldl (fun acc s => acc + s.shopping_time) 0 + (plan.length - 1) * travel_time
  start_time + total_time ≤ close_time

def gift_cost (plan : List Store) : Nat :=
  plan.foldl (fun acc s => acc + s.mom_gift + s.dad_gift + s.brother_gift + s.sister_gift) 0

theorem minimum_gift_cost :
  ∃ (plan : List Store),
    plan.length = 4 ∧
    (∀ s : Store, s ∈ plan → s ∈ stores) ∧
    is_valid_shopping_plan plan ∧
    gift_cost plan = 3435 ∧
    (∀ other_plan : List Store,
      other_plan.length = 4 →
      (∀ s : Store, s ∈ other_plan → s ∈ stores) →
      is_valid_shopping_plan other_plan →
      gift_cost other_plan ≥ 3435) :=
sorry

end NUMINAMATH_CALUDE_minimum_gift_cost_l2488_248893


namespace NUMINAMATH_CALUDE_unique_postal_codes_exist_l2488_248801

def PostalCode := Fin 6 → Fin 7

def validDigits (code : PostalCode) : Prop :=
  ∀ i : Fin 6, code i < 7 ∧ code i ≠ 4

def distinctDigits (code : PostalCode) : Prop :=
  ∀ i j : Fin 6, i ≠ j → code i ≠ code j

def matchingPositions (code1 code2 : PostalCode) : Nat :=
  (List.range 6).filter (λ i => code1 i = code2 i) |>.length

def A : PostalCode := λ i => [3, 2, 0, 6, 5, 1][i]
def B : PostalCode := λ i => [1, 0, 5, 2, 6, 3][i]
def C : PostalCode := λ i => [6, 1, 2, 3, 0, 5][i]
def D : PostalCode := λ i => [3, 1, 6, 2, 5, 0][i]

theorem unique_postal_codes_exist : 
  ∃! (M N : PostalCode), 
    validDigits M ∧ validDigits N ∧
    distinctDigits M ∧ distinctDigits N ∧
    M ≠ N ∧
    (matchingPositions A M = 2 ∧ matchingPositions A N = 2) ∧
    (matchingPositions B M = 2 ∧ matchingPositions B N = 2) ∧
    (matchingPositions C M = 2 ∧ matchingPositions C N = 2) ∧
    (matchingPositions D M = 3 ∧ matchingPositions D N = 3) := by
  sorry

end NUMINAMATH_CALUDE_unique_postal_codes_exist_l2488_248801


namespace NUMINAMATH_CALUDE_line_point_value_l2488_248817

/-- Given a line containing points (2, 9), (15, m), and (35, 4), prove that m = 232/33 -/
theorem line_point_value (m : ℚ) : 
  (∃ (line : ℝ → ℝ), line 2 = 9 ∧ line 15 = m ∧ line 35 = 4) → m = 232/33 := by
  sorry

end NUMINAMATH_CALUDE_line_point_value_l2488_248817


namespace NUMINAMATH_CALUDE_solution_of_square_eq_zero_l2488_248895

theorem solution_of_square_eq_zero :
  ∀ x : ℝ, x^2 = 0 ↔ x = 0 := by sorry

end NUMINAMATH_CALUDE_solution_of_square_eq_zero_l2488_248895


namespace NUMINAMATH_CALUDE_pythagorean_triple_square_l2488_248836

theorem pythagorean_triple_square (a b c : ℕ+) (h : a^2 + b^2 = c^2) :
  ∃ m n : ℤ, (1/2 : ℚ) * ((c : ℚ) - (a : ℚ)) * ((c : ℚ) - (b : ℚ)) = (n^2 * (m - n)^2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_triple_square_l2488_248836


namespace NUMINAMATH_CALUDE_race_distance_race_distance_proof_l2488_248853

/-- The total distance of a race where:
    - The ratio of speeds of contestants A and B is 2:4
    - A has a start of 300 m
    - A wins by 100 m
-/
theorem race_distance : ℝ :=
  let speed_ratio : ℚ := 2 / 4
  let head_start : ℝ := 300
  let winning_margin : ℝ := 100
  500

theorem race_distance_proof (speed_ratio : ℚ) (head_start winning_margin : ℝ) :
  speed_ratio = 2 / 4 →
  head_start = 300 →
  winning_margin = 100 →
  race_distance = 500 := by
  sorry

end NUMINAMATH_CALUDE_race_distance_race_distance_proof_l2488_248853


namespace NUMINAMATH_CALUDE_ny_mets_fans_count_l2488_248885

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

end NUMINAMATH_CALUDE_ny_mets_fans_count_l2488_248885


namespace NUMINAMATH_CALUDE_andrey_numbers_l2488_248894

/-- Represents a five-digit number as a tuple of five natural numbers, each between 0 and 9 inclusive. -/
def FiveDigitNumber := (Nat × Nat × Nat × Nat × Nat)

/-- Checks if a given FiveDigitNumber is valid (all digits between 0 and 9). -/
def isValidFiveDigitNumber (n : FiveDigitNumber) : Prop :=
  let (a, b, c, d, e) := n
  0 ≤ a ∧ a ≤ 9 ∧
  0 ≤ b ∧ b ≤ 9 ∧
  0 ≤ c ∧ c ≤ 9 ∧
  0 ≤ d ∧ d ≤ 9 ∧
  0 ≤ e ∧ e ≤ 9

/-- Converts a FiveDigitNumber to its numerical value. -/
def toNumber (n : FiveDigitNumber) : Nat :=
  let (a, b, c, d, e) := n
  10000 * a + 1000 * b + 100 * c + 10 * d + e

/-- Checks if two FiveDigitNumbers differ by exactly two digits. -/
def differByTwoDigits (n1 n2 : FiveDigitNumber) : Prop :=
  let (a1, b1, c1, d1, e1) := n1
  let (a2, b2, c2, d2, e2) := n2
  (a1 ≠ a2 ∧ b1 = b2 ∧ c1 = c2 ∧ d1 = d2 ∧ e1 = e2) ∨
  (a1 = a2 ∧ b1 ≠ b2 ∧ c1 = c2 ∧ d1 = d2 ∧ e1 = e2) ∨
  (a1 = a2 ∧ b1 = b2 ∧ c1 ≠ c2 ∧ d1 = d2 ∧ e1 = e2) ∨
  (a1 = a2 ∧ b1 = b2 ∧ c1 = c2 ∧ d1 ≠ d2 ∧ e1 = e2) ∨
  (a1 = a2 ∧ b1 = b2 ∧ c1 = c2 ∧ d1 = d2 ∧ e1 ≠ e2)

/-- Checks if a FiveDigitNumber contains a zero. -/
def containsZero (n : FiveDigitNumber) : Prop :=
  let (a, b, c, d, e) := n
  a = 0 ∨ b = 0 ∨ c = 0 ∨ d = 0 ∨ e = 0

theorem andrey_numbers (n1 n2 : FiveDigitNumber) 
  (h1 : isValidFiveDigitNumber n1)
  (h2 : isValidFiveDigitNumber n2)
  (h3 : differByTwoDigits n1 n2)
  (h4 : toNumber n1 + toNumber n2 = 111111) :
  containsZero n1 ∨ containsZero n2 := by
  sorry

end NUMINAMATH_CALUDE_andrey_numbers_l2488_248894


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_specific_l2488_248869

def arithmetic_sequence_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_sum_specific : 
  arithmetic_sequence_sum 1 2 17 = 289 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_specific_l2488_248869


namespace NUMINAMATH_CALUDE_square_properties_l2488_248839

theorem square_properties (a b : ℤ) (h : 2*a^2 + a = 3*b^2 + b) :
  ∃ (x y : ℤ), (a - b = x^2) ∧ (2*a + 2*b + 1 = y^2) := by
  sorry

end NUMINAMATH_CALUDE_square_properties_l2488_248839


namespace NUMINAMATH_CALUDE_intersection_parameter_value_l2488_248816

/-- Given two lines that intersect at a specific x-coordinate, 
    prove the value of the parameter m in the first line equation. -/
theorem intersection_parameter_value 
  (x : ℝ) 
  (h1 : x = -7.5) 
  (h2 : ∃ y : ℝ, 3 * x - y = m ∧ -0.4 * x + y = 3) : 
  m = -22.5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_parameter_value_l2488_248816


namespace NUMINAMATH_CALUDE_robert_reading_capacity_l2488_248806

/-- Represents the number of books Robert can read in a given time -/
def books_read (reading_speed : ℕ) (available_time : ℕ) (book_type1_pages : ℕ) (book_type2_pages : ℕ) : ℕ :=
  let books_type1 := available_time / (book_type1_pages / reading_speed)
  let books_type2 := available_time / (book_type2_pages / reading_speed)
  books_type1 + books_type2

/-- Theorem stating that Robert can read 5 books in 6 hours given the specified conditions -/
theorem robert_reading_capacity : 
  books_read 120 6 240 360 = 5 := by
  sorry

end NUMINAMATH_CALUDE_robert_reading_capacity_l2488_248806


namespace NUMINAMATH_CALUDE_max_min_product_l2488_248846

theorem max_min_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (sum_eq : a + b + c = 12) (sum_prod : a * b + b * c + c * a = 30) :
  ∃ (m : ℝ), m = min (a * b) (min (b * c) (c * a)) ∧ m ≤ 6 ∧
  ∀ (m' : ℝ), m' = min (a * b) (min (b * c) (c * a)) → m' ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_max_min_product_l2488_248846


namespace NUMINAMATH_CALUDE_total_pears_picked_l2488_248863

theorem total_pears_picked (alyssa nancy michael : ℕ) 
  (h1 : alyssa = 42)
  (h2 : nancy = 17)
  (h3 : michael = 31) :
  alyssa + nancy + michael = 90 := by
  sorry

end NUMINAMATH_CALUDE_total_pears_picked_l2488_248863


namespace NUMINAMATH_CALUDE_stamp_collection_theorem_l2488_248879

/-- Calculates the total value of a stamp collection given the following conditions:
    - The total number of stamps in the collection
    - The number of stamps in a subset of the collection
    - The total value of the subset of stamps
    Assumes that all stamps have the same value. -/
def stamp_collection_value (total_stamps : ℕ) (subset_stamps : ℕ) (subset_value : ℚ) : ℚ :=
  (total_stamps : ℚ) * (subset_value / subset_stamps)

/-- Proves that a collection of 18 stamps, where 6 stamps are worth 18 dollars,
    has a total value of 54 dollars. -/
theorem stamp_collection_theorem :
  stamp_collection_value 18 6 18 = 54 := by
  sorry

end NUMINAMATH_CALUDE_stamp_collection_theorem_l2488_248879


namespace NUMINAMATH_CALUDE_blanket_price_problem_l2488_248844

theorem blanket_price_problem (price1 price2 unknown_price avg_price : ℚ) :
  price1 = 100 →
  price2 = 150 →
  avg_price = 150 →
  (5 * price1 + 5 * price2 + 2 * unknown_price) / 12 = avg_price →
  unknown_price = 275 := by
sorry

end NUMINAMATH_CALUDE_blanket_price_problem_l2488_248844


namespace NUMINAMATH_CALUDE_stairs_ratio_l2488_248857

theorem stairs_ratio (samir_stairs veronica_stairs : ℕ) (ratio : ℚ) : 
  samir_stairs = 318 →
  veronica_stairs = (samir_stairs / 2 : ℚ) * ratio →
  samir_stairs + veronica_stairs = 495 →
  ratio = 177 / 159 := by
sorry

end NUMINAMATH_CALUDE_stairs_ratio_l2488_248857


namespace NUMINAMATH_CALUDE_two_pairs_satisfy_equation_l2488_248826

theorem two_pairs_satisfy_equation :
  ∃ (x₁ y₁ x₂ y₂ : ℕ),
    (x₁ = 2 ∧ y₁ = 2 ∧ 2 * x₁^3 = y₁^4) ∧
    (x₂ = 32 ∧ y₂ = 16 ∧ 2 * x₂^3 = y₂^4) :=
by sorry

end NUMINAMATH_CALUDE_two_pairs_satisfy_equation_l2488_248826


namespace NUMINAMATH_CALUDE_power_of_four_in_expression_l2488_248896

theorem power_of_four_in_expression (x : ℕ) : 
  (2 * x + 5 + 2 = 29) → x = 11 := by sorry

end NUMINAMATH_CALUDE_power_of_four_in_expression_l2488_248896


namespace NUMINAMATH_CALUDE_find_m_l2488_248851

theorem find_m (A B : Set ℕ) (m : ℕ) : 
  A = {1, 3, m} → 
  B = {3, 4} → 
  A ∪ B = {1, 2, 3, 4} → 
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_find_m_l2488_248851


namespace NUMINAMATH_CALUDE_monotonic_sufficient_not_necessary_for_maximum_l2488_248852

theorem monotonic_sufficient_not_necessary_for_maximum 
  (f : ℝ → ℝ) (hf : ContinuousOn f (Set.Icc 0 1)) :
  (MonotoneOn f (Set.Icc 0 1) → ∃ x ∈ Set.Icc 0 1, ∀ y ∈ Set.Icc 0 1, f y ≤ f x) ∧
  ¬(∃ x ∈ Set.Icc 0 1, ∀ y ∈ Set.Icc 0 1, f y ≤ f x → MonotoneOn f (Set.Icc 0 1)) :=
sorry

end NUMINAMATH_CALUDE_monotonic_sufficient_not_necessary_for_maximum_l2488_248852


namespace NUMINAMATH_CALUDE_grid_game_winner_l2488_248823

/-- Represents a player in the game -/
inductive Player
| First
| Second

/-- Represents the game state on a rectangular grid -/
structure GridGame where
  m : ℕ  -- number of rows
  n : ℕ  -- number of columns

/-- Determines the winner of the game based on the grid dimensions -/
def winner (game : GridGame) : Player :=
  if (game.m + game.n) % 2 = 0 then Player.Second else Player.First

/-- Theorem stating the winning condition for the grid game -/
theorem grid_game_winner (game : GridGame) :
  (winner game = Player.Second ↔ (game.m + game.n) % 2 = 0) ∧
  (winner game = Player.First ↔ (game.m + game.n) % 2 = 1) :=
sorry

end NUMINAMATH_CALUDE_grid_game_winner_l2488_248823


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2488_248812

theorem complex_equation_solution (z : ℂ) :
  (Complex.I * 3 + Real.sqrt 3) * z = Complex.I * 3 →
  z = 3 / 4 + Complex.I * (Real.sqrt 3 / 4) := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2488_248812


namespace NUMINAMATH_CALUDE_distance_difference_l2488_248832

-- Define the dimensions
def street_width : ℕ := 25
def block_length : ℕ := 450
def block_width : ℕ := 350
def alley_width : ℕ := 25

-- Define Sarah's path
def sarah_long_side : ℕ := block_length + alley_width
def sarah_short_side : ℕ := block_width

-- Define Sam's path
def sam_long_side : ℕ := block_length + 2 * street_width
def sam_short_side : ℕ := block_width + 2 * street_width

-- Calculate total distances
def sarah_total : ℕ := 2 * sarah_long_side + 2 * sarah_short_side
def sam_total : ℕ := 2 * sam_long_side + 2 * sam_short_side

-- Theorem to prove
theorem distance_difference :
  sam_total - sarah_total = 150 := by
  sorry

end NUMINAMATH_CALUDE_distance_difference_l2488_248832


namespace NUMINAMATH_CALUDE_polynomial_remainder_l2488_248877

theorem polynomial_remainder (x : ℝ) : 
  let p := fun x : ℝ => 8*x^4 - 10*x^3 + 16*x^2 - 18*x + 5
  let d := fun x : ℝ => 4*x - 8
  (p x) % (d x) = 81 := by
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l2488_248877


namespace NUMINAMATH_CALUDE_T_increasing_T_not_perfect_square_non_perfect_square_in_T_T_2012th_term_l2488_248838

/-- The sequence of positive integers that are not perfect squares -/
def T : ℕ → ℕ := sorry

/-- T is increasing -/
theorem T_increasing : ∀ n : ℕ, T n < T (n + 1) := sorry

/-- T consists of non-perfect squares -/
theorem T_not_perfect_square : ∀ n : ℕ, ¬ ∃ m : ℕ, T n = m^2 := sorry

/-- Every non-perfect square is in T -/
theorem non_perfect_square_in_T : ∀ k : ℕ, (¬ ∃ m : ℕ, k = m^2) → ∃ n : ℕ, T n = k := sorry

/-- The 2012th term of T is 2057 -/
theorem T_2012th_term : T 2011 = 2057 := sorry

end NUMINAMATH_CALUDE_T_increasing_T_not_perfect_square_non_perfect_square_in_T_T_2012th_term_l2488_248838


namespace NUMINAMATH_CALUDE_digit_difference_in_base_d_l2488_248847

/-- Represents a digit in a given base --/
def Digit (d : ℕ) := { n : ℕ // n < d }

/-- Represents a two-digit number in a given base --/
def TwoDigitNumber (d : ℕ) (A B : Digit d) : ℕ := A.val * d + B.val

theorem digit_difference_in_base_d 
  (d : ℕ) 
  (h_d : d > 7) 
  (A B : Digit d) 
  (h_sum : TwoDigitNumber d A B + TwoDigitNumber d A A = 175) : 
  A.val - B.val = 2 := by
sorry

end NUMINAMATH_CALUDE_digit_difference_in_base_d_l2488_248847


namespace NUMINAMATH_CALUDE_plumber_pipe_cost_l2488_248866

theorem plumber_pipe_cost 
  (copper_length : ℝ) 
  (plastic_length : ℝ) 
  (copper_cost_per_meter : ℝ) 
  (plastic_cost_per_meter : ℝ) 
  (discount_rate : ℝ) 
  (h1 : copper_length = 10) 
  (h2 : plastic_length = 15) 
  (h3 : copper_cost_per_meter = 5) 
  (h4 : plastic_cost_per_meter = 3) 
  (h5 : discount_rate = 0.1) : 
  (copper_length * copper_cost_per_meter + plastic_length * plastic_cost_per_meter) * (1 - discount_rate) = 85.50 := by
  sorry

end NUMINAMATH_CALUDE_plumber_pipe_cost_l2488_248866


namespace NUMINAMATH_CALUDE_two_six_digit_squares_decomposable_l2488_248891

/-- A function that checks if a number is a two-digit square -/
def isTwoDigitSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, 4 ≤ k ∧ k ≤ 9 ∧ n = k^2

/-- A function that checks if a 6-digit number can be decomposed into three two-digit squares -/
def isDecomposableIntoThreeTwoDigitSquares (n : ℕ) : Prop :=
  ∃ a b c : ℕ,
    isTwoDigitSquare a ∧
    isTwoDigitSquare b ∧
    isTwoDigitSquare c ∧
    n = a * 10000 + b * 100 + c

/-- The main theorem stating that there are exactly two 6-digit perfect squares
    that can be decomposed into three two-digit perfect squares -/
theorem two_six_digit_squares_decomposable :
  ∃! (s : Finset ℕ),
    s.card = 2 ∧
    (∀ n ∈ s, 100000 ≤ n ∧ n < 1000000) ∧
    (∀ n ∈ s, ∃ k : ℕ, n = k^2) ∧
    (∀ n ∈ s, isDecomposableIntoThreeTwoDigitSquares n) :=
  sorry

end NUMINAMATH_CALUDE_two_six_digit_squares_decomposable_l2488_248891


namespace NUMINAMATH_CALUDE_max_students_l2488_248867

/-- Represents the relation of knowing each other among students -/
def knows (n : ℕ) := Fin n → Fin n → Prop

/-- For any group of 3 students, at least 2 know each other -/
def at_least_two_know (n : ℕ) (k : knows n) : Prop :=
  ∀ a b c : Fin n, a ≠ b ∧ b ≠ c ∧ a ≠ c → k a b ∨ k b c ∨ k a c

/-- For any group of 4 students, at least 2 do not know each other -/
def at_least_two_dont_know (n : ℕ) (k : knows n) : Prop :=
  ∀ a b c d : Fin n, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d → 
    ¬(k a b) ∨ ¬(k a c) ∨ ¬(k a d) ∨ ¬(k b c) ∨ ¬(k b d) ∨ ¬(k c d)

/-- The maximum number of students satisfying the conditions is 8 -/
theorem max_students : 
  ∀ n : ℕ, (∃ k : knows n, at_least_two_know n k ∧ at_least_two_dont_know n k) → n ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_max_students_l2488_248867


namespace NUMINAMATH_CALUDE_binary_to_decimal_110011_l2488_248825

/-- Converts a list of binary digits to a decimal number -/
def binaryToDecimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of the number we want to convert -/
def binaryNumber : List Bool := [true, true, false, false, true, true]

/-- Theorem stating that the binary number 110011₂ is equal to the decimal number 51 -/
theorem binary_to_decimal_110011 : binaryToDecimal binaryNumber = 51 := by
  sorry

end NUMINAMATH_CALUDE_binary_to_decimal_110011_l2488_248825


namespace NUMINAMATH_CALUDE_water_tank_capacity_l2488_248834

theorem water_tank_capacity : ∀ (C : ℝ),
  (∃ (x : ℝ), x / C = 1 / 3 ∧ (x + 6) / C = 1 / 2) →
  C = 36 := by
sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l2488_248834


namespace NUMINAMATH_CALUDE_tetrahedron_properties_l2488_248892

def A₁ : ℝ × ℝ × ℝ := (2, -1, 2)
def A₂ : ℝ × ℝ × ℝ := (1, 2, -1)
def A₃ : ℝ × ℝ × ℝ := (3, 2, 1)
def A₄ : ℝ × ℝ × ℝ := (-4, 2, 5)

def tetrahedron_volume (A B C D : ℝ × ℝ × ℝ) : ℝ := sorry

def tetrahedron_height (A B C D : ℝ × ℝ × ℝ) : ℝ := sorry

theorem tetrahedron_properties :
  tetrahedron_volume A₁ A₂ A₃ A₄ = 11 ∧
  tetrahedron_height A₄ A₁ A₂ A₃ = 3 * Real.sqrt (11 / 2) := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_properties_l2488_248892


namespace NUMINAMATH_CALUDE_coinciding_directrices_l2488_248897

/-- Given a hyperbola and a parabola with coinciding directrices, prove that p = 3 -/
theorem coinciding_directrices (p : ℝ) : p > 0 → ∃ (x y : ℝ),
  (x^2 / 3 - y^2 = 1 ∧ y^2 = 2*p*x ∧ 
   (x = -3/2 ∨ x = 3/2) ∧ x = -p/2) → p = 3 := by
  sorry

end NUMINAMATH_CALUDE_coinciding_directrices_l2488_248897


namespace NUMINAMATH_CALUDE_pascal_triangle_15th_row_4th_number_l2488_248821

def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem pascal_triangle_15th_row_4th_number : binomial 15 3 = 455 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_15th_row_4th_number_l2488_248821


namespace NUMINAMATH_CALUDE_value_added_to_number_l2488_248875

theorem value_added_to_number : ∃ v : ℝ, 3 * (9 + v) = 9 + 24 ∧ v = 2 := by
  sorry

end NUMINAMATH_CALUDE_value_added_to_number_l2488_248875


namespace NUMINAMATH_CALUDE_shaded_area_of_concentric_circles_l2488_248890

theorem shaded_area_of_concentric_circles (r1 r2 r3 : ℝ) (shaded unshaded : ℝ) : 
  r1 = 4 → r2 = 5 → r3 = 6 →
  shaded + unshaded = π * (r1^2 + r2^2 + r3^2) →
  shaded = (3/7) * unshaded →
  shaded = (1617 * π) / 70 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_of_concentric_circles_l2488_248890


namespace NUMINAMATH_CALUDE_amanda_keeps_33_candy_bars_l2488_248882

/-- Calculates the number of candy bars Amanda keeps for herself after a series of events --/
def amanda_candy_bars : ℕ :=
  let initial := 7
  let after_first_give := initial - (initial / 3)
  let after_buying := after_first_give + 30
  let after_second_give := after_buying - (after_buying / 4)
  let after_gift := after_second_give + 15
  let final := after_gift - ((15 * 3) / 5)
  final

/-- Theorem stating that Amanda keeps 33 candy bars for herself --/
theorem amanda_keeps_33_candy_bars : amanda_candy_bars = 33 := by
  sorry

end NUMINAMATH_CALUDE_amanda_keeps_33_candy_bars_l2488_248882


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_smallest_odd_primes_l2488_248803

theorem smallest_four_digit_divisible_by_smallest_odd_primes : 
  ∃ (n : ℕ), 
    (1000 ≤ n) ∧ 
    (n < 10000) ∧ 
    (n % 3 = 0) ∧ 
    (n % 5 = 0) ∧ 
    (n % 7 = 0) ∧ 
    (n % 11 = 0) ∧
    (∀ m : ℕ, 
      (1000 ≤ m) ∧ 
      (m < 10000) ∧ 
      (m % 3 = 0) ∧ 
      (m % 5 = 0) ∧ 
      (m % 7 = 0) ∧ 
      (m % 11 = 0) → 
      n ≤ m) ∧
    n = 1155 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_smallest_odd_primes_l2488_248803


namespace NUMINAMATH_CALUDE_parabola_focus_line_l2488_248813

/-- Given a parabola and a line passing through its focus, prove the value of p -/
theorem parabola_focus_line (p : ℝ) (A B : ℝ × ℝ) : 
  p > 0 →  -- p is positive
  (∀ x y, y = x^2 / (2*p)) →  -- equation of parabola
  (A.1^2 = 2*p*A.2) →  -- A is on the parabola
  (B.1^2 = 2*p*B.2) →  -- B is on the parabola
  (A.1 + B.1 = 2) →  -- midpoint of AB has x-coordinate 1
  ((A.2 + B.2) / 2 = 1) →  -- midpoint of AB has y-coordinate 1
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 36 →  -- length of AB is 6
  p = 2 := by
sorry

end NUMINAMATH_CALUDE_parabola_focus_line_l2488_248813


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l2488_248855

theorem solve_exponential_equation :
  ∃! y : ℝ, (64 : ℝ)^(3*y) = (16 : ℝ)^(4*y - 5) :=
by
  -- The unique solution is y = -10
  use -10
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l2488_248855


namespace NUMINAMATH_CALUDE_square_1225_identity_l2488_248804

theorem square_1225_identity (x : ℤ) (h : x^2 = 1225) : (x + 2) * (x - 2) = 1221 := by
  sorry

end NUMINAMATH_CALUDE_square_1225_identity_l2488_248804


namespace NUMINAMATH_CALUDE_shoe_price_is_50_l2488_248811

/-- Represents the original price of a pair of shoes -/
def original_shoe_price : ℝ := sorry

/-- Represents the discount rate for shoes -/
def shoe_discount : ℝ := 0.4

/-- Represents the discount rate for dresses -/
def dress_discount : ℝ := 0.2

/-- Represents the number of pairs of shoes bought -/
def num_shoes : ℕ := 2

/-- Represents the original price of the dress -/
def dress_price : ℝ := 100

/-- Represents the total amount spent -/
def total_spent : ℝ := 140

/-- Theorem stating that the original price of a pair of shoes is $50 -/
theorem shoe_price_is_50 : 
  (num_shoes : ℝ) * original_shoe_price * (1 - shoe_discount) + 
  dress_price * (1 - dress_discount) = total_spent → 
  original_shoe_price = 50 := by
  sorry

end NUMINAMATH_CALUDE_shoe_price_is_50_l2488_248811


namespace NUMINAMATH_CALUDE_nested_radical_floor_l2488_248886

theorem nested_radical_floor (y : ℝ) (B : ℤ) : 
  y > 0 → y^2 = 10 + y → B = ⌊10 + y⌋ → B = 13 := by sorry

end NUMINAMATH_CALUDE_nested_radical_floor_l2488_248886


namespace NUMINAMATH_CALUDE_second_pipe_rate_l2488_248887

def well_capacity : ℝ := 1200
def first_pipe_rate : ℝ := 48
def filling_time : ℝ := 5

theorem second_pipe_rate : 
  ∃ (rate : ℝ), 
    rate * filling_time + first_pipe_rate * filling_time = well_capacity ∧ 
    rate = 192 :=
by sorry

end NUMINAMATH_CALUDE_second_pipe_rate_l2488_248887


namespace NUMINAMATH_CALUDE_data_value_proof_l2488_248865

theorem data_value_proof (a b c : ℝ) 
  (h1 : a + b = c)
  (h2 : b = 3 * a)
  (h3 : a + b + c = 96) :
  a = 12 := by
  sorry

end NUMINAMATH_CALUDE_data_value_proof_l2488_248865


namespace NUMINAMATH_CALUDE_set_equality_implies_values_l2488_248899

/-- Given two sets A and B, prove that if they are equal and have the specified form,
    then x = 2 and y = 2 -/
theorem set_equality_implies_values (x y : ℝ) : 
  ({x, y^2, 1} : Set ℝ) = ({1, 2*x, y} : Set ℝ) → x = 2 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_values_l2488_248899


namespace NUMINAMATH_CALUDE_cf_length_l2488_248850

/-- A rectangle ABCD with point F such that C is on DF and B is on DE -/
structure SpecialRectangle where
  /-- Point A -/
  A : ℝ × ℝ
  /-- Point B -/
  B : ℝ × ℝ
  /-- Point C -/
  C : ℝ × ℝ
  /-- Point D -/
  D : ℝ × ℝ
  /-- Point E -/
  E : ℝ × ℝ
  /-- Point F -/
  F : ℝ × ℝ
  /-- ABCD is a rectangle -/
  is_rectangle : (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2
  /-- AB = 8 -/
  ab_length : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 64
  /-- BC = 6 -/
  bc_length : (B.1 - C.1)^2 + (B.2 - C.2)^2 = 36
  /-- C is on DF -/
  c_on_df : ∃ t : ℝ, C = (1 - t) • D + t • F
  /-- B is the quarter-point of DE -/
  b_quarter_point : B = (3/4) • D + (1/4) • E
  /-- DEF is a right triangle -/
  def_right_triangle : (D.1 - E.1) * (E.1 - F.1) + (D.2 - E.2) * (E.2 - F.2) = 0

/-- The length of CF is 12 -/
theorem cf_length (rect : SpecialRectangle) : 
  (rect.C.1 - rect.F.1)^2 + (rect.C.2 - rect.F.2)^2 = 144 := by
  sorry

end NUMINAMATH_CALUDE_cf_length_l2488_248850


namespace NUMINAMATH_CALUDE_survey_result_l2488_248859

theorem survey_result (total_surveyed : ℕ) 
  (believed_spread_diseases : ℕ) 
  (believed_flu : ℕ) : 
  (believed_spread_diseases : ℝ) / total_surveyed = 0.905 →
  (believed_flu : ℝ) / believed_spread_diseases = 0.503 →
  believed_flu = 26 →
  total_surveyed = 57 := by
sorry

end NUMINAMATH_CALUDE_survey_result_l2488_248859


namespace NUMINAMATH_CALUDE_remaining_area_in_square_configuration_l2488_248861

/-- The area of the remaining space in a square configuration -/
theorem remaining_area_in_square_configuration : 
  ∀ (s : ℝ) (small_square : ℝ) (rect1_length rect1_width : ℝ) (rect2_length rect2_width : ℝ),
  s = 4 →
  small_square = 1 →
  rect1_length = 2 ∧ rect1_width = 1 →
  rect2_length = 1 ∧ rect2_width = 2 →
  s^2 - (small_square^2 + rect1_length * rect1_width + rect2_length * rect2_width) = 11 :=
by sorry

end NUMINAMATH_CALUDE_remaining_area_in_square_configuration_l2488_248861


namespace NUMINAMATH_CALUDE_not_even_if_symmetric_to_x_squared_l2488_248880

-- Define the function g(x) = x^2 for x ≥ 0
def g (x : ℝ) : ℝ := x^2

-- Define symmetry with respect to y = x
def symmetricToYEqualsX (f g : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ g y = x

-- Define an even function
def isEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Theorem statement
theorem not_even_if_symmetric_to_x_squared (f : ℝ → ℝ) 
  (h_sym : symmetricToYEqualsX f g) : ¬ (isEven f) := by
  sorry

end NUMINAMATH_CALUDE_not_even_if_symmetric_to_x_squared_l2488_248880


namespace NUMINAMATH_CALUDE_sqrt2_power0_plus_neg2_power3_l2488_248802

theorem sqrt2_power0_plus_neg2_power3 : (Real.sqrt 2) ^ 0 + (-2) ^ 3 = -7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt2_power0_plus_neg2_power3_l2488_248802


namespace NUMINAMATH_CALUDE_least_marbles_theorem_l2488_248884

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

end NUMINAMATH_CALUDE_least_marbles_theorem_l2488_248884


namespace NUMINAMATH_CALUDE_mystery_number_proof_l2488_248837

theorem mystery_number_proof : ∃ x : ℕ, x * 48 = 173 * 240 ∧ x = 865 := by
  sorry

end NUMINAMATH_CALUDE_mystery_number_proof_l2488_248837


namespace NUMINAMATH_CALUDE_circle_condition_chord_length_l2488_248878

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*m*x - 4*y + 5*m = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  2*x - y + 1 = 0

-- Theorem for the range of m
theorem circle_condition (m : ℝ) :
  (∃ x y, circle_equation x y m) ↔ (m < 1 ∨ m > 4) :=
sorry

-- Theorem for the chord length
theorem chord_length :
  let m : ℝ := -2
  let center : ℝ × ℝ := (-2, 2)
  let radius : ℝ := 3 * Real.sqrt 2
  let d : ℝ := Real.sqrt 5
  2 * Real.sqrt (radius^2 - d^2) = 2 * Real.sqrt 13 :=
sorry

end NUMINAMATH_CALUDE_circle_condition_chord_length_l2488_248878


namespace NUMINAMATH_CALUDE_monic_quartic_with_given_roots_l2488_248829

-- Define the polynomial
def p (x : ℝ) : ℝ := x^4 - 10*x^3 + 17*x^2 + 18*x - 12

-- Theorem statement
theorem monic_quartic_with_given_roots :
  -- The polynomial is monic
  (∀ x, p x = x^4 + (-10)*x^3 + 17*x^2 + 18*x + (-12)) ∧
  -- The polynomial has rational coefficients
  (∃ a b c d : ℚ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + d) ∧
  -- 3+√5 is a root
  p (3 + Real.sqrt 5) = 0 ∧
  -- 2-√7 is a root
  p (2 - Real.sqrt 7) = 0 :=
by sorry

end NUMINAMATH_CALUDE_monic_quartic_with_given_roots_l2488_248829


namespace NUMINAMATH_CALUDE_skew_iff_a_neq_zero_l2488_248815

def line1 (a t : ℝ) : Fin 4 → ℝ := fun i =>
  match i with
  | 0 => 1 + 2*t
  | 1 => 3 + 4*t
  | 2 => 0 + 1*t
  | 3 => a + 3*t

def line2 (u : ℝ) : Fin 4 → ℝ := fun i =>
  match i with
  | 0 => 3 + 4*u
  | 1 => 4 + 5*u
  | 2 => 1 + 2*u
  | 3 => 0 + 1*u

def are_skew (a : ℝ) : Prop :=
  ∀ t u : ℝ, line1 a t ≠ line2 u

theorem skew_iff_a_neq_zero (a : ℝ) :
  are_skew a ↔ a ≠ 0 := by sorry

end NUMINAMATH_CALUDE_skew_iff_a_neq_zero_l2488_248815


namespace NUMINAMATH_CALUDE_price_reduction_equation_l2488_248840

theorem price_reduction_equation (x : ℝ) : 
  (∀ (original_price final_price : ℝ),
    original_price = 100 ∧ 
    final_price = 81 ∧ 
    final_price = original_price * (1 - x)^2) →
  100 * (1 - x)^2 = 81 :=
by sorry

end NUMINAMATH_CALUDE_price_reduction_equation_l2488_248840


namespace NUMINAMATH_CALUDE_S_is_circle_l2488_248828

-- Define the set of complex numbers satisfying the condition
def S : Set ℂ := {z : ℂ | Complex.abs (z - Complex.I) = Complex.abs (3 - 4 * Complex.I)}

-- Theorem stating that S is a circle
theorem S_is_circle : ∃ (c : ℂ) (r : ℝ), S = {z : ℂ | Complex.abs (z - c) = r} := by
  sorry

end NUMINAMATH_CALUDE_S_is_circle_l2488_248828


namespace NUMINAMATH_CALUDE_original_price_calculation_l2488_248870

theorem original_price_calculation (reduced_price : ℝ) (reduction_percentage : ℝ) 
  (h1 : reduced_price = 6)
  (h2 : reduction_percentage = 0.25)
  (h3 : reduced_price = reduction_percentage * original_price) :
  original_price = 24 := by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l2488_248870


namespace NUMINAMATH_CALUDE_simplify_sqrt_difference_l2488_248835

theorem simplify_sqrt_difference : 
  (Real.sqrt 528 / Real.sqrt 64) - (Real.sqrt 242 / Real.sqrt 121) = 1.461 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_difference_l2488_248835


namespace NUMINAMATH_CALUDE_locus_of_circle_center_l2488_248868

-- Define the two given circles
def circle1 (x y : ℝ) : Prop := (x + 3)^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9

-- Define the condition for a circle to be tangent to both given circles
def is_tangent_to_both (cx cy r : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ),
    circle1 x1 y1 ∧ circle2 x2 y2 ∧
    (cx - x1)^2 + (cy - y1)^2 = (r + 1)^2 ∧
    (cx - x2)^2 + (cy - y2)^2 = (r + 3)^2

-- State the theorem
theorem locus_of_circle_center :
  ∀ (x y : ℝ), x < 0 →
    (∃ (r : ℝ), is_tangent_to_both x y r) ↔ x^2 - y^2/8 = 1 :=
sorry

end NUMINAMATH_CALUDE_locus_of_circle_center_l2488_248868


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l2488_248856

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 138) 
  (h2 : a*b + b*c + c*a = 131) : 
  a + b + c = 20 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l2488_248856


namespace NUMINAMATH_CALUDE_alcohol_percentage_after_dilution_l2488_248876

/-- Calculates the alcohol percentage in a mixture after adding water -/
theorem alcohol_percentage_after_dilution
  (initial_volume : ℝ)
  (initial_alcohol_percentage : ℝ)
  (water_added : ℝ)
  (h1 : initial_volume = 9)
  (h2 : initial_alcohol_percentage = 57)
  (h3 : water_added = 3) :
  let alcohol_volume := initial_volume * (initial_alcohol_percentage / 100)
  let total_volume := initial_volume + water_added
  let new_alcohol_percentage := (alcohol_volume / total_volume) * 100
  new_alcohol_percentage = 42.75 := by
sorry

end NUMINAMATH_CALUDE_alcohol_percentage_after_dilution_l2488_248876


namespace NUMINAMATH_CALUDE_square_difference_120_pairs_l2488_248862

theorem square_difference_120_pairs :
  ∃! (pairs : Finset (ℕ × ℕ)), 
    (∀ (m n : ℕ), (m, n) ∈ pairs ↔ m > n ∧ m^2 - n^2 = 120) ∧
    pairs.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_120_pairs_l2488_248862


namespace NUMINAMATH_CALUDE_value_of_a_l2488_248818

theorem value_of_a (a b c d : ℤ) 
  (eq1 : 2 * a + 2 = b)
  (eq2 : 2 * b + 2 = c)
  (eq3 : 2 * c + 2 = d)
  (eq4 : 2 * d + 2 = 62) : 
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_l2488_248818


namespace NUMINAMATH_CALUDE_expression_value_l2488_248849

theorem expression_value (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 3) :
  x + (x^4 / y^3) + (y^4 / x^3) + y = 27665/27 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2488_248849


namespace NUMINAMATH_CALUDE_wang_lei_pastries_l2488_248874

/-- Represents the number of pastries in a large box -/
def large_box_pastries : ℕ := 32

/-- Represents the number of pastries in a small box -/
def small_box_pastries : ℕ := 15

/-- Represents the cost of a large box in yuan -/
def large_box_cost : ℚ := 85.6

/-- Represents the cost of a small box in yuan -/
def small_box_cost : ℚ := 46.8

/-- Represents the total amount spent by Wang Lei in yuan -/
def total_spent : ℚ := 654

/-- Represents the total number of boxes bought by Wang Lei -/
def total_boxes : ℕ := 9

/-- Theorem stating that Wang Lei got 237 pastries -/
theorem wang_lei_pastries : 
  ∃ (large_boxes small_boxes : ℕ), 
    large_boxes + small_boxes = total_boxes ∧
    large_box_cost * large_boxes + small_box_cost * small_boxes = total_spent ∧
    large_box_pastries * large_boxes + small_box_pastries * small_boxes = 237 :=
by sorry

end NUMINAMATH_CALUDE_wang_lei_pastries_l2488_248874


namespace NUMINAMATH_CALUDE_equal_angles_point_exists_l2488_248871

-- Define the structure for a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the function to calculate the angle between three points
def angle (p1 p2 p3 : Point2D) : ℝ := sorry

-- Define the theorem
theorem equal_angles_point_exists (A B C D : Point2D) 
  (h_collinear : ∃ (t : ℝ), B.x = A.x + t * (D.x - A.x) ∧ 
                             B.y = A.y + t * (D.y - A.y) ∧ 
                             C.x = A.x + t * (D.x - A.x) ∧ 
                             C.y = A.y + t * (D.y - A.y)) :
  ∃ (M : Point2D), angle A M B = angle B M C ∧ angle B M C = angle C M D := by
  sorry

end NUMINAMATH_CALUDE_equal_angles_point_exists_l2488_248871


namespace NUMINAMATH_CALUDE_interest_calculation_l2488_248854

/-- Compound interest calculation --/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Interest earned calculation --/
def interest_earned (total : ℝ) (principal : ℝ) : ℝ :=
  total - principal

theorem interest_calculation (P : ℝ) (h1 : P > 0) :
  let rate : ℝ := 0.08
  let time : ℕ := 2
  let total : ℝ := 19828.80
  compound_interest P rate time = total →
  interest_earned total P = 2828.80 := by
sorry


end NUMINAMATH_CALUDE_interest_calculation_l2488_248854


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l2488_248873

/-- The distance between foci of an ellipse with given parameters -/
theorem ellipse_foci_distance (a b : ℝ) (h1 : a = 10) (h2 : b = 8) (h3 : a > b) :
  2 * Real.sqrt (a^2 - b^2) = 12 := by
  sorry

#check ellipse_foci_distance

end NUMINAMATH_CALUDE_ellipse_foci_distance_l2488_248873


namespace NUMINAMATH_CALUDE_max_points_in_configuration_l2488_248883

/-- A configuration of points in the plane with associated real numbers -/
structure PointConfiguration where
  n : ℕ
  points : Fin n → ℝ × ℝ
  radii : Fin n → ℝ
  distance_property : ∀ (i j : Fin n), i ≠ j →
    Real.sqrt ((points i).1 - (points j).1)^2 + ((points i).2 - (points j).2)^2 = radii i + radii j

/-- The maximum number of points in a valid configuration is 4 -/
theorem max_points_in_configuration :
  (∃ (c : PointConfiguration), c.n = 4) ∧
  (∀ (c : PointConfiguration), c.n ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_max_points_in_configuration_l2488_248883


namespace NUMINAMATH_CALUDE_sqrt_D_irrational_l2488_248810

theorem sqrt_D_irrational (x : ℤ) : 
  let a : ℤ := x
  let b : ℤ := x + 2
  let c : ℤ := a * b
  let d : ℤ := b + c
  let D : ℤ := a^2 + b^2 + c^2 + d^2
  Irrational (Real.sqrt D) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_D_irrational_l2488_248810


namespace NUMINAMATH_CALUDE_root_relation_iff_p_values_l2488_248820

-- Define the quadratic equation
def quadratic_equation (p : ℝ) (x : ℝ) : ℝ := x^2 + p*x + 2*p

-- Define the condition for one root being three times the other
def root_condition (p : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), 
    quadratic_equation p x₁ = 0 ∧ 
    quadratic_equation p x₂ = 0 ∧ 
    x₂ = 3 * x₁

-- Theorem statement
theorem root_relation_iff_p_values :
  ∀ p : ℝ, root_condition p ↔ (p = 0 ∨ p = 32/3) :=
by sorry

end NUMINAMATH_CALUDE_root_relation_iff_p_values_l2488_248820


namespace NUMINAMATH_CALUDE_copy_pages_with_discount_l2488_248805

/-- Calculates the number of pages that can be copied given a certain amount of cents,
    considering a discount where for every 100 pages, an additional 10 pages are free. -/
def pages_copied (cents : ℕ) : ℕ :=
  let base_pages := (cents * 5) / 10
  let free_pages := (base_pages / 100) * 10
  base_pages + free_pages

/-- Proves that 5000 cents allows copying 2750 pages with the given pricing and discount. -/
theorem copy_pages_with_discount :
  pages_copied 5000 = 2750 := by
  sorry

end NUMINAMATH_CALUDE_copy_pages_with_discount_l2488_248805


namespace NUMINAMATH_CALUDE_cos_negative_75_degrees_l2488_248872

theorem cos_negative_75_degrees :
  Real.cos (-(75 * π / 180)) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_75_degrees_l2488_248872


namespace NUMINAMATH_CALUDE_B_power_15_minus_3_power_14_l2488_248889

def B : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 0, 2]

theorem B_power_15_minus_3_power_14 :
  B^15 - 3 • B^14 = !![0, 4; 0, -1] := by sorry

end NUMINAMATH_CALUDE_B_power_15_minus_3_power_14_l2488_248889


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2488_248845

theorem inequality_system_solution (x : ℝ) :
  (3 * x - (x - 2) ≥ 6) ∧ (x + 1 > (4 * x - 1) / 3) → 2 ≤ x ∧ x < 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2488_248845


namespace NUMINAMATH_CALUDE_absolute_value_equation_product_l2488_248833

theorem absolute_value_equation_product (x : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   (abs (2 * x₁) + 4 = 38) ∧ 
   (abs (2 * x₂) + 4 = 38) ∧ 
   x₁ * x₂ = -289) := by
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_product_l2488_248833


namespace NUMINAMATH_CALUDE_multiple_without_zero_l2488_248814

/-- A function that checks if a natural number contains the digit 0 in its decimal representation -/
def containsZero (n : ℕ) : Prop :=
  ∃ (k : ℕ), n % (10^(k+1)) / (10^k) = 0

theorem multiple_without_zero (n : ℕ) (h : n % 10 ≠ 0) :
  ∃ (k : ℕ), k % n = 0 ∧ ¬containsZero k := by
  sorry

end NUMINAMATH_CALUDE_multiple_without_zero_l2488_248814


namespace NUMINAMATH_CALUDE_words_per_page_l2488_248881

theorem words_per_page (total_pages : ℕ) (max_words_per_page : ℕ) (total_words_mod : ℕ) :
  total_pages = 150 →
  max_words_per_page = 120 →
  total_words_mod = 270 →
  ∃ (words_per_page : ℕ),
    words_per_page ≤ max_words_per_page ∧
    (total_pages * words_per_page) % 221 = total_words_mod % 221 ∧
    words_per_page = 107 :=
by sorry

end NUMINAMATH_CALUDE_words_per_page_l2488_248881


namespace NUMINAMATH_CALUDE_only_pairD_not_opposite_l2488_248898

-- Define a structure for a pair of quantities
structure QuantityPair where
  first : String
  second : String

-- Define the function to check if a pair has opposite meanings
def hasOppositeMeanings (pair : QuantityPair) : Bool :=
  match pair with
  | ⟨"Income of 200 yuan", "expenditure of 20 yuan"⟩ => true
  | ⟨"Rise of 10 meters", "fall of 7 meters"⟩ => true
  | ⟨"Exceeding 0.05 mm", "falling short of 0.03 m"⟩ => true
  | ⟨"Increase of 2 years", "decrease of 2 liters"⟩ => false
  | _ => false

-- Define the pairs
def pairA : QuantityPair := ⟨"Income of 200 yuan", "expenditure of 20 yuan"⟩
def pairB : QuantityPair := ⟨"Rise of 10 meters", "fall of 7 meters"⟩
def pairC : QuantityPair := ⟨"Exceeding 0.05 mm", "falling short of 0.03 m"⟩
def pairD : QuantityPair := ⟨"Increase of 2 years", "decrease of 2 liters"⟩

-- Theorem statement
theorem only_pairD_not_opposite : 
  (hasOppositeMeanings pairA = true) ∧ 
  (hasOppositeMeanings pairB = true) ∧ 
  (hasOppositeMeanings pairC = true) ∧ 
  (hasOppositeMeanings pairD = false) :=
sorry

end NUMINAMATH_CALUDE_only_pairD_not_opposite_l2488_248898


namespace NUMINAMATH_CALUDE_truck_rental_percentage_l2488_248888

/-- The percentage of trucks returned given the total number of trucks,
    the number of trucks rented out, and the number of trucks returned -/
def percentage_returned (total : ℕ) (rented : ℕ) (returned : ℕ) : ℚ :=
  (returned : ℚ) / (rented : ℚ) * 100

theorem truck_rental_percentage (total : ℕ) (rented : ℕ) (returned : ℕ)
  (h_total : total = 24)
  (h_rented : rented = total)
  (h_returned : returned ≥ 12) :
  percentage_returned total rented returned = 50 := by
sorry

end NUMINAMATH_CALUDE_truck_rental_percentage_l2488_248888


namespace NUMINAMATH_CALUDE_number_of_book_combinations_l2488_248831

-- Define the number of books and the number to choose
def total_books : ℕ := 15
def books_to_choose : ℕ := 3

-- Theorem statement
theorem number_of_book_combinations :
  Nat.choose total_books books_to_choose = 455 := by
  sorry

end NUMINAMATH_CALUDE_number_of_book_combinations_l2488_248831


namespace NUMINAMATH_CALUDE_quadratic_root_l2488_248800

theorem quadratic_root (a b c : ℝ) (h : a ≠ 0 ∧ b + c ≠ 0) :
  let f : ℝ → ℝ := λ x => a * (b + c) * x^2 - b * (c + a) * x - c * (a + b)
  (f (-1) = 0) → (f (c * (a + b) / (a * (b + c))) = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_l2488_248800


namespace NUMINAMATH_CALUDE_perimeter_MNO_value_l2488_248843

/-- A right prism with regular hexagonal bases -/
structure HexagonalPrism where
  height : ℝ
  base_side_length : ℝ

/-- A point on an edge of the prism -/
structure EdgePoint where
  fraction : ℝ  -- Fraction of the edge length from the base

/-- Triangle MNO formed by three points on different edges of the prism -/
structure TriangleMNO where
  prism : HexagonalPrism
  m : EdgePoint
  n : EdgePoint
  o : EdgePoint

/-- Calculate the perimeter of triangle MNO -/
def perimeter_MNO (t : TriangleMNO) : ℝ :=
  sorry

theorem perimeter_MNO_value (t : TriangleMNO) 
  (h1 : t.prism.height = 20)
  (h2 : t.prism.base_side_length = 10)
  (h3 : t.m.fraction = 1/3)
  (h4 : t.n.fraction = 1/4)
  (h5 : t.o.fraction = 1/2) :
  perimeter_MNO t = 10 + Real.sqrt (925/9) + 5 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_perimeter_MNO_value_l2488_248843


namespace NUMINAMATH_CALUDE_verandah_area_l2488_248827

/-- The area of a verandah surrounding a rectangular room -/
theorem verandah_area (room_length room_width verandah_width : ℝ) :
  room_length = 15 ∧ room_width = 12 ∧ verandah_width = 2 →
  (room_length + 2 * verandah_width) * (room_width + 2 * verandah_width) -
  room_length * room_width = 124 := by sorry

end NUMINAMATH_CALUDE_verandah_area_l2488_248827


namespace NUMINAMATH_CALUDE_matthew_rebecca_age_difference_l2488_248858

/-- Represents the ages of three children and their properties --/
structure ChildrenAges where
  freddy : ℕ
  matthew : ℕ
  rebecca : ℕ
  total_age : freddy + matthew + rebecca = 35
  freddy_age : freddy = 15
  matthew_younger : matthew = freddy - 4
  matthew_older : matthew > rebecca

/-- Theorem stating that Matthew is 2 years older than Rebecca --/
theorem matthew_rebecca_age_difference (ages : ChildrenAges) : ages.matthew = ages.rebecca + 2 := by
  sorry

end NUMINAMATH_CALUDE_matthew_rebecca_age_difference_l2488_248858


namespace NUMINAMATH_CALUDE_mistake_in_report_l2488_248808

def reported_numbers : List Nat := [3, 3, 3, 3, 3, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6]

def num_boys : Nat := 7
def num_girls : Nat := 8

theorem mistake_in_report :
  (List.sum reported_numbers) % 2 = 0 →
  ¬(∃ (boys_sum : Nat), 
    boys_sum * 2 = List.sum reported_numbers ∧
    boys_sum = num_girls * (List.sum reported_numbers / (num_boys + num_girls))) :=
by sorry

end NUMINAMATH_CALUDE_mistake_in_report_l2488_248808


namespace NUMINAMATH_CALUDE_repair_cost_equals_profit_l2488_248807

/-- Proves that the repair cost equals the profit under given conditions --/
theorem repair_cost_equals_profit (original_cost : ℝ) : 
  let repair_cost := 0.1 * original_cost
  let selling_price := 1.2 * original_cost
  let profit := selling_price - (original_cost + repair_cost)
  profit = 1100 ∧ profit / original_cost = 0.2 → repair_cost = 1100 := by
sorry

end NUMINAMATH_CALUDE_repair_cost_equals_profit_l2488_248807


namespace NUMINAMATH_CALUDE_polynomial_irreducibility_l2488_248822

/-- A polynomial of the form x^n + 5x^(n-1) + 3 where n > 1 is irreducible over the integers -/
theorem polynomial_irreducibility (n : ℕ) (hn : n > 1) :
  Irreducible (Polynomial.monomial n 1 + Polynomial.monomial (n-1) 5 + Polynomial.C 3 : Polynomial ℤ) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_irreducibility_l2488_248822


namespace NUMINAMATH_CALUDE_subtracted_number_l2488_248848

theorem subtracted_number (a b x : ℕ) : 
  (a : ℚ) / b = 6 / 5 →
  (a - x : ℚ) / (b - x) = 5 / 4 →
  a - b = 5 →
  x = 5 := by
sorry

end NUMINAMATH_CALUDE_subtracted_number_l2488_248848
