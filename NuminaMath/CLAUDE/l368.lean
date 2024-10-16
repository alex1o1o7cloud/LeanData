import Mathlib

namespace NUMINAMATH_CALUDE_parallelogram_side_sum_l368_36864

/-- Given a parallelogram with sides measuring 7, 9, 8y-1, and 2x+3 units consecutively,
    prove that x + y = 4 -/
theorem parallelogram_side_sum (x y : ℝ) : 
  (7 : ℝ) = 8*y - 1 → (9 : ℝ) = 2*x + 3 → x + y = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_side_sum_l368_36864


namespace NUMINAMATH_CALUDE_alley_width_equals_ladder_height_l368_36813

/-- Proof that the width of an alley equals the height of a ladder against one wall 
    when it forms specific angles with both walls. -/
theorem alley_width_equals_ladder_height 
  (l : ℝ) -- length of the ladder
  (x y : ℝ) -- heights on the walls
  (h_x_pos : x > 0)
  (h_y_pos : y > 0)
  (h_angle_Q : x / w = Real.sqrt 3) -- tan 60° = √3
  (h_angle_R : y / w = 1) -- tan 45° = 1
  : w = y :=
sorry

end NUMINAMATH_CALUDE_alley_width_equals_ladder_height_l368_36813


namespace NUMINAMATH_CALUDE_constant_term_expansion_l368_36853

theorem constant_term_expansion (x : ℝ) (x_neq_zero : x ≠ 0) :
  ∃ (f : ℝ → ℝ), (∀ y, f y = (y + 4/y - 4)^3) ∧
  (∃ c, ∀ y ≠ 0, f y = c + y * (f y - c) / y) ∧
  c = -160 := by
sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l368_36853


namespace NUMINAMATH_CALUDE_seating_arrangement_l368_36862

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def total_arrangements (n : ℕ) : ℕ := factorial n

def restricted_arrangements (n : ℕ) (k : ℕ) : ℕ := 
  factorial (n - k + 1) * factorial k

theorem seating_arrangement (n : ℕ) (k : ℕ) 
  (h1 : n = 8) (h2 : k = 4) : 
  total_arrangements n - restricted_arrangements n k = 37440 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangement_l368_36862


namespace NUMINAMATH_CALUDE_f_upper_bound_and_g_monotonicity_l368_36838

noncomputable section

def f (x : ℝ) : ℝ := 2 * Real.log x + 1

def g (a x : ℝ) : ℝ := (f x - f a) / (x - a)

theorem f_upper_bound_and_g_monotonicity :
  (∃ c : ℝ, ∀ x : ℝ, x > 0 → f x ≤ 2 * x + c) ∧
  (∀ c : ℝ, c < -1 → ∃ x : ℝ, x > 0 ∧ f x > 2 * x + c) ∧
  (∀ a : ℝ, a > 0 → 
    (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < a → g a x₁ > g a x₂) ∧
    (∀ x₁ x₂ : ℝ, a < x₁ ∧ x₁ < x₂ → g a x₁ > g a x₂)) :=
sorry

end NUMINAMATH_CALUDE_f_upper_bound_and_g_monotonicity_l368_36838


namespace NUMINAMATH_CALUDE_initial_amount_80_leads_to_128_each_l368_36886

/-- Represents the amount of money each person has at each stage -/
structure Money where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Performs the first transaction where A gives to B and C -/
def transaction1 (m : Money) : Money :=
  { a := m.a - m.b - m.c,
    b := 2 * m.b,
    c := 2 * m.c }

/-- Performs the second transaction where B gives to A and C -/
def transaction2 (m : Money) : Money :=
  { a := 2 * m.a,
    b := m.b - m.a - m.c,
    c := 2 * m.c }

/-- Performs the third transaction where C gives to A and B -/
def transaction3 (m : Money) : Money :=
  { a := 2 * m.a,
    b := 2 * m.b,
    c := m.c - m.a - m.b }

/-- The main theorem stating that if the initial amount for A is 80,
    after all transactions, each person will have 128 cents -/
theorem initial_amount_80_leads_to_128_each (m : Money)
    (h_total : m.a + m.b + m.c = 128 + 128 + 128)
    (h_initial_a : m.a = 80) :
    let m1 := transaction1 m
    let m2 := transaction2 m1
    let m3 := transaction3 m2
    m3.a = 128 ∧ m3.b = 128 ∧ m3.c = 128 := by
  sorry


end NUMINAMATH_CALUDE_initial_amount_80_leads_to_128_each_l368_36886


namespace NUMINAMATH_CALUDE_remainder_of_9876543210_div_101_l368_36854

theorem remainder_of_9876543210_div_101 : 9876543210 % 101 = 68 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_9876543210_div_101_l368_36854


namespace NUMINAMATH_CALUDE_product_of_integers_l368_36884

theorem product_of_integers (a b : ℕ+) : 
  a + b = 30 → 
  2 * (a * b) + 14 * a = 5 * b + 290 → 
  a * b = 104 := by
sorry

end NUMINAMATH_CALUDE_product_of_integers_l368_36884


namespace NUMINAMATH_CALUDE_apple_pie_pieces_l368_36818

/-- Calculates the number of pieces each pie is cut into -/
def piecesPer (totalApples : ℕ) (numPies : ℕ) (applesPerSlice : ℕ) : ℕ :=
  (totalApples / numPies) / applesPerSlice

/-- Proves that each pie is cut into 6 pieces given the problem conditions -/
theorem apple_pie_pieces : 
  piecesPer (4 * 12) 4 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_apple_pie_pieces_l368_36818


namespace NUMINAMATH_CALUDE_article_largeFont_wordsPerPage_l368_36817

/-- Calculates the number of words per page in the large font given the article constraints. -/
def largeFont_wordsPerPage (totalWords smallFont_wordsPerPage totalPages largeFont_pages : ℕ) : ℕ :=
  let smallFont_pages := totalPages - largeFont_pages
  let smallFont_words := smallFont_pages * smallFont_wordsPerPage
  let largeFont_words := totalWords - smallFont_words
  largeFont_words / largeFont_pages

/-- Proves that the number of words per page in the large font is 1800 given the article constraints. -/
theorem article_largeFont_wordsPerPage :
  largeFont_wordsPerPage 48000 2400 21 4 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_article_largeFont_wordsPerPage_l368_36817


namespace NUMINAMATH_CALUDE_beka_jackson_miles_difference_l368_36882

/-- The difference in miles flown between two people -/
def miles_difference (beka_miles jackson_miles : ℕ) : ℕ :=
  beka_miles - jackson_miles

/-- Theorem stating the difference in miles flown between Beka and Jackson -/
theorem beka_jackson_miles_difference :
  miles_difference 873 563 = 310 := by
  sorry

end NUMINAMATH_CALUDE_beka_jackson_miles_difference_l368_36882


namespace NUMINAMATH_CALUDE_nine_circles_problem_l368_36879

/-- Represents a 3x3 grid of numbers -/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- Checks if all numbers from 1 to 9 are used exactly once in the grid -/
def is_valid_grid (g : Grid) : Prop :=
  ∀ n : Fin 9, ∃! (i j : Fin 3), g i j = n

/-- Represents a triangle in the grid by its three vertex coordinates -/
structure Triangle where
  v1 : Fin 3 × Fin 3
  v2 : Fin 3 × Fin 3
  v3 : Fin 3 × Fin 3

/-- List of all 7 triangles in the grid -/
def triangles : List Triangle := sorry

/-- Checks if the sum of numbers at the vertices of a triangle is 15 -/
def triangle_sum_is_15 (g : Grid) (t : Triangle) : Prop :=
  (g t.v1.1 t.v1.2).val + (g t.v2.1 t.v2.2).val + (g t.v3.1 t.v3.2).val = 15

/-- The main theorem: there exists a valid grid where all triangles sum to 15 -/
theorem nine_circles_problem :
  ∃ (g : Grid), is_valid_grid g ∧ ∀ t ∈ triangles, triangle_sum_is_15 g t :=
sorry

end NUMINAMATH_CALUDE_nine_circles_problem_l368_36879


namespace NUMINAMATH_CALUDE_call_charge_for_550_seconds_l368_36891

-- Define the local call charge rule
def local_call_charge (duration : ℕ) : ℚ :=
  let base_charge : ℚ := 22/100  -- 0.22 yuan for first 3 minutes
  let per_minute_charge : ℚ := 11/100  -- 0.11 yuan per minute after
  let full_minutes : ℕ := (duration + 59) / 60  -- Round up to nearest minute
  if full_minutes ≤ 3 then
    base_charge
  else
    base_charge + per_minute_charge * (full_minutes - 3 : ℚ)

-- Theorem statement
theorem call_charge_for_550_seconds :
  local_call_charge 550 = 99/100 := by
  sorry

end NUMINAMATH_CALUDE_call_charge_for_550_seconds_l368_36891


namespace NUMINAMATH_CALUDE_sugar_spilled_correct_l368_36828

/-- The amount of sugar Pamela spilled on the floor -/
def sugar_spilled (original : ℝ) (left : ℝ) : ℝ := original - left

/-- Theorem stating that the amount of sugar spilled is correct -/
theorem sugar_spilled_correct (original left : ℝ) 
  (h1 : original = 9.8)
  (h2 : left = 4.6) : 
  sugar_spilled original left = 5.2 := by
  sorry

end NUMINAMATH_CALUDE_sugar_spilled_correct_l368_36828


namespace NUMINAMATH_CALUDE_village_population_l368_36877

theorem village_population (partial_population : ℕ) (percentage : ℚ) (total_population : ℕ) :
  percentage = 90 / 100 →
  partial_population = 45000 →
  (percentage : ℚ) * total_population = partial_population →
  total_population = 50000 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l368_36877


namespace NUMINAMATH_CALUDE_jeffs_score_l368_36800

theorem jeffs_score (jeff tim : ℕ) (h1 : jeff = tim + 60) (h2 : (jeff + tim) / 2 = 112) : jeff = 142 := by
  sorry

end NUMINAMATH_CALUDE_jeffs_score_l368_36800


namespace NUMINAMATH_CALUDE_village_population_percentage_l368_36896

theorem village_population_percentage :
  let total_population : ℕ := 24000
  let part_population : ℕ := 23040
  let percentage : ℚ := (part_population : ℚ) / total_population * 100
  percentage = 96 := by
  sorry

end NUMINAMATH_CALUDE_village_population_percentage_l368_36896


namespace NUMINAMATH_CALUDE_marbles_cost_marbles_cost_value_l368_36815

def total_spent : ℚ := 20.52
def football_cost : ℚ := 4.95
def baseball_cost : ℚ := 6.52

theorem marbles_cost : ℚ :=
  total_spent - (football_cost + baseball_cost)

#check marbles_cost

theorem marbles_cost_value : marbles_cost = 9.05 := by sorry

end NUMINAMATH_CALUDE_marbles_cost_marbles_cost_value_l368_36815


namespace NUMINAMATH_CALUDE_min_value_of_P_l368_36897

/-- The polynomial function P(x,y) -/
def P (x y : ℝ) : ℝ := 4 + x^2 * y^4 + x^4 * y^2 - 3 * x^2 * y^2

/-- Theorem stating that the minimal value of P(x,y) is 3 -/
theorem min_value_of_P :
  (∀ x y : ℝ, P x y ≥ 3) ∧ (∃ x y : ℝ, P x y = 3) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_P_l368_36897


namespace NUMINAMATH_CALUDE_vector_perpendicular_l368_36860

def a : Fin 2 → ℝ := ![2, 1]
def b : Fin 2 → ℝ := ![0, -2]

def perpendicular (v w : Fin 2 → ℝ) : Prop :=
  (v 0) * (w 0) + (v 1) * (w 1) = 0

theorem vector_perpendicular :
  perpendicular (λ i => a i + 2 * b i) ![3, 2] := by
  sorry

end NUMINAMATH_CALUDE_vector_perpendicular_l368_36860


namespace NUMINAMATH_CALUDE_symmetric_circle_l368_36801

/-- Given a circle and a line of symmetry, find the equation of the symmetric circle -/
theorem symmetric_circle (x y : ℝ) : 
  -- Original circle
  ((x - 2)^2 + (y - 3)^2 = 1) →
  -- Line of symmetry
  (x + y - 1 = 0) →
  -- Symmetric circle
  ((x + 2)^2 + (y + 1)^2 = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_symmetric_circle_l368_36801


namespace NUMINAMATH_CALUDE_block_depth_l368_36826

theorem block_depth (cube_volume : ℕ) (length width : ℕ) (fewer_cubes : ℕ) (d : ℕ) : 
  cube_volume = 5 →
  length = 7 →
  width = 7 →
  fewer_cubes = 194 →
  length * width * d * cube_volume - fewer_cubes * cube_volume = length * width * (d - 1) * cube_volume →
  d = 5 :=
by sorry

end NUMINAMATH_CALUDE_block_depth_l368_36826


namespace NUMINAMATH_CALUDE_equation_solutions_l368_36878

theorem equation_solutions : 
  let f : ℝ → ℝ := λ x => (x - 3)^2 + 2*x*(x - 3)
  ∀ x : ℝ, f x = 0 ↔ x = 3 ∨ x = 1 := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l368_36878


namespace NUMINAMATH_CALUDE_decreased_equilateral_angle_l368_36824

/-- The measure of an angle in an equilateral triangle -/
def equilateral_angle : ℝ := 60

/-- The amount by which angle E is decreased -/
def angle_decrease : ℝ := 15

/-- Theorem: In an equilateral triangle where one angle is decreased by 15 degrees, 
    the measure of the decreased angle is 45 degrees -/
theorem decreased_equilateral_angle :
  equilateral_angle - angle_decrease = 45 := by sorry

end NUMINAMATH_CALUDE_decreased_equilateral_angle_l368_36824


namespace NUMINAMATH_CALUDE_notebook_cost_is_one_thirty_l368_36806

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The number of nickels used to buy the notebook -/
def nickels_used : ℕ := 26

/-- The cost of the notebook in cents -/
def notebook_cost_cents : ℕ := nickel_value * nickels_used

/-- The cost of the notebook in dollars -/
def notebook_cost_dollars : ℚ := notebook_cost_cents / 100

/-- Theorem: The cost of a notebook bought with 26 nickels is $1.30 -/
theorem notebook_cost_is_one_thirty : notebook_cost_dollars = 13/10 := by
  sorry

end NUMINAMATH_CALUDE_notebook_cost_is_one_thirty_l368_36806


namespace NUMINAMATH_CALUDE_perpendicular_sufficient_not_necessary_l368_36820

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism and perpendicularity relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_to_plane : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_sufficient_not_necessary
  (l m : Line) (α : Plane)
  (h_different : l ≠ m)
  (h_parallel : parallel m α) :
  (∀ l m α, perpendicular_to_plane l α → perpendicular l m) ∧
  (∃ l m α, perpendicular l m ∧ ¬ perpendicular_to_plane l α) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_sufficient_not_necessary_l368_36820


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l368_36808

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 5) → x ≥ 5 := by
sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l368_36808


namespace NUMINAMATH_CALUDE_max_beads_in_pile_l368_36867

/-- Represents a pile of beads -/
structure BeadPile :=
  (size : ℕ)
  (has_lighter_bead : Bool)

/-- Represents a balance scale measurement -/
inductive Measurement
  | Balanced
  | Unbalanced

/-- A function that performs a measurement on a subset of beads -/
def perform_measurement (subset_size : ℕ) : Measurement :=
  sorry

/-- A function that represents the algorithm to find the lighter bead -/
def find_lighter_bead (pile : BeadPile) (max_measurements : ℕ) : Bool :=
  sorry

/-- Theorem stating the maximum number of beads in the pile -/
theorem max_beads_in_pile :
  ∀ (pile : BeadPile),
    pile.has_lighter_bead →
    (∃ (algorithm : BeadPile → ℕ → Bool),
      (∀ p, algorithm p 2 = find_lighter_bead p 2) →
      algorithm pile 2 = true) →
    pile.size ≤ 7 :=
sorry

end NUMINAMATH_CALUDE_max_beads_in_pile_l368_36867


namespace NUMINAMATH_CALUDE_square_perimeters_sum_l368_36839

theorem square_perimeters_sum (x : ℝ) : 
  let area1 := x^2 + 8*x + 16
  let area2 := 4*x^2 - 12*x + 9
  let area3 := 9*x^2 - 6*x + 1
  let perimeter1 := 4 * Real.sqrt area1
  let perimeter2 := 4 * Real.sqrt area2
  let perimeter3 := 4 * Real.sqrt area3
  perimeter1 + perimeter2 + perimeter3 = 48 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeters_sum_l368_36839


namespace NUMINAMATH_CALUDE_max_salad_servings_is_56_l368_36894

/-- Represents the ingredients required for one serving of salad -/
structure SaladServing where
  cucumbers : ℕ
  tomatoes : ℕ
  brynza : ℕ  -- in grams
  peppers : ℕ

/-- Represents the available ingredients in the restaurant's warehouse -/
structure WarehouseStock where
  cucumbers : ℕ
  tomatoes : ℕ
  brynza : ℕ  -- in grams
  peppers : ℕ

/-- Calculates the maximum number of salad servings that can be made -/
def maxSaladServings (serving : SaladServing) (stock : WarehouseStock) : ℕ :=
  min
    (stock.cucumbers / serving.cucumbers)
    (min
      (stock.tomatoes / serving.tomatoes)
      (min
        (stock.brynza / serving.brynza)
        (stock.peppers / serving.peppers)))

/-- Theorem stating that the maximum number of salad servings is 56 -/
theorem max_salad_servings_is_56 :
  let serving := SaladServing.mk 2 2 75 1
  let stock := WarehouseStock.mk 117 116 4200 60
  maxSaladServings serving stock = 56 := by
  sorry

#eval maxSaladServings (SaladServing.mk 2 2 75 1) (WarehouseStock.mk 117 116 4200 60)

end NUMINAMATH_CALUDE_max_salad_servings_is_56_l368_36894


namespace NUMINAMATH_CALUDE_total_trees_formula_l368_36846

/-- The total number of trees planted by three teams under specific conditions -/
def total_trees (a : ℕ) : ℕ :=
  let team1 := a
  let team2 := 2 * a + 8
  let team3 := (team2 / 2) - 6
  team1 + team2 + team3

/-- Theorem stating the total number of trees planted by the three teams -/
theorem total_trees_formula (a : ℕ) : total_trees a = 4 * a + 6 := by
  sorry

#eval total_trees 100  -- Should output 406

end NUMINAMATH_CALUDE_total_trees_formula_l368_36846


namespace NUMINAMATH_CALUDE_opposite_roots_quadratic_l368_36805

theorem opposite_roots_quadratic (k : ℝ) : 
  (∃ x y : ℝ, x^2 + (k^2 - 1)*x + k + 1 = 0 ∧ 
               y^2 + (k^2 - 1)*y + k + 1 = 0 ∧ 
               x = -y ∧ x ≠ y) → 
  k = -1 := by
sorry

end NUMINAMATH_CALUDE_opposite_roots_quadratic_l368_36805


namespace NUMINAMATH_CALUDE_bella_items_after_purchase_l368_36825

def total_items (marbles frisbees deck_cards action_figures : ℕ) : ℕ :=
  marbles + frisbees + deck_cards + action_figures

theorem bella_items_after_purchase : 
  ∀ (marbles frisbees deck_cards action_figures : ℕ),
    marbles = 60 →
    marbles = 2 * frisbees →
    frisbees = deck_cards + 20 →
    marbles = 5 * action_figures →
    total_items (marbles + (2 * marbles) / 5)
                (frisbees + (2 * frisbees) / 5)
                (deck_cards + (2 * deck_cards) / 5)
                (action_figures + action_figures / 3) = 156 := by
  sorry

#check bella_items_after_purchase

end NUMINAMATH_CALUDE_bella_items_after_purchase_l368_36825


namespace NUMINAMATH_CALUDE_sector_to_inscribed_circle_area_ratio_l368_36807

/-- Given a sector with a central angle of 120° and its inscribed circle,
    the ratio of the area of the sector to the area of the inscribed circle
    is (7 + 4√3) / 9. -/
theorem sector_to_inscribed_circle_area_ratio :
  ∀ (R r : ℝ), R > 0 → r > 0 →
  (2 * π / 3 : ℝ) = 2 * Real.arcsin (r / R) →
  (π * R^2 * (2 * π / 3) / (2 * π)) / (π * r^2) = (7 + 4 * Real.sqrt 3) / 9 := by
  sorry

end NUMINAMATH_CALUDE_sector_to_inscribed_circle_area_ratio_l368_36807


namespace NUMINAMATH_CALUDE_hot_dogs_remainder_l368_36893

theorem hot_dogs_remainder (total : Nat) (package_size : Nat) : 
  total = 25197624 → package_size = 4 → total % package_size = 0 := by
  sorry

end NUMINAMATH_CALUDE_hot_dogs_remainder_l368_36893


namespace NUMINAMATH_CALUDE_johns_weekly_sleep_l368_36895

/-- Calculates the total sleep John got in a week given the specified conditions --/
def johnsTotalSleep (daysInWeek : ℕ) (shortSleepDays : ℕ) (shortSleepHours : ℝ) 
  (recommendedSleep : ℝ) (percentOfRecommended : ℝ) : ℝ :=
  let normalSleepDays := daysInWeek - shortSleepDays
  let normalSleepHours := recommendedSleep * percentOfRecommended
  shortSleepDays * shortSleepHours + normalSleepDays * normalSleepHours

/-- Theorem stating that John's total sleep for the week equals 30 hours --/
theorem johns_weekly_sleep :
  johnsTotalSleep 7 2 3 8 0.6 = 30 := by
  sorry

#eval johnsTotalSleep 7 2 3 8 0.6

end NUMINAMATH_CALUDE_johns_weekly_sleep_l368_36895


namespace NUMINAMATH_CALUDE_max_triangle_area_l368_36876

theorem max_triangle_area (a b : ℝ) (ha : a = 1984) (hb : b = 2016) :
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ π → (1/2) * a * b * Real.sin θ ≤ 1998912) ∧
  (∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ π ∧ (1/2) * a * b * Real.sin θ = 1998912) := by
  sorry

end NUMINAMATH_CALUDE_max_triangle_area_l368_36876


namespace NUMINAMATH_CALUDE_f_inequality_iff_a_range_l368_36874

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - (a + 1) * x + (1/2) * x^2

theorem f_inequality_iff_a_range (a : ℝ) :
  (a > 0 ∧ ∀ x > 1, f a x ≥ x^a - Real.exp x + (1/2) * x^2 - a * x) ↔ 0 < a ∧ a ≤ Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_f_inequality_iff_a_range_l368_36874


namespace NUMINAMATH_CALUDE_prob_vertical_side_from_start_l368_36875

/-- Represents a point on the grid -/
structure Point where
  x : ℕ
  y : ℕ

/-- The probability of jumping in each direction -/
def jump_prob : Fin 4 → ℝ
| 0 => 0.3  -- up
| 1 => 0.3  -- down
| 2 => 0.2  -- left
| 3 => 0.2  -- right

/-- The dimensions of the grid -/
def grid_size : ℕ := 6

/-- The starting point of the frog -/
def start : Point := ⟨2, 3⟩

/-- Predicate to check if a point is on the vertical side of the grid -/
def on_vertical_side (p : Point) : Prop :=
  p.x = 0 ∨ p.x = grid_size

/-- The probability of reaching a vertical side first from a given point -/
noncomputable def prob_vertical_side (p : Point) : ℝ := sorry

/-- The main theorem: probability of reaching a vertical side first from the starting point -/
theorem prob_vertical_side_from_start :
  prob_vertical_side start = 5/8 := by sorry

end NUMINAMATH_CALUDE_prob_vertical_side_from_start_l368_36875


namespace NUMINAMATH_CALUDE_quadratic_solution_l368_36816

theorem quadratic_solution (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0)
  (h1 : (2 * c)^2 + c * (2 * c) + d = 0)
  (h2 : (-3 * d)^2 + c * (-3 * d) + d = 0) :
  c = -1/6 ∧ d = -1/6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_l368_36816


namespace NUMINAMATH_CALUDE_fraction_value_l368_36848

theorem fraction_value : (0.5 ^ 4) / (0.05 ^ 2.5) = 559.06 := by sorry

end NUMINAMATH_CALUDE_fraction_value_l368_36848


namespace NUMINAMATH_CALUDE_no_consistent_solution_l368_36847

theorem no_consistent_solution :
  ¬ ∃ (x y : ℕ+) (z : ℤ),
    (∃ (q : ℕ), x = 11 * y + 4 ∧ x = 11 * q + 4) ∧
    (∃ (q : ℕ), 2 * x = 8 * (3 * y) + 3 ∧ 2 * x = 8 * q + 3) ∧
    (∃ (q : ℕ), x + z = 17 * (2 * y) + 5 ∧ x + z = 17 * q + 5) :=
by sorry

end NUMINAMATH_CALUDE_no_consistent_solution_l368_36847


namespace NUMINAMATH_CALUDE_max_a_equals_min_f_l368_36822

theorem max_a_equals_min_f : 
  let f (x : ℝ) := x^2 + 2*x - 6
  (∃ (a_max : ℝ), (∀ (a : ℝ), (∀ (x : ℝ), f x ≥ a) → a ≤ a_max) ∧ 
    (∀ (x : ℝ), f x ≥ a_max)) ∧ 
  (∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x) →
  ∃ (a_max x_min : ℝ), (∀ (a : ℝ), (∀ (x : ℝ), f x ≥ a) → a ≤ a_max) ∧ 
    (∀ (x : ℝ), f x ≥ a_max) ∧ 
    (∀ (x : ℝ), f x_min ≤ f x) ∧ 
    a_max = f x_min :=
by sorry

end NUMINAMATH_CALUDE_max_a_equals_min_f_l368_36822


namespace NUMINAMATH_CALUDE_average_weight_group_B_proof_l368_36880

/-- The average weight of additional friends in Group B -/
def average_weight_group_B : ℝ := 141

theorem average_weight_group_B_proof
  (initial_group : ℕ) (additional_group : ℕ) (group_A : ℕ) (group_B : ℕ)
  (avg_weight_increase : ℝ) (avg_weight_gain_A : ℝ) (final_avg_weight : ℝ)
  (h1 : initial_group = 50)
  (h2 : additional_group = 40)
  (h3 : group_A = 20)
  (h4 : group_B = 20)
  (h5 : avg_weight_increase = 12)
  (h6 : avg_weight_gain_A = 15)
  (h7 : final_avg_weight = 46)
  (h8 : additional_group = group_A + group_B) :
  average_weight_group_B = 141 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_group_B_proof_l368_36880


namespace NUMINAMATH_CALUDE_conditional_probability_good_air_quality_l368_36833

-- Define the probability of good air quality on any given day
def p_good_day : ℝ := 0.75

-- Define the probability of good air quality for two consecutive days
def p_two_good_days : ℝ := 0.6

-- State the theorem
theorem conditional_probability_good_air_quality :
  (p_two_good_days / p_good_day : ℝ) = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_conditional_probability_good_air_quality_l368_36833


namespace NUMINAMATH_CALUDE_car_cost_difference_l368_36887

/-- Represents the cost and characteristics of a car --/
structure Car where
  initialCost : ℕ
  fuelConsumption : ℕ
  annualInsurance : ℕ
  annualMaintenance : ℕ
  resaleValue : ℕ

/-- Calculates the total cost of owning a car for 5 years --/
def totalCost (c : Car) (annualDistance : ℕ) (fuelCost : ℕ) (years : ℕ) : ℕ :=
  c.initialCost +
  (annualDistance * c.fuelConsumption * fuelCost * years) / 10000 +
  c.annualInsurance * years +
  c.annualMaintenance * years -
  c.resaleValue

/-- The statement to be proved --/
theorem car_cost_difference :
  let carA : Car := {
    initialCost := 900000,
    fuelConsumption := 9,
    annualInsurance := 35000,
    annualMaintenance := 25000,
    resaleValue := 500000
  }
  let carB : Car := {
    initialCost := 600000,
    fuelConsumption := 10,
    annualInsurance := 32000,
    annualMaintenance := 20000,
    resaleValue := 350000
  }
  let annualDistance := 15000
  let fuelCost := 40
  let years := 5
  totalCost carA annualDistance fuelCost years - totalCost carB annualDistance fuelCost years = 160000 := by
  sorry

end NUMINAMATH_CALUDE_car_cost_difference_l368_36887


namespace NUMINAMATH_CALUDE_relationship_holds_l368_36844

/-- The function describing the relationship between x and y -/
def f (x : ℕ) : ℕ := x^2 - 3*x + 2

/-- The set of x values given in the table -/
def X : Set ℕ := {2, 3, 4, 5, 6}

/-- The proposition that the function f correctly describes the relationship for all x in X -/
theorem relationship_holds (x : ℕ) (h : x ∈ X) : 
  (x = 2 → f x = 0) ∧ 
  (x = 3 → f x = 2) ∧ 
  (x = 4 → f x = 6) ∧ 
  (x = 5 → f x = 12) ∧ 
  (x = 6 → f x = 20) :=
by sorry

end NUMINAMATH_CALUDE_relationship_holds_l368_36844


namespace NUMINAMATH_CALUDE_apple_box_weight_l368_36861

theorem apple_box_weight : 
  ∀ (x : ℝ), 
  (x > 0) →  -- Ensure positive weight
  (3 * x - 3 * 4 = x) → 
  x = 6 := by
  sorry

end NUMINAMATH_CALUDE_apple_box_weight_l368_36861


namespace NUMINAMATH_CALUDE_bat_survey_result_l368_36868

theorem bat_survey_result :
  ∀ (total : ℕ) 
    (blind_believers : ℕ) 
    (ebola_believers : ℕ),
  (blind_believers : ℚ) = 0.750 * total →
  (ebola_believers : ℚ) = 0.523 * blind_believers →
  ebola_believers = 49 →
  total = 125 :=
by
  sorry

end NUMINAMATH_CALUDE_bat_survey_result_l368_36868


namespace NUMINAMATH_CALUDE_total_cost_theorem_l368_36883

def original_price : ℝ := 10
def child_discount : ℝ := 0.3
def senior_discount : ℝ := 0.1
def handling_fee : ℝ := 5
def num_child_tickets : ℕ := 2
def num_senior_tickets : ℕ := 2

def senior_ticket_price : ℝ := 14

theorem total_cost_theorem :
  let child_ticket_price := (1 - child_discount) * original_price + handling_fee
  let total_cost := num_child_tickets * child_ticket_price + num_senior_tickets * senior_ticket_price
  total_cost = 52 := by sorry

end NUMINAMATH_CALUDE_total_cost_theorem_l368_36883


namespace NUMINAMATH_CALUDE_penny_species_count_l368_36872

/-- The number of shark species Penny identified -/
def shark_species : ℕ := 35

/-- The number of eel species Penny identified -/
def eel_species : ℕ := 15

/-- The number of whale species Penny identified -/
def whale_species : ℕ := 5

/-- The total number of species Penny identified -/
def total_species : ℕ := shark_species + eel_species + whale_species

/-- Theorem stating that the total number of species Penny identified is 55 -/
theorem penny_species_count : total_species = 55 := by
  sorry

end NUMINAMATH_CALUDE_penny_species_count_l368_36872


namespace NUMINAMATH_CALUDE_maximal_colored_squares_correct_l368_36821

/-- Given positive integers n and k where n > k^2 > 4, maximal_colored_squares
    returns the maximal number of unit squares that can be colored in an n × n grid,
    such that in any k-group there are two squares with the same color and
    two squares with different colors. -/
def maximal_colored_squares (n k : ℕ+) (h1 : n > k^2) (h2 : k^2 > 4) : ℕ :=
  n * (k - 1)^2

/-- Theorem stating that maximal_colored_squares gives the correct result -/
theorem maximal_colored_squares_correct (n k : ℕ+) (h1 : n > k^2) (h2 : k^2 > 4) :
  maximal_colored_squares n k h1 h2 = n * (k - 1)^2 := by
  sorry

#check maximal_colored_squares
#check maximal_colored_squares_correct

end NUMINAMATH_CALUDE_maximal_colored_squares_correct_l368_36821


namespace NUMINAMATH_CALUDE_house_wall_planks_l368_36869

/-- The number of large planks needed for the house wall. -/
def large_planks : ℕ := 12

/-- The number of small planks needed for the house wall. -/
def small_planks : ℕ := 17

/-- The total number of planks needed for the house wall. -/
def total_planks : ℕ := large_planks + small_planks

theorem house_wall_planks : total_planks = 29 := by
  sorry

end NUMINAMATH_CALUDE_house_wall_planks_l368_36869


namespace NUMINAMATH_CALUDE_cookies_per_pack_l368_36845

theorem cookies_per_pack (trays : ℕ) (cookies_per_tray : ℕ) (packs : ℕ) 
  (h1 : trays = 8) 
  (h2 : cookies_per_tray = 36) 
  (h3 : packs = 12) :
  (trays * cookies_per_tray) / packs = 24 := by
  sorry

#check cookies_per_pack

end NUMINAMATH_CALUDE_cookies_per_pack_l368_36845


namespace NUMINAMATH_CALUDE_height_on_hypotenuse_of_right_triangle_l368_36837

theorem height_on_hypotenuse_of_right_triangle (a b h c : ℝ) : 
  a = 6 → b = 8 → c^2 = a^2 + b^2 → (1/2) * a * b = (1/2) * c * h → h = 4.8 := by
  sorry

end NUMINAMATH_CALUDE_height_on_hypotenuse_of_right_triangle_l368_36837


namespace NUMINAMATH_CALUDE_line_AB_slope_and_equation_l368_36863

/-- Given points A(0,-2) and B(√3,1), prove the slope of line AB is √3 and its equation is y = √3x - 2 -/
theorem line_AB_slope_and_equation :
  let A : ℝ × ℝ := (0, -2)
  let B : ℝ × ℝ := (Real.sqrt 3, 1)
  let slope : ℝ := (B.2 - A.2) / (B.1 - A.1)
  let equation (x : ℝ) : ℝ := slope * x + (A.2 - slope * A.1)
  slope = Real.sqrt 3 ∧ ∀ x, equation x = Real.sqrt 3 * x - 2 := by
  sorry

end NUMINAMATH_CALUDE_line_AB_slope_and_equation_l368_36863


namespace NUMINAMATH_CALUDE_inequality_proof_l368_36841

theorem inequality_proof (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  (a + b + c) / 3 - (a * b * c) ^ (1/3) ≤ max ((a^(1/2) - b^(1/2))^2) (max ((b^(1/2) - c^(1/2))^2) ((c^(1/2) - a^(1/2))^2)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l368_36841


namespace NUMINAMATH_CALUDE_max_value_2x_minus_y_l368_36835

theorem max_value_2x_minus_y (x y : ℝ) 
  (h1 : x - y + 1 ≥ 0) 
  (h2 : y + 1 ≥ 0) 
  (h3 : x + y + 1 ≤ 0) : 
  ∃ (m : ℝ), m = 1 ∧ ∀ (x' y' : ℝ), 
    x' - y' + 1 ≥ 0 → y' + 1 ≥ 0 → x' + y' + 1 ≤ 0 → 2*x' - y' ≤ m :=
by sorry

end NUMINAMATH_CALUDE_max_value_2x_minus_y_l368_36835


namespace NUMINAMATH_CALUDE_one_face_colored_count_l368_36849

/-- Represents a cube that has been painted and cut into smaller cubes -/
structure PaintedCube where
  edge_count : Nat
  is_painted : Bool

/-- Counts the number of small cubes with exactly one face colored -/
def count_one_face_colored (cube : PaintedCube) : Nat :=
  sorry

/-- Theorem: A cube painted on all faces and cut into 5x5x5 smaller cubes
    will have 54 small cubes with exactly one face colored -/
theorem one_face_colored_count (cube : PaintedCube) :
  cube.edge_count = 5 → cube.is_painted → count_one_face_colored cube = 54 := by
  sorry

end NUMINAMATH_CALUDE_one_face_colored_count_l368_36849


namespace NUMINAMATH_CALUDE_common_terms_k_polygonal_fermat_l368_36852

/-- k-polygonal number sequence -/
def kPolygonalSeq (k : ℕ) (n : ℕ) : ℕ :=
  (k - 2) * n * (n - 1) / 2 + n

/-- Fermat number sequence -/
def fermatSeq (m : ℕ) : ℕ :=
  2^(2^m) + 1

/-- Proposition: The only positive integers k > 2 for which there exist common terms
    between the k-polygonal numbers sequence and the Fermat numbers sequence are 3 and 5 -/
theorem common_terms_k_polygonal_fermat :
  {k : ℕ | k > 2 ∧ ∃ (n m : ℕ), kPolygonalSeq k n = fermatSeq m} = {3, 5} := by
  sorry

end NUMINAMATH_CALUDE_common_terms_k_polygonal_fermat_l368_36852


namespace NUMINAMATH_CALUDE_equality_of_sides_from_equal_angles_l368_36881

-- Define a structure for 3D points
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a function to calculate the angle between three points
def angle (p1 p2 p3 : Point3D) : ℝ := sorry

-- Define a function to calculate the distance between two points
def distance (p1 p2 : Point3D) : ℝ := sorry

-- Define a predicate to check if four points are non-coplanar
def nonCoplanar (p1 p2 p3 p4 : Point3D) : Prop := sorry

theorem equality_of_sides_from_equal_angles 
  (A B C D : Point3D) 
  (h1 : nonCoplanar A B C D)
  (h2 : angle A B C = angle A D C)
  (h3 : angle B A D = angle B C D) :
  distance A B = distance C D ∧ distance B C = distance A D := by
  sorry

end NUMINAMATH_CALUDE_equality_of_sides_from_equal_angles_l368_36881


namespace NUMINAMATH_CALUDE_repeating_decimal_37_l368_36811

/-- The repeating decimal 0.373737... expressed as a rational number -/
theorem repeating_decimal_37 : ∃ (x : ℚ), x = 37 / 99 ∧ 
  ∀ (n : ℕ), (100 * x - ⌊100 * x⌋ : ℚ) * 10^n = (37 * 10^n : ℚ) % 100 / 100 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_37_l368_36811


namespace NUMINAMATH_CALUDE_vaccine_comparison_l368_36812

/-- Represents a vaccine trial result -/
structure VaccineTrial where
  vaccinated : Nat
  infected : Nat

/-- Determines if a vaccine is considered effective based on trial results and population infection rate -/
def is_effective (trial : VaccineTrial) (population_rate : Real) : Prop :=
  (trial.infected : Real) / trial.vaccinated < population_rate

/-- Compares the effectiveness of two vaccines -/
def more_effective (trial1 trial2 : VaccineTrial) (population_rate : Real) : Prop :=
  is_effective trial1 population_rate ∧ is_effective trial2 population_rate ∧
  (trial1.infected : Real) / trial1.vaccinated < (trial2.infected : Real) / trial2.vaccinated

theorem vaccine_comparison :
  let population_rate : Real := 0.2
  let vaccine_I : VaccineTrial := ⟨8, 0⟩
  let vaccine_II : VaccineTrial := ⟨25, 1⟩
  more_effective vaccine_II vaccine_I population_rate :=
by
  sorry

end NUMINAMATH_CALUDE_vaccine_comparison_l368_36812


namespace NUMINAMATH_CALUDE_sum_of_quadratic_solutions_l368_36803

theorem sum_of_quadratic_solutions (x : ℝ) : 
  (x^2 + 6*x - 22 = 4*x - 18) → 
  (∃ a b : ℝ, (a + b = -2) ∧ (x = a ∨ x = b)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_quadratic_solutions_l368_36803


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_over_one_plus_i_l368_36855

theorem imaginary_part_of_i_over_one_plus_i (i : ℂ) :
  i * i = -1 →
  Complex.im (i / (1 + i)) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_over_one_plus_i_l368_36855


namespace NUMINAMATH_CALUDE_set_intersection_problem_l368_36809

def A : Set ℕ := {1, 6, 8, 10}
def B : Set ℕ := {2, 4, 8, 10}

theorem set_intersection_problem : A ∩ B = {8, 10} := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_problem_l368_36809


namespace NUMINAMATH_CALUDE_g_increasing_range_l368_36843

/-- A piecewise function g(x) defined on [0, +∞) -/
noncomputable def g (m : ℝ) (x : ℝ) : ℝ :=
  if x ≥ m then (1/4) * x^2 else x

/-- The theorem stating the range of m for which g is increasing on [0, +∞) -/
theorem g_increasing_range (m : ℝ) :
  (m > 0) →
  (∀ x y, 0 ≤ x ∧ x < y → g m x ≤ g m y) →
  m ∈ Set.Ici 4 :=
sorry

end NUMINAMATH_CALUDE_g_increasing_range_l368_36843


namespace NUMINAMATH_CALUDE_right_triangle_is_stable_l368_36842

-- Define the concept of a shape
structure Shape :=
  (name : String)

-- Define the property of stability
def is_stable (s : Shape) : Prop := sorry

-- Define a right triangle
def right_triangle : Shape :=
  { name := "Right Triangle" }

-- Define structural rigidity
def has_structural_rigidity (s : Shape) : Prop := sorry

-- Define resistance to deformation
def resists_deformation (s : Shape) : Prop := sorry

-- Theorem: A right triangle is stable
theorem right_triangle_is_stable :
  has_structural_rigidity right_triangle →
  resists_deformation right_triangle →
  is_stable right_triangle :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_is_stable_l368_36842


namespace NUMINAMATH_CALUDE_ashley_wedding_champagne_servings_l368_36836

/-- The number of servings in one bottle of champagne for Ashley's wedding toast. -/
def servings_per_bottle (guests : ℕ) (glasses_per_guest : ℕ) (total_bottles : ℕ) : ℕ :=
  (guests * glasses_per_guest) / total_bottles

/-- Theorem stating that there are 6 servings in one bottle of champagne for Ashley's wedding toast. -/
theorem ashley_wedding_champagne_servings :
  servings_per_bottle 120 2 40 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ashley_wedding_champagne_servings_l368_36836


namespace NUMINAMATH_CALUDE_equal_abc_l368_36859

theorem equal_abc (a b c x : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : (x * b + (1 - x) * c) / a = (x * c + (1 - x) * a) / b)
  (h5 : (x * b + (1 - x) * c) / a = (x * a + (1 - x) * b) / c) :
  a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_equal_abc_l368_36859


namespace NUMINAMATH_CALUDE_original_index_is_12_l368_36832

/-- Given an original sequence and a new sequence formed by inserting 3 numbers
    between every two adjacent terms of the original sequence, 
    this function returns the index in the original sequence that corresponds
    to the 49th term in the new sequence. -/
def original_index_of_49th_new_term : ℕ :=
  let x := (49 - 1) / 4
  x + 1

theorem original_index_is_12 : original_index_of_49th_new_term = 12 := by
  sorry

end NUMINAMATH_CALUDE_original_index_is_12_l368_36832


namespace NUMINAMATH_CALUDE_value_standard_deviations_below_mean_l368_36810

/-- For a normal distribution with mean 14.5 and standard deviation 1.5,
    the value 11.5 is 2 standard deviations less than the mean. -/
theorem value_standard_deviations_below_mean
  (μ : ℝ) (σ : ℝ) (x : ℝ)
  (h_mean : μ = 14.5)
  (h_std_dev : σ = 1.5)
  (h_value : x = 11.5) :
  (μ - x) / σ = 2 := by
sorry

end NUMINAMATH_CALUDE_value_standard_deviations_below_mean_l368_36810


namespace NUMINAMATH_CALUDE_special_square_difference_l368_36899

theorem special_square_difference : 123456789^2 - 123456788 * 123456790 = 1 := by
  sorry

end NUMINAMATH_CALUDE_special_square_difference_l368_36899


namespace NUMINAMATH_CALUDE_f_odd_f_inequality_iff_a_range_l368_36888

noncomputable section

def f (x : ℝ) : ℝ := (Real.exp x - 1) / (Real.exp x + 1)

theorem f_odd : ∀ x : ℝ, f (-x) = -f x := by sorry

theorem f_inequality_iff_a_range :
  ∀ a : ℝ, (∀ x : ℝ, x > 1 ∧ x < 2 → f (a * x^2 + 2) + f (2 * x - 1) > 0) ↔ a > -5/4 := by sorry

end NUMINAMATH_CALUDE_f_odd_f_inequality_iff_a_range_l368_36888


namespace NUMINAMATH_CALUDE_snackles_leftover_candies_l368_36870

theorem snackles_leftover_candies (m : ℕ) (h : m % 9 = 8) : (2 * m) % 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_snackles_leftover_candies_l368_36870


namespace NUMINAMATH_CALUDE_curve_symmetric_about_origin_l368_36851

/-- A curve defined by the equation xy - x^2 = 1 -/
def curve (x y : ℝ) : Prop := x * y - x^2 = 1

/-- Symmetry about the origin for the curve -/
theorem curve_symmetric_about_origin :
  ∀ x y : ℝ, curve x y ↔ curve (-x) (-y) :=
sorry

end NUMINAMATH_CALUDE_curve_symmetric_about_origin_l368_36851


namespace NUMINAMATH_CALUDE_continued_fraction_value_l368_36823

theorem continued_fraction_value : ∃ x : ℝ, 
  x = 3 + 4 / (1 + 4 / (3 + 4 / ((1/2) + x))) ∧ 
  x = (43 + Real.sqrt 4049) / 22 := by
  sorry

end NUMINAMATH_CALUDE_continued_fraction_value_l368_36823


namespace NUMINAMATH_CALUDE_mixture_ratio_weight_l368_36856

theorem mixture_ratio_weight (total_weight : ℝ) (ratio_a ratio_b : ℕ) (weight_b : ℝ) : 
  total_weight = 58.00000000000001 →
  ratio_a = 9 →
  ratio_b = 11 →
  weight_b = (ratio_b : ℝ) / ((ratio_a : ℝ) + (ratio_b : ℝ)) * total_weight →
  weight_b = 31.900000000000006 := by
sorry

end NUMINAMATH_CALUDE_mixture_ratio_weight_l368_36856


namespace NUMINAMATH_CALUDE_twelfth_term_of_sequence_l368_36871

-- Define an arithmetic sequence
def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  a₁ + (n - 1 : ℚ) * d

-- State the theorem
theorem twelfth_term_of_sequence (a₁ d : ℚ) (h₁ : a₁ = 1/4) :
  arithmetic_sequence a₁ d 12 = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_twelfth_term_of_sequence_l368_36871


namespace NUMINAMATH_CALUDE_solve_for_k_l368_36830

theorem solve_for_k (k : ℝ) (h1 : k ≠ 0) :
  (∀ x : ℝ, (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 6)) → k = 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_k_l368_36830


namespace NUMINAMATH_CALUDE_total_bird_families_l368_36802

/-- The number of bird families that migrated to Africa -/
def africa : ℕ := 42

/-- The number of bird families that migrated to Asia -/
def asia : ℕ := 31

/-- The difference between the number of families that migrated to Africa and Asia -/
def difference : ℕ := 11

/-- Theorem: The total number of bird families before migration is 73 -/
theorem total_bird_families : africa + asia = 73 ∧ africa = asia + difference := by
  sorry

end NUMINAMATH_CALUDE_total_bird_families_l368_36802


namespace NUMINAMATH_CALUDE_max_a_value_l368_36890

-- Define the quadratic polynomial
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

-- State the theorem
theorem max_a_value :
  (∃ (a_max : ℝ), ∀ (a b : ℝ),
    (∀ (x : ℝ), ∃ (y : ℝ), f a b y = f a b x + y) →
    a ≤ a_max ∧
    (∃ (b : ℝ), ∀ (x : ℝ), ∃ (y : ℝ), f a_max b y = f a_max b x + y)) ∧
  (∀ (a_greater : ℝ),
    (∃ (a b : ℝ), a > a_greater ∧
      (∀ (x : ℝ), ∃ (y : ℝ), f a b y = f a b x + y)) →
    a_greater < 1/2) :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l368_36890


namespace NUMINAMATH_CALUDE_parabola_symmetry_l368_36873

-- Define the parabolas
def C₁ (x : ℝ) : ℝ := x^2 - 2*x + 3
def C₂ (x : ℝ) : ℝ := C₁ (x + 1)
def C₃ (x : ℝ) : ℝ := C₂ (-x)

-- State the theorem
theorem parabola_symmetry :
  ∀ x : ℝ, C₃ x = x^2 + 2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_symmetry_l368_36873


namespace NUMINAMATH_CALUDE_rajan_income_l368_36865

/-- Represents the financial situation of two individuals -/
structure FinancialSituation where
  income_ratio : Rat
  expenditure_ratio : Rat
  savings : ℕ

/-- Calculates the income based on the given financial situation -/
def calculate_income (fs : FinancialSituation) : ℕ :=
  sorry

/-- Theorem stating that given the specific financial situation, Rajan's income is 7000 -/
theorem rajan_income (fs : FinancialSituation) 
  (h1 : fs.income_ratio = 7/6)
  (h2 : fs.expenditure_ratio = 6/5)
  (h3 : fs.savings = 1000) :
  calculate_income fs = 7000 := by
  sorry

end NUMINAMATH_CALUDE_rajan_income_l368_36865


namespace NUMINAMATH_CALUDE_range_of_a_l368_36831

-- Define the inequality and its solution set
def inequality (a x : ℝ) : Prop := (a - 1) * x > a - 1
def solution_set (x : ℝ) : Prop := x < 1

-- Define the theorem
theorem range_of_a (a : ℝ) : 
  (∀ x, inequality a x ↔ solution_set x) → a < 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l368_36831


namespace NUMINAMATH_CALUDE_equation_solutions_l368_36829

theorem equation_solutions :
  (∀ x : ℝ, 12 * (x - 1)^2 = 3 ↔ x = 3/2 ∨ x = 1/2) ∧
  (∀ x : ℝ, (x + 1)^3 = 0.125 ↔ x = -0.5) := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l368_36829


namespace NUMINAMATH_CALUDE_divisibility_by_twelve_l368_36866

theorem divisibility_by_twelve (n : ℕ) : 
  (713 * 10 + n ≥ 1000) ∧ 
  (713 * 10 + n < 10000) ∧ 
  (713 * 10 + n) % 12 = 0 ↔ 
  n = 4 := by
sorry

end NUMINAMATH_CALUDE_divisibility_by_twelve_l368_36866


namespace NUMINAMATH_CALUDE_john_hiking_probability_l368_36857

theorem john_hiking_probability (p_rain : ℝ) (p_hike_given_rain : ℝ) (p_hike_given_sunny : ℝ)
  (h_rain : p_rain = 0.3)
  (h_hike_rain : p_hike_given_rain = 0.1)
  (h_hike_sunny : p_hike_given_sunny = 0.9) :
  p_rain * p_hike_given_rain + (1 - p_rain) * p_hike_given_sunny = 0.66 := by
  sorry

end NUMINAMATH_CALUDE_john_hiking_probability_l368_36857


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l368_36889

theorem square_area_from_perimeter (perimeter : ℝ) (area : ℝ) :
  perimeter = 36 →
  area = (perimeter / 4) ^ 2 →
  area = 81 := by
sorry

end NUMINAMATH_CALUDE_square_area_from_perimeter_l368_36889


namespace NUMINAMATH_CALUDE_typing_time_calculation_l368_36885

theorem typing_time_calculation (original_speed : ℕ) (speed_reduction : ℕ) (document_length : ℕ) :
  original_speed = 212 →
  speed_reduction = 40 →
  document_length = 3440 →
  (document_length : ℚ) / ((original_speed - speed_reduction) : ℚ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_typing_time_calculation_l368_36885


namespace NUMINAMATH_CALUDE_bella_stamps_count_l368_36834

/-- Represents the number of stamps of each type Bella bought -/
structure StampCounts where
  snowflake : ℕ
  truck : ℕ
  rose : ℕ
  butterfly : ℕ

/-- Calculates the total number of stamps bought -/
def totalStamps (counts : StampCounts) : ℕ :=
  counts.snowflake + counts.truck + counts.rose + counts.butterfly

/-- Theorem stating the total number of stamps Bella bought -/
theorem bella_stamps_count : ∃ (counts : StampCounts),
  (counts.snowflake : ℚ) * (105 / 100) = 1575 / 100 ∧
  counts.truck = counts.snowflake + 11 ∧
  counts.rose = counts.truck - 17 ∧
  (counts.butterfly : ℚ) = (3 / 2) * counts.rose ∧
  totalStamps counts = 64 := by
  sorry

#check bella_stamps_count

end NUMINAMATH_CALUDE_bella_stamps_count_l368_36834


namespace NUMINAMATH_CALUDE_sum_of_squares_l368_36814

theorem sum_of_squares (x y : ℕ+) 
  (h1 : x * y + x + y = 119)
  (h2 : x^2 * y + x * y^2 = 1680) :
  x^2 + y^2 = 1057 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_l368_36814


namespace NUMINAMATH_CALUDE_trigonometric_identity_l368_36892

theorem trigonometric_identity : 
  Real.sin (-1200 * π / 180) * Real.cos (1290 * π / 180) + 
  Real.cos (-1020 * π / 180) * Real.sin (-1050 * π / 180) + 
  Real.tan (945 * π / 180) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l368_36892


namespace NUMINAMATH_CALUDE_correct_average_l368_36858

theorem correct_average (n : ℕ) (initial_avg : ℚ) (incorrect_num correct_num : ℚ) :
  n = 10 ∧ initial_avg = 19 ∧ incorrect_num = 26 ∧ correct_num = 76 →
  (n : ℚ) * initial_avg - incorrect_num + correct_num = n * 24 :=
by sorry

end NUMINAMATH_CALUDE_correct_average_l368_36858


namespace NUMINAMATH_CALUDE_crayon_selection_count_l368_36804

/-- The number of crayons in the box -/
def total_crayons : ℕ := 15

/-- The number of crayons Karl must select -/
def crayons_to_select : ℕ := 5

/-- The number of non-red crayons to select -/
def non_red_to_select : ℕ := crayons_to_select - 1

/-- The number of non-red crayons available -/
def available_non_red : ℕ := total_crayons - 1

theorem crayon_selection_count :
  (Nat.choose available_non_red non_red_to_select) = 1001 := by
  sorry

end NUMINAMATH_CALUDE_crayon_selection_count_l368_36804


namespace NUMINAMATH_CALUDE_ellipse_with_y_axis_focus_l368_36827

/-- Given that θ is an interior angle of a triangle ABC and sin θ + cos θ = 3/4,
    prove that x^2 * sin θ - y^2 * cos θ = 1 represents an ellipse with focus on the y-axis -/
theorem ellipse_with_y_axis_focus (θ : Real) (x y : Real) 
  (h1 : 0 < θ ∧ θ < π) -- θ is an interior angle of a triangle
  (h2 : Real.sin θ + Real.cos θ = 3/4) -- given condition
  (h3 : x^2 * Real.sin θ - y^2 * Real.cos θ = 1) -- equation of the curve
  : ∃ (a b : Real), 
    0 < b ∧ b < a ∧ 
    (x^2 / a^2) + (y^2 / b^2) = 1 ∧ 
    (a^2 - b^2) / a^2 > 0 :=
sorry

end NUMINAMATH_CALUDE_ellipse_with_y_axis_focus_l368_36827


namespace NUMINAMATH_CALUDE_tan_product_seventh_pi_l368_36819

theorem tan_product_seventh_pi : 
  Real.tan (π / 7) * Real.tan (2 * π / 7) * Real.tan (3 * π / 7) = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_seventh_pi_l368_36819


namespace NUMINAMATH_CALUDE_mathematics_letter_probability_l368_36850

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of distinct letters in 'MATHEMATICS' -/
def distinct_letters : ℕ := 8

/-- The word we're considering -/
def word : String := "MATHEMATICS"

/-- Theorem: The probability of randomly selecting a letter from the alphabet
    that appears in 'MATHEMATICS' is 4/13 -/
theorem mathematics_letter_probability :
  (distinct_letters : ℚ) / (alphabet_size : ℚ) = 4 / 13 := by
  sorry

end NUMINAMATH_CALUDE_mathematics_letter_probability_l368_36850


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l368_36898

theorem imaginary_part_of_complex_fraction (i : ℂ) :
  i * i = -1 →
  Complex.im (10 * i / (1 - 2 * i)) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l368_36898


namespace NUMINAMATH_CALUDE_abc_product_l368_36840

theorem abc_product (a b c : ℝ) 
  (h1 : a - b = 4)
  (h2 : a^2 + b^2 = 18)
  (h3 : a + b + c = 8) :
  a * b * c = 92 - 50 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_l368_36840
