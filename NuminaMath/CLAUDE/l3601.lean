import Mathlib

namespace NUMINAMATH_CALUDE_shaded_area_calculation_l3601_360113

/-- The area of the shaded region in a configuration where a 5x5 square adjoins a 15x15 square,
    with a line drawn from the top left corner of the larger square to the bottom right corner
    of the smaller square, is 175/8 square inches. -/
theorem shaded_area_calculation : 
  let large_square_side : ℝ := 15
  let small_square_side : ℝ := 5
  let total_width : ℝ := large_square_side + small_square_side
  let triangle_base : ℝ := large_square_side * small_square_side / total_width
  let triangle_area : ℝ := 1/2 * triangle_base * small_square_side
  let small_square_area : ℝ := small_square_side ^ 2
  let shaded_area : ℝ := small_square_area - triangle_area
  shaded_area = 175/8 := by sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l3601_360113


namespace NUMINAMATH_CALUDE_expansion_equals_cube_l3601_360119

theorem expansion_equals_cube : 101^3 + 3*(101^2) + 3*101 + 1 = 102^3 := by sorry

end NUMINAMATH_CALUDE_expansion_equals_cube_l3601_360119


namespace NUMINAMATH_CALUDE_initialMenCountIs8_l3601_360126

/-- The initial number of men in a group where:
  - The average age increases by 2 years when two women replace two men
  - The two men being replaced are aged 20 and 24 years
  - The average age of the women is 30 years
-/
def initialMenCount : ℕ := by
  -- Define the increase in average age
  let averageAgeIncrease : ℕ := 2
  -- Define the ages of the men being replaced
  let replacedManAge1 : ℕ := 20
  let replacedManAge2 : ℕ := 24
  -- Define the average age of the women
  let womenAverageAge : ℕ := 30
  
  -- The proof goes here
  sorry

/-- Theorem stating that the initial number of men is 8 -/
theorem initialMenCountIs8 : initialMenCount = 8 := by sorry

end NUMINAMATH_CALUDE_initialMenCountIs8_l3601_360126


namespace NUMINAMATH_CALUDE_new_average_after_adding_l3601_360115

theorem new_average_after_adding (n : ℕ) (original_avg : ℝ) (added_value : ℝ) :
  n > 0 →
  n = 15 →
  original_avg = 40 →
  added_value = 14 →
  (n * original_avg + n * added_value) / n = 54 := by
  sorry

end NUMINAMATH_CALUDE_new_average_after_adding_l3601_360115


namespace NUMINAMATH_CALUDE_fabian_walnuts_amount_l3601_360162

/-- The amount of walnuts in grams that Fabian wants to buy -/
def walnuts_amount (apple_kg : ℕ) (sugar_packs : ℕ) (total_cost : ℕ) 
  (apple_price : ℕ) (walnut_price : ℕ) (sugar_discount : ℕ) : ℕ :=
  let apple_cost := apple_kg * apple_price
  let sugar_price := apple_price - sugar_discount
  let sugar_cost := sugar_packs * sugar_price
  let walnut_cost := total_cost - apple_cost - sugar_cost
  let walnut_grams_per_dollar := 1000 / walnut_price
  walnut_cost * walnut_grams_per_dollar

/-- Theorem stating that Fabian wants to buy 500 grams of walnuts -/
theorem fabian_walnuts_amount : 
  walnuts_amount 5 3 16 2 6 1 = 500 := by
  sorry

end NUMINAMATH_CALUDE_fabian_walnuts_amount_l3601_360162


namespace NUMINAMATH_CALUDE_millet_exceeds_half_on_thursday_l3601_360111

/-- Represents the state of the bird feeder on a given day -/
structure FeederState where
  day : ℕ
  millet : ℚ
  other : ℚ

/-- Calculates the next day's feeder state -/
def nextDay (state : FeederState) : FeederState :=
  { day := state.day + 1,
    millet := state.millet / 2 + 2 / 5,
    other := 3 / 5 }

/-- Checks if millet proportion exceeds 50% -/
def milletExceedsHalf (state : FeederState) : Prop :=
  state.millet > (state.millet + state.other) / 2

/-- Initial state of the feeder on Monday -/
def initialState : FeederState :=
  { day := 1, millet := 2 / 5, other := 3 / 5 }

theorem millet_exceeds_half_on_thursday :
  let thursday := nextDay (nextDay (nextDay initialState))
  milletExceedsHalf thursday ∧
  ∀ (prevDay : FeederState), prevDay.day < thursday.day →
    ¬ milletExceedsHalf prevDay := by
  sorry

end NUMINAMATH_CALUDE_millet_exceeds_half_on_thursday_l3601_360111


namespace NUMINAMATH_CALUDE_no_integer_solutions_for_3x_minus_12y_eq_7_l3601_360142

theorem no_integer_solutions_for_3x_minus_12y_eq_7 :
  ¬ ∃ (x y : ℤ), 3 * x - 12 * y = 7 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_for_3x_minus_12y_eq_7_l3601_360142


namespace NUMINAMATH_CALUDE_expression_simplification_l3601_360170

theorem expression_simplification (a b : ℚ) (h1 : a = -1) (h2 : b = 1/4) :
  (a + 2*b)^2 + (a + 2*b)*(a - 2*b) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3601_360170


namespace NUMINAMATH_CALUDE_cell_population_after_9_days_l3601_360153

/-- Represents the growth and mortality of a cell population over time -/
def cell_population (initial_cells : ℕ) (growth_rate : ℚ) (mortality_rate : ℚ) (cycles : ℕ) : ℕ :=
  sorry

/-- Theorem stating the cell population after 9 days -/
theorem cell_population_after_9_days :
  cell_population 5 2 (9/10) 3 = 28 :=
sorry

end NUMINAMATH_CALUDE_cell_population_after_9_days_l3601_360153


namespace NUMINAMATH_CALUDE_inequality_proof_l3601_360118

theorem inequality_proof (x y z : ℝ) 
  (non_neg_x : x ≥ 0) (non_neg_y : y ≥ 0) (non_neg_z : z ≥ 0)
  (sum_squares : x^2 + y^2 + z^2 = 1) :
  x / (1 - x^2) + y / (1 - y^2) + z / (1 - z^2) ≥ 3 * Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3601_360118


namespace NUMINAMATH_CALUDE_andrews_cheese_pops_l3601_360101

theorem andrews_cheese_pops (hotdogs chicken_nuggets total : ℕ) 
  (hotdogs_count : hotdogs = 30)
  (chicken_nuggets_count : chicken_nuggets = 40)
  (total_count : total = 90)
  (sum_equation : hotdogs + chicken_nuggets + (total - hotdogs - chicken_nuggets) = total) :
  total - hotdogs - chicken_nuggets = 20 := by
  sorry

end NUMINAMATH_CALUDE_andrews_cheese_pops_l3601_360101


namespace NUMINAMATH_CALUDE_quadratic_root_existence_l3601_360152

theorem quadratic_root_existence (a b c x₁ x₂ : ℝ) (ha : a ≠ 0)
  (h1 : a * x₁^2 + b * x₁ + c = 0)
  (h2 : -a * x₂^2 + b * x₂ + c = 0) :
  ∃ x₃ : ℝ, (a / 2) * x₃^2 + b * x₃ + c = 0 ∧
    ((x₁ ≤ x₃ ∧ x₃ ≤ x₂) ∨ (x₁ ≥ x₃ ∧ x₃ ≥ x₂)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_existence_l3601_360152


namespace NUMINAMATH_CALUDE_town_population_problem_l3601_360130

theorem town_population_problem (original_population : ℕ) : 
  (((original_population + 2000) * 85 / 100) : ℕ) = original_population - 50 →
  original_population = 11667 :=
by
  sorry

end NUMINAMATH_CALUDE_town_population_problem_l3601_360130


namespace NUMINAMATH_CALUDE_relationship_abc_l3601_360163

theorem relationship_abc (a b c : ℝ) :
  (∃ u v : ℝ, u - v = a ∧ u^2 - v^2 = b ∧ u^3 - v^3 = c) →
  3 * b^2 + a^4 = 4 * a * c := by
sorry

end NUMINAMATH_CALUDE_relationship_abc_l3601_360163


namespace NUMINAMATH_CALUDE_mean_score_is_215_div_11_l3601_360151

def points : List ℕ := [15, 20, 25, 30]
def players : List ℕ := [5, 3, 2, 1]

theorem mean_score_is_215_div_11 : 
  (List.sum (List.zipWith (· * ·) points players)) / (List.sum players) = 215 / 11 := by
  sorry

end NUMINAMATH_CALUDE_mean_score_is_215_div_11_l3601_360151


namespace NUMINAMATH_CALUDE_paint_weight_l3601_360190

theorem paint_weight (total_weight : ℝ) (half_empty_weight : ℝ) 
  (h1 : total_weight = 24)
  (h2 : half_empty_weight = 14) :
  total_weight - half_empty_weight = 10 ∧ 
  2 * (total_weight - half_empty_weight) = 20 := by
  sorry

#check paint_weight

end NUMINAMATH_CALUDE_paint_weight_l3601_360190


namespace NUMINAMATH_CALUDE_f_properties_l3601_360134

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem f_properties (x₁ x₂ : ℝ) (h₁ : 0 < x₁) (h₂ : x₁ < x₂) :
  (x₂ * f x₁ < x₁ * f x₂) ∧
  (x₁ > Real.exp (-1) → x₁ * f x₁ + x₂ * f x₂ > x₂ * f x₁ + x₁ * f x₂) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l3601_360134


namespace NUMINAMATH_CALUDE_enclosing_polygons_sides_l3601_360184

/-- The number of sides of the central polygon -/
def m : ℕ := 12

/-- The number of polygons enclosing the central polygon -/
def enclosing_polygons : ℕ := 12

/-- The number of sides of each enclosing polygon -/
def n : ℕ := 12

/-- The interior angle of a regular polygon with k sides -/
def interior_angle (k : ℕ) : ℚ := (k - 2) * 180 / k

/-- The exterior angle of a regular polygon with k sides -/
def exterior_angle (k : ℕ) : ℚ := 360 / k

/-- Theorem: In a configuration where a regular polygon with m sides is exactly
    enclosed by 'enclosing_polygons' number of regular polygons each with n sides,
    the value of n must be equal to the number of sides of the central polygon. -/
theorem enclosing_polygons_sides (h1 : m = 12) (h2 : enclosing_polygons = 12) :
  n = m := by sorry

end NUMINAMATH_CALUDE_enclosing_polygons_sides_l3601_360184


namespace NUMINAMATH_CALUDE_number_sum_15_equals_96_l3601_360183

theorem number_sum_15_equals_96 : ∃ x : ℝ, x + 15 = 96 ∧ x = 81 := by
  sorry

end NUMINAMATH_CALUDE_number_sum_15_equals_96_l3601_360183


namespace NUMINAMATH_CALUDE_twelfth_term_of_ap_l3601_360147

-- Define the arithmetic progression
def arithmeticProgression (a : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

-- State the theorem
theorem twelfth_term_of_ap : arithmeticProgression 2 8 12 = 90 := by
  sorry

end NUMINAMATH_CALUDE_twelfth_term_of_ap_l3601_360147


namespace NUMINAMATH_CALUDE_root_implies_a_values_l3601_360127

theorem root_implies_a_values (a : ℝ) :
  ((-1)^2 * a^2 + 2011 * (-1) * a - 2012 = 0) →
  (a = 2012 ∨ a = -1) :=
by
  sorry

end NUMINAMATH_CALUDE_root_implies_a_values_l3601_360127


namespace NUMINAMATH_CALUDE_no_quadratic_transform_l3601_360112

/-- A polynomial function of degree 2 or less -/
def QuadraticPolynomial (a b c : ℚ) : ℚ → ℚ := λ x => a * x^2 + b * x + c

/-- Theorem stating that no quadratic polynomial can transform (1,4,7) to (1,10,7) -/
theorem no_quadratic_transform :
  ¬ ∃ (a b c : ℚ), 
    (QuadraticPolynomial a b c 1 = 1) ∧ 
    (QuadraticPolynomial a b c 4 = 10) ∧ 
    (QuadraticPolynomial a b c 7 = 7) := by
  sorry

end NUMINAMATH_CALUDE_no_quadratic_transform_l3601_360112


namespace NUMINAMATH_CALUDE_scientific_notation_of_80_million_l3601_360186

theorem scientific_notation_of_80_million :
  ∃ (n : ℕ), 80000000 = 8 * (10 ^ n) ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_80_million_l3601_360186


namespace NUMINAMATH_CALUDE_local_minimum_implies_b_range_l3601_360195

-- Define the function f(x)
def f (b : ℝ) (x : ℝ) : ℝ := x^3 - 3*b*x + 3*b

-- State the theorem
theorem local_minimum_implies_b_range :
  ∀ b : ℝ, (∃ x : ℝ, x ∈ (Set.Ioo 0 1) ∧ IsLocalMin (f b) x) → 0 < b ∧ b < 1 :=
by sorry

end NUMINAMATH_CALUDE_local_minimum_implies_b_range_l3601_360195


namespace NUMINAMATH_CALUDE_part1_part2_part3_l3601_360168

/-- Represents a supermarket's sales and profit scenario -/
structure SupermarketSales where
  initialProfit : ℝ  -- Initial profit per item
  initialSales : ℝ   -- Initial daily sales volume
  priceReduction : ℝ -- Amount of price reduction per item
  salesIncrease : ℝ  -- Increase in sales per dollar of price reduction

/-- Calculates the new sales volume after price reduction -/
def newSalesVolume (s : SupermarketSales) : ℝ :=
  s.initialSales + s.salesIncrease * s.priceReduction

/-- Calculates the new profit per item after price reduction -/
def newProfitPerItem (s : SupermarketSales) : ℝ :=
  s.initialProfit - s.priceReduction

/-- Calculates the total daily profit after price reduction -/
def totalDailyProfit (s : SupermarketSales) : ℝ :=
  newSalesVolume s * newProfitPerItem s

/-- Theorem: For given initial conditions, calculate the new sales volume and total daily profit -/
theorem part1 (s : SupermarketSales) 
    (h1 : s.initialProfit = 50)
    (h2 : s.initialSales = 30)
    (h3 : s.priceReduction = 5)
    (h4 : s.salesIncrease = 2) :
    newSalesVolume s = 40 ∧ totalDailyProfit s = 1800 := by
  sorry

/-- Theorem: Find the price reduction that achieves a specific daily profit -/
theorem part2 (s : SupermarketSales) 
    (h1 : s.initialProfit = 50)
    (h2 : s.initialSales = 30)
    (h3 : s.salesIncrease = 2) :
    ∃ x : ℝ, x > 0 ∧ totalDailyProfit {s with priceReduction := x} = 2100 := by
  sorry

/-- Theorem: Prove that a certain daily profit is unachievable -/
theorem part3 (s : SupermarketSales) 
    (h1 : s.initialProfit = 50)
    (h2 : s.initialSales = 30)
    (h3 : s.salesIncrease = 2) :
    ¬ ∃ x : ℝ, x > 0 ∧ totalDailyProfit {s with priceReduction := x} = 2200 := by
  sorry

end NUMINAMATH_CALUDE_part1_part2_part3_l3601_360168


namespace NUMINAMATH_CALUDE_quadratic_roots_l3601_360109

/-- Given a quadratic function f(x) = ax² - 2ax + c where a ≠ 0,
    if f(3) = 0, then the solutions to f(x) = 0 are x₁ = -1 and x₂ = 3 -/
theorem quadratic_roots (a c : ℝ) (ha : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 - 2 * a * x + c
  f 3 = 0 → (∀ x, f x = 0 ↔ x = -1 ∨ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_l3601_360109


namespace NUMINAMATH_CALUDE_lewis_weekly_earnings_l3601_360165

/-- Lewis's earnings during harvest -/
def harvest_earnings : ℕ := 178

/-- Duration of harvest in weeks -/
def harvest_duration : ℕ := 89

/-- Lewis's weekly earnings during harvest -/
def weekly_earnings : ℚ := harvest_earnings / harvest_duration

theorem lewis_weekly_earnings :
  weekly_earnings = 2 :=
sorry

end NUMINAMATH_CALUDE_lewis_weekly_earnings_l3601_360165


namespace NUMINAMATH_CALUDE_paper_stack_height_l3601_360150

/-- Given a ream of paper with 400 sheets that is 4 cm thick,
    prove that a stack of 6 cm will contain 600 sheets. -/
theorem paper_stack_height (sheets_per_ream : ℕ) (ream_thickness : ℝ) 
  (stack_height : ℝ) (h1 : sheets_per_ream = 400) (h2 : ream_thickness = 4) 
  (h3 : stack_height = 6) : 
  (stack_height / ream_thickness) * sheets_per_ream = 600 :=
sorry

end NUMINAMATH_CALUDE_paper_stack_height_l3601_360150


namespace NUMINAMATH_CALUDE_ratatouille_cost_per_quart_l3601_360149

/-- Calculate the cost per quart of ratatouille --/
theorem ratatouille_cost_per_quart :
  let eggplant_weight : ℝ := 5.5
  let eggplant_price : ℝ := 2.20
  let zucchini_weight : ℝ := 3.8
  let zucchini_price : ℝ := 1.85
  let tomato_weight : ℝ := 4.6
  let tomato_price : ℝ := 3.75
  let onion_weight : ℝ := 2.7
  let onion_price : ℝ := 1.10
  let basil_weight : ℝ := 1.0
  let basil_price : ℝ := 2.70 * 4  -- Price per pound (4 quarters)
  let pepper_weight : ℝ := 0.75
  let pepper_price : ℝ := 3.15
  let total_yield : ℝ := 4.5

  let total_cost : ℝ := 
    eggplant_weight * eggplant_price +
    zucchini_weight * zucchini_price +
    tomato_weight * tomato_price +
    onion_weight * onion_price +
    basil_weight * basil_price +
    pepper_weight * pepper_price

  let cost_per_quart : ℝ := total_cost / total_yield

  cost_per_quart = 11.67 := by sorry

end NUMINAMATH_CALUDE_ratatouille_cost_per_quart_l3601_360149


namespace NUMINAMATH_CALUDE_bird_round_trips_l3601_360197

/-- Given two birds collecting nest materials, this theorem proves the number of round trips each bird made. -/
theorem bird_round_trips (distance_to_materials : ℕ) (total_distance : ℕ) : 
  distance_to_materials = 200 →
  total_distance = 8000 →
  ∃ (trips_per_bird : ℕ), 
    trips_per_bird * 2 * (2 * distance_to_materials) = total_distance ∧
    trips_per_bird = 10 := by
  sorry

end NUMINAMATH_CALUDE_bird_round_trips_l3601_360197


namespace NUMINAMATH_CALUDE_strip_overlap_area_l3601_360128

theorem strip_overlap_area (β : Real) : 
  let strip1_width : Real := 1
  let strip2_width : Real := 2
  let circle_radius : Real := 1
  let rhombus_area : Real := (1/2) * strip1_width * strip2_width * Real.sin β
  let circle_area : Real := Real.pi * circle_radius^2
  rhombus_area - circle_area = Real.sin β - Real.pi := by sorry

end NUMINAMATH_CALUDE_strip_overlap_area_l3601_360128


namespace NUMINAMATH_CALUDE_smallest_x_satisfying_abs_equation_l3601_360124

theorem smallest_x_satisfying_abs_equation : 
  ∃ x : ℝ, (∀ y : ℝ, |5*y + 2| = 28 → x ≤ y) ∧ |5*x + 2| = 28 := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_satisfying_abs_equation_l3601_360124


namespace NUMINAMATH_CALUDE_average_transformation_l3601_360172

theorem average_transformation (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h : (x₁ + x₂ + x₃ + x₄ + x₅) / 5 = 2) : 
  ((3*x₁ + 1) + (3*x₂ + 1) + (3*x₃ + 1) + (3*x₄ + 1) + (3*x₅ + 1)) / 5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_average_transformation_l3601_360172


namespace NUMINAMATH_CALUDE_S_is_three_rays_l3601_360116

/-- The set S of points (x,y) in the coordinate plane satisfying the given conditions -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (4 = x + 1 ∧ y - 3 ≤ 4) ∨
               (4 = y - 3 ∧ x + 1 ≤ 4) ∨
               (x + 1 = y - 3 ∧ 4 ≤ x + 1)}

/-- A ray starting from a point in a given direction -/
def Ray (start : ℝ × ℝ) (direction : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, t ≥ 0 ∧ p = (start.1 + t * direction.1, start.2 + t * direction.2)}

/-- The theorem stating that S consists of three rays with a common point -/
theorem S_is_three_rays :
  ∃ (r₁ r₂ r₃ : Set (ℝ × ℝ)) (common_point : ℝ × ℝ),
    S = r₁ ∪ r₂ ∪ r₃ ∧
    (∃ d₁ d₂ d₃ : ℝ × ℝ, r₁ = Ray common_point d₁ ∧
                         r₂ = Ray common_point d₂ ∧
                         r₃ = Ray common_point d₃) ∧
    common_point = (3, 7) := by
  sorry

end NUMINAMATH_CALUDE_S_is_three_rays_l3601_360116


namespace NUMINAMATH_CALUDE_martha_collected_90_cans_l3601_360129

/-- The number of cans Martha collected -/
def martha_cans : ℕ := sorry

/-- The number of cans Diego collected -/
def diego_cans (m : ℕ) : ℕ := m / 2 + 10

/-- The total number of cans collected -/
def total_cans : ℕ := 145

theorem martha_collected_90_cans :
  martha_cans = 90 ∧ 
  diego_cans martha_cans = martha_cans / 2 + 10 ∧
  martha_cans + diego_cans martha_cans = total_cans :=
sorry

end NUMINAMATH_CALUDE_martha_collected_90_cans_l3601_360129


namespace NUMINAMATH_CALUDE_least_positive_multiple_least_positive_multiple_when_x_24_l3601_360102

theorem least_positive_multiple (x y : ℤ) : ∃ (k : ℤ), k > 0 ∧ k * (x + 16 * y) = 8 ∧ ∀ (m : ℤ), m > 0 ∧ (∃ (n : ℤ), m * (x + 16 * y) = n * 8) → m ≥ k :=
  by sorry

theorem least_positive_multiple_when_x_24 : ∃ (k : ℤ), k > 0 ∧ k * (24 + 16 * (-1)) = 8 ∧ ∀ (m : ℤ), m > 0 ∧ (∃ (n : ℤ), m * (24 + 16 * (-1)) = n * 8) → m ≥ k :=
  by sorry

end NUMINAMATH_CALUDE_least_positive_multiple_least_positive_multiple_when_x_24_l3601_360102


namespace NUMINAMATH_CALUDE_modified_pattern_cannot_form_polyhedron_l3601_360141

/-- Represents a flat pattern of squares -/
structure FlatPattern where
  squares : ℕ
  foldingLines : ℕ

/-- Represents a modified flat pattern with an extra square and a removed folding line -/
def ModifiedPattern (fp : FlatPattern) : FlatPattern :=
  { squares := fp.squares + 1
  , foldingLines := fp.foldingLines - 1 }

/-- Represents whether a pattern can form a simple polyhedron -/
def CanFormPolyhedron (fp : FlatPattern) : Prop := sorry

/-- Theorem stating that a modified pattern cannot form a simple polyhedron -/
theorem modified_pattern_cannot_form_polyhedron (fp : FlatPattern) : 
  ¬(CanFormPolyhedron (ModifiedPattern fp)) := by
  sorry

end NUMINAMATH_CALUDE_modified_pattern_cannot_form_polyhedron_l3601_360141


namespace NUMINAMATH_CALUDE_projected_revenue_increase_l3601_360198

theorem projected_revenue_increase (last_year_revenue : ℝ) :
  let actual_revenue := 0.9 * last_year_revenue
  let projected_revenue := last_year_revenue * (1 + 0.2)
  actual_revenue = 0.75 * projected_revenue :=
by sorry

end NUMINAMATH_CALUDE_projected_revenue_increase_l3601_360198


namespace NUMINAMATH_CALUDE_square_perimeter_from_rectangle_division_l3601_360107

/-- Given a square divided into four congruent rectangles, each with its longer side
    parallel to the sides of the square and having a perimeter of 40 inches,
    the perimeter of the square is 64 inches. -/
theorem square_perimeter_from_rectangle_division (s : ℝ) :
  s > 0 →
  (2 * (s + s/4) = 40) →
  (4 * s = 64) :=
by sorry

end NUMINAMATH_CALUDE_square_perimeter_from_rectangle_division_l3601_360107


namespace NUMINAMATH_CALUDE_hyperbola_a_plus_h_l3601_360199

/-- A hyperbola with given asymptotes and a point it passes through -/
structure Hyperbola where
  -- Asymptote equations
  asymptote1 : Real → Real
  asymptote2 : Real → Real
  -- Point the hyperbola passes through
  point : Real × Real
  -- Conditions on asymptotes
  asymptote1_eq : ∀ x, asymptote1 x = 2 * x + 5
  asymptote2_eq : ∀ x, asymptote2 x = -2 * x + 1
  -- Condition on the point
  point_eq : point = (0, 7)

/-- The standard form of a hyperbola -/
def standard_form (h k a b : Real) (x y : Real) : Prop :=
  (y - k)^2 / a^2 - (x - h)^2 / b^2 = 1

/-- Theorem stating the sum of a and h for the given hyperbola -/
theorem hyperbola_a_plus_h (H : Hyperbola) :
  ∃ (h k a b : Real), a > 0 ∧ b > 0 ∧
  (∀ x y, standard_form h k a b x y ↔ H.point = (x, y)) →
  a + h = 2 * Real.sqrt 3 - 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_a_plus_h_l3601_360199


namespace NUMINAMATH_CALUDE_finite_perfect_squares_y_l3601_360135

theorem finite_perfect_squares_y (x : ℕ) : 
  let y := x^4 + 2*x^3 + 2*x^2 + 2*x + 1
  Finite {x : ℕ | ∃ (z : ℕ), y = z^2} :=
sorry

end NUMINAMATH_CALUDE_finite_perfect_squares_y_l3601_360135


namespace NUMINAMATH_CALUDE_matrix_fourth_power_l3601_360140

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -1; 1, 1]

theorem matrix_fourth_power :
  A ^ 4 = !![0, -9; 9, -9] := by sorry

end NUMINAMATH_CALUDE_matrix_fourth_power_l3601_360140


namespace NUMINAMATH_CALUDE_elevator_capacity_l3601_360196

theorem elevator_capacity (adult_avg_weight child_avg_weight next_person_max_weight : ℝ)
  (num_adults num_children : ℕ) :
  adult_avg_weight = 140 →
  child_avg_weight = 64 →
  next_person_max_weight = 52 →
  num_adults = 3 →
  num_children = 2 →
  (num_adults : ℝ) * adult_avg_weight + (num_children : ℝ) * child_avg_weight + next_person_max_weight = 600 :=
by sorry

end NUMINAMATH_CALUDE_elevator_capacity_l3601_360196


namespace NUMINAMATH_CALUDE_solution_value_l3601_360175

theorem solution_value (a b : ℝ) : 
  (2 * 2 * a - 2 * b - 20 = 0) → (2023 - 2 * a + b = 2013) := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l3601_360175


namespace NUMINAMATH_CALUDE_average_sale_is_6900_l3601_360178

def sales : List ℕ := [6435, 6927, 6855, 7230, 6562, 7391]

theorem average_sale_is_6900 :
  (sales.sum : ℚ) / sales.length = 6900 := by
  sorry

end NUMINAMATH_CALUDE_average_sale_is_6900_l3601_360178


namespace NUMINAMATH_CALUDE_complex_power_to_rectangular_l3601_360138

theorem complex_power_to_rectangular : 
  (3 * (Complex.cos (30 * π / 180) + Complex.I * Complex.sin (30 * π / 180)))^4 = 
    Complex.mk (-40.5) (40.5 * Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_complex_power_to_rectangular_l3601_360138


namespace NUMINAMATH_CALUDE_max_min_difference_c_l3601_360100

theorem max_min_difference_c (a b c : ℝ) 
  (sum_eq : a + b + c = 6) 
  (sum_sq_eq : a^2 + b^2 + c^2 = 18) : 
  (∃ (a₁ b₁ : ℝ), a₁ + b₁ + 6 = 6 ∧ a₁^2 + b₁^2 + 6^2 = 18) ∧ 
  (∃ (a₂ b₂ : ℝ), a₂ + b₂ + (-2) = 6 ∧ a₂^2 + b₂^2 + (-2)^2 = 18) ∧
  (∀ (a₃ b₃ c₃ : ℝ), a₃ + b₃ + c₃ = 6 → a₃^2 + b₃^2 + c₃^2 = 18 → c₃ ≤ 6 ∧ c₃ ≥ -2) ∧
  (6 - (-2) = 8) := by
  sorry

end NUMINAMATH_CALUDE_max_min_difference_c_l3601_360100


namespace NUMINAMATH_CALUDE_sample_size_for_295_students_l3601_360156

/-- Calculates the sample size for systematic sampling --/
def calculateSampleSize (totalStudents : Nat) (samplingRatio : Nat) : Nat :=
  totalStudents / samplingRatio

/-- Theorem: The sample size for 295 students with a 1:5 sampling ratio is 59 --/
theorem sample_size_for_295_students :
  calculateSampleSize 295 5 = 59 := by
  sorry


end NUMINAMATH_CALUDE_sample_size_for_295_students_l3601_360156


namespace NUMINAMATH_CALUDE_dawsons_b_students_l3601_360157

/-- Proves that given the conditions from the problem, the number of students
    receiving a 'B' in Mr. Dawson's class is 18. -/
theorem dawsons_b_students
  (carter_total : ℕ)
  (carter_b : ℕ)
  (dawson_total : ℕ)
  (h1 : carter_total = 20)
  (h2 : carter_b = 12)
  (h3 : dawson_total = 30)
  (h4 : (carter_b : ℚ) / carter_total = dawson_b / dawson_total) :
  dawson_b = 18 := by
  sorry

#check dawsons_b_students

end NUMINAMATH_CALUDE_dawsons_b_students_l3601_360157


namespace NUMINAMATH_CALUDE_factor_expression_l3601_360166

theorem factor_expression (x y z : ℝ) :
  ((x^2 - y^2)^3 + (y^2 - z^2)^3 + (z^2 - x^2)^3) / ((x - y)^3 + (y - z)^3 + (z - x)^3) = (x + y) * (y + z) * (z + x) :=
by sorry

end NUMINAMATH_CALUDE_factor_expression_l3601_360166


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l3601_360123

theorem inequality_and_equality_condition (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_product : a * b * c = 1/8) :
  a^2 + b^2 + c^2 + a^2*b^2 + a^2*c^2 + b^2*c^2 ≥ 15/16 ∧
  (a^2 + b^2 + c^2 + a^2*b^2 + a^2*c^2 + b^2*c^2 = 15/16 ↔ a = 1/2 ∧ b = 1/2 ∧ c = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l3601_360123


namespace NUMINAMATH_CALUDE_relay_team_selection_l3601_360120

/-- The number of sprinters available for selection -/
def total_sprinters : ℕ := 6

/-- The number of athletes needed for the relay race -/
def selected_athletes : ℕ := 4

/-- The number of mandatory athletes (A and B) -/
def mandatory_athletes : ℕ := 2

/-- The number of positions in the relay race -/
def race_positions : ℕ := 4

/-- The number of different arrangements for the relay race -/
def relay_arrangements : ℕ := 72

theorem relay_team_selection :
  (total_sprinters.choose (selected_athletes - mandatory_athletes)) *
  (mandatory_athletes.choose 1) *
  ((selected_athletes - 1).factorial) = relay_arrangements :=
sorry

end NUMINAMATH_CALUDE_relay_team_selection_l3601_360120


namespace NUMINAMATH_CALUDE_six_player_four_games_tournament_l3601_360194

/-- Represents a chess tournament --/
structure ChessTournament where
  numPlayers : Nat
  gamesPerPlayer : Nat

/-- Calculates the total number of games in a chess tournament --/
def totalGames (t : ChessTournament) : Nat :=
  (t.numPlayers * t.gamesPerPlayer) / 2

/-- Theorem: In a tournament with 6 players where each plays 4 others, there are 10 games total --/
theorem six_player_four_games_tournament :
  ∀ (t : ChessTournament),
    t.numPlayers = 6 →
    t.gamesPerPlayer = 4 →
    totalGames t = 10 := by
  sorry

#check six_player_four_games_tournament

end NUMINAMATH_CALUDE_six_player_four_games_tournament_l3601_360194


namespace NUMINAMATH_CALUDE_greatest_perimeter_of_special_triangle_l3601_360137

theorem greatest_perimeter_of_special_triangle :
  ∀ a b : ℕ,
    a > 0 →
    b > 0 →
    b = 2 * a →
    17 + a > b →
    b + 17 > a →
    a + b > 17 →
    a + b + 17 ≤ 65 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_perimeter_of_special_triangle_l3601_360137


namespace NUMINAMATH_CALUDE_perfume_cost_calculation_l3601_360132

/-- The cost of a bottle of perfume given initial savings, earnings from jobs, and additional amount needed --/
def perfume_cost (christian_initial : ℕ) (sue_initial : ℕ) 
                 (yards_mowed : ℕ) (yard_price : ℕ) 
                 (dogs_walked : ℕ) (dog_price : ℕ) 
                 (additional_needed : ℕ) : ℕ :=
  christian_initial + sue_initial + 
  yards_mowed * yard_price + 
  dogs_walked * dog_price + 
  additional_needed

/-- Theorem stating the cost of the perfume given the problem conditions --/
theorem perfume_cost_calculation : 
  perfume_cost 5 7 4 5 6 2 6 = 50 := by
  sorry

end NUMINAMATH_CALUDE_perfume_cost_calculation_l3601_360132


namespace NUMINAMATH_CALUDE_jacque_suitcase_weight_l3601_360167

/-- The weight of Jacque's suitcase when he arrived in France -/
def initial_weight : ℝ := 5

/-- The weight of one bottle of perfume in ounces -/
def perfume_weight : ℝ := 1.2

/-- The number of bottles of perfume Jacque bought -/
def perfume_count : ℕ := 5

/-- The weight of chocolate in pounds -/
def chocolate_weight : ℝ := 4

/-- The weight of one bar of soap in ounces -/
def soap_weight : ℝ := 5

/-- The number of bars of soap Jacque bought -/
def soap_count : ℕ := 2

/-- The weight of one jar of jam in ounces -/
def jam_weight : ℝ := 8

/-- The number of jars of jam Jacque bought -/
def jam_count : ℕ := 2

/-- The number of ounces in a pound -/
def ounces_per_pound : ℝ := 16

/-- The total weight of Jacque's suitcase on the return flight in pounds -/
def return_weight : ℝ := 11

theorem jacque_suitcase_weight :
  initial_weight + 
  (perfume_weight * perfume_count + soap_weight * soap_count + jam_weight * jam_count) / ounces_per_pound + 
  chocolate_weight = return_weight := by
  sorry

end NUMINAMATH_CALUDE_jacque_suitcase_weight_l3601_360167


namespace NUMINAMATH_CALUDE_triangle_arithmetic_angle_sequence_l3601_360174

theorem triangle_arithmetic_angle_sequence (A B C : ℝ) : 
  A + B + C = 180 →  -- Sum of angles in a triangle is 180°
  2 * B = A + C →    -- Angles form an arithmetic sequence
  (max A (max B C) + min A (min B C) = 120) := by
sorry

end NUMINAMATH_CALUDE_triangle_arithmetic_angle_sequence_l3601_360174


namespace NUMINAMATH_CALUDE_position_of_2007_l3601_360154

/-- Represents the position of a number in the table -/
structure Position where
  row : ℕ
  column : ℕ

/-- The arrangement of positive odd numbers in 5 columns -/
def arrangement (n : ℕ) : Position :=
  let cycle := (n - 1) / 8
  let position := (n - 1) % 8
  match position with
  | 0 => ⟨cycle * 2 + 1, 2⟩
  | 1 => ⟨cycle * 2 + 1, 3⟩
  | 2 => ⟨cycle * 2 + 1, 4⟩
  | 3 => ⟨cycle * 2 + 1, 5⟩
  | 4 => ⟨cycle * 2 + 2, 1⟩
  | 5 => ⟨cycle * 2 + 2, 2⟩
  | 6 => ⟨cycle * 2 + 2, 3⟩
  | 7 => ⟨cycle * 2 + 2, 4⟩
  | _ => ⟨0, 0⟩  -- This case should never occur

theorem position_of_2007 : arrangement 2007 = ⟨251, 5⟩ := by
  sorry

end NUMINAMATH_CALUDE_position_of_2007_l3601_360154


namespace NUMINAMATH_CALUDE_eight_digit_repeating_divisible_by_10001_l3601_360181

/-- An 8-digit positive integer whose first four digits are the same as its last four digits -/
def EightDigitRepeating (n : ℕ) : Prop :=
  10000000 ≤ n ∧ n < 100000000 ∧
  ∃ (a b c d : ℕ), a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧
    n = 10000000 * a + 1000000 * b + 100000 * c + 10000 * d +
        1000 * a + 100 * b + 10 * c + d

theorem eight_digit_repeating_divisible_by_10001 (n : ℕ) (h : EightDigitRepeating n) :
  10001 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_eight_digit_repeating_divisible_by_10001_l3601_360181


namespace NUMINAMATH_CALUDE_max_training_cost_l3601_360144

def training_cost (x : ℕ) : ℕ :=
  if x ≤ 30 then 1400 * x
  else 2000 * x - 20 * x * x

theorem max_training_cost :
  ∃ (x : ℕ), x ≤ 60 ∧ ∀ (y : ℕ), y ≤ 60 → training_cost y ≤ training_cost x ∧ training_cost x = 50000 := by
  sorry

end NUMINAMATH_CALUDE_max_training_cost_l3601_360144


namespace NUMINAMATH_CALUDE_max_condition_implies_a_range_l3601_360114

/-- Given a function f with derivative f'(x) = a(x+1)(x-a), 
    if f has a maximum at x = a, then a is in the open interval (-1, 0) -/
theorem max_condition_implies_a_range (f : ℝ → ℝ) (a : ℝ) 
  (h_deriv : ∀ x, deriv f x = a * (x + 1) * (x - a))
  (h_max : IsLocalMax f a) : 
  a ∈ Set.Ioo (-1 : ℝ) 0 := by
sorry

end NUMINAMATH_CALUDE_max_condition_implies_a_range_l3601_360114


namespace NUMINAMATH_CALUDE_dodecahedron_edge_probability_l3601_360192

/-- A regular dodecahedron -/
structure RegularDodecahedron where
  /-- The number of vertices in a regular dodecahedron -/
  num_vertices : ℕ
  /-- The number of edges connected to each vertex -/
  edges_per_vertex : ℕ
  /-- Properties of a regular dodecahedron -/
  vertex_count : num_vertices = 20
  edge_count : edges_per_vertex = 3

/-- The probability of randomly choosing two vertices that form an edge in a regular dodecahedron -/
def edge_probability (d : RegularDodecahedron) : ℚ :=
  3 / 19

/-- Theorem stating the probability of randomly choosing two vertices that form an edge in a regular dodecahedron -/
theorem dodecahedron_edge_probability (d : RegularDodecahedron) :
  edge_probability d = 3 / 19 := by
  sorry

end NUMINAMATH_CALUDE_dodecahedron_edge_probability_l3601_360192


namespace NUMINAMATH_CALUDE_total_weight_is_540_l3601_360122

def back_squat_initial : ℝ := 200
def back_squat_increase : ℝ := 50
def front_squat_ratio : ℝ := 0.8
def triple_ratio : ℝ := 0.9
def number_of_triples : ℕ := 3

def calculate_total_weight : ℝ :=
  let back_squat_new := back_squat_initial + back_squat_increase
  let front_squat := back_squat_new * front_squat_ratio
  let triple_weight := front_squat * triple_ratio
  triple_weight * number_of_triples

theorem total_weight_is_540 :
  calculate_total_weight = 540 := by sorry

end NUMINAMATH_CALUDE_total_weight_is_540_l3601_360122


namespace NUMINAMATH_CALUDE_cloth_cost_price_calculation_l3601_360191

/-- Given the selling price and loss per metre for a certain length of cloth,
    calculate the cost price per metre. -/
def cost_price_per_metre (selling_price total_length loss_per_metre : ℚ) : ℚ :=
  (selling_price + loss_per_metre * total_length) / total_length

/-- Theorem stating that under the given conditions, 
    the cost price per metre of cloth is 95. -/
theorem cloth_cost_price_calculation :
  let selling_price : ℚ := 18000
  let total_length : ℚ := 200
  let loss_per_metre : ℚ := 5
  cost_price_per_metre selling_price total_length loss_per_metre = 95 :=
by
  sorry


end NUMINAMATH_CALUDE_cloth_cost_price_calculation_l3601_360191


namespace NUMINAMATH_CALUDE_min_value_xy_plus_two_over_xy_l3601_360155

theorem min_value_xy_plus_two_over_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (∀ z w : ℝ, z > 0 → w > 0 → z + w = 1 → x * y + 2 / (x * y) ≤ z * w + 2 / (z * w)) ∧
  x * y + 2 / (x * y) = 33 / 4 := by
sorry

end NUMINAMATH_CALUDE_min_value_xy_plus_two_over_xy_l3601_360155


namespace NUMINAMATH_CALUDE_equation_rewrite_l3601_360110

theorem equation_rewrite :
  ∃ (m n : ℝ), (∀ x, x^2 - 12*x + 33 = 0 ↔ (x + m)^2 = n) ∧ m = -6 ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_rewrite_l3601_360110


namespace NUMINAMATH_CALUDE_expression_factorization_l3601_360121

theorem expression_factorization (x : ℝ) :
  (20 * x^3 - 100 * x^2 + 30) - (5 * x^3 - 10 * x^2 + 3) = 3 * (5 * x^2 * (x - 6) + 9) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l3601_360121


namespace NUMINAMATH_CALUDE_simplest_fraction_sum_l3601_360171

theorem simplest_fraction_sum (p q : ℕ+) : 
  (p : ℚ) / q = 83125 / 100000 ∧ 
  ∀ (a b : ℕ+), (a : ℚ) / b = p / q → a ≤ p ∧ b ≤ q →
  p + q = 293 := by
sorry

end NUMINAMATH_CALUDE_simplest_fraction_sum_l3601_360171


namespace NUMINAMATH_CALUDE_average_of_numbers_l3601_360173

def numbers : List ℝ := [12, 14, 510, 520, 530, 1115, 1120, 1, 1252140, 2345]

theorem average_of_numbers : 
  (numbers.sum / numbers.length : ℝ) = 125830.7 := by sorry

end NUMINAMATH_CALUDE_average_of_numbers_l3601_360173


namespace NUMINAMATH_CALUDE_circular_road_width_l3601_360164

theorem circular_road_width 
  (inner_radius outer_radius : ℝ) 
  (h1 : 2 * Real.pi * inner_radius + 2 * Real.pi * outer_radius = 88) 
  (h2 : inner_radius = (1/3) * outer_radius) : 
  outer_radius - inner_radius = 22 / Real.pi := by
sorry

end NUMINAMATH_CALUDE_circular_road_width_l3601_360164


namespace NUMINAMATH_CALUDE_max_abs_sum_on_circle_l3601_360105

theorem max_abs_sum_on_circle (x y : ℝ) (h : x^2 + y^2 = 4) :
  |x| + |y| ≤ 2 * Real.sqrt 2 ∧ ∃ x y : ℝ, x^2 + y^2 = 4 ∧ |x| + |y| = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_abs_sum_on_circle_l3601_360105


namespace NUMINAMATH_CALUDE_unique_intersecting_line_l3601_360117

/-- A line in 3D space -/
structure Line3D where
  -- Define a line using two points
  point1 : ℝ × ℝ × ℝ
  point2 : ℝ × ℝ × ℝ
  ne : point1 ≠ point2

/-- Two lines are skew if they are not parallel and do not intersect -/
def are_skew (l1 l2 : Line3D) : Prop :=
  -- Definition of skew lines
  sorry

/-- A line intersects another line -/
def intersects (l1 l2 : Line3D) : Prop :=
  -- Definition of line intersection
  sorry

theorem unique_intersecting_line (a b c : Line3D) 
  (hab : are_skew a b) (hbc : are_skew b c) (hac : are_skew a c) :
  ∃! l : Line3D, intersects l a ∧ intersects l b ∧ intersects l c :=
sorry

end NUMINAMATH_CALUDE_unique_intersecting_line_l3601_360117


namespace NUMINAMATH_CALUDE_unique_number_l3601_360143

def is_three_digit_number (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def has_distinct_digits (n : ℕ) : Prop :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let units := n % 10
  hundreds ≠ tens ∧ hundreds ≠ units ∧ tens ≠ units

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

def person_a_initially_unsure (n : ℕ) : Prop :=
  ∃ m : ℕ, m ≠ n ∧ 
    is_three_digit_number m ∧ 
    is_perfect_square m ∧ 
    has_distinct_digits m ∧ 
    m / 100 = n / 100

def person_b_knows_a_unsure (n : ℕ) : Prop :=
  ∀ m : ℕ, (m / 10) % 10 = (n / 10) % 10 → person_a_initially_unsure m

def person_c_knows_number (n : ℕ) : Prop :=
  ∀ m : ℕ, m ≠ n →
    ¬(is_three_digit_number m ∧ 
      is_perfect_square m ∧ 
      has_distinct_digits m ∧ 
      person_a_initially_unsure m ∧ 
      person_b_knows_a_unsure m ∧ 
      m % 10 = n % 10)

def person_a_knows_after_c (n : ℕ) : Prop :=
  ∀ m : ℕ, m ≠ n →
    ¬(is_three_digit_number m ∧ 
      is_perfect_square m ∧ 
      has_distinct_digits m ∧ 
      person_a_initially_unsure m ∧ 
      person_b_knows_a_unsure m ∧ 
      person_c_knows_number m ∧ 
      m / 100 = n / 100)

def person_b_knows_after_a (n : ℕ) : Prop :=
  ∀ m : ℕ, m ≠ n →
    ¬(is_three_digit_number m ∧ 
      is_perfect_square m ∧ 
      has_distinct_digits m ∧ 
      person_a_initially_unsure m ∧ 
      person_b_knows_a_unsure m ∧ 
      person_c_knows_number m ∧ 
      person_a_knows_after_c m ∧ 
      (m / 10) % 10 = (n / 10) % 10)

theorem unique_number : 
  ∃! n : ℕ, 
    is_three_digit_number n ∧ 
    is_perfect_square n ∧ 
    has_distinct_digits n ∧ 
    person_a_initially_unsure n ∧ 
    person_b_knows_a_unsure n ∧ 
    person_c_knows_number n ∧ 
    person_a_knows_after_c n ∧ 
    person_b_knows_after_a n ∧ 
    n = 289 := by sorry

end NUMINAMATH_CALUDE_unique_number_l3601_360143


namespace NUMINAMATH_CALUDE_vegetable_sale_ratio_l3601_360159

theorem vegetable_sale_ratio : 
  let carrots : ℝ := 15
  let zucchini : ℝ := 13
  let broccoli : ℝ := 8
  let total_installed : ℝ := carrots + zucchini + broccoli
  let sold : ℝ := 18
  sold / total_installed = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_vegetable_sale_ratio_l3601_360159


namespace NUMINAMATH_CALUDE_cone_height_l3601_360108

theorem cone_height (r : ℝ) (h : ℝ) :
  r = 1 →
  (2 * Real.pi * r = (2 * Real.pi / 3) * 3) →
  h = Real.sqrt (3^2 - r^2) →
  h = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_cone_height_l3601_360108


namespace NUMINAMATH_CALUDE_expected_attacked_squares_theorem_l3601_360177

/-- The number of squares on a chessboard -/
def chessboardSize : ℕ := 64

/-- The number of rooks placed on the board -/
def numRooks : ℕ := 3

/-- The probability that a specific square is not attacked by a single rook -/
def probNotAttacked : ℚ := (49 : ℚ) / 64

/-- The expected number of squares under attack by rooks on a chessboard -/
def expectedAttackedSquares : ℚ :=
  chessboardSize * (1 - probNotAttacked ^ numRooks)

/-- Theorem stating the expected number of squares under attack -/
theorem expected_attacked_squares_theorem :
  expectedAttackedSquares = 64 * (1 - (49/64)^3) :=
sorry

end NUMINAMATH_CALUDE_expected_attacked_squares_theorem_l3601_360177


namespace NUMINAMATH_CALUDE_abcd_power_2018_l3601_360160

theorem abcd_power_2018 (a b c d : ℝ) 
  (ha : (5 : ℝ) ^ a = 4)
  (hb : (4 : ℝ) ^ b = 3)
  (hc : (3 : ℝ) ^ c = 2)
  (hd : (2 : ℝ) ^ d = 5) :
  (a * b * c * d) ^ 2018 = 1 := by
  sorry

end NUMINAMATH_CALUDE_abcd_power_2018_l3601_360160


namespace NUMINAMATH_CALUDE_triangle_properties_l3601_360136

theorem triangle_properties (A B C : Real) (a b c : Real) :
  -- Triangle ABC is acute
  0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2 →
  -- a, b, c are sides opposite to A, B, C
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Given equation
  1 + (Real.sqrt 3 / 3) * Real.sin (2 * A) = 2 * (Real.sin ((B + C) / 2))^2 →
  -- Radius of circumcircle
  2 * Real.sqrt 3 = 2 * a / Real.sin A →
  -- Prove A = π/3
  A = π/3 ∧
  -- Prove maximum area is 9√3
  (1/2) * b * c * Real.sin A ≤ 9 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3601_360136


namespace NUMINAMATH_CALUDE_school_relationship_l3601_360133

/-- In a school with teachers and students, prove the relationship between
    the number of teachers, students, students per teacher, and teachers per student. -/
theorem school_relationship (m n k ℓ : ℕ) 
  (h1 : m > 0) 
  (h2 : n > 0) 
  (h3 : k > 0) 
  (h4 : ℓ > 0) 
  (teacher_students : ∀ t, t ≤ m → (∃ s, s ≤ n ∧ s = k))
  (student_teachers : ∀ s, s ≤ n → (∃ t, t ≤ m ∧ t = ℓ)) :
  m * k = n * ℓ := by
  sorry


end NUMINAMATH_CALUDE_school_relationship_l3601_360133


namespace NUMINAMATH_CALUDE_journey_fraction_by_rail_l3601_360103

theorem journey_fraction_by_rail 
  (total_journey : ℝ) 
  (bus_fraction : ℝ) 
  (foot_distance : ℝ) : 
  total_journey = 130 ∧ 
  bus_fraction = 17/20 ∧ 
  foot_distance = 6.5 → 
  (total_journey - bus_fraction * total_journey - foot_distance) / total_journey = 1/10 := by
sorry

end NUMINAMATH_CALUDE_journey_fraction_by_rail_l3601_360103


namespace NUMINAMATH_CALUDE_equal_one_two_digit_prob_l3601_360148

-- Define a 12-sided die
def twelveSidedDie : Finset ℕ := Finset.range 12

-- Define one-digit numbers on the die
def oneDigitNumbers : Finset ℕ := Finset.filter (λ n => n < 10) twelveSidedDie

-- Define two-digit numbers on the die
def twoDigitNumbers : Finset ℕ := Finset.filter (λ n => n ≥ 10) twelveSidedDie

-- Define the probability of rolling a one-digit number
def probOneDigit : ℚ := (oneDigitNumbers.card : ℚ) / (twelveSidedDie.card : ℚ)

-- Define the probability of rolling a two-digit number
def probTwoDigit : ℚ := (twoDigitNumbers.card : ℚ) / (twelveSidedDie.card : ℚ)

-- Theorem stating the probability of rolling 4 dice and getting an equal number of one-digit and two-digit numbers
theorem equal_one_two_digit_prob : 
  (Finset.card oneDigitNumbers * Finset.card twoDigitNumbers * 6 : ℚ) / (twelveSidedDie.card ^ 4 : ℚ) = 27 / 128 :=
by sorry

end NUMINAMATH_CALUDE_equal_one_two_digit_prob_l3601_360148


namespace NUMINAMATH_CALUDE_distance_to_origin_l3601_360182

/-- Given that point A has coordinates (√3, 2, 5) and its projection on the x-axis is (√3, 0, 0),
    prove that the distance from A to the origin is 4√2. -/
theorem distance_to_origin (A : ℝ × ℝ × ℝ) (h : A = (Real.sqrt 3, 2, 5)) :
  Real.sqrt ((Real.sqrt 3)^2 + 2^2 + 5^2) = 4 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_distance_to_origin_l3601_360182


namespace NUMINAMATH_CALUDE_hexagon_circumscribable_l3601_360139

-- Define a hexagon
structure Hexagon where
  vertices : Fin 6 → ℝ × ℝ

-- Define the property of parallel opposite sides
def has_parallel_opposite_sides (h : Hexagon) : Prop :=
  ∀ i : Fin 3, 
    let v1 := h.vertices i
    let v2 := h.vertices ((i + 1) % 6)
    let v3 := h.vertices ((i + 3) % 6)
    let v4 := h.vertices ((i + 4) % 6)
    (v2.1 - v1.1) * (v4.2 - v3.2) = (v2.2 - v1.2) * (v4.1 - v3.1)

-- Define the property of equal diagonals
def has_equal_diagonals (h : Hexagon) : Prop :=
  ∀ i : Fin 3,
    let v1 := h.vertices i
    let v2 := h.vertices ((i + 3) % 6)
    (v1.1 - v2.1)^2 + (v1.2 - v2.2)^2 = 
    (h.vertices 0).1^2 + (h.vertices 0).2^2 + 
    (h.vertices 3).1^2 + (h.vertices 3).2^2 - 
    2 * ((h.vertices 0).1 * (h.vertices 3).1 + (h.vertices 0).2 * (h.vertices 3).2)

-- Theorem statement
theorem hexagon_circumscribable 
  (h : Hexagon) 
  (parallel : has_parallel_opposite_sides h) 
  (equal_diagonals : has_equal_diagonals h) : 
  ∃ (center : ℝ × ℝ) (radius : ℝ), 
    ∀ i : Fin 6, 
      (h.vertices i).1^2 + (h.vertices i).2^2 - 
      2 * (center.1 * (h.vertices i).1 + center.2 * (h.vertices i).2) + 
      center.1^2 + center.2^2 = radius^2 :=
sorry

end NUMINAMATH_CALUDE_hexagon_circumscribable_l3601_360139


namespace NUMINAMATH_CALUDE_choose_four_different_suits_standard_deck_l3601_360179

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (suits : Nat)
  (cards_per_suit : Nat)
  (h1 : cards = suits * cards_per_suit)

/-- The number of ways to choose 4 cards of different suits from a standard deck -/
def choose_four_different_suits (d : Deck) : Nat :=
  d.cards_per_suit ^ d.suits

/-- Theorem stating the number of ways to choose 4 cards of different suits from a standard deck -/
theorem choose_four_different_suits_standard_deck :
  ∃ (d : Deck), d.cards = 52 ∧ d.suits = 4 ∧ d.cards_per_suit = 13 ∧ choose_four_different_suits d = 28561 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_different_suits_standard_deck_l3601_360179


namespace NUMINAMATH_CALUDE_birds_on_branch_l3601_360131

theorem birds_on_branch (initial_parrots : ℕ) (remaining_parrots : ℕ) (remaining_crows : ℕ) :
  initial_parrots = 7 →
  remaining_parrots = 2 →
  remaining_crows = 1 →
  ∃ (initial_crows : ℕ) (flew_away : ℕ),
    flew_away = initial_parrots - remaining_parrots ∧
    flew_away = initial_crows - remaining_crows ∧
    initial_parrots + initial_crows = 13 :=
by sorry

end NUMINAMATH_CALUDE_birds_on_branch_l3601_360131


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l3601_360189

theorem cubic_equation_solution (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hmn : m ≠ n) :
  ∃! (a b c : ℝ), ∀ x : ℝ, 
    (x + m)^3 - (x + n)^3 = (m + n)^3 ∧ x = a * m + b * n + c :=
by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l3601_360189


namespace NUMINAMATH_CALUDE_product_congruence_l3601_360185

theorem product_congruence : 56 * 89 * 94 ≡ 21 [ZMOD 25] := by
  sorry

end NUMINAMATH_CALUDE_product_congruence_l3601_360185


namespace NUMINAMATH_CALUDE_units_digit_sum_in_base_7_l3601_360188

/-- The base of the number system we're working in -/
def base : ℕ := 7

/-- Function to get the units digit of a number in the given base -/
def unitsDigit (n : ℕ) : ℕ := n % base

/-- First number in the sum -/
def num1 : ℕ := 52

/-- Second number in the sum -/
def num2 : ℕ := 62

/-- Theorem stating that the units digit of the sum of num1 and num2 in base 7 is 4 -/
theorem units_digit_sum_in_base_7 : 
  unitsDigit (num1 + num2) = 4 := by sorry

end NUMINAMATH_CALUDE_units_digit_sum_in_base_7_l3601_360188


namespace NUMINAMATH_CALUDE_four_spheres_cover_point_source_l3601_360146

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a sphere in 3D space
structure Sphere where
  center : Point3D
  radius : ℝ

-- Define a ray emanating from a point
structure Ray where
  origin : Point3D
  direction : Point3D

-- Function to check if a ray intersects a sphere
def rayIntersectsSphere (r : Ray) (s : Sphere) : Prop := sorry

-- Theorem statement
theorem four_spheres_cover_point_source (source : Point3D) :
  ∃ (s1 s2 s3 s4 : Sphere),
    ∀ (r : Ray),
      r.origin = source →
      rayIntersectsSphere r s1 ∨
      rayIntersectsSphere r s2 ∨
      rayIntersectsSphere r s3 ∨
      rayIntersectsSphere r s4 := by
  sorry

end NUMINAMATH_CALUDE_four_spheres_cover_point_source_l3601_360146


namespace NUMINAMATH_CALUDE_b_value_proof_l3601_360193

theorem b_value_proof (b : ℚ) (h : b + b/4 = 5/2) : b = 2 := by
  sorry

end NUMINAMATH_CALUDE_b_value_proof_l3601_360193


namespace NUMINAMATH_CALUDE_transform_graph_point_l3601_360145

/-- Given a function g : ℝ → ℝ such that g(8) = 5, prove that (8/3, 14/9) is on the graph of
    3y = g(3x)/3 + 3 and the sum of its coordinates is 38/9 -/
theorem transform_graph_point (g : ℝ → ℝ) (h : g 8 = 5) :
  let f : ℝ → ℝ := λ x => (g (3 * x) / 3 + 3) / 3
  f (8/3) = 14/9 ∧ 8/3 + 14/9 = 38/9 := by
  sorry

end NUMINAMATH_CALUDE_transform_graph_point_l3601_360145


namespace NUMINAMATH_CALUDE_muffin_count_l3601_360187

/-- Given a number of doughnuts and a ratio of doughnuts to muffins, 
    calculate the number of muffins -/
def calculate_muffins (num_doughnuts : ℕ) (doughnut_ratio : ℕ) (muffin_ratio : ℕ) : ℕ :=
  (num_doughnuts / doughnut_ratio) * muffin_ratio

/-- Theorem: Given 50 doughnuts and a ratio of 5 doughnuts to 1 muffin, 
    the number of muffins is 10 -/
theorem muffin_count : calculate_muffins 50 5 1 = 10 := by
  sorry

end NUMINAMATH_CALUDE_muffin_count_l3601_360187


namespace NUMINAMATH_CALUDE_factory_production_quota_l3601_360176

theorem factory_production_quota (x : ℕ) : 
  ((x - 3) * 31 + 60 = (x + 3) * 25 - 60) → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_factory_production_quota_l3601_360176


namespace NUMINAMATH_CALUDE_sin_cos_equivalence_l3601_360161

/-- The function f(x) = sin(2x) + √3 * cos(2x) is equivalent to 2 * sin(2(x + π/6)) for all real x -/
theorem sin_cos_equivalence (x : ℝ) : 
  Real.sin (2 * x) + Real.sqrt 3 * Real.cos (2 * x) = 2 * Real.sin (2 * (x + Real.pi / 6)) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_equivalence_l3601_360161


namespace NUMINAMATH_CALUDE_melted_ice_cream_height_l3601_360104

/-- The height of a cylindrical region formed by a melted spherical ice cream scoop -/
theorem melted_ice_cream_height (r_sphere r_cylinder : ℝ) (h_sphere : r_sphere = 3)
    (h_cylinder : r_cylinder = 9) : 
    (4 / 3) * Real.pi * r_sphere^3 = Real.pi * r_cylinder^2 * (4 / 9) := by
  sorry

#check melted_ice_cream_height

end NUMINAMATH_CALUDE_melted_ice_cream_height_l3601_360104


namespace NUMINAMATH_CALUDE_tulip_probability_l3601_360180

structure FlowerSet where
  roses : ℕ
  tulips : ℕ
  daisies : ℕ
  lilies : ℕ

def total_flowers (fs : FlowerSet) : ℕ :=
  fs.roses + fs.tulips + fs.daisies + fs.lilies

def probability_of_tulip (fs : FlowerSet) : ℚ :=
  fs.tulips / (total_flowers fs)

theorem tulip_probability (fs : FlowerSet) (h : fs = ⟨3, 2, 4, 6⟩) :
  probability_of_tulip fs = 2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_tulip_probability_l3601_360180


namespace NUMINAMATH_CALUDE_james_passenger_count_l3601_360158

/-- Calculates the total number of passengers James has seen given the vehicle counts and passenger capacities. -/
def total_passengers (total_vehicles : ℕ) (trucks : ℕ) (buses : ℕ) (cars : ℕ) 
  (truck_capacity : ℕ) (bus_capacity : ℕ) (taxi_capacity : ℕ) (motorbike_capacity : ℕ) (car_capacity : ℕ) : ℕ :=
  let taxis := 2 * buses
  let motorbikes := total_vehicles - trucks - buses - taxis - cars
  trucks * truck_capacity + buses * bus_capacity + taxis * taxi_capacity + motorbikes * motorbike_capacity + cars * car_capacity

theorem james_passenger_count :
  total_passengers 52 12 2 30 2 15 2 1 3 = 156 := by
  sorry

end NUMINAMATH_CALUDE_james_passenger_count_l3601_360158


namespace NUMINAMATH_CALUDE_coin_arrangement_count_l3601_360106

/-- Represents the number of ways to arrange 5 gold and 5 silver coins -/
def colorArrangements : ℕ := Nat.choose 10 5

/-- Represents the number of valid face orientations for 10 coins -/
def validOrientations : ℕ := 144

/-- The total number of distinguishable arrangements -/
def totalArrangements : ℕ := colorArrangements * validOrientations

/-- Theorem stating the number of distinguishable arrangements -/
theorem coin_arrangement_count :
  totalArrangements = 36288 :=
sorry

end NUMINAMATH_CALUDE_coin_arrangement_count_l3601_360106


namespace NUMINAMATH_CALUDE_peach_difference_l3601_360125

theorem peach_difference (jill_peaches steven_peaches jake_peaches : ℕ) : 
  jill_peaches = 12 →
  steven_peaches = jill_peaches + 15 →
  jake_peaches + 1 = jill_peaches →
  steven_peaches - jake_peaches = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_peach_difference_l3601_360125


namespace NUMINAMATH_CALUDE_perimeter_plus_area_sum_l3601_360169

/-- A parallelogram with integer coordinates -/
structure Parallelogram where
  v1 : ℤ × ℤ
  v2 : ℤ × ℤ
  v3 : ℤ × ℤ
  v4 : ℤ × ℤ

/-- The specific parallelogram from the problem -/
def specificParallelogram : Parallelogram :=
  { v1 := (1, 3)
    v2 := (6, 8)
    v3 := (13, 8)
    v4 := (8, 3) }

/-- Calculate the perimeter of the parallelogram -/
def perimeter (p : Parallelogram) : ℝ :=
  sorry

/-- Calculate the area of the parallelogram -/
def area (p : Parallelogram) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem perimeter_plus_area_sum (p : Parallelogram) :
  p = specificParallelogram →
  perimeter p + area p = 10 * Real.sqrt 2 + 49 :=
by
  sorry

end NUMINAMATH_CALUDE_perimeter_plus_area_sum_l3601_360169
