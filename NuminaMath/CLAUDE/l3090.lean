import Mathlib

namespace NUMINAMATH_CALUDE_inequality_proof_l3090_309074

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (b * c) + b / (a * c) + c / (a * b) ≥ 2 / a + 2 / b - 2 / c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3090_309074


namespace NUMINAMATH_CALUDE_genetically_modified_microorganisms_percentage_l3090_309005

/-- Represents the budget allocation for Megatech Corporation --/
structure BudgetAllocation where
  microphotonics : ℝ
  homeElectronics : ℝ
  foodAdditives : ℝ
  industrialLubricants : ℝ
  basicAstrophysicsDegrees : ℝ

/-- Theorem stating the percentage allocated to genetically modified microorganisms --/
theorem genetically_modified_microorganisms_percentage 
  (budget : BudgetAllocation)
  (h1 : budget.microphotonics = 13)
  (h2 : budget.homeElectronics = 24)
  (h3 : budget.foodAdditives = 15)
  (h4 : budget.industrialLubricants = 8)
  (h5 : budget.basicAstrophysicsDegrees = 39.6) :
  100 - (budget.microphotonics + budget.homeElectronics + budget.foodAdditives + 
         budget.industrialLubricants + (budget.basicAstrophysicsDegrees / 360 * 100)) = 29 := by
  sorry

end NUMINAMATH_CALUDE_genetically_modified_microorganisms_percentage_l3090_309005


namespace NUMINAMATH_CALUDE_total_jeans_purchased_l3090_309034

-- Define the regular prices and quantities
def fox_price : ℝ := 15
def pony_price : ℝ := 18
def fox_quantity : ℕ := 3
def pony_quantity : ℕ := 2

-- Define the total savings and discount rates
def total_savings : ℝ := 8.55
def total_discount_rate : ℝ := 0.22
def pony_discount_rate : ℝ := 0.15

-- Define the theorem
theorem total_jeans_purchased :
  fox_quantity + pony_quantity = 5 := by sorry

end NUMINAMATH_CALUDE_total_jeans_purchased_l3090_309034


namespace NUMINAMATH_CALUDE_equilateral_triangle_ratio_l3090_309025

/-- Given two equilateral triangles with side lengths A and a, and altitudes h_A and h_a respectively,
    if h_A = 2h_a, then the ratio of their perimeters is equal to the ratio of their altitudes. -/
theorem equilateral_triangle_ratio (A a h_A h_a : ℝ) 
  (h_positive : h_a > 0)
  (h_eq : h_A = 2 * h_a) :
  3 * A / (3 * a) = h_A / h_a := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_ratio_l3090_309025


namespace NUMINAMATH_CALUDE_negation_of_positive_product_l3090_309071

theorem negation_of_positive_product (x y : ℝ) :
  ¬(x > 0 ∧ y > 0 → x * y > 0) ↔ (x ≤ 0 ∨ y ≤ 0 → x * y ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_positive_product_l3090_309071


namespace NUMINAMATH_CALUDE_combinatorial_identity_l3090_309080

theorem combinatorial_identity (n k : ℕ) (h : k ≤ n) :
  Nat.choose (n + 1) k = Nat.choose n k + Nat.choose n (k - 1) := by
  sorry

end NUMINAMATH_CALUDE_combinatorial_identity_l3090_309080


namespace NUMINAMATH_CALUDE_symmetric_points_sqrt_l3090_309029

/-- Given that point P(3, -1) is symmetric to point Q(a+b, 1-b) about the y-axis,
    prove that the square root of -ab equals √10. -/
theorem symmetric_points_sqrt (a b : ℝ) : 
  (3 = -(a + b) ∧ -1 = 1 - b) → Real.sqrt (-a * b) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sqrt_l3090_309029


namespace NUMINAMATH_CALUDE_newspaper_conference_max_both_l3090_309068

theorem newspaper_conference_max_both (total : ℕ) (writers : ℕ) (editors : ℕ) (neither : ℕ) (both : ℕ) :
  total = 90 →
  writers = 45 →
  editors > 38 →
  neither = 2 * both →
  total = writers + editors + neither - both →
  both ≤ 4 ∧ (∃ (e : ℕ), editors = 38 + e ∧ both = 4) :=
by sorry

end NUMINAMATH_CALUDE_newspaper_conference_max_both_l3090_309068


namespace NUMINAMATH_CALUDE_cows_and_sheep_bushels_l3090_309023

/-- Represents the farm animals and their food consumption --/
structure Farm where
  cows : ℕ
  sheep : ℕ
  chickens : ℕ
  chicken_bushels : ℕ
  total_bushels : ℕ

/-- Calculates the bushels eaten by cows and sheep --/
def bushels_for_cows_and_sheep (farm : Farm) : ℕ :=
  farm.total_bushels - (farm.chickens * farm.chicken_bushels)

/-- Theorem stating that the bushels eaten by cows and sheep is 14 --/
theorem cows_and_sheep_bushels (farm : Farm) 
  (h1 : farm.cows = 4)
  (h2 : farm.sheep = 3)
  (h3 : farm.chickens = 7)
  (h4 : farm.chicken_bushels = 3)
  (h5 : farm.total_bushels = 35) :
  bushels_for_cows_and_sheep farm = 14 := by
  sorry

end NUMINAMATH_CALUDE_cows_and_sheep_bushels_l3090_309023


namespace NUMINAMATH_CALUDE_g_satisfies_conditions_l3090_309075

/-- A monic polynomial of degree 3 satisfying specific conditions -/
def g (x : ℝ) : ℝ := x^3 + 4*x^2 + 3*x + 6

/-- Theorem stating that g satisfies the given conditions -/
theorem g_satisfies_conditions :
  (∀ x, g x = x^3 + 4*x^2 + 3*x + 6) ∧
  g 0 = 6 ∧
  g 1 = 14 ∧
  g (-1) = 6 :=
by sorry

end NUMINAMATH_CALUDE_g_satisfies_conditions_l3090_309075


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l3090_309092

theorem square_area_from_diagonal (d : ℝ) (h : d = 7) : 
  (d^2 / 2) = 24.5 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l3090_309092


namespace NUMINAMATH_CALUDE_lionel_distance_walked_l3090_309090

/-- The distance between Lionel's and Walt's houses -/
def total_distance : ℝ := 48

/-- Lionel's walking speed in miles per hour -/
def lionel_speed : ℝ := 2

/-- Walt's running speed in miles per hour -/
def walt_speed : ℝ := 6

/-- The time Walt waits before starting to run, in hours -/
def walt_wait_time : ℝ := 2

/-- The theorem stating that Lionel walked 15 miles when he met Walt -/
theorem lionel_distance_walked : ℝ := by
  sorry

end NUMINAMATH_CALUDE_lionel_distance_walked_l3090_309090


namespace NUMINAMATH_CALUDE_quadratic_polynomial_unique_l3090_309044

theorem quadratic_polynomial_unique (q : ℝ → ℝ) : 
  (∀ x, q x = 2 * x^2 - 6 * x - 36) →
  q (-3) = 0 ∧ q 6 = 0 ∧ q 2 = -40 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_unique_l3090_309044


namespace NUMINAMATH_CALUDE_eight_faucets_fill_time_correct_l3090_309038

/-- The time (in seconds) it takes for eight faucets to fill a 50-gallon tank,
    given that four faucets fill a 200-gallon tank in 8 minutes and all faucets
    dispense water at the same rate. -/
def eight_faucets_fill_time : ℕ := by sorry

/-- Four faucets fill a 200-gallon tank in 8 minutes. -/
def four_faucets_fill_time : ℕ := 8 * 60  -- in seconds

/-- The volume of the tank filled by four faucets. -/
def four_faucets_volume : ℕ := 200  -- in gallons

/-- The volume of the tank to be filled by eight faucets. -/
def eight_faucets_volume : ℕ := 50  -- in gallons

/-- All faucets dispense water at the same rate. -/
axiom faucets_equal_rate : True

theorem eight_faucets_fill_time_correct :
  eight_faucets_fill_time = 60 := by sorry

end NUMINAMATH_CALUDE_eight_faucets_fill_time_correct_l3090_309038


namespace NUMINAMATH_CALUDE_simplify_fraction_with_sqrt_two_l3090_309053

theorem simplify_fraction_with_sqrt_two : 
  (1 / (1 + Real.sqrt 2)) * (1 / (1 - Real.sqrt 2)) = -1 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_with_sqrt_two_l3090_309053


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l3090_309082

theorem cube_root_equation_solution :
  ∃ y : ℝ, (5 + 2 / y)^(1/3) = -3 ↔ y = -1/16 := by sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l3090_309082


namespace NUMINAMATH_CALUDE_profit_calculation_l3090_309024

def profit_division (total_profit : ℚ) : 
  (ℚ × ℚ × ℚ × ℚ) := 
  let equal_share := total_profit / 12
  let investment_total := 800 + 200 + 600 + 400
  let mary_inv_share := (800 / investment_total) * (total_profit / 3)
  let mike_inv_share := (200 / investment_total) * (total_profit / 3)
  let anna_inv_share := (600 / investment_total) * (total_profit / 3)
  let ben_inv_share := (400 / investment_total) * (total_profit / 3)
  let ratio_total := 2 + 1 + 3 + 4
  let mary_ratio_share := (2 / ratio_total) * (total_profit / 3)
  let mike_ratio_share := (1 / ratio_total) * (total_profit / 3)
  let anna_ratio_share := (3 / ratio_total) * (total_profit / 3)
  let ben_ratio_share := (4 / ratio_total) * (total_profit / 3)
  (
    equal_share + mary_inv_share + mary_ratio_share,
    equal_share + mike_inv_share + mike_ratio_share,
    equal_share + anna_inv_share + anna_ratio_share,
    equal_share + ben_inv_share + ben_ratio_share
  )

theorem profit_calculation :
  ∃ (total_profit : ℚ), 
    let (mary_share, mike_share, anna_share, ben_share) := profit_division total_profit
    mary_share - mike_share = 900 ∧
    anna_share - ben_share = 600 ∧
    total_profit = 6000 := by
  sorry

end NUMINAMATH_CALUDE_profit_calculation_l3090_309024


namespace NUMINAMATH_CALUDE_expand_expression_l3090_309097

theorem expand_expression (x : ℝ) : (7*x - 3) * 5*x^2 = 35*x^3 - 15*x^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3090_309097


namespace NUMINAMATH_CALUDE_infinite_series_convergence_l3090_309093

theorem infinite_series_convergence : 
  let f (n : ℕ) := (n^3 + 2*n^2 + 5*n + 2) / (3^n * (n^3 + 3))
  ∑' (n : ℕ), f n = 1/2 := by sorry

end NUMINAMATH_CALUDE_infinite_series_convergence_l3090_309093


namespace NUMINAMATH_CALUDE_time_to_eat_half_l3090_309065

/-- Represents the eating rate of a bird in terms of fraction of nuts eaten per hour -/
structure BirdRate where
  fraction : ℚ
  hours : ℚ

/-- Calculates the rate at which a bird eats nuts per hour -/
def eatRate (br : BirdRate) : ℚ :=
  br.fraction / br.hours

/-- Represents the rates of the three birds -/
structure BirdRates where
  crow : BirdRate
  sparrow : BirdRate
  parrot : BirdRate

/-- Calculates the combined eating rate of all three birds -/
def combinedRate (rates : BirdRates) : ℚ :=
  eatRate rates.crow + eatRate rates.sparrow + eatRate rates.parrot

/-- The main theorem stating the time taken to eat half the nuts -/
theorem time_to_eat_half (rates : BirdRates) 
  (h_crow : rates.crow = ⟨1/5, 4⟩) 
  (h_sparrow : rates.sparrow = ⟨1/3, 6⟩)
  (h_parrot : rates.parrot = ⟨1/4, 8⟩) : 
  (1/2) / combinedRate rates = 2880 / 788 := by
  sorry

end NUMINAMATH_CALUDE_time_to_eat_half_l3090_309065


namespace NUMINAMATH_CALUDE_probability_x_gt_5y_l3090_309050

/-- The probability of selecting a point (x,y) from a rectangle with vertices
    (0,0), (2020,0), (2020,2021), and (0,2021) such that x > 5y is 101/1011. -/
theorem probability_x_gt_5y : 
  let rectangle_area := 2020 * 2021
  let triangle_area := (1 / 2) * 2020 * 404
  triangle_area / rectangle_area = 101 / 1011 := by
sorry

end NUMINAMATH_CALUDE_probability_x_gt_5y_l3090_309050


namespace NUMINAMATH_CALUDE_reflection_of_A_l3090_309039

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

theorem reflection_of_A : reflect_x (-4, 3) = (-4, -3) := by
  sorry

end NUMINAMATH_CALUDE_reflection_of_A_l3090_309039


namespace NUMINAMATH_CALUDE_ball_drawing_game_l3090_309037

theorem ball_drawing_game (x : ℕ) : 
  (2 : ℕ) > 0 ∧ x > 0 →
  (4 * x : ℚ) / ((x + 2) * (x + 1)) ≥ 1/5 ∧
  (4 * x : ℚ) / ((x + 2) * (x + 1)) ≤ 33/100 →
  9 ≤ x ∧ x ≤ 16 :=
by sorry

end NUMINAMATH_CALUDE_ball_drawing_game_l3090_309037


namespace NUMINAMATH_CALUDE_notebook_marker_cost_l3090_309016

theorem notebook_marker_cost (notebook_cost marker_cost : ℝ) 
  (h1 : 3 * notebook_cost + 2 * marker_cost = 7.20)
  (h2 : 2 * notebook_cost + 3 * marker_cost = 6.90) :
  notebook_cost + marker_cost = 2.82 := by
  sorry

end NUMINAMATH_CALUDE_notebook_marker_cost_l3090_309016


namespace NUMINAMATH_CALUDE_bc_length_l3090_309052

-- Define the points
variable (A B C D : ℝ × ℝ)

-- Define the conditions
def is_right_triangle (X Y Z : ℝ × ℝ) : Prop := sorry

-- Define the lengths
def length (X Y : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem bc_length 
  (h1 : is_right_triangle A B C)
  (h2 : is_right_triangle A B D)
  (h3 : length A D = 50)
  (h4 : length C D = 25)
  (h5 : length A C = 20)
  (h6 : length A B = 15) :
  length B C = 25 :=
sorry

end NUMINAMATH_CALUDE_bc_length_l3090_309052


namespace NUMINAMATH_CALUDE_decimal_expansion_four_seventeenths_l3090_309076

/-- The decimal expansion of 4/17 has a repeating block of 235. -/
theorem decimal_expansion_four_seventeenths :
  ∃ (a b : ℕ), (4 : ℚ) / 17 = (a : ℚ) / 999 + (b : ℚ) / (999 * 1000) ∧ a = 235 ∧ b < 999 := by
  sorry

end NUMINAMATH_CALUDE_decimal_expansion_four_seventeenths_l3090_309076


namespace NUMINAMATH_CALUDE_jack_initial_yen_l3090_309045

/-- Represents Jack's currency holdings and exchange rates --/
structure CurrencyHoldings where
  pounds : ℕ
  euros : ℕ
  total_yen : ℕ
  pounds_per_euro : ℕ
  yen_per_pound : ℕ

/-- Calculates Jack's initial yen amount --/
def initial_yen (h : CurrencyHoldings) : ℕ :=
  h.total_yen - (h.pounds * h.yen_per_pound + h.euros * h.pounds_per_euro * h.yen_per_pound)

/-- Theorem stating that Jack's initial yen amount is 3000 --/
theorem jack_initial_yen :
  let h : CurrencyHoldings := {
    pounds := 42,
    euros := 11,
    total_yen := 9400,
    pounds_per_euro := 2,
    yen_per_pound := 100
  }
  initial_yen h = 3000 := by sorry

end NUMINAMATH_CALUDE_jack_initial_yen_l3090_309045


namespace NUMINAMATH_CALUDE_line_plane_perpendicular_parallel_l3090_309099

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicular_parallel 
  (b c : Line) (α β : Plane) :
  perpendicular c β → parallel c α → plane_perpendicular α β :=
by sorry

end NUMINAMATH_CALUDE_line_plane_perpendicular_parallel_l3090_309099


namespace NUMINAMATH_CALUDE_logarithm_equation_solution_l3090_309041

theorem logarithm_equation_solution (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (∀ x : ℝ, x > 0 → 5 * (Real.log x / Real.log a)^2 + 2 * (Real.log x / Real.log b)^2 = 
    (10 * (Real.log x)^2) / (Real.log a * Real.log b) + (Real.log x)^2) →
  b = a^(2 / (5 + Real.sqrt 17)) ∨ b = a^(2 / (5 - Real.sqrt 17)) :=
by sorry

end NUMINAMATH_CALUDE_logarithm_equation_solution_l3090_309041


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3090_309043

theorem polynomial_factorization (x : ℝ) : 
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = 
  (x^2 + 6*x + 12) * (x^2 + 6*x + 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3090_309043


namespace NUMINAMATH_CALUDE_seulgi_winning_score_l3090_309036

/-- Represents a player's scores in a two-round darts game -/
structure PlayerScores where
  round1 : ℕ
  round2 : ℕ

/-- Calculates the total score for a player -/
def totalScore (scores : PlayerScores) : ℕ :=
  scores.round1 + scores.round2

/-- Theorem: Seulgi needs at least 25 points in the second round to win -/
theorem seulgi_winning_score 
  (hohyeon : PlayerScores) 
  (hyunjeong : PlayerScores)
  (seulgi_round1 : ℕ) :
  hohyeon.round1 = 23 →
  hohyeon.round2 = 28 →
  hyunjeong.round1 = 32 →
  hyunjeong.round2 = 17 →
  seulgi_round1 = 27 →
  ∀ seulgi_round2 : ℕ,
    (totalScore ⟨seulgi_round1, seulgi_round2⟩ > totalScore hohyeon ∧
     totalScore ⟨seulgi_round1, seulgi_round2⟩ > totalScore hyunjeong) →
    seulgi_round2 ≥ 25 :=
by
  sorry


end NUMINAMATH_CALUDE_seulgi_winning_score_l3090_309036


namespace NUMINAMATH_CALUDE_digit_1234_is_4_l3090_309067

/-- The number of digits in the representation of an integer -/
def num_digits (n : ℕ) : ℕ := sorry

/-- The nth digit in the decimal expansion of x -/
def nth_digit (x : ℝ) (n : ℕ) : ℕ := sorry

/-- The number formed by concatenating the decimal representations of integers from 1 to n -/
def concat_integers (n : ℕ) : ℝ := sorry

theorem digit_1234_is_4 :
  let x := concat_integers 500
  nth_digit x 1234 = 4 := by sorry

end NUMINAMATH_CALUDE_digit_1234_is_4_l3090_309067


namespace NUMINAMATH_CALUDE_square_less_than_four_times_l3090_309002

theorem square_less_than_four_times : ∀ n : ℤ, n^2 < 4*n ↔ n = 1 ∨ n = 2 ∨ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_less_than_four_times_l3090_309002


namespace NUMINAMATH_CALUDE_tiling_problem_l3090_309018

/-- Number of ways to tile a 3 × n rectangle -/
def tiling_ways (n : ℕ) : ℚ :=
  (2^(n+2) + (-1)^(n+1)) / 3

/-- Proof of the tiling problem -/
theorem tiling_problem (n : ℕ) (h : n > 3) :
  tiling_ways n = (2^(n+2) + (-1)^(n+1)) / 3 :=
by sorry

end NUMINAMATH_CALUDE_tiling_problem_l3090_309018


namespace NUMINAMATH_CALUDE_leftover_value_calculation_l3090_309085

/-- Calculates the value of leftover coins after making complete rolls --/
def leftover_value (quarters_per_roll dimes_per_roll toledo_quarters toledo_dimes brian_quarters brian_dimes : ℕ) : ℚ :=
  let total_quarters := toledo_quarters + brian_quarters
  let total_dimes := toledo_dimes + brian_dimes
  let leftover_quarters := total_quarters % quarters_per_roll
  let leftover_dimes := total_dimes % dimes_per_roll
  (leftover_quarters : ℚ) * (1 / 4) + (leftover_dimes : ℚ) * (1 / 10)

theorem leftover_value_calculation :
  leftover_value 30 50 95 172 137 290 = 17/10 := by
  sorry

end NUMINAMATH_CALUDE_leftover_value_calculation_l3090_309085


namespace NUMINAMATH_CALUDE_smallest_next_divisor_after_391_l3090_309063

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem smallest_next_divisor_after_391 (m : ℕ) 
  (h1 : is_even m) 
  (h2 : is_four_digit m) 
  (h3 : m % 391 = 0) : 
  ∃ d : ℕ, d > 391 ∧ m % d = 0 ∧ (∀ k : ℕ, 391 < k ∧ k < d → m % k ≠ 0) → d = 782 :=
sorry

end NUMINAMATH_CALUDE_smallest_next_divisor_after_391_l3090_309063


namespace NUMINAMATH_CALUDE_largest_radius_is_61_l3090_309009

/-- A circle containing specific points and the unit circle -/
structure SpecialCircle where
  center : ℝ × ℝ
  radius : ℝ
  contains_points : center.1^2 + 11^2 = radius^2
  contains_unit_circle : ∀ (x y : ℝ), x^2 + y^2 < 1 → 
    (x - center.1)^2 + (y - center.2)^2 < radius^2

/-- The largest possible radius of a SpecialCircle is 61 -/
theorem largest_radius_is_61 : 
  (∃ (c : SpecialCircle), true) → 
  (∀ (c : SpecialCircle), c.radius ≤ 61) ∧ 
  (∃ (c : SpecialCircle), c.radius = 61) :=
sorry

end NUMINAMATH_CALUDE_largest_radius_is_61_l3090_309009


namespace NUMINAMATH_CALUDE_equation_solution_l3090_309028

theorem equation_solution : 
  ∃ x : ℚ, x + 5/6 = 7/18 + 1/2 ∧ x = -7/18 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l3090_309028


namespace NUMINAMATH_CALUDE_convex_polygon_equal_area_division_l3090_309014

-- Define a convex polygon
structure ConvexPolygon where
  -- Add necessary properties to define a convex polygon
  is_convex : Bool

-- Define a line in 2D space
structure Line where
  -- Add necessary properties to define a line
  slope : ℝ
  intercept : ℝ

-- Define the concept of perpendicular lines
def perpendicular (l1 l2 : Line) : Prop :=
  -- Add condition for perpendicularity
  sorry

-- Define the concept of a region in the polygon
structure Region where
  -- Add necessary properties to define a region
  area : ℝ

-- Define the division of a polygon by two lines
def divide_polygon (p : ConvexPolygon) (l1 l2 : Line) : List Region :=
  -- Function to divide the polygon into regions
  sorry

-- Theorem statement
theorem convex_polygon_equal_area_division (p : ConvexPolygon) :
  ∃ (l1 l2 : Line), 
    perpendicular l1 l2 ∧ 
    let regions := divide_polygon p l1 l2
    regions.length = 4 ∧ 
    ∀ (r1 r2 : Region), r1 ∈ regions → r2 ∈ regions → r1.area = r2.area :=
by
  sorry

end NUMINAMATH_CALUDE_convex_polygon_equal_area_division_l3090_309014


namespace NUMINAMATH_CALUDE_faster_train_speed_l3090_309055

/-- The speed of the faster train when two trains cross each other --/
theorem faster_train_speed
  (train_length : ℝ)
  (crossing_time : ℝ)
  (h1 : train_length = 100)
  (h2 : crossing_time = 8)
  (h3 : crossing_time > 0) :
  ∃ (v : ℝ), v > 0 ∧ 2 * v * crossing_time = 2 * train_length ∧ v = 25 / 3 :=
by sorry

end NUMINAMATH_CALUDE_faster_train_speed_l3090_309055


namespace NUMINAMATH_CALUDE_mary_has_more_than_marco_l3090_309040

/-- Proves that Mary has $10 more than Marco after transactions --/
theorem mary_has_more_than_marco (marco_initial : ℕ) (mary_initial : ℕ) 
  (marco_gives : ℕ) (mary_spends : ℕ) : ℕ :=
by
  -- Define initial amounts
  have h1 : marco_initial = 24 := by sorry
  have h2 : mary_initial = 15 := by sorry
  
  -- Define amount Marco gives to Mary
  have h3 : marco_gives = marco_initial / 2 := by sorry
  
  -- Define amount Mary spends
  have h4 : mary_spends = 5 := by sorry
  
  -- Calculate final amounts
  let marco_final := marco_initial - marco_gives
  let mary_final := mary_initial + marco_gives - mary_spends
  
  -- Prove Mary has $10 more than Marco
  have h5 : mary_final - marco_final = 10 := by sorry
  
  exact 10

end NUMINAMATH_CALUDE_mary_has_more_than_marco_l3090_309040


namespace NUMINAMATH_CALUDE_equation_root_l3090_309033

theorem equation_root (a b c d x : ℝ) 
  (h1 : a + d = 2015)
  (h2 : b + c = 2015)
  (h3 : a ≠ c) :
  (x - a) * (x - b) = (x - c) * (x - d) ↔ x = 1007.5 := by
sorry

end NUMINAMATH_CALUDE_equation_root_l3090_309033


namespace NUMINAMATH_CALUDE_solution_form_l3090_309084

-- Define the system of equations
def equation1 (x y z : ℝ) : Prop :=
  x / y + y / z + z / x = x / z + z / y + y / x

def equation2 (x y z : ℝ) : Prop :=
  x^2 + y^2 + z^2 = x*y + y*z + z*x + 4

-- Theorem statement
theorem solution_form (x y z : ℝ) :
  equation1 x y z ∧ equation2 x y z →
  (∃ t : ℝ, (x = t ∧ y = t - 2 ∧ z = t - 2) ∨ (x = t ∧ y = t + 2 ∧ z = t + 2)) :=
by sorry

end NUMINAMATH_CALUDE_solution_form_l3090_309084


namespace NUMINAMATH_CALUDE_white_marbles_in_bag_a_l3090_309086

/-- Represents the number of marbles of each color in Bag A -/
structure BagA where
  red : ℕ
  white : ℕ
  blue : ℕ

/-- Represents the ratios of marbles in Bag A -/
structure BagARatios where
  red_to_white : ℚ
  white_to_blue : ℚ

/-- Theorem stating that if Bag A contains 5 red marbles, it must contain 15 white marbles -/
theorem white_marbles_in_bag_a 
  (bag : BagA) 
  (ratios : BagARatios) 
  (h1 : ratios.red_to_white = 1 / 3) 
  (h2 : ratios.white_to_blue = 2 / 3) 
  (h3 : bag.red = 5) : 
  bag.white = 15 := by
  sorry

#check white_marbles_in_bag_a

end NUMINAMATH_CALUDE_white_marbles_in_bag_a_l3090_309086


namespace NUMINAMATH_CALUDE_exists_valid_grid_l3090_309057

def is_valid_grid (grid : Matrix (Fin 3) (Fin 3) ℕ) : Prop :=
  (∀ i j, grid i j ≤ 25) ∧
  (∀ i j, grid i j > 0) ∧
  (∀ i₁ j₁ i₂ j₂, i₁ ≠ i₂ ∨ j₁ ≠ j₂ → grid i₁ j₁ ≠ grid i₂ j₂) ∧
  (∀ i j, i < 2 → (grid i j ∣ grid (i+1) j) ∨ (grid (i+1) j ∣ grid i j)) ∧
  (∀ i j, j < 2 → (grid i j ∣ grid i (j+1)) ∨ (grid i (j+1) ∣ grid i j))

theorem exists_valid_grid : ∃ (grid : Matrix (Fin 3) (Fin 3) ℕ), is_valid_grid grid := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_grid_l3090_309057


namespace NUMINAMATH_CALUDE_largest_product_of_three_primes_digit_sum_l3090_309095

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m > 1 → m < p → ¬(p % m = 0)

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem largest_product_of_three_primes_digit_sum :
  ∃ (n d e : ℕ),
    d < 10 ∧
    e < 10 ∧
    is_prime d ∧
    is_prime e ∧
    is_prime (10 * d + e) ∧
    n = d * e * (10 * d + e) ∧
    (∀ (m : ℕ), m = d' * e' * (10 * d' + e') → 
      is_prime d' ∧ 
      is_prime e' ∧ 
      is_prime (10 * d' + e') ∧ 
      d' < 10 ∧ 
      e' < 10 → 
      m ≤ n) ∧
    sum_of_digits n = 12 :=
sorry

end NUMINAMATH_CALUDE_largest_product_of_three_primes_digit_sum_l3090_309095


namespace NUMINAMATH_CALUDE_team_pizza_consumption_l3090_309069

theorem team_pizza_consumption (total_slices : ℕ) (slices_left : ℕ) : 
  total_slices = 32 → slices_left = 7 → total_slices - slices_left = 25 := by
  sorry

end NUMINAMATH_CALUDE_team_pizza_consumption_l3090_309069


namespace NUMINAMATH_CALUDE_bike_ride_distance_l3090_309087

/-- Calculates the total distance ridden given a constant riding rate and time, including breaks -/
def total_distance (rate : ℚ) (total_time : ℚ) (break_time : ℚ) (num_breaks : ℕ) : ℚ :=
  rate * (total_time - (break_time * num_breaks))

/-- The theorem to be proved -/
theorem bike_ride_distance :
  let rate : ℚ := 2 / 10  -- 2 miles per 10 minutes
  let total_time : ℚ := 40  -- 40 minutes total time
  let break_time : ℚ := 5  -- 5 minutes per break
  let num_breaks : ℕ := 2  -- 2 breaks
  total_distance rate total_time break_time num_breaks = 6 := by
  sorry


end NUMINAMATH_CALUDE_bike_ride_distance_l3090_309087


namespace NUMINAMATH_CALUDE_sum_of_imaginary_parts_zero_l3090_309006

theorem sum_of_imaginary_parts_zero (z : ℂ) : 
  (z^2 - 2*z = -1 + Complex.I) → 
  (∃ z₁ z₂ : ℂ, (z₁^2 - 2*z₁ = -1 + Complex.I) ∧ 
                (z₂^2 - 2*z₂ = -1 + Complex.I) ∧ 
                (z₁ ≠ z₂) ∧
                (Complex.im z₁ + Complex.im z₂ = 0)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_imaginary_parts_zero_l3090_309006


namespace NUMINAMATH_CALUDE_negative_quadratic_symmetry_implies_inequality_l3090_309096

/-- A quadratic function with a negative leading coefficient -/
structure NegativeQuadraticFunction where
  f : ℝ → ℝ
  is_quadratic : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c
  negative_leading_coeff : ∃ a b c : ℝ, (∀ x, f x = a * x^2 + b * x + c) ∧ a < 0

/-- The theorem statement -/
theorem negative_quadratic_symmetry_implies_inequality
  (f : NegativeQuadraticFunction)
  (h_symmetry : ∀ x : ℝ, f.f (2 - x) = f.f (2 + x)) :
  ∀ x : ℝ, -2 < x → x < 0 → f.f (1 + 2*x - x^2) < f.f (1 - 2*x^2) :=
sorry

end NUMINAMATH_CALUDE_negative_quadratic_symmetry_implies_inequality_l3090_309096


namespace NUMINAMATH_CALUDE_towel_area_decrease_l3090_309004

/-- Represents the properties of a fabric material -/
structure Material where
  cotton_percent : Real
  polyester_percent : Real
  cotton_length_shrinkage : Real
  cotton_breadth_shrinkage : Real
  polyester_length_shrinkage : Real
  polyester_breadth_shrinkage : Real

/-- Calculates the area decrease percentage of a fabric after shrinkage -/
def calculate_area_decrease (m : Material) : Real :=
  let effective_length_shrinkage := 
    m.cotton_length_shrinkage * m.cotton_percent + m.polyester_length_shrinkage * m.polyester_percent
  let effective_breadth_shrinkage := 
    m.cotton_breadth_shrinkage * m.cotton_percent + m.polyester_breadth_shrinkage * m.polyester_percent
  1 - (1 - effective_length_shrinkage) * (1 - effective_breadth_shrinkage)

/-- The towel material properties -/
def towel : Material := {
  cotton_percent := 0.60
  polyester_percent := 0.40
  cotton_length_shrinkage := 0.35
  cotton_breadth_shrinkage := 0.45
  polyester_length_shrinkage := 0.25
  polyester_breadth_shrinkage := 0.30
}

/-- Theorem: The area decrease of the towel after bleaching is approximately 57.91% -/
theorem towel_area_decrease : 
  ∃ ε > 0, |calculate_area_decrease towel - 0.5791| < ε :=
by sorry

end NUMINAMATH_CALUDE_towel_area_decrease_l3090_309004


namespace NUMINAMATH_CALUDE_roots_of_equation_l3090_309081

theorem roots_of_equation : ∀ x : ℝ, (x - 3)^2 = 25 ↔ x = 8 ∨ x = -2 := by sorry

end NUMINAMATH_CALUDE_roots_of_equation_l3090_309081


namespace NUMINAMATH_CALUDE_target_number_is_294_l3090_309089

/-- Represents the list of numbers starting with digit 2 in increasing order -/
def digit2List : List ℕ := sorry

/-- Returns the nth digit in the concatenated representation of digit2List -/
def nthDigit (n : ℕ) : ℕ := sorry

/-- The three-digit number formed by the 1498th, 1499th, and 1500th digits -/
def targetNumber : ℕ := 100 * (nthDigit 1498) + 10 * (nthDigit 1499) + (nthDigit 1500)

theorem target_number_is_294 : targetNumber = 294 := by sorry

end NUMINAMATH_CALUDE_target_number_is_294_l3090_309089


namespace NUMINAMATH_CALUDE_shaded_area_of_divided_triangle_l3090_309035

theorem shaded_area_of_divided_triangle (leg_length : ℝ) (total_divisions : ℕ) (shaded_divisions : ℕ) : 
  leg_length = 10 → 
  total_divisions = 20 → 
  shaded_divisions = 12 → 
  (1/2 * leg_length * leg_length * (shaded_divisions / total_divisions : ℝ)) = 30 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_of_divided_triangle_l3090_309035


namespace NUMINAMATH_CALUDE_quadratic_function_minimum_value_l3090_309021

/-- A quadratic function f(x) = ax² + bx + c with a ≠ 0 -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- The function f(x) defined by the quadratic function -/
def f (q : QuadraticFunction) (x : ℝ) : ℝ :=
  q.a * x^2 + q.b * x + q.c

/-- The derivative of f(x) -/
def f' (q : QuadraticFunction) (x : ℝ) : ℝ :=
  2 * q.a * x + q.b

theorem quadratic_function_minimum_value (q : QuadraticFunction)
  (h1 : f' q 0 > 0)
  (h2 : ∀ x : ℝ, f q x ≥ 0) :
  2 ≤ (f q 1) / (f' q 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_minimum_value_l3090_309021


namespace NUMINAMATH_CALUDE_range_of_a_l3090_309027

theorem range_of_a (a : ℝ) 
  (h1 : ∀ x ∈ Set.Icc 0 1, a ≥ Real.exp x)
  (h2 : ∃ x : ℝ, x^2 + 4*x + a = 0) :
  a ∈ Set.Icc (Real.exp 1) 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3090_309027


namespace NUMINAMATH_CALUDE_little_john_money_little_john_initial_money_l3090_309011

/-- Little John's money problem -/
theorem little_john_money : ℝ → Prop :=
  fun initial_money =>
    let spent_on_sweets : ℝ := 3.25
    let given_to_each_friend : ℝ := 2.20
    let number_of_friends : ℕ := 2
    let money_left : ℝ := 2.45
    initial_money = spent_on_sweets + (given_to_each_friend * number_of_friends) + money_left ∧
    initial_money = 10.10

/-- Proof of Little John's initial money amount -/
theorem little_john_initial_money : ∃ (m : ℝ), little_john_money m :=
  sorry

end NUMINAMATH_CALUDE_little_john_money_little_john_initial_money_l3090_309011


namespace NUMINAMATH_CALUDE_large_long_furred_brown_dogs_l3090_309000

/-- Represents the characteristics of dogs in a kennel -/
structure DogKennel where
  total : ℕ
  longFurred : ℕ
  brown : ℕ
  neitherLongFurredNorBrown : ℕ
  large : ℕ
  small : ℕ
  smallAndBrown : ℕ
  onlyLargeAndLongFurred : ℕ

/-- Theorem stating the number of large, long-furred, brown dogs -/
theorem large_long_furred_brown_dogs (k : DogKennel)
  (h1 : k.total = 60)
  (h2 : k.longFurred = 35)
  (h3 : k.brown = 25)
  (h4 : k.neitherLongFurredNorBrown = 10)
  (h5 : k.large = 30)
  (h6 : k.small = 30)
  (h7 : k.smallAndBrown = 14)
  (h8 : k.onlyLargeAndLongFurred = 7) :
  ∃ n : ℕ, n = 6 ∧ n = k.large - k.onlyLargeAndLongFurred - (k.brown - k.smallAndBrown) :=
by sorry


end NUMINAMATH_CALUDE_large_long_furred_brown_dogs_l3090_309000


namespace NUMINAMATH_CALUDE_min_value_theorem_range_theorem_l3090_309073

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 5 = 1

-- Define the left focus F₁
def F₁ : ℝ × ℝ := (-2, 0)

-- Define point P
def P : ℝ × ℝ := (1, 1)

-- Define a point M on the ellipse
def M : ℝ × ℝ := sorry

-- Distance between two points
def distance (p₁ p₂ : ℝ × ℝ) : ℝ := sorry

-- Statement for the minimum value
theorem min_value_theorem :
  ∀ M, is_on_ellipse M.1 M.2 →
  ∃ m : ℝ, m = distance M P + (3/2) * distance M F₁ ∧
  m ≥ 11/2 ∧
  ∃ M₀, is_on_ellipse M₀.1 M₀.2 ∧ distance M₀ P + (3/2) * distance M₀ F₁ = 11/2 :=
sorry

-- Statement for the range of values
theorem range_theorem :
  ∀ M, is_on_ellipse M.1 M.2 →
  ∃ r : ℝ, r = distance M P + distance M F₁ ∧
  6 - Real.sqrt 2 < r ∧ r < 6 + Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_range_theorem_l3090_309073


namespace NUMINAMATH_CALUDE_toucan_count_l3090_309019

/-- Given that there are initially 2 toucans on a tree limb and 1 more toucan joins them,
    prove that the total number of toucans is 3. -/
theorem toucan_count (initial : ℕ) (joined : ℕ) (h1 : initial = 2) (h2 : joined = 1) :
  initial + joined = 3 := by
  sorry

end NUMINAMATH_CALUDE_toucan_count_l3090_309019


namespace NUMINAMATH_CALUDE_continuity_at_seven_l3090_309048

/-- The function f(x) = 4x^2 + 6 is continuous at x₀ = 7 -/
theorem continuity_at_seven (ε : ℝ) (hε : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧ ∀ x : ℝ, |x - 7| < δ → |4 * x^2 + 6 - (4 * 7^2 + 6)| < ε := by
  sorry

end NUMINAMATH_CALUDE_continuity_at_seven_l3090_309048


namespace NUMINAMATH_CALUDE_middle_school_run_time_average_l3090_309058

/-- Represents the average number of minutes run per day by students in a specific grade -/
structure GradeRunTime where
  grade : Nat
  average_minutes : ℝ

/-- Represents the ratio of students between two grades -/
structure GradeRatio where
  higher_grade : Nat
  lower_grade : Nat
  ratio : Nat

/-- Calculates the average run time for all students given the run times for each grade and the ratios between grades -/
def calculate_average_run_time (run_times : List GradeRunTime) (ratios : List GradeRatio) : ℝ :=
  sorry

theorem middle_school_run_time_average :
  let sixth_grade := GradeRunTime.mk 6 20
  let seventh_grade := GradeRunTime.mk 7 18
  let eighth_grade := GradeRunTime.mk 8 16
  let ratio_sixth_seventh := GradeRatio.mk 6 7 3
  let ratio_seventh_eighth := GradeRatio.mk 7 8 3
  let run_times := [sixth_grade, seventh_grade, eighth_grade]
  let ratios := [ratio_sixth_seventh, ratio_seventh_eighth]
  calculate_average_run_time run_times ratios = 250 / 13 := by
    sorry

end NUMINAMATH_CALUDE_middle_school_run_time_average_l3090_309058


namespace NUMINAMATH_CALUDE_rhombus_area_l3090_309010

/-- The area of a rhombus with vertices at (0, 4.5), (8, 0), (0, -4.5), and (-8, 0) is 72 square units. -/
theorem rhombus_area : ℝ := by
  -- Define the vertices of the rhombus
  let v1 : ℝ × ℝ := (0, 4.5)
  let v2 : ℝ × ℝ := (8, 0)
  let v3 : ℝ × ℝ := (0, -4.5)
  let v4 : ℝ × ℝ := (-8, 0)

  -- Define the diagonals of the rhombus
  let d1 : ℝ := ‖v1.2 - v3.2‖ -- Distance between y-coordinates of v1 and v3
  let d2 : ℝ := ‖v2.1 - v4.1‖ -- Distance between x-coordinates of v2 and v4

  -- Calculate the area of the rhombus
  let area : ℝ := (d1 * d2) / 2

  -- Prove that the area is 72 square units
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l3090_309010


namespace NUMINAMATH_CALUDE_expression_change_l3090_309003

/-- The change in the expression x^3 - 3x + 1 when x changes by a -/
def expressionChange (x a : ℝ) : ℝ := 
  (x + a)^3 - 3*(x + a) + 1 - (x^3 - 3*x + 1)

theorem expression_change (x a : ℝ) (h : a > 0) : 
  expressionChange x a = 3*a*x^2 + 3*a^2*x + a^3 - 3*a ∧
  expressionChange x (-a) = -3*a*x^2 + 3*a^2*x - a^3 + 3*a := by
  sorry

end NUMINAMATH_CALUDE_expression_change_l3090_309003


namespace NUMINAMATH_CALUDE_sum_largest_smallest_prime_factors_546_l3090_309091

theorem sum_largest_smallest_prime_factors_546 :
  ∃ (smallest largest : ℕ),
    smallest.Prime ∧
    largest.Prime ∧
    (smallest ∣ 546) ∧
    (largest ∣ 546) ∧
    (∀ p : ℕ, p.Prime → p ∣ 546 → p ≤ largest) ∧
    (∀ p : ℕ, p.Prime → p ∣ 546 → p ≥ smallest) ∧
    smallest + largest = 15 :=
by sorry

end NUMINAMATH_CALUDE_sum_largest_smallest_prime_factors_546_l3090_309091


namespace NUMINAMATH_CALUDE_remaining_money_l3090_309049

def initial_amount : ℕ := 11
def spent_amount : ℕ := 2
def lost_amount : ℕ := 6

theorem remaining_money :
  initial_amount - spent_amount - lost_amount = 3 := by sorry

end NUMINAMATH_CALUDE_remaining_money_l3090_309049


namespace NUMINAMATH_CALUDE_circle_divides_rectangle_sides_l3090_309062

/-- A circle touching two adjacent sides of a rectangle --/
structure CircleTouchingRectangle where
  radius : ℝ
  rect_side1 : ℝ
  rect_side2 : ℝ
  (radius_positive : 0 < radius)
  (rect_sides_positive : 0 < rect_side1 ∧ 0 < rect_side2)
  (radius_fits : radius < rect_side1 ∧ radius < rect_side2)

/-- The segments into which the circle divides the rectangle sides --/
structure RectangleSegments where
  seg1 : ℝ
  seg2 : ℝ
  seg3 : ℝ
  seg4 : ℝ
  seg5 : ℝ
  seg6 : ℝ

/-- Theorem stating how the circle divides the rectangle sides --/
theorem circle_divides_rectangle_sides (c : CircleTouchingRectangle) 
  (h : c.radius = 26 ∧ c.rect_side1 = 36 ∧ c.rect_side2 = 60) :
  ∃ (s : RectangleSegments), 
    s.seg1 = 26 ∧ s.seg2 = 34 ∧ 
    s.seg3 = 26 ∧ s.seg4 = 10 ∧ 
    s.seg5 = 2 ∧ s.seg6 = 48 :=
sorry

end NUMINAMATH_CALUDE_circle_divides_rectangle_sides_l3090_309062


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3090_309088

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- State the theorem
theorem solution_set_of_inequality
  (h_even : ∀ x, f (-x) = f x)  -- f is even
  (h_derivative : ∀ x, HasDerivAt f (f' x) x)  -- f' is the derivative of f
  (h_condition : ∀ x, x < 0 → x * f' x - f x > 0)  -- condition for x < 0
  (h_f_1 : f 1 = 0)  -- f(1) = 0
  : {x : ℝ | f x / x < 0} = {x | x < -1 ∨ (0 < x ∧ x < 1)} :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3090_309088


namespace NUMINAMATH_CALUDE_total_legs_calculation_l3090_309094

/-- The total number of legs of Camden's dogs, Rico's dogs, and Samantha's cats -/
def totalLegs : ℕ := by sorry

theorem total_legs_calculation :
  let justin_dogs : ℕ := 14
  let rico_dogs : ℕ := justin_dogs + 10
  let camden_dogs : ℕ := (3 * rico_dogs) / 4
  let camden_legs : ℕ := 5 * 3 + 7 * 4 + 2 * 2
  let rico_legs : ℕ := rico_dogs * 4
  let samantha_cats : ℕ := 8
  let samantha_legs : ℕ := 6 * 4 + 2 * 3
  totalLegs = camden_legs + rico_legs + samantha_legs := by sorry

end NUMINAMATH_CALUDE_total_legs_calculation_l3090_309094


namespace NUMINAMATH_CALUDE_tom_bought_ten_candies_l3090_309098

/-- Calculates the number of candy pieces Tom bought -/
def candy_bought (initial : ℕ) (from_friend : ℕ) (final_total : ℕ) : ℕ :=
  final_total - (initial + from_friend)

/-- Theorem stating that Tom bought 10 pieces of candy -/
theorem tom_bought_ten_candies : candy_bought 2 7 19 = 10 := by
  sorry

end NUMINAMATH_CALUDE_tom_bought_ten_candies_l3090_309098


namespace NUMINAMATH_CALUDE_substitution_method_correctness_l3090_309030

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := 2 * x - y = 5
def equation2 (x y : ℝ) : Prop := y = 1 + x

-- Define the correct substitution
def correct_substitution (x : ℝ) : Prop := 2 * x - 1 - x = 5

-- Theorem statement
theorem substitution_method_correctness :
  ∀ x y : ℝ, equation1 x y ∧ equation2 x y → correct_substitution x :=
by sorry

end NUMINAMATH_CALUDE_substitution_method_correctness_l3090_309030


namespace NUMINAMATH_CALUDE_first_box_weight_l3090_309008

theorem first_box_weight (total_weight second_weight third_weight : ℕ) 
  (h1 : total_weight = 18)
  (h2 : second_weight = 11)
  (h3 : third_weight = 5)
  : ∃ first_weight : ℕ, first_weight + second_weight + third_weight = total_weight ∧ first_weight = 2 := by
  sorry

end NUMINAMATH_CALUDE_first_box_weight_l3090_309008


namespace NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l3090_309056

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

theorem geometric_sequence_increasing_condition
  (a : ℕ → ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_positive : a 1 > 0) :
  (is_increasing_sequence a → a 1^2 < a 2^2) ∧
  (a 1^2 < a 2^2 → ¬(is_increasing_sequence a → False)) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l3090_309056


namespace NUMINAMATH_CALUDE_third_day_income_l3090_309032

def cab_driver_problem (day1 day2 day3 day4 day5 : ℝ) : Prop :=
  day1 = 300 ∧ 
  day2 = 150 ∧ 
  day4 = 200 ∧ 
  day5 = 600 ∧ 
  (day1 + day2 + day3 + day4 + day5) / 5 = 400

theorem third_day_income (day1 day2 day3 day4 day5 : ℝ) 
  (h : cab_driver_problem day1 day2 day3 day4 day5) : day3 = 750 := by
  sorry

end NUMINAMATH_CALUDE_third_day_income_l3090_309032


namespace NUMINAMATH_CALUDE_fraction_equality_l3090_309026

theorem fraction_equality (x : ℝ) (h : x ≠ 1) : -2 / (2 * x - 2) = 1 / (1 - x) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3090_309026


namespace NUMINAMATH_CALUDE_exists_different_degree_same_characteristic_l3090_309072

/-- A characteristic of a polynomial -/
def characteristic (P : Polynomial ℝ) : ℝ := sorry

/-- Theorem: There exist two polynomials with different degrees but the same characteristic -/
theorem exists_different_degree_same_characteristic :
  ∃ (P1 P2 : Polynomial ℝ), 
    (Polynomial.degree P1 ≠ Polynomial.degree P2) ∧ 
    (characteristic P1 = characteristic P2) := by
  sorry

end NUMINAMATH_CALUDE_exists_different_degree_same_characteristic_l3090_309072


namespace NUMINAMATH_CALUDE_cassette_tape_cost_cassette_tape_cost_proof_l3090_309066

theorem cassette_tape_cost 
  (initial_amount : ℝ) 
  (headphone_cost : ℝ) 
  (num_tapes : ℕ) 
  (remaining_amount : ℝ) : ℝ :=
  let total_tape_cost := initial_amount - headphone_cost - remaining_amount
  total_tape_cost / num_tapes

#check cassette_tape_cost 50 25 2 7 = 9

theorem cassette_tape_cost_proof 
  (initial_amount : ℝ)
  (headphone_cost : ℝ)
  (num_tapes : ℕ)
  (remaining_amount : ℝ)
  (h1 : initial_amount = 50)
  (h2 : headphone_cost = 25)
  (h3 : num_tapes = 2)
  (h4 : remaining_amount = 7) :
  cassette_tape_cost initial_amount headphone_cost num_tapes remaining_amount = 9 := by
  sorry

end NUMINAMATH_CALUDE_cassette_tape_cost_cassette_tape_cost_proof_l3090_309066


namespace NUMINAMATH_CALUDE_complex_square_l3090_309078

-- Define the complex number i
axiom i : ℂ
axiom i_squared : i * i = -1

-- State the theorem
theorem complex_square : (1 + i) * (1 + i) = 2 * i := by sorry

end NUMINAMATH_CALUDE_complex_square_l3090_309078


namespace NUMINAMATH_CALUDE_patio_rows_l3090_309064

theorem patio_rows (r c : ℕ) : 
  r * c = 30 →
  (r + 4) * (c - 2) = 30 →
  r = 3 :=
by sorry

end NUMINAMATH_CALUDE_patio_rows_l3090_309064


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3090_309054

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b = 1) :
  ∃ (min : ℝ), min = 3 + 2 * Real.sqrt 2 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → 2 * x + y = 1 → 1 / x + 1 / y ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3090_309054


namespace NUMINAMATH_CALUDE_subset_intersection_theorem_l3090_309077

theorem subset_intersection_theorem (α : ℝ) 
  (h_pos : α > 0) (h_bound : α < (3 - Real.sqrt 5) / 2) :
  ∃ (n p : ℕ+) (S T : Finset (Finset (Fin n))),
    p > α * 2^(n : ℝ) ∧
    S.card = p ∧
    T.card = p ∧
    (∀ s ∈ S, ∀ t ∈ T, (s ∩ t).Nonempty) :=
sorry

end NUMINAMATH_CALUDE_subset_intersection_theorem_l3090_309077


namespace NUMINAMATH_CALUDE_terminating_decimals_count_l3090_309007

theorem terminating_decimals_count : 
  let n_count := Finset.filter (fun n => Nat.gcd n 420 % 3 = 0 ∧ Nat.gcd n 420 % 7 = 0) (Finset.range 419)
  Finset.card n_count = 19 := by
  sorry

end NUMINAMATH_CALUDE_terminating_decimals_count_l3090_309007


namespace NUMINAMATH_CALUDE_triangle_max_area_l3090_309079

/-- Given a triangle ABC with sides a, b, c and area S, 
    if S = a² - (b-c)² and b + c = 8, 
    then the maximum possible value of S is 64/17 -/
theorem triangle_max_area (a b c S : ℝ) : 
  S = a^2 - (b-c)^2 → b + c = 8 → (∀ S' : ℝ, S' = a'^2 - (b'-c')^2 ∧ b' + c' = 8 → S' ≤ S) → S = 64/17 :=
by sorry


end NUMINAMATH_CALUDE_triangle_max_area_l3090_309079


namespace NUMINAMATH_CALUDE_thirteen_sided_polygon_diagonals_l3090_309046

/-- The number of diagonals in a polygon with n sides. -/
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of diagonals connected to a single vertex in a polygon with n sides. -/
def diagonals_per_vertex (n : ℕ) : ℕ := n - 3

/-- The number of diagonals in a polygon with n sides where one vertex is not connected to any diagonal. -/
def diagonals_with_disconnected_vertex (n : ℕ) : ℕ :=
  diagonals n - diagonals_per_vertex n

theorem thirteen_sided_polygon_diagonals :
  diagonals_with_disconnected_vertex 13 = 55 := by
  sorry

#eval diagonals_with_disconnected_vertex 13

end NUMINAMATH_CALUDE_thirteen_sided_polygon_diagonals_l3090_309046


namespace NUMINAMATH_CALUDE_aunt_gemma_dog_food_l3090_309070

/-- The number of sacks of dog food Aunt Gemma bought -/
def num_sacks : ℕ := 2

/-- The number of dogs Aunt Gemma has -/
def num_dogs : ℕ := 4

/-- The number of times Aunt Gemma feeds her dogs per day -/
def feeds_per_day : ℕ := 2

/-- The amount of food each dog consumes per meal in grams -/
def food_per_meal : ℕ := 250

/-- The weight of each sack of dog food in kilograms -/
def sack_weight : ℕ := 50

/-- The number of days the dog food will last -/
def days_lasting : ℕ := 50

theorem aunt_gemma_dog_food :
  num_sacks = (num_dogs * feeds_per_day * food_per_meal * days_lasting) / (sack_weight * 1000) := by
  sorry

end NUMINAMATH_CALUDE_aunt_gemma_dog_food_l3090_309070


namespace NUMINAMATH_CALUDE_expand_expression_l3090_309012

theorem expand_expression (x : ℝ) : (7 * x - 3) * (3 * x^2) = 21 * x^3 - 9 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3090_309012


namespace NUMINAMATH_CALUDE_number_calculation_l3090_309083

theorem number_calculation (x : ℝ) : (0.5 * x - 10 = 25) → x = 70 := by
  sorry

end NUMINAMATH_CALUDE_number_calculation_l3090_309083


namespace NUMINAMATH_CALUDE_intersection_of_lines_l3090_309047

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a line in 2D space -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Represents a circle in 2D space -/
structure Circle :=
  (center : Point) (radius : ℝ)

/-- ABC is an acute-angled scalene triangle -/
def Triangle (A B C : Point) : Prop := sorry

/-- AH is an altitude of triangle ABC -/
def IsAltitude (H : Point) (A B C : Point) : Prop := sorry

/-- AM is a median of triangle ABC -/
def IsMedian (M : Point) (A B C : Point) : Prop := sorry

/-- O is the center of the circumscribed circle ω of triangle ABC -/
def IsCircumcenter (O : Point) (A B C : Point) (ω : Circle) : Prop := sorry

/-- Two lines intersect at a point -/
def Intersect (l1 l2 : Line) (P : Point) : Prop := sorry

/-- A line intersects a circle at a point -/
def IntersectCircle (l : Line) (c : Circle) (P : Point) : Prop := sorry

theorem intersection_of_lines 
  (A B C H M O D E F X Y : Point) 
  (ω : Circle) :
  Triangle A B C →
  IsAltitude H A B C →
  IsMedian M A B C →
  IsCircumcenter O A B C ω →
  Intersect (Line.mk 0 0 0) (Line.mk 0 0 0) D →  -- OH and AM
  Intersect (Line.mk 0 0 0) (Line.mk 0 0 0) E →  -- AB and CD
  Intersect (Line.mk 0 0 0) (Line.mk 0 0 0) F →  -- BD and AC
  IntersectCircle (Line.mk 0 0 0) ω X →  -- EH and ω
  IntersectCircle (Line.mk 0 0 0) ω Y →  -- FH and ω
  ∃ P, Intersect (Line.mk 0 0 0) (Line.mk 0 0 0) P ∧  -- BY and CX
      Intersect (Line.mk 0 0 0) (Line.mk 0 0 0) P ∧  -- CX and AH
      Intersect (Line.mk 0 0 0) (Line.mk 0 0 0) P    -- AH and BY
:= by sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l3090_309047


namespace NUMINAMATH_CALUDE_tangent_point_coordinates_l3090_309020

/-- The curve function f(x) = x^4 + x -/
def f (x : ℝ) : ℝ := x^4 + x

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 4 * x^3 + 1

theorem tangent_point_coordinates :
  ∃ (x y : ℝ), f y = f x ∧ f' x = -3 → x = -1 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_tangent_point_coordinates_l3090_309020


namespace NUMINAMATH_CALUDE_range_of_r_l3090_309031

-- Define the function r(x)
def r (x : ℝ) : ℝ := x^4 + 6*x^2 + 9

-- State the theorem
theorem range_of_r :
  Set.range (fun x : ℝ => r x) = Set.Ici 9 := by
  sorry

end NUMINAMATH_CALUDE_range_of_r_l3090_309031


namespace NUMINAMATH_CALUDE_husband_saves_225_monthly_l3090_309022

/-- Represents the savings and investment scenario of a married couple -/
structure SavingsScenario where
  wife_weekly_savings : ℕ
  savings_period_months : ℕ
  stock_price : ℕ
  stocks_bought : ℕ

/-- Calculates the husband's monthly savings based on the given scenario -/
def husband_monthly_savings (scenario : SavingsScenario) : ℕ :=
  let total_savings := 2 * scenario.stock_price * scenario.stocks_bought
  let wife_total_savings := scenario.wife_weekly_savings * 4 * scenario.savings_period_months
  let husband_total_savings := total_savings - wife_total_savings
  husband_total_savings / scenario.savings_period_months

/-- Theorem stating that given the specific scenario, the husband's monthly savings is $225 -/
theorem husband_saves_225_monthly (scenario : SavingsScenario) 
  (h1 : scenario.wife_weekly_savings = 100)
  (h2 : scenario.savings_period_months = 4)
  (h3 : scenario.stock_price = 50)
  (h4 : scenario.stocks_bought = 25) :
  husband_monthly_savings scenario = 225 := by
  sorry

end NUMINAMATH_CALUDE_husband_saves_225_monthly_l3090_309022


namespace NUMINAMATH_CALUDE_exactly_one_greater_than_one_l3090_309060

theorem exactly_one_greater_than_one (x₁ x₂ x₃ : ℝ) 
  (h_positive : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0)
  (h_product : x₁ * x₂ * x₃ = 1)
  (h_sum : x₁ + x₂ + x₃ > 1/x₁ + 1/x₂ + 1/x₃) :
  (x₁ > 1 ∧ x₂ ≤ 1 ∧ x₃ ≤ 1) ∨
  (x₁ ≤ 1 ∧ x₂ > 1 ∧ x₃ ≤ 1) ∨
  (x₁ ≤ 1 ∧ x₂ ≤ 1 ∧ x₃ > 1) :=
by sorry

end NUMINAMATH_CALUDE_exactly_one_greater_than_one_l3090_309060


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3090_309017

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

/-- The sum of specific terms in the sequence equals 120 -/
def sum_condition (a : ℕ → ℝ) : Prop :=
  a 4 + a 6 + a 8 + a 10 + a 12 = 120

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) (h2 : sum_condition a) :
  2 * a 10 - a 12 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3090_309017


namespace NUMINAMATH_CALUDE_right_triangle_identification_l3090_309042

def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

theorem right_triangle_identification :
  ¬(is_right_triangle 6 15 17) ∧
  ¬(is_right_triangle 7 12 15) ∧
  is_right_triangle 7 24 25 ∧
  ¬(is_right_triangle 13 15 20) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_identification_l3090_309042


namespace NUMINAMATH_CALUDE_sum_divisors_bound_l3090_309015

/-- σ(n) is the sum of the divisors of n -/
def sigma (n : ℕ+) : ℕ := sorry

/-- ω(n) is the number of distinct prime divisors of n -/
def omega (n : ℕ+) : ℕ := sorry

/-- The sum of divisors of n is less than n multiplied by one more than
    the number of its distinct prime divisors -/
theorem sum_divisors_bound (n : ℕ+) : sigma n < n * (omega n + 1) := by sorry

end NUMINAMATH_CALUDE_sum_divisors_bound_l3090_309015


namespace NUMINAMATH_CALUDE_complex_product_theorem_l3090_309059

theorem complex_product_theorem (a b : ℝ) :
  let z₁ : ℂ := Complex.mk a b
  let z₂ : ℂ := Complex.mk a (-b)
  let z₃ : ℂ := Complex.mk (-a) b
  let z₄ : ℂ := Complex.mk (-a) (-b)
  (z₁ * z₂ * z₃ * z₄).re = (a^2 + b^2)^2 ∧ (z₁ * z₂ * z₃ * z₄).im = 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_product_theorem_l3090_309059


namespace NUMINAMATH_CALUDE_exponent_equality_l3090_309051

theorem exponent_equality : 8^5 * 3^5 * 8^3 * 3^7 = 8^8 * 3^12 := by
  sorry

end NUMINAMATH_CALUDE_exponent_equality_l3090_309051


namespace NUMINAMATH_CALUDE_initial_concentration_proof_l3090_309013

/-- Proves that the initial concentration of a hydrochloric acid solution is 20%
    given the conditions of the problem. -/
theorem initial_concentration_proof (
  initial_amount : ℝ)
  (drained_amount : ℝ)
  (added_concentration : ℝ)
  (final_concentration : ℝ)
  (h1 : initial_amount = 300)
  (h2 : drained_amount = 25)
  (h3 : added_concentration = 80 / 100)
  (h4 : final_concentration = 25 / 100)
  : ∃ (initial_concentration : ℝ),
    initial_concentration = 20 / 100 ∧
    (initial_amount - drained_amount) * initial_concentration +
    drained_amount * added_concentration =
    initial_amount * final_concentration :=
by sorry

end NUMINAMATH_CALUDE_initial_concentration_proof_l3090_309013


namespace NUMINAMATH_CALUDE_power_equality_l3090_309061

theorem power_equality (q : ℕ) : 64^4 = 8^q → q = 8 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l3090_309061


namespace NUMINAMATH_CALUDE_tank_plastering_cost_l3090_309001

/-- Calculates the cost of plastering a rectangular tank's walls and bottom -/
def plasteringCost (length width height rate : ℝ) : ℝ :=
  let wallArea := 2 * (length * height + width * height)
  let bottomArea := length * width
  let totalArea := wallArea + bottomArea
  totalArea * rate

/-- Theorem stating the cost of plastering the specific tank -/
theorem tank_plastering_cost :
  plasteringCost 25 12 6 0.75 = 558 := by
  sorry

end NUMINAMATH_CALUDE_tank_plastering_cost_l3090_309001
