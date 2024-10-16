import Mathlib

namespace NUMINAMATH_CALUDE_water_equals_sugar_in_new_recipe_l1300_130050

/-- Represents a recipe with ratios of flour, water, and sugar -/
structure Recipe :=
  (flour : ℚ)
  (water : ℚ)
  (sugar : ℚ)

/-- Creates a new recipe by doubling the flour to water ratio and halving the flour to sugar ratio -/
def newRecipe (r : Recipe) : Recipe :=
  { flour := r.flour * 2,
    water := r.water,
    sugar := r.sugar / 2 }

/-- Calculates the amount of water needed given the amount of sugar and the recipe ratios -/
def waterNeeded (r : Recipe) (sugarAmount : ℚ) : ℚ :=
  (r.water / r.sugar) * sugarAmount

theorem water_equals_sugar_in_new_recipe (originalRecipe : Recipe) (sugarAmount : ℚ) :
  let newRecipe := newRecipe originalRecipe
  waterNeeded newRecipe sugarAmount = sugarAmount :=
by sorry

#check water_equals_sugar_in_new_recipe

end NUMINAMATH_CALUDE_water_equals_sugar_in_new_recipe_l1300_130050


namespace NUMINAMATH_CALUDE_average_daily_low_temp_l1300_130049

def daily_low_temperatures : List ℝ := [40, 47, 45, 41, 39]

theorem average_daily_low_temp : 
  (daily_low_temperatures.sum / daily_low_temperatures.length : ℝ) = 42.4 := by
  sorry

end NUMINAMATH_CALUDE_average_daily_low_temp_l1300_130049


namespace NUMINAMATH_CALUDE_child_grandmother_weight_ratio_l1300_130056

/-- Represents the weights of family members and their relationships -/
structure FamilyWeights where
  grandmother : ℝ
  daughter : ℝ
  child : ℝ
  total_weight : grandmother + daughter + child = 130
  daughter_child_weight : daughter + child = 60
  daughter_weight : daughter = 46

/-- The ratio of the child's weight to the grandmother's weight is 1:5 -/
theorem child_grandmother_weight_ratio (fw : FamilyWeights) :
  fw.child / fw.grandmother = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_child_grandmother_weight_ratio_l1300_130056


namespace NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l1300_130055

theorem smallest_positive_integer_with_remainders : ∃ M : ℕ+,
  (M : ℕ) % 3 = 2 ∧
  (M : ℕ) % 4 = 3 ∧
  (M : ℕ) % 5 = 4 ∧
  (M : ℕ) % 6 = 5 ∧
  (M : ℕ) % 7 = 6 ∧
  (∀ n : ℕ+, n < M →
    (n : ℕ) % 3 ≠ 2 ∨
    (n : ℕ) % 4 ≠ 3 ∨
    (n : ℕ) % 5 ≠ 4 ∨
    (n : ℕ) % 6 ≠ 5 ∨
    (n : ℕ) % 7 ≠ 6) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l1300_130055


namespace NUMINAMATH_CALUDE_chord_length_l1300_130012

/-- The polar equation of line l is √3ρcosθ + ρsinθ - 1 = 0 -/
def line_l (ρ θ : ℝ) : Prop :=
  Real.sqrt 3 * ρ * Real.cos θ + ρ * Real.sin θ - 1 = 0

/-- The polar equation of curve C is ρ = 4 -/
def curve_C (ρ : ℝ) : Prop := ρ = 4

/-- The length of the chord formed by the intersection of l and C is 3√7 -/
theorem chord_length : 
  ∃ (A B : ℝ × ℝ), 
    (∃ (ρ_A θ_A ρ_B θ_B : ℝ), 
      line_l ρ_A θ_A ∧ line_l ρ_B θ_B ∧ 
      curve_C ρ_A ∧ curve_C ρ_B ∧
      A = (ρ_A * Real.cos θ_A, ρ_A * Real.sin θ_A) ∧
      B = (ρ_B * Real.cos θ_B, ρ_B * Real.sin θ_B)) →
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 3 * Real.sqrt 7 :=
sorry

end NUMINAMATH_CALUDE_chord_length_l1300_130012


namespace NUMINAMATH_CALUDE_student_ranking_l1300_130057

theorem student_ranking (total : Nat) (rank_right : Nat) (rank_left : Nat) : 
  total = 31 → rank_right = 21 → rank_left = total - rank_right + 1 → rank_left = 11 := by
  sorry

end NUMINAMATH_CALUDE_student_ranking_l1300_130057


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l1300_130025

/-- Given a geometric sequence with first term 512 and sixth term 32, 
    the fourth term is 64. -/
theorem geometric_sequence_fourth_term : ∀ (a : ℝ → ℝ),
  (∀ n : ℕ, a (n + 1) = a n * (a 1)⁻¹ * a 0) →  -- Geometric sequence property
  a 0 = 512 →                                  -- First term is 512
  a 5 = 32 →                                   -- Sixth term is 32
  a 3 = 64 :=                                  -- Fourth term is 64
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l1300_130025


namespace NUMINAMATH_CALUDE_dealer_gain_percent_l1300_130021

theorem dealer_gain_percent (list_price : ℝ) (list_price_pos : list_price > 0) :
  let purchase_price := (3/4) * list_price
  let selling_price := (3/2) * list_price
  let gain := selling_price - purchase_price
  let gain_percent := (gain / purchase_price) * 100
  gain_percent = 100 := by sorry

end NUMINAMATH_CALUDE_dealer_gain_percent_l1300_130021


namespace NUMINAMATH_CALUDE_no_integer_solution_for_equation_l1300_130077

theorem no_integer_solution_for_equation : ¬ ∃ (x y : ℤ), x^2 - y^2 = 1998 := by sorry

end NUMINAMATH_CALUDE_no_integer_solution_for_equation_l1300_130077


namespace NUMINAMATH_CALUDE_pentagon_right_angles_l1300_130079

/-- The sum of interior angles in a pentagon in degrees -/
def pentagonAngleSum : ℝ := 540

/-- The measure of a right angle in degrees -/
def rightAngle : ℝ := 90

/-- The set of possible numbers of right angles in a pentagon -/
def possibleRightAngles : Set ℕ := {0, 1, 2, 3}

/-- Theorem: The set of possible numbers of right angles in a pentagon is {0, 1, 2, 3} -/
theorem pentagon_right_angles :
  ∀ n : ℕ, n ∈ possibleRightAngles ↔ 
    (n : ℝ) * rightAngle ≤ pentagonAngleSum ∧ 
    (n + 1 : ℝ) * rightAngle > pentagonAngleSum :=
by sorry

end NUMINAMATH_CALUDE_pentagon_right_angles_l1300_130079


namespace NUMINAMATH_CALUDE_cookies_eaten_l1300_130070

/-- Given a package of cookies where some were eaten, this theorem proves
    the number of cookies eaten, given the initial count and remaining count. -/
theorem cookies_eaten (initial : ℕ) (remaining : ℕ) (h : initial = 18) (h' : remaining = 9) :
  initial - remaining = 9 := by
  sorry

end NUMINAMATH_CALUDE_cookies_eaten_l1300_130070


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l1300_130037

def vector_a : Fin 2 → ℝ := ![2, -1]
def vector_b (x : ℝ) : Fin 2 → ℝ := ![6, x]

theorem vector_difference_magnitude 
  (h_parallel : ∃ (k : ℝ), ∀ i, vector_a i = k * vector_b x i) :
  ∃ (x : ℝ), ‖vector_a - vector_b x‖ = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l1300_130037


namespace NUMINAMATH_CALUDE_orange_banana_relationship_l1300_130052

/-- The cost of fruits at Frank's Fruit Market -/
structure FruitCost where
  banana_to_apple : ℚ  -- ratio of bananas to apples
  apple_to_orange : ℚ  -- ratio of apples to oranges

/-- Given the cost ratios, calculate how many oranges cost as much as 24 bananas -/
def oranges_for_24_bananas (cost : FruitCost) : ℚ :=
  24 * (cost.banana_to_apple * cost.apple_to_orange)

/-- Theorem stating the relationship between banana and orange costs -/
theorem orange_banana_relationship (cost : FruitCost)
  (h1 : cost.banana_to_apple = 4 / 3)
  (h2 : cost.apple_to_orange = 5 / 2) :
  oranges_for_24_bananas cost = 36 / 5 := by
  sorry

#eval oranges_for_24_bananas ⟨4/3, 5/2⟩

end NUMINAMATH_CALUDE_orange_banana_relationship_l1300_130052


namespace NUMINAMATH_CALUDE_jakes_snake_sales_l1300_130010

def snakes : ℕ := 3
def eggs_per_snake : ℕ := 2
def regular_price : ℕ := 250
def rare_price_multiplier : ℕ := 4

def total_eggs : ℕ := snakes * eggs_per_snake
def rare_snakes : ℕ := 1
def regular_snakes : ℕ := total_eggs - rare_snakes

def total_sales : ℕ :=
  regular_snakes * regular_price + rare_snakes * (rare_price_multiplier * regular_price)

theorem jakes_snake_sales : total_sales = 2250 := by
  sorry

end NUMINAMATH_CALUDE_jakes_snake_sales_l1300_130010


namespace NUMINAMATH_CALUDE_intersecting_line_passes_through_fixed_point_l1300_130020

/-- An ellipse with eccentricity 1/2 passing through (1, 3/2) -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0
  h_ecc : (a^2 - b^2) / a^2 = 1/4
  h_point : 1^2 / a^2 + (3/2)^2 / b^2 = 1

/-- A line intersecting the ellipse -/
structure IntersectingLine (E : Ellipse) where
  k : ℝ
  m : ℝ
  h_intersect : ∃ (x₁ y₁ x₂ y₂ : ℝ),
    x₁ ≠ x₂ ∧
    y₁ = k * x₁ + m ∧
    y₂ = k * x₂ + m ∧
    x₁^2 / E.a^2 + y₁^2 / E.b^2 = 1 ∧
    x₂^2 / E.a^2 + y₂^2 / E.b^2 = 1

/-- The theorem stating that the line passes through a fixed point -/
theorem intersecting_line_passes_through_fixed_point (E : Ellipse) (l : IntersectingLine E) :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    x₁ ≠ x₂ ∧
    y₁ = l.k * x₁ + l.m ∧
    y₂ = l.k * x₂ + l.m ∧
    x₁^2 / E.a^2 + y₁^2 / E.b^2 = 1 ∧
    x₂^2 / E.a^2 + y₂^2 / E.b^2 = 1 ∧
    (x₁ - E.a) * (x₂ - E.a) + y₁ * y₂ = 0 →
    l.k * (2/7) + l.m = 0 :=
by sorry

end NUMINAMATH_CALUDE_intersecting_line_passes_through_fixed_point_l1300_130020


namespace NUMINAMATH_CALUDE_loan_interest_calculation_l1300_130011

/-- Calculate simple interest --/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Prove that the total interest for a $12500 loan at 12% for 1 year is $1500 --/
theorem loan_interest_calculation :
  let principal : ℝ := 12500
  let rate : ℝ := 0.12
  let time : ℝ := 1
  simple_interest principal rate time = 1500 := by
sorry

end NUMINAMATH_CALUDE_loan_interest_calculation_l1300_130011


namespace NUMINAMATH_CALUDE_movie_theater_revenue_l1300_130051

/-- 
Calculates the total revenue from movie ticket sales given the prices and quantities sold.
-/
theorem movie_theater_revenue 
  (matinee_price : ℕ) 
  (evening_price : ℕ) 
  (three_d_price : ℕ)
  (matinee_quantity : ℕ)
  (evening_quantity : ℕ)
  (three_d_quantity : ℕ)
  (h1 : matinee_price = 5)
  (h2 : evening_price = 12)
  (h3 : three_d_price = 20)
  (h4 : matinee_quantity = 200)
  (h5 : evening_quantity = 300)
  (h6 : three_d_quantity = 100) :
  matinee_price * matinee_quantity + 
  evening_price * evening_quantity + 
  three_d_price * three_d_quantity = 6600 :=
by
  sorry

#check movie_theater_revenue

end NUMINAMATH_CALUDE_movie_theater_revenue_l1300_130051


namespace NUMINAMATH_CALUDE_cubic_function_increasing_iff_a_nonpositive_l1300_130089

/-- Theorem: For the function f(x) = x^3 - ax + 1 where a ∈ ℝ, 
    f(x) is increasing in its domain if and only if a ≤ 0 -/
theorem cubic_function_increasing_iff_a_nonpositive (a : ℝ) :
  (∀ x : ℝ, HasDerivAt (fun x => x^3 - a*x + 1) (3*x^2 - a) x) →
  (∀ x y : ℝ, x < y → (x^3 - a*x + 1) < (y^3 - a*y + 1)) ↔ a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_increasing_iff_a_nonpositive_l1300_130089


namespace NUMINAMATH_CALUDE_pairball_longest_time_l1300_130029

/-- Represents the pairball game setup -/
structure PairballGame where
  totalTime : ℕ
  numChildren : ℕ
  longestPlayRatio : ℕ

/-- Calculates the playing time of the child who played the longest -/
def longestPlayingTime (game : PairballGame) : ℕ :=
  let totalChildMinutes := 2 * game.totalTime
  let adjustedChildren := game.numChildren - 1 + game.longestPlayRatio
  (totalChildMinutes * game.longestPlayRatio) / adjustedChildren

/-- Theorem stating that the longest playing time in the given scenario is 68 minutes -/
theorem pairball_longest_time :
  let game : PairballGame := {
    totalTime := 120,
    numChildren := 6,
    longestPlayRatio := 2
  }
  longestPlayingTime game = 68 := by
  sorry

end NUMINAMATH_CALUDE_pairball_longest_time_l1300_130029


namespace NUMINAMATH_CALUDE_larger_number_problem_l1300_130031

theorem larger_number_problem (x y : ℝ) (h1 : x - y = 8) (h2 : (1/4) * (x + y) = 6) : max x y = 16 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l1300_130031


namespace NUMINAMATH_CALUDE_remaining_unit_area_l1300_130016

theorem remaining_unit_area (total_units : ℕ) (total_area : ℝ) 
  (units_10x7 : ℕ) (units_14x6 : ℕ) 
  (h1 : total_units = 120)
  (h2 : total_area = 15300)
  (h3 : units_10x7 = 50)
  (h4 : units_14x6 = 35) :
  let remaining_units := total_units - units_10x7 - units_14x6
  let remaining_area := total_area - (units_10x7 * 10 * 7) - (units_14x6 * 14 * 6)
  remaining_area / remaining_units = (15300 - 50 * 10 * 7 - 35 * 14 * 6) / (120 - 50 - 35) :=
by sorry

end NUMINAMATH_CALUDE_remaining_unit_area_l1300_130016


namespace NUMINAMATH_CALUDE_equation_solution_l1300_130080

theorem equation_solution :
  ∀ x : ℝ, x ≠ 2 ∧ x ≠ 4/5 →
  (x^2 - 11*x + 24)/(x - 2) + (5*x^2 + 22*x - 48)/(5*x - 4) = -7 →
  x = -4/3 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1300_130080


namespace NUMINAMATH_CALUDE_museum_artifacts_l1300_130069

theorem museum_artifacts (total_wings : ℕ) (painting_wings : ℕ) (large_painting : ℕ) 
  (small_paintings_per_wing : ℕ) (artifact_multiplier : ℕ) :
  total_wings = 8 →
  painting_wings = 3 →
  large_painting = 1 →
  small_paintings_per_wing = 12 →
  artifact_multiplier = 4 →
  let total_paintings := large_painting + 2 * small_paintings_per_wing
  let total_artifacts := artifact_multiplier * total_paintings
  let artifact_wings := total_wings - painting_wings
  total_artifacts / artifact_wings = 20 :=
by sorry

end NUMINAMATH_CALUDE_museum_artifacts_l1300_130069


namespace NUMINAMATH_CALUDE_circle_tangents_and_chord_l1300_130005

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 5 = 0

-- Define the tangent lines
def tangent1 (x y : ℝ) : Prop := 4*x + 3*y - 23 = 0
def tangent2 (x : ℝ) : Prop := x = 5

-- Define the chord line
def chord_line (x y : ℝ) : Prop := x + y - 4 = 0

-- Theorem statement
theorem circle_tangents_and_chord :
  -- Part 1: Tangent lines
  (∀ x y, C x y → (tangent1 x y → x^2 + y^2 = 25)) ∧
  (∀ x y, C x y → (tangent2 x → x^2 + y^2 = 25)) ∧
  tangent1 5 1 ∧ tangent2 5 ∧
  -- Part 2: Chord line
  (∀ x₁ y₁ x₂ y₂, C x₁ y₁ ∧ C x₂ y₂ ∧ (x₁ + x₂)/2 = 3 ∧ (y₁ + y₂)/2 = 1 →
    chord_line x₁ y₁ ∧ chord_line x₂ y₂) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangents_and_chord_l1300_130005


namespace NUMINAMATH_CALUDE_packages_sold_correct_l1300_130022

/-- The number of packages of gaskets sold during a week -/
def packages_sold : ℕ := 66

/-- The price per package of gaskets -/
def price_per_package : ℚ := 20

/-- The discount factor for packages in excess of 10 -/
def discount_factor : ℚ := 4/5

/-- The total payment received for the gaskets -/
def total_payment : ℚ := 1096

/-- Calculates the total cost for the given number of packages -/
def total_cost (n : ℕ) : ℚ :=
  if n ≤ 10 then n * price_per_package
  else 10 * price_per_package + (n - 10) * (discount_factor * price_per_package)

/-- Theorem stating that the number of packages sold satisfies the given conditions -/
theorem packages_sold_correct : 
  total_cost packages_sold = total_payment := by sorry

end NUMINAMATH_CALUDE_packages_sold_correct_l1300_130022


namespace NUMINAMATH_CALUDE_geometry_relations_l1300_130026

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (line_perpendicular : Line → Line → Prop)
variable (line_parallel : Line → Line → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem geometry_relations 
  (m l : Line) (α β : Plane)
  (h1 : perpendicular m α)
  (h2 : contained_in l β) :
  (((parallel α β) → (line_perpendicular m l)) ∧
   ((line_parallel m l) → (plane_perpendicular α β))) ∧
  ¬(((plane_perpendicular α β) → (line_parallel m l)) ∧
    ((line_perpendicular m l) → (parallel α β))) :=
sorry

end NUMINAMATH_CALUDE_geometry_relations_l1300_130026


namespace NUMINAMATH_CALUDE_money_distribution_l1300_130061

theorem money_distribution (a b c : ℕ) 
  (total : a + b + c = 700)
  (ac_sum : a + c = 300)
  (bc_sum : b + c = 600) :
  c = 200 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l1300_130061


namespace NUMINAMATH_CALUDE_ball_rolling_cycloid_l1300_130008

/-- Represents the path of a ball rolling down a smooth cycloidal trough -/
noncomputable def path (a g t : ℝ) : ℝ :=
  4 * a * (1 - Real.cos (t * Real.sqrt (g / (4 * a))))

/-- Time for the ball to roll from the start to the lowest point along the cycloid -/
noncomputable def time_cycloid (a g : ℝ) : ℝ :=
  Real.pi * Real.sqrt (a / g)

/-- Time for the ball to roll from the start to the lowest point along a straight line -/
noncomputable def time_straight (a g : ℝ) : ℝ :=
  Real.sqrt (a * (4 + Real.pi^2) / g)

theorem ball_rolling_cycloid (a g : ℝ) (ha : a > 0) (hg : g > 0) :
  (∀ t, path a g t = 4 * a * (1 - Real.cos (t * Real.sqrt (g / (4 * a))))) ∧
  time_cycloid a g = Real.pi * Real.sqrt (a / g) ∧
  time_straight a g = Real.sqrt (a * (4 + Real.pi^2) / g) ∧
  time_cycloid a g < time_straight a g :=
sorry

end NUMINAMATH_CALUDE_ball_rolling_cycloid_l1300_130008


namespace NUMINAMATH_CALUDE_rectangular_plot_perimeter_l1300_130090

/-- Given a rectangular plot with length 10 meters more than width,
    and fencing cost of Rs. 910 at Rs. 6.5 per meter,
    prove that the perimeter is 140 meters. -/
theorem rectangular_plot_perimeter : 
  ∀ (width length : ℝ),
  length = width + 10 →
  910 = (2 * (length + width)) * 6.5 →
  2 * (length + width) = 140 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_perimeter_l1300_130090


namespace NUMINAMATH_CALUDE_parabola_focus_l1300_130060

/-- The focus of the parabola x^2 = 8y has coordinates (0, 2) -/
theorem parabola_focus (x y : ℝ) :
  (x^2 = 8*y) → (∃ p : ℝ, p = 2 ∧ (0, p) = (0, 2)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_l1300_130060


namespace NUMINAMATH_CALUDE_binomial_variance_example_l1300_130047

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- The variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

/-- Theorem: The variance of X ~ B(8, 0.7) is 1.68 -/
theorem binomial_variance_example :
  let X : BinomialRV := ⟨8, 0.7, by norm_num⟩
  variance X = 1.68 := by
  sorry

end NUMINAMATH_CALUDE_binomial_variance_example_l1300_130047


namespace NUMINAMATH_CALUDE_function_property_l1300_130054

def isEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def hasPeriod (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem function_property (f : ℝ → ℝ) 
  (h_even : isEven f)
  (h_period : hasPeriod f 2)
  (h_interval : ∀ x ∈ Set.Icc 2 3, f x = x) :
  ∀ x ∈ Set.Icc (-2) 0, f x = 3 - |x + 1| := by
sorry

end NUMINAMATH_CALUDE_function_property_l1300_130054


namespace NUMINAMATH_CALUDE_solve_for_a_l1300_130030

-- Define the equation
def equation (a b c : ℝ) : Prop :=
  a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1)

-- Define the theorem
theorem solve_for_a :
  ∀ a : ℝ, equation a 15 7 → a * 15 * 7 = 1.5 → a = 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l1300_130030


namespace NUMINAMATH_CALUDE_total_baseball_cards_l1300_130042

def mary_cards : ℕ := 15
def sam_cards : ℕ := 15
def keith_cards : ℕ := 15
def alyssa_cards : ℕ := 15
def john_cards : ℕ := 12
def sarah_cards : ℕ := 18
def emma_cards : ℕ := 10

theorem total_baseball_cards :
  mary_cards + sam_cards + keith_cards + alyssa_cards +
  john_cards + sarah_cards + emma_cards = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_baseball_cards_l1300_130042


namespace NUMINAMATH_CALUDE_smallest_inverse_undefined_l1300_130099

theorem smallest_inverse_undefined (a : ℕ) : a = 6 ↔ 
  a > 0 ∧ 
  (∀ k < a, k > 0 → (Nat.gcd k 72 = 1 ∨ Nat.gcd k 90 = 1)) ∧
  Nat.gcd a 72 > 1 ∧ 
  Nat.gcd a 90 > 1 :=
sorry

end NUMINAMATH_CALUDE_smallest_inverse_undefined_l1300_130099


namespace NUMINAMATH_CALUDE_coefficient_sum_l1300_130081

-- Define the sets A and B
def A : Set ℝ := {x | x^3 + 3*x^2 + 2*x > 0}
def B : Set ℝ := {x | ∃ (a b : ℝ), x^2 + a*x + b ≤ 0}

-- Define the intersection and union of A and B
def intersection : Set ℝ := {x | 0 < x ∧ x ≤ 2}
def union : Set ℝ := {x | x > -2}

-- State the theorem
theorem coefficient_sum (a b : ℝ) : 
  A ∩ B = intersection → A ∪ B = union → a + b = -3 := by sorry

end NUMINAMATH_CALUDE_coefficient_sum_l1300_130081


namespace NUMINAMATH_CALUDE_tangent_lines_range_l1300_130002

/-- The function f(x) = x^3 - 3x --/
def f (x : ℝ) : ℝ := x^3 - 3*x

/-- The derivative of f(x) --/
def f' (x : ℝ) : ℝ := 3*x^2 - 3

/-- The function g(x) = 2x^3 - 6x^2 --/
def g (x : ℝ) : ℝ := 2*x^3 - 6*x^2

/-- The derivative of g(x) --/
def g' (x : ℝ) : ℝ := 6*x^2 - 12*x

/-- Theorem: If three distinct tangent lines to f(x) pass through A(2, m), then -6 < m < 2 --/
theorem tangent_lines_range (m : ℝ) : 
  (∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    (m - (f a)) = (f' a) * (2 - a) ∧
    (m - (f b)) = (f' b) * (2 - b) ∧
    (m - (f c)) = (f' c) * (2 - c)) →
  -6 < m ∧ m < 2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_lines_range_l1300_130002


namespace NUMINAMATH_CALUDE_ecommerce_sales_analysis_l1300_130035

/-- Represents the sales model for an e-commerce platform. -/
structure SalesModel where
  cost_price : ℝ
  initial_price : ℝ
  initial_sales : ℝ
  price_sensitivity : ℝ

/-- Calculates the daily sales volume for a given price. -/
def daily_sales (model : SalesModel) (price : ℝ) : ℝ :=
  model.initial_sales + model.price_sensitivity * (model.initial_price - price)

/-- Calculates the daily profit for a given price. -/
def daily_profit (model : SalesModel) (price : ℝ) : ℝ :=
  (price - model.cost_price) * daily_sales model price

/-- The e-commerce platform's sales model. -/
def ecommerce_model : SalesModel := {
  cost_price := 40
  initial_price := 60
  initial_sales := 20
  price_sensitivity := 2
}

/-- Xiao Ming's store price. -/
def xiaoming_price : ℝ := 62.5

theorem ecommerce_sales_analysis 
  (h1 : daily_sales ecommerce_model 45 = 50)
  (h2 : ∃ x, x ≥ 40 ∧ x < 60 ∧ daily_profit ecommerce_model x = daily_profit ecommerce_model 60 ∧
             ∀ y, y ≥ 40 ∧ y < 60 ∧ daily_profit ecommerce_model y = daily_profit ecommerce_model 60 → x ≤ y)
  (h3 : ∃ d : ℝ, d ≥ 0 ∧ d ≤ 1 ∧ xiaoming_price * (1 - d) ≤ 50 ∧
             ∀ e, e ≥ 0 ∧ e < d ∧ xiaoming_price * (1 - e) ≤ 50 → False) :
  (daily_sales ecommerce_model 45 = 50) ∧
  (∃ x, x = 50 ∧ daily_profit ecommerce_model x = daily_profit ecommerce_model 60 ∧
        ∀ y, y ≥ 40 ∧ y < 60 ∧ daily_profit ecommerce_model y = daily_profit ecommerce_model 60 → x ≤ y) ∧
  (∃ d : ℝ, d = 0.2 ∧ xiaoming_price * (1 - d) ≤ 50 ∧
            ∀ e, e ≥ 0 ∧ e < d ∧ xiaoming_price * (1 - e) ≤ 50 → False) := by
  sorry

end NUMINAMATH_CALUDE_ecommerce_sales_analysis_l1300_130035


namespace NUMINAMATH_CALUDE_random_variables_comparison_l1300_130045

-- Define the random variables ξ and η
def ξ (a b c : ℝ) : ℝ → ℝ := sorry
def η (a b c : ℝ) : ℝ → ℝ := sorry

-- Define the probability measure
def P : Set ℝ → ℝ := sorry

-- Define expected value
def E (X : ℝ → ℝ) : ℝ := sorry

-- Define variance
def D (X : ℝ → ℝ) : ℝ := sorry

theorem random_variables_comparison (a b c : ℝ) 
  (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ a) :
  (E (ξ a b c) = E (η a b c)) ∧ (D (ξ a b c) > D (η a b c)) := by
  sorry

end NUMINAMATH_CALUDE_random_variables_comparison_l1300_130045


namespace NUMINAMATH_CALUDE_calculation_proof_l1300_130075

theorem calculation_proof : (1/4) * 6.16^2 - 4 * 1.04^2 = 5.16 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1300_130075


namespace NUMINAMATH_CALUDE_problem_statement_l1300_130028

theorem problem_statement (x y : ℝ) (hx : x = 20) (hy : y = 8) :
  (x - y) * (x + y) = 336 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1300_130028


namespace NUMINAMATH_CALUDE_cooking_cleaning_arrangements_l1300_130092

theorem cooking_cleaning_arrangements (n : ℕ) (h : n = 8) : 
  Nat.choose n (n / 2) = 70 := by
  sorry

end NUMINAMATH_CALUDE_cooking_cleaning_arrangements_l1300_130092


namespace NUMINAMATH_CALUDE_binomial_18_10_l1300_130064

theorem binomial_18_10 (h1 : Nat.choose 16 7 = 8008) (h2 : Nat.choose 16 9 = 11440) :
  Nat.choose 18 10 = 43758 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_10_l1300_130064


namespace NUMINAMATH_CALUDE_least_perimeter_triangle_l1300_130088

theorem least_perimeter_triangle (d e f : ℕ) : 
  d > 0 → e > 0 → f > 0 →
  (d^2 + e^2 - f^2) / (2 * d * e : ℚ) = 24/25 →
  (d^2 + f^2 - e^2) / (2 * d * f : ℚ) = 3/5 →
  (e^2 + f^2 - d^2) / (2 * e * f : ℚ) = -2/5 →
  d + e + f ≥ 32 :=
by sorry

end NUMINAMATH_CALUDE_least_perimeter_triangle_l1300_130088


namespace NUMINAMATH_CALUDE_distinct_subscription_selections_l1300_130018

def number_of_providers : ℕ := 25
def number_of_siblings : ℕ := 4

theorem distinct_subscription_selections :
  (number_of_providers - 0) *
  (number_of_providers - 1) *
  (number_of_providers - 2) *
  (number_of_providers - 3) = 303600 := by
  sorry

end NUMINAMATH_CALUDE_distinct_subscription_selections_l1300_130018


namespace NUMINAMATH_CALUDE_max_pages_copied_l1300_130046

-- Define the cost per 2 pages in cents
def cost_per_2_pages : ℕ := 7

-- Define the fixed fee in cents
def fixed_fee : ℕ := 500

-- Define the total budget in cents
def total_budget : ℕ := 3500

-- Define the function to calculate the number of pages
def pages_copied (budget : ℕ) : ℕ :=
  ((budget - fixed_fee) * 2) / cost_per_2_pages

-- Theorem statement
theorem max_pages_copied :
  pages_copied total_budget = 857 := by
  sorry

end NUMINAMATH_CALUDE_max_pages_copied_l1300_130046


namespace NUMINAMATH_CALUDE_no_such_function_l1300_130063

theorem no_such_function : ¬∃ f : ℝ → ℝ, (f 0 > 0) ∧ (∀ x y : ℝ, f (x + y) ≥ f x + y * f (f x)) := by
  sorry

end NUMINAMATH_CALUDE_no_such_function_l1300_130063


namespace NUMINAMATH_CALUDE_max_value_abc_l1300_130096

theorem max_value_abc (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 3) :
  a + b^2 + c^4 ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_max_value_abc_l1300_130096


namespace NUMINAMATH_CALUDE_equality_of_cyclic_sum_powers_l1300_130058

theorem equality_of_cyclic_sum_powers (n : ℕ+) (p : ℕ) (a b c : ℤ) 
  (h_prime : Nat.Prime p)
  (h_cycle : a^n.val + p * b = b^n.val + p * c ∧ b^n.val + p * c = c^n.val + p * a) :
  a = b ∧ b = c := by sorry

end NUMINAMATH_CALUDE_equality_of_cyclic_sum_powers_l1300_130058


namespace NUMINAMATH_CALUDE_product_of_divisors_equal_implies_equal_l1300_130003

/-- Product of divisors function -/
def p (x : ℤ) : ℤ := sorry

/-- Theorem: If the product of divisors of two integers are equal, then the integers are equal -/
theorem product_of_divisors_equal_implies_equal (m n : ℤ) : p m = p n → m = n := by
  sorry

end NUMINAMATH_CALUDE_product_of_divisors_equal_implies_equal_l1300_130003


namespace NUMINAMATH_CALUDE_chicken_difference_l1300_130034

/-- The number of Rhode Island Reds Susie has -/
def susie_rir : ℕ := 11

/-- The number of Golden Comets Susie has -/
def susie_gc : ℕ := 6

/-- The number of Rhode Island Reds Britney has -/
def britney_rir : ℕ := 2 * susie_rir

/-- The number of Golden Comets Britney has -/
def britney_gc : ℕ := susie_gc / 2

/-- The total number of chickens Susie has -/
def susie_total : ℕ := susie_rir + susie_gc

/-- The total number of chickens Britney has -/
def britney_total : ℕ := britney_rir + britney_gc

theorem chicken_difference : britney_total - susie_total = 8 := by
  sorry

end NUMINAMATH_CALUDE_chicken_difference_l1300_130034


namespace NUMINAMATH_CALUDE_p_and_q_true_l1300_130027

theorem p_and_q_true :
  (∃ x₀ : ℝ, x₀^2 < x₀) ∧ (∀ x : ℝ, x^2 - x + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_p_and_q_true_l1300_130027


namespace NUMINAMATH_CALUDE_letter_writing_is_permutation_problem_l1300_130095

/-- A function that represents the number of letters written when n people write to each other once -/
def letters_written (n : ℕ) : ℕ := n * (n - 1)

/-- A function that represents whether a scenario is a permutation problem -/
def is_permutation_problem (scenario : ℕ → ℕ) : Prop :=
  ∃ n : ℕ, n > 1 ∧ scenario n ≠ scenario (n - 1)

theorem letter_writing_is_permutation_problem :
  is_permutation_problem letters_written :=
sorry


end NUMINAMATH_CALUDE_letter_writing_is_permutation_problem_l1300_130095


namespace NUMINAMATH_CALUDE_unique_square_divisible_by_six_l1300_130083

theorem unique_square_divisible_by_six : ∃! x : ℕ,
  (∃ n : ℕ, x = n^2) ∧ 
  (∃ k : ℕ, x = 6 * k) ∧
  50 ≤ x ∧
  x ≤ 150 := by
  sorry

end NUMINAMATH_CALUDE_unique_square_divisible_by_six_l1300_130083


namespace NUMINAMATH_CALUDE_find_2a_plus_c_l1300_130013

theorem find_2a_plus_c (a b c : ℝ) 
  (eq1 : 3 * a + b + 2 * c = 3) 
  (eq2 : a + 3 * b + 2 * c = 1) : 
  2 * a + c = 2 := by sorry

end NUMINAMATH_CALUDE_find_2a_plus_c_l1300_130013


namespace NUMINAMATH_CALUDE_x_squared_gt_4_necessary_not_sufficient_for_x_gt_2_l1300_130078

theorem x_squared_gt_4_necessary_not_sufficient_for_x_gt_2 :
  (∀ x : ℝ, x > 2 → x^2 > 4) ∧ (∃ x : ℝ, x^2 > 4 ∧ x ≤ 2) := by sorry

end NUMINAMATH_CALUDE_x_squared_gt_4_necessary_not_sufficient_for_x_gt_2_l1300_130078


namespace NUMINAMATH_CALUDE_platform_length_l1300_130062

/-- Given a train crossing a platform and a signal pole, calculate the platform length -/
theorem platform_length 
  (train_length : ℝ) 
  (time_platform : ℝ) 
  (time_pole : ℝ) 
  (h1 : train_length = 300) 
  (h2 : time_platform = 54) 
  (h3 : time_pole = 18) : 
  ∃ platform_length : ℝ, platform_length = 600 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_l1300_130062


namespace NUMINAMATH_CALUDE_equation_has_two_distinct_real_roots_l1300_130093

/-- Custom multiplication operation -/
def star_op (a b : ℝ) := a^2 - a*b

/-- Theorem stating that the equation (x+1)*3 = -2 has two distinct real roots -/
theorem equation_has_two_distinct_real_roots :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ star_op (x₁ + 1) 3 = -2 ∧ star_op (x₂ + 1) 3 = -2 :=
sorry

end NUMINAMATH_CALUDE_equation_has_two_distinct_real_roots_l1300_130093


namespace NUMINAMATH_CALUDE_barangay_speed_l1300_130082

/-- Proves that the speed going to the barangay is 5 km/h given the problem conditions -/
theorem barangay_speed 
  (total_time : ℝ) 
  (distance : ℝ) 
  (rest_time : ℝ) 
  (return_speed : ℝ) 
  (h1 : total_time = 6)
  (h2 : distance = 7.5)
  (h3 : rest_time = 2)
  (h4 : return_speed = 3) : 
  distance / (total_time - rest_time - distance / return_speed) = 5 := by
sorry

end NUMINAMATH_CALUDE_barangay_speed_l1300_130082


namespace NUMINAMATH_CALUDE_train_carriages_l1300_130041

theorem train_carriages (initial_seats : ℕ) (additional_capacity : ℕ) (total_passengers : ℕ) (num_trains : ℕ) :
  initial_seats = 25 →
  additional_capacity = 10 →
  total_passengers = 420 →
  num_trains = 3 →
  (total_passengers / (num_trains * (initial_seats + additional_capacity))) = 4 :=
by sorry

end NUMINAMATH_CALUDE_train_carriages_l1300_130041


namespace NUMINAMATH_CALUDE_latus_rectum_equation_l1300_130048

/-- The equation of the latus rectum of the parabola y = -1/4 * x^2 -/
theorem latus_rectum_equation (x y : ℝ) :
  y = -1/4 * x^2 → (∃ (p : ℝ), p = -1/2 ∧ y = p) :=
by sorry

end NUMINAMATH_CALUDE_latus_rectum_equation_l1300_130048


namespace NUMINAMATH_CALUDE_sum_of_digits_N_l1300_130065

def N : ℕ := 9 + 99 + 999 + 9999 + 99999 + 999999

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_N : sum_of_digits N = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_N_l1300_130065


namespace NUMINAMATH_CALUDE_divisibility_theorem_l1300_130072

theorem divisibility_theorem (a b c : ℕ+) 
  (h1 : a ∣ b^5)
  (h2 : b ∣ c^5)
  (h3 : c ∣ a^5) :
  (a * b * c) ∣ (a + b + c)^31 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_theorem_l1300_130072


namespace NUMINAMATH_CALUDE_ratio_difference_problem_l1300_130006

theorem ratio_difference_problem (A B : ℚ) : 
  A / B = 3 / 5 → B - A = 12 → A = 18 := by
  sorry

end NUMINAMATH_CALUDE_ratio_difference_problem_l1300_130006


namespace NUMINAMATH_CALUDE_unique_solution_implies_prime_l1300_130004

theorem unique_solution_implies_prime (n : ℕ) :
  (∃! (x y : ℕ), (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / n) →
  Nat.Prime n :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_implies_prime_l1300_130004


namespace NUMINAMATH_CALUDE_inequality_multiplication_l1300_130000

theorem inequality_multiplication (a b c d : ℝ) :
  (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a > b ∧ c > d → a * c > b * d) ∧
  (a < 0 ∧ b < 0 ∧ c < 0 ∧ d < 0 ∧ a < b ∧ c < d → a * c > b * d) :=
sorry

end NUMINAMATH_CALUDE_inequality_multiplication_l1300_130000


namespace NUMINAMATH_CALUDE_correct_change_marys_change_l1300_130023

def change_calculation (cost_berries : ℚ) (cost_peaches : ℚ) (amount_paid : ℚ) : ℚ :=
  amount_paid - (cost_berries + cost_peaches)

theorem correct_change (cost_berries cost_peaches amount_paid : ℚ) :
  change_calculation cost_berries cost_peaches amount_paid =
  amount_paid - (cost_berries + cost_peaches) :=
by
  sorry

-- Example with Mary's specific values
theorem marys_change :
  change_calculation 7.19 6.83 20 = 5.98 :=
by
  sorry

end NUMINAMATH_CALUDE_correct_change_marys_change_l1300_130023


namespace NUMINAMATH_CALUDE_jack_buttons_theorem_l1300_130087

/-- The number of buttons Jack must use for all shirts -/
def total_buttons (num_kids : ℕ) (shirts_per_kid : ℕ) (buttons_per_shirt : ℕ) : ℕ :=
  num_kids * shirts_per_kid * buttons_per_shirt

/-- Theorem stating the total number of buttons Jack must use -/
theorem jack_buttons_theorem :
  total_buttons 3 3 7 = 63 := by
  sorry

end NUMINAMATH_CALUDE_jack_buttons_theorem_l1300_130087


namespace NUMINAMATH_CALUDE_range_of_a_l1300_130059

def f (a x : ℝ) : ℝ := a^2 * x - 2*a + 1

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ f a x ≤ 0) → a ≥ 1/2 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l1300_130059


namespace NUMINAMATH_CALUDE_factorial_fraction_equals_seven_l1300_130068

theorem factorial_fraction_equals_seven : (4 * Nat.factorial 7 + 28 * Nat.factorial 6) / Nat.factorial 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_equals_seven_l1300_130068


namespace NUMINAMATH_CALUDE_rahul_deepak_age_ratio_l1300_130036

theorem rahul_deepak_age_ratio : 
  ∀ (rahul_age deepak_age : ℕ),
  deepak_age = 3 →
  rahul_age + 22 = 26 →
  (rahul_age : ℚ) / (deepak_age : ℚ) = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_rahul_deepak_age_ratio_l1300_130036


namespace NUMINAMATH_CALUDE_photo_arrangement_count_l1300_130033

/-- Represents the number of people in the arrangement -/
def total_people : ℕ := 6

/-- Represents the number of students in the arrangement -/
def num_students : ℕ := 4

/-- Represents the number of teachers in the arrangement -/
def num_teachers : ℕ := 2

/-- Represents the number of students that must stand together -/
def students_together : ℕ := 2

/-- Calculates the number of ways to arrange the people given the constraints -/
def arrangement_count : ℕ :=
  (num_teachers.factorial) *    -- Ways to arrange teachers in the middle
  2 *                           -- Ways to place students A and B (left or right of teachers)
  (students_together.factorial) * -- Ways to arrange A and B within their unit
  ((num_students - students_together).factorial) -- Ways to arrange remaining students

theorem photo_arrangement_count :
  arrangement_count = 8 := by sorry

end NUMINAMATH_CALUDE_photo_arrangement_count_l1300_130033


namespace NUMINAMATH_CALUDE_equation_solution_l1300_130032

theorem equation_solution : ∃ x : ℝ, (x - 3)^4 = 16 ∧ x = 5 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1300_130032


namespace NUMINAMATH_CALUDE_min_value_theorem_l1300_130024

-- Define the quadratic inequality solution set condition
def solution_set (a b : ℝ) : Prop :=
  ∀ x, (a * x^2 + 2 * x + b > 0) ↔ (x ≠ -1/a)

-- Define the theorem
theorem min_value_theorem (a b : ℝ) (h1 : solution_set a b) (h2 : a > b) :
  ∃ min_val : ℝ, min_val = 2 * Real.sqrt 2 ∧
  ∀ x : ℝ, (a^2 + b^2) / (a - b) ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1300_130024


namespace NUMINAMATH_CALUDE_toyota_not_less_than_honda_skoda_combined_l1300_130053

/-- Proves that the number of Toyotas is not less than the number of Hondas and Skodas combined in a parking lot with specific conditions. -/
theorem toyota_not_less_than_honda_skoda_combined 
  (C T H S X Y : ℕ) 
  (h1 : C - H = (3 * (C - X)) / 2)
  (h2 : C - S = (3 * (C - Y)) / 2)
  (h3 : C - T = (X + Y) / 2) :
  T ≥ H + S := by sorry

end NUMINAMATH_CALUDE_toyota_not_less_than_honda_skoda_combined_l1300_130053


namespace NUMINAMATH_CALUDE_investment_scientific_notation_l1300_130066

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem investment_scientific_notation :
  toScientificNotation 909000000000 = ScientificNotation.mk 9.09 11 sorry := by
  sorry

end NUMINAMATH_CALUDE_investment_scientific_notation_l1300_130066


namespace NUMINAMATH_CALUDE_sum_of_specific_S_l1300_130038

def S (n : ℕ) : ℤ :=
  if n % 2 = 0 then -n else n + 1

theorem sum_of_specific_S : S 18 + S 34 + S 51 = 0 := by sorry

end NUMINAMATH_CALUDE_sum_of_specific_S_l1300_130038


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l1300_130098

theorem binomial_coefficient_equality (x : ℕ) : 
  (Nat.choose 14 x = Nat.choose 14 (2*x - 4)) → (x = 4 ∨ x = 6) :=
by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l1300_130098


namespace NUMINAMATH_CALUDE_unique_permutations_four_letter_two_pairs_is_six_l1300_130085

/-- The number of unique permutations of a four-letter word with two pairs of identical letters -/
def unique_permutations_four_letter_two_pairs : ℕ :=
  Nat.factorial 4 / (Nat.factorial 2 * Nat.factorial 2)

/-- Theorem: The number of unique permutations of a four-letter word with two pairs of identical letters is 6 -/
theorem unique_permutations_four_letter_two_pairs_is_six :
  unique_permutations_four_letter_two_pairs = 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_permutations_four_letter_two_pairs_is_six_l1300_130085


namespace NUMINAMATH_CALUDE_triangle_existence_condition_l1300_130084

def triangle_exists (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def valid_x_values : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}

theorem triangle_existence_condition (x : ℕ) :
  x > 0 → (triangle_exists 7 (x + 3) 10 ↔ x ∈ valid_x_values) := by sorry

end NUMINAMATH_CALUDE_triangle_existence_condition_l1300_130084


namespace NUMINAMATH_CALUDE_symmetric_points_fourth_quadrant_l1300_130040

/-- Given points A(a, 3) and B(2, b) are symmetric with respect to the x-axis,
    prove that point M(a, b) is in the fourth quadrant. -/
theorem symmetric_points_fourth_quadrant (a b : ℝ) :
  (a = 2 ∧ b = -3) →  -- Symmetry conditions
  a > 0 ∧ b < 0       -- Fourth quadrant conditions
  := by sorry

end NUMINAMATH_CALUDE_symmetric_points_fourth_quadrant_l1300_130040


namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l1300_130039

theorem arithmetic_evaluation : 
  -(18 / 3 * 11 - 48 / 4 + 5 * 9) = -99 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l1300_130039


namespace NUMINAMATH_CALUDE_zeros_not_adjacent_probability_l1300_130014

/-- The number of ones in the arrangement -/
def num_ones : ℕ := 3

/-- The number of zeros in the arrangement -/
def num_zeros : ℕ := 3

/-- The total number of digits in the arrangement -/
def total_digits : ℕ := num_ones + num_zeros

/-- The probability that the zeros are not adjacent when randomly arranged -/
def prob_zeros_not_adjacent : ℚ := 1 / 5

/-- Theorem stating that the probability of zeros not being adjacent is 1/5 -/
theorem zeros_not_adjacent_probability :
  prob_zeros_not_adjacent = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_zeros_not_adjacent_probability_l1300_130014


namespace NUMINAMATH_CALUDE_inequality_proof_l1300_130017

theorem inequality_proof (a b c d : ℝ) 
  (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hd : d ≥ 0)
  (hca : c + d ≤ a) (hcb : c + d ≤ b) : 
  a * d + b * c ≤ a * b := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1300_130017


namespace NUMINAMATH_CALUDE_binomial_probability_problem_l1300_130019

/-- A binomial distribution with n trials and probability p -/
def binomial_distribution (n : ℕ) (p : ℝ) : Type := Unit

/-- The probability of getting at least k successes in a binomial distribution -/
def P_at_least (dist : binomial_distribution n p) (k : ℕ) : ℝ := sorry

theorem binomial_probability_problem 
  (p : ℝ) 
  (ξ : binomial_distribution 2 p) 
  (η : binomial_distribution 4 p) 
  (h : P_at_least ξ 1 = 5/9) :
  P_at_least η 2 = 11/27 := by
  sorry

end NUMINAMATH_CALUDE_binomial_probability_problem_l1300_130019


namespace NUMINAMATH_CALUDE_cube_root_27_times_fourth_root_81_times_sqrt_9_l1300_130094

theorem cube_root_27_times_fourth_root_81_times_sqrt_9 :
  ∃ (a b c : ℝ), a^3 = 27 ∧ b^4 = 81 ∧ c^2 = 9 → a * b * c = 27 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_27_times_fourth_root_81_times_sqrt_9_l1300_130094


namespace NUMINAMATH_CALUDE_find_y_l1300_130071

theorem find_y (x : ℝ) (y : ℝ) (h1 : 3 * x = 0.75 * y) (h2 : x = 24) : y = 96 := by
  sorry

end NUMINAMATH_CALUDE_find_y_l1300_130071


namespace NUMINAMATH_CALUDE_basketball_game_scores_l1300_130091

/-- Represents the scores of a team in a basketball game -/
structure TeamScores :=
  (q1 q2 q3 q4 : ℕ)

/-- Checks if a sequence of four numbers is an arithmetic progression -/
def isArithmeticSequence (s : TeamScores) : Prop :=
  s.q2 - s.q1 = s.q3 - s.q2 ∧ s.q3 - s.q2 = s.q4 - s.q3 ∧ s.q2 > s.q1

/-- Checks if a sequence of four numbers is a geometric progression -/
def isGeometricSequence (s : TeamScores) : Prop :=
  ∃ r : ℚ, r > 1 ∧ s.q2 = s.q1 * r ∧ s.q3 = s.q2 * r ∧ s.q4 = s.q3 * r

/-- The main theorem statement -/
theorem basketball_game_scores 
  (falcons tigers : TeamScores) 
  (h1 : falcons.q1 = tigers.q1)
  (h2 : isArithmeticSequence falcons)
  (h3 : isGeometricSequence tigers)
  (h4 : falcons.q1 + falcons.q2 + falcons.q3 + falcons.q4 = 
        tigers.q1 + tigers.q2 + tigers.q3 + tigers.q4 + 2)
  (h5 : falcons.q1 + falcons.q2 + falcons.q3 + falcons.q4 ≤ 100)
  (h6 : tigers.q1 + tigers.q2 + tigers.q3 + tigers.q4 ≤ 100) :
  falcons.q1 + falcons.q2 + tigers.q1 + tigers.q2 = 14 :=
by
  sorry


end NUMINAMATH_CALUDE_basketball_game_scores_l1300_130091


namespace NUMINAMATH_CALUDE_soap_box_length_l1300_130001

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Proves that the length of each soap box is 7 inches -/
theorem soap_box_length 
  (carton : BoxDimensions)
  (soap : BoxDimensions)
  (h_carton_length : carton.length = 25)
  (h_carton_width : carton.width = 42)
  (h_carton_height : carton.height = 60)
  (h_soap_width : soap.width = 6)
  (h_soap_height : soap.height = 5)
  (h_max_boxes : ↑300 * boxVolume soap = boxVolume carton) :
  soap.length = 7 := by
  sorry

end NUMINAMATH_CALUDE_soap_box_length_l1300_130001


namespace NUMINAMATH_CALUDE_trip_length_l1300_130044

theorem trip_length (total : ℚ) 
  (h1 : (1 / 4 : ℚ) * total + 25 + (1 / 6 : ℚ) * total = total) :
  total = 300 / 7 := by
  sorry

end NUMINAMATH_CALUDE_trip_length_l1300_130044


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l1300_130074

/-- The hyperbola C with parameter m -/
def hyperbola (m : ℝ) (x y : ℝ) : Prop := x^2 / m - y^2 = 1

/-- The asymptote of the hyperbola C -/
def asymptote (m : ℝ) (x y : ℝ) : Prop := Real.sqrt 3 * x + m * y = 0

/-- The focal length of a hyperbola -/
def focal_length (m : ℝ) : ℝ := sorry

theorem hyperbola_focal_length (m : ℝ) (h1 : m > 0) :
  (∀ x y : ℝ, hyperbola m x y ↔ asymptote m x y) →
  focal_length m = 4 := by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l1300_130074


namespace NUMINAMATH_CALUDE_gas_tank_capacity_l1300_130076

/-- Represents the gas prices at each station -/
def gas_prices : List ℝ := [3, 3.5, 4, 4.5]

/-- Represents the total amount spent on gas -/
def total_spent : ℝ := 180

/-- Theorem: If a car owner fills up their tank 4 times at the given prices and spends $180 in total,
    then the gas tank capacity is 12 gallons -/
theorem gas_tank_capacity :
  ∀ (C : ℝ),
  (C > 0) →
  (List.sum (List.map (λ price => price * C) gas_prices) = total_spent) →
  C = 12 := by
  sorry

end NUMINAMATH_CALUDE_gas_tank_capacity_l1300_130076


namespace NUMINAMATH_CALUDE_math_book_cost_l1300_130043

-- Define the total number of books
def total_books : ℕ := 90

-- Define the number of math books
def math_books : ℕ := 53

-- Define the cost of each history book
def history_book_cost : ℕ := 5

-- Define the total price of all books
def total_price : ℕ := 397

-- Theorem to prove
theorem math_book_cost :
  ∃ (x : ℕ), x * math_books + (total_books - math_books) * history_book_cost = total_price ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_math_book_cost_l1300_130043


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l1300_130015

theorem binomial_expansion_coefficient (a : ℝ) : 
  (20 : ℝ) * a^3 = 160 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l1300_130015


namespace NUMINAMATH_CALUDE_power_calculation_l1300_130073

theorem power_calculation : 4^2009 * (-0.25)^2008 - 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l1300_130073


namespace NUMINAMATH_CALUDE_mAssignment_is_valid_l1300_130097

/-- Represents a variable in a programming language -/
structure Variable where
  name : String

/-- Represents an expression in a programming language -/
inductive Expression where
  | Var : Variable → Expression
  | Const : ℤ → Expression
  | Neg : Expression → Expression

/-- Represents an assignment statement -/
structure Assignment where
  lhs : Variable
  rhs : Expression

/-- Checks if an expression is valid on the right-hand side of an assignment -/
def isValidRHS : Expression → Bool
  | Expression.Var _ => true
  | Expression.Const _ => true
  | Expression.Neg e => isValidRHS e

/-- Checks if an assignment is valid according to programming rules -/
def isValidAssignment (a : Assignment) : Bool :=
  isValidRHS a.rhs

/-- The specific assignment M = -M -/
def mAssignment : Assignment :=
  { lhs := { name := "M" },
    rhs := Expression.Neg (Expression.Var { name := "M" }) }

/-- Theorem stating that M = -M is a valid assignment -/
theorem mAssignment_is_valid : isValidAssignment mAssignment := by sorry

end NUMINAMATH_CALUDE_mAssignment_is_valid_l1300_130097


namespace NUMINAMATH_CALUDE_largest_reciprocal_l1300_130067

theorem largest_reciprocal (a b c d e : ℚ) : 
  a = 1/4 → b = 3/7 → c = 2 → d = 7 → e = 1000 →
  (1/a > 1/b) ∧ (1/a > 1/c) ∧ (1/a > 1/d) ∧ (1/a > 1/e) :=
by
  sorry

#check largest_reciprocal

end NUMINAMATH_CALUDE_largest_reciprocal_l1300_130067


namespace NUMINAMATH_CALUDE_bill_equation_l1300_130086

/-- Represents the monthly telephone bill calculation -/
def monthly_bill (rental_fee : ℝ) (per_call_cost : ℝ) (num_calls : ℝ) : ℝ :=
  rental_fee + per_call_cost * num_calls

/-- Theorem stating the relationship between monthly bill and number of calls -/
theorem bill_equation (x : ℝ) :
  monthly_bill 10 0.2 x = 10 + 0.2 * x :=
by sorry

end NUMINAMATH_CALUDE_bill_equation_l1300_130086


namespace NUMINAMATH_CALUDE_halfway_fraction_l1300_130007

theorem halfway_fraction : (3 / 4 + 5 / 6) / 2 = 19 / 24 := by
  sorry

end NUMINAMATH_CALUDE_halfway_fraction_l1300_130007


namespace NUMINAMATH_CALUDE_two_burritos_five_quesadillas_cost_l1300_130009

/-- The price of a burrito in dollars -/
def burrito_price : ℝ := sorry

/-- The price of a quesadilla in dollars -/
def quesadilla_price : ℝ := sorry

/-- The condition that one burrito and four quesadillas cost $3.50 -/
axiom condition1 : burrito_price + 4 * quesadilla_price = 3.50

/-- The condition that four burritos and one quesadilla cost $4.10 -/
axiom condition2 : 4 * burrito_price + quesadilla_price = 4.10

/-- The theorem stating that two burritos and five quesadillas cost $5.02 -/
theorem two_burritos_five_quesadillas_cost :
  2 * burrito_price + 5 * quesadilla_price = 5.02 := by sorry

end NUMINAMATH_CALUDE_two_burritos_five_quesadillas_cost_l1300_130009
