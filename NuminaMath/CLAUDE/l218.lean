import Mathlib

namespace NUMINAMATH_CALUDE_jacket_cost_l218_21877

theorem jacket_cost (shorts_cost total_cost : ℚ) 
  (shorts_eq : shorts_cost = 14.28)
  (total_eq : total_cost = 19.02) :
  total_cost - shorts_cost = 4.74 := by
  sorry

end NUMINAMATH_CALUDE_jacket_cost_l218_21877


namespace NUMINAMATH_CALUDE_parallel_lines_m_equal_intercepts_equal_intercept_equations_l218_21894

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 2 * x - y - 3 = 0
def l₂ (m x y : ℝ) : Prop := x - m * y + 1 - 3 * m = 0

-- Part 1: Parallel lines
theorem parallel_lines_m (m : ℝ) : 
  (∀ x y : ℝ, l₁ x y ↔ l₂ m x y) → m = 1/2 :=
sorry

-- Part 2: Equal intercepts
theorem equal_intercepts :
  ∃ m : ℝ, m ≠ 0 ∧ 
  ((∃ y : ℝ, l₂ m 0 y) ∧ (∃ x : ℝ, l₂ m x 0)) ∧
  (∀ y : ℝ, l₂ m 0 y → y = 3 * m - 1) ∧
  (∀ x : ℝ, l₂ m x 0 → x = 3 * m - 1) →
  (m = -1 ∨ m = 1/3) :=
sorry

-- Final equations for l₂ with equal intercepts
theorem equal_intercept_equations (x y : ℝ) :
  (x + y + 4 = 0 ∨ 3 * x - y = 0) ↔
  (l₂ (-1) x y ∨ l₂ (1/3) x y) :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_m_equal_intercepts_equal_intercept_equations_l218_21894


namespace NUMINAMATH_CALUDE_power_calculation_l218_21839

theorem power_calculation : 7^3 - 5*(6^2) + 2^4 = 179 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l218_21839


namespace NUMINAMATH_CALUDE_common_tangents_count_l218_21800

-- Define the circles
def C1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y + 4 = 0

-- Define the number of common tangents
def num_common_tangents : ℕ := 2

-- Theorem statement
theorem common_tangents_count : 
  ∃ (n : ℕ), n = num_common_tangents ∧ 
  (∀ x y : ℝ, C1 x y ∨ C2 x y → n = 2) :=
sorry

end NUMINAMATH_CALUDE_common_tangents_count_l218_21800


namespace NUMINAMATH_CALUDE_distance_representation_l218_21884

theorem distance_representation (A B : ℝ) (hA : A = 3) (hB : B = -2) :
  |A - B| = |3 - (-2)| := by sorry

end NUMINAMATH_CALUDE_distance_representation_l218_21884


namespace NUMINAMATH_CALUDE_specific_systematic_sample_l218_21821

/-- Systematic sampling function -/
def systematicSample (totalItems : ℕ) (sampleSize : ℕ) (firstNumber : ℕ) (nthItem : ℕ) : ℕ :=
  let k := totalItems / sampleSize
  firstNumber + k * (nthItem - 1)

/-- Theorem for the specific systematic sampling problem -/
theorem specific_systematic_sample :
  systematicSample 1000 50 15 40 = 795 := by
  sorry

end NUMINAMATH_CALUDE_specific_systematic_sample_l218_21821


namespace NUMINAMATH_CALUDE_probability_higher_first_lower_second_l218_21861

def card_set : Finset ℕ := Finset.range 7

def favorable_outcomes : Finset (ℕ × ℕ) :=
  Finset.filter (fun p => p.1 > p.2) (card_set.product card_set)

theorem probability_higher_first_lower_second :
  (favorable_outcomes.card : ℚ) / (card_set.card * card_set.card) = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_probability_higher_first_lower_second_l218_21861


namespace NUMINAMATH_CALUDE_lost_weights_solution_l218_21811

/-- Represents the types of weights in the set -/
inductive WeightType
  | Light : WeightType  -- 43g
  | Medium : WeightType -- 57g
  | Heavy : WeightType  -- 70g

/-- The weight in grams for each type -/
def weight_value (w : WeightType) : ℕ :=
  match w with
  | WeightType.Light => 43
  | WeightType.Medium => 57
  | WeightType.Heavy => 70

/-- The total number of each type of weight initially -/
def initial_count : ℕ := sorry

/-- The total weight of all weights initially -/
def initial_total_weight : ℕ := initial_count * (weight_value WeightType.Light + weight_value WeightType.Medium + weight_value WeightType.Heavy)

/-- The remaining weight after some weights were lost -/
def remaining_weight : ℕ := 20172

/-- Represents the number of weights lost for each type -/
structure LostWeights :=
  (light : ℕ)
  (medium : ℕ)
  (heavy : ℕ)

/-- The total number of weights lost -/
def total_lost (lw : LostWeights) : ℕ := lw.light + lw.medium + lw.heavy

/-- The total weight of lost weights -/
def lost_weight (lw : LostWeights) : ℕ :=
  lw.light * weight_value WeightType.Light +
  lw.medium * weight_value WeightType.Medium +
  lw.heavy * weight_value WeightType.Heavy

/-- Theorem stating that the only solution is losing 4 weights of 57g each -/
theorem lost_weights_solution :
  ∃! lw : LostWeights,
    total_lost lw < 5 ∧
    initial_total_weight - lost_weight lw = remaining_weight ∧
    lw.light = 0 ∧ lw.medium = 4 ∧ lw.heavy = 0 :=
  sorry

end NUMINAMATH_CALUDE_lost_weights_solution_l218_21811


namespace NUMINAMATH_CALUDE_zero_in_interval_l218_21816

-- Define the function f(x) = 2x - 5
def f (x : ℝ) : ℝ := 2 * x - 5

-- State the theorem
theorem zero_in_interval :
  (∀ x y, x < y → f x < f y) →  -- f is monotonically increasing
  Continuous f →                -- f is continuous
  ∃ c ∈ Set.Ioo 2 3, f c = 0    -- there exists a c in (2, 3) such that f(c) = 0
:= by sorry

end NUMINAMATH_CALUDE_zero_in_interval_l218_21816


namespace NUMINAMATH_CALUDE_parabola_equation_and_range_l218_21848

/-- A parabola with equation y = x^2 - 2mx + m^2 - 1 -/
def Parabola (m : ℝ) (x y : ℝ) : Prop :=
  y = x^2 - 2*m*x + m^2 - 1

/-- The parabola intersects the y-axis at (0, 3) -/
def IntersectsYAxisAt3 (m : ℝ) : Prop :=
  Parabola m 0 3

/-- The vertex of the parabola is in the fourth quadrant -/
def VertexInFourthQuadrant (m : ℝ) : Prop :=
  let x_vertex := m  -- x-coordinate of vertex is m for this parabola
  let y_vertex := -1  -- y-coordinate of vertex is -1 for this parabola
  x_vertex > 0 ∧ y_vertex < 0

theorem parabola_equation_and_range (m : ℝ) 
  (h1 : IntersectsYAxisAt3 m) 
  (h2 : VertexInFourthQuadrant m) :
  (∀ x y, Parabola m x y ↔ y = x^2 - 4*x + 3) ∧
  (∀ x y, 0 ≤ x ∧ x ≤ 3 ∧ Parabola m x y → -1 ≤ y ∧ y ≤ 3) := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_and_range_l218_21848


namespace NUMINAMATH_CALUDE_circles_tangent_radii_product_l218_21898

/-- Given two circles in a plane with radii r₁ and r₂, and distance d between their centers,
    if their common external tangent has length 2017 and their common internal tangent has length 2009,
    then the product of their radii is 8052. -/
theorem circles_tangent_radii_product (r₁ r₂ d : ℝ) 
  (h_external : d^2 - (r₁ + r₂)^2 = 2017^2)
  (h_internal : d^2 - (r₁ - r₂)^2 = 2009^2) :
  r₁ * r₂ = 8052 := by
  sorry

end NUMINAMATH_CALUDE_circles_tangent_radii_product_l218_21898


namespace NUMINAMATH_CALUDE_total_blue_marbles_l218_21850

-- Define the number of marbles collected by each friend
def jenny_red : ℕ := 30
def jenny_blue : ℕ := 25
def mary_red : ℕ := 2 * jenny_red
def anie_red : ℕ := mary_red + 20
def anie_blue : ℕ := 2 * jenny_blue
def mary_blue : ℕ := anie_blue / 2

-- Theorem to prove
theorem total_blue_marbles :
  jenny_blue + mary_blue + anie_blue = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_blue_marbles_l218_21850


namespace NUMINAMATH_CALUDE_ship_arrangement_count_l218_21844

/-- The number of ways to select and arrange ships for tasks -/
def arrange_ships (destroyers frigates selected : ℕ) (tasks : ℕ) : ℕ :=
  (Nat.choose (destroyers + frigates) selected - Nat.choose frigates selected) * Nat.factorial tasks

/-- Theorem stating the correct number of arrangements -/
theorem ship_arrangement_count :
  arrange_ships 2 6 3 3 = 216 := by
  sorry

end NUMINAMATH_CALUDE_ship_arrangement_count_l218_21844


namespace NUMINAMATH_CALUDE_total_fruits_is_41_l218_21856

/-- The number of oranges Mike received -/
def mike_oranges : ℕ := 3

/-- The number of apples Matt received -/
def matt_apples : ℕ := 2 * mike_oranges

/-- The number of bananas Mark received -/
def mark_bananas : ℕ := mike_oranges + matt_apples

/-- The number of grapes Mary received -/
def mary_grapes : ℕ := mike_oranges + matt_apples + mark_bananas + 5

/-- The total number of fruits received by all four children -/
def total_fruits : ℕ := mike_oranges + matt_apples + mark_bananas + mary_grapes

theorem total_fruits_is_41 : total_fruits = 41 := by
  sorry

end NUMINAMATH_CALUDE_total_fruits_is_41_l218_21856


namespace NUMINAMATH_CALUDE_only_proposition_3_true_l218_21841

theorem only_proposition_3_true : 
  (¬∀ (a b c : ℝ), a > b ∧ c ≠ 0 → a * c > b * c) ∧ 
  (¬∀ (a b c : ℝ), a > b → a * c^2 > b * c^2) ∧ 
  (∀ (a b c : ℝ), a * c^2 > b * c^2 → a > b) ∧ 
  (¬∀ (a b : ℝ), a > b → 1 / a < 1 / b) ∧ 
  (¬∀ (a b c d : ℝ), a > b ∧ b > 0 ∧ c > d → a * c > b * d) :=
by sorry

end NUMINAMATH_CALUDE_only_proposition_3_true_l218_21841


namespace NUMINAMATH_CALUDE_jeans_cost_theorem_l218_21864

def total_cost : ℕ := 110
def coat_cost : ℕ := 40
def shoes_cost : ℕ := 30
def num_jeans : ℕ := 2

theorem jeans_cost_theorem :
  ∃ (jeans_cost : ℕ),
    jeans_cost * num_jeans + coat_cost + shoes_cost = total_cost ∧
    jeans_cost = 20 :=
by sorry

end NUMINAMATH_CALUDE_jeans_cost_theorem_l218_21864


namespace NUMINAMATH_CALUDE_factorization_sum_l218_21842

theorem factorization_sum (x y : ℝ) : 
  ∃ (a b c d e f g h j k : ℤ), 
    125 * x^9 - 216 * y^9 = (a*x + b*y) * (c*x^3 + d*x*y^2 + e*y^3) * (f*x + g*y) * (h*x^3 + j*x*y^2 + k*y^3) ∧
    a + b + c + d + e + f + g + h + j + k = 16 :=
by sorry

end NUMINAMATH_CALUDE_factorization_sum_l218_21842


namespace NUMINAMATH_CALUDE_benny_crayons_l218_21838

theorem benny_crayons (initial : ℕ) (final : ℕ) (added : ℕ) : 
  initial = 9 → final = 12 → added = final - initial → added = 3 := by
  sorry

end NUMINAMATH_CALUDE_benny_crayons_l218_21838


namespace NUMINAMATH_CALUDE_book_difference_proof_l218_21802

def old_town_books : ℕ := 750
def riverview_books : ℕ := 1240
def downtown_books : ℕ := 1800
def eastside_books : ℕ := 1620

def library_books : List ℕ := [old_town_books, riverview_books, downtown_books, eastside_books]

theorem book_difference_proof :
  (List.maximum library_books).get! - (List.minimum library_books).get! = 1050 := by
  sorry

end NUMINAMATH_CALUDE_book_difference_proof_l218_21802


namespace NUMINAMATH_CALUDE_find_number_multiplied_by_9999_l218_21801

theorem find_number_multiplied_by_9999 : ∃ x : ℕ, x * 9999 = 724797420 ∧ x = 72480 := by
  sorry

end NUMINAMATH_CALUDE_find_number_multiplied_by_9999_l218_21801


namespace NUMINAMATH_CALUDE_function_values_l218_21809

/-- The linear function f(x) = 2x - 3 -/
def f (x : ℝ) : ℝ := 2 * x - 3

theorem function_values :
  (f 4 = 5) ∧ (f (3/2) = 0) := by sorry

end NUMINAMATH_CALUDE_function_values_l218_21809


namespace NUMINAMATH_CALUDE_polynomial_simplification_l218_21834

theorem polynomial_simplification (s r : ℝ) :
  (2 * s^2 + 5 * r - 4) - (3 * s^2 + 9 * r - 7) = -s^2 - 4 * r + 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l218_21834


namespace NUMINAMATH_CALUDE_cubic_function_property_l218_21852

/-- Given a cubic function f(x) = ax³ + bx - 2 where f(2015) = 7, prove that f(-2015) = -11 -/
theorem cubic_function_property (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x^3 + b * x - 2
  f 2015 = 7 → f (-2015) = -11 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_property_l218_21852


namespace NUMINAMATH_CALUDE_solve_equation_l218_21832

/-- Proves that for x = 3.3333333333333335, the equation √((x * y) / 3) = x is satisfied when y = 10 -/
theorem solve_equation (x : ℝ) (y : ℝ) (h1 : x = 3.3333333333333335) (h2 : y = 10) :
  Real.sqrt ((x * y) / 3) = x := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l218_21832


namespace NUMINAMATH_CALUDE_max_toys_purchasable_l218_21836

theorem max_toys_purchasable (initial_amount : ℚ) (game_cost : ℚ) (book_cost : ℚ) (toy_cost : ℚ) :
  initial_amount = 57.45 →
  game_cost = 26.89 →
  book_cost = 12.37 →
  toy_cost = 6 →
  ⌊(initial_amount - game_cost - book_cost) / toy_cost⌋ = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_max_toys_purchasable_l218_21836


namespace NUMINAMATH_CALUDE_new_jasmine_percentage_l218_21833

/-- Calculates the new jasmine percentage in a solution after adding jasmine and water -/
theorem new_jasmine_percentage
  (initial_volume : ℝ)
  (initial_jasmine_percentage : ℝ)
  (added_jasmine : ℝ)
  (added_water : ℝ)
  (h1 : initial_volume = 90)
  (h2 : initial_jasmine_percentage = 5)
  (h3 : added_jasmine = 8)
  (h4 : added_water = 2) :
  let initial_jasmine := initial_volume * (initial_jasmine_percentage / 100)
  let new_jasmine := initial_jasmine + added_jasmine
  let new_volume := initial_volume + added_jasmine + added_water
  let new_percentage := (new_jasmine / new_volume) * 100
  new_percentage = 12.5 := by
sorry

end NUMINAMATH_CALUDE_new_jasmine_percentage_l218_21833


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_sum_l218_21890

-- Define the vectors a and b
def a (m n : ℝ) : Fin 3 → ℝ := ![2*m - 3, n + 2, 3]
def b (m n : ℝ) : Fin 3 → ℝ := ![2*m + 1, 3*n - 2, 6]

-- Define parallel vectors
def parallel (u v : Fin 3 → ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ (∀ i, v i = k * u i)

-- Theorem statement
theorem parallel_vectors_imply_sum (m n : ℝ) :
  parallel (a m n) (b m n) → 2*m + n = 13 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_sum_l218_21890


namespace NUMINAMATH_CALUDE_age_puzzle_solution_l218_21867

/-- Represents a person's age --/
structure Age :=
  (tens : Nat)
  (ones : Nat)
  (is_valid : tens ≤ 9 ∧ ones ≤ 9)

/-- The age after 10 years --/
def age_after_10_years (a : Age) : Nat :=
  10 * a.tens + a.ones + 10

/-- Helen's age is the reverse of Ellen's age --/
def is_reverse (helen : Age) (ellen : Age) : Prop :=
  helen.tens = ellen.ones ∧ helen.ones = ellen.tens

/-- In 10 years, Helen will be three times as old as Ellen --/
def future_age_relation (helen : Age) (ellen : Age) : Prop :=
  age_after_10_years helen = 3 * age_after_10_years ellen

/-- The current age difference --/
def age_difference (helen : Age) (ellen : Age) : Int :=
  (10 * helen.tens + helen.ones) - (10 * ellen.tens + ellen.ones)

theorem age_puzzle_solution :
  ∃ (helen ellen : Age),
    is_reverse helen ellen ∧
    future_age_relation helen ellen ∧
    age_difference helen ellen = 54 :=
  sorry

end NUMINAMATH_CALUDE_age_puzzle_solution_l218_21867


namespace NUMINAMATH_CALUDE_arctan_sum_special_case_l218_21820

theorem arctan_sum_special_case (a b : ℝ) : 
  a = 1/3 → (a + 1) * (b + 1) = 5/2 → Real.arctan a + Real.arctan b = Real.arctan (29/17) := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_special_case_l218_21820


namespace NUMINAMATH_CALUDE_blackboard_sum_divisibility_l218_21896

theorem blackboard_sum_divisibility (n : ℕ) (h_odd : Odd n) (h_gt_one : n > 1) :
  ∃ x ∈ Finset.range n, ¬ ∃ y ∈ Finset.range n,
    (((n * (3 * n - 1)) / 2 - (n + x)) % (n + y) = 0) := by
  sorry

end NUMINAMATH_CALUDE_blackboard_sum_divisibility_l218_21896


namespace NUMINAMATH_CALUDE_acute_triangles_in_cuboid_l218_21863

-- Define a rectangular cuboid
structure RectangularCuboid where
  vertices : Finset (ℝ × ℝ × ℝ)
  vertex_count : vertices.card = 8

-- Define a function to count acute triangles
def count_acute_triangles (rc : RectangularCuboid) : ℕ := sorry

-- Theorem statement
theorem acute_triangles_in_cuboid (rc : RectangularCuboid) :
  count_acute_triangles rc = 8 := by sorry

end NUMINAMATH_CALUDE_acute_triangles_in_cuboid_l218_21863


namespace NUMINAMATH_CALUDE_hunter_always_catches_grasshopper_l218_21807

/-- A point in the 2D integer plane -/
structure Point where
  x : Int
  y : Int

/-- The grasshopper's trajectory -/
structure Trajectory where
  start : Point
  jump : Point

/-- Spiral search function that returns the nth point in the spiral -/
def spiralSearch (n : Nat) : Point :=
  sorry

/-- Predicate to check if a point is on a trajectory at a given time -/
def onTrajectory (p : Point) (t : Trajectory) (time : Nat) : Prop :=
  p.x = t.start.x + t.jump.x * time ∧ p.y = t.start.y + t.jump.y * time

theorem hunter_always_catches_grasshopper :
  ∀ (t : Trajectory), ∃ (time : Nat), onTrajectory (spiralSearch time) t time :=
sorry

end NUMINAMATH_CALUDE_hunter_always_catches_grasshopper_l218_21807


namespace NUMINAMATH_CALUDE_inverse_proposition_l218_21830

theorem inverse_proposition (a b : ℝ) : 
  (a = 0 → a * b = 0) ↔ (a * b = 0 → a = 0) :=
sorry

end NUMINAMATH_CALUDE_inverse_proposition_l218_21830


namespace NUMINAMATH_CALUDE_oliver_workout_total_l218_21886

/-- Oliver's workout schedule over four days -/
def workout_schedule (monday tuesday wednesday thursday : ℕ) : Prop :=
  monday = 4 ∧ 
  tuesday = monday - 2 ∧ 
  wednesday = 2 * monday ∧ 
  thursday = 2 * tuesday

/-- The total workout hours over four days -/
def total_hours (monday tuesday wednesday thursday : ℕ) : ℕ :=
  monday + tuesday + wednesday + thursday

/-- Theorem: Given Oliver's workout schedule, the total hours worked out is 18 -/
theorem oliver_workout_total :
  ∀ (monday tuesday wednesday thursday : ℕ),
  workout_schedule monday tuesday wednesday thursday →
  total_hours monday tuesday wednesday thursday = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_oliver_workout_total_l218_21886


namespace NUMINAMATH_CALUDE_julio_has_more_soda_l218_21806

/-- Calculates the total liters of soda for a person given the number of orange and grape soda bottles -/
def totalSoda (orangeBottles grapeBottles : ℕ) : ℕ := 2 * (orangeBottles + grapeBottles)

theorem julio_has_more_soda : 
  let julioTotal := totalSoda 4 7
  let mateoTotal := totalSoda 1 3
  julioTotal - mateoTotal = 14 := by
  sorry

end NUMINAMATH_CALUDE_julio_has_more_soda_l218_21806


namespace NUMINAMATH_CALUDE_acute_angle_vector_range_l218_21895

def a (x : ℝ) : ℝ × ℝ := (x, 2)
def b : ℝ × ℝ := (-3, 6)

theorem acute_angle_vector_range (x : ℝ) :
  (∃ θ : ℝ, 0 < θ ∧ θ < π / 2 ∧ Real.cos θ = (a x).1 * b.1 + (a x).2 * b.2 / (Real.sqrt ((a x).1^2 + (a x).2^2) * Real.sqrt (b.1^2 + b.2^2))) →
  x < 4 ∧ x ≠ -1 :=
by sorry

end NUMINAMATH_CALUDE_acute_angle_vector_range_l218_21895


namespace NUMINAMATH_CALUDE_geometry_propositions_l218_21891

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the basic relations
variable (intersect : Plane → Plane → Line)
variable (contains : Plane → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (perpendicularPL : Plane → Line → Prop)
variable (perpendicularPP : Plane → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (parallelPL : Plane → Line → Prop)
variable (parallelPP : Plane → Plane → Prop)

-- Theorem statement
theorem geometry_propositions 
  (m n : Line) (α β : Plane) 
  (h_diff_lines : m ≠ n) 
  (h_diff_planes : α ≠ β) :
  (((intersect α β = m) ∧ (contains α n) ∧ (perpendicular n m)) → (perpendicularPP α β)) ∧
  ((perpendicularPL α m) ∧ (perpendicularPL β m) → (parallelPP α β)) ∧
  ((perpendicularPL α m) ∧ (perpendicularPL β n) ∧ (perpendicular m n) → (perpendicularPP α β)) ∧
  (∃ (m n : Line) (α β : Plane), (parallelPL α m) ∧ (parallelPL β n) ∧ (parallel m n) ∧ ¬(parallelPP α β)) :=
by sorry

end NUMINAMATH_CALUDE_geometry_propositions_l218_21891


namespace NUMINAMATH_CALUDE_a_squared_gt_b_squared_necessity_not_sufficiency_l218_21892

theorem a_squared_gt_b_squared_necessity_not_sufficiency (a b : ℝ) :
  (∀ a b : ℝ, a > |b| → a^2 > b^2) ∧
  (∃ a b : ℝ, a^2 > b^2 ∧ a ≤ |b|) :=
sorry

end NUMINAMATH_CALUDE_a_squared_gt_b_squared_necessity_not_sufficiency_l218_21892


namespace NUMINAMATH_CALUDE_max_rooms_needed_l218_21826

/-- Represents a group of football fans -/
structure FanGroup where
  team : Fin 3
  gender : Bool
  count : Nat

/-- The maximum number of fans that can be accommodated in one room -/
def max_fans_per_room : Nat := 3

/-- The total number of fans -/
def total_fans : Nat := 100

/-- Calculate the number of rooms needed for a group of fans -/
def rooms_needed (group : FanGroup) : Nat :=
  (group.count + max_fans_per_room - 1) / max_fans_per_room

/-- The main theorem stating the maximum number of rooms needed -/
theorem max_rooms_needed (fans : List FanGroup) 
  (h1 : fans.length = 6)
  (h2 : fans.all (λ g ↦ g.count > 0))
  (h3 : (fans.map FanGroup.count).sum = total_fans) :
  (fans.map rooms_needed).sum ≤ 37 := by
  sorry

end NUMINAMATH_CALUDE_max_rooms_needed_l218_21826


namespace NUMINAMATH_CALUDE_chairs_produced_in_six_hours_l218_21837

/-- The number of chairs produced by a group of workers over a given time period -/
def chairs_produced (num_workers : ℕ) (chairs_per_worker_per_hour : ℕ) (additional_chairs : ℕ) (hours : ℕ) : ℕ :=
  num_workers * chairs_per_worker_per_hour * hours + additional_chairs

/-- Theorem stating that 3 workers producing 4 chairs per hour, with an additional chair every 6 hours, produce 73 chairs in 6 hours -/
theorem chairs_produced_in_six_hours :
  chairs_produced 3 4 1 6 = 73 := by
  sorry

end NUMINAMATH_CALUDE_chairs_produced_in_six_hours_l218_21837


namespace NUMINAMATH_CALUDE_always_satisfies_condition_l218_21871

-- Define the set of colors
inductive Color
| Red
| Blue
| Green
| Yellow

-- Define a point with a color
structure ColoredPoint where
  color : Color

-- Define a colored line segment
structure ColoredSegment where
  endpoint1 : ColoredPoint
  endpoint2 : ColoredPoint
  color : Color

-- Define the coloring property for segments
def validSegmentColoring (segment : ColoredSegment) : Prop :=
  segment.color = segment.endpoint1.color ∨ segment.color = segment.endpoint2.color

-- Define the configuration of points and segments
structure Configuration where
  points : Fin 4 → ColoredPoint
  segments : Fin 6 → ColoredSegment
  allColorsUsed : ∀ c : Color, ∃ s : Fin 6, (segments s).color = c
  distinctPointColors : ∀ i j : Fin 4, i ≠ j → (points i).color ≠ (points j).color
  validSegments : ∀ s : Fin 6, validSegmentColoring (segments s)

-- Define the conditions to be satisfied
def satisfiesConditionA (config : Configuration) (p : Fin 4) : Prop :=
  ∃ s1 s2 s3 : Fin 6,
    (config.segments s1).color = Color.Red ∧
    (config.segments s2).color = Color.Blue ∧
    (config.segments s3).color = Color.Green ∧
    ((config.segments s1).endpoint1 = config.points p ∨ (config.segments s1).endpoint2 = config.points p) ∧
    ((config.segments s2).endpoint1 = config.points p ∨ (config.segments s2).endpoint2 = config.points p) ∧
    ((config.segments s3).endpoint1 = config.points p ∨ (config.segments s3).endpoint2 = config.points p)

def satisfiesConditionB (config : Configuration) (p : Fin 4) : Prop :=
  ∃ s1 s2 s3 : Fin 6,
    (config.segments s1).color = Color.Red ∧
    (config.segments s2).color = Color.Blue ∧
    (config.segments s3).color = Color.Green ∧
    (config.segments s1).endpoint1 ≠ config.points p ∧
    (config.segments s1).endpoint2 ≠ config.points p ∧
    (config.segments s2).endpoint1 ≠ config.points p ∧
    (config.segments s2).endpoint2 ≠ config.points p ∧
    (config.segments s3).endpoint1 ≠ config.points p ∧
    (config.segments s3).endpoint2 ≠ config.points p

-- The main theorem
theorem always_satisfies_condition (config : Configuration) :
  ∃ p : Fin 4, satisfiesConditionA config p ∨ satisfiesConditionB config p := by
  sorry

end NUMINAMATH_CALUDE_always_satisfies_condition_l218_21871


namespace NUMINAMATH_CALUDE_butane_molecular_weight_l218_21854

/-- The molecular weight of Butane in grams per mole. -/
def molecular_weight_butane : ℝ := 65

/-- The number of moles used in the given condition. -/
def given_moles : ℝ := 4

/-- The total molecular weight of the given moles in grams. -/
def given_total_weight : ℝ := 260

/-- Theorem stating that the molecular weight of Butane is 65 grams/mole. -/
theorem butane_molecular_weight : 
  molecular_weight_butane = given_total_weight / given_moles := by
  sorry

end NUMINAMATH_CALUDE_butane_molecular_weight_l218_21854


namespace NUMINAMATH_CALUDE_expand_product_l218_21882

theorem expand_product (x : ℝ) : 2 * (x - 3) * (x + 7) = 2 * x^2 + 8 * x - 42 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l218_21882


namespace NUMINAMATH_CALUDE_friend_balloon_count_l218_21825

theorem friend_balloon_count (my_balloons : ℕ) (difference : ℕ) : my_balloons = 7 → difference = 2 → my_balloons - difference = 5 := by
  sorry

end NUMINAMATH_CALUDE_friend_balloon_count_l218_21825


namespace NUMINAMATH_CALUDE_largest_inscribed_circle_radius_for_specific_quad_l218_21810

/-- A quadrilateral with given side lengths -/
structure Quadrilateral where
  ab : ℝ
  bc : ℝ
  cd : ℝ
  da : ℝ

/-- The radius of the largest inscribed circle in a quadrilateral -/
def largest_inscribed_circle_radius (q : Quadrilateral) : ℝ :=
  sorry

/-- Theorem stating the largest inscribed circle radius for a specific quadrilateral -/
theorem largest_inscribed_circle_radius_for_specific_quad :
  let q : Quadrilateral := ⟨15, 10, 8, 13⟩
  largest_inscribed_circle_radius q = 5.7 := by
  sorry

end NUMINAMATH_CALUDE_largest_inscribed_circle_radius_for_specific_quad_l218_21810


namespace NUMINAMATH_CALUDE_num_shortest_paths_A_to_B_l218_21860

/-- Represents a point on the grid --/
structure Point where
  x : ℕ
  y : ℕ

/-- Represents the road network --/
def RoadNetwork : Type := Set (Point × Point)

/-- Calculates the number of shortest paths between two points on a given road network --/
def numShortestPaths (start finish : Point) (network : RoadNetwork) : ℕ :=
  sorry

/-- The specific road network described in the problem --/
def specificNetwork : RoadNetwork :=
  sorry

/-- The start point A --/
def pointA : Point :=
  ⟨0, 0⟩

/-- The end point B --/
def pointB : Point :=
  ⟨11, 8⟩

/-- Theorem stating that the number of shortest paths from A to B on the specific network is 22023 --/
theorem num_shortest_paths_A_to_B :
  numShortestPaths pointA pointB specificNetwork = 22023 :=
by sorry

end NUMINAMATH_CALUDE_num_shortest_paths_A_to_B_l218_21860


namespace NUMINAMATH_CALUDE_inequality_implication_l218_21829

theorem inequality_implication (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a - d > b - c := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l218_21829


namespace NUMINAMATH_CALUDE_problem_solution_l218_21804

theorem problem_solution (p q : ℝ) 
  (h1 : 3 / p = 6)
  (h2 : 3 / q = 18)
  (h3 : abs (p - q - 0.33333333333333337) < 1e-14) :
  p = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l218_21804


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l218_21897

/-- A hyperbola with foci on the y-axis and asymptotes y = ±4x has eccentricity √17/4 -/
theorem hyperbola_eccentricity (a b : ℝ) (h1 : a = 4*b) : 
  let e := (Real.sqrt (a^2 + b^2)) / a
  e = Real.sqrt 17 / 4 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l218_21897


namespace NUMINAMATH_CALUDE_parent_selection_theorem_l218_21831

def total_parents : ℕ := 12
def num_couples : ℕ := 6
def parents_to_select : ℕ := 4

theorem parent_selection_theorem :
  let ways_to_select_couple := num_couples
  let remaining_parents := total_parents - 2
  let ways_to_select_others := (remaining_parents.choose (parents_to_select - 2))
  ways_to_select_couple * ways_to_select_others = 240 := by
  sorry

end NUMINAMATH_CALUDE_parent_selection_theorem_l218_21831


namespace NUMINAMATH_CALUDE_point_on_line_max_product_l218_21873

/-- Given points A(a,b) and B(4,c) lie on the line y = kx + 3, where k is a constant and k ≠ 0,
    and the maximum value of ab is 9, then c = 2. -/
theorem point_on_line_max_product (k a b c : ℝ) : 
  k ≠ 0 → 
  b = k * a + 3 → 
  c = k * 4 + 3 → 
  (∀ x y, x * y ≤ 9 → a * b ≥ x * y) → 
  c = 2 := by
sorry

end NUMINAMATH_CALUDE_point_on_line_max_product_l218_21873


namespace NUMINAMATH_CALUDE_second_pipe_fill_time_l218_21862

/-- Represents a system of pipes filling or draining a tank -/
structure PipeSystem where
  fill_time_1 : ℝ  -- Time for first pipe to fill the tank
  drain_time : ℝ   -- Time for drain pipe to empty the tank
  combined_time : ℝ -- Time to fill the tank with all pipes open
  fill_time_2 : ℝ  -- Time for second pipe to fill the tank (to be proven)

/-- The theorem stating the relationship between the pipes' fill times -/
theorem second_pipe_fill_time (ps : PipeSystem) 
  (h1 : ps.fill_time_1 = 5)
  (h2 : ps.drain_time = 20)
  (h3 : ps.combined_time = 2.5) : 
  ps.fill_time_2 = 4 := by
  sorry


end NUMINAMATH_CALUDE_second_pipe_fill_time_l218_21862


namespace NUMINAMATH_CALUDE_odd_number_probability_l218_21870

/-- The set of digits used to form the number -/
def digits : Finset Nat := {1, 4, 6, 9}

/-- The set of odd digits from the given set -/
def oddDigits : Finset Nat := {1, 9}

/-- The probability of forming an odd four-digit number -/
def probabilityOdd : ℚ := (oddDigits.card : ℚ) / (digits.card : ℚ)

/-- Theorem stating that the probability of forming an odd four-digit number is 1/2 -/
theorem odd_number_probability : probabilityOdd = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_odd_number_probability_l218_21870


namespace NUMINAMATH_CALUDE_sum_can_equal_fifty_l218_21845

theorem sum_can_equal_fifty : ∃ (scenario : Type) (sum : scenario → ℝ), ∀ (s : scenario), sum s = 50 := by
  sorry

end NUMINAMATH_CALUDE_sum_can_equal_fifty_l218_21845


namespace NUMINAMATH_CALUDE_add_neg_two_three_l218_21828

theorem add_neg_two_three : -2 + 3 = 1 := by sorry

end NUMINAMATH_CALUDE_add_neg_two_three_l218_21828


namespace NUMINAMATH_CALUDE_least_k_for_error_bound_l218_21817

def u : ℕ → ℚ
  | 0 => 1/8
  | n + 1 => 3 * u n - 3 * (u n)^2

def L : ℚ := 1/3

theorem least_k_for_error_bound :
  (∀ k < 9, |u k - L| > 1/2^500) ∧
  |u 9 - L| ≤ 1/2^500 := by
  sorry

end NUMINAMATH_CALUDE_least_k_for_error_bound_l218_21817


namespace NUMINAMATH_CALUDE_pencils_on_desk_l218_21858

theorem pencils_on_desk (drawer : ℕ) (added : ℕ) (total : ℕ) (initial : ℕ) : 
  drawer = 43 → added = 16 → total = 78 → initial + added + drawer = total → initial = 19 := by
sorry

end NUMINAMATH_CALUDE_pencils_on_desk_l218_21858


namespace NUMINAMATH_CALUDE_largest_power_of_five_l218_21847

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_of_factorials (n : ℕ) : ℕ := factorial n + factorial (n + 1) + factorial (n + 2)

theorem largest_power_of_five (n : ℕ) : 
  (∃ k : ℕ, sum_of_factorials 105 = 5^n * k) ∧ 
  (∀ m : ℕ, m > n → ¬∃ k : ℕ, sum_of_factorials 105 = 5^m * k) ↔ 
  n = 25 :=
sorry

end NUMINAMATH_CALUDE_largest_power_of_five_l218_21847


namespace NUMINAMATH_CALUDE_number_line_positions_l218_21872

theorem number_line_positions (x : ℝ) : 
  (x > 0 → (0 = -4*x + 1/4 * (12*x - (-4*x)) ∧ x = 0 + 1/4 * (4*x - 0))) ∧
  (x < 0 → (0 = 12*x + 3/4 * (-4*x - 12*x) ∧ x = 4*x + 3/4 * (0 - 4*x))) :=
by sorry

end NUMINAMATH_CALUDE_number_line_positions_l218_21872


namespace NUMINAMATH_CALUDE_cube_root_of_a_times_sqrt_a_l218_21855

theorem cube_root_of_a_times_sqrt_a (a : ℝ) (ha : a > 0) : 
  (a * a^(1/2))^(1/3) = a^(1/2) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_a_times_sqrt_a_l218_21855


namespace NUMINAMATH_CALUDE_fourth_sphere_radius_l218_21865

/-- A cone with four spheres inside, where three spheres have radius 3 and touch the base. -/
structure ConeFourSpheres where
  /-- Radius of the three identical spheres -/
  r₁ : ℝ
  /-- Radius of the fourth sphere -/
  r₂ : ℝ
  /-- Angle between the slant height and the base of the cone -/
  θ : ℝ
  /-- The three identical spheres touch the base of the cone -/
  touch_base : True
  /-- All spheres touch each other externally -/
  touch_externally : True
  /-- All spheres touch the lateral surface of the cone -/
  touch_lateral : True
  /-- The radius of the three identical spheres is 3 -/
  r₁_eq_3 : r₁ = 3
  /-- The angle between the slant height and the base of the cone is π/3 -/
  θ_eq_pi_div_3 : θ = π / 3

/-- The radius of the fourth sphere in the cone arrangement is 9 - 4√2 -/
theorem fourth_sphere_radius (c : ConeFourSpheres) : c.r₂ = 9 - 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_fourth_sphere_radius_l218_21865


namespace NUMINAMATH_CALUDE_highest_a_divisible_by_8_l218_21824

def is_divisible_by_8 (n : ℕ) : Prop := n % 8 = 0

def last_three_digits (n : ℕ) : ℕ := n % 1000

def construct_number (a : ℕ) : ℕ := 365000 + a * 100 + 16

theorem highest_a_divisible_by_8 :
  ∀ a : ℕ, a ≤ 9 →
    (is_divisible_by_8 (construct_number a) ↔ a ≤ 8) ∧
    (a = 8 → is_divisible_by_8 (construct_number a)) ∧
    (a = 9 → ¬is_divisible_by_8 (construct_number a)) :=
sorry

end NUMINAMATH_CALUDE_highest_a_divisible_by_8_l218_21824


namespace NUMINAMATH_CALUDE_triangle_length_l218_21835

-- Define the curve y = x^3
def curve (x : ℝ) : ℝ := x^3

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the conditions of the problem
structure ProblemConditions where
  triangle : Triangle
  on_curve : 
    curve triangle.A.1 = triangle.A.2 ∧
    curve triangle.B.1 = triangle.B.2 ∧
    curve triangle.C.1 = triangle.C.2
  A_at_origin : triangle.A = (0, 0)
  BC_parallel_x : triangle.B.2 = triangle.C.2
  area : ℝ

-- Define the theorem
theorem triangle_length (conditions : ProblemConditions) 
  (h : conditions.area = 125) : 
  let BC_length := |conditions.triangle.C.1 - conditions.triangle.B.1|
  BC_length = 10 := by sorry

end NUMINAMATH_CALUDE_triangle_length_l218_21835


namespace NUMINAMATH_CALUDE_x_range_l218_21814

theorem x_range (x : ℝ) : 
  (6 - 3 * x ≥ 0) ∧ (¬(1 / (x + 1) < 0)) → x ∈ Set.Icc (-1) 2 := by
sorry

end NUMINAMATH_CALUDE_x_range_l218_21814


namespace NUMINAMATH_CALUDE_arthur_dinner_cost_theorem_l218_21887

/-- Calculates the total cost of Arthur's dinner, including tips --/
def arthurDinnerCost (appetizer_cost dessert_cost entree_cost wine_cost : ℝ)
  (entree_discount appetizer_discount dessert_discount bill_discount tax_rate waiter_tip_rate busser_tip_rate : ℝ) : ℝ :=
  let discounted_entree := entree_cost * (1 - entree_discount)
  let subtotal := discounted_entree + 2 * wine_cost
  let discounted_subtotal := subtotal * (1 - bill_discount)
  let tax := discounted_subtotal * tax_rate
  let total_with_tax := discounted_subtotal + tax
  let original_cost := appetizer_cost + entree_cost + 2 * wine_cost + dessert_cost
  let original_with_tax := original_cost * (1 + tax_rate)
  let waiter_tip := original_with_tax * waiter_tip_rate
  let total_with_waiter_tip := total_with_tax + waiter_tip
  let busser_tip := total_with_waiter_tip * busser_tip_rate
  total_with_waiter_tip + busser_tip

/-- Theorem stating that Arthur's dinner cost is $38.556 --/
theorem arthur_dinner_cost_theorem :
  arthurDinnerCost 8 7 30 4 0.4 1 1 0.1 0.08 0.2 0.05 = 38.556 := by
  sorry


end NUMINAMATH_CALUDE_arthur_dinner_cost_theorem_l218_21887


namespace NUMINAMATH_CALUDE_remaining_watch_time_l218_21899

/-- Represents a time duration in hours and minutes -/
structure Duration where
  hours : ℕ
  minutes : ℕ

/-- Converts a Duration to minutes -/
def Duration.toMinutes (d : Duration) : ℕ :=
  d.hours * 60 + d.minutes

/-- The total duration of the series -/
def seriesDuration : Duration := { hours := 6, minutes := 0 }

/-- The durations of Hannah's watching periods -/
def watchingPeriods : List Duration := [
  { hours := 2, minutes := 24 },
  { hours := 1, minutes := 25 },
  { hours := 0, minutes := 55 }
]

/-- Theorem stating the remaining time to watch the series -/
theorem remaining_watch_time :
  seriesDuration.toMinutes - (watchingPeriods.map Duration.toMinutes).sum = 76 := by
  sorry

end NUMINAMATH_CALUDE_remaining_watch_time_l218_21899


namespace NUMINAMATH_CALUDE_base9_to_base3_conversion_l218_21823

/-- Converts a digit from base 9 to base 4 -/
def base9To4Digit (d : Nat) : Nat :=
  d / 4 * 10 + d % 4

/-- Converts a number from base 9 to base 4 -/
def base9To4 (n : Nat) : Nat :=
  let d1 := n / 81
  let d2 := (n / 9) % 9
  let d3 := n % 9
  base9To4Digit d1 * 10000 + base9To4Digit d2 * 100 + base9To4Digit d3

/-- Converts a digit from base 4 to base 3 -/
def base4To3Digit (d : Nat) : Nat :=
  d / 3 * 10 + d % 3

/-- Converts a number from base 4 to base 3 -/
def base4To3 (n : Nat) : Nat :=
  let d1 := n / 10000
  let d2 := (n / 100) % 100
  let d3 := n % 100
  base4To3Digit d1 * 100000000 + base4To3Digit d2 * 10000 + base4To3Digit d3

theorem base9_to_base3_conversion :
  base4To3 (base9To4 758) = 01101002000 := by
  sorry

end NUMINAMATH_CALUDE_base9_to_base3_conversion_l218_21823


namespace NUMINAMATH_CALUDE_farm_oxen_count_l218_21805

/-- Represents the daily fodder consumption of one buffalo -/
def B : ℝ := sorry

/-- Represents the number of oxen on the farm -/
def O : ℕ := sorry

/-- The total amount of fodder available on the farm -/
def total_fodder : ℝ := sorry

theorem farm_oxen_count : O = 8 := by
  have h1 : 3 * B = 4 * (3/4 * B) := sorry
  have h2 : 3 * B = 2 * (3/2 * B) := sorry
  have h3 : total_fodder = (33 * B + 3/2 * O * B) * 48 := sorry
  have h4 : total_fodder = (108 * B + 3/2 * O * B) * 18 := sorry
  sorry

end NUMINAMATH_CALUDE_farm_oxen_count_l218_21805


namespace NUMINAMATH_CALUDE_complement_A_complement_A_intersect_B_l218_21893

-- Define the universal set U
def U : Set ℝ := {x | x ≤ 4}

-- Define set A
def A : Set ℝ := {x | 2 * x + 4 < 0}

-- Define set B
def B : Set ℝ := {x | x^2 + 2*x - 3 ≤ 0}

-- Theorem for the complement of A with respect to U
theorem complement_A : (U \ A) = {x : ℝ | -2 ≤ x ∧ x ≤ 4} := by sorry

-- Theorem for the complement of (A ∩ B) with respect to U
theorem complement_A_intersect_B : (U \ (A ∩ B)) = {x : ℝ | x < -3 ∨ (-2 ≤ x ∧ x ≤ 4)} := by sorry

end NUMINAMATH_CALUDE_complement_A_complement_A_intersect_B_l218_21893


namespace NUMINAMATH_CALUDE_problem_2021_l218_21818

theorem problem_2021 : (2021^2 - 2020) / 2021 + 7 = 2027 := by
  sorry

end NUMINAMATH_CALUDE_problem_2021_l218_21818


namespace NUMINAMATH_CALUDE_triangle_movement_l218_21822

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Check if a triangle is isosceles and right-angled at C -/
def isIsoscelesRightTriangle (t : Triangle) : Prop :=
  (t.A.x - t.C.x)^2 + (t.A.y - t.C.y)^2 = (t.B.x - t.C.x)^2 + (t.B.y - t.C.y)^2 ∧
  (t.A.x - t.C.x) * (t.B.x - t.C.x) + (t.A.y - t.C.y) * (t.B.y - t.C.y) = 0

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The main theorem -/
theorem triangle_movement (t : Triangle) (a b : Line) (c : ℝ) :
  isIsoscelesRightTriangle t →
  (∀ (t' : Triangle), isIsoscelesRightTriangle t' →
    pointOnLine t'.A a → pointOnLine t'.B b →
    (t'.A.x - t'.B.x)^2 + (t'.A.y - t'.B.y)^2 = c^2) →
  ∃ (l : Line),
    (l.a = 1 ∧ l.b = 1 ∧ l.c = 0) ∨ (l.a = 1 ∧ l.b = -1 ∧ l.c = 0) ∧
    ∀ (p : Point), pointOnLine p l →
      -c/2 ≤ p.x ∧ p.x ≤ c/2 →
      ∃ (t' : Triangle), isIsoscelesRightTriangle t' ∧
        pointOnLine t'.A a ∧ pointOnLine t'.B b ∧
        t'.C = p :=
sorry

end NUMINAMATH_CALUDE_triangle_movement_l218_21822


namespace NUMINAMATH_CALUDE_range_of_a_l218_21885

theorem range_of_a (p q : Prop) (h_p : ∀ x : ℝ, x ∈ Set.Icc 0 1 → ∃ a : ℝ, a ≥ Real.exp x) 
  (h_q : ∃ (a : ℝ) (x : ℝ), x^2 + 4*x + a = 0) (h_pq : p ∧ q) :
  ∃ a : ℝ, a ∈ Set.Icc (Real.exp 1) 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l218_21885


namespace NUMINAMATH_CALUDE_composition_result_l218_21846

-- Define the functions f and g
def f (b : ℝ) (x : ℝ) : ℝ := 5 * x + b
def g (b : ℝ) (x : ℝ) : ℝ := b * x + 3

-- State the theorem
theorem composition_result (b e : ℝ) :
  (∀ x, f b (g b x) = 15 * x + e) → e = 18 := by
  sorry

end NUMINAMATH_CALUDE_composition_result_l218_21846


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l218_21883

/-- A quadratic function satisfying certain conditions -/
def f (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

/-- The theorem stating the properties of the quadratic function -/
theorem quadratic_function_properties :
  ∀ a b c : ℝ,
  (∀ x : ℝ, f a b c (x + 1) - f a b c x = 2 * x) →
  f a b c 0 = 1 →
  (∃ m : ℝ, 
    (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → f a b c x ≥ 2 * x + m) ∧
    (∀ m' : ℝ, m' > m → ∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ f a b c x < 2 * x + m')) →
  (∀ x : ℝ, f a b c x = x^2 - x + 1) ∧
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → f a b c x ≥ 2 * x + (-1)) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_function_properties_l218_21883


namespace NUMINAMATH_CALUDE_sam_march_aug_earnings_l218_21879

/-- Represents Sam's work and financial situation --/
structure SamFinances where
  hourly_rate : ℝ
  march_aug_hours : ℕ := 23
  sept_feb_hours : ℕ := 8
  additional_hours : ℕ := 16
  console_cost : ℕ := 600
  car_repair_cost : ℕ := 340

/-- Theorem stating Sam's earnings from March to August --/
theorem sam_march_aug_earnings (sam : SamFinances) :
  sam.hourly_rate * sam.march_aug_hours = 460 :=
by
  have total_needed : ℝ := sam.console_cost + sam.car_repair_cost
  have total_hours : ℕ := sam.march_aug_hours + sam.sept_feb_hours + sam.additional_hours
  have : sam.hourly_rate * total_hours = total_needed :=
    sorry
  sorry

#check sam_march_aug_earnings

end NUMINAMATH_CALUDE_sam_march_aug_earnings_l218_21879


namespace NUMINAMATH_CALUDE_box_volume_formula_l218_21869

/-- The volume of an open box formed from a rectangular cardboard sheet. -/
def box_volume (y : ℝ) : ℝ :=
  (20 - 2*y) * (12 - 2*y) * y

theorem box_volume_formula (y : ℝ) 
  (h : 0 < y ∧ y < 6) : -- y is positive and less than half the smaller dimension
  box_volume y = 4*y^3 - 64*y^2 + 240*y := by
  sorry

end NUMINAMATH_CALUDE_box_volume_formula_l218_21869


namespace NUMINAMATH_CALUDE_product_scaling_l218_21868

theorem product_scaling (a b c : ℝ) (h : (268 : ℝ) * 74 = 19732) :
  2.68 * 0.74 = 1.9732 := by
  sorry

end NUMINAMATH_CALUDE_product_scaling_l218_21868


namespace NUMINAMATH_CALUDE_halfway_fraction_l218_21875

theorem halfway_fraction : (3 / 4 + 5 / 6) / 2 = 19 / 24 := by sorry

end NUMINAMATH_CALUDE_halfway_fraction_l218_21875


namespace NUMINAMATH_CALUDE_remainder_of_123456789012_div_210_l218_21849

theorem remainder_of_123456789012_div_210 :
  123456789012 % 210 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_123456789012_div_210_l218_21849


namespace NUMINAMATH_CALUDE_fabric_cost_per_yard_l218_21803

theorem fabric_cost_per_yard 
  (total_spent : ℝ) 
  (total_yards : ℝ) 
  (h1 : total_spent = 120) 
  (h2 : total_yards = 16) : 
  total_spent / total_yards = 7.50 := by
sorry

end NUMINAMATH_CALUDE_fabric_cost_per_yard_l218_21803


namespace NUMINAMATH_CALUDE_inequality_proof_l218_21819

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a / (a^4 + b^2) + b / (a^2 + b^4) ≤ 1 / (a * b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l218_21819


namespace NUMINAMATH_CALUDE_two_digit_multiplication_l218_21878

theorem two_digit_multiplication (a b c : ℕ) : 
  (10 * a + b) * (10 * a + c) = 10 * a * (10 * a + c + b) + b * c := by
  sorry

end NUMINAMATH_CALUDE_two_digit_multiplication_l218_21878


namespace NUMINAMATH_CALUDE_max_value_sqrt_x2_y2_l218_21889

theorem max_value_sqrt_x2_y2 (x y : ℝ) (h : 3 * x^2 + 2 * y^2 = 6 * x) :
  ∃ (max : ℝ), max = 2 ∧ ∀ (x' y' : ℝ), 3 * x'^2 + 2 * y'^2 = 6 * x' → Real.sqrt (x'^2 + y'^2) ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_x2_y2_l218_21889


namespace NUMINAMATH_CALUDE_abs_two_x_minus_one_lt_one_l218_21866

theorem abs_two_x_minus_one_lt_one (x y : ℝ) 
  (h1 : |x - y - 1| ≤ 1/3) 
  (h2 : |2*y + 1| ≤ 1/6) : 
  |2*x - 1| < 1 := by
sorry

end NUMINAMATH_CALUDE_abs_two_x_minus_one_lt_one_l218_21866


namespace NUMINAMATH_CALUDE_union_of_sets_l218_21851

theorem union_of_sets : 
  let P : Set ℕ := {1, 2, 3, 4}
  let Q : Set ℕ := {3, 4, 5, 6}
  P ∪ Q = {1, 2, 3, 4, 5, 6} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l218_21851


namespace NUMINAMATH_CALUDE_fixed_salary_is_1000_l218_21888

/-- Represents the earnings structure and goal of a sales executive -/
structure SalesExecutive where
  commissionRate : Float
  targetEarnings : Float
  targetSales : Float

/-- Calculates the fixed salary for a sales executive -/
def calculateFixedSalary (exec : SalesExecutive) : Float :=
  exec.targetEarnings - exec.commissionRate * exec.targetSales

/-- Theorem: The fixed salary for the given sales executive is $1000 -/
theorem fixed_salary_is_1000 :
  let exec : SalesExecutive := {
    commissionRate := 0.05,
    targetEarnings := 5000,
    targetSales := 80000
  }
  calculateFixedSalary exec = 1000 := by
  sorry

#eval calculateFixedSalary {
  commissionRate := 0.05,
  targetEarnings := 5000,
  targetSales := 80000
}

end NUMINAMATH_CALUDE_fixed_salary_is_1000_l218_21888


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l218_21843

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := (2 + Complex.I) / Complex.I
  Complex.im z = -2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l218_21843


namespace NUMINAMATH_CALUDE_average_monthly_sales_l218_21827

def january_sales : ℝ := 150
def february_sales : ℝ := 90
def march_sales : ℝ := 60
def april_sales : ℝ := 140
def may_sales_before_discount : ℝ := 100
def discount_rate : ℝ := 0.2

def may_sales : ℝ := may_sales_before_discount * (1 - discount_rate)

def total_sales : ℝ := january_sales + february_sales + march_sales + april_sales + may_sales

def number_of_months : ℕ := 5

theorem average_monthly_sales :
  total_sales / number_of_months = 104 := by sorry

end NUMINAMATH_CALUDE_average_monthly_sales_l218_21827


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l218_21874

/-- A geometric sequence with specific properties -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 1 + a 2 = 1) →
  (a 3 + a 4 = 2) →
  (a 9 + a 10 = 16) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l218_21874


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l218_21840

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_a2 : a 2 = 2)
  (h_a4 : a 4 = 4)
  (h_b : ∀ n, b n = 2^(a n)) :
  (∀ n, a n = n) ∧ (b 1 + b 2 + b 3 + b 4 + b 5 = 62) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l218_21840


namespace NUMINAMATH_CALUDE_grid_arithmetic_sequences_l218_21853

/-- Given a 7x1 grid of numbers with two additional columns of length 3 and 5,
    prove that the value M satisfies the arithmetic sequence properties. -/
theorem grid_arithmetic_sequences (a : ℤ) (b c : ℚ) (M : ℚ) : 
  a = 25 ∧ 
  b = 16 ∧ 
  c = 20 ∧ 
  (∀ i : Fin 7, ∃ d : ℚ, a + i.val * d = a + 6 * d) ∧  -- row is arithmetic
  (∀ j : Fin 3, ∃ e : ℚ, a + j.val * e = b) ∧  -- first column is arithmetic
  (∀ k : Fin 5, ∃ f : ℚ, M + k.val * f = -20) ∧  -- second column is arithmetic
  (a + 3 * (b - a) / 3 = b) ∧  -- 4th element in row equals top of middle column
  (a + 6 * (M - a) / 6 = M) →  -- last element in row equals top of right column
  M = -6.25 := by
sorry

end NUMINAMATH_CALUDE_grid_arithmetic_sequences_l218_21853


namespace NUMINAMATH_CALUDE_grape_juice_percentage_l218_21880

-- Define the initial mixture volume
def initial_volume : ℝ := 30

-- Define the initial percentage of grape juice
def initial_grape_percentage : ℝ := 0.1

-- Define the volume of grape juice added
def added_grape_volume : ℝ := 10

-- Define the resulting percentage of grape juice
def resulting_grape_percentage : ℝ := 0.325

theorem grape_juice_percentage :
  let initial_grape_volume := initial_volume * initial_grape_percentage
  let total_grape_volume := initial_grape_volume + added_grape_volume
  let final_volume := initial_volume + added_grape_volume
  (total_grape_volume / final_volume) = resulting_grape_percentage := by
sorry

end NUMINAMATH_CALUDE_grape_juice_percentage_l218_21880


namespace NUMINAMATH_CALUDE_hole_pattern_symmetry_l218_21857

/-- Represents a rectangular piece of paper --/
structure Paper where
  length : ℝ
  width : ℝ

/-- Represents a point on the paper --/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a fold operation --/
inductive Fold
  | LeftToRight
  | TopToBottom
  | Diagonal

/-- Represents the hole pattern after unfolding --/
inductive HolePattern
  | SymmetricAll
  | SingleCenter
  | VerticalOnly
  | HorizontalOnly

/-- Performs a series of folds on the paper --/
def foldPaper (p : Paper) (folds : List Fold) : Paper :=
  sorry

/-- Punches a hole at a specific location on the folded paper --/
def punchHole (p : Paper) (loc : Point) : Paper :=
  sorry

/-- Unfolds the paper and determines the resulting hole pattern --/
def unfoldAndAnalyze (p : Paper) : HolePattern :=
  sorry

/-- Main theorem: The hole pattern is symmetrical across all axes --/
theorem hole_pattern_symmetry 
  (initialPaper : Paper)
  (folds : List Fold)
  (holeLocation : Point) :
  initialPaper.length = 8 ∧ 
  initialPaper.width = 4 ∧
  folds = [Fold.LeftToRight, Fold.TopToBottom, Fold.Diagonal] ∧
  holeLocation.x = 1/4 ∧ 
  holeLocation.y = 3/4 →
  unfoldAndAnalyze (punchHole (foldPaper initialPaper folds) holeLocation) = HolePattern.SymmetricAll :=
by
  sorry

end NUMINAMATH_CALUDE_hole_pattern_symmetry_l218_21857


namespace NUMINAMATH_CALUDE_square_value_l218_21859

theorem square_value (r : ℝ) (h1 : r + r = 75) (h2 : (r + r) + 2 * r = 143) : r = 41 := by
  sorry

end NUMINAMATH_CALUDE_square_value_l218_21859


namespace NUMINAMATH_CALUDE_complement_of_M_l218_21808

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define the set M
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 - 1}

-- State the theorem
theorem complement_of_M : 
  Set.compl M = {y : ℝ | y < -1} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l218_21808


namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocal_sum_l218_21815

theorem smallest_sum_of_reciprocal_sum (x y : ℕ+) 
  (h1 : x ≠ y) 
  (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 10) : 
  (∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 10 → (x : ℤ) + y ≤ (a : ℤ) + b) ∧ 
  ∃ p q : ℕ+, p ≠ q ∧ (1 : ℚ) / p + (1 : ℚ) / q = (1 : ℚ) / 10 ∧ (p : ℤ) + q = 45 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocal_sum_l218_21815


namespace NUMINAMATH_CALUDE_hillary_saturday_reading_l218_21876

/-- Calculates the number of minutes read on Saturday given the total assignment time and time read on Friday and Sunday. -/
def minutes_read_saturday (total_assignment : ℕ) (friday_reading : ℕ) (sunday_reading : ℕ) : ℕ :=
  total_assignment - (friday_reading + sunday_reading)

/-- Theorem stating that given the specific conditions of Hillary's reading assignment, she read for 28 minutes on Saturday. -/
theorem hillary_saturday_reading :
  minutes_read_saturday 60 16 16 = 28 := by
  sorry

end NUMINAMATH_CALUDE_hillary_saturday_reading_l218_21876


namespace NUMINAMATH_CALUDE_grocery_to_gym_speed_l218_21812

/-- Represents Angelina's walking scenario --/
structure WalkingScenario where
  initial_distance : ℝ
  second_distance : ℝ
  initial_speed : ℝ
  second_speed : ℝ
  third_speed : ℝ
  first_second_time_diff : ℝ
  second_third_time_diff : ℝ

/-- Theorem stating the speed from grocery to gym --/
theorem grocery_to_gym_speed (w : WalkingScenario) 
  (h1 : w.initial_distance = 100)
  (h2 : w.second_distance = 180)
  (h3 : w.second_speed = 2 * w.initial_speed)
  (h4 : w.third_speed = 3 * w.initial_speed)
  (h5 : w.initial_distance / w.initial_speed - w.second_distance / w.second_speed = w.first_second_time_diff)
  (h6 : w.first_second_time_diff = 40)
  (h7 : w.second_third_time_diff = 20) :
  w.second_speed = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_grocery_to_gym_speed_l218_21812


namespace NUMINAMATH_CALUDE_remainder_problem_l218_21881

theorem remainder_problem (n : ℕ) 
  (h1 : n^2 % 5 = 4)
  (h2 : n^3 % 5 = 2) : 
  n % 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l218_21881


namespace NUMINAMATH_CALUDE_min_value_expression_lower_bound_achievable_l218_21813

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 4) :
  ((x + 1) * (2*y + 1)) / (x * y) ≥ 9/2 :=
sorry

theorem lower_bound_achievable :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + 2*y = 4 ∧ ((x + 1) * (2*y + 1)) / (x * y) = 9/2 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_lower_bound_achievable_l218_21813
