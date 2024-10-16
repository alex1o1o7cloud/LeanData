import Mathlib

namespace NUMINAMATH_CALUDE_two_digit_perfect_square_conditions_l2331_233105

theorem two_digit_perfect_square_conditions : ∃! n : ℕ, 
  10 ≤ n ∧ n ≤ 99 ∧ 
  (∃ m : ℕ, 2 * n + 1 = m * m) ∧ 
  (∃ k : ℕ, 3 * n + 1 = k * k) ∧ 
  n = 40 := by
sorry

end NUMINAMATH_CALUDE_two_digit_perfect_square_conditions_l2331_233105


namespace NUMINAMATH_CALUDE_complementary_angles_imply_right_triangle_l2331_233125

-- Define a triangle
structure Triangle where
  angles : Fin 3 → ℝ
  sum_180 : angles 0 + angles 1 + angles 2 = 180

-- Define complementary angles
def complementary (a b : ℝ) : Prop := a + b = 90

-- Define a right triangle
def is_right_triangle (t : Triangle) : Prop :=
  ∃ i : Fin 3, t.angles i = 90

-- Theorem statement
theorem complementary_angles_imply_right_triangle (t : Triangle) 
  (h : ∃ i j : Fin 3, i ≠ j ∧ complementary (t.angles i) (t.angles j)) : 
  is_right_triangle t := by
  sorry

end NUMINAMATH_CALUDE_complementary_angles_imply_right_triangle_l2331_233125


namespace NUMINAMATH_CALUDE_no_savings_on_joint_purchase_l2331_233177

/-- Calculates the cost of windows under a "buy 3, get 1 free" offer -/
def windowCost (regularPrice : ℕ) (quantity : ℕ) : ℕ :=
  ((quantity + 3) / 4 * 3) * regularPrice

/-- Proves that buying windows together does not save money under the given offer -/
theorem no_savings_on_joint_purchase (regularPrice : ℕ) :
  windowCost regularPrice 19 = windowCost regularPrice 9 + windowCost regularPrice 10 :=
by sorry

end NUMINAMATH_CALUDE_no_savings_on_joint_purchase_l2331_233177


namespace NUMINAMATH_CALUDE_experienced_sailors_monthly_earnings_l2331_233126

/-- Calculate the total combined monthly earnings of experienced sailors --/
theorem experienced_sailors_monthly_earnings
  (total_sailors : ℕ)
  (inexperienced_sailors : ℕ)
  (inexperienced_hourly_wage : ℚ)
  (weekly_hours : ℕ)
  (weeks_per_month : ℕ)
  (h_total : total_sailors = 17)
  (h_inexperienced : inexperienced_sailors = 5)
  (h_wage : inexperienced_hourly_wage = 10)
  (h_hours : weekly_hours = 60)
  (h_weeks : weeks_per_month = 4) :
  let experienced_sailors := total_sailors - inexperienced_sailors
  let experienced_hourly_wage := inexperienced_hourly_wage * (1 + 1/5)
  let weekly_earnings := experienced_hourly_wage * weekly_hours
  let total_monthly_earnings := weekly_earnings * experienced_sailors * weeks_per_month
  total_monthly_earnings = 34560 :=
by sorry

end NUMINAMATH_CALUDE_experienced_sailors_monthly_earnings_l2331_233126


namespace NUMINAMATH_CALUDE_sum_of_smallest_and_largest_l2331_233148

-- Define the property of three consecutive even numbers
def ConsecutiveEvenNumbers (a b c : ℕ) : Prop :=
  ∃ n : ℕ, a = 2 * n ∧ b = 2 * n + 2 ∧ c = 2 * n + 4

theorem sum_of_smallest_and_largest (a b c : ℕ) :
  ConsecutiveEvenNumbers a b c → a + b + c = 1194 → a + c = 796 := by
  sorry

#check sum_of_smallest_and_largest

end NUMINAMATH_CALUDE_sum_of_smallest_and_largest_l2331_233148


namespace NUMINAMATH_CALUDE_correct_calculation_l2331_233191

theorem correct_calculation (x y : ℝ) : 3 * x - (-2 * y + 4) = 3 * x + 2 * y - 4 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2331_233191


namespace NUMINAMATH_CALUDE_uruguay_goals_conceded_l2331_233108

theorem uruguay_goals_conceded : 
  ∀ (x : ℕ), 
  (5 + 5 + 4 + 0 = 2 + 4 + x + 3) → 
  x = 5 := by
sorry

end NUMINAMATH_CALUDE_uruguay_goals_conceded_l2331_233108


namespace NUMINAMATH_CALUDE_triangle_property_l2331_233171

-- Define a triangle ABC
structure Triangle :=
  (A B C : Real)  -- angles
  (a b c : Real)  -- sides

-- Define the theorem
theorem triangle_property (t : Triangle) 
  (h : t.a^2 + t.b^2 = 2018 * t.c^2) :
  (2 * Real.sin t.A * Real.sin t.B * Real.cos t.C) / (1 - Real.cos t.C ^ 2) = 2017 := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l2331_233171


namespace NUMINAMATH_CALUDE_minuend_not_integer_l2331_233197

theorem minuend_not_integer (M S : ℝ) : M + S + (M - S) = 555 → ¬(∃ n : ℤ, M = n) := by
  sorry

end NUMINAMATH_CALUDE_minuend_not_integer_l2331_233197


namespace NUMINAMATH_CALUDE_evaluate_expression_l2331_233162

theorem evaluate_expression (x z : ℝ) (hx : x = 5) (hz : z = 4) :
  z^2 * (z^2 - 4*x) = -64 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2331_233162


namespace NUMINAMATH_CALUDE_exponential_graph_not_in_second_quadrant_l2331_233185

/-- Given a > 1 and b < -1, the graph of y = a^x + b does not intersect the second quadrant -/
theorem exponential_graph_not_in_second_quadrant 
  (a b : ℝ) (ha : a > 1) (hb : b < -1) :
  ∀ x y : ℝ, y = a^x + b → ¬(x < 0 ∧ y > 0) :=
by sorry

end NUMINAMATH_CALUDE_exponential_graph_not_in_second_quadrant_l2331_233185


namespace NUMINAMATH_CALUDE_sin_cos_inequality_implies_obtuse_l2331_233100

-- Define a triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  angle_sum : A + B + C = Real.pi

-- Define the property of being an obtuse triangle
def is_obtuse_triangle (t : Triangle) : Prop :=
  t.A > Real.pi / 2 ∨ t.B > Real.pi / 2 ∨ t.C > Real.pi / 2

-- Theorem statement
theorem sin_cos_inequality_implies_obtuse (t : Triangle) 
  (h : Real.sin t.A * Real.sin t.B < Real.cos t.A * Real.cos t.B) : 
  is_obtuse_triangle t := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_inequality_implies_obtuse_l2331_233100


namespace NUMINAMATH_CALUDE_triangle_inequality_l2331_233131

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  a + b > c ∧ b + c > a ∧ c + a > b :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2331_233131


namespace NUMINAMATH_CALUDE_k_lower_bound_l2331_233101

/-- Piecewise function f(x) -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then Real.log x else k * x

/-- Theorem stating the lower bound of k -/
theorem k_lower_bound (k : ℝ) :
  (∃ x₀ : ℝ, f k (-x₀) = f k x₀) → k ≥ -Real.exp (-1) :=
by sorry

end NUMINAMATH_CALUDE_k_lower_bound_l2331_233101


namespace NUMINAMATH_CALUDE_table_chair_price_ratio_l2331_233146

/-- The price ratio of tables to chairs in a store -/
theorem table_chair_price_ratio :
  ∀ (chair_price table_price : ℝ),
  chair_price > 0 →
  table_price > 0 →
  2 * chair_price + table_price = 0.6 * (chair_price + 2 * table_price) →
  table_price = 7 * chair_price :=
by
  sorry

end NUMINAMATH_CALUDE_table_chair_price_ratio_l2331_233146


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_ratio_l2331_233106

/-- Given a geometric sequence {a_n} with positive terms and common ratio q,
    if 3a_1, (1/2)a_3, and 2a_2 form an arithmetic sequence, then q = 3. -/
theorem geometric_arithmetic_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →  -- All terms are positive
  (∀ n, a (n + 1) = q * a n) →  -- Geometric sequence with ratio q
  (3 * a 1 - (1/2) * a 3) = ((1/2) * a 3 - 2 * a 2) →  -- Arithmetic sequence condition
  q = 3 :=
by sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_ratio_l2331_233106


namespace NUMINAMATH_CALUDE_lowest_degree_is_four_l2331_233137

/-- A polynomial with coefficients in ℤ -/
def IntPolynomial := Polynomial ℤ

/-- The set of coefficients of a polynomial -/
def coefficientSet (P : IntPolynomial) : Set ℤ :=
  {a : ℤ | ∃ (i : ℕ), a = P.coeff i}

/-- The property that a polynomial satisfies the given conditions -/
def satisfiesCondition (P : IntPolynomial) : Prop :=
  ∃ (b : ℤ), 
    (∃ (x y : ℤ), x ∈ coefficientSet P ∧ y ∈ coefficientSet P ∧ x < b ∧ b < y) ∧
    b ∉ coefficientSet P

/-- The theorem stating that the lowest degree of a polynomial satisfying the condition is 4 -/
theorem lowest_degree_is_four :
  ∃ (P : IntPolynomial), satisfiesCondition P ∧ P.degree = 4 ∧
  ∀ (Q : IntPolynomial), satisfiesCondition Q → Q.degree ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_lowest_degree_is_four_l2331_233137


namespace NUMINAMATH_CALUDE_midpoint_of_fractions_l2331_233147

theorem midpoint_of_fractions :
  (1 / 6 + 1 / 12) / 2 = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_of_fractions_l2331_233147


namespace NUMINAMATH_CALUDE_lemonade_percentage_in_solution1_l2331_233192

/-- Represents a solution mixture of lemonade and carbonated water -/
structure Solution :=
  (lemonade : ℝ)
  (carbonated_water : ℝ)
  (h_sum : lemonade + carbonated_water = 100)

/-- Represents a mixture of two solutions -/
structure Mixture :=
  (solution1 : Solution)
  (solution2 : Solution)
  (proportion1 : ℝ)
  (proportion2 : ℝ)
  (h_prop_sum : proportion1 + proportion2 = 100)

theorem lemonade_percentage_in_solution1
  (s1 : Solution)
  (s2 : Solution)
  (mix : Mixture)
  (h1 : s2.lemonade = 45)
  (h2 : s2.carbonated_water = 55)
  (h3 : mix.solution1 = s1)
  (h4 : mix.solution2 = s2)
  (h5 : mix.proportion1 = 40)
  (h6 : mix.proportion2 = 60)
  (h7 : mix.proportion1 / 100 * s1.carbonated_water + mix.proportion2 / 100 * s2.carbonated_water = 65) :
  s1.lemonade = 20 := by
sorry

end NUMINAMATH_CALUDE_lemonade_percentage_in_solution1_l2331_233192


namespace NUMINAMATH_CALUDE_prob_sum_greater_than_four_proof_l2331_233181

/-- The probability of rolling two dice and getting a sum greater than four -/
def prob_sum_greater_than_four : ℚ := 5/6

/-- The number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := 36

/-- The number of outcomes where the sum is less than or equal to four -/
def outcomes_sum_le_four : ℕ := 6

theorem prob_sum_greater_than_four_proof :
  prob_sum_greater_than_four = 1 - (outcomes_sum_le_four / total_outcomes) :=
by sorry

end NUMINAMATH_CALUDE_prob_sum_greater_than_four_proof_l2331_233181


namespace NUMINAMATH_CALUDE_ticket_price_is_six_l2331_233116

/-- The price of a concert ticket, given the following conditions:
  * Lana bought 8 tickets for herself and friends
  * Lana bought 2 extra tickets
  * Lana spent $60 in total
-/
def ticket_price : ℚ := by
  -- Define the number of tickets for Lana and friends
  let lana_friends_tickets : ℕ := 8
  -- Define the number of extra tickets
  let extra_tickets : ℕ := 2
  -- Define the total amount spent
  let total_spent : ℚ := 60
  -- Calculate the total number of tickets
  let total_tickets : ℕ := lana_friends_tickets + extra_tickets
  -- Calculate the price per ticket
  exact total_spent / total_tickets
  
-- Prove that the ticket price is $6
theorem ticket_price_is_six : ticket_price = 6 := by
  sorry

end NUMINAMATH_CALUDE_ticket_price_is_six_l2331_233116


namespace NUMINAMATH_CALUDE_well_depth_proof_well_depth_l2331_233132

theorem well_depth_proof (total_time : ℝ) (stone_fall_law : ℝ → ℝ) (sound_velocity : ℝ) : ℝ :=
  let depth := 2000
  let stone_fall_time := Real.sqrt (depth / 20)
  let sound_travel_time := depth / sound_velocity
  
  by
    have h1 : total_time = 10 := by sorry
    have h2 : ∀ t, stone_fall_law t = 20 * t^2 := by sorry
    have h3 : sound_velocity = 1120 := by sorry
    have h4 : stone_fall_time + sound_travel_time = total_time := by sorry
    
    -- The proof would go here
    sorry

-- The theorem states that given the conditions, the depth of the well is 2000 feet
theorem well_depth : well_depth_proof 10 (λ t => 20 * t^2) 1120 = 2000 := by sorry

end NUMINAMATH_CALUDE_well_depth_proof_well_depth_l2331_233132


namespace NUMINAMATH_CALUDE_intersection_line_slope_l2331_233159

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y - 11 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 14*x + 12*y + 60 = 0

-- Define the line passing through the intersection points
def intersectionLine (x y : ℝ) : Prop := 10*x - 10*y - 71 = 0

-- Theorem statement
theorem intersection_line_slope :
  ∀ (x1 y1 x2 y2 : ℝ),
  circle1 x1 y1 ∧ circle1 x2 y2 ∧
  circle2 x1 y1 ∧ circle2 x2 y2 ∧
  intersectionLine x1 y1 ∧ intersectionLine x2 y2 ∧
  x1 ≠ x2 →
  (y2 - y1) / (x2 - x1) = 1 :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_slope_l2331_233159


namespace NUMINAMATH_CALUDE_a_squared_b_plus_ab_squared_l2331_233182

theorem a_squared_b_plus_ab_squared (a b : ℝ) (h1 : a + b = 6) (h2 : a * b = 7) :
  a^2 * b + a * b^2 = 42 := by
  sorry

end NUMINAMATH_CALUDE_a_squared_b_plus_ab_squared_l2331_233182


namespace NUMINAMATH_CALUDE_pet_store_combinations_l2331_233168

def num_puppies : ℕ := 12
def num_kittens : ℕ := 8
def num_hamsters : ℕ := 10
def num_birds : ℕ := 5

theorem pet_store_combinations : 
  num_puppies * num_kittens * num_hamsters * num_birds * 4 * 3 * 2 * 1 = 115200 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_combinations_l2331_233168


namespace NUMINAMATH_CALUDE_mei_wendin_equation_theory_l2331_233165

theorem mei_wendin_equation_theory :
  ∀ x y : ℚ,
  (3 * x + 6 * y = 47/10) →
  (5 * x + 3 * y = 11/2) →
  (x = 9/10 ∧ y = 1/3) :=
by
  sorry

end NUMINAMATH_CALUDE_mei_wendin_equation_theory_l2331_233165


namespace NUMINAMATH_CALUDE_area_ratio_bound_l2331_233144

/-- A convex quadrilateral -/
structure ConvexQuadrilateral where
  area : ℝ
  area_pos : area > 0

/-- The result of reflecting each vertex of a quadrilateral 
    with respect to the diagonal that does not contain it -/
def reflect_vertices (q : ConvexQuadrilateral) : ℝ := 
  sorry

theorem area_ratio_bound (q : ConvexQuadrilateral) : 
  reflect_vertices q / q.area < 3 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_bound_l2331_233144


namespace NUMINAMATH_CALUDE_elevator_weight_problem_l2331_233183

/-- Given the initial conditions and new averages after each person enters,
    prove that the weights of X, Y, and Z are 195 lbs, 141 lbs, and 126 lbs respectively. -/
theorem elevator_weight_problem (initial_people : Nat) (initial_avg : ℝ)
    (avg_after_X : ℝ) (avg_after_Y : ℝ) (avg_after_Z : ℝ)
    (h1 : initial_people = 6)
    (h2 : initial_avg = 160)
    (h3 : avg_after_X = 165)
    (h4 : avg_after_Y = 162)
    (h5 : avg_after_Z = 158) :
    ∃ (X Y Z : ℝ),
      X = 195 ∧
      Y = 141 ∧
      Z = 126 ∧
      (initial_people : ℝ) * initial_avg + X = (initial_people + 1 : ℝ) * avg_after_X ∧
      ((initial_people + 1 : ℝ) * avg_after_X + Y = (initial_people + 2 : ℝ) * avg_after_Y) ∧
      ((initial_people + 2 : ℝ) * avg_after_Y + Z = (initial_people + 3 : ℝ) * avg_after_Z) :=
by sorry

end NUMINAMATH_CALUDE_elevator_weight_problem_l2331_233183


namespace NUMINAMATH_CALUDE_carol_rectangle_length_l2331_233121

theorem carol_rectangle_length 
  (carol_width jordan_length jordan_width : ℕ) 
  (h1 : carol_width = 24)
  (h2 : jordan_length = 8)
  (h3 : jordan_width = 15)
  (h4 : carol_width * carol_length = jordan_length * jordan_width) :
  carol_length = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_carol_rectangle_length_l2331_233121


namespace NUMINAMATH_CALUDE_rectangle_exists_in_octagon_decomposition_l2331_233184

/-- A regular octagon -/
structure RegularOctagon where
  -- Add necessary fields

/-- A parallelogram -/
structure Parallelogram where
  -- Add necessary fields

/-- A decomposition of a regular octagon into parallelograms -/
structure OctagonDecomposition where
  octagon : RegularOctagon
  parallelograms : Finset Parallelogram
  is_valid : Bool  -- Predicate to check if the decomposition is valid

/-- Predicate to check if a parallelogram is a rectangle -/
def is_rectangle (p : Parallelogram) : Prop :=
  sorry

/-- Main theorem: In any valid decomposition of a regular octagon into parallelograms,
    there exists at least one rectangle among the parallelograms -/
theorem rectangle_exists_in_octagon_decomposition (d : OctagonDecomposition) 
    (h : d.is_valid) : ∃ p ∈ d.parallelograms, is_rectangle p :=
  sorry

end NUMINAMATH_CALUDE_rectangle_exists_in_octagon_decomposition_l2331_233184


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_sqrt_two_l2331_233170

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  positive_a : 0 < a
  positive_b : 0 < b

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- A point on the plane -/
structure Point (α : Type*) where
  x : α
  y : α

/-- The area of a triangle given three points -/
def triangle_area (A B C : Point ℝ) : ℝ := sorry

/-- Theorem: The eccentricity of a hyperbola with specific properties is √2 -/
theorem hyperbola_eccentricity_sqrt_two 
  (a b c : ℝ) (h : Hyperbola a b) 
  (M N : Point ℝ) (A : Point ℝ) :
  (∃ F₁ F₂ : Point ℝ, 
    -- M and N are on the asymptote
    -- MF₁NF₂ is a rectangle
    -- A is a vertex of the hyperbola
    triangle_area A M N = (1/2) * c^2) →
  eccentricity h = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_sqrt_two_l2331_233170


namespace NUMINAMATH_CALUDE_rectangle_intersection_theorem_l2331_233140

/-- Represents a rectangle with a given area -/
structure Rectangle where
  area : ℝ

/-- Represents the configuration of three rectangles in a square -/
structure Configuration where
  square_side : ℝ
  rect1 : Rectangle
  rect2 : Rectangle
  rect3 : Rectangle

/-- The theorem to be proved -/
theorem rectangle_intersection_theorem (config : Configuration) 
  (h1 : config.square_side = 4)
  (h2 : config.rect1.area = 6)
  (h3 : config.rect2.area = 6)
  (h4 : config.rect3.area = 6) :
  ∃ (inter_area : ℝ), inter_area ≥ 2/3 ∧ 
  ((inter_area = (config.rect1.area + config.rect2.area - (config.square_side^2 - config.rect3.area)) / 2 ∨
    inter_area = (config.rect2.area + config.rect3.area - (config.square_side^2 - config.rect1.area)) / 2 ∨
    inter_area = (config.rect3.area + config.rect1.area - (config.square_side^2 - config.rect2.area)) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_intersection_theorem_l2331_233140


namespace NUMINAMATH_CALUDE_optimal_strategy_with_budget_optimal_strategy_without_budget_l2331_233195

/-- Revenue function -/
def R (x₁ x₂ : ℝ) : ℝ := -2 * x₁^2 - x₂^2 + 13 * x₁ + 11 * x₂ - 28

/-- Profit function -/
def profit (x₁ x₂ : ℝ) : ℝ := R x₁ x₂ - (x₁ + x₂)

/-- Theorem for part 1 -/
theorem optimal_strategy_with_budget :
  ∀ x₁ x₂ : ℝ, x₁ + x₂ = 5 → profit x₁ x₂ ≤ profit 2 3 :=
sorry

/-- Theorem for part 2 -/
theorem optimal_strategy_without_budget :
  ∀ x₁ x₂ : ℝ, profit x₁ x₂ ≤ profit 3 5 :=
sorry

end NUMINAMATH_CALUDE_optimal_strategy_with_budget_optimal_strategy_without_budget_l2331_233195


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2331_233160

def geometric_sequence (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_a2 : a 2 = 2) 
  (h_a5 : a 5 = 1/4) : 
  ∃ q : ℝ, q = 1/2 ∧ ∀ n : ℕ, a (n + 1) = a n * q :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2331_233160


namespace NUMINAMATH_CALUDE_mary_flour_amount_l2331_233199

/-- The amount of flour Mary puts in the cake. -/
def total_flour (recipe_flour extra_flour : ℝ) : ℝ :=
  recipe_flour + extra_flour

/-- Theorem stating the total amount of flour Mary uses. -/
theorem mary_flour_amount :
  let recipe_flour : ℝ := 7.0
  let extra_flour : ℝ := 2.0
  total_flour recipe_flour extra_flour = 9.0 := by
  sorry

end NUMINAMATH_CALUDE_mary_flour_amount_l2331_233199


namespace NUMINAMATH_CALUDE_sphere_volume_l2331_233123

theorem sphere_volume (d r h : ℝ) (h1 : d = 2 * Real.sqrt 5) (h2 : h = 2) 
  (h3 : r^2 = (d/2)^2 + h^2) : (4/3) * Real.pi * r^3 = 36 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_l2331_233123


namespace NUMINAMATH_CALUDE_total_study_hours_is_three_l2331_233152

-- Define the time spent on each subject in minutes
def science_time : ℕ := 60
def math_time : ℕ := 80
def literature_time : ℕ := 40

-- Define the total study time in minutes
def total_study_time : ℕ := science_time + math_time + literature_time

-- Define the conversion factor from minutes to hours
def minutes_per_hour : ℕ := 60

-- Theorem to prove
theorem total_study_hours_is_three :
  (total_study_time : ℚ) / minutes_per_hour = 3 := by sorry

end NUMINAMATH_CALUDE_total_study_hours_is_three_l2331_233152


namespace NUMINAMATH_CALUDE_not_all_x_heartsuit_zero_eq_x_l2331_233102

-- Define the heartsuit operation
def heartsuit (x y : ℝ) : ℝ := |x - y|

-- Theorem stating that "x ♡ 0 = x for all x" is false
theorem not_all_x_heartsuit_zero_eq_x : ¬ ∀ x : ℝ, heartsuit x 0 = x := by
  sorry

end NUMINAMATH_CALUDE_not_all_x_heartsuit_zero_eq_x_l2331_233102


namespace NUMINAMATH_CALUDE_quadratic_function_ratio_bound_l2331_233107

/-- A quadratic function f(x) = ax² + bx + c with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  derivative_at_zero_positive : b > 0
  nonnegative : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0

/-- The ratio of f(1) to f'(0) for a QuadraticFunction is always at least 2 -/
theorem quadratic_function_ratio_bound (f : QuadraticFunction) :
  (f.a + f.b + f.c) / f.b ≥ 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_ratio_bound_l2331_233107


namespace NUMINAMATH_CALUDE_negation_equivalence_l2331_233104

theorem negation_equivalence (a b x : ℝ) :
  ¬(x ≠ a ∧ x ≠ b → x^2 - (a+b)*x + a*b ≠ 0) ↔ (x = a ∨ x = b → x^2 - (a+b)*x + a*b = 0) :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2331_233104


namespace NUMINAMATH_CALUDE_cone_angle_theorem_l2331_233175

/-- A cone with vertex A -/
structure Cone where
  vertexAngle : ℝ

/-- The configuration of four cones as described in the problem -/
structure ConeConfiguration where
  cone1 : Cone
  cone2 : Cone
  cone3 : Cone
  cone4 : Cone
  cone1_eq_cone2 : cone1 = cone2
  cone3_angle : cone3.vertexAngle = π / 3
  cone4_angle : cone4.vertexAngle = 5 * π / 6
  external_tangent : True  -- Represents that cone1, cone2, and cone3 are externally tangent
  internal_tangent : True  -- Represents that cone4 is internally tangent to the other three

theorem cone_angle_theorem (config : ConeConfiguration) :
  config.cone1.vertexAngle = 2 * Real.arctan (Real.sqrt 3 - 1) := by
  sorry

end NUMINAMATH_CALUDE_cone_angle_theorem_l2331_233175


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l2331_233157

theorem trigonometric_equation_solution (x : ℝ) : 
  (4 * (Real.tan (8 * x))^4 + 4 * Real.sin (2 * x) * Real.sin (6 * x) - 
   Real.cos (4 * x) - Real.cos (12 * x) + 2) / Real.sqrt (Real.cos x - Real.sin x) = 0 ∧
  Real.cos x - Real.sin x > 0 ↔
  (∃ n : ℤ, x = -π/2 + 2*n*π ∨ x = -π/4 + 2*n*π ∨ x = 2*n*π) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l2331_233157


namespace NUMINAMATH_CALUDE_minRainfallDay4_is_21_l2331_233143

/-- Represents the rainfall data and conditions for a 4-day storm --/
structure RainfallData where
  capacity : ℝ  -- Area capacity in inches
  drain_rate : ℝ  -- Daily drainage rate in inches
  day1_rain : ℝ  -- Rainfall on day 1 in inches
  day2_rain : ℝ  -- Rainfall on day 2 in inches
  day3_rain : ℝ  -- Rainfall on day 3 in inches

/-- Calculates the minimum rainfall on day 4 for flooding to occur --/
def minRainfallDay4 (data : RainfallData) : ℝ :=
  data.capacity - (data.day1_rain + data.day2_rain + data.day3_rain - 3 * data.drain_rate)

/-- Theorem stating the minimum rainfall on day 4 for flooding --/
theorem minRainfallDay4_is_21 (data : RainfallData) :
  data.capacity = 72 ∧
  data.drain_rate = 3 ∧
  data.day1_rain = 10 ∧
  data.day2_rain = 2 * data.day1_rain ∧
  data.day3_rain = 1.5 * data.day2_rain →
  minRainfallDay4 data = 21 := by
  sorry

#eval minRainfallDay4 {
  capacity := 72,
  drain_rate := 3,
  day1_rain := 10,
  day2_rain := 20,
  day3_rain := 30
}

end NUMINAMATH_CALUDE_minRainfallDay4_is_21_l2331_233143


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l2331_233115

-- Define the quadratic function
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

-- State the theorem
theorem quadratic_roots_range (a b : ℝ) :
  (∃ x y, x ∈ Set.Icc 0 1 ∧ y ∈ Set.Icc 0 1 ∧ x ≠ y ∧ f a b x = 0 ∧ f a b y = 0) →
  a^2 - 2*b ∈ Set.Icc 0 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l2331_233115


namespace NUMINAMATH_CALUDE_square_perimeters_sum_l2331_233124

theorem square_perimeters_sum (a b : ℝ) (h1 : a^2 + b^2 = 130) (h2 : a^2 - b^2 = 50) :
  4*a + 4*b = 20 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeters_sum_l2331_233124


namespace NUMINAMATH_CALUDE_bobby_has_two_pizzas_l2331_233110

-- Define the number of slices per pizza
def slices_per_pizza : ℕ := 6

-- Define Mrs. Kaplan's number of slices
def kaplan_slices : ℕ := 3

-- Define the ratio of Mrs. Kaplan's slices to Bobby's slices
def kaplan_to_bobby_ratio : ℚ := 1 / 4

-- Define Bobby's number of pizzas
def bobby_pizzas : ℕ := 2

-- Theorem to prove
theorem bobby_has_two_pizzas :
  kaplan_slices = kaplan_to_bobby_ratio * (bobby_pizzas * slices_per_pizza) :=
by sorry

end NUMINAMATH_CALUDE_bobby_has_two_pizzas_l2331_233110


namespace NUMINAMATH_CALUDE_houses_with_dogs_l2331_233145

/-- Given a group of houses, prove the number of houses with dogs -/
theorem houses_with_dogs 
  (total_houses : ℕ) 
  (houses_with_cats : ℕ) 
  (houses_with_both : ℕ) 
  (h1 : total_houses = 60) 
  (h2 : houses_with_cats = 30) 
  (h3 : houses_with_both = 10) : 
  ∃ (houses_with_dogs : ℕ), houses_with_dogs = 40 ∧ 
    houses_with_dogs + houses_with_cats - houses_with_both ≤ total_houses :=
by sorry

end NUMINAMATH_CALUDE_houses_with_dogs_l2331_233145


namespace NUMINAMATH_CALUDE_student_group_size_l2331_233150

/-- The number of students in a group with overlapping class registrations --/
def num_students (history math english all_three two_classes : ℕ) : ℕ :=
  history + math + english - two_classes - 2 * all_three + all_three

theorem student_group_size :
  let history := 19
  let math := 14
  let english := 26
  let all_three := 3
  let two_classes := 7
  num_students history math english all_three two_classes = 46 := by
  sorry

end NUMINAMATH_CALUDE_student_group_size_l2331_233150


namespace NUMINAMATH_CALUDE_phone_number_A_equals_9_l2331_233139

def phone_number (A B C D E F G H I J : ℕ) : Prop :=
  A > B ∧ B > C ∧
  D > E ∧ E > F ∧
  G > H ∧ H > I ∧ I > J ∧
  D % 3 = 0 ∧ E % 3 = 0 ∧ F % 3 = 0 ∧
  E = D - 3 ∧ F = E - 3 ∧
  J % 2 = 0 ∧ G = J + 3 ∧ H = J + 2 ∧ I = J + 1 ∧
  A + B + C = 15 ∧
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧ A ≠ H ∧ A ≠ I ∧ A ≠ J ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧ B ≠ H ∧ B ≠ I ∧ B ≠ J ∧
  C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ C ≠ H ∧ C ≠ I ∧ C ≠ J ∧
  D ≠ E ∧ D ≠ F ∧ D ≠ G ∧ D ≠ H ∧ D ≠ I ∧ D ≠ J ∧
  E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ E ≠ I ∧ E ≠ J ∧
  F ≠ G ∧ F ≠ H ∧ F ≠ I ∧ F ≠ J ∧
  G ≠ H ∧ G ≠ I ∧ G ≠ J ∧
  H ≠ I ∧ H ≠ J ∧
  I ≠ J

theorem phone_number_A_equals_9 :
  ∀ A B C D E F G H I J : ℕ,
  phone_number A B C D E F G H I J → A = 9 :=
by sorry

end NUMINAMATH_CALUDE_phone_number_A_equals_9_l2331_233139


namespace NUMINAMATH_CALUDE_product_is_solution_quotient_is_solution_l2331_233149

/-- A type representing solutions of the equation x^2 - 5y^2 = 1 -/
structure Solution where
  x : ℝ
  y : ℝ
  property : x^2 - 5*y^2 = 1

/-- The product of two solutions is also a solution -/
theorem product_is_solution (s₁ s₂ : Solution) :
  ∃ (m n : ℝ), m^2 - 5*n^2 = 1 ∧ m + n * Real.sqrt 5 = (s₁.x + s₁.y * Real.sqrt 5) * (s₂.x + s₂.y * Real.sqrt 5) :=
by sorry

/-- The quotient of two solutions can be represented as p + q√5 and is also a solution -/
theorem quotient_is_solution (s₁ s₂ : Solution) (h : s₂.x^2 - 5*s₂.y^2 ≠ 0) :
  ∃ (p q : ℝ), p^2 - 5*q^2 = 1 ∧ p + q * Real.sqrt 5 = (s₁.x + s₁.y * Real.sqrt 5) / (s₂.x + s₂.y * Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_product_is_solution_quotient_is_solution_l2331_233149


namespace NUMINAMATH_CALUDE_probability_of_vowels_in_all_sets_l2331_233189

-- Define the sets
def set1 : Finset Char := {'a', 'b', 'o', 'd', 'e', 'f', 'g'}
def set2 : Finset Char := {'k', 'l', 'm', 'n', 'u', 'p', 'r', 's'}
def set3 : Finset Char := {'t', 'v', 'w', 'i', 'x', 'y', 'z'}
def set4 : Finset Char := {'a', 'c', 'e', 'u', 'g', 'h', 'j'}

-- Define vowels
def vowels : Finset Char := {'a', 'e', 'i', 'o', 'u'}

-- Function to count vowels in a set
def countVowels (s : Finset Char) : Nat :=
  (s ∩ vowels).card

-- Theorem statement
theorem probability_of_vowels_in_all_sets :
  let p1 := (countVowels set1 : ℚ) / set1.card
  let p2 := (countVowels set2 : ℚ) / set2.card
  let p3 := (countVowels set3 : ℚ) / set3.card
  let p4 := (countVowels set4 : ℚ) / set4.card
  p1 * p2 * p3 * p4 = 9 / 2744 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_vowels_in_all_sets_l2331_233189


namespace NUMINAMATH_CALUDE_int_poly5_root_count_l2331_233172

/-- A polynomial of degree 5 with integer coefficients -/
structure IntPoly5 where
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ

/-- The set of possible numbers of integer roots for an IntPoly5 -/
def possibleRootCounts : Set ℕ := {0, 1, 2, 4, 5}

/-- The number of integer roots (counting multiplicity) of an IntPoly5 -/
def numIntegerRoots (p : IntPoly5) : ℕ := sorry

/-- Theorem stating that the number of integer roots of an IntPoly5 is in the set of possible root counts -/
theorem int_poly5_root_count (p : IntPoly5) : numIntegerRoots p ∈ possibleRootCounts := by sorry

end NUMINAMATH_CALUDE_int_poly5_root_count_l2331_233172


namespace NUMINAMATH_CALUDE_largest_angle_in_special_quadrilateral_l2331_233136

/-- The measure of the largest angle in a quadrilateral with angles in the ratio 3:4:5:6 -/
theorem largest_angle_in_special_quadrilateral : 
  ∀ (a b c d : ℝ), 
  (a + b + c + d = 360) →  -- Sum of angles in a quadrilateral is 360°
  (∃ (k : ℝ), a = 3*k ∧ b = 4*k ∧ c = 5*k ∧ d = 6*k) →  -- Angles are in the ratio 3:4:5:6
  max a (max b (max c d)) = 120  -- The largest angle is 120°
:= by sorry

end NUMINAMATH_CALUDE_largest_angle_in_special_quadrilateral_l2331_233136


namespace NUMINAMATH_CALUDE_jasmine_swims_300_laps_l2331_233188

/-- The number of laps Jasmine swims in five weeks -/
def jasmine_laps : ℕ :=
  let laps_per_day : ℕ := 12
  let days_per_week : ℕ := 5
  let num_weeks : ℕ := 5
  laps_per_day * days_per_week * num_weeks

/-- Theorem stating that Jasmine swims 300 laps in five weeks -/
theorem jasmine_swims_300_laps : jasmine_laps = 300 := by
  sorry

end NUMINAMATH_CALUDE_jasmine_swims_300_laps_l2331_233188


namespace NUMINAMATH_CALUDE_a_6_value_l2331_233169

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem a_6_value (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 4 = 1 ∧ a 8 = 3) ∨ (a 4 = 3 ∧ a 8 = 1) →
  a 6 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_a_6_value_l2331_233169


namespace NUMINAMATH_CALUDE_oprah_car_giveaway_l2331_233173

theorem oprah_car_giveaway (initial_cars final_cars years : ℕ) 
  (h1 : initial_cars = 3500)
  (h2 : final_cars = 500)
  (h3 : years = 60) :
  (initial_cars - final_cars) / years = 50 :=
by sorry

end NUMINAMATH_CALUDE_oprah_car_giveaway_l2331_233173


namespace NUMINAMATH_CALUDE_value_of_a_l2331_233138

theorem value_of_a (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a^3 / b = 1) (h2 : b^3 / c = 8) (h3 : c^3 / a = 27) :
  a = (24^(1/8 : ℝ))^(1/3 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_value_of_a_l2331_233138


namespace NUMINAMATH_CALUDE_smallest_prime_after_six_nonprimes_l2331_233153

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

-- Define a function to check if a number is the start of six consecutive non-primes
def isSixConsecutiveNonPrimes (n : ℕ) : Prop :=
  ∀ k : ℕ, k ≥ n → k < n + 6 → ¬(isPrime k)

-- Theorem statement
theorem smallest_prime_after_six_nonprimes :
  ∃ p : ℕ, isPrime p ∧ 
    (∃ n : ℕ, isSixConsecutiveNonPrimes n ∧ p = n + 6) ∧
    (∀ q : ℕ, q < p → ¬(∃ m : ℕ, isSixConsecutiveNonPrimes m ∧ isPrime (m + 6) ∧ q = m + 6)) :=
  sorry

end NUMINAMATH_CALUDE_smallest_prime_after_six_nonprimes_l2331_233153


namespace NUMINAMATH_CALUDE_exponential_equation_solutions_l2331_233167

theorem exponential_equation_solutions :
  ∀ x y : ℕ+, (3 : ℕ) ^ x.val = 2 ^ x.val * y.val + 1 ↔ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 2) ∨ (x = 4 ∧ y = 5) := by
  sorry

end NUMINAMATH_CALUDE_exponential_equation_solutions_l2331_233167


namespace NUMINAMATH_CALUDE_geometric_sequence_iff_k_eq_neg_one_l2331_233103

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

def S (n : ℕ) (k : ℝ) : ℝ := 3^n + k

def a (n : ℕ) (k : ℝ) : ℝ :=
  if n = 1 then S 1 k else S n k - S (n-1) k

theorem geometric_sequence_iff_k_eq_neg_one (k : ℝ) :
  is_geometric_sequence (a · k) ↔ k = -1 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_iff_k_eq_neg_one_l2331_233103


namespace NUMINAMATH_CALUDE_remainder_3_800_mod_17_l2331_233163

theorem remainder_3_800_mod_17 : 3^800 % 17 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_800_mod_17_l2331_233163


namespace NUMINAMATH_CALUDE_min_value_theorem_l2331_233109

theorem min_value_theorem (x y z : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0) (h_prod : x * y * z = 27) : 
  ∀ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 27 → 3 * a + 2 * b + c ≥ 18 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2331_233109


namespace NUMINAMATH_CALUDE_angle_C_measure_l2331_233174

theorem angle_C_measure (A B C : ℝ) (h1 : 4 * Real.sin A + 2 * Real.cos B = 4)
  (h2 : (1/2) * Real.sin B + Real.cos A = Real.sqrt 3 / 2) : C = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_measure_l2331_233174


namespace NUMINAMATH_CALUDE_harriet_drive_time_l2331_233154

theorem harriet_drive_time (total_time : ℝ) (outbound_speed return_speed : ℝ) 
  (h1 : total_time = 5)
  (h2 : outbound_speed = 100)
  (h3 : return_speed = 150) :
  let distance := (total_time * outbound_speed * return_speed) / (outbound_speed + return_speed)
  let outbound_time := distance / outbound_speed
  outbound_time * 60 = 180 := by
sorry

end NUMINAMATH_CALUDE_harriet_drive_time_l2331_233154


namespace NUMINAMATH_CALUDE_street_house_numbers_l2331_233133

/-- Calculates the sum of digits for all numbers in an arithmetic sequence -/
def sumOfDigits (start : Nat) (diff : Nat) (count : Nat) : Nat :=
  sorry

theorem street_house_numbers (eastStart : Nat) (eastDiff : Nat) (westStart : Nat) (westDiff : Nat) 
  (houseCount : Nat) (costPerDigit : Nat) :
  eastStart = 5 → eastDiff = 7 → westStart = 7 → westDiff = 8 → houseCount = 30 → costPerDigit = 1 →
  sumOfDigits eastStart eastDiff houseCount + sumOfDigits westStart westDiff houseCount = 149 :=
by sorry

end NUMINAMATH_CALUDE_street_house_numbers_l2331_233133


namespace NUMINAMATH_CALUDE_floor_abs_negative_real_l2331_233179

theorem floor_abs_negative_real : ⌊|(-45.7 : ℝ)|⌋ = 45 := by sorry

end NUMINAMATH_CALUDE_floor_abs_negative_real_l2331_233179


namespace NUMINAMATH_CALUDE_coin_value_calculation_l2331_233122

theorem coin_value_calculation (num_quarters num_nickels : ℕ) 
  (quarter_value nickel_value : ℚ) : 
  num_quarters = 8 → 
  num_nickels = 13 → 
  quarter_value = 25 / 100 → 
  nickel_value = 5 / 100 → 
  num_quarters * quarter_value + num_nickels * nickel_value = 265 / 100 := by
  sorry

end NUMINAMATH_CALUDE_coin_value_calculation_l2331_233122


namespace NUMINAMATH_CALUDE_quadratic_equation_equivalence_l2331_233142

/-- Given two quadratic equations and a relationship between their roots, 
    prove the condition for the equations to be identical -/
theorem quadratic_equation_equivalence 
  (p q r s : ℝ) 
  (x₁ x₂ y₁ y₂ : ℝ) : 
  (x₁^2 + p*x₁ + q = 0) →
  (x₂^2 + p*x₂ + q = 0) →
  (y₁^2 + r*y₁ + s = 0) →
  (y₂^2 + r*y₂ + s = 0) →
  (y₁ = x₁/(x₁-1)) →
  (y₂ = x₂/(x₂-1)) →
  (x₁ ≠ 1) →
  (x₂ ≠ 1) →
  (p = -r ∧ q = s) →
  (p + q = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_equivalence_l2331_233142


namespace NUMINAMATH_CALUDE_election_result_l2331_233156

theorem election_result (total_votes : ℕ) (second_candidate_votes : ℕ) :
  total_votes = 1200 →
  second_candidate_votes = 480 →
  (total_votes - second_candidate_votes) / total_votes = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_election_result_l2331_233156


namespace NUMINAMATH_CALUDE_quadratic_root_bound_l2331_233196

theorem quadratic_root_bound (a b c x : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (hx : a * x^2 + b * x + c = 0) : 
  |x| ≤ (2 * |a * c| + b^2) / (|a * b|) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_bound_l2331_233196


namespace NUMINAMATH_CALUDE_sector_central_angle_l2331_233114

-- Define the sector
structure Sector where
  perimeter : ℝ
  area : ℝ

-- Theorem statement
theorem sector_central_angle (s : Sector) (h1 : s.perimeter = 8) (h2 : s.area = 4) :
  ∃ (r l θ : ℝ), r > 0 ∧ l > 0 ∧ θ > 0 ∧ 
  2 * r + l = s.perimeter ∧
  1 / 2 * l * r = s.area ∧
  θ = l / r ∧
  θ = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l2331_233114


namespace NUMINAMATH_CALUDE_cube_volume_from_lateral_area_l2331_233127

/-- 
Given a cube with lateral surface area of 100 square units, 
prove that its volume is 125 cubic units.
-/
theorem cube_volume_from_lateral_area : 
  ∀ s : ℝ, 
  (4 * s^2 = 100) → 
  (s^3 = 125) :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_lateral_area_l2331_233127


namespace NUMINAMATH_CALUDE_angle_B_range_l2331_233164

theorem angle_B_range (A B C : ℝ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C)
  (h4 : A + B + C = 180) (h5 : A ≤ B) (h6 : B ≤ C) (h7 : 2 * B = 5 * A) :
  0 < B ∧ B ≤ 75 := by
sorry

end NUMINAMATH_CALUDE_angle_B_range_l2331_233164


namespace NUMINAMATH_CALUDE_range_of_m_l2331_233166

-- Define the propositions P and Q
def P (m : ℝ) : Prop := ∃ (x y : ℝ), x^2 + y^2 - 2*m*x + 2*m^2 - 2*m = 0

def Q (m : ℝ) : Prop := 
  let e := Real.sqrt (1 + m/5)
  1 < e ∧ e < 2

-- State the theorem
theorem range_of_m : 
  ∀ m : ℝ, (¬(P m) ∧ Q m) → 2 ≤ m ∧ m < 15 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2331_233166


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2331_233119

theorem arithmetic_calculation : 3 * 11 + 3 * 12 + 3 * 15 + 11 = 125 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2331_233119


namespace NUMINAMATH_CALUDE_race_distance_is_17_l2331_233120

/-- Represents the relay race with given conditions -/
structure RelayRace where
  totalTime : Real
  sadieTime : Real
  sadieSpeed : Real
  arianaTime : Real
  arianaSpeed : Real
  sarahSpeed : Real

/-- Calculates the total distance of the relay race -/
def totalDistance (race : RelayRace) : Real :=
  let sadieDistance := race.sadieTime * race.sadieSpeed
  let arianaDistance := race.arianaTime * race.arianaSpeed
  let sarahTime := race.totalTime - race.sadieTime - race.arianaTime
  let sarahDistance := sarahTime * race.sarahSpeed
  sadieDistance + arianaDistance + sarahDistance

/-- Theorem stating that the total distance of the given race is 17 miles -/
theorem race_distance_is_17 (race : RelayRace) 
  (h1 : race.totalTime = 4.5)
  (h2 : race.sadieTime = 2)
  (h3 : race.sadieSpeed = 3)
  (h4 : race.arianaTime = 0.5)
  (h5 : race.arianaSpeed = 6)
  (h6 : race.sarahSpeed = 4) :
  totalDistance race = 17 := by
  sorry

#eval totalDistance { totalTime := 4.5, sadieTime := 2, sadieSpeed := 3, arianaTime := 0.5, arianaSpeed := 6, sarahSpeed := 4 }

end NUMINAMATH_CALUDE_race_distance_is_17_l2331_233120


namespace NUMINAMATH_CALUDE_square_perimeters_sum_l2331_233134

theorem square_perimeters_sum (x y : ℝ) 
  (h1 : x^2 + y^2 = 130)
  (h2 : x^2 / y^2 = 4) :
  4*x + 4*y = 12 * Real.sqrt 26 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeters_sum_l2331_233134


namespace NUMINAMATH_CALUDE_expressions_equality_l2331_233111

theorem expressions_equality :
  -- Expression 1
  (1 + Real.sqrt 3) * (2 - Real.sqrt 3) = -1 + Real.sqrt 3 ∧
  -- Expression 2
  2 * (Real.sqrt (9/2) - Real.sqrt 8 / 3) * (2 * Real.sqrt 2) = 10/3 ∧
  -- Expression 3
  Real.sqrt 18 - Real.sqrt 8 + Real.sqrt (1/8) = 5 * Real.sqrt 2 / 4 ∧
  -- Expression 4
  (Real.sqrt 6 - 2 * Real.sqrt 15) * Real.sqrt 3 - 6 * Real.sqrt (1/2) = -6 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_expressions_equality_l2331_233111


namespace NUMINAMATH_CALUDE_catholic_tower_height_l2331_233158

/-- Given two towers and a grain between them, prove the height of the second tower --/
theorem catholic_tower_height 
  (church_height : ℝ) 
  (total_distance : ℝ) 
  (grain_distance : ℝ) 
  (h : ℝ → church_height = 150 ∧ total_distance = 350 ∧ grain_distance = 150) :
  ∃ (catholic_height : ℝ), 
    catholic_height = 50 * Real.sqrt 5 ∧ 
    (church_height^2 + grain_distance^2 = 
     catholic_height^2 + (total_distance - grain_distance)^2) :=
by sorry

end NUMINAMATH_CALUDE_catholic_tower_height_l2331_233158


namespace NUMINAMATH_CALUDE_total_pepper_pieces_l2331_233112

-- Define the number of bell peppers
def num_peppers : ℕ := 5

-- Define the number of large slices per pepper
def slices_per_pepper : ℕ := 20

-- Define the number of smaller pieces each half-slice is cut into
def smaller_pieces_per_slice : ℕ := 3

-- Theorem to prove
theorem total_pepper_pieces :
  let total_large_slices := num_peppers * slices_per_pepper
  let half_large_slices := total_large_slices / 2
  let smaller_pieces := half_large_slices * smaller_pieces_per_slice
  let remaining_large_slices := total_large_slices - half_large_slices
  remaining_large_slices + smaller_pieces = 200 := by
  sorry

end NUMINAMATH_CALUDE_total_pepper_pieces_l2331_233112


namespace NUMINAMATH_CALUDE_chord_line_equation_l2331_233130

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in the 2D plane represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given a circle and a point that is the midpoint of a chord, 
    return the line containing that chord -/
def chordLine (c : Circle) (p : ℝ × ℝ) : Line :=
  sorry

theorem chord_line_equation (c : Circle) (p : ℝ × ℝ) :
  let circle : Circle := { center := (3, 0), radius := 3 }
  let midpoint : ℝ × ℝ := (4, 2)
  let line := chordLine circle midpoint
  line.a = 1 ∧ line.b = 2 ∧ line.c = -8 := by sorry

end NUMINAMATH_CALUDE_chord_line_equation_l2331_233130


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_area_l2331_233198

/-- Represents an isosceles trapezoid -/
structure IsoscelesTrapezoid where
  leg_length : ℝ
  diagonal_length : ℝ
  longer_base : ℝ

/-- Calculates the area of an isosceles trapezoid -/
def area (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem stating the area of the specific isosceles trapezoid -/
theorem isosceles_trapezoid_area :
  let t : IsoscelesTrapezoid := {
    leg_length := 25,
    diagonal_length := 34,
    longer_base := 40
  }
  ∃ ε > 0, |area t - 569.275| < ε :=
sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_area_l2331_233198


namespace NUMINAMATH_CALUDE_y_intercept_of_specific_line_l2331_233187

/-- A line in a 2D plane. -/
structure Line where
  slope : ℝ
  x_intercept : ℝ × ℝ

/-- The y-intercept of a line. -/
def y_intercept (l : Line) : ℝ × ℝ :=
  (0, l.slope * (-l.x_intercept.1) + l.x_intercept.2)

/-- Theorem: For a line with slope -3 and x-intercept (4,0), the y-intercept is (0,12). -/
theorem y_intercept_of_specific_line :
  let l : Line := { slope := -3, x_intercept := (4, 0) }
  y_intercept l = (0, 12) := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_specific_line_l2331_233187


namespace NUMINAMATH_CALUDE_car_average_speed_l2331_233113

/-- The average speed of a car given its speeds in two consecutive hours -/
theorem car_average_speed (speed1 speed2 : ℝ) (h1 : speed1 = 145) (h2 : speed2 = 60) :
  (speed1 + speed2) / 2 = 102.5 := by
  sorry

end NUMINAMATH_CALUDE_car_average_speed_l2331_233113


namespace NUMINAMATH_CALUDE_sum_of_solutions_l2331_233186

theorem sum_of_solutions (x y : ℝ) 
  (hx : (x - 1)^3 + 2015*(x - 1) = -1) 
  (hy : (y - 1)^3 + 2015*(y - 1) = 1) : 
  x + y = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l2331_233186


namespace NUMINAMATH_CALUDE_monotonic_decreasing_condition_l2331_233176

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 3 * x + 4

-- State the theorem
theorem monotonic_decreasing_condition (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ < 6 → f a x₁ > f a x₂) ↔ 0 ≤ a ∧ a ≤ 1/4 :=
by sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_condition_l2331_233176


namespace NUMINAMATH_CALUDE_algebraic_simplification_l2331_233193

theorem algebraic_simplification (a b : ℝ) : -3*a*(2*a - 4*b + 2) + 6*a = -6*a^2 + 12*a*b := by
  sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l2331_233193


namespace NUMINAMATH_CALUDE_student_average_weight_l2331_233180

theorem student_average_weight 
  (n : ℕ) 
  (teacher_weight : ℝ) 
  (weight_increase : ℝ) : 
  n = 24 → 
  teacher_weight = 45 → 
  weight_increase = 0.4 → 
  (n * 35 + teacher_weight) / (n + 1) = 35 + weight_increase :=
by sorry

end NUMINAMATH_CALUDE_student_average_weight_l2331_233180


namespace NUMINAMATH_CALUDE_range_of_c_l2331_233178

/-- Given c > 0, if the function y = c^x is monotonically decreasing on ℝ or 
    the function g(x) = lg(2cx^2 + 2x + 1) has domain ℝ, but not both, 
    then c ≥ 1 or 0 < c ≤ 1/2 -/
theorem range_of_c (c : ℝ) (h_c : c > 0) : 
  (∀ x y : ℝ, x < y → c^x > c^y) ∨ 
  (∀ x : ℝ, 2*c*x^2 + 2*x + 1 > 0) ∧ 
  ¬((∀ x y : ℝ, x < y → c^x > c^y) ∧ 
    (∀ x : ℝ, 2*c*x^2 + 2*x + 1 > 0)) → 
  c ≥ 1 ∨ (0 < c ∧ c ≤ 1/2) := by
  sorry

end NUMINAMATH_CALUDE_range_of_c_l2331_233178


namespace NUMINAMATH_CALUDE_range_of_a_l2331_233117

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ Real.exp x * (x + a) < 1) → a < 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2331_233117


namespace NUMINAMATH_CALUDE_shopping_price_difference_l2331_233155

/-- Proves that the difference between shoe price and bag price is $17 --/
theorem shopping_price_difference 
  (initial_amount : ℕ) 
  (shoe_price : ℕ) 
  (remaining_amount : ℕ) 
  (bag_price : ℕ) 
  (lunch_price : ℕ) 
  (h1 : initial_amount = 158)
  (h2 : shoe_price = 45)
  (h3 : remaining_amount = 78)
  (h4 : lunch_price = bag_price / 4)
  (h5 : initial_amount = shoe_price + bag_price + lunch_price + remaining_amount) :
  shoe_price - bag_price = 17 := by
  sorry

end NUMINAMATH_CALUDE_shopping_price_difference_l2331_233155


namespace NUMINAMATH_CALUDE_first_nonzero_digit_not_eventually_periodic_l2331_233141

/-- The first non-zero digit from the unit's place in the decimal representation of n! -/
def first_nonzero_digit (n : ℕ) : ℕ :=
  sorry

/-- The sequence of first non-zero digits is eventually periodic if there exists an N such that
    the sequence {a_n}_{n>N} is periodic -/
def eventually_periodic (a : ℕ → ℕ) : Prop :=
  ∃ N T : ℕ, T > 0 ∧ ∀ n > N, a (n + T) = a n

theorem first_nonzero_digit_not_eventually_periodic :
  ¬ eventually_periodic first_nonzero_digit :=
sorry

end NUMINAMATH_CALUDE_first_nonzero_digit_not_eventually_periodic_l2331_233141


namespace NUMINAMATH_CALUDE_ball_distribution_l2331_233118

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- The number of ways to choose m items from n items -/
def choose (n : ℕ) (m : ℕ) : ℕ :=
  sorry

theorem ball_distribution :
  let total_balls : ℕ := 6
  let num_boxes : ℕ := 4
  distribute_balls total_balls num_boxes = choose num_boxes 2 + num_boxes := by
  sorry

end NUMINAMATH_CALUDE_ball_distribution_l2331_233118


namespace NUMINAMATH_CALUDE_school_garbage_plan_l2331_233135

/-- Represents a purchasing plan for warm reminder signs and garbage bins -/
structure PurchasePlan where
  signs : ℕ
  bins : ℕ

/-- Calculates the total cost of a purchasing plan given the prices -/
def totalCost (plan : PurchasePlan) (signPrice binPrice : ℕ) : ℕ :=
  plan.signs * signPrice + plan.bins * binPrice

theorem school_garbage_plan :
  ∃ (signPrice binPrice : ℕ) (bestPlan : PurchasePlan),
    -- Conditions
    (2 * signPrice + 3 * binPrice = 550) ∧
    (binPrice = 3 * signPrice) ∧
    (bestPlan.signs + bestPlan.bins = 100) ∧
    (bestPlan.bins ≥ 48) ∧
    (totalCost bestPlan signPrice binPrice ≤ 10000) ∧
    -- Conclusions
    (signPrice = 50) ∧
    (binPrice = 150) ∧
    (bestPlan.signs = 52) ∧
    (bestPlan.bins = 48) ∧
    (totalCost bestPlan signPrice binPrice = 9800) ∧
    (∀ (plan : PurchasePlan),
      (plan.signs + plan.bins = 100) →
      (plan.bins ≥ 48) →
      (totalCost plan signPrice binPrice ≤ 10000) →
      (totalCost plan signPrice binPrice ≥ totalCost bestPlan signPrice binPrice)) :=
by
  sorry

end NUMINAMATH_CALUDE_school_garbage_plan_l2331_233135


namespace NUMINAMATH_CALUDE_certain_number_problem_l2331_233151

theorem certain_number_problem (x N : ℤ) (h1 : 3 * x = (N - x) + 18) (h2 : x = 11) : N = 26 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2331_233151


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2331_233190

/-- Simplification of polynomial expression -/
theorem polynomial_simplification (x : ℝ) :
  (3 * x^10 + 5 * x^9 + 2 * x^8) + (7 * x^12 - x^10 + 4 * x^9 + x^7 + 6 * x^4 + 9) =
  7 * x^12 + 2 * x^10 + 9 * x^9 + 2 * x^8 + x^7 + 6 * x^4 + 9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2331_233190


namespace NUMINAMATH_CALUDE_ninety_ninth_digit_sum_l2331_233129

/-- The decimal expansion of 2/9 -/
def decimal_expansion_2_9 : ℚ := 2/9

/-- The decimal expansion of 3/11 -/
def decimal_expansion_3_11 : ℚ := 3/11

/-- The 99th digit after the decimal point in a rational number -/
def digit_99 (q : ℚ) : ℕ :=
  sorry

/-- Theorem: The 99th digit after the decimal point in the decimal expansion of 2/9 + 3/11 is 4 -/
theorem ninety_ninth_digit_sum :
  digit_99 (decimal_expansion_2_9 + decimal_expansion_3_11) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ninety_ninth_digit_sum_l2331_233129


namespace NUMINAMATH_CALUDE_candy_distribution_l2331_233128

theorem candy_distribution (N : ℕ) : N > 1 ∧ 
  N % 2 = 1 ∧ 
  N % 3 = 1 ∧ 
  N % 5 = 1 ∧ 
  (∀ m : ℕ, m > 1 → m % 2 = 1 → m % 3 = 1 → m % 5 = 1 → m ≥ N) → 
  N = 31 := by
sorry

end NUMINAMATH_CALUDE_candy_distribution_l2331_233128


namespace NUMINAMATH_CALUDE_train_speed_l2331_233161

/-- Proves that a train with given specifications travels at 45 km/hr -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (total_length : ℝ) :
  train_length = 150 ∧ 
  crossing_time = 30 ∧ 
  total_length = 225 →
  (total_length - train_length) / crossing_time * 3.6 = 45 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2331_233161


namespace NUMINAMATH_CALUDE_ned_bomb_diffusion_l2331_233194

/-- Ned's bomb diffusion problem -/
theorem ned_bomb_diffusion (total_flights : ℕ) (time_per_flight : ℕ) (bomb_timer : ℕ) (time_spent : ℕ)
  (h1 : total_flights = 20)
  (h2 : time_per_flight = 11)
  (h3 : bomb_timer = 72)
  (h4 : time_spent = 165) :
  bomb_timer - (total_flights - time_spent / time_per_flight) * time_per_flight = 17 := by
  sorry

#check ned_bomb_diffusion

end NUMINAMATH_CALUDE_ned_bomb_diffusion_l2331_233194
