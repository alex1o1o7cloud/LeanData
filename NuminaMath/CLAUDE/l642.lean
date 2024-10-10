import Mathlib

namespace percy_dish_cost_l642_64202

/-- The cost of a meal for three people with a 10% tip --/
def meal_cost (leticia_cost scarlett_cost percy_cost : ℝ) : ℝ :=
  (leticia_cost + scarlett_cost + percy_cost) * 1.1

/-- The theorem stating the cost of Percy's dish --/
theorem percy_dish_cost : 
  ∃ (percy_cost : ℝ), 
    meal_cost 10 13 percy_cost = 44 ∧ 
    percy_cost = 17 := by
  sorry

end percy_dish_cost_l642_64202


namespace olives_per_jar_l642_64239

/-- Proves that the number of olives in a jar is 20 given the problem conditions --/
theorem olives_per_jar (
  total_money : ℝ)
  (olives_needed : ℕ)
  (jar_cost : ℝ)
  (change : ℝ)
  (h1 : total_money = 10)
  (h2 : olives_needed = 80)
  (h3 : jar_cost = 1.5)
  (h4 : change = 4)
  : (olives_needed : ℝ) / ((total_money - change) / jar_cost) = 20 := by
  sorry

end olives_per_jar_l642_64239


namespace contest_result_l642_64235

/-- The total number of baskets made by Alex, Sandra, and Hector -/
def totalBaskets (alex sandra hector : ℕ) : ℕ := alex + sandra + hector

/-- Theorem: Given the conditions, the total number of baskets is 80 -/
theorem contest_result : ∃ (sandra hector : ℕ),
  sandra = 3 * 8 ∧ 
  hector = 2 * sandra ∧
  totalBaskets 8 sandra hector = 80 := by
  sorry

end contest_result_l642_64235


namespace david_current_age_l642_64227

/-- David's current age -/
def david_age : ℕ := sorry

/-- David's daughter's current age -/
def daughter_age : ℕ := 12

/-- Number of years until David's age is twice his daughter's -/
def years_until_double : ℕ := 16

theorem david_current_age :
  david_age = 40 ∧
  david_age + years_until_double = 2 * (daughter_age + years_until_double) :=
by sorry

end david_current_age_l642_64227


namespace binary_representation_of_51_l642_64211

/-- Converts a natural number to its binary representation as a list of booleans -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec toBinaryAux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinaryAux (m / 2)
  toBinaryAux n

/-- The decimal number to be converted -/
def decimalNumber : ℕ := 51

/-- The expected binary representation -/
def expectedBinary : List Bool := [true, true, false, false, true, true]

/-- Theorem stating that the binary representation of 51 is [true, true, false, false, true, true] -/
theorem binary_representation_of_51 :
  toBinary decimalNumber = expectedBinary := by
  sorry

end binary_representation_of_51_l642_64211


namespace sum_reciprocals_equals_one_l642_64226

theorem sum_reciprocals_equals_one (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = x * y) : 
  1 / x + 1 / y = 1 := by
sorry

end sum_reciprocals_equals_one_l642_64226


namespace travel_ways_l642_64279

theorem travel_ways (ways_AB ways_BC : ℕ) (h1 : ways_AB = 3) (h2 : ways_BC = 2) : 
  ways_AB * ways_BC = 6 := by
  sorry

end travel_ways_l642_64279


namespace janabel_widget_sales_l642_64214

theorem janabel_widget_sales :
  let a : ℕ → ℕ := fun n => 2 * n - 1
  let S : ℕ → ℕ := fun n => n * (a 1 + a n) / 2
  S 20 = 400 := by
  sorry

end janabel_widget_sales_l642_64214


namespace mikes_money_duration_l642_64290

/-- The number of weeks Mike's money will last given his earnings and weekly spending. -/
def weeks_money_lasts (lawn_earnings weed_eating_earnings weekly_spending : ℕ) : ℕ :=
  (lawn_earnings + weed_eating_earnings) / weekly_spending

/-- Theorem stating that Mike's money will last 8 weeks given his earnings and spending. -/
theorem mikes_money_duration :
  weeks_money_lasts 14 26 5 = 8 :=
by sorry

end mikes_money_duration_l642_64290


namespace solve_equation_l642_64298

theorem solve_equation (x : ℝ) : (3 - 5 + 7 = 6 - x) → x = 1 := by
  sorry

end solve_equation_l642_64298


namespace remainder_proof_l642_64222

theorem remainder_proof : ∃ (q : ℕ), 4351 = 101 * q + 8 :=
by
  -- We define the greatest common divisor G as 101
  let G : ℕ := 101

  -- We define the condition that G divides 5161 with remainder 10
  have h1 : ∃ (q : ℕ), 5161 = G * q + 10 := by sorry

  -- We prove that 4351 divided by G has remainder 8
  sorry

end remainder_proof_l642_64222


namespace sin_690_degrees_l642_64256

theorem sin_690_degrees : Real.sin (690 * π / 180) = -1/2 := by
  sorry

end sin_690_degrees_l642_64256


namespace trapezium_side_length_l642_64246

theorem trapezium_side_length (a b h area : ℝ) : 
  b = 20 → h = 12 → area = 228 → area = (a + b) * h / 2 → a = 18 := by
  sorry

end trapezium_side_length_l642_64246


namespace repeating_decimal_sum_l642_64230

theorem repeating_decimal_sum : 
  (1 / 3 : ℚ) + (4 / 99 : ℚ) + (5 / 999 : ℚ) = (14 / 37 : ℚ) := by
  sorry

end repeating_decimal_sum_l642_64230


namespace two_triangles_exist_l642_64280

/-- A triangle with a given angle, height, and circumradius. -/
structure SpecialTriangle where
  /-- One of the angles of the triangle -/
  angle : ℝ
  /-- The height corresponding to one side of the triangle -/
  height : ℝ
  /-- The radius of the circumcircle -/
  circumradius : ℝ
  /-- The angle is positive and less than π -/
  angle_pos : 0 < angle
  angle_lt_pi : angle < π
  /-- The height is positive -/
  height_pos : 0 < height
  /-- The circumradius is positive -/
  circumradius_pos : 0 < circumradius

/-- There exist two distinct triangles satisfying the given conditions -/
theorem two_triangles_exist (α m r : ℝ) 
  (h_α_pos : 0 < α) (h_α_lt_pi : α < π) 
  (h_m_pos : 0 < m) (h_r_pos : 0 < r) : 
  ∃ (t1 t2 : SpecialTriangle), t1 ≠ t2 ∧ 
    t1.angle = α ∧ t1.height = m ∧ t1.circumradius = r ∧
    t2.angle = α ∧ t2.height = m ∧ t2.circumradius = r := by
  sorry

end two_triangles_exist_l642_64280


namespace gcd_315_2016_l642_64283

theorem gcd_315_2016 : Nat.gcd 315 2016 = 63 := by sorry

end gcd_315_2016_l642_64283


namespace complex_roots_isosceles_triangle_l642_64268

theorem complex_roots_isosceles_triangle (a b z₁ z₂ : ℂ) : 
  z₁^2 + a*z₁ + b = 0 → 
  z₂^2 + a*z₂ + b = 0 → 
  Complex.abs z₁ = Complex.abs (2*z₂) → 
  a^2 / b = 4.5 := by sorry

end complex_roots_isosceles_triangle_l642_64268


namespace shaded_area_is_32_l642_64213

/-- Represents a rectangle in the grid --/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents the grid configuration --/
structure Grid where
  totalWidth : ℕ
  totalHeight : ℕ
  rectangles : List Rectangle

def triangleArea (base height : ℕ) : ℕ :=
  base * height / 2

def rectangleArea (r : Rectangle) : ℕ :=
  r.width * r.height

def totalGridArea (g : Grid) : ℕ :=
  g.rectangles.foldl (fun acc r => acc + rectangleArea r) 0

theorem shaded_area_is_32 (g : Grid) 
    (h1 : g.totalWidth = 16)
    (h2 : g.totalHeight = 8)
    (h3 : g.rectangles = [⟨5, 4⟩, ⟨6, 6⟩, ⟨5, 8⟩])
    (h4 : triangleArea g.totalWidth g.totalHeight = 64) :
    totalGridArea g - triangleArea g.totalWidth g.totalHeight = 32 := by
  sorry

end shaded_area_is_32_l642_64213


namespace mean_proportional_problem_l642_64218

theorem mean_proportional_problem (x : ℝ) : 
  (156 : ℝ) = Real.sqrt (234 * x) → x = 104 := by
  sorry

end mean_proportional_problem_l642_64218


namespace new_person_weight_l642_64253

/-- Given a group of 8 people, if replacing a person weighing 45 kg with a new person
    increases the average weight by 2.5 kg, then the new person weighs 65 kg. -/
theorem new_person_weight (initial_count : Nat) (weight_replaced : ℝ) (avg_increase : ℝ) :
  initial_count = 8 →
  weight_replaced = 45 →
  avg_increase = 2.5 →
  (initial_count : ℝ) * avg_increase + weight_replaced = 65 := by
  sorry

#check new_person_weight

end new_person_weight_l642_64253


namespace angle_cosine_equivalence_l642_64252

-- Define a structure for a triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  angle_sum : A + B + C = π
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

-- Define the theorem
theorem angle_cosine_equivalence (t : Triangle) :
  (t.A > t.B ↔ Real.cos t.A < Real.cos t.B) :=
by sorry

end angle_cosine_equivalence_l642_64252


namespace apples_picked_theorem_l642_64295

/-- The number of apples picked by Mike -/
def mike_apples : ℕ := 7

/-- The number of apples picked by Nancy -/
def nancy_apples : ℕ := 3

/-- The number of apples picked by Keith -/
def keith_apples : ℕ := 6

/-- The total number of apples picked -/
def total_apples : ℕ := mike_apples + nancy_apples + keith_apples

theorem apples_picked_theorem : total_apples = 16 := by
  sorry

end apples_picked_theorem_l642_64295


namespace set_equality_proof_l642_64228

def A : Set ℝ := {x | x^2 - 5*x + 6 = 0}

def B (a : ℝ) : Set ℝ := {x | x < a}

def C : Set ℝ := {2, 3}

theorem set_equality_proof : 
  ∀ a : ℝ, (A ∪ B a = A) ↔ (a ∈ C) :=
by sorry

end set_equality_proof_l642_64228


namespace vector_cosine_and_projection_l642_64231

/-- Given vectors a and b with their components, prove the cosine of the angle between them
    and the scalar projection of a onto b. -/
theorem vector_cosine_and_projection (a b : ℝ × ℝ) (h : a = (3, 1) ∧ b = (-2, 4)) :
  let θ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  (Real.cos θ = -Real.sqrt 2 / 10) ∧
  ((a.1 * b.1 + a.2 * b.2) / (b.1^2 + b.2^2) * Real.sqrt (b.1^2 + b.2^2) = -Real.sqrt 5 / 5) := by
  sorry

end vector_cosine_and_projection_l642_64231


namespace town_distance_interval_l642_64203

def distance_to_town (d : ℝ) : Prop :=
  (¬ (d ≥ 8)) ∧ (¬ (d ≤ 7)) ∧ (¬ (d ≤ 6)) ∧ (d ≠ 5)

theorem town_distance_interval :
  ∀ d : ℝ, distance_to_town d → (7 < d ∧ d < 8) :=
by sorry

end town_distance_interval_l642_64203


namespace f_max_at_three_halves_l642_64206

/-- The quadratic function f(x) = -3x^2 + 9x - 1 -/
def f (x : ℝ) : ℝ := -3 * x^2 + 9 * x - 1

/-- The theorem states that f(x) attains its maximum value when x = 3/2 -/
theorem f_max_at_three_halves :
  ∃ (c : ℝ), c = 3/2 ∧ ∀ (x : ℝ), f x ≤ f c :=
by
  sorry

end f_max_at_three_halves_l642_64206


namespace chocolate_difference_l642_64238

/-- The number of chocolates Robert ate -/
def robert_chocolates : ℕ := 10

/-- The number of chocolates Nickel ate -/
def nickel_chocolates : ℕ := 5

/-- Theorem stating the difference in chocolate consumption -/
theorem chocolate_difference : robert_chocolates - nickel_chocolates = 5 := by
  sorry

end chocolate_difference_l642_64238


namespace sqrt_3_irrational_l642_64264

theorem sqrt_3_irrational : Irrational (Real.sqrt 3) := by
  sorry

end sqrt_3_irrational_l642_64264


namespace quadratic_is_perfect_square_l642_64204

theorem quadratic_is_perfect_square (a : ℚ) : 
  (∃ r s : ℚ, ∀ x, a * x^2 + 26 * x + 9 = (r * x + s)^2) → a = 169 / 9 := by
  sorry

end quadratic_is_perfect_square_l642_64204


namespace circle_center_height_l642_64237

/-- Represents a circle inside a parabola y = 2x^2, tangent at two points -/
structure CircleInParabola where
  /-- x-coordinate of one tangency point -/
  a : ℝ
  /-- y-coordinate of the circle's center -/
  b : ℝ
  /-- Radius of the circle -/
  r : ℝ
  /-- Condition: The circle is tangent to the parabola -/
  tangent : (a^2 + (2*a^2 - b)^2 = r^2) ∧ ((-a)^2 + (2*(-a)^2 - b)^2 = r^2)
  /-- Condition: The circle's center is on the y-axis -/
  center_on_y_axis : True

/-- Theorem: The y-coordinate of the circle's center equals the y-coordinate of the tangency points -/
theorem circle_center_height (c : CircleInParabola) : c.b = 2 * c.a^2 := by
  sorry

end circle_center_height_l642_64237


namespace kerosene_mixture_l642_64234

theorem kerosene_mixture (x : ℝ) : 
  (((6 * (x / 100)) + (4 * 0.3)) / 10 = 0.27) → x = 25 := by
  sorry

end kerosene_mixture_l642_64234


namespace square_field_problem_l642_64247

theorem square_field_problem (a p : ℝ) (x : ℝ) : 
  p = 36 →                           -- perimeter is 36 feet
  a = (p / 4) ^ 2 →                  -- area formula for square
  6 * a = 6 * (2 * p + x) →          -- given equation
  x = 9 := by sorry                  -- prove x = 9

end square_field_problem_l642_64247


namespace seventh_term_is_eight_l642_64217

/-- An arithmetic sequence with given conditions -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  a 1 = 2 ∧
  a 3 + a 4 = 9

/-- Theorem: For an arithmetic sequence satisfying the given conditions, the 7th term is 8 -/
theorem seventh_term_is_eight (a : ℕ → ℝ) (h : ArithmeticSequence a) : a 7 = 8 := by
  sorry

end seventh_term_is_eight_l642_64217


namespace water_consumption_correct_l642_64210

/-- Water consumption per person per year in cubic meters for different regions -/
structure WaterConsumption where
  west : ℝ
  nonWest : ℝ
  russia : ℝ

/-- Given water consumption data -/
def givenData : WaterConsumption :=
  { west := 21428
    nonWest := 26848.55
    russia := 302790.13 }

/-- Theorem stating that the given water consumption data is correct -/
theorem water_consumption_correct (data : WaterConsumption) :
  data.west = givenData.west ∧
  data.nonWest = givenData.nonWest ∧
  data.russia = givenData.russia :=
by sorry

#check water_consumption_correct

end water_consumption_correct_l642_64210


namespace square_minus_a_nonpositive_implies_a_geq_four_l642_64285

theorem square_minus_a_nonpositive_implies_a_geq_four :
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0) → a ≥ 4 := by
  sorry

end square_minus_a_nonpositive_implies_a_geq_four_l642_64285


namespace products_not_equal_l642_64205

def is_valid_table (t : Fin 10 → Fin 10 → ℕ) : Prop :=
  ∀ i j, 102 ≤ t i j ∧ t i j ≤ 201 ∧ (∀ i' j', (i ≠ i' ∨ j ≠ j') → t i j ≠ t i' j')

def row_product (t : Fin 10 → Fin 10 → ℕ) (i : Fin 10) : ℕ :=
  (Finset.univ.prod fun j => t i j)

def col_product (t : Fin 10 → Fin 10 → ℕ) (j : Fin 10) : ℕ :=
  (Finset.univ.prod fun i => t i j)

def row_products (t : Fin 10 → Fin 10 → ℕ) : Finset ℕ :=
  Finset.image (row_product t) Finset.univ

def col_products (t : Fin 10 → Fin 10 → ℕ) : Finset ℕ :=
  Finset.image (col_product t) Finset.univ

theorem products_not_equal :
  ∀ t : Fin 10 → Fin 10 → ℕ, is_valid_table t → row_products t ≠ col_products t :=
sorry

end products_not_equal_l642_64205


namespace hexagon_diagonal_intersection_probability_l642_64249

/-- A convex hexagon -/
structure ConvexHexagon where
  -- Add necessary fields if needed

/-- A diagonal in a convex hexagon -/
structure Diagonal (H : ConvexHexagon) where
  -- Add necessary fields if needed

/-- Predicate to check if two diagonals intersect inside the hexagon -/
def intersect_inside (H : ConvexHexagon) (d1 d2 : Diagonal H) : Prop :=
  sorry

/-- The set of all diagonals in a hexagon -/
def all_diagonals (H : ConvexHexagon) : Set (Diagonal H) :=
  sorry

/-- The number of diagonals in a hexagon -/
def num_diagonals (H : ConvexHexagon) : ℕ :=
  9

/-- The number of pairs of diagonals that intersect inside the hexagon -/
def num_intersecting_pairs (H : ConvexHexagon) : ℕ :=
  15

/-- The probability of two randomly chosen diagonals intersecting inside the hexagon -/
def prob_intersect (H : ConvexHexagon) : ℚ :=
  15 / 36

theorem hexagon_diagonal_intersection_probability (H : ConvexHexagon) :
  prob_intersect H = 5 / 12 :=
sorry

end hexagon_diagonal_intersection_probability_l642_64249


namespace stopping_time_maximizes_distance_l642_64251

/-- The distance function representing the distance traveled by a car after braking. -/
def S (t : ℝ) : ℝ := -3 * t^2 + 18 * t

/-- The time at which the distance function reaches its maximum value. -/
def stopping_time : ℝ := 3

/-- Theorem stating that the stopping time maximizes the distance function. -/
theorem stopping_time_maximizes_distance :
  ∀ t : ℝ, S t ≤ S stopping_time :=
sorry

end stopping_time_maximizes_distance_l642_64251


namespace library_visitors_average_l642_64224

theorem library_visitors_average (monday_visitors : ℕ) (total_visitors : ℕ) :
  monday_visitors = 50 →
  total_visitors = 250 →
  (total_visitors - (monday_visitors + 2 * monday_visitors)) / 5 = 20 :=
by
  sorry

end library_visitors_average_l642_64224


namespace f_composition_negative_two_l642_64288

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 1 - Real.sqrt x else 3^x

theorem f_composition_negative_two : f (f (-2)) = 2/3 := by
  sorry

end f_composition_negative_two_l642_64288


namespace point_location_implies_coordinate_signs_l642_64209

/-- A point in a 2D coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is to the right of the y-axis -/
def isRightOfYAxis (p : Point) : Prop := p.x > 0

/-- Predicate to check if a point is below the x-axis -/
def isBelowXAxis (p : Point) : Prop := p.y < 0

/-- Theorem stating that if a point is to the right of the y-axis and below the x-axis,
    then its x-coordinate is positive and y-coordinate is negative -/
theorem point_location_implies_coordinate_signs (p : Point) :
  isRightOfYAxis p → isBelowXAxis p → p.x > 0 ∧ p.y < 0 := by
  sorry

end point_location_implies_coordinate_signs_l642_64209


namespace at_least_one_third_l642_64291

theorem at_least_one_third (a b c : ℝ) (h : a + b + c = 1) :
  a ≥ 1/3 ∨ b ≥ 1/3 ∨ c ≥ 1/3 := by
  sorry

end at_least_one_third_l642_64291


namespace correct_statements_l642_64212

-- Define a differentiable function
variable (f : ℝ → ℝ) (hf : Differentiable ℝ f)

-- Define extremum
def HasExtremumAt (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∀ x, f x ≤ f x₀ ∨ f x ≥ f x₀

-- Define inductive and deductive reasoning
def InductiveReasoning : Prop :=
  ∃ (specific general : Prop), specific → general

def DeductiveReasoning : Prop :=
  ∃ (general specific : Prop), general → specific

-- Define synthetic and analytic methods
def SyntheticMethod : Prop :=
  ∃ (cause effect : Prop), cause → effect

def AnalyticMethod : Prop :=
  ∃ (effect cause : Prop), effect → cause

-- Theorem statement
theorem correct_statements
  (x₀ : ℝ)
  (h_extremum : HasExtremumAt f x₀) :
  (deriv f x₀ = 0) ∧
  InductiveReasoning ∧
  DeductiveReasoning ∧
  SyntheticMethod ∧
  AnalyticMethod :=
sorry

end correct_statements_l642_64212


namespace problem_solution_l642_64257

theorem problem_solution (c d : ℚ) 
  (eq1 : 5 + c = 6 - d) 
  (eq2 : 6 + d = 10 + c) : 
  5 - c = 13/2 := by
  sorry

end problem_solution_l642_64257


namespace twenty_two_students_remain_l642_64262

/-- The number of remaining students after some leave early -/
def remaining_students (total_groups : ℕ) (students_per_group : ℕ) (students_who_left : ℕ) : ℕ :=
  total_groups * students_per_group - students_who_left

/-- Theorem stating that given 3 groups of 8 students with 2 leaving early, 22 students remain -/
theorem twenty_two_students_remain :
  remaining_students 3 8 2 = 22 := by
  sorry

end twenty_two_students_remain_l642_64262


namespace remainder_proof_l642_64232

theorem remainder_proof : (9^5 + 8^6 + 7^7) % 7 = 5 := by
  sorry

end remainder_proof_l642_64232


namespace puzzle_solution_l642_64299

/-- Given a permutation of the digits 1 to 6, prove that it satisfies the given conditions
    and corresponds to the number 132465 --/
theorem puzzle_solution (E U L S R T : Nat) : 
  ({E, U, L, S, R, T} : Finset Nat) = {1, 2, 3, 4, 5, 6} →
  E + U + L = 6 →
  S + R + U + T = 18 →
  U * T = 15 →
  S * L = 8 →
  E * 100000 + U * 10000 + L * 1000 + S * 100 + R * 10 + T = 132465 :=
by sorry

end puzzle_solution_l642_64299


namespace chris_birthday_money_l642_64258

theorem chris_birthday_money (x : ℕ) : 
  x + 25 + 20 + 75 = 279 → x = 159 := by sorry

end chris_birthday_money_l642_64258


namespace power_of_product_l642_64216

theorem power_of_product (x y : ℝ) : (-2 * x * y^3)^2 = 4 * x^2 * y^6 := by
  sorry

end power_of_product_l642_64216


namespace min_value_expression_l642_64243

theorem min_value_expression (x y z : ℝ) (h1 : 2 ≤ x) (h2 : x ≤ y) (h3 : y ≤ z) (h4 : z ≤ 5) :
  (x - 2)^2 + (y/x - 1)^2 + (z/y - 1)^2 + (5/z - 1)^2 ≥ 9 ∧
  ∃ x y z : ℝ, 2 ≤ x ∧ x ≤ y ∧ y ≤ z ∧ z ≤ 5 ∧
    (x - 2)^2 + (y/x - 1)^2 + (z/y - 1)^2 + (5/z - 1)^2 = 9 :=
by sorry

end min_value_expression_l642_64243


namespace diana_garden_area_l642_64255

/-- Represents a rectangular garden with fence posts --/
structure Garden where
  total_posts : ℕ
  post_distance : ℝ
  short_side_posts : ℕ
  long_side_posts : ℕ

/-- Calculates the area of the garden --/
def garden_area (g : Garden) : ℝ :=
  (g.short_side_posts - 1) * g.post_distance * (g.long_side_posts - 1) * g.post_distance

/-- Theorem stating the area of Diana's garden --/
theorem diana_garden_area :
  ∀ g : Garden,
  g.total_posts = 24 ∧
  g.post_distance = 3 ∧
  g.long_side_posts = (3 * g.short_side_posts + 1) / 2 ∧
  2 * g.short_side_posts + 2 * g.long_side_posts - 4 = g.total_posts →
  garden_area g = 135 := by
  sorry

end diana_garden_area_l642_64255


namespace quadratic_function_properties_l642_64244

/-- Given a quadratic function f(x) = ax^2 + bx + c with specific conditions,
    prove properties about its coefficients, roots, and values. -/
theorem quadratic_function_properties
  (a b c : ℝ) (m₁ m₂ : ℝ)
  (h_order : a > b ∧ b > c)
  (h_points : a^2 + (a * m₁^2 + b * m₁ + c + a * m₂^2 + b * m₂ + c) * a +
              (a * m₁^2 + b * m₁ + c) * (a * m₂^2 + b * m₂ + c) = 0)
  (h_root : a + b + c = 0) :
  (b ≥ 0) ∧
  (2 ≤ |1 - c/a| ∧ |1 - c/a| < 3) ∧
  (max (a * (m₁ + 3)^2 + b * (m₁ + 3) + c) (a * (m₂ + 3)^2 + b * (m₂ + 3) + c) > 0) :=
by sorry

end quadratic_function_properties_l642_64244


namespace hundred_three_square_partitions_l642_64201

/-- A function that returns the number of ways to write a given number as the sum of three positive perfect squares, where the order doesn't matter. -/
def count_three_square_partitions (n : ℕ) : ℕ :=
  sorry

/-- The theorem stating that there is exactly one way to write 100 as the sum of three positive perfect squares, where the order doesn't matter. -/
theorem hundred_three_square_partitions : count_three_square_partitions 100 = 1 := by
  sorry

end hundred_three_square_partitions_l642_64201


namespace burger_share_length_l642_64215

-- Define the length of a foot in inches
def foot_in_inches : ℕ := 12

-- Define the burger length in feet
def burger_length_feet : ℕ := 1

-- Define the number of people sharing the burger
def num_people : ℕ := 2

-- Theorem to prove
theorem burger_share_length :
  (burger_length_feet * foot_in_inches) / num_people = 6 := by
  sorry

end burger_share_length_l642_64215


namespace modular_arithmetic_equivalence_l642_64240

theorem modular_arithmetic_equivalence : 144 * 20 - 17^2 + 5 ≡ 4 [ZMOD 16] := by
  sorry

end modular_arithmetic_equivalence_l642_64240


namespace sum_c_d_equals_three_l642_64236

theorem sum_c_d_equals_three (a b c d : ℝ)
  (h1 : a + b = 12)
  (h2 : b + c = 9)
  (h3 : a + d = 6) :
  c + d = 3 := by
sorry

end sum_c_d_equals_three_l642_64236


namespace arithmetic_sequence_nth_term_l642_64271

theorem arithmetic_sequence_nth_term (x : ℚ) (n : ℕ) : 
  (3*x - 5 : ℚ) = (7*x - 17) - ((7*x - 17) - (3*x - 5)) → 
  (7*x - 17 : ℚ) = (4*x + 3) - ((4*x + 3) - (7*x - 17)) → 
  (∃ a d : ℚ, a = 3*x - 5 ∧ d = (7*x - 17) - (3*x - 5) ∧ 
    a + (n - 1) * d = 4033) → 
  n = 641 := by
sorry

end arithmetic_sequence_nth_term_l642_64271


namespace two_hearts_three_different_probability_l642_64269

/-- A standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (hearts : Nat)
  (other_suits : Nat)
  (cards_eq : cards = 52)
  (hearts_eq : hearts = 13)
  (other_suits_eq : other_suits = 39)

/-- The probability of the specified event -/
def probability_two_hearts_three_different (d : Deck) : ℚ :=
  135 / 1024

/-- Theorem statement -/
theorem two_hearts_three_different_probability (d : Deck) :
  probability_two_hearts_three_different d = 135 / 1024 := by
  sorry

end two_hearts_three_different_probability_l642_64269


namespace dividend_proof_l642_64221

theorem dividend_proof : (10918788 : ℕ) / 12 = 909899 := by
  sorry

end dividend_proof_l642_64221


namespace least_8bit_number_proof_l642_64223

/-- The least positive base-10 number requiring 8 binary digits -/
def least_8bit_number : ℕ := 128

/-- Convert a natural number to its binary representation -/
def to_binary (n : ℕ) : List Bool := sorry

/-- Count the number of digits in a binary representation -/
def binary_digit_count (n : ℕ) : ℕ := (to_binary n).length

theorem least_8bit_number_proof :
  (∀ m : ℕ, m < least_8bit_number → binary_digit_count m < 8) ∧
  binary_digit_count least_8bit_number = 8 := by sorry

end least_8bit_number_proof_l642_64223


namespace binary_1110011_is_115_l642_64233

def binary_to_decimal (binary_digits : List Bool) : ℕ :=
  binary_digits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

theorem binary_1110011_is_115 :
  binary_to_decimal [true, true, false, false, true, true, true] = 115 := by
  sorry

end binary_1110011_is_115_l642_64233


namespace alcohol_solution_concentration_l642_64286

theorem alcohol_solution_concentration 
  (initial_volume : ℝ)
  (initial_percentage : ℝ)
  (added_alcohol : ℝ)
  (h1 : initial_volume = 6)
  (h2 : initial_percentage = 0.4)
  (h3 : added_alcohol = 1.2) :
  let final_volume := initial_volume + added_alcohol
  let initial_alcohol := initial_volume * initial_percentage
  let final_alcohol := initial_alcohol + added_alcohol
  let final_percentage := final_alcohol / final_volume
  final_percentage = 0.5 := by sorry

end alcohol_solution_concentration_l642_64286


namespace power_of_product_l642_64276

theorem power_of_product (a b : ℝ) : (a * b^3)^2 = a^2 * b^6 := by
  sorry

end power_of_product_l642_64276


namespace point_on_line_l642_64219

/-- For any point (m,n) on the line y = 2x + 1, 2m - n = -1 -/
theorem point_on_line (m n : ℝ) : n = 2 * m + 1 → 2 * m - n = -1 := by
  sorry

end point_on_line_l642_64219


namespace linear_equation_solution_l642_64261

/-- If the equation (m+1)x^2 + 2mx + 1 = 0 is linear with respect to x, then its solution is 1/2. -/
theorem linear_equation_solution (m : ℝ) : 
  (m + 1 = 0) → (2*m ≠ 0) → 
  ∃ (x : ℝ), ((m + 1) * x^2 + 2*m*x + 1 = 0) ∧ (x = 1/2) :=
by sorry

end linear_equation_solution_l642_64261


namespace pure_imaginary_complex_number_l642_64260

theorem pure_imaginary_complex_number (a : ℝ) : 
  (Complex.mk (a^2 - 4) (a - 2)).im ≠ 0 ∧ (Complex.mk (a^2 - 4) (a - 2)).re = 0 → a = -2 := by
  sorry

end pure_imaginary_complex_number_l642_64260


namespace factorial_difference_quotient_l642_64250

theorem factorial_difference_quotient (n : ℕ) (h : n ≥ 8) :
  (Nat.factorial (n + 3) - Nat.factorial (n + 2)) / Nat.factorial n = n^2 + 3*n + 2 := by
  sorry

end factorial_difference_quotient_l642_64250


namespace sequence_arrangement_count_l642_64225

theorem sequence_arrangement_count : ℕ :=
  let n : ℕ := 40
  let k : ℕ := 31
  let m : ℕ := 20
  Nat.choose n (n - k) * Nat.factorial (k - 2) * Nat.factorial (n - k)

#check sequence_arrangement_count

end sequence_arrangement_count_l642_64225


namespace circle_radius_l642_64220

theorem circle_radius (x y : ℝ) : 
  (x^2 + y^2 - 8 = 2*x + 4*y) → 
  ∃ (center_x center_y : ℝ), (x - center_x)^2 + (y - center_y)^2 = 13 :=
by
  sorry

end circle_radius_l642_64220


namespace wire_around_square_field_l642_64241

/-- Proves that a wire of length 15840 m goes around a square field of area 69696 m^2 exactly 15 times -/
theorem wire_around_square_field (field_area : ℝ) (wire_length : ℝ) : 
  field_area = 69696 → wire_length = 15840 → 
  (wire_length / (4 * Real.sqrt field_area) : ℝ) = 15 := by
  sorry

end wire_around_square_field_l642_64241


namespace ice_cream_sales_l642_64277

def daily_sales : List ℝ := [100, 92, 109, 96, 0, 96, 105]

theorem ice_cream_sales (x : ℝ) :
  let sales := daily_sales.set 4 x
  sales.length = 7 ∧ 
  sales.sum / sales.length = 100.1 →
  x = 102.7 := by
sorry

end ice_cream_sales_l642_64277


namespace mans_rate_l642_64287

/-- The man's rate in still water given his speeds with and against the stream -/
theorem mans_rate (speed_with_stream speed_against_stream : ℝ) 
  (h1 : speed_with_stream = 18)
  (h2 : speed_against_stream = 8) :
  (speed_with_stream + speed_against_stream) / 2 = 13 := by
  sorry

end mans_rate_l642_64287


namespace sum_of_coefficients_l642_64282

theorem sum_of_coefficients (a c : ℚ) : 
  (3 : ℚ) ∈ {x | a * x^2 - 6 * x + c = 0} →
  (1/3 : ℚ) ∈ {x | a * x^2 - 6 * x + c = 0} →
  a + c = 18/5 := by
sorry

end sum_of_coefficients_l642_64282


namespace regular_triangular_pyramid_volume_l642_64296

/-- Volume of a regular triangular pyramid -/
theorem regular_triangular_pyramid_volume
  (a b : ℝ) (h_positive : 0 < a ∧ 0 < b) (h_height_constraint : a * Real.sqrt 2 / 2 ≤ b ∧ b < a * Real.sqrt 3 / 2) :
  ∃ V : ℝ, V = (a^3 * b) / (12 * Real.sqrt (3 * a^2 - 4 * b^2)) ∧ V > 0 := by
  sorry

end regular_triangular_pyramid_volume_l642_64296


namespace no_perfect_cube_in_range_l642_64278

theorem no_perfect_cube_in_range : 
  ¬ ∃ (n : ℤ), 4 ≤ n ∧ n ≤ 11 ∧ ∃ (k : ℤ), n^2 + 3*n + 2 = k^3 := by
  sorry

end no_perfect_cube_in_range_l642_64278


namespace sum_base4_equals_1232_l642_64248

/-- Converts a base 4 number represented as a list of digits to its decimal equivalent -/
def base4ToDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 4 * acc + d) 0

/-- The sum of 111₄, 323₄, and 132₄ is equal to 1232₄ in base 4 -/
theorem sum_base4_equals_1232 :
  let a := base4ToDecimal [1, 1, 1]
  let b := base4ToDecimal [3, 2, 3]
  let c := base4ToDecimal [1, 3, 2]
  let sum := base4ToDecimal [1, 2, 3, 2]
  a + b + c = sum := by
  sorry

end sum_base4_equals_1232_l642_64248


namespace max_b_value_l642_64281

theorem max_b_value (a b : ℤ) (h : (127 : ℚ) / a - (16 : ℚ) / b = 1) : b ≤ 2016 := by
  sorry

end max_b_value_l642_64281


namespace power_of_m_divisible_by_24_l642_64254

theorem power_of_m_divisible_by_24 (m : ℕ+) 
  (h1 : ∃ k : ℕ, (24 : ℕ) ∣ m^k)
  (h2 : (8 : ℕ) = Nat.gcd m ((Finset.range m).sup (λ i => Nat.gcd m i))) : 
  (∀ k < 1, ¬((24 : ℕ) ∣ m^k)) ∧ ((24 : ℕ) ∣ m^1) :=
sorry

end power_of_m_divisible_by_24_l642_64254


namespace product_equals_3408_l642_64242

theorem product_equals_3408 : 213 * 16 = 3408 := by
  sorry

end product_equals_3408_l642_64242


namespace quadratic_root_implies_q_value_l642_64272

theorem quadratic_root_implies_q_value (p q : ℝ) (h : Complex.I ^ 2 = -1) :
  (3 * (1 + 4 * Complex.I) ^ 2 + p * (1 + 4 * Complex.I) + q = 0) → q = 51 := by
  sorry

end quadratic_root_implies_q_value_l642_64272


namespace two_correct_statements_l642_64259

theorem two_correct_statements (a b : ℝ) 
  (h : (a - Real.sqrt (a^2 - 1)) * (b - Real.sqrt (b^2 - 1)) = 1) :
  (a = b ∧ a * b = 1) ∧ 
  (a + b ≠ 0 ∧ a * b ≠ -1) :=
by sorry

end two_correct_statements_l642_64259


namespace euler_family_mean_age_l642_64200

def euler_family_children : ℕ := 7
def girls_aged_8 : ℕ := 4
def boys_aged_11 : ℕ := 2
def girl_aged_16 : ℕ := 1

def total_age : ℕ := girls_aged_8 * 8 + boys_aged_11 * 11 + girl_aged_16 * 16

theorem euler_family_mean_age :
  (total_age : ℚ) / euler_family_children = 10 := by sorry

end euler_family_mean_age_l642_64200


namespace polynomial_divisibility_l642_64229

def p (x : ℝ) : ℝ := 2*x^3 - 6*x^2 + 6*x - 18

theorem polynomial_divisibility :
  (∃ q : ℝ → ℝ, p = fun x ↦ (x - 3) * q x) ∧
  (∃ r : ℝ → ℝ, p = fun x ↦ (2*x^2 + 6) * r x) := by
  sorry

end polynomial_divisibility_l642_64229


namespace karen_graded_eight_tests_l642_64275

/-- Represents the bonus calculation for a teacher based on test scores. -/
def bonus_calculation (n : ℕ) : Prop :=
  let base_bonus := 500
  let extra_bonus_per_point := 10
  let base_threshold := 75
  let max_score := 150
  let current_average := 70
  let last_two_tests_score := 290
  let target_bonus := 600
  let total_current_points := n * current_average
  let total_points_after := total_current_points + last_two_tests_score
  let final_average := total_points_after / (n + 2)
  (final_average > base_threshold) ∧
  (target_bonus = base_bonus + (final_average - base_threshold) * extra_bonus_per_point) ∧
  (∀ m : ℕ, m ≤ n + 2 → m * max_score ≥ total_points_after)

/-- Theorem stating that Karen has graded 8 tests. -/
theorem karen_graded_eight_tests : ∃ (n : ℕ), bonus_calculation n ∧ n = 8 :=
  sorry

end karen_graded_eight_tests_l642_64275


namespace inequality_proof_l642_64263

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 * (a - b)/(a + b) + b^2 * (b - c)/(b + c) + c^2 * (c - a)/(c + a) ≥ 0 ∧
  (a^2 * (a - b)/(a + b) + b^2 * (b - c)/(b + c) + c^2 * (c - a)/(c + a) = 0 ↔ a = b ∧ b = c) :=
by sorry

end inequality_proof_l642_64263


namespace min_distance_to_tangent_point_l642_64289

/-- The minimum distance from a point on the line y = x + 1 to a tangent point 
    on the circle (x - 3)² + y² = 1 is √7. -/
theorem min_distance_to_tangent_point : 
  ∃ (P : ℝ × ℝ) (T : ℝ × ℝ),
    (P.2 = P.1 + 1) ∧ 
    ((T.1 - 3)^2 + T.2^2 = 1) ∧
    (∀ (Q : ℝ × ℝ), (Q.2 = Q.1 + 1) → (Q.1 - 3)^2 + Q.2^2 = 1 → 
      dist P T ≤ dist Q T) ∧
    dist P T = Real.sqrt 7 := by
  sorry


end min_distance_to_tangent_point_l642_64289


namespace max_value_implies_a_l642_64208

def f (a : ℝ) (x : ℝ) : ℝ := -4 * x^2 + 4 * a * x - 4 * a - a^2

theorem max_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f a x ≤ -5) ∧
  (∃ x ∈ Set.Icc 0 1, f a x = -5) →
  a = 5/4 ∨ a = -5 := by
  sorry

end max_value_implies_a_l642_64208


namespace min_S_value_l642_64297

noncomputable def S (x y z : ℝ) : ℝ := (z + 1)^2 / (2 * x * y * z)

theorem min_S_value (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_constraint : x^2 + y^2 + z^2 = 1) :
  (∀ x' y' z' : ℝ, x' > 0 → y' > 0 → z' > 0 → x'^2 + y'^2 + z'^2 = 1 → 
    S x y z ≤ S x' y' z') →
  x = Real.sqrt (Real.sqrt 2 - 1) :=
sorry

end min_S_value_l642_64297


namespace min_distance_ellipse_line_l642_64274

/-- The minimum distance between a point on the ellipse x²/8 + y²/4 = 1 
    and a point on the line x - √2 y - 5 = 0 is √3/3 -/
theorem min_distance_ellipse_line : 
  ∃ (d : ℝ), d = Real.sqrt 3 / 3 ∧ 
  ∀ (P Q : ℝ × ℝ), 
    (P.1^2 / 8 + P.2^2 / 4 = 1) → 
    (Q.1 - Real.sqrt 2 * Q.2 - 5 = 0) → 
    Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≥ d ∧
    ∃ (P₀ Q₀ : ℝ × ℝ), 
      (P₀.1^2 / 8 + P₀.2^2 / 4 = 1) ∧
      (Q₀.1 - Real.sqrt 2 * Q₀.2 - 5 = 0) ∧
      Real.sqrt ((P₀.1 - Q₀.1)^2 + (P₀.2 - Q₀.2)^2) = d :=
by sorry

end min_distance_ellipse_line_l642_64274


namespace class_size_l642_64266

/-- The number of boys in the class -/
def n : ℕ := sorry

/-- The initial (incorrect) average weight -/
def initial_avg : ℚ := 584/10

/-- The correct average weight -/
def correct_avg : ℚ := 587/10

/-- The difference between the correct and misread weight -/
def weight_diff : ℚ := 62 - 56

theorem class_size :
  (n : ℚ) * initial_avg + weight_diff = n * correct_avg ∧ n = 20 := by sorry

end class_size_l642_64266


namespace sin_squared_sum_l642_64292

theorem sin_squared_sum (α : ℝ) : 
  Real.sin α ^ 2 + Real.sin (α + Real.pi / 3) ^ 2 + Real.sin (α + 2 * Real.pi / 3) ^ 2 = 3 / 2 := by
  sorry

end sin_squared_sum_l642_64292


namespace f_neg_two_eq_four_l642_64284

noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log (1/2)

def symmetric_about_y_eq_x (f g : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ g y = x

theorem f_neg_two_eq_four 
  (f : ℝ → ℝ) 
  (h : symmetric_about_y_eq_x f g) : 
  f (-2) = 4 := by
sorry

end f_neg_two_eq_four_l642_64284


namespace boat_upstream_distance_l642_64245

/-- Proves that given a boat with speed 36 kmph in still water and a stream with speed 12 kmph,
    if the boat covers 80 km downstream in the same time as it covers a certain distance upstream,
    then that upstream distance is 40 km. -/
theorem boat_upstream_distance
  (boat_speed : ℝ)
  (stream_speed : ℝ)
  (downstream_distance : ℝ)
  (h1 : boat_speed = 36)
  (h2 : stream_speed = 12)
  (h3 : downstream_distance = 80)
  (h4 : downstream_distance / (boat_speed + stream_speed) =
        upstream_distance / (boat_speed - stream_speed)) :
  upstream_distance = 40 :=
sorry

end boat_upstream_distance_l642_64245


namespace weight_replaced_is_75_l642_64270

/-- The weight of the replaced person in a group, given the following conditions:
  * There are 7 persons initially
  * The average weight increases by 3.5 kg when a new person replaces one of them
  * The weight of the new person is 99.5 kg
-/
def weight_of_replaced_person (num_persons : ℕ) (avg_weight_increase : ℝ) (new_person_weight : ℝ) : ℝ :=
  new_person_weight - (num_persons * avg_weight_increase)

/-- Theorem stating that the weight of the replaced person is 75 kg -/
theorem weight_replaced_is_75 :
  weight_of_replaced_person 7 3.5 99.5 = 75 := by sorry

end weight_replaced_is_75_l642_64270


namespace parabola_point_range_l642_64273

/-- Parabola type representing y² = 8x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ
  directrix : ℝ → ℝ → Prop

/-- Point on a parabola -/
structure PointOnParabola (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : p.equation x y

/-- Circle type -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Predicate to check if a circle intersects a line -/
def circle_intersects_line (c : Circle) (l : ℝ → ℝ → Prop) : Prop :=
  ∃ x y, l x y ∧ ((x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2)

/-- Main theorem -/
theorem parabola_point_range (p : Parabola) (m : PointOnParabola p) :
  let c : Circle := { center := p.focus, radius := Real.sqrt ((m.x - p.focus.1)^2 + (m.y - p.focus.2)^2) }
  circle_intersects_line c p.directrix → m.x > 2 := by
  sorry


end parabola_point_range_l642_64273


namespace money_fraction_after_two_years_l642_64267

/-- The simple interest rate per annum as a decimal -/
def interest_rate : ℝ := 0.08333333333333337

/-- The time period in years -/
def time_period : ℝ := 2

/-- The fraction of the sum of money after the given time period -/
def money_fraction : ℝ := 1 + interest_rate * time_period

theorem money_fraction_after_two_years :
  money_fraction = 1.1666666666666667 := by sorry

end money_fraction_after_two_years_l642_64267


namespace sqrt_square_nine_l642_64293

theorem sqrt_square_nine : Real.sqrt 9 ^ 2 = 9 := by
  sorry

end sqrt_square_nine_l642_64293


namespace compare_absolute_values_l642_64294

theorem compare_absolute_values (m n : ℝ) 
  (h1 : m * n < 0) 
  (h2 : m + n < 0) 
  (h3 : n > 0) : 
  |m| > |n| := by
  sorry

end compare_absolute_values_l642_64294


namespace expand_and_simplify_l642_64265

theorem expand_and_simplify (a : ℝ) : 
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = 
  a^5 + 19*a^4 + 137*a^3 + 461*a^2 + 702*a + 360 := by
  sorry

end expand_and_simplify_l642_64265


namespace middle_number_problem_l642_64207

theorem middle_number_problem (a b c : ℕ) : 
  a < b ∧ b < c ∧ 
  a + b = 16 ∧ 
  a + c = 21 ∧ 
  b + c = 27 → 
  b = 11 := by
sorry

end middle_number_problem_l642_64207
