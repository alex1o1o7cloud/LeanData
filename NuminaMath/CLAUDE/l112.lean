import Mathlib

namespace equation_describes_parabola_l112_11268

/-- Represents a conic section type -/
inductive ConicSection
  | Circle
  | Parabola
  | Ellipse
  | Hyperbola
  | None

/-- Determines the type of conic section based on the equation |y-3| = √((x+4)² + y²) -/
def determineConicSection : ConicSection := by sorry

/-- Theorem stating that the equation |y-3| = √((x+4)² + y²) describes a parabola -/
theorem equation_describes_parabola : determineConicSection = ConicSection.Parabola := by sorry

end equation_describes_parabola_l112_11268


namespace simplified_expression_l112_11225

theorem simplified_expression (x : ℝ) :
  Real.sqrt (4 * x^2 - 8 * x + 4) + Real.sqrt (4 * x^2 + 8 * x + 4) + 5 =
  2 * |x - 1| + 2 * |x + 1| + 5 := by
  sorry

end simplified_expression_l112_11225


namespace f_is_even_l112_11209

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = -g x

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem f_is_even (g : ℝ → ℝ) (h : is_odd_function g) :
  is_even_function (fun x ↦ |g (x^5)|) :=
by sorry

end f_is_even_l112_11209


namespace candidate_marks_l112_11273

theorem candidate_marks (max_marks : ℝ) (pass_percentage : ℝ) (fail_margin : ℕ) 
  (h1 : max_marks = 152.38)
  (h2 : pass_percentage = 0.42)
  (h3 : fail_margin = 22) : 
  ∃ (secured_marks : ℕ), secured_marks = 42 :=
by
  sorry

end candidate_marks_l112_11273


namespace max_value_of_f_l112_11234

/-- The function we're analyzing -/
def f (x : ℝ) : ℝ := -4 * x^2 + 8 * x + 3

/-- The domain of x -/
def X : Set ℝ := Set.Ioo 0 3

theorem max_value_of_f :
  ∃ (M : ℝ), M = 7 ∧ ∀ x ∈ X, f x ≤ M ∧ ∃ x₀ ∈ X, f x₀ = M :=
by sorry

end max_value_of_f_l112_11234


namespace binomial_p_value_l112_11222

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  ξ : ℝ

/-- The expected value of a binomial random variable -/
def expectedValue (X : BinomialRV) : ℝ := X.n * X.p

/-- The variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

/-- Theorem: If a binomial random variable has E[ξ] = 8 and D[ξ] = 1.6, then p = 0.8 -/
theorem binomial_p_value (X : BinomialRV) 
  (h1 : expectedValue X = 8) 
  (h2 : variance X = 1.6) : 
  X.p = 0.8 := by
  sorry

end binomial_p_value_l112_11222


namespace quadratic_equation_properties_l112_11217

theorem quadratic_equation_properties (m : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 + (m-2)*x - m
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  (∀ (y : ℝ), f y = 0 → y = x₁ ∨ y = x₂) ∧
  x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ x₁ - x₂ = 5/2 :=
by
  sorry

end quadratic_equation_properties_l112_11217


namespace plane_equation_proof_l112_11219

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a vector in 3D space -/
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the vector between two points -/
def vectorBetweenPoints (p1 p2 : Point3D) : Vector3D :=
  { x := p2.x - p1.x
    y := p2.y - p1.y
    z := p2.z - p1.z }

/-- Checks if a vector is perpendicular to a plane -/
def isPerpendicularToPlane (v : Vector3D) (a b c : ℝ) : Prop :=
  a * v.x + b * v.y + c * v.z = 0

/-- Checks if a point lies on a plane -/
def isPointOnPlane (p : Point3D) (a b c d : ℝ) : Prop :=
  a * p.x + b * p.y + c * p.z + d = 0

theorem plane_equation_proof (A B C : Point3D) 
    (h1 : A.x = -4 ∧ A.y = -2 ∧ A.z = 5)
    (h2 : B.x = 3 ∧ B.y = -3 ∧ B.z = -7)
    (h3 : C.x = 9 ∧ C.y = 3 ∧ C.z = -7) :
    let BC := vectorBetweenPoints B C
    isPerpendicularToPlane BC 1 1 0 ∧ isPointOnPlane A 1 1 0 (-6) := by
  sorry

#check plane_equation_proof

end plane_equation_proof_l112_11219


namespace central_circle_radius_l112_11202

/-- The radius of a circle tangent to six semicircles evenly arranged inside a regular hexagon -/
theorem central_circle_radius (side_length : ℝ) (h : side_length = 3) :
  let apothem := side_length * (Real.sqrt 3 / 2)
  let semicircle_radius := side_length / 2
  let central_radius := apothem - semicircle_radius
  central_radius = (3 * (Real.sqrt 3 - 1)) / 2 := by
  sorry

end central_circle_radius_l112_11202


namespace basketball_free_throws_l112_11232

theorem basketball_free_throws :
  ∀ (two_points three_points free_throws : ℕ),
    3 * three_points = 2 * two_points →
    free_throws = 2 * two_points - 1 →
    2 * two_points + 3 * three_points + free_throws = 71 →
    free_throws = 23 := by
  sorry

end basketball_free_throws_l112_11232


namespace equation_solution_l112_11237

theorem equation_solution (a b : ℝ) (h : (a^2 * b^2) / (a^4 - 2*b^4) = 1) :
  (a^2 - b^2) / (a^2 + b^2) = 1/3 :=
by sorry

end equation_solution_l112_11237


namespace distance_circle_center_to_line_l112_11244

/-- The distance from the center of the circle (x+4)^2 + (y-3)^2 = 9 to the line 4x + 3y - 1 = 0 is 8/5 -/
theorem distance_circle_center_to_line : 
  let circle := fun (x y : ℝ) => (x + 4)^2 + (y - 3)^2 = 9
  let line := fun (x y : ℝ) => 4*x + 3*y - 1 = 0
  let center := (-4, 3)
  abs (4 * center.1 + 3 * center.2 - 1) / Real.sqrt (4^2 + 3^2) = 8/5 := by
sorry

end distance_circle_center_to_line_l112_11244


namespace prob_all_cocaptains_l112_11286

def team_sizes : List Nat := [4, 6, 7, 9]
def num_teams : Nat := 4
def num_cocaptains : Nat := 3

def prob_select_cocaptains (n : Nat) : Rat :=
  6 / (n * (n - 1) * (n - 2))

theorem prob_all_cocaptains : 
  (1 : Rat) / num_teams * (team_sizes.map prob_select_cocaptains).sum = 143 / 1680 := by
  sorry

end prob_all_cocaptains_l112_11286


namespace feb_first_is_wednesday_l112_11226

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to get the next day
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

-- Define a function to get the previous day
def prevDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Saturday
  | DayOfWeek.Monday => DayOfWeek.Sunday
  | DayOfWeek.Tuesday => DayOfWeek.Monday
  | DayOfWeek.Wednesday => DayOfWeek.Tuesday
  | DayOfWeek.Thursday => DayOfWeek.Wednesday
  | DayOfWeek.Friday => DayOfWeek.Thursday
  | DayOfWeek.Saturday => DayOfWeek.Friday

-- Define a function to go back n days
def goBackDays (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | Nat.succ m => prevDay (goBackDays d m)

-- Theorem statement
theorem feb_first_is_wednesday (h : DayOfWeek) :
  h = DayOfWeek.Tuesday → goBackDays h 27 = DayOfWeek.Wednesday :=
by
  sorry

end feb_first_is_wednesday_l112_11226


namespace sqrt_x_plus_5_equals_3_l112_11241

theorem sqrt_x_plus_5_equals_3 (x : ℝ) : 
  Real.sqrt (x + 5) = 3 → (x + 5)^2 = 81 := by sorry

end sqrt_x_plus_5_equals_3_l112_11241


namespace solve_for_k_l112_11262

theorem solve_for_k (k : ℝ) (h1 : k ≠ 0) :
  (∀ x : ℝ, (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 6)) → k = 6 := by
  sorry

end solve_for_k_l112_11262


namespace given_point_in_fourth_quadrant_l112_11247

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def is_in_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The given point -/
def given_point : Point :=
  { x := 1, y := -2 }

/-- Theorem: The given point is in the fourth quadrant -/
theorem given_point_in_fourth_quadrant :
  is_in_fourth_quadrant given_point := by
  sorry

end given_point_in_fourth_quadrant_l112_11247


namespace equation_solution_l112_11221

theorem equation_solution :
  let f (x : ℝ) := (x - 4)^4 + (x - 6)^4
  ∃ x₁ x₂ : ℝ, 
    (f x₁ = 240 ∧ f x₂ = 240) ∧
    x₁ = 5 + Real.sqrt (5 * Real.sqrt 2 - 3) ∧
    x₂ = 5 - Real.sqrt (5 * Real.sqrt 2 - 3) ∧
    ∀ x : ℝ, f x = 240 → (x = x₁ ∨ x = x₂) :=
by
  sorry

end equation_solution_l112_11221


namespace three_W_seven_equals_thirteen_l112_11216

/-- Definition of operation W -/
def W (x y : ℤ) : ℤ := y + 5*x - x^2

/-- Theorem: 3W7 equals 13 -/
theorem three_W_seven_equals_thirteen : W 3 7 = 13 := by
  sorry

end three_W_seven_equals_thirteen_l112_11216


namespace root_product_equation_l112_11290

theorem root_product_equation (p q : ℝ) (α β γ δ : ℂ) : 
  (α^2 + p*α + 2 = 0) →
  (β^2 + p*β + 2 = 0) →
  (γ^2 + q*γ + 2 = 0) →
  (δ^2 + q*δ + 2 = 0) →
  (α - γ)*(β - γ)*(α + δ)*(β + δ) = 4 + 2*(p^2 - q^2) :=
by sorry

end root_product_equation_l112_11290


namespace pie_eating_contest_l112_11265

theorem pie_eating_contest (student1_session1 student1_session2 student2_session1 student2_session2 : ℚ)
  (h1 : student1_session1 = 7/8)
  (h2 : student1_session2 = 3/4)
  (h3 : student2_session1 = 5/6)
  (h4 : student2_session2 = 2/3) :
  (student1_session1 + student1_session2) - (student2_session1 + student2_session2) = 1/8 := by
  sorry

end pie_eating_contest_l112_11265


namespace square_sum_of_solution_l112_11283

theorem square_sum_of_solution (x y : ℝ) 
  (h1 : x * y = 6)
  (h2 : x^2 - y^2 + x + y = 44) : 
  x^2 + y^2 = 109 := by
sorry

end square_sum_of_solution_l112_11283


namespace simplify_expression_l112_11210

theorem simplify_expression (y : ℝ) : y - 3*(2+y) + 4*(2-y) - 5*(2+3*y) = -21*y - 8 := by
  sorry

end simplify_expression_l112_11210


namespace arrange_five_balls_three_boxes_l112_11207

/-- The number of ways to put n distinguishable objects into k distinguishable containers -/
def arrange_objects (n k : ℕ) : ℕ := k^n

/-- Theorem: There are 243 ways to put 5 distinguishable balls into 3 distinguishable boxes -/
theorem arrange_five_balls_three_boxes : arrange_objects 5 3 = 243 := by
  sorry

end arrange_five_balls_three_boxes_l112_11207


namespace delta_quotient_on_curve_l112_11284

/-- Given a point (1,3) on the curve y = x^2 + 2, and a nearby point (1 + Δx, 3 + Δy) on the same curve,
    prove that Δy / Δx = 2 + Δx. -/
theorem delta_quotient_on_curve (Δx Δy : ℝ) : 
  (3 + Δy = (1 + Δx)^2 + 2) → (Δy / Δx = 2 + Δx) := by
  sorry

end delta_quotient_on_curve_l112_11284


namespace chocolate_difference_l112_11215

/-- The number of chocolates Nick has -/
def nick_chocolates : ℕ := 10

/-- The factor by which Alix's chocolates exceed Nick's -/
def alix_factor : ℕ := 3

/-- The number of chocolates mom took from Alix -/
def mom_took : ℕ := 5

/-- The number of chocolates Alix has after mom took some -/
def alix_chocolates : ℕ := alix_factor * nick_chocolates - mom_took

theorem chocolate_difference : alix_chocolates - nick_chocolates = 15 := by
  sorry

end chocolate_difference_l112_11215


namespace descending_order_always_possible_ascending_order_sometimes_impossible_l112_11274

-- Define the grid
def Grid := Fin 10 → Fin 10 → Bool

-- Define piece sizes
inductive PieceSize
| One
| Two
| Three
| Four

-- Define a piece
structure Piece where
  size : PieceSize
  position : Fin 10 × Fin 10
  horizontal : Bool

-- Define the set of pieces
def PieceSet := List Piece

-- Check if a placement is valid
def is_valid_placement (grid : Grid) (pieces : PieceSet) : Prop := sorry

-- Define descending order placement
def descending_order_placement (grid : Grid) (pieces : PieceSet) : Prop := sorry

-- Define ascending order placement
def ascending_order_placement (grid : Grid) (pieces : PieceSet) : Prop := sorry

-- Theorem for descending order placement
theorem descending_order_always_possible (grid : Grid) (pieces : PieceSet) :
  descending_order_placement grid pieces → is_valid_placement grid pieces := by sorry

-- Theorem for ascending order placement
theorem ascending_order_sometimes_impossible : 
  ∃ (grid : Grid) (pieces : PieceSet), 
    ascending_order_placement grid pieces ∧ ¬is_valid_placement grid pieces := by sorry

end descending_order_always_possible_ascending_order_sometimes_impossible_l112_11274


namespace fractional_part_theorem_l112_11231

theorem fractional_part_theorem (x : ℝ) (n : ℕ) (hn : n > 0) :
  ∃ k : ℕ, 1 ≤ k ∧ k < n ∧ ∃ m : ℤ, |k * x - m| ≤ 1 / n := by sorry

end fractional_part_theorem_l112_11231


namespace cafe_pricing_l112_11271

theorem cafe_pricing (s c p : ℝ) 
  (eq1 : 5 * s + 9 * c + 2 * p = 6.50)
  (eq2 : 7 * s + 14 * c + 3 * p = 9.45)
  (eq3 : 4 * s + 8 * c + p = 5.20) :
  s + c + p = 1.30 := by
  sorry

end cafe_pricing_l112_11271


namespace number_division_problem_l112_11291

theorem number_division_problem (N : ℕ) (D : ℕ) (h1 : N % D = 0) (h2 : N / D = 2) (h3 : N % 4 = 2) : D = 3 := by
  sorry

end number_division_problem_l112_11291


namespace evaluate_expression_l112_11282

theorem evaluate_expression (x y z : ℚ) 
  (hx : x = 1/2) (hy : y = 1/3) (hz : z = -3) : 
  x^2 * y^3 * z^2 = 1/12 := by
  sorry

end evaluate_expression_l112_11282


namespace candidates_appeared_l112_11214

theorem candidates_appeared (x : ℝ) 
  (h1 : 0.07 * x = 0.06 * x + 82) : x = 8200 := by
  sorry

end candidates_appeared_l112_11214


namespace isosceles_triangle_perimeter_l112_11212

/-- An isosceles triangle with two sides of length 9 and one side of length 2 has a perimeter of 20. -/
theorem isosceles_triangle_perimeter :
  ∀ (a b c : ℝ), 
    a = 9 → b = 9 → c = 2 →
    (a + b > c) ∧ (a + c > b) ∧ (b + c > a) →  -- Triangle inequality
    a = b →  -- Isosceles condition
    a + b + c = 20 := by
  sorry

end isosceles_triangle_perimeter_l112_11212


namespace parabola_intersection_length_l112_11259

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 16 * x

-- Define the line
def line (x : ℝ) : Prop := x = -1

-- Define the focus of the parabola
def focus : ℝ × ℝ := (4, 0)

-- Define points A and B
variable (A B : ℝ × ℝ)

-- State the theorem
theorem parabola_intersection_length :
  parabola B.1 B.2 →
  line A.1 →
  (∃ t : ℝ, 0 < t ∧ t < 1 ∧ B = (1 - t) • focus + t • A) →
  (A - focus) = 5 • (B - focus) →
  ‖A - B‖ = 28 :=
sorry

end parabola_intersection_length_l112_11259


namespace total_revenue_equals_8189_35_l112_11208

-- Define the types of ground beef
structure GroundBeef where
  regular : ℝ
  lean : ℝ
  extraLean : ℝ

-- Define the prices
def regularPrice : ℝ := 3.50
def leanPrice : ℝ := 4.25
def extraLeanPrice : ℝ := 5.00

-- Define the sales for each day
def mondaySales : GroundBeef := { regular := 198.5, lean := 276.2, extraLean := 150.7 }
def tuesdaySales : GroundBeef := { regular := 210, lean := 420, extraLean := 150 }
def wednesdaySales : GroundBeef := { regular := 230, lean := 324.6, extraLean := 120.4 }

-- Define the discount for Tuesday
def tuesdayDiscount : ℝ := 0.1

-- Define the sale price for lean ground beef on Wednesday
def wednesdayLeanSalePrice : ℝ := 3.75

-- Function to calculate revenue for a single day
def calculateDayRevenue (sales : GroundBeef) (regularPrice leanPrice extraLeanPrice : ℝ) : ℝ :=
  sales.regular * regularPrice + sales.lean * leanPrice + sales.extraLean * extraLeanPrice

-- Theorem statement
theorem total_revenue_equals_8189_35 :
  let mondayRevenue := calculateDayRevenue mondaySales regularPrice leanPrice extraLeanPrice
  let tuesdayRevenue := calculateDayRevenue tuesdaySales (regularPrice * (1 - tuesdayDiscount)) (leanPrice * (1 - tuesdayDiscount)) (extraLeanPrice * (1 - tuesdayDiscount))
  let wednesdayRevenue := calculateDayRevenue wednesdaySales regularPrice wednesdayLeanSalePrice extraLeanPrice
  mondayRevenue + tuesdayRevenue + wednesdayRevenue = 8189.35 := by
  sorry

end total_revenue_equals_8189_35_l112_11208


namespace travel_distance_ratio_l112_11223

theorem travel_distance_ratio :
  ∀ (total_distance plane_distance bus_distance train_distance : ℕ),
    total_distance = 900 →
    plane_distance = total_distance / 3 →
    bus_distance = 360 →
    train_distance = total_distance - (plane_distance + bus_distance) →
    (train_distance : ℚ) / bus_distance = 2 / 3 := by
  sorry

end travel_distance_ratio_l112_11223


namespace david_recreation_spending_l112_11220

-- Define the wages from last week as a parameter
def last_week_wages : ℝ := sorry

-- Define the percentage spent on recreation last week
def last_week_recreation_percent : ℝ := 0.40

-- Define the wage reduction percentage
def wage_reduction_percent : ℝ := 0.05

-- Define the increase in recreation spending
def recreation_increase_percent : ℝ := 1.1875

-- Calculate this week's wages
def this_week_wages : ℝ := last_week_wages * (1 - wage_reduction_percent)

-- Calculate the amount spent on recreation last week
def last_week_recreation_amount : ℝ := last_week_wages * last_week_recreation_percent

-- Calculate the amount spent on recreation this week
def this_week_recreation_amount : ℝ := last_week_recreation_amount * recreation_increase_percent

-- Define the theorem
theorem david_recreation_spending :
  this_week_recreation_amount / this_week_wages = 0.5 := by sorry

end david_recreation_spending_l112_11220


namespace prob_ace_heart_queen_l112_11263

-- Define the structure of a standard deck
def StandardDeck : Type := Unit

-- Define card types
inductive Rank
| Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King

inductive Suit
| Hearts | Diamonds | Clubs | Spades

structure Card where
  rank : Rank
  suit : Suit

-- Define the probability of drawing specific cards
def prob_first_ace (deck : StandardDeck) : ℚ := 4 / 52

def prob_second_heart (deck : StandardDeck) : ℚ := 13 / 51

def prob_third_queen (deck : StandardDeck) : ℚ := 4 / 50

-- State the theorem
theorem prob_ace_heart_queen (deck : StandardDeck) :
  prob_first_ace deck * prob_second_heart deck * prob_third_queen deck = 1 / 663 := by
  sorry

end prob_ace_heart_queen_l112_11263


namespace product_xyz_equals_25_l112_11275

/-- Given complex numbers x, y, and z satisfying specific equations, prove that their product is 25. -/
theorem product_xyz_equals_25 
  (x y z : ℂ) 
  (eq1 : 2 * x * y + 5 * y = -20)
  (eq2 : 2 * y * z + 5 * z = -20)
  (eq3 : 2 * z * x + 5 * x = -20) :
  x * y * z = 25 := by
  sorry

end product_xyz_equals_25_l112_11275


namespace line_intercept_sum_l112_11279

/-- Given a line 3x + 5y + d = 0, if the sum of its x-intercept and y-intercept is 15,
    then d = -225/8 -/
theorem line_intercept_sum (d : ℚ) : 
  (∃ x y : ℚ, 3 * x + 5 * y + d = 0 ∧ x + y = 15) → d = -225/8 := by
  sorry

end line_intercept_sum_l112_11279


namespace model_a_better_fit_l112_11240

/-- Represents a regression model --/
structure RegressionModel where
  rsquare : ℝ
  (rsquare_nonneg : 0 ≤ rsquare)
  (rsquare_le_one : rsquare ≤ 1)

/-- Defines when one model has a better fit than another --/
def better_fit (model1 model2 : RegressionModel) : Prop :=
  model1.rsquare > model2.rsquare

/-- Theorem stating that model A has a better fit than model B --/
theorem model_a_better_fit (model_a model_b : RegressionModel)
  (ha : model_a.rsquare = 0.98)
  (hb : model_b.rsquare = 0.80) :
  better_fit model_a model_b :=
sorry

end model_a_better_fit_l112_11240


namespace sum_of_constants_l112_11201

theorem sum_of_constants (a b : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y = a + b / x^2) →
  (2 = a + b) →
  (6 = a + b / 9) →
  a + b = 2 := by
sorry

end sum_of_constants_l112_11201


namespace product_of_fractions_l112_11269

theorem product_of_fractions : 
  (7 : ℚ) / 4 * 14 / 35 * 21 / 12 * 28 / 56 * 49 / 28 * 42 / 84 * 63 / 36 * 56 / 112 = 1201 / 12800 := by
  sorry

end product_of_fractions_l112_11269


namespace surprise_shop_revenue_l112_11264

/-- Represents the daily potential revenue of a shop during Christmas holidays -/
def daily_potential_revenue (closed_days_per_year : ℕ) (total_years : ℕ) (total_revenue_loss : ℕ) : ℚ :=
  total_revenue_loss / (closed_days_per_year * total_years)

/-- Theorem stating that the daily potential revenue for the given conditions is 5000 dollars -/
theorem surprise_shop_revenue : 
  daily_potential_revenue 3 6 90000 = 5000 := by
  sorry

end surprise_shop_revenue_l112_11264


namespace eggs_per_box_l112_11278

theorem eggs_per_box (total_eggs : ℕ) (num_boxes : ℕ) (h1 : total_eggs = 15) (h2 : num_boxes = 5) :
  total_eggs / num_boxes = 3 := by
  sorry

end eggs_per_box_l112_11278


namespace prime_sum_product_l112_11204

theorem prime_sum_product (p q : ℕ) : 
  Prime p → Prime q → p + q = 91 → p * q = 178 := by sorry

end prime_sum_product_l112_11204


namespace largest_number_problem_l112_11280

theorem largest_number_problem (a b c : ℝ) 
  (h_order : a < b ∧ b < c)
  (h_sum : a + b + c = 67)
  (h_diff_large : c - b = 7)
  (h_diff_small : b - a = 5) :
  c = 86 / 3 := by
sorry

end largest_number_problem_l112_11280


namespace opposite_of_negative_three_l112_11258

theorem opposite_of_negative_three : 
  (∃ x : ℤ, -3 + x = 0) → (∃ x : ℤ, -3 + x = 0 ∧ x = 3) :=
by sorry

end opposite_of_negative_three_l112_11258


namespace find_divisor_l112_11238

theorem find_divisor (dividend : ℕ) (quotient : ℕ) (h1 : dividend = 1254) (h2 : quotient = 209) 
  (h3 : dividend % (dividend / quotient) = 0) : dividend / quotient = 6 := by
  sorry

end find_divisor_l112_11238


namespace smallest_x_value_l112_11267

theorem smallest_x_value (x : ℝ) : 
  (3 * x^2 + 36 * x - 90 = x * (x + 15)) → x ≥ -15 := by
  sorry

end smallest_x_value_l112_11267


namespace quadrilateral_bf_length_l112_11224

-- Define the points
variable (A B C D E F : ℝ × ℝ)

-- Define the conditions
variable (h1 : A.1 = 0 ∧ A.2 = 0)  -- A is at (0,0)
variable (h2 : C.1 = 10 ∧ C.2 = 0)  -- C is at (10,0)
variable (h3 : E.1 = 3 ∧ E.2 = 0)  -- E is at (3,0)
variable (h4 : F.1 = 7 ∧ F.2 = 0)  -- F is at (7,0)
variable (h5 : D.1 = 3 ∧ D.2 = -5)  -- D is at (3,-5)
variable (h6 : B.1 = 7 ∧ B.2 = 4.2)  -- B is at (7,4.2)

-- Define the geometric conditions
variable (h7 : (B.2 - A.2) * (D.1 - A.1) = (D.2 - A.2) * (B.1 - A.1))  -- ∠BAD is right
variable (h8 : (B.2 - C.2) * (D.1 - C.1) = (D.2 - C.2) * (B.1 - C.1))  -- ∠BCD is right
variable (h9 : (D.2 - E.2) * (C.1 - E.1) = (C.2 - E.2) * (D.1 - E.1))  -- DE ⊥ AC
variable (h10 : (B.2 - F.2) * (A.1 - F.1) = (A.2 - F.2) * (B.1 - F.1))  -- BF ⊥ AC

-- Define the length conditions
variable (h11 : Real.sqrt ((E.1 - A.1)^2 + (E.2 - A.2)^2) = 3)  -- AE = 3
variable (h12 : Real.sqrt ((E.1 - D.1)^2 + (E.2 - D.2)^2) = 5)  -- DE = 5
variable (h13 : Real.sqrt ((C.1 - E.1)^2 + (C.2 - E.2)^2) = 7)  -- CE = 7

-- Theorem statement
theorem quadrilateral_bf_length : 
  Real.sqrt ((F.1 - B.1)^2 + (F.2 - B.2)^2) = 4.2 :=
sorry

end quadrilateral_bf_length_l112_11224


namespace multiple_compounds_same_weight_l112_11218

/-- Represents a chemical compound -/
structure Compound where
  molecular_weight : ℕ

/-- Represents the set of all possible compounds -/
def AllCompounds : Set Compound := sorry

/-- The given molecular weight -/
def given_weight : ℕ := 391

/-- Compounds with the given molecular weight -/
def compounds_with_given_weight : Set Compound :=
  {c ∈ AllCompounds | c.molecular_weight = given_weight}

/-- Theorem stating that multiple compounds can have the same molecular weight -/
theorem multiple_compounds_same_weight :
  ∃ (c1 c2 : Compound), c1 ≠ c2 ∧ c1 ∈ compounds_with_given_weight ∧ c2 ∈ compounds_with_given_weight :=
sorry

end multiple_compounds_same_weight_l112_11218


namespace number_wall_solution_l112_11296

/-- Represents a block in the Number Wall --/
structure Block where
  value : ℕ

/-- Represents the Number Wall --/
structure NumberWall where
  n : Block
  block1 : Block
  block2 : Block
  block3 : Block
  block4 : Block
  top : Block

/-- The sum of two adjacent blocks equals the block above them --/
def sum_rule (b1 b2 b_above : Block) : Prop :=
  b1.value + b2.value = b_above.value

/-- The Number Wall satisfies all given conditions --/
def valid_wall (w : NumberWall) : Prop :=
  w.block1.value = 4 ∧
  w.block2.value = 8 ∧
  w.block3.value = 7 ∧
  w.block4.value = 15 ∧
  w.top.value = 46 ∧
  sum_rule w.n w.block1 { value := w.n.value + 4 } ∧
  sum_rule { value := w.n.value + 4 } w.block2 w.block4 ∧
  sum_rule w.block4 w.block3 { value := 27 } ∧
  sum_rule { value := w.n.value + 16 } { value := 27 } w.top

theorem number_wall_solution (w : NumberWall) (h : valid_wall w) : w.n.value = 3 := by
  sorry


end number_wall_solution_l112_11296


namespace find_a_l112_11257

theorem find_a : ∃ a : ℕ, 
  (∀ k : ℤ, k ≠ 27 → ∃ m : ℤ, a - k^1964 = m * (27 - k)) → 
  a = 27^1964 := by
sorry

end find_a_l112_11257


namespace fraction_thousandths_digit_l112_11236

def fraction : ℚ := 57 / 5000

/-- The thousandths digit of a rational number is the third digit after the decimal point in its decimal representation. -/
def thousandths_digit (q : ℚ) : ℕ :=
  sorry

theorem fraction_thousandths_digit :
  thousandths_digit fraction = 1 := by
  sorry

end fraction_thousandths_digit_l112_11236


namespace abs_inequality_equivalence_l112_11295

theorem abs_inequality_equivalence :
  ∀ x : ℝ, |5 - 2*x| < 3 ↔ 1 < x ∧ x < 4 := by sorry

end abs_inequality_equivalence_l112_11295


namespace taxi_driver_theorem_l112_11256

def driving_distances : List Int := [5, -3, 6, -7, 6, -2, -5, -4, 6, -8]

def starting_price : ℕ := 8
def base_distance : ℕ := 3
def additional_rate : ℚ := 3/2

theorem taxi_driver_theorem :
  (List.sum driving_distances = -6) ∧
  (List.sum (List.take 7 driving_distances) = 0) ∧
  (starting_price + (8 - base_distance) * additional_rate = 31/2) ∧
  (∀ x : ℕ, x > base_distance → starting_price + (x - base_distance) * additional_rate = (3 * x + 7) / 2) :=
by sorry

end taxi_driver_theorem_l112_11256


namespace equation_solutions_l112_11261

theorem equation_solutions :
  (∀ x : ℝ, 12 * (x - 1)^2 = 3 ↔ x = 3/2 ∨ x = 1/2) ∧
  (∀ x : ℝ, (x + 1)^3 = 0.125 ↔ x = -0.5) := by
sorry

end equation_solutions_l112_11261


namespace earth_surface_area_scientific_notation_l112_11206

/-- The surface area of the Earth in square kilometers. -/
def earth_surface_area : ℝ := 510000000

/-- The scientific notation representation of the Earth's surface area. -/
def earth_surface_area_scientific : ℝ := 5.1 * (10 ^ 8)

/-- Theorem stating that the Earth's surface area is correctly represented in scientific notation. -/
theorem earth_surface_area_scientific_notation : 
  earth_surface_area = earth_surface_area_scientific := by sorry

end earth_surface_area_scientific_notation_l112_11206


namespace math_books_count_l112_11255

theorem math_books_count (total_books : ℕ) (math_cost history_cost total_price : ℕ) :
  total_books = 80 ∧ 
  math_cost = 4 ∧ 
  history_cost = 5 ∧ 
  total_price = 368 →
  ∃ (math_books : ℕ), 
    math_books * math_cost + (total_books - math_books) * history_cost = total_price ∧ 
    math_books = 32 :=
by sorry

end math_books_count_l112_11255


namespace truncated_cone_radius_l112_11293

/-- Given three cones touching each other with base radii 6, 24, and 24,
    and a truncated cone sharing a common generator with each,
    the radius of the smaller base of the truncated cone is 2. -/
theorem truncated_cone_radius (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 6) (h₂ : r₂ = 24) (h₃ : r₃ = 24) :
  ∃ (r : ℝ), r = 2 ∧ 
  (r = r₂ - 24) ∧ 
  (r = r₃ - 24) ∧
  ((24 + r)^2 = 24^2 + (12 - r)^2) :=
by sorry

end truncated_cone_radius_l112_11293


namespace triangle_angle_theorem_l112_11276

/-- A triple of integers representing the angles of a triangle in degrees. -/
structure TriangleAngles where
  a : ℕ
  b : ℕ
  c : ℕ
  sum_eq_180 : a + b + c = 180
  all_positive : 0 < a ∧ 0 < b ∧ 0 < c
  all_acute : a < 90 ∧ b < 90 ∧ c < 90

/-- The set of valid angle combinations for the triangle. -/
def validCombinations : Set TriangleAngles := {
  ⟨42, 72, 66, by norm_num, by norm_num, by norm_num⟩,
  ⟨49, 54, 77, by norm_num, by norm_num, by norm_num⟩,
  ⟨56, 36, 88, by norm_num, by norm_num, by norm_num⟩,
  ⟨84, 63, 33, by norm_num, by norm_num, by norm_num⟩
}

/-- Theorem stating that the only valid angle combinations for the triangle
    are those in the validCombinations set. -/
theorem triangle_angle_theorem :
  ∀ t : TriangleAngles,
    (∃ k : ℕ, t.a = 7 * k) ∧
    (∃ l : ℕ, t.b = 9 * l) ∧
    (∃ m : ℕ, t.c = 11 * m) →
    t ∈ validCombinations := by
  sorry

end triangle_angle_theorem_l112_11276


namespace sum_of_specific_S_values_l112_11252

def S (n : ℕ) : ℤ :=
  if n % 2 = 0 then
    -n / 2
  else
    (n + 1) / 2

theorem sum_of_specific_S_values : S 17 + S 33 + S 50 = 1 := by
  sorry

end sum_of_specific_S_values_l112_11252


namespace lukas_points_l112_11254

/-- Given a basketball player's average points per game and a number of games,
    calculates the total points scored. -/
def total_points (avg_points : ℕ) (num_games : ℕ) : ℕ :=
  avg_points * num_games

/-- Proves that a player averaging 12 points per game scores 60 points in 5 games. -/
theorem lukas_points : total_points 12 5 = 60 := by
  sorry

end lukas_points_l112_11254


namespace sum_inequality_l112_11250

theorem sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*a*c)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 := by
  sorry

end sum_inequality_l112_11250


namespace constant_term_expansion_l112_11260

theorem constant_term_expansion (x : ℝ) (x_neq_zero : x ≠ 0) :
  ∃ (f : ℝ → ℝ), (∀ y, f y = (y + 4/y - 4)^3) ∧
  (∃ c, ∀ y ≠ 0, f y = c + y * (f y - c) / y) ∧
  c = -160 := by
sorry

end constant_term_expansion_l112_11260


namespace f_range_and_triangle_area_l112_11270

noncomputable def f (x : ℝ) : ℝ := 
  Real.sqrt 3 * (Real.cos x)^2 + Real.sin x * Real.cos x

def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  A > 0 ∧ A < Real.pi ∧
  B > 0 ∧ B < Real.pi ∧
  C > 0 ∧ C < Real.pi ∧
  A + B + C = Real.pi

theorem f_range_and_triangle_area 
  (h1 : ∀ x, x ∈ Set.Icc 0 (Real.pi / 2) → f x ∈ Set.Icc 0 (1 + Real.sqrt 3 / 2))
  (h2 : ∃ A B C a b c, 
    triangle_ABC a b c A B C ∧ 
    f (A / 2) = Real.sqrt 3 ∧
    a = 4 ∧
    b + c = 5) :
  ∃ A B C a b c, 
    triangle_ABC a b c A B C ∧
    f (A / 2) = Real.sqrt 3 ∧
    a = 4 ∧
    b + c = 5 ∧
    (1/2) * b * c * Real.sin A = 3 * Real.sqrt 3 / 4 := by
  sorry

end f_range_and_triangle_area_l112_11270


namespace events_B_C_complementary_l112_11227

-- Define the sample space (cube faces)
def Ω : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define events A, B, and C
def A : Set ℕ := {n ∈ Ω | n % 2 = 1}
def B : Set ℕ := {n ∈ Ω | n ≤ 3}
def C : Set ℕ := {n ∈ Ω | n ≥ 4}

-- Theorem to prove
theorem events_B_C_complementary : B ∪ C = Ω ∧ B ∩ C = ∅ := by
  sorry

end events_B_C_complementary_l112_11227


namespace noah_holidays_l112_11213

/-- Calculates the number of holidays taken in a year given monthly holidays. -/
def holidays_per_year (monthly_holidays : ℕ) : ℕ :=
  monthly_holidays * 12

/-- Theorem: Given 3 holidays per month for a full year, the total holidays is 36. -/
theorem noah_holidays :
  holidays_per_year 3 = 36 := by
  sorry

end noah_holidays_l112_11213


namespace p_and_q_false_iff_a_range_l112_11200

/-- The logarithm function with base 10 -/
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

/-- The function f(x) = lg(ax^2 - x + a/16) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := lg (a * x^2 - x + a/16)

/-- Proposition p: The range of f(x) is ℝ -/
def p (a : ℝ) : Prop := ∀ y : ℝ, ∃ x : ℝ, f a x = y

/-- Proposition q: 3^x - 9^x < a holds for all real numbers x -/
def q (a : ℝ) : Prop := ∀ x : ℝ, 3^x - 9^x < a

/-- Theorem: "p and q" is false iff a > 2 or a ≤ 1/4 -/
theorem p_and_q_false_iff_a_range (a : ℝ) : ¬(p a ∧ q a) ↔ a > 2 ∨ a ≤ 1/4 := by
  sorry

end p_and_q_false_iff_a_range_l112_11200


namespace correct_equation_proof_l112_11249

def quadratic_equation (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

def has_roots (a b c : ℝ) (r₁ r₂ : ℝ) : Prop :=
  quadratic_equation a b c r₁ = 0 ∧ quadratic_equation a b c r₂ = 0

theorem correct_equation_proof :
  ∃ (a₁ b₁ c₁ : ℝ) (a₂ b₂ c₂ : ℝ),
    has_roots a₁ b₁ c₁ 8 2 ∧
    has_roots a₂ b₂ c₂ (-9) (-1) ∧
    (a₁ = 1 ∧ b₁ = -10 ∧ c₁ ≠ 9) ∧
    (a₂ = 1 ∧ b₂ ≠ -10 ∧ c₂ = 9) ∧
    quadratic_equation 1 (-10) 9 = quadratic_equation a₁ b₁ c₁ ∧
    quadratic_equation 1 (-10) 9 = quadratic_equation a₂ b₂ c₂ :=
by sorry

end correct_equation_proof_l112_11249


namespace felix_lifting_capacity_l112_11245

/-- Felix's lifting capacity problem -/
theorem felix_lifting_capacity 
  (felix_lift_ratio : ℝ) 
  (brother_weight_ratio : ℝ) 
  (brother_lift_ratio : ℝ) 
  (brother_lift_weight : ℝ) 
  (h1 : felix_lift_ratio = 1.5)
  (h2 : brother_weight_ratio = 2)
  (h3 : brother_lift_ratio = 3)
  (h4 : brother_lift_weight = 600) :
  felix_lift_ratio * (brother_lift_weight / (brother_lift_ratio * brother_weight_ratio)) = 150 := by
  sorry


end felix_lifting_capacity_l112_11245


namespace negation_of_universal_proposition_cubic_inequality_negation_l112_11203

theorem negation_of_universal_proposition (p : ℝ → Prop) :
  (¬∀ x, p x) ↔ (∃ x, ¬p x) := by sorry

theorem cubic_inequality_negation :
  (¬∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) := by sorry

end negation_of_universal_proposition_cubic_inequality_negation_l112_11203


namespace machine_work_time_solution_l112_11235

theorem machine_work_time_solution : ∃ x : ℝ, 
  (x > 0) ∧ 
  (1 / (x + 4) + 1 / (x + 2) + 1 / (2 * x + 6) = 1 / x) ∧ 
  (x = 2) := by
  sorry

end machine_work_time_solution_l112_11235


namespace prob_vertical_side_from_start_l112_11298

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

end prob_vertical_side_from_start_l112_11298


namespace m_plus_n_values_l112_11294

theorem m_plus_n_values (m n : ℤ) (hm : m = 3) (hn : |n| = 1) :
  m + n = 4 ∨ m + n = 2 := by
sorry

end m_plus_n_values_l112_11294


namespace f_inequality_iff_a_range_l112_11297

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - (a + 1) * x + (1/2) * x^2

theorem f_inequality_iff_a_range (a : ℝ) :
  (a > 0 ∧ ∀ x > 1, f a x ≥ x^a - Real.exp x + (1/2) * x^2 - a * x) ↔ 0 < a ∧ a ≤ Real.exp 1 :=
sorry

end f_inequality_iff_a_range_l112_11297


namespace sector_area_theorem_l112_11277

/-- A sector is a portion of a circle enclosed by two radii and an arc. -/
structure Sector where
  centralAngle : ℝ
  perimeter : ℝ

/-- The area of a sector. -/
def sectorArea (s : Sector) : ℝ := sorry

theorem sector_area_theorem (s : Sector) :
  s.centralAngle = 2 ∧ s.perimeter = 8 → sectorArea s = 4 := by
  sorry

end sector_area_theorem_l112_11277


namespace banana_bread_theorem_l112_11229

/-- The number of bananas needed to make one loaf of banana bread -/
def bananas_per_loaf : ℕ := 4

/-- The number of loaves of banana bread made on Monday -/
def monday_loaves : ℕ := 3

/-- The number of loaves of banana bread made on Tuesday -/
def tuesday_loaves : ℕ := 2 * monday_loaves

/-- The total number of bananas used for banana bread on both days -/
def total_bananas : ℕ := bananas_per_loaf * (monday_loaves + tuesday_loaves)

theorem banana_bread_theorem : total_bananas = 36 := by
  sorry

end banana_bread_theorem_l112_11229


namespace ratio_of_multiples_l112_11281

theorem ratio_of_multiples (w x y z : ℝ) 
  (hx : x = 4 * y) 
  (hy : y = 3 * z) 
  (hz : z = 5 * w) : 
  (x * z) / (y * w) = 20 := by
sorry

end ratio_of_multiples_l112_11281


namespace call_charge_for_550_seconds_l112_11248

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

end call_charge_for_550_seconds_l112_11248


namespace profit_with_discount_theorem_l112_11239

/-- Calculates the profit percentage with discount given the discount rate and profit percentage without discount -/
def profit_percentage_with_discount (discount_rate : ℝ) (profit_no_discount : ℝ) : ℝ :=
  ((1 - discount_rate) * (1 + profit_no_discount) - 1) * 100

/-- Theorem stating that given a 5% discount and 28% profit without discount, the profit percentage with discount is 21.6% -/
theorem profit_with_discount_theorem :
  profit_percentage_with_discount 0.05 0.28 = 21.6 := by
  sorry

end profit_with_discount_theorem_l112_11239


namespace complement_union_M_N_l112_11266

def U : Set Nat := {1, 2, 3, 4, 5}
def M : Set Nat := {1, 2}
def N : Set Nat := {3, 4}

theorem complement_union_M_N : (U \ (M ∪ N)) = {5} := by
  sorry

end complement_union_M_N_l112_11266


namespace parallel_vectors_x_value_l112_11228

theorem parallel_vectors_x_value (x : ℝ) :
  let a : Fin 2 → ℝ := ![x, 1]
  let b : Fin 2 → ℝ := ![1, -1]
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) → x = -1 := by
  sorry

end parallel_vectors_x_value_l112_11228


namespace kelly_apples_l112_11292

/-- Given Kelly's initial apples and the number of apples she needs to pick,
    calculate the total number of apples she will have. -/
def total_apples (initial : ℕ) (to_pick : ℕ) : ℕ :=
  initial + to_pick

/-- Theorem stating that Kelly will have 105 apples altogether -/
theorem kelly_apples :
  total_apples 56 49 = 105 := by
  sorry

end kelly_apples_l112_11292


namespace smallest_staircase_steps_l112_11243

theorem smallest_staircase_steps (n : ℕ) : 
  (n > 15) ∧ 
  (n % 6 = 4) ∧ 
  (n % 7 = 3) ∧ 
  (∀ m : ℕ, m > 15 ∧ m % 6 = 4 ∧ m % 7 = 3 → m ≥ n) → 
  n = 52 := by
sorry

end smallest_staircase_steps_l112_11243


namespace rectangle_side_length_l112_11230

/-- 
Given a rectangular arrangement with all right angles, where the top length 
consists of segments 3 cm, 2 cm, Y cm, and 1 cm sequentially, and the total 
bottom length is 11 cm, prove that Y = 5 cm.
-/
theorem rectangle_side_length (Y : ℝ) : 
  (3 : ℝ) + 2 + Y + 1 = 11 → Y = 5 := by
  sorry

end rectangle_side_length_l112_11230


namespace isosceles_triangle_altitude_ratio_l112_11253

/-- An isosceles triangle with base to side ratio 4:3 has its altitude dividing the side in ratio 2:1 -/
theorem isosceles_triangle_altitude_ratio :
  ∀ (a b h m n : ℝ),
  a > 0 → b > 0 → h > 0 → m > 0 → n > 0 →
  b = (4/3) * a →  -- base to side ratio is 4:3
  h^2 = a^2 - (b/2)^2 →  -- height formula
  a^2 = h^2 + m^2 →  -- right triangle formed by altitude
  a = m + n →  -- side divided by altitude
  m / n = 2 / 1 :=  -- ratio in which altitude divides the side
by
  sorry


end isosceles_triangle_altitude_ratio_l112_11253


namespace houses_built_l112_11285

theorem houses_built (original : ℕ) (current : ℕ) (built : ℕ) : 
  original = 20817 → current = 118558 → built = current - original → built = 97741 := by
  sorry

end houses_built_l112_11285


namespace no_common_real_root_l112_11205

theorem no_common_real_root (a b : ℚ) : ¬∃ (r : ℝ), r^5 - r - 1 = 0 ∧ r^2 + a*r + b = 0 := by
  sorry

end no_common_real_root_l112_11205


namespace sum_of_x_and_y_is_two_l112_11272

theorem sum_of_x_and_y_is_two (x y : ℝ) (h : x^2 + y^2 = 12*x - 8*y - 56) : x + y = 2 := by
  sorry

end sum_of_x_and_y_is_two_l112_11272


namespace greater_number_on_cards_l112_11246

theorem greater_number_on_cards (x y : ℤ) 
  (sum_eq : x + y = 1443) 
  (diff_eq : x - y = 141) : 
  x = 792 ∧ x > y :=
by sorry

end greater_number_on_cards_l112_11246


namespace fraction_equivalence_l112_11233

theorem fraction_equivalence : 
  ∀ (n : ℚ), (2 + n) / (7 + n) = 3 / 8 → n = 1 := by sorry

end fraction_equivalence_l112_11233


namespace solution_exists_l112_11211

theorem solution_exists : ∃ c : ℝ, 
  (∃ x : ℤ, (x = ⌊c⌋ ∧ 3 * (x : ℝ)^2 - 9 * (x : ℝ) - 30 = 0)) ∧
  (∃ y : ℝ, (y = c - ⌊c⌋ ∧ 4 * y^2 - 8 * y + 1 = 0)) ∧
  (c = -1 - Real.sqrt 3 / 2 ∨ c = 6 - Real.sqrt 3 / 2) :=
by sorry

end solution_exists_l112_11211


namespace alissa_earrings_l112_11289

/-- Represents the number of pairs of earrings Barbie bought -/
def barbie_pairs : ℕ := 12

/-- Represents the number of earrings Barbie gave to Alissa -/
def earrings_given : ℕ := barbie_pairs * 2 / 2

/-- Represents Alissa's total number of earrings after receiving the gift -/
def alissa_total : ℕ := 3 * earrings_given

theorem alissa_earrings : alissa_total = 36 := by
  sorry

end alissa_earrings_l112_11289


namespace triangle_angles_l112_11242

/-- Given a triangle with sides 5, 5, and √17 - √5, prove that its angles are θ, φ, φ, where
    θ = arccos((14 + √85) / 25) and φ = (180° - θ) / 2 -/
theorem triangle_angles (a b c : ℝ) (θ φ : ℝ) : 
  a = 5 → b = 5 → c = Real.sqrt 17 - Real.sqrt 5 →
  θ = Real.arccos ((14 + Real.sqrt 85) / 25) →
  φ = (π - θ) / 2 →
  ∃ (α β γ : ℝ), 
    (α = θ ∧ β = φ ∧ γ = φ) ∧
    (α + β + γ = π) ∧
    (Real.cos α = (b^2 + c^2 - a^2) / (2 * b * c)) ∧
    (Real.cos β = (a^2 + c^2 - b^2) / (2 * a * c)) ∧
    (Real.cos γ = (a^2 + b^2 - c^2) / (2 * a * b)) :=
by sorry

end triangle_angles_l112_11242


namespace valerie_laptop_savings_l112_11299

/-- Proves that Valerie needs 30 weeks to save for a laptop -/
theorem valerie_laptop_savings :
  let laptop_price : ℕ := 800
  let parents_money : ℕ := 100
  let uncle_money : ℕ := 60
  let siblings_money : ℕ := 40
  let weekly_tutoring_income : ℕ := 20
  let total_graduation_money : ℕ := parents_money + uncle_money + siblings_money
  let remaining_amount : ℕ := laptop_price - total_graduation_money
  let weeks_needed : ℕ := remaining_amount / weekly_tutoring_income
  weeks_needed = 30 := by
sorry


end valerie_laptop_savings_l112_11299


namespace geometric_sequence_log_sum_l112_11251

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_log_sum
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_pos : ∀ n, a n > 0)
  (h_prod : a 10 * a 11 = Real.exp 5) :
  (Finset.range 20).sum (λ i => Real.log (a (i + 1))) = 50 := by
  sorry

end geometric_sequence_log_sum_l112_11251


namespace grocery_store_problem_l112_11287

/-- Represents the price of an item after applying discount and tax -/
structure ItemPrice where
  base : ℝ
  discount : ℝ
  tax : ℝ

/-- Calculates the final price of an item after applying discount and tax -/
def finalPrice (item : ItemPrice) : ℝ :=
  item.base * (1 - item.discount) * (1 + item.tax)

/-- Represents the grocery store problem -/
theorem grocery_store_problem :
  let spam : ItemPrice := { base := 3, discount := 0.1, tax := 0 }
  let peanutButter : ItemPrice := { base := 5, discount := 0, tax := 0.05 }
  let bread : ItemPrice := { base := 2, discount := 0, tax := 0 }
  let milk : ItemPrice := { base := 4, discount := 0.2, tax := 0.08 }
  let eggs : ItemPrice := { base := 3, discount := 0.05, tax := 0 }
  
  let totalAmount :=
    12 * finalPrice spam +
    3 * finalPrice peanutButter +
    4 * finalPrice bread +
    2 * finalPrice milk +
    1 * finalPrice eggs
  
  totalAmount = 65.92 := by sorry

end grocery_store_problem_l112_11287


namespace smallest_undefined_value_l112_11288

theorem smallest_undefined_value : 
  let f (x : ℝ) := (x - 3) / (9*x^2 - 90*x + 225)
  ∃ (y : ℝ), (∀ (x : ℝ), x < y → f x ≠ 0⁻¹) ∧ f y = 0⁻¹ ∧ y = 5 :=
by sorry

end smallest_undefined_value_l112_11288
