import Mathlib

namespace hyperbola_foci_l1502_150259

/-- Given a hyperbola with equation x²/4 - y² = 1, prove that its foci are at (±√5, 0) -/
theorem hyperbola_foci (x y : ℝ) : 
  (x^2 / 4 - y^2 = 1) → (∃ (s : ℝ), s^2 = 5 ∧ ((x = s ∨ x = -s) ∧ y = 0)) :=
by sorry

end hyperbola_foci_l1502_150259


namespace number_of_observations_l1502_150219

theorem number_of_observations
  (initial_mean : ℝ)
  (incorrect_value : ℝ)
  (correct_value : ℝ)
  (corrected_mean : ℝ)
  (h1 : initial_mean = 41)
  (h2 : incorrect_value = 23)
  (h3 : correct_value = 48)
  (h4 : corrected_mean = 41.5) :
  ∃ n : ℕ, n * initial_mean - incorrect_value + correct_value = n * corrected_mean ∧ n = 50 := by
  sorry

end number_of_observations_l1502_150219


namespace mike_ride_mileage_l1502_150282

/-- Represents the cost of a taxi ride -/
structure TaxiRide where
  startFee : ℝ
  tollFee : ℝ
  mileage : ℝ
  costPerMile : ℝ

/-- Calculates the total cost of a taxi ride -/
def totalCost (ride : TaxiRide) : ℝ :=
  ride.startFee + ride.tollFee + ride.mileage * ride.costPerMile

theorem mike_ride_mileage :
  let mikeRide : TaxiRide := {
    startFee := 2.5,
    tollFee := 0,
    mileage := m,
    costPerMile := 0.25
  }
  let annieRide : TaxiRide := {
    startFee := 2.5,
    tollFee := 5,
    mileage := 26,
    costPerMile := 0.25
  }
  totalCost mikeRide = totalCost annieRide → m = 36 := by
  sorry

#check mike_ride_mileage

end mike_ride_mileage_l1502_150282


namespace repeating_decimal_division_l1502_150204

/-- Represents a repeating decimal with a single repeating digit -/
def RepeatingDecimal (whole : ℚ) (repeating : ℕ) : ℚ :=
  whole + (repeating : ℚ) / 999

theorem repeating_decimal_division (h : RepeatingDecimal 0 4 / RepeatingDecimal 1 6 = 4 / 15) : 
  RepeatingDecimal 0 4 / RepeatingDecimal 1 6 = 4 / 15 := by
  sorry

end repeating_decimal_division_l1502_150204


namespace card_sum_theorem_l1502_150256

theorem card_sum_theorem (a b c d e f g h : ℕ) :
  (a + b) * (c + d) * (e + f) * (g + h) = 330 →
  a + b + c + d + e + f + g + h = 21 := by
sorry

end card_sum_theorem_l1502_150256


namespace olivia_savings_account_l1502_150292

/-- The compound interest function -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- The problem statement -/
theorem olivia_savings_account :
  let principal : ℝ := 5000
  let rate : ℝ := 0.07
  let time : ℕ := 15
  let final_amount := compound_interest principal rate time
  ∃ ε > 0, |final_amount - 13795.15| < ε :=
by sorry

end olivia_savings_account_l1502_150292


namespace largest_fraction_l1502_150272

theorem largest_fraction (a b c d : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b = c) (h4 : c < d) :
  let f1 := (a + b) / (c + d)
  let f2 := (a + d) / (b + c)
  let f3 := (b + c) / (a + d)
  let f4 := (b + d) / (a + c)
  let f5 := (c + d) / (a + b)
  (f4 = f5) ∧ (f4 ≥ f1) ∧ (f4 ≥ f2) ∧ (f4 ≥ f3) :=
by sorry

end largest_fraction_l1502_150272


namespace point_on_axis_l1502_150276

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of a point being on the x-axis -/
def onXAxis (p : Point2D) : Prop := p.y = 0

/-- Definition of a point being on the y-axis -/
def onYAxis (p : Point2D) : Prop := p.x = 0

/-- Theorem: If xy = 0, then the point is on the x-axis or y-axis -/
theorem point_on_axis (p : Point2D) (h : p.x * p.y = 0) :
  onXAxis p ∨ onYAxis p := by
  sorry

end point_on_axis_l1502_150276


namespace flour_for_cookies_l1502_150258

/-- Given a recipe where 20 cookies require 3 cups of flour,
    calculate the number of cups of flour needed for 100 cookies. -/
theorem flour_for_cookies (original_cookies : ℕ) (original_flour : ℕ) (target_cookies : ℕ) :
  original_cookies = 20 →
  original_flour = 3 →
  target_cookies = 100 →
  (target_cookies * original_flour) / original_cookies = 15 :=
by sorry

end flour_for_cookies_l1502_150258


namespace adjacent_vertices_probability_l1502_150288

/-- A decagon is a polygon with 10 sides and vertices -/
def Decagon := Nat

/-- The number of vertices in a decagon -/
def num_vertices : Decagon → Nat := fun _ => 10

/-- The number of adjacent vertices for each vertex in a decagon -/
def num_adjacent_vertices : Decagon → Nat := fun _ => 2

/-- The total number of ways to choose the second vertex -/
def total_second_vertex_choices : Decagon → Nat := fun d => num_vertices d - 1

theorem adjacent_vertices_probability (d : Decagon) :
  (num_adjacent_vertices d : ℚ) / (total_second_vertex_choices d) = 2 / 9 := by
  sorry

end adjacent_vertices_probability_l1502_150288


namespace omega_sum_simplification_l1502_150230

theorem omega_sum_simplification (ω : ℂ) (h1 : ω^8 = 1) (h2 : ω ≠ 1) :
  ω^17 + ω^21 + ω^25 + ω^29 + ω^33 + ω^37 + ω^41 + ω^45 + ω^49 + ω^53 + ω^57 + ω^61 + ω^65 = ω :=
by sorry

end omega_sum_simplification_l1502_150230


namespace increasing_condition_l1502_150244

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-2)*x + 5

-- State the theorem
theorem increasing_condition (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 4 < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂) ↔ a ≥ -2 :=
by sorry

end increasing_condition_l1502_150244


namespace diagonal_sum_inequality_l1502_150262

-- Define a convex quadrilateral
structure ConvexQuadrilateral where
  vertices : Fin 4 → ℝ × ℝ
  is_convex : sorry

-- Define the sum of diagonal lengths for a quadrilateral
def sum_of_diagonals (q : ConvexQuadrilateral) : ℝ := sorry

-- Define the "inside" relation for quadrilaterals
def inside (inner outer : ConvexQuadrilateral) : Prop := sorry

-- Theorem statement
theorem diagonal_sum_inequality {P P' : ConvexQuadrilateral} 
  (h_inside : inside P' P) : 
  sum_of_diagonals P' < 2 * sum_of_diagonals P := by
  sorry

end diagonal_sum_inequality_l1502_150262


namespace decimal_multiplication_l1502_150240

theorem decimal_multiplication : (0.5 : ℝ) * 0.7 = 0.35 := by sorry

end decimal_multiplication_l1502_150240


namespace perfect_pairs_iff_even_l1502_150232

/-- A pair of integers (a, b) is perfect if ab + 1 is a perfect square. -/
def IsPerfectPair (a b : ℤ) : Prop :=
  ∃ k : ℤ, a * b + 1 = k ^ 2

/-- The set {1, ..., 2n} can be divided into n perfect pairs. -/
def CanDivideIntoPerfectPairs (n : ℕ) : Prop :=
  ∃ f : Fin n → Fin (2 * n) × Fin (2 * n),
    (∀ i : Fin n, IsPerfectPair (f i).1.val.succ (f i).2.val.succ) ∧
    (∀ i j : Fin n, i ≠ j → (f i).1 ≠ (f j).1 ∧ (f i).1 ≠ (f j).2 ∧ 
                            (f i).2 ≠ (f j).1 ∧ (f i).2 ≠ (f j).2)

/-- The main theorem: The set {1, ..., 2n} can be divided into n perfect pairs 
    if and only if n is even. -/
theorem perfect_pairs_iff_even (n : ℕ) :
  CanDivideIntoPerfectPairs n ↔ Even n :=
sorry

end perfect_pairs_iff_even_l1502_150232


namespace shenile_score_theorem_l1502_150277

/-- Represents the number of points Shenille scored in a basketball game -/
def shenilesScore (threePointAttempts twoPointAttempts : ℕ) : ℝ :=
  0.6 * (threePointAttempts + twoPointAttempts)

theorem shenile_score_theorem :
  ∀ threePointAttempts twoPointAttempts : ℕ,
  threePointAttempts + twoPointAttempts = 30 →
  shenilesScore threePointAttempts twoPointAttempts = 18 :=
by
  sorry

#check shenile_score_theorem

end shenile_score_theorem_l1502_150277


namespace quadratic_root_zero_l1502_150268

/-- A quadratic equation with parameter k -/
def quadratic_equation (k : ℝ) (x : ℝ) : ℝ := (k - 1) * x^2 + 6 * x + k^2 - k

theorem quadratic_root_zero (k : ℝ) :
  (quadratic_equation k 0 = 0) ∧ (k - 1 ≠ 0) → k = 0 := by
  sorry

end quadratic_root_zero_l1502_150268


namespace min_value_3x_plus_4y_l1502_150208

theorem min_value_3x_plus_4y (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : x + 3 * y = x * y) :
  25 ≤ 3 * x + 4 * y := by
  sorry

end min_value_3x_plus_4y_l1502_150208


namespace lillian_sugar_bags_lillian_sugar_bags_proof_l1502_150225

/-- Lillian's cupcake sugar problem -/
theorem lillian_sugar_bags : ℕ :=
  let sugar_at_home : ℕ := 3
  let sugar_per_bag : ℕ := 6
  let sugar_per_dozen_batter : ℕ := 1
  let sugar_per_dozen_frosting : ℕ := 2
  let dozens_of_cupcakes : ℕ := 5

  let total_sugar_needed := dozens_of_cupcakes * (sugar_per_dozen_batter + sugar_per_dozen_frosting)
  let sugar_to_buy := total_sugar_needed - sugar_at_home
  let bags_to_buy := sugar_to_buy / sugar_per_bag

  2

theorem lillian_sugar_bags_proof : lillian_sugar_bags = 2 := by
  sorry

end lillian_sugar_bags_lillian_sugar_bags_proof_l1502_150225


namespace our_circle_center_and_radius_l1502_150296

/-- A circle in the xy-plane --/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- The center of a circle --/
def center (c : Circle) : ℝ × ℝ := sorry

/-- The radius of a circle --/
def radius (c : Circle) : ℝ := sorry

/-- Our specific circle --/
def our_circle : Circle :=
  { equation := λ x y => x^2 + y^2 - 6*x = 0 }

theorem our_circle_center_and_radius :
  center our_circle = (3, 0) ∧ radius our_circle = 3 := by sorry

end our_circle_center_and_radius_l1502_150296


namespace quadrilateral_is_parallelogram_l1502_150297

-- Define the points
variable (A B C D M N P : ℝ × ℝ)

-- Define the conditions
def is_convex_quadrilateral (A B C D : ℝ × ℝ) : Prop := sorry

def is_midpoint (M B C : ℝ × ℝ) : Prop := sorry

def lines_intersect (A M B N P : ℝ × ℝ) : Prop := sorry

def ratio_equals (P M A : ℝ × ℝ) (r : ℚ) : Prop := sorry

def is_parallelogram (A B C D : ℝ × ℝ) : Prop := sorry

-- State the theorem
theorem quadrilateral_is_parallelogram 
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : is_midpoint M B C)
  (h3 : is_midpoint N C D)
  (h4 : lines_intersect A M B N P)
  (h5 : ratio_equals P M A (1/5))
  (h6 : ratio_equals B P N (2/5))
  : is_parallelogram A B C D := by sorry

end quadrilateral_is_parallelogram_l1502_150297


namespace odd_cube_minus_n_div_24_l1502_150293

theorem odd_cube_minus_n_div_24 (n : ℤ) (h : Odd n) : ∃ k : ℤ, n^3 - n = 24 * k := by
  sorry

end odd_cube_minus_n_div_24_l1502_150293


namespace smallest_integer_l1502_150243

theorem smallest_integer (a b : ℕ) (ha : a = 60) (h_lcm_gcd : Nat.lcm a b / Nat.gcd a b = 60) : 
  b ≥ 60 ∧ ∀ c : ℕ, c < 60 → Nat.lcm a c / Nat.gcd a c ≠ 60 := by
  sorry

end smallest_integer_l1502_150243


namespace f_le_one_l1502_150213

noncomputable def f (x : ℝ) : ℝ := (1 + Real.log x) / x

theorem f_le_one (x : ℝ) (hx : x > 0) : f x ≤ 1 := by
  sorry

end f_le_one_l1502_150213


namespace tan_five_pi_fourths_l1502_150209

theorem tan_five_pi_fourths : Real.tan (5 * π / 4) = 1 := by
  sorry

end tan_five_pi_fourths_l1502_150209


namespace baking_difference_l1502_150242

/-- Calculates the difference between remaining flour to be added and total sugar required -/
def flour_sugar_difference (total_flour sugar_required flour_added : ℕ) : ℤ :=
  (total_flour - flour_added : ℤ) - sugar_required

/-- Proves that the difference between remaining flour and total sugar is 1 cup -/
theorem baking_difference (total_flour sugar_required flour_added : ℕ) 
  (h1 : total_flour = 10)
  (h2 : sugar_required = 2)
  (h3 : flour_added = 7) :
  flour_sugar_difference total_flour sugar_required flour_added = 1 := by
  sorry

end baking_difference_l1502_150242


namespace geometric_series_sum_l1502_150273

/-- The sum of the geometric series 15 + 15r + 15r^2 + 15r^3 + ... for -1 < r < 1 -/
noncomputable def S (r : ℝ) : ℝ := 15 / (1 - r)

/-- For -1 < a < 1, if S(a)S(-a) = 2025, then S(a) + S(-a) = 270 -/
theorem geometric_series_sum (a : ℝ) (h1 : -1 < a) (h2 : a < 1) 
  (h3 : S a * S (-a) = 2025) : S a + S (-a) = 270 := by
  sorry

end geometric_series_sum_l1502_150273


namespace inequalities_theorem_l1502_150264

theorem inequalities_theorem (a b m : ℝ) :
  (b < a ∧ a < 0 → 1 / a < 1 / b) ∧
  (b > a ∧ a > 0 ∧ m > 0 → (a + m) / (b + m) > a / b) := by
  sorry

end inequalities_theorem_l1502_150264


namespace simplify_and_evaluate_evaluate_at_two_l1502_150222

theorem simplify_and_evaluate (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 0) :
  (1 - 1 / (x + 1)) / (x / (x^2 + 2*x + 1)) = x + 1 := by
  sorry

-- Evaluation for x = 2
theorem evaluate_at_two :
  (1 - 1 / (2 + 1)) / (2 / (2^2 + 2*2 + 1)) = 3 := by
  sorry

end simplify_and_evaluate_evaluate_at_two_l1502_150222


namespace watermelon_seeds_count_l1502_150246

/-- Represents a watermelon with its properties -/
structure Watermelon :=
  (slices : ℕ)
  (black_seeds_per_slice : ℕ)
  (white_seeds_per_slice : ℕ)

/-- Calculates the total number of seeds in a watermelon -/
def total_seeds (w : Watermelon) : ℕ :=
  w.slices * (w.black_seeds_per_slice + w.white_seeds_per_slice)

/-- Theorem stating that a watermelon with 40 slices, 20 black seeds and 20 white seeds per slice has 1600 total seeds -/
theorem watermelon_seeds_count :
  ∀ (w : Watermelon),
  w.slices = 40 →
  w.black_seeds_per_slice = 20 →
  w.white_seeds_per_slice = 20 →
  total_seeds w = 1600 :=
by
  sorry


end watermelon_seeds_count_l1502_150246


namespace triangle_area_solution_l1502_150298

/-- Given a triangle with vertices (0, 0), (x, 3x), and (x, 0), where x > 0,
    if the area of this triangle is 100 square units, then x = 10√6/3 -/
theorem triangle_area_solution (x : ℝ) (h1 : x > 0) : 
  (1/2 : ℝ) * x * (3*x) = 100 → x = (10 * Real.sqrt 6) / 3 := by
  sorry

#check triangle_area_solution

end triangle_area_solution_l1502_150298


namespace binomial_variance_four_third_l1502_150294

/-- A random variable following a binomial distribution B(n, p) -/
structure BinomialRV (n : ℕ) (p : ℝ) where
  (p_nonneg : 0 ≤ p)
  (p_le_one : p ≤ 1)

/-- The variance of a binomial random variable -/
def variance (n : ℕ) (p : ℝ) (ξ : BinomialRV n p) : ℝ :=
  n * p * (1 - p)

theorem binomial_variance_four_third (ξ : BinomialRV 4 (1/3)) :
  variance 4 (1/3) ξ = 8/9 := by
  sorry

end binomial_variance_four_third_l1502_150294


namespace evan_future_books_l1502_150278

/-- Calculates the number of books Evan will have in ten years given the initial conditions --/
def books_in_ten_years (initial_books : ℕ) (reduction : ℕ) (multiplier : ℕ) (addition : ℕ) : ℕ :=
  let current_books := initial_books - reduction
  let books_after_halving := current_books / 2
  multiplier * books_after_halving + addition

/-- Theorem stating that Evan will have 1080 books in ten years --/
theorem evan_future_books :
  books_in_ten_years 400 80 6 120 = 1080 := by
  sorry

#eval books_in_ten_years 400 80 6 120

end evan_future_books_l1502_150278


namespace sarah_interview_combinations_l1502_150289

/-- Represents the number of interview choices for each day of the week -/
structure WeekChoices where
  monday : Nat
  tuesday : Nat
  wednesday : Nat
  thursday : Nat
  friday : Nat

/-- Calculates the total number of interview combinations for the week -/
def totalCombinations (choices : WeekChoices) : Nat :=
  choices.monday * choices.tuesday * choices.wednesday * choices.thursday * choices.friday

/-- Represents Sarah's interview choices for the week -/
def sarahChoices : WeekChoices :=
  { monday := 1
  , tuesday := 2
  , wednesday := 5  -- 2 + 3, accounting for both Tuesday possibilities
  , thursday := 5
  , friday := 1 }  -- No interviews, but included for completeness

/-- Theorem stating that Sarah's total interview combinations is 50 -/
theorem sarah_interview_combinations :
  totalCombinations sarahChoices = 50 := by
  sorry

#eval totalCombinations sarahChoices  -- Should output 50

end sarah_interview_combinations_l1502_150289


namespace roberto_outfits_l1502_150280

/-- The number of different outfits Roberto can assemble -/
def number_of_outfits : ℕ :=
  let trousers : ℕ := 4
  let shirts : ℕ := 8
  let jackets : ℕ := 3
  let belts : ℕ := 2
  trousers * shirts * jackets * belts

theorem roberto_outfits :
  number_of_outfits = 192 := by
  sorry

end roberto_outfits_l1502_150280


namespace pages_per_notebook_l1502_150285

/-- Given that James buys 2 notebooks, pays $5 in total, and each page costs 5 cents,
    prove that the number of pages in each notebook is 50. -/
theorem pages_per_notebook :
  let notebooks : ℕ := 2
  let total_cost : ℕ := 500  -- in cents
  let cost_per_page : ℕ := 5 -- in cents
  let total_pages : ℕ := total_cost / cost_per_page
  let pages_per_notebook : ℕ := total_pages / notebooks
  pages_per_notebook = 50 := by
sorry

end pages_per_notebook_l1502_150285


namespace investment_plans_count_l1502_150251

theorem investment_plans_count (n_projects : ℕ) (n_cities : ℕ) (max_per_city : ℕ) : 
  n_projects = 3 → n_cities = 5 → max_per_city = 2 →
  (Nat.choose n_cities 3 * Nat.factorial 3 + 
   Nat.choose n_cities 1 * Nat.choose (n_cities - 1) 1 * 3) = 120 := by
  sorry

end investment_plans_count_l1502_150251


namespace diorama_building_time_l1502_150249

/-- Represents the time spent on the diorama project -/
structure DioramaTime where
  planning : ℕ  -- Planning time in minutes
  building : ℕ  -- Building time in minutes

/-- Defines the conditions of the diorama project -/
def validDioramaTime (t : DioramaTime) : Prop :=
  t.building = 3 * t.planning - 5 ∧
  t.building + t.planning = 67

/-- Theorem stating that the building time is 49 minutes -/
theorem diorama_building_time :
  ∀ t : DioramaTime, validDioramaTime t → t.building = 49 :=
by
  sorry


end diorama_building_time_l1502_150249


namespace other_number_problem_l1502_150241

theorem other_number_problem (a b : ℕ) : 
  a + b = 96 → 
  (a = b + 12 ∨ b = a + 12) → 
  (a = 42 ∨ b = 42) → 
  (a = 54 ∨ b = 54) :=
by sorry

end other_number_problem_l1502_150241


namespace percentage_calculation_l1502_150202

theorem percentage_calculation (total : ℝ) (result : ℝ) (percentage : ℝ) :
  total = 50 →
  result = 2.125 →
  percentage = 4.25 →
  (percentage / 100) * total = result :=
by
  sorry

end percentage_calculation_l1502_150202


namespace number_of_amateurs_l1502_150239

/-- The number of chess amateurs in the tournament -/
def n : ℕ := sorry

/-- The number of other amateurs each amateur plays with -/
def games_per_amateur : ℕ := 4

/-- The total number of possible games in the tournament -/
def total_games : ℕ := 10

/-- Theorem stating the number of chess amateurs in the tournament -/
theorem number_of_amateurs :
  n = 5 ∧
  games_per_amateur = 4 ∧
  total_games = 10 ∧
  n.choose 2 = total_games :=
sorry

end number_of_amateurs_l1502_150239


namespace complex_number_problems_l1502_150238

open Complex

theorem complex_number_problems (z₁ z₂ z : ℂ) (b : ℝ) :
  z₁ = 1 - I ∧ z₂ = 4 + 6 * I ∧ z = 1 + b * I ∧ (z + z₁).im = 0 →
  z₂ / z₁ = -1 + 5 * I ∧ abs z = Real.sqrt 2 := by
  sorry

end complex_number_problems_l1502_150238


namespace intersection_of_parallel_lines_l1502_150255

/-- The number of parallelograms formed by the intersection of two sets of parallel lines -/
def parallelograms (n m : ℕ) : ℕ := n.choose 2 * m

/-- Given two sets of parallel lines intersecting in a plane, 
    where one set has 8 lines and they form 280 parallelograms, 
    prove that the other set must have 10 lines -/
theorem intersection_of_parallel_lines : 
  ∃ (n : ℕ), n > 0 ∧ parallelograms n 8 = 280 ∧ n = 10 :=
sorry

end intersection_of_parallel_lines_l1502_150255


namespace committee_size_l1502_150206

theorem committee_size (n : ℕ) : 
  (n * (n - 1) = 42) → n = 7 := by
  sorry

#check committee_size

end committee_size_l1502_150206


namespace sum_interior_angles_formula_l1502_150287

/-- The sum of interior angles of an n-sided polygon -/
def sum_interior_angles (n : ℕ) : ℝ :=
  (n - 2) * 180

/-- Theorem: The sum of the interior angles of an n-sided polygon is (n-2) × 180° -/
theorem sum_interior_angles_formula (n : ℕ) (h : n ≥ 3) :
  sum_interior_angles n = (n - 2) * 180 := by
  sorry

end sum_interior_angles_formula_l1502_150287


namespace min_value_theorem_l1502_150291

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  1 / (b^2 + b*c) + 1 / (c^2 + c*a) + 1 / (a^2 + a*b) ≥ 3/2 := by
  sorry

end min_value_theorem_l1502_150291


namespace purple_balls_count_l1502_150223

theorem purple_balls_count (total_balls : ℕ) (white_balls : ℕ) (green_balls : ℕ) (yellow_balls : ℕ) (red_balls : ℕ) 
  (h1 : total_balls = 60)
  (h2 : white_balls = 22)
  (h3 : green_balls = 18)
  (h4 : yellow_balls = 2)
  (h5 : red_balls = 15)
  (h6 : (white_balls + green_balls + yellow_balls : ℚ) / total_balls = 7/10) :
  ∃ (purple_balls : ℕ), purple_balls = 3 ∧ total_balls = white_balls + green_balls + yellow_balls + red_balls + purple_balls :=
by
  sorry


end purple_balls_count_l1502_150223


namespace difference_solution_eq_one_difference_solution_eq_two_difference_solution_eq_three_l1502_150261

/-- Definition of a difference solution equation -/
def is_difference_solution_equation (a b : ℝ) : Prop :=
  b / a = b - a

/-- Theorem for 4x = m -/
theorem difference_solution_eq_one (m : ℝ) :
  is_difference_solution_equation 4 m ↔ m = 16 / 3 := by sorry

/-- Theorem for 4x = ab + a -/
theorem difference_solution_eq_two (a b : ℝ) :
  is_difference_solution_equation 4 (a * b + a) → 3 * (a * b + a) = 16 := by sorry

/-- Theorem for 4x = mn + m and -2x = mn + n -/
theorem difference_solution_eq_three (m n : ℝ) :
  is_difference_solution_equation 4 (m * n + m) →
  is_difference_solution_equation (-2) (m * n + n) →
  3 * (m * n + m) - 9 * (m * n + n)^2 = 0 := by sorry

end difference_solution_eq_one_difference_solution_eq_two_difference_solution_eq_three_l1502_150261


namespace move_point_l1502_150295

/-- Moving a point left decreases its x-coordinate --/
def move_left (x : ℝ) (units : ℝ) : ℝ := x - units

/-- Moving a point up increases its y-coordinate --/
def move_up (y : ℝ) (units : ℝ) : ℝ := y + units

/-- A 2D point --/
structure Point where
  x : ℝ
  y : ℝ

/-- The initial point P --/
def P : Point := ⟨-2, -3⟩

/-- Theorem: Moving P 1 unit left and 3 units up results in (-3, 0) --/
theorem move_point :
  let new_x := move_left P.x 1
  let new_y := move_up P.y 3
  (new_x, new_y) = (-3, 0) := by sorry

end move_point_l1502_150295


namespace system_solution_difference_l1502_150211

theorem system_solution_difference (a b x y : ℝ) : 
  (2 * x + y = b) → 
  (x - b * y = a) → 
  (x = 1) → 
  (y = 0) → 
  (a - b = -1) := by
sorry

end system_solution_difference_l1502_150211


namespace lily_reads_28_books_l1502_150271

/-- Represents Lily's reading habits and goals over two months -/
structure LilyReading where
  last_month_weekday : Nat
  last_month_weekend : Nat
  this_month_weekday_factor : Nat
  this_month_weekend_factor : Nat

/-- Calculates the total number of books Lily reads in two months -/
def total_books_read (r : LilyReading) : Nat :=
  let last_month_total := r.last_month_weekday + r.last_month_weekend
  let this_month_weekday := r.last_month_weekday * r.this_month_weekday_factor
  let this_month_weekend := r.last_month_weekend * r.this_month_weekend_factor
  let this_month_total := this_month_weekday + this_month_weekend
  last_month_total + this_month_total

/-- Theorem stating that Lily reads 28 books in total over two months -/
theorem lily_reads_28_books :
  ∀ (r : LilyReading),
    r.last_month_weekday = 4 →
    r.last_month_weekend = 4 →
    r.this_month_weekday_factor = 2 →
    r.this_month_weekend_factor = 3 →
    total_books_read r = 28 :=
  sorry


end lily_reads_28_books_l1502_150271


namespace expected_heads_value_l1502_150267

/-- The probability of a coin landing heads -/
def p_heads : ℚ := 1/3

/-- The number of coins -/
def num_coins : ℕ := 100

/-- The maximum number of flips allowed for each coin -/
def max_flips : ℕ := 4

/-- The probability of a coin showing heads after up to four flips -/
def p_heads_after_four_flips : ℚ :=
  p_heads + (1 - p_heads) * p_heads + (1 - p_heads)^2 * p_heads + (1 - p_heads)^3 * p_heads

/-- The expected number of coins showing heads after all flips -/
def expected_heads : ℚ := num_coins * p_heads_after_four_flips

theorem expected_heads_value : expected_heads = 6500/81 := by
  sorry

end expected_heads_value_l1502_150267


namespace range_of_m_l1502_150252

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x - 3| ≥ |m - 1|) → 
  -3 ≤ m ∧ m ≤ 5 := by
sorry

end range_of_m_l1502_150252


namespace tangent_sphere_radius_l1502_150201

/-- A truncated cone with a sphere tangent to its surfaces -/
structure TruncatedConeWithSphere where
  bottom_radius : ℝ
  top_radius : ℝ
  slant_height : ℝ
  sphere_radius : ℝ

/-- The sphere is tangent to the top, bottom, and lateral surface of the truncated cone -/
def is_tangent_sphere (cone : TruncatedConeWithSphere) : Prop :=
  cone.sphere_radius > 0 ∧
  cone.sphere_radius ≤ cone.bottom_radius ∧
  cone.sphere_radius ≤ cone.top_radius ∧
  cone.sphere_radius ≤ cone.slant_height

/-- The theorem stating the radius of the tangent sphere -/
theorem tangent_sphere_radius (cone : TruncatedConeWithSphere) 
  (h1 : cone.bottom_radius = 20)
  (h2 : cone.top_radius = 5)
  (h3 : cone.slant_height = 25)
  (h4 : is_tangent_sphere cone) :
  cone.sphere_radius = 10 := by
  sorry

end tangent_sphere_radius_l1502_150201


namespace completing_square_equivalence_l1502_150266

theorem completing_square_equivalence (x : ℝ) : 
  (x^2 + 6*x + 3 = 0) ↔ ((x + 3)^2 = 6) :=
by sorry

end completing_square_equivalence_l1502_150266


namespace gcd_204_85_l1502_150218

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := by
  sorry

end gcd_204_85_l1502_150218


namespace scalene_triangle_c_equals_four_l1502_150299

/-- A scalene triangle with integer side lengths satisfying a specific equation -/
structure ScaleneTriangle where
  a : ℤ
  b : ℤ
  c : ℤ
  scalene : a ≠ b ∧ b ≠ c ∧ a ≠ c
  equation : a^2 + b^2 - 6*a - 4*b + 13 = 0

/-- Theorem: If a scalene triangle satisfies the given equation, then c = 4 -/
theorem scalene_triangle_c_equals_four (t : ScaleneTriangle) : t.c = 4 := by
  sorry

end scalene_triangle_c_equals_four_l1502_150299


namespace cubic_roots_of_27_l1502_150290

theorem cubic_roots_of_27 :
  let z₁ : ℂ := 3
  let z₂ : ℂ := -3/2 + (3*Complex.I*Real.sqrt 3)/2
  let z₃ : ℂ := -3/2 - (3*Complex.I*Real.sqrt 3)/2
  (z₁^3 = 27 ∧ z₂^3 = 27 ∧ z₃^3 = 27) ∧
  ∀ z : ℂ, z^3 = 27 → (z = z₁ ∨ z = z₂ ∨ z = z₃) :=
by sorry

end cubic_roots_of_27_l1502_150290


namespace function_properties_l1502_150233

def f (x : ℝ) := -5 * x + 1

theorem function_properties :
  (∃ (x₁ x₂ x₃ : ℝ), f x₁ > 0 ∧ x₁ > 0 ∧ f x₂ < 0 ∧ x₂ > 0 ∧ f x₃ < 0 ∧ x₃ < 0) ∧
  (∀ x : ℝ, x > 1 → f x < 0) :=
sorry

end function_properties_l1502_150233


namespace gcd_of_polynomial_and_multiple_l1502_150220

theorem gcd_of_polynomial_and_multiple : ∀ x : ℤ, 
  18432 ∣ x → 
  Nat.gcd (Int.natAbs ((3*x+5)*(7*x+2)*(13*x+7)*(2*x+10))) (Int.natAbs x) = 28 :=
by sorry

end gcd_of_polynomial_and_multiple_l1502_150220


namespace water_intake_calculation_l1502_150235

theorem water_intake_calculation (morning_intake : Real) 
  (h1 : morning_intake = 1.5)
  (h2 : afternoon_intake = 3 * morning_intake)
  (h3 : evening_intake = 0.5 * afternoon_intake) :
  morning_intake + afternoon_intake + evening_intake = 8.25 := by
  sorry

end water_intake_calculation_l1502_150235


namespace largest_angle_in_triangle_l1502_150205

theorem largest_angle_in_triangle : ∀ (a b c : ℝ),
  a + b + c = 180 →  -- Sum of angles in a triangle is 180°
  a + b = 120 →      -- Sum of two angles is 4/3 of right angle (90° * 4/3 = 120°)
  b = a + 36 →       -- One angle is 36° larger than the other
  max a (max b c) = 78 := by
sorry

end largest_angle_in_triangle_l1502_150205


namespace opposite_unit_vector_l1502_150207

def vector_a : Fin 2 → ℝ := ![4, 2]

theorem opposite_unit_vector :
  let magnitude := Real.sqrt (vector_a 0 ^ 2 + vector_a 1 ^ 2)
  let opposite_unit_vector := fun i => -vector_a i / magnitude
  opposite_unit_vector 0 = -2 * Real.sqrt 5 / 5 ∧
  opposite_unit_vector 1 = -Real.sqrt 5 / 5 := by
  sorry

end opposite_unit_vector_l1502_150207


namespace quadratic_equation_roots_l1502_150265

theorem quadratic_equation_roots :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (3 * x₁^2 - 2 * x₁ - 1 = 0) ∧ 
  (3 * x₂^2 - 2 * x₂ - 1 = 0) := by
  sorry

end quadratic_equation_roots_l1502_150265


namespace simplify_expression_l1502_150270

theorem simplify_expression : 
  (Real.sqrt (32^(1/5)) - Real.sqrt 7)^2 = 11 - 4 * Real.sqrt 7 := by
  sorry

end simplify_expression_l1502_150270


namespace remainder_problem_l1502_150284

theorem remainder_problem (n : ℤ) (h : n % 7 = 5) : (3 * n + 2) % 7 = 3 := by
  sorry

end remainder_problem_l1502_150284


namespace product_calculation_l1502_150227

theorem product_calculation : 10 * 0.2 * 0.5 * 4 / 2 = 2 := by
  sorry

end product_calculation_l1502_150227


namespace cookies_left_for_birthday_l1502_150245

theorem cookies_left_for_birthday 
  (pans : ℕ) 
  (cookies_per_pan : ℕ) 
  (eaten_cookies : ℕ) 
  (burnt_cookies : ℕ) 
  (h1 : pans = 12)
  (h2 : cookies_per_pan = 15)
  (h3 : eaten_cookies = 9)
  (h4 : burnt_cookies = 6) :
  (pans * cookies_per_pan) - (eaten_cookies + burnt_cookies) = 165 := by
  sorry

end cookies_left_for_birthday_l1502_150245


namespace project_estimated_hours_l1502_150250

/-- The number of extra hours Anie needs to work each day -/
def extra_hours : ℕ := 5

/-- The number of hours in Anie's normal work schedule each day -/
def normal_hours : ℕ := 10

/-- The number of days it would take Anie to finish the job -/
def days_to_finish : ℕ := 100

/-- The total number of hours Anie works each day -/
def total_hours_per_day : ℕ := normal_hours + extra_hours

/-- Theorem: The project is estimated to take 1500 hours -/
theorem project_estimated_hours : 
  days_to_finish * total_hours_per_day = 1500 := by
  sorry


end project_estimated_hours_l1502_150250


namespace perfect_square_condition_l1502_150214

/-- A polynomial is a perfect square trinomial if it can be written as (px + q)^2 for some real p and q. -/
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  ∃ p q : ℝ, ∀ x, a * x^2 + b * x + c = (p * x + q)^2

/-- If 4x^2 + bx + 1 is a perfect square trinomial, then b = ±4. -/
theorem perfect_square_condition (b : ℝ) :
  is_perfect_square_trinomial 4 b 1 → b = 4 ∨ b = -4 := by
  sorry

#check perfect_square_condition

end perfect_square_condition_l1502_150214


namespace volleyball_teams_l1502_150283

theorem volleyball_teams (managers : ℕ) (employees : ℕ) (team_size : ℕ) : 
  managers = 23 → employees = 7 → team_size = 5 → 
  (managers + employees) / team_size = 6 := by
  sorry

end volleyball_teams_l1502_150283


namespace nested_root_simplification_l1502_150247

theorem nested_root_simplification (y : ℝ) (h : y ≥ 0) :
  Real.sqrt (y * Real.sqrt (y^3 * Real.sqrt (y^5))) = (y^15)^(1/8) := by
  sorry

end nested_root_simplification_l1502_150247


namespace fixed_point_theorem_l1502_150237

/-- Given a triangle ABC and a point M, prove that a certain line always passes through a fixed point -/
theorem fixed_point_theorem (a b c t m : ℝ) : 
  let A : ℝ × ℝ := (0, a)
  let B : ℝ × ℝ := (b, 0)
  let C : ℝ × ℝ := (c, 0)
  let M : ℝ × ℝ := (t, m)
  let D : ℝ × ℝ := ((b + c) / 3, a / 3)  -- Centroid
  let E : ℝ × ℝ := ((t + b) / 2, m / 2)  -- Midpoint of MB
  let F : ℝ × ℝ := ((t + c) / 2, m / 2)  -- Midpoint of MC
  let P : ℝ × ℝ := ((t + b) / 2, a * (1 - (t + b) / (2 * b)))  -- Intersection of AB and perpendicular through E
  let Q : ℝ × ℝ := ((t + c) / 2, a * (1 - (t + c) / (2 * c)))  -- Intersection of AC and perpendicular through F
  let slope_PQ : ℝ := (a * t) / (b * c)
  let perpendicular_slope : ℝ := -b * c / (a * t)
  True → ∃ k : ℝ, (0, m + b * c / a) = (t + k, m + k * perpendicular_slope) :=
by
  sorry


end fixed_point_theorem_l1502_150237


namespace kate_change_l1502_150200

/-- The amount Kate gave to the clerk in cents -/
def amount_given : ℕ := 100

/-- The cost of Kate's candy in cents -/
def candy_cost : ℕ := 54

/-- The change Kate should receive in cents -/
def change : ℕ := amount_given - candy_cost

theorem kate_change : change = 46 := by
  sorry

end kate_change_l1502_150200


namespace complex_polynomial_solution_l1502_150228

theorem complex_polynomial_solution (c₀ c₁ c₂ c₃ c₄ a b : ℝ) :
  let z : ℂ := Complex.mk a b
  let i : ℂ := Complex.I
  let f : ℂ → ℂ := λ w => c₄ * w^4 + i * c₃ * w^3 + c₂ * w^2 + i * c₁ * w + c₀
  f z = 0 → f (Complex.mk (-a) b) = 0 := by sorry

end complex_polynomial_solution_l1502_150228


namespace model2_best_fit_l1502_150217

-- Define the coefficient of determination for each model
def R2_model1 : ℝ := 0.78
def R2_model2 : ℝ := 0.85
def R2_model3 : ℝ := 0.61
def R2_model4 : ℝ := 0.31

-- Define a function to calculate the distance from 1
def distance_from_one (x : ℝ) : ℝ := |1 - x|

-- Theorem stating that Model 2 has the best fitting effect
theorem model2_best_fit :
  distance_from_one R2_model2 < distance_from_one R2_model1 ∧
  distance_from_one R2_model2 < distance_from_one R2_model3 ∧
  distance_from_one R2_model2 < distance_from_one R2_model4 :=
by sorry


end model2_best_fit_l1502_150217


namespace mental_health_survey_is_comprehensive_l1502_150257

/-- Represents a survey --/
structure Survey where
  description : String
  population : Set String
  environment : String

/-- Conditions for a comprehensive survey --/
def is_comprehensive (s : Survey) : Prop :=
  s.population.Finite ∧
  s.population.Nonempty ∧
  (∀ x ∈ s.population, ∃ y, y = x) ∧
  s.environment = "Contained"

/-- The survey on students' mental health --/
def mental_health_survey : Survey :=
  { description := "Survey on the current status of students' mental health in a school in Huicheng District"
  , population := {"Students in a school in Huicheng District"}
  , environment := "Contained" }

/-- Theorem stating that the mental health survey is comprehensive --/
theorem mental_health_survey_is_comprehensive :
  is_comprehensive mental_health_survey :=
sorry

end mental_health_survey_is_comprehensive_l1502_150257


namespace smallest_positive_d_l1502_150234

theorem smallest_positive_d : ∃ d : ℝ,
  d > 0 ∧
  (2 * Real.sqrt 7)^2 + (d + 5)^2 = (2 * d + 1)^2 ∧
  ∀ d' : ℝ, d' > 0 → (2 * Real.sqrt 7)^2 + (d' + 5)^2 = (2 * d' + 1)^2 → d ≤ d' ∧
  d = 1 + Real.sqrt 660 / 6 :=
by sorry

end smallest_positive_d_l1502_150234


namespace existence_of_point_l1502_150224

theorem existence_of_point (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  ∃ x ∈ Set.Icc 0 1, (4 / Real.pi) * (f 1 - f 0) = (1 + x^2) * (deriv f x) := by
  sorry

end existence_of_point_l1502_150224


namespace contractor_problem_l1502_150253

/-- The number of days initially planned to complete the work -/
def initial_days : ℕ := 15

/-- The number of absent laborers -/
def absent_laborers : ℕ := 5

/-- The number of days taken to complete the work with reduced laborers -/
def actual_days : ℕ := 20

/-- The original number of laborers employed -/
def original_laborers : ℕ := 15

theorem contractor_problem :
  (original_laborers - absent_laborers) * initial_days = original_laborers * actual_days :=
sorry

end contractor_problem_l1502_150253


namespace three_lines_intersection_l1502_150210

theorem three_lines_intersection (x : ℝ) : 
  (∀ (a b c d e f : ℝ), a = x ∧ b = x ∧ c = x ∧ d = x ∧ e = x ∧ f = x) →  -- opposite angles are equal
  (a + b + c + d + e + f = 360) →                                       -- sum of angles around a point is 360°
  x = 60 :=                                                            -- prove that x = 60°
by
  sorry

end three_lines_intersection_l1502_150210


namespace rectangle_dimension_change_l1502_150216

theorem rectangle_dimension_change (L B : ℝ) (L' B' : ℝ) (h_positive : L > 0 ∧ B > 0) :
  B' = 1.15 * B →
  L' * B' = 1.2075 * (L * B) →
  L' = 1.05 * L := by
sorry

end rectangle_dimension_change_l1502_150216


namespace arithmetic_equality_l1502_150229

theorem arithmetic_equality : 253 - 47 + 29 + 18 = 253 := by sorry

end arithmetic_equality_l1502_150229


namespace zero_function_theorem_l1502_150212

theorem zero_function_theorem (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h1 : ∀ x : ℤ, deriv f x = 0)
  (h2 : ∀ x : ℝ, deriv f x = 0 → f x = 0) :
  ∀ x : ℝ, f x = 0 := by
sorry

end zero_function_theorem_l1502_150212


namespace nPointedStar_interiorAngleSum_l1502_150248

/-- Represents an n-pointed star formed from an n-sided convex polygon -/
structure NPointedStar where
  n : ℕ
  h_n : n ≥ 6

/-- The sum of interior angles at the vertices of an n-pointed star -/
def interiorAngleSum (star : NPointedStar) : ℝ :=
  180 * (star.n - 2)

/-- Theorem: The sum of interior angles at the vertices of an n-pointed star
    formed by extending every third side of an n-sided convex polygon (n ≥ 6)
    is equal to 180°(n-2) -/
theorem nPointedStar_interiorAngleSum (star : NPointedStar) :
  interiorAngleSum star = 180 * (star.n - 2) := by
  sorry

end nPointedStar_interiorAngleSum_l1502_150248


namespace machine_A_production_rate_l1502_150260

-- Define the production rates and times for machines A, P, and Q
variable (A : ℝ) -- Production rate of Machine A (sprockets per hour)
variable (P : ℝ) -- Production rate of Machine P (sprockets per hour)
variable (Q : ℝ) -- Production rate of Machine Q (sprockets per hour)
variable (T_Q : ℝ) -- Time taken by Machine Q to produce 440 sprockets

-- State the conditions
axiom total_sprockets : 440 = Q * T_Q
axiom time_difference : 440 = P * (T_Q + 10)
axiom production_ratio : Q = 1.1 * A

-- State the theorem to be proved
theorem machine_A_production_rate : A = 4 := by
  sorry

end machine_A_production_rate_l1502_150260


namespace minimum_cost_is_correct_l1502_150279

/-- Represents the dimensions and cost of a box --/
structure Box where
  length : ℕ
  width : ℕ
  height : ℕ
  cost : ℚ

/-- Represents the capacity of a box for different painting sizes --/
structure BoxCapacity where
  small : ℕ
  medium : ℕ
  large : ℕ

/-- Represents the collection of paintings --/
structure PaintingCollection where
  small : ℕ
  medium : ℕ
  large : ℕ

def smallBox : Box := ⟨20, 20, 15, 4/5⟩
def mediumBox : Box := ⟨22, 22, 17, 11/10⟩
def largeBox : Box := ⟨24, 24, 20, 27/20⟩

def smallBoxCapacity : BoxCapacity := ⟨3, 2, 0⟩
def mediumBoxCapacity : BoxCapacity := ⟨5, 4, 3⟩
def largeBoxCapacity : BoxCapacity := ⟨8, 6, 5⟩

def collection : PaintingCollection := ⟨1350, 2700, 3150⟩

/-- Calculates the minimum cost to move the entire collection --/
def minimumCost (collection : PaintingCollection) (largeBox : Box) (largeBoxCapacity : BoxCapacity) : ℚ :=
  let smallBoxes := (collection.small + largeBoxCapacity.small - 1) / largeBoxCapacity.small
  let mediumBoxes := (collection.medium + largeBoxCapacity.medium - 1) / largeBoxCapacity.medium
  let largeBoxes := (collection.large + largeBoxCapacity.large - 1) / largeBoxCapacity.large
  (smallBoxes + mediumBoxes + largeBoxes) * largeBox.cost

theorem minimum_cost_is_correct :
  minimumCost collection largeBox largeBoxCapacity = 1686.15 := by
  sorry

end minimum_cost_is_correct_l1502_150279


namespace element_in_set_l1502_150269

def U : Set Nat := {1, 2, 3, 4, 5}

theorem element_in_set (M : Set Nat) (h : Set.compl M = {1, 3}) : 2 ∈ M := by
  sorry

end element_in_set_l1502_150269


namespace betty_pays_nothing_l1502_150281

-- Define the ages and cost
def doug_age : ℕ := 40
def alice_age : ℕ := doug_age / 2
def total_age_sum : ℕ := 130
def cost_decrease_per_year : ℕ := 5

-- Define Betty's age
def betty_age : ℕ := total_age_sum - doug_age - alice_age

-- Define the original cost of a pack of nuts
def original_nut_cost : ℕ := 2 * betty_age

-- Define the age difference between Betty and Alice
def age_difference : ℕ := betty_age - alice_age

-- Define the total cost decrease
def total_cost_decrease : ℕ := age_difference * cost_decrease_per_year

-- Define the new cost of a pack of nuts
def new_nut_cost : ℕ := max 0 (original_nut_cost - total_cost_decrease)

-- Theorem to prove
theorem betty_pays_nothing : new_nut_cost * 20 = 0 := by
  sorry

end betty_pays_nothing_l1502_150281


namespace hotel_room_charges_l1502_150263

theorem hotel_room_charges (P R G : ℝ) 
  (h1 : P = R - 0.5 * R) 
  (h2 : P = G - 0.1 * G) : 
  R = G + 0.8 * G := by
sorry

end hotel_room_charges_l1502_150263


namespace three_digit_divisible_by_17_l1502_150231

theorem three_digit_divisible_by_17 : 
  (Finset.filter (fun k => 100 ≤ 17 * k ∧ 17 * k ≤ 999) (Finset.range 1000)).card = 53 := by
  sorry

end three_digit_divisible_by_17_l1502_150231


namespace smallest_number_in_S_l1502_150275

def S : Set ℝ := {3.2, 2.3, 3, 2.23, 3.22}

theorem smallest_number_in_S : 
  ∃ (x : ℝ), x ∈ S ∧ ∀ y ∈ S, x ≤ y ∧ x = 2.23 := by
  sorry

end smallest_number_in_S_l1502_150275


namespace sticker_distribution_l1502_150286

/-- The number of ways to distribute n identical objects into k distinct bins -/
def stars_and_bars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The problem statement -/
theorem sticker_distribution :
  stars_and_bars 10 4 = Nat.choose 13 3 := by
  sorry

end sticker_distribution_l1502_150286


namespace parallel_planes_k_value_l1502_150203

/-- Given two planes α and β with normal vectors n₁ and n₂ respectively,
    prove that if the planes are parallel, then k = 6. -/
theorem parallel_planes_k_value (n₁ n₂ : ℝ × ℝ × ℝ) (k : ℝ) :
  n₁ = (1, 2, -3) →
  n₂ = (-2, -4, k) →
  (∃ (c : ℝ), c ≠ 0 ∧ n₁ = c • n₂) →
  k = 6 := by sorry

end parallel_planes_k_value_l1502_150203


namespace two_balls_same_box_probability_l1502_150254

theorem two_balls_same_box_probability :
  let num_balls : ℕ := 3
  let num_boxes : ℕ := 5
  let total_outcomes : ℕ := num_boxes ^ num_balls
  let favorable_outcomes : ℕ := (num_balls.choose 2) * num_boxes * (num_boxes - 1)
  favorable_outcomes / total_outcomes = 12 / 25 := by
sorry

end two_balls_same_box_probability_l1502_150254


namespace apple_value_in_cake_slices_l1502_150236

/-- Represents the value of one apple in terms of cake slices -/
def apple_value : ℚ := 15 / 4

/-- Represents the number of apples that can be traded for juice bottles -/
def apples_per_juice_trade : ℕ := 4

/-- Represents the number of juice bottles received in trade for apples -/
def juice_bottles_per_apple_trade : ℕ := 3

/-- Represents the number of cake slices that can be traded for one juice bottle -/
def cake_slices_per_juice_bottle : ℕ := 5

theorem apple_value_in_cake_slices :
  apple_value = (juice_bottles_per_apple_trade * cake_slices_per_juice_bottle : ℚ) / apples_per_juice_trade :=
sorry

#eval apple_value -- Should output 3.75

end apple_value_in_cake_slices_l1502_150236


namespace negation_of_forall_positive_negation_of_greater_than_zero_l1502_150221

theorem negation_of_forall_positive (P : ℝ → Prop) :
  (¬ ∀ x : ℝ, P x) ↔ (∃ x : ℝ, ¬ P x) :=
by sorry

theorem negation_of_greater_than_zero :
  (¬ ∀ x : ℝ, x^2 + x + 1 > 0) ↔ (∃ x : ℝ, x^2 + x + 1 ≤ 0) :=
by sorry

end negation_of_forall_positive_negation_of_greater_than_zero_l1502_150221


namespace sum_of_five_cubes_l1502_150226

theorem sum_of_five_cubes (n : ℤ) : ∃ (a b c d e : ℤ), n = a^3 + b^3 + c^3 + d^3 + e^3 := by
  sorry

end sum_of_five_cubes_l1502_150226


namespace a_sufficient_not_necessary_l1502_150215

/-- The function f(x) = x³ + a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a

/-- f is strictly increasing on ℝ -/
def strictly_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem a_sufficient_not_necessary :
  (∃ a : ℝ, a > 1 ∧ strictly_increasing (f a)) ∧
  (∃ a : ℝ, strictly_increasing (f a) ∧ ¬(a > 1)) :=
sorry

end a_sufficient_not_necessary_l1502_150215


namespace rods_in_one_mile_l1502_150274

/-- Conversion factor from miles to chains -/
def mile_to_chain : ℚ := 10

/-- Conversion factor from chains to rods -/
def chain_to_rod : ℚ := 22

/-- The number of rods in one mile -/
def rods_in_mile : ℚ := mile_to_chain * chain_to_rod

theorem rods_in_one_mile :
  rods_in_mile = 220 :=
by sorry

end rods_in_one_mile_l1502_150274
