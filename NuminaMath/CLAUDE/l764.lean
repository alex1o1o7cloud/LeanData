import Mathlib

namespace NUMINAMATH_CALUDE_pants_cost_is_20_l764_76493

/-- Represents the cost of a school uniform -/
structure UniformCost where
  pants : ℝ
  shirt : ℝ
  tie : ℝ
  socks : ℝ

/-- Calculates the total cost of a given number of uniforms -/
def totalCost (u : UniformCost) (n : ℕ) : ℝ :=
  n * (u.pants + u.shirt + u.tie + u.socks)

/-- Theorem: Given the uniform cost constraints, the pants cost $20 -/
theorem pants_cost_is_20 :
  ∃ (u : UniformCost),
    u.shirt = 2 * u.pants ∧
    u.tie = (1/5) * u.shirt ∧
    u.socks = 3 ∧
    totalCost u 5 = 355 ∧
    u.pants = 20 := by
  sorry

end NUMINAMATH_CALUDE_pants_cost_is_20_l764_76493


namespace NUMINAMATH_CALUDE_min_sum_squares_l764_76450

theorem min_sum_squares (a b c t : ℝ) (h : a + b + c = t) :
  ∃ (min : ℝ), min = t^2 / 3 ∧ 
  (∀ (x y z : ℝ), x + y + z = t → x^2 + y^2 + z^2 ≥ min) ∧
  (∃ (x y z : ℝ), x + y + z = t ∧ x^2 + y^2 + z^2 = min) := by
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l764_76450


namespace NUMINAMATH_CALUDE_range_of_3a_plus_2b_l764_76422

theorem range_of_3a_plus_2b (a b : ℝ) (h : a^2 + b^2 = 4) :
  ∃ (x : ℝ), x ∈ Set.Icc (-2 * Real.sqrt 13) (2 * Real.sqrt 13) ∧ x = 3*a + 2*b :=
by sorry

end NUMINAMATH_CALUDE_range_of_3a_plus_2b_l764_76422


namespace NUMINAMATH_CALUDE_exact_four_white_probability_l764_76481

-- Define the number of balls
def n : ℕ := 8

-- Define the probability of a ball being white (or black)
def p : ℚ := 1/2

-- Define the number of white balls we're interested in
def k : ℕ := 4

-- State the theorem
theorem exact_four_white_probability :
  (n.choose k) * p^k * (1 - p)^(n - k) = 35/128 := by
  sorry

end NUMINAMATH_CALUDE_exact_four_white_probability_l764_76481


namespace NUMINAMATH_CALUDE_parabola_directrix_l764_76456

/-- The equation of the directrix of the parabola y = x^2 is 4y + 1 = 0 -/
theorem parabola_directrix : 
  ∀ (x y : ℝ), y = x^2 → (∃ (k : ℝ), k * y + 1 = 0 ∧ k = 4) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l764_76456


namespace NUMINAMATH_CALUDE_simplify_product_of_square_roots_l764_76412

theorem simplify_product_of_square_roots (x : ℝ) (h : x > 0) :
  Real.sqrt (50 * x^3) * Real.sqrt (18 * x^2) * Real.sqrt (35 * x) = 30 * x^3 * Real.sqrt 35 := by
  sorry

end NUMINAMATH_CALUDE_simplify_product_of_square_roots_l764_76412


namespace NUMINAMATH_CALUDE_simplify_fraction_l764_76466

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l764_76466


namespace NUMINAMATH_CALUDE_prob_ace_ten_jack_value_l764_76473

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Number of Aces in a standard deck -/
def NumAces : ℕ := 4

/-- Number of 10s in a standard deck -/
def NumTens : ℕ := 4

/-- Number of Jacks in a standard deck -/
def NumJacks : ℕ := 4

/-- Probability of drawing an Ace, then a 10, then a Jack from a standard deck without replacement -/
def prob_ace_ten_jack : ℚ :=
  (NumAces : ℚ) / StandardDeck *
  NumTens / (StandardDeck - 1) *
  NumJacks / (StandardDeck - 2)

theorem prob_ace_ten_jack_value : prob_ace_ten_jack = 16 / 33150 := by
  sorry

end NUMINAMATH_CALUDE_prob_ace_ten_jack_value_l764_76473


namespace NUMINAMATH_CALUDE_pentagon_area_form_pentagon_area_sum_l764_76469

/-- A pentagon constructed from 15 line segments of length 3 -/
structure Pentagon :=
  (F G H I J : ℝ × ℝ)
  (segments : List (ℝ × ℝ))
  (segment_length : ℝ)
  (segment_count : ℕ)
  (is_valid : segment_count = 15 ∧ segment_length = 3)

/-- The area of the pentagon -/
def pentagon_area (p : Pentagon) : ℝ := sorry

/-- The area can be expressed as √p + √q where p and q are positive integers -/
theorem pentagon_area_form (p : Pentagon) : 
  ∃ (a b : ℕ), pentagon_area p = Real.sqrt a + Real.sqrt b ∧ a > 0 ∧ b > 0 := sorry

/-- The sum of p and q is 48 -/
theorem pentagon_area_sum (p : Pentagon) :
  ∃ (a b : ℕ), pentagon_area p = Real.sqrt a + Real.sqrt b ∧ a > 0 ∧ b > 0 ∧ a + b = 48 := sorry

end NUMINAMATH_CALUDE_pentagon_area_form_pentagon_area_sum_l764_76469


namespace NUMINAMATH_CALUDE_selection_count_l764_76485

def class_size : ℕ := 38
def selection_size : ℕ := 5
def remaining_students : ℕ := class_size - 2  -- Excluding students A and B
def remaining_selection_size : ℕ := selection_size - 1  -- We always select student A

def binomial (n k : ℕ) : ℕ := 
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem selection_count : 
  binomial remaining_students remaining_selection_size = 58905 := by
  sorry

end NUMINAMATH_CALUDE_selection_count_l764_76485


namespace NUMINAMATH_CALUDE_wage_period_theorem_l764_76419

/-- Represents the number of days a sum of money can pay wages -/
structure WagePeriod where
  b : ℕ  -- Days for B's wages
  c : ℕ  -- Days for C's wages
  both : ℕ  -- Days for both B and C's wages

/-- Given conditions on wage periods, proves the number of days both can be paid -/
theorem wage_period_theorem (w : WagePeriod) (hb : w.b = 12) (hc : w.c = 24) :
  w.both = 8 := by
  sorry

#check wage_period_theorem

end NUMINAMATH_CALUDE_wage_period_theorem_l764_76419


namespace NUMINAMATH_CALUDE_students_with_glasses_l764_76435

theorem students_with_glasses (total : ℕ) (difference : ℕ) : total = 36 → difference = 24 → 
  ∃ (with_glasses : ℕ) (without_glasses : ℕ), 
    with_glasses + without_glasses = total ∧ 
    with_glasses + difference = without_glasses ∧ 
    with_glasses = 6 := by
  sorry

end NUMINAMATH_CALUDE_students_with_glasses_l764_76435


namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l764_76484

/-- Systematic sampling function -/
def systematicSample (totalStudents : ℕ) (sampleSize : ℕ) (firstGroupSelection : ℕ) (groupNumber : ℕ) : ℕ :=
  firstGroupSelection + (groupNumber - 1) * (totalStudents / sampleSize)

/-- Theorem: In a systematic sampling of 400 students with a sample size of 20,
    if the selected number from the first group is 12,
    then the selected number from the 14th group is 272. -/
theorem systematic_sampling_theorem :
  systematicSample 400 20 12 14 = 272 := by
  sorry

#eval systematicSample 400 20 12 14

end NUMINAMATH_CALUDE_systematic_sampling_theorem_l764_76484


namespace NUMINAMATH_CALUDE_home_run_differential_l764_76492

/-- Represents the number of home runs scored by a team in a specific inning -/
structure InningHomeRuns :=
  (inning : ℕ)
  (home_runs : ℕ)

/-- Represents a baseball team -/
inductive Team
| Cubs
| Cardinals

/-- Calculates the total number of home runs for a team -/
def total_home_runs (team_home_runs : List InningHomeRuns) : ℕ :=
  team_home_runs.foldl (fun acc ihr => acc + ihr.home_runs) 0

/-- The main theorem stating the home run differential -/
theorem home_run_differential
  (cubs_home_runs : List InningHomeRuns)
  (cardinals_home_runs : List InningHomeRuns)
  (h_cubs_3rd : InningHomeRuns.mk 3 2 ∈ cubs_home_runs)
  (h_cubs_5th : InningHomeRuns.mk 5 1 ∈ cubs_home_runs)
  (h_cubs_8th : InningHomeRuns.mk 8 2 ∈ cubs_home_runs)
  (h_cardinals_2nd : InningHomeRuns.mk 2 1 ∈ cardinals_home_runs)
  (h_cardinals_5th : InningHomeRuns.mk 5 1 ∈ cardinals_home_runs) :
  total_home_runs cubs_home_runs - total_home_runs cardinals_home_runs = 3 :=
sorry

end NUMINAMATH_CALUDE_home_run_differential_l764_76492


namespace NUMINAMATH_CALUDE_correct_factorization_l764_76475

/-- Given a quadratic expression x^2 - ax + b, if (x+6)(x-1) and (x-2)(x+1) are incorrect
    factorizations due to misreading a and b respectively, then the correct factorization
    is (x+2)(x-3). -/
theorem correct_factorization (a b : ℤ) : 
  (∃ a', (x^2 - a'*x + b = (x+6)*(x-1)) ∧ (a' ≠ a)) →
  (∃ b', (x^2 - a*x + b' = (x-2)*(x+1)) ∧ (b' ≠ b)) →
  (x^2 - a*x + b = (x+2)*(x-3)) :=
sorry

end NUMINAMATH_CALUDE_correct_factorization_l764_76475


namespace NUMINAMATH_CALUDE_baseball_games_played_l764_76403

theorem baseball_games_played (runs_1 runs_4 runs_5 : ℕ) (avg_runs : ℚ) : 
  runs_1 = 1 → runs_4 = 2 → runs_5 = 3 → avg_runs = 4 → 
  (runs_1 * 1 + runs_4 * 4 + runs_5 * 5 : ℚ) / (runs_1 + runs_4 + runs_5) = avg_runs → 
  runs_1 + runs_4 + runs_5 = 6 := by
sorry

end NUMINAMATH_CALUDE_baseball_games_played_l764_76403


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_tangency_l764_76411

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := x^2 + 9*y^2 = 9

-- Define the hyperbola equation
def hyperbola (x y n : ℝ) : Prop := x^2 - n*(y-1)^2 = 4

-- Define the tangency condition
def are_tangent (n : ℝ) : Prop := 
  ∃ x y, ellipse x y ∧ hyperbola x y n

-- Theorem statement
theorem ellipse_hyperbola_tangency (n : ℝ) :
  are_tangent n → n = 45/4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_tangency_l764_76411


namespace NUMINAMATH_CALUDE_math_team_combinations_l764_76441

theorem math_team_combinations (girls : ℕ) (boys : ℕ) : girls = 4 → boys = 5 → (girls.choose 3) * (boys.choose 1) = 20 := by
  sorry

end NUMINAMATH_CALUDE_math_team_combinations_l764_76441


namespace NUMINAMATH_CALUDE_inscribed_prism_volume_l764_76416

/-- Regular triangular prism inscribed in a sphere -/
structure InscribedPrism where
  /-- Radius of the sphere -/
  R : ℝ
  /-- Distance from vertex A to point D on the sphere -/
  AD : ℝ
  /-- Assertion that CD is a diameter of the sphere -/
  is_diameter : Bool

/-- Volume of the inscribed prism -/
def prism_volume (p : InscribedPrism) : ℝ :=
  sorry

/-- Theorem stating the volume of the specific inscribed prism -/
theorem inscribed_prism_volume :
  ∀ (p : InscribedPrism),
    p.R = 3 ∧ p.AD = 2 * Real.sqrt 6 ∧ p.is_diameter = true →
    prism_volume p = 6 * Real.sqrt 15 :=
  sorry

end NUMINAMATH_CALUDE_inscribed_prism_volume_l764_76416


namespace NUMINAMATH_CALUDE_exponential_inequality_l764_76495

theorem exponential_inequality (x₁ x₂ : ℝ) (h₁ : 0 < x₁) (h₂ : x₁ < x₂) (h₃ : x₂ < 1) :
  x₂ * Real.exp x₁ > x₁ * Real.exp x₂ := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l764_76495


namespace NUMINAMATH_CALUDE_towns_distance_l764_76424

/-- Given a map distance and a scale, calculate the actual distance between two towns. -/
def actual_distance (map_distance : ℝ) (scale : ℝ) : ℝ :=
  map_distance * scale

/-- Theorem stating that for a map distance of 20 inches and a scale of 1 inch = 10 miles,
    the actual distance between the towns is 200 miles. -/
theorem towns_distance :
  let map_distance : ℝ := 20
  let scale : ℝ := 10
  actual_distance map_distance scale = 200 := by
sorry

end NUMINAMATH_CALUDE_towns_distance_l764_76424


namespace NUMINAMATH_CALUDE_product_of_five_reals_l764_76487

theorem product_of_five_reals (a b c d e : ℝ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e)
  (h1 : a * b + b = a * c + a)
  (h2 : b * c + c = b * d + b)
  (h3 : c * d + d = c * e + c)
  (h4 : d * e + e = d * a + d) :
  a * b * c * d * e = 1 := by
sorry

end NUMINAMATH_CALUDE_product_of_five_reals_l764_76487


namespace NUMINAMATH_CALUDE_obtuseTrianglesIn120Gon_l764_76410

/-- The number of vertices in the regular polygon -/
def n : ℕ := 120

/-- A function that calculates the number of ways to choose three vertices 
    forming an obtuse triangle in a regular n-gon -/
def obtuseTrianglesCount (n : ℕ) : ℕ :=
  n * (n / 2 - 1) * (n / 2 - 2) / 2

/-- Theorem stating that the number of ways to choose three vertices 
    forming an obtuse triangle in a regular 120-gon is 205320 -/
theorem obtuseTrianglesIn120Gon : obtuseTrianglesCount n = 205320 := by
  sorry

end NUMINAMATH_CALUDE_obtuseTrianglesIn120Gon_l764_76410


namespace NUMINAMATH_CALUDE_sum_of_four_integers_l764_76446

theorem sum_of_four_integers (k l m n : ℕ+) 
  (h1 : k + l + m + n = k * m) 
  (h2 : k + l + m + n = l * n) : 
  k + l + m + n = 16 ∨ k + l + m + n = 18 ∨ k + l + m + n = 24 ∨ k + l + m + n = 30 :=
sorry

end NUMINAMATH_CALUDE_sum_of_four_integers_l764_76446


namespace NUMINAMATH_CALUDE_at_least_one_negative_l764_76421

theorem at_least_one_negative (a b : ℝ) (h1 : a ≠ b) (h2 : a ≠ 0) (h3 : b ≠ 0)
  (h4 : a^2 + 1/b = b^2 + 1/a) : a < 0 ∨ b < 0 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_negative_l764_76421


namespace NUMINAMATH_CALUDE_fraction_equality_l764_76400

theorem fraction_equality (a b c : ℝ) (h : a/2 = b/3 ∧ b/3 = c/4) :
  (a + b + c) / (2*a + b - c) = 3 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l764_76400


namespace NUMINAMATH_CALUDE_highway_time_greater_than_swamp_time_l764_76468

/-- Represents the hunter's journey with different terrains and speeds -/
structure HunterJourney where
  swamp_speed : ℝ
  forest_speed : ℝ
  highway_speed : ℝ
  total_time : ℝ
  total_distance : ℝ

/-- Theorem stating that the time spent on the highway is greater than the time spent in the swamp -/
theorem highway_time_greater_than_swamp_time (journey : HunterJourney) 
  (h1 : journey.swamp_speed = 2)
  (h2 : journey.forest_speed = 4)
  (h3 : journey.highway_speed = 6)
  (h4 : journey.total_time = 4)
  (h5 : journey.total_distance = 17) : 
  ∃ (swamp_time highway_time : ℝ), 
    swamp_time * journey.swamp_speed + 
    (journey.total_time - swamp_time - highway_time) * journey.forest_speed + 
    highway_time * journey.highway_speed = journey.total_distance ∧
    highway_time > swamp_time :=
by sorry

end NUMINAMATH_CALUDE_highway_time_greater_than_swamp_time_l764_76468


namespace NUMINAMATH_CALUDE_salary_changes_l764_76480

def initial_salary : ℝ := 1800

def may_raise : ℝ := 0.30
def june_cut : ℝ := 0.25
def july_increase : ℝ := 0.10

def final_salary : ℝ := initial_salary * (1 + july_increase)

theorem salary_changes :
  final_salary = 1980 := by sorry

end NUMINAMATH_CALUDE_salary_changes_l764_76480


namespace NUMINAMATH_CALUDE_rhombus_other_diagonal_l764_76454

/-- Represents a rhombus with given diagonals and area -/
structure Rhombus where
  d1 : ℝ
  d2 : ℝ
  area : ℝ
  h_area_formula : area = (d1 * d2) / 2

/-- Theorem: In a rhombus with one diagonal of 12 cm and an area of 90 cm², the other diagonal is 15 cm -/
theorem rhombus_other_diagonal
  (r : Rhombus)
  (h1 : r.d1 = 12)
  (h2 : r.area = 90) :
  r.d2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_other_diagonal_l764_76454


namespace NUMINAMATH_CALUDE_even_product_probability_l764_76482

def ten_sided_die := Finset.range 10

theorem even_product_probability :
  let outcomes := ten_sided_die.product ten_sided_die
  (outcomes.filter (fun (x, y) => (x + 1) * (y + 1) % 2 = 0)).card / outcomes.card = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_even_product_probability_l764_76482


namespace NUMINAMATH_CALUDE_system_solutions_l764_76432

/-- The system of equations -/
def system (x y z : ℝ) : Prop :=
  x^2 - y = z^2 ∧ y^2 - z = x^2 ∧ z^2 - x = y^2

/-- The set of solutions -/
def solutions : Set (ℝ × ℝ × ℝ) :=
  {(0, 0, 0), (1, 0, -1), (0, -1, 1), (-1, 1, 0)}

/-- Theorem stating that the solutions are correct and complete -/
theorem system_solutions :
  ∀ x y z : ℝ, system x y z ↔ (x, y, z) ∈ solutions := by
  sorry

end NUMINAMATH_CALUDE_system_solutions_l764_76432


namespace NUMINAMATH_CALUDE_min_value_implies_a_range_l764_76428

/-- Piecewise function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 2 then x^2 - 2*a*x - 2 else x + 36/x - 6*a

/-- Theorem stating that if f(2) is the minimum value of f(x), then a ∈ [2, 5] -/
theorem min_value_implies_a_range (a : ℝ) :
  (∀ x : ℝ, f a x ≥ f a 2) → 2 ≤ a ∧ a ≤ 5 := by
  sorry


end NUMINAMATH_CALUDE_min_value_implies_a_range_l764_76428


namespace NUMINAMATH_CALUDE_range_of_a_l764_76440

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 - x + a ≥ 0) → a ≥ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l764_76440


namespace NUMINAMATH_CALUDE_product_of_elements_is_zero_l764_76434

theorem product_of_elements_is_zero
  (n : ℕ)
  (M : Finset ℝ)
  (h_odd : Odd n)
  (h_gt_one : n > 1)
  (h_card : M.card = n)
  (h_sum_invariant : ∀ x ∈ M, M.sum id = (M.erase x).sum id + (M.sum id - x)) :
  M.prod id = 0 := by
sorry

end NUMINAMATH_CALUDE_product_of_elements_is_zero_l764_76434


namespace NUMINAMATH_CALUDE_largest_three_digit_base6_l764_76448

/-- Converts a base-6 number to base-10 --/
def base6ToBase10 (d1 d2 d3 : Nat) : Nat :=
  d1 * 6^2 + d2 * 6^1 + d3 * 6^0

/-- The largest digit in base-6 --/
def maxBase6Digit : Nat := 5

theorem largest_three_digit_base6 :
  base6ToBase10 maxBase6Digit maxBase6Digit maxBase6Digit = 215 := by
  sorry

end NUMINAMATH_CALUDE_largest_three_digit_base6_l764_76448


namespace NUMINAMATH_CALUDE_electronic_cat_run_time_l764_76459

/-- Proves that an electronic cat running on a circular track takes 36 seconds to run the last 120 meters -/
theorem electronic_cat_run_time (track_length : ℝ) (speed1 speed2 : ℝ) :
  track_length = 240 →
  speed1 = 5 →
  speed2 = 3 →
  let avg_speed := (speed1 + speed2) / 2
  let total_time := track_length / avg_speed
  let half_time := total_time / 2
  let first_half_distance := speed1 * half_time
  let second_half_distance := track_length - first_half_distance
  let time_at_speed1 := (first_half_distance - track_length / 2) / speed1
  let time_at_speed2 := (track_length / 2 - (first_half_distance - track_length / 2)) / speed2
  (time_at_speed1 + time_at_speed2 : ℝ) = 36 :=
by sorry

end NUMINAMATH_CALUDE_electronic_cat_run_time_l764_76459


namespace NUMINAMATH_CALUDE_price_after_nine_years_l764_76401

/-- The price of a product after a certain number of three-year periods, given an initial price and a decay rate. -/
def price_after_periods (initial_price : ℝ) (decay_rate : ℝ) (periods : ℕ) : ℝ :=
  initial_price * (1 - decay_rate) ^ periods

/-- Theorem stating that if a product's price decreases by 25% every three years and its current price is 640 yuan, then its price after 9 years will be 270 yuan. -/
theorem price_after_nine_years :
  let initial_price : ℝ := 640
  let decay_rate : ℝ := 0.25
  let periods : ℕ := 3
  price_after_periods initial_price decay_rate periods = 270 := by
  sorry


end NUMINAMATH_CALUDE_price_after_nine_years_l764_76401


namespace NUMINAMATH_CALUDE_book_cost_calculation_l764_76407

/-- Calculates the cost of each book given the total customers, return rate, and total sales after returns. -/
theorem book_cost_calculation (total_customers : ℕ) (return_rate : ℚ) (total_sales : ℚ) : 
  total_customers = 1000 → 
  return_rate = 37 / 100 → 
  total_sales = 9450 → 
  (total_sales / (total_customers * (1 - return_rate))) = 15 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_calculation_l764_76407


namespace NUMINAMATH_CALUDE_car_worth_calculation_l764_76414

/-- Brendan's earnings and expenses in June -/
structure BrendanFinances where
  total_earnings : ℕ  -- Total earnings in June
  remaining_money : ℕ  -- Remaining money at the end of June
  car_worth : ℕ  -- Worth of the used car

/-- The worth of the car is the difference between total earnings and remaining money -/
theorem car_worth_calculation (b : BrendanFinances) 
  (h1 : b.total_earnings = 5000)
  (h2 : b.remaining_money = 1000)
  (h3 : b.car_worth = b.total_earnings - b.remaining_money) :
  b.car_worth = 4000 := by
  sorry

#check car_worth_calculation

end NUMINAMATH_CALUDE_car_worth_calculation_l764_76414


namespace NUMINAMATH_CALUDE_max_daily_sales_l764_76429

def salesVolume (t : ℕ) : ℝ := -2 * t + 200

def price (t : ℕ) : ℝ :=
  if t ≤ 30 then 12 * t + 30 else 45

def dailySales (t : ℕ) : ℝ :=
  salesVolume t * price t

theorem max_daily_sales :
  ∃ (t : ℕ), 1 ≤ t ∧ t ≤ 50 ∧ ∀ (s : ℕ), 1 ≤ s ∧ s ≤ 50 → dailySales s ≤ dailySales t ∧ dailySales t = 54600 := by
  sorry

end NUMINAMATH_CALUDE_max_daily_sales_l764_76429


namespace NUMINAMATH_CALUDE_divisibility_of_T_members_l764_76491

/-- The set of all numbers which are the sum of the squares of four consecutive integers -/
def T : Set ℤ := {x | ∃ n : ℤ, x = (n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2}

theorem divisibility_of_T_members :
  (∃ x ∈ T, 5 ∣ x) ∧ (∀ x ∈ T, ¬(7 ∣ x)) := by sorry

end NUMINAMATH_CALUDE_divisibility_of_T_members_l764_76491


namespace NUMINAMATH_CALUDE_tennis_tournament_matches_l764_76415

theorem tennis_tournament_matches (n : Nat) (byes : Nat) :
  n = 100 →
  byes = 28 →
  ∃ m : Nat, m = n - 1 ∧ m % 11 = 0 :=
by sorry

end NUMINAMATH_CALUDE_tennis_tournament_matches_l764_76415


namespace NUMINAMATH_CALUDE_new_average_and_variance_l764_76483

/-- Given three numbers with average 5 and variance 2, prove that adding 1 results in four numbers with average 4 and variance 4.5 -/
theorem new_average_and_variance 
  (x y z : ℝ) 
  (h_avg : (x + y + z) / 3 = 5)
  (h_var : ((x - 5)^2 + (y - 5)^2 + (z - 5)^2) / 3 = 2) :
  let new_numbers := [x, y, z, 1]
  ((x + y + z + 1) / 4 = 4) ∧ 
  (((x - 4)^2 + (y - 4)^2 + (z - 4)^2 + (1 - 4)^2) / 4 = 4.5) := by
sorry


end NUMINAMATH_CALUDE_new_average_and_variance_l764_76483


namespace NUMINAMATH_CALUDE_overtime_hours_calculation_l764_76479

/-- Calculates the number of overtime hours worked by an employee given their gross pay and pay rates. -/
theorem overtime_hours_calculation (regular_rate overtime_rate gross_pay : ℚ) : 
  regular_rate = 11.25 →
  overtime_rate = 16 →
  gross_pay = 622 →
  ∃ (overtime_hours : ℕ), 
    overtime_hours = 11 ∧ 
    gross_pay = (40 * regular_rate) + (overtime_hours : ℚ) * overtime_rate :=
by sorry

end NUMINAMATH_CALUDE_overtime_hours_calculation_l764_76479


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l764_76413

theorem quadratic_roots_sum_of_squares (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    2 * x₁^2 + 4 * m * x₁ + m = 0 ∧
    2 * x₂^2 + 4 * m * x₂ + m = 0 ∧
    x₁^2 + x₂^2 = 3/16) →
  m = -1/8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l764_76413


namespace NUMINAMATH_CALUDE_marvin_bottle_caps_l764_76438

theorem marvin_bottle_caps (initial : ℕ) (remaining : ℕ) (taken : ℕ) : 
  initial = 16 → remaining = 10 → taken = initial - remaining → taken = 6 := by
  sorry

end NUMINAMATH_CALUDE_marvin_bottle_caps_l764_76438


namespace NUMINAMATH_CALUDE_max_elevation_is_288_l764_76462

-- Define the elevation function
def s (t : ℝ) : ℝ := 144 * t - 18 * t^2

-- Theorem stating that the maximum elevation is 288
theorem max_elevation_is_288 : 
  ∃ t_max : ℝ, ∀ t : ℝ, s t ≤ s t_max ∧ s t_max = 288 :=
sorry

end NUMINAMATH_CALUDE_max_elevation_is_288_l764_76462


namespace NUMINAMATH_CALUDE_hexagon_angle_measure_l764_76476

-- Define the hexagon and its angles
structure ConvexHexagon where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  E : ℝ
  F : ℝ

-- Define the theorem
theorem hexagon_angle_measure (h : ConvexHexagon) :
  h.A = h.B ∧ h.B = h.C ∧  -- A, B, C are congruent
  h.D = h.E ∧ h.E = h.F ∧  -- D, E, F are congruent
  h.A + 20 = h.D ∧         -- A is 20° less than D
  h.A + h.B + h.C + h.D + h.E + h.F = 720 -- Sum of angles in a hexagon
  →
  h.D = 130 := by sorry

end NUMINAMATH_CALUDE_hexagon_angle_measure_l764_76476


namespace NUMINAMATH_CALUDE_athlete_team_division_l764_76436

theorem athlete_team_division (n : ℕ) (k : ℕ) (total : ℕ) (specific : ℕ) :
  n = 10 →
  k = 5 →
  total = n →
  specific = 2 →
  (Nat.choose (n - specific) (k - 1)) = 70 := by
  sorry

end NUMINAMATH_CALUDE_athlete_team_division_l764_76436


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_ratio_l764_76460

theorem hyperbola_eccentricity_ratio (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (mx^2 - ny^2 = 1 ∧ (m + n) / n = 4) → m / n = 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_ratio_l764_76460


namespace NUMINAMATH_CALUDE_point_on_circle_x_value_l764_76471

-- Define the circle
def circle_center : ℝ × ℝ := (12, 0)
def circle_radius : ℝ := 15

-- Define the point on the circle
def point_on_circle (x : ℝ) : ℝ × ℝ := (x, 12)

-- Theorem statement
theorem point_on_circle_x_value (x : ℝ) :
  (point_on_circle x).1 - circle_center.1 ^ 2 + 
  (point_on_circle x).2 - circle_center.2 ^ 2 = circle_radius ^ 2 →
  x = 3 ∨ x = 21 := by
  sorry

end NUMINAMATH_CALUDE_point_on_circle_x_value_l764_76471


namespace NUMINAMATH_CALUDE_distance_from_center_to_chords_l764_76427

/-- A circle with two chords drawn through the ends of a diameter -/
structure CircleWithChords where
  /-- The radius of the circle -/
  radius : ℝ
  /-- The length of the first chord -/
  chord1_length : ℝ
  /-- The length of the second chord -/
  chord2_length : ℝ
  /-- The chords intersect on the circumference -/
  chords_intersect_on_circumference : True
  /-- The chords are drawn through the ends of a diameter -/
  chords_through_diameter_ends : True
  /-- The first chord has length 12 -/
  chord1_is_12 : chord1_length = 12
  /-- The second chord has length 16 -/
  chord2_is_16 : chord2_length = 16

/-- The theorem stating the distances from the center to the chords -/
theorem distance_from_center_to_chords (c : CircleWithChords) :
  ∃ (d1 d2 : ℝ), d1 = 8 ∧ d2 = 6 ∧
  d1 = c.chord2_length / 2 ∧
  d2 = c.chord1_length / 2 :=
sorry

end NUMINAMATH_CALUDE_distance_from_center_to_chords_l764_76427


namespace NUMINAMATH_CALUDE_max_value_constrained_l764_76490

theorem max_value_constrained (a b c : ℝ) (h : 9 * a^2 + 4 * b^2 + 25 * c^2 = 1) :
  (∃ x y z : ℝ, 9 * x^2 + 4 * y^2 + 25 * z^2 = 1 ∧ 8 * x + 3 * y + 5 * z > 8 * a + 3 * b + 5 * c) ∨
  8 * a + 3 * b + 5 * c = 7 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_constrained_l764_76490


namespace NUMINAMATH_CALUDE_multiples_property_l764_76431

def is_multiple_of (a b : ℤ) : Prop := ∃ k : ℤ, a = b * k

theorem multiples_property (c d : ℤ) 
  (hc : is_multiple_of c 5) 
  (hd : is_multiple_of d 10) : 
  (is_multiple_of d 5) ∧ 
  (is_multiple_of (c - d) 5) ∧ 
  (is_multiple_of (c + d) 5) := by
  sorry

end NUMINAMATH_CALUDE_multiples_property_l764_76431


namespace NUMINAMATH_CALUDE_angle_B_is_80_l764_76444

-- Define the quadrilateral and its properties
structure Quadrilateral where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  x : ℝ
  BEC : ℝ

-- Define the conditions
def quadrilateral_conditions (q : Quadrilateral) : Prop :=
  q.A = 60 ∧
  q.B = 2 * q.C ∧
  q.D = 2 * q.C - q.x ∧
  q.x > 0 ∧
  q.A + q.B + q.C + q.D = 360 ∧
  q.BEC = 20

-- Theorem statement
theorem angle_B_is_80 (q : Quadrilateral) 
  (h : quadrilateral_conditions q) : q.B = 80 :=
sorry

end NUMINAMATH_CALUDE_angle_B_is_80_l764_76444


namespace NUMINAMATH_CALUDE_area_square_on_hypotenuse_l764_76494

/-- Given a right triangle XYZ with right angle at Y, prove that the area of the square on XZ is 201
    when the sum of areas of a square on XY, a rectangle on YZ, and a square on XZ is 450,
    and YZ is 3 units longer than XY. -/
theorem area_square_on_hypotenuse (x y z : ℝ) (h1 : x^2 + y^2 = z^2)
    (h2 : y = x + 3) (h3 : x^2 + x * y + z^2 = 450) : z^2 = 201 := by
  sorry

end NUMINAMATH_CALUDE_area_square_on_hypotenuse_l764_76494


namespace NUMINAMATH_CALUDE_eighteenth_digit_is_five_l764_76472

/-- The sequence of digits in the decimal expansion of 10000/9899 -/
def decimalSequence : ℕ → ℕ
| 0 => 1  -- The digit before the decimal point
| 1 => 0
| 2 => 1
| n+3 => (decimalSequence n + decimalSequence (n+1)) % 10

/-- The 18th digit after the decimal point in the expansion of 10000/9899 is 5 -/
theorem eighteenth_digit_is_five : decimalSequence 18 = 5 := by
  sorry

end NUMINAMATH_CALUDE_eighteenth_digit_is_five_l764_76472


namespace NUMINAMATH_CALUDE_distinct_number_probability_l764_76498

-- Define the number of balls of each color and the number to be selected
def total_red_balls : ℕ := 5
def total_black_balls : ℕ := 5
def balls_to_select : ℕ := 4

-- Define the total number of balls
def total_balls : ℕ := total_red_balls + total_black_balls

-- Define the function to calculate binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Theorem statement
theorem distinct_number_probability :
  (binomial total_balls balls_to_select : ℚ) ≠ 0 →
  (binomial total_red_balls balls_to_select * 2^balls_to_select : ℚ) /
  (binomial total_balls balls_to_select : ℚ) = 8/21 := by sorry

end NUMINAMATH_CALUDE_distinct_number_probability_l764_76498


namespace NUMINAMATH_CALUDE_gcd_372_684_l764_76433

theorem gcd_372_684 : Nat.gcd 372 684 = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcd_372_684_l764_76433


namespace NUMINAMATH_CALUDE_cylinder_volume_ratio_l764_76453

/-- The ratio of volumes of cylinders formed by rolling a 6x9 rectangle -/
theorem cylinder_volume_ratio : 
  let length : ℝ := 6
  let width : ℝ := 9
  let volume1 := π * (length / (2 * π))^2 * width
  let volume2 := π * (width / (2 * π))^2 * length
  volume2 / volume1 = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_ratio_l764_76453


namespace NUMINAMATH_CALUDE_locus_of_point_on_line_intersecting_ellipse_l764_76458

/-- The locus of point P on a line segment AB, where the line intersects an ellipse --/
theorem locus_of_point_on_line_intersecting_ellipse 
  (x y : ℝ) 
  (h_ellipse : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2 + y₁^2/4 = 1 ∧ 
    x₂^2 + y₂^2/4 = 1 ∧ 
    y₁ - x₁ = y₂ - x₂)
  (h_slope : ∃ (c : ℝ), y = x + c)
  (h_ratio : ∃ (x_a y_a x_b y_b : ℝ), 
    (x - x_a)^2 + (y - y_a)^2 = 4 * ((x_b - x)^2 + (y_b - y)^2))
  (h_bound : |y - x| < Real.sqrt 5) :
  4*x + y = (2/3) * Real.sqrt (5 - (y-x)^2) :=
sorry

end NUMINAMATH_CALUDE_locus_of_point_on_line_intersecting_ellipse_l764_76458


namespace NUMINAMATH_CALUDE_hyperbola_foci_l764_76418

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 12 = 1

/-- The foci of the hyperbola -/
def foci : Set (ℝ × ℝ) :=
  {(4, 0), (-4, 0)}

/-- Theorem: The foci of the given hyperbola are (4,0) and (-4,0) -/
theorem hyperbola_foci :
  ∀ (p : ℝ × ℝ), p ∈ foci ↔
    (∃ (c : ℝ), ∀ (x y : ℝ),
      hyperbola_equation x y →
      (x - p.1)^2 + y^2 = (x + p.1)^2 + y^2 + 4*c) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_foci_l764_76418


namespace NUMINAMATH_CALUDE_fourth_person_height_l764_76425

/-- Given 4 people with heights in increasing order, prove that the 4th person is 85 inches tall. -/
theorem fourth_person_height (h₁ h₂ h₃ h₄ : ℝ) : 
  (h₁ < h₂) ∧ (h₂ < h₃) ∧ (h₃ < h₄) ∧  -- Heights in increasing order
  (h₂ - h₁ = 2) ∧                      -- Difference between 1st and 2nd
  (h₃ - h₂ = 2) ∧                      -- Difference between 2nd and 3rd
  (h₄ - h₃ = 6) ∧                      -- Difference between 3rd and 4th
  ((h₁ + h₂ + h₃ + h₄) / 4 = 79)       -- Average height
  → h₄ = 85 := by
sorry

end NUMINAMATH_CALUDE_fourth_person_height_l764_76425


namespace NUMINAMATH_CALUDE_min_buses_required_l764_76457

theorem min_buses_required (total_students : ℕ) (bus_capacity : ℕ) (h1 : total_students = 625) (h2 : bus_capacity = 47) :
  ∃ (n : ℕ), n * bus_capacity ≥ total_students ∧ ∀ (m : ℕ), m * bus_capacity ≥ total_students → n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_min_buses_required_l764_76457


namespace NUMINAMATH_CALUDE_total_cost_is_24_l764_76452

/-- The cost of a burger meal and soda order for two people -/
def total_cost (burger_price : ℚ) : ℚ :=
  let soda_price := burger_price / 3
  let paulo_total := burger_price + soda_price
  let jeremy_total := 2 * paulo_total
  paulo_total + jeremy_total

/-- Theorem: The total cost of Paulo and Jeremy's orders is $24 when a burger meal costs $6 -/
theorem total_cost_is_24 : total_cost 6 = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_24_l764_76452


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l764_76439

theorem min_value_sum_reciprocals (a b c d e f : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_d : 0 < d) (pos_e : 0 < e) (pos_f : 0 < f)
  (sum_eq_9 : a + b + c + d + e + f = 9) :
  (1/a + 9/b + 25/c + 49/d + 81/e + 121/f) ≥ 286^2/9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l764_76439


namespace NUMINAMATH_CALUDE_basketball_tryouts_l764_76426

/-- The number of boys who tried out for the basketball team -/
def boys_tried_out : ℕ := sorry

/-- The number of girls who tried out for the basketball team -/
def girls_tried_out : ℕ := 9

/-- The number of students who got called back -/
def called_back : ℕ := 2

/-- The number of students who didn't make the cut -/
def didnt_make_cut : ℕ := 21

theorem basketball_tryouts :
  boys_tried_out = 14 ∧
  girls_tried_out + boys_tried_out = called_back + didnt_make_cut :=
by sorry

end NUMINAMATH_CALUDE_basketball_tryouts_l764_76426


namespace NUMINAMATH_CALUDE_positive_sum_inequality_sqrt_difference_inequality_l764_76489

-- Problem 1
theorem positive_sum_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) :
  min ((1 + x) / y) ((1 + y) / x) < 2 := by sorry

-- Problem 2
theorem sqrt_difference_inequality (n : ℕ+) :
  Real.sqrt (n + 1) - Real.sqrt n > Real.sqrt (n + 2) - Real.sqrt (n + 1) := by sorry

end NUMINAMATH_CALUDE_positive_sum_inequality_sqrt_difference_inequality_l764_76489


namespace NUMINAMATH_CALUDE_quadratic_roots_l764_76420

theorem quadratic_roots (m : ℝ) : 
  (∃! x : ℝ, (m + 2) * x^2 - 2 * (m + 1) * x + m = 0) →
  (∃ x : ℝ, (m + 1) * x^2 - 2 * m * x + (m - 2) = 0 ∧
             ∀ y : ℝ, (m + 1) * y^2 - 2 * m * y + (m - 2) = 0 → y = x) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_l764_76420


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l764_76470

def A : Matrix (Fin 3) (Fin 3) ℤ := !![0, 1, 2; 1, 0, 1; 2, 1, 0]

theorem matrix_equation_solution :
  ∃ (p q r : ℤ), A^3 + p • A^2 + q • A + r • (1 : Matrix (Fin 3) (Fin 3) ℤ) = 0 ∧ p = 0 ∧ q = -6 ∧ r = -4 := by
  sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l764_76470


namespace NUMINAMATH_CALUDE_inequality_proof_l764_76423

theorem inequality_proof (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  (a^2 * b + b^2 * c + c^2 * a) * (a * b^2 + b * c^2 + c * a^2) ≥ 9 * (a * b * c)^2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l764_76423


namespace NUMINAMATH_CALUDE_jack_bought_36_books_l764_76430

/-- The number of books Jack bought each month -/
def books_per_month : ℕ := sorry

/-- The price of each book in dollars -/
def price_per_book : ℕ := 20

/-- The total sale price at the end of the year in dollars -/
def total_sale_price : ℕ := 500

/-- The total loss in dollars -/
def total_loss : ℕ := 220

/-- Theorem stating that Jack bought 36 books each month -/
theorem jack_bought_36_books : books_per_month = 36 := by
  sorry

end NUMINAMATH_CALUDE_jack_bought_36_books_l764_76430


namespace NUMINAMATH_CALUDE_min_value_quadratic_l764_76461

theorem min_value_quadratic (x y : ℝ) (h : x + y = 4) :
  ∃ (m : ℝ), m = 12 ∧ ∀ (a b : ℝ), a + b = 4 → 3 * a^2 + b^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l764_76461


namespace NUMINAMATH_CALUDE_min_burgers_recovery_l764_76488

/-- The minimum whole number of burgers Sarah must sell to recover her initial investment -/
def min_burgers : ℕ := 637

/-- Sarah's initial investment in dollars -/
def initial_investment : ℕ := 7000

/-- Sarah's earnings per burger in dollars -/
def earnings_per_burger : ℕ := 15

/-- Sarah's ingredient cost per burger in dollars -/
def ingredient_cost_per_burger : ℕ := 4

/-- Theorem stating that min_burgers is the minimum whole number of burgers
    Sarah must sell to recover her initial investment -/
theorem min_burgers_recovery :
  (min_burgers * (earnings_per_burger - ingredient_cost_per_burger) ≥ initial_investment) ∧
  ∀ n : ℕ, n < min_burgers → n * (earnings_per_burger - ingredient_cost_per_burger) < initial_investment :=
by sorry

end NUMINAMATH_CALUDE_min_burgers_recovery_l764_76488


namespace NUMINAMATH_CALUDE_chipmunk_acorns_l764_76478

/-- Represents the number of acorns hidden in each hole by an animal -/
structure AcornsPerHole where
  chipmunk : ℕ
  squirrel : ℕ

/-- Represents the number of holes dug by each animal -/
structure HolesDug where
  chipmunk : ℕ
  squirrel : ℕ

/-- The main theorem about the number of acorns hidden by the chipmunk -/
theorem chipmunk_acorns (aph : AcornsPerHole) (h : HolesDug) : 
  aph.chipmunk = 3 → 
  aph.squirrel = 4 → 
  h.chipmunk = h.squirrel + 4 → 
  aph.chipmunk * h.chipmunk = aph.squirrel * h.squirrel → 
  aph.chipmunk * h.chipmunk = 48 := by
  sorry

#check chipmunk_acorns

end NUMINAMATH_CALUDE_chipmunk_acorns_l764_76478


namespace NUMINAMATH_CALUDE_midpoint_distance_to_y_axis_l764_76447

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line passing through the focus
def line_through_focus (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ∃ (t : ℝ), x₁ = t ∧ x₂ = t ∧ y₁^2 = 4*x₁ ∧ y₂^2 = 4*x₂

-- Theorem statement
theorem midpoint_distance_to_y_axis 
  (x₁ y₁ x₂ y₂ : ℝ) 
  (h_parabola₁ : parabola x₁ y₁) 
  (h_parabola₂ : parabola x₂ y₂)
  (h_line : line_through_focus x₁ y₁ x₂ y₂)
  (h_sum : x₁ + x₂ = 3) :
  (x₁ + x₂) / 2 = 3 / 2 :=
sorry

end NUMINAMATH_CALUDE_midpoint_distance_to_y_axis_l764_76447


namespace NUMINAMATH_CALUDE_f_2_equals_17_l764_76486

/-- A function f with an extremum of 9 at x = 1 -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2 - 1

/-- The function f has an extremum of 9 at x = 1 -/
def has_extremum_at_1 (a b : ℝ) : Prop :=
  f a b 1 = 9 ∧ ∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a b x ≤ f a b 1

theorem f_2_equals_17 (a b : ℝ) (h : has_extremum_at_1 a b) : f a b 2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_f_2_equals_17_l764_76486


namespace NUMINAMATH_CALUDE_shopping_theorem_l764_76445

def shopping_problem (total_amount : ℝ) : Prop :=
  let clothing_percent : ℝ := 0.40
  let food_percent : ℝ := 0.20
  let electronics_percent : ℝ := 0.10
  let cosmetics_percent : ℝ := 0.20
  let household_percent : ℝ := 0.10

  let clothing_discount : ℝ := 0.10
  let food_discount : ℝ := 0.05
  let electronics_discount : ℝ := 0.15
  let cosmetics_discount : ℝ := 0
  let household_discount : ℝ := 0

  let clothing_tax : ℝ := 0.06
  let food_tax : ℝ := 0
  let electronics_tax : ℝ := 0.10
  let cosmetics_tax : ℝ := 0.08
  let household_tax : ℝ := 0.04

  let clothing_amount := total_amount * clothing_percent
  let food_amount := total_amount * food_percent
  let electronics_amount := total_amount * electronics_percent
  let cosmetics_amount := total_amount * cosmetics_percent
  let household_amount := total_amount * household_percent

  let clothing_tax_paid := clothing_amount * (1 - clothing_discount) * clothing_tax
  let food_tax_paid := food_amount * (1 - food_discount) * food_tax
  let electronics_tax_paid := electronics_amount * (1 - electronics_discount) * electronics_tax
  let cosmetics_tax_paid := cosmetics_amount * (1 - cosmetics_discount) * cosmetics_tax
  let household_tax_paid := household_amount * (1 - household_discount) * household_tax

  let total_tax_paid := clothing_tax_paid + food_tax_paid + electronics_tax_paid + cosmetics_tax_paid + household_tax_paid
  let total_tax_percentage := (total_tax_paid / total_amount) * 100

  total_tax_percentage = 5.01

theorem shopping_theorem : ∀ (total_amount : ℝ), total_amount > 0 → shopping_problem total_amount := by
  sorry

end NUMINAMATH_CALUDE_shopping_theorem_l764_76445


namespace NUMINAMATH_CALUDE_least_five_digit_divisible_by_12_15_18_l764_76477

theorem least_five_digit_divisible_by_12_15_18 : ∃ n : ℕ,
  (n ≥ 10000 ∧ n < 100000) ∧  -- 5-digit number
  (n % 12 = 0 ∧ n % 15 = 0 ∧ n % 18 = 0) ∧  -- divisible by 12, 15, and 18
  (∀ m : ℕ, m ≥ 10000 ∧ m < n ∧ m % 12 = 0 ∧ m % 15 = 0 ∧ m % 18 = 0 → false) ∧  -- least such number
  n = 10080 :=  -- the answer
by sorry

end NUMINAMATH_CALUDE_least_five_digit_divisible_by_12_15_18_l764_76477


namespace NUMINAMATH_CALUDE_quadratic_inequality_l764_76404

/-- A quadratic function of the form y = a(x-3)² + c where a < 0 -/
def quadratic_function (a c : ℝ) (h : a < 0) : ℝ → ℝ := 
  fun x => a * (x - 3)^2 + c

theorem quadratic_inequality (a c : ℝ) (h : a < 0) :
  let f := quadratic_function a c h
  let y₁ := f (Real.sqrt 5)
  let y₂ := f 0
  let y₃ := f 4
  y₂ < y₃ ∧ y₃ < y₁ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l764_76404


namespace NUMINAMATH_CALUDE_sixteen_students_not_liking_sports_l764_76465

/-- The number of students who do not like basketball, cricket, or football -/
def students_not_liking_sports (total : ℕ) (basketball cricket football : ℕ) 
  (basketball_cricket cricket_football basketball_football : ℕ) (all_three : ℕ) : ℕ :=
  total - (basketball + cricket + football - basketball_cricket - cricket_football - basketball_football + all_three)

/-- Theorem stating that 16 students do not like any of the three sports -/
theorem sixteen_students_not_liking_sports : 
  students_not_liking_sports 50 20 18 12 8 6 5 3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_sixteen_students_not_liking_sports_l764_76465


namespace NUMINAMATH_CALUDE_jennifer_gave_away_six_fruits_l764_76474

/-- Represents the number of fruits Jennifer gave to her sister -/
def fruits_given_away (initial_pears initial_oranges initial_apples fruits_left : ℕ) : ℕ :=
  initial_pears + initial_oranges + initial_apples - fruits_left

/-- Proves that Jennifer gave away 6 fruits -/
theorem jennifer_gave_away_six_fruits :
  ∀ (initial_pears initial_oranges initial_apples fruits_left : ℕ),
    initial_pears = 10 →
    initial_oranges = 20 →
    initial_apples = 2 * initial_pears →
    fruits_left = 44 →
    fruits_given_away initial_pears initial_oranges initial_apples fruits_left = 6 :=
by
  sorry

#check jennifer_gave_away_six_fruits

end NUMINAMATH_CALUDE_jennifer_gave_away_six_fruits_l764_76474


namespace NUMINAMATH_CALUDE_jake_not_drop_coffee_l764_76409

/-- The probability of Jake tripping over his dog on any given morning. -/
def prob_trip : ℝ := 0.40

/-- The probability of Jake dropping his coffee when he trips over his dog. -/
def prob_drop_when_trip : ℝ := 0.25

/-- The probability of Jake missing a step on the stairs on any given morning. -/
def prob_miss_step : ℝ := 0.30

/-- The probability of Jake spilling his coffee when he misses a step. -/
def prob_spill_when_miss : ℝ := 0.20

/-- Theorem: The probability of Jake not dropping his coffee on any given morning is 0.846. -/
theorem jake_not_drop_coffee :
  (1 - prob_trip * prob_drop_when_trip) * (1 - prob_miss_step * prob_spill_when_miss) = 0.846 := by
  sorry

end NUMINAMATH_CALUDE_jake_not_drop_coffee_l764_76409


namespace NUMINAMATH_CALUDE_chairs_per_round_table_l764_76455

/-- Proves that each round table has 6 chairs in the office canteen -/
theorem chairs_per_round_table : 
  ∀ (x : ℕ),
  (2 * x + 2 * 7 = 26) →
  x = 6 := by
sorry

end NUMINAMATH_CALUDE_chairs_per_round_table_l764_76455


namespace NUMINAMATH_CALUDE_magic_square_sum_l764_76437

/-- Represents a 3x3 magic square -/
structure MagicSquare :=
  (a b c d e : ℕ)
  (sum : ℕ)
  (row1_sum : a + 22 + b = sum)
  (row2_sum : 20 + c + e = sum)
  (row3_sum : 28 + d + 19 = sum)
  (col1_sum : a + 20 + 28 = sum)
  (col2_sum : 22 + c + d = sum)
  (col3_sum : b + e + 19 = sum)
  (diag1_sum : a + c + 19 = sum)
  (diag2_sum : 28 + c + b = sum)

/-- Theorem: In the given magic square, d + e = 70 -/
theorem magic_square_sum (ms : MagicSquare) : ms.d + ms.e = 70 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_sum_l764_76437


namespace NUMINAMATH_CALUDE_quadratic_solution_set_l764_76406

/-- A quadratic function f(x) = ax^2 - bx + 1 -/
def f (a b x : ℝ) : ℝ := a * x^2 - b * x + 1

theorem quadratic_solution_set (a b : ℝ) :
  (∀ x, f a b x < 0 ↔ 1/4 < x ∧ x < 1/3) →
  a = 12 ∧ b = 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_set_l764_76406


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l764_76499

theorem smallest_four_digit_divisible_by_53 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 → n ≥ 1007 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l764_76499


namespace NUMINAMATH_CALUDE_committee_arrangement_count_l764_76402

def num_women : ℕ := 7
def num_men : ℕ := 3
def num_rocking_chairs : ℕ := 7
def num_stools : ℕ := 3
def num_unique_chair : ℕ := 1
def total_seats : ℕ := num_women + num_men + num_unique_chair

def arrangement_count : ℕ := total_seats * (Nat.choose (total_seats - 1) num_stools)

theorem committee_arrangement_count :
  arrangement_count = 1320 :=
sorry

end NUMINAMATH_CALUDE_committee_arrangement_count_l764_76402


namespace NUMINAMATH_CALUDE_cos_angle_sum_diff_vectors_l764_76442

def a : ℝ × ℝ := (3, 1)
def b : ℝ × ℝ := (2, 2)

theorem cos_angle_sum_diff_vectors :
  let sum := (a.1 + b.1, a.2 + b.2)
  let diff := (a.1 - b.1, a.2 - b.2)
  (sum.1 * diff.1 + sum.2 * diff.2) / 
  (Real.sqrt (sum.1^2 + sum.2^2) * Real.sqrt (diff.1^2 + diff.2^2)) = 
  Real.sqrt 17 / 17 := by
  sorry

end NUMINAMATH_CALUDE_cos_angle_sum_diff_vectors_l764_76442


namespace NUMINAMATH_CALUDE_taco_truck_problem_l764_76449

/-- The price of a hard shell taco, given the conditions of the taco truck problem -/
def hard_shell_taco_price : ℝ := 5

theorem taco_truck_problem :
  let soft_taco_price : ℝ := 2
  let family_hard_tacos : ℕ := 4
  let family_soft_tacos : ℕ := 3
  let other_customers : ℕ := 10
  let other_customer_soft_tacos : ℕ := 2
  let total_earnings : ℝ := 66

  family_hard_tacos * hard_shell_taco_price +
  family_soft_tacos * soft_taco_price +
  other_customers * other_customer_soft_tacos * soft_taco_price = total_earnings :=
by
  sorry

#eval hard_shell_taco_price

end NUMINAMATH_CALUDE_taco_truck_problem_l764_76449


namespace NUMINAMATH_CALUDE_necessary_and_sufficient_condition_l764_76497

theorem necessary_and_sufficient_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧
  (∃ x₀ : ℝ, x₀^2 + 2*a*x₀ + 2 - a = 0) ↔
  a ≤ -2 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_necessary_and_sufficient_condition_l764_76497


namespace NUMINAMATH_CALUDE_ice_cream_bill_calculation_l764_76467

/-- Calculate the final bill amount for an ice cream outing --/
theorem ice_cream_bill_calculation
  (alicia_total brant_total josh_total yvette_total : ℝ)
  (discount_rate tax_rate tip_rate : ℝ)
  (h_alicia : alicia_total = 16.50)
  (h_brant : brant_total = 20.50)
  (h_josh : josh_total = 16.00)
  (h_yvette : yvette_total = 19.50)
  (h_discount : discount_rate = 0.10)
  (h_tax : tax_rate = 0.08)
  (h_tip : tip_rate = 0.20) :
  let subtotal := alicia_total + brant_total + josh_total + yvette_total
  let discounted_subtotal := subtotal * (1 - discount_rate)
  let tax_amount := discounted_subtotal * tax_rate
  let tip_amount := subtotal * tip_rate
  discounted_subtotal + tax_amount + tip_amount = 84.97 := by
  sorry


end NUMINAMATH_CALUDE_ice_cream_bill_calculation_l764_76467


namespace NUMINAMATH_CALUDE_fraction_equality_l764_76463

theorem fraction_equality (m n r t : ℚ) 
  (h1 : m / n = 5 / 2) 
  (h2 : r / t = 7 / 8) : 
  (3 * m * r - n * t) / (4 * n * t - 7 * m * r) = -89 / 181 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l764_76463


namespace NUMINAMATH_CALUDE_lives_per_player_l764_76443

theorem lives_per_player (initial_players : ℕ) (quitting_players : ℕ) (total_lives : ℕ) 
  (h1 : initial_players = 8)
  (h2 : quitting_players = 5)
  (h3 : total_lives = 15)
  (h4 : initial_players > quitting_players) :
  total_lives / (initial_players - quitting_players) = 5 := by
  sorry

end NUMINAMATH_CALUDE_lives_per_player_l764_76443


namespace NUMINAMATH_CALUDE_not_proposition_example_l764_76464

-- Define what a proposition is in this context
def is_proposition (s : String) : Prop :=
  ∀ (interpretation : Type), (∃ (truth_value : Bool), true)

-- State the theorem
theorem not_proposition_example : ¬ (is_proposition "x^2 + 2x - 3 < 0") :=
sorry

end NUMINAMATH_CALUDE_not_proposition_example_l764_76464


namespace NUMINAMATH_CALUDE_chess_tournament_games_l764_76496

theorem chess_tournament_games (n : ℕ) (h : n = 16) : 
  (n * (n - 1)) / 2 = 120 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l764_76496


namespace NUMINAMATH_CALUDE_equation_one_root_range_l764_76451

-- Define the equation
def equation (k : ℝ) (x : ℝ) : Prop :=
  Real.log (k * x) = 2 * Real.log (x + 1)

-- Define the condition for having only one real root
def has_one_real_root (k : ℝ) : Prop :=
  ∃! x : ℝ, equation k x

-- Define the range of k
def k_range : Set ℝ :=
  Set.Iio 0 ∪ {4}

-- Theorem statement
theorem equation_one_root_range :
  {k : ℝ | has_one_real_root k} = k_range :=
sorry

end NUMINAMATH_CALUDE_equation_one_root_range_l764_76451


namespace NUMINAMATH_CALUDE_cyclists_meeting_time_l764_76405

theorem cyclists_meeting_time 
  (course_length : ℝ) 
  (speed1 speed2 : ℝ) 
  (h1 : course_length = 45)
  (h2 : speed1 = 14)
  (h3 : speed2 = 16) :
  ∃ t : ℝ, t * speed1 + t * speed2 = course_length ∧ t = 1.5 :=
by sorry

end NUMINAMATH_CALUDE_cyclists_meeting_time_l764_76405


namespace NUMINAMATH_CALUDE_assignment_methods_eq_twelve_l764_76408

/-- The number of ways to assign doctors and nurses to schools. -/
def assignment_methods (num_doctors num_nurses num_schools : ℕ) : ℕ :=
  (num_doctors.choose 1) * (num_nurses.choose 2)

/-- Theorem stating the number of assignment methods for the given problem. -/
theorem assignment_methods_eq_twelve :
  assignment_methods 2 4 2 = 12 :=
by sorry

end NUMINAMATH_CALUDE_assignment_methods_eq_twelve_l764_76408


namespace NUMINAMATH_CALUDE_probability_arts_and_sciences_is_two_thirds_l764_76417

/-- Represents a class subject -/
inductive Subject
  | Mathematics
  | Chinese
  | Politics
  | Geography
  | English
  | History
  | PhysicalEducation

/-- Represents the time of day for a class -/
inductive TimeOfDay
  | Morning
  | Afternoon

/-- Defines the class schedule -/
def schedule : TimeOfDay → List Subject
  | TimeOfDay.Morning => [Subject.Mathematics, Subject.Chinese, Subject.Politics, Subject.Geography]
  | TimeOfDay.Afternoon => [Subject.English, Subject.History, Subject.PhysicalEducation]

/-- Determines if a subject is related to arts and sciences -/
def isArtsAndSciences : Subject → Bool
  | Subject.Politics => true
  | Subject.History => true
  | Subject.Geography => true
  | _ => false

/-- The probability of selecting at least one arts and sciences class -/
def probabilityArtsAndSciences : ℚ := 2/3

theorem probability_arts_and_sciences_is_two_thirds :
  probabilityArtsAndSciences = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_probability_arts_and_sciences_is_two_thirds_l764_76417
