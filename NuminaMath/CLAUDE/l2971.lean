import Mathlib

namespace NUMINAMATH_CALUDE_he_more_apples_l2971_297194

/-- The number of apples Adam and Jackie have together -/
def total_adam_jackie : ℕ := 12

/-- The number of apples Adam has more than Jackie -/
def adam_more_than_jackie : ℕ := 8

/-- The number of apples He has -/
def he_apples : ℕ := 21

/-- The number of apples Jackie has -/
def jackie_apples : ℕ := 2

/-- The number of apples Adam has -/
def adam_apples : ℕ := jackie_apples + adam_more_than_jackie

theorem he_more_apples : he_apples - total_adam_jackie = 9 := by
  sorry

end NUMINAMATH_CALUDE_he_more_apples_l2971_297194


namespace NUMINAMATH_CALUDE_binomial_minus_five_l2971_297126

theorem binomial_minus_five : Nat.choose 10 3 - 5 = 115 := by sorry

end NUMINAMATH_CALUDE_binomial_minus_five_l2971_297126


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_l2971_297153

/-- A quadratic radical is in its simplest form if it has no fractions inside 
    the radical and no coefficients outside the radical. -/
def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∃ (n : ℕ), x = Real.sqrt n ∧ n ≠ 0 ∧ n ≠ 1 ∧ ∀ (m : ℕ), m * m ≤ n → m = 1

theorem simplest_quadratic_radical :
  is_simplest_quadratic_radical (Real.sqrt 5) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt 0.2) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt (1/2)) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt 12) :=
by sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_l2971_297153


namespace NUMINAMATH_CALUDE_jackson_charity_collection_l2971_297182

/-- Proves the number of houses Jackson needs to visit per day to meet his goal -/
theorem jackson_charity_collection (total_goal : ℕ) (days_per_week : ℕ) (monday_earnings : ℕ) (tuesday_earnings : ℕ) (houses_per_collection : ℕ) (earnings_per_collection : ℕ) : 
  total_goal = 1000 →
  days_per_week = 5 →
  monday_earnings = 300 →
  tuesday_earnings = 40 →
  houses_per_collection = 4 →
  earnings_per_collection = 10 →
  ∃ (houses_per_day : ℕ), 
    houses_per_day = 88 ∧ 
    (total_goal - monday_earnings - tuesday_earnings) = 
      (days_per_week - 2) * houses_per_day * (earnings_per_collection / houses_per_collection) :=
by
  sorry

end NUMINAMATH_CALUDE_jackson_charity_collection_l2971_297182


namespace NUMINAMATH_CALUDE_intersection_of_two_lines_l2971_297123

/-- The intersection point of two lines in a 2D plane -/
structure IntersectionPoint where
  x : ℝ
  y : ℝ

/-- Checks if a point satisfies the equation of a line -/
def satisfiesLine (p : IntersectionPoint) (a b c : ℝ) : Prop :=
  a * p.x + b * p.y = c

/-- The unique intersection point of two lines -/
def uniqueIntersection (p : IntersectionPoint) (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  satisfiesLine p a1 b1 c1 ∧
  satisfiesLine p a2 b2 c2 ∧
  ∀ q : IntersectionPoint, satisfiesLine q a1 b1 c1 ∧ satisfiesLine q a2 b2 c2 → q = p

theorem intersection_of_two_lines :
  uniqueIntersection ⟨3, -1⟩ 2 (-1) 7 3 2 7 :=
sorry

end NUMINAMATH_CALUDE_intersection_of_two_lines_l2971_297123


namespace NUMINAMATH_CALUDE_movie_ticket_revenue_l2971_297156

/-- Calculates the total revenue from movie ticket sales --/
theorem movie_ticket_revenue
  (adult_price : ℕ)
  (child_price : ℕ)
  (total_tickets : ℕ)
  (child_tickets : ℕ)
  (h1 : adult_price = 7)
  (h2 : child_price = 4)
  (h3 : total_tickets = 900)
  (h4 : child_tickets = 400) :
  adult_price * (total_tickets - child_tickets) + child_price * child_tickets = 5100 :=
by sorry

end NUMINAMATH_CALUDE_movie_ticket_revenue_l2971_297156


namespace NUMINAMATH_CALUDE_train_length_l2971_297133

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 72 → time = 12 → speed * time * (1000 / 3600) = 240 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l2971_297133


namespace NUMINAMATH_CALUDE_parabola_param_valid_l2971_297154

/-- A parameterization of the curve y = x^2 -/
def parabola_param (t : ℝ) : ℝ × ℝ := (t, t^2)

/-- The curve y = x^2 -/
def parabola (x : ℝ) : ℝ := x^2

theorem parabola_param_valid :
  ∀ (x : ℝ), ∃ (t : ℝ), parabola_param t = (x, parabola x) :=
sorry

end NUMINAMATH_CALUDE_parabola_param_valid_l2971_297154


namespace NUMINAMATH_CALUDE_constant_t_equation_l2971_297119

theorem constant_t_equation : ∃! t : ℝ, 
  ∀ x : ℝ, (2*x^2 - 3*x + 4)*(5*x^2 + t*x + 9) = 10*x^4 - t^2*x^3 + 23*x^2 - 27*x + 36 ∧ t = -5 := by
  sorry

end NUMINAMATH_CALUDE_constant_t_equation_l2971_297119


namespace NUMINAMATH_CALUDE_subset_pairs_count_for_six_elements_l2971_297159

-- Define a function that counts the number of valid subset pairs
def countValidSubsetPairs (n : ℕ) : ℕ :=
  if n = 0 then 1 else 3 * countValidSubsetPairs (n - 1) - 1

-- Theorem statement
theorem subset_pairs_count_for_six_elements :
  countValidSubsetPairs 6 = 365 := by
  sorry

end NUMINAMATH_CALUDE_subset_pairs_count_for_six_elements_l2971_297159


namespace NUMINAMATH_CALUDE_symmetric_point_correct_l2971_297168

/-- The symmetric point of (a, b) with respect to the y-axis is (-a, b) -/
def symmetric_point_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-(p.1), p.2)

/-- The given point A -/
def point_A : ℝ × ℝ := (2, -3)

/-- The expected symmetric point -/
def expected_symmetric_point : ℝ × ℝ := (-2, -3)

theorem symmetric_point_correct :
  symmetric_point_y_axis point_A = expected_symmetric_point := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_correct_l2971_297168


namespace NUMINAMATH_CALUDE_grass_seed_price_five_pound_bag_price_l2971_297161

/-- Represents the price of a bag of grass seed -/
structure BagPrice where
  weight : ℕ
  price : ℚ

/-- Represents the customer's purchase -/
structure Purchase where
  bags5lb : ℕ
  bags10lb : ℕ
  bags25lb : ℕ

def total_weight (p : Purchase) : ℕ :=
  5 * p.bags5lb + 10 * p.bags10lb + 25 * p.bags25lb

def total_cost (p : Purchase) (price5lb : ℚ) : ℚ :=
  price5lb * p.bags5lb + 20.42 * p.bags10lb + 32.25 * p.bags25lb

def is_valid_purchase (p : Purchase) : Prop :=
  65 ≤ total_weight p ∧ total_weight p ≤ 80

theorem grass_seed_price (price5lb : ℚ) : Prop :=
  ∃ (p : Purchase),
    is_valid_purchase p ∧
    total_cost p price5lb = 98.77 ∧
    ∀ (q : Purchase), is_valid_purchase q → total_cost q price5lb ≥ 98.77 →
    price5lb = 2.02

/-- The main theorem stating that the price of the 5-pound bag is $2.02 -/
theorem five_pound_bag_price : ∃ (price5lb : ℚ), grass_seed_price price5lb :=
  sorry

end NUMINAMATH_CALUDE_grass_seed_price_five_pound_bag_price_l2971_297161


namespace NUMINAMATH_CALUDE_triangle_theorem_l2971_297104

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.a * Real.sin (2 * t.B) = Real.sqrt 3 * t.b * Real.sin t.A)
  (h2 : Real.cos t.A = 1 / 3) :
  t.B = π / 6 ∧ Real.sin t.C = (2 * Real.sqrt 6 + 1) / 6 := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l2971_297104


namespace NUMINAMATH_CALUDE_corn_cobs_picked_l2971_297166

theorem corn_cobs_picked (bushel_weight : ℝ) (ear_weight : ℝ) (bushels_picked : ℝ) : 
  bushel_weight = 56 → 
  ear_weight = 0.5 → 
  bushels_picked = 2 → 
  (bushels_picked * bushel_weight / ear_weight : ℝ) = 224 := by
sorry

end NUMINAMATH_CALUDE_corn_cobs_picked_l2971_297166


namespace NUMINAMATH_CALUDE_mystery_compound_is_nh4_l2971_297186

/-- Represents the atomic weight of an element -/
structure AtomicWeight where
  value : ℝ
  positive : value > 0

/-- Represents a chemical compound -/
structure Compound where
  molecularWeight : ℝ
  nitrogenCount : ℕ
  otherElementCount : ℕ
  otherElementWeight : AtomicWeight

/-- The atomic weight of nitrogen -/
def nitrogenWeight : AtomicWeight :=
  { value := 14.01, positive := by norm_num }

/-- The atomic weight of hydrogen -/
def hydrogenWeight : AtomicWeight :=
  { value := 1.01, positive := by norm_num }

/-- The compound in question -/
def mysteryCompound : Compound :=
  { molecularWeight := 18,
    nitrogenCount := 1,
    otherElementCount := 4,
    otherElementWeight := hydrogenWeight }

/-- Theorem stating that the mystery compound must be NH₄ -/
theorem mystery_compound_is_nh4 :
  ∀ (c : Compound),
    c.molecularWeight = 18 →
    c.nitrogenCount = 1 →
    c.otherElementWeight.value * c.otherElementCount + nitrogenWeight.value = c.molecularWeight →
    c = mysteryCompound :=
  sorry

end NUMINAMATH_CALUDE_mystery_compound_is_nh4_l2971_297186


namespace NUMINAMATH_CALUDE_sqrt_50_between_consecutive_integers_product_l2971_297167

theorem sqrt_50_between_consecutive_integers_product : ∃ n : ℕ, 
  (n : ℝ) < Real.sqrt 50 ∧ Real.sqrt 50 < (n + 1 : ℝ) ∧ n * (n + 1) = 56 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_50_between_consecutive_integers_product_l2971_297167


namespace NUMINAMATH_CALUDE_smallest_prime_dividing_sum_l2971_297169

theorem smallest_prime_dividing_sum : 
  ∃ (p : Nat), Nat.Prime p ∧ p ∣ (7^15 + 9^17) ∧ ∀ (q : Nat), Nat.Prime q → q ∣ (7^15 + 9^17) → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_dividing_sum_l2971_297169


namespace NUMINAMATH_CALUDE_customers_without_tip_waiter_tip_problem_l2971_297164

theorem customers_without_tip (initial_customers : ℕ) (additional_customers : ℕ) (customers_with_tip : ℕ) : ℕ :=
  let total_customers := initial_customers + additional_customers
  total_customers - customers_with_tip

theorem waiter_tip_problem : customers_without_tip 29 20 15 = 34 := by
  sorry

end NUMINAMATH_CALUDE_customers_without_tip_waiter_tip_problem_l2971_297164


namespace NUMINAMATH_CALUDE_polynomial_subtraction_l2971_297135

theorem polynomial_subtraction (x : ℝ) :
  (2 * x^6 + 3 * x^5 + x^4 - x^2 + 15) - (x^6 + x^5 - 2 * x^4 + x^3 + 5) =
  x^6 + 2 * x^5 + 3 * x^4 - x^3 + x^2 + 10 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_subtraction_l2971_297135


namespace NUMINAMATH_CALUDE_same_color_probability_l2971_297180

/-- Represents the number of sides on each die -/
def totalSides : ℕ := 12

/-- Represents the number of red sides on each die -/
def redSides : ℕ := 3

/-- Represents the number of blue sides on each die -/
def blueSides : ℕ := 4

/-- Represents the number of green sides on each die -/
def greenSides : ℕ := 3

/-- Represents the number of purple sides on each die -/
def purpleSides : ℕ := 2

/-- Theorem stating the probability of rolling the same color on both dice -/
theorem same_color_probability : 
  (redSides^2 + blueSides^2 + greenSides^2 + purpleSides^2) / totalSides^2 = 19 / 72 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l2971_297180


namespace NUMINAMATH_CALUDE_tan_double_angle_second_quadrant_l2971_297113

/-- Given an angle α in the second quadrant with sin(π + α) = -3/5, prove that tan(2α) = -24/7 -/
theorem tan_double_angle_second_quadrant (α : Real) 
  (h1 : π/2 < α ∧ α < π) -- α is in the second quadrant
  (h2 : Real.sin (π + α) = -3/5) : 
  Real.tan (2 * α) = -24/7 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_second_quadrant_l2971_297113


namespace NUMINAMATH_CALUDE_cindy_calculation_l2971_297125

theorem cindy_calculation (x : ℝ) : 
  (x - 12) / 4 = 28 → (x - 5) / 8 = 14.875 := by sorry

end NUMINAMATH_CALUDE_cindy_calculation_l2971_297125


namespace NUMINAMATH_CALUDE_smallest_k_for_power_inequality_l2971_297128

theorem smallest_k_for_power_inequality : ∃ k : ℕ, k = 14 ∧ 
  (∀ n : ℕ, n < k → (7 : ℝ)^n ≤ 4^19) ∧ (7 : ℝ)^k > 4^19 := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_for_power_inequality_l2971_297128


namespace NUMINAMATH_CALUDE_A_D_mutually_exclusive_not_complementary_l2971_297132

-- Define the sample space for a die throw
def Ω : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define the events
def A : Set Nat := {n ∈ Ω | n % 2 = 1}
def B : Set Nat := {n ∈ Ω | n % 2 = 0}
def C : Set Nat := {n ∈ Ω | n % 2 = 0}
def D : Set Nat := {2, 4}

-- Define mutually exclusive
def mutually_exclusive (X Y : Set Nat) : Prop := X ∩ Y = ∅

-- Define complementary
def complementary (X Y : Set Nat) : Prop := X ∪ Y = Ω

-- Theorem to prove
theorem A_D_mutually_exclusive_not_complementary :
  mutually_exclusive A D ∧ ¬complementary A D :=
by sorry

end NUMINAMATH_CALUDE_A_D_mutually_exclusive_not_complementary_l2971_297132


namespace NUMINAMATH_CALUDE_dried_fruit_percentage_l2971_297137

/-- Represents the composition of a trail mix -/
structure TrailMix where
  nuts : ℚ
  dried_fruit : ℚ
  chocolate_chips : ℚ
  composition_sum : nuts + dried_fruit + chocolate_chips = 1

/-- The combined mixture of two trail mixes -/
def combined_mixture (mix1 mix2 : TrailMix) : TrailMix :=
  { nuts := (mix1.nuts + mix2.nuts) / 2,
    dried_fruit := (mix1.dried_fruit + mix2.dried_fruit) / 2,
    chocolate_chips := (mix1.chocolate_chips + mix2.chocolate_chips) / 2,
    composition_sum := by sorry }

theorem dried_fruit_percentage
  (sue_mix : TrailMix)
  (jane_mix : TrailMix)
  (h_sue_nuts : sue_mix.nuts = 0.3)
  (h_sue_dried_fruit : sue_mix.dried_fruit = 0.7)
  (h_jane_nuts : jane_mix.nuts = 0.6)
  (h_jane_chocolate : jane_mix.chocolate_chips = 0.4)
  (h_combined_nuts : (combined_mixture sue_mix jane_mix).nuts = 0.45) :
  (combined_mixture sue_mix jane_mix).dried_fruit = 0.35 := by
  sorry

end NUMINAMATH_CALUDE_dried_fruit_percentage_l2971_297137


namespace NUMINAMATH_CALUDE_intersection_and_union_range_of_a_l2971_297197

-- Define sets A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}

-- Define set C with parameter a
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

-- Theorem for part (1)
theorem intersection_and_union :
  (A ∩ B = {x | 3 ≤ x ∧ x < 6}) ∧
  ((Set.univ \ B) ∪ A = {x | x ≤ 2 ∨ (3 ≤ x ∧ x < 6) ∨ 9 ≤ x}) := by sorry

-- Theorem for part (2)
theorem range_of_a (a : ℝ) :
  (C a ⊆ B) ↔ (2 ≤ a ∧ a ≤ 8) := by sorry

end NUMINAMATH_CALUDE_intersection_and_union_range_of_a_l2971_297197


namespace NUMINAMATH_CALUDE_min_value_expression_l2971_297102

theorem min_value_expression (x y : ℝ) : 
  x^2 - 2*x*y + y^2 + 2*y + 1 ≥ 0 ∧ 
  ∃ (a b : ℝ), a^2 - 2*a*b + b^2 + 2*b + 1 = 0 := by
sorry

end NUMINAMATH_CALUDE_min_value_expression_l2971_297102


namespace NUMINAMATH_CALUDE_last_round_win_ratio_l2971_297101

/-- Represents the number of matches in a kickboxing competition --/
structure KickboxingCompetition where
  firstTwoRoundsMatches : ℕ  -- Total matches in first two rounds
  lastRoundMatches : ℕ      -- Total matches in last round
  totalWins : ℕ             -- Total matches won by Brendan

/-- Theorem stating the ratio of matches won in the last round --/
theorem last_round_win_ratio (comp : KickboxingCompetition)
  (h1 : comp.firstTwoRoundsMatches = 12)
  (h2 : comp.lastRoundMatches = 4)
  (h3 : comp.totalWins = 14) :
  (comp.totalWins - comp.firstTwoRoundsMatches) * 2 = comp.lastRoundMatches := by
  sorry

#check last_round_win_ratio

end NUMINAMATH_CALUDE_last_round_win_ratio_l2971_297101


namespace NUMINAMATH_CALUDE_units_digit_2019_power_2019_l2971_297192

theorem units_digit_2019_power_2019 : (2019^2019) % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_2019_power_2019_l2971_297192


namespace NUMINAMATH_CALUDE_change5_is_census_change5_most_suitable_for_census_l2971_297141

/-- Represents a survey method -/
inductive SurveyMethod
  | Sample
  | Census

/-- Represents a survey target -/
structure SurveyTarget where
  name : String
  method : SurveyMethod

/-- Definition of a census -/
def isCensus (target : SurveyTarget) : Prop :=
  target.method = SurveyMethod.Census

/-- The "Chang'e 5" probe components survey -/
def change5Survey : SurveyTarget :=
  { name := "All components of the Chang'e 5 probe"
    method := SurveyMethod.Census }

/-- Theorem: The "Chang'e 5" probe components survey is a census -/
theorem change5_is_census : isCensus change5Survey := by
  sorry

/-- Theorem: The "Chang'e 5" probe components survey is the most suitable for a census -/
theorem change5_most_suitable_for_census (other : SurveyTarget) :
    isCensus other → other = change5Survey := by
  sorry

end NUMINAMATH_CALUDE_change5_is_census_change5_most_suitable_for_census_l2971_297141


namespace NUMINAMATH_CALUDE_spherical_shell_surface_area_l2971_297106

/-- The surface area of a spherical shell formed by two hemispheres -/
theorem spherical_shell_surface_area 
  (r : ℝ) -- radius of the inner hemisphere
  (h1 : r > 0) -- radius is positive
  (h2 : r^2 * π = 200 * π) -- base area of inner hemisphere is 200π
  : 2 * π * ((r + 1)^2 - r^2) = 2 * π + 40 * Real.sqrt 2 * π :=
by sorry

end NUMINAMATH_CALUDE_spherical_shell_surface_area_l2971_297106


namespace NUMINAMATH_CALUDE_remainder_when_consecutive_primes_l2971_297191

theorem remainder_when_consecutive_primes (n : ℕ) :
  Nat.Prime (n + 3) ∧ Nat.Prime (n + 7) → n % 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_when_consecutive_primes_l2971_297191


namespace NUMINAMATH_CALUDE_equilateral_triangle_vertex_l2971_297138

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if three points form an equilateral triangle -/
def isEquilateral (a b c : Point) : Prop :=
  (a.x - b.x)^2 + (a.y - b.y)^2 = (b.x - c.x)^2 + (b.y - c.y)^2 ∧
  (b.x - c.x)^2 + (b.y - c.y)^2 = (c.x - a.x)^2 + (c.y - a.y)^2

/-- Checks if a point is on the altitude from another point to a line segment -/
def isOnAltitude (a d : Point) (b c : Point) : Prop :=
  (d.x - b.x) * (c.x - b.x) + (d.y - b.y) * (c.y - b.y) = 0 ∧
  (a.x - d.x) * (c.x - b.x) + (a.y - d.y) * (c.y - b.y) = 0

theorem equilateral_triangle_vertex (a b d : Point) : 
  a = Point.mk 10 4 →
  b = Point.mk 1 (-5) →
  d = Point.mk 0 (-2) →
  ∃ c : Point, 
    isEquilateral a b c ∧ 
    isOnAltitude a d b c ∧ 
    c = Point.mk (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_vertex_l2971_297138


namespace NUMINAMATH_CALUDE_no_natural_solutions_l2971_297183

theorem no_natural_solutions : ¬∃ (x y : ℕ), (2 * x + y) * (2 * y + x) = 2017^2017 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solutions_l2971_297183


namespace NUMINAMATH_CALUDE_prob_same_color_is_34_100_l2971_297157

/-- Represents an urn with balls of different colors -/
structure Urn :=
  (blue : ℕ)
  (red : ℕ)
  (green : ℕ)

/-- Calculates the total number of balls in an urn -/
def Urn.total (u : Urn) : ℕ := u.blue + u.red + u.green

/-- Calculates the probability of drawing a ball of a specific color from an urn -/
def prob_color (u : Urn) (color : ℕ) : ℚ :=
  color / u.total

/-- Calculates the probability of drawing balls of the same color from two urns -/
def prob_same_color (u1 u2 : Urn) : ℚ :=
  prob_color u1 u1.blue * prob_color u2 u2.blue +
  prob_color u1 u1.red * prob_color u2 u2.red +
  prob_color u1 u1.green * prob_color u2 u2.green

/-- The main theorem stating that the probability of drawing balls of the same color
    from the given urns is 0.34 -/
theorem prob_same_color_is_34_100 :
  let u1 : Urn := ⟨2, 3, 5⟩
  let u2 : Urn := ⟨4, 2, 4⟩
  prob_same_color u1 u2 = 34/100 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_color_is_34_100_l2971_297157


namespace NUMINAMATH_CALUDE_mary_bike_rental_cost_l2971_297117

/-- Calculates the total cost of bike rental given the fixed fee, hourly rate, and duration. -/
def bikeRentalCost (fixedFee : ℕ) (hourlyRate : ℕ) (duration : ℕ) : ℕ :=
  fixedFee + hourlyRate * duration

/-- Theorem stating that the bike rental cost for Mary is $80 -/
theorem mary_bike_rental_cost :
  bikeRentalCost 17 7 9 = 80 := by
  sorry

end NUMINAMATH_CALUDE_mary_bike_rental_cost_l2971_297117


namespace NUMINAMATH_CALUDE_prime_pythagorean_inequality_l2971_297199

theorem prime_pythagorean_inequality (p m n : ℕ) 
  (prime_p : Nat.Prime p) 
  (pos_m : m > 0) 
  (pos_n : n > 0) 
  (pyth_eq : p^2 + m^2 = n^2) : 
  m > p := by
sorry

end NUMINAMATH_CALUDE_prime_pythagorean_inequality_l2971_297199


namespace NUMINAMATH_CALUDE_sqrt_primes_not_arithmetic_progression_l2971_297188

theorem sqrt_primes_not_arithmetic_progression (a b c : ℕ) 
  (ha : Prime a) (hb : Prime b) (hc : Prime c) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) : 
  ¬∃ (d : ℝ), (Real.sqrt (a : ℝ) + d = Real.sqrt (b : ℝ) ∧ 
               Real.sqrt (b : ℝ) + d = Real.sqrt (c : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_primes_not_arithmetic_progression_l2971_297188


namespace NUMINAMATH_CALUDE_math_books_count_l2971_297107

theorem math_books_count (total_books : ℕ) (math_cost history_cost total_price : ℚ)
  (h1 : total_books = 80)
  (h2 : math_cost = 4)
  (h3 : history_cost = 5)
  (h4 : total_price = 390) :
  ∃ (math_books : ℕ),
    math_books * math_cost + (total_books - math_books) * history_cost = total_price ∧
    math_books = 10 := by
  sorry

end NUMINAMATH_CALUDE_math_books_count_l2971_297107


namespace NUMINAMATH_CALUDE_inequality_solution_set_sum_of_coordinates_l2971_297149

-- Problem 1
theorem inequality_solution_set (x : ℝ) :
  x + |2*x - 1| < 3 ↔ -2 < x ∧ x < 4/3 := by sorry

-- Problem 2
theorem sum_of_coordinates (x y z : ℝ) 
  (h1 : x^2 + y^2 + z^2 = 1) 
  (h2 : x + 2*y + 3*z = Real.sqrt 14) : 
  x + y + z = 3 * Real.sqrt 14 / 7 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_sum_of_coordinates_l2971_297149


namespace NUMINAMATH_CALUDE_problem_solution_l2971_297114

/-- Given constants a, b, and c satisfying the specified conditions, prove that a + 2b + 3c = 74 -/
theorem problem_solution (a b c : ℝ) 
  (h1 : ∀ x, (x - a) * (x - b) / (x - c) ≤ 0 ↔ (x < -6 ∨ |x - 30| ≤ 2))
  (h2 : a < b) : 
  a + 2 * b + 3 * c = 74 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2971_297114


namespace NUMINAMATH_CALUDE_socks_cost_l2971_297174

theorem socks_cost (num_players : ℕ) (jersey_cost shorts_cost total_cost : ℚ) : 
  num_players = 16 →
  jersey_cost = 25 →
  shorts_cost = 15.20 →
  total_cost = 752 →
  ∃ (socks_cost : ℚ), 
    num_players * (jersey_cost + shorts_cost + socks_cost) = total_cost ∧ 
    socks_cost = 6.80 := by
  sorry

end NUMINAMATH_CALUDE_socks_cost_l2971_297174


namespace NUMINAMATH_CALUDE_tangent_line_constraint_l2971_297170

/-- Given a cubic function f(x) = x³ - (1/2)x² + bx + c, 
    if f has a tangent line parallel to y = 1, then b ≤ 1/12 -/
theorem tangent_line_constraint (b c : ℝ) : 
  (∃ x : ℝ, (3*x^2 - x + b) = 1) → b ≤ 1/12 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_constraint_l2971_297170


namespace NUMINAMATH_CALUDE_min_value_sum_fractions_l2971_297158

theorem min_value_sum_fractions (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + 1) / c + (a + c + 1) / b + (b + c + 1) / a ≥ 9 ∧
  ∃ (a₀ b₀ c₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ 0 < c₀ ∧
    (a₀ + b₀ + 1) / c₀ + (a₀ + c₀ + 1) / b₀ + (b₀ + c₀ + 1) / a₀ = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_fractions_l2971_297158


namespace NUMINAMATH_CALUDE_bob_jacket_purchase_percentage_l2971_297178

/-- Calculates the percentage of the suggested retail price that Bob paid for a jacket -/
theorem bob_jacket_purchase_percentage (P : ℝ) (P_pos : P > 0) : 
  let marked_price := P * (1 - 0.4)
  let bob_price := marked_price * (1 - 0.4)
  bob_price / P = 0.36 := by
  sorry

end NUMINAMATH_CALUDE_bob_jacket_purchase_percentage_l2971_297178


namespace NUMINAMATH_CALUDE_remaining_problems_calculation_l2971_297187

/-- Given a number of worksheets, problems per worksheet, and graded worksheets,
    calculate the number of remaining problems to grade. -/
def remaining_problems (total_worksheets : ℕ) (problems_per_worksheet : ℕ) (graded_worksheets : ℕ) : ℕ :=
  (total_worksheets - graded_worksheets) * problems_per_worksheet

theorem remaining_problems_calculation :
  remaining_problems 16 4 8 = 32 := by
  sorry

end NUMINAMATH_CALUDE_remaining_problems_calculation_l2971_297187


namespace NUMINAMATH_CALUDE_reverse_sum_divisibility_l2971_297195

def reverse_number (n : ℕ) : ℕ := sorry

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem reverse_sum_divisibility (n : ℕ) (m : ℕ) (h1 : n ≥ 10^(m-1)) (h2 : n < 10^m) :
  (81 ∣ (n + reverse_number n)) ↔ (81 ∣ sum_of_digits n) := by sorry

end NUMINAMATH_CALUDE_reverse_sum_divisibility_l2971_297195


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2971_297115

/-- Given a hyperbola with the equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    left and right foci F₁ and F₂, and a point P(3,4) on its asymptote,
    prove that if |PF₁ + PF₂| = |F₁F₂|, then the equation of the hyperbola is x²/9 - y²/16 = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (F₁ F₂ : ℝ × ℝ) (hF : F₁.1 < F₂.1)
  (P : ℝ × ℝ) (hP : P = (3, 4))
  (h_asymptote : ∃ (k : ℝ), P.2 = k * P.1 ∧ k^2 * a^2 = b^2)
  (h_foci : |P - F₁ + (P - F₂)| = |F₂ - F₁|) :
  ∀ (x y : ℝ), x^2 / 9 - y^2 / 16 = 1 ↔ x^2 / a^2 - y^2 / b^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2971_297115


namespace NUMINAMATH_CALUDE_quadratic_roots_for_negative_k_l2971_297131

theorem quadratic_roots_for_negative_k (k : ℝ) (h : k < 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + x₁ + k - 1 = 0 ∧ x₂^2 + x₂ + k - 1 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_for_negative_k_l2971_297131


namespace NUMINAMATH_CALUDE_local_max_value_l2971_297144

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x

-- State the theorem
theorem local_max_value (a : ℝ) :
  (∃ (x : ℝ), x = 2 ∧ IsLocalMin (f a) x) →
  (∃ (y : ℝ), IsLocalMax (f a) y ∧ f a y = 16) :=
by sorry

end NUMINAMATH_CALUDE_local_max_value_l2971_297144


namespace NUMINAMATH_CALUDE_not_first_class_probability_l2971_297147

theorem not_first_class_probability 
  (P_A P_B P_C : ℝ) 
  (h_A : P_A = 0.65) 
  (h_B : P_B = 0.2) 
  (h_C : P_C = 0.1) :
  1 - P_A = 0.35 := by
  sorry

end NUMINAMATH_CALUDE_not_first_class_probability_l2971_297147


namespace NUMINAMATH_CALUDE_chess_tournament_theorem_l2971_297127

/-- A chess tournament with the given conditions -/
structure ChessTournament where
  num_players : ℕ
  games_per_player : ℕ
  losses_per_player : ℕ
  no_ties : Bool

/-- Calculates the total number of games in the tournament -/
def total_games (t : ChessTournament) : ℕ :=
  t.num_players * (t.num_players - 1) / 2

/-- Calculates the number of wins for each player -/
def wins_per_player (t : ChessTournament) : ℕ :=
  t.games_per_player - t.losses_per_player

/-- The theorem to be proved -/
theorem chess_tournament_theorem (t : ChessTournament) 
  (h1 : t.num_players = 200)
  (h2 : t.games_per_player = 199)
  (h3 : t.losses_per_player = 30)
  (h4 : t.no_ties = true) :
  total_games t = 19900 ∧ wins_per_player t = 169 := by
  sorry


end NUMINAMATH_CALUDE_chess_tournament_theorem_l2971_297127


namespace NUMINAMATH_CALUDE_parallel_vectors_l2971_297152

theorem parallel_vectors (x : ℝ) : 
  (∃ k : ℝ, k ≠ 0 ∧ k • (1, x) = (x - 1, 2)) → x = 1 ∨ x = 2 :=
sorry

end NUMINAMATH_CALUDE_parallel_vectors_l2971_297152


namespace NUMINAMATH_CALUDE_common_root_exists_polynomial_common_root_l2971_297109

def coefficients : Finset Int := {-7, 4, -3, 6}

theorem common_root_exists (a b c d : Int) 
  (h : {a, b, c, d} = coefficients) : 
  (a : ℝ) + b + c + d = 0 := by sorry

theorem polynomial_common_root (a b c d : Int) 
  (h : {a, b, c, d} = coefficients) :
  ∃ (x : ℝ), a * x^3 + b * x^2 + c * x + d = 0 := by sorry

end NUMINAMATH_CALUDE_common_root_exists_polynomial_common_root_l2971_297109


namespace NUMINAMATH_CALUDE_solve_triangle_equation_l2971_297134

-- Define the ∆ operation
def triangle (A B : ℕ) : ℕ := 2 * A + B

-- Theorem statement
theorem solve_triangle_equation : 
  ∃ x : ℕ, triangle (triangle 3 2) x = 20 ∧ x = 4 := by
sorry

end NUMINAMATH_CALUDE_solve_triangle_equation_l2971_297134


namespace NUMINAMATH_CALUDE_cubic_equation_root_l2971_297160

theorem cubic_equation_root (a b : ℚ) : 
  (2 - 3 * Real.sqrt 3) ^ 3 + a * (2 - 3 * Real.sqrt 3) ^ 2 + b * (2 - 3 * Real.sqrt 3) - 37 = 0 →
  a = -55/23 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l2971_297160


namespace NUMINAMATH_CALUDE_second_player_cannot_lose_l2971_297111

/-- Represents a player in the game -/
inductive Player : Type
| First : Player
| Second : Player

/-- Represents a move in the game -/
structure Move where
  player : Player
  moveNumber : Nat

/-- Represents the state of the game -/
structure GameState where
  currentMove : Move
  isGameOver : Bool

/-- The game can only end on an even-numbered move -/
axiom game_ends_on_even_move : 
  ∀ (gs : GameState), gs.isGameOver → gs.currentMove.moveNumber % 2 = 0

/-- The first player makes even-numbered moves -/
axiom first_player_even_moves :
  ∀ (m : Move), m.player = Player.First → m.moveNumber % 2 = 0

/-- Theorem: The second player cannot lose -/
theorem second_player_cannot_lose :
  ∀ (gs : GameState), gs.isGameOver → gs.currentMove.player ≠ Player.Second :=
by sorry


end NUMINAMATH_CALUDE_second_player_cannot_lose_l2971_297111


namespace NUMINAMATH_CALUDE_boat_journey_distance_l2971_297150

/-- Calculates the total distance covered by a man rowing a boat in a river with varying currents. -/
theorem boat_journey_distance
  (man_speed : ℝ)
  (current1_speed : ℝ)
  (current1_time : ℝ)
  (current2_speed : ℝ)
  (current2_time : ℝ)
  (h1 : man_speed = 15)
  (h2 : current1_speed = 2.5)
  (h3 : current1_time = 2)
  (h4 : current2_speed = 3)
  (h5 : current2_time = 1.5) :
  (man_speed + current1_speed) * current1_time +
  (man_speed - current2_speed) * current2_time = 53 := by
sorry


end NUMINAMATH_CALUDE_boat_journey_distance_l2971_297150


namespace NUMINAMATH_CALUDE_percentage_calculation_l2971_297177

theorem percentage_calculation (total : ℝ) (part : ℝ) (h1 : total = 600) (h2 : part = 150) :
  (part / total) * 100 = 25 := by
sorry

end NUMINAMATH_CALUDE_percentage_calculation_l2971_297177


namespace NUMINAMATH_CALUDE_complement_of_A_l2971_297139

def U : Set Nat := {1,2,3,4,5,6,7}
def A : Set Nat := {1,2,4,5}

theorem complement_of_A :
  (U \ A) = {3,6,7} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l2971_297139


namespace NUMINAMATH_CALUDE_rationalization_factor_l2971_297105

theorem rationalization_factor (a b : ℝ) :
  (Real.sqrt a - Real.sqrt b) * (Real.sqrt a + Real.sqrt b) = a - b :=
by sorry

end NUMINAMATH_CALUDE_rationalization_factor_l2971_297105


namespace NUMINAMATH_CALUDE_regular_hexagon_perimeter_l2971_297140

/-- The perimeter of a regular hexagon with radius √3 is 6√3 -/
theorem regular_hexagon_perimeter (r : ℝ) (h : r = Real.sqrt 3) : 
  6 * r = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_regular_hexagon_perimeter_l2971_297140


namespace NUMINAMATH_CALUDE_fraction_inequality_l2971_297122

theorem fraction_inequality (a b : ℝ) (h : a > b) : a / 4 > b / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l2971_297122


namespace NUMINAMATH_CALUDE_mandy_sister_age_difference_l2971_297145

/-- Represents the ages and relationships in Mandy's family -/
structure Family where
  mandy_age : ℕ
  brother_age : ℕ
  sister_age : ℕ
  brother_age_relation : brother_age = 4 * mandy_age
  sister_age_relation : sister_age = brother_age - 5

/-- Calculates the age difference between Mandy and her sister -/
def age_difference (f : Family) : ℕ :=
  f.sister_age - f.mandy_age

/-- Theorem stating the age difference between Mandy and her sister -/
theorem mandy_sister_age_difference (f : Family) (h : f.mandy_age = 3) :
  age_difference f = 4 := by
  sorry

#check mandy_sister_age_difference

end NUMINAMATH_CALUDE_mandy_sister_age_difference_l2971_297145


namespace NUMINAMATH_CALUDE_gasoline_tank_capacity_l2971_297176

theorem gasoline_tank_capacity :
  ∀ x : ℚ,
  (5/6 : ℚ) * x - (2/3 : ℚ) * x = 15 →
  x = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_gasoline_tank_capacity_l2971_297176


namespace NUMINAMATH_CALUDE_james_truck_mpg_james_truck_mpg_proof_l2971_297112

/-- Proves that given the conditions of James's truck driving job, his truck's fuel efficiency is 20 miles per gallon. -/
theorem james_truck_mpg : ℝ → Prop :=
  λ mpg : ℝ =>
    let pay_per_mile : ℝ := 0.5
    let gas_cost_per_gallon : ℝ := 4
    let trip_distance : ℝ := 600
    let profit : ℝ := 180
    let earnings : ℝ := pay_per_mile * trip_distance
    let gas_cost : ℝ := (trip_distance / mpg) * gas_cost_per_gallon
    earnings - gas_cost = profit → mpg = 20

/-- The proof of james_truck_mpg. -/
theorem james_truck_mpg_proof : james_truck_mpg 20 := by
  sorry

end NUMINAMATH_CALUDE_james_truck_mpg_james_truck_mpg_proof_l2971_297112


namespace NUMINAMATH_CALUDE_total_oranges_count_l2971_297163

/-- The number of oranges picked by Joan -/
def joan_oranges : ℕ := 37

/-- The number of oranges picked by Sara -/
def sara_oranges : ℕ := 10

/-- The total number of oranges picked -/
def total_oranges : ℕ := joan_oranges + sara_oranges

theorem total_oranges_count : total_oranges = 47 := by
  sorry

end NUMINAMATH_CALUDE_total_oranges_count_l2971_297163


namespace NUMINAMATH_CALUDE_power_of_power_five_l2971_297151

theorem power_of_power_five : (5^2)^4 = 390625 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_five_l2971_297151


namespace NUMINAMATH_CALUDE_square_root_divided_by_18_l2971_297121

theorem square_root_divided_by_18 : Real.sqrt 5184 / 18 = 4 := by sorry

end NUMINAMATH_CALUDE_square_root_divided_by_18_l2971_297121


namespace NUMINAMATH_CALUDE_min_value_bn_Sn_l2971_297162

def a (n : ℕ) : ℕ := n * (n + 1)

def S (n : ℕ) : ℚ := 1 - 1 / (n + 1)

def b (n : ℕ) : ℤ := n - 8

theorem min_value_bn_Sn :
  (∀ n : ℕ, (b n : ℚ) * S n ≥ -4) ∧
  (∃ n : ℕ, (b n : ℚ) * S n = -4) :=
sorry

end NUMINAMATH_CALUDE_min_value_bn_Sn_l2971_297162


namespace NUMINAMATH_CALUDE_largest_b_value_l2971_297171

theorem largest_b_value (b : ℝ) (h : (3*b + 4)*(b - 2) = 7*b) :
  b ≤ (9 + Real.sqrt 177) / 6 ∧
  ∃ (b : ℝ), (3*b + 4)*(b - 2) = 7*b ∧ b = (9 + Real.sqrt 177) / 6 :=
by sorry

end NUMINAMATH_CALUDE_largest_b_value_l2971_297171


namespace NUMINAMATH_CALUDE_rachel_books_total_l2971_297108

/-- The number of books Rachel has in total -/
def total_books (mystery_shelves picture_shelves scifi_shelves bio_shelves books_per_shelf : ℕ) : ℕ :=
  (mystery_shelves + picture_shelves + scifi_shelves + bio_shelves) * books_per_shelf

/-- Theorem stating that Rachel has 135 books in total -/
theorem rachel_books_total :
  total_books 6 2 3 4 9 = 135 := by
  sorry

end NUMINAMATH_CALUDE_rachel_books_total_l2971_297108


namespace NUMINAMATH_CALUDE_parkway_elementary_girls_not_playing_soccer_l2971_297173

theorem parkway_elementary_girls_not_playing_soccer 
  (total_students : ℕ) 
  (total_boys : ℕ) 
  (total_soccer_players : ℕ) 
  (boys_soccer_percentage : ℚ) :
  total_students = 450 →
  total_boys = 320 →
  total_soccer_players = 250 →
  boys_soccer_percentage = 86 / 100 →
  ∃ (girls_not_playing_soccer : ℕ), 
    girls_not_playing_soccer = 
      total_students - total_boys - 
      (total_soccer_players - (boys_soccer_percentage * total_soccer_players).floor) :=
by
  sorry

end NUMINAMATH_CALUDE_parkway_elementary_girls_not_playing_soccer_l2971_297173


namespace NUMINAMATH_CALUDE_calculator_game_sum_l2971_297103

/-- Represents the state of the three calculators -/
structure CalculatorState where
  first : ℕ
  second : ℕ
  third : ℤ

/-- Applies the operations to the calculator state -/
def applyOperations (state : CalculatorState) : CalculatorState :=
  { first := state.first ^ 2,
    second := state.second ^ 3,
    third := -state.third }

/-- Applies the operations n times to the initial state -/
def nOperations (n : ℕ) : CalculatorState :=
  match n with
  | 0 => { first := 2, second := 1, third := -2 }
  | n + 1 => applyOperations (nOperations n)

theorem calculator_game_sum (N : ℕ) :
  ∃ (n : ℕ), n > 0 ∧ 
    let finalState := nOperations 50
    finalState.first = N ∧ 
    finalState.second = 1 ∧ 
    finalState.third = -2 ∧
    (finalState.first : ℤ) + finalState.second + finalState.third = N - 1 :=
  sorry

end NUMINAMATH_CALUDE_calculator_game_sum_l2971_297103


namespace NUMINAMATH_CALUDE_sum_condition_iff_divisible_l2971_297148

/-- An arithmetic progression with first term a and common difference d. -/
structure ArithmeticProgression (α : Type*) [Ring α] where
  a : α
  d : α

/-- The nth term of an arithmetic progression. -/
def ArithmeticProgression.nthTerm {α : Type*} [Ring α] (ap : ArithmeticProgression α) (n : ℕ) : α :=
  ap.a + n • ap.d

/-- Condition for the sum of two terms to be another term in the progression. -/
def SumCondition {α : Type*} [Ring α] (ap : ArithmeticProgression α) : Prop :=
  ∀ n k : ℕ, ∃ p : ℕ, ap.nthTerm n + ap.nthTerm k = ap.nthTerm p

/-- Theorem: The sum condition holds if and only if the first term is divisible by the common difference. -/
theorem sum_condition_iff_divisible {α : Type*} [CommRing α] (ap : ArithmeticProgression α) :
    SumCondition ap ↔ ∃ m : α, ap.a = m * ap.d :=
  sorry

end NUMINAMATH_CALUDE_sum_condition_iff_divisible_l2971_297148


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2971_297165

theorem trigonometric_identity (x y z : Real) 
  (hm : m = Real.sin x / Real.sin (y - z))
  (hn : n = Real.sin y / Real.sin (z - x))
  (hp : p = Real.sin z / Real.sin (x - y)) :
  m * n + n * p + p * m = -1 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2971_297165


namespace NUMINAMATH_CALUDE_volume_between_spheres_l2971_297116

theorem volume_between_spheres (π : ℝ) (h : π > 0) :
  let volume_sphere (r : ℝ) := (4 / 3) * π * r^3
  (volume_sphere 10 - volume_sphere 4) = (3744 / 3) * π :=
by
  sorry

end NUMINAMATH_CALUDE_volume_between_spheres_l2971_297116


namespace NUMINAMATH_CALUDE_bacteria_count_theorem_l2971_297184

/-- The number of bacteria after growth, given the original count and increase. -/
def bacteria_after_growth (original : ℕ) (increase : ℕ) : ℕ :=
  original + increase

/-- Theorem stating that the number of bacteria after growth is 8917,
    given the original count of 600 and an increase of 8317. -/
theorem bacteria_count_theorem :
  bacteria_after_growth 600 8317 = 8917 := by
  sorry

end NUMINAMATH_CALUDE_bacteria_count_theorem_l2971_297184


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2971_297146

theorem hyperbola_equation (a b c : ℝ) : 
  (2 * c = 10) →  -- focal length is 10
  (b / a = 2) →   -- slope of asymptote is 2
  (a^2 + b^2 = c^2) →  -- relation between a, b, and c
  (a^2 = 5 ∧ b^2 = 20) := by
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2971_297146


namespace NUMINAMATH_CALUDE_polygon_with_six_diagonals_has_nine_vertices_l2971_297181

/-- The number of vertices in a polygon given the number of diagonals from one vertex -/
def vertices_from_diagonals (diagonals : ℕ) : ℕ := diagonals + 3

/-- Theorem: A polygon with 6 diagonals drawn from one vertex has 9 vertices -/
theorem polygon_with_six_diagonals_has_nine_vertices :
  vertices_from_diagonals 6 = 9 := by
  sorry

end NUMINAMATH_CALUDE_polygon_with_six_diagonals_has_nine_vertices_l2971_297181


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2971_297136

/-- An isosceles triangle with two sides of length 8 cm and perimeter 25 cm has a base of length 9 cm. -/
theorem isosceles_triangle_base_length 
  (a b c : ℝ) 
  (h_isosceles : a = b) 
  (h_congruent_sides : a = 8) 
  (h_perimeter : a + b + c = 25) : 
  c = 9 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2971_297136


namespace NUMINAMATH_CALUDE_pictures_on_sixth_day_l2971_297185

def artists_group1 : ℕ := 6
def artists_group2 : ℕ := 8
def days_interval1 : ℕ := 2
def days_interval2 : ℕ := 3
def days_observed : ℕ := 5
def pictures_in_5_days : ℕ := 30

theorem pictures_on_sixth_day :
  let total_6_days := artists_group1 * (6 / days_interval1) + artists_group2 * (6 / days_interval2)
  (total_6_days - pictures_in_5_days : ℕ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_pictures_on_sixth_day_l2971_297185


namespace NUMINAMATH_CALUDE_circle_on_parabola_passes_through_focus_l2971_297190

/-- A circle with center on the parabola y^2 = 8x and tangent to x + 2 = 0 passes through (2, 0) -/
theorem circle_on_parabola_passes_through_focus (c : ℝ × ℝ) (r : ℝ) :
  c.2^2 = 8 * c.1 →  -- center is on the parabola y^2 = 8x
  r = c.1 + 2 →      -- circle is tangent to x + 2 = 0
  (c.1 - 2)^2 + c.2^2 = r^2  -- point (2, 0) is on the circle
  := by sorry

end NUMINAMATH_CALUDE_circle_on_parabola_passes_through_focus_l2971_297190


namespace NUMINAMATH_CALUDE_area_triangle_AOB_l2971_297196

/-- Given a sector AOB with area 2π/3 and radius 2, the area of triangle AOB is √3. -/
theorem area_triangle_AOB (S : ℝ) (r : ℝ) (h1 : S = 2 * π / 3) (h2 : r = 2) :
  (1 / 2) * r^2 * Real.sin (S / r^2) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_area_triangle_AOB_l2971_297196


namespace NUMINAMATH_CALUDE_coyote_prints_time_l2971_297193

/-- The time elapsed since the coyote left the prints -/
def time_elapsed : ℝ := 2

/-- The speed of the coyote in miles per hour -/
def coyote_speed : ℝ := 15

/-- The speed of Darrel in miles per hour -/
def darrel_speed : ℝ := 30

/-- The time it takes Darrel to catch up to the coyote in hours -/
def catch_up_time : ℝ := 1

theorem coyote_prints_time :
  time_elapsed * coyote_speed = darrel_speed * catch_up_time :=
sorry

end NUMINAMATH_CALUDE_coyote_prints_time_l2971_297193


namespace NUMINAMATH_CALUDE_smallest_sum_of_a_and_b_l2971_297120

theorem smallest_sum_of_a_and_b (a b : ℝ) : 
  a > 0 → b > 0 → 
  ((3 * a)^2 ≥ 16 * b) → 
  ((4 * b)^2 ≥ 12 * a) → 
  a + b ≥ 70/3 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_of_a_and_b_l2971_297120


namespace NUMINAMATH_CALUDE_defective_shipped_percentage_l2971_297189

theorem defective_shipped_percentage
  (total_units : ℕ)
  (defective_rate : ℚ)
  (shipped_rate : ℚ)
  (h1 : defective_rate = 5 / 100)
  (h2 : shipped_rate = 4 / 100) :
  (defective_rate * shipped_rate) * 100 = 0.2 := by
sorry

end NUMINAMATH_CALUDE_defective_shipped_percentage_l2971_297189


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_product_l2971_297143

theorem quadratic_roots_sum_product (a b : ℝ) : 
  a^2 + a - 1 = 0 → b^2 + b - 1 = 0 → a ≠ b → ab + a + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_product_l2971_297143


namespace NUMINAMATH_CALUDE_no_solution_for_equation_l2971_297172

theorem no_solution_for_equation : ¬ ∃ (x : ℝ), (x - 8) / (x - 7) - 8 = 1 / (7 - x) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_equation_l2971_297172


namespace NUMINAMATH_CALUDE_product_and_multiple_l2971_297110

theorem product_and_multiple : ∃ (ε : ℝ) (x : ℝ), 
  (ε > 0 ∧ ε < 1) ∧ 
  (abs (198 * 2 - 400) < ε) ∧ 
  (2 * x = 56) ∧ 
  (9 * x = 252) := by
  sorry

end NUMINAMATH_CALUDE_product_and_multiple_l2971_297110


namespace NUMINAMATH_CALUDE_calculation_result_l2971_297130

/-- The smallest two-digit prime number -/
def smallest_two_digit_prime : ℕ := 11

/-- The largest one-digit prime number -/
def largest_one_digit_prime : ℕ := 7

/-- The smallest one-digit prime number -/
def smallest_one_digit_prime : ℕ := 2

/-- Theorem stating the result of the calculation -/
theorem calculation_result :
  smallest_two_digit_prime * (largest_one_digit_prime ^ 2) - smallest_one_digit_prime = 537 := by
  sorry


end NUMINAMATH_CALUDE_calculation_result_l2971_297130


namespace NUMINAMATH_CALUDE_arccos_one_half_eq_pi_third_l2971_297142

theorem arccos_one_half_eq_pi_third : Real.arccos (1/2) = π/3 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_half_eq_pi_third_l2971_297142


namespace NUMINAMATH_CALUDE_distance_from_center_l2971_297155

-- Define the circle and points
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 50}

-- Define the conditions
def Conditions (A B C : ℝ × ℝ) : Prop :=
  A ∈ Circle ∧ 
  C ∈ Circle ∧ 
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 36 ∧  -- AB = 6
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = 4 ∧   -- BC = 2
  (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0  -- Angle ABC is right

-- State the theorem
theorem distance_from_center (A B C : ℝ × ℝ) 
  (h : Conditions A B C) : B.1^2 + B.2^2 = 26 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_center_l2971_297155


namespace NUMINAMATH_CALUDE_spirits_bottle_cost_l2971_297124

/-- Calculates the cost of a bottle of spirits given the number of servings,
    price per serving, and profit per bottle. -/
def bottle_cost (servings : ℕ) (price_per_serving : ℚ) (profit : ℚ) : ℚ :=
  servings * price_per_serving - profit

/-- Proves that the cost of a bottle of spirits is $30.00 under given conditions. -/
theorem spirits_bottle_cost :
  bottle_cost 16 8 98 = 30 := by
  sorry

end NUMINAMATH_CALUDE_spirits_bottle_cost_l2971_297124


namespace NUMINAMATH_CALUDE_chicken_burger_price_proof_l2971_297118

/-- The cost of a chicken burger in won -/
def chicken_burger_cost : ℕ := 3350

/-- The cost of a bulgogi burger in won -/
def bulgogi_burger_cost : ℕ := chicken_burger_cost + 300

/-- The total cost of three bulgogi burgers and three chicken burgers in won -/
def total_cost : ℕ := 21000

theorem chicken_burger_price_proof :
  chicken_burger_cost = 3350 ∧
  bulgogi_burger_cost = chicken_burger_cost + 300 ∧
  3 * chicken_burger_cost + 3 * bulgogi_burger_cost = total_cost :=
by sorry

end NUMINAMATH_CALUDE_chicken_burger_price_proof_l2971_297118


namespace NUMINAMATH_CALUDE_arithmetic_sequence_n_value_l2971_297175

/-- An arithmetic sequence with first term a₁, common difference d, and nth term aₙ -/
structure ArithmeticSequence where
  a₁ : ℝ
  d : ℝ
  n : ℕ
  aₙ : ℝ
  seq_def : aₙ = a₁ + (n - 1) * d

/-- The theorem stating that for the given arithmetic sequence, n = 100 -/
theorem arithmetic_sequence_n_value
  (seq : ArithmeticSequence)
  (h1 : seq.a₁ = 1)
  (h2 : seq.d = 3)
  (h3 : seq.aₙ = 298) :
  seq.n = 100 := by
  sorry

#check arithmetic_sequence_n_value

end NUMINAMATH_CALUDE_arithmetic_sequence_n_value_l2971_297175


namespace NUMINAMATH_CALUDE_compare_expressions_l2971_297179

theorem compare_expressions : -|(-3/4)| < -(-4/5) := by
  sorry

end NUMINAMATH_CALUDE_compare_expressions_l2971_297179


namespace NUMINAMATH_CALUDE_f_is_quadratic_l2971_297129

/-- Definition of a quadratic function -/
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function y = 2x² -/
def f (x : ℝ) : ℝ := 2 * x^2

/-- Theorem: f is a quadratic function -/
theorem f_is_quadratic : is_quadratic f := by
  sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l2971_297129


namespace NUMINAMATH_CALUDE_negation_equivalence_l2971_297198

theorem negation_equivalence (m : ℝ) :
  (¬ ∃ x < 0, x^2 + 2*x - m > 0) ↔ (∀ x < 0, x^2 + 2*x - m ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2971_297198


namespace NUMINAMATH_CALUDE_eighteen_to_binary_l2971_297100

def decimal_to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 2) ((m % 2) :: acc)
    aux n []

def binary_to_decimal (l : List ℕ) : ℕ :=
  l.foldl (fun acc d => 2 * acc + d) 0

theorem eighteen_to_binary :
  decimal_to_binary 18 = [1, 0, 0, 1, 0] ∧
  binary_to_decimal [1, 0, 0, 1, 0] = 18 :=
sorry

end NUMINAMATH_CALUDE_eighteen_to_binary_l2971_297100
