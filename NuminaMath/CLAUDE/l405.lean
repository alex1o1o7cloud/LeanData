import Mathlib

namespace NUMINAMATH_CALUDE_driver_total_stops_l405_40566

/-- The total number of stops made by a delivery driver -/
def total_stops (initial_stops additional_stops : ℕ) : ℕ :=
  initial_stops + additional_stops

/-- Theorem: The delivery driver made 7 stops in total -/
theorem driver_total_stops :
  total_stops 3 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_driver_total_stops_l405_40566


namespace NUMINAMATH_CALUDE_sampling_suitable_for_yangtze_l405_40592

/-- Represents a survey method -/
inductive SurveyMethod
| Census
| Sampling

/-- Represents a scenario to be surveyed -/
structure Scenario where
  name : String
  population_size : ℕ
  geographical_spread : ℕ
  measurement_difficulty : ℕ

/-- Determines if a survey method is suitable for a given scenario -/
def is_suitable (method : SurveyMethod) (scenario : Scenario) : Prop :=
  match method with
  | SurveyMethod.Sampling => 
      scenario.population_size > 1000000 ∧ 
      scenario.geographical_spread > 100 ∧
      scenario.measurement_difficulty > 5
  | SurveyMethod.Census => 
      scenario.population_size ≤ 1000000 ∧ 
      scenario.geographical_spread ≤ 100 ∧
      scenario.measurement_difficulty ≤ 5

/-- The Yangtze River Basin scenario -/
def yangtze_river_basin : Scenario :=
  { name := "Yangtze River Basin Fish Population"
    population_size := 10000000
    geographical_spread := 1000
    measurement_difficulty := 9 }

/-- Theorem stating that sampling is more suitable for the Yangtze River Basin scenario -/
theorem sampling_suitable_for_yangtze : 
  is_suitable SurveyMethod.Sampling yangtze_river_basin ∧ 
  ¬is_suitable SurveyMethod.Census yangtze_river_basin :=
sorry

end NUMINAMATH_CALUDE_sampling_suitable_for_yangtze_l405_40592


namespace NUMINAMATH_CALUDE_exactly_two_correct_propositions_l405_40539

-- Define the basic geometric concepts
def Line : Type := sorry
def intersect (l1 l2 : Line) : Prop := sorry
def perpendicular (l1 l2 : Line) : Prop := sorry
def angle (l1 l2 : Line) : ℝ := sorry
def supplementary_angle (α : ℝ) : ℝ := sorry
def adjacent_angle (l1 l2 : Line) (α : ℝ) : ℝ := sorry
def alternate_interior_angles (l1 l2 : Line) (α β : ℝ) : Prop := sorry
def same_side_interior_angles (l1 l2 : Line) (α β : ℝ) : Prop := sorry
def angle_bisector (l : Line) (α : ℝ) : Line := sorry
def complementary (α β : ℝ) : Prop := sorry

-- Define the four propositions
def proposition1 : Prop :=
  ∀ l1 l2 : Line, intersect l1 l2 →
    ∀ α : ℝ, adjacent_angle l1 l2 α = adjacent_angle l1 l2 (supplementary_angle α) →
      perpendicular l1 l2

def proposition2 : Prop :=
  ∀ l1 l2 : Line, intersect l1 l2 →
    ∀ α : ℝ, α = supplementary_angle α →
      perpendicular l1 l2

def proposition3 : Prop :=
  ∀ l1 l2 : Line, ∀ α β : ℝ,
    alternate_interior_angles l1 l2 α β → α = β →
      perpendicular (angle_bisector l1 α) (angle_bisector l2 β)

def proposition4 : Prop :=
  ∀ l1 l2 : Line, ∀ α β : ℝ,
    same_side_interior_angles l1 l2 α β → complementary α β →
      perpendicular (angle_bisector l1 α) (angle_bisector l2 β)

-- The main theorem
theorem exactly_two_correct_propositions :
  (proposition1 = False ∧
   proposition2 = True ∧
   proposition3 = False ∧
   proposition4 = True) :=
sorry

end NUMINAMATH_CALUDE_exactly_two_correct_propositions_l405_40539


namespace NUMINAMATH_CALUDE_game_total_score_l405_40568

/-- Represents the scores of three teams in a soccer game -/
structure GameScores where
  teamA_first : ℕ
  teamB_first : ℕ
  teamC_first : ℕ
  teamA_second : ℕ
  teamB_second : ℕ
  teamC_second : ℕ

/-- Calculates the total score of all teams -/
def totalScore (scores : GameScores) : ℕ :=
  scores.teamA_first + scores.teamB_first + scores.teamC_first +
  scores.teamA_second + scores.teamB_second + scores.teamC_second

/-- Theorem stating the total score of the game -/
theorem game_total_score :
  ∀ (scores : GameScores),
  scores.teamA_first = 8 →
  scores.teamB_first = scores.teamA_first / 2 →
  scores.teamC_first = 2 * scores.teamB_first →
  scores.teamA_second = scores.teamC_first →
  scores.teamB_second = scores.teamA_first →
  scores.teamC_second = scores.teamB_second + 3 →
  totalScore scores = 47 := by
  sorry


end NUMINAMATH_CALUDE_game_total_score_l405_40568


namespace NUMINAMATH_CALUDE_unripe_oranges_count_l405_40521

/-- The number of sacks of ripe oranges harvested per day -/
def ripe_oranges : ℕ := 44

/-- The difference between the number of sacks of ripe and unripe oranges harvested per day -/
def difference : ℕ := 19

/-- The number of sacks of unripe oranges harvested per day -/
def unripe_oranges : ℕ := ripe_oranges - difference

theorem unripe_oranges_count : unripe_oranges = 25 := by
  sorry

end NUMINAMATH_CALUDE_unripe_oranges_count_l405_40521


namespace NUMINAMATH_CALUDE_journey_distance_l405_40515

/-- Prove that the total distance traveled is 300 km given the specified conditions -/
theorem journey_distance (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ)
  (h_total_time : total_time = 11)
  (h_speed1 : speed1 = 30)
  (h_speed2 : speed2 = 25)
  (h_half_distance : ∀ d : ℝ, d / speed1 + d / speed2 = total_time → d = 300) :
  ∃ d : ℝ, d = 300 ∧ d / (2 * speed1) + d / (2 * speed2) = total_time :=
by sorry

end NUMINAMATH_CALUDE_journey_distance_l405_40515


namespace NUMINAMATH_CALUDE_masha_can_pay_with_five_ruble_coins_l405_40519

theorem masha_can_pay_with_five_ruble_coins 
  (p c n : ℕ+) 
  (h : 2 * p.val + c.val + 7 * n.val = 100) : 
  5 ∣ (p.val + 3 * c.val + n.val) := by
  sorry

end NUMINAMATH_CALUDE_masha_can_pay_with_five_ruble_coins_l405_40519


namespace NUMINAMATH_CALUDE_gmat_scores_l405_40575

theorem gmat_scores (x y z : ℝ) (h1 : x - y = 1/3) (h2 : z = (x + y) / 2) :
  y = x - 1/3 ∧ z = x - 1/6 := by
  sorry

end NUMINAMATH_CALUDE_gmat_scores_l405_40575


namespace NUMINAMATH_CALUDE_binomial_coefficient_16_10_l405_40590

theorem binomial_coefficient_16_10 :
  (Nat.choose 15 8 = 6435) →
  (Nat.choose 15 9 = 5005) →
  (Nat.choose 17 10 = 19448) →
  Nat.choose 16 10 = 8008 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_16_10_l405_40590


namespace NUMINAMATH_CALUDE_little_john_money_l405_40533

/-- Little John's money problem -/
theorem little_john_money (sweet_cost : ℚ) (friend_gift : ℚ) (num_friends : ℕ) (money_left : ℚ) 
  (h1 : sweet_cost = 105/100)
  (h2 : friend_gift = 1)
  (h3 : num_friends = 2)
  (h4 : money_left = 205/100) :
  sweet_cost + num_friends * friend_gift + money_left = 51/10 := by
  sorry

end NUMINAMATH_CALUDE_little_john_money_l405_40533


namespace NUMINAMATH_CALUDE_max_value_of_fraction_l405_40536

theorem max_value_of_fraction (x y z : ℕ) : 
  (10 ≤ x ∧ x ≤ 99) → 
  (10 ≤ y ∧ y ≤ 99) → 
  (10 ≤ z ∧ z ≤ 99) → 
  ((x + y + z) / 3 = 60) → 
  ((x + y) / z ≤ 17) ∧ 
  (∃ (a b c : ℕ), (10 ≤ a ∧ a ≤ 99) ∧ 
                  (10 ≤ b ∧ b ≤ 99) ∧ 
                  (10 ≤ c ∧ c ≤ 99) ∧ 
                  ((a + b + c) / 3 = 60) ∧ 
                  ((a + b) / c = 17)) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_fraction_l405_40536


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l405_40511

theorem repeating_decimal_sum (c d : ℕ) : 
  (5 : ℚ) / 13 = 0.1 * (10 * c + d : ℚ) / 99 → c + d = 11 := by
sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l405_40511


namespace NUMINAMATH_CALUDE_intersection_condition_longest_chord_l405_40552

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := 4 * x^2 + y^2 = 1

-- Define the line
def line (x y m : ℝ) : Prop := y = x + m

-- Theorem 1: Intersection condition
theorem intersection_condition (m : ℝ) :
  (∃ x y : ℝ, ellipse x y ∧ line x y m) ↔ -Real.sqrt 5 / 2 ≤ m ∧ m ≤ Real.sqrt 5 / 2 :=
sorry

-- Theorem 2: Longest chord
theorem longest_chord :
  ∃ x y : ℝ, ellipse x y ∧ line x y 0 ∧
  ∀ m x' y' : ℝ, ellipse x' y' ∧ line x' y' m →
    (x - y)^2 ≥ (x' - y')^2 :=
sorry

end NUMINAMATH_CALUDE_intersection_condition_longest_chord_l405_40552


namespace NUMINAMATH_CALUDE_polynomial_equality_l405_40502

/-- Given a polynomial Q(x) such that Q(x) = Q(0) + Q(1)x + Q(3)x^2 and Q(-2) = 2,
    prove that Q(x) = -3x^2 - (1/2)x + 3 -/
theorem polynomial_equality (Q : ℝ → ℝ) 
    (h1 : ∀ x, Q x = Q 0 + Q 1 * x + Q 3 * x^2)
    (h2 : Q (-2) = 2) :
    ∀ x, Q x = -3 * x^2 - (1/2) * x + 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l405_40502


namespace NUMINAMATH_CALUDE_final_sale_price_is_correct_l405_40525

/-- Calculates the final sale price of an item given its original price and the pricing rules --/
def finalSalePrice (originalPrice : ℝ) : ℝ :=
  let increasedPrice := originalPrice * 1.3
  let priceWithCharge := increasedPrice + 5
  let salePrice := priceWithCharge * 0.75
  salePrice

/-- Theorem stating that the final sale price of an item originally priced at $40 is $42.75 --/
theorem final_sale_price_is_correct : finalSalePrice 40 = 42.75 := by
  sorry

end NUMINAMATH_CALUDE_final_sale_price_is_correct_l405_40525


namespace NUMINAMATH_CALUDE_quadratic_monotonicity_l405_40507

/-- A function f is monotonic on an interval [a,b] if it is either
    non-decreasing or non-increasing on that interval. -/
def IsMonotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y) ∨
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f y ≤ f x)

theorem quadratic_monotonicity (m : ℝ) :
  let f := fun (x : ℝ) ↦ -2 * x^2 + m * x + 1
  IsMonotonic f (-1) 4 ↔ m ∈ Set.Iic (-4) ∪ Set.Ici 16 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_monotonicity_l405_40507


namespace NUMINAMATH_CALUDE_cube_equation_solution_l405_40530

theorem cube_equation_solution (a e : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 45 * e) : e = 49 := by
  sorry

end NUMINAMATH_CALUDE_cube_equation_solution_l405_40530


namespace NUMINAMATH_CALUDE_roxy_garden_plants_l405_40547

def garden_problem (initial_flowering : ℕ) (initial_fruiting_multiplier : ℕ)
  (bought_flowering : ℕ) (bought_fruiting : ℕ)
  (given_flowering : ℕ) (given_fruiting : ℕ) : ℕ :=
  let initial_fruiting := initial_flowering * initial_fruiting_multiplier
  let after_buying_flowering := initial_flowering + bought_flowering
  let after_buying_fruiting := initial_fruiting + bought_fruiting
  let final_flowering := after_buying_flowering - given_flowering
  let final_fruiting := after_buying_fruiting - given_fruiting
  final_flowering + final_fruiting

theorem roxy_garden_plants :
  garden_problem 7 2 3 2 1 4 = 21 := by
  sorry

end NUMINAMATH_CALUDE_roxy_garden_plants_l405_40547


namespace NUMINAMATH_CALUDE_point_on_curve_iff_function_zero_l405_40586

variable (f : ℝ × ℝ → ℝ)
variable (x₀ y₀ : ℝ)

theorem point_on_curve_iff_function_zero :
  f (x₀, y₀) = 0 ↔ (x₀, y₀) ∈ {p : ℝ × ℝ | f p = 0} := by sorry

end NUMINAMATH_CALUDE_point_on_curve_iff_function_zero_l405_40586


namespace NUMINAMATH_CALUDE_largest_n_for_factorization_l405_40588

/-- 
Given a quadratic expression 3x^2 + nx + 54, this theorem states that 163 is the largest 
value of n for which the expression can be factored as the product of two linear factors 
with integer coefficients.
-/
theorem largest_n_for_factorization : 
  ∀ n : ℤ, (∃ a b c d : ℤ, 3*x^2 + n*x + 54 = (a*x + b) * (c*x + d)) → n ≤ 163 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_factorization_l405_40588


namespace NUMINAMATH_CALUDE_intersection_of_symmetric_lines_l405_40531

/-- Two lines that are symmetric about the x-axis -/
structure SymmetricLines where
  k : ℝ
  b : ℝ
  l₁ : ℝ → ℝ := fun x ↦ k * x + 2
  l₂ : ℝ → ℝ := fun x ↦ -x + b
  symmetric : l₁ 0 = -l₂ 0

/-- The intersection point of two symmetric lines is (-2, 0) -/
theorem intersection_of_symmetric_lines (lines : SymmetricLines) :
  ∃ x y, lines.l₁ x = lines.l₂ x ∧ x = -2 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_symmetric_lines_l405_40531


namespace NUMINAMATH_CALUDE_hawk_pregnancies_l405_40579

theorem hawk_pregnancies (num_kettles : ℕ) (babies_per_pregnancy : ℕ) 
  (survival_rate : ℚ) (total_expected_babies : ℕ) :
  num_kettles = 6 →
  babies_per_pregnancy = 4 →
  survival_rate = 3/4 →
  total_expected_babies = 270 →
  (total_expected_babies : ℚ) / (num_kettles * babies_per_pregnancy * survival_rate) = 15 := by
  sorry

end NUMINAMATH_CALUDE_hawk_pregnancies_l405_40579


namespace NUMINAMATH_CALUDE_area_of_triangle_l405_40559

-- Define the triangle ABC and point P
structure Triangle :=
  (A B C P : ℝ × ℝ)

-- Define the properties of the triangle
def isRightTriangle (t : Triangle) : Prop :=
  -- Right angle at B
  sorry

def pointOnHypotenuse (t : Triangle) : Prop :=
  -- P is on AC
  sorry

def angleABP (t : Triangle) : ℝ :=
  -- Angle ABP in radians
  sorry

def lengthAP (t : Triangle) : ℝ :=
  -- Length of AP
  sorry

def lengthCP (t : Triangle) : ℝ :=
  -- Length of CP
  sorry

def areaABC (t : Triangle) : ℝ :=
  -- Area of triangle ABC
  sorry

-- Theorem statement
theorem area_of_triangle (t : Triangle) 
  (h1 : isRightTriangle t)
  (h2 : pointOnHypotenuse t)
  (h3 : angleABP t = π / 6)  -- 30° in radians
  (h4 : lengthAP t = 2)
  (h5 : lengthCP t = 1) :
  areaABC t = 9 / 5 :=
by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_l405_40559


namespace NUMINAMATH_CALUDE_power_of_256_l405_40524

theorem power_of_256 : (256 : ℝ) ^ (5/4 : ℝ) = 1024 :=
by
  have h : 256 = 2^8 := by sorry
  sorry

end NUMINAMATH_CALUDE_power_of_256_l405_40524


namespace NUMINAMATH_CALUDE_negation_equivalence_l405_40578

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x > 0 → (x - 2) / x ≥ 0) ↔ (∃ x : ℝ, x > 0 ∧ 0 ≤ x ∧ x < 2) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l405_40578


namespace NUMINAMATH_CALUDE_satisfactory_fraction_is_four_fifths_l405_40542

/-- Represents the distribution of grades in a classroom --/
structure GradeDistribution where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  f : ℕ

/-- Calculates the fraction of satisfactory grades --/
def satisfactoryFraction (g : GradeDistribution) : ℚ :=
  let satisfactory := g.a + g.b + g.c + g.d
  let total := satisfactory + g.f
  satisfactory / total

/-- Theorem stating that for the given grade distribution, 
    the fraction of satisfactory grades is 4/5 --/
theorem satisfactory_fraction_is_four_fifths :
  let g : GradeDistribution := ⟨8, 7, 5, 4, 6⟩
  satisfactoryFraction g = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_satisfactory_fraction_is_four_fifths_l405_40542


namespace NUMINAMATH_CALUDE_parabola_point_to_directrix_distance_l405_40503

/-- Proves that for a parabola y² = 2px containing the point (1, √5), 
    the distance from this point to the directrix is 9/4 -/
theorem parabola_point_to_directrix_distance :
  ∀ (p : ℝ), 
  (5 : ℝ) = 2 * p →  -- Condition from y² = 2px with (1, √5)
  (1 : ℝ) + p / 2 = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_to_directrix_distance_l405_40503


namespace NUMINAMATH_CALUDE_sum_of_distances_l405_40509

theorem sum_of_distances (saham_distance mother_distance : ℝ) 
  (h1 : saham_distance = 2.6)
  (h2 : mother_distance = 5.98) :
  saham_distance + mother_distance = 8.58 := by
sorry

end NUMINAMATH_CALUDE_sum_of_distances_l405_40509


namespace NUMINAMATH_CALUDE_min_balls_for_fifteen_colors_l405_40562

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  black : Nat
  purple : Nat

/-- The minimum number of balls needed to guarantee at least n balls of a single color -/
def minBallsForColor (counts : BallCounts) (n : Nat) : Nat :=
  sorry

theorem min_balls_for_fifteen_colors (counts : BallCounts) 
  (h_red : counts.red = 35)
  (h_green : counts.green = 18)
  (h_yellow : counts.yellow = 15)
  (h_blue : counts.blue = 17)
  (h_white : counts.white = 12)
  (h_black : counts.black = 12)
  (h_purple : counts.purple = 8) :
  minBallsForColor counts 15 = 89 := by
  sorry

end NUMINAMATH_CALUDE_min_balls_for_fifteen_colors_l405_40562


namespace NUMINAMATH_CALUDE_tree_height_problem_l405_40597

theorem tree_height_problem (h₁ h₂ : ℝ) : 
  h₁ = h₂ + 24 →  -- One tree is 24 feet taller than the other
  h₂ / h₁ = 2 / 3 →  -- The heights are in the ratio 2:3
  h₁ = 72 :=  -- The height of the taller tree is 72 feet
by
  sorry

end NUMINAMATH_CALUDE_tree_height_problem_l405_40597


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l405_40561

theorem arithmetic_calculation : (7.356 - 1.092) + 3.5 = 9.764 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l405_40561


namespace NUMINAMATH_CALUDE_isosceles_triangle_proof_l405_40537

/-- Represents the sides of an isosceles triangle --/
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ

/-- Checks if the given sides form a valid triangle --/
def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem isosceles_triangle_proof :
  let rope_length : ℝ := 20
  let triangle1 : IsoscelesTriangle := { base := 8, leg := 6 }
  let triangle2 : IsoscelesTriangle := { base := 4, leg := 8 }
  
  -- Part 1
  (triangle1.base + 2 * triangle1.leg = rope_length) ∧
  (triangle1.base - triangle1.leg = 2) ∧
  (is_valid_triangle triangle1.base triangle1.leg triangle1.leg) ∧
  
  -- Part 2
  (triangle2.base + 2 * triangle2.leg = rope_length) ∧
  (is_valid_triangle triangle2.base triangle2.leg triangle2.leg) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_proof_l405_40537


namespace NUMINAMATH_CALUDE_honda_cars_sold_l405_40574

/-- Represents the total number of cars sold -/
def total_cars : ℕ := 300

/-- Represents the percentage of Audi cars sold -/
def audi_percent : ℚ := 10 / 100

/-- Represents the percentage of Toyota cars sold -/
def toyota_percent : ℚ := 20 / 100

/-- Represents the percentage of Acura cars sold -/
def acura_percent : ℚ := 30 / 100

/-- Represents the percentage of BMW cars sold -/
def bmw_percent : ℚ := 15 / 100

/-- Theorem stating that the number of Honda cars sold is 75 -/
theorem honda_cars_sold : 
  (total_cars : ℚ) * (1 - (audi_percent + toyota_percent + acura_percent + bmw_percent)) = 75 := by
  sorry

end NUMINAMATH_CALUDE_honda_cars_sold_l405_40574


namespace NUMINAMATH_CALUDE_product_equation_l405_40570

theorem product_equation : 935420 * 625 = 584638125 := by
  sorry

end NUMINAMATH_CALUDE_product_equation_l405_40570


namespace NUMINAMATH_CALUDE_three_equidistant_points_l405_40541

/-- A color type with two possible values -/
inductive Color
| red
| blue

/-- A point on a straight line -/
structure Point where
  x : ℝ

/-- A coloring function that assigns a color to each point on the line -/
def Coloring := Point → Color

/-- The distance between two points -/
def distance (p q : Point) : ℝ := |p.x - q.x|

theorem three_equidistant_points (c : Coloring) :
  ∃ (A B C : Point), c A = c B ∧ c B = c C ∧ distance A B = distance B C :=
sorry

end NUMINAMATH_CALUDE_three_equidistant_points_l405_40541


namespace NUMINAMATH_CALUDE_geometric_progression_identity_l405_40534

theorem geometric_progression_identity 
  (a r : ℝ) (n p k : ℕ) (A B C : ℝ) 
  (hA : A = a * r^(n - 1)) 
  (hB : B = a * r^(p - 1)) 
  (hC : C = a * r^(k - 1)) :
  A^(p - k) * B^(k - n) * C^(n - p) = 1 := by
  sorry


end NUMINAMATH_CALUDE_geometric_progression_identity_l405_40534


namespace NUMINAMATH_CALUDE_difference_of_equal_distinct_prime_factors_l405_40506

def distinctPrimeFactors (n : ℕ) : Finset ℕ :=
  sorry

theorem difference_of_equal_distinct_prime_factors :
  ∀ n : ℕ, ∃ a b : ℕ, n = a - b ∧ (distinctPrimeFactors a).card = (distinctPrimeFactors b).card :=
sorry

end NUMINAMATH_CALUDE_difference_of_equal_distinct_prime_factors_l405_40506


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l405_40582

theorem arithmetic_calculations :
  (24 - |(-2)| + (-16) - 8 = -2) ∧
  ((-2) * (3/2) / (-3/4) * 4 = 16) ∧
  (-1^2016 - (1 - 0.5) / 3 * (2 - (-3)^2) = 1/6) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l405_40582


namespace NUMINAMATH_CALUDE_shifted_line_equation_and_intercept_l405_40517

/-- A line obtained by shifting a direct proportion function -/
structure ShiftedLine where
  k : ℝ
  b : ℝ
  k_neq_zero : k ≠ 0
  passes_through_one_two : k * 1 + b = 2 + 5
  shifted_up_five : b = 5

theorem shifted_line_equation_and_intercept (l : ShiftedLine) :
  (l.k = 2 ∧ l.b = 5) ∧ 
  (∃ (x : ℝ), x = -2.5 ∧ l.k * x + l.b = 0) := by
  sorry

end NUMINAMATH_CALUDE_shifted_line_equation_and_intercept_l405_40517


namespace NUMINAMATH_CALUDE_necessary_not_implies_sufficient_l405_40589

theorem necessary_not_implies_sufficient (A B : Prop) : 
  (A → B) → ¬(∀ A B, (A → B) → (B → A)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_implies_sufficient_l405_40589


namespace NUMINAMATH_CALUDE_intersection_in_second_quadrant_l405_40587

/-- The slope of line l₁ -/
def m₁ : ℚ := 2/3

/-- The y-intercept of line l₁ in terms of a -/
def b₁ (a : ℚ) : ℚ := (1 - a) / 3

/-- The slope of line l₂ -/
def m₂ : ℚ := -1/2

/-- The y-intercept of line l₂ in terms of a -/
def b₂ (a : ℚ) : ℚ := a

/-- The x-coordinate of the intersection point of l₁ and l₂ -/
def x_intersect (a : ℚ) : ℚ := (b₂ a - b₁ a) / (m₁ - m₂)

/-- The y-coordinate of the intersection point of l₁ and l₂ -/
def y_intersect (a : ℚ) : ℚ := m₁ * x_intersect a + b₁ a

/-- The theorem stating the condition for the intersection point to be in the second quadrant -/
theorem intersection_in_second_quadrant (a : ℚ) :
  (x_intersect a > 0 ∧ y_intersect a > 0) ↔ a > 1/4 := by sorry

end NUMINAMATH_CALUDE_intersection_in_second_quadrant_l405_40587


namespace NUMINAMATH_CALUDE_stating_two_students_math_course_l405_40591

/-- The number of students -/
def num_students : ℕ := 4

/-- The number of courses -/
def num_courses : ℕ := 4

/-- The number of students who should choose mathematics -/
def math_students : ℕ := 2

/-- The number of remaining courses after mathematics -/
def remaining_courses : ℕ := 3

/-- Function to calculate the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := 
  Nat.choose n k

/-- 
Theorem stating that the number of ways in which exactly two out of four students 
can choose a mathematics tutoring course, while the other two choose from three 
remaining courses, is equal to 54.
-/
theorem two_students_math_course : 
  (choose num_students math_students) * (remaining_courses^(num_students - math_students)) = 54 := by
  sorry

end NUMINAMATH_CALUDE_stating_two_students_math_course_l405_40591


namespace NUMINAMATH_CALUDE_quadrilateral_qt_length_l405_40560

-- Define the quadrilateral PQRS
structure Quadrilateral :=
  (P Q R S : ℝ × ℝ)

-- Define the conditions
def is_convex (quad : Quadrilateral) : Prop := sorry

def is_perpendicular (A B C D : ℝ × ℝ) : Prop := sorry

def distance (A B : ℝ × ℝ) : ℝ := sorry

def line_through (A B : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

def intersect_point (l₁ l₂ : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

-- State the theorem
theorem quadrilateral_qt_length 
  (quad : Quadrilateral)
  (h_convex : is_convex quad)
  (h_perp_rs_pq : is_perpendicular quad.R quad.S quad.P quad.Q)
  (h_perp_pq_rs : is_perpendicular quad.P quad.Q quad.R quad.S)
  (h_rs_length : distance quad.R quad.S = 39)
  (h_pq_length : distance quad.P quad.Q = 52)
  (h_t : ∃ T : ℝ × ℝ, T ∈ line_through quad.Q (intersect_point (line_through quad.P quad.S) (line_through quad.Q quad.Q)) ∧
                       T = intersect_point (line_through quad.P quad.Q) (line_through quad.Q quad.Q))
  (h_pt_length : ∀ T : ℝ × ℝ, T ∈ line_through quad.P quad.Q → distance quad.P T = 13) :
  ∃ T : ℝ × ℝ, T ∈ line_through quad.P quad.Q ∧ distance quad.Q T = 195 :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_qt_length_l405_40560


namespace NUMINAMATH_CALUDE_equidistant_point_x_coord_l405_40572

/-- The point (x, y) that is equidistant from the x-axis, y-axis, and the line 2x + 3y = 6 -/
def equidistant_point (x y : ℝ) : Prop :=
  let d_x_axis := |y|
  let d_y_axis := |x|
  let d_line := |2*x + 3*y - 6| / Real.sqrt 13
  d_x_axis = d_y_axis ∧ d_x_axis = d_line

/-- The x-coordinate of the equidistant point is 6/5 -/
theorem equidistant_point_x_coord :
  ∃ y : ℝ, equidistant_point (6/5) y :=
sorry

end NUMINAMATH_CALUDE_equidistant_point_x_coord_l405_40572


namespace NUMINAMATH_CALUDE_smallest_numbers_with_percentage_property_l405_40514

theorem smallest_numbers_with_percentage_property :
  ∃ (a b : ℕ), a = 21 ∧ b = 19 ∧
  (∀ (x y : ℕ), (95 * x = 105 * y) → (x ≥ a ∨ y ≥ b)) ∧
  (95 * a = 105 * b) := by
  sorry

end NUMINAMATH_CALUDE_smallest_numbers_with_percentage_property_l405_40514


namespace NUMINAMATH_CALUDE_triangle_solution_l405_40516

/-- Given a triangle with sides a, b, c, angle γ, and circumscribed circle diameter d,
    if a² - b² = 19, γ = 126°52'12", and d = 21.25,
    then a ≈ 10, b ≈ 9, and c ≈ 17 -/
theorem triangle_solution (a b c : ℝ) (γ : Real) (d : ℝ) : 
  a^2 - b^2 = 19 →
  γ = 126 * π / 180 + 52 * π / (180 * 60) + 12 * π / (180 * 60 * 60) →
  d = 21.25 →
  (abs (a - 10) < 0.5 ∧ abs (b - 9) < 0.5 ∧ abs (c - 17) < 0.5) :=
by sorry

end NUMINAMATH_CALUDE_triangle_solution_l405_40516


namespace NUMINAMATH_CALUDE_tournament_result_l405_40549

-- Define the tournament structure
structure Tournament :=
  (teams : Fin 9 → ℕ)  -- Each team's points
  (t1_wins : ℕ)
  (t1_draws : ℕ)
  (t1_losses : ℕ)
  (t9_wins : ℕ)
  (t9_draws : ℕ)
  (t9_losses : ℕ)

-- Define the conditions of the tournament
def valid_tournament (t : Tournament) : Prop :=
  t.teams 0 = 3 * t.t1_wins + t.t1_draws ∧  -- T1's score
  t.teams 8 = t.t9_draws ∧  -- T9's score
  t.t1_wins = 3 ∧ t.t1_draws = 4 ∧ t.t1_losses = 1 ∧
  t.t9_wins = 0 ∧ t.t9_draws = 5 ∧ t.t9_losses = 3 ∧
  (∀ i j, i < j → t.teams i > t.teams j) ∧  -- Strict ordering
  (∀ i, t.teams i ≤ 24)  -- Maximum possible points

-- Define the theorem
theorem tournament_result (t : Tournament) (h : valid_tournament t) :
  (¬ ∃ (t3_defeats_t4 : Bool), t.teams 2 > t.teams 3) ∧
  (∃ (t4_defeats_t3 : Bool), t.teams 3 > t.teams 2) :=
sorry

end NUMINAMATH_CALUDE_tournament_result_l405_40549


namespace NUMINAMATH_CALUDE_exactly_one_inscribed_rhombus_l405_40593

/-- Represents a hyperbola in the xy-plane -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  eq : (x y : ℝ) → Prop

/-- The first hyperbola C₁: x²/a² - y²/b² = 1 -/
def C₁ (a b : ℝ) : Hyperbola :=
  { a := a
    b := b
    eq := fun x y ↦ x^2 / a^2 - y^2 / b^2 = 1 }

/-- The second hyperbola C₂: y²/b² - x²/a² = 1 -/
def C₂ (a b : ℝ) : Hyperbola :=
  { a := a
    b := b
    eq := fun x y ↦ y^2 / b^2 - x^2 / a^2 = 1 }

/-- A predicate indicating whether a hyperbola has an inscribed rhombus -/
def has_inscribed_rhombus (h : Hyperbola) : Prop := sorry

/-- The main theorem stating that exactly one of C₁ or C₂ has an inscribed rhombus -/
theorem exactly_one_inscribed_rhombus (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  (has_inscribed_rhombus (C₁ a b) ∧ ¬has_inscribed_rhombus (C₂ a b)) ∨
  (has_inscribed_rhombus (C₂ a b) ∧ ¬has_inscribed_rhombus (C₁ a b)) :=
sorry

end NUMINAMATH_CALUDE_exactly_one_inscribed_rhombus_l405_40593


namespace NUMINAMATH_CALUDE_karl_drove_517_miles_l405_40557

/-- Represents Karl's car and journey --/
structure KarlsCar where
  miles_per_gallon : ℝ
  tank_capacity : ℝ
  initial_distance : ℝ
  refuel_amount : ℝ
  final_tank_fraction : ℝ

/-- Calculates the total distance Karl drove --/
def total_distance (car : KarlsCar) : ℝ :=
  car.initial_distance + 
  (car.refuel_amount - car.final_tank_fraction * car.tank_capacity) * car.miles_per_gallon

/-- Theorem stating that Karl drove 517 miles --/
theorem karl_drove_517_miles :
  let car : KarlsCar := {
    miles_per_gallon := 25,
    tank_capacity := 16,
    initial_distance := 400,
    refuel_amount := 10,
    final_tank_fraction := 1/3
  }
  total_distance car = 517 := by
  sorry

end NUMINAMATH_CALUDE_karl_drove_517_miles_l405_40557


namespace NUMINAMATH_CALUDE_range_of_a_l405_40529

def proposition_p (a : ℝ) : Prop :=
  ∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0

def proposition_q (a : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 + (a - 1) * x + 1 = 0 ∧ 
             y^2 + (a - 1) * y + 1 = 0 ∧ 
             0 < x ∧ x < 1 ∧ 1 < y ∧ y < 2

theorem range_of_a :
  ∀ a : ℝ, (proposition_p a ∨ proposition_q a) ∧ 
           ¬(proposition_p a ∧ proposition_q a) →
           (a ∈ Set.Ioc (-2) (-3/2) ∪ Set.Icc (-1) 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l405_40529


namespace NUMINAMATH_CALUDE_decimal_division_l405_40528

theorem decimal_division (x y : ℚ) (hx : x = 0.25) (hy : y = 0.005) : x / y = 50 := by
  sorry

end NUMINAMATH_CALUDE_decimal_division_l405_40528


namespace NUMINAMATH_CALUDE_expression_value_l405_40598

theorem expression_value (b : ℚ) (h : b = 1/3) : 
  (3 * b⁻¹ - b⁻¹ / 3) / b^2 = 72 := by sorry

end NUMINAMATH_CALUDE_expression_value_l405_40598


namespace NUMINAMATH_CALUDE_robot_path_lengths_l405_40565

/-- Represents the direction the robot is facing -/
inductive Direction
| North
| East
| South
| West

/-- Represents a point in the plane -/
structure Point where
  x : Int
  y : Int

/-- Represents the state of the robot -/
structure RobotState where
  position : Point
  direction : Direction

/-- The robot's path -/
def RobotPath := List RobotState

/-- Function to check if a path is valid according to the problem conditions -/
def is_valid_path (path : RobotPath) : Bool :=
  sorry

/-- Function to check if a path returns to the starting point -/
def returns_to_start (path : RobotPath) : Bool :=
  sorry

/-- Function to check if a path visits any point more than once -/
def no_revisits (path : RobotPath) : Bool :=
  sorry

/-- Theorem stating the possible path lengths for the robot -/
theorem robot_path_lengths :
  ∀ (n : Nat), 
    (∃ (path : RobotPath), 
      path.length = n ∧ 
      is_valid_path path ∧ 
      returns_to_start path ∧ 
      no_revisits path) ↔ 
    (∃ (k : Nat), n = 4 * k ∧ k ≥ 3) :=
  sorry

end NUMINAMATH_CALUDE_robot_path_lengths_l405_40565


namespace NUMINAMATH_CALUDE_age_difference_l405_40526

theorem age_difference (A B : ℕ) : 
  B = 39 → 
  A + 10 = 2 * (B - 10) → 
  A - B = 9 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l405_40526


namespace NUMINAMATH_CALUDE_complex_purely_imaginary_l405_40512

theorem complex_purely_imaginary (x : ℝ) :
  let z : ℂ := Complex.mk (x^2 + x - 2) (x^2 + 3*x + 2)
  (z.re = 0 ∧ z.im ≠ 0) → x = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_purely_imaginary_l405_40512


namespace NUMINAMATH_CALUDE_even_product_probability_l405_40558

def eight_sided_die := Finset.range 8

theorem even_product_probability :
  let outcomes := eight_sided_die.product eight_sided_die
  (outcomes.filter (fun (x, y) => (x + 1) * (y + 1) % 2 = 0)).card / outcomes.card = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_even_product_probability_l405_40558


namespace NUMINAMATH_CALUDE_kareem_has_largest_number_l405_40567

def jose_final (start : ℕ) : ℕ :=
  ((start - 2) * 4) + 5

def thuy_final (start : ℕ) : ℕ :=
  ((start * 3) - 3) - 4

def kareem_final (start : ℕ) : ℕ :=
  ((start - 3) + 4) * 3

theorem kareem_has_largest_number :
  kareem_final 20 > jose_final 15 ∧ kareem_final 20 > thuy_final 15 :=
by sorry

end NUMINAMATH_CALUDE_kareem_has_largest_number_l405_40567


namespace NUMINAMATH_CALUDE_jeff_bought_seven_one_yuan_socks_l405_40596

/-- Represents the number of sock pairs at each price point -/
structure SockPurchase where
  one_yuan : ℕ
  three_yuan : ℕ
  four_yuan : ℕ

/-- Checks if a SockPurchase satisfies the given conditions -/
def is_valid_purchase (p : SockPurchase) : Prop :=
  p.one_yuan + p.three_yuan + p.four_yuan = 12 ∧
  p.one_yuan * 1 + p.three_yuan * 3 + p.four_yuan * 4 = 24 ∧
  p.one_yuan ≥ 1 ∧ p.three_yuan ≥ 1 ∧ p.four_yuan ≥ 1

/-- The main theorem stating that the only valid purchase has 7 pairs of 1-yuan socks -/
theorem jeff_bought_seven_one_yuan_socks :
  ∀ p : SockPurchase, is_valid_purchase p → p.one_yuan = 7 := by
  sorry

end NUMINAMATH_CALUDE_jeff_bought_seven_one_yuan_socks_l405_40596


namespace NUMINAMATH_CALUDE_value_of_expression_l405_40544

theorem value_of_expression (x : ℝ) (h : x = 4) : 3 * x + 5 = 17 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l405_40544


namespace NUMINAMATH_CALUDE_janet_masud_sibling_ratio_l405_40540

/-- The number of Masud's siblings -/
def masud_siblings : ℕ := 60

/-- The number of Carlos' siblings -/
def carlos_siblings : ℕ := (3 * masud_siblings) / 4

/-- The number of Janet's siblings -/
def janet_siblings : ℕ := carlos_siblings + 135

/-- The ratio of Janet's siblings to Masud's siblings -/
def sibling_ratio : ℚ := janet_siblings / masud_siblings

theorem janet_masud_sibling_ratio :
  sibling_ratio = 3 / 1 := by sorry

end NUMINAMATH_CALUDE_janet_masud_sibling_ratio_l405_40540


namespace NUMINAMATH_CALUDE_ab_leq_one_l405_40555

theorem ab_leq_one (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hab : a + b = 2) : a * b ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ab_leq_one_l405_40555


namespace NUMINAMATH_CALUDE_complex_exponentiation_l405_40518

theorem complex_exponentiation (i : ℂ) (h : i * i = -1) : 
  (1 + i) ^ (2 * i) = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_exponentiation_l405_40518


namespace NUMINAMATH_CALUDE_alternating_power_difference_l405_40577

theorem alternating_power_difference : (-1 : ℤ)^2010 - (-1 : ℤ)^2011 = 2 := by
  sorry

end NUMINAMATH_CALUDE_alternating_power_difference_l405_40577


namespace NUMINAMATH_CALUDE_quadratic_inequality_roots_l405_40535

theorem quadratic_inequality_roots (b : ℝ) : 
  (∀ x, -x^2 + b*x - 12 < 0 ↔ x < 3 ∨ x > 7) → b = 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_roots_l405_40535


namespace NUMINAMATH_CALUDE_paul_clothing_expense_l405_40510

def shirt_price : ℝ := 15
def pants_price : ℝ := 40
def suit_price : ℝ := 150
def sweater_price : ℝ := 30

def num_shirts : ℕ := 4
def num_pants : ℕ := 2
def num_suits : ℕ := 1
def num_sweaters : ℕ := 2

def store_discount : ℝ := 0.2
def coupon_discount : ℝ := 0.1

def total_before_discount : ℝ := 
  shirt_price * num_shirts + 
  pants_price * num_pants + 
  suit_price * num_suits + 
  sweater_price * num_sweaters

def total_after_store_discount : ℝ :=
  total_before_discount * (1 - store_discount)

def final_total : ℝ :=
  total_after_store_discount * (1 - coupon_discount)

theorem paul_clothing_expense : final_total = 252 :=
sorry

end NUMINAMATH_CALUDE_paul_clothing_expense_l405_40510


namespace NUMINAMATH_CALUDE_melanie_has_41_balloons_l405_40556

/-- The number of blue balloons Joan has -/
def joan_balloons : ℕ := 40

/-- The total number of blue balloons -/
def total_balloons : ℕ := 81

/-- The number of blue balloons Melanie has -/
def melanie_balloons : ℕ := total_balloons - joan_balloons

theorem melanie_has_41_balloons : melanie_balloons = 41 := by
  sorry

end NUMINAMATH_CALUDE_melanie_has_41_balloons_l405_40556


namespace NUMINAMATH_CALUDE_triangles_in_decagon_l405_40599

/-- The number of triangles that can be formed using vertices of a regular decagon -/
def trianglesInDecagon : ℕ := 120

/-- A regular decagon has 10 vertices -/
def decagonVertices : ℕ := 10

/-- Theorem: The number of triangles that can be formed using vertices of a regular decagon is 120 -/
theorem triangles_in_decagon :
  (decagonVertices.choose 3) = trianglesInDecagon := by sorry

end NUMINAMATH_CALUDE_triangles_in_decagon_l405_40599


namespace NUMINAMATH_CALUDE_time_to_fill_tank_l405_40554

/-- Represents the tank and pipe system -/
structure TankSystem where
  capacity : ℝ
  pipeA_rate : ℝ
  pipeB_rate : ℝ
  pipeC_rate : ℝ
  pipeA_time : ℝ
  pipeB_time : ℝ
  pipeC_time : ℝ

/-- Calculates the net volume filled in one cycle -/
def netVolumeFilled (system : TankSystem) : ℝ :=
  system.pipeA_rate * system.pipeA_time +
  system.pipeB_rate * system.pipeB_time -
  system.pipeC_rate * system.pipeC_time

/-- Calculates the time for one cycle -/
def cycleTime (system : TankSystem) : ℝ :=
  system.pipeA_time + system.pipeB_time + system.pipeC_time

/-- Theorem stating the time to fill the tank -/
theorem time_to_fill_tank (system : TankSystem)
  (h1 : system.capacity = 2000)
  (h2 : system.pipeA_rate = 200)
  (h3 : system.pipeB_rate = 50)
  (h4 : system.pipeC_rate = 25)
  (h5 : system.pipeA_time = 1)
  (h6 : system.pipeB_time = 2)
  (h7 : system.pipeC_time = 2) :
  (system.capacity / netVolumeFilled system) * cycleTime system = 40 := by
  sorry

end NUMINAMATH_CALUDE_time_to_fill_tank_l405_40554


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_range_l405_40553

theorem quadratic_distinct_roots_range (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 3*x + 2*m = 0 ∧ y^2 - 3*y + 2*m = 0) ↔ m < 9/8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_range_l405_40553


namespace NUMINAMATH_CALUDE_right_angle_vector_coord_l405_40504

/-- Given two vectors OA and OB in a 2D Cartesian coordinate system, 
    if they form a right angle at B, then the y-coordinate of A is 5. -/
theorem right_angle_vector_coord (t : ℝ) : 
  let OA : ℝ × ℝ := (-1, t)
  let OB : ℝ × ℝ := (2, 2)
  let AB : ℝ × ℝ := (OB.1 - OA.1, OB.2 - OA.2)
  (OB.1 * AB.1 + OB.2 * AB.2 = 0) → t = 5 := by
  sorry

end NUMINAMATH_CALUDE_right_angle_vector_coord_l405_40504


namespace NUMINAMATH_CALUDE_triangle_inequality_l405_40522

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l405_40522


namespace NUMINAMATH_CALUDE_complex_number_operations_l405_40551

theorem complex_number_operations (z₁ z₂ : ℂ) 
  (hz₁ : z₁ = 2 - 3*I) 
  (hz₂ : z₂ = (15 - 5*I) / (2 + I^2)) : 
  (z₁ - z₂ = -13 + 2*I) ∧ (z₁ * z₂ = 15 - 55*I) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_operations_l405_40551


namespace NUMINAMATH_CALUDE_tangent_problem_l405_40584

theorem tangent_problem (α β : ℝ) 
  (h1 : Real.tan (α - 2 * β) = 4)
  (h2 : Real.tan β = 2) :
  (Real.tan α - 2) / (1 + 2 * Real.tan α) = -6/7 := by
  sorry

end NUMINAMATH_CALUDE_tangent_problem_l405_40584


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l405_40583

theorem unique_three_digit_number : ∃! n : ℕ,
  100 ≤ n ∧ n < 1000 ∧
  (∃ (π b γ : ℕ),
    π ≠ b ∧ π ≠ γ ∧ b ≠ γ ∧
    0 ≤ π ∧ π ≤ 9 ∧
    0 ≤ b ∧ b ≤ 9 ∧
    0 ≤ γ ∧ γ ≤ 9 ∧
    n = 100 * π + 10 * b + γ ∧
    n = (π + b + γ) * (π + b + γ + 1)) ∧
  n = 156 :=
by sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l405_40583


namespace NUMINAMATH_CALUDE_oil_production_fraction_l405_40527

def initial_concentration : ℝ := 0.02
def first_replacement : ℝ := 0.03
def second_replacement : ℝ := 0.015

theorem oil_production_fraction (x : ℝ) 
  (hx_pos : x > 0)
  (hx_le_one : x ≤ 1)
  (h_first_replacement : initial_concentration * (1 - x) + first_replacement * x = initial_concentration + x * (first_replacement - initial_concentration))
  (h_second_replacement : (initial_concentration + x * (first_replacement - initial_concentration)) * (1 - x) + second_replacement * x = initial_concentration) :
  x = 1/2 := by
sorry

end NUMINAMATH_CALUDE_oil_production_fraction_l405_40527


namespace NUMINAMATH_CALUDE_divisibility_of_power_tower_plus_one_l405_40581

theorem divisibility_of_power_tower_plus_one (a : ℕ) : 
  ∃ n : ℕ, ∀ k : ℕ, a ∣ n^(n^k) + 1 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_power_tower_plus_one_l405_40581


namespace NUMINAMATH_CALUDE_divisor_problem_l405_40523

theorem divisor_problem (x : ℕ) : x > 0 ∧ 83 = 9 * x + 2 → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l405_40523


namespace NUMINAMATH_CALUDE_line_intersection_y_axis_l405_40595

/-- The line passing through points (4, 2) and (6, 14) intersects the y-axis at (0, -22) -/
theorem line_intersection_y_axis :
  let p1 : ℝ × ℝ := (4, 2)
  let p2 : ℝ × ℝ := (6, 14)
  let m : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)
  let b : ℝ := p1.2 - m * p1.1
  let line (x : ℝ) : ℝ := m * x + b
  (0, line 0) = (0, -22) :=
by sorry

end NUMINAMATH_CALUDE_line_intersection_y_axis_l405_40595


namespace NUMINAMATH_CALUDE_platter_size_is_26_l405_40501

/-- Represents the quantity of each type of fruit --/
structure FruitQuantities where
  greenApples : ℕ
  redApples : ℕ
  yellowApples : ℕ
  redOranges : ℕ
  yellowOranges : ℕ
  greenKiwis : ℕ
  purpleGrapes : ℕ
  greenGrapes : ℕ

/-- Represents the desired ratio of apples in the platter --/
structure AppleRatio where
  green : ℕ
  red : ℕ
  yellow : ℕ

/-- Calculates the total number of fruits in the platter --/
def calculatePlatterSize (initialQuantities : FruitQuantities) (appleRatio : AppleRatio) : ℕ :=
  let greenApples := appleRatio.green
  let redApples := appleRatio.red
  let yellowApples := appleRatio.yellow
  let redOranges := 1
  let yellowOranges := 2
  let kiwisAndGrapes := min initialQuantities.greenKiwis initialQuantities.purpleGrapes
  greenApples + redApples + yellowApples + redOranges + yellowOranges + 2 * kiwisAndGrapes

theorem platter_size_is_26 (initialQuantities : FruitQuantities) (appleRatio : AppleRatio) :
  initialQuantities.greenApples = 2 →
  initialQuantities.redApples = 3 →
  initialQuantities.yellowApples = 14 →
  initialQuantities.redOranges = 4 →
  initialQuantities.yellowOranges = 8 →
  initialQuantities.greenKiwis = 10 →
  initialQuantities.purpleGrapes = 7 →
  initialQuantities.greenGrapes = 5 →
  appleRatio.green = 2 →
  appleRatio.red = 4 →
  appleRatio.yellow = 3 →
  calculatePlatterSize initialQuantities appleRatio = 26 := by
  sorry


end NUMINAMATH_CALUDE_platter_size_is_26_l405_40501


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_product_product_of_means_2_8_l405_40571

theorem arithmetic_geometric_mean_product (x y : ℝ) (x_pos : 0 < x) (y_pos : 0 < y) :
  let arithmetic_mean := (x + y) / 2
  let geometric_mean := Real.sqrt (x * y)
  arithmetic_mean * geometric_mean = (x + y) * Real.sqrt (x * y) / 2 :=
by sorry

theorem product_of_means_2_8 :
  let arithmetic_mean := (2 + 8) / 2
  let geometric_mean := Real.sqrt (2 * 8)
  (arithmetic_mean * geometric_mean = 20 ∨ arithmetic_mean * geometric_mean = -20) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_product_product_of_means_2_8_l405_40571


namespace NUMINAMATH_CALUDE_rectangle_measurement_error_l405_40545

theorem rectangle_measurement_error (L W : ℝ) (x : ℝ) 
  (h1 : L > 0) (h2 : W > 0) (h3 : x > 0) :
  L * (1 + x / 100) * W * 0.95 = L * W * 1.045 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_measurement_error_l405_40545


namespace NUMINAMATH_CALUDE_fraction_equality_l405_40576

theorem fraction_equality (a b c d : ℝ) (hb : b ≠ 0) (hd : d ≠ 0) 
  (h1 : (a / b)^2 = (c / d)^2) (h2 : a * c < 0) : 
  a / b = -(c / d) := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l405_40576


namespace NUMINAMATH_CALUDE_fixed_point_parabola_l405_40550

/-- The fixed point theorem for a parabola -/
theorem fixed_point_parabola 
  (p a b : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : b ≠ 0) 
  (h3 : b^2 ≠ 2*p*a) :
  ∃ C : ℝ × ℝ, 
    ∀ (M M₁ M₂ : ℝ × ℝ),
      (M.2)^2 = 2*p*M.1 →  -- M is on the parabola
      (M₁.2)^2 = 2*p*M₁.1 →  -- M₁ is on the parabola
      (M₂.2)^2 = 2*p*M₂.1 →  -- M₂ is on the parabola
      M₁ ≠ M →
      M₂ ≠ M →
      M₁ ≠ M₂ →
      (∃ t : ℝ, M₁.2 - b = t * (M₁.1 - a)) →  -- M₁ is on line AM
      (∃ t : ℝ, M₂.2 = t * (M₂.1 + a)) →  -- M₂ is on line BM
      (∃ t : ℝ, M₂.2 - M₁.2 = t * (M₂.1 - M₁.1)) →  -- M₁M₂ is a line
      C = (a, 2*p*a/b) ∧ 
      ∃ t : ℝ, C.2 - M₁.2 = t * (C.1 - M₁.1)  -- C is on line M₁M₂
  := by sorry

end NUMINAMATH_CALUDE_fixed_point_parabola_l405_40550


namespace NUMINAMATH_CALUDE_ellipse_properties_l405_40594

-- Define the ellipse C
def ellipse_C (x y a : ℝ) : Prop :=
  x^2 / a^2 + y^2 / (7 - a^2) = 1 ∧ a > 0

-- Define the eccentricity
def eccentricity (a : ℝ) : ℝ := 2

-- Define the standard form of the ellipse
def standard_ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

-- Define a line passing through (4,0)
def line_through_R (x y k : ℝ) : Prop :=
  y = k * (x - 4)

-- Define a point on the ellipse
def point_on_ellipse (x y : ℝ) : Prop :=
  standard_ellipse x y

-- Define a perpendicular line to x-axis
def perpendicular_to_x (x y x₁ : ℝ) : Prop :=
  x = x₁

-- Define the right focus of the ellipse
def right_focus : ℝ × ℝ := (1, 0)

-- State the theorem
theorem ellipse_properties (a x y x₁ y₁ x₂ y₂ k : ℝ) :
  ellipse_C x y a →
  eccentricity a = 2 →
  line_through_R x y k →
  point_on_ellipse x₁ y₁ →
  point_on_ellipse x₂ y₂ →
  perpendicular_to_x x y x₁ →
  point_on_ellipse x₁ (-y₁) →
  (∀ x y, ellipse_C x y a ↔ standard_ellipse x y) ∧
  (∃ t, t * (x₁, -y₁) + (1 - t) * right_focus = (x₂, y₂)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l405_40594


namespace NUMINAMATH_CALUDE_car_distance_proof_l405_40563

/-- Proves that the initial distance covered by a car is 180 km, given the conditions of the problem. -/
theorem car_distance_proof (initial_time : ℝ) (new_speed : ℝ) :
  initial_time = 6 →
  new_speed = 20 →
  ∃ (D : ℝ),
    D = new_speed * (3/2 * initial_time) ∧
    D = 180 :=
by
  sorry

#check car_distance_proof

end NUMINAMATH_CALUDE_car_distance_proof_l405_40563


namespace NUMINAMATH_CALUDE_matrix_determinant_sixteen_l405_40585

def matrix (x : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![3*x, 2; x, 4*x]

theorem matrix_determinant_sixteen (x : ℝ) : 
  Matrix.det (matrix x) = 16 ↔ x = 4/3 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_matrix_determinant_sixteen_l405_40585


namespace NUMINAMATH_CALUDE_problem_solution_l405_40538

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m - 4 ≤ x ∧ x ≤ 3 * m + 2}

theorem problem_solution :
  (∀ m : ℝ, A ∪ B m = B m → m ∈ Set.Icc 1 2) ∧
  (∀ m : ℝ, A ∩ B m = B m → m < -3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l405_40538


namespace NUMINAMATH_CALUDE_cats_adoption_proof_l405_40548

def adopt_cats (initial_cats : ℕ) (added_cats : ℕ) (cats_per_adopter : ℕ) (final_cats : ℕ) : ℕ :=
  ((initial_cats + added_cats) - final_cats) / cats_per_adopter

theorem cats_adoption_proof :
  adopt_cats 20 3 2 17 = 3 := by
  sorry

end NUMINAMATH_CALUDE_cats_adoption_proof_l405_40548


namespace NUMINAMATH_CALUDE_fifteenth_even_multiple_of_four_l405_40543

-- Define a function that represents the nth positive integer that is both even and a multiple of 4
def evenMultipleOfFour (n : ℕ) : ℕ := 4 * n

-- State the theorem
theorem fifteenth_even_multiple_of_four : evenMultipleOfFour 15 = 60 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_even_multiple_of_four_l405_40543


namespace NUMINAMATH_CALUDE_vector_equation_l405_40573

def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (1, -1)
def c : ℝ × ℝ := (-2, 4)

theorem vector_equation : c = a - 3 • b := by sorry

end NUMINAMATH_CALUDE_vector_equation_l405_40573


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l405_40546

/-- Prove that in a geometric sequence with first term 1 and fourth term 64, the common ratio is 4 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- Geometric sequence condition
  a 1 = 1 →                     -- First term is 1
  a 4 = 64 →                    -- Fourth term is 64
  q = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l405_40546


namespace NUMINAMATH_CALUDE_ali_baba_cave_theorem_l405_40580

/-- Represents the state of a barrel (herring head up or down) -/
inductive BarrelState
| Up
| Down

/-- Represents a configuration of n barrels -/
def Configuration (n : ℕ) := Fin n → BarrelState

/-- Represents a move by Ali Baba -/
def Move (n : ℕ) := Fin n → Bool

/-- Apply a move to a configuration -/
def applyMove (n : ℕ) (config : Configuration n) (move : Move n) : Configuration n :=
  fun i => if move i then match config i with
    | BarrelState.Up => BarrelState.Down
    | BarrelState.Down => BarrelState.Up
  else config i

/-- Check if all barrels are in the same state -/
def allSameState (n : ℕ) (config : Configuration n) : Prop :=
  (∀ i : Fin n, config i = BarrelState.Up) ∨ (∀ i : Fin n, config i = BarrelState.Down)

/-- Ali Baba can win in a finite number of moves -/
def canWin (n : ℕ) : Prop :=
  ∃ (strategy : ℕ → Move n), ∀ (initialConfig : Configuration n),
    ∃ (k : ℕ), allSameState n (Nat.rec initialConfig (fun i config => applyMove n config (strategy i)) k)

/-- n is a power of 2 -/
def isPowerOfTwo (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

theorem ali_baba_cave_theorem (n : ℕ) :
  canWin n ↔ isPowerOfTwo n :=
sorry

end NUMINAMATH_CALUDE_ali_baba_cave_theorem_l405_40580


namespace NUMINAMATH_CALUDE_function_periodicity_l405_40569

def periodic_function (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f (x + c) = f x

theorem function_periodicity 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, |f x| ≤ 1)
  (h2 : ∀ x, f (x + 13/42) + f x = f (x + 1/6) + f (x + 1/7)) :
  ∃ c > 0, periodic_function f c ∧ c = 1 :=
sorry

end NUMINAMATH_CALUDE_function_periodicity_l405_40569


namespace NUMINAMATH_CALUDE_last_digit_of_power_of_two_plus_one_l405_40500

theorem last_digit_of_power_of_two_plus_one (n : ℕ) (h : n ≥ 2) :
  (2^(2^n) + 1) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_power_of_two_plus_one_l405_40500


namespace NUMINAMATH_CALUDE_max_trailing_zeros_consecutive_two_digit_numbers_l405_40532

/-- Two-digit number type -/
def TwoDigitNumber := {n : ℕ // 10 ≤ n ∧ n ≤ 99}

/-- Function to count trailing zeros of a natural number -/
def countTrailingZeros (n : ℕ) : ℕ := sorry

/-- Theorem: The maximum number of consecutive zeros at the end of the product 
    of two consecutive two-digit numbers is 2 -/
theorem max_trailing_zeros_consecutive_two_digit_numbers : 
  ∃ (a : TwoDigitNumber), 
    let b : TwoDigitNumber := ⟨a.val + 1, sorry⟩
    countTrailingZeros (a.val * b.val) = 2 ∧ 
    ∀ (x : TwoDigitNumber), 
      let y : TwoDigitNumber := ⟨x.val + 1, sorry⟩
      countTrailingZeros (x.val * y.val) ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_max_trailing_zeros_consecutive_two_digit_numbers_l405_40532


namespace NUMINAMATH_CALUDE_f_max_min_on_interval_l405_40513

def f (x : ℝ) := 4 * x - x^3

theorem f_max_min_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (0 : ℝ) 2, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (0 : ℝ) 2, f x = max) ∧
    (∀ x ∈ Set.Icc (0 : ℝ) 2, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (0 : ℝ) 2, f x = min) ∧
    max = 16 * Real.sqrt 3 / 9 ∧
    min = 0 :=
by sorry

end NUMINAMATH_CALUDE_f_max_min_on_interval_l405_40513


namespace NUMINAMATH_CALUDE_greatest_integer_problem_l405_40520

theorem greatest_integer_problem : 
  ∃ (n : ℕ), n < 150 ∧ 
  (∃ (k l : ℕ), n = 9 * k - 2 ∧ n = 6 * l - 4) ∧
  (∀ (m : ℕ), m < 150 → 
    (∃ (k' l' : ℕ), m = 9 * k' - 2 ∧ m = 6 * l' - 4) → 
    m ≤ n) ∧
  n = 146 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_problem_l405_40520


namespace NUMINAMATH_CALUDE_snake_eating_time_l405_40505

/-- Represents the number of weeks it takes for a snake to eat one mouse. -/
def weeks_per_mouse (mice_per_decade : ℕ) : ℚ :=
  (10 * 52) / mice_per_decade

/-- Proves that a snake eating 130 mice in a decade takes 4 weeks to eat one mouse. -/
theorem snake_eating_time : weeks_per_mouse 130 = 4 := by
  sorry

end NUMINAMATH_CALUDE_snake_eating_time_l405_40505


namespace NUMINAMATH_CALUDE_problem_solution_l405_40508

def f (k : ℝ) (x : ℝ) : ℝ := |3*x - 1| + |3*x + k|
def g (x : ℝ) : ℝ := x + 4

theorem problem_solution :
  (∀ x : ℝ, f (-3) x ≥ 4 ↔ (x ≤ 0 ∨ x ≥ 4/3)) ∧
  (∀ k : ℝ, k > -1 → 
    (∀ x : ℝ, x ∈ Set.Icc (-k/3) (1/3) → f k x ≤ g x) →
    k ∈ Set.Ioo (-1) (9/4)) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l405_40508


namespace NUMINAMATH_CALUDE_family_change_is_71_l405_40564

/-- Represents a family member with their age and ticket price. -/
structure FamilyMember where
  age : ℕ
  ticketPrice : ℕ

/-- Calculates the change received after a family visit to an amusement park. -/
def amusementParkChange (family : List FamilyMember) (regularPrice discountAmount paidAmount : ℕ) : ℕ :=
  let totalCost := family.foldl (fun acc member => acc + member.ticketPrice) 0
  paidAmount - totalCost

/-- Theorem: The family receives $71 in change. -/
theorem family_change_is_71 :
  let family : List FamilyMember := [
    { age := 6, ticketPrice := 114 },
    { age := 10, ticketPrice := 114 },
    { age := 13, ticketPrice := 129 },
    { age := 8, ticketPrice := 114 },
    { age := 30, ticketPrice := 129 },  -- Assuming parent age
    { age := 30, ticketPrice := 129 }   -- Assuming parent age
  ]
  let regularPrice := 129
  let discountAmount := 15
  let paidAmount := 800
  amusementParkChange family regularPrice discountAmount paidAmount = 71 := by
sorry

end NUMINAMATH_CALUDE_family_change_is_71_l405_40564
