import Mathlib

namespace NUMINAMATH_CALUDE_totalCost_equals_64_l1215_121587

-- Define the side length of each square
def squareSide : ℝ := 4

-- Define the number of squares
def numSquares : ℕ := 4

-- Define the areas of overlap
def centralOverlap : ℕ := 1
def tripleOverlap : ℕ := 6
def doubleOverlap : ℕ := 12
def singleArea : ℕ := 18

-- Define the cost function
def costFunction (overlappingSquares : ℕ) : ℕ := overlappingSquares

-- Theorem statement
theorem totalCost_equals_64 :
  (centralOverlap * costFunction numSquares) +
  (tripleOverlap * costFunction 3) +
  (doubleOverlap * costFunction 2) +
  (singleArea * costFunction 1) = 64 := by
  sorry

end NUMINAMATH_CALUDE_totalCost_equals_64_l1215_121587


namespace NUMINAMATH_CALUDE_price_two_bracelets_is_eight_l1215_121556

/-- Represents the bracelet selling scenario -/
structure BraceletSale where
  initialStock : ℕ
  singlePrice : ℕ
  singleRevenue : ℕ
  totalRevenue : ℕ

/-- Calculates the price for two bracelets -/
def priceTwoBracelets (sale : BraceletSale) : ℕ :=
  let singleSold := sale.singleRevenue / sale.singlePrice
  let remainingBracelets := sale.initialStock - singleSold
  let pairRevenue := sale.totalRevenue - sale.singleRevenue
  let pairsSold := remainingBracelets / 2
  pairRevenue / pairsSold

/-- Theorem stating that the price for two bracelets is 8 -/
theorem price_two_bracelets_is_eight (sale : BraceletSale) 
  (h1 : sale.initialStock = 30)
  (h2 : sale.singlePrice = 5)
  (h3 : sale.singleRevenue = 60)
  (h4 : sale.totalRevenue = 132) : 
  priceTwoBracelets sale = 8 := by
  sorry

end NUMINAMATH_CALUDE_price_two_bracelets_is_eight_l1215_121556


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l1215_121505

theorem cube_root_equation_solution :
  ∃ x : ℝ, x ≠ 0 ∧ (5 - 1/x)^(1/3 : ℝ) = -6 ↔ x = 1/221 :=
by sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l1215_121505


namespace NUMINAMATH_CALUDE_sqrt_product_plus_one_l1215_121580

theorem sqrt_product_plus_one : 
  Real.sqrt ((21 : ℝ) * 20 * 19 * 18 + 1) = 379 := by sorry

end NUMINAMATH_CALUDE_sqrt_product_plus_one_l1215_121580


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1215_121588

theorem contrapositive_equivalence :
  (∀ x : ℝ, x < 3 → x^2 ≤ 9) ↔ (∀ x : ℝ, x^2 > 9 → x ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l1215_121588


namespace NUMINAMATH_CALUDE_union_with_complement_l1215_121546

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4}

-- Define set A
def A : Set Nat := {1, 2}

-- Define set B
def B : Set Nat := {2, 3}

-- Theorem statement
theorem union_with_complement : A ∪ (U \ B) = {1, 2, 4} := by
  sorry

end NUMINAMATH_CALUDE_union_with_complement_l1215_121546


namespace NUMINAMATH_CALUDE_vegetables_per_week_l1215_121557

theorem vegetables_per_week (total_points : ℕ) (points_per_vegetable : ℕ) 
  (num_students : ℕ) (num_weeks : ℕ) 
  (h1 : total_points = 200)
  (h2 : points_per_vegetable = 2)
  (h3 : num_students = 25)
  (h4 : num_weeks = 2) :
  (total_points / points_per_vegetable / num_students) / num_weeks = 2 :=
by
  sorry

#check vegetables_per_week

end NUMINAMATH_CALUDE_vegetables_per_week_l1215_121557


namespace NUMINAMATH_CALUDE_uniform_motion_parametric_equation_l1215_121527

/-- Parametric equation of a point undergoing uniform linear motion -/
def parametric_equation (initial_x initial_y vx vy : ℝ) : ℝ → ℝ × ℝ :=
  λ t => (initial_x + vx * t, initial_y + vy * t)

/-- The correct parametric equation for the given conditions -/
theorem uniform_motion_parametric_equation :
  parametric_equation 1 1 9 12 = λ t => (1 + 9 * t, 1 + 12 * t) := by
  sorry

end NUMINAMATH_CALUDE_uniform_motion_parametric_equation_l1215_121527


namespace NUMINAMATH_CALUDE_range_of_a_l1215_121586

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc (-2) 3, 2 * x > x^2 + a) → a < -8 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1215_121586


namespace NUMINAMATH_CALUDE_fraction_comparison_l1215_121542

theorem fraction_comparison : 
  (111110 : ℚ) / 111111 < (333331 : ℚ) / 333334 ∧ (333331 : ℚ) / 333334 < (222221 : ℚ) / 222223 := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l1215_121542


namespace NUMINAMATH_CALUDE_negation_of_absolute_value_less_than_zero_l1215_121569

theorem negation_of_absolute_value_less_than_zero :
  (¬ ∀ x : ℝ, |x| < 0) ↔ (∃ x₀ : ℝ, |x₀| ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_absolute_value_less_than_zero_l1215_121569


namespace NUMINAMATH_CALUDE_largest_integer_below_sqrt_two_l1215_121528

theorem largest_integer_below_sqrt_two :
  ∀ n : ℕ, n > 0 ∧ n < Real.sqrt 2 → n = 1 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_below_sqrt_two_l1215_121528


namespace NUMINAMATH_CALUDE_motel_room_rate_problem_l1215_121545

theorem motel_room_rate_problem (total_rent : ℕ) (lower_rate : ℕ) (num_rooms_changed : ℕ) (rent_decrease_percent : ℚ) (higher_rate : ℕ) : 
  total_rent = 400 →
  lower_rate = 50 →
  num_rooms_changed = 10 →
  rent_decrease_percent = 1/4 →
  (total_rent : ℚ) * rent_decrease_percent = (num_rooms_changed : ℚ) * (higher_rate - lower_rate) →
  higher_rate = 60 := by
sorry

end NUMINAMATH_CALUDE_motel_room_rate_problem_l1215_121545


namespace NUMINAMATH_CALUDE_intersection_implies_a_equals_two_l1215_121567

def A : Set ℝ := {2, 4}
def B (a : ℝ) : Set ℝ := {a, a^2 + 3}

theorem intersection_implies_a_equals_two (a : ℝ) :
  A ∩ B a = {2} → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_equals_two_l1215_121567


namespace NUMINAMATH_CALUDE_leap_year_53_sundays_probability_l1215_121590

/-- A leap year has 366 days -/
def leapYearDays : ℕ := 366

/-- A week has 7 days -/
def daysInWeek : ℕ := 7

/-- A leap year has 52 complete weeks and 2 extra days -/
def leapYearWeeks : ℕ := 52
def leapYearExtraDays : ℕ := 2

/-- The probability of a randomly selected leap year having 53 Sundays -/
def probLeapYear53Sundays : ℚ := 2 / 7

theorem leap_year_53_sundays_probability :
  probLeapYear53Sundays = 2 / 7 := by sorry

end NUMINAMATH_CALUDE_leap_year_53_sundays_probability_l1215_121590


namespace NUMINAMATH_CALUDE_sum_of_odd_coefficients_l1215_121533

theorem sum_of_odd_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₃ + a₅ = 122 := by
sorry

end NUMINAMATH_CALUDE_sum_of_odd_coefficients_l1215_121533


namespace NUMINAMATH_CALUDE_bricks_needed_for_wall_l1215_121548

/-- Represents the dimensions of a brick -/
structure BrickDimensions where
  length : ℝ
  height : ℝ
  thickness : ℝ

/-- Represents the dimensions of the wall -/
structure WallDimensions where
  baseLength : ℝ
  topLength : ℝ
  height : ℝ
  thickness : ℝ

/-- Calculates the number of bricks needed to build the wall -/
def calculateBricksNeeded (brickDim : BrickDimensions) (wallDim : WallDimensions) (mortarThickness : ℝ) : ℕ :=
  sorry

/-- Theorem stating the number of bricks needed for the given wall -/
theorem bricks_needed_for_wall 
  (brickDim : BrickDimensions)
  (wallDim : WallDimensions)
  (mortarThickness : ℝ)
  (h1 : brickDim.length = 125)
  (h2 : brickDim.height = 11.25)
  (h3 : brickDim.thickness = 6)
  (h4 : wallDim.baseLength = 800)
  (h5 : wallDim.topLength = 650)
  (h6 : wallDim.height = 600)
  (h7 : wallDim.thickness = 22.5)
  (h8 : mortarThickness = 1.25) :
  calculateBricksNeeded brickDim wallDim mortarThickness = 1036 :=
sorry

end NUMINAMATH_CALUDE_bricks_needed_for_wall_l1215_121548


namespace NUMINAMATH_CALUDE_brody_battery_usage_l1215_121599

/-- Represents the battery life of Brody's calculator -/
def BatteryLife : Type := ℚ

/-- The total battery life of the calculator when fully charged (in hours) -/
def full_battery : ℚ := 60

/-- The duration of Brody's exam (in hours) -/
def exam_duration : ℚ := 2

/-- The remaining battery life after the exam (in hours) -/
def remaining_battery : ℚ := 13

/-- The fraction of battery Brody has used up -/
def battery_used_fraction : ℚ := 3/4

/-- Theorem stating that the fraction of battery Brody has used up is 3/4 -/
theorem brody_battery_usage :
  (full_battery - (remaining_battery + exam_duration)) / full_battery = battery_used_fraction := by
  sorry

end NUMINAMATH_CALUDE_brody_battery_usage_l1215_121599


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l1215_121560

theorem smallest_integer_with_remainders : ∃ (x : ℕ), 
  (x > 0) ∧ 
  (x % 5 = 2) ∧ 
  (x % 7 = 3) ∧ 
  (∀ y : ℕ, y > 0 ∧ y % 5 = 2 ∧ y % 7 = 3 → x ≤ y) ∧
  (x = 17) := by
sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l1215_121560


namespace NUMINAMATH_CALUDE_edward_initial_money_l1215_121589

def books_cost : ℕ := 6
def pens_cost : ℕ := 16
def notebook_cost : ℕ := 5
def pencil_case_cost : ℕ := 3
def money_left : ℕ := 19

theorem edward_initial_money :
  books_cost + pens_cost + notebook_cost + pencil_case_cost + money_left = 49 := by
  sorry

end NUMINAMATH_CALUDE_edward_initial_money_l1215_121589


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_250_l1215_121538

theorem closest_integer_to_cube_root_250 : 
  ∀ n : ℤ, |n^3 - 250| ≥ |6^3 - 250| := by sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_250_l1215_121538


namespace NUMINAMATH_CALUDE_total_points_after_perfect_games_l1215_121576

/-- The number of points in a perfect score -/
def perfect_score : ℕ := 21

/-- The number of consecutive perfect games -/
def consecutive_games : ℕ := 3

/-- Theorem: The total points after 3 perfect games is 63 -/
theorem total_points_after_perfect_games :
  perfect_score * consecutive_games = 63 := by
  sorry

end NUMINAMATH_CALUDE_total_points_after_perfect_games_l1215_121576


namespace NUMINAMATH_CALUDE_sisters_candy_count_l1215_121508

theorem sisters_candy_count 
  (debbys_candy : ℕ) 
  (eaten_candy : ℕ) 
  (remaining_candy : ℕ) 
  (h1 : debbys_candy = 32) 
  (h2 : eaten_candy = 35) 
  (h3 : remaining_candy = 39) : 
  ∃ (sisters_candy : ℕ), 
    sisters_candy = 42 ∧ 
    debbys_candy + sisters_candy = eaten_candy + remaining_candy :=
by
  sorry

end NUMINAMATH_CALUDE_sisters_candy_count_l1215_121508


namespace NUMINAMATH_CALUDE_expand_product_l1215_121532

theorem expand_product (x : ℝ) : (x + 3) * (x + 6) = x^2 + 9*x + 18 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l1215_121532


namespace NUMINAMATH_CALUDE_ordering_abc_l1215_121581

theorem ordering_abc : ∃ (a b c : ℝ), 
  a = Real.sqrt 1.2 ∧ 
  b = Real.exp 0.1 ∧ 
  c = 1 + Real.log 1.1 ∧ 
  b > a ∧ a > c := by
  sorry

end NUMINAMATH_CALUDE_ordering_abc_l1215_121581


namespace NUMINAMATH_CALUDE_other_roots_of_polynomial_l1215_121584

def f (a b x : ℝ) : ℝ := x^3 + 4*x^2 + a*x + b

theorem other_roots_of_polynomial (a b : ℚ) :
  (f a b (2 + Real.sqrt 3) = 0) →
  (f a b (2 - Real.sqrt 3) = 0) ∧ (f a b (-8) = 0) :=
by sorry

end NUMINAMATH_CALUDE_other_roots_of_polynomial_l1215_121584


namespace NUMINAMATH_CALUDE_total_dividend_income_l1215_121541

-- Define the investments for each stock
def investment_A : ℕ := 2000
def investment_B : ℕ := 2500
def investment_C : ℕ := 1500
def investment_D : ℕ := 2000
def investment_E : ℕ := 2000

-- Define the dividend yields for each stock for each year
def yield_A : Fin 3 → ℚ
  | 0 => 5/100
  | 1 => 4/100
  | 2 => 3/100

def yield_B : Fin 3 → ℚ
  | 0 => 3/100
  | 1 => 5/100
  | 2 => 4/100

def yield_C : Fin 3 → ℚ
  | 0 => 4/100
  | 1 => 6/100
  | 2 => 4/100

def yield_D : Fin 3 → ℚ
  | 0 => 6/100
  | 1 => 3/100
  | 2 => 5/100

def yield_E : Fin 3 → ℚ
  | 0 => 2/100
  | 1 => 7/100
  | 2 => 6/100

-- Calculate the total dividend income for a single stock over 3 years
def total_dividend (investment : ℕ) (yield : Fin 3 → ℚ) : ℚ :=
  (yield 0 * investment) + (yield 1 * investment) + (yield 2 * investment)

-- Theorem: The total dividend income from all stocks over 3 years is 1330
theorem total_dividend_income :
  total_dividend investment_A yield_A +
  total_dividend investment_B yield_B +
  total_dividend investment_C yield_C +
  total_dividend investment_D yield_D +
  total_dividend investment_E yield_E = 1330 := by
  sorry


end NUMINAMATH_CALUDE_total_dividend_income_l1215_121541


namespace NUMINAMATH_CALUDE_evaluate_expression_l1215_121543

theorem evaluate_expression : 3002^3 - 3001 * 3002^2 - 3001^2 * 3002 + 3001^3 + 1 = 6004 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1215_121543


namespace NUMINAMATH_CALUDE_unfair_coin_probability_l1215_121555

theorem unfair_coin_probability (p : ℝ) : 
  0 < p ∧ p < 1 →
  (6 : ℝ) * p^2 * (1 - p)^2 = (4 : ℝ) * p^3 * (1 - p) →
  p = 3/5 := by
sorry

end NUMINAMATH_CALUDE_unfair_coin_probability_l1215_121555


namespace NUMINAMATH_CALUDE_wall_width_calculation_l1215_121514

/-- Calculates the width of a wall given brick dimensions and wall specifications -/
theorem wall_width_calculation (brick_length brick_width brick_height : ℝ)
  (wall_length wall_height : ℝ) (num_bricks : ℕ) :
  brick_length = 0.2 →
  brick_width = 0.1 →
  brick_height = 0.075 →
  wall_length = 25 →
  wall_height = 2 →
  num_bricks = 25000 →
  ∃ (wall_width : ℝ), wall_width = 0.75 :=
by sorry

end NUMINAMATH_CALUDE_wall_width_calculation_l1215_121514


namespace NUMINAMATH_CALUDE_boat_speed_l1215_121531

/-- Given a boat that travels 11 km/h along a stream and 5 km/h against the same stream,
    its speed in still water is 8 km/h. -/
theorem boat_speed (along_stream : ℝ) (against_stream : ℝ) (still_water : ℝ)
    (h1 : along_stream = 11)
    (h2 : against_stream = 5)
    (h3 : along_stream = still_water + (along_stream - still_water))
    (h4 : against_stream = still_water - (along_stream - still_water)) :
    still_water = 8 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_l1215_121531


namespace NUMINAMATH_CALUDE_elenas_garden_lily_petals_l1215_121582

/-- Proves that each lily has 6 petals given the conditions of Elena's garden --/
theorem elenas_garden_lily_petals :
  ∀ (lily_petals : ℕ),
    (8 * lily_petals + 5 * 3 = 63) →
    lily_petals = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_elenas_garden_lily_petals_l1215_121582


namespace NUMINAMATH_CALUDE_total_combinations_l1215_121515

/-- The number of color choices available for painting the box. -/
def num_colors : ℕ := 4

/-- The number of decoration choices available for the box. -/
def num_decorations : ℕ := 3

/-- The number of painting method choices available. -/
def num_methods : ℕ := 3

/-- Theorem stating the total number of combinations for painting the box. -/
theorem total_combinations : num_colors * num_decorations * num_methods = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_combinations_l1215_121515


namespace NUMINAMATH_CALUDE_largest_sum_l1215_121501

theorem largest_sum : 
  let sum1 := (1/4 : ℚ) + (1/5 : ℚ)
  let sum2 := (1/4 : ℚ) + (1/6 : ℚ)
  let sum3 := (1/4 : ℚ) + (1/3 : ℚ)
  let sum4 := (1/4 : ℚ) + (1/8 : ℚ)
  let sum5 := (1/4 : ℚ) + (1/7 : ℚ)
  sum3 > sum1 ∧ sum3 > sum2 ∧ sum3 > sum4 ∧ sum3 > sum5 := by
  sorry

end NUMINAMATH_CALUDE_largest_sum_l1215_121501


namespace NUMINAMATH_CALUDE_tan_x_minus_pi_sixth_l1215_121519

theorem tan_x_minus_pi_sixth (x : ℝ) 
  (h : Real.sin (π / 3 - x) = (1 / 2) * Real.cos (x - π / 2)) : 
  Real.tan (x - π / 6) = Real.sqrt 3 / 9 := by
  sorry

end NUMINAMATH_CALUDE_tan_x_minus_pi_sixth_l1215_121519


namespace NUMINAMATH_CALUDE_vector_projection_l1215_121529

/-- Given two vectors a and e in a real inner product space, 
    where |a| = 4, e is a unit vector, and the angle between a and e is 2π/3,
    prove that the projection of a + e on a - e is 5√21 / 7 -/
theorem vector_projection (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (a e : V) (h1 : ‖a‖ = 4) (h2 : ‖e‖ = 1) 
  (h3 : Real.cos (Real.arccos (inner a e / (‖a‖ * ‖e‖))) = Real.cos (2 * Real.pi / 3)) :
  ‖a + e‖ * (inner (a + e) (a - e) / (‖a + e‖ * ‖a - e‖)) = 5 * Real.sqrt 21 / 7 := by
  sorry

end NUMINAMATH_CALUDE_vector_projection_l1215_121529


namespace NUMINAMATH_CALUDE_nyusha_ate_28_candies_l1215_121547

-- Define the number of candies eaten by each person
variable (K E N B : ℕ)

-- Define the conditions
axiom total_candies : K + E + N + B = 86
axiom minimum_candies : K ≥ 5 ∧ E ≥ 5 ∧ N ≥ 5 ∧ B ≥ 5
axiom nyusha_ate_most : N > K ∧ N > E ∧ N > B
axiom kros_yozhik_total : K + E = 53

-- Theorem to prove
theorem nyusha_ate_28_candies : N = 28 := by
  sorry


end NUMINAMATH_CALUDE_nyusha_ate_28_candies_l1215_121547


namespace NUMINAMATH_CALUDE_sum_of_coordinates_X_l1215_121500

/-- Given points Y and Z, and the condition that XZ/XY = ZY/XY = 1/3,
    prove that the sum of the coordinates of point X is 10. -/
theorem sum_of_coordinates_X (Y Z X : ℝ × ℝ) : 
  Y = (2, 8) →
  Z = (0, -4) →
  (X.1 - Z.1) / (X.1 - Y.1) = 1/3 →
  (X.2 - Z.2) / (X.2 - Y.2) = 1/3 →
  (Z.1 - Y.1) / (X.1 - Y.1) = 1/3 →
  (Z.2 - Y.2) / (X.2 - Y.2) = 1/3 →
  X.1 + X.2 = 10 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_X_l1215_121500


namespace NUMINAMATH_CALUDE_joanie_wants_three_cups_l1215_121518

-- Define the relationship between tablespoons of kernels and cups of popcorn
def kernels_to_popcorn (tablespoons : ℕ) : ℕ := 2 * tablespoons

-- Define the amount of popcorn each person wants
def mitchell_popcorn : ℕ := 4
def miles_davis_popcorn : ℕ := 6
def cliff_popcorn : ℕ := 3

-- Define the total amount of kernels needed
def total_kernels : ℕ := 8

-- Define Joanie's popcorn amount
def joanie_popcorn : ℕ := kernels_to_popcorn total_kernels - (mitchell_popcorn + miles_davis_popcorn + cliff_popcorn)

-- Theorem statement
theorem joanie_wants_three_cups :
  joanie_popcorn = 3 := by sorry

end NUMINAMATH_CALUDE_joanie_wants_three_cups_l1215_121518


namespace NUMINAMATH_CALUDE_points_collinear_l1215_121522

/-- Prove that points A(-1, -2), B(2, -1), and C(8, 1) are collinear. -/
theorem points_collinear : 
  let A : ℝ × ℝ := (-1, -2)
  let B : ℝ × ℝ := (2, -1)
  let C : ℝ × ℝ := (8, 1)
  ∃ (t : ℝ), C - A = t • (B - A) :=
by sorry

end NUMINAMATH_CALUDE_points_collinear_l1215_121522


namespace NUMINAMATH_CALUDE_werewolf_identity_l1215_121585

/-- Represents a forest dweller -/
inductive Dweller
| A
| B
| C

/-- Represents the status of a dweller -/
structure Status where
  is_werewolf : Bool
  is_knight : Bool

/-- The statement made by B -/
def b_statement (status : Dweller → Status) : Prop :=
  (status Dweller.C).is_werewolf

theorem werewolf_identity (status : Dweller → Status) :
  (∃! d : Dweller, (status d).is_werewolf ∧ (status d).is_knight) →
  (∀ d : Dweller, d ≠ Dweller.A → d ≠ Dweller.B → ¬(status d).is_knight) →
  b_statement status →
  (status Dweller.A).is_werewolf := by
  sorry

end NUMINAMATH_CALUDE_werewolf_identity_l1215_121585


namespace NUMINAMATH_CALUDE_count_satisfying_integers_is_five_l1215_121554

/-- The count of positive integers n satisfying (n + 1050) / 90 = ⌊√n⌋ -/
def count_satisfying_integers : ℕ := 5

/-- Predicate defining when a positive integer satisfies the equation -/
def satisfies_equation (n : ℕ+) : Prop :=
  (n + 1050) / 90 = ⌊Real.sqrt n⌋

/-- Theorem stating that exactly 5 positive integers satisfy the equation -/
theorem count_satisfying_integers_is_five :
  (∃! (S : Finset ℕ+), S.card = count_satisfying_integers ∧ 
    ∀ n, n ∈ S ↔ satisfies_equation n) :=
by sorry

end NUMINAMATH_CALUDE_count_satisfying_integers_is_five_l1215_121554


namespace NUMINAMATH_CALUDE_cow_spots_l1215_121539

theorem cow_spots (left_spots : ℕ) : 
  (left_spots + (3 * left_spots + 7) = 71) → left_spots = 16 := by
  sorry

end NUMINAMATH_CALUDE_cow_spots_l1215_121539


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l1215_121578

/-- 
For a quadratic equation x^2 - x + n = 0, if it has two equal real roots,
then n = 1/4.
-/
theorem equal_roots_quadratic (n : ℝ) : 
  (∃ x : ℝ, x^2 - x + n = 0 ∧ (∀ y : ℝ, y^2 - y + n = 0 → y = x)) → n = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l1215_121578


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_first_four_prime_reciprocals_l1215_121506

-- Define the first four prime numbers
def first_four_primes : List Nat := [2, 3, 5, 7]

-- Define a function to calculate the arithmetic mean of reciprocals
def arithmetic_mean_of_reciprocals (numbers : List Nat) : ℚ :=
  let reciprocals := numbers.map (λ x => (1 : ℚ) / x)
  reciprocals.sum / numbers.length

-- Theorem statement
theorem arithmetic_mean_of_first_four_prime_reciprocals :
  arithmetic_mean_of_reciprocals first_four_primes = 247 / 840 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_first_four_prime_reciprocals_l1215_121506


namespace NUMINAMATH_CALUDE_average_weight_e_f_l1215_121536

theorem average_weight_e_f (d e f : ℝ) 
  (h1 : (d + e + f) / 3 = 42)
  (h2 : (d + e) / 2 = 35)
  (h3 : e = 26) :
  (e + f) / 2 = 41 := by
sorry

end NUMINAMATH_CALUDE_average_weight_e_f_l1215_121536


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l1215_121570

theorem profit_percentage_calculation (C S : ℝ) (h1 : C > 0) (h2 : S > 0) :
  20 * C = 16 * S →
  (S - C) / C * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l1215_121570


namespace NUMINAMATH_CALUDE_line_y_axis_intersection_l1215_121577

/-- A line passing through two given points intersects the y-axis at a specific point -/
theorem line_y_axis_intersection (x₁ y₁ x₂ y₂ : ℝ) :
  x₁ = 4 ∧ y₁ = 20 ∧ x₂ = -6 ∧ y₂ = -2 →
  ∃ (y : ℝ), y = 11.2 ∧ 
    (y - y₁) / (0 - x₁) = (y₂ - y₁) / (x₂ - x₁) :=
by sorry

end NUMINAMATH_CALUDE_line_y_axis_intersection_l1215_121577


namespace NUMINAMATH_CALUDE_prob_less_than_three_heads_in_eight_flips_prob_less_than_three_heads_in_eight_flips_proof_l1215_121568

/-- The probability of getting fewer than 3 heads in 8 fair coin flips -/
theorem prob_less_than_three_heads_in_eight_flips : ℚ :=
  37 / 256

/-- Proof that the probability of getting fewer than 3 heads in 8 fair coin flips is 37/256 -/
theorem prob_less_than_three_heads_in_eight_flips_proof :
  prob_less_than_three_heads_in_eight_flips = 37 / 256 := by
  sorry


end NUMINAMATH_CALUDE_prob_less_than_three_heads_in_eight_flips_prob_less_than_three_heads_in_eight_flips_proof_l1215_121568


namespace NUMINAMATH_CALUDE_fundamental_theorem_of_calculus_l1215_121551

open Set
open Interval
open MeasureTheory
open Real

-- Define the theorem
theorem fundamental_theorem_of_calculus 
  (f : ℝ → ℝ) (f' : ℝ → ℝ) (a b : ℝ) 
  (h1 : ContinuousOn f (Icc a b))
  (h2 : DifferentiableOn ℝ f (Ioc a b))
  (h3 : ∀ x ∈ Ioc a b, deriv f x = f' x) :
  ∫ x in a..b, f' x = f b - f a :=
sorry

end NUMINAMATH_CALUDE_fundamental_theorem_of_calculus_l1215_121551


namespace NUMINAMATH_CALUDE_game_score_product_l1215_121572

def g (n : ℕ) : ℕ :=
  if n % 2 = 0 ∧ n % 3 = 0 then 6
  else if n % 3 = 0 then 3
  else if n % 2 = 0 then 2
  else 0

def allie_rolls : List ℕ := [6, 3, 2, 4]
def betty_rolls : List ℕ := [5, 2, 3, 6]

theorem game_score_product : 
  (allie_rolls.map g).sum * (betty_rolls.map g).sum = 143 := by
  sorry

end NUMINAMATH_CALUDE_game_score_product_l1215_121572


namespace NUMINAMATH_CALUDE_kite_profit_theorem_l1215_121520

/-- Cost price of type B kite -/
def cost_B : ℝ := 80

/-- Cost price of type A kite -/
def cost_A : ℝ := cost_B + 20

/-- Selling price of type B kite -/
def sell_B : ℝ := 120

/-- Selling price of type A kite -/
def sell_A (m : ℝ) : ℝ := 2 * (130 - m)

/-- Total number of kites -/
def total_kites : ℕ := 300

/-- Profit function -/
def profit (m : ℝ) : ℝ := (sell_A m - cost_A) * m + (sell_B - cost_B) * (total_kites - m)

/-- Theorem stating the cost prices and maximum profit -/
theorem kite_profit_theorem :
  (∀ m : ℝ, 50 ≤ m → m ≤ 150 → profit m ≤ 13000) ∧
  (20000 / cost_A = 2 * (8000 / cost_B)) ∧
  (profit 50 = 13000) := by sorry

end NUMINAMATH_CALUDE_kite_profit_theorem_l1215_121520


namespace NUMINAMATH_CALUDE_factorization_example_l1215_121594

theorem factorization_example : ∀ x : ℝ, x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_example_l1215_121594


namespace NUMINAMATH_CALUDE_pretzel_price_is_two_l1215_121563

/-- Represents the revenue and quantity information for a candy store --/
structure CandyStore where
  fudgePounds : ℕ
  fudgePrice : ℚ
  trufflesDozens : ℕ
  trufflePrice : ℚ
  pretzelsDozens : ℕ
  totalRevenue : ℚ

/-- Calculates the price of each chocolate-covered pretzel --/
def pretzelPrice (store : CandyStore) : ℚ :=
  let fudgeRevenue := store.fudgePounds * store.fudgePrice
  let trufflesRevenue := store.trufflesDozens * 12 * store.trufflePrice
  let pretzelsRevenue := store.totalRevenue - fudgeRevenue - trufflesRevenue
  let pretzelsCount := store.pretzelsDozens * 12
  pretzelsRevenue / pretzelsCount

/-- Theorem stating that the price of each chocolate-covered pretzel is $2 --/
theorem pretzel_price_is_two (store : CandyStore)
  (h1 : store.fudgePounds = 20)
  (h2 : store.fudgePrice = 5/2)
  (h3 : store.trufflesDozens = 5)
  (h4 : store.trufflePrice = 3/2)
  (h5 : store.pretzelsDozens = 3)
  (h6 : store.totalRevenue = 212) :
  pretzelPrice store = 2 := by
  sorry


end NUMINAMATH_CALUDE_pretzel_price_is_two_l1215_121563


namespace NUMINAMATH_CALUDE_T_10_mod_5_l1215_121512

def T : ℕ → ℕ
  | 0 => 0
  | 1 => 2
  | (n+2) =>
    let c₁ := T n
    let c₂ := T (n+1)
    c₁ + c₂

theorem T_10_mod_5 : T 10 % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_T_10_mod_5_l1215_121512


namespace NUMINAMATH_CALUDE_expression_evaluation_l1215_121509

theorem expression_evaluation (a b : ℤ) (h1 : a = 3) (h2 : b = 2) :
  (a^3 + b)^2 - (a^3 - b)^2 = 216 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1215_121509


namespace NUMINAMATH_CALUDE_compound_interest_rate_interest_rate_calculation_l1215_121550

/-- Compound interest calculation -/
theorem compound_interest_rate (P : ℝ) (A : ℝ) (n : ℝ) (t : ℝ) (h1 : P > 0) (h2 : A > P) (h3 : n > 0) (h4 : t > 0) :
  A = P * (1 + 0.2 / n) ^ (n * t) → 
  A - P = 240 ∧ P = 1200 ∧ n = 1 ∧ t = 1 :=
by sorry

/-- Main theorem: Interest rate calculation -/
theorem interest_rate_calculation (P : ℝ) (A : ℝ) (n : ℝ) (t : ℝ) 
  (h1 : P > 0) (h2 : A > P) (h3 : n > 0) (h4 : t > 0) 
  (h5 : A - P = 240) (h6 : P = 1200) (h7 : n = 1) (h8 : t = 1) :
  ∃ r : ℝ, A = P * (1 + r) ∧ r = 0.2 :=
by sorry

end NUMINAMATH_CALUDE_compound_interest_rate_interest_rate_calculation_l1215_121550


namespace NUMINAMATH_CALUDE_max_tickets_purchasable_l1215_121511

def ticket_price : ℚ := 15.75
def processing_fee : ℚ := 1.25
def budget : ℚ := 150.00

theorem max_tickets_purchasable :
  ∀ n : ℕ, n * (ticket_price + processing_fee) ≤ budget ↔ n ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_max_tickets_purchasable_l1215_121511


namespace NUMINAMATH_CALUDE_harmonic_mean_closest_to_ten_l1215_121558

theorem harmonic_mean_closest_to_ten :
  let a := 5
  let b := 2023
  let harmonic_mean := 2 * a * b / (a + b)
  ∀ n : ℤ, n ≠ 10 → |harmonic_mean - 10| < |harmonic_mean - n| :=
by sorry

end NUMINAMATH_CALUDE_harmonic_mean_closest_to_ten_l1215_121558


namespace NUMINAMATH_CALUDE_abs_neg_five_l1215_121565

theorem abs_neg_five : |(-5 : ℤ)| = 5 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_five_l1215_121565


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1215_121562

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b, a > b ∧ b > 0 → a^2 > b^2) ∧
  (∃ a b, a^2 > b^2 ∧ ¬(a > b ∧ b > 0)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1215_121562


namespace NUMINAMATH_CALUDE_car_mpg_difference_l1215_121510

/-- Proves that the difference between highway and city miles per gallon is 9 --/
theorem car_mpg_difference (highway_miles : ℕ) (city_miles : ℕ) (city_mpg : ℕ) :
  highway_miles = 462 →
  city_miles = 336 →
  city_mpg = 24 →
  (highway_miles / (city_miles / city_mpg)) - city_mpg = 9 := by
  sorry

#check car_mpg_difference

end NUMINAMATH_CALUDE_car_mpg_difference_l1215_121510


namespace NUMINAMATH_CALUDE_dot_product_problem_l1215_121540

/-- Given vectors a and b in ℝ², prove that the dot product of (2a + b) and a is 6. -/
theorem dot_product_problem (a b : ℝ × ℝ) (h1 : a = (2, -1)) (h2 : b = (-1, 2)) :
  (2 • a + b) • a = 6 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_problem_l1215_121540


namespace NUMINAMATH_CALUDE_expand_binomials_l1215_121571

theorem expand_binomials (x : ℝ) : (2*x + 3) * (4*x - 7) = 8*x^2 - 2*x - 21 := by
  sorry

end NUMINAMATH_CALUDE_expand_binomials_l1215_121571


namespace NUMINAMATH_CALUDE_bus_driver_worked_48_hours_l1215_121566

/-- Represents the pay structure and hours worked by a bus driver --/
structure BusDriverPay where
  regularRate : ℝ
  regularHours : ℝ
  overtimeRate : ℝ
  overtimeHours : ℝ
  totalCompensation : ℝ

/-- Calculates the total hours worked by the bus driver --/
def totalHours (pay : BusDriverPay) : ℝ :=
  pay.regularHours + pay.overtimeHours

/-- Theorem stating that under the given conditions, the bus driver worked 48 hours --/
theorem bus_driver_worked_48_hours :
  ∀ (pay : BusDriverPay),
    pay.regularRate = 16 →
    pay.regularHours = 40 →
    pay.overtimeRate = pay.regularRate * 1.75 →
    pay.totalCompensation = 864 →
    pay.regularHours * pay.regularRate + pay.overtimeHours * pay.overtimeRate = pay.totalCompensation →
    totalHours pay = 48 := by
  sorry

end NUMINAMATH_CALUDE_bus_driver_worked_48_hours_l1215_121566


namespace NUMINAMATH_CALUDE_remainder_425421_div_12_l1215_121513

theorem remainder_425421_div_12 : 425421 % 12 = 9 := by
  sorry

end NUMINAMATH_CALUDE_remainder_425421_div_12_l1215_121513


namespace NUMINAMATH_CALUDE_common_root_of_quadratic_equations_l1215_121544

theorem common_root_of_quadratic_equations (x : ℚ) :
  (6 * x^2 + 5 * x - 1 = 0) ∧ (18 * x^2 + 41 * x - 7 = 0) → x = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_common_root_of_quadratic_equations_l1215_121544


namespace NUMINAMATH_CALUDE_rectangle_shorter_side_length_l1215_121596

/-- Given a rectangle, square, and equilateral triangle with the same perimeter,
    if the square's side length is 9 cm, the rectangle's shorter side is 6 cm. -/
theorem rectangle_shorter_side_length
  (rectangle : Real × Real)
  (square : Real)
  (equilateral_triangle : Real)
  (h1 : 2 * (rectangle.1 + rectangle.2) = 4 * square)
  (h2 : 2 * (rectangle.1 + rectangle.2) = 3 * equilateral_triangle)
  (h3 : square = 9) :
  min rectangle.1 rectangle.2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_shorter_side_length_l1215_121596


namespace NUMINAMATH_CALUDE_michael_truck_meet_once_l1215_121530

-- Define the constants
def michael_speed : ℝ := 6
def truck_speed : ℝ := 12
def bench_distance : ℝ := 180
def truck_stop_time : ℝ := 40

-- Define the positions of Michael and the truck as functions of time
def michael_position (t : ℝ) : ℝ := michael_speed * t

-- The truck's position is more complex due to stops, so we'll define it as a noncomputable function
noncomputable def truck_position (t : ℝ) : ℝ := 
  let cycle_time := bench_distance / truck_speed + truck_stop_time
  let full_cycles := ⌊t / cycle_time⌋
  let remaining_time := t - full_cycles * cycle_time
  bench_distance * (full_cycles + 1) + 
    if remaining_time ≤ bench_distance / truck_speed 
    then truck_speed * remaining_time
    else bench_distance

-- Define the theorem
theorem michael_truck_meet_once :
  ∃! t : ℝ, t > 0 ∧ michael_position t = truck_position t :=
sorry


end NUMINAMATH_CALUDE_michael_truck_meet_once_l1215_121530


namespace NUMINAMATH_CALUDE_sin_sum_specific_angles_l1215_121523

theorem sin_sum_specific_angles (θ φ : ℝ) :
  Complex.exp (θ * Complex.I) = (4 / 5 : ℂ) + (3 / 5 : ℂ) * Complex.I ∧
  Complex.exp (φ * Complex.I) = -(5 / 13 : ℂ) - (12 / 13 : ℂ) * Complex.I →
  Real.sin (θ + φ) = -(63 / 65) := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_specific_angles_l1215_121523


namespace NUMINAMATH_CALUDE_three_circles_tangency_theorem_l1215_121574

-- Define the structure for a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the structure for a point
structure Point where
  x : ℝ
  y : ℝ

-- Define the function to check if two circles are tangent
def are_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

-- Define the function to get the tangency point of two circles
def tangency_point (c1 c2 : Circle) : Point :=
  sorry

-- Define the function to check if a point is on a circle
def point_on_circle (p : Point) (c : Circle) : Prop :=
  let (x, y) := c.center
  (p.x - x)^2 + (p.y - y)^2 = c.radius^2

-- Define the function to check if two points form a diameter of a circle
def is_diameter (p1 p2 : Point) (c : Circle) : Prop :=
  let (x, y) := c.center
  (p1.x + p2.x) / 2 = x ∧ (p1.y + p2.y) / 2 = y

-- Theorem statement
theorem three_circles_tangency_theorem (S1 S2 S3 : Circle) :
  are_tangent S1 S2 ∧ are_tangent S2 S3 ∧ are_tangent S3 S1 →
  let C := tangency_point S1 S2
  let A := tangency_point S2 S3
  let B := tangency_point S3 S1
  let A1 := sorry -- Intersection of line CA with S3
  let B1 := sorry -- Intersection of line CB with S3
  point_on_circle A1 S3 ∧ point_on_circle B1 S3 ∧ is_diameter A1 B1 S3 :=
sorry

end NUMINAMATH_CALUDE_three_circles_tangency_theorem_l1215_121574


namespace NUMINAMATH_CALUDE_min_value_case1_min_value_case2_l1215_121553

/-- The function f(x) defined as x^2 + |x-a| + 1 -/
def f (a x : ℝ) : ℝ := x^2 + |x-a| + 1

/-- The minimum value of f(x) when a ≤ -1/2 and x ≥ a -/
theorem min_value_case1 (a : ℝ) (h : a ≤ -1/2) :
  ∀ x ≥ a, f a x ≥ 3/4 - a :=
sorry

/-- The minimum value of f(x) when a > -1/2 and x ≥ a -/
theorem min_value_case2 (a : ℝ) (h : a > -1/2) :
  ∀ x ≥ a, f a x ≥ a^2 + 1 :=
sorry

end NUMINAMATH_CALUDE_min_value_case1_min_value_case2_l1215_121553


namespace NUMINAMATH_CALUDE_horseshoe_selling_price_l1215_121559

/-- Proves that the selling price per set of horseshoes is $50 given the specified conditions. -/
theorem horseshoe_selling_price
  (initial_outlay : ℕ)
  (cost_per_set : ℕ)
  (num_sets : ℕ)
  (profit : ℕ)
  (h1 : initial_outlay = 10000)
  (h2 : cost_per_set = 20)
  (h3 : num_sets = 500)
  (h4 : profit = 5000) :
  ∃ (selling_price : ℕ),
    selling_price * num_sets = initial_outlay + cost_per_set * num_sets + profit ∧
    selling_price = 50 :=
by sorry

end NUMINAMATH_CALUDE_horseshoe_selling_price_l1215_121559


namespace NUMINAMATH_CALUDE_greeting_cards_group_size_l1215_121595

theorem greeting_cards_group_size (n : ℕ) : 
  n * (n - 1) = 72 → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_greeting_cards_group_size_l1215_121595


namespace NUMINAMATH_CALUDE_arcsin_one_half_equals_pi_sixth_l1215_121579

theorem arcsin_one_half_equals_pi_sixth : Real.arcsin (1/2) = π/6 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_one_half_equals_pi_sixth_l1215_121579


namespace NUMINAMATH_CALUDE_prob_at_least_7_heads_theorem_l1215_121552

/-- A fair coin is flipped 10 times. -/
def total_flips : ℕ := 10

/-- The number of heads required for the event. -/
def min_heads : ℕ := 7

/-- The number of fixed heads at the end. -/
def fixed_heads : ℕ := 2

/-- The probability of getting heads on a single flip of a fair coin. -/
def prob_heads : ℚ := 1/2

/-- The probability of getting at least 7 heads in 10 flips, given that the last two are heads. -/
def prob_at_least_7_heads_given_last_2_heads : ℚ := 93/256

/-- 
Theorem: The probability of getting at least 7 heads in 10 flips of a fair coin, 
given that the last two flips are heads, is equal to 93/256.
-/
theorem prob_at_least_7_heads_theorem : 
  prob_at_least_7_heads_given_last_2_heads = 93/256 :=
sorry

end NUMINAMATH_CALUDE_prob_at_least_7_heads_theorem_l1215_121552


namespace NUMINAMATH_CALUDE_special_vector_exists_l1215_121524

/-- Define a new operation * for 2D vectors -/
def vec_mult (m n : Fin 2 → ℝ) : Fin 2 → ℝ := 
  λ i => if i = 0 then m 0 * n 0 + m 1 * n 1 else m 0 * n 1 + m 1 * n 0

/-- Theorem: If m * p = m for all m, then p = (1, 0) -/
theorem special_vector_exists :
  ∃ p : Fin 2 → ℝ, (∀ m : Fin 2 → ℝ, vec_mult m p = m) → p 0 = 1 ∧ p 1 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_special_vector_exists_l1215_121524


namespace NUMINAMATH_CALUDE_bike_distance_l1215_121525

/-- Proves that the distance covered by a bike is 88 miles given the conditions -/
theorem bike_distance (time : ℝ) (truck_distance : ℝ) (speed_difference : ℝ) : 
  time = 8 → 
  truck_distance = 112 → 
  speed_difference = 3 → 
  (truck_distance / time - speed_difference) * time = 88 := by
  sorry

#check bike_distance

end NUMINAMATH_CALUDE_bike_distance_l1215_121525


namespace NUMINAMATH_CALUDE_fraction_equality_l1215_121521

theorem fraction_equality (p q s u : ℚ) 
  (h1 : p / q = 5 / 6) 
  (h2 : s / u = 7 / 15) : 
  (5 * p * s - 3 * q * u) / (6 * q * u - 5 * p * s) = -19 / 73 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1215_121521


namespace NUMINAMATH_CALUDE_lines_parallel_if_planes_parallel_and_coplanar_l1215_121591

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the necessary relations
variable (subset : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (line_parallel : Line → Line → Prop)
variable (coplanar : Line → Line → Prop)

-- State the theorem
theorem lines_parallel_if_planes_parallel_and_coplanar
  (m n : Line) (α β : Plane)
  (h_diff_lines : m ≠ n)
  (h_diff_planes : α ≠ β)
  (h_m_in_α : subset m α)
  (h_n_in_β : subset n β)
  (h_planes_parallel : parallel α β)
  (h_coplanar : coplanar m n) :
  line_parallel m n :=
sorry

end NUMINAMATH_CALUDE_lines_parallel_if_planes_parallel_and_coplanar_l1215_121591


namespace NUMINAMATH_CALUDE_proportion_solution_l1215_121537

theorem proportion_solution (x : ℝ) : (0.75 / x = 5 / 7) → x = 1.05 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l1215_121537


namespace NUMINAMATH_CALUDE_compare_sqrt_expressions_l1215_121516

theorem compare_sqrt_expressions : 3 * Real.sqrt 5 > 2 * Real.sqrt 11 := by sorry

end NUMINAMATH_CALUDE_compare_sqrt_expressions_l1215_121516


namespace NUMINAMATH_CALUDE_meeting_percentage_is_37_5_l1215_121573

/-- Represents the duration of a work day in minutes -/
def work_day_minutes : ℕ := 8 * 60

/-- Represents the duration of the first meeting in minutes -/
def first_meeting_minutes : ℕ := 45

/-- Calculates the duration of the second meeting in minutes -/
def second_meeting_minutes : ℕ := 3 * first_meeting_minutes

/-- Calculates the total time spent in meetings in minutes -/
def total_meeting_minutes : ℕ := first_meeting_minutes + second_meeting_minutes

/-- Represents the percentage of work day spent in meetings -/
def meeting_percentage : ℚ := (total_meeting_minutes : ℚ) / (work_day_minutes : ℚ) * 100

theorem meeting_percentage_is_37_5 : meeting_percentage = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_meeting_percentage_is_37_5_l1215_121573


namespace NUMINAMATH_CALUDE_hospital_patient_distribution_l1215_121504

/-- Represents the number of patients each doctor takes care of in a hospital -/
def patients_per_doctor (total_patients : ℕ) (total_doctors : ℕ) : ℕ :=
  total_patients / total_doctors

/-- Theorem stating that given 400 patients and 16 doctors, each doctor takes care of 25 patients -/
theorem hospital_patient_distribution :
  patients_per_doctor 400 16 = 25 := by
  sorry

end NUMINAMATH_CALUDE_hospital_patient_distribution_l1215_121504


namespace NUMINAMATH_CALUDE_arc_length_of_sector_l1215_121534

/-- Given a circle with radius 4 cm and a sector with an area of 7 square centimeters,
    the length of the arc forming this sector is 3.5 cm. -/
theorem arc_length_of_sector (r : ℝ) (area : ℝ) (arc_length : ℝ) : 
  r = 4 → area = 7 → arc_length = (area * 2) / r → arc_length = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_arc_length_of_sector_l1215_121534


namespace NUMINAMATH_CALUDE_partnership_profit_l1215_121507

/-- Represents the profit share of a partner in a business partnership. -/
structure ProfitShare where
  investment : ℕ
  share : ℕ

/-- Calculates the total profit of a partnership business given the investments and one partner's profit share. -/
def totalProfit (a b c : ProfitShare) : ℕ :=
  sorry

/-- Theorem stating that given the specific investments and A's profit share, the total profit is 12500. -/
theorem partnership_profit (a b c : ProfitShare) 
  (ha : a.investment = 6300)
  (hb : b.investment = 4200)
  (hc : c.investment = 10500)
  (ha_share : a.share = 3750) :
  totalProfit a b c = 12500 := by
  sorry

end NUMINAMATH_CALUDE_partnership_profit_l1215_121507


namespace NUMINAMATH_CALUDE_range_of_expression_l1215_121517

theorem range_of_expression (x y a b : ℝ) : 
  x ≥ 1 →
  y ≥ 2 →
  x + y ≤ 4 →
  2*a + b ≥ 1 →
  3*a - b ≥ 2 →
  5*a ≤ 4 →
  (b + 2) / (a - 1) ≥ -12 ∧ (b + 2) / (a - 1) ≤ -9/2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_expression_l1215_121517


namespace NUMINAMATH_CALUDE_shoe_percentage_gain_l1215_121598

/-- Prove that the percentage gain on the selling price of a shoe is approximately 16.67% -/
theorem shoe_percentage_gain :
  let manufacturing_cost : ℝ := 210
  let transportation_cost_per_100 : ℝ := 500
  let selling_price : ℝ := 258
  let total_cost : ℝ := manufacturing_cost + transportation_cost_per_100 / 100
  let gain : ℝ := selling_price - total_cost
  let percentage_gain : ℝ := gain / selling_price * 100
  ∃ ε > 0, abs (percentage_gain - 16.67) < ε :=
by sorry

end NUMINAMATH_CALUDE_shoe_percentage_gain_l1215_121598


namespace NUMINAMATH_CALUDE_quadrilateral_side_length_l1215_121593

/-- Given a quadrilateral ABCD with specific side lengths and angles, prove that AD = √7 -/
theorem quadrilateral_side_length (A B C D : ℝ × ℝ) : 
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let CD := Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)
  let AD := Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2)
  let angle_ABC := Real.arccos ((AB^2 + BC^2 - (A.1 - C.1)^2 - (A.2 - C.2)^2) / (2 * AB * BC))
  let angle_BCD := Real.arccos ((BC^2 + CD^2 - (B.1 - D.1)^2 - (B.2 - D.2)^2) / (2 * BC * CD))
  AB = 1 ∧ BC = 2 ∧ CD = Real.sqrt 3 ∧ angle_ABC = 2 * Real.pi / 3 ∧ angle_BCD = Real.pi / 2 →
  AD = Real.sqrt 7 := by
sorry


end NUMINAMATH_CALUDE_quadrilateral_side_length_l1215_121593


namespace NUMINAMATH_CALUDE_max_value_reciprocal_l1215_121564

theorem max_value_reciprocal (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → 1 / (x + 2*y - 3*x*y) ≤ 3/2) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 1 ∧ 1 / (x + 2*y - 3*x*y) = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_reciprocal_l1215_121564


namespace NUMINAMATH_CALUDE_product_correction_l1215_121583

theorem product_correction (a b : ℕ) : 
  a > 9 ∧ a < 100 ∧ (a - 3) * b = 224 → a * b = 245 := by
  sorry

end NUMINAMATH_CALUDE_product_correction_l1215_121583


namespace NUMINAMATH_CALUDE_imaginary_number_condition_l1215_121592

theorem imaginary_number_condition (a : ℝ) : 
  let z : ℂ := (a - 2*I) / (2 + I)
  (∃ b : ℝ, z = b * I ∧ b ≠ 0) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_number_condition_l1215_121592


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_fractions_l1215_121526

theorem min_value_of_sum_of_fractions (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) : 
  (a / b + b / c + c / d + d / a) ≥ 4 ∧ 
  ((a / b + b / c + c / d + d / a) = 4 ↔ a = b ∧ b = c ∧ c = d) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_fractions_l1215_121526


namespace NUMINAMATH_CALUDE_joyce_apples_to_larry_l1215_121575

/-- The number of apples Joyce gave to Larry -/
def apples_given_to_larry (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

theorem joyce_apples_to_larry : 
  apples_given_to_larry 75 23 = 52 := by
  sorry

end NUMINAMATH_CALUDE_joyce_apples_to_larry_l1215_121575


namespace NUMINAMATH_CALUDE_vector_parallel_implies_x_y_values_l1215_121561

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (v w : Fin 3 → ℝ) : Prop :=
  ∃ (k : ℝ), ∀ (i : Fin 3), v i = k * w i

/-- Vector a defined as (1, 2, -y) -/
def a (y : ℝ) : Fin 3 → ℝ
  | 0 => 1
  | 1 => 2
  | 2 => -y
  | _ => 0

/-- Vector b defined as (x, 1, 2) -/
def b (x : ℝ) : Fin 3 → ℝ
  | 0 => x
  | 1 => 1
  | 2 => 2
  | _ => 0

/-- The sum of two vectors -/
def vec_add (v w : Fin 3 → ℝ) : Fin 3 → ℝ :=
  λ i => v i + w i

/-- The scalar multiplication of a vector -/
def vec_smul (k : ℝ) (v : Fin 3 → ℝ) : Fin 3 → ℝ :=
  λ i => k * v i

theorem vector_parallel_implies_x_y_values (x y : ℝ) :
  parallel (vec_add (a y) (vec_smul 2 (b x))) (vec_add (vec_smul 2 (a y)) (vec_smul (-1) (b x))) →
  x = 1/2 ∧ y = -4 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_implies_x_y_values_l1215_121561


namespace NUMINAMATH_CALUDE_apple_purchase_cost_l1215_121502

/-- The cost of apples in dollars per 7 pounds -/
def apple_cost : ℚ := 5

/-- The rate of apples in pounds per cost unit -/
def apple_rate : ℚ := 7

/-- The amount of apples we want to buy in pounds -/
def apple_amount : ℚ := 21

/-- Theorem: The cost of 21 pounds of apples is $15 -/
theorem apple_purchase_cost : (apple_amount / apple_rate) * apple_cost = 15 := by
  sorry

end NUMINAMATH_CALUDE_apple_purchase_cost_l1215_121502


namespace NUMINAMATH_CALUDE_profit_percent_when_cost_is_quarter_of_selling_price_l1215_121535

/-- If the cost price is 25% of the selling price, then the profit percent is 300%. -/
theorem profit_percent_when_cost_is_quarter_of_selling_price :
  ∀ (selling_price : ℝ) (cost_price : ℝ),
    selling_price > 0 →
    cost_price = 0.25 * selling_price →
    (selling_price - cost_price) / cost_price * 100 = 300 := by
  sorry

end NUMINAMATH_CALUDE_profit_percent_when_cost_is_quarter_of_selling_price_l1215_121535


namespace NUMINAMATH_CALUDE_coefficient_x_squared_is_seven_l1215_121549

/-- The coefficient of x^2 in the expansion of (1 - 3x)^7 -/
def coefficient_x_squared : ℕ :=
  Nat.choose 7 6

/-- Theorem: The coefficient of x^2 in the expansion of (1 - 3x)^7 is 7 -/
theorem coefficient_x_squared_is_seven : coefficient_x_squared = 7 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_is_seven_l1215_121549


namespace NUMINAMATH_CALUDE_tan_sum_from_sin_cos_sum_l1215_121503

theorem tan_sum_from_sin_cos_sum (α β : Real) 
  (h1 : Real.sin α + Real.sin β = (4/5) * Real.sqrt 2)
  (h2 : Real.cos α + Real.cos β = (4/5) * Real.sqrt 3) :
  Real.tan α + Real.tan β = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_from_sin_cos_sum_l1215_121503


namespace NUMINAMATH_CALUDE_probability_A_and_B_selected_l1215_121597

/-- The number of students in the group -/
def total_students : ℕ := 5

/-- The number of students to be selected -/
def selected_students : ℕ := 3

/-- The probability of selecting both A and B -/
def prob_select_A_and_B : ℚ := 3 / 10

/-- Theorem stating that the probability of selecting both A and B
    when randomly choosing 3 students from a group of 5 students is 3/10 -/
theorem probability_A_and_B_selected :
  (Nat.choose (total_students - 2) (selected_students - 2) : ℚ) /
  (Nat.choose total_students selected_students : ℚ) = prob_select_A_and_B :=
sorry

end NUMINAMATH_CALUDE_probability_A_and_B_selected_l1215_121597
