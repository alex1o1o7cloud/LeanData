import Mathlib

namespace NUMINAMATH_CALUDE_marathon_distance_theorem_l2796_279607

/-- The length of a marathon in miles -/
def marathon_miles : ℕ := 26

/-- The additional length of a marathon in yards -/
def marathon_yards : ℕ := 312

/-- The number of yards in a mile -/
def yards_per_mile : ℕ := 1760

/-- The number of marathons Joanna has run -/
def num_marathons : ℕ := 8

/-- The total distance Joanna has run in yards -/
def total_distance : ℕ := num_marathons * (marathon_miles * yards_per_mile + marathon_yards)

theorem marathon_distance_theorem :
  ∃ (m : ℕ) (y : ℕ), total_distance = m * yards_per_mile + y ∧ y = 736 ∧ y < yards_per_mile :=
by sorry

end NUMINAMATH_CALUDE_marathon_distance_theorem_l2796_279607


namespace NUMINAMATH_CALUDE_car_trip_equation_correct_l2796_279690

/-- Represents a car trip with a break -/
structure CarTrip where
  totalDistance : ℝ
  totalTime : ℝ
  breakDuration : ℝ
  speedBefore : ℝ
  speedAfter : ℝ

/-- The equation representing the relationship between time before break and total distance -/
def tripEquation (trip : CarTrip) (t : ℝ) : Prop :=
  trip.speedBefore * t + trip.speedAfter * (trip.totalTime - trip.breakDuration / 60 - t) = trip.totalDistance

theorem car_trip_equation_correct (trip : CarTrip) : 
  trip.totalDistance = 295 ∧ 
  trip.totalTime = 3.25 ∧ 
  trip.breakDuration = 15 ∧ 
  trip.speedBefore = 85 ∧ 
  trip.speedAfter = 115 → 
  ∃ t, tripEquation trip t ∧ t > 0 ∧ t < trip.totalTime - trip.breakDuration / 60 :=
sorry

end NUMINAMATH_CALUDE_car_trip_equation_correct_l2796_279690


namespace NUMINAMATH_CALUDE_rachel_money_theorem_l2796_279697

def rachel_money_problem (initial_earnings : ℝ) 
  (lunch_fraction : ℝ) (clothes_percent : ℝ) (dvd_cost : ℝ) (supplies_percent : ℝ) : Prop :=
  let lunch_cost := initial_earnings * lunch_fraction
  let clothes_cost := initial_earnings * (clothes_percent / 100)
  let supplies_cost := initial_earnings * (supplies_percent / 100)
  let total_expenses := lunch_cost + clothes_cost + dvd_cost + supplies_cost
  let money_left := initial_earnings - total_expenses
  money_left = 74.50

theorem rachel_money_theorem :
  rachel_money_problem 200 0.25 15 24.50 10.5 := by
  sorry

end NUMINAMATH_CALUDE_rachel_money_theorem_l2796_279697


namespace NUMINAMATH_CALUDE_fraction_ratio_equality_l2796_279640

theorem fraction_ratio_equality : 
  let certain_fraction : ℚ := 84 / 25
  let given_fraction : ℚ := 6 / 5
  let comparison_fraction : ℚ := 2 / 5
  let answer : ℚ := 1 / 7  -- 0.14285714285714288 is approximately 1/7
  (certain_fraction / given_fraction) = (comparison_fraction / answer) :=
by sorry

end NUMINAMATH_CALUDE_fraction_ratio_equality_l2796_279640


namespace NUMINAMATH_CALUDE_triangle_median_inequality_l2796_279670

/-- Given a triangle with side lengths a, b, c, medians m_a, m_b, m_c, 
    and circumscribed circle diameter D, the following inequality holds. -/
theorem triangle_median_inequality 
  (a b c m_a m_b m_c D : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_pos_m_a : 0 < m_a) (h_pos_m_b : 0 < m_b) (h_pos_m_c : 0 < m_c)
  (h_pos_D : 0 < D)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_median_a : 4 * m_a^2 = 2 * b^2 + 2 * c^2 - a^2)
  (h_median_b : 4 * m_b^2 = 2 * c^2 + 2 * a^2 - b^2)
  (h_median_c : 4 * m_c^2 = 2 * a^2 + 2 * b^2 - c^2)
  (h_circumradius : D = 2 * (a * b * c) / (4 * area))
  (h_area : area = Real.sqrt ((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c)) / 4) :
  (a^2 + b^2) / m_c + (b^2 + c^2) / m_a + (c^2 + a^2) / m_b ≤ 6 * D := by
  sorry

end NUMINAMATH_CALUDE_triangle_median_inequality_l2796_279670


namespace NUMINAMATH_CALUDE_initial_pencils_count_l2796_279680

/-- The number of pencils Eric takes from the box -/
def pencils_taken : ℕ := 4

/-- The number of pencils left in the box after Eric takes some -/
def pencils_left : ℕ := 75

/-- The initial number of pencils in the box -/
def initial_pencils : ℕ := pencils_taken + pencils_left

theorem initial_pencils_count : initial_pencils = 79 := by
  sorry

end NUMINAMATH_CALUDE_initial_pencils_count_l2796_279680


namespace NUMINAMATH_CALUDE_g_neg_501_l2796_279653

-- Define the function g
variable (g : ℝ → ℝ)

-- State the conditions
axiom func_eq : ∀ x y : ℝ, g (x * y) + 2 * x = x * g y + g x
axiom g_neg_one : g (-1) = 7

-- State the theorem to be proved
theorem g_neg_501 : g (-501) = 507 := by sorry

end NUMINAMATH_CALUDE_g_neg_501_l2796_279653


namespace NUMINAMATH_CALUDE_three_digit_permutations_l2796_279627

def digits : Finset Nat := {1, 5, 8}

theorem three_digit_permutations (d : Finset Nat) (h : d = digits) :
  (d.toList.permutations.filter (fun l => l.length = 3)).length = 6 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_permutations_l2796_279627


namespace NUMINAMATH_CALUDE_tiffany_max_points_l2796_279617

/-- Represents the game setup and Tiffany's current state -/
structure GameState where
  initialMoney : ℕ
  costPerGame : ℕ
  ringsPerPlay : ℕ
  redBucketPoints : ℕ
  greenBucketPoints : ℕ
  gamesPlayed : ℕ
  redBucketsHit : ℕ
  greenBucketsHit : ℕ

/-- Calculates the maximum points achievable given a GameState -/
def maxPoints (state : GameState) : ℕ :=
  let pointsFromRed := state.redBucketsHit * state.redBucketPoints
  let pointsFromGreen := state.greenBucketsHit * state.greenBucketPoints
  let moneySpent := state.gamesPlayed * state.costPerGame
  let moneyLeft := state.initialMoney - moneySpent
  let gamesLeft := moneyLeft / state.costPerGame
  let maxPointsLastGame := state.ringsPerPlay * max state.redBucketPoints state.greenBucketPoints
  pointsFromRed + pointsFromGreen + gamesLeft * maxPointsLastGame

/-- Tiffany's game state -/
def tiffanyState : GameState where
  initialMoney := 3
  costPerGame := 1
  ringsPerPlay := 5
  redBucketPoints := 2
  greenBucketPoints := 3
  gamesPlayed := 2
  redBucketsHit := 4
  greenBucketsHit := 5

/-- Theorem stating that the maximum points Tiffany can achieve is 38 -/
theorem tiffany_max_points :
  maxPoints tiffanyState = 38 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_max_points_l2796_279617


namespace NUMINAMATH_CALUDE_arithmetic_computation_l2796_279658

theorem arithmetic_computation : 2 + 4 * 3^2 - 1 + 7 * 2 / 2 = 44 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l2796_279658


namespace NUMINAMATH_CALUDE_janes_calculation_l2796_279698

theorem janes_calculation (x y z : ℝ) 
  (h1 : x - (y - z) = 15) 
  (h2 : x - y - z = 7) : 
  x - y = 11 := by
sorry

end NUMINAMATH_CALUDE_janes_calculation_l2796_279698


namespace NUMINAMATH_CALUDE_range_of_k_for_two_distinct_roots_l2796_279661

/-- The quadratic equation (k-1)x^2 + 2x - 2 = 0 has two distinct real roots -/
def has_two_distinct_real_roots (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (k - 1) * x₁^2 + 2 * x₁ - 2 = 0 ∧ (k - 1) * x₂^2 + 2 * x₂ - 2 = 0

/-- The range of k values for which the quadratic equation has two distinct real roots -/
theorem range_of_k_for_two_distinct_roots :
  ∀ k : ℝ, has_two_distinct_real_roots k ↔ k > 1/2 ∧ k ≠ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_k_for_two_distinct_roots_l2796_279661


namespace NUMINAMATH_CALUDE_shop_equations_correct_l2796_279635

/-- A shop with rooms and guests satisfying certain conditions -/
structure Shop where
  rooms : ℕ
  guests : ℕ
  seven_per_room_overflow : 7 * rooms + 7 = guests
  nine_per_room_empty : 9 * (rooms - 1) = guests

/-- The theorem stating that the system of equations correctly describes the shop's situation -/
theorem shop_equations_correct (s : Shop) : 
  (7 * s.rooms + 7 = s.guests) ∧ (9 * (s.rooms - 1) = s.guests) := by
  sorry

end NUMINAMATH_CALUDE_shop_equations_correct_l2796_279635


namespace NUMINAMATH_CALUDE_mans_speed_against_current_l2796_279650

/-- Calculates the man's speed against the current with wind and waves -/
def speed_against_current_with_wind_and_waves 
  (speed_with_current : ℝ) 
  (current_speed : ℝ) 
  (wind_effect : ℝ) 
  (wave_effect : ℝ) : ℝ :=
  speed_with_current - current_speed - wind_effect - current_speed - wave_effect

/-- Theorem stating the man's speed against the current with wind and waves -/
theorem mans_speed_against_current 
  (speed_with_current : ℝ) 
  (current_speed : ℝ) 
  (wind_effect : ℝ) 
  (wave_effect : ℝ) 
  (h1 : speed_with_current = 20) 
  (h2 : current_speed = 5) 
  (h3 : wind_effect = 2) 
  (h4 : wave_effect = 1) : 
  speed_against_current_with_wind_and_waves speed_with_current current_speed wind_effect wave_effect = 7 := by
  sorry

end NUMINAMATH_CALUDE_mans_speed_against_current_l2796_279650


namespace NUMINAMATH_CALUDE_sqrt_15_simplest_l2796_279644

-- Define what it means for a square root to be in its simplest form
def is_simplest_sqrt (n : ℝ) : Prop :=
  ∀ (a b : ℝ), a > 0 ∧ b > 0 → n ≠ a * b^2

-- Theorem statement
theorem sqrt_15_simplest : is_simplest_sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_15_simplest_l2796_279644


namespace NUMINAMATH_CALUDE_min_bailing_rate_is_8_l2796_279662

/-- Represents the fishing scenario with Steve and LeRoy -/
structure FishingScenario where
  distance_to_shore : ℝ
  water_intake_rate : ℝ
  max_water_capacity : ℝ
  rowing_speed : ℝ

/-- Calculates the minimum bailing rate required to reach the shore without sinking -/
def min_bailing_rate (scenario : FishingScenario) : ℝ :=
  -- The actual calculation is not implemented here
  sorry

/-- Theorem stating that the minimum bailing rate for the given scenario is 8 gallons per minute -/
theorem min_bailing_rate_is_8 (scenario : FishingScenario) 
  (h1 : scenario.distance_to_shore = 1)
  (h2 : scenario.water_intake_rate = 10)
  (h3 : scenario.max_water_capacity = 30)
  (h4 : scenario.rowing_speed = 4) :
  min_bailing_rate scenario = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_bailing_rate_is_8_l2796_279662


namespace NUMINAMATH_CALUDE_integral_of_root_and_polynomial_l2796_279695

open Real

theorem integral_of_root_and_polynomial (x : ℝ) :
  let f := λ x : ℝ => x^(1/2) * (3 + 2*x^(3/4))^(1/2)
  let F := λ x : ℝ => (2/15) * (3 + 2*x^(3/4))^(5/2) - (2/3) * (3 + 2*x^(3/4))^(3/2)
  deriv F x = f x := by sorry

end NUMINAMATH_CALUDE_integral_of_root_and_polynomial_l2796_279695


namespace NUMINAMATH_CALUDE_fourth_six_probability_l2796_279631

/-- Represents a six-sided die --/
structure Die :=
  (prob_six : ℚ)
  (prob_other : ℚ)
  (valid_probs : prob_six + 5 * prob_other = 1)

/-- The fair die --/
def fair_die : Die :=
  { prob_six := 1/6,
    prob_other := 1/6,
    valid_probs := by norm_num }

/-- The biased die --/
def biased_die : Die :=
  { prob_six := 3/4,
    prob_other := 1/20,
    valid_probs := by norm_num }

/-- The probability of rolling three sixes with a given die --/
def prob_three_sixes (d : Die) : ℚ := d.prob_six^3

/-- The probability of the fourth roll being a six given the first three were sixes --/
def prob_fourth_six (fair : Die) (biased : Die) : ℚ :=
  let p_fair := prob_three_sixes fair
  let p_biased := prob_three_sixes biased
  let total := p_fair + p_biased
  (p_fair / total) * fair.prob_six + (p_biased / total) * biased.prob_six

theorem fourth_six_probability :
  prob_fourth_six fair_die biased_die = 685 / 922 :=
sorry

end NUMINAMATH_CALUDE_fourth_six_probability_l2796_279631


namespace NUMINAMATH_CALUDE_five_integers_sum_20_product_420_l2796_279675

theorem five_integers_sum_20_product_420 :
  ∃ (a b c d e : ℕ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
    a + b + c + d + e = 20 ∧
    a * b * c * d * e = 420 :=
by
  sorry

end NUMINAMATH_CALUDE_five_integers_sum_20_product_420_l2796_279675


namespace NUMINAMATH_CALUDE_room_dimensions_l2796_279608

theorem room_dimensions (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_not_equal : b ≠ c) :
  let x := (b^2 * (b^2 - a^2) / (b^2 - c^2))^(1/4)
  let y := a / x
  let z := b / x
  let u := c * x / b
  ∃ (room_I room_II room_III : ℝ × ℝ),
    room_I.1 * room_I.2 = a ∧
    room_II.1 * room_II.2 = b ∧
    room_III.1 * room_III.2 = c ∧
    room_I.1 = room_II.1 ∧
    room_II.2 = room_III.2 ∧
    room_I.1^2 + room_I.2^2 = room_III.1^2 + room_III.2^2 ∧
    room_I = (x, y) ∧
    room_II = (x, z) ∧
    room_III = (u, z) := by
  sorry

#check room_dimensions

end NUMINAMATH_CALUDE_room_dimensions_l2796_279608


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l2796_279664

/-- The line passing through (-1, 0) and perpendicular to x+y=0 has equation x-y+1=0 -/
theorem perpendicular_line_through_point (x y : ℝ) : 
  (x + y = 0 → (x + 1 = 0 ∧ y = 0) → x - y + 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l2796_279664


namespace NUMINAMATH_CALUDE_team_games_total_l2796_279624

theorem team_games_total (first_games : ℕ) (first_win_rate : ℚ) (remaining_win_rate : ℚ) (total_win_rate : ℚ) : 
  first_games = 30 →
  first_win_rate = 2/5 →
  remaining_win_rate = 4/5 →
  total_win_rate = 3/5 →
  ∃ (total_games : ℕ), total_games = 60 ∧ 
    (first_win_rate * first_games + remaining_win_rate * (total_games - first_games) = total_win_rate * total_games) :=
by
  sorry

#check team_games_total

end NUMINAMATH_CALUDE_team_games_total_l2796_279624


namespace NUMINAMATH_CALUDE_factorization_proof_l2796_279638

theorem factorization_proof (x : ℝ) : 
  x^2 + 6*x + 9 - 64*x^4 = (-8*x^2 + x + 3) * (8*x^2 + x + 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l2796_279638


namespace NUMINAMATH_CALUDE_kaleb_toy_purchase_l2796_279614

def max_toys_purchasable (initial_savings : ℕ) (allowance : ℕ) (toy_cost : ℕ) : ℕ :=
  (initial_savings + allowance) / toy_cost

theorem kaleb_toy_purchase :
  max_toys_purchasable 21 15 6 = 6 := by
  sorry

end NUMINAMATH_CALUDE_kaleb_toy_purchase_l2796_279614


namespace NUMINAMATH_CALUDE_product_of_three_integers_summing_to_seven_l2796_279689

theorem product_of_three_integers_summing_to_seven (a b c : ℕ) :
  a > 0 → b > 0 → c > 0 →
  a ≠ b → b ≠ c → a ≠ c →
  a + b + c = 7 →
  a * b * c = 8 := by
sorry

end NUMINAMATH_CALUDE_product_of_three_integers_summing_to_seven_l2796_279689


namespace NUMINAMATH_CALUDE_standard_deviation_of_dataset_l2796_279688

def dataset : List ℝ := [3, 4, 5, 5, 6, 7]

theorem standard_deviation_of_dataset :
  let n : ℕ := dataset.length
  let mean : ℝ := (dataset.sum) / n
  let variance : ℝ := (dataset.map (fun x => (x - mean)^2)).sum / n
  Real.sqrt variance = Real.sqrt (5/3) := by sorry

end NUMINAMATH_CALUDE_standard_deviation_of_dataset_l2796_279688


namespace NUMINAMATH_CALUDE_total_different_movies_l2796_279604

-- Define the number of people
def num_people : ℕ := 5

-- Define the number of movies watched by each person
def dalton_movies : ℕ := 15
def hunter_movies : ℕ := 19
def alex_movies : ℕ := 25
def bella_movies : ℕ := 21
def chris_movies : ℕ := 11

-- Define the number of movies watched together
def all_together : ℕ := 5
def dalton_hunter_alex : ℕ := 3
def bella_chris : ℕ := 2

-- Theorem to prove
theorem total_different_movies : 
  dalton_movies + hunter_movies + alex_movies + bella_movies + chris_movies
  - (num_people - 1) * all_together
  - (3 - 1) * dalton_hunter_alex
  - (2 - 1) * bella_chris = 63 := by
sorry

end NUMINAMATH_CALUDE_total_different_movies_l2796_279604


namespace NUMINAMATH_CALUDE_some_students_not_honor_society_l2796_279682

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Student : U → Prop)
variable (Scholarship : U → Prop)
variable (HonorSociety : U → Prop)

-- State the theorem
theorem some_students_not_honor_society :
  (∃ x, Student x ∧ ¬Scholarship x) →
  (∀ x, HonorSociety x → Scholarship x) →
  (∃ x, Student x ∧ ¬HonorSociety x) :=
by
  sorry


end NUMINAMATH_CALUDE_some_students_not_honor_society_l2796_279682


namespace NUMINAMATH_CALUDE_red_apples_count_l2796_279646

theorem red_apples_count (red : ℕ) (green : ℕ) : 
  green = red + 12 →
  red + green = 44 →
  red = 16 := by
sorry

end NUMINAMATH_CALUDE_red_apples_count_l2796_279646


namespace NUMINAMATH_CALUDE_sqrt_of_nine_l2796_279626

theorem sqrt_of_nine : 
  {x : ℝ | x^2 = 9} = {3, -3} := by sorry

end NUMINAMATH_CALUDE_sqrt_of_nine_l2796_279626


namespace NUMINAMATH_CALUDE_periodic_trig_function_l2796_279699

/-- Given a function f(x) = a*sin(π*x + α) + b*cos(π*x + β) + 4,
    where a, b, α, and β are constants, 
    if f(2009) = 5, then f(2010) = 3 -/
theorem periodic_trig_function 
  (a b α β : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * Real.sin (π * x + α) + b * Real.cos (π * x + β) + 4) 
  (h2 : f 2009 = 5) : 
  f 2010 = 3 := by
sorry

end NUMINAMATH_CALUDE_periodic_trig_function_l2796_279699


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_intersection_eccentricity_l2796_279625

/-- An ellipse with foci and eccentricity -/
structure Ellipse :=
  (F₁ F₂ : ℝ × ℝ)
  (e : ℝ)

/-- A hyperbola with foci and eccentricity -/
structure Hyperbola :=
  (F₁ F₂ : ℝ × ℝ)
  (e : ℝ)

/-- The dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  (v.1 * w.1) + (v.2 * w.2)

theorem ellipse_hyperbola_intersection_eccentricity 
  (C₁ : Ellipse) (C₂ : Hyperbola) (P : ℝ × ℝ) :
  C₁.F₁ = C₂.F₁ →
  C₁.F₂ = C₂.F₂ →
  dot_product (P.1 - C₁.F₁.1, P.2 - C₁.F₁.2) (P.1 - C₁.F₂.1, P.2 - C₁.F₂.2) = 0 →
  (1 / C₁.e^2) + (1 / C₂.e^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_intersection_eccentricity_l2796_279625


namespace NUMINAMATH_CALUDE_equation_solution_l2796_279620

theorem equation_solution : 
  let f : ℝ → ℝ := λ x => (x^2 - x + 1) * (3*x^2 - 10*x + 3) - 20*x^2
  ∀ x : ℝ, f x = 0 ↔ x = (5 + Real.sqrt 21) / 2 ∨ x = (5 - Real.sqrt 21) / 2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2796_279620


namespace NUMINAMATH_CALUDE_existence_of_solution_l2796_279672

/-- Floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- Condition: For any positive integers k₁ and k₂, ⌊k₁α⌋ ≠ ⌊k₂β⌋ -/
def condition (α β : ℝ) : Prop :=
  ∀ (k₁ k₂ : ℕ), k₁ > 0 ∧ k₂ > 0 → floor (k₁ * α) ≠ floor (k₂ * β)

/-- Theorem: If the condition holds for positive real numbers α and β,
    then there exist positive integers m₁ and m₂ such that (m₁/α) + (m₂/β) = 1 -/
theorem existence_of_solution (α β : ℝ) (hα : α > 0) (hβ : β > 0) 
    (h : condition α β) : 
    ∃ (m₁ m₂ : ℕ), m₁ > 0 ∧ m₂ > 0 ∧ (m₁ : ℝ) / α + (m₂ : ℝ) / β = 1 :=
  sorry

end NUMINAMATH_CALUDE_existence_of_solution_l2796_279672


namespace NUMINAMATH_CALUDE_line_equation_through_A_and_B_l2796_279674

/-- Two-point form equation of a line passing through two points -/
def two_point_form (x1 y1 x2 y2 : ℝ) (x y : ℝ) : Prop :=
  (x - x1) / (x2 - x1) = (y - y1) / (y2 - y1)

/-- Theorem: The two-point form equation of the line passing through A(1,2) and B(-1,1) -/
theorem line_equation_through_A_and_B :
  two_point_form 1 2 (-1) 1 x y ↔ (x - 1) / (-2) = (y - 2) / (-1) := by
  sorry

end NUMINAMATH_CALUDE_line_equation_through_A_and_B_l2796_279674


namespace NUMINAMATH_CALUDE_sqrt_two_squared_l2796_279667

theorem sqrt_two_squared : (Real.sqrt 2)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_squared_l2796_279667


namespace NUMINAMATH_CALUDE_main_theorem_l2796_279637

/-- Proposition p: for all positive x, x + a/x ≥ 2 -/
def p (a : ℝ) : Prop :=
  ∀ x > 0, x + a / x ≥ 2

/-- Proposition q: for all real k, the line kx - y + 2 = 0 intersects with the ellipse x^2 + y^2/a^2 = 1 -/
def q (a : ℝ) : Prop :=
  ∀ k : ℝ, ∃ x y : ℝ, k * x - y + 2 = 0 ∧ x^2 + y^2 / a^2 = 1

/-- The main theorem: (p ∨ q) ∧ ¬(p ∧ q) is true if and only if 1 ≤ a < 2 -/
theorem main_theorem (a : ℝ) (h : a > 0) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) ↔ 1 ≤ a ∧ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_main_theorem_l2796_279637


namespace NUMINAMATH_CALUDE_a_4_equals_8_l2796_279622

def geometric_sequence (a : ℝ) (q : ℝ) (n : ℕ) : ℝ := a * q^(n - 1)

theorem a_4_equals_8 
  (a : ℝ) (q : ℝ) 
  (h1 : geometric_sequence a q 6 + geometric_sequence a q 2 = 34)
  (h2 : geometric_sequence a q 6 - geometric_sequence a q 2 = 30) :
  geometric_sequence a q 4 = 8 :=
sorry

end NUMINAMATH_CALUDE_a_4_equals_8_l2796_279622


namespace NUMINAMATH_CALUDE_sin_330_degrees_l2796_279609

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l2796_279609


namespace NUMINAMATH_CALUDE_expected_adjacent_red_pairs_l2796_279692

/-- The number of cards in a standard deck -/
def deckSize : ℕ := 52

/-- The number of red cards in a standard deck -/
def redCards : ℕ := 26

/-- The probability that a card adjacent to a red card is also red -/
def probAdjacentRed : ℚ := 25 / 51

theorem expected_adjacent_red_pairs (deckSize : ℕ) (redCards : ℕ) (probAdjacentRed : ℚ) :
  deckSize = 52 → redCards = 26 → probAdjacentRed = 25 / 51 →
  (redCards : ℚ) * probAdjacentRed = 650 / 51 := by
  sorry

end NUMINAMATH_CALUDE_expected_adjacent_red_pairs_l2796_279692


namespace NUMINAMATH_CALUDE_reciprocal_inequality_l2796_279671

theorem reciprocal_inequality (a b : ℝ) (ha : a < 0) (hb : b > 0) : 1 / a < 1 / b := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_inequality_l2796_279671


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l2796_279681

/-- A quadratic function symmetric about the y-axis -/
def QuadraticFunction (a c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + c

theorem quadratic_function_theorem (a c : ℝ) :
  (QuadraticFunction a c 0 = -2) →
  (QuadraticFunction a c 1 = -1) →
  (∃ (x : ℝ), QuadraticFunction a c x = QuadraticFunction a c (-x)) →
  (QuadraticFunction a c = fun x ↦ x^2 - 2) ∧
  (∃! (x y : ℝ), x ≠ y ∧ QuadraticFunction a c x = 0 ∧ QuadraticFunction a c y = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l2796_279681


namespace NUMINAMATH_CALUDE_shoe_price_problem_l2796_279632

theorem shoe_price_problem (first_pair_price : ℝ) (total_paid : ℝ) :
  first_pair_price = 40 →
  total_paid = 60 →
  ∃ (second_pair_price : ℝ),
    second_pair_price ≥ first_pair_price ∧
    total_paid = (3/4) * (first_pair_price + (1/2) * second_pair_price) ∧
    second_pair_price = 80 :=
by
  sorry

#check shoe_price_problem

end NUMINAMATH_CALUDE_shoe_price_problem_l2796_279632


namespace NUMINAMATH_CALUDE_floor_of_e_l2796_279649

theorem floor_of_e : ⌊Real.exp 1⌋ = 2 := by sorry

end NUMINAMATH_CALUDE_floor_of_e_l2796_279649


namespace NUMINAMATH_CALUDE_triangle_area_l2796_279656

/-- The area of a triangle with vertices at (0,0), (8,8), and (-8,8) is 64 -/
theorem triangle_area : 
  let O : ℝ × ℝ := (0, 0)
  let A : ℝ × ℝ := (8, 8)
  let B : ℝ × ℝ := (-8, 8)
  let base := |A.1 - B.1|
  let height := A.2
  (1 / 2) * base * height = 64 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l2796_279656


namespace NUMINAMATH_CALUDE_smallest_student_count_l2796_279628

/-- Represents the number of students in each grade --/
structure GradeCount where
  sixth : ℕ
  eighth : ℕ
  ninth : ℕ

/-- Checks if the given counts satisfy the required ratios --/
def satisfiesRatios (counts : GradeCount) : Prop :=
  5 * counts.sixth = 3 * counts.eighth ∧
  7 * counts.ninth = 4 * counts.eighth

/-- The total number of students --/
def totalStudents (counts : GradeCount) : ℕ :=
  counts.sixth + counts.eighth + counts.ninth

/-- Theorem stating the smallest possible number of students --/
theorem smallest_student_count :
  ∃ (counts : GradeCount),
    satisfiesRatios counts ∧
    totalStudents counts = 76 ∧
    ∀ (other : GradeCount),
      satisfiesRatios other →
      totalStudents other ≥ 76 :=
by sorry

end NUMINAMATH_CALUDE_smallest_student_count_l2796_279628


namespace NUMINAMATH_CALUDE_square_sum_or_product_l2796_279636

theorem square_sum_or_product (a b c : ℕ+) (p : ℕ) :
  a + b = b * (a - c) →
  c + 1 = p^2 →
  Nat.Prime p →
  (∃ k : ℕ, (a + b : ℕ) = k^2) ∨ (∃ k : ℕ, (a * b : ℕ) = k^2) := by
  sorry

end NUMINAMATH_CALUDE_square_sum_or_product_l2796_279636


namespace NUMINAMATH_CALUDE_collinear_vectors_l2796_279655

/-- Given vectors a and b in ℝ², if a + b is collinear with a, then the second component of a is 1. -/
theorem collinear_vectors (k : ℝ) : 
  let a : Fin 2 → ℝ := ![1, k]
  let b : Fin 2 → ℝ := ![2, 2]
  (∃ (t : ℝ), (a + b) = t • a) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_l2796_279655


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_properties_l2796_279651

/-- A right triangle with two equal angles and hypotenuse length 12 -/
structure IsoscelesRightTriangle where
  -- Side lengths
  a : ℝ
  b : ℝ
  c : ℝ
  -- Angle equality
  angle_equality : a = b
  -- Hypotenuse length
  hypotenuse_length : c = 12

/-- Theorem about the properties of an isosceles right triangle -/
theorem isosceles_right_triangle_properties (t : IsoscelesRightTriangle) :
  t.a = 6 * Real.sqrt 2 ∧ (1/2 : ℝ) * t.a * t.b = 36 := by
  sorry

#check isosceles_right_triangle_properties

end NUMINAMATH_CALUDE_isosceles_right_triangle_properties_l2796_279651


namespace NUMINAMATH_CALUDE_three_digit_powers_of_three_l2796_279602

theorem three_digit_powers_of_three (n : ℕ) : 
  (∃ k, 100 ≤ 3^k ∧ 3^k ≤ 999) ∧ (∀ m, 100 ≤ 3^m ∧ 3^m ≤ 999 → m = n ∨ m = n+1) :=
sorry

end NUMINAMATH_CALUDE_three_digit_powers_of_three_l2796_279602


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2796_279677

theorem purely_imaginary_complex_number (m : ℝ) :
  let z : ℂ := (m - 3 * Complex.I) / (2 + Complex.I)
  (∃ (y : ℝ), z = Complex.I * y) → m = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2796_279677


namespace NUMINAMATH_CALUDE_collinear_points_k_value_l2796_279601

/-- Three points (x1, y1), (x2, y2), and (x3, y3) are collinear if and only if
    the slope between any two pairs of points is equal. -/
def collinear (x1 y1 x2 y2 x3 y3 : ℝ) : Prop :=
  (y2 - y1) * (x3 - x2) = (y3 - y2) * (x2 - x1)

/-- The theorem states that for three collinear points (1, 2), (3, k), and (10, 5),
    the value of k must be 8/3. -/
theorem collinear_points_k_value :
  collinear 1 2 3 k 10 5 → k = 8/3 :=
by
  sorry

end NUMINAMATH_CALUDE_collinear_points_k_value_l2796_279601


namespace NUMINAMATH_CALUDE_olly_minimum_cost_l2796_279647

/-- Represents the number of each type of pet Olly has -/
structure Pets where
  dogs : Nat
  cats : Nat
  ferrets : Nat

/-- Represents the pricing and discount structure for Pack A -/
structure PackA where
  small_shoe_price : ℝ
  medium_shoe_price : ℝ
  small_shoe_discount : ℝ
  medium_shoe_discount : ℝ

/-- Represents the pricing and discount structure for Pack B -/
structure PackB where
  small_shoe_price : ℝ
  medium_shoe_price : ℝ
  small_shoe_free_ratio : Nat
  medium_shoe_free_ratio : Nat

/-- Calculates the minimum cost for Olly to purchase shoes for all his pets -/
def minimum_cost (pets : Pets) (pack_a : PackA) (pack_b : PackB) : ℝ := by
  sorry

/-- Theorem stating that the minimum cost for Olly to purchase shoes for all his pets is $64 -/
theorem olly_minimum_cost :
  let pets := Pets.mk 3 2 1
  let pack_a := PackA.mk 12 16 0.2 0.15
  let pack_b := PackB.mk 7 9 3 4
  minimum_cost pets pack_a pack_b = 64 := by
  sorry

end NUMINAMATH_CALUDE_olly_minimum_cost_l2796_279647


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l2796_279634

def is_decreasing (a : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, a n ≥ a (n + 1)

theorem contrapositive_equivalence (a : ℕ+ → ℝ) :
  (∀ n : ℕ+, (a n + a (n + 2)) / 2 < a (n + 1) → is_decreasing a) ↔
  (¬ is_decreasing a → ∀ n : ℕ+, (a n + a (n + 2)) / 2 ≥ a (n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l2796_279634


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l2796_279659

-- Define the function f
def f (x t : ℝ) : ℝ := |x - 1| + |x - t|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f x 2 > 2} = {x : ℝ | x < (1/2) ∨ x > (5/2)} := by sorry

-- Part 2
theorem range_of_a_part2 :
  ∀ a : ℝ, (∀ t x : ℝ, t ∈ [1, 2] → x ∈ [-1, 3] → f x t ≥ a + x) → a ≤ -1 := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l2796_279659


namespace NUMINAMATH_CALUDE_probability_correct_l2796_279648

/-- Represents a standard six-sided die --/
def Die := Fin 6

/-- The probability of the event described in the problem --/
def probability : ℚ :=
  (5 * 4^9) / (6^11)

/-- The function that calculates the probability of the event --/
def calculate_probability : ℚ :=
  -- First roll: any number (1)
  -- Rolls 2 to 10: different from previous, not 4 on 11th (5/6 * (4/5)^9)
  -- 11th and 12th rolls both 4 (1/6 * 1/6)
  1 * (5/6) * (4/5)^9 * (1/6)^2

theorem probability_correct :
  calculate_probability = probability := by sorry

end NUMINAMATH_CALUDE_probability_correct_l2796_279648


namespace NUMINAMATH_CALUDE_number_problem_l2796_279654

theorem number_problem : ∃ x : ℝ, x^2 + 95 = (x - 20)^2 ∧ x = 7.625 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2796_279654


namespace NUMINAMATH_CALUDE_smallest_y_squared_l2796_279668

/-- An isosceles trapezoid with a inscribed circle --/
structure IsoscelesTrapezoidWithCircle where
  -- Length of the longer base
  AB : ℝ
  -- Length of the shorter base
  CD : ℝ
  -- Length of the legs
  y : ℝ
  -- The circle's center is on AB and it's tangent to AD and BC
  has_inscribed_circle : Bool

/-- The smallest possible y value for the given trapezoid configuration --/
def smallest_y (t : IsoscelesTrapezoidWithCircle) : ℝ :=
  sorry

/-- Theorem stating that the square of the smallest y is 900 --/
theorem smallest_y_squared (t : IsoscelesTrapezoidWithCircle) 
  (h1 : t.AB = 100)
  (h2 : t.CD = 64)
  (h3 : t.has_inscribed_circle = true) :
  (smallest_y t) ^ 2 = 900 :=
sorry

end NUMINAMATH_CALUDE_smallest_y_squared_l2796_279668


namespace NUMINAMATH_CALUDE_odd_sum_of_squares_implies_odd_sum_l2796_279606

theorem odd_sum_of_squares_implies_odd_sum (n m : ℤ) 
  (h : Odd (n^2 + m^2)) : Odd (n + m) := by
  sorry

end NUMINAMATH_CALUDE_odd_sum_of_squares_implies_odd_sum_l2796_279606


namespace NUMINAMATH_CALUDE_union_with_complement_l2796_279613

theorem union_with_complement (I A B : Set ℕ) : 
  I = {1, 2, 3, 4} →
  A = {1} →
  B = {2, 4} →
  A ∪ (I \ B) = {1, 3} := by
  sorry

end NUMINAMATH_CALUDE_union_with_complement_l2796_279613


namespace NUMINAMATH_CALUDE_rain_probability_rain_probability_in_both_areas_l2796_279630

theorem rain_probability (P₁ P₂ : ℝ) 
  (h₁ : 0 < P₁ ∧ P₁ < 1) 
  (h₂ : 0 < P₂ ∧ P₂ < 1) 
  (h_independent : True) -- Representing independence condition
  : ℝ :=
(1 - P₁) * (1 - P₂)

theorem rain_probability_in_both_areas (P₁ P₂ : ℝ) 
  (h₁ : 0 < P₁ ∧ P₁ < 1) 
  (h₂ : 0 < P₂ ∧ P₂ < 1) 
  (h_independent : True) -- Representing independence condition
  : rain_probability P₁ P₂ h₁ h₂ h_independent = (1 - P₁) * (1 - P₂) :=
sorry

end NUMINAMATH_CALUDE_rain_probability_rain_probability_in_both_areas_l2796_279630


namespace NUMINAMATH_CALUDE_prove_a_minus_b_l2796_279657

-- Define the equation
def equation (a b c x : ℝ) : Prop :=
  (2*x - 3)^2 = a*x^2 + b*x + c

-- Theorem statement
theorem prove_a_minus_b (a b c : ℝ) 
  (h : ∀ x : ℝ, equation a b c x) : a - b = 16 := by
  sorry

end NUMINAMATH_CALUDE_prove_a_minus_b_l2796_279657


namespace NUMINAMATH_CALUDE_height_comparison_l2796_279676

theorem height_comparison (height_a height_b : ℝ) (h : height_a = 0.75 * height_b) :
  (height_b - height_a) / height_a = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_height_comparison_l2796_279676


namespace NUMINAMATH_CALUDE_bakery_earnings_for_five_days_l2796_279687

/-- Represents the daily production and prices of baked goods in Uki's bakery --/
structure BakeryData where
  cupcake_price : ℝ
  cookie_price : ℝ
  biscuit_price : ℝ
  cupcakes_per_day : ℕ
  cookie_packets_per_day : ℕ
  biscuit_packets_per_day : ℕ

/-- Calculates the total earnings for a given number of days --/
def total_earnings (data : BakeryData) (days : ℕ) : ℝ :=
  let daily_earnings := 
    data.cupcake_price * data.cupcakes_per_day +
    data.cookie_price * data.cookie_packets_per_day +
    data.biscuit_price * data.biscuit_packets_per_day
  daily_earnings * days

/-- Theorem stating that the total earnings for 5 days is $350 --/
theorem bakery_earnings_for_five_days :
  let data := BakeryData.mk 1.5 2 1 20 10 20
  total_earnings data 5 = 350 := by
  sorry

end NUMINAMATH_CALUDE_bakery_earnings_for_five_days_l2796_279687


namespace NUMINAMATH_CALUDE_farm_trip_chaperones_l2796_279686

theorem farm_trip_chaperones (num_students : ℕ) (student_fee adult_fee total_fee : ℚ) : 
  num_students = 35 →
  student_fee = 5 →
  adult_fee = 6 →
  total_fee = 199 →
  ∃ (num_adults : ℕ), num_adults * adult_fee + num_students * student_fee = total_fee ∧ num_adults = 4 :=
by sorry

end NUMINAMATH_CALUDE_farm_trip_chaperones_l2796_279686


namespace NUMINAMATH_CALUDE_ten_by_ten_not_tileable_l2796_279669

/-- Represents a checkerboard -/
structure Checkerboard :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a tile -/
structure Tile :=
  (width : ℕ)
  (height : ℕ)

/-- Defines a function to check if a checkerboard can be tiled with given tiles -/
def can_tile (board : Checkerboard) (tile : Tile) : Prop :=
  ∃ (n : ℕ), board.rows * board.cols = n * tile.width * tile.height

/-- Theorem stating that a 10x10 checkerboard cannot be tiled with 1x4 tiles -/
theorem ten_by_ten_not_tileable :
  ¬ can_tile (Checkerboard.mk 10 10) (Tile.mk 1 4) :=
sorry

end NUMINAMATH_CALUDE_ten_by_ten_not_tileable_l2796_279669


namespace NUMINAMATH_CALUDE_lauryns_company_men_count_l2796_279621

/-- The number of men employed by Lauryn's computer company. -/
def num_men : ℕ := 80

/-- The number of women employed by Lauryn's computer company. -/
def num_women : ℕ := num_men + 20

/-- The total number of employees in Lauryn's computer company. -/
def total_employees : ℕ := 180

/-- Theorem stating that the number of men employed by Lauryn is 80,
    given the conditions of the problem. -/
theorem lauryns_company_men_count :
  (num_men + num_women = total_employees) ∧ 
  (num_women = num_men + 20) →
  num_men = 80 := by
  sorry

end NUMINAMATH_CALUDE_lauryns_company_men_count_l2796_279621


namespace NUMINAMATH_CALUDE_game_winner_l2796_279673

/-- Given a game with three players and three cards, prove who received q marbles in the first round -/
theorem game_winner (p q r : ℕ) (total_rounds : ℕ) : 
  0 < p → p < q → q < r →
  total_rounds > 1 →
  total_rounds * (p + q + r) = 39 →
  2 * p + r = 10 →
  2 * q + p = 9 →
  q = 4 →
  (∃ (x : ℕ), x = total_rounds ∧ x = 3) →
  (∃ (player : String), player = "A" ∧ 
    (∀ (other : String), other ≠ "A" → 
      (other = "B" → (∃ (y : ℕ), y = r ∧ y = 8)) ∧ 
      (other = "C" → (∃ (z : ℕ), z = p ∧ z = 1)))) :=
by sorry

end NUMINAMATH_CALUDE_game_winner_l2796_279673


namespace NUMINAMATH_CALUDE_harkamal_mangoes_purchase_l2796_279600

/-- The amount of mangoes purchased by Harkamal -/
def mangoes : ℕ := sorry

theorem harkamal_mangoes_purchase :
  let grapes_kg : ℕ := 8
  let grapes_rate : ℕ := 70
  let mango_rate : ℕ := 50
  let total_paid : ℕ := 1010
  grapes_kg * grapes_rate + mangoes * mango_rate = total_paid →
  mangoes = 9 := by sorry

end NUMINAMATH_CALUDE_harkamal_mangoes_purchase_l2796_279600


namespace NUMINAMATH_CALUDE_travel_time_ratio_is_one_to_one_l2796_279618

-- Define the time spent on each leg of the journey
def walk_to_bus : ℕ := 5
def bus_ride : ℕ := 20
def walk_to_job : ℕ := 5

-- Define the total travel time per year in hours
def total_travel_time_per_year : ℕ := 365

-- Define the number of days worked per year
def days_per_year : ℕ := 365

-- Define the total travel time for one way (morning or evening)
def one_way_travel_time : ℕ := walk_to_bus + bus_ride + walk_to_job

-- Theorem to prove
theorem travel_time_ratio_is_one_to_one :
  one_way_travel_time = (total_travel_time_per_year * 60) / (2 * days_per_year) :=
by
  sorry

#check travel_time_ratio_is_one_to_one

end NUMINAMATH_CALUDE_travel_time_ratio_is_one_to_one_l2796_279618


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l2796_279691

theorem pure_imaginary_fraction (a : ℝ) : 
  (Complex.I * ((a + 2 * Complex.I) / (1 + Complex.I))).re = ((a + 2 * Complex.I) / (1 + Complex.I)).re → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l2796_279691


namespace NUMINAMATH_CALUDE_least_people_for_cheaper_second_service_l2796_279684

/-- Represents a catering service with a basic fee and per-person charge -/
structure CateringService where
  basicFee : ℕ
  perPersonCharge : ℕ

/-- Calculates the total cost for a catering service given the number of people -/
def totalCost (service : CateringService) (people : ℕ) : ℕ :=
  service.basicFee + service.perPersonCharge * people

/-- The first catering service -/
def service1 : CateringService := { basicFee := 150, perPersonCharge := 18 }

/-- The second catering service -/
def service2 : CateringService := { basicFee := 250, perPersonCharge := 15 }

/-- Theorem stating that 34 is the least number of people for which the second service is cheaper -/
theorem least_people_for_cheaper_second_service :
  (∀ n : ℕ, n < 34 → totalCost service1 n ≤ totalCost service2 n) ∧
  totalCost service2 34 < totalCost service1 34 := by
  sorry

end NUMINAMATH_CALUDE_least_people_for_cheaper_second_service_l2796_279684


namespace NUMINAMATH_CALUDE_component_usage_impossibility_l2796_279683

theorem component_usage_impossibility (p q r : ℕ) : 
  ¬∃ (x y z : ℕ), 
    (2 * x + 2 * z = 2 * p + 2 * r + 2) ∧
    (2 * x + y = 2 * p + q + 1) ∧
    (y + z = q + r) := by
  sorry

end NUMINAMATH_CALUDE_component_usage_impossibility_l2796_279683


namespace NUMINAMATH_CALUDE_max_sum_of_four_digit_integers_l2796_279605

/-- A function that returns true if a number is a 4-digit integer -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- A function that returns the set of digits in a number -/
def digits (n : ℕ) : Finset ℕ :=
  Finset.filter (fun d => d < 10) (Finset.range (n + 1))

/-- The theorem statement -/
theorem max_sum_of_four_digit_integers (a c : ℕ) :
  is_four_digit a ∧ is_four_digit c ∧
  (digits a ∪ digits c = Finset.range 10) →
  a + c ≤ 18395 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_four_digit_integers_l2796_279605


namespace NUMINAMATH_CALUDE_ellipse_properties_l2796_279611

/-- Definition of an ellipse C with given parameters -/
def Ellipse (a b : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

/-- Definition of an equilateral triangle with given side length -/
def EquilateralTriangle (side : ℝ) (p1 p2 p3 : ℝ × ℝ) :=
  ‖p1 - p2‖ = side ∧ ‖p2 - p3‖ = side ∧ ‖p3 - p1‖ = side

/-- Theorem about the properties of a specific ellipse -/
theorem ellipse_properties (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (F1 F2 B : ℝ × ℝ) (h3 : EquilateralTriangle 2 B F1 F2) :
  ∃ (C : Set (ℝ × ℝ)) (e : ℝ) (l1 l2 : ℝ → ℝ),
    C = Ellipse 2 (Real.sqrt 3) ∧
    e = (1 : ℝ) / 2 ∧
    (∀ x, l1 x = (Real.sqrt 5 * x - Real.sqrt 5) / 2) ∧
    (∀ x, l2 x = (-Real.sqrt 5 * x + Real.sqrt 5) / 2) ∧
    (∃ P Q : ℝ × ℝ, P ∈ C ∧ Q ∈ C ∧
      (P.2 = l1 P.1 ∨ P.2 = l2 P.1) ∧
      (Q.2 = l1 Q.1 ∨ Q.2 = l2 Q.1) ∧
      ((P.1 - 2) * (Q.2 + 1) = (P.2) * (Q.1 + 1))) :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2796_279611


namespace NUMINAMATH_CALUDE_system_solution_l2796_279663

theorem system_solution (x y z : ℝ) : 
  (Real.sqrt (2 * x^2 + 2) = y + 1 ∧
   Real.sqrt (2 * y^2 + 2) = z + 1 ∧
   Real.sqrt (2 * z^2 + 2) = x + 1) →
  (x = 1 ∧ y = 1 ∧ z = 1) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2796_279663


namespace NUMINAMATH_CALUDE_equation_solution_l2796_279685

theorem equation_solution :
  ∃ y : ℝ, ∀ x : ℝ, x + 0.35 * y - (x + y) = 200 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2796_279685


namespace NUMINAMATH_CALUDE_football_game_attendance_l2796_279641

/-- Prove the total attendance at a football game -/
theorem football_game_attendance
  (adult_price : ℚ)
  (child_price : ℚ)
  (total_collected : ℚ)
  (num_adults : ℕ)
  (h1 : adult_price = 60 / 100)
  (h2 : child_price = 25 / 100)
  (h3 : total_collected = 140)
  (h4 : num_adults = 200) :
  num_adults + ((total_collected - (↑num_adults * adult_price)) / child_price) = 280 :=
by sorry

end NUMINAMATH_CALUDE_football_game_attendance_l2796_279641


namespace NUMINAMATH_CALUDE_midway_point_distance_l2796_279629

/-- The distance from Yooseon's house to the midway point of her path to school -/
def midway_distance (house_to_hospital : ℕ) (hospital_to_school : ℕ) : ℕ :=
  (house_to_hospital + hospital_to_school) / 2

theorem midway_point_distance :
  let house_to_hospital := 1700
  let hospital_to_school := 900
  midway_distance house_to_hospital hospital_to_school = 1300 := by
  sorry

end NUMINAMATH_CALUDE_midway_point_distance_l2796_279629


namespace NUMINAMATH_CALUDE_min_omega_l2796_279633

theorem min_omega (f : ℝ → ℝ) (ω : ℝ) :
  (∀ x, f x = 2 * Real.sin (ω * x)) →
  ω > 0 →
  (∀ x ∈ Set.Icc (-π/3) (π/4), f x ≥ -2) →
  (∃ x ∈ Set.Icc (-π/3) (π/4), f x = -2) →
  ω ≥ 3/2 ∧ ∀ ω' > 0, (∀ x ∈ Set.Icc (-π/3) (π/4), 2 * Real.sin (ω' * x) ≥ -2) →
    (∃ x ∈ Set.Icc (-π/3) (π/4), 2 * Real.sin (ω' * x) = -2) → ω' ≥ 3/2 :=
by sorry

end NUMINAMATH_CALUDE_min_omega_l2796_279633


namespace NUMINAMATH_CALUDE_expression_simplification_l2796_279639

theorem expression_simplification (x : ℝ) :
  2 * x * (4 * x^2 - 3) - 6 * (x^2 - 3 * x + 8) = 8 * x^3 - 6 * x^2 + 12 * x - 48 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2796_279639


namespace NUMINAMATH_CALUDE_rectangle_area_l2796_279660

/-- Given a rectangle with width 5 inches and length 3 times its width, prove its area is 75 square inches. -/
theorem rectangle_area (width : ℝ) (length : ℝ) (area : ℝ) : 
  width = 5 → length = 3 * width → area = length * width → area = 75 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2796_279660


namespace NUMINAMATH_CALUDE_bus_students_count_l2796_279615

theorem bus_students_count (initial_students : Real) (students_boarding : Real) : 
  initial_students = 10.0 → students_boarding = 3.0 → initial_students + students_boarding = 13.0 := by
  sorry

end NUMINAMATH_CALUDE_bus_students_count_l2796_279615


namespace NUMINAMATH_CALUDE_set_A_equals_circle_B_l2796_279616

-- Define the circle D
def circle_D (Q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {P | dist P Q = 10}

-- Define point B
def point_B (Q : ℝ × ℝ) : ℝ × ℝ :=
  let v : ℝ × ℝ := (6, 0)  -- Arbitrary direction, 6 units from Q
  (Q.1 + v.1, Q.2 + v.2)

-- Define the set of points A satisfying the condition
def set_A (Q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {A | ∀ P ∈ circle_D Q, dist A (point_B Q) ≤ dist A P}

-- Define the circle with center B and radius 4
def circle_B (Q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {P | dist P (point_B Q) ≤ 4}

-- The theorem to prove
theorem set_A_equals_circle_B (Q : ℝ × ℝ) : set_A Q = circle_B Q := by
  sorry

end NUMINAMATH_CALUDE_set_A_equals_circle_B_l2796_279616


namespace NUMINAMATH_CALUDE_min_equal_fruits_cost_l2796_279694

/-- Represents a package of fruits -/
structure Package where
  apples : ℕ
  oranges : ℕ
  cost : ℕ

/-- The two available packages -/
def package1 : Package := ⟨3, 12, 5⟩
def package2 : Package := ⟨20, 5, 13⟩

/-- The minimum nonzero amount to spend for equal apples and oranges -/
def minEqualFruitsCost : ℕ := 64

/-- Theorem stating the minimum cost for equal fruits -/
theorem min_equal_fruits_cost :
  ∀ x y : ℕ,
    x * package1.apples + y * package2.apples = x * package1.oranges + y * package2.oranges →
    x > 0 ∨ y > 0 →
    x * package1.cost + y * package2.cost ≥ minEqualFruitsCost :=
sorry

end NUMINAMATH_CALUDE_min_equal_fruits_cost_l2796_279694


namespace NUMINAMATH_CALUDE_rachel_age_proof_l2796_279603

/-- Rachel's age in years -/
def rachel_age : ℕ := 12

/-- Rachel's grandfather's age in years -/
def grandfather_age (r : ℕ) : ℕ := 7 * r

/-- Rachel's mother's age in years -/
def mother_age (r : ℕ) : ℕ := grandfather_age r / 2

/-- Rachel's father's age in years -/
def father_age (r : ℕ) : ℕ := mother_age r + 5

theorem rachel_age_proof :
  rachel_age = 12 ∧
  grandfather_age rachel_age = 7 * rachel_age ∧
  mother_age rachel_age = grandfather_age rachel_age / 2 ∧
  father_age rachel_age = mother_age rachel_age + 5 ∧
  father_age rachel_age = rachel_age + 35 ∧
  father_age 25 = 60 :=
by sorry

end NUMINAMATH_CALUDE_rachel_age_proof_l2796_279603


namespace NUMINAMATH_CALUDE_lines_planes_perpendicular_l2796_279645

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem lines_planes_perpendicular 
  (m n : Line) (α β : Plane) :
  parallel m n →
  contains α m →
  perpendicular n β →
  plane_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_lines_planes_perpendicular_l2796_279645


namespace NUMINAMATH_CALUDE_apple_orange_difference_l2796_279666

theorem apple_orange_difference (total : Nat) (apples : Nat) (h1 : total = 301) (h2 : apples = 164) (h3 : apples > total - apples) : apples - (total - apples) = 27 := by
  sorry

end NUMINAMATH_CALUDE_apple_orange_difference_l2796_279666


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l2796_279643

def point (x y : ℝ) := (x, y)

def symmetric_point_x_axis (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1, -P.2)

theorem symmetric_point_coordinates :
  let P : ℝ × ℝ := point (-2) 1
  let Q : ℝ × ℝ := symmetric_point_x_axis P
  Q = point (-2) (-1) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l2796_279643


namespace NUMINAMATH_CALUDE_handshake_count_l2796_279696

/-- Represents the number of women in each age group -/
def women_per_group : ℕ := 5

/-- Represents the number of age groups -/
def num_groups : ℕ := 3

/-- Calculates the number of inter-group handshakes -/
def inter_group_handshakes : ℕ := women_per_group * women_per_group * (num_groups.choose 2)

/-- Calculates the number of intra-group handshakes for a single group -/
def intra_group_handshakes : ℕ := women_per_group.choose 2

/-- Calculates the total number of handshakes -/
def total_handshakes : ℕ := inter_group_handshakes + num_groups * intra_group_handshakes

/-- Theorem stating that the total number of handshakes is 105 -/
theorem handshake_count : total_handshakes = 105 := by
  sorry

end NUMINAMATH_CALUDE_handshake_count_l2796_279696


namespace NUMINAMATH_CALUDE_kennedy_drive_home_l2796_279693

/-- Calculates the remaining miles that can be driven given the car's efficiency,
    initial gas amount, and distance already driven. -/
def remaining_miles (efficiency : ℝ) (initial_gas : ℝ) (driven_miles : ℝ) : ℝ :=
  efficiency * initial_gas - driven_miles

theorem kennedy_drive_home 
  (efficiency : ℝ) 
  (initial_gas : ℝ) 
  (school_miles : ℝ) 
  (softball_miles : ℝ) 
  (restaurant_miles : ℝ) 
  (friend_miles : ℝ) 
  (h1 : efficiency = 19)
  (h2 : initial_gas = 2)
  (h3 : school_miles = 15)
  (h4 : softball_miles = 6)
  (h5 : restaurant_miles = 2)
  (h6 : friend_miles = 4) :
  remaining_miles efficiency initial_gas (school_miles + softball_miles + restaurant_miles + friend_miles) = 11 := by
  sorry

end NUMINAMATH_CALUDE_kennedy_drive_home_l2796_279693


namespace NUMINAMATH_CALUDE_units_digit_of_k_squared_plus_two_to_k_l2796_279652

def k : ℕ := 2015^2 + 2^2015

theorem units_digit_of_k_squared_plus_two_to_k (k : ℕ) : k = 2015^2 + 2^2015 → (k^2 + 2^k) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_k_squared_plus_two_to_k_l2796_279652


namespace NUMINAMATH_CALUDE_max_rectangles_in_6x6_square_l2796_279623

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a square with integer side length -/
structure Square where
  side : ℕ

/-- Represents the maximum number of rectangles that can fit in a square -/
def max_rectangles_in_square (r : Rectangle) (s : Square) : ℕ :=
  sorry

/-- The theorem stating the maximum number of 4×1 rectangles in a 6×6 square -/
theorem max_rectangles_in_6x6_square :
  let r : Rectangle := ⟨4, 1⟩
  let s : Square := ⟨6⟩
  max_rectangles_in_square r s = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_rectangles_in_6x6_square_l2796_279623


namespace NUMINAMATH_CALUDE_percent_of_x_l2796_279619

theorem percent_of_x (x : ℝ) (h : x > 0) : (x / 50 + x / 25) / x * 100 = 6 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_x_l2796_279619


namespace NUMINAMATH_CALUDE_coefficient_a2_value_l2796_279665

theorem coefficient_a2_value (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x : ℝ, x^2 + (x+1)^7 = a₀ + a₁*(x+2) + a₂*(x+2)^2 + a₃*(x+2)^3 + a₄*(x+2)^4 + a₅*(x+2)^5 + a₆*(x+2)^6 + a₇*(x+2)^7) →
  a₂ = -20 := by
sorry

end NUMINAMATH_CALUDE_coefficient_a2_value_l2796_279665


namespace NUMINAMATH_CALUDE_simplify_fraction_with_sqrt_three_l2796_279610

theorem simplify_fraction_with_sqrt_three : 
  (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1/2 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_with_sqrt_three_l2796_279610


namespace NUMINAMATH_CALUDE_similar_triangle_leg_length_l2796_279612

theorem similar_triangle_leg_length
  (a b c : ℝ)  -- sides of the first triangle
  (d e f : ℝ)  -- sides of the second triangle
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hd : d > 0) (he : e > 0) (hf : f > 0)
  (right_triangle1 : a^2 + b^2 = c^2)  -- first triangle is right triangle
  (right_triangle2 : d^2 + e^2 = f^2)  -- second triangle is right triangle
  (similar : a / d = b / e ∧ b / e = c / f)  -- triangles are similar
  (leg1 : a = 15)  -- one leg of first triangle
  (hyp1 : c = 17)  -- hypotenuse of first triangle
  (hyp2 : f = 51)  -- hypotenuse of second triangle
  : e = 24 :=  -- corresponding leg in second triangle
by sorry

end NUMINAMATH_CALUDE_similar_triangle_leg_length_l2796_279612


namespace NUMINAMATH_CALUDE_square_equals_product_plus_seven_l2796_279679

theorem square_equals_product_plus_seven (a b : ℕ) : 
  (a^2 = b * (b + 7)) ↔ ((a = 0 ∧ b = 0) ∨ (a = 12 ∧ b = 9)) :=
sorry

end NUMINAMATH_CALUDE_square_equals_product_plus_seven_l2796_279679


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l2796_279678

theorem quadratic_real_roots_condition 
  (a b c : ℝ) : 
  (∃ x : ℝ, a * x^2 + b * x + c = 0) ↔ (a ≠ 0 ∧ b^2 - 4*a*c ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l2796_279678


namespace NUMINAMATH_CALUDE_exactly_three_combinations_l2796_279642

/-- Represents a combination of banknotes -/
structure BanknoteCombination :=
  (n_2000 : Nat)
  (n_1000 : Nat)
  (n_500  : Nat)
  (n_200  : Nat)

/-- Checks if a combination is valid according to the problem conditions -/
def isValidCombination (c : BanknoteCombination) : Prop :=
  c.n_2000 + c.n_1000 + c.n_500 + c.n_200 = 10 ∧
  2000 * c.n_2000 + 1000 * c.n_1000 + 500 * c.n_500 + 200 * c.n_200 = 5000

/-- The set of all valid combinations -/
def validCombinations : Set BanknoteCombination :=
  { c | isValidCombination c }

/-- The three specific combinations mentioned in the solution -/
def solution1 : BanknoteCombination := ⟨0, 0, 10, 0⟩
def solution2 : BanknoteCombination := ⟨1, 0, 4, 5⟩
def solution3 : BanknoteCombination := ⟨0, 3, 2, 5⟩

/-- Theorem stating that there are exactly three valid combinations -/
theorem exactly_three_combinations :
  validCombinations = {solution1, solution2, solution3} := by sorry

#check exactly_three_combinations

end NUMINAMATH_CALUDE_exactly_three_combinations_l2796_279642
