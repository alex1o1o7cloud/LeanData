import Mathlib

namespace divisibility_by_3804_l2500_250086

theorem divisibility_by_3804 (n : ℕ+) :
  ∃ k : ℤ, (n.val ^ 3 - n.val : ℤ) * (5 ^ (8 * n.val + 4) + 3 ^ (4 * n.val + 2)) = 3804 * k := by
  sorry

end divisibility_by_3804_l2500_250086


namespace no_valid_grid_l2500_250099

/-- Represents a 3x3 grid with elements from 1 to 4 -/
def Grid := Fin 3 → Fin 3 → Fin 4

/-- Checks if all elements in a list are distinct -/
def allDistinct (l : List (Fin 4)) : Prop :=
  l.Nodup

/-- Checks if a row in the grid contains distinct elements -/
def rowDistinct (g : Grid) (i : Fin 3) : Prop :=
  allDistinct [g i 0, g i 1, g i 2]

/-- Checks if a column in the grid contains distinct elements -/
def colDistinct (g : Grid) (j : Fin 3) : Prop :=
  allDistinct [g 0 j, g 1 j, g 2 j]

/-- Checks if the main diagonal contains distinct elements -/
def mainDiagDistinct (g : Grid) : Prop :=
  allDistinct [g 0 0, g 1 1, g 2 2]

/-- Checks if the anti-diagonal contains distinct elements -/
def antiDiagDistinct (g : Grid) : Prop :=
  allDistinct [g 0 2, g 1 1, g 2 0]

/-- A grid is valid if all rows, columns, and diagonals contain distinct elements -/
def validGrid (g : Grid) : Prop :=
  (∀ i, rowDistinct g i) ∧
  (∀ j, colDistinct g j) ∧
  mainDiagDistinct g ∧
  antiDiagDistinct g

theorem no_valid_grid : ¬∃ g : Grid, validGrid g := by
  sorry

end no_valid_grid_l2500_250099


namespace garden_area_calculation_l2500_250067

/-- The total area of Mancino's and Marquita's gardens -/
def total_garden_area (mancino_garden_length mancino_garden_width mancino_garden_count
                       marquita_garden_length marquita_garden_width marquita_garden_count : ℕ) : ℕ :=
  (mancino_garden_length * mancino_garden_width * mancino_garden_count) +
  (marquita_garden_length * marquita_garden_width * marquita_garden_count)

/-- Theorem stating that the total area of Mancino's and Marquita's gardens is 304 square feet -/
theorem garden_area_calculation :
  total_garden_area 16 5 3 8 4 2 = 304 := by
  sorry

end garden_area_calculation_l2500_250067


namespace sin_double_angle_plus_pi_sixth_l2500_250080

theorem sin_double_angle_plus_pi_sixth (α : Real) 
  (h : Real.sin (α - π/6) = 1/3) : 
  Real.sin (2*α + π/6) = 7/9 := by
  sorry

end sin_double_angle_plus_pi_sixth_l2500_250080


namespace feet_heads_difference_l2500_250036

theorem feet_heads_difference : 
  let birds : ℕ := 4
  let dogs : ℕ := 3
  let cats : ℕ := 18
  let humans : ℕ := 7
  let bird_feet : ℕ := 2
  let dog_feet : ℕ := 4
  let cat_feet : ℕ := 4
  let human_feet : ℕ := 2
  let total_heads : ℕ := birds + dogs + cats + humans
  let total_feet : ℕ := birds * bird_feet + dogs * dog_feet + cats * cat_feet + humans * human_feet
  total_feet - total_heads = 74 :=
by sorry

end feet_heads_difference_l2500_250036


namespace repeating_decimal_equiv_fraction_l2500_250094

/-- The repeating decimal 0.3̄6 is equal to the fraction 11/30 -/
theorem repeating_decimal_equiv_fraction : 
  (∃ (x : ℚ), x = 0.3 + (6 / 9) / 10 ∧ x = 11 / 30) := by sorry

end repeating_decimal_equiv_fraction_l2500_250094


namespace counterfeit_coin_identification_l2500_250056

/-- Represents the outcome of a weighing on a balance scale -/
inductive WeighingResult
  | Equal : WeighingResult
  | Unequal : WeighingResult

/-- Represents a coin, which can be either real or counterfeit -/
inductive Coin
  | Real : Coin
  | Counterfeit : Coin

/-- Represents a weighing action on the balance scale -/
def weighing (c1 c2 : Coin) : WeighingResult :=
  match c1, c2 with
  | Coin.Real, Coin.Real => WeighingResult.Equal
  | Coin.Counterfeit, Coin.Real => WeighingResult.Unequal
  | Coin.Real, Coin.Counterfeit => WeighingResult.Unequal
  | Coin.Counterfeit, Coin.Counterfeit => WeighingResult.Equal

/-- Theorem stating that the counterfeit coin can be identified in at most 2 weighings -/
theorem counterfeit_coin_identification
  (coins : Fin 4 → Coin)
  (h_one_counterfeit : ∃! i, coins i = Coin.Counterfeit) :
  ∃ (w1 w2 : Fin 4 × Fin 4),
    let r1 := weighing (coins w1.1) (coins w1.2)
    let r2 := weighing (coins w2.1) (coins w2.2)
    ∃ i, coins i = Coin.Counterfeit ∧
         ∀ j, j ≠ i → coins j = Coin.Real :=
  sorry

end counterfeit_coin_identification_l2500_250056


namespace largest_mediocre_number_l2500_250089

def is_mediocre (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  n = (100 * a + 10 * b + c +
       100 * a + 10 * c + b +
       100 * b + 10 * a + c +
       100 * b + 10 * c + a +
       100 * c + 10 * a + b +
       100 * c + 10 * b + a) / 6

theorem largest_mediocre_number :
  is_mediocre 629 ∧ ∀ n : ℕ, is_mediocre n → n ≤ 629 :=
sorry

end largest_mediocre_number_l2500_250089


namespace jake_initial_balloons_count_l2500_250081

/-- The number of balloons Jake brought initially to the park -/
def jake_initial_balloons : ℕ := 3

/-- The number of balloons Allan brought to the park -/
def allan_balloons : ℕ := 6

/-- The number of additional balloons Jake bought at the park -/
def jake_additional_balloons : ℕ := 4

theorem jake_initial_balloons_count :
  jake_initial_balloons = 3 ∧
  allan_balloons = 6 ∧
  jake_additional_balloons = 4 ∧
  jake_initial_balloons + jake_additional_balloons = allan_balloons + 1 :=
by sorry

end jake_initial_balloons_count_l2500_250081


namespace chord_length_l2500_250016

-- Define the circle C
def circle_C (x y a : ℝ) : Prop :=
  x^2 + y^2 - 2*a*x + 4*a*y + 5*a^2 - 25 = 0

-- Define line l₁
def line_l1 (x y : ℝ) : Prop :=
  x + y + 2 = 0

-- Define line l₂
def line_l2 (x y : ℝ) : Prop :=
  3*x + 4*y - 5 = 0

-- Define the center of the circle
def center (a : ℝ) : ℝ × ℝ :=
  (a, -2*a)

-- State that the center of circle C lies on line l₁
axiom center_on_l1 (a : ℝ) :
  line_l1 (center a).1 (center a).2

-- Theorem: The length of the chord formed by intersecting circle C with line l₂ is 8
theorem chord_length : ℝ := by
  sorry

end chord_length_l2500_250016


namespace timothys_journey_speed_l2500_250041

/-- Proves that given the conditions of Timothy's journey, his average speed during the first part was 10 mph. -/
theorem timothys_journey_speed (v : ℝ) (T : ℝ) (h1 : T > 0) :
  v * (0.25 * T) + 50 * (0.75 * T) = 40 * T →
  v = 10 := by
  sorry

end timothys_journey_speed_l2500_250041


namespace mangoes_kelly_can_buy_l2500_250071

def mangoes_cost_per_half_pound : ℝ := 0.60
def kelly_budget : ℝ := 12

theorem mangoes_kelly_can_buy :
  let cost_per_pound : ℝ := 2 * mangoes_cost_per_half_pound
  let pounds_kelly_can_buy : ℝ := kelly_budget / cost_per_pound
  pounds_kelly_can_buy = 10 := by sorry

end mangoes_kelly_can_buy_l2500_250071


namespace diophantine_equation_solutions_l2500_250040

theorem diophantine_equation_solutions :
  ∀ x y : ℤ, 2 * x^3 + x * y - 7 = 0 ↔ 
    ((x = -7 ∧ y = -99) ∨ 
     (x = -1 ∧ y = -9) ∨ 
     (x = 1 ∧ y = 5) ∨ 
     (x = 7 ∧ y = -97)) :=
by sorry

end diophantine_equation_solutions_l2500_250040


namespace age_difference_proof_l2500_250088

theorem age_difference_proof : ∃ (a b : ℕ), 
  (a ≥ 10 ∧ a < 100) ∧ 
  (b ≥ 10 ∧ b < 100) ∧ 
  (a / 10 = b % 10) ∧ 
  (a % 10 = b / 10) ∧ 
  (a + 7 = 3 * (b + 7)) ∧ 
  (a - b = 36) := by
sorry

end age_difference_proof_l2500_250088


namespace solution_set_inequality_inequality_with_parameter_l2500_250064

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1|

-- Theorem for part I
theorem solution_set_inequality (x : ℝ) :
  (f x + f (x - 1) ≤ 2) ↔ (1/2 ≤ x ∧ x ≤ 5/2) :=
sorry

-- Theorem for part II
theorem inequality_with_parameter (a x : ℝ) (h : a > 0) :
  f (a * x) - a * f x ≤ f a :=
sorry

end solution_set_inequality_inequality_with_parameter_l2500_250064


namespace line_through_points_specific_line_equation_l2500_250002

/-- A line passing through two given points has a specific equation -/
theorem line_through_points (x₁ y₁ x₂ y₂ : ℝ) (h : x₁ ≠ x₂) :
  ∃ k b : ℝ, ∀ x y : ℝ, y = k * x + b ↔ (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂) :=
sorry

/-- The line passing through (2, 5) and (1, 1) has the equation y = 4x - 3 -/
theorem specific_line_equation :
  ∃ k b : ℝ, (k = 4 ∧ b = -3) ∧
    (∀ x y : ℝ, y = k * x + b ↔ (x = 2 ∧ y = 5) ∨ (x = 1 ∧ y = 1)) :=
sorry

end line_through_points_specific_line_equation_l2500_250002


namespace barbell_to_rack_ratio_is_one_to_ten_l2500_250031

/-- Given a squat rack cost and total cost, calculates the ratio of barbell cost to squat rack cost -/
def barbellToRackRatio (rackCost totalCost : ℚ) : ℚ × ℚ :=
  let barbellCost := totalCost - rackCost
  (barbellCost, rackCost)

/-- Theorem: The ratio of barbell cost to squat rack cost is 1:10 for given costs -/
theorem barbell_to_rack_ratio_is_one_to_ten :
  barbellToRackRatio 2500 2750 = (1, 10) := by
  sorry

#eval barbellToRackRatio 2500 2750

end barbell_to_rack_ratio_is_one_to_ten_l2500_250031


namespace sqrt_square_not_always_equal_l2500_250033

theorem sqrt_square_not_always_equal (a : ℝ) : ¬(∀ a, Real.sqrt (a^2) = a) := by
  sorry

end sqrt_square_not_always_equal_l2500_250033


namespace coat_price_theorem_l2500_250001

theorem coat_price_theorem (price : ℝ) : 
  (price - 150 = price * (1 - 0.3)) → price = 500 := by
  sorry

end coat_price_theorem_l2500_250001


namespace quadratic_points_relation_l2500_250093

theorem quadratic_points_relation (c : ℝ) (y₁ y₂ y₃ : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^2 + 2*x + c
  (f (-3) = y₁) → (f (1/2) = y₂) → (f 2 = y₃) → 
  (y₂ < y₁) ∧ (y₁ < y₃) := by
sorry

end quadratic_points_relation_l2500_250093


namespace remaining_budget_for_accessories_l2500_250060

def total_budget : ℕ := 250
def frame_cost : ℕ := 85
def front_wheel_cost : ℕ := 35
def rear_wheel_cost : ℕ := 40
def seat_cost : ℕ := 25
def handlebar_tape_cost : ℕ := 15
def water_bottle_cage_cost : ℕ := 10
def bike_lock_cost : ℕ := 20
def future_expenses : ℕ := 10

def total_expenses : ℕ :=
  frame_cost + front_wheel_cost + rear_wheel_cost + seat_cost +
  handlebar_tape_cost + water_bottle_cage_cost + bike_lock_cost + future_expenses

theorem remaining_budget_for_accessories :
  total_budget - total_expenses = 10 := by sorry

end remaining_budget_for_accessories_l2500_250060


namespace xy_sum_square_l2500_250018

theorem xy_sum_square (x y : ℤ) 
  (h1 : x * y + x + y = 106) 
  (h2 : x^2 * y + x * y^2 = 1320) : 
  x^2 + y^2 = 748 ∨ x^2 + y^2 = 5716 := by
sorry

end xy_sum_square_l2500_250018


namespace palindrome_product_sum_l2500_250046

/-- A function that checks if a number is a three-digit palindrome -/
def isThreeDigitPalindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n / 100 = n % 10)

/-- The theorem stating the sum of the two three-digit palindromes whose product is 522729 -/
theorem palindrome_product_sum : 
  ∃ (a b : ℕ), isThreeDigitPalindrome a ∧ 
                isThreeDigitPalindrome b ∧ 
                a * b = 522729 ∧ 
                a + b = 1366 := by
  sorry

end palindrome_product_sum_l2500_250046


namespace triangle_angle_measure_l2500_250007

theorem triangle_angle_measure (a b : ℝ) (area : ℝ) (h1 : a = 5) (h2 : b = 8) (h3 : area = 10) :
  ∃ (C : ℝ), (C = π / 6 ∨ C = 5 * π / 6) ∧ 
  (1 / 2 * a * b * Real.sin C = area) ∧ 
  (0 < C) ∧ (C < π) := by
sorry

end triangle_angle_measure_l2500_250007


namespace intersection_point_is_unique_l2500_250090

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (-3/8, 17/8)

/-- The first line equation -/
def line1 (x y : ℚ) : Prop := y = -3 * x + 1

/-- The second line equation -/
def line2 (x y : ℚ) : Prop := y = 5 * x + 4

theorem intersection_point_is_unique :
  let (x, y) := intersection_point
  (line1 x y ∧ line2 x y) ∧
  ∀ x' y', (line1 x' y' ∧ line2 x' y') → (x' = x ∧ y' = y) :=
by sorry

end intersection_point_is_unique_l2500_250090


namespace large_posters_count_l2500_250082

theorem large_posters_count (total : ℕ) (small_fraction : ℚ) (medium_fraction : ℚ) : 
  total = 50 →
  small_fraction = 2 / 5 →
  medium_fraction = 1 / 2 →
  (total : ℚ) * small_fraction + (total : ℚ) * medium_fraction + 5 = total :=
by
  sorry

end large_posters_count_l2500_250082


namespace julie_count_correct_l2500_250008

/-- Represents the number of people with a given name in the crowd -/
structure NameCount where
  barry : ℕ
  kevin : ℕ
  julie : ℕ
  joe : ℕ

/-- Represents the proportion of nice people for each name -/
structure NiceProportion where
  barry : ℚ
  kevin : ℚ
  julie : ℚ
  joe : ℚ

/-- The total number of nice people in the crowd -/
def totalNicePeople : ℕ := 99

/-- The actual count of people with each name -/
def actualCount : NameCount where
  barry := 24
  kevin := 20
  julie := 80  -- This is what we want to prove
  joe := 50

/-- The proportion of nice people for each name -/
def niceProportion : NiceProportion where
  barry := 1
  kevin := 1/2
  julie := 3/4
  joe := 1/10

/-- Calculates the number of nice people for a given name -/
def niceCount (count : ℕ) (proportion : ℚ) : ℚ :=
  (count : ℚ) * proportion

/-- Theorem stating that the number of people named Julie is correct -/
theorem julie_count_correct :
  actualCount.julie = 80 ∧
  (niceCount actualCount.barry niceProportion.barry +
   niceCount actualCount.kevin niceProportion.kevin +
   niceCount actualCount.julie niceProportion.julie +
   niceCount actualCount.joe niceProportion.joe : ℚ) = totalNicePeople :=
by sorry

end julie_count_correct_l2500_250008


namespace sum_of_interior_angles_l2500_250012

-- Define a triangle
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the interior angles of a triangle
def interior_angles (t : Triangle) : ℝ := sorry

-- Theorem: The sum of interior angles of a triangle is 180°
theorem sum_of_interior_angles (t : Triangle) : interior_angles t = 180 := by
  sorry

end sum_of_interior_angles_l2500_250012


namespace no_integer_solution_l2500_250096

theorem no_integer_solution : ∀ (x y z : ℤ), x ≠ 0 → 2*x^4 + 2*x^2*y^2 + y^4 ≠ z^2 := by
  sorry

end no_integer_solution_l2500_250096


namespace car_speed_problem_l2500_250006

/-- Proves that car R's average speed is 50 miles per hour given the problem conditions -/
theorem car_speed_problem (distance : ℝ) (time_difference : ℝ) (speed_difference : ℝ) :
  distance = 800 ∧ 
  time_difference = 2 ∧ 
  speed_difference = 10 →
  ∃ (speed_r : ℝ),
    speed_r > 0 ∧
    distance / speed_r - time_difference = distance / (speed_r + speed_difference) ∧
    speed_r = 50 := by
  sorry

end car_speed_problem_l2500_250006


namespace matchstick_houses_l2500_250058

theorem matchstick_houses (initial_matchsticks : ℕ) (num_houses : ℕ) 
  (h1 : initial_matchsticks = 600)
  (h2 : num_houses = 30) :
  (initial_matchsticks / 2) / num_houses = 10 := by
  sorry

end matchstick_houses_l2500_250058


namespace initial_observations_count_l2500_250021

theorem initial_observations_count 
  (initial_avg : ℝ) 
  (new_obs : ℝ) 
  (avg_decrease : ℝ) 
  (h1 : initial_avg = 12)
  (h2 : new_obs = 5)
  (h3 : avg_decrease = 1) :
  ∃ n : ℕ, 
    (n : ℝ) * initial_avg = ((n : ℝ) + 1) * (initial_avg - avg_decrease) - new_obs ∧ 
    n = 6 :=
by sorry

end initial_observations_count_l2500_250021


namespace first_day_earnings_10_l2500_250024

/-- A sequence of 5 numbers where each subsequent number is 4 more than the previous one -/
def IceCreamEarnings (first_day : ℕ) : Fin 5 → ℕ
  | ⟨0, _⟩ => first_day
  | ⟨n + 1, h⟩ => IceCreamEarnings first_day ⟨n, Nat.lt_trans n.lt_succ_self h⟩ + 4

/-- The theorem stating that if the sum of the sequence is 90, the first day's earnings were 10 -/
theorem first_day_earnings_10 :
  (∃ (first_day : ℕ), (Finset.sum Finset.univ (IceCreamEarnings first_day)) = 90) →
  (∃ (first_day : ℕ), (Finset.sum Finset.univ (IceCreamEarnings first_day)) = 90 ∧ first_day = 10) :=
by sorry


end first_day_earnings_10_l2500_250024


namespace system_solutions_l2500_250050

/-- The system of equations -/
def system (x y z : ℝ) : Prop :=
  x^3 + y^3 = 3*y + 3*z + 4 ∧
  y^3 + z^3 = 3*z + 3*x + 4 ∧
  z^3 + x^3 = 3*x + 3*y + 4

/-- The solutions to the system of equations -/
theorem system_solutions :
  (∀ x y z : ℝ, system x y z ↔ (x = -1 ∧ y = -1 ∧ z = -1) ∨ (x = 2 ∧ y = 2 ∧ z = 2)) :=
by sorry

end system_solutions_l2500_250050


namespace system_solution_condition_l2500_250098

theorem system_solution_condition (n p : ℕ) :
  (∃ x y : ℕ+, x + p * y = n ∧ x + y = p^2) ↔
  (p > 1 ∧ (p - 1) ∣ (n - 1) ∧ ∀ k : ℕ+, n ≠ p^(k : ℕ)) :=
sorry

end system_solution_condition_l2500_250098


namespace average_children_in_families_with_children_l2500_250085

theorem average_children_in_families_with_children 
  (total_families : ℕ) 
  (average_children : ℚ) 
  (childless_families : ℕ) 
  (h1 : total_families = 15)
  (h2 : average_children = 3)
  (h3 : childless_families = 3) :
  (total_families * average_children) / (total_families - childless_families) = 3.75 := by
sorry

end average_children_in_families_with_children_l2500_250085


namespace biking_problem_solution_l2500_250077

/-- Represents the problem of Andrea and Lauren biking in a park -/
def BikingProblem (park_length : ℝ) (distance_decrease_rate : ℝ) (andrea_initial_time : ℝ) (andrea_wait_time : ℝ) : Prop :=
  ∃ (lauren_speed : ℝ),
    lauren_speed > 0 ∧
    2 * lauren_speed + lauren_speed = distance_decrease_rate ∧
    let initial_distance := distance_decrease_rate * andrea_initial_time
    let remaining_distance := park_length - initial_distance
    let lauren_time := remaining_distance / lauren_speed
    andrea_initial_time + andrea_wait_time + lauren_time = 79

/-- The theorem stating the solution to the biking problem -/
theorem biking_problem_solution :
  BikingProblem 24 0.8 7 3 := by
  sorry

end biking_problem_solution_l2500_250077


namespace tan_theta_three_expression_l2500_250045

theorem tan_theta_three_expression (θ : Real) (h : Real.tan θ = 3) :
  (1 - Real.cos θ) / Real.sin θ - Real.sin θ / (1 + Real.cos θ ^ 2) = (11 * Real.sqrt 10 - 101) / 33 := by
  sorry

end tan_theta_three_expression_l2500_250045


namespace original_strip_length_is_57_l2500_250020

/-- Represents the folded strip configuration -/
structure FoldedStrip where
  width : ℝ
  folded_length : ℝ
  trapezium_count : ℕ

/-- Calculates the length of the original strip before folding -/
def original_strip_length (fs : FoldedStrip) : ℝ :=
  sorry

/-- Theorem stating the length of the original strip -/
theorem original_strip_length_is_57 (fs : FoldedStrip) 
  (h_width : fs.width = 3)
  (h_folded_length : fs.folded_length = 27)
  (h_trapezium_count : fs.trapezium_count = 4) :
  original_strip_length fs = 57 :=
sorry

end original_strip_length_is_57_l2500_250020


namespace f_composition_result_l2500_250025

noncomputable def f (z : ℂ) : ℂ :=
  if z.im = 0 then -z^2 else z^2

theorem f_composition_result :
  f (f (f (f (1 + 2*I)))) = 503521 + 420000*I :=
by sorry

end f_composition_result_l2500_250025


namespace stream_speed_l2500_250043

/-- Proves that the speed of a stream is 8 kmph given the conditions of the boat's travel -/
theorem stream_speed (boat_speed : ℝ) (downstream_distance : ℝ) (upstream_distance : ℝ) 
  (h1 : boat_speed = 24)
  (h2 : downstream_distance = 64)
  (h3 : upstream_distance = 32)
  (h4 : downstream_distance / (boat_speed + x) = upstream_distance / (boat_speed - x)) :
  x = 8 := by
  sorry

end stream_speed_l2500_250043


namespace symmetry_axes_symmetry_origin_l2500_250070

-- Define the curve C
def C (x y : ℝ) : Prop :=
  ((x + 1)^2 + y^2) * ((x - 1)^2 + y^2) = 2

-- Theorem for symmetry with respect to axes
theorem symmetry_axes :
  (∀ x y, C x y ↔ C (-x) y) ∧ (∀ x y, C x y ↔ C x (-y)) :=
sorry

-- Theorem for symmetry with respect to origin
theorem symmetry_origin :
  ∀ x y, C x y ↔ C (-x) (-y) :=
sorry

end symmetry_axes_symmetry_origin_l2500_250070


namespace brenda_weighs_220_l2500_250073

def mel_weight : ℕ := 70

def brenda_weight (m : ℕ) : ℕ := 3 * m + 10

theorem brenda_weighs_220 : brenda_weight mel_weight = 220 := by
  sorry

end brenda_weighs_220_l2500_250073


namespace division_problem_solution_l2500_250032

theorem division_problem_solution :
  ∀ (D d q r : ℕ),
    D + d + q + r = 205 →
    q = d →
    D = d * q + r →
    D = 174 ∧ d = 13 :=
by
  sorry

end division_problem_solution_l2500_250032


namespace mary_change_theorem_l2500_250078

-- Define the ticket prices and discounts
def adult_price : ℚ := 2
def child_price : ℚ := 1
def first_child_discount : ℚ := 0.5
def second_child_discount : ℚ := 0.75
def third_child_discount : ℚ := 1
def sales_tax_rate : ℚ := 0.08
def amount_paid : ℚ := 20

-- Calculate the total cost before tax
def total_cost_before_tax : ℚ :=
  adult_price +
  child_price * first_child_discount +
  child_price * second_child_discount +
  child_price * third_child_discount

-- Calculate the sales tax
def sales_tax : ℚ := total_cost_before_tax * sales_tax_rate

-- Calculate the total cost including tax
def total_cost_with_tax : ℚ := total_cost_before_tax + sales_tax

-- Calculate the change
def change : ℚ := amount_paid - total_cost_with_tax

-- Theorem to prove
theorem mary_change_theorem : change = 15.41 := by sorry

end mary_change_theorem_l2500_250078


namespace ann_shorts_purchase_l2500_250011

/-- Calculates the maximum number of shorts Ann can buy -/
def max_shorts (total_spent : ℕ) (shoe_cost : ℕ) (shorts_cost : ℕ) (num_tops : ℕ) : ℕ :=
  ((total_spent - shoe_cost) / shorts_cost)

theorem ann_shorts_purchase :
  let total_spent := 75
  let shoe_cost := 20
  let shorts_cost := 7
  let num_tops := 4
  max_shorts total_spent shoe_cost shorts_cost num_tops = 7 := by
  sorry

#eval max_shorts 75 20 7 4

end ann_shorts_purchase_l2500_250011


namespace jill_earnings_l2500_250005

/-- Calculates the total earnings of a waitress given her work conditions --/
def waitress_earnings (hourly_wage : ℝ) (tip_rate : ℝ) (shifts : ℕ) (hours_per_shift : ℕ) (average_orders_per_hour : ℝ) : ℝ :=
  let total_hours : ℝ := shifts * hours_per_shift
  let wage_earnings : ℝ := total_hours * hourly_wage
  let total_orders : ℝ := total_hours * average_orders_per_hour
  let tip_earnings : ℝ := tip_rate * total_orders
  wage_earnings + tip_earnings

/-- Theorem stating that Jill's earnings for the week are $240.00 --/
theorem jill_earnings :
  waitress_earnings 4 0.15 3 8 40 = 240 := by
  sorry

end jill_earnings_l2500_250005


namespace farm_animals_l2500_250013

theorem farm_animals (cows chickens : ℕ) : 
  cows + chickens = 12 →
  4 * cows + 2 * chickens = 20 + 2 * (cows + chickens) →
  cows = 10 := by sorry

end farm_animals_l2500_250013


namespace mixed_doubles_pairing_methods_l2500_250003

-- Define the number of male and female players
def num_male_players : ℕ := 5
def num_female_players : ℕ := 4

-- Define the number of players to be selected for each gender
def male_players_to_select : ℕ := 2
def female_players_to_select : ℕ := 2

-- Define the total number of pairing methods
def total_pairing_methods : ℕ := 120

-- Theorem statement
theorem mixed_doubles_pairing_methods :
  (Nat.choose num_male_players male_players_to_select) *
  (Nat.choose num_female_players female_players_to_select) * 2 =
  total_pairing_methods := by
  sorry


end mixed_doubles_pairing_methods_l2500_250003


namespace collinearity_condition_perpendicularity_condition_l2500_250057

-- Define the points as functions of a
def A (a : ℝ) : ℝ × ℝ := (1, -2*a)
def B (a : ℝ) : ℝ × ℝ := (2, a)
def C (a : ℝ) : ℝ × ℝ := (2+a, 0)
def D (a : ℝ) : ℝ × ℝ := (2*a, 1)

-- Define collinearity of three points
def collinear (p q r : ℝ × ℝ) : Prop :=
  (r.2 - p.2) * (q.1 - p.1) = (q.2 - p.2) * (r.1 - p.1)

-- Define perpendicularity of two lines
def perpendicular (p1 q1 p2 q2 : ℝ × ℝ) : Prop :=
  (q1.2 - p1.2) * (q2.2 - p2.2) = -(q1.1 - p1.1) * (q2.1 - p2.1)

-- Theorem 1: Collinearity condition
theorem collinearity_condition :
  ∀ a : ℝ, collinear (A a) (B a) (C a) ↔ a = -1/3 :=
sorry

-- Theorem 2: Perpendicularity condition
theorem perpendicularity_condition :
  ∀ a : ℝ, perpendicular (A a) (B a) (C a) (D a) ↔ a = 1/2 :=
sorry

end collinearity_condition_perpendicularity_condition_l2500_250057


namespace pencil_purchase_count_l2500_250062

/-- Represents the number of pencils and pens purchased -/
structure Purchase where
  pencils : ℕ
  pens : ℕ

/-- Represents the cost in won -/
@[reducible] def Won := ℕ

theorem pencil_purchase_count (p : Purchase) 
  (h1 : p.pencils + p.pens = 12)
  (h2 : 1000 * p.pencils + 1300 * p.pens = 15000) :
  p.pencils = 2 := by
  sorry

#check pencil_purchase_count

end pencil_purchase_count_l2500_250062


namespace line_inclination_angle_l2500_250055

/-- The inclination angle of a line with equation x + √3 * y + c = 0 is 5π/6 --/
theorem line_inclination_angle (c : ℝ) : 
  let line := {(x, y) : ℝ × ℝ | x + Real.sqrt 3 * y + c = 0}
  ∃ θ : ℝ, 0 ≤ θ ∧ θ < π ∧ 
    (∀ (x y : ℝ), (x, y) ∈ line → Real.tan θ = -(1 / Real.sqrt 3)) ∧
    θ = 5 * π / 6 := by
  sorry

end line_inclination_angle_l2500_250055


namespace green_balls_count_l2500_250039

theorem green_balls_count (blue_count : ℕ) (total_count : ℕ) 
  (h1 : blue_count = 8)
  (h2 : (blue_count : ℚ) / total_count = 1 / 3) :
  total_count - blue_count = 16 := by
  sorry

end green_balls_count_l2500_250039


namespace triangle_square_side_ratio_l2500_250042

theorem triangle_square_side_ratio :
  ∀ (t s : ℝ),
  (3 * t = 15) →  -- Perimeter of equilateral triangle
  (4 * s = 12) →  -- Perimeter of square
  (t / s = 5 / 3) :=  -- Ratio of side lengths
by
  sorry

end triangle_square_side_ratio_l2500_250042


namespace expression_defined_iff_l2500_250000

theorem expression_defined_iff (a : ℝ) :
  (∃ x : ℝ, x = (Real.sqrt (a + 1)) / (a - 2)) ↔ (a ≥ -1 ∧ a ≠ 2) := by
  sorry

end expression_defined_iff_l2500_250000


namespace power_sum_equality_l2500_250092

theorem power_sum_equality : (-1)^49 + 2^(4^3 + 3^2 - 7^2) = 16777215 := by
  sorry

end power_sum_equality_l2500_250092


namespace polynomial_division_remainder_l2500_250079

theorem polynomial_division_remainder : ∃ q : Polynomial ℤ, 
  X^5 + 3 = (X + 1)^2 * q + 2 := by
  sorry

end polynomial_division_remainder_l2500_250079


namespace expression_evaluation_l2500_250009

theorem expression_evaluation : 2 - (-3) - 4 - (-5) - 6 - (-7) - 8 = -1 := by
  sorry

end expression_evaluation_l2500_250009


namespace unique_number_property_l2500_250034

theorem unique_number_property : ∃! x : ℝ, x / 3 = x - 3 := by sorry

end unique_number_property_l2500_250034


namespace wendy_count_problem_l2500_250083

theorem wendy_count_problem (total_heads : ℕ) (total_legs : ℕ) 
  (h1 : total_heads = 28) 
  (h2 : total_legs = 92) : 
  ∃ (people animals : ℕ), 
    people + animals = total_heads ∧ 
    2 * people + 4 * animals = total_legs ∧ 
    people = 10 := by
  sorry

end wendy_count_problem_l2500_250083


namespace revenue_decrease_l2500_250053

theorem revenue_decrease (tax_reduction : Real) (consumption_increase : Real) :
  tax_reduction = 0.24 →
  consumption_increase = 0.12 →
  let new_tax_rate := 1 - tax_reduction
  let new_consumption := 1 + consumption_increase
  let revenue_change := 1 - (new_tax_rate * new_consumption)
  revenue_change = 0.1488 := by
  sorry

end revenue_decrease_l2500_250053


namespace family_average_age_unchanged_l2500_250049

theorem family_average_age_unchanged 
  (initial_members : ℕ) 
  (initial_avg_age : ℝ) 
  (years_passed : ℕ) 
  (baby_age : ℝ) 
  (h1 : initial_members = 5)
  (h2 : initial_avg_age = 17)
  (h3 : years_passed = 3)
  (h4 : baby_age = 2) : 
  initial_avg_age = 
    (initial_members * (initial_avg_age + years_passed) + baby_age) / (initial_members + 1) := by
  sorry

#check family_average_age_unchanged

end family_average_age_unchanged_l2500_250049


namespace bread_weight_equals_antons_weight_l2500_250061

/-- Prove that the weight of bread eaten by Vladimir equals Anton's weight before his birthday -/
theorem bread_weight_equals_antons_weight 
  (A : ℝ) -- Anton's weight
  (B : ℝ) -- Vladimir's weight before eating bread
  (F : ℝ) -- Fyodor's weight
  (X : ℝ) -- Weight of the bread
  (h1 : X + F = A + B) -- Condition 1: Bread and Fyodor weigh as much as Anton and Vladimir
  (h2 : B + X = A + F) -- Condition 2: Vladimir's weight after eating equals Anton and Fyodor
  : X = A := by
  sorry

end bread_weight_equals_antons_weight_l2500_250061


namespace alcohol_solution_proof_l2500_250048

/-- Proves that adding 1.8 liters of pure alcohol to a 6-liter solution
    that is 35% alcohol will result in a solution that is 50% alcohol. -/
theorem alcohol_solution_proof :
  let initial_volume : ℝ := 6
  let initial_concentration : ℝ := 0.35
  let target_concentration : ℝ := 0.50
  let added_alcohol : ℝ := 1.8

  let final_volume : ℝ := initial_volume + added_alcohol
  let initial_alcohol : ℝ := initial_volume * initial_concentration
  let final_alcohol : ℝ := initial_alcohol + added_alcohol

  (final_alcohol / final_volume) = target_concentration :=
by sorry


end alcohol_solution_proof_l2500_250048


namespace race_conditions_satisfied_l2500_250030

/-- The speed of Xiao Ying in meters per second -/
def xiao_ying_speed : ℝ := 4

/-- The speed of Xiao Liang in meters per second -/
def xiao_liang_speed : ℝ := 6

/-- Theorem stating that the given speeds satisfy the race conditions -/
theorem race_conditions_satisfied : 
  (5 * xiao_ying_speed + 10 = 5 * xiao_liang_speed) ∧ 
  (6 * xiao_ying_speed = 4 * xiao_liang_speed) := by
  sorry

end race_conditions_satisfied_l2500_250030


namespace binomial_and_permutation_l2500_250037

theorem binomial_and_permutation :
  (Nat.choose 8 5 = 56) ∧
  (Nat.factorial 5 / Nat.factorial 2 = 60) := by
  sorry

end binomial_and_permutation_l2500_250037


namespace pregnant_cows_l2500_250044

theorem pregnant_cows (total_cows : ℕ) (female_ratio : ℚ) (pregnant_ratio : ℚ) : 
  total_cows = 44 →
  female_ratio = 1/2 →
  pregnant_ratio = 1/2 →
  (↑total_cows * female_ratio * pregnant_ratio : ℚ) = 11 :=
by
  sorry

end pregnant_cows_l2500_250044


namespace jacket_price_change_l2500_250023

theorem jacket_price_change (P : ℝ) (x : ℝ) (h : x > 0) :
  P * (1 - (x / 100)^2) * 0.9 = 0.75 * P →
  x = 100 * Real.sqrt (1 / 6) := by
  sorry

end jacket_price_change_l2500_250023


namespace cars_between_black_and_white_l2500_250054

theorem cars_between_black_and_white :
  ∀ (n : ℕ) (black_pos_right : ℕ) (white_pos_left : ℕ),
    n = 20 →
    black_pos_right = 16 →
    white_pos_left = 11 →
    (n - black_pos_right) - (white_pos_left - 1) = 5 := by
  sorry

end cars_between_black_and_white_l2500_250054


namespace valid_parts_characterization_valid_parts_complete_l2500_250065

/-- A type representing the possible numbers of equal parts. -/
inductive ValidParts : Nat → Prop where
  | two : ValidParts 2
  | three : ValidParts 3
  | four : ValidParts 4
  | six : ValidParts 6
  | eight : ValidParts 8
  | twelve : ValidParts 12
  | twentyfour : ValidParts 24

/-- The total number of cells in the figure. -/
def totalCells : Nat := 24

/-- A function that checks if a number divides the total number of cells evenly. -/
def isDivisor (n : Nat) : Prop := totalCells % n = 0

/-- The main theorem stating that the valid numbers of parts are exactly those that divide the total number of cells evenly. -/
theorem valid_parts_characterization (n : Nat) : 
  ValidParts n ↔ (isDivisor n ∧ n > 1) :=
sorry

/-- The theorem stating that the list of valid parts is complete. -/
theorem valid_parts_complete : 
  ∀ n, isDivisor n ∧ n > 1 → ValidParts n :=
sorry

end valid_parts_characterization_valid_parts_complete_l2500_250065


namespace arcade_spending_fraction_l2500_250076

theorem arcade_spending_fraction (allowance : ℚ) (remaining : ℚ) 
  (h1 : allowance = 480 / 100)
  (h2 : remaining = 128 / 100)
  (h3 : remaining = (2/3) * (1 - (arcade_fraction : ℚ)) * allowance) :
  arcade_fraction = 3/5 := by
sorry

end arcade_spending_fraction_l2500_250076


namespace new_person_weight_l2500_250017

theorem new_person_weight (original_count : ℕ) (original_average : ℝ) (leaving_weight : ℝ) (average_increase : ℝ) :
  original_count = 20 →
  leaving_weight = 92 →
  average_increase = 4.5 →
  (original_count * (original_average + average_increase) - (original_count - 1) * original_average) = 182 :=
by sorry

end new_person_weight_l2500_250017


namespace unique_division_problem_l2500_250047

theorem unique_division_problem :
  ∀ (a b : ℕ),
  (a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 1 ∧ b ≤ 9) →
  (∃ (p : ℕ), 111111 * a = 1111 * b * 233 + p) →
  (∃ (q : ℕ), 11111 * a = 111 * b * 233 + (q - 1000)) →
  (a = 7 ∧ b = 3) := by
sorry

end unique_division_problem_l2500_250047


namespace betty_payment_l2500_250038

-- Define the given conditions
def doug_age : ℕ := 40
def sum_of_ages : ℕ := 90
def num_packs : ℕ := 20

-- Define Betty's age
def betty_age : ℕ := sum_of_ages - doug_age

-- Define the cost of a pack of nuts
def pack_cost : ℕ := 2 * betty_age

-- Theorem to prove
theorem betty_payment : betty_age * num_packs * 2 = 2000 := by
  sorry

end betty_payment_l2500_250038


namespace afternoon_to_morning_ratio_l2500_250027

def total_pears : ℕ := 420
def afternoon_pears : ℕ := 280

theorem afternoon_to_morning_ratio :
  let morning_pears := total_pears - afternoon_pears
  (afternoon_pears : ℚ) / morning_pears = 2 := by
  sorry

end afternoon_to_morning_ratio_l2500_250027


namespace contact_list_count_is_38_l2500_250019

/-- The number of people on Jerome's contact list at the end of the month -/
def contact_list_count : ℕ :=
  let classmates : ℕ := 20
  let out_of_school_friends : ℕ := classmates / 2
  let immediate_family : ℕ := 3
  let added_contacts : ℕ := 5 + 7
  let removed_contacts : ℕ := 3 + 4
  classmates + out_of_school_friends + immediate_family + added_contacts - removed_contacts

/-- Theorem stating that the number of people on Jerome's contact list at the end of the month is 38 -/
theorem contact_list_count_is_38 : contact_list_count = 38 := by
  sorry

end contact_list_count_is_38_l2500_250019


namespace mr_grey_purchases_l2500_250014

/-- The cost of Mr. Grey's purchases -/
theorem mr_grey_purchases (polo_price : ℝ) : polo_price = 26 :=
  let necklace_price := 83
  let game_price := 90
  let rebate := 12
  let total_cost := 322
  let num_polos := 3
  let num_necklaces := 2
  have h : num_polos * polo_price + num_necklaces * necklace_price + game_price - rebate = total_cost :=
    by sorry
  sorry

end mr_grey_purchases_l2500_250014


namespace pencil_sharpening_mean_l2500_250066

def pencil_sharpening_data : List ℕ := [13, 8, 13, 21, 7, 23, 15, 19, 12, 9, 28, 6, 17, 29, 31, 10, 4, 20, 16, 12, 2, 18, 27, 22, 5, 14, 31, 29, 8, 25]

theorem pencil_sharpening_mean :
  (pencil_sharpening_data.sum : ℚ) / pencil_sharpening_data.length = 543 / 30 := by
  sorry

end pencil_sharpening_mean_l2500_250066


namespace find_m_value_l2500_250035

/-- Given two functions f and g, prove the value of m when f(5) - g(5) = 20 -/
theorem find_m_value (f g : ℝ → ℝ) (m : ℝ) : 
  (∀ x, f x = 4*x^2 + 3*x + 5) →
  (∀ x, g x = x^2 - m*x - 9) →
  f 5 - g 5 = 20 →
  m = -16.8 := by
sorry

end find_m_value_l2500_250035


namespace geometric_progression_sum_change_l2500_250069

/-- Given a geometric progression with 3000 terms, all positive, prove that
    if increasing every third term by 50 times increases the sum by 10 times,
    then doubling every even term increases the sum by 11/8 times. -/
theorem geometric_progression_sum_change (b₁ : ℝ) (q : ℝ) (S : ℝ) : 
  b₁ > 0 ∧ q > 0 ∧ S > 0 →
  S = b₁ * (1 - q^3000) / (1 - q) →
  S + 49 * b₁ * q^2 * (1 - q^3000) / ((1 - q) * (1 + q + q^2)) = 10 * S →
  S + 2 * b₁ * q * (1 - q^3000) / (1 - q^2) = 11 * S / 8 := by
sorry

end geometric_progression_sum_change_l2500_250069


namespace line_slope_calculation_l2500_250026

/-- Given a line in the xy-plane with y-intercept 20 and passing through the point (150, 600),
    its slope is equal to 580/150. -/
theorem line_slope_calculation (line : Set (ℝ × ℝ)) : 
  (∀ p ∈ line, ∃ m b : ℝ, p.2 = m * p.1 + b) →  -- Line equation
  (0, 20) ∈ line →                              -- y-intercept condition
  (150, 600) ∈ line →                           -- Point condition
  ∃ m : ℝ, m = 580 / 150 ∧                      -- Slope existence and value
    ∀ (x y : ℝ), (x, y) ∈ line → y = m * x + 20 -- Line equation with calculated slope
  := by sorry

end line_slope_calculation_l2500_250026


namespace symmetry_about_x_equals_one_l2500_250068

-- Define a function f over the reals
variable (f : ℝ → ℝ)

-- State the theorem
theorem symmetry_about_x_equals_one (x : ℝ) : f (x - 1) = f (-(x - 2) + 1) := by
  sorry

end symmetry_about_x_equals_one_l2500_250068


namespace abc_sum_product_bound_l2500_250063

theorem abc_sum_product_bound (a b c : ℝ) (h : a + b + c = 1) :
  ∃ (M : ℝ), ∀ (ε : ℝ), ε > 0 → ∃ (a' b' c' : ℝ),
    a' + b' + c' = 1 ∧ 
    ab + ac + bc ≤ 1/2 ∧
    a' * b' + a' * c' + b' * c' < -M + ε :=
sorry

end abc_sum_product_bound_l2500_250063


namespace complex_fraction_calculation_l2500_250091

theorem complex_fraction_calculation : 
  (13/6 : ℚ) + ((((432/100 - 168/100 - 33/25) * 5/11 - 2/7) / (44/35)) : ℚ) = 521/210 := by
  sorry

end complex_fraction_calculation_l2500_250091


namespace john_spending_l2500_250074

def supermarket_spending (x : ℝ) (total : ℝ) : Prop :=
  let fruits_veg := x / 100 * total
  let meat := (1 / 3) * total
  let bakery := (1 / 6) * total
  let candy := 6
  fruits_veg + meat + bakery + candy = total ∧
  candy = 0.1 * total ∧
  x = 40 ∧
  fruits_veg = 24 ∧
  total = 60

theorem john_spending :
  ∃ (x : ℝ) (total : ℝ), supermarket_spending x total :=
sorry

end john_spending_l2500_250074


namespace system_solution_l2500_250004

theorem system_solution : ∃! (x y : ℝ), x - y = 2 ∧ 2*x + y = 7 := by
  sorry

end system_solution_l2500_250004


namespace f_of_3_equals_8_l2500_250059

-- Define the function f
def f (x : ℝ) : ℝ := 3 * (x - 1) + 2

-- State the theorem
theorem f_of_3_equals_8 : f 3 = 8 := by sorry

end f_of_3_equals_8_l2500_250059


namespace profit_calculation_l2500_250029

/-- Represents the profit made from commercial farming -/
def profit_from_farming 
  (total_land : ℝ)           -- Total land area in hectares
  (num_sons : ℕ)             -- Number of sons
  (profit_per_son : ℝ)       -- Annual profit per son
  (land_unit : ℝ)            -- Land unit for profit calculation in m^2
  : ℝ :=
  -- The function body is left empty as we only need the statement
  sorry

/-- Theorem stating the profit from farming under given conditions -/
theorem profit_calculation :
  let total_land : ℝ := 3                    -- 3 hectares
  let num_sons : ℕ := 8                      -- 8 sons
  let profit_per_son : ℝ := 10000            -- $10,000 per year per son
  let land_unit : ℝ := 750                   -- 750 m^2 unit
  let hectare_to_sqm : ℝ := 10000            -- 1 hectare = 10,000 m^2
  profit_from_farming total_land num_sons profit_per_son land_unit = 500 :=
by
  sorry  -- The proof is omitted as per instructions

end profit_calculation_l2500_250029


namespace xyz_value_l2500_250010

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 36)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 24) :
  x * y * z = 4 := by
  sorry

end xyz_value_l2500_250010


namespace pizza_fraction_l2500_250072

theorem pizza_fraction (initial_parts : ℕ) (cuts_per_part : ℕ) (pieces_eaten : ℕ) : 
  initial_parts = 12 →
  cuts_per_part = 2 →
  pieces_eaten = 3 →
  (pieces_eaten : ℚ) / (initial_parts * cuts_per_part : ℚ) = 1 / 8 := by
sorry

end pizza_fraction_l2500_250072


namespace fraction_problem_l2500_250097

theorem fraction_problem (N : ℝ) (F : ℝ) : 
  N = 8 → 0.5 * N = F * N + 2 → F = 1/4 := by
  sorry

end fraction_problem_l2500_250097


namespace movie_length_after_cut_l2500_250075

theorem movie_length_after_cut (final_length cut_length : ℕ) (h1 : final_length = 57) (h2 : cut_length = 3) :
  final_length + cut_length = 60 := by
  sorry

end movie_length_after_cut_l2500_250075


namespace apples_theorem_l2500_250022

def apples_problem (initial_apples : ℕ) (ricki_removes : ℕ) (days : ℕ) : Prop :=
  let samson_removes := 2 * ricki_removes
  let bindi_removes := 3 * samson_removes
  let total_daily_removal := ricki_removes + samson_removes + bindi_removes
  let total_weekly_removal := total_daily_removal * days
  total_weekly_removal = initial_apples + 2150

theorem apples_theorem : apples_problem 1000 50 7 := by
  sorry

end apples_theorem_l2500_250022


namespace line_through_point_l2500_250095

/-- 
Given a line with equation -1/3 - 3kx = 4y that passes through the point (1/3, -8),
prove that k = 95/3.
-/
theorem line_through_point (k : ℚ) : 
  (-1/3 : ℚ) - 3 * k * (1/3 : ℚ) = 4 * (-8 : ℚ) → k = 95/3 := by
  sorry

end line_through_point_l2500_250095


namespace one_seventh_difference_l2500_250084

theorem one_seventh_difference : ∃ (ε : ℚ), 1/7 - 0.14285714285 = ε ∧ ε > 0 ∧ ε < 1/(7*10^10) := by
  sorry

end one_seventh_difference_l2500_250084


namespace points_four_units_away_l2500_250028

theorem points_four_units_away (P : ℝ) : 
  P = -3 → {x : ℝ | |x - P| = 4} = {1, -7} := by sorry

end points_four_units_away_l2500_250028


namespace calculate_expression_l2500_250015

theorem calculate_expression : (121^2 - 110^2 + 11) / 10 = 255.2 := by
  sorry

end calculate_expression_l2500_250015


namespace logistics_center_equidistant_l2500_250087

def rectilinear_distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  |x₁ - x₂| + |y₁ - y₂|

def town_A : ℝ × ℝ := (2, 3)
def town_B : ℝ × ℝ := (-6, 9)
def town_C : ℝ × ℝ := (-3, -8)
def logistics_center : ℝ × ℝ := (-5, 0)

theorem logistics_center_equidistant :
  let (x, y) := logistics_center
  rectilinear_distance x y town_A.1 town_A.2 =
  rectilinear_distance x y town_B.1 town_B.2 ∧
  rectilinear_distance x y town_B.1 town_B.2 =
  rectilinear_distance x y town_C.1 town_C.2 :=
by sorry

end logistics_center_equidistant_l2500_250087


namespace a_5_value_l2500_250051

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem a_5_value (a : ℕ → ℝ) :
  geometric_sequence a →
  a 3 * a 7 = 2 →
  a 3 + a 7 = -4 →
  a 5 = Real.sqrt 2 := by
  sorry

end a_5_value_l2500_250051


namespace sheep_value_is_16_l2500_250052

/-- Represents the agreement between Kuba and the shepherd -/
structure Agreement where
  fullYearCoins : ℕ
  fullYearSheep : ℕ
  monthsWorked : ℕ
  coinsReceived : ℕ
  sheepReceived : ℕ

/-- Calculates the value of a sheep in gold coins based on the agreement -/
def sheepValue (a : Agreement) : ℕ :=
  let monthlyRate := a.fullYearCoins / 12
  let expectedCoins := monthlyRate * a.monthsWorked
  expectedCoins - a.coinsReceived

/-- The main theorem stating that the value of a sheep is 16 gold coins -/
theorem sheep_value_is_16 (a : Agreement) 
  (h1 : a.fullYearCoins = 20)
  (h2 : a.fullYearSheep = 1)
  (h3 : a.monthsWorked = 7)
  (h4 : a.coinsReceived = 5)
  (h5 : a.sheepReceived = 1) :
  sheepValue a = 16 := by
  sorry

#eval sheepValue { fullYearCoins := 20, fullYearSheep := 1, monthsWorked := 7, coinsReceived := 5, sheepReceived := 1 }

end sheep_value_is_16_l2500_250052
